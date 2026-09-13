"""Per-sample chance-constraint problem (paper eq. (7)/(8)) for an n-dimensional input and one or more rows.

    min_{mu, P}  ||mu − mu0||_1 + ||P − P0||_F
    s.t.         a (ubar + mu) − pen(std of a·du) >= b        for every row (a, b)
                 pen(s) = z s          ("std" form — the chance constraint the paper claims, z = Phi^{-1}(1−delta))
                 pen(s) = alpha s^2    ("variance" form — the constraint as printed)

with mu0 = 0, P0 = diag(s0), du ~ N(mu, P P^T).  Closed form for ONE active row (proof in the README):
  * the std of a·du under P0 is ||w||, w = s0 ∘ a;  reducing it to s costs (||w|| − s)/||a|| in Frobenius norm
    (rank-one update P = P0 − (1 − s/||w||) a^T w^T / ||a||^2);
  * the l1-cheapest mean shift for the residual margin r puts it on the coordinate with the largest |a_i|,
    cost |r| / max_i |a_i|.
  The optimum over s is found on a grid (exact for the "std" form: shrink first, then shift).
Several active rows (m <= 4): a near-optimal heuristic — candidates for the m target stds; for each candidate
the factor update Delta = A^T (A A^T)^{-1} V^T is the Frobenius-minimal one among updates that shrink each row
along its own w_j (not the global minimiser), and the l1-minimal mean shift satisfying every row is exact
(LP vertex enumeration, `l1_min_shift`).  Always feasible; more conservative than the exact SOCP optimum by a
median of about 10 % on real two-row instances (experiment V0 reports the gap split by the number of active
rows).  The vessel experiments report the fraction of sample-timesteps with >= 2 active rows (`multi_violation`).
"""
import numpy as np


def _solve_one_row(ubar, a_s, b_s, Pfac, mu, s0, z, alpha, form, n_s):
    """Solve one row (K,n)/(K,) against the CURRENT proposal (mu, Pfac): shrink the std along a_s by a rank-one
    update of Pfac and add an l1-cheapest mean shift.  Returns updated (mu, Pfac, s_opt, s_nom, active, cost)."""
    K, n = ubar.shape
    w = np.einsum("kij,ki->kj", np.transpose(Pfac, (0, 2, 1)), a_s)      # P^T a^T  (K,n)
    wn = np.linalg.norm(w, axis=-1)                                      # current std of a·du
    an = np.linalg.norm(a_s, axis=-1); amax = np.abs(a_s).max(-1)
    slack = (a_s * (ubar + mu)).sum(-1) - b_s
    pen_nom = z * wn if form == "std" else alpha * wn ** 2
    active = (slack - pen_nom) < 0.0
    zero_row = an < 1e-12
    idx = np.arange(K)
    if form == "std":
        # exact: the cost (wn − s)/|a| + max(0, z s − slack)/|a|_inf is piecewise linear in s with its minimum at
        # s* = clip(slack / z, 0, wn)  (shrink first, then shift)
        s_opt = np.clip(slack / z, 0.0, wn)
        r_opt = np.maximum(z * s_opt - slack, 0.0)
        cost_grid = None
    else:
        grid = np.linspace(0.0, 1.0, n_s)[None, :]
        s_grid = wn[:, None] * grid
        pen = alpha * s_grid ** 2
        r = pen - slack[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            cost_grid = (wn[:, None] - s_grid) / np.where(an > 1e-12, an, np.inf)[:, None] \
                      + np.maximum(r, 0.0) / np.where(amax > 1e-12, amax, np.inf)[:, None]
        cost_grid = np.where(np.isfinite(cost_grid), cost_grid, np.inf)
        k = cost_grid.argmin(-1)
        s_opt = s_grid[idx, k]; r_opt = np.maximum(r[idx, k], 0.0)
    s_opt = np.where(active & ~zero_row, s_opt, wn)
    r_opt = np.where(active & ~zero_row, r_opt, 0.0)
    i_star = np.abs(a_s).argmax(-1); coef = a_s[idx, i_star]
    mu = mu.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        mu[idx, i_star] += np.where(np.abs(coef) > 1e-12, r_opt / coef, 0.0)
    shrink = np.where(wn > 1e-12, 1.0 - s_opt / np.maximum(wn, 1e-12), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        outer = a_s[:, :, None] * w[:, None, :] / np.where(an > 1e-12, an, np.inf)[:, None, None] ** 2
    Pfac = Pfac - shrink[:, None, None] * np.where(np.isfinite(outer), outer, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cost_opt = (wn - s_opt) / np.where(an > 1e-12, an, np.inf) + r_opt / np.where(amax > 1e-12, amax, np.inf)
    cost_opt = np.where(active & ~zero_row, np.where(np.isfinite(cost_opt), cost_opt, 0.0), 0.0)
    return mu, Pfac, s_opt, wn, active, zero_row, cost_opt


def _joint_eval_batch(ubar, A, b, Aa, W, Ginv, s0, z, alpha, form, F, idx_act):
    """Evaluate the joint objective for every sample and every grid point at once.
    F: (K,Gp,m) target-std fractions.  Returns cost (K,Gp), mu (K,Gp,n), P (K,Gp,n,n)."""
    K, J, n = A.shape; Gp = F.shape[1]
    V = (F[:, :, :, None] - 1.0) * W[:, None, :, :]                              # (K,Gp,m,n)
    Delta = np.einsum("kmi,kml,kglj->kgij", Aa, Ginv, V, optimize=True)           # (K,Gp,n,n)
    P = np.diag(s0)[None, None] + Delta
    Sig = P @ np.swapaxes(P, -1, -2)
    SA = np.einsum("kgil,kjl->kgij", Sig, A, optimize=True)                                    # (K,Gp,n,J)
    std_all = np.sqrt(np.maximum(np.einsum("kji,kgij->kgj", A, SA, optimize=True), 0.0))     # (K,Gp,J)
    pen_all = z * std_all if form == "std" else alpha * std_all ** 2
    rreq = b[:, None, :] + pen_all - (A * ubar[:, None, :]).sum(-1)[:, None, :]    # (K,Gp,J)
    Ab = np.broadcast_to(A[:, None], (K, Gp, J, n))
    mu, ok = l1_min_shift(Ab.reshape(K * Gp, J, n), rreq.reshape(K * Gp, J))      # exact over ALL rows
    mu = mu.reshape(K, Gp, n); ok = ok.reshape(K, Gp)
    cost = np.abs(mu).sum(-1) + np.linalg.norm(Delta.reshape(K, Gp, -1), axis=-1)
    cost = np.where(ok, cost, np.inf)
    return cost, mu, P


def _solve_joint(ubar, A, b, act, s0, z, alpha, form, n_g=None, refine=None):
    """Joint solve for samples with m active rows (all samples in this call have the same m).
    ubar (K,n), A (K,J,n), b (K,J), act (K,J) bool with exactly m True per sample.
    Candidate target stds per active row j: {0, s_j* = clip(slack_j / z, 0, s_j^nom) [the single-row optimum of
    the std form], s_j^nom} plus, for form 'variance', the single-row grid optimum; all m-fold combinations are
    evaluated, then a local refinement (fractions +/- 1/6 around the best, 3 points per row).  For each candidate
    the Frobenius-minimal factor update Delta = A_act^T (A_act A_act^T)^{-1} V^T (V: shrink vectors along w_j)
    is exact and the l1-minimal mean shift for ALL rows is exact (vertex enumeration on the active rows,
    feasibility on all rows).  Vectorised over samples and candidates.  Returns mu, Pfac, s_first, s_nom_first,
    ok, cost."""
    import itertools
    K, J, n = A.shape
    m = int(act.sum(-1)[0])
    idx_act = np.argsort(~act, axis=-1, kind="stable")[:, :m]
    Aa = np.take_along_axis(A, idx_act[:, :, None], axis=1)
    W = Aa * s0[None, None, :]
    Wn = np.linalg.norm(W, axis=-1)                                              # (K,m)
    G = np.einsum("kmi,kli->kml", Aa, Aa)
    Ginv = np.linalg.pinv(G)
    slack = (Aa * ubar[:, None, :]).sum(-1) - np.take_along_axis(b, idx_act, axis=1)   # (K,m)
    if form == "std":
        f_star = np.clip(slack / (z * np.maximum(Wn, 1e-12)), 0.0, 1.0)
    else:
        f_star = np.clip(np.sqrt(np.maximum(slack, 0.0) / (alpha * np.maximum(Wn, 1e-12) ** 2)), 0.0, 1.0)
    cands = np.stack([np.zeros_like(f_star), f_star, np.ones_like(f_star), 0.5 * f_star], axis=-1)   # (K,m,4)
    combos = np.array(list(itertools.product(range(cands.shape[-1]), repeat=m)))                # (Gp,m)
    F = np.stack([cands[:, j, combos[:, j]] for j in range(m)], axis=-1)                       # (K,Gp,m)
    cost, mu, P = _joint_eval_batch(ubar, A, b, Aa, W, Ginv, s0, z, alpha, form, F, idx_act)
    g_best = cost.argmin(-1); idx = np.arange(K)
    best_cost = cost[idx, g_best]; best_mu = mu[idx, g_best]; best_P = P[idx, g_best]; f_best = F[idx, g_best]
    if refine is None:
        refine = {1: 9, 2: 5, 3: 3, 4: 3}.get(m, 3)
    if refine > 1:
        h = 1.0 / 6.0
        offs = np.array(list(itertools.product(np.linspace(-h, h, refine), repeat=m)))
        F2 = np.clip(f_best[:, None, :] + offs[None, :, :], 0.0, 1.0)
        cost2, mu2, P2 = _joint_eval_batch(ubar, A, b, Aa, W, Ginv, s0, z, alpha, form, F2, idx_act)
        g2 = cost2.argmin(-1)
        better = cost2[idx, g2] < best_cost
        best_cost = np.where(better, cost2[idx, g2], best_cost)
        best_mu = np.where(better[:, None], mu2[idx, g2], best_mu)
        best_P = np.where(better[:, None, None], P2[idx, g2], best_P)
        f_best = np.where(better[:, None], F2[idx, g2], f_best)
    s_first = f_best[:, 0] * Wn[:, 0]
    return best_mu, best_P, s_first, Wn[:, 0], np.isfinite(best_cost), best_cost


def solve_rows_nd(ubar, A, b, s0, z=None, alpha=None, form="std", n_s=65, n_g=None):
    """ubar (K,n) nominal input; A (K,J,n); b (K,J); s0 (n,) stds of P0 = diag(s0).
    Samples with one active row: exact closed form.  Samples with m >= 2 active rows (m <= 4): joint candidate
    solve (`_solve_joint`) — feasible and near-optimal, see V0 for the gap to the conic optimum.  Returns dict: mu (K,n), Pfac (K,n,n), active (K,),
    infeasible (K,), multi_violation (K,) [>= 2 rows active], residual_violation (K,), s (K,) std kept along the
    (first) solved row, s_nom (K,), cost (K,), row (K,)."""
    K, J, n = A.shape
    s0 = np.asarray(s0, float); idx = np.arange(K)
    Pfac = np.broadcast_to(np.diag(s0), (K, n, n)).copy(); mu = np.zeros((K, n))
    w0 = A * s0[None, None, :]; wn0 = np.linalg.norm(w0, axis=-1)
    slack0 = (A * ubar[:, None, :]).sum(-1) - b
    pen0 = z * wn0 if form == "std" else alpha * wn0 ** 2
    margin0 = slack0 - pen0
    act_rows = margin0 < 0.0                                                   # (K,J)
    n_act = act_rows.sum(-1)
    jstar = margin0.argmin(-1)
    a_s = A[idx, jstar]; b_s = b[idx, jstar]
    mu, Pfac, s_opt, s_nom, active, zero_row, cost = _solve_one_row(ubar, a_s, b_s, Pfac, mu, s0, z, alpha, form, n_s)
    infeasible = zero_row & (b_s > 0.0)
    multi = n_act >= 2
    residual = np.zeros(K, bool)
    if J > 1:
        # rows violated by the single-row solution (a mean shift can cross another row) join the active set
        Sig = Pfac @ np.transpose(Pfac, (0, 2, 1))
        std_all = np.sqrt(np.maximum(np.einsum("kji,kil,kjl->kj", A, Sig, A), 0.0))
        pen_all = z * std_all if form == "std" else alpha * std_all ** 2
        marg_all = (A * (ubar + mu)[:, None, :]).sum(-1) - b - pen_all
        viol = marg_all < -1e-9 * (1.0 + np.abs(b))
        joint_rows = act_rows | viol
        need_joint = (multi | viol.any(-1)) & ~zero_row
        n_j = joint_rows.sum(-1)
        for m in range(1, J + 1):
            sel = np.where(need_joint & (n_j == m))[0]
            if len(sel) == 0:
                continue
            mu_j, P_j, s_j, sn_j, ok_j, cost_j = _solve_joint(ubar[sel], A[sel], b[sel], joint_rows[sel], s0, z, alpha, form)
            good = sel[ok_j]
            mu[good] = mu_j[ok_j]; Pfac[good] = P_j[ok_j]; cost[good] = cost_j[ok_j]
            # std kept along the most violated row (jstar), from the final factor — consistent with the single-row case
            aj = A[good, jstar[good]]
            s_opt[good] = np.linalg.norm(np.einsum("kij,ki->kj", np.transpose(P_j[ok_j], (0, 2, 1)), aj), axis=-1)
            s_nom[good] = np.linalg.norm(aj * s0[None, :], axis=-1)
            residual[sel[~ok_j]] = True
        infeasible = infeasible | residual
    return {"mu": mu, "Pfac": Pfac, "active": active, "infeasible": infeasible, "multi_violation": multi,
            "residual_violation": residual, "s": s_opt, "s_nom": s_nom, "cost": cost, "row": jstar}


def l1_min_shift(A, r, tol=1e-9, enum_rows=None):
    """Exact  min ||mu||_1  s.t.  A mu >= r  for A (K,J,n) with n = 3 and small J, by enumerating the LP vertices:
    at an optimum at most min(n, J) coordinates are nonzero and as many rows are tight.  Candidates: one coordinate
    (one tight row, interval intersection), two coordinates with two tight rows, three with three.  Returns
    (mu (K,n), ok (K,)) — ok False when no candidate is feasible (then the caller falls back)."""
    K, J, n = A.shape
    assert n == 3
    if enum_rows is not None:
        # enumerate vertices from a subset of rows (K,m) (the active ones); feasibility is checked on ALL rows
        Ae = np.take_along_axis(A, enum_rows[:, :, None], axis=1); re = np.take_along_axis(r, enum_rows, axis=1)
    else:
        Ae, re = A, r
    Je = Ae.shape[1]
    best = np.full(K, np.inf); mu_best = np.zeros((K, n))
    def consider(mu_c):
        feas = ((np.einsum("kji,ki->kj", A, mu_c) - r) >= -tol * (1.0 + np.abs(r))).all(-1)
        cst = np.abs(mu_c).sum(-1)
        better = feas & (cst < best)
        mu_best[better] = mu_c[better]; best[better] = cst[better]
    # (0) mu = 0
    consider(np.zeros((K, n)))
    # (1) single coordinate
    for i in range(n):
        ai = Ae[:, :, i]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = re / ai
        lo = np.where(ai > tol, ratio, -np.inf).max(-1); hi = np.where(ai < -tol, ratio, np.inf).min(-1)
        val = np.clip(0.0, lo, hi)
        val = np.where(lo <= hi, val, np.nan)
        mu_c = np.zeros((K, n)); mu_c[:, i] = np.where(np.isfinite(val), val, 0.0)
        mu_c[~np.isfinite(val)] = np.nan
        consider(np.nan_to_num(mu_c, nan=1e300))
    # (2) two coordinates, two tight rows
    import itertools
    for (i, k) in itertools.combinations(range(n), 2):
        for (j, l) in itertools.combinations(range(Je), 2):
            Mx = np.stack([np.stack([Ae[:, j, i], Ae[:, j, k]], -1), np.stack([Ae[:, l, i], Ae[:, l, k]], -1)], -2)  # (K,2,2)
            det = Mx[:, 0, 0] * Mx[:, 1, 1] - Mx[:, 0, 1] * Mx[:, 1, 0]
            good = np.abs(det) > 1e-14
            rhs = np.stack([re[:, j], re[:, l]], -1)
            sol = np.zeros((K, 2))
            sol[good, 0] = (rhs[good, 0] * Mx[good, 1, 1] - Mx[good, 0, 1] * rhs[good, 1]) / det[good]
            sol[good, 1] = (Mx[good, 0, 0] * rhs[good, 1] - Mx[good, 1, 0] * rhs[good, 0]) / det[good]
            mu_c = np.zeros((K, n)); mu_c[:, i] = sol[:, 0]; mu_c[:, k] = sol[:, 1]
            mu_c[~good] = 1e300
            consider(mu_c)
    # (3) three coordinates, three tight rows
    if Je >= 3:
        for rows in itertools.combinations(range(Je), 3):
            Mx = Ae[:, list(rows), :]                                  # (K,3,3)
            det = np.linalg.det(Mx); good = np.abs(det) > 1e-18
            mu_c = np.full((K, n), 1e300)
            if good.any():
                mu_c[good] = np.linalg.solve(Mx[good], re[good][:, list(rows)][..., None])[..., 0]
            consider(mu_c)
    ok = np.isfinite(best)
    return mu_best, ok


def quantile(delta):
    from scipy.stats import norm
    return float(norm.ppf(1.0 - delta))


def solve_exact_cvxpy(ubar, P0, A_rows, b_rows, form="std", z=None, alpha=None, solver="CLARABEL"):
    import cvxpy as cp
    n = ubar.shape[0]
    mu = cp.Variable(n); P = cp.Variable((n, n))
    cons = []
    for A, bb in zip(A_rows, b_rows):
        A = np.asarray(A, float).reshape(1, n)
        aff = A @ (ubar + mu) - bb
        if form == "std":
            cons.append(z * cp.norm(P.T @ A.T) <= aff)
        else:
            cons.append(alpha * cp.sum_squares(P.T @ A.T) <= aff)
    prob = cp.Problem(cp.Minimize(cp.norm1(mu) + cp.norm(P - P0, "fro")), cons)
    prob.solve(solver=solver)
    return prob.status, mu.value, P.value, prob.value



LAST_SOLVE_TIME = None


def solve_sdp_literal(ubar, P0, A_rows, b_rows, alpha, solver="CLARABEL"):
    """Problem (8) exactly as printed (one PSD block per row) — for the wall-clock experiment only.
    Sets LAST_SOLVE_TIME to the solver-only time reported by cvxpy."""
    import cvxpy as cp
    global LAST_SOLVE_TIME
    n = ubar.shape[0]
    mu = cp.Variable(n); P = cp.Variable((n, n)); cons = []; sa = np.sqrt(alpha)
    for A, bb in zip(A_rows, b_rows):
        A = np.asarray(A, float).reshape(1, n)
        top = cp.hstack([np.eye(n), sa * (P.T @ A.T)])
        bot = cp.hstack([sa * (A @ P), cp.reshape(A @ (ubar + mu) - bb, (1, 1), order="C")])
        cons.append(cp.vstack([top, bot]) >> 0)
    prob = cp.Problem(cp.Minimize(cp.norm1(mu) + cp.norm(P - P0, "fro")), cons)
    prob.solve(solver=solver)
    LAST_SOLVE_TIME = getattr(prob.solver_stats, "solve_time", None)
    return prob.status, mu.value, P.value, prob.value


def validate_against_cvxpy(n_inst=150, seed=0, s0=(200.0, 200.0, 60.0), form="std", z=2.748, alpha=None,
                           rows=1, verbose=False):
    """Random instances with the input scale of the vessel; compares the closed-form objective with the exact
    conic optimum. rows = 1 is the exact case; rows = 2 measures the approximation of taking one row."""
    rng = np.random.default_rng(seed)
    s0 = np.asarray(s0, float)
    gaps = []; act = 0; multi = 0
    for _ in range(n_inst):
        A = rng.normal(size=(1, rows, 3)) * np.array([1e-3, 1e-3, 2e-4])      # scale of n^T R2 M^{-1}
        ubar = rng.normal(size=(1, 3)) * np.array([200, 200, 50.0])
        slack = (A[0] * ubar[0]).sum(-1)
        b = slack + rng.uniform(-0.5, 1.0, size=rows) * (z * np.linalg.norm(A[0] * s0, axis=-1))
        res = solve_rows_nd(ubar, A, b[None, :], s0, z=z, alpha=alpha, form=form, n_s=401)
        st, mu, P, val = solve_exact_cvxpy(ubar[0], np.diag(s0), [A[0, j] for j in range(rows)], list(b),
                                          form=form, z=z, alpha=alpha)
        if st not in ("optimal", "optimal_inaccurate"):
            continue
        # our objective, evaluated exactly (not the surrogate): l1 of mu + Frobenius of P − P0
        mine = np.abs(res["mu"][0]).sum() + np.linalg.norm(res["Pfac"][0] - np.diag(s0))
        # residual violation of any row by the returned proposal (should be none after the sweeps)
        Sig = res["Pfac"][0] @ res["Pfac"][0].T
        for j in range(rows):
            sd = np.sqrt(max(A[0, j] @ Sig @ A[0, j], 0.0)); penj = z * sd if form == "std" else alpha * sd ** 2
            if A[0, j] @ (ubar[0] + res["mu"][0]) - penj < b[j] - 1e-7 * (1 + abs(b[j])):
                multi += 1; break
        gaps.append((mine, val)); act += int(res["active"][0])
    g = np.array(gaps)
    rel = (g[:, 0] - g[:, 1]) / np.maximum(g[:, 1], 1e-9)
    out = {"n": len(g), "n_active": act, "n_multi_violation": multi,
           "mean_abs_gap": float(np.abs(g[:, 0] - g[:, 1]).mean()), "max_abs_gap": float(np.abs(g[:, 0] - g[:, 1]).max()),
           "mean_rel_gap": float(rel.mean()), "max_rel_gap": float(rel.max())}
    if verbose:
        print(out)
    return out
