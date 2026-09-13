"""Per-sample chance-constraint solvers.

The paper's per-sample problem (7)/(8): choose the proposal N(mu, Sigma) closest to N(mu0, Sigma0)
such that the SCBF chance constraint  Pr(A u >= b) >= 1 - delta  holds for u = ubar + du, du ~ N(mu, Sigma).

Two encodings of the constraint:
  'variance' (as printed, eq. (6)/(8)):   A(ubar+mu) - alpha * A Sigma A^T  >= b
  'std'      (the correct one):           A(ubar+mu) - z * sqrt(A Sigma A^T) >= b,   z = Phi^{-1}(1-delta)

For the unicycle the omega column of A = L_g h is identically zero, so the problem is one-dimensional
in the v-channel: mean shift m of the v-perturbation and std s <= s0 of the v-perturbation.
Objective as in (8): ||mu - mu0||_1 + ||P - P0||_F with Sigma = P P^T, P0 = diag(s0, s_omega).
Because A kills the omega column, the exact optimum has P[0,1] = 0, P[1,:] = P0[1,:], mu_omega = 0, so the
one-dimensional search below is exact up to grid resolution in s.  `validate_against_cvxpy` checks that.
"""
import numpy as np

BIG = 1e6
LAST_SOLVE_TIME = None   # solver-only time of the last cvxpy call (seconds), for the timing experiment

def solve_rows(ubar_v, c, b, s0, z=None, alpha=None, form="std", n_s=65):
    """Vectorised over K samples.
    ubar_v: (K,) nominal v;  c: (K,2) v-coefficients of the two rows;  b: (K,2) right-hand sides;
    s0: scalar std of the v-perturbation in Sigma0.
    Returns dict with m (K,), s (K,), active (K,) bool, infeasible (K,) bool, cost (K,)."""
    K = ubar_v.shape[0]
    s_grid = np.linspace(0.0, s0, n_s)                     # (n_s,)
    cc = c[:, :, None]                                     # (K,2,1)
    bb = b[:, :, None]
    ub = ubar_v[:, None, None]
    if form == "std":
        pen = z * np.abs(cc) * s_grid[None, None, :]       # (K,2,n_s)
    elif form == "variance":
        pen = alpha * cc ** 2 * s_grid[None, None, :] ** 2
    else:
        raise ValueError(form)
    r = bb + pen - cc * ub                                  # need c*m >= r
    pos = cc > 1e-12
    neg = cc < -1e-12
    zero = ~(pos | neg)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = r / cc
    lo = np.where(pos, ratio, -np.inf).max(axis=1)          # (K,n_s)
    hi = np.where(neg, ratio, np.inf).min(axis=1)
    zero_ok = np.where(zero, r <= 0, True).all(axis=1)      # rows with c = 0 must hold by themselves
    feas = (lo <= hi) & zero_ok
    m_star = np.clip(0.0, lo, hi)                           # closest to zero inside [lo, hi]
    m_star = np.where(np.isfinite(m_star), m_star, 0.0)
    m_inf = np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi), np.where(np.isfinite(lo), lo, np.where(np.isfinite(hi), hi, 0.0)))
    m_use = np.where(feas, m_star, m_inf)
    cost = np.abs(m_use) + (s0 - s_grid[None, :]) + np.where(feas, 0.0, BIG)
    j = cost.argmin(axis=1)                                 # (K,)
    idx = np.arange(K)
    m = m_use[idx, j]; s = s_grid[j]; cst = cost[idx, j]
    infeasible = ~feas[idx, j]
    # active: is (m=0, s=s0) infeasible?
    active = ~(feas[:, -1] & (lo[:, -1] <= 0.0) & (hi[:, -1] >= 0.0))
    return {"m": m, "s": s, "active": active, "infeasible": infeasible, "cost": cst}


def quantile(delta):
    from scipy.stats import norm
    return float(norm.ppf(1.0 - delta))


# ---------------------------------------------------------------------------------------------
# exact solvers (cvxpy) for validation and wall-clock timing
# ---------------------------------------------------------------------------------------------
def solve_exact_cvxpy(ubar, mu0, P0, A_rows, b_rows, form="std", z=None, alpha=None, solver="CLARABEL"):
    """Exact per-sample problem in the factor P (Sigma = P P^T), conic form.
    variance form: alpha ||P^T A_k^T||^2 <= A_k(ubar+mu) - b_k   (rotated second-order cone)
    std form:      z     ||P^T A_k^T||   <= A_k(ubar+mu) - b_k   (second-order cone)"""
    import cvxpy as cp
    n = mu0.shape[0]
    mu = cp.Variable(n); P = cp.Variable((n, n))
    cons = []
    for A, b in zip(A_rows, b_rows):
        A = np.asarray(A, float).reshape(1, n)
        aff = A @ (ubar + mu) - b
        if form == "std":
            cons.append(z * cp.norm(P.T @ A.T) <= aff)
        else:
            cons.append(alpha * cp.sum_squares(P.T @ A.T) <= aff)
    obj = cp.Minimize(cp.norm1(mu - mu0) + cp.norm(P - P0, "fro"))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver)
    return prob.status, (None if mu.value is None else mu.value), (None if P.value is None else P.value), prob.value


def solve_sdp_literal(ubar, mu0, P0, A_rows, b_rows, alpha, solver="CLARABEL"):
    """The paper's problem (8) exactly as printed: one PSD block per row,
        [[ I, sqrt(alpha) A P ], [ sqrt(alpha) P^T A^T, A mu - b ]] >> 0,
    objective ||mu - mu0||_1 + ||P - P0||_F.  Used only for wall-clock timing (M1)."""
    import cvxpy as cp
    n = mu0.shape[0]
    mu = cp.Variable(n); P = cp.Variable((n, n))
    cons = []
    sa = np.sqrt(alpha)
    for A, b in zip(A_rows, b_rows):
        A = np.asarray(A, float).reshape(1, n)
        top = cp.hstack([np.eye(n), sa * (P.T @ A.T)])                    # (n, n+1)
        bot = cp.hstack([sa * (A @ P), cp.reshape(A @ (ubar + mu) - b, (1, 1), order="C")])  # (1, n+1)
        M = cp.vstack([top, bot])
        cons.append(M >> 0)
    obj = cp.Minimize(cp.norm1(mu - mu0) + cp.norm(P - P0, "fro"))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver)
    global LAST_SOLVE_TIME
    LAST_SOLVE_TIME = getattr(prob.solver_stats, "solve_time", None)
    return prob.status, mu.value, P.value, prob.value


def validate_against_cvxpy(n_inst=200, seed=0, s0=0.5, s_om=1.5, form="std", z=2.748, alpha=None, verbose=False):
    """Random instances; compare the vectorised solver's objective with the exact conic optimum."""
    rng = np.random.default_rng(seed)
    gaps = []; act = 0
    for _ in range(n_inst):
        ubar_v = rng.uniform(-0.5, 1.5)
        c1 = rng.uniform(-2.5, 2.5); c = np.array([[c1, -c1]])
        # right-hand sides: make a good fraction active
        b = np.array([[rng.uniform(-2.0, 1.0), rng.uniform(-2.0, 1.0)]])
        res = solve_rows(np.array([ubar_v]), c, b, s0, z=z, alpha=alpha, form=form, n_s=201)
        if res["infeasible"][0]:
            continue
        P0 = np.diag([s0, s_om]); mu0 = np.zeros(2); ubar = np.array([ubar_v, 0.0])
        st, mu, P, val = solve_exact_cvxpy(ubar, mu0, P0, [[c1, 0.0], [-c1, 0.0]], [b[0, 0], b[0, 1]],
                                            form=form, z=z, alpha=alpha)
        if st not in ("optimal", "optimal_inaccurate"):
            continue
        mine = res["cost"][0]
        gaps.append((mine, val))
        act += int(res["active"][0])
    g = np.array(gaps)
    rel = (g[:, 0] - g[:, 1]) / np.maximum(g[:, 1], 1e-9)
    out = {"n": len(g), "n_active": act, "mean_abs_gap": float(np.abs(g[:, 0] - g[:, 1]).mean()),
           "max_abs_gap": float(np.abs(g[:, 0] - g[:, 1]).max()), "mean_rel_gap": float(rel.mean()),
           "max_rel_gap": float(rel.max())}
    if verbose:
        print(out)
    return out
