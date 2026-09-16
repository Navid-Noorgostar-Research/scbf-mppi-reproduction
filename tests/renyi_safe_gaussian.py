"""The sample-efficient safe Gaussian: the published table, reproduced, and what it does in flight.

    python tests/renyi_safe_gaussian.py                 # the table in FINDINGS.md section 2
    python tests/renyi_safe_gaussian.py --closed-loop   # what it would choose at Algorithm 1's states

WHY THIS EXISTS.  FINDINGS.md section 2 states that the proposal minimising the exact weight second
moment subject to the SAME corrected row INFLATES the covariance, by 1.63x to 4.28x, exactly where
program (8) collapses it to zero.  That paragraph said "solved numerically" and shipped no solver, and
it was the one substantive claim in that document with nothing runnable behind it.  This is the solver.

WHAT THE SECOND HALF ADDS, AND WHY IT MATTERS MORE THAN THE FIRST.  The table is evaluated at four
hand-chosen slack values.  The obvious question -- what does the cheapest safe Gaussian choose at the
states a controller actually visits? -- was never asked.  The counterfactual sweep answers it: at every
state Algorithm 1 reaches, on the same rows, the same nominal plan and the same delta, both programs are
solved and compared.  The answer is that the inflation very largely does not fire.  The table is correct
and the factor is real at tight slack; in flight the optimum sits just above the nominal covariance.
So the contrast is a statement about particular states, not a design rule, and it is reported that way.

THE OBJECTIVE.  For p = N(0, sigma0^2) and q = N(m, (beta sigma0)^2),

    E_q[(dp/dq)^2] = beta^2 / sqrt(2 beta^2 - 1) * exp( m^2 / (sigma0^2 (2 beta^2 - 1)) ),

finite only for beta > 1/sqrt(2).  This is the exponentiated Renyi-2 divergence between two Gaussians
(Liese and Vajda 1987); none of it is new, and section 7 of FINDINGS.md says so.  It is the same
quantity the ESS budget inverts, so the two programs are stated in one currency.  Raising beta also
raises the row penalty z|c| beta sigma0, so wide proposals eventually become infeasible; the grid
search resolves that trade-off per sample rather than assuming either side wins.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import norm


def solve_renyi(xp, ubar_v, c, b, s0, z, n_beta=257, beta_max=8.0, beta_eps=1e-3):
    """Minimise E_q[(dp/dq)^2] subject to the corrected chance row, over beta NOT capped at 1.

    Same signature and same two-row convention as gpu_closed_loop.solve_paper, so the two can be
    called side by side on identical inputs.  Works with numpy or cupy as `xp`.
    Returns m, s, beta, active -- `active` being True where the unmodified proposal (m = 0, s = s0)
    does not already satisfy both rows.
    """
    beta = xp.linspace(1.0 / np.sqrt(2.0) + beta_eps, beta_max, n_beta)
    s_grid = beta * s0
    d = 2.0 * beta ** 2 - 1.0
    c1 = c[..., 0][..., None]
    b1 = b[..., 0][..., None]
    b2 = b[..., 1][..., None]
    ub = ubar_v[..., None]
    a1 = xp.abs(c1)
    pen = z * a1 * s_grid
    r1 = b1 + pen - c1 * ub
    r2 = b2 + pen + c1 * ub                                   # row 2 has coefficient -c1
    big = a1 > 1e-12
    den = xp.where(big, c1, 1.0)
    lo = xp.where(big, xp.where(c1 > 0, r1 / den, -r2 / den), -xp.inf)
    hi = xp.where(big, xp.where(c1 > 0, -r2 / den, r1 / den), xp.inf)
    zero_ok = xp.where(big, True, (r1 <= 0) & (r2 <= 0))
    feas = (lo <= hi) & zero_ok
    m_star = xp.clip(xp.zeros_like(lo), lo, hi)               # |m| is what the exponent charges for
    m_star = xp.where(xp.isfinite(m_star), m_star, 0.0)
    m_inf = xp.where(xp.isfinite(lo) & xp.isfinite(hi), 0.5 * (lo + hi),
                     xp.where(xp.isfinite(lo), lo, xp.where(xp.isfinite(hi), hi, 0.0)))
    m_use = xp.where(feas, m_star, m_inf)
    logE = 2.0 * xp.log(beta) - 0.5 * xp.log(d) + m_use ** 2 / (s0 ** 2 * d)
    j = (logE + xp.where(feas, 0.0, 1e6)).argmin(axis=-1)
    m = xp.take_along_axis(m_use, j[..., None], -1)[..., 0]
    pen0 = z * a1[..., 0] * s0
    ok0 = (c1[..., 0] * ubar_v >= b[..., 0] + pen0) & (-c1[..., 0] * ubar_v >= b[..., 1] + pen0)
    return m, s_grid[j], beta[j], ~ok0


def validate(z):
    from gpu_closed_loop import solve_paper
    print("")
    print("FINDINGS.md section 2, reproduced   (sigma0 = 1, one active row, z = %.10f)" % z)
    print("")
    print("%8s%13s%11s%11s%13s" % ("slack", "beta* here", "FINDINGS", "analytic", "(8) gives s"))
    for slack, ref in ((0.50, 1.63), (0.00, 2.28), (-0.50, 3.17), (-1.00, 4.28)):
        c = np.array([[1.0, -1.0]])
        b = np.array([[0.0, -1e6]])
        ub = np.array([slack])
        _, _, bt, _ = solve_renyi(np, ub, c, b, 1.0, z, n_beta=40001, beta_max=12.0)
        # at slack 0 the stationarity condition is exactly 2 y^2 - (3 + z^2) y + 1 = 0 with y = beta^2
        an = np.sqrt(((3 + z * z) + np.sqrt((3 + z * z) ** 2 - 8)) / 4) if slack == 0.0 else np.nan
        _, sp, _ = solve_paper(np, ub, c, b, 1.0, z, z, "std")
        print("%8.2f%13.4f%11.2f%11.4f%13.4f" % (slack, float(bt[0]), ref, an, float(sp[0])))
    print("")
    print("  Program (8) returns s = 0 at every one of them.  The two programs disagree maximally,")
    print("  and in opposite directions, at exactly the states where the barrier is doing work.")


def closed_loop(cp, seeds, z):
    """At every state Algorithm 1 visits, solve BOTH programs on the same rows."""
    from gpu_closed_loop import CFG, DT, inside, rows, solve_paper, step
    c_ = CFG
    S, K, T = len(seeds), c_["K"], c_["T"]
    s0, s_om, lam, pen = c_["sigma_v"], c_["sigma_om"], c_["lam"], c_["penalty"]
    goal = cp.asarray(c_["goal"])
    R = cp.asarray(np.array([lam / s0 ** 2, lam / s_om ** 2]))
    prop = [np.random.default_rng(int(s)) for s in seeds]
    plant = [np.random.default_rng(10_000 + int(s)) for s in seeds]
    X = cp.asarray(np.tile(np.array([0.0, 0.5, 0.0]), (S, 1)))
    U = cp.zeros((S, T, 2))
    done = cp.zeros(S, bool)
    n_act = 0.0
    n_all = 0.0
    b_sum = 0.0
    s8_sum = 0.0
    g11 = 0.0
    g15 = 0.0
    g20 = 0.0
    for _ in range(c_["max_steps"]):
        xi = cp.asarray(np.stack([r.standard_normal((K, T, 2)) for r in prop]))
        Xs = cp.repeat(X[:, None, :], K, axis=1)
        eps = cp.empty((S, K, T, 2))
        run = cp.zeros((S, K))
        for t in range(T):
            cc, bb = rows(cp, Xs, c_["sigma_env"])
            ubk = cp.repeat(U[:, t, 0][:, None], K, axis=1)
            m, s, _ = solve_paper(cp, ubk, cc, bb, s0, z, z, "std")       # what the controller does
            _, _, bt, act = solve_renyi(cp, ubk, cc, bb, s0, z)           # what the cheapest one wants
            af = act.astype(cp.float64)
            n_act += float(af.sum())
            n_all += af.size
            b_sum += float((bt * af).sum())
            s8_sum += float((s / s0 * af).sum())
            g11 += float((af * (bt > 1.1)).sum())
            g15 += float((af * (bt > 1.5)).sum())
            g20 += float((af * (bt > 2.0)).sum())
            ev = m + s * xi[:, :, t, 0]
            eps[:, :, t, 0] = ev
            eps[:, :, t, 1] = s_om * xi[:, :, t, 1]
            Xs = step(cp, Xs, U[:, None, t, :] + eps[:, :, t, :], DT)
            run = run + ((Xs[..., :2] - goal) ** 2).sum(-1) + pen * (~inside(cp, Xs))
        Sc = (run + ((Xs[..., :2] - goal) ** 2).sum(-1)
              + ((U[:, None] * R * eps).sum(-1) + 0.5 * (U[:, None] ** 2 * R).sum(-1)).sum(-1))
        logw = -(Sc - Sc.min(axis=1, keepdims=True)) / lam
        logw -= logw.max(axis=1, keepdims=True)
        w = cp.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        U = U + (w[:, :, None, None] * eps).sum(1)
        noise = cp.asarray(np.stack([r.standard_normal(3) for r in plant]))
        X = cp.where(done[:, None], X,
                     step(cp, X, U[:, 0, :], DT) + c_["sigma_env"] * np.sqrt(DT) * noise)
        done = done | (cp.sqrt(((X[:, :2] - goal) ** 2).sum(-1)) < c_["goal_radius"])
        U = cp.concatenate([U[:, 1:], cp.zeros((S, 1, 2))], axis=1)
        if bool(done.all()):
            break
    print("")
    print("At the states Algorithm 1 visits   (%d seeds, %s sample-timesteps)"
          % (len(seeds), format(int(n_all), ",")))
    print("")
    print("  barrier active                                   %8.2f%%" % (100 * n_act / n_all))
    print("  mean beta the cheapest safe Gaussian wants        %8.4f" % (b_sum / n_act))
    print("  mean s/sigma0 program (8) actually returns        %8.4f" % (s8_sum / n_act))
    print("  of active samples, beta > 1.1                    %8.2f%%" % (100 * g11 / n_act))
    print("  of active samples, beta > 1.5                    %8.2f%%" % (100 * g15 / n_act))
    print("  of active samples, beta > 2.0                    %8.2f%%" % (100 * g20 / n_act))
    print("")
    print("  The 1.63-4.28 of the table is real at tight slack and essentially absent in flight.")
    print("  The gap between the two programs in flight is the collapse, not the inflation.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed-loop", action="store_true")
    ap.add_argument("--seeds", type=int, default=64)
    a = ap.parse_args()
    z = float(norm.ppf(1.0 - 0.003))
    validate(z)
    if a.closed_loop:
        from gpu_closed_loop import load_cupy
        closed_loop(load_cupy(), list(range(a.seeds)), z)


if __name__ == "__main__":
    main()
