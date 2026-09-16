"""How much of a barrier-row violation is an actual collision?

    python tests/certificate_conservatism.py --seeds 64

WHY THIS EXISTS.  Every safety number in this repository is reported against one of two different
events, and they are not the same event.  `sat_active` and `delivered Pr(a u >= b)` count violations of
the BARRIER ROW -- the per-step sufficient condition the SCBF imposes.  `collision_rate` and `min_h`
count the trajectory actually leaving the safe set.  The row is a certificate: satisfying it is enough
for safety, and violating it is not evidence of anything.  Nothing in this repository has ever measured
the gap, so "the sampler violated the row 8% of the time" has had no safety interpretation attached.

WHAT IS MEASURED.  Plain MPPI is run closed loop, so the visited states are the ones a real controller
produces rather than a sweep.  At every control cycle each of the K rollouts is scored twice: M, the
number of horizon timesteps at which the drawn control violates at least one row (the same test the
controller applies, scbf_mppi/mppi.py), and whether that rollout's own trajectory ever leaves the safe
set.  The conditional exit probabilities are then the certificate's precision.

WHY PLAIN MPPI.  Under Algorithm 1 the proposal is reshaped so the row holds by construction -- with
the covariance collapsed to zero it holds deterministically -- so there are almost no violations to
score.  The unconstrained proposal is the only one that visits both sides of the row often enough for
the conditional to be estimated, and the question is about the certificate, not about a controller.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from gpu_closed_loop import CFG, DT, inside, load_cupy, rows, solve_paper, step


def measure(cp, seeds, kind="mppi", cfg=None):
    c_ = dict(CFG, **(cfg or {}))
    S, K, T = len(seeds), c_["K"], c_["T"]
    s0, s_om = c_["sigma_v"], c_["sigma_om"]
    lam, pen = c_["lam"], c_["penalty"]
    goal = cp.asarray(c_["goal"])
    R = cp.asarray(np.array([lam / s0 ** 2, lam / s_om ** 2]))
    from scipy.stats import norm
    z = float(norm.ppf(1.0 - c_["delta"]))
    prop = [np.random.default_rng(int(s)) for s in seeds]
    plant = [np.random.default_rng(10_000 + int(s)) for s in seeds]

    X = cp.asarray(np.tile(np.array([0.0, 0.5, 0.0]), (S, 1)))
    U = cp.zeros((S, T, 2))
    done = cp.zeros(S, bool)
    nM = cp.zeros(T + 1)          # rollouts by violation count
    nE = cp.zeros(T + 1)          # of those, how many left the safe set

    for _ in range(c_["max_steps"]):
        xi = cp.asarray(np.stack([r.standard_normal((K, T, 2)) for r in prop]))
        Xs = cp.repeat(X[:, None, :], K, axis=1)
        eps = cp.empty((S, K, T, 2))
        run = cp.zeros((S, K))
        M = cp.zeros((S, K))
        exit_ = cp.zeros((S, K), bool)
        for t in range(T):
            cc, bb = rows(cp, Xs, c_["sigma_env"])                 # rows at the rollout's own state
            if kind == "paper":
                ubk = cp.repeat(U[:, t, 0][:, None], K, axis=1)
                mm, ss, _ = solve_paper(cp, ubk, cc, bb, s0, z, z, "std")
                ev = mm + ss * xi[:, :, t, 0]
            else:
                ev = s0 * xi[:, :, t, 0]
            eps[:, :, t, 0] = ev
            eps[:, :, t, 1] = s_om * xi[:, :, t, 1]
            u_v = U[:, None, t, 0] + ev
            ok = (cc * u_v[..., None] >= bb - 1e-7 * (1.0 + cp.abs(bb))).all(-1)
            M = M + (~ok)
            Ut = U[:, None, t, :] + eps[:, :, t, :]
            Xs = step(cp, Xs, Ut, DT)
            exit_ = exit_ | (~inside(cp, Xs))
            run = run + ((Xs[..., :2] - goal) ** 2).sum(-1) + pen * (~inside(cp, Xs))
        live = (~done)[:, None]
        Mi = M.astype(cp.int32)
        nM += cp.bincount(Mi[cp.broadcast_to(live, Mi.shape)].ravel(), minlength=T + 1)
        nE += cp.bincount(Mi[cp.broadcast_to(live, Mi.shape) & exit_].ravel(), minlength=T + 1)

        Sc = run + ((Xs[..., :2] - goal) ** 2).sum(-1) \
             + ((U[:, None] * R * eps).sum(-1) + 0.5 * (U[:, None] ** 2 * R).sum(-1)).sum(-1)
        logw = -(Sc - Sc.min(axis=1, keepdims=True)) / lam
        logw -= logw.max(axis=1, keepdims=True)
        w = cp.exp(logw); w /= w.sum(axis=1, keepdims=True)
        U = U + (w[:, :, None, None] * eps).sum(1)
        noise = cp.asarray(np.stack([r.standard_normal(3) for r in plant]))
        Xn = step(cp, X, U[:, 0, :], DT) + c_["sigma_env"] * np.sqrt(DT) * noise
        X = cp.where(done[:, None], X, Xn)
        done = done | (cp.sqrt(((X[:, :2] - goal) ** 2).sum(-1)) < c_["goal_radius"])
        U = cp.concatenate([U[:, 1:], cp.zeros((S, 1, 2))], axis=1)
        if bool(done.all()):
            break
    return cp.asnumpy(nM), cp.asnumpy(nE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--kind", default="mppi", choices=("mppi", "paper"))
    a = ap.parse_args()
    cp = load_cupy()
    nM, nE = measure(cp, list(range(a.seeds)), kind=a.kind)
    tot, ext = nM.sum(), nE.sum()
    EM = float((np.arange(len(nM)) * nM).sum() / tot)

    def cond(lo, hi=None):
        s = slice(lo, hi)
        d, n = nM[s].sum(), nE[s].sum()
        return (n / d if d else float("nan")), d

    print(f"\n{a.kind}, {a.seeds} seeds, K={CFG['K']}, T={CFG['T']}: "
          f"{tot:,.0f} scored rollouts\n")
    print(f"  E[M], violating timesteps per rollout       {EM:10.4f} of {CFG['T']}")
    print(f"  Pr(rollout leaves the safe set)             {ext / tot:10.6f}")
    print(f"  Pr(M >= 1), i.e. the certificate fires      {1 - nM[0] / tot:10.6f}\n")
    for nm, lo, hi in (("M = 0", 0, 1), ("M >= 1", 1, None), ("M >= 2", 2, None),
                       ("M >= 4", 4, None), ("M >= 8", 8, None)):
        p, d = cond(lo, hi)
        print(f"  Pr(leaves the safe set | {nm:<7})            {p:10.6f}   ({d:,.0f} rollouts)")
    p1, _ = cond(1, None)
    print(f"\n  A row violation is a genuine safety event {p1 * 100:.2f}% of the time; "
          f"{(1 - p1) * 100:.2f}% of the\n  time the certificate refuses a rollout that never leaves "
          f"the safe set.  That number is\n  the price of a sufficient condition, and it is what "
          f"Algorithm 1's proposal is paying for.")


if __name__ == "__main__":
    main()
