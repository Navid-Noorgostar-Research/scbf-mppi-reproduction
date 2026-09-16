"""What the defensive-mixture escape actually buys: coverage, but not accuracy.

    python tests/mixture_recovery.py

WHY THIS EXISTS.  Section 2b of FINDINGS.md answers the first objection anyone who knows importance
sampling will raise -- use a mixture and the infinite weight variance disappears -- and answers it with
algebra: q = w p + (1 - w) q1 gives dp/dq <= 1/w (Hesterberg 1995), the chance constraint caps
w <= delta/delta0, and the second moment lands between 188 and 250.  That bounds the PRICE of the
escape.  It says nothing about whether the escaped estimator gets the right answer, and the obvious
follow-up -- so does the mixture work? -- had no measurement behind it.

WHY THIS IS NOT MEASURED ON THE CORRIDOR.  It cannot be.  Judging an estimator needs a target whose
value is known, and the corridor's full MPPI target is not knowable: the probes at that state pooled an
effective sample size of 1.17 out of two million draws, with a single weight carrying 92% of the mass.
That is why this uses a constructed scalar problem whose target is exactly Gaussian, and it is the
reason the numbers below are about a mechanism rather than about the corridor.

THE PROBLEM.  x_{t+1} = x_t + 0.05 v_t, x_0 = 0, safe box [-0.5, 0.5], goal 0.4, T = 20, nominal
controls iid N(0,1), cost sum_t (x_t - 0.4)^2 + (x_T - 0.4)^2, lambda = 1.  Writing positions as x = B v
with W = diag(1,...,1,2), the target is exactly N(mu, H^-1) with H = I + 2 B' W B and mu = H^-1 2 B' W g1,
so the first-control mean is known in closed form.  The rows are the discrete barrier
h(x_+) >= (1 - alpha) h(x) at alpha = dt = 0.05, that is, a continuous-time class-K gain of 1.

THE PROPOSALS.  q_safe draws each step from the nominal density restricted to one of three regions --
the admissible interval, the left tail, the right tail -- with the two tails given probability d_tail
each, so its per-row violation rate is exactly d_tail.  The mixture picks a source per rollout.  Its
weights are the exact path-density ratio against the mixture, so every estimate below is unbiased in K.

WHAT IT SHOWS.  A mixture of PATH measures is not a per-step mixture of kernels: along a prefix that has
already violated, the posterior over sources reverts to the nominal component, so the mixture meets the
MARGINAL per-step cap while breaking the conditional one.  Coverage therefore returns at an admissible
weight.  Accuracy does not: the estimate stays put and the error gets worse, because the rare nominal
draws arrive carrying enormous weight.  Recovering the target mean needs w near 0.10, whose per-row
violation rate is about ten times the cap the chance constraint allows.
"""
import numpy as np
from scipy.stats import norm

N, DT, GOAL, DELTA = 20, 0.05, 0.4, 0.003
ALPHA = DT                      # h(x_+) >= (1 - alpha) h(x); alpha = gamma dt with gamma = 1
SQ2 = np.sqrt(2.0)
LOG2PI = np.log(2 * np.pi)


def exact_target():
    B = DT * np.tril(np.ones((N, N)))
    W = np.diag([1.0] * (N - 1) + [2.0])
    H = np.eye(N) + 2 * B.T @ W @ B
    return np.linalg.solve(H, 2 * B.T @ W @ (GOAL * np.ones(N)))


def run(nb, K, d_tail, w, rng):
    """nb independent batches of K rollouts from q = w p + (1 - w) q_safe, weighted exactly."""
    src_p = rng.random((nb, K)) < w
    x = np.zeros((nb, K))
    lp = np.zeros((nb, K))
    lq = np.zeros((nb, K))
    S = np.zeros((nb, K))
    M = np.zeros((nb, K))
    rate = 0.0
    v1 = None
    for t in range(N):
        u = 20 * ALPHA * (0.5 - x)
        l = -20 * ALPHA * (0.5 + x)
        Fu, Fl = norm.cdf(u), norm.cdf(l)
        m_mid = np.maximum(Fu - Fl, 1e-300)
        m_lo = np.maximum(Fl, 1e-300)
        m_hi = np.maximum(1 - Fu, 1e-300)
        r, U = rng.random((nb, K)), rng.random((nb, K))
        lo_s, hi_s = r < d_tail, (r >= d_tail) & (r < 2 * d_tail)
        p_q = np.where(lo_s, U * m_lo, np.where(hi_s, Fu + U * m_hi, Fl + U * m_mid))
        v = np.where(src_p, rng.standard_normal((nb, K)),
                     norm.ppf(np.clip(p_q, 1e-300, 1 - 1e-16)))
        in_lo, in_hi = v < l, v > u                       # region decided by where v LANDED
        in_mid = ~(in_lo | in_hi)
        mass = np.where(in_lo, m_lo, np.where(in_hi, m_hi, m_mid))
        asg = np.where(in_mid, 1 - 2 * d_tail, d_tail)
        lphi = -0.5 * v * v - 0.5 * LOG2PI
        lp += lphi
        lq += lphi + np.log(asg) - np.log(mass)
        viol = (in_lo | in_hi).astype(float)
        M += viol
        rate += viol.mean()
        if t == 0:
            v1 = v
        x = x + DT * v
        S += (x - GOAL) ** 2
    S += (x - GOAL) ** 2                                  # terminal counted twice: W = diag(1..1,2)
    lqm = np.logaddexp(np.log(max(w, 1e-300)) + lp, np.log(1 - w) + lq)
    lw = -S + lp - lqm
    lw -= lw.max(axis=1, keepdims=True)
    ww = np.exp(lw)
    ww /= ww.sum(axis=1, keepdims=True)
    return ((ww * v1).sum(axis=1), 1.0 / (ww ** 2).sum(axis=1),
            (M.max(axis=1) >= 4).mean(), rate / N / 2.0)


def main():
    truth = exact_target()[0]
    rng = np.random.default_rng(3)
    print("")
    print("Exact first-control mean of the full target: %.6f" % truth)
    print("Per-row chance cap delta = %.3f; a mixture is admissible only at or below it." % DELTA)
    print("")
    print("%9s%8s%11s%9s%9s%18s%16s"
          % ("w", "d_tail", "mean est", "RMSE", "ESS/500", "batches w/ M>=4", "per-row rate"))
    for w, dt_ in ((0.0, 0.003), (0.0, 0.0015), (0.0015, 0.0015),
                   (0.0049, 0.0015), (0.02, 0.0015), (0.10, 0.0015)):
        e, ess, hit, rate = run(2000, 500, dt_, w, rng)
        flag = "" if rate <= DELTA * 1.001 else "  over cap"
        print("%9.4f%8.4f%11.6f%9.4f%9.2f%17.2f%%%14.5f%s"
              % (w, dt_, e.mean(), np.sqrt(((e - truth) ** 2).mean()), ess.mean(),
                 100 * hit, rate, flag))
    print("")
    print("  At w = 0.0015 the mixture is admissible and coverage is restored -- the class the")
    print("  chance-constrained proposal could not reach now appears in half the batches.  The")
    print("  estimate does not move and the error gets worse.  Only at w = 0.10, ten times")
    print("  the cap, does the mean come back.  The escape in section 2b buys support, not accuracy.")


if __name__ == "__main__":
    main()
