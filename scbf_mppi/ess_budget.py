"""Budgeted SCBF: a per-sample problem whose importance weights exist and whose variance is bounded.

WHY THIS FILE EXISTS

Algorithm 1 of arXiv:2206.11985 reshapes the sampling distribution per sample and per timestep, and then
uses MPPI's cost-weighted average unchanged.  That average is a path-integral estimator only if the
samples are reweighted by the density ratio between the distribution actually sampled and the one the
derivation assumes.  The paper omits the reweighting, and dismisses the consequence in one sentence as
"sub-optimal" and "more conservative".

It is neither.  The per-sample problem's objective rewards shrinking the proposal's standard deviation,
because a smaller spread makes the chance constraint easier to meet, and its optimum reaches s = 0 on the
majority of sample-timesteps.  A proposal with zero variance is a point mass; it has no density with
respect to the Gaussian the derivation assumes, so the weight that would repair the update does not exist.
Measured in the corridor, the asymptotic effective sample size of the corrected estimator is one, at
every sample count from 500 to 10^6, while plain MPPI's grows linearly (125 -> 212,326).

THE FIX, IN TWO STATEMENTS.  Neither is new mathematics.  Both are standard importance-sampling facts,
written out here because the argument needs them exact and verified, not because nobody had them; see
Owen, Monte Carlo Theory Methods and Examples, ch. 9, and Sanz-Alonso & Wang, Prop. 3.

Theorem A (variance feasibility), SCALAR CASE.  Let p = N(0, s0^2) and q = N(m, s^2), and let w = p/q.
Then E_q[w^2] < infinity if and only if s > s0/sqrt(2).  Proof: the integrand of int p^2/q has exponent
x^2 (1/(2 s^2) - 1/s0^2) + ..., which is integrable exactly when 1/(2 s^2) < 1/s0^2.

  The 1/sqrt(2) is the scalar threshold ONLY, i.e. Sigma_q = f^2 Sigma_p.  In general the condition is
  the matrix one, 2 Sigma_q - Sigma_p positive definite, and an anisotropic or oblique shrink can fail
  it while its largest ratio still sits above 1/sqrt(2): diag(0.9^2, 0.6^2) against the identity has
  eigenvalues (0.62, -0.28) and is therefore infinite.  On the three-input vessel the rank-one update of
  solver_nd is oblique in the Sigma0 metric, so the effective whitened eigenvalue is smaller than f^2
  suggests.  Quote the matrix condition, not the scalar number, whenever the channel is not scalar.

Theorem B (closed form).  For beta = s/s0 > 1/sqrt(2),

        E_q[w^2] = beta^2 / sqrt(2 beta^2 - 1) * exp( m^2 / (s0^2 (2 beta^2 - 1)) )

and for self-normalised importance sampling ESS/K -> 1/E_q[w^2].  Verified against 4x10^7 Monte Carlo
draws to a relative error of 2e-5 away from the boundary.

  That arrow is an ASYMPTOTIC statement about the population quantity 1/(1 + chi^2), not a cap on any
  finite run.  A measured ESS can sit far above it: for a proposal that attains the bound by moving mass
  off the violation set, every weight is equal whenever no sample lands in that set, which at K = 500 and
  delta = 0.003 happens in 21.5 % of runs and returns a measured ESS of exactly 500.  Report it as an
  asymptotic efficiency, never as "at most N usable samples in this run".

Over a horizon of T steps the per-step ratios multiply, so a per-step budget kappa gives
E[w^2] <= kappa^T and hence ESS/K >= kappa^(-T) asymptotically.  This is a CONSTANT fraction of K, so the
effective sample size grows linearly in the sample count again, which is the property Algorithm 1 destroys.

WHAT THIS BUDGET DOES AND DOES NOT BUY, measured over 512 corridor seeds (tests/gpu_closed_loop.py).
It does what it is designed to do: the weight budget binds to five decimals and the effective sample size
rises from 2.0 to 65.  It does NOT improve the controller.  On every closed-loop metric the budgeted
controller lands on plain MPPI (97.5 % reached, median min h -0.049, 0 runaway episodes of 512), while the
degenerate estimator it was meant to repair reaches the goal every time with a median min h of +0.167.
Restoring the estimator removes the advantage, because in this problem the advantage comes from the
near-argmax selection that the degeneracy produces.  That is the finding; the repair is the instrument
that isolates it.  See BudgetedSCBFMPPI below, and the intervention penalty in tests/gpu_closed_loop.py
for the one-parameter family that spans the two regimes.

THE ALGORITHM

Per sample and per timestep, with barrier rows c_k u >= b_k, nominal ubar, base std s0 and quantile z:

    minimise    |m| + (s0 - s)                        the paper's own objective
    subject to  c_k (ubar + m) - z |c_k| s >= b_k     the chance constraint, corrected to a std
                beta^2/sqrt(2 beta^2 - 1) exp(m^2/(s0^2 (2 beta^2 - 1))) <= kappa    the weight budget
                s <= s0,   beta = s/s0

The budget is a closed-form box on the mean shift for each beta,

    |m| <= m_max(beta) = s0 sqrt( (2 beta^2 - 1) * log( kappa sqrt(2 beta^2 - 1) / beta^2 ) ),

so the problem is a one-dimensional search over beta, exactly like the original, and stays vectorised.

WHEN IT IS INFEASIBLE, WHICH IS THE HONEST PART

Tightening delta and holding a sample budget are in genuine conflict: the chance constraint wants a small
spread and a large mean shift, and both spend weight budget.  For some states no beta satisfies both.  The
paper's formulation hides this by taking s = 0 and silently forfeiting the estimator.  Here the solver
instead returns the point that MAXIMISES the delivered probability within budget, and reports that
probability.  An honest 0.97 that keeps the estimator is worth more than a nominal 0.997 that does not.
"""
import numpy as np

SQRT_HALF = 1.0 / np.sqrt(2.0)


def m_max_of_beta(beta, s0, kappa):
    """The weight budget, solved for the largest admissible mean shift at each beta.

    E[w^2] <= kappa  <=>  |m| <= s0 sqrt( d * log(kappa sqrt(d) / beta^2) ),  d = 2 beta^2 - 1.

    Returns NaN where no mean shift at all fits, which happens in two ways and both must be excluded
    rather than clamped to zero.  Below beta = 1/sqrt(2) the second moment is infinite.  And above it,
    but at small beta, the shrinkage ALONE already costs more than the budget: beta^2/sqrt(d) > kappa,
    so the logarithm is negative and even m = 0 is inadmissible.  Returning 0 there, as an earlier
    version did, silently admitted proposals whose weight variance was eight times the budget."""
    beta = np.asarray(beta, float)
    d = 2.0 * beta ** 2 - 1.0
    ok = d > 0
    dd = np.where(ok, d, 1.0)
    inner = np.log(np.maximum(kappa * np.sqrt(dd) / np.maximum(beta ** 2, 1e-300), 1e-300))
    admissible = ok & (inner > 0.0)
    return np.where(admissible, s0 * np.sqrt(np.maximum(dd * inner, 0.0)), np.nan)


def ess_fraction_bound(kappa, T):
    """The guarantee the budget buys: ESS/K >= kappa^(-T) asymptotically."""
    return float(kappa) ** (-int(T))


def kappa_for_target(ess_frac, T):
    """Inverse of the above: the per-step budget that delivers a target ESS fraction over T steps."""
    return float(ess_frac) ** (-1.0 / int(T))


def solve_rows_budgeted(ubar_v, c, b, s0, z, kappa, n_beta=193, beta_eps=1e-3):
    """Vectorised over K samples, two rows, one constrained channel.

    ubar_v: (K,)   nominal value of the constrained input channel
    c:      (K,R)  row coefficients in that channel
    b:      (K,R)  right-hand sides
    s0:     scalar base standard deviation of that channel
    z:      scalar Phi^{-1}(1 - delta)
    kappa:  scalar per-step budget on E_q[w^2]

    Returns m, s, active, within_budget, delivered  (each (K,)), where `delivered` is the probability the
    returned proposal actually gives the row that binds hardest -- equal to 1-delta when the chance
    constraint was met, and the best achievable within budget otherwise.
    """
    K = ubar_v.shape[0]
    # beta grid strictly inside the feasible half-line, plus the unmodified point beta = 1
    beta = np.linspace(SQRT_HALF + beta_eps, 1.0, n_beta)
    s_grid = beta * s0                                              # (n_beta,)
    mmax = m_max_of_beta(beta, s0, kappa)                           # (n_beta,)

    cc = c[:, :, None]                                              # (K,R,1)
    bb = b[:, :, None]
    ub = ubar_v[:, None, None]
    r = bb + z * np.abs(cc) * s_grid[None, None, :] - cc * ub       # need c*m >= r
    pos = cc > 1e-12
    neg = cc < -1e-12
    zero = ~(pos | neg)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = r / cc
    lo = np.where(pos, ratio, -np.inf).max(axis=1)                  # (K,n_beta)
    hi = np.where(neg, ratio, np.inf).min(axis=1)
    zero_ok = np.where(zero, r <= 0, True).all(axis=1)

    # intersect the row interval [lo, hi] with the budget box [-mmax, +mmax]
    LO = np.maximum(lo, -mmax[None, :])
    HI = np.minimum(hi, mmax[None, :])
    feas = (LO <= HI) & zero_ok & np.isfinite(mmax)[None, :]
    m_in = np.clip(0.0, LO, HI)                                     # smallest |m| inside
    m_in = np.where(np.isfinite(m_in), m_in, 0.0)

    cost = np.abs(m_in) + (s0 - s_grid[None, :]) + np.where(feas, 0.0, 1e9)
    j = cost.argmin(axis=1)
    idx = np.arange(K)
    ok_any = feas[idx, j]

    # ---- fallback: no beta satisfies both.  Maximise the delivered probability within budget. ----
    # delivered for row k is Phi( (c_k(ubar+m) - b_k) / (|c_k| s) ); the binding row is the min over k.
    mb = np.clip(np.where(np.isfinite(lo), lo, 0.0), -mmax[None, :], mmax[None, :])   # push to the wall
    mb = np.where(np.isfinite(mb), mb, 0.0)
    marg = (cc * (ub + mb[:, None, :]) - bb) / np.maximum(np.abs(cc) * s_grid[None, None, :], 1e-300)
    zmin = marg.min(axis=1)                                          # (K,n_beta) worst row, in sigmas
    zmin = np.where(np.isfinite(mmax)[None, :], zmin, -np.inf)
    jb = zmin.argmax(axis=1)

    m = np.where(ok_any, m_in[idx, j], mb[idx, jb])
    s = np.where(ok_any, s_grid[j], s_grid[jb])
    zeff = np.where(ok_any, z, zmin[idx, jb])
    from scipy.stats import norm
    delivered = norm.cdf(np.clip(zeff, -10.0, 10.0))

    # active: would the unmodified proposal (m = 0, s = s0) already satisfy the rows?
    active = ~(zero_ok[:, -1] & (lo[:, -1] <= 0.0) & (hi[:, -1] >= 0.0))
    return {"m": m, "s": s, "active": active, "within_budget": ok_any, "delivered": delivered}


# ==================================================================================================
# The controller.  Same MPPI, same barrier, same cost; the per-sample problem is the budgeted one and
# the importance weights are ALWAYS applied, because with the budget in place they exist.
# ==================================================================================================
class BudgetedSCBFMPPI:
    """SCBF-MPPI with a weight budget, wrapping the repository's own MPPI.

    Differences from Algorithm 1, and only these:
      * the chance constraint uses the standard deviation, as Theorem 2's own proof requires;
      * the proposal is confined to the set where the importance weight has finite second moment, with
        a per-step budget kappa chosen from a target ESS fraction and the horizon;
      * the importance weights are applied, because they now exist;
      * where the budget cannot deliver 1 - delta, the solver returns the best probability it can and
        records it, instead of taking a point mass and forfeiting the estimator.
    """

    def __init__(self, env, ess_target=0.10, delta=0.003, sigma_env=1.0, **kw):
        from .mppi import MPPI
        from .scbf import quantile
        self._base = MPPI(env, **kw)
        self.env = env
        self.sigma_env = sigma_env
        self.delta = delta
        self.z = quantile(delta)
        self.ess_target = float(ess_target)
        self.kappa = kappa_for_target(ess_target, self._base.T)
        # forward the attributes the base plan() touches
        for a in ("K", "T", "lam", "dt", "Sigma0", "R", "goal", "penalty", "u_max", "rng", "U"):
            setattr(self, a, getattr(self._base, a))
        self.last = {}

    q = property(lambda self: self._base.q)
    phi = property(lambda self: self._base.phi)
    control_terms = property(lambda self: self._base.control_terms)

    def sample_eps(self, x0):
        from .dynamics import step
        K, T = self.K, self.T
        s0 = float(np.sqrt(self.Sigma0[0]))
        s_om = float(np.sqrt(self.Sigma0[1]))
        eps = np.empty((K, T, 2))
        X = np.broadcast_to(x0, (K, 3)).copy()
        traj = np.empty((K, T + 1, 3))
        traj[:, 0] = X
        xi = self.rng.standard_normal((K, T, 2))
        logq = np.zeros(K)
        n_active = 0
        delivered = []
        in_budget = []
        for t in range(T):
            c, b = self.env.scbf_rows(X, self.sigma_env)
            res = solve_rows_budgeted(np.full(K, self.U[t, 0]), c, b, s0, self.z, self.kappa)
            m, s = res["m"], res["s"]
            n_active += int(res["active"].sum())
            delivered.append(res["delivered"])
            in_budget.append(res["within_budget"])
            ev = m + s * xi[:, t, 0]
            eps[:, t, 0] = ev
            eps[:, t, 1] = s_om * xi[:, t, 1]
            logq += ((-0.5 * (ev / s0) ** 2 - np.log(s0))
                     - (-0.5 * ((ev - m) / s) ** 2 - np.log(s)))
            U = self.U[t][None] + eps[:, t]
            if self.u_max is not None:
                U = np.clip(U, -self.u_max, self.u_max)
            X = step(X, U, self.dt)
            traj[:, t + 1] = X
        info = {"activation_frac": n_active / (K * T),
                "delivered_mean": float(np.mean(delivered)),
                "delivered_min": float(np.min(delivered)),
                "in_budget_frac": float(np.mean(in_budget)),
                "logq": logq}
        return eps, traj, info

    def plan(self, x0):
        x0 = np.asarray(x0, float)
        self._base.U = self.U
        eps, traj, info = self.sample_eps(x0)
        S = self.q(traj[:, 1:]).sum(-1) + self.phi(traj[:, -1]) + self.control_terms(self.U, eps)
        logw = -(S - S.min()) / self.lam + info.pop("logq")
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        self.U = self.U + (w[:, None, None] * eps).sum(0)
        if self.u_max is not None:
            self.U = np.clip(self.U, -self.u_max, self.u_max)
        u0 = self.U[0].copy()
        self.last = {"ess": float(1.0 / (w ** 2).sum()), "w_max": float(w.max()),
                     "frac_unsafe_samples": float((~self.env.inside(traj[:, 1:])).any(-1).mean()),
                     "traj": traj, "w": w, "S": S, **info}
        self.U = np.vstack([self.U[1:], np.zeros((1, 2))])
        return u0
