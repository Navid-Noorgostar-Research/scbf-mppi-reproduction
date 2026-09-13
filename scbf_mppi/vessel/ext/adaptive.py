"""A temperature that holds the effective sample size (experiment V12).

Why.  Algorithm 1 of the paper changes the proposal distribution of the rollouts but weights them with the
plain MPPI weights, i.e. it omits the importance-sampling correction that the change of proposal requires.
Putting the correction back is arithmetically easy, and the reproduction does it -- but the corrected
estimator then runs on a median effective sample size of 13 out of K = 500 in the vessel scenario.  An
estimator that uses 13 of 500 samples is not an estimator.  This module asks whether that is fatal or merely
a tuning problem: can the temperature be chosen on line so the corrected weights keep a usable ESS?

The algebra that makes it cheap.  In controllers.py the sample cost is

    S = J + lam C,        J = sum_t q + phi   (independent of lam),        C = sum_t (u^T Sigma0^-1 eps + u^T Sigma0^-1 u / 2)

because R = lam / Sigma0.  The weights are w ∝ exp(-(S - min S)/lam + logq), so

    log w(lam)  =  - J / lam  -  C  +  logq  +  const

-- the temperature enters only through the state cost, and the control term and the importance weight are
fixed offsets.  One rollout batch therefore gives the whole family {w(lam)} for free: no re-sampling, no
re-simulation.  This keeps lam consistent in the cost and in the weights, which is what the free-energy
derivation of MPPI requires; decoupling them (tempering the weights while the cost keeps lam0) would make
the update no longer the minimiser of the KL bound it is derived from, so it is not done here.

Monotonicity.  As lam -> 0 the weights collapse onto argmin J and ESS -> 1.  As lam -> infinity the state
cost drops out and w -> softmax(logq - C), so ESS tends to a CEILING set by the importance weights alone,
not to K.  ESS is therefore bounded above by something the temperature cannot change, and the search must
report when the target is unreachable -- which is the interesting outcome, not a failure of the search.  The
implementation does a log-spaced grid followed by bisection between the bracketing points, so it is correct
whether or not ESS happens to be monotone on a given cycle.

The honest failure mode.  Raising lam to buy ESS flattens the weights toward the proposal mean, and a
controller whose weights are flat does not steer: the MPPI update (w . eps) shrinks toward zero.  A "safe"
result obtained that way is safe the way a moored boat is safe.  V12 therefore reports, beside ESS and the
collision statistics, the norm of the control update per cycle and the fraction of episodes that reach the
goal.  If ESS goes up while ||dU|| and the reached fraction go down, the correction has not been rescued.
"""
import numpy as np


def _ess_from_logw(logw):
    m = logw.max()
    if not np.isfinite(m):
        return 1.0
    w = np.exp(logw - m)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return 1.0
    w = w / s
    return float(1.0 / (w ** 2).sum())


class AdaptiveTemperatureMixin:
    """Choose lam each cycle so that ESS(lam) ~ ess_target.  ess_target=None restores the fixed temperature."""

    def __init__(self, *a, ess_target=None, lam_bounds=(1.0, 1.0e9), n_grid=64, n_bisect=20, **kw):
        self.ess_target = None if ess_target is None else float(ess_target)
        self.lam_bounds = (float(lam_bounds[0]), float(lam_bounds[1]))
        self.n_grid = int(n_grid); self.n_bisect = int(n_bisect)
        self.lam_eff = None; self.target_met = None
        super().__init__(*a, **kw)

    def _logw(self, lam, Jc, C, logq):
        return -Jc / lam - C + logq

    def weights(self, S, info):
        if self.ess_target is None:
            self.lam_eff = float(self.lam); self.target_met = None
            return super().weights(S, info)

        ct = getattr(self, "_ctrl_terms", None)
        logq = info.get("logq", None)
        logq = np.zeros_like(S) if logq is None else np.asarray(logq, float)
        if ct is None:
            J = np.asarray(S, float); C = np.zeros_like(J)
        else:
            ct = np.asarray(ct, float)
            J = np.asarray(S, float) - ct
            C = ct / self.lam                       # C is lam-free; ct was built with self.lam
        Jc = J - J.min()

        lo, hi = self.lam_bounds
        grid = np.exp(np.linspace(np.log(lo), np.log(hi), self.n_grid))
        ess = np.array([_ess_from_logw(self._logw(l, Jc, C, logq)) for l in grid])
        tgt = self.ess_target

        if ess.max() <= tgt:                        # the ceiling set by the importance weights is below target
            lam = float(grid[int(np.argmax(ess))]); self.target_met = False
        elif ess.min() >= tgt:                      # even the coldest temperature is above target
            lam = float(grid[int(np.argmin(ess))]); self.target_met = True
        else:
            k = int(np.argmax(ess >= tgt))          # first grid point at or above the target
            a = grid[k - 1] if k > 0 else lo
            b = grid[k]
            for _ in range(self.n_bisect):          # bisection in log lam
                m = float(np.sqrt(a * b))
                if _ess_from_logw(self._logw(m, Jc, C, logq)) >= tgt:
                    b = m
                else:
                    a = m
            lam = float(b); self.target_met = True

        self.lam_eff = lam
        logw = self._logw(lam, Jc, C, logq)
        logw = logw - logw.max()
        w = np.exp(logw)
        tot = w.sum()
        if not np.isfinite(tot) or tot <= 0:        # degenerate cycle: fall back to the fixed temperature
            self.lam_eff = float(self.lam); self.target_met = False
            return super().weights(S, info)
        return w / tot

    def plan(self, x0):
        u = super().plan(x0)
        self.last["lam_eff"] = float(self.lam_eff if self.lam_eff is not None else self.lam)
        self.last["ess_target_met"] = self.target_met
        return u
