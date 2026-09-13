"""The barrier as an OUTPUT safety filter instead of a sampler modification (experiment V10).

The research question the critique opens.  arXiv:2206.11985 pushes the barrier into the sampling
distribution of MPPI: every rollout sample is shifted and its covariance shrunk so that the stochastic CBF
condition holds with probability 1 - delta.  The alternative, and the one a model-predictive-control group
would reach for first, is to leave the controller alone and project its OUTPUT onto the barrier condition
once per control interval -- the standard CBF quadratic program.  The two are not variants of one idea: one
pays K x T constrained solves per cycle to bias 500 rollouts, the other pays a single constrained solve per
cycle and touches only the input that is actually applied.  V10 measures what the extra 7500 solves buy.

Filter 1, nominal CBF-QP:

    min_u  || u - u_mppi ||^2_W  +  rho 1^T s
    s.t.   (a_j B) u  >=  b_j - s_j          for every obstacle j      (the same rows harbour.rows returns)
           || (u_X, u_Y) ||  <=  700 N                                 (the azimuth force disk)
           | u_BT |  <=  244.4 exp(-0.62 u_surge^2) N                  (the speed-dependent bow thruster)
           s >= 0

W = diag(1/s0^2) with s0 the MPPI sampling std (350, 350, 120) N, so "smallest change" is measured in the
same metric the controller samples in; using the identity instead would silently make the bow thruster
three times more expensive to move than the azimuth thruster, for no reason.  The slack s with a large
linear penalty rho is an exact penalty: it is zero whenever the constraint set is non-empty, so the filter
is ALWAYS feasible and never throws.

Conditioning.  Solved in the raw variables the problem is badly scaled: the row coefficients a B are of
order 3e-4 per newton while the inputs are of order 700 N and the cone radius is 700, which is enough to
make an interior-point solver report an inaccurate solution.  It is therefore solved in the dimensionless
step v = (u - u_mppi) / s0, and every row is divided by the norm of its coefficient vector in those
units.  The objective is then ||v||^2 (one sampling std costs 1) and the slack is measured in the same
units (one std of missing margin costs rho), so rho = 100 is an honest exact penalty rather than a number
chosen to fight the scaling.

Filter 2, chance-constrained CBF-QP.  The disturbance enters as a generalised force, so the row under
disturbance is a (tau + tau_d) >= b and the tightening is exact rather than a bound:

    a tau - b  >=  z sqrt(a Sigma_d a^T),      z = Phi^{-1}(1 - delta)

with a in TAU coordinates (not u coordinates -- the disturbance is a force, it does not pass through the
thruster allocation).  Sigma_d depends on which disturbance model is meant, and this is where the paper's own
assumption bites:

  coloured gust (OU, tau_c = 5 s):  tau_d is an honest force with stationary covariance diag(s_F^2),
        s_F = (120 N, 300 N, 400 N m).   Sigma_d = diag(s_F^2).
  white force noise:  tau_d is a distribution, not a function, and the instantaneous chance constraint is
        not well posed.  What is well posed is the average force over one control interval h, whose
        covariance is sigma_F^2 / h with sigma_F = s_F sqrt(2 tau_c), i.e. diag((s_F sqrt(2 tau_c / h))^2).
        At tau_c = 5 s and h = 1 s that is sqrt(10) = 3.2 times the gust tightening.

That factor is not a detail: the model the paper assumes (white) demands a three times larger safety margin
than the disturbance a vessel actually sees (a gust with a correlation time), for the same delta.  Both are
computed here and V10 reports both.

What would make this comparison a strawman, and what is done about it.  (i) Giving the filter a different
MPPI: it gets the identical one, same seeds, same cost, same K.  (ii) Giving the filter a different barrier:
it uses harbour.rows, the same second-order rows with the same alpha1, alpha2 that the SCBF sampler uses.
(iii) Letting the filter act on a horizon while the sampler acts per sample: that asymmetry is the point of
the experiment and is reported as such, not hidden.  (iv) Input limits: the filter respects the real
azimuth disk and bow-thruster cap, so it cannot buy safety with thrust the vessel does not have.

Known limitation, stated rather than hidden: the inner MPPI is not told that its output was filtered, so its
warm start is the unfiltered plan.  This is the standard cascade, and it re-plans from the measured state
every second, so the stale warm start is a second-order effect -- but it is an effect, and a controller
co-designed with its filter would do better than the numbers reported here.
"""
import numpy as np

from .. import solgenia as sg
from .. import solver_nd

try:
    import cvxpy as cp
    _HAVE_CVXPY = True
except Exception:                                                   # pragma: no cover
    _HAVE_CVXPY = False


class CBFQPFilter:
    """Projection of an input onto the second-order barrier rows and the true input set."""

    def __init__(self, harbour, s0=(350.0, 350.0, 120.0), delta=None, disturbance="white",
                 s_F=(120.0, 300.0, 400.0), tau_c=5.0, dt=sg.DT_CTRL, rho=100.0, solver="CLARABEL"):
        if not _HAVE_CVXPY:
            raise RuntimeError("the CBF-QP filter needs cvxpy (already a requirement of this package)")
        self.H = harbour
        self.s0 = np.asarray(s0, float)
        self.J = len(harbour.obs)
        self.delta = delta                                   # None -> nominal filter, no tightening
        self.z = 0.0 if delta is None else float(solver_nd.quantile(delta))
        self.dt = float(dt)
        s_F = np.asarray(s_F, float)
        if disturbance == "ou":
            self.sig_d = s_F.copy()                             # stationary std of the OU force
        elif disturbance == "white":
            self.sig_d = s_F * np.sqrt(2.0 * tau_c / self.dt)   # std of the mean force over one interval
        else:
            self.sig_d = np.zeros(3)
        self.rho = float(rho)
        self.solver = solver
        self._build()
        self.n_solve = 0
        self.n_fail = 0
        self.n_slack = 0
        self.n_active = 0

    def _build(self):
        J = self.J
        v = cp.Variable(3, name="v")                     # u = u_ref + s0 * v, so ||v|| is in sampling stds
        s = cp.Variable(J, nonneg=True, name="s")
        self.p_G = cp.Parameter((J, 3), name="G")        # row coefficients on v, each row unit norm
        self.p_r = cp.Parameter(J, name="r")             # required margin in the same units
        self.p_uref = cp.Parameter(3, name="uref")
        self.p_cap = cp.Parameter(nonneg=True, name="cap")
        u = self.p_uref + cp.multiply(self.s0, v)
        obj = cp.Minimize(cp.sum_squares(v) + self.rho * cp.sum(s))
        cons = [self.p_G @ v >= self.p_r - s,
                cp.norm(u[:2], 2) <= sg.F_AT_MAX,
                cp.abs(u[2]) <= self.p_cap]
        self.prob = cp.Problem(obj, cons)
        self._v = v
        self._s = s

    def __call__(self, x, t, u_ref):
        """x: (6,) measured state, t: absolute time, u_ref: (3,) the controller's input.  -> (u, info)"""
        x = np.asarray(x, float)
        u_ref = np.asarray(u_ref, float)
        a_tau, b, h, psi1 = self.H.rows(x[None], t)             # a in tau coordinates
        a_tau = a_tau[0]
        b = b[0]                                                # (J,3), (J,)
        tight = self.z * np.linalg.norm(a_tau * self.sig_d[None, :], axis=-1) if self.z else np.zeros(self.J)
        A_u = a_tau @ sg.B_ALLOC                                # rows in u coordinates
        b_t = b + tight
        margin_ref = A_u @ u_ref - b_t
        active = bool((margin_ref < 0).any())
        info = {"filter_active": float(active), "filter_min_margin_ref": float(margin_ref.min()),
                "filter_h_min": float(h.min()), "filter_tighten_max": float(tight.max())}
        u_leg = sg.clip_inputs(u_ref[None], x[None])[0]      # the filter never emits an illegal input
        if not active:                                 # the controller's own input already satisfies every row
            self.n_solve += 1
            info.update({"filter_dU": 0.0, "filter_slack": 0.0, "filter_failed": 0.0})
            return u_leg, info
        self.n_active += 1
        # dimensionless step v = (u - u_ref)/s0 with each row scaled to unit coefficient norm
        g = np.linalg.norm(A_u * self.s0[None, :], axis=1)
        gs = np.maximum(g, 1e-12)
        self.p_G.value = (A_u * self.s0[None, :]) / gs[:, None]
        self.p_r.value = (b_t - A_u @ u_ref) / gs
        self.p_uref.value = u_ref
        self.p_cap.value = float(max(sg.bow_cap(x[3]), 1e-9))
        try:
            self.prob.solve(solver=self.solver, warm_start=True)
            ok = self._v.value is not None and np.all(np.isfinite(self._v.value))
        except Exception:
            ok = False
        self.n_solve += 1
        if not ok:                                     # does not happen in practice; degrade to the raw input
            self.n_fail += 1
            info.update({"filter_dU": 0.0, "filter_slack": 0.0, "filter_failed": 1.0})
            return u_leg, info
        u = u_ref + self.s0 * np.asarray(self._v.value, float)
        slack = float(np.asarray(self._s.value, float).sum())
        if slack > 1e-6:
            self.n_slack += 1
        u = sg.clip_inputs(u[None], x[None])[0]        # numerical safety: the cone is enforced exactly
        info.update({"filter_dU": float(np.linalg.norm((u - u_ref) / self.s0)), "filter_slack": slack,
                     "filter_failed": 0.0})
        return u, info


class FilteredController:
    """A controller plus an output filter, exposing the interface `simulate.run_episode` expects."""

    def __init__(self, inner, filt):
        self.inner = inner
        self.filt = filt
        self.dt = inner.dt
        self.H = inner.H
        self.last = {}

    @property
    def t(self):
        return self.inner.t

    @t.setter
    def t(self, v):
        self.inner.t = v

    def plan(self, x0):
        t_now = self.inner.t                           # absolute time of THIS control instant
        u_ref = self.inner.plan(x0)
        u, info = self.filt(x0, t_now, u_ref)
        self.last = dict(self.inner.last)
        self.last.update(info)
        return u
