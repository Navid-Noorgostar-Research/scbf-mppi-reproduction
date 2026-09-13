"""On-line estimate of the unknown water current (experiment V11).

The problem.  In the "gust + 0.5 m/s current" scenario the plant integrates

    pdot = R(psi) nu_xy + c,          c = (0, -0.5) m/s

while every controller rolls out with c = 0.  The controller's predicted position is therefore biased, and so
is its barrier derivative: for h_j = ||p - c_j|| - R_j the true hdot is n . (R nu + c - w_j) but the
controller computes n . (R nu - w_j), an error of n . c, up to 0.5 m/s.  With alpha1 = 0.1 1/s that is an
error of the same size as the entire alpha1 h term at h = 5 m.  The chance constraint's nominal
delta = 0.003 says nothing about this: delta bounds the effect of the noise the controller models, not of a
drift it does not model.  V11 measures the resulting gap and how much of it an observer closes.

Identifiability.  The controller receives the full state x = (x, y, psi, u, v, r) every second.  A constant
water current enters ONLY the position kinematics; a constant force bias enters only the velocity dynamics,
where it would change nu -- which is measured.  Forming the residual of the position kinematics with the
MEASURED nu therefore isolates c from any force disturbance: the two are distinguishable given this
measurement set.  (With position-only measurement over a short window they are not: a force bias produces a
ramp in position, a current a straight line, and the two are separated only by the second derivative.)

The estimator.  Over one control interval h the plant satisfies exactly

    p_{k+1} - p_k = integral over the step of R(psi(s)) nu_xy(s) ds  +  c h

so with the trapezoidal rule on the measured endpoints

    y_k = [ (p_{k+1} - p_k)/h ]  -  0.5 [ R(psi_k) nu_k + R(psi_{k+1}) nu_{k+1} ]  =  c  +  e_k

where e_k is the trapezoidal error of the body-velocity integral -- second order in h and driven by the
curvature of the velocity, i.e. by the gust.  The estimate is the first-order low pass

    chat_{k+1} = chat_k + L ( y_k - chat_k ),         L in (0, 1]

a time constant of h/L seconds.  L must be small enough to average the 5 s gust away and large enough to
converge inside a 100 s transit: L = 0.2 gives a 5 s time constant and roughly 15 s to settle, which is what
V11 reports.  L is swept over 0.05 / 0.2 / 0.5 so the conclusion does not rest on one gain.

Note what the estimator does NOT need: the vessel model.  It is a kinematic residual, so it is unaffected by
errors in the damping or added-mass parameters -- the honest way to do this on a real boat, where those are
identified and imperfect.

Injection.  The estimate has to reach two places, and V11 reports both separately because they answer
different questions:
  rollouts   solgenia.step(..., current=chat)   -- the predicted positions stop drifting
  barrier    harbour.rows(..., current=chat)    -- the barrier derivative hdot stops being wrong
Only the second one repairs the safety certificate; the first one only repairs the goal-seeking.
"""
import numpy as np
from .. import solgenia as sg


class CurrentObserver:
    """Trapezoidal kinematic-residual estimator of a constant ENU current."""

    def __init__(self, gain=0.2, dt=sg.DT_CTRL, c0=(0.0, 0.0)):
        self.L = float(gain); self.dt = float(dt)
        self.c = np.asarray(c0, float).copy()
        self.prev = None                 # (p, R nu_xy) of the previous control instant
        self.history = []

    @staticmethod
    def _body_vel_enu(x):
        psi = x[2]; cs, sn = np.cos(psi), np.sin(psi)
        return np.array([cs * x[3] - sn * x[4], sn * x[3] + cs * x[4]])

    def update(self, x):
        """Call once per control instant with the measured state, BEFORE planning."""
        x = np.asarray(x, float)
        v = self._body_vel_enu(x)
        if self.prev is not None:
            p0, v0 = self.prev
            y = (x[:2] - p0) / self.dt - 0.5 * (v0 + v)
            self.c = self.c + self.L * (y - self.c)
        self.prev = (x[:2].copy(), v)
        self.history.append(self.c.copy())
        return self.c


class CurrentObserverMixin:
    """Gives a controller an estimate of the current and feeds it to the rollouts and/or the barrier rows.

    mode = "off"       no estimate (the shipped behaviour)
           "rollouts"  the estimate is used to propagate the samples only
           "barrier"   the estimate is used in the barrier rows only
           "both"      both (the sensible controller)
    `oracle=True` replaces the estimator by the true current, which upper-bounds what any observer can buy.
    """

    def __init__(self, *a, obs_mode="off", obs_gain=0.2, oracle=False, true_current=(0.0, 0.0), **kw):
        self.obs_mode = obs_mode
        self.oracle = bool(oracle)
        self.true_current = tuple(float(v) for v in true_current)
        super().__init__(*a, **kw)
        self.observer = CurrentObserver(gain=obs_gain, dt=self.dt)
        self._c_hat = np.zeros(2)

    # what the rollouts believe -----------------------------------------------------------------------
    @property
    def est_current(self):
        if self.obs_mode in ("rollouts", "both"):
            return (float(self._c_hat[0]), float(self._c_hat[1]))
        return (0.0, 0.0)

    @est_current.setter
    def est_current(self, v):                      # the base class declares it as a class attribute
        self._c_hat = np.asarray(v, float)

    # what the barrier believes -----------------------------------------------------------------------
    def _barrier_current(self):
        if self.obs_mode in ("barrier", "both"):
            return (float(self._c_hat[0]), float(self._c_hat[1]))
        return (0.0, 0.0)

    def _rows(self, X, t_abs):
        c = self._barrier_current()
        return self.H.rows(X, t_abs, self.sigma_pos, current=(c if (c[0] or c[1]) else None))

    def _rows_printed(self, X, t_abs):
        c = self._barrier_current()
        return self.H.rows_printed(X, t_abs, self.alpha, current=(c if (c[0] or c[1]) else None))

    def plan(self, x0):
        if self.obs_mode != "off":
            if self.oracle:
                self._c_hat = np.asarray(self.true_current, float)
                self.observer.history.append(self._c_hat.copy())
            else:
                self._c_hat = self.observer.update(x0).copy()
        u = super().plan(x0)
        self.last["c_hat"] = self._c_hat.copy()
        return u
