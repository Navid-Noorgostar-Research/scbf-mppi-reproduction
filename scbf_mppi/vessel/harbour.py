"""Obstacles and barrier functions for the vessel scenario.

Every obstacle is a circle of inflated radius R_j = r_obstacle + r_ego (the group's own collision-avoidance
scheme, Section 9 of the Solgenia paper: circles around the ego vessel and every track, constant-velocity
prediction).  Static obstacles have zero velocity; the crossing ferry moves at constant velocity.

Barrier for obstacle j:      h_j(x, t) = ||p − c_j(t)|| − R_j                    (relative degree 2 in the force input)
Second-order (exponential / HOCBF) stochastic CBF condition used by the controllers:
    psi1 = dh/dt + alpha1 h,          dpsi1/dt + alpha2 psi1 >= 0
which is affine in the generalised force tau:   a tau >= b   (derived in `rows`, checked against finite
differences in vessel/selftest.py).  The Itô term (1/2) tr(sigma sigma^T Hess psi1) is identically zero when the
process noise enters the velocity states, because psi1 is linear in nu — `ito_term` computes it in general
(position noise included) and the self-test confirms the zero.

Relative-degree-1 condition exactly as printed in the paper (eq. (3)/(6)): L_g h u >= −L_f h − alpha h − Itô.
For a force input L_g h = (dh/dnu) M^{-1} B = 0, so the row is  0 >= b  — `rows_printed` returns it, and the
controller in mode "printed_rd1" applies Algorithm 1 to it literally.
"""
import numpy as np
from . import solgenia as sg

S2 = np.array([[0.0, -1.0], [1.0, 0.0]])


class Obstacle:
    def __init__(self, center, radius, velocity=(0.0, 0.0), name=""):
        self.c0 = np.asarray(center, float); self.R = float(radius)
        self.w = np.asarray(velocity, float); self.name = name

    def center(self, t):
        return self.c0 + self.w * t


class Harbour:
    """Obstacle field + goal.  alpha1, alpha2: class-K gains of the second-order barrier."""
    def __init__(self, obstacles, goal, alpha1=0.1, alpha2=0.4, goal_radius=5.0, r_ego=5.0):
        self.obs = list(obstacles); self.goal = np.asarray(goal, float)
        self.alpha1 = alpha1; self.alpha2 = alpha2; self.goal_radius = goal_radius; self.r_ego = r_ego

    # geometry -------------------------------------------------------------------------------------
    def h_all(self, X, t):
        """h_j for every obstacle: X (...,6) -> (..., J)."""
        p = X[..., :2]
        out = []
        for o in self.obs:
            d = np.linalg.norm(p - o.center(t), axis=-1)
            out.append(d - o.R)
        return np.stack(out, axis=-1)

    def min_h(self, X, t):
        return self.h_all(X, t).min(-1)

    def inside(self, X, t):
        return self.min_h(X, t) >= 0.0

    # second-order SCBF rows --------------------------------------------------------------------------
    def rows(self, X, t, sigma_pos=0.0, current=None):
        """Rows (a, b) of  a tau >= b  for every obstacle at state X (K,6) and time t.
        a: (K, J, 3) coefficients of tau = (X, Y, N);  b: (K, J);  also returns h, psi1 for statistics.
        `current` is a KNOWN water current (m/s, ENU) entering the position kinematics, so that
        pdot = R(psi) nu_xy + c.  It is None (and the arithmetic untouched) unless a controller has an
        estimate of it — see vessel/ext/observer.py.  A controller that does not know the current has a
        systematically wrong hdot, which is exactly what experiment V11 measures."""
        K = X.shape[0]; J = len(self.obs)
        p = X[:, :2]; psi = X[:, 2]; nu = X[:, 3:6]
        c, s = np.cos(psi), np.sin(psi)
        R2 = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)          # (K,2,2)
        nu_xy = nu[:, :2]; r = nu[:, 2]
        pdot = np.einsum("kij,kj->ki", R2, nu_xy)                                  # (K,2)
        if current is not None and (current[0] or current[1]):
            pdot = pdot + np.asarray(current, float)
        fnu = sg.f_nu(nu)                                                          # (K,3)  = −M^{-1}(C nu + N)
        Minv_xy = sg.M_INV[:2, :]                                                  # (2,3)
        a = np.empty((K, J, 3)); b = np.empty((K, J)); H = np.empty((K, J)); PSI1 = np.empty((K, J))
        for j, o in enumerate(self.obs):
            rel = p - o.center(t)                                                  # (K,2)
            d = np.maximum(np.linalg.norm(rel, axis=-1), 1e-6)
            n = rel / d[:, None]
            h = d - o.R
            v_rel = pdot - o.w[None, :]
            hdot = (n * v_rel).sum(-1)
            psi1 = hdot + self.alpha1 * h
            # d/dt(n) = (I − n n^T) v_rel / d  ->  (dn/dt)^T v_rel = (|v_rel|^2 − (n^T v_rel)^2)/d
            term_n = ((v_rel ** 2).sum(-1) - hdot ** 2) / d
            # n^T d/dt(R2 nu_xy) = n^T R2 S nu_xy r + n^T R2 (dnu_xy/dt)
            nR2 = np.einsum("ki,kij->kj", n, R2)                                   # (K,2) = n^T R2
            term_rot = (nR2 * (nu_xy @ S2.T)).sum(-1) * r
            term_f = (nR2 * fnu[:, :2]).sum(-1)
            a[:, j, :] = nR2 @ Minv_xy                                             # n^T R2 [M^{-1}]_{xy,:}
            drift = term_n + term_rot + term_f + self.alpha1 * hdot + self.alpha2 * psi1
            b[:, j] = -drift
            H[:, j] = h; PSI1[:, j] = psi1
        if sigma_pos:
            b -= self.ito_term(X, t, sigma_pos)
        return a, b, H, PSI1

    def ito_term(self, X, t, sigma_pos, eps=1e-3):
        """(1/2) sigma_pos^2 tr(Hess_p psi1) by central differences in the position — zero for velocity noise."""
        K = X.shape[0]; J = len(self.obs); out = np.zeros((K, J))
        for i in range(2):
            Xp = X.copy(); Xm = X.copy(); Xp[:, i] += eps; Xm[:, i] -= eps
            out += (self._psi1(Xp, t) - 2 * self._psi1(X, t) + self._psi1(Xm, t)) / eps ** 2
        return 0.5 * sigma_pos ** 2 * out

    def _psi1(self, X, t):
        return self.rows(X, t)[3]

    def rows_printed(self, X, t, alpha, current=None):
        """The relative-degree-1 condition as printed: L_g h u >= −L_f h − alpha h.  L_g h = 0 for a force input."""
        K = X.shape[0]; J = len(self.obs)
        p = X[:, :2]; psi = X[:, 2]; nu = X[:, 3:6]
        c, s = np.cos(psi), np.sin(psi)
        pdot = np.stack([c * nu[:, 0] - s * nu[:, 1], s * nu[:, 0] + c * nu[:, 1]], -1)
        if current is not None and (current[0] or current[1]):
            pdot = pdot + np.asarray(current, float)
        a = np.zeros((K, J, 3)); b = np.empty((K, J))
        for j, o in enumerate(self.obs):
            rel = p - o.center(t); d = np.maximum(np.linalg.norm(rel, axis=-1), 1e-6); n = rel / d[:, None]
            h = d - o.R; hdot = (n * (pdot - o.w[None, :])).sum(-1)
            b[:, j] = -(hdot + alpha * h)
        return a, b

    def walls_for_plot(self):
        return [(o.center(0.0), o.R, o.w, o.name) for o in self.obs]


# ---- the scenarios -------------------------------------------------------------------------------------
def harbour_static(alpha1=0.1, alpha2=0.4):
    """Transit through three moored obstacles to a goal 120 m east.  Radii include r_ego = 5 m (hull 8.5 m)."""
    obs = [Obstacle((40.0, 8.0), 12.0, name="moored boat"),
           Obstacle((72.0, -12.0), 14.0, name="barge"),
           Obstacle((92.0, 14.0), 11.0, name="buoy field")]
    return Harbour(obs, goal=(120.0, 0.0), alpha1=alpha1, alpha2=alpha2)


def harbour_crossing(alpha1=0.1, alpha2=0.4, ferry_speed=3.0, ferry_R=25.0, t_cross=22.0):
    """Static field plus the group's Experiment-III ferry: R = 25 m, 3 m/s, crossing the track from the south.
    It reaches the track (y = 0) at x = 66 m at t = t_cross."""
    H = harbour_static(alpha1, alpha2)
    y0 = -ferry_speed * t_cross
    H.obs.append(Obstacle((66.0, y0), ferry_R, velocity=(0.0, ferry_speed), name="ferry (3 m/s)"))
    return H


X0 = np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0])      # start: origin, heading east, 1.5 m/s
