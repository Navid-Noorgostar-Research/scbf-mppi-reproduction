"""A heading term for the vessel running cost (experiment V9).

Why this exists.  The reproduction's vessel cost is

    q(x) = ||p - p_goal||^2  +  w_v max(0, |u| - v_max)^2  +  penalty * 1[min_j h_j < 0]        [m^2]

The speed term uses |u|, so running ASTERN costs exactly what running ahead costs, and the azimuth thruster
is an isotropic 700 N force disk with damping that is symmetric in surge.  Nothing in the problem therefore
prefers bow-first motion, and all three controllers spend large parts of a run going backwards.  That is a
property of the force-input abstraction, not of the barrier, and it affects MPPI and the barrier variants
alike -- but a reviewer is entitled to ask whether the comparison survives a cost that does prefer bow-first
motion.  This module supplies that cost so the question can be answered with numbers.

Two terms, independently switchable:

  heading-to-goal   w_psi * (1 - cos(psi - atan2(p_goal - p)))        in [0, 2 w_psi]
      Zero when the bow points at the goal, maximal when it points away.  The 1 - cos form needs no angle
      unwrapping (psi is already wrapped to (-pi, pi] by solgenia.step) and is smooth through +-pi, unlike a
      squared angle error.  It is gated off inside the goal radius, where the bearing is ill-conditioned and
      the vessel is finished anyway.

  astern            w_astern * max(0, -u_surge)^2                     [m^2/s^2 scaled]
      Penalises negative surge directly, with the same weight and the same one-sided quadratic shape as the
      existing speed-limit term.  This isolates the astern question from the turning question: a vessel can
      satisfy the heading term by turning, but the astern term can only be satisfied by going forwards.

Scale.  The position cost is ||p - p_goal||^2 in m^2 and runs from 120^2 = 14400 at the start to 0 at the
goal, i.e. of order 3000-4000 during the transit through the obstacle field.  w_psi = 400 m^2 makes a fully
reversed heading cost 800 m^2, about a fifth of the mid-transit position cost: enough to turn the vessel,
too little to override obstacle avoidance (penalty = 20000).  V9 sweeps w_psi over 100 / 400 / 1600 so the
conclusion cannot rest on one weight.

Yaw authority (so the term is achievable, not merely expensive): the azimuth thruster gives at most
700 N x 2.9 m = 2030 N m and the bow thruster 244 N x 3.7 m = 904 N m, against a yaw inertia of
21179 kg m^2 -- about 0.14 rad/s^2, so a 180 deg turn is a matter of a few seconds of dedicated effort.
The vessel can obey the term; whether obeying it is worth the detour is what the experiment measures.
"""
import numpy as np


class HeadingCostMixin:
    """Adds the two optional terms to `q`.  With w_psi = w_astern = 0 the cost is untouched."""

    def __init__(self, *a, w_psi=0.0, w_astern=0.0, **kw):
        self.w_psi = float(w_psi)
        self.w_astern = float(w_astern)
        super().__init__(*a, **kw)

    def q(self, X, t_abs):
        base = super().q(X, t_abs)
        if self.w_psi <= 0.0 and self.w_astern <= 0.0:
            return base
        extra = np.zeros_like(base)
        if self.w_psi > 0.0:
            d = self.H.goal - X[..., :2]
            dist = np.linalg.norm(d, axis=-1)
            bearing = np.arctan2(d[..., 1], d[..., 0])
            # gate: the bearing is meaningless once the vessel is inside the goal circle
            gate = (dist > self.H.goal_radius).astype(float)
            extra = extra + self.w_psi * (1.0 - np.cos(X[..., 2] - bearing)) * gate
        if self.w_astern > 0.0:
            extra = extra + self.w_astern * np.maximum(0.0, -X[..., 3]) ** 2
        return base + extra
