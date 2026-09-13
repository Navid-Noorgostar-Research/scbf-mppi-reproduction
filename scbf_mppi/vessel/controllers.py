"""MPPI, SCBF-MPPI (Algorithm 1 with the second-order barrier; or the relative-degree-1 rows exactly as
printed) and deterministic MPPI for the Solgenia model.  Same weight/update equations as the corridor code
(scbf_mppi/mppi.py); n = 3 inputs u = (X_AT, Y_AT, F_BT) [N], 6 states.

Cost:   q(x) = ||p − p_g||^2 + w_v max(0, |u| − v_max)^2 + penalty · 1[min_j h_j < 0]      [m^2]
        phi(x_T) = ||p_T − p_g||^2
Control interval 1 s (the model is integrated in four RK4 sub-steps of 0.25 s), horizon T = 15 s (the group's
NMPC horizon, Table 3 of the Solgenia paper), K = 500 samples.
Sampling covariance Sigma0 = diag(s0^2) in newtons; rollouts on the nominal (noise-free) model, as in the paper.
All controllers draw the same base normals xi ~ N(0, I) per cycle, so that a mode that never modifies the
proposal (e.g. "printed_rd1") reproduces plain MPPI bit for bit under the same seed.
"""
import numpy as np
from . import solgenia as sg
from . import solver_nd


class VesselMPPI:
    def __init__(self, harbour, K=500, T=15, lam=300.0, s0=(350.0, 350.0, 120.0), dt=sg.DT_CTRL,
                 penalty=20000.0, v_max=2.5, w_v=2000.0, seed=0):
        self.H = harbour; self.K = K; self.T = T; self.lam = lam; self.dt = dt
        self.s0 = np.asarray(s0, float); self.Sigma0 = self.s0 ** 2; self.R = lam / self.Sigma0
        self.penalty = penalty; self.v_max = v_max; self.w_v = w_v
        self.rng = np.random.default_rng(seed)
        self.U = np.zeros((T, 3)); self.last = {}; self.t = 0.0

    # ---- hook points -------------------------------------------------------------------------------
    # `est_current` is the controller's BELIEF about the water current (m/s, ENU).  It is (0, 0) for every
    # controller in the paper's setting and in all shipped experiments, and the two hooks below then reduce
    # to the exact calls they replaced.  vessel/ext/observer.py overrides it with an on-line estimate.
    est_current = (0.0, 0.0)

    def _step(self, X, U, dt=None):
        """Propagate the rollouts.  Identical to sg.step unless the controller believes in a current."""
        d = self.dt if dt is None else dt
        c = self.est_current
        if c[0] or c[1]:
            return sg.step(X, U, d, current=c)
        return sg.step(X, U, d)

    # costs --------------------------------------------------------------------------------------------
    def q(self, X, t_abs):
        """X (K,T,6); t_abs (T,) absolute times of the states (for moving obstacles)."""
        d2 = ((X[..., :2] - self.H.goal) ** 2).sum(-1)
        over = np.maximum(np.abs(X[..., 3]) - self.v_max, 0.0)
        hmin = np.stack([self.H.min_h(X[:, i], t_abs[i]) for i in range(X.shape[1])], axis=1)
        return d2 + self.w_v * over ** 2 + self.penalty * (hmin < 0.0)

    def phi(self, X):
        return ((X[..., :2] - self.H.goal) ** 2).sum(-1)

    def control_terms(self, U, eps):
        return ((U[None] * self.R * eps).sum(-1) + 0.5 * (U[None] ** 2 * self.R).sum(-1)).sum(-1)

    # proposal -------------------------------------------------------------------------------------------
    def sample_eps(self, x0, xi):
        K, T = self.K, self.T
        eps = xi * self.s0
        X = np.broadcast_to(x0, (K, 6)).copy()
        traj = np.empty((K, T + 1, 6)); traj[:, 0] = X
        for t in range(T):
            X = self._step(X, self.U[t][None] + eps[:, t])
            traj[:, t + 1] = X
        return eps, traj, {}

    def weights(self, S, info):
        beta = S.min()
        logw = -(S - beta) / self.lam
        if "logq" in info:
            logw = logw + info["logq"]
        logw -= logw.max()
        w = np.exp(logw); return w / w.sum()

    def plan(self, x0):
        x0 = np.asarray(x0, float)
        xi = self.rng.standard_normal((self.K, self.T, 3))
        eps, traj, info = self.sample_eps(x0, xi)
        t_abs = self.t + self.dt * np.arange(1, self.T + 1)
        # S = J + lam * C:  the control term is linear in lam, which vessel/ext/adaptive.py exploits to
        # retune the temperature within a cycle without re-rolling the samples.
        self._ctrl_terms = self.control_terms(self.U, eps)
        S = self.q(traj[:, 1:], t_abs).sum(-1) + self.phi(traj[:, -1]) + self._ctrl_terms
        w = self.weights(S, info)
        dU = (w[:, None, None] * eps).sum(0)
        self.U = self.U + dU
        self.U = sg.clip_inputs(self.U, np.broadcast_to(x0, (self.T, 6)))
        u0 = self.U[0].copy()
        hmin_s = np.stack([self.H.min_h(traj[:, i + 1], t_abs[i]) for i in range(self.T)], axis=1)
        self.last = {"ess": float(1.0 / (w ** 2).sum()), "w_max": float(w.max()),
                     "dU_norm": float(np.linalg.norm(dU)), "lam_used": float(self.lam),
                     "frac_unsafe_samples": float((hmin_s < 0).any(-1).mean()), "traj": traj, "w": w, "S": S,
                     **{k: v for k, v in info.items() if k != "logq"}}
        self.U = np.vstack([self.U[1:], self.U[-1:]])
        self.t += self.dt
        return u0


class VesselSCBFMPPI(VesselMPPI):
    """Algorithm 1 on the vessel.
    mode = "hocbf":        second-order SCBF rows (a tau >= b), per sample and per timestep
    mode = "printed_rd1":  the relative-degree-1 rows exactly as printed (L_g h = 0 for a force input)
    form = "std" | "variance";  is_correction: importance-sampling weight log N(eps;0,Sigma0) − log q(eps).
    Statistics per cycle: activation (any row active), infeasible, multi (>= 2 rows active), var_ratio (std kept along
    the most violated row, squared), sat_all / sat_active (all rows satisfied by the clipped draw), clip_frac."""
    def __init__(self, harbour, delta=0.003, form="std", mode="hocbf", alpha=None, is_correction=False,
                 sigma_pos=0.0, **kw):
        super().__init__(harbour, **kw)
        self.delta = delta; self.form = form; self.mode = mode
        self.z = solver_nd.quantile(delta); self.alpha = self.z if alpha is None else alpha
        self.is_correction = is_correction; self.sigma_pos = sigma_pos

    def _rows(self, X, t_abs):
        """Second-order rows at the controller's BELIEF about the world (see VesselMPPI.est_current)."""
        c = self.est_current
        return self.H.rows(X, t_abs, self.sigma_pos, current=(c if (c[0] or c[1]) else None))

    def _rows_printed(self, X, t_abs):
        c = self.est_current
        return self.H.rows_printed(X, t_abs, self.alpha, current=(c if (c[0] or c[1]) else None))

    def sample_eps(self, x0, xi):
        K, T = self.K, self.T
        eps = np.empty((K, T, 3))
        X = np.broadcast_to(x0, (K, 6)).copy()
        traj = np.empty((K, T + 1, 6)); traj[:, 0] = X
        n_active = 0; n_infeas = 0; n_multi = 0; var_ratio = []; sat = []; sat_active = []
        logq = np.zeros(K); demand_over = 0
        for t in range(T):
            t_abs = self.t + self.dt * t
            ubar = np.broadcast_to(self.U[t], (K, 3))
            if self.mode == "printed_rd1":
                A, b = self._rows_printed(X, t_abs)
            else:
                A, b, _, _ = self._rows(X, t_abs)
                A = A @ sg.B_ALLOC                          # rows in u-coordinates: a_u = a B
            res = solver_nd.solve_rows_nd(ubar, A, b, self.s0, z=self.z, alpha=self.alpha, form=self.form)
            mu, P = res["mu"], res["Pfac"]
            n_active += int(res["active"].sum()); n_infeas += int(res["infeasible"].sum())
            n_multi += int(res["multi_violation"].sum())
            var_ratio.append(float(np.mean((res["s"] / np.maximum(res["s_nom"], 1e-12)) ** 2)))
            e = mu + np.einsum("kij,kj->ki", P, xi[:, t])
            eps[:, t] = e
            U = ubar + e
            Uc = sg.clip_inputs(U, X)
            demand_over += int((np.abs(Uc - U).max(-1) > 1e-9).sum())
            # delivered satisfaction of EVERY row by the (clipped) drawn input; "active" = any row active
            ok = ((A * Uc[:, None, :]).sum(-1) >= b - 1e-7 * (1.0 + np.abs(b))).all(-1)
            sat.append(ok); sat_active.append(ok[res["active"]])
            if self.is_correction:
                # log N(e; 0, Sigma0) − log N(e; mu, P P^T) with e − mu = P xi:  log q = −0.5|xi|^2 − log|det P|.
                # |det P| from slogdet (valid for the rank-one AND the joint multi-row update); a collapsed
                # direction (singular P, a delta proposal) is floored at det P0 · 1e-6.
                sign, logdet = np.linalg.slogdet(P)
                logdet = np.where(sign == 0, -np.inf, logdet)
                floor = np.log(self.s0).sum() + np.log(1e-6)
                lq = -0.5 * (xi[:, t] ** 2).sum(-1) - np.maximum(logdet, floor)
                lp = -0.5 * ((e / self.s0) ** 2).sum(-1) - np.log(self.s0).sum()
                logq += lp - lq
            X = self._step(X, U)
            traj[:, t + 1] = X
        info = {"activation_frac": n_active / (K * T), "infeasible_frac": n_infeas / (K * T),
                "multi_violation_frac": n_multi / (K * T), "var_ratio": float(np.mean(var_ratio)),
                "sat_all": float(np.concatenate(sat).mean()),
                "sat_active": float(np.concatenate(sat_active).mean()) if sum(len(a) for a in sat_active) else np.nan,
                "clip_frac": demand_over / (K * T)}
        if self.is_correction:
            info["logq"] = logq
        return eps, traj, info


class VesselDetMPPI(VesselMPPI):
    """Homburger, Messerer, Diehl, Reuter (L-CSS 2025) Algorithm 1 as a receding-horizon planner:
    I iterations, beta = nu^j, lambda_j = beta^2 lambda0, Sigma_j = beta^2 Sigma0, correction lambda W^T Sigma^{-1} U."""
    def __init__(self, harbour, iters=4, shrink=0.5, M=125, **kw):
        super().__init__(harbour, K=M, **kw)
        self.iters = iters; self.shrink = shrink; self.M = M; self.lam0 = self.lam; self.s00 = self.s0.copy()

    def plan(self, x0):
        x0 = np.asarray(x0, float); ess_list = []; traj = None
        t_abs = self.t + self.dt * np.arange(1, self.T + 1)
        for j in range(self.iters):
            beta = self.shrink ** j
            lam = beta ** 2 * self.lam0; s = beta * self.s00
            eps = self.rng.standard_normal((self.M, self.T, 3)) * s
            X = np.broadcast_to(x0, (self.M, 6)).copy()
            traj = np.empty((self.M, self.T + 1, 6)); traj[:, 0] = X
            for t in range(self.T):
                X = self._step(X, self.U[t][None] + eps[:, t]); traj[:, t + 1] = X
            S = self.q(traj[:, 1:], t_abs).sum(-1) + self.phi(traj[:, -1])
            S = S + lam * ((eps / s ** 2) * self.U[None]).sum((-1, -2))
            w = np.exp(-(S - S.min()) / lam); w /= w.sum()
            self.U = self.U + (w[:, None, None] * eps).sum(0)
            self.U = sg.clip_inputs(self.U, np.broadcast_to(x0, (self.T, 6)))
            ess_list.append(float(1.0 / (w ** 2).sum()))
        u0 = self.U[0].copy()
        hmin_s = np.stack([self.H.min_h(traj[:, i + 1], t_abs[i]) for i in range(self.T)], axis=1)
        self.last = {"ess": float(np.mean(ess_list)), "ess_iters": ess_list,
                     "frac_unsafe_samples": float((hmin_s < 0).any(-1).mean()), "traj": traj, "w": w, "S": S}
        self.U = np.vstack([self.U[1:], self.U[-1:]])
        self.t += self.dt
        return u0
