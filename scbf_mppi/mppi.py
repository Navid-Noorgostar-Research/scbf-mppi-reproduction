"""MPPI (Williams et al. 2017 form, as in Section III-B of the paper), SCBF-MPPI (Algorithm 1),
its mean-shift-only / variance-only variants, and deterministic MPPI with covariance annealing
(Homburger, Messerer, Diehl, Reuter, L-CSS 2025, Algorithm 1).

Cost (paper, Section V-B):  q(x) = ||X - Xg||^2 + 1000 * 1[X not in C];  terminal phi(x_T) = ||X_T - Xg||^2  (assumption).
Running cost with control terms (paper, Section III-B, nu = 1):  q~ = q + v^T R eps + 0.5 v^T R v,  R = lambda Sigma0^{-1}.
Weights  w_i = exp(-(S~_i - min_j S~_j)/lambda);  update  u_t <- u_t + sum_i w_i eps_{i,t} / sum_i w_i.
"""
import numpy as np
from .dynamics import step, DT
from . import scbf as scbf_mod


class MPPI:
    def __init__(self, env, K=500, T=20, lam=1.0, sigma_v=0.5, sigma_om=1.5, dt=DT,
                 goal=(4.0, 0.5), penalty=1000.0, seed=0, u_max=None):
        self.env = env; self.K = K; self.T = T; self.lam = lam; self.dt = dt
        self.Sigma0 = np.array([sigma_v, sigma_om]) ** 2     # diagonal
        self.R = lam / self.Sigma0                              # R = lambda Sigma0^{-1} (diagonal)
        self.goal = np.asarray(goal, float); self.penalty = penalty
        self.rng = np.random.default_rng(seed)
        self.u_max = u_max
        self.U = np.zeros((T, 2))
        self.last = {}

    # costs ----------------------------------------------------------------------------
    def q(self, X):
        d2 = ((X[..., :2] - self.goal) ** 2).sum(-1)
        return d2 + self.penalty * (~self.env.inside(X))

    def phi(self, X):
        return ((X[..., :2] - self.goal) ** 2).sum(-1)

    def control_terms(self, U, eps):
        """sum_t v^T R eps + 0.5 v^T R v  (nu = 1)."""
        return ((U[None] * self.R * eps).sum(-1) + 0.5 * (U[None] ** 2 * self.R).sum(-1)).sum(-1)

    # proposal -------------------------------------------------------------------------
    def sample_eps(self, x0):
        eps = self.rng.standard_normal((self.K, self.T, 2)) * np.sqrt(self.Sigma0)
        X = np.broadcast_to(x0, (self.K, 3)).copy()
        traj = np.empty((self.K, self.T + 1, 3)); traj[:, 0] = X
        for t in range(self.T):
            U = self.U[t][None] + eps[:, t]
            if self.u_max is not None:
                U = np.clip(U, -self.u_max, self.u_max)
            X = step(X, U, self.dt)
            traj[:, t + 1] = X
        return eps, traj, {}

    # one planning cycle ---------------------------------------------------------------
    def plan(self, x0):
        eps, traj, info = self.sample_eps(np.asarray(x0, float))
        S = self.q(traj[:, 1:]).sum(-1) + self.phi(traj[:, -1]) + self.control_terms(self.U, eps)
        beta = S.min()
        w = np.exp(-(S - beta) / self.lam)
        w = w / w.sum()
        self.U = self.U + (w[:, None, None] * eps).sum(0)
        if self.u_max is not None:
            self.U = np.clip(self.U, -self.u_max, self.u_max)
        u0 = self.U[0].copy()
        ess = 1.0 / (w ** 2).sum()
        frac_unsafe = float((~self.env.inside(traj[:, 1:])).any(-1).mean())
        self.last = {"ess": float(ess), "w_max": float(w.max()), "frac_unsafe_samples": frac_unsafe,
                     "traj": traj, "w": w, "S": S, **info}
        # receding horizon shift
        self.U = np.vstack([self.U[1:], np.zeros((1, 2))])
        return u0


class SCBFMPPI(MPPI):
    """Algorithm 1: per sample and per timestep, re-choose the proposal (mean shift m, std s of the
    v-perturbation) so that the SCBF chance constraint holds, then draw the perturbation."""
    def __init__(self, env, sigma_env=1.0, delta=0.003, form="std", mode="both", alpha=None,
                 is_correction=False, **kw):
        super().__init__(env, **kw)
        self.sigma_env = sigma_env; self.delta = delta; self.form = form; self.mode = mode
        self.z = scbf_mod.quantile(delta)
        self.alpha = self.z if alpha is None else alpha       # 'alpha' of eq. (6): we take it = z unless given
        self.is_correction = is_correction

    def sample_eps(self, x0):
        K, T = self.K, self.T
        s0 = np.sqrt(self.Sigma0[0]); s_om = np.sqrt(self.Sigma0[1])
        eps = np.empty((K, T, 2))
        X = np.broadcast_to(x0, (K, 3)).copy()
        traj = np.empty((K, T + 1, 3)); traj[:, 0] = X
        xi = self.rng.standard_normal((K, T, 2))
        n_active = 0; n_infeas = 0; var_ratio = []; logq = np.zeros(K); sat = []; sat_active = []
        for t in range(T):
            c, b = self.env.scbf_rows(X, self.sigma_env)
            res = scbf_mod.solve_rows(np.full(K, self.U[t, 0]), c, b, s0,
                                      z=self.z, alpha=self.alpha, form=self.form)
            m, s = res["m"], res["s"]
            if self.mode == "mean_only":            # keep Sigma0, move only the mean
                s = np.full(K, s0)
                m = self._mean_only(c, b, self.U[t, 0], s0)
            elif self.mode == "var_only":
                m = np.zeros(K)
            n_active += int(res["active"].sum()); n_infeas += int(res["infeasible"].sum())
            var_ratio.append(float((s ** 2).mean() / s0 ** 2))
            ev = m + s * xi[:, t, 0]
            eo = s_om * xi[:, t, 1]
            eps[:, t, 0] = ev; eps[:, t, 1] = eo
            # empirical satisfaction of the row constraints by the drawn control
            u_v = self.U[t, 0] + ev
            ok = (c * u_v[:, None] >= b - 1e-7 * (1.0 + np.abs(b))).all(-1)   # tolerance: s = 0 puts u exactly on the boundary
            sat.append(ok); sat_active.append(ok[res["active"]])
            if self.is_correction:
                # log density ratio  log N(eps; 0, Sigma0) - log N(eps; m, s^2)  (v channel only)
                logq += (-0.5 * (ev / s0) ** 2 - np.log(s0)) - (-0.5 * ((ev - m) / np.maximum(s, 1e-9)) ** 2 - np.log(np.maximum(s, 1e-9)))
            U = self.U[t][None] + eps[:, t]
            if self.u_max is not None:
                U = np.clip(U, -self.u_max, self.u_max)
            X = step(X, U, self.dt)
            traj[:, t + 1] = X
        info = {"activation_frac": n_active / (K * T), "infeasible_frac": n_infeas / (K * T),
                "var_ratio": float(np.mean(var_ratio)),
                "sat_all": float(np.concatenate(sat).mean()),
                "sat_active": float(np.concatenate(sat_active).mean()) if sum(len(a) for a in sat_active) else np.nan,
                "n_active_pairs": n_active}
        if self.is_correction:
            info["logq"] = logq
        return eps, traj, info

    def _mean_only(self, c, b, ubar_v, s0):
        pen = (self.z * np.abs(c) * s0) if self.form == "std" else (self.alpha * c ** 2 * s0 ** 2)
        r = b + pen - c * ubar_v
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = r / c
        lo = np.where(c > 1e-12, ratio, -np.inf).max(1); hi = np.where(c < -1e-12, ratio, np.inf).min(1)
        m = np.clip(0.0, lo, hi)
        m = np.where(lo <= hi, m, 0.5 * (np.where(np.isfinite(lo), lo, hi) + np.where(np.isfinite(hi), hi, lo)))
        return np.where(np.isfinite(m), m, 0.0)

    def plan(self, x0):
        eps, traj, info = self.sample_eps(np.asarray(x0, float))
        S = self.q(traj[:, 1:]).sum(-1) + self.phi(traj[:, -1]) + self.control_terms(self.U, eps)
        beta = S.min()
        logw = -(S - beta) / self.lam
        if self.is_correction:
            logw = logw + info.pop("logq")
        logw -= logw.max()
        w = np.exp(logw); w /= w.sum()
        self.U = self.U + (w[:, None, None] * eps).sum(0)
        if self.u_max is not None:
            self.U = np.clip(self.U, -self.u_max, self.u_max)
        u0 = self.U[0].copy()
        frac_unsafe = float((~self.env.inside(traj[:, 1:])).any(-1).mean())
        self.last = {"ess": float(1.0 / (w ** 2).sum()), "w_max": float(w.max()),
                     "frac_unsafe_samples": frac_unsafe, "traj": traj, "w": w, "S": S, **info}
        self.U = np.vstack([self.U[1:], np.zeros((1, 2))])
        return u0


class DeterministicMPPI(MPPI):
    """Homburger et al., L-CSS 2025, Algorithm 1: I iterations with beta = nu^j, lambda_j = beta^2 lambda0,
    Sigma_j = beta^2 Sigma0, importance-sampling correction lambda W^T Sigma^{-1} U for the shifted proposal.
    Used here as a receding-horizon planner on the nominal model; total rollouts per cycle = I * M."""
    def __init__(self, env, iters=4, shrink=0.5, M=125, **kw):
        super().__init__(env, K=M, **kw)
        self.iters = iters; self.shrink = shrink; self.M = M
        self.lam0 = self.lam; self.Sigma00 = self.Sigma0.copy()

    def plan(self, x0):
        x0 = np.asarray(x0, float)
        ess_list = []; frac_unsafe = 0.0; traj = None
        for j in range(self.iters):
            beta = self.shrink ** j
            lam = beta ** 2 * self.lam0; Sig = beta ** 2 * self.Sigma00
            eps = self.rng.standard_normal((self.M, self.T, 2)) * np.sqrt(Sig)
            X = np.broadcast_to(x0, (self.M, 3)).copy()
            traj = np.empty((self.M, self.T + 1, 3)); traj[:, 0] = X
            for t in range(self.T):
                U = self.U[t][None] + eps[:, t]
                if self.u_max is not None:
                    U = np.clip(U, -self.u_max, self.u_max)
                X = step(X, U, self.dt); traj[:, t + 1] = X
            S = self.q(traj[:, 1:]).sum(-1) + self.phi(traj[:, -1])
            S = S + lam * ((eps / Sig) * self.U[None]).sum((-1, -2))     # correction lambda W^T Sigma^{-1} U
            psi = S.min()
            w = np.exp(-(S - psi) / lam); w /= w.sum()
            self.U = self.U + (w[:, None, None] * eps).sum(0)
            if self.u_max is not None:
                self.U = np.clip(self.U, -self.u_max, self.u_max)
            ess_list.append(float(1.0 / (w ** 2).sum()))
            frac_unsafe = float((~self.env.inside(traj[:, 1:])).any(-1).mean())
        u0 = self.U[0].copy()
        self.last = {"ess": float(np.mean(ess_list)), "ess_iters": ess_list, "frac_unsafe_samples": frac_unsafe,
                     "traj": traj, "w": w, "S": S}
        self.U = np.vstack([self.U[1:], np.zeros((1, 2))])
        return u0
