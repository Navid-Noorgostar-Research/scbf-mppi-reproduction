"""Closed-loop episodes on the Solgenia model with three disturbance models of equal low-frequency intensity.

disturbance = "white":  force noise  tau_d dt = sigma_F dW,   sigma_F = s_F sqrt(2 tau_c)          (the paper's assumption)
disturbance = "ou":     Ornstein–Uhlenbeck force  d tau_d = −tau_d/tau_c dt + s_F sqrt(2/tau_c) dW   (a gust that persists)
                        — same spectral density at zero frequency as the white case, so the same long-horizon
                        displacement variance, but correlated over tau_c seconds.
disturbance = "none":   deterministic plant.
current: constant water current (m/s, ENU) added to the position kinematics (the controller does not know it).
"""
import numpy as np
from . import solgenia as sg
from .harbour import X0 as X0_DEFAULT

S_F_DEFAULT = np.array([120.0, 300.0, 400.0])    # N, N, N m — stationary std of the environmental force (experiments.DEFAULT)
TAU_C_DEFAULT = 5.0                              # s — gust correlation time


def run_episode(controller, harbour, seed=0, disturbance="white", s_F=S_F_DEFAULT, tau_c=TAU_C_DEFAULT,
                current=(0.0, 0.0), max_time=150.0, x0=None, record_samples=False, sample_stride=1,
                extra_keys=(), step_hook=None):
    """`extra_keys` additionally averages those keys of `controller.last` over the episode and keeps their
    series; `step_hook(k, t, x, u, controller)` is called once per control instant with the measured state
    and the applied input.  Both default to inert, so every shipped experiment is unaffected."""
    rng = np.random.default_rng(10_000 + seed)
    dt = controller.dt
    x = np.array(X0_DEFAULT if x0 is None else x0, float)
    s_F = np.asarray(s_F, float)
    tau_d = s_F * rng.standard_normal(3) if disturbance == "ou" else np.zeros(3)   # OU started in its stationary law
    max_steps = int(round(max_time / dt))
    traj = [x.copy()]; ctrl = []; ess = []; act = []; vr = []; sat_all = []; sat_act = []; infeas = []
    multi = []; clipf = []; dist = []; hmin_series = []; hmin_sub = []; samples = []; unsafe = []
    extra = {k: [] for k in extra_keys}
    ttf = max_time; reached = False; controller.t = 0.0
    for k in range(max_steps):
        u = controller.plan(x)
        info = controller.last
        ess.append(info.get("ess", np.nan)); act.append(info.get("activation_frac", np.nan))
        vr.append(info.get("var_ratio", np.nan)); sat_all.append(info.get("sat_all", np.nan))
        sat_act.append(info.get("sat_active", np.nan)); infeas.append(info.get("infeasible_frac", np.nan))
        multi.append(info.get("multi_violation_frac", np.nan)); clipf.append(info.get("clip_frac", np.nan))
        unsafe.append(info.get("frac_unsafe_samples", np.nan))
        for _k in extra_keys:
            _v = info.get(_k, np.nan)
            extra[_k].append(float(_v) if _v is not None and np.isscalar(_v) else np.nan)
        if record_samples and k % sample_stride == 0:
            samples.append((k, info["traj"][:, :, :2].astype(np.float32).copy(), info["w"].astype(np.float32).copy()))
        t_now = k * dt
        if step_hook is not None:
            step_hook(k, t_now, x, np.asarray(u, float), controller)
        # disturbance for this step
        n_sub = max(1, int(round(dt / sg.DT))); h = dt / n_sub
        if disturbance == "white":
            x_new, subs = sg.step(x[None], u[None], dt, current=current, sigma_F=s_F * np.sqrt(2.0 * tau_c), rng=rng, return_sub=True)
            x_new = x_new[0]; subs = [z[0] for z in subs]
        elif disturbance == "ou":
            # exact OU update at every integration sub-step (h = 0.25 s), force held over the sub-step
            x_new = x.copy(); subs = []
            for _ in range(n_sub):
                x_new = sg.step(x_new[None], u[None], h, tau_d=tau_d[None], current=current, n_sub=1)[0]
                subs.append(x_new.copy())
                tau_d = tau_d * np.exp(-h / tau_c) + s_F * np.sqrt(1.0 - np.exp(-2.0 * h / tau_c)) * rng.standard_normal(3)
        else:
            x_new, subs = sg.step(x[None], u[None], dt, current=current, return_sub=True)
            x_new = x_new[0]; subs = [z[0] for z in subs]
        dist.append(tau_d.copy())
        x = x_new
        traj.append(x.copy()); ctrl.append(np.asarray(u, float).copy())
        hmin_series.append(float(harbour.min_h(x[None], t_now + dt)[0]))
        # barrier value at every integration sub-step (0.25 s): collisions between control samples count too
        for j, xs in enumerate(subs):
            hmin_sub.append(float(harbour.min_h(xs[None], t_now + (j + 1) * h)[0]))
        if np.linalg.norm(x[:2] - harbour.goal) < harbour.goal_radius:
            ttf = (k + 1) * dt; reached = True
            break
    traj = np.array(traj); ctrl = np.array(ctrl); hmin_series = np.array(hmin_series); hmin_sub = np.array(hmin_sub)
    times = dt * np.arange(1, len(hmin_series) + 1)
    ess_a = np.asarray(ess, float); x_obs = traj[1:len(ess_a) + 1, 0]
    obst = x_obs < 100.0                                          # the obstacle field (x < 100 m); beyond it only the goal
    out = {
        "seed": seed, "steps": int(len(ctrl)), "ttf": float(ttf), "reached": bool(reached),
        "collision_rate": float((hmin_sub < 0).mean()),           # fraction of 0.25 s integration steps inside a circle
        "collision_rate_ctrl": float((hmin_series < 0).mean()),   # the same at the 1 s control instants only
        "collided": bool((hmin_sub < 0).any()),
        "min_h": float(hmin_sub.min()),
        "ess_mean": float(np.nanmean(ess)), "ess_median": float(np.nanmedian(ess)),
        "ess_obstacle_mean": float(np.nanmean(ess_a[obst])) if obst.any() else None,
        "unsafe_sample_frac_mean": float(np.nanmean(unsafe)),
        "activation_frac_mean": _nm(act), "var_ratio_mean": _nm(vr), "sat_all_mean": _nm(sat_all),
        "sat_active_mean": _nm(sat_act), "infeasible_frac_mean": _nm(infeas), "multi_violation_frac_mean": _nm(multi),
        "clip_frac_mean": _nm(clipf),
        "effort": float((ctrl ** 2).sum() * dt),                   # sum |u|^2 dt  [N^2 s]
        "path_length": float(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum()),
        "mean_speed": float(np.abs(traj[1:, 3]).mean()),
        # astern statistics: the vessel cost has no heading term and the surge penalty uses |u|, so nothing
        # in the problem prefers bow-first motion (see vessel/ext/costs.py and experiment V9)
        "astern_frac": float((traj[1:, 3] < 0.0).mean()),
        "astern_frac_strong": float((traj[1:, 3] < -0.3).mean()),
        "min_surge": float(traj[:, 3].min()),
        "traj": traj, "ctrl": ctrl, "hmin_series": hmin_series, "times": times, "dist": np.array(dist),
        "ess_series": np.array(ess),
    }
    for _k in extra_keys:
        out[_k + "_mean"] = _nm(extra[_k])
    if extra_keys:
        out["_extra_series"] = {k: np.asarray(v, float) for k, v in extra.items()}
    if record_samples:
        out["samples"] = samples
    return out


def _nm(v):
    v = np.asarray(v, float)
    return float(np.nanmean(v)) if not np.all(np.isnan(v)) else None


def summarize(runs, keys=("collision_rate", "collision_rate_ctrl", "ttf", "min_h", "ess_mean", "ess_median", "ess_obstacle_mean", "activation_frac_mean", "var_ratio_mean",
                          "sat_active_mean", "infeasible_frac_mean", "clip_frac_mean", "effort", "path_length",
                          "mean_speed", "unsafe_sample_frac_mean", "multi_violation_frac_mean"), n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = {"n": len(runs)}
    for k in keys:
        v = np.array([r[k] for r in runs if r.get(k) is not None], float)
        if len(v) == 0:
            continue
        boots = np.array([rng.choice(v, len(v)).mean() for _ in range(n_boot)])
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        out[k] = {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "lo": lo, "hi": hi, "ci95": [lo, hi], "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
    out["collided_frac"] = float(np.mean([r["collided"] for r in runs]))
    out["reached_frac"] = float(np.mean([r["reached"] for r in runs]))
    return out
