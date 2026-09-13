"""Closed-loop episodes and metrics."""
import numpy as np
from .dynamics import step, DT

def bridge_crossings(X0, X1, U, dt, sigma, env, n_sub=20, rng=None):
    """Given consecutive states X0 -> X1 (both (3,)), applied control U (2,), process noise sigma,
    sample a Brownian bridge for the noise part with n_sub sub-steps and report whether the continuous
    path left C between the two sample times (endpoints both inside)."""
    if not (env.inside(X0[None])[0] and env.inside(X1[None])[0]):
        return False
    drift = np.array([U[0] * np.cos(X0[2]), U[0] * np.sin(X0[2]), U[1]])
    B_end = X1 - X0 - drift * dt             # = sigma * sqrt(dt) * xi  (the realised noise increment)
    h = dt / n_sub
    B = np.zeros(3); tau = 0.0
    for j in range(1, n_sub):
        rem = dt - tau
        mean = B + (B_end - B) * (h / rem)
        var = sigma ** 2 * h * (rem - h) / rem      # bridge of a Brownian motion with diffusion sigma
        B = mean + np.sqrt(max(var, 0.0)) * rng.standard_normal(3)
        tau += h
        Xs = X0 + drift * tau + B
        if not env.inside(Xs[None])[0]:
            return True
    return False


def run_episode(controller, env, sigma=1.0, seed=0, x0=(0.0, 0.5, 0.0), max_steps=250, goal_radius=0.15,
                record_samples=False, bridge=True, n_sub=20):
    rng = np.random.default_rng(10_000 + seed)
    x = np.array(x0, float)
    traj = [x.copy()]; ctrl = []; ess = []; act = []; vr = []; unsafe_frac = []; sat_all = []; sat_act = []
    infeas = []; samples = []; crossings = 0; steps_inside_pairs = 0
    ttf = max_steps; reached = False
    for k in range(max_steps):
        u = controller.plan(x)
        info = controller.last
        ess.append(info.get("ess", np.nan)); unsafe_frac.append(info.get("frac_unsafe_samples", np.nan))
        act.append(info.get("activation_frac", np.nan)); vr.append(info.get("var_ratio", np.nan))
        sat_all.append(info.get("sat_all", np.nan)); sat_act.append(info.get("sat_active", np.nan))
        infeas.append(info.get("infeasible_frac", np.nan))
        if record_samples:
            samples.append(info["traj"][:, :, :2].copy())
        x_new = step(x[None], u[None], DT, sigma, rng)[0]
        if bridge and sigma > 0:
            if env.inside(x[None])[0] and env.inside(x_new[None])[0]:
                steps_inside_pairs += 1
                crossings += int(bridge_crossings(x, x_new, u, DT, sigma, env, n_sub, rng))
        x = x_new
        traj.append(x.copy()); ctrl.append(u.copy())
        if np.linalg.norm(x[:2] - controller.goal) < goal_radius:
            ttf = k + 1; reached = True
            break
    traj = np.array(traj); ctrl = np.array(ctrl)
    inside = env.inside(traj)
    hmin = env.min_h(traj)
    out = {
        "seed": seed, "steps": int(len(ctrl)), "ttf": int(ttf), "reached": bool(reached),
        "collision_rate": float((~inside).mean()),          # paper's definition: fraction of states outside C
        "collided": bool((~inside).any()),
        "min_h": float(hmin.min()),
        "ess_mean": float(np.nanmean(ess)), "unsafe_sample_frac_mean": float(np.nanmean(unsafe_frac)),
        "activation_frac_mean": float(np.nanmean(act)) if not np.all(np.isnan(act)) else None,
        "var_ratio_mean": float(np.nanmean(vr)) if not np.all(np.isnan(vr)) else None,
        "sat_all_mean": float(np.nanmean(sat_all)) if not np.all(np.isnan(sat_all)) else None,
        "sat_active_mean": float(np.nanmean(sat_act)) if not np.all(np.isnan(sat_act)) else None,
        "infeasible_frac_mean": float(np.nanmean(infeas)) if not np.all(np.isnan(infeas)) else None,
        "bridge_crossings": int(crossings), "inside_pairs": int(steps_inside_pairs),
        "traj": traj, "ctrl": ctrl, "hmin_series": hmin, "ess_series": np.array(ess),
    }
    if record_samples:
        out["samples"] = samples
    return out


def summarize(runs, keys=("collision_rate", "ttf", "min_h", "ess_mean", "activation_frac_mean", "var_ratio_mean",
                          "sat_all_mean", "sat_active_mean", "bridge_crossings", "inside_pairs")):
    """Mean and 95 % bootstrap CI over runs for each key."""
    rng = np.random.default_rng(0)
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs if r.get(k) is not None], float)
        if len(vals) == 0:
            continue
        boots = [rng.choice(vals, len(vals)).mean() for _ in range(2000)]
        out[k] = {"mean": float(vals.mean()), "lo": float(np.percentile(boots, 2.5)),
                  "hi": float(np.percentile(boots, 97.5)), "n": int(len(vals))}
    out["collided_frac"] = float(np.mean([r["collided"] for r in runs]))
    out["reached_frac"] = float(np.mean([r["reached"] for r in runs]))
    return out
