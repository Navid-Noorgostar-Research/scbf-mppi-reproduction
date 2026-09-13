"""Export the runs behind the 3-D view (live_demo/3d/viewer_3d.html).
    python -m scbf_mppi.export3d          -> live_demo/3d/scene_data.json   (~8 min: the runs are re-simulated with
                                                                              the rollout fan recorded)
Three scenes, the same controllers, seeds and code path as the experiments and the 2-D animations:
    current  — V3 seed 0: MPPI / SCBF-MPPI (2nd-order barrier) / + IS under the coloured gust and the 0.5 m/s current
    ferry    — V4 seed 2: the same three through the crossing ferry (white force noise)
    corridor — E1 seed 6: MPPI / SCBF-MPPI as printed / as claimed (corrected) in the paper's corridor
Per boat: the 1-s control states (x, y, psi, u, v, r), the executed control, the disturbance force, h and ESS per step,
and the 40 highest-weight rollouts of every cycle with their normalised weights.  The 30-seed statistics for the
titles and footers come from the stored result files.  Nothing here changes any experiment.
"""
import os, sys, json, time
import numpy as np
from .env import Corridor
from .mppi import MPPI, SCBFMPPI
from .simulate import run_episode as run_corridor
from .experiments import DEFAULT as CDEF, RES
from .vessel import solgenia as sg, harbour as hb
from .vessel.experiments import make_controller, DEFAULT as VDEF
from .vessel.simulate import run_episode as run_vessel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "live_demo", "3d", "scene_data.json")
COL = {"mppi": "#1B8AA6", "scbf": "#5B4FB0", "is": "#2E7D4F", "var": "#C2482A", "std": "#5B4FB0", "ink": "#20303C"}
N_FAN_VESSEL, N_FAN_CORRIDOR = 40, 30
KEYS = ["MPPI", "SCBF-MPPI", "SCBF-MPPI + IS"]


def _r(a, nd):
    return np.round(np.asarray(a, float), nd).tolist()


def _fan_vessel(samples, n):
    """the n highest-weight rollouts of every cycle, weights normalised to the largest"""
    out = []
    for k, S, w in samples:
        idx = np.argsort(w)[::-1][:n]
        out.append({"k": int(k), "S": _r(S[idx], 2), "w": _r(w[idx] / max(float(w[idx].max()), 1e-12), 3)})
    return out


def _fan_corridor(samples, n):
    out = []
    for i, S in enumerate(samples):
        idx = np.linspace(0, S.shape[0] - 1, n).astype(int)
        out.append({"k": int(i), "S": _r(S[idx], 3)})
    return out


def _vessel_boat(key, label, color, cfg, seed, disturbance, current):
    c, H = make_controller(dict(cfg, disturbance=disturbance, current=current), seed)
    t = time.time()
    r = run_vessel(c, H, seed=seed, disturbance=disturbance, s_F=np.asarray(VDEF["s_F"], float), tau_c=VDEF["tau_c"],
                   current=tuple(current), max_time=VDEF["max_time"], record_samples=True)
    print(f"  {label:<32} steps {r['steps']:3d} reached {r['reached']!s:5} ttf {r['ttf']:.0f} min_h {r['min_h']:+.2f} "
          f"touched {r['collided']!s:5} ({time.time()-t:.0f} s)", file=sys.stderr)
    return dict(key=key, label=label, color=color, traj=_r(r["traj"], 3), ctrl=_r(r["ctrl"], 1), dist=_r(r["dist"], 1),
                h=_r(r["hmin_series"], 2), ess=_r(r["ess_series"], 1), reached=bool(r["reached"]), ttf=float(r["ttf"]),
                collided=bool(r["collided"]), min_h=float(r["min_h"]), collision_rate=float(r["collision_rate"]),
                activation=r.get("activation_frac_mean"), var_kept=r.get("var_ratio_mean"),
                fan=_fan_vessel(r["samples"], N_FAN_VESSEL)), H


def _obstacles(H):
    kinds = {"barge": "barge", "moored boat": "moored", "buoy field": "buoys"}
    out = []
    for o in H.obs:
        kind = "ferry" if o.name.startswith("ferry") else kinds.get(o.name, "generic")
        out.append(dict(name=o.name, kind=kind, c0=_r(o.c0, 2), R=float(o.R), vel=_r(o.w, 2)))
    return out


def _vessel_stats(fname, names):
    """touched / reached / paired ordering from the stored 30-seed result file (names = its three config keys)"""
    try:
        d = json.load(open(os.path.join(RES, fname)))
        R = [d[k]["runs"] for k in names]
        n = len(R[0])
        return dict(n=n, touched=[int(sum(x["collided"] for x in rr)) for rr in R],
                    reached=[int(sum(x["reached"] for x in rr)) for rr in R],
                    order=int(sum(1 for i in range(n) if R[0][i]["min_h"] < R[1][i]["min_h"] < R[2][i]["min_h"])))
    except Exception as e:  # pragma: no cover
        print("stats skipped:", fname, e, file=sys.stderr); return None


def scene_vessel(name, title, seed, disturbance, current, scenario, stats_file, stats_names):
    print(f"== {name} (seed {seed})", file=sys.stderr)
    boats = []
    for key, label, short, color, cfg in [("MPPI", "MPPI", "MPPI", COL["mppi"], dict(kind="mppi")),
                                          ("SCBF-MPPI", "SCBF-MPPI, 2nd-order barrier", "SCBF", COL["scbf"], dict(kind="scbf")),
                                          ("SCBF-MPPI + IS", "SCBF-MPPI + IS correction", "SCBF + IS", COL["is"], dict(kind="scbf", is_correction=True))]:
        b, H = _vessel_boat(key, label, color, dict(cfg, scenario=scenario), seed, disturbance, current)
        b["short"] = short; boats.append(b)
    return dict(kind="vessel", title=title, seed=seed, disturbance=disturbance, current=list(current), t_max=VDEF["max_time"],
                dt=sg.DT_CTRL, dt_sub=sg.DT, goal=_r(H.goal, 2), goal_r=float(H.goal_radius), r_ego=float(H.r_ego),
                alpha1=float(H.alpha1), alpha2=float(H.alpha2), obstacles=_obstacles(H), x0=_r(hb.X0, 3),
                stats=_vessel_stats(stats_file, stats_names), boats=boats)


def scene_corridor(seed=6):
    print(f"== corridor (seed {seed})", file=sys.stderr)
    env = Corridor(); s0, som, sgm = CDEF["sigma_v"], CDEF["sigma_om"], CDEF["sigma_env"]
    robots = []
    for key, label, color, kind, kw in [("MPPI", "MPPI, K = 500", COL["mppi"], "mppi", {}),
                                        ("as printed", "SCBF-MPPI as printed (variance form)", COL["var"], "scbf", dict(form="variance")),
                                        ("as claimed", "SCBF-MPPI as claimed (corrected, std form)", COL["std"], "scbf", dict(form="std"))]:
        # (the robots' short on-screen names are their keys)
        c = MPPI(env, K=500, seed=seed, sigma_v=s0, sigma_om=som) if kind == "mppi" else \
            SCBFMPPI(env, K=500, seed=seed, sigma_env=sgm, sigma_v=s0, sigma_om=som, **kw)
        t = time.time()
        r = run_corridor(c, env, sigma=sgm, seed=seed, record_samples=True)          # bridge=True: the E1 convention, so this IS the stored seed-6 run
        print(f"  {label:<44} steps {r['steps']:3d} reached {r['reached']!s:5} min_h {r['min_h']:+.2f} ({time.time()-t:.0f} s)", file=sys.stderr)
        robots.append(dict(key=key, label=label, color=color, traj=_r(r["traj"], 4), h=_r(r["hmin_series"], 3), ess=_r(r["ess_series"], 1),
                           reached=bool(r["reached"]), steps=int(r["steps"]), collision_rate=float(r["collision_rate"]), min_h=float(r["min_h"]),
                           activation=r.get("activation_frac_mean"), var_kept=r.get("var_ratio_mean"), fan=_fan_corridor(r["samples"], N_FAN_CORRIDOR)))
    xs, lo, hi = env.walls()
    stats = None
    try:
        d = json.load(open(os.path.join(RES, "E1_table1.json")))
        stats = dict(n=30, reached=[int(sum(x["reached"] for x in d[k]["runs"])) for k in ("MPPI K=500", "SCBF-var K=500", "SCBF-std K=500")])
    except Exception as e:  # pragma: no cover
        print("E1 stats skipped:", e, file=sys.stderr)
    return dict(kind="corridor", title="The paper's corridor — MPPI · as printed · as claimed (seed 6)", seed=seed, dt=0.05, T=20,
                sigma=sgm, goal=[4.0, 0.5], goal_r=0.15, walls=dict(xs=_r(xs, 3), lo=_r(lo, 3), hi=_r(hi, 3)), stats=stats, robots=robots)


def main():
    t0 = time.time()
    data = dict(meta=dict(generated="2026-09-13", loa=sg.LOA, beam=sg.BEAM, f_at_max=float(sg.F_AT_MAX), dt_ctrl=sg.DT_CTRL, dt_sub=sg.DT,
                          colors=COL, note="3-DOF planar model (surge, sway, yaw): no roll, pitch or heave is simulated; the hull is stylised to LOA x BEAM. "
                                           "Positions between the 1-s control states are interpolated for display."),
                scenes={})
    data["scenes"]["current"] = scene_vessel("current", "Unknown current — three boats, the same gust and a 0.5 m/s current the model does not know (V3, seed 0)",
                                             0, "ou", (0.0, -0.5), "static", "vessel_V3_disturbance.json",
                                             [f"{k} | gust + 0.5 m/s current" for k in KEYS])
    data["scenes"]["ferry"] = scene_vessel("ferry", "Crossing ferry — the group's Experiment-III obstacle, 3 m/s, R = 25 m (V4, seed 2)",
                                           2, "white", (0.0, 0.0), "crossing", "vessel_V4_crossing.json",
                                           ["MPPI", "SCBF-MPPI (2nd-order barrier)", "SCBF-MPPI + IS correction"])
    data["scenes"]["corridor"] = scene_corridor(6)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB) in {time.time()-t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
