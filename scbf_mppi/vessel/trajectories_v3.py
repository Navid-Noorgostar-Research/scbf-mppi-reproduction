"""The 30-seed trajectories of experiment V3's 'gust + 0.5 m/s current' scenario, for the ghost overlay of
animate_current --ghosts.  vessel_V3_disturbance.json keeps traj_xy only for seeds 0-2 (size); this reruns the same
three configurations on the same 30 seeds through the same code path (make_controller + run_episode) and stores the
planar paths, then cross-checks every seed's min_h / ttf / collided against the stored V3 run.
    python -m scbf_mppi.vessel.trajectories_v3 [procs]     -> results/vessel_V3_trajectories.json   (~5 min on 8 cores)
"""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool
from .experiments import make_controller, RES, DEFAULT
from .simulate import run_episode

SCEN = "gust + 0.5 m/s current"
KEYS = ["MPPI", "SCBF-MPPI", "SCBF-MPPI + IS"]
CFGS = {"MPPI": dict(kind="mppi", disturbance="ou", current=(0.0, -0.5)),
        "SCBF-MPPI": dict(kind="scbf", disturbance="ou", current=(0.0, -0.5)),
        "SCBF-MPPI + IS": dict(kind="scbf", is_correction=True, disturbance="ou", current=(0.0, -0.5))}


def _job(args):
    key, seed = args
    cfg = CFGS[key]
    c, H = make_controller(cfg, seed)
    r = run_episode(c, H, seed=seed, disturbance=cfg["disturbance"], s_F=np.asarray(DEFAULT["s_F"], float),
                    tau_c=DEFAULT["tau_c"], current=tuple(cfg["current"]), max_time=DEFAULT["max_time"])
    return key, seed, dict(seed=seed, traj_xy=np.round(r["traj"][:, :2], 3).tolist(), collided=bool(r["collided"]),
                           min_h=float(r["min_h"]), ttf=float(r["ttf"]), reached=bool(r["reached"]))


def main(procs=8, n_seeds=30):
    jobs = [(k, s) for k in KEYS for s in range(n_seeds)]
    t = time.time()
    with Pool(procs) as p:
        outs = p.map(_job, jobs, chunksize=1)
    print(f"  {len(jobs)} episodes in {time.time()-t:.0f}s", file=sys.stderr)
    out = {"scenario": SCEN, "seeds": list(range(n_seeds)), "configs": {k: {"runs": []} for k in KEYS}, "cross_check": {}}
    for key, seed, rec in outs:
        out["configs"][key]["runs"].append(rec)
    V3 = json.load(open(os.path.join(RES, "vessel_V3_disturbance.json")))
    for key in KEYS:
        runs = sorted(out["configs"][key]["runs"], key=lambda r: r["seed"]); out["configs"][key]["runs"] = runs
        stored = V3[f"{key} | {SCEN}"]["runs"]
        cc = dict(n_touched=int(sum(r["collided"] for r in runs)),
                  n_touched_stored=int(sum(r["collided"] for r in stored)),
                  collided_agree=int(sum(a["collided"] == b["collided"] for a, b in zip(runs, stored))),
                  max_abs_dmin_h=float(max(abs(a["min_h"] - b["min_h"]) for a, b in zip(runs, stored))),
                  max_abs_dttf=float(max(abs(a["ttf"] - b["ttf"]) for a, b in zip(runs, stored))))
        out["cross_check"][key] = cc
        print(f"  {key:<16} touched {cc['n_touched']}/{n_seeds} (stored {cc['n_touched_stored']}) | collided agree "
              f"{cc['collided_agree']}/{n_seeds} | max|Δmin_h| {cc['max_abs_dmin_h']:.2e} m  max|Δttf| {cc['max_abs_dttf']:.0f} s", file=sys.stderr)
    path = os.path.join(RES, "vessel_V3_trajectories.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote", path, f"({os.path.getsize(path)/1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main(procs=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
