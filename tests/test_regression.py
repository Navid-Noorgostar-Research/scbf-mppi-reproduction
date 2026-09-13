"""Bit-exactness guard for the shipped vessel pipeline.

Any change to solgenia / harbour / controllers / simulate that is meant to be *additive* must leave the
shipped configurations bit-for-bit unchanged.  This script captures a reference snapshot of a handful of
(config, seed) pairs and later re-checks them.

    python tests/test_regression.py capture --out tests/regression_ref.json
    python tests/test_regression.py check   --ref tests/regression_ref.json

The snapshot is a SHA-256 of the raw float64 bytes of the closed-loop trajectory and the control sequence,
plus the scalar summaries.  It is therefore sensitive to the last bit of every state.

NOTE ON LIBRARY DRIFT: the reference must be captured with the SAME interpreter and the same BLAS thread
settings as the check, otherwise a difference proves nothing about the source change.  Run both through
`reproduce.py` (or with OMP_NUM_THREADS=1) so the two are comparable.
"""
import argparse, hashlib, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scbf_mppi.vessel.experiments import make_controller, DEFAULT
from scbf_mppi.vessel.simulate import run_episode

# Configurations chosen to touch every shipped code path: plain MPPI, the second-order barrier, the
# importance-sampling correction, the printed relative-degree-1 rows, the variance form, the deterministic
# planner, the moving-obstacle scenario, and the unknown-current disturbance.
CASES = [
    ("MPPI static white",        dict(name="a", kind="mppi"), 0),
    ("MPPI static white",        dict(name="a", kind="mppi"), 2),
    ("SCBF hocbf std",           dict(name="b", kind="scbf"), 0),
    ("SCBF hocbf std",           dict(name="b", kind="scbf"), 1),
    ("SCBF + IS",                dict(name="c", kind="scbf", is_correction=True), 0),
    ("SCBF variance form",       dict(name="d", kind="scbf", form="variance"), 0),
    ("SCBF printed rd1",         dict(name="e", kind="scbf", mode="printed_rd1", form="variance"), 0),
    ("Det MPPI 4x125",           dict(name="f", kind="det"), 0),
    ("SCBF crossing ferry",      dict(name="g", kind="scbf", scenario="crossing"), 2),
    ("MPPI crossing ferry",      dict(name="h", kind="mppi", scenario="crossing"), 2),
    ("SCBF gust + current",      dict(name="i", kind="scbf", disturbance="ou", current=(0.0, -0.5)), 0),
    ("SCBF + IS gust + current", dict(name="j", kind="scbf", is_correction=True, disturbance="ou", current=(0.0, -0.5)), 0),
]

SCALARS = ("ttf", "reached", "collision_rate", "collision_rate_ctrl", "collided", "min_h",
           "ess_mean", "ess_median", "effort", "path_length", "mean_speed",
           "activation_frac_mean", "var_ratio_mean", "sat_active_mean", "infeasible_frac_mean")


def _digest(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:32]


def run_case(cfg, seed):
    c, H = make_controller(cfg, seed)
    r = run_episode(c, H, seed=seed,
                    disturbance=cfg.get("disturbance", DEFAULT["disturbance"]),
                    s_F=np.asarray(cfg.get("s_F", DEFAULT["s_F"]), float),
                    tau_c=cfg.get("tau_c", DEFAULT["tau_c"]),
                    current=tuple(cfg.get("current", (0.0, 0.0))),
                    max_time=cfg.get("max_time", DEFAULT["max_time"]))
    out = {k: (float(r[k]) if not isinstance(r[k], bool) else bool(r[k])) for k in SCALARS if r.get(k) is not None}
    out["traj_sha"] = _digest(r["traj"])
    out["ctrl_sha"] = _digest(r["ctrl"])
    out["hmin_sha"] = _digest(r["hmin_series"])
    out["steps"] = int(r["steps"])
    return out


def capture(path):
    snap = {"cases": {}, "env": {"numpy": np.__version__, "python": sys.version.split()[0],
                                 "threads": {k: os.environ.get(k) for k in
                                             ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}}
    for label, cfg, seed in CASES:
        t0 = time.time()
        snap["cases"][f"{label} | seed {seed}"] = run_case(cfg, seed)
        print(f"  captured {label} seed {seed}  ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(snap, f, indent=1)
    print(f"reference written: {path}")
    return snap


def check(path, verbose=True):
    with open(path) as f:
        ref = json.load(f)
    bad = []
    for label, cfg, seed in CASES:
        key = f"{label} | seed {seed}"
        if key not in ref["cases"]:
            continue
        got = run_case(cfg, seed); want = ref["cases"][key]
        diffs = [k for k in want if k in got and got[k] != want[k]]
        if diffs:
            bad.append((key, {k: (want[k], got[k]) for k in diffs}))
            if verbose:
                print(f"  MISMATCH {key}: {diffs}", flush=True)
        elif verbose:
            print(f"  ok       {key}", flush=True)
    if bad:
        print(f"\nFAILED: {len(bad)} of {len(CASES)} cases changed.")
        for key, d in bad:
            for k, (w, g) in d.items():
                print(f"  {key}  {k}: reference {w}  ->  now {g}")
        return 1
    print(f"\nPASS: all {len(CASES)} shipped cases reproduce bit for bit.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "check"])
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "regression_ref.json"))
    ap.add_argument("--ref", default=os.path.join(here, "regression_ref.json"))
    a = ap.parse_args()
    if a.mode == "capture":
        capture(a.out)
    else:
        sys.exit(check(a.ref))
