"""Wall-clock per control cycle, measured at the SAME operating point for every controller.

    python tests/timing_matched.py                     # static harbour, 40 cycles
    python tests/timing_matched.py --scenario crossing --cycles 40

Why this exists.  The shipped timing experiment (vessel/experiments.py, V5) starts every controller at the
same initial state and then lets each one follow its OWN trajectory for 25 cycles.  After a few seconds the
controllers are no longer in the same place, and the cost of the per-sample chance-constraint solve depends
on how many barrier rows are ACTIVE, which depends on where the boat is.  The comparison therefore mixes
"how expensive is this algorithm" with "how close to an obstacle did this algorithm end up".

The symptom is visible in both result trees: SCBF-MPPI is reported at 0.91 s (shipped) and 0.68 s
(regenerated) per cycle while SCBF-MPPI + IS, which does everything SCBF-MPPI does and then adds a
log-determinant per timestep, is reported at 0.31 s and 0.23 s -- three times FASTER, which no amount of
noise explains.  The importance-sampling variant keeps a larger margin (closest approach 4.6 m against
1.0 m in V2), so far fewer rows are active, so its solves are cheaper.  That is a fact about the
trajectories, not about the algorithms.

This script removes the confound: one reference state sequence is generated once, and every controller is
then asked to plan from each of those states in turn.  The states are identical across controllers, so the
only thing that differs is the work each one does.  It also reports the mean activation fraction, so the
operating point is visible rather than implied.

Both numbers are worth having.  The matched one answers "what does this algorithm cost per cycle".  The
shipped one answers "what did this controller cost in the run it actually produced".  They should be quoted
with the right question attached; this script prints them side by side.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from scbf_mppi.vessel import solgenia as sg
from scbf_mppi.vessel import harbour as hb
from scbf_mppi.vessel.controllers import VesselMPPI, VesselSCBFMPPI, VesselDetMPPI


def reference_states(scenario, cycles, seed=0):
    """A single trajectory through the field, generated once by plain MPPI, used by every controller."""
    H = hb.harbour_crossing() if scenario == "crossing" else hb.harbour_static()
    c = VesselMPPI(H, seed=seed)
    x = hb.X0.copy()
    xs = [x.copy()]
    for _ in range(cycles):
        u = c.plan(x)
        x = sg.step(x[None], u[None], c.dt)[0]
        xs.append(x.copy())
    return H, np.array(xs[:cycles])


def time_controller(make, xs, H, warmup=5):
    c = make(H)
    c.t = 0.0
    ts = []
    act = []
    for k, x in enumerate(xs):
        t0 = time.perf_counter()
        c.plan(x)                       # the state is FED IN, so every controller sees the same sequence
        dt = time.perf_counter() - t0
        if k >= warmup:
            ts.append(dt)
            act.append(c.last.get("activation_frac", np.nan))
    return float(np.median(ts)), float(np.mean(ts)), float(np.nanmean(act)) if act else float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="static", choices=["static", "crossing"])
    ap.add_argument("--cycles", type=int, default=40)
    a = ap.parse_args()

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if not os.environ.get(v):
            print(f"note: {v} is not set; a multi-threaded BLAS makes these numbers noisier")

    print(f"generating {a.cycles} reference states ({a.scenario} harbour, plain MPPI) ...", flush=True)
    H, xs = reference_states(a.scenario, a.cycles)

    rows = [
        ("MPPI K=500", lambda H: VesselMPPI(H, seed=0)),
        ("MPPI K=2000", lambda H: VesselMPPI(H, K=2000, seed=0)),
        ("SCBF-MPPI (2nd-order barrier)", lambda H: VesselSCBFMPPI(H, seed=0)),
        ("SCBF-MPPI + IS correction", lambda H: VesselSCBFMPPI(H, is_correction=True, seed=0)),
        ("SCBF-MPPI, variance form", lambda H: VesselSCBFMPPI(H, form="variance", seed=0)),
        ("Deterministic MPPI 4x125", lambda H: VesselDetMPPI(H, seed=0)),
    ]
    out = {"scenario": a.scenario, "cycles": a.cycles, "control interval s": sg.DT_CTRL, "matched": {}}
    print(f"\n{'controller':<34s}{'median s':>10s}{'mean s':>10s}{'activation':>12s}")
    for name, mk in rows:
        med, mean, act = time_controller(mk, xs, H)
        out["matched"][name] = {"median_s": med, "mean_s": mean, "activation_frac": act}
        print(f"{name:<34s}{med:10.4f}{mean:10.4f}{act:12.3f}", flush=True)

    # the shipped, unmatched numbers, for comparison
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for tree in ("results", "results_shipped_2026_09_12"):
        p = os.path.join(here, tree, "vessel_V5_wallclock.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            out[f"unmatched_{tree}"] = {k: v for k, v in d.items() if isinstance(v, (int, float))}

    dst = os.path.join(here, "results", "vessel_V13_timing_matched.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwritten: {dst}")
    print("\nEach controller planned from the SAME 40 states, so the only difference is the work done.")
