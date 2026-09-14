"""Does this controller hold a real-time deadline?  The tail, not the median.

    python tests/timing_deadline.py                          # both scenarios, the full study
    python tests/timing_deadline.py --scenario crossing       # one scenario
    python tests/timing_deadline.py --no-sweep                # skip the K sweep

Why this exists.  `tests/timing_matched.py` answers "what does one cycle cost on average", and reports a
median and a mean.  A controller that runs in a fixed 1 s interval is not judged on its average.  It is
judged on the cycle that overruns, because that is the cycle where the boat gets no new command and holds
the previous one for another second while a ferry closes at 3 m/s.

The shipped figures give the median only, and in the crossing scene the shipped MEAN (0.60 s) sits BELOW
the shipped MEDIAN (0.68 s).  A left-skewed sample says the slow tail was never characterised: the mean is
being pulled down by cheap cycles, and nothing in the stored results says how expensive the dear ones got.

What this measures, at the same matched operating point the other script uses:

  * the whole per-cycle distribution, not two summary numbers: median, p90, p95, p99, max
  * how many cycles would have MISSED a deadline of one control interval, and by how much
  * the largest sample count K whose 99th percentile still fits inside the interval

Method.  One reference state sequence per scenario is generated once by plain MPPI, and every controller is
then asked to plan from each of those states in turn, so the only thing that differs between controllers is
the work each one does rather than where its own trajectory took it.  The sequence is replayed `--repeats`
times to separate machine noise from the geometry: the states are identical on every pass, so the spread
within a pass is the scene and the spread across passes is the machine.

Honest limits, which belong beside any number this prints:

  * This is CPython on a laptop, with BLAS pinned to one thread so the figures are comparable with the
    shipped ones.  It is not an estimate of what optimised code on the vessel's computer would cost.
  * A general-purpose OS gives no deadline guarantee at all.  Even a controller whose 99th percentile fits
    can be descheduled.  What is measured here is the algorithm's demand, not a scheduling guarantee.
  * The per-sample solve cost depends on how many barrier rows are active, which depends on the scene, so
    every figure below is per scenario and never a single number for "the method".
"""
import argparse
import json
import os
import statistics as stats
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # must precede the numpy import

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from scbf_mppi.vessel import solgenia as sg
from scbf_mppi.vessel import harbour as hb
from scbf_mppi.vessel.controllers import VesselMPPI, VesselSCBFMPPI, VesselDetMPPI


def pct(xs, q):
    """Percentile by nearest rank on the sorted sample; no interpolation, so the value printed is one that
    was actually measured."""
    s = sorted(xs)
    if not s:
        return float("nan")
    i = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
    return s[i]


def reference_states(scenario, cycles, seed=0):
    H = hb.harbour_crossing() if scenario == "crossing" else hb.harbour_static()
    c = VesselMPPI(H, seed=seed)
    x = hb.X0.copy()
    xs = [x.copy()]
    for _ in range(cycles):
        u = c.plan(x)
        x = sg.step(x[None], u[None], c.dt)[0]
        xs.append(x.copy())
    return H, np.array(xs[:cycles])


def time_all_cycles(make, xs, H, repeats, warmup=5):
    """Every per-cycle time, over `repeats` passes of the same fixed state sequence.

    A fresh controller is built for each pass and its clock restarted at zero.  Both matter.  plan()
    advances self.t, and in the crossing scenario the ferry's position is a function of that clock, so a
    replay on a controller carried over from the previous pass would meet the ferry somewhere else and the
    passes would not be comparable.  The warm start self.U would carry over too.  Rebuilding makes every
    pass do bit-identical work, so the spread ACROSS passes is machine noise and the spread WITHIN a pass
    is the scene.  The first few cycles of every pass are discarded, not just of the first, so that every
    recorded sample comes from a warm-started controller."""
    out, act, multi = [], [], []
    for _ in range(repeats):
        c = make(H)
        c.t = 0.0
        for k, x in enumerate(xs):
            t0 = time.perf_counter()
            c.plan(x)
            dt = time.perf_counter() - t0
            if k >= warmup:
                out.append(dt)
                act.append(float(c.last.get("activation_frac", float("nan"))))
                multi.append(float(c.last.get("multi_violation_frac", float("nan"))))
    return out, act, multi


def correlate(ts, ys):
    """Pearson correlation between per-cycle time and a per-cycle scene statistic, ignoring NaNs.

    This is what turns "some cycles are slow" into "the slow cycles are the ones where two barrier rows
    are active at once", which is a mechanism rather than an observation."""
    pairs = [(t, y) for t, y in zip(ts, ys) if y == y]
    if len(pairs) < 3:
        return float("nan")
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    ma, mb = stats.fmean(a), stats.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in pairs)
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else float("nan")


def summarise(ts, budget):
    miss = [t for t in ts if t > budget]
    return {
        "n": len(ts),
        "median_s": stats.median(ts),
        "mean_s": stats.fmean(ts),
        "p90_s": pct(ts, 90),
        "p95_s": pct(ts, 95),
        "p99_s": pct(ts, 99),
        "max_s": max(ts),
        "deadline_s": budget,
        "misses": len(miss),
        "miss_frac": len(miss) / len(ts),
        "worst_overrun_s": (max(miss) - budget) if miss else 0.0,
        "p99_frac_of_budget": pct(ts, 99) / budget,
    }


ROWS = [
    ("MPPI K=500",                     lambda H: VesselMPPI(H, seed=0)),
    ("MPPI K=2000",                    lambda H: VesselMPPI(H, K=2000, seed=0)),
    ("SCBF-MPPI (2nd-order barrier)",  lambda H: VesselSCBFMPPI(H, seed=0)),
    ("SCBF-MPPI + IS correction",      lambda H: VesselSCBFMPPI(H, is_correction=True, seed=0)),
    ("SCBF-MPPI, variance form",       lambda H: VesselSCBFMPPI(H, form="variance", seed=0)),
    ("Deterministic MPPI 4x125",       lambda H: VesselDetMPPI(H, seed=0)),
]

SWEEP_K = [125, 250, 500, 1000]


def run(scenario, cycles, repeats, do_sweep, budget):
    print(f"\n{'='*92}\n{scenario.upper()} harbour   deadline = {budget:.2f} s   "
          f"{cycles} states x {repeats} passes\n{'='*92}", flush=True)
    print(f"generating {cycles} reference states (plain MPPI) ...", flush=True)
    H, xs = reference_states(scenario, cycles)

    out = {"scenario": scenario, "cycles": cycles, "repeats": repeats,
           "control_interval_s": float(sg.DT_CTRL), "deadline_s": budget,
           "blas_threads": os.environ.get("OMP_NUM_THREADS"), "controllers": {}}

    hdr = (f"{'controller':<32s}{'n':>5s}{'med':>8s}{'p90':>8s}{'p95':>8s}{'p99':>8s}"
           f"{'max':>8s}{'miss':>7s}{'p99/budget':>12s}{'r(t,2rows)':>12s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, mk in ROWS:
        ts, act, multi = time_all_cycles(mk, xs, H, repeats)
        s = summarise(ts, budget)
        s["corr_time_vs_activation"] = correlate(ts, act)
        s["corr_time_vs_two_rows"] = correlate(ts, multi)
        s["mean_activation_frac"] = stats.fmean([a for a in act if a == a]) if any(a == a for a in act) else float("nan")
        s["mean_two_row_frac"] = stats.fmean([m for m in multi if m == m]) if any(m == m for m in multi) else float("nan")
        out["controllers"][name] = s
        out["controllers"][name]["samples_s"] = [round(t, 6) for t in ts]
        r2 = s["corr_time_vs_two_rows"]
        print(f"{name:<32s}{s['n']:5d}{s['median_s']:8.3f}{s['p90_s']:8.3f}{s['p95_s']:8.3f}"
              f"{s['p99_s']:8.3f}{s['max_s']:8.3f}{s['misses']:7d}{s['p99_frac_of_budget']:11.0%}"
              f"{('  n/a' if r2 != r2 else f'{r2:11.2f}')}",
              flush=True)

    if do_sweep:
        print(f"\nK sweep, SCBF-MPPI second-order barrier, {scenario} harbour:")
        print(f"{'K':>6s}{'med':>9s}{'p99':>9s}{'max':>9s}{'miss':>7s}{'fits at p99':>13s}")
        out["sweep_K"] = {}
        for K in SWEEP_K:
            ts, _a, _m = time_all_cycles(lambda H, K=K: VesselSCBFMPPI(H, K=K, seed=0), xs, H, repeats)
            s = summarise(ts, budget)
            out["sweep_K"][str(K)] = {k: v for k, v in s.items()}
            print(f"{K:6d}{s['median_s']:9.3f}{s['p99_s']:9.3f}{s['max_s']:9.3f}"
                  f"{s['misses']:7d}{'yes' if s['p99_s'] <= budget else 'no':>13s}", flush=True)
        fits = [K for K in SWEEP_K if out["sweep_K"][str(K)]["p99_s"] <= budget]
        out["largest_K_fitting_p99"] = max(fits) if fits else None
        print(f"\n  largest K whose 99th percentile fits {budget:.2f} s: "
              f"{out['largest_K_fitting_p99'] if fits else 'none of those tried'}")

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="both", choices=["static", "crossing", "both"])
    ap.add_argument("--cycles", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--deadline", type=float, default=None,
                    help="seconds; defaults to the control interval")
    a = ap.parse_args()

    budget = a.deadline if a.deadline else float(sg.DT_CTRL)
    scenarios = ["static", "crossing"] if a.scenario == "both" else [a.scenario]

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t_start = time.time()
    for sc in scenarios:
        out = run(sc, a.cycles, a.repeats, not a.no_sweep, budget)
        dst = os.path.join(here, "results", f"vessel_V14_deadline_{sc}.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"\nwritten: {dst}")
    print(f"\ntotal wall clock: {time.time() - t_start:.0f} s")
