"""The deadline figure.   python tests/figure_deadline.py

Draws results/vessel_V14_deadline_*.json.  Nothing is recomputed; run tests/timing_deadline.py first.

Three panels, because the result has three parts and the first one alone would mislead:

  left    the whole per-cycle distribution against the one-second interval, as a survival curve.  A median
          is one point on this curve and in the crossing scene it is the flattering one.
  middle  cycle time against the fraction of sample-timesteps with two barrier rows active at once.  This
          is the mechanism: two active rows leave the closed form for a candidate search.
  right   the sweep over sample count K.  This is the engineering answer: how much sampling fits.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

COL = {"mppi": "#1B8AA6", "var": "#C2482A", "std": "#5B4FB0", "is": "#2E7D4F", "det": "#9C6A12",
       "ink": "#20303C", "muted": "#5D7180", "grid": "#D5DEE4", "bad": "#B3261E"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11.5, "axes.edgecolor": "#5D7180",
                     "axes.labelcolor": "#20303C", "xtick.color": "#20303C", "ytick.color": "#20303C",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#D5DEE4", "grid.linewidth": 0.6, "legend.frameon": False})

STYLE = {
    "MPPI K=500":                    (COL["mppi"], "MPPI"),
    "SCBF-MPPI, variance form":      (COL["var"],  "SCBF-MPPI, variance form (as printed)"),
    "SCBF-MPPI (2nd-order barrier)": (COL["std"],  "SCBF-MPPI, std form (as claimed)"),
    "SCBF-MPPI + IS correction":     (COL["is"],   "SCBF-MPPI + IS correction"),
    "Deterministic MPPI 4x125":      (COL["det"],  "deterministic MPPI"),
}


def load(scenario):
    p = os.path.join(RES, f"vessel_V14_deadline_{scenario}.json")
    if not os.path.exists(p):
        sys.exit(f"missing {p} — run tests/timing_deadline.py first")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def survival(ax, d, title):
    """P(cycle time > t): the fraction of cycles that would still be computing at time t."""
    budget = d["deadline_s"]
    for name, (col, lab) in STYLE.items():
        blk = d["controllers"].get(name)
        if not blk:
            continue
        ts = np.sort(np.asarray(blk["samples_s"]))
        surv = 1.0 - np.arange(len(ts)) / len(ts)
        ax.step(ts, surv, where="post", color=col, lw=1.9, label=lab)
    ax.axvline(budget, color=COL["bad"], lw=1.4, ls="--")
    ax.text(budget, 1.06, f" {budget:.0f} s control interval", color=COL["bad"], fontsize=10, va="top")
    ax.set_yscale("log")
    ax.set_ylim(1.0 / 400, 1.4)
    ax.set_xlabel("wall clock for one control cycle  [s]")
    ax.set_ylabel("fraction of cycles still computing")
    ax.set_title(title, loc="left", fontsize=12)


def mechanism(ax, d):
    for name, (col, lab) in STYLE.items():
        blk = d["controllers"].get(name)
        if not blk or blk.get("mean_two_row_frac") != blk.get("mean_two_row_frac"):
            continue
        ax.scatter(blk["mean_two_row_frac"], blk["median_s"], s=70, color=col, zorder=3)
        ax.errorbar(blk["mean_two_row_frac"], blk["median_s"],
                    yerr=[[0.0], [blk["p99_s"] - blk["median_s"]]],
                    color=col, lw=1.6, capsize=4, alpha=0.7, zorder=2)
        r = blk.get("corr_time_vs_two_rows", float("nan"))
        ax.annotate(f"r = {r:.2f}", (blk["mean_two_row_frac"], blk["p99_s"]),
                    textcoords="offset points", xytext=(6, 4), color=col, fontsize=10)
    ax.axhline(d["deadline_s"], color=COL["bad"], lw=1.4, ls="--")
    ax.set_xlabel("sample-timesteps with TWO barrier rows active")
    ax.set_ylabel("median cycle, bar to the 99th percentile  [s]")
    ax.set_title("the slow cycles are the cluttered ones", loc="left", fontsize=12)


def sweep(ax, ds):
    budget = ds[0][1]["deadline_s"]
    for (label, d), mk in zip(ds, ("o-", "s--")):
        sw = d.get("sweep_K")
        if not sw:
            continue
        ks = sorted(int(k) for k in sw)
        ax.plot(ks, [sw[str(k)]["p99_s"] for k in ks], mk, color=COL["std"] if label == "crossing" else COL["mppi"],
                lw=1.9, ms=6, label=f"{label} harbour, 99th percentile")
        ax.plot(ks, [sw[str(k)]["median_s"] for k in ks], mk, color=COL["std"] if label == "crossing" else COL["mppi"],
                lw=1.1, ms=4, alpha=0.45, label=f"{label} harbour, median")
    ax.axhline(budget, color=COL["bad"], lw=1.4, ls="--")
    ax.text(ax.get_xlim()[0], budget * 1.05, f" {budget:.0f} s control interval", color=COL["bad"], fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([125, 250, 500, 1000])
    ax.set_xticklabels(["125", "250", "500", "1000"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())   # log minor ticks collide with these four
    ax.set_xlabel("rollouts per cycle  K")
    ax.set_ylabel("cycle wall clock  [s]")
    ax.set_title("how much sampling fits in the interval", loc="left", fontsize=12)


def main():
    st, cr = load("static"), load("crossing")
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2))
    survival(axes[0], cr, "crossing harbour: what the median does not say")
    mechanism(axes[1], cr)
    sweep(axes[2], [("crossing", cr), ("static", st)])
    axes[0].legend(loc="lower left", fontsize=9.5)
    axes[2].legend(loc="upper left", fontsize=9.5)

    miss = cr["controllers"]["SCBF-MPPI (2nd-order barrier)"]
    fig.text(0.008, 0.015,
             f"Matched operating point: every controller plans from the same {cr['cycles']} states, "
             f"replayed {cr['repeats']} times, BLAS pinned to one thread, CPython on a laptop.  "
             f"In the crossing harbour SCBF-MPPI with the corrected constraint has a median of "
             f"{miss['median_s']:.2f} s but a 90th percentile of {miss['p90_s']:.2f} s, and "
             f"{miss['misses']} of {miss['n']} cycles ({miss['miss_frac']:.0%}) exceed the interval.  "
             f"A general-purpose OS gives no deadline guarantee in any case; this measures the algorithm's "
             f"demand, not a scheduling guarantee.",
             fontsize=8.6, color=COL["muted"], wrap=True)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    os.makedirs(FIG, exist_ok=True)
    dst = os.path.join(FIG, "fig_V14_deadline.png")
    fig.savefig(dst, dpi=170)
    print("written:", dst)


if __name__ == "__main__":
    main()
