"""Figures for V9-V12.   python -m scbf_mppi.vessel.ext.figures_ext

Same palette and typography as the shipped vessel figures, so a new figure sits beside an old one without
looking out of place.  Every panel is drawn from results/vessel_V9..V12*.json; nothing is
recomputed here.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..experiments import RES, FIG, load

COL = {"mppi": "#1B8AA6", "var": "#C2482A", "std": "#5B4FB0", "is": "#2E7D4F", "det": "#9C6A12",
       "ink": "#20303C", "muted": "#5D7180", "grid": "#D5DEE4", "water": "#EAF2F6", "obst": "#8A9BA8",
       "filt": "#B57A00", "obs": "#2E7D4F"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11.5, "axes.edgecolor": "#5D7180",
                     "axes.labelcolor": "#20303C", "xtick.color": "#20303C", "ytick.color": "#20303C",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#D5DEE4", "grid.linewidth": 0.6, "legend.frameon": False})


def _exists(n):
    return os.path.exists(os.path.join(RES, n + ".json"))


def _rows(res):
    return [(k, v) for k, v in res.items() if not k.startswith("_")]


def _m(blk, key, default=np.nan):
    s = blk["summary"]
    return s[key]["mean"] if key in s else default


def _ci(blk, key):
    s = blk["summary"]
    if key not in s:
        return (np.nan, np.nan)
    return s[key]["lo"], s[key]["hi"]


def _touched(blk):
    return sum(1 for r in blk["runs"] if r["collided"]), len(blk["runs"])


def _ctrl_colour(name):
    n = name.lower()
    if "+ is" in n:
        return COL["is"]
    if "filter" in n:
        return COL["filt"]
    if "scbf" in n:
        return COL["std"]
    return COL["mppi"]


# ------------------------------------------------------------------------------------------------
def fig_V9():
    res = load("vessel_V9_heading")
    scen = [("A", "static harbour, gust + 0.5 m/s unknown current"),
            ("B", "crossing ferry, white force noise")]
    ctrls = ["MPPI", "SCBF-MPPI", "SCBF-MPPI + IS"]
    variants = [("no heading term", COL["muted"]), ("heading to goal w=400", COL["std"]),
                ("astern penalty w=2000", COL["is"])]
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 10.4))
    metrics = [("astern_frac", "fraction of the run spent going astern", "{:.2f}"),
               ("touched", "seeds that touched a circle (of 30)", "{:.0f}"),
               ("ttf", "time to the goal [s]", "{:.0f}")]
    for col, (sk, stitle) in enumerate(scen):
        for row, (key, ylab, fmt) in enumerate(metrics):
            ax = axes[row, col]
            w = 0.26
            for iv, (vname, vcol) in enumerate(variants):
                xs, ys, los, his = [], [], [], []
                for ic, c in enumerate(ctrls):
                    name = next((k for k, _ in _rows(res)
                                 if k.startswith(sk + " ") and f"| {c} |" in k and k.endswith(vname)), None)
                    if name is None:
                        continue
                    blk = res[name]
                    xs.append(ic + (iv - 1) * w)
                    if key == "touched":
                        t, n = _touched(blk)
                        ys.append(t); los.append(t); his.append(t)
                    else:
                        ys.append(_m(blk, key)); lo, hi = _ci(blk, key); los.append(lo); his.append(hi)
                ax.bar(xs, ys, width=w * 0.92, color=vcol, alpha=0.9,
                       label=vname if (row == 0 and col == 0) else None)
                if key != "touched":
                    ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)],
                                fmt="none", ecolor=COL["ink"], elinewidth=1.0, capsize=2.5)
                for x, y in zip(xs, ys):
                    ax.text(x, y, " " + fmt.format(y), rotation=90, ha="center", va="bottom",
                            fontsize=8.5, color=COL["ink"])
            ax.set_xticks(range(len(ctrls)))
            ax.set_xticklabels(ctrls, fontsize=10)
            ax.set_ylabel(ylab, fontsize=10)
            ax.margins(y=0.22)
            if row == 0:
                ax.set_title(f"{sk}  {stitle}", fontsize=11.5, color=COL["ink"])
    axes[0, 0].legend(loc="upper left", fontsize=9.5, ncol=1)
    fig.suptitle("V9  the vessel cost has no heading term, so the boats run astern — does adding one change "
                 "the comparison?", fontsize=12.5, color=COL["ink"], y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(os.path.join(FIG, "fig_V9_heading.png"), dpi=200)
    plt.close(fig)


def fig_V10():
    res = load("vessel_V10_filter")
    order = ["MPPI (no barrier)", "SCBF-MPPI (barrier in the sampler)", "SCBF-MPPI + IS",
             "MPPI + CBF-QP filter", "MPPI + chance CBF-QP filter", "SCBF-MPPI + CBF-QP filter"]
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.4))
    for col, sk in enumerate(["A", "B"]):
        ax = axes[col]
        for a in order:
            name = next((k for k, _ in _rows(res) if k.startswith(sk + " ") and k.endswith(a)), None)
            if name is None:
                continue
            blk = res[name]
            t, n = _touched(blk)
            x = _m(blk, "ttf")
            y = t / n
            ax.scatter([x], [y], s=110, color=_ctrl_colour(a), zorder=3,
                       marker="s" if "filter" in a else "o",
                       edgecolor=COL["ink"], linewidth=0.7)
            ax.annotate(a.replace("SCBF-MPPI", "SCBF").replace(" (barrier in the sampler)", " sampler")
                        .replace(" (no barrier)", ""), (x, y), textcoords="offset points",
                        xytext=(7, 5), fontsize=8.8, color=COL["ink"])
        ax.set_xlabel("time to the goal [s]   (conservatism)")
        ax.set_ylabel("seeds that touched a circle")
        ax.set_title(("A  gust + 0.5 m/s unknown current" if sk == "A" else "B  crossing ferry"),
                     fontsize=11.5, color=COL["ink"])
        ax.margins(0.22)
    ax = axes[2]
    tim = res.get("_timing", {})
    labs = [k for k in tim if k != "control interval s"]
    vals = [tim[k] for k in labs]
    ys = np.arange(len(labs))[::-1]
    ax.barh(ys, vals, color=[_ctrl_colour(l) for l in labs], alpha=0.9, height=0.6)
    ax.axvline(tim.get("control interval s", 1.0), color=COL["var"], lw=1.4, ls="--")
    ax.text(tim.get("control interval s", 1.0), len(labs) - 0.3, " control interval",
            color=COL["var"], fontsize=9, va="top")
    ax.set_yticks(ys)
    ax.set_yticklabels([l.replace("SCBF-MPPI", "SCBF") for l in labs], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("wall clock per control cycle [s]")
    ax.set_title("what the architecture costs", fontsize=11.5, color=COL["ink"])
    for y, v in zip(ys, vals):
        ax.text(v, y, f"  {v*1000:.0f} ms", fontsize=8.8, va="center", color=COL["ink"])
    fig.suptitle("V10  the same barrier, in the sampler or on the output", fontsize=12.5,
                 color=COL["ink"], y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(FIG, "fig_V10_filter.png"), dpi=200)
    plt.close(fig)


def fig_V11():
    res = load("vessel_V11_observer")
    names = [k for k, _ in _rows(res) if k.startswith("SCBF-MPPI |") and "1.0 m/s" not in k]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))

    ax = axes[0]
    ys = np.arange(len(names))[::-1]
    errs = [_m(res[n], "hdot_err_mean") for n in names]
    psis = [_m(res[n], "psi1_abs_mean") for n in names]
    ax.barh(ys, errs, color=COL["var"], alpha=0.85, height=0.55)
    ax.plot(psis, ys, "o", color=COL["ink"], ms=5, label="|psi1| for scale")
    ax.set_yticks(ys)
    ax.set_yticklabels([n.split("| ", 1)[1] for n in names], fontsize=9)
    ax.set_xlabel("error in the barrier derivative  |n · (c_believed − c_true)|  [m/s]")
    ax.set_title("what the unknown current does to hdot", fontsize=11.5, color=COL["ink"])
    ax.legend(fontsize=9, loc="lower right")
    for y, v in zip(ys, errs):
        ax.text(v, y, f"  {v:.3f}", fontsize=8.8, va="center", color=COL["ink"])

    ax = axes[1]
    for n in names:
        r0 = next((r for r in res[n]["runs"] if r.get("obs_hist")), None)
        if r0 is None:
            continue
        h = np.asarray(r0["obs_hist"], float)
        ax.plot(np.arange(len(h)), h[:, 1], lw=1.6, label=n.split("| ", 1)[1])
    ax.axhline(-0.5, color=COL["ink"], ls="--", lw=1.1)
    ax.text(2, -0.455, "true current  −0.5 m/s", fontsize=9, color=COL["ink"])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("estimated northward current [m/s]")
    ax.set_title("the estimate, seed 0", fontsize=11.5, color=COL["ink"])
    ax.legend(fontsize=8.2, loc="upper right", ncol=1)

    ax = axes[2]
    ys = np.arange(len(names))[::-1]
    ttf = [_m(res[n], "ttf") for n in names]
    ax.barh(ys, ttf, color=COL["mppi"], alpha=0.85, height=0.55)
    ax.set_yticks(ys)
    ax.set_yticklabels([n.split("| ", 1)[1] for n in names], fontsize=9)
    ax.set_xlabel("time to the goal [s]")
    ax.set_title("the cost of not knowing", fontsize=11.5, color=COL["ink"])
    ax.set_xlim(0, max(ttf) * 1.45)
    for y, v, n in zip(ys, ttf, names):
        s = res[n]["summary"]
        t, tot = _touched(res[n])
        ax.text(v, y, f"  {v:.0f} s · reached {s['reached_frac']*100:.0f}% · touched {t}/{tot}",
                fontsize=8.5, va="center", color=COL["ink"])

    fig.suptitle("V11  an unmodelled 0.5 m/s current against a nominal delta of 0.003", fontsize=12.5,
                 color=COL["ink"], y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, "fig_V11_observer.png"), dpi=200)
    plt.close(fig)


def fig_V12():
    res = load("vessel_V12_ess")
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))
    groups = [("SCBF-MPPI + IS", COL["is"]), ("SCBF-MPPI (no IS)", COL["std"]), ("MPPI,", COL["mppi"])]
    for col, sk in enumerate(["A", "B"]):
        ax = axes[col]
        for g, c in groups:
            names = [k for k, _ in _rows(res) if k.startswith(sk + " ") and g in k]
            if not names:
                continue
            xs = [_m(res[n], "ess_median") for n in names]
            ys = [_m(res[n], "dU_norm_mean") for n in names]
            rs = [res[n]["summary"]["reached_frac"] for n in names]
            ax.plot(xs, ys, "-", color=c, lw=1.2, alpha=0.6)
            ax.scatter(xs, ys, s=[40 + 220 * r for r in rs], color=c, edgecolor=COL["ink"],
                       linewidth=0.7, zorder=3, label=g.rstrip(","))
            for i, (x, y, n, r) in enumerate(zip(xs, ys, names, rs)):
                tag = n.split("| ")[-1]
                tag = (tag.replace("SCBF-MPPI + IS, ", "").replace("SCBF-MPPI (no IS), ", "")
                          .replace("MPPI, ", "").replace("fixed lambda = 300", "fixed λ")
                          .replace("ESS target ", "→"))
                if r < 0.999:                                  # only the interesting ones carry the reach
                    tag += f"  (reach {r:.0%})"
                off = (8, 6) if i % 2 == 0 else (8, -12)
                ax.annotate(tag, (x, y), textcoords="offset points", xytext=off,
                            fontsize=8.2, color=COL["ink"])
        ax.set_xlabel("median effective sample size  (of K = 500)")
        ax.set_ylabel("‖control update‖ per cycle  [N]")
        ax.set_title(("A  gust + 0.5 m/s unknown current" if sk == "A" else "B  crossing ferry"),
                     fontsize=11.5, color=COL["ink"])
        ax.margins(0.25)
        if col == 0:
            ax.legend(fontsize=9, loc="lower right")

    ax = axes[2]
    names = [k for k, _ in _rows(res) if "ESS target" in k]
    ys = np.arange(len(names))[::-1]
    met = [_m(res[n], "ess_target_met_mean") for n in names]
    ax.barh(ys, met, color=[COL["is"] if "+ IS" in n else COL["std"] if "SCBF" in n else COL["mppi"]
                            for n in names], alpha=0.85, height=0.55)
    ax.set_yticks(ys)
    ax.set_yticklabels([n.replace("gust + 0.5 m/s current ", "").replace("crossing ferry ", "")
                        .replace("SCBF-MPPI + IS, ", "+IS ").replace("SCBF-MPPI (no IS), ", "SCBF ")
                        .replace("MPPI, ", "MPPI ").replace("ESS target ", "→") for n in names], fontsize=8.5)
    ax.set_xlabel("cycles where the target was reachable")
    ax.set_xlim(0, 1.12)
    ax.set_title("the ceiling the temperature cannot raise", fontsize=11.5, color=COL["ink"])
    for y, v in zip(ys, met):
        if np.isfinite(v):
            ax.text(v, y, f"  {v:.2f}", fontsize=8.8, va="center", color=COL["ink"])

    fig.suptitle("V12  raising the temperature buys effective sample size by flattening the weights — "
                 "the size of the marker is the fraction of seeds that still reach the goal",
                 fontsize=12.0, color=COL["ink"], y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, "fig_V12_ess.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    made = []
    for tag, fn in [("vessel_V9_heading", fig_V9), ("vessel_V10_filter", fig_V10),
                    ("vessel_V11_observer", fig_V11), ("vessel_V12_ess", fig_V12)]:
        if not _exists(tag):
            print(f"  skipped {tag} (no results yet)")
            continue
        try:
            fn()
            made.append(tag)
            print(f"  wrote figure for {tag}")
        except Exception as exc:
            print(f"  FAILED {tag}: {exc.__class__.__name__}: {exc}")
    print(f"{len(made)} figures written to {FIG}")
