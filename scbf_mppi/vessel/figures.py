"""Figures from results/vessel_V*.json.   python -m scbf_mppi.vessel.figures"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from . import solgenia as sg, harbour as hb
from .experiments import RES, FIG, load, DEFAULT

# same validated palette as the corridor figures; fixed assignment by controller identity
COL = {"mppi": "#1B8AA6", "var": "#C2482A", "std": "#5B4FB0", "is": "#2E7D4F", "det": "#9C6A12",
       "ink": "#20303C", "muted": "#5D7180", "grid": "#D5DEE4", "water": "#EAF2F6", "obst": "#8A9BA8"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": "#5D7180", "axes.labelcolor": "#20303C",
                     "xtick.color": "#20303C", "ytick.color": "#20303C", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#D5DEE4", "grid.linewidth": 0.6, "legend.frameon": False})


def colour_for(name):
    n = name.lower()
    if n.startswith("mppi"): return COL["mppi"]
    if "printed" in n and "rel" in n: return COL["mppi"]
    if "variance" in n: return COL["var"]
    if "is" in n.split() or "+ is" in n: return COL["is"]
    if "deterministic" in n: return COL["det"]
    return COL["std"]


def short(name):
    return (name.replace("SCBF-MPPI (2nd-order barrier)", "SCBF-MPPI, 2nd-order barrier")
                .replace("SCBF-MPPI + IS correction", "SCBF-MPPI + IS correction")
                .replace("SCBF-MPPI, variance form as printed", "SCBF-MPPI, variance form (as printed)"))


def dotci(ax, names, res, key, xlabel, fmt="{:.3f}", labels=None):
    ys = np.arange(len(names))[::-1]
    for y, n in zip(ys, names):
        s = res[n]["summary"][key]
        ax.plot([s["lo"], s["hi"]], [y, y], color=colour_for(n), lw=2, solid_capstyle="round")
        ax.plot(s["mean"], y, "o", color=colour_for(n), ms=6)
        ax.text(s["hi"], y + 0.18, fmt.format(s["mean"]), fontsize=9, color=COL["ink"], ha="left", va="bottom")
    ax.set_yticks(ys); ax.set_yticklabels(labels if labels else [short(n) for n in names]); ax.set_xlabel(xlabel); ax.grid(axis="y", visible=False)


def draw_harbour(ax, H, t=0.0, ferry_track=True):
    ax.set_facecolor(COL["water"])
    for o in H.obs:
        c = o.center(t)
        ax.add_patch(Circle(c, o.R, facecolor=COL["obst"], edgecolor=COL["ink"], lw=0.8, alpha=0.55))
        ax.add_patch(Circle(c, max(o.R - H.r_ego, 0.5), facecolor=COL["ink"], edgecolor="none", alpha=0.35))
        if np.linalg.norm(o.w) > 0 and ferry_track:
            ax.annotate("", xy=c + o.w * 8, xytext=c, arrowprops=dict(arrowstyle="->", color=COL["ink"], lw=1.2))
        ax.text(c[0], c[1] + o.R + 2.0, o.name, fontsize=8, color=COL["ink"], ha="center", bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.75))
    ax.plot(H.goal[0], H.goal[1], "x", color=COL["ink"], ms=10, mew=2)
    ax.add_patch(Circle(H.goal, H.goal_radius, facecolor="none", edgecolor=COL["ink"], lw=0.8, ls="--"))
    ax.plot(hb.X0[0], hb.X0[1], "o", color=COL["ink"], ms=5)
    ax.set_aspect("equal"); ax.set_xlabel("east, m"); ax.set_ylabel("north, m")


def fig_V1():
    res = load("vessel_V1_printed")
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    H = hb.harbour_static(); draw_harbour(ax, H)
    for n, lw, ls in [("MPPI", 3.2, "-"), ("SCBF as printed (rel. degree 1)", 1.4, "--")]:
        t = np.array(res[n]["runs"][0]["traj_xy"]); ax.plot(t[:, 0], t[:, 1], color=colour_for(n) if n == "MPPI" else "white", lw=lw, ls=ls, label=n)
    inf = res["SCBF as printed (rel. degree 1)"]["summary"]["infeasible_frac_mean"]["mean"]
    ax.set_title(f"As printed, L_g h = 0 on a vessel: the two runs coincide (max |Δ| = {res['_max_abs_difference']:g}); the row was unfixable on {100*inf:.0f} % of pairs",
                 fontsize=9.5, loc="left")
    ax.set_xlim(-10, 130); ax.set_ylim(-35, 35); ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_V1_printed.png"), dpi=200); plt.close(fig)


def fig_V2():
    res = load("vessel_V2_harbour"); H = hb.harbour_static()
    names = [n for n in res if not n.startswith("_")]
    # (a) map with one run each
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    draw_harbour(ax, H)
    for n, lab in [("MPPI K=500", "MPPI"), ("SCBF-MPPI (2nd-order barrier)", "SCBF-MPPI, 2nd-order barrier"),
                   ("SCBF-MPPI + IS correction", "+ importance-sampling correction")]:
        t = np.array(res[n]["runs"][0]["traj_xy"]); ax.plot(t[:, 0], t[:, 1], color=colour_for(n), lw=1.8, label=lab)
    ax.set_xlim(-10, 130); ax.set_ylim(-35, 35); ax.legend(loc="lower left", fontsize=9)
    d = DEFAULT
    ax.set_title("Solgenia model (Homburger et al. 2025), seed 0 · K = 500 · 15 s horizon · white force noise · circles: obstacle + 5 m ego radius",
                 fontsize=9.5, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_V2_map.png"), dpi=200); plt.close(fig)
    # (b) dot-CI charts
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    dotci(axes[0], names, res, "collision_rate", "collision rate")
    dotci(axes[1], names, res, "ttf", "time to goal, s", fmt="{:.0f}")
    dotci(axes[2], names, res, "min_h", "closest approach, m", fmt="{:.1f}")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_V2_table.png"), dpi=200); plt.close(fig)
    # (c) speed and clearance traces (seed 0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for n, lab in [("MPPI K=500", "MPPI"), ("SCBF-MPPI (2nd-order barrier)", "SCBF-MPPI"), ("SCBF-MPPI + IS correction", "+ IS correction")]:
        r = res[n]["runs"][0]
        axes[0].plot(np.arange(1, len(r["speed_series"]) + 1), r["speed_series"], color=colour_for(n), lw=1.6, label=lab)
        axes[1].plot(np.arange(1, len(r["hmin_series"]) + 1), r["hmin_series"], color=colour_for(n), lw=1.6, label=lab)
    axes[0].set_xlabel("time, s"); axes[0].set_ylabel("surge speed, m/s"); axes[0].legend(fontsize=9)
    axes[1].axhline(0, color=COL["ink"], lw=0.8); axes[1].set_xlabel("time, s"); axes[1].set_ylabel("min h, m")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_V2_traces.png"), dpi=200); plt.close(fig)


def fig_V3():
    res = load("vessel_V3_disturbance")
    dists = ["white", "gust (OU, 5 s)", "gust + 0.5 m/s current"]
    ctrls = [("MPPI", COL["mppi"]), ("SCBF-MPPI", COL["std"]), ("SCBF-MPPI + IS", COL["is"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
    for ax, key, xl, fmt in [(axes[0], "collision_rate", "collision rate · mean and 95 % CI", "{:.3f}"), (axes[1], "ttf", "time to goal, s", "{:.0f}")]:
        y = 0; ticks = []; labels = []
        for dn in dists:
            for cn, col in ctrls:
                s = res[f"{cn} | {dn}"]["summary"][key]
                ax.plot([s["lo"], s["hi"]], [y, y], color=col, lw=2, solid_capstyle="round"); ax.plot(s["mean"], y, "o", color=col, ms=6)
                ax.text(s["hi"], y + 0.15, fmt.format(s["mean"]), fontsize=10, color=COL["ink"], va="bottom")
                ticks.append(y); labels.append(f"{cn.replace('SCBF-MPPI + IS', '+ IS').replace('SCBF-MPPI', 'SCBF')} · {dn.replace('gust (OU, 5 s)', 'gust').replace('gust + 0.5 m/s current', 'gust + current')}"); y -= 1
            y -= 0.6
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=10.5); ax.set_xlabel(xl, fontsize=11); ax.grid(axis="y", visible=False)
    fig.suptitle("Equal low-frequency intensity: white (the paper's model) · a 5 s gust · gust + an unknown 0.5 m/s cross-current", fontsize=11, x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_V3_disturbance.png"), dpi=200); plt.close(fig)


def fig_V4():
    res = load("vessel_V4_crossing"); H = hb.harbour_crossing()
    names = [n for n in res if not n.startswith("_")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]; draw_harbour(ax, H, t=0.0)
    for n, lab in [("MPPI", "MPPI"), ("SCBF-MPPI (2nd-order barrier)", "SCBF-MPPI"), ("SCBF-MPPI + IS correction", "+ IS correction")]:
        t = np.array(res[n]["runs"][0]["traj_xy"]); ax.plot(t[:, 0], t[:, 1], color=colour_for(n), lw=1.8, label=lab)
        # mark the position at t = 22 s (ferry on the track)
        if len(t) > 22: ax.plot(t[22, 0], t[22, 1], "s", color=colour_for(n), ms=5)
    ferry = H.obs[-1]
    for tt in (10, 22, 34):
        c = ferry.center(tt); ax.add_patch(Circle(c, ferry.R, facecolor="none", edgecolor=COL["ink"], lw=0.8, ls=":"))
        ax.text(c[0] + ferry.R + 1, c[1], f"t = {tt} s", fontsize=8, color=COL["ink"], va="center", bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.8))
    ax.set_xlim(-10, 130); ax.set_ylim(-92, 50); ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Crossing ferry (R = 25 m incl. margin, 3 m/s — the group's Experiment III obstacle); squares: position at t = 22 s", fontsize=9.5, loc="left")
    dotci(axes[1], names, res, "collision_rate", "collision rate · mean and 95 % CI")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_V4_crossing.png"), dpi=200); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    dotci(axes[0], names, res, "ttf", "time to goal, s", fmt="{:.0f}")
    dotci(axes[1], names, res, "min_h", "closest approach, m", fmt="{:.1f}")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_V4_crossing_b.png"), dpi=200); plt.close(fig)


def fig_V5():
    w = load("vessel_V5_wallclock")
    items = [("MPPI K=500", w["MPPI K=500"], COL["mppi"]), ("MPPI K=2000", w["MPPI K=2000"], COL["mppi"]),
             ("Deterministic MPPI 4x125", w["Deterministic MPPI 4x125"], COL["det"]),
             ("SCBF-MPPI, closed-form\nper-sample solve (NumPy)", w["SCBF-MPPI closed-form per-sample solve"], COL["std"]),
             ("SCBF-MPPI + IS", w["SCBF-MPPI + IS"], COL["is"]),
             ("Algorithm 1 with the SDP (8)\nas printed, solver time only", w["Algorithm 1 cycle with literal SDP, solver-only s"], COL["var"]),
             ("Algorithm 1 with the SDP (8)\nas printed, through cvxpy", w["Algorithm 1 cycle with literal SDP, cvxpy wall s"], COL["var"])]
    fig, ax = plt.subplots(figsize=(9.4, 4.3))
    ys = np.arange(len(items))[::-1]
    for y, (n, v, c) in zip(ys, items):
        ax.barh(y, v, color=c, height=0.62)
        ax.text(v * 1.15, y, f"{v:.3g} s" if v < 100 else f"{v:.0f} s", va="center", fontsize=9.5, color=COL["ink"], bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    ax.axvline(w["control interval s"], color=COL["ink"], lw=1, ls="--"); ax.text(w["control interval s"] * 1.06, ys[-1] - 0.62, "control interval 1 s", fontsize=8.5, color=COL["ink"], va="top")
    ax.set_xscale("log"); ax.set_yticks(ys); ax.set_yticklabels([i[0] for i in items], fontsize=9.5); ax.set_xlabel("wall-clock per control cycle, s (log) · 7,500 per-sample problems per cycle")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_V5_wallclock.png"), dpi=200); plt.close(fig)


def fig_V6():
    res = load("vessel_V6_gains"); names = list(res.keys())
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for n in names:
        s = res[n]["summary"]
        ax.errorbar(s["ttf"]["mean"], s["collision_rate"]["mean"], xerr=[[s["ttf"]["mean"] - s["ttf"]["lo"]], [s["ttf"]["hi"] - s["ttf"]["mean"]]],
                    yerr=[[s["collision_rate"]["mean"] - s["collision_rate"]["lo"]], [s["collision_rate"]["hi"] - s["collision_rate"]["mean"]]],
                    fmt="o", color=COL["std"], ms=6, capsize=2, lw=1.2)
        off = {"SCBF-MPPI a1=0.1 a2=1.0": (6, -14), "SCBF-MPPI a1=0.05 a2=0.2": (-70, 8)}.get(n, (6, 6))
        ax.annotate(n.replace("SCBF-MPPI ", "").replace("a1=", "α₁=").replace(" a2=", ", α₂="), (s["ttf"]["mean"], s["collision_rate"]["mean"]), xytext=off, textcoords="offset points", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    ax.set_xlim(70, 158); ax.set_xlabel("time to goal, s"); ax.set_ylabel("collision rate"); ax.set_title("Barrier gains α₁ (closing-speed limit) and α₂ (decay of ψ₁)", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_V6_gains.png"), dpi=200); plt.close(fig)


def fig_V7():
    res = load("vessel_V7_delta"); names = list(res.keys())
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), sharey=True)
    dotci(axes[0], names, res, "collision_rate", "collision rate")
    dotci(axes[1], names, res, "ttf", "time to goal, s", fmt="{:.0f}")
    dotci(axes[2], names, res, "var_ratio_mean", "Var kept along the row, fraction of Σ₀", fmt="{:.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_V7_delta.png"), dpi=200); plt.close(fig)


def fig_V8():
    res = load("vessel_V8_det_sweep"); names = list(res.keys())
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    markers = ["o", "s", "^", "v", "D", "P", "X"]
    for n, mk in zip(names, markers):
        s = res[n]["summary"]
        ax.errorbar(s["ttf"]["mean"], s["collision_rate"]["mean"], xerr=[[s["ttf"]["mean"] - s["ttf"]["lo"]], [s["ttf"]["hi"] - s["ttf"]["mean"]]],
                    yerr=[[s["collision_rate"]["mean"] - s["collision_rate"]["lo"]], [s["collision_rate"]["hi"] - s["collision_rate"]["mean"]]],
                    fmt=mk, color=COL["det"], ms=7, capsize=2, lw=1.0, alpha=0.9, label=n.replace("Det. MPPI ", "").replace("lam0=", "λ₀=").replace(".0 ", " ").replace("nu=", "ν="))
    v2 = load("vessel_V2_harbour")
    for key, lab, col in [("MPPI K=500", "MPPI K=500", COL["mppi"]), ("SCBF-MPPI (2nd-order barrier)", "SCBF-MPPI", COL["std"])]:
        s = v2[key]["summary"]; ax.plot(s["ttf"]["mean"], s["collision_rate"]["mean"], "*", color=col, ms=12, label=lab)
    ax.set_xlabel("time to goal, s (150 = not reached)"); ax.set_ylabel("collision rate")
    ax.set_title("Deterministic MPPI (Homburger et al.), 500 rollouts per cycle: λ₀, ν and schedule", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper center", ncol=2, frameon=True, framealpha=1, edgecolor="none")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_V8_det.png"), dpi=200); plt.close(fig)


def summary_text():
    lines = []
    for fn in ["vessel_V1_printed", "vessel_V2_harbour", "vessel_V3_disturbance", "vessel_V4_crossing", "vessel_V6_gains", "vessel_V7_delta", "vessel_V8_det_sweep"]:
        if not os.path.exists(os.path.join(RES, fn + ".json")):
            continue
        res = load(fn); lines.append(f"== {fn}")
        for n, d in res.items():
            if n.startswith("_"):
                lines.append(f"  {n}: {d}"); continue
            s = d["summary"]
            l = (f"  {n:<42} n={s['n']:2d} coll {s['collision_rate']['mean']:.3f} [{s['collision_rate']['lo']:.3f},{s['collision_rate']['hi']:.3f}] (ctrl-instants {s['collision_rate_ctrl']['mean']:.3f}) "
                 f"runs-touching {s['collided_frac']:.2f} ttf {s['ttf']['mean']:.0f} [{s['ttf']['lo']:.0f},{s['ttf']['hi']:.0f}] reached {s['reached_frac']:.2f} "
                 f"min_h {s['min_h']['mean']:.1f} ess mean {s['ess_mean']['mean']:.0f} median {s['ess_median']['mean']:.0f} obstacle-field {s['ess_obstacle_mean']['mean']:.0f}")
            if "activation_frac_mean" in s:
                l += (f" | act {s['activation_frac_mean']['mean']:.2f} multi {s['multi_violation_frac_mean']['mean']:.2f} var {s['var_ratio_mean']['mean']:.2f} "
                      f"sat {s['sat_active_mean']['mean']:.3f} infeas {s['infeasible_frac_mean']['mean']:.3f} clip {s['clip_frac_mean']['mean']:.2f}")
            lines.append(l)
    for fn in ["vessel_V0_solver", "vessel_V5_wallclock"]:
        if os.path.exists(os.path.join(RES, fn + ".json")):
            lines.append(f"== {fn}"); lines.append("  " + json.dumps(load(fn)))
    open(os.path.join(RES, "vessel_summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    for f, fn in [(fig_V1, "vessel_V1_printed"), (fig_V2, "vessel_V2_harbour"), (fig_V3, "vessel_V3_disturbance"),
                  (fig_V4, "vessel_V4_crossing"), (fig_V5, "vessel_V5_wallclock"), (fig_V6, "vessel_V6_gains"), (fig_V7, "vessel_V7_delta"), (fig_V8, "vessel_V8_det_sweep")]:
        if os.path.exists(os.path.join(RES, fn + ".json")):
            f(); print("wrote", f.__name__)
    summary_text()


if __name__ == "__main__":
    main()
