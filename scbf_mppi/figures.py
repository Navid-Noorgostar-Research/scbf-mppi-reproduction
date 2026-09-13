"""Figures from results/*.json.   python -m scbf_mppi.figures"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .env import Corridor
from .experiments import RES, FIG, load

# categorical palette (validated for CVD separation / contrast): fixed assignment, never cycled
COL = {"mppi": "#1B8AA6", "var": "#C2482A", "std": "#5B4FB0", "det": "#9C6A12", "ink": "#20303C", "muted": "#5D7180", "grid": "#D5DEE4"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": "#5D7180", "axes.labelcolor": "#20303C",
                     "xtick.color": "#20303C", "ytick.color": "#20303C", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#D5DEE4", "grid.linewidth": 0.6, "legend.frameon": False})

PAPER_TABLE_I = {"MPPI K=200": (0.0454, 140.4), "MPPI K=500": (0.0294, 129.6),
                 "SCBF-var K=200": (0.0, 163.6), "SCBF-var K=500": (0.0, 156.1)}

def colour_for(name):
    if name.startswith("MPPI"): return COL["mppi"]
    if "std" in name: return COL["std"]
    if "Deterministic" in name: return COL["det"]
    return COL["var"]

def dotci(ax, names, res, key, xlabel, paper=None, fmt="{:.3f}"):
    ys = np.arange(len(names))[::-1]
    for y, n in zip(ys, names):
        s = res[n]["summary"][key]
        ax.plot([s["lo"], s["hi"]], [y, y], color=colour_for(n), lw=2, solid_capstyle="round")
        ax.plot(s["mean"], y, "o", color=colour_for(n), ms=6)
        ax.text(s["hi"], y + 0.18, fmt.format(s["mean"]), fontsize=9, color=COL["ink"], ha="left", va="bottom")
        if paper and n in paper:
            ax.plot(paper[n], y, "D", mfc="white", mec=COL["ink"], ms=6, mew=1.2)
    ax.set_yticks(ys); ax.set_yticklabels(names); ax.set_xlabel(xlabel); ax.grid(axis="y", visible=False)

def fig_E1():
    res = load("E1_table1"); env = Corridor()
    names = [n for n in res if not n.startswith("_")]
    # trajectories
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    xs, lo, hi = env.walls(); ax.plot(xs, lo, color=COL["ink"], lw=1.2); ax.plot(xs, hi, color=COL["ink"], lw=1.2)
    for n, lab in [("MPPI K=500", "MPPI"), ("SCBF-var K=500", "SCBF-MPPI as printed"), ("SCBF-std K=500", "SCBF-MPPI corrected")]:
        t = np.array(res[n]["runs"][0]["traj_xy"]); ax.plot(t[:, 0], t[:, 1], color=colour_for(n), lw=1.6, label=lab)
    ax.plot(4, 0.5, "x", color=COL["ink"], ms=9, mew=2); ax.plot(0, 0.5, "o", color=COL["ink"], ms=5)
    ax.set_aspect("equal"); ax.set_xlim(-0.3, 4.4); ax.set_ylim(-1.15, 2.2); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="lower right", fontsize=9, framealpha=1, frameon=True, edgecolor="none"); ax.set_title("One run each (seed 0) · K = 500 · σ = 0.1 · Δt = 0.05 · T = 20", fontsize=10.5, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_E1_trajectories.png"), dpi=200); plt.close(fig)
    # table as dot-CI chart
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    dotci(axes[0], names, res, "collision_rate", "collision rate · mean and 95 % CI",
          paper={k: v[0] for k, v in PAPER_TABLE_I.items()})
    dotci(axes[1], names, res, "ttf", "time to finish, steps (250 = not reached)", paper={k: v[1] for k, v in PAPER_TABLE_I.items()}, fmt="{:.0f}")
    axes[0].plot([], [], "D", mfc="white", mec=COL["ink"], label="paper, Table I"); axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9.5, ncol=1)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_E1_table.png"), dpi=200); plt.close(fig)
    # summary text
    rows = []
    for n in names:
        s = res[n]["summary"]
        rows.append(f"{n:<18} coll {s['collision_rate']['mean']:.3f} [{s['collision_rate']['lo']:.3f},{s['collision_rate']['hi']:.3f}]  runs-with-collision {s['collided_frac']:.1f}  ttf {s['ttf']['mean']:.0f}  reached {s['reached_frac']:.1f}"
                    + (f"  act {s['activation_frac_mean']['mean']:.2f} varratio {s['var_ratio_mean']['mean']:.2f} sat_active {s['sat_active_mean']['mean']:.3f}" if 'activation_frac_mean' in s else "")
                    + f"  ess {s['ess_mean']['mean']:.0f}  min_h {s['min_h']['mean']:.2f}  bridge {s['bridge_crossings']['mean']:.1f}/{s['inside_pairs']['mean']:.0f}")
    open(os.path.join(RES, "E1_summary.txt"), "w").write("\n".join(rows)); print("\n".join(rows))

def fig_E2():
    res = load("E2_sigma_sweep"); ref = res["_free_diffusion_exit_by_130_steps"]
    sig = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for kind, lab, col in [("MPPI", "MPPI, K = 500", COL["mppi"]), ("SCBF-var", "SCBF-MPPI as printed, K = 500", COL["var"])]:
        m = [res[f"{kind} sigma={s}"]["summary"]["collision_rate"]["mean"] for s in sig]
        lo = [res[f"{kind} sigma={s}"]["summary"]["collision_rate"]["lo"] for s in sig]
        hi = [res[f"{kind} sigma={s}"]["summary"]["collision_rate"]["hi"] for s in sig]
        ax.fill_between(sig, lo, hi, color=col, alpha=0.15, lw=0); ax.plot(sig, m, "o-", color=col, lw=2, ms=5, label=lab)
    ax.axhspan(0.029, 0.046, color=COL["ink"], alpha=0.08, lw=0); ax.text(0.021, 0.055, "Table I, MPPI: 0.029–0.045", fontsize=8.5, color=COL["ink"])
    ax.axvline(1.0, color=COL["ink"], lw=1, ls=":"); ax.text(0.93, 0.62, "σ = I as stated", rotation=90, fontsize=8.5, color=COL["ink"], va="center", ha="right")
    ax.text(0.021, 0.86, "No control at all, start on the centre line — leaves C within 130 steps\nwith probability "
            + f"{ref['1.0']:.2f} at σ = 1 · {ref['0.3']:.2f} at σ = 0.3 · {ref['0.1']:.2f} at σ = 0.1 · {ref['0.05']:.2f} at σ = 0.05", fontsize=8.3, color=COL["muted"], va="top")
    ax.set_xscale("log"); ax.set_xlabel("process-noise level σ  (paper: 'σ is the identity matrix')"); ax.set_ylabel("collision rate")
    ax.set_ylim(-0.02, 0.9); ax.legend(fontsize=8, loc="center left"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_E2_sigma.png"), dpi=200); plt.close(fig)

def fig_E3():
    e = load("E3_wallclock")
    items = [("MPPI, K = 500", e["mppi_cycle_K500_s"], COL["mppi"]), ("MPPI, K = 5000", e["mppi_cycle_K5000_s"], COL["mppi"]),
             ("SCBF-MPPI, K = 500, closed-form per-sample solve (this code)", e["scbf_mppi_cycle_K500_closed_form_s"], COL["var"]),
             ("Algorithm 1, K = 200: 4,000 SDPs (8) per cycle", e["algorithm1_cycle_K200_T20_with_literal_SDP_s"], COL["var"]),
             ("Algorithm 1, K = 500: 10,000 SDPs, solver time only", e["algorithm1_cycle_K500_T20_solver_only_s"], COL["var"]),
             ("Algorithm 1, K = 500: 10,000 SDPs (8) through cvxpy", e["algorithm1_cycle_K500_T20_with_literal_SDP_s"], COL["var"]),
             ("Algorithm 1, K = 500, as 10,000 SOCPs (corrected form)", e["algorithm1_cycle_K500_T20_with_SOCP_s"], COL["std"])]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ys = np.arange(len(items))[::-1]
    for y, (lab, v, col) in zip(ys, items):
        ax.barh(y, v, color=col, height=0.6); ax.text(v * 1.15, y, f"{v*1000:.1f} ms" if v < 1 else f"{v:.0f} s", va="center", fontsize=9, color=COL["ink"])
    ax.set_yticks(ys); ax.set_yticklabels([i[0] for i in items], fontsize=8.5); ax.set_xscale("log"); ax.set_xlim(1e-3, 1e3)
    ax.set_xlabel("seconds per control cycle (T = 20, one core)"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_E3_wallclock.png"), dpi=200); plt.close(fig)

def fig_E4():
    e = load("E4_theorem2"); e1 = load("E1_table1")
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    s = np.array(e["s_proj"]); ax.plot(s, e["pr_printed"], color=COL["var"], lw=2.2, label="printed constraint (6)/(8): A·μ − α·AΣAᵀ ≥ b  →  Pr = Φ(α·s)")
    ax.plot(s, e["pr_correct"], color=COL["std"], lw=2.2, label="chance constraint it claims: A·μ − z·√(AΣAᵀ) ≥ b  →  Pr = 1 − δ")
    ax.axvline(1.0, color=COL["muted"], lw=1, ls=":"); ax.text(1.03, 0.56, "s = 1: the two agree", fontsize=8.5, color=COL["muted"])
    sv = e1["SCBF-var K=500"]["summary"]["sat_active_mean"]["mean"]; ss = e1["SCBF-std K=500"]["summary"]["sat_active_mean"]["mean"]
    ax.axhline(sv, color=COL["var"], lw=1, ls="--"); ax.text(1.05, sv + 0.02, f"measured in the loop, printed form: {sv:.2f}", fontsize=9, color=COL["var"])
    ax.axhline(ss, color=COL["std"], lw=1, ls="--"); ax.text(0.05, 0.905, f"measured in the loop, corrected form: {ss:.3f}", fontsize=9, color=COL["std"])
    ax.set_xlabel("projected standard deviation of the sampled control,  s = √(A Σ Aᵀ)"); ax.set_ylabel("Pr(A u ≥ b) actually delivered")
    ax.set_ylim(0.3, 1.02); ax.set_xlim(0, 2.0); ax.legend(fontsize=8.5, loc="lower right", frameon=True, framealpha=1, edgecolor="none")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_E4_theorem2.png"), dpi=200); plt.close(fig)

def fig_E5():
    res = load("E1_table1")
    bins = res["MPPI K=500"]["runs"][0]["h_bins"]; labels = ["outside", "0–0.05", "0.05–0.1", "0.1–0.2", "0.2–0.3", "0.3–0.5", "> 0.5"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for n, lab in [("MPPI K=500", "MPPI"), ("SCBF-var K=500", "SCBF-MPPI as printed"), ("SCBF-std K=500", "SCBF-MPPI corrected")]:
        for ax, key in zip(axes, ("v_by_h", "om_by_h")):
            arr = np.array([[np.nan if v is None else v for v in r[key]] for r in res[n]["runs"]], float)
            m = np.nanmean(arr, 0)
            ax.plot(range(len(labels)), m, "o-", color=colour_for(n), lw=2, ms=5, label=lab)
    axes[0].set_ylabel("|v| applied (m/s)"); axes[1].set_ylabel("|ω| applied (rad/s)")
    for ax in axes:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.5); ax.set_xlabel("distance to the nearest wall, min(h₁, h₂)")
    axes[0].legend(fontsize=8.5); axes[0].set_title("speed drops as the wall approaches", fontsize=10, loc="left"); axes[1].set_title("steering does not respond", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_E5_braking.png"), dpi=200); plt.close(fig)

def fig_E7():
    res = load("E1_table1")
    names = ["MPPI K=500", "MPPI K=5000", "SCBF-var K=200", "SCBF-var K=500", "SCBF-std K=500"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ys = np.arange(len(names))[::-1]
    for y, n in zip(ys, names):
        s = res[n]["summary"]
        axes[0].barh(y, s["bridge_crossings"]["mean"], color=colour_for(n), height=0.6)
        axes[0].text(s["bridge_crossings"]["mean"] + 0.15, y, f"{s['bridge_crossings']['mean']:.1f} of {s['inside_pairs']['mean']:.0f} intervals · rate {s['collision_rate']['mean']:.3f}", va="center", fontsize=9.5)
    axes[0].set_yticks(ys); axes[0].set_yticklabels(names, fontsize=10); axes[0].set_xlabel("between-sample excursions per run (bridge, 20 sub-steps)"); axes[0].grid(axis="y", visible=False)
    axes[0].set_xlim(0, max(res[n]["summary"]["bridge_crossings"]["mean"] for n in names) * 3.2)
    d = np.linspace(0, 0.5, 200)
    for sg, col in [(1.0, COL["ink"]), (0.3, COL["muted"]), (0.1, COL["mppi"])]:
        axes[1].plot(d, np.exp(-2 * d ** 2 / (sg ** 2 * 0.05)), lw=2, color=col, label=f"σ = {sg}")
    axes[1].set_xlabel("distance d of both samples from the wall"); axes[1].set_ylabel("P(cross and return)"); axes[1].legend(fontsize=9.5)
    axes[1].set_title("exp(−2d²/(σ²Δt)), Δt = 0.05", fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_E7_bridge.png"), dpi=200); plt.close(fig)

def fig_E8():
    res = load("E8_mechanisms"); names = [n for n in res if not n.startswith("_")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    dotci(axes[0], names, res, "collision_rate", f"collision rate, mean and 95 % CI ({res[names[0]]['summary']['collision_rate']['n']} runs)")
    dotci(axes[1], names, res, "ttf", "time to finish (steps; 250 = not reached)", fmt="{:.0f}")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_E8_mechanisms.png"), dpi=200); plt.close(fig)
    rows = []
    for n in names:
        s = res[n]["summary"]
        rows.append(f"{n:<40} coll {s['collision_rate']['mean']:.3f} [{s['collision_rate']['lo']:.3f},{s['collision_rate']['hi']:.3f}] collided-runs {s['collided_frac']:.1f} ttf {s['ttf']['mean']:.0f} reached {s['reached_frac']:.1f} ess {s['ess_mean']['mean']:.0f} min_h {s['min_h']['mean']:.2f}"
                    + (f" act {s['activation_frac_mean']['mean']:.2f} varratio {s['var_ratio_mean']['mean']:.2f}" if 'activation_frac_mean' in s else ""))
    open(os.path.join(RES, "E8_summary.txt"), "w").write("\n".join(rows)); print("\n".join(rows))

def fig_E11():
    res = load("E11_delta_sweep"); names = [n for n in res if not n.startswith("_")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    dotci(axes[0], names, res, "collision_rate", "collision rate")
    dotci(axes[1], names, res, "ttf", "time to finish (steps)", fmt="{:.0f}")
    dotci(axes[2], names, res, "var_ratio_mean", "Var[δv] / Var₀ left", fmt="{:.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_E11_delta.png"), dpi=200); plt.close(fig)
    rows = []
    for n in names:
        s = res[n]["summary"]
        rows.append(f"{n:<32} coll {s['collision_rate']['mean']:.3f} ttf {s['ttf']['mean']:.0f} reached {s['reached_frac']:.1f} act {s['activation_frac_mean']['mean']:.2f} varratio {s['var_ratio_mean']['mean']:.2f} sat_active {s['sat_active_mean']['mean']:.3f} ess {s['ess_mean']['mean']:.0f} min_h {s['min_h']['mean']:.2f}")
    open(os.path.join(RES, "E11_summary.txt"), "w").write("\n".join(rows)); print("\n".join(rows))

def fig_E9():
    e = load("E9_clark")
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    T = np.array(e["T"]); Tf = np.logspace(0, np.log10(500), 200)
    from scipy.stats import norm
    ax.plot(Tf, 2 * norm.cdf(-1.0 / np.sqrt(Tf)), color=COL["ink"], lw=2, label="exact: 2Φ(−x₀/√T)")
    ax.plot(T, e["mc_hit_frac"], "o", color=COL["var"], ms=6, label="Monte Carlo, 20,000 paths")
    ax.set_xscale("log"); ax.set_xlabel("horizon T"); ax.set_ylabel("P(h hits 0 by T)  ·  x₀ = 1"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("(3) holds for every x > 0 — the set is still left, w.p. → 1", fontsize=10.5, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_E9_clark.png"), dpi=200); plt.close(fig)

def fig_E12():
    res = load("E12_printed_barriers"); env = Corridor(printed=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    xs, lo, hi = env.walls(); ax.plot(xs, lo, color=COL["ink"], lw=1.2); ax.plot(xs, hi, color=COL["ink"], lw=1.2)
    t = np.array(res["SCBF-var K=500, printed barriers"]["runs"][0]["traj_xy"]); ax.plot(t[:, 0], t[:, 1], color=COL["var"], lw=1.4, label="SCBF-MPPI, printed barriers h = y − sin x, sin x + 1 − y")
    ax.plot(4, 0.5, "x", color=COL["ink"], ms=9, mew=2); ax.text(3.55, 0.62, f"goal: h₂ = {res['_goal_h_printed']['h2']:.3f}", fontsize=8.5, color=COL["var"])
    ax.set_aspect("equal"); ax.set_xlim(-0.3, 4.4); ax.set_ylim(-1.2, 2.2); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_E12_printed.png"), dpi=200); plt.close(fig)

def main():
    have = lambda n: os.path.exists(os.path.join(RES, n + ".json"))
    if have("E1_table1"): fig_E1(); fig_E5(); fig_E7()
    if have("E2_sigma_sweep"): fig_E2()
    if have("E3_wallclock"): fig_E3()
    if have("E4_theorem2") and have("E1_table1"): fig_E4()
    if have("E8_mechanisms"): fig_E8()
    if have("E11_delta_sweep"): fig_E11()
    if have("E9_clark"): fig_E9()
    if have("E12_printed_barriers"): fig_E12()

if __name__ == "__main__":
    main()
