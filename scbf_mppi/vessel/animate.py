"""Side-by-side animation on the Solgenia model: MPPI vs SCBF-MPPI (second-order barrier), same seed, crossing
ferry scenario, with the sampled rollout fan of every cycle drawn (60 of the 500 samples, weight-shaded).
    python -m scbf_mppi.vessel.animate            -> figures/anim_vessel_mppi_vs_scbf.mp4 and .gif
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Polygon
from . import solgenia as sg, harbour as hb
from .controllers import VesselMPPI, VesselSCBFMPPI
from .simulate import run_episode
from .experiments import FIG, DEFAULT

COL = {"mppi": "#1B8AA6", "std": "#5B4FB0", "ink": "#20303C", "muted": "#5D7180", "water": "#EAF2F6", "obst": "#8A9BA8"}


def main(seed=2, n_show=60, fps=6, gif_every=1, scenario="crossing"):
    H = hb.harbour_crossing() if scenario == "crossing" else hb.harbour_static()
    kw = dict(K=DEFAULT["K"], T=DEFAULT["T"], lam=DEFAULT["lam"], s0=DEFAULT["s0"], seed=seed)
    runs = []
    for kind in ("mppi", "scbf"):
        c = VesselMPPI(H, **kw) if kind == "mppi" else VesselSCBFMPPI(H, **kw)
        runs.append(run_episode(c, H, seed=seed, disturbance="white", s_F=np.asarray(DEFAULT["s_F"]), record_samples=True))
        print(kind, "steps", runs[-1]["steps"], "collision", round(runs[-1]["collision_rate"], 3), "min_h", round(runs[-1]["min_h"], 1), file=sys.stderr)
    n_frames = max(r["steps"] for r in runs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    arts = []
    ferry = H.obs[-1] if scenario == "crossing" else None
    for ax, r, name, col in zip(axes, runs, ("MPPI, K = 500", "SCBF-MPPI, second-order barrier, K = 500"), (COL["mppi"], COL["std"])):
        ax.set_facecolor(COL["water"])
        for o in H.obs:
            if np.linalg.norm(o.w) == 0:
                ax.add_patch(Circle(o.center(0), o.R, facecolor=COL["obst"], edgecolor=COL["ink"], lw=0.8, alpha=0.5))
                ax.add_patch(Circle(o.center(0), max(o.R - H.r_ego, 0.5), facecolor=COL["ink"], edgecolor="none", alpha=0.35))
        ax.plot(H.goal[0], H.goal[1], "x", color=COL["ink"], ms=10, mew=2)
        ax.set_aspect("equal"); ax.set_xlim(-10, 130); ax.set_ylim(-60, 40)
        ax.set_title(name, loc="left", fontsize=11, color=COL["ink"]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        samp = [ax.plot([], [], color=col, lw=0.6, alpha=0.15)[0] for _ in range(n_show)]
        path, = ax.plot([], [], color=col, lw=2.0)
        hull = Polygon(np.zeros((5, 2)), closed=True, facecolor=col, edgecolor=COL["ink"], lw=0.8); ax.add_patch(hull)
        fpatch = Circle((0, -1000), ferry.R, facecolor=COL["obst"], edgecolor=COL["ink"], lw=0.8, alpha=0.5) if ferry else None
        fcore = Circle((0, -1000), ferry.R - H.r_ego, facecolor=COL["ink"], edgecolor="none", alpha=0.35) if ferry else None
        if ferry: ax.add_patch(fpatch); ax.add_patch(fcore)
        flabel = ax.text(0, -1000, "ferry, 3 m/s", fontsize=8, color=COL["ink"], ha="center", va="center") if ferry else None
        txt = ax.text(0.0, -0.02, "", transform=ax.transAxes, fontsize=9, color=COL["ink"], va="top", ha="left")
        arts.append((samp, path, hull, fpatch, fcore, txt, r, flabel))
    fig.text(0.5, 0.01, "faint lines: 60 of the 500 sampled rollouts of the current cycle, opacity ∝ weight",
             ha="center", fontsize=10, color=COL["ink"])

    def update(f):
        out = []
        for samp, path, hull, fpatch, fcore, txt, r, flabel in arts:
            k = min(f, r["steps"] - 1)
            tr = r["traj"]
            path.set_data(tr[:k + 1, 0], tr[:k + 1, 1])
            hull.set_xy(sg.hull_polygon(tr[k]))
            if fpatch is not None:
                c = ferry.center(k * sg.DT_CTRL); fpatch.center = c; fcore.center = c
                if flabel is not None: flabel.set_position((c[0], c[1]))
            kk, S, w = r["samples"][k]
            idx = np.argsort(w)[::-1][:n_show]
            wmax = max(float(w[idx].max()), 1e-12)
            for line, i in zip(samp, idx):
                line.set_data(S[i, :, 0], S[i, :, 1]); line.set_alpha(0.06 + 0.5 * float(w[i]) / wmax)
            hm = r["hmin_series"][k]
            status = f"t = {k * sg.DT_CTRL:.0f} s   speed {abs(tr[k, 3]):.1f} m/s   min h {hm:+.1f} m   collisions so far {int((r['hmin_series'][:k + 1] < 0).sum())}   ESS {r['ess_series'][k]:.0f}"
            txt.set_text(status)
            out += samp + [path, hull, txt]
        return out

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    mp4 = os.path.join(FIG, "anim_vessel_mppi_vs_scbf.mp4"); gif = os.path.join(FIG, "anim_vessel_mppi_vs_scbf.gif")
    try:
        anim.save(mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=1800), dpi=110)
        print("wrote", mp4, file=sys.stderr)
    except Exception as e:
        print("mp4 failed:", e, file=sys.stderr)
    anim2 = animation.FuncAnimation(fig, update, frames=range(0, n_frames, gif_every), interval=1000 / fps * gif_every, blit=False)
    anim2.save(gif, writer=animation.PillowWriter(fps=max(1, fps // gif_every)), dpi=72)
    print("wrote", gif, file=sys.stderr)
    update(n_frames - 1); fig.savefig(os.path.join(FIG, "anim_vessel_last_frame.png"), dpi=150)


if __name__ == "__main__":
    main()
