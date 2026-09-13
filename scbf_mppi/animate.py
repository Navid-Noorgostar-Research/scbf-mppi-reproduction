"""Side-by-side animation on the same seed.
    python -m scbf_mppi.animate                     -> figures/anim_mppi_vs_scbf.mp4 / .gif / anim_last_frame.png
                                                       MPPI  |  SCBF-MPPI as printed (variance form)          [unchanged]
    python -m scbf_mppi.animate --three [--seed N]  -> figures/anim_corridor_three.mp4 / .gif / anim_corridor_three_last.png
                                                       MPPI  |  as printed (variance form)  |  as CLAIMED (corrected, std form)
The third panel is that corrected form as a moving picture: with the constraint the paper says it enforces (margin z·σ, 1−δ = 0.997)
the objective in (8) shrinks the variance before it moves the mean, the speed channel keeps ~8 % of Σ₀ and the robot
stalls — over the 30 seeds of E1 ('SCBF-std K=500') 12 of 30 reach the goal, against 22 of 30 as printed and 30 of 30
for MPPI.  The panel titles carry those counts and the caption names the seed, so the single run is disclosed.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
try:    # the laptop may have no ffmpeg on PATH; imageio-ffmpeg ships one with libx264 (optional dependency)
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib import animation
from .env import Corridor
from .mppi import MPPI, SCBFMPPI
from .simulate import run_episode
from .experiments import FIG, RES, DEFAULT

COL = {"mppi": "#1B8AA6", "var": "#C2482A", "std": "#5B4FB0", "ink": "#20303C", "muted": "#5D7180"}
FFMPEG_ARGS = ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"]   # even frame size; plays in PowerPoint
E1_KEYS = ("MPPI K=500", "SCBF-var K=500", "SCBF-std K=500")                   # the 30-seed rows behind the three panels


def _e1_reach():
    """(reached, n) per panel from results/E1_table1.json, or None if the file is not there."""
    try:
        d = json.load(open(os.path.join(RES, "E1_table1.json")))
        return [(int(sum(r["reached"] for r in d[k]["runs"])), len(d[k]["runs"])) for k in E1_KEYS]
    except Exception:
        return None


def main(seed=1, K=500, n_show=60, fps=12, gif_every=2, three=False):
    env = Corridor(); s0, som, sg = DEFAULT["sigma_v"], DEFAULT["sigma_om"], DEFAULT["sigma_env"]
    specs = [("mppi", "MPPI, K = 500", COL["mppi"], {}),
             ("scbf", "SCBF-MPPI as printed" + (" (variance form)" if three else "") + ", K = 500", COL["var"], dict(form="variance"))]
    if three:
        specs.append(("scbf", "SCBF-MPPI as claimed (corrected, std form)", COL["std"], dict(form="std")))
    runs = []
    for kind, name, _, kw in specs:
        c = MPPI(env, K=K, seed=seed, sigma_v=s0, sigma_om=som) if kind == "mppi" else \
            SCBFMPPI(env, K=K, seed=seed, sigma_env=sg, sigma_v=s0, sigma_om=som, **kw)
        r = run_episode(c, env, sigma=sg, seed=seed, record_samples=True, bridge=False)
        runs.append(r)
        vr = r.get("var_ratio_mean")
        print(f"{name}: steps {r['steps']} reached {r['reached']} collision {r['collision_rate']:.3f} min_h {r['min_h']:.2f}"
              + (f" Var[dv] kept {100*vr:.0f} %" if vr is not None else ""), file=sys.stderr)
    reach = _e1_reach() if three else None
    n_run = max(r["steps"] for r in runs)
    hold = 2 * fps if three else 0                        # hold the last frame so 'time limit' can be read
    n_frames = n_run + hold
    if three:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
        fig.subplots_adjust(left=0.01, right=0.99, top=0.885, bottom=0.225, wspace=0.04)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    xs, lo, hi = env.walls()
    arts = []
    for pi, (ax, r, (kind, name, col, _)) in enumerate(zip(axes, runs, specs)):
        ax.plot(xs, lo, color=COL["ink"], lw=1.3); ax.plot(xs, hi, color=COL["ink"], lw=1.3)
        ax.plot(4, 0.5, "x", color=COL["ink"], ms=10, mew=2)
        ax.set_aspect("equal"); ax.set_xlim(-0.3, 4.4); ax.set_ylim(-1.15, 2.2)
        title = name
        if reach is not None:
            title += f"  —  {reach[pi][0]} of {reach[pi][1]} reach the goal"
        ax.set_title(title, loc="left", fontsize=10.5 if three else 11, color=col if three else COL["ink"])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        samp = [ax.plot([], [], color=col, lw=0.6, alpha=0.18)[0] for _ in range(n_show)]
        path, = ax.plot([], [], color=col, lw=2.0)
        robot, = ax.plot([], [], "o", color=COL["ink"], ms=6)
        txt = ax.text(0.0, -0.02, "", transform=ax.transAxes, fontsize=8.8 if three else 9, color=COL["ink"], va="top", ha="left", linespacing=1.35)
        arts.append((samp, path, robot, txt, r))
    if three:
        cap = (f"σ = 0.1 (the level at which the paper's Table I reproduces), Δt = 0.05 s, T = 20, λ = 1, K = 500, 1−δ = 0.997 · seed {seed}, the same seed and process noise "
               "in all three panels · faint lines: 60 of the 500 rollouts of the current cycle · the counts in the titles are the 30 seeds of E1\n"
               "Right: the constraint the paper claims to enforce — margin z·σ where the printed (8) has α·σ². A binding constraint is cheapest to satisfy by shrinking the "
               "variance, not by moving the mean, so the speed channel keeps 8 % of Σ₀ over 30 seeds and the robot cannot speed up.")
        fig.text(0.5, 0.008, cap, ha="center", va="bottom", fontsize=8.5, color=COL["muted"], linespacing=1.4)
    else:
        fig.text(0.5, 0.01, "σ = 0.1 (the level at which the paper's Table I reproduces), Δt = 0.05 s, horizon T = 20, λ = 1. Faint lines: 60 of the 500 sampled rollouts of the current cycle.",
                 ha="center", fontsize=8.5, color=COL["muted"])
        fig.tight_layout(rect=(0, 0.06, 1, 1))

    def update(k):
        out = []
        for samp, path, robot, txt, r in arts:
            i = min(k, r["steps"] - 1)
            S = r["samples"][i]
            idx = np.linspace(0, S.shape[0] - 1, n_show).astype(int)
            for line, j in zip(samp, idx):
                line.set_data(S[j, :, 0], S[j, :, 1])
            path.set_data(r["traj"][:i + 2, 0], r["traj"][:i + 2, 1])
            robot.set_data([r["traj"][i + 1, 0]], [r["traj"][i + 1, 1]])
            hmin = r["hmin_series"][i + 1]
            end = "" if i + 1 < r["steps"] else ("goal reached" if r["reached"] else "TIME LIMIT — goal not reached")
            if three:
                l1 = f"step {i+1:3d}   min h {hmin:+.2f}   ESS {r['ess_series'][i]:.0f}" + ("   ·  COLLISION" if hmin < 0 else "")
                parts = []
                if r.get("activation_frac_mean") is not None: parts.append(f"constraint active {100*r['activation_frac_mean']:.0f} %")
                if r.get("var_ratio_mean") is not None: parts.append(f"Var[δv] kept {100*r['var_ratio_mean']:.0f} %")
                if end: parts.append(end)
                txt.set_text(l1 + "\n" + "   ".join(parts))
            else:
                s = f"step {i+1:3d}   min h = {hmin:+.2f}   ESS = {r['ess_series'][i]:.0f}"
                if r.get("activation_frac_mean") is not None:
                    s += f"   constraint active {100*r['activation_frac_mean']:.0f} %"
                if hmin < 0: s += "   ·  COLLISION"
                if end: s += "   ·  " + end
                txt.set_text(s)
            out += samp + [path, robot, txt]
        return out

    stem = "anim_corridor_three" if three else "anim_mppi_vs_scbf"
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    mp4 = os.path.join(FIG, stem + ".mp4")
    try:
        ani.save(mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2400, extra_args=FFMPEG_ARGS), dpi=110)
        print("wrote", mp4)
    except Exception as e:
        print("mp4 failed:", e, file=sys.stderr)
    gif = os.path.join(FIG, stem + ".gif")
    ani2 = animation.FuncAnimation(fig, update, frames=range(0, n_frames, gif_every), interval=1000 / (fps / gif_every), blit=False)
    ani2.save(gif, writer=animation.PillowWriter(fps=fps / gif_every), dpi=70)
    print("wrote", gif)
    # a still image: last frame
    update(n_frames - 1); fig.savefig(os.path.join(FIG, ("anim_corridor_three_last" if three else "anim_last_frame") + ".png"), dpi=200)


if __name__ == "__main__":
    args = sys.argv[1:]
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 1
    main(seed=seed, three=("--three" in args))
