"""Three boats under a coloured gust and a 0.5 m/s cross-current the controller does not know (experiment V3, seed 0):
MPPI · SCBF-MPPI (second-order barrier) · SCBF-MPPI + importance-sampling correction, same seed, same disturbance
realisation.  The water drifts, the gust force is drawn on the hull, the sampled rollout fan of every cycle is shown.
    python -m scbf_mppi.vessel.animate_current           -> figures/anim_vessel_current.mp4 / .gif / _last.png / _mid.png
    python -m scbf_mppi.vessel.animate_current --ghosts  -> figures/anim_vessel_current_ghosts.mp4 / .gif / _last.png / _mid.png
        the last three seconds hold the final frame and fade in all 30 seeds of V3 behind the played one — runs that
        touched a circle drawn darker.  Needs results/vessel_V3_trajectories.json
        (python -m scbf_mppi.vessel.trajectories_v3, ~5 min on 8 cores).
Both variants read the 30-seed statistics from results/vessel_V3_disturbance.json and print them in the panel titles
(runs touching a circle) and in the footer (runs reaching the goal, the paired ordering, and where the played seed
sits among the 30) — the single seed is disclosed, not called representative.  The readout shows the clearance now and
the closest approach so far, and the executed azimuth thrust against Solgenia's 700 N disk.
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
from matplotlib.patches import Circle, Polygon, FancyArrow
from . import solgenia as sg, harbour as hb
from .simulate import run_episode
from .experiments import FIG, RES, DEFAULT, make_controller

COL = {"mppi": "#1B8AA6", "scbf": "#5B4FB0", "is": "#2E7D4F", "ink": "#20303C", "muted": "#5D7180", "water": "#EAF2F6", "obst": "#8A9BA8", "gust": "#5D7180"}
CURRENT = (0.0, -0.5)
KEYS = ["MPPI", "SCBF-MPPI", "SCBF-MPPI + IS"]
SCEN = "gust + 0.5 m/s current"
FFMPEG_ARGS = ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"]   # even frame size; plays in PowerPoint


def main(seed=0, n_show=60, fps=8, ghosts=False):
    cfgs = [("MPPI", dict(kind="mppi"), COL["mppi"]),
            ("SCBF-MPPI, 2nd-order barrier", dict(kind="scbf"), COL["scbf"]),
            ("SCBF-MPPI + IS correction", dict(kind="scbf", is_correction=True), COL["is"])]
    H = hb.harbour_static()
    runs = []
    for name, cfg, _ in cfgs:
        c, _ = make_controller(dict(cfg, disturbance="ou", current=CURRENT), seed)
        r = run_episode(c, H, seed=seed, disturbance="ou", s_F=np.asarray(DEFAULT["s_F"]), tau_c=DEFAULT["tau_c"],
                        current=CURRENT, record_samples=True)
        runs.append(r)
        print(name, "steps", r["steps"], "collision", round(r["collision_rate"], 3), "min_h", round(r["min_h"], 2), "reached", r["reached"], "ttf", r["ttf"], file=sys.stderr)
    # the 30-seed statistics (always available) and the cross-check of this seed against the stored V3 run
    stats = None
    try:
        V3 = json.load(open(os.path.join(RES, "vessel_V3_disturbance.json")))
        R = [V3[f"{k} | {SCEN}"]["runs"] for k in KEYS]
        n = len(R[0])
        mine = [next(x for x in rr if x["seed"] == seed) for rr in R]
        stats = dict(n=n,
                     touched=[int(sum(x["collided"] for x in rr)) for rr in R],
                     reached=[int(sum(x["reached"] for x in rr)) for rr in R],
                     rank=[sorted((x["min_h"] for x in rr), reverse=True).index(m["min_h"]) + 1 for rr, m in zip(R, mine)],
                     order=int(sum(1 for i in range(n) if R[0][i]["min_h"] < R[1][i]["min_h"] < R[2][i]["min_h"])))
        for key, r, m in zip(KEYS, runs, mine):
            print(f"   V3 stored: {key}: min_h {m['min_h']:.2f} ttf {m['ttf']:.0f} | this run: min_h {r['min_h']:.2f} ttf {r['ttf']:.0f}", file=sys.stderr)
        print(f"   30 seeds: touched {stats['touched']} reached {stats['reached']} | seed {seed} rank by closest approach {stats['rank']} | MPPI<barrier<+IS in {stats['order']}/{n}", file=sys.stderr)
    except Exception as e:  # pragma: no cover
        print("V3 statistics skipped:", e, file=sys.stderr)
    # the 30-seed ghost paths, shown in the final hold
    ghost_runs = None
    if ghosts:
        gpath = os.path.join(RES, "vessel_V3_trajectories.json")
        try:
            G = json.load(open(gpath)); ghost_runs = [G["configs"][k]["runs"] for k in KEYS]
            print(f"ghosts: {len(ghost_runs[0])} seeds per controller from {gpath}", file=sys.stderr)
        except Exception as e:
            print(f"ghosts skipped — {gpath} not readable ({e}); run python -m scbf_mppi.vessel.trajectories_v3", file=sys.stderr)
    n_run = max(r["steps"] for r in runs)
    hold = 3 * fps if ghost_runs is not None else 0            # three seconds on the last frame with the ghosts
    n_frames = n_run + hold
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.215, wspace=0.03)
    arts = []
    xmin, xmax, ymin, ymax = -10, 130, -42, 40
    spacing = 9.0
    gxs = np.arange(xmin - spacing, xmax + spacing, spacing); gys = np.arange(ymin - spacing, ymax + spacing, spacing)
    GX, GY = np.meshgrid(gxs, gys)
    rng0 = np.random.default_rng(3); keep = rng0.random(GX.shape) > 0.35; jit = (rng0.random(GX.shape + (2,)) - 0.5) * spacing * 0.6
    cur = np.asarray(CURRENT); spd = np.linalg.norm(cur); ux, uy = cur / spd
    for pi, (ax, r, (name, _, col)) in enumerate(zip(axes, runs, cfgs)):
        ax.set_facecolor(COL["water"])
        for o in H.obs:
            ax.add_patch(Circle(o.center(0), o.R, facecolor=COL["obst"], edgecolor=COL["ink"], lw=0.8, alpha=0.5))
            ax.add_patch(Circle(o.center(0), max(o.R - H.r_ego, 0.5), facecolor=COL["ink"], edgecolor="none", alpha=0.35))
            ax.text(o.center(0)[0], o.center(0)[1] + o.R + 2.5, o.name, fontsize=8, color=COL["ink"], ha="center", va="bottom")
        ax.plot(H.goal[0], H.goal[1], "x", color=COL["ink"], ms=10, mew=2)
        ax.set_aspect("equal"); ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        title = name if stats is None else f"{name} — {stats['touched'][pi]} of {stats['n']} seeds touch a circle"
        ax.set_title(title, loc="left", fontsize=10.5, color=col); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        # the drifting water: streaks advected by the current (redrawn every frame with a periodic offset)
        streaks = ax.quiver(GX[keep] + jit[keep][:, 0], GY[keep] + jit[keep][:, 1], np.full(keep.sum(), ux), np.full(keep.sum(), uy),
                            color=COL["ink"], alpha=0.15, scale=40, width=0.003, headwidth=3.5, headlength=4, pivot="mid")
        # the ghosts: every seed's path, invisible until the final hold, under the fan and the played path
        gl = []
        if ghost_runs is not None:
            for gr in ghost_runs[pi]:
                xy = np.asarray(gr["traj_xy"], float)
                ln, = ax.plot(xy[:, 0], xy[:, 1], color=col, lw=0.9, alpha=0.0, zorder=1.9, solid_capstyle="round")
                gl.append((ln, bool(gr["collided"])))
        gtxt = ax.text(0.99, 0.975, "", transform=ax.transAxes, fontsize=9.5, color=col, ha="right", va="top", fontweight="bold", alpha=0.0)
        samp = [ax.plot([], [], color=col, lw=0.6, alpha=0.15)[0] for _ in range(n_show)]
        path, = ax.plot([], [], color=col, lw=2.0)
        hull = Polygon(np.zeros((5, 2)), closed=True, facecolor=col, edgecolor=COL["ink"], lw=0.8); ax.add_patch(hull)
        gust = ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=COL["gust"], lw=2.2, alpha=0.9))
        glabel = ax.text(0, 0, "", fontsize=8, color=COL["ink"], ha="left", va="center")
        txt = ax.text(0.0, -0.03, "", transform=ax.transAxes, fontsize=9.2, color=COL["ink"], va="top", ha="left", linespacing=1.35)
        arts.append((samp, path, hull, gust, glabel, txt, r, streaks, gl, gtxt))
    fig.text(0.5, 0.955, f"Solgenia · moored obstacles · coloured gust (OU, 5 s) + 0.5 m/s current towards the south that the controller's model does not know · seed {seed} · same disturbance realisation for all three",
             ha="center", fontsize=10.5, color=COL["ink"])
    fig.text(0.5, 0.098, "faint lines: 60 of the 500 sampled rollouts of the current cycle, opacity ∝ weight · arrow on the hull: the gust force (100 N = 2.5 m) · grey arrows: the water, drifting at 0.5 m/s · thrust: executed azimuth force against the 700 N disk",
             ha="center", fontsize=8.8, color=COL["ink"])
    if stats is not None:
        t, g, n = stats["touched"], stats["reached"], stats["n"]
        foot = (f"seed {seed} plays — the first seed, not a selected one, and a mild draw: each boat is among its {max(stats['rank'])} clearest of {n} here. "
                f"Over the {n} seeds, each giving all three boats the same gust and current:\n"
                f"touched a circle {t[0]} / {t[1]} / {t[2]} · reached the goal within 150 s {g[0]} / {g[1]} / {g[2]} (MPPI / barrier / + IS) · "
                f"MPPI < barrier < + IS on closest approach in {stats['order']} of {n} paired seeds"
                + (" · the last three seconds show all 30 paths, those that touched a circle darker" if ghost_runs is not None else ""))
    else:
        foot = "over the 30 seeds of experiment V3 the runs touching a circle are 29 / 9 / 3 of 30 (MPPI / SCBF-MPPI / + IS)"
    fig.text(0.5, 0.008, foot, ha="center", va="bottom", fontsize=8.8, color=COL["muted"], linespacing=1.4)

    def update(f):
        out = []
        t_abs = f * sg.DT_CTRL
        offx = (cur[0] * t_abs) % spacing; offy = (cur[1] * t_abs) % spacing
        fade = min(1.0, (f - n_run + 1) / (0.75 * fps)) if f >= n_run else 0.0
        for pi, (samp, path, hull, gust, glabel, txt, r, streaks, gl, gtxt) in enumerate(arts):
            k = min(f, r["steps"] - 1)
            tr = r["traj"]
            path.set_data(tr[:k + 1, 0], tr[:k + 1, 1])
            hull.set_xy(sg.hull_polygon(tr[k]))
            streaks.set_offsets(np.column_stack([GX[keep] + jit[keep][:, 0] + offx, GY[keep] + jit[keep][:, 1] + offy]))
            kk, S, w = r["samples"][k]
            idx = np.argsort(w)[::-1][:n_show]
            wmax = max(float(w[idx].max()), 1e-12)
            for line, i in zip(samp, idx):
                line.set_data(S[i, :, 0], S[i, :, 1]); line.set_alpha(0.06 + 0.5 * float(w[i]) / wmax)
            # the gust force acting as this interval begins (body frame -> world)
            td = r["dist"][k - 1] if k > 0 else np.zeros(3)
            psi = tr[k, 2]; fx = np.cos(psi) * td[0] - np.sin(psi) * td[1]; fy = np.sin(psi) * td[0] + np.cos(psi) * td[1]
            fmag = float(np.hypot(fx, fy)); L = fmag / 100.0 * 2.5
            x0, y0 = tr[k, 0], tr[k, 1]
            if fmag > 15:
                gust.xy = (x0 + fx / fmag * L, y0 + fy / fmag * L); gust.set_position((x0, y0)); gust.set_visible(True)
                glabel.set_position((x0 + fx / fmag * L + 1.5, y0 + fy / fmag * L)); glabel.set_text(f"gust {fmag:.0f} N")
            else:
                gust.set_visible(False); glabel.set_text("")
            # clearance now and the closest approach so far; executed azimuth thrust against the 700 N disk
            hm = float(r["hmin_series"][k]); h_closest = float(np.min(r["hmin_series"][:k + 1]))
            n_coll = int((np.asarray(r["hmin_series"][:k + 1]) < 0).sum())
            u = r["ctrl"][min(k, len(r["ctrl"]) - 1)]; f_at = float(np.hypot(u[0], u[1]))
            line1 = f"t = {k * sg.DT_CTRL:.0f} s   speed {abs(tr[k, 3]):.1f} m/s   thrust {f_at:.0f} / {sg.F_AT_MAX:.0f} N" + ("  · at limit" if f_at >= 0.98 * sg.F_AT_MAX else "")
            if k >= r["steps"] - 1:
                line1 += "   ·  " + ("goal reached" if r["reached"] else "time limit")
            line2 = f"h now {hm:+.1f} m   closest so far {h_closest:+.1f} m   collisions {n_coll}   ESS {r['ess_series'][k]:.0f}"
            txt.set_text(line1 + "\n" + line2)
            # the ghosts fade in over the final hold
            if gl:
                for ln, touched in gl:
                    ln.set_alpha(fade * (0.45 if touched else 0.13))
                gtxt.set_alpha(fade); gtxt.set_text(f"all {len(gl)} seeds" if fade > 0 else "")
                out += [ln for ln, _ in gl] + [gtxt]
            out += samp + [path, hull, txt, streaks, gust, glabel]
        return out

    stem = "anim_vessel_current_ghosts" if ghost_runs is not None else "anim_vessel_current"
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    mp4 = os.path.join(FIG, stem + ".mp4"); gif = os.path.join(FIG, stem + ".gif")
    try:
        anim.save(mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2200, extra_args=FFMPEG_ARGS), dpi=110)
        print("wrote", mp4, file=sys.stderr)
    except Exception as e:
        print("mp4 failed:", e, file=sys.stderr)
    anim2 = animation.FuncAnimation(fig, update, frames=range(0, n_frames, 2), interval=2000 / fps, blit=False)
    anim2.save(gif, writer=animation.PillowWriter(fps=max(1, fps // 2)), dpi=72)
    print("wrote", gif, file=sys.stderr)
    update(n_frames - 1); fig.savefig(os.path.join(FIG, stem + "_last.png"), dpi=150)
    # a mid-run frame (the moment MPPI is being set onto the barge)
    k_mid = int(np.argmin(runs[0]["hmin_series"])); update(min(k_mid, n_run - 1)); fig.savefig(os.path.join(FIG, stem + "_mid.png"), dpi=150)
    print("mid frame at MPPI's closest approach, k =", k_mid, file=sys.stderr)


if __name__ == "__main__":
    main(ghosts=("--ghosts" in sys.argv))
