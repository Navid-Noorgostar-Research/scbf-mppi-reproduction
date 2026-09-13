"""Inside one cycle: the map with the sampled fan (left) and, for the row the filter works on at t = 0 of that cycle,
the distribution of a·u along the row's normal (right), in units of the nominal spread σ₀ — the nominal proposal N(a·ū, aᵀΣ₀a), the filtered one
N(a·(ū+μ), |Pᵀa|²) after problem (8) with the corrected constraint, the threshold b, the mass left of b, and the two
requirement margins at the nominal spread: z·σ (corrected, standard-deviation form) and α·σ² (variance form as
printed; α = z, so their ratio is σ in the row's units — about 0.1 on the vessel).
SCBF-MPPI, second-order barrier, crossing-ferry scene, seed 2, white force noise.
    python -m scbf_mppi.vessel.animate_rowspace   -> figures/anim_vessel_rowspace.mp4 / .gif / anim_vessel_rowspace_last.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Polygon
from scipy.stats import norm
from . import solgenia as sg, harbour as hb, solver_nd
from .controllers import VesselSCBFMPPI
from .experiments import FIG, DEFAULT

COL = {"std": "#5B4FB0", "ink": "#20303C", "muted": "#5D7180", "water": "#EAF2F6", "obst": "#8A9BA8", "bad": "#C2482A", "var": "#C2482A"}


def simulate(seed=2, n_show=60, scenario="crossing"):
    """One SCBF-MPPI run that also records, per cycle, the t = 0 row picture: (a, b, ū, μ, Pfac, active, row name)."""
    H = hb.harbour_crossing() if scenario == "crossing" else hb.harbour_static()
    c = VesselSCBFMPPI(H, K=DEFAULT["K"], T=DEFAULT["T"], lam=DEFAULT["lam"], s0=DEFAULT["s0"], delta=DEFAULT["delta"], seed=seed)
    rng = np.random.default_rng(10_000 + seed)
    s_F = np.asarray(DEFAULT["s_F"]); tau_c = DEFAULT["tau_c"]; dt = c.dt
    x = np.array(hb.X0, float)
    traj = [x.copy()]; frames = []; hmin = []; ess = []
    c.t = 0.0
    for k in range(int(round(DEFAULT["max_time"] / dt))):
        t_now = k * dt
        # the row picture the K samples share at t = 0 of this plan: rows at x, nominal ū = U[0] before the update
        ubar = c.U[0].copy()
        A, b, hh, psi1 = H.rows(x[None], t_now, 0.0)
        A = A @ sg.B_ALLOC
        res = solver_nd.solve_rows_nd(ubar[None], A, b, c.s0, z=c.z, alpha=c.alpha, form=c.form)
        a_all = A[0]; b_all = b[0]
        if bool(res["active"][0]):
            j = int(res["row"][0])
        else:   # inactive: show the most violated row by nominal z-score
            m = a_all @ ubar - b_all; sa = np.linalg.norm(a_all * c.s0[None, :], axis=-1)
            j = int(np.argmin(np.where(sa > 1e-12, m / np.maximum(sa, 1e-12), m)))
        a = a_all[j]; bb = float(b_all[j]); mu = res["mu"][0]; P = res["Pfac"][0]
        m0 = float(a @ ubar); s0a = float(np.linalg.norm(a * c.s0)); m1 = float(a @ (ubar + mu)); s1 = float(np.linalg.norm(P.T @ a))
        u = c.plan(x)
        info = c.last
        S = info["traj"][:, :, :2].astype(np.float32); w = info["w"].astype(np.float32)
        frames.append(dict(k=k, x=x.copy(), name=H.obs[j].name, h=float(hh[0, j]), a=a.copy(), b=bb, m0=m0, s0=s0a, m1=m1, s1=s1,
                           active=bool(res["active"][0]), z=float(c.z), alpha=float(c.alpha), S=S, w=w))
        x_new, subs = sg.step(x[None], u[None], dt, current=(0.0, 0.0), sigma_F=s_F * np.sqrt(2.0 * tau_c), rng=rng, return_sub=True)
        x = x_new[0]; traj.append(x.copy())
        hmin.append(min(float(H.min_h(z[None], t_now + (i + 1) * dt / len(subs))[0]) for i, z in enumerate([q[0] for q in subs])))
        ess.append(info["ess"])
        if np.linalg.norm(x[:2] - H.goal) < H.goal_radius:
            break
    return H, np.array(traj), frames, np.array(hmin), np.array(ess)


def main(seed=2, n_show=60, fps=6):
    H, traj, frames, hmin, ess = simulate(seed, n_show)
    n = len(frames)
    print("steps", n, "min h", hmin.min().round(2), "collision steps", int((hmin < 0).sum()), "active cycles", sum(f["active"] for f in frames), file=sys.stderr)
    fig = plt.figure(figsize=(14.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], left=0.01, right=0.985, top=0.9, bottom=0.21, wspace=0.08)
    ax = fig.add_subplot(gs[0]); rx = fig.add_subplot(gs[1])
    col = COL["std"]
    ax.set_facecolor(COL["water"])
    ferry = H.obs[-1] if len(H.obs) == 4 else None
    for o in H.obs:
        if np.linalg.norm(o.w) == 0:
            ax.add_patch(Circle(o.center(0), o.R, facecolor=COL["obst"], edgecolor=COL["ink"], lw=0.8, alpha=0.5))
            ax.add_patch(Circle(o.center(0), max(o.R - H.r_ego, 0.5), facecolor=COL["ink"], edgecolor="none", alpha=0.35))
            ax.text(o.center(0)[0], o.center(0)[1] + o.R + 2.5, o.name, fontsize=8, color=COL["ink"], ha="center", va="bottom")
    ax.plot(H.goal[0], H.goal[1], "x", color=COL["ink"], ms=10, mew=2)
    ax.set_aspect("equal"); ax.set_xlim(-10, 130); ax.set_ylim(-60, 40); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title("SCBF-MPPI, second-order barrier · crossing ferry · seed 2 · K = 500", loc="left", fontsize=11, color=col)
    samp = [ax.plot([], [], color=col, lw=0.6, alpha=0.15)[0] for _ in range(n_show)]
    path, = ax.plot([], [], color=col, lw=2.0)
    hull = Polygon(np.zeros((5, 2)), closed=True, facecolor=col, edgecolor=COL["ink"], lw=0.8); ax.add_patch(hull)
    fpatch = Circle((0, -1000), ferry.R, facecolor=COL["obst"], edgecolor=COL["ink"], lw=0.8, alpha=0.5) if ferry else None
    fcore = Circle((0, -1000), ferry.R - H.r_ego, facecolor=COL["ink"], edgecolor="none", alpha=0.35) if ferry else None
    if ferry: ax.add_patch(fpatch); ax.add_patch(fcore)
    flabel = ax.text(0, -1000, "ferry, 3 m/s", fontsize=8, color=COL["ink"], ha="center", va="center") if ferry else None
    rowmark, = ax.plot([], [], "o", ms=9, mfc="none", mec=COL["bad"], mew=2)
    txt = ax.text(0.0, -0.02, "", transform=ax.transAxes, fontsize=9, color=COL["ink"], va="top", ha="left")
    fig.text(0.5, 0.965, "Inside one cycle: the row the filter works on at t = 0, along its normal — nominal proposal (dashed), after problem (8) with the corrected constraint (solid), threshold b, mass left of b (red)",
             ha="center", fontsize=10.5, color=COL["ink"])
    fig.text(0.5, 0.045, "left: 60 of the 500 rollouts of the cycle, opacity ∝ weight; the ringed obstacle is the row shown · right: the margin (a·u − b)/σ₀ in units of the nominal spread, one fixed axis for the whole run",
             ha="center", fontsize=9, color=COL["ink"])
    fig.text(0.5, 0.012, "ticks: the margins at the nominal spread — z·σ for the corrected form, α·σ² for the variance form as printed (α = z, so their ratio is σ in the row's units: about 0.1 here)",
             ha="center", fontsize=9, color=COL["muted"])

    # standardised units: x = (a·u − b)/σ₀ — b at 0, the nominal proposal a unit Gaussian, the corrected margin the constant z,
    # the printed one α·σ₀; one fixed axis for the whole run, so the picture moves smoothly instead of re-scaling every cycle
    zs0 = [(f["m0"] - f["b"]) / f["s0"] for f in frames]; zs1 = [(f["m1"] - f["b"]) / f["s0"] for f in frames]
    zmax = max(frames[0]["z"], max(zs0) + 3.4, max(zs1) + 3.4); zmin = min(-4.5, min(zs0) - 3.4, min(zs1) - 3.4)
    lo, hi = max(zmin, -5.0), min(max(zmax, 7.5), 8.0)   # inactive rows far from b may run off the right edge; the action is near b
    xs = np.linspace(lo, hi, 700)

    def draw_row(f):
        rx.cla()
        rx.set_facecolor("white")
        for sp in ["top", "right", "left"]: rx.spines[sp].set_visible(False)
        rx.set_yticks([])
        m0, s0, m1, s1, b, z, al = f["m0"], f["s0"], f["m1"], f["s1"], f["b"], f["z"], f["alpha"]
        z0 = (m0 - b) / s0; z1 = (m1 - b) / s0; s1z = s1 / s0
        collapsed = s1z < 0.06
        tick_std, tick_var = z, al * s0
        f0 = norm.pdf(xs, z0, 1.0); p0 = norm.pdf(0, 0, 1.0)
        for v in range(int(np.ceil(lo)), int(np.floor(hi)) + 1):
            if v != 0: rx.axvline(v, color="#E3E8EC", lw=0.8, zorder=0)
        rx.fill_between(xs, 0, f0, color=col, alpha=0.12); rx.plot(xs, f0, color=col, lw=1.6, ls="--")
        pr0 = norm.cdf(z0)
        ymax = 2.6 * p0
        if collapsed:
            pr1 = 1.0 if m1 >= b - 1e-7 * (1 + abs(b)) else 0.0
            rx.axvline(z1, color=col if pr1 >= 0.5 else COL["bad"], lw=5, ymin=0, ymax=0.86)
            rx.plot([z1], [ymax], "o", color=col, ms=7)
            rx.text(z1 + 0.02 * (hi - lo), ymax * 0.98, f"after the filter · point mass · Pr {pr1:.3f}", fontsize=9, color=COL["ink"], va="center")
        else:
            f1 = norm.pdf(xs, z1, s1z); p1 = norm.pdf(0, 0, s1z); scale = min(1.0, 2.6 * p0 / p1); f1 = f1 * scale
            pr1 = norm.cdf(z1 / s1z)
            rx.fill_between(xs, 0, f1, color=col, alpha=0.3); rx.plot(xs, f1, color=col, lw=2.4)
            rx.fill_between(xs[xs < 0], 0, f1[xs < 0], color=COL["bad"], alpha=0.45)
            rx.text(z1 + 0.02 * (hi - lo), p1 * scale * 0.98, f"after the filter · Pr {pr1:.3f}" + (" (peak capped)" if scale < 1 else ""), fontsize=9, color=COL["ink"], va="center")
        rx.fill_between(xs[xs < 0], 0, f0[xs < 0], color=COL["bad"], alpha=0.18)
        rx.text(z0 - 0.02 * (hi - lo), p0 * 1.02, f"nominal · Pr {pr0:.3f}", fontsize=9, color=COL["ink"], va="bottom", ha="right")
        rx.axvline(0, color=COL["bad"], lw=2.2, ymin=0, ymax=0.86); rx.text(-0.012 * (hi - lo) if 0 <= z1 < 1.4 else 0.012 * (hi - lo), ymax * 1.13, "b", color=COL["bad"], fontsize=11, fontweight="bold", ha="right" if 0 <= z1 < 1.4 else "left", va="center")
        rx.plot([tick_std], [0], marker="^", color=COL["ink"], ms=9, clip_on=False); rx.text(tick_std, -0.10 * ymax, "z·σ — corrected form", fontsize=8.5, color=COL["ink"], ha="center", va="top")
        dy = -0.22 * ymax if abs(tick_std - tick_var) < 0.24 * (hi - lo) else -0.10 * ymax   # the two tick labels are ~0.2 of the axis wide each: stagger them when they would touch
        rx.plot([tick_var], [0], marker="^", color=COL["var"], ms=9, clip_on=False); rx.text(tick_var, dy, "α·σ² — as printed", fontsize=8.5, color=COL["var"], ha="center", va="top")
        rx.set_xlim(lo, hi); rx.set_ylim(-0.02 * ymax, ymax * 1.38); rx.set_xticks(range(int(np.ceil(lo)), int(np.floor(hi)) + 1)); rx.tick_params(axis='x', labelsize=8.5, colors=COL['muted'])
        status = "active" if f["active"] else "inactive at t = 0 (the nominal already satisfies the chance constraint)"
        rx.set_title(f"{f['name']} row · h = {f['h']:.1f} m · {status}", loc="left", fontsize=10, color=COL["ink"])
        rx.text(0.0, 0.985, f"σ₀ = {s0:.3f} m/s² · shift a·μ = {m1 - m0:+.3f} m/s² ({(m1 - m0) / s0:+.2f} σ₀) · " + ("σ → 0: point mass" + (" on b" if abs(m1 - b) < 1e-6 * (1 + abs(b)) else "") if collapsed else f"σ kept {100 * s1z:.0f} %") + f" · z·σ = {z * s0:.3f}, α·σ² = {al * s0 ** 2:.3f}",
                transform=rx.transAxes, fontsize=9, color=COL["muted"], va="top", ha="left")

    def update(i):
        f = frames[i]; k = f["k"]
        path.set_data(traj[:k + 1, 0], traj[:k + 1, 1]); hull.set_xy(sg.hull_polygon(traj[k]))
        if fpatch is not None:
            c = ferry.center(k * sg.DT_CTRL); fpatch.center = c; fcore.center = c; flabel.set_position((c[0], c[1]))
        S, w = f["S"], f["w"]
        idx = np.argsort(w)[::-1][:n_show]; wmax = max(float(w[idx].max()), 1e-12)
        for line, j in zip(samp, idx):
            line.set_data(S[j, :, 0], S[j, :, 1]); line.set_alpha(0.06 + 0.5 * float(w[j]) / wmax)
        o = [ob for ob in H.obs if ob.name == f["name"]][0]; cc = o.center(k * sg.DT_CTRL); rowmark.set_data([cc[0]], [cc[1]])
        n_coll = int((hmin[:k + 1] < 0).sum())
        txt.set_text(f"t = {k * sg.DT_CTRL:.0f} s   speed {abs(traj[k, 3]):.1f} m/s   min h {hmin[k]:+.1f} m   collision steps so far {n_coll}   ESS {ess[k]:.0f}")
        draw_row(f)
        return samp + [path, hull, txt, rowmark]

    anim = animation.FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)
    mp4 = os.path.join(FIG, "anim_vessel_rowspace.mp4"); gif = os.path.join(FIG, "anim_vessel_rowspace.gif")
    try:
        anim.save(mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2200), dpi=110)
        print("wrote", mp4, file=sys.stderr)
    except Exception as e:
        print("mp4 failed:", e, file=sys.stderr)
    anim2 = animation.FuncAnimation(fig, update, frames=range(0, n, 2), interval=2000 / fps, blit=False)
    anim2.save(gif, writer=animation.PillowWriter(fps=max(1, fps // 2)), dpi=72)
    print("wrote", gif, file=sys.stderr)
    # the still frame: the cycle with the largest mean shift or the deepest shrink while active (the filter at work)
    act = [i for i, f in enumerate(frames) if f["active"] and not f["s1"] < 0.06 * f["s0"]]
    i_show = max(act, key=lambda i: (frames[i]["m1"] - frames[i]["m0"]) / max(frames[i]["s0"], 1e-9) + (1 - frames[i]["s1"] / max(frames[i]["s0"], 1e-9))) if act else n - 1
    update(i_show); fig.savefig(os.path.join(FIG, "anim_vessel_rowspace_last.png"), dpi=150)
    print("still frame: cycle", frames[i_show]["k"], frames[i_show]["name"], "active", frames[i_show]["active"], file=sys.stderr)


if __name__ == "__main__":
    main()
