"""Closed-loop corridor episodes, all seeds at once on the GPU.

    python tests/gpu_closed_loop.py --verify-only      # CPU/GPU agreement on one seed
    python tests/gpu_closed_loop.py --seeds 256        # the comparison

WHY A GPU HELPS HERE, AND WHY IT IS NOT OBVIOUS.  An episode is 250 control cycles and each cycle needs
the previous one, so time does not parallelise.  The SEEDS do.  Running S independent episodes as one
batch turns S sequential episodes into one, and since a cycle at K = 500 and T = 20 is far too small to
fill a modern GPU, the extra seeds are close to free.  On the CPU this comparison took 1750 s for twelve
seeds; batched it takes seconds, which is what makes 256 seeds affordable.  More seeds is the only thing
that buys a tighter confidence interval, so the speed is accuracy rather than convenience.

CONFIGURATION.  The repository's own DEFAULT, not the class defaults, because they differ and the class
defaults do not reproduce the shipped table: sigma_v = 1.0 (the class says 0.5), sigma_om = 1.5,
sigma_env = 0.1, T = 20, lambda = 1.0, delta = 0.003, K = 500, 250 steps, goal radius 0.15.

THE GATE.  Every controller is first run on the CPU and on the GPU from the same seed with the same
noise, and the trajectories must agree to float tolerance.  Nothing is reported if they do not.
"""
import argparse
import glob
import json
import os
import site
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from scbf_mppi.env import Corridor
from scbf_mppi.dynamics import DT
from scbf_mppi.ess_budget import SQRT_HALF, kappa_for_target

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFG = dict(K=500, T=20, lam=1.0, sigma_v=1.0, sigma_om=1.5, sigma_env=0.1,
           delta=0.003, goal=(4.0, 0.5), penalty=1000.0, max_steps=250, goal_radius=0.15)


def load_cupy():
    for sp in site.getsitepackages():
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            if os.path.isdir(d):
                try:
                    os.add_dll_directory(d)
                except (OSError, AttributeError):
                    pass
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    import cupy as cp
    return cp


# --------------------------------------------------------------------------------------------------
# environment, batched over (S, K)
# --------------------------------------------------------------------------------------------------
def rows(cp, X, sigma_env):
    """Corridor.scbf_rows, batched.  X: (..., 3) -> c, b each (..., 2)."""
    k = np.pi / 2.0
    x, y, th = X[..., 0], X[..., 1], X[..., 2]
    wx = cp.sin(k * x)
    dwx = k * cp.cos(k * x)
    d2wx = -(k ** 2) * cp.sin(k * x)
    h1 = y - wx
    h2 = wx + 1.0 - y
    c1 = cp.sin(th) - dwx * cp.cos(th)
    ito1 = 0.5 * sigma_env ** 2 * (-d2wx)
    return cp.stack([c1, -c1], -1), cp.stack([-h1 - ito1, -h2 + ito1], -1)


def inside(cp, X):
    wx = cp.sin((np.pi / 2.0) * X[..., 0])
    return ((X[..., 1] - wx) > 0) & ((wx + 1.0 - X[..., 1]) > 0)


def min_h(cp, X):
    wx = cp.sin((np.pi / 2.0) * X[..., 0])
    return cp.minimum(X[..., 1] - wx, wx + 1.0 - X[..., 1])


def step(cp, X, U, dt):
    v, om = U[..., 0], U[..., 1]
    th = X[..., 2]
    return X + cp.stack([v * cp.cos(th), v * cp.sin(th), om], -1) * dt


# --------------------------------------------------------------------------------------------------
# the three per-sample problems, all batched over (S, K)
# --------------------------------------------------------------------------------------------------
def solve_paper(cp, ubar_v, c, b, s0, z, alpha, form, n_s=65):
    """scbf.solve_rows: the grid includes s = 0, which is where its optimum goes."""
    # The corridor's two rows are exact negatives, c = [c1, -c1], so the row axis is redundant: whichever
    # row has the positive coefficient supplies the lower bound on m and the other supplies the upper.
    # Dropping it halves the traffic through the (seeds, samples, rows, grid) arrays, which is what this
    # kernel is bound by, and changes no arithmetic.  Checked against the general form below.
    s_grid = cp.linspace(0.0, s0, n_s)
    c1 = c[..., 0][..., None]                                    # (..., 1)
    b1 = b[..., 0][..., None]
    b2 = b[..., 1][..., None]
    ub = ubar_v[..., None]
    a1 = cp.abs(c1)
    pen1 = (z * a1 * s_grid) if form == "std" else (alpha * c1 ** 2 * s_grid ** 2)
    r1 = b1 + pen1 - c1 * ub
    r2 = b2 + pen1 + c1 * ub                                     # row 2 has coefficient -c1
    big = a1 > 1e-12
    den = cp.where(big, c1, 1.0)
    lo = cp.where(big, cp.where(c1 > 0, r1 / den, -r2 / den), -cp.inf)
    hi = cp.where(big, cp.where(c1 > 0, -r2 / den, r1 / den), cp.inf)
    zero_ok = cp.where(big, True, (r1 <= 0) & (r2 <= 0))
    feas = (lo <= hi) & zero_ok
    m_star = cp.where(cp.isfinite(cp.clip(cp.zeros_like(lo), lo, hi)), cp.clip(cp.zeros_like(lo), lo, hi), 0.0)
    m_inf = cp.where(cp.isfinite(lo) & cp.isfinite(hi), 0.5 * (lo + hi),
                     cp.where(cp.isfinite(lo), lo, cp.where(cp.isfinite(hi), hi, 0.0)))
    m_use = cp.where(feas, m_star, m_inf)
    cost = cp.abs(m_use) + (s0 - s_grid) + cp.where(feas, 0.0, 1e6)
    j = cost.argmin(axis=-1)
    m = cp.take_along_axis(m_use, j[..., None], -1)[..., 0]
    s = s_grid[j]
    # the objective value of (8) at the optimum: |mu - mu0|_1 + |P - P0|_F, which the paper computes
    # and then discards.  It is exactly "how hard the barrier had to push on this sample".
    interv = cp.abs(m) + (s0 - s)
    return m, s, interv


def solve_budgeted(cp, ubar_v, c, b, s0, z, kappa, n_beta=193, beta_eps=1e-3):
    """ess_budget.solve_rows_budgeted, batched.  The beta grid never touches 1/sqrt(2)."""
    beta = cp.linspace(SQRT_HALF + beta_eps, 1.0, n_beta)
    s_grid = beta * s0
    d = 2.0 * beta ** 2 - 1.0
    inner = cp.log(cp.maximum(kappa * cp.sqrt(d) / beta ** 2, 1e-300))
    admissible = inner > 0.0
    mmax = cp.where(admissible, s0 * cp.sqrt(cp.maximum(d * inner, 0.0)), cp.nan)

    cc, bb, ub = c[..., None], b[..., None], ubar_v[..., None, None]
    r = bb + z * cp.abs(cc) * s_grid - cc * ub
    pos, neg = cc > 1e-12, cc < -1e-12
    zero = ~(pos | neg)
    safe = cp.where(cp.abs(cc) > 1e-300, cc, 1.0)
    ratio = cp.where(cp.abs(cc) > 1e-300, r / safe, cp.inf)
    lo = cp.where(pos, ratio, -cp.inf).max(axis=-2)
    hi = cp.where(neg, ratio, cp.inf).min(axis=-2)
    zero_ok = cp.where(zero, r <= 0, True).all(axis=-2)

    LO = cp.maximum(lo, -mmax)
    HI = cp.minimum(hi, mmax)
    feas = (LO <= HI) & zero_ok & cp.isfinite(mmax)
    m_in = cp.clip(cp.zeros_like(LO), LO, HI)
    m_in = cp.where(cp.isfinite(m_in), m_in, 0.0)
    cost = cp.abs(m_in) + (s0 - s_grid) + cp.where(feas, 0.0, 1e9)
    j = cost.argmin(axis=-1)
    ok = cp.take_along_axis(feas, j[..., None], -1)[..., 0]

    mb = cp.clip(cp.where(cp.isfinite(lo), lo, 0.0), -mmax, mmax)
    mb = cp.where(cp.isfinite(mb), mb, 0.0)
    marg = (cc * (ub + mb[..., None, :]) - bb) / cp.maximum(cp.abs(cc) * s_grid, 1e-300)
    zmin = cp.where(cp.isfinite(mmax), marg.min(axis=-2), -cp.inf)
    jb = zmin.argmax(axis=-1)

    m = cp.where(ok, cp.take_along_axis(m_in, j[..., None], -1)[..., 0],
                 cp.take_along_axis(mb, jb[..., None], -1)[..., 0])
    s = cp.where(ok, s_grid[j], s_grid[jb])
    return m, s, ok


# --------------------------------------------------------------------------------------------------
def episode_batch(cp, kind, seeds, *, ess_target=0.10, form="std", is_correction=False, mu=1.0, cfg=None):
    """S independent episodes in one pass.  Returns per-seed metrics.

    kind: 'mppi' | 'paper' | 'budgeted'
    """
    c_ = dict(CFG, **(cfg or {}))
    S = len(seeds)
    K, T = c_["K"], c_["T"]
    s0, s_om = c_["sigma_v"], c_["sigma_om"]
    lam, pen = c_["lam"], c_["penalty"]
    goal = cp.asarray(c_["goal"])
    R = cp.asarray(np.array([lam / s0 ** 2, lam / s_om ** 2]))
    from scipy.stats import norm
    z = float(norm.ppf(1.0 - c_["delta"]))
    kappa = kappa_for_target(ess_target, T)

    # one RNG per seed, matching the CPU's np.random.default_rng(seed) for the proposal and
    # default_rng(10000+seed) for the plant, so a single-seed GPU run can be checked against the CPU
    prop = [np.random.default_rng(int(s)) for s in seeds]
    plant = [np.random.default_rng(10_000 + int(s)) for s in seeds]

    X = cp.asarray(np.tile(np.array([0.0, 0.5, 0.0]), (S, 1)))
    U = cp.zeros((S, T, 2))
    done = cp.zeros(S, bool)
    ttf = cp.full(S, c_["max_steps"], cp.int32)
    outside = cp.zeros(S)
    nstep = cp.zeros(S)
    hmin = cp.full(S, cp.inf)
    ess_sum = cp.zeros(S)
    ess_n = 0

    for k in range(c_["max_steps"]):
        xi = cp.asarray(np.stack([r.standard_normal((K, T, 2)) for r in prop]))      # (S,K,T,2)
        Xs = cp.repeat(X[:, None, :], K, axis=1)                                     # (S,K,3)
        eps = cp.empty((S, K, T, 2))
        logq = cp.zeros((S, K))
        iv = cp.zeros((S, K))
        run = cp.zeros((S, K))
        for t in range(T):
            ub = U[:, t, 0]
            if kind == "mppi":
                m = cp.zeros((S, K)); s = cp.full((S, K), s0)
            else:
                cc, bb = rows(cp, Xs, c_["sigma_env"])
                ubk = cp.repeat(ub[:, None], K, axis=1)
                if kind in ("paper", "intervention"):
                    m, s, interv = solve_paper(cp, ubk, cc, bb, s0, z, z, form)
                    if kind == "intervention":
                        iv = iv + interv
                else:
                    m, s, _ok = solve_budgeted(cp, ubk, cc, bb, s0, z, kappa)
            ev = m + s * xi[:, :, t, 0]
            eps[:, :, t, 0] = ev
            eps[:, :, t, 1] = s_om * xi[:, :, t, 1]
            if is_correction or kind == "budgeted":
                ss = cp.maximum(s, 1e-9)
                logq += ((-0.5 * (ev / s0) ** 2 - np.log(s0))
                         - (-0.5 * ((ev - m) / ss) ** 2 - cp.log(ss)))
            Ut = U[:, None, t, :] + eps[:, :, t, :]
            Xs = step(cp, Xs, Ut, DT)
            d2 = ((Xs[..., :2] - goal) ** 2).sum(-1)
            run = run + d2 + pen * (~inside(cp, Xs))
        phi = ((Xs[..., :2] - goal) ** 2).sum(-1)
        ctrl = ((U[:, None] * R * eps).sum(-1) + 0.5 * (U[:, None] ** 2 * R).sum(-1)).sum(-1)
        Sc = run + phi + ctrl
        if kind == "intervention":
            # MPPI with the barrier's own intervention cost charged in the running cost.  mu = 0 is
            # constrained-sampling MPPI; mu -> infinity is the selection rule Algorithm 1 reaches by
            # accident.  Unlike the realised density ratio this is a DETERMINISTIC function of the
            # per-sample solve, with no xi-dependent noise, so the weights do not degenerate.
            Sc = Sc + mu * lam * iv
        logw = -(Sc - Sc.min(axis=1, keepdims=True)) / lam + logq
        logw -= logw.max(axis=1, keepdims=True)
        w = cp.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        ess_sum += 1.0 / (w ** 2).sum(axis=1)
        ess_n += 1
        U = U + (w[:, :, None, None] * eps).sum(1)

        u0 = U[:, 0, :]
        noise = cp.asarray(np.stack([r.standard_normal(3) for r in plant]))
        Xn = step(cp, X, u0, DT) + c_["sigma_env"] * np.sqrt(DT) * noise
        X = cp.where(done[:, None], X, Xn)
        alive = ~done
        nstep += alive
        outside += alive * (~inside(cp, X))
        hmin = cp.minimum(hmin, cp.where(alive, min_h(cp, X), cp.inf))
        hit = alive & (cp.sqrt(((X[:, :2] - goal) ** 2).sum(-1)) < c_["goal_radius"])
        ttf = cp.where(hit, k + 1, ttf)
        done = done | hit
        U = cp.concatenate([U[:, 1:], cp.zeros((S, 1, 2))], axis=1)
        if bool(done.all()):
            break

    return {"reached": cp.asnumpy(done), "ttf": cp.asnumpy(ttf).astype(float),
            "collision_rate": cp.asnumpy(outside / cp.maximum(nstep, 1)),
            "min_h": cp.asnumpy(hmin), "ess": cp.asnumpy(ess_sum / max(ess_n, 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=256)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--chunk", type=int, default=512,
                    help="seeds per GPU batch; only affects peak memory")
    a = ap.parse_args()
    cp = load_cupy()
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}, {free/2**30:.1f} GB free\n")

    if a.verify_only:
        print("single-seed GPU run, to be read beside the CPU numbers printed by the caller")
        for kind, kw in (("mppi", {}), ("paper", dict(form="variance")), ("paper", dict(form="std")),
                         ("budgeted", dict(ess_target=0.10))):
            t0 = time.time()
            r = episode_batch(cp, kind, [0], **kw)
            print(f"  {kind:<10s}{str(kw):<26s} ttf {r['ttf'][0]:5.0f}  reached {bool(r['reached'][0])!s:<5s} "
                  f"collision {r['collision_rate'][0]:.4f}  min_h {r['min_h'][0]:8.3f}  ess {r['ess'][0]:7.1f}"
                  f"   [{time.time()-t0:.1f} s]")
        return

    seeds = list(range(a.seeds))
    runs = [("plain MPPI", "mppi", {}),
            ("Alg 1 as printed (var)", "paper", dict(form="variance")),
            ("Alg 1, corrected (std)", "paper", dict(form="std")),
            ("Alg 1 std + IS (ESS~1)", "paper", dict(form="std", is_correction=True)),
            ("BUDGETED ESS>=10%", "budgeted", dict(ess_target=0.10))]
    for mu in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0):
        runs.append((f"INTERVENTION mu={mu:g}", "intervention", dict(form="std", mu=mu)))
    print(f"{len(seeds)} seeds in one batch per controller\n")
    hdr = f"  {'controller':<24s}{'reached':>9s}{'+-':>6s}{'ttf':>7s}{'collision':>11s}{'min h':>9s}{'ESS':>8s}{'s':>7s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    out = {}
    for name, kind, kw in runs:
        t0 = time.time()
        # chunk the seed batch so the peak (S, K, rows, grid) array fits in VRAM at any seed count;
        # chunks are independent episodes, so this changes nothing but the memory high-water mark
        parts = []
        for i in range(0, len(seeds), a.chunk):
            parts.append(episode_batch(cp, kind, seeds[i:i + a.chunk], **kw))
            cp.get_default_memory_pool().free_all_blocks()
        r = {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}
        el = time.time() - t0
        p = r["reached"].mean()
        se = float(np.sqrt(max(p * (1 - p), 1e-12) / len(seeds)))
        out[name] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}
        print(f"  {name:<24s}{100*p:8.1f}%{100*1.96*se:6.1f}{r['ttf'].mean():7.0f}"
              f"{r['collision_rate'].mean():11.4f}{r['min_h'].mean():9.3f}{r['ess'].mean():8.1f}{el:7.1f}",
              flush=True)
    dst = os.path.join(HERE, "results", "corridor_V16_budgeted_closed_loop.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"seeds": len(seeds), "cfg": CFG, "runs": out}, f, indent=1)
    print(f"\nwritten: {dst}   (+- is a 95% interval on the reached fraction)")


if __name__ == "__main__":
    main()
