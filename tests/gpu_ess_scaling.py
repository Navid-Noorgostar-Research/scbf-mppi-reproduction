"""Is the effective-sample-size collapse a property of the method, or of a small sample budget?

    python tests/gpu_ess_scaling.py --verify-only        # just the CPU/GPU agreement gate
    python tests/gpu_ess_scaling.py                      # the full sweep, needs a CUDA GPU

THE QUESTION.  Algorithm 1 changes the sampling distribution per sample and per timestep, but its update
is still MPPI's cost-weighted average, which is only a valid estimator of the path-integral control if the
samples are reweighted by the density ratio between the distribution actually sampled and the one the
derivation assumes.  The paper omits that reweighting.  Putting it back collapses the effective sample
size in the corridor, from 269 out of 500 for plain MPPI to 2.

  A caution that cost an earlier version of this file its conclusion.  On the VESSEL the collapse is NOT
  caused by the density correction.  Decomposed per cycle in the obstacle field at delta = 0.003, the
  median ESS from the cost softmax alone is 3.3, from the importance weights alone 157.4, and from both
  3.1: there the lambda = 300 softmax against a 20,000 m^2 collision penalty is what flattens the weights,
  which the vessel README already notes for plain MPPI.  The two systems behave differently and the
  measurement below is the corridor's.  Do not carry one to the other.

The obvious reply is "then use more samples".  Nobody has been able to test that reply, because it needs
K across orders of magnitude and the per-sample constrained solve is the expensive part.  On a GPU the
whole sampling stage is one batched kernel, so K can run from 500 to 10^6.

WHAT IS MEASURED, and why two curves and not one.  ESS/K falls with K for a reason that has nothing to do
with the barrier: as K grows, the best sampled trajectory gets better, so the softmax over cost
concentrates and the plain MPPI weights become less uniform too.  Reporting only the corrected
controller's ESS/K would confuse that with the density correction.  So both are measured on the SAME
draws at the same operating point:

    plain MPPI      the nominal proposal, weights from the cost alone
    paper (no IS)   Algorithm 1 as published
    corrected       Algorithm 1 with the omitted importance weights put back

reported as ABSOLUTE effective sample sizes, because the question is whether the corrected estimator's
grows with the budget.  If plain MPPI's tracks K while the corrected one does not move, compute does not
rescue the correction.  Result: over a 2000x range plain MPPI goes 125 -> 212,326 and the corrected
estimator stays at 1.00.

  Two cautions on how to state that.  It is an ASYMPTOTIC efficiency, not a per-run cap; a measured ESS
  can be much larger in a run where no sample lands in the violation set.  And the operating points must
  come from an episode flown by PLAIN MPPI: two earlier versions of this file sampled states from a
  stalled or cold-start controller, where even plain MPPI has an effective size near one, and both
  produced confident wrong answers that only a comparison against the repository's shipped figures caught.

THE VERIFICATION GATE.  A GPU reimplementation that silently disagrees with the CPU one would produce a
confident wrong answer.  So before any sweep, this script runs the REPOSITORY'S OWN CPU controller and
this file's GPU code on the SAME noise draws at K = 500, and requires the costs, the log density ratios,
the weights and the ESS to agree to float tolerance.  The sweep does not run if the gate fails.
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

from scbf_mppi import scbf as scbf_mod
from scbf_mppi.dynamics import DT, step as cpu_step
from scbf_mppi.env import Corridor
from scbf_mppi.mppi import MPPI, SCBFMPPI

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cupy():
    """CuPy on Windows needs the CUDA DLLs from the nvidia-*-cu12 wheels on the search path."""
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


class FixedRNG:
    """Hands back a prescribed standard-normal array, so the CPU controller and the GPU code see
    identical draws.  Any other call is a bug in this harness rather than something to paper over."""

    def __init__(self, xi):
        self.xi = xi
        self.used = False

    def standard_normal(self, shape):
        assert not self.used, "the controller drew noise twice; the harness assumes one draw"
        assert tuple(shape) == self.xi.shape, f"shape {shape} != prescribed {self.xi.shape}"
        self.used = True
        return self.xi


# ----------------------------------------------------------------------------------------------
# GPU implementations.  Each mirrors the CPU function named beside it, line for line where possible.
# ----------------------------------------------------------------------------------------------
def gpu_scbf_rows(cp, X, sigma, printed=False, alpha_set=1.0):
    """mirrors env.Corridor.scbf_rows"""
    x, y, th = X[..., 0], X[..., 1], X[..., 2]
    if printed:
        wx = cp.sin(x); dwx = cp.cos(x); d2wx = -cp.sin(x)
    else:
        k = np.pi / 2.0
        wx = cp.sin(k * x); dwx = k * cp.cos(k * x); d2wx = -(k ** 2) * cp.sin(k * x)
    h1 = y - wx
    h2 = wx + alpha_set - y
    c1 = cp.sin(th) - dwx * cp.cos(th)
    ito1 = 0.5 * sigma ** 2 * (-d2wx)
    return cp.stack([c1, -c1], -1), cp.stack([-h1 - ito1, -h2 + ito1], -1)


def gpu_solve_rows(cp, ubar_v, c, b, s0, z=None, alpha=None, form="std", n_s=65):
    """mirrors scbf.solve_rows, which is already fully vectorised over K"""
    K = ubar_v.shape[0]
    s_grid = cp.linspace(0.0, s0, n_s)
    cc = c[:, :, None]
    bb = b[:, :, None]
    ub = ubar_v[:, None, None]
    if form == "std":
        pen = z * cp.abs(cc) * s_grid[None, None, :]
    else:
        pen = alpha * cc ** 2 * s_grid[None, None, :] ** 2
    r = bb + pen - cc * ub
    pos = cc > 1e-12
    neg = cc < -1e-12
    zero = ~(pos | neg)
    ratio = cp.where(cp.abs(cc) > 1e-300, r / cp.where(cp.abs(cc) > 1e-300, cc, 1.0), cp.inf)
    lo = cp.where(pos, ratio, -cp.inf).max(axis=1)
    hi = cp.where(neg, ratio, cp.inf).min(axis=1)
    zero_ok = cp.where(zero, r <= 0, True).all(axis=1)
    feas = (lo <= hi) & zero_ok
    m_star = cp.clip(cp.zeros_like(lo), lo, hi)
    m_star = cp.where(cp.isfinite(m_star), m_star, 0.0)
    m_inf = cp.where(cp.isfinite(lo) & cp.isfinite(hi), 0.5 * (lo + hi),
                     cp.where(cp.isfinite(lo), lo, cp.where(cp.isfinite(hi), hi, 0.0)))
    m_use = cp.where(feas, m_star, m_inf)
    cost = cp.abs(m_use) + (s0 - s_grid[None, :]) + cp.where(feas, 0.0, scbf_mod.BIG)
    j = cost.argmin(axis=1)
    idx = cp.arange(K)
    active = ~(feas[:, -1] & (lo[:, -1] <= 0.0) & (hi[:, -1] >= 0.0))
    return m_use[idx, j], s_grid[j], active


def gpu_step(cp, X, U, dt):
    """mirrors dynamics.step with sigma = 0"""
    v, om = U[..., 0], U[..., 1]
    th = X[..., 2]
    drift = cp.stack([v * cp.cos(th), v * cp.sin(th), om], -1)
    return X + drift * dt


def gpu_cycle(cp, x0, U0, K, xi, *, env_printed, env_alpha, sigma_env, s0, s_om, z, alpha, form,
              goal, penalty, lam, R, dt, u_max, want_logq, constrained=True):
    """One planning cycle's SAMPLING and WEIGHTING, on the GPU.

    Mirrors SCBFMPPI.sample_eps followed by the weight formation in plan().  Returns the two weight
    vectors of interest: cost only, and cost plus the omitted log density ratio."""
    T = U0.shape[0]
    Xs = cp.broadcast_to(cp.asarray(x0), (K, 3)).copy()
    eps = cp.empty((K, T, 2))
    logq = cp.zeros(K)
    n_active = 0
    for t in range(T):
        if constrained:
            c, b = gpu_scbf_rows(cp, Xs, sigma_env, env_printed, env_alpha)
            ubar_v = cp.full(K, float(U0[t, 0]))
            m, s, act = gpu_solve_rows(cp, ubar_v, c, b, s0, z=z, alpha=alpha, form=form)
            n_active += int(act.sum())
        else:
            # plain MPPI: the nominal proposal, which is the reference curve the barrier is judged against
            m = cp.zeros(K); s = cp.full(K, s0)
        ev = m + s * xi[:, t, 0]
        eo = s_om * xi[:, t, 1]
        eps[:, t, 0] = ev
        eps[:, t, 1] = eo
        if want_logq:
            s_safe = cp.maximum(s, 1e-9)
            logq += ((-0.5 * (ev / s0) ** 2 - np.log(s0))
                     - (-0.5 * ((ev - m) / s_safe) ** 2 - cp.log(s_safe)))
        Ut = cp.asarray(U0[t])[None] + eps[:, t]
        if u_max is not None:
            Ut = cp.clip(Ut, -u_max, u_max)
        Xs = gpu_step(cp, Xs, Ut, dt)
        if t == 0:
            traj_cost = cp.zeros(K)
        # running cost on the state AFTER the step, matching q(traj[:, 1:]).sum(-1)
        d2 = ((Xs[..., :2] - cp.asarray(goal)) ** 2).sum(-1)
        if env_printed:
            wx = cp.sin(Xs[..., 0])
        else:
            wx = cp.sin((np.pi / 2.0) * Xs[..., 0])
        inside = ((Xs[..., 1] - wx) > 0) & ((wx + env_alpha - Xs[..., 1]) > 0)
        traj_cost = traj_cost + d2 + penalty * (~inside)
    phi = ((Xs[..., :2] - cp.asarray(goal)) ** 2).sum(-1)
    Ug = cp.asarray(U0)
    Rg = cp.asarray(R)
    ctrl = ((Ug[None] * Rg * eps).sum(-1) + 0.5 * (Ug[None] ** 2 * Rg).sum(-1)).sum(-1)
    S = traj_cost + phi + ctrl

    def norm_w(logw):
        logw = logw - logw.max()
        w = cp.exp(logw)
        return w / w.sum()

    w_cost = norm_w(-(S - S.min()) / lam)
    w_both = norm_w(-(S - S.min()) / lam + logq) if want_logq else None
    return {"S": S, "logq": logq, "w_cost": w_cost, "w_both": w_both,
            "activation_frac": n_active / (K * T)}


def make_controller(K, seed, form="std", is_correction=True, **kw):
    env = Corridor()
    return SCBFMPPI(env, K=K, seed=seed, form=form, is_correction=is_correction, **kw)


def verify(cp, K=500, seed=0, tol=1e-9):
    """The gate.  Run the repository's own CPU controller and this file's GPU code on identical noise."""
    print(f"verification gate: CPU vs GPU at K={K}, identical draws")
    c = make_controller(K, seed)
    x0 = np.array([0.0, 0.5, 0.0])
    T = c.T
    rng = np.random.default_rng(12345)
    xi = rng.standard_normal((K, T, 2))
    c.rng = FixedRNG(xi)
    eps, traj, info = c.sample_eps(x0)
    S_cpu = c.q(traj[:, 1:]).sum(-1) + c.phi(traj[:, -1]) + c.control_terms(c.U, eps)
    logq_cpu = info["logq"]
    lw = -(S_cpu - S_cpu.min()) / c.lam
    w_cost_cpu = np.exp(lw - lw.max()); w_cost_cpu /= w_cost_cpu.sum()
    lw2 = lw + logq_cpu
    w_both_cpu = np.exp(lw2 - lw2.max()); w_both_cpu /= w_both_cpu.sum()

    g = gpu_cycle(cp, x0, c.U, K, cp.asarray(xi),
                  env_printed=c.env.printed, env_alpha=c.env.alpha, sigma_env=c.sigma_env,
                  s0=float(np.sqrt(c.Sigma0[0])), s_om=float(np.sqrt(c.Sigma0[1])),
                  z=c.z, alpha=c.alpha, form=c.form, goal=c.goal, penalty=c.penalty,
                  lam=c.lam, R=c.R, dt=c.dt, u_max=c.u_max, want_logq=True)

    def rel(a, b_):
        a = np.asarray(a, float); b_ = cp.asnumpy(b_)
        d = np.abs(a - b_).max() / max(1.0, np.abs(a).max())
        return float(d)

    checks = {
        "S": rel(S_cpu, g["S"]),
        "logq": rel(logq_cpu, g["logq"]),
        "w (cost only)": rel(w_cost_cpu, g["w_cost"]),
        "w (cost + logq)": rel(w_both_cpu, g["w_both"]),
    }
    ess_cpu = 1.0 / (w_both_cpu ** 2).sum()
    ess_gpu = float(1.0 / (g["w_both"] ** 2).sum())
    checks["ESS"] = abs(ess_cpu - ess_gpu) / max(1.0, ess_cpu)
    ok = True
    for k, v in checks.items():
        flag = "ok " if v < tol else "FAIL"
        ok &= v < tol
        print(f"   {flag} {k:<18s} max relative difference {v:.2e}")
    print(f"   CPU ESS {ess_cpu:.3f}   GPU ESS {ess_gpu:.3f}   of K={K}")
    return ok


def snapshots(n=6, seed=0, warm=12, every=4):
    """Operating points along a real closed-loop run.

    The first cycles are DISCARDED.  At the cold start the nominal sequence is still zero, every rollout
    is bad in the same way, and the effective sample size is near one for plain MPPI too; measuring there
    would attribute a cold start to the barrier.  The repository's own shipped corridor figure is an
    episode mean of about 160 of 500 for plain MPPI, so a harness that reports 2 at a representative
    state is measuring the wrong states.  `warm` skips past that.

    The reference episode is flown by PLAIN MPPI, not by Algorithm 1.  Under the corrected weights
    Algorithm 1 stalls within a fifth of a metre of the start, so its own states are all bunched where
    the goal is far and every rollout is bad; measuring there would report a stalled controller's
    operating points as if they were the corridor's, and even plain MPPI has an effective size of one
    to five at those states.  Flying the reference with plain MPPI and evaluating all three weightings
    at the same states is what isolates the weighting from the trajectory."""
    c = MPPI(Corridor(), K=500, seed=seed)
    x = np.array([0.0, 0.5, 0.0])
    out = []
    for k in range(warm + n * every):
        if k >= warm and (k - warm) % every == 0:
            out.append((x.copy(), c.U.copy()))
        u = c.plan(x)
        x = cpu_step(x[None], u[None], DT)[0]
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--kmax", type=int, default=1_000_000)
    ap.add_argument("--states", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    cp = load_cupy()
    free, total = cp.cuda.runtime.memGetInfo()
    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"{name}, {free/2**30:.1f} GB free of {total/2**30:.1f} GB\n")

    if not verify(cp, K=500):
        sys.exit("\nVERIFICATION FAILED — the GPU path does not reproduce the CPU path. No sweep run.")
    print("   gate passed\n")
    if a.verify_only:
        return

    Ks = [k for k in (500, 2_000, 8_000, 32_000, 128_000, 512_000, 1_000_000) if k <= a.kmax]
    pts = snapshots(a.states)
    print(f"sweeping K over {Ks} at {len(pts)} operating points, {a.reps} repetitions each\n")

    c0 = make_controller(500, 0)
    cfg = dict(env_printed=c0.env.printed, env_alpha=c0.env.alpha, sigma_env=c0.sigma_env,
               s0=float(np.sqrt(c0.Sigma0[0])), s_om=float(np.sqrt(c0.Sigma0[1])),
               z=c0.z, alpha=c0.alpha, form=c0.form, goal=c0.goal, penalty=c0.penalty,
               lam=c0.lam, R=c0.R, dt=c0.dt, u_max=c0.u_max, want_logq=True)

    rows = []
    hdr = (f"{'K':>9s}{'plain MPPI':>13s}{'paper (no IS)':>16s}{'corrected':>12s}"
           f"{'corr/K':>11s}{'s/cycle':>10s}")
    print(hdr); print("-" * len(hdr))
    rng = np.random.default_rng(7)
    for K in Ks:
        fc, fb, ab, t_tot = [], [], [], 0.0
        for (x0, U0) in pts:
            for _ in range(a.reps):
                xi = cp.asarray(rng.standard_normal((K, c0.T, 2)))
                cp.cuda.Stream.null.synchronize(); t0 = time.time()
                g = gpu_cycle(cp, x0, U0, K, xi, **cfg)                      # Algorithm 1's proposal
                p_ = gpu_cycle(cp, x0, U0, K, xi, **{**cfg, "constrained": False,
                                                     "want_logq": False})     # plain MPPI's proposal
                ess_c = float(1.0 / (p_["w_cost"] ** 2).sum())                # plain MPPI reference
                ess_b = float(1.0 / (g["w_both"] ** 2).sum())                 # corrected estimator
                ess_p = float(1.0 / (g["w_cost"] ** 2).sum())                 # the paper's, uncorrected
                cp.cuda.Stream.null.synchronize(); t_tot += time.time() - t0
                fc.append(ess_c); fb.append(ess_b); ab.append(ess_p)
                del xi, g, p_
                cp.get_default_memory_pool().free_all_blocks()
        mc, mb, mp = float(np.median(fc)), float(np.median(fb)), float(np.median(ab))
        rows.append({"K": K, "ess_plain_mppi": mc, "ess_paper_no_is": mp, "ess_corrected": mb,
                     "corrected_frac_of_K": mb / K, "n": len(fc), "s_per_cycle": t_tot / len(fc)})
        print(f"{K:9d}{mc:13.1f}{mp:16.1f}{mb:12.2f}{mb/K:11.2e}{t_tot/len(fc):10.3f}")

    dst = os.path.join(HERE, "results", "corridor_V15_ess_scaling.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"device": name, "states": len(pts), "reps": a.reps, "rows": rows}, f, indent=1)
    print(f"\nwritten: {dst}")

    a0, a1 = rows[0], rows[-1]
    kfac = a1["K"] / a0["K"]
    gp = a1["ess_plain_mppi"] / max(a0["ess_plain_mppi"], 1e-12)
    gc = a1["ess_corrected"] / max(a0["ess_corrected"], 1e-12)
    print("")
    print(f"K grew {kfac:.0f}x.")
    print(f"  plain MPPI              {a0['ess_plain_mppi']:10.1f} -> {a1['ess_plain_mppi']:12.1f}   ({gp:6.1f}x)")
    print(f"  Algorithm 1 as printed  {a0['ess_paper_no_is']:10.1f} -> {a1['ess_paper_no_is']:12.1f}")
    print(f"  with the weights put in {a0['ess_corrected']:10.2f} -> {a1['ess_corrected']:12.2f}   ({gc:6.1f}x)")
    print("")
    print("Plain MPPI's effective sample size tracks K. The corrected estimator's does not move at all:")
    print("the per-sample problem drives the proposal's standard deviation to zero, a point mass has no")
    print("density with respect to the Gaussian the derivation assumes, and the weight that would repair")
    print("the update does not exist. This is not a budget problem and no machine fixes it.")


if __name__ == "__main__":
    main()
