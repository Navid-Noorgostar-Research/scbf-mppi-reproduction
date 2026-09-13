"""Thirty-second self-test:  python -m scbf_mppi.selftest

Checks, in order, that
  1. the per-sample constraint solver agrees with the exact conic optimum (cvxpy + Clarabel);
  2. the Brownian-bridge crossing test reproduces the closed-form crossing probability
     P(bridge from 0 to b crosses the level -d) = exp(-2 d (d + b) / (sigma^2 dt)), averaged over b > -d;
  3. the delivered probability of the printed constraint is Phi(alpha s), not 1 - delta (Theorem 2 check);
  4. one short closed-loop episode of MPPI and of SCBF-MPPI runs and reports the standard metrics.
Exit code 0 when all pass."""
import sys, time
import numpy as np
from scipy.stats import norm

from . import scbf
from .env import Corridor
from .dynamics import DT
from .mppi import MPPI, SCBFMPPI
from .simulate import run_episode


def check_solver():
    t = time.time()
    out = scbf.validate_against_cvxpy(120, form="std", z=2.748)
    ok = out["max_abs_gap"] < 5e-3
    print(f"[1] solver vs cvxpy: n={out['n']} active={out['n_active']} max|gap|={out['max_abs_gap']:.2e} "
          f"({time.time()-t:.1f}s) -> {'ok' if ok else 'FAIL'}")
    return ok


def check_bridge(sigma=0.1, dt=DT, d=0.03, n=4000, seed=0, n_sub=200):
    """Straight wall y = 0 at distance d below the start point; drift-free 1-D bridge.
    A discretised bridge can only under-count crossings, so the estimate converges to the exact value from below
    as n_sub grows (n_sub = 20 in the experiments gives a conservative count; 200 here)."""
    rng = np.random.default_rng(seed)
    hits = 0; h = dt / n_sub
    for _ in range(n):
        B = 0.0; tau = 0.0; B_end = sigma * np.sqrt(dt) * rng.standard_normal()
        crossed = False
        for j in range(1, n_sub):
            rem = dt - tau
            mean = B + (B_end - B) * (h / rem)
            var = sigma ** 2 * h * (rem - h) / rem
            B = mean + np.sqrt(max(var, 0.0)) * rng.standard_normal()
            tau += h
            if d + B < 0.0:
                crossed = True; break
        # endpoints must both be inside for a "between-sample" crossing
        if d + B_end > 0.0 and crossed:
            hits += 1
    # exact: P(min < -d | B_end = b) = exp(-2 d (d + b) / (sigma^2 dt)); average over b with b > -d
    b = sigma * np.sqrt(dt) * rng.standard_normal(200_000)
    exact = np.mean(np.where(b > -d, np.exp(-2 * d * (d + b) / (sigma ** 2 * dt)), 0.0))
    mc = hits / n
    ok = (mc <= exact + 0.01) and (mc > 0.8 * exact - 0.01)
    print(f"[2] Brownian bridge crossing (n_sub={n_sub}): MC {mc:.3f} vs exact {exact:.3f} (MC converges from below) -> {'ok' if ok else 'FAIL'}")
    return ok


def check_theorem2(delta=0.003, s=0.5, c=1.0, n=200_000, seed=1):
    """Constraint c*(m + s*xi) >= b with xi ~ N(0,1). Printed form sets the margin alpha*c^2*s^2,
    the correct form z*|c|*s. Delivered probability of the printed form is Phi(alpha*s) (for c = 1)."""
    z = norm.ppf(1 - delta); alpha = z
    rng = np.random.default_rng(seed); xi = rng.standard_normal(n)
    pr_printed = np.mean(alpha * c * c * s * s + c * s * xi >= 0.0)   # m chosen so that equality holds
    pr_correct = np.mean(z * abs(c) * s + c * s * xi >= 0.0)
    ok = abs(pr_printed - norm.cdf(alpha * s)) < 5e-3 and abs(pr_correct - (1 - delta)) < 5e-3
    print(f"[3] Theorem 2: printed form delivers {pr_printed:.3f} (= Phi(alpha s) = {norm.cdf(alpha*s):.3f}); "
          f"corrected form {pr_correct:.3f} (claimed {1-delta:.3f}) -> {'ok' if ok else 'FAIL'}")
    return ok


def check_episode():
    env = Corridor()
    t = time.time()
    m = MPPI(env, K=200, T=20, lam=1.0, sigma_v=1.0, sigma_om=1.5)
    r1 = run_episode(m, env, sigma=0.1, seed=0, max_steps=60, bridge=False)
    s_ = SCBFMPPI(env, sigma_env=0.1, delta=0.003, form="variance", K=200, T=20, lam=1.0, sigma_v=1.0, sigma_om=1.5)
    r2 = run_episode(s_, env, sigma=0.1, seed=0, max_steps=60, bridge=False)
    act = r2["activation_frac_mean"]; vr = r2["var_ratio_mean"]
    ok = np.isfinite(r1["collision_rate"]) and np.isfinite(r2["collision_rate"]) and act is not None and 0 <= act <= 1
    print(f"[4] 60-step episodes (K=200): MPPI collision {r1['collision_rate']:.3f}, min h {r1['min_h']:.2f}; "
          f"SCBF-MPPI collision {r2['collision_rate']:.3f}, min h {r2['min_h']:.2f}, active {100*act:.0f} %, "
          f"Var[dv]/Var0 {100*vr:.0f} % ({time.time()-t:.1f}s) -> {'ok' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [check_solver(), check_bridge(), check_theorem2(), check_episode()]
    print("all passed" if all(results) else "FAILED")
    sys.exit(0 if all(results) else 1)
