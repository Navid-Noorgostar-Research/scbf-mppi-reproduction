r"""FINDINGS section 4 claims the selection the restored weights perform is driven by the log s term.
Only variation ACROSS samples can select, so decompose the per-sample log density ratio and compare the
spread of each piece.  With sigma_0 = 1 and ev = m + s*xi,

    log p/q = sum_t [ -0.5*((m + s*xi)/s0)^2 + log s0 ]  +  sum_t [ 0.5*xi^2 ]  +  sum_t [ -log s ]
              |__ mean-shift __|                            |__ noise __|          |__ shrink __|

This replays SCBFMPPI.sample_eps exactly (same solve_rows call, same RNG stream) so the total it rebuilds
must equal the controller's own logq.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scbf_mppi.env import Corridor
from scbf_mppi.mppi import SCBFMPPI
from scbf_mppi import scbf as scbf_mod
from scbf_mppi.dynamics import step, DT

env = Corridor()
c = SCBFMPPI(env, K=500, T=20, lam=1.0, sigma_v=1.0, sigma_om=1.5,
             sigma_env=0.1, delta=0.003, form="std", is_correction=True, seed=0)
# plan() pops logq out of info before it reaches c.last, so stash it on the way through
_orig_sample = c.sample_eps
_stash = {}
def _wrapped(x0):
    eps, traj, info = _orig_sample(x0)
    if "logq" in info:
        _stash["logq"] = np.array(info["logq"], copy=True)
    return eps, traj, info
c.sample_eps = _wrapped

plant = np.random.default_rng(10_000)
x = np.array([0.0, 0.5, 0.0])
s0 = float(np.sqrt(c.Sigma0[0]))
out = []
for k in range(60):
    # replay one planning cycle's proposal, from the controller's state BEFORE it plans
    U0 = c.U.copy(); rng_state = c.rng.bit_generator.state
    K, T = c.K, c.T
    X = np.broadcast_to(x, (K, 3)).copy()
    xi = np.random.default_rng(); xi.bit_generator.state = rng_state
    XI = xi.standard_normal((K, T, 2))
    shift = np.zeros(K); noise = np.zeros(K); shrink = np.zeros(K)
    for t in range(T):
        cc, bb = env.scbf_rows(X, c.sigma_env)
        res = scbf_mod.solve_rows(np.full(K, U0[t, 0]), cc, bb, s0, z=c.z, alpha=c.alpha, form=c.form)
        m, s = res["m"], res["s"]
        ev = m + s * XI[:, t, 0]
        ss = np.maximum(s, 1e-9)
        shift += -0.5 * (ev / s0) ** 2 - np.log(s0)
        noise += 0.5 * ((ev - m) / ss) ** 2
        shrink += np.log(ss)
        U = np.stack([U0[t, 0] + ev, U0[t, 1] + np.sqrt(c.Sigma0[1]) * XI[:, t, 1]], -1)
        X = step(X, U, c.dt)
    total = shift + noise + shrink
    u = c.plan(x)                                  # the real cycle, which consumes the same draws
    if k >= 12 and k % 8 == 0:
        lq = _stash["logq"]
        err = float(np.abs(total - lq).max())
        out.append((k, shift.std(), noise.std(), shrink.std(), total.std(),
                    float(np.corrcoef(shrink, total)[0, 1]), err))
    x = step(x[None], u[None], DT)[0] + 0.1 * np.sqrt(DT) * plant.standard_normal(3)

print(f"  {'cycle':>6s}{'sd(shift)':>11s}{'sd(noise)':>11s}{'sd(log s)':>11s}{'sd(total)':>11s}"
      f"{'corr(log s,tot)':>17s}{'rebuild err':>13s}")
print("  " + "-"*80)
for k, a, b, d, t, r, e in out:
    print(f"  {k:6d}{a:11.3f}{b:11.3f}{d:11.3f}{t:11.3f}{r:17.3f}{e:13.2e}")
A = np.array([o[1:] for o in out])
print(f"\n  means:  sd(shift) {A[:,0].mean():.2f}   sd(noise) {A[:,1].mean():.2f}   "
      f"sd(log s) {A[:,2].mean():.2f}   sd(total) {A[:,3].mean():.2f}")
print(f"  mean corr(log s, total) = {A[:,4].mean():+.3f};  worst rebuild error vs the controller's own logq = {A[:,5].max():.1e}")
