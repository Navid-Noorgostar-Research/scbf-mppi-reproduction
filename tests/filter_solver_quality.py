"""How often does the CBF-QP filter's solver return something it is not sure about, and does it matter?

    python tests/filter_solver_quality.py --episodes 6

Clarabel occasionally reports "solved_inaccurate" on the filter's little three-variable problem, and cvxpy
prints a UserWarning when it does.  Python's warning filter prints that once per worker process, so the
count in an experiment log says nothing about the rate.  This script measures it directly.

For every control instant of a set of closed-loop episodes it records the solver status, and then re-checks
the returned input against the raw barrier rows, in sampling-standard-deviation units (the rows have
coefficients of order 3e-4 per newton, so an absolute tolerance would be meaningless).  Three things are
reported:

  status distribution      how often the solver was confident
  post-filter margin       the smallest row margin AFTER filtering, excluding instants where the exact
                           penalty had to use slack because the constraint set was genuinely empty
  input-set violation      how far outside the 700 N azimuth disc or the bow cap the returned input was
                           before the final projection

The third should be exactly zero by construction, because `CBFQPFilter.__call__` ends with
`sg.clip_inputs`; it is measured anyway, because a filter that silently emits an input the vessel cannot
produce would invalidate the whole V10 comparison.
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from scbf_mppi.vessel import solgenia as sg
from scbf_mppi.vessel.ext.controllers_ext import make_controller_ext
from scbf_mppi.vessel.simulate import run_episode


def audit(cfg, seeds):
    stats = {"status": Counter(), "margins": [], "slack_used": 0, "instants": 0,
             "cone_excess": 0.0, "bow_excess": 0.0, "active": 0}

    for seed in seeds:
        c, H = make_controller_ext(cfg, seed)
        filt = c.filt
        inner_call = filt.__call__

        def wrapped(x, t, u_ref, _f=filt, _orig=inner_call):
            u, info = _orig(x, t, u_ref)
            stats["instants"] += 1
            if info["filter_active"]:
                stats["active"] += 1
                stats["status"][str(_f.prob.status)] += 1
                if info["filter_slack"] > 1e-6:
                    stats["slack_used"] += 1
                else:
                    a, b, _, _ = _f.H.rows(np.asarray(x, float)[None], t)
                    tight = (_f.z * np.linalg.norm(a[0] * _f.sig_d[None, :], axis=-1)
                             if _f.z else np.zeros(b.shape[1]))
                    g = np.maximum(np.linalg.norm((a[0] @ sg.B_ALLOC) * _f.s0[None, :], axis=1), 1e-12)
                    m = (a[0] @ (sg.B_ALLOC @ u) - b[0] - tight) / g
                    stats["margins"].append(float(m.min()))
            stats["cone_excess"] = max(stats["cone_excess"],
                                       float(np.hypot(u[0], u[1]) - sg.F_AT_MAX))
            stats["bow_excess"] = max(stats["bow_excess"],
                                      float(abs(u[2]) - sg.bow_cap(np.asarray(x, float)[3])))
            return u, info

        filt.__call__ = wrapped
        c.filt = filt
        # FilteredController calls self.filt(...), which resolves to the instance attribute we just set
        run_episode(c, H, seed=seed, disturbance=cfg.get("disturbance", "white"),
                    current=tuple(cfg.get("current", (0.0, 0.0))), max_time=150.0)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    a = ap.parse_args()
    seeds = list(range(a.episodes))

    for label, cfg in [
        ("nominal filter, crossing ferry",
         dict(name="q", kind="mppi_filter", filter=dict(delta=None), scenario="crossing", disturbance="white")),
        ("chance filter, crossing ferry",
         dict(name="q", kind="mppi_filter", filter=dict(delta=0.003), scenario="crossing", disturbance="white")),
        ("chance filter, unknown current",
         dict(name="q", kind="mppi_filter", filter=dict(delta=0.003), scenario="static",
              disturbance="ou", current=(0.0, -0.5))),
    ]:
        s = audit(cfg, seeds)
        m = np.array(s["margins"]) if s["margins"] else np.array([np.nan])
        print(f"\n=== {label}   ({a.episodes} episodes)")
        print(f"    control instants                 {s['instants']}")
        print(f"    filter active                    {s['active']}  ({s['active']/max(s['instants'],1):.1%})")
        print(f"    of those, slack was needed       {s['slack_used']}  "
              f"({s['slack_used']/max(s['active'],1):.1%})")
        print(f"    solver status                    {dict(s['status'])}")
        print(f"    worst post-filter row margin     {np.nanmin(m):+.2e} sampling std")
        print(f"    margins below -1e-6              {int((m < -1e-6).sum())} of {len(m)}")
        print(f"    azimuth disc exceeded by         {s['cone_excess']:.2e} N")
        print(f"    bow cap exceeded by              {s['bow_excess']:.2e} N")
