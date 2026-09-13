"""One command that reproduces this package, in the order a reviewer would want it.

    python reproduce.py                 # ~3 min: environment, self-tests, bit-exactness guard
    python reproduce.py --full          # ~90 min: the above, then every experiment and every figure
    python reproduce.py --exp vessel    # environment, self-tests, then only the vessel experiments

It is deliberately boring: it prints what it is about to do, runs it, and stops at the first failure with
the command that failed, so that a reader can run that command alone.  Everything it calls is a module of
this package -- there is no build step and nothing is downloaded.

Why it exists.  This package argues that a published result should be checkable, so it has to be
checkable itself.  Three things are checked, in increasing strength:

  1. self-tests        the model, the barrier rows and the per-sample solver against independent references
                       (MATLAB trajectories from the group's repository, finite differences, cvxpy)
  2. regression        twelve shipped configurations must reproduce BIT FOR BIT, including a SHA-256 of the
                       closed-loop trajectory.  This is what makes it safe to add code to a package whose
                       numbers have already been reported.
  3. experiments       the full set, re-deriving every reported number from scratch

Note on run-to-run identity.  The rollouts are seeded and the integrator is deterministic, so results are
reproducible to the last bit ON THE SAME STACK.  They are not reproducible across library versions: the
barrier path calls a convex solver, and its iterates move between Clarabel releases.  requirements.txt is
pinned for exactly this reason, and REPRODUCIBILITY.md measures how far the numbers moved when it was not.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

PINNED = {"numpy": "2.5.3", "scipy": "1.18.1", "matplotlib": "3.11.2",
          "cvxpy": "1.9.2", "clarabel": "0.11.1"}


def run(label, args, env=None, optional=False):
    print(f"\n=== {label}\n    {' '.join(args)}", flush=True)
    t = time.time()
    e = dict(os.environ)
    e.setdefault("PYTHONUTF8", "1")
    if env:
        e.update(env)
    r = subprocess.run(args, cwd=HERE, env=e)
    dt = time.time() - t
    if r.returncode != 0:
        print(f"    FAILED after {dt:.0f}s (exit {r.returncode})", flush=True)
        if optional:
            print("    (optional step -- continuing)", flush=True)
            return False
        print(f"\nStopped.  Re-run just this step with:\n    {' '.join(args)}")
        sys.exit(r.returncode)
    print(f"    ok  ({dt:.0f}s)", flush=True)
    return True


def check_environment():
    print("=== environment")
    print(f"    python      {sys.version.split()[0]}  ({sys.platform})")
    bad = []
    for mod, want in PINNED.items():
        try:
            m = __import__(mod)
            got = getattr(m, "__version__", "?")
        except Exception as exc:                                   # pragma: no cover
            got = f"MISSING ({exc.__class__.__name__})"
        mark = "  " if got == want else " <- differs from requirements.txt"
        if got != want:
            bad.append((mod, want, got))
        print(f"    {mod:<12s}{got:<12s}{mark}")
    try:
        import numpy as np
        print(f"    BLAS        {np.__config__.get_info('blas_opt_info').get('libraries', ['?'])[0]}"
              if hasattr(np.__config__, "get_info") else "    BLAS        (numpy >= 2, see numpy.show_config())")
    except Exception:
        pass
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(v):
            print(f"    {v} = {os.environ[v]}")
    if bad:
        print("\n    NOTE: the stack differs from the pinned one.  The self-tests and the pure-MPPI numbers")
        print("    will still match; the barrier numbers may move by a fraction of their confidence")
        print("    interval.  REPRODUCIBILITY.md records how much when this was last measured.")
    return not bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also run every experiment and rebuild every figure")
    ap.add_argument("--exp", default=None, choices=["corridor", "vessel", "ext", "all"],
                    help="run only this group of experiments")
    ap.add_argument("--runs", type=int, default=30, help="seeds per configuration")
    ap.add_argument("--procs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()

    t0 = time.time()
    clean = check_environment()

    run("self-test: corridor (solver vs cvxpy, bridge crossing law, Theorem 2)",
        [PY, "-m", "scbf_mppi.selftest"])
    run("self-test: vessel (Solgenia model vs the group's MATLAB, barrier rows, chance constraint)",
        [PY, "-m", "scbf_mppi.vessel.selftest"])
    run("self-test: the V9-V12 additions (defaults identical to shipped, rows under a current, filter, observer)",
        [PY, "-m", "scbf_mppi.vessel.ext.selftest_ext"])
    run("regression: twelve shipped configurations, bit for bit",
        [os.path.join(os.path.dirname(PY), "python.exe") if os.name == "nt" else PY,
         os.path.join("tests", "test_regression.py"), "check"],
        env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        optional=not clean)

    groups = ["corridor", "vessel", "ext"] if (a.full or a.exp == "all") else ([a.exp] if a.exp else [])
    for g in groups:
        if g == "corridor":
            run("experiments: corridor E1-E14", [PY, "-m", "scbf_mppi.experiments", "--exp", "all",
                                                 "--runs", str(a.runs), "--procs", str(a.procs)])
            run("figures: corridor", [PY, "-m", "scbf_mppi.figures"])
        elif g == "vessel":
            run("experiments: vessel V0-V8", [PY, "-m", "scbf_mppi.vessel.experiments", "--exp", "all",
                                              "--runs", str(a.runs), "--procs", str(a.procs)])
            run("figures: vessel", [PY, "-m", "scbf_mppi.vessel.figures"])
        elif g == "ext":
            run("experiments: V9-V12 (heading cost, output filter, current observer, adaptive temperature)",
                [PY, "-m", "scbf_mppi.vessel.ext.experiments_ext", "--exp", "all",
                 "--runs", str(a.runs), "--procs", str(a.procs)])
            run("figures: V9-V12", [PY, "-m", "scbf_mppi.vessel.ext.figures_ext"], optional=True)

    print(f"\n=== done in {(time.time()-t0)/60:.1f} min")
    if not groups:
        print("    (self-tests and the regression guard only -- add --full to re-run the experiments)")
