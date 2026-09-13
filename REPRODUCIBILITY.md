# What "it reproduces" means in this package

This package argues that a published result should be checkable, so it has to hold itself to that. This note says exactly what reproduces, what does not, and where the boundary is. All
numbers below are produced by the two scripts named, on the machine described at the end; nothing here is
asserted from memory.

There are three different claims hiding inside the word "reproducible", and they have three different
answers.

| claim | tool | answer |
|---|---|---|
| same code, same pinned stack, same seed → same bits | `tests/test_regression.py` | yes, always |
| same code, **different** library versions, same seed → same run | `tests/drift_report.py` | only without the solver |
| same code, different library versions → same conclusion | `tests/drift_report.py` | yes, 92 % of quantities |

## 1. On a pinned stack: bit for bit

```
python tests/test_regression.py check
```

Twelve configurations covering every code path — plain MPPI, the second-order barrier, the importance-sampling
correction, the relative-degree-1 rows as printed, the variance form, the deterministic planner, the moving
ferry and the unknown current — are re-run and compared against a stored reference. The comparison includes a
SHA-256 of the raw float64 bytes of the closed-loop trajectory, the control sequence and the barrier series,
so a single changed bit anywhere in the loop fails the check.

This is what makes it safe to keep developing a package whose numbers have already been reported. Every addition
made for V9–V12 (a heading term, an output filter, a current observer, an adaptive temperature) was written
against this guard, and the guard passed unchanged after each one.

## 2. Across library versions: only the solver-free paths

`requirements.txt` originally carried floors (`numpy>=1.26`, `clarabel>=0.6`). A fresh install two days after the
first result set therefore produced a different stack, and the package was re-run on it. Both result trees are kept:

* `results_shipped_2026_09_12/` — the numbers first reported, and quoted in the referee-style review
* `results/` — the same experiments regenerated on the pinned stack of 14 Sep 2026

```
python tests/drift_report.py
```

compares them seed by seed. The same seed is the same deterministic computation, so any disagreement is the
stack, not sampling noise. Of 85 configurations:

| | configurations |
|---|---|
| identical to the last bit on every shared seed | 5 |
| differ only at rounding level (relative < 1e-6) | 11 |
| materially different on at least one seed | 69 |

The split is not random. **Every one of the 16 reproducible configurations is one that never calls the
per-sample chance-constraint solver**: plain MPPI at each sample count, MPPI under bounded speed, MPPI under
the gust and the current, the relative-degree-1 rows exactly as printed (which reduce to MPPI, because
`L_g h = 0` for a force input — that is the finding of experiment V1), and the variance-only SCBF variant,
which shrinks the covariance without shifting the mean.

Everything that solves for a shifted mean moves. The mechanism is visible in the size of the differences:

* MPPI, worst relative change over all seeds: **1.3e-7** — floating-point rounding from a different BLAS.
  It becomes visible at all only because time-to-goal is a threshold crossing, so a run can finish one
  control step earlier (seed 4 of `MPPI K=200`: 110 s → 109 s).
* SCBF-MPPI on the vessel, seed 0, closest approach: **0.652 m → 0.561 m**. That is not rounding. The
  interior-point iterates differ between Clarabel releases, the shifted mean differs, and from there the two
  runs are different trajectories.

Worst single-seed changes across the whole set: 176 s in time-to-goal (deterministic MPPI, corridor), 0.45 in
collision rate (SCBF variance form at sigma = 0.5), and closest-approach swings of metres.

## 3. The conclusions survive

Comparing summary means with their bootstrap intervals, restricted to the 78 configurations that were run
with the same number of seeds in both trees:

* **385 of 417** summary quantities have each tree's mean inside the other's 95 % interval.
* **32** do not.

So the aggregate statements this package makes are stable; the individual runs behind them are not. That is the
honest position, and it is worth saying out loud rather than hiding: a method whose output depends on an
interior-point solver's iterates is reproducible **in distribution**, not run by run, unless the solver
version is pinned.

## 4. The specific numbers that are fragile

These should be given with a caveat, or given to fewer
digits, because they moved between stacks.

1. **Crossing ferry, SCBF-MPPI + IS correction: "collision rate 0.000, 0 % of runs touch a circle."**
   On the current stack this is **0.0034, and 3 of 30 runs touch**. The qualitative claim (the corrected
   variant is the safest of the four) holds; the exact zero does not. Say "no collisions in 30 seeds on the
   stack the shipped results were made with, 3 of 30 on a newer solver" or simply "the lowest of the four".
2. **Deterministic MPPI time-to-goal in the corridor sweep** (`E13`). Per-seed changes of up to 176 s. This
   controller runs at an effective sample size of 2–5, so one or two samples decide the update and a rounding
   difference flips which. Its collision rates are stable; its times are not.
3. **Closest approach (`min h`) for any SCBF variant in the corridor.** It is a minimum over a whole run —
   an extreme-value statistic set by a single worst excursion in a single seed — and it swings by metres.
   The vessel `min h` values are stable, because the vessel runs keep a real margin; the corridor ones sit
   near zero where the minimum is decided by one step.
4. **Gust scenario, SCBF-MPPI + IS: "collision rate 0.00073."** Now 0.000. This one moved in the reassuring
   direction, which is exactly why it should be reported as an interval rather than a point.

Everything else reported here — the vessel table (V2), the crossing table (V4) apart from item 1 above, the
relative-degree finding (V1), the wall-clock figures, the effective-sample-size figures, and every pure-MPPI
baseline — is consistent between the two stacks.

## 5. What was changed so this does not happen again

* `requirements.txt` is pinned with `==` to the stack that produced `results/`; `requirements-lock.txt` is the
  full freeze of the environment, including transitive dependencies.
* `results_shipped_2026_09_12/` is kept in the repository and is never overwritten by a re-run, so the claim
  "the reported numbers match the code" stays checkable after any future run.
* `tests/test_regression.py` guards the code against accidental change, independently of the library versions.
* `tests/drift_report.py` re-derives every number in this note, so it can be re-checked rather than believed.

Two portability bugs found while doing this are also fixed: `scbf_mppi/experiments.py` called `os.uname()`,
which does not exist on Windows and killed the corridor sweep at E3; and the self-tests printed mathematical
minus signs into a cp1252 console and died with a `UnicodeEncodeError` before printing a single result. The
console encoding is now set in `scbf_mppi/__init__.py`, so `python -m scbf_mppi.selftest` works on a stock
Windows install without setting `PYTHONUTF8`.

## 6. The machine

```
Windows 11 Pro 26200, 16 logical cores, 64 GB RAM, NVIDIA RTX 3080 Laptop (16 GB)
Python 3.12.6 (CPython, win_amd64)
numpy 2.5.3   scipy 1.18.1   matplotlib 3.11.2   cvxpy 1.9.2   clarabel 0.11.1
OMP_NUM_THREADS=1  MKL_NUM_THREADS=1  OPENBLAS_NUM_THREADS=1   (set for every run in this note)
```

The thread settings matter: a multi-threaded BLAS changes the order of floating-point reductions, which is a
second, independent source of the rounding-level differences of section 2. Pin them when you want the
bit-exact guarantee.
