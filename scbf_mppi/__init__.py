"""Reproduction of Tao, Yoon, Kim, Hovakimyan, Voulgaris,
'Path Integral Methods with Stochastic Control Barrier Functions' (arXiv:2206.11985 / CDC 2022),
together with a referee-style review of its derivations and stated claims, run here as experiments.

Modules
  env        corridor safe set, barrier functions, Lie derivatives, Ito terms, collision tests
  dynamics   unicycle with additive Brownian noise, Euler-Maruyama, vectorised rollouts
  scbf       per-sample chance-constraint solvers (variance form as printed, std form as corrected),
             exact cvxpy solvers for validation and wall-clock timing
  mppi       MPPI (Williams form), SCBF-MPPI (Algorithm 1), mean-shift-only variant,
             deterministic MPPI with covariance annealing (Homburger et al., L-CSS 2025, Alg. 1)
  simulate   closed-loop episodes and metrics (collision rate, TTF, activation, min h, bridge crossings)
  experiments E1..E10 and figures
  animate    side-by-side animation
"""
__version__ = "1.0.0"

# Windows consoles default to cp1252, and several of these modules print mathematical minus signs and
# Greek letters; without this the self-tests die with UnicodeEncodeError before printing a single result.
# Reconfiguring here (rather than asking every user to set PYTHONUTF8=1) makes `python -m scbf_mppi.*`
# work out of the box on a stock Windows install.
import sys as _sys
for _stream in ("stdout", "stderr"):
    try:
        getattr(_sys, _stream).reconfigure(encoding="utf-8")
    except Exception:                                          # pragma: no cover - not a tty, or Python < 3.7
        pass
del _sys, _stream
