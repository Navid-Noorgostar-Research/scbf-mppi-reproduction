"""Additions to the vessel study that are NOT part of the reproduction of arXiv:2206.11985.

Everything in this package is new work built on top of the reproduction. It changes nothing in the shipped experiments: `tests/test_regression.py` re-runs twelve shipped
configurations and requires bit-for-bit identical trajectories.

  costs.py       a heading term for the vessel running cost           -> experiment V9
  filters.py     the barrier as an output safety filter (CBF-QP)      -> experiment V10
  observer.py    an on-line estimate of the unknown water current     -> experiment V11
  adaptive.py    a temperature that holds the effective sample size   -> experiment V12
"""
