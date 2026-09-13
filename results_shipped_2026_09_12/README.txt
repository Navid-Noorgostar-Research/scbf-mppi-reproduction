The result files exactly as they were first produced, on 12 Sep 2026, before the dependency stack was
pinned (generated with the then-unpinned requirements.txt, i.e. an older NumPy / Clarabel).  Every
number reported for that stack comes from THIS tree.

results/ in the repository root is the same set regenerated on 14 Sep 2026 with the now-pinned stack
(requirements.txt).  The difference between the two is the subject of tests/drift_report.py and of
REPRODUCIBILITY.md.  Nothing here is ever overwritten by a re-run.
