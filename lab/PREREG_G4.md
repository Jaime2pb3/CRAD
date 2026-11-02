# Hardware replication of the September preregistered test

Written 2026-11-02, BEFORE running. Nothing in PREREG.md changes.

Hypothesis, feature, statistic, test, alpha and decision rules: exactly PREREG.md, applied by analyze_prereg.py revision 2.
Data: seeds 43 to 46, 48 draws, on a Colab G4, fresh runs.
Configs: lab/configs, rebuilt from the session that wrote the originals.

Reading:
- H1 REJECTED: the September rejection holds on other hardware.
- H1 SUPPORTED: it does not hold. Report it as a conflict, not as a new finding.
- INCONCLUSIVE: no claim.

Draws are not compared one to one with September, because sampled outcomes at T=0.7 differ between GPUs.
Any other feature is exploratory and carries no claim.
