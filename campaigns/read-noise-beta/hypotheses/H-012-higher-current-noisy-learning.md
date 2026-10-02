---
id: "H-012"
title: "Increasing current amplification beyond ours improves noisy learning"
---
# H-012 — Higher current amplification and noisy learning

Prospective September 28, 2026, motivated by [X-010](../explorations/X-010-higher-current-amplification.md).
For at least one Conv1/2/3 cell at eta=1e-5 or 1e-4, A=4,B=2 with exactly
2x the ours weight-rate vector exceeds ours' fixed-horizon validation accuracy.
Use matched initial output D/F=1/4/6, seed0, centered EqProp, relative endpoint
noise and uninterrupted Adam. Horizons are 10/30/30 epochs. Report legacy as a
second comparator and all clean penalties, noisy degradation and failures.

A positive paired gap supports a sampled exploratory advantage; a gain >=0.2 pp
prioritizes later independent-seed confirmation. This practical threshold is
proposed, not a significance criterion. No favorable finite cell contradicts
the sampled claim; incomplete/invalid comparisons remain inconclusive. One seed
cannot establish generality. This tests the selected amplification-plus-LR
policy, not amplification independent of learning rate. Cosine improvement alone
does not support the training claim.
