---
id: "H-007"
title: "Initialization-calibrated beta changes its effective displacement during training"
---
# Fixed beta across training checkpoints

The same physical nudging coefficient may produce different relative output
displacements as the trained state and loss-gradient forcing change. Therefore
initialization alignment need not predict alignment later. Small beta may also
be vulnerable to read noise, whereas large nudging may bias gradient direction;
the present noiseless replay addresses drift and alignment, not noise tolerance.

[exp-009](../series/001-beta-selection/experiments/exp-009-fixed-beta-transfer.md)
freezes each scheme/block's initialization beta for RMS 0.03 or 0.09 and measures
epochs 10/30/50. Report displacement ratios to initialization and cosine changes,
using 0.95 pooled worst-layer cosine as a descriptive high-alignment threshold.
Do not attribute displacement changes solely to intrinsic susceptibility: the
injected loss-gradient vector also changes. These are BPTT trajectories, not
predictions of trajectories trained with EqProp.
