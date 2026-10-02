---
id: "H-021"
title: "Packing centered EqProp phases improves throughput within V100 memory"
---
# Packed positive and negative nudged phases

Prospective extension of the [precision/performance pilot](../../pilots/cifar-float64-performance-20261001.md).
At fixed free T and nudged K, the positive and negative trajectories are independent.
Packing them along the batch dimension may reduce overhead and improve throughput,
while increasing live state and workspace memory. Batch normalization stays in the
unchanged free pass; no statistics are computed across packed phases.

For ours CIFAR L8, batch32, float32 free/digital/Adam and float64 nudging/formula,
predict at least5% lower mean update time than sequential phases and successful
execution on one V10032GB. Proposed engineering fidelity criterion: every nonzero
convolution gradient at the identical first update has cosine>=0.99999 and relative
L2 error<=1e-4. These thresholds are frozen before the new measurements.
Insufficient speedup, OOM or failed fidelity contradicts the practical claim in this
regime. No claim of double speed or improved learning accuracy.

[exp-027](../series/001-beta-selection/experiments/exp-027-phase-k-performance.md)
owns execution and measurements.
