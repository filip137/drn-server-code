---
id: "H-022"
title: "Shorter nudged solves reduce update cost while retaining initial gradient direction"
---
# Reduce nudged K to [4,4,3]

Prospective extension of the [precision/performance pilot](../../pilots/cifar-float64-performance-20261001.md).
Nudged solves account for most of the mixed-precision V100 update time.
Reduce only K from[6,6,4] to[4,4,3], keeping free T=[6,6,4], initialization,
beta, data, optimizer and precision fixed. Predict at least10% lower mean update
time, without a material increase in peak allocated memory (>5%).

A separate initial-gradient screen requires every convolution cosine>=0.95 and
norm ratio0.9–1.1 against the original-K EqProp update on the same minibatch.
These proposed thresholds are frozen before measurement. Passing supports only
this initial screen; it does not establish agreement with BPTT, trained-checkpoint
validity or training accuracy. A speed gain with failed gradient screen supports
only the performance mechanism, not adoption for training.

[exp-027](../series/001-beta-selection/experiments/exp-027-phase-k-performance.md)
owns execution and measurements.
