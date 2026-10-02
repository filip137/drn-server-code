---
id: "H-008"
title: "A modest renewed schedule improves on the floor for legacy or ours"
---
# H-008 — The LR floor limits late accuracy gains

## Provenance and claim

Prospective continuation test motivated by
[X-003](../explorations/X-003-training-horizon.md). From each identical epoch 50
checkpoint, a cosine renewal from 10% to 2% of original group LRs over 20
epochs improves mean validation accuracy at 66–70 by at least 0.20 pp relative
to the unchanged 2% floor for at least one of raw legacy and ours.

## Decision

Support the scoped claim if either paired difference reaches +0.20 pp and
both scheme pairs have reconciled qualified coverage; contradict it if both
qualified differences are below +0.20 pp. Missing or invalid pairs are
inconclusive. A nonfinite/solver-invalid renewal is a failed intervention,
not evidence for a successful schedule or a reason to tune another rate.
Report each pair, validation CE and training CE, including negative effects.

If the floor also improves, support is for incremental benefit of renewal,
not an assertion that useful learning requires renewal. A null result only
limits this schedule, horizon and two checkpoints; it does not prove an
optimal LR. No claims about baseline or normalized legacy are made.
See the exact schedule and restore requirements in
[exp-007](../series/003-training-horizon/experiments/exp-007-schedule-renewal.md).
