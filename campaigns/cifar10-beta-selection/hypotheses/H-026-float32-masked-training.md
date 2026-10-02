---
id: "H-026"
title: "The inactive-channel force mask improves float32 hybrid EqProp learning"
---
# Does the mask make float32 training competitive?

Prospective, following [X-002](../explorations/X-002-inactive-channel-force.md)
and [H-025](H-025-inactive-channel-force-mask.md). Earlier float32 ours EqProp
reached 54.12% at epoch5 ([exp-013](../series/001-beta-selection/results/exp-013-all-scheme-long-eqprop.md))
versus 66.68% for matched native BPTT; full float64 was too slow to train
([exp-026](../series/001-beta-selection/results/exp-026-float64-periodic-vs-fixed-beta.md)).

Intervention: fresh ours hybrid EqProp, float32 throughout, periodic beta
recalibration to shared output-RMS targets chosen from float32 masked replays,
with `force_mask: "inactive_output_channels"`. Matched control: identical
initializer, data order, targets, calibration schedule, optimizer and seed,
without the mask. Five epochs on the CIFAR-10 45k/5k training/validation split.

Proposed decision rule, frozen before training: **support** if the masked arm's
epoch5 validation accuracy exceeds the control's by ≥2.0 percentage points and
its validation CE is lower; **contradict** if it exceeds by ≤0.5 points or is
worse; otherwise inconclusive. A control stopped by failed beta calibration is
a negative control outcome compared at the last common epoch; an operational
failure of either arm is inconclusive. Report, without grading the hypothesis,
the gap to native BPTT, inactive-channel counts, masked force share, betas and
achieved RMS. One seed per arm: a positive result is a candidate, not a general
training rule or read-noise result.

[exp-031](../series/001-beta-selection/experiments/exp-031-float32-masked-training.md)
owns execution and review.
