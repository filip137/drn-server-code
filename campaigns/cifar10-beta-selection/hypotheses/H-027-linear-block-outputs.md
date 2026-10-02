---
id: "H-027"
title: "Linear analog block outputs widen the usable beta window"
---
# Does removing the diode at the injection site widen the beta window?

Prospective, proposed by Filip (October2) after
[X-002](../explorations/X-002-inactive-channel-force.md),
[exp-029](../series/001-beta-selection/results/exp-029-block3-diode-switching.md)
and [exp-031](../series/001-beta-selection/results/exp-031-float32-masked-training.md).
The nudging force is injected into each block's output layer. With perfect diodes
there, the most-perturbed units are the ones that can switch, and a perfect-diode
circuit is exact in beta only until the first switch. A linear analog output layer
cannot switch, so the first switch moves into the hidden layers, which receive an
attenuated perturbation. A digital rectifier after the linear output
(`digital_relu`) restores the block nonlinearity and gates the force exactly with
its derivative, so units that are off receive none.

Intervention: `block_output_activation` set to `digital_relu` or `linear`, against
`perfect_diode` (the original), at the same ours seed-0 initializer, cohort,
T/K and calibration rule, separately in float32 and float64.

Proposed decision rule, frozen before measurement, using U = the upper end of the
contiguous run of sampled targets with every layer's pooled local cosine ≥0.99:

- **Support** if, in both precisions, both linear-output variants have block3 U at
  least two grid steps above the control's (or pass the whole grid when the
  control does not), and no block's U falls below the control's.
- **Contradict** if either variant's block3 U is at or below the control's in
  float64.
- Otherwise **inconclusive**.

Also report, without grading: the lower end of each window (the float32 floor),
whether a single float32 target passes the strict rule (minibatch ≥0.95, norm
ratio 0.9–1.1) in all three blocks, native-BPTT agreement, and calibrated betas.

Scope limit: initialization has no exactly-dead block3 channels (X-002 census), so
this tests only output-unit switching, not the dead-channel BatchNorm amplification
seen after training. A trained-checkpoint replay is needed for that part. Gradient
agreement does not establish learning.

[exp-032](../series/001-beta-selection/experiments/exp-032-block-output-activation-replay.md)
owns execution and review.
