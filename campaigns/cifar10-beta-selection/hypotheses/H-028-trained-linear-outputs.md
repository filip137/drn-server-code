---
id: "H-028"
title: "Digital rectification after a linear analog output restores trained block3 float32 windows"
---
# Do linear block outputs keep the beta window open after training?

Prospective, proposed by Filip (October2). The collapse of block3's usable output
displacement appears only after training ([X-002](../explorations/X-002-inactive-channel-force.md),
[exp-030](../series/001-beta-selection/results/exp-030-float32-force-mask-replay.md)),
so the initialization test of [H-027](H-027-linear-block-outputs.md) cannot see it.
With a linear analog output layer the nudged units cannot switch, and a digital
rectifier gates every unit's force by its derivative. That removes both the
output-unit switching and the BatchNorm-amplified force on dead or near-dead
channels, including those the exp-030 channel mask missed.

Intervention A (`transplant`): the 15 trained diode-architecture exp-025
checkpoints, loaded unchanged into the `digital_relu` architecture. The forward
function differs from the one the weights were trained in, so this is an
approximation of a trained digital-ReLU network. Matched float32 control: exp-030's
unmasked diode replays of the same checkpoints. Float32 only.

Proposed decision rule, frozen before measurement, using U = upper end of the
contiguous run of targets with every layer's pooled local cosine ≥0.99:

- **Support** if block3 U ≥1e-3 in every transplanted checkpoint, it exceeds the
  control's U by at least two grid steps wherever the control's U is below 1e-3
  (or the control has no passing target), and no block's U falls more than one grid
  step below its control.
- **Contradict** if block3 U <1e-3 in at least 8 of the 15 checkpoints.
- Otherwise **partial**, reported per checkpoint.

Observation B (`linear`, descriptive, not graded): Codex's linear-output BPTT
checkpoints (ours/legacy, epochs1/5/10), replayed in their own architecture. There
is no matched diode control trained in the same run. Report windows per block and
epoch, and whether one float32 target passes the strict rule (every minibatch ≥0.95,
norm ratio 0.9–1.1) in all blocks and epochs.

Gradient agreement does not establish learning, and transplanted weights are not
trained digital-ReLU networks.

[exp-033](../series/001-beta-selection/experiments/exp-033-trained-linear-output-replay.md)
owns execution and review.
