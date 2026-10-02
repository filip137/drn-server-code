---
id: "H-029"
title: "One output-RMS target per block keeps digital-ReLU EqProp gradients accurate across training"
---
# Is there a per-block beta rule that works at every epoch?

Prospective (Filip's goal, October2): with linear analog block outputs and a digital
ReLU, find for each block the beta that works across epochs. exp-032/exp-033 show
windows reaching relative output RMS 1.0 at initialization and on transplanted
trained weights, with 0.02–0.2 passing the strict rule everywhere in the transplants.
This hypothesis tests the architecture's own BPTT trajectory.

Claim: for each block there is a relative output-RMS target T_b that passes the
strict rule (every layer's pooled local cosine ≥0.99, every minibatch ≥0.95, pooled
norm ratio 0.9–1.1) at every replayed checkpoint of the digital-ReLU ours BPTT
trajectory (epochs 0/1/2/3/5/7/10), float32.

Frozen selection rule: per block, take the sampled targets that pass at every
checkpoint. If the set is contiguous, recommend its log-midpoint, rounded to the
nearest sampled target; otherwise recommend the log-midpoint of its longest
contiguous run. Report the margin: grid steps to each end. Also test a fixed beta:
the calibrated beta range for that passing set at each checkpoint, and whether the
ranges share a common beta. If they do, a fixed beta suffices; if not, beta must be
recalibrated to T_b during training.

- **Support** if every block has a non-empty passing set, with at least two sampled
  targets.
- **Contradict** if any block has no target that passes at every checkpoint.
- Otherwise **partial**.

Gradient agreement under BPTT-trained weights is not EqProp learning. A learning
test with the selected targets is a separate experiment.

[exp-034](../series/001-beta-selection/experiments/exp-034-digital-relu-bptt-trajectory.md)
owns execution and review.
