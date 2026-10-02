---
id: "H-024"
title: "Diode switching limits block3's useful output displacement"
---
# Why can a tiny output response spoil local EP gradients?

Prospective mechanism test following the full64
[local comparison](../series/001-beta-selection/results/exp-025-all-scheme-layer-rms.md).
At ours EqProp epoch1, block3's worst minibatch/layer cosine falls from about
0.99998 at output relative RMS1e-5 to0.92461 at3e-5 and0.43006 at1e-4.
This compares isolated-block BPTT with EP at matched K4, not full-network BPTT.

Hypothesis: changes in diode decisions during the nudged trajectory materially
cause this upper displacement limit. Record response linearity, per-update
switches and unclamped margins, zero-nudge settling drift and phase asymmetry.
At the first failing target, impose the zero-nudge trajectory's diode decisions
while holding input, forcing, beta, K and local BPTT fixed. Recovery of every
layer/minibatch to cosine>0.95 and norm ratio0.9–1.1 supports switching as a
cause in this checkpoint/block. Improvement without recovery is partial evidence;
no improvement leaves the mechanism unsupported here. Fixed-mask linearity
alone is not sufficient evidence, and large common settling alone is not a
causal explanation. This intervention is artificial and is not a training rule.

[exp-029](../series/001-beta-selection/experiments/exp-029-block3-diode-switching.md)
owns the bounded execution and review. Other blocks, schemes and checkpoints
are outside this test.
