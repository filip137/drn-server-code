---
id: "H-019"
title: "A block-output displacement range can preserve all of its layer gradients"
---
# A usable output-displacement range for sizing beta

Filip asks whether each block has an output d/f range where **every convolution
inside that block** has high EqProp/BPTT cosine. Internal layers have different
responses, so inspect their own d/f curves while selecting a single output
target and beta for the whole block. This extends
[H-017](H-017-small-trained-layer-response.md) and the earlier
[common-RMS exploration](../explorations/X-001-beta-sensitivity.md).

Test progressively: a common range across layers at one checkpoint; persistence
across checkpoints within a scheme; and transfer across the three schemes.
Analyse EqProp epochs 0/1/5 and BPTT epochs 0/10/30/50 separately, as Filip
explicitly selected. These histories cannot form one training curve.

[exp-025](../series/001-beta-selection/experiments/exp-025-all-scheme-layer-rms.md)
uses full float64 throughout the replay, following the precision evidence in
[exp-024](../series/001-beta-selection/results/exp-024-precision-response.md),
with a matched output-d/f grid and every layer's own response. This tests
a displacement criterion without the observed float32 response floor. A sampled
target passes the primary descriptive criterion only when every layer's pooled
native-BPTT cosine is >=0.95. Report >=0.99, norm ratios and minibatch spread
separately. A nonempty intersection supports a shared target in that scope;
an empty intersection contradicts transfer on the sampled grid, without proving
that no narrow unsampled interval exists. This is beta-sizing evidence, not
proof of training-accuracy recovery or robustness to injected read noise.
