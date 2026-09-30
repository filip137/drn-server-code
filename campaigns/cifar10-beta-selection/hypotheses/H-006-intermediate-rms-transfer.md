---
id: "H-006"
title: "Epoch50 RMS candidates remain near-optimal at intermediate training checkpoints"
---
# Intermediate-checkpoint transfer

The useful RMS changes substantially between initialization and epoch50. Test
whether each scheme/block's epoch50 selected nominal RMS is within0.01 pooled
worst-layer cosine of that checkpoint's best sampled value at epochs10 and30.
Support requires this across all three schemes/blocks at both intermediate
checkpoints. Failure indicates checkpoint-dependent selection; report where it
fails rather than treating schemes/blocks as interchangeable.

Recommendations compare per-checkpoint sampled peaks with a fixed per-block
target maximizing the worst checkpoint's pooled cosine. Use actual common nominal
targets, report minibatch limitations and boundary peaks, and keep diagnostic
alignment distinct from EqProp training performance.

Test: [exp-008](../series/001-beta-selection/experiments/exp-008-intermediate-rms.md).
