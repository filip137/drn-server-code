---
id: "H-023"
title: "Float64 nudging only in block3 can preserve useful gradients at shorter K"
---
# Select precision by block

The [precision pilot](../../pilots/cifar-float64-performance-20261001.md) found
that block1's float32 gradients pass at ours epochs0/1/5, block2 passes at1/5
but fails at initialization, and block3 fails at all three. The
[K reduction test](../series/001-beta-selection/results/exp-027-phase-k-performance.md)
found lower cost and close initial gradient agreement at K[4,4,3].

Prospective test: with free T=[6,6,4], nudged K=[4,4,3], float32 blocks1/2
and float64 block3 nudging/formula, calibrating each checkpoint to output relative
RMS[0.01,0.005,0.00001] preserves every convolution's local-reference cosine>0.95
and norm ratio0.9–1.1. The initialization block2 result makes universal success
uncertain; negative early evidence must remain visible. Compare a control with
all blocks nudged in64 at the same K/beta to separate precision from shorter-K
error. The full64 block-local BPTT reference retains original K=[6,6,4].

Use ours initialization, EqProp1/5 and BPTT10 (distinct histories). This is not a
single training trajectory or a training-accuracy test. Support is per checkpoint;
all four must pass to support the claim across this requested checkpoint set.
Measure potential speed/memory savings separately on a matched fresh prefix.

[exp-028](../series/001-beta-selection/experiments/exp-028-block3-float64-k443.md)
owns execution, calibration and review.
