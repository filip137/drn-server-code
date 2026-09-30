---
id: "H-009"
title: "Epoch10 gradient-aligned beta improves short EqProp continuations"
---
# Does a trained-checkpoint beta preserve BPTT-like learning?

Exp-010 used an initialization beta or common RMS0.09 and learned substantially
less well than historical BPTT. Exp-008 instead measured high epoch10 cosine for
ours with blockwise RMS[0.02,0.035,0.001]. Test the corresponding beta vector and
smaller/larger multiples from that same epoch10 checkpoint.

Compare exact half-epoch continuations with identical Adam/BN/RNG/data/LRs.
Rank closeness to a matched native BPTT continuation by absolute validation-CE
gap, with accuracy gap, Conv-update alignment and starting/ending gradient
cosine reported separately. Improved accuracy is useful even if farther from
BPTT; do not confuse matching with better performance. A free-T autograd control
separates the existing hybrid learner's solver-differentiation change from its
Conv gradient replacement. No claim about fresh training, other schemes or noise.

[exp-011](../series/001-beta-selection/experiments/exp-011-trained-beta-half-epoch.md)
owns the bounded sweep. This exploratory ranking has no invented pass threshold.
