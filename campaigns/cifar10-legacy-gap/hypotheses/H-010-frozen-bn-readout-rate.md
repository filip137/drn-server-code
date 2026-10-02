---
id: "H-010"
title: "Reducing the dense-readout rate preferentially improves early legacy fitting"
---
# H-010 — Dense-readout rate with frozen BN affine

Prospective September28 follow-up to [H-009](H-009-classifier-update-imbalance.md)
and [X-002](../explorations/X-002-optimization-and-regime.md). The earlier one-step
test was inconclusive across epochs30/50. Current frozen-BN trajectories show
larger raw legacy convolution gradients but smaller legacy readout gradients;
Adam normalization means neither those magnitudes nor nominal LR ratios prove
that the readout is too fast.

Mechanism proposed: legacy's selected readout LR lets the classifier adapt to
transient minibatch features faster than the convolution stack, impairing fitting
on separate examples. Hold BN affine frozen and change only `head.weight` LR.

Prediction in [exp-016](../series/002-residual-gap/experiments/exp-016-dense-readout-lr.md):
from the original unnormalized legacy epoch10 state, setting its base readout LR
to ours' original readout LR divided by9 improves fixed training-cohort CE after
400 updates by at least1% versus unchanged legacy. Require validation CE no more
than1% higher and accuracy no more than0.5pp lower. The relative training-CE
improvement must exceed ours' improvement under the same multiplicative reduction
by at least0.005 (0.5 percentage points). These are exploratory thresholds frozen
before observing branches, not significance tests. A one-third LR is a secondary
dose comparison, not a replacement primary endpoint.

Meeting all criteria supports this short-window prediction. If both lower rates
worsen legacy training-cohort CE by at least1%, that contradicts the proposed
benefit in this regime. Other outcomes are inconclusive. Invalid or unmatched
branches cannot adjudicate it. One seed and small cohorts do not establish an
optimal LR, long-run validation gain, or a cause of the historical scheme gap.
