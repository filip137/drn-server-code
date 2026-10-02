---
id: "H-011"
title: "Screen-selected readout rates improve fresh trainable-BN learning"
---
# H-011 — Transfer of the readout-rate candidates

Prospective follow-up to [exp-016](../series/002-residual-gap/results/exp-016-dense-readout-lr.md).
The doses were selected after a short frozen-BN continuation screen; their
benefit with trainable BN affine and fresh optimization is untested.

Prediction: fresh ten-epoch legacy training with head LR divided by3, and ours
with head LR multiplied by0.037042420152718095, each improve terminal validation
accuracy by at least0.5pp over their respective original-rate historical runs,
with lower validation CE. Historical reference scores are74.68% and75.36%.
All other scientific settings, including convolution LRs and seed0, stay fixed.
The numerical threshold is a proposed exploratory criterion frozen before runs.

Both improvements support the scoped prediction; both accuracy declines of at
least0.5pp contradict it; mixed/smaller effects are inconclusive. No additional
control runs are authorized. Historical controls and possible runtime differences
limit causal attribution, and dose selection used the same seed/data. This is
not independent replication or proof of optimal rates. No official test read.
