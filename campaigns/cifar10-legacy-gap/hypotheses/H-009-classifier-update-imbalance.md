---
id: "H-009"
title: "Legacy classifier updates impair transfer to separate training examples"
---
# H-009 — Classifier update imbalance

Prospective September27 BPTT hypothesis, narrowing the optimizer explanation in
[X-002](../explorations/X-002-optimization-and-regime.md). Legacy's selected head LR
is about3× ours while its convolution rates are about1/9. Raw LR ratios alone do
not measure functional movement. The proposed mechanism is that the classifier
adapts to a minibatch faster than its feature extractor, reducing the usefulness
of the combined update on separate examples. No EqProp or noise argument is used.

Local falsifiable prediction: at the original epoch30 and50 checkpoints, halving
only the classifier conductance LR for one saved-state Adam step lowers separate
training-cohort CE relative to the full step in at least3/4 directions at each
epoch, and gives a larger mean benefit for legacy than ours. All other parameter
updates must be byte-identical to the full step. Primary evaluation uses minibatch
BN without buffer updates; saved-stat evaluation is a secondary mode control.
Require a mean legacy benefit and excess benefit over ours above
`max(1e-5 CE, 10 × repeated-no-update discrepancy)` at each epoch.

Failure to meet this prediction is inconclusive unless halving instead worsens
legacy CE above that tolerance in at least3/4 directions at both epochs, which
contradicts the proposed local oversize mechanism. Zero/double head-step controls
are secondary and cannot replace a failed primary test after seeing results.
Support is local evidence only, not proof of the historical accuracy gap's cause
or of the outcome of sustained LR changes. Four minibatches share one cohort;
they are not independent seeds. Epoch50 tests a hypothetical step at the LR floor.
