---
id: "H-006"
title: "Useful validation gains remain beyond epoch 50 at the existing LR floor"
---
# H-006 — Useful late learning

## Provenance and claim

Proposed September 24 after the historical trends in
[X-003](../explorations/X-003-training-horizon.md), before any epoch 51–70 data.
At least one of baseline, ours, legacy and normalized legacy gains at least
0.20 percentage points in mean validation accuracy over epochs 66–70 versus
46–50 after full-state continuation at the existing 2% LR floor.

## Decision

Report the change for every arm. Support the bounded existential claim if at
least one qualified arm reaches +0.20 pp and all four arms have reconciled
coverage. Contradict it if all four qualified changes are below +0.20 pp;
missing or invalid coverage is inconclusive. Report which arms meet the
criterion rather than implying all schemes improve. The threshold is a
predeclared practical effect, not a confidence bound or historical tolerance.

Validation CE, epoch 70 accuracy and training CE are secondary. If training CE
falls at least 5% between those windows without a +0.20 pp accuracy gain, label
that arm continued fitting without demonstrated practical accuracy benefit.
An accuracy gain with worsening validation CE is a mixed generalization
outcome, not an improvement in both. Correlated epochs and one seed do not
support population significance. See [exp-006](../series/003-training-horizon/experiments/exp-006-floor-continuation.md).
