---
id: "exp-013"
title: "Locate the CIFAR performance gap in existing training trajectories"
status: "complete"
hypotheses: []
---
# Existing training trajectories

Filip accepted a comparison of existing CIFAR-10 legacy, ours and baseline
training trajectories after exp-012 failed to explain the endpoint ranking.
CPU-only analysis of the selected seed-0 trajectories, epochs 1–50, assembled
from their recorded 1–10, 11–30 and 31–50 segments. No new training, checkpoint
evaluation, validation selection or official-test access.

Check canonical bundle integrity, exact epoch/step coverage, parent checkpoint
provenance, shared initialization/split and resolved scientific config differences.
Distinguish selected optimizer settings and runtime/source differences from a
controlled amplification-only comparison. Source paths and hashes belong in
`results/cifar10-training-trajectories-20260927-v1/analysis/summary.json`.

Plot training CE/accuracy and validation CE/accuracy. Retain raw epochs and show
fixed nonoverlapping five-epoch means; describe when a persistent difference
appears without fitting a retrospective change point. Report common epochs
10/20/30/40/50 and five-epoch bins. Training metrics are averages over changing
weights and augmented minibatches with training BN; validation uses endpoint
weights, unaugmented examples and saved BN. Their subtraction is not a clean
generalization estimate. Worse training and validation measurements suggest a
fitting component, but cannot isolate optimizer, parameterization or BN causes.

Use available completed baseline/legacy low-LR floor continuations (51–70) only
as separately labelled context. Missing local ours/renewal coverage must not be
treated as failed runs or silently pooled. These continuations are not replicates.

Budget: CPU analysis only, no queue/GPU allocation. Root registered before creation:
`results/cifar10-training-trajectories-20260927-v1/`; runner `analysis/review.py`;
outputs `analysis/summary.json`, `analysis/trajectories.csv` and JPG figures.
Fail on source metric corruption, duplicate/missing epochs or invalid provenance;
retain scientific mismatches as explicit comparison limits. Record conclusions
and the next useful intervention in the result note; do not launch it from analysis.

Completed: nine primary bundles and all 150 epoch records validated, plus two
separate floor-continuation bundles (40 epochs). Plots inspected and scientific
review recorded in [result](../results/exp-013-training-trajectories.md).
No new simulation launched.

September28 review extension, requested by Filip: reconcile the original frozen
affine-BN comparisons (epochs1–10) and review the next discriminating tests.
Same result root; `analysis/bn_review.py`, `analysis/bn_review.json` and
`analysis/bn_affine_review.jpg`. Four frozen bundles include legacy's recovered
epochs1–4 prefix. Canonical bundle checks, complete epoch coverage, actual epoch0
model tensor equality, shared core configs and frozen gamma/beta were verified.
Original trainable reference validation is reused from the trajectory analysis.
This extension is CPU-only; proposed training tests remain unlaunched.
