---
id: "exp-004"
title: "Extend the complete coordinate-and-optimizer control"
status: "blocked"
hypotheses: ["H-002", "H-004"]
---

# exp-004 — Extend the mapped training control

## Question and dependencies
Does the already demonstrated baseline/normalized-legacy mapping remain practically
matched over the original horizon, and can the residual optimizer explanation be tested
without a broad retuning exercise? Requires exp-003's exact mapping, tolerances and
source configs. No exact executable command is asserted before that audit.

## Cases and factors held fixed
Two seed0 arms at most: baseline and normalized legacy, both with the complete baseline
optimizer/mapped config from the existing identity study. Same initialization, split,
minibatch/augmentation sequence, precision, runtime environment, gain and BN state,
solver budgets, 50-epoch scheduler horizon and output-selection rule. Follow the
source mapping exactly; merely copying a headline LR does not reproduce the control.

Reuse an audited baseline trajectory only if every required factor and source identity
matches; otherwise rerun the paired baseline within the two-arm budget. Do not count
a continuation as a second seed. The normalized-legacy/legacy-optimizer endpoint may
be a secondary causal control only if the audit establishes equal non-intervened factors.
If that reference is not matched, mark H-004 not-tested instead of attributing a
historical score difference to the optimizer. A third reference rerun would need a
separately amended bounded plan before measurement.

## Metrics and decision
At the original prefix boundary compare logits, gradients, BN state, optimizer state
and actual updates using recovered source tolerances. Continue only if the mapping is
qualified. Primary extension outcome is the absolute paired epoch50 accuracy difference;
proposed practical-equivalence margin 0.2pp, frozen before launch. Report CE, per-layer
state/update divergence and saved-state qualifications at the declared checkpoints.

H-002 is supported only within the tested numerical/horizon scope if both prefix and
endpoint criteria pass. This is not a proof of identity for all possible training.
H-004 additionally needs the matched legacy-optimizer reference and improvement relative
to it. An optimizer-transplant success demonstrates sufficiency of a bundled change,
not which single parameter caused the difference or a global optimum.

## Execution, budget and stop
Reuse/adapt the actual existing mechanism runner after audit, freeze exact configs,
command and hashes in this file, and index a new results root before execution.
Proposed cap: two 50-epoch runs / 100 scheme-epochs and 24 GPU-hours including checks.
Measure smoke throughput first; if infeasible, revise scope before execution or leave
unstarted. Stop on mapping/solver failure or nonfinite values, retain evidence and
investigate the first divergence; no silent tolerance relaxation. Write
results/exp-004-mapped-control.md with coverage, qualification and scoped verdicts.
