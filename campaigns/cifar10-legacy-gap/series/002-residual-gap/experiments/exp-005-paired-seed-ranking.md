---
id: "exp-005"
title: "Test reproducibility of the fixed-config epoch50 gap"
status: "blocked"
hypotheses: ["H-003"]
---

# exp-005 — Paired-seed endpoint reproducibility

## Question and dependencies
Is the current observed endpoint deficit robust across new matched seeds? Depends on
exp-003's resolved configs/environment decision, not on legacy being expected to win.
Freeze all source configurations and the precise seed/pairing convention before running.

## Cases and controls
Preferred new matrix: baseline, ours, legacy on seeds1 and2, six runs, each 50 epochs;
include seed0 only when audit proves comparability. If seed0 cannot match the declared
environment/config, revise the matrix explicitly to all three seeds {0,1,2}, nine runs
maximum. Do not pool the unmatched old seed0 with new runs or replace difficult seeds.

Use the current per-scheme selected LR vectors unchanged. This tests reproducibility of
selected configurations, not equality of learning rates or global per-scheme optima.
Keep data split, augmentation/minibatch pairing, architecture, T/K, precision, BN/gains,
Adam and the full 50-epoch cosine horizon fixed. Pair initialization appropriately in
physical coordinates under the source protocol, not by assuming a common integer seed
is enough. No official-test reads and no hyperparameter search during confirmation.

## Metrics and acceptance
Primary: final epoch50 validation accuracy, paired legacy-minus-baseline and
legacy-minus-ours differences by seed, mean and sample dispersion. Secondary: validation
and training CE, fixed early endpoints, solver checks and failures. Do not substitute
best validation or treat epochs as independent replicates.

H-003's strict observed-seed criterion requires both differences negative on all three
seeds. A sign reversal contradicts that criterion, but does not establish a population
winner. Small effects, few seeds and uncertainty remain explicit. Invalid/missing arms
prevent a complete qualified comparison; keep them in coverage rather than dropping them.

## Execution, budget and stop
After source audit, record exact config files/commands, code/input hashes, target,
measured duration estimate and indexed results root. Proposed cap: six new runs,
300 scheme-epochs, 48 GPU-hours; a prespecified full-seed0 fallback is capped at nine
runs, 450 scheme-epochs and 72 GPU-hours. Choose one design before new measurements.
No automatic extension or retuning if the result is inconvenient. Stop failed arms,
preserve full state where possible, and distinguish infrastructure retries from seeds.
Expected note: results/exp-005-replication.md with every declared case and a scoped verdict.
