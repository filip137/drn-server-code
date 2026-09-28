---
id: "exp-005"
title: "Compare the current and matched-initial-RMS beta policies in training"
status: "blocked"
hypotheses: ["H-002"]
---

# exp-005 — Bounded noisy-training qualification

## Question and dependencies
Does matched initial output RMS improve stable noisy training under the frozen current
non-beta settings? Requires exp-003/004, actual training config identities, and a new
execution task. These missing scientific inputs prevent launch-ready status.

## Cases and controls
First proposed pilot: ours and legacy, current p99 versus initial-unit-RMS beta policy,
seed0, sigma=5e-4, ten epochs: four runs / 40 scheme-epochs. Freeze the original
50-or-other-horizon scheduler semantics from the source audit; do not silently restart
or reinterpret its horizon to fit a ten-epoch pilot. Use fresh paired starts, identical
within-scheme minibatch/augmentation order, Adam LR vectors, T/K and noise draw policy.
Exact source seeds and initialization hashes must match. The known RMS betas are
ours=1.385 and legacy=0.02173; p99 controls are 0.987333678708 and 0.1.

If qualified, separately freeze a confirmation matrix: all three schemes, two policies,
three paired seeds {0,1,2}, common 30-epoch endpoint, at most 18 runs / 540 scheme-epochs.
Seed0 reuse is allowed only if the pilot was planned with precisely that continuation
contract and saves full state. Otherwise count fresh runs and disclose wasted pilot cost.
For new seeds, apply the same calibration rule and effort to both policies; do not
claim a per-initialization matching policy while copying seed0 beta values blindly.
The baseline p99 source and the exact calibration rule remain audit dependencies.

## Metrics, interpretation and acceptance
Use final validation accuracy at the common endpoint as primary, CE and best checkpoint
as secondary. Preserve all seeds including failures; report paired differences and
sample dispersion, not a p-value from correlated epochs. Pilot results qualify stability
and feasibility, not a three-scheme superiority claim. Confirmation tests H-002 per
scheme: positive mean paired accuracy change with valid finite runs is provisional
support; nonpositive qualified change or reproducible candidate-specific instability
contradicts improvement. Mixed/uncertain effects remain inconclusive.

Keep non-beta settings fixed to isolate this policy comparison. Failure may motivate a
separate beta/LR study, but does not authorize changing rates mid-comparison. Never
optimize for legacy winning. No official-test evaluation.

## Execution, budget and stop
Recover the actual scientific runner/config path, write exact commands and identities,
and index a new results root before launching under the existing execution skill.
Proposed pilot cap: four runs, ten epochs each, eight GPU-hours total. Confirmation
cap: 18 runs, 30 epochs each, 96 GPU-hours, only after a feasible measured smoke and
explicitly scoped execution request. These caps are planning defaults, not assigned
resources. Stop a failed arm, preserve its evidence and do not silently replace its seed.
Stop the campaign stage when budget expires or qualification fails; no extra sweep.
Expected note: results/exp-005-training.md with full coverage and a scoped decision.
