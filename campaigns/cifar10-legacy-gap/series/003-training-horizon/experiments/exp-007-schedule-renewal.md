---
id: "exp-007"
title: "Compare renewed learning rates with the matched floor continuations"
status: "running"
hypotheses: ["H-008"]
---

> **September 25, 2026 — pre-launch policy update:** Filip removed CIFAR solver
> and migration audits as launch prerequisites, including the epoch-50 full-validation
> replay and accuracy/CE drift thresholds. Future launches retain source/config/data
> identity, full checkpoint restore, execution smoke, resource admission and runtime
> nonfinite checks. The older gate requirements and observations below are historical;
> do not rerun them automatically or record an omitted audit as passed. Already launched
> jobs retain their frozen execution; post-training audits are outside this pre-launch change.

## September25 user revision — authoritative current scope

The user cancelled Fifi and all floor runs, added baseline to renewal, and
authorized immediate daytime execution of baseline/local, legacy/nom-cool-1,
ours/Riri. Three original epoch50→70 renewal branches,6h each including
gates;18h total. Follow [current handoff](../HANDOFF.md).
The earlier allocation and six-case contract below are preserved as history.
H-008 renewal-minus-floor cannot be evaluated without matched floor controls.
# exp-007 — A bounded schedule intervention

## Cases and scientific contrast

Test [H-008](../../../hypotheses/H-008-lr-floor-limits-progress.md) on raw legacy
and ours only. Add two 20-epoch branches, each starting from its **original
epoch 50 final checkpoint**, with the exact same data/augmentation sequence
and optimizer history as its floor control in
[exp-006](exp-006-floor-continuation.md). Do not start from an epoch 70 result.
Reuse those two controls; do not train duplicates or choose which schemes
receive renewal based on observed floor outcomes. Baseline and normalized
legacy renewal are outside this bounded test.

Keep all Adam moments and steps, weights, gains, BN state, group LR ratios,
split, precision and solver settings. Only the multiplier applied to every
optimizer group's original starting LR changes, including BN and gain groups.
This tests a joint group schedule, not a conductance-only intervention.

## Exact renewed schedule

For training epoch e in 51…70, define

`r(e) = 0.02 + 0.08 * (1 + cos(pi * (e - 51) / 19)) / 2`.

Thus epoch 51 uses 10% of each original group LR (five times the floor),
epoch 70 uses 2%, and the schedule clamps at 2% afterwards. This moderate,
single prespecified schedule is not a search. Epoch 50's saved LR is restored
and verified first; renewal is applied exactly once at the branch boundary.
Store the epoch 51 learning rates in provenance separately from restored LRs.
Preserve 98490 final Adam steps, as in the floor arms.

## Implemented interface and qualification

Implemented September24 under the subsequent scheduling authorization:
`--schedule-branch` explicitly qualifies the intervention while ordinary
restore still rejects changed scientific configs. Renewal configs and
`schedule_branch.json` now exist beside the four floor configs. The focused
and regression CPU suite passed28 tests (two CUDA-only tests skipped), including
exact epoch60 recovery with70350→98490 Adam steps. Scientific smokes and
remote gates remain separate admission requirements.

The implemented CIFAR runner accepts one explicit optional
`--schedule-branch` JSON input. Its fixed content for these two cases is:

```json
{"kind":"cosine_renewal","from_epoch":50,"first_epoch":51,"last_epoch":70,"start_lr_ratio":0.10,"end_lr_ratio":0.02}
```

Accept it only with an epoch 50 `--resume-run`, require the declared 70-epoch
target, and retain strict equality checks on all original scientific fields.
Restore and verify the parent optimizer/scheduler/RNG first, then install
this explicit branch schedule using original optimizer-group initial rates.
Record the specification in immutable manifest/resolved config and saved
checkpoints, with parent and input hashes. Checkpoint recovery must restore
the already active branch and absolute epoch without applying the jump twice;
reject any changed branch specification. Existing no-branch runs must remain
identical. Allow recoveries of branched checkpoints without requiring the
original epoch 50 launch condition again, while verifying their saved branch.

Derive `legacy_renewal.json` and `ours_renewal.json` from the prepared floor
configs, changing study/arm/evidence labels only; use study
`cifar10-l8-schedule-renewal-e50-e70-seed0-20260924-v1` and arms
`{legacy,ours}_renewal_e50_e70`. Keep epoch 70, runtime 6h and original parent
hashes. Store the explicit branch input beside those configs. Freeze all
files in a reviewed source snapshot before either experiment's production,
so both floor and renewal branches use the same runner version. If exp-006
already ran with another version, prove no-branch behavior equivalence before
reusing it; otherwise leave the pair unqualified, without silent reruns.

The production command will be the exp-006 template with the two frozen
renewal configs, matching original parent/assets, the separate study root
and `--schedule-branch` pointing to that exact JSON. The flag is implemented; the dated queue supplies these exact inputs.
No production can begin before22:00 and successful common admission gates.

## Tests and gates required at implementation/execution

- CPU tests verify exact epoch 51/70 ratios, floor after 70, unchanged group
  ratios, strict rejection of unrelated config changes, and rejection of a
  branch at the wrong checkpoint epoch.
- Using a small synthetic model, compare uninterrupted versus save/restore
  at epoch 60 for parameters, Adam state, schedule/LRs and RNG. Check all 20
  updates and final epoch/step count; ensure renewal happens only once.
- Test unchanged no-branch scheduling/restore and the existing deadline pause
  behavior. Never reinterpret a failed/paused ancestor as a successful result.
- After execution authorization, pass the same input, boundary-replay,
  migration and smoke gates as exp-006; pair both schedules on Fifi under
  the same software snapshot. Retain nonfinite and solver-invalid renewal
  outcomes; do not retry them with gentler rates as the same experiment.

## Decision, allocation and outputs

For each scheme, compare mean validation accuracy66–70 between renewal and
floor. H-008's practical threshold is +0.20 pp. Report both pairs, secondary
validation/training CE, epoch 70 outcomes and the full learning-rate traces.
Support for incremental benefit does not mean the floor cannot improve;
mixed accuracy/CE outcomes remain explicit. All inference is limited to
these two existing seed 0 trajectories and this one renewal policy.

Target is Fifi, sequentially after the four floor arms, under the common
[allocation](../README.md). Forty added scheme-epochs, about 8.4 GPUh expected,
**12 GPUh cap** including checks/retries, at most6 GPUh per added case. Floor
controls are charged only to exp-006. Stop at nonfinite/solver failure,
source mismatch or the common deadline/cap; no automatic schedule tuning.

Expected root `results/cifar10-l8-schedule-renewal-e50-e70-seed0-20260924-v1/`
contains the two `cells/{case}_renewal_e50_e70` bundles and diagnostics.
Nothing has been created there or remotely. Collect and validate locally,
reconcile all four paired bundles (two reused) and write
`series/003-training-horizon/results/exp-007-schedule-renewal.md`, explicitly
naming failures/exclusions and H-008's scoped verdict. The previous runner-capability blocker is resolved. The experiment remains
planned pending its scheduled window and admission checks.


## Execution and monitoring handoff

Use the shared [current run handoff](../HANDOFF.md) for exact cases, source identity,
queue/monitor handles, inspect/collect/validate commands, budgets and recovery rules.
September25: Fifi queue PID18566 waiting for22:00 Paris; no production or remote
gates started. Local monitor PID2028277 owns the30-minute checks.
