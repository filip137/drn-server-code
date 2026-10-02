---
id: "exp-006"
title: "Continue four epoch 50 trajectories to 70 at the unchanged LR floor"
status: "planned"
hypotheses: ["H-006", "H-007"]
---

> **Latest execution scope:** Filip reauthorized baseline, legacy and ours for
> 20 epochs at the unchanged low-LR floor via the shared automatically placing
> GPU queue. Fifi-only placement and the floor cancellation below are superseded
> for these three cases. Normalized legacy is not submitted. See the current
> [execution handoff](../HANDOFF.md#shared-queue-three-low-lr-floor-jobs).

> **September 25, 2026 — pre-launch policy update:** Filip removed CIFAR solver
> and migration audits as launch prerequisites, including the epoch-50 full-validation
> replay and accuracy/CE drift thresholds. Future launches retain source/config/data
> identity, full checkpoint restore, execution smoke, resource admission and runtime
> nonfinite checks. The older gate requirements and observations below are historical;
> do not rerun them automatically or record an omitted audit as passed. Already launched
> jobs retain their frozen execution; post-training audits are outside this pre-launch change.

## Earlier September25 revision — superseded for three floor cases

The user cancelled Fifi and all floor runs, added baseline to renewal, and
authorized immediate daytime execution of baseline/local, legacy/nom-cool-1,
ours/Riri. Three original epoch50→70 renewal branches,6h each including
gates;18h total. Follow [current handoff](../HANDOFF.md).
The earlier allocation and six-case contract below are preserved as history.
H-008 renewal-minus-floor cannot be evaluated without matched floor controls.
# exp-006 — Unchanged-schedule continuation

## Question and frozen cases

Does more training improve validation accuracy, and does raw legacy catch up?
Test [H-006](../../../hypotheses/H-006-useful-late-learning.md) and
[H-007](../../../hypotheses/H-007-legacy-late-catchup.md) with exactly four
seed 0 branches: baseline, ours, legacy, normalized legacy. Each restores its
own **final epoch 50** checkpoint and completes epochs 51–70, not a best
checkpoint or a fresh optimizer. One20-epoch extension per condition; no
selection among horizons after seeing results.

Architecture, BN, gains, augmentation/order, split, precision, batch32/eval16,
solver T=K=[6,6,4], all explicit diode dictionaries, Adam settings and exact
group LR vectors stay unchanged. The original scheduler remains
`{"final_lr_ratio":0.02,"horizon_epochs":50}`; epochs 51–70 use its 2% floor.
Restore all Adam moments/steps, BN buffers, model/gains and RNG state. Final
Adam step count is 98490 (70350+20×1407). No official test read.

## Sources and preparation checks

Paths below are relative to repository root. Four parent canonical bundles
and their four prepared-asset bundles were read-only validated on September 24
using `experiments.reporting.validate_run`; all returned no errors. SHA-256
values were recomputed from the actual files. This is a narrow input check,
not completion of exp-003's full historical comparability audit.

| Case | Parent run under `results/` | Final checkpoint SHA-256 |
|---|---|---|
| baseline | `cifar10-l8-analog-baseline-legacy-e30-e50-seed0-20260922-v1/cells/baseline_e30_e50` | `9c9cb07a2ad7646ec2c58411d2cf31307ba337686f81c75d810b3d84d8cf2560` |
| ours | `cifar10-l8-analog-ours-e30-e50-seed0-20260922-v1/cells/ours_e30_e50` | `740f536b0cae14f9289d8e84c826bb25025ce0e62fc73e880447565d5d74bcf5` |
| legacy | `cifar10-l8-analog-baseline-legacy-e30-e50-seed0-20260922-v1/cells/legacy_e30_e50` | `66b8315f4cb52b308c2592bae3e269168b616b247aa6150e638d17f2cf50262f` |
| normalized_legacy | `cifar10-l8-bn-gain-followup-seed0-20260923-v1/cells/legacy_voltage_normalized_e48_e50` | `10f480c32a85293c942a80e028389776ba229556fd0324fa5bc3213cad9e8415` |

Parent manifest hashes in the same order:
`0a8fb9b14f2dd001d4d3ffc6ab052f88c94e21019aaa9d2ba9d8456c6e6e2bd7`,
`945cf7a9aafed13c86893349e89e83f0600571a9aea16223df5c9f967e242735`,
`355de29a7b05ee824e8412f724cd0456dd5794aa22dce6789fc018c30f676825`,
`2587699d86061122a6c08718345a8755ef66ac0f6f80f747f76b43667f1530c4`.

Baseline/ours/legacy assets are respectively `prepare-baseline`, `prepare-ours`,
`prepare-legacy` under
`results/cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1/`.
Normalized legacy uses the collected
`results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/inputs/normalized-assets/`.
That differs from its original remote pathname; its actual initializer,
split and qualification hashes match the parent's `artifacts/assets.json`.
Preserve the files, not the obsolete transport path.

Prepared JSON configs live in
`configs/cifar/horizon_e50_e70_20260924/{baseline,ours,legacy,normalized_legacy}_floor.json`.
They are exact parent resolved configs with changes limited to study/arm,
evidence label, epoch limit, runtime cap and continuation provenance. The
adjacent `source_identity.json` pins current source including dirty changes
and the four configs at commit `6be7d41038018b225cef7d1771fdb905084b0db7`.
It is a design-time pin, not a claim that source has been staged or tested.

## Execution contract — commands prepared, not executed

Use the CIFAR-specific `experiments.train_cifar_l8_analog` runner, not the
MNIST exact-run entrypoint. Stage one isolated source snapshot on Fifi from
the pinned files without replacing its working checkout. Preserve parent
bundles and assets in their relative paths. Source changes require an explicit
new snapshot/identity and compatibility checks; do not loosen hash checking.

Local and remote result root:
`results/cifar10-l8-horizon-e50-e70-seed0-20260924-v1/`, remote relative to
`/home/filip/server_code`. Each child is `cells/{case}_floor_e50_e70`.
With the isolated snapshot as working directory, the command template is:

```bash
/home/filip/miniconda3/envs/py312/bin/python -m experiments.train_cifar_l8_analog \
  configs/cifar/horizon_e50_e70_20260924/CASE_floor.json \
  --mode train --device cuda --target fifi --require-gpu-resident \
  --output-dir /home/filip/server_code/results/cifar10-l8-horizon-e50-e70-seed0-20260924-v1/cells/CASE_floor_e50_e70 \
  --resume-run PARENT_RUN --assets PREPARED_ASSETS \
  --dataset-root /home/filip/server_code/results/cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1/data/cifar-10-batches-py \
  --source-identity configs/cifar/horizon_e50_e70_20260924/source_identity.json \
  --stop-before NIGHT_END_ISO --initial-epoch-seconds 900 --deadline-reserve-seconds 180
```

Substitute CASE from the four frozen names and absolute parent/asset paths
from the table. NIGHT_END_ISO is the 08:00 Europe/Paris cutoff for the actual
authorized night, with its UTC offset. Freeze concrete dated commands at
execution; none is armed by this document. For a local smoke, use the same
config/parent/assets, `--mode smoke --smoke-batches 1 --device cpu`, omit
`--require-gpu-resident` and deadline options, and use a distinct smoke path.
Then repeat that one-batch smoke on Fifi with CUDA and GPU residency required.
Neither smoke advances the production checkpoint; production restores the
original epoch 50 parent. Stage the unchanged training data for each target.

## Required gates before production

- After execution authorization, revalidate parent/asset bundles and hashes,
  verify full checkpoint restore at epoch 50/70350 Adam steps and the group LRs
  at the floor. Reuse the accepted preparation qualification; do not search T/K.
- On disposable copies before training, run the existing audit function on
  the prepared eight-example cohort at accepted and reference iteration
  counts, and evaluate epoch 50 on the full validation split on Fifi. These
  are validation-only checks, with no official-test access or BN updates.
- Predeclare migration acceptance: absolute validation accuracy drift at most
  0.10 pp and relative validation CE drift at most1% against the saved epoch 50
  metrics, plus the unchanged native solver-audit thresholds. These are new
  practical guardrails, not a proof of trajectory identity. Failures block
  the affected comparison; do not automatically relax them or replace scores.
- Preserve original mixed-runtime history in the report even when guards
  pass. The new suffix uses a common Fifi environment, but hardware migration
  prevents treating prefix-to-suffix gains as an isolated causal time effect.
- Require semantic success from both smokes and record measured memory and
  shared throughput. Check all planned source pins after any runner change.
  Final native solver audits remain mandatory at epoch 70.

The design task ran no scientific gates. Under the later scheduling
authorization, CPU smokes are in progress and remote gates are mandatory
before production. An initial local sandbox IPC failure is retained and
excluded; unchanged CPU smokes are retried outside the sandbox. The runner directly supports
the training/smoke commands; the boundary evaluation is a read-only diagnostic
using existing model/data/audit functions, prepared in the study-local `launch/boundary_gate.py`.

## Metrics, rules and cost

Primary H-006 measurement: mean validation accuracy66–70 minus46–50 per arm,
with the frozen +0.20 pp practical threshold. Primary H-007: changes in raw
legacy-minus-baseline and legacy-minus-ours mean accuracy differences; both
must improve by at least 0.20 pp. Historical late means are baseline89.364%,
ours89.700%, legacy88.676%, normalized legacy88.796%. Apply the exact support,
contradiction and missing-coverage rules in the linked hypotheses.

Also report mean validation CE, training CE, epoch 70 accuracy/CE, learning
rates, runtime and solver checks. Use every planned case; no per-arm best
checkpoint selection, no significance tests treating epochs as replicates.
Reconcile epochs 51–70 exactly once per arm across pause/recovery segments.

Budget:80 added scheme-epochs, approximately 16.9 GPUh expected, **24 GPUh cap**
including all checks/retries, at most6 GPUh per case. Follow the common
[allocation and night-window plan](../README.md). Stop on nonfinite values,
failed qualification, budget/deadline or source mismatch. Preserve failed
and partial evidence; no automatic new seeds, LR changes or extra epochs.
Future monitoring checks every 30 minutes, with full-state pause before 08:00.
Collect canonical bundles locally, validate coverage, and write
`series/003-training-horizon/results/exp-006-floor-continuation.md` with plots
and scoped H-006/H-007 verdicts before marking complete.


## Execution and monitoring handoff

Use the shared [current run handoff](../HANDOFF.md) for exact cases, source identity,
queue/monitor handles, inspect/collect/validate commands, budgets and recovery rules.
Earlier September25 history: Fifi queue PID18566 and local monitor PID2028277
were subsequently stopped. They do not own the new shared-queue jobs. See the
current handoff for shared-service ownership and submission verification.

Shared-queue submission verified 2026-09-25T18:20:44+02:00: baseline, legacy and ours are
queued under `cifar-floor20-{baseline,legacy,ours}-20260925`; service PID3007181
owns automatic placement and monitoring. No production has started. The case
remains planned pending GPU admission; this is not completed scientific evidence.
