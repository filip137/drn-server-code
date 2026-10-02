---
id: "exp-025"
title: "Seed0 rescaled-target relative-noise training across four amplification schemes"
status: "running"
hypotheses: ["H-011", "H-012"]
---
# Exp025 — Rescaled-target relative-noise training

## Question and authority

Filip adopted the plan and requested implementation on October 1, 2026.
Measure whether amplification-dependent noise robustness persists with smaller
output targets. This extends exp019/020; their unit-target results are historical
comparators, not substitutes for the fresh clean controls here.

## Frozen cases and scientific contract

48 fresh seed0 centered frozen-current EqProp trainings: Conv1 target1/4 for10
epochs, Conv2 target1/16 for30, Conv3 target1/64 for30. Each depth crosses
baseline(A1,B1), ours(A4,B1), legacy(A4,B0.25), amp2(A4,B2) with
eta0,1e-6,1e-5,1e-4. Zero noise is clean EqProp; no BPTT trainings or extra seeds.
Use float64, training/validation batch16/64, input gains40/100/360, T=K4/6/8,
explicit perfect-diode dictionaries, bounds[0,100], exact-zero frozen biases,
Adam and identical accepted seed0 initial weights/data order within each depth.
Fresh Adam per case; retain uninterrupted optimizer state through the horizon.

Original per-scheme named/ordered rates come from
`configs/conv/relative_noise_eta5em4_20260928/`; amp2 uses exactly2x ours.
Scale the actual native paired20-output targets, with no compensating loss or LR
normalization. Reporting may separately divide loss by target_scale squared.
Use ordinary-MNIST deterministic55k/5k splitseed0, model/shuffleseed0;
official-test access is disabled.

For all12 depth/scheme contexts, rematch beta at initialization on the accepted
guarded576-example validation cohort: pooled centered half-difference output RMS
divided by free output RMS, targetsD/F1/4/6, relative tolerance1%, at most3
correction rounds. The actual loss and teaching force must use the rescaled
targets. Freeze the matched injected/internal beta across epochs and all etas;
beta_inj=beta_internal*(A/B)^depth. A calibration failure blocks its four cases.

Noise remains v_read=v+eta*abs(v)*xi, independently sampled by endpoint phase,
node and layer including readout, one global eta, acquisitionseed2026081601,
std0/floor0. Input, physical relaxation, frozen teaching force and validation
remain clean. Retired T/K/residual/rho probes are not assigned.

## Metrics and decision

Primary: fixed epoch10/30/30 validation accuracy. Report all six pairwise scheme
gaps per depth/eta, degradation relative to each scheme's own clean control,
complete learning curves and failures; best validation is secondary. Compare
only matched inputs and contracts. Larger positive gaps at eta1e-5 or1e-4 would
motivate replicated testing; flat or reversed gaps are retained. The existing H-012 heuristic of a gain >=0.2 percentage points prioritizes
replication; it is not a significance threshold. One seed cannot establish
significance. Rankings compare amplification together with its fixed LR policy.

## Execution, budget and stop rules

Local/result root: `results/conv123-rescaled-target-relative-noise-jeanzay-20261001-v1/`.
Configs: `configs/conv/rescaled_target_noise_20261001/`.
Remote results: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/conv123-rescaled-target-relative-noise-jeanzay-20261001-v1/`.
Frozen source: `/lustre/fswork/projects/rech/umg/ucy17uy/server_code_sources/conv123-rescaled-target-relative-noise-jeanzay-20261001-v1/source/`.
Every calibration, smoke, failed/replaced attempt, launch receipt and collected
artifact remains below the local/remote study root.

Use Jean Zay gpu_p13, qos_gpu-t3, v100-16g, oneGPU/fourCPUs/task,
module pytorch-gpu/py3/2.5.0 and single-thread BLAS/OpenMP. Both fmu@v100 and
umg@v100 have verified associations. Training is six arrays of8 tasks:
depth x account. Scheme order baseline,ours,legacy,amp2 and eta order
0,1e-6,1e-5,1e-4; even index sums use fmu, odd sums umg. Thus each account
has24 cases, eight per depth, with scheme/noise balance.

Expected training cost109GPUh from matched prior batch16 V100 timings.
Hard total allowance160GPUh including calibration and recovery. First-attempt
allocation ceilings: 12calibrations x30min +16trainings x(30min+3h+6h)=158GPUh.
Retries require accounting reconciliation and remaining budget; never renew the
study allowance. Calibration full runs use two arrays of6 tasks. The deadline
is seven days from first production submission; record its absolute timestamp
at launch and constrain allocations accordingly. Queue time is not a runtime
promise. No unrelated work may be changed.

Use local synchronous scientific smokes and static submission checks before
launch; remote canaries are required only after a target-specific failure per
docs/jean-zay.md. Production retains full model/Adam/global and private RNG
state each epoch. Operational continuation uses a new preserved attempt and
exact compatible latest epoch state; no epoch0 restart, Adam reset, or retuning.
Nonfinite outcomes are scientific failures, not automatic retries.

## Execution and monitoring handoff

Current phase: all48 production cases submitted to Jean Zay; calibration complete. No production accuracy is collected yet. Arrays: Conv1 fmu458249/umg458267; Conv2 fmu458270/umg458271; Conv3 fmu458278/umg458281, eight cases each.
Latest check: October2 10:14 Europe/Paris. Live squeue shows all48 PENDING
for Priority, no current start-time estimate; sacct shows zero production
allocation seconds and no failed tasks. The earlier October10–11 forecasts
are historical estimates, not current promised start dates. No training
results or production plots yet. The October8 deadline remains unchanged.
Owner: persistent run-watch PID2475035, dispatch enabled; latest observation
October2 10:12:09, no probe errors or recovery agent active. Polling
after launch:300seconds, semantic-stall threshold45minutes. Source/config/launch
identities and exact sbatch arguments will be recorded under `launch/jean-zay/`
before submission. Calibration and training have separate immutable receipts.
Use the existing exact-run/reporting interfaces; no new scientific control plane.

Full collection must reconcile48 outcomes, validate bundles, checkpoint/RNG and
source identity, and produce JPG accuracy/noise, degradation, scheme-gap and
learning-curve figures. Review belongs in the matching campaign result note;
refresh the campaign ledger. Historical central documents and human-authored
dashboard remain outside this assignment.

### Pre-submission validation

Source frozen under `source/`; calibration source identity SHA256
`51ff7f161feb28a9e8d2ef2d3d7414542da274b1e313c21e6319f840703845b3`.
All451 staged source/input hashes verified on Jean Zay. All12 local CPU
calibration smokes passed (5–8seconds/context). Existing target/noise/resume
tests passed33/33; extended strict-resume tests passed11/11, and preparation
checks passed20/20. Detailed smoke receipts are under `local-gates/`.

Latest available account accounting (October1 08:00) shows remaining V100
credit1847.2GPUh on fmu and8312.6GPUh on umg, sufficient for this study.
Local GPU driver is unavailable; CPU smokes cover the same scientific runner.
Full calibration and production use the declared Jean Zay V100 resources.

### Planned calibration submission

Two6-task arrays, `rscale-cal-fmu` and `rscale-cal-umg`, are ready for
submission using source-bound `local-gates/calibration-gate.json`. Exact
argv/mappings: `launch/jean-zay/calibration-commands.json` and
`calibration-cases.json`; worker/wrapper hashes are in those contracts.
Outputs: remote study `calibration/conv{depth}_{scheme}/` and
`calibration-workers/conv{depth}_{scheme}/attempt-0/`; logs `slurm/cal-*.out`.
Transport checks passed38/38; source identity and all12 local calibrations
passed. Submission receipts and actual Slurm handles follow immediately.

Calibration submissions accepted: `457445_0..5` (fmu@v100) and
`457464_0..5` (umg@v100). Receipts: `launch/jean-zay/submission-cal-{fmu,umg}.json`.
V100 t3 scheduling forecasts were October10–11 at submission; they are
tentative, not allocated start times. Short qos_gpu-dev read-only forecast
was October1 17:04; resource/queue options are under operational review.

Operational update: pending calibration parent457445 moved in place to
`qos_gpu-dev` (verified), preserving all6 cases, V10016 resources and30min caps.
The short-job queue has a2h maximum and10-task/user submission limit; parent
457464 remains t3 until its six tasks fit. No duplicate submission or science
change. Receipt: `launch/jean-zay/calibration-qos-update.json`. A question on
V100 deadline extension versus alternate Jean Zay accelerator availability is
pending with Filip; production placement/deadline will follow that answer.

First6 calibrations (parent457445) completed with Slurm exit0:0 in32–39s
per task. Parent457464 then moved to the same development QoS after the
submitted/running count fit its10-task limit. Production remains the adopted
V100/t3/seven-day plan unless Filip changes the optional scheduling preference.

### Calibration complete; production preparation

All12 full576-example calibrations completed on Jean Zay and validated locally.
Maximum relative D/F matching error9.822e-7 (tolerance0.01); no blocked cells.
All48 exact configs are emitted and frozen. Calibration charged0.105833GPUh
including module/startup time; all12 Slurm tasks COMPLETED/exit0:0.
Production source identity SHA256
`7b89c18232cefdff7e9666e82973f9781a3dbb1faca68f11438c238f23491c33`
extends the unchanged calibration source with48 emitted config hashes.
Receipts: `local-gates/calibration-validation.json`,
`local-gates/calibration-accounting.json`.

Six production arrays are prepared as `rscale-conv{1,2,3}-{fmu,umg}`,
indices0..7 each. Exact argv and case mapping are in
`launch/jean-zay/production-commands.json` and `production-cases.json`.
Slurm test-only accepted all six resource requests. Runtime result paths will
be remote study `full/<arm>/attempt-0/runs/000_<arm>_<configSHA8>/`.
Their24 final-config CPU training smokes cover zero/maxnoise across all12
contexts and include optimizer steps and epoch checkpoint writes. Production
will require their passed source-bound gate before any submission.

Production gate passed:24/24 finalized-config CPU smokes, source/config identity
verified remotely for499 files and all six array contracts. All gates completed
without scientific retuning. Proceed with adopted V100/t3/seven-day deadline;
the optional scheduling question does not suspend existing launch authority.
No deadline extension or alternate GPU is assumed without Filip choosing it.

Production conv1-umg: parent458267, indices0..7; accountumg@v100, v100-16g, limit00:30:00.

Production conv2-fmu: parent458270, indices0..7; accountfmu@v100, v100-16g, limit03:00:00.

Production conv2-umg: parent458271, indices0..7; accountumg@v100, v100-16g, limit03:00:00.

Production conv3-fmu: parent458278, indices0..7; accountfmu@v100, v100-16g, limit06:00:00.

Production conv3-umg: parent458281, indices0..7; accountumg@v100, v100-16g, limit06:00:00.

### Production submission and monitor handoff

All48 production cases submitted once; Conv1 fmu parent458249 completes the
array handles listed above. Immutable receipts and48 task mappings are under
`launch/jean-zay/production-submissions.json`; no duplicates or replacement
attempts. Total first-attempt reserved/charged ceiling is152.105833GPUh, below
the160GPUh limit. Deadline: 2026-10-08T14:42:56.874132+00:00.

Monitoring configuration: `launch/jean-zay/production-watch.json`, state
`production-state/`, existing run-watch with300s polling and45min stall alerts.
Owner changes to that service only after its PID/first observation is verified.
The calibration watch only performed one observation; no calibration daemon
was started, so there is no competing watcher.

Inspect with `python3 /home/filip/.codex/skills/run-watch/scripts/run_watch.py
status /home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/conv123-rescaled-target-relative-noise-jeanzay-20261001-v1/launch/jean-zay/production-watch.json`.
Account all attempts with `python3 /home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/conv123-rescaled-target-relative-noise-jeanzay-20261001-v1/launch/jean-zay/accounting.py`.
Collect terminal remote study `full/` into local `production/` using rsync;
retain remote/failed artifacts. From this checkout run
`/home/filip/miniconda3/envs/py312/bin/python -m experiments.analyze_rescaled_target_noise
--study-root /home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/conv123-rescaled-target-relative-noise-jeanzay-20261001-v1 --allow-partial` for intermediate collection.
Final validation removes `--allow-partial` and must exit0 with48 valid cases
and72 valid pairwise comparisons. The current analyzer checks complete epoch
coverage, model/Adam/zero-bias/RNG and exact-resume evidence; use it for review,
with the frozen source/configs for execution. Final JPGs and interpretation
belong in the matching campaign result note. No scientific conclusion yet.

Persistent monitor armed and verified: PID2475035, correct installed run-watch
command with `--dispatch`, first fresh observation2026-10-01T14:45:42Z.
All48 cases are queued, no probe or artifact errors. Polls every300s; only this
watch owns monitoring and bounded recovery/collection. Calibration watch was
never started. Local evidence currently has0/48 training outcomes; queue
forecasts do not establish completed or healthy-running training. Deadline
remains October8 16:42:56 Europe/Paris, budget160GPUh. Optional question about
a later deadline or alternate accelerator is unanswered; no change assumed.
