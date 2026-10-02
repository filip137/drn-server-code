---
id: "exp-019"
title: "Relative endpoint noise eta=5e-4 across all schemes and Conv1/2/3"
status: "complete"
hypotheses: ["H-011"]
---
# Exp019 — Relative endpoint noise eta=5e-4

## Question and dependencies

Filip authorized nine additional training runs on Jean Zay V100 GPUs using both
accounts: baseline, ours and legacy for Conv1/2/3 at relative endpoint noise
eta=5e-4, with 10/30/30 epochs respectively. Seed 0 and initial output D/F=1/4/6
reuse the preceding comparisons. All nine start from the accepted initialization
with uninterrupted Adam; this is the recorded execution interpretation.

The [idea map](../../../ideas.md#where-the-training-advantage-may-peak) records the
retrospective hypothesis that ours has its largest advantage over legacy near
eta=1e-4. This additional level explores the transition between eta=3e-4 and
1e-3; a single added level and seed cannot establish a noise optimum.
Dependencies are the accepted exp013 beta matching, exp014 numerical/noise
contract, exp016 uninterrupted controls, and exp017/018 baseline controls.

## Cases and controls

| Architecture | Initial output D/F | Epochs | Cases |
|---|---:|---:|---|
| Conv1 | 1 | 10 | baseline, ours, legacy; seed 0; eta=5e-4 |
| Conv2 | 4 | 30 | baseline, ours, legacy; seed 0; eta=5e-4 |
| Conv3 | 6 | 30 | baseline, ours, legacy; seed 0; eta=5e-4 |

Use ordinary MNIST with deterministic 55,000/5,000 train/validation split,
shuffle/split seed 0, batch sizes 16/64, float64, centered EqProp, perfect diodes,
weight bounds [0,100], frozen zero biases, and input gains 40/100/360 for depths
1/2/3. T=K=4/6/8 respectively. Preserve each scheme's source LR vector and
accepted beta. Baseline has V/I=1/1, ours 4/1, legacy 4/0.25. No new beta, LR,
T/K or initializer selection. Explicit diode dictionaries remain in every config.

Noise is `v_read = v + eta * abs(v) * xi`: independent Gaussian draws for each
phase/layer/node, same eta at every noninput layer, no absolute floor or additive
component, acquisition seed 2026081601. Inputs, physical relaxation, nudging force
and validation stay clean. Matched standard-normal streams follow the existing
contract. All schemes share the identical seed-0 initializer within each depth.
The official test split is disabled.

## Execution contract

Nine exact configs:
`configs/conv/relative_noise_eta5em4_20260928/conv{1,2,3}_{baseline,ours,legacy}_relative{1,4,6}_eta5em4_seed0.json`,
where relative target follows depth. Each embeds its exact source config and SHA256.
Conv1 derives from exp018 seed-0 configs; Conv2/3 baseline derives from exp017;
Conv2/3 ours/legacy derives from exp016 seed-0 configs. Only eta and study/metric
metadata change from those sources; obsolete clean/control metadata is removed.
Inherited test/canary waiver metadata is removed: there is no current waiver.

Stage these existing assets at the portable config paths:

| Depth | Portable destination | Existing local source | SHA256 |
|---|---|---|---|
| Conv1 | `inputs/initializers/conv1_seed0.pt` | `results/conv1-clean-and-baseline-noise-20260927-v1/source-training-v1/inputs/initializers/conv1_seed0.pt` | `f453d719befc1b81f3f2780fd4c5edc061a73cffdcfeb6a5655ee3bd929d4800` |
| Conv2 | `inputs/initializers/conv2_seed0.pt` | `results/conv23-noiseless-three-seed-20260927-v1/source-training-v1/inputs/initializers/conv2_seed0.pt` | `7717c90d3a7daa6fd9e848469013525beb36d549e39077873c55e0c3ab62aec0` |
| Conv3 | `inputs/initializers/conv3_seed0.pt` | `results/conv23-noiseless-three-seed-20260927-v1/source-training-v1/inputs/initializers/conv3_seed0.pt` | `5e5782bd9bf166b8a432cbf283d745ffe4d25fe4a69ed656f14f9e7e9407392f` |

Dataset transport overrides only its root. Required training assets are
`MNIST/raw/train-images-idx3-ubyte` and `MNIST/raw/train-labels-idx1-ubyte` under
`/home/filip/datasets/mnist` (their gzip copies are also available). Neither
training nor validation reads official test examples.

From the staged source directory, the per-case production command is:

```bash
python -m experiments.exact_run \
  configs/conv/relative_noise_eta5em4_20260928/CASE.json \
  --output-root RESULTS_ROOT --device cuda --dataset-root MNIST_ROOT \
  --checkpoint-every-epoch \
  --target jean-zay
```

`CASE`, `RESULTS_ROOT` and `MNIST_ROOT` are transport substitutions, not literal
paths. Preserve full configured epochs and optimizer/RNG state across any
verified operational resumption. Save model and optimizer checkpoints at every
epoch including initialization. Do not substitute epoch-10 warm starts.

## Metrics and decision

Primary: fixed epoch 10 (Conv1) or epoch 30 (Conv2/3) validation accuracy and
paired ours-minus-legacy / ours-minus-baseline differences at eta=5e-4. Retain
all outcomes, including negative ones; no independent best-epoch selection.
The inherited runner may save its maximum-validation checkpoint, but that is
not the primary comparison. Conv2/3 epoch-10 metrics are secondary.

Compare with previous noise-level curves descriptively. Historical exp015
ours/legacy epoch-30 comparators reset Adam and RNG after epoch 10, whereas all
new cases run uninterrupted; cross-study differences cannot be attributed solely
to eta or amplification. These are single-seed exploratory validation results,
not significance, an established optimum, or official-test paper accuracy.
A positive ours-minus-legacy gap supports transfer at this specific tested cell;
a zero/negative gap does not. Missing/invalid outcomes leave that cell unresolved.
No new global claim threshold is introduced after observing exp014–018.

## Budget, stop and expected outputs

Scientific bound: nine cases, 210 total training epochs, one weight/shuffle seed,
one added noise level. Total cap: **30 GPU-hours**, including smoke, startup and
failed attempts; no budget reset on recovery. Per-job Slurm limits: Conv1 30 minutes,
Conv2 3 hours, Conv3 6 hours (28.5 GPU-hours maximum for nine initial jobs).
Expected longest runtime is 4–6 hours once admitted, based on earlier V100 Conv3
30-epoch runs taking 4h24; queue time is additional. Execution/queue window is
7 calendar days from admission, through **2026-10-05T13:29:10.524563+00:00**. The initial
72-hour planning allowance was superseded before submission because Slurm estimated
October 2 starts. This adds queue allowance only; the 30 GPU-hour cap is unchanged.
`fmu@v100` receives all three Conv1 cases and Conv3 ours/legacy (5 jobs, 13.5h
maximum); `umg@v100` receives all three Conv2 cases and Conv3 baseline (4 jobs,
15h maximum). Both use `gpu_p13`, `qos_gpu-t3`, `v100-16g`, one GPU, one task,
four CPUs, no memory directive, and `pytorch-gpu/py3/2.5.0`. Stop/escalate on nonfinite values, missing identity,
failed smoke, unexpected exit, stalled artifacts or threatened remaining budget;
retain every attempt and do not silently change the scientific config.

Registered result root, including staging/smoke/production and collected evidence:
`results/conv123-relative-noise-eta5em4-jeanzay-20260928-v1/`.
Reviewed result note:
`../results/exp-019-relative-noise-eta5em4.md`.

## Execution and monitoring handoff

Prepared locally on September 28; all nine CPU semantic smokes and canonical
artifact checks passed, as did remote source hashes, wrapper syntax and nine
`sbatch --test-only` requests. All nine production jobs were submitted once on September 28 at 15:29 Paris;
initial live verification showed all nine PENDING (Priority), no duplicates.
Job IDs: Conv1 baseline/legacy/ours = 272329/272330/272331 (`fmu@v100`);
Conv2 baseline/legacy/ours = 272332/272333/272334 (`umg@v100`);
Conv3 baseline = 272335 (`umg@v100`), legacy/ours = 272336/272337 (`fmu@v100`).
Live scheduling advanced sooner than test-only estimates: Conv1 baseline/legacy,
all Conv2 and Conv3 baseline began on September 28 and published real batch
progress. Conv1 ours and Conv3 ours/legacy remain queued at this observation.
Test-only October 2 estimates were provisional; queued jobs have no guaranteed start. Source snapshot
and nine configs: study root `source/`; archive/fingerprint: `transport/source-identity.json`.
The numerical runner, trainer, optimizer and minimizer match the prior exp016 source.
Remote source: `/lustre/fswork/projects/rech/umg/ucy17uy/server_code_sources/conv123-relative-noise-eta5em4-jeanzay-20260928-v1/source`.
Remote result root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/conv123-relative-noise-eta5em4-jeanzay-20260928-v1`.
Wrapper: `transport/run.sh`; nine separate single-task jobs, no arrays. Each writes
`production/ARM/{worker_started,training.log,summary.json,runs/000_*}`. Slurm logs
are under `slurm/exp019-*-JOBID.{out,err}`. The deployment command list and returned
job IDs are preserved in `transport/planned-submissions.json` and
`transport/submissions.json`. Remote dataset: `/lustre/fsn1/projects/rech/umg/ucy17uy/datasets/mnist`.
Local smoke paths: `local-smoke/` and `local-smoke.log` within the registered root.
Persistent run-watch owns routine monitoring: PID **562837**, dispatch enabled,
five-minute cadence. First daemon observation at **2026-09-28T13:31:30.221269+00:00** confirmed
six running jobs with actual epoch-1 batch progress and three queued jobs,
no probe or training errors. Fourteen focused adapter tests passed.
Root startup checks are complete; bounded watcher incidents own subsequent
operational recovery/collection within the original scientific scope and budget. No separate live remote canary is planned.
The same-runner local smoke with these exact configs is required before launch;
use the command above with local output/dataset paths, `--device cpu --smoke`
(or an authorized local CUDA device). Require semantic completion artifacts.
Under repository execution policy, no separate remote canary is submitted absent
a preceding target-specific failure. Preserve identity and finite-progress checks.

Config-only validation passed for all nine: source beta, initializer identity,
all LR/optimizer fields, model/diodes, dataset/split, T/K, calibration proof,
noise contract except eta, and official-test-disabled status are unchanged.
This is not a smoke, deployment or completed-result validation claim.

The coordinator owns launch and monitoring until acknowledged transfer to one
Monitor. Observe actual artifacts and bounded logs at startup and every 30 minutes;
escalate faster on failures. Collect full local coverage before review. Validate
each collected canonical bundle with:

```bash
python -m experiments.reporting validate-run RUN_DIR
```

Reconcile all nine final epoch rows and per-epoch model/optimizer checkpoints,
report failed/replaced attempts, and generate a fixed-epoch comparison table and
JPG figures before writing the result note and refreshing the ledger.

### Latest monitoring observation

September 28 completion collection and review are complete. Fresh Slurm accounting
confirmed all nine jobs 272329–272337 COMPLETED with exit 0:0 and none queued/running.
The exact remote study was rsynced into the registered root's `collected/`;
`analysis/analyze.py` passed 9/9 canonical bundles, all 210 epoch records and 438
model/optimizer checkpoints, with matched initialization and no official-test reads.
The analyzer's policy-field lookup was corrected to the production schema;
initial validation-failure evidence is retained. No scientific settings changed.

Allocation wall time totals 20.4467 GPU-hours including overhead, below the original
30-hour cap (launcher receipts 20.4400 hours; charge the full allocation). All nine
production cases ran once, with no retries or remote canary; prelaunch smokes were CPU.
Receipts, complete measurements and the JPG are in `analysis/` under the study root.
See the [reviewed result](../results/exp-019-relative-noise-eta5em4.md) for scoped
interpretation. The bounded incident returns complete to the sole Python watcher;
no further exp019 checks or launches are needed.

### Persistent monitor and collection

The single intended owner is `monitor/watch.json` under the registered study root,
using the installed run-watch service and the read-only `monitor/probe_slurm.py`.
Check every five minutes; queued Slurm jobs are not missing jobs. Probe scheduler
identity/account, wrapper state, run status, real batch/epoch progression and error
logs. Enforce a 45-minute active progress limit. Completion dispatch occurs only
after all nine expected arms have terminal artifacts and successful accounting.
A bounded incident agent handles operational diagnosis or final collection/review;
no training runs or duplicates are launched by the read-only probe.

Before interpreting results, collect the exact remote root into local
`results/conv123-relative-noise-eta5em4-jeanzay-20260928-v1/collected/` with rsync,
retaining remote results. Reconcile elapsed allocation times from `sacct`, failed
attempts and the 30 GPU-hour cap. Run the study-local analyzer:

```bash
/home/filip/miniconda3/envs/py312/bin/python \
  results/conv123-relative-noise-eta5em4-jeanzay-20260928-v1/analysis/analyze.py
```

Require nine validated fixed-epoch outcomes or explicitly recorded scientific
failures; never call missing coverage complete. Record findings in
`../results/exp-019-relative-noise-eta5em4.md` and regenerate the campaign ledger.
Watcher PID and verified first observation are recorded above; monitor state is
in `monitor/state/`, with `monitor/startup-status.json` as the startup receipt. There is no
external messaging callback; no automatic chat notification is claimed.
