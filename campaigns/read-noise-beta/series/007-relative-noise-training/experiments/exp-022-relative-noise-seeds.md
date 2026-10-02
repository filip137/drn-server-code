---
id: "exp-022"
title: "Two additional full-training seeds for relative read noise"
status: "complete"
hypotheses: ["H-011"]
---
# Exp022 — Read-noise seeds 1 and 2

## Question and adopted decisions

Filip approved implementation September 30: repeat the original batch-16
comparison at eta=1e-5, 1e-4, 3e-4 and 5e-4 with two full-training seeds.
Baseline, ours and legacy × Conv1/2/3 × four etas × seeds1/2 gives **72 cases**,
1680 training epochs. This tests whether the earlier scheme differences recur
under independent initialization, shuffle and read-noise draws. Link H-011 for
the original epoch10 transfer question; epoch30 replication is an extension,
not a retrospective change to that hypothesis.

All new runs use uninterrupted Adam from initialization. Filip chose only the
72 new runs: do not replace historical seed0. Conv2/3 ours/legacy seed0 at the
first three etas reset Adam/RNG after epoch10 and must remain separate from
new uninterrupted-seed averages and paired comparisons.

## Scientific contract

| Architecture | Cases | Epochs | Seed0-reference D/F | T=K | Input gain |
|---|---:|---:|---:|---:|---:|
| Conv1 | 24 | 10 | 1 | 4 | 40 |
| Conv2 | 24 | 30 | 4 | 6 | 100 |
| Conv3 | 24 | 30 | 6 | 8 | 360 |

Use accepted exp016/018 seed1/2 initializers and config contracts, and exp019's
nodewise relative endpoint noise. Model/shuffle seeds1/2 correspond to read-noise
seeds2026081602/2026081603. Within each depth/seed, all schemes and etas share
the initializer and matched random streams. Split seed stays0. Preserve source
scheme-specific LR vectors, beta, model, perfect-diode dictionaries, centered
float64 EqProp, frozen-zero biases, weight bounds[0,100], and batch16/validation64.
Noise is v_read=v+eta*abs(v)*xi, independent across phases/layers/nodes, with no
absolute component/floor. Inputs, relaxation and validation stay clean.
The 55k/5k ordinary-MNIST split is fixed; official-test evaluation is disabled.
Frozen betas reference seed0 calibration; actual seed1/2 D/F is unmeasured.

## Execution contract and evidence paths

Preparation: `python -m experiments.prepare_relative_noise_seeds`.
Configs: `configs/conv/relative_noise_seeds12_20260930/`.
Registered result root (all preparation, smokes, attempts and collected evidence):
`results/conv123-relative-noise-seeds12-jeanzay-20260930-v1/`.
Reviewed result note: `../results/exp-022-relative-noise-seeds.md`.

Freeze the effective source including relevant uncommitted files, configs and
initializer hashes. Run `experiments.exact_run CONFIG --index 0 --device cuda
--checkpoint-every-epoch --epoch-chunk-size N --target jean-zay`, with N equal
to the full10/30epoch horizon and explicit output/data-root transport paths.
Save epoch0..N model/Adam/global and private RNG state. No planned epoch10
restart. Operational continuations require exact complete epoch-state recovery;
do not use approximate noise-stream resets. Preserve prior attempts and check
the old worker terminal before any replacement.

## Metrics and review

Primary: final epoch10/30/30 validation accuracy, per-seed values, mean/sample
standard deviation over seeds1/2, and paired ours-minus-legacy/baseline differences.
Report signs and magnitudes for every cell; mixed signs indicate inconsistent
replication, not a hidden selected win. Best accuracy and Conv2/3 epoch10 results
are secondary. No significance claim from two seeds. Three-seed supplementary
summaries may include compatible uninterrupted seed0 evidence only; paired
comparisons must use a common seed set. Preserve hardware/source differences.
Retain numerical failures as scientific outcomes, never silently exclude them
to improve averages. Missing/invalid evidence leaves the affected comparison open.
Generate tables and matplotlib JPGs; validate all successful canonical bundles
and reconcile all72 terminal outcomes before review and ledger refresh.

## Budget, gates and stop rules

Expected164GPUh from exp019's measured V100 timings; cap240GPUh including all
overhead/failures/recovery. Initial job upper bounds total228GPUh. Seven calendar
days from first submission; record resolved timestamps in transport/window.json.
Per-case limits: Conv1 30min, Conv2 3h, Conv3 6h. Submit independent single-case
jobs, seed1 on fmu@v100 and seed2 on umg@v100; each gpu_p13/qos_gpu-t3/v100-16g,
one GPU, one task, four CPUs, no memory directive, pytorch-gpu/py3/2.5.0.
Slurm account limits control concurrency. Queued jobs are not absent workers.

Check frozen config/input identity, same-config local semantic smokes, wrapper
syntax, relevant runner/restore tests and scheduler test-only commands. Repository
execution policy supersedes the older global live-canary prerequisite: restore
a live target gate only following an applicable target-specific production failure.
No beta/LR/TK retuning, gradient/residual audits, batch changes or extra seeds.
Escalate nonfinite, unexpected exit, missing identity,45min active progress stall,
or budget/deadline risk. Do not extend the budget on retries; scientific failure
is terminal. Allow one understood operational retry per affected case within the
remaining allowance, with full-state continuation if training began.

## Execution and monitoring handoff

Current owner: root Reviewer; collection, validation and scientific review complete.
The hourly watcher was stopped cleanly with no incident inflight to avoid duplicate transfers.
Root completed startup checks and transferred monitoring after its first fresh probe. Exact submission commands/handles will live under
`transport/planned-submissions.json` and `transport/submissions.json` in the study.
Remote source: `/lustre/fswork/projects/rech/umg/ucy17uy/server_code_sources/conv123-relative-noise-seeds12-jeanzay-20260930-v1/source`.
Remote result root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/conv123-relative-noise-seeds12-jeanzay-20260930-v1`.
Remote dataset: `/lustre/fsn1/projects/rech/umg/ucy17uy/datasets/mnist`.
Each arm writes `production/ARM/{worker_started,training.log,summary.json,runs/000_*}`.

Use one persistent run-watch service at monitor/watch.json,5min cadence,45min
semantic-progress threshold. Verify its PID, first real observation, dispatch
and single ownership before handoff. The monitor owns bounded recovery and
collection/review dispatch; there is no external messaging callback. Collect
the exact remote root into local collected/ without removing remote evidence.
Run `python -m experiments.analyze_relative_noise_seeds` and canonical
`python -m experiments.reporting validate-run RUN_DIR`; retain failures and all
allocation charges. The agent must acknowledge monitoring before root relinquishes
ownership. Scheduling is not completed scientific evidence.

### Preflight identity

Frozen numerical source: study source-v2/ (source/ is a convenience symlink).
Archive SHA256 `524c978c8fd3831741c0b0b3cb9e1eba374c2f6202681b0005309db0f57d566b`;
commit `08aad531a4f1fe809c37eb5289d26f4ffd5cbddc` plus recorded uncommitted content.
All82 focused frozen-source tests and72 remote scheduler test-only requests pass.
The final source is staged and hashed remotely; final-config CPU smokes are in
local-smoke-v2/. Initial preparation had stale nested test-waiver metadata; its
72 successful CPU smokes are preserved under local-smoke/, and its source/configs
under superseded-preparation-v1/. None was submitted. No numerical fields changed.
Study recovery transport has24 passing tests and monitor adapter20; both are
outside frozen scientific source. The ledger validator now accepts the existing
ready-for-review lifecycle state;18 generator tests pass, preserving exp021 state.

### Release gate

Final72/72 CPU semantic smokes passed, each3epochs with2training batches per
epoch and1validation batch. Canonical bundles, exact config identities and
all288 epoch0..3 checkpoint pairs passed; every trained checkpoint preserves
Adam steps, global RNG, sampler/worker RNG and private noise RNG/draw count.
Receipt: transport/local-gate.json. A smoke-only validation lookup was corrected
to require both configured policy disabled and runtime skipped_for_diagnostic;
its initial failure is retained and no scientific config changed.

All72 remote test-only requests passed, commands frozen in
transport/planned-submissions.json. Admission window opened2026-09-30T17:00:22Z;
Slurm deadline2026-10-07T19:00:22 Europe/Paris, matching transport/window.json.
No separate remote canary: the accepted independent-job transport is unchanged
except the reviewed full-state checkpoint flag and optional-task env handling.
Collector tests16/16 pass;24 compatible historical seed0 cases validate separately.

### Submission

All72 initial jobs were submitted once on September30, distributed36 per account.
IDs range384676–384754 with gaps from unrelated submissions; the exact72-ID map
is transport/submissions.json, not that inclusive range. Root retains startup
ownership until the persistent monitor has a fresh observation and verified PID.

### Verified monitoring handoff

Persistent watcher **PID1920169**, dispatch enabled, confirmed alive on the host.
First daemon observation **2026-09-30T17:07:29.767624+00:00**: **8 running,64 queued**, no probe
failures or training errors. The running jobs published manifests/checkpoints and
advanced to at least epoch1 batch1000; progress differs from the initial direct
probe, satisfying the second startup check. Next poll **2026-09-30T17:12:29.767624+00:00**,
then every5min. Charged0.16GPUh plus227.84GPUh outstanding upper bounds totals228,
below240GPUh. Monitor receipt: monitor/ownership-receipt.json; full observations:
monitor/startup-status.json and monitor/state/. No second polling owner remains.

The active service owns future bounded operational recovery and completion dispatch.
Use the prepared transport/resume_entry.py for exact-state recovery, including its
fresh terminal-observation guard and dataset-root correction; preserve all attempts.
Collection command: `/home/filip/miniconda3/envs/py312/bin/python -m
experiments.analyze_relative_noise_seeds` after syncing into collected/. All60 combined
collector/monitor/recovery tests pass. No official-test evaluation or completed
scientific outcome is claimed at this handoff.

Final startup resource verification:72/72 live scontrol records match account,
partition/QoS, v100-16g,1GPU/1task/4CPUs and per-depth wall limits. At that
observation15 jobs were RUNNING and57 PENDING;10 had already written verified
V10016GiB gpu.json files, with the other newly admitted workers starting.
Receipt: transport/live-resource-verification.json. This later scheduling count
supersedes the8/64 count in the first daemon observation; sole monitoring
ownership remains with PID1920169. Initial bulk scontrol query syntax was rejected
by this Slurm version; individual read-only queries passed, with no job changes.

### October1 — user-requested hourly monitoring

Filip requested checks about hourly and a completion notification. The existing
watch was stopped cleanly and restarted as PID2145368 with poll_seconds=3600,
dispatch enabled and all state/incident/budget history retained. No training job
was changed. The hourly cadence supersedes previous five-minute instructions.
Keep the45min semantic-stall criterion; assess it at the hourly checks.

Latest successful prior observation at2026-10-01T07:48:48Z found71 complete and
one running: Conv3 legacy eta1e-5 seed1, epoch10 validation starting,5782seconds
charged for that allocation. Total study charge162.650833GPUh, below240GPUh.
Transient SSH timeouts had delayed some probes; the latest prior probe succeeded.
Completion still requires collection and local validation/review.

Notification delivery is **pending the user’s channel choice**. The background
watcher cannot proactively post in this chat, and no email/Slack/desktop callback
is configured. A concise channel question was sent; do not claim delivery armed
until a real destination and mechanism are configured and verified.

### October1 — all training terminal, collection in progress

The2026-10-01T10:51:29Z watcher observation found72/72 completed artifacts and
165.6375GPUh charged, below240GPUh. Its bounded collection attempt copied1.8GiB
but used an unnecessarily short20s transfer limit; no training failed. On Filip’s
update request, root took sole collection/review ownership and resumed the exact
rsync with an inactivity timeout, preserving partial and remote evidence. Local
validation and scientific review remain pending; do not call the study complete yet.

### October1 — completed and reviewed

All72 original allocations completed successfully; final end12:48:03 Europe/Paris,
165.6375GPUh charged. Full collection (17,747,129,975 bytes) succeeded, then the
collector validated72/72 cases,1680epochs and1752 model/optimizer checkpoint pairs
with no errors. The separate24-case compatible seed0 supplement also validates.
Analysis-only CLI/config transport assumptions were corrected with23 regression
tests; original failures remain in analysis/validation-corrections.json.

[Reviewed result](../results/exp-022-relative-noise-seeds.md) owns measurements,
interpretation and limits. Primary evidence is seeds1/2 only; no official test was
read. The hourly watcher remains stopped because training and review are complete.
Filip's update request is answered in this chat; no external notification channel
was configured. No next poll or outstanding experiment action remains.
