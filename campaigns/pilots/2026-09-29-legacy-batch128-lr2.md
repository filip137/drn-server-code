# Legacy batch128 doubled learning-rate pilot

Status: both cases complete, collected, validated and reviewed September29,10:35Paris.
Both original runs launched09:24Paris; Conv3 continued on loulou after epoch2.

## Question and frozen design

Filip requested legacy Conv2 and Conv3 BPTT training for 10 epochs with twice
the original learning rates, using the local GPU and nom-cool-1, to test whether
the larger rates improve results. This is a two-candidate pilot, not a broader
automatic LR sweep. Inherit batch128, seed0 and noiseless training from
[exp021](../read-noise-beta/series/007-relative-noise-training/experiments/exp-021-batch128-read-noise.md).

Copy the accepted exp021 legacy clean BPTT configs and frozen numerical source.
Multiply every weight LR by exactly two, keep all bias LRs and biases zero,
and set the horizon to 10 epochs. Preserve initializers, Adam settings, legacy
amplification, explicit perfect-diode parameters, float64, wide bounds [0,100],
T/K6/8, input gains100/360, deterministic ordinary-MNIST55k/5k cohort,
training/validation batches128/64 and seed0. No official-test access.
Start fresh from the matched initializer, with fresh Adam, not from trained weights.

Primary comparison: fixed epoch10 validation accuracy against the original-rate
legacy BPTT epoch10 checkpoints/metrics from exp021. An accuracy increase is a
descriptive improvement for this seed; report its magnitude and loss alongside
finite training and runtime. Preserve ties and negative outcomes. Do not substitute
best-epoch accuracy or noiseless EP for the BPTT control. The Conv2 original-rate
epoch10 accuracy is97.42%; Conv3's control was still queued at preparation and
must be collected before its LR verdict. Hardware differs from the V100 controls,
so retain host/source caveats. No claim of an optimal LR or statistical significance.

## Budget and execution

Filip subsequently requested moving Conv3 to an RTX5090. This expands its allowed
targets to an available configured lab5090 under normal ownership/memory admission
and the weekday two-device fleet cap. Preserve the latest complete epoch and
model/Adam state; no restart from initialization or extra epochs. Prepare and
verify the destination, then cancel only the original Conv3 queue worker through
its owner and verify it stopped before launching the replacement. Retain and
charge that attempt and all migration checks within the original4GPUh Conv3
allowance. The existing diagnostic resume bridge reconstructs missing private
RNG and labels the continuation approximate; the hardware/framework change is
an additional comparison caveat. Conv2 remains on nom-cool-1.

Two GPU jobs in parallel on the requested RTX3090 targets: prefer Conv3 local
and Conv2 nom-cool-1. Both have24GiB; check full-batch memory before production.
Ben's identified nom-cool-1 workloads may be shared when peak plus headroom fits;
leave those jobs and MPS settings unchanged. Do not reduce batch size or precision
to obtain a fit. Use the existing exact_run runner, explicit conv2/conv3 queue
profiles and a single shared-queue monitoring owner.

Operational ceiling6GPUh total (Conv2 up to2h, Conv3 up to4h), including smokes,
failed attempts and validation overhead. Preliminary expectation about1–2h in
parallel after admission, to be replaced by actual throughput; sharing may extend
the duration. No legacy 09:00 deadline applies to this new request. Timeout
recovery, if needed within the same budget, restores the latest complete
model/Adam checkpoint for only the remaining original epochs. Full private RNG
restoration was planned, but the launched commands omit `--epoch-chunk-size`:
they save global RNG plus model/Adam each epoch, without sampler/worker private
RNG state. Keep the healthy uninterrupted runs. If a timeout occurs, the user's
standing saved-epoch recovery instruction permits the existing explicitly
approximate fallback; record missing private RNG and never claim exact trajectory
continuation. No cancellation or replay is justified solely by this limitation.
Preserve failed
attempts and diagnose OOM/nonfinite without changing scientific settings.

Result root, including every smoke and attempt:
`results/conv23-legacy-batch128-lr2-20260929-v1/`.
Configs: `configs/conv/legacy_batch128_lr2_20260929/`.
Use `source-training-v1/` beneath the result root for frozen execution, keeping
unrelated current worktree edits out of the scientific snapshot.

## Execution and monitoring handoff

Both cases passed synchronous full-batch CPU and target CUDA smokes, with
canonical local validators reporting zero errors. CPU receipt:
`local-smoke/validation.json`; CUDA receipt: `gpu-smoke/validation.json` beneath
the result root. Conv2 on nom-cool-1 used722MiB allocated/792MiB reserved;
Conv3 local used1911MiB/1968MiB. No change to batch128 or float64 was needed.
Allocator setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and py312
PyTorch2.5.1+cu121 are used on both targets. Conv2 shares with Ben's identified
mumax3 jobs and MPS; their settings and jobs are untouched.

Frozen source identity SHA256:
`be051de7192c741934c13a52669840f69062b2dd7e7acc3aae266aa6d1af9cfa`.
`preparation.json` records the499 unchanged numerical source files and matched
initializers. Source identity takes precedence over ambient parent-worktree Git
metadata. Exact commands and prepared-file hashes are in `launch/jobs/`.

| Case | Shared queue ID | Target | Production allowance |
|---|---|---|---|
| Conv2 | `legacy-conv2-bptt-lr2-b128-s0-20260929` | nom-cool-1 |7140s|
| Conv3 | `legacy-conv3-bptt-lr2-b128-s0-20260929` | local |14340s|

Both launched at09:24Paris. The60s per-case reserve includes the short GPU smoke
and startup allowance within the original2h/4h shares. Production paths follow
`cases.json`: `full/<case>/attempt-0/runs/<bundle>/`. Remote root is
`/home/filip/server_code/results/conv23-legacy-batch128-lr2-20260929-v1/`;
Conv2 terminal evidence is copied to local `collected/nom-cool-1/full/`.

The existing shared run-watch servicePID585501 is the sole routine monitoring
owner, with60s cadence, verified admission receipts and live workers. Inspect:
`python /home/filip/.codex/skills/run-watch/scripts/gpu_queue.py status JOB_ID`.
`launch/*-status.json` and `launch/monitoring-handoff.json` retain compact startup
evidence. No second watcher is assigned. Fresh queue observation at09:26:29Paris confirms both worker processes alive
and their first full128-example optimizer batches finite (loss0.5027 Conv2,
0.4946 Conv3). First epoch timing is pending; the1–2h estimate stays provisional.

The queue invokes `collect.py` on CPU after each successful target validator;
it collects remote bundles, runs the frozen `experiments.reporting validate-run`,
and verifies10epochs,4300Adam updates,11checkpoints including epoch0, finite
terminal metrics and zero official-test evaluations. Local receipts appear at
`launch/collected/JOB_ID.json`. To repeat a failed collection safely, use
`gpu_queue.py collect JOB_ID`; collection failure must not rerun GPU training.

Unexpected exit, numerical error, missing progress or budget pressure escalates
to the Codifier through the queue's bounded recovery. These initial commands
omitted `--epoch-chunk-size`; per-epoch checkpoints retain model, Adam and global
RNG, but not private sampler/worker RNG. `exact_resume_available:false` is recorded
in the startup and collection receipts. Under the previously authorized approximate
saved-epoch fallback, a timeout continuation may restore the latest complete
model/Adam checkpoint after verifying the original attempt stopped, label the
missing private RNG explicitly, and use only remaining original epochs/budget.
Healthy jobs are preserved. No fresh restart, scientific change or extra case. Root owns comparison/review;
Conv3's original-rate epoch10 control may still be pending independently.

### Conv3 requested migration to RTX5090

The local Conv3 attempt was stopped through its own queue cancellation after
hash-validating completed epoch2, including Adam step860. Queue supervisor and
all matching Python descendants were verified stopped before migration. The old
attempt is preserved; its charged runtime is987.37seconds. Full validation and
stop receipts are `launch/migration-selected-checkpoint.json`,
`launch/migration-stop-verification.json`, and `launch/migration-old-status.json`.
Conv2 remains unchanged on nom-cool-1.

The unchanged source and stopped bundle were staged on loulou. Its environment
is PyTorch2.11.0+cu128 with RTX5090. The existing diagnostic-resume helper passed
a CPU restore gate on that host:2epochs complete,8remaining,next epoch3,model
and Adam restored, explicitly approximate stochastic continuation. The original
scientific config and10-epoch horizon are unchanged; resumed execution adds
`--epoch-chunk-size 10` to save complete future continuation state.

Prepared job: `legacy-conv3-bptt-lr2-b128-s0-20260929-loulou-resume`, with13320s
remaining production allowance after charging the local attempt and smoke
reserves. Its queue profile remains `conv3`. The permanently updated queue policy uses
measured peak plus headroom, preserving one worker and the weekday fleet limit.
Policy receipt: `launch/reservation-policy-update.json`.
Source input: remote `migration-input/attempt-0/`; new output:
`full/conv3_legacy_bp_eta0_lr2_batch128_seed0/attempt-1-loulou/runs/`.
The resumed job is now running on loulou RTX5090: the worker explicitly reports
continuation after epoch2 and actual epoch3 batch1/430, loss0.0466,accuracy98.44%.
Queue childPID407449 and trainingPID407489 are live; observed process memory2728MiB.
The restarted shared queuePID1124310 is the sole monitoring owner. Fresh service,
worker and budget evidence: `launch/migration-monitoring-handoff.json`. The CPU
completion collector now collects loulou outputs and validates the complete
10-epoch history including its restored two-epoch prefix. Approximate migration
provenance remains visible even though future checkpoints save private RNG state.

## Results and review

Both authorized cases completed10epochs and passed target and CPU collection
validation: finite metrics,4300Adam updates,11indexed checkpoints including
initialization, zero biases and no official-test evaluations. Shared queue jobs
and their completion hooks are terminal complete. Conv2 finished09:40:52Paris;
Conv3 finished10:21:24Paris. No further pilot simulations are authorized.

| Model | Original LR, epoch10 | Doubled LR, epoch10 | Difference | Validation loss original → doubled |
|---|---:|---:|---:|---:|
| Conv2 |97.42%|97.82%|+0.40pp|0.059650 → 0.052253|
| Conv3 |98.20%|98.46%|+0.26pp|0.035572 → 0.031693|

These are batch128, seed0, noiseless BPTT validation results. Both original-rate
controls use their actual epoch10 metrics from exp021, not their30-epoch endpoint
or an EP substitute. Config comparison verified unchanged cohort settings,
seed/initializer references, parameter order and float64, and exactly doubled
LR vectors; dataset-root overrides differ by host. Evidence and exact bundle
paths: `analysis/comparison.json` and `launch/collected/*.json` under the pilot root.

Both doubled rates improve fixed-epoch accuracy and validation loss for this seed.
This supports the proposed increase as a candidate at batch128, not an optimum,
statistical significance or a30-epoch/noisy benefit. Controls ran on V100s;
Conv3 also migrated after epoch2 with reconstructed private RNG, so these are
comparisons with execution differences. Scientific settings were unchanged at
migration, and completed training/Adam history was retained.

Median completed-epoch intervals: Conv2 on nom-cool-1,92.48seconds; Conv3 on
loulou,224.89seconds. Conv3's original local epoch1→2 interval was455.13seconds,
so the observed5090 continuation was about2.02× faster per epoch. Sharing and
hardware/framework differences prevent treating this as a controlled benchmark.
The independent exp021 campaign remains active under its existing monitor.
