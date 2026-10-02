---
id: "exp-018"
title: "Ours CIFAR batch size versus time and accuracy"
status: "blocked"
hypotheses: ["H-012"]
---
# Batch-size comparison

Question: can larger batches shorten ten-epoch ours CIFAR training without a
material validation penalty? Filip requested batches128/256, selected ours, and
authorized V100 execution. Use fresh initialization, trainable BN affine and the
latest ours readout LR1.9758828827095137e-6. Other scientific fields inherit
exp-017: seed0, 45k/5k, T/K=[6,6,4], fixed convolution/gain/BN rates, original
50-epoch cosine horizon. No smokes, LR scaling, gradient accumulation or prelaunch
scientific audits. Each case uses the same original epoch-zero model and split.

Configs: `configs/cifar/batch_size_20260928/ours_bs{32,64,128,256}.json`.
Result root registered before creation:
`results/cifar10-ours-batch-size-20260928-v1/`.

Compare terminal full-validation accuracy/CE, training CE, total and epoch times,
examples/second, peak GPU memory and actual Adam steps. Proposed exploratory
success criterion: >=10% lower mean epoch time (epochs2–10) than batch32 on the
same GPU, with accuracy loss <=0.5pp. One seed is provisional evidence. Fixed
epochs intentionally change optimizer-update count and BN statistics; this is an
operational batch-size comparison, not a test of equal-update optimization.

## Execution and monitoring handoff

Planned one sequential Slurm allocation on Jean Zay, account fmu@v100,
partition gpu_p13, qos_gpu-t3, explicit constraint v100-32g, one GPU, ten CPUs,
20-hour combined hard ceiling. Estimated10–16 GPU-hours if only32/64 fit;
large-batch runtime remains unmeasured. Native cases each target ten epochs.
Attempt128 then256; preserve CUDA-OOM bundles as capacity exclusions, not accuracy
results. If both fail specifically with CUDA OOM, run64 as the practical fallback;
run32 as the matched control. Abort on other failures, nonfinite values or the
shared deadline. No automatic CPU offload, retries or budget extension.
The previous batch64 legacy peak21.47GiB suggests128/256 may not fit, but that
extrapolation is not a direct ours measurement. CPU offload remains a separate
execution choice pending Filip's response; no offload case is currently assigned.

Prepared source/assets/launch live under this result root. Remote counterpart:
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/cifar10-ours-batch-size-20260928-v1/`.
Persistent run-watch will inspect Slurm plus native case progress every5min and
collect/review on termination. Slurm job289265 accepted and observed PENDING. Persistent run-watch PID770633 owns monitoring; verified live outside the sandbox process namespace with a successful scheduler observation and zero probe failures. Config: `results/cifar10-ours-batch-size-20260928-v1/monitor/watch.json`. Poll every5min; queue wait allowed through October5 18:00 Paris. This scheduled V100 assignment is not overnight-bounded; its single allocation may continue during the day within the declared20-hour cap.

Collect with rsync from the remote counterpart into the same local root, without --delete. Then run `/home/filip/miniconda3/envs/py312/bin/python results/cifar10-ours-batch-size-20260928-v1/analysis/review.py`; inspect the result note and refresh/check the campaign ledger. Expected OOM attempts remain capacity exclusions. Unexpected failures require local partial coverage and an incident report; no retries are assigned.


### Terminal incident — 2026-09-29

Job289265 is FAILED, exit1:0; fresh sacct confirms node r6i6n7 and
00:57:04–00:57:32 UTC (28 seconds allocation wall time); squeue has no entry.
Batch128 used CUDA-visible device0 and failed before any completed epoch with a
non-OOM deterministic CuBLAS RuntimeError (missing required
`CUBLAS_WORKSPACE_CONFIG`). The mandatory abort prevented256 and32; conditional64
was not eligible because both larger cases did not CUDA-OOM. No retry is authorized.
Collected only the registered remote root into the identical local result root
with rsync -a, without --delete, successfully. Preserved initial checkpoint,
empty metrics, failed status, source, allocation and logs. Local failure-coverage
checks passed; complete-allocation review.py is inapplicable and was not run.
Charged allocation time is28 seconds including startup/teardown, not the shorter
inner-runner interval; unused budget does not authorize a retry.
Run-watch retains passive monitoring; incident decision is blocked.
See [reviewed partial evidence](../results/exp-018-batch-size.md) and
`results/cifar10-ours-batch-size-20260928-v1/analysis/incident-289265.json`.

### Coordinator recovery — September29

The missing deterministic CuBLAS environment variable is an operational launcher
error. The earlier no-retry restriction was coordinator-imposed, not a user
restriction; standing experiment-lifecycle authority covers this specific repair.
Preserve job289265 and all v1 evidence. One replacement attempt uses identical
science/source/assets with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before Python.
No smoke is added, respecting Filip's explicit instruction. New result root:
`results/cifar10-ours-batch-size-20260928-v2/`, with its identically named remote
counterpart. Allocation limit19:59:00 plus the original28s remains below20h;
controller deadline is shortened accordingly. Same conditional cases/target.
No automatic retries by the Monitor; escalate any new failure to coordinator.
Replacement Slurm job294989 submitted and observed queued. Persistent monitor PID948699 verified alive; config `results/cifar10-ours-batch-size-20260928-v2/monitor/watch.json`. It owns observation/collection and checks every5min; v1 passive monitor stopped. Recovery collection/review use the v2 root and its `analysis/review.py`. Historical failed result
note describes attempt1 only; no scientific verdict is possible yet.

### Move to two lab5090 workers — September29

Filip explicitly requests local5090s, two in parallel. V100 job294989 confirmed
CANCELLED with zero allocation time; v2 monitor stopped. New registered result
root `results/cifar10-ours-batch-size-5090-20260929-v1/`.
Worker candidate on Fifi attempts128 then256 and runs64 if both CUDA-OOM.
Worker reference on Loulou runs32 concurrently. Same fresh ten-epoch configs,
trainable BN and unchanged rates; no smokes/offload. Fifi is currently idle;
Loulou shares with identified Ben processes within available memory. Preserve
those workloads. Timing comparisons across shared versus idle GPUs are descriptive
and cannot alone establish H-012's matched speed criterion. Accuracy remains
comparable with the stated single-seed/environment limitations.
Budget10h candidate plus9h reference, plus28s original failure, below20h total;
expected2–4h per worker, with shared runtime uncertainty, deadline October1 08:00
Paris. Shared run-watch queue becomes sole owner; recovery may repair routine
operational faults and resume compatible full-state checkpoints within each
original budget, preserving all failed attempts. No scientific retuning or blind
replays. Queue jobs `cifar-batch-candidate-5090-20260929` (Fifi) and `cifar-batch-reference-5090-20260929` (Loulou) submitted. Shared queue service owns placement, 60s monitoring, bounded operational recovery and CPU collection/review. Specs: `results/cifar10-ours-batch-size-5090-20260929-v1/launch/{candidate,reference}.json`. Native cases live under `lanes/{candidate,reference}/cells/`; expected CUDA-OOM attempts remain preserved there. Controller collection validates coverage and Adam/BN updates, then writes the reviewed result with timing-contention limitations.


### Replacement terminal incident — September29

Fresh Slurm accounting confirms job294989 CANCELLED by UID306943 before start,
with no assigned node, zero allocation seconds, and no squeue entry. Scheduler
end timestamp: 2026-09-29T07:22:28 (cluster local time). The reason for cancellation
is not established. No replacement or retry is authorized to this incident agent.
Collected only the registered v2 remote root into the identical local root using
rsync -a without --delete (exit0). Local validation confirms no allocation.json
and no cells directory: batches128,256,32 never started; conditional64 was not
eligible. No training metrics, checkpoints or complete coverage exist in v2.
Total charged allocation time remains28 seconds across v1 and v2; unused budget
does not authorize resubmission. Complete-allocation review.py is inapplicable.
Evidence: `results/cifar10-ours-batch-size-20260928-v2/analysis/incident-294989.json`.
Decision: blocked; run-watch retains passive observation. H-012 remains
inconclusive; see the existing result note. No other jobs were changed.

Startup verified: Fifi batch128 and256 both raised actual CUDA OOM on an otherwise idle RTX5090; preserved native failed bundles/logs. Fifi automatically entered batch64 and completed its first training update. Loulou batch32 reached epoch1 batch100/1407. Both workers are training concurrently under shared-queue ownership. These are capacity measurements, not accuracy exclusions inferred from estimates.

### Scope reduction — September29

Filip: “just do64, cancel32”. Cancellation requested through the owning queue for
`cifar-batch-reference-5090-20260929`; do not restart or replace that control.
`cifar-batch-candidate-5090-20260929` continues unchanged on Fifi at batch64 toward
10 epochs. Batch128/256 CUDA-OOM evidence remains preserved; no further attempts.
Controller-only analysis/review.py now waits only for the candidate collection;
it reports64 descriptively and marks H-012 inconclusive without a complete32
reference. Training source/configs and the active worker are unchanged. Preserve
cancelled32 artifacts separately, excluded from ten-epoch scientific comparisons.

Cancellation confirmed: batch32 worker stopped, queue state cancelled, reservation released. Native pause records epoch0 (no completed epoch). Batch64 remained running at epoch1 batch600/704.

Filip authorized20 additional epochs after this run. The separately queued full-state continuation to epoch30 is owned by [exp-019](exp-019-batch64-continuation.md); this ten-epoch run and its checkpoint remain unchanged.

Authorized batch64 worker collected and validated; batch32 cancelled at user request; see [review](../results/exp-018-batch-size.md).
