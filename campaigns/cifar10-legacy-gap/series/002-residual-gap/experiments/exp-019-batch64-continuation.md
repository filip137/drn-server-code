---
id: "exp-019"
title: "Continue ours batch64 from epoch10 to epoch30"
status: "cancelled"
hypotheses: []
---
# Batch64 continuation

Filip requested20 more epochs after the current10-epoch run. Continue the exact
ours batch64 trajectory from exp-018's final epoch10 checkpoint to epoch30.
Preserve model, Adam moments/step, original50-epoch cosine schedule, RNG, trainable
BN affine/running state, seed0, data split, T/K and all learning rates. No restart,
smokes, offload or new scientific diagnostics. Batch32 remains cancelled.

Result root registered before creation:
`results/cifar10-ours-batch64-e10-e30-20260929-v1/`.
Config template: `configs/cifar/batch64_e10_e30_20260929/ours.json`.
The launcher adds the actual final checkpoint SHA to the runtime config only
after verifying the parent has completed epoch10 and7040 optimizer steps.
Parent: `results/cifar10-ours-batch-size-5090-20260929-v1/lanes/candidate/cells/ours_bs64`.
Report epoch30 validation accuracy/CE, training CE and full epoch11–30 curve,
relative to the same trajectory at10. This is one continuation, not a replicate
or a completed batch-size comparison.

## Execution and monitoring handoff

One shared-queue job on Fifi RTX5090, same device as parent, default exclusive
queue profile. Current parent reserves this GPU until its worker/validation ends;
the continuation's launcher also requires a completed parent bundle before
training. Expected4h15m plus overhead, cap8 GPU-hours, deadline October1 08:00
Paris. Peak22GiB plus2GiB headroom. Result path remote prefix:
`/home/filip/server_code/results/cifar10-ours-batch64-e10-e30-20260929-v1/`.
Shared queue owns monitoring, operational recovery and collection. Recovery may
repair launch faults or resume full-state checkpoints within the original budget;
no scientific changes or blind replays. Validate epochs11–30, all45k/5k examples,
Adam/scheduler and BN steps, restored parent identity. Collect and review in this
experiment's result note. Queue job `cifar-ours-batch64-e10-e30-20260929` accepted, currently queued behind the active Fifi worker. Persistent queue service PID585501 has a fresh observation; no second monitor. Parent last observed epoch9 batch200/704. Spec: `results/cifar10-ours-batch64-e10-e30-20260929-v1/launch/ours.json`. Inspect with `gpu_queue.py status cifar-ours-batch64-e10-e30-20260929`. The CPU completion hook collects and validates epoch11–30 and writes the result note.

## Cancellation requested — September29

Filip explicitly cancelled batch64. Cancellation requested through queue job
`cifar-ours-batch64-e10-e30-20260929`. No further training, retries or continuation
are authorized. Preserve all completed-epoch checkpoints and partial metrics;
do not run the epoch30 completion/review hook on incomplete coverage.
Confirmed cancelled: worker stopped, GPU reservation released. Collected paused native bundle and inputs locally. Verified final checkpoint epoch14, scheduler14 and9,856 Adam steps; completed continuation metrics cover11–14. Epoch14 validation accuracy79.44%. Interrupted epoch15 is not a completed epoch. No epoch30 result or completed continuation claim. Cancellation evidence: `results/cifar10-ours-batch64-e10-e30-20260929-v1/analysis/cancellation.json`.
