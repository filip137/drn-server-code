---
id: "exp-017"
title: "Two fresh ten-epoch readout-LR runs with trainable BN affine"
status: "cancelled"
hypotheses: ["H-011"]
---
# Fresh readout-rate candidates

Filip's final instruction: **two runs only, from scratch, ten epochs, train BN**.
No extra controls or continuations. No smokes. Test
[H-011](../../../hypotheses/H-011-fresh-readout-rates.md), motivated by exp-016.

| Scheme | Readout base LR | Change from original |
|---|---:|---:|
| Original unnormalized legacy | 0.00005333333333333333 | divide by3 |
| Ours | 0.0000019758828827095137 | multiply by0.037042420152718095 |

Use each original LR-search epoch0 initialization, fresh Adam moments/step0 and
scheduler phase0. BN gamma/beta and running statistics train normally; BN LR0.001.
Preserve original convolution vectors, gain LR5e-5, T/K=[6,6,4], batch32/eval16,
seed0, 45k/5k split, crop/flip, loss, architecture, amplification and50-epoch
cosine horizon/floor0.02. No normalized-end-voltage intervention. End at epoch10;
rank terminal full-validation accuracy/CE, not best-epoch metrics. Compare with
historical trainable-BN original-rate scores, not the frozen-BN screen's cohorts.

Exact configs: `configs/cifar/fresh_readout_lr_20260928/{legacy,ours}.json`.
Root registered before creation:
`results/cifar10-fresh-readout-lr-bn-trainable-20260928-v1/`.
Frozen source, original preparation bundles, job commands and outputs live under
`source/`, `assets/`, `launch/`, and `cells/{legacy,ours}`. Source runner is native
`experiments.train_cifar_l8_analog`; queue transport is
`experiments.run_cifar_fresh_readout_lr`. No resume argument is passed. Retain
native source/initializer checks and nonfinite guards; no pre-launch science
audits. The runner's existing final audit remains part of completion.

## Execution and monitoring handoff

Two sequential jobs on Trex RTX5090, expected2–3 hours each, capped at4 GPU-hours
each (8 total); deadline September30 08:00 Paris. Reserve12,288MiB plus2,048MiB
headroom per worker, default profile. Keep the existing baseline continuation
on nom-cool-1 unchanged. Queue jobs:
`cifar-fresh-readout-legacy-20260928`, `cifar-fresh-readout-ours-20260928`.

Shared run-watch queue owns monitoring and bounded operational recovery. Inspect
with `gpu_queue.py status JOB_ID`. No blind retries or extra scientific cases.
Collection hook validates epochs1–10, full data coverage,14,070 Adam steps,
fresh initial optimizer, and trained BN affine/statistics. It collects each
case, then summarizes both against the historical references, writes a JPG and
the result/H-011 verdict, updates ideas.md and refreshes the ledger. Collection
allowance10 CPU-minutes per case. Failed or paused runs stay visibly incomplete;
native deadline protection preserves completed-epoch states. No automatic extension.
Current handoff: cancelled at Filip's request while he develops a new amplification.
Both queue jobs are cancelled, both workers confirmed stopped, and GPU reservations
released. No active or queued CIFAR jobs remain in the shared queue. Ours paused
with completed epoch2 saved; legacy had already failed after a Trex reboot during
epoch2 (last reported batch1,100), with completed epoch1 saved. Legacy's native
status still says running because the reboot bypassed clean shutdown; queue
receipt/boot identity establish that it is stopped. Preserve this discrepancy.
No retry, recovery or extension is authorized. Partial bundles are retained under
`cells/{legacy,ours}` and copied locally. There is no complete ten-epoch comparison;
H-011 remains untested at its declared endpoint. The new amplification is not yet
specified and no replacement experiment is launched.
