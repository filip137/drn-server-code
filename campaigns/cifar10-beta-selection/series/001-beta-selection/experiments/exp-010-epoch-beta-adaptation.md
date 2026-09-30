---
id: "exp-010"
title: "Ours ten-epoch learning: recalibrate beta every epoch versus fixed beta"
status: "cancelled"
hypotheses: ["H-008"]
---
# Epoch-wise beta adaptation

September29: Filip requested early termination of all four runs after the
epoch5 comparison showed a substantial deficit versus historical BPTT.
Cancellation confirmed through the owning queue: all workers stopped and all
four reservations released. Bundles collected locally; final checkpoints load
with epoch2 on both3090s and epoch5 on both5090s, including model, optimizer,
scheduler, RNG and beta state. nom-cool-1 reports a termination-time DataLoader
exception, with its epoch2 checkpoint intact. Partial evidence and interpretation
are in [the result note](../results/exp-010-epoch-beta-adaptation.md).
No retries or replacement runs are authorized.

Filip approved two fresh10-epoch ours arms, RMS0.09 in all blocks, convolution-only
EqProp and autograd boundary signals/BN/gains/dense-readout updates. His latest
placement clarification adds local GPU and nom-cool-1 runs IN ADDITION to the
two5090 runs. There are two policies repeated on two hardware pairs, four runs
total, all seed0; these are not independent scientific seed replicates.
The earlier six-arm training comparison in exp-002 is deferred.

**Question:** does recalibrating beta before each epoch improve learning relative
to keeping the initialization coefficients? The competing noise/bias tradeoff and
changing effective response remain the overarching beta-selection question; no
read noise is injected in this first controlled comparison.

Both arms share the original epoch0 seed0 tensors, deterministic 45k/5k training/
validation split, batch32, validation batch16, augmentation/order, native endpoints,
ours V4/C1, blocks[3,3,2], free/nudged T/K[6,6,4], trainable BN/gains and the original
ours_edge_head_down_e10 Adam rates and50-epoch cosine schedule. Stop at epoch10;
no LR tuning, official test access, training smokes or retired solver audits.

Training differentiates one free T forward pass. Summed-CE autograd supplies
the physical output forces; divide parameter gradients by actual batch size.
Replace the eight Conv weight gradients with stable centered EqProp estimates,
using both phases from identical saved free states; retain autograd updates for
all other trainable tensors. BN running statistics update once per training
batch. This explicitly differs from native BPTT's detached-free plus tracked K
training and from full end-to-end EqProp. Both new arms use exactly this method.

Reuse verified initialization beta vector
`[0.005572005081287171,0.0010216520845670086,0.02058830659364593]` in both arms.
Fixed retains it for all10 epochs. Adaptive holds it during epoch1 and searches
before epochs2–10 using current parameters, same256 training examples/order,
batch32/no augmentation, relative RMS tolerance3%, max24 evaluations per block,
warm-starting previous beta. Searches preserve parameters, BN buffers, RNG and
optimizer state. Commit beta only if all blocks succeed; pause with the previous
complete checkpoint on failure. Checkpoints include beta vector/policy/epoch.

Measure beta/RMS at initialization and after every epoch; pooled/minibatch native
and local BPTT cosine at epochs0/1/5/10. Diagnostics use each arm's actual beta.
Separate calibration, observation and training/validation times. Compare final
epoch10 accuracy/CE using H-008's decision rule; no best-epoch selection.

## Execution and monitoring handoff

Active root `results/cifar-beta-adaptation-20260929-v2/`, cells `epoch` and `fixed`;
configs `configs/cifar/beta_adaptation_20260929/`. Adaptive uses local RTX3090;
fixed uses nom-cool-1 RTX3090, with authorized Ben sharing and verified memory.
Reserve16GiB plus2GiB headroom per worker. Queue IDs
`cifar-beta-epoch-local-20260929-v2` and `cifar-beta-fixed-nomcool1-20260929-v2`.
Budget8GPU-hours/arm including preparation/diagnostics/attempts; total16GPU-hours.
First full epoch determines ETA; pause at an epoch boundary if the next epoch
cannot fit the original budget. Initial estimate40min/epoch, subject to measured
3090/shared-host performance. Shared run-watch owns admission, monitoring and
collection; no competing watch. Prepared source and input identities are frozen.

Entrypoint `experiments.run_cifar_beta_adaptation` supports run/validate/collect.
Validate10 epochs,14070 updates, all45k/5k examples,11 beta/RMS boundary records,
four cosine records, nine adaptive calibrations and zero fixed calibrations,
fresh Adam/BN state, full-state checkpoint and no solver audit/test read.
Collectors write local validation receipts and regenerate JPG comparisons when
both arms are complete. Scientific review and campaign verdict remain separate
from scheduling completion. Source/checkpoint failures, nonfinite values,
calibration failure or budget risk stop/escalate with preserved evidence.

Implementation checks:35 focused tests passed,2 existing CUDA-only tests skipped
in the CPU-only suite. Verified same configs except policy/arm ID and exact frozen
source identity. Asset preparation restored the shared initializer on CPU without
training or solver qualification.

Preserved failed root `results/cifar-beta-adaptation-20260929-v1/`: both hosts
failed before any optimizer update with CUDNN_STATUS_NOT_INITIALIZED. Their
PyTorch2.5.1+cu121 declares cuDNN9.1.0.70, but loaded9.19 libraries despite9.1
package metadata. V2 privately pins the declared9.1 libraries with LD_PRELOAD/
LD_LIBRARY_PATH; shared environments are untouched. Local metadata inspection
confirms runtime90100. Original attempts are collected and queue jobs cancelled;
their3.057s/3.240s charges are deducted from the respective V2 eight-hour budgets.
Scientific code/settings are unchanged between attempts. Exact dependency hashes
are in `runtime/identity.json` and queue prepared-file checks.

### Additional RTX5090 pair

Filip clarified the placements are additive. Keep both3090 jobs running and add
`results/cifar-beta-adaptation-5090-20260929-v1/`: adaptive `epoch` on Loulou,
queue ID `cifar-beta-epoch-loulou-5090-20260929`; fixed on Riri, queue ID
`cifar-beta-fixed-riri-5090-20260929`. Same initializer, seed0, data order, beta
values and scientific config; only study ID and placement differ. Configs are in
`configs/cifar/beta_adaptation_5090_20260929/`. Each new job receives8GPU-hours,
so total authorized allowance is32GPU-hours across four runs, including the
original failed charges. Respect the two-distinct5090 daytime cap and Ben sharing.

The5090 hosts use their existing working PyTorch2.11/cu128/cuDNN9.19 environment;
do not preload the3090-specific cuDNN9.1 dependency. Within each pair the runtime
versions match. Analyze policies within each hardware pair and report runtime
differences explicitly; do not pool repeated seed0 runs as independent seeds.
The validated scientific implementation is reused; only the launch wrapper gains
an optional config-directory environment variable. No new smokes or audits.

Launch verified: all four queue workers are alive and have reached epoch-1
optimizer updates. The queue service has a fresh heartbeat and owns monitoring
and collection for both pairs; both original3090 jobs remain running.

### Fixed-arm queue incident 1 (September 29)

**Resolved:** the incident below describes the preserved V1 attempt. V2's private
runtime repair and replacement jobs supersede its blocked decision; the local
arm has reached optimizer updates and nom-cool-1 has progressed through its
initialization diagnostic. Do not retry or reopen the cancelled V1 job.

Attempt `a887c166483547feb1620784e83c7a46` on nom-cool-1 failed with
exit1 after3.240 seconds, reporting `CUDNN_STATUS_NOT_INITIALIZED` in BatchNorm.
A live SSH read confirmed the failed receipt and no matching worker, supervisor,
process-group/session or training command in the process listing. The remote
`cells/fixed` bundle exists with manifest/status/checkpoints and empty metrics;
preserve it and the queue attempt log/receipt. No successful compute or collection
is established. The cuDNN root cause remains unresolved; subsequent SSH reads
failed to resolve nom-cool-1.

Incident decision: blocked. The frozen `run fixed` entrypoint asserts that its
output directory does not exist, the job declares `retry_safe: false`, and no
`resume_argv` is prepared. Thus the existing command cannot recover while retaining
the failed bundle at its current path. Recovery needs restored host access and a
reviewed evidence-preserving retry/resume handoff through the scheduler. No worker
was launched, cancelled or modified, and no queue state or scientific setting was
changed. Retain the original8GPU-hour allowance, including the3.240-second attempt;
this incident does not grant a fresh budget.
