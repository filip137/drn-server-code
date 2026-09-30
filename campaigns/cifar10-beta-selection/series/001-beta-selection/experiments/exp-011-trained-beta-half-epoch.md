---
id: "exp-011"
title: "Epoch10 ours beta sweep with matched half-epoch BPTT continuations"
status: "complete"
hypotheses: ["H-009"]
---
# Trained-checkpoint beta and short learning

Filip requested trained-epoch10 beta candidates, a beta sweep, half an epoch of
training and maximum available parallelism. Continue ours (the current scope)
from its original BPTT epoch10 checkpoint; interpret1/2 as half an epoch.

Seven independent branches restore the same full model, Adam moments/steps,
BN buffers, RNG and original50-epoch LR schedule. Use epoch11's deterministic
permutation/augmentation, first22,500 training examples:704 updates at batch32,
final batch4. Keep scheduler at completed epoch10 and save explicit partial-epoch
metadata; these outputs are not native completed-epoch resume checkpoints.
Preserve native T/K[6,6,4], trainable BN/gains/readout, original LRs, no read noise.

Exp-008's best sampled ours epoch10 RMS[0.02,0.035,0.001] gives beta vector
[0.09452322002142666,3.729986372379273,1.1695196708152935]. Freeze this vector
times[0.1,0.3,1,3,10] throughout each branch. Two controls: native BPTT and
free-T autograd using the existing hybrid's forward/digital-gradient semantics.
This control avoids attributing all differences from native BPTT to beta.
There is no recalibration during the half epoch.

Evaluate full5,000 validation images before/after, fixed256 training cohort,
and initial/final per-layer native/local BPTT cosine, norms and RMS for EqProp
arms. Compare Conv parameter updates against the common parent's native BPTT
branch. Rank absolute validation-CE difference, report accuracy difference and
gradient alignment separately. No official test data, smokes or retired audits.

## Execution and monitoring handoff

Root `results/cifar-beta-half-epoch-e10-20260929-v1/`; inputs include frozen cases,
parent checkpoint, split/cohort indices and hashes. Configs:
`configs/cifar/beta_half_epoch_e10_20260929/cases.json`. Native parent SHA256
`5f9a9c8bfe247fc029007ff8b06b3ba5ca9256aaac51aee16642d187f1944e2d`.
Each case has its own `cells/CASE` bundle and shared-queue job
`cifar-beta-half-e10-CASE-20260929` (underscores in CASE replaced with hyphens).

Use local and nom-cool-1 RTX3090 placements, one worker/GPU,16GiB
measured/bounded peak plus2GiB headroom. All5090s currently have unapproved other
workloads. Inputs were staged there but no jobs submitted for them; restrict the
seven branches to the matching3090 software environment for this comparison.
Future5090 placement would also require a BPTT control in its different runtime.
Budget1GPU-hour/case,
7GPU-hours total including diagnostics/attempts; expected20–30min/case on3090,
10–15min on5090, deadline September30 08:00 Paris. No scientific changes to fit
other GPUs. Reuse the verified private cuDNN9.1 on both3090s.

Implementation checks:14 focused CPU tests passed, covering identical epoch11
order, exact half-epoch length, restored next Adam update, partial checkpoint
bookkeeping, free-T control and existing hybrid gradient/RNG/BN contracts.

Queue owns admission/monitoring/collection. Preserve failed/partial attempts;
no blind retry of occupied paths. Validators require704updates/22,500examples,
unchanged parent, matching order/LRs, scheduler10, Adam14774 and full-state
partial checkpoint. CPU collection compares only completed validated cases and
does not claim a finished sweep until all seven are reconciled.

All seven jobs accepted. First placements: native BPTT on nom-cool-1 and1x
beta on local; five pending cases fill the next available slot automatically.
Verified live workers and fresh service heartbeat. Queue entrypoint
`experiments.run_cifar_beta_half_epoch` supplies bounded run/validate/collect;
CPU collection writes per-case receipts and `analysis/comparison.json`/JPG.
Only validated completed cases enter the report; runtime logs remain in bundles.
Startup verified: both first cases reproduced75.36% validation accuracy and
CE0.691126888 before training, then reached optimizer updates. Source/input and
full-state restore checks passed; no training smoke was performed.

All seven branches are complete, collected and validated. Each covers22,500
examples/704 updates with scheduler10 and Adam14774; all GPU workers released.
[Reviewed results](../results/exp-011-trained-beta-half-epoch.md) distinguish
closeness from performance. Fresh training follows separately in exp-012.
