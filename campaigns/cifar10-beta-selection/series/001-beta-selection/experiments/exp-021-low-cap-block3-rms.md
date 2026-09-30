---
id: "exp-021"
title: "Fresh current-batch training with the lower ceiling and larger block3 target"
status: "complete"
hypotheses: ["H-016"]
---
# Complete the missing beta-selection cell

September30,09:24 Paris: completed and locally validated at1,407 updates,
28.92% validation accuracy, CE2.149137,3,233.2seconds. Both matched-BPTT
gates fail. [Review](../results/exp-021-low-cap-block3-rms.md) records the
completed four-cell comparison and substantial block3 ceiling misses.
No extension; Filip's09:00 stop on new runs remains in effect.

One fresh ours epoch uses targets[.02,.035,.005] and beta bounds[1e-8,10000].
Relative to [exp018](exp-018-block3-rms-training.md), change only the upper
beta bound30,000→10,000. Relative to exp015, only block3 targets in
`rms_targets` and `beta_search.targets` change.001→.005.

Reuse exp018's404 frozen source files byte-for-byte, original model snapshot,
fresh Adam, seed0,45,000/5,000 split, order and augmentation, T/K[6,6,4],
batch32/eval16, trainable BN/gains/head, rates and50-epoch schedule. The
starting beta vector, five-pair search cap,25% tolerance, ratio bounds and
all other controller fields remain unchanged. There is no smoke, additional
diagnostic, noise intervention, native-control rerun or official test read.

The sole case is `current_batch`; `comparison_only:true` bounds execution
to the full first epoch:1,407 updates and45,000 training examples. Existing
initial/final validation and diagnostic behavior are unchanged. Compare the
5,000-image final validation with the matched native37.06%, CE1.726060.
The unchanged gate is accuracy>=35.06% and CE<=1.812363172. Report RMS
misses, beta values and pair counts with the result. Main reviews a passing
candidate before any stage5/10 continuation; this job cannot auto-extend.

## Execution and monitoring handoff

Root: `results/cifar-beta-current-batch-low-cap-block3-rms-20260930-v1/`.
Small configs: `configs/cifar/beta_current_batch_low_cap_block3_rms_20260930/`.
Prepared job: `launch/current_batch-1.json` under the result root.
Use the existing shared queue with local or nom-cool-1 RTX3090 placements,
torch2.5.1+cu121/privatecuDNN9.1. Reserve16GiB plus2GiB headroom; preserve
Ben's work and the existing owner guard. The baseline stage5 jobs retain
their allocations and higher priority. Do not submit for admission until
the matched baseline native job has actually started; then wait for the
next free3090 without preemption or reordering baseline jobs.

One5,400second allowance, expected about3,200seconds. Deadline October2
08:00 Paris. Transfer5,400seconds by reducing all nine unsubmitted ours
epoch50 caps46,400→44,600seconds, over at most three promoted branches.
The combined141.5GPU-hour ceiling remains unchanged. Record exact before/
after budget hashes in `analysis/budget_transfer.json`.

Reuse passed tests of the unchanged source. Static source/config/input and
prepared-file checks establish identity; do not add a scientific preflight.
The existing queue wrapper and collector validate full coverage, beta-state
continuity, optimizer/scheduler/BN state and all5,000 validation examples.
Queue is the sole runtime monitor; main owns scientific review and subsequent
promotion.

Accepted ID: `cifar-beta-fresh-current-batch-low-cap-block3-rms-e1-20260930`.
It was queued at priority1 behind the running priority8 baseline jobs.
Nom-cool native was confirmed at update1420
(PID461028); local baseline EqProp was already beyond update1500 (PID1605898)
before submission. Neither allocation was displaced. The shared queue service
was alive with a25-second observation age and reports waiting for eligible
GPU memory/ownership. The existing30-minute queue summary includes this ID by
prefix; no additional monitor or service change was introduced.

September30 startup: the queue admitted this job on nom-cool-1 after the
baseline native5 process completed. Attempt `0862aa7cda75436ca68caeade82ac8e0`,
launcher PID481505; initial validation and beta-observation artifacts are
advancing. No baseline job was interrupted. The queue remains the sole monitor.

All404 source files and `source_identity.json` are byte-identical to exp018;
source-identity SHA256 is
`46239ed3aa94bc386f1e36270b5b7deec8c3189d1885b55e4ab5190f5c0bec11`.
Six input hashes and13 prepared-file hashes verified on each allowed target.
`analysis/preparation.json`, `analysis/preflight.json` and
`analysis/budget_transfer.json` retain compact identity, placement and funding
evidence. All nine donor epoch50 caps are now44,600seconds. Reuse the existing
wrapper's `validate 1 current_batch` and `collect 1 current_batch`; collection
records execution completion only, leaving the scientific gate to main.
