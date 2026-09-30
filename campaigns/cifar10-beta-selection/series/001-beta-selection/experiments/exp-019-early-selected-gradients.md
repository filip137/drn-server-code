---
id: "exp-019"
title: "Early actual-gradient replay of the failed beta-ceiling trajectories"
status: "complete"
hypotheses: ["H-015"]
---
# Measure the gradients that reached Adam

Replay the first101 fresh updates of exp-015 (ceiling10,000) and exp-017
(ceiling30,000), using their exact frozen inputs, initializer, fresh optimizer,
data order/augmentation, scientific source and matched local runtime. Preserve
all scientific settings. This is a post-failure diagnostic, not a prelaunch
qualification or another full-epoch training candidate. Exp-018 subsequently
completed at32.88% validation accuracy and CE1.945375, failing both gates.

At0/1/20/50/100 completed updates, probe the next actual augmented training
minibatch. Before the real update, obtain mean-CE native BPTT and reset-free-T
reference gradients with `autograd.grad`, preserving model modes, buffers,
BN running state and all RNG state. Use the established replay context together
with minibatch BN statistics; do not replace the batch with a fixed cohort.
Then run unchanged `training_update` and capture the actual gradients at Adam's
input, after beta selection. Measure actual parameter deltas after Adam and
projection. References must not alter the replay trajectory or optimizer.

Record per-layer and pooled-block cosine, both gradient norms and their ratio,
zero fractions, selected beta/RMS/trials, and separate Conv versus BN/gain/head
gradient agreement and update norms. Zero reference norm means undefined cosine.
The references are finite-iteration computations, not equilibrium ground truth.
Ten probe batches across two101-update trajectories are the declared coverage.
No full validation, solver sweep, beta tuning or noise test is included.

CPU checks must establish probed/unprobed trajectory identity (parameters,
Adam, controller, BN, order and RNG), exact capture of Adam's selected gradient,
chosen-beta consistency when the chosen trial was not last, and reference loss
scaling. On the real replay, compare every available prefix loss with the
original per-update metrics; a mismatch blocks interpretation as an exact
trajectory replay. Report incomplete coverage explicitly on deadline/failure.

## Execution and monitoring handoff

Root: `results/cifar-beta-early-selected-gradients-20260930-v1/`.
Use the free local3090 with torch2.5.1+cu121/privatecuDNN9.1. One GPU job,
maximum1,200seconds; expect about8–12minutes for both trajectories. Fund20
GPU-minutes from unused ours epoch50 allowances by reducing each13h cap by
400seconds across at most three promoted branches (46,800→46,400seconds).
The combined141.5GPU-hour ceiling remains unchanged. Deadline October2
08:00 Paris. Use the shared queue as sole owner; retain source/input identities,
progress and declared-coverage validation in the existing bundle format.
No smoke or retired solver audit. Record accepted ID and actual progress here.

Prepared transport: `results/cifar-beta-early-selected-gradients-20260930-v1/launch/replay.json`.
The frozen source preserves all404 exp017 files byte-for-byte and adds only
`experiments/replay_cifar_selected_gradients.py`;10 new and46 related CPU tests
passed. The helper runs for at most1,110seconds inside the1,200second queue
allowance. Its output bundle is `cells/replay/` under the root above.
The nine unsubmitted ours epoch50 specifications now have46,400second caps;
the maximum three promoted branches release1,200seconds. The exact before/after
specification hashes are in `analysis/budget_transfer.json`.

Queue ID: `cifar-beta-fresh-early-selected-gradients-20260930` (complete and validated).
Attempt `8c6e6aecd00b4eb9863c37caeb8d63bd` started September30 at04:01:30UTC
on local RTX3090 `GPU-7cb044b0-02d2-61d7-a982-e2a4f1a2d4a0`, child PID1601193.
At04:02:02UTC the first trajectory had completed6/101 updates with exact
original-prefix losses. Queue service heartbeat was fresh and ownership verified.
The shared queue is the sole runtime monitor; the existing30-minute summary
includes this ID by prefix. The queue's target validator and CPU completion
command both use `python -m experiments.replay_cifar_selected_gradients
--validate-run RUN`, where `RUN` is the absolute `cells/replay` path.
Completion requires202 exact original-prefix loss matches,10 probes and both
101-update full-state checkpoints. A mismatch retains failed/partial evidence
and does not authorize an automatic retry.

Both101-update trajectories completed, with all202 original losses exactly
matched and all10 probes present. Scientific runtime473.24seconds; queue
charged478.324seconds, below the1,200second allowance. Peak allocation10.37GiB
was below the16GiB bound. Target validation and CPU collection both exited0;
collection `04dd0059bb5c46199c373a4428637df4` confirmed declared coverage,
beta/counter continuity and checkpoint state. The local reservation is released.
Compact receipt: `analysis/collection.json`; scientific review:
[exp-019 result](../results/exp-019-early-selected-gradients.md).
