---
id: "exp-014"
title: "Fresh EqProp with beta feedback from each training minibatch"
status: "complete"
hypotheses: ["H-011"]
---
# Beta feedback during early learning

Completed September30: lagged feedback34.34%, CE1.981908; fixed34.22%,
CE1.845956. Neither passes the matched-native gate. See the
[reviewed comparison](../results/exp-014-minibatch-beta-feedback.md).

Follow up exp-012's exhausted fixed search within Filip's authorized beta search
and long-training objective. Two fresh ours arms, one full epoch each: fixed
and lagged feedback. Both start at0.1x the exp-012 base vector, the best completed
fixed candidate. Preserve the same initializer, fresh Adam, seed0,45k/5k split,
augmentation/order, original LRs,50-epoch schedule, trainable BN/gains/head,
batch32/eval16 and T/K[6,6,4]. No read noise or extra solver qualifications.

Both arms measure relative RMS from the same centered endpoints used for the
Conv gradient. Feedback changes only the next minibatch's beta, after a finite
optimizer/project update. Targets by block: [.02,.035,.001], preserving the
epoch10 anchors tested by exp-012. These remain candidates, not proven optima.
For each block:

`beta_next = clip(beta * clip(sqrt(target / max(RMS,1e-12)), .5, 2), 1e-8, 1e4)`.

The fixed arm records the same measurements without changing beta. No extra
nudged solve, forward pass or RNG draw is introduced. Save controller values,
successful-update counters and clipping counts with Adam/BN/RNG/scheduler.
Log beta used/next, actual RMS and clipping each update; retain initial/final
gradient diagnostics. A maintained RMS alone does not demonstrate good gradients.

Primary outcome: full5,000-example validation accuracy/CE after1,407 updates.
Compare with the instrumented fixed arm and existing same-runtime native/free-T
controls (37.06%/37.84%). Eligibility for longer training remains accuracy at
least35.06% and CE at most1.812363 (native+2pp tolerance and1.05x CE).
Also require finite execution and verified instrumentation equivalence. Improvement
over fixed without reaching this gate is partial evidence, not long-run success.
No automatic grid expansion or relaxed gate. If eligible, prepare exact-state
extensions under the existing5/10/20/30/50 policy and remaining campaign allowance.

## Execution and monitoring handoff

September30 03:00 Paris: the fixed arm is collected and validated at34.22%,
CE1.845956, exactly reproducing exp-012's fixed0.1x result. Feedback retry1 is
confirmed training locally (PID1520847), beyond440 updates with live RMS/beta
telemetry; the earlier CUDA initialization failure is resolved for this attempt.
The shared queue remains the monitoring owner. Next scheduled summary03:30 Paris.

Root: `results/cifar-beta-minibatch-feedback-20260930-v1/`.
Two local3090 jobs, sequential if no matching3090 capacity is available; expected
45–60minutes each, maximum5,400seconds each (3GPU-hours total). Charge this bounded
follow-up to the unused ours long-training allocation: reduce each prepared ours
epoch50 allowance from16h to15h across at most three promoted branches. The
combined141.5GPU-hour ceiling remains unchanged. Do not duplicate pending
ours5090 screens. Deadline October2 08:00 Paris. Same torch2.5.1+cu121 and private
cuDNN9.1 as exp-012. The shared queue owns monitoring, collection and available
placement; the half-hour summary includes this root after submission.
Job IDs, frozen identities, implementation checks and launch evidence follow here.

Submitted September30 01:58 Paris: `cifar-beta-fresh-minibatch-e1-feedback-20260930`
and `cifar-beta-fresh-minibatch-e1-fixed-20260930`. The queue accepted both and
started feedback locally. Frozen source identity covers404 files; six inputs
preserve the original initializer/config/split/cohort and declare the two cases.
Fifty-six CPU tests passed, including bitwise gradient/BN/RNG equivalence,
unchanged solver-call count, post-update commit and exact resumed next updates.
Collector validation includes every-update beta continuity, counters and clipping.
The durable half-hour summary now includes this root. No GPU smoke was run.

The first feedback attempt failed during CUDA driver initialization before any
training update; this is operational evidence, not a beta rejection. Recovery
retains its spent allowance and failed bundle; exact replacement handles follow.

### Startup recovery, September30

Feedback attempt`2d71f00c834d4a65a712fe942d614086` exited during CUDA driver
initialization after3.2613337seconds, before any metrics or checkpoint. Queue
evidence confirmed both worker processes absent and the reservation released;
no automatic recovery agent was active. The failed bundle and queue receipt are
preserved under the original root's
`failed_attempts/feedback/2d71f00c834d4a65a712fe942d614086/`.

A bounded availability check under the exact launch environment subsequently
reported CUDA available, torch2.5.1+cu121, CUDA12.1 and cuDNN90100. It performed
no GPU computation. No persistent cause was identified; no driver/GPU/MPS
settings, scientific source or inputs were changed. Diagnostic evidence is
`analysis/cuda-startup-diagnostic.json` in the original root.

Replacement root: `results/cifar-beta-minibatch-feedback-retry-20260930-v1/`;
job`cifar-beta-fresh-minibatch-e1-feedback-20260930-retry1`. Source and inputs
are byte-identical to the canonical study. Its allowance is5,396seconds after
rounding the original3.2613337seconds upward to4seconds; the deadline and total
budget remain unchanged. The frozen retry collector validates the new bundle,
atomically restores its canonical cell, retains both receipts and retry
provenance, and runs the original comparison-only completion policy. It cannot
promote or launch a longer scientific run.

The healthy fixed arm remains untouched: attempt
`67e93eb570474e21a4971ee6bfd2bde9`, worker1503874, with initial checkpoint
and validation in progress. The feedback retry uses the same local3090 queue
and waits for that worker to release its reservation. The failed feedback job
is held while replacement preparation is verified; final cancellation and
replacement acceptance are recorded below. Shared run-watch remains the sole
launch and recovery owner.

The original feedback job is now cancelled after verified terminal failure; its
replacement was accepted and is queued behind fixed. This confirms queue
ownership, not retry startup or successful computation. The existing minute
queue checks and half-hour summary cover the retry ID; the canonical comparison
remains pending until both collected receipts validate. Latest handoff state is
`results/cifar-beta-minibatch-feedback-retry-20260930-v1/analysis/queue-handoff.json`.
