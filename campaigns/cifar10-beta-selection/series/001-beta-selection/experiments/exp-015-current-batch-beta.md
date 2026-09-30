---
id: "exp-015"
title: "Current-minibatch beta selection with bounded centered solves"
status: "complete"
hypotheses: ["H-012"]
---
# Remove the one-minibatch response delay

Completed and reviewed September30 04:35 Paris:32.64%, CE1.788320 misses
the accuracy gate. See the [result](../results/exp-015-current-batch-beta.md).
The bounded ceiling follow-up is recorded in exp-017; no long extension here.

One fresh ours arm follows exp-014 within the authorized beta search. Preserve
its exact initializer, fresh Adam, seed0,45k/5k data/order/augmentation, original
LRs,50-epoch schedule, trainable BN/gains/head, batch32/eval16 and T/K[6,6,4].
Targets remain [.02,.035,.001]; initial beta is the same0.1x trained vector.
No read noise. Reuse the validated fixed and lagged controls from exp-014 and
same-runtime native/free-T controls from exp-012; no redundant control rerun.

After the single differentiable free forward/backward, freeze each block's free
states and boundary force. Measure centered EqProp at the proposed beta, restoring
those same free states for every trial. Up to five centered pairs per block and
batch are allowed. Stop when RMS is within25% of target. Otherwise propose beta
times target/max(RMS,1e-12), with multiplier clipped to[.001,1000] and beta to
[1e-8,1e4]. When observed beta values bracket the target in increasing order,
use their geometric midpoint. Stop if bounds repeat a tested beta.

If the evaluation cap is reached, choose the tested beta with smallest absolute
log RMS/target error (RMS floor1e-12; ties prefer smaller beta). Use the cached
gradient computed at that exact beta; never change only its denominator. Record
whether the selected trial met tolerance. This is bounded selection, not an
assertion that every target was reached. All trials retain nonfinite guards.
Reuse the chosen beta as the next batch's starting proposal only after a finite
optimizer/project update. No additional model/BN forward or random draw occurs.

Checkpoint controller state with model/Adam/BN/RNG/scheduler. Log starting,
selected and next beta, all trial RMS/beta pairs, target error, solver-pair count
and tolerance misses. Retain initial/final gradient diagnostics. CPU checks must
verify gradient/selected-beta consistency, state restoration between trials,
unchanged digital gradients/BN/RNG and exact continuation.

Train one full epoch:45,000 examples,1,407 updates and5,000 validation examples.
The unchanged long-run eligibility gate is accuracy>=35.06% and CE<=1.812363,
relative to matched native37.06%, CE1.726060. Compare also against fixed34.22%,
CE1.845956 and lagged34.34%, CE1.981908. Better RMS alone is insufficient.
If eligible, continue under the existing staged policy and remaining allowance;
otherwise review the measured failure without silently relaxing the gate.

## Execution and monitoring handoff

Root: `results/cifar-beta-current-batch-20260930-v1/`. Local3090, matching
torch2.5.1+cu121/private cuDNN9.1. One job, maximum10,800seconds; expect1–3hours
depending on trial count. Deduct this3GPU-hour ceiling from the unused ours
allocation by reducing each prepared epoch50 allowance15h→14h across at most
three branches; the combined141.5GPU-hour ceiling is unchanged. Deadline
October2 08:00 Paris. The queue owns launch, collection and monitoring, with
this root included in the existing half-hour summary. No GPU smoke.

Submitted September30 03:44 Paris as `cifar-beta-fresh-current-batch-e1-20260930`.
The shared queue admitted it locally; CUDA initialization succeeded and the
initial validation is in progress. Source identity covers404 files and six
inputs. Sixty-seven CPU tests passed, including exact selected-beta gradients,
state restoration across trials, unchanged BN/digital gradients/RNG, bounded
solver calls, controller commits and checkpoint continuation. The updated
collector verifies trial selection and every-update controller continuity.
The existing half-hour monitor includes this root; next scheduled check04:00 Paris.

Actual training startup verified locally at32 updates, PID1543688. The first two
batches reached all three target tolerances using[2,3,3] and[3,3,3] centered
pairs; this verifies execution, not improved accuracy. Boundary cosine artifacts
evaluate the stored warm-start beta on the fixed diagnostic cohort; they do not
measure every training batch's selected-gradient cosine.
