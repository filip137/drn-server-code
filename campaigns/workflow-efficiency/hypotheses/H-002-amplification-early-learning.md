---
id: "H-002"
title: "Amplification improves early relative validation-loss reduction"
---
# H-002 — Early learning

Accepted research question: does bidirectional amplification improve early learning under matched initialization, data and optimizer settings? Prospective test compares baseline v1/c1 and ours v4/c1 on a tiny seed-0 perfect-diode Conv1 with identical explicit Adam rates.

For each arm r=(validation_loss_epoch1-validation_loss_epoch4)/validation_loss_epoch1. Primary contrast D=100*(r_ours-r_baseline), in percentage points of relative loss reduction. Support if D>2pp; contradict if D<−2pp; otherwise inconclusive. Exactly ±2pp is inconclusive. The threshold is an arbitrary predeclared pilot-scale practical threshold. Report all four epoch losses and initial/final levels so normalization cannot hide differences. Epoch1 is the step-8 reference, not initialization. No test-split access or best-checkpoint selection.

Validity requires both arms complete four epochs/eight minibatches each, finite metrics, identical initial-parameter/split/all-epoch-order hashes, and equal scientific config except amplification. Nonfinite failure or failed matching gives inconclusive scientific evidence with preserved failure. No tuning or retry of a scientifically failed arm. One seed and sixteen validation examples permit only this conditional pilot conclusion.
