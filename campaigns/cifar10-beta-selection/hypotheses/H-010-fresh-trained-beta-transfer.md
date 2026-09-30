---
id: "H-010"
title: "Trained-checkpoint beta candidates can support learning from scratch"
---
# Can trained-checkpoint betas work from initialization?

Exp-011's trained continuations do not test the difficult initial learning phase.
Compare fresh ours BPTT against fixed0.3x and1x versions of its epoch10 beta vector.
Practical screening rule, chosen before these runs: at the same training boundary,
EqProp accuracy is at least BPTT minus2pp and validation CE at most1.05x BPTT.
Require finite execution. This is a resource-allocation rule, not significance.

Passing epoch1 candidates continue to epoch5; passing epoch5 candidates to10.
If neither succeeds at an epoch1/5 gate, broaden the fixed-beta search before committing
more long training. Preserve matching initial tensors, optimizer history,
data/order/augmentation, BN, rates and T/K. No read noise.
[exp-012](../series/001-beta-selection/experiments/exp-012-fresh-beta-staged.md)
owns candidates, execution budgets and conditional decisions.

[exp-013](../series/001-beta-selection/experiments/exp-013-all-scheme-long-eqprop.md)
extends the test to baseline, ours and legacy, each with its own epoch10 beta
candidates and runtime-matched native control. Passing runs can continue to50
epochs through the same accuracy/CE gates. This tests transfer within each scheme;
it does not assume one scheme's beta or learning rate transfers to another.

September30: exp013 records one posthoc allocation exception for ours0.3x,
which missed epoch1 accuracy by one validation image while passing CE.
Its bounded epoch5 follow-up retains the failed epoch1 classification and
unchanged epoch5 gates; report this changed selection path explicitly.
