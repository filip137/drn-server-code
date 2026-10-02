# X-002 — Check the ranking before explaining it

The current [review](../../../docs/cifar_experiment_review_20260924.md) shows legacy
above baseline at epoch30 but below it at epoch50, on one seed. This is an endpoint
and trajectory observation, not yet a stable property of an amplification scheme.
A fresh paired-seed comparison addresses the premise directly. It should hold the
scientific surface and environment fixed and report all seed-level outcomes.

The present networks also differ structurally from earlier MNIST studies: separately
voltage-clamped analog blocks, digital normalization/pooling, and BPTT across boundaries.
An expected advantage in a fully coupled EqProp network is a motivation, not a prior
CIFAR control. Isolating the topology itself would require its own matched architecture
study, rather than attaching a causal label to unmatched datasets and training methods.

The LR search selected separate conv/head vectors using short horizons. Nearby legacy
half/double changes did not improve the ten-epoch setting, but this does not establish
joint LR/epsilon or 50-epoch optimality. Importantly, original trajectories use different
optimizer vectors; the optimizer transplant in H-004 is a concrete test, whereas a broad
retuning exercise would ask a different, much more expensive optimization question.

Existing native solver/checkpoint audits and synthetic KCL checks weaken a simple
implementation-error explanation on tested cases. Larger legacy raw gradients do not
imply larger Adam-relative updates, and sampled nominal directions improve same-batch
loss. That does not prove generalization is optimal or rule out trajectory effects.

Input gains barely move under their original parameterization. Stronger log-gain
learning improves a matched fixed-gain frozen-BN control but not the historical softplus
reference. Neither weak gain adaptation nor the raw gradient scale has been shown to
be the unique cause of the final gap. Preserve these limits rather than relaunching the
same negative tests without a new discriminating prediction.

See [H-003](../hypotheses/H-003-reproducible-gap.md) and
[H-004](../hypotheses/H-004-optimizer-transplant.md). Stop with a bounded unresolved
explanation set when the data do not support a unique mechanism.

September27: [H-009](../hypotheses/H-009-classifier-update-imbalance.md) narrows the
optimizer account to classifier-versus-feature update balance. Its saved-checkpoint
intervention changes only the head.weight learning rate and uses both minibatch
and saved-stat BN evaluation; it is distinct from the completed whole-update probes.

## September28 synthesis and next tests

The gap remains a selected seed0 CIFAR L8 result, not an established property of
legacy. [Trajectory review](../series/002-residual-gap/results/exp-013-training-trajectories.md)
finds worse legacy fitting before the persistent validation deficit appears after
epoch30. Existing low-LR baseline/legacy continuations narrow the late mean gap by
only0.12pp; ours' floor job is incomplete, so a three-arm horizon verdict is not
available. The historical mapped baseline/normalized-legacy identity with a common
optimizer constrains an intrinsic-amplification account of their difference.
Scheme-selected LR vectors, effective BN epsilon and training histories remain
interacting factors. Normalizing legacy endpoints alone added only0.10pp at50;
the tested BN-stat recalibration did not repair accuracy.

Recent local tests do not identify the cause: [head probe](../series/002-residual-gap/results/exp-009-classifier-update-probe.md)
is inconclusive across checkpoints, and the convolution response changed with the
evaluation cohort ([exp-012](../series/002-residual-gap/results/exp-012-conv-lr-cohorts.md)).
The actual [one-epoch 3× convolution LR screen](../series/002-residual-gap/results/exp-014-one-epoch-conv-lr.md)
provides a clearer negative: worse online and fixed-cohort fitting, and-4.20pp
validation accuracy versus its matched control. Reject this abrupt increase from
that checkpoint; do not infer that the original LR is optimal or too large.

The reviewed frozen-affine-BN runs keep normalization/statistics active. Freezing
from initialization worsens all three schemes, but baseline much more than legacy;
the resulting legacy/baseline rank reversal does not show that freezing helps legacy.
Legacy/ours have similar epoch10 penalties, and only ten epochs were measured.
Removing BN entirely is a separate, confounded intervention: baseline and legacy
collapsed, while ours completed with a failed reference audit. Do not promote it
to a qualified BN-free scheme comparison or repeat it with unchanged settings.

Proposed tests, in order, not launches:

1. **Late affine-BN freezing, one epoch.** For legacy, ours and baseline, branch
   each epoch30 checkpoint into continuing affine learning versus fixing its
   already learned gamma/beta. Keep minibatch normalization and running-stat
   updates active; never reset gamma/beta to1/0. Keep optimizer history, rates,
   data order and evaluation cohorts fixed. Six branch-epochs, potentially five
   new ones if exp-014's legacy control is exactly reusable. Measure at0/100/400/
   1407 updates with the existing fixed-cohort instrument. A selective improvement
   for legacy in both training CE and validation supports a late co-adaptation
   explanation; equal changes across schemes do not explain its relative gap.
   This asks a new question from the completed from-initialization ablation.

2. **Half convolution LR, conditional on test1's outcome.** Original versus0.5×
   convolution LR for legacy and ours, leaving BN/head/gain learning unchanged.
   Reuse matched controls from test1 when valid; only two new branch-epochs may
   be needed. Require a sustained fitting improvement without validation damage
   at400 and1407 updates, rather than a favorable single step. Compare the effect
   between schemes. A harmful3× step alone is not evidence that0.5× will help.

3. **Replication before a general mechanism claim.** After short screens identify
   a useful comparison, use two additional paired seeds with frozen settings and
   a common software stack. This tests whether the original endpoint ranking and
   any proposed repair survive seed variation. Do not treat additional epochs,
   minibatches or evaluation cohorts as independent seeds.

Do not launch all tests as one grid. Review test1 before choosing the next branch;
only extend a promising short intervention to3–5 epochs. Full-run claims still
require the declared endpoint and seed coverage. Existing solver qualification,
epsilon-only and BN-recalibration checks need no routine repetition.

September28 follow-up: [H-010](../hypotheses/H-010-frozen-bn-readout-rate.md)
tests sustained dense-readout rate reductions with BN affine frozen, using
matched epoch10 states and400 updates per branch in
[exp-016](../series/002-residual-gap/experiments/exp-016-dense-readout-lr.md).
Raw gradient magnitudes motivated inspection but are not treated as Adam update
magnitudes. Original unnormalized legacy and ours remain distinct schemes.
