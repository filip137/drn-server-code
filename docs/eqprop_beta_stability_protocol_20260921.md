# Provisional EqProp beta selection by training stability

Recorded September 21, 2026 at Filip's request. Status: scientific direction
recorded; numerical choices and the scope of stability qualification remain
provisional. This document records the proposed protocol and the evidence
needed to support its wording. It does not launch or replace any experiment.

## Requested decision

The selection objective is to choose the largest beta for which training does
not diverge. Filip proposed the following approximate depth-dependent guide:

- Conv1: about one decade below p99.
- Conv2: about p99.
- Conv3: still being tested, with p95 as the current working candidate.

Here p99 and p95 refer to cosine thresholds .99 and .95, not percentiles.
They are reference points for choosing candidates; a cosine threshold alone
does not certify training stability. The approximate relationships are a
working hypothesis, not established maximum-stability boundaries.

## Scope and definitions

Current scope is ordinary-MNIST wide [0,100] Conv1/Conv2/Conv3, Adam with the
unchanged scheme-specific learning-rate vectors, corrected physical-KCL
dynamics, float64 centered frozen-current EqProp, and frozen exact-zero
biases. Keep T=K=4/6/8 and full epoch budgets 10/30/30. Bounded-weight and
BPTT runs retain their separate contracts.

The current reference calibration uses zero endpoint noise, every weight
matrix, all 36 batches at initialization and all 36 at the saved BPTT checkpoint.
It takes the largest measured beta with cosine strictly above the threshold,
without a norm gate. Preserve the checkpoint/cohort/grid identity and report
norm mismatch separately. These reference betas are seed-0 measurements.

The operational objective is the **largest tested stable beta under the
declared training conditions**, not a proven continuous or universal maximum.
Numerical nondivergence requires the full declared epoch budget to complete
with finite states, gradients, losses, and checkpoints. Ten-epoch Conv2/3
pilots are early evidence, not thirty-epoch qualification. Preserve the
existing additional screen (final validation strictly less than 5 percentage
points below the run's own best) and temporary drawdowns as separate reported
measures; do not conflate numerical nondivergence with satisfactory accuracy.

Freeze the beta grid, training/noise conditions, seeds, horizon, and inclusion
rule before evaluating the final comparison. Keep scientific failures in the
record. Candidate selection must use training/validation data only. Official
test evaluation remains sealed until the final scientific contract and
checkpoint rule are frozen. Report results at other betas as sensitivity
evidence rather than silently mixing them into the selected-beta comparison.

## Numerical interpretation under review

All values below are actual injected beta; columns are baseline/ours/legacy.

| Architecture | Proposed reference | Candidate beta values |
|---|---|---|
| Conv1 | Latest measured p99 divided by 10 | 30 / 30 / 6 |
| Conv1, alternative reading | Retain historical stable choices described as one decade below the earlier limits | 100 / 30 / 3 |
| Conv2 | Latest measured p99 | 100 / 10 / 3 |
| Conv3 | Refined p95, pending qualification | 147.682614594 / 2.49274796756 / 2.81845428732 |

The Conv1 interpretation is unresolved: the latest p99 values are 300/300/60,
whereas the older candidate limits were 1000/300/30. The asynchronous
clarification asks which reference Filip intends. The companion audit provides
both interpretations; it provisionally uses the latest p99 reference for the
literal formula and does not silently substitute historical values.

A second clarification asks whether nondivergence is required only in clean
training or throughout the noise range through sigma 1e-3. This materially
changes the selection objective. If selection is clean-only, noisy failures
are robustness results and do not trigger automatic reselection. If stability
under noise is part of selection, declare the noise range and realization(s)
and require the selected beta to qualify under those conditions too. No
additional noise grid is authorized by this record.

## What the existing evidence supports

- Historical Conv1 Adam candidates 1000/300/30 diverged, while their 100/30/3
  controls completed ten epochs. Values another decade lower were also
  stable. Those earlier tests used T=K=8 and trainable biases; legacy also
  preceded the physical-KCL correction. They support the original safety
  choice but do not locate the present configuration's stability boundary.
- Current Conv1 baseline 100 and 200 both completed ten-epoch clean/noisy
  studies. Therefore literal latest-p99/10=30 is not demonstrated to be its
  largest stable beta. Current ours 30 and legacy 3 have complete evidence;
  proposed legacy 6 has no matching full-budget evidence in this audit.
- Current Conv2 baseline 500 and ours 30/50 completed ten clean epochs,
  although their p99 references are 100 and 10. Thus p99 is not established as
  the maximum for ten-epoch clean stability. Full thirty-epoch/noisy evidence
  for those larger values is not supplied by those pilots. Legacy 30 failed
  in epoch 5; legacy 3 remains a candidate needing training qualification.
- All three Conv3 p90 values completed thirty clean epochs. At sigma 1e-3,
  p90 baseline completed thirty epochs, while legacy and ours became
  nonfinite in epochs 8 and 18. Hence p95 cannot be described as the demonstrated
  clean-training limit; it is a candidate for a more robust operating point.
- All three exact Conv3 p95 values completed ten clean epochs. A separate
  six-case study covers thirty-epoch clean and sigma 1e-3 conditions. The local
  summary has no verified completed production cases at this review. Its
  monitor recorded an SSH timeout; this is not a scientific divergence result
  and does not establish whether the remote jobs stopped. Reconcile that
  already-declared work before scheduling duplicates.

We can currently claim empirically stable tested operating points and coarse
failure brackets. We cannot yet claim that a common depth-dependent factor
finds each scheme's largest stable beta, or that one beta is optimal for noisy
accuracy. Additional intermediate points would be needed to locate a closer
stability boundary under an unchanged training contract.

## Draft methods wording, conditional on completing qualification

> We use gradient-fidelity measurements to nominate scheme-specific nudging
> strengths, and select the largest tested value that satisfies a predeclared
> training-stability criterion under the fixed optimizer, learning rates, and
> relaxation schedule. Candidate values are guided by per-matrix EP–BPTT
> cosine thresholds measured at initialization and a trained reference
> checkpoint. Stability is assessed over the full training horizon, and the
> selected beta is then held fixed across the reported comparison. Calibration
> and selection use the training/validation partition; the official test set
> is reserved for the final frozen protocol.

Before using this as completed-methods text, state whether stability selection
includes noise and document the candidate search and outcomes. Add the
Conv1≈p99/10, Conv2≈p99, Conv3≈p95 relationship only as an observed approximate
summary once the numerical references and qualifying evidence are settled.

## Evidence and run review

- [Run coverage and conditional redo counts](../paper_ready_results/beta_stability_protocol_run_audit_20260921.md)
- [Existing p99 compatibility review](../paper_ready_results/p99_beta_run_coverage_20260921.md)
- [Historical safety-factor comparisons](beta_study.md)
- [Calibration](../paper_ready_results/beta_rule_comparison_20260918.md) and [refinement](../paper_ready_results/beta_refinement_20260919.md)
- [Current ten-epoch training comparisons](../paper_ready_results/layerwise_beta_training_20260919.md)
- [Conv3 p95/p90 pilots](../paper_ready_results/conv3_refined_beta_training_20260919.md)
- [Existing Conv3 p95 thirty-epoch plan](eqprop_conv3_p95_read_noise_1em3_plan_20260921.md)
