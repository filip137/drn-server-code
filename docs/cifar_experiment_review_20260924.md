# CIFAR-10 experiment review, September 24, 2026

This review consolidates the CIFAR evidence through September 24: the digital
L12/L8 references, analog L8 batch and learning-rate searches, continuations,
implementation checks, and BN/epsilon/gradient/gain investigations. It uses
the collected results and completed analyses; no new training or replay was
needed. Historical MNIST experiments supplied LR priors and a motivation for
the amplification comparison, but their results are not CIFAR evidence.

This file contains the consolidated measurements and interpretation. Detailed
evidence links resolve in the research worktree; generated reports, plots and
checkpoints are stored separately from Git.

The wider L8 retained the observed digital L12 performance with approximately
half the parameters. Analog L8 reached 89.48% baseline, 89.92% proposed and
88.72% legacy validation accuracy at epoch 50. Trainable BN helps all three
under their selected rates. Epsilon interacts with learning rate, but correcting
it alone leaves normalized legacy at 88.82%. The tests have not identified one
cause of the remaining legacy gap, or established a reproducible scheme ranking
across seeds.

The main conclusions are:

- Wider digital L8 retains the observed L12 accuracy with approximately half
  the parameters. The digital test results and analog validation results use
  different splits and must remain separate.
- Trainable BN affine parameters help every scheme at its selected rates.
  Removing BN produces two early collapses and one solver-unqualified result.
- BN epsilon interacts strongly with LR, but correcting effective epsilon
  alone gives only +0.10pp for legacy at epoch 50. Recalibrating BN statistics
  does not repair its final accuracy gap either.
- The KCL checks and trained-checkpoint audits pass. Legacy's larger raw
  gradients do not imply larger relative Adam updates; the sampled nominal
  steps do not overshoot same-batch loss.
- Original scalar input gains barely adapt. Stronger log-gain learning beats
  fixed gains in the matched frozen-BN pair, but remains below the existing
  softplus-gain reference.
- Baseline is flattest over the final ten epochs by averaged validation
  accuracy; legacy gains least over epochs 30–50. All four analog runs still
  reduce training loss. These are distinct definitions of stagnation.
- One seed, mixed runtime environments and the tested hyperparameter ranges
  limit broader claims. Legacy's final disadvantage remains unexplained;
  there is no established global LR/epsilon optimum or multi-seed ranking.

**What was held fixed.** The current analog network has eight convolutions in
three blocks of depths [3, 3, 2], logical widths 128/256/512 and differential
physical widths 256/512/1024. All convolutions and the 8192-to-10 classifier
are analog resistive layers. Max-pooling and boundary BN are digital; the
circuits are separately voltage-clamped. Thus electrical feedback stays within
each block, while BPTT crosses the digital boundaries.

The main analog comparisons use seed 0, the same 45,000/5,000 split of CIFAR
training data, Adam, crop/flip augmentation, cross-entropy without label
smoothing, batch 32/evaluation batch 16, and a 50-epoch cosine horizon ending
at 2% of the starting LR. BN affine parameters are trainable except in the
declared frozen/no-BN controls. Four scalar boundary gains start at 100.
Convolution solver budgets are T=K=[6,6,4], with higher-iteration audits.
Baseline uses voltage/current amplification (1,1), proposed (4,1), and legacy
(4,0.25). Each scheme uses its selected conductance/head LR vector.

Digital references instead use batch 512 and all 50,000 training / 10,000
official-test images. They monitor the test set each epoch. Their accuracy
must remain separate from analog validation accuracy; these are exploratory
references rather than untouched-test estimates. Mixed hardware/software and
one seed also limit causal comparisons.

**The completed test families are listed below.** Counts describe scientific
cases or diagnostic coverage; continuations and repeated executions from one
seed are not independent replications. Setup smokes are not accuracy evidence.

| Test | Coverage | Result and interpretation |
|---|---|---|
| Digital depth comparison | One 50-epoch L12 and one 50-epoch wider L8 | L12 final test 92.30%; L8 final 92.57%, best 92.62%. L8 has 5.40M versus 10.85M parameters. Supports the smaller-depth reference, without isolating depth from the changed BN/pooling arrangement. |
| Analog batch size | Legacy, batches 16/32/64, five epochs each | Validation 61.02/65.50/60.02%; batch 32 chosen provisionally. Later matching 5090 batch-32 case reached 65.62%. |
| Three-scheme LR search | 27 five-epoch grid cases, two proposed edge cases, six ten-epoch finalists | Selected baseline/proposed/legacy reach 75.26/75.36/74.68% at ten epochs. All 35 scientific cases complete and pass final solver audits. |
| Legacy LR refinement | Five ten-epoch cases: current rates and separate conv/head half/double changes | Current rates win at 74.68%; doubling conv rates is worst at 70.80%. No tested individual change improves the ten-epoch setting. |
| Longer training | Original three trajectories resumed 10→30→50 with full optimizer/scheduler/BN/RNG state | All improve substantially. At 50, proposed leads baseline by 0.44pp and legacy by 1.20pp. |
| Physical equations and solver implementation | 136 synthetic checks, including 21 CIFAR architecture checks | KCL currents, energy derivatives, coordinate updates, states and gradients agree with independent checks and the fully analog MNIST solver path on the tested small circuits. |
| Matched-coordinate training identity | Baseline versus normalized legacy for 200 real minibatches; matched-weight replay at 0/10/30/50 | With the complete baseline optimizer, normalization gives identical logits, gradients, BN state and updates over the tested window. |
| Trained-checkpoint solver/voltage/gradient audit | Three original schemes × epochs 0/10/30/50 = 12 checkpoints, fixed cohort | Every native solver audit passes. No systematic legacy conductance-gradient collapse or inadequate settling is found on this cohort. |
| BN affine learning | Three frozen-affine controls, ten epochs each | Baseline/proposed/legacy drop 4.54/1.60/1.66pp. Trainable affine BN helps all three at these rates. |
| Removing BN | Three attempted ten-epoch cases | Baseline and legacy collapse to 10% after one epoch. Proposed reaches 52.08% at ten but fails final solver qualification. No complete qualified BN-free comparison. |
| Effective BN epsilon | Reduced-epsilon baseline to ten; normalized legacy to ten and then fifty | Strong epsilon/rate interaction early; normalized legacy ends at 88.82% versus raw legacy 88.72%. No sustained late validation improvement from this correction alone. |
| BN inference-statistics recalibration | Seven checkpoints, 4,096 training examples to fit statistics, full validation before/after | CE improves in all seven. Legacy epoch-50 accuracy falls 88.72→88.34%, so this policy does not repair its final gap. |
| Raw gradients versus actual Adam movement | Layerwise analysis of the 12 checkpoint replays; nine saved-Adam checkpoints at 10/30/50, four directions each | Legacy's larger gradients do not imply proportionally larger updates. All 36 sampled directions improve same-batch CE with half/nominal/double steps; no nominal same-batch overshoot is observed. |
| Learned gains and stronger gain adaptation | Original gain extraction at 0/10/30/50; matched fixed/log-gain legacy controls to ten epochs with frozen affine BN | Original block gains change at most 0.353%. Fixed/log gains reach 68.00/69.72%; log gains adapt substantially, but remain below the historical softplus-gain reference at 73.02%. |

**Digital references.** Both use ordinary feed-forward BP, Adam LR 0.001 with
cosine decay, crop/flip augmentation and cross-entropy. L12 uses non-affine BN;
L8 trains BN gamma and beta, with all three pairs verified to change.

| Digital reference | Epoch-10 test accuracy | Final test accuracy, epoch 50 | Best test accuracy | Parameters | Run time on Akib RTX 3080 |
|---|---:|---:|---:|---:|---:|
| Wider L8, trainable BN | 82.65% | 92.57% | 92.62% at 48 | 5,395,594 | 12m15s |
| L12, non-affine BN | 80.46% | 92.30% | 92.30% at 50 | 10,849,674 | 16m16s |

This is a joint architecture/BN comparison. It does not independently measure
BN's contribution or establish an analog L12-to-L8 speedup. Earlier records
cite best hybrid/L12 "all-analog" accuracies of 92.32/85.04%; these are historical
context from the L12 report, not revalidated matched controls in this review.
[Digital L8 evidence](../results/cifar10-digital-l8-wide-affinebn-adam-seed0-20260921-v1/analysis/report.md)
and [digital L12 evidence](../results/cifar10-digital-l12-logical-width-adam-seed0-20260921-v1/analysis/report.md).

**Batch and learning rates.** Batch 32 led throughout the five-epoch batch
comparison. Its first run used a different GPU/software stack; the later
65.62% 5090 control closely reproduced the 65.50% result. This supports using
32 for this setup, without proving a long-run optimum. The LR grid varied
conv and head rates independently around MNIST-informed priors, while BN and
gain LRs stayed fixed at 0.001 and 0.00005.

The selected conv vectors are approximately 3.001× / 9.001× / 1× the legacy
anchor for baseline/proposed/legacy. The legacy anchor in convolution order is
[0.0045, 0.00069, 0.00069, 0.00069, 0.00049, 0.00049, 0.00049, 0.000345].
Selected classifier LRs are 8.336e-4 / 5.334e-5 / 1.600e-4. Proposed therefore
uses larger convolution rates but a smaller head rate; there is no single
scalar LR that describes the comparison. Exact vectors remain in the
[LR report](../results/cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1/analysis/report.md).

| Legacy ten-epoch refinement | Validation accuracy |
|---|---:|
| Current selected rates | 74.68% |
| Head LR ×0.5 | 74.44% |
| Head LR ×2 | 74.34% |
| Conv LRs ×0.5 | 74.26% |
| Conv LRs ×2 | 70.80% |

The center is reproduced, and the nearest alternatives differ only slightly.
Joint conv/head changes and long-horizon optima are not established by this
local refinement. The proposed head winner also lies at the smallest tested
rate in its bounded search.
[Batch evidence](../results/cifar10-l8-analog-conv-and-head-adam-bs16-32-64-seed0-v1/analysis/report.md)
and [legacy refinement](../results/cifar10-l8-legacy-lr-refinement-e10-seed0-20260922-v1/analysis/report.md).

**Longer training and epsilon.** The primary comparison uses the same final
epoch, rather than choosing a different best checkpoint for each case.

| Analog condition | Epoch 10 validation | Epoch 30 validation | Epoch 50 validation | Validation CE at 50 | Training CE at 50 |
|---|---:|---:|---:|---:|---:|
| Baseline | 75.26% | 86.54% | 89.48% | 0.368819 | 0.060967 |
| Proposed | 75.36% | 87.20% | 89.92% | 0.372177 | 0.036016 |
| Legacy | 74.68% | 86.88% | 88.72% | 0.385642 | 0.071951 |
| Normalized legacy | 75.82% | 86.44% | 88.82% | 0.390240 | 0.060160 |

Legacy exceeds baseline at 30 but falls behind by 50. Its best original
accuracy is 88.82% at 44; normalized legacy's best is 88.94% at 49. These best
values are separate from the declared final endpoint. Continuing all three
selected optimizers was productive, but their ranking is still single-seed
evidence. Baseline has slightly better final CE than proposed despite lower
accuracy. The [10→30](../results/cifar10-l8-analog-continue-e10-e30-seed0-20260922-v1/analysis/report.md),
[baseline/legacy 30→50](../results/cifar10-l8-analog-baseline-legacy-e30-e50-seed0-20260922-v1/analysis/report.md)
and [proposed 30→50](../results/cifar10-l8-analog-ours-e30-e50-seed0-20260922-v1/analysis/report.md)
records preserve the full-state continuation evidence.

Baseline and legacy both have A×B=1. At matched weights, legacy changes block
output units by 16/16/4. With production BN epsilon 1e-5, its effective epsilon
in baseline units is [1e-5/256, 1e-5/256, 1e-5/16]. Dividing outputs before BN
restores baseline effective epsilon while preserving legacy KCL equations and
its selected rates.

| Selected optimizer, ten epochs | Baseline effective epsilon | Smaller legacy effective epsilon |
|---|---:|---:|
| Baseline rates | 75.26% | 70.18% |
| Legacy rates | 75.82% | 74.68% |

The 5.08pp versus 1.14pp penalties show optimizer/epsilon coupling in these
cases. Reducing baseline epsilon also worsens training CE substantially, so
the effect is not limited to inference statistics. Raw BN running-variance
initialization has a transient difference in units; matching training prefixes
support the practical coordinate comparison. At 50, correcting legacy epsilon
gives only +0.10pp, lowers training CE, and slightly worsens validation CE.
Mean accuracy improvement over epochs 41–50 is +0.008pp; these correlated
epochs are not independent replications. Epsilon matters, but this correction
alone does not close the final gap at the selected legacy rates.
[Early controls](cifar_l8_bn_conclusions_and_next_experiments_20260923.md)
and [completed normalized continuation](../results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/analysis/report.md).

**BN learning, removal and inference statistics answer different questions.**
The affine parameters are the per-channel learned scale gamma and shift beta
applied after normalization. They are distinct from the scalar input gain of
each analog block. Frozen affine BN keeps minibatch normalization and
running-statistic updates, while fixing gamma=1 and beta=0. Removing BN
removes all of those operations.

| Scheme | Trainable BN at ten epochs | Frozen affine BN at ten epochs | No-BN result |
|---|---:|---:|---|
| Baseline | 75.26% | 70.72% | 10% after one epoch; zero-output/gradient collapse |
| Proposed | 75.36% | 73.76% | 52.08% after ten; final solver audit failed |
| Legacy | 74.68% | 73.02% | 10% after one epoch; zero-output/gradient collapse |

Frozen affine BN is worse at every measured epoch in each scheme. Learning
its affine parameters is not supported as a legacy-specific harmful mechanism.
The no-BN tests use BN-selected rates and radically different signal scales.
Proposed's reference-versus-sentinel free-logit discrepancy is 8.61%, above the
1% limit; 52.08% is an observed finite-iteration result, not qualified
steady-state accuracy. The two collapsed runs were stopped at the first
epoch-2 batch, with exact failure replay, and never reached final audits.
[BN ablation evidence](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/report.md).

Recalibration is read-only: fit sequential BN1→BN3 moments using 4,096 training
examples without augmentation, then compare full validation accuracy on
disposable model copies. Original checkpoint bytes and primary scores remain
unchanged.

| Checkpoint | Saved BN accuracy | Recalibrated BN accuracy |
|---|---:|---:|
| Baseline 10 | 75.26% | 76.08% |
| Proposed 10 | 75.36% | 76.66% |
| Legacy 10 | 74.68% | 75.82% |
| Normalized legacy 10 | 75.82% | 75.54% |
| Baseline 50 | 89.48% | 89.62% |
| Proposed 50 | 89.92% | 89.76% |
| Legacy 50 | 88.72% | 88.34% |

CE improves in all seven, but legacy's final accuracy gap widens. Thus this
recalibration policy does not repair the late ranking. At ten, the normalized
versus raw legacy advantage also disappears after recalibration, qualifying
the early saved-stat result.
[Paired BN and Adam diagnostic tables](../results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/analysis/diagnostic_report.md).

**The implementation and gradient tests weaken several simple explanations.**
The [independent KCL audit](cifar_l8_amplification_audit_20260923.md) supports
the implemented physics in the tested circuits. All 12 original trainable-BN
checkpoints pass solver/gradient audits. These are cohort-based numerical
checks, not full hardware validation or a proof for every input.

The blocks share the MNIST solver engine, but have a different topology:
digital clamps stop electrical feedback between blocks. The isolated one-edge
analog classifier has no internal amplified edge, so A/B have no effect at
fixed classifier input/weights/gain. Within a block, normalized legacy has the
same matched-weight loading as baseline. With baseline's complete optimizer,
the 200-batch equality check gave zero observed differences in logits, loss,
all trainable gradients, BN buffers, parameters and projected Adam updates.
This explains why intuition from a fully coupled MNIST circuit need not carry
over directly; it does not establish equality of the separately trained models.

Legacy/proposed median convolution gradient ratios at epochs 10/30/50 are
6.34/6.51/8.63, whereas relative projected Adam-update ratios are
0.903/1.111/0.664. Legacy's smaller learned conductance norms contribute to
the raw gradient scale: multiplying gradient RMS by conductance norm reduces
the epoch-50 median ratio from 8.63 to 1.06. The trained networks are not
asserted to be uniform rescalings.
[Layerwise gradient analysis](cifar_l8_gradient_scale_interpretation_20260923.md).

The subsequent saved-Adam probes test three schemes at 10/30/50, four gradient
batches each, step multipliers 0/0.5/1/2, and a separate fixed training probe
cohort. All 36 directions strictly lower same-batch CE as the step increases
through the tested multipliers. Separate-cohort effects are mixed, and early
legacy steps hurt that cohort more. By 50, legacy's nominal same-batch logit
movement is 0.0842%, close to proposed's 0.0819%, despite much larger raw
gradients. These probes do not support uniform nominal-step overshoot; they
also do not justify doubling training LR. Gradients use train-mode BN while
paired evaluations use saved BN buffers, and epoch-50 steps are hypothetical
next steps at the cosine floor. Noise and accumulated optimization history
remain open.

**Scalar gains barely learned under the original parameterization.** The
original softplus gains started at 100. Their final values were:

| Scheme | Block 1 | Block 2 | Block 3 | Classifier input |
|---|---:|---:|---:|---:|
| Baseline | 100.051 | 100.123 | 100.353 | 100.328 |
| Proposed | 100.069 | 100.113 | 100.274 | 100.277 |
| Legacy | 99.994 | 100.095 | 100.001 | 100.299 |

The original gain LR was 5e-5, decaying to 1e-6. The largest block-gain change
is 0.353%; late block-gain update proposals round to zero in float32. This
occurs in all schemes. Later scalar gains are functionally redundant with
preceding BN gamma/beta, so almost constant scalar gains do not imply fixed
downstream voltage scales. See [checkpoint gain measurements](cifar_l8_mechanism_findings_20260923.md).

The new gain pair retains frozen affine BN and legacy conductance/head rates;
it changes only the three convolution gain policies. The classifier gain
keeps its original parameterization and LR.

| Gain policy, ten epochs | Validation accuracy | Final block gains |
|---|---:|---|
| Fixed at 100 | 68.00% | 100 / 100 / 100 |
| 100×exp(theta), theta LR 1e-3 | 69.72% | 114.114 / 191.112 / 33.360 |
| Existing softplus-gain reference | 73.02% | 100.005 / 100.032 / 99.961 |

Log gains beat fixed gains by 1.72pp and lead throughout epochs 6–10, but do
not beat the historical softplus reference. Both new final solver audits pass.
The fixed/log pair is primary; the softplus run is a secondary historical
reference with different host history. Initializers, cohorts, first-batch
summaries and software settings were checked, but complete cross-host
numerical equivalence is not independently proven. Parameterization and
relative learning rate change together. These results show that substantial
adaptation is possible, without establishing that it repairs legacy's gap.
[Gain comparison and completed follow-up](../results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/analysis/report.md).

**Coverage and exclusions.** All declared cases have a recorded outcome, and
the authoritative results are local. The BN-free scientific failures above
remain failures; they are not counted as qualified ten-epoch successes.
The normalized continuation paused at 48 before the overnight cutoff, then
completed 49–50 after explicit daytime authorization. Its native final audit
passes, the paused ancestor is preserved, and no required continuation remains.
The overnight/final follow-up used 13.191 of its 16 GPUh allowance and has no
remaining workers.

Cancelled proposed/legacy reference repeats were not run. The stopped
supplementary legacy-at-baseline-LR prefix is auxiliary and overlaps the
baseline-epsilon condition; it is not an independent primary result. A
redundant baseline repeat is also auxiliary. Akib's CPU-offloaded continuation
attempt was stopped before a complete added epoch and replaced with GPU-only
execution. Environment-smoke failures and interrupted prefixes remain in the
individual records, excluded from scientific ranking. Neither cancelled work
nor numerical failures have been silently replaced by successful scores.

**What remains unresolved.** The current evidence supports keeping L8,
trainable affine BN, augmentation and cross-entropy as the working setup.
It supports continued training at the selected rates and gives no basis for
blanket LR reduction from raw gradient size alone. It does not isolate the
benefit of augmentation or cross-entropy: neither was separately ablated.
We have not established an optimal BN epsilon, a globally optimal long-horizon
LR vector, a qualified BN-free comparison, a causal account of the final
legacy gap, or a multi-seed scheme advantage. Digital/analog differences also
include datasets, parameterizations, optimizers and batches, so subtracting
their accuracies is not a matched estimate of an analog penalty. No further
jobs are scheduled by this review.

**Which run stagnated most near epoch 50?** A follow-up comparison uses the
four completed analog trajectories, with no new training or evaluation.
Define the late accuracy trend as mean validation accuracy over epochs 46–50
minus its mean over 41–45, reducing sensitivity to one noisy endpoint. Also
report the broader epoch-30-to-50 endpoint gain rather than treating those
two questions as interchangeable.

| Run | Accuracy gain, 30→50 | Mean accuracy, 41–45 | Mean accuracy, 46–50 | Late mean gain | Best-accuracy epoch |
|---|---:|---:|---:|---:|---:|
| Baseline | +2.94pp | 89.256% | 89.364% | +0.108pp | 50 |
| Proposed | +2.72pp | 89.228% | 89.700% | +0.472pp | 50 |
| Legacy | +1.84pp | 88.496% | 88.676% | +0.180pp | 44 |
| Normalized legacy | +2.38pp | 88.392% | 88.796% | +0.404pp | 49 |

Baseline is flattest by this last-ten-epoch accuracy measure, followed by
legacy. Linear slopes across epochs 41–50 give the same ordering:
baseline 0.0205, legacy 0.0273, normalized legacy 0.0770 and proposed
0.0941 percentage points per epoch. Legacy gains least over the broader
30→50 interval and has the earliest best-accuracy checkpoint, but it is not
uniformly the flattest under every definition. Simple endpoint gains 40→50
are baseline +0.70pp, proposed +0.84pp, legacy +0.76pp and normalized legacy
+0.70pp, illustrating the sensitivity to individual epochs.

No run has stopped fitting the augmented training stream: from epochs 40
to 50, training CE falls 42.5% for baseline, 49.0% for proposed, 38.8% for
legacy and 40.5% for normalized legacy. Validation CE gives a different
perspective. Mean CE over 46–50 minus 41–45 is -0.00194 baseline, +0.00106
proposed, -0.00420 legacy and -0.00103 normalized legacy. Proposed still
improves accuracy most near the end while its mean validation CE worsens
slightly. These are descriptive, correlated measurements from one seed under
a cosine schedule approaching its floor, not a statistical ranking or proof
that extending any run will improve generalization.

[Late-training measurements and method](../results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/analysis/stagnation_e50.json)
and [curves](../results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/analysis/stagnation_e50.png).
