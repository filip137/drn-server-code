# Experimental Manifest

Updated: 2026-09-24

This is the manually curated ledger of analyzed studies. It is not generated
from raw metrics and does not track transient job state.

Each finished entry records:

- scientific question and evidence class;
- frozen setup, arms, seeds, and inclusion rule;
- outcome: `positive`, `negative`, `mixed`, or `inconclusive`;
- headline measurements;
- interpretation;
- limitations and protocol deviations;
- raw run and study-analysis locations; and
- links to detailed tables, plots, or reviewed historical result cards.

As of 2026-08-17, ordinary MNIST is the active paper dataset. Its held-out
5,000-example validation accuracy remains selection/mechanism evidence and
must never be presented as paper accuracy. Only a once-read official-test
metric from a sealed eligible checkpoint is paper-facing. Existing entries
retain their recorded evidence class until a separate reuse audit and test
evaluation explicitly promote them. An operational failure needs an entry
only when it materially affects the scientific conclusion.

The [CIFAR experiment review, September 24](cifar_experiment_review_20260924.md)
consolidates the digital references, analog searches and continuations, and
BN/epsilon/gradient/gain controls, with completed and failed cases distinguished.

## CIFAR L8 overnight BN, Adam-step and gain controls, September 24

Outcome: **mixed; declared coverage complete through epoch 50**.
This seed 0 follow-up tests whether BN inference statistics, excessive Adam
steps, voltage normalization or input-gain adaptation explain legacy's CIFAR
disadvantage. All cases retain the existing 45,000/5,000 split and selected
rates; training uses augmentation, cross-entropy, batch 32 and the original
50-epoch cosine horizon. Normalized legacy retains trainable affine BN and
divides block outputs by 16/16/4 without changing legacy KCL physics. The
gain pair fixes BN gamma 1/beta 0 while updating its running statistics.

- **BN recalibration:** all seven checkpoints completed. At epoch 50,
  legacy accuracy changes 88.72→88.34%, baseline 89.48→89.62%, and
  proposed 89.92→89.76%. CE improves, but legacy's accuracy gap widens.
  This recalibration policy does not repair the observed disadvantage.
- **Saved-Adam probes:** all nine checkpoints at 10/30/50 completed. Every
  one of 36 sampled directions lowers same-batch CE with half, nominal and
  double projected steps, with double improving most. Separate-cohort effects
  are mixed. Larger raw gradients alone do not establish excessive effective
  updates; these local probes do not establish optimal training rates.
- **Frozen-BN gain pair:** fixed gains reach 68.00%, log gains 69.72% at 10;
  log gains finish 114.114/191.112/33.360 from 100. The log case leads over
  epochs 6–10 by an average 1.02pp, but both remain below the historical
  softplus-gain reference 73.02%. Parameterization and relative step size
  change together; historical reference placement is a comparison limitation.
- **Normalized legacy:** full-state continuation completes epochs 11–50,
  ending at 88.82%; matched epoch 50 raw legacy/baseline/proposed are
  88.72/89.48/89.92%. Normalized training CE is lower than raw legacy
  (.06016/.07195), but validation CE is higher (.39024/.38564).
  The endpoint gain is only .10pp, and mean normalized-minus-raw accuracy
  over 41–50 is +.008pp. Correcting effective epsilon alone at the selected
  legacy rates does not demonstrate a sustained late validation benefit.
- **Epsilon interpretation:** the reused ten-epoch controls lose 5.08pp when
  baseline uses legacy's smaller effective BN epsilon, versus 1.14pp with
  legacy's rates. Epsilon affects training and interacts with rates; the
  completed continuation does not establish that epsilon is irrelevant or
  identify an optimum. The original four-way comparison remains single-seed
  evidence, with a transient BN running-variance initialization distinction.

All nine production bundles validate locally: eight complete and one preserved
paused ancestor. Both gain endpoints, the separately replayed epoch 48
checkpoint and the native epoch 50 endpoint pass solver/gradient audits.
The deliberate overnight pause (exit 75) preserves model/Adam/scheduler/BN/
gain/RNG state without a successful result file; its approved child completes
the trajectory without modifying that ancestor. One local cuDNN smoke failure
is retained and excluded; the unchanged remote smoke passed. The prepared
recovery selector never ran. Overnight workers on Fifi, Trex, Loulou and Riri
were gone by 07:49 Paris, before the 08:00 deadline. Filip explicitly approved
the final two epochs beyond that cutoff; they completed on Fifi within the
additional one-hour cap, and its workers have exited. Recorded total compute
13.191 GPUh is below 16 GPUh; shared-host timing is not an exclusive benchmark.

Keep trainable BN, augmentation and CE as the current default. These results
do not justify blanket LR reduction based on raw gradient scale or replacement
of the existing gain policy. A matched additional seed should precede strong
claims about small accuracy differences; any future epsilon tuning should
account for its interaction with LR. No further jobs are scheduled.
This is exploratory, single-seed validation evidence; no official test was read.

The subsequent late-training comparison distinguishes accuracy plateau from
overall improvement: mean accuracy at 46–50 minus 41–45 is baseline +0.108pp,
legacy +0.180pp, normalized legacy +0.404pp and proposed +0.472pp. Baseline
is flattest by that measure; legacy gains least over 30→50 (+1.84pp).
All four still reduce training CE substantially, while proposed's late mean
validation CE worsens slightly despite improving accuracy. Window/metric
dependence and single-seed limits are retained in the
[consolidated review](cifar_experiment_review_20260924.md).

[Full report and next steps](../results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/analysis/report.md) ·
[Training and gain curves](../results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/analysis/training_and_gains.png) ·
[Coverage, hashes and budget](../results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/analysis/collection_validation.json) ·
[Execution plan](cifar_l8_bn_gain_overnight_plan_20260923.md).

## Conv3 T8 check at output-displacement-one betas, September 23

Outcome: **negative for a material T effect on gradient quality** at these
initial and p99-trained checkpoints. Four new T8/K8 cases compare directly
against the four completed T16/K8 RMS-one-beta replays. Ours injected beta
1.385 and legacy .02173, initialization and p99 epoch30, 36 matched validation
batches, float64 centered frozen-current EqProp, clean reads plus four paired
sigma5e-4 draws. Source/checkpoint bytes, runner, cohort, noise seeds, Akib
RTX3080 device and PyTorch/CUDA/Python environment match. Only T changes.

Maximum noisy layer-median cosine change is **1.50245e-7**; maximum clean
median change is **8.72897e-8**. Maximum individual or paired-batch cosine
change is **1.41724e-6**. Noisy Conv3 medians at T8/T16 are legacy
initialization .238445325/.238445314, ours initialization
.697435152/.697435147, legacy epoch30 .276864701/.276864710, and ours
epoch30 .753430189/.753430039. Ours retains the strong noisy Conv3 advantage;
both early layers remain noise-dominated. Gradient RMS, norm ratios and
noise/clean ratios change by at most7.35e-6 relatively across individual
measurements. The largest relative gradient-error change is1.97% only because
a tiny clean readout error changes from4.81314e-6 to4.90782e-6; maximum
absolute change in any relative gradient error is7.83857e-6.

T affects a separate displacement diagnostic: legacy initialization hidden1
positive-free RMS is1.69770e-7 at T8 versus6.75692e-8 at T16, while centered
phase RMS is6.75742e-8 at both. Zero-nudge continuation RMS1.54506e-7 at T8
versus5.30729e-13 at T16 identifies residual common relaxation. Across all
layers/checkpoints, pooled centered phase RMS changes by at most2.72798e-7
relatively. All2,304 new residual checks pass, maximum7.92642e-6 below .01.

All four new production/eight smoke bundles validate locally. All2,880 new
rows pair against the2,880 validated T16 rows;119 remote file hashes match.
No failures, exclusions or missing cases. Akib exited zero and released the
GPU;337.22s production,625.35s combined CPU/GPU smoke and replay within1,800s.
This is exploratory, single-seed validation replay with no training, accuracy
evaluation or official-test read. It does not overturn the earlier large T
effect at a different p90-trained ours checkpoint or qualify training at the
new betas.

[Report and plot](../results/eqprop-conv3-rms1-t8-replay-20260923-v1/analysis/report.md) ·
[Paired T effects](../results/eqprop-conv3-rms1-t8-replay-20260923-v1/analysis/T_effects.csv) ·
[Source/runtime pairing](../results/eqprop-conv3-rms1-t8-replay-20260923-v1/analysis/T_validation.json) ·
[Collection validation](../results/eqprop-conv3-rms1-t8-replay-20260923-v1/collection_validation.json) ·
[Plan](eqprop_conv3_rms1_t8_replay_plan_20260923.md).

## Conv3 gradient quality at output-displacement-one betas, September 23

Outcome: **mixed overall, with a clear noisy third-convolution advantage for
ours**. Injected beta 1.385 for ours and .02173 for legacy were calibrated to
output displacement approximately one at the shared initialization. Read-only
replay at those fixed values compares initialization and both existing p99
epoch-30 final checkpoints, T16/K8, float64 centered frozen-current EqProp,
clean reads and four paired endpoint-noise draws at sigma 5e-4. The same 36
validation batches contain 576 ordinary-MNIST examples. No optimizer step or
official-test read occurred.

Median noisy Conv3 EqProp–BPTT cosine is **.697435 ours versus .238445 legacy
at initialization**, and **.753430 versus .276865 at epoch 30**. Ours improves
on all 36 paired batches at each checkpoint. Median noise/clean gradient
ratios are 1.0221/4.0851 initially and .8517/3.4199 at epoch 30 (ours/legacy).
The corresponding noisy relative errors against BPTT are 1.0203/4.0859 and
.8604/3.4107. Ours reduces early-layer noise error too, but both first and
second convolutional gradients remain noise-dominated in all measurements,
with cosines near zero. Both readouts stay nearly perfectly aligned, with a
small advantage for legacy. Legacy also has smaller clean finite-beta error;
ours epoch-30 Conv3 clean cosine is .996677 median but .948477 minimum.

Initial centered output displacement is .999812 ours and .999881 legacy,
while ours has 6.70/5.15/3.92 times the hidden-layer phase signal. This supports
an upstream signal advantage under the initial output-displacement constraint.
At the trained checkpoints, output positive-free RMS is .212876 ours and
.558918 legacy: fixed initialization-calibrated beta no longer matches the
trained output excursion. The p99 parents trained at different original
betas (.987333678708 ours and .1 legacy), with inherited scheme-specific
learning rates and distinct learned weights. This replay does not establish
training stability or accuracy at the new beta choices, nor reverse the
earlier independently optimized comparison with unconstrained output motion.

All four production cells and eight CPU/GPU smoke bundles validate locally;
2,880 gradient comparisons, 2,304 passing residual checks, maximum selected
residual 2.82041e-7, and 119 matching remote file hashes. Source/checkpoint,
cohort, frozen-force, zero-bias, parameter and matched-noise guards pass.
Akib RTX3080 exited zero and released the GPU; production 344.60s, total
charged smoke plus replay 634.77s within 1,800s. Thirteen focused tests pass.
No failures, exclusions or missing cases. This is single-seed exploratory
validation evidence; batch/draw spreads are not independent-seed uncertainty.

[Report and plots](../results/eqprop-conv3-rms1-gradient-replay-20260923-v1/analysis/report.md) ·
[Layer statistics](../results/eqprop-conv3-rms1-gradient-replay-20260923-v1/analysis/layer_summary.csv) ·
[Paired effects](../results/eqprop-conv3-rms1-gradient-replay-20260923-v1/analysis/paired_effects.csv) ·
[Collection validation](../results/eqprop-conv3-rms1-gradient-replay-20260923-v1/collection_validation.json) ·
[Plan](eqprop_conv3_rms1_gradient_replay_plan_20260923.md).

## Conv3 initialization beta/T/K replay, September 23

Outcome: **negative for meaningful T/K dependence at initialization on the
tested grid**, supporting the checkpoint-specific interpretation of the older
ours result. Baseline, ours, and legacy share the exact saved seed-0 initializer,
injected beta `.987333678708` and `5.26875648112`, T8/16/32, K8/32, float64
centered frozen-current EqProp, zero biases, input gain 360, and the existing
576-example validation cohort. Clean endpoints and four matched draws at
sigma `5e-4` yield all 36 cells and 25,920 layer comparisons. This is read-only
selection/mechanism evidence; no optimizer step or official-test access occurs.

Across all schemes, betas, and layers, the largest noisy median-cosine change
is `5.87e-8` across T and `6.41e-6` across K; the largest clean change is
`2.49e-6`. A fixed K32 BPTT reference gives the same practical conclusion.
The maximum absolute paired-batch change is `2.35e-5`, averaging the four
noise draws within each noisy batch. Thus the pooled result does not conceal
a large effect in a few examples. These are descriptive batch comparisons,
not independent training-seed replications.

Ours' noisy readout stays `.999978` at beta `.987334` and `.999923` at beta
`5.268756` throughout the grid. At the latter beta, the older p90-trained
checkpoint instead changes `-.293154 → .820463` with T8→32 at K8; paired
readout improvement occurs on 35/36 batches at T16 and 36/36 at T32. The large
T/K sensitivity is therefore absent at initialization and appears later in
that trajectory. Its onset epoch and mechanism remain unmeasured. Historical
replay environments differ, although cohort bytes match exactly. Beta itself
still matters at initialization: ours' noisy Conv3 median rises
`.571775 → .960893` between these betas, while early layers remain noise-limited.
This neither freezes a training beta nor predicts accuracy at another T/K.

All 36 production and 18 local GPU smoke bundles validate; all 509 collected
file hashes and 606 frozen source/input files match. Fifi RTX 5090 exited 0
and released its worker after 1851.02 seconds of production; including smoke,
1927.27 seconds (0.5354 GPUh) is within the one-hour cap. Twelve focused tests
pass. No scientific case failed or was excluded; two prelaunch staging
corrections are preserved in the plan. Earlier partial analyses are superseded.

[Report](../results/eqprop-conv3-init-beta-tk-20260923-v1/analysis/report.md) ·
[Initialization versus trained](../results/eqprop-conv3-init-beta-tk-20260923-v1/analysis/ours_initialization_vs_trained.png) ·
[Paired effects](../results/eqprop-conv3-init-beta-tk-20260923-v1/analysis/effects.csv) ·
[Collection validation](../results/eqprop-conv3-init-beta-tk-20260923-v1/collection_validation.json) ·
[Beta notes](beta_study.md#initialization-check-at-the-same-replay-betas) ·
[Plan](eqprop_conv3_init_beta_tk_plan_20260923.md).

## CIFAR L8 amplification and KCL implementation audit, September23

Outcome: **positive** for the implemented blockwise circuit. Synthetic CPU
checks compare independent physical KCL currents with the energy derivatives
and coordinate updates in two- and three-convolution circuits, all three
production schemes plus a non-reciprocal control. CIFAR states and input/weight
gradients match the fully analog MNIST interaction/tracking-solver path
bit-for-bit on these small circuits. The separate analog classifier matches
the standard quadratic update and an independent branch-current balance.
The focused suite passes **136/136** checks, including 21 new architecture
checks. Nine relevant runtime files match all five frozen production source
snapshots byte-for-byte; no training implementation or checkpoints changed.

The architectural distinction is explicit: digital max-pool/affine-BN bridges
and voltage clamps isolate the analog circuits. Electrical feedback and
amplification do not extend across those boundaries. The isolated one-edge
classifier has no internal amplified edge, so A/B do not affect its solution
at fixed input/weights/gain. Matched-weight legacy blocks reproduce baseline
outputs scaled by 16/16/4; ideal BN cancels those factors. This supports a
mechanism hypothesis, not a causal attribution of the observed accuracy gap
or a full-width hardware validation. See the [audit and reproducible
checks](cifar_l8_amplification_audit_20260923.md).

Exploratory follow-up, September23: baseline and legacy with
block-output normalization and the complete baseline optimizer matched exactly
through200 real CIFAR training minibatches of32 examples. Maximum observed
relative error was0 for logits, loss, all trainable gradients, BN buffers,
parameters and projected updates. This validates the scaling identity through
Adam training over the tested window with production BN epsilon. All12 saved
checkpoint replays subsequently passed their solver audits. The first-block
epoch50 training BN epsilon share was73.77% for baseline,61.43% for proposed,
and2.30% for legacy. Existing reference and LR evidence are reused; redundant
reference repeats and the supplementary LR intervention were cancelled.

**BN follow-up is terminal, with limited no-BN evidence.** Five ten-epoch controls are locally
collected and validated, with passing final solver audits: voltage-normalized
legacy75.82% validation versus74.68% for original legacy (+1.14pp); baseline
with reciprocal epsilon70.18% versus75.26% (-5.08pp); frozen-affine legacy
73.02% versus74.68% (-1.66pp); frozen-affine baseline70.72% versus75.26%
(-4.54pp); frozen-affine proposed73.76% versus75.36% (-1.60pp).
All three frozen checkpoints confirm gamma1, beta0 and14070 BN
running-stat updates. The epsilon control overlaps with the
effective condition sampled by the cancelled legacy-baseline-LR prefix; those
five matching epochs are not independent evidence.

Baseline/no-BN and legacy/no-BN each stopped at the first epoch2 batch after
epoch1 validation10% and CE log(10). One exact read-only replay per failed
batch confirmed zero block outputs, zero logits and zero gradients for all13
parameter tensors, including float64 norm/nonzero-count checks. The failed
bundles and stopped checkpoints are local; no LR retry or ten-epoch
extrapolation is included. Their final higher-iteration audits were not reached.
Proposed/no-BN completed ten epochs at52.08%, but failed final solver
qualification: free-logit relative L2 difference between reference and
sentinel iterations was8.61% (limit1%). Its bundle and checkpoint are local
and internally valid; the accuracy is provisional at configured iterations,
not a qualified steady-state comparison. All seven new cases have terminal
artifacts and valid local bundles: four qualified ten-epoch results, two
early collapses, and one completed but solver-unqualified result. The separate
normalized-legacy control is also complete. No training remains active; obsolete
Riri/local queue items were cancelled without repeating outsourced cases.
New-study compute15.2119GPUh is within24GPUh, including checks and recovery.
All cases use the original selected optimizer,
augmentation, cross-entropy, seed and50-epoch cosine horizon.

Input-gain extraction from the original epoch0/10/30/50 checkpoints finds
negligible endpoint adaptation: all start100; the largest final block-gain
change is0.353%, and legacy's largest is0.095%. Gain Adam LR stays5e-5 initially
and decays to1e-6; endpoint next-step block-gain proposals round to zero in
float32 for all three schemes on the diagnostic cohort. This is a shared
parameterization/update limitation, not evidence of a legacy-specific gradient
failure. Input gain can change effective BN epsilon; the normalized-legacy
control tests a much larger fixed scale change than the gains learned. Later
input gains can also be absorbed into preceding trainable BN gamma/beta;
constant scalar gains do not imply constant downstream voltage scales.

The completed evidence supports a BN/voltage-scale interaction under the fixed
optimizers and a trainable-affine benefit for all three schemes at ten epochs.
Freezing affine parameters hurts baseline more, while proposed and legacy
lose similar amounts; learned affine BN is not supported as specifically
harming legacy. This does
not establish the full cause of the original fifty-epoch gap or general
BN-free trainability after retuning. No official CIFAR test split was read.

Trajectory analysis refines the normalization conclusion: normalized legacy
wins at4/10 measured epochs, averages-0.636pp versus its reference over6–10,
and has nearly unchanged epoch10 training CE. Its+1.14pp endpoint is promising
but does not establish consistent acceleration. Baseline with reduced epsilon
averages-6.268pp over6–10 and increases epoch10 training CE by0.2036.
Viewed in common voltage units, the four existing cases give a descriptive
epsilon-by-optimizer comparison: baseline-selected rates yield75.26/70.18%
at baseline/legacy effective epsilon, while legacy-selected rates yield
75.82/74.68%. The3.94pp difference of endpoint effects supports optimizer/BN
scale coupling at these settings; it is not an additional replicate or a
population interaction estimate. Frozen affine BN is worse at every measured
epoch for each scheme. Correlated epoch averages do not replace the declared
endpoint or measure seed uncertainty.

Proposed next work is a checkpoint-only BN-statistics comparison, continuation
of normalized legacy10→50 with the original optimizer/scheduler state, and a
fixed-versus-log block-gain pair in frozen-affine legacy. Suggested combined
cap16GPUh; no jobs or configs were launched for this proposal. Defer broad LR
sweeps and BN-free training pending a separate settling/scale diagnosis.
See the [conclusions and concrete proposal](cifar_l8_bn_conclusions_and_next_experiments_20260923.md).

Gradient-scale follow-up reuses the same128-example replay CSV. Across eight
convolution layers, median paired legacy/proposed gradient ratios at10/30/50
are6.34/6.51/8.63, while projected update/weight ratios are0.903/1.111/0.664.
Legacy's conductance norms at50 are roughly6–10 times smaller; scaling the
gradient norm by parameter norm reduces the median50 ratio to1.06. This is
consistent with a substantial parameter-scale contribution, not a proof that
the trained models are equivalent rescalings. Large raw gradients do not imply
uniformly oversized Adam steps. Noise, direction and functional sensitivity
remain unresolved; the proposed next probe measures loss/logits along actual
Adam directions. [Detailed measurements](cifar_l8_gradient_scale_interpretation_20260923.md).

See the [mechanism plan](cifar_l8_mechanism_plan_20260923.md),
[BN ablation plan](cifar_l8_bn_ablation_plan_20260923.md),
[comparison](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/report.md),
[coverage validation](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/collection_validation.json)
and [findings with gain values](cifar_l8_mechanism_findings_20260923.md).

## Conv3 noise versus free/nudged phase RMS, September 23

Outcome: **positive for phase amplitude as an explanation of measured gradient
noise sensitivity**. Local CPU aggregation joins the saved phase-voltage
measurements to actual noisy-minus-clean EqProp gradient RMS from 21 p99
T16/K8 replay cells (three checkpoints, seven betas), with nine K64 controls.
All use sigma5e-4, the same 36 validation batches and four noise draws. There
is no new model replay, training, checkpoint change, or official-test access.

Noise dominance is RMS(g_noisy-g_clean)/RMS(g_clean)>=1. The centered
phase signal is D=RMS((v_plus-v_minus)/2), with RMS noise sigma/sqrt(2) in
that contrast. Free-to-positive and free-to-negative RMS differences are
reported separately. Gradient acquisition error is separated from clean
finite-beta EqProp–BPTT error.

The noise-versus-D curves nearly overlap despite differing strongly versus
beta. Log-interpolated, bracketed median Conv3 crossovers are D=1.172e-3
for legacy epoch30, 1.068e-3 for ours epoch30, and 1.063e-3 for ours best27.
The corresponding estimated betas are .0745/1.179/.870. These are estimates
between sampled points at fixed sigma, not newly measured thresholds or a
noise-amplitude sweep. Ours' readout crossovers are D≈1.32–1.34e-3;
legacy's readout is signal-dominated throughout the tested grid.

At their training betas (.1 legacy, .987333678708 ours), median Conv3 D is
1.571e-3 versus 8.958e-4, and gradient noise/clean ratios are .746 versus
1.193. Noise dominates 16.0% versus 72.2% of batch/draw measurements.
Ours best27 is intermediate: D=1.204e-3, ratio .881, and 32.6% dominated.
At each method's best sampled Conv3 beta, legacy D=.014898 and ratio .0779;
ours30 D=.008564 and ratio .1211. Both are above the crossover, and finite-beta
fidelity remains a separate limitation.

The first two layers remain noise-dominated in median at every sampled beta;
no crossover is extrapolated for them. Even at beta10, noise/clean ratios
are 16.48/2.04 for legacy and 76.44/10.73 for ours30. Absolute free voltages
are much larger than their tiny phase contrasts, so free-state RMS alone is
not a useful criterion for gradient acquisition. At common beta.9873,
legacy's Conv3 phase signal is about 16.6 times larger than ours30's.

Interpretation: phase RMS is a better comparison coordinate for this noise
mechanism than numerical beta. An exactly matched phase-signal comparison
has not been run; one beta need not match all layers. The apparent common
Conv3 threshold is not universal: pre/post-state correlations, batch/spatial
averaging, finite-beta effects, and different trained weights remain relevant.
The checkpoints and betas are not newly qualified for training stability.

All 30 source bundles and the canonical aggregation bundle validate; 17,280
noisy parameter/batch/draw rows and 36 batch identities reconcile. Source
result hashes are unchanged. K64 changes median D by at most 2.32e-5
relatively and median noise ratio by at most 1.15e-4. All seven unbracketed
crossovers are labeled rather than extrapolated. Runtime is 6.01s on local
CPU within the 300s budget, with no failures or exclusions.

[Report and figures](../results/eqprop-conv3-p99-noise-rms-analysis-20260923-v1/analysis/report.md)
· [Crossover brackets](../results/eqprop-conv3-p99-noise-rms-analysis-20260923-v1/analysis/measurements/crossovers.csv)
· [Full RMS and gradient statistics](../results/eqprop-conv3-p99-noise-rms-analysis-20260923-v1/analysis/measurements/summary.csv)

## Conv3 p99 beta/K replay, September 23

Outcome: **negative for insufficient K as the explanation of the measured
gradient gap**. The requested read-only follow-up uses common T=16,
K=8/16/32/64, injected beta=.987333678708/3/10, legacy epoch30, ours epoch30,
and ours best epoch27. It preserves the preceding 576-example validation
cohort, clean control, sigma5e-4, four matched endpoint-noise draws, float64
centered frozen-current EqProp, and exact-zero biases. No training or
accuracy evaluation is performed.

Across all three checkpoints and betas, increasing K8 to64 changes every
noisy layerwise median cosine by less than **9.62e-7**. The largest clean
EqProp gradient relative differences are 1.83e-6 (legacy), 1.58e-4 (ours30),
and 1.19e-3 (ours27); by K16 these fall below 6.75e-8. Every K32 clean
EqProp gradient equals K64 exactly. BPTT K32-to64 relative error is below
2.6e-23. Comparisons against fixed K64 BPTT give the same conclusion,
separating actual convergence from changes in the reference unroll.

At each method's independently best sampled beta, noisy Conv3 median cosine
at K64 is **legacy .977483 at beta.987333678708**, **ours30 .967538 at beta10**,
and **ours27 .955650 at beta10**. The best beta is unchanged across K. For the
minimum of all four layerwise medians, every checkpoint selects beta10;
legacy scores .049025, ours30 .005245, and ours27 -.003270. Ours still wins
Conv3/readout at common beta10 but loses the first two layers. These are
best tested gradient scores, not global beta optima or training accuracy.

All 36 production cells, 25,920 comparisons, and 12 local smoke cases
validate. All 20,736 residual records pass. The maximum nudged residual
falls from 4.75e-5 at K8 to 1.09e-11 at K32/K64 without a material change
in noisy alignment. All nine K8 cells reproduce the preceding replay's
checked metrics exactly. Fifi exited0 and released its GPU; all 509 remote
output hashes match locally. Production took 2,200.07s; charged smoke plus
production/allowance was 2,265.25s within the 7,200s cap. One sandbox-blocked
local smoke is preserved and excluded; no scientific cells failed or are
missing. Twelve focused regression tests passed.

Interpretation: numerical K convergence is already adequate at these frozen
p99 weights, including beta10. Endpoint noise and finite-beta error remain
the relevant measured tradeoff. This does not overturn the earlier strong
T effect at the different p90-trained checkpoint, certify a new training
operating point, or establish the best beta/K for a new training trajectory.
The grid stops at beta10, uses one parent seed, and retains the parents'
different weights and learning-rate histories. Official-test access and
optimizer steps remain zero.

[Report and plots](../results/eqprop-conv3-p99-beta-k-replay-20260923-v1/analysis/report.md)
· [K effects](../results/eqprop-conv3-p99-beta-k-replay-20260923-v1/analysis/k_effects.csv)
· [Validation](../results/eqprop-conv3-p99-beta-k-replay-20260923-v1/collection_validation.json)

## Conv3 p99 trained-checkpoint beta replay, September 23

Outcome: **mixed**. The seven-point injected-beta replay supports a partial
noise-sensitivity explanation for legacy's validation advantage. Clean
gradients agree closely with BPTT at each training beta, and doubling T has
negligible effect. Legacy's third convolution retains better noisy alignment;
both schemes' first two convolutions remain strongly noise-dominated.

At T8/K8, sigma5e-4, and each checkpoint's training beta:

| Checkpoint | Training beta | Noisy Conv1/Conv2/Conv3/readout median cosine | Conv3 added noise / BPTT norm |
|---|---:|---|---:|
| Legacy final/best epoch30 | .1 | .00520 / .00452 / **.79946** / .999999 | .7411 |
| Ours final epoch30 | .987333678708 | -.00564 / .00884 / **.63167** / .999948 | 1.1969 |
| Ours best epoch27 | .987333678708 | -.02262 / .01020 / **.74381** / .999950 | .8755 |

Minimum clean layer-median cosines are .99836/.99785/.99651. Ours at beta3
improves noisy Conv3 cosine to .91897 at epoch30 and .94487 at epoch27,
without repairing the first two layers. At beta10, the best sampled
worst-layer medians are only .04903/.00525/-.00327, all upper-bound optima;
no tested beta provides an all-layer .90 median window. Legacy's larger-beta
noisy Conv3/readout cosines fall to .76274/.74043 at10; the clean controls
show the same decline.
Maximum T8→T16 median change is8.77e-6 clean and8.57e-7 noisy. All24,192
residual records pass, with largest individual selected residual .000130872
below the .01 threshold. These p99 checkpoints do not reproduce the earlier
p90 checkpoint's free-phase settling limitation. This is a checkpoint-specific
conclusion: the earlier p90-trained ours epoch30 checkpoint at beta5.26875648112
changed noisy readout cosine from-.293154 to.684545 when T8→T16 at K8,
whereas the current p99-trained checkpoint stays at.999948. The earlier
checkpoint also failed the small-beta clean/free-phase control at T8.
Different trained weights change the unnudged dynamics; varying replay beta
does not reproduce that difference. T8 sufficiency is not established for
other ours checkpoints or throughout training. See the
[earlier T/K comparison](../results/eqprop-conv3-cosine-tk-schemes-20260922-v1/analysis/report.md).

The frozen scope is three saved checkpoints, beta
[.001,.01,.1,.3,.987333678708,3,10], T8/16 and K8, all36 original batches
of16 ordinary-MNIST validation examples and four matched read-noise draws
plus clean controls. Float64 centered frozen-current EqProp, explicit perfect
diodes, input gain360, exact-zero biases and unchanged source weights are
preserved. BPTT differentiates eight zero-nudge steps from the same post-T
state. There are no optimizer steps, new accuracy evaluations or official-test
reads. The source validation results remain legacy96.86%, ours final96.04%
and best96.32%; the gap already existed before ours' final .28-point drop.

The interpretation is conditional on one trained seed, different learned
weights, amplification, training betas and scheme-specific learning rates.
Ours' roughly3x convolution and9x readout rates are a causal confound, not
proof of excessive relative updates. Snapshot replay cannot reconstruct
Adam's accumulated moments or establish training recovery. A matched training
comparison of ours' current beta and beta3 is the most direct next test of
this hypothesis; no new training or automatic range extension was launched.

All42 production cells/30,240 gradient comparisons and24 GPU smoke bundles
validate locally;593 remote file hashes match,177 runtime/four analyzer
source files and checkpoint bytes are unchanged. Fifi RTX5090 exited0 and
released the worker. Production took1560.60s; charged1748.92s, including
smoke and100s for the stopped CPU attempt, is below7200s. The initial missing-
dependency import and interrupted CPU smoke (one complete, one failed cell)
are retained/excluded. There are no missing or excluded scientific cases.

[Report and plots](../results/eqprop-conv3-p99-trained-beta-replay-20260923-v1/analysis/report.md)
· [Layer statistics](../results/eqprop-conv3-p99-trained-beta-replay-20260923-v1/analysis/layer_summary.csv)
· [Validation](../results/eqprop-conv3-p99-trained-beta-replay-20260923-v1/collection_validation.json)
· [Plan](eqprop_conv3_p99_trained_beta_replay_plan_20260923.md).

## Legacy CIFAR L8 ten-epoch LR refinement, September23

Outcome: **negative for improving the current rates with these four local
perturbations**. All five seed0 ten-epoch evaluations completed before the
02:00CEST deadline. Current rates remain selected: validation **74.68%**,
CE0.717528. Classifier half/double gives74.44%/74.34%; convolution half/double
gives74.26%/70.80%. Every alternative has higher CE. The fresh center matches
the earlier ten-epoch74.68% result. Retain the existing convolution LR vector
and classifier0.00016, BN0.001 and gains0.00005.

Matched saved initialization,45k/5k split, seed0, batch32, legacy amplification
4/0.25, analog L8, trainable BN/gains, Adam, cosine horizon50, T/K[6,6,4],
augmentation, bounds and loss are fixed. Only conv or classifier LR changes
by a factor of two. Rank by terminal epoch10 validation accuracy, then CE.
All five production and seven smoke bundles are collected and validate;
14070 Adam steps each, all final solver audits pass, both lane exits0.
No failures, exclusions, missing cases or deadline cancellations; no official
test reads. Production11.3943GPUh within budget. No further runs launched.

User-authorized parallel placement uses matching RTX5090/software on Fifi
(center/conv probes) and Riri (classifier probes). One seed, host split and
validation selection limit inference; the nearest gap0.24pp is12 examples.
Joint LR interactions and longer training are not resolved. This result does
not establish inherent legacy-scheme underperformance or a global LR optimum.

[Report](../results/cifar10-l8-legacy-lr-refinement-e10-seed0-20260922-v1/analysis/report.md)
· [Comparison](../results/cifar10-l8-legacy-lr-refinement-e10-seed0-20260922-v1/analysis/comparison.csv)
· [Verification](../results/cifar10-l8-legacy-lr-refinement-e10-seed0-20260922-v1/analysis/verification.json)
· [Plan](cifar_l8_legacy_lr_refinement_plan_20260922.md).

## Conv3 ours p99 at T16/K8 for ten epochs, September 23

Outcome: **mixed**. The requested seed0 run completed ten finite epochs at
injected beta0.987333678708 and endpoint read noise sigma5e-4. Doubling T
from8 to16 gave almost identical ten-epoch validation trajectories in this
comparison: the maximum absolute epochwise difference is0.12pp.

| T | K | Best validation, epochs1–10 | Epoch10 validation | Best epoch |
|---:|---:|---:|---:|---:|
| 8, existing reference | 8 | 95.64% | 95.44% | 9 |
| 16, requested run | 8 | 95.58% | 95.38% | 9 |

The final difference is-0.06pp; both finish0.20pp below their ten-epoch best.
The T8 values are the first ten epochs of the existing completed30-epoch p99
run. Both use the same frozen source on Tesla V100-SXM2-16GB with PyTorch2.5.0
and CUDA12.2. Initial parameter hashes, train/validation split and all ten
minibatch-order hashes match. The inherited Adam vector, beta, noise seeds,
batch sizes, input gain360, explicit perfect diodes, float64 centered
frozen-current EqProp, [0,100] weights and frozen exact-zero biases are
unchanged. Scientific config differences are T and terminal epoch horizon;
learning rates are constant. Official-test reads are zero.

This is exploratory ordinary-MNIST training/validation evidence. The inherited
p99 beta was calibrated at T8/K8; p99 is a clean cosine threshold, not a
stability probability. T16 changes both training free phases and validation
inference. This single-seed result supports ten-epoch nondivergence at the
requested setting, with no observed validation gain. It neither establishes
thirty-epoch stability nor answers whether higher T rescues larger-beta
failures.

Jean Zay jobs73267/73276 completed0:0 as two five-epoch continuation segments.
The second restored the exact epoch5 optimizer/RNG/noise state. Total GPU
allocation time is6,539seconds (1.8164GPUh), within the3GPUh budget. The local
same-config and continuation smokes pass, as do17 exact-run and5 continuation
tests. The new production bundle, existing T8 reference and both smoke
bundles validate locally. Remote file checksums match the collected copy;
the dry-run's sole difference is the local study directory's timestamp.
The expected275,040 endpoint-noise draws are recorded. No failures, retries,
excluded scientific cases or pending jobs remain.

[Report and trajectories](../results/eqprop-conv3-ours-p99-t16k8-10ep-20260922-v1/analysis/report.md)
· [Epoch measurements](../results/eqprop-conv3-ours-p99-t16k8-10ep-20260922-v1/analysis/epochs.csv)
· [Validation](../results/eqprop-conv3-ours-p99-t16k8-10ep-20260922-v1/collection_validation.json)
· [Config](../configs/conv/eqprop_conv3_ours_p99_t16k8_10ep_20260922_v1/conv3_ours_p99_sigma5em4_t16k8_10ep_seed0.json)
· [Plan](eqprop_conv3_ours_p99_t16k8_plan_20260922.md).

## Conv3 T/K cosine interaction across amplification schemes, September 22

Outcome: **mixed**. A complete T/K grid at each trained checkpoint's original
beta shows essentially unchanged baseline and legacy cosines, while ours
has strong T/K interaction and opposing effects across layers. Increasing
both phase lengths does not recover legacy's alignment at this fixed beta.

The grid is T=8/16/32 × K=8/16/32 for baseline, ours and legacy at training/read
sigma 5e-4. Maximum range in any noisy layer median over the nine points is
1.61e-4 for baseline and 6.55e-6 for legacy; their largest clean ranges are
2.02e-5 and 1.84e-4. Ours has the following noisy readout medians:

| T | K=8 | K=16 | K=32 |
|---:|---:|---:|---:|
| 8 | -0.29315 | -0.48089 | -0.67259 |
| 16 | 0.68455 | 0.73575 | 0.80633 |
| 32 | 0.82046 | 0.83396 | 0.85758 |

At T8, larger K worsens the readout; at T16 and T32 it improves it. This is
not a uniform layerwise improvement. For ours at T32, K8→K32 changes noisy
Conv1/Conv2/Conv3/readout medians from [.06055,.07781,.30246,.82046] to
[.01057,.14756,.20515,.85758]. Clean Conv1 falls from .96967 to .47268.
Clean/noisy readout medians differ by at most 1.41e-6 on this grid, so this
interaction persists without endpoint read noise. No single K maximizes all
four layer medians at T32.

The three final epoch-30, seed-0 A100 checkpoints remain frozen, with injected
betas baseline404.141105702, ours5.26875648112, legacy4.42250110273. The same
576 validation examples, four matched noise draws, float64 centered
frozen-current EP, exact-zero biases, explicit perfect diodes, input gain360
and paired20-output squared loss are used. At each point both EP and BPTT use
the requested K from the same post-T state. K therefore changes both
estimators; this grid does not isolate finite-beta EP error from changes in
the finite-unroll BPTT reference, identify an exact minimum T/K, or establish
training recovery. The result is conditional on these trained weights and
betas, with one trained seed per scheme. No training, accuracy evaluation or
official-test read occurs; residuals do not filter or select the points.

All 27 grid cells are included: 22 new plus five matched cached anchors. All
44 new production/smoke bundles and five cached bundles validate locally;
all664 output hashes match Akib and all12 checkpoint source files remain
unchanged. All324 scheme/T/batch groups have identical post-T state and frozen
force hashes across K; all972 batch replays verify both requested minimizer
lengths. Coverage is19,440 layer comparisons (15,840 new), with no failures,
exclusions or missing points. Akib exited0 and released the worker. Replay
including smokes took2,400.39s (0.667GPUh), total wall time2,462s, within the
5,400s cap. No further run is pending.

[Noisy cosine grid](../results/eqprop-conv3-cosine-tk-schemes-20260922-v1/analysis/noisy_cosine_TK.png)
· [Clean cosine grid](../results/eqprop-conv3-cosine-tk-schemes-20260922-v1/analysis/clean_cosine_TK.png)
· [Report and fixed-T/K curves](../results/eqprop-conv3-cosine-tk-schemes-20260922-v1/analysis/report.md)
· [Measurements](../results/eqprop-conv3-cosine-tk-schemes-20260922-v1/analysis/layer_summary.csv)
· [Collection validation](../results/eqprop-conv3-cosine-tk-schemes-20260922-v1/collection_validation.json)
· [Plan](eqprop_conv3_cosine_tk_schemes_plan_20260922.md).

## Conv3 readout cosine transition by T12, September 22

Outcome: **positive for localizing the observed median sign change**, with
continued T dependence beyond it. Filip requested exactly T=12/16/20/32 for
the final epoch-30 ours/sigma5e-4 checkpoint, keeping the training injected
beta 5.26875648112 and K=8. The cached T8/T64 cells supply matched anchors.

| T | Conv1 | Conv2 | Conv3 | Readout | Positive readout comparisons |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.02007 | 0.02578 | 0.24569 | -0.29315 | 44.4% |
| 12 | 0.02726 | 0.03409 | 0.56056 | 0.61105 | 94.4% |
| 16 | 0.03308 | 0.04468 | 0.43053 | 0.68455 | 97.2% |
| 20 | 0.04147 | 0.05296 | 0.36083 | 0.73593 | 100.0% |
| 32 | 0.06055 | 0.07781 | 0.30246 | 0.82046 | 100.0% |
| 64 | 0.08031 | 0.11167 | 0.28897 | 0.88544 | 100.0% |

Layer values are median noisy EP–BPTT cosines over the same 36 validation
batches and four matched noise draws. T12 is the first sampled positive
readout median; T9–11 are unmeasured. T20 is the first sampled value where
all 144 readout comparisons are positive. The readout still changes by
0.06498 from T32 to T64. Clean and noisy readout medians nearly coincide,
so the T dependence is also present without endpoint read noise. Conv3
changes non-monotonically, peaking among these sampled values at T12 and
decreasing afterward. Early layers remain poorly aligned under read noise.

The single seed-0 A100-trained checkpoint, beta, K, cohort, noise seeds,
float64 centered frozen-current estimator, explicit perfect diodes,
exact-zero biases, input gain 360 and paired 20-output loss remain fixed.
BPTT uses K zero-nudge steps from the post-T state at each sampled T. These
are exploratory fixed-checkpoint measurements, not a training result or an
exact minimum-T claim. No residual threshold filters the points, and no
training, accuracy evaluation or official-test read occurs.

All four production and four smoke bundles validate locally; all 136 output
file hashes match Akib and all four source checkpoint files remain unchanged.
The unchanged Akib numerical environment also produced both cached anchors.
The launcher exited 0, the worker released the GPU, and total wall time was
394 seconds against a 900-second cap (363.18 seconds of replay including
smokes). No requested point is missing, failed or excluded; no further T
sweep is pending.

[Cosine plot](../results/eqprop-conv3-cosine-t-transition-20260922-v1/analysis/cosine_vs_T.png)
· [Report](../results/eqprop-conv3-cosine-t-transition-20260922-v1/analysis/report.md)
· [Measurements](../results/eqprop-conv3-cosine-t-transition-20260922-v1/analysis/cosine_vs_T.csv)
· [Collection validation](../results/eqprop-conv3-cosine-t-transition-20260922-v1/collection_validation.json)
· [Plan](eqprop_conv3_cosine_t_transition_plan_20260922.md).

## Conv3 noisy cosine changes from T8 to T64, September 22

Outcome: **mixed**. Increasing free-phase T from 8 to 64 substantially changes
cosine similarity for ours at training/read sigma 5e-4. Legacy at both noise
levels, baseline at both noise levels, and ours at sigma 3e-4 remain almost
unchanged. The scientific objective is cosine change, as explicitly clarified
by Filip; no residual threshold filters or selects these points.

At the original training beta, ours/sigma 5e-4 changes as follows:

| T | Conv1 | Conv2 | Conv3 | Readout |
|---:|---:|---:|---:|---:|
| 8 | 0.02007 | 0.02578 | 0.24569 | -0.29315 |
| 64 | 0.08031 | 0.11167 | 0.28897 | 0.88544 |

These are noisy median EP–BPTT cosines over 36 batches and four read draws.
Across all four layers and the five matched beta values, the largest absolute
median change is 1.17860 for this checkpoint. Legacy changes by at most
3.48e-8/2.26e-8 at sigma 3e-4/5e-4; its output-side loss of alignment at larger
beta persists. Baseline changes by at most 1.61e-4/1.02e-4, and ours/sigma 3e-4
by at most 3.21e-6. Thus the legacy pattern survives this higher-T check;
the pronounced T-sensitive readout discrepancy belongs to ours/sigma 5e-4.
The latter still has poor noisy early-layer alignment, and its Conv3 cosine
remains low at the training beta. Increasing T does not resolve every layer.

This is a read-only, same-cohort mechanism diagnostic on six final epoch-30,
seed-0 p90 Conv3 checkpoints trained on A100s. Baseline/ours/legacy each use
training/read sigma 3e-4 and 5e-4. The narrowed beta factors are
`[.1,.3,1,3,10]` relative to each source training beta; the cached T8 data are
restricted to these same factors. K=8, float64 centered frozen-current EqProp,
explicit perfect diodes, exact-zero biases, input gain 360, original [0,100]
weights and 576 validation examples remain fixed. BPTT differentiates K
zero-nudge steps from the same post-T state. Cohort SHA-256 is
`95af3eebd9416ec643b7d698d061b7dac3fe0d52311dd888293609bada629acd`.
Clean controls and four matched noisy endpoint draws are included. No
training, accuracy evaluation or official-test read occurs.

The user authorized parallel placement: all schemes at factors .1/3 ran on
Trex, .3/10 on Loulou, and 1 on Akib. Common smoke comparisons bound cross-host
cosine disagreement by 2.23e-10; production BPTT norms agree within 2.46e-14
relative, with numerical rather than bitwise agreement across environments.
All 30 production and 54 smoke bundles validate locally, all 1,194 output
hashes match the remote files, all 24 checkpoint source files remain unchanged,
and all launchers exited 0. Runtime including smoke was 0.516 GPUh against a
1.5 GPUh cap. Two preflight MPS admission failures occurred before scientific
computation; logs are retained and no scientific case is excluded or missing.

Limitations: one trained seed per condition, five sampled betas, fixed K, and
two T values in the noisy comparison. This does not establish the minimum T
for a cosine threshold, continuous-beta coverage, exact gradients, or improved
training at a changed beta. The clean small-beta T study below is separate.

[T8/T64 overlay](../results/eqprop-conv3-trained-beta-noise-t64-20260922-v1/analysis/cosine_T8_vs_T64.png)
· [Report and clean/noisy curves](../results/eqprop-conv3-trained-beta-noise-t64-20260922-v1/analysis/report.md)
· [Matched cosine changes](../results/eqprop-conv3-trained-beta-noise-t64-20260922-v1/analysis/cosine_change_with_T.csv)
· [Collection validation](../results/eqprop-conv3-trained-beta-noise-t64-20260922-v1/collection_validation.json)
· [Config](../configs/conv/eqprop_conv3_trained_beta_noise_t64_20260922.json)
· [Plan](eqprop_conv3_trained_beta_noise_t64_plan_20260922.md).

## Conv3 clean cosine changes with free-phase T, September 22

Outcome: **positive for repairing the clean directional comparison by increasing
T** at the final epoch-30 ours/sigma5e-4 checkpoint. On the same 576 validation
examples, at injected beta0.000526875648112 and fixed K=8, layer median clean
EP–BPTT cosines are:

| T | Conv1 | Conv2 | Conv3 | Readout |
|---:|---:|---:|---:|---:|
| 8 | 0.73785 | 0.76034 | 0.38812 | 0.00705 |
| 32 | 0.99595 | 0.99820 | 0.99986 | 0.99972 |
| 41 | 0.99598 | 0.99903 | 0.99994 | 0.99987 |
| 64 | 0.99634 | 0.99930 | 0.99998 | 0.99996 |

The medians change little beyond T32. This does not identify the minimum T
for a chosen cosine threshold: full-cohort cosine was not sampled between8
and32. Minimum individual layer/batch cosines at32/41/64 are0.98586/0.98557/
0.98324, so improvement is not monotonic for every comparison. These are
clean, small-beta measurements; the subsequent T64 noisy-beta sweep addresses
Filip's clarified question about cosine changes under read noise.

Frozen checkpoint/float64 runtime, zero biases, perfect diodes and validation
cohort are unchanged. Four production and four smoke bundles validate locally,
with39 exact direct-state equivalence checks. Both training and official-test
access are absent. Runtime674.32s including smoke is below1,200s. An initial
launcher failed before computation because the new config lacked the frozen
runtime path; its log and initial payload are retained and the corrected retry
exited0. The initial integer-T residual trace is retained as ancillary evidence;
Filip explicitly clarified that residual thresholds are not his decision metric.

[Report and cosine plot](../results/eqprop-conv3-trained-minimum-t-20260922-v1/analysis/report.md)
· [Measurements](../results/eqprop-conv3-trained-minimum-t-20260922-v1/analysis/clean_cosine_vs_T.csv)
· [Validation](../results/eqprop-conv3-trained-minimum-t-20260922-v1/collection_validation.json).

## Conv3 trained-checkpoint beta sweep under read noise, September 22

Outcome: **mixed**. A read-only sweep of thirteen beta values on six final
epoch-30 p90 checkpoints supports the proposed conflict for legacy: small
beta preserves clean gradient direction but leaves early layers noise-dominated;
larger beta produces large, misdirected gradients near the output. No sampled
shared beta gives legacy four layer median noisy cosines above 0.90. Baseline
has a useful measured overlap; ours/sigma 3e-4 is intermediate. Ours/sigma 5e-4
also fails its native-T8 clean control and requires a settling qualification.

Best shared beta maximizes the minimum of four layer median noisy cosines
over the sampled grid. Beta below is actual injected B, not base beta.

| Training/read sigma | Scheme | Best sampled B | Worst layer median cosine | Residual qualification |
|---:|---|---:|---:|---|
| 3e-4 | baseline | 12124.23317 | 0.97090 | Free T8 fails |
| 3e-4 | ours | 526.87565 | 0.62706 | Pass; upper grid boundary |
| 3e-4 | legacy | 4.42250 | 0.02010 | Pass |
| 5e-4 | baseline | 12124.23317 | 0.96245 | Free T8 fails |
| 5e-4 | ours | 0.000526876 | -0.00496 | Fails clean/settling control; lower boundary |
| 5e-4 | legacy | 0.44225 | 0.01247 | Pass |

Legacy's minimum clean layer median cosine at the smallest beta is
0.999999985/0.999993377 for sigma 3e-4/5e-4. For sigma 3e-4, at one tenth of
training beta its clean Conv1/Conv2 cosines remain 0.9758/0.9728 while Conv3 is
0.5029. At training beta its clean Conv3/readout cosines are 0.0536/0.0977,
with EP/BPTT norm ratios 10.01/10.93. This supports finite-beta distortion as
the late-layer limitation even when the readout is clean. It does not prove
that every continuous beta fails or establish training recovery at another beta.

The six seed-0 source checkpoints were trained on A100s. Included replays all
use one Akib RTX3080, with unchanged final weights, exact-zero biases, explicit
perfect diodes, input gain360, float64 centered frozen-current EqProp and
T=K=8. Beta factors span1e-4 to100 relative to each training beta. The same
36 validation batches of16 (576 examples; cohort SHA-256
`95af3eebd9416ec643b7d698d061b7dac3fe0d52311dd888293609bada629acd`)
are replayed at clean endpoints and four matched noisy draws per beta. BPTT
differentiates K zero-nudge steps from the same post-T state. Physical
relaxation and clamped input remain clean. No training, accuracy evaluation
or official-test read occurs. Medians/ranges describe batches and noise draws,
not uncertainty across training seeds.

Baseline's free-state residuals fail at Layer_1 and Layer_3 throughout this
grid; its zero and nudged endpoint checks pass. Its measured alignment is
therefore a finite-T/K diagnostic without full equilibrium qualification.
For ours/sigma 5e-4, a separate clean check on the first 48 examples at
T=8/32/128/512, K=8 and the minimum beta shows that longer free settling
helps: at T128 all 12 layer/batch cosines exceed0.99623 and residuals pass,
versus minimum 0.67334 at T8 on exactly that subset. Norm ratios at T128
remain 0.882–1.070. This supports a settling contribution, without establishing
exact gradients or full-cohort/K convergence. A subsequent solver-qualified
noisy sweep is needed before interpreting that checkpoint solely as a
read-noise/beta tradeoff; none is launched in this closeout.

All 78 main bundles (56160 comparisons),18 Akib smoke bundles and four settling
bundles validate locally; all 1392 collected file hashes match Akib. The24
checkpoint source files,177 runtime files and four analyzers are unchanged.
All 4320 training-beta comparisons exactly reproduce prior RTX3090 cosine,
BPTT norm and EP norm results. Both Akib jobs exited0 and released the GPU.
Total charged runtime 6811.13s (1.892GPUh) is below the10800s cap. The27
successful Loulou smoke bundles, failed noise-identity smoke and stopped
production attempt after unrelated GPU occupancy are preserved and excluded;
their local bundles validate. No main scientific case is missing or excluded.

[Report and curves](../results/eqprop-conv3-trained-beta-noise-20260922-v1/analysis/report.md)
· [Layerwise measurements](../results/eqprop-conv3-trained-beta-noise-20260922-v1/analysis/layer_summary.csv)
· [Collection validation](../results/eqprop-conv3-trained-beta-noise-20260922-v1/collection_validation.json)
· [Config](../configs/conv/eqprop_conv3_trained_beta_noise_20260922.json)
· [Plan and recovery](eqprop_conv3_trained_beta_noise_plan_20260922.md).

## CIFAR L8 proposed continuation to 50 epochs, September 22

Outcome: **positive for continued optimization of this selected seed-0 run**.
The user-requested 20 additional epochs on Fifi restore the final epoch-30
model, Adam state and original 50-epoch cosine schedule. Epoch 50 reaches
**89.92% validation accuracy**, versus 87.20% at 30 (**+2.72pp**); cross-entropy
changes from 0.393454 to 0.372177. Epoch 50 is the best accuracy within 31–50.

Fully analog L8, proposed voltage/current amplification 4/1, trainable BN/gains,
batch 32, unchanged selected learning-rate vector, T/K [6,6,4], crop/flip and
45k/5k split remain fixed. All 20 new epochs and 70,350 cumulative Adam steps
validate, including final checkpoint and scheduler state; final solver audit
passes. Full production/smoke artifacts are local and valid, exit 0, no failed
or excluded cases/retries. Runtime 4.2140 GPUh, Fifi released. No official test
read. This is one selected seed and validation evidence; only proposed was
extended, so the gain does not establish superiority over other schemes.
No additional experiment is launched as part of this closeout.

[Report and curves](../results/cifar10-l8-analog-ours-e30-e50-seed0-20260922-v1/analysis/report.md)
· [Validation](../results/cifar10-l8-analog-ours-e30-e50-seed0-20260922-v1/collection_validation.json)
· [Plan](cifar_l8_ours_e30_e50_plan_20260922.md).

## Conv3 beta choices for unit output displacement, September 22

Outcome: **positive for matching the pooled output displacement at the saved
initialization**. Direct replay of the user-selected injected betas
baseline `88.7`, ours `1.385`, and legacy `.02173` confirms output
`RMS(v_plus - v_free)` of `1.0003009717`, `.9998121959`, and `.9998813740`,
respectively. Deviations from the target of one are `+.030097%`, `-.018780%`,
and `-.011863%`; the worst absolute error is below `.031%`. Negative-nudge
and centered-half-difference RMS agree at the displayed six-decimal precision.
This supports the interpolation-based choices for this initializer.

The exploratory diagnostic uses the identical saved seed-0 initialization
SHA-256 `5e5782bd9bf166b8a432cbf283d745ffe4d25fe4a69ed656f14f9e7e9407392f`
and exact prior 36 validation batches of 16 (576 examples), with cohort hash
`95af3eebd9416ec643b7d698d061b7dac3fe0d52311dd888293609bada629acd`.
Conv3 T=K=8, float64 centered frozen-current EqProp, input gain360, wide
[0,100] weights, explicit perfect-diode dictionaries, exact-zero biases and
clean endpoints are unchanged. Beta is the actual injected strength;
base beta is divided by `(voltage_amp/current_amp)^3`. No optimizer step,
accuracy evaluation or official-test access occurred.

All three production bundles and three one-batch smoke bundles validate
locally: 108 replay batches, 432 layer-signal records, and zero failures
among 1,728 projected-KKT residual checks. Input hashes, source hashes,
parameter immutability and cohort checks pass. Independent aggregation of
the raw displacement CSV reproduces the output RMS to within `1e-12`.
The authoritative local archive matches Akib's SHA-256
`e3c651e8ec69096478a6f70f6e7f60f6e361778449e0b9294287e77113ea73c9`.

All cases ran on Akib's idle RTX3080. An initial smoke OOM prevented
production; its failed bundle is retained and validates as failed. The
replacement used expandable allocator segments and a 256-MiB cuDNN workspace
cap, preserving the scientific configuration and batch size. Successful
smoke plus production took199.47s, within the900s budget. No scientific
case was excluded. The original attempt is under `collected/workspace/output/`;
the successful replacement is under `collected/workspace/output-retry/`.

The match is a pooled initial-state measurement, not a claim about every
example or later training states. Per-batch output RMS ranges are
baseline `.996891–1.005512`, ours `.995796–1.005771`, and legacy
`.987583–1.013621`. Equal output displacement does not establish equal
hidden-layer signals or stability under noisy training; this diagnostic
does not replace those separate qualifications.

[Report](../results/eqprop-conv3-output-rms1-init-20260922-v1/analysis/report.md) ·
[Layer measurements](../results/eqprop-conv3-output-rms1-init-20260922-v1/analysis/layer_summary.csv) ·
[Coverage validation](../results/eqprop-conv3-output-rms1-init-20260922-v1/analysis/validation.json) ·
[Config](../configs/conv/eqprop_conv3_output_rms1_init_20260922.json) ·
[Run and recovery plan](../results/eqprop-conv3-output-rms1-init-20260922-v1/plan.md).

## Conv3 initialization beta and read-noise replay, September 21

Outcome: **mixed**. The user-requested fixed-initialization diagnostic is
complete: baseline/legacy/ours at injected beta B=.01/.1, with no training,
trained-checkpoint loading, accuracy evaluation or official-test access.
All six production bundles and six one-batch smoke bundles validate locally.
The scientific summaries contain exactly6048 defined layer comparisons:
6 cases ×36 matched validation batches ×7 read-noise levels ×4 weight matrices.
Runtime including smoke was633.78s (10.56min) on one local RTX3090, within the
one-GPU-hour cap. No failures or replacement attempts occurred; operational
smokes are retained and excluded from scientific aggregation.

The shared saved seed-0 initializer,576 ordinary-MNIST validation examples,
T=K=8, float64 centered frozen-current EP, same-post-T K-step BPTT, perfect
diodes, input gain360, wide[0,100] weights and exact-zero biases match the
previous diagnostic contract. The initializer is loaded in its native
float32 and promoted exactly to float64. Beta denotes injected B, with base
beta=B/(v/c)^3. The previous noise grid is retained by stated assumption:
sigma0,1e-5,3e-5,1e-4,3e-4,5e-4,1e-3. One readout draw per batch/sigma uses
seed2026092101; standard normals are matched across schemes/betas and reused
across sigma, while phases/layers remain independent. Only copied endpoint
voltages are perturbed. All initializer/parameter/cohort/noise guards pass,
BPTT references are bitwise invariant across beta, and all3456 projected-KKT
batch/layer/phase residual rows pass. The worst residual p90 is6.98e-6.

Clean EP–BPTT cosine is>.99 for every case, batch and matrix; the minimum is
.991160714 for legacy B=.1. At the smallest nonzero sigma1e-5, all first/second
Conv-layer median cosines are already below.23. This grid therefore does not
resolve their transition from high-fidelity to noise-limited readout. Small
negative medians near zero should not be interpreted as a reproducible
anti-gradient mechanism from one draw per batch.

The third convolution shows a clear separation. At sigma1e-5 and B=.01,
baseline/legacy/ours median cosines are.018007/.984603/.335984; at B=.1 they
are.163659/.999695/.962394. At sigma1e-4 and B=.1 they are
.018008/.984408/.335767. Dense readout is more robust: at sigma1e-3 and B=.1,
baseline/legacy/ours are.863322/.999999/.999543. Increasing beta by10 gives
approximately a decade of noise tolerance in these measured curves, with
small finite-beta deviations; this observation applies to these two betas.

These fixed-weight results show no general noisy-gradient advantage for ours
over legacy under equal injected B. They also demonstrate why the previous
large-beta baseline result cannot be transferred to B=.01/.1. Equal B does
not match physical response: at B=.1 the pooled centered output displacement
RMS is.001127735/4.601386908/.072188604 for baseline/legacy/ours, while the
first-hidden values are4.87212e-9/3.10829e-7/3.28561e-8. Legacy therefore has
larger absolute phase signals despite weaker first-hidden response relative
to its output than ours. The study isolates endpoint readout noise at the
shared initial weights; it does not establish training accuracy, explain
the chronology of the earlier trained-checkpoint failures, or equalize
physical output nudging. Single seed, one draw per batch, the missing
lower-noise transition and finiteT/K limit broader conclusions. A lower-sigma
fixed-initializer replay or an explicitly equal-output-response comparison
would answer different follow-up questions; neither is launched here.

[Figure and complete tables](../paper_ready_results/conv3_init_beta_noise_20260921.md) ·
[CSV](../paper_ready_results/conv3_init_beta_noise_20260921_summary.csv) ·
[Plan](eqprop_conv3_init_beta_noise_plan_20260921.md) ·
[Raw bundles](../results/eqprop-conv3-init-beta-noise-20260921-v1/) ·
[Coverage validation](../results/eqprop-conv3-init-beta-noise-20260921-v1/analysis/validation.json).

## Provisional stability-based EqProp beta selection, September 21

Scientific direction recorded at Filip's request; **the maximum stable betas
remain unestablished**. The objective is the largest tested beta that avoids
training divergence under fixed conditions. Working reference points are
Conv1 approximately one decade below p99, Conv2 approximately p99, and Conv3
approximately p95, with Conv3 still under test. Here p99/p95 mean per-matrix
EP–BPTT cosine thresholds, not percentiles or stability guarantees.

The [protocol and evidence limits](eqprop_beta_stability_protocol_20260921.md)
record the numerical candidates and two unresolved interpretations: which
Conv1 p99 reference to use, and whether stability qualification includes
read noise. Larger candidates already complete clean training in several
cases, so the proposed depth-dependent relationship is provisional.

The [local run-coverage audit](../paper_ready_results/beta_stability_protocol_run_audit_20260921.md)
preserves three clean seeds and five noisy seed-0 conditions per group.
Retaining historical Conv1 betas leaves 32 full-budget completions needed;
literal latest-p99/10 leaves 48. Each count includes three clean controls
already declared in the separate Conv3 p95 study. These are conditional
replacement counts for wide-weight EqProp, not additions to the paused
clean-paper completion tally. No experiment was launched by this review.

## Conv3 p90 final-checkpoint gradient cosine versus noise, September 21

Outcome: **mixed**. Read-only replay of all25 completed epoch-30 checkpoints
in the p90 noise sweep measures12,816 per-matrix EP/BPTT comparisons. All25
bundles validate; source checkpoints and parameter tensors remain unchanged.
No training or official-test evaluation was performed. The two sigma1e-3
legacy/ours failures have no final checkpoint and remain explicitly missing.
The run took approximately36.4 minutes on the local RTX3090, within the
one-GPU-hour cap; summed per-case runtime is0.5993 GPU-hours, excluding smoke
and shared setup/collection time.

Frozen setup: seed0, T=K=8, native float64, centered frozen-current EP, wide
weights[0,100], zero biases, injected beta baseline404.141105702,
legacy4.42250110273, ours5.26875648112. Every final checkpoint uses the same36
validation batches of16, with clean readout and four independent noise draws
per batch at its training sigma. BPTT differentiates the matched K-step
zero-nudge continuation from the same post-T state. No Adam or LR transform
is applied to the compared gradients.

At sigma1e-5, noisy median cosines for Conv1/Conv2/Conv3/Dense are baseline
.985441/.996814/.999482/1.000000, legacy .602313/.869716/.997049/.999999,
and ours .258183/.414092/.997959/1.000000. Thus early-layer direction can
degrade substantially while the previously recorded validation accuracies
remain97.50/98.72/98.38%. This is descriptive evidence, not a causal estimate
of which layer determines accuracy.

At sigma5e-4, noisy medians are baseline .236215/.457898/.985971/.999985,
legacy .001441/.083346/-.003131/.082845, and ours
.020070/.025777/.245687/-.293154. The latter two final checkpoints also
lose clean-readout alignment: legacy -.165296/.171045/-.003132/.082845;
ours .723292/.688886/.245614/-.293156. The discrepancy at those weights
cannot be attributed solely to the added endpoint noise. This replay does
not separate finite-beta distortion from finite-T/K effects. Residual gates
are retained separately in the raw evidence and summary.

Baseline at sigma1e-3 has noisy medians .179658/.357652/.971671/.999960
and clean medians .999753/.999544/.998677/1.000000. Legacy/ours are missing
at that noise because training became nonfinite during epochs8/18; neither
best nor earlier checkpoints substitute for a final measurement.

The initial/BPTT-checkpoint p90 admission criterion does not certify
alignment at final EP-trained states or under noisy readout. Single seed,
four Monte Carlo draws per batch, finite T/K and different training GPU
classes limit generalization. All replays use one GPU class; all nine
original GPU-specific clean controls are retained. The figure's main zero
point uses RTX5090 controls, and lines are visual guides across training
conditions rather than a hardware-independent causal noise curve.

[Report, per-layer tables and figure](../paper_ready_results/conv3_p90_final_noise_cosine_20260921.md)
and [raw bundles/provenance](../results/eqprop-conv3-p90-final-noise-cosine-20260921-v1/).
The two-checkpoint smoke exercised native final loading and noisy readout;
final weights matched their NPZ representation exactly, all12816 cosines
are defined, and the source/runtime/analyzer hashes are preserved.

## Conv3 p90 versus p99: existing evidence comparison, September 21

Outcome: **inconclusive for a training-accuracy comparison between thresholds**;
the initial clean-gradient and physical-displacement comparison is available.
This is an aggregation of existing seed-0, T=K=8 diagnostics, with no new
simulations or official-test evaluations. Six calibration bundles validate
and share the same 36 initialization replay batches.

Largest measured betas passing every matrix on all 72 initialization/BPTT
checkpoint replays are baseline/legacy/ours 404.141105702/4.42250110273/
5.26875648112 for cosine >.90, and 10/.1/.987333678708 for >.99. The last
value is a retrospective lookup in the refined measurements; the earlier
coarse-grid p99 choice was .9. No targeted p99 refinement is claimed.
The p90 initial minimum first-convolution cosines are .901977/.902633/.900348;
their medians are .992760/.993805/.994578. The worst-batch criterion should
not be described as a typical-batch cosine.

No matching initial noisy-gradient cosine measurements at sigma1e-3 are
available in the inspected evidence. At fixed initial weights, physical
phase displacement is unaffected by the readout-only noise model. Apparent
noisy-readout RMS expectations are derived analytically and explicitly
separated from measured clean-state displacement. Neither determines a
missing noisy-gradient cosine.

The p90 clean/noisy accuracy outcomes are reused from the completed sweep.
For p99, baseline beta10 has 97.64% final validation at epoch30; the selected
legacy/ours values have no matching training outcomes. Ours beta.9 has a
separate ten-epoch clean result of98.46%, not evidence for beta.987334 or a
30-epoch endpoint. Missing entries remain missing, as requested.

[Comparison and source links](../paper_ready_results/conv3_p90_p99_existing_20260921.md)
include per-layer minimum/median cosines, free-to-nudged and centered phase
RMS, accuracy horizons, source hashes, and validation receipts. Reproduce
with `python -m experiments.summarize_conv3_p90_p99_existing`.

## `cifar10-l8-analog-baseline-legacy-e30-e50-seed0-20260922-v1` — final50epoch continuation comparison

Analyzed September23,2026. Outcome: **positive for additional training at the
frozen selected rates**. Baseline and legacy each complete20 more epochs from
their final30 checkpoints on Nom/local RTX3090s, preserving model/conductances,
Adam moments and counters, trainable affine BN/buffers, trainable gains, saved
RNG state and the original50epoch cosine schedule. Batch32, crop/flip input
augmentation, cross-entropy, bounds and T/K[6,6,4] remain unchanged. The separately
completed proposed/Fifi5090 continuation supplies the matched50epoch reference.
All measurements below are seed0 CIFAR45000/5000 train/validation, not test.

| Scheme | Epoch30 validation | Epoch50 validation | Gain30→50 | CE50 | Best through50 |
|---|---:|---:|---:|---:|---:|
| Baseline |86.54%|89.48%|+2.94pp|0.368819|89.48% at50|
| Proposed, separate reference |87.20%|89.92%|+2.72pp|0.372177|89.92% at50|
| Legacy |86.88%|88.72%|+1.84pp|0.385642|88.82% at44|

Continuing the existing rates improves both requested arms, with smaller gains
than the earlier10→30 interval. All three reach broadly similar performance.
Proposed has the highest final accuracy by0.44pp over baseline and1.20pp over
legacy; baseline has slightly lower final cross-entropy. One seed and different
GPU/software stacks do not establish a robust amplification advantage or a
global LR optimum. Legacy's final accuracy is0.10pp below its best44checkpoint;
keep endpoint and best-checkpoint comparisons distinct. No further training
or learning-rate search is launched by this closeout.

Both new runs and all3smokes validate locally (5canonical bundles). All40new
full epochs contain45000train/5000validation examples; both final checkpoints
have70350Adam steps,19optimizer groups,3BN counters at70350, finite bounded
conductances and scheduler epoch50. Exact parent model/Adam/RNG/scheduler
restoration and preserved per-epoch cosine rates verify; the separately
completed proposed checkpoint passes the same checks. All final solver audits
pass. New production plus smokes cost20.3131workerGPUh, below27h allowance.
No scientific failures, retries or exclusions. No official-test reads.

Both GPU workers are absent and resources released. Nom's wrapper records
exit0. Local legacy's wrapper omitted its exit-code/finished-at sidecars;
its native exit status is unknown. Successful canonical terminal artifacts,
complete log, final audit and actual checkpoint verification independently
establish scientific completion. This exploratory launcher-receipt omission
is retained in closeout_validation.json and logs/local.terminal_observation.json;
no rerun was made solely to recreate it. Nom briefly had unrelated mumax3 work
at allocation, so the launcher checked for an idle lane before smoke/production;
no unrelated job was interrupted and no CPU offloading was used.

[Full1–50 curves and comparison](../results/cifar10-l8-analog-baseline-legacy-e30-e50-seed0-20260922-v1/analysis/report.md) ·
[Checkpoint verification](../results/cifar10-l8-analog-baseline-legacy-e30-e50-seed0-20260922-v1/analysis/checkpoint_verification.json) ·
[Coverage validation](../results/cifar10-l8-analog-baseline-legacy-e30-e50-seed0-20260922-v1/closeout_validation.json) ·
[Plan](cifar_l8_baseline_legacy_e30_e50_plan_20260922.md).

## `cifar10-l8-analog-continue-e10-e30-seed0-20260922-v1` — selected LR continuation to30

Analyzed September22,2026. Outcome: **continuing the selected rates improves
all three schemes substantially**. Each resumes its own final epoch10 model,
Adam moments/counters, trainable affine BN, gains, RNG and the original50epoch
cosine schedule. Batch32, crop/flip augmentation, cross-entropy and accepted
T/K[6,6,4] remain fixed. Seed0,45000/5000 training/validation; no official test.

| Scheme | Epoch10 validation | Epoch20 | Epoch30 | Gain10→30 | CE30 | Best through30 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline |75.26%|83.16%|86.54%|+11.28pp|0.398059|87.14% at29|
| Proposed |75.36%|83.10%|87.20%|+11.84pp|0.393454|87.20% at30|
| Legacy |74.68%|82.06%|86.88%|+12.20pp|0.399088|86.88% at30|

The selected learning rates support useful longer training; these results do
not establish a global LR optimum. Proposed's final lead is0.32pp over legacy
and0.66pp over baseline. One seed and mixed GPU/software stacks do not establish
a robust amplification advantage. Epoch-to-epoch fluctuations matter: baseline's
best29 exceeds its final30. Compare fixed endpoints separately from best checkpoints.

All60 added full epochs,42210 cumulative Adam steps per scheme, exact restored
model/moments/RNG and numerical scheduler state,19 optimizer groups,3 BN counters,
finite bounded conductances and final solver audits verify locally. PyTorch2.5
adds only an inert scheduler verbose=False field relative to the2.11 parent;
this precise metadata difference is recorded in checkpoint verification.
All11 canonical bundles validate:3 production,7 smoke and1 intentionally
cancelled Akib attempt. Akib used GPU compute with CPU saved-tensor storage;
Filip disallowed offloading, so it was stopped before a complete new epoch and
replaced by GPU-only local legacy. All production workers have exited0; the
excluded Akib attempt exited143. Successful production time24.7336GPUh;
recorded production,smoke and cancelled-attempt time25.2037GPUh, below60h cap.

Filip subsequently requested20 more epochs from the final30 checkpoints for
baseline and legacy. Those are a separate study; proposed already completed50
in its separately recorded continuation. Preserve this completed30endpoint.

[Report and curves](../results/cifar10-l8-analog-continue-e10-e30-seed0-20260922-v1/analysis/report.md) ·
[Checkpoint verification](../results/cifar10-l8-analog-continue-e10-e30-seed0-20260922-v1/analysis/checkpoint_verification.json) ·
[Coverage validation](../results/cifar10-l8-analog-continue-e10-e30-seed0-20260922-v1/closeout_validation.json) ·
[Plan](cifar_l8_analog_continuation_plan_20260922.md).

## `cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1` — analog L8 Adam LR selection

Analyzed September 22, 2026. Outcome: **positive for finding short-horizon
learning-rate settings for all three amplification schemes**. All 35 declared
scientific runs complete: 27 five-epoch grid points, two triggered proposed
edge probes and six fresh ten-epoch finalists. Select by terminal epoch-10
validation accuracy, then lower cross-entropy, using the frozen rule.

| Scheme (voltage/current amplification) | Selected candidate | Epoch-10 validation | Cross-entropy | Initial analog-head LR |
|---|---|---:|---:|---:|
| Baseline 1/1 | `baseline_c1_d1_e10` | **75.26%** | 0.722593 | 0.0008335640532273448 |
| Proposed 4/1 | `ours_edge_head_down_e10` | **75.36%** | 0.691127 | 0.00005334108501991406 |
| Legacy 4/0.25 | `legacy_c1_d1_e10` | **74.68%** | 0.717528 | 0.00016 |

Retain the baseline and legacy center vectors. Relative to the legacy
convolution anchor `[[.0045,.00069,.00069],[.00069,.00049,.00049],[.00049,.000345]]`,
the selected conv multipliers are 3.001373182460858 for baseline,
9.000971237736372 for proposed, and 1 for legacy. Complete raw vectors and
scientific fields are preserved in the selected
[baseline](../configs/cifar/lr_search_20260921/confirmation/baseline_c1_d1_e10.json),
[proposed](../configs/cifar/lr_search_20260921/confirmation/ours_edge_head_down_e10.json)
and [legacy](../configs/cifar/lr_search_20260921/confirmation/legacy_c1_d1_e10.json)
configs. Use these as the tested short-horizon settings for later analog L8
comparisons; 50-epoch optimality and seed variability remain unmeasured.

The network keeps eight analog convs in [3,3,2] blocks at logical widths
128/256/512 (physical 256/512/1024), plus the analog 8192-to-10 dense classifier.
Digital pooling and **trainable affine BN** follow the blocks. Training uses
**Adam, batch 32, crop/flip augmentation and cross-entropy without label
smoothing**. BN LR stays .001 and four positive boundary-gain LRs stay .00005;
gains start at 100. Analog biases are zero and conductances stay in [1e-7,10].
All runs retain the 50-epoch cosine horizon/final ratio .02, one saved seed-0
initializer, matched input order/augmentation and the stratified 45k/5k
training/validation split. The official CIFAR test split is never read.

MNIST Conv1/Conv2 Adam ratios supply approximate conv/head priors; every
scheme receives the complete 3x3 multiplier grid {1/3,1,3}. Proposed needs a
higher conv and lower classifier rate than its transferred center: epoch-5
accuracy rises from 59.84% at that center to 66.68% at the selected edge point.
The other proposed edge, another 3x increase in conv rates, reaches 54.26%.
At epoch 10 the selected proposed head beats its 3x larger alternative by
0.34 pp (75.36% versus 75.02%). The baseline/legacy centers beat their
alternatives by 0.64/0.56 pp. Baseline's alternate has marginally lower CE;
accuracy is the declared primary criterion. Proposed's 0.10 pp lead over
baseline is only five validation examples and does not establish an
amplification advantage. Its head winner is the lowest tested rate; the
bounded search permits no further expansion. This is one-seed, short-horizon
validation evidence, with no paper-facing test accuracy.

All initial/final fixed-eight-example T/K audits pass at [6,6,4] against
[24,24,16], with reference stability checked at [48,48,32]. All 35 saved initial
model states have identical tensor hashes; all six ten-epoch executions
exactly reproduce their source run's first five epochs, including metrics,
gradients, clipping, LR vectors and step counts. Bounds, conductance changes,
BN scale/shift changes and update counts, finiteness, complete cohorts,
resolved configs, source hashes and native exits validate locally.

Peak production concurrency is **seven workers on six GPUs**: Fifi/Loulou/
Riri/Trex RTX 5090s plus Nom/local RTX 3090s, including two qualified concurrent
Trex workers. Finalists use only RTX 5090/PyTorch 2.11/cu128/cuDNN 9.19;
Trex's driver differs, and the core grid's mixed hardware/software remains a
qualification of the promotion stage. Akib's offload smoke fits but projects
beyond the per-case cap, so no production arm uses it. All **58 canonical
bundles** validate: 35 scientific, three preparation, 19 smoke and one
preserved failed pre-step startup. A missing deterministic cuBLAS environment
caused that local failure; its unchanged successful replacement is
`cells/legacy_c2_d2`, with receipt `launcher/local-retry1.json`. Planned old-queue
no-overwrite stops are explicitly superseded; no active training was killed.
Riri's clock offset is handled with monotonic timing.

Successful native scientific time is 54.4460 worker GPUh; failed startup and
reported preparation/smoke time bring recorded time to **54.6697**, below the
72-hour cap. Concurrent worker times are counted separately. The complete
local copy is authoritative; no study worker remains on any production host.
The legacy center also supplies the prior batch study's missing 5090 batch-32
control: 65.62% versus 65.50% on Nom, with identical scientific config and
initializer/split/qualification hashes.

[Report and full layerwise rates](../results/cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1/analysis/report.md) ·
[Learning curves](../results/cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1/analysis/ten_epoch_learning_curves.png) ·
[Comparison CSV](../results/cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1/analysis/comparison.csv) ·
[Coverage validation](../results/cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1/closeout_validation.json) ·
[Plan](cifar_l8_analog_lr_plan_20260921.md).

## `cifar10-l8-analog-conv-and-head-adam-bs16-32-64-seed0-v1` — analog L8 batch comparison

Outcome: **positive for the requested short comparison**. All three seed-0
runs completed five epochs; batch **32** has the highest frozen epoch-5
validation accuracy and lowest cross-entropy. This is exploratory CIFAR-10
validation evidence, not paper-facing accuracy or a long-run optimum.

| Training batch | Host / GPU | Epoch-5 validation | Cross-entropy | Total runtime |
|---|---|---:|---:|---:|
| 16 | Fifi / RTX 5090 | 61.02% | 1.0940 | 66.1 min |
| 32 | Nom-cool-1 / RTX 3090 | **65.50%** | **0.9718** | 157.5 min |
| 64 | Fifi / RTX 5090 | 60.02% | 1.1275 | 62.9 min |

The requested wider L8 uses independent analog convolution blocks [3,3,2]
at logical widths 128/256/512, followed by an analog resistive dense
8192-to-10 classifier. Physical differential widths are 256/512/1024;
the dense circuit receives 16384 physical input voltages. Digital max-pool
and **trainable affine BN** follow each block. All BN scale/shift pairs and
all nine conductance tensors changed during training. The objective is
**cross-entropy without label smoothing**, with **crop/flip augmentation**.
Adam uses the inherited layerwise analog L12 rates; no LR recalibration was
performed. Voltage/current amplification is 4/0.25, boundary gains start at
100, analog biases are frozen zero, and conductances stay within [1e-7,10].
T=K=[6,6,4] passed the initial qualification and every final checkpoint audit.

All runs share the saved initializer, deterministic stratified 45k/5k split,
epoch sample ordering and stateless per-example augmentation. Only CIFAR
training files were read; the official test split was never accessed. The
ranking rule was epoch-5 validation accuracy, then lower CE, then runtime.
Each arm completed 225000 training-example presentations and five complete
5000-example validations. All 19 trainable tensors had finite nonzero
first-batch gradients at each epoch; checkpoints, bounds, BN update counts,
initialization/split identities and native exits validate locally. All nine
canonical bundles validate, including the preserved failed Nom smoke.
Production elapsed time totals **4.7748 GPU-hours**.

Batch 32 leads batch 16 by 4.48 percentage points and batch 64 by 5.48 points;
it led at every measured epoch. Use **batch 32 as the provisional choice for
this five-epoch setup**. The user-authorized move to Nom introduced a host and
software confound (PyTorch 2.5.1/CUDA 12.1/cuDNN 9.1 versus Fifi's
2.11/CUDA 12.8/cuDNN 9.19). The numerical trajectory and runtime differences
cannot be attributed exclusively to batch size. Initializer/split hashes
match, but cross-host bitwise equivalence is not claimed. A matched batch-32
repeat on a 5090 was still outstanding at this study's closeout. The later
three-scheme Adam LR study above supplies that reference at 65.62%, with
identical scientific config and input hashes. One seed, five epochs,
different optimizer-step counts and batch-dependent BN statistics limit
generalization. The digital L8's
92.62% monitored test accuracy used a different split and 50 epochs.

Nom's first smoke failed before any optimizer step because actual cuDNN
libraries disagreed with package metadata. A study-local cuDNN 9.1 runtime,
verified by a conv/BN backward probe and repeated smoke, fixed the environment
without changing global packages. Preserve and exclude that smoke from
scientific aggregation. The first Fifi queue-handoff attempt failed a
protective working-directory assertion before modifying jobs; the corrected
handoff preserved batch 16 and ran only batch 64, avoiding duplicate batch 32.
All three production runs and both final launchers exited zero. The full
local artifact copy is authoritative; no run from this study remains active.

[Report](../results/cifar10-l8-analog-conv-and-head-adam-bs16-32-64-seed0-v1/analysis/report.md) ·
[Learning curves](../results/cifar10-l8-analog-conv-and-head-adam-bs16-32-64-seed0-v1/analysis/learning_curves.png) ·
[Comparison CSV](../results/cifar10-l8-analog-conv-and-head-adam-bs16-32-64-seed0-v1/analysis/comparison.csv) ·
[Coverage validation](../results/cifar10-l8-analog-conv-and-head-adam-bs16-32-64-seed0-v1/closeout_validation.json) ·
[Plan](cifar_l8_analog_batch_plan.md).

## `cifar10-digital-l8-wide-affinebn-adam-seed0-20260921-v1` — wider digital L8

Analyzed September 21, 2026. Outcome: **positive** for retaining digital
CIFAR-10 accuracy with fewer convolutions. The requested wider `[3,3,2]`
network reaches **92.62% best official-test accuracy at epoch48**, and
**92.57% final**, versus the existing L12's92.30% best/final. This descriptive
single-seed difference is +.32pp best / +.27pp final, not a significance claim.

Logical widths128/256/512; eight bias-free3x3 ReLU convolutions, each block
followed by2x2 max-pooling then trainable affine BN, Linear8192-to-10 head.
There are5,395,594 parameters versus10,849,674 for L12. Both gamma and beta
changed in all three BN layers; each accumulated4900 statistic updates.
Adam .001, cosine50 epochs to.00002, weight decay.0003 except zero on BN,
batch512, seed0, float32/TF32-off, unchanged CIFAR crop/flip/normalization.

One50-epoch production run and a passing target smoke completed on Akib's
RTX3080. Runtime734.55s (12m15s,.2040GPUh), mean epoch14.60s versus L12's
19.36s; peak reserved6.65GiB. All50 full50k/10k epochs and4900 steps are
present, with16 finite nonzero first-batch parameter gradients each epoch.
Native launcher exit0; no worker remains. Production/smoke bundles are
collected and validate locally, including artifact hashes and finite final
checkpoint state. Local CPU shape/count/BN-gradient check passed. The target
smoke served as execution gate because of the prior local cuDNN problem and
remote CIFAR data availability. Initial sandbox-network-blocked launcher
attempt created no job; permitted retry launched unchanged; no training retry.

The narrower depth preserves the observed digital performance in this run.
Architecture, pooling and BN changed together, so the comparison does not
isolate BN's contribution. Official test is monitored every epoch and selects
the checkpoint: exploratory evidence, not an untouched-test paper estimate.
Analog accuracy, solver convergence and analog runtime are untested. No analog
or additional digital run is scheduled.

**Recommendation, recorded at Filip's request on September 21:** use this
wider L8 as the reference architecture for the next **all-analog convolution
tests**. Keep three coupled analog blocks of depths **[3, 3, 2]**, with logical
channels `[128,128,128] / [256,256,256] / [512,512]`. Differential physical
channels are twice these widths. Retain digital 2x2 max-pooling and trainable
affine BN after each block, the digital Linear(8192,10) classifier, and the
same crop/flip augmentation. Here “all-analog” refers to all eight
convolutions; normalization, pooling and the classifier remain digital.

Use cross-entropy on the 10-class logits, with no label smoothing, as in
this digital run. Its final test cross-entropy is **0.3224**, versus **0.4268**
for L12. The digital accuracy and reduced depth motivate this recommendation;
they do not establish analog accuracy or simulation speed. Qualify the new
analog blocks' solver convergence and T/K, and choose analog-specific learning
rates and a fitting batch size before full training; digital batch 512 and
Adam LR .001 are reference settings, not qualified analog settings. Keep the
measured digital result as the comparison baseline. This records the next
architecture recommendation and does not launch another experiment.

[Report](../results/cifar10-digital-l8-wide-affinebn-adam-seed0-20260921-v1/analysis/report.md) ·
[Learning curves](../results/cifar10-digital-l8-wide-affinebn-adam-seed0-20260921-v1/analysis/learning_curves.png) ·
[Validation](../results/cifar10-digital-l8-wide-affinebn-adam-seed0-20260921-v1/closeout-validation.json) ·
[Config](../configs/cifar_digital_l8_wide_affinebn_20260921.json).

## `cifar10-digital-l12-logical-width-adam-seed0-20260921-v1` — digital CIFAR L12 baseline

Analyzed September 21, 2026. Outcome: **positive** for the requested digital
trainability baseline. The fully digital feed-forward BP network completes
50 epochs on Akib's RTX 3080 and reaches **92.30% best/final official-test
accuracy**, both at epoch 50. Final augmented-train accuracy is **99.754%**.

The user explicitly chose to match the analog **logical** widths 128/256/512,
halving its physical channel counts. Twelve bias-free 3x3 convolutions with
ReLU are grouped as `[128,128] / [128,128] / [256,256,256] / [256,512,512] /
[512,512]`; pooling and non-affine BatchNorm positions match the all-analog
graph. The digital readout is Linear(8192,10), with bias. There are
**10,849,674 trainable parameters**, no amplification or trainable voltage
gains, no differential duplication, and no conductance bounds.

Frozen setup: seed 0, Adam LR .001, betas .9/.999, eps 1e-8, weight decay
.0003, cosine decay to .00002 over 50 epochs, batch **512**, float32 with
TF32 disabled, and the historical CIFAR flip/crop/normalization. Batch 512
was selected from fitting target memory smokes at 256/512, with **6.68 GiB**
peak reserved memory. No learning-rate scaling or search was performed.

All 4900 optimizer steps and 50 full train/test epochs are present; the runner
took **975.60 seconds (16m16s, .2710 GPU-hours)**. Metrics and recorded first-
batch gradients remain finite. Final checkpoint state is finite and contains
4900 BatchNorm updates. The native wrapper exited zero; no GPU worker remains.
Production and three successful target smoke bundles validate locally, as
does the preserved failed auxiliary local bundle. All **26 remote production
and launcher file hashes** match the authoritative local copy. The local
auxiliary smoke failed at cuDNN initialization before any optimizer step;
its preceding sandbox CUDA-access failure and both logs are preserved and
excluded. Production required no retry.

For context, historical hybrid/all-analog best accuracies are 92.32/85.04%:
the digital result is .02pp below the hybrid and 7.26pp above all-analog.
This is a descriptive one-seed comparison. Parameter count, activation,
learning rates, batch size and GPU model differ, so it does not isolate a
causal effect of analog computation or amplification. The official test set
is monitored every epoch and selects the checkpoint; this is exploratory
evidence, not an untouched-test paper estimate. No further runs are scheduled.

[Report](../results/cifar10-digital-l12-logical-width-adam-seed0-20260921-v1/analysis/report.md) ·
[Learning curves](../results/cifar10-digital-l12-logical-width-adam-seed0-20260921-v1/analysis/learning_curves.png) ·
[Epoch table](../results/cifar10-digital-l12-logical-width-adam-seed0-20260921-v1/analysis/epoch_metrics.csv) ·
[Validation](../results/cifar10-digital-l12-logical-width-adam-seed0-20260921-v1/closeout-validation.json) ·
[Config](../configs/cifar_digital_l12_logical_width_20260921.json).

## `eqprop-conv3-p90-read-noise-1em3-20260920-v1` — higher-noise extension

Analyzed September 21, 2026. Outcome: **mixed**. All three requested cases
are terminal: baseline completed 30 epochs; legacy and ours became non-finite.
These are ordinary-MNIST validation diagnostics, seed 0, with no official-test
read. No clean control or previously tested noisy setting was repeated.

The only scientific change from the matched RTX5090 sweep is endpoint-read
noise sigma=1e-3. The same frozen runner, saved initialization, train/validation
split and ordering, float64 centered EqProp, T=K=8, [0,100] weights, zero frozen
biases and exact Adam rates were reused. Injected beta remains
404.141105702/4.42250110273/5.26875648112 for baseline/legacy/ours. Three short
local and three Fifi smoke tests passed. Production ran sequentially on Fifi's
RTX5090; other 5090s were occupied. The completed RTX5090 clean controls were
reused (97.72/98.70/98.52% final validation).

| Scheme | Outcome at sigma=1e-3 | Last complete validation | Best observed validation |
|---|---|---:|---:|
| Baseline | 30 epochs complete; clean-relative drop 0.56pp | 97.16% (epoch30) | 97.20% |
| Legacy | Non-finite at epoch8, minibatch2791 | 94.76% (epoch7) | 97.22% |
| Ours | Non-finite at epoch18, minibatch1638 | 95.14% (epoch17) | 96.40% |

Both failures were `NonFiniteTrainingError` in the inference voltages of
`Layer_1` (200704 non-finite elements), not OOM or transport failures. Their
partial accuracies are not 30-epoch results. Both failed bundles, logs and best
checkpoints are retained; no retries or accuracy-based exclusions were made.
Baseline's maximum running-best decline was 0.20pp; its final own-best drop
was 0.04pp and its predeclared endpoint screen passes.

The result places a clear limit on the zero-noise cosine argument: passing
cosine >.90 across the calibration cohort and passing clean training does not
guarantee finite training at sigma1e-3. Only baseline completed this setting;
ours failed later than legacy, which is a single-seed observation rather than
a general robustness guarantee. Scheme-specific betas/LRs remain part of the
comparison. The earlier sigma<=5e-4 sweep is a separate completed study;
comparisons against its A100 high-noise points also change GPU class. No
noise-specific beta tuning or official-test evaluation is implied.

All three local canonical bundles validate, and all97 production-file hashes
match the finished remote copy. The launcher exited0 because both failures
were classified scientific terminal outcomes; native codes were0/1/1.
Production used7.640955 GPU-hours, with short preparation smokes counted
separately. No experiment worker remains. Coverage is3/3 terminal,1/3 full
30-epoch completions,2/3 scientific failures,zero operational failures.

[Results and trajectories](../paper_ready_results/conv3_p90_read_noise_1em3_20260920.md) ·
[CSV](../paper_ready_results/conv3_p90_read_noise_1em3_20260920.csv) ·
[Raw study](../results/eqprop-conv3-p90-read-noise-1em3-20260920-v1/) ·
[Validation](../results/eqprop-conv3-p90-read-noise-1em3-20260920-v1/closeout-validation.json).

## `eqprop-conv3-p90-read-noise-20260919-v1` — refined beta read-noise sensitivity

Analyzed September 20, 2026. Outcome: **mixed**. All 24 thirty-epoch runs
complete: 15 noisy settings and nine GPU-matched clean controls. The refined
p90 betas work as empirical Conv3 operating points in this seed, but the
robustness ordering depends on the scheme, noise level and chosen beta.
These are ordinary-MNIST validation diagnostics; the official test is unread.

The contract freezes seed 0, the 55,000/5,000 split, matched initialization
and minibatch order, T=K=8, float64 centered frozen-current EqProp, perfect
diodes, zero biases, wide [0,100] weights and the existing scheme-specific
Adam rates. Injected betas are 404.141105702 (baseline), 4.42250110273 (legacy),
and 5.26875648112 (ours): the largest measured values passing every matrix at
cosine >.90 across 36 zero-noise batches at initialization and the saved BPTT
checkpoint. The six p90/p95 ten-epoch clean pilots are separate admission
and sensitivity evidence. Beta remains fixed across noise levels. Independent
Gaussian noise affects copied positive/negative non-input endpoint voltages
for gradient readout; relaxation and validation remain clean.

Final validation drops from each scheme's matching clean control are:

| Read-noise sigma | GPU class | Baseline (pp) | Legacy (pp) | Ours (pp) |
|---:|---|---:|---:|---:|
| 1e-5 | V100 | 0.06 | 0.04 | 0.20 |
| 3e-5 | RTX5090 | 0.16 | 0.00 | 0.40 |
| 1e-4 | A100 | 0.02 | 0.08 | 0.64 |
| 3e-4 | A100 | 0.02 | 2.46 | 1.02 |
| 5e-4 | A100 | 0.18 | 3.00 | 1.38 |

At sigma5e-4, final validation is 97.40/95.72/97.12% for
baseline/legacy/ours. Baseline loses less accuracy than ours at every level;
legacy loses less than ours at the three lower levels, while ours is less
impaired than legacy at the two higher levels. No universal robustness
ordering follows. The new highest-noise accuracies exceed the earlier
beta10/.001/3 outcomes by 1.36/20.28/.52pp, respectively, but the environments
also changed. This comparison cannot isolate a causal beta-only effect.

All runs are finite and pass the predeclared final-best drop <5pp screen.
That screen misses temporary deterioration: legacy falls 7.34pp below its
running best at epoch17 for sigma3e-4, and 9.16pp at epoch20 for sigma5e-4,
then recovers to final best-to-final drops of 1.78/1.86pp. Full curves and
these diagnostics are retained; no low-accuracy run was excluded. Static
cosine and finite completion therefore do not certify smooth noisy training.
Report the actual fidelity-constrained selection followed by empirical
training qualification, without retrofitting it as the historical beta rule
or claiming it optimizes noisy performance. Multi-seed confirmation, a noisy
p95 comparison and noise-aware selection remain separate possible studies.
The six refined Conv2 values remain calibration-only; finite-T8 caveats persist.

All 24 canonical bundles, 30-epoch histories, best/final float64 checkpoints,
source/config/initializer/cohort/order checks, zero official-test reads and
noise draw counts validate locally. All 54 Slurm allocations and four local
launchers exit successfully. Remote/local production checksums match.
Production consumed 62.58194 HPC plus 26.41622 local physical GPU-hours,
88.99817 total; preparation checks are separate from this subtotal. Four
RTX5090 hosts supplemented A100/V100. Measured paired-worker contention led
to temporary in-memory clean pauses on Fifi/Riri, then automatic resumes;
all training states and coverage were preserved. Superseded 3090 preparation,
smokes, timing/continuation canaries and canceled pending canaries remain
excluded and named in the closeout. No production run failed or was replaced.

[Full table, curves and CSV](../paper_ready_results/conv3_p90_read_noise_20260919.md) ·
[Beta protocol assessment](../paper_ready_results/beta_selection_protocol_assessment_20260919.md) ·
[Plan](eqprop_conv3_p90_read_noise_plan_20260919.md) ·
[Coverage and accounting](../results/eqprop-conv3-p90-read-noise-20260919-v1/closeout-validation.json).

## `eqprop-conv3-refined-beta-training-20260919-v1` — refined beta and clean training

Analyzed September 19, 2026. Outcome: **positive** for the narrow ten-epoch
stability question. All six refined Conv3 settings completed ten finite epochs
and passed the final-drop <5 pp screen. This is ordinary-MNIST validation
evidence, with no official-test evaluation.

The frozen setup uses seed 0, zero read noise, T=K=8, wide [0,100] weights,
zero biases, float64 centered EqProp, unchanged scheme-specific Adam rates,
and matched initialization and minibatch order. The betas are the largest
measured values passing every matrix on 36 batches at two reference states;
they are not proven mathematical maxima.

| Scheme | Beta at cosine >.95 | Final validation | Beta at cosine >.90 | Final validation | Larger minus smaller |
|---|---:|---:|---:|---:|---:|
| Baseline | 147.682614594 | 97.14% | 404.141105702 | 97.16% | +.02 pp |
| Ours | 2.49274796756 | 98.48% | 5.26875648112 | 98.52% | +.04 pp |
| Legacy | 2.81845428732 | 98.58% | 4.42250110273 | 98.54% | -.04 pp |

Historical controls contribute only their first ten epochs: baseline beta 100
reaches 97.02%, ours beta 3 reaches 98.42%, and legacy beta .001 reaches 98.64%.
There is no consistent performance penalty from the larger refined beta on
this seed. Differences of one or two examples do not establish reproducible
effects. Earlier Conv1/Conv2 failures despite good static cosine still refute
a universal cosine-based stability guarantee. Keep empirical training and
validation as selection evidence, with gradient fidelity a separate claim.
The finite-T8 residual caveat remains.

All six canonical bundles, epoch histories, finite float64 zero-bias best/final
checkpoints, initializer/split/order checks, native receipts and Slurm receipts
validate locally. Remote/local checksums match. H100 allocations consumed
3.44556 GPU-hours including failed startups and the remote canary; local smokes
are separate preparation. Original task 2178212_0 is retained. Its five siblings
failed before training from double array indexing; fixed-wrapper canary
2178310 passed, then replacements 2178358_1–5 completed. Every attempt remains
preserved. No scientific failure was excluded or replaced.

The separate **30-epoch read-noise follow-up** is now complete: 15 noisy
settings and nine GPU-matched clean controls across V100, A100 and RTX5090.
See the September20 entry above. No paper beta was automatically promoted,
and no official-test read or Overleaf edit occurred. The six refined Conv2
values still lack new fine-grid training repeats; their earlier coarse-grid
training evidence remains unchanged.

[Report, plots and CSV](../paper_ready_results/conv3_refined_beta_training_20260919.md) ·
[Protocol assessment](../paper_ready_results/beta_selection_protocol_assessment_20260919.md) ·
[Local evidence](../results/eqprop-conv3-refined-beta-training-20260919-v1/) ·
[Validation and accounting](../results/eqprop-conv3-refined-beta-training-20260919-v1/closeout-validation.json).

## `eqprop-beta-refinement-20260919-v1` — refined cosine boundaries and beta-selection protocol

- Analyzed: 2026-09-19. Outcome: **mixed**. The refined grid resolves all
  twelve requested static cosine boundaries, but the joined training evidence
  rules out treating static cosine alone as a stability certificate. This is
  exploratory calibration and validation evidence; no official test was read.
- Scope: Conv2/Conv3 × baseline/ours/legacy, seed 0, zero read noise,
  unchanged centered float64 frozen-current EqProp/BPTT replay, T=K=6/8,
  36 fixed validation-partition batches at matched initialization and the
  saved BPTT checkpoint. Every matrix on every replay must exceed .90/.95;
  no norm gate. Added 31 Conv2 and 57 Conv3 beta settings (6,336 replays),
  reusing 102 settings for 190 total. All twelve passing/failing brackets
  have relative beta width 4.14–4.74%; none remains open.

| Model / scheme | Largest measured passing beta, cosine >.90 | Cosine >.95 |
|---|---:|---:|
| Conv2 baseline | 720.198 | 542.236 |
| Conv2 ours | 63.7712 | 34.0866 |
| Conv2 legacy | 43.0177 | 31.3825 |
| Conv3 baseline | 404.141 | 147.683 |
| Conv3 ours | 5.26876 | 2.49275 |
| Conv3 legacy | 4.42250 | 2.81845 |

- All selected points are limited by the first convolutional matrix at
  initialization. The curves are not globally assumed monotone: Conv3 legacy
  improves from cosine .960632 at beta 1 to .970996 at beta 1.5. These are
  measured brackets, not mathematical maxima or training-qualified betas.
- Audited 23 distinct existing clean seed-0 training settings: 17 met the
  ten-epoch stability screen and six became nonfinite. All 23 pass whole-
  gradient cosine >.99 and symmetric norm mismatch <=.10. Per-matrix >.95
  still admits two failures. Conv1 ours beta 1500 even passes per-matrix
  >.95 plus norm mismatch <=.10, but fails in epoch 1. Conversely, Conv3
  ours beta 22.5 trains for ten epochs at 98.38% final validation despite
  worst matrix cosine .454946; its beta-3 control reaches 98.42%.
- Worst-case diagnostics obscure distributions: the stable Conv3 baseline
  beta-750 and ours beta-22.5 initializer median batch-minimum cosines are
  .973566 and .965799. Their saved BPTT checkpoint passes .95 on every batch.
  Failed Conv2 legacy beta 30 passes .95 on every batch at both checkpoints.
  Neither a minimum nor a median is established as a universal selector.
- Recommended prospective protocol: declare equal candidate/search budgets,
  screen using matched ten-epoch training, then confirm a small shortlist at
  the full paper horizon across seeds 0/1/2 and choose by validation under
  fixed checkpoint rules. Retain gradient diagnostics as fidelity evidence.
  For existing fixed-beta noise curves, claim robustness of the listed
  operating points. Claims about optimized noisy performance need equally
  budgeted noise-aware tuning; Conv3 legacy's clean-stable beta 1 has not been
  tested under the same noise sweep. Do not retrofit a new selection rule
  onto the historical runs or infer a reliable gain from one-seed differences.
- Validation: all 88 new canonical bundles, six successful smokes, 23 reused
  training bundles, source guards, coverage and native receipts validate
  locally. All study workers ended. A conservative physical GPU-time
  envelope is 2.6462 hours, below the four-hour cap, counting overlapping
  local workers once and adding remote smoke durations.
- Placement/deviations: both production workers ran on the local RTX3090.
  Trex/Riri received only smokes because unrelated clients arrived before
  production. Preserved snapshot-Git metadata aborts produced no measurements;
  the documented Trex diagnostic-tolerance review affected no production
  selection. Exclusions and recovery receipts are named in the closeout.
- Limits and remaining work: calibration uses two reference parameter states,
  not the actual new EqProp trajectories. At calibration closeout the refined
  values had not been newly trained; the subsequent Conv3 training entry above
  now supplies six ten-epoch outcomes. Full-horizon multi-seed finalist confirmation and any
  noise-aware selection remain proposed. No additional training, paper edit,
  automatic beta promotion or official-test evaluation was launched.
- [Protocol assessment](../paper_ready_results/beta_selection_protocol_assessment_20260919.md),
  [refined results and plots](../paper_ready_results/beta_refinement_20260919.md),
  [plan](eqprop_beta_refinement_plan_20260919.md),
  [local evidence](../results/eqprop-beta-refinement-20260919-v1/),
  [closeout validation](../results/eqprop-beta-refinement-20260919-v1/closeout-validation.json).

## `eqprop-layerwise-beta-training-20260919-v1` — per-matrix cosine and ten-epoch performance

- Analyzed: 2026-09-19. Outcome: **mixed**. Twelve threshold conditions map
  to nine distinct settings: eight passed ten finite epochs, and one reused
  matching pilot became nonfinite. All five newly trained settings completed.
  Larger beta did not consistently reduce final validation accuracy among
  completed comparisons; passing the static per-matrix cosine rule still
  does not guarantee training stability.

| Architecture / scheme | beta for cosine >.90 | Epoch-10 validation | beta for cosine >.95 | Epoch-10 validation |
|---|---:|---|---:|---|
| Conv2 baseline | 500 | 96.96% | 500 | Same run |
| Conv2 ours | 50 | 97.96% | 30 | 97.84% |
| Conv2 legacy | 30 | Nonfinite epoch 5, batch 1910 | 30 | Same failed run |
| Conv3 baseline | 300 | 97.02% | 100 | 97.02% |
| Conv3 ours | 3 | 98.42% | .9 | 98.46% |
| Conv3 legacy | 1 | 98.64% | 1 | Same run |

- Frozen selection: largest passing **tested injected beta**, with strict
  cosine `>` for every weight matrix on all 36 fixed calibration batches
  from the validation partition, at both initialization and the saved BPTT checkpoint (72 replays). Reused
  September 18 raw calibration measurements; no additional norm gate.
  Worst symmetric norm mismatch is reported separately and exceeds .10
  for every selected point. Legacy selections are open upper grid edges,
  not established maxima. Equal selections are not independent replicas.
- Training: seed 0, ordinary-MNIST 55k/5k, zero read noise, batch16,
  float64 centered frozen-current EqProp, T=K=6/8, wide [0,100] weights,
  zero biases, inherited fixed Adam vectors, saved initializers and matching
  epochwise minibatch orders. This is exploratory validation evidence;
  official test data was not read.
- Larger-minus-smaller final accuracy for the three distinct .90/.95 pairs
  is +.12pp (Conv2 ours), .00pp (Conv3 baseline), and -.04pp (Conv3 ours).
  Relative to historical controls, Conv2 baseline beta500 matches beta100
  at 96.96%; Conv2 ours beta50/beta30 differ by +.08/-.04pp from beta10's
  97.88%; Conv3 legacy beta1 matches beta.001 at 98.64%. Small differences
  from one seed do not establish a reliable improvement or degradation.
- The reused Conv2 legacy beta30 pilot has worst matrix cosine .954935,
  yet failed in epoch 5. Its beta.03 control reached 98.00% at epoch 10.
  The two calibrated parameter states do not cover every state reached by
  the new EqProp training trajectory; this study does not identify the
  unique mechanism of the failure.
- Reuse: Conv3 baseline beta100 and ours beta3 use only the first ten
  epoch records of historical 30-epoch controls, with unchanged fixed rates
  and data order. Their full-run best checkpoints are not called epoch-10
  checkpoints. Conv2 legacy beta30 and Conv3 legacy beta1 reuse explicitly
  named prior ten-epoch pilots, including the numerical failure.
- Placement: Riri/Trex/Fifi completed three settings; both Conv2 ours
  settings completed together on the local RTX3090. Two partial Loulou
  attempts were superseded after unrelated clients arrived, using a
  predeclared throughput-based decision. Partial results are preserved and
  excluded; two short operational probes and a transport abort before
  training are also excluded. No selection between attempts used accuracy.
- Validation: all nine included bundles, six historical controls, native
  receipts, and preserved exclusions reconcile locally. Successful runs
  match initializers and ten epochwise orders; their metric histories and
  applicable best/final checkpoints are finite. Frozen numerical source
  hashes reverify locally and on all three completed remote targets.
  All study workers exited. Seven production attempts consumed 8.67 worker
  hours including superseded work, within the 20-hour budget; occupancy
  includes shared GPU time and excludes short smoke/probe overhead.
- Limits: one seed, ten epochs, zero noise, exploratory host differences;
  no full-horizon/noisy-training qualification or automatic beta promotion.
- [Report, plots and epoch CSV](../paper_ready_results/layerwise_beta_training_20260919.md),
  [plan](eqprop_layerwise_beta_training_plan_20260919.md),
  [local evidence](../results/eqprop-layerwise-beta-training-20260919-v1/),
  [coverage audit](../results/eqprop-layerwise-beta-training-20260919-v1/closeout-validation.json).

## `eqprop-beta-training-stability-20260918-v1` — larger-beta training stability

- Analyzed: 2026-09-19 (Europe/Paris). Outcome: **mixed**. All nine pilots
  have terminal, locally validated outcomes: six Conv1/Conv2 candidates
  became nonfinite, while all three Conv3 candidates passed ten epochs.
  Static whole-gradient calibration is insufficient to qualify training
  stability across these architectures.

| Architecture | Scheme | Injected beta | Outcome | Final validation / smaller-beta epoch-10 control |
|---|---|---:|---|---|
| Conv1 | baseline | 5000 | Nonfinite: epoch 1, batch 376 | No completed epoch |
| Conv1 | ours | 1500 | Nonfinite: epoch 1, batch 546 | No completed epoch |
| Conv1 | legacy | 900 | Nonfinite: epoch 1, batch 1696 | No completed epoch |
| Conv2 | baseline | 1000 | Nonfinite: epoch 7, batch 813 | Best before failure 96.48% |
| Conv2 | ours | 100 | Nonfinite: epoch 3, batch 481 | Best before failure 95.96% |
| Conv2 | legacy | 30 | Nonfinite: epoch 5, batch 1910 | Best before failure 97.18% |
| Conv3 | baseline | 750 | Ten-epoch stable | 97.08% / 97.02% (beta 100) |
| Conv3 | ours | 22.5 | Ten-epoch stable | 98.38% / 98.42% (beta 3) |
| Conv3 | legacy | 1 | Ten-epoch stable | 98.64% / 98.64% (beta .001) |

- Frozen contract: seed 0, ten epochs, zero endpoint read noise, inherited
  Adam vectors, zero biases, wide [0,100] weights, T/K=4/6/8, centered
  frozen-current float64 EP, ordinary-MNIST 55k/5k. The candidates are the
  largest tested betas passing whole-gradient cosine >=.99 and symmetric
  norm mismatch <=.10 on all 36 batches at both initialization and a saved
  BPTT checkpoint. Legacy Conv2/Conv3 were open upper calibration-grid edges.
  No optimizer steps occurred during calibration.
- Early stability required ten finite epochs and a final validation drop
  strictly below 5pp from the run's own best. All three Conv3 final drops
  were zero; maximum transient drops were .04/.00/.12pp for baseline/ours/
  legacy. Final differences from their smaller-beta controls were
  +.06/-.04/.00pp. These single-seed results do not establish an accuracy gain.
- Instability can follow good early learning. Conv2 legacy reached 97.18%
  before failing in epoch 5; baseline fell from 96.48% to 93.92% before its
  epoch-7 failure; ours fell from 95.96% to 84.74% before its epoch-3 failure.
  Conv1 legacy's sampled first-layer gradient L2 rose from 1.159 to about
  6962 before failure. Sparse traces do not identify a unique causal mechanism.
- Conv1 ours beta 1500 also passes the .95 per-matrix cosine rule with its
  norm gate (worst cosine .951337; mismatch .084590), yet fails in epoch 1.
  This rejects that particular looser per-matrix candidate too; the pilots
  do not test every threshold/grouping combination.
- Placement: eight cases on Riri RTX5090 with two concurrent workers; Conv3
  ours moved before training to a newly free Trex RTX5090. Trex/Riri smoke
  losses and gradient/update norms match exactly. Local/Riri smoke differences
  are below 5.8e-12 for loss and 1.57e-10 for gradient/update norms. This is
  exploratory evidence with an explicit host-placement revision.
- Validation: nine canonical production bundles and native exit receipts
  reconcile; six scientific failures are retained, with no exclusions or
  production retries. All initializers match historical controls; successful
  runs additionally match all ten minibatch-order hashes, have ten finite
  metric records, and finite float64 best/final checkpoints with zero biases.
  Frozen source/config hashes reverify locally and on both hosts. The Riri
  supervisor captured both native successful exits and retired the obsolete
  queues; every case finished within its original four-hour limit. No owned
  training, queue, or supervisor process remains. Production host occupancy
  was approximately 6.03 GPU-hours; worker time 10.28 hours, within budget.
- Limits: one seed, zero read noise, ten epochs only; no full 30-epoch
  qualification, no noisy-training qualification, no automatic beta or
  three-seed promotion. Official test data remains unread. The global
  criterion can hide layerwise errors, but these pilots do not isolate
  that mechanism from finite-nudge effects along the new training trajectory.
- [Report and plots](../paper_ready_results/beta_training_stability_20260918.md),
  [plan](eqprop_beta_training_stability_plan_20260918.md),
  [local evidence](../results/eqprop-beta-training-stability-20260918-v1/), and
  [coverage audit](../results/eqprop-beta-training-stability-20260918-v1/closeout-validation.json).

## `eqprop-beta-rule-comparison-20260918-v1` — beta thresholds and gradient grouping

- Analyzed: 2026-09-18. Outcome: **positive for strong dependence on the
  acceptance rule**, without establishing that larger betas improve training.
  Whole-gradient selection permits a larger tested beta in 50/54 matched
  comparisons; adding symmetric norm mismatch <=.10 changes 16/54 selections.
- Evidence class: zero-noise seed-0 ordinary-MNIST diagnostic. All nine wide
  [0,100] Conv1/2/3 × baseline/ours/legacy cases use the exact saved initializer
  and BPTT best-validation checkpoint, fixed zero biases, perfect diodes,
  centered frozen-current float64 EqProp, and unchanged T=K=4/6/8. Replay the
  four historical plus 32 additional batches of 16. BPTT uses the same post-T
  state and K; no optimizer steps or official-test reads.
- Coverage: all 153 beta cases complete and validate locally: 11,016
  checkpoint/batch gradients and 33,048 matrix comparisons. Compare .90/.95/.99
  cosine per matrix versus raw concatenated weight gradient, with and without
  the norm gate at the same grouping. Require every batch at both checkpoints;
  report 108 joint selections and 324 joint/checkpoint-specific selections.
  Choose the largest passing tested beta without another decade reduction.
- At cosine >=.99 plus norm mismatch <=.10, the selected injected betas are:

  | Case | Every matrix | Whole gradient |
  |---|---:|---:|
  | Conv1 baseline | 300 | 5000 |
  | Conv1 ours | 300 | 1500 |
  | Conv1 legacy | 60 | 900 |
  | Conv2 baseline | 100 | 1000 |
  | Conv2 ours | 10 | 100 |
  | Conv2 legacy | 3 | 30* |
  | Conv3 baseline | 10 | 750 |
  | Conv3 ours | .3 | 22.5 |
  | Conv3 legacy | .1 | 1* |

  Stars mark passing upper grid edges, not identified maxima. Across all 108
  rules there are 18 open upper edges and no disconnected passing regions.
- Interpretation: aggregate alignment can hide errors in matrices with small
  gradient norms. Conv3 baseline beta 750 has minimum whole-gradient cosine
  .994088 but worst matrix cosine .696630. Conv3 ours beta 3 reproduces its
  previously documented initialization C0 failure (cosine .939168, norm
  mismatch .278178); its .99 cosine-only matrix limit is .9, reduced to .3 by
  the norm constraint. This is separate from its observed training stability.
- Validation: 26 focused tests pass. All 13 overlapping historical full cases
  reproduce within 1.10e-14. The BPTT norm is exactly unchanged in 31,104
  repeated layer comparisons across beta, with identical input payload hashes.
  Runtime, checkpoint, frozen-force, parameter and zero-bias guards pass.
  No numerical or operational case failures, exclusions, or replacements.
- Execution: user-authorized parallel placement kept all 51 Conv3 cases on
  local RTX3090 and all 102 Conv1/Conv2 cases on Akib RTX3080. Remote smoke
  reproduced the historical batch exactly; five path substitutions are the
  only transported-config changes. Both launchers exited zero. Charged cost
  2.7499/6 physical GPU-hours, including smokes. An initial local two-worker
  benchmark yielded 1.068× throughput and retained both completed cases;
  subsequent execution used one worker per GPU. Superseded admission
  coordinators were stopped between cases; their archived states remain in
  the study directory and do not represent failed scientific runs.
- Limits: one seed, reused selection cohort, only initialization and the
  selected trained checkpoint. Raw gradients are not learning-rate-scaled or
  Adam updates. Conv3 baseline retains its independent T8 residual caveat.
  These are calibration limits, not full-training or multi-seed qualifications;
  no training beta was promoted and no paper table or training run was changed.
- Artifacts: [report and all threshold tables](../paper_ready_results/beta_rule_comparison_20260918.md),
  [beta-limit figure](../paper_ready_results/beta_rule_comparison_20260918_limits.png),
  [verification](../paper_ready_results/beta_rule_comparison_20260918_verification.json),
  [raw canonical bundles](../results/eqprop-beta-rule-comparison-20260918-v1/production/),
  [plan and transport amendment](eqprop_beta_rule_comparison_plan_20260918.md).

## `section43-conv3-initialization-magnitudes-20260918-v1` — Conv3 gradient magnitudes at initialization

- Analyzed: 2026-09-18. Outcome: **negative for uniformly weak clean legacy
  gradients** as the explanation of its greater read-noise sensitivity.
  Legacy already has the largest clean gradients in every layer at epoch 0.
- Matched setup: all schemes share exact saved seed-0 initializer bytes and
  training fingerprints; same 36 validation batches of 16, local RTX 3090,
  float64, perfect diodes, frozen zero biases, [0,100] weights, T=K=8,
  centered frozen-current EqProp and matched finite-K BPTT. Injected betas
  10/3/.001 retained. No optimizer, read noise, or official-test access.
- Initialization mean EP per-weight RMS for Conv1/Conv2/Conv3/readout:
  baseline 4.372e-5/2.806e-6/2.625e-6/9.464e-5;
  balanced 2.753e-4/1.864e-5/1.746e-5/6.328e-4;
  legacy 2.429e-3/1.703e-4/1.650e-4/5.914e-3.
  Legacy is 56–63× baseline. Clean BPTT mean norms agree within .856%.
- Ratios of trained mean norm to initial mean norm are
  1.018/11.289/68.966/.6944 for baseline, .2156/1.903/4.162/.2398 for
  balanced, and .1346/.6078/.8696/.1802 for legacy. Baseline's first-layer
  gradient remains similar; its second/third grow. Legacy's decrease in all
  layers. The figure separately reports means of paired minibatch ratios.
- Coverage: 72 new balanced/legacy initialization batches, 288 EP/BPTT layer
  comparisons and 576 detailed gradient-statistic rows; reuse 36 baseline
  initialization batches and all trained clean values at epochs 30/23/23.
  Three-scheme smoke reproduces four baseline comparisons and eight new
  production comparisons exactly. Four replay bundles validate, all hashes
  and parameter guards pass, trained statistics reproduce the previous table.
  All 1,728 initialization residual-p90 checks pass. No exclusions or failures;
  154.203 seconds (2.570/10 GPU-minutes) including smoke, exit zero, GPU released.
- Limits: one seed, only initial/selected-trained endpoints, raw gradient norms
  rather than Adam updates; amplification, beta, states and losses differ.
  Trained baseline retains its known free-state residual caveat. Gradient size
  alone does not explain noisy accuracy or establish a causal mechanism.
- Artifacts: [report and tables](../paper_ready_results/section43_conv3_initialization_magnitudes.md),
  [JPG figure](../paper_ready_results/figures/section43_conv3_initialization_magnitudes.jpg),
  [CSV](../paper_ready_results/section43_conv3_initialization_magnitudes.csv),
  [provenance](../paper_ready_results/section43_conv3_initialization_magnitudes_provenance.json),
  [canonical replay](../results/section43-conv3-initialization-magnitudes-20260918-v1/full/).

## `section43-conv3-baseline-initialization-20260918-v1` — Conv3 baseline initialization check

- Analyzed: 2026-09-18. Outcome: **negative for the proposed late-training-only
  explanation** of near-zero early-layer gradient cosine. The phenomenon is
  already present at the exact saved initialization. This does not establish
  what preserves noisy-training accuracy.
- Scope: Conv3 baseline only, seed 0, injected beta 10, T=K=8, perfect diode,
  centered frozen-current float64 EqProp, fixed zero biases, wide [0,100]
  weights. Compare the original saved initialization (epoch 0, float32 values
  exactly promoted to float64 as in training) with the clean maximum-validation
  checkpoint (epoch 30). Both training initialization fingerprints match.
- Matched replay: the same 36 validation batches of 16 examples and six sigma
  levels as the Section 4.3 replay; eight draws per nonzero sigma, exactly
  paired with the earlier trained replay. Reuse its trained noisy measurements
  and repeat all 36 trained clean computations: every clean metric and endpoint
  state hash reproduces exactly. No optimizer steps or official-test reads.
- At sigma=5e-4, mean cosine to matched noiseless BPTT for convolution weights
  1/2/3/readout is .005745/.004053/.306445/.999882 at initialization versus
  .003998/.004179/.099958/.961369 at epoch 30. At sigma=1e-5 the first two
  initialization cosines are .1626/.2676, versus .1482/.3210 trained. Clean
  initialization mean cosines all exceed .9989.
- H1/H2 centered phase-contrast RMS is 4.8671e-7/2.2782e-6 at initialization
  versus 4.0733e-7/2.1194e-6 trained. First-weight clean BPTT mean L2 is
  .00148534 versus .00151070: it does not shrink. Second/third clean gradient
  norms increase, while third-state/output phase contrast decreases. The
  earlier suggestion that late-stage gradient shrinkage mainly causes the
  first two near-zero cosines is unsupported by this matched comparison.
- Limits: only two actual checkpoints, no saved intermediate epoch trajectory;
  one seed and one scheme. The raw-gradient cosine does not directly measure
  Adam updates, their accumulation, or the effect on classification accuracy.
  All initialization residual-p90 checks pass; the trained post-T baseline
  caveat persists (72/144 batch-layer failures at .01, largest p90 .26808).
  Subsequent phase endpoints pass. No scientific cases were excluded.
- Completion: 5,904 new initialization comparisons and 36 paired clean
  controls; 164 smoke comparisons identical to production. Four relevant
  canonical bundles validate. One regression test passes. Local RTX 3090
  usage including smoke is 208.707 seconds (3.478/10 GPU-minutes); process
  exited 0. A pre-GPU hash-namespace comparison failure was corrected and
  regression-tested; its original log remains in the result root.
- Artifacts: [report](../paper_ready_results/section43_conv3_baseline_initialization.md),
  [cosine comparison](../paper_ready_results/figures/section43_conv3_baseline_initialization_cosine.jpg),
  [signal comparison](../paper_ready_results/figures/section43_conv3_baseline_initialization_signal.jpg),
  [verification](../paper_ready_results/section43_conv3_baseline_initialization_verification.json),
  and [canonical replay](../results/section43-conv3-baseline-initialization-20260918-v1/full/).
  Gradient, displacement and signed-mean/RMS voltage CSVs and PDF/JPG/PNG/SVG
  exports are retained.

## `section43-eqprop-mechanism-20260918-v1` — Trained-checkpoint gradient alignment and phase contrast

- Analyzed: 2026-09-18. Outcome: **mixed**, supporting greater legacy noise
  sensitivity at the selected operating points while retaining early-layer
  sensitivity in the other schemes. Evidence: ordinary-MNIST validation
  mechanism replay; one training seed, no official-test access.
- Frozen setup: all nine clean seed-0 EqProp best-validation checkpoints,
  Conv1/2/3 × baseline/balanced/legacy. Conv3 baseline uses the new injected
  beta 10 clean control. Injected betas are 100/30/3, 100/10/.03, and 10/3/.001;
  T=K 4/6/8, native float64, centered frozen-current EqProp, perfect diodes,
  [0,100] weights, exact-zero biases, no optimizer. The preserved September 16
  cohort contains 36 batches of 16 validation examples. Six sigmas 0 through 5e-4,
  eight independent draws per nonzero sigma, paired across schemes/sigmas.
- Complete evidence: 324 checkpoint/batch evaluations, 39,852 individual
  layerwise gradient comparisons, 162 mean-cosine cells and 162 displacement
  groups. All 11 source/smoke/full canonical bundles validate; source files and
  native tensors remain exact. All 492 Conv3 smoke comparisons are bitwise
  identical to production. Eight targeted tests pass. No failures or exclusions.
  Local RTX 3090 budget settled at 678.147 seconds (.188374/1 GPU-hours), including
  smokes; the launcher exited 0 and released the GPU.
- Measurements: every clean layerwise mean cosine is at least .9986. At
  sigma 1e-5, Conv2 first-weight cosine is .9980/.9622/.0851 and Conv3
  third-weight cosine is .9778/.9915/.0426 (baseline/balanced/legacy).
  Conv3 H1 centered phase-contrast RMS is 4.0733e-7/1.5771e-7/2.9194e-10;
  legacy is 540.2 times smaller than balanced. Contrast decreases toward the
  input in all nine cases. Baseline and balanced Conv3 first-layer cosines
  are also low (.1482/.0672 at sigma 1e-5), and baseline has better Conv2
  convolution-gradient alignment than balanced across this grid.
- Interpretation: the figures support a small phase-signal explanation of
  legacy's greater susceptibility to injected endpoint noise, not intrinsic
  clean-gradient misalignment or a complete prediction of training accuracy.
  RMS uses the centered half-difference of nudged phases; raw free-to-nudged,
  matched-zero, zero-nudge drift and output-normalized measures are retained.
  Baseline Conv3 H1 raw displacement is 289.3 times the centered contrast,
  dominated by continued relaxation. The sigma/sqrt(2) band is a voltage
  noise reference, not an exact gradient SNR or failure threshold.
- Limitations: source beta and trained weights differ; a clean-checkpoint
  readout sweep does not isolate amplification or reproduce noisy training
  trajectories. Within-batch/draw spread is not training-seed uncertainty.
  Finite-K BPTT is the reference, not a convergence certificate: Conv3
  baseline has 72/144 failing post-T batch-layer residual-p90 checks at .01
  (largest p90 .26808), while subsequent endpoint p90 checks pass. Source
  beta qualification and seed-confirmation caveats remain; no cases are dropped.
- Conv3 magnitude follow-up (same replay, no new compute): mean clean EP
  per-weight RMS for Conv1/Conv2/Conv3/readout is
  4.451e-5/3.168e-5/1.810e-4/6.572e-5 (baseline),
  5.936e-5/3.547e-5/7.267e-5/1.518e-4 (balanced), and
  3.270e-4/1.035e-4/1.435e-4/1.065e-3 (legacy). Clean BPTT mean norms
  agree within .108%. Legacy's clean gradients are not uniformly smaller;
  at sigma5e-4 its mean paired noisy/clean EP norm ratios are
  627,987/157,530/1,290/1.016. These are raw gradients, not Adam updates.
  All 17,712 Conv3 comparisons reaggregate consistently into 72 cells;
  [report, tables and figure](../paper_ready_results/section43_conv3_gradient_magnitudes.md).
- Trained Conv3 displacement follow-up: pooled RMS of `(v_plus-v_minus)/2`
  at H1/H2/H3/output is 4.073e-7/2.119e-6/3.971e-5/.001845 (baseline),
  1.577e-7/2.455e-6/1.889e-4/.08672 (balanced), and
  2.919e-10/7.508e-9/9.570e-7/.008611 (legacy). Legacy hidden contrasts
  are 540/327/197× smaller than balanced. Raw baseline H1 free-to-positive
  movement is 1.179e-4, dominated by zero-nudge drift; centered and matched-zero
  contrasts are about 4.073e-7. All 2,592 raw rows reaggregate into 72 cells,
  with source-bundle and checkpoint hashes verified; no new compute.
  [Focused two-definition figure and tables](../paper_ready_results/section43_conv3_trained_displacement.md).
- Artifacts: [report](../paper_ready_results/section43_mechanism.md),
  [cosine figure](../paper_ready_results/figures/section43_gradient_cosine.pdf),
  [phase-contrast figure](../paper_ready_results/figures/section43_phase_displacement.pdf),
  [captions](../paper_ready_results/section43_mechanism_captions.tex),
  [verification](../paper_ready_results/section43_mechanism_verification.json),
  and [canonical replay](../results/section43-eqprop-mechanism-20260918-v1/full-1/).
  Both figures also have JPG/PNG/SVG exports and supporting CSVs.

## `eqprop-read-noise-seed0-baseline-20260916-v1` — Baseline endpoint read-noise training

- Manuscript integration: pulled the latest Overleaf revision and completed
  Table 3 in `bidir_paper_theory_revised.tex`, pushed as `1d160a9` on
  2026-09-18. The table preserves best-checkpoint validation accuracy,
  includes both Conv1 baseline betas and matching clean controls, and covers
  all50 noisy trainings. All60 accuracy cells were verified against collected
  metrics; table structure checks pass. No local LaTeX compiler was available.
- Analyzed: 2026-09-18. Outcome: **mixed** by depth: negligible observed
  sensitivity at Conv1/2 and increasing degradation at Conv3. Evidence class:
  `ordinary_mnist_eqprop_read_noise_training_diagnostic`; validation only.
- Question: how do baseline Conv1/2/3 trainings respond to endpoint-voltage
  noise under the user-selected betas and fixed T/K, compared descriptively
  with the completed ours/legacy curves?
- Frozen setup: centered frozen-current float64 EqProp, baseline ampV/C1/1,
  perfect diode, wide conductances [0,100], exact-zero frozen biases, original
  shared initializer and unchanged exact Adam rates. Conv1 beta100/200,
  Conv2 beta100, Conv3 beta10; T=K4/6/8,10/30/30 epochs, train/validation
  batch16/64, ordinary MNIST55k/5k. Training seed0 and noise seed2026081601.
  Independent positive/negative endpoint read noise affects gradient readout
  only; sigma1e-5/3e-5/1e-4/3e-4/5e-4. Inputs, relaxation and validation are
  noiseless. Maximum validation accuracy selects the best checkpoint.
- Coverage: all22 declared trainings are included, comprising20 noisy runs
  and2 new clean controls (Conv1 beta200 and Conv3 beta10). Audited historical
  beta100 controls are reused for Conv1/2. All440 epochs/1,512,720 optimizer
  steps complete. Final validation accuracy (%), in increasing sigma order:

  | Architecture / beta | Clean | 1e-5 | 3e-5 | 1e-4 | 3e-4 | 5e-4 |
  |---|---:|---:|---:|---:|---:|---:|
  | Conv1 /100 |96.04|96.04|96.04|96.00|96.02|96.04|
  | Conv1 /200 |96.02|96.02|96.02|96.02|96.02|96.04|
  | Conv2 /100 |97.22|97.30|97.38|97.36|97.34|97.24|
  | Conv3 /10 |97.64|97.48|97.04|96.64|96.48|96.04|

- Interpretation: Conv1's largest observed final loss is.04pp; Conv2 shows
  no observed loss against its historical clean reference. Conv3's loss rises
  from.16 to1.60pp. At sigma5e-4, Conv3 baseline/ours/legacy have final
  accuracies96.04/96.60/75.44% and clean-relative losses1.60/2.04/23.34pp.
  Ours retains higher absolute accuracy; baseline has less observed loss
  from its own control. These are descriptions of inherited contracts, not
  an isolated causal test of amplification or statistical equivalence.
- Limitations: one training/noise seed; different beta values, rates and
  GPU/software noise streams across some comparisons. Conv1's beta contrast
  crosses3080/3090 groups; Conv1/2 beta100 clean references use historical
  environments. All Conv3 baseline cases share Fifi, including their new
  beta10 control. Conv3 beta10's seed1 gradient-confirmation failure and T8
  free-state residual caveat remain; this study does not establish universal
  beta qualification. The older beta100 Conv3 score is not its noise control.
  Official-test reads are zero; none of these accuracies is paper-test evidence.
- Verification: all22 source and collected bundles pass full revalidation,
  including configs/source identity, initializer/cohort/order, complete epochs
  and noise draws, finite float64 checkpoints, zero biases, conductance bounds
  and PT/NPZ agreement. Remote production trees match local checksums. All39
  smokes and6 timing runs validate. All queues exit0; final Fifi completion is
  September18,00:39:15 CEST, with its GPU released. Three consecutive pairs
  retain the requested two-worker concurrency. Settled usage43.294845/60
  physical GPU-hours. No full training failed, was excluded or replaced;
  the pretraining Fifi wrapper quoting failure and corrected replacement are
  retained. The separate clean campaign remains paused at202/216.
- Evidence: [completed report](../paper_ready_results/baseline_read_noise_results_20260916.md),
  [per-run metrics](../paper_ready_results/baseline_read_noise_run_status_20260916.csv),
  [all-scheme comparison](../paper_ready_results/read_noise_all_schemes_validation_20260916.png),
  [validated bundles](../paper_ready_results/bundles/baseline_read_noise_20260916/),
  [coverage and terminal proof](../paper_ready_results/provenance/baseline_read_noise_closeout_20260918.json),
  [exact launch contract](eqprop_baseline_read_noise_launch_plan_20260916.md),
  [source/logs/receipts](../results/eqprop-read-noise-seed0-baseline-20260916-v1/).

## `digital-relu-table12-20260917-v1` — Half-channel digital ReLU references

- Manuscript inclusion: Table 1 alone retains the three MSE aggregates
  (nine trainings); CE results remain supporting evidence. Applied to the
  current Overleaf revision and pushed as `8a889d7`; Tables 2/3 unchanged. DRNs also use
  squared error, with paired-output encoding and a different class reduction.

- Analyzed: 2026-09-17. Outcome: **mixed** for loss ordering; all requested
  reference values are available. Evidence class: `ordinary_mnist_selection`.
- Frozen setup: Conv1/2/3 channels 32, 32/64, 32/64/128 with the DRN convolution
  geometry, single signed normalized input at unit gain, ten logits, signed
  unconstrained PyTorch-default weights, and every bias initialized at zero
  with LR 0. Adam weight LR .001, batch 16, 10/30/30 epochs, model/shuffle seeds 0–2.
  Both losses use the exact DRN train/validation split and minibatch orders;
  each MSE/CE pair starts from identical tensors. MSE uses one-hot 0/1 targets
  averaged over classes/examples; CE uses raw logits. No LR search.
- Best validation accuracy, mean ± sample SD (%): Conv1 MSE **96.47±0.12**,
  CE **97.59±0.09**; Conv2 MSE **98.19±0.16**, CE **98.38±0.11**;
  Conv3 MSE **99.01±0.03**, CE **98.77±0.08**. CE has the higher observed
  mean at depths 1/2, MSE at depth 3; this fixed-rate comparison does not select
  one uniformly superior loss or establish statistical significance.
- All 18/18 full runs, 420 epochs and 1,443,960 optimizer steps validate.
  Initial/best/final parameters and final Adam state are finite; biases are
  exactly zero. Independent replay reproduces all 18 selected-checkpoint
  accuracies exactly. Akib/Trex terminal trees are checksum-identical locally,
  all three launchers exited 0, and production occupied .4975 GPU-hours.
- No full training failed, was excluded or retried. Ten same-path smokes passed;
  one initial local py312 cuDNN failure before an optimizer step is retained at
  `results/digital-relu-table12-20260917-v1/smoke/conv1_mse_seed0` and replaced
  operationally by the unchanged six-case local py309 smoke. Conv1/2 use
  torch 2.5.1 on local/Akib and Conv3 torch 2.11.0 on Trex; each complete depth
  surface stayed on one host. Unrelated Trex contention is recorded and was
  not disturbed. Seven CPU regression checks passed.
- Interpretation/limits: these are conventional digital references with half
  the hidden channels, not an isolated amplification ablation or equal
  parameter-count comparison. Input gain, loss/output representation,
  initialization, signed support and dynamics differ from the DRNs. Only Table 1
  includes the unconstrained digital reference; there is no digital
  weight-bound mapping. Official-test reads remain zero.
- Authoritative local evidence:
  [`results/digital-relu-table12-20260917-v1/`](../results/digital-relu-table12-20260917-v1/).
  [Report and verification](../paper_ready_results/digital_relu_table12_20260917.md),
  [aggregate CSV](../paper_ready_results/digital_relu_table12_20260917.csv),
  [per-seed CSV](../paper_ready_results/digital_relu_table12_per_seed_20260917.csv),
  [frozen plan](digital_relu_table12_plan_20260917.md). The Table 1 input
  and comparison description are updated on the current Overleaf version. No local TeX engine was
  available; table structure and every numeric cell were checked in source.


## `eqprop-phase-displacement-t-sweep-20260916-v1` — T sensitivity of residual phase drift

- Analyzed: 2026-09-16. Outcome: **positive** for the requested mechanism
  question: longer free-phase relaxation removes the problematic drift while
  leaving controlled nudging-response RMS essentially unchanged.
- Cases: Conv3 baseline/legacy initialization and best BPTT checkpoints at
  T8/10/12/16/24, fixed K8 and injected beta10/.001; Conv2 legacy initialization
  at T6/8/10/12/16, fixed K6 and beta.03. Only cases with a previous hidden-layer
  raw/control RMS difference >=1% are included. Seed0, same36 batches of16,
  wide [0,100], exact-zero biases and float64 centered frozen-current EqProp.
  Source checkpoints, input gains, beta and K remain fixed along each curve.
- Measurement: add direct zero-nudge drift RMS(Z_K−F_T) to the previous
  positive/negative raw and matched-zero displacement comparisons. Pool by
  element count and retain batchwise ratios. Raw/control RMS close to one is
  not sufficient to infer small drift because the vector terms can differ
  in direction. The direct drift/response ratio avoids that ambiguity.
- Main result: trained Conv3 baseline H1 drift/response falls from224.456 at
  T8 to15.107 at T10,1.0673 at T12,.005871 at T16 and2.451e-7 at T24. T12
  passes the separate equilibrium-residual gate but still has drift comparable
  to the signal. At T16 pooled drift is.587%, but worst-batch drift is1.30%.
- Descriptive <=1% criterion, sustained at larger tested T, all layers and both
  signs: smallest pooled/every-batch T is12/12 for Conv3 baseline initialization,
  16/24 for Conv3 baseline best,16/16 for Conv3 legacy initialization,10/12 for
  Conv3 legacy best, and10/10 for Conv2 legacy initialization. These are tested
  grid points, not exact minimum T values or training qualification gates.
- Controlled-response RMS changes by **<.001%** across every observed layer,
  sign and checkpoint relative to the largest tested T (maximum.000915105%).
  Thus the large raw-displacement changes arise from residual relaxation.
  At trained Conv3 legacy T24 the measured drift reaches exact float64 zero;
  finite-step numerical stationarity does not establish arbitrary-time or
  exact-real convergence.
- Scope limits: one seed and fixed saved checkpoints; no retraining, noise
  injection or official-test read. All15 included gradient checks pass;
  baseline T8/T10 retain residual failures. Conv3 baseline beta10's prior
  seed1 confirmation failure remains unresolved. No training T or beta is
  adopted, and unaffected schemes were not assigned a direct-drift pass.
- Coverage/integrity: all15 included bundles and both smokes validate, with
  900 included replays,3,420 gradient comparisons,20,520 individual-layer rows,
  570 pooled contexts and190 sign-specific comparisons. Cohort, source-byte,
  zero-bias and finite/dtype guards pass. All three native-T cases reproduce
  previous shared measurements exactly; eight focused checks pass.
- Recovery: `runs/conv2_legacy_seed0_T6` completed36 replays but failed its
  inherited two-role completion-count check. It remains preserved without
  a successful result. Corrected metadata in `recovery_v2/` gives36/108 counts;
  `runs/conv2_legacy_seed0_T6_v2` replaces that attempt. Four unstarted Conv2
  configs received the same correction. A new regression test passes;
  numerical source and science are unchanged and two completed Conv3 cases
  were retained. Original/recovery inputs and the first driver are preserved.
- Cost: local RTX3090; recovery driver1264547 exited0 at14:58 CEST, GPU idle.
  **.460455/1 GPU-hours** settled, including the failed attempt and both smokes.
- Artifacts: [report and plots](../paper_ready_results/phase_t_sweep_20260916.md),
  [per-layer drift/response CSV](../paper_ready_results/phase_t_sweep_20260916_drift_response_comparison.csv),
  [all local evidence](../results/eqprop-phase-displacement-t-sweep-20260916-v1/),
  [completed plan and recovery](eqprop_phase_displacement_t_sweep_plan_20260916.md).
- Subsequent gradient review and user decision (2026-09-16): **retain T=K=4,
  6,8 for Conv1,Conv2,Conv3 across baseline,legacy,ours**, respectively, for
  the present wide-weight comparisons. The tentative Conv3 baseline T12/K8
  choice is not adopted. The descriptive1% drift target is not a training
  gate. For trained Conv3 baseline, increasing T8 to24 reduces drift/response
  from224.456 to2.451e-7 while minimum gradient cosine remains about.999944
  and maximum norm discrepancy changes only from.00137827 to.00127581.
  Centered subtraction cancels the sign-independent component but does not
  prove that incomplete relaxation has no gradient effect. Direct agreement
  with the matched finite-K BPTT reference is the relevant evidence here;
  all15 included configurations pass that check. The baseline residual and
  seed1 beta failures and Conv3 ours gradient exception remain labeled.
  No training resumes and historical bounded-run settings remain distinct.
  [Interpretation and decision report](../paper_ready_results/eqprop_drift_gradient_report_20260916.md)
  and [25 checkpoint/T gradient comparisons](../paper_ready_results/eqprop_drift_gradient_comparison_20260916.csv).

## `eqprop-phase-displacement-conv123-20260916-v1` — per-layer free/nudged displacement

- Analyzed: 2026-09-16. Outcome: **mixed**; clean read-only mechanism evidence.
  Filip requested all three amplification schemes on Conv1, Conv2 and Conv3
  after the fixed-T8/K8 Conv3 beta sweep.
- Cases: all nine architecture/scheme combinations, seed0 initialization and
  saved BPTT best-validation checkpoints; 36 matched batches of16, wide
  [0,100], exact-zero biases and float64 centered frozen-current EqProp.
  Native T/K is4/4,6/6,8/8. Injected betas (baseline/ours/legacy) are
  100/30/3, 100/10/.03 and10/3/.001. Conv3 baseline beta10 is a seed0
  candidate that failed seed1 confirmation; it is not promoted for training.
- Measurements: every hidden/output layer, both nudge signs relative to the
  common post-T free state and a matched K-step zero-nudge endpoint, plus
  positive-minus-negative span. RMS and relative displacement are pooled
  from sums of squares and element counts. Signed means, voltage RMS/spread,
  extrema, active-set transitions and cohort percentiles are retained.
- Main finding: trained Conv3 baseline H1 raw RMS displacement is1.02807e-4
  versus4.57975e-7 relative to the matched zero-nudge endpoint: **224.5×**.
  H2 is8.10819e-5 versus2.44316e-6 (**33.2×**). Continued relaxation dominates
  these raw free-to-nudged differences at T8. The matched endpoint comparison
  does not remove the separate free-state equilibrium qualification failure.
- Scheme comparison: at trained Conv3 checkpoints, ours exceeds legacy's
  matched-zero positive-nudge RMS by about545×/321×/191×/9.88× at H1/H2/H3/output.
  Baseline H1 is3.50× ours, so there is no uniform ours-over-baseline ordering.
  These are comparisons under the declared different beta/training contracts,
  not an isolated amplification effect or a read-noise accuracy prediction.
- Qualification and limits: seed0 diagnostic gates pass in seven cases;
  Conv3 baseline fails equilibrium and Conv3 ours fails gradient fidelity.
  Both remain included and labeled. Only one model seed and initialization/
  best checkpoints are observed; no training trajectory or seed uncertainty
  is inferred. No endpoint noise, optimizer step or official-test read.
- Integrity: all nine formal local bundles and smoke validate, covering648
  replays,1,944 gradient comparisons,9,720 individual-layer rows and270 pooled
  layer/context rows. Cohort payloads match across all cases; source bytes
  remain unchanged. Added voltage statistics reproduce the prior smoke's
  shared gradient, residual and displacement measurements exactly. No missing
  cases, operational failures, exclusions or replacement directories.
- Cost: one local RTX3090; driver1239772 exited0 at14:02 CEST, GPU released.
  **.202105/1 GPU-hours** settled, including smoke; no training resumed.
- Artifacts: [report with all per-layer tables and figures](../paper_ready_results/phase_displacement_20260916.md),
  [pooled CSV](../paper_ready_results/phase_displacement_20260916_layer_displacement_summary.csv),
  [checkpoint provenance](../paper_ready_results/phase_displacement_20260916_sources.csv),
  [full evidence](../results/eqprop-phase-displacement-conv123-20260916-v1/),
  [execution plan](eqprop_phase_displacement_plan_20260916.md).

## `eqprop-conv3-baseline-beta-tk8-20260916-v1` — Conv3 baseline beta sweep

- Analyzed: 2026-09-16. Outcome: **mixed**; clean read-only gradient-selection
  evidence. Filip explicitly retains T=K=8 and the separate equilibrium caveat.
- Cases: wide Conv3 baseline beta10/30/50/75, with the unchanged beta100
  reference reused. Seed0 initialization and BPTT best checkpoint,36 matched
  batches of16. Float64 centered frozen-current EqProp, exact-zero biases,
  same-state BPTT, unchanged source/LR/input/cohort contracts.
- Selection: beta10 passes (worst cosine .994886, norm mismatch .057006).
  Betas30/50/75/100 fail respectively1/2/5/7 of288 layer/batch comparisons,
  all at initialization; trained gradients pass. Beta10 is frozen before
  confirmation on seeds0/1/2 with32 fresh batches plus the four regressions.
- Confirmation: seeds0/2 pass (worst cosine .991247/.992655), but seed1 fails
  two initialization ConvWeight_0 rows: batch16 norm mismatch .101208, and
  batch29 cosine .968773. No beta is confirmed across all three seeds and
  no fallback is tuned on that confirmation cohort. Beta10 remains a seed-0
  passing candidate, not a promoted common training beta.
- Equilibrium and interpretation: trained free-state residuals fail at T8;
  seed0 values are identical across beta, max p90 .188054. Confirmation maxima
  are .195286/.259774/1.858277. Thresholds are preserved. Lower beta improves
  seed0 gradient fidelity but this grid does not deliver a robust three-seed
  choice. Full-training stability and values below10 remain untested here.
- Integrity and coverage: all7 new formal bundles and smoke validate;
  504 new replays/2,016 layer comparisons,10 retained failures. Reused100
  is separate. Matched selection payloads, source immutability, finite
  float64/zero-bias guards pass; no optimizer step or official-test access.
- Cost: local RTX3090, driver1230581 exited0; **.275828/1 GPU-hours** settled.
  The requested displacement comparison follows separately, using candidate
  beta10 for seed0 with its confirmation failure retained. Training is paused.
- Artifacts: [report and figure](../paper_ready_results/conv3_baseline_beta_tk8_20260916.md),
  [case table](../paper_ready_results/conv3_baseline_beta_tk8_20260916.csv),
  [full evidence](../results/eqprop-conv3-baseline-beta-tk8-20260916-v1/),
  [plan](eqprop_conv3_baseline_beta_tk8_plan_20260916.md).

## `eqprop-baseline-wide-t-relaxation-20260916-v1` — Conv3 baseline T-only audit

- Analyzed: 2026-09-16. Outcome: **mixed**; clean read-only gradient and
  equilibrium diagnostics, not training or official-test performance.
- Question: Before baseline EqProp read-noise training, can increased free
  relaxation alone make the wide Conv3 baseline beta100 setting credible?
  Filip directed increasing T first; this study keeps K=8 and beta100 fixed.
- Setup: wide [0,100], exact-zero biases, centered frozen-current float64
  EqProp versus identical-state, same-T/K BPTT. Four new seed-0 cases at
  T12/16/24/32 use the same36 batches of16 at the verified initializer and
  BPTT best checkpoint as the preserved T8 reference. Source checkpoints
  were trained at T8/K8; larger-T replay is explicitly diagnostic.
- Measurements: trained free-state worst residual p90 falls from **.188054**
  at T8 to **.00111043, 6.71346e-6, 3.33387e-10, 2.36469e-11** at the four
  larger T values. Every new case passes all equilibrium checks. The worst
  initialization cosine stays **.967393** and maximum norm mismatch **.184260**,
  failing the .99/.10 criteria at every T. The same seven ConvWeight_0 batch
  comparisons fail per case; all trained gradient comparisons pass.
- Interpretation: longer free relaxation fixes the observed trained
  equilibrium failure, but does not fix beta100 initialization gradient
  fidelity. T12 is the smallest tested residual-passing extension, not a
  fully qualified operating point. No T passes both gates, so the predeclared
  three-seed confirmation is omitted; its fresh cohort remains unmeasured.
  A lower-beta check at T12/K8 (starting with10/30) is the proposed next
  diagnostic, not a launched or qualified replacement. No beta is changed.
- Coverage and integrity: all4 new formal bundles plus the separate smoke
  validate. New formal work is288 checkpoint/batch replays and1,152 layer
  comparisons, including28 retained failures. The reused T8 reference is
  excluded from those counts. Exact selection payload matching, immutable
  source/config/checkpoints, float64/zero-bias guards and15 focused tests pass.
  No operational failure, optimizer step or official-test read occurred.
- Limits and next work: seed0 and two checkpoint roles do not establish
  full-training stability or a three-seed result. K and beta were not swept.
  Any longer-T noisy training requires dependent operating-point/LR checks
  and a matching clean training control; earlier T8 accuracies cannot supply
  that control. Cross-scheme T/compute differences must remain explicit.
- Execution: local RTX3090, driver1218622 exited0 after597.733 seconds;
  **.166037/2 physical GPU-hours** settled in a separate baseline-noise
  preparation allowance. Local GPU is idle; clean training remains paused.
- Artifacts: [report and figure](../paper_ready_results/baseline_wide_t_audit_20260916.md),
  [measurements](../paper_ready_results/baseline_wide_t_audit_20260916.csv),
  [full evidence](../results/eqprop-baseline-wide-t-relaxation-20260916-v1/),
  [execution plan](eqprop_baseline_read_noise_beta_plan_20260916.md).

## `eqprop-read-noise-seed0-ours-legacy-20260914-v1` — completed single-seed noise sweep

- Analyzed: 2026-09-16. Outcome: **mixed**. All **30/30** declared full
  trainings are collected and validated; no run is active, missing or excluded.
  Evidence class: ordinary-MNIST validation robustness, not official-test
  performance. This interpretation is the agent's assessment under the
  repository's standing scientific authority.
- Question and gate: How do ours and legacy training respond to Gaussian
  endpoint-voltage read noise under their inherited EqProp contracts? The
  completion gate is all 30 full trainings, local canonical/checkpoint
  validation and reconciled remote receipts.
- Frozen setup: Conv1/2/3 × ours/legacy × sigma
  `1e-5,3e-5,1e-4,3e-4,5e-4`, training seed 0 and noise seed 2026081601.
  Wide [0,100] weights, exact-zero frozen biases, centered frozen-current
  float64 EqProp, unchanged Adam vectors, 10/30/30 epochs, T/K 4/4,6/6,8/8,
  batch 16, deterministic ordinary-MNIST 55k/5k split and matched complete
  minibatch orders. Injected beta is ours 30/10/3 and legacy 3/.03/.001.
  Noise affects gradient readout only; sigma is in simulator voltage units.
  Source-v2 SHA-256:
  `4055090e16606583c276b06c3aac59396e87dcedf58b8cde9be94c5824948fb1`.
- Measurements, in increasing sigma order: Conv1 final accuracy changes
  by at most .08 pp from its historical clean reference. Conv2 ours final is
  **98.08/98.08/98.00/97.80/97.60%**, versus legacy
  **97.62/97.38/96.52/95.94/95.72%**. Conv3 ours final is
  **98.26/97.82/97.62/96.82/96.60%**, versus legacy
  **95.68/93.42/83.08/82.34/75.44%**. At 5e-4, clean-relative final losses
  are **.42 vs 2.34 pp** for Conv2 and **2.04 vs 23.34 pp** for Conv3.
  Conv3 legacy at 5e-4 reaches a best 76.36%, with epoch validation values
  spanning 31.26–76.36%; all checkpoints remain finite.
- Interpretation: Conv1 changes little across this grid at this seed.
  Conv2 supports better observed robustness for ours under the inherited
  contracts at every noise level. Conv3 shows a much larger legacy loss,
  but its comparisons cross environments. The broad robustness hypothesis
  is therefore supported descriptively for Conv2/3, while a causal claim
  isolating amplification remains unresolved. Low accuracy and oscillation
  are included scientific outcomes, not operational failures.
- Limits: One training/noise seed cannot quantify seed-to-seed uncertainty;
  differing beta choices prevent isolating amplification. Conv3 ours retains
  its inherited beta-3 clean-gradient qualification exception. Clean controls
  ran on Jean Zay/PyTorch 2.5.0: initializer/cohort/order match, but environment
  differs, so small clean-relative differences cannot be attributed uniquely
  to noise. Audited CUDA samples match within the 5090 group and within
  local/Nom; Akib is a third group. Every Conv1/2 within-sigma pair shares a
  host. All five Conv3 pairs differ in environment and noise realization:
  at 1e-5/1e-4/5e-4 ours uses 5090 and legacy uses Nom; at 3e-5/3e-4
  ours uses local/Akib and legacy uses Trex/Fifi after the authorized deadline
  transfers. The figure distinguishes groups and connects within groups only.
- Execution and inclusion: The first 12 completed by September 15,
  **05:52:58 CEST**; the final continuation run completed September 16,
  **06:20:46 CEST**, 99 minutes before the 08:00 deadline. All five
  continuation GPUs are released. All full training packs exited 0; none
  was discarded or retried. The preserved local nohup attempt ended before
  training and was replaced by unchanged tmux execution. Local/Akib queues
  intentionally exited 1 at documented output guards after their retained
  full trainings, preventing duplicates of the transferred Trex/Fifi cases.
- Integrity and cost: All 30 included bundles pass canonical/source/config,
  initializer/cohort/order, full-epoch/noise-count, finite-float64 checkpoint,
  zero-bias, bounds and PT/NPZ equality checks. All remote production trees
  and terminal receipts match authoritative local copies by checksum.
  First window: **22.142636/36 physical GPU-hours**. Continuation:
  **55.089836/78**, including .341005 hours of checks/probes/recovery.
  Total: **77.232472 physical GPU-hours**. Official-test evaluations: zero.
- Remaining decisions: No required run remains within this sweep. For a
  future causal study, resolve the inherited beta qualification issue and
  match Conv3 environment/noise draws, then add seeds to quantify variability.
  These follow-ups are not launched. Baseline, sigma 0/1e-3, extra seeds,
  official-test evaluation and the paused clean campaign remain outside scope.
- Artifacts: [full results and interpretation](../paper_ready_results/read_noise_sweep_results.md),
  [all 30 cases and bundle links](../paper_ready_results/read_noise_run_status_20260914.md),
  [final figure](../paper_ready_results/read_noise_validation.png),
  [collection proofs](../paper_ready_results/provenance/read_noise_collection_20260914.json),
  [final remote reconciliation](../paper_ready_results/provenance/read_noise_final_reconciliation_20260916.json),
  [final GPU accounting](../paper_ready_results/provenance/read_noise_gpu_budget_continuation_20260916.json),
  [raw evidence](../results/eqprop-read-noise-seed0-ours-legacy-20260914-v1/),
  [completed continuation record](eqprop_read_noise_continuation_plan_20260915.md),
  [preserved first-window report](../paper_ready_results/read_noise_overnight_results_20260915.md).

## `eqprop-beta-selection-audit-20260914-v1` — larger-cohort baseline beta selection

- Analyzed: 2026-09-14. Outcome: **mixed**; ordinary-MNIST read-only
  gradient-selection evidence, not training or official-test performance.
- Question: Do wide Conv1/2/3 baseline injected betas 100/200/300 satisfy the
  unchanged cosine >= .99 and symmetric norm difference <= .10 criteria with
  broader batch and model-seed coverage?
- Controls: centered frozen-current float64 EqProp versus matched same-T/K
  BPTT, native 4/4, 6/6, 8/8, exact-zero biases, original Adam checkpoint/LR
  contracts, initialization plus best-validation checkpoints. Selection uses
  seed 0 and 32 new batches of16 plus the original four. Freeze the largest
  passing candidate before 32 disjoint confirmation batches plus the original
  four on seeds 0/1/2. Other schemes, bounds, rates and training remain unchanged.
- Coverage: all 9 selection and 6 conditionally required confirmation cases
  complete locally and validate, with 1,080 replays and 3,024 layer comparisons.
  All 72 historical baseline beta 100 comparisons reproduce exactly in input
  payload, cosine and norm mismatch. No optimizer step or official-test read.
- Conv1: all three candidates pass seed 0 selection, but nominated beta 300
  fails confirmation on seed 2 at the trained C0 checkpoint: three new batches
  have cosines .976462, .986547, .987500. Seeds0/1 pass. Beta 100/200 have only
  seed 0 selection coverage in this audit; neither is relabeled as confirmed.
- Conv2: beta 100 passes selection and all three confirmation seeds, with worst
  confirmation cosine .991173 and maximum norm mismatch .060712. Betas200/300
  fail selection in 3/216 and 7/216 comparisons (worst cosine .984396/.973355).
- Conv3: beta 100/200/300 fail in 7/288, 16/288, 29/288 selection comparisons,
  with worst initial cosine .967393/.940800/.931421 and maximum norm mismatch
  .184260/.261803/.286851. All failures of the direct gradient gate occur at
  initialization; trained gradient checks pass. The separate trained free-state
  residual limitation at T=8 persists. No candidate enters confirmation.
- Interpretation: additional batches and seeds expose failures hidden by the
  smaller cohort. This audit supports retaining wide Conv2 beta 100 and does
  not support promoting Conv1 beta 300 or increasing Conv3 beta in this grid.
  No automatic fallback uses the now-observed confirmation data, and no new
  training beta is adopted. Conv1 lower candidates would require a separately
  declared confirmation round; Conv3 remains unresolved under this grid/T/K.
- Paper review, 2026-09-16: Filip requested preserving the wide Conv3 baseline
  qualification caveat and raised **using only BPTT for the bounded-weight
  results** as a possible paper scope. This remains tentative; no EqProp
  results are excluded and the clean completion contract is unchanged. The
  wide T8/K8, beta-100 failure and bounded T24/K8, beta-.1 qualification are
  separate evidence. A BPTT-only bounded comparison would not resolve the
  wide EqProp caveat or establish equilibrium convergence. See the
  [recorded scope discussion and settings](paper_ready_results_manifest.md#baseline-qualification-caveat-and-possible-bptt-only-scope-2026-09-16).
- Limitations: sampled initial/best checkpoints rather than complete training
  trajectories; validation data were previously used for checkpoint selection.
  Numerical qualification does not establish full-training stability. The
  initial smoke's stale runtime path was rejected before computation; its log
  and configs are retained, and the corrected smoke/production use verified
  frozen source. Fifteen focused tests pass. Main is idle after completion;
  .279801 GPU-hours settled within the 2-hour reservation. Training remains paused.
- Artifacts: [report and figures](../paper_ready_results/baseline_beta_audit_20260914.md),
  [case table](../paper_ready_results/baseline_beta_audit_20260914.csv),
  [layer/cohort table](../paper_ready_results/baseline_beta_layer_cohort_summary_20260914.csv),
  [complete raw and analysis evidence](../results/eqprop-beta-selection-audit-20260914-v1/),
  [execution plan](eqprop_baseline_beta_audit_20260914.md).

## Active paper completion: collected training and EqProp pilot evidence (2026-09-11–13)

Evening paper review, September 14: **202/216** current-contract training
results are now collected and validated, with **14 full trainings remaining**.
The newly collected Conv3 baseline EqProp seed-0/Gmax1e-4 pilot at T24/K8,
beta .1 completes 30 epochs at **77.68/77.68%** best/final validation, equal
to its BPTT control. Full split/order matching, exact initializer/config,
finite float64 checkpoints, zero biases, bounds, PT/NPZ equality, remote/local
checksums, and both terminal receipts pass. Its zero best-to-final drop passes
the pilot stability rule; Conv3 pilot coverage is **1/3**, so the other two
ceilings and all six repetitions remain required. The 6.6025 GPU-hours are
settled. Across the 97 completed algorithm pairs, the maximum best-validation
difference remains .66 pp; the maximum mean difference across 31 complete
three-seed condition pairs is .193 pp. This is partial validation evidence;
no official test has been read and no clean follow-up was launched.
See the [paper review](paper_experimental_review_20260914.md),
[pilot proof](../paper_ready_results/baseline_tk_revision/conv3_seed0_tight_pair_check_20260914.json),
and [review verification](../paper_ready_results/paper_review_verification_20260914.json).

September 14 update: the [61 locally validated baseline T/K replays](../paper_ready_results/baseline_tk_beta_qualification.md)
resolve the original numerical gate failures. Common beta .1 passes every
ceiling, checkpoint role and fixed validation batch at Conv2 T=12/K=6
(worst cosine .997674, maximum symmetric norm difference .019507) and Conv3
T=24/K=8 (.997096 and .071091). All projected-residual gates pass. Larger
beta candidates 100, 10 and 1 fail; Conv3 T=20/K=8 still fails the norm gate.
The native GPU controls reproduce the earlier CPU failures. For the tight
Conv2 ceiling, increasing T alone succeeds while increasing K alone fails,
supporting insufficient free-phase relaxation in that case.

These are diagnostic replays on previously trained checkpoints, not new
training outcomes. Filip authorized revised T/K for both BPTT and EqProp.
Eighteen older baseline BPTT runs are preserved but require matched
replacements. The [36-cell revision](../paper_ready_results/baseline_tk_revision/README.md)
requires new BPTT seed-0 best-checkpoint beta checks, full EqProp pilots,
and then replication. Exact learning rates, initializer/cohort/order, bounds,
seeds and training budgets remain unchanged. The revision initially added 36 runs alongside eight admitted original
repetitions. Twenty-one revised results and all 198 original bundles are collected: 201/216
cells satisfy the current contract and 15 full training completions remain. No official test has been read.

Main completed the final seed-1 Conv3 baseline BPTT ceiling at 13:15:08 CEST
on September 14: 89.58% best/final validation, 30 epochs, both terminal receipts
zero, and 2.166111 GPU-hours settled. All six Conv3 BPTT seed-0/1 cases are
collected. New admissions are paused for Filip’s review; the three current
EqProp cases may finish and stop at their run boundaries.

The original bounded Conv3 ours and legacy BPTT–EqProp comparisons now each
contain all 18 validated runs: two algorithms, three ceilings, and three seeds.
The largest paired best-validation difference is .66 pp for ours and .18 pp
for legacy, with no excluded seed. All nine pairs per scheme match initializers,
cohorts and all 30 minibatch orders. The original host/environment split is
retained in provenance. These validation trajectories support close agreement
under the frozen controls; three seeds do not establish statistical equivalence
or official-test performance. [Ours](../paper_ready_results/bounded_conv3_ours_three_seed_validation.md)
· [Legacy](../paper_ready_results/bounded_conv3_legacy_three_seed_validation.md).

The first revised Conv2 EqProp repetition (seed1/Gmax1e-4) is collected at
85.36/85.04% best/final validation, versus paired BPTT85.42/85.06%.
Its complete split and 30 epoch-order fingerprints match; all canonical,
finite-float64, zero-bias, bounds and PT/NPZ checks pass. The later middle seed-1 and tight seed-2 repetitions also complete at 91.50% and 85.22% best/final validation, each .02 pp below its paired BPTT. Both full cohort/order audits pass and costs settle at 2.243056 and 2.120000 GPU-hours. Three Conv2 EqProp repetitions remain outstanding; those hosts are now idle under the pause.

The September 14 agreement review compares new seeds 1/2 with 32 earlier reused seed-0 references, excluding each reused seed from the new-run
mean. Wide changes are at most .18 pp. The largest bounded change is Conv3
ours at Gmax1e-4: 84.54% earlier versus 85.43% for the new-seed mean (+.89 pp).
Fifteen same-seed original/revised baseline BPTT pairs match initialization,
exact learning rates, membership and all 30 minibatch orders; the largest
best-accuracy change is -.50 pp. Conv2 complete means change by -.06, .00,
+.09 pp; Conv3 remains partial. Across 96 completed algorithm pairs, the
largest BPTT–EqProp best-accuracy difference is .66 pp; the largest mean
difference among 31 complete three-seed conditions is .193 pp. These are
descriptive validation comparisons, not proof of equivalence or official-test
performance. [Full results, matched pairs and figure](../paper_ready_results/results_agreement_20260914.md).

The collected Nom timing diagnostic, with 256 training batches and one
validation batch, projects9.788hours for a full revised Conv3 EqProp case
(11.256hours with15% margin). This exceeds the current8h cap and is a
placement observation, not a training completion or accuracy result.
[Timing evidence](../results/paper-training-completion-20260911-v1/checks/nom_conv3_baseline_ep_timing_20260914/timing_estimate.json).

The first revised Conv3 baseline BPTT seed-1 repeat (Gmax=1e-4) is
collected and validated at 77.50/76.70% best/final validation after all
30 epochs. This remains a partial seed group; no three-seed aggregate is
reported for this ceiling yet.

The revised Conv3 baseline BPTT seed1 middle ceiling (Gmax=5e-4) is
also collected and validated after all30 epochs: 87.36/87.36% best/final
validation. Both terminal receipts are zero and its allocated runtime is
2.2725 GPU-hours. This is a second seed for that ceiling; seed2 is still
required before reporting a three-seed aggregate.

The middle-ceiling Conv3 ours EqProp seed-1 repeat on Loulou is collected
and validated at 94.28/94.28% best/final validation. Tight and middle
ceilings now have all three seeds; the final-ceiling repeat remains active.
The complete three-ceiling paired comparison awaits that last result.

The largest revised Conv3 BPTT reference is also collected and validated:
89.38/89.38% best/final validation, completing all three new Conv3 seed-0
references. Its full beta .1 gate passes all eight replays and 32 weight-layer
comparisons: minimum cosine .9970958, maximum symmetric norm difference
.0687167, all residual gates passing. Both Conv2 and Conv3 now have complete
three-ceiling numerical qualification at common beta .1. These are validation observations, not official-test results.

The Loulou packing benchmark used 10.1 GiB for two EqProp workers but gave
0.775x aggregate throughput, so the requested nom-cool-1 fallback is used.
The earlier Trex tight legacy seed-1 case is retained; the remaining ceilings
on Nom have an explicitly recorded host/environment split. Nom's middle-ceiling
legacy seed-1 result is collected and validated at 97.84/97.82% best/final
validation; the full three-ceiling comparison awaits its last case.

The first new Conv2 baseline BPTT reference (seed 0, Gmax=1e-4,
T=12/K=6) is now locally collected and validated after 30 epochs. Best/final
validation is 85.52/85.08%, versus 85.60/85.04% at the original T=K=6;
this single-seed difference is -.08/+.04 percentage points and is not a
cross-seed conclusion. Its [new-reference beta .1 recheck](../paper_ready_results/baseline_tk_revision/beta_qualification.md)
passes all eight replays and 24 layer comparisons, including both checkpoint
roles and all four batches: minimum cosine .9976736, maximum symmetric norm
difference .0195065, all residual gates passing. The middle ceiling (Gmax=5e-4) is now also collected after 30 epochs:
92.38/92.16% best/final validation, versus 92.44/92.32% under native T/K.
Its full beta .1 check passes all eight replays and 24 layer comparisons,
with all residual gates passing. The largest ceiling is also collected at 93.54/93.36% best/final validation,
and its full beta .1 check passes all eight replays and 24 layer comparisons.
All three revised Conv2 ceilings therefore qualify at common beta .1.
The three full seed-0 EqProp pilots remain required before seeds 1/2. These accuracy differences remain
single-seed validation observations. Official-test evaluations remain zero.

All three revised Conv2 baseline EqProp pilots are now collected and stable
at common beta .1, T=12/K=6. Best/final validation across ascending ceilings
is 85.42/85.00%, 92.54/92.30%, and 93.92/93.54%; drops .42/.24/.38pp pass
the strict five-point rule. All 30 epochs, finite float64 states/histories,
zero biases, bounds, shared initialization and PT/NPZ equality validate.
This supports the revised relaxation setting for full seed0 training and
releases the other seeds under the frozen group guard. Replication variability
remains unmeasured; official-test evaluations remain zero.
The [matched-cohort audit](../paper_ready_results/baseline_tk_revision/conv2_matched_cohort_check.json)
also confirms identical split and all30 epoch-order fingerprints for all
three completed BPTT/EqProp pilot pairs. All nine repetition smoke fingerprints
match their corresponding BPTT first epoch.

The first revised Conv2 baseline EqProp pilot (tight ceiling, seed 0,
T=12/K=6, beta .1) is collected and passes full stability: 85.42/85.00%
best/final validation, a .42-point drop. Its 30-epoch histories and float64
checkpoints are finite; initialization, zero biases, bounds and PT/NPZ equality
validate. The paired revised BPTT result is 85.52/85.08%, so the single-seed
EqProp-minus-BPTT difference is -.10/-.08 pp. This supports stability at the
tight ceiling; all three ceilings remain required before replication.
[Full-pilot report](../paper_ready_results/baseline_tk_revision/pilot_stability.md).

The middle-ceiling revised Conv2 EqProp seed-0 pilot is also collected and
stable: 92.54/92.30% best/final validation, a .24-point drop. All 30 epochs,
float64 checkpoint/history finiteness, zero biases, bounds, shared initializer
and PT/NPZ equality validate. Against matched revised BPTT 92.38/92.16%,
the single-seed difference is +.16/+.14 pp. Stable pilot coverage is 2/3;
replication still requires the largest ceiling at the same beta.

All three revised Conv2 BPTT seed-1 repetitions are collected and validated
from Jean Zay: best/final validation is 85.42/85.06%, 91.52/91.46%, and
93.62/93.62% across ascending ceilings. Their configs, source and shared
initializers match the revision. The three-seed aggregation awaits seed 2;
single-seed differences are not a completed cross-seed comparison.

All three revised Conv2 BPTT seed-2 ceilings are now collected and validated
at 85.24/85.24%, 92.22/92.22% and 93.50/93.40% best/final validation.
This completes all nine revised Conv2 BPTT runs. Across three seeds, best
validation mean ± sample SD is 85.39±.14%, 92.04±.46% and 93.55±.06% for
ascending ceilings; final means are 85.13%, 91.95% and 93.46%. These remain
ordinary-MNIST validation observations. All initializer, source and canonical
checks pass; no official test has been read. [Three-seed values](../paper_ready_results/baseline_tk_revision/conv2_bptt_three_seed_summary.json).

The first revised Conv3 baseline BPTT reference (seed 0, Gmax=1e-4,
T=24/K=8) is also collected and validated. Best/final validation is
77.68/77.68%, versus 77.66/77.66% at native T=K=8. Its new-reference beta .1
check passes all eight replays and 32 layer comparisons: minimum cosine
.9970958, maximum symmetric norm difference .0473293, with every residual
gate passing. This is 1/3 Conv3 ceilings and remains single-seed validation
evidence. The middle reference is now collected at 87.12/87.12% best/final
validation, versus 87.44/87.44% at native T/K (a single-seed -.32 pp
difference). Its full beta .1 recheck also passes all eight replays and
32 layer comparisons: minimum cosine .9970958, maximum symmetric norm
difference .0473293, every residual check passing. Conv3 is qualified at
two of three ceilings; the largest reference and full EqProp pilots remain
outstanding. Both full CPU replays leave source bytes unchanged and apply
no optimizer updates or official-test reads.

The three Conv3 ours EqProp seed-2 cases are now locally collected and
validated after all 30 epochs. Best validation across ascending ceilings is
85.60/95.04/95.88%; these remain validation evidence. Three-seed aggregation
waits for the remaining seed-1 cases on Loulou. The tight seed-1 case is
now collected at 85.18/85.18% best/final validation, with finite float64
checkpoints, zero biases, projection bounds and exact local/remote checksums
passing. The tight-ceiling three-seed condition is complete; the two larger
ceilings still lack seed 1.

Earlier completion evidence follows; its 190/216 counts and native-T/K
baseline qualifications describe the original inventory before this revision.

Study `paper-training-completion-20260911-v1` remains active. This is partial
ordinary-MNIST validation evidence; no official-test result is included.
The [continuing manifest](paper_ready_results_manifest.md) tracks coverage
and operational state under the [approved plan](paper_training_completion_launch_plan_20260911.md).
As of September 13 at 22:09 CEST, **190/216** training bundles are collected
and validated. This includes all 108 BPTT cases, all 27 wide EqProp cases,
all 27 bounded Conv1 EqProp cases, and all nine cases in each of bounded
Conv2 ours and legacy. The middle Conv3 ours pilot is now collected and
passes the complete 30-epoch gate at beta .003: 94.68/94.50% best/final
validation, decline .18 pp, finite float64 checkpoints/histories and zero
biases. Its exact copied bytes and shared initializer validate. The final
ceiling also passes at 95.72/95.72% best/final, zero decline, with all thirty
epochs, finite float64 checkpoints, zero biases, bounds and initializer
checks validated locally. All three Conv3 ours ceilings therefore pass the
full-training gate at injected beta .003. This completes all 21 qualified
seed-0 pilots and permits the remaining six ours repetitions, subject to
resource and budget admission. Eighteen baseline cells remain held after
their declared numerical ladders failed to qualify a common beta.
The [complete Conv2 ours comparison](../paper_ready_results/bounded_conv2_ours_three_seed_validation.md)
uses frozen injected beta .001. Mean best-validation EqProp minus BPTT
differences from tight to largest ceiling are -.007, -.033 and +.027 pp.
All nine initializer/cohort/order pairs match; over 270 matched epochs the
maximum accuracy difference is .46 pp, maximum paired best difference .08 pp,
and maximum EqProp final decline .34 pp. All three seeds are retained.
This supports consistency under the declared selection-split contract;
it does not establish equivalence or official-test performance.

Conv3 legacy's tight ceiling now has all three EqProp seeds: best/final
93.54/93.54% (seed 0), 94.84/94.72% (seed 1), and 93.20/93.20% (seed 2).
Both repetitions are stable and fully validated against their frozen
30-epoch contracts and shared initializers, with exact remote/local
checksums. The middle seed-2 repetition is also collected and stable at
98.00/97.88%, decline .12 pp, and its largest seed-2 repetition is complete
at 98.06/98.06%. All seed-2 ceilings are stable. Middle/largest ceilings still lack their
complete three-seed coverage. The
[global validation tables](../paper_ready_results/validation_tables.md)
have 61/72 complete three-seed conditions.

The corrected wide Conv1 legacy centered-float64 EP pilot at injected beta 3
passes its numerical gate (minimum cosine .9998157; maximum symmetric norm
difference .0022935) and ten-epoch stability rule: best/final validation
96.48/96.44%. With the eight retained pilots, the nine-condition wide pilot
block is locally validated and its replications are released.

The full wide Conv1 comparison now has all three seeds for both algorithms.
EqProp minus BPTT mean best-validation differences are +.013, -.040, and
-.007 percentage points for baseline, ours, and legacy; the largest paired
seed difference is .06 points. This supports consistency of the frozen
settings across these seeds, without establishing statistical equivalence
or test performance. See the [three-seed table](../paper_ready_results/wide_conv1_validation.md).

The wide Conv2 comparison is also complete for all three seeds and both
algorithms. Mean EqProp minus BPTT best-validation differences are -.047,
-.033, and +.040 percentage points for baseline, ours, and legacy; the largest
paired-seed difference is .14 points. All nine algorithm/seed pairs have
identical recorded training/validation cohorts and all 30 epoch minibatch
orders. This supports consistency on the selection split, with the same
equivalence and official-test limitations. See the
[Conv2 table and pairing audit](../paper_ready_results/wide_conv2_validation.md).

The wide Conv3 comparison is now complete: mean EqProp minus BPTT
best-validation differences are -.073, +.040 and +.020 pp for baseline,
ours and legacy, with a maximum paired-seed difference of .14 pp. All nine
pairs share recorded cohorts and all 30 minibatch orders. All six repeated
seed pairs have verified shared initializer assets and numerical BPTT
initial-state records. The retained baseline/ours seed-0 BPTT bundles lack
initial-state hashes, so exact initializer reuse remains an audit limitation.
The known wide Conv3 gradient/residual caveats also remain. This reports
training consistency on the selection split, not official-test eligibility.
See the [wide Conv3 table and pairing audit](../paper_ready_results/wide_conv3_validation.md).

For bounded Conv1 baseline and ours, injected betas .1 and .03 respectively
pass the gradient and residual gates at the shared initializer and matched
BPTT best checkpoints for all three ceilings. All six ten-epoch seed-0 pilots
are collected and finite with exact-zero biases. The maximum best-to-final
drop is 2.44 percentage points, below the declared strict 5-point limit.
Across the 60 matched epoch pairs, EP and BPTT validation accuracies differ by
at most .06 percentage points. The tight-ceiling decline appears in both
algorithms, supporting the selected beta settings for these pilots. This
does not establish three-seed equivalence. See the
[pilot report and curves](../paper_ready_results/bounded_ep_pilot_validation.md).

Bounded Conv1 legacy also passes numerical qualification at injected beta
.03. Its three full seed-0 pilots are now collected and stable, with a maximum
best-to-final drop of .74 percentage points. All nine Conv1 seed-0 pilots
therefore pass, releasing the three qualified groups for seeds 1/2 under
the September 12 authorization and the remaining budget. Across all nine
Conv1 pairs (90 matched epochs), the maximum absolute EP–BPTT validation
difference is .08 percentage points. This supports training consistency at
the qualified beta values for seed 0. All eighteen seed-1/2 repetitions are
now collected and stable; maximum drops are .86/.40 pp. The complete
54-run bounded Conv1 comparison verifies 27 paired initializer assets,
cohorts and all minibatch orders. Across all three seeds, the maximum paired
best-validation difference is .06 pp and the maximum over 270 epoch pairs
is .10 pp. Mean best-validation differences range from -.013 to +.020 pp
across the nine scheme/ceiling groups. This supports consistency of the
frozen BPTT and EqProp trajectories across these seeds, without establishing
statistical equivalence or official-test performance. See the
[complete three-seed table and curves](../paper_ready_results/bounded_conv1_three_seed_validation.md).
Conv2 ours qualifies at .001. All three full pilots are now collected and
stable: best/final validation is 90.34/90.34%, 96.58/96.58%, and
97.20/97.08%; the maximum decline is .12 pp. Conv3 legacy also qualifies
at .001, and all three full pilots are collected and stable at
93.54/93.54%, 97.68/97.68%, and 97.72/97.72%, with no final decline.
The same betas are released for seeds 1/2 under the approved group gate.
Conv2 ours repetitions run on Main/Nom; Conv3 legacy runs seed 1 on Trex
and seed 2 on JZ. This establishes full seed-0 stability across ceilings;
three-seed conclusions still require those repetitions. Conv3 legacy's worst scored cosine is .9974632 and
maximum symmetric norm difference is .0516333 across the three ceilings.
Conv3 ours also qualifies at beta .003, with worst cosine .9969936 and
maximum norm difference .0850763. Its tight-ceiling full pilot is now collected and stable: 85.20/84.64%
best/final validation, decline .56 pp, finite float64 and zero biases. The
paired BPTT reference is 84.54/84.30%; all initializer/cohort/order/control
checks match, with maximum epoch difference 1.20 pp. This is seed-0
stability evidence, without a three-seed or official-test claim. The middle
pilot is running on Loulou and the largest follows; all six repetitions stay
gated until the complete three-ceiling proof passes.
After collecting all corrected Conv2 legacy references, that group qualifies
at injected beta .003: worst cosine .9979283 and maximum norm difference
.0283384, with passing residuals across all three ceilings. The initial .03
point fails the cosine threshold at .9841046. All three full 30-epoch pilots are collected and stable: best/final
validation is 95.68/95.68%, 97.60/97.60%, and 97.86/97.76%, with maximum
drop .10 pp. All three pairs have identical recorded initializers, cohorts
and minibatch orders; the maximum absolute difference over 90 EP–BPTT
epoch pairs is .24 pp. This supports seed-0 stability across all ceilings
and released seeds 1/2 at the same beta. All six repetitions are now
collected, finite and stable, with maximum best-to-final decline .12 pp.
All nine Conv2 legacy EqProp runs therefore pass the stability rule across
three seeds. Their complete 18-run comparison verifies all nine shared
initializer assets, cohorts and thirty minibatch orders. Mean best-validation
EP minus BPTT differences at increasing ceilings are -.013, +.013 and -.013
pp; the maximum paired best difference is .10 pp, and the maximum over 270
matched epochs is .24 pp. This supports consistency across the three seeds
under the frozen contract, without establishing statistical equivalence or
official-test performance. See the [three-seed table and curves](../paper_ready_results/bounded_conv2_legacy_three_seed_validation.md) and the
[complete group proof](../results/paper-training-completion-20260911-v1/source-v6/guards/bounded_pilots.json)
and [three-ceiling pairing audit](../results/paper-training-completion-20260911-v1/checks/conv2_legacy_seed0_pairing.json).

## Published Learning-Rate Handoffs

These are the current parameter-wise optimizer inputs for downstream
unbounded/wide-range work. Publishing a handoff does not promote its
ordinary-MNIST validation measurements to paper evidence or erase its stated
review and confirmation limits.

| Handoff | Scope | Evidence status | Machine-readable authority |
|---|---|---|---|
| `perfectdiode-conv12-unbounded-fixed-lr-20260729-v1` | Conv1/Conv2, `[0,100]` | Conv1 ten-epoch confirmations complete, review pending; Conv2 three-epoch evidence only | [`perfectdiode_conv12_unbounded_fixed_lr_handoff_20260729_v1.json`](../configs/conv/perfectdiode_conv12_unbounded_fixed_lr_handoff_20260729_v1.json) |
| `perfectdiode-conv3-unbounded-selected-lr-20260729-v1` | Conv3, downstream `[0,100]` authorized from a `weight_max=null` source study | six-surface rho selection complete; source-contract mismatch recorded; long confirmation pending | [`perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json`](../configs/conv/perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json) |

### Conv1/Conv2 current fixed vectors

Notation is `C_i=ConvWeight_i`, `B_i=Bias_i`, and `D=DenseWeight_0`.
Runtime order is `[C0,D,B0]` for Conv1 and `[C0,C1,D,B0,B1]` for
Conv2. The machine-readable handoff derives each ordered vector from the named
mapping in that runtime order.

| Architecture | Scheme | Optimizer | Parameter-wise LR vector |
|---|---|---|---|
| Conv1 | baseline | SGD | `C0=B0=0.142696, D=0.0120238` |
| Conv1 | baseline | Adam | `C0=B0=8.66763e-4, D=1.09401e-4` |
| Conv1 | ours | SGD | `C0=B0=0.0375994, D=0.00311655` |
| Conv1 | ours | Adam | `C0=B0=8.66753e-4, D=1.09388e-4` |
| Conv1 | legacy | SGD | `C0=B0=2.05753e-4, D=4.76259e-5` |
| Conv1 | legacy | Adam | `C0=B0=2.88917e-4, D=3.64618e-5` |
| Conv2 | baseline | SGD | `C0=B0=7.90864, C1=B1=3.81476, D=0.820630` |
| Conv2 | baseline | Adam | `C0=B0=2.60078e-3, C1=B1=4.59053e-4, D=4.67660e-4` |
| Conv2 | ours | SGD | `C0=B0=0.523358, C1=B1=0.246043, D=0.0520201` |
| Conv2 | ours | Adam | `C0=B0=2.60036e-3, C1=B1=4.58651e-4, D=1.55136e-4` |
| Conv2 | legacy | SGD | `C0=B0=0.00496733, C1=0.00367839, B1=6.74726e-4, D=0.00236044` |
| Conv2 | legacy | Adam | `C0=B0=8.66750e-4, C1=B1=1.52841e-4, D=5.16982e-5` |

## Finished Studies

### Physical-KCL rerun scope closeout — 2026-09-02

The required corrected physical-KCL compatibility scope is complete.  It
consists of the Conv1 clean-wide checkpoint and same-LR BPTT diagnostic; the
Conv2/Conv3 clean-wide same-LR BPTT runs; the Conv2/Conv3 clean and
endpoint-noisy centered-EqProp repeats; and the three fixed-initialization
bounded Conv3 BPTT cells at `Gmax={1e-4,5e-4,1e-3}`.  Every scheduled run is
terminal, collected locally, and validated; the official MNIST test split
remained sealed.  The original 18-arm table was an exhaustive impact audit,
not the final execution contract.  Its unrun Conv1 EqProp and Conv1/Conv2
bounded cells, together with deferred wide SGD qualification, are optional
future work rather than missing required reruns.  See the
[closed inventory](legacy_physical_kcl_rerun_inventory.md) and the terminal
[operational record](current_simulations.md#physical-kcl-rerun-closeout).

## `perfectdiode-conv1-legacy-physical-kcl-same-weights-and-bptt-rerun-seed0-20260825-v1` — Conv1 legacy checkpoint compatibility and exact-LR retraining under the corrected physical KCL energy

- Analyzed: 2026-08-25
- Evidence class:
  `ordinary_mnist_physical_kcl_same_lr_diagnostic`; implementation-change
  compatibility evidence, non-paper-facing.
- Outcome: `mixed`.  Corrected-energy Adam remains equivalent at the
  predeclared `0.5 pp` resolution, whereas both unchanged-checkpoint and
  exact-LR SGD comparisons change materially.
- Scientific question: Does replacing the formerly implemented coherent but
  differently parameterized legacy energy with the physical KCL energy
  materially change Conv1 validation accuracy when old checkpoint tensors
  are replayed unchanged or training is repeated with the exact old rates?
- Frozen setup: seed-0 ordinary MNIST deterministic `55,000/5,000`
  train/validation split; legacy `(A,B)=(4,.25)`; perfect-diode BPTT;
  exact-zero hidden bias; `T=K=4`; batch size `16`; validation batch size
  `64`; ten epochs; source SGD rates
  `[2.05753e-4,4.76259e-5,0]` and Adam rates
  `[2.88917e-4,3.64618e-5,0]`.  Checkpoint replays use the source study's
  exact best/final bytes, reset state each batch, and apply no optimizer step.
  The official test split remains sealed.
- Coverage and integrity: both exact-config one-batch smokes, four full
  5,000-example checkpoint replays, and both ten-epoch retrains complete.
  All six scientific production bundles validate; every production manifest
  records clean fix commit `f016bf6a`; config, seed, split/order, epoch,
  optimizer, and LR contracts match the source arms.  Both new runs report
  zero official-test evaluations, and every best/final `Bias_0` tensor is
  exactly zero.  The initial checkpoint smoke is retained but excluded from
  inference because it compared one batch with a full-split source accuracy;
  corrected `smoke_v2` is the accepted gate.
- Headline unchanged-checkpoint measurements: source-to-corrected best/final
  validation is SGD `95.92/95.60 -> 93.84/93.88%`
  (`-2.08/-1.72 pp`) and Adam `96.44/96.38 -> 96.12/96.24%`
  (`-.32/-.14 pp`).
- Headline exact-LR retraining measurements: source-to-corrected best/final
  validation is SGD `95.92/95.60 -> 91.78/91.78%`
  (`-4.14/-3.82 pp`) and Adam `96.44/96.38 -> 96.50/96.44%`
  (`+.06/+.06 pp`).  Corrected SGD selects epoch 10 and corrected Adam epoch
  9.
- Interpretation: historical convergence is expected because the former code
  still minimized a coherent energy; it does not validate the manuscript's
  physical circuit equations.  Conv1 Adam's learned decision rule and frozen
  optimizer scale are empirically robust to the correction at one seed.  SGD
  is not: its same-checkpoint loss is material, and the larger retraining loss
  shows that optimizer scaling is more sensitive than checkpoint reuse.
  Corrected Conv1 legacy SGD would therefore require LR reselection before it
  could be used as corrected evidence; it is excluded from the closed required
  scope.  The old Adam rate is accepted for the completed compatibility scope,
  not as a generally requalified legacy result.
- Limits and scope: one architecture, one seed, two BPTT optimizers, and
  validation-only evidence.  This study alone neither estimates uncertainty
  nor clears Conv2/Conv3, EqProp phase gradients and beta, read noise, bounded
  weights, or an official-test evaluation.  The selected Conv2/Conv3 checks
  were completed separately and are recorded in the closeout above.  Broader
  Conv1 SGD, EqProp, and bounded follow-up is optional rather than required for
  the current paper integration.
- Artifacts: [study report](../results/perfectdiode-conv1-legacy-physical-kcl-same-weights-and-bptt-rerun-seed0-20260825-v1/analysis/report.md),
  [checkpoint replays](../results/perfectdiode-conv1-legacy-physical-kcl-same-weights-and-bptt-rerun-seed0-20260825-v1/checkpoint_replay/),
  [exact-LR retrains](../results/perfectdiode-conv1-legacy-physical-kcl-same-weights-and-bptt-rerun-seed0-20260825-v1/same_lr_bptt/), and
  [legacy rerun inventory](legacy_physical_kcl_rerun_inventory.md).

## `perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1` — Conv1/Conv3 one-decade EqProp training under endpoint read noise

- Analyzed: 2026-08-17
- Evidence class:
  `ordinary_mnist_eqprop_zero_bias_read_noise_training_diagnostic`;
  user-directed exploratory robustness evidence, non-paper-facing.
- Outcome: `mixed`. All six runs complete finite `10/30`-epoch budgets and
  pass the frozen within-run stability rule. Conv1 remains effectively
  unchanged from clean, but Conv3 loses validation accuracy, with a severe
  legacy penalty.
- Scientific question: At the already clean-stable one-decade beta, do the
  exact-zero-bias shared-`T/K` Conv1 and Conv3 Adam EqProp cases remain stable
  when endpoint voltage reads used by the local update have sigma `5e-4`
  Gaussian noise?
- Frozen setup: seed-0 ordinary MNIST's deterministic `55,000/5,000`
  train/validation split; baseline/ours/legacy; exact inherited BPTT weight
  learning rates; exact-zero bias rates and tensors; centered frozen-current
  true-float64 EqProp; `T=K=4/8` for Conv1/Conv3; `10/30` epochs; injected
  beta Conv1 `100/30/3` and Conv3 `100/3/.001`; independent negative/positive
  layer-endpoint noise with seed `2026081601`; input noise disabled; noiseless
  validation; no official-test read. The six exact clean controls come from
  the reviewed zero-noise one-decade qualification.
- Coverage and integrity: Jean Zay UMG V100-32GB array `1037835_[0-2]`
  completed all tasks with exit `0:0` in `05:00:24`--`05:07:34`, after
  canary `1037825_0` passed. The authoritative local copy has zero checksum
  differences from Jean Zay. All `6/6` canonical bundles, six run receipts,
  three pack receipts, three semantic summaries, `2,887,920` expected noise
  draws, initialization/split/order matching, zero-bias, completion, and
  no-test-read guards validate.
- Headline measurements: noisy best/final validation for Conv1
  baseline/ours/legacy is `96.18/96.10`, `96.46/96.28`, and
  `96.40/96.38%`; clean-relative final changes are `+.06/-.06/-.04 pp`, and
  no epoch differs by more than `.08 pp`. Conv3 noisy best/final is
  `96.94/96.84`, `96.96/96.82`, and `88.30/87.64%`; clean-relative
  best/final changes are `-.82/-.84`, `-1.76/-1.82`, and
  `-10.78/-11.38 pp`.
- Interpretation: the one-decade tier is a finite optimizer-level training
  point at sigma `5e-4`, but it is not a common practical robustness point.
  Conv3 legacy still learns and remains only `.66 pp` below its own best at
  the end, so it does not meet the declared collapse rule; nevertheless, its
  `11.38 pp` final clean-relative penalty is a large performance failure.
  Training stability and read-noise tolerance must therefore be recorded
  separately, and a within-run best-to-final rule alone is inadequate for the
  noise decision.
- Limits and retained deviations: one model seed and one fixed noise
  realization do not estimate uncertainty. The study does not isolate depth,
  beta, or amplification causally, freeze a paper beta, or authorize an
  official-test read. The one-decade `213/216` gradient result and trained
  Conv3-baseline `T=8` residual caveat remain explicitly unresolved.
- Artifacts: [report](../results/perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1/analysis/report.md),
  [summary table](../results/perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1/analysis/summary.csv),
  [trajectory plot](../results/perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1/analysis/validation_accuracy_trajectories.png),
  [machine summary](../results/perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1/analysis/summary.json), and
  [collection receipt](../results/perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1/analysis/collection_validation_receipt.json).

## `perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1` — Conv3 broadened all-layer cosine versus beta under endpoint read noise

- Analyzed: 2026-08-17
- Evidence class:
  `ordinary_mnist_learning_algorithm_read_noise_gradient_diagnostic`;
  exploratory read-only mechanism study, non-paper-facing.
- Outcome: `negative` for finding a per-minibatch all-layer faithful beta and
  `mixed` for locating layer-specific optima. None of the six scheme x
  checkpoint contexts reaches noisy EqProp/BPTT cosine `>=.99` on every
  `C0/C1/C2/Dense` layer and all four batches. The enlarged range does bound
  the trained C2 average optimum for baseline and ours, but not the early
  trained convolutional-layer optima.
- Scientific question: How do all Conv3 layerwise EqProp/BPTT cosine curves
  continue when beta is swept one decade above the original maximum under
  endpoint-voltage read noise sigma `5e-4`?
- Frozen setup: the reviewed nine-point parent is joined with eight new
  `1/8`-decade points, producing beta-scale factors `10^(i/8)`, `i=0..16`.
  Injected ranges are baseline `100--10000`, ours `3--300`, and legacy
  `.001--.1`. The same seed-0 exact-zero-bias Adam BPTT source checkpoints,
  reconstructed initialization, best-validation checkpoint, deterministic
  four-batch ordinary-MNIST validation cohort, `T=K=8`, centered
  frozen-current true-float64 EqProp, exact BPTT reference, and all
  `C0/C1/C2/Dense` weights are used throughout. Independent sigma-`5e-4`
  negative/positive endpoint reads use seed `2026081601`; the standard-normal
  tensors are identical across schemes and all 17 beta points. A clean EqProp
  control remains embedded at every point.
- Coverage and integrity: the local RTX 3090 extreme-point smoke and all eight
  extension production bundles validate canonically; the reviewed nine parent
  bundles also revalidate in the join. Combined coverage is `408/408`
  checkpoint-batch replays, `1632/1632` layer-batch comparisons, and
  `3264/3264` endpoint-noise records. All 17 noise signatures match exactly.
  All 96 BPTT reference coordinates are beta-invariant with maximum norm delta
  `0`; source/checkpoint bytes, in-memory parameters, exact-zero biases,
  true-float64 arithmetic, shared `T/K`, frozen force, cohort, input
  exactness, no-optimizer-step, and no-official-test-read guards pass.
- Headline conservative measurements: best all-layer worst-batch cosine at
  initialization/best is baseline `.162914 @ B=1778.28` / `.786156 @
  B=10000`, ours `-.015895 @ B=53.3484` / `-.039670 @ B=300`, and legacy
  `.012911 @ B=.1` / `-.071393 @ B=.1`. The corresponding clean worst cosine
  at each trained best point is baseline `.978357`, ours `.926021`, and legacy
  `.969854`. Dense passes from the lower endpoint throughout; trained baseline
  C2 remains the only convolutional all-batch `.99` crossing, first at
  `B=1000` and peaking conservatively at `.994911 @ B=1778.28`.
- Headline four-batch averages at the trained checkpoint: baseline peaks at
  C0/C1/C2/Dense `.897801 @ 10000`, `.980900 @ 10000`, `.996141 @ 1778.28`,
  and `.999999 @ 1778.28`; ours at `.091214 @ 300`, `.189009 @ 300`,
  `.976147 @ 168.702`, and `.999996 @ 30`; legacy at `.056734 @ .1`,
  `.003716 @ .1`, `.577923 @ .1`, and `.999992 @ .023714`. Baseline/ours C2
  fall to `.983442/.970383` at the enlarged endpoint after their interior
  optima.
- Interpretation: the extra decade exposes the expected read-noise versus
  finite-beta tradeoff and identifies some deep-layer peaks, but it does not
  produce a common all-layer useful beta. Early C0/C1 estimates remain
  noise-limited even where their averages continue rising; initialization
  curves eventually deteriorate strongly from finite-beta distortion. These
  above-maximum values are diagnostic only and are not training-qualified.
  Prior training already shows optimizer-dependent collapse at the original
  boundary, so this study does not nominate larger training betas.
- Residual and limitations: trained baseline retains the known
  beta-independent post-`T=8` residual caveat. The study has one model seed,
  one noise seed, four minibatches, and one absolute noise scale. Some trained
  early-layer curves remain ceiling-bound, so the experiment bounds neither
  every layer-specific optimum nor a read-noise-only asymptote.
- Artifacts: [analysis report](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/report.md),
  [four-batch-average table](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/best_checkpoint_average_cosine_vs_beta.csv),
  [average-per-layer plot](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/plots/best_checkpoint_average_cosine_vs_injected_beta_all_layers.png),
  [all-layer table](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/layerwise_cosine_vs_beta.csv),
  [per-layer thresholds](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/per_layer_thresholds.csv),
  [median/spread plot](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/plots/median_cosine_vs_injected_beta_all_layers.png),
  [worst-batch plot](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/plots/worst_batch_cosine_vs_injected_beta_all_layers.png),
  [machine summary](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-above-max-to-10xmax-sigma-5em4-seed0-20260817-v1/analysis/summary.json),
  and [frozen extension config](../configs/conv/perfectdiode_conv3_zero_bias_adam_eqprop_bptt_gradient_beta_sweep_above_max_to_10xmax_sigma_5em4_seed0_20260817_v1.json).

## `perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-one-decade-to-max-sigma-5em4-seed0-20260817-v1` — Conv3 all-layer cosine versus beta under endpoint read noise

- Analyzed: 2026-08-17
- Evidence class:
  `ordinary_mnist_learning_algorithm_read_noise_gradient_diagnostic`;
  exploratory read-only mechanism study, non-paper-facing.
- Outcome: `negative` for finding a per-minibatch all-layer faithful beta in
  the declared one-decade-to-original-maximum interval. None of the six
  scheme x checkpoint contexts reaches cosine `>=.99` on every
  `C0/C1/C2/Dense` layer and all four batches at any tested point.
- Scientific question: For every Conv3 amplification scheme and weight
  layer, how does centered EqProp versus same-`T/K` BPTT cosine change when
  injected beta rises logarithmically from the accepted one-decade tier to
  the original maximum under endpoint-voltage read noise sigma `5e-4`?
- Frozen setup: the same seed-0 exact-zero-bias Adam BPTT source checkpoints
  and deterministic four-batch ordinary-MNIST validation cohort as the
  accepted direct gradient gate; Conv3 baseline/ours/legacy; reconstructed
  initialization and best-validation checkpoints; `T=K=8`; centered
  frozen-current true-float64 EqProp; exact BPTT reference; all
  `C0/C1/C2/Dense` weights; nine uniformly log-spaced beta-scale factors
  `10^(i/8)`, `i=0..8`; injected ranges baseline `100--1000`, ours `3--30`,
  and legacy `.001--.01`. Independent sigma-`5e-4` negative/positive
  endpoint reads use seed `2026081601`; standard-normal tensors are identical
  across schemes and all beta points. The clean EqProp replay is retained at
  each beta.
- Coverage and integrity: the local RTX 3090 semantic smoke and all nine
  production bundles validate canonically. The joined surface contains
  `216/216` checkpoint-batch replays, `864/864` layer-batch comparisons, and
  `1728/1728` endpoint-noise records. All nine noise signatures match exactly
  across beta; all 96 BPTT reference norms are beta-invariant with maximum
  delta `0`; source/checkpoint bytes, in-memory parameters, exact-zero biases,
  true-float64 arithmetic, shared-`T/K`, frozen force, cohort, input
  exactness, no-optimizer-step, and no-official-test-read guards pass.
- Headline measurements: the best observed all-layer worst-batch cosine at
  initialization/best is baseline `.1157/.0841` at `B=1000`, ours
  `-.0243/-.1892` at `B=30`, and legacy `-.00497/-.0720` at `B=.01`.
  Dense passes cosine `.99` from the one-decade anchor in all six contexts.
  Among convolutional layers, only trained baseline C2 reaches an all-batch
  `.99` crossing, at `B=1000` with minimum `.9923`. Initialization C2 peaks
  below threshold at baseline `.9863` (`B=316.2`), ours `.9804`
  (`B=12.65`), and legacy `.8369` (`B=.01`). No C0 or C1 context crosses
  `.99`; their best minima remain at most `.1157/.4875` for baseline,
  `-.0243/-.00355` for ours, and `.0233/-.00109` for legacy across
  initialization/best.
- Interpretation: raising beta improves read-noise signal margin in several
  deeper layers, most visibly C2, but the early convolutional readouts remain
  noise-dominated throughout the already qualified training interval. At the
  original maximum, the clean initialization worst-layer cosine has also
  fallen to baseline/ours/legacy `.8022/.8016/.9767`, exposing the opposing
  finite-beta distortion. Therefore choosing the largest trainable beta does
  not recover strict single-minibatch all-layer fidelity at sigma `5e-4`;
  reducing the endpoint noise or averaging multiple independent reads or
  gradients is the more direct remedy. This does not imply optimizer-level
  collapse, because the separate noisy training study measures multi-step
  averaging rather than one instantaneous gradient.
- Residual and limitations: trained baseline retains the known
  beta-independent post-`T=8` residual caveat. The sweep has one model seed,
  one noise seed, four batches, one absolute noise scale, and stops at the
  original maximum because larger betas are outside the previously qualified
  training interval and clean finite-beta distortion is already visible.
- Artifacts: [analysis report](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-one-decade-to-max-sigma-5em4-seed0-20260817-v1/analysis/report.md),
  [all-layer table](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-one-decade-to-max-sigma-5em4-seed0-20260817-v1/analysis/layerwise_cosine_vs_beta.csv),
  [per-layer thresholds](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-one-decade-to-max-sigma-5em4-seed0-20260817-v1/analysis/per_layer_thresholds.csv),
  [median/spread plot](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-one-decade-to-max-sigma-5em4-seed0-20260817-v1/analysis/plots/median_cosine_vs_injected_beta_all_layers.png),
  [worst-batch plot](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-one-decade-to-max-sigma-5em4-seed0-20260817-v1/analysis/plots/worst_batch_cosine_vs_injected_beta_all_layers.png),
  and [machine summary](../results/perfectdiode-conv3-zero-bias-adam-centered-float64-eqprop-bptt-gradient-beta-sweep-one-decade-to-max-sigma-5em4-seed0-20260817-v1/analysis/summary.json).

## `perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1` — Conv2/Conv3 gradient fidelity with endpoint read noise and Conv2 high-beta extension

- Analyzed: 2026-08-17
- Conv2 extension analyzed: 2026-08-18
- Evidence class:
  `ordinary_mnist_learning_algorithm_read_noise_gradient_diagnostic`; formal
  read-only hyperparameter/robustness diagnostic, non-paper-facing.
- Outcome: `negative` for direct per-minibatch gradient fidelity at endpoint
  read-noise sigma `5e-4`. None of the 12 architecture x scheme x checkpoint
  contexts passes the all-layer noisy-versus-clean acquisition gate or the
  noisy-versus-BPTT task gate.
- Scientific question: If the current one-decade EqProp beta is retained, do
  Conv2 and Conv3 gradients remain faithful when the established independent
  Gaussian endpoint-voltage read-noise model is applied at sigma `5e-4`?
- Frozen setup: seed-0 zero-bias Adam BPTT source checkpoints; Conv2/Conv3 x
  baseline/ours/legacy; reconstructed initialization and best validation;
  four fixed 16-example ordinary-MNIST validation batches; shared
  `T/K=6/6,8/8`; centered frozen-current EqProp in true float64; injected
  one-decade beta Conv2 `100/10/.03` and Conv3 `100/3/.001`; all Conv/Dense
  weights scored; cosine `>=.99` and symmetric norm delta `<=.10`; input and
  BPTT reference exact. Independent sigma-`5e-4` noise is added to every
  negative/positive free-layer endpoint voltage after equilibrium and before
  the local energy-gradient readout, with seed `2026081601` and matched
  standard-normal draws across schemes.
- Coverage and integrity: the local RTX 3090 smoke and production bundles
  validate canonically. Production completes `48/48` replay contexts,
  `168/168` layer-batch comparisons, and `336/336` noised endpoint tensors in
  54 seconds. All source/checkpoint and in-memory parameter bytes remain
  unchanged; exact-zero-bias, true-float64, shared-`T/K`, frozen-force,
  cohort, matched-noise, finite-gradient, input-exactness, and zero-test-read
  guards pass. The embedded clean replay is exactly identical to all 168
  accepted one-decade parent rows.
- Conv2 high-beta extension: five additional local RTX 3090 production
  bundles cover relative beta scales `10,17.7828,31.6228,56.2341,100`, from
  the original maximum through ten times that maximum. Their Conv2 slice has
  `120` checkpoint-batch replays, `360` layer-batch rows, and `720` endpoint
  noise records. All five bundles validate canonically; Conv2 residuals and
  all scientific/read-only guards pass; BPTT norms are exactly beta-invariant;
  and all noise signatures match byte-for-byte across beta. The bundle-level
  residual failures are solely the retained Conv3 baseline caveat. The
  scale-10 embedded clean replay agrees with the accepted clean maximum to
  `1.30e-14` cosine and exactly in BPTT norm and norm delta. One first smoke
  attempt is an excluded pre-case operational failure because sandboxed CUDA
  was unavailable; the replacement noisy endpoint smoke completes and
  validates.
- Headline measurements: layer-batch acquisition/task/combined passes are
  `52/50/50` of `168`. Every Dense row passes (`48/48`), and Conv2-baseline
  C1 contributes the only two additional task passes; every other
  convolutional layer/scheme group has `0/8` task passes across the two
  checkpoints. Conv2 minimum noisy EqProp/BPTT cosine at initialization/best
  is baseline `.5476/.2136`, ours `.2423/-.0888`, and legacy
  `.00715/-.1713`. Conv3 is baseline `-.0862/-.0554`, ours
  `-.0551/-.2066`, and legacy `-.00696/-.0720`. Maximum context norm deltas
  reach `1.99999`.
- Interpretation: a single centered two-phase gradient estimate is strongly
  noise-dominated in the convolutional layers at sigma `5e-4`; the Dense
  gradient remains well resolved. The effect is especially severe for the
  smaller scheme-specific base betas because endpoint-gradient readout noise
  is divided by the finite-difference denominator. This is not a training
  collapse result: separate noisy long-training runs can remain finite and
  accurate because stochastic gradient errors may average over many
  minibatches and optimizer steps. It does show that sigma `5e-4` cannot be
  called per-minibatch all-layer gradient-faithful at the one-decade tier.
- High-beta interpretation: larger beta materially improves noisy upstream
  direction before finite-nudge bias dominates. At the trained checkpoint,
  baseline C0 noisy mean peaks at `.983554` (`B=3162.28`), ours C0 reaches
  `.953794` (`B=1000`), and legacy C0 reaches only `.332723` (`B=3`). By
  relative scale `31.6228`, only Dense layers retain the conservative
  all-batch `.99` gate; by `56.2341`, every trained convolution-layer mean is
  below `.99`. At the ten-times-maximum endpoint, clean trained C0/C1 means
  are baseline `.969744/.946317`, ours `.965700/.943066`, and legacy
  `.932222/.938292`. This is unambiguous direct-gradient failure at high beta,
  not a new training-collapse measurement. It supports a layer-dependent
  signal-to-noise/locality tradeoff but does not qualify a layer-specific
  training estimator.
- Residual and limitations: trained Conv3 baseline retains the known
  beta/noise-independent post-`T=8` residual caveat. One model seed, one noise
  seed, four batches, one absolute uncalibrated noise scale, and a strict
  per-layer/per-batch gate; no multi-step optimizer-noise averaging or paper
  accuracy is estimated.
- Artifacts: [analysis report](../results/perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1/analysis/report.md),
  [layer table](../results/perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1/analysis/layer_summary.csv),
  [clean-versus-noisy plot](../results/perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1/analysis/plots/layerwise_clean_vs_noisy_cosine.png),
  [machine summary](../results/perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1/analysis/summary.json),
  [production bundle](../results/perfectdiode-conv23-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-one-decade-sigma-5em4-seed0-20260817-v1/production/),
  [paper Conv2 clean/noisy beta figure](../papers/amplification_overleaf/figures/beta_cosine/conv2_by_scheme_all_layers_cosine_vs_beta.pdf),
  [paper summary table](../papers/amplification_overleaf/figures/beta_cosine/beta_cosine_summary.csv),
  and [paper provenance](../papers/amplification_overleaf/figures/beta_cosine/provenance.json).

## `perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1` — zero-bias shared-T/K one-decade Adam EqProp training qualification

- Analyzed: 2026-08-17
- Evidence class: `ordinary_mnist_eqprop_zero_bias_training_qualification`;
  user-directed exploratory hyperparameter-selection evidence,
  non-paper-facing.
- Outcome: `positive` for the declared training-stability question. All nine
  Conv1/Conv2/Conv3 x baseline/ours/legacy cases complete their full epoch
  budget with finite metrics and finish less than `.14 pp` below their own
  best validation, far inside the predeclared `<5 pp` rule.
- Scientific question: At the one-decade scheme-specific injected beta, do all
  nine Adam EqProp cases learn stably when BPTT and EqProp use identical
  learning-rate vectors, every bias is fixed at exact zero, and
  `T/K=4/4,6/6,8/8` is shared by architecture?
- Frozen setup: seed-0 ordinary MNIST; Conv1/2/3 baseline/ours/legacy;
  inherited zero-bias BPTT Adam vectors unchanged; injected beta Conv1
  `100/30/3`, Conv2 `100/10/.03`, Conv3 `100/3/.001` (base beta
  `100/7.5/.1875`, `100/.625/.0001171875`, and
  `100/.046875/2.44140625e-7`); centered frozen-current EqProp; true float64;
  `T/K=4/4,6/6,8/8`; Kaiming `[0,100]` weights; 10/30/30 epochs; endpoint
  read-noise sigma `0`; no official-test read; best-validation checkpoint
  selection.
- Coverage and integrity: Jean Zay UMG array `1030857_[0-4]` completed all
  five V100-32GB tasks with exit `0:0` in `02:17:19`--`05:02:24`; four packs
  ran two cases concurrently on one GPU and one pack ran Conv2 legacy alone.
  The 106 MB authoritative local copy has zero checksum differences from the
  remote tree. All nine canonical bundles, five pack summaries/receipts, nine
  run receipts, frozen source/config hashes, exact-zero bias rates and
  best/final checkpoint tensors, architecture-matched initialization,
  common train/validation split, common ten-epoch minibatch prefix, complete
  Conv2/Conv3 30-epoch order, zero-noise, and zero-test-read guards pass.
- Headline measurements: best/final validation is Conv1 baseline
  `96.18/96.04%`, ours `96.42/96.34%`, legacy `96.44/96.42%`; Conv2
  `97.22/97.22%`, `98.10/98.02%`, `98.44/98.42%`; and Conv3
  `97.76/97.68%`, `98.72/98.64%`, `99.08/99.02%`. Final drops from best are
  `.14/.08/.02 pp`, `0/.08/.02 pp`, and `.08/.08/.06 pp`, respectively.
- Interpretation: the one-decade tier is demonstrably training-stable under
  the exact zero-bias/shared-`T/K` Adam contract, so ordinary-MNIST training
  stability is not a reason by itself to reduce beta another decade. This does
  not freeze a paper beta: the retained direct gradient result is `213/216`
  at one decade versus `216/216` at two decades, and trained Conv3 baseline
  retains its beta-independent post-`T=8` residual deviation. The remaining
  decision is a signal-margin/read-noise versus local-gradient-fidelity
  tradeoff; a matched two-decade stability qualification would complete the
  other branch.
- Limitations and deviations: one seed and ordinary-MNIST selection accuracy;
  no BPTT accuracy comparison in this study; no harder-dataset transfer test.
  The user-directed launch retained three failed Conv3-initialization C0 rows
  from the one-decade gradient gate and did not reclassify them or the Conv3
  residual deviation as passing or paper-ready.
- Artifacts: [report](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/report.md),
  [summary table](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/summary.csv),
  [machine summary](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/summary.json),
  [trajectories](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/validation_accuracy_trajectories.png),
  and [verification](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/verification.json).

## `perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1` — Conv2 SGD endpoint-read-noise repeat and optimizer-contract comparison

- Analyzed: 2026-08-16
- Evidence class: `ordinary_mnist_eqprop_read_noise_training_diagnostic`;
  exploratory and non-paper-facing.
- Outcome: `mixed`. Sigma `5e-4` causes no observed instability with either
  historical optimizer contract and preserves the same scheme ordering, but
  the clean-relative penalty magnitude changes with optimizer/LR contract.
- Scientific question: At the same clean-stable one-decade scheme-specific
  betas used by the Adam read-noise study, do Conv2 SGD runs remain finite at
  endpoint-noise sigma `5e-4`, and does the earlier qualitative robustness
  conclusion survive the optimizer/LR-contract change?
- Frozen setup: baseline/ours/legacy injected beta `100/10/.03` (base beta
  `100/.625/.0001171875`); historical parameter-wise SGD vectors; centered
  frozen-current EqProp; endpoint read-noise sigma `5e-4` and seed
  `2026081201`; true float64; `T=K=8`; seed-0 ordinary MNIST; exact common
  Kaiming `[0,100]` initialization, split, and ten minibatch orders; ten
  epochs; no official-test read. Existing one-decade clean SGD/Adam controls
  and matched noisy Adam runs are reused read-only.
- Coverage and integrity: Jean Zay `fmu@v100` canary `1029073_0` and
  production array `1029094_[0-2]` all complete with exit `0:0`. The three
  production tasks take `00:57:51/00:57:00/01:10:18`; all three canonical
  bundles, summaries, transport receipts, source/config hashes, optimizer
  identity, 206,280 noise draws per run, initialization/split/order hashes,
  finite histories, and zero-test-read guards pass. The authoritative local
  copy has zero checksum-dry-run differences from Jean Zay, and the final
  focused suite passes `29/29`.
- Headline measurements: SGD clean -> noisy final validation is baseline
  `96.52 -> 96.34%` (`-.18 pp`), ours `97.80 -> 97.46%` (`-.34 pp`), and
  legacy `97.52 -> 96.72%` (`-.80 pp`). Against the same later paired clean
  controls, Adam changes are `96.88 -> 96.78%` (`-.10 pp`),
  `97.92 -> 97.44%` (`-.48 pp`), and `98.40 -> 96.90%` (`-1.50 pp`). The
  noisy SGD-minus-Adam endpoints are `-.44/+.02/-.18 pp`.
- Interpretation: ours has the highest noisy final validation accuracy and
  legacy the largest clean-relative loss under both contracts. The earlier
  qualitative conclusion is therefore not Adam-specific, but its numeric
  size is optimizer/LR-contract dependent, especially for legacy. The
  original Adam extension's `-.22/-.42/-1.44 pp` values remain valid against
  its own earlier clean parent; its noisy endpoints are unchanged.
- Limitations: one seed; ordinary-MNIST diagnostic accuracy; optimizer and
  historical LR vector change together, so this is not an optimizer-algorithm
  ablation. Clean and noisy anchors come from separate retained launches and
  source archives rather than a single co-scheduled factorial, so small
  endpoint differences should not be overinterpreted.
- Artifacts: [report](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/report.md),
  [terminal table](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/terminal_summary.csv),
  [trajectories](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/validation_accuracy_trajectories.png),
  [verification](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/verification.json),
  and [collection receipt](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/collection_validation_receipt.json).

## `perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1` — zero-bias shared-T/K EqProp--BPTT gradient qualification

- Analyzed: 2026-08-16
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_qualification`;
  formal hyperparameter-selection diagnostic, non-paper-facing.
- Outcome: `mixed`. The original maximum tier fails broadly, the one-decade
  beta tier fails three Conv3-initialization first-convolution comparisons,
  and the two-decade tier passes all direct gradient comparisons. Trained
  Conv3 baseline remains residual-limited at the shared `T=8`, so the
  two-decade result is gradient-qualified but not an unqualified
  equilibrium-qualified paper handoff.
- Scientific question: Before matched long training, do centered
  frozen-current EqProp and BPTT weight gradients agree when both use exact
  zero biases and shared Conv1/2/3 `T/K=4/4,6/6,8/8`, and does the answer
  distinguish the clean-stable one- and two-decade beta candidates?
- Frozen setup: all nine seed-0 Conv1/Conv2/Conv3 x baseline/ours/legacy Adam
  source cases from the completed zero-bias ordinary-MNIST study;
  reconstructed initialization and best-validation checkpoints; four fixed
  16-example validation batches; true-float64 dynamics; centered
  frozen-current EqProp; exact common post-`T` state and exact `K`-step BPTT
  reference; Conv/Dense weights scored; cosine `>=.99` and symmetric norm
  delta `<=.10`; projected-residual p90 `<.01` recorded separately; no
  optimizer or official-test read.
- Coverage and integrity: each of the three tiers completes 72
  checkpoint-batch replays and 216 layer-batch comparisons after a completed
  Conv3-legacy semantic smoke. All six canonical bundles validate. All nine
  source bundles and cohort
  hashes reproduce; all source/checkpoint bytes and in-memory parameter
  tensors remain unchanged; every bias LR and initial/best tensor is exact
  zero; and all float64, iteration, common-state, and frozen-force guards pass.
- Headline measurements: at the original maximum tier Conv1 `1000/300/30`,
  Conv2 `1000/100/.3`, Conv3 `1000/30/.01`, only `159/216` rows and `9/18`
  checkpoint contexts pass; worst cosine is `.801608` and maximum norm delta
  `.533422`. At the one-decade injected-beta tier Conv1
  `100/30/3`, Conv2 `100/10/.03`, Conv3 `100/3/.001`, `213/216` rows pass.
  Conv3-baseline initialization C0 fails one batch (worst cosine `.987983`),
  and Conv3-ours initialization C0 fails two (worst cosine `.939168`, norm
  ratio `1.32312`, symmetric norm delta `.278178`). Every Conv1/Conv2 row and
  every best-checkpoint context passes. At the two-decade tier Conv1
  `10/3/.3`, Conv2 `10/1/.003`, Conv3 `10/.3/1e-4`, all `216/216` pass;
  global worst cosine is `.998877` and maximum norm delta `.039421`.
- Residual result: only trained Conv3 baseline fails, and only at the beta-
  independent post-`T=8` free endpoint. Layer_1 p90 is `.106--.156` and
  Layer_3 p90 `.0179--.0247` across the four batches. Its direct gradients
  pass at both beta tiers.
- Interpretation: clean training stability at the one-decade tier was not
  sufficient to make it universally gradient-faithful under the new matching
  contract. The two-decade tier is the supported candidate for the next
  Adam-first zero-bias ordinary-MNIST training qualification. Sealing paper
  checkpoints and reading the official MNIST test split remain blocked until
  that stability run completes and the Conv3-baseline truncated-T residual
  caveat is explicitly accepted or a new shared BPTT/EqProp T/K is
  requalified.
- Limitations: one seed, Adam-trained endpoint checkpoints, 64 validation
  examples, and two checkpoint roles rather than an online full-training
  trace. This measures gradient fidelity, not training stability or accuracy;
  SGD endpoint checkpoints require their own gate before an SGD-first launch.
- Artifacts: [review](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/review.md),
  [maximum-beta summary](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production-max-beta/case_summary.csv),
  [maximum-beta layer table](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production-max-beta/layer_metrics.csv),
  [one-decade summary](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production/case_summary.csv),
  [one-decade layer table](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production/layer_metrics.csv),
  [two-decade summary](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production-two-decades/case_summary.csv),
  [two-decade layer table](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/production-two-decades/layer_metrics.csv),
  and [validation receipt](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/validation_receipt.json).

## `perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1` — Conv1 legacy factor-three Conv-LR counterfactual

- Analyzed: 2026-08-16
- Evidence class: `ordinary_mnist_learning_rate_counterfactual`; exploratory
  and non-paper-facing.
- Outcome: `negative` for adopting a general factor-three Conv-LR increase.
  SGD becomes worse and Adam is effectively unchanged, so the current Conv1
  legacy learning-rate handoff is not changed.
- Scientific question: Does the open factor-three upper-Conv plateau from the
  bounded, zero-bias Conv1 legacy-SGD surface justify tripling the wide-range
  `ConvWeight_0` learning rate when bias learning is excluded, and does the
  same transfer help Adam?
- Bounded-evidence verification: legacy-SGD ties at `94.50%` between
  `(rho_conv,rho_dense)=(.003,.0033333)` and the factor-three upper Conv point
  `(.009,.0033333)`, leaving that edge open.  Legacy-Adam is already bracketed:
  its selected `(.009,.01)` point reaches `95.42%`, while the factor-three
  upper neighbor `(.027,.01)` reaches `95.14%` (`-.28 pp`).  The prior result
  therefore motivated an SGD sentinel only; it was not a general LR handoff.
- Frozen setup: two seed-0 ordinary-MNIST Conv1 legacy BPTT/BP arms, SGD and
  Adam, ten epochs, accepted `T=K=4`, Kaiming weights projected to `[0,100]`,
  no official-test read, exact-zero Bias_0 LR/tensors, and unchanged Dense LR.
  Only ConvWeight_0 LR changes relative to the reviewed zero-bias controls:
  SGD `2.05753e-4 -> 6.17259e-4` and Adam
  `2.88917e-4 -> 8.66751e-4`.
- Coverage and integrity: exact-config sequential CUDA smokes and a concurrent
  two-process packing smoke passed.  `main` handles `@169/@168` completed with
  launcher exit `0`; both ten-epoch production bundles validate, share current
  initial-state SHA `594521a3...91ccb7b`, and have finite best/final checkpoint
  arrays with exact-zero Bias_0.  All four new/historical BPTT runs share the
  exact train/validation split and all ten minibatch-order hashes.
- Headline measurements: SGD best validation changes
  `95.92 -> 95.52%` (`-.40 pp`) and final changes
  `95.60 -> 95.52%` (`-.08 pp`); the times-three run is lower at every epoch
  with a mean epochwise delta of `-.548 pp`.  Adam best changes
  `96.44 -> 96.46%` (`+.02 pp`) and final changes
  `96.38 -> 96.46%` (`+.08 pp`); epochwise differences are mixed
  (`4/1/5` higher/tied/lower) with mean `-.010 pp`.
- Interpretation: the bounded SGD tie does not transfer into a better
  wide-range SGD setting.  Adam supplies no practically meaningful gain and
  is consistent with its bounded surface already being bracketed.  Retain the
  original Conv1 legacy SGD and Adam Conv LRs for bias-free work unless a new
  matched, multi-seed selection study establishes otherwise.
- Limitations and deviations: one seed and ordinary MNIST only.  Historical
  controls ran on Jean Zay V100s at `ac941dff`; counterfactuals ran on the local
  RTX 3090 from a dirty worktree based at `68d2f855`, although data orders and
  scientific configs are matched.  An unrelated GPU process started after
  launch and affected runtime only.  The production commands omitted explicit
  `--study-id`, so valid bundles have top-level `study_id=production` while
  their resolved configs retain the intended full study ID; this reporting
  label deviation is recorded and does not affect metrics or checkpoints.
- Artifacts: [review](../results/perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1/analysis/report.md),
  [comparison table](../results/perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1/analysis/comparison.csv),
  [epoch table](../results/perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1/analysis/epoch_metrics.csv),
  [bounded points](../results/perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1/analysis/bounded_evidence.csv),
  [trajectory plot](../results/perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1/analysis/plots/validation_accuracy_by_epoch.png),
  and [validation receipt](../results/perfectdiode-conv1-legacy-zero-bias-conv-lr-times3-bptt-sgd-adam-10ep-seed0-20260816-v1/analysis/validation_receipt.json).

## `perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1` — direct EqProp beta-boundary training test

- Analyzed: 2026-08-15
- Evidence class: ordinary-MNIST exploratory beta selection; non-paper-facing.
- Outcome: `mixed`. Seven of eighteen candidate/control pairs meet the frozen
  beta-collapse criterion, one additional candidate is practically unusable
  at chance, and ten do not confirm collapse. All eighteen one-decade-lower
  controls are stable.
- Scientific question: Do the original worst-layer cosine-`.99` candidate
  betas actually cause training collapse in Conv1, Conv2, and Conv3, and does
  moving one decade lower remove that instability under both SGD and Adam?
- Frozen setup: Conv1/Conv2/Conv3 x baseline/ours/legacy x SGD/Adam x original
  candidate/one-decade-lower beta, 36 cases total; seed-0 ordinary-MNIST
  55,000/5,000 split; ten epochs; exact fixed optimizer vectors; centered
  frozen-current EqProp; true float64; `T=K=8`; `[0,100]` Kaiming weights;
  zero endpoint read noise; and no official-test read. Candidate/control
  injected betas are Conv1 `1000/100`, `300/30`, `30/3`; Conv2 `1000/100`,
  `100/10`, `.3/.03`; and Conv3 `1000/100`, `30/3`, `.01/.001` for
  baseline, ours, and legacy. Collapse is an explicit
  `NonFiniteTrainingError` or a final-validation drop of at least `5.00 pp`
  from that run's best epoch.
- Coverage and integrity: all 18 two-case V100-32GB allocations in Jean Zay
  UMG array `1017012_[0-17]` completed with exit `0:0` in
  `00:31:45--02:21:11`. The exact staged source passed 43 focused local tests,
  31 staged Jean Zay tests, and `sbatch --test-only` job `1017005`. All 36
  one-batch scientific smokes passed sequentially on Trex. The first
  concurrent Trex pack is retained and excluded after a Trex-only Conv3 CUDA
  OOM; its Conv1 peer passed, and no production V100 had that failure. The
  authoritative local production copy has zero itemized differences under an
  rsync checksum dry-run against Jean Zay. All 36 canonical bundles, 36 run
  receipts, 18 pack receipts, source/config hashes, architecture-shared
  initialization hashes, split/order hashes, true-float64 guards, and zero
  official-test-read guards validate. Production contains 30 finite
  completions and six explicit scientific non-finite endpoints.
- Headline measurements: confirmed collapse occurs for all six Conv1/Conv2
  Adam candidates and Conv3-ours SGD. The non-finite endpoints are Conv1
  baseline Adam (`e2/b164`), ours Adam (`e1/b2137`), legacy Adam
  (`e3/b3047`); Conv2 baseline Adam (`e4/b3422`) and ours Adam (`e4/b2493`);
  and Conv3 ours SGD (`e1/b75`). Conv2-legacy Adam completes but drops
  `13.00 pp`, from `97.74%` best to `84.74%` final. Conv1-legacy SGD remains
  finite but flat at `10.00%`, finishing `85.62 pp` below its matched control;
  this is classified as degradation without internal collapse because it has
  no within-run drop. The remaining ten candidate pairs are stable under the
  frozen rule. Every one-decade-lower control completes ten finite epochs and
  finishes within `.28 pp` of its own best validation.
- Interpretation: the original cosine candidate is not a uniformly safe
  training beta, and its usable bound is optimizer-dependent. A one-decade
  margin is a clean-stable safety point for every tested MNIST contract, but
  the ten stable candidate pairs show that this experiment does not locate a
  universal maximum. The result supports scheme-, depth-, and optimizer-aware
  beta qualification. It does not select between the one-decade signal margin,
  the two-decade perturbative margin, or a matched-output-displacement rule.
- Limitations: one seed, ordinary MNIST, fixed LR vectors, and only ten epochs;
  therefore diagnostic rather than paper-facing. A harder dataset could
  narrow the safety margin. Conv3 baseline retains the predeclared
  residual-unqualified `T=K=8` cosine-boundary caveat. The excluded Trex OOM
  means smoke transport was sequential while production used demonstrated
  two-worker V100 packs; scientific configs were unchanged.
- Artifacts: [beta decision record](beta_study.md),
  [analysis report](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/report.md),
  [pair table](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/pairs.csv),
  [case table](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/cases.csv),
  [full summary](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/summary.json),
  [collection receipt](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/collection_validation_receipt.json),
  and [trajectory plot](../results/perfectdiode-conv123-centered-float64-eqprop-beta-collapse-boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1/analysis/validation_accuracy_trajectories.png).
- Provenance: frozen base commit `68d2f855`; staged source archive SHA-256
  `e0d4e1762c87e4247a8eccf09c34adb2ac540b9c3ef77a037fe2568327433577`;
  study-config SHA-256
  `ddc0f81e1ed2b7b6e548c5ef5ca780bb55dfe39bd3135804de827546dafb7326`;
  materialized-config-set SHA-256
  `009b77ba430eb72558dbb4927586d659c9a0e9d68b39f0e5aa838cd5345c7135`;
  Jean Zay job `1017012` under `umg@v100`.

## `perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1` — matched Conv1 historical-bias ablation

- Analyzed: 2026-08-15
- Evidence class: `ordinary_mnist_bias_ablation`; diagnostic and non-paper-facing.
- Outcome: `positive` for the predeclared practical-similarity criterion. No
  material validation-accuracy effect is observed when the active historical
  positive-only bias is replaced by an exact-zero bias in this matched Conv1
  case.
- Scientific question: Does the current nonnegative-projected hidden-bias
  implementation materially change baseline perfect-diode Conv1 validation
  performance relative to fixing the otherwise identical Bias_0 at zero?
- Frozen setup: baseline Conv1, BPTT, plain SGD, seed 0, ordinary-MNIST
  55,000/5,000 train/validation split, ten epochs, `T=K=4`, input gain `40`,
  `[0,100]` Kaiming-initialized conductance weights, and weight LRs
  `C0=.142696,D=.0120238`. The only scientific difference is Bias_0 LR
  `.142696` versus `0`; both biases initialize exactly to zero. The official
  test is disabled. The predeclared threshold is an absolute best-validation
  difference no larger than `.5 pp`, conditional on the learned arm acquiring
  positive bias values.
- Coverage and integrity: both one-batch CPU smokes and both sequential Akib
  RTX 3080 production arms pass. Production ran under `nohup` PID `3957295`
  from `2026-08-15T14:35:25Z` to `14:44:47Z` with exit `0`. Both ten-epoch
  canonical bundles validate locally and remotely; all parameter tensors are
  exactly matched at initialization; train, validation, first-epoch, and full
  ten-epoch minibatch-order hashes match; every metric is finite; and the
  official test is never read. The 112-path local/remote content manifests
  share SHA-256 `5ff107f12d6f...8e9fc68557d`.
- Headline measurements: both arms select epoch 7 at `96.28%` best validation
  and finish at `96.00%`. Accuracy is identical in 7/10 epochs and differs by
  at most `.02 pp` in any epoch. Final loss is `.0923912023` with learned
  positive-only bias and `.0923836840` with fixed-zero bias; the maximum
  epoch-level loss difference is `9.574e-5`. At best/final, respectively,
  `94.85%/96.76%` of the learned arm's 12,544 Bias_0 entries are positive;
  the final range is `[0,.0163113]`. Every learned-bias checkpoint is
  nonnegative, and every control-bias checkpoint is exactly zero.
- Interpretation: this active comparison meets the practical-similarity
  criterion and supports describing the affected simulation matrix explicitly
  as bias-free or biases fixed at zero. It does not validate the separate
  signed/amplification-scaled learned-bias contract. Historical learned-bias
  and fixed-zero rows must not be pooled as one contract.
- Limitations: one architecture/scheme/optimizer and one seed; ordinary MNIST
  only; no uncertainty estimate or official-test read. This is neither a
  statistical equivalence test nor paper-facing accuracy evidence. Baseline
  has `voltage_amp=current_amp=1`, so this case tests active positive-only
  learning versus zero but not the historical depth-scaling confound.
- Artifacts: [bias state review](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/bias_state_review.md),
  [analysis report](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/analysis/report.md),
  [run table](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/analysis/summary.csv),
  [epoch table](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/analysis/epoch_metrics.csv),
  [bias audit](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/analysis/bias_checkpoint_stats.csv),
  [trajectory plot](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/analysis/accuracy_loss_trajectories.png),
  and [collection receipt](../results/perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-seed0-20260815-v1/analysis/collection_validation_receipt.json).
- Provenance: frozen base commit `68d2f855`; source archive SHA-256
  `f10761eedb120ded72a9c6e728ebe2ed79e158cffa7d48ebf6c5a6460a74a9b3`;
  Akib launcher PID `3957295`; config SHAs `366054ee...af8f` and
  `4f17359f...7dc`.

## `perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1` — Conv1/2/3 Adam upper-weight-limit sensitivity

- Analyzed: 2026-08-15
- Evidence class: ordinary-MNIST exploratory / noncanonical.
- Outcome: `positive`. The tight `1e-4` training ceiling materially suppresses
  accuracy, increasingly so with depth for baseline and ours, and explains a
  substantial part of legacy's preliminary advantage.
- Scientific question: With each architecture/scheme's best observed Adam
  learning-rate vector held fixed, how does validation accuracy change when
  only the post-update upper weight limit increases from `1e-4` to `5e-4` or
  `1e-3`?
- Frozen setup: Conv1/2/3 perfect-diode BPTT at accepted `T=K=4/6/8`;
  baseline, ours, and legacy; Adam only; seed 0; ordinary MNIST; respectively
  10/30/30 epochs; exact-zero biases; exact architecture-shared checkpoints
  sampled once from `Uniform[1e-5,1e-4)`; fixed transferred raw LR vectors;
  lower projection limit `1e-5`; and no official-test read. The upper limit is
  the only within-surface factor.
- Coverage and integrity: Jean Zay R3 arrays `979501_[0-8]`, `979502_[0-8]`,
  and `979503_[0-8]` completed all 27 tasks with exit `0:0`, consuming
  `51.479` GPU-hours. All 27 canonical local bundles, task summaries, and
  validation receipts pass. Every run has complete epoch/checkpoint coverage,
  preserves its architecture's exact initial-state hash, shares the same MNIST
  split and first-epoch order, keeps biases exactly zero, obeys its configured
  interval, and records no official-test read. The local and remote production
  trees both contain `3,331,146,916` apparent bytes under `runs/`; their 108
  canonical files share aggregate SHA-256
  `8e27fab62fa13863d7561365e487ec9fbf24a7382e47d041c0e10cd50bf2dbee`.
- Headline measurements: best validation from `wmax=1e-4` to `1e-3` changes
  by Conv1 baseline/ours/legacy `+5.12/+2.10/+.90 pp`, Conv2
  `+7.92/+6.88/+1.36 pp`, and Conv3 `+11.94/+11.26/+4.18 pp`. Best values at
  `1e-3` are Conv1 `94.80/95.02/96.68%`, Conv2 `93.52/97.18/98.20%`, and
  Conv3 `89.60/95.80/97.86%` for baseline/ours/legacy.
- Mechanism and interpretation: final exact upper-endpoint occupancy across
  the nine surfaces falls from `0.89%`--`7.79%` at `wmax=1e-4` to
  `0.03%`--`1.64%` at `1e-3`. Together with the paired accuracy gains, this
  supports upper-bound projection throttling as the mechanism. The
  legacy-minus-ours best-accuracy gap contracts from `2.86 -> 1.66 pp` in
  Conv1, `6.54 -> 1.02 pp` in Conv2, and `9.14 -> 2.06 pp` in Conv3. The tight
  ceiling therefore explains much, but not all, of legacy's advantage.
  Legacy gains only `0.00/.04/.18 pp` from `5e-4` to `1e-3` for Conv1/2/3, so
  `5e-4` is effectively sufficient for legacy; deeper baseline/ours still
  benefit from `1e-3`.
- Limitations: one seed and ordinary MNIST make this diagnostic rather than
  paper-facing evidence. Architecture epoch budgets differ, so depth-level
  comparisons are descriptive. Within-surface deltas isolate the ceiling
  conditional on a fixed initializer and transferred LR vector; cross-scheme
  levels remain confounded by different LR vectors. Conv2 baseline and Conv3
  ours inherit open-upper-Conv-rho source searches, while Conv3 legacy inherits
  a noncanonical below-gate LR source. This study is not a new LR handoff.
- Artifacts: [analysis report](../results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/analysis/report.md),
  [full table](../results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/analysis/summary.csv),
  [surface table](../results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/analysis/surface_summary.csv),
  [accuracy plot](../results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/analysis/accuracy_vs_weight_max.png),
  [endpoint-occupancy table](../results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/analysis/endpoint_occupancy.csv),
  and [collection/validation receipt](../results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/analysis/collection_validation_receipt.json).

## `perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1` — Conv1/Conv3 clean-beta qualification

- Analyzed: 2026-08-14
- Evidence class: ordinary-MNIST exploratory / noncanonical.
- Outcome: `positive` for clean stability and `inconclusive` for choosing
  between one- and two-decade beta margins.
- Scientific question: For Conv1 and Conv3 baseline/ours/legacy, do centered
  true-float64 EqProp training trajectories remain finite and competitive at
  one and two decades below each independently measured worst-layer
  cosine-`.99` injected-beta boundary?
- Frozen setup: seed 0; ordinary MNIST deterministic 55k/5k train/validation
  split; exact current Adam vectors; ten epochs; `T=K=8`; centered
  frozen-current EqProp; endpoint read-noise sigma `0`; Kaiming initialization
  and projection interval `[0,100]`; explicit diode dictionaries; no
  official-test read. Injected beta is Conv1 baseline/ours/legacy
  `100/30/3` versus `10/3/.3`, and Conv3 `100/3/.001` versus
  `10/.3/1e-4`, at one versus two decades below the measured boundary.
- Coverage and integrity: Jean Zay UMG production array `980913_[0-5]`
  completed six V100-32GB tasks with exit `0:0`; each GPU ran one Conv1 and one
  Conv3 case concurrently. All 12 canonical bundles, 12 logical receipts, and
  six pack receipts validate. Every trajectory has ten finite epochs, an
  applied optimizer step, zero read-noise draws, and zero official-test
  evaluations. Initialization hashes match within architecture, and all cases
  share exact split and ten-epoch minibatch-order hashes. All copied result
  subtrees match Jean Zay by relative path and content manifest. Frozen source
  archive SHA-256 is `68c17dc5...fb69`; exact-config set SHA-256 is
  `b18c91be...fefc7`.
- Headline measurements: Conv1 best validation at one/two decades is baseline
  `96.24/96.22%`, ours `96.44/96.42%`, and legacy `96.64/96.62%`.
  Conv3 is `97.02/97.06%`, `98.40/98.44%`, and `98.46/98.42%`.
  Two-minus-one differences span only `-.04` to `+.04 pp`; all twelve runs
  remain finite and complete.
- Interpretation: both tested margins are clean-qualified. There is no clean
  accuracy evidence for selecting one over the other, so no beta is selected
  here. The one-decade tier retains a tenfold larger phase signal; the
  two-decade tier retains the larger small-perturbation margin and matches the
  prior Conv2 rule. A subsequent read-noise comparison, not this clean result,
  should resolve that tradeoff.
- Limitations: one seed, ordinary MNIST, ten epochs, and exploratory accuracy;
  not paper-facing evidence. Conv3 baseline's source boundary is cosine-only
  and residual-unqualified at `T=K=8` because its shared free state was
  under-relaxed. These training results show stability at the tested betas but
  do not repair that boundary qualification.
- Operational provenance: first live canary `980540` completed both packed GPU
  smokes but failed only in the post-run receipt validator, which read launch
  beta keys rather than trainer-runtime beta keys. The attempt is retained and
  excluded. The regression fix passed `58/58` focused tests and validated the
  preserved GPU outputs end to end; isolated replacement canary `980865`
  passed the scheduler and semantic gates before production.
- Artifacts: [analysis report](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/report.md),
  [full table](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/summary.csv),
  [one-vs-two-decade table](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/one_vs_two_decades.csv),
  [trajectory plot](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/validation_accuracy_trajectories.png),
  [scientific verification](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/verification.json),
  and [collection verification](../results/perfectdiode-conv13-centered-float64-eqprop-beta-qualification-1to2decades-10ep-seed0-20260814-v1/analysis/collection_verification.json).

## `perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1` — Conv2 high endpoint-read-noise extension

- Analyzed: 2026-08-17
- Evidence class: `ordinary_mnist_eqprop_read_noise_training_diagnostic`;
  exploratory and non-paper-facing.
- Outcome: `positive` for a selective robustness gap and `negative` for
  literal noisy-training failure. All six runs remain finite and complete,
  while legacy has the largest clean-relative penalty at both added noise
  levels.
- Scientific question: Does the Conv2 one-decade beta remain trainable when
  endpoint read-noise sigma increases from the earlier `1e-4` cap to
  `3e-4` and `5e-4`, and does amplification scheme change the resulting
  degradation?
- Frozen setup: Conv2 baseline/ours/legacy; injected beta `100/10/.03`
  (base beta `100/.625/.0001171875`); current scheme-specific Adam LR
  vectors; centered frozen-current EqProp; true float64; `T=K=8`; seed-0
  ordinary MNIST; exact matched initialization, split, minibatch order, and
  endpoint-noise draws; ten epochs; noiseless validation; no official-test
  read. Noise is added only to copied equilibrium endpoint voltages used by
  the local EqProp gradient.
- Coverage and integrity: Jean Zay array `910848_[0-5]` completed all six
  tasks with exit `0:0` in about 58--60 minutes each. The authoritative local
  copy is byte-identical to the remote files. All six canonical bundles,
  summaries, receipt hashes, initialization/split/order/noise matching, finite
  histories, and zero-test-read guards validate.
- Headline measurements: final validation at sigma `3e-4` is baseline/ours/
  legacy `96.76/97.60/97.14%`; at `5e-4` it is
  `96.78/97.44/96.90%`. Against this study's matched clean anchors, the
  sigma-`5e-4` penalties are `-.22/-.42/-1.44 pp`. Legacy penalties are
  `-1.20/-1.44 pp` at the two added sigmas, versus baseline
  `-.24/-.22 pp` and ours `-.26/-.42 pp`.
- Interpretation: ours has the highest final validation at both new noise
  levels, and legacy is the most noise-sensitive on this matched seed-0
  surface. Together with the later SGD repeat, this qualitative result holds
  under both tested optimizer/LR contracts, although its magnitude changes.
- Limitations and deviations: one seed, one architecture, ordinary-MNIST
  diagnostic accuracy, and an absolute simulator-voltage noise scale that is
  not a calibrated hardware or ENOB claim. Small non-monotonic differences
  between adjacent noise levels are not a trend. The separate live canary was
  explicitly bypassed for this exploratory launch; six local semantic smokes
  and every production live guard passed.
- Artifacts: [analysis report](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1/analysis/report.md),
  [terminal table](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1/analysis/terminal_summary.csv),
  [noise curve](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1/analysis/final_validation_vs_noise.png),
  and [verification](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-3em4-5em4-10ep-seed0-20260813-v1/analysis/verification.json).

## `perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1` — bounded-Kaiming raw-LR transfer diagnostic

- Analyzed: 2026-08-17
- Evidence class: ordinary-MNIST bounded-initializer diagnostic; explicit
  noncanonical cross-initializer LR transfer.
- Outcome: `negative` for the hypothesis that replacing bounded-uniform with
  bounded-Kaiming initialization explains the preliminary legacy advantage at
  the transferred Adam rates.
- Scientific question: If the bounded-uniform source cells' exact raw Adam LR
  vectors are held fixed, does changing only to gain-1 bounded-Kaiming
  initialization reduce the legacy-minus-baseline or legacy-minus-ours gap?
- Frozen setup: Conv1/Conv2 x baseline/ours/legacy; seed 0; ordinary MNIST;
  three epochs; Adam; accepted `T=K=4/6`; exact-zero biases; projection to
  `[1e-5,1e-4]`; architecture-shared bounded-Kaiming initial states; exact
  bounded-uniform source LR vectors transferred without reselection; matched
  dataset/order hashes; no official-test read.
- Coverage and integrity: all six Jean Zay `umg@v100` array `911653` tasks
  complete with exit `0:0`. All six canonical bundles validate, remain within
  the configured interval, preserve exact-zero biases, share the intended
  architecture-level initial states and data-order hashes, and select epoch 3
  as best.
- Headline measurements: Kaiming-minus-uniform final-accuracy changes are
  Conv1 baseline/ours/legacy `-.42/-.28/-.16 pp` and Conv2
  `-.22/-.32/-.18 pp`. Legacy margins change by only `+.04` to
  `+.26 pp`, and none shrinks. Final bound occupancy is baseline/ours/legacy
  `28.20/15.23/4.73%` for Conv1 and `45.50/22.04/15.04%` for Conv2.
- Interpretation: the scheme ordering and gaps reproduce almost unchanged,
  so initializer family alone does not explain legacy's advantage conditional
  on these transferred rates. The occupancy ordering is consistent with
  stronger projection pressure in baseline and ours, but it is associative
  rather than causal because schemes use different LR vectors.
- Limitations and deviations: the user explicitly authorized raw LR transfer
  across initializers, overriding the canonical protocol requirement to
  reselect rho independently. This is a three-epoch, one-seed, Conv1/Conv2,
  Adam-only diagnostic, not a bounded LR handoff, convergence result, or
  paper-facing accuracy result. Conv2 baseline retains the source search's
  open upper-Conv-rho caveat.
- Artifacts: [analysis report](../results/perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1/analysis/report.md),
  [summary table](../results/perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1/analysis/summary.csv),
  [legacy-margin table](../results/perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1/analysis/legacy_margins.csv),
  [epoch trajectories](../results/perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1/analysis/epoch_trajectory.csv),
  and [comparison plot](../results/perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1/analysis/accuracy_comparison.png).

## `perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1` — Conv1 bounded absolute-scale repeat

- Analyzed: 2026-08-13
- Evidence class: ordinary-MNIST exploratory / noncanonical.
- Outcome: `negative` for the hypothesis that small absolute conductance values
  materially cause the weak bounded Conv1 results through numerical
  imprecision.
- Scientific question: Do the recent bounded-uniform Conv1 Adam results improve
  when the full weight interval and initialization are scaled from
  `[1e-5,1e-4]` to `[1e-1,1]` and every nonzero raw LR is scaled by the same
  `1e4` factor?
- Frozen setup: baseline, ours, and legacy; seed 0; ordinary MNIST; three
  epochs; accepted `T=K=4`; exact-zero biases; the three source cells' selected
  raw Adam vectors; no official-test read. The shared initializer preserves
  every source quantile by multiplying the exact source checkpoint tensors by
  `1e4`; the dynamic range remains 10:1.
- Coverage and integrity: all three local RTX 3090 production bundles from
  `main` handle `@142` completed with launcher exit `0`, validate canonically,
  contain three epoch rows and four checkpoints, share initial-state SHA-256
  `71bfa790...6d7e9c`, obey `[.1,1]`, and match the source split and first-epoch
  order hashes. The unchanged three-case CUDA smoke passed. A preceding
  restricted-sandbox attempt failed before model construction because CUDA was
  hidden and is retained and excluded.
- Headline measurements: final small/scaled accuracies are baseline
  `88.84/88.84%`, ours `91.60/91.60%`, and legacy `95.42/95.40%`; deltas are
  `0.00/0.00/-.02 pp`. The largest absolute difference across all nine epoch
  accuracies is `.04 pp`. Final exact-bound occupancy changes only
  `29.8716 -> 29.8549%`, `19.2797 -> 19.2733%`, and
  `4.1844 -> 4.1610%` for baseline/ours/legacy.
- Interpretation: increasing the absolute weight and LR scale by four decades
  recovers no performance and leaves the epoch trajectories, endpoint
  occupancy, and `legacy > ours > baseline` ordering effectively invariant.
  This is evidence against material small-value numerical imprecision in these
  cells. The bounded 10:1 dynamic range, projection behavior, or another
  scale-invariant part of the bounded contract remains more plausible.
- Limitations: Conv1, Adam, seed 0, the selected vectors, and three epochs only;
  no SGD, depth, long-convergence, or medium-affine claim. Scaling weights and
  LRs together tests the complete absolute-scale transformation rather than
  isolating one arithmetic operation. Adam epsilon and solver tolerances remain
  fixed. Normalized final weights differ by `.796%`--`3.007%` RMS rather than
  bitwise reproducing, and the reviewed small-scale sources and repeat retain
  different code-state provenance.
- Artifacts: [review](../results/perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1/review.md),
  [analysis report](../results/perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1/analysis/report.md),
  [summary table](../results/perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1/analysis/summary.csv),
  [epoch comparison](../results/perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1/analysis/epochs.csv),
  and [accuracy plot](../results/perfectdiode-conv1-bounded-uniform-scale1e4-adam-3ep-seed0-20260813-v1/analysis/accuracy_comparison.png).

## `perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1` — Conv2 upper-weight-limit sensitivity

- Analyzed: 2026-08-13
- Evidence class: ordinary-MNIST exploratory / noncanonical.
- Outcome: `positive`. The tight `1e-4` training ceiling materially suppresses
  accuracy, especially for ours, and explains a substantial part of legacy's
  preliminary advantage under the bounded initializer.
- Scientific question: With the best observed fixed learning-rate vector for
  each Conv2 scheme/optimizer, how does ten-epoch validation accuracy change
  when only the post-update upper weight limit is increased from `1e-4` to
  `{2e-4,5e-4,1e-3}`?
- Frozen setup: Conv2 perfect-diode BPTT at accepted `T=K=6`; baseline, ours,
  and legacy; SGD and Adam; seed 0; ordinary MNIST; ten epochs; exact-zero
  biases; one byte-identical checkpoint whose non-bias weights were sampled
  once from `Uniform[1e-5,1e-4)`; fixed transferred raw LR vectors; lower
  projection limit `1e-5`; and no official-test read. The upper limit is the
  only within-surface factor. User-authorized exploratory execution waived
  repeated scientific security and post-training T/K gates.
- Coverage and integrity: all 24 clean production cases from Jean Zay array
  `921005_[0-23]` completed with exit `0:0`, contain exactly ten epoch records
  and eleven checkpoints, and validate canonically and scientifically. All
  cases share the initializer, split, and minibatch-order hashes; final
  weights obey their configured intervals. The authoritative local and remote
  production-run trees have identical aggregate SHA-256
  `fbea6096d7aa17116f6c54b906905e3b0a3cb1803c956f6f3463ca915652af25`.
- Headline measurements: from `wmax=1e-4` to `1e-3`, epoch-10 accuracy changes
  by baseline `+4.94/+6.44 pp`, ours `+8.30/+9.18 pp`, and legacy
  `+3.20/+1.34 pp` for SGD/Adam. The legacy-minus-ours gap contracts from
  `8.34` to `3.24 pp` with SGD and from `8.88` to `1.04 pp` with Adam.
  Highest epoch-10 values are baseline-SGD `86.92%` at `5e-4`, baseline-Adam
  `90.46%` at `1e-3`, ours-SGD `94.20%` at `1e-3`, ours-Adam `96.74%` at
  `1e-3`, legacy-SGD `97.46%` at `5e-4`, and legacy-Adam `97.78%` at `1e-3`.
- Mechanism and interpretation: final upper-endpoint occupancy falls from
  `0.99%`--`5.27%` across surfaces at `1e-4` to `0.00%`--`0.45%` at `1e-3`.
  The paired accuracy gains support projection throttling as a major reason
  ours and baseline looked worse at the tight ceiling. Baseline-SGD and
  legacy-SGD are effectively flat from `5e-4` to `1e-3`; ours-SGD/Adam and
  baseline-Adam still gain about half a point, so `1e-3` is the best single
  tested ceiling for follow-up, without proving that it is globally optimal.
- Limitations: one seed and ordinary MNIST make this diagnostic rather than
  paper-facing evidence. Within-surface deltas isolate the ceiling conditional
  on a fixed initializer and LR vector; cross-scheme levels remain confounded
  by different LR vectors. Baseline-Adam's source rho search remains open at
  its upper Conv edge. The initializer's `1e-4` sampling ceiling is fixed even
  when the subsequent training ceiling is larger.
- Operational provenance: array `917600` failed before training because of a
  submitted archive-path defect. Array `920119` exposed double array-index
  selection; its cancelled task-0 partial is retained under `failed_attempts/`
  and excluded. Full recovery canary `917728_0` and two-index selector canary
  `920841_[0-1]` passed before clean production `921005`.
- Artifacts: [report](../results/perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1/analysis/report.md),
  [full table](../results/perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1/analysis/summary.csv),
  [surface table](../results/perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1/analysis/surface_summary.csv),
  [accuracy plot](../results/perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1/analysis/accuracy_vs_weight_max.png),
  [occupancy plot](../results/perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1/analysis/endpoint_occupancy_vs_weight_max.png),
  and [collection/validation receipt](../results/perfectdiode-conv2-fixed-uniform-init-wmax-sweep-sgd-adam-10ep-seed0-20260813-v1/analysis/collection_validation_receipt.json).

## `perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1` — Conv1/Conv2 directed bounded-uniform LR map

- Analyzed: 2026-08-13
- Evidence class: ordinary-MNIST exploratory / noncanonical.
- Outcome: `positive` for locating the Conv2 SGD basins and `mixed` for exact
  endpoint occupancy as a learning-rate diagnostic. This is not an LR handoff
  or paper-facing accuracy.
- Frozen setup: Conv1/Conv2 baseline, ours, and legacy with SGD/Adam; exact
  architecture-shared `Uniform[1e-5,1e-4)` initializers; projection to
  `[1e-5,1e-4]`; exact-zero bias rates; seed 0; three epochs; Conv1 `T=K=4`
  and Conv2 `T=K=6`; no official-test read. The exploratory extension disables
  canaries, scientific safety rejection, and post-training T/K rejection.
- Coverage: all 12 shards, 12 terminal semantic receipts, and 257/257 directed
  numeric cells are collected locally with zero nonfinite outcome. The join
  with 114 prior same-contract numeric cells contains 371 distinct coordinates
  and no numeric duplicate. The archival checker passes 12/12 authorities and
  257/257 directed bundles.
- Headline result: reducing rates moves Conv2 baseline-SGD from the prior
  `49.34%` maximum to a bracketed `82.84%` at
  `(rho_conv,rho_dense)=(3.70370e-5,4.11523e-5)` with `20.28%` final clipping,
  and ours-SGD from `69.50%` to a bracketed `85.72%` at
  `(3.33333e-4,1.23457e-4)` with `8.77%` clipping. Thus the earlier Conv2 SGD
  rates were too high for this fixed weight interval.
- Adam result: ours-Adam improves only `+.32 pp` to a bracketed `87.70%` at
  `(0.027,0.01)`. Baseline-Adam improves only `+.18 pp` to `84.54%` at the
  maximum Conv rho `0.243`; it remains formally open but already has `46.83%`
  final clipping. Legacy retains the best observed accuracy: `95.42%` with
  Adam in both architectures.
- Range result: ten of twelve combined surfaces are bracketed. Conv1
  legacy-SGD has a tied upper-Conv plateau, and Conv2 baseline-Adam remains
  open toward higher Conv rho. These are the only scientifically motivated
  sparse extensions if complete boundary closure is required.
- Clipping interpretation: combined within-surface Pearson coefficients span
  `-.744` to `+.449` and Spearman coefficients span `-.645` to `+.623`.
  Clipping is therefore not a universal proxy for LR quality. Conv2 ours-SGD
  is the clearest case where entering the correct lower-rate basin both raises
  accuracy and reduces clipping (`33.07% -> 8.77%`); baseline-Adam instead
  gains slightly while clipping increases. Correlations are descriptive and
  noncausal, and combined Adam rows are host/runtime-confounded across studies.
- Operational note: the jobs finished successfully, but the conversational
  watchdog paused after its second checkpoint and delayed collection/reporting
  until the following day. This affected latency, not the scientific bundles.
  Future exploratory runs must report terminal receipts immediately and defer
  deep post-transfer auditing until after the preliminary result.
- Artifacts: [review](../results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/review.md),
  [combined report](../results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/analysis/combined/report.md),
  [surface table](../results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/analysis/combined/combined_surface_summary.csv),
  [cell table](../results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/analysis/combined/combined_cells.csv),
  [correlations](../results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/analysis/combined/combined_correlations.csv),
  and [accuracy/clipping plot](../results/perfectdiode-conv12-directed-exploratory-lr-search-seed0-20260812-v1/analysis/combined/accuracy_vs_final_clipping_combined.png).

## `perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1` — Conv2 lower-beta read-noise training

- Analyzed: 2026-08-13
- Evidence class: `ordinary_mnist_eqprop_read_noise_training_diagnostic`
- Evidence tier: exploratory / noncanonical.
- Outcome: `positive` for a selective robustness gap, `negative` for literal
  legacy training failure. All lower-beta clean controls and noisy cases stay
  finite and complete ten epochs.
- Scientific question: Can beta be selected by one relative rule across
  amplification schemes while keeping clean centered-float64 EqProp stable
  and exposing a larger endpoint-read-noise penalty for legacy than for ours?
- Frozen setup: Conv2; seed-0 ordinary-MNIST 55k/5k split; current selected
  scheme-specific Adam LR vectors; shared initialization and ten minibatch
  orders; centered frozen-current EqProp; true float64; `T=K=8`; noiseless
  validation; matched endpoint-noise seed/draws; no official-test read. The
  one-decade injected betas are baseline/ours/legacy `100/10/.03`; the
  two-decade betas are `10/1/.003`. These are common nominal 10% and 1%
  safety-margin rules relative to the independently measured Conv2
  worst-layer cosine-`.99` upper boundaries (`>=1000/100/.3`), not
  accuracy-tuned betas; the open baseline boundary makes its margins at least
  one and two decades.
- User-truncated coverage: clean, sigma `1e-5`, and sigma `1e-4` for both beta
  tiers and all three schemes (`18` cases). Six materialized sigma-`1e-6`
  configs were never launched. A partial index-10 bundle interrupted solely
  to change launcher scope is retained and excluded; its clean exact
  replacement is canonical.

| Beta margin | sigma | Baseline final change | Ours final change | Legacy final change | Legacy excess penalty vs ours |
|---|---:|---:|---:|---:|---:|
| one decade | `1e-5` | `-.04 pp` | `+.02 pp` | `-.18 pp` | `.20 pp` |
| one decade | `1e-4` | `-.14 pp` | `-.06 pp` | `-.82 pp` | `.76 pp` |
| two decades | `1e-5` | `-.06 pp` | `-.14 pp` | `-.94 pp` | `.80 pp` |
| two decades | `1e-4` | `-.12 pp` | `-.76 pp` | `-1.92 pp` | `1.16 pp` |

- Interpretation: the two-decade rule gives stable clean controls and the
  clearest noise separation. Clean legacy leads ours by `.54 pp`, but ours
  leads legacy under noise by `.26 pp` at `1e-5` and `.62 pp` at `1e-4`.
  Smaller beta increases sensitivity to fixed absolute read noise, but the
  effect is disproportionate for legacy, consistent with prior layerwise
  displacement/read-noise diagnostics. This study does not itself localize
  the degradation to a layer.
- Recommendation: use one relative beta rule, not one common numeric beta nor
  independently accuracy-picked values. The exploratory paper candidate is
  the two-decade rule (`10/1/.003` in Conv2) with clean versus sigma `1e-4`;
  sigma `1e-5` is a milder supplementary point. Freeze and confirm under the
  final ordinary-MNIST zero-bias/shared-`T/K` contract before the sealed
  official-test evaluation. Treat `1e-4` as a controlled stress level unless
  hardware calibration justifies it. The supported claim is
  robustness/degradation, not failure.
- Integrity: `18/18` canonical bundles validate with zero failures; local and
  Trex hashes match for all 72 canonical manifest/status/metrics/result files;
  initialization, split, and minibatch-order guards pass; frozen-source tests
  passed `17/17`, representative CUDA smokes `12/12`, and final runner,
  correctness-guard, and analysis tests `21/21`.
- Limitations: one seed, ten epochs, ordinary MNIST, absolute uncalibrated
  noise, and post-exploration choice of the recommended margin/noise point.
- Artifacts: [review](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/review.md),
  [terminal table](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/analysis/terminal_summary.csv),
  [penalty plot](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/analysis/noise_penalty_by_beta_tier.png),
  [training curves](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/analysis/terminal_training_side_by_side.png),
  and [verification](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-1to2decades-10ep-seed0-20260812-v1/analysis/verification.json).

## `perfectdiode-conv123-legacy-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1` — legacy Conv1/2/3 beta fidelity and displacement

- Analyzed: 2026-08-12
- Evidence class: `ordinary_mnist_exploratory_checkpoint_gradient_diagnostic`
- Outcome: `negative`. Legacy's centered true-float64 gradient loses
  all-layer agreement with same-T/K BPTT at progressively smaller injected
  beta as depth grows.
- Scientific question: On the same T8 beta grid already used for baseline and
  ours, what is the largest sampled injected beta whose legacy EqProp gradient
  has minimum all-weight-layer cosine `.90`, `.95`, or `.99`, and what signed
  state displacement accompanies it?
- Frozen setup: trained legacy best-validation Conv1/2/3 checkpoints; seed-0
  ordinary-MNIST batch 0 with 16 fixed examples; centered frozen-current
  phases from a common post-free-state start; `T=K=8`; injected
  `B={.001,.003,.01,.03,.1,.3,1,3,10,30,100,300,1000}`. Base beta is
  `B/16`, `B/256`, and `B/4096` for Conv1/2/3. Biases remain active in
  dynamics but are excluded from scoring.

| Architecture | max B, cosine >= .90 | max B, cosine >= .95 | max B, cosine >= .99 |
|---|---:|---:|---:|
| Conv1 legacy | `300` | `100` | `30` |
| Conv2 legacy | `3` | `1` | `.3` |
| Conv3 legacy | `.1` | `.03` | `.01` |

- Headline boundary cosines: Conv1 `.913462/.977821/.994894` at
  `B=300/100/30`; Conv2 `.910511/.976413/.990301` at `B=3/1/.3`; Conv3
  `.933893/.983855/.991996` at `B=.1/.03/.01`. Every lower sampled beta
  through each boundary also passes, and all nine boundaries are
  equilibrium-residual-qualified.
- Interpretation: at the `.99` gate, legacy permits only `30/.3/.01` versus
  ours `300/100/30` across Conv1/2/3. The signed displacement curves show
  rapidly growing downstream perturbations beyond these boundaries. This is
  a checkpoint-gradient diagnostic, not evidence about high-beta training
  accuracy.
- Integrity: authoritative GPU smoke and `39/39` local RTX 3090 production
  bundles validate; handle `@118` exited `0`. The analysis reconciles 39
  curve, 117 per-layer cosine, 117 per-layer signed-displacement, and nine
  threshold rows. Source/checkpoint/cohort/parameter/common-start/frozen-force/
  iteration/dtype/native-arithmetic guards pass. No optimizer was constructed
  or stepped and no official-test example was read.
- Operational exclusions: the restricted-shell smoke could not see CUDA and
  produced no bundle; detached handle `@117` then used the wrong workdir and
  exited `127` before starting any case. The unchanged absolute-path retry
  `@118` is the sole production authority.
- Limitations/deviations: one seed/checkpoint and one minibatch; maxima are
  sampled-grid values rather than interpolated crossings. The reusable
  runner's internal study ID is source-compatible rather than equal to the
  enclosing study ID; the exact inventory and summary bind the extension.
- Artifacts: [review](../results/perfectdiode-conv123-legacy-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/review.md),
  [threshold table](../results/perfectdiode-conv123-legacy-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/max_beta_by_cosine_threshold.csv),
  [per-layer cosine](../results/perfectdiode-conv123-legacy-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/layer_cosine.csv),
  [per-layer displacement](../results/perfectdiode-conv123-legacy-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/layer_displacement.csv), and
  [figure](../results/perfectdiode-conv123-legacy-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/float64_cosine_and_displacement_vs_beta.png).

## `perfectdiode-conv123-baseline-ours-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1` — matched Conv1/2/3 beta fidelity and displacement

- Analyzed: 2026-08-12
- Evidence class: `ordinary_mnist_exploratory_checkpoint_gradient_diagnostic`
- Outcome: `mixed`. Baseline remains above worst-layer cosine `.99` at the
  tested `B=1000` ceiling for every architecture, while ours has a clear
  depth-dependent finite-perturbation boundary; Conv3 baseline is not
  equilibrium-qualified at `T=K=8`.
- Scientific question: On one common low-to-high beta grid, what is the
  largest sampled injected beta whose centered true-float64 EqProp gradient
  has minimum all-weight-layer cosine `.90`, `.95`, or `.99` against
  same-T/K BPTT, and what state displacement accompanies it?
- Frozen setup: trained best-validation Conv1/2/3 baseline and ours
  checkpoints; seed-0 ordinary MNIST batch 0 with 16 fixed examples;
  `T=K=8`; centered frozen-current phases from a common post-free-state start;
  injected `B={.001,.003,.01,.03,.1,.3,1,3,10,30,100,300,1000}`. Baseline
  base beta is `B`; ours uses `B/4`, `B/16`, and `B/64` for Conv1, Conv2, and
  Conv3. Biases remain active in dynamics but are excluded from scoring.

| Architecture | Scheme | max B, cosine >= .90 | max B, cosine >= .95 | max B, cosine >= .99 |
|---|---|---:|---:|---:|
| Conv1 | baseline | `>=1000` | `>=1000` | `>=1000` |
| Conv1 | ours | `>=1000` | `>=1000` | `300` |
| Conv2 | baseline | `>=1000` | `>=1000` | `>=1000` |
| Conv2 | ours | `>=1000` | `300` | `100` |
| Conv3 | baseline | `>=1000`* | `>=1000`* | `>=1000`* |
| Conv3 | ours | `100` | `100` | `30` |

- `>=1000` is an open tested edge, not an estimated crossing. Conv3
  baseline's marked values are cosine-only: the shared free state has
  float64 residual p90 `1.75151` and a hard failure at every beta, leaving no
  residual-qualified point. Every other reported boundary passes all float64
  residual gates.
- Headline cosines at non-baseline limits: ours reaches `.979564` at Conv1
  `B=1000` and `.994410` at `B=300`; `.911775/.984057/.995390` at Conv2
  `B=1000/300/100`; and `.967556/.995566` at Conv3 `B=100/30`. The limiting
  layers are C0, C1, and C2, respectively.
- Interpretation: Ours' `.99` boundary moves downward with depth from
  `B=300` to `100` to `30`, matching the increase in layerwise displacement.
  This is a read-only BPTT-checkpoint diagnostic, not evidence that high-beta
  training is valid or beneficial.
- Integrity: local RTX 3090 launcher exit `0`; authoritative GPU smoke and
  `78/78` production bundles validate. The analysis reconciles 78 curve rows,
  234 per-layer cosine rows, 234 per-layer signed-displacement rows, and 18
  threshold rows. Checkpoint bytes, source files, cohort, parameter tensors,
  phase starts, dtype, and native arithmetic guards pass. No optimizer was
  constructed or stepped and no official-test example was read. An
  interrupted CPU smoke with no result is retained and excluded.
- Limitations/deviations: one seed/checkpoint and one minibatch; sampled
  thresholds are not interpolated crossings. The reusable runner's internal
  study ID is source-compatible rather than equal to the enclosing study ID;
  the exact inventory and analysis summary bind the evidence.
- Artifacts: [review](../results/perfectdiode-conv123-baseline-ours-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/review.md),
  [threshold table](../results/perfectdiode-conv123-baseline-ours-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/max_beta_by_cosine_threshold.csv),
  [per-layer cosine](../results/perfectdiode-conv123-baseline-ours-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/layer_cosine.csv),
  [per-layer displacement](../results/perfectdiode-conv123-baseline-ours-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/layer_displacement.csv), and
  [figure](../results/perfectdiode-conv123-baseline-ours-centered-float64-beta-1em3-to-1e3-tk8-seed0-20260812-v1/analysis/float64_cosine_and_displacement_vs_beta.png).

## `perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1` — Conv3 baseline and ours exploratory LR ladder

- Analyzed: 2026-08-12
- Evidence class: `ordinary_mnist_exploratory`
- Evidence tier: `exploratory_noncanonical`
- Evidence label: **EXPLORATORY / NONCANONICAL ordinary-MNIST diagnostic.**
- Outcome: `positive`. Ours has a higher observed accuracy frontier than
  baseline within both matched optimizer/runtime pairs and is better at most
  shared grid coordinates, although neither the study nor its open boundaries
  authorize an LR selection.
- Scientific question: How do baseline and ours respond to a broad low-to-high
  Conv3 LR grid when occupancy and the other scientific safety rules are
  report-only or disabled?
- Frozen setup: Seed-0 ordinary MNIST; baseline/ours x SGD/Adam; ordered `8x8`
  `(rho_conv,rho_dense)` grid; 64 independent three-epoch restarts per
  surface; exact shared `Uniform[1e-5,1e-4)` initializer SHA-256
  `8ebe9dd916e1e9299c31053e2bb37b4f5121f8322f26af7746a92875e555438f`;
  projection to `[1e-5,1e-4]`; zero biases and bias LRs; `T=K=8`; fixed
  split/shuffle seed 0. No official-test examples were read.
- Coverage and integrity: All `4/4` surfaces and `256/256` declared cells are
  canonical, numeric, finite, and reconciled. Four terminal receipts, nine
  exact collected roots, both launchers, three target-gate attempt trees,
  source `result.json` hashes, CSV/Markdown regeneration, and all four PNGs
  validate. The inventory retains 13 excluded operational/non-authoritative
  bundles and exactly two canceled zero-artifact scheduler attempts.

| Scheme / optimizer | Target | Highest observed rho pair | Actual weight LR vector `(C0,C1,C2,D)` | Accuracy | Loss | Exact final either-bound occupancy | Median projection efficiency | Range status |
|---|---|---:|---|---:|---:|---:|---:|---|
| baseline / SGD | Akib | `(3.7037037e-5,1.3717421e-5)` | `(9.476769e-7,2.8743254e-4,2.6388743e-4,2.6545789e-6)` | `72.64%` | `.286757` | `60.79%` | `.2928` | open toward lower dense rho |
| ours / SGD | Akib | `(1.2345679e-5,4.1152263e-5)` | `(6.4107133e-8,1.9908568e-5,1.8416615e-5,1.6719532e-6)` | **`76.84%`** | `.249718` | `18.97%` | `.9418` | locally bracketed |
| baseline / Adam | Trex | `(1e-3,3.3333333e-3)` | `(6.0547261e-8,6.1436962e-8,6.1456227e-8,2.0994218e-7)` | `76.16%` | `.279628` | `69.75%` | `.2803` | locally bracketed |
| ours / Adam | Trex | `(9e-3,3.3333333e-3)` | `(5.4491566e-7,5.4897250e-7,5.4913819e-7,2.0396740e-7)` | **`80.90%`** | `.229190` | `38.55%` | `.3887` | open toward higher Conv rho |

- Within-optimizer comparison: The observed ours-minus-baseline frontier gain
  is `+4.20 pp` for SGD and `+4.74 pp` for Adam. At each baseline maximum's
  same rho coordinate, ours is `+3.20 pp` for SGD and `+3.32 pp` for Adam;
  at each ours maximum, the matched gain is `+7.58 pp` and `+5.36 pp`.
  Ours is better/tied/worse over the full shared grid in `42/13/9` SGD cells
  and `53/0/11` Adam cells. Baseline and ours share initialization, cohort,
  order, target, and validated host/runtime within each optimizer.
- Endpoint occupancy at the maxima: Per-parameter `(C0,C1,C2,D)` percentages
  are `(94.44,88.89,94.53,12.71)` baseline-SGD,
  `(29.25,31.20,28.85,3.71)` ours-SGD,
  `(100.00,87.13,97.12,32.32)` baseline-Adam, and
  `(72.22,44.01,64.95,5.75)` ours-Adam. This is exact final-checkpoint
  either-bound occupancy, not the percentage clipped at any time during
  training.
- Clipping diagnostics: Accuracy versus pooled exact endpoint occupancy has
  Pearson/Spearman `+.381/+.386` for baseline-SGD, `-.098/-.171` for
  ours-SGD, `+.914/+.928` for baseline-Adam, and `+.774/+.610` for
  ours-Adam (`n=64` each). Descriptive and noncausal: `rho_conv` and
  `rho_dense` jointly vary, so the correlations do not isolate an effect of
  clipping on accuracy.
- Range/LR status: The unique baseline-Adam and ours-SGD maxima are locally
  bracketed by four worse axial neighbors. Baseline-SGD is open toward lower
  dense rho, while ours-Adam peaks at the highest tested Conv rho `0.009` and
  is open upward. These are highest observed accuracies, not selected learning
  rates.
- Target assignment: SGD ran on Akib RTX 3080/PyTorch 2.5.1 and Adam on Trex
  RTX 5090/PyTorch 2.11.0. Within-optimizer scheme comparisons are matched;
  SGD-versus-Adam differences are host/runtime-confounded and must not be
  interpreted causally as optimizer effects.
- Operational provenance: Jean Zay jobs `844032` and `844396` were canceled
  before allocation with exit `0:0`, elapsed `00:00:00`, no start, and zero
  scientific artifacts. The checksum-converged Akib/Trex shards are the sole
  scientific authority.
- Interpretation: The broad grid confirms that the previous narrow Conv3
  searches understated baseline/ours performance. Ours reaches higher
  accuracy with much lower endpoint occupancy at matched high-performing
  coordinates. Positive Adam occupancy correlations are consistent with the
  lowest rates undertraining while both learning and boundary contact rise;
  they are not evidence that clipping itself improves accuracy. The open
  ours-Adam edge motivates a later controlled higher-Conv-rho extension.
- Limitations and protocol deviations: Single seed, ordinary MNIST, three
  epochs, broad joint-rho grid, deliberately disabled canaries/scientific
  safety rejections/post-training T/K, and a cross-optimizer host/runtime
  confound. This study creates neither a canonical LR handoff nor paper-facing
  evidence.
- Results and analysis: [report](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/analysis/exploratory_lr_ladder/report.md),
  [surface table](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/analysis/exploratory_lr_ladder/surfaces.csv),
  [cell table](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/analysis/exploratory_lr_ladder/cells.csv),
  [collection validation](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/provenance/local_collection_validation.json),
  [accuracy heatmaps](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/analysis/exploratory_lr_ladder/plots/final_validation_accuracy_heatmaps.png),
  [endpoint-occupancy heatmaps](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/analysis/exploratory_lr_ladder/plots/endpoint_bound_occupancy_heatmaps.png),
  [diagnostic scatter](../results/perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1/analysis/exploratory_lr_ladder/plots/accuracy_vs_diagnostics.png),
  and [study config](../configs/conv/perfectdiode_conv3_exploratory_lr_ladder_seed0_20260811_v1.json).

## `perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1` — Conv1/Conv2 higher-beta true-dtype EqProp

- Analyzed: 2026-08-12
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Evidence tier: `exploratory_noncanonical`
- Outcome: `mixed`. Centered baseline remains all-layer faithful through
  injected beta `1000` for both Conv1 and Conv2. Centered ours passes through
  `300` for Conv1 but only `100` for Conv2; positive one-sided differencing
  has no passing point on the tested high-beta grid.
- Scientific question: Do the baseline/ours higher-beta findings previously
  measured for Conv3 repeat in Conv1 and Conv2, and how do the one-sided,
  centered, float32, and float64 results change with depth?
- Frozen setup: Accepted BPTT-trained best-validation Conv1/Conv2
  baseline/ours checkpoints, one fixed 16-example ordinary-MNIST validation
  batch, `T=K=64`, frozen-current nudging, active biases excluded from
  gradient scoring, injected `B={100,300,1000}`, and independent native
  float32/float64 replay. Baseline uses base beta `B`; ours uses `B/4` for
  Conv1 and `B/16` for Conv2.
- Centered result: At `B=100/300/1000`, worst-layer float64 cosine is
  `.998945/.997131/.990309` for Conv1 baseline,
  `.998584/.994410/.979564` for Conv1 ours,
  `.999678/.997736/.993496` for Conv2 baseline, and
  `.995390/.984057/.911775` for Conv2 ours. With the all-layer cosine
  `>=.99`, symmetric norm delta `<=.1`, and residual gate, baseline passes all
  six architecture/beta points, ours Conv1 passes `100/300`, and ours Conv2
  passes only `100` in both dtypes.
- One-sided result: There is no passing point. At beta `100`, worst-layer
  float64 cosine is `.914346/.943015` for baseline Conv1/Conv2 and
  `.758829/.263565` for ours; at beta `1000` it is
  `.264351/.397352` and `.141423/.141112`. Centering therefore removes the
  leading one-sided finite-beta error but not the eventual higher-order and
  active-set bias.
- Precision and displacement: Float32 and float64 are nearly coincident over
  this large-signal grid, so failure is finite-perturbation bias rather than
  small-contrast float32 cancellation. At beta `100`, positive float64 output
  displacement is `1.9368/4.8651` for Conv1 baseline/ours and
  `.5663/17.9988` for Conv2; at `1000` it reaches
  `19.371/48.744` and `5.664/180.14`.
- Coverage and guards: The CPU smoke and all 24 local RTX 3090 production
  bundles validate; `main` handle `@114` exited zero without retries. Source,
  checkpoint, parameter, cohort, phase-start, frozen-force, iteration,
  beta-scaling, residual, dtype, and arithmetic guards pass. No optimizer was
  constructed or stepped and the official test split was not read. The
  normalized analysis has 24 context rows and 120 layer/precision rows.
- Reporting deviation: The reusable runner retains its source-compatible
  manifest study ID. The enclosing directory, exact launch script, complete
  inventory, and result hashes bind this Conv1/Conv2 extension.
- Limitations: One seed, one minibatch, ordinary MNIST, BPTT-trained
  checkpoints, and deliberately out-of-source-grid diagnostic betas. This is
  not EqProp training accuracy and does not propose these betas for training.
- Results and analysis: [review](../results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1/review.md),
  [report](../results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1/analysis/report.md),
  [context table](../results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1/analysis/summary.csv),
  [layer table](../results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1/analysis/layer_metrics.csv), and
  [figure](../results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1/analysis/cosine_vs_injected_beta.png).

## `perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1` — lower-T/K endpoint-read-noise replay

- Analyzed: 2026-08-12
- Evidence class: `ordinary_mnist_learning_algorithm_read_noise_tk_diagnostic`
- Evidence tier: `exploratory_noncanonical`
- Outcome: `mixed`. The complete gradient/noise response is effectively
  unchanged when equal iteration budgets are reduced from `T=K=64` to `12`
  or `8`, but the trained baseline checkpoint is not equilibrium-valid at
  either lower budget.
- Scientific question: Does the centered-float64 Conv3 endpoint-voltage
  read-noise hierarchy between baseline, ours, and legacy persist at the
  lower iteration budgets used by training?
- Frozen setup: The same six reconstructed-initialization/best-validation
  checkpoints, frozen 16-example ordinary-MNIST validation batch,
  `T=K=64`-selected injected betas, centered frozen-current nudge, true
  float64 dynamics and endpoint subtraction, 32 matched trials per nonzero
  `sigma_v`, and biases active but excluded from gradient scoring. EqProp,
  its endpoints, residuals, and BPTT are recomputed at each `T=K in {8,12}`;
  the accepted `64/64` tensors are used only for drift comparisons.
- Threshold result: Every acquisition and usable threshold is identical at
  `T=K=8/12/64`: baseline initialization `1e-6/1e-6/1e-6`, baseline best
  `3e-6/3e-6/3e-6`, ours initialization and best
  `3e-7/3e-7/3e-7`, legacy initialization `3e-9/3e-9/3e-9`, and legacy best
  `3e-10/3e-10/3e-10`. All 192 lower-T/K tested-sigma rows exactly match
  their `64/64` acquisition, task, usable, and all-trials-usable gate
  classifications. Ours therefore retains its `100x` initialization and
  `1000x` best-checkpoint usable-noise margin over legacy.
- Clean-gradient and state result: All six lower-T/K clean EqProp gradients
  pass the all-layer cosine `>=.99` and symmetric norm-delta `<=.1` gate
  against same-T/K BPTT. The worst clean cosine across all three budgets is
  legacy-initialization C0 `.990074641`. The largest `8/8` EqProp drift
  versus `64/64` is trained-baseline C0 (cosine `.9999428`, relative L2
  `.0112593`); at `12/12` it shrinks to cosine `.999999945` and relative L2
  `3.45e-4`. Layerwise positive-minus-negative delta RMS changes by at most
  `.053344%` at `8/8` and `.0014916%` at `12/12`.
- Residual qualification: Five of six cases pass the strict residual
  `p90<.01` gate at both lower budgets. Trained baseline is under-relaxed:
  its limiting free Layer 1 p90 is `1.75151` at `8/8` (hard failure) and
  `.0734647` at `12/12`, versus `2.64e-11` at `64/64`. Its numeric `3e-6`
  lower-budget threshold is retained as a gradient-only diagnostic but is
  explicitly null as equilibrium-valid evidence. All other lower-budget
  thresholds are equilibrium-valid.
- Coverage and guards: Corrected runs `tk8-v2` and `tk12-v2` each validate
  with six cases, 11,544 gradient rows and 23,088 state-noise rows; source,
  checkpoint, parameter, input, float64, zero-noise, reference-hash, and
  no-update guards pass. The completed `smoke-tk8-v2` validates. An earlier
  smoke had a manifest-only K64 task-reference label and the first `tk8`
  production attempt was intentionally interrupted after that review finding;
  both are excluded and retained as provenance.
- Interpretation: The baseline/ours/legacy read-noise separation is not an
  artifact of using `T=K=64`; the noisy-gradient curves overlap across
  `8/12/64`. Reducing T/K does expose a separate convergence problem for the
  trained baseline checkpoint, so gradient-only agreement must not be used as
  evidence that every lower-budget state is equilibrated.
- Limitations: One ordinary-MNIST minibatch, one seed/checkpoint family, a
  fixed discrete sigma grid and 32 trials, fixed `64/64`-selected betas, and
  BPTT-trained weights. This is not EqProp training accuracy, formal tail
  evidence, physical-voltage calibration, or hardware ENOB.
- Results and analysis: [comparison report](../results/perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1/analysis/report.md),
  [overview plot](../results/perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1/analysis/tk_overview.png),
  [overlaid noise curves](../results/perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1/analysis/noise_task_cosine_vs_sigma_by_tk.png),
  [threshold table](../results/perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1/analysis/threshold_comparison.csv),
  [residual table](../results/perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1/analysis/residual_summary.csv),
  and [gradient-drift table](../results/perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1/analysis/tk64_gradient_drift.csv).

## `perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1` — Conv3 LR search without occupancy rejection

- Analyzed: 2026-08-11
- Evidence class: `ordinary_mnist_selection`
- Outcome: `mixed`. Removing occupancy as a rejection rule exposes three
  accuracy-eligible legacy-SGD cells and a best observation of `93.62%`, but
  no Conv3 surface yields a formal selected LR because that best remains on
  the expanded lower Conv-rho boundary.
- Scientific question: Under the exact parent initializer, zero bias LRs, and
  unchanged Conv3 LR-search contract, what training performance and endpoint
  clipping appear when bound occupancy is recorded but no longer rejects a
  candidate?
- Frozen setup: Seed-0 ordinary MNIST with the official test unread; exact
  shared `Uniform[1e-5,1e-4)` initializer SHA
  `8ebe9dd9...438f`; projection to `[1e-5,1e-4]`; exact-zero bias tensors and
  bias LRs; baseline/ours/legacy x SGD/Adam; `T=K=8`; 640-step canaries;
  three-epoch candidates; 90% gate; one adaptive boundary-expansion wave.
  Projection-efficiency, non-finite, loss, gradient, and epochwise Conv3 T/K
  gates remain active.
- Legacy-SGD result: Eleven numeric cells all pass three epochwise K64/T/K
  checks. At fixed `rho_dense=.0033333`, reducing `rho_conv` from
  `3.333e-4` to `1.111e-4` improves accuracy `92.56 -> 93.62%` and reduces
  exact endpoint bound occupancy `11.45 -> 7.65%`. The best cell's actual
  LRs are `C0=3.469e-8`, `C1=1.224e-5`, `C2=1.153e-5`, and
  `Dense=9.448e-6`; bias LRs are zero. Its layerwise either-bound occupancy
  is `3.21/9.79/9.39/5.01%`. The middle dense rho is bracketed, but the Conv
  rho is not; runner status is `unresolved_after_boundary_expansion` and
  `selected=null`.
- Other surfaces: Baseline-SGD remains probe-unresolved. Baseline-Adam and
  ours-Adam have no numeric cell because every candidate trips the retained
  projection gate. Ours-SGD completes six cells at `9.94--10.00%`, all of
  which fail post-training T/K. Legacy-Adam completes four cells, peaks at
  `88.66%` with `26.25%` exact clipping, and has no accuracy-eligible row.
- Clipping interpretation: Within legacy-SGD, accuracy versus exact final
  bound occupancy is Pearson/Spearman `-0.695/-0.664` (`n=11`), and the best
  row has the lowest endpoint occupancy. This argues against more endpoint
  clipping being beneficial, but remains descriptive: LRs vary jointly and
  exact final equality is not the removed persistent occupancy-increase
  statistic. Ours-SGD's positive `0.757/0.778` correlation is scientifically
  meaningless because all six runs are at chance and fail T/K.
- Interpretation: On this current V100 surface, the Conv LR was not too small;
  a threefold decrease improves both accuracy and clipping. Cell `046` is a
  next-search center, not a final handoff. The next range test should add a
  lower row near `rho_conv=3.70e-5` while keeping the already bracketed dense
  neighborhood around `.0033333`.
- Counterfactual limitation: The parent legacy-SGD path ran on Trex RTX 5090
  with PyTorch 2.11, while this path ran on Jean Zay V100 with PyTorch 2.5 and
  no bitwise CUDA determinism. Init, split/order, seed, normalized signature,
  and training/model code match, but the retained projection gate branches
  differently at the nominal center and shifts the adaptive grid down. The
  `93.62%` result therefore cannot be attributed to occupancy-rule removal
  alone. A causal comparison requires both policies on one runtime with fixed
  LRs and no re-probe/adaptive-grid drift.
- Guards and validation: Jean Zay job `832379` tasks 0--1 completed `0:0`.
  Remote/staging checksum reconciliation is empty; all `47/47` canonical
  bundles validate. Final analysis has six terminal surfaces, six valid
  receipts, 21 numeric bundles, 84 per-parameter occupancy rows, and zero
  missing, duplicate, in-progress, or invalid authorities. Akib's pre-search
  T/K-audit OOM produced no candidate and is retained; unchanged ours
  surfaces completed on main.
- Limitations: One seed, three-epoch ordinary-MNIST selection only; no long
  confirmation or paper-facing accuracy. Endpoint occupancy is not a causal
  LR diagnostic and report-only artifacts do not reconstruct whether the old
  persistence rule would reject the best row.
- Results and analysis: [study review](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1/review.md),
  [final report](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1/analysis/final/report.md),
  [candidate table](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1/analysis/final/candidate_rows.csv),
  [occupancy correlations](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1/analysis/final/accuracy_occupancy_correlations.csv),
  and [collection validation](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1/provenance/collection_validation.json).

## `perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1` — complete Conv3 current-nudge beta and precision sweep

- Analyzed: 2026-08-11
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Evidence tier: `exploratory_noncanonical`
- Outcome: `mixed`. No baseline, ours, or legacy context has an all-layer
  float32 estimator, but centered float64 has a common all-scheme,
  initialization-and-best window at injected current
  `B in {1e-3,3e-3,1e-2}`.
- Scientific question: Across the complete requested injected-beta range, how
  do bias-excluded Conv3 EqProp--BPTT cosine, matched free/nudged relative
  displacement, and delta RMS vary by layer, checkpoint, amplification
  scheme, native precision, and one- versus two-sided differencing?
- Frozen setup: Hash-verified reconstructed initialization and best-validation
  BPTT weights for Conv3 baseline/ours/legacy; one fixed 16-example
  ordinary-MNIST validation minibatch; `T=K=64`; injected
  `B={1e-3,3e-3,1e-2,3e-2,1e-1,.3,1,3,10,30,100}`; positive one-sided and
  centered frozen-current nudges; independent native float32/float64
  dynamics. Base beta is `B`, `B/64`, and `B/4096` for baseline, ours, and
  legacy. Biases remain active in the dynamics but are excluded from scoring.
- Float32 result: Neither estimator has an all-layer point. For centered
  differencing, the best worst-layer cosine over the full grid is
  baseline `.164965/.335705`, ours `.596808/.355025`, and legacy
  `.161268/.149566` at initialization/best. All 33 undefined cosines are
  exact-zero float32 `C0` EqProp vectors.
- Float64 result: Centered all-layer cosine is at least `.99` with symmetric
  norm error at most `.1` over `.001--100` for both baseline checkpoints,
  `.001--10/.001--30` for ours initialization/best, and
  `.001--.03/.001--.01` for legacy. The common window is therefore
  `.001--.01` on the tested log grid. One-sided legacy has no all-layer point
  at `B>=.001`; its best maximin is `.877393/.847661`, limited by the
  Dense/readout gradient.
- Displacement result: At centered float64 `B=.001`, legacy H0/H1/H2/output
  relative displacement is
  `9.45e-9/2.51e-5/.00469/1.34555` at initialization and
  `1.50e-10/2.37e-7/.000391/.520166` at best. The corresponding output delta
  RMS is `.738209/.187204`. Thus the centered directional window is not a
  small-output-perturbation window; positive and negative motion is nearly
  symmetric and centered differencing cancels the leading one-sided error.
- Residual result: All float64 residual gates pass. Only trained baseline
  float32 `Layer_2` is slightly residual-limited, with p90
  `.01025--.01147` and maximum `.01660`; the shared free/zero state also fails,
  so this is not signed-nudge blow-up. No row crosses the `.1` hard-failure
  threshold, and all ours/legacy float32 residual contexts pass.
- Interpretation: The float32 early-layer failure is a shared numerical
  contrast problem, not legacy-specific evidence. Legacy nevertheless has the
  sharpest tradeoff: its output moves by order one at the smallest common
  centered float64 beta while H0 remains barely displaced. This replay does
  not establish EqProp training accuracy.
- Coverage and guards: Both valid smokes and all 22 production bundles pass
  canonical validation. The aggregate reconciles 132 logical contexts, 264
  dtype contexts, 1,056 joined layer rows, 3,696 residual rows, 4,620
  displacement rows, 660 phase rows, and 132 scaling rows. Positive endpoints
  agree exactly between estimators; source/checkpoint/parameter/dtype guards
  pass; no optimizer step or official-test read occurs. The first sandboxed
  smoke failed before canonical bundle creation because CUDA was not visible;
  the unchanged retry and stress smoke pass and are retained.
- Reporting deviation: Bundle manifests retain the generic source-compatible
  study ID. The aggregate SHA-binds the enclosing plan and launcher and
  reconstructs exact `11 x 2` bundle coverage. At `B=.01`, inherited
  `beta_capped` metadata is true for ours/legacy, but requested, actual, and
  effective beta are unchanged and explicitly validated.
- Limitations: One seed, one fixed ordinary-MNIST minibatch, read-only replay,
  and no EqProp optimizer-step or accuracy experiment.
- Results and analysis: [review](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/review.md),
  [full table](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/analysis/summary.csv),
  [validation summary](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/analysis/summary.json),
  [float32 cosine plot](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/analysis/plots/cosine_vs_injected_beta_float32.png),
  and [float64 cosine plot](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/analysis/plots/cosine_vs_injected_beta_float64.png).

## `perfectdiode-conv3-baseline-ours-float64-beta-observables-20260811-v1` — float64 phase-displacement and squared-drop sweep

- Analyzed: 2026-08-11
- Evidence class: `ordinary_mnist_learning_algorithm_state_diagnostic`
- Evidence tier: `exploratory_noncanonical`
- Outcome: `mixed`. Correctly scaled float64 positive one-sided EqProp has a
  broad all-layer window, but very small beta is cancellation-limited and the
  ours readout becomes finite-beta-limited at the largest tested point.
- Scientific question: Across a very wide float64 beta range, how large is
  the matched free-to-nudged state motion layer by layer, and how much
  separately read phase-local squared-voltage-drop contrast remains for the
  normalized EqProp update?
- Frozen setup: Best-validation BPTT Conv3 baseline epoch `30` and ours epoch
  `28`, one fixed 16-example ordinary-MNIST validation minibatch, positive
  one-sided frozen-current EqProp, common residual-clean `T=K=64`, float64,
  and weight-only scoring. Biases remain active in the dynamics.
- Scaling and grid: Effective/injected beta is
  `B=beta_base*(voltage_amp/current_amp)^3`; baseline therefore uses
  `beta_base=B` and ours uses `B/64`. Nineteen common `B` values span
  `2.8046421e-9` through `1` over `8.55` decades.
- Fidelity result: With the exploratory all-layer criterion cosine `>=.99`
  and symmetric norm error `<=.1`, both schemes first pass at
  `B=2.8046421e-5`. Baseline remains faithful through `B=1` (worst cosine
  `.999999`); ours passes through `.3` (worst `.995143`) but its Dense cosine
  falls to `.947298` at `1`, where output relative displacement is `.197275`.
- State result: At `B=.1`, relative H0/H1/H2/output displacement is
  `1.94e-10/2.01e-9/5.40e-7/1.04e-4` baseline and
  `4.21e-10/6.11e-8/3.13e-5/1.97e-2` ours. Output/H0 absolute displacement
  ratios are about `3.48e3` and `9.62e5`, respectively.
- Squared-drop result: The exact endpoint statistic is `q=dE/dw`, and the
  update is `(q_plus-q_zero)/B`. At `B=.1`, C0
  `RMS(delta q)/RMS(q_zero)` is `7.48e-12` baseline and `7.91e-12` ours,
  exposing a severe common-mode readout requirement even though float64 is
  directionally stable there.
- Coverage and guards: The CUDA smoke and all 19 production bundles validate;
  the analysis reconciles 152 scheme/layer/beta rows and verifies the requested
  beta scaling and archived update identity. No optimizer is stepped and the
  official test split is not read.
- Limitations: One seed/checkpoint per scheme, one 16-example ordinary-MNIST
  minibatch, positive one-sided replay only, and no training-accuracy sweep.
  This characterizes numerical endpoint subtraction and finite-beta bias; it
  is not yet a hardware ADC/ENOB measurement.
- Results and analysis: [review](../results/perfectdiode-conv3-baseline-ours-float64-beta-observables-20260811-v1/review.md),
  [complete layer-by-beta table](../results/perfectdiode-conv3-baseline-ours-float64-beta-observables-20260811-v1/analysis/observables_table.md),
  [state CSV](../results/perfectdiode-conv3-baseline-ours-float64-beta-observables-20260811-v1/analysis/state_displacement_wide.csv),
  [learning-rule CSV](../results/perfectdiode-conv3-baseline-ours-float64-beta-observables-20260811-v1/analysis/learning_rule_wide.csv), and
  [figure](../results/perfectdiode-conv3-baseline-ours-float64-beta-observables-20260811-v1/analysis/beta_observables_loglog.png).

## `perfectdiode-conv3-baseline-ours-eqprop-one-vs-two-sided-higher-beta-seed0-20260811-v1` — higher-beta one- versus two-sided true-dtype EqProp

- Analyzed: 2026-08-11
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Evidence tier: `exploratory_noncanonical`
- Outcome: `negative` for float32 training.  Centered two-sided differencing is
  materially better than positive one-sided differencing, but neither trained
  Conv3 baseline nor ours has an all-layer float32 point through the tested
  order-one-displacement boundary.
- Scientific question: Does increasing the frozen-current nudge beyond the
  previous grid recover early-layer float32 contrast, how large is the
  free-to-nudged motion at each layer, and does centered `-beta/+beta`
  differencing repair the one-sided estimate?
- Frozen setup: Best-validation BPTT Conv3 baseline epoch `30` and ours epoch
  `28`, one fixed 16-example ordinary-MNIST validation minibatch, `T=K=64`,
  true float32 and float64 dynamics, and weight-only scoring.  All primary
  points run on one Trex RTX 5090 surface after the occupied local 3090 lane
  was excluded as transport.
- Nudge and scaling: One `F=-dC/dy` tensor is frozen at the common post-T state
  and reused byte-for-byte for zero and every signed phase in the final-output
  `b_k` current interaction.  Baseline tests injected beta `{100,300,1000}`;
  ours tests `{3,5,10}` using base beta divided by the guarded Conv3 factor
  `64`.  The estimators are `(G+ - G0)/beta_injected` and
  `(G+ - G-)/(2*beta_injected)`, formed entirely in the endpoint dtype.
- Float32 result: Centered baseline C0 rises
  `.314963/.644491/.915589`, but never reaches `.99`.  Centered ours C0 is
  `.024906/.066527/.046875`, while C1 is only
  `.234048/.321744/.494860`.  Increasing beta therefore does not expose a
  usable all-layer float32 gradient.
- Float64 result: Centered gradients remain extremely well aligned at all
  tested points; the worst cosine is `.997707` for baseline beta `1000` and
  `.998990` for ours beta `10`.  This is cancellation of centered finite-beta
  error, not evidence that the perturbation is small.
- Displacement/nonlinearity: Float64 matched-zero output displacement grows
  `.103767 -> .311282 -> 1.037796` for baseline and
  `.591827 -> .986378 -> 1.972761` for ours.  At the largest beta, one-sided
  Dense cosine has fallen to `.709103/.194545`, while centered Dense remains
  essentially one.  Negative and positive displacement magnitudes are nearly
  symmetric; matched positive endpoint hashes are exactly identical across
  estimator runs.
- Residual interpretation: Ours passes every residual gate in both dtypes and
  baseline float64 passes.  Baseline float32 is residual-limited because the
  shared post-T/zero Layer 2 p90 is already `.0119629` against the strict
  `.01` threshold; signed endpoints remain finite in the same narrow range
  rather than blowing up with beta.
- Guards and coverage: Six production bundles and two smokes validate locally.
  All checkpoint/source/cohort, common-state, frozen-force, iteration,
  parameter, and native-dtype guards pass; biases are active but excluded from
  scoring, no optimizer is constructed or stepped, and the official test split
  is not read.  The normalized report reconciles 168 rows and explicitly
  supersedes duplicate local one-sided anchors with matched Trex points.
- Reporting deviation: Individual runner manifests retain the compatible
  source-study ID `perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1--float64-shadow`;
  their enclosing directory, run inventory, hashes, and this manifest provide
  the higher-beta study association.  Canonical bundle validation passes.
- Limitations: One seed, one fixed minibatch, ordinary MNIST, BPTT-trained
  checkpoints, and no EqProp optimizer steps.  Betas above the source `.01`
  injected-current cap are explicit diagnostic extrapolations.
- Results and analysis: [review](../results/perfectdiode-conv3-baseline-ours-eqprop-one-vs-two-sided-higher-beta-seed0-20260811-v1/review.md),
  [layer/displacement report](../results/perfectdiode-conv3-baseline-ours-eqprop-one-vs-two-sided-higher-beta-seed0-20260811-v1/analysis/conv3_eqprop_one_vs_two_sided_higher_beta.md),
  [figure](../results/perfectdiode-conv3-baseline-ours-eqprop-one-vs-two-sided-higher-beta-seed0-20260811-v1/analysis/conv3_eqprop_one_vs_two_sided_higher_beta.png), and
  [normalized CSV](../results/perfectdiode-conv3-baseline-ours-eqprop-one-vs-two-sided-higher-beta-seed0-20260811-v1/analysis/conv3_eqprop_one_vs_two_sided_higher_beta.csv).
  CSV SHA-256 is
  `86296c131142bcfdd9251fe0d22a51fec644095fb4d7d8732d9e02bbb55f1f4e`.

## `perfectdiode-conv3-ours-eqprop-beta-above10-bug-audit-seed0-20260811-v1` — ours beta-above-10 injection and displacement audit

- Analyzed: 2026-08-11
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Evidence tier: `exploratory_noncanonical`
- Outcome: `negative` for a beta-injection bug and for float32 recovery.
  Injected beta is applied exactly once, but the output-to-early-layer response
  is severely attenuated; increasing beta introduces finite-perturbation bias
  before producing an all-layer float32 estimator.
- Scientific question: Was the missing trained-Conv3 ours tail above injected
  beta `10` hiding a beta-scaling or displacement bug, and can still larger
  one- or two-sided nudges recover the early float32 layers?
- Frozen setup: BPTT-trained Conv3 ours best-validation epoch `28`, one fixed
  16-example ordinary-MNIST validation minibatch, `T=K=64`, final-output
  frozen-current nudging, true float32 and float64 dynamics, and weights-only
  scoring.  The new production surface used one free Trex RTX 5090 lane.
- Scaling: The injected magnitude is
  `B=beta_base*(voltage_amp/current_amp)^3=64*beta_base`.  New injected/base
  pairs are `30/.46875`, `100/1.5625`, `300/4.6875`, and `1000/15.625`.
  The output `b_k` interaction and finite-difference denominator use the same
  `B`; the runtime factor, frozen-force identity, common phase starts, and
  paired endpoint hashes all pass.
- Displacement result: Positive float64 H0 relative displacement at
  `B=30/100/300/1000` is
  `1.263e-7/4.147e-7/2.498e-6/1.367e-5`, while output displacement is
  `5.918/19.745/59.441/198.624`.  Absolute RMS shows the same attenuation.
  Output displacement remains almost proportional to `B`, while hidden-layer
  asymmetry and active-set transitions grow at the largest points.
- Gradient result: Centered float64 is still all-layer `>=.99` at `B=30`
  (`.999566/.997234/.995566/.999992`) but fails that threshold by `B=100`.
  Centered float32 never has an all-layer point; its best worst-layer cosine is
  `.481631` at `B=300`.  One-sided Dense cosine is already `-.006012` at
  `B=30` and remains negative, exposing its uncanceled finite-beta term.
- Residual interpretation: Every ours phase and layer passes the strict
  residual gates in both dtypes through `B=1000`.  The high-beta loss of
  direction is finite-perturbation and diode-active-set bias, not residual
  blow-up or insufficient equilibrium iterations.
- Coverage and guards: Eight new production bundles and one CPU smoke validate
  locally; all eight remote/local `result.json` hashes match.  The joined
  analysis adds the paired `B={3,5,10}` anchors, yielding 14 included bundles
  and 112 normalized rows.  Biases remain active but excluded from scoring;
  no optimizer is constructed or stepped and the official test split is not
  read.
- Reporting deviation: Runner manifests retain the compatible source-study
  identity rather than the enclosing tail directory name.  Manifest and
  result IDs agree and validate; the top-level directory, launch script,
  analysis inventory, and review record the study association.
- Limitations: One seed/checkpoint and one minibatch; ordinary-MNIST diagnostic
  replay of a BPTT-trained model, not EqProp training.  These deliberately huge
  betas exceed the source current cap and are failure-audit points, not
  candidate training settings.
- Results and analysis: [review](../results/perfectdiode-conv3-ours-eqprop-beta-above10-bug-audit-seed0-20260811-v1/review.md),
  [exact joined table](../results/perfectdiode-conv3-ours-eqprop-beta-above10-bug-audit-seed0-20260811-v1/analysis/conv3_ours_eqprop_beta_above10.md),
  [normalized CSV](../results/perfectdiode-conv3-ours-eqprop-beta-above10-bug-audit-seed0-20260811-v1/analysis/conv3_ours_eqprop_beta_above10.csv), and
  [figure](../results/perfectdiode-conv3-ours-eqprop-beta-above10-bug-audit-seed0-20260811-v1/analysis/conv3_ours_eqprop_beta_above10.png).
  CSV SHA-256 is
  `0ed5c7207955e32bcd90adf686fa50075f9c4668968e7e69b6c12390a28ce0ec`.

## `perfectdiode-conv3-baseline-ours-eqprop-high-beta-high-tk-seed0-20260811-v1` — high-beta/high-T/K true-dtype EqProp extension

- Analyzed: 2026-08-11
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Evidence tier: `exploratory_noncanonical`
- Outcome: `negative`; neither baseline nor ours has a tested true-float32
  all-layer EqProp--BPTT window, and raising `T/K` through `128/128` does not
  repair the early convolutions.
- Scientific question: Can a much larger positive one-sided frozen-current
  nudge make the trained Conv3 float32 early-layer contrast measurable, and is
  the previous failure instead caused by insufficient free/nudged relaxation?
- Frozen setup: Best-validation Conv3 baseline and ours checkpoints, one fixed
  16-example ordinary-MNIST validation minibatch, true float32 and float64
  dynamics, and weight-only scoring. Biases remain active in the dynamics;
  no optimizer is constructed or stepped and the official test split is not
  read.
- Scaling and grid: `beta_injected=beta_base*(voltage_amp/current_amp)^3`, so
  the factors are baseline `1` and ours `64`. Common injected beta
  `{.01,.1,1}` is replayed at native and `64/64`; `.1` is confirmed at
  `128/128`; the adaptive tail adds beta `3` for both and `{10,30,100}` for
  baseline at `64/64`. Values above the source `.01` cap are explicit
  diagnostic extrapolations, not proposed training nudges.
- T/K result: At injected beta `.1`, the complete baseline and ours layer
  vectors at `64/64` and `128/128` are identical at recorded precision.
  Baseline float64 worst projected-KKT p90 improves from about `.07346` at
  native `12/8` to `2.7e-11` at `64/64`, while float32 stalls near `.01099`
  and its direction remains bad. Ours passes its float32 residual gates yet
  also retains bad C0/C1. More iterations are not the remedy.
- High-beta result: At beta `100`, baseline float32 `C0/C1/C2/D` cosines are
  `.245752/.773328/.999596/.992606`; C0 has norm ratio `4.67` and `83.5%`
  exactly-zero contrast entries, while float64 Dense is already `.992606`.
  Ours reaches the finite-beta crossover earlier: at beta `3`, output relative
  displacement is `.591827`, float64 Dense cosine is `.674475`, and float32
  C0/C1 remain only `.086006/.157173`.
- Displacement mechanism: At beta `.1`, matched-free/nudged float64 relative
  displacements `H0/H1/H2/output` are
  `1.94e-10/2.01e-9/5.40e-7/1.04e-4` for baseline and
  `4.21e-10/6.11e-8/3.13e-5/.01973` for ours. The nudge is strongly
  depth-skewed: the readout leaves linear response before the earliest
  float32 contrast becomes reliable.
- Interpretation: This extension strengthens the numerical
  cancellation/transport explanation. Float64 is converged and aligned at
  the same T/K where float32 is dead or nearly orthogonal. Do not choose a
  larger one-sided beta or extra iterations as the float32 training remedy;
  change precision/contrast acquisition or the learning-rule transport.
- Guards and validation: All 11 included canonical bundles validate, source
  checkpoint and in-memory parameter hashes remain unchanged, all iteration
  contracts pass, and the normalized analysis contains 152 layerwise rows.
  The failed `tk64-baseline-injected-30` and
  `tk64-baseline-injected-300` directories are unrelated-lane CUDA-OOM
  attempts with no result and are excluded; beta 30 is represented by the
  successful `-retry1` authority.
- Limitations: One seed, one fixed 16-example minibatch, ordinary MNIST, and a
  checkpoint gradient diagnostic rather than EqProp training. No beta above
  `100` completed, so the measured claim is the absence of a usable window
  through `100`, with the finite-beta crossover already visible there.
- Results and analysis: [review](../results/perfectdiode-conv3-baseline-ours-eqprop-high-beta-high-tk-seed0-20260811-v1/review.md),
  [layerwise report](../results/perfectdiode-conv3-baseline-ours-eqprop-high-beta-high-tk-seed0-20260811-v1/analysis/conv3_eqprop_high_beta_high_tk.md),
  [figure](../results/perfectdiode-conv3-baseline-ours-eqprop-high-beta-high-tk-seed0-20260811-v1/analysis/conv3_eqprop_high_beta_high_tk.png), and
  [normalized CSV](../results/perfectdiode-conv3-baseline-ours-eqprop-high-beta-high-tk-seed0-20260811-v1/analysis/conv3_eqprop_high_beta_high_tk.csv).
  CSV SHA-256 is
  `450634102b25b8784e5441d7d30fc09102418c878737e7b00f71928be654abe2`;
  the machine-readable summary records every input result hash.

## `perfectdiode-conv3-legacy-eqprop-adc-readout-precision-20260811-v1` — EqProp endpoint-readout precision

- Analyzed: 2026-08-11
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Outcome: `mixed`; independent absolute endpoint conversion needs 44
  nominal ideal-ADC bits at the accepted primary beta, whereas an oracle
  analog-difference front end needs 8.  The latter is an architectural control,
  not a realizable hardware specification.
- Scientific question: How does ideal node-voltage ADC resolution affect each
  minibatch EqProp weight update when the matched zero and positive endpoints
  are read separately, and how much resolution could be saved by subtracting
  the endpoints before conversion?
- Frozen setup: Trained Conv3 legacy best-validation checkpoint, native
  `T=K=8`, one fixed 64-example ordinary-MNIST validation cohort in four
  minibatches, and true-float64 positive one-sided frozen-current EqProp at
  base beta hats `{3e-10,1e-9,3e-9}`.  Biases remain active in the dynamics,
  but the four weight gradients alone are scored.  The official test split is
  not read and no optimizer is constructed.
- Acquisition contracts: The primary arm uses one signed symmetric mid-tread
  quantizer per state layer.  Its range is calibrated on the same cohort's
  zero endpoint with `1.05x` headroom and frozen across phases, betas, and
  minibatches.  The control subtracts states before conversion, retains the
  common state exactly, and uses an oracle per-layer/per-beta difference
  range.  Both recompute the actual local squared-voltage-drop statistics from
  quantized nodes before forming the EqProp numerator.
- Headline measurements:

  | Base beta hat | Effective beta | Worst clean EqProp/BPTT cosine | Separate endpoint threshold | Oracle difference threshold |
  |---:|---:|---:|---:|---:|
  | `3e-10` | `3.4463442e-5` | `.999282` | `44` | `8` |
  | `1e-9` | `1.1487814e-4` | `.998395` | `42` | `8` |
  | `3e-9` | `3.4463442e-4` | `.985886` | `41` | `8` |

  A threshold requires every weight and minibatch to have cosine at least
  `.99` and symmetric norm delta at most `.10` relative to clean float64
  EqProp, and to remain passing at every higher tested integer bit.  No
  pass-to-fail reversal occurs.
- Mechanism: At the primary beta, Layer 1 has full scale `121.9267` but mean
  free-to-nudged RMS displacement only `3.3065e-11`, a `2^41.75`
  range-to-signal ratio.  At 43 absolute bits, its mean p90 displacement is
  `.896` LSB, only `20.63%` of endpoint codes change, and the limiting
  `ConvWeight_0` minibatch cosine is `.983829`.  At 44 bits these become
  `1.792` LSB, `29.08%`, and `.994313`.  Increasing beta improves acquisition
  SNR approximately as expected, but the largest beta's clean Dense EqProp
  cosine to BPTT falls to `.985886`, exposing the finite-beta/readout tradeoff.
- Interpretation: Independent absolute conversion spends nearly all codes on
  common mode and is unattractive for this small-nudge operating point.
  Correlated sample/hold plus differential sensing, or analog/mixed-signal
  formation of the factorized local observable
  `(z_plus-z_zero)(z_plus+z_zero)/(2 beta_eff)`, is the supported direction.
  The next gate should add differential offset/noise, CMRR, temporal
  decorrelation, and a causal range policy before any DAC/write study.
- Limitations: Nominal quantizer bits are not hardware ENOB.  Both range
  calibrations are in-sample; the control is oracle and beta-dependent; read
  noise, converter nonlinearity, gain/offset mismatch, sample/hold error, and
  conductance programming are absent.  The primary path subtracts separately
  computed endpoint statistics in float64, so its threshold combines ADC
  quantization and digital cancellation.  This is one checkpoint and one
  ordinary-MNIST cohort, not paper-facing evidence or a training result.  The
  exact executed analyzer source snapshot was not retained; promotion to
  formal evidence therefore requires a fresh run under the formal tier.
- Guards and validation: The passing `smoke-02` and production bundles
  validate.  Production reconciles all 1,968 expected layer/minibatch rows;
  every residual, coverage, beta-scale, source, and parameter guard passes.
  The current batch-0 primary-beta replay exactly equals all 12 accepted
  archived float64 EqProp, BPTT, and numerator tensors.  Result SHA-256 is
  `1ecbe8557e777316ab7dbef5ee7e2a97d70ffb4fa11781962d63c7bd3790e291`.
- Results and analysis: [study review](../results/perfectdiode-conv3-legacy-eqprop-adc-readout-precision-20260811-v1/review.md),
  [production report](../results/perfectdiode-conv3-legacy-eqprop-adc-readout-precision-20260811-v1/production/report.md),
  [threshold table](../results/perfectdiode-conv3-legacy-eqprop-adc-readout-precision-20260811-v1/production/minimum_passing_bits.csv),
  [state measurements](../results/perfectdiode-conv3-legacy-eqprop-adc-readout-precision-20260811-v1/production/state_adc_metrics.csv),
  and [fidelity plot](../results/perfectdiode-conv3-legacy-eqprop-adc-readout-precision-20260811-v1/production/gradient_fidelity_vs_adc_bits.png).

## `perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1` — bounded-uniform zero-bias LR selection

- Analyzed: 2026-08-10
- Evidence class: `ordinary_mnist_selection`
- Outcome: `mixed`; six of eighteen surfaces produced reportable three-epoch
  selections, while six completed numeric surfaces stayed below the 90% gate
  and all six Conv3 surfaces remained scientifically unresolved.
- Scientific question: With one matched initializer per architecture, which
  weight learning rates are viable for baseline, ours, and legacy
  amplification under the narrow `[1e-5,1e-4]` conductance interval, and does
  final exact-bound occupancy diagnose whether a learning rate is adequate?
- Frozen setup: Seed 0 ordinary MNIST with 55,000/5,000 train/validation
  examples and no official-test read; Conv1/2/3 use accepted `T/K=4/4,6/6,8/8`.
  Each architecture shares one scheme/optimizer-matched checkpoint whose
  weights were sampled from `Uniform[1e-5,1e-4)` and projected to
  `[1e-5,1e-4]`. Hidden biases start at exactly zero and every `Bias_*` learning
  rate is exactly zero. The core rho grid is conv `{0.001,0.003,0.009}` by
  dense `{0.0033333,0.01,0.03}`, with one declared edge-expansion wave;
  Conv3 baseline/ours use their predeclared fixed higher grid. Candidates train
  for three epochs after a restarted 640-step gate, and reportable selections
  must meet 90% validation accuracy.
- Reportable selections:

  | Architecture | Scheme / optimizer | Selected `(rho_conv,rho_dense)` | Actual weight LRs | Accuracy | Exact lower / upper / either bound |
  |---|---|---:|---|---:|---:|
  | Conv1 | ours / SGD | `(0.001,0.0033333)` | `C0 3.980e-8, D 2.017e-6` | `90.18%` | `5.99 / 0.55 / 6.54%` |
  | Conv1 | ours / Adam | `(0.001,0.01)` | `C0 6.055e-8, D 6.086e-7` | `91.60%` | `18.25 / 1.03 / 19.28%` |
  | Conv1 | legacy / SGD | `(0.003,0.0033333)` | `C0 2.582e-8, D 4.386e-7` | `94.50%` | `0.54 / 0.13 / 0.67%` |
  | Conv1 | legacy / Adam | `(0.009,0.01)` | `C0 5.449e-7, D 6.086e-7` | `95.42%` | `3.32 / 0.86 / 4.18%` |
  | Conv2 | legacy / SGD | `(0.003,0.0033333)` | `C0 1.253e-7, C1 5.458e-6, D 1.572e-6` | `93.34%` | `5.41 / 0.80 / 6.21%` |
  | Conv2 | legacy / Adam | `(0.009,0.01)` | `C0 5.449e-7, C1 5.476e-7, D 6.092e-7` | `95.42%` | `13.08 / 1.23 / 14.31%` |

- Unresolved numeric surfaces: Conv1 baseline SGD/Adam peak at
  `87.52/88.84%`; Conv2 baseline SGD/Adam and ours SGD/Adam peak at
  `49.34/84.36/69.50/87.38%`. These are highest observations, not winners.
  Conv2 ours-Adam peaks on the tested conv-rho edge at `(0.009,0.01)` with
  `87.38%` accuracy and `22.19%` pooled exact-bound occupancy, so a larger
  convolutional rate is a weak follow-up hypothesis, but the `0.28 pp` gain
  from conv rho `0.003` to `0.009` at dense rho `0.01` is not a settled
  under-learning-rate diagnosis.
- Conv3 outcome: baseline-SGD fails the stability probe through
  `ConvWeight_0`. Baseline-Adam and ours SGD/Adam have all nine fixed-grid
  cells rejected by the Conv3 safety contract. Legacy SGD/Adam downshift to
  smaller adaptive grids, but each has five canary and four full-candidate
  safety rejections. There is therefore no numeric Conv3 accuracy or clipping
  result and no bounded-uniform Conv3 LR selection.
- Clipping interpretation: Final pooled either-bound occupancy has mixed-sign
  within-surface correlations with accuracy. Pearson/Spearman range from
  `-0.953/-0.817` (Conv1 baseline-SGD) and `-0.932/-0.883` (Conv2 legacy-SGD)
  to `+0.982/+0.967` (Conv1 baseline-Adam); Conv2 ours-Adam is nearly null at
  `-0.081/-0.117`, while Conv2 legacy-Adam is `+0.527/+0.615`. The strongest
  Conv1 legacy rows clip only `0.67/4.18%`, whereas Conv2 baseline-Adam clips
  `44.04%` and remains below gate. Layer occupancy also differs sharply: at
  the selected Conv2 legacy-SGD row, exact either-bound occupancy is
  `8.51/10.93/3.41%` for `C0/C1/D`; for legacy-Adam it is
  `24.91/26.24/7.21%`. Endpoint clipping is consequently a useful saturation
  symptom, but not a causal or standalone LR selector: it loses clipping
  history, update direction and magnitude, near-bound mass, and layer-size
  weighting.
- Interpretation: The uniform initializer removes the earlier initializer
  confound and shows a coherent legacy advantage at this narrow bound, but it
  does not establish viable rates for baseline/ours Conv2 or any Conv3 scheme.
  The grid geometry, not clipping percentage, supports only a limited LR
  follow-up: extend Conv2 ours-Adam conv rho while holding dense rho near
  `0.01`. Do not publish a complete bounded-weight handoff from this study.
- Guards and validation: All 18 expected surfaces are terminal with 18 valid
  semantic receipts. The final analyzer accepts 114/114 canonical completed
  candidates, retains 37 canary and 11 full-candidate scientific safety
  rejections, records 282 per-parameter occupancy rows, and finds zero invalid
  bundles or duplicate authorities. Akib and Trex checksum comparisons are
  empty, and the Jean Zay content mirror is complete; every receipt preserves
  the frozen source, config, and architecture-shared initialization hashes. No
  10/30/30-epoch confirmation was launched.
- Operational provenance: The original main/Akib wrappers completed only their
  first surface before a receipt-writer `hashlib` import defect; the failed
  attempts remain retained and are excluded. Recovery source commit
  `950b845519b6ed91c424cf7accdc59a193b8377f` and archive SHA-256
  `f9f73f5c...6e407fe` passed regression tests and replacement smokes. Conv1
  ran on main, Conv2 on Akib, Conv3 baseline/ours on Jean Zay job `819812`
  tasks 0--1, and Conv3 legacy on Trex. Pending Jean Zay task `819812_2` was
  canceled before allocation and replaced without duplicate execution.
- Limitations: One seed, three-epoch ordinary-MNIST selection accuracy, strong
  safety censoring in Conv3, and no long confirmation or paper-facing
  medium-affine evidence. Cross-surface comparisons mix different physical LR
  mappings and should not be read as causal effects of occupancy.
- Results and analysis: [study root](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/),
  [final report](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/analysis/final/report.md),
  [surface table](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/analysis/final/surface_summary.csv),
  [candidate table](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/analysis/final/candidate_rows.csv),
  [per-parameter occupancy](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/analysis/final/parameter_endpoint_occupancy.csv),
  [accuracy--occupancy correlations](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/analysis/final/accuracy_occupancy_correlations.csv),
  and [correlation plot](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/analysis/final/plots/accuracy_vs_either_bound_occupancy.png).
- Provenance: Study config SHA-256 `ee8688a1...fac6a05`; initializer checkpoint
  SHA-256 is `235d7ae5...cb1f16e` for Conv1,
  `6469bc8f...c4d88d2` for Conv2, and `8ebe9dd9...e555438f` for Conv3.

## `perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1` — positive one-sided EqProp precision and transport

- Analyzed: 2026-08-11
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Outcome: `mixed`; literal amplification-scaled frozen-current EqProp has a
  case-specific small-beta window with all-layer float64 cosine above `0.99`
  for every trained Conv2/Conv3 case, but no precision-certified deep float32
  update and no tested common base beta that works across schemes.
- Scientific question: After replacing centered EqProp with a positive
  one-sided estimator, reducing beta, and scaling its output current by
  `(voltage_amp/current_amp)^L` as the bias current is scaled, do EqProp
  gradients match BPTT layer-by-layer; does the answer depend on T/K,
  residual convergence, amplification scheme, state displacement,
  convolution, or arithmetic precision?
- Frozen setup: seed-0 baseline `v1/c1`, ours `v4/c1`, and legacy `v4/c0.25`
  Conv1/2/3 Adam leaders at reconstructed initialization and best validation;
  one shared 64-example ordinary-MNIST validation cohort; native T/K plus a
  fixed strong-beta cross through `64/64`; true-dtype shadows use the first
  fixed 16-example batch. Biases are active but excluded from scored
  gradients.
- Learning rule and scaling: `nudging_mode=current` freezes
  `force=-dC/dy` at the common post-T state, adds
  `-beta_injected<y,force>`, and reports
  `[dtheta E(s_plus)-dtheta E(s_zero)]/beta_injected`. The literal scaling is
  `beta_injected=beta_base*(voltage_amp/current_amp)^L`, using the guarded
  output bias-current-row exponent `L=1/2/3`. Multipliers for
  baseline/ours/legacy are `1/4/16`, `1/16/256`, and `1/64/4096` for depths
  one, two, and three. This is frozen-current one-sided EqProp, not full-cost
  `E+beta*C` EqProp.
- Supersession: earlier cost-mode bundles only reported the current scaling;
  the runtime ignored `current_scale` in their dynamics. They remain controls
  but are superseded for the literal-current question. The current-mode
  `conv-current-*`, `fc-current-*`, and `conv-shadow-current-*` bundles are
  authoritative.
- Float32 beta result: both the literal common-base and equal-injected Conv
  sweeps select `0/18` contexts by the residual + SNR + adjacent-consistency
  rule. Readouts align first, while trained early convolutions remain poor or
  exactly zero. The strong-endpoint trained vectors `C0/C1[/C2]/D` are:
  Conv2 baseline `-.027/.065/.996`, ours `.007/.184/1`, legacy
  `-.015/.992/.990`; Conv3 baseline `zero/.010/.025/.978`, ours
  `zero/.013/.268/1`, and legacy `zero/.036/.996/.217`.
- Precision result: independent full-dynamics float32/float64 shadows compute
  EqProp and BPTT from identical hashed starts in their native dtype. Every
  trained Conv2/Conv3 case has a tested case-specific beta with all-layer
  float64 cosine above `0.99`. For trained Conv3 legacy, base beta hat
  `3e-10` means base beta `8.4139263e-9`, injected beta `3.4463442e-5`, and
  float64 cosines `.999791/1/1/.999764`; the identical float32 cosines are
  `zero/.006617/.561176/.999764`.
- Complete trained-Conv3 curve: the canonical true-dtype grid has 14 unique
  baseline, 11 ours, and 7 legacy points. All four float64 layers have cosine
  at least `.99` for baseline at base beta hat `1e-6`--`3e-4`, ours at
  `1e-8`--`1e-5` (capped at the last point), and legacy at
  `1e-10`--`1e-9`. Float32 has no all-layer point; `ConvWeight_0` is exactly
  zero at every tested beta for all three schemes. Thus the float32
  first-layer failure is not legacy-specific.
  These are directional windows only: the trained baseline shadow remains
  residual-limited at native `T=12`.
- Common-beta result: no tested common base beta works across all schemes. At
  base beta hat `3e-10`, trained Conv3 baseline/ours first-layer float64
  cosines are `.112/.419`, versus `.999791` for legacy. At `3e-8`, baseline
  is still `.543` and ours is `.999702`, while legacy's Dense cosine falls to
  `.426627` because the `4096x` scale overnudges the output.
- Displacement and finite-beta result: trained Conv3-legacy state displacement
  at the good beta is `5.19e-12/8.17e-9/1.35e-5/.01793`. At injected beta
  `.003446`, output displacement is `1.793` and Dense cosine `.426627`; at
  `.01`, they are `5.202` and `.167644`. All float64 residuals still pass. A
  two-point Richardson extrapolation of the saved Dense gradients recovers
  BPTT with cosine `.999999843` and norm ratio `.999999972`, identifying the
  failure as one-sided finite-beta bias, not an incorrect zero-beta direction.
- T/K and residual result: the fixed strong-beta T/K cross changes finite
  cosines by at most `.007988` between native and `64/64`. Raising T repairs
  trained Conv3-baseline free residuals without repairing its direction;
  trained ours and legacy pass residual gates while their float32 C0 is zero.
  At the good Conv3-legacy beta, the worst float32 projected-KKT p90 is
  `3.05e-4` and float64 p90 is at most `4.55e-13`. Residual convergence does
  not imply a perturbatively small nudge.
- FC control: literal common-base selects only FC1 legacy in native and
  `64/64`; equal-injected selects `0/18`. Best descriptive native worst-layer
  cosine falls from FC1 legacy `.99990` to FC2 legacy `.98181` and FC3 legacy
  `.31322`; native and `64/64` are materially unchanged. Convolution is not
  necessary. FC evidence is initialization-only because matched trained FC
  checkpoints do not exist.
- Interpretation: use base beta hat `3e-10` as the frozen-current float64
  legacy Conv3 pilot point. Do not launch naive float32 deep training. Select
  beta using injected current and layerwise displacement; the `.01` cap is
  residual-safe but not a small-nudge cap for amplified legacy. Treat
  full-cost EqProp as a separate learning-rule experiment.
- Guards and validation: all authoritative Conv, FC, T/K, and shadow bundles
  pass canonical artifact-hash validation, source/checkpoint and in-memory
  parameters remain unchanged, all phases restore their declared common
  state, no optimizer step occurs, and the official test split is never read.
- Limitations: one seed; Conv/FC beta and T/K replays use 64 examples, while
  the true-dtype shadow uses one fixed 16-example batch; FC is initialization
  only; the T/K cross uses the strong endpoint rather than the recommended
  Conv3-legacy beta; no EqProp optimizer step or training-accuracy claim is
  made.
- Results and analysis: [study root](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/),
  [review](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/review.md),
  [Conv beta table](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/conv-current-commonbase/parameter_summary.csv),
  [T/K table](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/conv-current-commonbase-tk/fixed_beta_tk_summary.csv),
  [FC table](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/fc-current-commonbase/parameter_summary.csv),
  [true-dtype heatmap](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/analysis/eqprop_current_commonbase_true_dtype_cosine.png),
  [trained Conv3 true-dtype curve](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/analysis/conv3_current_true_dtype_beta_curve.png),
  [exact Conv3 table](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/analysis/conv3_current_true_dtype_beta_curve.md),
  and [recommended-beta float64 table](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/conv-shadow-current-commonbase-3em10/parameter_precision_comparison.csv).
- Authoritative result SHA-256 values: literal Conv beta
  `9f24538070cc9eeb85b6e11d155e36e228caaebbc133949f3230e9857ad5c80c`;
  equal-injected Conv `2011fbbe0e8f67fb08fb32714d4843cd36e916bd6a49d4ce7666d3928234b946`;
  T/K `685ebceb80aececb9357a778d096ec46e33d33cf2222139122af1b3e813236d8`;
  literal FC `c73e76f1ad4d9195e8e55360e23c9d4da10d820b06f3cee9047316c91b6585cd`;
  equal-injected FC `7124617c736bc1683b7b5d261c10769cc1e325270108fee45fc04ba10a50fc0c`;
  common-base true-dtype `3e-10` shadow
  `85cd345df45300c35717262d5a52765a750c140b134a015e011a481cb2ea0144`;
  completed Conv3 curve CSV
  `369fa9f3001fa8ee82b4a7d5006d0db98ead7d322ae7aa9c763597b669310975`.

## `perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1` — all-scheme EqProp gradient transport

- Analyzed: 2026-08-10
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Outcome: `mixed`; positive for all three Conv1 schemes, negative for a direct
  centered-EqProp replacement of BPTT in Conv2/Conv3.
- Scientific question: How does weight-only, layerwise EqProp--BPTT gradient
  direction vary with beta, independent free/nudged phase lengths T/K, and
  amplification scheme; and how far do the positive and negative states move
  from the free state?
- Frozen setup: baseline `voltage/current=1/1`, ours `4/1`, and legacy
  `4/0.25`; Conv1/Conv2/Conv3 coherent Adam leaders at hash-verified
  reconstructed initialization and best-validation weights; one identical
  64-example/four-batch ordinary-MNIST validation cohort; centered cost nudge;
  beta `{0.003,0.01,0.03,0.1,0.25,0.5,1}`. Native T/K is `4/4`, `6/6`,
  Conv3-baseline `12/8`, and Conv3-ours/legacy `8/8`; the full Cartesian grids
  independently vary Conv1 T,K over `{4,8,16,64}`, Conv2 over
  `{6,12,24,64}`, and Conv3 T over `{8,12,16,32,64}` with K over
  `{8,16,32,64}`. Biases remain active in the dynamics but are excluded from
  gradient scoring.
- Native maximin results:

  | Architecture | Scheme | Init beta / worst cosine | Best beta / worst cosine |
  |---|---|---:|---:|
  | Conv1 | baseline | `1 / 0.998771` | `1 / 0.993006` |
  | Conv1 | ours | `1 / 0.999917` | `1 / 0.995962` |
  | Conv1 | legacy | `0.03 / 0.999936` | `0.1 / 0.974661` |
  | Conv2 | baseline | `1 / 0.202739` | `0.5 / 0.143147` |
  | Conv2 | ours | `0.5 / 0.476067` | `1 / 0.286438` |
  | Conv2 | legacy | `0.003 / 0.321898` | `0.1 / 0.005915` |
  | Conv3 | baseline | `0.25 / 0.042111` | `0.5 / 0.039319` |
  | Conv3 | ours | `0.1 / 0.057662` | `1 / 0.034167` |
  | Conv3 | legacy | `0.003 / -0.998542` | `none / dead C0` |

- Layer/beta interpretation: Conv1 baseline/ours have a readout cosine near one
  throughout and a first convolution that improves toward beta one. Legacy is
  strong only in a narrow beta window and reverses its Conv1 first-convolution
  direction at larger beta. Conv2/Conv3 later layers again align before the
  early convolutions. Trained Conv2 ours is the least-bad deep result, but its
  worst cosine is only `0.286438`; trained legacy is essentially orthogonal.
  Conv3 legacy has a dead C0 at best and an initialization C2 cosine of
  `-0.998542`.
- T/K result: with beta fixed from the shared anchor, K dominates 41/50
  complete layer grids and has range `>0.01` in 17/50. T has only one material
  effect, trained Conv3-baseline C0 (`delta-T=0.059510`,
  `delta-K=0.050393`). Beta retuning can hide K sensitivity, most clearly for
  trained Conv2 legacy. The best retuned trained Conv3-ours coordinate is
  `T/K=8/16`, beta `1`, worst cosine `0.112146`, still not viable. Conv3 legacy
  best is dead at K `8/16`; K `32/64` revives all layers only at beta `0.003`,
  with worst cosine `-0.889231/-0.095818`.
- State displacement: both post-T-free and matched-zero-K references are saved;
  the latter isolates nudging. At selected native points, displacement grows
  monotonically with depth and the concatenated global norm understates the
  output-state motion by `19.6x`--`30599x`. Baseline is nearly phase-symmetric;
  ours and legacy show strong negative-phase asymmetry. Conv2 legacy
  initialization has global matched-zero displacement `0.00476/0.804`
  (positive/negative) and output displacement `0.0931/15.7`. Trained Conv3
  baseline is the important reference exception: residual zero-nudge K
  relaxation makes post-T displacement about `76x` the actual matched-nudge
  displacement. Displacement and cosine are not monotone, so both must be
  gated layerwise.
- Interpretation: Legacy's superior BPTT training does not yield superior deep
  EqProp gradient fidelity. The failure is localized to backward transport
  through early convolutions, not merely one bad global beta or too little T.
  More K can rotate gradients but also produces asymmetric and unstable
  negative phases. Deep EqProp training should wait for a frozen-checkpoint
  intervention such as staged/layerwise nudging or an explicit feedback path,
  with joint gates for finite phases, non-dead gradients, direction, scale,
  and per-layer state displacement.
- Guards and validation: 18 checkpoint cases, 312 T/K contexts, 6,720
  parameter rows, and 36,888 state-summary rows. The canonical production
  bundle and authoritative local copy validate; all source bytes and parameter
  tensors are unchanged; optimizer-step and official-test-read guards are
  false. Of 8,736 phase records, 8,584 are finite and all 152 unstable records
  are negative; no native/shared anchor is unstable. Jean Zay job `823643`
  completed in `01:10:46`. Result SHA-256 is
  `b4071c5fcf9e167e47fed2f7d0a1169edbd703b955e929c2bb6f8ede1238e8de`.
- Limitations: one seed and one 64-example ordinary-MNIST cohort;
  initialization is deterministic reconstruction; cosine measures direction,
  not scale; and this replay does not establish EqProp training accuracy or
  paper-facing medium-affine evidence. The earlier legacy-only replay remains
  provenance but is superseded by this matched all-scheme T/K analysis for
  cross-scheme and phase-length conclusions.
- Raw results and analysis: [study root](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/),
  [reviewed report](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/review.md),
  [runner report](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/analysis/report.md),
  [layer/beta table](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/analysis/parameter_summary.csv),
  [T/K table](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/analysis/fixed_beta_tk_summary.csv),
  [state table](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/analysis/state_displacement_summary.csv),
  and [read-only guards](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/analysis/read_only_guards.json).
- Provenance: clean scientific runtime head
  `154d80e845cbec95ef9ba08981efd3079d765ddb`; analyzer checkout
  `8a5d39c3dc10ecb683cd626fa5bc17ba5946e23b`; runner SHA-256
  `aeb8c2d9...a042d08a`, config SHA-256 `17c372d3...229b1335`, and target
  `jean-zay` V100.

## `perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1` — EqProp/BPTT checkpoint-gradient replay

- Analyzed: 2026-08-10
- Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic`
- Outcome: `mixed`; positive for Conv1 gradient direction, negative for directly
  reusing the accepted BPTT phase contract in Conv2/Conv3 EqProp.
- Scientific question: At reconstructed initialization and the
  maximum-validation checkpoint of the coherent signed/scaled-bias `wide`
  legacy-Adam Conv1/Conv2/Conv3 leaders, which EqProp nudging strength best
  matches BPTT layer-by-layer when bias gradients are excluded?
- Cases and inclusion: Six checkpoint cases, comprising three deterministic
  reconstructed initializations and three best checkpoints at epochs
  `9/29/22` with ordinary-MNIST validation accuracies
  `96.86/98.42/99.04%`. Every declared case, beta, operating phase, weight
  layer, and fixed cohort batch was retained. The three source bundles and
  their configs, results, manifests, PT checkpoints, and NPZ tensors were
  hash-verified; PT and NPZ tensors match exactly. Reconstructed initialization
  includes the active signed/scaled biases and matches the previously recorded
  complete-parameter tensor digests.
- Setup: One deterministic 256-example subset (16 batches of 16, cohort SHA-256
  `034c4791...fd37229`) from the stored 5,000-example ordinary-MNIST validation
  order; no official-test read. Accepted free/gradient iterations remained
  Conv1 `4/4`, Conv2 `6/6`, and Conv3 `8/8`. Every BPTT, zero, negative, and
  positive branch started from an identical cloned post-T state. Centered
  cost-nudged EqProp was primary over
  `beta={0.003,0.01,0.03,0.1,0.25,0.5,1.0}`; positive and negative one-sided
  estimates used the same phase endpoints. `Bias_*` tensors remained loaded
  and active in all dynamics, while only `ConvWeight_*` and `DenseWeight_*`
  entered gradient summaries. No optimizer was constructed or stepped.
- Headline direction measurements: Beta selection maximized the minimum
  centered cohort-gradient cosine over weight layers for each checkpoint.

  | Architecture / checkpoint | Selected beta | Layerwise centered cosine |
  |---|---:|---|
  | Conv1 reconstructed init | `0.01` | `C0 0.999948`, `D 1.000000` |
  | Conv1 best | `0.1` | `C0 0.988112`, `D 0.998487` |
  | Conv2 reconstructed init | `0.003` | `C0 0.981387`, `C1 0.961676`, `D 0.324420` |
  | Conv2 best | `0.003` | `C0 0.020944`, `C1 0.993538`, `D 0.973267` |
  | Conv3 reconstructed init | `0.003` | `C0 0.032305`, `C1 -0.101655`, `C2 -0.998761`, `D 0.973938` |
  | Conv3 best | none | `C0` EqProp gradient is exactly zero for every beta despite nonzero BPTT |

- Gradient scale and depth: Conv1 initialization also matches scale
  (`EqProp/BPTT` norms `1.035/1.036` for `C0/D`); at its best checkpoint the
  high-direction gradients are about `1.43/1.44x` BPTT. Conv2 does not admit a
  useful all-layer match. Its selected initialization norm ratios are
  `21.52/38.84/118.80` for `C0/C1/D`, and its selected best-checkpoint ratios
  are `89.95/1.98/2.04`. At Conv3 initialization the selected `C2` gradient is
  nearly exactly reversed (`-0.998761`); at the best checkpoint, centered
  EqProp has an exactly dead `C0`, near-orthogonal `C1`, negative `C2`, and
  well-aligned `D`. No beta corrects that layerwise split. Positive and
  negative one-sided variants also fail: their best worst-layer cosines at
  Conv2 best are only `0.0439/0.0158`, and neither has a viable all-layer
  Conv3-best candidate.
- Phase-length diagnostic: On four fixed minibatches, all 18 BPTT
  checkpoint/layer comparisons pass the accepted-K versus K64 gates
  (`cosine>=0.9`, relative norm delta `<=0.1`, zero-fraction delta `<=0.02`).
  Centered EqProp passes only `16/126` rows. At the selected beta it passes
  `2/2` Conv1-init layers, `1/2` Conv1-best layers, `3/3` Conv2-best layers,
  and `0/3` plus `0/4` Conv2/Conv3-init layers. Sixteen K64 negative-phase
  executions are non-finite across Conv1 init beta `0.1`, Conv1 best beta
  `0.25/0.5`, and Conv2 best beta `0.01`; these remain explicit structured
  outcomes. K64 therefore demonstrates severe EqProp phase-length sensitivity
  but is not a uniformly viable equilibrium reference.
- Interpretation: The accepted BPTT operating points transfer to centered
  EqProp for Conv1 direction, but not with depth. The observed pattern—aligned
  deep/readout gradients alongside orthogonal, reversed, badly scaled, or dead
  early-convolution gradients—is consistent with inadequate backward signal
  transport under the present reciprocal dynamics. This is a checkpoint-level
  necessary-condition failure, so deep EqProp training should not start by
  merely swapping estimators and retaining BPTT `T/K`. The next diagnostic
  should separate positive-phase iteration count from beta and require finite,
  non-dead, direction- and scale-viable gradients in every weight layer before
  training. Extra centered K alone is not the answer because the negative phase
  becomes unstable.
- Guards and validation: The production reporting bundle validates; all source
  bytes and all six in-memory parameter states are unchanged; all 1,800 phase
  records start from the declared common-state hash; all explicit beta-zero
  endpoints pass the `1e-5` relative-L2 equivalence gate; all selected gradient
  arrays exclude biases; and official-test/optimizer-step guards are false.
  Result SHA-256 is `cc17aebe...b37e7e5`. Smoke attempts `01`--`04` are retained
  as operational failures while the corrected all-case/all-beta
  `smoke-attempt-05` validates and is excluded from production aggregation.
- Limitations: One seed, one 256-example ordinary-MNIST validation cohort, and
  only the best coherent legacy-Adam family. Initialization is deterministic
  reconstruction rather than a saved epoch-0 file. Cosine does not measure
  scale, hence the separately reported norms. K64 uses the accepted-T common
  state and is a phase-length sensitivity test, not the independent protocol
  `T=K=64` operating-point sentinel. This study measures gradients; it does not
  establish EqProp training accuracy or compare amplification schemes under
  EqProp.
- Raw results and analysis: [study root](../results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1/),
  [reviewed report](../results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1/review.md),
  [canonical runner report](../results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1/analysis/report.md),
  [layer/beta table](../results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1/analysis/parameter_summary.csv),
  [phase-length table](../results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1/analysis/phase_sentinel_summary.csv),
  [phase diagnostics](../results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1/analysis/phase_diagnostics.csv),
  [read-only guards](../results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1/analysis/read_only_guards.json),
  and [cosine plot](../results/perfectdiode-conv123-legacy-adam-eqprop-bptt-beta-cosine-seed0-20260810-v1/analysis/cosine_by_beta_layer.png).
- Provenance: Exact compatible runtime worktree
  `/home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/signed-scaled-bias`
  was clean at `154d80e845cbec95ef9ba08981efd3079d765ddb`; frozen production
  source commit `08dd52e44818adf43b011542748e140dc0e01aeb`, source archive
  `e80f0510...ecb45ce`; analyzer SHA-256
  `6f93afd2...9d08f4`, config SHA-256 `f28e684d...dabaa0`; target `main`
  RTX 3090.

## `perfectdiode-conv123-zero-bias-lr-counterfactual-ordinary-mnist-seed0-20260807-v1` + `perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-long50-ordinary-mnist-seed0-20260808-v1` — baseline LR/time counterfactuals

- Analyzed: 2026-08-09
- Evidence class: `ordinary_mnist_lr_counterfactual` and
  `ordinary_mnist_lr_duration_counterfactual`
- Outcome: `mixed` overall and negative for the hypothesis that a simple
  learning-rate rescaling or extra time explains the Conv2/Conv3 zero-bias
  baseline gap.
- Scientific question: Can baseline recover the zero-bias ours/legacy
  trajectory by tripling its convolutional rates, matching the amplified
  schemes' initial relative-proposal vectors, or dividing its convolutional
  rates by three and training for 50 epochs?
- Runs and exclusions: All 15 requested seed-0 production arms completed: four
  30-epoch baseline-SGD LR vectors for each of Conv1/Conv2/Conv3 plus one
  50-epoch Conv-LR/3 baseline per architecture. The six fresh ours/legacy
  controls at array indices `4,5,10,11,16,17` were intentionally not launched;
  the reviewed prior controls are reused only with their duration limits made
  explicit.
- Setup and guards: Ordinary-MNIST 55,000/5,000 train/validation split,
  accepted `T/K`, exact-zero bias rates, `[0,100]` conductance projection, five
  traced transitions per epoch, every epoch checkpointed, and no official-test
  access. The authoritative local copies contain 15/15 successful bundles and
  510/510 epoch records. Canonical and semantic validation re-hashed every
  indexed artifact and checkpoint and found every saved bias tensor exactly
  zero. The repeated current-LR metric values exactly reproduce all matched
  epochs of the prior baseline controls.
- Thirty-epoch measurements: Best validation accuracy for current / Conv-LR x3
  / ours-proposal / legacy-proposal is
  `96.60/96.56/96.60/96.66%` for Conv1,
  `97.22/96.82/97.08/97.00%` for Conv2, and
  `97.04/96.56/96.92/96.86%` for Conv3. Current is best at Conv2/Conv3;
  tripling Conv rates changes those two values by `-0.40/-0.48` points.
- Duration measurements: Conv-LR/3 best accuracy through epoch 30 and epoch 50
  is `96.62/96.64%`, `97.16/97.30%`, and `96.44/96.78%` for Conv1/2/3.
  Epochs 31--50 therefore add only `+0.02/+0.14/+0.34` points. Relative to the
  current 30-epoch baseline, the 50-epoch values are `+0.04/+0.08/-0.26`
  points.
- Scheme context: Even with 50 epochs, the best Conv2 baseline remains
  `0.68/0.66` points below the prior 30-epoch ours/legacy SGD controls. Conv3's
  best tested baseline remains `1.16/1.72` points below those controls, and the
  Conv-LR/3 run itself is `1.42/1.98` points below. Conv1's longer baseline
  exceeds the prior 10-epoch amplified controls, but that comparison is not
  duration-matched.
- Interpretation: Conv1 is insensitive across the tested vectors (a complete
  0.10-point span), so its small earlier gap can disappear with extra epochs.
  At Conv2/Conv3, larger and proposal-matched rates do not transfer the
  amplified schemes' behavior into baseline coordinates. Lower rates plus
  extra time give a modest Conv2 gain and fail at Conv3. The depth-dependent
  separation is therefore unlikely to be ordinary baseline under-training at
  these rates. The next useful step is read-only analysis of the saved traces
  and checkpoints, not another longer Conv3 /3 allocation.
- Limitations: One seed and one fixed validation cohort; no uncertainty
  estimate; ordinary-MNIST diagnostic only. The Conv-LR/3 arm combines a rate
  change with a longer total budget, while its within-arm epoch-30/50 contrast
  isolates the added time. Reused Conv2/Conv3 controls are protocol- and
  duration-matched; reused Conv1 controls have only 10 epochs.
- Raw results and analysis: [30-epoch study](../results/perfectdiode-conv123-zero-bias-lr-counterfactual-ordinary-mnist-seed0-20260807-v1/),
  [50-epoch study](../results/perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-long50-ordinary-mnist-seed0-20260808-v1/),
  [reviewed report](../results/perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-long50-ordinary-mnist-seed0-20260808-v1/analysis/report.md),
  [run table](../results/perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-long50-ordinary-mnist-seed0-20260808-v1/analysis/summary.csv),
  [curve plot](../results/perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-long50-ordinary-mnist-seed0-20260808-v1/analysis/plots/validation_accuracy_by_epoch.png),
  and [validation receipt](../results/perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-long50-ordinary-mnist-seed0-20260808-v1/analysis/validation_receipt.json).
- Provenance: Jean Zay `fmu@v100` jobs `747602` and `748806`; staged commits
  `b90099fab1901bd3ceaf25385dc6ef064c600caa` and
  `338b5a57b13c227d3d8de6d9259b4b0697c43952`.

## `perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1` — all-depth zero-bias training diagnostic

- Analyzed: 2026-08-06
- Evidence class: `ordinary_mnist_bias_ablation`
- Outcome: `positive`
- Scientific question: Can Conv1/Conv2/Conv3 baseline, ours, and legacy train
  with every hidden bias initialized at zero and frozen there when all accepted
  weight learning rates are retained unchanged?
- Runs and seeds: All 18 seed-0 production arms completed: three
  architectures x three schemes x SGD/Adam. Conv1 ran 10 epochs and
  Conv2/Conv3 ran 30. Eighteen one-batch local smokes were excluded. All 36
  canonical bundles validate.
- Setup and guards: Deterministic ordinary-MNIST 55,000/5,000
  train/validation split, no affine corruption, `[0,100]` conductance
  projection, accepted `T/K` values, and no official-test access. Independent
  checks compared all 18 configs with their sources, confirmed every non-bias
  learning rate unchanged, and found every bias learning rate and all bias
  tensors in the 36 best/final production NPZ checkpoints exactly zero. All
  checked numeric checkpoint arrays are finite.
- Headline measurements: Every arm learns strongly; best validation accuracy
  spans `95.92%`--`99.00%`. Taking the better optimizer per scheme gives
  baseline/ours/legacy values of `96.28%/96.50%/96.44%` for Conv1,
  `97.36%/98.10%/98.46%` for Conv2, and `97.80%/98.66%/99.00%` for Conv3.
  Under matched SGD, legacy minus ours is `-0.58/-0.02/+0.56` points from
  Conv1 to Conv3; under Adam it is `0.00/+0.36/+0.34` points.
- Curve and mechanism follow-up: Ours exceeds baseline validation accuracy at
  all `140/140` matched epochs. Legacy exceeds ours throughout Conv3 and Conv2
  Adam, but not Conv1 or Conv2 SGD. An exact zero-bias coordinate transform
  gives effective edge weights `W'_l=g^l W_l`, with Conv3 multipliers
  `[1,1,1,1]`, `[1,4,16,64]`, and `[1,16,256,4096]`. Read-only replay on one
  deterministic 128-example cohort retained all checkpoint and parameter
  bytes. Reconstructed initialization has nearly identical scale-invariant
  class separation across schemes; at best Conv3 checkpoints the deepest
  between/within-class variance orders baseline below ours below legacy. The
  initial Conv3-SGD layerwise relative-proposal spread is
  `41.6x/5.3x/4.6x` for baseline/ours/legacy. A separate guarded endpoint
  comparison finds Conv3-SGD baseline/ours/legacy final total raw conductance
  `13277.6/15068.0/4125.9`, endpoint absolute displacement
  `15318.2/16262.2/4663.3`, and exact-zero occupancy
  `7.91%/2.70%/0.92%`. These are raw-coordinate endpoint facts, not training
  path length, write energy, or circuit power.
- Interpretation: Trainable biases are not necessary for strong ordinary-
  MNIST learning at the accepted weight rates. The amplified schemes separate
  more clearly from baseline with depth, and legacy retains a small Conv2/3
  Adam lead plus a Conv3 SGD lead in this diagnostic. This is evidence about
  optimization without biases, not evidence for the proposed signed/scaled
  contract or paper accuracy. The zero-bias schemes have the same
  classification function family after the coordinate change; the depth trend
  is consistent with effective-conductance, fixed-target output-scale, and
  learning-rate conditioning rather than extra feed-forward capacity.
- Limitations: One model/loader seed and one fixed 5,000-example validation
  cohort; no uncertainty estimate or official-test read; architecture budgets
  differ by protocol; ordinary-MNIST validation accuracy is never
  paper-facing. Any checkpoint reuse now requires the separate fail-closed
  paper audit and once-only official-test evaluation. The
  mechanism replay uses 128 examples and current-source seed reconstruction
  because no production initialization checkpoint/hash was saved; its
  associations are not a causal matched-coordinate ablation.
- Raw results and analysis: [authoritative local study](../results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/),
  [reviewed report](../results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/analysis/report.md),
  [mechanism report](../results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/analysis/mechanism/report.md),
  [run table](../results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/analysis/summary.csv),
  and [machine summary](../results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/analysis/summary.json).
- Provenance: Jean Zay `fmu@v100` array `604461_[0-17]`; frozen dirty-worktree
  snapshot based at commit `ac941dff`; source archive SHA-256
  `382df19e2baaea3b5d544a9e38fc5a348c7e1f8d6868254fcb2176d8c7ad213b`;
  ordered config-set SHA-256
  `0d29871f8393c6052244c8478a1317a6780800a434bf225904aa198a1b017483`.

## `perfectdiode-conv23-legacy-zero-bias-medium-affine-seed0-20260731-v1` — legacy training necessity of biases

- Analyzed: 2026-08-06
- Evidence class: `diagnostic_medium_affine_bias_ablation`
- Outcome: `positive`
- Scientific question: Does legacy's large post-training sensitivity to its
  learned deepest biases imply that trainable biases are necessary, or can its
  weights compensate when Conv2/Conv3 are trained from initialization with all
  biases fixed at zero?
- Runs and seeds: Two seed-0 legacy-SGD production arms, Conv2 and Conv3, each
  for 30 epochs. Four smoke/canary bundles were excluded. All six retained
  bundles validate. The original Conv3 array task `490130_1` failed before
  training because the config index was applied twice and produced no
  scientific output; corrected live canary `490690_1` passed and unchanged
  production replacement `490787_1` completed.
- Setup and guards: Deterministic medium-affine MNIST, original `[0,100]`
  weight contract and exact weight learning rates, every hidden-bias rate and
  checkpoint tensor exactly zero, five gradient-trace samples per epoch, and
  epoch checkpoints. The official test was deliberately skipped.
- Headline measurements: Conv2 zero-bias best validation accuracy is `91.58%`
  versus `91.42%` in the original learned-bias run, with final loss
  `0.164916` versus `0.165221`. Conv3 is `96.82%` versus `96.70%`, with final
  loss `0.081811` versus `0.082174`. The accuracy differences are only eight
  and six predictions out of 5,000, respectively, and all four runs select
  epoch 30.
- Interpretation: There is no observed zero-bias performance deficit. Legacy
  weights compensate when biases are absent from the start, so the severe
  post-hoc checkpoint intervention establishes checkpoint dependence but not
  training necessity. The historical unscaled-bias contract remains a
  scientific confound, but it is not by itself an adequate explanation for
  legacy's Conv2/Conv3 SGD advantage.
- Limitations: Legacy + SGD only, one seed, validation-only comparison, and no
  baseline/ours or Adam medium-affine zero-bias controls. This does not test
  signed or amplification-scaled biases and does not replace official-test
  paper rows.
- Raw results and analysis: [authoritative local study](../results/perfectdiode-conv23-legacy-zero-bias-medium-affine-seed0-20260731-v1/),
  [reviewed report](../results/perfectdiode-conv23-legacy-zero-bias-medium-affine-seed0-20260731-v1/analysis/report.md),
  [matched table](../results/perfectdiode-conv23-legacy-zero-bias-medium-affine-seed0-20260731-v1/analysis/summary.csv),
  and [machine summary](../results/perfectdiode-conv23-legacy-zero-bias-medium-affine-seed0-20260731-v1/analysis/summary.json).
- Provenance: Jean Zay `umg@v100`; source commit `ade6248a`; archive SHA-256
  `f7f9d80b439ba8f0f4d1a3ed1e1dde4b63135f4b5e9fc5e028e9c76ee4e1df5a`;
  initial array `490130_[0-1]` and Conv3 production replacement `490787_1`.

## `perfectdiode-conv12-bounded-uniform-signed-bias-rho-seed0-v1` — signed-only bounded bias screen

- Analyzed: 2026-08-06
- Evidence class: `ordinary_mnist_selection_mechanism_diagnostic`
- Outcome: `mixed`
- Scientific question: With bounded-uniform conductances, do signed,
  still-unscaled biases improve the three-epoch Conv1/Conv2 baseline/ours
  operating points, and are the searched bias-rate ranges sufficient?
- Runs and seeds: Seed 0; 53 production cells across eight Conv1/Conv2 x
  baseline/ours x SGD/Adam surfaces plus two excluded smokes. All 55 bundles
  validate, all eight surfaces publish one selected checkpoint, and the
  official test was not read.
- Setup: Ordinary MNIST, fixed accepted `T=K=4/6`, conductance bounds
  `[1e-5,1e-4]`, bounded-uniform initialization, signed unclamped biases, and
  three epochs per cell. Conductance rho pairs came from a finite previously
  useful shortlist while bias rho scales were screened independently. This
  implementation does not apply depth-dependent amplification scaling to
  biases.
- Headline measurements: Every selected checkpoint uses both bias signs;
  `36.99%`--`48.96%` of biases are negative. Relative to the recorded
  nonnegative selections, all SGD changes lie in `[-0.06,+0.08]` points,
  while three Adam surfaces gain `+1.06`--`+1.84` points and Conv1 ours/Adam
  is unchanged. Conv2 ours/Adam reaches `89.64%`, the largest gain
  (`+1.84` points). Five of eight surfaces select the tested upper bias-scale
  edge `9`, so those bias ranges remain open. The worst projection efficiency
  is `0.228` for Conv2 baseline/Adam; Conv2 ours/SGD remains high at `0.959`
  and changes only `+0.08` points.
- Interpretation: Signed biases are actively used and targeted Adam rows may
  benefit from a separately selected bias rate. They do not repair the Conv2
  ours/SGD shortfall, and the open upper edges prevent terminal handoff. The
  comparison changes bias sign and rate together; two selected rows also use
  alternate shortlisted conductance-rho pairs, so it is not a pure causal
  sign ablation.
- Limitations: One seed, three epochs, ordinary-MNIST selection evidence,
  bounded-uniform only, no legacy or Conv3, and no amplification-scaled bias
  energy. It cannot select the global bounded initializer or validate the
  proposed corrected bias contract.
- Raw results and analysis: [authoritative local study](../results/perfectdiode-conv12-bounded-uniform-signed-bias-rho-seed0-v1/),
  [reviewed report](../results/perfectdiode-conv12-bounded-uniform-signed-bias-rho-seed0-v1/analysis/report.md),
  [surface table](../results/perfectdiode-conv12-bounded-uniform-signed-bias-rho-seed0-v1/analysis/surface_summary.csv),
  and [machine summary](../results/perfectdiode-conv12-bounded-uniform-signed-bias-rho-seed0-v1/analysis/summary.json).
- Provenance: clean implementation commit `3564bd55`; working-tree SHA-256
  `a6f642dfbab3ab923b5b608a0c953eafad55e345889756e9c2b5312505515618`.

## `perfectdiode-paper-gradient-trace-limited-20260731-v1` — optimizer-state and depth-wise gradient/update diagnosis

- Analyzed: 2026-07-31
- Evidence class: `diagnostic_medium_affine_gradient_trace`
- Outcome: `mixed`
- Scientific question: Are the apparent seed-0 SGD advantages caused by an
  optimizer learning-rate mismatch, and does legacy's Conv2/Conv3 advantage
  coincide with abnormal gradients, projection failure, or better layerwise
  step conditioning?
- Runs and seeds: Twelve seed-0 production diagnostics: all six Conv1
  scheme/optimizer rows and baseline/ours/legacy SGD for Conv2 and Conv3. Every
  row executed 10 epochs, recorded 50 optimizer transitions, and saved model
  plus optimizer checkpoints at epochs 0--10. The passing one-batch Jean Zay
  canary was excluded. All 13 bundles validate; all diagnostic metric prefixes
  are bit-identical to the source paper trajectories.
- Setup and provenance: Deterministic medium-affine MNIST with the exact paper
  configurations, frozen commit `ade6248a5e9d80ec8157b8a5273f095e6b30715b`,
  source archive SHA-256
  `f7f9d80b439ba8f0f4d1a3ed1e1dde4b63135f4b5e9fc5e028e9c76ee4e1df5a`,
  and Jean Zay `umg@v100`. The trace recorded raw gradients, real optimizer
  proposals, post-projection applied updates, fresh-state rho shadows, bounds,
  and Adam moments. All 132 epoch checkpoints were replayed read-only on one
  frozen 256-example validation cohort; source bytes and parameter states were
  unchanged.
- Rho and weight contract: Every included row uses the wide/unbounded rho
  handoff family; none belongs to the separate bounded-weight initializer/rho-
  selection study whose conductance interval is `[1e-5,1e-4]`. “Unbounded”
  describes selector provenance, not complete rho bracketing: most Conv1/2
  surfaces remained at an outer search edge. Downstream conductances are still
  projected to `[0,100]`. Conv3 retains the caveat that its rates came from a
  `weight_max=null` source selection before transfer to `[0,100]`.
- Headline measurements: Across the original nine matched scheme/optimizer
  paper comparisons, Adam has lower official-test loss in all nine and higher
  accuracy in seven. SGD's only accuracy wins are Conv1 baseline and ours, by
  `0.74` and `0.84` points; their best-validation differences are only `0.12`
  and `0.10` points at one seed. In the Conv1 traces, real stateful-Adam
  proposals have median
  `0.211` times the fresh rho-shadow RMS and median direction cosine `0.312`,
  with `38.2%` momentum-sign disagreement. Conv3 median applied-update/current-
  weight spreads are `14.95x/8.55x/3.08x` for baseline/ours/legacy. On the
  fixed cohort, epoch-10/epoch-0 raw-gradient RMS for `C1/C2` is
  `38.7x/81.0x`, `4.33x/8.64x`, and `2.10x/3.33x`, respectively. Gradients
  remain finite and essentially dense. Median projection efficiency stays
  high, no upper-bound occupancy appears, and lower-bound pressure is least in
  legacy.
- Interpretation: The results do not support a general claim that SGD
  outperforms Adam. They expose a structural Adam-calibration mismatch: rho was
  evaluated with a fresh Adam shadow, whereas training uses accumulated
  moments, producing both a smaller and differently directed real step. A
  simple global LR multiplication is therefore not justified. Legacy's deeper
  advantage is associated with more conservative, better-balanced effective
  layerwise updates and lower boundary pressure. Given the strongly
  scheme-specific LR and bias vectors, this is more consistent with
  scheme-by-LR/bias conditioning than intrinsic superiority of legacy.
  Adam transition tracing is Conv1-only, so the mismatch magnitude is not
  directly measured for Conv2/Conv3.
- End-of-training bias follow-up: A guarded fixed-cohort replay of all six
  Conv2/Conv3 SGD best and final checkpoints shows that the legacy advantage is
  strongly dependent on its deepest learned bias, despite that tensor being
  smaller in raw RMS. Removing only the deepest bias changes the deepest state
  by `25.41%` (Conv2) and `67.01%` (Conv3) of learned-state RMS and changes
  `17.19%` and `16.41%` of predictions. Removing every bias reduces diagnostic
  fixed-cohort accuracy by `9.38` and `17.58` points, whereas baseline and ours
  change by at most one of 256 examples. The trained local-curvature estimate
  predicts the same scheme/depth ordering: legacy's deepest coordinate shift
  is `41.0%` (Conv2) and `158.7%` (Conv3) of learned-state RMS. The mechanism is
  structural: the linear bias drive is unscaled while legacy's deep resistive
  curvature is strongly attenuated. Retaining only spatial channel means keeps
  most legacy accuracy, with losses of `0.78` and `1.56` points. This establishes
  checkpoint dependence, not training necessity. The subsequent matched
  zero-bias legacy retrains reached essentially unchanged Conv2/Conv3
  validation accuracy and are reviewed separately above.
- Convergence: All source paper runs and diagnostics completed their fixed
  budgets with finite measurements, but completion is not optimizer
  convergence. Conv1 peaks around epochs 8--9 and fluctuates thereafter; nine
  of the 12 original Conv2/Conv3 surfaces select their best checkpoint at
  epochs 28--30 of 30. The 10-epoch deep diagnostics are prefixes, not
  convergence tests. Accepted `T/K` gates support the operating points, but
  per-step circuit residuals were not recorded, so direct equilibrium
  convergence throughout training is not established here.
- Limitations and deviations: One seed; five sampled optimizer steps per
  epoch; Conv2/Conv3 optimizer-step traces are SGD-only; fixed-cohort replay is
  associative rather than causal. Official-test values come from the original
  paper bundles—the diagnostics did not read the official test. Conv2 lacks
  long LR confirmation, and Conv3 retains the source `weight_max=null` to
  downstream `[0,100]` mismatch. A decisive comparison requires stateful Adam
  calibration or a matched effective-update/per-layer LR sweep including
  biases and multiple seeds.
- Raw results: [authoritative local study](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/);
  remote source
  `/lustre/fsn1/projects/rech/umg/ucy17uy/server_code/results/perfectdiode-paper-gradient-trace-limited-20260731-v1`.
- Analysis: [reviewed report](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/report.md),
  [study summary](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/summary.json),
  [source inventory](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/summary.csv),
  [online trace summary](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/online_trace/gradient_trace_summary.json),
  [anomaly inventory](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/online_trace/gradient_trace_anomalies.json),
  [fixed-cohort gradient replay](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/fixed_validation_replay/analysis_summary.json),
  and [end-of-training bias-effect report](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/bias_implementation_audit/paper_end_bias_effect/report.md).

## `perfectdiode-paper-medium-affine-wide-seed0-20260729-v1-analysis` — Conv1/Conv2 final weights and projection pressure

- Analyzed: 2026-07-31
- Evidence class: `paper_medium_affine_mechanism_diagnostic`
- Outcome: `mixed`
- Scientific question: Do the final `[0,100]` conductance distributions of the
  seed-0 Conv1/Conv2 baseline, ours, and legacy paper models show boundary
  pinning or poor gradient/update projection that explains their accuracy
  differences?
- Runs and seeds: All 12 production rows for Conv1/Conv2 x
  baseline/ours/legacy x SGD/Adam, seed 0. The 12 locally staged reporting
  bundles validated, and all 48 checkpoint artifacts matched their recorded
  size and SHA-256. Conv1/Conv2 smoke bundles were excluded.
- Setup: Deterministic medium-affine MNIST paper configs with the original
  fixed handoffs, `T=K=4` for Conv1 and `T=K=6` for Conv2, and conductance
  projection to `[0,100]`. The exact production source commit `1fd3e034` was
  used to reconstruct the seed-0 initializer and replay every final checkpoint
  without optimizer steps on the same first 16 training minibatches (256
  examples). Source checkpoint hashes and replayed tensors were unchanged.
- Headline measurements: The reconstructed initializer has `49.92%` (Conv1)
  and `49.86%` (Conv2) exact lower-bound occupancy. Every final model moved
  away from that initial atom: aggregate final occupancy is `0.17%`--`9.35%`
  for Conv1 and `0.79%`--`25.36%` for Conv2. No final weight equals the upper
  bound; the maximum is `65.30`. The worst layer median retains `96.7%` of raw
  tangent-gradient L2 norm. Exact momentum-free SGD proposal efficiency is at
  least `97.1%`. The strongest proxy clipping is Conv2 baseline Adam, where a
  fresh-moment Adam proposal retains `87.1%`--`88.5%` L2 and clips
  `24.7%`--`27.0%` of components.
- Interpretation: The general boundary-pinning/projection-collapse hypothesis
  is not supported at the final checkpoints. The clearer anomaly is the
  split, inflated Conv2 baseline distribution: its three layer RMS values are
  `37.4x/11.5x/22.9x` initialization for SGD and `53.1x/7.0x/19.2x` for Adam.
  Conv2 baseline-Adam's top 10% of first-layer output-channel filters carry
  `60.7%` of layer L2 energy. This scale and filter imbalance is more
  consistent with the poor baseline result than an active upper ceiling,
  while remaining associative rather than causal.
- Limitations: One seed; final-checkpoint replay rather than a recorded
  training trajectory; local float32 CPU replay rather than V100. Historical
  Adam optimizer moments were not saved, so fresh-Adam proposal efficiency is
  a labeled proxy. Official test accuracy belongs to the best-validation
  checkpoint; distributions and replay conclusions use final checkpoints.
  The separate `[1e-5,1e-4]` ordinary-MNIST bounded-rho studies are not mixed
  into this comparison.
- Raw results: Jean Zay
  `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/perfectdiode-paper-medium-affine-wide-seed0-20260729-v1`.
- Analysis:
  [reviewed report](../results/perfectdiode-paper-medium-affine-wide-seed0-20260729-v1-analysis/final_weight_projection/report.md),
  [surface summary](../results/perfectdiode-paper-medium-affine-wide-seed0-20260729-v1-analysis/final_weight_projection/summary.csv),
  [layer distributions](../results/perfectdiode-paper-medium-affine-wide-seed0-20260729-v1-analysis/final_weight_projection/weight_distribution_by_layer.csv),
  [projection table](../results/perfectdiode-paper-medium-affine-wide-seed0-20260729-v1-analysis/final_weight_projection/gradient_projection_summary.csv),
  [weight histograms](../results/perfectdiode-paper-medium-affine-wide-seed0-20260729-v1-analysis/final_weight_projection/final_weight_histograms.png),
  and
  [projection diagnostics](../results/perfectdiode-paper-medium-affine-wide-seed0-20260729-v1-analysis/final_weight_projection/final_gradient_projection.png).

## `perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1` — bounded Conv1/Conv2 checkpoint mechanisms

- Analyzed: 2026-07-31
- Evidence class: `ordinary_mnist_selection_mechanism_diagnostic`
- Outcome: `mixed`
- Scientific question: Do the selected `[1e-5,1e-4]` bounded-rho Conv1/Conv2
  checkpoints fail because conductances accumulate at the physical limits or
  because projection removes most proposed updates, and do any failures have a
  different gradient mechanism?
- Runs and seeds: Seed 0; all 16 canonical baseline/ours selections across
  Conv1/Conv2, two bounded initializers, and SGD/Adam, plus the five completed
  historical legacy selections (four Conv1 and bounded-uniform/SGD Conv2).
  Initialization, best-validation, and final checkpoints gave 63 guarded
  checkpoint roles. Historical legacy remains a separate, incomplete evidence
  panel.
- Setup: Ordinary MNIST, `T=K=4` for Conv1 and `T=K=6` for Conv2, physical
  conductance projection to `[1e-5,1e-4]`, and no official-test access. Raw
  gradients were replayed without optimizer steps on the same first 16
  deterministic training minibatches (256 examples). All source result and
  selection hashes matched; all 63 source files and replayed tensor states were
  unchanged. SGD proposal replay is exact for the saved zero-momentum,
  zero-weight-decay contract. Adam moments were not saved, so only recorded
  training projection diagnostics are authoritative for Adam; fresh-state
  proposal replay is labeled as a shadow.
- Headline measurements: Whole-model exact-bound occupancy never exceeds
  `41.96%`, so no final model has a majority of all conductances exactly pinned.
  Pinning is nevertheless severe within specific Conv2 layers: the maximum is
  `94.44%`, and baseline/Adam has `92.26%`--`93.40%` exact occupancy in
  `ConvWeight_1`. Conv2 baseline/Adam records only `0.091`--`0.096` aggregate
  projection efficiency; its first two convolution tensors retain as little as
  `0.006`--`0.079` of proposed RMS during training and `0.382`--`0.416` of
  feasible raw-gradient L2 at the final checkpoint. Matched ours/Adam lowers
  aggregate occupancy by `4.48`--`16.27` points, raises recorded projection
  efficiency by `0.320`--`0.403`, and improves accuracy by `2.50`--`2.84`
  points.
- Failure diagnosis: Kaiming/ours/SGD Conv2 is the clear non-clipping failure.
  It remains at `10.08%`; all conductance tensors have zero median proposed
  update RMS, final `ConvWeight_0` has zero gradient on all 16 replay batches,
  and the deeper tensors have zero gradient on 15 of 16. Their means moved
  upward to normalized conductance positions `0.561` and `0.569`. The raw
  conductance rates are approximately `304x/456x/127x` the corresponding
  successful bounded-uniform/ours/SGD rates. This is consistent with an early
  oversized operating-point shift followed by dead-gradient collapse, not
  projection clipping. The high recorded projection ratio is conditional on
  nonzero proposals and is therefore not evidence of active learning here.
- Broader interpretation: Conv1 ours improves all four matched rows by
  `2.38`--`3.34` points while reducing bound occupancy and modestly improving
  projection. Historical legacy Conv1 combines `94.46%`--`95.60%` accuracy,
  `0.64%`--`6.12%` aggregate exact occupancy, and `0.931`--`0.983` projection
  efficiency, but its predecessor training policy prevents a causal matched
  claim. Historical bounded-uniform/legacy/SGD Conv2 and active
  bounded-uniform/ours/SGD have nearly equal training projection efficiency
  (`0.957` versus `0.958`) despite `93.80%` versus `85.80%` accuracy, so clipping
  cannot explain the full historical gap. Every non-collapsed active Conv2 row
  selects epoch three as best, consistent with the short selection budget also
  limiting accuracy.
- Rho-range status: Four active choices remain boundary-selected: both Conv1
  Kaiming/SGD rows at lower/lower, Conv2 Kaiming/baseline/SGD at dense lower,
  and Conv2 bounded-uniform/ours/SGD at dense lower. The other 12 active choices
  are bracketed (`unbounded` in the search terminology). Bound occupancy and
  projection efficiency remained report-only and excluded no checkpoint.
- Limitations and deviations: One seed and three training epochs; ordinary
  MNIST is diagnostic rather than paper-facing evidence; only
  initialization/best/final checkpoints exist; historical coverage and policy
  are unmatched; checkpoint replay establishes association rather than
  causality. The first local smoke attempt failed before replay because the
  optional OpenMP workaround hid CUDA on this host; the second completed replay
  but exposed an SGD-only report-builder bug. Both are retained, and the fixed
  third smoke passed all nine guards before the full run.
- Analysis:
  [reviewed report](../results/perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1/analysis/report.md),
  [surface summary](../results/perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1/analysis/surface_summary.csv),
  [layer distributions](../results/perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1/analysis/weight_distribution_by_layer.csv),
  [recorded training projection](../results/perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1/analysis/training_projection_summary.csv),
  [gradient replay](../results/perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1/analysis/checkpoint_gradient_summary.csv),
  [checkpoint guards](../results/perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1/analysis/checkpoint_guards.json),
  and
  [plots](../results/perfectdiode-conv12-bounded-rho-checkpoint-analysis-seed0-v1/analysis/).
- Provenance: implementation commits `7b03095b` and `6e7cc751`; the full
  replay manifest records a clean `6e7cc751` checkout and the frozen cohort
  hash `ffac016d871e7c49c2574c8dfe00b892bb38865e1f533458d2cb83c5362da6dc`.

## `perfectdiode-conv2-bounded-rho-continuation-wave3-seed0-v1` — bounded Conv2 rho continuation and weight-distribution diagnosis

- Analyzed: 2026-07-30
- Evidence class: `ordinary_mnist_selection`
- Outcome: `mixed`
- Scientific question: After the initial bounded Conv2 baseline/ours rho
  search failed to reach 90% validation accuracy, do factor-of-three rho
  extensions locate better three-epoch operating points, and do the selected
  conductance distributions explain the remaining shortfall?
- Runs and seeds: Seed 0; two bounded initializers, baseline and ours, SGD and
  Adam. Wave two added 35 terminal cells over eight surfaces. User-directed
  wave three added 30 terminal cells over the six wave-two boundary-selected
  surfaces; all 30 canonical wave-three bundles validated. Legacy, Conv1, and
  Conv3 were excluded. Bound occupancy and projection efficiency were
  report-only diagnostics.
- Setup: Ordinary MNIST 55,000/5,000 deterministic train/validation split;
  official test split not read; Conv2 perfect-diode BPTT with fixed
  `T=K=6`; input gain 100; channels `[64,128]`; asynchronous minimizer;
  batch size 16; validation batch size 64; three epochs per cell. Conductance
  weights were projected to `[1e-5,1e-4]`. Initializers were full-range
  bounded uniform and bounded Kaiming uniform. Schemes were baseline
  `(voltage_amp=1,current_amp=1)` and ours
  `(voltage_amp=4,current_amp=1)`.
- Headline measurements: No surface reached the 90% reporting threshold.
  “Unbounded” below means that the selected rho is bracketed by the tested
  search range, not that the conductance weights lacked physical bounds.

  | Initializer | Scheme | Optimizer | `rho_conv` | `rho_dense` | Validation accuracy | Rho range |
  |---|---|---|---:|---:|---:|---|
  | bounded uniform | baseline | SGD | 0.001 | 0.000370370 | 77.28% | unbounded/bracketed |
  | bounded uniform | baseline | Adam | 0.081 | 0.00333333 | 85.30% | unbounded/bracketed |
  | bounded uniform | ours | SGD | 0.000333333 | 0.000123457 | 85.80% | bounded at dense lower edge |
  | bounded uniform | ours | Adam | 0.027 | 0.01 | 87.80% | unbounded/bracketed |
  | bounded Kaiming | baseline | SGD | 0.000111111 | 0.00111111 | 63.66% | bounded at dense lower edge |
  | bounded Kaiming | baseline | Adam | 0.081 | 0.00333333 | 85.26% | unbounded/bracketed |
  | bounded Kaiming | ours | SGD | 0.003 | 0.01 | 10.08% | unbounded/bracketed |
  | bounded Kaiming | ours | Adam | 0.081 | 0.03 | 88.10% | unbounded/bracketed |

- Weight-distribution measurements: The Kaiming initializer starts
  `ConvWeight_1` and `DenseWeight_0` in very narrow normalized supports
  `[0.449,0.551]` and `[0.485,0.515]`, respectively, inside the physical
  interval. Every 85%--88% selected row moved the final deeper-layer means
  downward: `ConvWeight_1` ended at `1.83e-5`--`2.47e-5`. The two weak
  Kaiming-SGD rows did not: baseline/SGD ended at `5.09e-5`, while
  ours/SGD shifted upward to `6.05e-5`; its dense mean also shifted upward to
  `6.12e-5`. The latter row remained at chance from epoch one
  (`10.04%,10.08%,10.08%`) with loss approximately `0.5`.
- Interpretation: Exact bound occupancy is not the primary explanation.
  Strong Adam rows have as much as 80%--93% exact occupancy in an early
  convolutional layer, and useful low-bound accumulation accompanies the
  downward deeper-layer shift. The clearest pathological case is
  Kaiming/ours/SGD: initializer-specific proposal normalization produced raw
  rates of `4.85e-5`, `3.27e-3`, and `8.47e-5` for the two convolutional and
  dense weights. These are roughly 304x, 456x, and 127x the corresponding
  rates in the successful bounded-uniform/ours/SGD selection. The final
  conductances are broadly redistributed but carry no useful class signal,
  and neighboring cells either remained at chance or encountered sustained
  gradient-RMS rejection. For the other seven selections, the best epoch was
  epoch three and validation accuracy was still increasing. Their shortfall
  from 90% is therefore primarily consistent with the deliberately short
  training budget; it is not evidence that projection alone prevents
  learning. Kaiming/baseline/SGD also remains rho-bounded and improved from
  `20.12%` to `48.46%` to `63.66%`, so it is visibly under-trained as well as
  poorly positioned.
- Limitations and protocol deviations: This is a one-seed, three-epoch
  selection diagnostic and is not paper-facing accuracy evidence. Weight
  distributions show association, not causality; a read-only gradient replay
  or longer matched confirmation would be needed to distinguish persistent
  gradient imbalance from an early transient. Wave three was an explicit
  extra continuation beyond the protocol's original single expansion wave.
  Trex was occupied by an unrelated job, so two Trex-planned wave-two
  surfaces ran on main with recorded target failover; scientific configs and
  cohorts were unchanged.
- Raw results:
  [`wave two`](../results/perfectdiode-conv2-bounded-rho-continuation-wave2-seed0-v1)
  and
  [`wave three`](../results/perfectdiode-conv2-bounded-rho-continuation-wave3-seed0-v1).
- Analysis:
  [machine-readable summary](../results/perfectdiode-conv2-bounded-rho-continuation-wave3-seed0-v1/analysis/weight_distribution/summary.json),
  [layer table](../results/perfectdiode-conv2-bounded-rho-continuation-wave3-seed0-v1/analysis/weight_distribution/layer_statistics.csv),
  [histograms](../results/perfectdiode-conv2-bounded-rho-continuation-wave3-seed0-v1/analysis/weight_distribution/selected_weight_histograms.png),
  and
  [diagnostic comparison](../results/perfectdiode-conv2-bounded-rho-continuation-wave3-seed0-v1/analysis/weight_distribution/weight_diagnostics.png).
- Provenance: continuation implementation commits `ca2c44f5`, `c9bd9da6`,
  and `a8c5b00f`; wave-three canonical summary reports
  `official_test_read=false`.

## `conv3_pd_unbounded_rho_t8k8_20260729T125734Z` — Conv3 perfect-diode rho selection

- Analyzed: 2026-07-29
- Completed: 2026-07-29 19:48 CEST
- Evidence class: `ordinary_mnist_selection`
- Outcome: `positive`
- Scientific question: Which `(rho_conv, rho_dense)` pairs minimize the
  three-epoch validation loss for Conv3 perfect-diode networks with `T=K=8`
  and nonnegative weights without an upper bound, for SGD and Adam across the
  baseline, ours, and legacy amplification schemes? Are the initial rho bounds
  sufficient?
- Runs and seeds: Seed 0; six optimizer/scheme surfaces; all 91 semantically
  complete cells included (54 initial cells and 37 boundary-expansion cells).
  All six final search bounds are closed, the retained cells share one cohort
  contract, and the final audit found zero semantic issues.
  Stability probes and canaries were operational gates and were excluded from
  the scientific comparison.
- Setup: Ordinary MNIST 55,000/5,000 deterministic train/validation split;
  official test split not read; input gain 360; Conv3 channels
  `[64, 128, 256]`; kernels 3; strides `[2, 2, 1]`; padding 1; output width
  20; Kaiming-uniform initialization; asynchronous minimizer; batch size 16;
  validation batch size 64; three epochs per cell. The schemes were baseline
  `(voltage_amp=1, current_amp=1)`, ours `(4, 1)`, and legacy `(4, 0.25)`.
  Weights used `weight_min=0` and `weight_max=null`. Runs used one
  `v100-16g` GPU each on Jean Zay under `fmu@v100`.
- Headline measurements: Selection used minimum final validation loss. Every
  selected point is interior to the final explored bounds. Accuracy is shown
  only as an ordinary-MNIST selection diagnostic.

  | Scheme | Optimizer | `rho_conv` | `rho_dense` | Validation loss | Validation accuracy |
  |---|---|---:|---:|---:|---:|
  | baseline | SGD | 0.001 | 0.03 | 0.09482917 | 94.74% |
  | baseline | Adam | 0.027 | 0.27 | 0.06915623 | 96.20% |
  | ours | SGD | 0.009 | 0.03 | 0.05100432 | 97.18% |
  | ours | Adam | 0.081 | 0.09 | 0.04339848 | 97.70% |
  | legacy | SGD | 0.003 | 0.01 | 0.04871860 | 97.44% |
  | legacy | Adam | 0.027 | 0.01 | 0.03983585 | 97.96% |

- Interpretation: Adam lowered the selected validation loss relative to SGD
  by `0.02567295` for baseline, `0.00760584` for ours, and `0.00888274`
  for legacy. Both amplified schemes improved on the baseline for both
  optimizers in this selection study. Legacy Adam produced the lowest
  three-epoch validation loss. The boundary expansions turned back around all
  six selected points, so no further rho expansion is indicated for this
  protocol.
- Limitations: This is a one-seed, three-epoch operating-point selection
  study, not final-training or paper-test evidence. Ordinary-MNIST selection
  accuracy is diagnostic only. “Unbounded” means nonnegative weights without
  an upper cap, not signed unconstrained weights. The run predates
  `docs/experiment_reporting.md`, so its raw bundle uses the reviewed
  rho-search layout rather than per-cell
  `manifest.json`/`status.json`/`result.json`; the frozen source hashes,
  semantic cell records, scheduler logs, final receipt, and 1,817-file
  checksum manifest provide the retained provenance. An initial canary failed
  because `git` was absent from the compute-node module path; the corrected
  canary passed before any scientific batch was released and does not affect
  the conclusion.
- Raw results:
  [validated local copy](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/remote_full/conv3_pd_unbounded_rho_t8k8_20260729T125734Z)
  and
  `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z`
  on Jean Zay.
- Analysis:
  [`results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis`](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis)
- Selected LR handoff:
  [`perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json`](../configs/conv/perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json).
  It copies the six exact named vectors from the selected cells, including
  bias rates modified by the `q90_cap` policy; rho is not recomputed
  downstream.
- Provenance: frozen source commit
  `20b495ec8f3bf4c6179480e2686bde1984b643c9`; source archive SHA-256
  `23ea58b784d189dd2e373e1e019af3b53bd6638a7d51390e654e19f1d2e9b9ac`;
  run-input archive SHA-256
  `43b8114218c566f88ab2454d82915b245ed9b8c9089254d8e2d70e0184ae62cd`;
  1,817-file manifest SHA-256
  `07bc63932bbe3df5dbc9d0013432315392fed3feca3555b16ca7cf8403120190`,
  verified locally with exit code 0.
- Detailed evidence:
  [summary](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis/study_summary.md),
  [selected-point table](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis/best_validation_by_surface.csv),
  [rho surfaces](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis/rho_validation_surfaces.png),
  [optimizer comparison](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis/best_validation_loss_by_surface.png),
  and
  [final receipt](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis/final_receipt.json).

Existing reviewed JSON cards under `result_registry/` remain historical
evidence and need not be rewritten.

<!--
## <study-id> — <title>

- Analyzed: YYYY-MM-DD
- Evidence class:
- Outcome:
- Runs and seeds:
- Setup:
- Headline measurements:
- Interpretation:
- Limitations:
- Raw results:
- Analysis:
- Detailed evidence:
-->
