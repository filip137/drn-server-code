# Choosing EqProp beta: evidence and a defensible paper protocol

Status: updated September 21, 2026. All88 new calibration settings, the
original23 audited training settings, six additional refined Conv3 ten-epoch
trainings, and all24 thirty-epoch p90 read-noise outcomes are locally collected
and validated. The later sigma1e-3 extension has three collected terminal
outcomes: one full completion and two non-finite training failures (see the
September 21 addendum). Proposed
selection and confirmation steps are recommendations, not experiments claimed
to have been performed.

**Recommendation.** Treat beta as a training hyperparameter, conditional on
the architecture, amplification, optimizer, conductance bounds, solver budget
and noise model. Select it by empirical stability and validation performance
under a declared objective. Use gradient agreement to characterize the
approximation and explain tradeoffs. A static cosine threshold alone does not
certify successful training, and the present evidence does not establish a
universal replacement threshold on another gradient statistic.

The claim matters: if a particular result is intended to demonstrate numerical
gradient equivalence itself, it still needs a separately declared fidelity
error budget on relevant states. Good validation performance cannot establish
that claim. Larger-beta settings that learn well despite substantial gradient
error should be described as empirical finite-nudge operating points, with
their approximation error reported, rather than as exact-gradient evidence.

## What our simulations actually establish

The audit joins 23 distinct clean seed-0 architecture/scheme/beta settings to
their September18 calibration measurements. Seventeen completed ten finite
epochs and met the existing final-drop rule; six became nonfinite. Historical
30-epoch controls contribute only their first ten epochs. Duplicate reused
settings are counted once. This is a retrospectively selected development
sample, not a random sample suitable for estimating a selector's general
failure probability.

| Example | Worst matrix cosine | Worst symmetric norm mismatch | Maximum relative gradient L2 error | Training outcome |
|---|---:|---:|---:|---|
| Conv1 ours, beta 1500 | .951337 | .084590 | .308621 | Nonfinite in epoch 1 |
| Conv2 legacy, beta 30 | .954935 | .127926 | .297104 | Nonfinite in epoch 5 |
| Conv2 ours, beta 50 | .929088 | .173502 | .452617 | Ten epochs, 97.96% final validation |
| Conv3 baseline, beta 750 | .696630 | .479017 | .923333 | Ten epochs, 97.08% |
| Conv3 ours, beta 22.5 | .454946 | .664652 | 1.370462 | Ten epochs, 98.38% |

The corresponding smaller-beta Conv3 controls finish at 97.02% and 98.42%.
Thus markedly poorer worst-layer gradient agreement can coexist with almost
unchanged ten-epoch accuracy, while good cosine and norm agreement can coexist
with a numerical training failure. These counterexamples concern the sampled
reference checkpoints; they do not identify the unique cause of instability.

All 23 settings pass whole-gradient cosine>.99 and whole-gradient symmetric
norm mismatch<=.10, including all six failures. This sample is enriched for
passing whole-gradient candidates by construction; the observation refutes a
stability guarantee, not every possible use of whole-gradient diagnostics.
Per-matrix cosine>.95 admits 14 settings, including two failures, and excludes
five stable settings. Adding per-matrix norm mismatch<=.10 admits eight,
including one failure, and excludes ten stable settings. Per-matrix cosine>.99
has no failures among its eight accepted observed settings, but excludes nine
stable settings. It is conservative in this sample; eight successes from one
seed per setting do not establish a guarantee.

In the requested .90/.95 training comparison, larger-minus-smaller epoch 10
accuracy is +.12pp for Conv2 ours, .00pp for Conv3 baseline, and -.04pp for
Conv3 ours. The evidence shows no consistent accuracy penalty from larger
beta among those completed comparisons. It does not establish equivalence or
an improvement with such small, single-seed differences.

[Joined measurements](beta_protocol_training_evidence_20260919.csv) and
[rule/outcome counts](beta_protocol_rule_outcomes_20260919.csv) are generated
by `experiments/analyze_beta_protocol_evidence.py`.

The batch distribution adds an important qualification. For each replay, take
the minimum cosine across matrices, then summarize those 36 batch minima
separately at each checkpoint:

| Setting | Initializer minimum / median | Initializer batches passing .95 | BPTT-checkpoint minimum / median | Ten-epoch outcome |
|---|---:|---:|---:|---|
| Conv1 ours, beta 1500 | .97493 / .98693 | 36/36 | .95134 / .97533 | Failed |
| Conv2 legacy, beta 30 | .95493 / .98941 | 36/36 | .97286 / .99145 | Failed |
| Conv3 baseline, beta 750 | .69663 / .97357 | 24/36 | .99449 / .99896 | Stable |
| Conv3 ours, beta 22.5 | .45495 / .96580 | 21/36 | .98743 / .99857 | Stable |

All four settings pass .95 on every batch at the saved BPTT checkpoint.
For the stable high-beta Conv3 settings, the very low minima do not describe
the typical batch: their initializer medians remain above .96. Nevertheless,
12/36 and 15/36 initializer batches fail .95, so this is not just one outlier.
Conversely, high median alignment does not rescue the two unstable settings.
Reporting distributions is more informative than one extreme, but these data
do not validate a median or percentile threshold as a stability selector.
See the [checkpoint-separated distributions](beta_protocol_batch_distributions_20260919.csv).

## What the finer grid changes

The completed refinement adds 31 Conv2 and 57 Conv3 beta settings, each with
72 checkpoint/batch replays: 6,336 new replays. Together with the 102 reused
settings, it provides 190 measured beta settings across six surfaces. All
twelve threshold boundaries have a next measured failure within 4.14–4.74%
of the selected passing value; no upper boundary remains open.

| Model / scheme | Largest measured beta passing every matrix >.90 | Largest measured beta passing every matrix >.95 |
|---|---:|---:|
| Conv2 baseline | 720.198 | 542.236 |
| Conv2 ours | 63.7712 | 34.0866 |
| Conv2 legacy | 43.0177 | 31.3825 |
| Conv3 baseline | 404.141 | 147.683 |
| Conv3 ours | 5.26876 | 2.49275 |
| Conv3 legacy | 4.42250 | 2.81845 |

The old grid had selected 500 for both Conv2 baseline thresholds; those limits
now separate. The Conv3 ours .95 selection changes from .9 to about 2.49.
Every selected point's limiting matrix is the first convolutional matrix at
initialization. These are limits on the recorded static diagnostic, not
training-stability limits. For example, the Conv2 legacy .95 boundary lies
above 31, while its already measured beta-30 training failed.

The curves need not be monotone: Conv3 legacy's worst cosine rises from
.960632 at beta 1 to .970996 at beta 1.5 before declining at larger values.
The table therefore reports the largest passing **measured** value and the
detailed report supplies its next failing sample; it does not prove a global
maximum. The six Conv3 refined selections have now completed ten-epoch
training (below); the six newly refined Conv2 values remain calibration-only.
See the [complete table, plots and CSVs](beta_refinement_20260919.md).

## Refined Conv3 selections: completed ten-epoch training

All six refined Conv3 values complete ten finite epochs and pass the declared
final-drop screen. They use seed0,zero read noise,T=K=8,the same saved
initializer/data order and fixed scheme-specific Adam rates, on Jean Zay H100.

| Scheme | Beta at per-matrix >.95 | Final validation | Beta at >.90 | Final validation | Larger minus smaller |
|---|---:|---:|---:|---:|---:|
| Baseline | 147.682614594 | 97.14% | 404.141105702 | 97.16% | +.02pp |
| Ours | 2.49274796756 | 98.48% | 5.26875648112 | 98.52% | +.04pp |
| Legacy | 2.81845428732 | 98.58% | 4.42250110273 | 98.54% | -.04pp |

The refined boundary choices show no consistent clean-accuracy penalty from
larger beta. Differences of one or two validation examples in a single seed
are descriptive, not evidence of equivalence or a reproducible advantage.
These six successes do not erase the prior Conv1/Conv2 counterexamples to a
static-cosine stability guarantee. Keep the empirical-training plus validation
protocol recommended below; use cosine as a separately reported fidelity
diagnostic. Do not retrofit this new rule as the historical selection method.

The p90 settings are admitted to the user-requested **30-epoch read-noise
sweep across V100, A100 and RTX 5090**, with fresh clean controls for every
scheme/GPU pair and the prior noise grid. The user-requested local acceleration
adds three RTX 5090 clean controls, for 24 outcomes in total. All24 now complete
thirty finite epochs. Their full-horizon evidence is separate from the clean
ten-epoch admission pilots; it remains single-seed validation evidence.
[Ten-epoch report](conv3_refined_beta_training_20260919.md) ·
[Completed noise sweep](conv3_p90_read_noise_20260919.md).

## Completed thirty-epoch read-noise evidence

All15 noisy runs and nine GPU-matched clean controls complete and pass the
predeclared final-best drop <5pp screen. The final validation drops below use
each scheme's own clean control in the same GPU/PyTorch/CUDA class. Positive
values mean that noise reduced accuracy.

| Read-noise sigma | GPU class | Baseline drop (pp) | Legacy drop (pp) | Ours drop (pp) |
|---:|---|---:|---:|---:|
| 1e-5 | V100 | 0.06 | 0.04 | 0.20 |
| 3e-5 | RTX5090 | 0.16 | 0.00 | 0.40 |
| 1e-4 | A100 | 0.02 | 0.08 | 0.64 |
| 3e-4 | A100 | 0.02 | 2.46 | 1.02 |
| 5e-4 | A100 | 0.18 | 3.00 | 1.38 |

At the highest noise, final validation is 97.40% for baseline, 95.72% for legacy,
and 97.12% for ours. Ours is less damaged than legacy at the two highest levels,
but baseline loses less accuracy than ours at every nonzero level in this sweep.
Legacy loses less than ours at the three lower levels. This is not evidence
for a universal robustness ordering. The curves combine three hardware/software
classes with matched controls; small nonmonotone differences should not be
interpreted as a precise noise threshold or a reproducible advantage.

The final-epoch screen also misses substantial temporary deterioration:
legacy drops 7.34pp below its running best at epoch17 for sigma3e-4 and 9.16pp
at epoch20 for sigma5e-4, then recovers to final best-to-final drops of 1.78pp
and 1.86pp. Thus finite completion and this endpoint screen do not establish
smooth training. These fluctuations are retained, with no new exclusion gate.

Relative to the earlier beta choices, the highest-noise final accuracies are
higher by 1.36pp (baseline), 20.28pp (legacy), and .52pp (ours). The injected
betas changed from 10/.001/3 to 404.141/4.42250/5.26876, respectively, and the
hardware/software environments also changed. This is a sensitivity comparison,
not an isolated causal estimate of beta's effect. It does show why conclusions
drawn from one set of beta values must identify those operating points.

These results support the faithfully reported p90 procedure below as a
qualified operating-point protocol for this Conv3 study. They do not establish
that .90 is optimal, that cosine guarantees stability, or that the largest
passing beta optimizes noisy training. No noisy p95 comparison, new refined
Conv2 training, multi-seed confirmation, or official-test evaluation is supplied
by this sweep. The earlier counterexamples to cosine-only selection remain.

## The selection rule actually used for the p90 sensitivity study

This follow-up has an auditable rule that can be described without claiming
that cosine predicts training stability:

1. At the frozen Conv3 solver, optimizer and weight contract, evaluate each
   candidate injected beta at zero noise on the same 36 batches of 16
   validation examples, at both initialization and the saved BPTT checkpoint.
   Require every weight matrix in all 72 replays to have cosine strictly
   greater than .90. Norm errors are recorded but are not an admission gate.
2. Select the largest **measured** passing beta in the declared refined grid:
   404.141105702 (baseline), 5.26875648112 (ours), and 4.42250110273 (legacy).
   The next measured failures are within 4.35%, 4.43%, and 4.61%, respectively.
   This does not prove monotonicity or identify a global maximum.
3. Qualify those choices with the completed, matched ten-epoch clean training
   pilots. All three pass the existing finite-training/final-drop screen.
   The .95 choices were also trained as a sensitivity control; differences
   at epoch10 are at most .04 percentage points in this seed.
4. Freeze each selected beta across the five read-noise levels and fresh
   thirty-epoch clean controls. Report the full curves and final accuracy
   relative to the GPU-matched clean control, including any poor outcome.

The rule chooses a finite-nudge operating point subject to a sampled gradient
fidelity constraint, then checks training empirically. It does not optimize
noisy validation accuracy and does not certify stability on unsampled states.
The thirty-epoch sweep is now complete, including all matched clean controls.
These are single-seed validation experiments, with no official-test evaluation.

The motivation for allowing larger nudges is a possible tradeoff between
finite-nudge distortion and noise in the gradient readout. That motivation is
not a theorem that the largest passing beta is best: the phase states also
change with beta, and the previous and current noise sweeps use different
hardware/software environments. The recommendation for a future selector
below therefore retains direct training validation and equal search budgets.

## Why cosine is an incomplete criterion

Cosine measures direction and discards magnitude. With
`r=||g_EP||/||g_ref||` and cosine `c`, the relative vector error satisfies
`||g_EP-g_ref||/||g_ref|| = sqrt(r*r + 1 - 2*r*c)`. Even with equal norms,
cosines .90/.95/.99 permit relative errors approximately .447/.316/.141.
Cosine>.95 therefore does not mean that every update differs by at most 5%.
The norm-mismatch and relative-error columns above are different quantities;
their worst cases need not occur in the same replay.

For plain SGD with scalar learning rate, the first-order loss change is
`g_ref·(-eta*g_EP)`. A positive gradient inner product gives a locally downhill
direction for sufficiently small steps when the reference is the relevant
loss gradient; .90 or .95 is not a special theoretical descent boundary.
Finite step size, curvature, changing states and the actual Adam/projection
update prevent promoting that local observation into a training guarantee.

Concatenating matrices weights the result through their gradient norms, so a
large layer can mask disagreement elsewhere. Requiring every matrix to pass
avoids that masking, but gives a worst-case diagnostic whose bottleneck need
not determine validation performance. Near-zero gradients also require explicit
handling: an undefined direction must not silently become a passing cosine.

The optimizer and the parameter constraints matter. Adam uses moment history,
coordinatewise preconditioning and learning rates, followed here by conductance
projection. Raw-gradient cosine is not the angle between the actual parameter
updates. A norm error can partly cancel under adaptive scaling, while a history-
dependent update can remain problematic despite good raw alignment. These are
mechanistic considerations, not causes isolated by the current experiments.
See [Kingma and Ba, Adam](https://arxiv.org/abs/1412.6980).

Calibration currently evaluates the matched initialization and a trained **BPTT**
checkpoint. It does not inspect all states visited by training with each new
EqProp beta. More batches on those same two states improve cohort coverage but
cannot eliminate this trajectory mismatch. The 36 batches are from the
validation partition of MNIST's official training split and have been reused
for selection; they are not an independent confirmation set.

Finite solver time remains a separate approximation. Our reference is BPTT
through exactly K zero-nudge iterations from the common post-T state. Classical
EP guarantees assume equilibrium and a limiting perturbation; the finite-T8
Conv3 residual caveat remains. Symmetric nudging reduces leading finite-beta
bias under the relevant regularity assumptions, but does not certify our
finite-iteration, piecewise-linear perfect-diode training. See
[Scellier and Bengio](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full)
and [Laborieux et al.](https://arxiv.org/html/2101.05536v1).

## Proposed selection procedure

1. **Freeze the experiment being optimized.** Keep the accepted T/K = 4/6/8,
   learning-rate vectors, optimizer, bounds, biases, initialization convention,
   data split and epoch/checkpoint rules. Report injected beta and its mapping
   to the base coefficient. Select separately for each architecture/scheme and
   weight contract; a wide-weight result does not qualify bounded weights.

2. **Declare a finite candidate grid and equal search budgets.** The refined
   gradient curves can inform the grid and identify approximation regimes.
   Include the conservative historical value and larger candidates, including
   an already trained larger value where available. Do not make the largest
   cosine-passing value the automatic winner. Use common multiplicative
   spacing and a common candidate/epoch budget across competing schemes.

3. **Screen using actual training.** Use the existing matched ten-epoch,
   seed-0 setup for initial screening. Retain all failures in the ledger.
   Reject nonfinite candidates and those failing the declared stability rule;
   the existing final-drop<5pp rule is only a gross-collapse screen, not a
   statement that a 4 pp loss is acceptable. Rank the survivors by a fixed
   validation endpoint. The present pilot comparison uses final epoch 10
   accuracy; its observations remain reported with that endpoint.

4. **Confirm finalists at the paper horizon.** Carry at most two candidates
   per surface to the intended 10/30/30-epoch budget with matched seeds 0, 1, 2.
   For paper selection, use the already frozen best-validation checkpoint rule
   consistently for all candidates. Require finite completion for every
   declared seed and report any failures and variability. Choose the candidate
   with the highest mean validation score across those runs; use a declared
   deterministic tie-break, such as the smaller beta for the clean-training
   objective. Small differences do not establish a scientific accuracy gain.
   These are development/confirmation seeds, not an independent seed holdout.

5. **Freeze beta before the final comparison and test access.** Preserve the
   complete development record. Apply the same selection rule to all schemes.
   Only after the beta, checkpoint-selection rule and inclusion contract are
   sealed may eligible runs use their one official-test evaluation. A new
   calibration limit does not automatically replace the betas in existing
   training or noise results.

This selects a useful operating point rather than asserting an optimizer-
independent gradient-fidelity threshold. The precise candidate grid and
selection/confirmation budget must be fixed before the proposed new training;
they have not been retrospectively declared to be the protocol of old runs.
If a practically meaningful near-best margin is preferred to strict ranking,
declare that margin in advance and support it with replication. There is no
evidence here for a universal .1pp/.2pp tolerance or for always selecting the
largest beta on an apparent single-seed plateau.

## The read-noise claim determines the selection objective

For **robustness of a fixed clean-selected configuration**, select beta using
clean training/validation and hold it fixed across the noise curve. This is a
defensible experiment and matches the structure of the existing noise sweeps.
The conclusion is robustness under the listed beta, optimizer and solver
contracts. It is not a comparison of each scheme's best achievable noisy
performance or a causal isolation of amplification from beta.

For **performance after adaptation to a noise level**, select beta under that
noise model using the same validation-search budget for every scheme. Declare
the primary noise level or an explicitly weighted noise grid, and validate
across training/noise seeds. If noise is unknown at deployment, use a declared
distribution or worst-case objective. Such retuned curves are a different
experiment. Our sigma values are simulator units, not a calibrated hardware
distribution, so an application-specific noise objective is not yet supplied.

The current endpoint-readout model perturbs copied non-input voltages for the
positive and negative phase measurements, while relaxation and validation
remain clean. In a local linear approximation with comparable numerator-noise
statistics, dividing their difference by 2B gives a variance contribution
scaling like 1/B². However, endpoint states and numerator sensitivity also
depend on beta; nonlinear products and optimizer effects remain. This
motivates investigating larger beta under noise but does not prove that the
largest clean cosine-passing beta maximizes noisy accuracy.

Existing Conv3 legacy noise results used B=.001 and deteriorated strongly;
the clean B=1 pilot nevertheless completed ten epochs at 98.64%. We have not
trained that B=1 candidate under the same noise sweep. Consequently, the
existing results do not establish that its noise penalty is unavoidable after
retuning beta. Conv3 baseline's noise sweep used B=10, whereas its completed
B=100/300/750 ten-epoch comparisons are clean. Keep these controls distinct.
See [ours/legacy noise results](read_noise_sweep_results.md) and
[baseline noise results](baseline_read_noise_results_20260916.md).

## More relevant diagnostics if a mechanism argument is needed

In addition to raw per-layer cosine, report relative gradient error, norm
ratios, gradient magnitude and the distribution across batches, not only a
single worst value. Repeat diagnostics at early/mid/late checkpoints of the
actual EqProp trajectory, preserving the matching optimizer state.

A closer diagnostic is the **actual projected optimizer step**: clone the
same parameter and Adam state, compute EP and reference updates, apply the
identical learning rates and projection, and compare update direction, size
and the ensuing loss change on separate probe batches. The first-order
quantity `-g_ref·delta_theta_EP` estimates descent only for the specified
reference objective and sufficiently small steps; it is not a guarantee for
finite-step training. Under noise, also estimate update variability and phase
signal relative to read noise. These diagnostics are proposals, not measured
results in this study, and should complement direct training validation.

For a noise-dependent auxiliary metric, the update error has the exact
bias–variance decomposition
`E||u_(B,sigma)-u_ref||² = ||E[u_(B,sigma)]-u_ref||² + tr Cov(u_(B,sigma))`
at fixed parameters, optimizer state and batch. Repeated endpoint-noise draws
can therefore expose the tradeoff between finite-nudge distortion and noisy
updates that a clean cosine omits. Report both layerwise and aggregate
quantities; a raw whole-vector MSE can also be dominated by large layers.
This is a proposed diagnostic, not a measured or validated replacement selector.

With the existing short T, displacement diagnostics should use the centered
odd phase signal `(s_plus-s_minus)/2` and a zero-nudge continuation control.
Raw free-to-nudged displacement can contain continued relaxation. A larger
observed displacement alone is therefore not evidence of a better learning
signal; see the existing [phase-displacement report](phase_displacement_20260916.md).

## What can be stated honestly in the current paper

Current evidence supports describing beta values as **empirically qualified
operating points**, with clean gradient comparisons as supporting diagnostics
and explicit sensitivity analyses. It does not support saying that a
universal cosine threshold guarantees stable or near-optimal training.
A new selector should be presented as a proposed or subsequently validated
extension, while the historical selection procedure is reported as it occurred.
The current single-seed data do not uniquely justify every historical beta
under one newly invented deterministic rule.

Suggested wording for the evidence available now:

> The nudging coefficient was treated as a numerical and optimization
> hyperparameter at fixed solver and optimizer settings. Gradient comparisons
> with the matched BPTT reference characterized the finite-nudge approximation,
> while training experiments assessed empirical stability and validation
> performance. Our sensitivity analysis shows that static gradient cosine
> similarity alone is insufficient to predict training stability. The reported
> read-noise curves keep their listed beta values fixed across noise levels and
> therefore characterize those operating points rather than performance after
> noise-specific beta optimization.

Outstanding work for the proposed stronger selector: full-horizon multi-seed
confirmation of its finalists, and noise-aware tuning only if the intended
claim is best performance under noise. Newly refined cosine-selected betas
require training qualification before promotion; the six refined Conv2 values
remain calibration-only, while the Conv3 p90 values now have the single-seed
thirty-epoch evidence reported above. No further selector-validation training,
paper edit or official-test evaluation was launched by this assessment.

[Refined calibration results](beta_refinement_20260919.md) ·
[Per-matrix training study](layerwise_beta_training_20260919.md) ·
[Larger-beta failure study](beta_training_stability_20260918.md) ·
[Plan](../docs/eqprop_beta_refinement_plan_20260919.md).

## September 21 addendum: sigma1e-3 exceeds the tested stability range

The subsequent three-case RTX5090 extension reused the same p90 betas and
completed clean controls. Baseline completed30 epochs at97.16% validation,
0.56pp below its clean control. Legacy became non-finite at epoch8/batch2791,
and ours at epoch18/batch1638, both in Layer_1 inference. Their last completed
validation values were94.76% and95.14%, not30-epoch outcomes. All three
terminal bundles are collected and validated, with both failures retained.

This strengthens the distinction between clean qualification and robustness
under read noise: even multi-batch zero-noise cosine checks and successful
clean training do not guarantee stable noisy training. A fixed-beta noise
curve should retain and mark non-finite outcomes. A claim of noise-qualified
stability needs training qualification at the stated noise level and horizon;
noise-specific beta tuning would be a separate study. No such tuning or
repeat was launched here. Single-seed and scheme-specific-optimizer limits
remain. See the [extension report](conv3_p90_read_noise_1em3_20260920.md).
