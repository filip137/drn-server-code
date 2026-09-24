# CIFAR L8 mechanism investigation: exploratory findings

Exploratory seed0 evidence, September23. GPU diagnostics completed on Fifi.
Scope corrected at the user's request: this is a BN investigation. Reuse the
existing LR sweep and selected legacy reference; the original mechanism study's
required new training run is voltage-normalized legacy at the unchanged selected legacy LR.
The subsequently approved seven-case BN study is reported separately below.
It started on Loulou RTX5090 at15:17UTC, worker1568364, tmux
`cifar-l8-bn-20260923`, state `launcher/loulou_bn.json`. Ten epochs,
batch32, expected about2.1h, cap4h within the original15GPUh total.
Queued proposed/legacy repeats were cancelled. The supplementary baseline-LR
control was stopped and its partial output is excluded from the primary
comparison. The nearly finished baseline repeat is retained as auxiliary data.
The old five-case generated report is a historical snapshot, not current coverage.
No official CIFAR test data was read. The normalized-legacy control completed
ten epochs and was collected/validated locally at18:27UTC. The seven-case BN
ablation study reached terminal coverage at20:42UTC: four qualified ten-epoch
results, two early scientific collapses, and one ten-epoch result with failed
solver qualification. All authoritative artifacts are local and validated;
failures are retained rather than counted as successful ten-epoch comparisons.

The subsequent [conclusions and proposed next experiments](cifar_l8_bn_conclusions_and_next_experiments_20260923.md)
combine the four optimizer/epsilon conditions and examine every matched epoch.
No new training or GPU replay was launched for that analysis.

## Voltage normalization restores baseline exactly

Using the same initializer, baseline learning rates, minibatches, augmentation
and Adam settings, baseline and voltage-normalized legacy matched exactly
through200 minibatches of32 examples. Maximum relative error was0 for free
and tracked logits, loss, all19 trainable gradients, BN buffers, parameters,
and projected updates.

Read-only comparisons using baseline's saved epoch0/10/30/50 weights gave the
same result on128 fixed training examples: normalized legacy matched every
logit and gradient exactly. Switching those weights to unnormalized legacy
changed logits by up to18.65%,14.62%,20.38%, and19.80% relative L2,
respectively. These percentages use the larger of the two logit norms as
denominator and the maximum across four batches32.

The corrected physical equations explain this identity. Within a zero-bias
perfect-diode block, substituting `v_l = A^(l-1) x_l` gives normalized edge
weights `(A*B)^p W_p`. Baseline and legacy both have `A*B=1`. Thus legacy
does not reduce normalized electrical loading relative to baseline in these
matched-weight blocks. It changes the output voltage units by16/16/4 before
the three BN boundaries. The [implementation audit](cifar_l8_amplification_audit_20260923.md)
documents the equations and the difference from a fully coupled MNIST circuit.

## Production BN epsilon materially changes normalization

For a positive constant `s`, training-mode BN obeys
`BN(s*x; epsilon) = BN(x; epsilon/s^2)` before its affine transform. Thus
legacy's output scales correspond to epsilon3.90625e-8 in the two depth3
blocks and6.25e-7 in the depth2 block, measured in baseline voltage units.
The configured epsilon is1e-5. Affine BN parameters remain trainable in the
original references and the normalization control; that control divides voltage
before pooling/BN and does not change epsilon. The newer frozen-affine
ablations are distinct cases described below.

Measured epoch50 training-mode epsilon shares, `epsilon/(variance+epsilon)`,
are below. Each entry is the median over channels and four batches on the
same128 examples, using the tracked pass. This compares separately trained
checkpoints and therefore includes their learned weight differences.

| Scheme | Block1 | Block2 | Block3 |
|---|---:|---:|---:|
| Baseline | 73.77% | 22.51% | 31.17% |
| Proposed | 61.43% | 8.54% | 14.04% |
| Legacy | 2.30% | 0.0117% | 1.64% |

The fixed epsilon is a substantial part of baseline's normalization, while
legacy is much closer to variance-only normalization. Evaluation using saved
running variances shows the same qualitative separation. This is a measured
scale/BN interaction; whether it helps accuracy is the causal-control question.

## Solver error and conductance-gradient collapse are not supported explanations

All12 actual checkpoints—three schemes at epochs0/10/30/50—passed the native
gradient/logit audit comparing operational `[6,6,4]`, reference `[24,24,16]`,
and sentinel `[48,48,32]` iterations. The largest operational projected KKT
relative residual in the full-batch tracked replay was6.72e-8; at trained
checkpoints it was about1e-8. These diagnostics do not indicate that legacy
is being disadvantaged by insufficient settling on this cohort. This conclusion
applies to the original trainable-BN checkpoints, not the later BN-free runs.
At epoch50 the native audit's free-logit relative errors were2.08e-7,
2.51e-7 and4.85e-7 for baseline, proposed and legacy, respectively; every
layer's mean gradient cosine exceeded0.99999999998.

Legacy's conductance gradients are not systematically smaller or more sparse.
For example, at epoch50 the first convolution's median gradient RMS is
2.55e-4 for legacy,1.64e-4 for baseline and4.26e-5 for proposed. Raw gradient
scale is not an Adam update size: the corresponding median projected next-step
update/weight L2 ratios are2.04e-5,2.06e-5 and4.24e-5. These proposals use the
saved Adam moments and next scheduled LR, without updating checkpoints.

A [full layerwise gradient/update comparison](cifar_l8_gradient_scale_interpretation_20260923.md)
confirms that legacy's convolution gradients are larger after training, but
the relative Adam steps do not grow proportionally. Median legacy/proposed
gradient ratios at10/30/50 are6.34/6.51/8.63, while relative-update ratios are
0.903/1.111/0.664. Legacy's much smaller conductance norms contribute to the
raw-scale distinction. This does not rule out differences in gradient noise,
update direction or functional sensitivity; a loss/logit probe along the
actual saved-Adam direction is a more direct next diagnostic than raw norms.

Boundary-gain gradients do differ substantially, as expected when BN nearly
cancels voltage scale. They should not be confused with conductance gradients.
At epoch50, the proposed updates of all three block-input gains round to zero
in float32 for every scheme on this cohort: the raw parameters are near100
and the final scheduled gain LR is small. This is a shared limitation of the
gain parameterization, not evidence of a legacy-specific failure.

The gradient convergence audit uses its native eight examples/microbatch2;
operational gradients, voltages and KKT measurements use128 examples/batch32.
They are kept separate. Parameters and checkpoint bytes were unchanged and
BN buffers were restored after replay.

## Learned input gains at the end of the original fifty-epoch runs

Read directly from the twelve collected epoch0/10/30/50 checkpoints using
[the extraction script](../experiments/summarize_cifar_l8_input_gains.py).
These are actual positive gains, `softplus(raw)`, not gradients or the fixed
amplification A/B. All four gains started at100. Their final values are:

| Scheme | Block1 input | Block2 input | Block3 input | Analog head input |
|---|---:|---:|---:|---:|
| Baseline | 100.051193 | 100.122528 | 100.352913 | 100.327759 |
| Proposed | 100.068626 | 100.113304 | 100.274414 | 100.277458 |
| Legacy | 99.994232 | 100.095139 | 100.001328 | 100.298927 |

The largest final block-gain displacement is0.353% of initialization; legacy's
largest is0.095%. All inspected epochs0/10/30/50 have gains near100; this
does not measure every intervening minibatch. The values
do not support a large learned gain difference as the explanation for the
accuracy gap. They also do not show that gain adaptation is unimportant:
the chosen optimizer barely explored that degree of freedom.

All gain groups use Adam with initial LR5e-5, decaying to1e-6. The prior CIFAR
conductance-LR search configs also kept this gain LR fixed at5e-5. Because
softplus'(100) is effectively1, this parameterization makes small absolute
updates to a parameter of size100. For equal unit-normalized Adam directions,
initial relative gain movement is5e-7, versus1e-3 for BN gamma initialized
at1: a2000-fold scale difference, not a measured ratio of actual updates.
Float32 spacing near the gain parameter is7.6294e-6.
In the existing epoch50 next-update replay (four training batches), all36
block-gain updates across the three schemes rounded to zero, despite nonzero
unrounded Adam proposals. Their largest magnitudes were6.14e-7,5.77e-7 and
2.76e-7 for baseline, proposed and legacy. This is a measured endpoint
limitation, not a claim that every earlier training update vanished.

Input gain can matter through precisely the BN mechanism under investigation.
For a zero-bias, homogeneous perfect-diode block at fixed weights, positive
input scaling g scales its pooled output by g. In training mode,
`BN(g*x; eps) = gamma*(x-mean(x))/sqrt(var(x)+eps/g^2) + beta`.
Gain therefore changes effective epsilon as well as voltage. Where variance
dominates epsilon, BN nearly cancels this scale and suppresses its learning
signal. Legacy's measured epsilon shares are much smaller than baseline's,
so this suppression is stronger. The head has no following BN; its gain
directly changes logit scale and hence cross-entropy. With fixed weights its
positive scaling preserves logit argmax, so it cannot directly change accuracy;
its effect on the optimization trajectory is a separate question. Without BN, the block
gains also propagate into logit scale: for these zero-bias homogeneous blocks,
their positive scalar product multiplies the final logits. At fixed weights,
changing these gains also preserves argmax in the BN-free model, while changing
cross-entropy and its training gradients. With BN, gain instead changes the
effective epsilon at each boundary. Evaluation additionally depends on
the saved running statistics; the formula above is a training-mode identity.

The completed voltage-normalized legacy control changes effective gain
substantially: by positive homogeneity, dividing block outputs by16/16/4 is
forward-equivalent to reducing input gains100 to6.25/6.25/25 at the same
weights. Its trainable raw gains still live near100 before the fixed output
division. This is not the same optimizer parameterization as directly
training raw gains initialized at6.25/6.25/25. Avoid counting a fixed input
rescaling and the equivalent output normalization as independent mechanisms.

A matched fixed-gain control versus stronger relative gain adaptation would
test whether this restricted adaptation matters for training. A possible
parameterization is `g=100*exp(theta)` with theta initialized at0, with a
separately chosen LR. This remains a proposed follow-up, not an implemented
change to these BN runs. Endpoint values alone cannot establish the
accuracy effect of changing gain training.

[Exact epoch0/10/30/50 values and checkpoint hashes](../results/cifar10-l8-amplification-mechanism-seed0-20260923-v1/analysis/input_gains.csv)
and [update summary](../results/cifar10-l8-amplification-mechanism-seed0-20260923-v1/analysis/input_gain_summary.json)
are retained locally. This extraction performed no training or checkpoint writes.

These scalar gains are distinct from BN's per-channel affine gamma. At epoch50,
median gamma values for boundaries1/2/3 are0.719/0.424/0.884 for baseline,
0.848/0.901/1.092 for proposed, and0.824/0.771/0.974 for legacy, starting
from1. Thus BN affine parameters did adapt substantially even though scalar
input gains barely changed. Channel distributions and beta RMS are preserved
in the [checkpoint summary](../results/cifar10-l8-amplification-mechanism-seed0-20260923-v1/analysis/bn_affine_checkpoint_summary.json).
This is descriptive evidence, not a matched intervention on gamma or gains.

The architecture also creates an exact functional redundancy for the later
gains: blocks2/3 and the classifier receive the preceding affine BN output
directly (with flattening before the classifier). Their clamp is
`g*(gamma*z+beta)`. Multiplying g by c gives the same clamp as keeping g
fixed and multiplying both preceding gamma and beta by c, in training or
evaluation, with the same BN statistics. Consequently nearly constant scalar
gains do not imply fixed downstream input voltage scales in the trainable-BN
network. The first block has no preceding BN; frozen-affine and no-BN variants
lack this trainable compensation. This is a forward-function identity from
the implemented boundary order, not equivalence of Adam optimization paths.

![Learned scalar gains at saved epochs](../results/cifar10-l8-amplification-mechanism-seed0-20260923-v1/analysis/input_gain_trajectories.png)

## Completed voltage-normalized legacy control

At ten epochs, voltage-normalized legacy reaches75.82% validation accuracy
and0.698843 validation CE, versus74.68% and0.717528 for the existing legacy
reference: +1.14percentage points. Its selected legacy optimizer, seed,
initializer, augmentation, cross-entropy and50-epoch cosine horizon were
unchanged. Training accuracy is74.00%, versus73.96% for the reference.
Final solver audit passed. All ten epochs, final checkpoint, Adam step14070
and the successful bundle were collected and validated locally; actual
runtime11294.81s, within the4h cap. Loulou GPU sharing ended at17:39UTC.

Actual learned gains end at100.052887,100.020485,100.038536 for the three
blocks and99.955627 for the head. Their movement is tiny; the substantial
intervention is the fixed output normalization16/16/4, which changes effective
input gain by those factors through homogeneity. This control supports a
beneficial BN/voltage-scale effect for legacy at ten epochs with its selected
optimizer. Combined with the reciprocal baseline-epsilon result below, it
supports the mechanism, but it does not establish how much of the original
fifty-epoch accuracy gap it explains or whether the effect generalizes across
seeds. The full learning trajectory qualifies the endpoint benefit: normalized
legacy wins at only4/10 epochs and averages -0.636pp relative to raw legacy
over epochs6–10. Its epoch10 training CE is essentially unchanged. These
descriptive means do not replace the declared endpoint or provide independent
replicates; a sustained optimization advantage is not yet established.
The BN-affine and no-BN outcomes are reported below.

[Locally validated control and final gains](../results/cifar10-l8-amplification-mechanism-seed0-20260923-v1/analysis/normalized_legacy_collection.json).

## Completed reciprocal BN ablation: reduced baseline epsilon

Collected and validated locally at18:10UTC: baseline_epsilon_legacy completed
all ten epochs (epoch1 recovered from Fifi, epochs2–10 on Riri), with exact
epoch coverage1–10, final checkpoint/Adam step14070 and passing solver audit.
Validation accuracy is70.18%, versus75.26% for the existing baseline at epoch10
(-5.08percentage points). Validation CE is0.871177 versus0.722593;
training accuracy66.62% versus74.08%. Only BN epsilon changed scientifically;
the baseline optimizer, initialization, data order, augmentation and schedule
were retained. This establishes an epsilon effect under that fixed optimizer,
not the best achievable accuracy after retuning or an explanation of the
whole fifty-epoch scheme gap.

Final input gains for blocks1/2/3 are100.002518,100.070824,99.987152;
head99.967903. Gain adaptation remained below0.071% despite the substantial
accuracy change. This result also extends the effective condition previously
sampled by the cancelled legacy-baseline-LR run; its five-epoch prefix matches
exactly and is not an independent replicate. The frozen/no-BN outcomes follow;
the reciprocal normalized-legacy control is reported above.

See [collection validation](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/baseline_epsilon_collection.json)
and the [live comparison](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/report.md).

## Completed legacy with frozen BN affine parameters

At ten epochs, frozen-affine legacy reaches73.02% validation accuracy and
CE0.783571, versus74.68% and0.717528 for trainable-BN legacy: -1.66percentage
points. Training accuracy is70.51%, versus73.96%. The final checkpoint confirms
all BN gamma values stayed1, beta stayed0, and each BN counter advanced exactly
14070 minibatches. Thus this freezes learned affine parameters while retaining
minibatch normalization and updating running statistics.

The saved epoch1–4 Loulou prefix and epoch5–10 Fifi continuation cover all ten
epochs exactly once. The complete bundle and checkpoint are local and valid;
final Adam step14070 and solver audit pass. Block gains are100.005264,
100.031868,99.960999; head99.712814. These are still close to their initial100.
This one-seed ten-epoch comparison supports a benefit from trained BN affine
parameters for legacy under its selected optimizer; the other schemes are
reported below.

[Frozen-BN checkpoint validation and gains](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/legacy_frozen_collection.json).

Baseline with frozen affine BN also completed ten epochs:70.72% validation,
CE0.849729, versus75.26%/0.722593 for trainable BN (-4.54pp). Training
accuracy69.22%. Its Riri checkpoint and full bundle are collected and valid;
final solver audit passed, all BN gamma1/beta0, running-stat counters14070,
and Adam14070. Block gains100.012764/99.990143/99.922058 and head99.746475
again show little scalar compensation. Under the respective fixed selected
optimizers, removing affine learning hurts baseline more than legacy at
epoch10. This does not support learned BN affine parameters specifically
hurting legacy; it does not identify the cause of the fifty-epoch ranking.

[Baseline frozen-BN validation and gains](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/baseline_frozen_collection.json).

Proposed with frozen affine BN completed ten epochs at73.76% validation,
CE0.748592, versus75.36% with trainable BN (-1.60pp). Training accuracy71.29%.
Its final solver audit passed; the local checkpoint confirms gamma1/beta0,
running-stat counters14070 and Adam14070. Block gains100.009857/100.017990/
99.957016 and head99.702438 remain close to initialization.

| Scheme | Trainable BN, epoch10 | Frozen affine BN, epoch10 | Difference |
|---|---:|---:|---:|
| Baseline | 75.26% | 70.72% | -4.54pp |
| Proposed | 75.36% | 73.76% | -1.60pp |
| Legacy | 74.68% | 73.02% | -1.66pp |

Learned affine BN benefits all three at these selected settings. Its removal
hurts baseline more; proposed and legacy have similar drops. This supports
an optimization/expressivity role for affine BN, not an explanation in which
learning its parameters specifically harms legacy. The scalar gains barely
compensated in any frozen-affine case. A single seed, different selected
conductance rates, and the3090/5090 software difference limit broader rankings.

[Proposed frozen-BN validation and gains](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/ours_frozen_collection.json).

## Baseline without BN: early zero-activation collapse

With the unchanged selected baseline optimizer, no-BN training completed one
epoch at10% validation accuracy and CE2.302585, then stopped at the first
epoch2 batch because the runner rejected a zero gain gradient. The failed
bundle and saved epoch1 checkpoint are preserved locally; this is not a
completed ten-epoch result.

A single read-only replay of that exact rejected batch found all three pooled
block outputs and all logits exactly zero. Every one of the13 trainable tensor
gradients had zero nonzero entries and zero float64 norm, ruling out float32
norm underflow as the cause of rejection. Adam first-moment entries were at
subnormal scale (maximum5.6052e-45). All four learned gains were99.999733.
The replay performed no optimizer updates and left parameters unchanged.
This supports a scientific zero-activation/gradient collapse under the selected
settings. No altered-LR retry or nominal ten-epoch extrapolation is included.
It does not establish that BN-free baseline cannot train with different
initialization, gains or optimizer settings. This replay used the declared
[6,6,4] iterations; the stopped run did not reach its final higher-iteration
solver audit, so no final solver-convergence claim is made for this case.

[Exact failed-batch replay](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/baseline_no_bn_zero_gradient_replay.json).

Legacy without BN also stopped on the first epoch2 batch after epoch1
validation10% and CE2.302585. Its exact failed-batch replay likewise has zero
outputs at all three block boundaries, zero logits, and exactly zero gradients
for all13 parameter tensors in float64/nonzero-count checks. Its four gains
are all99.999657; Adam first moments are at the same subnormal scale. The
epoch1 checkpoint (Adam1407), failed bundle and log are collected locally.
The single replay took1.031s and performed no optimizer updates. No retry,
ten-epoch extrapolation or final solver-convergence claim is made for this case.
Both failures are outcomes under the selected BN-trained hyperparameters;
they do not establish a general inability to train without BN.

[Legacy failed-batch replay](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/legacy_no_bn_zero_gradient_replay.json).

## Proposed without BN: ten epochs completed, final solver check failed

Proposed/no-BN completed ten epochs at52.08% validation accuracy and CE1.538326
(training49.05%). The full checkpoint, metrics, log and bundle are collected
locally and internally valid, with Adam step14070. However, the final native
solver audit failed its reference-stability check: free-logit relative L2
difference was0.08608 between reference[24,24,16] and sentinel[48,48,32], above
the0.01 threshold. The tracked-logit difference was0.009407; all reported
reference gradient-layer checks passed, but the unstable free reference
prevented qualification of operational[6,6,4].

Consequently52.08% is an observed result at the configured iterations, not a
qualified steady-state accuracy or a clean estimate of BN's effect alone.
The original trainable-BN proposed reference is75.36% at ten epochs, but that
comparison carries this additional solver limitation. No training iteration
budget, LR or checkpoint was changed retrospectively. Future BN-free causal
comparisons need settling qualification as well as attention to changed
activation/loss scales; the two collapsed BN-free runs also lack final audits.

Final learned gains are100.024506,100.024422,100.017387 for the blocks and
100.017365 for the head. Their actual endpoint values remain informative even
though the run's steady-state qualification failed.
[Collected result and gains](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/ours_no_bn_collection.json).

## BN ablations: earlier preliminary observations at 16:25 UTC

The [new ablation report](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/report.md)
reuses the existing references and compares each intervention at the same epoch.
After one epoch, reduced-epsilon baseline has31.86% validation versus27.92%
for its reference; frozen-affine legacy30.48% versus35.52%; frozen-affine
proposed34.46% versus36.16%. Normalized legacy reaches52.84% at epoch3,
versus50.84% for the existing legacy reference. These are early optimization
observations, not final ten-epoch conclusions or proof of the fifty-epoch gap.

The accepted no-BN smokes also expose a scale effect before any optimizer
update, at identical initialization and augmented minibatch. Logit RMS is
8.52383e-7 for baseline,5.93856e-6 for proposed and8.72840e-4 for legacy.
Legacy/baseline equals1024 exactly, consistent with multiplying the positive
block factors16*16*4 through homogeneous pooling and the analog head.
Consequently removing BN also changes the scale entering cross-entropy;
the no-BN runs cannot isolate learned affine parameters alone. The separate
frozen-affine runs retain batch normalization and update running statistics,
with gamma1/beta0 fixed, to address that narrower question.

The reciprocal baseline-epsilon control overlaps mathematically with the
previously stopped legacy-baseline-LR intervention: at identical weights and
optimizer settings, legacy's scaled block outputs with epsilon1e-5 give the
same training normalization as baseline with epsilon/s^2. Running-statistic
initialization has a transient unit-conversion distinction in evaluation.
Observed train accuracy, train CE, validation accuracy and validation CE match
exactly at epochs1–3 across the two records. The old record stops after five
completed epochs; the current run supplies the ten-epoch endpoint. Do not
count the two prefixes as independent evidence or as two different effective
training conditions. The earlier partial LR-control artifact remains labelled
cancelled, not relabelled a completed run.

## Coverage and remaining scientific questions

The ten-epoch normalized-legacy control is complete and compared above against
existing legacy training. Cancelled reference repeats and the stopped
supplementary LR intervention are not required coverage.
The separately authorized [BN ablations](cifar_l8_bn_ablation_plan_20260923.md)
attempted all seven cases. Four qualified ten-epoch results, two early
zero-activation collapses, and one completed but solver-unqualified no-BN
result account for all coverage. All seven bundles and stopped/final
checkpoints are local and internally valid; interrupted prefixes and original
failure receipts are retained. The Riri/local parent queues were stopped only
after their frozen-BN children completed, so outsourced no-BN cases were not
repeated. No training remains active. Charged compute15.2119GPUh is below
the24GPUh cap, including checks, interrupted work, and failed-batch replays.

All interventions preserve selected LRs, trainable gains, augmentation, CE
and the original50epoch cosine horizon. A single seed and ten epochs cannot
establish a general scheme ranking or fully explain the final50epoch gap.
The no-BN results do not provide three qualified converged endpoints; that
comparison remains scientifically incomplete. Relative/log gain adaptation
with a fixed-gain control is a possible follow-up, especially for the first
block or frozen-affine BN, and was not launched. Any future BN-free accuracy
comparison also needs solver qualification at the trained operating point.

[Terminal coverage, checkpoints, and budget validation](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/collection_validation.json).

[Plan](cifar_l8_mechanism_plan_20260923.md) ·
[Historical initial-scope report](../results/cifar10-l8-amplification-mechanism-seed0-20260923-v1/analysis/report.md) ·
[BN measurements](../results/cifar10-l8-amplification-mechanism-seed0-20260923-v1/analysis/batchnorm.csv) ·
[Gradient/Adam measurements](../results/cifar10-l8-amplification-mechanism-seed0-20260923-v1/analysis/gradients_adam.csv) ·
[Matched-weight measurements](../results/cifar10-l8-amplification-mechanism-seed0-20260923-v1/analysis/matched_weights.csv).
