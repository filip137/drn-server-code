# CIFAR L8: conclusions and proposed next experiments

Analysis of the completed exploratory seed0 studies, September 23, 2026.
This is a proposal, not a launch record. No new training or GPU replay was
started for this analysis. Existing references and checkpoints are reused.

Execution update: Filip subsequently authorized this round for the September23
night, with an08:00 Paris stop. See the [overnight execution plan](cifar_l8_bn_gain_overnight_plan_20260923.md)
and [collected findings](../results/cifar10-l8-bn-gain-followup-seed0-20260923-v1/analysis/report.md).
All declared cases are now complete, including the separately approved last
two epochs on September 24. Normalized legacy finishes epoch 50 at 88.82%
versus raw legacy 88.72%, without a sustained late validation improvement.
The proposals below are the historical plan, not a pending queue of repeats;
the linked findings contain the updated interpretation.

## Main conclusion

Keep trainable affine BN for the next CIFAR runs. The clearest measured
mechanism is an interaction between voltage scale, BN epsilon, and the chosen
Adam rates. The evidence does not show that learning BN affine parameters
specifically hurts legacy, nor that legacy's original training fails because
of its KCL solver or vanishing conductance gradients. The cause of its final
50-epoch accuracy gap is not yet established.

All results are validation on the existing 45,000/5,000 split, with augmentation,
cross-entropy, batch32, and the original 50-epoch cosine horizon. There is one
seed and mixed GPU/software placement; no official CIFAR test data was read.

## The existing cases form a useful four-way comparison

At matched weights, baseline and legacy have the same normalized block
loading because both have A*B=1. Legacy scales the three block outputs by
16/16/4. Expressed in baseline voltage units, its uncorrected BN epsilon is
[1e-5/256, 1e-5/256, 1e-5/16], instead of [1e-5, 1e-5, 1e-5]. The existing
controls therefore provide these epoch10 validation accuracies:

| Selected Adam vector | Baseline effective epsilon | Legacy effective epsilon |
|---|---:|---:|
| Baseline rates | 75.26%: baseline reference | 70.18%: baseline epsilon control |
| Legacy rates | 75.82%: voltage-normalized legacy | 74.68%: legacy reference |

The epsilon change costs 5.08pp with baseline rates and 1.14pp with legacy
rates: a descriptive difference of effects of 3.94pp at this endpoint. The
legacy rates improve the low-epsilon condition by 4.50pp, versus 0.56pp at
baseline epsilon. This is evidence of optimizer/normalization coupling under
these settings, not another LR sweep or independent replication. The raw
running-variance initialization has a transient unit-conversion distinction;
the previously observed identical five-epoch prefix supports the practical
mapping used here. Rates, all other scientific settings, and provenance are
available in the original configs and plans.

The normalization benefit for legacy must be qualified. It wins at only four
of ten measured epochs. Its mean accuracy difference over epochs6–10 is
-0.636pp, despite +1.14pp at epoch10; training CE at epoch10 is essentially
unchanged, 0.74813 versus 0.74621. Thus the endpoint is promising, but there is
no consistent acceleration of training so far. In contrast, baseline's
low-epsilon control loses 6.268pp on average over epochs6–10 and raises
epoch10 training CE from 0.74068 to 0.94429. Its effect includes a substantial
training penalty, not merely an evaluation-statistics difference.

These late-epoch averages describe trajectory sensitivity. They do not replace
the declared epoch10 endpoint or treat correlated epochs as independent seeds.
The original ranking also changes with time: baseline/legacy are 86.54/86.88%
at30 and 89.48/88.72% at50; proposed reaches89.92% at50. A ten-epoch intervention
cannot yet explain that late difference.

## What the other controls establish

Frozen affine BN retains minibatch normalization and updates running statistics,
but fixes gamma1/beta0. It reduces epoch10 accuracy by 4.54/1.60/1.66pp for
baseline/proposed/legacy, respectively. All three frozen cases are below their
trainable references at every measured epoch and have worse training CE.
Learned affine BN helps under all three selected optimizers. It is not supported
as a legacy-specific harmful mechanism; a stronger claim about generalization
or optimal frozen-BN rates would require different evidence.

No-BN is not a clean test of affine learning. Baseline and legacy collapsed
to zero outputs/gradients after epoch1 at10% accuracy. Proposed reached52.08%
at10 but failed final solver reference stability (8.61% free-logit discrepancy,
limit1%). All are outcomes at BN-selected hyperparameters, with dramatically
changed signal scales. They do not establish that BN-free training is impossible
or that the observed proposed accuracy is a converged-equilibrium result.

The original trainable-BN checkpoints all passed solver/gradient audits.
There is no measured systematic conductance-gradient collapse in legacy.
The CIFAR blocks are isolated by digital pooling/BN and voltage clamps;
electrical feedback does not cross those boundaries. The MNIST intuition
about a fully coupled, more feed-forward circuit therefore does not transfer
automatically. In these blocks, normalized legacy with the complete baseline
optimizer matched baseline exactly over200 real training minibatches.

Scalar gain adaptation remains a plausible untested opportunity, not an
established explanation. Original gains start100 and move at most0.353% by50;
late block-gain Adam proposals round to zero in float32. Later gains also have
functional redundancy with preceding BN gamma/beta, unlike the first block.
Frozen-BN gains likewise barely move. A useful intervention must permit
meaningful relative adaptation and separate it from classifier logit scaling.

## Recommended next round, in priority order

Gradient-scale follow-up: the [layerwise Adam analysis](cifar_l8_gradient_scale_interpretation_20260923.md)
finds much larger legacy raw convolution gradients without proportional
relative updates. Within the proposed diagnostic allowance, prioritize a
small loss/logit probe along actual saved-Adam directions at epoch30 before
fresh gain training; expand to10/50 only if needed and within that allowance.
This is an additional proposed diagnostic, not a new launch or an LR change.

1. **Check BN running statistics without further training.** On copies of the
   three original epoch10 and epoch50 checkpoints plus normalized legacy at10
   (seven checkpoints), compare stored BN statistics with statistics recomputed
   from one fixed, predetermined 4,096-example training-only cohort, without
   augmentation. Recompute moments sequentially BN1→BN3 so each later layer
   sees the recalibrated preceding layers; keep weights, affine parameters,
   gains and solver settings fixed. Evaluate the same full validation split
   before/after, keeping original checkpoint bytes unchanged. Use no validation
   labels to fit statistics. Report all paired accuracy/CE changes and changes
   in scheme gaps; retain the original evaluations as the primary records.
   This tests whether inference statistics contribute to the observed gap or
   the noisy epoch-to-epoch evaluations. It is a hypothesis, not a diagnosed
   defect. Suggested cap:1 GPUh on one RTX5090.

2. **Continue voltage-normalized legacy from epoch10 to50.** This is the
   highest-value additional training case for the original question. Restore
   the existing checkpoint's conductances, BN parameters/buffers, gains, Adam,
   scheduler and RNG; preserve its selected legacy rates, output divisors
   16/16/4, augmentation, CE and original50-epoch cosine horizon. Inspect30 as
   a milestone, but keep50 as the primary endpoint because the original
   scheme ranking changes after30. Compare with the already completed raw
   legacy88.72%, baseline89.48% and proposed89.92%; rerun no references. Keep
   the recalibration diagnostic separate from the saved training continuation.
   At measured exclusive5090 throughput,40 additional epochs take about8.4h;
   suggested cap10 GPUh including the resume smoke and final audit. Shared
   placement needs a revised measured duration. Use checkpointed overnight
   segments if the complete continuation does not fit the allowed window.

3. **A small gain-adaptation comparison in legacy with frozen affine BN.**
   Use two fresh matched cases from the same original seed0 initializer:
   (a) all three convolution-block input gains fixed exactly100;
   (b) those three gains g=100*exp(theta), theta initialized0, Adam LR1e-3
   with the existing cosine horizon/floor. Keep the analog classifier's
   original gain parameterization and5e-5 LR in both arms, so classifier
   gain policy is not another intervention. Keep the exact selected legacy
   conductance/head rates, gamma1/beta0 with running-stat updates, batch32,
   augmentation and CE. Ten epochs each; compare gains, layer voltage/BN
   epsilon shares, training/validation CE and accuracy, and solver convergence.
   Reuse the completed legacy_frozen_bn raw-gain run as a secondary reference.
   This deliberately tests stronger relative gain adaptation; it does not
   isolate parameterization from step-size effects. If it helps, extend the
   fixed/log pair to other schemes before making a general claim. Expected
   about4.2 GPUh total on5090; suggested cap5 GPUh including smokes/audits.

The proposed first round has a16 GPUh total cap, with about12.6 GPUh of
training plus diagnostics/checks expected on exclusive5090s. The long
continuation and gain pair can occupy two matched5090 environments in parallel
after live admission checks. No hosts are reserved and no new result roots,
configs or jobs have been created by this proposal. Initial/grown gain operating
points still require the applicable scientific solver checks; a new failure
must not be hidden by changing iterations, LRs or bounds mid-comparison.

Defer further BN-free training. First replay the existing failed checkpoints
at qualified longer settling budgets, then, in a separately specified follow-up,
test signal-scale and optimizer settings that are appropriate without BN.
Also defer broad LR sweeps, all-case50-epoch extensions, and equivalent
input/output rescaling repeats. Once a useful normalization or gain effect
persists, a matched additional seed is more informative than another seed0
reference repeat. Continuing the baseline-epsilon control would complete the
long-horizon four-way comparison if that interaction becomes the main question.

## Evidence

- [Full findings and original gains](cifar_l8_mechanism_findings_20260923.md).
- [Cases, metrics, gains and trajectory plots](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/report.md).
- [Every matched-epoch difference](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/matched_epoch_differences.csv).
- [Optimizer/epsilon comparison](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/epsilon_optimizer_comparison.json).
- [Terminal coverage and validation](../results/cifar10-l8-bn-ablation-seed0-20260923-v1/analysis/collection_validation.json).
- [Original comparison through50](../results/cifar10-l8-analog-baseline-legacy-e30-e50-seed0-20260922-v1/analysis/report.md).
