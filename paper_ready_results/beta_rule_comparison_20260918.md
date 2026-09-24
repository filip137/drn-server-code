# Seed-0 beta calibration: matrix versus whole-gradient criteria

All 153 declared beta cases are terminal: 153 complete measurement bundles and 0 retained numerical failures. The sweep measured 11,016 checkpoint/batch gradients and 33,048 weight-matrix comparisons.

Whole-gradient selection admits a larger tested beta in **50/54** matched comparisons. Adding the norm constraint changes **16/54** selections. These counts compare decision rules on identical measurements.

All values below are injected beta; require every batch at both initialization and the BPTT best-validation checkpoint. A star denotes a passing upper grid edge, not an identified maximum. “None” means no tested beta passed.

The frozen grid multiplies each current anchor by `.01, .03, .1, .3, 1, 2, 3, 5, 7.5, 10, 20, 30, 50, 75, 100, 300, 1000`. These are largest passing tested points, with no additional decade reduction.

## Cosine only

| Architecture/scheme | Current anchor | Matrix .90 | Matrix .95 | Matrix .99 | Whole .90 | Whole .95 | Whole .99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| conv1 baseline | 100 | 3000 | 1000 | 300 | 10000 | 10000 | 5000 |
| conv1 ours | 30 | 2250 | 1500 | 300 | 3000 | 3000 | 1500 |
| conv1 legacy | 3 | 900 | 300 | 60 | 3000* | 900 | 900 |
| conv2 baseline | 100 | 500 | 500 | 100 | 3000 | 2000 | 1000 |
| conv2 ours | 10 | 50 | 30 | 10 | 200 | 100 | 100 |
| conv2 legacy | 0.03 | 30* | 30* | 3 | 30* | 30* | 30* |
| conv3 baseline | 10 | 300 | 100 | 10 | 1000 | 1000 | 750 |
| conv3 ours | 3 | 3 | 0.9 | 0.9 | 30 | 30 | 22.5 |
| conv3 legacy | 0.001 | 1* | 1* | 0.1 | 1* | 1* | 1* |

## Cosine plus symmetric norm mismatch ≤ 0.10

| Architecture/scheme | Current anchor | Matrix .90 | Matrix .95 | Matrix .99 | Whole .90 | Whole .95 | Whole .99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| conv1 baseline | 100 | 3000 | 1000 | 300 | 10000 | 10000 | 5000 |
| conv1 ours | 30 | 2250 | 1500 | 300 | 3000 | 3000 | 1500 |
| conv1 legacy | 3 | 300 | 300 | 60 | 3000* | 900 | 900 |
| conv2 baseline | 100 | 300 | 300 | 100 | 2000 | 2000 | 1000 |
| conv2 ours | 10 | 20 | 20 | 10 | 100 | 100 | 100 |
| conv2 legacy | 0.03 | 3 | 3 | 3 | 30* | 30* | 30* |
| conv3 baseline | 10 | 50 | 50 | 10 | 1000 | 1000 | 750 |
| conv3 ours | 3 | 0.3 | 0.3 | 0.3 | 30 | 30 | 22.5 |
| conv3 legacy | 0.001 | 0.3 | 0.3 | 0.1 | 1* | 1* | 1* |

## Interpretation and limits

Whole-gradient cosine weights each matrix through its gradient magnitude. It can conceal a poorly aligned small-gradient matrix. The layer CSV retains individual cosines, squared-norm contributions, RMS, sparsity and scale relative to the same initialization minibatch. The whole-gradient norm gate can similarly conceal matrix-specific scale errors. With layer-specific learning rates or Adam, raw-gradient alignment also differs from alignment of the actual optimizer update.

For EP gradient e and BPTT gradient b, cosine is `(e · b) / (||e|| ||b||)` and symmetric norm mismatch is `2 | ||e|| - ||b|| | / (||e|| + ||b||)`. Matrix checks evaluate these separately for every weight tensor; whole-gradient checks concatenate all trainable weight tensors before evaluating them. Each rule must pass all 72 checkpoint/batch comparisons; a single failing matrix also rejects a matrixwise candidate.

The Conv3 balanced/ours anchor beta 3 retains its already documented initialization failure: first-convolution cosine 0.939168 on historical batch 2, with norm mismatch 0.278178. Thus its smaller matrixwise selection is not evidence that previously stable training diverged; training stability and this all-batch direct-gradient requirement are distinct. See the [earlier qualification record](../docs/conv_paper_one_seed_bptt_eqprop_protocol.md).

The comparison changes the acceptance rule only. Raw gradients have the usual EqProp amplification normalization, but receive no layer normalization, learning-rate scaling or Adam transformation. Frozen biases are excluded. No additional decade margin is applied to the selected values. Passing regions are recorded without assuming monotonicity.

These are zero-noise seed-0 diagnostics on the September 14 selection cohort (four historical plus 32 additional batches of 16), using the original wide [0,100] initialization/BPTT checkpoints, float64 centered frozen-current EqProp, and T=K=4/6/8. Every comparison starts from the same post-T state. Equilibrium residuals remain separate; the Conv3 baseline T8 free-state caveat is not removed by a relaxed cosine criterion. This study establishes neither full-training stability nor multi-seed qualification and does not change any training beta. Official-test reads and optimizer steps are zero.

## Verification and artifacts

Historical regression: 13 complete overlapping cases, maximum absolute metric difference 1.099120794378905e-14. All included canonical bundles validate, and source-byte, parameter, float64, cohort, frozen-force and zero-bias guards pass. The aggregation/selection and existing cohort/TK tests passed (26 tests).

The BPTT reference is beta-independent across 31,104 repeated layer comparisons: maximum norm difference 0, with identical input payload hashes.

Charged combined GPU time: **2.7499/6 physical GPU-hours**, including smokes. Conv3 ran on local RTX3090; all Conv1/Conv2 schemes and betas ran on Akib RTX3080 after an exact historical smoke regression. The user-authorized parallel placement changes five filesystem paths only; source, checkpoints and cohorts remain identical. A two-worker local benchmark gave 1.068x throughput, so production used one worker per GPU. The earlier local coordinators were retired between cases without discarding any measurement. Per-case commands, source identities, logs and receipts are retained in the study directory.

- [108 joint beta selections](beta_rule_comparison_20260918_selections.csv)
- [Initialization/trained/joint selections](beta_rule_comparison_20260918_checkpoint_selections.csv)
- [Case summaries](beta_rule_comparison_20260918_case_summary.csv)
- [Layer measurements and norm contributions](beta_rule_comparison_20260918_layer_metrics.csv)
- [Whole-gradient measurements](beta_rule_comparison_20260918_whole_gradient_metrics.csv)
- [Verification](beta_rule_comparison_20260918_verification.json)
- [Beta-limit figure](beta_rule_comparison_20260918_limits.png) / [PDF](beta_rule_comparison_20260918_limits.pdf)
- [Cosine curves](beta_rule_comparison_20260918_cosine.png) / [PDF](beta_rule_comparison_20260918_cosine.pdf)
- [Norm-mismatch curves](beta_rule_comparison_20260918_norm.png) / [PDF](beta_rule_comparison_20260918_norm.pdf)
- [Layer contributions at current beta](beta_rule_comparison_20260918_contributions.png) / [PDF](beta_rule_comparison_20260918_contributions.pdf)
- [Raw study](../results/eqprop-beta-rule-comparison-20260918-v1/)
- [Execution plan and transport amendment](../docs/eqprop_beta_rule_comparison_plan_20260918.md)

In the curve figures, green vertical lines mark the current anchors. Horizontal cosine lines mark .90/.95/.99; the horizontal norm line marks .10. Contribution bars average per-batch squared-norm fractions.
