# Current Perfect-Diode Conv State

Updated: 2026-09-24

This is a dashboard, not a second protocol. Selection rules remain in the
[active protocol index](conv_paper_hyperparameter_protocol.md). Live running
state is generated separately in
[`current_simulations.md`](current_simulations.md).

## Current Focus: Finding the Right Beta

Recorded September 23, 2026, reflecting Filip's interpretation and proposed
direction.

The current focus is finding the correct beta. Historically, the betas we
chose for the legacy schemes were too small, which unfairly penalized them
under read noise. In the regimes where we inject read noise, we obtain much
better performance with higher betas.

The issue with higher betas is not necessarily a decrease in accuracy:
training can diverge, somewhat randomly and sometimes only at high read-noise
levels. This behavior appears quite stochastic.

The objective is to find a defensible way to choose the right beta. We are
considering choosing beta so that the RMS displacement at the output at
initialization is fixed across all runs.

### Evidence from the recent Conv3 read-noise runs

The September 19–23 evidence below uses seed-0 Conv3, thirty-epoch training
budgets, T=K=8, centered float64 EqProp, frozen-zero biases and inherited
scheme-specific Adam rates. Noise affects endpoint gradient readout;
validation is clean. Beta values are actual injected values. The labels
p90/p95/p99 denote clean cosine thresholds .90/.95/.99, not probabilities
of stable training. All accuracies are ordinary-MNIST validation measurements;
the official test set was not evaluated.

At read-noise sigma `5e-4`:

| Scheme | Candidate | Injected beta | Thirty-epoch outcome | Best / final validation |
|---|---|---:|---|---:|
| Legacy | p99 | 0.1 | Completed; stability screen passed | 96.86% / 96.86% |
| Legacy | p95 | 2.81845428732 | Nonfinite at epoch 28 | 97.70% / no final result |
| Legacy | p90 | 4.42250110273 | Completed; stability screen passed | 97.58% / 95.72% |
| Ours | p99 | 0.987333678708 | Completed; stability screen passed | 96.32% / 96.04% |
| Ours | p95 | 2.49274796756 | Completed; stability screen passed | 96.78% / 96.66% |
| Ours | p90 | 5.26875648112 | Completed; stability screen passed | 97.16% / 97.12% |

Sources: [p99 completed runs and local validation](../results/eqprop-conv3-p99-read-noise-5em4-20260922-v1/production-02-summary.md),
[p95 terminal outcomes](../results/eqprop-conv3-p95-read-noise-5em4-20260921-v1/status-update-20260922.md),
and [p90 completed sweep](../paper_ready_results/conv3_p90_read_noise_20260919.md).
The stability screen requires thirty finite epochs and a final drop strictly
below 5 percentage points from the run's own best; it does not require a
smooth trajectory.

For the historical small-beta reference, legacy at injected beta `.001`
finished at only **75.44%** at the same sigma, compared with **96.86%** at
beta `.1` and **95.72%** at beta `4.4225` in the recent runs. At sigma `1e-4`,
the old `.001` and recent `4.4225` results are **83.08% versus 98.64%**.
These measured gains support the interpretation that the earlier small beta
penalized legacy under noise. They do not imply that accuracy increases
monotonically with beta. The [old/new comparison](../paper_ready_results/conv3_p90_read_noise_20260919.md)
also records GPU/runtime differences: the p90 high-noise cases use A100s,
the p95/p99 cases use V100s, and the older reference uses local GPU contracts.
Consequently these are descriptive comparisons, not an isolated beta-only
effect with identical realized noise.

### Divergence and the apparent stochasticity

The p90 legacy/ours betas above completed thirty clean epochs and all tested
noise levels through `5e-4`. At sigma `1e-3`, however, legacy became nonfinite
in **epoch 8** and ours in **epoch 18**, after reaching best validation
accuracies of **97.22%** and **96.40%**. Baseline completed that high-noise
run at 97.16%. These are actual numerical failures, not merely lower final
accuracy. See the [Conv3 high-noise extension](../paper_ready_results/conv3_p90_read_noise_1em3_20260920.md).

The late p95 legacy failure at sigma `5e-4` occurred after a best validation
of 97.70%; its last completed epoch, 27, had fallen to 67.08%. By contrast,
the larger p90 legacy beta completed at that sigma, despite a temporary
**9.16 pp** decline from its running best at epoch 20. This irregular pattern
motivates the description of divergence as apparently stochastic. These
studies do not include repeated identical-contract noise-seed trials that
would establish a failure probability or separate stochasticity from
runtime and trajectory differences. “Quite stochastic” remains the working
interpretation, rather than a measured failure law.

### Evidence for considering a fixed initial output displacement

A direct Conv3 initialization replay already demonstrates that a common
output-displacement target can be matched. Injected betas **88.7 / 1.385 /
0.02173** for baseline/ours/legacy give pooled output
`RMS(v_plus - v_free)` of **1.000301 / 0.999812 / 0.999881**, respectively,
on the same 576 validation examples at T=K=8. All three are within 0.031%
of one. See the [measured initialization displacement](conv3_unit_output_displacement_beta_20260922.md).

The recent p99-trained checkpoint analysis also shows why displacement is
relevant: at common replay beta `.987333678708`, legacy's third-convolution
centered phase signal is about **16.6 times** ours epoch 30's. The measured
gradient-noise curves align more closely against phase RMS than against beta.
See the [phase-signal analysis](../results/eqprop-conv3-p99-noise-rms-analysis-20260923-v1/analysis/report.md).
This motivates comparing physical responses, but does not establish that
matching output displacement at initialization matches hidden-layer signals
or prevents divergence during training. The common target and its training
qualification remain to be decided.

## CIFAR-10: Architecture and 50-Epoch DRN Results

We selected the **L8 architecture**: eight convolutions grouped into blocks
of depths **[3,3,2]**, with logical widths **128/256/512**, max-pooling and
trainable affine batch normalization at each block boundary. Its digital
feed-forward reference reached **92.62% best / 92.57% final test accuracy**
after fifty epochs. That exploratory run monitored the test set each epoch;
it is not a held-out architecture-selection result. See the
[architecture decision](cifar_digital_l8_plan_20260921.md) and
[feed-forward results](../results/cifar10-digital-l8-wide-affinebn-adam-seed0-20260921-v1/analysis/report.md).

The September 22 DRN continuations brought all three schemes to **50 epochs**,
using analog convolutions and an analog classifier, with digital pooling and
BN between blocks. The seed-0 runs use Adam, batch 32 and a 45,000/5,000
training/validation split:

| Scheme | Epoch-50 validation accuracy |
|---|---:|
| Baseline | 89.48% |
| Ours | 89.92% |
| Legacy | 88.72% |

The **very surprising observation is that legacy is not superior**: it finishes
below both baseline and ours in these runs. These are single-seed validation
results across different GPU/software environments; they do not establish a
general ranking. The DRN runs did not read the official test set, so their
accuracies are not a matched comparison with the digital reference above.
See the [completed fifty-epoch comparison and curves](../results/cifar10-l8-analog-baseline-legacy-e30-e50-seed0-20260922-v1/analysis/report.md).

Our working hypothesis is that **batch normalization may change or cancel
the benefit of legacy's voltage scaling**. The follow-up now includes the
completed fifty-epoch normalized-legacy control: **88.82% validation**, versus
**88.72%** for raw legacy. BN epsilon interacts with learning rate, but this
correction alone does not close the final gap. Trainable affine BN helps all
three schemes at their selected rates; recalibrating running statistics also
does not repair legacy's final accuracy gap.

The [consolidated CIFAR evidence summary](cifar_experiment_review_20260924.md)
covers the digital references, architecture, batch-size and LR searches,
fifty-epoch continuations, amplification/KCL checks, BN and epsilon controls,
gradient/Adam diagnostics, learned gains and late-training stagnation. It
includes measured results, failed and excluded cases, conclusions, limitations,
and links to the underlying reports and curves. The cause of legacy's final
disadvantage remains unresolved in this single-seed evidence.

## Detailed Records

Historical experiment conclusions and evidence are in the
[experimental manifest](experimental_manifest.md); detailed beta diagnostics
are in the [beta study](beta_study.md). Compute inventory and placement
guidance are maintained in [AGENTS.md](../AGENTS.md).
