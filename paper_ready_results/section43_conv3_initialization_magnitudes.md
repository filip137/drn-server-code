# Conv3 gradient magnitudes at initialization

Completed 2026-09-18. Outcome: **legacy has the largest clean gradients in
every layer already at initialization**. Its gradients then decrease in all
four layers at the selected trained checkpoint. Baseline instead has large
increases in its second and third convolutional layers.

[Matplotlib JPG](figures/section43_conv3_initialization_magnitudes.jpg) ·
[PDF](figures/section43_conv3_initialization_magnitudes.pdf) ·
[PNG](figures/section43_conv3_initialization_magnitudes.png) ·
[CSV](section43_conv3_initialization_magnitudes.csv) ·
[Provenance](section43_conv3_initialization_magnitudes_provenance.json).

## Initialization measurements

Mean clean centered-EqProp gradient RMS per weight, computed separately for
each of 36 matched minibatches as `||g||_2 / sqrt(N)`, then averaged:

| Layer within Conv3 | Baseline | Balanced | Legacy |
|---|---:|---:|---:|
| Conv 1 | 4.372e-5 | 2.753e-4 | 2.429e-3 |
| Conv 2 | 2.806e-6 | 1.864e-5 | 1.703e-4 |
| Conv 3 | 2.625e-6 | 1.746e-5 | 1.650e-4 |
| Readout | 9.464e-5 | 6.328e-4 | 5.914e-3 |

Mean whole-tensor L2 norms:

| Layer within Conv3 | Baseline | Balanced | Legacy |
|---|---:|---:|---:|
| Conv 1 | 0.001484 | 0.009343 | 0.08246 |
| Conv 2 | 0.0007620 | 0.005061 | 0.04625 |
| Conv 3 | 0.001425 | 0.009483 | 0.08963 |
| Readout | 0.04741 | 0.31697 | 2.96202 |

Legacy is approximately 56–63 times baseline across these layers. Balanced
is approximately 6.3–6.7 times baseline. Matched noiseless finite-K BPTT mean
norms agree with clean EqProp within 0.856% in every initialization cell.
The CSV contains both estimators, medians, p10/p90, sample standard deviations,
exact-zero fractions, and fractions with `abs(g) <= 1e-12`. The largest
initialization cell mean near-zero fraction is 2.326e-6 (0.000233%).

## Changes by the selected trained checkpoint

Ratios of the trained mean norm to the initialization mean norm:

| Layer within Conv3 | Baseline | Balanced | Legacy |
|---|---:|---:|---:|
| Conv 1 | 1.018 | 0.2156 | 0.1346 |
| Conv 2 | 11.289 | 1.903 | 0.6078 |
| Conv 3 | 68.966 | 4.162 | 0.8696 |
| Readout | 0.6944 | 0.2398 | 0.1802 |

Thus the first-layer mean gradient barely changes for baseline, but becomes
about 4.6 times smaller for balanced and 7.4 times smaller for legacy. Legacy
still has a larger clean first-layer gradient than the other schemes when
trained. Small raw gradient magnitude alone does not explain its poor noisy
alignment. These are raw gradients, not Adam updates or direct measurements
of their effect on accuracy.

The figure's first two panels share a vertical scale and show means with
p10–p90 ranges. Its third panel, and the CSV's `norm_relative_to_initial`
columns, summarize **individual paired minibatch ratios**, rather than the
ratios of means in the table above. For example, baseline Conv 1 has a ratio
of means 1.018 but a mean paired ratio 1.219. Both are retained explicitly.
Percentile ranges describe minibatch variation, not uncertainty across seeds.

## Matched contract and limitations

- Epoch 0 uses the exact common saved seed-0 wide-Kaiming initializer:
  `../results/paper-training-completion-20260911-v1/assets/wide_kaiming/conv3/seed0.pt`.
  SHA-256: `5e5782bd9bf166b8a432cbf283d745ffe4d25fe4a69ed656f14f9e7e9407392f`.
  Float32 saved weights are promoted exactly to float64 as in training; the
  initialization fingerprint agrees with all three original training records.
- The same 576 ordinary-MNIST validation examples in 36 ordered batches of
  16, with matching source-index and input-payload hashes, are used throughout.
  Cohort and configuration hashes are recorded in the provenance and bundles.
- Perfect diodes, explicit saved diode parameters, zero frozen biases,
  conductance bounds [0,100], float64, T=K=8, centered frozen-current EqProp
  and matched finite-K BPTT. Injected beta remains 10 / 3 / 0.001. No optimizer
  steps or official-test reads. No noise is added in this magnitude comparison.
- Baseline initialization is reused from the earlier validated replay;
  balanced and legacy each have 36 new initialization evaluations on the same
  local RTX 3090. Trained measurements are reused from the original replay at
  epochs 30 / 23 / 23, with [source paths](section43_checkpoint_sources.csv).
  This compares the available endpoints, not the intervening training trajectory.
- Amplification and beta differ across schemes. Shared initial parameter bytes
  do not imply equal state amplitudes, losses, or parameter gradients. These
  measurements do not isolate one cause of the gradient-scale differences.
- All 1,728 initialization phase/layer residual-p90 checks pass the 0.01
  threshold. The existing trained-baseline free-state residual caveat remains
  attached to the trained comparison; see [the earlier report](section43_mechanism.md).

## Validation and reproducibility

The three-case one-batch smoke exactly reproduces four archived baseline
layer comparisons and all corresponding endpoint hashes. Its eight new
balanced/legacy layer comparisons exactly reproduce in the full run. The full
run covers all 72 new batches, 288 clean EP/BPTT comparisons, and 576 detailed
estimator/layer gradient-statistic rows. Combined with baseline, analysis
covers all 108 initialization batches and 48 scheme/role/estimator/layer cells.

All four relevant replay bundles validate; all input hashes and per-batch
unchanged-parameter guards pass. Reaggregated trained statistics reproduce
the preceding trained-magnitude table. Figure exports were visually checked,
and their hashes verified. No failed attempts or exclusions. Total new GPU
time including smoke is 154.203 seconds (2.570/10 GPU-minutes); the process
exited zero and released the GPU.

- [Canonical new replay](../results/section43-conv3-initialization-magnitudes-20260918-v1/full/)
- [Baseline initialization source](../results/section43-conv3-baseline-initialization-20260918-v1/full/)
- [Trained source](../results/section43-eqprop-mechanism-20260918-v1/full-1/)
- [Frozen config](section43_conv3_initialization_magnitudes_config.json)
- [Replay runner](run_section43_conv3_initialization_magnitudes.py)
- [Matplotlib generator](plot_section43_conv3_initialization_magnitudes.py)

Regenerate without GPU work:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/section43-matplotlib /home/filip/miniconda3/envs/py312/bin/python plot_section43_conv3_initialization_magnitudes.py
```
