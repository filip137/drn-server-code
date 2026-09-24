# Trained Conv3 gradient magnitudes

Analyzed 2026-09-18 from the existing, validated Section 4.3 replay. No new
simulation, optimizer step, or official-test read was needed.

Legacy does not have uniformly smaller clean gradients: its first two weight
layers and readout have larger magnitudes than baseline and balanced. The
stronger distinction is how much endpoint read noise inflates those gradients,
especially in the early layers. These measurements describe the selected
checkpoints and operating points; they do not independently explain accuracy.

[Matplotlib figure, JPG](figures/section43_conv3_gradient_magnitudes.jpg) ·
[PDF](figures/section43_conv3_gradient_magnitudes.pdf) ·
[PNG](figures/section43_conv3_gradient_magnitudes.png) ·
[CSV](section43_conv3_gradient_magnitudes.csv) ·
[Provenance](section43_conv3_gradient_magnitudes_provenance.json).

## Clean gradients

The table gives mean per-weight gradient RMS, computed as
`||g||_2 / sqrt(number of weights)` for each minibatch and then averaged.
This removes the effect of different tensor sizes from the layer comparison.
Here Conv 1/2/3 are the three weight layers within the Conv3 architecture.

| Weight layer | Number of weights | Baseline | Balanced | Legacy |
|---|---:|---:|---:|---:|
| Conv 1 | 1,152 | 4.451e-5 | 5.936e-5 | 3.270e-4 |
| Conv 2 | 73,728 | 3.168e-5 | 3.547e-5 | 1.035e-4 |
| Conv 3 | 294,912 | 1.810e-4 | 7.267e-5 | 1.435e-4 |
| Readout | 250,880 | 6.572e-5 | 1.518e-4 | 1.065e-3 |

The corresponding mean whole-tensor L2 norms are:

| Weight layer | Baseline | Balanced | Legacy |
|---|---:|---:|---:|
| Conv 1 | 0.001511 | 0.002015 | 0.01110 |
| Conv 2 | 0.008602 | 0.009632 | 0.02811 |
| Conv 3 | 0.09831 | 0.03947 | 0.07794 |
| Readout | 0.03292 | 0.07601 | 0.53365 |

These are centered EqProp gradients at zero read noise. Matched noiseless
finite-K BPTT mean norms agree within 0.108% in every layer/scheme. The CSV
includes both estimators' means, medians, p10, p90, and sample standard
deviations. The plotted whiskers are p10–p90 across minibatches, not confidence
intervals across training seeds. The reported value is a mean of individual
gradient norms, not the norm of the averaged gradient.

## Effect of read noise

At endpoint noise sigma=5e-4, the mean ratio of each noisy EP gradient norm to
its paired clean EP gradient norm is:

| Weight layer | Baseline | Balanced | Legacy |
|---|---:|---:|---:|
| Conv 1 | 402.2× | 1,005.5× | 627,986.8× |
| Conv 2 | 159.4× | 441.1× | 157,529.9× |
| Conv 3 | 10.48× | 6.32× | 1,289.66× |
| Readout | 1.0414× | 1.0002× | 1.0159× |

These are means of paired ratios, not ratios of the preceding table's means.
The CSV retains all six noise levels. A large noisy gradient norm is not a
large useful learning signal: the companion cosine measurements show severe
directional corruption in the early layers. The readout norm changes much less.

The earlier small voltage-displacement measurements do not imply equally small
parameter gradients: the centered EP estimator divides its energy-gradient
difference by `2 × base_beta × output_row_current_scale`. Injected beta is 10, 3,
and 0.001 for baseline, balanced, and legacy, respectively. Differing beta,
amplification, learned weights, and operating points prevent attributing all
cross-scheme differences to amplification alone. These raw gradients also
do not directly measure Adam updates or their cumulative effect on accuracy.

## Sources and checks

- The same 576 ordinary-MNIST validation examples, in 36 batches of 16,
  used for the original cosine figure. Cohort identity and hashes are in
  [the provenance](section43_conv3_gradient_magnitudes_provenance.json) and
  [the source replay](../results/section43-eqprop-mechanism-20260918-v1/full-1/).
- Clean training seed-0 maximum-validation checkpoints: baseline epoch 30,
  balanced epoch 23, legacy epoch 23. Paths and hashes are preserved in
  [the checkpoint table](section43_checkpoint_sources.csv). These are fixed
  clean checkpoints, not separately noise-trained checkpoints.
- Float64, perfect diodes, T=K=8, centered frozen-current EqProp, matched
  finite-K BPTT, fixed zero biases, conductance bounds [0,100]. Eight paired
  draws at each nonzero noise level; 36 clean or 288 noisy comparisons per cell.
- Canonical source run validates, all three checkpoint NPZ hashes match,
  weight shapes match across schemes, and all 17,712 Conv3 raw comparisons
  have complete, unique batch/draw coverage. All 72 reaggregated cells agree
  with the earlier summary for both estimators' norm statistics.
- The original finite-T caveat persists: baseline Conv3 has 72/144 failing
  post-T batch-layer residual-p90 checks at threshold 0.01; subsequent phase
  endpoints pass. See [the original report](section43_mechanism.md).
- Exact-zero fractions are retained in the CSV. Raw gradient coordinates were
  not saved, so near-zero fractions at a new threshold cannot be reconstructed
  from these norms. This analysis is limited to trained checkpoints.

Regenerate the CSV and PDF/JPG/PNG/SVG exports from this directory:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/section43-matplotlib /home/filip/miniconda3/envs/py312/bin/python plot_section43_conv3_gradient_magnitudes.py
```
