# Trained Conv3 voltage-displacement RMS

Checked 2026-09-18 using the same three clean trained checkpoints and 576
validation examples as the gradient-magnitude comparison. No new simulation
or optimizer step was needed. All 2,592 raw displacement records reaggregate
consistently into the 72 scheme/layer/displacement-definition cells; the source
canonical replay and all three checkpoint NPZ hashes validate.

[Matplotlib JPG](figures/section43_conv3_trained_displacement.jpg) ·
[PDF](figures/section43_conv3_trained_displacement.pdf) ·
[PNG](figures/section43_conv3_trained_displacement.png) ·
[CSV](section43_conv3_trained_displacement.csv) ·
[Provenance](section43_conv3_trained_displacement_provenance.json).

## Centered phase contrast

For the centered EqProp estimator, the relevant voltage contrast is
`delta_v = (v_plus - v_minus)/2`. The table reports
`sqrt(sum(delta_v**2)/number_of_state_coordinates)`, pooled over examples,
channels, and spatial positions within each layer. Units are absolute simulator
voltage units. H1/H2/H3 are hidden states within the Conv3 architecture.

| State layer | Baseline | Balanced | Legacy |
|---|---:|---:|---:|
| H1 | 4.073e-7 | 1.577e-7 | 2.919e-10 |
| H2 | 2.119e-6 | 2.455e-6 | 7.508e-9 |
| H3 | 3.971e-5 | 1.889e-4 | 9.570e-7 |
| Output | 1.845e-3 | 8.672e-2 | 8.611e-3 |

Legacy has 540×, 327×, and 197× smaller hidden-state phase contrast than
balanced at H1/H2/H3, respectively. Every scheme has smaller contrast toward
the input. The full positive-minus-negative RMS, without dividing by two,
is exactly twice these values.

## Raw free-to-nudged displacement

For `delta_v = v_plus - v_free`, where free is the state after T=8 iterations:

| State layer | Baseline | Balanced | Legacy |
|---|---:|---:|---:|
| H1 | 1.179e-4 | 1.577e-7 | 4.027e-10 |
| H2 | 9.410e-5 | 2.455e-6 | 7.498e-9 |
| H3 | 6.734e-5 | 1.889e-4 | 9.570e-7 |
| Output | 1.846e-3 | 8.672e-2 | 8.611e-3 |

The baseline free state continues to relax during the K=8 phase iterations.
Its zero-nudge drift RMS is 1.179e-4 / 9.408e-5 / 5.437e-5 / 1.821e-5
for H1/H2/H3/output. Consequently, the large raw H1 movement is dominated
by continued relaxation: it is 289× the centered contrast. Subtracting the
matched zero-nudge endpoint gives H1 RMS 4.074e-7, close to the centered
value 4.073e-7. RMS differences cannot be corrected by subtracting the scalar
RMS values; these quantities were computed from the actual state differences.

The CSV retains both signed nudges relative to free, both relative to the
matched zero-nudge endpoint, centered contrast, and zero-nudge drift. It also
retains signed displacement means, mean absolute displacement, per-batch RMS
medians and p10/p90, and output-normalized RMS.

## Interpretation and scope

Small voltage contrast can coexist with large parameter gradients. The EP
estimator converts energy-gradient differences to a parameter gradient using
the scheme-specific beta and output-current normalization. The
[gradient-magnitude table](section43_conv3_gradient_magnitudes.md) and these
voltage displacements measure different quantities.

These measurements support greater susceptibility of legacy's small hidden
phase signal to imposed read noise at the selected operating points. They do
not isolate amplification from beta, trained weights, or operating-point
differences, and they do not by themselves explain noisy-training accuracy.

Baseline/balanced/legacy use epochs 30/23/23, seed 0, injected beta 10/3/.001,
T=K=8, native float64, perfect diodes, zero frozen biases, and [0,100]
conductance weights. All use the same ordered 36 validation batches of 16.
These are clean trained checkpoints and noiseless endpoint displacements;
the separate noise sweep adds noise only to endpoint reads. Official test
data remain unread. The known baseline post-T free-state residual caveat
remains: 72/144 batch-layer residual-p90 checks fail the 0.01 threshold,
whereas subsequent endpoints pass.

Sources: [checkpoint paths and epochs](section43_checkpoint_sources.csv),
[original replay and qualifications](section43_mechanism.md), and
[raw canonical bundle](../results/section43-eqprop-mechanism-20260918-v1/full-1/).

Regenerate the verified table and PDF/JPG/PNG/SVG figure:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/section43-matplotlib /home/filip/miniconda3/envs/py312/bin/python plot_section43_conv3_trained_displacement.py
```
