---
experiment: "exp-013"
evidence: "validated-local"
summary: "Both stages validated: 177876 comparisons; all six fresh-noise convolution patches pass the predeclared rule, supporting H-010 for all three architectures."
verdicts: {"H-010": "supports"}
---
# Relative-noise advantage map: completed review

## Preserved Stage A screen (before refinement)

The initialization screen completed on all three architectures on September 25.
All 63 scheme/displacement contexts are collected locally: 102,060 layer
comparisons with no undefined alignment values. Output displacement matching
uses 576 examples; noisy estimates use 192 examples and three paired draws.
One global relative coefficient eta applies to every noninput layer, including
the readout. All architectures used the same eta/displacement grid.

These are the sampled maxima of mean ours-minus-legacy gradient cosine with
respect to clean finite-K BPTT, selected separately for each convolutional layer:

| Architecture | Convolution | Output D/F | eta | Ours | Legacy | Gap |
|---|---:|---:|---:|---:|---:|---:|
| Conv1 | 1 | 2 | 1e-3 | 0.7378 | 0.3205 | +0.4173 |
| Conv2 | 1 | 4 | 1e-5 | 0.8348 | 0.2854 | +0.5494 |
| Conv2 | 2 | 2 | 3e-5 | 0.7589 | 0.3294 | +0.4295 |
| Conv3 | 1 | 2 | 3e-8 | 0.6423 | 0.3750 | +0.2673 |
| Conv3 | 2 | 1 | 1e-8 | 0.6633 | 0.5349 | +0.1284 |
| Conv3 | 3 | 1 | 1e-4 | 0.8136 | 0.3397 | +0.4739 |

Ours also exceeds baseline at each of these selected points. The useful noise
scale differs greatly across layers, particularly in Conv3: its last-layer
peak at eta=1e-4 does not establish useful alignment in earlier layers at that
same noise level. For the minimum layer cosine within each scheme, the largest
sampled ours-minus-legacy gap in Conv3 is +0.1463 at D/F=3, eta=3e-8
(ours 0.6627, legacy 0.5164, baseline 0.5569). Minima may occur in different
layers for different schemes.

Concretely, at Conv3 D/F=1 and eta=1e-4, ours has convolutional cosines
-0.0196, 0.0075 and 0.8136. Its third-layer advantage coexists with essentially
unaligned early-layer gradients. In Conv2, the peak first-layer point D/F=4,
eta=1e-5 retains second-layer alignment of 0.9893 for ours versus 0.8979
for legacy. Layer-specific peaks must therefore be interpreted alongside the
other layers at the same global noise coefficient.

This screen identifies candidate regions, not confirmed optima. The frozen
refinement contains 24 common coordinates for all three architectures and
schemes, with the full 576-example cohort and three fresh paired noise draws.
H-010's neighborhood criterion remains pending that refinement. There is one
initializer and no training or accuracy evidence; qualification and cross-device
BPTT equivalence were not checked under the user's explicit waiver.

[Screen report and plots](../../../../../results/conv123-relative-noise-advantage-map-20260925-v1/analysis/stage_a/report.md)
provide all layers, baseline comparisons, coverage and the frozen selection.
Operational state remains in the [experiment note](../experiments/exp-013-relative-noise-advantage-map.md).

## Completed Stage B and scoped review

Stage B collected all 54 contexts (18 per architecture), 24 frozen common coordinates for every scheme and depth, and 75,816 layer comparisons: 69,984 noisy rows plus 5,832 clean controls. Stage A retains 63 contexts and 102,060 comparisons. Coverage, acquisition seeds, declared artifacts and terminal criteria passed; no undefined alignment values or nonfinite numeric CSV fields were found. No source-hash or cross-device parity qualification was added. All six launcher exits were zero; no cases were excluded.

H-010 **supports** separately for Conv1, Conv2 and Conv3, and hence for the full predeclared claim. Each of the six selected convolution patches has 12/12 positive paired ours-minus-legacy differences (four cells times three fresh draws). This exceeds the rule requiring at least one supported patch per architecture.

Stage B sampled layer maxima, including readout (each row can have a different global operating point):

| Architecture | Layer | D/F | eta | Ours | Legacy | Baseline | Gap [draw min, max] |
|---|---|---:|---:|---:|---:|---:|---:|
| conv1 | ConvWeight_0 | 2 | 0.001 | 0.74329 | 0.3188 | 0.29415 | +0.42449 [+0.40718, +0.43424] |
| conv1 | DenseWeight_0 | 0.5 | 1e-08 | 1 | 1 | 1 | -1.1793e-11 [-1.1835e-11, -1.1769e-11] |
| conv2 | ConvWeight_0 | 3.5 | 1e-05 | 0.82662 | 0.27344 | 0.30603 | +0.55317 [+0.53688, +0.57139] |
| conv2 | ConvWeight_1 | 2 | 3e-05 | 0.77012 | 0.32971 | 0.33193 | +0.44041 [+0.43513, +0.44531] |
| conv2 | DenseWeight_0 | 2 | 0.0017321 | 0.99993 | 0.99992 | 0.99992 | +6.2908e-06 [+6.0504e-06, +6.5212e-06] |
| conv3 | ConvWeight_0 | 2.5 | 3e-08 | 0.69194 | 0.42955 | 0.47305 | +0.26239 [+0.26047, +0.26578] |
| conv3 | ConvWeight_1 | 1 | 1e-08 | 0.66271 | 0.52339 | 0.54182 | +0.13932 [+0.13229, +0.14337] |
| conv3 | ConvWeight_2 | 1 | 0.0001 | 0.81391 | 0.34179 | 0.34618 | +0.47212 [+0.46977, +0.47379] |
| conv3 | DenseWeight_0 | 2 | 0.0017321 | 0.99999 | 0.99999 | 0.99999 | +7.4592e-07 [+7.3176e-07, +7.5654e-07] |

Ours exceeds baseline at all six sampled convolutional peaks. Readout gaps are negligible near the cosine ceiling; Conv1 readout has a tiny negative peak gap. A positive convolutional margin does not establish dominance of every layer. Stage A uses 192 noisy examples, whereas Stage B uses all 576 with fresh draws; do not pool these estimates or interpret their difference as improvement over time.

For W = minimum mean cosine across all layers including readout, Stage B sampled maximum gaps are +0.42449 (Conv1, D/F=2, eta=1e-3), +0.55317 (Conv2, D/F=3.5, eta=1e-5), and +0.17917 (Conv3, D/F=2, eta=1.73205e-8; W ours=0.71277, legacy=0.53361, baseline=0.57953). The Conv3 Stage A W maximum at D/F=3, eta=3e-8 was not in the frozen union, so that particular screen maximum remains unconfirmed. H-010 tests convolutional patches, not a separate W criterion.

The eta/(D/F) scatter is descriptive and does not establish a universal ridge. Sampled maxima and the 95%-of-positive-peak band are conditional on the searched coordinates, not confidence intervals or global optima; edge directions remain unresolved. One initialization, three acquisition draws, one validation cohort, fixed finite-K references and unchecked cross-GPU equivalence limit the interpretation. No training benefit, official-test accuracy, causal isolation or all-layer dominance follows.

Decision: close the authorized two-stage study after this scoped review; no additional simulations are authorized or required. Preserve the historical blocked Stage B admission evidence and the accounting correction. No scientific retuning or further launch occurred during closeout.

[Final report and maps](../../../../../results/conv123-relative-noise-advantage-map-20260925-v1/analysis/stage_b/report.md) · [Coverage validation](../../../../../results/conv123-relative-noise-advantage-map-20260925-v1/monitor/completion-validation.json) · [Physical state controls](../../../../../results/conv123-relative-noise-advantage-map-20260925-v1/analysis/stage_b/physical_state_controls.csv) · [Noise scale controls](../../../../../results/conv123-relative-noise-advantage-map-20260925-v1/analysis/stage_b/noise_scale_controls.csv). All layers, clean/noisy reference comparisons, gradient errors, draw ranges and near-peak tables are retained in the linked analysis directory.

[Cosine-versus-displacement curves](../../../../../results/conv123-relative-noise-advantage-map-20260925-v1/analysis/displacement_curves/README.md) show every sampled noise level and layer for all three schemes, with screen and refinement distinguished. Available x-axes are measured output D/F, absolute output RMS displacement, and each layer's RMS displacement. Three-draw ranges are descriptive; stages are not pooled. Reproduce with `python -m experiments.plot_relative_noise_displacement_curves --study-root results/conv123-relative-noise-advantage-map-20260925-v1`.

## Layerwise absolute and fractional displacement — September 28

[JPG comparison](figures/exp013-layer-displacement/conv123_absolute_and_fractional_displacement.jpg)
shows absolute D and fractional D/P against layer depth for all three schemes.
Conv1/2/3 use matched initial output D/F targets 1/4/6, respectively. The 27
physical-state rows come from stage A's 576-example clean-state cohort, with no
new simulation or pooling with stage B. Both axes are logarithmic. The
[figure bundle](figures/exp013-layer-displacement/README.md) contains individual
architecture JPGs, plotted measurements and a reproducible command.

At these initialization settings, ours has larger D/P in every convolutional
layer than legacy. For Conv3, the ratios are approximately 2.06, 5.18 and 4.03
for convolutions 1, 2 and 3. Legacy has larger absolute D than ours in those
three layers, while baseline and legacy nearly overlap in D/P. This supports
distinguishing absolute voltage response from fractional signal under relative
read noise. It does not establish gradient alignment or explain trained
accuracy on its own; those require the separate gradient and training evidence.

## Gradient-quality companion figures — September 28

[Cosine to BPTT across layers](figures/exp013-gradient-quality/cosine_to_bptt_across_layers.jpg)
uses the same initial output D/F targets 1/4/6 as the displacement figure, at
eta=1e-6, 1e-5, 1e-4 and 3e-4, as requested for the revised figure. The
[clean-EqProp reference](figures/exp013-gradient-quality/cosine_to_clean_ep_across_layers.jpg)
separates noise-induced direction changes from clean EqProp/BPTT disagreement.
[Conv1](figures/exp013-gradient-quality/conv1_gradient_quality_vs_noise.jpg),
[Conv2](figures/exp013-gradient-quality/conv2_gradient_quality_vs_noise.jpg) and
[Conv3](figures/exp013-gradient-quality/conv3_gradient_quality_vs_noise.jpg)
noise sweeps show every layer's BPTT cosine and
`||g_noisy_EP - g_clean_EP|| / ||g_clean_EP||` over all 14 nonzero noise levels.
Cosine-only versions for
[Conv1](figures/exp013-gradient-quality/conv1_cosine_to_bptt_vs_noise.jpg),
[Conv2](figures/exp013-gradient-quality/conv2_cosine_to_bptt_vs_noise.jpg) and
[Conv3](figures/exp013-gradient-quality/conv3_cosine_to_bptt_vs_noise.jpg)
show each architecture's BPTT-alignment panels without the gradient-error row.

These plots reuse 405 stage-A summary cells, including 27 unplotted zero-noise
controls. Noisy measurements use 192 examples and three acquisition draws;
shading spans draw means, not model-seed confidence intervals. The clean
controls and the displacement figure use 576 examples and are not pooled with
the noisy cohort. No new simulation was run. A
[small source CSV and reproduction command](figures/exp013-gradient-quality/README.md)
are retained with the JPGs.

The Conv3 noise sweep exposes a large separation between layers: the first two
convolutions lose useful alignment at much lower eta than the third convolution,
while readout alignment stays near one. Ours delays the third-convolution
degradation relative to legacy, but this initialization advantage cannot by
itself establish better gradients throughout training or explain the final
accuracy ranking. Existing seed/cohort and finite-K reference limitations apply.
