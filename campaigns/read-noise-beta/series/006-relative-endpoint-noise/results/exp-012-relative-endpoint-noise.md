---
experiment: "exp-012"
evidence: "imported-summary"
summary: "Relative endpoint noise reduces legacy third-convolution alignment advantage in all 12 paired comparisons per competitor; ours leads all four relative-noise cells. At output D/F=2, ours has 4.04x legacy's hidden D/P and 3.95x lower relative gradient error at eta=1e-4, consistent with a fractional-signal explanation. Cross-device BPTT equivalence remains unchecked by explicit user waiver."
verdicts: {"H-009": "supports"}
---
# Exploratory endpoint-noise comparison: exp012

All six production contexts completed successfully and are collected locally:
three schemes at each output D/F target 2 and 6, 36 batches of 16 examples,
three paired noise draws at each of two levels for both acquisition models.
The analysis includes 11,232 finite layer/batch/draw rows and 120 layer cells;
there are no missing production contexts. Canonical/source revalidation was
waived by Filip; the collection receipts confirm terminal exit0 and coverage.

The predeclared third-convolution pattern supports H-009 within this exploratory
grid: legacy-minus-competitor alignment advantage decreases under relative
noise in all 12 paired target/level/draw comparisons for both baseline and ours,
against both BPTT and each scheme's clean EqProp reference. BPTT gap reductions
range from +0.074437 to +0.519310 versus baseline and +0.253338 to +0.670011
versus ours. Legacy leads all four absolute-noise cells; ours leads all four
relative-noise cells in this layer.

| Output D/F | Noise level | Absolute BPTT cosine, baseline / ours / legacy | Relative BPTT cosine, baseline / ours / legacy |
|---|---:|---|---|
| 2 | 0.0001 | 0.1718 / 0.3738 / 0.6826 | 0.5892 / 0.9412 / 0.5838 |
| 2 | 0.001 | 0.0198 / 0.0413 / 0.0945 | 0.0741 / 0.2758 / 0.0730 |
| 6 | 0.0001 | 0.4594 / 0.7681 / 0.9411 | 0.9067 / 0.9926 / 0.9045 |
| 6 | 0.001 | 0.0541 / 0.1209 / 0.2716 | 0.2163 / 0.6472 / 0.2133 |

These are means over the fixed cohort and three noise draws, not model seeds.
All layers and both reference metrics remain in the linked CSV/plots. This
supports the predicted pattern, not a causal separation of voltage scale from
noise severity/covariance. Equal numerical absolute sigma and relative eta
have different units and do not match noise power. There is one initializer,
one cohort and no training/official-test result.

## Interpretation: fractional hidden-layer response

This mechanism interpretation was developed after observing exp012, on
September 25. Output D/F matching leaves hidden-layer fractional displacement
unmatched. Define D = RMS((v_plus-v_minus)/2) and P as the pooled RMS of the
two nudged endpoints. With independent relative endpoint errors
`eta * abs(v) * xi`, the centered voltage-contrast noise has RMS scale
`eta * P / sqrt(2)`, giving a signal-to-noise scale of
`sqrt(2) * D / (eta * P)`. Gradient alignment also depends on adjacent-layer
states and spatial structure, so D/P alone is not a formula for cosine.

At output D/F=2 and eta=1e-4, all three schemes were measured on the local GPU:

| Scheme | Third-convolution D/P | Mean noisy-minus-clean gradient norm / clean norm | Mean cosine to BPTT |
|---|---:|---:|---:|
| Baseline | 0.0550% | 1.387 | 0.5892 |
| Ours | 0.2222% | 0.356 | 0.9412 |
| Legacy | 0.0550% | 1.407 | 0.5838 |

Ours has 4.04 times legacy's fractional displacement and a relative gradient
error smaller by a factor of 3.95. Clean third-convolution EqProp/BPTT cosine
exceeds 0.9999 for every scheme. Together, these measurements support better
gradient preservation under relative read noise as the explanation of ours'
advantage. See [physical-state controls](../../../../../results/conv3-relative-endpoint-noise-20260925-v1/analysis/physical_state_controls.csv)
and [layer summaries](../../../../../results/conv3-relative-endpoint-noise-20260925-v1/analysis/layer_summary.csv).

Legacy's free-state voltages are approximately 1, 4, 16 and 64 times baseline's
across the three convolutions and readout. Their displacements scale nearly
identically, leaving hidden D/P almost unchanged. This is consistent with their
nearly overlapping relative-noise curves: the noise scales with the voltage,
removing the benefit of a simple voltage rescaling present under absolute noise.

The amplification equations suggest why ours differs. Removing successive
voltage gains from the state coordinates leaves interaction weights involving
voltage_amp * current_amp: 1 for baseline and legacy, 4 for ours. Ours changes
normalized coupling and loading between layers. Its larger fractional response
is measured; attributing the exact response ratio to this coupling remains a
mechanistic inference, not an independently isolated causal result.

The scoped H-009 verdict remains support for the predicted exploratory pattern.
The first two convolutions remain strongly noise-dominated; a third-layer
alignment advantage does not establish better whole-network training. These
are MNIST initialization measurements, not CIFAR evidence.

## Next decision

Propose a focused initialization replay matching third-convolution D/P across
schemes at fixed eta. If fractional signal is the dominant explanation, ours'
alignment advantage should shrink. Compare against both clean EqProp and BPTT,
retain clean-gradient controls for beta-dependent error, and measure adjacent
layers' D/P because matching one layer does not match the complete gradient-noise
geometry. This follow-up is proposed, not launched; the present evidence does
not justify a training-performance claim. The idea is indexed in
[ideas.md](../../../ideas.md#why-ours-outperforms-legacy-under-relative-read-noise).

## Execution limitations and provenance

Target2 used local RTX3090 source-v4 with exact old-screen reference gates.
Target6 used Akib RTX3080 source-v5 after an explicit user waiver of additional
checks: native CUDA BPTT remained computed, but cross-device equivalence to the
old RTX3090 BPTT reference is unchecked. The inherited top-level summary flag
`screen_reference_hashes_matched=true` is not proof of BPTT equality; per-run
facts and this waiver are authoritative. Physical states and residual-reuse
identity guards remained active. Do not infer same-device cross-target parity.

Three failed local memory smokes and one failed Akib reference smoke are
preserved separately from the six successful production cases. Production
wall time was 935.230 seconds local and 496.380 seconds Akib; all preparation
cost 97.077 seconds, totaling approximately 0.42464 GPU-hours, below the 1-hour
cap. The collector's SSH route was corrected only inside its study script to
use the known explicit SSH config; no jobs were rerun or unrelated jobs altered.

Authoritative artifacts are under
`results/conv3-relative-endpoint-noise-20260925-v1/local/` and
`results/conv3-relative-endpoint-noise-20260925-v1/collected/akib/akib/`.
See the [analysis report](../../../../../results/conv3-relative-endpoint-noise-20260925-v1/analysis/report.md),
[BPTT plots](../../../../../results/conv3-relative-endpoint-noise-20260925-v1/analysis/cosine.png),
and [paired comparisons](../../../../../results/conv3-relative-endpoint-noise-20260925-v1/analysis/paired_advantage_reduction.csv).
No additional simulations are authorized or needed for this closeout.
