---
experiment: "exp-008"
evidence: "validated-local"
summary: "Epoch10/30 support fixed trained RMS vectors baseline[0.02,0.02,0.01] and ours[0.02,0.035,0.005]. Legacy block3 requires checkpoint-dependent targets; epoch50 targets transfer in16/18 intermediate cases."
verdicts: {"H-006": "contradicts"}
---
# Intermediate checkpoints and RMS recommendation

All six original epoch10/30 checkpoint replays completed and are locally validated:
15 targets x8 Conv tensors x8 batches each,720 pooled/5760 batch rows total,
270 calibrated block targets. No undefined cosines, failures or exclusions;
all target errors<3%, calibration/replay RMS agree, and source checkpoint hashes,
weights and BN tensors are unchanged. Same256-image cohort, batch32 and native
T=K=[6,6,4] as epochs0/50. Frozen exp-006 numerical implementation reused unchanged.
All queue jobs/collections complete;1535.5s charged within3600s, GPUs released.

New best sampled **nominal RMS (worst-layer pooled cosine)**:

| Scheme | Epoch | Block1 | Block2 | Block3 |
|---|---:|---:|---:|---:|
| Baseline | 10 | 0.01 (0.9926) | 0.02 (0.9917) | 0.0025 (0.9973) |
| Baseline | 30 | 0.02 (0.9937) | 0.02 (0.9814) | 0.01 (0.9931) |
| Ours | 10 | 0.02 (0.9932) | 0.035 (0.9819) | 0.001 (0.9951) |
| Ours | 30 | 0.02 (0.9870) | 0.035 (0.9710) | 0.005 (0.9919) |
| Legacy | 10 | 0.01 (0.9971) | 0.005 (0.9963) | 0.0005 (0.9907)* |
| Legacy | 30 | 0.01 (0.9758) | 0.01 (0.9890) | 0.005 (0.9912) |

*Legacy epoch10 block3 peaks at the lower search boundary. Its true optimum may
be smaller;0.0005 is a measured candidate, not a bracketed optimum.

Choose a fixed target per block by maximizing the worst pooled score across
epochs10/30/50. Recommended first candidates for noiseless EqProp learning tests:

| Scheme | RMS blocks1/2/3 | Worst pooled cosine across trained checkpoints | Worst minibatch cosine |
|---|---|---|---|
| Baseline | [0.02,0.02,0.01] | [0.9796,0.9727,0.9891] | [0.9131,0.8190,0.9424] |
| Ours | [0.02,0.035,0.005] | [0.9729,0.9498,0.9883] | [0.8824,0.9002,0.9605] |

Both vectors stay within0.01 pooled cosine of the sampled peak at every trained
checkpoint. All baseline targets are directly supported at initialization too.
Ours block1/2 initialization cosine is0.9993/0.9566 at the recommended targets.
Ours block3 R0.005 was not measured exactly at initialization; neighboring0.004
and0.007 give0.9991/0.9995. This bracket supports it as a candidate without certifying
the missing point. Restricting mechanically to targets sampled at all four epochs
would choose0.01 for ours block3, reducing its epoch10 cosine to0.9680 solely to
avoid that grid gap; this is not a reason to prefer0.01 scientifically.

Legacy: block1 can use0.01. Block2's best fixed **trained** target is0.0025,
giving worst pooled cosine0.9722, but it is poorly supported at initialization:
neighboring0.002/0.004 give0.431/0.584. No tested fixed block2 target covering all
four checkpoints has worst cosine above0.847. Block3's best fixed trained target
is0.001, with worst cosine only0.902, versus0.984–0.991 with checkpoint-specific
targets. Therefore prefer stage-specific calibration for legacy blocks2/3:

| Checkpoint | Block2 best sampled RMS | Block3 best sampled RMS |
|---|---:|---:|
| Initialization | 0.065 | 0.014 |
| Epoch10 | 0.005 | 0.0005* |
| Epoch30 | 0.01 | 0.005 |
| Epoch50 | 0.0025 | 0.01 |

These are calibration anchors, not a validated interpolation schedule. The large
exception is legacy block3 at epoch10: using epoch50's R0.01 gives cosine0.1786,
compared with0.9907 at0.0005. Its optimal scale is not monotonically decreasing.

Review: H-006's universal transfer claim is contradicted, although16/18 cases pass
the0.01-regret rule. Failures are legacy block2 at epoch30 (regret0.0168) and legacy
block3 at epoch10 (regret0.8120). Baseline and ours pass all intermediate cases.
Depth alone and a single commonR remain inadequate explanations of alignment.

Decision: use the baseline/ours vectors above as initial training candidates;
use checkpoint-specific legacy targets. These are **RMS targets**, not fixed
betas: recalibrate injectedB as the state changes. Betas vary greatly even whenR
is held fixed; exact values are retained in `analysis/comparison.json`/`points.csv`.
Single-seed, noiseless, pooled-gradient evidence does not establish uniformly good
minibatch directions or EqProp learning performance. No training was launched.

[RMS/cosine curves across epochs](../../../../../results/cifar-block-rms-intermediate-20260929-v1/analysis/rms_across_epochs.jpg)
and [movement of sampled peaks](../../../../../results/cifar-block-rms-intermediate-20260929-v1/analysis/peak_rms_across_epochs.jpg).
Evidence root: `results/cifar-block-rms-intermediate-20260929-v1/`; source paths/hashes
in `analysis/checkpoint_inventory.json`, raw/validated bundles under `e10/` and `e30/`.
Epoch0/50 evidence is reused from exp-005/006, not rerun or pooled across epochs.

## Per-layer plots and gradient magnitudes, September 30

At Filip's request, replot all eight Conv tensors independently at epochs0/10/30/50.
This is CPU postprocessing of the same evidence: no new replay or training.
All1,848 plotted layer measurements match the earlier comparison exactly, cover
96 scheme/epoch/layer groups and retain eight minibatches each. Reference norms
are unchanged across beta within each checkpoint/layer. Reproduce with
`python experiments/plot_cifar_layer_rms_across_epochs.py`.

The original block score is the **minimum of separately computed layer cosines**,
not the cosine of concatenated layer gradients. Thus a large-norm layer cannot
dominate that score. Each layer's gradient vectors are still averaged across
the256 examples before cosine/norm calculation; batch minima and medians are
retained in the new `analysis/layer_curves/plotted_values.csv`.
The x-axis remains **block-output** relative centered RMS. These older sweeps
did not measure each internal layer's own displacement.

| Scheme | Layer cosines, full range | High-cosine zoom | EqProp/BPTT norm ratios |
|---|---|---|---|
| Baseline | [JPG](../../../../../results/cifar-block-rms-intermediate-20260929-v1/analysis/layer_curves/baseline_cosine.jpg) | [JPG](../../../../../results/cifar-block-rms-intermediate-20260929-v1/analysis/layer_curves/baseline_cosine_zoom.jpg) | [JPG](../../../../../results/cifar-block-rms-intermediate-20260929-v1/analysis/layer_curves/baseline_norm_ratio.jpg) |
| Ours | [JPG](../../../../../results/cifar-block-rms-intermediate-20260929-v1/analysis/layer_curves/ours_cosine.jpg) | [JPG](../../../../../results/cifar-block-rms-intermediate-20260929-v1/analysis/layer_curves/ours_cosine_zoom.jpg) | [JPG](../../../../../results/cifar-block-rms-intermediate-20260929-v1/analysis/layer_curves/ours_norm_ratio.jpg) |
| Legacy | [JPG](../../../../../results/cifar-block-rms-intermediate-20260929-v1/analysis/layer_curves/legacy_cosine.jpg) | [JPG](../../../../../results/cifar-block-rms-intermediate-20260929-v1/analysis/layer_curves/legacy_cosine_zoom.jpg) | [JPG](../../../../../results/cifar-block-rms-intermediate-20260929-v1/analysis/layer_curves/legacy_norm_ratio.jpg) |

Initialization often has a wider high-cosine plateau, particularly blocks1/3 and
the last convolution of block2. Within the shared sampled domain R<=0.41, its
widest connected sampled cosine>=0.95 span exceeds epoch50's in21/24 layers
(baseline7/8, ours8/8, legacy6/8). Unequal sampling grids and boundary-truncated
spans prevent treating these as exact continuous widths. This is not universal
across training stages: ours block2 Conv1 has a wider passing span at epoch10
than at initialization. At epoch50 its three layers prefer R≈0.0841,0.0200 and
0.000499 respectively; the last maximum lies at the lower search boundary.

[BPTT gradient magnitudes (JPG)](../../../../../results/cifar-block-rms-intermediate-20260929-v1/analysis/layer_curves/bptt_gradient_magnitudes.jpg)
show epoch0/50 norm reductions of109–2,470× for baseline,444–7,788× for ours and
35–461× for legacy. This is not solely cancellation from averaging batches:
the corresponding mean minibatch norms also fall, by72–2,507×,424–14,722× and
29–1,093×. Positive scalar rescaling alone leaves cosine unchanged; smaller
gradient scales could increase relative estimation error, but this reanalysis
does not establish that mechanism. Direction and magnitude also differ:
ours epoch50 block1 Conv1 at R≈0.02 has cosine0.9729 but norm ratio0.8107.

Numeric evidence, source mapping, sampled passing spans and reference scales are
in `analysis/layer_curves/summary.json` and `plotted_values.csv` under the existing
exp-008 result root. Earlier hypothesis verdicts and training limitations stand.

## Block-output voltage magnitudes, September 30

For Filip's voltage-magnitude question, reuse the block-output free-state RMS
from the same epoch0/10/30/50 replays. These older artifacts did not retain
internal-layer free statistics or signed means. All36 scheme/epoch/block values
share the same ordered256-image cohort and freeT=[6,6,4]; source runtime matches
torch2.11.0+cu128/CUDA12.8. The recorded free RMS is invariant across beta cases
and duplicate layer records. No replay or training was launched.

For ours, block-output RMS vectors are [0.013175,0.005143,0.011987] at epoch0,
[0.005084,0.021962,0.039512] at epoch10,
[0.002268,0.007227,0.006554] at epoch30, and
[0.001901,0.007113,0.004390] at epoch50. Across all three schemes, block1 output
is lower after training; block2 output is higher at every measured trained
checkpoint. Block3 output is highest at epoch10 among these samples and falls
below initialization at epochs30/50. These BPTT trajectories are not combined
with exp-022's EqProp epochs0/1/5 into one training-evolution curve.

[All-scheme BPTT block-output voltage RMS (JPG)](../../../../../results/cifar-block-rms-intermediate-20260929-v1/analysis/voltage_evolution/bptt_block_output_voltage_rms.jpg).
`analysis/voltage_evolution/voltage_epoch_comparison.csv` and `summary.json`
under this result root retain values, initialization ratios, provenance and
missing-statistic labels. Reproduce with
`python experiments/plot_cifar_free_voltage_evolution.py`.
