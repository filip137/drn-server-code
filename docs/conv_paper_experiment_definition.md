# Conv Paper Experiment Definition

Updated: 2026-07-18

Status: frozen.

This document freezes the comparison axes that must be identical before `input_gain`, operational `T/K`, optimizer, or learning rate is selected.

## Dataset

Use deterministic **medium affine MNIST**, matching the medium transform used by the Hopfield depth-gap experiments.

- Base dataset: MNIST train and test splits.
- Rotation: sampled uniformly from `[-25 deg, +25 deg]`.
- Translation: sampled independently up to `20%` of image width and height in either direction.
- Scale: sampled uniformly from `[0.8, 1.2]`.
- Shear: `0`.
- Interpolation: bilinear.
- Fill outside the transformed image: `0`.
- Affine seed: `1729`.
- Determinism: each sample receives a fixed transform determined by sample index, split, and affine seed; the transform does not depend on model seed and is not resampled by epoch.

Apply the affine transform before tensor conversion and DRN preprocessing.

Use the established signed two-channel DRN input representation:

- MNIST normalization mean: `0.1307`;
- MNIST normalization standard deviation: `0.3081`;
- post-normalization scale: `0.3`;
- model input shape: `[2, 28, 28]`.

These constants and their order are unchanged from the recovered ordinary-MNIST DRN preprocessing. For an affine-transformed single-channel tensor `x`, the preprocessing is:

```text
x_normalized = 0.3 * (x - 0.1307) / 0.3081
x_signed = concat(x_normalized, -x_normalized)
```

The model's separately calibrated `input_gain` is applied to `x_signed`; it is not part of dataset normalization.

This dataset is not pixel-permuted MNIST. Result paths, configs, tables, and paper text must use the term `medium affine MNIST`.

## Architecture

All convolutional layers use kernel size `3`, padding `1`, and no pooling. Hidden channel counts stay fixed across nonlinearities and amplification schemes.

| Architecture | Hidden channels | Strides | Hidden shapes | Output shape |
|---|---|---|---|---|
| Conv1 | `[64]` | `[2]` | `[64,14,14]` | `[20]` |
| Conv2 | `[64,128]` | `[2,2]` | `[64,14,14]`, `[128,7,7]` | `[20]` |
| Conv3 | `[64,128,256]` | `[2,2,1]` | `[64,14,14]`, `[128,7,7]`, `[256,7,7]` | `[20]` |

The classifier uses 20 paired outputs for 10 digit classes and the paired squared-error loss. Output-10 checkpoints and results are a different protocol and are not comparable rows.

## Amplification Schemes

Only these three schemes belong to the main comparison:

| Label | Run name | Voltage amplification | Current amplification |
|---|---|---:|---:|
| baseline | `mnist_bp_amp_v1_c1` | `1` | `1` |
| proposed amplification / ours | `mnist_bp_amp_v4_c1` | `4` | `1` |
| legacy amplification | `mnist_bp_amp_v4_c0p25` | `4` | `0.25` |

The old five-setting grid is not part of this protocol.

## Nonlinearities

Run the complete architecture and amplification grid separately for:

- `hard_sigmoid`;
- `perfect_diode`.

This creates 18 frozen input-gain configurations: 3 architectures x 3 amplification schemes x 2 nonlinearities.

## Comparison Contract

Dataset realization, preprocessing, architecture, output encoding, initialization rule, and amplification values must not change across rows. Every resolved source config must record the affine transform, affine seed, model seed, convolution pipeline, hidden shapes, output dimension, nonlinearity, voltage amplification, and current amplification.

The following are not defined here: calibrated raw `input_gain`, operational `T/K`, training batch size, optimizer, learning rate, epoch budget, final model seeds, and checkpoint inclusion rules. Those decisions belong to later protocols linked from the [protocol index](conv_paper_hyperparameter_protocol.md).
