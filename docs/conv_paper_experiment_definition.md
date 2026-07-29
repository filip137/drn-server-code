# Conv Paper Experiment Definition

Updated: 2026-07-29

Status: frozen comparison contract. Learning-rate handoffs and some final-run
choices remain pending in the [protocol index](conv_paper_hyperparameter_protocol.md).

## Question

The paper compares bidirectional amplification in perfect-diode dissipative
resistive Conv networks under a wide conductance range and a narrow physical
conductance contract.

Training uses backpropagation through time (BPTT) through the explicit,
fixed-`K` unrolled equilibrium iterations. It does not use EqProp, nudging
updates, or a feed-forward surrogate.

## Paper Dataset

Paper runs use deterministic **medium affine MNIST**:

- base dataset: MNIST train and test splits;
- rotation: uniform in `[-25 deg,+25 deg]`;
- translation: independently up to `20%` of image width and height;
- scale: uniform in `[0.8,1.2]`;
- shear: `0`;
- interpolation: bilinear;
- fill: `0`;
- affine seed: `1729`; and
- each sample's transform is fixed by sample index, split, and affine seed and
  is not resampled by epoch.

Apply the affine transform before tensor conversion and DRN preprocessing:

```text
x_normalized = 0.3 * (x - 0.1307) / 0.3081
x_signed = concat(x_normalized, -x_normalized)
```

The model's frozen `input_gain` is applied after this preprocessing. This
dataset is not pixel-permuted MNIST.

Ordinary MNIST is used only for `T/K`, rho, learning-rate, and bounded
initializer selection. Its validation accuracy is diagnostic and is not
paper-facing evidence.

## Architecture

All convolutional layers use kernel size `3`, padding `1`, and no pooling.

| Architecture | Hidden channels | Strides | Hidden shapes | Output |
|---|---|---|---|---|
| Conv1 | `[64]` | `[2]` | `[64,14,14]` | `[20]` |
| Conv2 | `[64,128]` | `[2,2]` | `[64,14,14]`, `[128,7,7]` | `[20]` |
| Conv3 | `[64,128,256]` | `[2,2,1]` | `[64,14,14]`, `[128,7,7]`, `[256,7,7]` | `[20]` |

The classifier uses 20 paired outputs for 10 classes and paired squared-error
loss. Output-10 checkpoints are a different protocol.

## Nonlinearity And Amplification

The active nonlinearity is `perfect_diode` with explicit diode parameter
dictionaries and the protocol's clamp epsilon.

| Label | Run name | Voltage amp | Current amp |
|---|---|---:|---:|
| baseline | `mnist_bp_amp_v1_c1` | `1` | `1` |
| proposed/ours | `mnist_bp_amp_v4_c1` | `4` | `1` |
| legacy | `mnist_bp_amp_v4_c0p25` | `4` | `0.25` |

The old five-setting grid and hard sigmoid are outside the active paper
contract.

## Optimizers

Every architecture and amplification scheme is run with:

- plain SGD with momentum `0` and weight decay `0`; and
- Adam with the exact optimizer settings frozen by the rho protocol.

Rho targets, proposal units, and raw learning-rate vectors are selected
independently for every architecture x scheme x optimizer surface.

## Weight Contracts

Each paper surface has two result conditions:

1. **Wide-range reference:** conductance weights are projected to `[0,100]`
   and use the reference initialization contract.
2. **Bounded hardware contract:** `ConvWeight_*` and `DenseWeight_*` are
   projected to `[1e-5,1e-4]`. Initialization is selected globally between:
   - `bounded_uniform`, sampled from `[1e-5,1e-4)`; and
   - `bounded_kaiming_uniform`, the midpoint-centered, fan-in-scaled bounded
     Kaiming initializer with gain `1`.

Biases are not conductance weights and are not projected to this interval.
The bounded initializer and bounded raw LR vectors are selected on ordinary
MNIST under the
[bounded-weight protocol](perfectdiode_bounded_weight_protocol.md).

If uniform initialization wins, the bounded condition differs from the
wide-range reference in both range and initialization. Report it as one
hardware-constrained contract, not as a bounds-only causal ablation.

## Comparison Contract

Before paper training, every row must freeze and record:

- the exact dataset realization and preprocessing;
- architecture, convolution pipeline, output encoding, and loss;
- nonlinearity and explicit diode parameters;
- amplification values;
- weight contract and initialization;
- operational `T/K`;
- optimizer and complete parameter-specific LR vector;
- model and loader seeds;
- epoch budget; and
- checkpoint and inclusion rules.

The medium-affine paper config consumes the corresponding ordinary-MNIST
handoff unchanged. It must not recalibrate rho or rewrite raw learning rates.
Changing architecture, scheme, optimizer, `T/K`, weight contract, or
initializer invalidates the dependent handoff.
