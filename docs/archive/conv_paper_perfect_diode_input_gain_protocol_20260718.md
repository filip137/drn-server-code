# Historical Archive — Conv Paper Perfect-Diode Input-Gain Protocol

Archived: 2026-07-26

Status: historical only. This document preserves the former pending
deterministic-medium-affine perfect-diode input-gain protocol exactly enough
to retain its provenance. It is not an active protocol, and the user-fixed
ordinary-MNIST diagnostic gains `40`, `100`, and `360` do not complete or
supersede the medium-affine calibration described below.

---

# Conv Paper Perfect-Diode Input-Gain Protocol

Updated: 2026-07-18

Status: protocol frozen; calibration measurements pending.

## Purpose And Dependency

Calibrate raw `input_gain` after the experiment definition is frozen and before operational `T/K` is selected. This protocol applies only to the `perfect_diode` rows in the frozen [experiment definition](../conv_paper_experiment_definition.md).

## Fixed Calibration Configuration

- Dataset: deterministic medium affine MNIST.
- Affine seed: `1729`.
- Model initialization seed: `0`.
- Split: training.
- Samples: `256`, selected deterministically under seed `0`.
- Calibration batch size: `64`.
- Minimizer: explicit fixed-step configuration with `adaptive_equilibrium=false`.
- Provisional calibration settling count: `T=64`.
- Training `K`: not used during initialization calibration.
- Clamp epsilon: `1e-8`.
- Target scope: first hidden layer only.
- Target initial clamped occupancy: `30%`.

For a paired perfect-diode hidden state, split the channel dimension into equal excitatory and inhibitory halves. Count a unit as clamped when:

- its excitatory state is `<=1e-8`; or
- its inhibitory state is `>=-1e-8`.

The first-hidden clamped fraction is the combined clamped count divided by the combined excitatory and inhibitory unit count after the 64-step free-state calibration equilibrium.

## Gain Granularity And Freeze Rule

Calibrate a separate raw gain for every architecture and amplification scheme:

- Conv1 x baseline `v1/c1`, ours `v4/c1`, legacy `v4/c0.25`;
- Conv2 x baseline `v1/c1`, ours `v4/c1`, legacy `v4/c0.25`;
- Conv3 x baseline `v1/c1`, ours `v4/c1`, legacy `v4/c0.25`.

This produces nine perfect-diode gains. Gain is not shared across amplification schemes. Each gain is selected using model seed `0` and then frozen unchanged for all later model seeds, LR screens, longer runs, and final training under this experiment definition.

Deeper-layer clamped occupancy must be measured and reported for every selected gain, but it does not participate in the gain-selection target.

## Search Procedure

Use the same deterministic bracketing and binary-search convention as the hard-sigmoid calibration:

1. Set the lower bracket to gain `0` and the initial upper bracket to gain `64`.
2. Double the upper bracket until first-hidden clamped occupancy reaches or exceeds `30%`.
3. Fail the calibration if a bracket is not found by gain `8192`.
4. Run `24` binary-search steps within the bracket.
5. Save the final measured occupancy and all per-hidden-layer occupancy fractions with the selected gain.

Do not select gain using accuracy, learning rate, trained checkpoints, or a different operational `T`.

## Required Output

The calibration table must contain:

- architecture and convolution pipeline;
- nonlinearity;
- amplification label, run name, voltage amplification, and current amplification;
- dataset name and serialized affine transform;
- affine seed and model seed;
- sample count and calibration batch size;
- provisional calibration `T` and explicit adaptive-equilibrium setting;
- clamp epsilon, target scope, and target occupancy;
- selected raw `input_gain`;
- measured first-hidden clamped occupancy;
- measured clamped occupancy for every deeper hidden layer;
- bracket and binary-search settings;
- calibration status and failure reason, if any.

## Post-T/K Check

After operational `T/K` is eventually selected, remeasure initial per-layer clamped occupancy at the selected operational `T`. Do not recalibrate or change the frozen gain. Report any difference from the `T=64` calibration measurement.
