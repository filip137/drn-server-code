# Conv Paper Hard-Sigmoid Input-Gain Protocol

Updated: 2026-07-18

Status: protocol and all nine calibration measurements frozen.

## Purpose And Dependency

Calibrate raw `input_gain` after the experiment definition is frozen and before operational `T/K` is selected. This protocol applies only to the `hard_sigmoid` rows in the frozen [experiment definition](../../conv_paper_experiment_definition.md).

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
- Hard-sigmoid conductances: `g_on=100`, `g_off=0`.
- `v_off=4.0`, giving active interval `[-4,4]`.
- Target scope: first hidden layer only.
- Target initial saturation: `30%`.
- Training-loader shuffle seed: `0`, supplied through a dedicated loader generator and independent of model RNG consumption.

A first-hidden state is saturated when it is below `-4` or above `4` after the 64-step free-state calibration equilibrium.

## Gain Granularity And Freeze Rule

Calibrate a separate raw gain for every architecture and amplification scheme:

- Conv1 x baseline `v1/c1`, ours `v4/c1`, legacy `v4/c0.25`;
- Conv2 x baseline `v1/c1`, ours `v4/c1`, legacy `v4/c0.25`;
- Conv3 x baseline `v1/c1`, ours `v4/c1`, legacy `v4/c0.25`.

This produces nine hard-sigmoid gains. Gain is not shared across amplification schemes. Each gain is selected using model seed `0` and then frozen unchanged for all later model seeds, LR screens, longer runs, and final training under this experiment definition.

Deeper-layer saturation must be measured and reported for every selected gain, but it does not participate in the gain-selection target.

Every architecture x amplification model must start from reset global layer and parameter name counters and a fresh model seed `0`. This is part of the calibration contract, not an implementation detail: amplification currently derives local layer position and input-layer handling from layer names.

## Frozen Hard-Sigmoid Gains

The 2026-07-18 Trex calibration and independent source-config rebuild selected the following values. Layer occupancies are listed in hidden-layer order at calibration `T=64`.

| Architecture | Scheme | Frozen `input_gain` | `T=64` hidden-layer saturation |
|---|---|---:|---|
| Conv1 | baseline `v1/c1` | `75.6030807495` | `[30.000025%]` |
| Conv1 | proposed/ours `v4/c1` | `84.8402175903` | `[30.000025%]` |
| Conv1 | legacy `v4/c0.25` | `31.8188591003` | `[29.999994%]` |
| Conv2 | baseline `v1/c1` | `253.3022308350` | `[29.999994%, 0.000000%]` |
| Conv2 | proposed/ours `v4/c1` | `716.3439331055` | `[29.999994%, 4.516103%]` |
| Conv2 | legacy `v4/c0.25` | `661.4369506836` | `[29.999994%, 59.268811%]` |
| Conv3 | baseline `v1/c1` | `251.3061370850` | `[29.999994%, 0.000000%, 0.000000%]` |
| Conv3 | proposed/ours `v4/c1` | `744.7390747070` | `[30.000025%, 0.000000%, 0.000000%]` |
| Conv3 | legacy `v4/c0.25` | `665.0302124023` | `[29.999994%, 0.000000%, 17.682196%]` |

Evidence:

- [tracked frozen gain handoff](../../conv_hardsigmoid_sat30_gains_20260718.csv);
- [tracked `T=64` rebuild and operational `T/K` evidence](../../conv_hardsigmoid_tk_selection_20260718.csv).

The complete local diagnostic bundle remains under `results/conv_hardsigmoid_gain_medium_affine_t64_deterministic_cohort_20260718` and `results/conv_hardsigmoid_tk_medium_affine_deterministic_cohort_20260718`; generated result bundles are intentionally not tracked by Git.

The earlier ordinary-MNIST/old-`T` handoff gains and the intermediate medium-affine amplified gain candidates around `165-185` are superseded diagnostics. The latter were produced after another model had already advanced global name counters. For amplified rows this changed both the inferred amplification exponent and whether the first interaction was recognized as the input interaction. The baseline rows were largely unaffected because their amplification ratio is one. A second reproducibility issue tied the shuffled 256-sample cohort to global model RNG consumption; the dedicated shuffle seed now prevents that coupling.

## Search Procedure

Use the existing deterministic bracketing and binary-search convention:

1. Set the lower bracket to gain `0` and the initial upper bracket to gain `64`.
2. Double the upper bracket until first-hidden saturation reaches or exceeds `30%`.
3. Fail the calibration if a bracket is not found by gain `8192`.
4. Run `24` binary-search steps within the bracket.
5. Save the final measured saturation and all per-hidden-layer saturation fractions with the selected gain.

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
- `v_off`, target scope, and target saturation;
- selected raw `input_gain`;
- measured first-hidden saturation;
- measured saturation for every deeper hidden layer;
- bracket and binary-search settings;
- calibration status and failure reason, if any.

## Post-T/K Check

The post-selection check is complete. Initial per-layer saturation was remeasured at every row's selected operational `T` without recalibrating or changing the frozen gain. The calibration-`T` and selected-`T` values are recorded together in [`conv_hardsigmoid_tk_selection_20260718.csv`](../../conv_hardsigmoid_tk_selection_20260718.csv).
