# Conv Paper Operational T/K Protocol

Updated: 2026-07-20

Status: **hard-sigmoid selection rule and all nine values frozen. Perfect-diode rule pending.**

## Scope

This revision defines the operational `T/K` selection protocol for `hard_sigmoid` only. It does not define perfect-diode `T/K`, training batch size, optimizer, learning rate, epoch budget, final seeds, or checkpoint selection.

The July 5 `T/K` values are superseded for the deterministic medium affine MNIST experiment and must not be reused:

| Architecture | Superseded hard-sigmoid value |
|---|---|
| Conv1 | `T=4,K=4` |
| Conv2 | `T=16,K=6` |
| Conv3 | `T=16,K=16` |

## Unit Of Selection

Select one operational `T` and one operational `K` independently for each architecture × amplification row:

- Conv1, Conv2, and Conv3;
- baseline `v1/c1`, proposed/ours `v4/c1`, and legacy `v4/c0.25`.

This produces nine hard-sigmoid `(T,K)` pairs. Do not aggregate across amplification schemes or force a shared value within an architecture.

Here, “per layer” means per Conv architecture: Conv1, Conv2, and Conv3. The current minimizer has one `T` and one `K` for the whole network, not a different value for each hidden layer. Residuals and gradients are nevertheless checked layer by layer, and every required layer must pass before the network-level value is frozen.

## Frozen Prerequisites

Every diagnostic row must use:

- the dataset, preprocessing, architecture, output, and loss contract in the [experiment definition](conv_paper_experiment_definition.md);
- `hard_sigmoid` with `v_off=4.0` and active interval `[-4,4]`;
- fixed-step minimization with `adaptive_equilibrium=false`;
- model initialization seed `0` and affine seed `1729`;
- that row's frozen raw `input_gain`, selected for `30%` first-hidden-layer saturation under the [hard-sigmoid input-gain protocol](conv_paper_hard_sigmoid_input_gain_protocol.md);
- reset states at the start of every batch;
- no training before the initialization diagnostics described here.

Input gain is calibrated first with `T=64`. It remains unchanged while operational `T` and `K` are selected. After `T` is selected, remeasure all hidden-layer saturation fractions at the selected `T`, but do not recalibrate the gain.

The diagnostics may proceed row by row after that row's input gain is frozen. A row does not need to wait for another architecture's gain or `T/K` result.

## Completed Diagnostic Workflow

The completed Trex workflow used the following reusable Python components:

- `experiments/search_mnist_bp_conv_hardsigmoid_saturation_targets.py` for gain calibration;
- `experiments/create_mnist_bp_conv_hardsigmoid_tk_preflight.py` for the nine initialization-only source configs;
- `experiments/evaluate_mnist_bp_conv_residual_vs_iterations.py` for the `T` residual sweep;
- `experiments/analyze_mnist_bp_conv_k_gradient_convergence.py` for the high-`K` gradient comparison;
- `experiments/select_mnist_bp_conv_tk_from_diagnostics.py` for row-specific selection;
- `experiments/analyze_mnist_bp_conv_good_run_saturation.py` for the independent `T=64` and selected-`T` saturation measurements.

The temporary shell orchestration used during the completed run was retired by the canonical MNIST Conv workflow migration. It is not an active launch surface and is not required to interpret the tracked evidence. Any future rerun must preserve the frozen gains, `T_ref=64`, the grids below, strict residual comparison, fixed-step minimization, reset state at every batch, and the independent selected-`T` saturation remeasurement. The July 5 launcher remains historical and must not be used for this setup.

## Phase A: Select T From Residuals

### Data and candidate grid

- Phase: initialization only.
- Split: deterministic medium-affine test set.
- Samples: the first 1024 examples in deterministic loader order.
- Evaluation batch size: `128` for Conv1 and Conv2 and `64` for Conv3. Conv3 uses smaller evaluation chunks to fit Trex GPU memory; the statistic remains per-sample and the candidate rule is unchanged.
- Candidate grid: `T in {4,6,8,10,16,24,32,48,64}`.
- Mandatory high-settling reference/sentinel: `T_ref=64`.

`T_ref=64` anchors the diagnostic and must always be measured, but it is not automatically the selected operational `T`.

### Residual statistic

For hard sigmoid, use the raw absolute equilibrium residual `|dE/dz|` for every free hidden layer and the output layer.

For every layer and sample:

1. take the maximum absolute residual over all units in that layer;
2. take the 90th percentile of those per-sample maxima over the 1024 examples.

Call this value `layer_residual_p90`. Also record mean, median, p99, maximum, and the overall per-sample maximum across all free layers, but those fields are diagnostics rather than selection criteria.

### Selection rule

For one architecture × amplification row, select the smallest tested `T` satisfying

`layer_residual_p90 < 1e-2`

for every free hidden layer and the output layer. `T=64` must also satisfy the same condition; this guards against a non-monotonic early threshold crossing.

Do not round residuals before applying the strict `< 1e-2` comparison. Missing, non-finite, or incomplete layer measurements fail the candidate.

If no candidate passes, or if `T=64` fails, mark the row unresolved. Run the extension grid `T in {96,128,192,256}` while preserving the original curve and select the smallest passing value only if the largest extension sentinel also passes. Do not silently assign `T=64` to a failed row.

After selection, remeasure first-hidden and every deeper-hidden saturation fraction at the selected `T`. Keep the raw gain frozen even if the measured saturation moves away from 30%.

## Phase B: Select K Against A High-K Reference

Run Phase B only after Phase A has frozen `T` for that same architecture × amplification row.

### Data and candidate grid

- Phase: initialization only.
- Split: deterministic medium-affine training set.
- Samples: the first 256 examples in deterministic loader order.
- Evaluation batch size: 32.
- Free-state settling: the row's selected operational `T`.
- Candidate grid: `K in {4,6,8,16,32,64}`.
- High-`K` reference: `K_ref=64`.

Use exactly the same prefetched batches, initial parameter values, free equilibria, targets, and parameter ordering for every `K`. Only the backpropagation/unroll count may change.

### Gradient statistic

For every `ConvWeight_*` parameter and batch, compare the candidate-`K` BP gradient with its `K=64` reference. Aggregate each parameter over batches and record:

- gradient L2 norm;
- reference-gradient L2 norm;
- relative L2-norm delta;
- gradient-vector relative error and cosine;
- fraction of entries with absolute gradient `<=1e-12` and its delta from the reference.

The vector relative error and cosine are required diagnostics, but the frozen selection gate follows the established norm and zero-fraction comparison below.

### Selection rule

Select the smallest `K` for which every `ConvWeight_*` parameter satisfies both:

- relative gradient L2-norm delta from `K=64` `<=0.10`;
- absolute zero-fraction delta from `K=64` `<=0.02`.

For this gate, relative L2-norm delta is

`|norm_K - norm_ref| / max(|norm_K|, |norm_ref|, 1e-30)`.

Missing, non-finite, or unmatched parameters fail the candidate. Selection is row-specific: a result from one architecture or amplification scheme cannot set another row's `K`.

If the smallest passing value is the boundary `K=64`, run a `K=128` sentinel. Freeze `K=64` only if it also passes the same two comparisons against `K=128`; otherwise extend the candidate/reference range and leave the row unresolved until a new high-`K` reference has been documented.

## Frozen Hard-Sigmoid T/K Values

The 2026-07-18 Trex run selected the following values. `selected residual` is the largest layer p90 at the chosen `T`; `gradient delta` and `zero delta` are the worst ConvWeight comparisons at the chosen `K` against `K=64`.

| Architecture | Scheme | Frozen gain | `T` | Selected residual | `K` | Gradient delta | Zero delta |
|---|---|---:|---:|---:|---:|---:|---:|
| Conv1 | baseline `v1/c1` | `75.603081` | `4` | `5.48e-5` | `4` | `1.40e-8` | `0` |
| Conv1 | proposed/ours `v4/c1` | `84.840218` | `4` | `1.68e-4` | `4` | `4.89e-6` | `0` |
| Conv1 | legacy `v4/c0.25` | `31.818859` | `4` | `7.16e-4` | `4` | `2.60e-4` | `0` |
| Conv2 | baseline `v1/c1` | `253.302231` | `16` | `2.32e-3` | `6` | `4.05e-2` | `0` |
| Conv2 | proposed/ours `v4/c1` | `716.343933` | `24` | `3.46e-3` | `6` | `6.24e-2` | `0` |
| Conv2 | legacy `v4/c0.25` | `661.436951` | `8` | `3.22e-3` | `4` | `6.41e-3` | `0` |
| Conv3 | baseline `v1/c1` | `251.306137` | `24` | `2.64e-3` | `8` | `9.75e-2` | `1.70e-6` |
| Conv3 | proposed/ours `v4/c1` | `744.739075` | `32` | `4.55e-3` | `8` | `5.02e-2` | `4.24e-7` |
| Conv3 | legacy `v4/c0.25` | `665.030212` | `8` | `7.33e-3` | `6` | `7.82e-2` | `0` |

All nine `T=64` reference residuals passed. No row selected boundary `K=64`, so the `K=128` sentinel was not triggered. The selected-`T` first-hidden occupancies remain within `0.007` percentage points of the 30% calibration target. The most material deeper-layer change is Conv3 legacy layer 3, from `17.6822%` at `T=64` to `9.2615%` at selected `T=8`; the gain remains frozen by protocol.

Evidence:

- [tracked frozen gain handoff](conv_hardsigmoid_sat30_gains_20260718.csv);
- [tracked row-specific `T/K`, residual, gradient, and saturation evidence](conv_hardsigmoid_tk_selection_20260718.csv).

The complete generated diagnostic bundle remains locally under `results/conv_hardsigmoid_tk_medium_affine_deterministic_cohort_20260718` and is intentionally not tracked by Git.

## Required Output

The final hard-sigmoid selection table must contain exactly nine rows and record at least:

- architecture, scheme label, voltage amplification, and current amplification;
- frozen input gain and its calibration artifact;
- selected `T`, every per-layer residual statistic at selected `T`, and the `T=64` values;
- all hidden-layer saturation fractions at calibration `T=64` and selected operational `T`;
- selected `K`, reference `K`, and every per-`ConvWeight_*` comparison used by the gate;
- dataset transform, affine seed, model seed, split, sample count, batch size, fixed-step setting, candidate grids, and any extension sentinels;
- selection status, boundary status, and evidence paths.

The table must retain failed and unresolved rows rather than dropping them.

## Current Gate

Hard-sigmoid input gain and operational `T/K` are frozen. The later Conv1/Conv2 hard-sigmoid seed-0 studies froze batch size 16 and plain SGD for the screen and completed the six-row LR handoff: two v1 baseline values plus four v3 parameter-relative-rho amplified values are `frozen_seed0_screen`. Hard-sigmoid paper training remains blocked on a later final-training protocol settling epoch budget, final seeds, checkpoint selection, and inclusion rules; this `T/K` document does not define those choices.

Perfect-diode `T/K` remains pending definition and is not authorized by this document.
