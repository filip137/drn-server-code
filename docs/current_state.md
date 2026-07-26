# Current Research State

Updated: 2026-07-27

This is a short dashboard, not a second protocol. Detailed evidence and
selection rules remain in the linked protocol documents.

## Current focus

The immediate scope is Conv1 and Conv2 with:

- baseline `v1/c1`, proposed/ours `v4/c1`, and legacy `v4/c0.25`;
- hard sigmoid and perfect diode;
- plain SGD and Adam;
- an unrestricted-conductance comparison followed by the same comparison with
  a limited weight range.

The current SGD/Adam learning protocols are defined on ordinary MNIST:

- [`hardsigmoid_learning_protocol.md`](hardsigmoid_learning_protocol.md);
- [`perfectdiode_learning_protocol.md`](perfectdiode_learning_protocol.md).

They do not by themselves authorize deterministic-medium-affine paper
training. Unless a later protocol says otherwise, the planned 10- and
30-epoch result runs below refer to the current ordinary-MNIST Conv1/Conv2
comparison.

## Locked medium-affine hard-sigmoid learning rates

Yes: the six Conv1/Conv2 hard-sigmoid seed-0 peak learning rates below are
locked with status `frozen_seed0_screen` for the deterministic-medium-affine,
plain-SGD handoff.

| Architecture | Scheme | Frozen gain | Operational `T/K` | Frozen peak LR |
|---|---|---:|---:|---:|
| Conv1 | baseline `v1/c1` | `75.6030807495` | `4/4` | `0.30174007288404703` |
| Conv1 | proposed/ours `v4/c1` | `84.8402175903` | `4/4` | `0.001838414260143094` |
| Conv1 | legacy `v4/c0.25` | `31.8188591003` | `4/4` | `4.639114646319357e-5` |
| Conv2 | baseline `v1/c1` | `253.3022308350` | `16/6` | `1.7347317646208873` |
| Conv2 | proposed/ours `v4/c1` | `716.3439331055` | `24/6` | `0.011865709277518458` |
| Conv2 | legacy `v4/c0.25` | `661.4369506836` | `8/4` | `0.0015293192084949393` |

Baseline values come from the v1 conductance-span screen. Amplified values
come from the v3 initial-parameter-RMS screen. The full-precision authority is
[`conv_hardsigmoid_lr_active_handoff_relative_rho_v3.csv`](conv_hardsigmoid_lr_active_handoff_relative_rho_v3.csv).

These are peak rates for the reviewed plain-SGD warm-up/cosine schedule. They
are not constant-LR values and are not Adam learning rates. The canonical
handoff and schedule are documented in
[`conv_paper_learning_rate_protocol.md`](conv_paper_learning_rate_protocol.md).

The associated five-epoch medium-affine screen accuracies were much lower than
the ordinary-MNIST results below. They are omitted here to avoid presenting
cross-dataset numbers as one comparable result table.

## Completed ordinary-MNIST hard-sigmoid diagnostic

This is the high-accuracy MNIST result. The completed v5 diagnostic disabled
the affine transform and trained every selected constant-vector plain-SGD
candidate for exactly five epochs (`17,190` optimizer steps). It selected:

- Conv1: historical-profile arm with architecture alpha `1e-3`;
- Conv2: strict-equal arm with architecture alpha `3e-3`.

The alpha is the architecture-level selection coordinate, not a raw optimizer
learning rate. The selected constant plain-SGD vectors were:

| Architecture | Scheme | Conv-weight and attached-bias LR(s) | Dense-weight LR |
|---|---|---|---:|
| Conv1 | baseline `v1/c1` | `ConvWeight_0/Bias_0 = 0.012753716099960177` | `0.010733445246971115` |
| Conv1 | proposed/ours `v4/c1` | `ConvWeight_0/Bias_0 = 0.0022473180071152475` | `0.0018430140298218768` |
| Conv1 | legacy `v4/c0.25` | `ConvWeight_0/Bias_0 = 0.00015948762988316354` | `0.00012492257381097838` |
| Conv2 | baseline `v1/c1` | `ConvWeight_0/Bias_0 = 0.28517673662900744`; `ConvWeight_1/Bias_1 = 0.17572188418530033` | `0.008570616719708114` |
| Conv2 | proposed/ours `v4/c1` | `ConvWeight_0/Bias_0 = 0.010165416619510566`; `ConvWeight_1/Bias_1 = 0.01066261593218855` | `0.0004309071720122689` |
| Conv2 | legacy `v4/c0.25` | `ConvWeight_0/Bias_0 = 0.0002140195123572878`; `ConvWeight_1/Bias_1 = 0.00030380191511292584` | `0.000013754700600274864` |

| Architecture | Scheme | Maximum validation accuracy | Accuracy epoch / trained epochs | Minimum-loss checkpoint epoch | Clean test accuracy | Observed checkpoint conductance range |
|---|---|---:|---:|---:|---:|---:|
| Conv1 | baseline `v1/c1` | `94.98%` | `5 / 5` | `4` | `95.37%` | `[0, 0.2740043104]` |
| Conv1 | proposed/ours `v4/c1` | `95.56%` | `4 / 5` | `3` | `96.07%` | `[0, 0.2474433184]` |
| Conv1 | legacy `v4/c0.25` | `94.82%` | `5 / 5` | `4` | `95.18%` | `[0, 0.2518008053]` |
| Conv2 | baseline `v1/c1` | `93.18%` | `5 / 5` | `5` | `93.99%` | `[0, 0.6000501513]` |
| Conv2 | proposed/ours `v4/c1` | `95.42%` | `5 / 5` | `5` | `95.72%` | `[0, 0.3044329584]` |
| Conv2 | legacy `v4/c0.25` | `94.32%` | `5 / 5` | `5` | `95.02%` | `[0, 0.2438901961]` |

The maximum validation accuracy is the largest accuracy observed across the
five epoch-end validations. The clean test accuracy comes from a later
read-only replay of each immutable minimum-validation-loss checkpoint, so its
checkpoint epoch need not equal the maximum-accuracy epoch. The observed
conductance range is the global minimum and maximum across all `ConvWeight`
and `DenseWeight` tensors in that same minimum-loss checkpoint; biases are
excluded. These are achieved values inside the configured training bounds
`[0,100]`, not imposed limits. The Conv1 and Conv2 replay details are in
[`conv1_hardsigmoid_finite_conductance_ordinary_mnist_diagnostic.md`](conv1_hardsigmoid_finite_conductance_ordinary_mnist_diagnostic.md)
and
[`conv2_conv3_hardsigmoid_finite_conductance_ordinary_mnist.md`](conv2_conv3_hardsigmoid_finite_conductance_ordinary_mnist.md).

The separate limited-conductance replay found a shared sampled window of
`[1e-3,0.3]` for Conv1 and `[1e-4,0.3]` for Conv2 in which every scheme stayed
within one percentage point of its own clean test accuracy. Those are tested
diagnostic windows, not yet the frozen bounds for the planned training
comparison.

These are selected ordinary-MNIST v5 SGD diagnostics, not
deterministic-medium-affine paper results. The newer ordinary-MNIST
SGD-and-Adam protocol still has measurements pending; using the v5 selection
there would have to be an explicit reuse rather than a new selection.

## Perfect-diode Conv1/Conv2 operating points

The current ordinary-MNIST perfect-diode protocol uses user-fixed diagnostic
operating points shared across the three amplification schemes:

| Architecture | Fixed `input_gain` | Fixed operational `T/K` |
|---|---:|---:|
| Conv1 | `40` | `4/4` |
| Conv2 | `100` | `6/6` |

These values are fixed inputs, not calibration measurements. Before its LR
search, every scheme must pass the fixed-`T/K` gradient security comparison
against the `(T=64,K=64)` reference: Conv1 compares `(4,4)` and Conv2 compares
`(6,6)`. Perfect-diode learning rates are not yet selected.

## Learning-rate status

| Block | SGD | Adam |
|---|---|---|
| Conv1/Conv2 hard sigmoid, deterministic medium affine | six row-specific peak LRs locked for the reviewed SGD schedule | no locked Adam handoff |
| Conv1/Conv2 hard sigmoid, ordinary-MNIST v5 diagnostic | selected: Conv1 historical profile / `1e-3`; Conv2 strict equal / `3e-3` | not studied |
| Conv1/Conv2 hard sigmoid, current ordinary-MNIST protocol | new optimizer-specific selection pending; v5 reuse must be explicit | pending optimizer-specific LR selection |
| Conv1/Conv2 perfect diode, ordinary MNIST | pending optimizer-specific LR selection | pending optimizer-specific LR selection |

The former v1-v7 studies and the Conv2 SGD/Adam boundary study remain useful
diagnostic evidence in
[`conv_learning_rate_diagnostics.md`](conv_learning_rate_diagnostics.md), but
they do not change this status table.

## Remaining protocol decisions

| Decision | Status |
|---|---|
| Perfect-diode Conv1/Conv2 gains and `T/K` | fixed diagnostic inputs: Conv1 `40`, `4/4`; Conv2 `100`, `6/6` |
| Perfect-diode Conv1/Conv2 SGD learning rates | pending |
| Perfect-diode Conv1/Conv2 Adam learning rates | pending |
| Hard-sigmoid Adam learning rates | pending |
| Conv1 result budget | fixed at 10 epochs |
| Conv2 result budget | fixed at 30 epochs |
| Optimizers in the result comparison | plain SGD and Adam |
| Limited weight range | comparison requested; exact bounds still need to be frozen |
| Final model seeds | pending |
| Final checkpoint rule | pending |
| Final inclusion categories | pending |
| Conv3 | outside the immediate Conv1/Conv2 result sequence; unresolved |

## Next actions

1. Complete the missing optimizer-specific LR handoffs:
   - select perfect-diode Conv1/Conv2 LRs separately for plain SGD and Adam;
   - select hard-sigmoid Adam LRs, because the frozen SGD rates cannot be
     transferred to Adam.
2. Run the unrestricted-weight result grid with both optimizers:
   - Conv1 for 10 epochs;
   - Conv2 for 30 epochs.
3. Freeze the exact limited weight bounds, then repeat the same Conv1/Conv2,
   nonlinearity, amplification, optimizer, epoch, seed, and checkpoint grid
   with only the weight range changed.
