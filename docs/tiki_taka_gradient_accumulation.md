# Tiki-Taka gradient accumulation

The optional `update_pipeline` configuration routes gradients produced by
EqProp/DRN through an auxiliary ("fast") tensor before changing the DRN's
visible ("slow") parameters. The DRN still computes one complete parameter
gradient tensor per minibatch. Tiki-Taka changes only how that tensor is
accumulated and applied.

The implementation is in `training/tiki_taka.py`. It is an ideal, tensor-level
emulator of the two-crossbar idea in AIHWKit's `TransferCompound`, not an
AIHWKit device simulation.

## Selecting the update pipeline

An omitted value, JSON `null`, or the following object selects the existing
direct SGD path:

```json
{
  "update_pipeline": {
    "type": "direct"
  }
}
```

The `direct` object cannot contain additional keys. To enable the auxiliary
array, use `"type": "tiki_taka"`.

For `labs/small_network.py`, `labs/tools/optuna_digits_train.py`, and
`labs/mnist_train.py`, place `update_pipeline` at the top level of the input
JSON. `labs/mnist_monitor_training.py` and
`plotting_functions/visualize_training_steps.py` read the same object from
`training.update_pipeline`.

Here is an explicit top-level small-network example. The values are
illustrative rather than implicit recommendations:

```json
{
  "learning_rate": [0.8, 0.3],
  "update_pipeline": {
    "type": "tiki_taka",
    "fast_lr": 1.0,
    "transfer_every": 4,
    "units_in_mbatch": true,
    "n_reads_per_transfer": 2,
    "gamma": 0.0,
    "transfer_lr": 1.0,
    "scale_transfer_lr": true,
    "transfer_columns": true,
    "with_reset_prob": 1.0,
    "random_selection": false,
    "fast_weight_min": -1.0,
    "fast_weight_max": 1.0,
    "accumulate_biases": false
  }
}
```

Unknown keys and values of the wrong type are rejected. Tiki-Taka currently
requires optimizer `momentum` and `weight_decay` both to be zero because their
placement in the two-array update rule is undefined.

## Exact update rule and signs

For a parameter, let:

- `G_t` be the complete gradient tensor estimated by EqProp/DRN for minibatch
  `t` and assigned to `parameter.state.grad`;
- `A_t` be the auxiliary fast array, initialized lazily to zero;
- `W_t` be the visible DRN parameter used for forward relaxation and
  inference;
- `eta_fast` be `fast_lr` when it is positive, or the parameter group's
  current optimizer learning rate when `fast_lr` is zero;
- `ell_t` be the parameter group's current optimizer learning rate; and
- `eta_transfer` be `transfer_lr`.

The fast write on every optimizer step that has a gradient is

```text
A_half = clip(A_(t-1) - eta_fast * G_t,
              fast_weight_min, fast_weight_max)
```

Either clipping bound may be absent. If both are absent, the operation is just

```text
A_half = A_(t-1) - eta_fast * G_t.
```

Thus the fast array stores the negative-gradient/update direction. A positive
gradient writes a negative value into `A`; transfer then *adds* that negative
value to `W`.

A transfer is due after the fast write when

```text
transfer_every > 0 and t % transfer_every == 0.
```

For a due transfer, its effective scale is

```text
alpha_t = eta_transfer * ell_t    if scale_transfer_lr is true
alpha_t = eta_transfer            if scale_transfer_lr is false.
```

For each selected logical column `j`,

```text
W[:, j] = W[:, j] + alpha_t * A_half[:, j].
```

For each selected logical row `i`,

```text
W[i, :] = W[i, :] + alpha_t * A_half[i, :].
```

This sign is the intended descent sign: since `A_half` contains
`-eta_fast * G` contributions, its additive transfer moves the visible
parameter in the negative-gradient direction. The code does not negate
`A_half` again during transfer.

After a selected column has been added to `W`, the complete selected column of
`A` is independently reset to zero with probability `with_reset_prob`.
Unselected slices, and selected slices that are not reset, retain their values
for later transfers. With the default reset probability of zero, a stored
value can contribute again when its slice is selected on a later pass.
AIHWKit supports reset only for column transfer, so configurations combining
row transfer with a nonzero reset probability are rejected.

The Tiki-Taka optimizer itself does not clamp the visible tensor. Existing DRN
trainers call each parameter's `clamp_()` after the optimizer step, so any
configured DRN parameter bounds are applied after the additive transfer. Fast
array bounds are separate and are controlled only by `fast_weight_min` and
`fast_weight_max`.

Biases use ordinary digital SGD by default:

```text
W_t = W_(t-1) - ell_t * G_t.
```

Setting `accumulate_biases` to true instead gives each bias tensor a fast
array and applies the same transfer equations as for weights. Pooling weights
are excluded from both optimizers and remain frozen.

## Minibatch and slice schedule

`transfer_every` counts optimizer steps with a gradient for each parameter.
In the provided trainers there is normally one such step per minibatch.
Counters and selection cursors are stored per parameter.

For example, with `transfer_every=4`, the first three minibatches only write
their gradients into `A`. Minibatch 4 first writes its gradient and then
transfers, so the transfer includes minibatch 4. The next transfers are due
after minibatches 8, 12, and so on.

At each due event:

1. The number of slices is
   `min(n_reads_per_transfer, number_of_slices_on_the_selected_axis)`.
2. With `random_selection=false`, selection begins at the saved cursor, takes
   consecutive slices, wraps at the edge, and advances the cursor by the
   number selected.
3. With `random_selection=true`, a new random starting slice is drawn for
   every event; the remaining slices in that event are consecutive with
   wraparound.
4. The visible add happens before the optional reset of each selected fast
   slice.

`transfer_every=0` disables all transfers: fast arrays continue to accumulate
and clip, while their visible parameters do not change. If the effective
transfer scale is zero, no visible add or probabilistic reset is performed,
although the selection cursor and transfer-event counter still advance.
Parameters whose gradient is `None` do not write, increment their step
counter, or transfer. Sparse gradients are unsupported.

The learning-rate scheduler changes `ell_t`. With a positive `fast_lr`, the
fast write scale remains fixed; with `fast_lr=0`, the fast write follows
`ell_t`. Independently, `scale_transfer_lr=true` makes scheduling change the
visible transfer scale. With `scale_transfer_lr=false`, `ell_t` does not affect
Tiki-Taka weight transfers, although it still controls any directly updated
biases.

## Logical crossbar layouts

Transfers operate on a logical two-dimensional `[out, in]` view. These are
views of the original parameter tensor, so updates change the original tensor
in place.

| Parameter | Stored tensor | Logical crossbar |
| --- | --- | --- |
| Dense weight | `pre_shape + post_shape` | `state.reshape(prod(pre_shape), -1).T`, with shape `[prod(post_shape), prod(pre_shape)]` |
| Convolution weight | `[out_channels, in_channels, kernel_height, kernel_width]` | `state.reshape(out_channels, -1)`, with shape `[out_channels, in_channels * kernel_height * kernel_width]` |
| Bias | Any bias shape | `state.reshape(-1, 1)`, with shape `[number_of_bias_values, 1]` |

With `transfer_columns=true`, a dense transfer selects flattened input
coordinates and a convolution transfer selects flattened
input-channel/kernel coordinates; all output rows for each selected coordinate
are updated. With `transfer_columns=false`, a transfer selects output units or
output channels and updates all input columns for each selected output.

A bias has one logical column. Column transfer therefore transfers the entire
bias tensor at every due event, while row transfer can select individual bias
values. Biases bypass the auxiliary path unless `accumulate_biases=true`.

The small-network trainer currently assigns bias parameter groups a learning
rate of zero. Consequently, its biases remain unchanged on the default direct
bias path. If bias accumulation is enabled there,
`scale_transfer_lr=true` also makes their effective visible transfer scale
zero; `scale_transfer_lr=false` uses `transfer_lr` directly.

## Configuration reference

The values below are the defaults used when a Tiki-Taka object omits a field.

| Field | Default | Meaning and constraint |
| --- | ---: | --- |
| `fast_lr` | `1.0` | Non-negative scale of each negative-gradient write into the fast array; zero uses the parameter group's current learning rate. |
| `transfer_every` | `1` | Non-negative integer number of optimizer steps between transfer events; zero disables transfer. |
| `units_in_mbatch` | `true` | Must be true: this integration receives one averaged EqProp gradient per minibatch and counts optimizer steps. |
| `n_reads_per_transfer` | `1` | Positive integer number of consecutive logical slices selected per event, capped by the axis size. |
| `gamma` | `0.0` | Must be zero so only the slow/main DRN array is visible to forward relaxation and inference. |
| `transfer_lr` | `1.0` | Non-negative base scale of the fast-to-visible add. |
| `scale_transfer_lr` | `true` | Multiply `transfer_lr` by the parameter group's current learning rate. |
| `transfer_columns` | `true` | Select logical columns; false selects logical rows. |
| `with_reset_prob` | `0.0` | Probability in `[0, 1]` of zeroing each selected fast column after its visible add; must be zero for row transfer. |
| `random_selection` | `false` | Draw a random starting slice per event instead of following the saved sequential cursor. |
| `fast_weight_min` | `null` | Optional lower clamp bound for the signed fast array. |
| `fast_weight_max` | `null` | Optional upper clamp bound for the signed fast array; when both bounds are present, min must be less than max. |
| `accumulate_biases` | `false` | Use auxiliary arrays for biases instead of direct digital SGD. |

These defaults define a Tiki-Taka pipeline, but they do not make it equivalent
to direct SGD. In particular, only one logical column is transferred per
minibatch by default, and the fast slice is not reset after transfer.

## Correspondence with AIHWKit

The design follows the pipeline described in
`/home/filip/aihwkit/DEVICE_WRITE_PIPELINE.md`:

```text
EqProp/DRN complete gradient tensor
  -> ideal write to signed fast array A
  -> periodic logical row/column selection
  -> ideal additive transfer
  -> visible DRN parameter W
```

The correspondence to AIHWKit's `TransferCompound` is:

| This implementation | AIHWKit concept |
| --- | --- |
| Auxiliary tensor `A` | First, fast/hidden gradient crossbar |
| Visible DRN parameter `W` | Second, slow/visible crossbar |
| Current gradient tensor `G_t` | The desired update that AIHWKit normally realizes from retained activation `x`, error signal `d`, and pulse coincidences |
| `transfer_every` | Periodic transfer cycle |
| `n_reads_per_transfer` | Number of consecutive one-hot row/column reads per event |
| `transfer_columns` and `random_selection` | Transfer direction and slice-selection policy |
| `fast_lr`, `transfer_lr`, and `scale_transfer_lr` | Fast-write and transfer learning-rate controls |
| `with_reset_prob` | Optional reset of a transferred fast slice |
| Forward/inference using only `W` | The usual two-device `gamma=0` interpretation, where the slow array is visible and the fast array remains hidden |

The configuration names intentionally mirror the corresponding
`TransferCompound` controls where practical. The schedule here is nevertheless
defined in integer optimizer/minibatch steps and therefore requires
`units_in_mbatch=true`. AIHWKit can also express transfer cycles in mat-vec
units and implements the transfer through device reads and pulsed writes.

Both arrays here are dimensionless signed model tensors. They are not raw
conductances in siemens. As with a signed analog device state, a physical
interpretation could use a differential conductance pair proportional to
`G_plus - G_minus`, but such a pair is not represented by this implementation.

## Ideal-emulator limitations

This pipeline deliberately does not model:

- activation/error pulse-train generation or coincident cross-point writes;
- finite update granularity, pulse counting, or stochastic pulse events;
- device-to-device or cycle-to-cycle variation;
- asymmetric, nonlinear, or state-dependent device updates;
- fast- or slow-device programming noise, drift, read noise, ADC/DAC effects,
  or transfer readout error;
- physical conductance conversion or differential conductance pairs;
- analog `transfer_forward` and `transfer_update` behavior;
- more than one fast and one visible array, or a nonzero AIHWKit `gamma`;
- separate hardware bounds and device models for the slow array; or
- momentum and weight decay inside the two-array rule.

The code materializes the complete EqProp/DRN gradient and applies exact
floating-point tensor addition. It is therefore suitable for studying the
algorithmic accumulation/transfer schedule, but it should not be interpreted
as predicting a physical crossbar's write accuracy, noise, or timing.

## Checkpoint caveat

The auxiliary tensor, per-parameter optimizer-step counter, transfer cursor,
and transfer-event counter are stored in `TikiTakaOptimizer.state`. A PyTorch
`optimizer.state_dict()` includes that state.

The repository's usual `energy_fn.save(...)`, `network.save(...)`, and
`model.pt` paths save visible model parameters only. Those files are valid for
inference, but they do not contain the fast arrays or transfer schedule state.
Loading only such a model and resuming training creates zero-valued auxiliary
arrays and restarts the counters/cursors, so it is not an exact continuation
of the interrupted Tiki-Taka run.

For resumable training, save the visible model plus
`optimizer.state_dict()`, reconstruct the optimizer with the same
`update_pipeline` configuration, and then load its state. Also save the
learning-rate scheduler state when one is used. Exact continuation of runs
using random selection or probabilistic reset additionally requires preserving
the relevant PyTorch random-number-generator state.
