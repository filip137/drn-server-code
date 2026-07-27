# Tiki-Taka gradient accumulation

The optional Tiki-Taka update backend routes gradients produced by EqProp/DRN
through an auxiliary ("fast") tensor before changing the DRN's visible
("slow") parameters. The DRN still computes one complete parameter gradient
tensor per minibatch. Tiki-Taka changes only how that tensor is accumulated
and applied.

The implementation is in `training/tiki_taka.py`. It has two backends:

- the default ideal tensor emulator of the two-crossbar schedule; and
- an optional native AIHWKit `TransferCompound` backend that performs pulsed
  device writes and transfers, then exposes the realized slow state to the DRN.

## Selecting the update pipeline

For the versioned `small_drn.v1` experiment, select the backend at
`modes.train.update_backend`. The direct SGD path is:

```json
{
  "modes": {
    "train": {
      "update_backend": {
        "type": "direct",
        "parameters": {}
      }
    }
  }
}
```

The direct backend requires an empty parameter object. To enable the auxiliary
array, use `"type": "tiki_taka"`. The values below are illustrative:

```json
{
  "modes": {
    "train": {
      "learning_rates": [0.8, 0.3],
      "update_backend": {
        "type": "tiki_taka",
        "parameters": {
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
    }
  }
}
```

Unknown keys and values of the wrong type are rejected. Tiki-Taka currently
requires optimizer `momentum` and `weight_decay` both to be zero because their
placement in the two-array update rule is undefined.

The still-live MNIST research tools retain their older `update_pipeline`
locations: `labs/mnist_train.py` reads it at the top level, while
`labs/mnist_monitor_training.py` and
`plotting_functions/visualize_training_steps.py` read
`training.update_pipeline`.

## Exact ideal-tensor update rule and signs

This section defines the default backend used when `aihwkit_preset` is null.

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

The ideal backend does not clamp the visible tensor. Existing DRN trainers call
each parameter's `clamp_()` after the optimizer step, so any configured DRN
parameter bounds are applied after the additive transfer. Fast array bounds are
separate and are controlled only by `fast_weight_min` and `fast_weight_max`.

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
| `aihwkit_preset` | `null` | Null selects the ideal backend. A supported AIHWKit Tiki-Taka preset selects native pulsed tiles. |
| `aihwkit_conductance_min` | `null` | Optional non-negative lower DRN conductance bound for the affine device-state mapping; defaults to the parameter clamp. |
| `aihwkit_conductance_max` | `null` | Optional upper DRN conductance bound; defaults to the finite parameter clamp and must exceed the lower bound. |
| `aihwkit_construction_seed` | `null` | Optional positive base seed for reproducible device construction. Each tile/device receives a distinct offset; AIHWKit reserves zero for nondeterministic seeding. |

These defaults define a Tiki-Taka pipeline, but they do not make it equivalent
to direct SGD. In particular, only one logical column is transferred per
minibatch by default, and the fast slice is not reset after transfer.

## Native AIHWKit pulsed backend

Set `aihwkit_preset` to route every accumulated weight through a native
AIHWKit tile. AIHWKit is imported lazily, so direct SGD and the ideal backend
do not require that optional package.

The supported presets are:

- `TikiTakaIdealizedPreset`
- `TikiTakaReRamESPreset`
- `TikiTakaReRamSBPreset`
- `TikiTakaCapacitorPreset`
- `TikiTakaEcRamPreset`
- `TikiTakaEcRamMOPreset`

The ReRAM, capacitor, and ECRAM presets include their AIHWKit device
granularity, asymmetry, state dependence, device-to-device and cycle-to-cycle
variation, bounds, write noise, transfer readout, and pulsed update behavior.
The common transfer fields in `update_pipeline` override the corresponding
fields on the selected preset.

For example:

```json
{
  "update_pipeline": {
    "type": "tiki_taka",
    "fast_lr": 1.0,
    "transfer_every": 1,
    "n_reads_per_transfer": 1,
    "transfer_lr": 1.0,
    "scale_transfer_lr": true,
    "aihwkit_preset": "TikiTakaReRamESPreset",
    "aihwkit_conductance_min": 1e-7,
    "aihwkit_conductance_max": 1.0,
    "aihwkit_construction_seed": 7
  }
}
```

AIHWKit pulsed training devices store normalized, signed device states, not raw
siemens. The DRN, in contrast, uses non-negative conductances. For each slow
crosspoint, the backend reads AIHWKit's realized per-cell bounds `q_min` and
`q_max` and uses the fixed mapping

```text
G = G_min + (q - q_min) / (q_max - q_min) * (G_max - G_min).
```

The result is clipped to `[G_min, G_max]`. Unless explicitly configured,
`G_min` and `G_max` come from the DRN parameter clamp. Consequently, the
trainer's normal post-step `clamp_()` is a no-op and cannot desynchronize the
tile. If the configured `G` values are expressed in siemens, this boundary map
is also the explicit normalized-state-to-siemens calibration; AIHWKit presets
do not supply such a calibration themselves.

To preserve the existing DRN learning-rate scale, the gradient written to the
normalized device is divided by the local mapping slope
`dG/dq = (G_max-G_min)/(q_max-q_min)`. For a logical full gradient
`H[out,in]`, the backend makes one native AIHWKit update call per minibatch
with an exact rank-one batch satisfying

```text
d.T @ x = H.
```

It uses `x=I, d=H.T` or `x=H, d=I`, whichever has the smaller batch dimension.
AIHWKit supplies the descent sign, pulse generation, fast write, transfer, and
slow-device response. The backend also calls `post_update_step()` once per
minibatch so configured diffusion and decay are applied.

`optimizer.aihwkit_device_state(parameter.state)` returns cloned AIHWKit hidden
parameters. `hidden_weights_0` and `hidden_weights_1` are the apparent fast and
slow states. Presets with write noise may also expose
`persistent_weights_0/1`, which are the underlying programmed states.

The package is intentionally not added to the repository's pinned environment:
the installed AIHWKit 1.1 package requires a newer PyTorch than the main
environment currently pins. Use a compatible AIHWKit environment. A CPU-only
AIHWKit build requires the DRN run itself to use `--device cpu`.

### MNIST smoke test

The checked-in smoke configuration uses real MNIST samples, the measured ReRAM
ES preset, two training batches, and one test batch:

```bash
python labs/mnist_train.py \
  --config labs/configs/mnist_tiki_taka_aihwkit_smoke.json \
  --output-dir /tmp/mnist_tiki_taka_aihwkit_smoke
```

`max_test_batches` is separate from `max_batches`, so smoke runs do not
silently evaluate all 10,000 test examples.

For a complete one-epoch run with batch size 128, use the uncapped companion
configuration:

```bash
python labs/mnist_train.py \
  --config labs/configs/mnist_tiki_taka_aihwkit_epoch1_batch128.json \
  --output-dir /tmp/mnist_tiki_taka_aihwkit_epoch1_batch128
```

### Perfect-diode reference benchmark

The direct-FP32 reference configuration reproduces a previously validated
perfect-diode MNIST operating point: one 100-unit hidden layer, 20 paired
outputs, batch size 16, four coordinate-descent iterations, and centered
EqProp with nudging 0.05. It intentionally does not use Tiki-Taka or AIHWKit:

```bash
python labs/mnist_tests.py \
  --config labs/configs/mnist_perfect_diode_reference_epoch1.json \
  --model-key drn-xs \
  --num-iterations 4 \
  --seed 0 \
  --output-dir /tmp/mnist_perfect_diode_reference_epoch1_seed0 \
  train-stats \
  --num-epochs 1 \
  --record-statistics
```

The empty `--record-statistics` value disables residual-current collection for
the timing run. The historical seed-0 result was 95.40% test accuracy after
one epoch; the reproduced fixed-four-sweep CPU run also reached 95.40%.

### Initializing Tiki-Taka from direct FP32 conductances

`labs/mnist_train.py` accepts a direct DRN `model.pt` through
`--initial-weights` or the top-level `initial_weights` config field. The
checkpoint must be a list or tuple with one floating-point tensor per DRN
parameter. The loader validates the tensor count, shapes, finite values, and
DRN bounds before copying any tensor in place. Loading happens after the DRN is
moved to its target device and before the optimizer is constructed, so the
native AIHWKit tiles are programmed from those conductances.

The matched one-epoch configuration uses batch size 128, the perfect-diode
operating point above, and the ReRAM ES Tiki-Taka preset:

```bash
python labs/mnist_train.py \
  --config labs/configs/mnist_tiki_taka_aihwkit_from_fp32_epoch1_batch128.json \
  --initial-weights /path/to/direct-fp32/model.pt \
  --output-dir /tmp/mnist_tiki_taka_aihwkit_from_fp32_epoch1_batch128
```

The run metadata records the checkpoint path and SHA-256, an evaluation before
AIHWKit programming, source-to-realized conductance errors, persistent
programming errors when the preset exposes them, the initial fast-array
magnitude, an evaluation immediately after programming, and the final
one-epoch metrics. The final visible DRN tensors are saved as `model.pt`.

For the reproduced 95.40% direct checkpoint (SHA-256
`5988bdd6f2ae068da0f1bc704ae0922f822ba73d48f762fbb5d4cb272f28686f`),
the batch-128 CPU run produced:

| Stage | MNIST test accuracy |
| --- | ---: |
| Loaded direct FP32 conductances | 95.40% |
| ReRAM ES apparent conductances before training | 77.04% |
| After one Tiki-Taka epoch | 85.74% |

The underlying persistent slow states reproduced the two source weight tensors
with maximum conductance errors of `5.12e-8` and `4.47e-8`; the fast arrays
started exactly at zero. The lower apparent accuracy is not a checkpoint
loading error: `TikiTakaReRamESPreset` applies write noise when exposing its
apparent slow conductances. This run took 40.21 seconds on CPU.

## Correspondence with AIHWKit

The design follows the pipeline described in
`/home/filip/aihwkit/DEVICE_WRITE_PIPELINE.md`:

```text
EqProp/DRN complete gradient tensor
  -> ideal tensor write or native AIHWKit pulsed write to fast array A
  -> periodic logical row/column selection
  -> ideal additive or native pulsed transfer
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
`TransferCompound` controls. The schedule is defined in integer
optimizer/minibatch steps and therefore requires `units_in_mbatch=true`.

## Ideal-emulator limitations

The default ideal backend deliberately does not model:

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

The native backend removes those ideal-device assumptions, but it still starts
from an already materialized EqProp gradient rather than the original retained
activation/error pair. Its exact outer-product factorization therefore does not
reproduce pulse correlations from an ordinary backpropagation layer. It also
uses an explicit affine conductance calibration, not an AIHWKit inference-time
conductance converter.

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
`update_pipeline` configuration, parameter shapes/layouts, and conductance
bounds, and then load its state. These structural properties and the AIHWKit
version are checked before optimizer state is mutated. The native backend stores
visible weights, all hidden device parameters, AIHWKit's extra transfer state,
and the tile learning rate. For devices without persistent write noise, the
custom restore order preserves the fast array that AIHWKit 1.1's ordinary tile
restore would reset. Also save the learning-rate scheduler state when one is
used.

AIHWKit does not expose the native pulsed tile's internal stochastic RNG stream
through this interface. A native checkpoint restores device values and transfer
cursors as far as the public API permits, but the next stochastic pulse
realization is not guaranteed to be bit-for-bit identical. The
`TikiTakaReRamESPreset` and `TikiTakaReRamSBPreset` redraw apparent write noise
while loading hidden parameters; AIHWKit 1.1's compound setter can therefore
leave both their apparent and persistent values different from the snapshot.
`load_state_dict()` rejects checkpoints for those two presets before mutating
the optimizer rather than silently performing a lossy resume. Their saved
visible model remains usable for inference or as a fresh-training
initialization. The ideal backend can continue exactly when the relevant
PyTorch random-number-generator state is also preserved.
