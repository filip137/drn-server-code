# Passive layerwise low-rank recovery of a two-edge DRN

The `passive_layerwise_low_rank` adapter adds one physical rank-node branch
in parallel with each dense edge of a DRN containing one hidden layer:

```text
             +-- A1 -- r1 -- B1 --+
             |                    |
input x -----+------- W1 ---------+----- hidden h
                                          |
             +-- A2 -- r2 -- B2 --+       |
             |                    |       |
output y ----+------- W2 ---------+-------+
```

`W1`, `W2`, `A1`, `B1`, `A2`, and `B2` are six distinct non-negative
conductance matrices. The two base matrices are programmed once with
AIHWKit's Wan-2022 ReRAM model. The four adapter matrices are ideal,
noise-free conductance arrays, but they remain part of the passive circuit and
are trained through equilibrium propagation.

There is deliberately no synthesized matrix
`clip(W_device + alpha/r * A @ B)`. The rank nodes are explicit equilibrium
variables, and each conductance matrix is clipped independently:

```text
W1, W2       -> [model.weight_min, model.weight_max]
A1, A2       -> [input_factor_min, conductance_max]
B1, B2       -> [output_factor_min, conductance_max]
```

For `output_factor_init="zero"`, the corresponding output-factor lower bound
is zero. For `output_factor_init="off_conductance"`, the configured off
conductance is both its initial value and lower bound.

## Circuit topology and initialization

The model contains five node layers in solver order:

```text
[input, rank1, hidden, rank2, output]
```

The hidden layer retains its configured diode nonlinearity and bias. Both rank
layers and the output layer are linear. The two frozen base edges use the
configured logical-layer voltage and current amplification. The internal
rank branches remain passive unity-gain connections. Logical amplification
depth is explicit and does not depend on process-global layer names.

Zero-initializing `B1` makes the first branch output-neutral because `A1`
connects only the clamped input sources to the floating first rank nodes.
For the second branch, however, `A2` is attached to the free hidden nodes.
Even with `B2=0`, those shared floating rank nodes can load and couple hidden
nodes. Consequently, a physical two-edge adapter is not guaranteed to be
exactly neutral at initialization. The recorded pre-training validation metric
captures this circuit-level effect.

## Stable per-edge configuration

The adapter requires exactly three declared DRN layer widths and two
stable-keyed branch configurations:

```json
{
  "type": "passive_layerwise_low_rank",
  "parameters": {
    "layers": {
      "base.dense_weight.0": {
        "rank": 4,
        "input_factor_gain": 0.01,
        "input_factor_min": 1e-7,
        "conductance_max": 1.0,
        "output_factor_init": "zero",
        "device_noise": {
          "type": "aihwkit_reram_wan2022",
          "programming_seed": 17,
          "g_max_us": 40.0,
          "drn_conductance_at_g_max": 0.1,
          "noise_scale": 1.0,
          "t_inference_seconds": 1.0
        }
      },
      "base.dense_weight.1": {
        "rank": 4,
        "input_factor_gain": 0.01,
        "input_factor_min": 1e-7,
        "conductance_max": 1.0,
        "output_factor_init": "zero",
        "device_noise": {
          "type": "aihwkit_reram_wan2022",
          "programming_seed": 29,
          "g_max_us": 40.0,
          "drn_conductance_at_g_max": 0.1,
          "noise_scale": 1.0,
          "t_inference_seconds": 1.0
        }
      }
    }
  }
}
```

The base edges may use different programming seeds, calibrations, noise
scales, or inference ages. The factors may independently use different ranks
and conductance bounds.

Both base matrices are programmed only when training starts from
`--base-weights`. Full `--weights` and epoch-boundary `--resume` checkpoints
already contain the persistent device realization and are not reprogrammed.
The four factor matrices are never passed through the Wan-2022 model.

## Parameters and checkpoints

The frozen base group is:

```text
base.dense_weight.0
base.dense_weight.1
base.bias.0
```

The trainable conductance arrays are:

```text
adapter.input_factor.0
adapter.output_factor.0
adapter.input_factor.1
adapter.output_factor.1
```

After every optimizer step, each factor clamps its own parameter tensor to its
own physical interval.

## Running

Use a clean named base checkpoint from the same `[input, hidden, output]`
topology:

```bash
python -m ebl train \
  --config examples/small_drn/passive_layerwise_lora_reram_wan2022_digits.json \
  --base-weights /path/to/clean-two-edge-base.pt \
  --output-dir /path/to/new-output-root
```

The registered combination is:

| Model adapter | Modifier | Algorithm | Update backend |
| --- | --- | --- | --- |
| `passive_layerwise_low_rank` | `none` | `ep` | `direct` |

The initialization metric reports each programmed base edge, aggregate
programming error, and held-out accuracy before any factor update.

## Recovery measurements

For every run, retain clean accuracy, programmed pre-training accuracy,
best/final recovery accuracy, and the recovered fraction of the
noise-induced accuracy gap:

```text
gap_recovery = (accuracy_recovered - accuracy_noisy)
               / (accuracy_clean - accuracy_noisy)
```

Because a second physical branch can load the hidden layer even before
training, also compare the clean base model against a zero-noise model with
the passive branches attached. This separates adapter-insertion error from
ReRAM programming error.
