# Hardware-aware additive weight noise

`small_drn.v1` can train clean master weights while measuring each minibatch
at a temporary additive-Gaussian weight realization. This is a focused
forward-path robustness experiment; it is not a ReRAM device, programming
noise, finite-step BPTT, or deployment model.

## Configuration

The modifier belongs in the nested training mode:

```json
{
  "modes": {
    "train": {
      "weight_modifier": {
        "type": "add_normal",
        "parameters": {
          "std_dev": 0.06,
          "seed": 7,
          "noisy_evaluation": true,
          "scale_mode": "tensor_abs_max"
        }
      }
    }
  }
}
```

See
[`examples/small_drn/hardware_aware.json`](../examples/small_drn/hardware_aware.json)
for a complete CPU example.

The parameter object is strict:

- `std_dev` is required, finite, and non-negative.
- `seed` is optional and is either `null` or an integer in
  `[0, 2**64 - 1]`.
- `noisy_evaluation` is optional and defaults to `false`.
- `scale_mode` is optional and defaults to `tensor_abs_max`. It accepts
  `tensor_abs_max` or `output_channel_abs_max`.
- No other keys are accepted.

The modifier seed takes precedence over `runtime.seed`. If both are `null`,
the resolved seed is `torch.initial_seed()`. A zero standard deviation uses
the unmodified runtime path.

## Numerical behavior

Only unique dense and convolutional weights are eligible. Biases and pooling
weights stay clean. With `tensor_abs_max`, the sampled noise standard
deviation for each eligible tensor is

```text
std_dev * max(abs(clean_tensor))
```

With `output_channel_abs_max`, every output channel instead uses its own
clean absolute maximum. Dense DRN tensors have `(pre, post)` layout, so the
scale is taken over the pre-synaptic dimension; convolutional tensors use
one scale per leading output-channel dimension. This matches the
AIHWKit-Lightning `ADD_NORMAL_PER_CHANNEL` convention.

The temporary value is clamped to the parameter's physical bounds. Tensor
objects are modified and restored in place, so optimizers and parameter
catalog bindings retain their identities.

One draw spans a complete training minibatch: input assignment, free
equilibrium, cost measurement, and the backpropagation gradient phase. The
clean master weights are restored before the direct optimizer step.
Restoration also occurs when training raises an exception.

The currently advertised experimental combination is direct updates with
backpropagation. Other algorithm or update-backend combinations fail closed
until they are listed in the `small_drn.v1` capability matrix.

The adapter axis remains `none`; LoRA plus hardware-aware noise is not an
advertised combination.

## Evaluation and selection

When `noisy_evaluation` is enabled, each epoch performs one diagnostic
`validation_noisy` pass with a realization fixed across the held-out loader.
It runs before the authoritative clean `validation` pass. Running clean
evaluation last leaves clean settled layer state at the epoch checkpoint.

Clean validation alone controls checkpoint selection. `weights.pt` and the
`selected_weights` embedded in `resume.pt` always contain clean weights.
Results record the normalized modifier configuration, whether it was active,
the resolved seed, whether noisy evaluation was requested and executed, and
both `last_validation` and `last_noisy_validation`.

Training and evaluation noise use independent RNG streams. Enabling diagnostic
noisy evaluation therefore does not alter later training draws.

## Exact continuation

The epoch-boundary checkpoint stores the modifier's normalized configuration,
resolved seed, and per-device CPU `uint8` generator states for both the
training and evaluation streams. Loading validates the full payload before
replacing live generators, and the saved resolved seed wins on restore.

Resume capability is `exact` for the currently advertised direct-update
combination.
