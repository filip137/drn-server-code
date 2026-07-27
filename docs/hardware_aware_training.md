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
          "noisy_evaluation": true
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
- No other keys are accepted.

The modifier seed takes precedence over `runtime.seed`. If both are `null`,
the resolved seed is `torch.initial_seed()`. A zero standard deviation uses
the unmodified runtime path.

## Numerical behavior

Only unique dense and convolutional weights are eligible. Biases and pooling
weights stay clean. For each eligible tensor, the sampled noise standard
deviation is

```text
std_dev * max(abs(clean_tensor))
```

The temporary value is clamped to the parameter's physical bounds. Tensor
objects are modified and restored in place, so optimizers and parameter
catalog bindings retain their identities.

One draw spans a complete training minibatch: input assignment, free
equilibrium, cost measurement, and the EP or backpropagation gradient phase.
The clean master weights are restored before the direct or Tiki-Taka optimizer
step. Restoration also occurs when training raises an exception.

The supported experimental combinations are:

- direct updates with equilibrium propagation;
- direct updates with backpropagation;
- Tiki-Taka updates with equilibrium propagation;
- Tiki-Taka updates with backpropagation.

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

Resume capability remains:

- `exact` for direct updates and ideal Tiki-Taka;
- `stateful_nondeterministic` for native AIHWKit Tiki-Taka, whose internal
  stochastic stream is not fully exposed.

See
[`tiki_taka_gradient_accumulation.md`](tiki_taka_gradient_accumulation.md)
for the independent update-backend configuration.
