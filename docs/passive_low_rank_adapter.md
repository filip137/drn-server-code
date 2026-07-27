# Passive low-rank adapter

The `passive_low_rank` model adapter adds a physical two-edge path in parallel
with one frozen dense base edge:

```text
                 frozen W
input x  ------------------------>  output y
    \                                  /
     \ trainable A        trainable B /
      \                              /
       ------ rank nodes z ----------
```

The resulting function has three layers in composition order: input `x`,
linear floating rank nodes `z`, and task output `y`. The base conductance
matrix `W` remains part of the model and every full checkpoint, but only the
input factor `A` and output factor `B` are exposed to EP and the optimizer.

The stable parameter catalog is:

| Key | Group and role | Trainable | Full checkpoint |
| --- | --- | --- | --- |
| `base.dense_weight.0` | `base`, `dense_weight` | no | yes |
| `adapter.input_factor.0` | `adapter`, `input_factor` | yes | yes |
| `adapter.output_factor.0` | `adapter`, `output_factor` | yes | yes |

## Passive circuit interpretation

Let `W` connect `x` directly to `y`, `A` connect `x` to `z`, and `B`
connect `z` to `y`. With the unit amplification required by this adapter,
their contribution to the energy is

\[
E(x,z,y)=
\frac{1}{2}\sum_{ij}W_{ij}(x_i-y_j)^2+
\frac{1}{2}\sum_{ik}A_{ik}(x_i-z_k)^2+
\frac{1}{2}\sum_{kj}B_{kj}(z_k-y_j)^2.
\]

In an unnudged linear equilibrium, KCL at the floating rank and output nodes
gives

\[
D_z z=A^\top x+B y,\qquad
D_y y=W^\top x+B^\top z,
\]

where

\[
D_z=\operatorname{diag}(A^\top\mathbf{1}+B\mathbf{1}),\qquad
D_y=\operatorname{diag}(W^\top\mathbf{1}+B^\top\mathbf{1}).
\]

Eliminating `z` produces the rank-bounded cross term
\(A D_z^{-1}B\), but it also produces the reciprocal output loading

\[
\operatorname{diag}(B^\top\mathbf{1})-B^\top D_z^{-1}B.
\]

The rank nodes are therefore part of the equilibrium solve. This is not the
usual digital LoRA update written as \(W+BA\), nor simply \(W+AB\) in the
input-by-output orientation used above. Replacing the branch by an algebraic
matrix addition would discard its KCL loading and generally change the
equilibrium.

All adapter entries are non-negative physical conductances. With zero output
initialization, `B` starts exactly at zero. The adapter is then output-neutral:
`y` initially has the same equilibrium as the frozen base model, while the
strictly positive connections in `A` keep every rank node well-defined. It is
not necessarily power-neutral, because the input-to-rank branch is already
present.

## Strict nested configuration

The checked-in configuration is
[`examples/small_drn/lora.json`](../examples/small_drn/lora.json). The adapter
and its two learning rates are nested under the versioned experiment schema:

```json
{
  "model": {
    "dims": [4, 2],
    "voltage_amp": 1.0,
    "current_amp": 1.0,
    "adapter": {
      "type": "passive_low_rank",
      "parameters": {
        "rank": 2,
        "input_factor_gain": 0.01,
        "input_factor_min": 1e-7,
        "conductance_max": 1.0,
        "output_factor_init": "zero"
      }
    }
  },
  "modes": {
    "train": {
      "algorithm": "ep",
      "learning_rates": [0.01, 0.01],
      "bias_learning_rates": [],
      "weight_modifier": {
        "type": "none",
        "parameters": {}
      },
      "update_backend": {
        "type": "direct",
        "parameters": {}
      }
    }
  }
}
```

The complete example also supplies all other required model, diode, solver,
data, and runtime fields. Adapter parameters are strict: unknown keys and
missing required keys are rejected.

| Field | Constraint and meaning |
| --- | --- |
| `rank` | Positive integer number of physical floating rank nodes. |
| `input_factor_gain` | Positive finite initialization gain for `A`. |
| `input_factor_min` | Positive finite lower clamp for `A`; must be less than `conductance_max`. |
| `conductance_max` | Positive finite common upper clamp for `A` and `B`. |
| `output_factor_init` | Exactly `"zero"` or `"off_conductance"`. |
| `output_off_conductance` | Conditionally required for `"off_conductance"`; it must be positive, finite, and less than `conductance_max`. |

For `"zero"`, `output_off_conductance` must be omitted or `null`, and the
resolved configuration records it canonically as `null`. `B` is initialized
to zero and may remain at zero. For `"off_conductance"`, every entry of `B`
starts at `output_off_conductance`, which also becomes its lower clamp. That
mode intentionally perturbs the base output equilibrium.

The declared model must have exactly two positive one-dimensional widths,
`[input_width, output_width]`, one base `weight_gains` entry, no convolution or
pooling pipeline, and both `voltage_amp` and `current_amp` equal to `1.0`.
`learning_rates` must contain exactly `[lr_A, lr_B]`, and
`bias_learning_rates` must be empty.

## Supported training combinations

Only these two `small_drn.v1` combinations are registered:

| Model adapter | Modifier | Algorithm | Update backend |
| --- | --- | --- | --- |
| `passive_low_rank` | `none` | `ep` | `direct` |
| `passive_low_rank` | `none` | `ep` | ideal-tensor `tiki_taka` |

The direct backend requires an empty parameter object. For Tiki-Taka,
`aihwkit_preset` must be omitted or `null`; the remaining ideal-tensor
Tiki-Taka settings follow
[`docs/tiki_taka_gradient_accumulation.md`](tiki_taka_gradient_accumulation.md).
Backpropagation, parameter modifiers such as `add_normal`, and native AIHWKit
Tiki-Taka are not validated combinations and fail closed during experiment
resolution.

## Initialization and named checkpoints

Adapter training must receive exactly one initialization source:

```bash
# Start from a named base-only artifact; A and B use the config initialization.
python -m ebl train \
  --config examples/small_drn/lora.json \
  --base-weights named-base.pt \
  --output-dir runs

# Start from a complete named adapted artifact containing W, A, and B.
python -m ebl train \
  --config examples/small_drn/lora.json \
  --weights named-full.pt \
  --output-dir runs

# Continue a prior adapted run from its complete epoch boundary.
python -m ebl train \
  --config examples/small_drn/lora.json \
  --resume runs/<prior-run>/checkpoints/resume.pt \
  --output-dir runs
```

Supplying none or more than one of `--weights`, `--base-weights`, and
`--resume` is rejected. Their semantics are intentionally different:

- `--weights` requires the complete named catalog `W`, `A`, and `B`.
- `--base-weights` requires only `base.dense_weight.0`; loading it does not
  change the independently initialized adapter factors.
- `--resume` restores the full epoch-boundary training state.

The selected `checkpoints/weights.pt` written by an adapted run is always a
full named artifact. It can initialize training with `--weights` and is the
artifact required by `linspace` and `validate`. A base-only artifact is a
separately scoped artifact, not a partial full checkpoint, and there is no
adapter-only checkpoint format.

Positional legacy files are never auto-detected. Convert a legacy base tensor
explicitly before adapter training:

```bash
python -m ebl checkpoint import-legacy \
  --config examples/small_drn/lora.json \
  --source legacy-base.pt \
  --kind base \
  --output named-base.pt
```

`--kind base` declares that the positional source contains only `W` and emits
a named base artifact. Use `--kind full` only when the legacy positional
source really contains the complete adapted catalog.

## Exact continuation

For direct updates and ideal-tensor Tiki-Taka, the adapter runtime advertises
exact resume. The epoch-boundary checkpoint includes all named model
parameters, optimizer and Tiki-Taka state when applicable, progress and
selected weights, random-number state, named data-loader-generator state, and
the carried tensor state of every composed layer: input `x`, rank `z`, and
output `y`.

Those layer tensors matter because training deliberately carries the settled
state between operations. `--weights` and `--base-weights` restore parameters
only; they are initialization paths, not exact continuation. Resume also
requires the numerical configuration to match, except that
`modes.train.num_epochs` may extend the training horizon.

## Current limitations

This focused extension does not support hidden dense stages, convolution,
pooling, amplified edges, signed or differential adapter factors, a trainable
base matrix, adapter-only artifacts, hardware-aware modifiers, backpropagation,
or native AIHWKit tiles.

It also does not port the legacy `small_network` recovery experiment layer:
there are no recovery-baseline configuration fields, `recovery_metrics.json`,
post-training recovery score, rank-sweep helper, or automatic algebraic
collapse of the passive branch into an “effective” dense matrix. Use the
normal versioned metrics and explicit cross-worktree campaigns for experimental
comparisons.
