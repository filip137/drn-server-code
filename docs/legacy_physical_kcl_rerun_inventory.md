# Legacy physical-KCL rerun inventory

Updated: 2026-08-25

## Scope and reason

This inventory covers active ordinary-MNIST manuscript evidence for the
`legacy` amplification setting `(voltage_amp, current_amp) = (4, 0.25)`.
The former interaction energy multiplied the receiving-node voltage by
`current_amp`.  It therefore minimized a coherent but differently
parameterized energy, rather than the physical KCL energy stated in the
manuscript.  Baseline `(1, 1)` and ours `(4, 1)` are unchanged because their
`current_amp` is one.

Historical medium-affine, hard-sigmoid, pooling, and superseded exploratory
studies are excluded unless the manuscript later elects to cite them.  The
official MNIST test split remains sealed.

## Primary training arms to replace

The active manuscript contains 21 affected legacy training arms.  Each
replacement starts from fresh matched initialization and retains its recorded
seed, data order, optimizer, epoch budget, and parameter-wise learning-rate
vector unless a preceding corrected-model selection gate changes the final
paper contract.

| Block | Architecture | Optimizer/condition | Affected arms | Source authority |
|---|---|---|---:|---|
| Clean wide BPTT | Conv1 | SGD, Adam | 2 | `configs/conv/perfectdiode_conv123_zero_bias_ordinary_mnist_seed0_20260805_v1/conv1/04_legacy_sgd_bias_zero_seed0.json`, `05_legacy_adam_bias_zero_seed0.json` |
| Clean wide BPTT | Conv2 | SGD, Adam | 2 | Same study, Conv2 legacy configs `04` and `05` |
| Clean wide BPTT | Conv3 | SGD, Adam | 2 | Same study, Conv3 legacy configs `04` and `05` |
| Clean centered EqProp | Conv1 | Adam, no endpoint noise | 1 | `perfectdiode_conv123_zero_bias_adam_eqprop_one_decade_ordinary_mnist_10_30_30ep_seed0_20260816_v1.json` |
| Clean centered EqProp | Conv2 | Adam, no endpoint noise | 1 | Same study authority |
| Clean centered EqProp | Conv3 | Adam, no endpoint noise | 1 | Same study authority |
| Noisy centered EqProp | Conv1 | Adam, `sigma_v=5e-4` | 1 | `perfectdiode_conv13_zero_bias_adam_eqprop_one_decade_sigma_5em4_ordinary_mnist_10_30ep_seed0_20260817_v1.json` |
| Noisy centered EqProp | Conv2 | Adam, `sigma_v=5e-4` | 1 | `perfectdiode_conv2_zero_bias_adam_eqprop_one_decade_sigma_5em4_ordinary_mnist_30ep_seed0_20260817_v1.json` |
| Noisy centered EqProp | Conv3 | Adam, `sigma_v=5e-4` | 1 | Conv1/Conv3 noisy study authority above |
| Fixed-initialization bounded BPTT | Conv1 | Adam, `Gmax={1e-4,5e-4,1e-3}` | 3 | `perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/conv1/*legacy*.json` |
| Fixed-initialization bounded BPTT | Conv2 | Adam, `Gmax={1e-4,5e-4,1e-3}` | 3 | Same study, Conv2 legacy configs |
| Fixed-initialization bounded BPTT | Conv3 | Adam, `Gmax={1e-4,5e-4,1e-3}` | 3 | Same study, Conv3 legacy configs |
| **Total** |  |  | **21** |  |

The bounded-Kaiming raw-LR transfer adds two supporting legacy arms (Conv1 and
Conv2) if that diagnostic remains in the manuscript discussion.  They are not
part of the 21 primary table cells.

## Corrected-model gates and derived analyses to repeat

The following legacy-only portions must be recomputed before corrected results
can replace manuscript evidence:

1. Conv1/2/3 legacy operational `T/K` checks and the six wide legacy LR
   selection surfaces (architecture by SGD/Adam).  The immediate same-LR
   repeats below are controlled diagnostics, not a claim that the old rates
   remain selected for the corrected model.
2. Conv1/2/3 legacy EqProp beta-gradient qualification, including the clean
   tiers, endpoint-noise tiers, and the one-decade state-displacement rows.
3. The legacy columns of the manuscript beta-cosine and displacement figures.
   Baseline and ours source rows can be retained.
4. The nine legacy bounded final-weight distributions and all table deltas or
   narrative statements derived from them.  Baseline and ours checkpoints can
   be retained.
5. Any official-test evaluation only after the corrected inclusion set,
   selected checkpoints, and all preceding gates are frozen.  No official-test
   replay is authorized by this inventory.

## Conv1 first-wave diagnostic

The first wave isolates two questions in order:

1. Replay the old Conv1 legacy SGD and Adam best/final checkpoints unchanged
   under the corrected physical equilibrium.  Compare the full 5,000-example
   validation accuracy and voltage/score scales with the recorded source
   accuracy.  A descriptive absolute change of `0.5` percentage points or
   more is predeclared as material.
2. Retrain the clean wide Conv1 legacy SGD and Adam arms for 10 epochs with
   their exact existing rates:
   - SGD: `[2.05753e-4, 4.76259e-5, 0]`;
   - Adam: `[2.88917e-4, 3.64618e-5, 0]`.

Both use ordinary MNIST, seed/order `0`, exact-zero bias, `T=K=4`, batch size
16, validation batch size 64, and no official-test access.  The only intended
model change is the physical-KCL interaction energy.
