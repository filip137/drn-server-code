# CONTAMINATED OLD-WORKTREE SNAPSHOT

This file was copied from the old worktree snapshot `5e2d76fc`. The old worktree included simulations affected by the `adaptive_equilibrium=True` problem. Treat any result interpretation, selected protocol, or run decision here as historical context only unless revalidated with the corrected fixed-equilibrium protocol.

# Hard-Sigmoid Experiments

Updated: 2026-07-01

## Learning `v_off`

Motivation: the network trains better when the hard-sigmoid saturation boundary is trainable. Fixed `v_off=4.0` often leaves later convolutional layers in a poor operating point, especially with little or no later-layer saturation, so the added depth may behave too linearly or with weak training signal. Learning layer-wise `v_off` lets the model move each layer's transition interval during training instead of depending on one hand-picked fixed boundary.

The clearest completed result so far is Conv2, trainable `v_off` only, fixed amplification `mnist_bp_amp_v2_c1`, target sat30, `K=6`, 50 epochs:

| Run | Strides | Paddings | Initial `v_off` | Best / final test | Learned `v_off` |
|---|---:|---:|---:|---:|---:|
| Conv2 `mnist_bp_amp_v2_c1` | `[2, 2]` | `[0, 0]` | `4.0` | `97.94% / 97.94%` | `[8.4189, 3.2176]` |

Final saturation on 256 training samples with the final checkpoint and `K=6`:

| Threshold used for saturation check | L1 | L2 | All hidden |
|---|---:|---:|---:|
| Learned layer-wise `v_off` | `17.66%` | `8.73%` | `14.99%` |
| Fixed `[-4, 4]` diagnostic | `47.51%` | `0.00%` | `33.32%` |

The learned-threshold check splits as L1 low/high `8.63% / 9.03%` and L2 low/high `5.84% / 2.89%`. The important point is that learning `v_off` made the later layer nontrivially saturated under its learned boundary, while the fixed `[-4, 4]` diagnostic still shows L2 at `0.00%`.

## Padded Conv1/Conv2 Follow-up

Run a padded hard-sigmoid control for Conv1 and Conv2 using the previously selected target-sat30 `input_gain` and LR rows that performed well. Sat50 is deliberately excluded; this follow-up is sat30 only.

Each row runs both fixed and trainable `v_off`:

- Fixed mode: `v_min=-4.0`, `v_max=4.0`.
- Trainable mode: initial `v_off=4.0`, `trainable_v_off=true`, `v_off_min=0.0`.

| Depth | Amp | Strides | Paddings | K | Epochs | Input gain | LR |
|---:|---|---:|---:|---:|---:|---:|---:|
| Conv1 | `v1/c1` | `[2]` | `[1]` | `4` | `10` | `65.65764617919922` | `0.00548298668852` |
| Conv1 | `v4/c1` | `[2]` | `[1]` | `4` | `10` | `18.72927474975586` | `0.0192212461406` |
| Conv2 | `v1/c1` | `[2, 2]` | `[1, 1]` | `6` | `30` | `221.06837463378906` | `0.0260552872365` |
| Conv2 | `v4/c1` | `[2, 2]` | `[1, 1]` | `6` | `30` | `158.0352325439453` | `0.00455594609132` |

Jean Zay launcher:

`/home/filip/server_code/experiments/run_mnist_bp_conv12_r3_hardsigmoid_pad1_sat30_voff_trainability_jeanzay.slurm`

Planned Jean Zay output root:

`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv12_s2s2_pad1_hardsigmoid_sat30_v1c1_v4c1_voff_trainability_k4k6_seed0_10_30epoch`

### Status 2026-06-30 18:10 CEST

Jean Zay array `1136832` is active. Conv1 tasks `0-3` completed successfully; Conv2 tasks `4-7` are running.

| Depth | Amp | Mode | State | Best / final test | Best epoch | Learned `v_off` |
|---:|---|---|---|---:|---:|---:|
| Conv1 | `v1/c1` | fixed | completed | `96.51% / 96.08%` | `9` | n/a |
| Conv1 | `v1/c1` | trainable | completed | `96.52% / 96.12%` | `9` | `4.6808` |
| Conv1 | `v4/c1` | fixed | completed | `89.40% / 88.67%` | `6` | n/a |
| Conv1 | `v4/c1` | trainable | completed | `92.02% / 91.29%` | `9` | `1.6636` |

Interim Conv2 log status at epoch `9/30` test:

| Depth | Amp | Mode | State | Latest logged test |
|---:|---|---|---|---:|
| Conv2 | `v1/c1` | fixed | running | `95.53%` |
| Conv2 | `v1/c1` | trainable | running | `95.81%` |
| Conv2 | `v4/c1` | fixed | running | `96.10%` |
| Conv2 | `v4/c1` | trainable | running | `96.10%` |

No task failures are visible. The only non-empty stderr content so far is module loading plus the standard PyTorch `torch.load(weights_only=False)` warning after completed Conv1 checkpoints.

Important correction: the completed Conv1 `v4/c1` rows above should not be treated as the intended "previously performed well" comparison. They used the per-amplification target-sat30 calibration row `input_gain=18.72927474975586`, LR `0.0192212461406`, from the unpadded calibration target CSV. The earlier strong Conv1 `v4/c1` hard-sigmoid row used `input_gain=65.65764617919922`, LR `0.00548298668852`, and reached about `97.08% / 97.08%`. So the low `89.40%` fixed and `92.02%` trainable results are a mismatched-hyperparameter run, not evidence that `v4/c1` or trainable `v_off` is intrinsically bad.

### Completion 2026-07-01 11:23 CEST

Jean Zay array `1136832` completed all eight tasks with exit code `0:0`; all eight `metrics.json` files are present. No task failures are visible. Stderr contains only module loading plus the standard PyTorch `torch.load(weights_only=False)` warning.

| Depth | Amp | Mode | Best / final test | Best epoch | Learned `v_off` |
|---:|---|---|---:|---:|---:|
| Conv1 | `v1/c1` | fixed | `96.51% / 96.08%` | `9` | n/a |
| Conv1 | `v1/c1` | trainable | `96.52% / 96.12%` | `9` | `4.6808` |
| Conv1 | `v4/c1` | fixed | `89.40% / 88.67%` | `6` | n/a |
| Conv1 | `v4/c1` | trainable | `92.02% / 91.29%` | `9` | `1.6636` |
| Conv2 | `v1/c1` | fixed | `96.23% / 96.13%` | `26` | n/a |
| Conv2 | `v1/c1` | trainable | `97.03% / 97.03%` | `29` | `[12.0603, 3.4755]` |
| Conv2 | `v4/c1` | fixed | `97.29% / 97.12%` | `29` | n/a |
| Conv2 | `v4/c1` | trainable | `97.34% / 97.10%` | `29` | `[3.7878, 4.0000]` |

Read:

- Conv2 `v1/c1` shows a clear trainable-`v_off` gain over its fixed control: `+0.80 pp` best and `+0.90 pp` final.
- Conv2 `v4/c1` is essentially tied: `+0.05 pp` best and `-0.02 pp` final.
- Conv1 `v1/c1` is essentially tied.
- Conv1 `v4/c1` remains the mismatched-hyperparameter row noted above and should be rerun with the earlier strong operating point before interpreting it.

Loss-stagnation check on job `1136832`:

- The bad Conv1 `v4/c1` fixed row did not clearly stagnate; test loss still decreased from `0.16323` at epoch 8 to `0.16124` at epoch 10, but accuracy stayed poor because the operating point was wrong.
- The bad Conv1 `v4/c1` trainable row was still improving in loss: `0.14297 -> 0.13952 -> 0.13735` over epochs `8-10`.
- Conv2 `v1/c1` trainable was still improving at the end: `0.07392 -> 0.07356 -> 0.07231` over epochs `28-30`, so it did not look saturated/stalled.
- Conv2 `v4/c1` fixed and trainable were near a plateau by epochs `29-30`; both changed by only about `0.0014` in test loss over the last three epochs.

### Learned Conductance Scale 2026-07-01

Computed from `weights_final.npz` for completed Jean Zay array `1136832`. The conductance tensors are the saved `ConvWeight_*` and `DenseWeight_*` arrays; `Bias_*` and `HardSigmoidVOff_*` are excluded here. These are raw learned conductance parameters, not the amp-factor-scaled energy coefficients.

| Depth | Amp | Mode | Layer | Mean | Median | P90 | P99 | Max |
|---:|---|---|---|---:|---:|---:|---:|---:|
| Conv1 | `v1/c1` | fixed | `ConvWeight_0` | `0.0652` | `0.0216` | `0.1935` | `0.2801` | `0.3453` |
| Conv1 | `v1/c1` | fixed | `DenseWeight_0` | `3.49e-3` | `2.91e-4` | `0.0104` | `0.0485` | `0.1700` |
| Conv1 | `v1/c1` | trainable | `ConvWeight_0` | `0.0651` | `0.0203` | `0.1995` | `0.2826` | `0.3424` |
| Conv1 | `v1/c1` | trainable | `DenseWeight_0` | `3.62e-3` | `3.33e-4` | `0.0109` | `0.0486` | `0.1685` |
| Conv1 | `v4/c1` | fixed | `ConvWeight_0` | `0.1011` | `0.0746` | `0.1957` | `0.2901` | `2.300` |
| Conv1 | `v4/c1` | fixed | `DenseWeight_0` | `6.78e-3` | `2.26e-3` | `0.0176` | `0.0534` | `0.7014` |
| Conv1 | `v4/c1` | trainable | `ConvWeight_0` | `0.1046` | `0.0798` | `0.1927` | `0.3571` | `1.805` |
| Conv1 | `v4/c1` | trainable | `DenseWeight_0` | `4.57e-3` | `7.15e-4` | `0.0119` | `0.0501` | `0.8413` |
| Conv2 | `v1/c1` | fixed | `ConvWeight_0` | `0.0687` | `7.49e-3` | `0.2113` | `0.5356` | `0.8440` |
| Conv2 | `v1/c1` | fixed | `ConvWeight_1` | `9.71e-3` | `2.39e-4` | `0.0266` | `0.1635` | `0.6894` |
| Conv2 | `v1/c1` | fixed | `DenseWeight_0` | `6.31e-3` | `3.16e-4` | `5.59e-3` | `0.1388` | `0.6217` |
| Conv2 | `v1/c1` | trainable | `ConvWeight_0` | `0.0947` | `4.73e-3` | `0.3301` | `0.6976` | `0.9914` |
| Conv2 | `v1/c1` | trainable | `ConvWeight_1` | `8.47e-3` | `4.04e-4` | `0.0212` | `0.1449` | `0.4969` |
| Conv2 | `v1/c1` | trainable | `DenseWeight_0` | `6.79e-3` | `3.46e-4` | `8.06e-3` | `0.1431` | `1.005` |
| Conv2 | `v4/c1` | fixed | `ConvWeight_0` | `0.1000` | `0.0388` | `0.2951` | `0.4804` | `0.5544` |
| Conv2 | `v4/c1` | fixed | `ConvWeight_1` | `7.51e-3` | `7.92e-4` | `0.0255` | `0.0646` | `0.1891` |
| Conv2 | `v4/c1` | fixed | `DenseWeight_0` | `5.44e-3` | `6.81e-4` | `0.0172` | `0.0510` | `0.1515` |
| Conv2 | `v4/c1` | trainable | `ConvWeight_0` | `0.0998` | `0.0395` | `0.2912` | `0.4726` | `0.5704` |
| Conv2 | `v4/c1` | trainable | `ConvWeight_1` | `7.51e-3` | `7.80e-4` | `0.0256` | `0.0651` | `0.1881` |
| Conv2 | `v4/c1` | trainable | `DenseWeight_0` | `5.41e-3` | `6.65e-4` | `0.0172` | `0.0510` | `0.1431` |

Read:

- First conv conductances are order `6e-2` to `1e-1` by mean. Conv1 `v4/c1` is the bad mismatched-gain row, but its first conv raw conductances are still only about `1.5x` the Conv1 `v1/c1` mean; the large max values are sparse tails.
- Later conv and dense conductances are order `5e-3` to `1e-2` by mean, with long tails up to `~0.1-1.0`.
- For `v4/c1`, the resistive energy uses an additional factor `(current_amp / voltage_amp)^layer_pre_index`. Thus `ConvWeight_0` has factor `1`, Conv1 `DenseWeight_0` and Conv2 `ConvWeight_1` have factor `0.25`, and Conv2 `DenseWeight_0` has factor `0.0625`. For Conv2 `v4/c1`, that makes the amp-factor-scaled means roughly `0.100`, `1.88e-3`, and `3.4e-4` across the three links.
- The corrected Conv1 `v4/c1` rerun with `input_gain=65.65764617919922` is still pending and is not included in this table.

### Final-Checkpoint Weight Gradients 2026-07-01

Computed with `experiments/diagnose_mnist_bp_conv_saturation_gradients.py` on the copied final checkpoints from Jean Zay array `1136832`, using the train split, batch size `4`, and `4` diagnostic batches per run on CPU. Results are under `/tmp/conv12_gradient_results`. The original TensorBoard event files only logged loss/accuracy/error, so these are recomputed final-checkpoint gradients, not logged training curves.

The main scale below is `lr_initial * ||grad||_2 / ||param||_2`, i.e. the relative size of one initial-LR update for each weight tensor.

| Depth | Amp | Mode | Final test | Global step | Max tensor step | `ConvWeight_0` | `ConvWeight_1` | `DenseWeight_0` |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| Conv1 | `v1/c1` | fixed | `96.08%` | `1.65e-3` | `1.94e-3` | `6.91e-4` | n/a | `1.94e-3` |
| Conv1 | `v1/c1` | trainable | `96.12%` | `1.42e-3` | `2.10e-3` | `7.35e-4` | n/a | `2.10e-3` |
| Conv1 | `v4/c1` | fixed | `88.67%` | `2.02e-3` | `2.44e-3` | `8.15e-4` | n/a | `2.44e-3` |
| Conv1 | `v4/c1` | trainable | `91.29%` | `2.46e-3` | `2.86e-3` | `1.55e-3` | n/a | `2.86e-3` |
| Conv2 | `v1/c1` | fixed | `96.13%` | `1.40e-3` | `2.35e-3` | `2.35e-3` | `8.60e-4` | `1.52e-3` |
| Conv2 | `v1/c1` | trainable | `97.03%` | `6.79e-3` | `1.23e-2` | `1.23e-2` | `6.85e-3` | `8.70e-3` |
| Conv2 | `v4/c1` | fixed | `97.12%` | `1.05e-3` | `1.50e-3` | `4.40e-4` | `1.50e-3` | `1.16e-3` |
| Conv2 | `v4/c1` | trainable | `97.10%` | `1.02e-3` | `1.70e-3` | `5.77e-4` | `1.70e-3` | `1.41e-3` |

Read:

- None of the conductance-weight gradients are collapsed. The per-weight gradient zero fraction was `0.0%` for every `ConvWeight_*` and `DenseWeight_0` row in this diagnostic.
- The bad Conv1 `v4/c1` rows do not look gradient-starved. Their relative update sizes are at least as large as Conv1 `v1/c1`, especially for the dense readout, which supports the view that the issue is the mismatched operating point rather than dead weight gradients.
- Conv2 `v1/c1` trainable has the largest residual gradients by far: trainable/fixed weight-gradient `L2` ratios are about `7.2x` for `ConvWeight_0`, `6.7x` for `ConvWeight_1`, and `6.0x` for `DenseWeight_0`. This matches the loss-stagnation check: it was still improving at epoch `30`, so it likely was not converged.
- Conv2 `v4/c1` fixed/trainable gradients are small and close to each other, consistent with the near-tied accuracy and near-plateaued loss.
- `HardSigmoidVOff_*` gradients from this diagnostic came out exactly zero. Do not over-interpret that as proof that `v_off` stopped learning; the current diagnostic is reliable for conductance-weight gradients, while `v_off` gradient handling needs a separate check.

### Corrected Conv1 `v4/c1` Rerun

Launched on Jean Zay R3 at 2026-07-01:

- Job: `1174247_[0-1%2]`, state at submission `PD (None)`.
- Launcher: `/home/filip/server_code/experiments/run_mnist_bp_conv1_r3_hardsigmoid_pad1_v4c1_corrected_gain_voff_trainability_jeanzay.slurm`.
- Remote launcher: `/lustre/fswork/projects/rech/umg/ucy17uy/server_code/experiments/run_mnist_bp_conv1_r3_hardsigmoid_pad1_v4c1_corrected_gain_voff_trainability_jeanzay.slurm`.
- Output root: `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_bp_conv1_s2_pad1_hardsigmoid_v4c1_shared_sat30_gain65p657_voff_trainability_seed0_10epoch`.
- Matrix: Conv1 `mnist_bp_amp_v4_c1`, stride `[2]`, padding `[1]`, `K=4`, 10 epochs, seed `0`, batch size `4`.
- Corrected operating point: `input_gain=65.65764617919922`, LR `0.00548298668852`.
- Modes: fixed `v_off=4.0` and trainable `v_off` initialized at `4.0`.
