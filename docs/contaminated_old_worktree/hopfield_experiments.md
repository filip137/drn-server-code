# CONTAMINATED OLD-WORKTREE SNAPSHOT

This file was copied from the old worktree snapshot `5e2d76fc`. The old worktree included simulations affected by the `adaptive_equilibrium=True` problem. Treat any result interpretation, selected protocol, or run decision here as historical context only unless revalidated with the corrected fixed-equilibrium protocol.

# Hopfield Experiments

Updated: 2026-07-04 17:24 CEST

## Goal

Study Hopfield-energy EqProp convolutional networks to understand:

- benchmark accuracy for Conv1/Conv2/Conv3 on MNIST and Fashion-MNIST;
- typical hard-sigmoid saturation values at initialization and after training;
- typical tanh near-bound values and signal sizes as a centered-nonlinearity
  comparison;
- how depth, stride, and spatial downsampling affect accuracy and saturation.

The main research question is whether deeper Hopfield EqProp conv networks fail
because the architecture is poor, because the hard-sigmoid operating point is
wrong, or because stride/downsampling changes the representation too strongly.

## Reference Convention

Axel Laborieux reference:

- Paper: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.633674/full
- Local code clone: `/home/filip/server_code/labs/Equilibrium-Propagation`

Observed reference details:

- Axel's MNIST CNN command uses `ToTensor()` followed by
  `Normalize(mean=(0.0,), std=(1.0,))`, so inputs remain in `[0,1]`.
- Axel's MNIST CNN example does not pass `--scale`; therefore it uses default
  PyTorch module initialization.
- Our current Hopfield runs use the same input scaling, but not identical
  initialization scale: weights use Kaiming-form initialization with current
  gains conv `0.6` and dense `1.5`, and biases use `0.5/sqrt(fan_in)`.

Treat an Axel-raw-init run as a separate diagnostic from the stride experiments.

## Core Protocol

- Datasets: MNIST and Fashion-MNIST, ordinary 1-channel inputs, 10 outputs.
- Loss: MSE.
- Activation: hard sigmoid clamped to `[0,1]`.
- Architectures:
  - Conv1: channels `[64]`
  - Conv2: channels `[64,128]`
  - Conv3: channels `[64,128,256]`
- Kernel: `3`.
- Pooling: none.
- Padding:
  - Conv1/Conv2: valid padding `0`
  - Conv3: padding `1`, matching the existing Conv3 convention
- EqProp: `T1=200`, `T2=10`, centered EqProp, beta `0.4`.
- Optimizer: Adam, LR multiplier `2.0` unless otherwise noted.
- Saturation measurement: init/best/final checkpoints, 2048 train and 2048 test
  samples, with low-bound, high-bound, total, sample p50/p90, and per-layer
  totals.

## Tanh Comparison Protocol

Purpose: use a centered hidden nonlinearity to test whether the hard-sigmoid
operating point is the main driver of the observed saturation and depth
behavior.

- Hidden activation: `tanh`; output layer remains linear.
- Primary inputs: same as the hard-sigmoid and Axel MNIST reference runs,
  `ToTensor()` plus `Normalize(mean=(0.0,), std=(1.0,))`, i.e. pixels in
  `[0,1]`.
- Fallback inputs: centered pixels via `Normalize(mean=(0.5,), std=(0.5,))`,
  only if the first `[0,1]` tests are clearly worse than hard sigmoid.
- First test: Conv1/Conv2/Conv3 default stride-2 geometry, seeds `0,1,2`, LR
  multipliers `0.25,0.5,1,2,4`, 5 epochs.
- Gate: if best `[0,1]` tanh is more than 2 percentage points below the
  hard-sigmoid 5-epoch reference for at least two depths, run the centered-input
  LR screen before final settings.
- Diagnostics: near-bound occupancy uses `h <= -0.99` and `h >= 0.99`; report
  hidden near-low/near-high/near-total, hidden state mean/abs/std, free-to-nudged
  hidden RMS, and EqProp parameter-gradient RMS/mean-abs. Do not report the
  2048-subset accuracy in these diagnostic tables.

Status on 2026-06-30 18:10 CEST: code and launchers are implemented; local
compile, Slurm syntax, tiny synthetic train, centered-input train, diagnostics
checks, and remote Jean Zay compile/syntax checks passed. Identity-input tanh
jobs were submitted as smoke `1136935`, LR screen `1136936`, and collector
`1136937`. Smoke `1136935` completed successfully with `5/5` metrics including
Conv3 stride-1 and Conv3 `[2,2,1]` padding1 memory checks. LR screen `1136936`
and collector `1136937` completed successfully with `45/45` LR metrics. The
identity-input best mean over seeds was Conv1 `88.07%`, Conv2 `93.20%`, Conv3
`95.29%`, all at LR multiplier `4.0`. Since Conv1 and Conv2 were clearly worse
than the hard-sigmoid 5-epoch references, centered-input fallback jobs were
submitted: LR screen `1146084` and collector `1146085`.

Update on 2026-07-01 12:10 CEST: centered-input fallback completed with
`45/45` LR metrics. Best mean over seeds at 5 epochs was Conv1 `87.23%`, Conv2
`92.15%`, Conv3 `95.90%`, all at LR multiplier `4.0`. Centering did not improve
Conv1/Conv2 but did improve Conv3. Final 30-epoch tanh training uses the best
preprocessing per depth: Conv1/Conv2 identity `[0,1]`, Conv3 centered
`[-1,1]`, LR multiplier `4.0` for all depths. Final job `1172067` is queued;
diagnostics job `1172068` is queued after it.

## Affine-MNIST Depth-Gap Protocol

Purpose: make the Conv2 versus Conv3 difference more visible by using a
deterministic affine-MNIST task where Conv3's larger receptive field should be
more useful than on ordinary MNIST.

- Base data: ordinary MNIST, 1-channel inputs, 10 linear outputs, MSE.
- Input scaling: same as the hard-sigmoid/Axel convention, pixels in `[0,1]`.
- Affine preset `medium`: rotation `+-25 deg`, translation up to `20%` of
  width/height, scale `[0.8,1.2]`, no shear, fill `0`.
- Affine preset `mnist_affine`: rotation `+-60 deg`, translation up to `15%` of
  width/height, scale `[0.8,1.2]`, shear `+-15 deg`, fill `0`.
- Transform determinism: per-sample transform is fixed by `affine_seed=1729`
  and does not depend on model seed.
- Conv2 geometry: channels `[64,128]`, padding `1`, effective strides `[2,2]`.
- Conv3 geometry: channels `[64,128,256]`, padding `1`, effective strides
  `[2,2,1]`.
- EqProp/training: hard sigmoid, `T1=200`, `T2=10`, centered EqProp beta
  `0.4`, LR multiplier `2.0`, 50 epochs, seeds `0,1,2`, `PACK_SIZE=2`.
- Saturation: init/best/final checkpoints on 2048 train/test samples.

Status on 2026-07-04 16:38 CEST: code and launchers implemented; local Python
compile, Slurm syntax, tiny CPU affine training smoke, and tiny affine
saturation smoke passed. Jean Zay remote compile/syntax checks passed. Packed
smoke `1317786` completed with `4/4` metrics, including the two-Conv3 packed
memory check. Final job `1317803` is running; saturation job `1317804` is
pending dependency. Added `mnist_affine` as a second deterministic preset.
Smoke `1318395` completed with `4/4` metrics. Final job `1318422` has tasks
`_0` and `_2` running; task `_1` failed before Python on a Slurm stdout I/O
error and was rerun as `1318449` after adding per-packed-task logs. Guarded
saturation job `1318467` will wait for all six final metrics before measuring
saturation.

## Completed MNIST Baseline

Result root:

`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_hopfield_eqprop_conv_saturation_20260628`

Local CSV copies:

`/home/filip/server_code/results/mnist_hopfield_eqprop_conv_saturation_20260628`

Final 30-epoch accuracy over seeds `0,1,2`:

| Model | Stride | Best test | Final test |
|---|---:|---:|---:|
| Conv1 | 2 | 97.29% | 97.27% |
| Conv2 | 2 | 97.61% | 97.50% |
| Conv3 | 2 | 97.01% | 97.01% |
| Conv2 | 1 | 98.88% | 98.32% |
| Conv3 | 1 | 99.13% | 99.04% |
| Conv1 | `[2]`, pad1 | 97.06% | 97.04% |
| Conv2 | `[2,2]`, pad1 | 97.43% | 97.39% |
| Conv3 | `[2,2,1]`, pad1 | 97.67% | 97.64% |

Final-checkpoint test saturation over seeds `0,1,2`:

These saturation rows intentionally omit the 2048-subset accuracy; use the
full-test accuracy table above for benchmark accuracy.

| Model | Stride | All sat | Low / high sat | Hidden-layer total saturation |
|---|---:|---:|---:|---|
| Conv1 | 2 | 59.83% | 59.43% / 0.40% | h1 59.83% |
| Conv2 | 2 | 52.44% | 50.22% / 2.22% | h1 40.09%, h2 81.45% |
| Conv3 | 2 | 41.29% | 36.13% / 5.16% | h1 30.63%, h2 32.31%, h3 87.70% |
| Conv2 | 1 | 77.28% | 76.93% / 0.35% | h1 49.62%, h2 93.51% |
| Conv3 | 1 | 78.83% | 78.02% / 0.82% | h1 57.01%, h2 49.31%, h3 99.05% |
| Conv1 | `[2]`, pad1 | 60.28% | 60.00% / 0.28% | h1 60.28% |
| Conv2 | `[2,2]`, pad1 | 55.19% | 53.13% / 2.06% | h1 41.50%, h2 82.57% |
| Conv3 | `[2,2,1]`, pad1 | 53.69% | 49.97% / 3.72% | h1 29.33%, h2 39.06%, h3 85.37% |

Interpretation so far:

- Conv2 stride-1 is clearly better than Conv2 stride-2.
- Conv2 stride-1 is also much more saturated, mostly at the low bound.
- Conv3 stride-2 has high saturation in hidden layer h3 but low saturation in
  h1/h2, which suggests that the deep representation is unevenly using the
  hard sigmoid.

## Active MNIST Runs

Conv1 stride-1:

- Smoke `1048787`: completed.
- Final `1048788`: completed.
- Saturation `1048789`: completed.
- Mean best/final test accuracy: `98.21%/98.16%`.
- Final-checkpoint test saturation: all-unit total `85.33%`, low/high
  `85.32%/0.001%`, sample p50/p90 `85.24%/86.51%`.
- Output group: `final_conv1_stride1`.
- Saturation output: `conv1_stride1_saturation_train_test_2048.csv`.

Conv3 stride-1:

- Smoke `1054867`: completed.
- Final `1054868`: completed.
- Saturation `1054869`: completed.
- Mean best/final test accuracy: `99.13%/99.04%`.
- Final-checkpoint test saturation: all-unit total `78.83%`, low/high
  `78.02%/0.82%`, layer totals h1 `57.01%`, h2 `49.31%`, h3 `99.05%`.
- Output group: `final_conv3_stride1`.
- Saturation output: `conv3_stride1_saturation_train_test_2048.csv`.

MNIST layerwise strides `2,2,1`, padding1:

- Result root:
  `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_hopfield_eqprop_conv_strides221_pad1_20260629`
- Requested/effective strides: Conv1 `[2]`, Conv2 `[2,2]`, Conv3 `[2,2,1]`.
- Padding: `1` on every convolutional layer.
- Hidden spatial sizes: Conv1 `14`; Conv2 `14 -> 7`; Conv3 `14 -> 7 -> 7`.
- Smoke `1114567`: completed.
- Final `1114568`: completed.
- Saturation `1114569`: completed.
- Mean best/final test accuracy: Conv1 `97.06%/97.04%`, Conv2
  `97.43%/97.39%`, Conv3 `97.67%/97.64%`.
- Final-checkpoint test saturation: Conv1 `60.28%`, Conv2 `55.19%`, Conv3
  `53.69%`.
- Saturation output: `mnist_strides2_2_1_pad1_saturation_train_test_2048.csv`.

Continuation from final weights:

- Goal: check whether the padding-1 30-epoch runs were still undertrained in
  loss.
- Launcher:
  `experiments/run_mnist_hopfield_eqprop_conv_strides221_pad1_continue20_jeanzay.slurm`
- Job `1129861`: submitted on 2026-06-30 11:40 CEST as array `0-8%9`,
  initially `PD (Priority)`.
- Source checkpoints: the previous 30-epoch `final_model.pt` files for Conv1,
  Conv2, and Conv3, seeds `0,1,2`.
- Output root:
  `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/mnist_hopfield_eqprop_conv_strides221_pad1_continue20_20260630`
- Matrix: Conv1 `[2]` pad1, Conv2 `[2,2]` pad1, Conv3 `[2,2,1]` pad1;
  20 more epochs; LR multiplier `2.0`.
- Completed on Jean Zay with `9/9` metrics.
- Mean best/final test accuracy after 50 total epochs: Conv1
  `97.44%/97.40%`, Conv2 `97.74%/97.63%`, Conv3 `97.95%/97.94%`.
- Per-seed final test accuracy after continuation: Conv1 seeds `0,1,2`
  `97.43%`, `97.40%`, `97.38%`; Conv2 `97.58%`, `97.61%`, `97.70%`;
  Conv3 `98.00%`, `98.01%`, `97.82%`.
- Mean loss did not stagnate. Test loss from epoch 30 to 50 fell Conv1
  `0.07143 -> 0.06344`, Conv2 `0.02765 -> 0.02291`, Conv3
  `0.02059 -> 0.01817`; train loss fell Conv1 `0.07085 -> 0.06127`,
  Conv2 `0.02598 -> 0.01976`, Conv3 `0.01693 -> 0.01165`.
- Final-checkpoint test saturation after 50 total epochs: Conv1 `66.27%`
  low/high `65.86%/0.41%`; Conv2 `55.28%` low/high `53.11%/2.17%`,
  layer totals h1 `41.34%`, h2 `83.15%`; Conv3 `54.77%` low/high
  `50.98%/3.79%`, layer totals h1 `30.43%`, h2 `40.88%`, h3 `86.04%`.
- Stagewise test saturation available from saved checkpoints: initial, 30-epoch
  final, and 50-epoch final. Conv1 total `49.84% -> 60.28% -> 66.27%`;
  Conv2 `50.13% -> 55.19% -> 55.28%`; Conv3 `50.11% -> 53.69% -> 54.77%`.
  True per-epoch saturation cannot be reconstructed for these runs because
  epoch checkpoints were not saved.
- Saturation output:
  `mnist_strides2_2_1_pad1_continue20_saturation_train_test_2048.csv`.

## Active Fashion-MNIST Runs

Fashion-MNIST result root:

`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/fmnist_hopfield_eqprop_conv_stride_20260629`

Dataset root:

`/lustre/fsn1/projects/rech/umg/ucy17uy/datasets/fashion_mnist`

Stride grid:

- Depths: Conv1/Conv2/Conv3.
- Strides: `1` and `2`.
- Seeds: `0,1,2`.
- Smoke `1054870`: completed for all six depth/stride pairs.
- Final `1054871`: completed, 18 tasks total.
- Saturation `1054872`: completed.
- Saturation output: `fmnist_stride_saturation_train_test_2048.csv`.

Completed training block so far:

| Dataset | Model | Stride | Seeds | Best test | Final test | Status |
|---|---|---:|---|---:|---:|---|
| Fashion-MNIST | Conv1 | 1 | `0,1,2` | 90.12% | 89.90% | saturation complete |
| Fashion-MNIST | Conv1 | 2 | `0,1,2` | 87.74% | 87.73% | saturation complete |
| Fashion-MNIST | Conv2 | 1 | `0,1,2` | 91.75% | 91.44% | saturation complete |
| Fashion-MNIST | Conv2 | 2 | `0,1,2` | 87.67% | 87.64% | saturation complete |
| Fashion-MNIST | Conv3 | 1 | `0,1,2` | 92.07% | 91.61% | saturation complete |
| Fashion-MNIST | Conv3 | 2 | `0,1,2` | 86.69% | 86.42% | saturation complete |

## Active Fashion-MNIST 2-2-0 Layerwise-Stride Runs

Result root:

`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/fmnist_hopfield_eqprop_conv_strides220_20260629`

Requested layer strides:

- First conv layer: `2`.
- Second conv layer: `2`.
- Third conv layer: `0`, recorded as requested but executed as effective
  stride `1` because PyTorch convolution stride `0` is invalid.

Effective strides by depth:

- Conv1: `[2]`
- Conv2: `[2,2]`
- Conv3: `[2,2,1]`

Jobs:

- Smoke `1114316`: completed.
- Final `1114317`: completed.
- Saturation `1114318`: completed.
- Mean best/final test accuracy: Conv1 `87.74%/87.73%`, Conv2
  `87.68%/87.64%`, Conv3 `87.96%/87.81%`.
- Final-checkpoint test saturation: Conv1 `61.05%`, Conv2 `54.04%`, Conv3
  `55.14%`.
- Saturation output: `fmnist_strides2_2_0_saturation_train_test_2048.csv`.

## Reporting Rules

- Keep `docs/current_experiments.md` as the scheduler/status log.
- Keep `docs/current_state.md` as the concise research-state summary.
- When a Hopfield run finishes, report:
  - final/best accuracy by depth, stride, dataset, and seed;
  - mean accuracy over seeds;
  - final checkpoint saturation on test split;
  - init/best/final saturation comparison when it changes the interpretation.
- For stride comparisons, always note batch size, because the stride-1 runs use
  smaller batches for memory and therefore more optimizer steps per epoch.
