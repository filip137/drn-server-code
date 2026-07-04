# CONTAMINATED OLD-WORKTREE SNAPSHOT

This file was copied from the old worktree snapshot `5e2d76fc`. The old worktree included simulations affected by the `adaptive_equilibrium=True` problem. Treat any result interpretation, selected protocol, or run decision here as historical context only unless revalidated with the corrected fixed-equilibrium protocol.

# Current Experiments

Updated: 2026-07-04 17:24 CEST

## Hopfield EqProp Conv MNIST Saturation

Research question: typical saturation in Hopfield EqProp conv MNIST networks.

Reference material:

- Paper: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.633674/full
- Official code clone: `/home/filip/server_code/labs/Equilibrium-Propagation`
- Local implementation uses the repo-native `model.hopfield` stack, not the older copied top-level `Equilibrium-Propagation` directory.

Jean Zay R3 defaults:

- Account/partition: `fmu@v100`, `gpu_p13`, `qos_gpu-t3`, `v100-16g`
- Source root: `/lustre/fswork/projects/rech/umg/${USER}/server_code`
- Dataset root: `/lustre/fsn1/projects/rech/umg/${USER}/datasets/mnist`
- Result root: `/lustre/fsn1/projects/rech/fmu/${USER}/server_code/results/mnist_hopfield_eqprop_conv_saturation_20260628`

Run matrix:

- Dataset/model convention: ordinary MNIST, 1 input channel, 10 linear outputs, MSE loss.
- Architectures: Conv1 `[64]`, Conv2 `[64,128]`, Conv3 `[64,128,256]`; kernel `3`; stride `2`; no pooling; Conv1/Conv2 padding `0`; Conv3 padding `1`.
- EqProp settings: hard sigmoid `[0,1]`, `T1=200`, `T2=10`, centered EqProp, beta `0.4`.
- Optimizer: Adam; base edge LRs `[5e-5, 1e-5, 8e-6]` from Laborieux MNIST CNN example, with bias LRs matched to incoming edge LRs.
- LR screen: depths `1,2,3`, seed `0`, LR multipliers `0.25,0.5,1,2`, 5 epochs.
- Final: depths `1,2,3`, seeds `0,1,2`, selected LR per depth, 30 epochs.
- Saturation analysis: init, best, and final checkpoints; train/test splits; 2048 samples each; records per-layer low/high/total hard-sigmoid saturation and sample-level saturation mean/p50/p90.

Launchers:

- Smoke gate: `experiments/run_mnist_hopfield_eqprop_conv_smoke_jeanzay.slurm`
- LR screen: `experiments/run_mnist_hopfield_eqprop_conv_lr_screen_jeanzay.slurm`
- Final training: `experiments/run_mnist_hopfield_eqprop_conv_final_jeanzay.slurm`
- Saturation analysis: `experiments/run_mnist_hopfield_eqprop_conv_saturation_jeanzay.slurm`
- Conv2-only stride-2 saturation: `experiments/run_mnist_hopfield_eqprop_conv2_saturation_jeanzay.slurm`
- Conv1 stride-1 smoke: `experiments/run_mnist_hopfield_eqprop_conv1_stride1_smoke_jeanzay.slurm`
- Conv1 stride-1 final: `experiments/run_mnist_hopfield_eqprop_conv1_stride1_final_jeanzay.slurm`
- Conv1 stride-1 saturation: `experiments/run_mnist_hopfield_eqprop_conv1_stride1_saturation_jeanzay.slurm`
- Conv3 stride-1 smoke: `experiments/run_mnist_hopfield_eqprop_conv3_stride1_smoke_jeanzay.slurm`
- Conv3 stride-1 final: `experiments/run_mnist_hopfield_eqprop_conv3_stride1_final_jeanzay.slurm`
- Conv3 stride-1 saturation: `experiments/run_mnist_hopfield_eqprop_conv3_stride1_saturation_jeanzay.slurm`
- MNIST layerwise strides `2,2,1` padding1 smoke: `experiments/run_mnist_hopfield_eqprop_conv_strides221_pad1_smoke_jeanzay.slurm`
- MNIST layerwise strides `2,2,1` padding1 final: `experiments/run_mnist_hopfield_eqprop_conv_strides221_pad1_final_jeanzay.slurm`
- MNIST layerwise strides `2,2,1` padding1 saturation: `experiments/run_mnist_hopfield_eqprop_conv_strides221_pad1_saturation_jeanzay.slurm`
- Fashion-MNIST stride-grid smoke: `experiments/run_fmnist_hopfield_eqprop_conv_stride_smoke_jeanzay.slurm`
- Fashion-MNIST stride-grid final: `experiments/run_fmnist_hopfield_eqprop_conv_stride_final_jeanzay.slurm`
- Fashion-MNIST stride-grid saturation: `experiments/run_fmnist_hopfield_eqprop_conv_stride_saturation_jeanzay.slurm`
- Fashion-MNIST layerwise strides `2,2,0` smoke: `experiments/run_fmnist_hopfield_eqprop_conv_strides220_smoke_jeanzay.slurm`
- Fashion-MNIST layerwise strides `2,2,0` final: `experiments/run_fmnist_hopfield_eqprop_conv_strides220_final_jeanzay.slurm`
- Fashion-MNIST layerwise strides `2,2,0` saturation: `experiments/run_fmnist_hopfield_eqprop_conv_strides220_saturation_jeanzay.slurm`
- Current-state report: `experiments/run_mnist_hopfield_eqprop_conv_report_jeanzay.slurm`
- Conv2 stride-1 smoke: `experiments/run_mnist_hopfield_eqprop_conv2_stride1_smoke_jeanzay.slurm`
- Conv2 stride-1 final: `experiments/run_mnist_hopfield_eqprop_conv2_stride1_final_jeanzay.slurm`
- Conv2 stride-1 saturation: `experiments/run_mnist_hopfield_eqprop_conv2_stride1_saturation_jeanzay.slurm`

Status:

- Local implementation complete.
- Local checks passed: `py_compile`, focused Hopfield unit tests, and synthetic CPU training smoke.
- Jean Zay access works through the local SSH alias `jean-zay`, which uses `ProxyJump integnano2`; direct `ucy17uy@jean-zay.idris.fr` is the wrong route from this machine.
- Smoke job `1005669` completed successfully: `COMPLETED`, exit `0:0`, elapsed `00:00:25`, metrics `2/2`.
- Packed smoke results: Conv1 best/final test accuracy `5.5% / 5.5%`; Conv3 best/final test accuracy `13.5% / 13.5%`. These are one-epoch capped sanity checks only.
- Packed Conv3 smoke passed with `PACK_SIZE=2`; LR screen is cleared to submit.
- LR screen job `1005921` completed successfully with metrics `12/12`; LR collector job `1005954` completed successfully.
- LR-screen selected multiplier `2.0` for all depths: Conv1 best/final `94.75%/94.75%`, Conv2 `96.73%/96.73%`, Conv3 `94.58%/94.58%`.
- Old final chain `1005955 -> 1005956 -> 1006029` was cancelled because the original 24h final job exceeded `qos_gpu-t3` walltime (`QOSMaxWallDurationPerJobLimit`).
- Replacement final training job `1017213` completed successfully with metrics `9/9`.
- Replacement saturation job `1017214` completed successfully; output `final_saturation_train_test_2048.csv`.
- Replacement current-state report job `1017219` completed successfully, but the tracked block was refreshed manually from the completed CSVs.
- Conv2-only stride-2 saturation job `1019118` completed successfully; output `conv2_stride2_saturation_train_test_2048.csv`.
- Final 30-epoch test accuracy over seeds `0,1,2`: Conv1 best/final `97.29%/97.27%`, Conv2 `97.61%/97.50%`, Conv3 `97.01%/97.01%`.

## Hopfield EqProp Conv MNIST Tanh Comparison

Question: does replacing hard sigmoid with a centered `tanh` hidden
nonlinearity change benchmark accuracy and typical near-bound occupancy?

Input convention:

- Primary run uses the same MNIST input scaling as Axel's reference and the
  previous Hopfield hard-sigmoid jobs: `ToTensor()` plus
  `Normalize(mean=(0.0,), std=(1.0,))`, so pixels remain in `[0,1]`.
- Centered fallback is implemented but not the first run:
  `Normalize(mean=(0.5,), std=(0.5,))`, mapping pixels to `[-1,1]`.
- Output layer remains linear with 10 outputs and MSE loss.

Implementation:

- `model.hopfield.layer.TanhLayer` added and wired through
  `model.hopfield.network.create_layer`.
- `experiments/train_mnist_hopfield_eqprop_conv_sweep.py` now accepts
  `--activation tanh` and `--input-preprocessing identity|centered`.
- `experiments/measure_mnist_hopfield_eqprop_diagnostics.py` measures
  activation-aware hidden near-bound fractions, hidden state mean/abs/std,
  nudged hidden signal RMS, and EqProp parameter-gradient RMS/mean-abs. It does
  not write subset accuracy columns.

First-test gate:

- Run identity-input tanh first on Conv1/Conv2/Conv3 stride-2 defaults, seeds
  `0,1,2`, LR multipliers `0.25,0.5,1,2,4`, 5 epochs, `PACK_SIZE=2`.
- Hard-sigmoid 5-epoch reference from the completed LR screen: Conv1 `94.75%`,
  Conv2 `96.73%`, Conv3 `94.58%`.
- If the best identity-input tanh LR-screen result is more than 2 percentage
  points below the hard-sigmoid reference for at least two depths, run the
  centered-input tanh fallback screen before choosing final settings.

Launchers:

- Identity smoke:
  `experiments/run_mnist_hopfield_eqprop_conv_tanh_identity_smoke_jeanzay.slurm`
- Identity LR screen:
  `experiments/run_mnist_hopfield_eqprop_conv_tanh_identity_lr_screen_jeanzay.slurm`
- Identity LR collector:
  `experiments/run_mnist_hopfield_eqprop_conv_tanh_identity_collect_jeanzay.slurm`
- Centered fallback LR screen:
  `experiments/run_mnist_hopfield_eqprop_conv_tanh_centered_lr_screen_jeanzay.slurm`
- Centered fallback LR collector:
  `experiments/run_mnist_hopfield_eqprop_conv_tanh_centered_collect_jeanzay.slurm`
- Identity final training, after LR selection:
  `experiments/run_mnist_hopfield_eqprop_conv_tanh_identity_final_jeanzay.slurm`
- Identity diagnostics, after final training:
  `experiments/run_mnist_hopfield_eqprop_conv_tanh_identity_diagnostics_jeanzay.slurm`

Status:

- Local checks passed on 2026-06-30 14:35 CEST: Python compile with the conda
  PyTorch interpreter, Slurm `bash -n`, tiny CPU synthetic identity-input tanh
  training, tiny CPU synthetic centered-input tanh training, and diagnostics
  CSV generation from a tiny tanh checkpoint.
- Remote Jean Zay checks passed on 2026-06-30 14:36 CEST: Python compile under
  `pytorch-gpu/py3/2.5.0` and Slurm `bash -n`.
- Result root for identity-input first tests:
  `/lustre/fsn1/projects/rech/fmu/${USER}/server_code/results/mnist_hopfield_eqprop_conv_tanh_identity_20260630`.
- Result root for centered fallback if needed:
  `/lustre/fsn1/projects/rech/fmu/${USER}/server_code/results/mnist_hopfield_eqprop_conv_tanh_centered_20260630`.
- Submitted on 2026-06-30 14:37 CEST.
- Smoke job `1136935`, array `0-2%3`, completed successfully with metrics
  `5/5`: Conv1 `[2]`, Conv2 `[2,2]`, Conv3 `[2,2,2]`, Conv3 `[1,1,1]`
  padding1, and Conv3 `[2,2,1]` padding1.
- Capped one-epoch smoke final test accuracies were Conv1 `[2]` `7.0%`,
  Conv2 `[2,2]` `16.0%`, Conv3 `[2,2,2]` `10.5%`, Conv3 `[1,1,1]`
  padding1 `8.0%`, and Conv3 `[2,2,1]` padding1 `8.0%`; these are sanity
  checks only.
- Identity LR screen job `1136936`, array `0-22%9`, is now `PD (Priority)`.
- Identity collector job `1136937` remains `PD (Dependency)` after LR.
- Update 2026-06-30 18:10 CEST: identity-input LR screen `1136936` and
  collector `1136937` completed successfully with `45/45` LR metrics.
- Identity-input tanh best mean over seeds at 5 epochs:
  - Conv1: LR multiplier `4.0`, best/final `88.07%/88.05%`
    versus hard-sigmoid reference `94.75%`.
  - Conv2: LR multiplier `4.0`, best/final `93.20%/93.20%`
    versus hard-sigmoid reference `96.73%`.
  - Conv3: LR multiplier `4.0`, best/final `95.29%/95.29%`
    versus hard-sigmoid reference `94.58%`.
- Because Conv1 and Conv2 are more than 2 percentage points below the
  hard-sigmoid reference, the centered-input fallback was launched.
- Update 2026-07-01 12:10 CEST: centered-input LR screen `1146084` and
  collector `1146085` completed successfully with `45/45` LR metrics.
- Centered-input tanh best mean over seeds at 5 epochs:
  - Conv1: LR multiplier `4.0`, best/final `87.23%/86.78%`.
  - Conv2: LR multiplier `4.0`, best/final `92.15%/92.15%`.
  - Conv3: LR multiplier `4.0`, best/final `95.90%/95.90%`.
- Preprocessing choice for the final 30-epoch tanh run: Conv1/Conv2 use
  identity `[0,1]` inputs because they beat centered inputs in the LR screen;
  Conv3 uses centered `[-1,1]` inputs because it beat identity inputs.
- Mixed final launcher:
  `experiments/run_mnist_hopfield_eqprop_conv_tanh_best_final_jeanzay.slurm`.
- Mixed final diagnostics launcher:
  `experiments/run_mnist_hopfield_eqprop_conv_tanh_best_diagnostics_jeanzay.slurm`.
- Result root:
  `/lustre/fsn1/projects/rech/fmu/${USER}/server_code/results/mnist_hopfield_eqprop_conv_tanh_best_20260701`.
- Final training job `1172067`, array `0-4%5`, submitted on
  2026-07-01 12:10 CEST with `PACK_SIZE=2`; tasks `0,1` running and tasks
  `2-4` pending on resources at the 12:10 CEST check.
- Diagnostics job `1172068` submitted with dependency `afterok:1172067`; state
  `PD (Dependency)`.

## Hopfield EqProp Conv2 Stride-1 Diagnostic

Question: is repeated stride-2 downsampling part of why adding more conv layers fails to help, or is the issue mainly the nonlinearity/operating point?

Plan:

- Keep the well-trained Hopfield Conv2 settings from the LR screen: channels `[64,128]`, hard sigmoid `[0,1]`, no pooling, `T1=200`, `T2=10`, centered EqProp, beta `0.4`, LR multiplier `2.0`.
- Change spatial stride only: Conv2 stride `1`, valid padding `0`.
- Because stride-1 Conv2 has much larger hidden states (`64x26x26` then `128x24x24`), use batch size `16` for training and `64` for testing; keep `PACK_SIZE=2` after a smoke gate.
- Smoke gate: seeds `0,1`, one epoch capped at 20 train batches and 5 test batches.
- Final diagnostic: seeds `0,1,2`, 30 epochs, `PACK_SIZE=2`, output group `final_conv2_stride1`.
- Saturation diagnostic: init/best/final checkpoints on 2048 train and 2048 test samples, output `conv2_stride1_saturation_train_test_2048.csv`.

Status:

- Local Slurm syntax checks passed for the stride-1 smoke/final/saturation launchers.
- Smoke job `1017855` completed successfully with metrics `2/2`; capped one-epoch smoke test accuracies were seed 0 `42.50%` and seed 1 `30.94%`.
- Final diagnostic job `1017856` completed successfully with metrics `3/3`.
- Saturation diagnostic job `1017857` completed successfully; output `conv2_stride1_saturation_train_test_2048.csv`.
- Final stride-1 Conv2 test accuracy over seeds `0,1,2`: best/final `98.88%/98.32%`.
- Final test saturation comparison: stride-2 Conv2 all-unit total `52.44%` with layer totals h1 `40.09%`, h2 `81.45%`; stride-1 Conv2 all-unit total `77.28%` with layer totals h1 `49.62%`, h2 `93.51%`.

## Affine-MNIST Hopfield EqProp Conv2/Conv3 Depth Gap

Question: can a deterministic affine-MNIST variant make the Conv2 versus Conv3
depth advantage more visible than ordinary MNIST?

Geometry:

- Conv2: channels `[64,128]`, padding `1`, effective strides `[2,2]`.
- Conv3: channels `[64,128,256]`, padding `1`, effective strides `[2,2,1]`.
- Kernel `3`, no pooling, hard sigmoid `[0,1]`, MSE, `T1=200`, `T2=10`,
  centered EqProp beta `0.4`, Adam LR multiplier `2.0`.

Dataset transform:

- Base dataset is ordinary MNIST with inputs kept in `[0,1]`.
- Deterministic per-sample affine preset `medium`: rotation `+-25 deg`,
  translation up to `20%` of width/height, scale sampled in `[0.8,1.2]`, no
  shear, fill `0`.
- Second deterministic preset `mnist_affine`: rotation `+-60 deg`, translation
  up to `15%` of width/height, scale sampled in `[0.8,1.2]`, shear `+-15 deg`,
  fill `0`.
- Affine samples are fixed by `affine_seed=1729` and independent of model seed,
  so Conv2 and Conv3 see the same transformed train/test examples.

Schedule:

- Smoke launcher:
  `experiments/run_mnist_hopfield_eqprop_conv_affine_s221_pad1_smoke_jeanzay.slurm`.
- Final launcher:
  `experiments/run_mnist_hopfield_eqprop_conv_affine_s221_pad1_final_jeanzay.slurm`.
- Saturation launcher:
  `experiments/run_mnist_hopfield_eqprop_conv_affine_s221_pad1_saturation_jeanzay.slurm`.
- Result root:
  `/lustre/fsn1/projects/rech/fmu/${USER}/server_code/results/mnist_hopfield_eqprop_conv_affine_s221_pad1_20260704`.
- Smoke tests Conv2 seed `0`, Conv3 seed `0`, and a packed two-Conv3 case
  before the full run.
- Final matrix: Conv2/Conv3, seeds `0,1,2`, 50 epochs, `PACK_SIZE=2`.
- Saturation: init/best/final checkpoints on 2048 train and 2048 test samples.

Status:

- Local checks passed on 2026-07-04 16:35 CEST: Python compile, Slurm syntax,
  tiny CPU affine-MNIST training smoke, and tiny saturation CSV generation.
- Jean Zay remote checks passed: Python compile under `pytorch-gpu/py3/2.5.0`
  and Slurm syntax checks.
- Smoke job `1317786`, array `0-1%2`, completed successfully with `4/4`
  metrics. The packed two-Conv3 task completed, so `PACK_SIZE=2` is cleared for
  the final.
- Capped one-epoch smoke final test accuracies were Conv2 seed `0` `8.50%`,
  Conv3 seed `0` `10.83%`, Conv3 seed `1` `11.17%`, Conv3 seed `2` `9.83%`;
  these are sanity checks only.
- Final training job `1317803`, array `0-2%3`, submitted on 2026-07-04
  16:38 CEST and running at the first check.
- Saturation job `1317804` submitted with dependency `afterok:1317803`; state
  `PD (Dependency)`.
- Additional `mnist_affine` preset added on 2026-07-04 17:24 CEST.
- `mnist_affine` smoke job `1318395`, array `0-1%2`, completed successfully
  with `4/4` metrics, including the packed two-Conv3 memory check.
- Capped one-epoch `mnist_affine` smoke final test accuracies were Conv2 seed
  `0` `7.83%`, Conv3 seed `0` `10.00%`, Conv3 seed `1` `7.67%`, Conv3 seed
  `2` `9.67%`; these are sanity checks only.
- `mnist_affine` final job `1318422` was submitted as array `0-2%3`; tasks
  `_0` and `_2` are running. Task `_1` failed before Python with a Slurm stdout
  I/O error at the status echo, so packed-run logs were redirected to per-task
  files and array task `1` was rerun as job `1318449`.
- The original dependent saturation job `1318423` was canceled because
  `afterok:1318422` cannot fire after one failed subtask. Replacement guarded
  saturation job `1318467` depends `afterany:1318422:1318449`, waits until all
  six final metrics exist, then writes
  `mnist_affine_s221_pad1_saturation_train_test_2048.csv`.
- `mnist_affine` result root:
  `/lustre/fsn1/projects/rech/fmu/${USER}/server_code/results/mnist_hopfield_eqprop_conv_mnist_affine_s221_pad1_20260704`.

## Axel Initialization And Input Scaling Check

Sources:

- Paper: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.633674/full
- Local official code clone: `/home/filip/server_code/labs/Equilibrium-Propagation`

Findings:

- The paper states that the CIFAR ConvNet weights use PyTorch's default uniform Kaiming initialization.
- Axel's MNIST CNN command is `labs/Equilibrium-Propagation/check/train_mnist_cnn.sh`: channels `32,64`, kernels `5,5`, max-pooling after both conv layers, stride `1,1`, Adam LRs `5e-5,1e-5,8e-6`, hard sigmoid, `T1=200`, `T2=10`, beta `0.4`, batch size `100`.
- That MNIST command does not pass `--scale`; in Axel's code, `--scale` is the optional multiplier that reapplies/scales Kaiming initialization. Therefore the MNIST CNN example uses default PyTorch module initialization.
- MNIST input transform in `labs/Equilibrium-Propagation/main.py` is `ToTensor()` followed by `Normalize(mean=(0.0,), std=(1.0,))`, i.e. pixel values remain in `[0,1]`.
- Our Hopfield MNIST script uses the same input scaling: `ToTensor()` plus identity normalization.
- Our current Hopfield initialization is not identical in scale to Axel's default PyTorch modules: weights use Kaiming-uniform form but layer gains default to conv `0.6` and dense `1.5`; biases are initialized with gain `0.5/sqrt(fan_in)`, i.e. half of PyTorch's default bias bound. Treat an Axel-raw-init diagnostic (`weight_gains=1`, PyTorch-scale biases) as a separate factor, not part of the stride-only repeat.

## Hopfield EqProp Conv1 Stride-1 Diagnostic

Question: does preserving spatial resolution also help Conv1, and how does Conv1 stride-1 saturation compare with the completed Conv2 stride-1 run?

Plan:

- Keep the current Hopfield protocol to isolate stride against the existing runs: hard sigmoid `[0,1]`, no pooling, `T1=200`, `T2=10`, centered EqProp, beta `0.4`, LR multiplier `2.0`.
- Conv1, channels `[64]`, kernel `3`, stride `1`, valid padding `0`.
- Use batch size `16` and `PACK_SIZE=2` to match the completed Conv2 stride-1 diagnostic protocol.
- Smoke gate: seeds `0,1`, one epoch capped at 20 train batches and 5 test batches.
- Final diagnostic: seeds `0,1,2`, 30 epochs, `PACK_SIZE=2`, output group `final_conv1_stride1`.
- Saturation diagnostic: init/best/final checkpoints on 2048 train and 2048 test samples, output `conv1_stride1_saturation_train_test_2048.csv`.

Status:

- Local Slurm syntax checks passed for the Conv1 stride-1 smoke/final/saturation launchers.
- Smoke job `1048787` completed successfully.
- Final diagnostic job `1048788` completed successfully with metrics `3/3`.
- Saturation diagnostic job `1048789` completed successfully; output `conv1_stride1_saturation_train_test_2048.csv`.
- Final stride-1 Conv1 test accuracy over seeds `0,1,2`: best/final `98.21%/98.16%` (`98.23/98.13`, `98.25/98.19`, `98.16/98.16` by seed).
- Final test saturation over seeds `0,1,2`: all-unit total `85.33%`, low/high `85.32%/0.001%`, sample p50/p90 `85.24%/86.51%`, layer total h1 `85.33%`. Initial test total saturation was `50.26%`; best-checkpoint test total saturation was `84.61%`.

## Hopfield EqProp Conv3 Stride-1 Diagnostic

Question: does preserving spatial resolution help Conv3 as it helped Conv2?

Plan:

- Keep the current Hopfield protocol: hard sigmoid `[0,1]`, no pooling, `T1=200`, `T2=10`, centered EqProp, beta `0.4`, LR multiplier `2.0`.
- Conv3, channels `[64,128,256]`, kernel `3`, stride `1`, padding `1`, matching the existing Conv3 padding convention while changing stride.
- Use batch size `8` because Conv3 stride-1 keeps large hidden maps; run one seed per GPU for final training rather than packing this heavy run.
- Smoke gate: seeds `0,1`, one epoch capped at 10 train batches and 3 test batches.
- Final diagnostic: seeds `0,1,2`, 30 epochs, output group `final_conv3_stride1`.
- Saturation diagnostic: init/best/final checkpoints on 2048 train and 2048 test samples, output `conv3_stride1_saturation_train_test_2048.csv`.

Status:

- Local Python compile and Slurm syntax checks passed.
- Smoke job `1054867` completed successfully with metrics `2/2`; capped smoke test accuracies were seed 0 `11.46%` and seed 1 `13.54%`.
- Final diagnostic job `1054868` completed successfully with metrics `3/3`.
- Saturation diagnostic job `1054869` completed successfully; output `conv3_stride1_saturation_train_test_2048.csv`.
- Final stride-1 Conv3 test accuracy over seeds `0,1,2`: best/final `99.13%/99.04%` (`99.14/99.08`, `99.17/99.11`, `99.07/98.92` by seed).
- Final test saturation over seeds `0,1,2`: all-unit total `78.83%`, low/high `78.02%/0.82%`, sample p50/p90 `78.84%/79.03%`, layer totals h1 `57.01%`, h2 `49.31%`, h3 `99.05%`.

## MNIST Hopfield EqProp Layerwise Strides 2-2-1 Padding1

Question: does the intermediate Conv3 geometry `[2,2,1]` with padding `1` on
all convolution layers improve MNIST performance while avoiding the full memory
cost of `[1,1,1]`?

Convention:

- Requested/effective layer strides are `2,2,1`.
- Padding is `1` for every convolutional layer and every depth.
- Hidden spatial sizes:
  - Conv1: `28 -> 14`
  - Conv2: `28 -> 14 -> 7`
  - Conv3: `28 -> 14 -> 7 -> 7`
- Result root: `/lustre/fsn1/projects/rech/fmu/${USER}/server_code/results/mnist_hopfield_eqprop_conv_strides221_pad1_20260629`.

Status:

- Local checks passed: Python compile, tiny CPU synthetic Conv3 run with
  `--strides 2 2 1 --padding 1`, and Slurm shell syntax checks.
- Staged to Jean Zay source root `/lustre/fswork/projects/rech/umg/ucy17uy/server_code`.
- Smoke job `1114567` completed successfully with metrics `3/3`.
- Final job `1114568` completed successfully with metrics `9/9`.
- Saturation job `1114569` completed successfully; output
  `mnist_strides2_2_1_pad1_saturation_train_test_2048.csv`.
- Final 30-epoch test accuracy over seeds `0,1,2`: Conv1 best/final
  `97.06%/97.04%`, Conv2 `97.43%/97.39%`, Conv3 `97.67%/97.64%`.
- Final test saturation: Conv1 all-unit total `60.28%`; Conv2 `55.19%`
  with layer totals h1 `41.50%`, h2 `82.57%`; Conv3 `53.69%` with layer
  totals h1 `29.33%`, h2 `39.06%`, h3 `85.37%`.

### Continuation From Final Weights

Question: were the 30-epoch padding-1 runs still undertrained in loss?

- Launcher: `experiments/run_mnist_hopfield_eqprop_conv_strides221_pad1_continue20_jeanzay.slurm`.
- Code change: `experiments/train_mnist_hopfield_eqprop_conv_sweep.py` now accepts
  `--init-checkpoint` and validates checkpoint tensor count/shapes before
  constructing the optimizer.
- Source checkpoints: the 30-epoch `final_model.pt` files under
  `/lustre/fsn1/projects/rech/fmu/${USER}/server_code/results/mnist_hopfield_eqprop_conv_strides221_pad1_20260629/final_mnist_strides2_2_1_pad1`.
- Output root:
  `/lustre/fsn1/projects/rech/fmu/${USER}/server_code/results/mnist_hopfield_eqprop_conv_strides221_pad1_continue20_20260630`.
- Matrix: Conv1 `[2]` pad1, Conv2 `[2,2]` pad1, Conv3 `[2,2,1]` pad1;
  seeds `0,1,2`; 20 more epochs; LR multiplier `2.0`; `T1=200`, `T2=10`,
  centered EqProp beta `0.4`.
- Local checks passed: Python compile, Slurm syntax check, and a tiny CPU
  synthetic checkpoint-continuation smoke.
- Jean Zay checks passed: remote Python compile, Slurm syntax check, and all
  nine source final checkpoints present.
- Slurm job `1129861` submitted on 2026-06-30 11:40 CEST as array `0-8%9`;
  completed with `9/9` metrics.
- Mean best/final test accuracy after 50 total epochs: Conv1 `[2]` pad1
  `97.44%/97.40%`, Conv2 `[2,2]` pad1 `97.74%/97.63%`, Conv3 `[2,2,1]`
  pad1 `97.95%/97.94%`.
- Mean loss did not stagnate over the continuation. Test loss from epoch 30 to
  50 fell Conv1 `0.07143 -> 0.06344`, Conv2 `0.02765 -> 0.02291`, Conv3
  `0.02059 -> 0.01817`; train loss fell Conv1 `0.07085 -> 0.06127`, Conv2
  `0.02598 -> 0.01976`, Conv3 `0.01693 -> 0.01165`.
- Final-checkpoint test saturation after 50 total epochs: Conv1 `66.27%`;
  Conv2 `55.28%` with layer totals h1 `41.34%`, h2 `83.15%`; Conv3
  `54.77%` with layer totals h1 `30.43%`, h2 `40.88%`, h3 `86.04%`.
- Stagewise test saturation from saved checkpoints: Conv1 initial/30-final/50-final
  `49.84%/60.28%/66.27%`; Conv2 `50.13%/55.19%/55.28%`; Conv3
  `50.11%/53.69%/54.77%`. Per-epoch saturation is not recoverable because
  the training jobs saved only `best_model.pt` and `final_model.pt`.
- Saturation CSV:
  `mnist_strides2_2_1_pad1_continue20_saturation_train_test_2048.csv`.

## Fashion-MNIST Hopfield EqProp Conv Stride Grid

Question: does the stride-1 versus stride-2 behavior replicate on Fashion-MNIST across Conv1/Conv2/Conv3?

Plan:

- Dataset: `torchvision.datasets.FashionMNIST`, input scaling `ToTensor()` plus `Normalize(mean=(0.0,), std=(1.0,))`, so pixel values remain in `[0,1]`.
- Architectures: Conv1 `[64]`, Conv2 `[64,128]`, Conv3 `[64,128,256]`; kernel `3`; no pooling; Conv1/Conv2 padding `0`; Conv3 padding `1`.
- Strides: run both `stride=1` and `stride=2` for each depth.
- EqProp/optimizer: same as MNIST Hopfield runs, `T1=200`, `T2=10`, centered EqProp, beta `0.4`, LR multiplier `2.0`.
- Smoke gate: six jobs, one for each depth/stride pair, seed `0`, one epoch capped at 10 train batches and 3 test batches.
- Final grid: 18 jobs, depths `1,2,3`, strides `1,2`, seeds `0,1,2`, 30 epochs.
- Saturation: one pass over both stride groups, init/best/final checkpoints on 2048 train and 2048 test samples, output `fmnist_stride_saturation_train_test_2048.csv`.
- Result root: `/lustre/fsn1/projects/rech/fmu/${USER}/server_code/results/fmnist_hopfield_eqprop_conv_stride_20260629`.

Status:

- Local Python compile and Slurm syntax checks passed.
- Fashion-MNIST dataset was downloaded/prepared on Jean Zay at `/lustre/fsn1/projects/rech/umg/ucy17uy/datasets/fashion_mnist`; launchers use `--no-download`.
- Smoke job `1054870` completed successfully with metrics `6/6`; capped smoke jobs passed for all depth/stride pairs.
- Final grid job `1054871` completed successfully with metrics `18/18`.
- Completed Fashion-MNIST training blocks so far:
  - Conv1 stride-1, seeds `0,1,2`: best/final mean test accuracy `90.12%/89.90%` (`90.07/89.68`, `90.13/89.94`, `90.16/90.08` by seed).
  - Conv1 stride-2, seeds `0,1,2`: best/final mean test accuracy `87.74%/87.73%` (`87.63/87.60`, `87.79/87.79`, `87.79/87.79` by seed).
  - Conv2 stride-1, seeds `0,1,2`: best/final mean test accuracy `91.75%/91.44%` (`91.73/91.24`, `91.73/91.52`, `91.80/91.57` by seed).
  - Conv2 stride-2, seeds `0,1,2`: best/final mean test accuracy `87.67%/87.64%` (`87.81/87.71`, `87.62/87.62`, `87.59/87.59` by seed).
  - Conv3 stride-1, seeds `0,1,2`: best/final mean test accuracy `92.07%/91.61%` (`92.25/91.72`, `92.03/91.50`, `91.94/91.60` by seed).
  - Conv3 stride-2, seeds `0,1,2`: best/final mean test accuracy `86.69%/86.42%` (`86.45/86.24`, `86.77/86.66`, `86.86/86.35` by seed).
- Saturation job `1054872` completed successfully; output `fmnist_stride_saturation_train_test_2048.csv`.
- Final test saturation summary:
  - stride1 Conv1/Conv2/Conv3 all-unit totals `76.72%`, `75.81%`, `76.15%`;
    Conv3 layer totals h1 `55.76%`, h2 `41.04%`, h3 `98.80%`.
  - stride2 Conv1/Conv2/Conv3 all-unit totals `61.05%`, `54.04%`, `43.44%`;
    Conv3 layer totals h1 `32.60%`, h2 `34.71%`, h3 `90.00%`.

## Fashion-MNIST Hopfield EqProp Layerwise Strides 2-2-0

Question: does a Conv3 Hopfield EqProp network improve when the first two
convolutional layers downsample by stride 2 but the third convolution preserves
the spatial grid?

Convention:

- Requested layer strides are `2,2,0` for the first, second, and third
  convolutional layers.
- Since PyTorch convolution stride `0` is invalid, the training CLI records
  requested stride `0` but executes it as effective stride `1`, meaning no
  further spatial downsampling.
- Effective strides by depth: Conv1 `[2]`, Conv2 `[2,2]`, Conv3 `[2,2,1]`.
- Conv1/Conv2 use valid padding `0`; Conv3 uses padding `1`.
- Result root: `/lustre/fsn1/projects/rech/fmu/${USER}/server_code/results/fmnist_hopfield_eqprop_conv_strides220_20260629`.

Status:

- Local checks passed: Python compile, tiny CPU synthetic Conv3 run with
  `--strides 2 2 0`, and Slurm shell syntax checks.
- Staged to Jean Zay source root `/lustre/fswork/projects/rech/umg/ucy17uy/server_code`.
- Smoke job `1114316` completed successfully with metrics `3/3`.
- Final job `1114317` completed successfully with metrics `9/9`.
- Saturation job `1114318` completed successfully; output
  `fmnist_strides2_2_0_saturation_train_test_2048.csv`.
- Final 30-epoch test accuracy over seeds `0,1,2`: Conv1 best/final
  `87.74%/87.73%`, Conv2 `87.68%/87.64%`, Conv3 `87.96%/87.81%`.
- Final test saturation: Conv1 all-unit total `61.05%`; Conv2 `54.04%`
  with layer totals h1 `40.21%`, h2 `86.49%`; Conv3 `55.14%` with layer
  totals h1 `29.73%`, h2 `39.82%`, h3 `88.21%`.
