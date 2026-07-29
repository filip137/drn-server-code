# Hopfield Experiments

Updated: 2026-06-29 11:26 CEST

## Goal

Study Hopfield-energy EqProp convolutional networks to understand:

- benchmark accuracy for Conv1/Conv2/Conv3 on MNIST and Fashion-MNIST;
- typical hard-sigmoid saturation values at initialization and after training;
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

Final-checkpoint test saturation over seeds `0,1,2`:

| Model | Stride | Test acc on 2048 subset | All sat | Low / high sat | Layer total saturation |
|---|---:|---:|---:|---:|---|
| Conv1 | 2 | 96.37% | 59.83% | 59.43% / 0.40% | h1 59.83% |
| Conv2 | 2 | 96.50% | 52.44% | 50.22% / 2.22% | h1 40.09%, h2 81.45% |
| Conv3 | 2 | 96.04% | 41.29% | 36.13% / 5.16% | h1 30.63%, h2 32.31%, h3 87.70% |
| Conv2 | 1 | 98.08% | 77.28% | 76.93% / 0.35% | h1 49.62%, h2 93.51% |

Interpretation so far:

- Conv2 stride-1 is clearly better than Conv2 stride-2.
- Conv2 stride-1 is also much more saturated, mostly at the low bound.
- Conv3 stride-2 has high saturation in h3 but low saturation in h1/h2, which
  suggests that the deep representation is unevenly using the hard sigmoid.

## Active MNIST Runs

Conv1 stride-1:

- Smoke `1048787`: completed.
- Final `1048788`: running, around epochs `9-10/30` at the latest sampled logs.
- Saturation `1048789`: pending after final.
- Output group: `final_conv1_stride1`.
- Saturation output: `conv1_stride1_saturation_train_test_2048.csv`.

Conv3 stride-1:

- Smoke `1054867`: completed.
- Final `1054868`: running for seeds `0,1,2`; latest sampled seed-0 log had reached epoch `2/30`.
- Saturation `1054869`: pending after final.
- Output group: `final_conv3_stride1`.
- Saturation output: `conv3_stride1_saturation_train_test_2048.csv`.

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
- Final `1054871`: running, 18 tasks total; tasks `3-5` completed, tasks
  `0-2` and `6-10` running, tasks `11-17` pending behind `%9` at the
  2026-06-29 11:26 CEST check.
- Saturation `1054872`: pending after final.
- Saturation output: `fmnist_stride_saturation_train_test_2048.csv`.

Completed training block so far:

| Dataset | Model | Stride | Seeds | Best test | Final test | Status |
|---|---|---:|---|---:|---:|---|
| Fashion-MNIST | Conv1 | 2 | `0,1,2` | 87.74% | 87.73% | saturation pending |

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
