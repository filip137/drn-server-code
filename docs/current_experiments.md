# Current Experiments

Updated: 2026-06-29 11:26 CEST

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
- Fashion-MNIST stride-grid smoke: `experiments/run_fmnist_hopfield_eqprop_conv_stride_smoke_jeanzay.slurm`
- Fashion-MNIST stride-grid final: `experiments/run_fmnist_hopfield_eqprop_conv_stride_final_jeanzay.slurm`
- Fashion-MNIST stride-grid saturation: `experiments/run_fmnist_hopfield_eqprop_conv_stride_saturation_jeanzay.slurm`
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
- Final diagnostic job `1048788` is running on Jean Zay R3. Latest sampled logs were around epochs `9-10/30`, with test accuracy already around `97.7%`.
- Saturation diagnostic job `1048789` is queued with `afterok:1048788`.

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
- Final diagnostic job `1054868` is running on Jean Zay R3 for seeds `0,1,2`. Latest sampled seed-0 log had reached epoch `2/30`; epoch 1 test accuracy was `97.70%`.
- Saturation diagnostic job `1054869` queued with `afterok:1054868`.

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
- Final grid job `1054871` is running on Jean Zay R3. At the 11:26 CEST check, tasks `3-5` had completed, tasks `0-2` and `6-10` were running, and tasks `11-17` were pending behind the `%9` array limit.
- Completed training block so far: Fashion-MNIST Conv1 stride-2, seeds `0,1,2`, best/final mean test accuracy `87.74%/87.73%` (`87.63/87.60`, `87.79/87.79`, `87.79/87.79` by seed).
- Saturation job `1054872` queued with `afterok:1054871`.
