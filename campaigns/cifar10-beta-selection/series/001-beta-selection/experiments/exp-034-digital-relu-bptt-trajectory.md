---
id: "exp-034"
title: "Digital-ReLU ours BPTT trajectory and per-block beta across epochs"
status: "planned"
hypotheses: ["H-029"]
---
# Which beta works for each block across the digital-ReLU training trajectory?

Filip's goal (October2): with linear analog block outputs plus a digital ReLU
(`block_output_activation: "digital_relu"`), find the beta that works for each block
across epochs. Test [H-029](../../../hypotheses/H-029-digital-relu-beta-across-epochs.md).
Follows [exp-032](../results/exp-032-block-output-activation-replay.md) and
[exp-033](exp-033-trained-linear-output-replay.md).

## Stage 1: BPTT training

Ours analog L8, seed0, fresh initializer prepared on CPU like the Codex linear-output
pilot. Contract identical to that pilot's ours arm
(`configs/cifar/linear_outputs_bptt_20261002/ours.json`): float32, T/K=[6,6,4],
V4/C1, Adam with its learning-rate vector, zero weight decay, trainable BN/gains/head,
batch32/eval16, 10 epochs of the 50-epoch cosine schedule (ratio 0.02), CIFAR10
45k/5k split, no noise or official test. Only `block_output_activation` and
`checkpoint_epochs` [1,2,3,5,7,10] differ. Config
`configs/cifar/digital_relu_outputs_bptt_20261002/ours.json`; study script
`experiments/run_cifar_digital_relu_bptt.py`; root
`results/cifar10-l8-digital-relu-bptt-20261002-v1/`.

Riri RTX5090 only (Filip, October2), shared with Ben. Queue smoke (four updates,
same trainer/config/assets/runtime) first; production after it validates. Reserve
11GiB plus 2GiB; budget 21590s (6GPU-hours), expected about 2.5–3.5h shared;
deadline October3 08:00. Stop on nonfinite values or exhausted budget; queue
recovery may resume from a checkpoint without changing the science.

## Stage 2: replay and selection

Float32 replays of the epoch 0/1/2/3/5/7/10 checkpoints in their own architecture,
with exp-033's runner, cohort, calibration and 23-target grid (1e-6 … 1.0), riri only.
Apply H-029's frozen rule. Also report: windows per block and epoch, calibrated beta
against epoch, and native-BPTT agreement. Validation accuracy against epoch is
reported beside the Codex linear-output arm, descriptively.

## Execution and monitoring handoff

Not yet staged.
