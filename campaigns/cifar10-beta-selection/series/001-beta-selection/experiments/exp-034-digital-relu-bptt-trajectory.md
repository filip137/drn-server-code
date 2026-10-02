---
id: "exp-034"
title: "Digital-ReLU ours BPTT trajectory and per-block beta across epochs"
status: "running"
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

**Staged October2 15:15.** Frozen source `results/cifar10-l8-digital-relu-bptt-20261002-v1/source/`
(commit `69b03fee` plus the study-script prepare fix), initializer prepared on CPU:
`assets/ours/checkpoints/initial_model.pt`, tensors identical to the Codex linear-output
ours initializer (only embedded config fields differ). Pushed to riri, 3 hashes verified.
Queue smoke `cifar-digital-relu-bptt-ours-smoke-20261002` (priority 7) submitted 15:17;
production `cifar-digital-relu-bptt-ours-20261002` (priority 6, riri only) is submitted
by the coordinator after the smoke completes. Production completion hook:
`run_cifar_digital_relu_bptt collect` → `analysis/ours-collection.json`.

Stage 2 root `results/cifar-digital-relu-trajectory-replay-20261002-v1/`, configs
`configs/cifar/digital_relu_trajectory_replay_20261002/`, script
`experiments/run_cifar_trained_output_replay.py --trajectory` (stage-trajectory/push/
submit --only riri/collect/analyze), jobs `cifar-relu-trajectory-relu-ours-e<epoch>-riri-20261002`.

Coordinator: session background task (scratchpad `coordinator.py`), stateless and
restartable, 2h session window. It submits production after the smoke, stages/pushes/
submits the seven replays after production is collected, and runs the analysis. It also
stages exp-033's `linear-legacy-e10`. Training has priority over the remaining exp-033
replays on riri.

**Smoke passed (15:24, riri):** four updates through the exact trainer/config/assets,
bundle validated by the queue, 3.1s, peak reserved 10.4GiB (16-image validation CE
71.8 after four updates, as in earlier smokes of this recipe). Production admitted on
riri at about 15:31.
