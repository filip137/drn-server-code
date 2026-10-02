---
id: "exp-031"
title: "Five-epoch float32 hybrid EqProp with versus without the force mask"
status: "complete"
hypotheses: ["H-026"]
---
# Can float32 training work once inactive channels are masked?

Filip asked (October1) for a short float32 training comparison after the masked
replays. Test [H-026](../../../hypotheses/H-026-float32-masked-training.md);
targets come from [exp-030](exp-030-float32-force-mask-replay.md).

## Cases and controls

Two fresh ours runs differing only in `conv_eqprop.force_mask`:
`mask` (`"inactive_output_channels"`) and `nomask` (`"none"`). Base contract:
[exp-026](exp-026-float64-periodic-vs-fixed-beta.md)'s periodic arm — seed0
initializer (SHA256 `c4d98ef6…37`), CIFAR10 45k/5k split, batch32, same
augmentation and data order, Adam with the ours_edge_head_down_e10 learning
rates, 50-epoch cosine schedule stopped after5 epochs (7035 updates), trainable
BN affine and gains, V4/C1, blocks[3,3,2], free/nudged T/K=[6,6,4], four
calibrations per epoch on the 256-image cohort (3%, ≤24 evaluations), training
RMS measured from the actual phase pairs, no cosine/boundary replays, no read
noise, no official test access. **Float32 throughout.** Hybrid gradients:
autograd for boundary forces/BN/gains/readout; stable centered EqProp for the
eight convolutions.

Output-RMS targets and initial betas: exp-030's frozen rule on the masked ours
replays; both arms use the same values. Reference points, not controls:
matched native BPTT ours epoch5 66.68%/CE0.941809 and earlier float32 EqProp
ours0.3× 54.12%/CE1.255383 ([exp-013](../results/exp-013-all-scheme-long-eqprop.md)).

## Metrics and decision

Primary: epoch5 validation accuracy and CE, evaluated every epoch; no best-epoch
selection. Apply H-026's frozen rule. Report per-epoch accuracy/CE, betas and
achieved calibration RMS, training-time RMS, inactive-channel counts and masked
Σf² share (mask arm), and runtime.

## Budget, stop and expected outputs

Placement: Jean Zay `fmu@v100`, `gpu_p13`, `v100-32g`, one GPU per arm, both arms
in the same `pytorch-gpu/py3/2.5.0` environment. At 19:10 Paris the lab 5090s
were occupied by other users' long jobs, and the local 3090 is serving exp-030.
Follow the Jean Zay pre-submit gate: frozen source, bash/py_compile/focused
tests, `sbatch --test-only`, one live `qos_gpu-dev` canary of the mask arm in
smoke mode (four updates, four calibrations), verified canary receipt, then
both production jobs once on `qos_gpu-t3`.

Production limit 9h per arm (trainer cap 8.5h with epoch-boundary planned
pause), expected about 5–6h per arm; canary ≤1h. Total ceiling 19 V100
GPU-hours. Admission deadline October2 12:00 Paris. Stop on nonfinite values,
changed initializer/source, invalid precision or exhausted budget; a failed
beta calibration is that arm's recorded outcome. No automatic retry or
extension. Root `results/cifar-float32-masked-training-20261001-v1/`; study
script `experiments/run_cifar_masked_training_pair.py`; result note
`../results/exp-031-float32-masked-training.md`.

## Execution and monitoring handoff

**Targets (frozen rule applied October1 19:41):** output RMS [0.005, 0.02, 0.0025]
for blocks1/2/3, initial betas [2.9285e-4, 2.1732e-4, 5.5135e-4] from masked
init-ours-e0. Worst ours minibatch/layer scores 0.9788/0.8949/0.9698; block2 is
limited by the float32 floor at initialization, where no sampled target passes
pooled 0.99. Block3 tied at [0.001, 0.0025]; larger middle chosen. Record:
`results/cifar-float32-masked-training-20261001-v1/inputs/target_selection.json`.
Configs `configs/cifar/float32_masked_training_20261001/{mask,nomask}.json`
differ only in `arm_id` and `force_mask`.

**Placement changed before any production submission.** Source, assets and
Slurm scripts were staged on Jean Zay with matching hashes and `sbatch
--test-only` passed, but every `qos_gpu-t3` estimate on `fmu@v100` was October11
for 1–9h limits; only `qos_gpu-dev` started promptly. No Jean Zay job was
submitted. Meanwhile nom-cool-1's RTX3090 became free (Ben shares it) and Trex
reopened. Both arms now run through the shared queue restricted to RTX3090s
(local, nom-cool-1) with the same pinned PyTorch2.11.0+cu128 runtime, priority10
above exp-030. Reserve 11GiB plus 2GiB (smoke peak 10.78GiB); queue cap 36000s
per arm, trainer stop-before 9.5h and config cap 8.5h with epoch-boundary
pause. Total ceiling 20 GPU-hours.

Smoke (October1 19:43, nom-cool-1, mask arm, same trainer/config/assets,
`--mode smoke --smoke-batches 4`): exit0, valid bundle, 137.8s, four
calibrations. Calibrated betas grew about 1000x within three updates and
16-image validation CE was 33.4, showing large early response changes under the
inherited Adam contract; production uses four calibrations per epoch as
declared. Remote bundle `.../smoke/mask` on nom-cool-1.

Jobs `cifar-fp32-masked-training-{mask,nomask}-20261001` submitted 19:47
(`launch/{mask,nomask}-lab.json`). The queue owns monitoring and the CPU
completion hook `run_cifar_masked_training_pair collect ARM`; root reviews.

**Completion (October1).** mask finished 23:33 Paris on nom-cool-1 (3.80h);
nomask finished 23:20 on the local nom-cool-2 RTX3090 (3.54h; the queue labels it
`local`). Both completed 5 epochs, 7035 updates and 20 calibrations, with exit 0,
and the CPU collection hook validated each bundle
(`analysis/{mask,nomask}-collection.json`). No retries, no official test read,
no Jean Zay submission. The only top-level result root is
`results/cifar-float32-masked-training-20261001-v1/`; the smoke bundle stays
remote on nom-cool-1. Reviewed October2:
[result note](../results/exp-031-float32-masked-training.md).
