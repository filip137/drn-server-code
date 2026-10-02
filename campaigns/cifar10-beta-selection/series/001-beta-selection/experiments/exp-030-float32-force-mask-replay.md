---
id: "exp-030"
title: "Float32 replay of all18 checkpoints with the inactive-channel force mask"
status: "complete"
hypotheses: ["H-025"]
---
# Does the mask restore block3 in float32 at every checkpoint?

Filip asked (October1) to add the mask to the nudging path, replay all18
checkpoints with it, then run a short float32 training comparison. Test
[H-025](../../../hypotheses/H-025-inactive-channel-force-mask.md), motivated by
[X-002](../../../explorations/X-002-inactive-channel-force.md) and
[exp-029](../results/exp-029-block3-diode-switching.md). Training is
[exp-031](exp-031-float32-masked-training.md).

## Cases and controls

The18 [exp-025](exp-025-all-scheme-layer-rms.md) checkpoints, reused in place
from `results/cifar-layer-rms-all-schemes-float64-20260930-v1/cases/*/inputs/`
with unchanged hashes: shared initializations, EqProp epochs1/5 and BPTT
epochs10/30/50 for baseline/ours/legacy. Same256-image cohort in eight batches
of32, no augmentation/noise/test access, T/K=[6,6,4], frozen BN buffers with
minibatch statistics, no optimizer. **Float32 throughout.** Same21 output
relative-RMS targets (1e-6 … 0.4), 3% tolerance, at most24 evaluations, warm
starts and layer-displacement records as exp-025.

- `masked`: all three blocks, `force_mask: "inactive_output_channels"`.
- `control`: block3 only, no mask; blocks1/2 are identical to `masked` whenever
  no channel is masked there, which each masked bundle records per batch.

Both block-local and native BPTT references use the **unmasked** force. Each
masked batch row also records the reference recomputed with the masked force
(H-025 exactness), inactive-channel count and masked share of Σf².

## Metrics and decision

Primary: per checkpoint/block/arm, the upper end U of the contiguous run of
passing sampled targets that starts at the first passing target, for every
layer's pooled local cosine ≥0.99; also every minibatch ≥0.95 and pooled norm
ratio 0.9–1.1. Apply H-025's three frozen criteria. Report full64 exp-025
limits alongside, without pooling precisions. Keep EqProp and BPTT groups apart.

For exp-031 only, apply this frozen rule to the masked ours cases
(init-ours-e0, eqprop-ours-e1, eqprop-ours-e5): per block, score each target
by its minimum over those checkpoints, layers and minibatches of local cosine;
take the median (larger middle) of targets scoring within 0.001 of the best.
Initial training betas are the masked init-ours-e0 calibrated betas there.

## Budget, stop and expected outputs

Root `results/cifar-force-mask-float32-replay-20261001-v1/`; configs
`configs/cifar/force_mask_float32_replay_20261001/`; runner
`experiments.compare_cifar_block_beta_gradients`; study script
`experiments/run_cifar_force_mask_replay.py` (stage/collect/analyze).
Shared queue, priority3 for the three ours masked cases, 2 other masked, 1
controls; placements Loulou/Trex RTX5090 (Ben sharing authorized) and local
RTX3090. Riri and Fifi are left to the other session's queued conv3 job.
Reserve 11GiB peak plus 2GiB headroom (measured 10.21GiB). Runner caps
1500s masked / 900s control; queue caps 1800s / 1200s. Expected ~8–15min per
masked case and a few minutes per control, about 4–5GPU-hours in total; the
summed queue caps (15GPU-hours) are the ceiling. Admission deadline October2
09:00 Paris. Stop on nonfinite values, identity mismatch, changed
checkpoint/model/BN or exhausted budget; failed cases stay visible and are not
retried automatically. Result note: `../results/exp-030-float32-force-mask-replay.md`.

## Execution and monitoring handoff

Focused CPU tests: `labs/tests/test_cifar_force_mask.py` (6 passed); all CIFAR
tests pass except the pre-existing `test_cifar_selected_gradient_replay.py`
failure in the untouched `replay_without_training_effects` helper. Local GPU
check (pinned torch2.11.0+cu128 runtime, RTX3090): eqprop-ours-e1 masked with
two targets completed in 120s, 10.21GiB peak; bundle valid; block3 had 4
inactive channels per batch carrying 99.96–99.98% of Σf²; masked-reference
minimum cosine 1−1.6e-13 and norm ratio exactly 1. Preserved in `smoke/`.

Staged October1 19:05 Paris: 36 configs, frozen source (465 files) with
`source/source_identity.json`, `cases.json` and `launch/<case>-<arm>.json`.
Source synced to Loulou/Trex; exp-025 inputs copied to Trex (hashes match).
All 36 jobs submitted 19:07 as `cifar-force-mask-fp32-<case>-<arm>-20261001`
(`launch/submission.log`). At 19:10 Loulou and Riri carried Kellian's new
24GiB jobs and Trex another user's, so only the local 3090 admits work; the
queue moves jobs to Loulou/Trex automatically if memory frees. Measured
throughput on the 3090 is about 15min per masked case, so local-only coverage
would take about 5–6h. The shared queue owns admission, monitoring and the
CPU completion hook (`run_cifar_force_mask_replay collect CASE ARM`, writing
`analysis/<case>-<arm>-collection.json`); root owns analysis (`analyze`) and
review. Inspect with `gpu_queue.py status JOB_ID`.

All 36 jobs completed and were collected/validated by the queue by October1
21:30 Paris; no failures, retries or remaining jobs. Reviewed in
`../results/exp-030-float32-force-mask-replay.md`.
