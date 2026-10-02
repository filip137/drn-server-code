---
id: "exp-032"
title: "Initialization replay with diode, digital-ReLU and linear block outputs"
status: "complete"
hypotheses: ["H-027"]
---
# How wide is the beta window when block outputs are linear?

Filip asked (October2) to test linear block-output layers (a) with a digital ReLU
at the end of each block and (b) without one, at initialization. Test
[H-027](../../../hypotheses/H-027-linear-block-outputs.md). BPTT training of the
`linear` variant is the separate pilot
[cifar-analog-linear-block-outputs-20261002](../../../../pilots/cifar-analog-linear-block-outputs-20261002.md).

## Cases and controls

One checkpoint: the ours seed-0 initializer (SHA256 `c4d98ef6…6c37`), reused from
exp-025's `init-ours-e0` inputs. Three architectures sharing its tensors:

- `perfect_diode`: original analog diode output layer (control);
- `digital_relu`: linear analog output layer; ReLU(E)+ReLU(−I) applied digitally
  before max-pool, which is the function the analog diode computes;
- `linear`: linear analog output layer, decoded E−I with no rectifier.

The five internal diode layers, widths, BN, gains, head, V4/C1 and T/K=[6,6,4]
are unchanged. Each in float32 and float64 (six cases). exp-025 procedure
otherwise: 256-image cohort in eight batches of32, no augmentation/noise/test
access, minibatch BN statistics with frozen buffers, no optimizer, block-local
centered EqProp from native-K endpoints versus the matched frozen-current local
BPTT reference (and native BPTT), 3% target tolerance, at most24 evaluations.
Grid: exp-025's 21 relative-RMS targets (1e-6 … 0.4) plus 0.7 and 1.0.

## Metrics and decision

Per case and block: worst-layer pooled local cosine across the grid; window
[first passing, U] for pooled ≥0.99; strict window (also every minibatch ≥0.95,
norm ratio 0.9–1.1). Apply H-027's frozen rule. Report float32 floors,
common strict float32 targets, native agreement and betas.

## Budget, stop and expected outputs

Shared run-watch queue, RTX5090 (trex, fifi, loulou, riri) or local RTX3090,
pinned PyTorch2.11.0+cu128. Reserve 11GiB (float32, measured 10.2) or 21GiB
(float64, measured 20.4) plus 2GiB headroom. Runner caps 1500s / 4500s; queue caps
1800s / 5400s; expected about 10min / 40min each. Total ceiling 7.2GPU-hours.
Admission deadline October2 22:00 Paris. Float32 jobs first; the first collected
float32 bundle is the execution smoke and float64 jobs stay held until it
validates. Stop on nonfinite values, identity mismatch, changed checkpoint/BN or
exhausted budget; no automatic retry. A failed calibration is that case's
recorded outcome.

Root `results/cifar-block-output-init-replay-20261002-v1/`; configs
`configs/cifar/block_output_init_replay_20261002/`; runner
`experiments.compare_cifar_block_beta_gradients`; study script
`experiments/run_cifar_block_output_replay.py` (stage/push/submit/collect/analyze).
Result note `../results/exp-032-block-output-activation-replay.md`.

## Execution and monitoring handoff

Model option `block_output_activation: "digital_relu"` added beside Codex's
`"linear"` (`labs/cifar_l8_analog.py`, default unchanged). Tests
`labs/tests/test_cifar_digital_relu_outputs.py` (7 passed) and
`test_cifar_analog_linear_outputs.py` (6 passed). The full CIFAR test set has two
failures that are unrelated and also fail at commit `2b400943` without this change:
`test_cifar_selected_gradient_replay` (known) and
`test_cifar_l8_amplification::test_legacy_block_scaling_and_bn_boundary_cancellation`
(BatchNorm eps=0). On a tiny model at small beta, all three activations match local
BPTT (cosine 1.000000 at beta ≤1e-6 in float64).

**Staged October2 11:40.** Frozen source `results/cifar-block-output-init-replay-20261002-v1/source/`
at commit `b26388b1`; input roots `cases/{perfect_diode,digital_relu,linear}/inputs/`
(checkpoint hard-linked from exp-025, model configs differ only in
`block_output_activation`). Pushed to trex/fifi/loulou/riri with all 13 hashes
verified remotely. Queue jobs `cifar-block-output-init-{activation}-{precision}-20261002`
(specs in `launch/`); the queue owns placement, monitoring and the completion hook
`run_cifar_block_output_replay collect CASE`. Float32 jobs queued at 11:47; float64
jobs submitted held. A background watcher releases float64 when the first float32
bundle validates and leaves it held on any float32 failure (deadline 22:00).
At submission no approved GPU had room: Codex's linear-output BPTT jobs hold the local
3090 and fifi until about 16:15 (expected), and the other 5090s carry other users'
jobs not covered by sharing permission.

**Scope reduced (October2 13:00, Filip).** Finish only `digital_relu-float32`
(running on trex since 12:54, shared with Ben and another user's job). The other
five jobs are held, not cancelled; the float64 auto-release watcher is stopped.
Filip expects the important features to appear only after training, so the next
step will be chosen after this case. H-027's frozen rule cannot be applied without
the control and float64 cases; this case is descriptive only.
