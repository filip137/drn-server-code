---
id: "exp-033"
title: "Float32 trained-checkpoint replays with linear block outputs"
status: "planned"
hypotheses: ["H-028"]
---
# Does the beta window stay open after training when block outputs are linear?

Filip asked (October2 13:00) for float32 replays of trained networks with linear
block outputs, since the block3 collapse appears only after training. Test
[H-028](../../../hypotheses/H-028-trained-linear-outputs.md); follows the
initialization case of [exp-032](exp-032-block-output-activation-replay.md).

## Cases and controls

- `transplant-<case>` (15): exp-025's trained checkpoints (baseline/ours/legacy,
  EqProp epochs1/5, BPTT epochs10/30/50), weights and BN unchanged, model config
  changed only to `block_output_activation: "digital_relu"`. Baseline is included
  because baseline e1 is one of the two checkpoints where the exp-030 mask failed.
  Control: exp-030 unmasked float32 diode replays (block3 from its `control` arm,
  blocks1/2 from its `masked` arm, which had no inactive channels there; asserted).
- `linear-<scheme>-e<epoch>` (6): Codex's linear-output BPTT checkpoints
  ([pilot](../../../../pilots/cifar-analog-linear-block-outputs-20261002.md)), ours
  and legacy, epochs1/5/10, with the exact config embedded in each checkpoint.
  Initial calibration guesses are each scheme's exp-025 BPTT e10 values.

exp-025 procedure otherwise: 256-image cohort in eight batches of32, no
augmentation/noise/test access, minibatch BN statistics with frozen buffers, no
optimizer, centered EqProp from native-K endpoints against the matched frozen-current
local BPTT reference (and native BPTT), T/K=[6,6,4], 3% target tolerance, at most24
evaluations. Grid: exp-025's 21 targets plus 0.7 and 1.0. **Float32 only.**

## Metrics and decision

Per case and block: worst-layer pooled local cosine across the grid, window
[first passing, U] for pooled ≥0.99, and the strict window. Apply H-028's frozen
rule to `transplant`; report `linear` descriptively.

## Budget, stop and expected outputs

Shared run-watch queue on RTX5090 (trex, fifi, loulou, riri) or the local RTX3090,
pinned PyTorch2.11.0+cu128; reserve 11GiB plus 2GiB headroom (measured 10.2GiB).
Runner cap 1500s, queue cap 1800s, expected about 10min each: 21 jobs, about
3.5GPU-hours expected, 10.5 ceiling. Linear cases are staged as Codex writes each
checkpoint (ours e10 about 13:40; legacy e5 about 14:00, e10 about 16:25). Admission
deadline October3 08:00 Paris. The first collected bundle is the execution smoke;
later jobs stay held until it validates. Stop on nonfinite values, identity
mismatch, changed checkpoint/BN or exhausted budget; no automatic retry. A failed
calibration is that case's recorded outcome.

Root `results/cifar-trained-output-replay-20261002-v1/`; configs
`configs/cifar/trained_output_replay_20261002/`; study script
`experiments/run_cifar_trained_output_replay.py`. Result note
`../results/exp-033-trained-linear-output-replay.md`.

## Execution and monitoring handoff

Not yet staged.
