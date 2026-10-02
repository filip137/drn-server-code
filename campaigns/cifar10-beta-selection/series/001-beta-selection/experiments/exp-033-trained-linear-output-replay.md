---
id: "exp-033"
title: "Float32 trained-checkpoint replays with linear block outputs"
status: "running"
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

**Staged October2 13:05** at commit `65db8aa0`: frozen code in `source/`, 18 input
roots in `cases/` (15 transplants; `linear-ours-e1/e5`, `linear-legacy-e1`), each
config identified by SHA256 in `cases.json`. CPU check: transplanted and linear
checkpoints restore into their architectures with finite logits. Pushed to
trex/fifi/loulou/riri with all 91 hashes verified on each (trex after one transient
SSH failure during its reboot). Queue jobs `cifar-trained-output-<case>-20261002`;
smoke `transplant-eqprop-ours-e1` admitted on trex at 13:12, the other 17 held.

Coordinator (session background task, scratchpad `exp033_watch.py`): releases the
held jobs once the smoke bundle validates, keeps them held on a smoke failure, and
stages/pushes/submits `linear-ours-e10`, `linear-legacy-e5` and `linear-legacy-e10`
as Codex writes them. The queue owns placement, monitoring and the completion hook
`run_cifar_trained_output_replay collect CASE`. The coordinator's session limit is
2h (until about 15:10); it must be restarted for legacy e10 (expected about 16:25).

**User-directed interruption, October 2 about 13:50 Paris:** Filip requested
freeing fifi for the linear-output BPTT continuation. Queue job
`cifar-trained-output-transplant-bptt-baseline-e50-20261002`, attempt
`c30900124e5d4cd3a9230064ed62b171` (PID 2043553), was cancelled through its verified
supervisor; the queue confirmed cancellation. Preserve its partial output and
receipts as an interrupted case, not a scientific failure. The user accepted
the already-sent cancellation and reaffirmed the 40-epoch training continuation.
Other replay jobs remain unchanged; do not automatically restart this cancelled
case onto fifi ahead of `cifar-linear-bptt-ours-e10-e50-20261002` (priority 10).

**Placement restricted to riri (Filip, October2 13:50).** By then the smoke had
validated (13:26) and released jobs had completed on trex/fifi/riri: baseline
e1/e5/e10/e30 and ours e1/e5 transplants. `transplant-bptt-baseline-e50` was
cancelled on fifi by someone outside this session. The 12 jobs still pending were
cancelled before starting and resubmitted with riri-only placements as
`cifar-trained-output-<case>-riri-20261002`, plus baseline e50. Their remote run
directories are `cells/<case>-riri/`; the local collection path is unchanged.
The coordinator restarted riri-only (stages legacy e5/e10, pushes to riri only).
