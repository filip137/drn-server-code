# Conv1 quarter-amplitude targets, batch16

Status: baseline and ours running; legacy queued for the next RTX3090 slot, September29,2026.

## Question and design

Filip requested Conv1 with target amplitude1/4 for all three amplification
schemes and batch16. Schedule baseline(1,1), ours(4,1) and legacy(4,0.25),
noiseless BPTT, seed0, ten epochs. These defaults retain the established Conv1
target-encoding recipe. This extends the
[Conv2 target-encoding intervention](../why_does_legacy_perform_better/series/001-clean-voltage-and-encoding/experiments/exp-001-conv2-target-encoding.md).

Copy the corrected seed0 wide BPTT Conv1 paper-ready configs from
`paper_ready_results/bundles/table1_wide_bptt/conv1/`. Set `target_scale=0.25`,
meaning quarter-amplitude one-hot labels in native paired-output MSE, without
compensating loss normalization. Retain original scheme-specific Adam rates,
nativefloat32, T/K, input gain, perfect-diode dictionaries, zero frozen biases,
[0,100] bounds and deterministic ordinary-MNIST55k/5k training/validation split.
Train/validation batch sizes16/64; official test disabled. Use one shared,
verified seed0 initializer and loader order across all three schemes. Do not
transfer the preceding Conv2/Conv3 doubled-rate pilot's learning rates here.

The source snapshot must include the established target-scale implementation,
preferably the verified Conv2 target-encoding snapshot, rather than importing
unrelated live worktree edits. Record precise source/config/init identities
and explicit precision. This is a separate regime from the float64 read-noise runs.

Primary outcome: final epoch10 validation accuracy and loss for each scheme;
also report best validation over the same ten epochs as a secondary metric.
Retain all outcomes, including negative or nonfinite results. Historical target1
references are descriptive only unless source/init/cohort matching is verified.
One seed cannot establish significance or a general scheme ranking. No new
target1 controls, alternative amplitudes, learning-rate tuning or extra seeds.

## Budget and execution

Three independent jobs, allowed placements local and nom-cool-1 RTX3090s,
sharing with identified Ben workloads when memory plus headroom fits. Use the
existing shared run-watch queue, one worker per GPU, so two can run in parallel
and the third starts on the first available placement. No interference with
other jobs or MPS. Standing laboratory admission rules apply.

Ceiling3GPUh total, including smokes, production, validation and any recovery;
at most1GPUh per case. Provisional expected5–15minutes per case after admission,
to be updated from actual epoch timing. Use synchronous same-runner smoke,
existing exact_run and CPU collection/validation. Preserve checkpoint/Adam state
and prior attempts if recovery is needed; complete only the original ten epochs.
No automatic scientific changes to fit memory or improve accuracy.

Root: `results/conv1-target-quarter-batch16-20260929-v1/`, including every
source, smoke, attempt, transport receipt and collected result.
Configs: `configs/conv/target_scale_conv1_20260929/`.

## Execution and monitoring handoff

Three separate jobs were submitted to the existing shared GPU queue; each can
use local or nom-cool-1 RTX3090. Queue IDs:
`conv1-target-quarter-baseline-b16-s0-20260929`,
`conv1-target-quarter-ours-b16-s0-20260929`,
`conv1-target-quarter-legacy-b16-s0-20260929`.
The default resource profile permits one worker per GPU, allowing two simultaneous
cases and then the third on the next available placement. Job specs and exact
commands are under `launch/jobs/` within the result root.

Both full-batch CPU and CUDA smokes passed for all three cases. Each smoke uses
three epochs with two optimizer batches per epoch; canonical validators report
zero errors. CUDA peak allocated87MiB/reserved120MiB. Real epoch003 optimizer
checkpoints contain private sampler/worker continuation state. Receipts:
`local-smoke/validation.json` and `gpu-smoke/validation.json`.
Production uses the same frozen runner with explicit `--index 0`,
`--checkpoint-every-epoch --epoch-chunk-size 10 --skip-terminal-official-test`.
Scientific config, batch16 and nativefloat32 remain unchanged.

Frozen source identity:
`77485c534de83b058bec7ebea18e5ae436248647272fa9fb6c88c2fcc846ce9e`.
`preparation.json` records origin verification, original LR vectors and exact
shared initializer equality. `cases.json` provides every case/config identity
and exact production bundle path, including the config filename prefixes.
Source and inputs are staged under the identical result-root name on nom-cool-1
at `/home/filip/server_code/results/conv1-target-quarter-batch16-20260929-v1/`.

Each queue job has3540seconds production/validation allowance plus60seconds
reserved for its short smoke/transport overhead, preserving the original1GPUh
per-case ceiling. The expectation remains provisional5–15minutes per case,
to be revised after actual epoch timing under observed sharing.

The shared queue is the sole routine monitor, with60second cadence; no second
watcher is assigned. Inspect with
`python /home/filip/.codex/skills/run-watch/scripts/gpu_queue.py status JOB_ID`.
At10:46Paris, baseline is running on nom-cool-1 (queue childPID176296),
with actual epoch1 batch500/3438 and finite loss; ours is running locally
(PID1164627), already in epoch2 after first-epoch validation95.14%.
The first local epoch took approximately52seconds including startup. Legacy
is queued for the next available3090. Shared servicePID1124310 has a fresh
heartbeat; handles and direct evidence are in `launch/monitoring-handoff.json`. Queue exceptions
return to the Codifier; preserve failed attempts and charge the same allowance.
Any recovery restores model/Adam/RNG and only the remaining original epochs.

The queue invokes CPU-only `collect.py` after target validation. Remote bundles
are copied to `collected/nom-cool-1/` and validated locally using the frozen
reporting helper. Completion requires all three10-epoch histories,34380Adam
steps each,11checkpoints including initialization, finite metrics, target_scale0.25
in the consumed config, float32, zero biases, matching shared initialization,
complete future continuation state and zero official-test evaluations.
Per-case receipts appear under `launch/collected/`; a failed safe collection can
be repeated via `gpu_queue.py collect JOB_ID` without rerunning training.
Scientific comparisons remain with the root Reviewer after all outcomes are
collected; submission alone is not evidence completion.

## Results and review

Pending. Scheduling does not imply completed evidence.
