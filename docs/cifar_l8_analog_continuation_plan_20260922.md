# CIFAR L8 continuation from epoch 10 to epoch 30

Filip requested twenty additional epochs for the three selected LR settings,
initially distributed over Akib, Nom-cool-1, and one RTX5090. Filip subsequently
required GPU-only training and authorized local/Godzilla. The active placement
is local RTX3090, Nom-cool-1 and Fifi; the stopped Akib attempt is preserved
below. This is exploratory
CIFAR optimization evidence, seed0, validation only; no official-test access.

Resume the final epoch10 model, all Adam moments/step counters, BN buffers,
boundary gains, scheduler state and CPU RNG state. Preserve all scientific
config fields, including batch32, affine trainable BN, crop/flip augmentation,
cross-entropy, solver T=K[6,6,4], conductance bounds and the original50epoch
cosine horizon. Epoch11 uses the epoch10 scheduler state, and epoch-specific
data ordering/augmentation use absolute epoch numbers. Do not restart training.

| Scheme | Parent cell in completed September21 LR study | Target | Runtime cap |
|---|---|---|---|
| baseline | baseline_c1_d1_e10 | Nom-cool-1 RTX3090 | 12h |
| proposed | ours_edge_head_down_e10 | Fifi RTX5090 | 5h |
| legacy | legacy_c1_d1_e10 | Local RTX3090, GPU-only replacement | 14h |

Current expected durations: about10.5h Nom,4.3h Fifi,10h local.
Initial Nom estimate7–9h was optimistic; complete epochs give about10.5h.
The initial Akib estimate33h is superseded by the local replacement.
Total cap60workerGPUh remains unchanged. Active per-run caps total31h;
account separately for the stopped Akib attempt and checks.
No fourth LR control or new search is included. All authorized simulation
hosts and Jean Zay were checked before allocation; requested three lanes idle.

Result root: `results/cifar10-l8-analog-continue-e10-e30-seed0-20260922-v1/`.
Exact configs live in `configs/cifar/continuation_20260922/`.
Parent study: `cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1`.
Record parent checkpoint hashes and accepted unchanged preparation assets.
Keep new source and output bundles isolated; preserve the closed parent study.

Gate resumption with bitwise CPU/CUDA model, Adam and scheduler agreement
between interrupted and uninterrupted training, including nonzero moments
and an epoch10 boundary. Then smoke each full-width config from its real
epoch10 checkpoint, without CPU offloading. Reuse the accepted solver
operating points; repeat the standard fixed-cohort final solver audit at30.
New metrics contain epochs11–30; cumulative steps retain the parent offset.
The new best checkpoint means best within epochs11–30, explicitly recorded;
epoch30 validation is the main endpoint. Parent metrics remain separate.

Hardware/software change on Nom and local can change floating-point trajectories;
do not claim bitwise cross-hardware continuation or a clean scheme advantage.
The original Akib attempt used saved-tensor CPU storage; the user disallowed
that execution mode and it was stopped before a complete new epoch.
The model has no stochastic CUDA training operations; old checkpoints store
CPU torch, NumPy and Python RNG states only.

Direct detached SSH commands record worker handles, logs and exit status.
Verify each intended process and semantic progress after launch. Check process,
GPU, artifact/heartbeat and log progress at least every30min until terminal;
45min without progress triggers diagnosis. Deadline is the recorded cap per
worker. Recover understood operational failures only within remaining budget.
Collect remote bundles locally, validate20 new complete epochs per scheme,
Adam cumulative steps, unchanged config/split/parent SHA, and final audits.
Report epoch20 and30 against parent10; update the dashboard and manually
record conclusions after collection. No unrelated jobs may be disturbed.

## Launch record

Seven CPU/CUDA regression tests pass. The initial local CUDA invocation used
the incompatible default cuDNN and failed before training; using the existing
study-local cuDNN9.1 runtime resolves it. All three local real-checkpoint
two-batch smokes pass. Destination ten-batch smokes pass: Fifi5.45s,
Nom12.13s, Akib43.10s. No scientific setting changed.

Production wrapper/worker PIDs: Fifi2382803/2382807,
Nom1867676/1867680, Akib436494/436498. Each has its own timeout, log and
exit-code sidecar in `logs/`; exact scripts are `launch-TARGET.sh`.
Restoration validates parent bundle/SHA, asset identity and Adam cumulative
step14070, then starts epoch11. Monitoring and collection remain active.

## User-directed GPU-only replacement

Filip disallowed CPU execution and CPU offloading, and authorized Godzilla or
the local GPU in addition to a Jean Zay V100 fallback. Stop any future training
that uses CPU fallback/offloading. Ordinary CPU data loading is unaffected.
Akib's GPU worker436498 and process group436497 were terminated; wrapper
exit143 and no surviving worker confirmed, GPU idle287MiB. No new epoch was
completed. Preserve the failed/cancelled bundle under `stopped-akib/` locally.
This is an operational/user-directed exclusion, not scientific failure.

Godzilla SSH authentication failed; local RTX3090 verified idle with no compute
worker and about23GiB free. Use local for the legacy replacement, original
epoch10 parent, same scientific configuration, runtime cap14h. Expected10–12h.
The existing60GPUh total cap remains sufficient and unchanged. No Jean Zay
job was submitted. Fifi/Nom continue unchanged without offloading.

New `source-gpu-only` adds an explicit `--require-gpu-resident` guard: rejects
CPU device, unavailable CUDA, CPU saved-tensor offload, or CPU trainable tensors.
The replacement smoke and production use it with `--device cuda` and no
offload flag. Immutable earlier source snapshots and results remain intact.

GPU-only replacement launched after eight passing tests and the exact full-width
ten-batch smoke (13.38s). Local wrapper572996, timeout572998, worker572999.
First training progress confirms epoch11/30, step14071, CUDA device and
`save_tensors_on_cpu=false`; GPU about12GiB/94% utilization. Akib bundle
collected and validated as intentionally cancelled with exit143.

## First epoch20 milestone (partial study)

Proposed/Fifi reaches epoch20 validation83.10%, CE0.487596, versus75.36%
at10 (+7.74pp). All45000 training and5000 validation examples are present;
cumulative Adam steps28140. This is a partial single-scheme milestone, not
a terminal comparison. The run continues unchanged to30; baseline and
legacy have not yet reached20. Full curve is in `analysis/report.md`.

## Proposed completes epoch30; other schemes still active

At10:24UTC September22, proposed/Fifi is complete and collected locally.
Canonical validation passes,20 new epochs (11–30), cumulative42210 steps,
final solver audit passes, native exit0. Epoch30 is also best: validation87.20%,
CE0.393454, versus75.36% at10 (+11.84pp). Native elapsed15134.09s
(4.2039GPUh). Fifi GPU is idle2MiB/0%, worker absent.
Baseline has completed18 at82.42% and trains19 on Nom; legacy has completed17
at80.34% and trains18 locally. Both CUDA, offloadfalse. Approximately6–6.5h
remain. This is still a partial study; do not rank unmatched epoch budgets.

Baseline/Nom reaches epoch20 validation83.16%, CE0.488511, compared with
75.26% at10 (+7.90pp). It continues unchanged toward30. Proposed's matched
epoch20 accuracy was83.10%; the0.06pp difference is three validation examples
and does not establish a scheme advantage with one seed and different GPUs.
Legacy's epoch20 remains pending at this milestone.

## All three epoch20 milestones available (study still partial)

Validation at20: baseline83.16%/CE0.488511, proposed83.10%/CE0.487596,
legacy82.06%/CE0.514879. Gains over the respective epoch10 parents are
7.90,7.74,7.38 percentage points. Longer training at the frozen rates improves
all three; this does not establish a global LR optimum or a robust ranking
with one seed and differing GPU/software stacks. Baseline and legacy
continue unchanged toward30, while proposed has completed30 at87.20%.

## Completed epoch30 comparison

All three complete and validate locally: baseline86.54%/CE0.398059,
proposed87.20%/CE0.393454, legacy86.88%/CE0.399088. All final audits pass;
42210 Adam steps each. Eleven bundles reconcile, including7 smokes and the
preserved excluded Akib cancellation. Checkpoint verification passes with
only inert PyTorch scheduler verbose=False metadata normalized for the2.5
restores. Recorded training/smoke/cancelled cost25.2037GPUh. All successful
workers exited0. Interpretation recorded in experimental_manifest.md.
Baseline and legacy epochs31–50 are authorized as a new separate study.
