# CIFAR L8 proposed continuation from epoch 30 to 50

Filip requests20 more epochs of the recently completed proposed amplification
run, still on Fifi. One exploratory seed0 CIFAR-10 validation run; no official
test access. Restore final epoch30 of
`cifar10-l8-analog-continue-e10-e30-seed0-20260922-v1/cells/ours_e10_e30`,
SHA256 `e793e4e2480824fdb2e2ff47555f0fb6703355acc8f876d6a0cb05109fe8bfc7`.
The parent completed and passed its final solver audit:87.20% validation,
CE0.3934539339,42210 Adam steps,4.2039GPUh for20epochs.

Restore model/conductances, BN buffers, gains, Adam moments and step counters,
the existing50epoch cosine scheduler and saved CPU RNG states. Preserve every
scientific field except terminal epoch50. Epoch numbers31–50 control ordering,
augmentation and schedule; expected final cumulative70350 Adam steps. Batch32,
proposed voltage/current amplification4/1, T=K[6,6,4], exact selected LR vector,
45k/5k split, crop/flip and cross-entropy remain unchanged. Best checkpoint
selection is validation accuracy then lower CE within the20 new epochs; final
epoch50 is the primary endpoint. Parent metrics remain intact and separate.

Target: Fifi RTX5090, same PyTorch2.11/cu128 environment. GPU-resident training
is required; CPU offloading/fallback is forbidden. Fifi was idle at allocation.
All other authorized hosts and Jean Zay were checked: local/Nom/Trex/Riri/Loulou
occupied, Akib idle, Jean Zay has an unrelated pending job. Do not affect them.
Use the existing tested `source-gpu-only` runner snapshot from the parent study.
Reuse accepted unchanged preparation and solver gate, and repeat final audit.
Smoke the actual epoch30 restore on Fifi with two train batches and one
validation batch. The normal local smoke is moved to the destination because
the local GPU is occupied by an unrelated continuation; no local GPU allocation.

Config: `configs/cifar/continuation_e30_e50_20260922/ours.json`.
Result root: `results/cifar10-l8-analog-ours-e30-e50-seed0-20260922-v1/`,
remote under `/home/filip/server_code/`. Expected4.2h, production cap5h,
total allowance5.1GPUh including checks. A detached direct SSH wrapper records
PID, start/end times, log and exit code. Check process and semantic progress
twice after launch, then process/GPU/heartbeat/metrics/log deltas every30min;
45min without progress requires diagnosis. Deadline is launch+5h1min.
Recover only understood operational faults within this budget and science.

Collect the completed remote bundle locally, validate canonical artifacts,
epochs31–50 with45000 train/5000 validation examples each,70350 final steps,
parent SHA/config/asset identity, final audit and worker exit. Report50 versus30
with single-seed and validation-only limits; record supported conclusions in
`docs/experimental_manifest.md`. No comparison against still-running schemes
at unequal epoch budgets is implied.

## Launch

Fifi resume smoke passes two train batches (64 examples),16 validation examples,
canonical completion and source/parent guards. Production starts10:43:46UTC
September22, wrapper2419188, timeout2419192, GPU worker2419193. First real
training batch is epoch31/50, cumulative step42211. GPU memory11862MiB.
Restoration artifact confirms scheduler epoch30 and all42,210 parent steps.
Expected finish14:56UTC (16:56CEST); timeout deadline15:44:46UTC.
Exact commands and logs are in the result root; monitoring and collection active.

The second launch check confirms batch100, then later batch500, with95–98%
GPU utilization and continuing heartbeat. Complete smoke bundle collected and
validated locally. Passive monitor/collector PID674900 runs from the result
root every30min, logs process/GPU/heartbeat/metrics/log evidence and collects
terminal bundles for validation. Its first cycle succeeds. This helper cannot
diagnose or retry failures: alerts require an agent follow-up. It does not
write scientific conclusions or mark the study reviewed. Status remains
running until terminal collection, then ready-for-review only after checks.

## Completed and collected

Completed 14:56:38 UTC September 22 (16:56 CEST), exit 0. All 20 epochs and
70,350 Adam steps verified in metrics and final checkpoint; scheduler epoch 50.
Final/best validation 89.92% at 50, CE 0.3721765, gain 2.72pp from 30.
Native time 15170.409s (4.2140 GPUh). Final solver audit passes, full local
production/smoke bundles validate, no failures/exclusions/retries, no test
reads. Passive collector 674900 stopped after manual final collection to
avoid stale status overwrites. Fifi GPU free. Analysis/report and manifest
record the single-seed validation-only interpretation. No further run launched.
