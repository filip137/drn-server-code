# CIFAR L8 baseline and legacy continuation from epoch 30 to 50

Filip requests another twenty epochs after the two active epoch30 runs finish,
starting from their last weights. Continue baseline on Nom-cool-1 RTX3090 and
legacy on local RTX3090. Proposed has already completed epoch50 in the separate
`cifar10-l8-analog-ours-e30-e50-seed0-20260922-v1` study; do not duplicate it.

Use each scheme's final epoch30 checkpoint from
`cifar10-l8-analog-continue-e10-e30-seed0-20260922-v1`, after native success,
final solver audit, local collection and canonical validation. Restore all
conductances, trainable affine BN parameters and buffers, trainable gains,
Adam moments/counters, scheduler and saved RNG state. Preserve the original
50-epoch cosine horizon and all scientific settings: seed0, batch32/eval16,
45k/5k CIFAR training/validation split, crop/flip augmentation, cross-entropy,
exact selected layerwise learning rates, bounds and accepted T/K [6,6,4].
Only the terminal epoch, study/arm metadata, parent reference and runtime cap
change. No official test access. This remains exploratory one-seed evidence.

Expected duration about10.5h per GPU. Production caps: baseline12h, legacy14h;
total allowance27 worker GPUh including checks. Results:
`results/cifar10-l8-analog-baseline-legacy-e30-e50-seed0-20260922-v1/`.
Configs: `configs/cifar/continuation_baseline_legacy_e30_e50_20260922/`.
Use the tested GPU-resident runner snapshot from the parent study. Require
`--device cuda --require-gpu-resident`; CPU fallback/offloading is forbidden.
Check live resources before allocation; do not disturb other workers.

After both parents finish, run each exact continuation config in two-batch
smoke mode from its actual final checkpoint, then launch unchanged production
on that scheme's existing GPU. The local smoke is available after legacy exits;
Nom also receives a destination smoke. Cite the unchanged accepted solver gate
and repeat the usual fixed-cohort final audit at50. The earlier eight passing
runner tests already cover Adam/scheduler resume and the GPU-residency guard;
no runner change is planned.

Keep detached wrappers with PID, log, start/end time and exit code. Verify
semantic progress twice after launch, then monitor GPU/process/artifact/log
deltas at least every30min. Diagnose45min stalls immediately. Stop any CPU
training/offloading detected. Recover only understood operational failures
within the declared scientific scope and remaining budget.

Collect remote bundles locally and validate exactly epochs31–50, full45000/
5000 cohorts,70350 final Adam steps, scheduler epoch50, restored checkpoint
state, parent SHA, unchanged science/assets, native exit0 and final audits.
Best checkpoint is selected only within epochs31–50; final50 is the primary
comparison. Preserve earlier best/final checkpoints separately. Compare all
three schemes at matched10/30/50 horizons using the already collected proposed
run, noting different GPU/software stacks and one-seed uncertainty. Record
the conclusion manually in the experimental manifest; no further runs implied.

Inventory checked during final epoch30: Nom/local contain only their intended
continuation compute workers (plus Nom MPS service); Trex/Fifi/Loulou/Riri/Akib
are occupied. Jean Zay has unrelated running job52725. No other job is touched.
Recheck Nom/local immediately after parent completion before new allocation.

## Launch and queue

Both epoch30 parents complete and validate locally, including actual restored
checkpoint tensors/Adam/RNG and final solver audits. Both exact local two-batch
resume smokes pass with parent42210 steps and scheduler epoch30, reaching
step42212. Legacy local wrapper870234/worker870237 starts16:49:42UTC,
first full training batch epoch31,step42211. No CPU offload/fallback.

At allocation Nom had acquired unrelated mumax3 GPU workers; do not share or
interrupt them. Baseline direct wrapper waits up to6h for an idle compute lane,
checks twice10s apart, then runs its exact destination smoke and unchanged
production. This is an explicitly queued job, not running GPU training. The
12h production cap starts only after the queue and smoke. Other authorized
GPU hosts remain occupied; Godzilla SSH authentication still fails. No Jean
Zay job submitted. Source staging first encountered a missing remote parent
directory; creating the declared directory resolved it before any compute.

Nom cleared before the queued wrapper needed a full wait cycle. Destination
smoke passed and validated. Baseline wrapper1963470/worker1963534 starts
16:51UTC at epoch31,step42211; GPU98%,offloadfalse. Both productions now
run concurrently. Local second check reaches batch100, GPU99%. Expected
finish around03:00–03:30UTC September23; hard deadlines local06:50UTC,
Nom04:52UTC. Both inherit all parent state and frozen science unchanged.

Second launch check: baseline batch100 and local legacy batch200, both99–100%
GPU utilization and offloadfalse. All3 smoke bundles collected and validated;
launch_validation.json records their hashes, timings, source identity and
worker handles. Partial50epoch comparison also verifies the separately
completed proposed checkpoint and its70350 Adam steps.

## Halfway milestone, epoch40

Baseline88.78%/CE0.374974 and legacy87.96%/CE0.398051 at40, versus
86.54%/86.88% at30. Both have10 complete new epochs and56280 cumulative
Adam steps, full45000/5000 cohorts. Both remain healthy on their original
GPU-only lanes and continue unchanged toward50. No retries or failures.

## Legacy completes50

Legacy final88.72%/CE0.385642, best88.82% at44. Twenty full new epochs,
70350 Adam steps, final solver audit and exact restored/final checkpoint
checks pass. Worker870237 and wrapper870234 are absent, local GPU has no
compute process. The wrapper did not leave exit-code/finished-at sidecars;
its native exit status is unknown, not inferred as0. Preserve this launcher
receipt omission in logs/local.terminal_observation.json. Canonical terminal
artifacts, checkpoints and clean training log establish scientific completion;
no exploratory training rerun is warranted solely to recreate the receipt.
Baseline remains active in its last two epochs.

## Complete and reviewed September23

Baseline finishes03:17UTC,exit0,final/best89.48% at50,CE0.368819. Legacy
final88.72%,best88.82% at44. Gains30→50 are2.94/1.84pp. Separate proposed
reference89.92% at50 (+2.72pp). Full local copies validate:5new bundles,
all40new epochs,70350final Adam steps, exact restored state and final audits.
Source identity and GPU-residency checks pass. Recorded20.3131workerGPUh
including smokes is within27h. No scientific failures/retries/exclusions;
legacy missing native exit receipt remains explicitly recorded above. Both
workers absent and GPUs released. Manifest interpretation records supported
improvement and one-seed/mixed-hardware limits. No further run launched.
