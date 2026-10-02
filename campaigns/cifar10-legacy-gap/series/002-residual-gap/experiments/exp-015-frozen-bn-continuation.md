---
id: "exp-015"
title: "Frozen-affine BN continuation from epoch 10 to 30"
status: "cancelled"
hypotheses: []
---
# Frozen-affine BN continuation

Filip requested 20 additional epochs for the three existing frozen-affine BN
cases on one RTX5090 and two RTX3090s. This extends the epoch-10 comparison in
[exp-013](../results/exp-013-training-trajectories.md): baseline 70.72%, ours
73.76%, legacy 73.02% validation accuracy. These are frozen-from-initialization
gamma=1, beta=0 runs; normalization and running-stat updates remain active.

Resume each epoch-10 final checkpoint, including model, Adam, scheduler and RNG.
Keep seed0, batch32/eval16, the 45,000/5,000 split, augmentation, amplification,
T/K=[6,6,4], selected LR vectors and original 50-epoch cosine horizon unchanged.
Primary outcome: epoch-30 full-validation accuracy and cross-entropy; report
training loss and scheme gaps, and compare with the existing trainable-BN
epoch-30 results (86.54%, 87.20%, 86.88%). Single-seed, mixed-device exploratory
evidence; no official test read, automatic LR change or further extension.

Parent root: `results/cifar10-l8-bn-ablation-seed0-20260923-v1/`.
New root: `results/cifar10-frozen-bn-e10-e30-20260928-v1/`.
Exact configs: `configs/cifar/frozen_bn_e10_e30_20260928/{baseline,ours,legacy}.json`.
The new root contains frozen `source/`, copied `parents/` and `assets/`,
`launch/{baseline,ours,legacy}.json`, and new `cells/*_frozen_bn_e10_e30/` bundles.
Native runner: `experiments.train_cifar_l8_analog`; queue transport:
`experiments.continue_cifar_frozen_bn`. **No smokes: explicitly waived by Filip.**
Native source/checkpoint identity, full-state restore and nonfinite guards remain.
No additional pre-launch scientific audits. The runner's existing final audit
is retained as part of completion.

## Execution and monitoring handoff

| Case | Target | Expected | Total GPU cap | Queue job |
|---|---|---|---|---|
| Ours | Trex RTX5090 | 5 hours | 8 hours | `cifar-frozen-bn-ours-e30-20260928` |
| Baseline | nom-cool-1 RTX3090 | 10 hours | 14 hours | `cifar-frozen-bn-baseline-e30-20260928` |
| Legacy | local/nom-cool-2 RTX3090 | 10 hours | 14 hours | `cifar-frozen-bn-legacy-e30-20260928` |

Hard deadline: September30 08:00 Europe/Paris; total cap36 GPU-hours across
the three cases, plus 10 CPU-minutes collection per case. Reserve 12,288 MiB
plus 2,048 MiB headroom each using the default queue profile (CIFAR L8).
Use the existing cuDNN9.1 runtime on both 3090s and the existing Trex runtime.
Standing Ben sharing policy and weekday two-5090 fleet cap apply.

Shared run-watch queue is the sole monitoring owner (60-second service cadence).
Inspect: `python /home/filip/.codex/skills/run-watch/scripts/gpu_queue.py status JOB_ID`.
Prepared job commands enforce the remaining queue deadline and preserve completed
epoch checkpoints on a planned pause. No blind retries; recovery returns to the
Codifier, preserves failed attempts, and must fit the original budget.
Completion hooks collect only their own case, validate epochs11–30/42,210 total
Adam steps/full validation and write `analysis/SCHEME-collection.json`.
Run `python -m experiments.continue_cifar_frozen_bn validate SCHEME` from the
frozen source for validation; collection uses the corresponding `collect` mode
with `RUN_WATCH_JOB_ID`. Reviewer owns the three-case comparison after collection.
Current handoff (September28, supersedes the previous launch handoff): Filip
stopped legacy and ours to run exp-016. Both queue jobs are cancelled and their
workers are confirmed stopped. Native paused bundles preserve legacy epoch13
and ours epoch16; incomplete epochs14/17 were discarded. Baseline on nom-cool-1
was also cancelled at Filip's request on September28 at 16:09 Europe/Paris.
Its worker and data-loader processes are confirmed stopped and its GPU allocation
is released. The native paused bundle preserves epoch15 in
`cells/baseline_frozen_bn_e10_e30/checkpoints/final_model.pt`; incomplete epoch16
was discarded. All three cases are now stopped, with partial evidence preserved.
No LR changes were made to exp-015. Do not automatically resume the cancelled cases.

The already-recorded matched epochs11–13 gradient norms and derived RMS values
are retained under `analysis/recorded-gradient-{norms.json,comparison.csv,cohorts.json}`.
New dense-LR work belongs to exp-016 and its separate result root.
