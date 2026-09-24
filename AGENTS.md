# AGENTS

## Purpose

This repository contains the simulations and evidence for a paper on
bidirectional amplification in dissipative resistive networks.

## Agent Authority

This is a trusted, agent-operated research worktree. Within an assigned task,
agents have standing permission to:

- read, create, edit, move, and remove files in this worktree while preserving
  unrelated user changes and irreplaceable results;
- run project commands, tests, diagnostics, and project-local dependency setup;
- use available CPUs/GPUs and the configured tmux, SSH, and Slurm targets;
- smoke, launch, monitor, retry, and collect experiments; and
- analyze outputs, generate plots and summaries, interpret results, and update
  the lightweight experiment record.

A request to run or complete an experiment authorizes that ordinary lifecycle
without a separate approval round trip. Before a substantial launch, make the
cases, target, budget, expected duration, and result path visible, but continue
without waiting for confirmation. If target or budget is omitted, choose the
leanest protocol-compliant option that can answer the question and record the
assumption.

Pause only when a missing scientific choice would materially change the
question, required access is unavailable, or an action would affect unrelated
work, data, jobs, or people. Repository scientific gates protect evidential
quality; they are not human approval gates.
Normal contention from the authorized RTX 5090 sharing policy below does not
require another approval; changing or stopping another person's jobs does.

## Repository Rules

- Keep source, small configs, documentation, and lightweight result summaries
  in Git. Keep datasets, checkpoints, logs, and generated result bundles out
  of Git.
- Read a directory's nested `AGENTS.md` before editing files there.
- Require explicit `*_diode_param` dictionaries in configs; do not invent
  diode defaults.
- Use `matplotlib` for generated plots unless the user requests another
  backend.
- Treat `docs/current_state.md` as human-authored: read it for context, but do
  not edit it unless Filip explicitly grants permission for that edit.
- Read or edit `docs/my_notes.md` only when Filip explicitly asks.

## Scientific Authority

The active Conv scope is perfect-diode BPTT training of Conv1, Conv2, and
Conv3. Start with:

- `docs/conv_paper_hyperparameter_protocol.md`
- `docs/conv_paper_experiment_definition.md`
- `docs/perfectdiode_learning_protocol.md`
- `docs/perfectdiode_conv3_learning_protocol.md`
- `docs/perfectdiode_bounded_weight_protocol.md`

`docs/current_state.md` is the concise dashboard and node inventory.
`docs/experiment_workflow.md` is the execution guide. Historical and archived
documents are provenance, not authority for new runs.
New runs follow `docs/experiment_reporting.md`; live state is generated in
`docs/current_simulations.md`, every agent-created top-level result directory
is indexed persistently there, and analyzed conclusions are curated manually in
`docs/experimental_manifest.md`.

Ordinary MNIST is used for both hyperparameter selection and the active paper
runs. Selection uses the deterministic 55,000/5,000 train/validation split and
must not read the official test split. A paper-facing accuracy is the official
10,000-example MNIST test result read exactly once after the complete contract
and checkpoint-selection rule are frozen; ordinary-MNIST validation accuracy
is never paper-facing evidence. Deterministic medium-affine MNIST is retained
as historical or optional robustness evidence, not a required paper dataset.

## Experiments

Follow `docs/experiment_workflow.md`.

- Keep scientific choices in a readable config or command. Do not introduce
  catalogs, receipt graphs, launch schemas, or experiment-specific control
  planes.
- When Filip supplies complete configs with exact learning-rate vectors, use
  `python -m experiments.exact_run`. Do not run a rho search, calibration, or
  broad test campaign first. Use its one-train-batch/one-validation-batch
  `--smoke` mode locally, then launch the unchanged configs.
- Prefer a direct runner. Extract shared code only when it serves more than
  one experiment.
- Before a long or scheduled run, record and report the scientific cases,
  target, budget, expected duration, and result path, then proceed once the
  applicable smoke and scientific gates pass.
- Before creating a top-level result directory, add or update its persistent
  row in `docs/current_simulations.md`. Maintain its study-level state through
  launch, remote collection, validation, review, and supersession; automatic
  per-run status reporting does not replace this handoff row.
- Before submitting any live Jean Zay job, including a production job, array,
  restored remote canary, or retry, verify that its `planned` row already
  exists in `docs/current_simulations.md`. Prefer a direct result-directory
  link. If no directory can be linked yet, record the launch name, config or
  wrapper, and expected remote output root or pattern; immediately after
  `sbatch`, add the Slurm job ID and resolve the result link when possible.
- Check the simulation GPU hosts listed below and the configured `jean-zay`
  target before allocating work. An RTX 5090 being used by Ben remains an
  authorized shared target under the policy below. Never replace, terminate,
  suspend, or reconfigure unrelated jobs or take over their launcher lanes.
- Run a short end-to-end smoke through the same scientific runner and config.
  For Jean Zay, the default operational gate is a synchronous local smoke
  followed immediately by one production submission; do not submit a separate
  remote canary job. Record the local command and require its semantic output
  before production. If a future production failure is attributable to a
  Jean Zay-only environment, staging, device, filesystem, or Slurm issue,
  restore a live Jean Zay canary for that affected execution contract.
  Conv amplification runs also need the active scientific `T/K` gate unless
  an unchanged accepted operating point applies. Exact-config repeats cite the
  accepted operating point instead of rerunning it.
- Use `python -m experiments.launch` for the common local, tmux, SSH/tmux, and
  Slurm cases. A direct command is fine when it is clearer.
- Keep one scientific surface on one recorded target. Host selection is
  transport, not a scheme-dependent scientific factor; initialization,
  cohorts, minibatch order, and resolved config must remain identical where
  the protocol requires matched comparisons.
- Save the resolved config, exact command, commit, environment summary, logs,
  metrics, and checkpoints in one run-specific directory.
- For new active runs, write the canonical `manifest.json`, `status.json`,
  `metrics.jsonl`, and successful-only `result.json` bundle defined by
  `docs/experiment_reporting.md`.
- Diagnose operational failures and retry when the cause is understood and the
  science, target class, and budget remain in scope. Agents may cancel and
  replace jobs they launched when those jobs are invalid, obsolete, or
  operationally broken; do not disturb unrelated jobs or delete irreplaceable
  data.
- While actively monitoring, keep progress observable and use a deadline.
- Copy remote results locally and validate the local copy before drawing final
  conclusions.
- Do not mark a scientific study `ready-for-review` until expected coverage is
  reconciled, the authoritative local copy exists, included run bundles
  validate, and failures, exclusions, and replacement directories are named.
- Interpret completed or clearly labeled partial results directly against the
  governing protocol. Distinguish measurements from inference and record
  uncertainty or protocol deviations instead of withholding a supported
  conclusion for another approval step.

Prefer placement by model size and GPU capacity:

- Conv1: prefer the smaller GPUs—Akib's RTX 3080 or the RTX 3090s on local
  (`local`/`main`) and `nom-cool-1`.
- Conv2: placement is flexible; choose an available GPU according to the
  run's memory needs and expected duration.
- Conv3: prefer the RTX 5090 machines (`trex`, `fifi`, `loulou`, or another
  host once verified to have an RTX 5090) or `jean-zay`.

These are placement preferences; preserve the scientific config and matched
comparison requirements when choosing a host.
Jean Zay defaults to `fmu@v100`; put outputs under
`/lustre/fsn1/projects/rech/fmu/$USER/server_code/results`.

## Sharing RTX 5090 GPUs With Ben

RTX 5090 GPUs may always be considered for our simulations while Ben is using
them, during both daytime and nighttime. His presence or high GPU utilization
alone is not a reason to reject the host or wait for it to become idle. This
is standing permission to share; do not ask again for each launch.

- Verify the GPU model, free memory, process owners and existing launchers.
  Identify Ben's workload from process ownership or known launcher metadata;
  do not assume that every occupied GPU belongs to him.
- Admit our job when its measured peak memory plus headroom fits alongside
  the existing workload. Preserve its scientific batch size, precision and
  config. If it does not fit, choose another target or wait for memory.
- Use our own launcher/session and result path. Leave Ben's processes, queues,
  priorities, MPS service and GPU-wide settings intact. Other users' occupied
  lanes retain their existing protection unless separately authorized.
- Measure our throughput under actual sharing and use that rate for duration
  estimates and deadlines. Record shared use in the run's environment notes;
  do not report shared-GPU timings as exclusive-GPU benchmarks.

## Overnight Simulations

Proactively use **22:00–08:00 Europe/Paris** for authorized long simulations.
Prepare configs, inputs, smoke checks and the run queue during the day, then
arm a detached, dated launcher for the next night window. Do not require Filip
to return at 22:00 or approve each night of an already assigned experiment.
This schedules work within its existing scientific scope, budget and deadline;
it does not authorize an open-ended search or invented follow-up experiments.

Default to completing or safely checkpointing and pausing overnight workers
by 08:00, unless the recorded run plan permits daytime continuation. A pause
must preserve the full state needed for scientifically equivalent resumption.
If the runner cannot do that, admit only runs expected to finish within the
window with a margin. Do not truncate epochs or change scientific configs to
fit the night. Pending work may use a later night within the original budget.

Nighttime is an opportunity, not evidence that a GPU is idle. Recheck live
memory, users and queues at actual launch; the Ben/RTX 5090 sharing permission
still applies. Keep jobs observable and monitored, and collect a morning
summary of completed, paused, failed and unstarted cases. Follow the concrete
[overnight queue workflow](docs/experiment_workflow.md#overnight-runs).

## Simulation GPU Inventory

The following machines are authorized simulation resources. This is a hardware
and access inventory, not a statement that their GPUs are idle. Check live GPU
memory, utilization, and running jobs immediately before allocating work.

| Machine | Access / launcher target | GPU inventory | Verification |
|---|---|---|---|
| Local (`nom-cool-2`) | `local` foreground or `main` tmux | 1 × NVIDIA GeForce RTX 3090, 24 GiB | Queried 2026-09-11 |
| `akibscomputer` (Akib) | SSH `akibscomputer` → `filiposana@172.24.6.229` (hostname `integnano-akib`); launcher `akib` still uses SSH `akib` | 1 × NVIDIA GeForce RTX 3080, 10 GiB | Queried via `akibscomputer` on 2026-09-11 |
| `nom-cool-1` | SSH `filip@nom-cool-1` | 1 × NVIDIA GeForce RTX 3090, 24 GiB | Queried 2026-09-11 |
| `trex` | SSH `filip@trex`; launcher `trex` | 1 × NVIDIA GeForce RTX 5090, 32 GiB | Queried 2026-09-11 |
| `riri` | SSH `filip@riri` | Model, count, and memory unverified | SSH authentication failed on 2026-09-11 |
| `fifi` | SSH `filip@fifi` | 1 × NVIDIA GeForce RTX 5090, 32 GiB | Queried 2026-09-11 |
| `loulou` | SSH `filip@loulou` | 1 × NVIDIA GeForce RTX 5090, 32 GiB | Queried 2026-09-11 |

Memory above is nominal capacity; `nvidia-smi` reported 10,240 MiB for the
RTX 3080, 24,576 MiB for the RTX 3090s, and 32,607 MiB for the RTX 5090s.
Re-query unverified hardware before choosing a workload for it.

`local` and `main` share one physical GPU host. `akibscomputer` is both the
historical tmux name and a working SSH alias for Akib, not an additional
machine. Prefer SSH `akibscomputer`: SSH `akib` was unreachable during the
earlier check on 2026-09-11. The configured launcher target `akib` still uses
SSH `akib`; verify that route or update its host before using the launcher.
`nom-cool-1`, `riri`, `fifi`, and `loulou` are not currently named targets in
`configs/experiment_targets.json`; use a direct SSH runner or configure the
target before using `experiments.launch`, and verify the remote checkout and
Python environment. Jean Zay remains available through the existing Slurm
target and its allocation-specific GPU resources.

## Environment

If OpenMP shared memory is blocked, set:

```text
KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```
