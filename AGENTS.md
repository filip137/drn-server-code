# AGENTS

## Purpose

This repository contains the simulations and evidence for a paper on
bidirectional amplification in dissipative resistive networks.

## Research Planning

Use [campaigns/README.md](campaigns/README.md) and the repository's
[experiment-loop skill](.agents/skills/experiment-loop/SKILL.md) for research
questions and the full Explorer → Codifier → Monitor → Reviewer lifecycle.
New planning for an adopted question belongs in its campaign; preserve historical
plans and result paths. Small exploratory pilots use one note in `campaigns/pilots/`
without full campaign scaffolding. Pass record paths and compact
evidence summaries between roles; reuse unchanged context rather than rereading it.
run-experiment remains a compatibility entry point into the same lifecycle.
For explicit EqProp and CIFAR tasks, use their declared task-specific scientific
contracts rather than substituting the general Conv BPTT scope below.

Campaign setup or planning alone does not launch simulations. Explicit execution
requests retain the standing authority below. Experiment/pilot notes own execution
handoffs and result paths; result/pilot notes own reviewed conclusions. The campaign
ledger is generated scientific navigation. `docs/current_simulations.md` and
`docs/experimental_manifest.md` are historical snapshots: do not refresh or append
to them. This supersedes older plans and external skills requiring those updates.

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

- The active paper manuscript is now `paper_bidir.tex` in the Overleaf project.
  Apply manuscript updates to this file; it supersedes
  `bidir_paper_theory_revised.tex`. Sync the Overleaf checkout before editing if
  the renamed file is not yet present locally.
- Keep source, small configs, documentation, and lightweight result summaries
  in Git. Keep datasets, checkpoints, logs, and generated result bundles out
  of Git.
- Read a directory's nested `AGENTS.md` before editing files there.
- Require explicit `*_diode_param` dictionaries in configs; do not invent
  diode defaults.
- Use `matplotlib` for generated plots unless the user requests another
  backend.
- Export plots as `.jpg` by default and link the JPG files in user-facing
  responses. Additional formats such as PDF may be provided when useful.
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

`docs/current_state.md` is the concise human-authored dashboard.
`docs/experiment_workflow.md` is the execution guide. Historical and archived
documents are provenance, not authority for new runs.
New runs follow `docs/experiment_reporting.md`. Runners write live state into their
own bundles; the owning experiment/pilot note links every top-level result directory,
including failed or superseded attempts. Keep conclusions in the result/pilot note
and refresh an established campaign's ledger when its inputs change.

Ordinary MNIST is used for both hyperparameter selection and the active paper
runs. Selection uses the deterministic 55,000/5,000 train/validation split and
must not read the official test split. A paper-facing accuracy is the official
10,000-example MNIST test result read exactly once after the complete contract
and checkpoint-selection rule are frozen; ordinary-MNIST validation accuracy
is never paper-facing evidence. Deterministic medium-affine MNIST is retained
as historical or optional robustness evidence, not a required paper dataset.

## Experiments

Routine launches do not require T/K gradient/residual qualification or replay,
optimizer/rho probes, or CIFAR solver/migration audits. Use declared iteration
counts and learning rates; run those diagnostics only when explicitly assigned.
The September 25 retirement in docs/experiment_workflow.md supersedes older
pre-launch requirements. Keep execution smoke, identity/restore checks and
runtime nonfinite guards.

Use [experiment-loop](.agents/skills/experiment-loop/SKILL.md) for role ownership
and handoffs. [docs/experiment_workflow.md](docs/experiment_workflow.md) is the
operational authority for exact-config execution, smoke/scientific gates, GPU
placement, launch, timing, monitoring, recovery and collection. The
[reporting contract](docs/experiment_reporting.md) defines evidence and validation.
Keep scientific choices in readable configs/commands; do not add catalogs,
receipt graphs, launch schemas or experiment-specific control planes.

## Sharing GPUs With Ben

For the exp014 relative-noise training campaign, Filip also explicitly permits
sharing with Kellian (September 25). Admit when measured memory plus headroom
fits, preserve both workloads, and keep the existing study budget and deadline.
This campaign permission overrides an exclusive-GPU reservation for its Conv3
runs; it does not authorize sharing with other unidentified or unapproved owners.

Standing sharing permission and its admission rules are maintained in the
[execution policy](docs/experiment_workflow.md#sharing-gpus-with-ben), covering
all authorized GPU models, including RTX 3080, 3090 and 5090.

Weekdays 09:00–18:00 Europe/Paris: admit new work only within two distinct RTX
5090s across all our jobs. Existing jobs may finish after 09:00; wait for usage
below two before occupying another 5090. A Conv2 pair counts as one GPU.
Nights/weekends have no fleet-count cap.
Prefer memory-heavy jobs on 5090s. Use the queue's explicit Conv2/Conv3 profiles:
two Conv2 workers per 5090 when memory fits; one Conv3 reserving measured peak
memory plus headroom. Filip removed the fixed 31 GiB reservation on September29.
See execution policy for admission details.
Prepared jobs can use the shared run-watch queue; record its job ID in the
existing handoff and keep one monitoring owner.

## Overnight Simulations

Use the [overnight policy](docs/experiment_workflow.md#overnight-runs) for
authorized long runs, including the 22:00–08:00 Europe/Paris window.

## Simulation GPU Inventory

See the [GPU inventory](docs/experiment_workflow.md#simulation-gpu-inventory)
and [placement preferences](docs/experiment_workflow.md#gpu-placement).
Inventory is not live availability; verify resources before allocation.

## Environment

If OpenMP shared memory is blocked, set:

```text
KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```
