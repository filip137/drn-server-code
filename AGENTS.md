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

## Repository Rules

- Keep source, small configs, documentation, and lightweight result summaries
  in Git. Keep datasets, checkpoints, logs, and generated result bundles out
  of Git.
- Read a directory's nested `AGENTS.md` before editing files there.
- Require explicit `*_diode_param` dictionaries in configs; do not invent
  diode defaults.
- Use `matplotlib` for generated plots unless the user requests another
  backend.
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
`docs/current_simulations.md`, and analyzed conclusions are curated manually
in `docs/experimental_manifest.md`.

Ordinary MNIST is used for `T/K`, rho, learning-rate, and bounded-initializer
selection. Deterministic medium-affine MNIST is used for paper runs. A
diagnostic accuracy from ordinary MNIST is never paper-facing evidence.

## Experiments

Follow `docs/experiment_workflow.md`.

- Keep scientific choices in a readable config or command. Do not introduce
  catalogs, receipt graphs, launch schemas, or experiment-specific control
  planes.
- When Filip supplies complete configs with exact learning-rate vectors, use
  `python -m experiments.exact_run`. Do not run a rho search, calibration, or
  broad test campaign first. Use its one-train-batch/one-validation-batch
  `--smoke` mode on the target, then launch the unchanged configs.
- Prefer a direct runner. Extract shared code only when it serves more than
  one experiment.
- Before a long or scheduled run, record and report the scientific cases,
  target, budget, expected duration, and result path, then proceed once the
  applicable smoke and scientific gates pass.
- Check the configured `main`, `akib`, `trex`, and `jean-zay` targets before
  allocating work. Never replace or interfere with an occupied lane or
  unrelated Slurm job.
- Run a short end-to-end smoke through the same runner, environment, device,
  and output path. Conv amplification runs also need the active scientific
  `T/K` gate unless an unchanged accepted operating point applies. Exact-config
  repeats cite the accepted operating point instead of rerunning it.
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
- Interpret completed or clearly labeled partial results directly against the
  governing protocol. Distinguish measurements from inference and record
  uncertainty or protocol deviations instead of withholding a supported
  conclusion for another approval step.

Prefer Conv1 on `main` or `akib`, Conv2 on `akib` or `trex`, and Conv3 on
`trex` or `jean-zay`. Jean Zay defaults to `fmu@v100`; put outputs under
`/lustre/fsn1/projects/rech/fmu/$USER/server_code/results`.

## Environment

If OpenMP shared memory is blocked, set:

```text
KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```
