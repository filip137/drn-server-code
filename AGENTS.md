# AGENTS

## Purpose

This repository contains the simulations and evidence for a paper on
bidirectional amplification in dissipative resistive networks.

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

For Conv amplification work, start with:

- `docs/conv_paper_hyperparameter_protocol.md`
- `docs/conv_paper_experiment_definition.md`
- `docs/conv_paper_hard_sigmoid_input_gain_protocol.md`
- `docs/conv_paper_tk_protocol.md`
- `docs/conv_paper_learning_rate_protocol.md`
- `docs/amplification_experiment_curation.md`

`docs/current_state.md` records direction and `docs/current_experiments.md`
records live operational state. Neither replaces the scientific protocols.
Historical and archived documents are provenance, not authority for new runs.

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
- Before a long or scheduled run, show the scientific cases, target, budget,
  expected duration, and result path and get explicit approval.
- Check `tmux main`, `tmux akibscomputer`, and `tmux trex` before allocating
  work. Never replace or interfere with an occupied lane.
- Run a short end-to-end smoke through the same runner, environment, device,
  and output path. Conv amplification runs also need the active scientific
  `T/K` gate unless an unchanged accepted operating point applies. Exact-config
  repeats cite the accepted operating point instead of rerunning it.
- Use `python -m experiments.launch` for the common local, tmux, SSH/tmux, and
  Slurm cases. A direct command is fine when it is clearer.
- Save the resolved config, exact command, commit, environment summary, logs,
  metrics, and checkpoints in one run-specific directory.
- Diagnose operational failures and retry when the cause is understood and
  the approved science, target class, and budget are unchanged. Ask again
  before changing scientific scope, materially expanding cost, cancelling a
  run, or deleting data.
- While actively monitoring, keep progress observable and use a deadline.
- Copy remote results locally and validate the local copy before drawing final
  conclusions.

For MNIST Conv work, prefer Conv1/Conv2 on `main` and `akibscomputer`, and
Conv3 on Jean Zay or `trex`. Jean Zay defaults to `fmu@v100`; put outputs
under `/lustre/fsn1/projects/rech/fmu/$USER/server_code/results`.

## Environment

If OpenMP shared memory is blocked, set:

```text
KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```
