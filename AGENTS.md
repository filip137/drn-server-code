# AGENTS

## Purpose
This repository contains code and tooling for coordinate-descent simulations of DRNs with dissipative non-linearities.

## Scope and Boundaries
- In scope: Python source, configs, scripts, and documentation.
- Out of scope: large generated outputs and datasets (see `.gitignore`).

## Repo Layout (key)
- `model/`: model components and parameterized model adapters
- `training/`: reusable engines, estimators, update backends, modifiers, probes,
  and checkpoint codecs
- `experiments/`: versioned experiment schemas and thin composition roots
- `campaigns/`: subprocess-only orchestration across clean Git worktrees
- `labs/`: specialized numerical and analysis utilities; experiment execution
  belongs under `experiments/` and orchestration under `campaigns/`
- `plotting_functions/`: analysis/plotting helpers
- `playbooks/`: SOPs for repeatable tasks
- `docs/`: lightweight state and notes
- `results/`: ignored raw LoRA/HWA study output; only its root guide is
  intended for Git

## Experiment Architecture

- The stable public CLI is `python -m ebl`.
- Config files select a registered versioned experiment ID. They contain
  scientific settings only; input checkpoints and output paths are CLI
  arguments.
- Keep config parsing pure. It must not import datasets, initialize a GPU,
  create directories, or discover modules dynamically.
- Keep `model/` and `training/` independent of experiment-specific config
  dataclasses. Adapt config into numerical objects in
  `experiments/<experiment>/`.
- Add an intervention at the narrowest extension boundary:
  - structural parameterization: model adapter;
  - free/nudged phase perturbation: parameter modifier;
  - gradient application/accumulation: update backend;
  - measurements: evaluation probe.
- New combinations fail closed until listed in the experiment definition and
  covered by a numerical parity or acceptance test.
- Do not use module globals or monkey patches to compose experiments.
- New exploratory studies start from a tracked strict plan under `studies/`.
  The plan states the initial hypothesis, exact arm configs, completion
  criteria, and analysis plan before native runs begin.
- Materialize plans with `python -m ebl study prepare`; write native runs only
  below their declared `results/<study-id>/runs/<arm-id>/` roots.
- After exact coverage is present, run `python -m ebl study summarize`, write
  a human review, and use `python -m ebl study finalize` so
  `experimental_manifest.md` retains both the initial hypothesis and final
  interpretation.
- Keep `current_state.md` as the concise human-readable big picture and next
  steps. Run status belongs in `current_simulations.md`; detailed final
  evidence belongs in `experimental_manifest.md`.

## Model-Local Indexing and Construction Parity

- Never let process-global object counters, generated names such as
  `Layer_3`, or model-construction order affect numerical equations. They may
  be used as diagnostic identifiers only. Amplifier stages, energy scales,
  nonlinearities, parameter roles, and connectivity must use explicit
  model-local topology indices.
- Treat repeated construction as a required lifecycle check. Learning-rate
  selection, canaries, production training, resume, validation, and test may
  build multiple models in one process or use fresh subprocesses. Deleting a
  model or reseeding random-number generators does not reset Python class
  counters.
- For models with depth-dependent amplification or scaling, verify that a
  model built second in the same process has the same resolved numerical
  semantics as a model built first. Also verify train/resume/validate parity
  from recorded metadata or numerical acceptance tests.
- Record resolved topology indices or equivalent stage scales in checkpoints
  and result artifacts whenever they can affect the equations. Fail closed on
  a mismatch between the requested experiment and checkpoint provenance.
- A deliberate replay of process-global indexing is allowed only as an
  explicitly named historical-control adapter with a required schema marker,
  capability-matrix entry, provenance metadata, and dedicated lifecycle test.
  Never present such a replay as the intended physical circuit.

## Artifacts and Checkpoints

- A command owns exactly one run directory. Never append to a prior run.
- `checkpoints/weights.pt` is the selected named-weights artifact.
- `checkpoints/resume.pt` is the latest full epoch-boundary state and is not a
  substitute for selected weights.
- Load parameters by stable catalog key, not positional order.
- Positional checkpoints are accepted only by
  `ebl checkpoint import-legacy` with an explicit `--kind`.
- Linspace and validation require an explicit `--weights` path. Never scan for
  the newest model.
- Campaigns invoke worktrees as subprocesses through the public CLI. Do not
  import Python modules from another worktree into the controller process.

## Change Checklist

1. Update the strict schema and explicit experiment capability matrix.
2. Implement against an existing extension protocol or add a focused one.
3. Add unit tests for lifecycle/order and a numerical parity test where
   behavior should remain unchanged. If a model uses layer-dependent scaling,
   build it at least twice in one process and verify identical model-local
   semantics; also check production versus fresh-process validation metadata.
4. Audit numerical code for dependence on generated names, class counters, or
   construction order, and make the relevant indices explicit and model-local.
5. Add or update a nested example config.
6. Run `python -m ebl describe --experiment small_drn.v1 --json`, the focused
   tests, and the legacy `labs/tests` suite.
7. Keep HWA and LoRA feature commits separate so each can be rebased onto the
   same foundation and compared by a campaign.

## Playbooks
- Optuna analysis SOP: `playbooks/optuna_analysis.md`
- Task overrides: `playbooks/AGENTS.override.md`

## Timing Plot Script
- Script: `labs/tools/plot_spice_vs_coordinate_descent_loglog.py`
- Purpose: log-log plot with:
- x-axis: hidden size
- y-axis: time (seconds)
- series: coordinate descent and SPICE for all hidden-layer counts in one figure
- Input: `simulation_results/.../extracted_timings/combined_latest_by_hidden.csv` (or any CSV with equivalent columns)
- Basic usage:
- `python /home/filip/server_code/labs/tools/plot_spice_vs_coordinate_descent_loglog.py --combined-csv /home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/extracted_timings/combined_latest_by_hidden.csv --output /home/filip/server_code/labs/figures_for_paper_digits/timings/double_diode_exponential/extracted_timings/spice_vs_coordinate_descent_loglog.png`
- Optional SPICE field selection:
- `--spice-time-field total` (default), `--spice-time-field simulation`, or `--spice-time-field netlist`

## Plotting Rule
- Use `matplotlib` for every generated plot (PNG/SVG/PDF) unless the user explicitly requests a different plotting backend.
- Prefer reusable plotting scripts under `labs/tools/` for plot generation commands.

## Data and Artifacts
New raw LoRA/HWA outputs live under `results/<study-id>/` and are ignored by
Git. Existing generated outputs remain under legacy folders such as
`simulation_results*`, `papers/`, and `labs/cases/`; do not move them solely
to adopt the new index.

The `Active` block in `docs/current_simulations.md` is generated from running
native statuses below `results/`. Do not edit inside its automatic markers.
The refresh is informational and must never block or fail a numerical run.
Paused, analyzing, and queued sections remain human-maintained.

## Error Messages
- When validating inputs, state the expected format first, then echo the provided value on failure.

## Config Rules
- Do not silently default diode parameter dicts; require explicit `*_diode_param` dicts in config.
- Reject unknown keys at every versioned experiment-config level.
- Error messages state the expected format before the provided value.

## Environment Issues
- OpenMP SHM error (`OMP: Error #179: Function Can't open SHM2 failed: System error #13: Permission denied`) can occur when running Python (e.g., matplotlib/numpy) in the sandbox. Fix options:
- Set env vars to disable shared memory: `KMP_DISABLE_SHM=1`, `KMP_SHM_DISABLE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`.
- Run the command outside the sandbox (escalated) if SHM is blocked.
- If you just need PNGs from existing SVGs, use `rsvg-convert` as a fallback.
