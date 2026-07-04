# AGENTS

## Purpose
This repository contains code and tooling for coordinate-descent simulations of DRNs with dissipative non-linearities.

## Scope and Boundaries
- In scope: Python source, configs, scripts, and documentation.
- Out of scope: large generated outputs and datasets (see `.gitignore`).

## Repo Layout (key)
- `model/`: core model components
- `training/`: training loops and monitoring
- `labs/`: experiments and utilities
- `plotting_functions/`: analysis/plotting helpers
- `playbooks/`: SOPs for repeatable tasks
- `docs/`: lightweight state and notes

## Playbooks
- Optuna analysis SOP: `playbooks/optuna_analysis.md`
- Conv amplification paper protocol: `docs/conv_paper_hyperparameter_protocol.md`
- Amplification result curation: `docs/amplification_experiment_curation.md`
- Current conv paper state: `docs/current_state.md`
- Task overrides: `playbooks/AGENTS.override.md`

## Experiment Launch Policy
- For large experiment batches, first check whether local GPU tmux targets are free:
  - `tmux main`
  - `tmux akibscomputer`
  - `tmux trex`
- If any of those targets are free and reachable, parallelize there as much as GPU memory allows.
- Use Jean Zay for the remaining jobs or for batches that exceed the local machines.
- Jean Zay runs default to the R3 project `fmu`; use `fmu@v100` for V100 GPU jobs and write run outputs under `/lustre/fsn1/projects/rech/fmu/$USER/server_code/results` unless the user explicitly requests another project/account. The source checkout may stay under the existing `umg` work path.
- For MNIST Conv amplification training, prefer running Conv1 and Conv2 jobs on `tmux main` and `tmux akibscomputer`; reserve Conv3 jobs for Jean Zay and `tmux trex` unless local availability or urgency clearly argues otherwise.
- Prefer keeping the same launcher/config contract across local tmux and Jean Zay so results can be aggregated into one summary.

## Conv Amplification Paper Work
- Before proposing, launching, or summarizing MNIST Conv amplification runs, read `docs/conv_paper_hyperparameter_protocol.md` and use it as the source of truth for choosing solver iteration count `K`, hard-sigmoid operating point, `input_gain`, learning rate, epoch budget, and final inclusion category.
- Use `docs/amplification_experiment_curation.md` to decide which historical runs are valid, diagnostic, excluded, or superseded. Do not use superseded rows as paper-facing quantitative evidence.
- Keep final comparison axes fixed inside each table: architecture, preprocessing, nonlinearity family, amplification grid, seed list, batch size, epoch budget, and checkpoint rule. Label mixed-protocol rows as diagnostics.
- For hard-sigmoid MNIST Conv amplification runs with a target saturation, calibrate raw `input_gain` separately for each amplification scheme using the same deterministic saturation-target procedure. Do not use a shared raw `input_gain` across amplifications for the main comparison; shared-gain runs are diagnostics only. LR sweeps and longer runs must preserve the per-amplification calibrated `input_gain` for the chosen target saturation.
- For main hard-sigmoid comparisons, keep `v_off` and target initial saturation fixed across amplification schemes within an architecture/nonlinearity table. The current paper-facing default is `v_off=4.0` with `50%` target initial saturation unless a newer dated protocol update supersedes it.
- For perfect-diode Conv runs, judge clamped hidden-layer convergence with projected KKT residuals. Raw residuals are still useful for unconstrained layers and diagnostics, but do not treat raw `|dE/dz|` on clamped variables as the convergence criterion.
- Choose `K` before operating point, `input_gain`, or LR. The residual-vs-K gate should include the intended `K` plus larger sentinel values such as `2K` and `4K`, and final training should wait until residual stagnation and the EP/BP cosine rule pass.
- For LR selection, follow the dated paper protocol. Seed-0 screens and seed-0 long checks may select LR candidates, but final paper claims require validating the frozen settings on multiple seeds when runtime is reasonable; do not present single-seed screens as final evidence.
- For final paper runs, use seeds `0, 1, 2` at minimum when runtime is reasonable, report both best-checkpoint and final-epoch metrics, and keep the same epoch budget for every amplification setting in a table.
- When updating `docs/current_state.md`, separate calibrated initial saturation from trained-checkpoint saturation. The overall best-known table must mark mixed raw-gain, saturation, `K`, LR, or epoch comparisons as `mixed protocol / diagnostic`.

## Plotting Rule
- Use `matplotlib` for every generated plot (PNG/SVG/PDF) unless the user explicitly requests a different plotting backend.
- Prefer reusable plotting scripts under `labs/tools/` for plot generation commands.

## Data and Artifacts
Generated outputs live under folders like `simulation_results*`, `papers/`, and `labs/cases/` and are ignored by git.

## Error Messages
- When validating inputs, state the expected format first, then echo the provided value on failure.

## Config Rules
- Do not silently default diode parameter dicts; require explicit `*_diode_param` dicts in config.

## Environment Issues
- OpenMP SHM error (`OMP: Error #179: Function Can't open SHM2 failed: System error #13: Permission denied`) can occur when running Python (e.g., matplotlib/numpy) in the sandbox. Fix options:
- Set env vars to disable shared memory: `KMP_DISABLE_SHM=1`, `KMP_SHM_DISABLE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`.
- Run the command outside the sandbox (escalated) if SHM is blocked.
- If you just need PNGs from existing SVGs, use `rsvg-convert` as a fallback.
