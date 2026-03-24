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
