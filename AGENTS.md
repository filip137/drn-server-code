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
- Conv amplification paper protocol index: `docs/conv_paper_hyperparameter_protocol.md`
- Frozen Conv experiment definition: `docs/conv_paper_experiment_definition.md`
- Hard-sigmoid input-gain protocol: `docs/conv_paper_hard_sigmoid_input_gain_protocol.md`
- Operational T/K protocol: `docs/conv_paper_tk_protocol.md`
- Hard-sigmoid learning-rate handoff: `docs/conv_paper_learning_rate_protocol.md`
- Learning-rate diagnostic ledger: `docs/conv_learning_rate_diagnostics.md`
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
- Before proposing, launching, or summarizing Conv amplification runs, read `docs/conv_paper_hyperparameter_protocol.md` and every active protocol it links. Use that protocol set as the source of truth.
- Use `docs/amplification_experiment_curation.md` to decide which historical runs are valid, diagnostic, excluded, or superseded. Do not use superseded rows as paper-facing quantitative evidence.
- The active experiment uses deterministic medium affine MNIST, the frozen Conv1/Conv2/Conv3 padding-1 architectures, paired 20-output loss, and only baseline `v1/c1`, proposed/ours `v4/c1`, and legacy `v4/c0.25`. Do not substitute ordinary or pixel-permuted MNIST.
- Choose and freeze `input_gain` before defining operational `T/K`. Calibrate raw gain separately for every architecture, nonlinearity, and amplification scheme using model seed `0`, provisional calibration `T=64`, and a `30%` first-hidden target; preserve each gain across later model seeds.
- Reset global layer and parameter name counters before every independently built calibration model, and use a dedicated train-loader shuffle seed `0` so the 256-sample cohort is independent of model RNG consumption.
- For hard sigmoid, use `v_off=4.0` and count states outside `[-4,4]`.
- For perfect-diode Conv runs, judge clamped hidden-layer convergence with projected KKT residuals. Raw residuals are still useful for unconstrained layers and diagnostics, but do not treat raw `|dE/dz|` on clamped variables as the convergence criterion.
- The nine hard-sigmoid gains and row-specific operational `T/K` values are frozen in the active protocols. The Conv1/Conv2 hard-sigmoid seed-0 LR handoff is complete: use the two v1 baseline values and four v3 parameter-relative-rho amplified values exactly as curated. Perfect-diode calibration and `T/K`, Conv3/perfect-diode LR rules, and final paper training remain pending.
- Batch size 16 and plain SGD are frozen only for the completed Conv1/Conv2 hard-sigmoid screen. The final-training epoch budget, seed list, checkpoint rule, and inclusion categories remain unresolved; do not launch long checks or final training until a later active protocol defines them. Historical defaults are not active decisions.
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
