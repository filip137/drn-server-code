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
- `skills/`: repo-local Codex workflows and deterministic helpers
- `docs/`: lightweight state and notes

## Nested Guidance
- Before changing files under a directory that contains its own `AGENTS.md`,
  read and follow that file in addition to this repository-wide guidance.

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
- For an already approved long simulation on `tmux main` or Trex, use the
  bounded, receipt-backed `python -m experiments.local_dispatch` workflow in
  `docs/local_dispatch.md`; do not reconstruct it with `send-keys` or a chain
  of one-check-per-SSH commands. Build its schema-validated v2 preflight
  envelope with the same tool before arming `start`.

## Long-Run Simulation Fail-Stop-Report Policy
- This policy applies only to an **armed long-run simulation attempt**: a
  numerical run or batch expected to occupy a compute lane for at least ten
  minutes, any scheduled/queued long-running simulation batch, or explicitly
  armed monitoring of such an active run. It does not apply merely because
  the repository contains experiment code.
- Before entering the final launch path, state `LONG-RUN ATTEMPT ARMED` with
  the experiment/attempt ID, target lane or scheduler, expected duration,
  hard deadline, and immutable state/receipt paths. The armed attempt covers
  final staging and revalidation of already-produced smoke/scientific gate
  receipts, submission, immediate scheduler or process readback, and
  active-run monitoring.
- Planning, code edits, unit/integration tests, linting, dependency and
  environment setup, plan-only/dry-run commands, short interactive probes,
  bounded developer smoke tests, and short smoke/scientific gate execution
  completed before `start` are
  outside the armed attempt. Failures there may be diagnosed, fixed, and
  retested with bounded observable work; they must never be represented as
  passed long-run launch gates. A gate that is itself a long or scheduled
  simulation is a separate armed long-run attempt.
- Offline collection and analysis after a run is no longer active are also
  outside the armed attempt. For any repairable failure outside the protected
  scope, diagnose the cause before rerunning and keep corrective work bounded
  and observable; never repeat blindly. If the cause remains unresolved after
  reasonable diagnosis-backed attempts, report it.
- Within an armed attempt, an unexpected failure is any nonzero exit,
  exception, timeout, unreachable required dependency, or missing, stale,
  ambiguous, or mismatched receipt, artifact, or scheduler readback that the
  approved plan does not explicitly classify as an independent scientific
  outcome. The first such failure is terminal for that long-run attempt and
  current launch turn.
- At that failure, stop the long-run launch/monitoring workflow. Perform only
  bounded, read-only inspection needed to identify the exact error and live
  job state. Do not edit code or configuration, substitute a path, tool,
  host, or environment, restage, refreeze, rerun, retry, resubmit, cancel, or
  delegate those actions in the same turn.
- Report and wait. Only a new user message sent after the long-run failure
  report may authorize diagnosis, repair, cancellation, or a new attempt. A
  plan, config, or launcher may define retry mechanics, but it never grants
  authority to initiate that retry.
- The report must name the exact stage, command, and error; completed stages;
  launched-job count, IDs, and live states (explicitly zero when none);
  last valid receipts, logs, and artifacts; files or external state changed;
  and the smallest proposed next action.
- Preserve any failed receipt and all diagnostic evidence. A later authorized
  attempt must use a new immutable attempt ID and receipt paths; never
  overwrite or resume a terminal failed attempt.
- A long-run controller may catch an unexpected exception only to atomically
  write one first-write-wins failure report, mark the attempt terminal, and
  exit nonzero. It must never catch and continue. A failed long-run state
  cannot submit or resume.
- Long-run polling and reconciliation loops require an explicit hard
  deadline, emit observable progress at least every 60 seconds, and stop and
  report if progress becomes unobservable or the deadline expires.
- A candidate-cell rejection that an approved sweep explicitly defines as an
  independent scientific outcome is not a pipeline failure. Record it as
  negative evidence and continue only as authorized by the frozen plan.

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
