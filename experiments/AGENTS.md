# AGENTS

## Scope
This guidance applies to experiment scripts, launchers, collectors, and monitors under `experiments/`, especially MNIST Conv DRN amplification paper work.

## Conv Amplification Run Rules
- Before changing launchers or interpreting results for Conv amplification, read `../docs/conv_paper_hyperparameter_protocol.md`, every active protocol it links, and `../docs/amplification_experiment_curation.md`.
- For current state, also check `../docs/current_state.md`. Do not treat documents under `../docs/contaminated_old_worktree/` or `../docs/old_worktree_snapshots/` as active protocols.
- Keep the frozen dataset transform, input representation, architecture, output encoding, nonlinearity, and three-scheme amplification grid—baseline `v1/c1`, proposed/ours `v4/c1`, and legacy `v4/c0.25`—explicit in every launcher or collector.
- Do not mix matched-operating-point hard-sigmoid rows, tuned upper-envelope rows, same-raw-`input_gain` diagnostics, and preliminary screens without labeling the category in outputs and summaries.
- Choose `input_gain` before operational `T/K`. For both nonlinearities, preserve the per-architecture and per-amplification 30%-target calibration contract and freeze the seed-0 gain across later model seeds.
- Reset global layer and parameter name counters before every independent calibration model, and seed the shuffled calibration loader independently of model RNG consumption.
- For perfect diode, use projected KKT residuals for clamped hidden-layer convergence checks and keep raw residuals as diagnostics.
- All nine hard-sigmoid gains and row-specific operational `T/K` values are frozen. The Conv1/Conv2 hard-sigmoid seed-0 LR handoff is complete through the two v1 baselines and four v3 parameter-relative-rho amplified rows. Perfect-diode calibration and `T/K`, plus Conv3/perfect-diode LR rules, remain pending.
- Batch size 16 and plain SGD are frozen only for the completed Conv1/Conv2 hard-sigmoid screen. Final-training epoch budget, seeds, checkpoint inclusion, and paper categories remain unresolved; do not launch long checks or final training until the later protocol is frozen.
- Calibration outputs must preserve enough provenance for later aggregation: resolved dataset transform, affine and model seeds, convolution pipeline, nonlinearity, amplification values, calibration T, adaptive-equilibrium setting, selected gain, target occupancy, and every hidden-layer measured occupancy.

## Launching
- Preserve the same launcher/config contract across local tmux lanes and Jean Zay whenever possible.
- For large batches, follow the root `AGENTS.md` launch policy for checking `tmux main`, `tmux akibscomputer`, `tmux trex`, and then using Jean Zay R3 `fmu@v100` for the remainder.
- Keep Jean Zay result outputs under `/lustre/fsn1/projects/rech/fmu/$USER/server_code/results` unless the user explicitly requests another project/account.
- Dispatch already approved long simulations on `tmux main` or Trex through
  `experiments.local_dispatch` and its checked-in profiles. Preserve its
  schema-validated preflight, one-SSH, fresh-window/session,
  immutable-attempt, and no-fallback contract.

## Fail-Stop Implementation
- Apply the root Long-Run Simulation Fail-Stop-Report policy only after a
  long-run attempt is explicitly armed. Ordinary implementation work, tests,
  environment setup, plan-only commands, and bounded developer probes may be
  diagnosed and corrected in the same turn.
- A controller supervising an armed long-run attempt may catch a first
  unexpected failure only to persist one immutable failure report and a
  terminal failed state, then exit nonzero. Catch-and-continue and automatic
  corrective retries are prohibited inside that armed attempt.
- Long-run polling and reconciliation must propagate the first exception and
  use a hard deadline. Default CLI behavior must be one bounded reconciliation
  pass; persistent following must be explicit and bounded.
- Before any scheduler side effect for an armed long-run attempt, reject an
  existing terminal failure report or failed state. A failed state cannot
  submit or resume. A new attempt requires a new state/receipt path and a new
  user message after the prior report.
- Keep the exact tracker entry `preflighting` through the payload canary and
  require that state immediately before its scheduler side effect. Stop at
  the successful canary boundary. Only a later invocation may launch
  production, and immediately before that production scheduler side effect it
  must validate the canonical tracker entry as `launch-ready`. Remote
  controllers consume a short-lived, hash-bound receipt created by the
  canonical tracker validator; they must never treat a staged Markdown copy
  as current authority. After live production readback, the external workflow
  changes it to `queued` or `running`. Within an armed attempt, missing,
  malformed, duplicate, blocked, or stale entries are terminal pre-launch
  failures.
- For the perfect-diode Conv1/Conv2 successor, staged Jean Zay source and
  bootstrap validation is part of its armed long-run launch gate and must run
  exactly once through
  `validate_mnist_conv_perfectdiode_successor_staged_jeanzay.sh`. Do not
  reconstruct or selectively retry its SSH subcommands.
