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
- All nine hard-sigmoid gains and row-specific operational `T/K` values are frozen; the perfect-diode calibration and `T/K` rule remain pending. Do not launch paper-facing LR screens, long checks, or final training until the later training protocol is frozen.
- Training batch size, optimizer, LR protocol, epoch budget, final seeds, and checkpoint inclusion rules remain unresolved.
- Calibration outputs must preserve enough provenance for later aggregation: resolved dataset transform, affine and model seeds, convolution pipeline, nonlinearity, amplification values, calibration T, adaptive-equilibrium setting, selected gain, target occupancy, and every hidden-layer measured occupancy.

## Launching
- Preserve the same launcher/config contract across local tmux lanes and Jean Zay whenever possible.
- For large batches, follow the root `AGENTS.md` launch policy for checking `tmux main`, `tmux akibscomputer`, `tmux trex`, and then using Jean Zay R3 `fmu@v100` for the remainder.
- Keep Jean Zay result outputs under `/lustre/fsn1/projects/rech/fmu/$USER/server_code/results` unless the user explicitly requests another project/account.
