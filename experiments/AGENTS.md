# AGENTS

## Scope
This guidance applies to experiment scripts, launchers, collectors, and monitors under `experiments/`, especially MNIST Conv DRN amplification paper work.

## Conv Amplification Run Rules
- Before changing launchers or interpreting results for Conv amplification, read `../docs/conv_paper_hyperparameter_protocol.md` and `../docs/amplification_experiment_curation.md`.
- For current state, also check `../docs/current_state.md`; when present, use `../docs/hard_sigmoid_experiments.md` and `../docs/perfect_diode_experiments.md` for active hard-sigmoid or perfect-diode threads.
- Keep architecture, preprocessing, nonlinearity, amplification grid, seed list, batch size, epoch budget, and checkpoint rule explicit in every launcher or collector.
- Do not mix matched-operating-point hard-sigmoid rows, tuned upper-envelope rows, same-raw-`input_gain` diagnostics, and preliminary screens without labeling the category in outputs and summaries.
- For hard sigmoid, preserve the per-amplification saturation-target calibration contract. Do not replace calibrated raw `input_gain` values with a shared raw `input_gain` unless the run is explicitly diagnostic.
- For perfect diode, use projected KKT residuals for clamped hidden-layer convergence checks and keep raw residuals as diagnostics.
- Choose or verify `K` before LR or final training. Residual-vs-K diagnostics should include the intended training `K` and larger sentinel values.
- LR screens may start on seed `0`, but final paper claims require validating the frozen settings on multiple seeds when runtime permits more seeds.
- Final paper launchers should write enough provenance for later aggregation: resolved config, metrics, best/final checkpoints, best/final weights, histories, selected `K`, `input_gain`, LR, best epoch, best accuracy, final accuracy, final train loss, and final test loss.

## Launching
- Preserve the same launcher/config contract across local tmux lanes and Jean Zay whenever possible.
- For large batches, follow the root `AGENTS.md` launch policy for checking `tmux main`, `tmux akibscomputer`, `tmux trex`, and then using Jean Zay R3 `fmu@v100` for the remainder.
- Keep Jean Zay result outputs under `/lustre/fsn1/projects/rech/fmu/$USER/server_code/results` unless the user explicitly requests another project/account.
