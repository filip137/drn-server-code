# TODO

- Add numerical safeguarding in the minimizer/training loop to prevent runaway state divergence.
- Add explicit non-finite guards (`isfinite`) immediately after each equilibration step and stop early with a clear diagnostic.
- Add a state-magnitude safety cap (or clamp) during debug mode to detect and block blow-up before `inf`/`nan` propagation.
- Add a config-level stability profile (lower `nudging`, safer learning rates) for deep/2-hidden double-diode runs.
- Log first divergence location (layer name, iteration index, max `|state|`, max `|b|`) to speed up root-cause analysis.
