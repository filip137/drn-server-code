# Energy-Based Learning Framework

Coordinate-descent simulations of dissipative resistive networks, including
the experiments used to study bidirectional amplification.

- Core models: `model/`
- Training code: `training/` and `labs/mnist_train.py`
- Scientific runners and analysis: `experiments/`
- Active Conv protocols: `docs/conv_paper_hyperparameter_protocol.md`
- Lean execution workflow: `docs/experiment_workflow.md`

The execution layer intentionally stays small. Scientific runners own their
configs and outputs; `python -m experiments.launch` only starts an existing
command on a configured target and reports its handle.

Complete configs with already-selected learning rates run through
`python -m experiments.exact_run`. It supports a cheap one-batch smoke and
automatic Slurm-array indexing without generating another study format.

Conv layerwise two-rho update-ratio searches are available through
`python -m experiments.rho_search`; see `docs/experiment_workflow.md`.
