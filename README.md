# Energy-Based Learning Framework

This repository simulates and trains dissipative resistive networks with
coordinate-descent equilibrium solvers. Its core layout follows the original
[energy-based-learning framework](https://github.com/rain-neuromorphics/energy-based-learning):
model mechanics live in `model/`, reusable learning mechanics live in
`training/`, and executable research compositions live in `experiments/`.

The supported experiment entrypoint is:

```bash
python -m ebl describe --experiment small_drn.v1
python -m ebl train \
  --config examples/small_drn/base.json \
  --output-dir runs
```

Training, linspace analysis, validation, and legacy checkpoint import are
separate commands. Evaluation commands always require an explicit named
weights artifact; they never search for the “latest” checkpoint.

```bash
python -m ebl linspace \
  --config examples/small_drn/base.json \
  --weights runs/<run-id>/checkpoints/weights.pt \
  --output-dir runs

python -m ebl validate \
  --config examples/small_drn/base.json \
  --weights runs/<run-id>/checkpoints/weights.pt \
  --output-dir runs
```

Every invocation creates an immutable run directory containing its resolved
config, source/runtime manifest, status, JSON-lines metrics, result record,
logs, checkpoints, and content-hashed artifacts. See
[`docs/experiment_runtime.md`](docs/experiment_runtime.md) for the extension
boundaries, checkpoint rules, worktree workflow, and campaign protocol.
