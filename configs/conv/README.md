# Conv Configs

These JSON files record scientific inputs for Conv studies. They are not
launch-authority documents and do not encode host orchestration.

Use a config only with the runner that documents its format. Configs tied to a
removed historical runner remain useful provenance but are not accepted
silently by a different runner.

For ordinary MNIST/Fashion-MNIST training, the maintained low-level entry
point is:

```bash
python labs/mnist_train.py --config CONFIG.json --output-dir RESULTS ...
```

Run one or more complete configs with already-selected learning rates using:

```bash
python -m experiments.exact_run CONFIG.json [CONFIG2.json ...] \
  --output-root RESULTS
```

The reusable Conv amplification sweep entry point is:

```bash
python experiments/train_mnist_bp_conv_amplification_sweep.py --help
```

Optimizer-aware layerwise/two-rho Conv grids use the ordinary generated source
config:

```bash
python -m experiments.rho_search SOURCE_CONFIG.json --help
```

Experiment scripts may build configs and call the same training function
directly. Use `python -m experiments.launch` only to transport that command to
local tmux, an SSH/tmux host, or Slurm.
