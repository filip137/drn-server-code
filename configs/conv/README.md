# Conv Configs

These JSON files record scientific inputs and immutable provenance for Conv
studies. They are not launch-authority documents and do not encode host
orchestration.

The active scientific contract is indexed by
[`docs/conv_paper_hyperparameter_protocol.md`](../../docs/conv_paper_hyperparameter_protocol.md).
Use a config only with the runner that documents its format. Configs tied to a
removed historical runner remain provenance and are not silently accepted by a
different runner.

The current data split is deliberate:

- ordinary MNIST configs select operational `T/K`, rho targets,
  parameter-specific learning rates, and the bounded initializer;
- deterministic medium-affine MNIST configs consume frozen handoffs for paper
  training.

The existing
`perfectdiode_conv12_initialization_rho_comparison_20260727_v1.json` is an
immutable Conv1/Conv2 predecessor for the bounded-initializer question. It
does not by itself implement the active all-depth global selector.

The immutable
`perfectdiode_conv12_best_observed_confirmation_20260727_v1.json` contains
diagnostic entries beyond the nine interim wide-range vectors currently
authorized in `docs/perfectdiode_learning_protocol.md`. Do not launch that
file wholesale or infer the three missing interim rows from it.

Maintained low-level training uses:

```bash
python labs/mnist_train.py --config CONFIG.json --output-dir RESULTS
```

Run one or more complete configs with already-selected learning-rate vectors
using:

```bash
python -m experiments.exact_run CONFIG.json [CONFIG2.json ...] \
  --output-root RESULTS
```

Optimizer-aware layerwise/two-rho grids use:

```bash
python -m experiments.rho_search SOURCE_CONFIG.json --help
```

That command probes units and executes declared cells. Scientific canaries,
expansion, terminal selection, and post-training `T/K` checks remain governed
by the active protocol documents.

Use `python -m experiments.launch` only to transport an existing scientific
command to local tmux, an SSH/tmux host, or Slurm.
