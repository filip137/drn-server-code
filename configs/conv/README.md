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

The current machine-readable Conv1/Conv2 wide-range LR authority is
`perfectdiode_conv12_unbounded_fixed_lr_handoff_20260729_v1.json`. It contains
all 12 user-authorized vectors at the supplied precision. It is a handoff, not
a complete runnable training config; generate one complete exact-run config
per entry and copy its named mapping and ordered vector unchanged.
The trainer runtime order is weights first, then the dense weight, then biases;
the handoff and `experiments.exact_run` validate that ordering against the
named mapping before training.

The immutable
`perfectdiode_conv12_best_observed_confirmation_20260727_v1.json` is the
full-precision diagnostic predecessor. It is not launch authority. Do not
launch it wholesale or silently replace the user-authorized values with its
longer decimal representations.

The selected Conv3 unbounded LR vectors are recorded in
`perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json`. Its six
vectors are copied exactly from the winning `cell.json` artifacts, including
the bias-specific `q90_cap` values. The source study used `weight_min=0`,
`weight_max=null`; Filip explicitly authorized using those fixed vectors in
the downstream `[0,100]` paper configs, and the source-contract mismatch stays
recorded in the handoff.

The 18 complete seed-0 wide-range paper configs are under
`paper_medium_affine_perfectdiode_wide_seed0_20260729_v1/`. They use
deterministic medium-affine MNIST, Conv1/Conv2/Conv3 budgets of 10/30/30
epochs, best-validation checkpoint selection, and one terminal official-test
evaluation.

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
