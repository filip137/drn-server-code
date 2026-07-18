# `small_network` archival snapshot

Archive tag: `archive/small-network-v1`

The tag records the final supported state of the legacy dense
Moons/Yin-Yang/Digits workflow before its paper-facing retirement. The snapshot
contains the source, configuration files, notes, shell helpers, and dependent
analysis/training tools through Git history, including:

- `labs/small_network.py`
- `labs/small_network_core.py`
- `labs/small_network_config.py`
- `labs/configs/`
- the legacy Digits/Moons tools in `labs/tools/` and `labs/shfiles/`
- `labs/REFactor_plan_small_network.md` and
  `labs/digits_training_validation_todo.md`

The active tree keeps dependency-free error shims at the two historical module
paths for one cleanup cycle. Generated experiment artifacts are intentionally
outside the archive commit and remain unchanged in their existing result roots.

To inspect or restore the archived implementation without changing the current
worktree:

```bash
git show archive/small-network-v1:labs/small_network.py
git worktree add --detach /tmp/small-network-v1 archive/small-network-v1
```

The supported MNIST Conv replacement is:

```bash
python -m experiments.mnist_conv run --config RUN.json --results-root results
```
