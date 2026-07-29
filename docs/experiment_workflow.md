# Experiment Workflow

The workflow is deliberately small: a scientific runner plus an optional
transport command.

## Exact-config fast path

When Filip provides the exact setups and learning-rate vectors, each complete
trainer config is the scientific contract. Run those configs unchanged:

```bash
python -m experiments.exact_run \
  CONFIG1.json CONFIG2.json \
  --output-root results/fixed-run --device cuda
```

This path does not perform a rho search, calibration, adaptive expansion, or
plan generation. Before a long or remote run, use the same command with
`--smoke`; it performs one real optimizer step and one validation batch and
writes normal trainer artifacts below `OUTPUT_ROOT/smoke/`:

```bash
python -m experiments.exact_run \
  CONFIG1.json CONFIG2.json \
  --output-root results/fixed-run --device cuda --smoke
```

Each config must contain matching explicit `lr` and
`optimizer.learning_rate` vectors. With no index, configs run sequentially.
`--index N` selects one. On Slurm, `SLURM_ARRAY_TASK_ID` selects the config
automatically. Every case writes `exact_run.json` beside the normal trainer
artifacts and refuses to merge into a non-empty case directory.

An unchanged accepted Conv `T/K` operating point is cited rather than rerun.
A scientific change to the architecture, nonlinearity, amplification, input
gain, `T/K`, or equilibrium/gradient algorithm still needs its applicable
scientific check.

## Before a long run

1. Write down the goal, cases/seeds, scientific config, budget, target,
   expected duration, and result directory.
2. Show that short proposal to Filip and get explicit approval.
3. Check that the target GPU or tmux lane is free.
4. Run a short end-to-end smoke through the same runner, environment, device,
   and output-writing path. Training smokes perform at least one optimizer
   step.
5. Run the active scientific `T/K` gate when the Conv protocol requires it.

## Launch

`experiments.launch` treats the experiment command as opaque:

```bash
python -m experiments.launch run local \
  --name conv-smoke -- \
  python labs/mnist_train.py --config CONFIG.json --output-dir RESULTS

python -m experiments.launch run trex \
  --name conv-run --log /home/filip/server_code/results/conv-run/run.log -- \
  python EXPERIMENT_SCRIPT.py --config CONFIG.json
```

For a Jean Zay array, pass the array and time limits directly to Slurm. The
exact-run worker maps each task ID to one config:

```bash
python -m experiments.launch run jean-zay \
  --name conv-fixed \
  --log /lustre/fsn1/projects/rech/fmu/$USER/server_code/results/conv-fixed/logs/%A_%a.log \
  --slurm-arg=--array=0-5%6 \
  --slurm-arg=--time=04:00:00 -- \
  python -m experiments.exact_run \
    CONFIG0.json CONFIG1.json CONFIG2.json CONFIG3.json CONFIG4.json CONFIG5.json \
    --output-root /lustre/fsn1/projects/rech/fmu/$USER/server_code/results/conv-fixed \
    --device cuda
```

Targets live in `configs/experiment_targets.json`. `--dry-run` prints the
exact transport command without executing it. Detached launches return a
tmux handle or Slurm job ID:

```bash
python -m experiments.launch status trex conv-run \
  --log /home/filip/server_code/results/conv-run/run.log
python -m experiments.launch status jean-zay JOB_ID
```

The launcher does not stage source, choose scientific settings, retry, update
a tracker, or collect results. Do those directly and visibly.

## Conv layerwise two-rho search

`experiments.rho_search` accepts a normal Conv `source_config.json`, probes
the initial minibatch gradients, converts separate Conv and Dense rho targets
into a complete per-parameter learning-rate vector, and runs the Cartesian
grid:

```bash
python -m experiments.rho_search SOURCE_CONFIG.json \
  --output-root results/conv1-rho \
  --rho-conv 0.001 0.003 0.009 \
  --rho-dense 0.003333333333 0.01 0.03 \
  --epochs 3 --device cuda
```

Use `--optimizer Adam` for fresh-state Adam proposal units; otherwise the
source config's optimizer is used. The default bias policy is the newer Q90
cap. `--bias-policy tied` reproduces the older hard-sigmoid convention.

The default probe checks stability at 32, 64, then 128 batches and stops at
the first stable point. It uses the deterministic 55k/5k split of MNIST's
training set; the official test set is not read. `probe.json`, each cell's
config and metrics, and `summary.{json,csv}` are written below the output
root. `--probe-only` stops after calibration and `--dry-run` only prints the
resolved search.

The measured unit is
`RMS(nominal-LR-one pre-projection proposal) / RMS(initial weight)`.
Conv/Dense weight units use the batch median. A bias uses its attached Conv
weight for normalization and its batch Q90 for the cap. Each cell restarts
from the same seed and shuffle state.

This is the optimizer-aware layerwise/two-rho diagnostic. It does not parse
the deleted historical study schemas or reproduce the older v3 scalar-LR
selection rule.

## During and after

- Record the resolved config, command, commit, environment, handle/job ID,
  logs, metrics, and checkpoints in one run-specific directory.
- Keep active monitoring observable and bounded by a deadline.
- Retry an understood operational failure when the approved science and
  budget are unchanged. Ask again for scientific changes, material budget
  expansion, cancellation, or deletion.
- Copy remote results locally and validate the local copy before using it as
  evidence.
- `docs/current_experiments.md` is a concise human tracker, not a launch gate.
