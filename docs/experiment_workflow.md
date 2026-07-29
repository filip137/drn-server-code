# Experiment Workflow

The execution layer is deliberately small: a scientific runner plus an
optional transport command.

## Standing Agent Authorization

An assigned experiment task authorizes its ordinary lifecycle: prepare the
readable config or command, run same-path smoke and scientific gates, launch
on an available configured target, monitor, retry understood operational
failures, collect and validate outputs, and interpret the results.

Before substantial compute starts, record and report the goal, cases, target,
budget, expected duration, and result directory. This is an observability
step, not a separate approval gate.

## Scientific Paths

There are two distinct training paths:

1. Ordinary MNIST selects `T/K`, optimizer-specific rho targets, raw
   parameter-specific LR vectors, and the bounded initializer.
2. Deterministic medium-affine MNIST runs the paper configs with those
   handoffs unchanged.

When Filip supplies complete configs with exact LR vectors, use the exact-run
path directly. Do not run a rho search or calibration first:

```bash
python -m experiments.exact_run \
  CONFIG1.json CONFIG2.json \
  --output-root results/fixed-run --device cuda
```

Use the same command with `--smoke` before a long or remote run:

```bash
python -m experiments.exact_run \
  CONFIG1.json CONFIG2.json \
  --output-root results/fixed-run --device cuda --smoke
```

The smoke performs one real optimizer step and one validation batch and writes
normal artifacts below `OUTPUT_ROOT/smoke/`. An unchanged accepted `T/K`
operating point is cited rather than rerun.

## Configured Targets

Targets live in `configs/experiment_targets.json`.

| Target | Kind | Working directory | Preferred use |
|---|---|---|---|
| `local` | foreground | current checkout | tests and short smokes |
| `main` | local tmux session `main` | `/home/filip/server_code` | Conv1 and diagnostics |
| `akib` | SSH/tmux through alias `akib` | `/home/filiposana/server_code` | Conv1/Conv2 |
| `trex` | SSH/tmux through `filip@trex` | `/home/filip/server_code` | Conv2/Conv3 |
| `jean-zay` | SSH/Slurm | `/lustre/fswork/projects/rech/umg/$USER/server_code` | Conv3 and arrays |

`local` and `main` are two execution modes on the same local machine. The
launcher target is `akib`; `akibscomputer` is historical tmux terminology,
not the current target name.

Check live GPU/lane/scheduler availability immediately before launch. Keep one
scientific surface on one target and record that target in its artifacts.

## Before A Long Run

1. Record goal, cases/seeds, scientific config, budget, target, expected
   duration, and result directory.
2. Verify the exact source/config identity on the target.
3. Check the target GPU, tmux lane, or Slurm allocation.
4. Run a same-runner, same-environment, same-device, same-output-path smoke.
5. Run the active `T/K`, rho-canary, or other scientific gate.
6. Launch and record the tmux handle or Slurm job ID.
7. Monitor to a declared deadline.

## Launch Examples

`experiments.launch` treats the experiment command as opaque. Add `--dry-run`
to any `run` command to inspect transport without launching.

Foreground local smoke:

```bash
python -m experiments.launch run local \
  --name conv-smoke --dry-run -- \
  python labs/mnist_train.py --config CONFIG.json --output-dir RESULTS
```

Detached local tmux:

```bash
python -m experiments.launch run main \
  --name conv1-run \
  --log /home/filip/server_code/results/conv1-run/run.log --dry-run -- \
  python -m experiments.exact_run CONFIG.json \
    --output-root /home/filip/server_code/results/conv1-run --device cuda
```

Akib:

```bash
python -m experiments.launch run akib \
  --name conv1-rho \
  --log /home/filiposana/server_code/results/conv1-rho/run.log --dry-run -- \
  python -m experiments.rho_search SOURCE_CONFIG.json \
    --output-root /home/filiposana/server_code/results/conv1-rho \
    --rho-conv 0.001 0.003 0.009 \
    --rho-dense 0.003333333333 0.01 0.03 \
    --epochs 3 --device cuda
```

Trex:

```bash
python -m experiments.launch run trex \
  --name conv2-run \
  --log /home/filip/server_code/results/conv2-run/run.log --dry-run -- \
  python -m experiments.exact_run CONFIG.json \
    --output-root /home/filip/server_code/results/conv2-run --device cuda
```

Jean Zay array:

```bash
python -m experiments.launch run jean-zay \
  --name conv3-fixed \
  --log /lustre/fsn1/projects/rech/fmu/$USER/server_code/results/conv3-fixed/logs/%A_%a.log \
  --slurm-arg=--array=0-5%6 \
  --slurm-arg=--time=04:00:00 --dry-run -- \
  python -m experiments.exact_run \
    CONFIG0.json CONFIG1.json CONFIG2.json CONFIG3.json CONFIG4.json CONFIG5.json \
    --output-root /lustre/fsn1/projects/rech/fmu/$USER/server_code/results/conv3-fixed \
    --device cuda
```

The launcher does not stage source, choose scientific settings, retry,
collect, or select results.

## Status

Detached tmux/SSH jobs require the handle returned by launch and the same log
path:

```bash
python -m experiments.launch status main TMUX_WINDOW_ID \
  --log /home/filip/server_code/results/conv1-run/run.log

python -m experiments.launch status akib conv1-rho \
  --log /home/filiposana/server_code/results/conv1-rho/run.log

python -m experiments.launch status trex conv2-run \
  --log /home/filip/server_code/results/conv2-run/run.log

python -m experiments.launch status jean-zay JOB_ID
```

For Jean Zay resource details, monitoring, and result collection, follow
[`jean-zay.md`](jean-zay.md).

## Rho Search Building Block

`experiments.rho_search` accepts a normal Conv `source_config.json`, probes
the initial optimizer proposals, converts separate Conv and Dense rho targets
into a complete per-parameter LR vector, and runs a declared Cartesian grid:

```bash
python -m experiments.rho_search SOURCE_CONFIG.json \
  --output-root results/conv-rho \
  --rho-conv 0.001 0.003 0.009 \
  --rho-dense 0.003333333333 0.01 0.03 \
  --epochs 3 --device cuda
```

Use `--optimizer Adam` for fresh-state Adam proposal units; otherwise the
source config's optimizer is used. The default perfect-diode bias policy is
the Q90 cap.

The default probe checks stability at 32, 64, then 128 batches. It uses the
deterministic 55,000/5,000 ordinary-MNIST split and does not read the official
test set. Each cell restarts from the same initializer-specific checkpoint and
shuffle state.

The measured weight unit is:

```text
median_batch(
  RMS(nominal-LR-one pre-projection proposal) / RMS(initial weight)
)
```

The command implements probe and grid execution. It does not by itself
implement the complete 640-step scientific-canary, factor-of-three expansion,
2% plateau selection, global bounded-initializer selection, or post-training
`T/K` audit. Those remain requirements of the active perfect-diode protocols.

## During And After

- Record the resolved config, command, commit, environment, target/job
  identity, logs, metrics, checkpoints, and completion record in one
  run-specific directory.
- Keep monitoring observable and bounded by a deadline.
- Retry an understood operational failure when science and budget remain in
  scope. Cancel and replace only jobs launched for the current task that are
  invalid, obsolete, or operationally broken.
- Copy remote results locally and validate the local copy before using it as
  evidence.
- Analyze outputs against declared completion criteria. Label partial
  evidence, confounds, and deviations explicitly.
- Update `docs/current_state.md` with terminal scientific status, not transient
  busy/idle node state.
