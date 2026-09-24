# Experiment Workflow

The execution layer is deliberately small: a scientific runner plus an
optional transport command.

## Standing Agent Authorization

An assigned experiment task authorizes its ordinary lifecycle: prepare the
readable config or command, run same-path smoke and scientific gates, launch
on an available configured target, monitor, retry understood operational
failures, collect and validate outputs, and interpret the results.

Before substantial compute starts, record and report the goal, cases, target,
budget, expected duration, and result directory. Add the top-level result
directory to the persistent Recent Experiment Directories table in
[`current_simulations.md`](current_simulations.md) with state `planned`. This
is an observability step, not a separate approval gate.

## Scientific Paths

There are two distinct evidence paths on ordinary MNIST:

1. The deterministic 55,000/5,000 training/validation split selects `T/K`,
   optimizer-specific rho targets, raw parameter-specific LR vectors, bounded
   initializer, EqProp beta, and checkpoints. It never reads the official test
   split.
2. After the complete paper contract, inclusion set, and checkpoint identities
   are sealed, a paper evaluation reads the official 10,000-example MNIST test
   split exactly once per eligible run. Existing checkpoints may enter this
   path only through the reuse gate in the paper experiment definition.

Deterministic medium-affine MNIST is historical or optional robustness work,
not the active paper path.

When Filip supplies complete configs with exact LR vectors, use the exact-run
path directly. Do not run a rho search or calibration first:

```bash
python -m experiments.exact_run \
  CONFIG1.json CONFIG2.json \
  --output-root results/fixed-run --device cuda
```

Use the same scientific command with `--smoke` locally before a long or remote
run:

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
Apply the [RTX 5090 sharing policy](../AGENTS.md#sharing-rtx-5090-gpus-with-ben):
Ben's workload or high utilization alone does not block admission when our
measured memory requirement plus headroom fits. Preserve his processes and
launcher lanes; measure expected runtime under sharing.

## Overnight Runs

Use **22:00–08:00 in Europe/Paris** for long runs within an assigned experiment.
Prepare during the day and arm the queue before the window begins; no new
human approval is needed for already authorized cases. The default is to
release overnight GPU workers by 08:00. Record any permitted daytime
continuation explicitly in the run plan.

1. List the exact pending configs/commands, target, priority, total budget,
   expected duration and result paths. Add the persistent `planned` directory
   row before creating outputs or arming the queue. Complete the applicable
   smoke and scientific gates and verify staged inputs. Retain the stricter
   immediate-smoke/submission rule for Jean Zay jobs.
2. Use one readable, study-local queue script launched through tmux or
   SSH/nohup, with a unique session, log, PID/handle and exit record. Record
   concrete start/end dates with the Europe/Paris offset; account for the
   midnight crossing and daylight-saving changes. If already inside the
   window, use the remaining time. An expired window advances to the next
   night within the study deadline; it must not trigger an immediate daytime
   launch. Existing dated experiment schedulers are examples, not reusable
   standing services with valid dates or cases.
3. At actual admission, recheck access, GPU memory, process owners and existing
   queues. Permit Ben/RTX 5090 sharing under the standing policy. Start only
   one copy of each intended run and use a lock to prevent duplicate queue
   controllers. Default to one of our training workers per physical GPU;
   this can coexist with Ben. Preserve matched-pair placement requirements.
4. Base admission on measured throughput, including contention, and reserve
   time for checkpointing and shutdown. Admit a complete run, or a verified
   resumable segment, that fits the remaining window. Resume with model,
   optimizer, scheduler, RNG/data-order and any BN/gain state intact. Retain
   the original training horizon. Do not assume a weights-only checkpoint,
   SIGSTOP, or a forced kill provides a scientific continuation or frees GPU
   memory. If exact resumption is unavailable, select a shorter authorized
   case that fits; leave other cases explicitly unstarted.
5. Keep the detached launcher and worker runtime limits effective after an
   SSH disconnect. Monitor processes and actual artifact/epoch progress at
   least every 30 minutes under the normal watchdog workflow. Preserve exit
   status and distinguish planned pauses, numerical divergence and operational
   failure. A passive queue script is not a substitute for agent monitoring.
6. By 08:00, finish or checkpoint/pause our overnight workers unless daytime
   continuation was recorded. Stop admitting new night jobs. Collect and
   validate results, update `current_simulations.md`, and provide the morning
   coverage summary. Carry remaining authorized cases to a later night only
   within the original budget/deadline; never mark partial work complete.

## Before A Long Run

1. Record goal, cases/seeds, scientific config, budget, target, expected
   duration, and result directory; add the persistent result-directory row with
   state `planned`.
2. Verify the exact source/config identity on the target.
3. Check the target GPU, tmux lane, or Slurm allocation.
4. Run a synchronous local smoke through the same scientific runner and
   config, and require its semantic completion artifact.
5. Run the active `T/K`, rho-canary, or other scientific gate.
6. Immediately launch production once, record the tmux handle or Slurm job ID,
   and update the persistent row to `running`. Before any live Jean Zay
   `sbatch`, verify that the `planned` row identifies the result link or,
   temporarily, the launch name, config or wrapper, and expected remote output
   location. Do not submit a separate remote canary unless a prior
   target-specific failure has invalidated the local-only policy.
7. Monitor to a declared deadline.

## Launch Examples

`experiments.launch` treats the experiment command as opaque. Its optional
local-canary command runs synchronously; production is submitted only if the
canary exits zero and every required artifact is non-empty. Add `--dry-run` to
inspect both steps without executing either.

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
  --slurm-arg=--time=04:00:00 \
  --local-canary-command \
    "python -m experiments.exact_run CONFIG0.json CONFIG1.json CONFIG2.json CONFIG3.json CONFIG4.json CONFIG5.json --output-root results/conv3-fixed-local-canary --device cpu --dataset-root /home/filip/datasets/mnist --smoke --summary-json results/conv3-fixed-local-canary/summary.json" \
  --local-canary-log results/conv3-fixed-local-canary/canary.log \
  --local-canary-require results/conv3-fixed-local-canary/summary.json \
  --dry-run -- \
  python -m experiments.exact_run \
    CONFIG0.json CONFIG1.json CONFIG2.json CONFIG3.json CONFIG4.json CONFIG5.json \
    --output-root /lustre/fsn1/projects/rech/fmu/$USER/server_code/results/conv3-fixed \
    --device cuda
```

Remove `--dry-run` to run the local canary and, on success, make one Jean Zay
submission. The returned JSON contains both the local artifact hashes and the
production job ID. A local failure returns `state: blocked` and never contacts
Jean Zay. Before removing `--dry-run`, verify that the production submission
has a `planned` row in `current_simulations.md`. Immediately after submission,
write its returned job ID into the short summary and update the row to
`running`.

`--dataset-root` is a recorded transport-only override for local canaries whose
frozen configs point at a remote dataset path. It does not alter preprocessing,
splits, affine seeds, minibatch order, or model and optimizer fields.

The launcher does not stage source, choose scientific settings, retry,
collect, or select results. If a future production failure is caused by the
Jean Zay module, CUDA/V100 environment, Lustre, staging, or Slurm contract,
restore a live Jean Zay canary for that affected execution contract.

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

The command implements the optimizer probe, a restarted safety canary for each
cell, and candidate-grid execution. Bound occupancy and projection efficiency
are report-only diagnostics. The focused bounded runner adds the
architecture-specific core, one factor-of-three expansion wave, and the 2%
loss plateau selector. Conv1/Conv2 use adaptive safe-center attempts:

```bash
python -m experiments.run_conv12_bounded_rho plan
python -m experiments.run_conv12_bounded_rho smoke --surface-index 0
python -m experiments.run_conv12_bounded_rho run-all
```

Conv3 uses `T=K=8` and its fixed high 3x3 core:

```bash
python -m experiments.run_conv12_bounded_rho \
  --study configs/conv/perfectdiode_conv3_bounded_rho_baseline_ours_20260729_v1.json \
  plan
python -m experiments.run_conv12_bounded_rho \
  --study configs/conv/perfectdiode_conv3_bounded_rho_baseline_ours_20260729_v1.json \
  smoke --surface-index 0
```

Both focused configs deliberately exclude legacy. The default config excludes
Conv3, while the Conv3 config excludes Conv1/Conv2. Neither result alone
publishes the all-depth global bounded-initializer decision.

For these focused bounded studies, 90% remains a reported performance
threshold. If no safety-clean core candidate reaches it, selection falls back
to the lowest-loss completed candidate. A fallback selected point on a tested
edge receives the same single factor-of-three expansion wave, and the terminal
result explicitly reports whether the explored rho range is still bounded or
is bracketed.

## During And After

- Follow the canonical run-bundle and promotion rules in
  [`experiment_reporting.md`](experiment_reporting.md). A failed run records
  its error in `status.json` and must not publish a successful `result.json`.
- Record the resolved config, command, commit, environment, target/job
  identity, logs, metrics, checkpoints, and completion record in one
  run-specific directory.
- Keep monitoring observable and bounded by a deadline.
- Retry an understood operational failure when science and budget remain in
  scope. Cancel and replace only jobs launched for the current task that are
  invalid, obsolete, or operationally broken.
- Copy remote results locally and validate the local copy before using it as
  evidence.
- Maintain the persistent Recent Experiment Directories row for every
  top-level directory created during the task, including smoke, failed,
  recovery, partial, and replacement directories. Link the exact local result
  directory and mention a remote-only source in the short summary rather than
  relying on chat history.
- Use `collecting` while remote results are not yet fully present and validated
  locally. Use `ready-for-review` only after expected coverage is reconciled,
  included bundles validate, and all failures, exclusions, and superseded
  directories are identified. Use `partial`, `complete`, `failed`, or
  `superseded` when those more accurately describe the terminal handoff.
- Analyze outputs against declared completion criteria. Label partial
  evidence, confounds, and deviations explicitly.
- Keep live `running` state in the generated block of
  [`current_simulations.md`](current_simulations.md); this does not replace the
  persistent result-directory row. A reviewing agent changes the row to
  `under-review` when work begins and to `reviewed` only after adding the study
  conclusion to [`experimental_manifest.md`](experimental_manifest.md).
- Update `docs/current_state.md` with terminal scientific status, not transient
  busy/idle node state.
