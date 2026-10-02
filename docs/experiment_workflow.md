# Experiment Workflow

This is the operational authority for GPU placement, sharing, timing, smoke,
launch, monitoring, recovery and collection. The
[experiment-loop skill](../.agents/skills/experiment-loop/SKILL.md) owns the
Explorer → Codifier → Monitor → Reviewer lifecycle and role-specific reading.

## Campaign-to-execution handoff

The pilot note or campaign experiment holds both the scientific specification and the Codifier's
execution/monitoring handoff. The Monitor receives that bounded assignment and
returns factual local evidence; the Reviewer adds interpretation to the same
pilot/result note. Use the skill's single-note pilot convention for small tests;
do not scaffold a full campaign just to execute one. For existing non-campaign runs,
retain the existing plan and paths. Handoffs link the current note and report deltas;
they do not duplicate its narrative or full metadata.
Planning/setup alone launches nothing. Use the existing runner and shared queue
below; do not introduce experiment-specific launch schemas or approval gates.

## Standing Agent Authorization

An assigned experiment task authorizes its ordinary lifecycle: prepare the
readable config or command, run same-path smoke and scientific gates, launch
on an available configured target, monitor, retry understood operational
failures, collect and validate outputs, and interpret the results.

Before substantial compute starts, record and report the goal, cases, target,
budget, expected duration, and result directory in the owning experiment/pilot
note. This is an observability step, not a separate approval gate. Use that same
record for the execution handoff; do not update the retired central documents.

## Scientific Paths

### Retired pre-launch diagnostics (September 25, 2026)

At Filip's request, routine launches no longer require Conv1/Conv2 fixed-T/K
gradient comparisons, Conv3 T-residual or K-gradient sweeps, Conv3 operating-point
replays, optimizer/rho proposal probes, or CIFAR solver and migration audits
(including full validation replay and accuracy/CE drift thresholds). Use the
declared T/K and explicit learning-rate vectors directly, including after an
initializer change; do not automatically retune or requalify them.

These diagnostics remain available only when explicitly assigned as scientific
work. The older selection protocols retain their historical methods and evidence;
their prerequisite language does not reinstate these launch requirements. An
explicit rho-selection task still needs its proposal measurements to derive LRs.
Omitting an audit records no convergence, gradient-fidelity or migration pass.

`experiments.train_cifar_l8_analog --mode prepare` now prepares the split and
initializer with configured iteration counts, recording `configured_without_audit`.
Use `--qualify-solver` with preparation only for an explicitly assigned solver
diagnostic. Existing prepared assets remain usable with their recorded iteration
counts and identity checks. Historical rho/T/K selection runners are retained for
explicit selection work; routine fixed-config training uses `experiments.exact_run`.

Keep source/config and dataset identity, checkpoint/full-state restore checks,
resource admission, same-runner execution smoke, runtime nonfinite guards and
artifact validation. Existing EqProp-specific work, paper matching/test isolation,
and during/after-run checks retain their declared scope. Do not rewrite historical
results or modify the frozen execution of already launched jobs.

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
path directly. Do not run a rho search, calibration or broad test campaign first:

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
normal artifacts below `OUTPUT_ROOT/smoke/`. Existing `T/K` evidence may be cited;
new qualification is not required for routine launches.

## GPU Placement

Prepared lab GPU jobs may use the shared
[run-watch queue](/home/filip/.codex/skills/run-watch/SKILL.md). The Codifier stages
and verifies eligible placements before submission. Keep the job ID in the existing
experiment/pilot handoff; the persistent service then owns launch and routine
monitoring. Do not assign another Monitor or watch to the same job. The queue stores
operational state, while the existing note remains the scientific record.

In Europe/Paris time, Monday–Friday 09:00–18:00, new admissions may occupy at most
two distinct RTX 5090s across all our jobs, including unmanaged jobs. Existing jobs
may finish after 09:00 even when usage exceeds two; wait until it falls below two
before occupying another 5090. A second Conv2 job may fill an already counted GPU
while fleet usage is at most two. Ben-only workloads do not consume our quota.
Evenings, nights and weekends have no fleet-count cap; ownership, memory and each
job's explicit budget/window restrictions still apply. Do not preempt other jobs.

Prefer memory-heavy work on 5090s and smaller work on 3080s/3090s. Within explicit
priority tiers, 5090-only jobs take precedence, followed by larger memory requests;
smaller jobs may use spare 5090 capacity. Reserve peak memory plus headroom and
preserve the scientific config. The shared queue's default headroom is the larger
of 1 GiB or 10% of declared peak memory.

Prefer placement by model size and GPU capacity:

- Conv1: prefer the smaller GPUs—Akib's RTX 3080 or the RTX 3090s on local
  (`local`/`main`) and `nom-cool-1`.
- Conv2: use `resource_profile: "conv2"` in the shared queue. On a 5090, schedule
  up to two in parallel when their combined peak memory plus headroom fits.
  Other GPU models retain one worker; placement remains flexible.
- Conv3: prefer the RTX 5090 machines (`trex`, `fifi`, `loulou`, or another
  host once verified to have an RTX 5090) or `jean-zay`. In the lab queue, set
  `resource_profile: "conv3"`: one worker on a 5090, reserving the measured or
  bounded workload peak plus headroom. Filip removed the fixed31GiB reservation
  on September29; do not infer memory from architecture alone. Admission still
  requires the reservation to fit alongside authorized existing users.

These are placement preferences; preserve the scientific config and matched
comparison requirements when choosing a host.
Jean Zay defaults to `fmu@v100`; put outputs under
`/lustre/fsn1/projects/rech/fmu/$USER/server_code/results`.

<a id="sharing-rtx-5090-gpus-with-ben"></a>

## Sharing GPUs With Ben

**Exp014 exception, September 25:** Filip explicitly authorized running the
relative-noise training campaign alongside Kellian. Its existing per-study queues
may admit Conv3 using measured peak memory plus headroom instead of the generic
then-current31GiB exclusive reservation (removed generally September29). Preserve scientific settings, both users' jobs and
the original study caps/deadline. This does not authorize sharing with Adrien or
other unknown/unapproved owners, or give a second scheduler ownership of the run.

Filip explicitly permits our simulations to share a GPU with Ben on all
authorized simulation hosts and GPU models, including RTX 3080, 3090 and 5090,
during both daytime and nighttime. His presence or even 100% GPU utilization
alone is not a reason to reject the host or wait for it to become idle. This
is standing permission to share; do not ask again for each launch. The September
25 clarification extends the earlier RTX 5090 permission to all these GPUs.

- Verify the GPU model, free memory, process owners and existing launchers.
  Identify Ben's workload from process ownership or known launcher metadata;
  do not assume that every occupied GPU belongs to him.
- Admit our job when its measured peak memory plus headroom fits alongside
  the existing workload. Preserve its scientific batch size, precision and
  config. If it does not fit, choose another target or wait for memory.
- Use our own launcher/session and result path. Leave Ben's processes, queues,
  priorities, MPS service and GPU-wide settings intact. Other users' occupied
  lanes retain their existing protection unless separately authorized.
- Measure our throughput under actual sharing and use that rate for duration
  estimates and deadlines. Record shared use in the run's environment notes;
  do not report shared-GPU timings as exclusive-GPU benchmarks.

## Simulation GPU Inventory

The following machines are authorized simulation resources. This is a hardware
and access inventory, not a statement that their GPUs are idle. Check live GPU
memory, utilization, and running jobs immediately before allocating work.

| Machine | Access / launcher target | GPU inventory | Verification |
|---|---|---|---|
| Local (`nom-cool-2`) | `local` foreground or `main` tmux | 1 × NVIDIA GeForce RTX 3090, 24 GiB | Queried 2026-09-11 |
| `akibscomputer` (Akib) | SSH `akibscomputer` → `filiposana@172.24.6.229` (hostname `integnano-akib`); launcher `akib` still uses SSH `akib` | 1 × NVIDIA GeForce RTX 3080, 10 GiB | Queried via `akibscomputer` on 2026-09-11 |
| `nom-cool-1` | SSH `filip@nom-cool-1` | 1 × NVIDIA GeForce RTX 3090, 24 GiB | Queried 2026-09-11 |
| `trex` | SSH `filip@trex`; launcher `trex` | 1 × NVIDIA GeForce RTX 5090, 32 GiB | Queried 2026-09-11 |
| `riri` | SSH `filip@riri` | Model, count, and memory unverified | SSH authentication failed on 2026-09-11 |
| `fifi` | SSH `filip@fifi` | 1 × NVIDIA GeForce RTX 5090, 32 GiB | Queried 2026-09-11 |
| `loulou` | SSH `filip@loulou` | 1 × NVIDIA GeForce RTX 5090, 32 GiB | Queried 2026-09-11 |

Memory above is nominal capacity; `nvidia-smi` reported 10,240 MiB for the
RTX 3080, 24,576 MiB for the RTX 3090s, and 32,607 MiB for the RTX 5090s.
Re-query unverified hardware before choosing a workload for it.

`local` and `main` share one physical GPU host. `akibscomputer` is both the
historical tmux name and a working SSH alias for Akib, not an additional
machine. Prefer SSH `akibscomputer`: SSH `akib` was unreachable during the
earlier check on 2026-09-11. The configured launcher target `akib` still uses
SSH `akib`; verify that route or update its host before using the launcher.
`nom-cool-1`, `riri`, `fifi`, and `loulou` are not currently named targets in
`configs/experiment_targets.json`; use a direct SSH runner or configure the
target before using `experiments.launch`, and verify the remote checkout and
Python environment. Jean Zay remains available through the existing Slurm
target and its allocation-specific GPU resources.

## Configured Targets

Read [configs/experiment_targets.json](../configs/experiment_targets.json) for
actual transport aliases and working directories; the inventory above identifies
hardware and access. Check the listed simulation hosts and configured jean-zay
target before allocating work. Verify the selected remote checkout, Python
environment and staged inputs. Configuration entries are not proof of available
capacity or a current deployment.

Use `python -m experiments.launch` for common local, tmux, SSH/tmux and Slurm
cases, or a direct command when clearer. Keep one scientific surface on one
recorded target; preserve matched initialization, cohorts, minibatch order and
resolved config. Never take over unrelated launcher lanes or replace, terminate,
suspend or reconfigure unrelated jobs.

## Overnight Runs

Use **22:00–08:00 in Europe/Paris** for long runs within an assigned experiment.
This remains the convention for explicitly overnight-bounded studies. Shared-queue
jobs without a narrower declared window use the availability calendar above; they
do not inherit a blanket 08:00 stop. Existing dated study windows remain binding.
Prepare during the day and arm the queue before the window begins; no new
human approval is needed for already authorized cases. The default is to
release overnight GPU workers by 08:00. Record any permitted daytime
continuation explicitly in the run plan.

1. List the exact pending configs/commands, target, priority, total budget,
   expected duration and result paths in the owning experiment/pilot note
   before creating outputs or arming the queue. Complete the applicable
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
   queues. Permit sharing with Ben on all GPU models under the standing policy. Start only
   one copy of each intended run and use a lock to prevent duplicate queue
   controllers. Default to one of our training workers per physical GPU, except
   the two-Conv2 policy on 5090s above; this can coexist with Ben when memory fits. Preserve matched-pair placement requirements.
4. Base admission on measured throughput, including contention, and reserve
   time for shutdown and, for resumable training, checkpointing. Admit a complete run, or a verified
   resumable segment, that fits the remaining window. Resume with model,
   optimizer, scheduler, RNG/data-order and any BN/gain state intact. Retain
   the original training horizon. Do not assume a weights-only checkpoint,
   SIGSTOP, or a forced kill provides a scientific continuation or frees GPU
   memory. If exact resumption is unavailable, select a shorter authorized
   case that fits; leave other cases explicitly unstarted.
   Initialization replays and independent diagnostic cases do not require training
   checkpoints. A bounded stop that preserves completed case artifacts and records
   incomplete coverage is sufficient; never describe partial work as complete.
5. Keep the detached launcher and worker runtime limits effective after an
   SSH disconnect. Monitor processes and actual artifact/epoch progress at
   least every 30 minutes under the normal watchdog workflow. Preserve exit
   status and distinguish planned pauses, numerical divergence and operational
   failure. Queue-owned jobs use the shared service's deterministic monitoring
   and bounded incident recovery; do not add a second monitoring owner. Other
   passive launch scripts still require the normal Monitor handoff.
6. By 08:00, finish or checkpoint/pause our overnight workers unless daytime
   continuation was recorded. Stop admitting new night jobs. Collect and
   validate results, update the experiment/pilot handoff, and provide the morning
   coverage summary. Carry remaining authorized cases to a later night only
   within the original budget/deadline; never mark partial work complete.

## Before A Long Run

Use the same measured launcher wall-time receipts for remaining-budget calculations
in both configs and launchers, including startup, teardown and charged retries.
If operational accounting differs, reconcile it to those receipts and reduce the
remaining allowance automatically within the authorized budget. Record the correction;
do not block unchanged scientific cases merely because inner-runner timing excludes
startup overhead. Keep scientific settings frozen and never reset the total budget.

1. Record goal, cases/seeds, scientific config, budget, target, expected
   duration, and result directory in the owning experiment/pilot note before launch.
2. Verify the exact source/config identity on the target, including relevant
   uncommitted changes. Prefer a direct runner; extract shared code only after
   a second real use.
3. Check the target GPU, tmux lane, or Slurm allocation.
4. Run a synchronous local smoke through the same scientific runner and
   config, and require its semantic completion artifact.
5. Apply only the remaining checks explicitly required by the assigned task.
   The retired diagnostics above are not launch gates. Use declared T/K and
   learning rates directly; a missing historical audit does not block launch.
6. Immediately launch production once, record the tmux handle or Slurm job ID,
   and update the experiment/pilot handoff to `running`. Before any live Jean Zay
   `sbatch`, including production, arrays, restored canaries and retries, verify
   that the handoff identifies the result link or, temporarily, the launch
   name, config or wrapper, and expected remote output location. Do not submit a
   separate remote canary unless a prior
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
has its configs, target and expected result path in the experiment/pilot handoff.
Immediately after submission, record the returned job ID there and mark it `running`.

`--dataset-root` is a recorded transport-only override for local canaries whose
frozen configs point at a remote dataset path. It does not alter preprocessing,
splits, affine seeds, minibatch order, or model and optimizer fields.

The launcher does not stage source, choose scientific settings, retry,
collect, or select results. If a future production failure is caused by the
Jean Zay module, CUDA/V100 environment, Lustre, staging, or Slurm contract,
restore a live Jean Zay canary for that affected execution contract.

## Status

Role ownership and escalation are defined in
[experiment-loop](../.agents/skills/experiment-loop/SKILL.md#monitor). The
Codifier verifies startup and supplies exact commands/handles to the Monitor.
Only one agent owns routine polling for a run.

Every 30 minutes (sooner near a deadline), synchronize the assigned remote status
and metric observations, then use
`python -m experiments.reporting inspect-runs RUN_DIR...` for explicit run
directories. Compare with the preceding observation, launcher/process state and
a bounded log tail. Update the owning note's current handoff at meaningful status
changes and collection; keep individual heartbeats in the run bundles.

Fresh artifacts alone do not prove a remote process is alive. Stale copies or
failed synchronization alone do not prove it has stalled; report the visibility
gap. Inspect reports observations, not permission to kill/retry. Exit 75 is a
planned pause only when the runner's checkpoint evidence confirms it. Follow the
[reporting contract](experiment_reporting.md#execution-records-and-status)
for state and validation semantics.

Send abnormal progress, failures, deadline risks and capacity for already assigned
pending cases to the Codifier through the coordinator. Continue observing unaffected
runs. Keep healthy updates short. Keep monitoring active or provide an actual
supported scheduled continuation; detached training does not keep an agent alive.

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

Read this section only for assigned rho-search experiments; it is not a prerequisite
for exact-config runs. Follow the experiment's current scientific protocol rather
than treating the example studies below as authorization for a new search.

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
- The Codifier retries an understood operational failure when science, target
  class and budget remain in scope. Cancel and replace only jobs launched for
  the current task that are invalid, obsolete, or operationally broken.
- Copy remote results locally and validate the local copy before using it as
  evidence.
- Link every top-level result directory from the owning experiment/pilot note,
  including smoke, failed, recovery, partial and replacement directories. Keep
  remote-only locations and replacement relationships recoverable there.
- Use `collecting` while remote results are not yet fully present and validated
  locally. Use `ready-for-review` only after expected coverage is reconciled,
  included bundles validate, and all failures, exclusions, and superseded
  directories are identified. Use `partial`, `complete`, `failed`, or
  `superseded` when those more accurately describe the terminal handoff.
- Analyze outputs against declared completion criteria. Label partial
  evidence, confounds, and deviations explicitly.
- Record the current execution/review phase in the existing handoff. A completed
  process alone does not mean the evidence is collected, validated or reviewed.
- Record factual evidence and the Reviewer’s interpretation in the existing
  result/pilot note. Distinguish
  measurements from inference and state uncertainty/deviations. Do not withhold
  supported conclusions for another approval step.
- Treat `docs/current_state.md` as human-authored; edit it only with Filip's
  explicit permission.
