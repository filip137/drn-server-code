---
name: scheduled-run-preflight
description: Validate unattended commands before they are queued and fail closed on scheduler, CLI, environment, or payload errors. Use when Codex creates or changes an at job, cron entry, systemd timer, Codex automation, delayed tmux command, Slurm submission, remote batch launcher, or any other scheduled run.
---

# Scheduled Run Preflight

Validate both the scheduling mechanics and the smallest safe payload before
queueing unattended work. Never use the full scheduled execution itself as the
first test.

## Required runner contract

Implement the scheduled entrypoint as a versioned runner with:

- a side-effect-free `--preflight` mode;
- `set -euo pipefail` for shell runners;
- one frozen executable/argument construction path for real execution;
- absolute paths for the working directory, executable, prompt/config, logs,
  and required inputs;
- nonzero exit status on every failed prerequisite;
- no silent fallback to a different command, environment, config, device, or
  output directory.

For non-experiment shell runners, build the target invocation once as an
array and reuse its parseable prefix in both branches:

```bash
set -euo pipefail

target=(/absolute/path/to/tool GLOBAL_OPTIONS subcommand SUBCOMMAND_OPTIONS)

if [[ "${1:-}" == "--preflight" ]]; then
    "${target[@]}" --help >/dev/null
    # Validate inputs, environment, permissions, and safe payload smoke here.
    exit 0
fi

"${target[@]}" ACTUAL_INPUTS
```

Do not duplicate the target tokens in two handwritten commands. That would
allow preflight to pass while the scheduled command still contains different
option placement or spelling.

For Jean Zay runners, use one shared environment bootstrap in login-node
preflight and the Slurm wrapper. It must initialize the non-interactive shell,
run the required `idrenv`, and load the reviewed module.
It must resolve Python only after module loading and verify the resulting
Python/PyTorch/CUDA identity. Do not construct an alternate Python path in an
ad-hoc SSH command.
Production defaults and remote-applicable tests must not reference
workstation-only locations such as a user's `.codex` directory; inject or
stage those tools through the environment contract.

For remote experiment runners, keep login-node preflight import-only. After
the shared environment bootstrap, start one Python process that imports the
entrypoint and the real model/runtime dependency chain. Do not start a second
interpreter merely for `--help`, load a dataset, construct output artifacts,
request a GPU, or execute a numerical step on the login node. Dataset, device,
output, and numerical compatibility belong to the scheduled bounded payload
canary.

## Local launch-admission budget

Before a scheduled experiment submission, run exactly three focused local
runner checks:

1. shell or entrypoint syntax is valid;
2. the frozen target, resources, run class, case set, and hard job cap are
   exact, with no unauthorized production or expansion path; and
3. remote preflight is one import-only process and execution has one frozen
   runtime/argument construction path.

These are launch-focused checks, not a repository regression suite. On a
passing path, do not run the catalog suite, common smoke suite, production
adapter suite, legacy supervisor suite, or unrelated experiment tests.
Request, manifest, catalog-route, tracker, and source-hash validators remain
separate deterministic validations when applicable; do not inflate the local
test count by relabeling them as tests.

Broader tests are diagnostic escalation only. If one of the three focused
checks, the scheduled-run preflight, or the payload canary fails before
arming, diagnose the failure and run only the smallest relevant regression
slice. After an armed failure, first stop and report; broader diagnostics wait
for a new user message. Never run every available suite merely because one
focused check failed.

## Gate 1: command and scheduler preflight

Use a new receipt path for each attempt, then run:

```bash
python skills/scheduled-run-preflight/scripts/preflight_scheduled_runner.py \
  --runner /absolute/path/to/runner.sh \
  --cwd /absolute/path/to/workdir \
  --receipt /absolute/path/to/preflight-receipt.json
```

The checker:

- validates shell syntax when applicable;
- invokes shell preflight with forced `errexit`, `nounset`, and `pipefail`,
  without a shell-expanded command string;
- requires exit code zero within the timeout;
- verifies the runner did not change during preflight;
- records the runner SHA-256, working directory, command, timestamp, and
  captured output in a receipt.

Do not queue anything if the checker fails or produces no receipt with
`status: passed`.

The stable Jean Zay validation fast path implements this gate inside
`experiments.jeanzay_validation`: it invokes the staged stable runner once in
import-only mode, writes the captured JSON receipt, verifies the one package
hash, and then runs `sbatch --test-only`. Do not wrap that path in a second
`preflight_scheduled_runner.py` invocation or repeat the import probe during
final revalidation.

For a Codex CLI task, place global flags before the subcommand and make the
shared preflight array parse the same prefix, for example:

```bash
target=(/home/USER/.local/bin/codex -a never exec -C /absolute/repo -s workspace-write)
```

This exact-prefix check rejects a runner that incorrectly places `-a never`
after `exec`.

## Gate 2: payload smoke test

If the scheduled payload launches code or experiments, run the smallest
representative test through the same entrypoint before queueing the full job.
Verify:

- imports and strict config validation;
- source/config/checkpoint identities;
- output-directory creation and write permissions;
- required dataset or input availability without substituting another source;
- device allocation and one minimal successful compute step when a GPU is
  required;
- expected completion marker, checkpoint, or other minimal output;
- cleanup or isolation of smoke outputs from scientific results.

A CLI parse, syntax check, plan/manifest generation, CPU-only test for a
GPU-only path, or unrelated unit test does not replace this payload smoke
test. Record the smoke command, exit code, relevant hashes, and output path in
or beside the preflight receipt.

## Queue and verify

Only after both gates pass:

1. Recompute the runner SHA-256 and require it to equal the receipt.
2. Queue the job using the tested runner path, not an inline reconstruction.
3. Read the scheduler's stored entry back.
4. Verify the exact execution time, timezone, working directory, runner path,
   and arguments.
5. Record the scheduler job ID and queued runner SHA-256.
6. Make the runner verify its own expected SHA-256 at execution time when the
   scheduler permits it.

If the runner, config, prompt, environment contract, or payload changes,
the queued job must eventually be removed or superseded and both gates must
be repeated. If the change or discovery occurs after the long-run attempt is
armed, first stop and report; cancellation, repair, and a successor attempt
require a new user message. Never claim a run is ready merely because the
scheduler accepted it.

## Launch deadlines and failure reporting

For an approved immediate remote Slurm long-running simulation, start a
durable launch-attempt clock at user approval or at the first
launch-controller invocation, whichever occurs first. Preserve that original
timestamp across controller restarts. Starting this clock does not arm the
attempt: terminal stop-and-wait behavior applies only after the explicit
`LONG-RUN ATTEMPT ARMED` declaration. A deadline failure before that
declaration, and every failure for a scheduled payload that is not a
simulation, remains ordinary bounded, repairable work under the rules below.

- Default the approval/preparation-to-canary-submission deadline to at most
  1,200 seconds.
- Default the `sbatch`-acceptance-to-first-running-task deadline to at most
  900 seconds.
- Treat a scheduled future launch separately: its approved schedule must bind
  the later clock start explicitly rather than silently disabling deadlines.
- If the preparation deadline expires, perform no `sbatch` side effect.
- If the scheduler-start deadline expires while the exact job is still
  pending, record that numeric job ID and its live state, but do not cancel it
  in the same turn. Report the terminal armed-attempt failure and wait for a
  new user message before any cancellation or successor attempt.
- During armed polling or reconciliation, emit observable progress at least
  every 60 seconds and stop at the applicable hard deadline.
- Exit nonzero and publish one first-write-wins durable JSON failure report
  containing the stage, limit, elapsed time, job ID when present, last
  scheduler state, scheduler action (explicitly `not_attempted` when none),
  and the exact launch-attempt receipt.
- Emit one machine-readable failure marker naming that report so monitoring
  does not have to infer failure from prose or an hourly tracker refresh.

Deadline values and any explicit scheduled-start exception are part of the
reviewed launch contract. Changing them invalidates the earlier preflight.

## Failure handling

- Fail before queueing on any preflight or smoke-test error.
- Preserve any failed receipt, the failing command, and all diagnostic logs.
- State that no job was queued, or identify the exact queued job and live
  state. After an armed failure, removal requires a new user message.
- For an explicitly armed long-running simulation launch, treat the first
  unexpected final revalidation, staging, queueing, readback, or monitoring
  failure as terminal for that attempt and launch turn. After bounded
  read-only inspection, report the exact stage and error, completed work,
  launched jobs and states, last valid artifacts, and the smallest proposed
  next action, then wait.
- Short preflight and smoke execution completed before `start` remains
  ordinary repairable work, as do setup and validation for scheduled payloads
  that are not simulations. Diagnose, correct, and retest with bounded work;
  never loosen or bypass a failing gate. A preflight or smoke that is itself a
  long or scheduled simulation must be armed as its own long-run attempt and
  then uses the terminal rule above.
- A later authorized long-run attempt must use new immutable attempt and
  receipt paths.
