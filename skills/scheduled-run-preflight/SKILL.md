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
- one shared executable/argument construction path for preflight and real
  execution;
- absolute paths for the working directory, executable, prompt/config, logs,
  and required inputs;
- nonzero exit status on every failed prerequisite;
- no silent fallback to a different command, environment, config, device, or
  output directory.

For shell runners, build the target invocation once as an array. Reuse that
array in both branches:

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
remove or supersede the queued job and repeat both gates. Never claim a run is
ready merely because the scheduler accepted it.

## Failure handling

- Fail before queueing on any preflight or smoke-test error.
- Preserve the failing command and logs.
- State that no job was queued, or identify and remove a stale queued job when
  removal is authorized.
- Correct the root cause and repeat the full procedure. Do not bypass the
  failing check with a looser command.
