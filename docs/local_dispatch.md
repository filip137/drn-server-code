# Streamlined long-simulation dispatch

`python -m experiments.local_dispatch` is the canonical transport for an
already approved long simulation on the local `tmux main` lane or on Trex. It
turns one validated request into at most one detached launch and returns after
an immediate readback; it does not follow the simulation to completion.

This dispatcher is operational plumbing, not scientific authorization. It
does not replace the experiment catalog, approved frozen plan, tracker state,
execution smoke test, scientific `T/K` reference gate, or any protocol-specific
preflight. `start` fail-closes unless it can revalidate the approved plan, its
positive v2 preflight receipt, a fresh production tracker-gate receipt, the exact
Git commit, and a clean tracked worktree on the target.

## Request contract

The public commands are `build-preflight`, `verify-preflight`, `plan`,
`start`, and `status`. A request binds:

- an immutable, safe `attempt_id` and experiment ID;
- exactly one target, `main` or `trex`;
- the profile's canonical repository root as working directory and the exact
  argv frozen in the approved plan (never a shell command or extra suffix);
- a small allowlisted runtime override map on top of the profile's pinned base
  environment; Python/import/loader injection variables are rejected and no
  ambient host environment is inherited;
- the exact lowercase 40-character Git commit;
- an expected duration of at least 600 seconds and a hard deadline;
- the SHA-256 identity of the selected versioned dispatch profile;
- the approved plan, its plan-declared preflight receipt, and a fresh
  `current-experiments-launch-gate/v1` production receipt.

The plan-declared receipt used by this transport is
`server-code-long-run-preflight/v2`. It binds the approved-plan and profile
hashes, source commit, target/device, exact argv, pinned effective environment
and its hash, output root, and schema-validated receipts for the required
`smoke`, `tk_reference`, and `scheduled_run_preflight` checks. A check the
plan does not require is explicitly `not_required`; a self-asserted
`{"status":"passed"}` receipt is rejected.

The approved plan must contain exact `execution.preflight.receipt_schemas`
and `execution.preflight.receipt_bindings` role maps. Every required role has
an exact `{receipt_path, subject_id, producer_source_id}` binding. The receipt
path is repository-relative below the experiment's `results/` bundle. Smoke
and T/K bindings name the producer's filesystem-safe subject and exact
40-character source commit. The smoke producer commit must equal the
downstream launch commit; an upstream T/K study may deliberately bind an
older, different commit. Scheduled-run bindings use null subject and producer
source IDs. The CLI receipt path must equal the approved binding. Every
nonrequired role has a null schema and null binding. The initial checked-in
adapters support:

- `mnist-conv-perfectdiode-preflight-receipt/v1` as a payload smoke;
- `mnist-conv-perfectdiode-tk-selection/v1` as the scientific T/K gate;
- `scheduled-run-preflight-receipt/v1` as the unattended-runner gate.

A `main` dispatch consumes the perfect-diode producer's `akib`/`conv1`
smoke row; a `trex` dispatch consumes its `trex`/`conv2` row. This is an
explicit adapter mapping because the producer's host vocabulary is not the
dispatch target vocabulary. The T/K adapter re-derives the three-row
selection from hash-validated entry artifacts without taking the producer's
writer lock or publishing files.

A required unknown schema, a path or subject mismatch, or a legacy approved
plan without both maps fails before arming and must be amended and reapproved.
Static configuration, memory-only, and success-like generic receipts are not
accepted as payload evidence.

Use a new attempt ID for every changed request or authorized retry. There is no
automatic target fallback or retry.

## Build and verify the preflight envelope

After the experiment-specific tools produce the required inner receipts,
build the outer envelope through the canonical command. It validates every
inner receipt and its live artifacts, publishes the plan-declared envelope
first-write-wins, and has no GPU probe, tmux launch, SSH transaction, or
attempt-state side effect:

The generic example below assumes the approved plan requires the
perfect-diode Conv1/Conv2 hparam smoke, while `tk_reference` and
`scheduled_run_preflight` are both not required and therefore have null
schemas and bindings. The checked-in T/K adapter is for the separate Conv3
T/K study; do not attach it to this Conv1/Conv2 smoke merely because both
schemas are supported. Angle-bracket values below are schematic: replace them
with the exact approved values and expand the final launcher placeholder into
the plan's complete argv token sequence.

```bash
python -m experiments.local_dispatch build-preflight \
  --target main \
  --experiment-id '<experiment-id>' \
  --cwd /home/filip/server_code \
  --source-id "$GIT_SHA" \
  --approved-plan \
    '/home/filip/server_code/docs/experiment_plans/<experiment-id>.md' \
  --smoke-receipt \
    '/home/filip/server_code/results/<experiment-id>/preflight/smoke/receipt.json' \
  --env OMP_NUM_THREADS=1 \
  -- \
  '<exact executable from execution.launcher>' \
  '<exact remaining argv tokens from execution.launcher>'
```

Omit the T/K or scheduled receipt option only when the approved plan marks
that role not required and sets both its schema and binding to `null`. Confirm
the immutable result read-only with the same production binding:

```bash
python -m experiments.local_dispatch verify-preflight \
  --target main \
  --experiment-id '<experiment-id>' \
  --cwd /home/filip/server_code \
  --source-id "$GIT_SHA" \
  --approved-plan \
    '/home/filip/server_code/docs/experiment_plans/<experiment-id>.md' \
  --env OMP_NUM_THREADS=1 \
  -- \
  '<exact executable from execution.launcher>' \
  '<exact remaining argv tokens from execution.launcher>'
```

For Trex, build and verify the local evaluation snapshot after staging the
same immutable plan, receipts, artifacts, profile, and exact commit at the
same canonical paths on Trex. `start` independently reruns the same
schema-specific validators on Trex before touching its GPU or tmux.

When `scheduled_run_preflight` is required, create its receipt below the
experiment output root with the checked-in
`skills/scheduled-run-preflight/scripts/preflight_scheduled_runner.py`.
Dispatch accepts only that producer's exact command shape (`/bin/bash -euo
pipefail RUNNER --preflight` for `.sh`, the current Python interpreter plus
`RUNNER --preflight` for `.py`, or `RUNNER --preflight` for another executable),
the live runner hash and canonical cwd, a positive bounded timeout, and a
timezone-aware receipt no older than ten minutes. Extra arguments, copied
receipts outside the output bundle, stale timestamps, or changed runner bytes
fail before arming.

## Create the short-lived tracker gate

After the plan-declared smoke and scientific gates pass and the tracker is
`launch-ready`, create the production gate in a prelaunch directory outside
the dispatch attempt directory. The gate's `state_path` must be the future
dispatch receipt and its `output_root` must equal the plan's
`storage.local_bundle_path`.

```bash
python skills/run-experiment-pipeline/scripts/validate_current_experiments.py \
  docs/current_experiments.md \
  --require-experiment-id '<experiment-id>' \
  --require-launch-ready \
  --write-gate-receipt \
    'results/<experiment-id>/prelaunch/<attempt-id>/tracker-production-gate.json' \
  --gate-stage production \
  --gate-state \
    '/home/filip/server_code/simulation_results/local_dispatch/main/<attempt-id>/dispatch.json' \
  --gate-output-root \
    '/home/filip/server_code/results/<experiment-id>'
```

Use `trex` instead of `main` in `--gate-state` for a Trex request. The gate is
valid for ten minutes. For Trex, stage the immutable approved plan, manifest,
preflight receipt, production-gate receipt, and exact committed source before
`start`; the dispatcher trusts the short-lived canonical gate receipt and does
not treat a copied tracker Markdown file as live authority.

## Plan with zero launch side effects

Planning validates and prints the canonical request and its hash. It does not
probe a GPU, contact Trex, create a tmux window, or write dispatch state.

```bash
python -m experiments.local_dispatch plan \
  --target main \
  --attempt-id '<attempt-id>' \
  --experiment-id '<experiment-id>' \
  --cwd /home/filip/server_code \
  --source-id "$GIT_SHA" \
  --approved-plan \
    '/home/filip/server_code/docs/experiment_plans/<experiment-id>.md' \
  --tracker-gate-receipt \
    '/home/filip/server_code/results/<experiment-id>/prelaunch/<attempt-id>/tracker-production-gate.json' \
  --expected-duration-seconds 7200 \
  --hard-deadline-seconds 9000 \
  --env OMP_NUM_THREADS=1 \
  -- \
  '<exact executable from execution.launcher>' \
  '<exact remaining argv tokens from execution.launcher>'
```

Review the printed request before arming the launch. Repeat the same arguments
with `start` only after the launch gates are valid:

```bash
python -m experiments.local_dispatch start \
  --target main \
  --attempt-id '<attempt-id>' \
  --experiment-id '<experiment-id>' \
  --cwd /home/filip/server_code \
  --source-id "$GIT_SHA" \
  --approved-plan \
    '/home/filip/server_code/docs/experiment_plans/<experiment-id>.md' \
  --tracker-gate-receipt \
    '/home/filip/server_code/results/<experiment-id>/prelaunch/<attempt-id>/tracker-production-gate.json' \
  --expected-duration-seconds 7200 \
  --hard-deadline-seconds 9000 \
  --env OMP_NUM_THREADS=1 \
  -- \
  '<exact executable from execution.launcher>' \
  '<exact remaining argv tokens from execution.launcher>'
```

For Trex, change `--target` and the gate's state binding to `trex`; every
request path must be valid in the canonical `/home/filip/server_code`
checkout on Trex.

The plan's `execution.launcher` must contain that complete argv, including
stage, config/manifest, result paths, and every infrastructure flag. `start`
does not allow extra arguments after it. An approved literal `python`
resolves only to the selected profile's pinned `python_path`; an approved
absolute executable must match byte-for-byte. Another executable with the
same basename is rejected even when it is under an allowed root. Request
environment names are limited to `CUDA_VISIBLE_DEVICES`, standard
OMP/MKL/OpenBLAS/NumExpr thread counts, the two KMP shared-memory flags, and
`PYTHONUNBUFFERED`.

## Launch behavior

The checked-in profiles are
[`configs/dispatch/main.json`](../configs/dispatch/main.json) and
[`configs/dispatch/trex.json`](../configs/dispatch/trex.json).

For `main`, `start` takes a bounded target lock, revalidates launch authority,
checks the configured GPU and unresolved target leases, and creates a new
named window in the existing `main` session with `tmux new-window`. It never
uses `send-keys`, kills a pane, or replaces an existing window.

Before the GPU or tmux side effect, target validation requires the working
directory to equal the profile's Git top-level, `HEAD` to equal `source_id`,
no tracked changes or untracked files, and the plan, helper, profile, and
canonical validators to be tracked. Bound receipt paths may not traverse
symlinked parents. The helper and payload are launched through
`/usr/bin/env -i`; the profile-pinned base environment plus approved overrides
is exactly the environment hashed by preflight and passed to the payload.

For Trex, the caller makes exactly one bounded SSH call with `BatchMode=yes`
and a connection timeout. The canonical JSON request is sent on standard
input to the checked-in remote dispatcher. Trex then performs the same
authority checks, lock, lease and GPU probe, and creates a fresh
attempt-named detached tmux session. The simulation argv is read from the
immutable request receipt by the worker and is executed without a shell; it
is not reconstructed inside SSH or tmux.

Both paths write first-write-wins request, dispatch, completion, or failure
receipts below:

```text
/home/filip/server_code/simulation_results/local_dispatch/TARGET/ATTEMPT_ID/
```

An identical second `start` returns the existing dispatch or completion
receipt. Reusing an attempt ID for different content, trying to resume a
terminal failure, or finding only a request after an interrupted dispatch is
rejected. Every armed target-side failure gets one immutable terminal report.
A busy GPU causes no tmux launch. If an SSH or tmux result is ambiguous, the
reported launched-job count is `null`; the dispatcher never retries or falls
back to another lane.

The worker is a Linux child subreaper. It tracks and cleans process-group and
detached descendants at the hard deadline, on `SIGTERM`/`SIGHUP`/`SIGINT`,
and before publishing any normal completion or failure. Cleanup repeatedly
rescans for children created during termination and publishes an immutable
quiescence receipt only after none remain. A payload whose parent exits while
descendants remain is cleaned and failed rather than reported complete.
Prelaunch failures with a known launch count of zero release their lease;
ambiguous or known-launched controller failures keep the target reserved until
a worker quiescence or completion receipt proves that no prior payload remains.

## Bounded read-only status

Status reads the immutable receipts and performs no launch, retry, kill, or
repair:

```bash
python -m experiments.local_dispatch status \
  --target main \
  --attempt-id '<attempt-id>'

python -m experiments.local_dispatch status \
  --target trex \
  --attempt-id '<attempt-id>'
```

The command is bounded; Trex status is exactly one read-only SSH transaction.
It reports `not_found`, `initializing`, `running`, `stale_running`,
`dispatched`, `complete`, or the attempt's terminal failure receipt.
`stale_running` means the immutable worker PID/start-time identity is no
longer live; it is a read-only warning and conservatively keeps the lane
reserved until quiescence is proven. A failed one-shot status query is
nonterminal because it does not change the attempt. Persistent monitoring of a
running long simulation must use an explicitly armed, hard-deadline controller
and is then subject to the fail-stop-report policy.

The fail-stop behavior begins only when `start` prints
`LONG-RUN ATTEMPT ARMED`. Missing files found by `plan`, code/test failures,
environment setup, and bounded developer probes remain ordinary repairable
work; they do not terminally fail a simulation attempt.
