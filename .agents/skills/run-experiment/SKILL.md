---
name: run-experiment
description: Run, resume, monitor, recover, or collect repository simulations from exact configs or a concrete scientific plan. Covers training, declared sweeps, and diagnostic replays; excludes open-ended experiment design and interpretation-only requests.
---

# Run Experiment

Carry the assigned experiment through validated local results and a concise
handoff. Follow [AGENTS.md](../../../AGENTS.md), the task's scientific protocol,
the [execution workflow](../../../docs/experiment_workflow.md), and the
[reporting contract](../../../docs/experiment_reporting.md). Read nested
instructions before editing runners. Explicit user instructions take precedence
over stale documentation or global skills; use the actual experiment's protocol.

## Delegate by phase

Only the coordinator spawns phase workers. Workers execute their assigned
phase and return or escalate to the coordinator; they do not redelegate.
Start at the requested phase. During an assigned experiment, monitoring may
dispatch already-authorized pending cases; status-only or collection-only
requests do not authorize launches. Spawn with `model="gpt-6-astra"` and
explicit `reasoning_effort`:

| Phase | Effort | Responsibility |
|---|---|---|
| Prepare and launch | `high` | Prepare configs/runner, pass gates, launch, verify initial progress. |
| Monitor and collect | `medium` | Check progress/capacity, dispatch pending work through the coordinator, collect and validate results. |
| Diagnose and recover | `high` | Investigate failures and execute justified recovery. |

Use `fork_turns="none"` so effort overrides are accepted; supply the task,
current instructions, protocol links, and handoff explicitly. Leave coordinator
and global model settings unchanged. If delegation is unavailable, continue at
the available effort and disclose the limitation.

The coordinator tracks scope/budget, reconciles inputs and evidence alongside
the worker, and relays progress. Keep one execution owner per case. Before
handoff, the previous owner stops mutations and returns cases, source/config
identity, commands, target/job handles, local/remote paths, remaining
budget/deadline, and next checks. Retain these in the existing experiment
record; add no handoff schema. For existing runs, recover this context and
inspect live state before acting.

## Prepare and launch — High

- Resolve cases, completion criteria, evidence class, budget, and deadline.
  Record lean operational defaults when omitted; clarify missing scientific
  choices only when they materially change the task.
- Reuse a runner or implement a small direct runner for a specified plan.
  Preserve supplied science. Complete Conv configs with exact LR vectors use
  `python -m experiments.exact_run` and local `--smoke`, without extra tuning.
- Follow current resource, staging, smoke, scientific-gate, sharing, and
  overnight policies. Load relevant host skills. Record actual source,
  including relevant uncommitted changes. Add the persistent `planned` row
  before creating outputs; report cases, target, budget, duration, and paths
  before substantial compute.
- Use the existing launcher or a clear direct command. Apply the repo's current
  Jean Zay smoke/submission policy over older global canary requirements.
  Record handles and verify real artifact progress before handing off.

## Monitor and collect — Medium

- Apply `experiment-run-watchdog`, retaining repository reporting requirements.
  Verify initial health; check launcher, resources, logs, and artifact progress
  every 30 minutes against the recorded deadline.
- At each check, look for newly available RTX 3090 or RTX 5090 capacity on the
  authorized hosts. Dispatch eligible pending cases there in parallel instead
  of leaving them in a sequential queue. Recheck GPU model, free memory,
  process owners, and launchers at admission; apply the existing 5090 sharing
  policy and fit measured memory plus headroom without changing scientific
  settings. Preserve protocol-required placement of matched groups, remaining
  compute budget, and permitted execution windows.
- Have the coordinator assign High preparation/launch for the additional
  target while Medium continues monitoring existing runs. Dispatch only
  unstarted cases; claim them and update the original queue before submission
  to prevent duplicate starts. Record the new target and handles, verify
  initial progress, then include the runs in Medium monitoring. Proceed within
  the assigned scope without another approval round trip.
- Send unexpected failures and evidence to the coordinator for High recovery.
  Transfer mutation ownership of affected cases; keep observing unaffected
  cases. Preserve divergence and poor performance as scientific outcomes.
- Collect remote outputs locally. Validate each included bundle with
  `python -m experiments.reporting validate-run RUN_DIR`. Reconcile original
  coverage, failures, exclusions, retries, and replacements. A zero exit or
  valid subset does not establish completion. Maintain the directory row
  through collection and its appropriate terminal state under the reporting guide.

## Recover and finish

High recovery identifies the operational cause, preserves failed attempts,
and passes applicable checks before retrying within the original science and
remaining budget/deadline. Verify full-state compatibility before resuming.
Return verified replacement handles to Medium. Stop retries when the cause is
unresolved or budget/deadline is reached.

Keep the coordinator active through monitoring and collection. If paused,
blocked, or budget-limited, leave an accurate partial handoff and follow the
recorded policy for active jobs. Report coverage, headline measurements,
deviations, validation status, and local evidence paths. Continue into
interpretation when included in the user's task, without another approval gate.
