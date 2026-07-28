---
name: run-experiment-pipeline
description: Define, approve, preflight, launch, track, repatriate, validate, and publish an exact research experiment or sweep. Use when Codex is asked to plan, run, schedule, continue, monitor, or report an experiment and must bind the hypothesis, ordered sweep values, controls, decision rule, artifact paths, live progress entry, and final comparison-card result.
---

# Run Experiment Pipeline

Use one durable plan, one live tracker, one raw bundle, and one reviewed result
destination:

| Need | Canonical location |
|---|---|
| Why and exactly what to run | `docs/experiment_plans/<experiment-id>.md` |
| What is happening now | `docs/current_experiments.md` |
| Raw immutable evidence | `results/.../<run-or-study-id>/` |
| Interpretable final result | comparison card in `docs/results/index.md` |

Numeric measurements come from immutable artifacts and
`result_registry/cards/`. Interpretation comes from
`result_registry/reviews/`. Do not create a parallel result format.

## User request format

Accept a request in this compact form:

```text
Use $run-experiment-pipeline.
Hypothesis: ...
Sweep: axis=[ordered values], cases=[...], seeds=[...].
Primary result / decision rule: ...
Evidence scope: diagnostic | paper-facing | historical replay.
Storage or compute preference (optional): ...
```

Resolve standard storage and reporting locations from this skill. Ask only
for missing scientific choices that would change the run or interpretation.

## 1. Resolve the scientific request

Read the root and nested `AGENTS.md` files, then the active protocol index,
active protocols, and curation rules for the requested evidence scope.

Require these user-owned scientific choices before launch:

- falsifiable hypothesis;
- control, treatments, and expected direction;
- ordered sweep values, cases, seeds, and any conditional expansion rule and
  hard cap;
- primary metric, data split, checkpoint role, seed aggregation, comparison,
  and minimum meaningful effect;
- completion, exclusion, failure, and inconclusive rules;
- every protocol choice that is still unresolved.

The agent may derive infrastructure details such as content IDs, expanded job
count, exact paths, host allocation, and validation commands. Do not invent a
scientific choice. Stop for user direction when a missing choice would change
the interpretation.

## 2. Create and freeze the experiment plan

Copy `assets/experiment-plan-template.md` to
`docs/experiment_plans/<experiment-id>.md` and fill its single JSON contract.
The plan is the readable, user-approved study design; the canonical config and
immutable expanded manifest remain execution authority.

Keep all scientific values in the config. List the ordered axes in the plan
for review, bind the plan to the config and manifest SHA-256 values, and
verify the manifest job count through its declared JSON pointer. State that
every undeclared field is fixed by the linked protocol/config.

Before approval:

1. run the canonical config validator and plan-only/manifest command;
2. assign the exact local result bundle;
3. assign every remote staging path and its intended local destination;
4. name the planned comparison cards, review records, and final index anchors;
5. confirm that the result registry already supports the required card schema;
6. show the user a compact hypothesis, sweep, run-count, storage, and
   reporting summary.

Record explicit approval in the plan. Then run:

```bash
python skills/run-experiment-pipeline/scripts/validate_experiment_plan.py \
  docs/experiment_plans/<experiment-id>.md \
  --require-approved \
  --verify-files
```

Do not launch an unapproved or invalid plan. After launch, treat the plan and
scientific config as frozen. A scientific amendment gets a new experiment ID
or an explicit successor plan; it also invalidates prior preflight.

## 3. Add the live tracker entry

Once an experiment is explicitly planned and has a stable ID, add exactly one
entry to `docs/current_experiments.md`. Keep the visible contract to three
fields:

```markdown
<!-- experiment-id: <experiment-id> -->
### <Human-readable title>

- **Testing:** <one-sentence hypothesis or controlled comparison, with the plan or protocol link>
- **Where:** <host and job/tmux identity, or unassigned; remote/local bundle link when known>
- **Status:** `<state>` — <concise verified progress, timestamp, and next action or blocker>
```

The HTML marker is the stable machine identity and is invisible when rendered.
Do not copy manifests, hashes, metrics, receipts, or interpretation into the
tracker; keep those in the plan, result bundle, and result registry.

Use one of these states:

`planned`, `preflighting`, `launch-ready`, `queued`, `running`,
`remote-closed`, `transferring`, `validating`, `review-pending`, or `blocked`.

Set `planned` while the plan or implementation is still being prepared and
`preflighting` before the first payload smoke. Never infer `queued` or
`running` from an old log.

Validate the whole tracker after every edit:

```bash
python skills/run-experiment-pipeline/scripts/validate_current_experiments.py \
  docs/current_experiments.md
```

## 4. Preflight and launch

Run the real end-to-end smoke and every scientific reference gate required by
root and experiment guidance. A plan-only command is not a payload smoke test.
For Conv amplification, run the active `T/K` reference gate for every unique
resolved architecture × nonlinearity × amplification configuration.

The approved plan must bind every required gate to an exact supported receipt
schema in `execution.preflight.receipt_schemas` and an exact
`{receipt_path, subject_id, producer_source_id}` entry in
`execution.preflight.receipt_bindings`. Smoke and T/K subject IDs identify
their producer studies and need not equal the downstream experiment ID;
the producer source ID pins the exact 40-character Git commit that created the
receipt. Smoke uses the downstream launch commit, while an upstream T/K study
may use an older explicitly approved commit. Scheduled-run subject and
producer source IDs are null. Both schema and binding are null for a role the
plan does not require. Receipt paths are repository-relative below the
experiment result bundle and must exactly match the inner receipts passed to
the dispatcher. Amend and reapprove a legacy plan rather than inferring these
values. For a local/Trex long simulation, create the outer
`server-code-long-run-preflight/v2` receipt only through the
`python -m experiments.local_dispatch build-preflight` command, then confirm
it with the read-only `verify-preflight` command. Do not hand-author the
envelope or substitute a generic success-like JSON receipt.

If execution is scheduled, delayed, unattended, remote-batch, or Slurm-based,
read and apply `../scheduled-run-preflight/SKILL.md`. Store its receipt and
runner hash with the result bundle and tracker entry. Do not duplicate or
weaken that skill's scheduler checks.

For a remote payload canary, keep the tracker `preflighting` and create one
new canonical gate receipt immediately before staging/submission:

```bash
python skills/run-experiment-pipeline/scripts/validate_current_experiments.py \
  docs/current_experiments.md \
  --require-experiment-id <experiment-id> \
  --write-gate-receipt <new-attempt-path>/tracker-canary-gate.json \
  --gate-stage canary \
  --gate-state <new-attempt-path>/supervisor-state.json \
  --gate-output-root <remote-result-root>
sha256sum <new-attempt-path>/tracker-canary-gate.json
```

Pass both that receipt and its SHA-256 to the remote supervisor. The receipt
is a short-lived, exact-state/output bearer snapshot; a staged copy of
`docs/current_experiments.md` is not live authority. If the canonical tracker
changes after receipt creation, stop the attempt and do not use that receipt.

After all required gates pass, change the exact entry to `launch-ready` and
let the canary supervisor exit at its `stage_complete` boundary. Run this
fail-closed gate immediately before production:

```bash
python skills/run-experiment-pipeline/scripts/validate_current_experiments.py \
  docs/current_experiments.md \
  --require-experiment-id <experiment-id> \
  --require-launch-ready \
  --write-gate-receipt <new-stage-path>/tracker-production-gate.json \
  --gate-stage production \
  --gate-state <new-attempt-path>/supervisor-state.json \
  --gate-output-root <remote-result-root>
sha256sum <new-stage-path>/tracker-production-gate.json
```

Resume the same terminal-free supervisor state in a new process with the
distinct production receipt/hash. Never make one process cross the canary
boundary, reuse the canary receipt for production, or overwrite either
receipt. Launch only the immutable manifest through the canonical public launcher.
No launcher or submission controller may perform its first production-launch
side effect unless this exact-ID gate passes and the short-lived receipt/hash
remain valid. Use the repository compute-allocation rules. After a live
scheduler or process readback, change the same entry to `queued` or `running`
and update `Where`; do not create a second entry.

## 5. Monitor, collect, and repatriate

Update only `docs/current_experiments.md` for verified transitions. Use
`remote-closed` when execution has ended but the remote bundle has not yet
passed local transfer and validation, then `transferring`, `validating`, and
`review-pending` as applicable. Use `blocked` at any phase when progress
requires intervention. Keep detailed logs beside the result bundle.

For remote work, read and apply `../sync-remote-results/SKILL.md` and
`docs/remote_result_workflow.md`; for Jean Zay also follow
`docs/jean-zay.md`. Remote storage is staging. Do not mark the experiment
complete until the complete bundle is locally copied, validated, and linked.
Keep failed, partial, excluded, and missing coverage explicit.

Run the canonical collector and integrity validator. Never turn process
completion into scientific acceptance automatically.

When the user explicitly requests automatic reconciliation, the repository
provides a conservative hourly fallback for environments without the Codex
automation service:

```bash
bash skills/run-experiment-pipeline/scripts/run_hourly_current_experiments_refresh.sh \
  --check
tmux new-session -d -s current-experiments-refresh \
  "bash '$PWD/skills/run-experiment-pipeline/scripts/run_hourly_current_experiments_refresh.sh' --loop"
```

It runs an ephemeral, tracker-only audit under a noninteractive
no-approval policy, serializes passes with `flock`, and leaves the tracker
byte-identical when no verified transition exists. Its prompt prohibits
launch, cancellation, restart, transfer, deletion, publication, and edits
outside `docs/current_experiments.md`. Logs live under
`results/_tracker_refresh/hourly/`.

## 6. Publish the comparison result

Use the existing comparison-card model described in
`result_registry/README.md`:

1. add qualified measurements to `result_registry/cards/`;
2. add the comparison-level observation, interpretation, decision, and
   limitations to `result_registry/reviews/`;
3. keep evidence scope ahead of accuracy and never mix incompatible protocols
   in one comparison card;
4. qualify every accuracy by split, checkpoint role, epoch budget, seed
   coverage, and official-test use;
5. render and check `docs/results/index.md`.

The final user-facing answer must link the comparison card, the canonical
local result bundle, and the plan. Leave the tracker entry in
`review-pending` until the comparison is reviewed; then remove it from the
current tracker because `docs/results/index.md` is the durable history.

## Failure rules

- Fail closed on missing approval, unknown sweep values, config/manifest drift,
  failed smoke, failed `T/K` gate, missing artifacts, or unsupported result
  schema.
- Fail closed before a production launch when the tracker is missing,
  malformed, duplicated, names a different experiment ID, or is not
  `launch-ready`.
- Never silently expand an adaptive sweep beyond its approved rule and cap.
- Never substitute datasets, protocols, hosts, configs, devices, or output
  roots after a gate fails.
- A material code, config, environment, initializer, architecture,
  nonlinearity, gain, or `T/K` change returns the plan to preflight required.
- Before final launch staging, explicitly declare the long-run attempt armed
  as required by the root policy. Planning, implementation, unit tests,
  environment setup, plan-only commands, and bounded developer probes before
  that declaration may be diagnosed, corrected, and retested normally.
- After an armed long-run attempt begins, its first unexpected staging,
  final gate-revalidation, submission, readback, or active-monitoring failure
  is terminal for that attempt and launch turn. Short smoke/scientific gate
  execution completed before `start` remains ordinary bounded preflight work
  unless that gate is itself a long or scheduled simulation. Preserve
  evidence, perform only bounded read-only inspection, issue the root-policy
  failure report, and wait for a new user message.
- A later authorized long-run attempt must use a new immutable attempt ID and
  receipt paths. Retry mechanics written in a plan do not authorize the agent
  to initiate that retry.
- Long-run supervisors must write one first-write-wins structured failure
  report and exit nonzero. Failed states are terminal, and polling or
  reconciliation loops must be explicitly requested and bounded by a hard
  deadline.
