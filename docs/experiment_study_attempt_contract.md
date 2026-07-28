# Scientific Study and Execution Attempt Contracts

Date: 2026-07-28
Implementation status: simplified continuation contract is active for fixed
ordinary diagnostics; strict native contracts remain available for novel,
adaptive, paper-facing, and reproducibility-critical studies

## Decision

There are two deliberately scoped paths:

| Path | Scientific authority | Execution authority |
|---|---|---|
| Simplified fixed continuation | generated `experiment-resolved-study/v1`, approved from the user-facing request summary | generated `functional-dispatch-request/v1` plus matching passed `functional-preflight-result/v1` |
| Full novel/adaptive/paper-facing study | `experiment-study/v1` | `experiment-execution-attempt/v1` |

The simplified path is implemented by
[`experiment_request.py`](../experiments/experiment_request.py),
[`experiment_study.py`](../experiments/experiment_study.py),
[`experiment_flow.py`](../experiments/experiment_flow.py),
[`experiment_executor.py`](../experiments/experiment_executor.py), and
[`functional_dispatch.py`](../experiments/functional_dispatch.py). It
generates three bounded validation requests for the local/Trex/Akib fixture
and cannot generate production requests until every target contributes a
passing real one-batch smoke.

The full-study control plane uses two independent JSON contracts:

| Contract | Owns | User action that authorizes it |
|---|---|---|
| `experiment-study/v1` | Scientific question, controls, treatments, ordered design, decision rule, scientific config/manifest identities, required scientific gates, completion, and reporting intent | Scientific approval |
| `experiment-execution-attempt/v1` | Source release, target, device, launcher, routed manifest, environment, preflight receipt bindings, output paths, run class, and deadlines | Attempt authorization |

A study is not an execution request. An attempt is not allowed to redefine
the study. Both validate independently.

The implementation lives in
[`experiments/control_plane/`](../experiments/control_plane/). Native
authoring templates live at:

- [`experiment-study-template-v1.json`](../skills/run-experiment-pipeline/assets/experiment-study-template-v1.json)
- [`execution-attempt-template-v1.json`](../skills/run-experiment-pipeline/assets/execution-attempt-template-v1.json)

## Scientific boundary

Only the `study` object is hashed by `study_spec_sha256`. Approval records that
hash in `approval.approved_study_sha256`.

The study object contains:

- the hypothesis and falsification outcomes;
- metric, split, checkpoint role, aggregation, comparison, and effect
  threshold;
- the ordered axes, cases, seeds, expansion rule, and hard job cap;
- the hashes of a **scientific-only config** and **study manifest**;
- identities and hashes of scientific protocol authorities;
- required scientific gates such as `tk_reference`;
- completion, exclusion, failure, and review rules; and
- planned comparison identities and card schemas.

The scientific config and study manifest must not contain a host, queue,
interpreter, tmux session, attempt ID, output path, receipt path, or deadline.
Those belong to an execution attempt. The legacy converter can verify old
mixed artifacts, but marks their successor study for review rather than
asserting that an old config already satisfies this clean boundary.

File locations are held in the separate `bindings` object. Its canonical hash
is `study_bindings_sha256`. This gives the two hashes different meanings:

```text
study_spec_sha256       scientific bytes and interpretation
study_bindings_sha256   current repository locations of those bytes
```

Moving an unchanged scientific config, manifest, protocol, or reporting
destination leaves scientific approval valid. A new execution attempt is
still required because each attempt binds both hashes.

Approval states are:

- `draft`: scientific review has not happened;
- `review_required`: deterministic legacy extraction succeeded, but old
  approval was not transferred;
- `approved`: the named approver and timestamp bind the exact scientific
  study hash.

## Execution boundary

The `attempt` object is hashed independently by `attempt_spec_sha256`.
Authorization records that hash in
`authorization.authorized_attempt_sha256`.

An attempt binds:

- study ID, study path, scientific hash, and binding hash;
- the Git commit and a file-verified control-plane release identity;
- one target profile, host, device, and target kind;
- exact launcher, collector, and validator argv;
- a routed manifest tied back to the study-manifest hash;
- the environment-contract path and hash;
- distinct inner preflight receipts and an outer receipt;
- a required smoke gate, every study-required scientific gate, and a
  scheduled-run gate for remote targets;
- an attempt-isolated result root and remote staging destinations; and
- an explicit `validation` or `production` duration and hard deadline.

`validation` is bounded below ten minutes: expected duration must be below
600 seconds and its deadline cannot exceed 600 seconds. `production` starts
at 600 seconds and remains subject to the repository's explicit
`LONG-RUN ATTEMPT ARMED` and fail-stop-report policy.

The attempt validator verifies the study and, when requested, the scientific
bindings, target profile, control-plane release, routed manifest, and
environment bytes. It does not execute a command or create a job.

### Multi-attempt study routing

One attempt may own every study job or an explicit proper subset. The routed
manifest records:

- `selection.mode=all|subset`;
- the manifest's job-ID field; and
- the ordered `selected_study_job_ids`.

For a study split across hosts, validating each attempt independently is not
enough. The attempt-set validator requires every scientific job exactly once
across the authorized attempts and rejects omissions, overlaps, unknown IDs,
mixed study hashes, or inconsistent job-ID fields:

```bash
python -m experiments.control_plane validate-attempt-set \
  docs/experiment_studies/<study-id>.json \
  docs/execution_attempts/<study-id>/<attempt-a>.json \
  docs/execution_attempts/<study-id>/<attempt-b>.json \
  --repo-root . --verify-files --require-authorized --json
```

Host assignment and concurrency remain operational. Moving a job between
attempts changes attempt authorization but not the study hash.

## Change semantics

| Change | Scientific reapproval | New attempt authorization |
|---|---:|---:|
| Hypothesis, dataset, model, seed, sweep value, `T/K`, metric, or decision rule | Yes | Yes |
| Scientific config or study-manifest bytes | Yes | Yes |
| Move unchanged scientific files | No | Yes |
| Host, GPU, queue, interpreter, environment, launcher, or collector | No | Yes |
| Output, staging, receipt, attempt ID, or deadline | No | Yes |
| Failed preflight receipt | No, unless the repair is scientific | A successor attempt only when its bound attempt fields change |

This removes the former cascade in which a scheduler or environment repair
looked like a scientific amendment.

## Legacy split

The converter accepts exactly one fenced `experiment-run-plan/v1` JSON object
and produces:

1. an `experiment-study/v1` document; and
2. an `experiment-legacy-execution-residue/v1` document with
   `status=not_launch_authority`.

It copies hypothesis, sweep, completion, and reporting fields without
inventing values. Only `docs/...` entries in
`undeclared_fields_fixed_by` become hashed scientific protocol authorities.
Primary config and manifest identities remain scientific artifacts.
Environment files, auxiliary configs, launchers, storage, old preflight
fields, tracker paths, and legacy approval remain in the residue.

An old `approved` plan becomes `review_required`. An old draft stays `draft`.
The converter never creates an approved study, an authorized attempt, or a
job.

Dry-run all current plans:

```bash
/home/filip/miniconda3/envs/py312/bin/python \
  -m experiments.control_plane check-legacy-corpus \
  --repo-root . --json
```

Publish one first-write-wins split:

```bash
/home/filip/miniconda3/envs/py312/bin/python \
  -m experiments.control_plane split-legacy-plan \
  docs/experiment_plans/<experiment-id>.md \
  --repo-root . \
  --study-output docs/experiment_studies/<experiment-id>.json \
  --residue-output docs/experiment_attempt_residue/<experiment-id>.json \
  --json
```

## Native authoring and validation

Create and validate the scientific-only config and study manifest first.
Then fill the study template and run:

```bash
python -m experiments.control_plane validate-study \
  docs/experiment_studies/<study-id>.json \
  --repo-root . --verify-bindings --json
```

The output reports the canonical study hash. After explicit scientific review,
record the approver, timestamp, and exact hash, then require approval:

```bash
python -m experiments.control_plane validate-study \
  docs/experiment_studies/<study-id>.json \
  --repo-root . --verify-bindings --require-approved --json
```

Create a new attempt from the attempt template. Its result path must include
the exact attempt ID as a complete path component. After separate
authorization, run:

```bash
python -m experiments.control_plane validate-attempt \
  docs/execution_attempts/<study-id>/<attempt-id>.json \
  --repo-root . --verify-files --require-authorized --json
```

When more than one attempt implements the study, also run
`validate-attempt-set` over the complete intended attempt set before any
preflight or launch.

These commands are read-only. Approval and authorization remain explicit
document edits; validation does not grant either.

## Cutover boundary

The simplified continuation path is active now for requests that meet every
fast-path condition in
[`run-experiment-pipeline/SKILL.md`](../skills/run-experiment-pipeline/SKILL.md).
It does not wait for conversion of unrelated legacy plans, a clean whole-repo
worktree, exact interpreter aliases, or exact environment versions. It keeps
scientific inputs exact and uses one target-local numerical smoke as the
functional admission test.

The old `experiment-run-plan/v1` path and the strict native
`experiment-study/v1`/`experiment-execution-attempt/v1` contracts remain the
full lifecycle for novel science, adaptive sweeps, paper-facing evidence, and
explicit reproduction work. Their corpus migration status cannot block an
otherwise valid simplified continuation, and the simplified path cannot be
used to weaken their stricter requirements.
