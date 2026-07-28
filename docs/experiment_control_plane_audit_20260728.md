# Experiment Control-Plane Audit

Date: 2026-07-28
Scope: experiment request, planning, preflight, launch, recovery, transfer, and
publication
Disposition: analysis only; no live plan, tracker, protocol, receipt, or job
state was changed

Follow-up: the side-by-side scientific/operational split proposed below is
implemented and tested in
[`experiment_study_attempt_contract.md`](experiment_study_attempt_contract.md).
It remains shadow tooling until the adoption gates in that document pass.

The six-run Conv1 local/Trex/Akib request provides a concrete walkthrough of
where that migration still blocks a simple experiment:
[`conv1-six-run-multihost-protocol-blockers-20260728.md`](incident_reports/conv1-six-run-multihost-protocol-blockers-20260728.md).

## Executive conclusion

The safety policy is not primarily failing because it is too strict. It is
failing because strict checks were added across several independently changing
contracts without one released, end-to-end-compatible control plane.

The repository currently has strong unit coverage for individual validators
and dispatch components, but no passing path through the real artifacts:

- 114 focused control-plane tests pass;
- 0 of 11 real experiment plans pass the current canonical plan validator;
- the catalog reports itself valid, but has no route to a canonically valid
  approved plan;
- one live tracker experiment ID has no exact catalog entry;
- no real plan contains the receipt bindings required by the generic local or
  Trex dispatcher; and
- launch-authority files are spread across a large, dirty working tree rather
  than one reproducible framework release.

This combination explains the apparent error loop. Each layer correctly fails
closed, but the output of one accepted layer is not necessarily valid input to
the next layer. Agents can repair the latest error only to expose another
contract mismatch, and once a long-run attempt is armed the intentional
fail-stop policy requires a new user turn.

## Current protocol

The intended lifecycle is:

```text
request
  -> catalog resolution
  -> protocol resolution
  -> config + manifest + approved plan
  -> live tracker entry
  -> payload smoke + scientific gates + scheduled-run receipt
  -> target compatibility and canary
  -> launch-ready tracker gate
  -> armed production launch
  -> monitoring
  -> remote close + local transfer + validation
  -> comparison card + review + generated result index
```

The stages and their current authorities are:

| Stage | Primary authority or implementation | Expected output | Typical failure boundary |
|---|---|---|---|
| Request routing | [`experiment_catalog.py`](../experiments/experiment_catalog.py), [`experiment_catalog.json`](../experiments/experiment_catalog.json), and [`experiment_launch_request.md`](experiment_launch_request.md) | One exact experiment route and next action | `ambiguous`, `not_found`, missing plan, or a route that resolves but remains launch-blocked |
| Scientific resolution | [`conv_paper_hyperparameter_protocol.md`](conv_paper_hyperparameter_protocol.md) and its active protocols | Complete hypothesis, controls, sweep, metric, and decision rules | An unresolved scientific choice or an evidence-scope mismatch |
| Study freeze | [`run-experiment-pipeline/SKILL.md`](../skills/run-experiment-pipeline/SKILL.md), its template, and [`validate_experiment_plan.py`](../skills/run-experiment-pipeline/scripts/validate_experiment_plan.py) | Hash-bound config, manifest, plan, storage, reporting, and approval | Schema drift, placeholders, path topology, file/hash drift, job-count mismatch, or missing approval |
| Live-state registration | [`current_experiments.md`](current_experiments.md) and [`validate_current_experiments.py`](../skills/run-experiment-pipeline/scripts/validate_current_experiments.py) | Exactly one three-field entry with a lifecycle state | Missing, duplicate, malformed, wrong-ID, stale, or non-`launch-ready` entry |
| Execution preflight | Root policy, experiment-specific smoke/T/K tools, [`scheduled-run-preflight/SKILL.md`](../skills/scheduled-run-preflight/SKILL.md), and receipt adapters | Valid inner receipts and one outer preflight envelope | Producer/consumer schema mismatch, stale receipt, wrong subject/source/path, failed smoke, or failed T/K gate |
| Target compatibility | Experiment-specific runners, Jean Zay wrappers, Trex runners, and dispatch profiles | Verified interpreter, CUDA/GPU, data, storage, scheduler/tmux, and source identity | Site syntax, alias-versus-target identity, CPU accounting, temporary storage, staged path layout, capacity, or dirty source |
| Canary | Tracker canary receipt plus target supervisor | A live same-path payload result while the tracker remains `preflighting` | Scheduler delay, allocation mismatch, payload error, missing artifact, or stale tracker receipt |
| Production | Production tracker receipt plus generic or experiment-specific supervisor | Immediate scheduler/process readback and immutable attempt state | Any final revalidation, submission, readback, or controller error |
| Monitor and collect | Supervisor/status tools and remote-sync workflow | Terminal remote bundle, local transfer receipt, and local validation | Unobservable progress, timeout, incomplete bundle, transfer mismatch, or validation failure |
| Publish | Result registry cards, reviews, and renderer | Reviewed comparison in `docs/results/index.md` | Unsupported schema, incomplete coverage, mixed evidence scope, or missing review |

The protocol distinguishes two important recovery domains:

- Before `LONG-RUN ATTEMPT ARMED`, implementation, environment, smoke, and
  preflight failures are repairable with bounded diagnosis and a new receipt.
- After arming, the first unexpected launch or monitoring failure is terminal
  for that attempt and turn. Diagnosis is read-only; repair, retry,
  cancellation, or a successor attempt requires a new user message.

That boundary is scientifically useful and should remain. The repair goal is
to move compatibility discovery and ordinary mistakes before the boundary,
not to weaken post-arm fail-stop behavior.

## Reproduced repository state

The read-only audit command is:

```bash
/home/filip/miniconda3/envs/py312/bin/python \
  labs/tools/audit_experiment_control_plane.py
```

Observed on 2026-07-28:

| Check | Result |
|---|---:|
| Real plans accepted by the canonical validator with file verification | 0 / 11 |
| Real plans accepted by the catalog-only legacy compatibility path | 10 / 11 |
| Catalog entries | 6 |
| Catalog plan states | 3 legacy receipt contracts, 1 draft, 2 missing |
| Catalog routes to a canonical approved plan | 0 |
| Valid tracker entries | 4 |
| Tracker IDs without an exact catalog entry | 1 |
| Launch-authority paths modified or untracked | 20 |
| Focused plan/catalog/tracker/scheduled/dispatch unit tests | 114 passed |
| Experiment-specific Trex runner/runtime unit tests | 10 passed |

The direct corpus failure is:

```text
Expected all required keys in execution.preflight; got
['receipt_bindings', 'receipt_schemas']
```

The listed keys are missing, not values that were received. The diagnostic
therefore also demonstrates an error-message defect.

The catalog has a private compatibility option that accepts the legacy
four-field preflight object for routing, while the public plan validator and
dispatcher reject it. This makes the catalog internally green without making
any real route executable.

The tracker entry
`perfectdiode-conv3-tk-ordinary-mnist-20260727-v3` is not an exact catalog
identity. Resolving it returns `not_found` and suggests a differently named
entry whose catalog plan is missing, even though the tracker links an approved
v3 plan.

## How the incidents map onto the flow

### Standardization rollout

The [rollout incident](incident_reports/experiment-standardization-rollout-20260727.md)
is a producer/consumer release failure at the study-freeze boundary.
`receipt_schemas` and `receipt_bindings` became mandatory while the schema
name remained `experiment-run-plan/v1`. The template and validator moved, but
the plan corpus and its approvals did not. A later catalog-only compatibility
path improved routing visibility but did not restore a launch path.

This is still reproducible in the current working tree.

### Conv1/Conv2 confirmation on Jean Zay

The [confirmation incident](incident_reports/perfectdiode-conv12-best-observed-confirmation-20260727.md)
crossed five target-integration boundaries serially:

1. unsupported scheduler memory flags;
2. requested physical CPUs versus allocated logical CPUs;
3. compute-node `/tmp` capacity instead of `$JOBSCRATCH`;
4. repo-relative manifest resolution in a split checkout/bundle topology; and
5. scheduler start time beyond the attempt deadline.

The first four are compatibility facts that a short target doctor should have
found before the production plan was frozen. The fifth is an external
capacity outcome that should have been compared with the deadline before live
submission.

Every operational correction changed hashes, attempt paths, plan text, and
receipts. That was safe, but unnecessarily expensive because the stable
scientific study and the mutable execution attempt are stored in one plan.
The approved plan contains Jean Zay-specific absolute paths, attempt numbers,
receipt locations, and a complete supervisor command. A host or staging fix
therefore invalidates the same document that records the scientific decision.

### Initialization-rho environment check

The [local environment incident](incident_reports/perfectdiode-conv12-init-rho-local-environment-20260727.md)
was a target-identity false positive: a stable `bin/python` alias and its
canonical `bin/python3.12` target were compared as strings. The immediate fix
was small, but it led to a large experiment-specific hardening pass because
there was no reusable host-identity contract.

## Why agents enter a loop

### 1. There is no real golden path

The unit tests construct synthetic plans and receipts that satisfy the new
interfaces. They do not require any checked-in plan to validate or any catalog
entry to reach a launchable state. Consequently, 114 tests can pass while
every real plan fails at the first canonical gate.

### 2. The plan mixes two lifetimes

Scientific choices should remain stable across operational retries. Host,
queue, attempt ID, deadline, staging path, and receipt location change per
attempt. The current plan stores both, so an operational discovery forces a
cascade:

```text
environment fix
  -> source/config/runner change
  -> new hashes and paths
  -> plan amendment and approval
  -> new preflight receipts
  -> new tracker gates
  -> another remote compatibility discovery
```

### 3. Compatibility is discovered too late

Jean Zay scheduler syntax, CPU representation, `$JOBSCRATCH`, archive
extraction, staged topology, and predicted start time were learned only after
the launch contract was frozen. Trex likewise has environment facts embedded
in experiment-specific runners rather than one reusable, expiring capability
receipt.

### 4. Several launch systems overlap

The repository currently has:

- a 4,456-line generic `experiments.local_dispatch`;
- experiment-specific Trex shell runners;
- experiment-specific Jean Zay submitters and supervisors; and
- an 8,600-line perfect-diode successor runtime.

The generic dispatcher requires receipt maps that no real plan has. Existing
plans instead point to experiment-specific controllers. An agent must infer
which orchestration generation is authoritative, and passing one controller's
tests does not prove another controller's contract.

### 5. There is no bounded short-run transport

`experiments.local_dispatch` deliberately accepts only runs expected to last
at least 600 seconds and immediately arms the long-run policy on `start`.
That is appropriate for production dispatch, but it cannot run a one- to
five-minute Trex compatibility or validation simulation. The only available
alternatives are experiment-specific SSH/tmux code or an ad hoc command, both
of which the standardization is intended to eliminate.

### 6. Routing and live identity diverge

The catalog, plan filename, tracker marker, result identity, and launcher do
not always share one exact experiment ID. A route can be `resolved` while its
next action is `repair_catalog_or_plan`, or a valid tracker plan can be
`not_found` by the catalog.

### 7. Framework state is not released atomically

The control plane spans root and nested `AGENTS.md` files, two skills,
templates, validators, the catalog, tracker, dispatch profiles, generic
transport, and experiment-specific controllers. In the audited tree, 20 of
those launch-authority paths are modified or untracked. There is no single
framework release identity proving that all producers and consumers came from
one compatible revision.

### 8. Errors lack recovery metadata

Most errors are prose. They do not consistently identify:

- a stable error code and stage;
- whether the failure is pre-arm and repairable or post-arm and terminal;
- whether any job or side effect occurred;
- the exact artifact that remains valid;
- the smallest safe next command; or
- whether the same error fingerprint has already repeated.

An agent can therefore treat a repairable preflight error as terminal, or keep
retrying an unchanged command that cannot succeed.

## Recommended target design

### Separate the study from the attempt

Introduce two contracts:

1. `experiment-study/v1` for hypothesis, comparisons, sweep, decision
   rule, config/manifest identity, storage class, and publication targets.
2. `experiment-execution-attempt/v1` for target, environment capability,
   source release, runner, staging/output paths, deadline, receipt bindings,
   tracker gates, and scheduler/tmux identity.

Filip approves the study. A compatible execution attempt may be regenerated
after an operational failure without rewriting scientific choices. A material
scientific change still requires a successor study and approval.

### Release the framework as one unit

Add an `experiment-control-plane-release/v1` manifest that binds the versions
and hashes of:

- plan and attempt schemas;
- templates and migration tools;
- catalog and tracker validators;
- receipt schemas and adapter registry;
- dispatch profiles and generic transport; and
- lifecycle guidance.

The release gate should fail until the full real plan corpus and a golden
end-to-end fixture pass. Launch receipts should record this release identity.

### Provide one public CLI

A small facade should own phase transitions and print one structured next
action:

```text
experimentctl resolve
experimentctl doctor --target trex
experimentctl prepare
experimentctl preflight
experimentctl launch
experimentctl status
experimentctl collect
experimentctl publish
```

Agents should not assemble validators, receipt hashes, tracker gates, SSH,
tmux, and supervisors independently.

### Distinguish validation from production transport

Keep the existing long-run semantics for production, and add an explicit
bounded class such as `probe` or `validation`:

| Run class | Maximum duration | Scientific authority | Arming behavior |
|---|---:|---|---|
| `doctor` | 60 seconds | Infrastructure only | Never arms |
| `validation` | 5 minutes | Diagnostic fixture; never publishable evidence | Bounded, repairable, no long-run terminal rule |
| `production` | 10 minutes or longer | Approved study and attempt | Explicitly arms and uses fail-stop-report |

All classes should use the same transport and structured receipts, but only
production should require the complete launch-ready gate and long-run
terminal behavior.

### Move host discovery ahead of experiment freeze

A versioned Trex or Jean Zay doctor should verify, in one bounded run:

- SSH and hostname;
- canonical interpreter identity;
- PyTorch/CUDA/GPU identity;
- dataset identity and read access;
- writable output and temporary storage;
- tmux or scheduler request/readback behavior; and
- source archive extraction and staged path topology.

The receipt should be reusable only for a short declared lifetime and exact
host/runtime fingerprint.

### Standardize structured failures

Every public command should emit a record like:

```json
{
  "stage": "plan_validation",
  "error_code": "missing_required_keys",
  "expected": ["receipt_bindings", "receipt_schemas"],
  "provided": ["receipt_path", "smoke_required"],
  "recovery_scope": "pre_arm_repairable",
  "launched_job_count": 0,
  "last_valid_artifacts": [],
  "next_action": "migrate the v1 plan with the checked-in v1-to-v2 tool"
}
```

If the same error fingerprint repeats without an input change, the controller
should stop and report rather than retry.

## Recommended order of repair

1. Preserve a real v1 plan reader and introduce a versioned v2 contract, or
   provide a deterministic, reviewed migration. Do not redefine v1 again.
2. Add the real-corpus compatibility gate and make the catalog fail when it
   has no canonical approved route.
3. Create one committed golden experiment-flow fixture and require it to pass
   template, plan, tracker, receipt, dispatch-plan, status, and collection
   stages.
4. Add exact catalog entries for every live tracker identity.
5. Split stable study plans from per-attempt execution contracts.
6. Add the bounded `doctor` and `validation` transport classes for Trex.
7. Consolidate experiment-specific launchers behind the generic transport and
   a small receipt-adapter interface.
8. Commit the framework as one clean release and bind that release in every
   new attempt.
9. Run the acceptance campaign in
   [`experiment_flow_acceptance_tests.md`](experiment_flow_acceptance_tests.md)
   before resuming production experiments.
