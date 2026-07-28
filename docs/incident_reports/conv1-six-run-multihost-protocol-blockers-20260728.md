# Case Study: Six Conv1 Runs Blocked Before Preflight

- Date: 2026-07-28
- Request: six perfect-diode Conv1 confirmations, covering baseline, ours,
  and legacy amplification with SGD and Adam, for 10 epochs across the local
  GPU, Trex, and Akib
- Evidence scope: ordinary-MNIST diagnostic
- Disposition: documentation only; no smoke, T/K gate, or production run was
  launched

## Executive summary

This request did **not** get stuck because its scientific definition was
missing. The existing reviewed config already identifies the exact six cells,
rho values, raw parameter learning rates, seed, initialization, dataset split,
input gain, `T/K`, epoch budget, and official-test exclusion.

It got stuck because the repository currently has incompatible generations of
experiment authority:

1. the catalog cannot express “run this subset of an existing study” and
   returns `not_found`;
2. the canonical plan validator rejects all 11 real plans, including the
   parent plan that contains the requested cells;
3. the proposed study/attempt replacement is explicitly shadow-only and has
   no production launcher;
4. the first replacement attempt contract could not represent a `2 + 2 + 2`
   multi-host shard;
5. the reusable numerical bundle and runtime are hard-coded to the original
   12-entry, Jean-Zay/Trex-specific confirmation;
6. the generic dispatcher supports only local `main` and Trex, not Akib, and
   supports no bounded sub-10-minute validation transport; and
7. there is no clean, atomic control-plane release containing a compatible
   catalog, schema, validator, receipt model, dispatcher, and runner.

The safety gates correctly prevented an unauthorized launch. The protocol
failure is that there is no currently passing route between those gates. The
agent can repair one interface only to encounter the next incompatible
interface, which creates the observed pre-arm error loop.

The investigation should have stopped and reported once the corpus audit
showed `0/11` valid plans and the replacement contract was confirmed to be
non-authoritative. Continuing into sharding and runner design expanded a
simple experiment request into control-plane development and was the main
reason the response took too long.

## The requested study was scientifically resolvable

The numerical authority is
[`perfectdiode_conv12_best_observed_confirmation_20260727_v1.json`](../../configs/conv/perfectdiode_conv12_best_observed_confirmation_20260727_v1.json),
whose current SHA-256 is
`3635c5866b27445c58c90de034c4f2e152919772b40258820f614a60a61ea740`.
The requested subset is its first six entries:

| Entry | Scheme | Optimizer | `(rho_conv, rho_dense)` | Epochs |
|---|---|---|---:|---:|
| `pdconfirm_conv1_baseline_sgd_seed0` | baseline | SGD | `(0.009, 0.03)` | 10 |
| `pdconfirm_conv1_baseline_adam_seed0` | baseline | Adam | `(0.009, 0.03)` | 10 |
| `pdconfirm_conv1_ours_sgd_seed0` | ours | SGD | `(0.009, 0.03)` | 10 |
| `pdconfirm_conv1_ours_adam_seed0` | ours | Adam | `(0.009, 0.03)` | 10 |
| `pdconfirm_conv1_legacy_sgd_seed0` | legacy | SGD | `(0.003, 0.03)` | 10 |
| `pdconfirm_conv1_legacy_adam_seed0` | legacy | Adam | `(0.003, 0.01)` | 10 |

The same config binds the raw learning-rate vector for every entry. The active
protocols resolve the remaining shared values as ordinary MNIST, Conv1,
perfect diode, input gain 40, `T=4`, `K=4`, seed 0, a fresh restart from the
architecture-shared initialization, 3,438 steps per epoch, and no official
test-set read.

Five cells are explicitly
`best_observed_parent_core_candidate_diagnostic_override` with an unresolved
upper rho boundary. Legacy Adam is a bracketed but unpublished diagnostic
selector. Therefore the correct study interpretation is:

- test exact completion, finite numerics, safety, and validation trajectories
  independently for the six prespecified cells;
- make no cross-scheme superiority claim; and
- do not promote these values to protocol-frozen learning rates.

A minimal diagnostic hypothesis and decision rule can be generated
deterministically from those facts. No scientific clarification from Filip
was needed before preparing the study.

## Where the current protocol stopped

| Lifecycle stage | Outcome for this request | Blocking? |
|---|---|---:|
| Catalog resolution | `not_found`; the 12-entry parent appeared only as a suggestion | Yes, because suggestions cannot be selected silently |
| Active-protocol resolution | Exact six cells and interpretation resolved | No |
| Scientific study freeze | A clean six-entry study is representable | No |
| Multi-host attempt routing | Initial replacement schema assumed one attempt owned the whole study | Yes; repaired only in shadow tooling |
| Authoritative plan validation | Existing parent plan rejected; corpus audit reports `0/11` valid plans | Yes |
| Production authority | Replacement study/attempt contracts explicitly remain shadow-only | Yes |
| Source and bundle selection | Clean numerical source exists, but its bundle is a 12-entry host-bound artifact | Yes |
| Target reachability/capacity | Local, Trex, and Akib were reachable and their GPUs were idle at inspection time | No |
| Scheduled/static preflight | Not run | Not reached |
| Same-path payload smoke | Not run | Not reached |
| Scientific `T/K` gate | Not run | Not reached |
| Tracker `launch-ready` gate | No new tracker entry was created | Not reached |
| Armed production launch | Not armed; zero jobs launched | Not reached |

## Blocking findings

### B01 — Catalog routing is not compositional

The required first command returned:

```text
status=not_found
reason=No entry matched every searchable query token.
```

It suggested
`perfectdiode-conv12-best-observed-confirmation-20260727-v1`, but catalog
policy correctly prohibits silently choosing a suggestion for a `not_found`
result.

The scientific payload already exists; what is new is only the six-entry
subset and its three-host routing. The catalog currently treats that as a
brand-new experiment instead of a derived view of the parent manifest.

Required improvement: support an exact parent-plus-subset route, with the
subset identified by immutable entry IDs. Host allocation must remain outside
the scientific route.

### B02 — The authoritative plan schema has no valid real artifact

The parent plan declares `experiment-run-plan/v1`, but the current validator
requires `execution.preflight.receipt_schemas` and
`execution.preflight.receipt_bindings`, fields that were added without a
schema-version change.

Direct validation fails with:

```text
Experiment plan validation failed: Expected all required keys in
execution.preflight; got ['receipt_bindings', 'receipt_schemas']
```

The repository-wide audit currently reports:

```text
declared-plan-schema-corpus: 0/11
canonical-approved-catalog-route: none
```

This is a global release blocker, not a defect in the requested experiment.
Creating another `experiment-run-plan/v1` document would either reproduce the
same incompatibility or encode the newer contract under the same misleading
version.

Required improvement: restore a real v1 validator or introduce v2 plus a
deterministic corpus migration. The framework release must not become active
until every in-scope real plan passes its declared schema.

### B03 — The replacement study/attempt model is not launch authority

The new
[`experiment-study/v1` and `experiment-execution-attempt/v1` design](../experiment_study_attempt_contract.md)
correctly separates scientific approval from operational authorization.
However, its own adoption gate states that it is shadow validation and
migration tooling until launchers consume it and the end-to-end acceptance
tests pass.

Using it to launch this request would bypass the currently authoritative
lifecycle while falsely claiming the migration was complete.

Required improvement: make one explicit cutover. Until then, there must be
exactly one documented authority path; agents must not be asked to combine
the legacy plan and replacement attempt models themselves.

### B04 — A study could not initially be sharded across attempts

The natural representation is one six-job study and three operational
attempts:

```text
local GPU: 2 study jobs
Trex:      2 study jobs
Akib:      2 study jobs
```

The initial attempt validator required every routed manifest's job count to
equal the complete study count. Three two-job attempts therefore all failed,
even though their union was exactly the study.

A shadow-only repair now adds:

- `selection.mode=all|subset`;
- explicit `selected_study_job_ids`; and
- `validate-attempt-set`, which rejects overlap, omission, unknown jobs, or
  mixed study identities.

The focused study/attempt suite passes 13 tests, including exact
non-overlapping sharded coverage. This is useful design evidence, but it is
unreleased and cannot authorize production.

Required improvement: make exact cross-attempt coverage a first-class schema
property before the replacement becomes authoritative.

### B05 — The “scientific” bundle contains operational identity

The reusable parent bundle is immutable and hash-valid, but its manifest
contains:

- exactly 12 entries;
- a production execution-source block;
- an environment-contract hash; and
- assumptions inherited from the Jean Zay confirmation.

A scientific subset for local, Trex, and Akib cannot reuse that manifest as
its scientific authority without also inheriting an unrelated execution
environment. Rebuilding it for each host would change what should be one
shared scientific identity.

Required improvement: make the study manifest contain only scientific job
payloads and immutable input evidence. Put source release, interpreter,
environment, target, output, deadline, and receipts exclusively in execution
attempts.

### B06 — The existing numerical runtime is fixed to the old experiment

[`perfectdiode_successor_confirmation.py`](../../experiments/mnist_conv/perfectdiode_successor_confirmation.py)
is not a generic selected-entry runner. Static checks in that module require:

- `entry_count == 12`;
- entry indices in `[0, 11]`;
- six predefined Conv1/Conv2 scheme rows;
- fixed two-entry pack indices;
- a fixed canary pack; and
- Jean Zay or Trex-specific receipt/backend semantics.

Running only the first six entries through a new six-job study would require
either weakening those assertions or writing an adapter around lower-level
training functions. Either choice is implementation work, not experiment
preflight.

Required improvement: expose one small public runner that accepts a
hash-verified scientific manifest plus explicit job IDs. Counts, pack
composition, target, and concurrency must be data, not module constants.

### B07 — The generic dispatcher cannot represent the requested target set

[`local_dispatch.py`](../../experiments/local_dispatch.py) accepts only:

```text
main + local transport
trex + SSH transport
```

There is no Akib profile or Akib target. The dispatcher also rejects
`expected_duration_seconds < 600`, so it cannot run the required one-step,
one- to five-minute same-path validation on a remote GPU.

The alternative is to use experiment-specific SSH/tmux runners or construct
commands manually, exactly the behavior the standardization is intended to
remove.

Required improvement: use one target-profile interface for local, Trex, and
Akib, and add bounded `doctor` and `validation` run classes that share the
production transport without arming the long-run policy.

### B08 — Preflight receipts are split into incompatible dialects

The pipeline skill requires role-specific schema and binding maps. The generic
dispatcher recognizes a small fixed adapter table. The successor runtime
produces its own smoke, canary, Trex, and production receipt schemas. Existing
plans contain the older four-field preflight object.

There is no checked-in compatibility proof showing that a receipt produced by
the selected runner is accepted by the selected dispatcher and plan
validator.

Required improvement: standardize one receipt envelope with a role, subject,
producer, artifact hash, environment identity, status, and payload-schema
version. Keep experiment-specific measurements inside the payload, behind a
versioned adapter registry tested producer-to-consumer.

### B09 — The available smoke is not the requested minimal same-path test

The successor “smoke pack” is fixed to the old canary pack, performs fresh
security replay for all six Conv1/Conv2 scheme rows, and runs two full
one-epoch Conv2 training canaries. The single-entry smoke also performs all
six security rows and a full epoch.

That is scientifically strong for the old 12-entry Jean Zay batch, but it is
not a cheap one-step validation for one selected Conv1 entry on each of three
hosts. Running it would consume substantial time before testing the new route
and would validate rows outside this study.

Required improvement: provide a production entry point with a
validation-only execution budget override, such as one optimizer step and
zero validation passes. The resolved validation config must be hashed, the
output must be non-publishable, and every other scientific field must remain
identical.

### B10 — There is no atomic control-plane release to stage

The numerical source archive is clean and content-addressed:

```text
source commit:
367abf8d0f9a76b562a65a0b80c0ebbaef6624dc

source archive SHA-256:
228d17636e036e49e0e16302f52bcf7f85efb4c150d8786220eb204e9d9e310e
```

It predates the replacement contracts. Conversely, the current control-plane
changes are in a dirty working tree. The latest audit reports 22 modified or
untracked launch-authority paths. There is no single release identity that
contains both the new contracts and compatible producers/consumers.

Required improvement: create a minimal, content-addressed control-plane
release artifact and validate its full real-artifact corpus before staging.
Do not require an experiment agent to assemble a release from a dirty tree
during a launch request.

### B11 — Approval is both mandatory and underspecified for derived studies

The user supplied all material scientific choices and explicitly asked to run
them. The lifecycle skill nevertheless requires an agent to materialize a
plan, show a compact generated summary, receive approval, and then record that
approval inside the plan.

That checkpoint is appropriate when scientific choices remain open. For an
exact subset/replay, however, it creates a second approval round after the
user has already given an executable request. It also becomes ambiguous
whether the initial request approves the not-yet-materialized plan bytes.

Required improvement: define two explicit cases:

- `approval_required`: a generated study contains a new material scientific
  choice; and
- `approval_satisfied_by_request`: the request names an immutable parent,
  exact subset, fixed overrides, decision template, and evidence scope, with
  no inferred scientific choice.

In the second case the controller should render and record the resolved
summary, but must not force another conversational round trip.

### B12 — Pre-arm repair has no concrete stopping budget

The post-arm policy has a precise terminal rule. The pre-arm policy permits
bounded diagnosis and repair but defines “bounded” only qualitatively.
Consequently, an agent can:

```text
repair catalog representation
  -> repair study sharding
  -> discover plan-schema incompatibility
  -> design a runner adapter
  -> discover dispatcher target gaps
  -> design new receipts
```

Each action is locally reasonable and still pre-arm, but their composition is
an unbounded framework rewrite.

Required improvement: add a mandatory pre-arm report boundary. For example,
stop and report before further edits when any of these is true:

- the real-plan corpus gate is red;
- the selected contract is explicitly non-authoritative;
- satisfying the request requires a new launch transport or receipt dialect;
- two independent control-plane incompatibilities have been found; or
- the next repair changes more than one lifecycle layer.

Continuing beyond that boundary should require an explicit user request to
repair the framework rather than to run the experiment.

## What was not blocking

At the time of the read-only inspection:

- the local RTX 3090, Trex RTX 5090, and Akib RTX 3080 were reachable and had
  no experiment compute process occupying them;
- passwordless access to Trex and Akib worked;
- the six requested rho values and raw learning rates were available;
- the clean numerical source archive and immutable parent bundle were
  present; and
- no official-test access or scientific result mutation occurred.

Akib's 10 GiB GPU may require its two Conv1 jobs to run sequentially. That is
an execution-concurrency decision, not a scientific ambiguity, and should be
resolved by a bounded memory canary rather than by changing the study.

## Why this investigation took too long

There were two separate causes.

### Protocol cause

No command reports the complete incompatibility set. Each gate exposes only
its immediate mismatch, while policy permits continued pre-arm repair. The
agent therefore followed a serial chain of individually legitimate checks
without a framework-level stop condition.

### Agent cause

After confirming that the replacement schema could not shard the study, the
agent implemented subset routing and coverage validation before first
reporting the blocker. It then explored adapting the old 12-entry runtime.
Those were useful control-plane investigations, but they were a scope
expansion from “run six existing cells.”

The correct response point was earlier:

```text
catalog not_found
  + 0/11 canonical plans valid
  + replacement marked shadow-only
  => stop, document, and ask whether to repair the framework
```

No GPU work was performed during this time.

## Recommended repair order

### P0 — Establish one passing authority path

1. Version the plan-schema break and migrate the entire real-plan corpus.
2. Choose the legacy-plan model or the study/attempt model as the sole
   production authority during a declared release; do not require both.
3. Publish one atomic control-plane release identity.
4. Gate the release on the real corpus and one golden end-to-end lifecycle.

### P1 — Make the target experiment structurally representable

1. Adopt exact subset selection and cross-attempt coverage.
2. Split scientific manifests from environment and host identity.
3. Parameterize the runner by job IDs and concurrency.
4. Add Akib to the common target-profile transport.
5. Add `doctor` and bounded `validation` run classes.
6. Standardize receipt envelopes and adapters.

### P2 — Prevent another error loop

1. Add a report-only `prepare` command that evaluates every static gate and
   returns all blockers in one structured response.
2. Add the explicit pre-arm stopping budget described in B12.
3. Make the catalog resolve immutable parent subsets.
4. Define when an exact user request itself satisfies scientific approval.

## Acceptance tests derived from this case

These tests supplement
[`experiment_flow_acceptance_tests.md`](../experiment_flow_acceptance_tests.md).
They should pass before retrying the six-run study.

| ID | Test | Pass condition |
|---|---|---|
| C01 | Exact request resolution | The request resolves to the six entry IDs above with no invented scientific value |
| C02 | Parent-subset catalog route | The catalog returns an exact parent and subset, not `not_found` or a suggestion |
| C03 | Corpus release gate | Every real plan passes its declared validator, or has been deterministically migrated |
| C04 | `2 + 2 + 2` coverage | Three attempts cover all six jobs exactly once; omission, overlap, and unknown IDs fail |
| C05 | Scientific portability | Changing local/Trex/Akib routing leaves the study hash unchanged |
| C06 | Common target doctor | The same public command validates local, Trex, and Akib profiles in bounded time |
| C07 | Same-path one-step smoke | One selected Conv1 entry performs one real optimizer step on each host within 360 seconds |
| C08 | Scoped T/K gate | Exactly the three Conv1 scheme rows pass fresh `4/4` versus `64/64` reference checks |
| C09 | Receipt compatibility | Every producer receipt is accepted by the selected plan/attempt validator and dispatcher |
| C10 | Dry production preparation | One command returns exact six-job ownership, argv, output roots, and `launched_job_count=0` |
| C11 | Clean release staging | All three hosts report the same control-plane release and numerical source identities |
| C12 | Pre-arm stop behavior | An injected second independent contract mismatch produces one blocker report and no further repair |

The decisive end-to-end test is:

```text
one approved six-job study
  -> three authorized two-job attempts
  -> three bounded same-path smokes
  -> three scoped T/K gates
  -> exact cross-attempt coverage
  -> one launch-ready transition
  -> six immediate process identities
```

Until C01–C12 pass, another attempt to launch this study will mostly retest the
control plane rather than the science.

## Current disposition and preserved evidence

- Long-run attempt armed: **no**
- Production jobs launched: **0**
- Smoke jobs launched: **0**
- T/K gates launched: **0**
- Tracker entry created for this six-run subset: **no**
- Scientific outputs created: **none**
- Remote state mutated: **none**

Shadow control-plane work produced during the investigation:

- subset selection in execution attempts;
- exact cross-attempt coverage validation;
- a `validate-attempt-set` CLI; and
- one sharding regression test.

The focused suite currently passes `13 passed`. These changes remain
non-authoritative until the adoption and release gates are completed.

Related evidence:

- [`experiment-standardization-rollout-20260727.md`](experiment-standardization-rollout-20260727.md)
- [`perfectdiode-conv12-best-observed-confirmation-20260727.md`](perfectdiode-conv12-best-observed-confirmation-20260727.md)
- [`experiment_control_plane_audit_20260728.md`](../experiment_control_plane_audit_20260728.md)
- [`experiment_study_attempt_contract.md`](../experiment_study_attempt_contract.md)
- [`experiment_flow_acceptance_tests.md`](../experiment_flow_acceptance_tests.md)
