# Incident report: experiment-standardization rollout incompatibilities

- **Rollout window:** 2026-07-27 through 2026-07-28
- **Report prepared:** 2026-07-28
- **Scope:** Experiment control plane: plans, validators, trackers, launchers,
  preflight receipts, and immutable-source identities
- **Classification:** Non-atomic framework rollout with an unversioned
  backward-incompatible contract change
- **Status:** Contained by fail-closed gates; production launches remain
  blocked until the control plane is made internally consistent

## Executive summary

The failures that began during the 2026-07-27 standardization work are not a
set of unrelated experiment defects. Two effects are occurring together:

1. the new fail-closed checks are correctly exposing older assumptions that
   were previously unchecked; and
2. the standardization framework itself was integrated in several
   incompatible states rather than as one versioned, atomic release.

The clearest framework-level failure is the experiment-plan contract. Commit
`a00c3854` accepted the original four-field `execution.preflight` contract.
Commit `e2e084fd` made `receipt_schemas` and `receipt_bindings` mandatory and
explicitly added a test requiring legacy plans to fail closed. Both versions
still identify the plan schema as `experiment-run-plan/v1`. The stricter
validator reached this branch in merge commit `531f8f65` at
2026-07-28T09:31:39+02:00 without a schema-version change, plan migration, or
compatibility gate over the existing plan corpus.

At the 2026-07-28 audit, all 11 plans under `docs/experiment_plans/` failed the
current validator at the same stage:

```text
Expected all required keys in execution.preflight; got
['receipt_bindings', 'receipt_schemas']
```

The word `got` in this diagnostic is misleading: the listed fields are the
missing required keys.

No production experiment job was launched as a result of these incidents.
The fail-closed behavior prevented an inconsistent plan or environment from
reaching the GPUs, so this is a control-plane integrity incident rather than
corrupted scientific output.

## What was newly detected versus newly introduced

| Category | Finding | Interpretation |
|---|---|---|
| Older defect exposed by standardization | The local environment validator compared the configured `bin/python` alias text with Python's canonical `bin/python3.12` target text | The stronger gate found a real weakness in the old identity model, but the first failure was a false positive |
| Older gaps exposed by audit | Attempt roots, dataset identity, route ownership, and launch-time revalidation were not bound strongly enough | These controls needed hardening independently of the rollout |
| Rollout incompatibility | Required receipt fields were added while retaining `experiment-run-plan/v1` | This was a backward-incompatible schema change without a new version |
| Rollout incompatibility | Existing plans were neither migrated nor tested as a corpus before merge | The framework's own accepted artifacts became invalid |
| Rollout incoherence | Experiment source archives freeze experiment code but not the lifecycle validator and orchestration contract | A source identity can remain unchanged while the rules used to approve it change |
| Integration risk | The audit snapshot contained 33 staged changes, 13 unstaged changes, and 52 untracked paths; several control-plane files existed only in those partial states | Validators, plans, runners, trackers, and documentation were not represented by one reproducible repository revision |
| Integration risk | The reflog records a reset and then the `531f8f65` merge at 09:31 on 2026-07-28 while successor artifacts were being prepared | The control plane changed underneath in-progress preparation |

## Incident sequence

1. Standardization work added stronger plan, tracker, preflight, dispatch, and
   fail-stop controls throughout 2026-07-27.
2. The Conv1/Conv2 initialization-rho environment check stopped on the
   interpreter alias-versus-target mismatch before any payload gate or
   production launch.
3. The immediate validator defect and the broader workload-identity gaps were
   corrected in an isolated successor.
4. During report and plan revalidation, the branch received the stricter
   receipt-binding validator through `531f8f65`.
5. The successor plan and every other existing plan still used the earlier
   preflight contract, despite all declaring schema `experiment-run-plan/v1`.
6. The tracker was observed with stash-conflict markers before the merge. The
   current tree has no unmerged index entries and those markers are now gone,
   but their temporary presence is evidence that live control-plane states
   overlapped during the rollout.

## Root cause

The standardization changed many safety boundaries at once without defining a
single framework release identity and upgrade transaction. In particular,
there was no enforced rule that:

- a breaking plan-contract change must increment the schema version;
- all existing approved and draft plans must validate or be migrated before
  the new validator is merged;
- lifecycle tooling must be frozen alongside the experiment source and named
  in the preflight authority;
- launch preparation must use a clean, conflict-free control-plane revision
  from start to finish.

The fail-closed policy limited the damage, but it cannot by itself guarantee
that all producers and consumers implement the same contract generation.

## Impact and current containment

- Production jobs launched from the affected Conv1/Conv2 attempt: **0**.
- Scientific result files corrupted or incorrectly accepted: **none known**.
- The corrected Conv1/Conv2 source and isolated roots remain available.
- The current v2 plan is a draft and cannot pass the current plan validator.
- The current plan corpus is not compatible with the merged validator.
- No further smoke, T/K, or production launch should proceed until the
  framework contract is stabilized and the exact plan validates.

The run-specific failure and remediation evidence remain in
[`perfectdiode-conv12-init-rho-local-environment-20260727.md`](perfectdiode-conv12-init-rho-local-environment-20260727.md).

## Required corrective actions

1. **Create an explicit framework release boundary.** Commit the plan schema,
   template, validator, tracker validator, dispatch contract, receipt schemas,
   and lifecycle guidance together and identify that revision in every
   preflight.
2. **Version the breaking plan change.** Prefer
   `experiment-run-plan/v2` for receipt bindings. Either retain an actual v1
   validator or provide a deterministic v1-to-v2 migration; do not redefine
   v1 in place.
3. **Add a corpus compatibility gate.** A framework change must validate every
   tracked plan. If a breaking migration is intended, the merge must fail
   until all in-scope plans are migrated and reviewed.
4. **Bind the control-plane identity.** Freeze the lifecycle-tool revision and
   relevant validator hashes in the plan/preflight receipt, not only the
   experiment source archive.
5. **Add a clean-state launch gate.** Reject unresolved conflicts, staged and
   unstaged drift in authority files, and untracked authority files before
   freezing or launching an experiment.
6. **Use isolated worktrees for framework and experiment changes.** Do not
   merge, reset, or apply a stash into the worktree that is producing an
   immutable launch candidate.
7. **Run one golden end-to-end canary after framework changes.** Exercise the
   exact template-to-plan-to-tracker-to-preflight-to-dispatch path before
   applying the release to active experiments.
8. **Improve the missing-key diagnostic.** Report missing and unknown keys
   explicitly so the validator does not describe missing fields as values it
   “got.”

## Smallest safe next action

Do not repair the current plan ad hoc. First choose and implement one plan
schema transition policy—recommended: preserve v1 and introduce v2—then
migrate the plan corpus in one reviewed change and run the corpus compatibility
gate. Only after that should the corrected Conv1/Conv2 successor be approved
and its payload preflight begin.
