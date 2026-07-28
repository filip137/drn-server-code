# Experiment-Flow Acceptance Tests

Date: 2026-07-28
Purpose: expose control-plane incompatibilities with the smallest possible
side effect before using real experiment batches
Default evidence scope: infrastructure validation only; never paper-facing

## Test philosophy

These tests are layered so one failure identifies one interface. Do not skip
ahead after a failed layer, and do not use a production experiment to discover
basic host or control-plane compatibility.

The first five tests are local and non-launching. Remote validation is bounded
and uses new immutable attempt IDs. No production attempt is armed until the
user approves the resolved continuation and every target smoke passes.

| ID | Test | Expected time | External side effect | What it isolates |
|---|---|---:|---|---|
| F00 | Three launch-focused runner tests | 10 seconds | None | Import runner, singleton Slurm contract, and request package |
| F01 | Legacy-corpus audit | 2 seconds | None | Full-path migration health; not a fast-path gate |
| F02 | Strict study/attempt split | 5 seconds | Temporary files only | Full-path versioning and migration |
| F03 | Golden six-run continuation | 30 seconds | Temporary files only | Request-to-dispatch integration |
| F04 | Failure localization and loop budget | 60 seconds | Temporary files only | Error quality, retry scope, and stop/ask behavior |
| T00 | Target functional admission | at most 6 minutes | One bounded real-batch validation and disposable receipt | Runtime, data, CUDA, numerics, checkpoint, and output compatibility |
| T01 | Trex one-step validation simulation | 5 minutes | One short GPU process and disposable artifacts | Real launcher, config, data, model, CUDA, and output path |
| T02 | Trex result round trip | 2 minutes | Copy of the disposable validation bundle | Transfer and local integrity |
| T03 | Trex production dispatch dry plan | 30 seconds | None | Approved argv, receipts, tracker gate, and request identity |
| T04 | Controlled armed-failure test | 2 minutes | One deliberately failing disposable long-run attempt | Fail-stop, reporting, and successor authorization |

## Global constraints

- Keep scientific config and selected job definitions immutable. A dirty
  unrelated worktree or a different compatible Python/PyTorch/CUDA version is
  recorded as provenance and is not by itself a diagnostic-path blocker.
- Use an available target-profile Python candidate; do not require one
  hard-coded interpreter alias when another candidate passes the same smoke.
- Never edit [`current_state.md`](current_state.md) or
  [`my_notes.md`](my_notes.md).
- Keep all test artifacts under `results/_control_plane_validation/` locally
  and on Trex. They are infrastructure evidence and must never enter the
  result registry.
- Every remote test has a hard timeout and a unique attempt ID.
- T00 and T01 are bounded pre-arm validation. They must not print
  `LONG-RUN ATTEMPT ARMED`.
- T04 is the only intentional armed failure. After it fails, stop and wait for
  a new user message before cleanup, repair, or a successor attempt.
- A failed test preserves its receipt and logs. A corrected run uses a new
  attempt ID and path.

## F00: Three launch-focused runner tests

Command:

```bash
KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/filip/miniconda3/envs/py312/bin/python -m pytest -q \
  labs/tests/test_jeanzay_validation_fastpath.py::test_stable_runner_is_syntax_valid_and_remote_preflight_is_import_only \
  labs/tests/test_jeanzay_validation_fastpath.py::test_cached_profile_builds_one_exact_singleton_sbatch_command \
  labs/tests/test_jeanzay_validation_fastpath.py::test_approved_request_prepares_one_package_without_its_own_catalog_entry
```

These are the complete local pytest launch gate for this runner. They check
syntax/import coverage, the exact singleton validation-only Slurm contract,
and direct preparation of one package from an approved request without a new
catalog entry. Request, parent resolution, manifest, tracker, and source-hash
validation are deterministic contract checks rather than additional pytest
suites.

Pass condition:

- exactly these three focused tests pass in the documented interpreter.

Failure interpretation:

- stop the passing launch path and diagnose the failed interface;
- run only the smallest relevant broader suite after a focused test,
  preflight, or canary failure;
- never run the broad catalog, common smoke, production adapter, legacy
  supervisor, or unrelated experiment suites merely as routine launch
  preparation; and
- after an armed failure, wait for a new user message before diagnostic test
  escalation.

## F01: Real-artifact control-plane audit

Report-only command:

```bash
/home/filip/miniconda3/envs/py312/bin/python \
  labs/tools/audit_experiment_control_plane.py
```

Release-gate command:

```bash
/home/filip/miniconda3/envs/py312/bin/python \
  labs/tools/audit_experiment_control_plane.py --strict
```

The strict command verifies the full legacy lifecycle:

- every real plan validates under its declared canonical schema with live file
  hashes;
- the catalog and tracker are individually valid;
- every tracker ID has an exact catalog identity;
- at least one catalog route binds a canonical approved plan; and
- launch-authority files form a clean worktree release.

Failures here identify legacy-corpus or release-migration work. They block a
full-plan/paper-facing launch that depends on those artifacts, but they do not
block the simplified continuation when its exact parent, selected jobs,
generated study, and functional path validate.

## F02: Scientific-study and execution-attempt split

The replacement does not reinterpret `experiment-run-plan/v1`. It introduces
the distinct `experiment-study/v1` and
`experiment-execution-attempt/v1` schemas documented in
[`experiment_study_attempt_contract.md`](experiment_study_attempt_contract.md).

Focused command:

```bash
KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/filip/miniconda3/envs/py312/bin/python -m pytest -q \
  labs/tests/test_experiment_study_attempt_contracts.py
```

Real-corpus dry run:

```bash
/home/filip/miniconda3/envs/py312/bin/python \
  -m experiments.control_plane check-legacy-corpus \
  --repo-root . --json
```

Required assertions:

1. Study approval binds only the canonical scientific `study` object.
2. Moving unchanged artifact bindings preserves scientific approval but
   requires a newly authorized attempt.
3. A scientific change invalidates scientific approval.
4. A host or launcher change invalidates attempt authorization without
   changing the study.
5. Every study-required scientific gate is required by the attempt.
6. `validation` and `production` duration boundaries are enforced.
7. Missing and unknown fields are reported separately with
   `pre_arm_repairable` and zero launched jobs.
8. The legacy split is byte-deterministic and preserves execution, storage,
   and old approval only in `not_launch_authority` residue.
9. Legacy approval is never transferred to a new study or attempt.
10. Both output paths are collision-checked before publication.
11. Multiple attempts may route proper study subsets, and the attempt-set
    validator requires exact non-overlapping coverage with no unknown jobs.

Current result:

- 13 focused tests pass;
- all 11 current legacy plans split and verify against their current config,
  manifest, protocol hashes, and manifest job counts;
- ten old approved plans become `review_required`;
- one old draft remains `draft`; and
- zero launch authorities are created.

Pass condition:

- the focused suite and all-real-plan dry run remain green, and a human reviews
  each migrated study that is still active before approving it.

## F03: Golden six-run continuation

The permanent fixture is
[`experiment_requests/perfectdiode-conv1-best-rho-continuation-20260728.md`](experiment_requests/perfectdiode-conv1-best-rho-continuation-20260728.md).
It uses the real parent config and staged bundle but performs no GPU or remote
side effect in F03.

Exercise the public facade in this order:

```text
request parse + aggregate validation
  -> compositional parent/subset resolution
  -> scientific-only study + automatic hash
  -> stop at one user-review boundary
  -> three target validation plans and dispatch requests
  -> six exact production jobs with non-overlapping ownership
  -> idempotent publication
```

Pass conditions:

- pending review produces no attempt, tracker, preflight, staging, or launch;
- the approved fixture resolves exactly six Conv1 jobs;
- local owns baseline SGD/Adam, Trex owns ours SGD/Adam, and Akib owns legacy
  SGD/Adam;
- validation contains exactly one real-batch job per target;
- production requests do not exist before aggregate functional preflight;
- no fresh `T/K` gate is requested;
- operational binding changes do not alter the scientific hash;
- changing a learning rate does alter the scientific hash; and
- identical publication is idempotent while conflicting content is rejected.

The focused implementation lives in
`labs/tests/test_experiment_flow.py`,
`labs/tests/test_experiment_study.py`, and
`labs/tests/test_functional_dispatch.py`.

## F04: Failure localization matrix

Run the focused fixtures with one fault at a time:

| Case | Injected fault | Required outcome | Launch count |
|---|---|---|---:|
| F04-01 | Missing or invalid intake fields | one aggregate `needs_user_input` response | 0 |
| F04-02 | Ambiguous parent | exact disambiguation request, no nearest-match substitution | 0 |
| F04-03 | Unknown subset/job | study validation reports the complete invalid selection | 0 |
| F04-04 | Scientific value changed after approval | scientific hash mismatch | 0 |
| F04-05 | Environment/version differs but capabilities pass | admitted and recorded as provenance | 0 |
| F04-06 | Dataset, CUDA, finite check, or checkpoint write fails | target functional preflight fails | 0 |
| F04-07 | Repeated identical pre-arm error | stop and ask the user | 0 |
| F04-08 | Two failed diagnosis-backed repairs or ten minutes | stop and ask the user | 0 |
| F04-09 | Production preflight missing or mismatched | no production dispatch request | 0 |
| F04-10 | Post-arm payload failure | immutable terminal report and wait for a new user message | at least 1 |

Each error record must state expected and provided values, an error code,
attempt ID, completed stages, live job count/state, valid artifacts, and one
smallest next action.

Pass condition: no case exposes a later contract failure first, no unchanged
failing command is retried, and the user sees only scientific choices or the
smallest actionable runtime blocker.

## T00: Target functional admission

Do not introduce a separate exact-environment doctor as another gate. The
bounded validation request is the capability check and uses the same public
job adapter as production.

For each of local, Trex, and Akib, the executor:

1. checks that its named lane is reachable and not occupied;
2. stages the adapter and immutable parent bundle as an operational action;
3. selects an available Python candidate from the target profile;
4. launches exactly one representative job in `validation` mode;
5. executes the real data loader, model, equilibrium phases, loss, gradients,
   and one optimizer step on CUDA;
6. requires finite outputs, a disposable checkpoint, and the compact
   `experiment-functional-preflight/v1` receipt;
7. records actual Python, PyTorch, CUDA, GPU identity/memory, dataset identity,
   elapsed time, and peak memory as provenance; and
8. uses immediate tmux/process readback and a hard deadline of 360 seconds.

Pass condition: all three target receipts pass, every validation request
contains one job, no official-test read occurs, and no production request is
created until the receipts aggregate successfully.

Software status: implemented. Live status: pending the user's approval of the
six-run continuation fixture.

## T01: Trex one-step validation simulation

This is the black-box agent test closest to “all details are supplied; launch
the validation on Trex.”

Frozen validation definition:

| Field | Value |
|---|---|
| ID | `control-plane-trex-one-step-v1` |
| Classification | infrastructure validation; never scientific evidence |
| Host | `filip@trex`, GPU 0 |
| Dataset | existing ordinary-MNIST training data only |
| Official test read | forbidden |
| Architecture | Conv1 |
| Nonlinearity | perfect diode |
| Amplification | baseline `v1/c1` |
| Model seed | 0 |
| Shuffle seed | 0 |
| Batch size | 16 |
| Equilibrium/unroll | `T=4`, `K=4` |
| Optimizer | plain SGD |
| Source config | `configs/conv/perfectdiode_conv12_best_observed_confirmation_20260727_v1.json` |
| Source config SHA-256 | `3635c5866b27445c58c90de034c4f2e152919772b40258820f614a60a61ea740` |
| Source entry | index 0, `pdconfirm_conv1_baseline_sgd_seed0` |
| Raw parameter learning rates | `Bias_0=0.14269596288753023`, `ConvWeight_0=0.14269596288753023`, `DenseWeight_0=0.012023754247240415` |
| Work | one real batch, forward/equilibrium phases, loss, gradients, one optimizer step |
| Required artifact | resolved config, metrics JSON, one checkpoint, completion receipt |
| Remote output | `/home/filip/server_code/results/_control_plane_validation/trex/<attempt-id>` |
| Intended local destination | `results/_control_plane_validation/trex/<attempt-id>` |
| Expected duration | at most 300 seconds |
| Hard timeout | 360 seconds |

Every field not listed as a validation-only override is inherited byte for
byte from the hash-bound source config and entry. The only allowed execution
overrides are:

```text
classification=control_plane_validation
maximum_optimizer_steps=1
validation_passes=0
official_test_read=false
output_root=<the attempt-specific root above>
```

The implementation must materialize and hash a resolved validation config
before execution. The agent must not infer or recompute a scientific value.

Copy/paste bounded agent request:

```text
Run experiment-flow acceptance test T01 exactly as defined in
docs/experiment_flow_acceptance_tests.md.
Attempt ID: control-plane-trex-one-step-<UTC timestamp>.
All scientific and operational values are frozen by the T01 fixture.
Run only the bounded one-step validation on Trex GPU 0.
Do not launch, resume, cancel, or modify any production experiment.
Do not ask for choices already fixed by the fixture.
Hard timeout: 360 seconds.
Return the remote receipt, immediate process readback, and intended local
destination.
```

Pass conditions:

- catalog/fixture resolution is exact;
- no clarification is requested;
- the same public launcher/config/device/output path used by a real smoke is
  exercised;
- one real optimizer step completes on CUDA;
- checkpoint and metrics hashes validate;
- `official_test_read=false`;
- immediate process readback is unambiguous;
- no production job or long-run dispatch occurs; the concise tracker may
  record the verified `preflighting` transition; and
- the run finishes before the timeout without arming the long-run policy.

Software status: the resolver, same-path numerical smoke, functional target
profile, sub-ten-minute dispatch request, and one-SSH submit transport are
implemented. Live execution remains pending the user's approval and
executor-owned staging of the adapter and immutable parent bundle.

## T02: Trex result round trip

After T01 reaches a terminal success:

1. Use the canonical remote-sync workflow.
2. Pull a metadata snapshot first.
3. Copy the complete disposable bundle into a local incoming directory.
4. Verify file count, bytes, SHA-256 identities, completion receipt, config,
   metric, and checkpoint.
5. Publish it only under the local control-plane-validation root.
6. Do not create a comparison card or review.

Pass condition:

- the local bundle is self-contained and matches the remote receipt; remote
  cleanup is a separate action and is not part of this test.

## T03: Trex production dispatch dry plan

Use the approved six-run fixture and the passing target functional receipts.
Build the aggregate preflight and generated production dispatch requests, then
inspect them without invoking `start`.

Pass conditions:

- `side_effects_performed=false` and launched-job count is zero;
- local, Trex, and Akib own exactly two non-overlapping jobs each;
- all six jobs use the common public adapter and exact approved scientific
  definitions;
- source, target profile, functional preflight, output, argv, expected
  duration, and hard deadline are generated and bound;
- concurrency is reduced when observed peak/free memory requires it; and
- changing request content under the same attempt ID is rejected at `start`.

This test proves production authority without occupying a GPU.

## T04: Controlled armed-failure test

Run only after F00 through T03 pass and after an explicit user message
authorizes this deliberate failure.

Payload:

- a disposable command expected to run for at least 600 seconds;
- it writes a start marker and then exits nonzero on a configured trigger;
- it cannot access scientific data or production paths.

Expected behavior:

1. `start` prints `LONG-RUN ATTEMPT ARMED` with immutable paths.
2. Exactly one fresh Trex tmux session/process is launched.
3. The nonzero exit produces one first-write-wins terminal failure receipt.
4. Status reports the exact failure and proves descendant quiescence.
5. No retry, fallback, repair, cleanup, or successor attempt occurs in that
   turn.
6. A later user message authorizes a new attempt ID.
7. The successor succeeds without overwriting any failed evidence.

Pass condition:

- the intentional post-arm failure stops once and reports completely; it does
  not become an automatic retry loop.

## Campaign decision table

| First failing test | Primary repair area |
|---|---|
| F00 | Component implementation |
| F01 | Legacy/full-path release and migration health |
| F02 | Strict full-path schema versioning and migration |
| F03 | Public lifecycle facade and real producer/consumer integration |
| F04 | Structured errors and recovery state machine |
| T00 | Target staging, capability smoke, or bounded transport |
| T01 | Real launcher/config/data/GPU/output integration |
| T02 | Remote result sync and local integrity |
| T03 | Production authority binding |
| T04 | Armed fail-stop and successor semantics |

For the simplified continuation, require F00, F03, F04, and the applicable
live T00 through T03 checks. F01 and F02 gate only work that consumes the
legacy/full-study contracts. T04 is a deliberate safety test and requires its
own explicit authorization; it is not silently run as a prerequisite to the
user's scientific experiment.
