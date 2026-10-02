# Campaign workflow migration

Current routing: the [experiment-loop skill](../.agents/skills/experiment-loop/SKILL.md)
now owns Explorer → Codifier → Monitor → Reviewer. run-experiment is a compatibility
entry point. GPU placement, sharing and timing policy is consolidated in
[experiment_workflow](experiment_workflow.md); root AGENTS.md keeps authority,
global invariants and compatible section links. The experiment Markdown carries
the Codifier-to-Monitor handoff; its result note carries factual evidence and then
Reviewer verdicts. Existing records, runners, bundle schemas and resource policies
are preserved. The original setup/migration account below is historical provenance.

Prepared September 24, 2026 against `filip137/drn-server-code`, branch
`codex/lean-experiment-workflow`, commit `6be7d41038018b225cef7d1771fdb905084b0db7`.

## Decision and implementation status

Add a research-planning and review layer above the existing lean execution layer.
The [experiment-loop skill](../.agents/skills/experiment-loop/SKILL.md) routes planning,
idea exploration, hypothesis formulation and review. Existing run-experiment still
executes a concrete assigned experiment, with its existing operational lifecycle.
The handoff is a campaign path and an experiment path, not another chat-only plan.
These are responsibilities, not a requirement to use different products for each role.

This is an independently written adaptation of Filip's requested campaign structure,
informed by the experiment-loop example discussed previously. It is not a vendored
upstream skill or a schema-compatibility claim. No softschema, database, new launcher,
additional scheduler or third-party dependency is introduced.

Supplied: two populated campaigns, generic skill/artifact templates, a small deterministic
ledger generator and tests, and navigation edits to eight existing Markdown files.
The implementation is supplied as a local patch/package: the GitHub integration rejected
the write with HTTP 403, so no commit or pull request is claimed. No training, checkpoint
replay, GPU launch or official-test evaluation was performed.

Historical summaries remain at their original paths. Their measurements are imported
with provenance and limitations; underlying untracked local bundles were not accessed
or revalidated. Prospective plans explicitly name unavailable source/config inputs.

## Local integration — September 24, 2026

Applied to the `codex/lean-experiment-workflow` worktree at the prepared base commit.
All 54 package checksums matched and `git apply --check` passed; application required
no conflict resolution. The 46 new files and eight existing Markdown edits were
integrated, with one repository-specific correction: `.gitignore` now permits campaign
result Markdown notes and `.gitkeep` files while excluding generated payloads in those
directories. The existing root `results/` exclusion remains effective.

Local verification passed: all 17 ledger unit tests, both ledger freshness/structure
checks, 156 relative Markdown links in the 42 new documents, and diff whitespace
checks. All supplied campaign notes and placeholders are visible to Git. These are
integration checks; imported scientific evidence remains `imported-summary`, and the
source audits have not been performed as part of this setup.

## File-by-file map: active entry points and methodology

| Markdown | Change in this patch | Ongoing responsibility |
|---|---|---|
| `README.md` | Add campaign navigation and both questions; retain protocol links and correct the obsolete medium-affine-only description. | Repository entry point. |
| `AGENTS.md` | Add a short research-planning section. Delete no operational, resource or authorization policy. | Global invariants, task authority and scientific gates. |
| `.gitignore` | Permit campaign result Markdown notes and placeholders; keep generated run bundles excluded. | Track the complete lightweight campaign record. |
| `.agents/skills/run-experiment/SKILL.md` | Add campaign handoff and result/ledger pointers; retain phase delegation/monitoring. | Execute a concrete assigned experiment. |
| `.agents/skills/experiment-loop/SKILL.md` | New generic planning/review router. | Explore → falsify → specify → interpret → choose next step. |
| `.agents/skills/experiment-loop/references/artifacts.md` | New minimal metadata and copyable templates. | Artifact definitions and generator instructions. |
| `campaigns/README.md`, `campaigns/AGENTS.md` | New index and local instructions. | Current questions and campaign editing rules. |
| `docs/experiment_workflow.md` | Add campaign-to-execution handoff; retain commands/policy. | Smoke, launch, monitor, recover, collect. |
| `docs/experiment_reporting.md` | Explain campaign result notes and generated ledger; no bundle-schema change. | Canonical evidence and validation. |
| `docs/current_experiments.md` | Add navigation; preserve old p99 handoff explicitly as a historical snapshot. | Compatibility entry point, not a second live queue. |
| `docs/current_simulations.md` | Retired September 25, 2026; preserve existing contents. | Historical execution snapshot; new handoffs and result paths live in experiment/pilot notes. |
| `docs/experimental_manifest.md` | Retired September 25, 2026; preserve existing contents. | Historical conclusions; new interpretation lives in result/pilot notes. |
| `campaigns/pilots/` | One evolving note per small bounded test. | Question, execution and review without full campaign scaffolding. |
| `docs/current_state.md` | Read for context only; unchanged. | Filip's human-authored dashboard. |
| `docs/my_notes.md` | Neither read nor changed for this setup. | Personal notes; explicit access only. |

## File-by-file map: read-noise beta

| Existing Markdown | Destination / treatment |
|---|---|
| `docs/beta_study.md` | Add only a navigation banner. Preserve synthesis, dates and evidence. Current scope goes prospectively to `campaigns/read-noise-beta/README.md`; ideas/reasoning to `ideas.md`/`explorations/`; new tests to `series/002-policy-qualification/experiments/`. |
| `docs/conv3_unit_output_displacement_beta_20260922.md` | Keep unchanged; import its completed calibration into historical exp-001 and its result. Do not rerun merely to populate the structure. |
| `docs/experiment_plans/eqprop-conv3-p99-read-noise-5em4-20260922-v1.md` | Keep original execution contract/amendments. Link from campaign; do not reactivate dated launch instructions. |
| `docs/eqprop_baseline_beta_audit_20260914.md` | Keep baseline provenance; read during audit before importing finer baseline records. |
| `docs/eqprop_baseline_read_noise_beta_plan_20260916.md` | Keep historical selection context, not a parallel current plan. |
| `docs/eqprop_baseline_read_noise_launch_plan_20260916.md` | Keep historical execution context. New runs use current operational policies. |

The beta campaign contains two explorations, four hypotheses, two historical experiment
records and three prospective plans. Initialization calibration, p99 training, p90-trained
replay and changing K-reference studies remain distinct. It does not declare that unit
output RMS has already solved noisy training.

## File-by-file map: CIFAR-10

| Existing Markdown | Destination / treatment |
|---|---|
| `docs/cifar_experiment_review_20260924.md` | Add only a navigation banner; preserve consolidated observations. Current scope goes to `campaigns/cifar10-legacy-gap/README.md`; new reasoning/tests to that campaign. |
| `docs/cifar_l8_mechanism_plan_20260923.md` | Retain controls, tolerances and amendments; link from imports/source audit. |
| `docs/cifar_l8_mechanism_findings_20260923.md` | Preserve full findings and limited 200-minibatch identity scope, not a universal conclusion. |
| `docs/cifar_l8_bn_ablation_plan_20260923.md` | Preserve frozen/no-BN controls and negative/unqualified results; do not schedule as untested. |
| `docs/cifar_l8_bn_conclusions_and_next_experiments_20260923.md` | Keep dated provenance; separate competing explanations in X-001/X-002 and current next tests in campaign records. |
| `docs/cifar_l8_bn_gain_overnight_plan_20260923.md` | Preserve completed/cancelled history; no expired-queue restart. |
| `docs/cifar_l8_amplification_audit_20260923.md` | Preserve tested implementation/solver scope, not a universal guarantee. |
| `docs/cifar_l8_gradient_scale_interpretation_20260923.md` | Preserve raw-gradient versus actual-Adam-step distinction. |
| `docs/cifar_l8_analog_batch_plan.md` | Preserve historical batch selection, not a proven long-horizon optimum. |
| `docs/cifar_l8_analog_lr_plan_20260921.md` | Preserve selection provenance; recover exact resolved vectors, not prose approximations. |
| `docs/cifar_l8_legacy_lr_refinement_plan_20260922.md` | Preserve negative ten-epoch nearby-rate tests; do not infer a joint or 50-epoch optimum. |
| `docs/cifar_l8_analog_continuation_plan_20260922.md` | Preserve original 10→30 full-state continuation. |
| `docs/cifar_l8_baseline_legacy_e30_e50_plan_20260922.md` | Preserve original continuation/checkpoint identities. |
| `docs/cifar_l8_ours_e30_e50_plan_20260922.md` | Preserve original identities; continuation segments are not independent seeds. |
| `docs/cifar_digital_l8_plan_20260921.md` | Keep exploratory digital official-test reference distinct from analog validation. |
| `docs/cifar_digital_l12_plan_20260921.md` | Same; different architecture/BN and repeatedly observed test set are not matched analog evidence. |

The CIFAR campaign contains two explorations, five hypotheses, two historical experiment
records and three prospective plans. It retains counterevidence to epsilon-only and
selective-affine-BN harm explanations, while leaving endpoint reproducibility and the
long-horizon mapped-optimizer explanation open.

Central summaries/plans and directory structure were inspected. The older candidate-import
documents above have not all received a new line-by-line or raw-data audit. This table
specifies migration treatment, not complete reconstruction of every historical arm.

## Avoid competing sources of truth

For these two questions, put all new ideas and experiment intent in campaigns. Old dated
plans remain historical records of what ran, not parallel mutable queues. An ongoing
run's owner or original scientific contract is not silently changed by this patch.

Campaign README owns current scope/accept/stop rules. ideas.md maps alternatives.
Explorations preserve reasoning without forcing every idea into a hypothesis. Hypotheses
state claims; experiments specify tests; results record observations and reviewed,
scoped verdicts. Only ledger.md is generated. It does not infer conclusions from remote
logs, generate human dashboard prose or collapse conflicting judgments.

Large logs, checkpoints and analyses stay in `results/<study-id>/`. Campaign results
hold small notes and pointers. Experiment/pilot notes own handoffs and result paths;
result/pilot notes own conclusions. The old central index and manifest are historical.
A campaign directory is not a simulation result directory.

## Rollout and acceptance

Apply the patch on a review branch after `git apply --check`. Run the supplied generator
tests and `--check` for both ledgers. Structural checks do not validate the scientific
results, raw source bundles or GPU availability.

The first scientific action in both campaigns is the source-audit experiment. Recover
exact config/hash/coverage inputs there, then finalize the next bounded test. Explicit
execution requests retain the repository's standing authority: no new approval
bureaucracy is created, and proposed budgets are not allocated resources.

Do not bulk move/delete old docs now. After source audit, split additional historical
families into individual imported records only where navigation improves. Preserve dates,
failed attempts, retrospective status and uncertainty. Archive redundant plans only
after reconciling inbound links and active run ownership.
