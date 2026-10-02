---
name: experiment-loop
description: Explore, specify, implement, launch, monitor and review hypothesis-driven research experiments. Owns the Explorer, Codifier, Monitor and Reviewer handoffs for campaigns and concrete assigned runs.
---

# Experiment loop

This skill owns **Explorer → Codifier → Monitor → Reviewer**. The Codifier also
handles failures escalated by the Monitor. Reuse existing runners, launchers, bundles
and records; do not add a control plane or launch schema.

Read root AGENTS.md. Enter at the role matching the request; do not repeat completed
phases. Planning, setup and review alone do not launch experiments. A request to run or
complete an experiment authorizes implementation through interpretation within its
scope and budget, without another approval round trip. Ask only about material missing
scientific choices or access blockers; record lean operational defaults.

## Records and reading

For a small pilot, use one evolving `campaigns/pilots/<name>.md` note for the question,
expected outcome, comparison, decision rule, execution handoff, results and review.
Do not create a campaign scaffold
or separate report for it. Use the [pilot convention](references/artifacts.md#small-pilots).
Existing campaigns retain their records; link established context instead of copying it.

| Role | Working context |
|---|---|
| Explorer | Current question and pilot/experiment note; use the [index](../../../campaigns/README.md) for discovery and open other campaign context only when needed. |
| Codifier | Assigned note and relevant [execution policy](../../../docs/experiment_workflow.md) sections. Open named contracts for unresolved scientific details and [reporting](../../../docs/experiment_reporting.md) when implementing/validating bundles. |
| Monitor | Assigned experiment's execution/monitoring handoff; execution policy's [Status](../../../docs/experiment_workflow.md#status) and [During And After](../../../docs/experiment_workflow.md#during-and-after) sections. Load reporting details for collection or validation issues. |
| Reviewer | Predeclared decision rule and compact collected evidence; inspect raw artifacts or other protocols only for missing facts, contradictions or unresolved checks. |

Read only the relevant [artifact convention](references/artifacts.md) when creating
records or refreshing a ledger, and host guidance only for the chosen target.
Repository execution policy overrides stale global skill advice. Reuse unchanged context
already read in the session; a role transition does not require rereading it.

Keep evidence reads and handoffs compact: coverage, selected measurements, matching and
validation outcomes, exceptions, and paths. Extract named JSON fields and bounded log
tails; leave full manifests, hash lists and raw tables in files. If output truncates,
narrow the query rather than repeat the dump. One role's validation of unchanged
artifacts is reused by the next; repeat it only after changes or a concrete concern.

## Explorer

Discuss the research question and alternatives with Filip. Put the hypothesis,
comparison, decision rule, bounded cases, budget and stop rules in the pilot/experiment
note before implementation. For established campaigns, link existing hypotheses and
update ideas/explorations only when the scientific reasoning changes.

A scientifically ready experiment can still need code. Specify the computation and
expected evidence sufficiently for the Codifier to implement it. Routine implementation
is not a missing scientific decision. Record the scope actually authorized; a ready
plan alone is not permission to launch. Hand off campaign and experiment paths.

## Codifier

Turn the assigned specification into readable code/configs, resolving routine engineering
choices while preserving the scientific contract. Supplied exact configs use the
exact-run path without additional tuning or calibration. For a non-campaign run, use
its existing plan as the handoff; do not require a campaign migration.

Follow execution policy for GPU placement/sharing, timing, gates, launch and
recovery. Record actual source identity, including relevant uncommitted changes; a commit
alone may not identify the launched code. Before substantial compute, report cases,
target, budget, duration and result path. For prepared lab GPU jobs, submit to the
shared [run-watch queue](/home/filip/.codex/skills/run-watch/SKILL.md), record the
job ID in the existing handoff, and verify the service has accepted monitoring
ownership. Declare `resource_profile: "conv2"` or `"conv3"` for those models:
two Conv2 workers may share a 5090; Conv3 permits one worker, reserving its
measured or bounded peak memory plus headroom. There is no fixed 31 GiB
minimum. Keep peak memory separate from headroom; follow execution policy.
The queue launches and monitors without repeated model calls. Direct
launches retain the startup check and explicit Monitor handoff below.

Filip permits sharing an authorized simulation GPU with Ben on all GPU models,
including RTX 3080, 3090 and 5090. Ben's presence or 100% utilization alone is
not a reason to queue or request permission again. Verify ownership and memory
headroom, preserve the scientific config, leave Ben's jobs/MPS/GPU settings
untouched, and budget using measured shared-GPU throughput. See the repository's
[sharing policy](../../../docs/experiment_workflow.md#sharing-gpus-with-ben).

Do not reinstate retired pre-launch diagnostics from older protocols: fixed-T/K
gradient checks, Conv3 residual/gradient selection or operating-point replay,
optimizer/rho probes, and CIFAR solver/migration audits are opt-in scientific work.
Use the declared T/K and LR vectors for routine launches. Keep execution smoke,
identity/restore checks and runtime nonfinite guards; absence of an audit is not a pass.

Fill an **Execution and monitoring handoff** section in the existing experiment record:

- Cases, target/handles and evidence paths; link exact configs, commands and identities.
- Progress/cadence, owner, last/next check, remaining budget/deadline and pause rules.
- Executable inspect/collect/validate instructions, exceptions and escalation destination.

Update this section in place; send only changed handles, decisions and the note path
at handoff, not another full narrative. Require the Monitor to acknowledge the assigned
handles, deadline and next check through the delegation channel, and begin its check/
wait loop before relinquishing polling ownership. A written next-check timestamp or
detached launcher alone does not complete that handoff.

## Monitor

For shared-queue jobs, the persistent service owns routine polling, placement and
bounded operational recovery. Verify a fresh service observation and the recorded
job ID before handing over; do not start a second polling agent or competing watch.
Inspect queue state on request and use the collected evidence for the Reviewer.
Scheduling completion alone does not establish complete study coverage or review.

Use a separate, lighter agent for routine monitoring. Check every 30 minutes, or sooner
when progress cadence or deadlines require it. Follow the Status procedure: synchronize
observations, inspect compact status, compare actual progress with launcher state and
bounded logs, and update the assigned note's current handoff. Avoid repeatedly reading
full logs, hashing active checkpoints or scanning all result directories.

Collect completed outputs and run the declared validation commands. Reconcile every
expected case, including failures, exclusions and replacements. Write factual coverage,
measurements and validation evidence into the existing pilot/result note; leave hypothesis
verdicts empty until review. Mark ready-for-review only after local evidence and coverage
requirements pass. Report partial or failed coverage honestly.

Escalate unexpected exits, missing/stalled progress, numerical or validation failures,
budget/deadline risks, missing handoff details and capacity for authorized pending work.
Include affected cases/handles, last confirmed progress, exit/status evidence, a short
log excerpt and remaining budget. Continue observing unaffected runs.

Do not edit experiment code/configs, choose new experiments, diagnose broadly, cancel
jobs or improvise retries. Execute a predeclared mechanical pause/resume command only
after its stated checks pass; unexpected recovery returns to the Codifier. Staleness
is not proof of a dead process; planned pauses need checkpoint evidence. Operational
completion does not imply a positive scientific outcome.

## Delegation and recovery

The coordinating agent may also be Explorer, Codifier or Reviewer; no separate coordinator
worker is required. Only the coordinator delegates; workers return without spawning
further workers. Keep one monitoring owner per run and wait for reports instead of
duplicating its polling.

For queue-owned work, the shared service is that operational coordinator. Its recovery
agents request retries/resumes through the queue and do not launch independently.

With the currently exposed tools, use `gpt-6-astra` with high reasoning for Codifier and
recovery work and medium reasoning for Monitor work, unless the user specifies another
available model. Start workers with `fork_turns="none"` and pass only the role, paths,
assignment and essential constraints. Medium reasoning is an effort setting, not a
guarantee of a cheaper model. Narrow context and deterministic checks limit repeated work.

The Codifier diagnoses escalations and repairs/retries within the assigned science,
target class, budget and deadline. Preserve failures, rerun affected gates and update
handles/handoff before monitoring resumes. Claim pending cases before additional
launches to prevent duplicates; the Monitor continues observing unaffected jobs.
Material scientific changes return to the Explorer/user.

Treat timing-accounting differences as routine operational repairs: reconcile
remaining allowances to measured launcher wall time, including overhead and charged
retries, and continue unchanged authorized cases when they fit. Do not require a
user response for this correction or grant a fresh budget. Initialization replays
need bounded stopping and preserved completed artifacts, not training checkpoints.
Require full-state checkpoint restoration only when continuing stateful training.

Keep monitoring active or arrange an actual supported scheduled continuation. If
delegation is unavailable, the coordinator owns the checks in its active session,
using supported waits between due checks. If no scheduled continuation is available,
keep that session active through the bounded assignment. If it must end, record the
last observation, unmonitored interval and next required action, and report the gap;
do not claim a future check is armed. Detached training itself keeps no agent alive.

## Reviewer

Review reconciled, locally validated evidence against the original hypotheses and decision
rules. Distinguish measurement from inference, validity from predictive performance and
negative findings from execution failures. Complete interpretation, scoped verdicts and
the next decision in the same pilot/result note; do not duplicate it in a separate review.

Update the existing experiment/pilot record's state and regenerate an established
campaign's ledger when its inputs change. Keep the conclusion in its result/pilot note.
Pilots do not require a ledger. Do not edit the human-authored
docs/current_state.md without explicit permission. A full experiment assignment continues
through review; collection is not its final deliverable. Propose another experiment only
when it would change a decision; launch it only within already authorized scope, budget
and stop rules.

## Evidence and storage discipline

Preserve negative, partial, failed, cancelled and superseded evidence. Operational retries
are not independent replicates. Label retrospective hypotheses; never imply preregistration.
Distinguish calibration, replay and training, and finite-iteration BPTT from qualified
equilibrium gradients. Keep datasets, splits, architectures, algorithms, seeds and
checkpoint/optimizer histories visible. Do not pool unlike regimes or equate a diagnostic
gain with accuracy. Missing raw evidence cannot be replaced by manufactured validation.

For full campaigns, X-NNN, H-NNN and exp-NNN are independent local sequences; never reuse IDs.
Experiment IDs stay unique across series. Create a series for a changed instrument or
comparison regime, not each sweep cell. Campaign result notes stay small; large artifacts
stay in existing results roots. Record every top-level result path, including failed
and superseded attempts, in the owning experiment/pilot note. The generated ledger is
navigation; it displays written verdicts and never infers them from jobs or metrics.
docs/current_simulations.md and docs/experimental_manifest.md are historical snapshots;
do not update them or create replacement central narrative records.
