# AGENTS

## Purpose
This repository develops the experiments, evidence, and tooling for a paper
studying the importance of bidirectional amplification in dissipative
resistive networks.

## Scope and Boundaries
- In scope: Python source, configs, scripts, and documentation.
- Generated outputs and datasets may be created, read, synchronized, validated,
  and summarized when the task requires them, but large artifacts must not be
  committed. Follow `.gitignore` and keep only appropriate lightweight
  metadata, summaries, and documentation in Git.

## Repo Layout (key)
- `model/`: core model components
- `training/`: training loops and monitoring
- `experiments/`: experiment definitions, launchers, controllers, and collectors
- `labs/`: experiments and utilities
- `plotting_functions/`: analysis/plotting helpers
- `playbooks/`: SOPs for repeatable tasks
- `skills/`: repo-local Codex workflows and deterministic helpers
- `docs/`: lightweight state and notes
- `result_registry/`: measured comparison cards and reviewed interpretations
- `results/`: generated experiment, dispatch, transfer, and validation artifacts

## Nested Guidance
- Before changing files under a directory that contains its own `AGENTS.md`,
  read and follow that file in addition to this repository-wide guidance.

## Playbooks
- Optuna analysis SOP: `playbooks/optuna_analysis.md`
- Conv amplification paper protocol index: `docs/conv_paper_hyperparameter_protocol.md`
- Frozen Conv experiment definition: `docs/conv_paper_experiment_definition.md`
- Hard-sigmoid input-gain protocol: `docs/conv_paper_hard_sigmoid_input_gain_protocol.md`
- Operational T/K protocol: `docs/conv_paper_tk_protocol.md`
- Hard-sigmoid learning-rate handoff: `docs/conv_paper_learning_rate_protocol.md`
- Learning-rate diagnostic ledger: `docs/results/conv_learning_rate_diagnostics.md`
- Amplification result curation: `docs/results/amplification_experiment_curation.md`
- Agent handoff: `docs/current_state.md`
- Current experiment tracker: `docs/current_experiments.md`
- Filip's personal research notes: `docs/my_notes.md`
- Remote result workflow: `docs/remote_result_workflow.md`
- Task overrides: `playbooks/AGENTS.override.md`

## Experiment Launch and Compute Allocation
- For large experiment batches, first check whether local GPU tmux targets are free:
  - `tmux main`
  - `tmux akibscomputer`
  - `tmux trex`
- If any of those targets are free and reachable, use their available GPU
  capacity and parallelize there as much as GPU memory allows.
- Never replace or interfere with an occupied tmux lane.
- Use Jean Zay for the remaining jobs, for multiple long-running jobs, or for
  batches that exceed local capacity or would otherwise serialize
  unnecessarily.
- Jean Zay runs default to the R3 project `fmu`; use `fmu@v100` for V100 GPU jobs and write run outputs under `/lustre/fsn1/projects/rech/fmu/$USER/server_code/results` unless the user explicitly requests another project/account. The source checkout may stay under the existing `umg` work path.
- For MNIST Conv amplification training, prefer running Conv1 and Conv2 jobs on `tmux main` and `tmux akibscomputer`; reserve Conv3 jobs for Jean Zay and `tmux trex` unless local availability or urgency clearly argues otherwise.
- Prefer keeping the same launcher/config contract across local tmux and Jean Zay so results can be aggregated into one summary.
- For an already approved long simulation on `tmux main` or Trex, use the
  bounded, receipt-backed `python -m experiments.local_dispatch` workflow in
  `docs/local_dispatch.md`; do not reconstruct it with `send-keys` or a chain
  of one-check-per-SSH commands. Build its schema-validated v2 preflight
  envelope with the same tool before arming `start`.

## Long-Run Simulation Fail-Stop-Report Policy
- This policy applies only to an **armed long-run simulation attempt**: a
  numerical run or batch expected to occupy a compute lane for at least ten
  minutes, any scheduled/queued long-running simulation batch, or explicitly
  armed monitoring of such an active run. It does not apply merely because
  the repository contains experiment code.
- Before entering the final launch path, state `LONG-RUN ATTEMPT ARMED` with
  the experiment/attempt ID, target lane or scheduler, expected duration,
  hard deadline, and immutable state/receipt paths. The armed attempt covers
  final staging and revalidation of already-produced smoke/scientific gate
  receipts, submission, immediate scheduler or process readback, and
  active-run monitoring.
- Planning, code edits, unit/integration tests, linting, dependency and
  environment setup, plan-only/dry-run commands, short interactive probes,
  bounded developer smoke tests, and short smoke/scientific gate execution
  completed before `start` are outside the armed attempt. Failures there may
  be diagnosed, fixed, and retested with bounded observable work; they must
  never be represented as passed long-run launch gates. A gate that is itself
  a long or scheduled simulation is a separate armed long-run attempt.
- Offline collection and analysis after a run is no longer active are also
  outside the armed attempt. For any repairable failure outside the protected
  scope, diagnose the cause before rerunning and keep corrective work bounded
  and observable; never repeat blindly. If the cause remains unresolved after
  reasonable diagnosis-backed attempts, report it.
- Within an armed attempt, an unexpected failure is any nonzero exit,
  exception, timeout, unreachable required dependency, or missing, stale,
  ambiguous, or mismatched receipt, artifact, or scheduler readback that the
  approved plan does not explicitly classify as an independent scientific
  outcome. The first such failure is terminal for that long-run attempt and
  current launch turn.
- At that failure, stop the long-run launch/monitoring workflow. Perform only
  bounded, read-only inspection needed to identify the exact error and live
  job state. Do not edit code or configuration, substitute a path, tool,
  host, or environment, restage, refreeze, rerun, retry, resubmit, cancel, or
  delegate those actions in the same turn.
- Report and wait. Only a new user message sent after the long-run failure
  report may authorize diagnosis, repair, cancellation, or a new attempt. A
  plan, config, or launcher may define retry mechanics, but it never grants
  authority to initiate that retry.
- The report must name the exact stage, command, and error; completed stages;
  launched-job count, IDs, and live states (explicitly zero when none);
  last valid receipts, logs, and artifacts; files or external state changed;
  and the smallest proposed next action.
- Preserve any failed receipt and all diagnostic evidence. A later authorized
  attempt must use a new immutable attempt ID and receipt paths; never
  overwrite or resume a terminal failed attempt.
- A long-run controller may catch an unexpected exception only to atomically
  write one first-write-wins failure report, mark the attempt terminal, and
  exit nonzero. It must never catch and continue. A failed long-run state
  cannot submit or resume.
- Long-run polling and reconciliation loops require an explicit hard
  deadline, emit observable progress at least every 60 seconds, and stop and
  report if progress becomes unobservable or the deadline expires.
- A candidate-cell rejection that an approved sweep explicitly defines as an
  independent scientific outcome is not a pipeline failure. Record it as
  negative evidence and continue only as authorized by the frozen plan.

## Large-Run Preflight Policy
- This policy applies to every large or long experiment batch, whether
  launched immediately or scheduled for later.
- Before the full run, complete a minimal end-to-end test through the same
  launcher, configuration path, environment, device, and output path.
- For Conv amplification training, also run the active scientific `T/K`
  reference gate before launching the full batch, except for a true
  continuation that reuses an accepted operating point and changes none of
  architecture, nonlinearity, amplification, input gain, `T/K`, or the
  equilibrium/gradient algorithm. Such a continuation records the reused
  `T/K` authority and runs the functional smoke on every target instead.
- Fail closed: do not launch if either the execution smoke test or the
  required scientific reference gate fails.
- An operational code, host, path, Python, PyTorch, CUDA, or GPU change
  invalidates the functional smoke for that target but does not by itself
  invalidate an accepted scientific `T/K` result. A scientific change to the
  initializer, architecture, nonlinearity, amplification, input gain, `T/K`,
  or equilibrium/gradient semantics invalidates both as applicable.
- Scheduled runs must additionally follow
  [`skills/scheduled-run-preflight/SKILL.md`](skills/scheduled-run-preflight/SKILL.md).

## Pre-Arm Failure Handling
- Before a long-run attempt is armed, a failed gate, preflight, smoke test, or
  launch check fails closed for production launch but does not invoke the
  armed-attempt stop-and-wait rule. Diagnose the cause before rerunning and
  keep fixes and retries bounded and observable; never repeat blindly or
  represent a failed check as passed.
- Preserve failed receipts and diagnostic artifacts as evidence. A corrected
  rerun must write a distinct receipt instead of overwriting the failure.
- Long-running prelaunch and debugging steps must state their expected
  duration, emit a concise progress update at least every 60 seconds, and use
  a defined hard timeout. If progress becomes unobservable or the timeout is
  reached, stop that step and report rather than waiting indefinitely.
- If bounded diagnosis does not resolve the cause, report the failure, the
  completed work, and the smallest proposed next action.
- Use a default pre-arm problem-solving budget of ten minutes or two failed
  diagnosis-backed repair attempts, whichever comes first. Stop immediately
  when the same error repeats unchanged. At the limit, summarize the blocker,
  completed work, valid artifacts, and safe options, then ask the user how to
  proceed instead of redesigning the control plane or continuing the loop.
- A candidate-cell rejection that the approved sweep explicitly treats as an
  independent scientific outcome is not a pipeline failure. Record it as
  negative evidence and continue only as authorized by the frozen plan; do
  not let it silently block or invalidate unrelated cells.

## Experiment Lifecycle
- For every new, continued, scheduled, or materially changed experiment,
  follow
  [`skills/run-experiment-pipeline/SKILL.md`](skills/run-experiment-pipeline/SKILL.md).
- For an ordinary diagnostic continuation or repeat with an immutable parent,
  use the simplified continuation path in that skill. Its validated
  `experiment-resolved-study/v1` document is the scientific plan; do not
  duplicate it as a legacy mixed Markdown plan or ask the user for hashes,
  receipt schemas, paths, environment versions, or runner identities.
- The simplified path requires one user review of the resolved scientific
  summary and execution proposal. One explicit `approved` response may bind
  both when both are shown together and unchanged. The executor then owns
  target routing, staging, paths, capability discovery, concurrency, compact
  receipts, tracker updates, bounded functional smokes, launch, immediate
  readback, monitoring, and collection without changing scientific fields.
- Use the full plan under `docs/experiment_plans/` for paper-facing work,
  novel scientific studies, adaptive sweeps, or requests whose scientific
  choices cannot be inherited exactly from a parent.
- Every explicitly planned or otherwise nonterminal experiment must have
  exactly one entry in `docs/current_experiments.md`. Keep its visible fields
  to `Testing`, `Where`, and `Status`; keep hashes, metrics, receipts, and
  interpretation in the study, result bundle, and result registry. On the
  simplified path this entry is generated and maintained by the executor; it
  is operational visibility rather than a second user-authored authorization
  contract.
- Full-plan production launches retain the exact-ID `launch-ready` tracker
  gate. Simplified continuation launches instead require the immutable
  authorized dispatch request and the matching passed aggregate functional
  preflight; they must still fail closed on missing or duplicate tracker state
  discovered by the executor, but the user is never asked to repair tracker
  syntax.
- Publish qualified measurements and interpretation through the existing
  comparison-card registry and generated `docs/results/index.md`; remove the
  tracker entry after review because the tracker is not permanent history.

## Experiment Launch Intake
- When the user asks to launch, run, schedule, continue, or materially change
  an experiment without supplying a completed
  [`docs/experiment_request_template.md`](docs/experiment_request_template.md)
  intake, stop before plan creation, tracker mutation, preflight, staging, or
  launch. Use read-only catalog and protocol inspection to prefill every value
  already known, present the compact conversational form, and ask the user to
  complete or correct only the remaining user-owned choices.
- Do not make the user restate values fixed by an identified parent; record
  `inherit from parent` and expose the resolved values in the agent summary.
  When every field is already known, present the fully prefilled form for one
  confirmation instead of bypassing intake.
- A previously completed request may be reused only when the user identifies
  it and says it is unchanged. Completing the intake is not scientific
  approval or execution authorization; those remain separate review fields
  after the agent produces the resolved study and execution proposal.
- While waiting for the completed form, do not continue launch preparation or
  attempt to resolve missing scientific choices autonomously.

## Experiment Catalog
- Resolve experiment requests through
  `python -m experiments.experiment_catalog resolve "<request>"` and inspect
  the linked authority before planning or launching work.
- Exception for an approved fixed continuation request: the Jean Zay
  validation fast path may resolve the named parent and exact case directly
  from the approved request without creating a new catalog entry for the
  validation attempt. The existing parent must still resolve through the
  catalog. Use `python -m experiments.jeanzay_validation launch ...`; its
  package receipt records `catalog_role=parent_discovery_only`.
- The catalog is routing metadata, not scientific authority or launch
  authorization. A catalog match still requires the applicable lifecycle
  above and user review.
- Feed completed structured intake to the compositional resolver. It may
  resolve an immutable parent plus an exact subset and classify differences
  into scientific-study changes and operational-attempt changes. A missing
  exact leaf entry is not an error when the parent, subset, and changes resolve
  unambiguously.
- A `resolved` status means only that routing succeeded. Follow the returned
  `next_step`. A `derived` result with no unresolved scientific choices may
  proceed to generated study review; it is not launch authority by itself.
- Never silently choose among an `ambiguous` result or substitute a suggested
  entry for `not_found`. Disambiguate the former; treat the latter as a new
  workflow, preserve choices already supplied by the user, and surface only
  the unresolved decisions. Add or amend the catalog entry as part of the
  approved workflow.
- Use the copy/paste request contract in
  [`docs/experiment_launch_request.md`](docs/experiment_launch_request.md).

## Conv Amplification Protocol Map
- Start with `docs/conv_paper_hyperparameter_protocol.md`.
- Read every protocol that the index marks active for the requested work.
- Use `docs/results/amplification_experiment_curation.md` before treating historical
  results as evidence.
- Read `docs/current_state.md` for the short user-approved direction and
  `docs/current_experiments.md` for verified operational state and result
  locations. Neither is an independent protocol or launch authorization.
- `docs/my_notes.md` is Filip's personal, non-authoritative notebook. Read or
  edit it only when Filip explicitly asks.
- Archived protocols are provenance only and cannot authorize new runs.

## Plotting Rule
- Use `matplotlib` for every generated plot (PNG/SVG/PDF) unless the user explicitly requests a different plotting backend.
- Prefer reusable plotting scripts under `labs/tools/` for plot generation commands.

## Data and Artifacts
Generated outputs live under folders like `simulation_results*`, `papers/`, and `labs/cases/` and are ignored by git.

Before copying or evaluating results produced on Akib, Trex, or Jean Zay,
follow
[`skills/sync-remote-results/SKILL.md`](skills/sync-remote-results/SKILL.md).
Pull a small evaluation snapshot first, resume the complete archive
independently, and make scientific conclusions only from local paths.

## Error Messages
- When validating inputs, state the expected format first, then echo the provided value on failure.

## Config Rules
- Do not silently default diode parameter dicts; require explicit `*_diode_param` dicts in config.

## Environment Issues
- OpenMP SHM error (`OMP: Error #179: Function Can't open SHM2 failed: System error #13: Permission denied`) can occur when running Python (e.g., matplotlib/numpy) in the sandbox. Fix options:
- Set env vars to disable shared memory: `KMP_DISABLE_SHM=1`, `KMP_SHM_DISABLE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`.
- Run the command outside the sandbox (escalated) if SHM is blocked.
- If you just need PNGs from existing SVGs, use `rsvg-convert` as a fallback.
