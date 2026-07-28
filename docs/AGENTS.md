# AGENTS

## Scope
This guidance applies to documentation under `docs/`. It supplements the
repository-root `AGENTS.md`.

## Document Authority
- `conv_paper_hyperparameter_protocol.md` identifies the active paper
  protocols and current paper gate.
- Each active protocol owns its exact scientific definitions, numerical
  values, selection rules, and protocol-specific status.
- `results/amplification_experiment_curation.md` owns evidence disposition.
- `results/conv_learning_rate_diagnostics.md` owns historical and ordinary-MNIST
  learning-rate diagnostic curation.
- `current_state.md` is the short, user-controlled agent handoff.
- `current_experiments.md` is the agent-maintained operational tracker.
- `experiment_plans/<experiment-id>.md` is the user-approved, durable study
  design linking the hypothesis to the executable config and manifest for the
  full paper-facing or novel-study lifecycle.
- `experiment_requests/` holds user-facing intake and review for the
  simplified continuation lifecycle. Its generated
  `experiment-resolved-study/v1` artifact is the durable scientific plan for
  an ordinary diagnostic continuation.
- `my_notes.md` is Filip's personal, non-authoritative research notebook.
- `remote_result_workflow.md` owns the shared Akib, Trex, and Jean Zay
  transfer and local-evaluation workflow.
- `results/index.md` is generated from `../result_registry/`; do not edit its
  measured values or conclusions directly.
- Keep live direction and execution status in the two `current_*` files.
  Keep durable measurements, interpretation, and dated diagnostics under
  `results/`; never use a result document as a live status ledger.
- An active protocol may retain the compact selected values and small
  execution-input tables it owns. Put the supporting measurements and broader
  interpretation under `results/` and link them rather than duplicating them.
- Top-level moved-document pointers are compatibility aids only. Add new
  conclusion content at the canonical path under `results/`.
- Prefer links to authoritative documents over copied tables or duplicated
  numerical values.

## Agent-Handoff Ownership

- Treat `current_state.md` as user-controlled. Edit it only when the user
  explicitly asks to add, remove, correct, or summarize current state.
- Preserve user-written direction faithfully. Do not turn tentative
  discussion, an agent suggestion, or an inferred next step into a recorded
  user decision.
- Keep it to a short repository goal, current priority, immediate next action,
  open blockers, and links. Do not add simulation tables or numerical result
  summaries.
- Do not update `current_state.md` automatically as a side effect of launching,
  monitoring, completing, analyzing, or documenting a run.

## Personal-Notes Ownership

- Read, interpret, or edit `my_notes.md` only when Filip explicitly asks.
- Never treat personal notes as a protocol, verified result, launch
  authorization, or permission to resolve an open choice.
- When Filip asks to record a note, preserve its status as a hypothesis,
  observation, question, or decision rather than silently strengthening it.

## Experiment-Tracker Ownership

- Agents may update `current_experiments.md` as part of an explicitly
  requested launch, monitor, transfer, validation, or result-recording task.
- Keep exactly one entry for every experiment that is explicitly planned,
  preflighting, launch-ready, queued, running, blocked, remotely finished,
  transferring, validating, or awaiting review. Remove it after its reviewed
  result is published; archive stale tracker documents rather than building a
  permanent scheduler log.
- Give each entry one invisible `<!-- experiment-id: ... -->` marker and only
  three visible fields: `Testing` for the concise purpose/comparison and plan
  or protocol link; `Where` for host/job or tmux identity plus compact bundle
  locations; and `Status` for one lifecycle token, last verified progress, and
  the next action or blocker.
- Keep manifest and source hashes, receipts, detailed metrics, and
  interpretation in plans, protocols, result bundles, and registry records,
  not in the tracker.

## Experiment-Plan Ownership

- Create plans through `../skills/run-experiment-pipeline/SKILL.md`.
- Treat the canonical config and immutable manifest as numeric execution
  authority; the Markdown plan records intent, identities, decision rules, and
  destinations.
- Require Filip's explicit approval before launch. After launch, do not edit
  scientific choices in place; create an explicit successor/amendment with a
  new identity.
- Do not use an experiment plan as a live status log or a result conclusion.

## Simulation Status Rules

- Use `queued` or `running` only after a live check or an explicit
  timestamped user report. Old monitor output does not establish current
  execution state.
- Distinguish execution state from scientific acceptance. A completed process
  may still be diagnostic, failed selection, excluded, or pending review.
- A Jean Zay run is not complete while its only validated copy remains remote.
  Track transfer and local validation according to `jean-zay.md`.
- If current execution state was not checked, say so explicitly rather than
  inferring that no jobs exist.
- Generated report scripts must never append to `current_state.md` or
  `current_experiments.md`. Write generated detail beside the result bundle;
  update the tracker separately with a concise, verified summary.
- A full-plan production launch must fail closed unless the pipeline tracker
  validator finds the exact experiment ID once and its status is
  `launch-ready`. A simplified diagnostic continuation uses its immutable
  authorized dispatch request and passed aggregate functional preflight; its
  tracker entry remains required operational state but is generated and
  repaired by the executor rather than the user.

## Editing Conventions

- Update each document's date only when its content changes under the ownership
  rules above.
- Scope every decision and blocker explicitly as paper-facing,
  ordinary-MNIST diagnostic, or another named evidence category.
- Archive historical trackers under `archive/` with a prominent warning that
  their execution states are historical and unverified.
