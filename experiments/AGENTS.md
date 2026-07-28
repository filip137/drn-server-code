# AGENTS

## Scope
This guidance applies to experiment scripts, launchers, collectors, monitors,
and result bundles under `experiments/`. It supplements the repository-root
`AGENTS.md`.

## Protocol Authority
- For Conv amplification work, start with
  `../docs/conv_paper_hyperparameter_protocol.md` and follow the protocols that
  it marks active for the requested evidence scope.
- `../docs/conv_paper_experiment_definition.md` owns the paper dataset,
  preprocessing, architectures, output encoding, loss, nonlinearities, and
  amplification schemes.
- `../docs/conv_paper_hard_sigmoid_input_gain_protocol.md` owns the
  hard-sigmoid-only input-gain calibration contract, including its target and
  freeze rule. Do not apply that target or procedure to perfect diode.
- `../docs/conv_paper_tk_protocol.md` owns the paper operational `T/K`
  selection rules within the scope it explicitly declares.
- `../docs/conv_paper_learning_rate_protocol.md` owns the paper hard-sigmoid
  learning-rate handoff within the rows and optimizer contract it explicitly
  declares.
- `../docs/results/amplification_experiment_curation.md` owns the evidence disposition
  of historical results. `../docs/current_state.md` is the user-controlled
  agent handoff and `../docs/current_experiments.md` is the operational
  tracker; neither is a protocol.
- Do not copy numerical protocol values or completion-status snapshots into
  this file. Read them from the active protocol documents at execution time.

## Evidence Scope
- Every launcher and manifest must identify its intended evidence scope as
  paper-facing, diagnostic, or historical replay.
- Collectors and summaries must preserve that intended scope together with any
  later valid, diagnostic, excluded, or superseded disposition assigned by the
  active protocols and curation ledger.
- A paper-facing run must resolve exactly to the active experiment definition
  and every applicable downstream protocol.
- Ordinary-MNIST and other off-protocol diagnostics must be labeled explicitly
  and must not be promoted to paper-facing evidence by a launcher or worker.
- Collectors must not silently combine rows with different datasets,
  initializers, gains, `T/K`, optimizers, learning rates, epoch budgets, or
  checkpoint rules.

## Launching
- Preserve the same launcher/config contract across local tmux lanes and Jean Zay whenever possible.
- For large batches, follow the root `AGENTS.md` launch policy for checking `tmux main`, `tmux akibscomputer`, `tmux trex`, and then using Jean Zay R3 `fmu@v100` for the remainder.
- Keep Jean Zay result outputs under `/lustre/fsn1/projects/rech/fmu/$USER/server_code/results` unless the user explicitly requests another project/account.
- Dispatch already approved long simulations on `tmux main` or Trex through
  `experiments.local_dispatch` and its checked-in profiles. Preserve its
  schema-validated preflight, one-SSH, fresh-window/session,
  immutable-attempt, and no-fallback contract.

## Fail-Stop Implementation
- Apply the root Long-Run Simulation Fail-Stop-Report policy only after a
  long-run attempt is explicitly armed. Ordinary implementation work, tests,
  environment setup, plan-only commands, and bounded developer probes may be
  diagnosed and corrected in the same turn.
- A controller supervising an armed long-run attempt may catch a first
  unexpected failure only to persist one immutable failure report and a
  terminal failed state, then exit nonzero. Catch-and-continue and automatic
  corrective retries are prohibited inside that armed attempt.
- Long-run polling and reconciliation must propagate the first exception and
  use a hard deadline. Default CLI behavior must be one bounded reconciliation
  pass; persistent following must be explicit and bounded.
- Before any scheduler side effect for an armed long-run attempt, reject an
  existing terminal failure report or failed state. A failed state cannot
  submit or resume. A new attempt requires a new state/receipt path and a new
  user message after the prior report.
- For a legacy/full-plan attempt, keep the exact tracker entry `preflighting`
  through the payload canary and
  require that state immediately before its scheduler side effect. Stop at
  the successful canary boundary. Only a later invocation may launch
  production, and immediately before that production scheduler side effect it
  must validate the canonical tracker entry as `launch-ready`. Remote
  controllers consume a short-lived, hash-bound receipt created by the
  canonical tracker validator; they must never treat a staged Markdown copy
  as current authority. After live production readback, the external workflow
  changes it to `queued` or `running`. Within an armed attempt, missing,
  malformed, duplicate, blocked, or stale entries are terminal pre-launch
  failures.
- A simplified diagnostic continuation does not use this legacy
  canary/production tracker-receipt split. Its functional dispatcher requires
  the approved generated study/proposal, one passing smoke per target, the
  matching aggregate preflight, and an executor-generated tracker entry.
- The following is a legacy full-plan rule for the existing perfect-diode
  Conv1/Conv2 Jean Zay successor only; it is not the launch path for a new
  simplified diagnostic continuation. Its staged source and
  bootstrap validation is part of its armed long-run launch gate and must run
  exactly once through
  `validate_mnist_conv_perfectdiode_successor_staged_jeanzay.sh`. Do not
  reconstruct or selectively retry its SSH subcommands.

## Large-Run Preflight Implementation
- Follow the root large-run preflight policy for every long or large batch,
  whether immediate or scheduled.
- The execution smoke test must use the same public launcher, configuration
  resolution, environment, device type, and output-writing code as the full
  run. Use a separate disposable output directory so smoke artifacts cannot
  contaminate production results.
- A training smoke test must construct the real dataset and model, execute the
  equilibrium phases, compute the real loss and gradients, complete at least
  one optimizer step, and write the expected checkpoint or result artifact. A
  parser check, import check, plan, dry run, or `--help` invocation is not
  sufficient.
- For a non-training experiment, the smoke test must execute the experiment's
  principal numerical computation and write its expected result artifact.
- Reject non-finite values, missing artifacts, schema failures, or a nonzero
  exit status. Do not launch the full workload after any preflight failure.
- Every unique resolved architecture x nonlinearity x amplification
  configuration in a Conv amplification batch must have a passing scientific
  `T/K` reference gate under the active protocol for that evidence scope.
  Hard-sigmoid criteria must not be reused for perfect diode.
- A true continuation may reuse its recorded accepted `T/K` result when
  learning rates and the operating point are fixed and no architecture,
  nonlinearity, amplification, input gain, `T/K`, or equilibrium/gradient
  semantic changes. A host, epoch extension, path, interpreter, library, CUDA,
  or GPU change alone requires a new target smoke, not a new scientific gate.
- For perfect-diode clamped hidden states, use projected KKT residuals. Raw
  residuals are diagnostics for clamped states, not the convergence gate.
- For paper-facing or explicitly reproducibility-critical work, reuse a
  preflight only when its strict recorded identities still match. For ordinary
  diagnostics, keep scientific inputs exact but treat Python, PyTorch, CUDA,
  GPU, paths, and operational adapter code as provenance: rerun the real
  one-batch smoke and admit the target when required capabilities, finite
  numerics, checkpoint writing, and result schema pass.
- Scheduled runs must also satisfy
  `../skills/scheduled-run-preflight/SKILL.md`.

## Reproducibility And Provenance
- When an active calibration protocol requires independent model builds, reset
  global layer and parameter name counters before each build and seed the
  shuffled loader independently of model RNG consumption.
- Preserve the resolved config and enough provenance to reproduce and classify
  every result: code identity, dataset and transform, model and data seeds,
  architecture, output encoding, nonlinearity, amplification values,
  initializer and bounds, input gain, `T/K`, optimizer contract, device and
  host, evidence scope, and preflight receipt.
- Calibration results must additionally record the calibration configuration,
  target definition, selected value, and every required layer measurement.
- Slurm runs must record the job and array-task identifiers, account, resource
  profile, remote result path, and intended local destination.

## Execution And Jean Zay Results
- Follow the root compute-allocation policy. Keep the launcher and resolved
  scientific config identical across local tmux and Jean Zay execution.
- Follow `../docs/jean-zay.md` for the current SSH alias, project/account,
  storage paths, monitoring, and result-repatriation procedure.
- Use `../skills/sync-remote-results/SKILL.md` for Akib, Trex, and Jean Zay
  transfers.
- Treat Jean Zay result storage as temporary staging, not as the permanent
  result archive. Before submission, assign both a run-specific remote output
  directory and its intended local destination.
- A Jean Zay run is not complete until its result bundle, resolved configs,
  manifests, checkpoints, metrics, and Slurm logs have been copied back and
  validated locally.
- Before transfer, require a terminal Slurm state and a passing remote
  collector, completion marker, or experiment-specific validator. A
  metadata-only evaluation snapshot may arrive first, but final archival
  acceptance requires the entire self-contained run or study root.
- Copy into a local staging directory first. Verify the transfer and run the
  experiment-specific collector or integrity validator before publishing the
  bundle into the local result store.
- Write a transfer receipt under the local result store containing the remote
  and local paths, job identifiers and terminal states, content identity, file
  count and bytes, transfer and verification commands, local validation
  result, and timestamp.
- If transfer or validation fails, keep the exact remote run directory,
  classify the run as incomplete, and retry the transfer. Do not treat
  remote-only results as completed or paper-facing evidence.
- Never use a failure-masking transfer such as `scp -r ... || true`.
- After the local copy passes validation, perform remote cleanup as a separate
  recorded step and remove only the exact run-specific remote directory.
  Never delete a shared Jean Zay result root, source checkout, or another
  run's output.

## Operational Tracking

- Before a new or materially changed experiment launch, follow
  `../skills/run-experiment-pipeline/SKILL.md` and require its approved,
  validated plan.
- Full-plan workflows retain the exact-ID `launch-ready` tracker validator
  before their first production side effect. A simplified ordinary-diagnostic
  continuation uses its authorized immutable dispatch request plus matching
  passed functional-preflight result as launch authority; the executor
  creates and validates the concise tracker entry automatically rather than
  making tracker syntax a user-facing gate.
- As part of an explicitly requested launch, monitoring, transfer, or
  validation workflow, update `../docs/current_experiments.md` after each
  verified state transition, reusing the same experiment marker and the
  `Testing`, `Where`, and `Status` fields. Do not put run status in
  `../docs/current_state.md`.
- Generated monitors and reporters write detailed output beside the result
  bundle. They must not append directly to either concise document.
- Record terminal remote execution as `remote-closed`, not complete, until
  transfer and local validation pass. Keep the entry through
  `review-pending`; remove it only after the reviewed result is published.
