# TODO

- Add numerical safeguarding in the minimizer/training loop to prevent runaway state divergence.
- Add explicit non-finite guards (`isfinite`) immediately after each equilibration step and stop early with a clear diagnostic.
- Add a state-magnitude safety cap (or clamp) during debug mode to detect and block blow-up before `inf`/`nan` propagation.
- Add a config-level stability profile (lower `nudging`, safer learning rates) for deep/2-hidden double-diode runs.
- Log first divergence location (layer name, iteration index, max `|state|`, max `|b|`) to speed up root-cause analysis.

## Simplified experiment flow — implemented 2026-07-28

- [x] Compositional catalog resolution of an immutable parent, exact subset,
  scientific changes, operational changes, and genuinely unresolved choices
  in `experiments.experiment_catalog`.
- [x] Structured request parser and aggregate validator in
  `experiments.experiment_request`, using
  [`experiment_request_template.md`](../docs/experiment_request_template.md)
  and the six-run continuation as its first real fixture.
- [x] Scientific-only generated study and aggregate validator in
  `experiments.experiment_study`. Users review scientific intent, cases,
  inherited/changed fields, seeds, budget, decision rule, and evidence scope;
  hashes and receipt plumbing are generated.
- [x] Ten-minute/two-failed-repair pre-arm budget with immediate stop on a
  repeated unchanged error in `experiments.prearm_problem_budget`.
- [x] Capability-based executor, local/Trex/Akib profiles, exact job
  ownership, adaptive concurrency, generated paths, and separate validation
  and production plans in `experiments.experiment_executor`.
- [x] Scientific identity separated from host, environment, runner, output,
  deadline, and receipt provenance. All three target attempts share one
  scientific study hash.
- [x] Functional compatibility for ordinary diagnostics: Python, PyTorch,
  CUDA, GPU, and code versions are provenance; the real numerical smoke is the
  admission check. Strict replication remains available for paper-facing or
  reproducibility-critical work.
- [x] Accepted `T/K` reuse for a true continuation with unchanged
  `T/K`-relevant science; target or epoch changes require a new smoke but not a
  new scientific gate.
- [x] One common validation/production job adapter and bounded supervisor for
  local, Trex, and Akib in
  `experiments.run_mnist_conv_perfectdiode_job` and
  `experiments.functional_dispatch`, including one profile-driven local/SSH
  `submit` command with immediate readback and no automatic retry.
- [x] Executor-owned concise tracker generation and validated atomic updates in
  `experiments.experiment_tracker`; tracker Markdown is no longer a user task.
- [x] A same-path one-batch numerical smoke that performs equilibrium phases,
  loss, gradients, one optimizer step, finite checks, a disposable checkpoint,
  and output-schema validation.
- [x] Compact generated functional-preflight, per-job completion, per-attempt
  completion, progress, dispatch, and first-write failure receipts.
- [x] Active policy cutover in root/nested `AGENTS.md`, the experiment pipeline
  skill, launch documentation, and request template.
- [x] Golden six-run fixture covering intake, parent/subset resolution, study
  validation, one approval boundary, three validation dispatch requests, six
  production jobs, exact host ownership, no fresh `T/K`, idempotent
  publication, process readback, failure stop, and completion aggregation.

Verification on 2026-07-28: 102 focused fast-path/control-plane tests and 110
surrounding tracker, dispatcher, scheduling, live-document, and fail-stop
policy tests pass.

## Simplified experiment flow — live acceptance remaining

- [ ] After Filip approves the filled six-run request, stage the current
  adapter and immutable parent bundle as an executor-owned operational step,
  then run one bounded one-batch smoke on local, Trex, and Akib.
- [ ] Pull the two disposable remote validation bundles locally and validate
  the result round trip before arming the six production jobs.
- [ ] Generalize the numerical job adapter beyond the current perfect-diode
  Conv continuation family when a second experiment family needs the fast
  path.
