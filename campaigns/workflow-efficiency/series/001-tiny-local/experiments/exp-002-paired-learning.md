---
id: "exp-002"
title: "Compare tiny-network early learning and audit the complete workflow"
status: "complete"
hypotheses: ["H-001", "H-002"]
---
# exp-002 — Amplification and early learning

## Explorer: question and decision

Filip accepted the question: does bidirectional amplification improve early learning under matched initialization, data and optimizer settings? Test H-002 using baseline v1/c1 versus ours v4/c1. Same fresh seed0 initialization, batches/order, architecture, optimizer and raw LR vector. A useful outcome can be positive, negative or inconclusive; workflow success does not require amplification to win.

Primary r=(L_epoch1−L_epoch4)/L_epoch1 on the fixed sixteen-example validation prefix; D=100*(r_ours−r_baseline). Support H-002 when D>2pp, contradict when D<−2pp; otherwise inconclusive. Strict, arbitrary pilot threshold frozen before execution. Report absolute losses and accuracy as secondary descriptive metrics, not paper evidence. Eight steps precede the first measurement; no initialization-loss measurement is claimed.

## Frozen contract and Codifier assignment

Two-hidden-channel Conv1, signed28×28 MNIST, paired20 outputs, kernel3/stride2/padding1, perfect diode with explicit dictionaries; finite-step BP T=K=2, gain1, float32, wide weights[0,100], fresh Kaiming. Adam rates[1e-4,1e-4,0] in ConvWeight/DenseWeight/Bias order, fresh state, bias LR0. Four truncated epochs × eight batches × two examples =32 updates per arm. Fixed16-example validation after each epoch. Ordinary deterministic55k/5k split; never read official test. No calibration, gradient replay or scientific pre-launch gate.

Use configs/conv/workflow_tiny_{baseline,ours}_20260925.json via experiments.exact_run, --device cpu --skip-terminal-official-test. Same-path --smoke first for each config (one training and validation batch). No runner changes, broad tests, synthetic delays or new launch framework. One two-config command per phase so module/config overhead is observable. Built-in initial_parameter_state_sha256 and dataset_provenance establish matching; no extra checkpoint-every-epoch or gradient tracing.

Result root: results/experiment-loop-tiny-audit-20260925-v1/. analysis/paired_paths.json binds exact config hashes and bundle paths. Evidence class workflow_diagnostic. Preserve unexecuted exp-001 assets separately; they are not failures or replicates.

## Bounds and audit

Design frozen 2026-09-25T12:07:26.996564+00:00; deadline 2026-09-25T12:22:26.996564+00:00. Expected each phase seconds to a minute; cap each smoke/production invocation at180seconds, total compute10minutes including at most one operational retry. Stop on nonfinite scientific failure with no tuning. CPU threads1; no GPU, remote allocation or unrelated-job changes. Overall target15minutes after design freeze.

For H-001 record design-to-review elapsed time, subprocess interval union and remaining elapsed overhead. Record narrower epoch runtimes separately. Log role events, tool/read repetition, Monitor preparation/acknowledgment, validation time, artifacts and unsuccessful commands. User-discussion waits and abandoned setup are separate. Instrumentation and new campaign scaffolding are disclosed costs. Audit only in this pass: recommendations are not automatically applied.

## Execution and monitoring handoff

Coordinator owns Explorer, Codifier, Reviewer and shared dashboard. Separate Monitor exercises the role handoff; no separate Codifier worker required by skill. Coordinator runs synchronous smoke, launches production once and verifies initial progress, then sends process/session handles. Monitor must acknowledge exact paths/deadline/next check. No missing-output polling before launch; inspect only these bundles and bounded logs. If already complete, report completion-only handoff. Otherwise inspect at most every15seconds until exit; avoid a persistent watchdog for a seconds-long bounded invocation under active session ownership.

Monitor validates four terminal bundles once each, reconciles both smokes and both32-step arms, initial/split/order hashes, finite metrics and official_test_read=false; writes factual result note with verdicts{} and analysis/monitor.json. No code edits, relaunch, dashboard refresh, interpretation or unrelated scans. Escalate any failure to coordinator. Coordinator begins review only after receipt of evidence, then updates same result note and manifest. Existing commands: py312 -m experiments.reporting inspect-runs RUN1 RUN2; py312 -m experiments.reporting validate-run BUNDLE. Invocation logs/exit and timing are under analysis/. Source identity is the runner's current git+dirty-tree record; relevant file hashes accompany phase timing. This is local collection, with no redundant copy.

### Executed handoff

Both smokes completed in9.6846s with exit0. Production handle: unified exec session8147. Initial baseline status was running/starting at12:09:02.525440Z; ours starts sequentially. Exact commands, four source hashes and phase start/end/exit records: analysis/{smoke,production}-invocation.json. Logs and exitcodes share those phase names. Sandbox PIDs are namespace-local; the exec session is the outer handle. Monitor acknowledged paths/deadline before production and received inspection ownership after initial status publication. Coordinator collects only tool exit, with no duplicate artifact polling.

### Review handoff and completion

Monitor ready12:11:51.440850Z; Reviewer started12:12:25.035260Z. Four bundles passed validation once; scientific H-002 inconclusive, workflow H-001 supported. See the same-series exp-002 result note. No additional run is queued.
