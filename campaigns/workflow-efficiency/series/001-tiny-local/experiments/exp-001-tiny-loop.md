---
id: "exp-001"
title: "Time one tiny real experiment-loop lifecycle"
status: "superseded"
hypotheses: ["H-001"]
---
# exp-001 — Tiny CPU lifecycle

**Scope correction before launch:** Filip requested starting with the scientific research question. The implementation below is an unexecuted proposal, pending Explorer discussion. No smoke or training has run. The original execution deadline and Monitor assignment are cancelled. Do not launch this config until the research question and resulting design are settled. Record premature implementation/delegation as audit evidence.

## Cases and execution contract

One seed-0 perfect-diode Conv1 baseline, BP finite-iteration BPTT, two hidden channels, signed MNIST input, paired 20-output loss, T=K=2, input gain1, Adam weight rates1e-4 and frozen-zero bias LR. Fresh Kaiming initialization. Real deterministic 55k/5k split; only two epochs × four training batches × two examples; one two-example validation batch each epoch. A separate smoke uses one training and one validation batch. All official-test access disabled. No scientific accuracy comparison or convergence claim.

Config: configs/conv/workflow_tiny_conv1_20260925.json. Runner: experiments.exact_run, CPU, py312 interpreter. No remote or GPU allocation. Use existing reporting bundles and local subprocess handles. Copy explicit diode dictionaries from the existing config; remove inherited calibration/checkpoint/handoff claims.

## Budget and decision

Expected subprocess runtime: seconds to a minute each; cap each invocation at180seconds, at most one operational retry, total compute cap10minutes. Overall review target20minutes from contract creation. Stop on numerical failure; do not tune. H-001 compares non-overlapping contract-to-review wall time outside measured smoke/training against subprocess time. Capture redundant reads/checks, handoff latency, artifact work and errors. Do not change production workflow during the baseline measurement.

Results: results/experiment-loop-tiny-audit-20260925-v1/. Machine-readable events and subprocess timings stay in analysis/. Reviewer interprets the same result note; no paper/test read.

## Execution and monitoring handoff

Coordinator owns preparation and synchronous smoke, then launch and initial semantic check. Monitor is delegated before the tiny run so it can acknowledge paths and deadline; no artificial training delay. Coordinator remains owner until acknowledgment. Monitor inspects once at handoff, then at most every15seconds if still running, sooner on process exit; check only this study. A run that completes before handoff is a completion-collection handoff, not proof of live monitoring.

Handles, exact commands and identities are appended below by Codifier. Monitor must preserve failures, reconcile one smoke plus one mini run, run terminal validate-run once per bundle and write factual results with empty verdicts. No code edits, retries, scientific verdicts or shared dashboard writes by Monitor. Coordinator is sole dashboard writer and handles escalation. Deadline: contract creation plus20minutes; no detached unattended continuation required.

Superseded before any execution by exp-002 after research-question agreement. Original configs and timing events are retained as workflow evidence.
