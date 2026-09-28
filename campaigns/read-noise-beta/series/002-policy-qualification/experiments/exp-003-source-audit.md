---
id: "exp-003"
title: "Audit beta source contracts, evidence and comparison coverage"
status: "blocked"
hypotheses: ["H-001", "H-002", "H-003", "H-004"]
---

# exp-003 — Source audit before another run

## Question and blocker
Which prior observations can be reproduced from the authoritative local worktree, and
which exact configs define the next comparison? Blocked here by unavailable untracked
local run bundles/checkpoints, not by the absence of a planning decision.

## Cases and execution
Read the source roots linked by exp-001 and exp-002. Reconcile each original arm, seed,
checkpoint, iteration count, noise protocol and environment with its resolved config,
metrics and source hashes. Recover the current baseline p99 beta; do not infer it from
legacy/ours. Audit the base-to-injected-beta conversion and the RMS definition.

Run the existing `python -m experiments.reporting validate-run RUN_DIR` only against
real included local bundles, retaining its output. This is not a whole-study coverage
check: separately reconcile the declared matrices, failed smokes, excluded cases and
replacement paths. Older noncanonical artifacts must be described, not fabricated into
successful canonical runs. No simulator or training invocation is needed.

## Acceptance and outputs
Produce results/exp-003-audit.md with a source table, original hashes, valid/missing
coverage, exact starting configs and named unresolved differences. No missing or
inconsistent row is silently included. Publish a clear ready/blocked decision for the
replay and training plans. The audit can corroborate records, but does not newly test
H-002 or establish a causal mechanism.

## Budget and stop
Proposed cap: two hours of local read-only audit, zero GPU time, no checkpoint mutation.
Stop at missing authoritative data or unverifiable config identity and name the blocker.
An execution task may resolve access and finish this audit; setup does not claim it ran.
