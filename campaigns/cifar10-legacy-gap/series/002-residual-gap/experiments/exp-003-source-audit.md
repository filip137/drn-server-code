---
id: "exp-003"
title: "Audit the endpoint comparison and complete coordinate mapping"
status: "blocked"
hypotheses: ["H-001", "H-002", "H-003", "H-004", "H-005"]
---

# exp-003 — Source and comparability audit

## Question and access blocker
Recover a trustworthy comparison table and exact configs from the authoritative local
research worktree. Untracked bundles/checkpoints are not available in this setup.
No new training or replay is required to perform the audit once access is available.

## Cases and checks
Reconcile the original baseline/ours/legacy 10→30→50 histories, normalized-legacy
continuation and the 200-minibatch mapped-coordinate identity test. Read resolved configs
and source hashes, not approximate LR vectors in prose. Check BN state initialization,
normalization sites, epsilon, gains, full Adam/scheduler/RNG state, split and batches.
Recover the exact equality/solver tolerances and code for the identity test.

Record hardware/software differences and whether each historical seed0 reference can
be included in a new matched environment. Reconcile all declared cases, smokes, failed
no-BN runs, unqualified results, exclusions and replacements. Use the existing
`python -m experiments.reporting validate-run RUN_DIR` on actual canonical included
bundles and save its output. Describe older noncanonical evidence without inventing
successful bundle files. Validation of a subset is not full expected-coverage validation.

## Acceptance and outputs
Write results/exp-003-audit.md containing original IDs/hashes, exact config/command paths,
coverage, what is comparable, what remains missing and a go/no-go decision for exp-004
and exp-005. Distinguish normalized-legacy-with-legacy-rates from the mapped-baseline
optimizer control. Audit corroborates sources; it does not newly test multi-seed H-003.

## Budget and stop
Proposed cap: two hours read-only local audit, zero GPU time. Stop at missing source
identity or irreconcilable coverage and name the blocker. Never edit checkpoint bytes,
restart old launchers or overwrite source interpretations to make the table agree.
