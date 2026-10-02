---
id: "exp-024"
title: "Final-epoch official-test correction for read noise"
status: "complete"
hypotheses: []
---
# Exp024 — Final-epoch test evaluation

## User correction and scientific contract

Filip clarified October1: “omg yes, we need final point”. This explicitly
supersedes the default best-validation checkpoint rule for this comparison.
Exp023 evaluated best-validation checkpoints; its immutable evidence remains
separate. The requested comparison uses final epoch10 for Conv1 and cumulative
final epoch30 for Conv2/3, matching the earlier fixed-final validation tables.

Retain exactly exp023's153 cases:27 clean controls,54 noisy seed0 cases across
eta1e-6/1e-5/1e-4/3e-4/5e-4/1e-3, and72 seeds1/2 at the middle four levels.
All three schemes and D/F reference1/4/6 remain unchanged. No training, retuning,
new seeds, outcome exclusions or checkpoint selection from test results.
Preserve historical20 seed0 Adam/RNG resets separately from uninterrupted means.

The official test split has already been used for exp023. This correction
explicitly records that prior exposure; it is an additional fixed-final evaluation,
not a newly untouched test set or compliance with the original once-per-training-run
best-checkpoint protocol. User intent takes precedence over that default rule.
Each new final checkpoint may be evaluated once; no partial read is silently retried.

## Coverage and reuse

All153 final checkpoint files match saved canonical hashes and declared horizons.
CPU comparison of every parameter tensor (dtype, shape, bytes, exact torch.equal)
found26 final models identical to their previously tested best models. Their
canonical test results are reused with exact-identity evidence; no fresh data read.
The other127 models receive one new official10000-image test pass:

| Target | Model | Existing identical results | New final evaluations |
|---|---|---:|---:|
| Akib RTX3080 | Conv1 | 6 | 45 |
| Local RTX3090 | Conv2 | 12 | 39 |
| nom-cool-1 RTX3090 | Conv3 | 8 | 43 |

Source training configs, clean evaluation, float64, zero biases, preprocessing,
T=4/6/8, amplification and frozen betas remain unchanged. Original capacity
receipts establish batch10000 for all schemes/targets, with validated inference
source hashes and exact capacity signatures. Carry those receipts forward without
another capacity search or model/test-based tuning. Check test files exist before
reserving test access. Explicit checkpoint-transfer approval from exp023 persists.

## Evidence, budget and review

Result root: `results/read-noise-final-test-20261001-v1/`, including all assets,
identity audits, reused canonical bundles, source, failures and new evaluations.
Final case list and contract are sealed before new tests. Source is exp023's frozen
snapshot plus the explicit final-mode evaluator and transport updates; preserve
inference files exactly. The reporting class records prior best-test exposure.

Budget6GPUh including transport/runtime overhead and any understood pre-read
recovery;24h deadline from first launch. Expected runtime under30min with the three
GPUs in parallel, based on exp023's measured throughput. Prepared group limits
60/60/90min sum3.5GPUh. Keep1GiB/10% measured-peak headroom and sharing policies.

Primary comparison: final validation versus final test accuracy for uninterrupted
seeds1/2, means/sampleSD and within-seed ours-minus-baseline/legacy differences.
Clean controls use seeds0/1/2; reset seed0 and other historical results stay labeled.
Report all153 cases with whether measured now or reused identically. Best-checkpoint
results are a separate comparison, never relabeled as final. No significance claim.
Expected review note: `../results/exp-024-final-test.md`.

## Execution and monitoring handoff

Root owns preparation/launch. The shared queue will own per-host execution and
collection after verified startup. Record exact IDs, source/contract hashes,
per-checkpoint prior exposure, collection commands and live ownership here.
The three new-evaluation groups have been submitted. A positive result is not a completion condition: require
all153 cases reconciled,127 new passes plus26 audited reuses, with zero invalid
or missing outcomes before final review and ledger refresh.

### Sealed and submitted

Contract SHA256:
`5e28132a7ecbeded2c54790cbd3f04a75908db50290e63ba448b1e0983e5411a`.
All127 new cases passed final-epoch/config/source/prior-exposure and capacity
carryforward guards. All inference files match exp023 byte-for-byte; only the
explicit final-mode evaluator and transport changed. Twenty-one CPU runner tests
pass. Metadata preflight confirmed official test files on all targets before
reservation. Remote contract hashes match; both approved transfers exited0.

Shared queue IDs: `exp024-test-local-20261001-v1` (39 new Conv2),
`exp024-test-akib-20261001-v1` (45 new Conv1), and
`exp024-test-nom-cool-1-20261001-v1` (43 new Conv3). Each has automatic
collection/local-validation completion hooks; root retains startup ownership
until process and semantic progress checks pass. Source/config identities and
full153-case inclusion set are immutable. [Final-case list](../../../../../results/read-noise-final-test-20261001-v1/preparation/final-test-list.csv).

### Verified monitoring ownership — October1, 15:05 Paris

The shared queue is now the sole execution/monitoring owner for these three IDs.
Service PID1124310 is alive with a fresh observation and no host errors; its
60-second loop retains collection hooks. Direct canonical status checks showed
25/39 local,39/45 Akib and6/43 nom-cool-1 evaluations completed, with all three
groups advancing. Root's startup ownership ends here; no second polling owner
or automatic test retry is active.

The collector's real-artifact audit validated all153 final checkpoint/config
identities, all26 reused outcomes and the first8 new local outcomes with zero
errors. Eight analysis semantic checks passed, including rejection of wrong
selection, missing prior-exposure disclosure and modified reuse evidence.
Automatic collection runs `analysis/collect.py --target TARGET --allow-partial`
inside this result root. Final review requires `analysis/analyze.py` without
`--allow-partial`, all153 outcomes and completed worker membership39/45/43.
Collection and validation are automatic; scientific review remains a separate
step. Chat push delivery is not configured.

### Completion and review — October1

All three queue IDs completed execution, collection and validation. The last
worker finished15:22:01 Paris. Strict local analysis passed153/153 outcomes,
127 unique new evaluations and26 audited reuses, zero pending cases/groups and
zero errors. Measured worker time1868.742seconds (0.5191GPUh); no exp024 retries.
Root completed scientific review in the [result note](../results/exp-024-final-test.md).
This study no longer needs monitoring; leave the shared service and unrelated jobs
untouched. The requested endpoint remains final epoch10/30/30.

### Post-completion display update — October1

User requested “include all three seeds”. Added descriptive seeds0/1/2 tables
and plots from108 existing validated final outcomes, with explicit mixed-history
labels for12 groups. No new compute/test read or change to the frozen evaluation
contract. See the result note's all-three-seed section; retain the original matched
uninterrupted seeds1/2 comparison as a separate view.
