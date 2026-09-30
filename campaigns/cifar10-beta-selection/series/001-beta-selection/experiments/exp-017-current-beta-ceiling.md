---
id: "exp-017"
title: "Current-batch beta selection with a higher ceiling"
status: "complete"
hypotheses: ["H-013"]
---
# Is early clipping limiting the current-batch method?

Completed and reviewed September30 05:37 Paris:28.04%, CE1.995131, with zero
RMS tolerance misses. Both learning gates fail; no continuation. See the
[result](../results/exp-017-current-beta-ceiling.md).

Repeat the single fresh ours arm in [exp-015](exp-015-current-batch-beta.md),
changing only `beta_search.beta_bounds` from[1e-8,1e4] to[1e-8,3e4]. Preserve
initializer, fresh Adam, seed0,45k/5k data/order/augmentation, batch32/eval16,
original learning rates, trainable BN/gains/head,50-epoch schedule, T/K[6,6,4],
RMS targets[.02,.035,.001], initial0.1x beta and five-pair selection rule.
No noise, smoke, solver audit or new control rerun. Use the same frozen numerical
implementation and matched runtime as exp-015, with new case/input identity.

Run one full epoch. Reuse native37.06%, CE1.726060 and current-batch32.64%,
CE1.788320 controls. The unchanged extension gate is accuracy>=35.06% and
CE<=1.812363. Report clipping/tolerance misses alongside accuracy and CE; a
mechanistic improvement without passing learning quality does not permit an
automatic extension. If it passes, continue exact state under the existing
milestones and allowance; otherwise review before another search change.

## Execution and monitoring handoff

Root: `results/cifar-beta-current-batch-high-cap-20260930-v1/`.
One local3090 or nom-cool-1 job with torch2.5.1+cu121/privatecuDNN9.1,
matching the controls. Prefer the now-free localGPU. Allow5,400seconds,
expected about50–65minutes from the exp-015 measurement. Fund1.5GPU-hours
by reducing unused ours epoch50 allowances14h→13.5h across at most three
branches, preserving the combined141.5GPU-hour ceiling. Deadline October2
08:00 Paris. Shared queue owns launch/collection/recovery; include this root
in the existing half-hour summary. Record accepted ID and startup here.


### Frozen launch and ownership (September30, 04:40 Paris)

Accepted queue ID: `cifar-beta-fresh-current-batch-high-cap-e1-20260930`.
Prepared job: `results/cifar-beta-current-batch-high-cap-20260930-v1/launch/current_batch-1.json`.
The sole case remains `current_batch`; versioned small configs are in
`configs/cifar/beta_current_batch_high_cap_20260930/`. The source and all inputs
were copied from exp015's frozen root. All404 source files and source identity
remain byte-identical; only `cases.current_batch.beta_search.beta_bounds[1]`
changes10000→30000, with the corresponding case/input identity refreshed.
Initializer, config, split, diagnostic cohort and study runtime are byte-identical.
The frozen implementation's existing67-test validation remains applicable;
no numerical source changed and no GPU smoke was run. CPU case validation,
all source/input hashes and private runtime library hashes passed. Actual CPU
runtime inspection confirmed torch2.5.1+cu121, CUDA12.1, cuDNN90100.
Evidence: `analysis/preparation.json` and `analysis/preflight.json`.

Placement is the currently free local RTX3090, with16GiB reservation plus2GiB
headroom. The job has a5400second allocation and the existing October2 08:00
Paris deadline. Nine unsubmitted ours5090 stage50 specs now cap at48,600seconds
(13.5hours); at most three promoted branches reclaim the5400seconds used here.
Earlier charged allocations remain included in the unchanged141.5GPU-hour ceiling.

The shared run-watch queue remains the only operational owner. The unchanged
comparison-only collector verifies full one-epoch coverage, exact initializer,
Adam/BN steps, beta-search history and expected runtime before recording completion.
Its complete state means validated measurement, not a passing learning-quality
claim. The existing30-minute summary service includes this new root and retains
all previous roots. Review the declared accuracy/CE gate before any continuation.


### Preserved pretraining failure and bounded retry

The first queue attempt `0e6444b12ab949cf800053c91c5b4503` failed CUDA driver
initialization before any optimizer update, checkpoint, or training metric. The
worker was verified dead/terminal and held before recovery. Its3.24675155seconds
are charged as4seconds, leaving5,396seconds. Original evidence is preserved under
`failed_attempts/current_batch/0e6444b12ab949cf800053c91c5b4503/`, including the
failed bundle, queue receipt and bounded driver-error log. This is not a numerical
or scientific rejection of the higher ceiling.

Retry root: `results/cifar-beta-current-batch-high-cap-retry-20260930-v1/`.
Retry ID: `cifar-beta-fresh-current-batch-high-cap-e1-20260930-retry1`.
Its source and inputs are byte-identical to the original exp017 root. The frozen
existing `collect_cifar_beta_retry.py` validates the retry before atomically
copying it to the empty original canonical cell, writing both receipts and invoking
the original comparison-only policy. The terminal original job was cancelled only
after retry preparation. No competing monitor or additional scientific allowance
was introduced.

A host-level availability-only check with the identical queue environment found
CUDA available on the local RTX3090 (driver575.57.08,23,237MiB free). The restricted
tool-sandbox check returned unavailable and is not treated as host evidence.
There was no GPU computation, reset, driver/library change, or smoke. The exact
cause of the transient worker initialization failure remains unproven. Diagnostic
outputs are in `analysis/cuda-readiness-host-diagnostic.json`; charged budget and
recovery mappings are in both roots' `analysis/recovery.json`.


### Confirmed retry progress (04:49 Paris)

Active ID: `cifar-beta-fresh-current-batch-high-cap-e1-20260930-retry1`;
attempt `d0c9041607f34652a69169db9cd96b69`, local PID1569369. The retry has a
real85,690,824-byte initial checkpoint and has completed96 training updates on
the RTX3090. Its manifest records torch2.5.1+cu121/CUDA12.1 and the requested
30,000 beta ceiling. Startup proof is saved in the retry root's
`analysis/startup-proof.json`. This confirms execution startup, not completion
or a passing learning result. The shared queue owns subsequent monitoring and
validated canonical collection; no extra monitor was created. The first56
training losses exactly match exp-015, before its first ceiling miss. At96
updates the higher-cap arm has no target-tolerance misses and has used block2
beta18,082, above the previous ceiling. This supports the intended mechanism
change; full validation is still pending.
