# Baseline beta audit: 100, 200, 300

Completed read-only clean wide-range Adam diagnostic, 2026-09-14. The other schemes, learning rates and training configs remain unchanged.

Each selection case uses model seed 0, initialization and the BPTT best checkpoint, with 32 new batches of 16 plus the original four batches. Thresholds are every-layer/batch cosine >= .99 and symmetric norm difference <= .10. Equilibrium residual qualification is separate.

| Architecture | Beta | Worst initial cosine | Worst trained cosine | Failed layer/batches | Gradient gate | Equilibrium gate |
|---|---:|---:|---:|---:|---|---|
| conv1 | 100 | 0.999711 | 0.997616 | 0/144 | pass | pass |
| conv1 | 200 | 0.999481 | 0.994516 | 0/144 | pass | pass |
| conv1 | 300 | 0.999284 | 0.992238 | 0/144 | pass | pass |
| conv2 | 100 | 0.992029 | 0.998146 | 0/216 | pass | pass |
| conv2 | 200 | 0.984396 | 0.996685 | 3/216 | fail | pass |
| conv2 | 300 | 0.973355 | 0.993634 | 7/216 | fail | pass |
| conv3 | 100 | 0.967393 | 0.999421 | 7/288 | fail | fail |
| conv3 | 200 | 0.940800 | 0.998943 | 16/288 | fail | fail |
| conv3 | 300 | 0.931421 | 0.998685 | 29/288 | fail | fail |

## Selection and confirmation

- conv1: candidate entering confirmation **300**; state `confirmation_failed`.
- conv2: candidate entering confirmation **100**; state `confirmed`.
- conv3: candidate entering confirmation **none**; state `no_passing_tested_beta`.

| Architecture | Candidate beta | Seed | Worst cosine | Largest norm mismatch | Gradient gate |
|---|---:|---:|---:|---:|---|
| conv1 | 300 | 0 | 0.993926 | 0.028822 | pass |
| conv1 | 300 | 1 | 0.991313 | 0.032866 | pass |
| conv1 | 300 | 2 | 0.976462 | 0.068430 | fail |
| conv2 | 100 | 0 | 0.991173 | 0.046374 | pass |
| conv2 | 100 | 1 | 0.994066 | 0.055568 | pass |
| conv2 | 100 | 2 | 0.996504 | 0.060712 | pass |

Conv2 beta 100 passes the numerical selection and all three confirmation seeds. Conv1 beta 300 fails three trained-checkpoint layer/batch comparisons on seed 2; beta 100 and 200 pass seed-0 selection but were not subsequently confirmed across seeds in this round. Conv3 fails all three candidates at initialization and retains a separate trained free-state residual failure. No training beta has been changed.

Historical check: all 72 original baseline beta-100 layer comparisons retain identical input payloads; maximum cosine change 0 and norm-mismatch change 0.

Coverage: **9/9 selection cases and 6/6 conditionally required confirmation cases**; 1080 checkpoint/batch replays and 3024 layer comparisons. All included canonical bundles validate.

Only a passing candidate is tested on the reserved confirmation cohort and seeds 0/1/2. An architecture with no passing candidate has no confirmed replacement beta. Confirmation failures are retained without trying another beta on the same cohort. These are sampled gradient checks, not proof of accuracy, full-training stability or equivalence. The official test split was not read and no training was launched.

The first preparation smoke stopped before computation because its inherited runtime path pointed at the working checkout. Its log/configs are retained; the corrected smoke and production use the hash-verified frozen runtime and version-2 configs.

![Baseline beta audit](baseline_beta_audit_20260914.png)

![Frozen-candidate confirmation](baseline_beta_confirmation_20260914.png)

[Per-case measurements](baseline_beta_audit_20260914.csv) · [Per-layer/cohort distributions](baseline_beta_layer_cohort_summary_20260914.csv) · [Full local evidence](../results/eqprop-beta-selection-audit-20260914-v1/analysis/) · [Execution plan](../docs/eqprop_baseline_beta_audit_20260914.md)
