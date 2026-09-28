---
experiment: "exp-018"
evidence: "validated-local"
summary: "14 ten-epoch Conv1 controls validated; legacy leads clean means and noisy baseline trails both saved comparators"
verdicts: {"H-011": "not-tested"}
---
# Exp018 reviewed completion

All 14 distinct cases were collected and validated locally: nine clean cases (baseline/ours/legacy × seeds 0, 1, 2) and five noisy baseline seed0 cases. All 140 epoch records are finite, with exact epochs 1–10. Canonical artifact validation passed without errors. CUDA execution on nom-cool-1, Adam, disabled official test, three distinct initial parameter hashes matched across schemes, actual shuffle seeds and splitseed0 were verified. Epoch0–10 model and optimizer checkpoints are present; commands start from initialization without resume. No missing cases, duplicates, numerical failures, exclusions or training retries were found.

Authoritative study root: `results/conv1-clean-and-baseline-noise-20260927-v1/`. Remote root: `filip@nom-cool-1:/home/filip/server_code/results/conv1-clean-and-baseline-noise-20260927-v1/training_workspace/`. Collected bundles: `collected/nom1/production/nom1/`; terminal launcher history: `collected/nom1/launch/nom1/`. All 14 case wall-time receipts sum to 5795.243729 seconds (1.609790 GPU-hours), matching queue accounting and below the unchanged four-hour allowance; all completed before the deadline. Queue exit was zero. No budget reset or recovery was needed.

## Fixed epoch-10 measurements

Ordinary-MNIST validation accuracy only; no best-epoch selection.

| Scheme | Clean mean (%) | Sample SD (pp), n=3 |
|---|---:|---:|
| baseline | 96.1867 | 0.1474 |
| ours | 96.3333 | 0.0231 |
| legacy | 96.4933 | 0.0643 |

| Paired difference | Seed0 (pp) | Seed1 (pp) | Seed2 (pp) | Mean (pp) | Sample SD (pp) |
|---|---:|---:|---:|---:|---:|
| baseline minus ours | -0.3000 | -0.0600 | -0.0800 | -0.1467 | 0.1332 |
| baseline minus legacy | -0.4000 | -0.2200 | -0.3000 | -0.3067 | 0.0902 |
| ours minus legacy | -0.1000 | -0.1600 | -0.2200 | -0.1600 | 0.0600 |

| Relative eta | Baseline (%) | Saved exp014 ours (%) | Saved exp014 legacy (%) |
|---|---:|---:|---:|
| 1e-06 | 96.04 | 96.32 | 96.40 |
| 1e-05 | 96.00 | 96.34 | 96.40 |
| 0.0001 | 96.00 | 96.28 | 96.38 |
| 0.0003 | 95.80 | 96.12 | 96.14 |
| 0.001 | 95.44 | 95.86 | 95.86 |

## Interpretation and decision

Legacy exceeds ours and baseline in all three matched clean seeds; ours exceeds baseline in all three. At the five declared noise levels, baseline trails both saved ten-epoch exp014 comparators. These are descriptive comparisons under the frozen scheme-specific learning rates and seed0-reference D/F=1 betas; they do not isolate amplification from all algorithmic differences. Actual initial D/F may differ for seeds1/2. Three clean seeds and one noisy seed do not establish significance or general superiority. Results are validation evidence, not official-test paper accuracy.

H-011's cross-architecture noisy ours-versus-legacy claim is not newly tested by these baseline/clean controls. The saved exp014 comparator evidence and its existing verdict remain unchanged; reusing those measurements does not create new replicates. The requested controls are complete; stop this watch with no further simulations.

Validation: declared `experiments.analyze_conv1_controls` completed with 14/14 and zero errors; `analysis/identity_coverage.json` records identity, checkpoint, finite-metric, runtime and paired-gap checks. Prior comparator validation is recorded in the exp014 result note. Full per-case measurements are in `analysis/accuracy.csv`, comparisons in `analysis/summary.json`.

[Epoch-10 JPG](../../../../../results/conv1-clean-and-baseline-noise-20260927-v1/analysis/epoch10_accuracy.jpg).
