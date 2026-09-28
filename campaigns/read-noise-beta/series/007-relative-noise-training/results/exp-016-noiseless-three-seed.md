---
experiment: "exp-016"
evidence: "validated-local"
summary: "All 18 clean controls validated; legacy has the highest mean, followed by ours and baseline for Conv2 and Conv3"
verdicts: {"H-011": "not-tested"}
---
# Exp016 scoped results

## Baseline supplement: six runs collected and reviewed

All six baseline cases (Conv2/Conv3 × seeds 0/1/2) completed 30 uninterrupted Adam epochs on CUDA. The declared analyzer validated all six canonical local bundles with no errors. Frozen config/source hashes, split/shuffle seeds, disabled official test and initializer hashes matching the original ours/legacy pairs were checked. All 31 indexed model/optimizer checkpoint pairs (initialization and epochs 1–30) per case passed size and SHA-256 checks. No missing cases, duplicate bundles or numerical failures were found. All five queue exits are zero, within their allocations and the deadline.

Authoritative supplement: `results/conv23-noiseless-baseline-three-seed-20260927-v1/`. Local Conv2 seeds0/2 remain in `production/local`; collected nom1, trex, loulou and riri outputs are under `collected/LANE/production/LANE`. Conv3 seed0 ran on Trex after the authorized withdrawal of the unstarted Fifi controller. Its zero-charge relocation receipts remain under `relocated-from-fifi/`; they are not extra scientific coverage. No training retry was required in this completion decision. Summed charged launcher time is 67,836.622293 seconds (18.843506 GPU-hours), within the unchanged 40-GPU-hour supplemental budget.

| Model | Seed0 (%) | Seed1 (%) | Seed2 (%) | Mean (%) | Sample SD (pp) |
|---|---:|---:|---:|---:|---:|
| Conv2 baseline | 97.30 | 97.26 | 97.20 | 97.2533 | 0.0503 |
| Conv3 baseline | 97.56 | 97.62 | 97.78 | 97.6533 | 0.1137 |

These are fixed epoch-30 ordinary-MNIST validation accuracies, not best-epoch selections or official-test results. They establish the clean baseline reference across the declared seeds. Baseline-only evidence cannot establish a scheme advantage or test H-011's relative-noise comparison. Three seeds do not support a statistical-significance claim. Beta was frozen from seed0, not rematched for seeds1/2. Comparisons with exp015 remain descriptive because its Adam history includes an epoch10 reset.

Decision: the baseline6 supplement is complete; launch no further simulations. The original12 collection/review and combined18 paired scheme comparisons remain with their existing owner and are not certified by this bounded incident. Whole exp016 is not marked complete.

Validation receipts: `monitor/collection-receipt.json`, `monitor/completion-validation.json`, `analysis/summary.json`, `analysis/accuracy.csv` and `analysis/epoch30_accuracy.jpg` under the supplement. The analyzer command is preserved in `monitor/handoff.md`; result hashes and host identities are in `monitor/completion-validation.json`.

## Completed combined review — September 28

Root collected and validated the original 12 canonical bundles locally with the
declared analyzer: 12/12, no errors, complete epochs 1–30. Together with the
six validated baseline runs, this reconciles all 18 cases. All original queues
exited zero with no numerical failures. Collection required no new training.

| Model | Baseline mean ± sample SD | Ours | Legacy |
|---|---:|---:|---:|
| Conv2 | 97.2533 ± 0.0503 | 98.0467 ± 0.0416 | 98.2400 ± 0.0693 |
| Conv3 | 97.6533 ± 0.1137 | 98.6067 ± 0.1361 | 98.8133 ± 0.0987 |

All values are fixed epoch-30 validation percentages across seeds 0/1/2. Legacy
has the highest mean at both depths; ours exceeds baseline. Ours-minus-legacy
seed gaps are −0.16/−0.20/−0.22 pp for Conv2 and +0.06/−0.36/−0.32 pp for Conv3.
Legacy's clean lead is consistent across Conv2 seeds, but not universal for
Conv3. Three seeds do not establish statistical significance. These clean
controls do not test H-011's noise-dependent hypothesis. Baseline Conv3 seed 0
relocated to Trex; ours/legacy seed 0 stayed on Fifi. This transport difference
is recorded. All clean runs used uninterrupted Adam.

Decision: complete this clean-control study; retain the clean ordering as a
reference and interpret noisy comparisons separately. No additional runs are
assigned by this review. Original 12 artifacts:
`results/conv23-noiseless-three-seed-20260927-v1/`, including
`monitor/collection-receipt.json` and
`analysis/{summary.json,accuracy.csv,report.md,epoch30_accuracy.jpg}`.
Whole-study coverage is 18/18; this section supersedes the earlier pending-review
statement.
