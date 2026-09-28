---
experiment: "exp-017"
evidence: "validated-local"
summary: "All ten uninterrupted baseline trajectories validated; baseline below both comparison schemes with epoch30 optimizer-history confound."
verdicts: {"H-011": "not-tested"}
---
# Exp017 reviewed completion

All ten seed0 baseline cases validated locally: Conv2 D/F4 and Conv3 D/F6 at eta 1e-6, 1e-5, 1e-4, 3e-4 and 1e-3. Each has exactly epochs 1–30, actual CUDA execution, finite metrics, successful completion criteria, and model/optimizer checkpoints for epochs 0–30 whose recorded sizes and SHA-256 hashes passed validation. Canonical manifest/result/artifact validation passed for all ten. No production failures, exclusions or training retries were reported. The withdrawn, unstarted Fifi placement is preserved under relocated-from-fifi and is not extra coverage.

Evidence under `results/conv23-baseline-relative-noise-20260927-v1/`: monitor/collection-receipt.json, monitor/coverage-validation.json, analysis/summary.json, analysis/accuracy.csv, analysis/fixed_epoch_comparisons.json and analysis/epoch30_accuracy.jpg. Declared analyzer: experiments.analyze_baseline_relative_noise. Collection initially hit sandbox DNS restrictions; authorized network access resolved collection without changing training.

Measured outer-launcher charges total 114302.646537 seconds = 31.750735 GPU-hours, below the unchanged 60 GPU-hour budget. All queues completed within the original September 30 14:34:46 UTC deadline. All lane caps were respected.

- local: `/home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/conv23-baseline-relative-noise-20260927-v1/production/local`; 19965.500640 charged seconds.
- nom1: `/home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/conv23-baseline-relative-noise-20260927-v1/collected/nom1/production/nom1`; 12010.413523 charged seconds.
- trex: `/home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/conv23-baseline-relative-noise-20260927-v1/collected/trex/production/trex`; 37255.435276 charged seconds.
- loulou: `/home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/conv23-baseline-relative-noise-20260927-v1/collected/loulou/production/loulou`; 29825.872713 charged seconds.
- riri: `/home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/conv23-baseline-relative-noise-20260927-v1/collected/riri/production/riri`; 15245.424384 charged seconds.

| Model | eta | Epoch10 validation (%) | Epoch30 validation (%) |
|---|---:|---:|---:|
| conv2 | 1e-06 | 96.86 | 97.24 |
| conv2 | 1e-05 | 96.30 | 96.70 |
| conv2 | 0.0001 | 94.14 | 95.32 |
| conv2 | 0.0003 | 91.40 | 93.86 |
| conv2 | 0.001 | 87.68 | 91.70 |
| conv3 | 1e-06 | 95.58 | 96.38 |
| conv3 | 1e-05 | 92.06 | 93.66 |
| conv3 | 0.0001 | 84.22 | 85.46 |
| conv3 | 0.0003 | 82.20 | 83.96 |
| conv3 | 0.001 | 79.60 | 82.16 |

## Scoped interpretation and decision

All ten baseline trajectories improve from fixed epoch10 to fixed epoch30. Baseline is below both ours and legacy in all 20 available depth/eta/scheme comparisons at epoch30. Conv2 baseline deficits span 0.84–4.00 percentage points; Conv3 deficits span 1.30–10.74 points. Baseline accuracy decreases with increasing eta in each architecture on this sampled grid.

Secondary fixed epoch10 comparisons use the recorded exp014 parent measurements in the already reviewed exp015 original and low-noise-extension CSVs. All 20 comparisons are available, and baseline is below both schemes in all of them; exact differences are in analysis/fixed_epoch_comparisons.json. No requested comparison cell is missing.

These are descriptive seed0 validation results. Exp017 trained uninterrupted for 30 epochs with Adam fresh only at initialization; exp015 ours/legacy reset Adam and RNG at epoch10. Thus epoch30 gaps cannot be attributed solely to amplification or noise. The accepted scheme-specific beta/LR settings also remain part of each treatment. No significance, best-epoch superiority, global optimum or official-test claim is made. H-011 specifically tests ours versus legacy at epoch10 across its original grid; these added baseline controls do not independently retest that claim, so its verdict here is not-tested.

Stop exp017 monitoring: collection, validation and this scoped review are complete. No additional experiments, tuning or retries are authorized by this conclusion. REVIEW_COMPLETE applies only to exp017.
