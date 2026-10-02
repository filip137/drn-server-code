---
experiment: "exp-022"
evidence: "validated-local"
summary: "All 72 new-seed cases validated; ours beats baseline in every pair, but legacy advantages do not consistently replicate and Conv3 mean gaps are negative at all four etas."
verdicts: {"H-011": "inconclusive"}
---
# Result: exp022 — Two additional read-noise seeds

## Evidence and coverage

Reviewed October 1, 2026 against the [frozen experiment](../experiments/exp-022-relative-noise-seeds.md).
All 72 cases completed once, without failed, replaced, resumed or excluded production
runs: Conv1/2/3 × baseline/ours/legacy × four etas × seeds 1/2. Horizons were
10/30/30 epochs, batch 16, with uninterrupted Adam and matched initialization,
shuffle and noise streams within depth/seed. The nudging targets remain seed-0
reference output D/F=1/4/6; beta was frozen, and actual seed-1/2 D/F was not measured.

Evidence root: `results/conv123-relative-noise-seeds12-jeanzay-20260930-v1/`.
Complete remote collection is in `collected/`; remote evidence is retained.
`python -m experiments.analyze_relative_noise_seeds` passed for 72/72 canonical
bundles with no validation errors: frozen source/config identities, all 1680 epoch
records, 1752 model/optimizer checkpoint pairs (3504 checkpoint files), exact Adam
step counts, global/private RNG state, zero biases and matched dataset/order hashes.
All official-test evaluation/example counts are zero. These are ordinary-MNIST
55k/5k training/validation measurements.

Fresh Slurm accounting verifies all 72 submitted allocations COMPLETED with exit
0:0 and the expected accounts and single-GPU resources. The last finished at
12:48:03 Europe/Paris on October 1. Total charge: **165.6375 GPU-hours** of the
240-hour cap, including allocation overhead. Receipts: `analysis/allocation-accounting.json`,
`analysis/terminal-accounting.psv`, and `analysis/collection-receipt.json`.

Source archive SHA256:
`524c978c8fd3831741c0b0b3cb9e1eba374c2f6202681b0005309db0f57d566b`.
Two collector assumptions were corrected without changing training or scientific
configs: disabled-test policy does not require a redundant CLI flag, and the
continuation config hash includes the explicitly declared dataset-root transport
override. Original failures are retained in `analysis/validation-corrections.json`;
23 focused regression tests pass. The automatic collection attempt hit a 20-second
transfer limit; root completed the same transfer and retained that operational history.

## Measurements

Primary fixed-final validation accuracy (%), mean over seeds 1/2 (n=2).
All per-seed values, sample SDs, paired differences and secondary best-epoch values
are retained in the [full report](../../../../../results/conv123-relative-noise-seeds12-jeanzay-20260930-v1/analysis/report.md).

| Model | η | Baseline | Ours | Legacy | Ours − legacy (pp) | Ours − baseline (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Conv1 | 1e-05 | 96.27 | 96.35 | 96.55 | -0.20 | +0.08 |
| Conv1 | 0.0001 | 96.22 | 96.31 | 96.44 | -0.13 | +0.09 |
| Conv1 | 0.0003 | 96.08 | 96.26 | 96.29 | -0.03 | +0.18 |
| Conv1 | 0.0005 | 95.95 | 96.27 | 96.26 | +0.01 | +0.32 |
| Conv2 | 1e-05 | 96.66 | 97.70 | 97.76 | -0.06 | +1.04 |
| Conv2 | 0.0001 | 95.33 | 97.12 | 97.07 | +0.05 | +1.79 |
| Conv2 | 0.0003 | 94.00 | 96.50 | 96.66 | -0.16 | +2.50 |
| Conv2 | 0.0005 | 93.36 | 96.16 | 95.90 | +0.26 | +2.80 |
| Conv3 | 1e-05 | 95.38 | 96.45 | 96.72 | -0.27 | +1.07 |
| Conv3 | 0.0001 | 87.22 | 93.06 | 94.43 | -1.37 | +5.84 |
| Conv3 | 0.0003 | 84.43 | 89.93 | 92.96 | -3.03 | +5.50 |
| Conv3 | 0.0005 | 83.79 | 91.16 | 91.54 | -0.38 | +7.37 |

[Final-accuracy JPG](../../../../../results/conv123-relative-noise-seeds12-jeanzay-20260930-v1/analysis/final_accuracy.jpg) ·
[Paired-gap JPG](../../../../../results/conv123-relative-noise-seeds12-jeanzay-20260930-v1/analysis/paired_gaps.jpg).

Secondary epoch-10 ours-minus-legacy differences, seeds 1/2 respectively:

| Model | 1e-5 | 1e-4 | 3e-4 | 5e-4 | Grid mean (pp) |
|---|---:|---:|---:|---:|---:|
| Conv1 | -0.18/-0.22 | -0.10/-0.16 | -0.06/+0.00 | +0.14/-0.12 | -0.088 |
| Conv2 | -0.22/-0.12 | +0.32/-0.02 | +0.12/+0.12 | -0.30/+0.44 | +0.042 |
| Conv3 | -0.60/-0.90 | -6.10/+1.72 | -0.60/-3.04 | +3.38/+0.78 | -0.670 |

These secondary values are extracted from the already validated epoch histories
in `analysis/epoch10-secondary.json`; they do not replace the primary final epochs.

The separate historical supplement validates 24 eligible uninterrupted seed-0
cases and reports n=3 only where compatible coverage exists. Conv2/3 ours/legacy
seed-0 cases at eta=1e-5, 1e-4 and 3e-4 reset Adam/RNG after epoch 10 and are
excluded from pooled uninterrupted comparisons. Source/hardware differences and
the absence of a complete historical private-RNG audit remain explicit. Paired
supplementary comparisons use common eligible seed sets.

## Interpretation

Ours exceeds baseline in all 24 paired seed/cell comparisons. Mean advantages
are small in Conv1 (+0.08 to +0.32 pp), larger in Conv2 (+1.04 to +2.80 pp), and
largest at higher noise in Conv3 (+5.50 to +7.37 pp at eta≥1e-4).

An advantage over legacy does **not** consistently replicate. Conv1 favors legacy
at the first three etas; its +0.01 pp mean at 5e-4 has opposite seed signs. Conv2
has small positive gaps in both seeds at 1e-4 and 5e-4, but negative gaps in both
at 3e-4. Conv3 has negative mean gaps at every eta, with both seeds favoring legacy
at 3e-4 (−3.90/−2.16 pp). At 1e-4, ours ranges from 90.54% to 95.58% across seeds,
so the earlier large single-seed gain is not a stable general conclusion. These
observations do not isolate noise, optimizer-history, hardware or source effects.

H-011 remains **inconclusive for all-architecture reproducibility** in this
restricted replication. Its original epoch-10 grid is broader than these four
etas. Positive epoch-10 pairs exist somewhere in each architecture, satisfying
its weak existence criterion within these observed pairs, but Conv1 has no cell
positive in both new seeds. Conv2 at 3e-4 and Conv3 at 5e-4 do have positive
epoch-10 pairs in both seeds. Do not reinterpret final epoch-30 results as direct
tests of the original epoch-10 claim, or these n=2 descriptive summaries as
significance, a noise optimum, or a universal ranking.

## Decision

Close exp022: all authorized coverage is collected, validated and reviewed; the
hourly watcher is stopped. Retain negative and mixed-sign comparisons. No further
training, retuning, official-test evaluation or manuscript edit follows automatically.
These results can support a clearly labeled validation analysis; they are not
official-test paper accuracies. Freeze the complete paper evaluation contract and
checkpoint-selection rule before any once-only official-test evaluation.
