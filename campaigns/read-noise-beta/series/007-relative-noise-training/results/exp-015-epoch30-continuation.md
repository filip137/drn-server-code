---
experiment: "exp-015"
evidence: "validated-local"
summary: "All 20 continuations reviewed; ours leads 3/5 Conv2 and 2/5 Conv3 noise cells, with the largest gap +6.80 pp for Conv3 at eta=1e-4."
verdicts: {"H-011": "not-tested"}
---
# Exp015 reviewed completion

The combined September 28 review below supersedes the historical partial
handoffs. All 20 retained cases are collected, validated and reviewed.

## Original twelve: collection handoff

Validated at 2026-09-27T01:41:46.804685+00:00. Operational completion applies only to this watcher’s twelve cases. Root Reviewer owns scientific review under the current run authorization; REVIEW_COMPLETE is not asserted. The independent eight-case extension and whole twenty-case study remain outside this decision.

## Coverage and validation

All 12 expected identities have one complete canonical bundle, epochs 1–20 exactly (cumulative 11–30), and six matched pairs. The declared analyzer invoked experiments.reporting.validate_run on each local bundle: zero validation errors. All five collected queue exit codes are zero, with no numerical failures or excluded production cases. Checkpoint/artifact hashes and finite metrics passed canonical validation.

Collection: monitor/collection-receipt.json. Analyzer: analysis/summary.json, analysis/accuracy.csv, analysis/report.md and analysis/epoch30_accuracy.jpg. Command: MPLCONFIGDIR=/tmp/exp015-mpl /home/filip/miniconda3/envs/py312/bin/python -m experiments.analyze_relative_noise_epoch30 --study-root results/conv23-relative-noise-epoch30-20260926-v1.

## Roots and charged runtime

- local: /home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/conv23-relative-noise-epoch30-20260926-v1/production/local; 17715.434581 charged seconds.
- nom1: /home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/conv23-relative-noise-epoch30-20260926-v1/collected/nom1/production/nom1; 8110.278297 charged seconds.
- trex: /home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/conv23-relative-noise-epoch30-20260926-v1/collected/trex/production/trex; 24950.294741 charged seconds.
- loulou: /home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/conv23-relative-noise-epoch30-20260926-v1/collected/loulou/production/loulou; 19985.598696 charged seconds.
- riri: /home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/lean-experiment-workflow/results/conv23-relative-noise-epoch30-20260926-v1/collected/riri/production/riri; 20255.607583 charged seconds.

Total measured outer-launcher charges: 91017.213898 seconds = 25.282559 GPU-hours, below 60 GPU-hours. Original deadline remains 2026-09-28T18:38:28.889076+00:00. Charges include preserved initial dtype-load-order failures; failed-startup-dtype and prior-startup receipts are retained locally and in each collected lane. No budget reset, launch, recovery or training changes were made.

## Fixed epoch measurements

| Case | Parent epoch10 (%) | Cumulative epoch30 (%) | Change (pp) |
|---|---:|---:|---:|
| conv2_legacy_relative4_eta1em3_seed0 | 93.54 | 94.98 | +1.44 |
| conv2_legacy_relative4_eta1em4_seed0 | 96.64 | 97.04 | +0.40 |
| conv2_legacy_relative4_eta3em4_seed0 | 95.56 | 96.26 | +0.70 |
| conv2_ours_relative4_eta1em3_seed0 | 94.46 | 95.70 | +1.24 |
| conv2_ours_relative4_eta1em4_seed0 | 96.76 | 97.00 | +0.24 |
| conv2_ours_relative4_eta3em4_seed0 | 96.18 | 96.36 | +0.18 |
| conv3_legacy_relative6_eta1em3_seed0 | 84.68 | 91.94 | +7.26 |
| conv3_legacy_relative6_eta1em4_seed0 | 90.66 | 89.00 | -1.66 |
| conv3_legacy_relative6_eta3em4_seed0 | 89.64 | 92.28 | +2.64 |
| conv3_ours_relative6_eta1em3_seed0 | 87.06 | 90.82 | +3.76 |
| conv3_ours_relative6_eta1em4_seed0 | 93.58 | 95.80 | +2.22 |
| conv3_ours_relative6_eta3em4_seed0 | 90.28 | 94.70 | +4.42 |

Single seed; retrospective selection; fresh Adam and reset noise/data streams at epoch10. These are validation accuracies, not official test results or uninterrupted thirty-epoch training. No scientific verdict is assigned by this incident decision.


## Eight-case lower-noise extension: validated and reviewed

Scoped review completed 2026-09-27T08:38:52.178244+00:00. All eight retained identities (Conv2 D/F4 and Conv3 D/F6, ours/legacy, eta1e-6/1e-5) have one locally collected canonical terminal bundle, exactly epochs1–20 (cumulative11–30), and four matched pairs. Canonical result/manifest/metrics/artifact validation passed with zero errors. Parent final epoch10 SHA-256 identities match preparation and executed configuration; fresh Adam/reset streams, unchanged beta and disabled official test were verified. All four queue exit codes are zero; no numerical failures or exclusions.

Extension root: `results/conv23-relative-noise-epoch30-low-noise-20260926-v1/`. Local production remains authoritative; remote evidence is in `collected/{nom1,trex,riri}/production/`. Collection receipt: `monitor/collection-receipt.json`; explicit epoch/parent/runtime reconciliation: `monitor/coverage-validation.json`; analyzer outputs: `analysis/summary.json`, `analysis/accuracy.csv`, `analysis/report.md`, `analysis/epoch30_accuracy.jpg`.

Measured outer-launcher charge is 61886.356588 seconds = 17.190655 GPU-hours of the unchanged40GPUh allowance. Lane charges: local8885.220237s, nom18025.241492s, trex24835.284928s, riri20140.609930s. Deadline remained September29 18:54:41UTC. No new compute, retries, budget reset or scientific changes were needed for completion.

| Case | Parent epoch10 (%) | Cumulative epoch30 (%) | Change (pp) |
|---|---:|---:|---:|
| conv2_legacy_relative4_eta1em5_seed0 | 97.68 | 97.70 | +0.02 |
| conv2_legacy_relative4_eta1em6_seed0 | 98.06 | 98.22 | +0.16 |
| conv2_ours_relative4_eta1em5_seed0 | 97.70 | 97.78 | +0.08 |
| conv2_ours_relative4_eta1em6_seed0 | 97.84 | 98.08 | +0.24 |
| conv3_legacy_relative6_eta1em5_seed0 | 95.72 | 96.72 | +1.00 |
| conv3_legacy_relative6_eta1em6_seed0 | 97.40 | 97.90 | +0.50 |
| conv3_ours_relative6_eta1em5_seed0 | 94.80 | 96.10 | +1.30 |
| conv3_ours_relative6_eta1em6_seed0 | 97.30 | 97.68 | +0.38 |

### Scoped interpretation and decision

All eight endpoint accuracies increased from their respective parent epoch10 values (+0.02 to +1.30pp). At fixed epoch30, ours-minus-legacy gaps are Conv2 eta1e-6 −0.14pp and eta1e-5 +0.08pp; Conv3 eta1e-6 −0.22pp and eta1e-5 −0.62pp. Thus these four low-noise pairs do not show a consistent advantage for ours. Improvements with extra training do not establish a scheme advantage or statistical significance. These are single-seed, retrospectively selected validation comparisons with an Adam/RNG reset, not uninterrupted training or official-test evidence.

Stop extension monitoring: its collection, validation and scoped review are complete. Combined H-011/20-case scientific review remains with the root Reviewer; preserve the original twelve-case sections and independent watcher. No additional experiment is authorized by this conclusion. REVIEW_COMPLETE applies only to this eight-case extension.

## Combined review — September 28

The original 12 cases and retained eight-case extension reconcile all 20
authorized continuations, forming ten matched pairs. Both local collections
passed their declared validation as recorded above. All runs use the final
epoch-10 parent weights and complete 20 additional epochs. The withdrawn
eta=1e-8/1e-7 extension cases were never trained and are not missing coverage.
The preserved dtype startup failures are operational attempts, not replicates.

| Model | Relative eta | Ours epoch 30 (%) | Legacy epoch 30 (%) | Ours minus legacy (pp) |
|---|---:|---:|---:|---:|
| Conv2, D/F=4 | 1e-6 | 98.08 | 98.22 | −0.14 |
| Conv2, D/F=4 | 1e-5 | 97.78 | 97.70 | +0.08 |
| Conv2, D/F=4 | 1e-4 | 97.00 | 97.04 | −0.04 |
| Conv2, D/F=4 | 3e-4 | 96.36 | 96.26 | +0.10 |
| Conv2, D/F=4 | 1e-3 | 95.70 | 94.98 | +0.72 |
| Conv3, D/F=6 | 1e-6 | 97.68 | 97.90 | −0.22 |
| Conv3, D/F=6 | 1e-5 | 96.10 | 96.72 | −0.62 |
| Conv3, D/F=6 | 1e-4 | 95.80 | 89.00 | +6.80 |
| Conv3, D/F=6 | 3e-4 | 94.70 | 92.28 | +2.42 |
| Conv3, D/F=6 | 1e-3 | 90.82 | 91.94 | −1.12 |

Ours leads in three of five Conv2 cells and two of five Conv3 cells. Its largest
advantage is Conv3 at eta=1e-4 (+6.80 pp); legacy leads at the two lowest noise
levels in Conv3 and regains the lead at eta=1e-3. These results show a region
of advantage for ours, not a monotonic or universal ranking. Nineteen of the
20 trajectories improve from epoch 10; Conv3 legacy at eta=1e-4 declines by
1.66 pp. Extra training therefore does not uniformly preserve the earlier ranking.

Both schemes reset Adam and deterministic data/noise streams after epoch 10;
beta and the inherited learning-rate vectors remain fixed. The displacement
targets were matched only at initialization. These are retrospectively selected,
single-seed validation comparisons, with no official-test evaluation or noisy
seed uncertainty. The uninterrupted baseline results in
[exp017](exp-017-baseline-relative-noise.md) have a different optimizer history.
H-011 concerns the original ten-epoch grid across all three architectures, so
this continuation does not independently retest it or change exp014's verdict.

Decision: the 20-case continuation and its review are complete. No additional
training is assigned. Combined charged runtime is 42.473214 GPU-hours; the
original and extension allocations and deadlines were separately respected.
