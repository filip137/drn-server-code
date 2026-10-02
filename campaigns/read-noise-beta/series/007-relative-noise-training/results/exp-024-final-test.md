---
experiment: "exp-024"
evidence: "validated-local"
summary: "All153 final tests validated; requested three-seed summary shows ours above baseline in12/12 means and above legacy in3/12, with12 scheme/eta groups mixing seed0 reset history."
verdicts: {}
---
# Result: exp024 — Final-checkpoint official MNIST tests

## Evidence and coverage

Reviewed October1,2026 against the [fixed-final contract](../experiments/exp-024-final-test.md).
All153 cases passed the strict local `analysis/analyze.py` audit with no partial
flag:127 new official-test passes and26 canonical reuses with exact parameter,
dtype, shape and configuration identity. Zero missing cases or validation errors;
all three queue groups completed collection and declared validation. No new-test
retries were needed. Each new evaluation used all10000 official test images in one
batch, unchanged float64 clean inference, and no training or optimizer steps.

These are **final epoch10 for Conv1 and final cumulative epoch30 for Conv2/3**,
with seed-0-reference D/F=1/4/6 and frozen betas. Actual seed1/2 D/F was not remeasured.
All27 clean controls,54 noisy seed0 cases and72 noisy seed1/2 cases are present.
The20 historical seed0 Adam/RNG-reset trajectories remain separately labeled.
Primary noisy means use uninterrupted seeds1/2; clean means use seeds0/1/2.

Result root: `results/read-noise-final-test-20261001-v1/`.
Local Conv2 artifacts are under `workers/test-local-v1`; collected Conv1/Conv3
artifacts are under `collected/akib/workers/test-akib-v1` and
`collected/nom-cool-1/workers/test-nom-cool-1-v1`. Exact-state reuse bundles are
under `reuse/`. Contract SHA256:
`5e28132a7ecbeded2c54790cbd3f04a75908db50290e63ba448b1e0983e5411a`.

The last worker finished October1 at **15:22:01 Europe/Paris**. The three workers
used1868.742seconds (0.5191 GPU-hours), within the6GPUh budget. Source/config,
final checkpoint identity, capacity carryforward, unique test reservations,
prior exposure and matching raw dataset hashes all pass the artifact audit.

Exp023 had already tested best-validation checkpoints on the same test set before
the user requested this correction. That prior exposure is explicitly recorded;
these are additional fixed-final measurements, not an untouched holdout. No model,
noise level or seed was selected from test outcomes. Exp023 remains separate.

## Original uninterrupted two-seed measurements

Final official-test accuracy (%), means over uninterrupted seeds1/2, n=2:

| Model | Training η | Baseline | Ours | Legacy | Ours − baseline (pp) | Ours − legacy (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Conv1 | 1e-05 | 96.690 | 96.835 | 96.900 | +0.145 | -0.065 |
| Conv1 | 0.0001 | 96.630 | 96.790 | 96.850 | +0.160 | -0.060 |
| Conv1 | 0.0003 | 96.490 | 96.745 | 96.740 | +0.255 | +0.005 |
| Conv1 | 0.0005 | 96.365 | 96.720 | 96.775 | +0.355 | -0.055 |
| Conv2 | 1e-05 | 97.255 | 98.300 | 98.375 | +1.045 | -0.075 |
| Conv2 | 0.0001 | 96.110 | 97.735 | 97.770 | +1.625 | -0.035 |
| Conv2 | 0.0003 | 94.680 | 97.095 | 97.190 | +2.415 | -0.095 |
| Conv2 | 0.0005 | 94.275 | 96.745 | 96.735 | +2.470 | +0.010 |
| Conv3 | 1e-05 | 96.005 | 97.140 | 97.490 | +1.135 | -0.350 |
| Conv3 | 0.0001 | 88.885 | 94.055 | 95.335 | +5.170 | -1.280 |
| Conv3 | 0.0003 | 86.330 | 90.955 | 93.945 | +4.625 | -2.990 |
| Conv3 | 0.0005 | 85.300 | 92.245 | 92.620 | +6.945 | -0.375 |

[Full paired final validation/test table](../../../../../results/read-noise-final-test-20261001-v1/analysis/final_validation_test.csv) ·
[Means and sample SDs](../../../../../results/read-noise-final-test-20261001-v1/analysis/primary_final_means.csv) ·
[Clean means and sample SDs](../../../../../results/read-noise-final-test-20261001-v1/analysis/clean_final_means.csv) ·
[Final validation/test JPG](../../../../../results/read-noise-final-test-20261001-v1/analysis/final_validation_test.jpg).

## Original uninterrupted two-seed interpretation

Ours exceeds baseline in all12 noisy architecture/eta means and all24 individual
matched seed comparisons. Relative to legacy, ours is above in only2/12 means:
Conv1 eta3e-4 (+0.005pp) and Conv2 eta5e-4 (+0.010pp). Those differences are tiny;
there is no general advantage over legacy. Individual ours-versus-legacy pairs
win8 and lose16. Conv3 remains below legacy at all four etas, by0.350/1.280/
2.990/0.375pp. Clean means favor ours over baseline and legacy over ours at each depth.

Relative to the earlier best-checkpoint test means, fixed-final noisy means shift
by−0.950 to+0.220pp across the36 scheme/depth/eta cells. The main qualitative
conclusion is unchanged; the small Conv1 advantage over legacy at eta5e-4 disappears.
Final-test means exceed matching final-validation means by0.350–1.900pp. This
comparison now holds the checkpoint fixed and changes only the evaluation split.

Small seed counts and retained scheme-specific learning rates/betas limit inference;
these are descriptive results, not evidence of statistical significance or an
isolated causal effect of amplification. Historical reset runs are not pooled with
uninterrupted runs. All153 requested final outcomes are collected, validated and
reviewed. Use this final-endpoint table for the requested paper comparison with
prior exposure and seed/history labels disclosed. No manuscript edit, further
training or additional test evaluation is performed by this closeout.

## Requested all-three-seed comparison — October1

Filip requested inclusion of all three seeds after discussing variability. The
updated descriptive summary uses seeds0/1/2 for every scheme/depth at the four
replicated noise levels:108 existing validated final outcomes in36 groups, n=3
each. No new training or test evaluation. The original two-seed analysis above
remains the matched uninterrupted comparison; the new display includes and labels
seed0's different training history rather than treating all groups as identical
replications. Twelve groups (Conv2/3 ours/legacy at eta1e-5,1e-4,3e-4) include a
seed0 Adam/RNG reset after epoch10. All groups at eta5e-4 are uninterrupted.

[Three-seed means, SDs and history labels](../../../../../results/read-noise-final-test-20261001-v1/analysis/three_seed_report.md) ·
[Three-seed accuracy JPG](../../../../../results/read-noise-final-test-20261001-v1/analysis/three_seed_final_test.jpg) ·
[Three-seed spread JPG](../../../../../results/read-noise-final-test-20261001-v1/analysis/three_seed_spread.jpg).

Across these descriptive three-seed averages, ours exceeds baseline in12/12
means and legacy in3/12 (Conv2 eta5e-4; Conv3 eta1e-4 and5e-4). Ours has greater
sample SD than legacy in5/12 groups, so greater variability is not general.
For Conv3, sample SDs (percentage points) for ours versus legacy are:

| Training η | Ours SD | Legacy SD | History |
|---|---:|---:|---|
| 1e-5 | 0.135 | 0.500 | Seed0 reset |
| 1e-4 | 2.930 | 2.713 | Seed0 reset |
| 3e-4 | 2.667 | 0.582 | Seed0 reset |
| 5e-4 | 1.358 | 4.985 | All uninterrupted |

Thus the conspicuous ours-versus-legacy spread at1e-4 in the two-seed view becomes
similar with seed0 included. Ours remains more variable at3e-4; legacy is much more
variable at5e-4. For mixed-history groups, observed spread includes the training
history difference and cannot be attributed to random seed alone. Three seeds
remain insufficient for a confident variance ranking.

Reproduce with `analysis/analyze_three_seeds.py` under the registered result root.
Its audit verifies108 unique valid cases, exact seeds0/1/2 per group, final epochs,
10000 test examples per case and all36 groups; it records the input table SHA256
in `analysis/three_seed_audit.json`. No immutable contract or run artifact changed.
