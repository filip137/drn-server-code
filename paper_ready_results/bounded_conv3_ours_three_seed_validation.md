# Bounded Conv3 ours: complete BPTT–EqProp three-seed comparison

All 18 trainings are collected and validated: both algorithms, the ours scheme, three ceilings, and seeds 0/1/2.
These are measurements on the fixed 5,000-example selection split. The official MNIST test split remains unread.

| Scheme | Gmax | BPTT best: mean ± SD | EqProp best: mean ± SD | EP − BPTT (pp) |
|---|---:|---:|---:|---:|
| ours | 0.0001 | 85.133 ± 0.590% | 85.327 ± 0.237% | +0.193 |
| ours | 0.0005 | 94.620 ± 0.208% | 94.667 ± 0.380% | +0.047 |
| ours | 0.001 | 95.733 ± 0.061% | 95.793 ± 0.081% | +0.060 |

SD is the sample standard deviation across all three seeds; no seed is excluded.

Filip authorized moving the remaining seed-1 repetitions from occupied Trex after measuring GPU packing. Conv3 ours seed 1 uses Loulou; Conv3 legacy seed 1 retains its completed tight-ceiling case on Trex and uses nom-cool-1 for the other ceilings. Initialization, cohorts, minibatch order and scientific configs are checked below, but host and software environments differ. Per-run targets and runtime metadata are retained in the analysis JSON.

All nine paired runs share the initializer, training/validation cohorts, and all thirty minibatch orders. Across 270 paired epoch measurements, the largest accuracy difference is 1.200 percentage points; the largest paired best-accuracy difference is 0.660 points.

The maximum EqProp best-to-final decline is 0.560 points. The frozen injected beta is .003. All ceiling conditions retain thirty epochs, T=K=8, the exact inherited Adam vectors, shared bounded initialization, and zero biases.

The paired differences above describe these BPTT and EqProp training trajectories under the frozen contract. Three seeds on a selection split do not establish statistical equivalence or official-test performance.

![Paired validation trajectories with sample standard deviation](figures/bounded_conv3_ours_three_seed_validation.png)

[Per-seed results](collected_results.csv) · [Best and final aggregates](validation_table_summary.csv)
