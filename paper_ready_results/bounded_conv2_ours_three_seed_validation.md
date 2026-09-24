# Bounded Conv2 ours: complete BPTT–EqProp three-seed comparison

All 18 trainings are collected and validated: both algorithms, the ours scheme, three ceilings, and seeds 0/1/2.
These are measurements on the fixed 5,000-example selection split. The official MNIST test split remains unread.

| Scheme | Gmax | BPTT best: mean ± SD | EqProp best: mean ± SD | EP − BPTT (pp) |
|---|---:|---:|---:|---:|
| ours | 0.0001 | 90.193 ± 0.168% | 90.187 ± 0.160% | -0.007 |
| ours | 0.0005 | 96.560 ± 0.120% | 96.527 ± 0.110% | -0.033 |
| ours | 0.001 | 97.133 ± 0.064% | 97.160 ± 0.053% | +0.027 |

SD is the sample standard deviation across all three seeds; no seed is excluded.

All nine paired runs share the initializer, training/validation cohorts, and all thirty minibatch orders. Across 270 paired epoch measurements, the largest accuracy difference is 0.460 percentage points; the largest paired best-accuracy difference is 0.080 points.

The maximum EqProp best-to-final decline is 0.340 points. The frozen injected beta is .001. All ceiling conditions retain thirty epochs, T=K=6, the exact inherited Adam vectors, shared bounded initialization, and zero biases.

The measurements support consistency of these BPTT and EqProp training trajectories under the tested contract. Three seeds on a selection split do not establish statistical equivalence or official-test performance.

![Paired validation trajectories with sample standard deviation](figures/bounded_conv2_ours_three_seed_validation.png)

[Per-seed results](collected_results.csv) · [Best and final aggregates](validation_table_summary.csv)
