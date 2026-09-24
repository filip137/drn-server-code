# Bounded Conv2 legacy: complete BPTT–EqProp three-seed comparison

All 18 trainings are collected and validated: both algorithms, the legacy scheme, three ceilings, and seeds 0/1/2.
These are measurements on the fixed 5,000-example selection split. The official MNIST test split remains unread.

| Scheme | Gmax | BPTT best: mean ± SD | EqProp best: mean ± SD | EP − BPTT (pp) |
|---|---:|---:|---:|---:|
| legacy | 0.0001 | 95.633 ± 0.205% | 95.620 ± 0.178% | -0.013 |
| legacy | 0.0005 | 97.680 ± 0.035% | 97.693 ± 0.083% | +0.013 |
| legacy | 0.001 | 97.873 ± 0.081% | 97.860 ± 0.060% | -0.013 |

SD is the sample standard deviation across all three seeds; no seed is excluded.

All nine paired runs share the initializer, training/validation cohorts, and all thirty minibatch orders. Across 270 paired epoch measurements, the largest accuracy difference is 0.240 percentage points; the largest paired best-accuracy difference is 0.100 points.

The maximum EqProp best-to-final decline is 0.120 points. The frozen injected beta is .003. All ceiling conditions retain thirty epochs, T=K=6, the exact inherited Adam vectors, shared bounded initialization, and zero biases.

The measurements support consistency of these BPTT and EqProp training trajectories under the tested contract. Three seeds on a selection split do not establish statistical equivalence or official-test performance.

![Paired validation trajectories with sample standard deviation](figures/bounded_conv2_legacy_three_seed_validation.png)

[Per-seed results](collected_results.csv) · [Best and final aggregates](validation_table_summary.csv)
