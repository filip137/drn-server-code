# Bounded Conv1: complete BPTT–EqProp three-seed comparison

All 54 trainings are collected and validated: both algorithms, three schemes, three ceilings, and seeds 0/1/2.
These are measurements on the fixed 5,000-example selection split. The official MNIST test split remains unread.

| Scheme | Gmax | BPTT best: mean ± SD | EqProp best: mean ± SD | EP − BPTT (pp) |
|---|---:|---:|---:|---:|
| baseline | 0.0001 | 89.847 ± 0.160% | 89.867 ± 0.167% | +0.020 |
| baseline | 0.0005 | 94.580 ± 0.122% | 94.567 ± 0.129% | -0.013 |
| baseline | 0.001 | 94.953 ± 0.232% | 94.960 ± 0.243% | +0.007 |
| ours | 0.0001 | 93.040 ± 0.317% | 93.053 ± 0.340% | +0.013 |
| ours | 0.0005 | 95.167 ± 0.223% | 95.173 ± 0.250% | +0.007 |
| ours | 0.001 | 95.167 ± 0.204% | 95.167 ± 0.239% | +0.000 |
| legacy | 0.0001 | 94.940 ± 0.087% | 94.940 ± 0.060% | +0.000 |
| legacy | 0.0005 | 96.360 ± 0.100% | 96.367 ± 0.076% | +0.007 |
| legacy | 0.001 | 96.420 ± 0.080% | 96.413 ± 0.081% | -0.007 |

SD is the sample standard deviation across all three seeds; no seed is excluded.

All 27 paired runs share the initializer, training/validation cohorts, and all ten minibatch orders. Across 270 paired epoch measurements, the largest accuracy difference is 0.100 percentage points; the largest paired best-accuracy difference is 0.060 points.

The maximum EqProp best-to-final decline is 2.440 points. The frozen injected betas are baseline .1, ours .03, and legacy .03. All ceiling conditions retain ten epochs, T=K=4, the exact inherited Adam vectors, shared bounded initialization, and zero biases.

The measurements support consistency of these BPTT and EqProp training trajectories under the tested contract. Three seeds on a selection split do not establish statistical equivalence or official-test performance.

![Paired validation trajectories with sample standard deviation](figures/bounded_conv1_three_seed_validation.png)

[Per-seed results](collected_results.csv) · [Best and final aggregates](validation_table_summary.csv)
