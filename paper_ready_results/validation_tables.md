# Three-seed validation tables — original inventory

Collected and validated: 198/216 training runs. Official-test results: 0/216.

Values are mean best-validation accuracy ± sample standard deviation over model/shuffle seeds 0, 1, 2.
The fixed 5,000-example validation split was used for selection; these are not official-test accuracies.
Incomplete groups show their collected seed count and have no aggregate. All individual seed outcomes remain in the run ledger.

[Per-seed results](collected_results.csv) · [Best and final aggregates](validation_table_summary.csv) · [Remaining runs](remaining_runs.csv)

This table preserves the original inventory. The matched baseline T/K revision is tracked in the [current-contract tables](current_contract_validation_tables.md).

## Table 1: wide BPTT

| Architecture | Gmax | Baseline | Ours | Legacy |
|---|---:|---:|---:|---:|
| conv1 | 100 | 96.267 ± 0.110% | 96.493 ± 0.076% | 96.540 ± 0.053% |
| conv2 | 100 | 97.340 ± 0.020% | 98.187 ± 0.076% | 98.300 ± 0.040% |
| conv3 | 100 | 97.767 ± 0.031% | 98.560 ± 0.100% | 98.893 ± 0.061% |

## Table 2: wide EqProp

| Architecture | Gmax | Baseline | Ours | Legacy |
|---|---:|---:|---:|---:|
| conv1 | 100 | 96.280 ± 0.092% | 96.453 ± 0.076% | 96.533 ± 0.061% |
| conv2 | 100 | 97.293 ± 0.081% | 98.153 ± 0.061% | 98.340 ± 0.040% |
| conv3 | 100 | 97.693 ± 0.083% | 98.600 ± 0.106% | 98.913 ± 0.070% |

## Table 3: bounded BPTT

| Architecture | Gmax | Baseline | Ours | Legacy |
|---|---:|---:|---:|---:|
| conv1 | 0.0001 | 89.847 ± 0.160% | 93.040 ± 0.317% | 94.940 ± 0.087% |
| conv1 | 0.0005 | 94.580 ± 0.122% | 95.167 ± 0.223% | 96.360 ± 0.100% |
| conv1 | 0.001 | 94.953 ± 0.232% | 95.167 ± 0.204% | 96.420 ± 0.080% |
| conv2 | 0.0001 | 85.453 ± 0.133% | 90.193 ± 0.168% | 95.633 ± 0.205% |
| conv2 | 0.0005 | 92.040 ± 0.458% | 96.560 ± 0.120% | 97.680 ± 0.035% |
| conv2 | 0.001 | 93.460 ± 0.104% | 97.133 ± 0.064% | 97.873 ± 0.081% |
| conv3 | 0.0001 | 77.600 ± 0.053% | 85.133 ± 0.590% | 93.920 ± 0.971% |
| conv3 | 0.0005 | 87.867 ± 0.430% | 94.620 ± 0.208% | 97.827 ± 0.140% |
| conv3 | 0.001 | 89.800 ± 0.223% | 95.733 ± 0.061% | 98.007 ± 0.181% |

## Table 3: bounded EqProp

| Architecture | Gmax | Baseline | Ours | Legacy |
|---|---:|---:|---:|---:|
| conv1 | 0.0001 | 89.867 ± 0.167% | 93.053 ± 0.340% | 94.940 ± 0.060% |
| conv1 | 0.0005 | 94.567 ± 0.129% | 95.173 ± 0.250% | 96.367 ± 0.076% |
| conv1 | 0.001 | 94.960 ± 0.243% | 95.167 ± 0.239% | 96.413 ± 0.081% |
| conv2 | 0.0001 | 0/3 seeds | 90.187 ± 0.160% | 95.620 ± 0.178% |
| conv2 | 0.0005 | 0/3 seeds | 96.527 ± 0.110% | 97.693 ± 0.083% |
| conv2 | 0.001 | 0/3 seeds | 97.160 ± 0.053% | 97.860 ± 0.060% |
| conv3 | 0.0001 | 0/3 seeds | 85.327 ± 0.237% | 93.860 ± 0.866% |
| conv3 | 0.0005 | 0/3 seeds | 94.667 ± 0.380% | 97.840 ± 0.160% |
| conv3 | 0.001 | 0/3 seeds | 95.793 ± 0.081% | 97.960 ± 0.209% |
