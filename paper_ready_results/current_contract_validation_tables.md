# Three-seed validation tables — revised contract

Collected and validated: 202/216 training runs. Official-test results: 0/216.

Values are mean best-validation accuracy ± sample standard deviation over model/shuffle seeds 0, 1, 2.
The fixed 5,000-example validation split was used for selection; these are not official-test accuracies.
Incomplete groups show their collected seed count and have no aggregate. All individual seed outcomes remain in the run ledger.

[Per-seed results](current_contract_run_status.csv) · [Best and final aggregates](current_contract_validation_table_summary.csv) · [Remaining runs](current_contract_remaining_runs.csv)

Bounded baseline uses T/K=12/6 for Conv2 and 24/8 for Conv3, for both algorithms. Other groups retain T/K=4/4, 6/6 and 8/8 by architecture. Cross-scheme comparisons therefore use different baseline relaxation counts; report training cost alongside accuracy. Earlier native-T/K baseline BPTT runs are excluded from these revised tables.

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
| conv2 | 0.0001 | 85.393 ± 0.142% | 90.193 ± 0.168% | 95.633 ± 0.205% |
| conv2 | 0.0005 | 92.040 ± 0.457% | 96.560 ± 0.120% | 97.680 ± 0.035% |
| conv2 | 0.001 | 93.553 ± 0.061% | 97.133 ± 0.064% | 97.873 ± 0.081% |
| conv3 | 0.0001 | 2/3 seeds | 85.133 ± 0.590% | 93.920 ± 0.971% |
| conv3 | 0.0005 | 2/3 seeds | 94.620 ± 0.208% | 97.827 ± 0.140% |
| conv3 | 0.001 | 2/3 seeds | 95.733 ± 0.061% | 98.007 ± 0.181% |

## Table 3: bounded EqProp

| Architecture | Gmax | Baseline | Ours | Legacy |
|---|---:|---:|---:|---:|
| conv1 | 0.0001 | 89.867 ± 0.167% | 93.053 ± 0.340% | 94.940 ± 0.060% |
| conv1 | 0.0005 | 94.567 ± 0.129% | 95.173 ± 0.250% | 96.367 ± 0.076% |
| conv1 | 0.001 | 94.960 ± 0.243% | 95.167 ± 0.239% | 96.413 ± 0.081% |
| conv2 | 0.0001 | 85.333 ± 0.103% | 90.187 ± 0.160% | 95.620 ± 0.178% |
| conv2 | 0.0005 | 2/3 seeds | 96.527 ± 0.110% | 97.693 ± 0.083% |
| conv2 | 0.001 | 1/3 seeds | 97.160 ± 0.053% | 97.860 ± 0.060% |
| conv3 | 0.0001 | 1/3 seeds | 85.327 ± 0.237% | 93.860 ± 0.866% |
| conv3 | 0.0005 | 0/3 seeds | 94.667 ± 0.380% | 97.840 ± 0.160% |
| conv3 | 0.001 | 0/3 seeds | 95.793 ± 0.081% | 97.960 ± 0.209% |
