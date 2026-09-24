# Best versus final validation accuracy

Snapshot: 2026-09-16. Comparison uses the same 165 trainings as the three manuscript tables: 54 wide BPTT/EqProp, 81 shared-T/K bounded BPTT, and 30 ours/legacy read-noise runs. All values and source-result hashes were rechecked against the saved result bundles. No manuscript or experiment settings were changed.

All values are validation accuracy percentages. Final means the last epoch of the full 10/30/30-epoch training budget. Clean tables use mean ± sample SD across seeds 0/1/2; noisy runs use seed 0 only. This is not an official-test evaluation.

| Comparison | Change from best to final | Consequence |
|---|---|---|
| Wide clean training | Condition means decrease by 0.013–0.167 percentage points | All six scheme rankings unchanged; largest BPTT/EqProp mean difference decreases from 0.073 to 0.047 points |
| Bounded BPTT | 26 of 27 condition means decrease by less than 0.5 points; largest decrease is 1.187 points | All nine scheme rankings unchanged |
| Read noise | Ours decreases by at most 0.30 points; legacy by at most 2.58 points | Ours remains ahead at every noise level in Conv2 and Conv3 |

The largest bounded change is Conv1 baseline at Gmax=1e-4: 89.85 ± 0.16 becomes 88.66 ± 1.24. Its seed-0 run drops 2.42 points from its own best. Thus the final metric reveals more between-seed variability in this condition.

The only noisy scheme-order reversal is Conv1 at sigma=5e-4: best accuracy puts ours ahead by 0.02 points (96.50 versus 96.48); final puts legacy ahead by 0.10 points (96.38 versus 96.28). These are small differences on one validation seed.

At sigma=5e-4, Conv3 ours changes from best 96.78 to final 96.60, and legacy from 76.36 to 75.44. Using final clean references consistently, the corresponding noise penalties are 2.04 and 23.34 points, versus 1.94 and 22.48 with best checkpoints.

The tables below show the complete final-accuracy alternative. The metric choice changes the checkpoint-selection rule: final reports the fixed-budget endpoint, while best reports a checkpoint selected by validation accuracy. Both checkpoints are already saved; changing the displayed validation metric needs no retraining. The final paper rule still needs to be frozen before official-test evaluation.

Table 1: wide-range BPTT and EqProp, final validation accuracy.

| Architecture | Algorithm | Baseline | Ours | Legacy |
|---|---|---:|---:|---:|
| Conv1 | BPTT | 96.18 ± 0.16 | 96.33 ± 0.03 | 96.49 ± 0.05 |
| Conv1 | EqProp | 96.20 ± 0.14 | 96.35 ± 0.04 | 96.49 ± 0.04 |
| Conv2 | BPTT | 97.29 ± 0.06 | 98.04 ± 0.11 | 98.17 ± 0.10 |
| Conv2 | EqProp | 97.25 ± 0.04 | 98.07 ± 0.05 | 98.22 ± 0.14 |
| Conv3 | BPTT | 97.67 ± 0.08 | 98.55 ± 0.12 | 98.82 ± 0.04 |
| Conv3 | EqProp | 97.66 ± 0.07 | 98.53 ± 0.12 | 98.85 ± 0.06 |

Table 2: bounded BPTT at shared T/K, final validation accuracy. Gmin=1e-5.

| Architecture | Gmax | Baseline | Ours | Legacy |
|---|---:|---:|---:|---:|
| Conv1 | 0.0001 | 88.66 ± 1.24 | 92.57 ± 0.96 | 94.60 ± 0.33 |
| Conv1 | 0.0005 | 94.29 ± 0.44 | 94.98 ± 0.41 | 96.27 ± 0.11 |
| Conv1 | 0.001 | 94.77 ± 0.49 | 95.03 ± 0.36 | 96.33 ± 0.16 |
| Conv2 | 0.0001 | 85.22 ± 0.16 | 90.07 ± 0.21 | 95.56 ± 0.31 |
| Conv2 | 0.0005 | 92.00 ± 0.41 | 96.49 ± 0.24 | 97.63 ± 0.08 |
| Conv2 | 0.001 | 93.43 ± 0.10 | 97.07 ± 0.08 | 97.80 ± 0.00 |
| Conv3 | 0.0001 | 77.24 ± 0.44 | 85.04 ± 0.71 | 93.92 ± 0.97 |
| Conv3 | 0.0005 | 87.87 ± 0.43 | 94.62 ± 0.21 | 97.81 ± 0.13 |
| Conv3 | 0.001 | 89.80 ± 0.22 | 95.71 ± 0.09 | 98.00 ± 0.18 |

Table 3: read-noise training, final validation accuracy, one seed. Baseline noisy coverage is unchanged and absent. Clean references also use final accuracy.

| Architecture | Scheme | Clean | 1e-5 | 3e-5 | 1e-4 | 3e-4 | 5e-4 |
|---|---|---:|---:|---:|---:|---:|---:|
| Conv1 | ours | 96.34 | 96.34 | 96.30 | 96.34 | 96.30 | 96.28 |
| Conv1 | legacy | 96.44 | 96.40 | 96.40 | 96.40 | 96.36 | 96.38 |
| Conv2 | ours | 98.02 | 98.08 | 98.08 | 98.00 | 97.80 | 97.60 |
| Conv2 | legacy | 98.06 | 97.62 | 97.38 | 96.52 | 95.94 | 95.72 |
| Conv3 | ours | 98.64 | 98.26 | 97.82 | 97.62 | 96.82 | 96.60 |
| Conv3 | legacy | 98.78 | 95.68 | 93.42 | 83.08 | 82.34 | 75.44 |

[Per-run best/final values and source hashes](three_table_overview_20260916.csv) · [Original three-table overview](../docs/paper_three_table_overview_20260916.md)
