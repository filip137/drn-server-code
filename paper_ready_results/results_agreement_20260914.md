# Results and agreement with earlier runs — September 14

**201/216 full training runs are collected and validated.** New launches remain paused; only the current Loulou Conv3 EqProp pilot is still running.

All accuracies below are best validation accuracy (%) on the fixed 5,000-example ordinary-MNIST split. Official-test evaluations remain zero. This is a descriptive agreement analysis, not a statistical equivalence test.

## Agreement with earlier evidence

There are 32 unchanged conditions with genuinely earlier, reused seed-0 evidence. Compare the two newly trained seeds 1/2 against that earlier seed; the earlier seed is excluded from the new-run mean. The fixed train/validation membership matches throughout. Wide experiments differ by at most 0.18 percentage points. The largest bounded difference is Conv3 ours at Gmax=1e-4: 84.54% previously versus 85.43% for the new-seed mean (+0.89 pp). Conv3 legacy at that ceiling changes from 93.56% to 94.10% (+0.54 pp). These larger differences indicate seed sensitivity at the tight ceiling, while the scheme ordering is retained.

The old pre-correction legacy energy is a different model and is not used as a matched reference. Newly trained bounded EqProp pilots also are not historical pre-campaign EqProp results.

## Baseline change after increasing T

All 15 available old/new BPTT pairs match initialization, exact learning-rate vectors, dataset membership and all 30 minibatch orders. Conv2 changes from T6/K6 to T12/K6; Conv3 from T8/K8 to T24/K8. Hosts/software differ and are recorded in the source manifests.

| Architecture | Gmax | Seed | Original T best (%) | Larger T best (%) | Change (pp) |
|---|---:|---:|---:|---:|---:|
| conv2 | 0.0001 | 0 | 85.60 | 85.52 | -0.08 |
| conv2 | 0.0001 | 1 | 85.42 | 85.42 | +0.00 |
| conv2 | 0.0001 | 2 | 85.34 | 85.24 | -0.10 |
| conv2 | 0.0005 | 0 | 92.44 | 92.38 | -0.06 |
| conv2 | 0.0005 | 1 | 91.54 | 91.52 | -0.02 |
| conv2 | 0.0005 | 2 | 92.14 | 92.22 | +0.08 |
| conv2 | 0.001 | 0 | 93.52 | 93.54 | +0.02 |
| conv2 | 0.001 | 1 | 93.52 | 93.62 | +0.10 |
| conv2 | 0.001 | 2 | 93.34 | 93.50 | +0.16 |
| conv3 | 0.0001 | 0 | 77.66 | 77.68 | +0.02 |
| conv3 | 0.0001 | 1 | 77.56 | 77.50 | -0.06 |
| conv3 | 0.0005 | 0 | 87.44 | 87.12 | -0.32 |
| conv3 | 0.0005 | 1 | 87.86 | 87.36 | -0.50 |
| conv3 | 0.001 | 0 | 89.60 | 89.38 | -0.22 |
| conv3 | 0.001 | 1 | 89.76 | 89.58 | -0.18 |

Conv2's complete three-seed means change by -0.06, 0.00 and +0.09 pp across increasing ceilings. Conv3 is partial: seeds 0/1 change by +0.02/-0.06 pp at 1e-4, -0.32/-0.50 pp at 5e-4, and -0.22/-0.18 pp at 1e-3. Its seed 2 is still missing; no full three-seed baseline aggregate is reported for Conv3.

Increasing T therefore has little effect on these measured validation accuracies, while the separate numerical beta gate now passes. It increases the computational cost and is an intentional contract change.

## BPTT and EqProp agreement

Across 96 completed same-seed pairs, the maximum absolute best-accuracy difference is 0.66 pp. Across 31 complete three-seed condition pairs, the maximum mean difference is 0.193 pp. Revised Conv3 baseline EqProp remains untested by full-run completion.

| Architecture | Gmax | Scheme | BPTT mean ± sample SD (%) | EqProp mean ± sample SD (%) | EP − BPTT (pp) |
|---|---:|---|---:|---:|---:|
| conv1 | 100 | baseline | 96.267 ± 0.110 | 96.280 ± 0.092 | +0.013 |
| conv1 | 100 | legacy | 96.540 ± 0.053 | 96.533 ± 0.061 | -0.007 |
| conv1 | 100 | ours | 96.493 ± 0.076 | 96.453 ± 0.076 | -0.040 |
| conv2 | 100 | baseline | 97.340 ± 0.020 | 97.293 ± 0.081 | -0.047 |
| conv2 | 100 | legacy | 98.300 ± 0.040 | 98.340 ± 0.040 | +0.040 |
| conv2 | 100 | ours | 98.187 ± 0.076 | 98.153 ± 0.061 | -0.033 |
| conv3 | 100 | baseline | 97.767 ± 0.031 | 97.693 ± 0.083 | -0.073 |
| conv3 | 100 | legacy | 98.893 ± 0.061 | 98.913 ± 0.070 | +0.020 |
| conv3 | 100 | ours | 98.560 ± 0.100 | 98.600 ± 0.106 | +0.040 |
| conv1 | 0.0001 | baseline | 89.847 ± 0.160 | 89.867 ± 0.167 | +0.020 |
| conv1 | 0.0001 | legacy | 94.940 ± 0.087 | 94.940 ± 0.060 | -0.000 |
| conv1 | 0.0001 | ours | 93.040 ± 0.317 | 93.053 ± 0.340 | +0.013 |
| conv1 | 0.0005 | baseline | 94.580 ± 0.122 | 94.567 ± 0.129 | -0.013 |
| conv1 | 0.0005 | legacy | 96.360 ± 0.100 | 96.367 ± 0.076 | +0.007 |
| conv1 | 0.0005 | ours | 95.167 ± 0.223 | 95.173 ± 0.250 | +0.007 |
| conv1 | 0.001 | baseline | 94.953 ± 0.232 | 94.960 ± 0.243 | +0.007 |
| conv1 | 0.001 | legacy | 96.420 ± 0.080 | 96.413 ± 0.081 | -0.007 |
| conv1 | 0.001 | ours | 95.167 ± 0.204 | 95.167 ± 0.239 | +0.000 |
| conv2 | 0.0001 | baseline | 85.393 ± 0.142 | 85.333 ± 0.103 | -0.060 |
| conv2 | 0.0001 | legacy | 95.633 ± 0.205 | 95.620 ± 0.178 | -0.013 |
| conv2 | 0.0001 | ours | 90.193 ± 0.168 | 90.187 ± 0.160 | -0.007 |
| conv2 | 0.0005 | legacy | 97.680 ± 0.035 | 97.693 ± 0.083 | +0.013 |
| conv2 | 0.0005 | ours | 96.560 ± 0.120 | 96.527 ± 0.110 | -0.033 |
| conv2 | 0.001 | legacy | 97.873 ± 0.081 | 97.860 ± 0.060 | -0.013 |
| conv2 | 0.001 | ours | 97.133 ± 0.064 | 97.160 ± 0.053 | +0.027 |
| conv3 | 0.0001 | legacy | 93.920 ± 0.971 | 93.860 ± 0.866 | -0.060 |
| conv3 | 0.0001 | ours | 85.133 ± 0.590 | 85.327 ± 0.237 | +0.193 |
| conv3 | 0.0005 | legacy | 97.827 ± 0.140 | 97.840 ± 0.160 | +0.013 |
| conv3 | 0.0005 | ours | 94.620 ± 0.208 | 94.667 ± 0.380 | +0.047 |
| conv3 | 0.001 | legacy | 98.007 ± 0.181 | 97.960 ± 0.209 | -0.047 |
| conv3 | 0.001 | ours | 95.733 ± 0.061 | 95.793 ± 0.081 | +0.060 |

The newly collected Conv2 EqProp seed-1/Gmax5e-4 result is 91.50% versus its paired BPTT 91.52%; seed-2/Gmax1e-4 is 85.22% versus BPTT 85.24%. Both differ by -0.02 pp, and their complete 30-epoch cohort/order checks pass.

The scientific pattern remains: the completed amplification schemes exceed the baseline means, particularly with tight conductance bounds, and EqProp closely tracks its matched BPTT control. Three seeds and a selection split do not establish statistical equivalence or official-test performance.

![Agreement comparisons](figures/results_agreement_20260914.png)

[All current validation tables](current_contract_validation_tables.md) · [All individual results](current_contract_run_status.csv) · [Review and remaining work](review_20260914.md)

Source comparison CSVs and hash-identified inputs: [analysis directory](../results/paper-training-completion-20260911-v1/analysis/results_agreement_20260914/).
