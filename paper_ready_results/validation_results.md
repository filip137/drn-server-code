# Collected validation results

Updated: 2026-09-11. Every value below is **best / final validation accuracy (%) for seed 0**, measured on the 5,000-example ordinary-MNIST validation split. These are selection results, not official-test accuracy. Seeds 1 and 2 are still missing throughout; no three-seed mean or standard deviation is available.

The collection follows the clean Adam [launch plan](provenance/references/papers/amplification_overleaf/experiment_launch_plan_20260911.md). Legacy entries use the corrected physical-KCL training runs. The [experimental tracker](../docs/paper_ready_results_manifest.md) records the remaining work, and [collected_results.csv](collected_results.csv) provides source paths and SHA-256 hashes for all 38 included bundles.

## Table 1 — wide BPTT, seed 0

Conductance interval `[0,100]`; float32 checkpoints; 10/30/30 epochs for Conv1/2/3.

| Architecture | Baseline `(1,1)` | Ours `(4,1)` | Legacy `(4,0.25)` |
|---|---|---|---|
| Conv1 | [96.16 / 96.00](bundles/table1_wide_bptt/conv1/baseline/seed0/result.json) | [96.44 / 96.32](bundles/table1_wide_bptt/conv1/ours/seed0/result.json) | [96.50 / 96.44](bundles/table1_wide_bptt/conv1/legacy/seed0/result.json) |
| Conv2 | [97.36 / 97.36](bundles/table1_wide_bptt/conv2/baseline/seed0/result.json) | [98.10 / 98.08](bundles/table1_wide_bptt/conv2/ours/seed0/result.json) | [98.30 / 98.08](bundles/table1_wide_bptt/conv2/legacy/seed0/result.json) |
| Conv3 | [97.80 / 97.70](bundles/table1_wide_bptt/conv3/baseline/seed0/result.json) | [98.66 / 98.66](bundles/table1_wide_bptt/conv3/ours/seed0/result.json) | [98.84 / 98.78](bundles/table1_wide_bptt/conv3/legacy/seed0/result.json) |

## Table 2 — wide centered EP, seed 0

Conductance interval `[0,100]`; true float64 centered frozen-current EP; inherited one-decade injected beta. Corrected Conv1 legacy EP remains to be trained.

| Architecture | Baseline `(1,1)` | Ours `(4,1)` | Legacy `(4,0.25)` |
|---|---|---|---|
| Conv1 | [96.18 / 96.04](bundles/table2_wide_ep/conv1/baseline/seed0/result.json) | [96.42 / 96.34](bundles/table2_wide_ep/conv1/ours/seed0/result.json) | Pending corrected training |
| Conv2 | [97.22 / 97.22](bundles/table2_wide_ep/conv2/baseline/seed0/result.json) | [98.10 / 98.02](bundles/table2_wide_ep/conv2/ours/seed0/result.json) | [98.30 / 98.06](bundles/table2_wide_ep/conv2/legacy/seed0/result.json) |
| Conv3 | [97.76 / 97.68](bundles/table2_wide_ep/conv3/baseline/seed0/result.json) | [98.72 / 98.64](bundles/table2_wide_ep/conv3/ours/seed0/result.json) | [98.84 / 98.78](bundles/table2_wide_ep/conv3/legacy/seed0/result.json) |

## Table 3 — bounded BPTT, seed 0

`G_min=1e-5`, initialization from the same architecture-specific `Uniform[1e-5,1e-4)` asset at every ceiling. Float32 BPTT checkpoints. Corrected Conv1/Conv2 legacy rows remain to be trained.

| Architecture | G_max | Baseline `(1,1)` | Ours `(4,1)` | Legacy `(4,0.25)` |
|---|---|---|---|---|
| Conv1 | 1e-4 | [89.68 / 87.26](bundles/table3_bounded_bptt/conv1/baseline/gmax_1em4/seed0/result.json) | [92.92 / 91.52](bundles/table3_bounded_bptt/conv1/ours/gmax_1em4/seed0/result.json) | Pending corrected training |
| Conv1 | 5e-4 | [94.44 / 93.80](bundles/table3_bounded_bptt/conv1/baseline/gmax_5em4/seed0/result.json) | [95.00 / 94.60](bundles/table3_bounded_bptt/conv1/ours/gmax_5em4/seed0/result.json) | Pending corrected training |
| Conv1 | 1e-3 | [94.80 / 94.24](bundles/table3_bounded_bptt/conv1/baseline/gmax_1em3/seed0/result.json) | [95.02 / 94.68](bundles/table3_bounded_bptt/conv1/ours/gmax_1em3/seed0/result.json) | Pending corrected training |
| Conv2 | 1e-4 | [85.60 / 85.04](bundles/table3_bounded_bptt/conv2/baseline/gmax_1em4/seed0/result.json) | [90.30 / 90.30](bundles/table3_bounded_bptt/conv2/ours/gmax_1em4/seed0/result.json) | Pending corrected training |
| Conv2 | 5e-4 | [92.44 / 92.32](bundles/table3_bounded_bptt/conv2/baseline/gmax_5em4/seed0/result.json) | [96.56 / 96.56](bundles/table3_bounded_bptt/conv2/ours/gmax_5em4/seed0/result.json) | Pending corrected training |
| Conv2 | 1e-3 | [93.52 / 93.46](bundles/table3_bounded_bptt/conv2/baseline/gmax_1em3/seed0/result.json) | [97.18 / 97.04](bundles/table3_bounded_bptt/conv2/ours/gmax_1em3/seed0/result.json) | Pending corrected training |
| Conv3 | 1e-4 | [77.66 / 77.66](bundles/table3_bounded_bptt/conv3/baseline/gmax_1em4/seed0/result.json) | [84.54 / 84.30](bundles/table3_bounded_bptt/conv3/ours/gmax_1em4/seed0/result.json) | [93.56 / 93.56](bundles/table3_bounded_bptt/conv3/legacy/gmax_1em4/seed0/result.json) |
| Conv3 | 5e-4 | [87.44 / 87.44](bundles/table3_bounded_bptt/conv3/baseline/gmax_5em4/seed0/result.json) | [94.74 / 94.74](bundles/table3_bounded_bptt/conv3/ours/gmax_5em4/seed0/result.json) | [97.68 / 97.66](bundles/table3_bounded_bptt/conv3/legacy/gmax_5em4/seed0/result.json) |
| Conv3 | 1e-3 | [89.60 / 89.60](bundles/table3_bounded_bptt/conv3/baseline/gmax_1em3/seed0/result.json) | [95.80 / 95.80](bundles/table3_bounded_bptt/conv3/ours/gmax_1em3/seed0/result.json) | [97.80 / 97.80](bundles/table3_bounded_bptt/conv3/legacy/gmax_1em3/seed0/result.json) |

## Table 3 — bounded EP

All 81 training results are pending. The 27 architecture/scheme/ceiling conditions need the short beta/gradient checks in the launch plan before their three seeds are trained.

## Evidence limits

The collection preserves source manifests and results without relabeling their evidence classes. Some legacy trainer filenames contain `test` (for example `accuracy_test.npy`); their canonical manifests identify those arrays as **validation**, and every included run records `official_test_read=false`.

BPTT and EP have different recorded precision. One-decade EP stability does not establish exact equilibrium-gradient fidelity; the existing Conv3 residual and gradient caveats remain in the source protocol. The final matching/reuse audit, contract and checkpoint seal, and one official-test evaluation per eligible run are still pending. Missing or poor outcomes must stay visible when completing the tables.
