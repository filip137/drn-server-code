# Wide Conv2: completed three-seed validation comparison

Updated 2026-09-11T17:10:39.127311+00:00. All 18 training bundles are collected and validated locally.

These are best-validation accuracies on the fixed 5,000-example selection split. Mean and sample standard deviation use model/shuffle seeds 0, 1, and 2, with each run retaining its validation-selected checkpoint. Official-test results remain unavailable.

| Scheme | Algorithm | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
|---|---|---:|---:|---:|---:|
| baseline | BPTT | [97.36%](bundles/table1_wide_bptt/conv2/baseline/seed0/result.json) | [97.32%](bundles/table1_wide_bptt/conv2/baseline/seed1/result.json) | [97.34%](bundles/table1_wide_bptt/conv2/baseline/seed2/result.json) | 97.340 ± 0.020% |
| baseline | EqProp | [97.22%](bundles/table2_wide_ep/conv2/baseline/seed0/result.json) | [97.38%](bundles/table2_wide_ep/conv2/baseline/seed1/result.json) | [97.28%](bundles/table2_wide_ep/conv2/baseline/seed2/result.json) | 97.293 ± 0.081% |
| ours | BPTT | [98.10%](bundles/table1_wide_bptt/conv2/ours/seed0/result.json) | [98.22%](bundles/table1_wide_bptt/conv2/ours/seed1/result.json) | [98.24%](bundles/table1_wide_bptt/conv2/ours/seed2/result.json) | 98.187 ± 0.076% |
| ours | EqProp | [98.10%](bundles/table2_wide_ep/conv2/ours/seed0/result.json) | [98.14%](bundles/table2_wide_ep/conv2/ours/seed1/result.json) | [98.22%](bundles/table2_wide_ep/conv2/ours/seed2/result.json) | 98.153 ± 0.061% |
| legacy | BPTT | [98.30%](bundles/table1_wide_bptt/conv2/legacy/seed0/result.json) | [98.26%](bundles/table1_wide_bptt/conv2/legacy/seed1/result.json) | [98.34%](bundles/table1_wide_bptt/conv2/legacy/seed2/result.json) | 98.300 ± 0.040% |
| legacy | EqProp | [98.30%](bundles/table2_wide_ep/conv2/legacy/seed0/result.json) | [98.34%](bundles/table2_wide_ep/conv2/legacy/seed1/result.json) | [98.38%](bundles/table2_wide_ep/conv2/legacy/seed2/result.json) | 98.340 ± 0.040% |

EqProp minus BPTT differences in mean best-validation accuracy are baseline -0.047, ours -0.033, legacy +0.040 percentage points. The largest absolute paired-seed difference is 0.14 percentage points. These measurements support consistency of the frozen wide Conv2 training settings across the tested seeds; they do not establish statistical equivalence or official-test performance.

All runs use the inherited Adam vectors, exact-zero biases, T=K=6, shared initialization rules, matched training/validation cohorts and all 30 epoch minibatch orders. BPTT uses float32 and EqProp true float64. EqProp injected betas are 100 / 10 / .03 for baseline / ours / legacy. Original seed-0 reuse caveats remain in the [completion manifest](../docs/paper_ready_results_manifest.md).

The [pairing audit](provenance/wide_conv2_pairing.json) records all nine algorithm/seed comparisons and their source result hashes.
