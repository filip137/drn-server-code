# Wide Conv1: completed three-seed validation comparison

Updated 2026-09-11T14:15:50.237874+00:00. All 18 training bundles are collected and validated locally.

These are best-validation accuracies on the fixed 5,000-example selection split. They are not official-test results. Mean and sample standard deviation are across model/shuffle seeds 0, 1, and 2; each run keeps its own validation-selected checkpoint.

| Scheme | Algorithm | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
|---|---|---:|---:|---:|---:|
| baseline | BPTT | [96.16%](bundles/table1_wide_bptt/conv1/baseline/seed0/result.json) | [96.26%](bundles/table1_wide_bptt/conv1/baseline/seed1/result.json) | [96.38%](bundles/table1_wide_bptt/conv1/baseline/seed2/result.json) | 96.267 ± 0.110% |
| baseline | EqProp | [96.18%](bundles/table2_wide_ep/conv1/baseline/seed0/result.json) | [96.30%](bundles/table2_wide_ep/conv1/baseline/seed1/result.json) | [96.36%](bundles/table2_wide_ep/conv1/baseline/seed2/result.json) | 96.280 ± 0.092% |
| ours | BPTT | [96.44%](bundles/table1_wide_bptt/conv1/ours/seed0/result.json) | [96.46%](bundles/table1_wide_bptt/conv1/ours/seed1/result.json) | [96.58%](bundles/table1_wide_bptt/conv1/ours/seed2/result.json) | 96.493 ± 0.076% |
| ours | EqProp | [96.42%](bundles/table2_wide_ep/conv1/ours/seed0/result.json) | [96.40%](bundles/table2_wide_ep/conv1/ours/seed1/result.json) | [96.54%](bundles/table2_wide_ep/conv1/ours/seed2/result.json) | 96.453 ± 0.076% |
| legacy | BPTT | [96.50%](bundles/table1_wide_bptt/conv1/legacy/seed0/result.json) | [96.52%](bundles/table1_wide_bptt/conv1/legacy/seed1/result.json) | [96.60%](bundles/table1_wide_bptt/conv1/legacy/seed2/result.json) | 96.540 ± 0.053% |
| legacy | EqProp | [96.48%](bundles/table2_wide_ep/conv1/legacy/seed0/result.json) | [96.52%](bundles/table2_wide_ep/conv1/legacy/seed1/result.json) | [96.60%](bundles/table2_wide_ep/conv1/legacy/seed2/result.json) | 96.533 ± 0.061% |

EqProp minus BPTT differences in mean best-validation accuracy are baseline +0.013, ours -0.040, legacy -0.007 percentage points. The largest absolute paired-seed difference is 0.06 percentage points. These small observed differences support consistency of the frozen wide Conv1 training settings across these seeds; three seeds on the selection split do not establish statistical equivalence or test-set performance.

All runs use the inherited Adam vectors, zero biases, T=K=4, matched initialization and split/order rules, and 10 epochs. BPTT uses float32 and EqProp true float64. EqProp injected betas are 100 / 30 / 3 for baseline / ours / legacy. Original seed-0 provenance and reuse caveats remain in the [completion manifest](../docs/paper_ready_results_manifest.md); no completed training block is yet sealed for paper-facing accuracy.
