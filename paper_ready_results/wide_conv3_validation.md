# Wide Conv3: completed three-seed validation comparison

All 18 trainings are collected and validated locally. Values use the fixed 5,000-example selection split; official-test results remain unavailable.

| Scheme | Algorithm | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
|---|---|---:|---:|---:|---:|
| baseline | BPTT | [97.80%](bundles/table1_wide_bptt/conv3/baseline/seed0/result.json) | [97.74%](bundles/table1_wide_bptt/conv3/baseline/seed1/result.json) | [97.76%](bundles/table1_wide_bptt/conv3/baseline/seed2/result.json) | 97.767 ± 0.031% |
| baseline | EP | [97.76%](bundles/table2_wide_ep/conv3/baseline/seed0/result.json) | [97.60%](bundles/table2_wide_ep/conv3/baseline/seed1/result.json) | [97.72%](bundles/table2_wide_ep/conv3/baseline/seed2/result.json) | 97.693 ± 0.083% |
| ours | BPTT | [98.66%](bundles/table1_wide_bptt/conv3/ours/seed0/result.json) | [98.46%](bundles/table1_wide_bptt/conv3/ours/seed1/result.json) | [98.56%](bundles/table1_wide_bptt/conv3/ours/seed2/result.json) | 98.560 ± 0.100% |
| ours | EP | [98.72%](bundles/table2_wide_ep/conv3/ours/seed0/result.json) | [98.52%](bundles/table2_wide_ep/conv3/ours/seed1/result.json) | [98.56%](bundles/table2_wide_ep/conv3/ours/seed2/result.json) | 98.600 ± 0.106% |
| legacy | BPTT | [98.84%](bundles/table1_wide_bptt/conv3/legacy/seed0/result.json) | [98.96%](bundles/table1_wide_bptt/conv3/legacy/seed1/result.json) | [98.88%](bundles/table1_wide_bptt/conv3/legacy/seed2/result.json) | 98.893 ± 0.061% |
| legacy | EP | [98.84%](bundles/table2_wide_ep/conv3/legacy/seed0/result.json) | [98.98%](bundles/table2_wide_ep/conv3/legacy/seed1/result.json) | [98.92%](bundles/table2_wide_ep/conv3/legacy/seed2/result.json) | 98.913 ± 0.070% |

Mean best-validation EP − BPTT differences (percentage points): baseline -0.073, ours +0.040, legacy +0.020. The largest absolute paired-seed difference is 0.140 points.

All nine pairs have identical recorded training/validation cohorts and all 30 epoch minibatch orders. The six seed-1/2 pairs also have verified shared initializer assets and matching BPTT numerical initial-state hashes. The retained baseline/ours BPTT seed-0 bundles lack recorded initial-state hashes; exact seed-0 initialization reuse eligibility remains a separate audit. All original wide Conv3 gradient/residual caveats remain in the completion manifest.

The inherited Adam vectors, T=K=8, [0,100] weight bounds and zero-bias contract remain fixed. BPTT uses float32 and EqProp float64, with injected betas baseline 100, ours 3, legacy .001. No seed is excluded. These validation measurements describe the tested runs; they do not establish statistical equivalence or official-test performance.

[Pairing audit](provenance/wide_conv3_pairing.json) · [Best/final aggregates](validation_table_summary.csv) · [Completion manifest](../docs/paper_ready_results_manifest.md)
