# Beta recheck on newly trained baseline references

Updated 2026-09-14T04:35:02.166064+00:00. 6 full ceiling/beta checks completed and collected. Smoke checks are excluded from qualification coverage.

Each full check covers initialization and the newly trained seed-0 BPTT best checkpoint on four fixed validation batches. Every weight layer must have cosine ≥ .99 and symmetric norm difference ≤ .10, with every projected-residual p90 ≤ .01. Source files and parameters remain unchanged; official-test evaluations and optimizer steps are zero.

| Architecture | Passing common candidates | Coverage and next gate |
|---|---|---|
| conv2 | 0.1 | Numerically qualified; all three full seed-0 EqProp pilots still required |
| conv3 | 0.1 | Numerically qualified; all three full seed-0 EqProp pilots still required |

| Architecture | Ceiling | T/K | Beta | Minimum cosine | Maximum norm difference | Residual | Combined gate | Result |
|---|---|---:|---:|---:|---:|---|---|---|
| conv2 | 1em3 | 12/6 | 0.1 | 0.997674 | 0.019507 | pass | pass | [bundle](../provenance/baseline_tk_revision_best_gate/runs/conv2_1em3_beta0.1/result.json) |
| conv2 | 1em4 | 12/6 | 0.1 | 0.997674 | 0.019507 | pass | pass | [bundle](../provenance/baseline_tk_revision_best_gate/runs/conv2_1em4_beta0.1/result.json) |
| conv2 | 5em4 | 12/6 | 0.1 | 0.997674 | 0.019507 | pass | pass | [bundle](../provenance/baseline_tk_revision_best_gate/runs/conv2_5em4_beta0.1/result.json) |
| conv3 | 1em3 | 24/8 | 0.1 | 0.997096 | 0.068717 | pass | pass | [bundle](../provenance/baseline_tk_revision_best_gate/runs/conv3_1em3_beta0.1/result.json) |
| conv3 | 1em4 | 24/8 | 0.1 | 0.997096 | 0.047329 | pass | pass | [bundle](../provenance/baseline_tk_revision_best_gate/runs/conv3_1em4_beta0.1/result.json) |
| conv3 | 5em4 | 24/8 | 0.1 | 0.997096 | 0.047329 | pass | pass | [bundle](../provenance/baseline_tk_revision_best_gate/runs/conv3_5em4_beta0.1/result.json) |

[Training revision tracker](README.md) · [All measurements](beta_qualification.csv) · [Earlier T/K diagnostics on native-trained checkpoints](../baseline_tk_beta_qualification.md)
