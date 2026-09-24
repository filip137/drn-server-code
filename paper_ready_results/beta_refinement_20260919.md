# Refined beta cosine boundaries

Status: complete. 88 new beta settings collected and validated.

Zero-noise Conv2/Conv3, seed0, T=K6/8; unchanged 36-batch validation-partition cohort at initializer and saved BPTT checkpoint. Every weight matrix and every replay must pass strictly. Norm mismatch is diagnostic, without a norm gate.

| Model | Scheme | Threshold | Previous beta | Refined passing beta | Next failing beta | Bracket width | Worst cosine |
|---|---|---:|---:|---:|---:|---:|---:|
| conv2 | baseline | 0.90 | 500 | 720.198 | 750 | 4.14% | 0.907825 |
| conv2 | baseline | 0.95 | 500 | 542.236 | 564.673 | 4.14% | 0.950824 |
| conv2 | ours | 0.90 | 50 | 63.7712 | 66.4101 | 4.14% | 0.902953 |
| conv2 | ours | 0.95 | 30 | 34.0866 | 35.5689 | 4.35% | 0.953031 |
| conv2 | legacy | 0.90 | 30 | 43.0177 | 45 | 4.61% | 0.902998 |
| conv2 | legacy | 0.95 | 30 | 31.3825 | 32.8286 | 4.61% | 0.951788 |
| conv3 | baseline | 0.90 | 300 | 404.141 | 421.716 | 4.35% | 0.901977 |
| conv3 | baseline | 0.95 | 100 | 147.683 | 154.221 | 4.43% | 0.951316 |
| conv3 | ours | 0.90 | 3 | 5.26876 | 5.50202 | 4.43% | 0.900348 |
| conv3 | ours | 0.95 | 0.9 | 2.49275 | 2.61089 | 4.74% | 0.950604 |
| conv3 | legacy | 0.90 | 1 | 4.4225 | 4.6263 | 4.61% | 0.902633 |
| conv3 | legacy | 0.95 | 1 | 2.81845 | 2.94833 | 4.61% | 0.951908 |

These are sampled crossing intervals, not a proof of a globally monotone cosine function or an exact maximum. At calibration closeout, passing values were calibration-qualified only. Subsequently, all six refined Conv3 values completed ten-epoch clean training, and the three p90 choices completed the 24-outcome thirty-epoch read-noise study with GPU-matched clean controls. The six refined Conv2 values still lack new training qualification. See the [training report](conv3_refined_beta_training_20260919.md) and [completed read-noise report](conv3_p90_read_noise_20260919.md).
The figure zooms around the threshold crossings; it does not display every measured beta or every training outcome. The CSVs retain the complete grids.
Original measurements and all new measurements are retained. Both production workers used the local RTX3090; remote staging smokes and refused/aborted admissions are excluded. See the plan and handoff for transport revisions.

[Figure](beta_refinement_20260919.png) · [PDF](beta_refinement_20260919.pdf) · [Measurements](beta_refinement_20260919_metrics.csv) · [Selections](beta_refinement_20260919_selections.csv) · [Plan](../docs/eqprop_beta_refinement_plan_20260919.md)
