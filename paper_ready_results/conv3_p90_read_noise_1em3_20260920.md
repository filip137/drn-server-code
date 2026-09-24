# Conv3 p90 read noise: sigma 1e-3 extension

Updated 2026-09-21T02:37:34.284568+00:00. 1/3 completed; 3/3 terminal.

Three new seed-0 RTX5090 runs, each with a planned 30-epoch budget, frozen p90 betas, T=K=8, float64 centered EqProp and unchanged Adam learning rates. Ordinary MNIST 55,000/5,000 train/validation; official test disabled. Independent Gaussian noise perturbs copied non-input endpoint voltages during gradient readout; relaxation and validation remain clean. Previous RTX5090 clean controls are reused; no prior noisy case is repeated. These are single-seed validation diagnostics.

| Scheme | Beta | State / completed epochs | Clean (%) | New final / best (%) | Clean-relative drop (pp) |
|---|---:|---|---:|---:|---:|
| baseline | 404.141 | complete / 30 | 97.72 | 97.16 / 97.20 | +0.56 |
| legacy | 4.4225 | failed / 7 | 98.70 | failed; no epoch-30 result | — |
| ours | 5.26876 | failed / 17 | 98.52 | failed; no epoch-30 result | — |

**legacy failure:** NonFiniteTrainingError: Expected inference layer 'Layer_1' to contain only finite values at epoch=8, batch=2791. Provided value: tensor with 200704 non-finite element(s).
Last complete validation epoch: 7, 94.76%; best observed: 97.22%. These partial values are not thirty-epoch results. The failure and saved best checkpoint are retained; no repeat is scheduled.

**ours failure:** NonFiniteTrainingError: Expected inference layer 'Layer_1' to contain only finite values at epoch=18, batch=1638. Provided value: tensor with 200704 non-finite element(s).
Last complete validation epoch: 17, 95.14%; best observed: 96.40%. These partial values are not thirty-epoch results. The failure and saved best checkpoint are retained; no repeat is scheduled.

All three planned cases have terminal outcomes; 1 reached epoch 30. Scientific failures are included in coverage and are not treated as missing runs.

The zero-noise per-matrix cosine >0.90 selection and successful clean training do not guarantee finite training at sigma=1e-3. These observations compare fixed scheme-specific beta/LR operating points in one seed; they do not establish an intrinsic or universal amplification ranking.

Positive drop means worse accuracy with noise. All scientific outcomes are retained. The endpoint screen requires the final accuracy to be strictly less than 5pp below its own best; the CSV also records temporary running-best drawdown. A passing endpoint screen does not establish smooth training or noise robustness.

[Previous completed sweep](conv3_p90_read_noise_20260919.md) · [Launch plan](../docs/eqprop_conv3_p90_read_noise_1em3_plan_20260920.md)

![Epoch trajectories](conv3_p90_read_noise_1em3_20260920_epochs.png)
