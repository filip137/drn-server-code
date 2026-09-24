# Revised baseline EqProp pilot stability

Only complete, collected and validated 30-epoch seed-0 runs count. Collection verifies finite histories, float64 checkpoints, exact-zero biases, bounds, shared initialization and PT/NPZ equality. Stability additionally requires a best-to-final validation drop strictly below five percentage points. Official-test evaluations remain zero.

| Architecture | Gmax | T/K | Beta | Best/final validation | Drop (pp) | Full pilot | Result |
|---|---:|---:|---:|---:|---:|---|---|
| conv2 | 0.0001 | 12/6 | 0.1 | 85.42/85.00% | 0.42 | pass | [bundle](bundles/T3_EP_conv2_baseline_gmax0.0001_seed0/result.json) |
| conv2 | 0.0005 | 12/6 | 0.1 | 92.54/92.30% | 0.24 | pass | [bundle](bundles/T3_EP_conv2_baseline_gmax0.0005_seed0/result.json) |
| conv2 | 0.001 | 12/6 | 0.1 | 93.92/93.54% | 0.38 | pass | [bundle](bundles/T3_EP_conv2_baseline_gmax0.001_seed0/result.json) |
| conv3 | 0.0001 | 24/8 | 0.1 | 77.68/77.68% | 0.00 | pass | [bundle](bundles/T3_EP_conv3_baseline_gmax0.0001_seed0/result.json) |
| conv3 | 0.0005 | 24/8 | 0.1 | pending | pending | numerically_qualified_seed0_pilot_pending | pending |
| conv3 | 0.001 | 24/8 | 0.1 | pending | pending | numerically_qualified_seed0_pilot_pending | pending |

Full stable pilot coverage: **Conv2 3/3; Conv3 1/3.** Each architecture needs all three ceilings at the same qualified beta before its seeds 1/2. Passing coverage still requires frozen pilot proofs, exact smokes and budget admission before a repetition launches.

[Numerical beta qualification](beta_qualification.md) · [Training revision ledger](README.md)
