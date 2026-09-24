# Per-matrix cosine beta selection: ten-epoch training

Updated 2026-09-19T11:32:31.772689+00:00. **9/9 distinct settings terminal; 12 threshold conditions.**

Seed 0, zero read noise, fixed Adam rates, T=K=6/8, float64 centered EqProp; ordinary-MNIST validation only.
Select the largest tested injected beta whose cosine is strictly above the threshold for every weight matrix on every one of 36 batches at both initialization and the saved BPTT checkpoint. No norm gate is imposed; norm mismatch is reported below.

| Model | Scheme | Cosine threshold | Beta | Worst cosine | Worst norm mismatch |
|---|---|---:|---:|---:|---:|
| conv2 | baseline | 0.90 | 500 | 0.957271 | 0.102711 |
| conv2 | baseline | 0.95 | 500 | 0.957271 | 0.102711 |
| conv2 | ours | 0.90 | 50 | 0.929088 | 0.173502 |
| conv2 | ours | 0.95 | 30 | 0.961044 | 0.122724 |
| conv2 | legacy | 0.90 | 30 | 0.954935 | 0.127926 |
| conv2 | legacy | 0.95 | 30 | 0.954935 | 0.127926 |
| conv3 | baseline | 0.90 | 300 | 0.931421 | 0.286851 |
| conv3 | baseline | 0.95 | 100 | 0.967393 | 0.184260 |
| conv3 | ours | 0.90 | 3 | 0.939168 | 0.278178 |
| conv3 | ours | 0.95 | 0.9 | 0.992230 | 0.108906 |
| conv3 | legacy | 0.90 | 1 | 0.960632 | 0.116789 |
| conv3 | legacy | 0.95 | 1 | 0.960632 | 0.116789 |

Legacy selections are passing upper grid edges, not known maxima. Equal selected betas share one outcome.

| Model | Scheme | Beta | Outcome | Final / best validation | Previous-beta control (epoch 10) | Delta (pp) | Evidence |
|---|---|---:|---|---|---|---:|---|
| conv2 | baseline | 500 | Ten-epoch stable | 96.96% / 96.96% | 96.96% (beta 100) | +0.00 | new |
| conv2 | ours | 50 | Ten-epoch stable | 97.96% / 97.96% | 97.88% (beta 10) | +0.08 | new |
| conv2 | ours | 30 | Ten-epoch stable | 97.84% / 97.86% | 97.88% (beta 10) | -0.04 | new |
| conv2 | legacy | 30 | Nonfinite epoch 5, batch 1910 | — | 98.00% (beta 0.03) | — | reuse_pilot |
| conv3 | baseline | 300 | Ten-epoch stable | 97.02% / 97.02% | 97.02% (beta 100) | +0.00 | new |
| conv3 | baseline | 100 | Ten-epoch stable | 97.02% / 97.02% | 97.02% (beta 100) | +0.00 | reuse_control_prefix |
| conv3 | ours | 3 | Ten-epoch stable | 98.42% / 98.42% | 98.42% (beta 3) | +0.00 | reuse_control_prefix |
| conv3 | ours | 0.9 | Ten-epoch stable | 98.46% / 98.46% | 98.42% (beta 3) | +0.04 | new |
| conv3 | legacy | 1 | Ten-epoch stable | 98.64% / 98.64% | 98.64% (beta 0.001) | +0.00 | reuse_pilot |

| Model | Scheme | beta .90 / .95 | Larger minus smaller final accuracy (pp) | Interpretation |
|---|---|---|---:|---|
| conv2 | baseline | 500 / 500 | — | Same selected beta; no distinct threshold comparison |
| conv2 | ours | 50 / 30 | +0.12 | Larger beta has higher final validation |
| conv2 | legacy | 30 / 30 | — | Same selected beta; no distinct threshold comparison |
| conv3 | baseline | 300 / 100 | +0.00 | Equal final validation |
| conv3 | ours | 3 / 0.9 | -0.04 | Larger beta has lower final validation |
| conv3 | legacy | 1 / 1 | — | Same selected beta; no distinct threshold comparison |

**Interpretation:** 8/9 distinct settings passed ten finite epochs and the final-drop criterion. The distinct .90-versus-.95 comparisons show no consistent accuracy penalty from larger beta: conv2 ours +0.12pp; conv3 baseline +0.00pp; conv3 ours -0.04pp.
Conv2 legacy beta30 is the exception to training stability: despite worst per-matrix cosine .954935, the matching prior pilot became nonfinite in epoch 5, batch 1910; its beta.03 control reached 98.00% at epoch 10. Thus neither tested static cosine threshold guarantees stable training. Calibration measures two fixed reference parameter states, not every state visited by the new EqProp training trajectory.

Five settings are newly trained; four reuse explicitly named matching evidence. Historical 30-epoch controls contribute only their first ten epoch records; their full-horizon best checkpoints are not treated as epoch-10 checkpoints.
The two Conv2 ours settings were restarted together on the local RTX 3090 after unrelated GPU clients arrived on Loulou. The three-epoch Loulou attempts are preserved but superseded solely for resource contention, not numerical failure. Two explicitly excluded 100-batch throughput probes informed this placement change; an initial local transport attempt exited before smoke or training. Thus five new scientific settings used seven full training attempts, with only the designated replacement outcomes included.
All new runs retain the original learning rates, initialization and data order. New/reused host placement is recorded in cases.json. This is an exploratory multi-host comparison.
Stability means ten finite epochs and final validation less than 5pp below the run's best. Numerical failures remain included; no epoch-10 accuracy is invented for them. Small accuracy differences describe one seed and are not statistically established effects. No official test, noisy training qualification, full 30-epoch qualification, or automatic beta promotion.

[Validation curves](layerwise_beta_training_20260919.png) · [PDF](layerwise_beta_training_20260919.pdf) · [Train/validation losses and accuracies by epoch](layerwise_beta_training_20260919_epochs.csv) · [Plan](../docs/eqprop_layerwise_beta_training_plan_20260919.md) · [Raw evidence](../results/eqprop-layerwise-beta-training-20260919-v1/)
