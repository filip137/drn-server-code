# Conv3 p95: read noise 1e-3 versus clean

0/6 completed; 0/6 terminal. Updated 2026-09-21T10:45:15.653094+00:00.

Thirty epochs, seed 0, fixed per-matrix cosine >.95 betas, T=K=8, float64 centered EqProp, unchanged Adam rates and matched V100 controls. MNIST 55k/5k validation; official test disabled. Endpoint noise leaves relaxation and validation clean. Finite-T residual caveat retained.

| Scheme | Sigma | State | Epochs | Final / best validation (%) | Clean-relative drop (pp) |
|---|---:|---|---:|---:|---:|
| baseline | 0.001 | pending | 0 | — | — |
| ours | 0.001 | pending | 0 | — | — |
| legacy | 0.001 | pending | 0 | — | — |
| baseline | 0 | pending | 0 | — | — |
| ours | 0 | pending | 0 | — | — |
| legacy | 0 | pending | 0 | — | — |

The stability screen requires 30 finite epochs and final accuracy strictly less than 5pp below its own best. Temporary drawdown is reported separately; passing does not imply smooth training. Scientific failures remain included. If a clean control is unstable, its paired noisy failure cannot be attributed solely to noise. Single-seed validation diagnostics only.

Historical p90/p99 runs used other GPU/runtime contracts and are contextual comparisons, not a controlled beta-only effect. No further noise level is scheduled.

![Accuracy and loss trajectories](conv3_p95_read_noise_1em3_20260921_epochs.png)
