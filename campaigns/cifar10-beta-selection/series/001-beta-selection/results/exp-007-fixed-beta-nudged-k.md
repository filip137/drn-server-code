---
experiment: "exp-007"
evidence: "validated-local"
summary: "Increasing trained block2 nudgedK6→48 at fixed beta has negligible effect in all schemes; maximum layer cosine change is8.91e-7."
verdicts: {"H-005": "contradicts"}
---
# Fixed-beta nudged-K result

All three epoch50 checkpoint replays completed on5090s. One fixed physical beta
per scheme, block2 only, nudgedK=[6,12,24,48]; same256 images, freeT=6, fixed native/
local BPTT K=6, saved initial states and output forces. No recalibration, noise,
training or parameter updates. Adaptive stopping is disabled in these solver
configs: higher requestedK executes the extra sweeps.

Minimum pooled cosine across block2's three Conv weights:

| Scheme | Fixed B | K6 | K12 | K24 | K48 |
|---|---:|---:|---:|---:|---:|
| Baseline | 1.0249903844 | 0.972750 | 0.972750 | 0.972750 | 0.972750 |
| Ours | 2.1340009061 | 0.949797 | 0.949797 | 0.949797 | 0.949797 |
| Legacy | 0.6540032061 | 0.979358 | 0.979358 | 0.979358 | 0.979358 |

Across all three layers/schemes, maximum cosine change fromK6 is8.91e-7.
The worst-layer pooled score changes by at most5.84e-9. K12/24/48 metrics are
identical within the recorded precision. Relative outputRMS remains baseline
0.0199282, ours0.0348773, legacy0.00248283; maximum relative change is1.62e-10.
Worst minibatch/layer cosine is also unchanged to practical precision:
baseline0.818962, ours0.900176, legacy0.761540. More iterations do not improve
the previously observed minibatch variability.

Review: contradicts H-005. The proposed>=0.01 gain for ours is absent; this is
far below the predeclared0.001 unchanged threshold. Insufficient nudged iteration
count does not explain the remaining gradient gap at these checkpoints/betas.
This does not test different freeT, arithmetic precision or other beta values.
Finite-beta nonlinearity and numerical precision at small perturbations remain
possible explanations; neither is established by this negative result.

All12 pooled/96 batch rows per scheme reconcile. Local bundles validate; K6
reproduces exp-006 cosine/norm/RMS at rel_tol1e-5/abs_tol1e-7, references are
unchanged acrossK, and checkpoint hashes/model/BN tensors are unchanged.
Six focused numerical tests passed. No failures or exclusions. Queue collection
complete and GPUs released; charged times75.4s ours,85.2s legacy,31.1s baseline,
191.7s combined within1800s budget.

Intermediate epochs: final epoch10 and30 checkpoints exist for all three exact
trajectories. Actual loaded epochs and hashes match their continuation records.
Their cosine/RMS curves have not been measured. Paths/hashes are preserved in
`results/cifar-block2-nudged-k-e50-20260929-v1/analysis/intermediate_checkpoints.json`.

[K comparison plot](../../../../../results/cifar-block2-nudged-k-e50-20260929-v1/analysis/fixed_beta_nudged_k.jpg).
Evidence root: `results/cifar-block2-nudged-k-e50-20260929-v1/`, with each scheme's
validated `run/` and `analysis/`; combined points are in `analysis/comparison.json`.
