---
experiment: "exp-006"
evidence: "validated-local"
summary: "Epoch50 changes the useful RMS scale. Legacy block2 improves to0.979 atR≈0.0025; ours remains limited near0.950. Initialization target0.075 fails to transfer."
verdicts: {"H-004": "contradicts"}
---
# Epoch50 BPTT checkpoint replay

Completed original seed0 final-epoch50 baseline, ours and native unnormalized
legacy checkpoints. Same256 training images/order, batch32, frozen BN buffers,
no augmentation/noise, and T=K=[6,6,4] as initialization. Original initializer
tensors match the earlier sweep bitwise. Parent paths/hashes are in the
[experiment handoff](../experiments/exp-006-trained-e50-rms.md) and each bundle's
`inputs/checkpoint_provenance.json`; no checkpoint or training state was modified.

All12 main targets plus3 adaptive low targets completed for every scheme:
45 calibrated block targets,120 pooled/960 minibatch Conv rows per scheme.
All six queue jobs/collections complete; local bundles, coverage, epoch50,
checkpoint hashes, unchanged tensors and calibration/replay RMS agreement pass.
All cosines are defined and all calibrated targets are within3%. No failures or
exclusions. Combined charged GPU time997.8s within3600s; GPUs released.

Best sampled **RMS (minimum pooled Conv cosine)** per block:

| Block | Baseline | Ours | Legacy |
|---|---:|---:|---:|
| 1:3 conv | 0.0199 (0.9796) | 0.0200 (0.9729) | 0.00973 (0.9756) |
| 2:3 conv | 0.0199 (0.9727) | 0.0349 (0.9498) | 0.00248 (0.9794) |
| 3:2 conv | 0.00982 (0.9904) | 0.00491 (0.9883) | 0.00977 (0.9842) |

All block-level sampled peaks are interior after the lower-target extension;
this finite grid does not prove exact optima. In particular ours0.9498 is just
below0.95, not evidence that no intermediate target could cross the threshold.

Legacy block2's peak rises from0.9730 at initialization to0.9794 after training;
its preferredR falls from0.0647 to0.00248. It is now better aligned than legacy
block1. Ours block2 drops from0.9684 to0.9498 and baseline block2 from0.9932 to0.9727.
Thus H-004's prediction of a persistent weakest block2 in both ours and legacy is
contradicted. The predicted selective deficit also fails because other blocks
generally lose alignment after training. A universal depth explanation is unsupported.

At the initialization candidate **R≈0.075**, minimum pooled cosine:

| Scheme | Block1 | Block2 | Block3 |
|---|---:|---:|---:|
| Baseline | 0.1982 | 0.8086 | 0.7537 |
| Ours | 0.8569 | 0.8625 | 0.7922 |
| Legacy | 0.3723 | 0.3811 | 0.7165 |

No tested commonR gives>=0.95 across all schemes/blocks; baseline alone has a
pooled passing overlap around0.010–0.019. No scheme has an all-block/all-minibatch
>=0.95 overlap. Per-batch alignment can be much worse than the pooled score:
at each block's pooled-optimalR, worst minibatch/layer values are baseline
[0.913,0.819,0.942], ours[0.882,0.900,0.960], legacy[0.668,0.762,0.955].
Do not interpret a pooled maximum as uniformly reliable per-step gradients.

Why block2 can look difficult: in trained ours its three Conv layers prefer very
different displacements. Their individual sampled peaks occur nearR0.0841,0.0200,
and0.000499, with cosine0.9581,0.9858,0.9995 respectively. The last layer's individual
peak remains at the low search edge; the block-wide peak is bracketed. At the block
compromiseR≈0.035 the first/last layer cosines are0.9552/0.9498. Matching outputRMS
does not ensure equally suitable perturbations throughout a block.

Native and matched-local BPTT reference cosine is>=0.999999926 for every pooled
Conv tensor, so their mismatch cannot explain these substantial EqProp errors.
A plausible mechanism is limited propagation of the nudged response at finiteK,
combined with layer-specific nonlinear response. This remains an inference:
neither internal displacement nor a changed-K control was measured here. Native
gradient norms also fall strongly after training (retained in `gradient_scales.csv`);
that scale change alone does not establish why cosine falls.

Decision: R≈0.075 is an initialization result, not a trained-network rule. Retain
block/checkpoint-specific candidates rather than assuming a depth-based universal
target. A causal follow-up would change only nudgedK in trained block2, recalibrating
to the same measuredR; it was not launched. No EqProp training accuracy is inferred.

[Epoch0/50 curves](../../../../../results/cifar-block-rms-e50-20260929-v1/analysis/e50_rms_cosine.jpg)
and [trained block2 layer curves](../../../../../results/cifar-block-rms-e50-20260929-v1/analysis/e50_block2_layers.jpg).
Validated evidence roots: `results/cifar-block-rms-e50-20260929-v1/` and
`results/cifar-block-rms-e50-low-20260929-v1/`; combined analysis/peak betas,
reference comparisons and gradient scales are under the first root's `analysis/`.
