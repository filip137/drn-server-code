# CIFAR L8: larger legacy gradients versus effective Adam updates

Existing read-only checkpoint replays, reanalyzed September23. No new GPU
replay, training, optimizer update or checkpoint write was performed. The
question is whether legacy's larger raw gradients imply more aggressive or
less stable optimization.

## What is measured

All three original trainable-BN schemes have real checkpoints at epochs
0/10/30/50. The replay uses the same128 training examples, four batches32,
no augmentation, training-mode BN, cross-entropy and operational T/K[6,6,4]
on Fifi RTX5090. Cohort SHA256 is
`82962c61d8dff52bdc730ec1b5bc1ae0222d792d663da455b6fa8babf5778b00`.
The original replay verified unchanged checkpoint bytes/parameters and
restored BN buffers; all12 native convergence/gradient audits passed.

For every checkpoint and batch, the saved data include the hypothetical next
Adam update using that checkpoint's own moments, scheduled LR and epsilon,
then conductance-bound projection and float32 rounding. Relative update means
norm(delta_G)/norm(G). Epoch50 proposals are hypothetical next steps at the
schedule floor, not updates logged during a subsequent training epoch.

The following ratios pair identical batches, take each layer's median across
four batches, then the median across the eight convolution layers:

| Saved epoch | Legacy/proposed gradient RMS | Legacy/proposed relative update | Legacy/baseline gradient RMS | Legacy/baseline relative update |
|---|---:|---:|---:|---:|
| 0 | 1.00 | 0.115 | 1.16 | 0.336 |
| 10 | 6.34 | 0.903 | 1.77 | 0.954 |
| 30 | 6.51 | 1.111 | 2.25 | 1.270 |
| 50 | 8.63 | 0.664 | 1.97 | 0.923 |

At50, the eight layer-median legacy/proposed gradient ratios span4.58–10.58,
while relative-update ratios span0.479–0.987. At30, relative updates do vary
more: legacy/proposed layer medians span0.566–1.655. Thus the endpoint does not
describe the entire optimization history. The per-batch spreads are retained;
these four batches and eight layers are not independent training replicates.

Example: first convolution at50, medians over the four matched batches:

| Scheme | Gradient RMS | Conductance L2 norm | Relative projected next Adam update |
|---|---:|---:|---:|
| Baseline | 1.6353e-4 | 214.774 | 2.0649e-5 |
| Proposed | 4.2647e-5 | 518.444 | 4.2410e-5 |
| Legacy | 2.5480e-4 | 74.576 | 2.0383e-5 |

The large-gradient statement is specific to the convolutions: at50 the
classifier's gradient RMS is larger in proposed than legacy. Legacy's
convolution projection retains99.756–99.999% of the unprojected proposal norm
(range across layer medians), so there is little late convolution-update norm
attenuation from clipping/rounding in these probes. This does not measure a
clipped-coordinate fraction or rule out earlier boundary effects.

## Interpretation

Adam uses delta_G = -lr*m_hat/(sqrt(v_hat)+eps_Adam), with eps_Adam=1e-8 here,
distinct from BN epsilon. Multiplying an entire coordinate's gradient history
by a constant positive factor multiplies m by that factor and v by its square;
the factors approximately cancel when Adam epsilon is negligible. Different
histories, directions, relative LRs, curvature and time-varying scales still
matter. A large current gradient alone does not determine the step.

Parameter scale is another material distinction. Legacy's trained convolution
norms at50 are about6–10 times smaller than proposed's. For these zero-bias
resistive blocks, uniformly scaling every conductance in a block by c leaves
the ideal equilibrium function unchanged, while its conductance gradients
scale as1/c, provided the rescaled parameters remain admissible. Actual saved
models are not asserted to be uniform rescalings of each other. The identity
explains why raw gradient size is not itself a coordinate-independent measure
of loss sensitivity. Empirically, multiplying each layer's gradient RMS by
its parameter norm reduces the median legacy/proposed ratio at50 from8.63 to
1.06. This norm product is a scale diagnostic, not a measured loss change or
the gradient with respect to an individual log-conductance.

Consequently the data do not support a simple account in which large legacy
gradients cause uniformly oversized Adam steps. They do not establish that
gradient differences are harmless: layer balance, gradient noise, momentum
alignment and functional sensitivity remain open. Distinct trained weights,
BN states and selected optimizer rates confound attribution to amplification
alone. The near-unit legacy/proposed gradient ratio at the shared initializer
also shows that the large trained ratio is not simply a fixed amplification
multiplier present from the beginning.

## More informative next diagnostic

Before prescribing a smaller LR or gradient clipping, measure loss and logits
along the actual projected Adam-update direction on disposable copies of
epoch10/30/50 checkpoints, starting with30 where relative steps differ most.
Use identical gradient batches, saved moments and frozen evaluation BN buffers;
compare step multipliers0/0.5/1/2 on the same and a separate fixed training
probe cohort. Record actual logit displacement, CE change, gradient/update
alignment and bound projection. Restore full model/BN/optimizer state between
every probe, and do not change the original checkpoints or reported validation
scores. The update direction should include the same parameter groups as
training; conv-layer summaries alone do not capture the full-network move.

If legacy's nominal step repeatedly overshoots while half steps improve both
probes, that supports an effective-step-size problem. If nominal steps behave
well despite large raw gradients, raw scaling is a weaker explanation. These
are conditional interpretations, not results. The diagnostic is proposed,
not launched; it can precede gain training in the proposed diagnostic budget.

![Gradients versus relative Adam updates](../results/cifar10-l8-amplification-mechanism-seed0-20260923-v1/analysis/gradient_vs_adam_updates.png)

[Per-layer medians and batch ranges](../results/cifar10-l8-amplification-mechanism-seed0-20260923-v1/analysis/gradient_update_layers.csv) ·
[Paired ratios](../results/cifar10-l8-amplification-mechanism-seed0-20260923-v1/analysis/gradient_update_paired_ratios.csv) ·
[Checkpoint paths/hashes, settings and aggregate values](../results/cifar10-l8-amplification-mechanism-seed0-20260923-v1/analysis/gradient_update_summary.json) ·
[Reproducible aggregation](../experiments/summarize_cifar_l8_gradient_updates.py).
