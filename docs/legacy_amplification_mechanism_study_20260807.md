# Why Legacy Amplification Trains Better

- Analyzed: 2026-08-07
- Status: complete, validated post-hoc mechanism study
- Study ID: `perfectdiode-signed-scaled-bias-legacy-mechanism-seed0-20260807-v1`

## Executive Summary

Legacy amplification does **not** train better because it reaches flatter
minima in trainable-parameter space. Its parameter-loss Hessian is substantially
sharper than baseline or ours, including on the free face after exactly pinned
bounded weights are removed. The sharpness difference is already present at
initialization.

The combined evidence instead supports a depth-dependent structural
regularization and preconditioning mechanism. Legacy strongly attenuates deep
cross-coupling and postsynaptic loading, lowers the raw stiffness of the
physical equilibrium problem, suppresses runaway conductance and bias growth,
and leaves many more conductances away from projection boundaries. At the same
time, its hidden voltages occupy sparse, stable perfect-diode faces. The most
consistent signature is therefore **free parameters with sparse, stable hidden
states**. This mechanism is strongest in deeper and bounded networks, where
legacy's accuracy advantage is also largest.

## Evidence Base

The analysis covered all 72 validated seed-0 runs in the signed, unclamped,
amplification-scaled bias campaign:

- ordinary MNIST and deterministic medium-affine MNIST;
- bounded and wide-range conductance contracts;
- Conv1, Conv2, and Conv3;
- baseline, ours, and legacy amplification; and
- SGD and Adam.

Initialization was reconstructed deterministically. Best-validation and final
checkpoints were hash-validated against their saved PT and NPZ artifacts. The
analysis was read-only: it applied no optimizer steps, changed no source
checkpoint, and made no additional official-test read. Ordinary-MNIST results
are optimization diagnostics; deterministic medium-affine results are the
paper-facing outcomes.

Mean recorded accuracy across the six architecture-by-optimizer cells in each
dataset and weight contract was:

| Dataset | Weights | Baseline | Ours | Legacy | Legacy - baseline | Legacy - ours |
|---|---|---:|---:|---:|---:|---:|
| Ordinary MNIST | bounded | 73.84% | 82.95% | **94.92%** | +21.08 points | +11.97 points |
| Ordinary MNIST | wide | 96.98% | 97.78% | **97.79%** | +0.80 points | +0.01 points |
| Medium affine | bounded | 41.21% | 53.17% | **72.26%** | +31.05 points | +19.09 points |
| Medium affine | wide | 80.99% | 87.52% | **88.45%** | +7.46 points | +0.94 points |

Legacy beats both alternatives in all 12 bounded cells. Across all 24 matched
cells, it beats baseline in 22 and ours in 21; the exceptions are shallow
wide-range SGD cases. Its mean advantage grows with depth: Conv1, Conv2, and
Conv3 improve by `7.17/13.48/24.65` points over baseline and
`3.53/5.86/14.61` points over ours.

## Broader Conclusion: Feed-Forward Dominance Helps Training

Within the amplification family studied here, the evidence is now sufficient
for a descriptive conclusion: **the more feed-forward-dominant the effective
network is, the easier it is to train and, in general, the higher its final
accuracy**. The ordering becomes clearer with depth and is dramatically
stronger when conductances are bounded. This is not yet a causal explanation.
In particular, the Hessian results below rule out the simplest claim that the
more feed-forward-dominant schemes merely find flatter parameter minima.

“Feed-forward-dominant” is deliberately narrower than “feed-forward.” These
models still derive both directions from a reciprocal energy. Amplification
changes the relative effective influence of downstream and return interactions
after normalization; it does not introduce independently trainable forward and
backward matrices. A useful operational quantity for the next study is therefore
the local round-trip gain or state-Jacobian block transfer around the trained
equilibrium, alongside the known analytical interaction coefficients.

### Connection to Liu and Chen, ICLR 2026

Liu and Chen's ICLR 2026 paper,
[*Toward Practical Equilibrium Propagation*](https://arxiv.org/abs/2508.11659),
reports a closely related empirical pattern. Their feedback-regulated RNNs
independently scale feedforward and feedback connections. Reducing feedback
lowers the recurrent spectral radius and convergence time, and their MNIST
experiments show that larger feedback scaling generally lowers accuracy.
However, the effect is not monotone without qualification: feedback that is
too weak causes vanishing gradients in deep networks, and residual connections
are needed to restore deep credit assignment.

The settling-speed result is not the important connection for the present
study: accepted `T/K` operating points already make equilibrium convergence a
controlled condition. The relevant parallel is the accuracy effect. Both
studies suggest that learning improves when the recurrent return path is
weakened relative to forward signal transmission, provided useful deep learning
signal is preserved.

The comparison is suggestive rather than an equivalence. Liu and Chen use
separately scaled forward and feedback matrices in general vector-field
dynamics and an EP-style local update. The present networks use one reciprocal
energy and BPTT. Their work therefore motivates the feed-forward-dominance
hypothesis but does not explain the mechanism in this repository.

## 1. Parameter Sharpness Does Not Explain the Advantage

The study measured exact BPTT Hessian-vector products on four fixed batch-16
training cohorts at initialization, best, and final checkpoints. Spectra used
20-step fully reorthogonalized Lanczos and eight Hutchinson probes. Curvature
was evaluated in raw parameters, physical interaction coefficients, nominal
learning-rate coordinates, and the free face of the bounded parameter box.

At the best checkpoint, geometric-mean top-eigenvalue ratios over all 24
matched cells were:

| Reference | Raw parameter coordinates | Interaction coordinates | Nominal-LR coordinates |
|---|---:|---:|---:|
| Legacy / baseline | **41.9x** | **4.85e5x** | 1.20x |
| Legacy / ours | **11.3x** | **1.66e3x** | 1.59x |

Legacy is sharper in all 24 raw and all 24 interaction-coordinate comparisons
against either reference. In bounded networks, removing exactly pinned weights
still leaves legacy `40.7x` sharper than baseline and `6.10x` sharper than ours.
At initialization, the raw ratios are already `143x` and `29.4x`, respectively,
showing that the difference is primarily structural rather than learned.

Learning-rate scaling partially explains why SGD remains stable. For SGD,
legacy's nominal-LR curvature is `0.270x` baseline and `0.870x` ours despite
much larger raw curvature. It does not close the mechanism question: legacy
and ours can have similar LR-scaled curvature but very different accuracy.
For Adam, legacy remains `5.30x` sharper than baseline and `2.91x` sharper than
ours in nominal-LR coordinates while winning every cell. Saved Adam moments do
not exist, so the true Adam-preconditioned Hessian cannot be reconstructed.

The leading raw-Hessian directions are carried by weights, not biases. Biases
account for only about `0.024%` of legacy's leading-eigenvector mass on average.

## 2. Physical-State Curvature Shows the Opposite Effect

The Hessian with respect to equilibrium voltages is a different object from
the loss Hessian with respect to trainable parameters. On the feasible
perfect-diode face, legacy has lower raw physical-energy top curvature in every
matched cell:

- `0.0188x`--`0.0259x` the baseline value, or about 40--53 times less stiff;
- `0.0340x`--`0.131x` the ours value, or about 8--29 times less stiff.

Jacobi normalization mostly removes this absolute-scale difference: qualified
top-eigenvalue ratios become `0.892x`--`1.008x`. Legacy's normalized condition
number is nevertheless modestly lower in four of six cells within each
dataset/contract group. Legacy therefore makes physical inference less stiff;
it does not make the trained parameter loss flatter.

## 3. Learned Magnitudes Show Attenuation, Not Compensation

For source-edge index `i`, let `r = current_amp / voltage_amp`. The energy
prefactor is `r^i`; the cross coefficient is `r^i * p_i * current_amp`, where
`p_i` is one at the input and `voltage_amp` thereafter. Legacy uses
`voltage_amp=4`, `current_amp=0.25`, and `r=1/16`; ours uses `4`, `1`, and
`1/4`; baseline uses ones. Deep legacy interactions are therefore attenuated
much more rapidly.

Training does not undo that attenuation:

- At the best checkpoint, legacy's expanded cross-coupling RMS is below both
  references in every one of the 72 matched weight-tensor comparisons.
- In bounded networks, raw weight RMS is comparable and sometimes slightly
  larger, but whole-network effective cross-coupling is typically only 4--5%
  of baseline and 8% of ours.
- In the deepest bounded Conv3 dense interaction, the corresponding median
  ratios fall to `1.86e-4` and `3.44e-3`.
- In wide-range networks, legacy's raw best-checkpoint weight RMS is only about
  22--29% of the references before amplification coefficients are applied.

Wide-range raw-weight growth from initialization to best is:

| Optimizer | Baseline | Ours | Legacy |
|---|---:|---:|---:|
| Adam | 20.35x | 13.52x | **3.45x** |
| SGD | 13.31x | 5.36x | **1.13x** |

Learned signed biases are also usually smaller under legacy, including Bias 0,
which has no explicit depth scaling. Deeper bias coefficients then contribute
additional factors of 16 and 256 attenuation. This indicates that both the
formula and the resulting optimization dynamics suppress parameter growth.

## 4. Bounded Legacy Weights Are Much Less Pinned

No weights start exactly at a bound. At the best checkpoint, legacy has less
exact endpoint mass than both references in all 12 bounded cells. Count-weighted
occupancy is:

| Dataset / optimizer | Baseline | Ours | Legacy |
|---|---:|---:|---:|
| Medium affine / Adam | 26.86% | 37.08% | **20.37%** |
| Medium affine / SGD | 27.08% | 13.91% | **5.04%** |
| Ordinary MNIST / Adam | 50.18% | 49.76% | **18.90%** |
| Ordinary MNIST / SGD | 34.13% | 16.82% | **11.33%** |

Lower-bound projection dominates. At the final checkpoint, legacy remains
below baseline in all 12 cells and below ours in 11 of 12. The separation is
largest in deep convolutional tensors: Conv2 `ConvWeight_1` averages 15.2%
endpoint mass under legacy versus 39.5% for baseline and 39.4% for ours;
Conv3 `ConvWeight_2` averages 20.6% versus 69.8% and 48.1%.

This is a strong and consistent marker of useful optimizer freedom, but not a
complete causal explanation. The size of the endpoint-mass reduction does not
strongly track the accuracy gain within matched surfaces.

## 5. Wide Weights and Hidden Diode Faces

The wide-range analogue gives the same parameter-side pattern. About 49.9% of
Kaiming weights begin at zero after nonnegative projection. At best, legacy has
less zero mass, activates more initially zero weights, and kills fewer initially
positive weights than both alternatives in all 12 cells.

Parameter sparsity and state sparsity move in opposite directions. Although
legacy frees more conductances, it usually places **more hidden voltages exactly
on the perfect-diode boundary**. Boundary membership changes negligibly between
detached inference and differentiable replay, showing that the Hessian is being
measured on a stable local diode face.

## Mechanism Supported by the Measurements

The evidence is most consistent with the following chain:

1. Legacy amplification attenuates deep cross and postsynaptic terms much more
   strongly than presynaptic diagonal loading.
2. The equilibrium problem becomes much less stiff in raw voltage units and
   remains at least as well conditioned after diagonal normalization.
3. Conductances and biases do not need to grow enough to cancel the fixed
   attenuation; wide-range runaway growth is strongly suppressed.
4. Far fewer bounded parameters become pinned, preserving trainable degrees of
   freedom, especially in the deepest layers.
5. Hidden states settle onto sparse, stable diode faces, yielding a physically
   stable operating regime while the parameters remain comparatively free.
6. Surface-specific learning rates compensate much of the raw parameter
   sharpness for SGD; Adam must additionally benefit from its adaptive
   preconditioner, although the missing moment states prevent direct testing.

This supports a depth-dependent structural regularization/preconditioning
interpretation. It rules out three simple explanations: legacy does not win by
finding flatter parameter minima, by learning larger effective couplings, or
solely because of its bias tensors.

## Next Study: Scaling Across Depth and Conductance Bounds

The current comparison already suggests both relevant scaling effects. The
legacy advantage grows from Conv1 to Conv3, and it grows sharply when moving
from the wide contract to the current hardware bound. Legacy's mean advantage
over ours is `11.97` points under bounded weights versus `0.01` points in the
wide ordinary-MNIST condition, and `19.09` versus `0.94` points on
medium-affine MNIST. Three depths and two weight contracts establish the
direction of the effect, but not a depth-by-bound scaling law.

The next experiment should cross Conv1/Conv2/Conv3 with several bound
severities while retaining matched model seeds, minibatch order, data cohorts,
and initialization quantiles. If both bounded initializer families remain
scientifically relevant, they should be treated as separate strata rather than
pooled. The primary estimand should be the architecture-depth by conductance-range
by amplification-scheme interaction:

- accuracy gaps `legacy - ours`, `legacy - baseline`, and `ours - baseline`;
- the fraction and layer location of exact-bound weights;
- projected-update efficiency and the fraction of gradients removed by the
  bounds;
- raw and effective cross-coupling magnitude;
- layerwise gradient and update transmission, especially in the deepest
  convolutional tensors;
- physical-state curvature and an equilibrium round-trip-gain diagnostic; and
- hidden diode-face occupancy and stability.

The deployment question and the causal mechanism question should be reported
separately. A protocol-optimized sweep should reselect rho and learning rates at
each bound to measure the best achievable accuracy. A smaller update-norm-matched
or fixed-rate panel should test whether the amplification-by-bound interaction
persists without optimizer-scale compensation.

The predicted signature is not simply that tighter bounds hurt baseline more.
It is that, as compensation by raw conductance magnitude becomes less available,
the advantage of feed-forward dominance grows together with lower projection
pinning and preserved deep-layer update signal. Failure of those quantities to
co-vary would reject the current mechanism hypothesis even if the accuracy
ordering remains.

## Limits and Decisive Follow-Up

The result is a mechanism hypothesis supported by matched descriptive evidence,
not a causal decomposition. Important limitations are:

- seed 0 only;
- surface-specific rho and parameter-learning-rate vectors;
- no intermediate checkpoints or saved Adam moments;
- different best epochs;
- parameter Hessians of the local piecewise-smooth truncated-BPTT surrogate;
- fixed Hessian cohorts covering 64 training examples; and
- Conv3 baseline using `T=12,K=8` while ours and legacy use `T=K=8`.

The decisive next experiment is a fixed-rho/fixed-LR or update-norm-matched
ablation that varies `current_amp` and bias scaling independently, ideally while
also matching effective cross coefficients.

## Provenance and Numerical Validation

The canonical bundle contains 3,456 parameter-Hessian rows, 17,280 block rows,
and 864 physical-Hessian rows. The worst top-Ritz residual relative to the
absolute eigenvalue is 7.68%, the worst finite-difference HVP canary error is
6.09%, and the worst symmetry relative error is `1.11e-3`, below the justified
`2e-3` float32 threshold. Fifty-four focused tests and the canonical reporting
validator pass.

The canonical analysis was produced on branch `codex/signed-scaled-bias` at
commit `41d1bf9a0aa6d7cd70e543135123c27667d93b81`; its curated record was finalized
at `154d80e8`. The result directory is
`results/perfectdiode-signed-scaled-bias-legacy-mechanism-seed0-20260807-v1`
within that worktree.
