---
experiment: "exp-022"
evidence: "validated-local"
summary: "Larger beta largely restores trained gradients; initialization instead needs smaller beta. All eight pooled layer gradients meet the descriptive criterion at each checkpoint with separate block betas."
verdicts: {"H-017": "supports"}
---
# Small trained responses explain a recoverable part of the gradient error

Read-only replay of the same ours fixed-0.3× trajectory at epochs 0, 1 and 5,
256 identical unaugmented training images in eight batches, 17 beta multipliers,
eight convolution weights. All three bundles validated locally: 408 pooled rows
and 3,264 batch rows, unchanged checkpoints/model/BN, consistent state/force
pooling, complete coverage and invariant reference statistics across beta.
No training, smoke, optimizer update, noise or official-test access occurred.
Riri RTX 5090, torch 2.11.0+cu128 / CUDA 12.8, native float32 T/K [6,6,4].
Scientific compute took 748 seconds in total; charged queue execution including
validation was about 754 seconds, within the 5,400-second allowance.

Artifacts: `results/cifar-ours-epoch-beta-gradients-20260930-v1/`.
Exact inputs/source identity and configs are recorded in the
[experiment](../experiments/exp-022-epoch-layer-response.md), `analysis/input_provenance.json`,
and each bundle manifest. `analysis/e{0,1,5}-collection.json` records validation.
`analysis/summary.json`, `layers.csv` and `batches.csv` retain full measurements.
All checks and plotting completed with `analysis/analyze.py`.

## Measured recovery

Each entry is the **worst pooled native-BPTT cosine among a block's layers**.
Select one common beta per block by maximizing that minimum over the declared
grid; selections use these images and are not independent validation.
Multipliers below are relative to the actual trajectory beta vector
[0.0283569660, 1.1189959117, 0.3508559012], not the earlier source anchor.

| Epoch | Block 1: original → selected | Block 2: original → selected | Block 3: original → selected | Selected multipliers, blocks 1/2/3 |
|---|---|---|---|---|
| 0 | 0.583 → 0.9990 | −0.090 → 0.9660 | 0.676 → 0.9994 | 0.03 / 0.001 / 0.01 |
| 1 | 0.992 → 0.9998 | 0.705 → 0.9995 | 0.765 → 0.9878 | 10 / 100 / 10 |
| 5 | 0.999 → 0.9995 | 0.919 → 0.9999 | 0.787 → 0.9987 | 3 / 300 / 30 |

All selected pooled gradients meet the proposed cosine >=0.95 and norm ratio
0.9–1.1 criterion. At epochs 1 and 5, every selected norm is within 0.9% of
native BPTT; epoch-0 block2 conv1 has ratio 0.908. No selected optimum is at a
grid boundary. Block1 was already well aligned at the trained checkpoints;
the substantial recoveries are blocks2/3.

The minibatch result is weaker at epoch1 block3: its worst-layer cosine ranges
0.925–0.981, median 0.948; only 4/8 batches meet both thresholds in every layer.
At epoch5 all eight batches meet both thresholds in all blocks; the weakest
selected cosine is 0.993. At initialization 7/8 block2 batches meet both
thresholds, although all selected minibatch cosines exceed 0.975.

## RMS explains the direction, but does not provide a universal target

At epoch5, block3 conv1's relative internal RMS increases from 1.54e-8 to
2.68e-7 and its cosine from 0.787 to 0.9987. Its unchanged endpoint fraction
drops from 84.5% to 56.0%. At epoch1, block2 conv1's internal RMS increases
from 1.27e-8 to 7.83e-7, while block2 conv2's cosine rises from 0.705 to 0.9995.
This intervention at fixed weights and images supports insufficient response
as a cause of much of the measured gradient error. Unchanged endpoints also
include inactive/clamped units; these counts do not isolate rounding error.

The same beta has opposite problems at different stages: excessive nudging at
initialization, insufficient response later. RMS is not monotonically falling:
for example, fixed-beta block3 output RMS is 1.148 at epoch0, 2.13e-6 at epoch1,
and 7.53e-6 at epoch5. Internal layers can respond orders of magnitude less than
their block output. Conversely, relative internal RMS around 3–4e-7 already
gives high alignment in the initialization's first convolutions.

The drift includes changing force and voltage scale. From epoch0 to epoch1,
block2's output-force RMS falls 0.710 → 6.76e-5 and its free-output RMS grows
0.00514 → 3.81. Block3's force falls only about 5× while its absolute endpoint
response falls about 3,367× at fixed beta. Relative RMS alone therefore cannot
separate loss forcing, free-voltage scale and the network's finite-beta response.

| Epoch | Block1 output relative RMS at selected beta | Block2 | Block3 |
|---|---|---|---|
| 0 | 0.01450 | 0.09898 | 0.01586 |
| 1 | 0.000442 | 0.001071 | 0.00002172 |
| 5 | 0.002776 | 0.008852 | 0.0002274 |

These are grid-selected descriptive optima, not universal RMS prescriptions.
At epoch1 block3, increasing the multiplier beyond 10 to 30 raises output RMS
to 0.000371 but reduces the worst cosine to about 0.300. Both insufficient
and excessive response matter. The recorded native/local BPTT cosines are
all above 0.997 at initialization and above 0.9997 after training; the large
active-beta EqProp errors therefore occur against both references.

Plots (each convolution is shown separately):

- [Cosine versus internal relative RMS](../../../../../results/cifar-ours-epoch-beta-gradients-20260930-v1/analysis/cosine_vs_state_relative_rms.jpg)
- [Cosine versus beta](../../../../../results/cifar-ours-epoch-beta-gradients-20260930-v1/analysis/cosine_vs_beta_multiplier.jpg)
- [Cosine versus absolute RMS](../../../../../results/cifar-ours-epoch-beta-gradients-20260930-v1/analysis/cosine_vs_state_absolute_rms.jpg)
- [Norm ratios and unchanged endpoints](../../../../../results/cifar-ours-epoch-beta-gradients-20260930-v1/analysis/norm_ratios_and_unchanged_fraction.jpg)
- [Gradient magnitudes relative to epoch0](../../../../../results/cifar-ours-epoch-beta-gradients-20260930-v1/analysis/gradient_scale_vs_epoch0.jpg)

## Interpretation and decision

H-017 is supported for **gradient quality on the tested ours trajectory**:
beta adjustment largely recovers the trained gradients, without changing T/K
or the estimator. This identifies a plausible contributor to poor learning,
but does not establish how much of the accuracy deficit it caused. Recovering
gradients at saved checkpoints does not undo earlier updates or demonstrate
that a practical adaptive controller will follow the useful range.

Retain the stop on new training. A next discriminating diagnostic would resolve
the narrow epoch1 block3 beta interval and test selection on separate images;
a future learning intervention should then check the first updates as well as
epoch boundaries. No follow-up is launched or scheduled by this experiment.

## Internal RMS bounds for early convolutions

Filip requested the small-displacement limits for Conv1/Conv2 in blocks1/2.
Postprocess the existing replay with
`python experiments/plot_cifar_internal_rms_bounds.py`; no additional simulation.
This uses the **EqProp trajectory at epochs0/1/5**, distinct from exp-008's
original BPTT checkpoints at epochs0/10/30/50.

At pooled native cosine0.95, the adjacent lower failing and first passing
**relative internal RMS** samples are:

| Layer | Epoch0: fail → pass | Epoch1: fail → pass | Epoch5: fail → pass |
|---|---|---|---|
| Block1 Conv1 | 1.76e-8 → 3.67e-8 | 8.25e-9 → 1.59e-8 | 5.49e-9 → 1.04e-8 |
| Block1 Conv2 | 4.20e-6 → 1.26e-5 | 1.94e-7 → 5.78e-7 | 4.53e-7 → 1.51e-6 |
| Block2 Conv1 | 1.30e-7 → 3.89e-7 | 1.27e-8 → 2.72e-8 | 2.16e-8 → 5.25e-8 |
| Block2 Conv2 | 4.78e-5 → 1.43e-4 | 7.28e-7 → 2.42e-6 | 9.96e-7 → 2.99e-6 |

R is RMS((s_plus-s_minus)/2)/RMS(s_free) for that convolution's postsynaptic
state. These are coarse sampled transitions, not interpolated exact cutoffs,
certified continuous intervals or independent validation. First passing points
also have pooled norm ratios within0.9–1.1, but some individual minibatch cosines
remain as low as0.80; pooled bounds do not certify every update.

The early-layer/later-layer interpretation holds for the trained common-beta
bands in these blocks: at the adjacent lower failing sample, Conv1 and/or Conv2
fails; at the adjacent upper failing sample, only Conv3 fails. It is not a rule
that a layer has only one bound. Expanding beta exposes low-displacement failure
in the last layers too, and upper limits in early layers. Initialization block2
is an exception: its early layers constrain both ends of the common passing band.

Block2 Conv1's first passing **absolute** RMS is1.10e-7,1.19e-7,1.29e-7 at
epochs0/1/5, despite the relative crossing shifting substantially. Its free-state
RMS changes, so relative RMS alone can obscure this near agreement. Block2 Conv2
instead has first passing absolute RMS6.16e-7,4.23e-6,2.28e-6. Neither a universal
relative floor nor a universal absolute floor across layers is established.

[Relative internal RMS bounds (JPG)](../../../../../results/cifar-ours-epoch-beta-gradients-20260930-v1/analysis/early_layer_relative_rms_bounds.jpg)
and [absolute internal RMS bounds (JPG)](../../../../../results/cifar-ours-epoch-beta-gradients-20260930-v1/analysis/early_layer_absolute_rms_bounds.jpg)
show the four early layers with the measured lower transitions highlighted.
`analysis/internal_rms_bounds.json` and `.csv` retain all24 layer/checkpoint
passing intervals, adjacent lower/upper failures, unbracketed edges, beta,
norm ratio and (JSON) batch cosine spread. These are derived from the already
validated408 pooled records; all sampled state RMS values are monotone in beta.

## Free-state voltage magnitudes across training

Filip requested voltage magnitudes in every layer and block. Reuse the saved
free-state squared sums with `python experiments/plot_cifar_free_voltage_evolution.py`.
This answers magnitude as RMS; signed means, standard deviations and extrema
were not recorded and cannot be reconstructed from RMS. No new replay is needed
for this magnitude comparison. Preserve the same256-image cohort, batch32,
float32, freeT=[6,6,4] and minibatch-BN convention. Only epochs0/1/5 are measured;
no per-epoch path through2–4 or later checkpoints is inferred.

RMS of raw physical free states, before differential decoding, pooling and BN:

| Layer | Epoch0 | Epoch1 | Epoch5 | Epoch1 / initial |
|---|---:|---:|---:|---:|
| Block1 Conv1 | 0.194352 | 0.303136 | 0.208274 | 1.56× |
| Block1 Conv2 | 0.007594 | 0.067260 | 0.019255 | 8.86× |
| Block1 Conv3/output | 0.013175 | 0.123754 | 0.021585 | 9.39× |
| Block2 Conv1 | 0.282125 | 4.395784 | 2.453001 | 15.58× |
| Block2 Conv2 | 0.004320 | 1.746403 | 0.764197 | 404.2× |
| Block2 Conv3/output | 0.005143 | 3.806130 | 1.380153 | 740.0× |
| Block3 Conv1 | 0.188808 | 3.439313 | 1.875423 | 18.22× |
| Block3 Conv2/output | 0.011987 | 1.917366 | 0.374771 | 160.0× |

All eight RMS magnitudes increase at epoch1, then decrease at epoch5 while
remaining above initialization. The strongest relative increases are in the
later layers of blocks2/3. Pooling squared voltages over all internal nodes
within a block, with state-element weighting, gives block RMS vectors
[0.11255,0.16293,0.13378] at epoch0, [0.19299,3.50521,2.78435] at epoch1 and
[0.12140,1.68385,1.35234] at epoch5. These all-node magnitudes differ from the
last-convolution/block-output values above.

This directly matters for normalized displacement: R=D/F changes when the
free-state voltage RMS F changes. At the block2 output, the same absolute
displacement would give a relative displacement740× smaller at epoch1 than
initialization. This algebraic observation does not claim that D stayed fixed,
identify why voltages changed or establish that the voltage rise caused the
learning deficit. Raw analog voltages are also not post-BN activations.

[Layer voltage RMS and ratios to initialization (JPG)](../../../../../results/cifar-ours-epoch-beta-gradients-20260930-v1/analysis/voltage_evolution/eqprop_layer_voltage_rms.jpg)
and [block output versus all-node RMS (JPG)](../../../../../results/cifar-ours-epoch-beta-gradients-20260930-v1/analysis/voltage_evolution/eqprop_block_voltage_rms.jpg).
The existing root's `analysis/voltage_evolution/voltage_epoch_comparison.csv`
contains24 layer,9 block-output and9 block-aggregate rows. `summary.json` includes
the checkpoint provenance, ordered cohort hash, batch RMS and measurement limits.
Squared-sum pooling over eight batches, beta invariance across17 repeated cases
and equality of last-layer/block-output RMS pass. Prior checkpoint/BN integrity
validation is reused. Earlier BPTT output-voltage evidence is kept separately
in the [exp-008 result](exp-008-intermediate-rms.md).

## Float32 interpretation of the lower displacement bounds

Filip asks whether an absolute precision of 1e-7 explains the small-displacement
failure. Reinspection of the existing curves and frozen estimator source does
not support a universal absolute cutoff. No additional simulation was run.
Float32 epsilon is 2^-23=1.1920929e-7, defined as the gap above **1**, not a
fixed absolute resolution. Local `numpy.nextafter` checks give upward spacings
9.313e-10 at0.01, 1.490e-8 at0.2, 1.192e-7 at1, 4.768e-7 at4 and 7.629e-6
at100. These are representation spacings, not bounds on accumulated solver
error. See [NumPy finfo](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html)
and [spacing](https://numpy.org/doc/2.3/reference/generated/numpy.spacing.html).

First sampled absolute centered RMS D with pooled native cosine>=0.95:

| Layer | Epoch0 | Epoch1 | Epoch5 |
|---|---:|---:|---:|
| Block1 Conv1 | 7.13e-9 | 4.83e-9 | 2.16e-9 |
| Block1 Conv2 | 9.57e-8 | 3.89e-8 | 2.90e-8 |
| Block2 Conv1 | 1.10e-7 | 1.19e-7 | 1.29e-7 |
| Block2 Conv2 | 6.16e-7 | 4.23e-6 | 2.28e-6 |

These are grid-dependent first passing samples, not exact cutoffs. In
particular block1 Conv1 at epoch5 has cosine0.9838 with D=2.16e-9.
Block2 Conv1's approximately constant absolute crossing does not establish
a float32 constant: its free RMS changes0.282→4.396→2.453, and its relative
first-pass values are3.89e-7→2.72e-8→5.25e-8. Neither absolute nor relative
postsynaptic RMS alone makes all curves coincide.

The frozen `stable_centered_conv_gradient` uses float32 states, differences,
products and reductions. It factors the difference of squares before reducing;
it does not subtract two separately reduced large energy gradients. Conversion
to float64 happens afterward for cosine/pooling, and before displacement
statistics; neither restores phase-state differences already lost in float32.
The factorization mitigates cancellation from fixed input-voltage squared
terms but cannot eliminate solver-state rounding or all cancellation in the
remaining products and reductions.

Each convolution-weight gradient depends on both presynaptic and postsynaptic
phase changes. Our horizontal axis measures only the postsynaptic state.
Consequently a Conv2 lower bound need not be set by Conv2's own state spacing;
its input response and the numerical accuracy of the propagated solution also
matter. For Conv1 the external boundary is fixed across phases and its input
difference is identically zero; the large fixed input-square term cancels
algebraically. BN stabilizes this boundary's magnitude but does not normalize
the internal states or their phase differences.

The observed recovery at larger beta is consistent with a numerical
small-response limit, but does not by itself isolate floating-point error.
Pooled RMS may be below the spacing at the layer RMS because individual
coordinates have different magnitudes, many are clamped/unchanged, D is a
half-difference, and parameter gradients pool many coordinates/examples.
At epoch5 block3 Conv1, raising the beta multiplier1→30 increases
D=2.89e-8→5.02e-7 and cosine0.7868→0.9987; unchanged coordinates fall
84.5%→56.0%, which is suggestive but includes legitimate inactive units.
The deterioration at excessive nudging is a separate observed upper constraint.

A discriminating follow-up would compare identical low-beta cases using
(1) the current float32 computation, (2) float64 gradient arithmetic on the
same float32 endpoints, and (3) float64 phase-state solving and gradient
arithmetic. Record coordinatewise phase differences relative to local float32
spacing on active units, and compare both pre/post responses. This would
separate gradient-arithmetic cancellation from phase-state precision and test
whether the lower alignment boundary moves. This is a proposed diagnostic,
not a launched experiment or an authorization to resume training.
