---
experiment: "exp-024"
evidence: "validated-local"
summary: "Full float64 essentially removes the practical lower nudging bound seen in float32: very small beta multipliers down to 1e-6 pass every layer and batch at epochs0/1/5. Arithmetic-only promotion does not."
verdicts: {"H-018": "supports"}
---
# Result: precision limits the small-response regime

**Full float64 essentially removes the practical lower nudging bound for good
gradients observed in float32, allowing accurate gradients with very small
nudging.** This is a precision-dependent bound on resolving the response,
not a lower bound on gradient magnitude. The matched evidence below supports
this conclusion across epochs 0/1/5; it does not imply unlimited precision
as beta tends to zero.

## Evidence and coverage

Float32 parent: [exp-022](exp-022-epoch-layer-response.md), with artifacts in
[cifar-ours-epoch-beta-gradients-20260930-v1](../../../../../results/cifar-ours-epoch-beta-gradients-20260930-v1/).
Matched precision root:
[results/cifar-precision-response-20260930-v1/](../../../../../results/cifar-precision-response-20260930-v1/).
All three checkpoint
cells completed and passed local canonical validation and the declared coverage
checks: 1440 pooled layer rows, 11520 minibatch rows, 24 pooled cross-precision
reference comparisons and 192 reference comparisons by minibatch. Each cell
reproduced all 1088 overlapping float32 minibatch rows from exp-022. The two
float32-state arms have exactly identical displacement measurements. Independent
pooling checks of displacement sums, counts and coordinate classifications pass.

Same ours fixed-0.3× EqProp-trained checkpoints at epochs 0/1/5, ordered 256
training images, eight batches of 32, V4/C1, finite T/K=[6,6,4], perfect diodes,
loss and saved block beta vector as exp-022. Twenty beta multipliers per arm:
the original 17 plus 1e-10, 1e-8 and 1e-6. Float64 exactly promotes existing
checkpoint and preprocessed image values. BN uses minibatch statistics with
running buffers frozen. No optimizer, training, augmentation, noise or test
split. Parameter, BN, checkpoint-byte, input-identity and full-model dtype
checks passed. These are finite-iteration gradient comparisons.

Source identity SHA256:
`1284c71953c57aeff2379fb366ab3a74857fc7dd297b4a9f13cc473c4fffe3c0`.
The frozen source identity records the dirty source files and the exp-022
parent identity; exact checkpoint/config hashes remain in the manifests.
GPU runtime was torch2.11.0+cu128/CUDA12.8 on RTX5090: epochs 0/5 on Loulou,
epoch 1 on Fifi. Runner times were 1469.42/1561.56/1441.72 seconds, with
21.51GiB peak reserved memory each. Queue-charged total including validation
was 4479.22 seconds, within the 7200-second budget. Terminal receipts are in
`launch/*-terminal.json`; no assigned jobs remain active.

Four numerical tests passed. Initial CPU `smoke/` and its `smoke-source-v0/`
are preserved; inspection then corrected scalar dtype handling in the mixed
control, and the corrected `smoke-v1/` passed validation before all production
runs. Combined CPU smokes stayed below 300 seconds. The original Loulou epoch 1
queue entry was cancelled before any attempt and replaced on Fifi; it consumed
no compute. No production cases failed, were excluded or required retry.

Reproduce analysis and validation with
`python -m experiments.analyze_cifar_precision_response RESULTS_ROOT`.
`analysis/summary.json` has `complete_coverage: true`; `layers.csv` and
`lower_boundaries.csv` retain all measurements and sampled crossings.

## Matched-beta comparison

At multiplier **1e-4** of the saved block-specific beta vector, the minimum
pooled native-BPTT cosine across all eight convolutions is:

| Checkpoint | Float32 | Float32 states, float64 gradient arithmetic | Full float64 |
|---|---:|---:|---:|
| Epoch 0 | 0.435988 | 0.435988 | 0.997051 |
| Epoch 1 | -0.011888 | -0.011888 | 0.999792 |
| Epoch 5 | 0.001764 | 0.001764 | 0.999981 |

Full float64 passes cosine >=0.95 and gradient norm ratio 0.9–1.1 in **every
layer of every minibatch at every checkpoint** at this multiplier. Across
those 192 comparisons, minimum cosine is 0.995422 and norm ratios span
0.971868–1.042624. Across the entire sweep, changing only gradient arithmetic
changes pooled cosine by at most 2.88e-8 and recovers none of the failing
float32 cases at multipliers <=1.

This is not an artifact of a substantially different reference gradient.
The minimum pooled float32/float64 native-reference cosine is 0.999934
(minimum by minibatch 0.999833), with maximum pooled norm change 0.644%.
Those largest changes occur at initialization. At epochs 1/5, pooled reference
cosines exceed 0.99999989. Maximum relative changes across all checkpoints are
0.00304% for free-state RMS and 0.00674% for output-force RMS. These scale and
direction checks support comparability, but do not establish coordinatewise
identity of the free computation.

## Absolute and relative displacement

The measured displacement is `D = RMS((v_plus - v_minus)/2)` and relative
displacement is `D / RMS(v_free)`. First sampled points with pooled cosine
>=0.95 at epoch 1 illustrate the shift:

| Layer | Float32 D | Full float64 D | Float32 relative D | Full float64 relative D |
|---|---:|---:|---:|---:|
| Block1 Conv1 | 4.83e-9 | 1.17e-16 | 1.59e-8 | 3.86e-16 |
| Block1 Conv2 | 3.89e-8 | 1.29e-15 | 5.78e-7 | 1.92e-14 |
| Block2 Conv1 | 1.19e-7 | 3.67e-16 | 2.72e-8 | 8.34e-17 |
| Block2 Conv2 | 4.23e-6 | 4.23e-15 | 2.42e-6 | 2.42e-15 |

These are sampled cosine crossings, not exact thresholds or guarantees on
gradient norm or every minibatch. At initialization every full-float64 layer
passes even at the lowest multiplier 1e-10, so its lower boundary remains
unbracketed. At epochs 1/5 some layers fail there and recover at larger beta:
float64 moves the precision limit rather than eliminating it. Large-nudge
failures remain in both precisions, visibly in the last block.

Coordinate evidence agrees with this interpretation. At epoch 0, block1 Conv1,
multiplier 1e-4, 92.75% of active float32 coordinates have identical plus/minus
states. In full float64 that fraction is 0.000123%; 98.07% of active coordinates
have nonzero plus/minus differences smaller than one float32 spacing at their
free-state magnitude. Here active means nonzero free state, not a claim about
all clamping or derivative states. Float32 measured D is actually larger
(1.04e-9 versus 2.02e-10), showing that nonzero or larger RMS alone does not
guarantee an accurately resolved response.

Each plot includes all eight convolutions and all three controls:

| Checkpoint | Absolute RMS | Relative RMS | Beta multiplier |
|---|---|---|---|
| Epoch 0 | [JPG](../../../../../results/cifar-precision-response-20260930-v1/analysis/e0_cosine_vs_absolute_rms.jpg) | [JPG](../../../../../results/cifar-precision-response-20260930-v1/analysis/e0_cosine_vs_relative_rms.jpg) | [JPG](../../../../../results/cifar-precision-response-20260930-v1/analysis/e0_cosine_vs_beta_multiplier.jpg) |
| Epoch 1 | [JPG](../../../../../results/cifar-precision-response-20260930-v1/analysis/e1_cosine_vs_absolute_rms.jpg) | [JPG](../../../../../results/cifar-precision-response-20260930-v1/analysis/e1_cosine_vs_relative_rms.jpg) | [JPG](../../../../../results/cifar-precision-response-20260930-v1/analysis/e1_cosine_vs_beta_multiplier.jpg) |
| Epoch 5 | [JPG](../../../../../results/cifar-precision-response-20260930-v1/analysis/e5_cosine_vs_absolute_rms.jpg) | [JPG](../../../../../results/cifar-precision-response-20260930-v1/analysis/e5_cosine_vs_relative_rms.jpg) | [JPG](../../../../../results/cifar-precision-response-20260930-v1/analysis/e5_cosine_vs_beta_multiplier.jpg) |

### Does the float32 lower crossing occur around relative RMS 1e-8?

October 1 retrospective analysis uses **matched block-local BPTT**, rather
than the native reference in the earlier crossing table. For each of eight
layers at epochs 0/1/5, take the first sampled pooled cosine >=0.95 while
increasing beta, and the immediately preceding failing point. No interpolation,
norm gate, new simulations or changes to the source evidence. All 24 crossings
are bracketed; some initialization brackets span two decades of beta.

For the first convolution of each block at epoch5:

| Layer | Lower failing relative RMS (cosine) | First passing relative RMS (cosine) |
|---|---:|---:|
| Block1 Conv1 | 5.49e-9 (0.934) | 1.04e-8 (0.984) |
| Block2 Conv1 | 2.16e-8 (0.919) | 5.25e-8 (0.980) |
| Block3 Conv1 | 3.18e-8 (0.941) | 9.14e-8 (0.991) |

Across epochs0/1/5, first-convolution passing points span1.04e-8–3.89e-7.
Across **all layers**, their own passing RMS spans1.04e-8–1.43e-4: this is not
a common scalar threshold. A weight gradient uses both pre/post states; for
example, initialization block2 Conv2 first passes at its own RMS1.43e-4 while
its presynaptic layer has RMS1.30e-7. Float32 epsilon describes coordinatewise
resolution, not a hard bound on pooled relative RMS or gradient cosine.

At the same betas as all24 lower failing float32 points, full64 local-BPTT
cosine is >=0.999730. This supports numerical precision as the cause of this
lower deterioration. The pooled result is weaker than requiring every batch:
23/24 layer/checkpoint cases have a sampled every-minibatch passing point,
but epoch1 block3 Conv1 never passes (best minimum0.925524). The all-minibatch
crossing table retains pooled layer RMS as its x coordinate.

[Lower-tail curves](../../../../../results/cifar-precision-response-20260930-v1/analysis/float32_lower_floor/lower_tail_curves.jpg)
and [sampled crossing brackets](../../../../../results/cifar-precision-response-20260930-v1/analysis/float32_lower_floor/lower_crossings.jpg)
show each layer's own relative RMS. Reproduce with
`python results/cifar-precision-response-20260930-v1/analysis/float32_lower_floor/analyze.py`.
Adjacent `summary.json`, `curves.csv` and `boundaries.csv` preserve all values;
the analysis checks source artifact hashes, cohort identity and complete
20-multiplier/eight-batch coverage. These bounds concern ours only.

### Shared block-output displacement axes

Requested replot from the same saved measurements: every convolution's cosine
is now also plotted against its block's output displacement. Output means the
last physical analog state **before pooling and BN**: Conv3 in blocks 1/2,
Conv2 in block 3. Match the epoch, precision arm, block and beta multiplier
when selecting that output measurement. The relative axis is
`RMS((v_output_plus-v_output_minus)/2) / RMS(v_output_free)`; the absolute
version uses its numerator. All layers within a block share the same x values
and axis limits for a given arm. Every y value remains that layer's own
pooled native-BPTT cosine. Precision arms retain their own measured output
states and free-state normalization.

| Checkpoint | Relative RMS at block output | Absolute RMS at block output |
|---|---|---|
| Epoch 0 | [JPG](../../../../../results/cifar-precision-response-20260930-v1/analysis/e0_cosine_vs_block_output_relative_rms.jpg) | [JPG](../../../../../results/cifar-precision-response-20260930-v1/analysis/e0_cosine_vs_block_output_absolute_rms.jpg) |
| Epoch 1 | [JPG](../../../../../results/cifar-precision-response-20260930-v1/analysis/e1_cosine_vs_block_output_relative_rms.jpg) | [JPG](../../../../../results/cifar-precision-response-20260930-v1/analysis/e1_cosine_vs_block_output_absolute_rms.jpg) |
| Epoch 5 | [JPG](../../../../../results/cifar-precision-response-20260930-v1/analysis/e5_cosine_vs_block_output_relative_rms.jpg) | [JPG](../../../../../results/cifar-precision-response-20260930-v1/analysis/e5_cosine_vs_block_output_absolute_rms.jpg) |

The expanded `analysis/layers.csv` records `block_output_layer` and
`block_output_displacement_*` beside the existing layer-local fields.
`analysis/summary.json` records the axis definition and matching fields.
This is a change of plotting coordinates using the completed exp-024 evidence;
no new simulations or changes to the layer-local crossing tables.

### Output targets preserving every layer

Follow-up analysis scores each block/beta by the **minimum** pooled native-BPTT
cosine across its convolutions. High agreement means >=0.95 for every layer;
also retain the stricter >=0.99, every-minibatch >=0.95, and norm-ratio 0.9–1.1
criteria separately. No concatenation or averaging of layer cosines.
`experiments/analyze_cifar_block_output_targets.py` reuses the validated
artifacts after checking their hashes, checks all 540 block/arm/beta/checkpoint
cases and reconciles both joint-criterion classifications with the earlier
analysis. Undefined cosines fail. The arithmetic-only control has exactly the
same passing displacement spans as float32 for all five criteria.

The best **observed** float32 point for each block/checkpoint, maximizing the
minimum pooled layer cosine, is shown as `relative output RMS (minimum cosine)`:

| Block | Epoch 0 | Epoch 1 | Epoch 5 |
|---|---:|---:|---:|
| 1 | 0.014503 (0.999029) | 0.00044228 (0.999784) | 0.0027764 (0.999451) |
| 2 | 0.098979 (0.966014) | 0.0010706 (0.999453) | 0.0088524 (0.999885) |
| 3 | 0.015859 (0.999447) | 0.000021721 (0.987774) | 0.00022745 (0.998687) |

These are measured choices, not optimized continuous thresholds. Each passes
the pooled cosine/norm criterion. Block3 at epoch1 has no sampled float32
point passing every minibatch: the best point's minimum minibatch/layer cosine
is 0.92546. Block2 at initialization never reaches pooled cosine >=0.99 for
all layers; its best minimum is 0.96601.

Only float32 block1 has an overlap of the pooled >=0.95 passing sample spans
across epochs0/1/5: 0.0014525–0.0123187. Its every-minibatch cosine-only overlap
is 0.0014525–0.0042682. Blocks2/3 have no demonstrated common span. Requiring
pooled cosine >=0.99 or requiring both cosine/norm for every minibatch leaves
no common float32 span in any block on this grid. This does not rule out
unsampled narrow bands.

Full float64 has broad overlapping spans even at pooled cosine >=0.99:

| Block | Overlap of passing sample spans across epochs0/1/5 | Candidate fixed output relative RMS |
|---|---:|---:|
| 1 | 4.84e-11–4.27e-3 | 1e-3 (0.1%) |
| 2 | 1.03e-8–8.85e-3 | 1e-3 (0.1%) |
| 3 | 1.59e-10–2.17e-5 | 1e-6 (0.0001%) |

The overlap is the intersection of spans between consecutive passing samples,
not a claim that every interior displacement has been tested. Initialization's
lower bounds are the smallest sampled values, not established failure floors.
The nearest sampled points below and above each proposed target pass pooled
cosine >=0.99 and both cosine >=0.95/norm0.9–1.1 in every minibatch at all
three checkpoints. Across those bounding samples, the lowest pooled layer
cosine is 0.997051 and lowest minibatch/layer cosine is 0.993993. Some brackets
span two orders of magnitude. **Exact candidate targets have not been replayed
or calibrated**, and no continuous-band or training guarantee follows.

[Worst-layer cosine versus output displacement](../../../../../results/cifar-precision-response-20260930-v1/analysis/block_output_worst_layer_cosine.jpg)
compares float32/full float64 and all three checkpoints, with proposed targets
marked. Detailed measurements, contiguous passing sample groups, common spans,
limiting layers and target-bracketing points are in `analysis/block_output_targets.csv`
and `analysis/block_output_targets.json`. This remains analysis of exp-024;
it does not change or launch the separate all-scheme exp-025 assignment.

## Interpretation and decision

### Would separate betas for individual convolutions help?

Retrospective analysis of the existing grid compares the best shared beta
`max_beta min_layer cosine(layer,beta)` with independent per-layer choices
`min_layer max_beta cosine(layer,beta)`. Both choices use the same reference
gradients and 256-image cohort on which they are scored. This is an optimistic
selection comparison on the sampled grid, not a deployable selection rule,
held-out validation or training experiment. See `per_layer_beta_oracles` in
`analysis/block_output_targets.json` for all 27 arm/block/checkpoint cases,
selected betas, individual cosines, norm ratios and distinct phase-pair counts.

The intended intervention is to vary a block's output nudge and read each
layer's gradient from its preferred response region, then combine the selected
gradients for one update. The above comparison evaluates exactly this selection
from the existing independently computed phase pairs. It does not evaluate
a continuous beta ramp or transient readout. Preferred output displacements
are now included for each selected layer in the analysis artifact.

For example, float32 epoch1 block2 has the following best sampled readouts:

| Layer | Selected block-output relative RMS | Cosine at selected readout | Cosine at shared output RMS0.0010706 |
|---|---:|---:|---:|
| Conv1 | 0.0106818 | 0.999984 | 0.999869 |
| Conv2 | 0.0106818 | 0.999919 | 0.999453 |
| Conv3 | 0.000107158 | 0.999882 | 0.999830 |

The preferred output RMS differs by approximately100×, but high-cosine regions
overlap substantially. Hence different peak locations do not alone imply a
large improvement over a well-selected shared point. For any selected weight
gradient, both pre/post states must come from that same beta's positive/negative
phase pair with matching normalization. Combining neighboring layers' states
from their different preferred betas is not the estimator evaluated here.

| Float32 case | Best shared beta: minimum cosine | Independent layer betas: minimum cosine |
|---|---:|---:|
| Epoch0, block2 | 0.966014 | 0.966014 |
| Epoch1, block3 | 0.987774 | 0.987774 |
| Epoch1, block2 | 0.999453 | 0.999882 |
| Epoch5, block1 | 0.999451 | 0.999885 |

The largest increase in the worst-layer cosine across nine float32 cases is
0.00043361. Individual layers can gain more: initialization block2 Conv3
improves from 0.994907 to 0.999659, while Conv1 remains the bottleneck at
0.966014. The two weakest block/checkpoint cases therefore do not improve
in the worst-layer measure. In full float64 the largest worst-layer gain is
only about1.61e-6. These findings do not support predicting a large learning
gain from per-layer beta compared with an already well-chosen block beta.
They do not exclude gains from a denser beta grid, other checkpoints or a
different error/magnitude criterion.

In this implementation beta controls the output nudge for the whole coupled
block (`precision_pair`), and the gradient normalization uses that same beta.
Separate layer choices would require solving the block at each selected beta
and retaining the corresponding layer gradients. Merely replacing the
denominator for individual layers rescales their gradients without changing
cosine or recovering lost state response. Independently forcing internal
layers is a different intervention and needs a consistent objective and
gradient derivation; it is not tested by this selection comparison. This is
consistent with the scalar output-nudging formulation in
[Scellier and Bengio](https://arxiv.org/html/1602.05179v5).

**H-018 is supported in this replay regime.** The limiting numerical error is
upstream of the final gradient arithmetic: promoting already rounded float32
endpoints cannot recover the response, while full float64 does. Full promotion
also changes free solves, BN and forcing, so this does not isolate endpoint
storage from all other upstream operations. Separate post-BN distributions
were not measured here. The small changes in reference gradients, free RMS
and force RMS make a broad change of operating point an unlikely explanation.

The evidence rejects a universal absolute 1e-7 displacement cutoff. Float32
epsilon is relative to scale, and the effective response floor depends on
local voltages, accumulated arithmetic, propagation and both pre/post states
entering each weight gradient. Neither postsynaptic absolute RMS nor its
relative normalization completely characterizes those effects.

The tested multipliers **1e-6, 1e-4 and 3e-4** each pass the joint cosine/norm
criterion for all layers and all minibatches across all three checkpoints in
full float64. No sampled multiplier is common across checkpoints for even one
complete block in float32, including the arithmetic-only control. Thus the
earlier apparent need to raise beta during training is substantially affected
by precision. A changing RMS response per unit beta remains real, but these
results do not require an epoch-dependent beta merely to obtain good gradient
agreement in full float64 on this cohort.

The matched diagnostic is complete. Retain precision as an explicit factor
when interpreting beta adaptation and displacement optima. Existing trained
checkpoints were only replayed; this establishes neither a float64 training
accuracy benefit nor the cost of adopting it for training. No training change
or additional simulation is launched by this review.
