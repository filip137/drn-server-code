# Ideas — choosing beta for CIFAR-10

We have seen in the MNIST experiments that beta can be relatively large and
still maintain good performance. We do not know whether this is also the case
for CIFAR-10; it could be more sensitive.

First check the cosine similarity between EqProp and BPTT gradients across beta.
Then compare short training runs of 10 epochs using different betas. Good gradient
alignment can guide candidate selection, but training results decide which beta
works well.

Beta selection must balance small-beta read-noise sensitivity against large-beta
gradient bias. The same beta may give different RMS during training because
effective response and loss-gradient forcing change. First replay frozen
initialization betas, then compare the requested 0.03/0.09 initialization-RMS
training arms in all three schemes.

## Linear outputs at block boundaries

Filip proposes removing each block's final nonlinearity so injected nudging/noise
acts on linear output units. The [digital pilot](../pilots/cifar-linear-block-outputs-20261002.md)
first measures L8 performance with the three final ReLUs removed, retaining the
eight convolutions and internal activations. One 50-epoch seed-0 run, zero weight
decay, local RTX 3090, training/validation only. Analog changes and the noise
mechanism remain deferred; [X-002](explorations/X-002-inactive-channel-force.md)
provides the inactive-channel motivation.

The follow-up [analog BPTT pilot](../pilots/cifar-analog-linear-block-outputs-20261002.md)
implements the same linear block outputs and queues fresh legacy/ours runs for
10 epochs on RTX 3090/5090 GPUs, with zero weight decay. This establishes analog
architecture baselines before any noisy or nudged-training comparison.

## Use float64 only where the responses require it

Test blocks1/2 in float32 and only block3 nudging in float64, with K[4,4,3],
at the existing output RMS targets. Block2's initial precision failure may prevent
using this from scratch even if trained checkpoints pass. See
[H-023](hypotheses/H-023-block3-only-float64.md) and
[exp-028](series/001-beta-selection/experiments/exp-028-block3-float64-k443.md).
The [review](series/001-beta-selection/results/exp-028-block3-float64-k443.md)
records checkpoint-dependent precision needs and the measured speed benefit.

## Reduce the cost of accurate nudging

Measure two independent changes: pack positive/negative phases into one doubled
solver batch, or reduce nudged K to[4,4,3] while keeping free T unchanged.
The hypotheses are lower overhead with more memory for packing, and less solver
work with possible gradient bias for shorter K. Compare speed, allocated/reserved
memory and matched initial gradients before adopting either for training.
See [H-021](hypotheses/H-021-packed-nudged-phases.md),
[H-022](hypotheses/H-022-reduced-nudged-k.md) and
[exp-027](series/001-beta-selection/experiments/exp-027-phase-k-performance.md);
the completed [review](series/001-beta-selection/results/exp-027-phase-k-performance.md)
separates performance gains from the initial gradient screen.

## What the precision comparison established

We suspected that float32 was responsible for the poor cosine alignment at
small nudges. Earlier MNIST Conv1/2/3 experiments used float64, whereas these
CIFAR replays used float32. **The matched float64 experiments confirmed this
hypothesis for the small-response failures in the tested CIFAR checkpoints.**
See [H-018](hypotheses/H-018-float32-response-floor.md) and the
[exp-024 results](series/001-beta-selection/results/exp-024-precision-response.md).

The comparison used the same ours checkpoints at epochs 0/1/5, 256 training
images, weights and solver iterations. At the same beta multiplier of 1e-4,
the worst cosine across all eight convolutions changed from
0.436 / -0.012 / 0.0018 in float32 to 0.9971 / 0.99979 / 0.99998 in full
float64. Every layer in every minibatch then passed cosine >=0.95 and gradient
norm ratio 0.9–1.1. Computing only the final gradient in float64, using the
same float32 states, did not recover alignment. The numerical limitation is
therefore upstream of the final gradient arithmetic. Full promotion includes
the free/nudged states, BN and forcing; this experiment does not isolate
rounding in one particular operation.

The subsequent [selective-precision pilot](../pilots/cifar-float64-performance-20261001.md)
narrows this down: float32 free states/forcing followed by float64 nudged solves
and gradient arithmetic passes the selected RMS targets at ours0/1/5. Across
every layer/minibatch, cosine against full64 EqProp is at least0.999851 and
norms differ by less than0.9%. This supports resolving the small nudged response
in64 without computing everything in64; long-training performance is untested.

This is not a universal absolute displacement limit of 1e-7. Float32 resolution
depends on voltage scale, and small propagated responses can be lost before
the gradient is formed. Float64 moves the lower useful-displacement boundary
by many orders of magnitude. The deterioration at excessive nudging remains.

The [local-reference lower-crossing analysis](series/001-beta-selection/results/exp-024-precision-response.md#does-the-float32-lower-crossing-occur-around-relative-rms-1e-8)
finds first-convolution pooled cosine crossings around relative RMS1e-8–4e-7
across ours0/1/5, broadly consistent with a precision floor. Other layers first
pass at their own RMS up to1.4e-4, so neither1e-8 nor float32 epsilon is a
universal RMS cutoff. At all24 adjacent lower failing float32 samples,
full64 local-reference cosine exceeds0.9997. Keep the every-minibatch criterion
separate; epoch1 block3 Conv1 still has no sampled passing point.

## Output RMS ranges that preserve every layer's gradient

Use the **relative centered RMS displacement at the output of each block**:
`r_output = RMS((v_output_plus - v_output_minus)/2) / RMS(v_output_free)`.
The output is the last analog state before pooling and BN: Conv3 in blocks
1/2 and Conv2 in block3. Plot every layer's cosine against that same output
displacement, and score the block by its **worst layer**, not its average.

In full float64, the passing sample spans overlap across epochs 0/1/5 as
follows. Values are fractions of the block output's free-state RMS:

| Block | Overlap for every layer cosine >=0.95 | Overlap for every layer cosine >=0.99 | Candidate fixed relative RMS |
|---|---:|---:|---:|
| 1 | 4.84e-11–1.23e-2 | 4.84e-11–4.27e-3 | **1e-3 (0.1%)** |
| 2 | 1.03e-8–2.97e-2 | 1.03e-8–8.85e-3 | **1e-3 (0.1%)** |
| 3 | 1.59e-10–2.17e-5 | 1.59e-10–2.17e-5 | **1e-6 (0.0001%)** |

These ranges intersect spans between consecutive passing samples; they do not
certify every interior displacement. The small lower endpoints are set by
the initialization sweep's smallest tested values, not measured failure floors.
The exact candidate targets have not been calibrated or replayed. However,
the measured points immediately below and above each target pass at every
checkpoint: pooled layer cosines are all >=0.99705, minibatch/layer cosines
are all >=0.99399, and every minibatch passes the 0.9–1.1 norm-ratio criterion.
Some bounding samples are separated by two orders of magnitude.

The corresponding float32 ranges are much more restrictive and change during
training. Only block1 has an overlap of pooled cosine >=0.95 passing sample
spans across all three checkpoints: approximately **0.00145–0.0123**.
Blocks2/3 have no demonstrated common span on this grid. At initialization,
block2's best worst-layer cosine is only 0.9660. At epoch1, block3's best
pooled worst-layer cosine is 0.9878, but the worst minibatch/layer is 0.9255.
Promoting gradient arithmetic alone leaves these passing spans unchanged.

The [worst-layer comparison plot](../../results/cifar-precision-response-20260930-v1/analysis/block_output_worst_layer_cosine.jpg)
and [detailed review](series/001-beta-selection/results/exp-024-precision-response.md)
contain the curves, measured choices and separate minibatch checks. These
are good **gradient-alignment** ranges for the tested noiseless ours trajectory;
they do not yet establish improved learning or transfer to other schemes.
The completed [all-scheme replay](series/001-beta-selection/results/exp-025-all-scheme-layer-rms.md)
now has local-reference plots for every layer. Against matched block-only BPTT,
all blocks have shared sampled targets in each checkpoint group; EqProp0/1/5
targets[0.001,0.005,0.00001] pass cosine>=0.99 in every layer/minibatch.
The baseline epoch1 block3 exception remains only against full-network BPTT.
Separate local beta/nudging error from changes between the two BPTT reference
setups, and keep the two training histories distinct.

## Read each layer at a different output nudge

The proposed use of per-layer beta is to **vary the block's output nudge and
read each layer's gradient at the point where that layer has high cosine**.
Keep the input, weights and reference free state fixed while collecting the
readouts, then assemble the layer gradients for one weight update. Early
layers could be read at a larger output response while later layers are read
at a smaller response. The existing replay uses separately computed phase
pairs; a continuous beta ramp that reuses states has not been tested.

The required pattern does occur. In float32 at epoch1, block2's best sampled
output relative RMS is **0.010682 for Conv1/Conv2**, versus **0.00010716 for
Conv3**, approximately100 times smaller. Their respective maximum cosines are
0.999984, 0.999919 and 0.999882. Thus different layers can be read near their
individual best points rather than all at one block-output response.

The size of the benefit depends on overlap between their high-quality regions,
not just different peak locations. In this example a shared output RMS of
0.0010706 already gives cosines0.999869/0.999453/0.999830. Separate readouts
improve these, but the broad plateaus overlap. The retrospective comparison
in exp-024 already evaluates this selection: the worst-layer cosine increases
by at most0.000434 across nine float32 block/checkpoint cases. This measure
does not capture every individual-layer improvement; initialization block2
Conv3 improves0.994907→0.999659 while Conv1 remains the limiting layer.
The weakest cases, initialization block2 and epoch1 block3, do not improve
in their worst-layer cosine on the existing grid. This is an optimistic
same-cohort comparison, not evidence of a large training gain.

For each convolution's gradient, read **both its presynaptic and postsynaptic
states from the same selected positive/negative beta pair**, and normalize
using that beta. Conv2's gradient must not mix its preferred Conv2 state with
a Conv1 state collected at a different beta. The readout is selected per
weight gradient, with the necessary neighboring states collected together.

## Initialization-only nonlinear readout pilot

A coupled nonlinear dense readout may change the output-force distribution and
widen the useful RMS range. The [four-batch initialization pilot](../pilots/cifar-nonlinear-readout-initialization-20261001.md)
found no widening for block 3 in float32 or float64. In float32, block 2's
sampled passing span widened 3.75x, while block 1 narrowed to 0.30x.
This does not establish a training benefit. Filip has now authorized matched
five-epoch BPTT training and the same local comparison at epochs1 and5; the
pilot note owns that follow-up.

## Inactive output channels absorb block3's nudging force

After training, some block3 output channels are off for the whole minibatch.
Their zero batch variance makes the bridge BatchNorm multiply their force by
gamma/sqrt(eps), so almost all injected current reaches units that cannot respond
linearly and carry no BPTT gradient, until they switch. This explains block3's
tiny upper RMS limit; masking that force restored a broad float32 window in an
interactive replay. See [X-002](explorations/X-002-inactive-channel-force.md),
[exp-030](series/001-beta-selection/experiments/exp-030-float32-force-mask-replay.md)
(all18 checkpoints) and [exp-031](series/001-beta-selection/experiments/exp-031-float32-masked-training.md)
(float32 training).

## Ideas and linked work

| Idea | Priority | Exploration | Hypothesis / experiment |
|---|---|---|---|
| Linear analog block-output layers (optionally followed by a digital ReLU) may widen the beta window, because the nudged units can no longer switch | Current: initialization replay of diode, digital-ReLU and linear outputs in float32/float64 | [X-002](explorations/X-002-inactive-channel-force.md) | [H-027](hypotheses/H-027-linear-block-outputs.md), [exp-032](series/001-beta-selection/experiments/exp-032-block-output-activation-replay.md) |
| Zeroing the injected force on inactive block-output channels may restore block3's usable float32 output displacement at every checkpoint | Current: float32 masked replay of all18 checkpoints with a block3 unmasked control | [X-002](explorations/X-002-inactive-channel-force.md) | [H-025](hypotheses/H-025-inactive-channel-force-mask.md), [exp-030](series/001-beta-selection/experiments/exp-030-float32-force-mask-replay.md) |
| With the force mask, float32 hybrid EqProp may learn far better than without it at shared float32-compatible targets | Contradicted: epoch5 60.88% masked versus 60.68% unmasked; both well above earlier fixed-beta float32. Next candidate: matched periodic-calibration versus fixed-beta float32 test | [X-002](explorations/X-002-inactive-channel-force.md) | [H-026](hypotheses/H-026-float32-masked-training.md), [exp-031](series/001-beta-selection/experiments/exp-031-float32-masked-training.md) |
| Diode switching during nudging may explain why trained block3 needs a tiny output RMS even in float64 | Quantify per-update margins, response linearity and fixed-decision gradient recovery | [X-001](explorations/X-001-beta-sensitivity.md) | [H-024](hypotheses/H-024-block3-diode-switching.md), [exp-029](series/001-beta-selection/experiments/exp-029-block3-diode-switching.md) |
| Recalibrating a shared output RMS several times per epoch may improve float64 learning over beta selected at epoch5 | Cancelled before epoch1: profile/fix float64 runtime before resuming | [X-001](explorations/X-001-beta-sensitivity.md) | [H-020](hypotheses/H-020-periodic-float64-beta-training.md), [exp-026](series/001-beta-selection/experiments/exp-026-float64-periodic-vs-fixed-beta.md) |
| A common output d/f for each block may keep all of its internal-layer gradients aligned and provide a rule for sizing beta | Use matched block-only comparisons to size beta; retain the full-network reference discrepancy as a separate question | [X-001](explorations/X-001-beta-sensitivity.md) | [H-019](hypotheses/H-019-common-layer-compatible-output-displacement.md), [exp-025](series/001-beta-selection/experiments/exp-025-all-scheme-layer-rms.md) |
| Float32 state precision sets the small-beta gradient-agreement floor in the tested CIFAR replays, unlike the earlier float64 MNIST runs | Confirmed by the matched float64 comparison at epochs0/1/5; arithmetic-only promotion does not recover alignment | [X-001](explorations/X-001-beta-sensitivity.md) | [H-018](hypotheses/H-018-float32-response-floor.md), [exp-024](series/001-beta-selection/experiments/exp-024-precision-response.md) |
| RMS displacement can change greatly during training; responses that become too small may explain poor EqProp gradients and learning | Current interpretation: fixed-checkpoint layer sweeps completed; see linked review | [X-001](explorations/X-001-beta-sensitivity.md) | [H-017](hypotheses/H-017-small-trained-layer-response.md), [exp-022](series/001-beta-selection/experiments/exp-022-epoch-layer-response.md) |
| Complete the missing lower-ceiling/larger-block3-target cell to separate target and ceiling effects | Current: one fresh epoch, queued below baseline extensions | [X-001](explorations/X-001-beta-sensitivity.md) | [H-016](hypotheses/H-016-low-ceiling-larger-block3-rms.md), [exp-021](series/001-beta-selection/experiments/exp-021-low-cap-block3-rms.md) |
| Trained-checkpoint RMS targets may fail to preserve the actual gradients used in early learning | Current: short exact-prefix comparison of failed10,000/30,000-ceiling trajectories | [X-001](explorations/X-001-beta-sensitivity.md) | [H-015](hypotheses/H-015-early-selected-gradients.md), [exp-019](series/001-beta-selection/experiments/exp-019-early-selected-gradients.md) |
| The larger block3 RMS target recommended across trained checkpoints may improve fresh learning | Completed:32.88% improves on28.04% at the same ceiling, but still misses matched-BPTT gates | [X-001](explorations/X-001-beta-sensitivity.md) | [H-014](hypotheses/H-014-block3-rms-training.md), [exp-018](series/001-beta-selection/experiments/exp-018-block3-rms-training.md) |
| Early beta clipping may limit current-batch response selection | Completed: all RMS misses removed, but accuracy worsened32.64→28.04%; no extension | [X-001](explorations/X-001-beta-sensitivity.md) | [H-013](hypotheses/H-013-current-beta-ceiling.md), [exp-017](series/001-beta-selection/experiments/exp-017-current-beta-ceiling.md) |
| Select beta from the current minibatch before its update to avoid response-delay overshoots | Completed: better CE and RMS tracking, but32.64% missed the accuracy gate | [X-001](explorations/X-001-beta-sensitivity.md) | [H-012](hypotheses/H-012-current-batch-beta.md), [exp-015](series/001-beta-selection/experiments/exp-015-current-batch-beta.md) |
| Updating beta from each training minibatch may track response changes missed by fixed or epoch-wise beta | Completed: median RMS tracked, but lagged34.34% versus fixed34.22% and worse CE; continuation gate failed | [X-001](explorations/X-001-beta-sensitivity.md) | [H-011](hypotheses/H-011-minibatch-beta-feedback.md), [exp-014](series/001-beta-selection/experiments/exp-014-minibatch-beta-feedback.md) |
| Trained-checkpoint betas may transfer to fresh learning, or require a different early-stage policy | Current: all three schemes, full-epoch screens and conditional extensions toward50 epochs | [X-001](explorations/X-001-beta-sensitivity.md) | [H-010](hypotheses/H-010-fresh-trained-beta-transfer.md), [exp-012](series/001-beta-selection/experiments/exp-012-fresh-beta-staged.md), [exp-013](series/001-beta-selection/experiments/exp-013-all-scheme-long-eqprop.md) |
| Betas aligned at epoch10 may preserve short BPTT-like learning | Current: five beta vectors and two autograd controls, identical half-epoch continuations | [X-001](explorations/X-001-beta-sensitivity.md) | [H-009](hypotheses/H-009-trained-beta-short-learning.md), [exp-011](series/001-beta-selection/experiments/exp-011-trained-beta-half-epoch.md) |
| Recalibrating beta each epoch may preserve effective nudging and improve actual learning | Stopped early by Filip: epoch5 adaptive51.82%, fixed53.56%, historical BPTT66.68%; epoch10 verdict inconclusive | [X-001](explorations/X-001-beta-sensitivity.md) | [H-008](hypotheses/H-008-epoch-beta-adaptation.md), [exp-010](series/001-beta-selection/experiments/exp-010-epoch-beta-adaptation.md) |
| Fixed initialization beta may drift away from its useful RMS/alignment range during training | Current: freeze initialization RMS 0.03/0.09 betas across epochs 10/30/50 | [X-001](explorations/X-001-beta-sensitivity.md) | [H-007](hypotheses/H-007-fixed-beta-transfer.md), [exp-009](series/001-beta-selection/experiments/exp-009-fixed-beta-transfer.md) |
| Large nudging may distort CIFAR gradients more than it does on MNIST | First: fixed-T/K cosine sweep | [X-001](explorations/X-001-beta-sensitivity.md) | [H-001](hypotheses/H-001-large-nudging-error.md), [exp-001](series/001-beta-selection/experiments/exp-001-smaller-beta-cosine.md) |
| Compare different betas directly through 10-epoch CIFAR learning | Next: select candidates from the diagnostic and define the EqProp training update | [X-001](explorations/X-001-beta-sensitivity.md) | [exp-002](series/001-beta-selection/experiments/exp-002-ten-epoch-beta.md); training hypothesis to be frozen with its contract |
| A shared output RMS range may select different betas with high cosine across CIFAR blocks | Current: focused sweep | [X-001](explorations/X-001-beta-sensitivity.md) | [H-002](hypotheses/H-002-common-output-displacement.md), [exp-003](series/001-beta-selection/experiments/exp-003-common-rms.md) |
| A common RMS target may transfer between ours, baseline and legacy | Current: repeat the focused sweep in the other schemes | [X-001](explorations/X-001-beta-sensitivity.md) | [H-003](hypotheses/H-003-displacement-transfer-across-schemes.md), [exp-004](series/001-beta-selection/experiments/exp-004-baseline-legacy-rms.md) |
| Block2's lower alignment may persist after BPTT training rather than being an initialization effect | Current: replay original epoch50 checkpoints | [X-001](explorations/X-001-beta-sensitivity.md) | [H-004](hypotheses/H-004-trained-block2-alignment.md), [exp-006](series/001-beta-selection/experiments/exp-006-trained-e50-rms.md) |
| More nudged iterations may improve block2's response at fixed beta | Current: epoch50 block2 K6/12/24/48 screen | [X-001](explorations/X-001-beta-sensitivity.md) | [H-005](hypotheses/H-005-nudged-k-block2.md), [exp-007](series/001-beta-selection/experiments/exp-007-fixed-beta-nudged-k.md) |
| Useful RMS may depend on training stage, requiring a compromise or changing target | Current: compare epochs0/10/30/50 and recommend per-block targets | [X-001](explorations/X-001-beta-sensitivity.md) | [H-006](hypotheses/H-006-intermediate-rms-transfer.md), [exp-008](series/001-beta-selection/experiments/exp-008-intermediate-rms.md) |
