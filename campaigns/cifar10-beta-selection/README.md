# How should beta be chosen for good CIFAR-10 results?

**Completed architecture pilot (October 2):** [linear block outputs](../pilots/cifar-linear-block-outputs-20261002.md)
remove the final ReLU in each L8 block. The digital variant, trained for 50 epochs
with zero weight decay on the local RTX 3090, reaches **91.36% best / 91.34% final
validation accuracy**. This single-arm pilot establishes digital feasibility;
relative accuracy and analog/noise effects remain untested.
The [analog follow-up](../pilots/cifar-analog-linear-block-outputs-20261002.md)
now implements linear block outputs for 10-epoch legacy/ours BPTT baselines,
queued only on RTX 3090/5090 GPUs with zero weight decay.

**Current direction (October 2):** block3's tiny usable output RMS is
explained by inactive output channels whose BatchNorm-amplified force reaches
units that cannot respond until they switch ([X-002](explorations/X-002-inactive-channel-force.md)).
[exp-030](series/001-beta-selection/experiments/exp-030-float32-force-mask-replay.md)
replayed all18 checkpoints in float32 with that force masked. The mask is exact
but restores block3 only where exactly-dead channels dominate. In five-epoch float32
training ([exp-031](series/001-beta-selection/results/exp-031-float32-masked-training.md))
it made no difference: 60.88% versus 60.68% unmasked at epoch5, H-026
contradicted. Both arms are well above earlier fixed-beta float32 ours (54.12%)
but 5.8 points below native BPTT. Periodic recalibration to small block targets,
not the mask, is the candidate lever (untested as a matched comparison).

Find beta values that support effective EqProp learning on CIFAR-10 L8.
**Full float64 essentially removes the practical lower nudging bound for good
gradients seen in float32: very small nudging still gives accurate gradients
in the tested noiseless replay.** The
[matched precision comparison](series/001-beta-selection/results/exp-024-precision-response.md)
passes cosine >=0.95 and gradient norm ratio 0.9–1.1 in every convolution and
every minibatch at epochs 0/1/5 with beta multipliers 1e-6, 1e-4 and 3e-4 of
the saved block-specific beta vector. At 1e-4, the worst minibatch cosine is
0.995422. Float64 gradient arithmetic applied to already-computed float32
nudged states does not recover the lost response. The subsequent
[selective-precision pilot](../pilots/cifar-float64-performance-20261001.md)
finds that float32 free/autograd with float64 nudged solves and gradient
arithmetic preserves full64 EqProp gradients at the selected RMS targets
for ours0/1/5 (minimum cosine0.999851, norms within0.9%). Full64 throughout
is therefore unnecessary for these tested checkpoint/target pairs.
Float64 still has a numerical floor at much smaller scales, and large nudging
can still bias the gradient. These results establish gradient agreement,
not a float64 training-accuracy gain or robustness to read noise.

Current question: **does displacement becoming too small explain poor gradient
quality and contribute to the learning gap?** The
[epoch 0/1/5 replay](series/001-beta-selection/experiments/exp-022-epoch-layer-response.md)
sweeps beta on saved ours checkpoints and the same 256 training images. It
measures every convolution's internal response and gradient agreement with
BPTT. The [completed replay](series/001-beta-selection/results/exp-022-epoch-layer-response.md)
finds that larger beta largely restores trained gradients in float32, whereas
initialization needs smaller beta. Its source bundle is
[cifar-ours-epoch-beta-gradients-20260930-v1](../../results/cifar-ours-epoch-beta-gradients-20260930-v1/).
The float64 follow-up shows that this apparent need to raise beta is substantially
precision-dependent: common very small multipliers work across all three
checkpoints. Accuracy recovery requires a separate training test.

The completed full-float64 [all-scheme replay](series/001-beta-selection/results/exp-025-all-scheme-layer-rms.md)
now separates local gradient estimation from full-network reference changes.
Against matched block-only BPTT, EqProp0/1/5 output d/f targets
**[0.001,0.005,0.00001]** and BPTT0/10/30/50 targets
**[0.0025,0.001,0.0005]** keep every layer/minibatch cosine>=0.99 and norm
ratio0.9–1.1 across all schemes. Against full-network BPTT, baseline EqProp
epoch1 block3 still prevents a common sampled target. Use the local comparison
to size beta while investigating this separate reference discrepancy.

Filip stopped the new [five-epoch float64 comparison](series/001-beta-selection/experiments/exp-026-float64-periodic-vs-fixed-beta.md): ours with four RMS recalibrations per epoch versus fixed beta selected on the saved epoch5 checkpoint. Both arms stopped before epoch1 completed because updates took about10.2s. The [performance pilot](../pilots/cifar-float64-performance-20261001.md) removed redundant gradients/coefficient work and then tested selective precision: about4.2s/update versus9.2s for optimized full64 in the latest short shared-GPU profiles. Gradient replay passes at ours0/1/5, but training accuracy has not been tested. Production precision is unchanged and training remains stopped. Earlier
[all-scheme training](series/001-beta-selection/experiments/exp-013-all-scheme-long-eqprop.md)
and the [current-batch target/ceiling comparison](series/001-beta-selection/results/exp-021-low-cap-block3-rms.md)
are evidence for this investigation. Maintaining a chosen block-output RMS did
not reliably recover matched BPTT performance. The
[early-gradient replay](series/001-beta-selection/results/exp-019-early-selected-gradients.md)
shows why each layer and individual minibatches need examination.

Scope: noiseless Conv-only EqProp with autograd boundaries, BN, gains and readout;
ours V4/C1, L8 blocks [3,3,2], operational T/K [6,6,4]. Keep schemes, runtime
cohorts, finite-iteration references and training histories distinct. Gradient
cosine guides diagnostics; matched validation accuracy and loss decide learning
performance. Read-noise sensitivity is not established by noiseless replay.
Use training/validation data only, without the official test split.

[Ideas](ideas.md) · [Exploration](explorations/X-001-beta-sensitivity.md) ·
[Current hypothesis](hypotheses/H-017-small-trained-layer-response.md) ·
[Ledger and earlier results](ledger.md) ·
[Starting pilot](../pilots/cifar-block-beta-20260929.md)
