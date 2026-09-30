# How should beta be chosen for good CIFAR-10 results?

Find beta values that support effective EqProp learning on CIFAR-10 L8.
Small responses may be difficult to resolve or susceptible to read noise;
large nudging may bias the gradient. A fixed beta does not fix RMS displacement:
network response and loss-gradient forcing change during training.

Current question: **does displacement becoming too small explain poor gradient
quality and contribute to the learning gap?** The
[epoch 0/1/5 replay](series/001-beta-selection/experiments/exp-022-epoch-layer-response.md)
sweeps beta on saved ours checkpoints and the same 256 training images. It
measures every convolution's internal response and gradient agreement with
BPTT. The [completed replay](series/001-beta-selection/results/exp-022-epoch-layer-response.md)
finds that larger beta largely restores trained gradients, whereas initialization
needs smaller beta. Epoch1 block3 remains less consistent across minibatches.
Accuracy recovery requires a separate training test.

New training launches remain stopped at Filip's September 30 request. Earlier
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
