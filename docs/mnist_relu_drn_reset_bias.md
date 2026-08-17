# Bias-inclusive RESET loss control

This exploratory study answers whether the archived approximately 95% cohort-A
MNIST result was reproduced simply by replacing its paired squared-error loss
with KL divergence. It was not. During the comparison, the archived result was
also traced to a process-global amplifier-index artifact, so it is retained as
an explicit provenance control rather than evidence for the intended circuit.

## Two distinct amplifier semantics

The corrected experiment is `mnist_relu_drn_reset_bias.v1`. It assigns the
three DRN layers the logical indices 0, 1, and 2 every time the model is built.
This is the intended interpretation of `voltage_amp = 4` and
`current_amp = 0.25`.

The historical code inferred amplifier exponents from process-global names
such as `Layer_3`. Its LR selector built one stack (`Layer_0..2`), destroyed
it, and rebuilt the production stack (`Layer_3..5`). Consequently:

- LR selection used dense edges 0→1 and 1→2;
- production training used dense edges 3→4 and 4→5;
- held-out validation in a fresh process again used 0→1 and 1→2.

The archived training and evaluation circuits therefore differed even though
the JSON still said `voltage_amp = 4` and `current_amp = 0.25`. The explicit
historical replay is registered separately as
`mnist_relu_drn_reset_bias_legacy.v1`, with
`amplification_indexing = legacy_process_global`. Checkpoint metadata and
result artifacts record the resolved indices so this behavior cannot be used
silently.

## Controlled setup

All three historical-control arms use the same single-device dual-rail
`[1568, 100, 20]` DRN:

- one non-negative measured ReRAM conductance per physical edge;
- one trainable digital bias on the 100-node hidden layer and no output bias;
- paired class score `s_k = y_(2k) - y_(2k+1)`;
- raw measured cohort-A traces with deterministic virtual-device assignment;
- global nearest-state projection after every minibatch;
- every conductance initialized at pulse index zero and the bias initialized at
  exact zero;
- `input_gain = 100`, four asynchronous iterations, and overrelaxation `1.1`;
- 20 production epochs with a validation-selected checkpoint.

Each objective independently runs the same bounded relative-update LR probe,
safety canaries, and 3×3 validation grid from an exactly restored RESET state.
The teacher, measured-device assignments, RESET fingerprint, and initial
metrics are identical across arms. Test data are used only after selection.

The objectives are:

- paired squared error:
  `0.5 * sum_k((s_k - one_hot(label)_k)^2)`, exactly the archived cost;
- cross-entropy on the hard label and paired scores;
- `KL(softmax(teacher) || softmax(s))` from the frozen bias-free
  `784 -> 50 -> 10` ReLU teacher.

## Results

The eight-stage local RTX 3090 campaign completed successfully. Training
artifacts report edges 3→4 and 4→5; fresh-process student tests report 0→1 and
1→2.

| Objective | Selected validation | Held-out test | Test gap vs paired | Selected LR vector (dense 0, dense 1, bias) |
|---|---:|---:|---:|---|
| Paired squared error | 95.10% | **95.97%** | — | `[1.84898e-7, 6.42698e-8, 1.84898e-7]` |
| Cross-entropy | 91.16% | **92.37%** | −3.60 pp | `[2.26008e-7, 8.70685e-8, 2.26008e-7]` |
| Teacher KL | 82.78% | **83.72%** | −12.25 pp | `[6.82423e-7, 8.82274e-8, 6.82423e-7]` |

The ReLU teacher reached 97.36% held-out accuracy. The paired arm reproduced
the archived selected validation accuracy, selected loss, epoch-1 trajectory,
and 95.97% test accuracy exactly. This validates the provenance replay.

The scientific interpretation is narrower: under the historical train/test
amplifier mismatch, the objective is material. Paired squared error succeeds,
cross-entropy loses 3.60 percentage points, and KL loses 12.25 points. Thus the
archived 95% run was not “the same run with KL”; changing only the loss does
not preserve its recovery. The replay does not establish that the intended
logical 0→1 / 1→2 circuit can reach 95%.

Canonical configs are in `examples/mnist_relu_drn_reset_bias_legacy/`, and the
campaign manifest is
`campaigns/manifests/mnist_relu_drn_reset_bias_legacy.json`. Raw results are
under
`results/mnist-relu-drn-reset-bias-legacy-loss-control-20260816-v1/`.
`labs/tools/plot_mnist_relu_drn_reset_bias_comparison.py` produces the checked
summary and validation/test comparison in the campaign's `analysis/` folder.

This remains a single-seed measured-device simulation. It establishes exact
historical provenance and a controlled objective comparison, not robustness
across seeds or fabricated on-chip behavior.
