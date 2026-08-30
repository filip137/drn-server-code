# IBM OM standard-crossbar versus DRN protocol

## Question

This protocol measures how much same-array pulse recovery matters when the
same bias-free logical `784-50-10` MNIST network is implemented as:

```text
IBM OM analog MVM -> digital ReLU -> IBM OM analog MVM
```

The primary comparison is the recent four-device full-conductance
perfect-diode DRN endpoint cohort. The crossbar is an architecture control,
not a replacement DRN implementation.

The study is predeclared in
[`studies/mnist-ibm-om-crossbar-relu-onchip-importance-20260830-v1.json`](../studies/mnist-ibm-om-crossbar-relu-onchip-importance-20260830-v1.json).
Its pinned DRN evidence receipt is
[`studies/references/drn_ibm_om_endpoint_recovery_20260829.json`](../studies/references/drn_ibm_om_endpoint_recovery_20260829.json).

## Matched architecture and device accounting

The frozen crossbar source is the bias-free ReLU teacher with weight matrices
`784 x 50` and `50 x 10`. AIHWKit's standard `512 x 512` mapping limit gives
three physical tiles:

| Layer | Standard-crossbar tiles | Logical weights |
| --- | --- | ---: |
| input to hidden | `392 x 50`, `392 x 50` | 39,200 |
| hidden to output | `50 x 10` | 500 |
| total | 3 tiles | 39,700 |

Each crossbar weight uses one programmable OM active state and the preset's
fixed sampled intrinsic reference value. The effective signed state is

```text
q = a - r
```

The reference is part of the AIHWKit `SoftBoundsReferenceDevice` model. This
protocol reports 39,700 programmable active states and 39,700 fitted
reference values; it does not reinterpret those reference values as a known
number of fabricated devices.

The DRN has the same count of 39,700 logical weights but uses four non-negative
conductance states per logical weight, or 158,800 programmable states. Its
`1568-100-20` stored tensor dimensions express dual-rail encoding, not a
larger logical network. Unlike the crossbar, all DRN conductances enter its
passive KCL numerator and denominator. The standard crossbar performs ordinary
signed MVMs and has no passive loading denominator.

The two architectures share the teacher task, logical dimensions, data, and
recovery objective, but not an identical trained tensor. The crossbar maps the
exact teacher weights. The pinned DRN deployment starts from a separately
recovered physical DRN checkpoint. Results therefore report each
architecture's own source-to-deployment loss in addition to absolute test
accuracy.

The inference-state semantics are explicit rather than falsely labelled as
matched: the standard-crossbar comparator uses AIHWKit's apparent `q`
forward, whereas the pinned DRN receipt reports its persistent physical
conductances. Their accuracies are architecture-specific outcomes. The
Winsorized policy matches the sampled-bound treatment, but it does not make
the two reported forward states identical.

## Source and mapping

The exact teacher checkpoint is:

```text
data/mnist_relu_teacher_fixed_init_20260816.pt
SHA-256: 9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52
```

The CLI path is explicit and the runtime rejects a digest mismatch. There is
one shared absolute-maximum scale per logical layer, including across the two
input-layer tiles:

```text
alpha_l = max(abs(W_l)) / omega_l
q_target_l = W_l / alpha_l
omega = [1, 1]
```

The digital `alpha_l` factors restore the logical scale around each MVM. A
tile-local scale is forbidden because it would make the split itself an
uncontrolled calibration intervention.

## Device treatments

All arms use AIHWKit 1.1.0 `ReRamArrayOMPresetDevice`, frozen per-binding
construction seeds, repaired published corrupt sites, the exact preset pulse
equation, additive cycle-to-cycle variation, and apparent post-write verify
noise. The evidence is a model-based AIHWKit-preset control, not raw measured
device replay.

The primary arms reproduce the recent DRN control's model treatment: after
sampling and repair, every active-state support is intersected with raw
`a in [-1, 1]` without resampling, changing the reference, or changing pulse
sizes. The native-bound arms retain the sampled AIHWKit supports and are a
sensitivity analysis. Only the Winsorized arms support a bound-treatment
match with the pinned DRN results; even there, the apparent-versus-persistent
forward-state difference remains part of the architecture comparison.

The crossbar uses assignment seed `87004` and stochastic P&V endpoint labels
`89402-89405`. These labels pair stochastic conditions for analysis; the DRN
and crossbar populations have different shapes and are not the same physical
cells.

## Test ladder

The study advances through the following controls before recovery is interpreted:

1. Frozen floating-point teacher.
2. Per-cell continuous support envelope, with unsupported requested `q`
   values clamped only for this analysis control.
3. Deterministic zero-noise 128-SET-pulse codebook, with the lowest pulse
   count winning exact projection ties.
4. Direct stochastic P&V from RESET, using apparent verification, a
   half-nominal-step tolerance of `0.023725` in raw `x=(q+1)/2`, and a
   128-pulse cap. The post-write apparent state is the AIHWKit-style forward
   state; the hidden persistent state is evaluated separately.
5. Five fixed epochs of off-chip continuous HWA, exposing exact per-cell
   support clamps on every forward while retaining an FP32 `q` shadow.
6. Five fixed epochs of deterministic QAT, exposing the exact per-cell
   128-pulse codebook state on every forward through a straight-through
   estimator. The fixed fifth epoch is the primary HWA-only target.
7. Stochastic P&V of that QAT target, with a frozen arm that has zero on-chip
   optimizer updates and separate apparent-forward and hidden-persistent
   results.
8. One fixed epoch of same-array open-loop pulse-Adam on both layers, plus
   input-only and output-only recovery ablations.
9. Frozen-master redeployment onto untouched assignments `87005-87007`, with
   four independent programming streams per assignment. Portability of a
   same-array recovered persistent state is reported as a different transfer
   source, never folded into frozen-master redeployment.
10. QAT-frozen and QAT-plus-full-recovery repeats under native sampled bounds.

The apparent post-write effective state is authoritative for every
AIHWKit-style network forward. It is sampled when a programming pulse touches
the device and remains the exposed state until the next touch; it is not
per-inference read noise. The hidden persistent state is authoritative for
subsequent pulse evolution and is retained as a separate diagnostic. The
controller sees only target, apparent normalized verify state, and its own
history. Exact support reachability, apparent acceptance, persistent target
error, saturation, and budget exhaustion remain separate reported quantities.

## Off-chip adaptation match

Continuous HWA and deterministic QAT both start from the exact teacher-mapped
`q` tensor and use the same train order. They differ only in the state exposed
to each minibatch forward:

| Field | Value |
| --- | --- |
| minibatch size and data seed | `16`, `42` |
| epochs | `5` |
| objective | teacher KL, temperature 1 |
| Adam | beta1 `0.9`, beta2 `0.999`, epsilon `1e-8` |
| logical-layer learning rates | `[1e-4, 1e-4]` |
| shadow bounds | effective `q` in `[-1,1]` |
| continuous HWA forward | exact per-cell support clamp |
| QAT forward | nearest exact per-cell 128-pulse code, lower pulse index on ties |
| device writes during training | none |
| checkpoint | fixed fifth epoch; validation and test do not select |

Adam is implemented on two effective-`q` parameter groups. Each `q` learning
rate is the declared logical rate divided by the fixed digital scale for that
layer, so the Adam displacement is expressed in the logical-weight coordinate
apart from the epsilon term. The deployed target is the exact fixed-final
exposed state, not the unconstrained shadow.

## Recovery match

The matched DRN control used one epoch of digital teacher-KL gradients and
digital Adam moments followed by open-loop stochastic pulse selection. The
crossbar uses the same hybrid recovery class:

| Field | Value |
| --- | --- |
| minibatch size and data seed | `16`, `42` |
| epochs | `1` |
| objective | teacher KL, temperature 1 |
| Adam | beta1 `0.9`, beta2 `0.999`, epsilon `1e-8` |
| DRN learning rate | `3e-5` in raw `x` |
| crossbar learning rate | `6e-5` in effective `q=2x-1` |
| stochastic pulse probability | absolute Adam command divided by nominal pulse size |
| recovery cap | 64 pulses per active state |
| verify reads during recovery | 0 |
| checkpoint | fixed final epoch; validation and test do not select |

This changes persistent deployed devices on the same sampled array. The
gradient is calculated through the current apparent forward state and then
drives the existing pulse optimizer as an identity straight-through path to
the hidden persistent state; Adam moments remain digital. It is therefore a
same-array pulse-recovery control, not a claim of a fully local analog
learning circuit.

## Primary outcomes

The pinned exploratory DRN ladder begins at 94.14% for its recovered source
checkpoint, is 92.02% after the ideal requested-remap control before writing,
and then has these endpoint results:

| Endpoint | DRN P0 | DRN one-epoch Adam |
| ---: | ---: | ---: |
| 89402 | 62.49% | 93.62% |
| 89403 | 63.74% | 92.83% |
| 89404 | 66.29% | 92.74% |
| 89405 | 60.12% | 93.07% |
| mean | 63.16% | 93.065% |

For the crossbar, report all four endpoint values rather than pooling test
examples. The primary on-chip importance measures use the deterministic-QAT
target and its stochastic apparent-forward P&V endpoints. Hidden-persistent
accuracy is reported beside them as a programming-state diagnostic:

```text
paired recovery gain = final accuracy - P0 accuracy
gap closure = (final accuracy - P0 accuracy) / (source accuracy - P0 accuracy)
```

For the QAT recovery comparison, crossbar source means the exact fixed-final
QAT codebook target before stochastic writing. The report also retains the
97.36% floating teacher and the direct source mapping so adaptation loss and
gain remain visible. DRN source means its 94.14% recovered pre-transfer
checkpoint; the additional 92.02% DRN requested-remap control keeps logical
remapping loss separate from stochastic programming loss.

The study calls apparent-forward recovery negligible only if its absolute mean gain is below
one percentage point and every absolute endpoint gain is below two points. It
calls recovery needed for this protocol only after the QAT-frozen and
QAT-plus-recovery target and P0 hashes match, the QAT-frozen mean misses 90%,
and the recovered mean crosses 90%. Direct-map or continuous-HWA failure is
not sufficient. These thresholds do not establish a general hardware claim.

Layer ablations localize the recovery opportunity; they are not assumed to be
additive. Frozen-master redeployment is summarized across distinct target
assignments and programming streams. Same-array-recovered persistent-state
transfer is a separate test of whether recovery learned a portable task state
or assignment-specific compensation.

## Running the predeclared study

Use the repository's PyTorch environment for the network runtime and identify
the pinned AIHWKit sampler explicitly:

```bash
export EBL_AIHWKIT_PYTHON=/home/filip/miniconda3/envs/aihwkit/bin/python

python -m ebl study prepare \
  --plan studies/mnist-ibm-om-crossbar-relu-onchip-importance-20260830-v1.json \
  --results-root results
```

Run each declared config once under its matching arm directory. For example:

```bash
python -m ebl train \
  --config examples/mnist_analog_relu/ibm_om_onchip_importance/matched_winsorized_onchip_all.json \
  --output-dir results/mnist-ibm-om-crossbar-relu-onchip-importance-20260830-v1/runs/matched-qat-onchip-all \
  --teacher-weights data/mnist_relu_teacher_fixed_init_20260816.pt
```

After all eight arms complete:

```bash
python -m ebl study summarize \
  --study-dir results/mnist-ibm-om-crossbar-relu-onchip-importance-20260830-v1 \
  --verify-artifacts
```

Do not finalize or claim a conclusion until exact study coverage is present;
the source hashes must match across the ladder, and the off-chip target,
codebook, population, and endpoint P0 hashes must match across each QAT-frozen
and QAT-plus-recovery comparison.

### Fresh-array transfer smoke

The operational transfer check is separately frozen in
[`studies/mnist-ibm-om-crossbar-fresh-array-smoke-20260830-v1.json`](../studies/mnist-ibm-om-crossbar-fresh-array-smoke-20260830-v1.json).
It redeploys the direct, continuous-HWA, and deterministic-QAT frozen FP32
masters onto assignments `87005`, `87006`, and `87007`, with four writes per
assignment. Every target independently rebuilds its support and codebook.
The first 1,000 official MNIST test examples make this an exploratory smoke,
not the production on-chip-importance result.

```bash
python -m ebl study prepare \
  --plan studies/mnist-ibm-om-crossbar-fresh-array-smoke-20260830-v1.json \
  --results-root results
```

Run each declared smoke config once in its prepared arm directory, then use
`python -m ebl study summarize` to check exact three-arm coverage. Do not
finalize this operational smoke into the experimental manifest.

The artifact-verified smoke completed with three distinct target-population
fingerprints and four programming streams per population. Values below are
mean apparent-forward accuracies on the first 1,000 official test examples:

| Frozen master | Source `87004` | Target `87005` | Target `87006` | Target `87007` | Worst target delta | Smoke gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Direct teacher | 95.850% | 96.675% | 94.200% | 95.750% | -1.650 pp | pass |
| Continuous HWA | 95.775% | 96.225% | 94.100% | 95.575% | -1.675 pp | pass |
| Deterministic QAT | 88.650% | 83.650% | 73.700% | 92.450% | -14.950 pp | fail |

The QAT failure is primarily target-codebook transfer, not stochastic P&V.
Before writing, the three target-specific deterministic projections score
84.900%, 75.700%, and 92.700%; their apparent programmed endpoints score
83.650%, 73.700%, and 92.450%. Direct and continuous-HWA target no-write
maps remain at 97.700% and 96.900--97.400%, respectively. Thus the
single-assignment deterministic QAT master is array-specific in this smoke,
whereas the direct and continuous-HWA masters satisfy the predeclared
two-percentage-point portability screen.

The hidden-persistent diagnostic remains much lower: pooled target means are
66.975% for direct, 66.783% for continuous HWA, and 55.200% for QAT. This does
not replace the apparent AIHWKit forward result, but it confirms that the
one-read P&V controller has not estimated a robust persistent code. The smoke
uses only 16 off-chip training examples and 1,000 test examples, so a full
multi-assignment training/test study is still required before a production
claim.

## Claim limits

The implementation does not model inference read noise, retention, ADC/DAC
quantization, line resistance, IR drop, peripheral energy, or an
absolute-conductance calibration. Consequently it supports normalized
accuracy, mismatch, and pulse-count comparisons only. It does not support
power, energy, endurance lifetime, fabricated-device-count, or raw measured
hardware claims.
