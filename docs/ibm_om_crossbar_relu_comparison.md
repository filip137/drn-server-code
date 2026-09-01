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
   optimizer updates and separate apparent-forward and persistent-`q`
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
`python -m ebl study summarize` to check exact three-arm coverage. Following
artifact verification and human review, record this smoke in the experimental
manifest only as exploratory model-based evidence; do not promote it to
production evidence about on-chip-learning necessity.

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

The persistent-`q` diagnostic remains much lower: pooled target means are
66.975% for direct, 66.783% for continuous HWA, and 55.200% for QAT. This does
not replace the apparent AIHWKit forward result, but it confirms that the
one-read P&V controller has not estimated a robust persistent code. The smoke
uses only 16 off-chip training examples and 1,000 test examples, so a full
multi-assignment training/test study is still required before a production
claim.

### Published-defect fresh-array extension

The matched defect-inclusive extension is predeclared in
[`studies/mnist-ibm-om-crossbar-fresh-array-published-defects-smoke-20260831-v1.json`](../studies/mnist-ibm-om-crossbar-fresh-array-published-defects-smoke-20260831-v1.json).
It repeats the direct, continuous-HWA, and deterministic-QAT transfer arms
without modifying or reusing the completed repaired study. Each new config is
identical to its repaired counterpart except that `corruption_policy` is
`published` rather than `counterfactual_repaired`. Consequently, sampled
stuck/corrupt identities remain at their physical addresses instead of being
replaced by healthy donor parameters.

The extension retains source assignment `87004`, targets `87005-87007`, all
source and target endpoint seeds, the 16-example off-chip adaptation budget,
and the first-1,000-test-example evaluation cohort. The sampled intrinsic
reference remains active through `q=a-r`; this subtraction does not make a
stuck active state programmable. Pairing by assignment and endpoint position
therefore isolates the corruption-policy intervention from ordinary identity
and programming-stream changes.

For every source and target population, report sampled published-corrupt and
final surviving-corrupt counts and fractions, including per-tile counts and
mask/fingerprint provenance. Report target-specific no-write accuracy and all
four apparent-forward and persistent-`q` diagnostic endpoint accuracies, together with
corrupt-device acceptance, saturation, budget exhaustion, and pulse counts.
The paired headline is published-minus-repaired apparent accuracy. Defects are
called negligible in this smoke only if no-write and mean apparent accuracy
are each no more than two percentage points below the repaired result for
every arm and assignment. The original target-versus-source two-point
portability criterion is applied separately within the published-defect
population. This remains a model-based operational smoke, with no defect-aware
remapping, spares, or fabricated-array claim.

The artifact-verified extension completed with every sampled defect retained:
`5,366/39,700` cells on source `87004`, followed by `5,480`, `5,446`, and
`5,284` cells on targets `87005-87007`. These are defect fractions of
`13.31-13.80%`. Exact artifact comparison proves that each published-corrupt
mask and every healthy cell's sampled parameters match the paired repaired
assignment; only the corrupt sites use donor parameters in the repaired
control. Per-tile counts and endpoint-level defect outcomes are retained in
the run summaries.

Mean apparent-forward accuracies on the first 1,000 test examples are:

| Frozen master | Published source `87004` | Target `87005` | Target `87006` | Target `87007` | Pooled target | Pooled published-minus-repaired | Published portability gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Direct teacher | 93.850% | 89.600% | 87.225% | 92.450% | 89.758% | -5.783 pp | fail |
| Continuous HWA | 85.650% | 80.200% | 82.725% | 87.125% | 83.350% | -11.950 pp | fail |
| Deterministic QAT | 76.000% | 73.300% | 72.275% | 83.325% | 76.300% | -6.967 pp | fail |

No arm satisfies the original two-point fresh-array criterion. Target-minus-
source changes are `-4.250/-6.625/-1.400` points for direct deployment,
`-5.450/-2.925/+1.475` for continuous HWA, and
`-2.700/-3.725/+7.325` for QAT. The defects are also non-negligible under the
paired repaired-control criterion: source apparent accuracy changes by
`-2.000`, `-10.125`, and `-12.650` points for direct, HWA, and QAT,
respectively.

Most HWA/QAT loss is present before stochastic programming. The source
no-write maps change from `97.200%` to `88.300%` for continuous HWA and from
`91.000%` to `78.600%` for QAT. Fresh-target HWA no-write accuracy is
`82.600/89.600/89.600%`, while QAT reaches
`74.800/73.800/85.000%`. These policies use the exact one-state support of a
collapsed cell, so every defective weight is mapped to its immutable stuck
`q`; P&V then accepts all of those persistent targets. For HWA and QAT the
paired comparison is the complete corruption-policy path, including rerunning
off-chip adaptation against the defective source array, not deployment of an
identical adapted master.

Direct deployment behaves differently. Its no-write metric remains the ideal
unconstrained master at `97.700%`; defects enter when that target is sent to
P&V. No corrupt direct target is exactly in support. Nevertheless, repeated
zero-motion pulse attempts resample the preset's apparent write-noise term,
so roughly `95-98%` of corrupt sites eventually pass apparent verification
even though most persistent states remain outside tolerance. This is the
literal modeled AIHWKit-style endpoint, but it is optimistic for a defective
cell unless later fresh-read, retention, and inference-noise behavior is also
validated. The result therefore motivates an explicit defect-remapping or
spare-row/column control before testing whether on-chip updates can compensate
for surviving defects.

## Open-loop supervised-BP pulse control and corrected retraining follow-up

The first supervised-BP control is declared in
[`studies/mnist-ibm-om-crossbar-supervised-retraining-recovery-20260831-v1.json`](../studies/mnist-ibm-om-crossbar-supervised-retraining-recovery-20260831-v1.json).
It asks whether the same chronological post-deployment damage is compensable
at all before interpreting failure of a local recovery objective. It is not
the older `pulse_adam` arm: that arm uses teacher KL and does not apply the
post-deployment fault. The new `supervised_ce_pulse_adam` policy performs:

```text
healthy programmed P0
-> matched published-companion corrupt-device transition
-> ground-truth cross-entropy BP on apparent q
-> digital Adam command
-> stochastic physical pulse on persistent q
```

The update helper receives labels, inputs, current apparent `q`, and a
restricted pulse port. It receives neither teacher logits nor the fault mask.
Ordinary BP nevertheless propagates the output error through `W2^T`, uses
autograd, and retains FP32 first and second Adam moments outside the array.
The plant alone enforces the immutable sampled faults; post-hoc analysis
separates commands sent to stuck sites from actual persistent movement of
healthy sites.

This policy is now classified precisely as **open-loop Bernoulli one-pulse
Adam**, not ordinary continuous Adam retraining. For every minibatch it maps a
digital Adam command `c_j` independently to at most one physical pulse with
probability `min(1, |c_j|/dw_nominal)`. It has neither a sub-pulse residual nor
a continuous weight shadow. In addition, every selected IBM-OM write refreshes
the apparent state with write-noise standard deviation
`1.4113*0.0949 = 0.13393 q`, larger than the nominal `0.0949 q` pulse. The
algorithm can therefore erase a favorable P&V-conditioned apparent endpoint
while attempting to improve its hidden persistent state.

Two budgets are frozen before results are inspected. The matched control uses
the same one shuffled 16-example minibatch and `6e-5` effective-`q` learning
rate as the local STAR smoke. Its expected pulse count is only on the order of
tens, so it matches the example cohort and learning rate but not STAR's
sequential update mechanism; it is not a meaningful reparability test. The
primary control keeps the healthy predeployment HWA restricted to
that original 16-example stream, but constructs a separate repair loader and
runs exactly one shuffled epoch over all 55,000 post-split MNIST training
examples (`3,438` minibatches). It retains `[6e-5,6e-5]`, Adam
`[0.9,0.999]`, epsilon `1e-8`, all layers, the 64-command per-cell cap, and
fixed-final evaluation with no learning-rate tuning or checkpoint selection.

Both arms reproduced bit-exact healthy P0 and immediate faulted states on
assignment `87004` and endpoints `89402-89405`. Primary accuracy uses the
first 1,000 official test examples and reports healthy-to-fault damage,
fault-to-final recovery gain, and recovery fraction per endpoint. The full
epoch is promising only if mean gain is positive, at least three endpoints
improve, no endpoint loses more than two percentage points, and mean recovery
fraction reaches `0.25`. Four programming streams on one assignment are not
four independent arrays.

The artifact-verified but not yet reviewed study reached `95.775%` healthy,
`82.050%` immediately after the fault, and `89.175%` after the full open-loop
epoch. It recovered `7.125` percentage points, but narrowly missed its separate
strong threshold. The 16-example control recovered `0.000` points on average.
These numbers establish that the old writer can compensate part of the damage;
they do not establish the ceiling of normal BP.

Exact replay on endpoint `89402` exposed the failure mechanism. The open-loop
writer peaked at `93.0%` after 1,000 minibatches and fell to `89.8%` at the
forced end of the epoch, even as persistent accuracy initially improved. The
same labels, order, faulted state, and scaled crossbar gradients under bounded
continuous Adam reached `96.3%`. Training every coordinate without access to
the fault mask and then applying one closed-loop program-and-verify pass to the
same faulted plant reached `95.7%` apparent accuracy (`39,685/39,700` cells
accepted). This is a development diagnostic, not workflow evidence.

The corrected matched study is declared in
[`studies/mnist-ibm-om-crossbar-shadow-program-verify-retraining-debug-20260831-v1.json`](../studies/mnist-ibm-om-crossbar-shadow-program-verify-retraining-debug-20260831-v1.json).
It preserves the open-loop arm and adds
`supervised_ce_shadow_program_verify`:

```text
healthy programmed P0
-> identical published-companion fault transition
-> copy faulted apparent q into an FP32 digital shadow
-> one 55,000-example epoch of label-CE Adam on every shadow coordinate
-> clamp only to the healthy array support, without a defect map
-> one final same-array closed-loop apparent-verify programming pass
```

The ordinary Adam learning rate is `1e-3` in logical-weight coordinates and is
converted per layer to the corresponding `q` coordinate using the frozen
digital scales. Adam uses `[0.9,0.999]`, epsilon `1e-8`, and no checkpoint
selection. Programming occurs only after the fixed final epoch, uses the same
one-pulse controller and verify tolerance as deployment, and raises the cap to
`128` pulses per cell. The recovery learner receives labels but neither teacher
logits nor the fault mask; all shadow coordinates remain trainable, while only
the plant enforces stuck persistent cells.

Because endpoint `89402` was used to diagnose and choose this fixed correction,
it is development evidence. Endpoints `89403-89405` are the untouched primary
cohort. The correction passes only if their mean final apparent accuracy is at
least `95%`, all three improve from the immediate fault, and their mean remains
within two percentage points of their healthy mean. This is a privileged
off-array retrain-then-reprogram ceiling, not fully on-chip learning. Apparent
post-write accuracy remains primary; persistent-q, fresh-read, retention, and
independent-array durability remain separate requirements.

## Planned stochastic-pulse and Tiki-Taka recovery

The formal three-arm follow-up is frozen in
[`mnist-ibm-om-crossbar-stochastic-tiki-taka-recovery-20260901-v1.json`](../studies/mnist-ibm-om-crossbar-stochastic-tiki-taka-recovery-20260901-v1.json).
Its declared configs are the
[`direct BL31`](../examples/mnist_analog_relu/ibm_om_onchip_importance/supervised_ce_stochastic_pulse_sgd_full_epoch.json),
[`repaired-fast Tiki-Taka`](../examples/mnist_analog_relu/ibm_om_onchip_importance/supervised_ce_tiki_taka_v1_repaired_fast_full_epoch.json),
and
[`published-fast Tiki-Taka`](../examples/mnist_analog_relu/ibm_om_onchip_importance/supervised_ce_tiki_taka_v1_published_fast_full_epoch.json)
arms. The formal study is planned and has no held-out result yet.
It starts from the same continuous-HWA healthy P0 and the same chronological
published-companion fault as the writer-debug study, but removes the FP32
weight shadow, autograd, Adam moments, and final P&V. All recovery arms stream
the same `55,000` labels in `3,438` minibatches and perform manual full-network
cross-entropy BP, including a slow-array `W2` transpose MVM and the stored
digital-ReLU mask. The learner has no teacher or fault-map access.

Endpoint `89402` alone selected the fixed-final rates. The development screen
used the first 1,000 validation examples and never selected a mid-epoch state:

| Policy | Shared W1/W2 q-rate candidates | Fixed-final apparent accuracy | Frozen rate |
| --- | --- | --- | --- |
| Direct BL31 pulse-SGD | `1e-4`, `3e-4`, `1e-2` | `88.5%`, `72.0%`, `52.1%` | `[1e-4,1e-4]` |
| Tiki-Taka, repaired fast OM | `1e-3`, `3e-3`, `1e-2` | `87.2%`, `82.3%`, `78.0%` | `[1e-3,1e-3]` |

The Tiki-Taka screen was rerun after replacing support-aware fast-array
initialization with literal-zero, mask-blind P&V; no earlier initialization
result is eligible. The published-defect fast arm inherits `[1e-3,1e-3]`
without tuning. Its `87.1%` development canary is descriptive and cannot
change the schedule. Endpoint `89402` remains excluded from primary
statistics; endpoints `89403-89405` are the held-out cohort.

The direct arm converts the manual outer-product factors into
stochastic-compressed coincidence pulse trains and writes them directly to
the damaged slow array. The frozen writer uses `desired_bl=31`, `fixed_bl=true`,
update-BL management and update management, `um_grad_scale=1`, and no
cumulative pulse cap. The network forward and reverse MVMs use apparent `q`;
hidden persistent `q` controls subsequent physical writes.

The Tiki-Taka v1-style arms add an independently sampled fast accumulator
array `A` to the deployed slow array `C`. Every `A` cell receives the literal
`q=0` target through fault-blind one-pulse apparent-verify P&V before recovery;
the controller receives neither a fault mask nor a support clamp/oracle.
Unreachable stuck cells remain persistently immutable and may accept only
through apparent noise or exhaust. Acceptance, exhaustion, persistent/apparent
residuals, and commissioning pulses are reported separately from recovery
cost. Gradients write stochastic BL31 pulses
to `A`; every minibatch transfers one sequential apparent input-column slice
from each physical tile into `C`. With the frozen `512`-input tile limit, W1
has two `392x50` tiles and W2 one `50x10` tile. Each tile has its own cursor, so
`3,438` updates traverse each W1 tile `8.77` times and W2 `68.76` times; there
is no global 784-column W1 cursor. The remaining contract is
`gamma=0`, `fast_lr=1`, `transfer_every=1`, one read per transfer,
`transfer_lr=1` scaled by the selected layer q rate, column transfer, no reset,
and deterministic sequential selection.

Both Tiki-Taka arms use fast assignment `88004` and endpoint seeds
`98402-98405`, paired with slow endpoints `89402-89405`. The repaired-fast arm
counterfactually replaces only sampled fast corrupt cells. The published-fast
arm retains the preset's explicitly enabled `0.1348` corruption probability
and `0.01` corrupt range. Everything else is matched. The fast array is not
visible to inference and doubles programmable model-state storage from 39,700
to 79,400 `q` cells; this is simulator state accounting, not a fabricated
device-count claim.

Primary analysis reports healthy, immediate-fault, and fixed-final apparent
accuracy per held-out endpoint, with persistent-q accuracy as a separate
diagnostic. A protocol is meaningfully reparative only if all three held-out
endpoints improve, mean recovery fraction reaches `0.40`, and mean final
apparent accuracy reaches `88%`. A repaired-fast Tiki-Taka advantage over
direct requires at least `+1.0` percentage point in the held-out mean. The
published-fast effect is material when its absolute paired mean difference
from repaired-fast reaches `1.0` point; its sign is reported rather than
assumed.

Every result must reconcile bit-line statistics, slow/fast pulse commands,
physical plant counters, immutable-site commands, actual healthy persistent
movement, fast commissioning cost, transfer events and per-tile cursors, RNG
state, and checkpoint replay. These are on-chip-compatible update simulations,
not a physical peripheral implementation or native AIHWKit `TransferCompound`
parity result. They remain silent on fresh-read noise, retention, endurance,
line resistance, energy, and independent-array generalization.

## Planned STAR-inspired local-state-only recovery

This protocol adapts [*STAR: Astrocyte-Inspired State-Augmented Repair for
Supervised Memristive AI Hardware Systems*](https://arxiv.org/abs/2607.15415)
(Yusuf Ahmed Khan, Zhuangyu Han, and Abhronil Sengupta, 2026) as a pure-local
state-repair ablation for the standard crossbar. The paper's update is
equilibrium-propagation-specific: it adds the stored-state nudge to the task
nudge and retains a three-phase contrastive update. The feedforward crossbar
arm below intentionally tests the stored-state idea alone with direct local
outer-product pulses; it is STAR-inspired rather than a paper-faithful STAR
implementation.

The first STAR follow-up is predeclared in
[`studies/mnist-ibm-om-crossbar-star-local-recovery-smoke-20260831-v1.json`](../studies/mnist-ibm-om-crossbar-star-local-recovery-smoke-20260831-v1.json).
It is a planned smoke, not a completed result. The implementation config is
[`smoke_star_local_recovery.json`](../examples/mnist_analog_relu/ibm_om_onchip_importance/smoke_star_local_recovery.json);
neither file supports a recovery claim until the declared run and artifact
gates pass.

For the crossbar forward

```text
u = x W1,  h = ReLU(u),  z = h W2,
```

This STAR-inspired crossbar adaptation uses only class-indexed local state
errors:

```text
e_h = rho_h (h - mu_y^h) * 1[u > 0]
e_o = rho_o (z - mu_y^o)
g_q1 = alpha_1 x^T e_h
g_q2 = alpha_2 h^T e_o
```

Here `W_l = alpha_l q_l`. The label selects row `y` of the stored state
targets and has no other role. The two outer products drive stochastic
pulse-SGD directly. For cell `j` in layer `l`, one Bernoulli draw converts
the local gradient into the ternary physical command

```text
d_lj = -sign(g_qlj) with probability
p_lj = min(1, eta_l * abs(g_qlj) / dw_nominal), otherwise d_lj = 0.
```

The smoke fixes `eta_1 = eta_2 = 6e-5`; `dw_nominal` is the sampled
population's scalar nominal OM pulse scale and is recorded in the run receipt.
The probability normalization does not replace the sampled cell-specific
soft-bounds pulse response, and the per-cell command cap remains 64. There is
no cross-entropy or softmax error, no
`W2^T` reverse MVM, and no ordinary end-to-end BP. In particular, the input
layer update is independent of `W2`; the two rules are gradients of separate
local state-matching objectives with the inter-layer state stopped. The local
recovery path must not use a teacher, autograd, BPTT, Adam moments, FP32 shadow
weights, or a fault map. The teacher is used for source initialization,
predeployment continuous-HWA teacher-KL adaptation, and post-hoc diagnostics;
it is outside the post-fault local recovery update.

The experiment is chronological within each programmed endpoint. A matched
published companion population is sampled and sealed before the endpoint loop,
but none of its faults is applied before healthy target capture:

1. Program and evaluate one counterfactually repaired, healthy persistent
   source array.
2. Stream the fixed labeled calibration cohort through that exact plant using
   its apparent-`q` forward, and accumulate class means over exactly 1,024
   deterministic training-calibration examples for the 50 hidden ReLU states
   and 10 output logits.
3. Save and validate a versioned target artifact before any fault is applied.
4. After capture, verify that the sealed same-assignment `published` companion's
   mask and all non-defect identities match the repaired source, then replay its
   AIHWKit-sampled corrupt-device states on the same plant. The OM preset's
   upstream default corruption probability is zero; this companion explicitly
   enables the published `0.1348` probability while retaining the preset's
   `0.01` corrupt-device range. Each selected active state has collapsed bounds
   and zero upward/downward increments. Its persistent
   `q = a - r` is therefore stuck, while attempted writes continue to resample
   the configured apparent write-noise term with the same marginal equation as
   the AIHWKit device. AIHWKit samples corruption during device construction;
   this chronological experiment replays those sampled identities later as a
   post-deployment overlay and does not claim bitwise-native RNG-stream parity.
5. Evaluate the immediate post-fault state, then stream the ordered,
   hash-bound 16-example repair cohort and apply only the local outer-product
   pulse rule. This is the same deterministic 16-example subset used for
   predeployment continuous HWA, with its repair ordering recorded separately;
   under data seed 42 it is disjoint from the 1,024-example target-calibration
   cohort.
6. Evaluate the fixed final state without checkpoint selection or fresh-array
   remapping.

The target artifact is bound to the population, assignment, endpoint,
pre-fault apparent and persistent states, deployed checkpoint, state
representation, gains, calibration cohort hash, and per-class counts. Its
primary tensors have shapes `[10, 50]` and `[10, 10]`. FP16 storage therefore
uses exactly `10 * (50 + 10) * 2 = 1,200` tensor bytes, excluding version and
provenance metadata. The companion fault mask and stuck persistent values
remain analysis provenance: the plant enforces them, but the learner receives only inputs,
labels, current local states, target rows, and pulse-update settings.

Local state-error norms are evaluated on the same deterministic first 32
validation examples at healthy, immediate post-fault, and fixed-final states.
Accuracy decisions use the separately declared first 1,000 official MNIST test
examples. The calibration, repair, validation, and test cohorts have separate
roles and provenance and must not be conflated.

Even if this smoke recovers accuracy, it establishes only an algorithmic
local-state-repair control under the model-based AIHWKit preset. It does not
demonstrate physical target SRAM, local subtractors, ReLU-mask storage,
row/column pulse-coincidence circuitry, fresh-read stability, retention, or a
fabricated array. The apparent state remains post-write apparent `q`, not an
independent inference read. Report the healthy-to-fault damage and
fault-to-final recovery separately for all endpoints; do not infer that a
truly stuck cell itself moved or that STAR supersedes remapping and spares.

The stronger next target for genuinely local recovery is the DRN. Apply the
same healthy-capture--fault--repair chronology, make raw physical rail voltages
the primary state, and use centered equilibrium propagation driven only by
each layer's label-addressed raw-rail state-error cost. For ten classes, 100
hidden plus 20 output rail voltages require 1,200 values, or 2,400 FP16 bytes.
Logical rail differences must be a separately named ablation. The repository
now has the analytic local nudge primitive, but not a complete DRN recovery
runtime or study. A paper-faithful centered/three-phase EP arm that combines
the ordinary task nudge with the repair nudge must be a separately named
comparator. Both arms must exclude a teacher, autograd/BPTT, Adam state, and a
learner-visible fault map during recovery before they are described as
on-chip-compatible.

## Planned long-HWA exact-P&V follow-up

The next off-chip HWA diagnostic is frozen in
[`mnist-ibm-om-crossbar-long-hwa-exact-pv-20260901-v1.json`](../studies/mnist-ibm-om-crossbar-long-hwa-exact-pv-20260901-v1.json).
It addresses two separate questions that must not be conflated.

First, arrays A--D are independent hardware realizations. They share the same
logical teacher, OM preset, corruption policy, and controller, but assignments
`87004`, `87005`, `87006`, and `87007` independently sample cell parameters,
reference values, defect locations, and P&V streams. No-HWA accuracy is
therefore expected to be exchangeable across arrays, not bit-identical. The
repaired no-HWA source/fresh-target difference in the preceding diagnostic was
only about 0.31 percentage point; the published-defect spread was larger
because defect-map placement changes which logical weights are immutable.

Second, the earlier no-HWA/HWA contrast used different deployment handoffs.
No HWA requested the exact teacher master, while support-aware HWA first
clamped the target to each array's cell support. The new six-arm study gives
all policies the same fault-blind operation: P&V receives the exact unchanged
teacher or exact fixed-final HWA master. Bounds, stuck values, and eligibility
masks are not exposed to the request path; saturation and exhaustion are
measured outcomes.

The HWA budget is increased from 1,280 updates on 4,096 examples to ten full
55,000-example epochs, or 34,380 fixed-final Adam updates. Deterministic HWA
uses source-A support-clamped forwards. Stochastic HWA additionally samples

```text
q_apparent = q_persistent + 1.4113 * 0.0949 * N(0, 1)
```

once for every cell and minibatch with a replayable RNG and identity STE.
Initial and per-epoch clean realized-state metrics use exactly 1,000 validation
and 1,000 test examples; final deployment uses the matched P&V controller on A
and untouched B--D. This tests longer training and the OM apparent-write-noise
equation. It does not yet resample full device identities, defect masks, or
acceptance-conditioned programmed endpoints per minibatch; those are the next
HWA intervention if fixed-A noise training fails to transfer.

## Claim limits

The implementation does not model inference read noise, retention, ADC/DAC
quantization, line resistance, IR drop, peripheral energy, or an
absolute-conductance calibration. Consequently it supports normalized
accuracy, mismatch, and pulse-count comparisons only. It does not support
power, energy, endurance lifetime, fabricated-device-count, or raw measured
hardware claims.
