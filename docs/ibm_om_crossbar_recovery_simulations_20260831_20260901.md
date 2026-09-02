# IBM-OM standard-crossbar simulation report: 31 August–1 September 2026

- **Reporting window:** 2026-08-31 through 2026-09-01, Europe/Paris
- **Last updated:** 2026-09-02
- **Branch:** `codex/ibm-om-crossbar-digital-relu`
- **Architecture:** bias-free `784-50-10` analog crossbar--digital ReLU--analog crossbar
- **Device source:** AIHWKit 1.1.0 `ReRamArrayOMPresetDevice`
- **Evidence class:** model-based fitted-preset simulation, not raw measured traces or fabricated hardware

## Results at a glance

This is the canonical accuracy ledger for the report. All values are top-1
accuracy on the same first `1,000` official MNIST test examples. P&V rows are
the mean post-write **apparent-state** accuracy over four programming streams;
the one pre-P&V row is explicitly labelled as a deterministic diagnostic.
`Not run` means that the required experiment is missing; it does not mean zero
accuracy.

The two physical-array roles are:

- **Array A:** source device-population/HWA-model assignment, seed `87004`,
  programmed with endpoint seeds `89402-89405`.
- **Array B:** first fresh deployment assignment, seed `87005`, programmed
  with endpoint seeds `89502-89505`.

HWA adapts an FP32 master against Array A's sampled device model; it does not
train a persistent Array-A state. The resulting fixed-final master is then
mapped and programmed onto A or a fresh array.

| Stage | Model and operation | Array | No stuck cells: repaired control | Published stuck cells retained | Evidence state |
| --- | --- | --- | ---: | ---: | --- |
| Digital reference | FP32 ReLU teacher inference | none | **97.700%** | same teacher | cohort reference |
| No HWA | Teacher weights deployed directly with P&V | A (`87004`) | **95.850%** | **93.850%** | reviewed |
| After original HWA, before P&V | Five-minibatch fixed-final HWA master mapped to A support | A (`87004`) | **97.200%** | **88.300%** | reviewed deterministic diagnostic |
| After original HWA | Five-minibatch fixed-final HWA master deployed with P&V | A (`87004`) | **95.775%** | **85.650%** | reviewed |
| No HWA, fresh deployment | Teacher weights deployed directly with P&V | B (`87005`) | **96.675%** | **89.600%** | reviewed |
| After original HWA, fresh deployment | Five-minibatch HWA master deployed with P&V | B (`87005`) | **96.225%** | **80.200%** | reviewed |
| Long stochastic HWA on repaired A | Ten-epoch repaired-A master deployed with P&V | B (`87005`) | **95.575%** | **91.650%** | full artifacts; crossed study reviewed |
| Long stochastic HWA on published-defect A | Ten-epoch published-A master deployed with P&V | B (`87005`) | **92.525%** | **87.025%** | full artifacts; crossed study reviewed |
| On-chip training (a) | Train the same persistent B state with stochastic pulses | B (`87005`) | **Not run** | **Not run** | required matched experiment |
| Continuous control (b) | Train from the same B state with continuous updates | B (`87005`) | **Not run** | **Not run** | required matched experiment |

“No stuck cells” is the `counterfactual_repaired` control: the sampled defect
locations are recorded, but their device parameters are replaced by healthy
donors. “Published stuck cells” explicitly enables the OM rate `0.1348` and
retains the sampled collapsed, zero-step devices. In the original-HWA rows,
the two columns use separately trained masters because HWA was rerun under
each device policy. The two long-HWA rows explicitly cross each fixed master
with both deployment policies.

For context, the same frozen models were also deployed on two more fresh
arrays. These values again average four P&V writes per array:

| Fresh array | No-HWA direct P&V, repaired / stuck | After-HWA P&V, repaired / stuck |
| --- | ---: | ---: |
| B (`87005`) | 96.675% / 89.600% | 96.225% / 80.200% |
| C (`87006`) | 94.200% / 87.225% | 94.100% / 82.725% |
| D (`87007`) | 95.750% / 92.450% | 95.575% / 87.125% |
| **Pooled over B–D** | **95.542% / 89.758%** | **95.300% / 83.350%** |

### Long-HWA exact-master follow-up: accuracy and teacher KL

The later matched ten-epoch study measures both top-1 accuracy and

```text
D_KL(teacher || deployed crossbar),
```

on the same first `1,000` test examples. Lower KL is better; the teacher
compared with itself has KL `0`. Each array entry below is the mean of four
P&V programming streams, and the B--D entry pools all twelve fresh-array
writes. Values are written as `accuracy / KL`.

| Device policy | Training | A (`87004`) | B (`87005`) | C (`87006`) | D (`87007`) | B–D pooled |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Repaired | No HWA | 95.850% / 0.0505 | 96.675% / 0.0257 | 94.200% / 0.1462 | 95.750% / 0.0530 | **95.542% / 0.0750** |
| Repaired | Deterministic HWA | 96.300% / 0.0437 | 95.925% / 0.0703 | 93.625% / 0.1845 | 94.875% / 0.1011 | **94.808% / 0.1186** |
| Repaired | Stochastic HWA | 96.225% / 0.0687 | 95.575% / 0.0904 | 94.925% / 0.1136 | 95.250% / 0.0969 | **95.250% / 0.1003** |
| Published stuck cells | No HWA | 93.850% / 0.1216 | 89.600% / 0.3233 | 87.225% / 0.5974 | 92.450% / 0.1655 | **89.758% / 0.3621** |
| Published stuck cells | Deterministic HWA | 91.275% / 0.2223 | 70.250% / 1.3437 | 75.500% / 1.4245 | 79.975% / 0.7943 | **75.242% / 1.1875** |
| Published stuck cells | Stochastic HWA | 94.575% / 0.1290 | 87.025% / 0.4089 | 88.500% / 0.3546 | 89.050% / 0.3665 | **88.192% / 0.3767** |

These KL values expose an important distinction. On the fixed published
Array-A training proxy, stochastic HWA improves support-forward accuracy from
`87.3%` to `96.1%` and reduces teacher KL from `0.3162` to `0.0595`. After
actual P&V, however, KL is `0.1290` on A and `0.3767` on fresh B--D, slightly
worse than the no-HWA fresh-array KL of `0.3621`. Deterministic HWA makes the
same mismatch more extreme: its final Array-A proxy KL is only `0.0133`, but
its fresh-array P&V KL is `1.1875`.

Thus Adam successfully optimizes the model used inside HWA. The failure is in
generalization from that proxy to deployment: training holds Array A's device
identities, support, references, and stuck-cell map fixed, and stochastic HWA
resamples only unconditioned additive apparent-write noise. Actual deployment
uses a P&V-conditioned apparent endpoint and fresh arrays have different
device identities and defect locations. The current stochastic-HWA arm is
therefore equation-compatible with the IBM-OM write-noise term, but it is not
a sample from the complete P&V deployment distribution.

This follow-up study is artifact-verified and `ready_for_review`; it has not
yet been scientifically finalized. Its generated report is
[`results/mnist-ibm-om-crossbar-long-hwa-exact-pv-20260901-v1/analysis/report.md`](../results/mnist-ibm-om-crossbar-long-hwa-exact-pv-20260901-v1/analysis/report.md).

### Crossed HWA training and deployment defects

The missing off-diagonal deployments are now complete. The table below holds
the ten-epoch stochastic-HWA recipe fixed and crosses the device policy used
to train the FP32 master against Array A with the device policy present on
fresh arrays B--D. This is **not** training B--D; it is P&V deployment of a
frozen HWA master. Values are `apparent accuracy / teacher KL`.

| HWA device model on A | Fresh B--D device policy | B (`87005`) | C (`87006`) | D (`87007`) | B--D pooled |
| --- | --- | ---: | ---: | ---: | ---: |
| Repaired A | Repaired targets | 95.575% / 0.0904 | 94.925% / 0.1136 | 95.250% / 0.0969 | **95.250% / 0.1003** |
| Repaired A | Published-defect targets | 91.650% / 0.2252 | 90.775% / 0.2516 | 89.825% / 0.2970 | **90.750% / 0.2580** |
| Published-defect A | Repaired targets | 92.525% / 0.2390 | 92.250% / 0.2467 | 92.075% / 0.2356 | **92.283% / 0.2405** |
| Published-defect A | Published-defect targets | 87.025% / 0.4089 | 88.500% / 0.3546 | 89.050% / 0.3665 | **88.192% / 0.3767** |

On the same published-defect B--D populations, training HWA against repaired
A instead of published-defect A raises the array means by
`+4.625/+2.275/+0.775` percentage points and the pooled mean by `+2.558`
points. Teacher KL falls on every array, from `0.3767` to `0.2580` pooled.
This passes the study's predeclared numerical gate: at least `+1` pooled
point, lower KL, improvement on at least two arrays, and no array more than
`2` points worse.

The reverse cross reinforces the same pattern. On repaired B--D, the repaired-A
master reaches `95.250% / 0.1003`, while the published-defect-A master reaches
only `92.283% / 0.2405`. Thus exposing HWA to one fixed source defect map did
not produce reusable defect robustness; the result is consistent with fitting
Array A's particular missing connections.

The comparison with no HWA is more modest. On published B--D, repaired-A HWA
reaches `90.750% / 0.2580`, versus `89.758% / 0.3621` for direct teacher
deployment: `+0.992` pooled points and substantially lower KL. B and C improve
by `+2.050` and `+3.550` points, but D falls by `-2.625` points. HWA without
source defects therefore helps on average in this three-array smoke, but it
is not yet uniformly robust across target arrays. On repaired B--D, it remains
slightly below no HWA (`95.250%` versus `95.542%`) and has higher KL (`0.1003`
versus `0.0750`).

The pooled persistent-q diagnostic follows the same source-policy ordering:
`86.483%` for repaired-A to repaired targets, `77.658%` for repaired-A to
published targets, `83.800%` for published-A to repaired targets, and
`75.267%` for published-A to published targets. Apparent post-P&V accuracy
remains the primary inference metric.

This crossed study is artifact-verified and reviewed with outcome `supported`.
Its generated report is
[`results/mnist-ibm-om-crossbar-long-hwa-cross-defect-transfer-20260901-v1/analysis/report.md`](../results/mnist-ibm-om-crossbar-long-hwa-cross-defect-transfer-20260901-v1/analysis/report.md).

#### What “HWA on Array A” actually updated

No physical Array-A cell was pulse-programmed during HWA. Adam updated a
digital FP32 master. On every minibatch, however, each master coordinate was
clamped through the lower and upper support of the same corresponding Array-A
cell, and apparent write noise was added to that realized forward state. The
straight-through backward pass then updated the master coordinate as though
the clamp were the identity.

That distinction matters for corrupt cells. In the published-A run, all
`5,366` corrupt coordinates in the FP32 master changed (`0.0784 q` RMS), while
their realized forward values remained exactly at their immutable stuck
states. At deployment, P&V received the fixed-final **master**, not the
realized Array-A state. The training procedure therefore remained tied to A's
exact identity map while also exporting shadow updates that A itself could
never express.

The next HWA model should instead be characterized from Array A but not remain
identity-aligned with it: estimate or bootstrap the healthy-cell population,
then resample/permutate device parameters across logical coordinates during
training. A separate arm may resample independent defect masks if generic
defect augmentation is desired. Since deployment uses P&V-conditioned
apparent endpoints, a full endpoint-distribution surrogate should also be
tested separately from the current additive-noise-only forward.

## Plain-language conclusion

- The digital ReLU teacher scores `97.700%` on this cohort.
- Programming its weights directly onto repaired arrays costs little: Array A
  reaches `95.850%` and fresh Array B reaches `96.675%`.
- HWA is also accurate on repaired arrays, but it does not improve top-1 over
  direct teacher deployment in this small smoke: `95.775%` on A and `96.225%`
  on B.
- Retaining published stuck devices changes the picture. Direct deployment on
  B falls to `89.600%`, while the separately trained HWA path falls to
  `80.200%`. Pooled HWA accuracy across B--D falls from `95.300%` to
  `83.350%`.
- With ten full stochastic-HWA epochs, training against repaired A and then
  deploying onto published-defect B--D reaches `90.750%`, compared with
  `88.192%` when HWA uses published-defect A and `89.758%` with no HWA. The
  paired result supports an Array-A defect-map fitting mechanism, while the
  heterogeneous comparison with no HWA shows that this is not yet universal
  target-array robustness.
- The two values needed to answer the on-chip-training question on Array B do
  not exist yet. Existing recovery numbers must not be inserted into those
  cells because they trained Array A after a different chronological fault
  transition.

This makes the present conclusion narrower and clearer: HWA works well for the
repaired-array control, while fitting one fixed source defect map is actively
harmful to fresh-array transfer. Training against repaired A helps on
published-defect targets on average, but is not uniformly better than no HWA.
Whether training the already deployed Array B can recover the remaining loss
remains an open experiment.

## Existing recovery evidence is from Array A, not Array B

All current recovery arms start from the same Array-A held-out mean of
`95.900%`, apply a post-deployment published-companion fault that reduces it
to `83.100%`, and then train assignment `87004`. The `55,000`-example arms use
the same example order and `3,438` update steps, but their optimizers, rates,
writers, and final programming operations are not fully matched.

| Existing Array-A recovery | Final apparent | Final persistent | What it establishes |
| --- | ---: | ---: | --- |
| Direct stochastic pulse-SGD | 79.967% | 56.633% | on-chip-compatible writer made apparent accuracy worse |
| Tiki-Taka v1, repaired fast array | 86.267% | 55.433% | better than direct pulses, but incomplete recovery |
| Tiki-Taka v1, published-defect fast array | 84.933% | 55.100% | fast-array defects reduce the mean benefit |
| Open-loop pulse-Adam | 88.967% | 69.133% | digital Adam plus coarse physical pulses; not fully on-chip |
| Continuous FP32 shadow Adam target | 97.233% | not a physical state | privileged continuous optimization ceiling |
| Shadow Adam followed by final P&V | 95.900% | 63.233% | privileged retrain-then-reprogram ceiling |

These values are useful writer diagnostics. They are not an apples-to-apples
comparison of stochastic and continuous training on Array B, and the final
`95.900%` remains a sticky post-write apparent endpoint rather than a fresh
read.

## Standardized Array-B comparison to run

In this report, **on-chip training** will mean: program Array B once, save its
exact persistent and apparent P&V state, and update that same physical state
without remapping, resetting, or replacing the array.

The term describes the chronological same-array lifecycle. Of the two primary
comparators below, only stochastic-pulse training is physically
on-chip-compatible; continuous training is the ideal numerical control.

The minimum matched comparison should use Array B seed `87005`, reserve
endpoint `89502` for development, and report endpoints `89503-89505` as held
out. Its starting point, or **B-P0**, is the exact after-HWA, post-P&V state
from the corresponding endpoint: the published-aware HWA master on published
B for the primary realistic run, repeated with the repaired HWA master on
repaired B as the control. Do not substitute the direct teacher deployment or
remap the HWA master between recovery arms. One immutable B-P0 checkpoint per
endpoint must be forked into:

1. **Stochastic-pulse training:** manual label-CE BP followed by BL31
   stochastic physical pulse updates on B.
2. **Continuous-update control:** the same manual gradients, examples, order,
   number of steps, and learning rate, but ideal continuous-q SGD. Bounds and
   stuck-cell immutability remain enforced by the plant without exposing the
   fault mask to the learner. This is an ideal control, not physical on-chip
   learning.

Use the same `55,000` examples, batch size `16`, `3,438` steps, one fixed-final
epoch, seed `42`, and ordered cohort hash
`e6cd5aaa8d7c33a0aa5f78ed3c47e7d3c1017e2ad5c60794765baef4ef9590d7`.
Freeze `q_lr=[1e-4,1e-4]` for both layers and both primary arms, inheriting the
choice made on the separate Array-A development endpoint; do not tune it on
held-out B endpoints. Do not add final P&V to only one member of this primary
pair. Conventional FP32 Adam plus final P&V may remain as a separately named
privileged ceiling, but it is not the continuous member of the strict writer
comparison. Tiki-Taka is likewise a separate third algorithm, not a substitute
for the direct pulse arm.

Every result should appear in the top ledger and report:

- B accuracy immediately after initial P&V and after training;
- apparent, persistent, and independently resampled fresh-read accuracy;
- the exact accuracy gain and fraction of deployment loss recovered;
- stuck-state immutability and actual movement of programmable cells; and
- pulse count, saturation, and any final target-reprogramming cost.

The frozen-HWA deployment `2x2`--HWA against repaired or published-defect A,
then P&V on repaired or published-defect B--D--is now complete above. The
remaining matched `2x2` concerns **training the deployed B state itself**: fork
the exact same repaired or published B-P0 checkpoint into stochastic-pulse and
continuous-update recovery arms. That experiment is still required to isolate
the writer from the B starting state.

The current runtime cannot execute this comparison directly: it forbids
recovery together with fresh-array transfer and does not persist the target
array's complete P&V state. The implementation prerequisite is therefore a
hash-bound, replayable Array-B deployment bundle containing its population,
persistent state, apparent state, pulse counters, RNG state, and endpoint
receipt.

The remainder of this document is the detailed audit trail. The accuracy
ledger above remains the single source for the headline values.

## Definitions and measurement contract

The standard comparator contains `39,700` logical weights split into two
`392x50` input tiles and one `50x10` output tile. Each effective device weight
is

```text
q = a - r,
```

where `a` is the programmable active state and `r` is the sampled fixed
intrinsic reference. The network forward uses the post-write **apparent**
state. The hidden **persistent** state determines subsequent pulse updates and
is reported separately as a robustness diagnostic.

The upstream AIHWKit preset defaults to zero corrupt devices. The defect arms
explicitly enable the published OM rate `0.1348` with range `0.01`. A selected
active cell has collapsed bounds and zero upward and downward increments; its
persistent `q=a-r` cannot move. Write attempts can still resample its apparent
noise. Reference subtraction therefore centers the effective state but does
not restore programmability.

The same-array recovery studies use source assignment `87004` and programming
streams `89402-89405`. These are four stochastic writes on one sampled array,
not four independent arrays. Accuracy is measured on the first `1,000`
official MNIST test examples. Endpoint `89402` was later used for writer and
learning-rate development; the Tiki-Taka and shadow-P&V studies consequently
use `89403-89405` as their primary held-out endpoint cohort.

“On-chip-compatible” means that the numerical update can be expressed using
local pulse coincidences and array MVMs. It still requires labeled examples,
digital softmax/error logic, ReLU-mask storage, and a transpose MVM. It is not
a fabricated peripheral implementation. The custom stochastic backend matches
AIHWKit 1.1.0 equations and distributions but is not bitwise-native
`TransferCompound` execution.

## Detailed record: 31 August 2026

### 1. Fresh-array transfer with published defective cells

The published-defect study was paired with the repaired-array transfer study
that ran on 30 August and was reviewed on 31 August. Source assignment `87004`
was redeployed to independently sampled assignments `87005-87007`, with four
programming streams per assignment. Each fresh array rebuilt its own support
and codebook from the frozen off-chip master.

| Frozen model | Repaired source | Repaired pooled targets | Published source | Published pooled targets | Target defect penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct teacher | 95.850% | 95.542% | 93.850% | 89.758% | -5.783 pp |
| Continuous HWA | 95.775% | 95.300% | 85.650% | 83.350% | -11.950 pp |
| Deterministic QAT | 88.650% | 83.267% | 76.000% | 76.300% | -6.967 pp |

The HWA target means changed from `96.225/94.100/95.575%` with repaired cells
to `80.200/82.725/87.125%` with published defects. The source and target arrays
retained `5,366`, `5,480`, `5,446`, and `5,284` stuck cells out of `39,700`, or
`13.31--13.80%`. No published-defect arm passed the original two-point
fresh-array portability gate.

The pooled persistent diagnostics were much lower than the apparent results:
continuous HWA measured `66.783%` with repaired cells and `55.358%` with
published cells. The HWA defect comparison also reran the adaptation path
under each corruption policy; it is not deployment of one bit-identical
adapted master. The reviewed conclusion is limited to this matched smoke:
healthy-cell variation and stochastic programming were tolerable, whereas
retained stuck cells and relocated defect masks were not.

### 2. STAR-inspired local-state-only recovery canary

A chronological canary captured class-conditional means of the healthy
hidden ReLU states and output logits, applied the matched published-companion
fault, and then used only label-indexed local state errors to select pulse-SGD
updates. The recovery kernel used no teacher errors, cross-entropy BP,
`W2^T`, autograd, Adam state, FP32 weight shadow, or learner-visible fault map.

Across all four programming streams:

```text
healthy apparent       95.775%
immediate fault        82.050%
fixed-final STAR       81.600%
recovery gain          -0.450 percentage points
persistent accuracy    48.075% -> 48.700%
mean pulse commands    27.25
```

The local state objective decreased slightly, but apparent test accuracy fell
on all four streams. This was an ad hoc technical canary under `/tmp`, not a
prepared workflow run. The tracked STAR study therefore remains unexecuted as
formal evidence. Its 16-example update budget is also too small to distinguish
a weak local objective from an insufficient number of writes.

### 3. Ordinary supervised BP with open-loop pulse-Adam

The ordinary retraining control applied ground-truth cross-entropy BP through
the apparent network, propagated the output error through `W2^T`, kept digital
FP32 Adam moments, and converted each minibatch command independently into at
most one Bernoulli physical pulse per cell. The teacher and fault mask were not
available to the update kernel, but autograd and Adam state make this a
privileged digital control rather than a fully on-chip implementation.

| Budget | Faulted apparent | Final apparent | Gain | Final persistent | Mean pulse commands |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 examples, one minibatch | 82.050% | 82.050% | 0.000 pp | 48.050% | 23.25 |
| 55,000 examples, 3,438 minibatches | 82.050% | 89.175% | +7.125 pp | 67.100% | 16,978.25 |

The full epoch improved all four streams and recovered a mean `49.889%` of
their fault damage. It passed the study's promising-repair gate but narrowly
missed its stronger gate: final apparent accuracy remained below `90%`, and
the mean endpoint recovery fraction was `0.498893`, just below `0.5`.

### 4. Writer diagnosis and shadow-Adam plus P&V ceiling

Exact replay on development endpoint `89402` showed that the open-loop pulse
writer, rather than the BP gradient, was the main limitation. Its apparent
accuracy peaked near `93.0%` after `1,000` minibatches and then fell to `89.8%`
at the forced epoch end. Bounded continuous Adam on the same faulted state,
labels, order, and gradients reached approximately `96.3--96.5%`. Increasing
the open-loop pulse rate made apparent accuracy worse while often improving
persistent accuracy, exposing a coarse-pulse/write-noise tradeoff.

The formal corrected control trained every coordinate in an FP32 shadow at a
logical learning rate of `1e-3`, then sent the fixed-final target through one
same-array closed-loop P&V pass. Endpoint `89402` was development-only. On
held-out endpoints `89403-89405`:

```text
healthy apparent                 95.900%
immediate fault                  83.100%
continuous shadow target         97.233%
final post-P&V apparent           95.900%
final persistent diagnostic      63.233%
```

All three held-out endpoints improved, with exact final apparent accuracies of
`96.2/95.5/96.0%`. The result demonstrates a compensation ceiling, not
on-chip Adam: it requires a full FP32 weight shadow, autograd, digital Adam,
and final closed-loop verify. Its reported `95.900%` is also the sticky
post-write apparent endpoint, not a fresh inference read.

## Detailed record: 1 September 2026

### 5. On-chip-compatible stochastic BP and Tiki-Taka v1

The next study removed autograd, Adam moments, FP32 weights, and final P&V from
the recovery path. It used manual label-supervised CE errors, the digital ReLU
mask, an apparent-state `W2^T` MVM, and stochastic-compressed coincidence
pulses. All arms consumed the same `55,000` examples in `3,438` steps.

Development endpoint `89402` selected fixed-final rates before held-out
execution. Direct q-rate candidates `1e-4/3e-4/1e-2` produced
`88.5/72.0/52.1%`, selecting `1e-4`. After correcting fast-array commissioning
to literal all-cell `q=0` with mask-blind P&V, repaired-fast Tiki-Taka rates
`1e-3/3e-3/1e-2` produced `87.2/82.3/78.0%`, selecting `1e-3`. The
published-fast arm inherited `1e-3`; it was not independently tuned. Earlier
support-clamped commissioning screens were invalidated because they leaked
per-cell support information and are excluded from the evidence.

Primary apparent results on held-out endpoints were:

| Endpoint | Healthy | Faulted | Direct stochastic SGD | TT repaired fast | TT published fast |
| --- | ---: | ---: | ---: | ---: | ---: |
| `89403` | 95.9% | 84.1% | 76.1% | 85.8% | 85.0% |
| `89404` | 95.9% | 85.0% | 84.5% | 89.0% | 83.3% |
| `89405` | 95.9% | 80.2% | 79.3% | 84.0% | 86.5% |
| **Mean** | **95.900%** | **83.100%** | **79.967%** | **86.267%** | **84.933%** |

| Protocol | Mean apparent gain | Mean recovery fraction | Endpoints improved | Final persistent | Declared meaningful-repair gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Direct stochastic SGD | -3.133 pp | -26.039% | 0/3 | 56.633% | fail |
| TT repaired fast | +3.167 pp | 25.103% | 3/3 | 55.433% | fail |
| TT published fast | +1.833 pp | 10.719% | 2/3 | 55.100% | fail |

Repaired-fast Tiki-Taka beat direct pulse-SGD by `6.300` points, passing its
separate advantage gate. It nevertheless failed the composite repair gate
because `86.267% < 88%` and `25.103% < 40%`. Published fast-array defects had
a material mean effect of `-1.333` points relative to repaired fast, but the
paired endpoint changes `-0.8/-5.7/+2.5` points were heterogeneous.

The physical cost was substantial. Direct recovery issued a held-out mean of
`1,710.7` slow-array pulses. Repaired-fast Tiki-Taka issued about `1,177` slow
pulses plus `14.140` million fast-array pulses; published-fast Tiki-Taka used
about `1,111` slow plus `14.725` million fast pulses. The fast array adds a
second `39,700`-state accumulator. The configured bit-line length `31` was a
maximum: realized direct and transfer updates used length `1`, while fast
updates reached maxima of roughly `15--20`.

The slow array retained `5,366` immutable cells in every arm. The published
fast array retained `5,332/39,700` defects and sent about `1.961` million pulse
commands to those immutable cells without moving their persistent state.
Persistent adaptation therefore occurred only in programmable cells; no method
moved a stuck persistent state. The apparent endpoint may additionally include
write-noise resampling caused by commands to immutable cells.

## Provisional evidence-only mechanistic synthesis

1. **The intrinsic reference is already used.** Every effective state is
   `q=a-r`; failure is not caused by omitting `r`. A collapsed active cell with
   zero increments remains stuck after subtraction.
2. **Stuck-cell mismatch produced the largest observed transfer loss.** In the
   matched smoke, continuous HWA remained near `95%` across repaired fresh
   arrays but fell by about `12` points when published defects were retained on
   the targets.
3. **Training HWA on a fixed defect map hurts transfer.** Repaired-A HWA beats
   published-defect-A HWA on both repaired and published B--D. On published
   targets the paired gain is `+2.558` points with lower KL on every array.
   Relative to no HWA, however, the gain is only `+0.992` pooled points and one
   target array is worse, so the benefit is not yet uniformly robust.
4. **The simulator admits a post-write apparent compensation ceiling.**
   Continuous shadow Adam found a high-accuracy target without a fault map.
   Persistent changes during final P&V occurred only in programmable cells,
   while the apparent endpoint may additionally include resampled write noise
   at immutable cells. This does not prove a durable fresh-read state.
5. **The evidence implicates the tested physical-pulse writers.** Exact replay
   separates their open-loop Bernoulli rounding, coarse OM pulses, and
   write-noise resampling from the successful continuous optimizer. Direct
   stochastic SGD can improve persistent state while degrading the apparent
   forward endpoint.
6. **Tiki-Taka improves the update mechanism but not enough.** Its fast array
   integrates many more pulses and clearly beats direct stochastic SGD, yet it
   recovers only one quarter of the apparent damage with repaired fast cells.
7. **Apparent and persistent state must remain separate.** The apparent state
   is the modeled AIHWKit network state, but the large final gaps and lack of a
   fresh read mean that post-write apparent recovery is not yet durability
   evidence.

## Evidence lifecycle and artifacts

| Study | Native execution | Lifecycle | Evidence location |
| --- | --- | --- | --- |
| Repaired fresh-array comparator | 2026-08-30; reviewed 2026-08-31 | reviewed | [plan](../studies/mnist-ibm-om-crossbar-fresh-array-smoke-20260830-v1.json), [workflow report](../results/mnist-ibm-om-crossbar-fresh-array-smoke-20260830-v1/analysis/report.md) |
| Published-defect fresh-array transfer | 2026-08-31 | reviewed | [plan](../studies/mnist-ibm-om-crossbar-fresh-array-published-defects-smoke-20260831-v1.json), [workflow report](../results/mnist-ibm-om-crossbar-fresh-array-published-defects-smoke-20260831-v1/analysis/report.md) |
| STAR-inspired local recovery | 2026-08-31 canary only | development-only; formal study unexecuted | [plan](../studies/mnist-ibm-om-crossbar-star-local-recovery-smoke-20260831-v1.json); latest canary under `/tmp/ibm-om-star-aihwkit-corrupt-smoke-final3-20260831/` |
| Supervised open-loop pulse retraining | 2026-08-31 | full artifacts; ready for review | [plan](../studies/mnist-ibm-om-crossbar-supervised-retraining-recovery-20260831-v1.json), [workflow report](../results/mnist-ibm-om-crossbar-supervised-retraining-recovery-20260831-v1/analysis/report.md) |
| Shadow-Adam/P&V writer debug | 2026-08-31 | full artifacts; ready for review; one failed attempt retained | [plan](../studies/mnist-ibm-om-crossbar-shadow-program-verify-retraining-debug-20260831-v1.json), [workflow report](../results/mnist-ibm-om-crossbar-shadow-program-verify-retraining-debug-20260831-v1/analysis/report.md) |
| Stochastic SGD and Tiki-Taka v1 | 2026-09-01 | full artifacts; ready for review | [plan](../studies/mnist-ibm-om-crossbar-stochastic-tiki-taka-recovery-20260901-v1.json), [workflow report](../results/mnist-ibm-om-crossbar-stochastic-tiki-taka-recovery-20260901-v1/analysis/report.md) |
| Long-HWA exact-P&V comparison | 2026-09-01 | full artifacts; ready for review | [plan](../studies/mnist-ibm-om-crossbar-long-hwa-exact-pv-20260901-v1.json), [workflow report](../results/mnist-ibm-om-crossbar-long-hwa-exact-pv-20260901-v1/analysis/report.md) |
| Crossed HWA/target defect policies | 2026-09-01; reviewed 2026-09-02 | reviewed; supported | [plan](../studies/mnist-ibm-om-crossbar-long-hwa-cross-defect-transfer-20260901-v1.json), [workflow report](../results/mnist-ibm-om-crossbar-long-hwa-cross-defect-transfer-20260901-v1/analysis/report.md) |

Artifact verification reports exact declared coverage for all workflow studies.
No simulations are currently active. Studies marked `ready for review` do not
yet have a human `review.json`, `final.json`, or experimental-manifest entry.
Their numerical gates are reported above, but their schema-level scientific
outcomes remain pending human review.

The earlier 1 September formal runs bind source commit
`f8733a0ed2a7afcf25acd79f2f10b131d903ce64`. The direct run was clean. The two
Tiki-Taka manifests recorded an identical dirty hash caused only by the
watchdog-generated `docs/current_simulations.md` update; executable, config,
study, and test sources were unchanged and all artifact checks passed. Both
crossed-HWA arms bind clean source commit
`d542bbeadb972ecb64711796e9d15df611ac3ca0`.

## Limitations and open decisions

- All device evidence comes from the fitted AIHWKit OM preset, not raw device
  traces or fabricated crossbars.
- Recovery used one slow-array assignment and repeated programming streams;
  it does not establish independent-array or independent-fault-mask
  generalization.
- Accuracy uses only the first `1,000` test examples. The original reviewed
  HWA comparator used only 16 examples for five one-minibatch epochs; the long
  stochastic-HWA studies used ten complete `55,000`-example epochs.
- There is no independent fresh-read, retention, drift, inference-noise,
  endurance, line-resistance, ADC/DAC, peripheral-energy, or absolute-
  conductance validation.
- The successful shadow path is a privileged off-array ceiling. The on-chip-
  compatible paths require supervised labels and digital error peripherals.
- The current studies do not include the matched DRN recovery arm.

Before the three unreviewed studies can be finalized into the experimental
manifest, a human review must choose their scientific outcome and next test.
The unresolved choices exposed by these data are: fresh-read/retention testing
of the repaired endpoints; a full-budget local-state or TTv2/chopped-Tiki-Taka
recovery; and replication across independent slow arrays and fault masks with
remapping/spare-cell controls.
