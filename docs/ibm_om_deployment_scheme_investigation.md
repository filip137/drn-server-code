# IBM OM deployment-scheme investigation

- Status: homogeneous ideal-mapping, shared-calibration bounded-codebook,
  exact-lower-bound, and corrected per-cell RESET-mean four-delta screens
  complete; the three-accuracy ladder is frozen and the shared-zero ideal
  initialization control is the active next stage; no new HWA or stochastic
  deployment P&V experiment has been launched
- Date: 2026-08-28
- Device source: AIHWKit 1.1.0 `ReRamArrayOMPresetDevice`
- Evidence class: normalized hardware-derived fitted model, not raw measured
  conductance traces and not an absolute conductance calibration

## Standard headline accuracies

Use the following three accuracy names for this investigation. Do not call
all three "baselines": `B` is reserved for the physical conductance baseline
of a cell or device group.

| Order | Metric | Question answered | Current status |
| --- | --- | --- | --- |
| 1 | `ideal_mapped_init_accuracy` | How accurate is the frozen logical model immediately after deterministic mapping into one hardware instance's physical bounds and discrete codebook? | active gate |
| 2 | `same_hardware_bptt_accuracy` | How much can BPTT recover when it adapts the exact mapped state against the same hardware instance? | blocked until metric 1 passes and the physical HWA writer is available |
| 3 | `cross_hardware_deployment_accuracy` | How much of the source-hardware BPTT result survives when its frozen logical checkpoint is mapped onto untouched hardware? | blocked until metrics 1 and 2 are valid |

All three metrics obey one physical-conductance invariant. Each branch enters
the circuit as its full nonnegative conductance

```text
G_i = B_i + d_i.
```

No mapper, training modifier, or evaluator may remove `B_i` before forming
transfer or loading. For a four-device rail quad,

```text
C = (B_pp+d_pp) - (B_pm+d_pm) - (B_mp+d_mp) + (B_mm+d_mm),
L = (B_pp+d_pp) + (B_pm+d_pm) + (B_mp+d_mp) + (B_mm+d_mm).
```

For an eight-device active/reference edge,

```text
D = (B_a+d_a) - (B_r+d_r),
S = (B_a+d_a) + (B_r+d_r).
```

Equal realized baselines may cancel algebraically from `C` or `D`; that is a
measured property of the full conductances, not an implementation shortcut.
The same baselines always remain in `L` or `S`.

### 1. Ideal bounded-codebook initialization accuracy

The canonical definition ID for the first metric is
`ibm_om.ideal_bounded_standard4delta_init.v1`.
`ideal_mapped_init_accuracy(H)` is the top-1 test accuracy of one explicitly
named pre-BPTT FP32 logical checkpoint after deterministic mapping onto one
frozen OM hardware instance `H`. A hardware instance comprises the sampled
identity population, physical-cell bindings, per-cell bounds and fitted
references, corruption/reassignment policy, baseline-commissioning receipt,
topology, and codebook.

The primary codebook is the declared uniform standard grid in
`x=(a+1)/2`, with adjacent targets exactly
`4 * (nominal_dw_min/2)` apart and with the frozen rounding, tie,
zero-capacity, and unsupported-group rules. The exact selected codebook state
is applied without stochastic target-write or inference-read noise. If noisy
RESET/read observations were used to commission `B`, their one frozen receipt
remains part of `H`: ideal evaluation removes later deployment noise, not the
already committed commissioning error.

The following are separate diagnostics and must not be reported under the
headline name:

- `continuous_envelope_accuracy`, which omits nearest-level rounding inside
  the same bounds and capacity envelope;
- an exact-bound or exact-baseline oracle;
- the native state-dependent 0--128-pulse codebook; and
- any historical effective-coordinate map that presents `(a-r+1)/2` to the
  DRN as though it were a physical conductance.

Each scheme searches the same predeclared layer-fraction grid and objective
on the development assignment only. Its selected fractions and positive KL
gain are then frozen before held-out evaluation. Test labels and held-out
hardware cannot affect baseline commissioning, mapping, codebook,
reassignment, calibration, or checkpoint selection. No optimizer update,
HWA modifier, endpoint sampler, P&V controller, retention/drift, or on-chip
recovery is active.

Report exact `correct/examples` per untouched assignment, followed by the
arithmetic mean and full assignment range. Development accuracy is
calibration evidence, not the headline result. The existing 90% gate applies
to the held-out mean, while the full range must remain visible.

The metric is invalid, rather than merely low, if any required source,
population, binding, commissioning, codebook, or calibration hash is absent
or mismatched; if an unsupported group is handled outside its predeclared
policy; or if any target is nonfinite, negative, outside its assigned bounds,
or violates `G_i=B_i+d_i` beyond the declared numerical tolerance. The
assignment artifact must therefore record or hash at least:

- the definition, scheme, topology, reference policy, and hardware-instance
  IDs;
- the source checkpoint, teacher, model, solver, dataset/split, runtime, and
  device-population identities and hashes;
- the baseline estimator, grouping rule, seed, sample count, noise policy,
  receipt, and realized per-cell `B_i` tensor;
- code indices, capacities, rounding/tie policy, requested and exact targets,
  zero-capacity groups, and unsupported/reassigned/failure counts;
- per-branch `B_i`, `d_i`, and `G_i` hashes, their maximum decomposition
  residual, finite/nonnegative/bounds checks, and derived `C/L` or `D/S`
  hashes and loading statistics; and
- development calibration receipt, exact held-out correct count, prediction
  hash, teacher agreement, KL, voltage diagnostics, and explicit zero counts
  for optimizer updates and every excluded deployment noise/process.

The completed per-cell RESET-mean v2 screen remains relevant evidence, but it
predates this definition and cannot be relabelled as its canonical output: it
does not save an explicit checked `B_i`, `d_i`, `G_i=B_i+d_i` decomposition,
one hardware-instance ID, an exact correct count, or a fail-closed validity
record. Its independent origins also do not implement the shared-zero
candidate selected below.

### 2. Same-hardware BPTT accuracy

`same_hardware_bptt_accuracy(H_s)` starts from the exact artifact-identified
state used for `ideal_mapped_init_accuracy(H_s)`, performs BPTT against that
same hardware instance, and evaluates the selected checkpoint
deterministically on `H_s`. Identities, bindings, baselines, bounds, codebook,
topology, and calibration remain fixed. The metric records the initialization,
optimizer, training, and checkpoint hashes. It measures adaptation to `H_s`,
not hardware transfer.

This metric cannot use the current reusable IBM OM modifier unchanged. That
legacy path writes the effective coordinate `(a-r+1)/2` into a DRN
conductance tensor and therefore omits the separate physical-reference
loading. A full-physical-`G` BPTT writer and parity tests are prerequisites.

### 3. Cross-hardware deployment accuracy

`cross_hardware_deployment_accuracy(H_s -> H_t)` freezes the BPTT-selected
logical/master checkpoint and source-selected calibration from `H_s`, maps it
without optimizer updates or target-specific calibration onto a disjoint,
untouched hardware instance `H_t`, and evaluates it deterministically. The
target uses its own commissioned baselines, bounds, identities, bindings, and
codebook under the same frozen scheme rule. Absolute source conductances are
never copied to `H_t`.

Compute `ideal_mapped_init_accuracy(H_t)` on the same target instances so the
three useful differences are identifiable:

```text
same-hardware recovery = same_hardware_bptt_accuracy(H_s)
                         - ideal_mapped_init_accuracy(H_s)

cross-hardware drop    = cross_hardware_deployment_accuracy(H_s -> H_t)
                         - same_hardware_bptt_accuracy(H_s)

held-out BPTT benefit  = cross_hardware_deployment_accuracy(H_s -> H_t)
                         - ideal_mapped_init_accuracy(H_t).
```

Any pulse-programmed, fresh-read, retention, or noisy-inference accuracy is a
later separately suffixed metric. It must not be folded into these three
deterministic structural accuracies.

## Three physical mapping decisions

Represent every physical branch conductance as

```text
G_i = B_i + n_i h,
```

where the three mapping decisions are:

1. the physical baseline `B_i` and its commissioning/grouping rule;
2. the adjacent-level conductance spacing `h`; and
3. the allowed integer index set for `n_i`, which determines the usable
   level count.

Change these decisions in separate stages. They are experimentally
separable, but not mathematically independent under bounded devices. For a
one-sided grid,

```text
M_i(B_i,h) = floor((u_i-B_i)/h)
```

limits the realizable positive level count. A baseline experiment must
therefore freeze the spacing and level-count rule while reporting any
mechanical capacity change caused by moving the baseline. The frozen
four-delta grid is a probe for this stage, not a final selection of spacing
or level count.

The first active decision is the baseline. Here, "weight zero is zero" means
zero signed transfer, not zero physical conductance or zero circuit loading.
For a four-device rail quad, the hard zero condition is

```text
C0 = B_pp - B_pm - B_mp + B_mm = 0.
```

One exactly realized `B_Q` shared by all four cells is a sufficient, robust
construction. For an eight-device active/reference edge, the robust local
condition is

```text
D0,e = B_a,e - B_r,e = 0,
```

so one exactly realized baseline shared by each active/reference pair is
sufficient. In both cases the full zero-state baseline remains in KCL
loading:

```text
L0,Q = B_pp + B_pm + B_mp + B_mm,
S0,e = B_a,e + B_r,e.
```

Consequently, zero transfer must be verified from the stored full
conductances; it must never be implemented by dropping `B` from the circuit.
Unless all baselines are physically zero, a zero logical weight still has a
nonzero electrical loading footprint.

### Baseline-first decision experiment

Stage 0E changes only the baseline rule. Its primary interventions are:

- four devices/no fixed `r`: replace four independent `B_i` values by one
  bound-feasible commissioned `B_Q` shared by the rail quad; and
- eight devices/no fixed `r`: replace independent active/reference origins
  by one bound-feasible `B_e` shared within each pair.

The fitted intrinsic `r_i` is observed rather than selected and is therefore
not part of this first baseline-choice intervention. A later fixed-`r` study
may test identity balancing or active-state compensation, but must retain the
sampled `r_i`.

Before accuracy is inspected, every assignment must pass exact zero-transfer
checks at `n=0`, full `G=B+n h` bounds and KCL parity checks, and complete
functional coverage or its frozen reassignment/failure policy. Report
zero-state loading, common-window feasibility, positive headroom, and the
mechanically resulting capacity.

The headline remains `ideal_mapped_init_accuracy`. Report the continuous
bounded-envelope diagnostic beside the frozen four-delta probe. If both
improve, baseline cancellation/loading was limiting. If only the continuous
result improves, spacing or level count is the next limitation. Do not select
a new spacing or level count from Stage 0E.

## Decision at this checkpoint

The current raw-p90 result does **not** select raw `a` as the deployment
coordinate. It also does not show that subtracting the symmetry point alone
causes the RESET-relative advantage.

Separate two axes rather than treating `a-r` and device count as the same
choice:

1. **Reference policy:** omit the fitted symmetry reference `r`, or retain a
   fixed physical reference conductance while tuning only the active state
   `a`.
2. **Topology:** use one device per dual-rail edge (four devices per original
   signed weight), or use an active/reference pair per edge (eight devices per
   original signed weight).

This gives the following 2-by-2 screen. `B` denotes the characterized lower
conductance used by the no-`r` control and `L` is the nonnegative dual-rail
lift of the requested signed weight.

| Topology | Without fixed `r` | With fixed `r` |
| --- | --- | --- |
| **Four devices; one per rail edge** | `G=B+delta L`; each edge has `D=S=G` | for positive weight, `[[G++,G+-],[G-+,G--]]=[[a,r],[r,a]]`; swap `a/r` for negative weight |
| **Eight devices; pair per rail edge** | lower-branch control `G+=B+delta L`, `G-=B` | `G+=a=r+delta L`, `G-=r` fixed; per edge `D=a-r`, `S=a+r` |

The proposed four-device fixed-reference construction is important: even
though each rail edge contains only one device, its logical signed contrast is

```text
w = (G++ - G+- - G-+ + G--) / 2 = a-r,
```

while every paired node sees physical loading `a+r`. Thus, reference use is
not synonymous with an eight-device differential pair.

The subtraction belongs only to signed transfer. Physical conductances always
add in the nodal voltage denominator. An eight-device simulation that uses
`a-r` in that denominator is not physically correct; similarly, the
four-device construction must retain the `a+r` row/column loading produced by
its `a` and `r` edges.

In schematic unit-gain form, a node update has the structure

```text
              sum_j D_ij v_j + other drive terms
v_i = ------------------------------------------------,
              sum_j S_ij     + other loading terms
```

For a one-device rail edge, `D_ij=S_ij=G_ij`. For a paired rail edge,
`D_ij=G_a,ij-G_r,ij` and `S_ij=G_a,ij+G_r,ij`. In both cases, summing the
physical `S` contributions—not the signed logical weight—forms the
denominator.

The AIHWKit OM fit stores normalized states that can be negative. If these
states are embedded into calibrated nonnegative physical conductances as

```text
G_a = G_0 + s a,
G_r = G_0 + s r,
```

then the same rule is

```text
coupling = G_a - G_r = s (a-r),
loading  = G_a + G_r = 2 G_0 + s (a+r).
```

When `a` and `r` already denote physical conductances, this reduces directly
to coupling `a-r` and denominator loading `a+r`.

The recent pipelines do not yet form this matched 2-by-2 comparison. Raw-p90
is a normalized candidate for the four-device/no-fixed-`r` cell, but its
global affine origin is not an absolute conductance calibration. The
RESET-relative pipeline maps the effective coordinate `a-r` into a
single-conductance DRN interaction, so the mapped effective value supplies
both transfer and loading. It is therefore an **effective-weight upper
control**, not any correctly loaded physical cell in the table. Its `91.00%`
programmed accuracy must not be labeled a four- or eight-device fixed-`r`
result.

The first matched comparison is ideal mapped accuracy from one frozen logical
source, with no HWA and no programming noise. Only schemes that survive this
gate should receive an identity-aware bounds screen and then P&V. The current
raw-p90 study remains a useful negative/stress control, but its
structural-failure arm is not the functional reassignment path specified by
the raw protocol.

## What the recent results actually separate

The following comparison is descriptive because coordinate, mapping,
learning rate, training endpoint model, controller, and checkpoint all differ.
It nevertheless localizes the size of the two losses:

| Quantized-QAT pipeline | Clean selected | Ideal mapped | Programmed mean | Ideal to programmed |
| --- | ---: | ---: | ---: | ---: |
| RESET-relative | 95.37% | 96.29% | 91.00% | -5.29 pp |
| Raw-p90 | 50.33% | 76.36% | 52.66% | -23.70 pp |

The total programmed gap is `38.34` points. Of that descriptive gap,
`19.93` points are already present at the ideal mapped state, and another
`18.41` points come from the larger raw-p90 ideal-to-programmed loss. It is
therefore incorrect to attribute the whole result either to the scalar P&V
controller or to symmetry subtraction.

There is also interrupted evidence from the earlier canonical
differential-pair pilot. That implementation correctly used branch difference
for transfer and branch sum for loading, but it used a different checkpoint,
teacher, cell grouping, and deployment protocol, and its HWA arm stopped after
seven of ten epochs. Its clean programmed deployment reached `49.94%`; the
partial HWA checkpoint selected at epoch index 5 mapped ideally to `95.16%`
and programmed to `55.72%`, with a post-hoc best of `60.36%`. The associated
decomposition found that mapping increased normalized branch-sum means to
`0.74341` in W1 and `0.71689` in W2 while leaving much smaller difference RMS
values of `0.04975` and `0.01377`. This is mechanistic evidence that the sum
conductance can dominate the voltage denominator, not a matched accuracy
comparison with either row above.

Further evidence narrows the mechanism:

- The raw-QAT saved development endpoint reached `89.66%`, while its held-out
  endpoint mean was `52.66%`. Its ideal map was stable (`75.90%` development,
  `76.36%` held out). The training and selection path therefore learned a
  strongly assignment/endpoint-specific compensation.
- Held-out raw-p90 deployment left `3,802` quads, or `15,208` cells, as
  explicit structural target-assignment failures. Among eligible cells,
  apparent acceptance was about `99.85%`; the large all-quad differential
  error was concentrated in the structural failures.
- Sending the raw-p90-trained weights through the RESET-relative controller
  gave `65.94%` ideal mapped and `60.40%` programmed accuracy. Its `5.54`-point
  programming change was nearly the same as the old RESET-trained pipeline's
  `5.29`-point change. The raw weights still mapped poorly, but the large
  additional raw-p90 endpoint loss disappeared.
- The common-cell nine-target experiment used `Delta_g = 0.0123806`, while
  `2 tau_g = 0.0158317`. Its verify windows necessarily overlapped:
  `38.25%` of accepted apparent reads were compatible with multiple codes and
  only `28.50%` of accepted exact-support trajectories had the requested code
  as the nearest persistent endpoint. Apparent acceptance is not evidence of
  a persistent nine-state cell.

Sources are the artifact-verified analyses under:

- `results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/`;
- `results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/`;
- `results/mnist-ibm-om-reset-relative-new-p90-qat-cross-deploy-20260826-v1/`;
- `results/ibm-om-raw-active-common-cell-9-program-verify-20260826-v1/`; and
- `results/mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1/`.

## Matched ideal-mapping screen

The first 2-by-2 screen is complete. It maps the same frozen bias-free ReLU
teacher (SHA-256
`9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52`)
into every arm, uses the same perfect-diode DRN topology and solver, selects
layer scales by ideal mapped accuracy on the fixed 1,024-example calibration
cohort with KL as a tie-break, and evaluates the selected map on all 10,000
MNIST test examples. It performs no device sampling, endpoint programming,
HWA, or training.

Because the OM preset does not provide an absolute conductance calibration,
the fixed references are explicit sensitivity points at 12.5%, 25%, and 50%
of the configured `[0, 110]` microSiemens model range. They are not claims
about the fabricated OM symmetry conductance.

| Reference policy | Four devices | Eight devices |
| --- | ---: | ---: |
| No fixed `r`; lower-bound control | 96.91% | 96.91% |
| Fixed `r` at 12.5% of span | 97.36% | 97.36% |
| Fixed `r` at 25% of span | 97.35% | 97.37% |
| Fixed `r` at 50% of span | 97.38% | 97.37% |

The result rules out an intrinsic noiseless accuracy collapse for either
topology under exactly matched reference placement. In particular, the
proposed four-device pattern `[[a,r],[r,a]]` performs as well as the
eight-device per-edge subtraction in this ideal screen. The recent
`96.29%` RESET-relative versus `76.36%` raw-p90 ideal-map gap is therefore not
an inherent four-versus-eight-device effect; it comes from the different
trained checkpoints and target-mapping protocols bundled into those studies.

Ideal accuracy is a useful first gate, but it is not sufficient to predict
programmed robustness. At the 50% reference point, the selected four-device
maps have mean per-edge denominator loads of `57.871/58.348` microSiemens in
W1/W2, while the eight-device maps have `112.871/113.348` microSiemens. Their
accuracies remain tied because the ideal reference terms are exactly matched
and the scale is calibrated. Real reference mismatch, reduced active
headroom, read noise, line resistance, and P&V error can expose that roughly
doubled loading even though ideal top-1 accuracy cannot.

The executable screen is
[`ibm_om_ideal_mapping_scheme_screen.py`](../experiments/mnist_relu_drn/ibm_om_ideal_mapping_scheme_screen.py),
and its ignored read-only artifact is
`results/ibm-om-ideal-mapping-scheme-screen-20260827-v1/analysis/summary.json`.
The next ideal-only step is to replace the scalar reference with frozen
per-identity `r`, bounds, and headroom from several untouched assignments.

## Why “symmetry subtraction” is not the isolated intervention

[AIHWKit documents](https://aihwkit.readthedocs.io/en/latest/api/aihwkit.simulator.configs.devices.html)
`SoftBoundsReferenceDevice` as exposing a differential read `w = a - r`. The
OM preset enables symmetry-point subtraction and adds a 5% reference error.
The hidden raw state `a` is nevertheless normalized; the preset does not
retain the absolute physical symmetry conductance or a unique microSiemens
origin and scale.

A read-only audit of the held-out repaired population shows that the sampled
`r` shift is not large enough to explain the network gap:

| Population diagnostic | Value |
| --- | ---: |
| `r` median | `-0.000158` |
| `r` 1st / 99th percentiles | `-0.12560 / +0.12518` |
| Correlation of `r` with raw window center | `-0.00038` |
| Raw-`a` four-cell common-capacity 10th percentile | `0.62913` native units |
| `(a-r)` common-capacity 10th percentile | `0.62060` native units |
| Raw support at the frozen native D90 displacement | `90.423%` |
| `(a-r)` support at the same native displacement | `89.997%` |

The audited population fingerprint is
`2721e735ba6722df27a879a5373de2d1da02481cb5eec05b592f967d27e74310`.
The calculation uses the declared `halves`/`paired` layouts and, for each
quad, compares `min(a_max) - max(a_min)` with
`min(a_max-r) - max(a_min-r)`; it performs no programming or resampling.

Subtracting `r` therefore barely changes the common-window yield at a matched
native displacement. This support calculation says nothing about the voltage
denominator: a physical differential-device realization also changes loading
from the active branch alone to the branch sum `a+r`. The labels “raw” and
“RESET-relative” bundle that circuit distinction with several other
deployment choices:

### 1. The raw coordinate adds an arbitrary common-mode offset

Raw-p90 maps

```text
g = (a - A_min) / S,
A_min = -3.455834150314331,
S     =  5.994294881820679.
```

`A_min` is an extreme sampled normalized bound, not a measured zero-
conductance point. The subtraction makes every value non-negative, but it
also adds approximately `-A_min/S = 0.5766` to every cell. That offset remains
in the passive numerator currents, conductance-sum denominators, voltage
scale, and power. It cannot be justified as physical conductance without an
absolute calibration.

### 2. Midpoint placement retains more common-mode loading

Raw-p90 reserves both signs around a midpoint baseline. RESET-relative starts
from a guarded common RESET baseline and raises only the two rails selected by
the sign. Both can represent later sign changes: changing sign can RESET the
previously raised pair and SET the opposite pair. A below-baseline target is
not required for bidirectional logical weights.

As a rough forward-coordinate diagnostic, the RESET-QAT nonzero contrast RMS
is about seven times larger relative to its mean baseline than the eligible
raw-QAT differential RMS in both layers. This is a mapping/loading difference,
not evidence that the raw pulse equation is intrinsically worse.

### 3. The raw study made unsupported quads part of the network

The p90 threshold deliberately guarantees support for about 90% of quads.
The executed study used `structural_failure`, leaving the remaining cells at
their conditioned state. A functional deployment must instead use a frozen
donor/reassignment policy, a true open-circuit selector policy, or fail the
assignment. It must not silently include approximately 10% arbitrary logical
weights and call their cells programming successes or failures.

### 4. Fixed-array endpoint selection encouraged compensation, not transfer

Raw HWA/QAT repeatedly trained and selected against one development
assignment and fixed endpoint streams. The development endpoint substantially
outperformed the same checkpoint's ideal map, but the benefit vanished on the
held-out array. Future HWA must separate the assignments used for minibatch
modification, checkpoint selection, and final testing.

### 5. The codebook was selected from support, not persistent separability

Support, apparent acceptance, persistent accuracy, and code distinguishability
are different facts. A deployment codebook must pass a persistent confusion
test before it is used for QAT or an on-chip update rule.

## Four-device topology: one device per dual-rail edge

Each physical rail edge contains one nonnegative conductance, so at the edge
level

```text
D_edge = G,
S_edge = G.
```

The no-fixed-`r` arm uses an independently programmed active conductance on
each of the four edges. The raw-p90 global affine coordinate is one numerical
candidate for that arm, but not yet a physical-conductance realization because
its origin comes from an extreme normalized bound rather than an absolute
calibration.

The fixed-reference arm instead uses, for a positive requested signed value,

```text
G++ = G-- = a,
G+- = G-+ = r,
```

and swaps `a` and `r` for a negative value. Therefore

```text
w = (G++ - G+- - G-+ + G--) / 2 = +/- (a-r),
```

while the source- and destination-rail denominator contribution from each
paired logical connection is `a+r`. Only the two sign-selected `a` cells are
raised for a fixed deployment; the other two stay at the declared reference
state. A later sign-changing update would have to return the old active pair
to `r` and raise the opposite pair, so this write-role exchange must be
specified before any on-chip training claim.

For a lower-reference control `B_Q`, the same construction is

```text
G++ = G-- = B_Q + max( D_Q, 0)
G+- = G-+ = B_Q + max(-D_Q, 0),
```

with total four-cell loading `4 B_Q + 2 abs(D_Q)`. Choose `B_Q` or `r` from a
declared commissioning/calibration rule, retain it in the forward KCL, and
freeze a global code range before network metrics are inspected. Unsupported
quads consume a predeclared donor stream; donor exhaustion is an assignment
failure. The current midpoint rule and structural-failure policy remain
separate matched ablations.

## Eight-device topology: a pair per dual-rail edge

Every rail edge contains a nonnegative active branch and a nonnegative second
branch. With a fixed OM reference,

```text
D_edge = G_a - G_r    represented coupling,
S_edge = G_a + G_r    voltage-denominator loading and power.
```

For a nonnegative lifted target `L`, tune only
`G_a = G_r + delta L` and keep `G_r` fixed. The no-fixed-`r` control uses the
same construction with the second branch held at the characterized lower
conductance `B` instead of the fitted symmetry reference. This makes reference
policy and branch count separately testable.

The solver must retain the two branch tensors. It must not construct
`G_a-G_r` and then use that difference for both off-diagonal transfer and
diagonal loading. The existing `SignedDenseResistive` interaction implements
the required difference/sum algebra.

The equation-level circuit contract in
[`differential_pair_amplifier_implementation.md`](differential_pair_amplifier_implementation.md)
also shows why a one-way MVM subtraction is insufficient for a reciprocal
DRN: the opposite-polarity branch and signed returned-current path need an
explicit active two-port or equivalent circuit.

This topology must pass:

- exact parity showing that every numerator coupling uses `G_a-G_r` and every
  denominator/self-loading term uses `G_a+G_r`;
- solver parity for raw KCL and projected KKT residuals;
- gain-mismatch, reference-mismatch, write/read-noise, and line-resistance
  controls;
- finite-bandwidth/stability checks before a circuit-level claim; and
- explicit accounting of branch count, conductance sums, voltage margins,
  current, and power.

Until those checks exist on matched active/reference identities, the current
RESET-relative accuracy remains an effective-weight upper control.

## Matched investigation sequence

### Stage 0A: homogeneous ideal mapping — completed

The scalar-reference 2-by-2 result above is the first gate. It establishes
that all four constructions can retain approximately `97%` ideal mapped test
accuracy and that the four-device fixed-`r` proposal is not intrinsically
inferior to the eight-device realization.

### Stage 0B: identity-aware ideal mapping — exploratory control complete

Still without training or P&V, bind all four cells to the same frozen logical
teacher/checkpoint, DRN topology, data cohorts, solver, scale-selection rule,
and untouched OM assignments. Replace scalar `r` with each assigned identity's
saved reference state and enforce its active/reference bounds and headroom.
Evaluate:

1. four devices without fitted `r`;
2. four devices with the sign-selected `[[a,r],[r,a]]` construction;
3. eight devices with the second branch at the lower-bound control `B`;
4. eight devices with an explicit fixed `r` branch; and
5. the existing effective-weight path only as a nonphysical upper control.

Report ideal mapped accuracy first, followed by teacher agreement, ideal KL,
support and reassignment counts, signed contrast, each layer's summed
denominator conductance, `RMS(D)/mean(S)`, voltage RMS/margins, and power
proxies. Reference imbalance must remain visible: do not replace heterogeneous
`r` values by a post-hoc common scalar after seeing accuracy. A positive global
output gain may be calibrated on the development cohort, but it cannot alter
the physical equilibrium or use test labels.

Only arms that preserve the declared ideal-accuracy gate and cover every
logical target proceed to programming. The interrupted differential-pair
pilot is useful implementation evidence but does not replace this comparison:
its branches, checkpoint, and mapping protocol were different.

The first Stage 0B control is now frozen in
[`ibm_om_bounded_codebook_scheme_screen.md`](ibm_om_bounded_codebook_scheme_screen.md)
and
[`screen.json`](../examples/mnist_relu_drn/ibm_om_bounded_codebook_scheme_screen/screen.json).
The logical source and solver remain FP32. Each literal repaired OM identity
contributes its own bounds, fitted reference, and deterministic lower-to-SET
pulse codes through pulse index 128. The cycle-to-cycle random term and
apparent write noise are exactly zero, and a requested target is projected to
the nearest deterministic code. Thus this control includes identity range and
pulse resolution but excludes the realistic controller and stochastic write
endpoint.

The nominal OM `dw_min` is not treated as a universal fixed level count. A
continuous target inside the same per-cell codebook baseline-to-maximum
envelope is retained as a diagnostic upper control. The v2 contract freezes
the initial-RESET receipt's `[1.0, 1.0]` layer fractions and
`4.46683592150963` positive output gain for every scheme and both endpoint
controls. Assignment `86001` supplies diagnostics only; topology-specific
assignments `87001`, `87002`, and `87003` are evaluated without retuning.
Four- and eight-device assignments have different hashes because the physical
cell counts and binding catalogs differ; with/without-`r` arms within one
topology reuse exactly the same identities. The earlier v1 exploratory run is
superseded because it performed scheme-specific calibration, even though all
four v1 schemes happened to select `[1.0, 1.0]` and positive gain could not
change their top-1 accuracies.

The v2 held-out bounded-codebook means are `35.98%` (four/no `r`), `57.85%`
(four/fixed `r`), `29.95%` (eight/no `r`), and `36.51%` (eight/fixed `r`).
All fail the predeclared 90% gate, including continuous-envelope means of
`41.02%`, `64.35%`, `31.82%`, and `40.73%`. The fitted reference improves the
four- and eight-device means by `21.87` and `6.56` percentage points,
respectively, while the eight-device topology is worse than four devices by
`6.03` points without `r` and `21.34` points with `r`.

These are scheme effects under heterogeneous OM bounds and correct physical
loading, not a direct benefit of subtracting `r`. Mean hidden/output voltage
RMS spans `2.959/1.426` for four/no `r`, `0.313/0.054` for four/fixed `r`,
`1.947/0.668` for eight/no `r`, and `0.172/0.006` for eight/fixed `r`.
The sum-loading-induced operating-point shift is therefore large enough that
the next ideal investigation must resolve baseline cancellation, headroom,
and voltage scaling before any stochastic programming study.

### Stage 0C: exact-lower-bound four-delta diagnostic — complete

The next ideal screen no longer projects the fitted reference onto a dense
SET-only pulse trajectory and no longer applies the initial-RESET scale/gain
to every representation. It implements the user's standardized rule:

1. no-fixed-`r` arms use each identity's sampled RESET/lower state as the
   active origin;
2. fixed-`r` arms retain the exact intrinsic sampled `r` and center the active
   grid on it, moving only `a` if the zero-offset active state is outside its
   bounds;
3. every adjacent standard target is exactly four nominal OM increments apart
   in one declared conductance coordinate;
4. logical sign remains encoded by rail placement, so the four-device arm
   retains `[[a,r],[r,a]]` and does not count both sides of `r` as duplicate
   sign levels; and
5. each scheme refits its layer-scale pair and positive KL gain on the same
   development assignment, then freezes them on the same three held-out
   assignments.

The fixed-reference conductance is never projected or optimized. The active
and reference conductances remain separate in the eight-device solver, so
transfer uses their difference and denominator loading uses their sum. The
executable and frozen contract are documented in
[`ibm_om_standard_level_scheme_screen.md`](ibm_om_standard_level_scheme_screen.md).

The v1 lower-bound-origin result is retained as a diagnostic, but it does not
implement the newly specified no-`r` commissioning method. Its held-out
standard/continuous means were `31.48/32.60%` (four/no `r`),
`23.03/71.11%` (four/fixed `r`), `25.18/27.16%` (eight/no `r`), and
`14.16/42.69%` (eight/fixed `r`). The large fixed-`r` continuous-to-standard
collapse exposed insufficient four-delta level capacity; it is not evidence
that the fitted reference itself is harmful.

### Stage 0D: per-cell raw-`a` RESET-mean origin — complete

The corrected no-`r` arm uses the same eight sequential one-RESET-pulse/read
sample cost as the well-performing historical RESET-relative study, but it
implements the user's distinct estimator: each physical cell retains its own
arithmetic mean in apparent raw `a`. There is no quad maximum, 3-SE guard,
or cross-cell broadcast. The mean is mapped to the nonnegative conductance
coordinate, bounded to that cell's usable active interval, frozen once per
topology/assignment, and followed by levels exactly four nominal increments
apart. Commissioning uses the preset stochastic pulse/read terms; ideal level
deployment remains noiseless. The fixed-`r` arms are unchanged and provide a
bit-identical control against Stage 0C.

This is a scheme-optimized comparison, not a characterization-cost-matched
causal ablation: the fixed-`r` arm consumes exact hidden `r`, whereas no-`r`
uses an eight-read RESET estimate. The frozen v2 contract, assignments,
calibration grid, bound policy, and claim boundary are documented in
[`ibm_om_standard_level_scheme_screen.md`](ibm_om_standard_level_scheme_screen.md).

The held-out v2 standard/continuous means are `25.76/27.23%`
(four/no `r`), `23.03/71.11%` (four/fixed `r`), `23.46/25.22%`
(eight/no `r`), and `14.16/42.69%` (eight/fixed `r`). All fail 90%.
The fixed-`r` results are bit-identical to Stage 0C. Relative to its exact
lower-bound control, per-cell RESET commissioning lowers no-`r` standard
accuracy by `5.71` points for four devices and `1.72` points for eight.

This rejects the hypothesis that independent per-cell RESET averaging is the
missing ingredient behind the historical 91% RESET-relative result. About
`60.32%` of observed means require public-coordinate or cell-bound
projection. More importantly, the surviving cell-specific origin offsets do
not cancel at logical zero: no-`r` baseline-contrast RMS and conductance-sum
loading both rise while median signed capacity stays at seven levels and
quantization RMS is essentially unchanged. The resulting voltage reduction
and accuracy loss are already present in the continuous envelope.

The historical high-accuracy mapper instead pooled four observed cells into
one guarded shared quad baseline. The next no-`r` control must therefore test
baseline **grouping**, not more independent averaging: one common origin per
four-device rail quad, and a matched shared zero per active/reference pair (or
stricter common group) for eight devices. Fixed-`r` needs a separate level-
capacity control because the one-sided four-delta grid supplies only three
median signed levels and leaves about `14--16%` of quads at zero capacity;
its large continuous envelope must not be dismissed as an `r` failure.

### Stage 0E: baseline decision — shared-zero ideal initialization

The workflow-managed implementation is frozen in
[`ibm_om_baseline_selection.md`](ibm_om_baseline_selection.md) under study
`mnist-ibm-om-four-device-baseline-selection-20260828-v1`. It expands the
original two-arm screen into a matched four-policy, four-device comparison:
independent cell origins, one shared quad origin, two destination-column
origins, and sign-selected partner-reference enforcement. One jointly
repaired identity assignment is shared across policies. Because the scientific
question at this stage is baseline choice rather than level choice, its
primary definition is `ibm_om.ideal_bounded_continuous_init.v1`; the frozen
four-delta map remains a secondary diagnostic using the same selected
calibration. The detailed contract in the linked document is authoritative
where this earlier version-3 sketch differs.

The earlier two-arm Stage 0E sketch used
`ibm_om.ideal_bounded_standard4delta_init.v1` as its primary metric. The
formal four-policy study retains that definition only for its secondary
codebook diagnostic. Both versions test the specific explanation exposed by
Stage 0D:

> The no-`r` ideal initialization fails primarily because independently
> commissioned baselines create a large zero-weight contrast. Enforcing one
> physically reachable zero within the relevant device group should remove
> that contrast while retaining every baseline in denominator loading.

Implement it as a new version-3 screen contract and config. Do not change the
meaning, schema, or artifacts of completed v1/v2 screens.

Keep the v2 logical source, teacher, topology, solver, assignments, eight
sequential RESET/read samples per cell, normalized conductance coordinate,
four-delta spacing, scale grid, per-scheme development refit, and noiseless
level deployment. The declared intervention is the baseline rule. Keep `h`
and the whole-level capacity algorithm frozen. Moving `B` may mechanically
change realizable capacity under the bounds; record that consequence, but do
not optimize the spacing or select a new level count in this stage. The
existing independent-baseline v2 results are the frozen parent controls; do
not rerun or retune them after inspecting Stage 0E.

For every commissioned cell, first reproduce the bounded estimate

```text
b_i = clip((mean(a_i)+1)/2, [l_i, u_i]).
```

For a four-device no-`r` rail quad, define

```text
L_Q = max_i(l_i),
U_Q = min_i(u_i),
B_Q_requested = max_i(b_i),
B_Q = clip(B_Q_requested, L_Q, U_Q).
```

The group is feasible only if `L_Q <= U_Q`. The requested-to-realized group
projection is part of this ideal bound-constrained mapper and must be recorded;
it is not evidence that a stochastic RESET/P&V sequence can reach the value.
For a signed integer magnitude `m` and spacing
`h=4*(nominal_dw_min/2)`, use

```text
positive: G++=B_Q+m h, G+-=B_Q,     G-+=B_Q,     G--=B_Q+m h
negative: G++=B_Q,     G+-=B_Q+m h, G-+=B_Q+m h, G--=B_Q.
```

All four full `G=B_Q+d` values enter KCL. The realized zero contrast is
exactly zero, while the four-cell loading is `4 B_Q + 2 m h`. Capacity is
the largest whole `m` for which every potentially raised cell remains within
its own upper bound.

For an eight-device no-`r` edge with active and reference branches, define

```text
B_e_requested = max(b_a, b_r),
B_e = clip(B_e_requested, max(l_a,l_r), min(u_a,u_r)),
```

and require a nonempty pairwise common interval. Keep the reference branch at
`G_r=B_e`; set the active branch to `G_a=B_e+d` on the sign-selected rail.
Thus logical zero has exact `D=0`, while the solver retains `S=2 B_e+d`.
Quad capacity is the minimum whole active-branch capacity over the four rail
edges. A stricter common zero across all eight devices may be reported later
as a separate grouping intervention; it is not required to test local
active/reference cancellation.

An empty exact common interval must fail the assignment or consume a
separately predeclared deterministic group-reassignment stream. The mapper
must never fall back to independent cell baselines, drop a logical weight, or
choose a donor after observing accuracy. Report the group-baseline projection
count and error separately from empty-window and reassignment counts. A
feasible group with zero positive capacity remains an explicitly reported
zero-only group.

The minimal new primary arms are four/no-`r` shared-quad zero and eight/no-`r`
shared-pair zero. The existing four/eight no-`r` independent-baseline results
provide their paired historical controls. The intrinsic-`r` arms remain
unchanged reference-policy controls, but they must not be called shared-zero
schemes:
heterogeneous intrinsic `r_i` can still create a nonzero rail contrast. A
separate fixed-`r` spacing or identity-balancing study follows only after the
shared-zero no-`r` question is answered.

In this normalized model, "exact `r`" means that `r` is not moved onto the
active integer grid. The repository mapping still intersects native state
with the public `[0,1]` coordinate; any `r` outside native `[-1,1]` is clipped
by that coordinate. The canonical metric must retain both raw and mapped `r`,
report the clipping count, and must not describe a clipped value as the
unchanged native reference.

Stage 0E is valid only if it records the complete first-metric artifact
contract above and additionally proves:

- exact zero baseline contrast in every feasible four-device shared quad and
  exact `G_a-G_r=0` in every feasible eight-device pair at `d=0`;
- finite, nonnegative, in-bound `B`, `d`, and `G=B+d` tensors and exact
  four-delta integer levels;
- full-conductance four-device KCL and eight-device `D/S` parity;
- explicit common-window, zero-capacity, reassignment, and failure counts;
- one frozen selected scale pair and gain per scheme from assignment `86001`,
  evaluated without refit on assignments `87001`, `87002`, and `87003`; and
- deterministic main/replay equality under the declared semantic path
  normalization.

The headline is the held-out mean and assignment range of
`ideal_mapped_init_accuracy`; `continuous_envelope_accuracy`, baseline
contrast, capacity, loading, voltage, and sign-flip statistics explain any
remaining loss. The existing 90% held-out-mean gate and complete functional
coverage remain the progression criteria. No BPTT, P&V, or on-chip recovery
arm is eligible while this gate fails.

### Stage 0F: fixed-`r` identity-balance diagnostic

The four-device intrinsic-reference idea has a distinct zero condition:

```text
r+++r-- = r+-+r-+.
```

Equal references are sufficient but not necessary. Stage 0F therefore tests
whether weight-blind physical identity assignment can make this signed sum
small enough for useful ideal initialization. It is not part of the no-`r`
shared-zero Stage 0E intervention and does not replace that pending canonical
gate.

The frozen `reference_balanced_binding_v1` policy globally stable-sorts mapped
references within each layer, forms consecutive quartets, chooses the minimum
signed-sum two-versus-two partition, and then deterministically shuffles the
completed quartets over logical addresses. Every identity field moves
together; no identity crosses a layer, and the rule reads no weights, labels,
bounds, calibration outcome, or accuracy. The sampled-order binding is the
matched control.

Stage 0F uses the bounded continuous positive-only map and explicitly records
every branch as `G=B+d`. It contains no four-delta rounding or level-count
cap, so it informs only the baseline decision. Both bindings receive their
own development refit, and both selected calibrations are cross-applied on
held-out assignments to expose calibration interaction. Main and replay run
through the native workflow under
`ibm_om.reference_balanced_continuous_init.v1`; exact inputs, algorithms,
artifacts, caveats, and the 90% diagnostic gate are frozen in
[`ibm_om_four_reference_balance.md`](ibm_om_four_reference_balance.md).

The completed main/replay study reduced intrinsic-reference zero-contrast RMS
by 58--150 times and increased held-out scheme-optimized ideal accuracy from
84.49% to 91.83%.  Under the sampled-order calibration frozen for both
bindings, the mean paired improvement was +7.68 percentage points.  This
strongly supports heterogeneous signed reference contrast as a failure
mechanism.  It does not establish a deployable assignment rule: the treatment
globally regrouped complete device identities within each layer, which is not
available when initializing a fixed physical array.  The workflow outcome is
therefore `inconclusive` for deployment despite passing the mechanistic 90%
gate.

### Stage 0G: fixed-binding local reference compensation

The next intervention leaves every sampled identity at its existing physical
address and leaves its intrinsic symmetry parameter `r_i` immutable.  It
changes only the programmed zero-state baseline.  For rail order
`(++,+-,-+,--)`, let `s=(+1,-1,-1,+1)`, map the symmetry point to `rho_i`, and
form the reachable control baseline

```text
B0_i = clip(rho_i, [l_i,u_i]).
```

The treatment solves independently inside each existing quad:

```text
B* = argmin_B 0.5 ||B-rho||_2^2
     subject to l_i <= B_i <= u_i and s^T B = 0.
```

Thus `B*` is the smallest local conductance movement that makes the normalized
float64 zero target exact.  Any last-bit residual after canonical float32
physical conversion is reported and retained in KCL.  No identity is
reassigned and no donor is selected.  Away from active bounds the correction
is the checkerboard update
`B*=rho-(s^T rho/4)s`; it preserves both row sums, both column sums, and total
zero-state loading.  Bound-active quads retain exact zero through the declared
box projection and explicitly report any loading redistribution.

The primary continuous comparison uses the same offset tensor `d` for both
baselines, limited by the headroom common to both policies.  Every physical
branch remains `G=B+d`, and every full `G` enters both signed transfer and
denominator loading.  Assignment 86001 selects the scale/gain; assignments
87001--87003 receive the frozen two-by-two policy/calibration matrix.  The
primary effect is `B*@cal_B0 - B0@cal_B0`; independently refitted accuracy is
secondary.  Quantization, pulse programming, training, and noise remain
excluded so this study resolves only the baseline decision.  Its tracked
contract is in
[`ibm_om_local_reference_compensation.md`](ibm_om_local_reference_compensation.md).

Read-only feasibility on the four frozen populations found an exact bounded
solution for all 158,800 quads.  The unconstrained load-preserving correction
was already in bounds for 97.66%; the bounded correction magnitude had median
0.00891 and p99 0.03491 in normalized `x`, below the nominal single-pulse
increment 0.04745 in most cases.  These are feasibility diagnostics, not
accuracy or programmability evidence.  In particular, a successful continuous
result must be followed by a discrete baseline/codebook and deterministic-P&V
test because the required trim is commonly sub-pulse.

### Stage 1: codebook and P&V gate

Screen at least 3-, 5-, and 7-level codebooks on development identities and
freeze the selected spacing before held-out validation. A candidate is
ineligible for network QAT unless:

- verify windows do not overlap (`2 tau <= Delta_cell`);
- at least 90% of accepted exact-support held-out trajectories have the
  requested nearest persistent code;
- apparent acceptance, persistent correctness, and fresh-read correctness are
  reported separately; and
- the functional assignment path covers every logical quad or fails
  explicitly.

The level count is chosen by held-out persistent distinguishability and
network validation, not by apparent acceptance alone.

### Stage 2: matched HWA and QAT transfer

Only schemes passing Stages 0 and 1 enter training. Continuous HWA and the
selected quantized QAT arm must:

- start from the same logical checkpoint and optimizer state;
- use multiple declared training assignments or resampled identities;
- select checkpoints on separate development assignments and endpoint streams;
- use the same base seeds, minibatch order, data split, and compute budget;
- deploy on untouched held-out assignments with matched endpoint seeds; and
- preserve ideal-map, programmed-endpoint, and assignment-transfer losses as
  separate outcomes.

No Tiki-Taka, LoRA, or direct-pulse recovery arm should be launched until a
deployment scheme passes this transfer gate. On-chip recovery must then start
from the same explicitly named persistent deployed bundle.

## Immediate implementation order

1. Keep `ideal_mapped_init_accuracy` as the only active accuracy gate; do not
   launch additional raw-p90 HWA, BPTT, or transfer runs yet.
2. Add a fail-closed physical-target artifact/API carrying explicit `B`, `d`,
   and `G=B+d`, exact correct counts, one hardware-instance ID, and the metric
   validity record. Quarantine effective-`(a-r)` writers from physical metric
   IDs.
3. Implement the Stage 0E matched shared-zero control: a bound-feasible common
   commissioned baseline per four-device rail quad and per eight-device
   active/reference pair, with explicit common-window failure or a separately
   frozen reassignment policy.
4. Add heterogeneous-baseline four-device KCL and eight-device branch-pair
   regression tests proving that every full conductance contributes to both
   transfer and loading.
5. Run the Stage 0E deterministic main/replay screen and compare it with the
   frozen Stage 0D independent-baseline artifacts. Preserve the completed
   homogeneous 2-by-2 result and every exact teacher/config hash.
6. If the first metric passes, implement the persistent codebook screen with
   non-overlapping verify windows. If it fails, remain in ideal-only analysis
   and test the declared capacity/identity controls before training.
7. Only after the first metric and persistent-code gate pass, implement the
   full-physical-`G` BPTT path required for
   `same_hardware_bptt_accuracy`, followed by untouched-hardware transfer.
8. Keep every scheme blocked from absolute physical claims until OM
   conductance traces or a versioned normalized-to-conductance calibration are
   supplied. Materialize a workflow-managed P&V study only after its selected
   arms, configs, artifact schema, and parity tests exist.

This sequence determines whether the remaining gap belongs to device
programming, passive loading, assignment transfer, or training. It deliberately
does not assume that on-chip recovery is needed.
