# IBM OM deployment-scheme investigation

- Status: homogeneous ideal-mapping screen complete; identity-aware ideal
  mapping is next; no new native training or P&V experiment has been launched
- Date: 2026-08-27
- Device source: AIHWKit 1.1.0 `ReRamArrayOMPresetDevice`
- Evidence class: normalized hardware-derived fitted model, not raw measured
  conductance traces and not an absolute conductance calibration

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

### Stage 0B: identity-aware ideal mapping — next

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

1. Stop treating additional raw-p90 HWA runs as the next scientific step.
2. Preserve the completed homogeneous 2-by-2 ideal-mapping result and its exact
   teacher/config hashes.
3. Implement the identity-aware ideal-only 2-by-2 mapper using frozen OM
   `r`, bounds, and headroom on several untouched assignments.
4. Add the matched load decomposition for lower versus fitted-`r` placement,
   four versus eight devices, and functional reassignment versus structural
   failure.
5. Implement the persistent codebook screen with non-overlapping verify
   windows.
6. Bind the existing difference/sum interaction to explicit active/reference
   OM branches and verify numerator/denominator parity.
7. Keep every scheme blocked from absolute physical claims until OM
   conductance traces or a versioned normalized-to-conductance calibration are
   supplied.
8. Materialize the exact workflow-managed P&V study only after the selected
   arms, configs, artifact schema, and parity tests exist.

This sequence determines whether the remaining gap belongs to device
programming, passive loading, assignment transfer, or training. It deliberately
does not assume that on-chip recovery is needed.
