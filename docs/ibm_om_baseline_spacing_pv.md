# IBM OM shared-destination baseline, spacing, and P&V study

- Study ID:
  `mnist-ibm-om-shared-destination-baseline-spacing-pv-20260828-v1`
- Experiment ID: `mnist_ibm_om_baseline_spacing_pv.v1`
- Date frozen: 2026-08-28
- Evidence class: `model_based_aihwkit_preset`
- Device model: AIHWKit 1.1.0 `ReRamArrayOMPresetDevice`
- Execution: validate-only CUDA for production and the excluded smoke canary;
  no CPU path
- Status: the clipped-range direct CUDA screen is retained as a historical
  diagnostic, and its no-clipping successor has completed all 27 CUDA
  configurations and 135 persistent endpoints. Both are
  `exploratory_noncanonical`; neither has been workflow-reviewed or finalized.

## Post-run scope correction: `x=1` was an imposed ceiling

The completed 3-by-3 screen deliberately inherited the earlier mapping

```text
x_raw = (a+1)/2,
x_public = clip(x_raw, 0, 1),
G = G_min + (G_max-G_min) x_public.
```

It clipped sampled cell bounds before constructing `L`, `U`, and `B`, required
every ideal target to remain in `[0,1]`, and hard-projected raw P&V endpoints
back to `[0,1]` at the circuit handoff. The tables below are an exact record of
that frozen intervention, but `x=1` is not an intrinsic OM SET bound.

Across the three held-out repaired assignments, 49.90% of 476,400 raw sampled
upper coordinates exceed `1` and were collapsed to exactly `1`; 23.25% of
individual clipped headrooms consequently equal exactly `1`. The raw upper
coordinate has `p90=1.224`, `p99=1.405`, and maximum `1.847`. A direct
same-identity diagnostic without public clipping gives individual-cell
RESET-to-upper-bound headroom `p10=0.584`, median `0.978`, and `p90=1.402`,
rather than `0.554`, `0.857`, and `1.000`. This diagnostic does not replace a
rerun because the original assignment repair, baselines, calibration, full-G
loading, and accuracy all used the clipped bounds.

The intended branch question permits any **nonnegative** conductance. Its
successor must retain raw sampled support and define one common affine
embedding such as

```text
G = G0 + s x_raw,    s > 0,
```

with a globally frozen `G0` large enough to keep every used branch
nonnegative. `G0` and `s` may not vary by cell or held-out assignment, and the
conversion scale must be separate from the largest conductance accepted by
the DRN. Raw P&V endpoints above `x=1` must be handed to the circuit through
that same affine map rather than clipped. Since the OM preset supplies no
absolute conductance calibration, this remains a normalized affine circuit
embedding rather than a microSiemens claim.

Accordingly, the completed screen is a **clipped-range diagnostic**, not the
final baseline/spacing selection. Its internal finding that one-sided
initialization did best at `B=L,h=delta_x` remains a historical measurement
about that capped contract only. The corrected direct CUDA successor is now
reported below. It independently selects the same nominal design, but rejects
the clipped screen's unrestricted-deployment accuracy and controller
interpretation.

## Question

Under the historical public-`[0,1]` embedding, for the four-device,
no-fixed-reference topology selected by the completed baseline study, how
should the shared destination-column baseline and uniform conductance-level
spacing be chosen when both ideal quantization and noisy program-and-verify
are considered?

The study is a matched 3-by-3 design:

```text
baseline position alpha in {0, 0.25, 0.5}
spacing h/delta_x       in {1, 2, 4}
```

where `delta_x=nominal_dw_min/2` in the repository's normalized physical
conductance coordinate `x=(a+1)/2`. The design intentionally couples baseline
placement to the mechanically available number of whole levels. It does not
fix an analyst-selected level count.

The initial hypothesis is that smaller spacing reduces deterministic
quantization error but eventually makes adjacent targets insufficiently
distinguishable under P&V, while increasing `alpha` provides downward RESET
headroom at the cost of upward SET capacity and larger zero-state loading. A
useful deployment design should therefore lie on a joint ideal/P&V frontier,
not necessarily at the smallest spacing or lowest baseline.

## Frozen source and hardware matching

Every arm uses the same frozen bias-free ReLU source and teacher:

```text
data/mnist_relu_teacher_fixed_init_20260816.pt
SHA-256: 9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52
```

Development assignment `86001`, held-out assignments `87001`, `87002`, and
`87003`, the eight-read RESET commissioning rule, and the complete-quad joint
repair are inherited semantically from
`mnist-ibm-om-four-device-baseline-selection-20260828-v1`. The new runtime
must reconstruct them deterministically and verify the same identity,
commissioning, donor-selection, and hardware-instance fingerprints. It must
not discover a prior run by recency or use its mapped targets as inputs.

The joint assignment is repaired before `alpha`, spacing, calibration, or P&V
is considered. Eligibility and donor selection therefore cannot depend on a
candidate's classification accuracy, apparent acceptance, pulse cost, or
written endpoint. Donor exhaustion invalidates the assignment.

Each held-out assignment has five frozen endpoint-repeat seeds:

```text
87001: 89101, 89102, 89103, 89104, 89105
87002: 89201, 89202, 89203, 89204, 89205
87003: 89301, 89302, 89303, 89304, 89305
```

The same five seeds are reused across all nine interventions at a given
assignment. Thus each arm has 15 stochastic P&V endpoints over three hardware
assignments, with P&V randomness paired across baseline and spacing choices.

## Baseline construction

For each destination-column group `j`, let `b_i` be the bounded commissioned
RESET mean of cell `i` and `u_i` its sampled upper conductance bound **after
intersection with the public `[0,1]` coordinate**. Define

```text
L_j = max_i b_i
U_j = min_i u_i
B_j(alpha) = L_j + alpha (U_j - L_j).
```

The groups are `(G++,G-+)` and `(G+-,G--)`. Both cells in a group receive the
same exact baseline, so the zero-state quad is

```text
(B+, B-, B+, B-)
```

and its signed contrast is exactly zero even when `B+ != B-`. The full
zero-state loading remains `2 B+ + 2 B-`; it is never cancelled from the
circuit.

`alpha=0` recovers the completed study's
`shared_destination_columns_reset_max` rule. `alpha=0.5` centers the baseline
between the highest commissioned RESET mean and lowest sampled upper state of
each destination group. `alpha=0.25` is the predeclared intermediate point.

## Levels and ideal mapping

For spacing multiplier `m` in `{1,2,4}`:

```text
h = m delta_x,
G_i = B_j(alpha) + n_i h.
```

Initial logical sign remains encoded by choosing the two active rails. The
initialization codebook uses nonnegative integer indices only; it does not use
negative levels to encode logical sign. Half-step ties round away from zero,
and indices are capped by the exact sign-selected active-pair headroom. No
short terminal interval is invented.

The runtime nevertheless records the mechanically available directional
capacities

```text
N_down,j = floor((B_j-L_j)/h)
N_up,j   = floor((U_j-B_j)/h).
```

`N_down` is latent rewrite headroom, not a level used by this initial logical
mapping. P&V may issue a RESET correction after overshoot, but this study does
not perform a logical in-place rewrite or on-chip update. Claims about
bidirectional updateability therefore require a later matched rewrite study.

Each `alpha` searches the same 16 layer-scale pairs from
`{0.125,0.25,0.5,1.0}^2` on the fixed 1,024-example development cohort using
the continuous bounded envelope before any spacing is applied. It selects
continuous accuracy first, calibrated KL second, and the lexicographically
smallest pair last. A positive logit gain is fitted on the same continuous
development map. The selected pair and gain are then identical and frozen
across `h` in `{1,2,4} delta_x`, all held-out assignments, and all P&V
repeats for that `alpha`. Quantized results, P&V outcomes, held-out labels,
and pulse costs cannot refit calibration.

Calibration is therefore scheme-optimized for baseline position but shared
across the spacing intervention. This preserves the completed baseline
study's per-policy continuous refit while making the spacing comparison
causal at fixed calibration. Continuous bounded-envelope accuracy is reported
for each `alpha`; its three spacing arms must be exactly identical.

As a hard backward-parity gate, `alpha=0,h=4 delta_x` must reproduce the
finalized `shared_destination_columns_reset_max` ideal result exactly:

| Assignment | Correct / 10000 | Prediction SHA-256 |
| --- | ---: | --- |
| 87001 | 9098 | `32714304647fba730bfd769b8d5b43cf5d6b3dad7cbf049da595a7f3f2c0a4d6` |
| 87002 | 8580 | `3be15ece3ad194e70c846d4207ef3b097162b72ae39764c0f405a759dfe0789f` |
| 87003 | 8889 | `43ddf6df2b0c685a185ed2d8fe946d848259e32d5df3c8c02286b1a92512980f` |

That parity arm must also select scale fractions `(1.0,1.0)` and fixed logit
gain `14.12537544622754`. Any mismatch invalidates the new study rather than
creating a revised historical result.

## The two matched accuracy endpoints

The deterministic endpoint is
`ibm_om.ideal_bounded_uniform_spacing_init.v1`, reported as
`ideal_bounded_uniform_spacing_init_accuracy`. It applies the exact bounded
target `G_target=B+n h` with no target-write or inference-read noise and no
optimizer update.

The stochastic endpoint is
`ibm_om.pv_persistent_uniform_spacing_init.v1`, reported as
`pv_persistent_uniform_spacing_init_accuracy`. Starting from the declared
conditioned lower state, the explicit pulse plant and one-pulse controller
program every physical rail toward the same `G_target`. The controller may
observe only the continuous target, apparent raw-`x` verify reads, and its own
history. Persistent state, corruption state, and hidden device parameters
remain logging-side facts.

The P&V contract is:

```text
controller                   one_pulse
start protocol                conditioned lower bound to target
verify tolerance              0.5 delta_x
maximum programming pulses    128
verify coordinate             raw x=(a+1)/2
persistent coordinate         raw x=(a+1)/2
endpoint used for accuracy    persistent
public circuit handoff        hard-clip raw x to [0,1], save masks/deltas
inference read noise          off
```

The verify tolerance is fixed in physical `delta_x` units rather than set to
half of `h`; otherwise changing spacing would also change controller
strictness. Cycle-to-cycle pulse variation and apparent verify noise remain
enabled. Apparent acceptance, exact persistent target reachability, and
verify-window intersection are saved and reported as separate facts. An
apparent noisy admission is never relabelled as a persistently correct code.

The persistent endpoint is applied to the DRN without adding a new inference
read-noise draw. Thus the ideal-to-P&V difference isolates stochastic writing
and controller behavior, including any persistent residual produced by noisy
verification. Apparent-endpoint accuracy is diagnostic only.

The OM native state can extend beyond `a in [-1,1]`, whereas the passive DRN
uses the public nonnegative coordinate `x in [0,1]`. The controller and saved
continuation state therefore retain the unmodified native raw endpoint. Only
at the passive-circuit handoff is raw `x` hard-projected to `[0,1]`. Every run
saves the below/above masks, displacement tensor, and projection counts for
both persistent and apparent endpoints. This projection is neither a P&V
success nor a controller operation; it is reported as a separate handoff
loss. Ideal targets or targets outside exact native support still invalidate
the run and are never repaired by this rule.

## Full-conductance and validity invariants

Every ideal and written rail is evaluated as a full nonnegative conductance.
For canonical quad order `(++,+-,-+,--)`:

```text
C(G) = (G++ - G+- - G-+ + G--)/2
L(G) = G++ + G+- + G-+ + G--.
```

No implementation may subtract `B` before the nodal solve. Both signed
transfer and conductance-sum loading are derived from the same saved full `G`
tensors. Each run must save or hash `B`, integer `n`, ideal offset `d=n h`,
`G_target=B+d`, apparent endpoint, persistent endpoint, and the derived
contrast/loading tensors. Nonfinite values, out-of-bound ideal targets,
`B+d!=G_target`, population mismatch, or incomplete rail coverage invalidate
the run rather than trigger post-hoc clipping or reassignment.

## Required reports

For the ideal endpoint and every assignment, and for every one of the five
P&V repeats at that assignment, report exact `correct/10000`, top-1 accuracy,
teacher agreement, KL, prediction hashes, paired flips, and ideal-to-P&V
accuracy change. Also report:

- target-to-persistent and target-to-apparent conductance residuals both by
  layer and, separately, by requested target level: signed mean, MAE, RMSE,
  p50, p95, and maximum;
- reconstructed DRN contrast error versus the frozen ReLU weights, using only
  full conductances, including relative L2, cosine similarity, sign reversals,
  and erased weights;
- baseline contrast, zero-state loading, `N_down`, `N_up`, zero-only groups,
  effective level counts, index histograms, clipping, and saturation;
- apparent acceptance, persistent reachability, verify-window intersection,
  requested-code/nearest-persistent-code confusion, and adjacent-level
  confusion;
- SET, RESET, total pulse and verify-read counts, reversals, budget exhaustion,
  failure class, and pulse-cost distributions; and
- per-layer loading, `RMS(C)/mean(L)`, and voltage statistics, without making
  an absolute-Siemens or fabricated-array claim.

The continuous-envelope diagnostic must be reported at the same frozen
calibration. Optimizer updates, BPTT, QAT, HWA, retention, drift, inference
read noise, and post-deployment recovery remain exactly absent.

## Decision rule

The ideal gate is a mean held-out
`ideal_bounded_uniform_spacing_init_accuracy` of at least 90%. If no design
passes, the study concludes that none of the tested baseline/spacing pairs is
a viable initialization and reports the best ideal design without selecting a
P&V winner.

Among designs that pass the ideal gate, the practical winner is selected by
the arithmetic mean of all 15 held-out
`pv_persistent_uniform_spacing_init_accuracy` values. At each assignment,
compare the sum of persistent correct counts over its five paired seeds, with
denominator 50,000. A unique winner must have a strictly larger assignment
sum than every other eligible design at each of the three assignments and
lead the runner-up mean over all 150,000 predictions by at least 1.00
percentage point. Otherwise the P&V result is `inconclusive/no unique winner`.
Pulse cost and device-error diagnostics explain this outcome but cannot
overturn it.

Report the best ideal design separately even if it differs from the practical
P&V winner. This prevents a noisy endpoint from hiding deterministic
quantization failure and prevents an ideal-only optimum from being presented
as programmable.

## Exploratory CUDA result under the clipped public range

A direct local CUDA screen completed the full numerical 3-by-3 design on
2026-08-28. It contains 27 complete configurations: three baseline positions,
three spacings, and three held-out OM assignments. Each configuration includes
five matched P&V endpoint seeds, giving 135 complete stochastic endpoints.
Five failed or interrupted attempts are preserved outside the selected
complete bundles. This run did not use the prepared-study/finalization
lifecycle and is therefore `exploratory_noncanonical`, not finalized workflow
evidence. It also cannot answer the unrestricted-positive-conductance question
because every baseline, capacity, target, and circuit endpoint below depends
on the imposed public `[0,1]` ceiling.

For clarity, the `alpha=0` baseline is the higher of the two bounded,
eight-read RESET means in each destination-column pair:

```text
B+ = max(b++, b-+)    for (G++,G-+)
B- = max(b+-, b--)    for (G+-,G--).
```

It is not the largest individual noisy read and is not one maximum over all
four cells. Both cells in a destination pair receive that common baseline.
The cell with the larger commissioned RESET mean is consequently requested at
approximately its RESET baseline when its level index is zero, while its
partner is raised to the same baseline. Nonzero sign-selected rails are then
raised by `n h`.

### Accuracy

All accuracies below are held-out arithmetic means. Continuous and ideal
quantized accuracies average the three hardware assignments. Persistent P&V
accuracy averages the five matched endpoint seeds nested within each of those
three assignments.

| `alpha` | Spacing | Continuous | Ideal quantized | Persistent P&V |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | `1 delta_x` | 94.97% | 94.77% | **78.16%** |
| 0.00 | `2 delta_x` | 94.97% | 93.06% | 76.81% |
| 0.00 | `4 delta_x` | 94.97% | 88.56% | 73.35% |
| 0.25 | `1 delta_x` | 95.04% | **94.88%** | 67.20% |
| 0.25 | `2 delta_x` | 95.04% | 90.40% | 62.40% |
| 0.25 | `4 delta_x` | 95.04% | 79.39% | 58.01% |
| 0.50 | `1 delta_x` | 94.80% | 93.60% | 45.05% |
| 0.50 | `2 delta_x` | 94.80% | 88.59% | 44.80% |
| 0.50 | `4 delta_x` | 94.80% | 56.90% | 27.94% |

Within the clipped contract, the ideal-only optimum is
`alpha=0.25, h=delta_x` at 94.8767%. Its advantage over
`alpha=0, h=delta_x` is only 0.1067 percentage point. Under persistent P&V in
the same clipped contract, the practical optimum is the lowest feasible
baseline and finest tested spacing, `alpha=0, h=delta_x`, at 78.1553%.
Raising the baseline
to `alpha=0.25` loses 10.9533 points and centering it at `alpha=0.5` loses
33.1047 points. The historical `alpha=0, h=4 delta_x` ideal counts are
reproduced exactly at 9098, 8580, and 8889 correct.

### Mechanism and weight error

At `h=delta_x`, increasing `alpha` from 0 to 0.25 to 0.5 raises mean W1
baseline loading from `8.37e-5` to `1.60e-4` to `2.36e-4`, reduces W1 upward
headroom from 0.633 to 0.484 to 0.333, and increases the required fitted gain
from 14.1 to 70.8 to 281.8. Mean P&V cost rises from 7.14 to 9.55 to 13.08
pulses per cell. Thus the shared baseline cancels in the zero-state signed
contrast but remains fully present in denominator loading, while the extra
downward headroom is unused by this one-sided initialization.

For the clipped-screen `alpha=0, h=delta_x` choice, ideal DRN-versus-ReLU
relative-L2 errors are 0.310 for W1 and 0.278 for W2; after persistent P&V
they rise to 0.661 and 0.388. Apparent-endpoint network accuracy is 93.994%
and apparent acceptance is 99.9639%, but persistent accuracy is only 78.1553%
and persistent-window success is 35.4711%. The 15.8387-point
apparent-to-persistent accuracy gap identifies noisy apparent admission of an
incorrect hidden persistent state as the main remaining write-path failure.

### Exploratory interpretation and invalidated carry-forward

Within the capped screen, one-shot initialization favored one baseline per
destination column at `B=L` with `h=delta_x`; moving toward the midpoint
created unused downward headroom while increasing loading and P&V cost. That
mechanism remains useful, but `B=L,h=delta_x` must not be frozen as the
unrestricted deployment choice. Removing the artificial upper ceiling changes
`U`, every baseline with `alpha>0`, active-pair capacity, full-conductance
loading, calibration, and endpoint handoff. The matched 3-by-3 comparison must
therefore be rerun first.

The apparent-to-persistent gap remains a valid diagnostic of the historical
write path, but its proposed controller follow-up was conditional on first
rerunning the matrix with a global nonnegative affine map. That corrected
rerun is reported below. QAT, BPTT, and on-chip recovery remain downstream of
its persistent-code gate.

Detailed local artifacts are in
[`post_run_analysis.md`](../results/mnist-ibm-om-baseline-spacing-pv-exploratory-20260828-v1/analysis/post_run_analysis.md)
and
[`exploratory_summary.json`](../results/mnist-ibm-om-baseline-spacing-pv-exploratory-20260828-v1/analysis/exploratory_summary.json).

## Claim boundary

This study compares nine four-device, no-fixed-reference,
shared-destination initialization designs in one normalized OM preset under an
explicitly clipped public `[0,1]` range. It does not select among those designs
for an unrestricted-positive-conductance embedding. It does not test four
versus eight devices, a fixed intrinsic reference, BPTT,
QAT, HWA, replacement hardware after training, repeated rewrites, or on-chip
recovery. It provides model-based P&V evidence, not raw measured-device or
absolute-conductance evidence.

## Corrected no-clipping exploratory successor

- Result ID:
  `mnist-ibm-om-baseline-spacing-pv-no-clip-exploratory-20260828-v1`
- Experiment ID: `mnist_ibm_om_baseline_spacing_pv_no_clip.v1`
- Lifecycle: `exploratory_noncanonical`; direct local CUDA, not a prepared,
  reviewed, or finalized workflow-managed study
- Source revision: `329c6912c944e00b2f1441af57bab5420cd6d1cd` with the
  run-specific dirty-state fingerprints retained in each native manifest
- Exact configs:
  `examples/mnist_relu_drn/ibm_om_baseline_spacing_pv_no_clip/`
- Frozen source/teacher: `data/mnist_relu_teacher_fixed_init_20260816.pt`,
  SHA-256
  `9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52`

The successor retains raw native support and applies one study-wide pure
translation, without range compression or an upper projection:

```text
x_raw = (a+1)/2
x_origin = -1.3759248719940185
s = 1.10e-4 conductance units per raw-x unit
G = s (x_raw - x_origin).
```

The origin is one `1e-6` raw-`x` margin below the exact frozen repaired-cohort
support floor `-1.3759238719940186`. The exact support ceiling is
`1.8474750518798828`; the resulting study-wide mapped minimum is `1.1e-10`
and the configured mapped ceiling is `0.0003545739916261292`. These are
normalized circuit-embedding values, not a microSiemens calibration. The same
origin and slope are used for every cell, assignment, target, and persistent
endpoint. An out-of-support value invalidates the run; nothing is clipped.

The identity reconstruction, eight-read RESET commissioning, complete-quad
joint repair, assignments `86001/87001/87002/87003`, and five endpoint seeds
per held-out assignment match the clipped predecessor semantically. Each
`alpha` refits one scale pair and positive gain on development assignment
`86001`, then freezes them across spacings, held-out assignments, and endpoint
repeats. All three gains reached the declared upper grid boundary of `1000`,
which is a response-scale limitation but cannot change top-1 predictions.
For `alpha=0`, the refit selected scale fractions `[1.0,0.5]` over
`[1.0,1.0]` by only one development example: 990 versus 989 correct out of
1,024. That weak choice halves the intended W2 contrast while the P&V
contrast-error scale remains about `6.2 uS`. Persistent accuracy therefore
still confounds the physical handoff with a nearly tied calibration choice.

All 27 configurations completed, covering three held-out assignments and five
P&V repeats per configuration: 135 persistent endpoints. Every saved ideal
and persistent full conductance was strictly positive; all targets and
persistent endpoints remained in native support; projection count was zero;
and the apparent verify endpoint was never applied to the DRN. Two earlier
attempts are preserved as pre-numerical CUDA sandbox-access failures. They are
not additional configurations, endpoint results, or CPU executions.

### Corrected 3-by-3 accuracy matrix

Continuous and ideal quantized values are arithmetic means over the three
held-out assignments. Persistent P&V values are arithmetic means over all 15
matched endpoints nested within those assignments.

| `alpha` | Spacing | Continuous | Ideal quantized | Persistent P&V |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | `1 delta_x` | 94.1133% | **93.6967%** | **54.7013%** |
| 0.00 | `2 delta_x` | 94.1133% | 85.3967% | 52.3760% |
| 0.00 | `4 delta_x` | 94.1133% | 68.6033% | 42.4633% |
| 0.25 | `1 delta_x` | 93.9167% | 85.9433% | 27.6213% |
| 0.25 | `2 delta_x` | 93.9167% | 55.5400% | 23.7733% |
| 0.25 | `4 delta_x` | 93.9167% | 15.1933% | 13.0167% |
| 0.50 | `1 delta_x` | 93.2100% | 87.4133% | 30.5787% |
| 0.50 | `2 delta_x` | 93.2100% | 73.7767% | 26.5440% |
| 0.50 | `4 delta_x` | 93.2100% | 25.3533% | 15.4440% |

Only `alpha=0,h=delta_x` clears the 90% ideal gate. Its ideal assignment
range is 93.38--93.92%, and its three assignment-level persistent means are
55.978%, 57.436%, and 50.690%. It is therefore the sole P&V-eligible design in
this matrix and independently reproduces the *nominal* one-sided,
finest-spacing choice. It does not reproduce the clipped result's performance:
the corrected persistent mean is 54.7013%, rather than 78.1553%.

The clipped predecessor's numerical deployment conclusion and its proposed
immediate controller interpretation are consequently **superseded and rejected
for unrestricted positive conductance**. Its measurements remain above as an
exact historical result of the capped `[0,1]` intervention. The fact that both
screens rank `B=L,h=delta_x` first does not rehabilitate the clipped handoff;
the corrected choice is based on a new, physically admissible mapping.

### Corrected weight error and programming diagnosis

For `alpha=0,h=delta_x`, the held-out DRN-versus-ReLU relative-L2 error
averages 0.3388 in W1 and 0.3325 in W2 at the ideal quantized endpoint. Across
the 15 persistent endpoints it rises to 0.6668 and 0.6162. Persistent
requested-code correctness is 49.9727%, persistent-window success is 35.3374%,
and apparent acceptance is 99.9627%; apparent values are controller-side
diagnostics only and no apparent-endpoint network accuracy is reported. Mean
programming cost is 7.106 pulses per cell and 0.0373% of cells exhaust the
128-pulse budget.

The affine translation itself adds `151.35 uS` per cell, or `605.41 uS` per
four-cell quad, in the configured conductance embedding. Mean selected-arm
zero-state loading is `676.94/679.09 uS` in W1/W2, versus `83.68/84.95 uS`
under the clipped handoff. These numerical micro-units describe the declared
embedding, not a fabricated-device calibration. The baseline still cancels
exactly from signed zero but remains in every denominator.

By contrast, the raw programming diagnostics barely move: persistent-window
success is 35.3374% versus 35.4711% previously, apparent acceptance is
99.9627% versus 99.9639%, mean cost is 7.106 versus about 7.14 pulses per cell,
and P&V contrast error remains about `6.2 uS`. The large persistent-accuracy
change is therefore not evidence for a suddenly worse pulse plant. It is
consistent with retaining the physical common-mode loading while the weak
`[1.0,0.5]` refit also halves W2 target contrast.

The selected design fails the later persistent-code progression criterion:
approximately half of persistent states, not at least 90%, resolve to the
requested nearest code. It is not yet eligible for QAT, HWA, BPTT, or on-chip
recovery. The immediate diagnostic is one fixed
`alpha=0,h=delta_x,[1.0,1.0]` arm across all three held-out assignments, using
the same five endpoint seeds and corrected affine loading. Only after that
matched scale control should the investigation attribute the remaining loss to
the controller/state estimator or change code spacing.

Raw native bundles, manifests, configs, and scientific summaries are under
`results/mnist-ibm-om-baseline-spacing-pv-no-clip-exploratory-20260828-v1/`.

### Corrected claim boundary

This successor compares nine four-device, no-fixed-reference,
shared-destination initialization designs in a normalized AIHWKit 1.1.0 OM
preset. It is complete exploratory coverage, not workflow-reviewed or
finalized evidence. It uses counterfactually repaired model identities rather
than raw measured conductance traces or fabricated arrays and supplies no
absolute-conductance calibration. It does not test four versus eight devices,
a fixed intrinsic reference, inference read noise, retention, drift, repeated
logical rewrites, BPTT, QAT, HWA, replacement-hardware transfer, or on-chip
recovery.

## Nominal-bound-Winsorized IBM OM counterfactual

- Result ID:
  `mnist-ibm-om-baseline-spacing-pv-winsorized-nominal-exploratory-20260828-v2`
- Experiment ID:
  `mnist_ibm_om_baseline_spacing_pv_truncated_nominal.v1`
- Lifecycle: `exploratory_noncanonical`; direct local CUDA, not a prepared,
  reviewed, or finalized workflow-managed study
- Exact configs:
  `examples/mnist_relu_drn/ibm_om_baseline_spacing_pv_truncated_nominal/`
- Frozen source/teacher: `data/mnist_relu_teacher_fixed_init_20260816.pt`,
  SHA-256
  `9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52`

This run tests the proposed normalized workaround, but it must not be called
the default IBM OM model. AIHWKit 1.1.0 declares nominal means
`w_min=-1`, `w_max=1`, with device-to-device variations
`w_min_dtod=0.5695` and `w_max_dtod=0.3499`. Its C++ population code samples
each cell's two bounds with independent Gaussian factors and, under
`enforce_consistency`, forces only their signs and ordering. It does **not**
cap the sampled values at `-1` and `+1`. AIHWKit also documents these as
abstract signed weight bounds which may physically be implemented by the
difference of two resistive elements. The OM preset is a
`SoftBoundsReferenceDevice` with symmetry-point subtraction enabled, so it
does not supply a unique absolute single-device conductance origin for the
passive DRN.

The counterfactual therefore freezes the same repaired identities first and
then Winsorizes their sampled pulse bounds:

```text
a_min' = max(a_min, -1)
a_max' = min(a_max, +1)
x = (a+1)/2
G = a+1 = 2x.
```

The source audit is reproducible from the official
[`ReRamArrayOMPresetDevice` definition](https://aihwkit.readthedocs.io/en/latest/_modules/aihwkit/simulator/presets/devices.html#ReRamArrayOMPresetDevice),
the official
[`PulsedDevice`/`SoftBoundsReferenceDevice` documentation](https://aihwkit.readthedocs.io/en/latest/_modules/aihwkit/simulator/configs/devices.html#SoftBoundsReferenceDevice),
and the AIHWKit 1.1.0
[`PulsedRPUDevice::populate` implementation](https://github.com/IBM/aihwkit/blob/v1.1.0/src/rpucuda/rpu_pulsed_device.cpp#L593-L640).

This is Winsorization, not truncation by rejection or resampling. It creates
point masses at `a=-1` and `a=+1` and changes the soft-bound pulse response of
affected cells. After transforming the bounds, the runtime repeats the full
eight RESET/read commissioning sequence and runs P&V against the transformed
pulse plant. Persistent endpoints are deployed directly through `G=a+1`;
there is no circuit-handoff clipping and the apparent verify endpoint remains
controller-side only.

For each destination pair, the requested zero is the larger of its two fresh
commissioned RESET estimates. Four pairs among 238,200 unique held-out pairs
placed that noisy request just above their partner's exact SET ceiling. Those
four requested baselines were moved downward once into the exact common
support before quantization or P&V; the largest movement was `0.034773` in
raw `x`. This target-construction adjustment is recorded separately from
endpoint projection. There were zero target-handoff projections, zero
persistent-endpoint projections, and zero persistent support violations.

### Coverage and Winsorization magnitude

All 27 configurations completed, covering nine designs, three held-out
assignments, and five P&V endpoint seeds per assignment: 135 persistent
deployments. The held-out sampled-bound changes were large:

| Assignment | Lower bound Winsorized | Upper bound Winsorized | Either | Both |
| ---: | ---: | ---: | ---: | ---: |
| 87001 | 50.13% | 49.97% | 75.11% | 24.99% |
| 87002 | 50.33% | 49.70% | 75.02% | 25.01% |
| 87003 | 50.39% | 50.02% | 75.21% | 25.20% |

The intervention is consequently not a tail-only cleanup. Roughly three
quarters of identities have at least one changed bound, and about one quarter
have both bounds changed.

### Winsorized 3-by-3 accuracy matrix

Continuous and ideal values are arithmetic means over the three held-out
assignments. Persistent P&V is the arithmetic mean over all 15 matched
endpoints in each design.

| `alpha` | Spacing | Continuous | Ideal quantized | Persistent P&V |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | `1 delta_x` | 94.9800% | 94.5267% | 74.7913% |
| 0.00 | `2 delta_x` | 94.9800% | 93.0933% | **75.7860%** |
| 0.00 | `4 delta_x` | 94.9800% | 87.8867% | 71.7620% |
| 0.25 | `1 delta_x` | 95.0667% | **94.9933%** | 67.9993% |
| 0.25 | `2 delta_x` | 95.0667% | 88.7467% | 58.6040% |
| 0.25 | `4 delta_x` | 95.0667% | 77.6333% | 54.9020% |
| 0.50 | `1 delta_x` | 94.6900% | 22.7767% | 11.9080% |
| 0.50 | `2 delta_x` | 94.6900% | 12.7867% | 10.3027% |
| 0.50 | `4 delta_x` | 94.6900% | 13.1533% | 8.9993% |

The best ideal-mapped result is `alpha=0.25,h=delta_x` at 94.9933%.
The best observed persistent result is instead `alpha=0,h=2 delta_x` at
75.7860%. Its assignment-level persistent range is 74.444--76.504%. This is
an approximately 21-point recovery over the global-extreme affine stress
test's best 54.7013%, but it remains about one point below the historical
public-range `alpha=0,h=2 delta_x` control and is not usable deployment
accuracy. The apparent one-point lead over `h=delta_x` is not a robust spacing
selection: it wins only 9 of 15 paired endpoint seeds, its paired deltas range
from -13.34 to +14.19 points, and a naive endpoint-level standard error is
1.72 points.

The `alpha=0` spacing tradeoff is now informative. Moving from one to two
`delta_x` raises requested-code correctness from 49.63% to 71.26% while the
persistent raw-`x` RMSE stays essentially fixed at 0.05533 versus 0.05523.
That robustness gain narrowly outweighs the ideal-resolution loss from
94.53% to 93.09%. At four `delta_x`, code correctness reaches 93.87%, but
ideal accuracy falls to 87.89%; the lost weight resolution dominates. Thus a
larger spacing can make the *code label* more reliable while making the
network worse.

The midpoint rows require a separate calibration caveat. `alpha=0` and
`alpha=0.25` both selected `[1.0,1.0]` layer fractions, with fitted gains
14.125 and 70.795. `alpha=0.5` selected `[1.0,0.125]` by only one development
example over several larger-W2 candidates and hit the gain ceiling of 1000.
That choice erases 93.33% of W2 at one-delta spacing and 99.67--100% at the
larger spacings, even though W2 has ample physical capacity. The 9--23% midpoint
quantized accuracies therefore combine high-baseline loading with an unstable
continuous-only scale selection; they are not a clean baseline-position
effect.

### Full-conductance and programming diagnosis

For `alpha=0,h=delta_x`, mean W1/W2 baseline quad loading is 1.598/1.619 in
the declared normalized full-`G` units, and the ideal full-load means are
1.856/2.229. Persistent RMS contrast divided by mean load is 0.0969/0.1624.
These ratios and voltage levels are close to the historical capped screen and
far healthier than the global-extreme affine stress test, where the same
ratios fell to about 0.016/0.017. The recovery therefore comes from removing
the extreme common-mode translation, not from subtracting the baseline: every
complete `G` remains in both signed transfer and denominator loading.

P&V remains the limiting stage. In the persistent winner
`alpha=0,h=2 delta_x`, apparent acceptance is 99.955%, persistent-window
success is 35.98%, requested-code correctness is 71.26%, and mean cost is
6.054 pulses per cell. Persistent target RMSE is 0.05523 raw `x`, nearly the
same absolute error as at one and four spacings. At one `delta_x`, ideal
DRN-versus-ReLU relative-L2 error is 0.306/0.270 in W1/W2 and rises after P&V
to 0.689/0.395. At two `delta_x`, the corresponding ideal errors are
0.417/0.299 and persistent errors are 0.725/0.409. The network-level
programming gap is therefore not explained by endpoint clipping or stuck
cells; it is the interaction of finite persistent write error, quantization,
and full-conductance circuit sensitivity.

### Comparison and interpretation

For `alpha=0,h=delta_x`, this counterfactual gives 94.53% ideal and 74.79%
persistent accuracy. The historical public-range run gave 94.77%/78.16%,
while the global-extreme affine no-clipping stress test gave 93.70%/54.70%.
The Winsorized result shows that the catastrophic global-affine loss was
largely caused by treating the most negative abstract OM weight bound as a
literal absolute-conductance origin. It does **not** show that IBM intended
hard clipping at nominal `[-1,1]`, nor that this is a fabricated-device
model: default AIHWKit samples beyond those nominal values, and this control
changes the bound and update distribution for most cells.

The practical conclusion is two-part:

1. In a deliberately normalized single-device `w=a` model, bound
   Winsorization plus `G=a+1` is a coherent no-post-handoff-clipping
   sensitivity control and preserves high ideal-mapped accuracy.
2. It still leaves persistent deployment near 76%, below the 90% progression
   criterion, so P&V/controller work remains necessary before QAT, HWA, BPTT,
   or on-chip recovery is informative.

The full local analysis is in
[`post_run_analysis.md`](../results/mnist-ibm-om-baseline-spacing-pv-winsorized-nominal-exploratory-20260828-v2/analysis/post_run_analysis.md),
[`exploratory_summary.json`](../results/mnist-ibm-om-baseline-spacing-pv-winsorized-nominal-exploratory-20260828-v2/analysis/exploratory_summary.json),
and
[`accuracy_by_assignment.csv`](../results/mnist-ibm-om-baseline-spacing-pv-winsorized-nominal-exploratory-20260828-v2/analysis/accuracy_by_assignment.csv).

### Winsorized-control claim boundary

This is a complete exploratory model-based sensitivity screen, not default
IBM behavior, measured-device evidence, or an absolute conductance
calibration. It uses counterfactually repaired identities, analyst-imposed
bound Winsorization, one-pulse P&V, three assignments and five endpoint seeds.
It excludes inference read noise, retention, drift, repeated logical writes,
QAT, HWA, BPTT, replacement-hardware transfer after training, and on-chip
recovery.

## Multi-assignment deterministic QAT exploratory follow-up

Evidence tier: `exploratory_noncanonical`. This follow-up deliberately tests
whether deterministic codebook adaptation can make the nominal-bound-
Winsorized control less assignment-specific. It is not default IBM behavior,
measured-device evidence, or a workflow-reviewed study.

One normalized FP32 logical master was trained for ten epochs while the
forward codebook alternated, minibatch by minibatch, between Winsorized
assignments 86001 and 87001. Assignment 87002 was reserved for fixed-epoch-10
spacing selection, and assignment 87003 was excluded until all epoch-10
checkpoints had been written. Training used the exact full-conductance
`G=a+1=2x` mapper and a deterministic straight-through codebook forward. It
did **not** sample P&V or read noise during minibatch training. The fixed
hardware templates still inherit their one frozen eight-read RESET
commissioning receipts, which were generated with the preset's stochastic
commissioning/read process before QAT.

Spacing was selected only by assignment-87002 correct count at the declared
epoch 10, with lower KL as the tie-breaker:

| Spacing | 87002 epoch 0 | 87002 epoch 10 | Change | Epoch-10 KL |
| ---: | ---: | ---: | ---: | ---: |
| `1 delta_x` | 4710/5000 (94.20%) | **4671/5000 (93.42%)** | -0.78 pp | 0.122207 |
| `2 delta_x` | 4548/5000 (90.96%) | 4598/5000 (91.96%) | +1.00 pp | 0.181462 |
| `4 delta_x` | 4128/5000 (82.56%) | 4280/5000 (85.60%) | +3.04 pp | 0.450342 |

The selected one-delta checkpoint is
`a070d0369a93201ac3abd4e36019cc1d66c1a8cf0a325ccdda5439b08cded75b`.
The diagnostic best epoch was never substituted for the fixed epoch-10
headline.

After selection was frozen, each spacing was mapped onto assignment 87003.
Five stochastic persistent P&V deployments used matched endpoint seeds
89301--89305:

| Spacing | Ideal epoch 0 | Ideal epoch 10 | P&V epoch 0 mean | P&V epoch 10 mean | P&V change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1 delta_x` | 94.88% | **94.47%** | 68.998% | **70.380%** | +1.382 pp |
| `2 delta_x` | 94.14% | 92.67% | 76.504% | 77.122% | +0.618 pp |
| `4 delta_x` | 89.19% | 86.20% | 71.892% | 69.524% | -2.368 pp |

For the selected one-delta arm, the five epoch-10 endpoint accuracies were
58.90%, 75.61%, 72.96%, 64.29%, and 80.14%; their arithmetic mean is 70.38%.
Here, a **P&V mean** is the mean over independent stochastic programming and
verify streams on the same logical target and physical assignment. It is not
the mean over five independently sampled arrays. P&V includes stochastic
pulse response and noisy apparent verify observations, while network accuracy
uses the saved persistent endpoint with no additional inference-read noise.

The deterministic QAT result is modest and assignment-sensitive. It reduces
ideal accuracy on 87003 for every spacing and improves the selected arm's P&V
mean by only 1.382 points. Although two-delta has the largest observed 87003
P&V mean, choosing it after seeing that target would violate the frozen
selection rule. The result therefore selects one-delta for the downstream
recovery control and does not establish broad replacement-array recovery.

The verified local analysis is in
[`report.md`](../results/mnist-ibm-om-winsorized-multi-assignment-qat-exploratory-20260829-v1/analysis/report.md)
and
[`summary.json`](../results/mnist-ibm-om-winsorized-multi-assignment-qat-exploratory-20260829-v1/analysis/summary.json).

## Persistent pulse-mediated Adam exploratory follow-up

The selected one-delta epoch-10 checkpoint was then programmed once onto
assignment 87003 with endpoint seed 89301. This named persistent starting
state, `P0`, was saved with its apparent state and RNG continuation and cloned
byte-for-byte for every recovery arm. There was no target remapping after
`P0`, and no floating-point weight shadow replaced the persistent cell state.
Its SHA-256 is
`c807b3f1f08b140473a07a3f438594e9ae742750bcb8c0f53c57a885d1cb0b0e`.

This `P0` is a newly programmed realization, not a reload of the nominally
same 89301 repeat in the preceding five-seed QAT table. The recovery plant
preallocated 385 random draws per cell so its trajectory could continue,
whereas the QAT evaluator preallocated 257. CPU `torch.randn` streams are
shape-dependent, so the same nominal seed does not make the shorter vector a
prefix of the longer vector. The exact saved `P0` state, rather than the seed
label alone, is therefore the authoritative recovery initializer.

The experiment is accurately described as **hardware-in-loop pulse-mediated
Adam**, not fully autonomous on-chip Adam. Forward inference always used the
plant's current persistent full conductances. BPTT, teacher logits, Adam
moments, and the Bernoulli pulse-selection RNG remained digital; the only
weight mutation after `P0` was a capability-limited SET or RESET pulse applied
to the persistent plant.

Technique and learning rate were selected on 5,000 validation examples before
the 10,000-example test labels were opened:

| Arm | Validation | Pulses | Pulses/cell | Changed cells | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen `P0` | 2910/5000 (58.20%) | 0 | 0 | 0 | 5920/10000 (59.20%) |
| Direct rail, `1e-4` | 4636/5000 (92.72%) | 186,489 | 1.174 | 105,931 | not opened |
| Coordinated contrast, `1e-4` | 4530/5000 (90.60%) | 206,512 | 1.300 | 111,277 | not opened |
| **Direct rail, `3e-5`** | **4674/5000 (93.48%)** | **54,834** | **0.345** | **44,398** | **9414/10000 (94.14%)** |
| Direct rail, `3e-4` | 4643/5000 (92.86%) | 512,568 | 3.228 | 148,059 | not opened |

The selected arm recovers **34.94 percentage points**, from 59.20% to 94.14%,
and finishes only 0.33 point below the corresponding deterministic ideal-
mapped accuracy of 94.47%. It used 27,217 SET and 27,617 RESET pulses, touched
27.96% of cells, required at most six pulses on any cell, and hit the 64-pulse
recovery cap zero times. The full-`G` quad-mean loading-change bias/RMS was
0.02767/0.06721, while signed-contrast change bias/RMS was
-0.000085/0.12108. The selected recovery checkpoint SHA-256 is
`13c7286725f537e2f296fcde24f9b492295a1650574373a2425cb66585f69c96`.

The coordinated four-rail contrast arm balanced pulse counts exactly, but it
did not preserve common loading: pulse increments are state-dependent,
asymmetric, and bound-limited. It used more pulses and produced larger loading
and contrast drift than independent rail control. The low-rate direct arm was
therefore both the most accurate and the least invasive trained arm.

This recovery **does include device-update noise**. `P0` was created by
stochastic pulse programming stopped by noisy apparent verify observations,
and Adam continued the same stochastic persistent pulse plant. The recovery
updates themselves were open-loop pulse decisions; Adam did not perform an
apparent verify-and-correct loop after every update pulse. The experiment does
not include additional read noise during final inference, retention, drift,
aging, or measured-array variability. The 94.14% result is one exact
assignment-87003/seed-89301 trajectory, not a five-seed P&V mean. It supports
recoverability of that named simulated state only; replication from multiple
predeclared persistent endpoints is required before a robust same-hardware or
on-chip-recovery claim.

The complete mechanism report is in
[`post_run_analysis.md`](../results/mnist-ibm-om-winsorized-onchip-adam-exploratory-20260829-v1/analysis/post_run_analysis.md)
and the machine summary is in
[`summary.json`](../results/mnist-ibm-om-winsorized-onchip-adam-exploratory-20260829-v1/analysis/summary.json).
