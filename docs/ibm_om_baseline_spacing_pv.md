# IBM OM shared-destination baseline, spacing, and P&V study

- Study ID:
  `mnist-ibm-om-shared-destination-baseline-spacing-pv-20260828-v1`
- Experiment ID: `mnist_ibm_om_baseline_spacing_pv.v1`
- Date frozen: 2026-08-28
- Evidence class: `model_based_aihwkit_preset`
- Device model: AIHWKit 1.1.0 `ReRamArrayOMPresetDevice`
- Execution: validate-only CUDA for production and the excluded smoke canary;
  no CPU path
- Status: frozen workflow-managed contract; no run is evidence until all
  declared coverage and the fail-closed analysis are complete

## Question

For the four-device, no-fixed-reference topology selected by the completed
baseline study, how should the shared destination-column baseline and uniform
conductance-level spacing be chosen when both ideal quantization and noisy
program-and-verify are considered?

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
RESET mean of cell `i` and `u_i` its sampled upper conductance bound. Define

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

## Claim boundary

This study selects among nine four-device, no-fixed-reference,
shared-destination initialization designs in one normalized OM preset. It
does not test four versus eight devices, a fixed intrinsic reference, BPTT,
QAT, HWA, replacement hardware after training, repeated rewrites, or on-chip
recovery. It provides model-based P&V evidence, not raw measured-device or
absolute-conductance evidence.
