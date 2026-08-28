# IBM OM four-device baseline-selection study

- Study ID: `mnist-ibm-om-four-device-baseline-selection-20260828-v1`
- Experiment ID: `mnist_ibm_om_baseline_selection.v1`
- Date frozen: 2026-08-28
- Evidence class: `model_based_aihwkit_preset`
- Device model: AIHWKit 1.1.0 `ReRamArrayOMPresetDevice`
- Status: implementation and preflight in progress; no result is evidence until
  every declared native run and the fail-closed study analysis complete

## Question

For a four-device dual-rail quad, which physically valid baseline rule makes
logical weight zero behave most like zero before any optimizer update?

This study changes the baseline policy only. It does not choose the final
level spacing or level count. The primary target is therefore continuous
within the exact active-pair headroom. A four-delta nearest-level map is
reported from the same selected calibration as a secondary diagnostic.

## Accuracy definitions

The primary metric is
`ibm_om.ideal_bounded_continuous_init.v1`, reported as
`ideal_bounded_continuous_init_accuracy`. It is exact top-1 `correct/10000`
on an untouched held-out OM assignment after bounded deterministic mapping of
the frozen pre-BPTT ReLU checkpoint. It includes the frozen error of the
eight-read RESET commissioning step, but excludes BPTT/QAT, HWA, stochastic
target writes, P&V, inference read noise, retention, and drift.

The secondary metric is
`ibm_om.ideal_bounded_standard4delta_init.v1`, reported as
`ideal_bounded_standard4delta_init_accuracy`. Adjacent target levels are
exactly `4 * (nominal_dw_min/2)` apart in `x=(a+1)/2`; half-step ties round
away from zero and indices are capped by the exact active-pair headroom. It
uses the scale pair and logit gain selected by the continuous development
map and is never independently refit.

The other two deployment-ladder metrics remain reserved for later studies:
`same_hardware_bptt_accuracy` and
`cross_hardware_deployment_accuracy`. Neither is measured here.

## Full-conductance invariant

Every rail is materialized and evaluated as

```text
G_i = B_i + d_i.
```

No baseline is removed before solving the circuit. For canonical quad order
`(++,+-,-+,--)`, the represented contrast and total loading are derived only
from the full conductances:

```text
C(G) = (G++ - G+- - G-+ + G--)/2,
L(G) = G++ + G+- + G-+ + G--.
```

The physical DRN tensor containing every `G_i` enters both numerator and
denominator KCL terms. Algebraic cancellation at zero is a checked property
of a policy, never an implementation shortcut.

## Frozen hardware and source

- Frozen logical source: `data/mnist_relu_teacher_fixed_init_20260816.pt`,
  SHA-256
  `9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52`.
- Development assignment: `86001`.
- Untouched held-out assignments: `87001`, `87002`, and `87003`.
- Corrupt published cells are replaced by the existing
  `counterfactual_repaired` OM sampler before the baseline intervention.
- Each sampled cell is commissioned by eight sequential cycles of one RESET
  pulse followed by one apparent raw-`a` read, without reinitialization. The
  arithmetic mean is converted to `x` and clipped only to that cell's public
  and sampled bounds.

One deterministic joint repair is constructed per assignment before any
calibration or accuracy evaluation. A destination quad is retained only if it
is feasible under all four policies. Otherwise the complete four-identity
quad and its matching commissioning observations are replaced by the first
feasible candidate in its private, disjoint 128-candidate donor block. The
same repaired assignment and hardware-instance fingerprint must appear in
all four policy runs. Donor exhaustion invalidates the assignment; no weight
may be dropped and no policy may consume its own donor stream.

## Baseline policies

Let `b_i` be a cell's bounded commissioned RESET mean, and let `l_i,u_i` be
its mapped bounds.

1. `independent_cell_reset_mean`: each rail uses `B_i=b_i`. This is the
   matched heterogeneous-origin control and is not required to have zero
   baseline contrast.
2. `shared_quad_reset_max`: all four rails use
   `B_Q=max_i b_i`, provided that value lies within all four cells' bounds.
3. `shared_destination_columns_reset_max`: the two rails contributing to
   each output numerator share a baseline:
   `B++=B-+=max(b++,b-+)` and
   `B+-=B--=max(b+-,b--)`. These are destination-column groups, not the
   historical source-row pairing. Their zero contrast is exact even if the
   two column baselines differ.
4. `reference_enforced_destination_columns`: two cells are fixed intrinsic
   references and two are tunable active cells. For a positive logical
   weight, `-+` and `+-` are references and the baselines of `++` and `--`
   equal their respective partner references. Negative weights swap active
   and reference roles. A sampled `r` must lie unmodified in both its own
   cell's and its partner's bounds and in the public coordinate; otherwise
   the joint donor rule applies. Zero ties use the positive role convention.

The first three policies use the same commissioned observations. The
reference-enforced policy generates and records the same commissioning
receipts for matched provenance but does not use RESET means to construct its
targets.

## Mapping and calibration

For logical `U=W/max|W|`, only the two sign-selected rails are raised. The
continuous magnitude is

```text
d = f_layer * H_active_pair * |U|,
```

where `H_active_pair` is the smaller positive headroom of the two rails that
encode that sign. Inactive-cell upper bounds do not reduce this envelope.
The development assignment searches all 16 layer-scale pairs from
`{0.125,0.25,0.5,1.0}^2`. Each policy selects continuous calibration accuracy
first, calibrated KL second, and the lexicographically smallest pair last;
its positive logit gain is fitted on the calibration subset. The selected
pair and gain are frozen for the held-out assignment and for the four-delta
diagnostic.

## Weight-error report

The study reports the difference between mapped DRN contrasts and the source
ReLU weights, not only classification accuracy. On development data only,
each layer fits the zero-intercept analysis scale

```text
s = <U, C_continuous-C_baseline> / <U,U>,   s > 0.
```

This scale is frozen for held-out reporting and never changes the circuit or
accuracy. With `A=max|W|`, the reconstructed teacher-unit weight is
`W_hat=A*C/s`. Reports separate baseline, continuous-headroom, four-delta
quantization, and total errors and include signed mean, MAE, RMSE, p50, p95,
maximum absolute error, relative L2, cosine similarity, sign reversals, and
erased weights per layer. Raw model conductance units and normalized
coordinates remain clearly labelled; no absolute-Siemens claim is made.

At synthetic `d=0`, the study additionally reports `C(B)/h`, full quad
loading, and the solved rail-voltage residual. Exact zero contrast is a
validity invariant for the shared-quad, destination-column, and
reference-enforced policies. It is a diagnostic, not a second winner metric.

## Decision rule

A policy is the baseline winner only if it has a strictly larger continuous
correct count than every other policy on each of the three paired held-out
assignments and its arithmetic-mean continuous accuracy exceeds the
runner-up mean by at least `0.0100`. Otherwise the result is
`inconclusive/no winner`. Separately, a winner passes the initialization gate
only at a held-out mean of at least 90%. Four-delta accuracy and weight-error
statistics explain the result but cannot overturn this predeclared rule.

## Prior evidence and what remains new

| Prior pre-training control | Ideal result | Why it is not this matched test |
| --- | ---: | --- |
| Homogeneous four-device no-`r`; scalar fixed `r` | 96.91%; 97.36--97.38% | No sampled identities, per-cell bounds, commissioning, or feasibility problem. |
| Raw-p90 requested shared quad | 91.11% | Seven symmetric levels, global `D90`, and about 10% deliberately unsupported quads. |
| Historical shared RESET-relative | 94.90% | Effective `(a-r+1)/2` coordinate; separate physical `r` and its loading were absent. |
| Native deterministic codebook, independent origins | no-`r` 35.98% (41.02% continuous); fixed-`r` 57.85% (64.35% continuous) | No grouped zero and a different state-dependent codebook. |
| Exact-lower four-delta | no-`r` 31.48% (32.60% continuous); fixed-`r` 23.03% (71.11% continuous) | Heterogeneous per-cell origins/references remain in zero contrast. |
| Per-cell RESET-mean four-delta | no-`r` 25.76% (27.23% continuous); fixed-`r` 23.03% (71.11% continuous) | Direct matched evidence that independent origins fail; no grouping. |

One requested baseline per four-cell quad has therefore been tested only
under unmatched historical codebooks. The exact destination-column grouping
has never been tested in the OM model. Heterogeneous intrinsic references
have been tested, but not the sign-selected partner-reference rule that
enforces zero exactly. Those two interventions, and their matched comparison
against independent and quad-shared origins, are the new evidence in this
study.

## Workflow and completion

The tracked study declares four arms and three native `ebl validate` configs
per arm. Every production config requires CUDA and the launcher fails closed
unless the task interpreter exposes a CUDA GPU. The excluded disposable smoke
config also requires CUDA; this experiment has no CPU execution config. Raw runs live below
`results/mnist-ibm-om-four-device-baseline-selection-20260828-v1/runs/`.
A fail-closed analyzer must verify every RunStore bundle, source and config
hash, population and commissioning receipt, joint hardware fingerprint,
`B+d=G` decomposition, bounds/zero gates, selected calibration, prediction
hash, exact count, and cross-arm matching before producing the held-out table
and winner decision. Failed attempts are retained. No interpretation enters
`docs/experimental_manifest.md` until the workflow closeout review is
explicitly completed with the user.
