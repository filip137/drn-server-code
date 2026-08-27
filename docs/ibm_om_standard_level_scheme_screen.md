# IBM OM RESET-anchored and reference-centered standard-level screen

- Status: executable and focused tests complete; native screen not launched
- Screen ID: `mnist-ibm-om-standard-level-scheme-screen-20260827-v1`
- Evidence class: hardware-derived fitted AIHWKit model
- Source: AIHWKit 1.1.0 `ReRamArrayOMPresetDevice`

## Question

The completed shared-calibration bounded-codebook screen mixed an
initial-RESET calibration with every reference policy and used each cell's
dense deterministic pulse trajectory as its quantizer. This follow-up asks a
different, scheme-optimized question:

> What ideal mapped accuracy can each four/eight-device, with/without-`r`
> representation attain when its baseline is physically appropriate and all
> arms use the same standard conductance-level spacing?

This remains an ideal mapping gate. It includes sampled OM identities, bounds,
references, and nominal resolution, but excludes write noise, stochastic pulse
trajectories, program-and-verify, HWA, optimizer updates, and on-chip recovery.

## Common level definition

AIHWKit expresses the nominal minimum update as `dw_min` in native
`a in [-1,1]` coordinates. The repository uses

```text
x = clip((a + 1) / 2, 0, 1),
```

so the same increment is

```text
delta_a = dw_min,
delta_x = dw_min / 2.
```

Every adjacent standard target is exactly four such increments apart:

```text
h_a = 4 delta_a,
h_x = 4 delta_x.
```

This is a conductance-coordinate separation, not a claim that four pulses
produce a constant displacement. The OM soft-bounds pulse response is
state-dependent.

## Baseline and level-count rules

| Reference policy | Frozen origin | Active levels used for mapping |
| --- | --- | --- |
| Without fixed `r` | the identity's sampled RESET/lower state `B_i` | `a_i(m) = B_i + m h`, for whole in-bound levels `m = 0,...,M_i` |
| With fixed `r` | the identity's exact sampled `r_i` | `a_i(m) = r_i + m h`, using whole in-bound levels and never projecting or changing `r_i` |

For a fixed-`r` identity whose zero-offset active state is outside its active
bounds, the fixed reference remains exactly `r_i`; only `a_i` moves to the
nearest in-bound integer level of the grid centered on `r_i`. This residual
`a_i-r_i` is retained and reported. If an identity has no whole grid level
inside its active range, it retains exact `r_i` as an explicitly reported
zero-only structural case. The screen never invents a shorter terminal level.

The implementation also records the available negative levels below `r_i`,
but logical sign remains encoded by dual-rail placement. The negative half is
therefore headroom, not a second set of logical sign codes. For each logical
quad, the usable positive step count is the minimum capacity of its four
potentially sign-selected cells. A zero-capacity quad remains a zero-only
target and is reported; this first control performs no donor reassignment.

This policy makes the standard logical offsets identical across schemes while
allowing the number of realizable offsets to expose the range cost of RESET
versus centered placement.

## Four- and eight-device circuit rules

For the four-device fixed-reference arm, a positive weight uses

```text
[[a, r],
 [r, a]],
```

and a negative weight swaps `a` and `r`. Thus transfer contains `a-r`, while
the physical row/column loading contains `a+r`. A zero quantized magnitude
uses `r`, not an independently projected active zero.

For an eight-device edge, the active and second branches remain separate:

```text
D = G_a - G_r,
S = G_a + G_r.
```

`D` supplies signed transfer and `S` supplies voltage-denominator loading.
The implementation never places `a-r` in a conductance denominator.

## Per-scheme calibration

Assignment `86001` is the development assignment. Each of the four schemes
independently searches the same Cartesian grid of layer scale fractions
`[0.125, 0.25, 0.5, 1.0]^2` after standard-level quantization. Selection
maximizes mapped calibration accuracy, then minimizes fitted KL, then uses the
lexicographically smaller scale pair. The positive output gain is fitted per
scheme for KL; it cannot change top-1 predictions.

The selected scale pair and gain for each scheme are frozen before evaluation
on assignments `87001`, `87002`, and `87003`. This is explicitly a
scheme-optimized comparison and supersedes the shared-RESET rule for this new
question; it must not be described as the earlier shared-calibration causal
screen.

For the selected scale pair, a continuous target inside the same per-quad
standard-level envelope is retained as a diagnostic. Its difference from the
uniform-level result isolates logical quantization, not programming noise.

## Required output

The executable records the source/config hashes, sampled population receipts,
exact `r`, RESET origins, integer grid indices, positive and negative level
capacities, zero-reference residuals, target hashes, selected scale/gain,
accuracy, teacher agreement, KL, prediction flips, logical sign flips,
`RMS(D)/mean(S)`, conductance-sum loading, voltage statistics, and a normalized
total-conductance proxy.

The implementation and strict config are:

- `experiments/mnist_relu_drn/ibm_om_standard_level_scheme_screen.py`;
- `examples/mnist_relu_drn/ibm_om_standard_level_scheme_screen/screen.json`.

The frozen source inputs are the bias-free teacher with SHA-256
`9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52`
and
`examples/mnist_relu_drn/ibm_om_differential_pair_hwa_pilot/clean.json` with
SHA-256
`59c0a6a2c324dfea6bce8ec982dc9cf15286a6f577bace31471c4f646ebd1cad`.

## Claim boundary

The OM preset is a normalized fitted model, not raw measured-device replay and
not an absolute conductance calibration. A completed result can compare the
declared model-relative schemes, but cannot establish fabricated-array
accuracy, write success, physical power, or a need for on-chip training.
