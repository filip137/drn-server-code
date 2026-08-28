# IBM OM per-cell RESET-commissioned and reference-centered standard-level screen

- Status: exact-lower-bound v1 diagnostic and corrected per-cell RESET-mean
  v2 production screen/replay complete; artifact-verified post-run analysis
  complete
- Screen IDs:
  `mnist-ibm-om-standard-level-scheme-screen-20260827-v1` and
  `mnist-ibm-om-standard-level-reset-mean-scheme-screen-20260827-v2`
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

This remains an ideal deployment gate. It includes sampled OM identities,
bounds, references, and nominal resolution. Version 2 additionally enables
the fitted cycle-to-cycle and apparent-write terms only while commissioning
the no-`r` baseline. The mapped level deployment remains deterministic and
excludes write noise, program-and-verify, HWA, optimizer updates, and on-chip
recovery.

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

## Completed v1/v2 baseline and capacity rules

These completed screens bundled baseline placement with the mechanically
resulting bounded capacity. Stage 0E changes only the baseline policy while
retaining four-delta spacing and the existing whole-level capacity algorithm
as frozen probes; spacing and level-count selection are subsequent technical
decisions.

| Reference policy | Frozen origin | Active levels used for mapping |
| --- | --- | --- |
| Without fixed `r`, v1 diagnostic | the identity's exact sampled lower state `B_i` | `a_i(m) = B_i + m h`, for whole in-bound levels `m = 0,...,M_i` |
| Without fixed `r`, v2 primary | the bounded per-cell mean `B_i` from eight apparent raw-`a` RESET/read observations | `a_i(m) = B_i + m h`, for whole in-bound levels `m = 0,...,M_i` |
| With fixed `r`, unchanged | the identity's exact sampled `r_i` | `a_i(m) = r_i + m h`, using whole in-bound levels and never projecting or changing `r_i` |

For v2, every cell starts at its sampled active lower bound. Each of eight
sequential samples applies one stochastic RESET pulse and then reads apparent
raw `a`; there is no SET or reinitialization between samples. The arithmetic
mean is computed independently for every cell. There is no quad maximum,
cross-cell pooling, or standard-error guard. The mapped mean
`(mean(a)+1)/2` is clipped only to the intersection of the public `[0,1]`
conductance coordinate and that cell's sampled active interval, and then
frozen once per topology/assignment for all scale candidates. The fixed-`r`
arms do not consume this commissioning result.

This deliberately differs from the earlier 91% RESET-relative protocol. That
protocol used the same eight-sample cost but formed one guarded shared quad
baseline `max_i(mean_i + 3 SE_i)` in the already reference-relative
coordinate. V2 tests the user's new per-cell raw-`a` proposal; it is not a
reproduction of the historical mapper.

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
strict per-assignment commissioning NPZ/receipts, exact `r`, RESET origins,
integer grid indices, positive and negative level
capacities, zero-reference residuals, target hashes, selected scale/gain,
accuracy, teacher agreement, KL, prediction flips, logical sign flips,
`RMS(D)/mean(S)`, conductance-sum loading, voltage statistics, and a normalized
total-conductance proxy.

The implementation and strict config are:

- `experiments/mnist_relu_drn/ibm_om_standard_level_scheme_screen.py`;
- `examples/mnist_relu_drn/ibm_om_standard_level_scheme_screen/screen.json`
  (v1 exact-lower-bound diagnostic); and
- `examples/mnist_relu_drn/ibm_om_standard_level_scheme_screen/screen_v2.json`
  (v2 per-cell RESET-mean primary screen).

The frozen source inputs are the bias-free teacher with SHA-256
`9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52`
and
`examples/mnist_relu_drn/ibm_om_differential_pair_hwa_pilot/clean.json` with
SHA-256
`59c0a6a2c324dfea6bce8ec982dc9cf15286a6f577bace31471c4f646ebd1cad`.

## Results

Both v2 executions completed all four development calibrations, all 12
held-out arms, eight population receipts, eight commissioning receipts, and
10,000 test examples per arm. The independent analyzer verified every source,
population, commissioning, mapping, calibration, prediction, metric, and
aggregate field. Main and replay satisfy the predeclared semantic-equivalence
criterion after normalizing only result-root path strings.

| Scheme | Standard mean (range) | Continuous mean (range) | Standard minus continuous |
| --- | ---: | ---: | ---: |
| Four devices, no fixed `r` | 25.76% (19.37--30.56%) | 27.23% (19.21--33.27%) | -1.46 pp |
| Four devices, fixed `r` | 23.03% (20.34--25.95%) | 71.11% (68.03--72.84%) | -48.09 pp |
| Eight devices, no fixed `r` | 23.46% (21.54--26.10%) | 25.22% (23.06--29.24%) | -1.76 pp |
| Eight devices, fixed `r` | 14.16% (9.80--16.72%) | 42.69% (29.14--54.20%) | -28.53 pp |

Every arm fails the 90% ideal gate. The fixed-`r` target hashes and results are
bit-identical between v1 and v2, confirming that commissioning changed only
the no-`r` intervention. Relative to exact lower-bound v1, bounded per-cell
RESET means reduce no-`r` standard accuracy by `5.71` points for four devices
and `1.72` points for eight devices; continuous accuracy falls by `5.37` and
`1.93` points. Per-cell RESET averaging therefore does not recover the raw-`a`
scheme.

The mechanism is baseline mismatch, not a new rounding failure. From v1 to
v2, no-`r` median signed level count remains seven and active-target
quantization RMS is essentially unchanged. In contrast, baseline logical-
contrast RMS rises from `0.1431/0.1422` to `0.1502/0.1502` in the two
four-device layers and from `0.2035/0.2100` to `0.2136/0.2203` in the
eight-device layers. Mean physical edge loading also rises from
`0.1209/0.1671` to `0.1319/0.1767` and from `0.2255/0.2724` to
`0.2484/0.2943`, respectively. Hidden/output voltage RMS falls accordingly.
Independently estimated cell baselines do not cancel in the four-edge rail
contrast or between independently sampled active/reference branches, so a
nominal zero weight already produces transfer and extra denominator loading.

Commissioning itself is strongly boundary dominated. Across held-out
artifacts, about `60.32%` of mapped means are projected by the public
coordinate or sampled active bounds; about `46.96%` lie below public `x=0`
before projection and about `26.2%` of native means fall below the sampled
minimum. The final origin is only `0.01199` above the exact mapped lower bound
on average and has median zero shift. These must be called **bounded per-cell
means**, not untouched observed averages.

The fixed-`r` mechanism remains different. Its continuous envelopes are much
better, particularly for four devices, but four-delta mapping leaves only
three median signed levels and roughly `14--16%` zero-capacity quads. The
large standard-versus-continuous loss is therefore a level-capacity/rounding
failure layered on reference mismatch and sum loading; it is not evidence
that retaining `r` is intrinsically harmful.

The next ideal control should enforce zero baseline contrast rather than
retain independent origins: one commissioned common baseline per four-device
rail quad, and at least one shared baseline per active/reference edge in the
eight-device topology so `G_a=G_r` at logical zero. It should preserve the
physical conductance sum in the denominator and report common-window failures
explicitly. A separate fixed-`r` sensitivity should test whether a
bidirectional centered codebook or a smaller frozen spacing can use its lower
and upper headroom before any P&V or HWA launch.

The versioned definition of `ideal_mapped_init_accuracy` and the frozen
shared-zero Stage 0E design are recorded in
[`ibm_om_deployment_scheme_investigation.md`](ibm_om_deployment_scheme_investigation.md#standard-headline-accuracies).
This completed v2 screen predates that artifact contract and remains its
independent-baseline parent evidence; it must not be relabelled as the
canonical metric output.

The ignored result bundles are:

- `results/mnist-ibm-om-standard-level-reset-mean-scheme-screen-20260827-v2/`;
- `results/mnist-ibm-om-standard-level-reset-mean-scheme-screen-20260827-v2-replay/`.

The verified report is
`post_run_analysis/post_run_analysis.md` under the main result root. Numerical
source was frozen at commit `3f389fbb`; the versioned analyzer was added at
commit `d6227bfd`.

## Claim boundary

The OM preset is a normalized fitted model, not raw measured-device replay and
not an absolute conductance calibration. V2 is also a scheme-optimized
commissioning comparison: exact hidden `r` is not characterization-cost
matched to an eight-read RESET estimate. A completed result can compare the
declared model-relative schemes and isolate the v1-to-v2 no-`r` baseline
change, but cannot establish a causal benefit of `r`, fabricated-array
accuracy, deployment write success, physical power, or a need for on-chip
training.
