# Current Perfect-Diode Conv State

Updated: 2026-08-12

This is a dashboard, not a second protocol. Selection rules remain in the
[active protocol index](conv_paper_hyperparameter_protocol.md). Live running
state is generated separately in
[`current_simulations.md`](current_simulations.md).

## Findings From 2026-08-12

### Higher beta confirms a depth-dependent centered-EqProp boundary for ours

The matched Conv1/Conv2 extension repeats the trained-Conv3 higher-beta
diagnostic at common injected `B={100,300,1000}`, `T=K=64`, one fixed
16-example ordinary-MNIST batch, and true float32/float64. Baseline uses base
beta `B`; ours uses `B/4` for Conv1 and `B/16` for Conv2. Both positive
one-sided and centered frozen-current estimators are evaluated from identical
phase starts.

Centered baseline remains all-layer faithful through beta `1000` for both
architectures. Its worst-layer float64 cosine at that point is `.990309` for
Conv1 and `.993496` for Conv2. Ours crosses the finite-perturbation boundary
earlier with depth: Conv1 passes the cosine `>=.99`, symmetric norm delta
`<=.1`, and residual gate at beta `100/300` but not `1000`; Conv2 passes only
at beta `100`. Its worst-layer centered float64 cosine is
`.998584/.994410/.979564` for Conv1 and
`.995390/.984057/.911775` for Conv2 over the three betas.

No one-sided point passes. At beta `100`, worst-layer float64 cosine is
`.914346/.943015` for baseline Conv1/Conv2 and `.758829/.263565` for ours,
then falls further with beta. Float32 and float64 nearly coincide throughout
this large-signal grid, so the failure is finite-perturbation and active-set
bias rather than small-contrast float32 cancellation. Positive output
displacement for ours is already `4.865` in Conv1 and `17.999` in Conv2 at
beta `100`, reaching `48.744/180.14` at beta `1000`.

Together with Conv3, the centered ours boundary moves downward with depth:
Conv1 is faithful through `300`, Conv2 only through `100`, and Conv3 through
`30` on the tested grids. This does not justify high-beta EqProp training;
these are BPTT-checkpoint gradient diagnostics far outside the source `.01`
current cap. See the [review](../results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1/review.md),
[layer table](../results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1/analysis/layer_metrics.csv),
and [comparison figure](../results/perfectdiode-conv12-baseline-ours-eqprop-higher-beta-seed0-20260812-v1/analysis/cosine_vs_injected_beta.png).

## Findings From 2026-08-11

### Occupancy report-only exposes a strong Conv3 legacy-SGD region, still open at lower Conv LR

The matched Conv3 repeat keeps the parent study's exact
`Uniform[1e-5,1e-4)` initializer, zero bias LRs, `T=K=8`, rho-search runner,
three-epoch budget, 90% gate, and retained projection/T/K checks. Its only
intended scientific change is that exact-bound occupancy is measured but does
not reject a run.

Legacy-SGD completes eleven numeric, epochwise-T/K-valid candidates. Three
meet 90%. At fixed `rho_dense=.0033333`, reducing `rho_conv` by three gives:

| `rho_conv` | Accuracy | Exact final either-bound occupancy |
|---:|---:|---:|
| `3.333e-4` | `92.56%` | `11.45%` |
| `1.111e-4` | **`93.62%`** | **`7.65%`** |

The best cell uses actual weight LRs
`C0=3.469e-8, C1=1.224e-5, C2=1.153e-5, Dense=9.448e-6` and zero bias LRs.
Its middle dense coordinate is bracketed by worse neighbors, but it sits on
the newly expanded lower Conv-rho edge. The surface therefore ends
`unresolved_after_boundary_expansion`, with no formal selected LR. The next
range test should extend Conv rho downward near `3.70e-5` on the same runtime;
cell `046` is a next-search center, not a published handoff.

Within legacy-SGD, accuracy versus exact final occupancy is
Pearson/Spearman `-0.695/-0.664` over eleven cells. The best row has the lowest
endpoint occupancy. This argues against more endpoint clipping being
beneficial, but is not causal and does not reproduce the removed persistent
occupancy-increase statistic.

The other Conv3 surfaces remain unresolved: baseline-SGD fails its probe;
baseline-Adam and ours-Adam have every cell rejected by the retained
projection gate; ours-SGD's six numeric endpoints remain at chance and fail
post-training T/K; legacy-Adam peaks at `88.66%`. Occupancy report-only
therefore does not create a complete Conv3 handoff.

The repeat is not a clean occupancy-only counterfactual for adaptive
legacy-SGD. Parent legacy ran on Trex RTX 5090/PyTorch 2.11; current legacy ran
on Jean Zay V100/PyTorch 2.5. Matched initialization, data order, signatures,
and code still branch differently at the retained projection threshold,
shifting the adaptive grid. A causal gate test must run both policies on one
runtime with fixed LR vectors and no re-probe. Final local validation covers
all six surfaces and receipts, 47 canonical bundles, 21 numeric endpoints,
and 84 layer-occupancy rows with no authority or integrity error. See the
[study review](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1/review.md)
and [curated manifest entry](experimental_manifest.md#perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-occupancy-report-only-seed0-20260811-v1--conv3-lr-search-without-occupancy-rejection).

## Findings From 2026-08-10

### Bounded-uniform zero-bias LR selection is incomplete beyond legacy

The matched seed-0 ordinary-MNIST search now has complete transport coverage
for all 18 Conv1/Conv2/Conv3 x baseline/ours/legacy x SGD/Adam surfaces. Each
architecture uses one shared `Uniform[1e-5,1e-4)` checkpoint, projection to
`[1e-5,1e-4]`, and exact-zero hidden-bias learning rates. Six surfaces meet the
predeclared 90% three-epoch handoff gate:

| Architecture | Scheme / optimizer | Selected `(rho_conv,rho_dense)` | Accuracy | Exact either-bound occupancy |
|---|---|---:|---:|---:|
| Conv1 | ours / SGD | `(0.001,0.0033333)` | `90.18%` | `6.54%` |
| Conv1 | ours / Adam | `(0.001,0.01)` | `91.60%` | `19.28%` |
| Conv1 | legacy / SGD | `(0.003,0.0033333)` | `94.50%` | `0.67%` |
| Conv1 | legacy / Adam | `(0.009,0.01)` | `95.42%` | `4.18%` |
| Conv2 | legacy / SGD | `(0.003,0.0033333)` | `93.34%` | `6.21%` |
| Conv2 | legacy / Adam | `(0.009,0.01)` | `95.42%` | `14.31%` |

Conv1 baseline peaks at `87.52/88.84%` for SGD/Adam. Conv2 baseline and ours
peak at `49.34/84.36%` and `69.50/87.38%`, respectively, so these are
observations rather than selected LRs. Conv2 ours-Adam has a weak upper-conv
edge hint at `(0.009,0.01)`, but increasing conv rho from `0.003` to `0.009`
there adds only `0.28 pp`. Conv3 produces no numeric candidates: baseline-SGD
fails its stability probe and every tested cell in the other five surfaces is
rejected by the Conv3 safety contract, including the smaller adaptive legacy
grids.

Final exact-bound occupancy is not a reliable standalone LR diagnostic.
Within-surface accuracy correlations switch sign, from Pearson/Spearman
`-0.953/-0.817` to `+0.982/+0.967`; Conv2 ours-Adam is approximately null at
`-0.081/-0.117`. The best Conv1 legacy rows have only `0.67/4.18%` pooled
clipping, whereas below-gate Conv2 baseline-Adam has `44.04%`. Endpoint
occupancy loses clipping history, update direction and magnitude, near-bound
mass, and layer weighting. It can flag saturation but cannot establish that an
LR is too high or too low. This study therefore does not publish a complete
bounded handoff, and no long confirmations were launched. See the
[final report](../results/perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1/analysis/final/report.md)
and [curated manifest entry](experimental_manifest.md#perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1--bounded-uniform-zero-bias-lr-selection).

### Literal amplification-scaled current EqProp has a float64 window, but not a float32 deep update

The latest read-only replay now applies the proposed scaling to the nudging
dynamics:
`beta_injected=beta_base*(voltage_amp/current_amp)^L`, with the exact output
bias-current-row exponent `L=1/2/3` for Conv1/2/3. The factors are
baseline `1/1/1`, ours `4/16/64`, and legacy `16/256/4096`. Earlier cost-mode
bundles only reported this factor and are superseded for the literal-current
question.

The runtime rule freezes `force=-dC/dy` at the common post-`T` state and
applies the linear output term `-beta_injected<y,force>`. The original replay
uses `[dtheta E(s_plus)-dtheta E(s_zero)]/beta_injected`; the complete Conv3
extension also evaluates the centered
`[dtheta E(s_plus)-dtheta E(s_minus)]/(2 beta_injected)` estimator. Biases
remain active but are excluded from scored gradients. These are frozen-current
EqProp diagnostics, not full-cost `E+beta*C` EqProp.

In float32, neither the literal common-base nor equal-injected Conv sweep
selects any of the 18 architecture/scheme/checkpoint contexts. Readouts align
first, while the earliest deep convolution remains poor or exactly zero.
Increasing to `T/K=64/64` changes finite cosines by at most `0.007988`.
FC1/2/3 controls reproduce the depth trend; only FC1 legacy selects a beta,
so convolution is not the cause.

True float64 resolves the direction. Every trained Conv2/Conv3 case has a
tested case-specific beta with all-layer cosine above `0.99`. The clearest
result is trained Conv3 legacy at base beta hat `3e-10`: base beta
`8.4139263e-9`, injected beta `3.4463442e-5`, and float64 layer cosines
`.999791/1/1/.999764`. The identical float32 replay gives
`zero/.006617/.561176/.999764`. Its matched-zero state displacement is
`5.19e-12/8.17e-9/1.35e-5/.01793` from first state through output.

The completed trained-Conv3 true-dtype curve contains 14 unique baseline,
11 ours, and 7 legacy points. In float64, every layer exceeds cosine `0.99`
for baseline at base beta hat `1e-6`--`3e-4`, ours at
`1e-8`--`1e-5` (the last point is capped), and legacy at
`1e-10`--`1e-9`. In float32, no scheme has an all-layer point and `C0` is an
exact-zero EqProp gradient at every tested beta for all three schemes. The
float32 early-layer failure is therefore shared by baseline, ours, and legacy,
not unique evidence against legacy amplification. These are cosine windows;
the trained baseline shadow remains residual-limited at native `T=12`, so its
float64 window is not a separate equilibrium-convergence certification.

The user-directed trained-Conv3 extension now raises common injected beta to
`{.01,.1,1,3}` for baseline and ours, extends baseline alone through
`{10,30,100}`, and raises the replay to `64/64` plus a `128/128` confirmation.
There is still no float32 all-layer window. At injected beta `100`, baseline
float32 `C0/C1/C2/D` cosines are `.245752/.773328/.999596/.992606`; C0 has
norm ratio `4.67` and `83.5%` exactly-zero contrast entries, while the float64
Dense cosine has already fallen to `.992606`. Ours reaches finite-beta bias
much sooner: at beta `3`, its output displacement is `.591827` and its
float64 Dense cosine is `.674475`, while float32 C0/C1 are only
`.086006/.157173`.

At injected beta `.1`, `64/64` and `128/128` are numerically identical.
Longer relaxation repairs baseline's float64 projected-KKT residual from a
native worst p90 near `.07346` to about `2.7e-11`, but does not repair its
float32 direction; the float32 residual stalls near `.01099`. Ours passes the
float32 residual gates and still has bad C0/C1. The failure is therefore not
an iteration-budget effect. Its matched-free/nudged float64 displacement at
beta `.1` is `4.21e-10/6.11e-8/3.13e-5/.01973` from H0 through output,
versus `1.94e-10/2.01e-9/5.40e-7/1.04e-4` for baseline.

The matched one- versus two-sided extension tests the next boundary on one
Trex surface: injected beta baseline `{100,300,1000}` and ours `{3,5,10}` at
`T=K=64`.  Centered `(G_plus-G_minus)/(2 beta_injected)` is much better than
one-sided differencing, especially at the readout, but it still does not yield
an all-layer float32 estimator.  Centered baseline float32 C0 rises only
`.314963 -> .644491 -> .915589`; centered ours C0 is
`.024906/.066527/.046875` and C1 reaches only `.494860`.  In float64, centered
worst-layer cosine remains `.997707` for baseline beta `1000` and `.998990`
for ours beta `10`.

Those float64 cosines occur outside a small-nudge regime.  At the largest
betas, matched-zero output displacement is `1.038` baseline and `1.973` ours;
one-sided Dense cosine has fallen to `.709103/.194545`, while centered Dense
remains essentially one by canceling even-order finite-beta error.  Negative
and positive displacement magnitudes are nearly symmetric, and every matched
positive endpoint hash is identical across estimator runs.  Ours passes every
residual gate; baseline float32 is already slightly residual-limited at the
common free/zero Layer 2 state, with no signed-phase residual blow-up.  Thus
neither a still larger beta nor two-sided subtraction repairs the float32
early-layer transport problem.  See the
[one-vs-two review](../results/perfectdiode-conv3-baseline-ours-eqprop-one-vs-two-sided-higher-beta-seed0-20260811-v1/review.md) and
[combined figure](../results/perfectdiode-conv3-baseline-ours-eqprop-one-vs-two-sided-higher-beta-seed0-20260811-v1/analysis/conv3_eqprop_one_vs_two_sided_higher_beta.png).

The complete Conv3 matrix now closes the common injected-current grid for all
three schemes and both checkpoint roles. It covers
`B={.001,.003,.01,.03,.1,.3,1,3,10,30,100}`, `T=K=64`, native
float32/float64, and both positive one-sided and centered estimators. Base beta
is scaled as `B`, `B/64`, and `B/4096` for baseline, ours, and legacy.
Centered float64 has a common all-layer cosine-and-norm window at
`B={.001,.003,.01}` across initialization and best weights. At `B=.001`, the
worst layer cosine is `.999999/.999999` for baseline init/best,
`.999998/.9999998` for ours, and `.999638/.999906` for legacy.

That common centered window is directional, not perturbatively small for
legacy. At `B=.001`, legacy H0/H1/H2/output relative displacement is
`9.45e-9/2.51e-5/.00469/1.34555` at initialization and
`1.50e-10/2.37e-7/.000391/.520166` at best; output delta RMS is
`.738209/.187204`. One-sided float64 therefore has no legacy all-layer point
on this grid: Dense limits its best maximin cosine to `.877393/.847661`.
Centered subtraction cancels that leading finite-beta error.

Float32 has no all-layer point for either estimator or any scheme. For the
centered estimator, the best worst-layer cosine over the full grid is only
`.164965/.335705` baseline, `.596808/.355025` ours, and
`.161268/.149566` legacy at initialization/best. All 33 undefined cosines are
exact-zero float32 `C0` updates. All float64 residual gates pass; only trained
baseline float32 Layer 2 is slightly above the strict residual gate
(`p90=.01025--.01147`, maximum `.01660`), including at the shared free/zero
state, while ours and legacy pass. The completed grid therefore strengthens
the conclusion that float32 early-layer failure is shared numerical contrast
loss, not a legacy-only mechanism. See the
[complete review](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/review.md),
[full layer table](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/analysis/summary.csv),
[float32 curves](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/analysis/plots/cosine_vs_injected_beta_float32.png),
and [float64 curves](../results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1/analysis/plots/cosine_vs_injected_beta_float64.png).

The dedicated ours-only tail now closes the unmeasured region above injected
beta `10`: `B={30,100,300,1000}`, base beta `B/64`, both one-sided and
centered estimators, true float32/float64, and `T=K=64`.  It finds no missing
or doubled beta factor.  The current scale and finite-difference denominator
both use the guarded factor `64`; zero and positive endpoint hashes match
exactly between estimators; displacement is proportional to `B` through the
linear-response portion; and every ours residual gate passes.

The apparently small motion is confined to the early layers.  Positive
float64 relative displacement at `B=30/100/300/1000` is only
`1.26e-7/4.15e-7/2.50e-6/1.37e-5` in H0, but already
`5.92/19.75/59.44/198.62` at the output.  Absolute RMS confirms real
output-to-H0 attenuation, not merely a cross-layer normalization artifact.
Centered float64 remains all-layer cosine `>=.99` at `B=30`, then loses that
property by `100` as signed diode active sets diverge.  Centered float32 never
has an all-layer point: its best worst-layer cosine is only `.481631` at
`B=300`, after later layers have become finite-beta biased.  Thus larger beta
trades float32 subtraction zeros for nonlinear bias before exposing a usable
deep update; it does not rescue ours or reveal an injection bug.  See the
[beta-tail review](../results/perfectdiode-conv3-ours-eqprop-beta-above10-bug-audit-seed0-20260811-v1/review.md) and
[cosine/displacement figure](../results/perfectdiode-conv3-ours-eqprop-beta-above10-bug-audit-seed0-20260811-v1/analysis/conv3_ours_eqprop_beta_above10.png).

Larger legacy nudges leave the readout's linear-response regime without
blowing up the residual. At injected beta `.003446`, output displacement is
`1.793` and Dense cosine `.426627`; at `.01`, they are `5.202` and `.167644`.
All float64 residual gates still pass. A small residual therefore proves
equilibration, not a perturbatively small nudge. No tested common base beta is
good for every scheme: baseline needs a larger signal while the `4096x`
legacy factor overnudges its readout.

Supported handoff: a positive one-sided float64 frozen-current legacy Conv3
pilot should still start at base beta hat `3e-10`. A centered float64 pilot can
use common injected `B=.001` (legacy base beta `B/4096`), but it must be labeled
as a symmetric finite-difference point with order-one legacy output motion,
not a small-nudge point. Do not launch naive float32 deep training. Select beta
using injected current and layerwise displacement, and treat full-cost EqProp
as a separate learning-rule experiment. See the
[review](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/review.md),
[Conv beta curves](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/conv-current-commonbase/artifacts/plots/),
[true-dtype heatmap](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/analysis/eqprop_current_commonbase_true_dtype_cosine.png),
[trained Conv3 true-dtype curve](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/analysis/conv3_current_true_dtype_beta_curve.png),
[exact Conv3 table](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/analysis/conv3_current_true_dtype_beta_curve.md),
[high-beta/high-T/K review](../results/perfectdiode-conv3-baseline-ours-eqprop-high-beta-high-tk-seed0-20260811-v1/review.md),
[high-beta layerwise figure](../results/perfectdiode-conv3-baseline-ours-eqprop-high-beta-high-tk-seed0-20260811-v1/analysis/conv3_eqprop_high_beta_high_tk.png),
[T/K table](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/conv-current-commonbase-tk/fixed_beta_tk_summary.csv),
and [FC curves](../results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/fc-current-commonbase/artifacts/plots/).

### Across schemes, centered EqProp matches BPTT only in Conv1

A read-only replay compared centered cost-nudged EqProp with BPTT at
hash-verified reconstructed initialization and maximum-validation weights for
baseline `voltage/current=1/1`, ours `4/1`, and legacy `4/0.25`. All 18 cases
used the same fixed 64-example ordinary-MNIST cohort and
`beta in {0.003,0.01,0.03,0.1,0.25,0.5,1.0}`. Biases remained active in the
dynamics but were excluded from scoring; no optimizer step or official-test
read occurred.

At each source-native T/K, the beta maximizing the worst weight-layer cosine
gives:

| Architecture | Scheme | Init: beta / worst cosine | Best: beta / worst cosine |
|---|---|---:|---:|
| Conv1 | baseline | `1 / 0.998771` | `1 / 0.993006` |
| Conv1 | ours | `1 / 0.999917` | `1 / 0.995962` |
| Conv1 | legacy | `0.03 / 0.999936` | `0.1 / 0.974661` |
| Conv2 | baseline | `1 / 0.202739` | `0.5 / 0.143147` |
| Conv2 | ours | `0.5 / 0.476067` | `1 / 0.286438` |
| Conv2 | legacy | `0.003 / 0.321898` | `0.1 / 0.005915` |
| Conv3 | baseline | `0.25 / 0.042111` | `0.5 / 0.039319` |
| Conv3 | ours | `0.1 / 0.057662` | `1 / 0.034167` |
| Conv3 | legacy | `0.003 / -0.998542` | `none / dead C0` |

Conv1 is the only architecture with strong all-layer direction for every
scheme. In Conv2/Conv3, later and readout layers can align while the early
convolutions are orthogonal, reversed, or dead. Legacy's excellent Conv1 match
exists only in a narrow beta window and its BPTT training advantage does not
translate into deep EqProp gradient fidelity. Conv3 baseline uses native
`T/K=12/8`; at the scheme-shared `8/8` trained anchor its worst cosine is
`-0.012847`, rather than the native `0.039319`.

The independent T x K grid resolves the earlier phase-length ambiguity. With
beta frozen at the shared-anchor selection, K produces the larger cosine range
in 41/50 complete layer grids and a range above `0.01` in 17/50. T has only one
material effect: trained Conv3-baseline C0. Retuning beta improves trained
Conv3 ours from `0.034167` at `8/8` to `0.112146` at `8/16`, but no deep scheme
reaches a useful all-layer match. More K can also destabilize the negative
phase; all 152 unstable production phase records are negative.

Matched-zero-K state displacement shows why the global state norm is
insufficient. The largest displacement is always in the final/output state,
which can move by `9.35x` for Conv2-ours initialization and `15.7x` for
Conv2-legacy initialization even when the concatenated global ratios are only
`0.0357` and `0.804`. Baseline phases are nearly symmetric; ours and legacy
develop strong negative-phase asymmetry with depth. Displacement magnitude is
not monotone with cosine: trained Conv3 baseline moves its matched output state
by only `3.94e-4` yet has a worst cosine of `0.039319`.

The supported conclusion is unchanged but now applies across amplification
schemes: do not launch deep EqProp training under the present centered
whole-output-nudge contract. The next diagnostic should intervene on backward
transport using staged/layerwise nudging or an explicit feedback/residual path,
and gate every weight layer jointly on finite dynamics, non-dead gradient,
direction, scale, and state displacement. See the
[reviewed report](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/review.md),
[layer/beta table](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/analysis/parameter_summary.csv),
[T/K table](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/analysis/fixed_beta_tk_summary.csv),
and [state-displacement table](../results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1/analysis/state_displacement_summary.csv).

## Findings From 2026-08-09

### Learning-rate rescaling and extra time do not close the deeper zero-bias baseline gap

All 15 requested seed-0 ordinary-MNIST baseline-SGD counterfactuals completed
and validate locally, including every epoch checkpoint, five traced transitions
per epoch, exact-zero saved bias tensors, and zero official-test reads. The new
current-LR controls reproduce the prior metric curves exactly over every
matched epoch.

| Architecture | Current, 30 epochs | Best other 30-epoch LR vector | Conv-LR/3, 50 epochs | Prior ours SGD | Prior legacy SGD |
|---|---:|---:|---:|---:|---:|
| Conv1 | `96.60%` | **`96.66%`** | `96.64%` | `96.50%` (10 ep) | `95.92%` (10 ep) |
| Conv2 | `97.22%` | `97.08%` | **`97.30%`** | `97.98%` | `97.96%` |
| Conv3 | **`97.04%`** | `96.92%` | `96.78%` | `98.20%` | `98.76%` |

Tripling Conv rates reduces Conv2/Conv3 best accuracy by `0.40/0.48` points,
and matching ours' or legacy's initial relative-proposal vectors does not help.
For Conv-LR/3, epochs 31--50 add only `0.02/0.14/0.34` points for
Conv1/Conv2/Conv3. The 50-epoch Conv2 baseline still trails the 30-epoch
ours/legacy controls by `0.68/0.66` points; Conv3 remains farther behind.
Conv1 is effectively insensitive across the tested vectors and its amplified
controls are not duration-matched. This is diagnostic evidence that ordinary
under-training is not the explanation for the deeper gap. See the
[combined review](../results/perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-long50-ordinary-mnist-seed0-20260808-v1/analysis/report.md).

### Aggregate performance follows `legacy > ours > baseline`

The completed 72-run signed/scaled-bias mechanism study rejects a
flat-parameter-minimum explanation for legacy amplification. Legacy is much
sharper in parameter space but makes the physical equilibrium problem less
stiff, suppresses runaway weight and bias growth, and leaves substantially
fewer bounded conductances pinned while producing sparse, stable hidden diode
states. The evidence supports depth-dependent structural regularization and
preconditioning, with the strongest benefit in deeper bounded networks. See
the [legacy amplification mechanism study](legacy_amplification_mechanism_study_20260807.md).

Together with the zero-bias and learning-rate counterfactuals, the aggregate
performance ordering for the tested family is now `legacy > ours > baseline`,
with the separation strongest in deeper and bounded networks. This is not the
ordering of every individual shallow or optimizer-matched cell, but the overall
pattern is established. The leading interpretation is that the networks are
easier to train as their effective dynamics become more feed-forward-dominant.
That interpretation still needs to be quantified and tested causally; flat
parameter minima, larger effective learned weights, ordinary under-training,
and a requirement for trainable biases are not sufficient explanations.

### Relation to feedback-regulated equilibrium propagation

Liu and Chen's ICLR 2026 paper,
[*Toward Practical Equilibrium Propagation*](https://iclr.cc/virtual/2026/poster/10008357),
provides useful evidence that attenuating feedback can make equilibrium-style
recurrent networks converge faster and improve accuracy up to a
depth-dependent optimum. Feedback that is too weak loses deep learning signal;
their residual connections compensate for that failure. Its relevance here is
suggestive rather than direct: it uses distinct forward and backward
connections in general vector-field dynamics with a local, STDP-compatible
learning rule, rather than deriving both directions from one reciprocal energy.
Settling speed is secondary in the present study. The transferable hypothesis
is that weakening the effective recurrent return path while preserving forward
and deep update transmission improves accuracy. This should be tested through
the amplification-by-bound interaction, not assumed from equilibrium speed.

## Findings From 2026-08-06

### Legacy can retrain without biases

The completed Conv2/Conv3 legacy-SGD medium-affine controls distinguish
post-training checkpoint dependence from training necessity. Both models were
trained from initialization with every original weight learning rate unchanged
and every hidden bias fixed exactly at zero:

| Architecture | Original learned-bias validation | Zero-bias validation | Difference | Original / zero-bias validation loss |
|---|---:|---:|---:|---:|
| Conv2 legacy SGD | `91.42%` | `91.58%` | `+0.16 pp` | `0.165221 / 0.164916` |
| Conv3 legacy SGD | `96.70%` | `96.82%` | `+0.12 pp` | `0.082174 / 0.081811` |

The differences are only eight and six predictions out of 5,000 and should be
read as no observed zero-bias deficit, not as an accuracy improvement. Legacy's
weights can compensate when biases are absent from the start. The historical
unscaled-bias contract remains a confound, but the large accuracy losses from
post-hoc bias removal do not show that biases caused legacy's advantage. See
the [reviewed matched control](../results/perfectdiode-conv23-legacy-zero-bias-medium-affine-seed0-20260731-v1/analysis/report.md).

### All schemes train strongly without biases on ordinary MNIST

All 18 Conv1/Conv2/Conv3 x scheme x optimizer seed-0 ordinary-MNIST controls
completed with exact-zero bias rates and tensors and with the accepted weight
rates unchanged. Best validation accuracy spans `95.92%`--`99.00%`. Broken
out by optimizer, the complete matrix is:

| Architecture | Epochs | Optimizer | Baseline | Ours | Legacy |
|---|---:|---|---:|---:|---:|
| Conv1 | 10 | SGD | `96.28%` | **`96.50%`** | `95.92%` |
| Conv1 | 10 | Adam | `96.16%` | **`96.44%`** | **`96.44%`** |
| Conv2 | 30 | SGD | `97.22%` | **`97.98%`** | `97.96%` |
| Conv2 | 30 | Adam | `97.36%` | `98.10%` | **`98.46%`** |
| Conv3 | 30 | SGD | `97.04%` | `98.20%` | **`98.76%`** |
| Conv3 | 30 | Adam | `97.80%` | `98.66%` | **`99.00%`** |

This establishes that trainable biases are not required for strong learning at
the accepted weight rates. The official test was disabled, so these values are
optimization diagnostics rather than paper evidence. See the
[18-arm review](../results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/analysis/report.md).

### The zero-bias depth ordering is a conditioning effect, not extra capacity

The complete curves show that ours exceeds baseline at all `140/140` matched
epochs. Legacy exceeds ours throughout Conv3 and Conv2 Adam, but not Conv1 or
Conv2 SGD, so `legacy > ours > baseline` is a depth-dependent pattern rather
than a universal ordering.

For zero biases, the implemented energy has an exact coordinate transform. If
`g=voltage_amp/current_amp`, then normalized states turn edge weights into
`W'_l=g^l W_l`. Conv3's four effective edge multipliers are `[1,1,1,1]`,
`[1,4,16,64]`, and `[1,16,256,4096]` for baseline, ours, and legacy. The
schemes therefore have the same classification function family, but the shared
raw initialization represents progressively stiffer deep effective models.
Their normalized outputs are also multiplied by `1`, `4^d`, and `4^(2d+1)`
respectively in the raw coordinate system. Paired argmax is invariant to that
positive scale; the fixed-one-hot squared loss and its gradients are not.

A guarded 128-example best/final/init replay found nearly identical
scale-invariant class separation and clamp occupancy at reconstructed
initialization. By the best Conv3 checkpoints, deepest-layer between/within
class variance orders baseline below ours below legacy for both optimizers.
For Conv3 SGD, the initial largest-to-smallest layerwise relative-proposal
spread is `41.6x/5.3x/4.6x`; this imbalance is absent in the baseline Conv1
negative control. These observations support depth-wise preconditioning and
loss/LR scaling as the mechanism. See the
[mechanism report](../results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/analysis/mechanism/report.md).

### Signed-only bounded screen is suggestive but incomplete

The separate bounded-uniform Conv1/Conv2 screen used signed but still-unscaled
biases. All 55 bundles validate. Every selected checkpoint uses both bias
signs. SGD changes versus the recorded nonnegative selections stay within
`-0.06` to `+0.08` points, while three Adam rows gain `+1.06`--`+1.84`
points. Five of eight surfaces select the maximum tested bias scale `9`, so the
bias search is not closed. The study excludes legacy, Conv3, bounded Kaiming,
and amplification-scaled biases; it cannot publish a global bounded handoff or
validate the proposed corrected bias contract. See the
[reviewed signed-only report](../results/perfectdiode-conv12-bounded-uniform-signed-bias-rho-seed0-v1/analysis/report.md).

## Findings From 2026-07-31

### Historical seed-0 wide-range paper outcome (unscaled nonnegative biases)

The first deterministic medium-affine seed-0 paper batch completed its fixed
budgets for all 18 Conv1/Conv2/Conv3 x scheme x optimizer rows. At every depth,
**legacy has the highest best-checkpoint official-test accuracy when each
scheme is represented by its better of SGD and Adam**:

| Architecture | Baseline best | Ours best | Legacy best | Legacy minus ours |
|---|---:|---:|---:|---:|
| Conv1 | `69.60%` (SGD) | `72.60%` (SGD) | **`74.64%` (Adam)** | `+2.04 pp` |
| Conv2 | `85.83%` (Adam) | `92.26%` (Adam) | **`94.06%` (Adam)** | `+1.80 pp` |
| Conv3 | `91.54%` (Adam) | `96.38%` (Adam) | **`97.01%` (Adam)** | `+0.63 pp` |

This is a surprising one-seed result, not evidence that legacy wins every
optimizer-matched row: Conv1 legacy-SGD reaches only `68.15%`, while legacy
wins both matched optimizer comparisons for Conv2 and Conv3. These runs belong
to the wide/unbounded **LR-handoff family**, but their paper conductance weights
are projected to `[0,100]`; they are not literally unconstrained weights. See
the [reviewed comparison](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/report.md#sgd-versus-adam)
and [validated metric inventory](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/summary.csv).

### Bias-contract audit and corrective branch

The audit found a potential implementation/protocol mismatch in the historical
models. Their hidden-bias energy is `-<z,b>` at every depth and is not scaled
with amplification, whereas the resistive interactions carry depth-dependent
amplification factors. This means the old comparisons do not vary only the
resistive amplification convention: an unscaled linear bias can become
progressively stronger relative to the deep resistive curvature, especially
for legacy.

The proposed replacement was deliberately isolated on branch
`codex/signed-scaled-bias` in worktree
`/home/filip/server_code_conv_learning_rate_protocol/.codex/worktrees/signed-scaled-bias`
at commit `b0bd5a61260c0a13db25a88b9feedcff363973cf`. It changes two coupled
scientific factors:

- biases become signed and unclamped rather than nonnegative-projected; and
- hidden bias `B_n` is scaled by `(current_amp / voltage_amp)^n` at zero-based
  hidden depth `n`.

The resulting `[B0,B1,B2]` coefficients are `[1,1,1]` for baseline,
`[1,1/4,1/16]` for ours, and `[1,1/16,1/256]` for legacy. Because both the sign
constraint and scaling change, this branch tests a proposed new bias contract
rather than a clean scaling-only ablation, and all bias learning rates must be
reselected.

The associated study
`perfectdiode-signed-scaled-bias-conv123-seed0-20260731-v1` is prepared to
search independent bias rho values while holding accepted weight rates fixed,
over bounded and wide x Conv1/Conv2/Conv3 x baseline/ours/legacy x SGD/Adam.
As verified at `2026-07-31 22:18 CEST`, it had **not yet been submitted**: no
Slurm ID, launch receipt, or result directory existed. Separate user-authorized
current-rho/T12 long-run successors later completed on 2026-08-06; their
current results are reported below. The terminal v3 selector remains a
separate incomplete lineage. A distinct signed-only, still-unscaled
bounded-uniform Conv1/Conv2 rho study also completed on 2026-07-31; its eight
surface records are [here](../results/perfectdiode-conv12-bounded-uniform-signed-bias-rho-seed0-v1/summary.json),
but they must not be cited as evidence for amplification-scaled biases.

### Bias sensitivity in the historical checkpoints

Read-only re-equilibration on the same guarded 256-example validation cohort
shows that the learned biases are much more important to legacy than to
baseline or ours, even though the deepest legacy bias tensors are smaller in
raw RMS:

| Intervention at final SGD checkpoint | Baseline | Ours | Legacy |
|---|---:|---:|---:|
| Conv2 deepest-bias removal: deep-state change | `0.12%` | `0.96%` | **`25.41%`** |
| Conv2 all-bias removal: accuracy change | `0.00 pp` | `+0.39 pp` | **`-9.38 pp`** |
| Conv3 deepest-bias removal: deep-state change | `0.033%` | `1.00%` | **`67.01%`** |
| Conv3 all-bias removal: accuracy change | `0.00 pp` | `-0.39 pp` | **`-17.58 pp`** |

For Conv3, legacy's deepest bias is `12.3x` smaller in RMS than ours, yet its
removal causes `39.5x` more absolute deep-state displacement. This establishes
strong checkpoint dependence, not training necessity: weights may compensate
when trained from the start under a different bias contract. See the
[end-of-training bias-effect report](../results/perfectdiode-paper-gradient-trace-limited-20260731-v1/analysis/bias_implementation_audit/paper_end_bias_effect/report.md).

## Current Focus

The active question is perfect-diode BPTT training for:

- Conv1, Conv2, and Conv3;
- strides `[2]`, `[2,2]`, and `[2,2,1]`;
- kernel `3`, padding `1`, and no pooling;
- baseline `v1/c1`, proposed/ours `v4/c1`, and legacy `v4/c0.25`;
- plain SGD and Adam; and
- wide-range reference and bounded hardware weight contracts.

Rho and learning-rate selection uses ordinary MNIST. Paper training uses
deterministic medium-affine MNIST.

## Wide-Range Rho Status

The wide-range reference projects conductance weights to `[0,100]`.

### Current long-run signed/scaled-bias accuracies

The current learned-bias matrix uses signed, unclamped,
amplification-scaled biases and wide `[0,100]` conductance weights. All 18
seed-0 wide arms completed and validate locally at exact `10/30/30`-epoch
budgets for Conv1/Conv2/Conv3. The runs use the user-authorized current-rho/T12
deviation and are awaiting direct study review. Best ordinary-MNIST validation
accuracy over the full training budget is:

| Architecture | Epochs | Optimizer | Baseline | Ours | Legacy |
|---|---:|---|---:|---:|---:|
| Conv1 | 10 | SGD | `96.30%` (ep 7) | **`96.60%` (ep 9)** | `95.80%` (ep 9) |
| Conv1 | 10 | Adam | `96.42%` (ep 9) | `96.74%` (ep 9) | **`96.86%` (ep 9)** |
| Conv2 | 30 | SGD | `97.08%` (ep 29) | **`98.20%` (ep 29)** | `97.96%` (ep 29) |
| Conv2 | 30 | Adam | `97.28%` (ep 30) | `98.16%` (ep 20) | **`98.42%` (ep 29)** |
| Conv3 | 30 | SGD | `97.06%` (ep 30) | `98.28%` (ep 23) | **`98.64%` (ep 20)** |
| Conv3 | 30 | Adam | `97.76%` (ep 30) | `98.68%` (ep 28) | **`99.04%` (ep 22)** |

Every Conv2/Conv3 maximum occurs after epoch 3, at epochs `20`--`30`, which
confirms that the three-epoch selector snapshot understated the long-run
accuracy.

Ordinary MNIST uses the deterministic 5,000-example validation split and never
reads the official test split, so these are optimization diagnostics rather
than paper accuracy. The corresponding deterministic medium-affine paper rows
read the official test exactly once from the maximum-validation checkpoint:

| Architecture | Epochs | Optimizer | Baseline | Ours | Legacy |
|---|---:|---|---:|---:|---:|
| Conv1 | 10 | SGD | `70.28%` | **`74.46%`** | `68.89%` |
| Conv1 | 10 | Adam | `70.46%` | `76.55%` | **`79.85%`** |
| Conv2 | 30 | SGD | `80.73%` | `91.00%` | **`92.64%`** |
| Conv2 | 30 | Adam | `86.43%` | `92.49%` | **`94.37%`** |
| Conv3 | 30 | SGD | `86.48%` | `94.30%` | **`97.02%`** |
| Conv3 | 30 | Adam | `91.58%` | `96.29%` | **`97.95%`** |

These are the newer corrected-bias-contract measurements, not the historical
unscaled-bias paper rows above. They remain one-seed results and retain the
current-rho/T12 deviation and direct-review limitation.

The higher long-run accuracies are real. The earlier
`96.70%/97.78%/97.96%` snapshot mixed a 10-epoch Conv1 confirmation with
three-epoch Conv2/Conv3 selectors and was not the best current long-run
summary. The current long runs are not literal continuations of those old
selectors: the bias contract and bias rates changed, and Conv3 baseline uses
`T/K=12/8`, so the cross-study increase cannot be attributed to duration
alone. The separate exact-zero-bias `10/30/30`-epoch controls reach
`96.50%/98.46%/99.00%`; their full scheme-by-optimizer table is reported in
[the zero-bias finding](#all-schemes-train-strongly-without-biases-on-ordinary-mnist).
Do not pool the signed/scaled learned-bias and zero-bias cells: they use
different bias contracts.

Authority: the validated `perfectdiode-ordinary-mnist-signed-scaled-bias-current-rho-t12-conv123-seed0-20260806-v1`
and `perfectdiode-signed-scaled-bias-current-rho-t12-conv123-seed0-20260806-v1`
study summaries in the `signed-scaled-bias` worktree, plus the reviewed
aggregate [legacy amplification mechanism study](legacy_amplification_mechanism_study_20260807.md).

### Conv1/Conv2

The fixed ordinary-MNIST operating points are:

| Architecture | Input gain | Operational `T/K` |
|---|---:|---:|
| Conv1 | `40` | `4/4` |
| Conv2 | `100` | `6/6` |

All six fixed-`T/K` security rows passed comparison with `(T=64,K=64)`.
The 12 scheme x optimizer core rho surfaces completed and produced generally
strong three-epoch diagnostic candidates. Filip has now authorized all 12
explicit parameter-wise LR vectors for current unbounded result rows. Use
those vectors unchanged, at the supplied precision, from the machine-readable
handoff below.

All six Conv1 vectors completed ten-epoch confirmations with final
ordinary-MNIST validation accuracy between `95.60%` and `96.48%`; final review
is pending. Conv2 remains three-epoch best-observed evidence, with final
validation accuracy between `95.96%` and `97.78%`. Terminal rho selection
remains incomplete, and the current handoff does not imply Conv2
long-confirmation evidence.

Authority:
[`perfectdiode_learning_protocol.md`](perfectdiode_learning_protocol.md) and
[`perfectdiode_conv12_unbounded_fixed_lr_handoff_20260729_v1.json`](../configs/conv/perfectdiode_conv12_unbounded_fixed_lr_handoff_20260729_v1.json).

### Conv3

The ordinary-MNIST Conv3 contract uses:

- channels `[64,128,256]`, strides `[2,2,1]`, and padding `[1,1,1]`;
- shared diagnostic input gain `360`; and
- the user-fixed `T=K=8` operating point for the completed rho study.

The six optimizer-specific rho surfaces completed on Jean Zay: 54 initial
cells plus 37 boundary-expansion cells, with all final search bounds closed.
The selected `(rho_conv,rho_dense)` pairs are baseline SGD
`(0.001,0.03)`, baseline Adam `(0.027,0.27)`, ours SGD `(0.009,0.03)`,
ours Adam `(0.081,0.09)`, legacy SGD `(0.003,0.01)`, and legacy Adam
`(0.027,0.01)`. These are one-seed, three-epoch ordinary-MNIST selection
results. The execution used `weight_min=0` and `weight_max=null`; it is
unbounded selector evidence rather than an identical `[0,100]` selection
surface. Filip explicitly authorized applying its six published vectors to
the downstream `[0,100]` paper configs; the source mismatch and missing long
confirmation remain recorded limitations.

Authority:
[`perfectdiode_conv3_learning_protocol.md`](perfectdiode_conv3_learning_protocol.md),
the
[`perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json`](../configs/conv/perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json)
handoff, and the
[`conv3_pd_unbounded_rho_t8k8_20260729T125734Z` manifest entry](experimental_manifest.md#conv3_pd_unbounded_rho_t8k8_20260729t125734z--conv3-perfect-diode-rho-selection).

## Bounded Hardware Status

The bounded condition projects `ConvWeight_*` and `DenseWeight_*` to
`[1e-5,1e-4]`. It independently screens:

- `bounded_uniform` sampled from `[1e-5,1e-4)`; and
- `bounded_kaiming_uniform` with gain `1`.

Both initializers require independent ordinary-MNIST rho selection for all 18
architecture x scheme x optimizer surfaces. A single initializer is then
chosen globally by matched 2% loss-plateau wins.

The repository retains an immutable Conv1/Conv2 predecessor config, but no
completed all-depth selector or global winner is recorded. Bounded
medium-affine paper runs therefore remain pending.

The main open bounded-weight question is whether the training difficulty comes
from the absolute scale of the weight interval or from its limited dynamic
range `wmax/wmin`, and how that dependence changes with amplification. The
current `[1e-5,1e-4]` evidence does not separate those effects.

Authority:
[`perfectdiode_bounded_weight_protocol.md`](perfectdiode_bounded_weight_protocol.md).

## Paper-Run Status

The deterministic medium-affine paper grid has 36 rows per model seed:

```text
3 architectures x 3 schemes x 2 optimizers x 2 weight contracts
```

Each row must import its matching ordinary-MNIST LR handoff unchanged. The
12 user-directed Conv1/Conv2 vectors may be used for current unbounded result
rows. Ordinary-MNIST validation accuracy is never reported as paper evidence.

Current epoch decisions:

| Architecture | Paper budget |
|---|---:|
| Conv1 | 10 epochs |
| Conv2 | 30 epochs |
| Conv3 | 30 epochs |

The first wide-range batch used model seed `0`, loader seed `0`, and affine
seed `1729`. All 18 rows completed their fixed budgets. They trained on the
deterministic medium-affine 55,000/5,000 train/validation split, selected the
maximum-validation-accuracy checkpoint, and read the official 10,000-example
medium-affine test split exactly once from that checkpoint. The result is
reported above, subject to the newly identified historical bias-contract
confound and the recorded one-seed limitation. The later zero-bias legacy-SGD
controls found no Conv2/Conv3 validation deficit after retraining, but they did
not read the official test and therefore do not replace these paper rows.

Filip authorized the Conv3 vectors selected under `weight_max=null` for use
with the `[0,100]` paper contract. That source-contract mismatch remains
explicit provenance rather than being silently erased. The completed batch
used the 18 exact seed-0 wide-range configs under
`configs/conv/paper_medium_affine_perfectdiode_wide_seed0_20260729_v1/`.
Bounded medium-affine rows remain blocked on the global bounded initializer and
LR handoff.

## Compute Targets

This table records stable access and routing information, not live occupancy.
Always check the target immediately before launch.

| Resource | Launcher target | Access / workdir | Preferred work |
|---|---|---|---|
| Local foreground | `local` | current checkout | unit tests and foreground smokes |
| Local GPU tmux | `main` | session `main`; `/home/filip/server_code` | Conv1, smokes, diagnostics |
| Akib | `akib` | SSH alias `akib`; `/home/filiposana/server_code` | Conv1/Conv2 surfaces |
| Trex | `trex` | `filip@trex`; `/home/filip/server_code` | Conv2/Conv3 surfaces and long confirmations |
| Jean Zay | `jean-zay` | SSH alias `jean-zay`; Slurm `fmu@v100` | Conv3, arrays, and scheduled production |

Jean Zay source checkout:
`/lustre/fswork/projects/rech/umg/$USER/server_code`.

Jean Zay result root:
`/lustre/fsn1/projects/rech/fmu/$USER/server_code/results`.

Keep one scientific surface on one recorded target. Host availability must not
change the surface's config, batch size, initialization, cohorts, or training
order.

## Next Actions

1. Before deep EqProp training, distinguish the frozen-current rule from
   full-cost `E+beta*C` EqProp. For frozen-current legacy Conv3, validate a
   float64 pilot at base beta hat `3e-10` (injected `3.4463442e-5`) with
   per-layer direction, scale, residual, and matched-zero displacement gates.
   The same point loses the first-layer numerator in float32; extra T/K is not
   a remedy. If float32 is required, change the contrast arithmetic or signal
   transport before training.
2. Determine whether bounded-weight training difficulty is caused by the
   absolute weight scale or by `wmax/wmin`, and measure how this depends on
   amplification. Use the depth-by-bound study described in the
   [legacy amplification mechanism note](legacy_amplification_mechanism_study_20260807.md#next-study-scaling-across-depth-and-conductance-bounds).
3. Quantify feed-forward dominance, for example through physical round-trip
   gain or local state-Jacobian transfer, and test whether it tracks the
   amplification performance ordering across depths and weight bounds.
4. Analyze the saved LR-counterfactual traces and checkpoints for layerwise
   relative updates, projection activity, and late-epoch gradients. Do not
   allocate a longer Conv3 Conv-LR/3 run unless that read-only analysis exposes
   a new non-plateau mechanism.
5. Finish terminal wide-range rho review, then revisit the bounded initializer
   execution in light of the absolute-scale-versus-dynamic-range result.
6. Decide later multi-seed scope after the weight-bound question and matching
   rho handoffs are resolved.
7. Treat any remaining bias-contract comparisons as secondary follow-up. Add
   Adam or additional medium-affine zero-bias controls only if needed to test a
   specific residual scheme-by-bias interaction.

## Bias Conclusion

Bias choices can strongly affect a trained checkpoint under post-hoc removal,
but matched retraining shows no material performance deficit when trainable
biases are absent. Biases therefore do not significantly alter retrained
performance and do not explain the aggregate `legacy > ours > baseline`
ordering. Remaining bias-contract questions are lower priority than resolving
the weight-bound dependence.
