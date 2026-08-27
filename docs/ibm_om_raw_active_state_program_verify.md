# IBM OM raw-active-state deployment protocol

- Status: authoritative frozen deployment protocol; runtime implementation and
  experimental evidence are deferred
- Scope: shared-scale four-cell differential target construction and
  deployment program-and-verify for each DRN conductance
- Device model: AIHWKit 1.1.0 `ReRamArrayOMPresetDevice`
- Evidence class: normalized, hardware-derived fitted model; not raw pulse
  traces and not an absolute conductance calibration in siemens

## Purpose and scope

This protocol defines how one independently evolving IBM optimized-material
(OM) cell represents one non-negative DRN conductance, how four such cells
represent one signed DRN weight with a global differential scale and a
quad-specific common-mode baseline, and how each cell is programmed to its
resulting target. It replaces the nominal `q = (w + 1) / 2` coordinate for
future IBM OM work. It does not reinterpret the completed [`q`-based
exact-bounds study](ibm_om_cell_aware_exact_bounds.md), and it does not define
DRN initialization, quantization-aware training (QAT), or a learning-rate
schedule.

The `D_90` and variable-`B_Q` decision changes target construction only. Once
the four scalar targets have been constructed and pass preflight, deployment
uses the unchanged lower-to-target P&V plant, controller, tolerance, and pulse
budget specified below. It does not introduce a quad-level pulse controller or
renormalize a target during programming.

The AIHWKit preset is a fitted normalized device model. It does not supply a
unique microSiemens origin or scale. Accordingly, `g` below is a shared
model-based OM conductance coordinate. A result in `g` units must not be
reported as a physical conductance without a separate versioned calibration.
The upstream preset semantics are documented in the
[AIHWKit device-preset API](https://aihwkit.readthedocs.io/en/latest/api/aihwkit.simulator.presets.devices.html).

## One active cell per DRN conductance

For cell `i`, let:

- `a_i` be the persistent pulsed active state;
- `a_min,i` and `a_max,i` be its sampled persistent bounds;
- `d_up,i` and `d_down,i` be its sampled SET and RESET response parameters;
  and
- `r_i` be AIHWKit's sampled fixed reference/symmetry correction.

AIHWKit exposes the tile-facing logical state

```text
w_i = a_i - r_i.
```

This successor protocol does not interpret `w_i` as one physical conductance
and does not interpret `r_i` as a second programmable member of a differential
pair. It represents the DRN conductance with `a_i` alone. The fixed `r_i` is
saved as unused model provenance so the original AIHWKit identity remains
auditable, but it is excluded from state conversion, targets, bounds,
controller observations, and acceptance decisions. Changing only `r_i` must
therefore leave every numerical state, controller decision, and outcome
unchanged, although the provenance artifact must disclose the changed value.

Every DRN conductance has its own `a_min,i`, `a_max,i`, `d_up,i`, `d_down,i`,
cycle-to-cycle response, write variation, construction seed, and persistent
state. No pulse parameter or coordinate scale is pooled across the cells that
form a signed logical weight. Only the shared-baseline four-cell target
construction below couples their requested targets.

## Frozen array-wide coordinate

### Calibration array

Freeze one offset and scale from the complete repaired development assignment,
not from individual cells and not independently for each later array:

```text
A_min = min_i(a_min,i)
A_max = max_i(a_max,i)
S     = A_max - A_min
g_i   = (a_i - A_min) / S.
```

The calibration artifact is immutable:

| Field | Frozen value |
| --- | --- |
| Assignment seed | `84001` |
| Corruption policy | `counterfactual_repaired` |
| Binding order | `base.dense_weight.0`, `base.dense_weight.1` |
| Binding shapes | `[1568, 100]`, `[100, 20]` |
| Number of cells | `158800` |
| Population NPZ SHA-256 | `7e0bcfb76a0a613f0ba319786a39fdb540cce0a74f15d9fb007426a46ad49d2f` |
| Population fingerprint | `4887ad89abdd16193448c54a0cbe97be915cb4b9bc04cf1151c5e2a96ee5a3ac` |
| `A_min` | `-3.455834150314331` |
| `A_max` | `2.5384607315063477` |
| `S` | `5.994294881820679` |
| `1 / S` | `0.16682529300197937` |

For every cell, use that same affine map:

```text
g_min,i = (a_min,i - A_min) / S
g_max,i = (a_max,i - A_min) / S
Delta g_i = Delta a_i / S.
```

The pulse plant continues to evaluate the original IBM equations in `a`
using the sampled raw parameters. The affine conversion is only a common
coordinate change for DRN state, targets, verify observations, tolerances,
and reporting; it does not refit or rescale a cell internally.

### Cell-wise normalization is prohibited

Do not use

```text
(a_i - a_min,i) / (a_max,i - a_min,i).
```

That map would force every cell to span `[0, 1]` and would make the same
numerical update mean a different raw active-state change in every cell. It
would erase the device-to-device span and pulse-scale variation that the OM
preset is intended to retain. Under the required array-wide map, every cell
has the same offset and unit, while its usable interval and pulse response
remain heterogeneous.

For scale context, the development array's median raw interval is
`[-1.0001208782, 1.0011744499]`, with median span `2.0025067329`. Its median
cell therefore occupies only about one third of the global `g` interval. The
arithmetic `1 / Delta g_nominal = 63.1643` describes the entire cross-cell
global envelope, not the number of pulses that any typical cell can use over
its own span.

### Reuse on later arrays

Reuse the frozen `A_min`, `A_max`, and `S` on every validation, replacement,
or deployed array. Never recompute them from the new array. Consequently:

- a later valid cell may have `g_max,i > 1`; retain that state and do not clip
  it;
- a healthy cell whose support extends below `g = 0` is incompatible with a
  passive non-negative conductance assignment under this frozen coordinate;
  and
- array-to-array changes in bounds and pulse sizes remain physical-model
  variation rather than being hidden by renormalization.

The existing repaired assignment `85001` is a validation fixture. Its raw
support `[-3.295332670211792, 2.4219136238098145]` maps with the frozen
development anchors to `[0.0267757064, 0.9805569946]`. Its population NPZ
SHA-256 is
`8cf02438727c4090c4721976e09eee066e2408dad9c1c8cd8328ef3268965cab`
and its population fingerprint is
`2721e735ba6722df27a879a5373de2d1da02481cb5eec05b592f967d27e74310`.
These held-out values validate the transform; they did not select its anchors.

## Assignment and defect policy

The functional deployment path and the matched defect stress control must be
declared before network metrics are inspected.

### Functional path

Reassign a cell before target construction if it is:

- corrupt or stuck, including a collapsed interval with zero SET and RESET
  response;
- non-finite; or
- structurally incompatible because `g_min,i < 0`.

Replacement must consume a predeclared donor stream. Save the original
identity, reason, donor identity, construction seeds, donor-stream seed, and
all original and replacement parameters. Continue until an eligible donor is
obtained or fail the assignment; do not clamp a bad cell and do not choose a
donor using network performance.

### Matched stress control

Retain the original identity at the corresponding site. A corrupt zero-step
cell remains an explicit persistent stuck-device outcome. A healthy identity
with `g_min,i < 0` is recorded as a structural passive-assignment failure and
is not pulsed through negative conductance states. This control must preserve
the functional path's remaining assignment and scientific inputs so the
reassignment intervention is identifiable.

The OM fit publishes a corruption probability of `0.1348`. The repaired
development fixture has zero active corrupt cells but retains `21482`
published-corrupt identities in its provenance; the repaired held-out fixture
retains `21443` such identities. Repair is therefore an explicit simulated
defect-management control, not evidence that fabricated arrays contain no
defects.

## Pulse plant in the raw active state

For a healthy cell with the usual OM signs `a_min,i < 0 < a_max,i`, evaluate
one fixed-amplitude pulse in `a` before applying the global affine map. With
independent standard-normal cycle draw `xi_i,k`, SET/up is

```text
response_up = d_up,i * (1 - a_i,k / a_max,i + dw_min_std * xi_i,k)
a_i,k+1    = clamp(a_i,k + response_up, a_min,i, a_max,i),
```

and RESET/down is

```text
response_down = d_down,i * (1 - a_i,k / a_min,i + dw_min_std * xi_i,k)
a_i,k+1      = clamp(a_i,k - response_down, a_min,i, a_max,i).
```

An implementation must retain the explicit plant's defensive zero-normalizer
guards for an unexpected non-positive upper or non-negative lower bound. Such
a cell is an assignment-policy event, not a reason to alter these equations.
Pulses in a requested batch are evaluated sequentially because the nonlinear
response changes after every pulse.

The OM preset constants are:

| Quantity | Raw `a` value | Shared `g` value |
| --- | ---: | ---: |
| Nominal `dw_min` | `0.0949` | `0.01583172030588784` |
| Half-step acceptance tolerance | `0.04745` | `0.00791586015294392` |
| Cycle-to-cycle `dw_min_std` | `0.4158` | multiplicative factor, unchanged |
| Apparent write-noise standard deviation | `1.4113 * 0.0949 = 0.13393237` | `0.02234330686769951` |

With independent standard-normal write draw `zeta_i,k`, the apparent active
state after a pulse is

```text
a_apparent,i = a_persistent,i + 0.13393237 * zeta_i,k
g_apparent,i = (a_apparent,i - A_min) / S.
```

Do not clip the apparent or persistent value before a controller decision or
before logging it. Keep both states: the controller sees only the apparent
`g`, while exact continuation loads the persistent `a` and the saved RNG
state.

## Shared-scale four-cell differential targets

### Signed coordinate and cell order

For one logical quad `Q`, keep the canonical cell order

```text
(++, +-, -+, --).
```

Define its signed differential coordinate and common-mode baseline as

```text
D_Q = (g_++ - g_+- - g_-+ + g_--) / 2
B_Q = (g_++ + g_+- + g_-+ + g_--) / 4.
```

For a requested signed target `D*_Q`, construct the four cell targets as

```text
g*_++ = g*_-- = B_Q + D*_Q / 2
g*_+- = g*_-+ = B_Q - D*_Q / 2.
```

The reconstruction therefore returns exactly `D*_Q`, independently of
`B_Q`. At the maximum negative value `D*_Q = -D_90`, the target order is

```text
(B_Q - D_90/2, B_Q + D_90/2,
 B_Q + D_90/2, B_Q - D_90/2).
```

Here `D_Q` is the logical differential magnitude; the displacement of one
cell from `B_Q` is `D_Q / 2`. Do not use `D` sometimes for the logical
contrast and sometimes for the one-cell displacement.

The same numerical `D*_Q` must mean the same absolute difference in the
frozen array-wide `g` coordinate for every quad. A target mapper or QAT
codebook may use a smaller magnitude or a shared global code spacing, but it
must not replace `D*_Q` by a quad-specific fraction of that quad's span.
Programming noise may make the realized differential coordinates differ;
that deviation is an outcome and not a new local scale.

### Global 90% differential budget

The support unit is a complete logical quad, not an individual cell. For a
quad whose four exact transformed supports are `[g_min,i, g_max,i]`, define

```text
C_Q = min_i(g_max,i) - max_i(g_min,i).
```

`C_Q` is the largest symmetric logical magnitude for which some one shared
baseline lets every cell support both polarities. This both preserves later
sign changes and avoids treating four unrelated cell ranges as one device.

Freeze the global budget from the repaired development assignment after its
physical quad layout has been fixed, and before inspecting network metrics.
For `N` finite candidate quads, let

```text
M = ceil(0.90 N)
```

and sort their capacities as `C_(1) <= ... <= C_(N)`. Use the largest
threshold supported by at least `M` quads:

```text
D_90 = C_(N - M + 1).
```

This is the lower-tail 10th-percentile order statistic, despite the shorthand
"90% supported." Do not use the upper 90th percentile.

For the frozen development assignment and canonical layouts
`base.dense_weight.0: halves` and `base.dense_weight.1: paired`, the
bound-only calculation gives:

| Quantity | Frozen value |
| --- | ---: |
| Candidate quads `N` | `39700` |
| Required eligible quads `M` | `35730` |
| Order-statistic index, one-based | `3971` |
| Global differential budget `D_90` in `g` | `0.10354409442884083` |
| One-cell full-scale displacement `D_90 / 2` in `g` | `0.051772047214420414` |
| Global differential budget in raw `a` | `0.6206738352775576` |
| Aggregate exact-support yield | `35730 / 39700 = 0.900000` |
| First-binding yield | `35270 / 39200 = 0.8997448979591837` |
| Second-binding yield | `460 / 500 = 0.92` |

The calculation consumes only the frozen development population bounds,
binding shapes, and declared rail layouts. It does not consume trained
weights, validation identities, P&V outcomes, or accuracy. Freeze `D_90` for
all later arrays; do not recompute it to make each array appear to have the
same yield. The realized held-out eligibility is a reported transfer outcome.

This 90% criterion means exact two-polarity target support. It does not assert
90% apparent P&V acceptance under the 128-pulse budget. Reachability,
acceptance-window intersection, and finite-budget stochastic success remain
separate outcomes under the artifact contract below.

### Variable common-mode baseline

For a quad eligible at `D_90`, its full-range feasible baseline interval is

```text
B_low,Q  = max_i(g_min,i) + D_90 / 2
B_high,Q = min_i(g_max,i) - D_90 / 2.
```

Choose one baseline for all four cells by the deterministic midpoint rule

```text
B_Q = (B_low,Q + B_high,Q) / 2.
```

This maximizes the quad's minimum residual headroom after reserving the global
differential range. Keep `B_Q` fixed while constructing every initial target
in `[-D_90, D_90]`; in particular, do not choose a new baseline from the
sign or magnitude of the current trained weight. All four cells must share
exactly the same requested `B_Q`. Four independent baselines would leak the
offset

```text
(B_++ - B_+- - B_-+ + B_--) / 2
```

into the represented weight.

The common mode cancels from the signed reconstruction, but it is not
electrically inert:

```text
g*_++ + g*_+- + g*_-+ + g*_-- = 4 B_Q.
```

Consequently variable `B_Q` changes node loading, voltage equilibria, current,
and power. HWA and inference must use the assigned per-quad baselines rather
than remove them with a claimed scalar normalization.

If `C_Q < D_90`, the quad is unsupported. Consume the predeclared donor or
reassignment stream, or record a structural target-assignment failure if it
is exhausted. Do not shrink `D_90`, change the code spacing, normalize by
`C_Q`, clamp a target, or retain a smaller-scale logical weight for that quad.
The assignment procedure and whether it optimizes quad grouping are separate
predeclared interventions; a random assignment must not be retrospectively
repacked to inflate the reported 90% yield.

The baseline construction uses exact characterized bounds outside the
realistic controller port. It is therefore an array-aware target provider,
not a RESET/read-only commissioning claim. The controller still receives none
of the bounds or the derived capacity.

## Per-cell target preflight

For a signed logical quad, this protocol supplies the four scalar `g*_i`
targets through the construction above. Any other eligible conductance must
receive its target from a separately declared provider using the same frozen
coordinate. All target generation occurs outside the program-and-verify
controller. Before pulsing, the deployment layer checks that every target is
finite, non-negative, and inside that cell's exact transformed persistent
support:

```text
0 <= g*_i
g_min,i <= g*_i <= g_max,i.
```

An invalid target is rejected as a target-contract failure with zero target
programming pulses. It is not projected, clipped, narrowed, or replaced. A
target above one is valid when the eligible cell's frozen-coordinate support
contains it.

The target provider may use characterized bounds to establish validity, but
the realistic controller never receives those bounds. A later network-level
mapper must separately declare which device information it consumes and how
its logical parameters enter the bounded global `D_Q` coordinate.

## Lower-to-target program-and-verify

The four members of a logical quad are programmed as four independent scalar
cells. Each receives its fixed `g*_i` from the shared-scale target construction
and then follows the same controller below. The controller does not observe
`D_Q`, `B_Q`, the other three cell states, or their histories. In particular,
allowing `B_Q` to vary between quads does not change the P&V algorithm.

### Boundary conditioning

For a new simulated trajectory, initialize the raw plant at `a = 0`, which is
inside every cell support in the frozen development and held-out fixtures. Do
not initialize through tile-facing `w = 0`, because that would make the raw
start depend on the otherwise unused `r_i`. A continuation instead loads its
explicit saved persistent `a` state.

Before target programming, apply RESET pulses until four consecutive
persistent raw changes are each smaller than `1e-6` of the nominal raw preset
span of two, or until 4096 conditioning pulses have been applied:

```text
abs(a_i,k+1 - a_i,k) < 2e-6.
```

The equivalent shared-coordinate threshold is
`3.336505860039587e-7`. Boundary conditioning has its own per-trajectory seed.
Record its persistent and apparent endpoint, pulse count, and success flag,
but exclude its pulses from the target-programming budget. A conditioning
failure stops the trajectory and remains an initialization failure rather
than a programming residual.

Persistent-state quiet detection is a simulation-side initialization tool. It
does not enlarge the capability of the target-programming controller.

### Canonical one-pulse verify controller

Use `tau_g = 0.00791586015294392`. At each verify, let `y_i` be the apparent
`g` value:

```text
if abs(y_i - g*_i) <= tau_g:
    accept
elif y_i < g*_i - tau_g:
    apply one SET pulse
else:
    apply one RESET pulse
verify again
```

Thus SET pulses continue one at a time while the apparent conductance is below
the window. A RESET pulse is requested after an apparent overshoot. Further
noise or overshoot may cause repeated polarity reversals; acceptance, rather
than the first reversal, terminates the procedure. This one-pulse controller
is the canonical precision protocol.

### Matched adaptive verify-cost control

Retain an adaptive controller only as a matched verify-cost control. Its first
step estimate is learned in `g` from a predeclared controller-calibration
identity partition; do not transform or reuse the historical `q`-coordinate
estimator. Freeze 41 direction/state bins over the calibration interval
`[0, 1]`, with empty-bin fallback

```text
s_fallback = 0.01583172030588784.
```

For a later `g > 1` observation, clamp only the estimator's bin lookup to its
edge bin; never clamp the state, target, or residual. After an observed batch,
the controller may update its trajectory-local estimate from apparent
displacement divided by pulse count.

For distance `D = abs(y_i - g*_i)`, choose

```text
n = floor(0.75 * D / max(s_hat, 1e-8))
n = clamp(n, 1, 32)
```

and force `n = 1` when `D` is at most two predicted steps. Apply every pulse
in the batch sequentially and truncate the batch to the remaining target
budget. Direction continues to be selected from the apparent residual after
each verify; batching does not expose intermediate persistent states.

### Termination

Both controllers stop on:

- apparent acceptance;
- a non-finite apparent state; or
- 128 total target-programming SET/RESET pulses.

The first verify counts as a verify operation. Conditioning pulses do not
consume the 128-pulse target budget. One-pulse and adaptive comparisons must
use the same eligible identities, targets, tolerance, conditioned starting
states, seed-derivation rules, and logging contract.

## Capability boundary

The realistic controller receives only:

- the continuous target `g*_i`;
- apparent verify values in the shared `g` coordinate;
- its own direction, pulse-count, verify, and reversal history; and
- the frozen population step-estimator artifact for the calibration
  partition.

It must not receive persistent `a`, `a_min,i`, `a_max,i`, `d_up,i`,
`d_down,i`, `r_i`, corruption flags, donor status, or the future RNG stream.
Eligibility and target-support checks occur outside the controller before its
port is created. Hidden parameters and persistent states remain on the
analysis and logging side.

## Outcome and artifact contract

Keep the following as separate facts; none implies another:

1. `exact_target_in_support`: the target lies inside the exact sampled
   persistent interval `[g_min,i, g_max,i]`;
2. `persistent_target_inside_conditioned_bounds`: when matched RESET/SET
   boundary copies are available on the analysis side, the target lies inside
   their paired persistent interval;
3. `acceptance_window_intersects_conditioned_bounds`: the interval
   `[g*_i - tau_g, g*_i + tau_g]` intersects that paired persistent interval;
   and
4. `apparent_accepted`: a noisy apparent verify value entered the acceptance
   window.

In particular, apparent acceptance must not be relabelled as exact persistent
reachability. A finite-pulse stochastic trajectory may also exhaust its
budget even when its target or window lies within the persistent bounds.

Every run must save:

- the frozen coordinate version, constants, calibration artifact hashes, and
  code/source hashes;
- assignment seed, binding order and shapes, population fingerprint, AIHWKit
  version, per-identity construction seed, and full sampled population;
- original/replacement identities, donor stream and seeds, defect reason, and
  both parameter sets for every functional reassignment;
- explicit conditioning and programming seeds and exact RNG continuation;
- raw and transformed per-cell bounds, pulse parameters, ignored `r_i`
  provenance, `C_Q`, `D_90`, baseline interval, selected `B_Q`, supplied
  target, and target hash;
- conditioning success, conditioning pulse cost, and conditioned apparent and
  persistent endpoints;
- every apparent verify, signed batch length, SET count, RESET count, total
  target pulses, verifies, reversals, and saturation diagnostics;
- final persistent `a` and `g`, final apparent `g`, and their residuals; and
- separate flags for invalid target, exact support, conditioned-bound
  reachability, acceptance-window intersection, apparent acceptance,
  corruption/stuck state, structural passive-assignment failure, reassignment,
  non-finite state, conditioning failure, and target-budget exhaustion.

Do not merge target-contract failures, assignment failures, conditioning
failures, program failures, and accepted endpoint residuals into one error
distribution. Preserve failed attempts and original identities.

## Deferred contracts and implementation gate

This protocol fixes the physical differential coordinate, its development
support budget, the variable-baseline construction, and programming of the
resulting per-conductance targets. A successor implementation or study must
separately version and review:

- DRN initialization and any network-level mapping into the bounded signed
  coordinate `D_Q`;
- global quantization or QAT code spacing inside `[-D_90, D_90]`;
- array-aware versus array-transfer training;
- learning-rate derivation in the fixed shared `g` unit; and
- persistent on-chip updates, including Tiki-Taka, LoRA, or direct SET/RESET
  rules.

Until the raw-active-state plant, global transform, target preflight,
controller port, estimator, artifact schema, and exact-replay tests implement
this contract, existing runtime results remain `q`-based and must not be
described as evidence for this successor protocol.
