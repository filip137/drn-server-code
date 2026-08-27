# Initialization protocols

Initialization is part of the scientific contract in this worktree. Every
study plan must name the exact initialization path, configs, input artifacts,
device data, seeds, mapping, and assignment. Do not select inputs by recency.

## Matched-comparison rule

HWA and non-HWA controls must begin from the same logical DRN state. Likewise,
post-deployment dense, Tiki-Taka, and LoRA recovery arms must begin from the
same explicitly named physical-device realization. If initialization is the
intervention, state that in the hypothesis and include the matched controls
needed to isolate it.

At minimum, record:

- teacher or base checkpoint path and SHA-256;
- measured-device dataset SHA-256;
- model, data, split, and assignment seeds;
- DRN topology, encoding, conductance bounds, amplification, and solver;
- mapping, layout, cohort, preprocessing, and initial pulse index;
- HWA modifier and its seed, when present; and
- metrics immediately before and after device programming.

## Supported protocol families

### Teacher-mapped DRN

Map an explicitly frozen teacher or base checkpoint into a perfect-diode DRN,
then evaluate the mapped state before further training. Use the same mapped
state for the HWA and non-HWA arms. The current strict examples are
[`ideal_teacher_map.json`](../examples/mnist_relu_drn/wan_cmo_teacher_initialized/ideal_teacher_map.json)
and its matched
[`hwa_add_normal.json`](../examples/mnist_relu_drn/wan_cmo_teacher_initialized/hwa_add_normal.json)
control. The scientific protocol and its limitations are documented in
[`mnist_wan_cmo_teacher_initialized.md`](mnist_wan_cmo_teacher_initialized.md).

### Measured programmed or RESET state

When initializing measured synapses, use the verified dataset described in
[`synapse_data.md`](synapse_data.md). Declare whether target conductances are
programmed onto reachable measured states or whether every device starts at
its measured pulse-zero/RESET state. Also declare the physical layout and
common-window rule. These paths answer different questions and must not be
presented as equivalent initializations.

The RESET comparison contract is documented in
[`mnist_relu_drn_reset_factorial.md`](mnist_relu_drn_reset_factorial.md). A
zero-learning-rate measured initialization example is
[`initialization_hold_1ep.json`](../examples/mnist_relu_drn/eight_device_cohort_a_one_pulse_down/initialization_hold_1ep.json).

### IBM OM off-chip HWA and deployment

The IBM OM HWA pilot starts every training arm from the same exact named
bounded perfect-diode DRN checkpoint. The optimizer retains that logical state
as clean FP32 master weights. Compact HWA does not create a persistent device
state: each minibatch is a fresh hypothetical program-and-verify endpoint on
the same fixed cell identities, and the clean master is restored before the
optimizer step.

Physical selection and validation are different. Each fixed OM cell starts
directly at its sampled lower persistent bound, representing a fully RESET
state after boundary conditioning. The adaptive controller then programs the
target with a separately budgeted maximum of 128 SET/RESET pulses. Both the
apparent endpoint used by the DRN and the persistent endpoint required for
continuation are saved. Conditioning cost is excluded from the target pulse
budget but the initialization rule is recorded.

The published-corruption and repaired HWA arms begin from one literal sampled
assignment. Repair replaces only cells identified as corrupt/stuck; all
healthy sampled parameters remain identical. This is the declared
intervention and must not be repeated after observing network metrics.

The exact seeds, checkpoint hashes, device-model prerequisite, and deployment
handoff are frozen in
[`ibm_om_hwa_program_verify_pilot.md`](ibm_om_hwa_program_verify_pilot.md).
A later on-chip recovery arm must load the saved persistent deployment bundle.
It must not remap the logical checkpoint, redraw the physical assignment, or
start from the apparent endpoint as though it were the persistent state.

### Four-device and eight-device dual-rail initialization

When comparing the four-device `single` encoding with the eight-device
`differential` encoding, the common starting object is the explicitly named
signed teacher or base matrix. The physical conductance tensors are different
representations and must not be copied between encodings or compared as if
equal conductances implied equal DRN states.

For each layer, first form the one shared logical target

```text
u = W / max(abs(W)).
```

Declare the layer scale fraction `f` and the range-placement rule before
programming. For one logical weight `u_ij`, define the signed targets of its
four dual-rail edges as

```text
z++ = z-- =  u_ij
z+- = z-+ = -u_ij.
```

The normalized device fractions are then:

| Scheme | Fraction assigned to each rail edge |
| --- | --- |
| Four devices, `single` | `q_e = q0 + f max(z_e, 0)` |
| Eight devices, `differential` | `q_e+ = f max(z_e, 0)` and `q_e- = f max(-z_e, 0)` |

That table describes the established no-fixed-symmetry-reference mappings.
For the IBM OM deployment investigation, reference policy is an additional
axis. A four-device fixed-`r` map uses `[[a,r],[r,a]]` for a positive logical
weight and swaps the entries for a negative weight; its logical contrast is
`a-r`, but the paired voltage denominator receives `a+r`. An eight-device
fixed-`r` map uses `G_a=r+delta L` and `G_r=r` on each rail edge; its transfer
is `G_a-G_r`, while its denominator loading is `G_a+G_r`. Never substitute the
signed difference for either physical conductance sum.

For the four-device mapping, `q0 = 0` for lower placement and
`q0 = (1 - f) / 2` for centered placement. The differential mapper requires
lower placement. Thus a positive weight raises the `++` and `--` cells in the
four-device scheme; a negative weight raises the `+-` and `-+` cells. In the
eight-device scheme, the sign instead selects which member of each local
`G+`/`G-` pair is raised. At `f = 1`, for example, `u = -0.2` gives fractions
`(0, 0.2, 0.2, 0)` in `(++,+-,-+,--)` order for four devices.

Do not map those fractions independently onto each measured curve. For every
declared device group, compute one reachable window from the assigned raw
curves:

```text
common_low  = max(curve_min_k)
common_high = min(curve_max_k)
common_span = max(common_high - common_low, 0)
target_k    = common_low + q_k * common_span.
```

The grouping is the essential difference:

| Scheme | Required measured group | Required config |
| --- | --- | --- |
| Four devices | All four `++`, `+-`, `-+`, and `--` cells belonging to one logical synapse | `encoding: "single"` and `initial_target_mapping: "dual_rail_quad_common_window"` |
| Eight devices | Each `G+`/`G-` pair on one of the four dual-rail edges; four two-cell windows per logical synapse | `encoding: "differential"` and `initial_target_mapping: "paired_affine_common_window"` |

For the four-device scheme, also freeze the model-local layout map. The
current MNIST DRN uses `halves` for `base.dense_weight.0` and `paired` for
`base.dense_weight.1`. A missing or incorrect layout groups unrelated cells
and invalidates the initialization.

The four-way window is mandatory for new four-versus-eight device-facing
comparisons. `per_device_affine` leaves four unrelated floors, while the
legacy `dual_rail_pairwise_common_window` permits the two source rows of a
logical synapse to acquire different baselines and spans. Either can create a
large signed-drive error before learning. Retain those mappings only as
explicit initialization ablations.

After target construction, start cohort-A cells at the declared RESET pulse
(normally `initial_pulse_index: 0`) and perform the one accounted
global-nearest target write. Project each target to the nearest state on its
assigned measured trace. This final projection can break exact four-way or
pairwise cancellation, so the realized programmed state, rather than the
nominal target, is the state from which training and the initialization
evaluation must begin. Cohort-B deployment must apply the same scheme-specific
grouping to the explicitly named cohort-A checkpoint on the independently
assigned cohort-B curves.

An empty common window has zero span and cannot encode a nonzero target. The
current backend records it, uses a zero-span midpoint target, and projects
each device independently. Every study must predeclare whether its empty-window
rate is an exclusion criterion or a measured condition to retain; never hide
it by dropping devices after seeing the result.

#### Keeping the comparison matched

Use the same teacher/base hash, normalized logical matrices, dataset and
cohort, preprocessing, split and assignment seeds, calibration examples,
solver, and programming budget. The encoding changes the number and grouping
of physical cells, so assignment hashes will differ; record every hash and
keep the assignment algorithm fixed.

Choose one scale-selection question in advance:

- To compare the two physical representations at the same requested logical
  amplitude, predeclare the same layer fractions `f` in both arms.
- To compare the best initialization attainable by each representation,
  search the same fraction grid and objective in both arms, evaluate every
  candidate after its scheme-specific measured projection, and report the
  selected fractions as part of the intervention.

Do not select one arm in the nominal conductance domain and the other after
measured projection. The current runtime previews measured candidates for
`dual_rail_quad_common_window`, but the
`paired_affine_common_window` differential path selects its scale in the
nominal domain and recalibrates only after programming. Until that path is
made symmetric, a strict new comparison must use predeclared fixed fractions
or explicitly classify scale selection as an initialization intervention.

At minimum, record per layer and per arm:

- the teacher absolute maximum, selected fraction, range placement, and
  nominal-versus-measured selection domain;
- common-window empty fraction and span minimum, mean, and maximum;
- assignment hashes, projection RMS and maximum error, and realized pulse
  range;
- the realized signed-drive and total-conductance statistics; and
- validation accuracy, teacher agreement, KL, raw score scale, and fitted
  output gain immediately after programming and before any update.

The earlier mismatch and the corrected quad-window result are analyzed in
[`dual_rail_input_four_vs_eight_devices.md`](dual_rail_input_four_vs_eight_devices.md).
The four-way window improved initial validation accuracy from `34.20%` to
`61.48%`, but the eight-device pairwise construction still initialized at
`89.12%` because it retained a wider reachable span and local differential
cancellation. Treat that remaining gap as an initialization/device-range
effect, not as evidence for or against on-chip recovery.

### Post-deployment on-chip recovery

Start every recovery arm from one explicit named deployed checkpoint; do not
remap independently per arm. Evaluate and record that common starting state
before applying an update.

For Tiki-Taka, initialize the slow array from the declared DRN conductances and
the fast accumulation array at zero, following
[`tiki_taka_gradient_accumulation.md`](tiki_taka_gradient_accumulation.md#initializing-tiki-taka-from-direct-fp32-conductances).

For LoRA, keep the named deployed base fixed and initialize the adapter so its
initial network contribution is exactly zero, following
[`passive_low_rank_adapter.md`](passive_low_rank_adapter.md#initialization-and-named-checkpoints).

### IBM OM cell-aware exact-bounds oracle

The `cell_aware_exact_bounds_quad` mapping is an explicit oracle ablation, not
a deployable uncharacterized-array protocol. It consumes every assigned cell's
simulated logical minimum and maximum, intersects each range with the DRN's
normalized `[0, 1]` coordinate, and maps the sign-selected rail into that
cell's own usable span. It intentionally retains different floors and spans
across the four cells; do not describe it as common-window or RESET-relative
initialization.

For nine-level operation, reconstruct one logical shadow value per canonical
quad, round `4u` half away from zero to an integer in `[-4, 4]`, and apply the
result before compact endpoint sampling or pulse-resolved programming. The
continuous control uses `4u` without rounding. Do not clip, narrow, reassign,
or replace the resulting targets. The immutable codebook artifact must retain
the exact bound tensors, usable ranges, code indices, targets, hashes, and
sign diagnostics and state unambiguously that hidden bounds generated the
targets. The formal protocol is
[`ibm_om_cell_aware_exact_bounds.md`](ibm_om_cell_aware_exact_bounds.md).

### IBM OM standard-level deployment screen

The scheme-refitted ideal standard-level control uses two distinct physical
origins while sharing one level-spacing definition. No-fixed-`r` arms anchor
the active grid at each identity's sampled RESET/lower state. Fixed-`r` arms
retain each identity's exact sampled `r` and center the active grid on it;
only `a` may move to an in-bound integer level. In the repository coordinate
`x=(a+1)/2`, adjacent targets are exactly
`4 * (nominal_dw_min / 2)` apart. This is a conductance-coordinate interval,
not four constant-size pulses.

Logical sign remains a dual-rail placement choice. The four-device fixed-`r`
arm therefore retains `[[a,r],[r,a]]` or its sign-swapped form, and the lower
half of a centered cell's grid is recorded as update headroom rather than
counted again as extra logical sign levels. Eight-device transfer uses the
active/reference difference, while loading always uses their sum.

Each scheme searches the same scale grid after its own standard-level mapping
on the development assignment. Its scale pair and positive KL gain are then
frozen on held-out assignments. This is a scheme-optimized initialization
comparison, not the historical shared-RESET comparison. The strict contract
is documented in
[`ibm_om_standard_level_scheme_screen.md`](ibm_om_standard_level_scheme_screen.md).

### IBM OM raw-active-state deployment candidate

The frozen raw-active candidate represents every DRN conductance with
one independently evolving raw active state and converts all cells with one
frozen array-wide affine coordinate. It is the candidate single-device scheme:
`w=a`, with the physical active conductance `a` included in the
conductance-sum voltage denominator. More precisely, it is the
four-device/no-fixed-`r` cell of the deployment screen. It is distinct both
from the four-device fixed-reference pattern `[[a,r],[r,a]]`, whose logical
contrast is `a-r` but whose paired denominator loading is `a+r`, and from the
eight-device per-edge differential topology. For a four-cell signed weight,
the raw candidate freezes one global differential budget at the development
assignment's 90% quad support threshold and permits one exact shared baseline
to vary by eligible quad.
Unsupported quads are reassigned or fail; they are not locally rescaled. This
target construction does not, by itself, define network initialization. The
logical-weight mapping, QAT code spacing, and learning-rate derivation must be
separately versioned and must consume the same frozen coordinate and global
differential budget; they must not introduce cell-wise normalization or
recompute the scale on a replacement array. The protocol and its
implementation gate are documented in
[`ibm_om_raw_active_state_program_verify.md`](ibm_om_raw_active_state_program_verify.md).
Its coordinate, baseline, and support choices are under matched review in
[`ibm_om_deployment_scheme_investigation.md`](ibm_om_deployment_scheme_investigation.md);
do not treat it as the selected primary deployment path.
The global differential and variable-baseline construction changes only the
targets supplied to P&V. Boundary conditioning, apparent verification, SET
below target, RESET after overshoot, tolerance, and pulse budget remain the
frozen scalar-cell procedure in that deployment protocol.

## Checkpoint roles

- Use `checkpoints/weights.pt` for the explicitly selected named state passed
  into a new arm.
- Use `checkpoints/resume.pt` only for exact continuation of the same run.
- Validate initialization provenance before mutating a model. A hash, topology,
  device-data, mapping, or assignment mismatch must fail closed.
