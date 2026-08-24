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

## Checkpoint roles

- Use `checkpoints/weights.pt` for the explicitly selected named state passed
  into a new arm.
- Use `checkpoints/resume.pt` only for exact continuation of the same run.
- Validate initialization provenance before mutating a model. A hash, topology,
  device-data, mapping, or assignment mismatch must fail closed.
