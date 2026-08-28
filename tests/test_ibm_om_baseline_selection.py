from __future__ import annotations

from dataclasses import fields, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    BASELINE_POLICIES,
    INDEPENDENT_CELL_RESET_MEAN,
    MAPPING_SCHEMA,
    MAPPING_SCHEMA_VERSION,
    REFERENCE_ENFORCED_DESTINATION_COLUMNS,
    SHARED_DESTINATION_COLUMNS_RESET_MAX,
    SHARED_QUAD_RESET_MAX,
    DonorExhaustionError,
    InfeasibleBaselineError,
    PhysicalMapping,
    all_policy_feasibility_from_quads,
    apply_checked_physical_targets,
    apply_joint_repair_field,
    build_joint_repair_plan,
    build_layer_physical_mapping,
    build_physical_mapping,
    build_weight_error_decomposition,
    fit_weight_reconstruction_scales,
    hardware_instance_fingerprint,
    policy_baseline_quads,
    quad_column_loading,
    quad_contrast,
    quad_loading,
    quad_row_loading,
    quad_stack,
    save_physical_mapping,
    scatter_quads,
)
from model.resistive.interaction import DenseResistive
from model.variable.layer import LinearLayer
from model.variable.parameter import DenseWeight


def _native(unit: torch.Tensor) -> torch.Tensor:
    return 2.0 * unit - 1.0


def _physical_from_quads(quads: torch.Tensor, *, layout: str) -> torch.Tensor:
    rows, columns, rails = quads.shape
    assert rails == 4
    return scatter_quads(
        quads,
        shape=(2 * rows, 2 * columns),
        layout=layout,
    )


def _uniform_physical(
    shape: tuple[int, int], value: float
) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.float32)


def _layer(
    logical_weight: torch.Tensor,
    *,
    layer_index: int = 0,
    layout: str = "halves",
    policy: str = SHARED_DESTINATION_COLUMNS_RESET_MAX,
    scale_fraction: float = 1.0,
    nominal_dw_min: float = 0.1,
    conductance_min: float = 0.0,
    conductance_max: float = 1.0,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
    reset: torch.Tensor | None = None,
    reference_unit: torch.Tensor | None = None,
):
    physical_shape = (
        2 * int(logical_weight.shape[0]),
        2 * int(logical_weight.shape[1]),
    )
    lower = (
        _uniform_physical(physical_shape, 0.0) if lower is None else lower
    )
    upper = (
        _uniform_physical(physical_shape, 1.0) if upper is None else upper
    )
    reset = (
        _uniform_physical(physical_shape, 0.1) if reset is None else reset
    )
    reference_unit = (
        _uniform_physical(physical_shape, 0.2)
        if reference_unit is None
        else reference_unit
    )
    return build_layer_physical_mapping(
        logical_weight,
        layer_index=layer_index,
        layout=layout,
        policy=policy,
        scale_fraction=scale_fraction,
        nominal_dw_min=nominal_dw_min,
        conductance_min=conductance_min,
        conductance_max=conductance_max,
        cell_lower_unit=lower,
        cell_upper_unit=upper,
        reset_baseline_unit=reset,
        intrinsic_reference_native=_native(reference_unit),
    )


def _two_layer_mapping(
    *, policy: str = SHARED_DESTINATION_COLUMNS_RESET_MAX
) -> tuple[tuple[torch.Tensor, torch.Tensor], PhysicalMapping]:
    weights = (
        torch.tensor([[1.0, -0.5]], dtype=torch.float32),
        torch.tensor([[0.25], [-1.0]], dtype=torch.float32),
    )
    shapes = ((2, 4), (4, 2))
    lower = tuple(_uniform_physical(shape, 0.0) for shape in shapes)
    upper = tuple(_uniform_physical(shape, 1.0) for shape in shapes)
    reset = (
        _physical_from_quads(
            torch.tensor(
                [[[0.10, 0.20, 0.30, 0.50], [0.12, 0.22, 0.32, 0.52]]]
            ),
            layout="halves",
        ),
        _physical_from_quads(
            torch.tensor(
                [
                    [[0.11, 0.21, 0.31, 0.51]],
                    [[0.13, 0.23, 0.33, 0.53]],
                ]
            ),
            layout="paired",
        ),
    )
    references = tuple(_uniform_physical(shape, 0.4) for shape in shapes)
    mapping = build_physical_mapping(
        weights,
        policy=policy,
        scale_fractions=(0.5, 0.75),
        nominal_dw_min=0.1,
        conductance_min=0.0,
        conductance_max=1.0,
        cell_lower_units=lower,
        cell_upper_units=upper,
        reset_baseline_units=reset,
        intrinsic_references_native=tuple(_native(value) for value in references),
    )
    return weights, mapping


@pytest.mark.parametrize("layout", ("halves", "paired"))
def test_quad_helpers_round_trip_and_use_full_conductances(layout: str) -> None:
    matrix = torch.arange(24, dtype=torch.float64).reshape(4, 6) + 0.25
    quads = quad_stack(matrix, layout=layout)

    assert quads.shape == (2, 3, 4)
    torch.testing.assert_close(
        scatter_quads(quads, shape=matrix.shape, layout=layout), matrix
    )
    if layout == "halves":
        expected_first = matrix[(0, 0)], matrix[(0, 3)], matrix[(2, 0)], matrix[(2, 3)]
    else:
        expected_first = matrix[(0, 0)], matrix[(0, 1)], matrix[(2, 0)], matrix[(2, 1)]
    torch.testing.assert_close(quads[0, 0], torch.stack(expected_first))

    expected_contrast = (
        quads[..., 0] - quads[..., 1] - quads[..., 2] + quads[..., 3]
    ) / 2.0
    torch.testing.assert_close(quad_contrast(matrix, layout=layout), expected_contrast)
    torch.testing.assert_close(quad_loading(matrix, layout=layout), quads.sum(-1))
    torch.testing.assert_close(
        quad_row_loading(matrix, layout=layout),
        torch.stack(
            (quads[..., 0] + quads[..., 1], quads[..., 2] + quads[..., 3]),
            dim=-1,
        ),
    )
    torch.testing.assert_close(
        quad_column_loading(matrix, layout=layout),
        torch.stack(
            (quads[..., 0] + quads[..., 2], quads[..., 1] + quads[..., 3]),
            dim=-1,
        ),
    )


def test_four_baseline_policies_have_exact_declared_groups_and_roles() -> None:
    sign = torch.tensor([[1.0]])
    lower = torch.zeros((1, 1, 4))
    upper = torch.ones((1, 1, 4))
    reset = torch.tensor([[[0.10, 0.20, 0.30, 0.50]]])
    reference = torch.tensor([[[0.15, 0.25, 0.35, 0.45]]])

    independent, active, independent_groups = policy_baseline_quads(
        sign, lower, upper, reset, reference, policy=INDEPENDENT_CELL_RESET_MEAN
    )
    torch.testing.assert_close(independent, reset.to(torch.float64))
    assert torch.equal(active, torch.tensor([[[True, False, False, True]]]))
    assert torch.equal(independent_groups, torch.tensor([[[0, 1, 2, 3]]]))

    shared, _active, shared_groups = policy_baseline_quads(
        sign, lower, upper, reset, reference, policy=SHARED_QUAD_RESET_MAX
    )
    torch.testing.assert_close(shared, torch.full_like(shared, 0.50))
    assert torch.equal(shared_groups, torch.zeros_like(shared_groups))

    columns, _active, column_groups = policy_baseline_quads(
        sign,
        lower,
        upper,
        reset,
        reference,
        policy=SHARED_DESTINATION_COLUMNS_RESET_MAX,
    )
    torch.testing.assert_close(
        columns, torch.tensor([[[0.30, 0.50, 0.30, 0.50]]], dtype=torch.float64)
    )
    assert torch.equal(column_groups, torch.tensor([[[0, 1, 0, 1]]]))

    partner, partner_active, partner_groups = policy_baseline_quads(
        sign,
        lower,
        upper,
        reset,
        reference,
        policy=REFERENCE_ENFORCED_DESTINATION_COLUMNS,
    )
    torch.testing.assert_close(
        partner, torch.tensor([[[0.35, 0.25, 0.35, 0.25]]], dtype=torch.float64)
    )
    assert torch.equal(partner_active, active)
    assert torch.equal(partner_groups, column_groups)


@pytest.mark.parametrize(
    ("sign", "expected_baseline", "expected_active"),
    (
        (
            -1.0,
            (0.15, 0.45, 0.15, 0.45),
            (False, True, True, False),
        ),
        (
            0.0,
            (0.35, 0.25, 0.35, 0.25),
            (True, False, False, True),
        ),
    ),
)
def test_reference_policy_switches_roles_and_zero_ties_positive(
    sign: float,
    expected_baseline: tuple[float, float, float, float],
    expected_active: tuple[bool, bool, bool, bool],
) -> None:
    baseline, active, _groups = policy_baseline_quads(
        torch.tensor([[sign]]),
        torch.zeros((1, 1, 4)),
        torch.ones((1, 1, 4)),
        torch.full((1, 1, 4), 0.1),
        torch.tensor([[[0.15, 0.25, 0.35, 0.45]]]),
        policy=REFERENCE_ENFORCED_DESTINATION_COLUMNS,
    )
    torch.testing.assert_close(
        baseline, torch.tensor([[expected_baseline]], dtype=torch.float64)
    )
    assert torch.equal(active, torch.tensor([[expected_active]]))


def test_policy_feasibility_checks_unclipped_reference_in_own_and_partner_cells() -> None:
    sign = torch.tensor([[1.0]])
    lower = torch.zeros((1, 1, 4))
    upper = torch.ones((1, 1, 4))
    reset = torch.full((1, 1, 4), 0.1)
    reference = torch.tensor([[[0.2, 0.3, 1.1, 0.4]]])

    feasible = all_policy_feasibility_from_quads(
        sign, lower, upper, reset, reference
    )
    assert all(
        bool(feasible[policy].item())
        for policy in BASELINE_POLICIES
        if policy != REFERENCE_ENFORCED_DESTINATION_COLUMNS
    )
    assert not bool(feasible[REFERENCE_ENFORCED_DESTINATION_COLUMNS].item())
    assert not bool(feasible["all_policies"].item())

    reference_matrix = _physical_from_quads(reference, layout="halves")
    with pytest.raises(InfeasibleBaselineError, match="1 quads"):
        _layer(
            torch.tensor([[1.0]]),
            policy=REFERENCE_ENFORCED_DESTINATION_COLUMNS,
            reference_unit=reference_matrix,
        )

    bounded_reference = torch.tensor([[[0.2, 0.3, 0.6, 0.4]]])
    partner_limited_upper = torch.tensor([[[0.5, 1.0, 0.7, 1.0]]])
    own_limited_upper = torch.tensor([[[0.7, 1.0, 0.5, 1.0]]])
    for constrained_upper in (partner_limited_upper, own_limited_upper):
        constrained = all_policy_feasibility_from_quads(
            sign, lower, constrained_upper, reset, bounded_reference
        )
        assert not bool(
            constrained[REFERENCE_ENFORCED_DESTINATION_COLUMNS].item()
        )


def test_zero_contrast_is_exact_for_grouped_and_reference_policies() -> None:
    logical = torch.tensor([[1.0, -0.5]])
    reset = _physical_from_quads(
        torch.tensor(
            [[[0.10, 0.20, 0.30, 0.50], [0.11, 0.21, 0.31, 0.51]]]
        ),
        layout="halves",
    )
    reference = _physical_from_quads(
        torch.tensor(
            [[[0.15, 0.25, 0.35, 0.45], [0.16, 0.26, 0.36, 0.46]]]
        ),
        layout="halves",
    )

    independent = _layer(
        logical,
        policy=INDEPENDENT_CELL_RESET_MEAN,
        reset=reset,
        reference_unit=reference,
    )
    assert bool(torch.any(independent.baseline_contrast != 0.0))
    for policy in (
        SHARED_QUAD_RESET_MAX,
        SHARED_DESTINATION_COLUMNS_RESET_MAX,
        REFERENCE_ENFORCED_DESTINATION_COLUMNS,
    ):
        mapped = _layer(
            logical,
            policy=policy,
            reset=reset,
            reference_unit=reference,
        )
        assert torch.equal(
            mapped.baseline_contrast,
            torch.zeros_like(mapped.baseline_contrast),
        )


def test_destination_column_grouping_balances_contrast_not_column_loading() -> None:
    baseline_quads = torch.tensor([[[0.30, 0.50, 0.30, 0.50]]])
    baseline = _physical_from_quads(baseline_quads, layout="halves")

    torch.testing.assert_close(
        quad_row_loading(baseline, layout="halves"),
        torch.tensor([[[0.80, 0.80]]]),
    )
    torch.testing.assert_close(
        quad_column_loading(baseline, layout="halves"),
        torch.tensor([[[0.60, 1.00]]]),
    )
    assert torch.equal(
        quad_contrast(baseline, layout="halves"), torch.zeros((1, 1))
    )
    torch.testing.assert_close(
        quad_loading(baseline, layout="halves"), torch.tensor([[1.60]])
    )


@pytest.mark.parametrize(
    ("logical", "upper_quad", "expected_offset"),
    (
        (
            1.0,
            (0.80, 0.21, 0.21, 0.70),
            (0.50, 0.0, 0.0, 0.50),
        ),
        (
            -1.0,
            (0.21, 0.80, 0.70, 0.21),
            (0.0, 0.50, 0.50, 0.0),
        ),
    ),
)
def test_continuous_envelope_uses_only_the_sign_selected_active_pair(
    logical: float,
    upper_quad: tuple[float, float, float, float],
    expected_offset: tuple[float, float, float, float],
) -> None:
    reset = _physical_from_quads(
        torch.full((1, 1, 4), 0.20), layout="halves"
    )
    upper = _physical_from_quads(
        torch.tensor([[upper_quad]]), layout="halves"
    )
    mapped = _layer(
        torch.tensor([[logical]]),
        policy=INDEPENDENT_CELL_RESET_MEAN,
        reset=reset,
        upper=upper,
    )

    torch.testing.assert_close(
        mapped.selected_headroom_unit, torch.tensor([[0.50]])
    )
    torch.testing.assert_close(
        quad_stack(mapped.continuous_offset, layout="halves"),
        torch.tensor([[expected_offset]]),
    )


def test_standard_four_delta_rounds_half_away_and_caps_to_active_headroom() -> None:
    logical = torch.tensor([[1.0, 0.2]])
    shape = (2, 4)
    mapped = _layer(
        logical,
        policy=INDEPENDENT_CELL_RESET_MEAN,
        nominal_dw_min=0.1,
        reset=_uniform_physical(shape, 0.1),
        upper=_uniform_physical(shape, 0.6),
    )

    assert mapped.level_spacing_unit == pytest.approx(0.2)
    torch.testing.assert_close(
        mapped.selected_headroom_unit, torch.tensor([[0.5, 0.5]])
    )
    assert torch.equal(mapped.selected_level_capacity, torch.tensor([[2, 2]]))
    assert torch.equal(
        quad_stack(mapped.integer_level_number, layout="halves"),
        torch.tensor([[[2, 0, 0, 2], [1, 0, 0, 1]]]),
    )
    torch.testing.assert_close(
        quad_stack(mapped.continuous_offset, layout="halves"),
        torch.tensor([[[0.5, 0.0, 0.0, 0.5], [0.1, 0.0, 0.0, 0.1]]]),
    )
    torch.testing.assert_close(
        quad_stack(mapped.quantized_offset, layout="halves"),
        torch.tensor([[[0.4, 0.0, 0.0, 0.4], [0.2, 0.0, 0.0, 0.2]]]),
    )


def test_mapping_materializes_full_physical_g_and_its_loading() -> None:
    logical = torch.tensor([[1.0]])
    mapped = _layer(
        logical,
        policy=SHARED_QUAD_RESET_MAX,
        scale_fraction=0.5,
        conductance_min=2.0,
        conductance_max=6.0,
        reset=_physical_from_quads(
            torch.tensor([[[0.1, 0.2, 0.3, 0.4]]]), layout="halves"
        ),
    )

    assert torch.equal(
        mapped.continuous_conductance,
        mapped.baseline + mapped.continuous_offset,
    )
    assert torch.equal(
        mapped.quantized_conductance,
        mapped.baseline + mapped.quantized_offset,
    )
    torch.testing.assert_close(
        mapped.continuous_contrast,
        quad_contrast(mapped.continuous_conductance, layout="halves"),
    )
    torch.testing.assert_close(
        mapped.continuous_loading,
        quad_loading(mapped.continuous_conductance, layout="halves"),
    )
    assert float(mapped.continuous_conductance.min()) >= 2.0
    assert float(mapped.continuous_conductance.max()) <= 6.0
    assert float(mapped.continuous_loading.item()) > float(
        2.0 * mapped.continuous_contrast.abs().item()
    )


def test_dense_resistive_kcl_uses_full_g_in_transfer_and_denominator() -> None:
    conductance = torch.tensor(
        [[0.7, 0.2], [0.3, 0.9]], dtype=torch.float32
    )
    pre_voltage = torch.tensor([[0.8, -0.4]], dtype=torch.float32)
    post_voltage = torch.tensor([[0.1, -0.2]], dtype=torch.float32)
    pre = LinearLayer((2,), batch_size=1, device="cpu")
    post = LinearLayer((2,), batch_size=1, device="cpu")
    pre.state = pre_voltage
    post.state = post_voltage
    weight = DenseWeight(
        (2,),
        (2,),
        gain=0.0,
        device="cpu",
        clamp=True,
        clamp_min=0.0,
        clamp_max=2.0,
    )
    weight.state = conductance.clone()
    interaction = DenseResistive(
        pre,
        post,
        weight,
        voltage_amp=1.0,
        current_amp=1.0,
        logical_pre_index=0,
        logical_post_index=1,
    )

    torch.testing.assert_close(
        interaction.b_coef_fn(post)(), -(pre_voltage @ conductance)
    )
    torch.testing.assert_close(
        interaction.a_coef_fn(post)(), 0.5 * conductance.sum(dim=0).unsqueeze(0)
    )
    changed_baseline = conductance + 0.4
    weight.state = changed_baseline
    torch.testing.assert_close(
        interaction.b_coef_fn(post)(), -(pre_voltage @ changed_baseline)
    )
    torch.testing.assert_close(
        interaction.a_coef_fn(post)(),
        0.5 * changed_baseline.sum(dim=0).unsqueeze(0),
    )


def test_checked_application_validates_all_layers_before_any_mutation() -> None:
    _weights, mapping = _two_layer_mapping()
    bindings = tuple(
        SimpleNamespace(state=torch.full_like(layer.baseline, -7.0))
        for layer in mapping.layers
    )

    apply_checked_physical_targets(bindings, mapping, endpoint="continuous")
    for binding, layer in zip(bindings, mapping.layers):
        torch.testing.assert_close(binding.state, layer.continuous_conductance)

    fresh = tuple(
        SimpleNamespace(state=torch.full_like(layer.baseline, -9.0))
        for layer in mapping.layers
    )
    broken_layer = replace(
        mapping.layers[1],
        continuous_conductance=mapping.layers[1].continuous_conductance + 0.01,
    )
    broken = replace(mapping, layers=(mapping.layers[0], broken_layer))
    with pytest.raises(ValueError, match=r"G=B\+d"):
        apply_checked_physical_targets(fresh, broken, endpoint="continuous")
    for binding in fresh:
        assert torch.equal(binding.state, torch.full_like(binding.state, -9.0))


def test_joint_repair_uses_first_candidate_in_private_disjoint_blocks() -> None:
    plan = build_joint_repair_plan(
        (
            torch.tensor([[True, False]]),
            torch.tensor([[False]]),
        ),
        torch.tensor(
            [
                [False, True, True],
                [True, False, True],
            ]
        ),
        maximum_attempts=3,
    )

    assert plan.failure_count == 2
    assert torch.equal(plan.failure_layer_index, torch.tensor([0, 1]))
    assert torch.equal(plan.failure_flat_index, torch.tensor([1, 0]))
    assert torch.equal(plan.selected_attempt, torch.tensor([1, 0]))
    assert torch.equal(plan.selected_donor_flat_index, torch.tensor([1, 3]))
    assert torch.equal(plan.attempts_by_layer[0], torch.tensor([[-1, 1]]))
    assert torch.equal(plan.attempts_by_layer[1], torch.tensor([[0]]))

    base_quads = (
        torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]),
        torch.tensor([[[9.0, 10.0, 11.0, 12.0]]]),
    )
    base_layers = (
        _physical_from_quads(base_quads[0], layout="halves"),
        _physical_from_quads(base_quads[1], layout="paired"),
    )
    donor = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) + 100
    repaired = apply_joint_repair_field(base_layers, donor, plan)
    repaired_quads = (
        quad_stack(repaired[0], layout="halves"),
        quad_stack(repaired[1], layout="paired"),
    )

    torch.testing.assert_close(repaired_quads[0][0, 0], base_quads[0][0, 0])
    torch.testing.assert_close(repaired_quads[0][0, 1], donor[0, 1])
    torch.testing.assert_close(repaired_quads[1][0, 0], donor[1, 0])
    assert torch.equal(base_layers[0], _physical_from_quads(base_quads[0], layout="halves"))

    base_traces = tuple(
        torch.stack((value, value + 0.5), dim=-1) for value in base_layers
    )
    donor_traces = torch.stack((donor, donor + 0.5), dim=-1)
    repaired_traces = apply_joint_repair_field(base_traces, donor_traces, plan)
    torch.testing.assert_close(
        torch.stack(
            (
                quad_stack(repaired_traces[0][..., 0], layout="halves")[0, 1],
                quad_stack(repaired_traces[0][..., 1], layout="halves")[0, 1],
            ),
            dim=-1,
        ),
        donor_traces[0, 1],
    )


def test_joint_repair_fails_closed_on_donor_exhaustion() -> None:
    with pytest.raises(DonorExhaustionError, match=r"failure indices \[0\]"):
        build_joint_repair_plan(
            (torch.tensor([[False]]), torch.tensor([[True]])),
            torch.zeros((1, 4), dtype=torch.bool),
            maximum_attempts=4,
        )


def test_weight_reconstruction_uses_full_contrast_and_exact_error_sum() -> None:
    weights, mapping = _two_layer_mapping(policy=INDEPENDENT_CELL_RESET_MEAN)
    scales = fit_weight_reconstruction_scales(weights, mapping)
    errors = build_weight_error_decomposition(weights, mapping, scales)

    assert len(scales) == len(errors) == 2
    assert all(scale > 0.0 for scale in scales)
    for source, error in zip(weights, errors):
        component_sum = (
            error.baseline_error
            + error.envelope_error
            + error.quantization_error
        )
        torch.testing.assert_close(error.total_error, component_sum)
        torch.testing.assert_close(
            error.continuous_total_error,
            error.baseline_error + error.envelope_error,
        )
        torch.testing.assert_close(
            error.reconstructed_quantized_weight,
            source + error.total_error,
        )
        assert error.report["component_sum_max_abs_residual"] < 1e-6
        assert "raw_contrast_error" in error.report
        assert "normalized_conductance_coordinate_error" in error.report
        assert "normalized_relu_weight_error" in error.report
        assert "teacher_weight_error" in error.report
        assert "continuous_endpoint" in error.report
        assert "standard4delta_endpoint" in error.report


def test_hardware_fingerprint_is_deterministic_and_tensor_sensitive() -> None:
    metadata_a = {"assignment_seed": 87001, "role": "heldout"}
    metadata_b = {"role": "heldout", "assignment_seed": 87001}
    tensors_a = {
        "upper": torch.tensor([1.0, 0.9]),
        "lower": torch.tensor([0.0, 0.1]),
    }
    tensors_b = {"lower": tensors_a["lower"], "upper": tensors_a["upper"]}

    first = hardware_instance_fingerprint(metadata_a, tensors_a)
    second = hardware_instance_fingerprint(metadata_b, tensors_b)
    assert first == second
    assert first != hardware_instance_fingerprint(
        metadata_a, {**tensors_a, "upper": torch.tensor([1.0, 0.8])}
    )


def test_mapping_artifact_is_pickle_free_and_records_every_tensor(
    tmp_path: Path,
) -> None:
    _weights, mapping = _two_layer_mapping()
    artifact_path = tmp_path / "physical_mapping.npz"
    record = save_physical_mapping(
        artifact_path,
        mapping,
        assignment_seed=87001,
        assignment_role="heldout",
        hardware_instance_id="joint-hardware-test",
    )

    expected_tensors = {
        f"layer_{layer.layer_index}_{field.name}"
        for layer in mapping.layers
        for field in fields(layer)
        if isinstance(getattr(layer, field.name), torch.Tensor)
    }
    expected_scalars = {
        "schema",
        "schema_version",
        "policy",
        "assignment_seed",
        "assignment_role",
        "hardware_instance_id",
        "conductance_min",
        "conductance_max",
    }
    with np.load(record["path"], allow_pickle=False) as artifact:
        assert set(artifact.files) == expected_scalars | expected_tensors
        assert artifact["schema"].item() == MAPPING_SCHEMA
        assert artifact["schema_version"].item() == MAPPING_SCHEMA_VERSION
        for layer in mapping.layers:
            np.testing.assert_array_equal(
                artifact[f"layer_{layer.layer_index}_continuous_conductance"],
                artifact[f"layer_{layer.layer_index}_baseline"]
                + artifact[f"layer_{layer.layer_index}_continuous_offset"],
            )

    receipt = json.loads(Path(record["receipt"]).read_text(encoding="utf-8"))
    assert receipt["schema"] == MAPPING_SCHEMA
    assert receipt["schema_version"] == MAPPING_SCHEMA_VERSION
    assert receipt["artifact_sha256"] == record["sha256"]
    assert set(receipt["tensor_hashes"]) == expected_tensors
    assert receipt["mapping"]["decomposition"] == "every physical branch G=B+d"
