from dataclasses import fields

import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    LayerPhysicalMapping,
    SHARED_DESTINATION_COLUMNS_RESET_MAX,
    build_physical_mapping,
    quad_contrast,
    quad_stack,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv import (
    build_baseline_spacing_mapping,
)


def _inputs(reference_shift: float = 0.0):
    weights = (
        torch.tensor([[0.13, -0.70], [1.00, -0.03]], dtype=torch.float32),
        torch.tensor([[-0.91, 0.27], [0.04, 0.63]], dtype=torch.float32),
    )
    lower = (
        torch.tensor(
            [
                [0.02, 0.04, 0.01, 0.03],
                [0.05, 0.01, 0.02, 0.04],
                [0.03, 0.02, 0.04, 0.01],
                [0.01, 0.05, 0.03, 0.02],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [0.02, 0.04, 0.01, 0.03],
                [0.05, 0.01, 0.02, 0.04],
                [0.03, 0.02, 0.04, 0.01],
                [0.01, 0.05, 0.03, 0.02],
            ],
            dtype=torch.float32,
        ),
    )
    upper = (
        torch.tensor(
            [
                [0.91, 0.83, 0.88, 0.86],
                [0.79, 0.94, 0.82, 0.90],
                [0.87, 0.80, 0.92, 0.84],
                [0.89, 0.85, 0.81, 0.93],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [0.91, 0.83, 0.88, 0.86],
                [0.79, 0.94, 0.82, 0.90],
                [0.87, 0.80, 0.92, 0.84],
                [0.89, 0.85, 0.81, 0.93],
            ],
            dtype=torch.float32,
        ),
    )
    reset = tuple(
        low
        + torch.tensor(
            [
                [0.08, 0.05, 0.09, 0.04],
                [0.06, 0.10, 0.05, 0.07],
                [0.07, 0.06, 0.08, 0.05],
                [0.09, 0.04, 0.06, 0.08],
            ],
            dtype=torch.float32,
        )
        for low in lower
    )
    references = tuple(
        torch.linspace(-0.2, 0.2, 16, dtype=torch.float32).reshape(4, 4)
        + reference_shift
        for _ in range(2)
    )
    return {
        "logical_weights": weights,
        "scale_fractions": (0.8, 0.6),
        "nominal_dw_min": 0.1,
        "conductance_min": 0.1,
        "conductance_max": 1.1,
        "cell_lower_units": lower,
        "cell_upper_units": upper,
        "reset_baseline_units": reset,
        "intrinsic_references_native": references,
    }


def test_alpha_zero_four_delta_is_tensor_exact_parent_parity() -> None:
    inputs = _inputs()
    parent = build_physical_mapping(
        **inputs,
        policy=SHARED_DESTINATION_COLUMNS_RESET_MAX,
    )
    candidate = build_baseline_spacing_mapping(
        **inputs,
        baseline_position_fraction=0.0,
        spacing_delta_multiples=4,
    ).physical

    for parent_layer, candidate_layer in zip(parent.layers, candidate.layers):
        for field in fields(LayerPhysicalMapping):
            expected = getattr(parent_layer, field.name)
            actual = getattr(candidate_layer, field.name)
            if isinstance(expected, torch.Tensor):
                assert torch.equal(actual, expected), field.name
            else:
                assert actual == expected, field.name


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5])
@pytest.mark.parametrize("multiple", [1, 2, 4])
def test_destination_baseline_capacity_and_raw_target_units(
    alpha: float,
    multiple: int,
) -> None:
    inputs = _inputs()
    mapping = build_baseline_spacing_mapping(
        **inputs,
        baseline_position_fraction=alpha,
        spacing_delta_multiples=multiple,
    )
    spacing = multiple * inputs["nominal_dw_min"] / 2.0
    span = inputs["conductance_max"] - inputs["conductance_min"]

    for layer, capacity, target_unit in zip(
        mapping.physical.layers,
        mapping.directional_capacity,
        mapping.ideal_target_units,
    ):
        reset_quad = quad_stack(
            layer.reset_baseline_unit.to(torch.float64), layout=layer.layout
        )
        upper_quad = quad_stack(
            layer.cell_upper_unit.to(torch.float64), layout=layer.layout
        )
        expected_lower = torch.stack(
            (
                torch.maximum(reset_quad[..., 0], reset_quad[..., 2]),
                torch.maximum(reset_quad[..., 1], reset_quad[..., 3]),
            ),
            dim=-1,
        )
        expected_upper = torch.stack(
            (
                torch.minimum(upper_quad[..., 0], upper_quad[..., 2]),
                torch.minimum(upper_quad[..., 1], upper_quad[..., 3]),
            ),
            dim=-1,
        )
        expected_baseline = expected_lower + alpha * (
            expected_upper - expected_lower
        )
        assert torch.equal(
            capacity.group_baseline_unit,
            expected_baseline.to(torch.float32),
        )
        assert torch.equal(
            capacity.downward_level_capacity,
            torch.floor(
                (expected_baseline - expected_lower) / spacing + 1e-12
            ).to(torch.int64),
        )
        assert torch.equal(
            capacity.upward_level_capacity,
            torch.floor(
                (expected_upper - expected_baseline) / spacing + 1e-12
            ).to(torch.int64),
        )
        assert torch.count_nonzero(layer.baseline_contrast) == 0
        assert bool(torch.all(layer.integer_level_number >= 0))
        expected_target_unit = (
            layer.baseline_unit.to(torch.float64)
            + layer.integer_level_number.to(torch.float64) * spacing
        ).to(torch.float32)
        assert torch.equal(target_unit, expected_target_unit)
        assert bool(torch.all(target_unit >= layer.cell_lower_unit - 1e-6))
        assert bool(torch.all(target_unit <= layer.cell_upper_unit + 1e-6))

        # The ideal circuit tensor and raw-x P&V target encode the same full
        # G.  The old mapper performs baseline and offset float32 rounding
        # separately, so require its declared physical tolerance, not bitwise
        # equality after an affine round trip.
        roundtrip = inputs["conductance_min"] + span * target_unit
        assert torch.allclose(
            roundtrip,
            layer.quantized_conductance,
            rtol=0.0,
            atol=span * 1e-6,
        )


def test_no_reference_mapping_and_strict_spacing_selector() -> None:
    first = build_baseline_spacing_mapping(
        **_inputs(reference_shift=0.0),
        baseline_position_fraction=0.25,
        spacing_delta_multiples=2,
    )
    second = build_baseline_spacing_mapping(
        **_inputs(reference_shift=0.4),
        baseline_position_fraction=0.25,
        spacing_delta_multiples=2,
    )
    for left, right in zip(first.physical.layers, second.physical.layers):
        assert torch.equal(left.baseline, right.baseline)
        assert torch.equal(left.continuous_conductance, right.continuous_conductance)
        assert torch.equal(left.quantized_conductance, right.quantized_conductance)

    with pytest.raises(ValueError, match="Expected spacing"):
        build_baseline_spacing_mapping(
            **_inputs(),
            baseline_position_fraction=0.25,
            spacing_delta_multiples=1.5,  # type: ignore[arg-type]
        )
