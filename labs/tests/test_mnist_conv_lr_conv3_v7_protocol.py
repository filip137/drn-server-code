from __future__ import annotations

import copy

import pytest

from experiments.mnist_conv.lr_protocol import (
    V6_RHO_CONV_GRID,
    V6_RHO_DENSE_GRID,
    layerwise_parameter_groups,
    select_two_rho_candidates,
    select_v6_conv2_baseline,
    two_rho_candidate_grid,
    two_rho_learning_rates,
    two_rho_upper_boundary_axes,
)


CORE_CONV_GRID = (5e-4, 3e-3, 1e-2)
CORE_DENSE_GRID = (3e-3, 1e-2, 3e-2)


def _grid_results(
    rho_conv_values: tuple[float, ...] = CORE_CONV_GRID,
    rho_dense_values: tuple[float, ...] = CORE_DENSE_GRID,
) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"conv-{rho_conv:g}--dense-{rho_dense:g}",
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "admissible": True,
            "final_validation_loss": 3.0,
            "final_validation_accuracy": 0.95,
            "median_projection_efficiency": 0.8,
            "inadmissible_reason": None,
        }
        for rho_conv, rho_dense in two_rho_candidate_grid(
            rho_conv_values, rho_dense_values
        )
    ]


def _candidate(
    candidates: list[dict[str, object]], rho_conv: float, rho_dense: float
) -> dict[str, object]:
    return next(
        candidate
        for candidate in candidates
        if candidate["rho_conv"] == rho_conv and candidate["rho_dense"] == rho_dense
    )


def test_conv3_groups_and_two_rho_rates_cover_all_seven_parameters() -> None:
    parameter_names = (
        "Bias_2",
        "DenseWeight_0",
        "ConvWeight_0",
        "Bias_0",
        "ConvWeight_2",
        "Bias_1",
        "ConvWeight_1",
    )
    assert layerwise_parameter_groups(parameter_names) == {
        "ConvWeight_0": ("ConvWeight_0", "Bias_0"),
        "ConvWeight_1": ("ConvWeight_1", "Bias_1"),
        "ConvWeight_2": ("ConvWeight_2", "Bias_2"),
        "DenseWeight_0": ("DenseWeight_0",),
    }

    units = {
        "ConvWeight_0": 0.1,
        "ConvWeight_1": 0.2,
        "ConvWeight_2": 0.4,
        "DenseWeight_0": 0.5,
    }
    rates = two_rho_learning_rates(
        units,
        rho_conv=3e-3,
        rho_dense=3e-2,
        parameter_names=parameter_names,
    )

    assert rates == pytest.approx(
        {
            "ConvWeight_0": 3e-2,
            "Bias_0": 3e-2,
            "ConvWeight_1": 1.5e-2,
            "Bias_1": 1.5e-2,
            "ConvWeight_2": 7.5e-3,
            "Bias_2": 7.5e-3,
            "DenseWeight_0": 6e-2,
        }
    )
    for weight_name, unit in units.items():
        expected_target = (
            3e-3 if weight_name.startswith("ConvWeight_") else 3e-2
        )
        assert rates[weight_name] * unit == pytest.approx(expected_target)


def test_generic_selector_applies_inclusive_ninety_percent_floor() -> None:
    candidates = _grid_results()
    below = _candidate(candidates, 5e-4, 3e-3)
    below.update(final_validation_loss=0.1, final_validation_accuracy=0.8999)
    exact = _candidate(candidates, 3e-3, 1e-2)
    exact.update(final_validation_loss=0.2, final_validation_accuracy=0.90)

    selection = select_two_rho_candidates(
        candidates,
        rho_conv_values=CORE_CONV_GRID,
        rho_dense_values=CORE_DENSE_GRID,
        minimum_accuracy=0.90,
    )

    assert selection["status"] == "selected"
    assert selection["selected_rho_conv"] == pytest.approx(3e-3)
    assert selection["selected_rho_dense"] == pytest.approx(1e-2)
    evaluated_below = next(
        candidate
        for candidate in selection["evaluated_candidates"]
        if candidate["rho_conv"] == 5e-4 and candidate["rho_dense"] == 3e-3
    )
    assert evaluated_below["passes"] is False
    assert evaluated_below["failure_reason"] == (
        "final_validation_accuracy_below_0.9"
    )
    assert selection["selected"]["final_validation_accuracy"] == pytest.approx(0.90)


def test_generic_selector_can_disable_the_accuracy_floor() -> None:
    candidates = _grid_results()
    low_accuracy = _candidate(candidates, 3e-3, 1e-2)
    low_accuracy.update(final_validation_loss=0.1, final_validation_accuracy=0.05)

    selection = select_two_rho_candidates(
        candidates,
        rho_conv_values=CORE_CONV_GRID,
        rho_dense_values=CORE_DENSE_GRID,
        minimum_accuracy=None,
    )

    assert selection["status"] == "selected"
    assert selection["selected"]["candidate_id"] == low_accuracy["candidate_id"]
    assert selection["selected"]["passes"] is True
    assert selection["selected"]["failure_reason"] is None


def test_upper_boundary_axes_distinguish_single_and_split_edges() -> None:
    kwargs = {
        "rho_conv_values": CORE_CONV_GRID,
        "rho_dense_values": CORE_DENSE_GRID,
    }
    assert two_rho_upper_boundary_axes(
        [
            {"rho_conv": 1e-2, "rho_dense": 3e-3},
            {"rho_conv": 1e-2, "rho_dense": 3e-2},
        ],
        **kwargs,
    ) == ("rho_conv",)
    assert two_rho_upper_boundary_axes(
        [
            {"rho_conv": 5e-4, "rho_dense": 3e-2},
            {"rho_conv": 3e-3, "rho_dense": 3e-2},
        ],
        **kwargs,
    ) == ("rho_dense",)
    assert two_rho_upper_boundary_axes(
        [
            {"rho_conv": 1e-2, "rho_dense": 1e-2},
            {"rho_conv": 3e-3, "rho_dense": 3e-2},
        ],
        **kwargs,
    ) == ("rho_conv", "rho_dense")
    assert two_rho_upper_boundary_axes(
        [{"rho_conv": 1e-2, "rho_dense": 3e-2}],
        **kwargs,
    ) == ("rho_conv", "rho_dense")
    assert two_rho_upper_boundary_axes(
        [
            {"rho_conv": 1e-2, "rho_dense": 1e-2},
            {"rho_conv": 3e-3, "rho_dense": 1e-2},
        ],
        **kwargs,
    ) == ()


def test_v6_wrapper_preserves_selected_payload_and_failure_reason() -> None:
    candidates = _grid_results(V6_RHO_CONV_GRID, V6_RHO_DENSE_GRID)
    best = _candidate(candidates, 3e-3, 1e-2)
    best.update(final_validation_loss=0.2, final_validation_accuracy=0.90)

    generic = select_two_rho_candidates(
        candidates,
        rho_conv_values=V6_RHO_CONV_GRID,
        rho_dense_values=V6_RHO_DENSE_GRID,
        minimum_accuracy=0.90,
    )
    assert select_v6_conv2_baseline(candidates) == generic

    failed = copy.deepcopy(candidates)
    for candidate in failed:
        candidate.update(
            admissible=False,
            final_validation_loss=None,
            final_validation_accuracy=None,
            median_projection_efficiency=None,
            inadmissible_reason="projection_efficiency",
        )
    failure = select_v6_conv2_baseline(failed)
    assert failure["status"] == "failed"
    assert failure["reason"] == "no_passing_baseline_candidate"
    with pytest.raises(ValueError, match="minimum_accuracy.*finite number"):
        select_v6_conv2_baseline(candidates, minimum_accuracy=None)  # type: ignore[arg-type]
