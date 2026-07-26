from __future__ import annotations

import copy

import pytest

from experiments.mnist_conv.lr_protocol import (
    V6_RHO_CONV_GRID,
    V6_RHO_DENSE_GRID,
    select_v6_conv2_baseline,
    two_rho_candidate_grid,
    two_rho_learning_rates,
)


def _grid_results() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"conv-{rho_conv:g}--dense-{rho_dense:g}",
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "admissible": True,
            "final_validation_loss": 2.0,
            "final_validation_accuracy": 0.90,
            "median_projection_efficiency": 0.8,
            "inadmissible_reason": None,
        }
        for rho_conv, rho_dense in two_rho_candidate_grid()
    ]


def _candidate(
    candidates: list[dict[str, object]], rho_conv: float, rho_dense: float
) -> dict[str, object]:
    return next(
        candidate
        for candidate in candidates
        if candidate["rho_conv"] == rho_conv and candidate["rho_dense"] == rho_dense
    )


def test_two_rho_vector_uses_each_median_unit_and_ties_each_bias() -> None:
    units = {
        "ConvWeight_0": 0.1,
        "ConvWeight_1": 0.2,
        "DenseWeight_0": 0.5,
    }
    rates = two_rho_learning_rates(
        units,
        rho_conv=0.003,
        rho_dense=0.03,
        parameter_names=(
            "DenseWeight_0",
            "Bias_1",
            "ConvWeight_0",
            "Bias_0",
            "ConvWeight_1",
        ),
    )

    assert rates == pytest.approx(
        {
            "ConvWeight_0": 0.03,
            "Bias_0": 0.03,
            "ConvWeight_1": 0.015,
            "Bias_1": 0.015,
            "DenseWeight_0": 0.06,
        }
    )
    assert rates["ConvWeight_0"] * units["ConvWeight_0"] == pytest.approx(0.003)
    assert rates["ConvWeight_1"] * units["ConvWeight_1"] == pytest.approx(0.003)
    assert rates["DenseWeight_0"] * units["DenseWeight_0"] == pytest.approx(0.03)

    assert two_rho_learning_rates(
        units,
        rho_conv=0.003,
        rho_dense=0.03,
        bias_weight_lr_groups={
            "ConvWeight_0": ["ConvWeight_0", "Bias_0"],
            "ConvWeight_1": ["ConvWeight_1", "Bias_1"],
            "DenseWeight_0": ["DenseWeight_0"],
        },
    ) == pytest.approx(rates)

    with pytest.raises(ValueError, match="exactly the bounded weights"):
        two_rho_learning_rates(
            {**units, "Bias_0": 1.0},
            0.003,
            0.03,
            ("ConvWeight_0", "Bias_0", "ConvWeight_1", "Bias_1", "DenseWeight_0"),
        )


def test_frozen_grid_is_four_by_four_in_conv_outer_dense_inner_order() -> None:
    grid = two_rho_candidate_grid()

    assert V6_RHO_CONV_GRID == (5e-4, 1e-3, 3e-3, 1e-2)
    assert V6_RHO_DENSE_GRID == (3e-3, 1e-2, 3e-2, 1e-1)
    assert len(grid) == 16
    assert grid[:4] == tuple((5e-4, value) for value in V6_RHO_DENSE_GRID)
    assert grid[-4:] == tuple((1e-2, value) for value in V6_RHO_DENSE_GRID)


def test_baseline_selection_uses_accuracy_then_efficiency_then_lower_targets() -> None:
    candidates = _grid_results()
    first = _candidate(candidates, 0.001, 0.01)
    first.update(
        final_validation_loss=0.5,
        final_validation_accuracy=0.91,
        median_projection_efficiency=0.9,
    )
    second = _candidate(candidates, 0.003, 0.01)
    second.update(
        final_validation_loss=0.505,
        final_validation_accuracy=0.92,
        median_projection_efficiency=0.7,
    )

    selection = select_v6_conv2_baseline(candidates)

    assert selection["status"] == "selected"
    assert selection["minimum_final_validation_loss"] == pytest.approx(0.5)
    assert len(selection["plateau"]) == 2
    assert selection["selected_rho_conv"] == 0.003
    assert selection["selected_rho_dense"] == 0.01

    second["final_validation_accuracy"] = 0.91
    second["median_projection_efficiency"] = 0.9
    lower_target_selection = select_v6_conv2_baseline(candidates)
    assert lower_target_selection["selected_rho_conv"] == 0.001
    assert lower_target_selection["selected_rho_dense"] == 0.01


def test_exactly_ninety_percent_passes_and_below_ninety_fails() -> None:
    candidates = _grid_results()
    below = _candidate(candidates, 0.001, 0.01)
    below.update(final_validation_loss=0.1, final_validation_accuracy=0.8999)
    boundary = _candidate(candidates, 0.003, 0.01)
    boundary.update(final_validation_loss=0.2, final_validation_accuracy=0.90)

    selection = select_v6_conv2_baseline(candidates)

    assert selection["status"] == "selected"
    assert selection["selected_rho_conv"] == 0.003
    assert below["final_validation_accuracy"] < 0.90
    evaluated_below = next(
        candidate
        for candidate in selection["evaluated_candidates"]
        if candidate["rho_conv"] == 0.001 and candidate["rho_dense"] == 0.01
    )
    assert evaluated_below["passes"] is False
    assert evaluated_below["failure_reason"] == "final_validation_accuracy_below_0.9"


def test_outer_edge_plateau_is_reported_unbracketed_without_freezing() -> None:
    candidates = _grid_results()
    upper_conv = _candidate(candidates, 0.01, 0.03)
    upper_conv.update(final_validation_loss=0.5, final_validation_accuracy=0.94)
    upper_dense = _candidate(candidates, 0.003, 0.1)
    upper_dense.update(final_validation_loss=0.505, final_validation_accuracy=0.95)

    selection = select_v6_conv2_baseline(candidates)

    assert selection["status"] == "unbracketed"
    assert selection["reason"] == "passing_plateau_confined_to_outer_boundary"
    assert selection["selected"] is None
    assert selection["selected_rho_conv"] is None
    assert selection["selected_rho_dense"] is None
    assert selection["diagnostic_best"]["rho_dense"] == 0.1


def test_selection_rejects_incomplete_grid_and_fails_when_every_run_fails() -> None:
    candidates = _grid_results()
    with pytest.raises(ValueError, match="exactly one result for every configured"):
        select_v6_conv2_baseline(candidates[:-1])

    failed = copy.deepcopy(candidates)
    for candidate in failed:
        candidate.update(
            admissible=False,
            final_validation_loss=None,
            final_validation_accuracy=None,
            median_projection_efficiency=None,
            inadmissible_reason="projection_efficiency",
        )
    selection = select_v6_conv2_baseline(failed)
    assert selection["status"] == "failed"
    assert selection["reason"] == "no_passing_baseline_candidate"
    assert selection["selected"] is None
