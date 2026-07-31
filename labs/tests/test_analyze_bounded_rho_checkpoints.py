from __future__ import annotations

import pytest
import torch

from experiments.analyze_bounded_rho_checkpoints import (
    LEGACY_SELECTED_CELLS,
    _distribution_stats,
    _gradient_comparison,
    _gradient_geometry,
    _transition_rows,
    Surface,
)


def _surface(tmp_path) -> Surface:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    result = tmp_path / "result.json"
    result.write_text("{}\n", encoding="utf-8")
    return Surface(
        surface_id="bounded_uniform__conv1__baseline__sgd",
        evidence_panel="active_matched_baseline_ours",
        architecture="conv1",
        initializer="bounded_uniform",
        scheme="baseline",
        optimizer="SGD",
        final_validation_accuracy=0.8,
        final_validation_loss=0.2,
        rho_conv=0.001,
        rho_dense=0.01,
        rho_range_classification="unbounded",
        rho_conv_edge=None,
        rho_dense_edge=None,
        cell_dir=tmp_path,
        initialization_checkpoint=checkpoint,
        best_checkpoint=checkpoint,
        final_checkpoint=checkpoint,
        best_epoch=3,
        final_epoch=3,
        config_path=None,
        run_spec_path=None,
        result_path=result,
        safety_path=None,
        step_log_path=None,
        selection_source_path=result,
        learning_rates_by_parameter={"ConvWeight_0": 0.1},
        source_policy_projection_is_report_only=True,
    )


def test_distribution_and_transition_metrics_resolve_exact_float32_bounds(tmp_path):
    lower = torch.tensor(1e-5, dtype=torch.float32)
    upper = torch.tensor(1e-4, dtype=torch.float32)
    midpoint = (lower + upper) / 2
    initial = torch.stack((lower, midpoint, upper, midpoint))
    final = torch.stack((lower, upper, upper, lower))

    statistics = _distribution_stats(final, initial)
    transitions = _transition_rows(
        _surface(tmp_path), "final", "ConvWeight_0", initial, final
    )
    lookup = {
        (row["source_region"], row["target_region"]): row["transition_count"]
        for row in transitions
    }

    assert statistics["exact_lower_fraction"] == pytest.approx(0.5)
    assert statistics["exact_upper_fraction"] == pytest.approx(0.5)
    assert statistics["combined_exact_bound_fraction"] == pytest.approx(1.0)
    assert lookup[("lower", "lower")] == 1
    assert lookup[("interior", "upper")] == 1
    assert lookup[("upper", "upper")] == 1
    assert lookup[("interior", "lower")] == 1


def test_gradient_geometry_separates_outward_pressure_from_projection():
    lower = torch.tensor(1e-5, dtype=torch.float32)
    upper = torch.tensor(1e-4, dtype=torch.float32)
    midpoint = (lower + upper) / 2
    state = torch.stack((lower, upper, midpoint, midpoint))
    # Descent is -gradient: the first two components point outward.
    gradient = torch.tensor((1.0, -2.0, 1.0, -1.0), dtype=torch.float32)

    result = _gradient_geometry(
        state=state,
        gradient=gradient,
        name="ConvWeight_0",
        learning_rate=1e-4,
        optimizer="SGD",
        adam_epsilon=1e-8,
    )

    assert result["outward_component_fraction_all"] == pytest.approx(0.5)
    assert result["outward_component_fraction_among_bound"] == pytest.approx(1.0)
    assert 0.0 < result["tangent_gradient_efficiency_l2"] < 1.0
    assert 0.0 < result["projection_efficiency_l2"] < 1.0
    assert result["proposal_kind"] == "exact_sgd_momentum0_weight_decay0"


def test_gradient_comparison_uses_matched_vectors():
    initial = torch.tensor((1.0, 0.0), dtype=torch.float64)
    current = torch.tensor((0.0, 2.0), dtype=torch.float64)
    result = _gradient_comparison(current, initial)

    assert result["gradient_l2_ratio_to_initial"] == pytest.approx(2.0)
    assert result["gradient_cosine_vs_initial"] == pytest.approx(0.0)
    assert result["gradient_relative_l2_delta_vs_initial"] == pytest.approx(
        5.0**0.5
    )


def test_historical_legacy_scope_is_explicitly_incomplete():
    assert len(LEGACY_SELECTED_CELLS) == 5
    assert sum(item["architecture"] == "conv1" for item in LEGACY_SELECTED_CELLS) == 4
    assert sum(item["architecture"] == "conv2" for item in LEGACY_SELECTED_CELLS) == 1
    assert {
        (item["architecture"], item["initializer"], item["optimizer"])
        for item in LEGACY_SELECTED_CELLS
        if item["architecture"] == "conv2"
    } == {("conv2", "bounded_uniform", "SGD")}
