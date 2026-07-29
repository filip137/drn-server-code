from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import experiments.run_conv12_bounded_rho as bounded_rho_runner
from experiments.reselect_bounded_rho_below_accuracy import (
    _select_best_safe as reselect_best_safe,
)
from experiments.reselect_bounded_rho_below_accuracy import (
    _selected_edge as reselect_selected_edge,
)
from experiments.run_conv12_bounded_rho import (
    _has_clean_canary,
    _rho_axes,
    _rho_index,
    build_source_config,
    collect,
    compare_fixed_tk_gradients,
    load_study,
    select_candidates,
    surface_specs,
)

CONV3_STUDY = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "conv"
    / "perfectdiode_conv3_bounded_rho_baseline_ours_20260729_v1.json"
)


def test_focused_study_has_exactly_sixteen_nonlegacy_surfaces() -> None:
    _path, study = load_study()
    surfaces = surface_specs(study)

    assert len(surfaces) == 16
    assert {surface["architecture"] for surface in surfaces} == {"conv1", "conv2"}
    assert {surface["scheme"] for surface in surfaces} == {"baseline", "ours"}
    assert {surface["optimizer"] for surface in surfaces} == {"SGD", "Adam"}
    assert all("legacy" not in surface["surface_id"] for surface in surfaces)


def test_conv3_study_has_eight_fixed_grid_surfaces() -> None:
    _path, study = load_study(CONV3_STUDY)
    surfaces = surface_specs(study)
    conv, dense = _rho_axes(study)

    assert len(surfaces) == 8
    assert {surface["architecture"] for surface in surfaces} == {"conv3"}
    assert study["model"]["architectures"]["conv3"]["T"] == 8
    assert study["model"]["architectures"]["conv3"]["K"] == 8
    assert conv == pytest.approx([0.003, 0.009, 0.027, 0.081, 0.243])
    assert dense == pytest.approx([0.01, 0.03, 0.09, 0.27, 0.81])


def test_source_config_freezes_bounded_perfect_diode_contract(tmp_path) -> None:
    _path, study = load_study()
    checkpoint = tmp_path / "initial.pt"
    config = build_source_config(
        study,
        initializer="bounded_kaiming_uniform",
        architecture="conv2",
        scheme="ours",
        optimizer="Adam",
        init_checkpoint_path=checkpoint,
    )

    assert config["model_base"]["weight_min"] == 1e-5
    assert config["model_base"]["weight_max"] == 1e-4
    assert config["model_base"]["weight_init_mode"] == "bounded_kaiming_uniform"
    assert config["model_base"]["num_iterations_inference"] == 6
    assert config["model_base"]["num_iterations_training"] == 6
    assert config["model_base"]["voltage_amp"] == 4.0
    assert config["model_base"]["current_amp"] == 1.0
    assert config["optimizer"]["name"] == "Adam"
    assert config["datasets"]["mnist"]["factory"].endswith(
        "MnistTrainValidationDataset"
    )
    assert config["init_checkpoint_path"] == str(checkpoint.resolve())
    for name in (
        "quadratic_diode_param",
        "exponential_diode_param",
        "hard_sigmoid_param",
    ):
        assert config["model_base"][name]


def test_conv3_source_config_freezes_t8_k8_and_three_convolutions(tmp_path) -> None:
    _path, study = load_study(CONV3_STUDY)
    config = build_source_config(
        study,
        initializer="bounded_uniform",
        architecture="conv3",
        scheme="baseline",
        optimizer="SGD",
        init_checkpoint_path=tmp_path / "initial.pt",
    )

    assert config["model_base"]["num_iterations_inference"] == 8
    assert config["model_base"]["num_iterations_training"] == 8
    assert config["model_base"]["input_gain"] == 360.0
    assert config["model_overrides"]["mnist_bp_conv_amp"]["layer_shapes"] == [
        [2, 28, 28],
        [64, 14, 14],
        [128, 7, 7],
        [256, 7, 7],
        [20],
    ]
    assert len(config["model_overrides"]["mnist_bp_conv_amp"]["conv_pipeline"]) == 3
    assert len(config["lr"]) == 7


def test_rho_axis_resolves_all_center_attempts_and_expansion_edges() -> None:
    _path, study = load_study()
    conv, dense = _rho_axes(study)
    center = study["rho_search"]["center"]

    for attempt in range(6):
        index = _rho_index(
            conv,
            dense,
            center["rho_conv"] / (3.0**attempt),
            center["rho_dense"] / (3.0**attempt),
        )
        assert 0 <= index < 100
    assert _rho_index(conv, dense, center["rho_conv"] * 9, center["rho_dense"]) >= 0
    assert _rho_index(conv, dense, center["rho_conv"], center["rho_dense"] * 9) >= 0


def test_completed_center_cell_reuses_its_clean_canary(tmp_path) -> None:
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    (cell_dir / "canary.json").write_text(
        '{"status":"clean","completed_steps":640,"requested_steps":640}\n',
        encoding="utf-8",
    )

    assert _has_clean_canary(cell_dir, {"status": "complete"}) is True
    assert _has_clean_canary(cell_dir, {"status": "canary_clean"}) is True


def test_conv3_fixed_grid_runs_all_nine_cells_without_center_search(
    tmp_path, monkeypatch
) -> None:
    _path, study = load_study(CONV3_STUDY)
    surface = surface_specs(study)[0]
    calls = []

    monkeypatch.setattr(bounded_rho_runner, "run_rho_search", lambda _args: {})

    def fake_run_cell(
        _study,
        _surface,
        _source_config,
        rho_root,
        _rho_conv_axis,
        _rho_dense_axis,
        rho_conv,
        rho_dense,
        *,
        device,
        canary_only,
        smoke=False,
    ):
        del device, smoke
        calls.append((float(rho_conv), float(rho_dense), canary_only))
        cell_dir = rho_root / "cells" / f"cell-{len(calls)}"
        return cell_dir, {
            "index": len(calls) - 1,
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "status": "complete",
            "selection_eligible": True,
        }

    def fake_candidate(cell_dir, cell):
        distance = abs(float(cell["rho_conv"]) - 0.027) + abs(
            float(cell["rho_dense"]) - 0.09
        )
        return {
            "cell_id": cell_dir.name,
            "index": cell["index"],
            "rho_conv": float(cell["rho_conv"]),
            "rho_dense": float(cell["rho_dense"]),
            "status": "complete",
            "selection_eligible": True,
            "final_validation_loss": 1.0 if distance < 1e-12 else 2.0 + distance,
            "final_validation_accuracy": 0.95,
            "median_projection_efficiency": 1.0,
            "path": str(cell_dir),
        }

    monkeypatch.setattr(bounded_rho_runner, "_run_rho_cell", fake_run_cell)
    monkeypatch.setattr(bounded_rho_runner, "_candidate_record", fake_candidate)

    result = bounded_rho_runner.run_rho_surface(
        study,
        tmp_path,
        surface,
        tmp_path / "source.json",
        device="cuda",
    )

    assert len(calls) == 9
    assert all(canary_only is False for _, _, canary_only in calls)
    assert {rho_conv for rho_conv, _, _ in calls} == {0.009, 0.027, 0.081}
    assert {rho_dense for _, rho_dense, _ in calls} == {0.03, 0.09, 0.27}
    assert result["status"] == "complete"
    assert result["selected"]["rho_conv"] == 0.027
    assert result["selected"]["rho_dense"] == 0.09


def test_suspiciously_low_fixed_grid_widens_toward_better_edges(
    tmp_path, monkeypatch
) -> None:
    _path, study = load_study(CONV3_STUDY)
    surface = surface_specs(study)[0]
    calls = []

    monkeypatch.setattr(bounded_rho_runner, "run_rho_search", lambda _args: {})

    def fake_run_cell(
        _study,
        _surface,
        _source_config,
        rho_root,
        _rho_conv_axis,
        _rho_dense_axis,
        rho_conv,
        rho_dense,
        *,
        device,
        canary_only,
        smoke=False,
    ):
        del device, smoke
        calls.append((float(rho_conv), float(rho_dense), canary_only))
        cell_dir = rho_root / "cells" / f"cell-{len(calls)}"
        return cell_dir, {
            "index": len(calls) - 1,
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "status": "complete",
            "selection_eligible": False,
        }

    def fake_candidate(cell_dir, cell):
        rho_conv = float(cell["rho_conv"])
        rho_dense = float(cell["rho_dense"])
        distance = abs(rho_conv - 0.027) + abs(rho_dense - 0.09)
        return {
            "cell_id": cell_dir.name,
            "index": cell["index"],
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "status": "complete",
            "selection_eligible": False,
            "final_validation_loss": 1.0 if distance < 1e-12 else 2.0 + distance,
            "final_validation_accuracy": 0.55 + 0.1 * rho_conv + 0.1 * rho_dense,
            "median_projection_efficiency": 1.0,
            "path": str(cell_dir),
        }

    monkeypatch.setattr(bounded_rho_runner, "_run_rho_cell", fake_run_cell)
    monkeypatch.setattr(bounded_rho_runner, "_candidate_record", fake_candidate)

    result = bounded_rho_runner.run_rho_surface(
        study,
        tmp_path,
        surface,
        tmp_path / "source.json",
        device="cuda",
    )

    assert len(calls) == 16
    assert max(rho_conv for rho_conv, _, _ in calls) == pytest.approx(0.243)
    assert max(rho_dense for _, rho_dense, _ in calls) == pytest.approx(0.81)
    assert result["expansion"]["trigger"] == "suspicious_low_accuracy"
    assert result["suspicious_accuracy_status"]["triggered"] is True
    assert result["suspicious_accuracy_status"]["remains_below_floor"] is True
    assert result["rho_range_status"]["classification"] == "unbounded"


def test_fixed_tk_comparison_uses_inclusive_layerwise_gates() -> None:
    contract = {
        "reference_T": 64,
        "reference_K": 64,
        "gradient_zero_epsilon": 1e-12,
        "relative_gradient_l2_norm_delta_maximum": 0.0,
        "absolute_zero_fraction_delta_maximum": 0.0,
        "gradient_vector_cosine_minimum": 1.0,
    }
    reference = {"ConvWeight_0": [torch.tensor([1.0, 0.0])]}
    operational = {"ConvWeight_0": [torch.tensor([1.0, 0.0])]}

    result = compare_fixed_tk_gradients(
        operational,
        reference,
        contract,
        operational_t=4,
        operational_k=4,
    )

    assert result["security_passed"] is True
    assert (
        result["parameter_diagnostics"][0]["relative_gradient_l2_norm_delta"]
        == 0.0
    )


def test_selector_uses_loss_plateau_then_accuracy_and_projection() -> None:
    candidates = [
        {
            "cell_id": "a",
            "selection_eligible": True,
            "final_validation_loss": 1.0,
            "final_validation_accuracy": 0.91,
            "median_projection_efficiency": 0.9,
            "rho_conv": 0.003,
            "rho_dense": 0.01,
        },
        {
            "cell_id": "b",
            "selection_eligible": True,
            "final_validation_loss": 1.02,
            "final_validation_accuracy": 0.92,
            "median_projection_efficiency": 0.1,
            "rho_conv": 0.001,
            "rho_dense": 0.01,
        },
    ]

    selection = select_candidates(candidates, 0.02)

    assert {item["cell_id"] for item in selection["plateau"]} == {"a", "b"}
    assert selection["selected"]["cell_id"] == "b"


def test_selector_falls_back_to_best_safe_candidate_below_accuracy() -> None:
    candidates = [
        {
            "cell_id": "low-loss",
            "status": "complete",
            "selection_eligible": False,
            "final_validation_loss": 0.2,
            "final_validation_accuracy": 0.88,
            "median_projection_efficiency": 0.7,
            "rho_conv": 0.001,
            "rho_dense": 0.01,
        },
        {
            "cell_id": "higher-loss",
            "status": "complete",
            "selection_eligible": False,
            "final_validation_loss": 0.3,
            "final_validation_accuracy": 0.89,
            "median_projection_efficiency": 0.9,
            "rho_conv": 0.003,
            "rho_dense": 0.01,
        },
    ]

    selection = select_candidates(
        candidates,
        0.02,
        select_best_safe_below_accuracy=True,
    )

    assert selection["accuracy_gate_met"] is False
    assert selection["selection_basis"] == "best_safe_below_accuracy"
    assert selection["selected"]["cell_id"] == "low-loss"


def test_compatibility_reselector_detects_below_accuracy_boundary() -> None:
    candidates = [
        {
            "cell_id": "best",
            "status": "complete",
            "final_validation_loss": 0.2,
            "final_validation_accuracy": 0.88,
            "median_projection_efficiency": 0.8,
            "rho_conv": 0.001,
            "rho_dense": 0.01,
        },
        {
            "cell_id": "other",
            "status": "complete",
            "final_validation_loss": 0.3,
            "final_validation_accuracy": 0.89,
            "median_projection_efficiency": 0.9,
            "rho_conv": 0.003,
            "rho_dense": 0.01,
        },
    ]

    selection = reselect_best_safe(candidates, 0.02)

    assert selection["selected"]["cell_id"] == "best"
    assert (
        reselect_selected_edge(
            selection,
            "rho_conv",
            [0.001, 0.003, 0.009],
        )
        == "lower"
    )


def test_reselector_detects_factor_constructed_boundary_with_float_tolerance() -> None:
    selection = {
        "selected": {
            "rho_conv": 0.027,
            "rho_dense": 0.03,
        }
    }

    assert (
        reselect_selected_edge(
            selection,
            "rho_conv",
            [0.001, 0.003, 0.009000000000000001, 0.027000000000000003],
        )
        == "upper"
    )


def test_collect_reports_range_and_below_accuracy_fields(tmp_path) -> None:
    _path, study = load_study()
    surface = surface_specs(study)[0]
    selection_path = (
        tmp_path / "surfaces" / surface["surface_id"] / "selection.json"
    )
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(
        """{
          "status": "complete_below_accuracy_range_bounded",
          "selected": {"rho_conv": 0.001, "rho_dense": 0.01},
          "selection": {"accuracy_gate_met": false},
          "rho_range_status": {
            "classification": "bounded",
            "bracketed": false,
            "rho_conv_edge": "lower",
            "rho_dense_edge": null
          },
          "suspicious_accuracy_status": {
            "floor": 0.8,
            "triggered": true,
            "remains_below_floor": true
          }
        }
        """,
        encoding="utf-8",
    )

    summary = collect(study, tmp_path)
    record = summary["surfaces"][0]

    assert summary["execution_status"] == "in_progress"
    assert record["accuracy_gate_met"] is False
    assert record["rho_range_status"]["classification"] == "bounded"
    assert record["suspicious_accuracy_status"]["remains_below_floor"] is True


def test_collect_treats_below_accuracy_fallbacks_as_complete(tmp_path) -> None:
    _path, study = load_study()
    statuses = (
        "complete",
        "complete_below_accuracy_range_bounded",
        "complete_below_accuracy_bracketed",
    )
    for surface in surface_specs(study):
        selection_path = (
            tmp_path / "surfaces" / surface["surface_id"] / "selection.json"
        )
        selection_path.parent.mkdir(parents=True)
        selection_path.write_text(
            json.dumps(
                {
                    "status": statuses[surface["index"] % len(statuses)],
                    "selected": {
                        "rho_conv": 0.001,
                        "rho_dense": 0.01,
                    },
                }
            ),
            encoding="utf-8",
        )

    summary = collect(study, tmp_path)

    assert summary["status"] == "complete"
    assert summary["execution_status"] == "terminal"
