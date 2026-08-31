from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pytest

from experiments.mnist_relu_drn.analyze_ibm_om_baseline_spacing_pv_truncated_nominal import (
    RESULT_SCHEMA,
    WinsorizedNominalAnalysisError,
    analyze,
)


ALPHAS = (0.0, 0.25, 0.5)
SPACINGS = (1, 2, 4)
ASSIGNMENTS = (87001, 87002, 87003)
ENDPOINTS = {
    87001: (89101, 89102, 89103, 89104, 89105),
    87002: (89201, 89202, 89203, 89204, 89205),
    87003: (89301, 89302, 89303, 89304, 89305),
}


def _metric(correct: int, *, voltage_scale: float) -> dict:
    return {
        "examples": 100,
        "student_correct": correct,
        "student_accuracy": correct / 100,
        "voltage": [
            {"layer": 0, "rms": 100.0},
            {"layer": 1, "rms": voltage_scale},
            {"layer": 2, "rms": voltage_scale / 10},
        ],
    }


def _ideal_weight_report() -> dict:
    return {
        "ideal_layers": [
            {
                "layer": layer,
                "normalized_relu_weight_error": {
                    "continuous_total": {"rmse": 0.05 + layer, "mae": 0.04 + layer},
                    "ideal_quantized_total": {
                        "rmse": 0.1 + layer,
                        "mae": 0.08 + layer,
                    },
                },
                "ideal_quantized_endpoint": {
                    "relative_l2": 0.2 + layer,
                    "cosine_similarity": 0.9 - layer * 0.1,
                },
            }
            for layer in range(2)
        ]
    }


def _persistent_weight_report(endpoint_seed: int) -> dict:
    return {
        "endpoint_seed": endpoint_seed,
        "layers": [
            {
                "layer": layer,
                "rmse": 0.3 + layer,
                "mae": 0.2 + layer,
                "relative_l2": 0.4 + layer,
                "cosine_similarity": 0.8 - layer * 0.1,
            }
            for layer in range(2)
        ],
    }


def _baseline_projection() -> dict:
    return {
        "policy": "project_requested_RESET_max_once_to_exact_common_support",
        "stage": "target_construction_before_quantization_and_pv",
        "persistent_endpoint_projection": False,
        "destination_pair_count": 50,
        "projected_pair_count": 2,
        "projected_pair_fraction": 0.04,
        "absolute_projection_sum_raw_x": 0.03,
        "absolute_projection_mean_all_pairs_raw_x": 0.0006,
        "absolute_projection_mean_projected_pairs_raw_x": 0.015,
        "absolute_projection_maximum_raw_x": 0.02,
        "layers": [
            {
                "layer": 0,
                "destination_pair_count": 40,
                "projected_pair_count": 2,
                "downward_projection_count": 2,
                "upward_projection_count": 0,
            },
            {
                "layer": 1,
                "destination_pair_count": 10,
                "projected_pair_count": 0,
                "downward_projection_count": 0,
                "upward_projection_count": 0,
            },
        ],
    }


def _summary(alpha: float, spacing: int, assignment: int) -> dict:
    ideal_correct = 95 - spacing - int(alpha * 4)
    persistent_correct = ideal_correct - 10
    endpoints = []
    repeats = []
    for endpoint_seed in ENDPOINTS[assignment]:
        metric = _metric(persistent_correct, voltage_scale=0.5)
        repeats.append(metric)
        endpoints.append(
            {
                "endpoint_seed": endpoint_seed,
                "cells": 100,
                "counts": {
                    "exact_target_in_support": 100,
                    "verify_window_intersects_support": 100,
                    "apparent_accepted": 99,
                    "persistent_inside_acceptance_window": 60,
                    "requested_code_correct": 70,
                    "nearest_code_adjacent_to_requested": 20,
                    "nearest_code_farther_than_adjacent": 10,
                    "budget_exhausted": 1,
                    "nonfinite": 0,
                    "saturated_lower": 2,
                    "saturated_upper": 3,
                },
                "pulse_totals": {
                    "set": 500,
                    "reset": 200,
                    "total": 700,
                    "verify": 800,
                    "reversals": 20,
                },
                "persistent_network_metrics": metric,
                "persistent_affine_handoff": {
                    "projected": 0,
                    "below_native_support": 0,
                    "above_native_support": 0,
                },
                "apparent_endpoint": {"applied_to_drn": False},
                "persistent_target_residual_unit": {"rmse": 0.04, "mae": 0.03},
                "full_conductance_residuals_by_layer": [
                    {
                        "layer": layer,
                        "persistent_loading": {
                            "mean": 1.5 + layer,
                            "rms": 1.6 + layer,
                        },
                        "persistent_rms_contrast_over_mean_loading": 0.2 + layer,
                    }
                    for layer in range(2)
                ],
            }
        )
    cells = 100
    winsorization = {
        "policy": "nominal_bound_winsorization",
        "identity_resampled": False,
        "cells": cells,
        "empty_intersection_count": 0,
        "lower_bound_clipped_count": 50,
        "lower_bound_clipped_fraction": 0.5,
        "upper_bound_clipped_count": 49,
        "upper_bound_clipped_fraction": 0.49,
        "both_bounds_clipped_count": 24,
        "both_bounds_clipped_fraction": 0.24,
        "any_bound_clipped_count": 75,
        "any_bound_clipped_fraction": 0.75,
    }
    invariant_layers = [
        {
            "layer": layer,
            "baseline_loading": {"mean": 1.0 + layer},
            "continuous_loading": {"mean": 1.1 + layer},
            "standard4delta_loading": {"mean": 1.2 + layer},
        }
        for layer in range(2)
    ]
    physical_layers = [
        {"layer": layer, "ideal_contrast": {"rms": 0.1 + layer}}
        for layer in range(2)
    ]
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "contract": {"assignments": {"endpoint_seeds": list(ENDPOINTS[assignment])}},
        "design": {
            "baseline_position_fraction": alpha,
            "spacing_delta_x_multiplier": spacing,
            "heldout_shared_destination_baseline_projection": (
                _baseline_projection()
            ),
        },
        "heldout": {
            "assignment_seed": assignment,
            "hardware": {"hardware_instance_id": f"hardware-{assignment}"},
            "continuous": _metric(96, voltage_scale=0.8),
            "ideal_quantized": _metric(ideal_correct, voltage_scale=0.7),
            "pv_persistent_repeats": repeats,
            "pv_persistent_mean_accuracy": persistent_correct / 100,
            "pv_endpoint_artifacts": endpoints,
            "target_affine_handoff": {"projected": 0},
            "native_coordinate": {"winsorization": winsorization},
            "invariants": {"layers": invariant_layers},
            "mapping": {
                "physical": {"layers": physical_layers},
                "shared_destination_baseline_projection": _baseline_projection(),
            },
            "weight_errors": _ideal_weight_report(),
            "pv_persistent_weight_errors": [
                _persistent_weight_report(seed) for seed in ENDPOINTS[assignment]
            ],
        },
        "validity": {
            "coverage_valid": True,
            "gates": {"cuda": True, "repeat_count": True},
            "apparent_endpoint_applied_to_drn": False,
            "inference_read_noise": 0.0,
        },
    }


def _write_matrix(root: Path) -> None:
    for alpha, spacing, assignment in product(ALPHAS, SPACINGS, ASSIGNMENTS):
        token = str(alpha).replace(".", "")
        run_id = f"run-{token}-{spacing}-{assignment}"
        run = root / "runs" / f"arm-{token}-{spacing}-{assignment}" / run_id
        artifacts = run / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "scientific_summary.json").write_text(
            json.dumps(_summary(alpha, spacing, assignment)), encoding="utf-8"
        )
        (run / "status.json").write_text(
            json.dumps({"status": "complete", "run_id": run_id}), encoding="utf-8"
        )
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "config": {"sha256": f"config-{run_id}"},
                    "source": {
                        "commit": "abc123",
                        "dirty": True,
                        "dirty_hash": "dirty123",
                    },
                }
            ),
            encoding="utf-8",
        )


def test_analyze_complete_matrix_writes_requested_reports(tmp_path: Path) -> None:
    root = tmp_path / "result"
    _write_matrix(root)

    result = analyze(root, comparison_roots={})

    assert result["status"] == "complete_exploratory_noncanonical_analysis"
    assert result["coverage"] == {
        "complete_configurations": 27,
        "expected_configurations": 27,
        "designs": 9,
        "heldout_assignments_per_design": 3,
        "persistent_repeats_per_assignment": 5,
        "persistent_deployments": 135,
    }
    assert (
        result["intervention"]["scientific_name"]
        == "nominal_bound_winsorization"
    )
    assert result["intervention"]["default_ibm_model"] is False
    assert len(result["assignment_rows"]) == 27
    assert len(result["design_rows"]) == 9
    assert result["winsorization_by_assignment"][0]["any_bound_clipped_fraction"] == 0.75
    assert result["requested_baseline_projection_unique_assignments"][
        "projected_pair_count"
    ] == 6
    first = result["design_rows"][0]
    assert first["program_verify"]["fractions"]["requested_code_correct"] == 0.7
    assert first["program_verify"]["pulses_per_cell"] == 7.0
    assert first["ideal_weight_errors"][0]["ideal_quantized_rmse"]["mean"] == pytest.approx(0.1)
    assert first["persistent_weight_errors"][0]["relative_l2"]["mean"] == pytest.approx(0.4)

    analysis = root / "analysis"
    assert (analysis / "exploratory_summary.json").is_file()
    assert len((analysis / "accuracy_by_assignment.csv").read_text().splitlines()) == 28
    markdown = (analysis / "post_run_analysis.md").read_text(encoding="utf-8")
    assert "not the default IBM" in markdown
    assert "DRN-written versus ReLU weight errors" in markdown


def test_analyze_rejects_missing_fifth_endpoint_repeat(tmp_path: Path) -> None:
    root = tmp_path / "result"
    _write_matrix(root)
    summary_path = next(root.rglob("artifacts/scientific_summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["heldout"]["pv_endpoint_artifacts"].pop()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(WinsorizedNominalAnalysisError, match="five persistent P&V"):
        analyze(root, comparison_roots={})
