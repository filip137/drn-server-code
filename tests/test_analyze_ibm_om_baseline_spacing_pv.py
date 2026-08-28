from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.mnist_relu_drn.analyze_ibm_om_baseline_spacing_pv import (
    DESIGNS,
    BaselineSpacingPvAnalysisError,
    _aggregate,
    _csv,
    _markdown,
    _validate_predictions,
    _zero_only_initialization_group_count,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_config import (
    ENDPOINT_SEEDS_BY_ASSIGNMENT,
    HELDOUT_ASSIGNMENT_SEEDS,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_runtime import (
    _save_predictions,
)


def _records(
    *, narrow_lead: bool = False, exact_threshold_lead: bool = False
) -> list[dict[str, object]]:
    def metrics(rms: float) -> dict[str, object]:
        return {
            "teacher_agreement": 0.9,
            "kl_teacher_student": 0.1,
            "voltage": [{"rms": rms + layer} for layer in range(3)],
        }

    def endpoint(endpoint_seed: int) -> dict[str, object]:
        return {
            "endpoint_seed": endpoint_seed,
            "cells": 100,
            "persistent_projection": {"projected": 0},
            "apparent_projection": {"projected": 0},
            "persistent_network_metrics": metrics(2.0),
            "apparent_network_metrics": metrics(2.1),
            "counts": {
                "requested_code_correct": 90,
                "apparent_accepted": 95,
                "budget_exhausted": 5,
            },
            "pulse_totals": {"set": 100, "reset": 50, "total": 150},
            "pulse_distributions": {
                "set_count": {"mean": 1.0, "rms": 1.4},
                "reset_count": {"mean": 0.5, "rms": 0.8},
                "total_pulses": {"mean": 1.5, "rms": 2.0},
                "verify_count": {"mean": 2.5, "rms": 3.0},
                "reversals": {"mean": 0.1, "rms": 0.2},
            },
            "full_conductance_residuals_by_layer": [
                {
                    "persistent_target_residual_full_conductance": {
                        "rmse": 0.01 + layer
                    },
                    "apparent_target_residual_full_conductance": {
                        "rmse": 0.02 + layer
                    },
                    "persistent_loading": {"mean": 2.0 + layer},
                    "apparent_loading": {"mean": 2.1 + layer},
                    "persistent_rms_contrast_over_mean_loading": 0.1 + layer,
                    "apparent_rms_contrast_over_mean_loading": 0.2 + layer,
                }
                for layer in range(2)
            ],
            "requested_to_nearest_code_confusion": [
                {"requested_index": 0, "nearest_index": 0, "cells": 90},
                {"requested_index": 1, "nearest_index": 0, "cells": 10},
            ],
            "persistent_target_residual_unit": {"rmse": 0.01},
            "apparent_target_residual_unit": {"rmse": 0.02},
            "persistent_residual_by_requested_level": [],
            "apparent_residual_by_requested_level": [],
        }

    result = []
    for design_index, (alpha, spacing, arm_id) in enumerate(DESIGNS):
        for assignment_index, seed in enumerate(HELDOUT_ASSIGNMENT_SEEDS):
            ideal = 9500 - 10 * design_index
            if design_index == 0:
                persistent = [9400] * 5
            elif design_index == 1:
                persistent = [
                    9395
                    if narrow_lead
                    else 9300
                    if exact_threshold_lead
                    else 9200
                ] * 5
            else:
                persistent = [9000 - 10 * design_index] * 5
            result.append(
                {
                    "run_id": f"run-{design_index}-{assignment_index}",
                    "arm_id": arm_id,
                    "alpha": alpha,
                    "spacing_delta_x_multiplier": spacing,
                    "heldout_assignment_seed": seed,
                    "selected_scale_fractions": [1.0, 1.0],
                    "fixed_logit_gain": 14.0,
                    "development_hardware_instance_id": "development",
                    "heldout_hardware_instance_id": f"heldout-{seed}",
                    "ideal_correct": ideal,
                    "persistent_correct": persistent,
                    "continuous_metrics": metrics(1.0),
                    "ideal_metrics": metrics(1.5),
                    "endpoint_diagnostics": [
                        endpoint(endpoint_seed)
                        for endpoint_seed in ENDPOINT_SEEDS_BY_ASSIGNMENT[seed]
                    ],
                    "mapping_diagnostics": [],
                    "paired_prediction_diagnostics": [],
                    "ideal_weight_errors": [
                        {
                            "ideal_quantized_endpoint": {
                                "relative_l2": 0.1 + layer
                            },
                            "teacher_weight_error": {
                                "ideal_quantized_total": {"rmse": 0.2 + layer}
                            },
                        }
                        for layer in range(2)
                    ],
                    "pv_persistent_weight_errors": [
                        [
                            {
                                "relative_l2": 0.3 + layer,
                                "rmse": 0.4 + layer,
                            }
                            for layer in range(2)
                        ]
                        for _ in ENDPOINT_SEEDS_BY_ASSIGNMENT[seed]
                    ],
                }
            )
    return result


def test_aggregate_applies_assignment_dominance_and_mean_lead() -> None:
    aggregate, decision = _aggregate(_records())

    assert decision["best_ideal_design"] == "alpha-000-spacing-1delta"
    assert decision["predeclared_unique_practical_winner"] == (
        "alpha-000-spacing-1delta"
    )
    assert aggregate["matrices"]["ideal_accuracy"][0][0] == 0.95


def test_aggregate_is_inconclusive_below_one_point_mean_lead() -> None:
    _aggregate_result, decision = _aggregate(_records(narrow_lead=True))

    assert decision["assignment_level_strict_dominance_candidates"] == [
        "alpha-000-spacing-1delta"
    ]
    assert decision["persistent_mean_lead_over_runner_up"] == pytest.approx(
        0.0005
    )
    assert decision["predeclared_unique_practical_winner"] is None


def test_aggregate_accepts_exact_1500_correct_boundary_without_float_rounding() -> None:
    _aggregate_result, decision = _aggregate(
        _records(exact_threshold_lead=True)
    )

    assert decision["persistent_correct_count_lead_over_runner_up"] == 1500
    assert decision["persistent_mean_lead_over_runner_up"] == pytest.approx(0.01)
    assert decision["predeclared_unique_practical_winner"] == (
        "alpha-000-spacing-1delta"
    )


def test_zero_only_initialization_ignores_latent_downward_headroom() -> None:
    upward = torch.tensor([0, 1, 0, 3], dtype=torch.int64)

    assert _zero_only_initialization_group_count(upward) == 2


def test_human_readable_outputs_surface_weight_and_voltage_diagnostics() -> None:
    records = _records()
    aggregate, decision = _aggregate(records)

    markdown = _markdown({"title": "Synthetic"}, aggregate, decision)
    assert "DRN weight error versus frozen ReLU weights" in markdown
    assert "ideal rel-L2 L0/L1" in markdown
    assert "persistent voltage RMS S0/S1/S2" in markdown
    assert "persistent full-G RMSE L0/L1" in markdown
    assert "mean SET/RESET/total pulses per cell" in markdown
    assert "Programmed endpoint loading" in markdown

    csv_text = _csv(records)
    header = csv_text.splitlines()[0]
    assert "ideal_relative_l2_layer0" in header
    assert "pv_persistent_rmse_mean_layer1" in header
    first = aggregate["designs"][DESIGNS[0][2]]["program_verify_diagnostics"]
    assert (
        first["by_assignment"][0]["repeats"][0][
            "apparent_residual_by_requested_level"
        ]
        == []
    )
    assert first["requested_to_nearest_code_confusion_total"] == [
        {"requested_index": 0, "nearest_index": 0, "cells": 1350},
        {"requested_index": 1, "nearest_index": 0, "cells": 150},
    ]


def test_prediction_artifact_is_recomputed_and_tamper_rejected(
    tmp_path: Path,
) -> None:
    labels = torch.arange(32, dtype=torch.int64) % 10
    teacher = labels.clone()
    continuous = labels.clone()
    ideal = labels.clone()
    endpoint_seeds = ENDPOINT_SEEDS_BY_ASSIGNMENT[87001]
    persistent = [labels.clone() for _ in endpoint_seeds]
    apparent = [labels.clone() for _ in endpoint_seeds]
    path = tmp_path / "predictions.npz"
    _save_predictions(
        path,
        labels=labels,
        teacher=teacher,
        continuous=continuous,
        ideal=ideal,
        persistent=persistent,
        apparent=apparent,
        endpoint_seeds=endpoint_seeds,
    )

    report = _validate_predictions(
        path,
        path.with_suffix(".receipt.json"),
        endpoint_seeds=endpoint_seeds,
        expected_examples=32,
    )
    assert report["ideal_correct"] == 32

    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(BaselineSpacingPvAnalysisError, match="artifact hash"):
        _validate_predictions(
            path,
            path.with_suffix(".receipt.json"),
            endpoint_seeds=endpoint_seeds,
            expected_examples=32,
        )
