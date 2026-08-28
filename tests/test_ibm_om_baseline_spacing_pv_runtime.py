import json
from types import SimpleNamespace

import pytest
import torch

from experiments.mnist_relu_drn.analyze_ibm_om_baseline_spacing_pv import (
    BaselineSpacingPvAnalysisError,
    _validate_ideal_weight_errors,
    _validate_summary_parity,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv import (
    build_baseline_spacing_mapping,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_runtime import (
    _endpoint_conductance_diagnostics,
    _endpoint_residual_reports,
    _historical_parity_gates,
    _historical_parity_summary,
    _maximum_supported_uniform_index,
    _pulse_distribution_reports,
    _requested_to_nearest_code_confusion,
    _save_ideal_weight_errors,
    _spacing_neutral_ideal_report,
)


def test_uniform_capacity_uses_raw_active_support_tolerance() -> None:
    # A float32 shared baseline can round a few 1e-8 above the native upper
    # endpoint even though it remains inside the P&V support tolerance.  Zero
    # must remain a valid code rather than becoming a spurious capacity -1.
    upper = torch.tensor([0.5405102670192719], dtype=torch.float64)
    baseline = torch.tensor([0.5405102968215942], dtype=torch.float32)

    maximum = _maximum_supported_uniform_index(
        upper,
        baseline,
        spacing_unit=0.04745,
    )

    assert torch.equal(maximum, torch.zeros(1, dtype=torch.int64))


def _parity():
    return SimpleNamespace(
        development_hardware_instance_id="development",
        heldout_hardware_instance_ids=("heldout",),
        selected_scale_fractions=(1.0, 1.0),
        fixed_logit_gain=14.0,
        ideal_correct=(9000,),
        ideal_prediction_sha256=("full-test-hash",),
    )


def test_smoke_parity_keeps_identity_and_calibration_but_skips_full_test_output() -> None:
    required, gates = _historical_parity_gates(
        require_full_test_output=False,
        parity=_parity(),
        parity_index=0,
        development_hardware_instance_id="development",
        heldout_hardware_instance_id="heldout",
        selected_pair=(1.0, 1.0),
        selected_gain=14.0,
        ideal_metrics={"student_correct": 29, "prediction_sha256": "smoke-hash"},
    )

    assert required is False
    assert all(gates.values())


def test_production_parity_requires_exact_full_test_output() -> None:
    required, gates = _historical_parity_gates(
        require_full_test_output=True,
        parity=_parity(),
        parity_index=0,
        development_hardware_instance_id="development",
        heldout_hardware_instance_id="heldout",
        selected_pair=(1.0, 1.0),
        selected_gain=14.0,
        ideal_metrics={"student_correct": 29, "prediction_sha256": "smoke-hash"},
    )

    assert required is True
    assert gates["ideal_correct"] is False
    assert gates["ideal_prediction_sha256"] is False


def test_smoke_parity_does_not_bypass_calibration_mismatch() -> None:
    _required, gates = _historical_parity_gates(
        require_full_test_output=False,
        parity=_parity(),
        parity_index=0,
        development_hardware_instance_id="development",
        heldout_hardware_instance_id="heldout",
        selected_pair=(0.5, 1.0),
        selected_gain=14.0,
        ideal_metrics={"student_correct": 29, "prediction_sha256": "smoke-hash"},
    )

    assert gates["selected_scales"] is False


def test_runtime_parity_block_matches_analyzer_contract() -> None:
    gates = {"hardware_identity": True, "ideal_correct": True}
    written = _historical_parity_summary(
        design_arm=True,
        full_test_output_required=True,
        gates=gates,
    )

    _validate_summary_parity(
        written,
        design_arm=True,
        full_test_output_required=True,
        gates=gates,
    )

    with pytest.raises(BaselineSpacingPvAnalysisError, match="summary parity"):
        _validate_summary_parity(
            {"required": True, "gates": gates},
            design_arm=True,
            full_test_output_required=True,
            gates=gates,
        )


def test_ideal_report_uses_spacing_neutral_names_recursively() -> None:
    renamed = _spacing_neutral_ideal_report(
        {
            "standard4delta_endpoint": {"relative_l2": 0.1},
            "teacher_weight_error": {
                "standard4delta_total": {"rmse": 0.2}
            },
            "items": [{"standard4delta_total": 3}],
        }
    )

    assert renamed["ideal_quantized_endpoint"] == {"relative_l2": 0.1}
    assert renamed["teacher_weight_error"]["ideal_quantized_total"] == {
        "rmse": 0.2
    }
    assert renamed["items"] == [{"ideal_quantized_total": 3}]
    serialized = json.dumps(renamed, sort_keys=True)
    assert "standard4delta_endpoint" not in serialized
    assert "standard4delta_total" not in serialized


def test_ideal_report_rename_rejects_key_collision() -> None:
    with pytest.raises(RuntimeError, match="key collision"):
        _spacing_neutral_ideal_report(
            {
                "standard4delta_endpoint": {},
                "ideal_quantized_endpoint": {},
            }
        )


def test_ideal_weight_writer_is_consumed_by_analyzer_with_neutral_names(
    tmp_path,
) -> None:
    weights = (
        torch.tensor([[0.5, -0.25], [0.1, -0.8]], dtype=torch.float32),
        torch.tensor([[-0.3, 0.7], [0.2, -0.4]], dtype=torch.float32),
    )
    lower = tuple(torch.full((4, 4), 0.02) for _ in range(2))
    upper = tuple(torch.full((4, 4), 0.95) for _ in range(2))
    reset = tuple(torch.full((4, 4), 0.10) for _ in range(2))
    reference = tuple(torch.zeros((4, 4)) for _ in range(2))
    scales = (0.8, 0.8)
    mapping = build_baseline_spacing_mapping(
        logical_weights=weights,
        scale_fractions=scales,
        nominal_dw_min=0.1,
        conductance_min=0.1,
        conductance_max=1.1,
        cell_lower_units=lower,
        cell_upper_units=upper,
        reset_baseline_units=reset,
        intrinsic_references_native=reference,
        baseline_position_fraction=0.25,
        spacing_delta_multiples=2,
    )
    path = tmp_path / "ideal_weight_errors.npz"

    _save_ideal_weight_errors(
        path,
        logical_weights=weights,
        mapping=mapping,
        analysis_scales=scales,
    )
    analyzer_mapping = {
        "spacing_physical": mapping.physical.layers[0].level_spacing_physical,
        "span": mapping.physical.conductance_max - mapping.physical.conductance_min,
        "layers": [
            {
                "source_weight": layer.source_weight.numpy(),
                "baseline_contrast": layer.baseline_contrast.numpy(),
                "continuous_contrast": layer.continuous_contrast.numpy(),
                "quantized_contrast": layer.quantized_contrast.numpy(),
            }
            for layer in mapping.physical.layers
        ],
    }

    reports = _validate_ideal_weight_errors(
        path,
        path.with_suffix(".receipt.json"),
        mapping=analyzer_mapping,
        analysis_scales=scales,
    )

    assert len(reports) == 2
    assert all("ideal_quantized_endpoint" in report for report in reports)
    assert all(
        "ideal_quantized_total" in report["teacher_weight_error"]
        for report in reports
    )


def test_endpoint_residual_reports_keep_apparent_levels_separate() -> None:
    requested = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    persistent = torch.tensor([0.0, 0.2, -0.1, 0.1], dtype=torch.float32)
    apparent = torch.tensor([0.3, -0.1, 0.4, -0.2], dtype=torch.float32)

    reports = _endpoint_residual_reports(requested, persistent, apparent)

    assert set(reports) == {
        "persistent_target_residual_unit",
        "apparent_target_residual_unit",
        "persistent_residual_by_requested_level",
        "apparent_residual_by_requested_level",
    }
    assert [
        value["requested_level"]
        for value in reports["apparent_residual_by_requested_level"]
    ] == [0, 1]
    assert (
        reports["apparent_residual_by_requested_level"]
        != reports["persistent_residual_by_requested_level"]
    )


def test_pulse_distributions_include_set_and_reset_counts() -> None:
    programming = SimpleNamespace(
        set_count=torch.tensor([0, 2, 3]),
        reset_count=torch.tensor([1, 0, 2]),
        total_pulses=torch.tensor([1, 2, 5]),
        verify_count=torch.tensor([2, 3, 6]),
        reversals=torch.tensor([0, 0, 1]),
    )

    reports = _pulse_distribution_reports(programming)

    assert tuple(reports) == (
        "set_count",
        "reset_count",
        "total_pulses",
        "verify_count",
        "reversals",
    )
    assert reports["set_count"]["mean"] == pytest.approx(5 / 3)
    assert reports["reset_count"]["mean"] == pytest.approx(1.0)


def test_code_confusion_and_full_g_loading_diagnostics_are_complete() -> None:
    confusion = _requested_to_nearest_code_confusion(
        torch.tensor([0, 1, 1, 2, 2, 2]),
        torch.tensor([0, 0, 1, 1, 2, 2]),
    )
    assert confusion == [
        {"requested_index": 0, "nearest_index": 0, "cells": 1},
        {"requested_index": 1, "nearest_index": 0, "cells": 1},
        {"requested_index": 1, "nearest_index": 1, "cells": 1},
        {"requested_index": 2, "nearest_index": 1, "cells": 1},
        {"requested_index": 2, "nearest_index": 2, "cells": 2},
    ]

    conductance = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    diagnostics = _endpoint_conductance_diagnostics(
        conductance,
        layout="halves",
    )
    assert diagnostics["loading"]["mean"] > 0.0
    assert diagnostics["rms_contrast_over_mean_loading"] >= 0.0
