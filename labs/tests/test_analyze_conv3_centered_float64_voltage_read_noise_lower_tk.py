from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "experiments/analyze_conv3_centered_float64_voltage_read_noise_lower_tk.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_lower_tk_voltage_noise_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lower_tk = _load_module()


def test_with_tk_overrides_parent_schema_and_iteration_fields():
    rows = [{"schema": "parent", "T": 64, "K": 64, "value": 3}]
    observed = lower_tk._with_tk(rows, 8)
    assert observed == [
        {"schema": lower_tk.SCHEMA, "T": 8, "K": 8, "value": 3}
    ]


def test_drift_metrics_reports_direction_norm_and_zero_fraction():
    reference = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    candidate = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    row = lower_tk._drift_metrics(candidate, reference)
    assert row["gate_passed"] is True
    assert row["cosine"] == 1.0
    assert row["candidate_zero_fraction"] == 2.0 / 3.0
    assert row["reference_zero_fraction"] == 2.0 / 3.0
    assert row["absolute_zero_fraction_delta"] == 0.0


def test_same_tk_gate_replaces_parent_k64_label_without_mutation():
    parent_gate = {"task_reference": "bptt_same_post_T_state_K64", "x": 1}
    observed = lower_tk._same_tk_gradient_gate(parent_gate, 12)
    assert observed["task_reference"] == "bptt_same_post_T_state_K12"
    assert parent_gate["task_reference"] == "bptt_same_post_T_state_K64"


def test_aggregation_retains_undefined_cosine_as_gate_failure():
    row = {
        "scheme": "legacy",
        "checkpoint_role": "best_validation",
        "clean_beta_selection_status": "fixed_from_tk64",
        "sigma_voltage": 0.0,
        "parameter_name": "ConvWeight_0",
        "acquisition_cosine": None,
        "acquisition_symmetric_norm_delta": 2.0,
        "task_cosine": None,
        "task_symmetric_norm_delta": 2.0,
        "acquisition_gate_passed": False,
        "task_gate_passed": False,
        "usable_gate_passed": False,
    }
    layers, configurations = lower_tk._aggregate_trials_allowing_undefined_cosine(
        [row], cosine_minimum=0.99, norm_delta_maximum=0.1
    )
    assert layers[0]["task_cosine_p05"] is None
    assert layers[0]["undefined_task_cosine_trial_count"] == 1
    assert layers[0]["task_gate_passed"] is False
    assert configurations[0]["minimum_layer_task_cosine_p05"] is None
    assert configurations[0]["all_layers_task_gate_passed"] is False


def test_usable_threshold_is_null_when_sigma_zero_task_gate_fails():
    configuration_rows = [
        {
            "scheme": "legacy",
            "checkpoint_role": "best_validation",
            "sigma_voltage": 0.0,
            "clean_beta_selection_status": "fixed_from_tk64",
            "all_layers_acquisition_gate_passed": True,
            "all_layers_task_gate_passed": False,
            "all_layers_usable_gate_passed": False,
        },
        {
            "scheme": "legacy",
            "checkpoint_role": "best_validation",
            "sigma_voltage": 1.0e-10,
            "clean_beta_selection_status": "fixed_from_tk64",
            "all_layers_acquisition_gate_passed": True,
            "all_layers_task_gate_passed": True,
            "all_layers_usable_gate_passed": True,
        },
    ]
    clean_state_rows = [
        {
            "scheme": "legacy",
            "checkpoint_role": "best_validation",
            "delta_rms": 2.0e-9,
        }
    ]
    row = lower_tk._thresholds_allowing_clean_failure(
        configuration_rows, clean_state_rows
    )[0]
    assert row["clean_eqprop_vs_same_tk_bptt_gate_passed"] is False
    assert row["usable_maximum_sustained_sigma"] is None
    assert row["usable_first_failing_sigma"] == 0.0
    assert row["usable_threshold_status"] == "clean_task_gate_failed_at_sigma_zero"


def test_residual_failure_nulls_only_equilibrium_valid_threshold():
    threshold = {
        "scheme": "baseline",
        "checkpoint_role": "best_validation",
        "acquisition_only_maximum_sustained_sigma": 1.0e-6,
        "usable_maximum_sustained_sigma": 3.0e-7,
        "clean_eqprop_vs_same_tk_bptt_gate_passed": True,
    }
    guard = {
        "scheme": "baseline",
        "checkpoint_role": "best_validation",
        "all_residual_gates_passed": False,
    }
    row = lower_tk._annotate_equilibrium_validity([threshold], [guard])[0]
    assert row["usable_maximum_sustained_sigma"] == 3.0e-7
    assert row["equilibrium_valid_usable_maximum_sustained_sigma"] is None
    assert row["under_relaxed_diagnostic"] is True
    assert row["equilibrium_valid_threshold_status"] == (
        "under_relaxed_residual_gate_failed"
    )
