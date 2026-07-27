from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.mnist_conv.perfectdiode_tk_spec import (
    CONV3_CONV_WEIGHTS,
    K_GRID,
    T_CORE_GRID,
    T_EXTENSION_GRID,
    PerfectDiodeTKStudySpec,
    PerfectDiodeTKValidationError,
    select_k_measurements,
    select_t_measurements,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv3_tk_ordinary_mnist_v1.json"
)


def _residual_stats(p90: float) -> dict[str, float]:
    return {
        "mean": p90 / 4,
        "median": p90 / 2,
        "p90": p90,
        "p99": p90,
        "max": p90,
    }


def _t_record(
    iteration: int,
    *,
    p90: float = 5.0e-3,
    cohort_sha: str = "a" * 64,
) -> dict:
    layers = []
    for role in ("hidden_0", "hidden_1", "hidden_2", "output"):
        item = {
            "role": role,
            "name": role,
            "selection_residual_mode": (
                "raw" if role == "output" else "projected_kkt"
            ),
            "selection_residual_p90": p90,
            "selection_residual": _residual_stats(p90),
            "raw_residual": _residual_stats(p90 * 2),
        }
        if role != "output":
            item["clamped_occupancy"] = {
                "fraction": 0.5,
                "excitation_fraction": 0.5,
                "inhibition_fraction": 0.5,
            }
        layers.append(item)
    return {
        "iteration_count": iteration,
        "num_examples": 1024,
        "batch_size": 64,
        "official_test_read": False,
        "cohort_source_indices_sha256": cohort_sha,
        "initialization_checkpoint_sha256": "b" * 64,
        "initialization_tensor_sha256": "c" * 64,
        "fixed_step_minimization": True,
        "batch_state_policy": "reset_each_batch",
        "layers": layers,
    }


def _batch(
    index: int,
    *,
    gradient_l2: float = 1.0,
    reference_l2: float = 1.0,
    cosine: float | None = 1.0,
    zero: float = 0.0,
    reference_zero: float = 0.0,
    reference_rms: float = 1.0,
) -> dict:
    norm_delta = abs(gradient_l2 - reference_l2) / max(
        gradient_l2, reference_l2, 1.0e-30
    )
    return {
        "batch_index": index,
        "gradient_l2": gradient_l2,
        "reference_gradient_l2": reference_l2,
        "relative_gradient_l2_norm_delta": norm_delta,
        "gradient_vector_relative_error": abs(gradient_l2 - reference_l2)
        / max(reference_l2, 1.0e-30),
        "gradient_vector_cosine": cosine,
        "gradient_zero_fraction": zero,
        "reference_gradient_zero_fraction": reference_zero,
        "absolute_zero_fraction_delta": abs(zero - reference_zero),
        "reference_gradient_rms": reference_rms,
    }


def _k_record(
    candidate_k: int,
    *,
    reference_k: int = 64,
    gradient_l2: float = 1.0,
    reference_l2: float = 1.0,
    cosine: float | None = 1.0,
    cohort_sha: str = "d" * 64,
) -> dict:
    batches = [
        _batch(
            index,
            gradient_l2=gradient_l2,
            reference_l2=reference_l2,
            cosine=cosine,
        )
        for index in range(8)
    ]
    norm_delta = abs(gradient_l2 - reference_l2) / max(
        gradient_l2, reference_l2, 1.0e-30
    )
    viable = reference_l2 > 1.0e-12
    passed = viable and norm_delta <= 0.1 and cosine is not None and cosine >= 0.9
    diagnostics = [
        {
            "parameter": name,
            "batch_count": 8,
            "batches": copy.deepcopy(batches),
            "gradient_l2": gradient_l2,
            "reference_gradient_l2": reference_l2,
            "relative_gradient_l2_norm_delta": norm_delta,
            "gradient_vector_relative_error": abs(
                gradient_l2 - reference_l2
            )
            / max(reference_l2, 1.0e-30),
            "gradient_vector_cosine": cosine,
            "gradient_zero_fraction": 0.0,
            "reference_gradient_zero_fraction": 0.0,
            "absolute_zero_fraction_delta": 0.0,
            "reference_median_gradient_rms": 1.0,
            "reference_q90_zero_fraction": 0.0,
            "initial_weight_rms": 2.0,
            "reference_nominal_update_unit": 0.5,
            "reference_viable": viable,
            "passed": passed,
        }
        for name in CONV3_CONV_WEIGHTS
    ]
    return {
        "candidate_k": candidate_k,
        "reference_k": reference_k,
        "batch_count": 8,
        "cohort_examples": 256,
        "batch_size": 32,
        "official_test_read": False,
        "cohort_source_indices_sha256": cohort_sha,
        "initialization_checkpoint_sha256": "b" * 64,
        "initialization_tensor_sha256": "c" * 64,
        "selected_t": 16,
        "fixed_step_minimization": True,
        "batch_state_policy": "reset_each_batch",
        "reference_viable": viable,
        "passed": passed,
        "parameter_diagnostics": diagnostics,
        "free_equilibrium_sha256_by_batch": ["e" * 64] * 8,
    }


def test_frozen_config_contract_and_geometry() -> None:
    spec = PerfectDiodeTKStudySpec.from_path(CONFIG)
    assert spec.study_id == (
        "tkstudy_407b09b9a5bb6127224ece23361c1f350"
        "a05b2e3bcc950a41cffd5d51744ef30"
    )
    architecture = spec.data["model"]["architectures"]["conv3"]
    assert architecture["channels"] == [64, 128, 256]
    assert architecture["strides"] == [2, 2, 1]
    assert architecture["paddings"] == [1, 1, 1]
    assert architecture["output_dim"] == 20
    assert spec.data["model"]["input_gain"]["conv3"] == 360.0
    assert spec.data["dataset"]["official_test"]["read_allowed"] is False
    assert spec.data["dataset"]["train"]["batch_size"] == 16


def test_config_is_canonically_frozen() -> None:
    value = json.loads(CONFIG.read_text())
    value["model"]["input_gain"]["conv3"] = 361.0
    with pytest.raises(PerfectDiodeTKValidationError):
        PerfectDiodeTKStudySpec.from_dict(value)


def test_t_selection_is_strict_complete_and_context_bound() -> None:
    curve = {count: _t_record(count) for count in T_CORE_GRID}
    curve[4] = _t_record(4, p90=1.0e-2)
    assert select_t_measurements(curve)["selected_t"] == 6

    missing = dict(curve)
    missing.pop(10)
    assert select_t_measurements(missing)["status"] == "unresolved_t_incomplete_grid"

    float_keyed = {float(key): value for key, value in curve.items()}
    assert (
        select_t_measurements(float_keyed)["status"]
        == "unresolved_t_incomplete_grid"
    )

    mixed = copy.deepcopy(curve)
    mixed[16]["cohort_source_indices_sha256"] = "f" * 64
    assert select_t_measurements(mixed)["status"] == "unresolved_t_mixed_context"


def test_t_extension_requires_complete_largest_sentinel() -> None:
    curve = {count: _t_record(count, p90=2.0e-2) for count in T_CORE_GRID}
    assert select_t_measurements(curve)["status"] == "needs_extension"
    curve.update(
        {
            count: _t_record(count, p90=5.0e-3)
            for count in T_EXTENSION_GRID
        }
    )
    curve[256] = _t_record(256, p90=1.0e-2)
    assert (
        select_t_measurements(curve)["status"]
        == "unresolved_t_extension_sentinel"
    )
    curve[256] = _t_record(256, p90=5.0e-3)
    selection = select_t_measurements(curve)
    assert selection["status"] == "selected"
    assert selection["selected_t"] == 96
    assert selection["extension_used"] is True


def test_t_missing_layer_or_samples_never_selects() -> None:
    curve = {count: _t_record(count) for count in T_CORE_GRID}
    curve[8]["layers"].pop()
    assert (
        select_t_measurements(curve)["status"]
        == "unresolved_t_incomplete_measurement"
    )
    curve = {count: _t_record(count) for count in T_CORE_GRID}
    curve[8]["num_examples"] = 1023
    assert (
        select_t_measurements(curve)["status"]
        == "unresolved_t_incomplete_measurement"
    )


def test_k_selection_requires_complete_grid_and_exact_context() -> None:
    measurements = {count: _k_record(count) for count in K_GRID}
    assert select_k_measurements(measurements)["selected_k"] == 4
    missing = dict(measurements)
    missing.pop(16)
    assert (
        select_k_measurements(missing)["status"]
        == "unresolved_k_incomplete_grid"
    )
    mixed = copy.deepcopy(measurements)
    mixed[16]["cohort_source_indices_sha256"] = "f" * 64
    assert select_k_measurements(mixed)["status"] == "unresolved_k_mixed_context"


def test_k_uses_inclusive_mean_batch_gates_and_cosine() -> None:
    measurements = {count: _k_record(count) for count in K_GRID}
    measurements[4] = _k_record(4, gradient_l2=0.9, reference_l2=1.0, cosine=0.9)
    selection = select_k_measurements(measurements)
    assert selection["status"] == "selected"
    assert selection["selected_k"] == 4

    measurements[4] = _k_record(4, cosine=0.899)
    assert select_k_measurements(measurements)["selected_k"] == 6


def test_k_boundary_requires_exact_k128_sentinel() -> None:
    measurements = {
        count: (
            _k_record(count, gradient_l2=0.5)
            if count < 64
            else _k_record(count)
        )
        for count in K_GRID
    }
    assert (
        select_k_measurements(measurements)["status"]
        == "needs_k128_sentinel"
    )
    wrong = _k_record(64, reference_k=256)
    assert (
        select_k_measurements(measurements, k128_sentinel=wrong)["status"]
        == "unresolved_k128_incomplete_measurement"
    )
    sentinel = _k_record(64, reference_k=128)
    selected = select_k_measurements(
        measurements, k128_sentinel=sentinel
    )
    assert selected["status"] == "selected"
    assert selected["selected_k"] == 64


def test_k_recomputes_reference_viability_and_rejects_duplicate_parameters() -> None:
    measurements = {count: _k_record(count) for count in K_GRID}
    forged = copy.deepcopy(measurements)
    forged[4]["parameter_diagnostics"][0]["reference_median_gradient_rms"] = 2.0
    assert (
        select_k_measurements(forged)["status"]
        == "unresolved_k_incomplete_measurement"
    )
    duplicate = copy.deepcopy(measurements)
    duplicate[4]["parameter_diagnostics"][1]["parameter"] = "ConvWeight_0"
    assert (
        select_k_measurements(duplicate)["status"]
        == "unresolved_k_incomplete_measurement"
    )
