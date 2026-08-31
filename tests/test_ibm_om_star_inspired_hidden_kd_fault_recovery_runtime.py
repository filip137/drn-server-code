from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from experiments.artifacts import sha256_file
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_star_inspired_hidden_kd_fault_recovery_runtime import (
    ARMS,
    _assert_evaluation_identity,
    _component_gradient_report,
    _load_healthy_initializer,
    _relative_pulse_mismatch,
    _selection_key,
    _summaries_by_arm,
)


def _mapping(path, *, negative: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    layer0 = torch.ones((1568, 100), dtype=torch.float32)
    layer1 = torch.full((100, 20), 1.25, dtype=torch.float32)
    if negative:
        layer0[0, 0] = -0.01
    np.savez_compressed(
        path,
        schema=np.asarray("ebl.mnist_relu_drn.ibm_om_baseline_physical_mapping"),
        schema_version=np.asarray(1, dtype=np.int64),
        assignment_seed=np.asarray(87004, dtype=np.int64),
        conductance_min=np.asarray(0.0, dtype=np.float64),
        conductance_max=np.asarray(2.0, dtype=np.float64),
        layer_0_quantized_conductance=layer0.numpy(),
        layer_1_quantized_conductance=layer1.numpy(),
    )
    return layer0, layer1


def test_initializer_keeps_raw_a_distinct_from_positive_full_conductance(tmp_path) -> None:
    path = tmp_path / "mapping.npz"
    expected = _mapping(path)
    full_g, raw_a, report = _load_healthy_initializer(
        path,
        expected_sha256=sha256_file(path),
        expected_tensor_sha256=tuple(_tensor_sha256(value) for value in expected),
        expected_assignment_seed=87004,
    )
    torch.testing.assert_close(full_g[0], torch.ones_like(full_g[0]))
    torch.testing.assert_close(raw_a[0], torch.zeros_like(raw_a[0]))
    torch.testing.assert_close(raw_a[1], torch.full_like(raw_a[1], 0.25))
    assert report["coordinate_transform"] == "G=raw_a+1=2*x"
    assert report["negative_conductance_count"] == 0


def test_initializer_fails_closed_on_negative_circuit_conductance(tmp_path) -> None:
    path = tmp_path / "mapping.npz"
    tensors = _mapping(path, negative=True)
    with pytest.raises(ValueError, match="non-negative"):
        _load_healthy_initializer(
            path,
            expected_sha256=sha256_file(path),
            expected_tensor_sha256=tuple(_tensor_sha256(value) for value in tensors),
            expected_assignment_seed=87004,
        )


def test_selection_and_aggregate_use_declared_axes() -> None:
    candidates = [
        {
            "lambda_multiplier": 0.3,
            "validation": {"student_correct": 4500, "output_kl": 0.2},
        },
        {
            "lambda_multiplier": 1.0,
            "validation": {"student_correct": 4501, "output_kl": 0.4},
        },
        {
            "lambda_multiplier": 3.0,
            "validation": {"student_correct": 4501, "output_kl": 0.3},
        },
    ]
    assert min(candidates, key=_selection_key)["lambda_multiplier"] == 3.0

    rows = []
    for arm_index, arm in enumerate(ARMS):
        for seed_index, seed in enumerate((91502, 91503)):
            rows.append(
                {
                    "arm": arm,
                    "fault_seed": seed,
                    "test": {"student_accuracy": 0.70 + 0.01 * arm_index + 0.001 * seed_index},
                    "training": {
                        "write_ledger": {"attempted_pulse_requests": 100 + arm_index}
                    },
                }
            )
    aggregate = _summaries_by_arm(rows)
    assert aggregate["output_kl_only"]["masks"] == 2
    assert aggregate["relu_sample_hidden_kl"]["mean_test_gain_vs_output_only"] == pytest.approx(0.01)


def test_pulse_mismatch_is_relative_to_output_only_baseline() -> None:
    assert _relative_pulse_mismatch(200, 100) == pytest.approx(1.0)
    assert _relative_pulse_mismatch(50, 100) == pytest.approx(0.5)


def test_healthy_identity_gate_checks_count_and_prediction_hash() -> None:
    report = {
        "examples": 5_000,
        "student_correct": 4_776,
        "prediction_sha256": "expected",
    }
    _assert_evaluation_identity(
        report,
        split="validation",
        expected_examples=5_000,
        expected_correct=4_776,
        expected_prediction_sha256="expected",
    )
    with pytest.raises(RuntimeError, match="prediction identity changed"):
        _assert_evaluation_identity(
            report,
            split="validation",
            expected_examples=5_000,
            expected_correct=4_775,
            expected_prediction_sha256="expected",
        )


def test_gradient_calibration_uses_mean_batch_rms(monkeypatch) -> None:
    import experiments.mnist_relu_drn.ibm_om_star_inspired_hidden_kd_fault_recovery_runtime as runtime

    stack = SimpleNamespace(
        device=torch.device("cpu"),
        bundle=SimpleNamespace(catalog=SimpleNamespace(trainable=(object(),)))
    )
    calls = iter(
        (
            (torch.tensor([1.0, 1.0]),),
            (torch.tensor([2.0, 2.0]),),
            (torch.tensor([3.0, 3.0]),),
            (torch.tensor([7.0, 7.0]),),
        )
    )
    monkeypatch.setattr(runtime, "_physical_gradients", lambda *args, **kwargs: next(calls))
    loader = (
        (torch.zeros((1, 1)), torch.zeros((1,), dtype=torch.long)),
        (torch.zeros((1, 1)), torch.zeros((1,), dtype=torch.long)),
    )
    report = _component_gradient_report(
        stack, object(), loader, maximum_batches=None
    )
    layer = report["layers"][0]
    assert layer["rms_aggregation"] == "mean_of_batch_rms"
    assert layer["output_gradient_rms"] == pytest.approx(2.0)
    assert layer["hidden_gradient_rms"] == pytest.approx(2.5)
    assert layer["pooled_output_gradient_rms"] == pytest.approx(5.0**0.5)
