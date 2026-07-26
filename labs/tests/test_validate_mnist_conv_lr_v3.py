from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from experiments.mnist_conv.lr_engine import parameter_tensor_digest
from experiments.mnist_conv.lr_protocol import (
    candidate_learning_rate_at_step,
    linear_quantile,
)
from labs.tools.validate_mnist_conv_lr_v3 import (
    AuditError,
    TOTAL_STEPS,
    _aggregate_observed_rho,
    _audit_initialization_parameter_diagnostics,
    _audit_step_log,
    _checkpoint_tensor_digest,
    main,
)


STEP_FIELDS = [
    "step",
    "epoch",
    "batch_in_epoch",
    "learning_rate",
    "scheduled_rho",
    "scheduled_rho_relative",
    "scheduled_rho_span",
    "loss",
    "loss_ema",
    "accuracy",
    "sample_count",
    "status",
    "failure_reason",
]


class _FakeParameter:
    def __init__(self, name: str, state: torch.Tensor) -> None:
        self.name = name
        self.state = state


def _write_checkpoint(path: Path, parameters: list[_FakeParameter]) -> None:
    torch.save(
        {
            "format": "drn.function.parameters",
            "version": 1,
            "schema": [
                {
                    "name": parameter.name,
                    "type": (
                        f"{type(parameter).__module__}."
                        f"{type(parameter).__qualname__}"
                    ),
                    "shape": list(parameter.state.shape),
                    "dtype": str(parameter.state.dtype),
                }
                for parameter in parameters
            ],
            "states": [parameter.state for parameter in parameters],
        },
        path,
    )


def _write_step_log(path: Path, peak: float, *, drift_step: int | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STEP_FIELDS)
        writer.writeheader()
        for step in range(1, TOTAL_STEPS + 1):
            batch = (step - 1) % 3_438 + 1
            learning_rate = candidate_learning_rate_at_step(peak, step)
            if step == drift_step:
                learning_rate *= 1.01
            writer.writerow(
                {
                    "step": step,
                    "epoch": (step - 1) // 3_438 + 1,
                    "batch_in_epoch": batch,
                    "learning_rate": learning_rate,
                    "scheduled_rho": None,
                    "scheduled_rho_relative": None,
                    "scheduled_rho_span": None,
                    "loss": 1.0,
                    "loss_ema": 1.0,
                    "accuracy": 0.5,
                    "sample_count": 8 if batch == 3_438 else 16,
                    "status": "complete",
                    "failure_reason": None,
                }
            )


def test_step_log_audit_accepts_exact_17190_step_schedule(tmp_path: Path) -> None:
    peak = 0.125
    path = tmp_path / "step_log.csv"
    _write_step_log(path, peak)

    learning_rates = _audit_step_log(path, peak, "candidate")

    assert len(learning_rates) == TOTAL_STEPS + 1
    assert learning_rates[1] == pytest.approx(peak / 860)
    assert learning_rates[860] == peak
    assert learning_rates[-1] == 0.0


def test_step_log_audit_fails_closed_on_schedule_drift(tmp_path: Path) -> None:
    peak = 0.125
    path = tmp_path / "step_log.csv"
    _write_step_log(path, peak, drift_step=861)

    with pytest.raises(
        AuditError,
        match=r"Expected candidate\.step_log\[861\]\.learning_rate .*Provided value",
    ):
        _audit_step_log(path, peak, "candidate")


def test_observed_rho_uses_per_parameter_q90_then_maximum() -> None:
    relative = {
        "ConvWeight_0": [0.1] * 32,
        "DenseWeight_0": [float(value) for value in range(1, 33)],
        "ConvWeight_1": [0.01] * 32,
    }
    span = [0.001 * value for value in range(96)]

    observed = _aggregate_observed_rho(relative, span, path="candidate")

    assert observed["observed_relative_by_parameter"] == {
        "ConvWeight_0": 0.1,
        "ConvWeight_1": 0.01,
        "DenseWeight_0": linear_quantile(range(1, 33), 0.9),
    }
    assert observed["observed_relative"] == linear_quantile(range(1, 33), 0.9)
    assert observed["observed_span"] == linear_quantile(span, 0.9)


def test_observed_rho_rejects_incomplete_warmup_window() -> None:
    with pytest.raises(AuditError, match="Expected .*Provided value"):
        _aggregate_observed_rho(
            {"DenseWeight_0": [0.1] * 31},
            [0.001] * 31,
            path="candidate",
        )


def test_checkpoint_audit_recomputes_exact_parameter_tensor_digest(
    tmp_path: Path,
) -> None:
    parameters = [
        _FakeParameter("DenseWeight_0 ", torch.tensor([0.0, 0.25, 1.0])),
        _FakeParameter("Bias_0", torch.tensor([-0.5, 0.5])),
    ]
    path = tmp_path / "checkpoint.pt"
    _write_checkpoint(path, parameters)

    assert _checkpoint_tensor_digest(path, "checkpoint") == parameter_tensor_digest(
        parameters
    )


def test_checkpoint_audit_rejects_nonfinite_tensor(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    _write_checkpoint(
        path,
        [_FakeParameter("DenseWeight_0", torch.tensor([0.0, torch.nan]))],
    )

    with pytest.raises(AuditError, match="Expected .*entirely finite.*Provided value"):
        _checkpoint_tensor_digest(path, "checkpoint")


def test_initialization_audit_recomputes_rho_scale_from_checkpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _write_checkpoint(
        path,
        [_FakeParameter("DenseWeight_0", torch.tensor([0.0, 0.25, 1.0]))],
    )
    rms = float(torch.sqrt(torch.mean(torch.tensor([0.0, 0.25, 1.0]).double() ** 2)))
    metadata = {
        "DenseWeight_0": {
            "name": "DenseWeight_0",
            "parameter_type": "_FakeParameter",
            "shape": [3],
            "rms": rms,
            "std": float(
                torch.tensor([0.0, 0.25, 1.0]).double().std(unbiased=False)
            ),
            "bounded_gate": True,
            "report_only": False,
            "lower_bound": 0.0,
            "upper_bound": 100.0,
            "lower_bound_occupancy": 1.0 / 3.0,
            "upper_bound_occupancy": 0.0,
            "combined_bound_occupancy": 1.0 / 3.0,
        }
    }

    _audit_initialization_parameter_diagnostics(
        path,
        metadata,
        label="checkpoint",
    )

    metadata["DenseWeight_0"]["rms"] *= 2.0
    with pytest.raises(AuditError, match=r"metadata DenseWeight_0\.rms"):
        _audit_initialization_parameter_diagnostics(
            path,
            metadata,
            label="checkpoint",
        )


def test_cli_failure_is_concise_json_and_read_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing-study"

    assert main(["--study", str(missing)]) == 1
    captured = capsys.readouterr()
    failure = json.loads(captured.err)
    assert failure["status"] == "failed"
    assert failure["error_type"] == "AuditError"
    assert "Expected --study" in failure["error"]
    assert "Provided value" in failure["error"]
    assert not missing.exists()
