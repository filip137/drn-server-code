from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.mnist_conv import lr_stages
from experiments.mnist_conv.lr_protocol import (
    RangeCandidateSet,
    RangeGateResult,
    range_normalized_update_schedule,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
V1_CONFIG = REPO_ROOT / "configs/conv/hardsigmoid_lr_study_sgd_bs16_v1.json"
RESCUE_CONFIG = (
    REPO_ROOT / "configs/conv/hardsigmoid_lr_rescue_warmin_sgd_bs16_v2.json"
)
V3_CONFIG = (
    REPO_ROOT / "configs/conv/hardsigmoid_lr_relative_rho_sgd_bs16_v3.json"
)


class _LoaderBundle:
    def __init__(self) -> None:
        self.train_loader = [object()] * 1_400
        self.validation_loader = object()
        self.train_indices_hash = "train-indices"
        self.validation_indices_hash = "validation-indices"
        self.reset_count = 0

    def reset_train_shuffle(self) -> None:
        self.reset_count += 1


def _install_safe_range_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Replace training and I/O while retaining the executor's real schedules."""

    observed: dict[str, Any] = {
        "training_learning_rates": [],
        "scheduled_steps": [],
        "gate_calls": [],
        "validation_steps": [],
        "initial_learning_rates": [],
        "transition_calls": [],
        "candidate_calls": [],
    }
    bundle = _LoaderBundle()
    observed["bundle"] = bundle
    transition_parameter = SimpleNamespace(
        name="ConvWeight_0",
        bounded_gate=True,
        gradient_rms=1.0,
        proposed_update_rms=1.0,
        normalized_update=0.01,
        lower_bound_occupancy=0.5,
        upper_bound_occupancy=0.0,
        projection_efficiency=1.0,
        proposed_bound_crossing_fraction=0.0,
    )

    def build_model_runtime(*args: Any, **kwargs: Any) -> SimpleNamespace:
        del args
        observed["initial_learning_rates"].append(kwargs["learning_rate"])
        return SimpleNamespace(parameters=object())

    def training_step(*args: Any, **kwargs: Any) -> SimpleNamespace:
        del args
        observed["training_learning_rates"].append(kwargs["learning_rate"])
        step = len(observed["training_learning_rates"])
        return SimpleNamespace(
            source_indices=(step,),
            loss=1.0,
            accuracy=0.5,
            sample_count=16,
            transition=SimpleNamespace(parameters=(transition_parameter,)),
        )

    def transition_rows(*args: Any, **kwargs: Any) -> list[Any]:
        del args
        observed["transition_calls"].append(dict(kwargs))
        observed["scheduled_steps"].append(
            (kwargs["step"], kwargs["rho_schedule"], kwargs["learning_rate"])
        )
        return []

    def evaluate_validation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        observed["validation_steps"].append(
            len(observed["training_learning_rates"])
        )
        return {
            "loss": 1.0,
            "accuracy": 0.5,
            "sample_count": 5_000,
            "source_indices": (),
        }

    def analyze_range_stop_gates(
        records: Any,
        initial_occupancy: Any,
        **kwargs: Any,
    ) -> RangeGateResult:
        del initial_occupancy
        observed["gate_calls"].append((len(records), dict(kwargs)))
        return RangeGateResult(
            failure=None,
            maximum_admissible_step=len(records),
            maximum_admissible_learning_rate=records[-1].learning_rate,
            maximum_admissible_rho=records[-1].rho,
            last_processed_step=len(records),
            extension_allowed=True,
        )

    monkeypatch.setattr(lr_stages, "build_model_runtime", build_model_runtime)
    monkeypatch.setattr(lr_stages, "build_loader_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        lr_stages,
        "parameter_state_diagnostics",
        lambda parameters: {
            "ConvWeight_0": {
                "bounded_gate": True,
                "combined_bound_occupancy": 0.5,
                "rms": 2.0,
            }
        },
    )
    monkeypatch.setattr(lr_stages, "training_step", training_step)
    monkeypatch.setattr(lr_stages, "transition_rows", transition_rows)
    monkeypatch.setattr(lr_stages, "evaluate_validation", evaluate_validation)
    monkeypatch.setattr(
        lr_stages, "analyze_range_stop_gates", analyze_range_stop_gates
    )
    def extract_range_candidates(*args: Any, **kwargs: Any) -> RangeCandidateSet:
        observed["candidate_calls"].append((args, dict(kwargs)))
        return RangeCandidateSet(
            status="unresolved",
            reason="test_harness",
            fast=None,
            middle=None,
            high=None,
            stable_targets=(),
        )

    monkeypatch.setattr(
        lr_stages,
        "extract_range_candidates",
        extract_range_candidates,
    )
    monkeypatch.setattr(lr_stages, "_batch_hash", lambda batches: "batch-order")
    monkeypatch.setattr(lr_stages, "atomic_write_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(lr_stages, "atomic_write_json", lambda *args, **kwargs: None)
    return observed


def _prepare_study_root(tmp_path: Path, study: dict[str, Any], row_id: str) -> Path:
    row = next(row for row in study["rows"] if row["row_id"] == row_id)
    root = tmp_path / "study"
    checkpoint = root / "initialization" / f"{row['architecture']}.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixed-test-checkpoint")
    probe_dir = root / "stages" / "probe" / "entries" / row_id
    probe_dir.mkdir(parents=True)
    (probe_dir / "summary.json").write_text(
        json.dumps({"rho_unit": 1.0}), encoding="utf-8"
    )
    return root


def test_rescue_executor_wires_exact_1199_step_tail_and_128_step_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = json.loads(RESCUE_CONFIG.read_text(encoding="utf-8"))
    row_id = study["rows"][0]["row_id"]
    root = _prepare_study_root(tmp_path, study, row_id)
    observed = _install_safe_range_harness(monkeypatch)

    summary = lr_stages.execute_range_entry(
        study,
        root,
        row_id,
        data_root=tmp_path / "unused-data",
        download=False,
        device="cpu",
    )

    scheduled = observed["scheduled_steps"]
    steps = tuple(item[0] for item in scheduled)
    rhos = tuple(item[1] for item in scheduled)
    learning_rates = tuple(item[2] for item in scheduled)
    v1_main = range_normalized_update_schedule()

    assert len(scheduled) == 1_327
    assert steps == tuple(range(1, 1_328))
    assert rhos[0] == 1e-8
    assert rhos[599] == 1e-5
    assert rhos[599:1_199] == v1_main
    assert rhos[1_198] == 1e-2
    assert steps[1_199] == 1_200
    assert rhos[1_199] == 1e-2
    assert steps[-1] == 1_327
    assert rhos[-1] == 3e-2
    assert learning_rates == rhos
    assert tuple(observed["training_learning_rates"]) == rhos
    assert observed["initial_learning_rates"] == [1e-8]
    assert observed["validation_steps"] == study["range_test"]["validation_steps"]
    assert observed["bundle"].reset_count == 1
    assert all(
        call_kwargs["main_range_steps"] == 1_199
        for _, call_kwargs in observed["gate_calls"]
    )
    assert sum(length == 1_199 for length, _ in observed["gate_calls"]) == 2
    assert summary["extension_ran"] is True
    assert summary["completed_steps"] == 1_327
    assert summary["range_schedule"]["extension_start_step"] == 1_200
    assert summary["range_schedule"]["extension_end_step"] == 1_327


def test_v1_executor_keeps_600_step_main_and_extension_at_global_step_601(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = json.loads(V1_CONFIG.read_text(encoding="utf-8"))
    row_id = study["rows"][0]["row_id"]
    root = _prepare_study_root(tmp_path, study, row_id)
    observed = _install_safe_range_harness(monkeypatch)

    summary = lr_stages.execute_range_entry(
        study,
        root,
        row_id,
        data_root=tmp_path / "unused-data",
        download=False,
        device="cpu",
    )

    scheduled = observed["scheduled_steps"]
    steps = tuple(item[0] for item in scheduled)
    rhos = tuple(item[1] for item in scheduled)
    v1_main = range_normalized_update_schedule()

    assert len(scheduled) == 728
    assert steps == tuple(range(1, 729))
    assert rhos[:600] == v1_main
    assert rhos[0] == 1e-5
    assert rhos[599] == 1e-2
    assert steps[600] == 601
    assert rhos[600] == 1e-2
    assert steps[-1] == 728
    assert rhos[-1] == 3e-2
    assert observed["validation_steps"] == study["range_test"]["validation_steps"]
    assert all(
        call_kwargs["main_range_steps"] == 600
        for _, call_kwargs in observed["gate_calls"]
    )
    assert sum(length == 600 for length, _ in observed["gate_calls"]) == 2
    assert summary["extension_ran"] is True
    assert summary["completed_steps"] == 728
    assert "range_schedule" not in summary


def test_v3_executor_wires_relative_rho_schedule_and_dual_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = json.loads(V3_CONFIG.read_text(encoding="utf-8"))
    row_id = study["rows"][0]["row_id"]
    root = _prepare_study_root(tmp_path, study, row_id)
    probe_summary = root / "stages" / "probe" / "entries" / row_id / "summary.json"
    probe_summary.write_text(
        json.dumps(
            {
                "rho_unit": 1.0,
                "rho_unit_relative": 1.0,
                "rho_span_unit_diagnostic": 0.25,
            }
        ),
        encoding="utf-8",
    )
    observed = _install_safe_range_harness(monkeypatch)

    summary = lr_stages.execute_range_entry(
        study,
        root,
        row_id,
        data_root=tmp_path / "unused-data",
        download=False,
        device="cpu",
    )

    scheduled = observed["scheduled_steps"]
    steps = tuple(item[0] for item in scheduled)
    relative_rhos = tuple(item[1] for item in scheduled)
    learning_rates = tuple(item[2] for item in scheduled)

    assert len(scheduled) == 928
    assert steps == tuple(range(1, 929))
    assert relative_rhos[0] == 3e-4
    assert relative_rhos[799] == 3.0
    assert relative_rhos[800] == 3.0
    assert relative_rhos[-1] == 9.0
    assert learning_rates == relative_rhos
    assert observed["initial_learning_rates"] == [3e-4]
    assert tuple(observed["training_learning_rates"]) == relative_rhos

    transition_calls = observed["transition_calls"]
    assert len(transition_calls) == 928
    assert all(
        call["bounded_initial_rms_scales"] == {"ConvWeight_0": 2.0}
        for call in transition_calls
    )
    assert tuple(call["rho_schedule_relative"] for call in transition_calls) == (
        relative_rhos
    )
    assert tuple(call["rho_schedule_span"] for call in transition_calls) == tuple(
        rho * 0.25 for rho in relative_rhos
    )
    assert transition_calls[0]["rho_schedule_span"] == pytest.approx(7.5e-5)
    assert transition_calls[-1]["rho_schedule_span"] == pytest.approx(2.25)

    assert observed["validation_steps"] == study["range_test"]["validation_steps"]
    assert observed["bundle"].reset_count == 1
    assert all(
        call_kwargs["main_range_steps"] == 800
        for _, call_kwargs in observed["gate_calls"]
    )
    assert sum(length == 800 for length, _ in observed["gate_calls"]) == 2

    assert len(observed["candidate_calls"]) == 1
    candidate_args, candidate_kwargs = observed["candidate_calls"][0]
    assert len(candidate_args[0]) == 928
    assert candidate_args[0][0].rho == 3e-4
    assert candidate_args[0][799].rho == 3.0
    assert candidate_args[0][-1].rho == 9.0
    assert candidate_kwargs == {
        "failure_onset_step": None,
        "targets": study["probe"]["rho_targets"],
        "reference_steps": 32,
        "summary_steps": 8,
        "ema_decay": 0.98,
    }

    assert summary["schema_version"] == "mnist-conv-lr-range-result/v2"
    assert summary["rho_definition"] == "initial_parameter_rms"
    assert summary["rho_unit"] == 1.0
    assert summary["rho_unit_relative"] == 1.0
    assert summary["rho_span_unit_diagnostic"] == 0.25
    assert summary["completed_steps"] == 928
    assert summary["extension_ran"] is True
    assert summary["maximum_admissible_rho"] == 9.0
    assert summary["maximum_admissible_rho_relative"] == 9.0
    assert summary["maximum_admissible_rho_span"] == pytest.approx(2.25)
    assert summary["candidates"]["status"] == "unresolved"
    assert summary["range_schedule"] == {
        "name": "geometric_including_endpoints",
        "main_steps": 800,
        "start_rho": 3e-4,
        "end_rho": 3.0,
        "extension_steps": 128,
        "extension_start_step": 801,
        "extension_end_step": 928,
    }
