from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import run_mnist_conv_perfectdiode_job as adapter


def _artifacts(output: Path) -> None:
    for name in (
        "result.json",
        "run_spec.json",
        "step_log.csv",
        "validation.json",
        "best_validation.pt",
        "final.pt",
    ):
        (output / name).write_bytes(name.encode())


def _entry() -> dict:
    return {
        "entry_id": "job-a",
        "row_id": "conv1_baseline_v1_c1",
        "architecture": "conv1",
        "scheme": "baseline",
        "optimizer": "sgd",
        "rho_conv": 0.009,
        "rho_dense": 0.03,
        "raw_learning_rates_by_parameter": {
            "ConvWeight_0": 0.1,
            "Bias_0": 0.1,
            "DenseWeight_0": 0.01,
        },
        "epochs": 10,
        "expected_steps": 34380,
        "probe_result": "stages/probe/result.json",
    }


def test_production_adapter_writes_simple_completion(monkeypatch, tmp_path) -> None:
    output = tmp_path / "job"

    def execute(*args, **kwargs):
        output.mkdir(parents=True)
        _artifacts(output)
        return {
            "status": "complete",
            "official_test_read": False,
            "completed_steps": 34380,
        }

    monkeypatch.setattr(adapter, "execute_successor_stage_entry", execute)
    monkeypatch.setattr(
        adapter,
        "capture_environment",
        lambda device: {"device": device, "pytorch_version": "functional"},
    )
    receipt = adapter.run_production_job(
        {"study": "fixture"},
        tmp_path / "bundle",
        _entry(),
        attempt_id="attempt-a",
        target="local",
        data_root=tmp_path / "data",
        device="cuda:0",
        output_dir=output,
    )

    assert receipt["status"] == "complete"
    assert receipt["outcome"] == "completed"
    assert receipt["attempt_id"] == "attempt-a"
    assert receipt["job_id"] == "job-a"
    assert len(receipt["artifacts"]) == 6
    assert json.loads((output / "job_completion.json").read_text())[
        "job_id"
    ] == "job-a"


def test_declared_safety_failure_is_negative_evidence(monkeypatch, tmp_path) -> None:
    output = tmp_path / "job"

    def execute(*args, **kwargs):
        output.mkdir(parents=True)
        _artifacts(output)
        return {
            "status": "safety_failure",
            "official_test_read": False,
            "completed_steps": 12,
        }

    monkeypatch.setattr(adapter, "execute_successor_stage_entry", execute)
    monkeypatch.setattr(
        adapter, "capture_environment", lambda device: {"device": device}
    )
    result = adapter.run_production_job(
        {},
        tmp_path / "bundle",
        _entry(),
        attempt_id="attempt-b",
        target="trex",
        data_root=tmp_path / "data",
        device="cuda",
        output_dir=output,
    )
    assert result["outcome"] == "negative_evidence"


def test_existing_completion_is_idempotent(monkeypatch, tmp_path) -> None:
    output = tmp_path / "job"
    output.mkdir()
    receipt = {
        "schema_version": "experiment-job-completion/v1",
        "status": "complete",
        "outcome": "completed",
        "attempt_id": "attempt-c",
        "job_id": "job-a",
        "official_test_read": False,
        "result": {"official_test_read": False},
        "artifacts": [{"path": "result.json", "sha256": "a" * 64}],
    }
    (output / "job_completion.json").write_text(json.dumps(receipt))

    monkeypatch.setattr(
        adapter,
        "execute_successor_stage_entry",
        lambda *args, **kwargs: pytest.fail("must not rerun a complete job"),
    )
    assert adapter.run_production_job(
        {},
        tmp_path / "bundle",
        _entry(),
        attempt_id="attempt-c",
        target="local",
        data_root=tmp_path / "data",
        device="cuda",
        output_dir=output,
    ) == receipt


def test_partial_output_fails_without_overwrite(tmp_path) -> None:
    output = tmp_path / "job"
    output.mkdir()
    (output / "partial.log").write_text("partial")
    with pytest.raises(FileExistsError, match="fresh job output"):
        adapter.run_production_job(
            {},
            tmp_path / "bundle",
            _entry(),
            attempt_id="attempt-d",
            target="akib",
            data_root=tmp_path / "data",
            device="cuda",
            output_dir=output,
        )


def test_runtime_payload_maps_existing_probe_path() -> None:
    value = adapter._runtime_payload(_entry())
    assert value["probe_result_path"] == "stages/probe/result.json"
