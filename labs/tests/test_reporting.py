from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.reporting import (
    ACTIVE_BEGIN,
    ACTIVE_END,
    append_metric,
    complete_run,
    fail_run,
    start_run,
    update_status_progress,
    validate_run,
)
from labs.mnist_train import _make_reporting_epoch_callback


def _manifest(study_id: str = "study-a", run_id: str = "run-a") -> dict:
    return {
        "study_id": study_id,
        "run_id": run_id,
        "arm_id": "conv1-baseline-sgd-seed0",
        "evidence_class": "ordinary_mnist_selection",
        "configuration": {"sha256": "a" * 64},
        "dataset": {
            "key": "mnist",
            "evaluation_split": "validation",
            "official_test_read": False,
        },
        "command": ["python", "-m", "example"],
        "runtime": {"target": "local", "host": "test-host"},
    }


def test_successful_bundle_records_metrics_and_indexes_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    start_run(run_dir, _manifest())
    append_metric(
        run_dir / "metrics.jsonl",
        {
            "kind": "epoch",
            "epoch": 1,
            "evaluation_split": "validation",
            "metrics": {"validation_accuracy": 0.9},
        },
    )
    (run_dir / "final_model.pt").write_bytes(b"checkpoint")
    (run_dir / "training.log").write_text("done\n", encoding="utf-8")

    result = complete_run(
        run_dir,
        terminal_metrics={"validation": {"final_accuracy": 0.9}},
    )

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "complete"
    assert (run_dir / "result.json").is_file()
    assert (run_dir / "checkpoints" / "final_model.pt").is_symlink()
    assert (run_dir / "artifacts" / "training.log").is_symlink()
    assert {item["kind"] for item in result["artifacts"]} == {
        "artifact",
        "checkpoint",
    }
    assert validate_run(run_dir) == []


def test_failed_bundle_retains_error_without_result(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    start_run(run_dir, _manifest())
    (run_dir / "partial.log").write_text("before failure\n", encoding="utf-8")

    fail_run(run_dir, error=ValueError("non-finite loss"))

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["error"] == {
        "type": "ValueError",
        "message": "non-finite loss",
    }
    assert not (run_dir / "result.json").exists()
    assert validate_run(run_dir) == []


def test_manifest_is_immutable(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    start_run(run_dir, _manifest())
    changed = _manifest()
    changed["arm_id"] = "different-arm"

    with pytest.raises(RuntimeError, match="manifest.json does not match"):
        start_run(run_dir, changed)


def test_epoch_callback_uses_validation_name_and_updates_heartbeat(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    start_run(run_dir, _manifest())
    callback = _make_reporting_epoch_callback(
        run_dir,
        {"schema": "mnist-train-validation-split/v1"},
        3,
        None,
    )

    assert (
        callback(
            {
                "epoch": 1,
                "train_loss": 0.5,
                "train_accuracy": 0.8,
                "test_loss": 0.4,
                "test_accuracy": 0.85,
                "is_best": True,
                "best_epoch": 1,
                "best_accuracy": 0.85,
            },
            {},
        )
        is False
    )

    metric = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8"))
    assert metric["evaluation_split"] == "validation"
    assert metric["metrics"]["validation_accuracy"] == 0.85
    assert "test_accuracy" not in metric["metrics"]
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["progress"] == {
        "stage": "training",
        "epoch": 1,
        "epochs": 3,
    }


def test_active_block_groups_running_arms_and_preserves_manual_text(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    document = tmp_path / "docs" / "current_simulations.md"
    document.parent.mkdir()
    document.write_text(
        "\n".join(
            [
                "# Current",
                "",
                ACTIVE_BEGIN,
                "old generated text",
                ACTIVE_END,
                "",
                "## Queued",
                "",
                "Keep this manual note.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    first = results_root / "study-a" / "run-a"
    second = results_root / "study-a" / "run-b"
    start_run(first, _manifest(run_id="run-a"))
    start_run(second, _manifest(run_id="run-b"))

    active = document.read_text(encoding="utf-8")
    assert active.count("### `study-a`") == 1
    assert "`run-a`" in active
    assert "`run-b`" in active
    assert "Keep this manual note." in active

    complete_run(first, terminal_metrics={})
    fail_run(second, error="cancelled")
    inactive = document.read_text(encoding="utf-8")
    assert "No reporting-contract runs" in inactive
    assert "Keep this manual note." in inactive


def test_progress_rejects_terminal_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    start_run(run_dir, _manifest())
    fail_run(run_dir, error="stopped")

    with pytest.raises(RuntimeError, match="running experiment"):
        update_status_progress(run_dir, {"stage": "training"})
