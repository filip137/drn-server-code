from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
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
    atomic_write_json,
    inspect_runs,
    observe_status,
    parse_args,
    pause_run,
    refresh_active_document,
    resume_run,
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


def test_refresh_requires_explicit_document(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(["refresh-active"])
    assert exc.value.code == 2
    document = tmp_path / "observations.md"
    assert parse_args(["refresh-active", "--document", str(document)]).document == document


@pytest.mark.parametrize("name", ["current_simulations.md", "experimental_manifest.md"])
def test_refresh_preserves_retired_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    document = tmp_path / name
    original = f"# Historical\n{ACTIVE_BEGIN}\nold observations\n{ACTIVE_END}\n"
    document.write_text(original)
    monkeypatch.setattr("experiments.reporting.RETIRED_DOCUMENTS", {document.resolve()})
    alias = tmp_path / "alias.md"
    alias.symlink_to(document)
    for destination in (document, alias):
        with pytest.raises(ValueError, match="retired historical snapshot"):
            refresh_active_document(tmp_path / "results", destination)
    assert document.read_text() == original


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

    assert "old generated text" in document.read_text()
    refresh_active_document(results_root, document)
    active = document.read_text(encoding="utf-8")
    assert active.count("### `study-a`") == 1
    assert "`run-a`" in active
    assert "`run-b`" in active
    assert "Keep this manual note." in active

    complete_run(first, terminal_metrics={})
    fail_run(second, error="cancelled")
    assert document.read_text() == active
    refresh_active_document(results_root, document)
    inactive = document.read_text(encoding="utf-8")
    assert "No reporting-contract runs" in inactive
    assert "Keep this manual note." in inactive


def test_progress_rejects_terminal_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    start_run(run_dir, _manifest())
    fail_run(run_dir, error="stopped")

    with pytest.raises(RuntimeError, match="running experiment"):
        update_status_progress(run_dir, {"stage": "training"})


def test_changed_terminal_result_is_rejected(tmp_path):
    start_run(tmp_path, _manifest())
    result = complete_run(tmp_path, terminal_metrics={"accuracy": .8})
    result["terminal_metrics"]["accuracy"] = .99
    atomic_write_json(tmp_path / "result.json", result)
    assert "result hash mismatch" in validate_run(tmp_path)


def test_rejected_scientific_criterion_is_valid_negative_evidence(tmp_path):
    start_run(tmp_path, _manifest())
    with pytest.raises(ValueError, match="criteria_met"):
        complete_run(tmp_path, terminal_metrics={}, completion={})
    assert not (tmp_path / "result.json").exists()
    complete_run(tmp_path, terminal_metrics={}, completion={"criteria_met": False})
    assert validate_run(tmp_path) == []


@pytest.mark.parametrize("record", ['not json', '[]', '{"loss": NaN}', '{"loss": 1e999}'])
def test_malformed_or_nonfinite_metrics_are_rejected(tmp_path, record):
    start_run(tmp_path, _manifest())
    (tmp_path / "metrics.jsonl").write_text(record + '\n')
    assert any("metrics.jsonl line 1" in error for error in validate_run(tmp_path))
    with pytest.raises(ValueError, match="metrics.jsonl line 1"):
        complete_run(tmp_path, terminal_metrics={})
    assert not (tmp_path / "result.json").exists()


def test_pause_is_valid_without_success_and_resume_checks_checkpoint(tmp_path):
    start_run(tmp_path, _manifest())
    checkpoint = tmp_path / "checkpoints/resume.pt"
    checkpoint.write_bytes(b"full-state-checkpoint")
    pause_run(tmp_path, checkpoint=checkpoint, progress={"epoch": 2}, reason="Night window ended")
    assert not validate_run(tmp_path)
    assert inspect_runs([tmp_path])[0]["observation"] == "paused"
    assert not inspect_runs([tmp_path])[0]["alerts"]
    with pytest.raises(RuntimeError, match="running"):
        update_status_progress(tmp_path, {"epoch": 3})
    checkpoint.write_bytes(b"changed")
    assert "paused checkpoint missing or hash mismatch" in validate_run(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        resume_run(tmp_path, runtime={"target": "local"})
    checkpoint.write_bytes(b"full-state-checkpoint")
    resume_run(tmp_path, runtime={"target": "local", "slurm_job_id": "new-job"})
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "running"
    assert status["progress"]["stage"] == "resuming"
    assert status["runtime"]["slurm_job_id"] == "new-job"
    assert not validate_run(tmp_path)


@pytest.mark.parametrize("legacy", [
    {"state": "running", "progress": {"stage": "checkpointed_pause"}},
    {"state": "failed", "error": {"type": "PlannedPause", "returncode": 75}},
])
def test_old_pauses_are_recognized_without_rewriting_evidence(legacy):
    observed = observe_status({**legacy, "heartbeat_at": "2026-01-01T00:00:00+00:00"})
    assert observed["observation"] == "paused"
    assert not observed["alerts"]


def test_fresh_stale_and_unknown_are_observations_not_mutations():
    now = datetime(2026, 9, 24, tzinfo=timezone.utc)
    status = {"state": "running", "heartbeat_at": (now - timedelta(minutes=46)).isoformat()}
    assert observe_status(status, now=now)["observation"] == "stale"
    assert status["state"] == "running"
    assert observe_status(status, now=now, stale_after_minutes=60)["observation"] == "running"
    assert observe_status({"state": "running"}, now=now)["observation"] == "unknown"


def test_refresh_excludes_staging_archives_and_labels_stale_records(tmp_path):
    root = tmp_path / "results"
    for folder, run_id in [("study/runs", "active"), ("study/.incoming", "staged"),
                           ("study/archive", "archived"), ("study/stopped-production", "stopped")]:
        run = root / folder / run_id
        start_run(run, _manifest(run_id=run_id))
        status = json.loads((run / "status.json").read_text())
        status["heartbeat_at"] = "2026-01-01T00:00:00+00:00"
        atomic_write_json(run / "status.json", status)
    document = tmp_path / "dashboard.md"
    document.write_text(f"Manual prefix\n{ACTIVE_BEGIN}\nold\n{ACTIVE_END}\nManual suffix\n")
    refresh_active_document(root, document)
    text = document.read_text()
    assert "`active`" in text and "| stale |" in text
    assert all(f"`{name}`" not in text for name in ("staged", "archived", "stopped"))
    assert text.startswith("Manual prefix\n") and text.endswith("Manual suffix\n")


def test_inspect_reports_missing_failed_and_incomplete_terminal_evidence(tmp_path):
    missing = tmp_path / "missing"
    failed = tmp_path / "failed"
    complete = tmp_path / "complete"
    start_run(failed, _manifest(run_id="failed"))
    fail_run(failed, error="OOM")
    start_run(complete, _manifest(run_id="complete"))
    complete_run(complete, terminal_metrics={})
    (complete / "result.json").unlink()
    rows = inspect_runs([missing, failed, complete])
    assert [r["observation"] for r in rows] == ["unknown", "failed", "complete"]
    assert all(r["alerts"] for r in rows)
    assert rows[1]["error"]["message"] == "OOM"


def test_inspect_cli_returns_alerts_without_crashing_on_bad_status(tmp_path, capsys):
    from experiments.reporting import main
    start_run(tmp_path, _manifest())
    status_path = tmp_path / "status.json"
    status = json.loads(status_path.read_text())
    status["progress"] = None
    atomic_write_json(status_path, status)
    assert main(["inspect-runs", str(tmp_path)]) == 1
    row = json.loads(capsys.readouterr().out)["runs"][0]
    assert row["observation"] == "unknown" and row["alerts"] == ["invalid status progress"]


def test_heartbeats_do_not_write_the_shared_dashboard(tmp_path):
    run = tmp_path / "results/study/run"
    document = tmp_path / "docs/current_simulations.md"
    document.parent.mkdir()
    # Deliberately not a valid dashboard: runner transitions must not even read it.
    document.write_text("Manual work in progress\n")
    start_run(run, _manifest())
    update_status_progress(run, {"stage": "training", "epoch": 1})
    complete_run(run, terminal_metrics={})
    assert document.read_text() == "Manual work in progress\n"
