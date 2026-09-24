import json
import hashlib

import pytest

from experiments.analyze_conv3_p90_read_noise import epoch_rows
from experiments import analyze_conv3_p90_read_noise as collector


def test_collection_keeps_full_thirty_epoch_horizon(tmp_path):
    rows = [{"kind": "epoch", "epoch": epoch} for epoch in range(1, 31)]
    (tmp_path / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
    assert [row["epoch"] for row in epoch_rows(tmp_path)] == list(range(1, 31))


def test_collection_before_first_epoch(tmp_path):
    assert epoch_rows(tmp_path) == []


@pytest.mark.parametrize("state,stage,epoch_count,accepted", [
    ("running", "starting", 0, True),
    ("failed", "starting", 0, True),
    ("complete", "complete", 0, False),
    ("running", "training", 1, False),
])
@pytest.mark.parametrize("layout", ["chunked", "single"])
def test_missing_resolved_config_is_only_allowed_at_startup(
        tmp_path, monkeypatch, state, stage, epoch_count, accepted, layout):
    config = tmp_path / "case.json"
    config.write_text(json.dumps({"lab": {"epochs": 30}}))
    monkeypatch.setattr(collector, "ROOT", tmp_path)
    monkeypatch.setattr(collector, "STUDY", tmp_path)
    # Canonical schema validation is independent of the collection-time race.
    monkeypatch.setattr(collector, "validate_run", lambda run: [])
    task = "local-acceleration/fifi/production" if layout == "single" else "jz/v100-production/task_0"
    run = tmp_path / task / "runs/000_case_12345678"
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"state": state, "progress": {"stage": stage}}))
    (run / "metrics.jsonl").write_text("".join(
        json.dumps({"kind": "epoch", "epoch": e}) + "\n"
        for e in range(1, epoch_count + 1)))
    case = dict(config="case.json", config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                epochs=30, production_root="v100-production", index=0, case="case")
    if layout == "single":
        case.update(layout="single", local_relative_root=task)
    if accepted:
        row = collector.collect_case(case)
        assert row["state"] == state and row["epochs_completed"] == 0
        assert "startup" in row["collection_note"]
        if state == "failed":
            assert row["failure"]["state"] == "failed"
    else:
        with pytest.raises(AssertionError):
            collector.collect_case(case)
