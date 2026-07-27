from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import campaigns.runner as runner
import pytest
from campaigns.schema import CampaignSpec, TargetSpec


def _target(tmp_path: Path) -> dict:
    return {
        "id": "target",
        "worktree": str(tmp_path / "worktree"),
        "python": str(tmp_path / "python"),
    }


def _preflight(_target, *, allow_dirty):
    return {
        "source": {
            "commit": "abc123",
            "dirty": bool(allow_dirty),
            "dirty_hash": "dirty" if allow_dirty else None,
        },
        "description": {
            "protocol_version": 1,
            "capabilities": {
                "supported_commands": [
                    "train",
                    "linspace",
                    "validate",
                ],
                "resume": {
                    "weights": True,
                    "base_weights": True,
                    "full_training_state": True,
                },
            },
            "commands": {
                "train": {
                    "available": True,
                    "required_options": ["--config", "--output-dir"],
                    "exclusive_input_options": [
                        "--weights",
                        "--base-weights",
                        "--resume",
                    ],
                },
                "linspace": {
                    "available": True,
                    "required_options": [
                        "--config",
                        "--output-dir",
                        "--weights",
                    ],
                },
                "validate": {
                    "available": True,
                    "required_options": [
                        "--config",
                        "--output-dir",
                        "--weights",
                    ],
                },
            },
        },
    }


def test_dry_run_plans_stage_references_without_launching(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "train.json").write_text("{}")
    (tmp_path / "validate.json").write_text("{}")
    manifest = {
        "schema_version": 1,
        "campaign_id": "dry",
        "targets": [_target(tmp_path)],
        "stages": [
            {
                "id": "train",
                "case_id": "digits",
                "target": "target",
                "command": "train",
                "config": "train.json",
            },
            {
                "id": "validate",
                "case_id": "digits",
                "target": "target",
                "command": "validate",
                "config": "validate.json",
                "depends_on": ["train"],
                "inputs": {
                    "weights": {
                        "stage": "train",
                        "artifact_kind": "weights",
                    }
                },
            },
        ],
    }
    spec = CampaignSpec.parse(manifest, base_dir=tmp_path)
    monkeypatch.setattr(runner, "_preflight", _preflight)

    records = runner.run_campaign(
        spec,
        output_root=tmp_path / "outputs",
        dry_run=True,
    )

    assert records["train"]["status"] == "dry_run"
    assert records["validate"]["status"] == "dry_run"
    assert "@stage:train:weights" in records["validate"]["command"]
    assert records["train"]["target_source"]["commit"] == "abc123"


def test_failure_skips_dependents_but_not_independent_stages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name in ("fail", "dependent", "independent"):
        (tmp_path / f"{name}.json").write_text("{}")
    manifest = {
        "schema_version": 1,
        "campaign_id": "continue",
        "targets": [_target(tmp_path)],
        "stages": [
            {
                "id": "fail",
                "case_id": "first",
                "target": "target",
                "command": "train",
                "config": "fail.json",
            },
            {
                "id": "dependent",
                "case_id": "first",
                "target": "target",
                "command": "validate",
                "config": "dependent.json",
                "depends_on": ["fail"],
                "inputs": {
                    "weights": {
                        "stage": "fail",
                        "artifact_kind": "weights",
                    }
                },
            },
            {
                "id": "independent",
                "case_id": "second",
                "target": "target",
                "command": "train",
                "config": "independent.json",
            },
        ],
    }
    spec = CampaignSpec.parse(manifest, base_dir=tmp_path)
    monkeypatch.setattr(runner, "_preflight", _preflight)

    def fake_run(command, **_kwargs):
        if any(str(argument).endswith("/fail.json") for argument in command):
            return SimpleNamespace(
                returncode=7,
                stdout="",
                stderr="intentional failure",
            )
        output_dir = Path(command[command.index("--output-dir") + 1])
        run_dir = output_dir / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "artifacts": [],
                }
            )
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    records = runner.run_campaign(
        spec,
        output_root=tmp_path / "outputs",
    )

    assert records["fail"]["status"] == "failed"
    assert records["dependent"]["status"] == "skipped_dependency"
    assert records["independent"]["status"] == "complete"


def test_preflight_rejects_unadvertised_stage_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "train.json").write_text("{}")
    manifest = {
        "schema_version": 1,
        "campaign_id": "unsupported",
        "targets": [_target(tmp_path)],
        "stages": [
            {
                "id": "train",
                "case_id": "case",
                "target": "target",
                "command": "train",
                "config": "train.json",
            }
        ],
    }
    spec = CampaignSpec.parse(manifest, base_dir=tmp_path)

    def unsupported(_target, *, allow_dirty):
        result = _preflight(_target, allow_dirty=allow_dirty)
        result["description"]["capabilities"]["supported_commands"] = [
            "validate"
        ]
        return result

    monkeypatch.setattr(runner, "_preflight", unsupported)
    with pytest.raises(RuntimeError, match="advertised by its target"):
        runner.run_campaign(
            spec,
            output_root=tmp_path / "outputs",
            dry_run=True,
        )


def test_preflight_rejects_advertised_but_unavailable_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "train.json").write_text("{}")
    spec = CampaignSpec.parse(
        {
            "schema_version": 1,
            "campaign_id": "unavailable",
            "targets": [_target(tmp_path)],
            "stages": [
                {
                    "id": "train",
                    "case_id": "case",
                    "target": "target",
                    "command": "train",
                    "config": "train.json",
                }
            ],
        },
        base_dir=tmp_path,
    )

    def unavailable(_target, *, allow_dirty):
        result = _preflight(_target, allow_dirty=allow_dirty)
        result["description"]["commands"]["train"]["available"] = False
        return result

    monkeypatch.setattr(runner, "_preflight", unavailable)
    with pytest.raises(RuntimeError, match="available on its target"):
        runner.run_campaign(
            spec,
            output_root=tmp_path / "outputs",
            dry_run=True,
        )


def test_base_weights_remains_an_explicit_hyphenated_stage_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "train.json").write_text("{}")
    (tmp_path / "base.pt").write_bytes(b"weights")
    manifest = {
        "schema_version": 1,
        "campaign_id": "base-weights",
        "targets": [_target(tmp_path)],
        "stages": [
            {
                "id": "train",
                "case_id": "case",
                "target": "target",
                "command": "train",
                "config": "train.json",
                "inputs": {"base_weights": {"path": "base.pt"}},
            }
        ],
    }
    spec = CampaignSpec.parse(manifest, base_dir=tmp_path)
    monkeypatch.setattr(runner, "_preflight", _preflight)

    records = runner.run_campaign(
        spec,
        output_root=tmp_path / "outputs",
        dry_run=True,
    )

    command = records["train"]["command"]
    assert "--base-weights" in command
    assert "--base_weights" not in command
    assert str((tmp_path / "base.pt").resolve()) in command


def test_fail_fast_marks_later_independent_stage_without_launching(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name in ("fail", "later"):
        (tmp_path / f"{name}.json").write_text("{}")
    manifest = {
        "schema_version": 1,
        "campaign_id": "fail-fast",
        "targets": [_target(tmp_path)],
        "stages": [
            {
                "id": "fail",
                "case_id": "first",
                "target": "target",
                "command": "train",
                "config": "fail.json",
            },
            {
                "id": "later",
                "case_id": "second",
                "target": "target",
                "command": "train",
                "config": "later.json",
            },
        ],
    }
    spec = CampaignSpec.parse(manifest, base_dir=tmp_path)
    monkeypatch.setattr(runner, "_preflight", _preflight)
    launches = []

    def fail(command, **_kwargs):
        launches.append(command)
        return SimpleNamespace(returncode=3, stdout="", stderr="failed")

    monkeypatch.setattr(runner.subprocess, "run", fail)
    records = runner.run_campaign(
        spec,
        output_root=tmp_path / "outputs",
        fail_fast=True,
    )

    assert records["fail"]["status"] == "failed"
    assert records["later"]["status"] == "skipped_fail_fast"
    assert len(launches) == 1


def test_resume_reuses_only_a_matching_complete_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "train.json").write_text("{}")
    manifest = {
        "schema_version": 1,
        "campaign_id": "resume",
        "targets": [_target(tmp_path)],
        "stages": [
            {
                "id": "train",
                "case_id": "case",
                "target": "target",
                "command": "train",
                "config": "train.json",
            }
        ],
    }
    spec = CampaignSpec.parse(manifest, base_dir=tmp_path)
    monkeypatch.setattr(runner, "_preflight", _preflight)
    launches = []

    def complete(command, **_kwargs):
        launches.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        run_dir = output_dir / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "result.json").write_text(
            json.dumps({"status": "complete", "artifacts": []})
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", complete)
    first = runner.run_campaign(
        spec,
        output_root=tmp_path / "outputs",
    )
    resumed = runner.run_campaign(
        spec,
        output_root=tmp_path / "outputs",
        resume=True,
    )

    assert first["train"]["status"] == "complete"
    assert resumed["train"]["status"] == "complete"
    assert resumed["train"]["fingerprint"] == first["train"]["fingerprint"]
    assert len(launches) == 1


def test_dirty_identity_hashes_untracked_source_contents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "new_module.py"
    source.write_text("VALUE = 1\n")
    target = TargetSpec(
        target_id="target",
        worktree=tmp_path,
        python=tmp_path / "python",
    )

    def check_output(command, **_kwargs):
        if command[-2:] == ("rev-parse", "HEAD"):
            return b"abc123\n"
        if "status" in command:
            return b"?? new_module.py\0"
        if "diff" in command:
            return b""
        raise AssertionError(command)

    monkeypatch.setattr(runner.subprocess, "check_output", check_output)
    first = runner._git_identity(target)
    source.write_text("VALUE = 2\n")
    second = runner._git_identity(target)

    assert first["dirty"] is True
    assert first["dirty_hash"] != second["dirty_hash"]


def test_preflight_requires_target_python_to_be_executable(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o644)
    target = TargetSpec(
        target_id="target",
        worktree=worktree,
        python=python,
    )

    with pytest.raises(ValueError, match="existing executable"):
        runner._preflight(target, allow_dirty=False)
