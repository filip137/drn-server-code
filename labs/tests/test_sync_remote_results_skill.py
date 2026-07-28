from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "skills"
    / "sync-remote-results"
    / "scripts"
    / "sync_remote_results.py"
)
SKILL = REPO_ROOT / "skills" / "sync-remote-results" / "SKILL.md"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sync_remote_results", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_cli(tmp_path: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--host",
        "akib",
        "--remote-path",
        "/home/filiposana/results/example-study/shards/akib",
        "--local-stage",
        str(tmp_path / "results/.incoming/example-study/shards/akib"),
        "--receipt",
        str(
            tmp_path
            / "results/_transfer_receipts/example-study/akib-metadata.json"
        ),
        "--mode",
        "metadata",
        "--completion-marker",
        "stages/rho_core_candidates/complete.json",
    ]


def test_plan_is_side_effect_free_and_metadata_skips_step_logs(
    tmp_path: Path,
) -> None:
    command = [*_base_cli(tmp_path), "--plan-only"]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    plan = json.loads(completed.stdout)

    assert plan["status"] == "planned"
    assert plan["ssh_target"] == "akib"
    assert plan["completion_state"] == "markers_required"
    sync = plan["commands"]["sync"]
    verify = plan["commands"]["verify"]
    assert "--exclude=**/step_log.csv" in sync
    assert "--include=*.csv" in sync
    assert sync.index("--exclude=**/step_log.csv") < sync.index(
        "--include=*.csv"
    )
    assert "--checksum" not in sync
    assert "--dry-run" in verify
    assert "--checksum" in verify
    completion_check = plan["commands"]["completion_checks"][0]
    assert completion_check[:4] == ["ssh", "-o", "BatchMode=yes", "akib"]
    assert completion_check[4].startswith("test -f /")
    assert "test -f --" not in completion_check[4]
    assert not (tmp_path / "results").exists()


def test_incomplete_copy_can_still_require_last_valid_marker(
    tmp_path: Path,
) -> None:
    command = [*_base_cli(tmp_path), "--allow-incomplete", "--plan-only"]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    plan = json.loads(completed.stdout)

    assert plan["completion_state"] == "incomplete_with_valid_markers"
    assert plan["commands"]["completion_checks"]


def test_requires_marker_or_incomplete_classification(tmp_path: Path) -> None:
    command = _base_cli(tmp_path)
    marker_index = command.index("--completion-marker")
    del command[marker_index : marker_index + 2]
    command.append("--plan-only")
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True
    )

    assert completed.returncode == 2
    assert "provide at least one marker or be paired" in completed.stderr


@pytest.mark.parametrize(
    ("field_args", "expected"),
    (
        (
            [
                "--remote-path",
                "/home/filiposana/results",
            ],
            "name one run, study, or shard",
        ),
        (
            [
                "--local-stage",
                "/tmp/final-result",
            ],
            "inside a '.incoming' staging directory",
        ),
    ),
)
def test_rejects_broad_or_final_paths(
    tmp_path: Path, field_args: list[str], expected: str
) -> None:
    command = _base_cli(tmp_path)
    field = field_args[0]
    index = command.index(field)
    command[index : index + 2] = field_args
    command.append("--plan-only")
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True
    )

    assert completed.returncode == 2
    assert f"Expected {field}" in completed.stderr
    assert expected in completed.stderr
    assert "Provided value:" in completed.stderr


def test_success_writes_a_content_hashed_transfer_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    stage = tmp_path / "results/.incoming/example-study/shards/akib"
    stage.mkdir(parents=True)
    (stage / "result.json").write_text('{"accuracy": 0.9}\n')
    (stage / "best_validation.pt").write_bytes(b"pre-existing full artifact")
    receipt = (
        tmp_path
        / "results/_transfer_receipts/example-study/akib-metadata.json"
    )
    calls: list[list[str]] = []

    def fake_run(
        command: list[str] | tuple[str, ...], *, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command, 0, stdout="" if capture_output else None, stderr=""
        )

    monkeypatch.setattr(module, "_run_command", fake_run)
    args = module._parser().parse_args(_base_cli(tmp_path)[2:])
    result = module.execute(args)

    assert result["status"] == "synced"
    assert result["verification"]["method"] == "rsync_checksum_dry_run"
    assert result["verification"]["scientific_validation"] == "pending"
    assert result["local_inventory"]["file_count"] == 1
    assert result["local_inventory"]["content_hashed"] is True
    assert len(result["local_inventory"]["inventory_sha256"]) == 64
    assert json.loads(receipt.read_text()) == result
    assert calls[0][0] == "ssh"
    assert calls[1][0] == "rsync"
    assert "--dry-run" in calls[2]


def test_verification_change_fails_without_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    stage = tmp_path / "results/.incoming/example-study/shards/akib"
    stage.mkdir(parents=True)
    receipt = (
        tmp_path
        / "results/_transfer_receipts/example-study/akib-metadata.json"
    )

    def fake_run(
        command: list[str] | tuple[str, ...], *, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        output = ">f+++++++++ missing.json\n" if capture_output else None
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(module, "_run_command", fake_run)
    args = module._parser().parse_args(_base_cli(tmp_path)[2:])
    with pytest.raises(RuntimeError, match="no remaining changes"):
        module.execute(args)

    assert not receipt.exists()


def test_skill_has_no_initializer_placeholders() -> None:
    text = SKILL.read_text()
    assert "TODO" not in text
    assert "sync_remote_results.py" in text
