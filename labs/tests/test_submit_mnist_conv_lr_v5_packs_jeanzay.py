from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

import experiments.submit_mnist_conv_lr_v5_packs_jeanzay as submission
from experiments.mnist_conv.identity import sha256_file
from experiments.mnist_conv.io import atomic_write_json


REPO_ROOT = Path(__file__).resolve().parents[2]


def _fixture(tmp_path: Path) -> argparse.Namespace:
    study_id = "lrstudy_" + "a" * 64
    study = tmp_path / ("v5-study--" + study_id)
    candidate_manifest = study / "stages/candidates/manifest.json"
    candidate_manifest.parent.mkdir(parents=True)
    atomic_write_json(
        candidate_manifest,
        {
            "schema_version": "mnist-conv-lr-stage-manifest/v1",
            "study_id": study_id,
            "stage_name": "candidates",
            "entries": [{"entry_index": 0}],
        },
        canonical=True,
    )
    pack_manifest = study / "stages/candidates/pack_manifest.json"
    atomic_write_json(
        pack_manifest,
        {
            "schema_version": "mnist-conv-lr-pack-manifest/v1",
            "study_id": study_id,
            "stage_manifest_sha256": sha256_file(candidate_manifest),
            "packs": [{"pack_index": 0, "entry_indices": [0]}],
        },
        canonical=True,
    )
    data_root = tmp_path / "mnist"
    data_root.mkdir()
    return argparse.Namespace(
        study=str(study),
        candidate_manifest=str(candidate_manifest),
        pack_manifest=str(pack_manifest),
        repo_root=str(REPO_ROOT),
        python="python",
        data_root=str(data_root),
        module="pytorch-gpu/py3/2.5.0",
        max_concurrent_packs=6,
        dry_run=False,
    )


def _completed(command: list[str], job_id: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout=f"{job_id}\n", stderr="")


def test_commands_reserve_slurm_for_gpu_training_and_collection_for_login_node(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    worker, login_finalizer = submission._commands(args)

    assert worker[0] == "sbatch"
    assert "--account=fmu@v100" in worker
    assert "--constraint=v100-32g" in worker
    assert "--gres=gpu:1" in worker
    assert not any("fmu@cpu" in item for item in worker)
    assert login_finalizer["execution_host"] == "jean_zay_login_node"
    assert login_finalizer["run_after"] == "worker_array_terminal"
    assert login_finalizer["module"] == "pytorch-gpu/py3/2.5.0"
    command = login_finalizer["command"]
    assert command[0] == "python"
    assert "--finalize-stage" in command
    assert command[command.index("--device") + 1] == "cpu"
    assert "sbatch" not in command


def test_submission_runs_exactly_one_sbatch_and_returns_login_finalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs == {"check": True, "text": True, "capture_output": True}
        return _completed(command, "2121299")

    monkeypatch.setattr(submission.subprocess, "run", fake_run)
    result = submission.submit(args)

    assert len(calls) == 1
    assert calls[0][0] == "sbatch"
    assert result["status"] == "complete"
    assert result["worker_submitted"] is True
    assert result["worker_job_id"] == "2121299"
    assert result["login_finalizer"]["execution_host"] == "jean_zay_login_node"


def test_dry_run_submits_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _fixture(tmp_path)
    args.dry_run = True
    monkeypatch.setattr(
        submission.subprocess,
        "run",
        lambda *unused_args, **unused_kwargs: pytest.fail("sbatch must not run"),
    )

    result = submission.submit(args)

    assert result["status"] == "dry_run"
    assert result["submitted"] is False
    assert result["worker"][0] == "sbatch"
    assert "--finalize-stage" in result["login_finalizer"]["command"]


def test_worker_failure_never_attempts_another_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        raise subprocess.CalledProcessError(
            1, command, output="", stderr="sbatch: rejected"
        )

    monkeypatch.setattr(submission.subprocess, "run", fake_run)
    result = submission.submit(args)

    assert len(calls) == 1
    assert result["status"] == "failed"
    assert result["worker_submitted"] is False
    assert result["worker_error"]["stderr"] == "sbatch: rejected"


def test_ambiguous_worker_submission_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path)
    monkeypatch.setattr(
        submission.subprocess,
        "run",
        lambda command, **unused_kwargs: _completed(command, "not-a-job-id"),
    )

    result = submission.submit(args)

    assert result["status"] == "submission_unconfirmed"
    assert result["worker_submitted"] is None


def _required_argv(tmp_path: Path) -> list[str]:
    return [
        "--study",
        str(tmp_path / "study"),
        "--candidate-manifest",
        str(tmp_path / "candidates.json"),
        "--pack-manifest",
        str(tmp_path / "packs.json"),
        "--python",
        "python",
        "--data-root",
        str(tmp_path / "mnist"),
    ]


def test_cli_failed_result_is_json_and_has_failure_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    expected = {
        "mode": "training_array_with_login_collection",
        "status": "failed",
        "submitted": False,
    }
    monkeypatch.setattr(submission, "submit", lambda unused_args: expected)

    assert submission.main(_required_argv(tmp_path)) == 2
    assert json.loads(capsys.readouterr().out) == expected
