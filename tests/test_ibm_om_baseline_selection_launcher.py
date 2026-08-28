from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.mnist_relu_drn import ibm_om_baseline_selection_launcher as launcher
from experiments.study_workflow import prepare_study


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_task_matrix_and_native_commands_are_exact(tmp_path: Path) -> None:
    tasks = launcher.tasks()

    assert len(tasks) == 12
    assert len({(task.arm_id, task.heldout_seed) for task in tasks}) == 12
    assert {task.heldout_seed for task in tasks} == {87001, 87002, 87003}
    assert [task.arm_id for task in tasks[::3]] == [
        arm_id for arm_id, _policy in launcher.ARM_POLICIES
    ]
    assert all(not task.config.name.startswith("smoke-") for task in tasks)

    command = launcher._command(tasks[0], study_dir=tmp_path)
    assert command[:4] == [
        "/home/filip/miniconda3/envs/py312/bin/python",
        "-m",
        "ebl",
        "validate",
    ]
    assert command[-2:] == ["--weights", str(launcher.TEACHER.resolve())]
    assert command[command.index("--output-dir") + 1] == str(
        (tmp_path / "runs" / tasks[0].arm_id).resolve()
    )


def test_prepared_study_has_exact_four_by_three_config_coverage(
    tmp_path: Path,
) -> None:
    study_dir = prepare_study(launcher.PLAN, tmp_path)

    hashes = launcher._validate_prepared_study(study_dir)

    assert set(hashes) == {task.label for task in launcher.tasks()}
    assert len(set(hashes.values())) == 12


def test_task_environment_pins_aihwkit_and_unbuffered_execution() -> None:
    environment = launcher._task_environment({"EXISTING": "kept"})

    assert environment == {
        "EXISTING": "kept",
        "EBL_AIHWKIT_PYTHON": "/home/filip/miniconda3/envs/aihwkit/bin/python",
        "EBL_DEFER_CURRENT_SIMULATIONS": "1",
        "PYTHONUNBUFFERED": "1",
    }


def test_prerequisite_probe_requires_and_records_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    task_python = tmp_path / "task-python"
    aihwkit_python = tmp_path / "aihwkit-python"
    task_python.write_text("", encoding="utf-8")
    aihwkit_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(launcher, "TEACHER", teacher)
    monkeypatch.setattr(launcher, "EXPECTED_TEACHER_SHA256", launcher.sha256_file(teacher))
    monkeypatch.setattr(launcher, "TASK_PYTHON", task_python)
    monkeypatch.setattr(launcher, "AIHWKIT_PYTHON", aihwkit_python)
    monkeypatch.setattr(
        launcher,
        "_require_executable",
        lambda path, *, label: path.resolve(),
    )
    responses = iter(
        (
            SimpleNamespace(returncode=0, stdout="1.1.0\n", stderr=""),
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "torch_version": "2.5.1+cu121",
                        "cuda_available": True,
                        "cuda_device_count": 1,
                        "cuda_device_name": "NVIDIA RTX 3090",
                    }
                ),
                stderr="",
            ),
        )
    )
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: next(responses))

    result = launcher._validate_prerequisites()

    assert result["cuda"]["cuda_available"] is True
    assert result["cuda"]["cuda_device_name"] == "NVIDIA RTX 3090"


def test_prerequisite_probe_rejects_cpu_only_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    task_python = tmp_path / "task-python"
    aihwkit_python = tmp_path / "aihwkit-python"
    task_python.write_text("", encoding="utf-8")
    aihwkit_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(launcher, "TEACHER", teacher)
    monkeypatch.setattr(launcher, "EXPECTED_TEACHER_SHA256", launcher.sha256_file(teacher))
    monkeypatch.setattr(launcher, "TASK_PYTHON", task_python)
    monkeypatch.setattr(launcher, "AIHWKIT_PYTHON", aihwkit_python)
    monkeypatch.setattr(
        launcher,
        "_require_executable",
        lambda path, *, label: path.resolve(),
    )
    responses = iter(
        (
            SimpleNamespace(returncode=0, stdout="1.1.0\n", stderr=""),
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "torch_version": "2.5.1+cpu",
                        "cuda_available": False,
                        "cuda_device_count": 0,
                        "cuda_device_name": None,
                    }
                ),
                stderr="",
            ),
        )
    )
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: next(responses))

    with pytest.raises(RuntimeError, match="CUDA GPU"):
        launcher._validate_prerequisites()


def _native_bundle(
    arm_root: Path,
    *,
    config_sha256: str,
    source_commit: str,
    status: str = "complete",
) -> Path:
    run_dir = arm_root / "run-001"
    artifact = run_dir / "artifacts" / "payload.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    _write_json(
        run_dir / "manifest.json",
        {
            "experiment_id": launcher.EXPERIMENT_ID,
            "source": {
                "commit": source_commit,
                "dirty": False,
            },
            "study": {
                "study_id": launcher.STUDY_ID,
                "arm_id": arm_root.name,
                "source_config_sha256": config_sha256,
            },
        },
    )
    _write_json(run_dir / "status.json", {"status": status})
    if status == "complete":
        _write_json(
            run_dir / "result.json",
            {
                "run_id": run_dir.name,
                "status": "complete",
                "artifacts": [
                    {
                        "path": "artifacts/payload.bin",
                        "sha256": digest,
                        "size_bytes": len(b"payload"),
                    }
                ],
            },
        )
    return run_dir


def test_exact_completed_skip_requires_commit_and_intact_artifacts(
    tmp_path: Path,
) -> None:
    config_sha = "a" * 64
    run_dir = _native_bundle(
        tmp_path,
        config_sha256=config_sha,
        source_commit="launch-commit",
    )

    assert launcher._matching_run(
        tmp_path,
        config_sha256=config_sha,
        source_commit="launch-commit",
        require_complete=True,
    ) == run_dir
    assert (
        launcher._matching_run(
            tmp_path,
            config_sha256=config_sha,
            source_commit="other-commit",
            require_complete=True,
        )
        is None
    )

    (run_dir / "artifacts" / "payload.bin").write_bytes(b"tampered")
    assert (
        launcher._matching_run(
            tmp_path,
            config_sha256=config_sha,
            source_commit="launch-commit",
            require_complete=True,
        )
        is None
    )


def test_running_attempt_is_detected_before_duplicate_launch(tmp_path: Path) -> None:
    config_sha = "b" * 64
    run_dir = _native_bundle(
        tmp_path,
        config_sha256=config_sha,
        source_commit="launch-commit",
        status="running",
    )

    assert launcher._running_run(tmp_path, config_sha256=config_sha) == run_dir
