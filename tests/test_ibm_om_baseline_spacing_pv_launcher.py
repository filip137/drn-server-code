from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.mnist_relu_drn import (
    ibm_om_baseline_spacing_pv_launcher as launcher,
)
from experiments.study_workflow import prepare_study


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_task_matrix_and_native_commands_are_exact(tmp_path: Path) -> None:
    tasks = launcher.tasks()

    assert len(launcher.POLICIES) == 9
    assert len(tasks) == 27
    assert len({(task.arm_id, task.heldout_seed) for task in tasks}) == 27
    assert {task.heldout_seed for task in tasks} == {87001, 87002, 87003}
    assert [task.arm_id for task in tasks[::3]] == [
        policy.arm_id for policy in launcher.POLICIES
    ]
    assert all(not task.config.name.startswith("smoke-") for task in tasks)
    assert {
        tuple(launcher.ENDPOINT_SEEDS_BY_ASSIGNMENT[task.heldout_seed])
        for task in tasks
        if task.heldout_seed == 87001
    } == {(89101, 89102, 89103, 89104, 89105)}

    command = launcher._task_command(tasks[0], study_dir=tmp_path)
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


def test_strict_configs_cover_nine_by_three_and_excluded_smoke() -> None:
    for task in launcher.tasks():
        digest = launcher._validate_task_config(task)
        assert len(digest) == 64

    smoke_digest = launcher._validate_smoke_config()
    assert len(smoke_digest) == 64
    assert launcher.SMOKE_CONFIG.name.startswith("smoke-")


def test_prepared_study_has_exact_nine_by_three_config_coverage(
    tmp_path: Path,
) -> None:
    study_dir = prepare_study(launcher.PLAN, tmp_path)

    hashes = launcher._validate_prepared_study(study_dir)

    assert set(hashes) == {
        *(task.label for task in launcher.tasks()),
        "smoke-canary",
    }
    assert len(hashes) == 28
    assert len({hashes[task.label] for task in launcher.tasks()}) == 27


def test_task_environment_pins_aihwkit_and_unbuffered_execution() -> None:
    environment = launcher._task_environment({"EXISTING": "kept"})

    assert environment == {
        "EXISTING": "kept",
        "EBL_AIHWKIT_PYTHON": "/home/filip/miniconda3/envs/aihwkit/bin/python",
        "EBL_DEFER_CURRENT_SIMULATIONS": "1",
        "PYTHONUNBUFFERED": "1",
    }


@pytest.mark.parametrize("cuda_available", [True, False])
def test_prerequisite_probe_requires_and_records_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
) -> None:
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    task_python = tmp_path / "task-python"
    aihwkit_python = tmp_path / "aihwkit-python"
    task_python.write_text("", encoding="utf-8")
    aihwkit_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(launcher, "TEACHER", teacher)
    monkeypatch.setattr(
        launcher,
        "EXPECTED_WEIGHTS_SHA256",
        launcher.sha256_file(teacher),
    )
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
                        "torch_version": (
                            "2.5.1+cu121" if cuda_available else "2.5.1+cpu"
                        ),
                        "cuda_available": cuda_available,
                        "cuda_device_count": 1 if cuda_available else 0,
                        "cuda_device_name": (
                            "NVIDIA RTX 3090" if cuda_available else None
                        ),
                    }
                ),
                stderr="",
            ),
        )
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: next(responses),
    )

    if not cuda_available:
        with pytest.raises(RuntimeError, match="CUDA GPU"):
            launcher._validate_prerequisites()
        return
    result = launcher._validate_prerequisites()
    assert result["cuda"]["cuda_available"] is True
    assert result["cuda"]["cuda_device_name"] == "NVIDIA RTX 3090"


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
            "source": {"commit": source_commit, "dirty": False},
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
                        "kind": "test_payload",
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


def test_custom_analyzer_is_a_required_launcher_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_dir = tmp_path / "study"
    attempt_dir = study_dir / "launch" / "attempt"
    attempt_dir.mkdir(parents=True)

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        output_dir = Path(command[command.index("--output-dir") + 1])
        _write_json(
            output_dir / launcher.ANALYSIS_RESULT,
            {
                "schema": "ebl.mnist_relu_drn.ibm_om_baseline_spacing_pv_analysis",
                "schema_version": 1,
                "study_id": launcher.STUDY_ID,
                "audit": {
                    "complete_run_count": 27,
                    "registered_artifacts_verified": True,
                    "matched_hardware_and_rng_streams": True,
                },
            },
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    result = launcher._run_custom_analysis(
        study_dir=study_dir,
        attempt_dir=attempt_dir,
        environment={},
    )

    assert result["valid"] is True
    assert launcher.ANALYZER_MODULE in result["command"]
    assert (attempt_dir / "custom-analysis.result.json").is_file()
