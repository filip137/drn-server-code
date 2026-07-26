from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import experiments.supervise_mnist_conv_lr_v7_jeanzay as supervisor_module
from experiments.mnist_conv.io import atomic_write_json, read_json


REPO_ROOT = Path(__file__).resolve().parents[2]
V7_CONFIG = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "hardsigmoid_lr_conv3_scheme_two_rho_constant_sgd_bs16_v7.json"
)
V7_PROFILE = (
    REPO_ROOT
    / "configs"
    / "executors"
    / "jeanzay-v100-conv3-lr-v7.json"
)


def _allocation(**unused: object) -> dict[str, object]:
    return {
        "verified": True,
        "allocation_id": "AD010913993R3",
        "slurm_account": "fmu@v100",
    }


def _supervisor(
    tmp_path: Path,
    *,
    runner: object = subprocess.run,
) -> supervisor_module.V7JeanZaySupervisor:
    data = tmp_path / "mnist"
    data.mkdir(exist_ok=True)
    value = supervisor_module.V7JeanZaySupervisor(
        config_path=V7_CONFIG,
        profile_path=V7_PROFILE,
        results_root=tmp_path / "results",
        data_root=data,
        runner=runner,  # type: ignore[arg-type]
        allocation_verifier=_allocation,
    )
    value.initialize()
    return value


def test_supervisor_rejects_compute_node_context() -> None:
    with pytest.raises(RuntimeError, match="login node"):
        supervisor_module.require_login_node_context(
            {"SLURM_JOB_ID": "12345", "SLURM_JOB_NODELIST": "r8i2n5"}
        )


def test_array_sacct_parser_requires_parent_and_every_exact_task() -> None:
    def runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        assert command[:5] == ["sacct", "-n", "-X", "-j", "123"]
        assert "--format=JobIDRaw,JobID,State" in command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "123|123|COMPLETED|\n"
                "124|123_0|COMPLETED|\n"
                "125|123_1|COMPLETED|\n"
                "124.batch|123_0.batch|FAILED|\n"
                "123.extern|123.extern|FAILED|\n"
            ),
            stderr="",
        )

    result = supervisor_module.query_array_worker_state(
        "123", 2, runner=runner
    )
    assert result["status"] == "complete"
    assert result["parent_state"] == "COMPLETED"
    assert result["task_states"] == {"0": "COMPLETED", "1": "COMPLETED"}
    assert result["task_raw_ids"] == {"0": "124", "1": "125"}
    assert result["completion_basis"] == "explicit_parent_and_all_indexed_tasks"


def test_array_sacct_parser_accepts_jeanzay_singleton_parent_task() -> None:
    def runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        assert command[:5] == ["sacct", "-n", "-X", "-j", "206197"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="206197|206197_0|COMPLETED|\n",
            stderr="",
        )

    result = supervisor_module.query_array_worker_state(
        "206197", 1, runner=runner
    )

    assert result["status"] == "complete"
    assert result["task_states"] == {"0": "COMPLETED"}
    assert result["observed_task_count"] == 1
    assert result["missing_task_indices"] == []
    assert result["singleton_task_recorded_as_parent"] is True
    assert result["completion_basis"] == "all_indexed_tasks_without_parent_row"


def test_array_sacct_parser_accepts_real_jeanzay_multi_task_mapping() -> None:
    def runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "206752|206740_0|COMPLETED|\n"
                "206758|206740_1|COMPLETED|\n"
                "206740|206740_2|COMPLETED|\n"
            ),
            stderr="",
        )

    result = supervisor_module.query_array_worker_state(
        "206740", 3, runner=runner
    )

    assert result["status"] == "complete"
    assert result["parent_state"] == "COMPLETED"
    assert result["task_states"] == {
        "0": "COMPLETED",
        "1": "COMPLETED",
        "2": "COMPLETED",
    }
    assert result["task_raw_ids"] == {
        "0": "206752",
        "1": "206758",
        "2": "206740",
    }
    assert result["completion_basis"] == (
        "all_indexed_tasks_without_parent_row"
    )


def test_array_sacct_parser_waits_for_active_missing_tasks_but_fails_closed_at_terminal() -> None:
    def pending_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="123|123_[0-2]|PENDING|\n",
            stderr="",
        )

    waiting = supervisor_module.query_array_worker_state(
        "123", 3, runner=pending_runner
    )
    assert waiting["status"] == "waiting"
    assert waiting["missing_task_indices"] == [0, 1, 2]
    assert waiting["compressed_rows"] == ["123_[0-2]"]

    def incomplete_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "123|123|COMPLETED|\n"
                "124|123_0|COMPLETED|\n"
                "126|123_2|COMPLETED|\n"
            ),
            stderr="",
        )

    with pytest.raises(RuntimeError, match="missing indices"):
        supervisor_module.query_array_worker_state(
            "123", 3, runner=incomplete_runner
        )


def test_array_sacct_parser_rejects_duplicate_and_unexpected_jobids() -> None:
    def duplicate_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "124|123_0|COMPLETED|\n"
                "125|123_0|COMPLETED|\n"
            ),
            stderr="",
        )

    with pytest.raises(RuntimeError, match="duplicate task"):
        supervisor_module.query_array_worker_state(
            "123", 2, runner=duplicate_runner
        )

    def unexpected_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "124|123_0|COMPLETED|\n"
                "125|123_1|COMPLETED|\n"
                "126|123_2|COMPLETED|\n"
            ),
            stderr="",
        )

    with pytest.raises(RuntimeError, match="unexpected indices"):
        supervisor_module.query_array_worker_state(
            "123", 2, runner=unexpected_runner
        )


def test_array_sacct_parser_rejects_failed_or_mixed_terminal_tasks() -> None:
    def failed_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "123|123|RUNNING|\n"
                "124|123_0|FAILED|\n"
                "125|123_1|RUNNING|\n"
            ),
            stderr="",
        )

    with pytest.raises(RuntimeError, match="failed tasks"):
        supervisor_module.query_array_worker_state(
            "123", 2, runner=failed_runner
        )

    def mixed_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "123|123|COMPLETED|\n"
                "124|123_0|COMPLETED|\n"
                "125|123_1|RUNNING|\n"
            ),
            stderr="",
        )

    with pytest.raises(RuntimeError, match="mixed task states"):
        supervisor_module.query_array_worker_state(
            "123", 2, runner=mixed_runner
        )


def test_full_preflight_gate_rechecks_flags_counts_and_numeric_limits(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    valid = {
        "schema_version": "mnist-conv-lr-v7-preflight/v1",
        "status": "passed",
        "failure_reason": None,
        "device_is_v100": True,
        "device_capacity_passed": True,
        "memory_passed": True,
        "runtime_passed": True,
        "safety_gate_failure": None,
        "measured_steps": 256,
        "completed_measured_steps": 256,
        "validation_metrics": {"loss": 1.0, "accuracy": 0.1},
        "memory_headroom_fraction": 0.2,
        "required_memory_headroom_fraction": 0.1,
        "projected_candidate_hours": 9.0,
        "maximum_projected_candidate_hours": 20.0,
    }
    atomic_write_json(path, valid, canonical=True)
    assert supervisor_module.require_full_preflight_pass(path)["status"] == "passed"

    for field, value in (
        ("memory_passed", False),
        ("runtime_passed", False),
        ("device_capacity_passed", False),
        ("completed_measured_steps", 255),
        ("memory_headroom_fraction", 0.09),
        ("projected_candidate_hours", 20.1),
    ):
        broken = dict(valid)
        broken[field] = value
        atomic_write_json(path, broken, canonical=True)
        with pytest.raises(RuntimeError, match="preflight contract"):
            supervisor_module.require_full_preflight_pass(path)


def test_extension_no_op_recognition_is_exact() -> None:
    no_op = {
        "entries": [
            {
                "entry_id": "no-expansion-required",
                "payload": {
                    "no_op": True,
                    "reason": "no_scheme_requested_expansion",
                },
            }
        ]
    }
    assert supervisor_module.extension_is_explicit_no_op(no_op) is True
    assert (
        supervisor_module.extension_is_explicit_no_op(
            {"entries": [{"entry_id": "candidate", "payload": {"no_op": False}}]}
        )
        is False
    )
    with pytest.raises(RuntimeError, match="exact single"):
        supervisor_module.extension_is_explicit_no_op(
            {
                "entries": [
                    *no_op["entries"],
                    {"entry_id": "candidate", "payload": {"no_op": False}},
                ]
            }
        )


def test_audit_submission_persists_worker_only_and_resume_does_not_resubmit(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def first_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[0] == "sbatch":
            return subprocess.CompletedProcess(
                command, 0, stdout="4001\n", stderr=""
            )
        if command[0] == "sacct":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="4001|4001_[0]|PENDING|\n",
                stderr="",
            )
        raise AssertionError(command)

    first = _supervisor(tmp_path, runner=first_runner)
    result = first.advance_once()
    assert result == {
        "status": "waiting",
        "stage": "audit",
        "worker_job_id": "4001",
        "worker_parent_state": "PENDING",
        "completed_task_count": 0,
        "expected_task_count": 1,
        "missing_task_indices": [0],
    }
    state = read_json(first.state_path)
    audit = state["stages"]["audit"]
    assert audit["worker_job_id"] == "4001"
    assert (first.study_dir / "slurm" / "audit").is_dir()
    assert "finalizer_job_id" not in audit
    assert audit["finalizer_submission_status"] == "disabled"
    assert (
        audit["finalizer_mode"]
        == "login_node_local_after_array_validation"
    )
    assert sum(command[0] == "sbatch" for command in calls) == 1
    assert "--hint=nomultithread" in audit["worker_command"]
    joined = "\n".join(audit["worker_command"])
    assert "OMP_NUM_THREADS=1" in joined

    resumed_calls: list[list[str]] = []

    def resumed_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        resumed_calls.append(command)
        assert command[0] == "sacct"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="4001|4001_0|RUNNING|\n",
            stderr="",
        )

    resumed = _supervisor(tmp_path, runner=resumed_runner)
    resumed_result = resumed.advance_once()
    assert resumed_result["worker_parent_state"] == "RUNNING"
    assert not any(command[0] == "sbatch" for command in resumed_calls)


def test_failed_array_task_stops_without_submitting_a_later_stage(
    tmp_path: Path,
) -> None:
    def submit_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "sbatch":
            return subprocess.CompletedProcess(
                command, 0, stdout="5001\n", stderr=""
            )
        return subprocess.CompletedProcess(
            command, 0, stdout="5001|5001_[0]|PENDING|\n", stderr=""
        )

    first = _supervisor(tmp_path, runner=submit_runner)
    first.advance_once()
    observed: list[list[str]] = []

    def failed_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        observed.append(command)
        assert command[0] == "sacct"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="5001|5001_0|FAILED|\n",
            stderr="",
        )

    resumed = _supervisor(tmp_path, runner=failed_runner)
    with pytest.raises(RuntimeError, match="failed tasks"):
        resumed.advance_once()
    assert not any(command[0] == "sbatch" for command in observed)
    state = read_json(resumed.state_path)
    assert (
        state["stages"]["audit"]["status"]
        == "failed_worker_array_validation"
    )


def test_rejected_legacy_cpu_finalizer_is_preserved_and_ignored_on_resume(
    tmp_path: Path,
) -> None:
    def submit_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "sbatch":
            return subprocess.CompletedProcess(
                command, 0, stdout="162895\n", stderr=""
            )
        return subprocess.CompletedProcess(
            command, 0, stdout="162895|162895_[0]|PENDING|\n", stderr=""
        )

    first = _supervisor(tmp_path, runner=submit_runner)
    first.advance_once()
    state = read_json(first.state_path)
    audit = state["stages"]["audit"]
    audit.pop("finalizer_mode")
    audit["finalizer_submission_status"] = "rejected"
    audit["finalizer_command"] = ["sbatch", "--account=fmu@cpu"]
    audit["finalizer_submission_error"] = "AssocGrpSubmitJobsLimit"
    atomic_write_json(first.state_path, state, canonical=True)
    observed: list[list[str]] = []

    def resumed_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        observed.append(command)
        assert command[0] == "sacct"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="162895|162895_0|PENDING|\n",
            stderr="",
        )

    resumed = _supervisor(tmp_path, runner=resumed_runner)
    result = resumed.advance_once()
    assert result["worker_job_id"] == "162895"
    assert not any(command[0] == "sbatch" for command in observed)
    converted = read_json(resumed.state_path)["stages"]["audit"]
    assert converted["finalizer_submission_status"] == "disabled"
    assert converted["legacy_cpu_finalizer"] == {
        "finalizer_command": ["sbatch", "--account=fmu@cpu"],
        "finalizer_submission_error": "AssocGrpSubmitJobsLimit",
        "finalizer_submission_status": "rejected",
    }


def test_completed_array_is_finalized_locally_without_cpu_sbatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def submit_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "sbatch":
            return subprocess.CompletedProcess(
                command, 0, stdout="6001\n", stderr=""
            )
        return subprocess.CompletedProcess(
            command, 0, stdout="6001|6001_[0]|PENDING|\n", stderr=""
        )

    first = _supervisor(tmp_path, runner=submit_runner)
    first.advance_once()
    finalized: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        supervisor_module,
        "finalize_stage",
        lambda study, manifest: finalized.append(
            (Path(study), Path(manifest))
        ),
    )
    monkeypatch.setattr(
        supervisor_module,
        "validate_stage_completion",
        lambda **unused: {},
    )
    observed: list[list[str]] = []

    def completed_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        observed.append(command)
        assert command[0] == "sacct"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="6001|6001_0|COMPLETED|\n",
            stderr="",
        )

    resumed = _supervisor(tmp_path, runner=completed_runner)
    result = resumed.advance_once()
    assert result["status"] == "stage_complete"
    assert result["worker_parent_state"] == "COMPLETED"
    assert len(finalized) == 1
    assert not any(command[0] == "sbatch" for command in observed)
    assert read_json(resumed.state_path)["stages"]["audit"]["status"] == "complete"


def test_ambiguous_submission_intent_fails_closed_instead_of_duplicating(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(tmp_path, runner=lambda *args, **kwargs: None)
    _manifest, manifest_path, record = supervisor._ensure_manifest("audit")
    record.update(
        {
            "manifest_path": str(manifest_path),
            "worker_submission_status": "started",
        }
    )
    supervisor._save()
    calls: list[list[str]] = []

    def forbidden_runner(
        command: list[str], **unused: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        raise AssertionError("must not call Slurm")

    resumed = _supervisor(tmp_path, runner=forbidden_runner)
    with pytest.raises(RuntimeError, match="duplicate submission"):
        resumed.advance_once()
    assert calls == []


def test_worker_contract_augmentation_is_idempotent() -> None:
    command = [
        "sbatch",
        "--parsable",
        "--export=ALL,EXISTING=1",
        "worker.sh",
    ]
    once = supervisor_module._augment_worker_contract(command)
    twice = supervisor_module._augment_worker_contract(once)
    assert once == twice
    assert once.count("--hint=nomultithread") == 1
    export = next(item for item in once if item.startswith("--export="))
    for key, value in supervisor_module.THREAD_EXPORTS.items():
        assert f"{key}={value}" in export
