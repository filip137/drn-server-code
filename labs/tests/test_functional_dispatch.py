from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import experiment_executor as executor
from experiments import functional_dispatch as dispatch


def _plan(tmp_path: Path, run_class: str = "validation") -> dict:
    jobs = []
    job_ids = ["job-a"] if run_class == "validation" else ["job-a", "job-b"]
    for job_id in job_ids:
        output = tmp_path / "results" / "jobs" / job_id
        if run_class == "validation":
            code = (
                "import json,pathlib; "
                f"p=pathlib.Path({str(output)!r}); "
                "assert not p.exists(),p; "
                "p.mkdir(parents=True,exist_ok=True); "
                "(p/'checkpoint.pt').write_bytes(b'x'); "
                "(p/'functional_preflight.json').write_text(json.dumps({"
                "'schema_version':'experiment-functional-preflight/v1',"
                "'status':'passed','passed':True,'official_test_read':False,"
                "'environment':{'python_version':'test'},"
                "'smoke':{'kind':'perfectdiode-conv-one-batch/v1',"
                "'status':'passed','completed_steps':1,'loss':1.0,"
                "'accuracy':0.5,'checkpoint':{'path':'checkpoint.pt',"
                "'disposable':True,'sha256':'" + "a" * 64 + "'}}}))"
            )
        else:
            code = (
                "import json,pathlib; "
                f"p=pathlib.Path({str(output)!r}); "
                "assert not p.exists(),p; "
                "p.mkdir(parents=True,exist_ok=True); "
                "(p/'result.json').write_text('{}'); "
                "(p/'job_completion.json').write_text(json.dumps({"
                "'schema_version':'experiment-job-completion/v1',"
                "'status':'complete','outcome':'completed',"
                "'attempt_id':'proposal-local-production',"
                f"'job_id':{job_id!r},'official_test_read':False,"
                "'adapter_provenance':{'python':'test'},"
                "'result':{'official_test_read':False},"
                "'artifacts':[{'path':'result.json','sha256':'"
                + "b" * 64
                + "'}]}))"
            )
        jobs.append(
            {
                "job_id": job_id,
                "output_dir": str(output),
                "worker_argv": [sys.executable, "-c", code],
                "dispatch_argv": [],
            }
        )
    return {
        "schema_version": executor.EXECUTION_PLAN_SCHEMA,
        "status": "planned",
        "study_id": "study-a",
        "proposal_id": "proposal",
        "run_class": run_class,
        "launch_after_checks": True,
        "preflight_policy": {
            "functional_smoke_required": True,
            "fresh_tk_gate_required": False,
        },
        "attempts": [
            {
                "attempt_id": f"proposal-local-{run_class}",
                "target_id": "local",
                "workspace_root": str(tmp_path),
                "result_root": str(tmp_path / "results"),
                "tmux_session": f"test-{run_class}",
                "jobs": jobs,
                "launch_batches": (
                    [job_ids] if run_class == "validation" else [[job_ids[0]], [job_ids[1]]]
                ),
            }
        ],
    }


def _preflight() -> dict:
    return {
        "schema_version": executor.PREFLIGHT_RESULT_SCHEMA,
        "status": "passed",
        "proposal_id": "proposal",
    }


def test_validation_request_executes_one_adapter_and_receipt(tmp_path) -> None:
    request = dispatch.build_dispatch_requests(
        _plan(tmp_path),
        expected_duration_seconds=60,
        hard_deadline_seconds=300,
    )[0]
    result = dispatch.execute_request(request)

    assert result["status"] == "complete"
    assert result["run_class"] == "validation"
    assert [item["job_id"] for item in result["job_results"]] == ["job-a"]
    assert Path(request["state_dir"], "attempt_result.json").is_file()


def test_production_requires_passing_preflight(tmp_path) -> None:
    with pytest.raises(ValueError, match="passing production"):
        dispatch.build_dispatch_requests(
            _plan(tmp_path, "production"),
            expected_duration_seconds=600,
            hard_deadline_seconds=900,
        )


def test_production_batches_complete_without_retry(tmp_path) -> None:
    requests = dispatch.build_dispatch_requests(
        _plan(tmp_path, "production"),
        expected_duration_seconds=600,
        hard_deadline_seconds=900,
        production_preflight=_preflight(),
    )
    result = dispatch.execute_request(requests[0])
    assert result["status"] == "complete"
    assert [item["job_id"] for item in result["job_results"]] == [
        "job-a",
        "job-b",
    ]
    state = Path(requests[0]["state_dir"])
    for job_id in ("job-a", "job-b"):
        assert (state / "worker_logs" / f"{job_id}.log").is_file()
        assert not (
            Path(tmp_path, "results", "jobs", job_id, "worker.log")
        ).exists()


def test_failure_is_first_write_terminal_and_second_batch_not_launched(
    tmp_path,
) -> None:
    plan = _plan(tmp_path, "production")
    plan["attempts"][0]["jobs"][0]["worker_argv"] = [
        sys.executable,
        "-c",
        "raise SystemExit(7)",
    ]
    request = dispatch.build_dispatch_requests(
        plan,
        expected_duration_seconds=600,
        hard_deadline_seconds=900,
        production_preflight=_preflight(),
    )[0]
    result = dispatch.execute_request(request)

    assert result["status"] == "failed"
    assert result["terminal"] is True
    assert result["stage"] == "payload_execution"
    assert result["launched_job_count"] == 1
    assert not Path(
        plan["attempts"][0]["jobs"][1]["output_dir"]
    ).exists()
    assert dispatch.execute_request(request) == result


def test_start_prints_arm_only_for_production(monkeypatch, capsys, tmp_path) -> None:
    request = dispatch.build_dispatch_requests(
        _plan(tmp_path, "production"),
        expected_duration_seconds=600,
        hard_deadline_seconds=900,
        production_preflight=_preflight(),
    )[0]
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["tmux", "list-panes"]:
            return subprocess.CompletedProcess(argv, 0, "123 python\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    receipt = dispatch.start_request(request, runner=runner)
    assert receipt["status"] == "dispatched"
    assert "LONG-RUN ATTEMPT ARMED" in capsys.readouterr().out
    assert calls[0][:3] == ["tmux", "new-session", "-d"]
    assert calls[1][:2] == ["tmux", "list-panes"]


def test_smoke_receipt_becomes_version_tolerant_host_facts() -> None:
    receipt = {
        "schema_version": "experiment-functional-preflight/v1",
        "status": "passed",
        "passed": True,
        "attempt_id": "attempt-a",
        "target": "local",
        "official_test_read": False,
        "capabilities": {
            "dataset_readable": True,
            "output_writable": True,
        },
        "environment": {
            "python_executable": "/different/python",
            "python_version": "3.13.2",
            "pytorch_version": "2.8.0",
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_runtime": "13.0",
            "cuda_devices": [
                {
                    "index": 0,
                    "name": "future-gpu",
                    "free_memory_bytes": 12 * 1024**3,
                }
            ],
        },
        "scientific_binding": {"train_indices_sha256": "c" * 64},
        "smoke": {
            "kind": "perfectdiode-conv-one-batch/v1",
            "status": "passed",
            "completed_steps": 1,
            "loss": 1.0,
            "accuracy": 0.5,
            "benchmark": {
                "cuda_peak_memory_reserved_bytes": 2 * 1024**3,
            },
            "checkpoint": {
                "path": "checkpoint.pt",
                "disposable": True,
                "sha256": "a" * 64,
            },
        },
    }
    profile = {
        "schema_version": executor.TARGET_PROFILE_SCHEMA,
        "target_id": "local",
        "target_kind": "local_tmux",
        "host": "local",
        "transport": {"kind": "local", "ssh_target": None},
        "workspace_root": "/workspace",
        "result_root": "/results",
        "data_root": "/data",
        "device": "cuda:0",
        "python_candidates": ["python"],
        "required_imports": ["torch"],
        "requires_cuda": True,
        "minimum_gpu_count": 1,
        "default_concurrency": 2,
        "tmux_session_prefix": "exp-",
    }
    facts = dispatch.host_facts_from_smoke(receipt, profile)
    assert facts["python"]["version"] == "3.13.2"
    assert facts["imports"]["torch"]["version"] == "2.8.0"
    assert facts["cuda"]["runtime"] == "13.0"
    assert facts["max_safe_concurrency"] == 2


def test_remote_submit_uses_one_profile_driven_ssh_call(tmp_path) -> None:
    request = dispatch.build_dispatch_requests(
        _plan(tmp_path),
        expected_duration_seconds=60,
        hard_deadline_seconds=300,
    )[0]
    request["target_id"] = "trex"
    profile = {
        "schema_version": executor.TARGET_PROFILE_SCHEMA,
        "target_id": "trex",
        "target_kind": "ssh_tmux",
        "host": "trex",
        "transport": {
            "kind": "ssh",
            "ssh_target": "filip@trex",
            "ssh_config": "/home/test/.ssh/config",
        },
        "workspace_root": str(tmp_path),
        "result_root": str(tmp_path / "results"),
        "data_root": "/data",
        "device": "cuda:0",
        "python_candidates": ["/opt/python", "python3"],
        "required_imports": ["torch"],
        "requires_cuda": True,
        "minimum_gpu_count": 1,
        "default_concurrency": 2,
        "tmux_session_prefix": "exp-trex-",
    }
    calls = []
    expected = {
        "schema_version": dispatch.DISPATCH_RECEIPT_SCHEMA,
        "status": "dispatched",
        "attempt_id": request["attempt_id"],
    }

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            json.dumps(expected),
            "",
        )

    assert dispatch.submit_request(request, profile, runner=runner) == expected
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[:4] == [
        "ssh",
        "-F",
        "/home/test/.ssh/config",
        "filip@trex",
    ]
    assert "/opt/python -m experiments.functional_dispatch remote-start" in (
        argv[4]
    )
    assert f"cd {tmp_path}" in argv[4]
    assert json.loads(kwargs["input"])["target_id"] == "trex"


def test_remote_query_uses_read_only_status_command(tmp_path) -> None:
    request = dispatch.build_dispatch_requests(
        _plan(tmp_path),
        expected_duration_seconds=60,
        hard_deadline_seconds=300,
    )[0]
    request["target_id"] = "trex"
    profile = {
        "schema_version": executor.TARGET_PROFILE_SCHEMA,
        "target_id": "trex",
        "target_kind": "ssh_tmux",
        "host": "trex",
        "transport": {
            "kind": "ssh",
            "ssh_target": "filip@trex",
            "ssh_config": "/home/test/.ssh/config",
        },
        "workspace_root": "/profile/default/is/not/the/deployment",
        "result_root": str(tmp_path / "results"),
        "data_root": "/data",
        "device": "cuda:0",
        "python_candidates": ["/opt/python"],
        "required_imports": ["torch"],
        "requires_cuda": True,
        "minimum_gpu_count": 1,
        "default_concurrency": 1,
        "tmux_session_prefix": "exp-trex-",
    }
    expected = {"schema_version": "functional-dispatch-status/v1", "status": "running"}
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, json.dumps(expected), "")

    assert dispatch.query_request(request, profile, runner=runner) == expected
    argv, kwargs = calls[0]
    assert "remote-status" in argv[-1]
    assert "remote-start" not in argv[-1]
    assert f"cd {tmp_path}" in argv[-1]
    assert json.loads(kwargs["input"])["target_id"] == "trex"


def test_status_prefers_live_progress_over_dispatch_receipt(tmp_path) -> None:
    request = dispatch.build_dispatch_requests(
        _plan(tmp_path),
        expected_duration_seconds=60,
        hard_deadline_seconds=300,
    )[0]
    state = Path(request["state_dir"])
    state.mkdir(parents=True)
    (state / "dispatch.json").write_text(
        json.dumps({"status": "dispatched", "attempt_id": request["attempt_id"]})
    )
    expected = {
        "status": "running",
        "attempt_id": request["attempt_id"],
        "live_jobs": [{"job_id": "job-a", "state": "running"}],
    }
    (state / "progress.json").write_text(json.dumps(expected))

    assert dispatch.status_request(request) == expected


def test_status_cli_accepts_profile_for_target_aware_query(
    monkeypatch, capsys, tmp_path
) -> None:
    request = dispatch.build_dispatch_requests(
        _plan(tmp_path),
        expected_duration_seconds=60,
        hard_deadline_seconds=300,
    )[0]
    profile = {"target_id": "local", "sentinel": "profile-read"}
    request_path = tmp_path / "request.json"
    profile_path = tmp_path / "profile.json"
    request_path.write_text(json.dumps(request))
    profile_path.write_text(json.dumps(profile))
    expected = {"status": "running", "attempt_id": request["attempt_id"]}
    calls = []

    def query_request(request_value, profile_value):
        calls.append((request_value, profile_value))
        return expected

    monkeypatch.setattr(dispatch, "query_request", query_request)

    assert (
        dispatch.main(
            [
                "status",
                "--request",
                str(request_path),
                "--profile",
                str(profile_path),
            ]
        )
        == 0
    )
    assert calls == [(request, profile)]
    assert json.loads(capsys.readouterr().out) == expected


def test_remote_query_preserves_structured_failure_exit_two(tmp_path) -> None:
    request = dispatch.build_dispatch_requests(
        _plan(tmp_path),
        expected_duration_seconds=60,
        hard_deadline_seconds=300,
    )[0]
    request["target_id"] = "trex"
    profile = {
        "schema_version": executor.TARGET_PROFILE_SCHEMA,
        "target_id": "trex",
        "target_kind": "ssh_tmux",
        "host": "trex",
        "transport": {"kind": "ssh", "ssh_target": "filip@trex"},
        "workspace_root": str(tmp_path),
        "result_root": str(tmp_path / "results"),
        "data_root": "/data",
        "device": "cuda:0",
        "python_candidates": ["/opt/python"],
        "required_imports": ["torch"],
        "requires_cuda": True,
        "minimum_gpu_count": 1,
        "default_concurrency": 1,
        "tmux_session_prefix": "exp-trex-",
    }
    expected = {"schema_version": dispatch.DISPATCH_FAILURE_SCHEMA, "status": "failed"}

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 2, json.dumps(expected), "")

    assert dispatch.query_request(request, profile, runner=runner) == expected


def test_submit_rejects_profile_target_mismatch(tmp_path) -> None:
    request = dispatch.build_dispatch_requests(
        _plan(tmp_path),
        expected_duration_seconds=60,
        hard_deadline_seconds=300,
    )[0]
    profile = {
        "schema_version": executor.TARGET_PROFILE_SCHEMA,
        "target_id": "trex",
        "target_kind": "ssh_tmux",
        "host": "trex",
        "transport": {"kind": "ssh", "ssh_target": "filip@trex"},
        "workspace_root": str(tmp_path),
        "result_root": str(tmp_path / "results"),
        "data_root": "/data",
        "device": "cuda:0",
        "python_candidates": ["python3"],
        "required_imports": ["torch"],
        "requires_cuda": True,
        "minimum_gpu_count": 1,
        "default_concurrency": 1,
        "tmux_session_prefix": "exp-trex-",
    }
    with pytest.raises(ValueError, match="Expected request target_id"):
        dispatch.submit_request(request, profile)


def test_scheduled_runner_preflight_is_side_effect_free() -> None:
    result = dispatch.runner_preflight()

    assert result["status"] == "passed"
    assert result["side_effects_performed"] is False
    assert set(result["profiles"]) == {"local", "trex", "akib"}


def test_runner_preflight_supports_absolute_script_invocation() -> None:
    runner = Path(dispatch.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(runner), "--preflight"],
        cwd=runner.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "passed"
    assert result["side_effects_performed"] is False


def test_detached_process_backend_has_immediate_pid_readback(tmp_path) -> None:
    request = dispatch.build_dispatch_requests(
        _plan(tmp_path),
        expected_duration_seconds=60,
        hard_deadline_seconds=300,
    )[0]
    request["launch_backend"] = "detached_process"

    class Process:
        pid = 4321

        @staticmethod
        def poll():
            return None

    calls = []

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return Process()

    receipt = dispatch.start_request(request, popen_factory=popen)
    assert receipt["status"] == "dispatched"
    assert receipt["launch_backend"] == "detached_process"
    assert receipt["tmux_session"] is None
    assert receipt["immediate_readback"] == "pid=4321 state=running"
    assert calls[0][0][:3] == [
        sys.executable,
        "-m",
        "experiments.functional_dispatch",
    ]
