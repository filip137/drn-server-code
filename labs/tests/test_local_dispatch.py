from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import pytest

import experiments.local_dispatch as dispatch


REPO_ROOT = Path(__file__).resolve().parents[2]


class MainRunner:
    """Deterministic stand-in for nvidia-smi and tmux."""

    def __init__(
        self,
        *,
        compute_pids: str = "",
        git_status_stdout: str = "",
        readback_returncode: int = 0,
        readback_stdout: str | None = None,
    ) -> None:
        self.compute_pids = compute_pids
        self.git_status_stdout = git_status_stdout
        self.readback_returncode = readback_returncode
        self.readback_stdout = readback_stdout
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self._lock = threading.Lock()

    def __call__(
        self,
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        argv = list(command)
        with self._lock:
            self.calls.append((argv, dict(kwargs)))
        if argv[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=self.compute_pids,
                stderr="",
            )
        if argv[0] == "git":
            if argv[-2:] == ["rev-parse", "--show-toplevel"]:
                stdout = argv[2] + "\n"
            elif argv[-2:] == ["rev-parse", "HEAD"]:
                stdout = "a" * 40 + "\n"
            elif "status" in argv:
                stdout = self.git_status_stdout
            elif "ls-files" in argv:
                stdout = "\n".join(argv[argv.index("--") + 1 :]) + "\n"
            else:
                stdout = ""
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=stdout,
                stderr="",
            )
        if len(argv) > 1 and argv[1].endswith(
            "validate_experiment_plan.py"
        ):
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="OK\n",
                stderr="",
            )
        if argv[:2] == ["tmux", "has-session"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="",
                stderr="",
            )
        if argv[:2] in (
            ["tmux", "new-window"],
            ["tmux", "new-session"],
        ):
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="@11|%22|333\n",
                stderr="",
            )
        if argv[:2] == ["tmux", "display-message"]:
            return subprocess.CompletedProcess(
                argv,
                self.readback_returncode,
                stdout=(
                    self.readback_stdout
                    if self.readback_stdout is not None
                    else "main|@11|%22|333|0\n"
                    if self.readback_returncode == 0
                    else ""
                ),
                stderr=(
                    ""
                    if self.readback_returncode == 0
                    else "pane disappeared"
                ),
            )
        raise AssertionError(f"unexpected command: {argv!r}")


def _write_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    target: str,
) -> tuple[dict[str, Any], Path, Path]:
    monkeypatch.setenv("USER", "dispatch-test")
    approved_root = tmp_path / "approved"
    cwd = approved_root
    binary = approved_root / "bin" / "python"
    helper = approved_root / "repo" / "experiments" / "local_dispatch.py"
    cwd.mkdir(parents=True)
    binary.parent.mkdir(parents=True)
    helper.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.write_text("# test helper\n", encoding="utf-8")
    binary.chmod(0o755)

    profile_path = approved_root / "configs" / "dispatch" / f"{target}.json"
    profile_path.parent.mkdir(parents=True)
    is_local = target == "main"
    profile_value = {
        "schema_version": dispatch.PROFILE_SCHEMA,
        "target": target,
        "transport": "local" if is_local else "ssh",
        "python_path_template": str(binary),
        "helper_path_template": str(helper),
        "profile_path_template": str(profile_path),
        "repo_root_template": str(approved_root),
        "state_root_template": str(
            approved_root / "simulation_results" / "local_dispatch" / target
        ),
        "allowed_cwd_roots": [str(approved_root)],
        "allowed_executable_roots": [str(approved_root)],
        "base_env": {
            "HOME": "/home/{user}",
            "USER": "{user}",
            "LOGNAME": "{user}",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
            "TMPDIR": "/tmp",
            "PYTHONNOUSERSITE": "1",
        },
        "tmux": {
            "mode": "existing_session" if is_local else "attempt_session",
            "session": "main" if is_local else "",
            "session_prefix": "" if is_local else "mnist-",
        },
        "gpu": {
            "require_no_compute_processes": True,
            "device_index": 0,
        },
        "ssh": (
            None
            if is_local
            else {
                "target": "dispatch-test@trex",
                "connect_timeout_seconds": 7,
            }
        ),
        "command_timeout_seconds": 19,
    }
    profile_path.write_text(
        json.dumps(profile_value, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dispatch.load_profile(profile_path), cwd, binary


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment_binding(
    profile: Mapping[str, Any],
    overrides: Mapping[str, str],
) -> dict[str, Any]:
    normalized_overrides = dict(sorted(overrides.items()))
    effective = dict(
        sorted({**profile["base_env"], **normalized_overrides}.items())
    )
    canonical = json.dumps(
        effective,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "overrides": normalized_overrides,
        "effective": effective,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _assert_isolated_shell_command(
    shell_command: str,
    *,
    environment: Mapping[str, str],
) -> list[str]:
    words = shlex.split(shell_command)
    expected_assignments = [
        f"{key}={value}" for key, value in sorted(environment.items())
    ]
    assert words[:2] == ["/usr/bin/env", "-i"]
    assert words[2 : 2 + len(expected_assignments)] == expected_assignments
    prefix = words[: 2 + len(expected_assignments)]
    for blocked in ("LD_PRELOAD=", "PYTHONPATH=", "PYTHONHOME="):
        assert not any(item.startswith(blocked) for item in prefix)
    return words[2 + len(expected_assignments) :]


def _write_perfectdiode_smoke(
    *,
    output_root: Path,
    experiment_id: str,
    target: str,
    source_id: str,
    fixture_name: str = "smoke",
) -> tuple[Path, Path]:
    producer_binding = {
        "main": {"host": "akib", "architecture": "conv1"},
        "trex": {"host": "trex", "architecture": "conv2"},
    }[target]
    smoke_root = output_root / "preflight" / fixture_name
    smoke_root.mkdir(parents=True, exist_ok=True)
    request_path = smoke_root / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "mnist-conv-perfectdiode-preflight-request/v1"
                ),
                "study_id": experiment_id,
                "host": producer_binding["host"],
                "architecture": producer_binding["architecture"],
                "code_provenance": {
                    "git_revision": source_id,
                    "dirty_source_digest": None,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = output_root / "artifacts" / f"{fixture_name}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps({"status": "passed", "fixture": fixture_name}) + "\n",
        encoding="utf-8",
    )
    receipt_path = smoke_root / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "mnist-conv-perfectdiode-preflight-receipt/v1"
                ),
                "study_id": experiment_id,
                "host": producer_binding["host"],
                "architecture": producer_binding["architecture"],
                "surface_id": "surface-001",
                "entry_id": f"{fixture_name}-entry",
                "execution_passed": True,
                "scientific_center_safety_clean": True,
                "completed_steps": 1,
                "expected_steps": 1,
                "benchmark": {"elapsed_seconds": 0.01},
                "request_sha256": _sha256(request_path),
                "outputs": [
                    {
                        "path": str(artifact.relative_to(output_root)),
                        "sha256": _sha256(artifact),
                        "bytes": artifact.stat().st_size,
                    }
                ],
                "official_test_read": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt_path, artifact


def _write_scheduled_preflight(
    *,
    output_root: Path,
    cwd: Path,
    receipt_root: Path | None = None,
) -> tuple[Path, Path]:
    scheduled_root = output_root / "preflight" / "scheduled"
    scheduled_root.mkdir(parents=True, exist_ok=True)
    runner = scheduled_root / "runner.sh"
    runner.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'test "${1:-}" = "--preflight"\n'
        "printf 'bounded scheduled preflight passed\\n'\n",
        encoding="utf-8",
    )
    runner.chmod(0o700)
    receipt_dir = receipt_root or scheduled_root
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = receipt_dir / "receipt.json"
    producer = (
        REPO_ROOT
        / "skills"
        / "scheduled-run-preflight"
        / "scripts"
        / "preflight_scheduled_runner.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(producer),
            "--runner",
            str(runner),
            "--cwd",
            str(cwd),
            "--receipt",
            str(receipt),
            "--timeout-seconds",
            "5",
        ],
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return receipt, runner


def _write_producer_tk_selection(
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_root: Path,
    study_id: str,
    source_id: str,
) -> Path:
    import experiments.run_mnist_conv_perfectdiode_tk as tk_producer

    root = output_root / "preflight" / "tk"
    root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    schemes = ("baseline", "ours", "legacy")
    lanes = ("main", "akibscomputer", "main")
    for index, (scheme, lane) in enumerate(zip(schemes, lanes)):
        entry_root = root / "entries" / scheme
        entry_root.mkdir(parents=True)
        environment = entry_root / "environment.json"
        audit = entry_root / "operating_point_audit.json"
        result_path = entry_root / "result.json"
        completion_path = entry_root / "completion.json"
        environment.write_text(
            json.dumps({"execution_host": lane}) + "\n",
            encoding="utf-8",
        )
        audit.write_text(
            json.dumps({"passed": True, "scheme": scheme}) + "\n",
            encoding="utf-8",
        )
        completion_path.write_text(
            json.dumps({"state": "complete", "scheme": scheme}) + "\n",
            encoding="utf-8",
        )
        relative_root = Path("entries") / scheme
        entry = {
            "entry_index": index,
            "entry_id": f"{study_id}-{scheme}",
            "row_id": f"conv3-{scheme}",
            "scheme": scheme,
            "execution": {"lane": lane},
            "completion_path": str(relative_root / completion_path.name),
            "outputs": [
                str(relative_root / environment.name),
                str(relative_root / audit.name),
                str(relative_root / result_path.name),
            ],
        }
        result = {
            "row_id": entry["row_id"],
            "scheme": scheme,
            "execution_backend": "tmux",
            "execution_host": lane,
            "execution_environment_sha256": _sha256(environment),
            "run_name": f"producer-{scheme}",
            "voltage_amp": 1.0 if scheme == "baseline" else 4.0,
            "current_amp": 0.25 if scheme == "legacy" else 1.0,
            "input_gain": 360.0,
            "status": "selected",
            "selected_t": 16,
            "selected_k": 8,
            "t_reference": 64,
            "k_reference": 64,
            "t_extension_used": False,
            "k128_sentinel_used": False,
            "operating_point_audit_sha256": _sha256(audit),
            "operating_point_audit_passed": True,
            "t_cohort_indices_sha256": "4" * 64,
            "k_cohort_indices_sha256": "5" * 64,
        }
        result_path.write_text(
            json.dumps(result, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        entries.append(entry)
        results[entry["entry_id"]] = result
    manifest = {
        "schema_version": "mnist-conv-perfectdiode-tk-manifest/v1",
        "study_id": study_id,
        "config_sha256": "6" * 64,
        "asset_binding": {
            "schema_version": "mnist-conv-perfectdiode-tk-assets/v1",
            "assets_sha256": "7" * 64,
        },
        "entries": entries,
        "entry_count": len(entries),
        "selection_path": "selection.json",
        "staged_source": {
            "commit": source_id,
            "archive_sha256": "8" * 64,
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = _sha256(manifest_path)

    def fake_load_manifest(
        supplied: str | Path,
        *,
        enforce_code_identity: bool,
    ) -> tuple[Path, dict[str, Any], object]:
        assert Path(supplied).resolve() == manifest_path.resolve()
        assert enforce_code_identity is False
        return manifest_path.resolve(), copy.deepcopy(manifest), object()

    def fake_validate_entry(
        supplied_root: Path,
        supplied_manifest: Mapping[str, Any],
        entry: Mapping[str, Any],
        *,
        manifest_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert supplied_root == root.resolve()
        assert supplied_manifest["study_id"] == study_id
        assert manifest_sha256 == _sha256(manifest_path)
        return (
            {"state": "complete"},
            copy.deepcopy(results[str(entry["entry_id"])]),
        )

    monkeypatch.setattr(tk_producer, "_load_manifest", fake_load_manifest)
    monkeypatch.setattr(
        tk_producer,
        "_validate_entry_completion",
        fake_validate_entry,
    )
    selection = tk_producer._publish_selection_if_complete(
        root,
        manifest,
        manifest_sha256=manifest_sha256,
    )
    assert selection is not None

    def forbidden_publisher(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError(
            "dispatch validation must not call the publishing T/K helper"
        )

    monkeypatch.setattr(
        tk_producer,
        "_publish_selection_if_complete",
        forbidden_publisher,
    )
    return root / "selection.json"


def _read_plan_contract(plan_path: Path) -> dict[str, Any]:
    matches = dispatch.PLAN_JSON_BLOCK_RE.findall(
        plan_path.read_text(encoding="utf-8")
    )
    assert len(matches) == 1
    return json.loads(matches[0])


def _write_plan_contract(
    plan_path: Path,
    contract: Mapping[str, Any],
) -> None:
    plan_path.write_text(
        "# Test plan\n\n```json\n"
        + json.dumps(contract, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )


def _preflight_arguments(
    *,
    profile: Mapping[str, Any],
    cwd: Path,
    binary: Path,
    plan: Path,
) -> dict[str, Any]:
    return {
        "profile": profile,
        "experiment_id": "experiment-001",
        "cwd": str(cwd),
        "argv": [str(binary), "-c", "print('payload')"],
        "env": {"OMP_NUM_THREADS": "1"},
        "source_id": "a" * 40,
        "approved_plan_path": str(plan),
    }


def _write_launch_authority(
    profile: Mapping[str, Any],
    *,
    cwd: Path,
    binary: Path,
    attempt_id: str,
    experiment_id: str,
    build_envelope: bool = True,
) -> tuple[Path, Path]:
    validator_root = (
        cwd
        / "skills"
        / "run-experiment-pipeline"
        / "scripts"
    )
    validator_root.mkdir(parents=True, exist_ok=True)
    plan_validator = validator_root / "validate_experiment_plan.py"
    tracker_validator = validator_root / "validate_current_experiments.py"
    plan_validator.write_text("# plan validator\n", encoding="utf-8")
    tracker_validator.write_text("# tracker validator\n", encoding="utf-8")

    tracker = cwd / "docs" / "current_experiments.md"
    tracker.parent.mkdir(parents=True, exist_ok=True)
    tracker.write_text("launch-ready fixture\n", encoding="utf-8")

    output_root = cwd / "results" / experiment_id
    preflight = output_root / "preflight" / "receipt.json"
    preflight.parent.mkdir(parents=True, exist_ok=True)
    smoke_receipt, _smoke_artifact = _write_perfectdiode_smoke(
        output_root=output_root,
        experiment_id=experiment_id,
        target=str(profile["target"]),
        source_id="a" * 40,
    )

    plan = cwd / "docs" / "experiment_plans" / f"{experiment_id}.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": "experiment-run-plan/v1",
        "experiment_id": experiment_id,
        "approval": {
            "status": "approved",
            "approved_by": "test",
            "approved_at": "2026-07-27T00:00:00+00:00",
            "amendment_of": None,
        },
        "execution": {
            "launcher": [
                str(binary),
                "-c",
                "print('payload')",
            ],
            "preflight": {
                "smoke_required": True,
                "tk_reference_required": False,
                "scheduled_run_preflight_required": False,
                "receipt_schemas": {
                    "smoke": (
                        "mnist-conv-perfectdiode-preflight-receipt/v1"
                    ),
                    "tk_reference": None,
                    "scheduled_run_preflight": None,
                },
                "receipt_bindings": {
                    "smoke": {
                        "receipt_path": str(
                            smoke_receipt.relative_to(cwd)
                        ),
                        "subject_id": experiment_id,
                        "producer_source_id": "a" * 40,
                    },
                    "tk_reference": None,
                    "scheduled_run_preflight": None,
                },
                "receipt_path": str(
                    preflight.relative_to(cwd)
                ),
            },
        },
        "storage": {
            "local_bundle_path": str(output_root.relative_to(cwd)),
        },
    }
    _write_plan_contract(plan, contract)
    if build_envelope:
        built = dispatch.build_preflight(
            profile=profile,
            experiment_id=experiment_id,
            cwd=str(cwd),
            argv=[str(binary), "-c", "print('payload')"],
            env={"OMP_NUM_THREADS": "1"},
            source_id="a" * 40,
            approved_plan_path=str(plan),
            smoke_receipt_path=str(smoke_receipt),
        )
        assert built["status"] in {"built", "already_built"}

    state_root = Path(str(profile["state_root"]))
    gate = output_root / "attempts" / attempt_id / "tracker-production-gate.json"
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(
        json.dumps(
            {
                "schema_version": dispatch.TRACKER_GATE_SCHEMA,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "experiment_id": experiment_id,
                "gate_stage": "production",
                "status": "launch-ready",
                "state_path": str(
                    state_root / attempt_id / "dispatch.json"
                ),
                "output_root": str(output_root),
                "tracker_path": str(tracker),
                "tracker_sha256": _sha256(tracker),
                "validator_path": str(tracker_validator),
                "validator_sha256": _sha256(tracker_validator),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return plan, gate


def _request(
    profile: Mapping[str, Any],
    *,
    cwd: Path,
    binary: Path,
    attempt_id: str = "attempt-001",
) -> dict[str, Any]:
    experiment_id = "experiment-001"
    plan, gate = _write_launch_authority(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id=attempt_id,
        experiment_id=experiment_id,
    )
    return dispatch.build_request(
        profile=profile,
        attempt_id=attempt_id,
        experiment_id=experiment_id,
        cwd=str(cwd),
        argv=[str(binary), "-c", "print('payload')"],
        env={"OMP_NUM_THREADS": "1"},
        source_id="a" * 40,
        expected_duration_seconds=600,
        hard_deadline_seconds=900,
        approved_plan_path=str(plan),
        tracker_gate_receipt_path=str(gate),
    )


def _commands(runner: MainRunner) -> list[list[str]]:
    return [command for command, _kwargs in runner.calls]


def test_main_dispatch_uses_one_fresh_window_without_send_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    monkeypatch.setenv("LD_PRELOAD", "/ambient/loader.so")
    monkeypatch.setenv("PYTHONPATH", "/ambient/python")
    monkeypatch.setenv("PYTHONHOME", "/ambient/home")
    runner = MainRunner()

    receipt = dispatch.start_request(
        request,
        profile=profile,
        runner=runner,
    )

    commands = _commands(runner)
    launches = [
        command for command in commands if command[:2] == ["tmux", "new-window"]
    ]
    assert receipt["schema_version"] == dispatch.DISPATCH_SCHEMA
    assert receipt["status"] == "dispatched"
    assert receipt["launched_job_count"] == 1
    assert receipt["tmux"] == {
        "window_id": "@11",
        "pane_id": "%22",
        "pane_pid": "333",
        "session": "main",
    }
    assert len(launches) == 1
    assert "-d" in launches[0]
    execute_tail = _assert_isolated_shell_command(
        launches[0][-1],
        environment={
            **profile["base_env"],
            **request["env"],
        },
    )
    assert execute_tail[:3] == [
        str(profile["python_path"]),
        str(profile["helper_path"]),
        "execute",
    ]
    assert "--request-sha256" in execute_tail
    assert all("send-keys" not in command for command in commands)
    assert all(command[:2] != ["tmux", "kill-window"] for command in commands)
    assert all(kwargs["timeout"] == 19 for _command, kwargs in runner.calls)
    attempt_root = Path(profile["state_root"]) / "attempt-001"
    assert (attempt_root / "request.json").is_file()
    assert (attempt_root / "dispatch.json").is_file()
    preflight = json.loads(
        Path(request["preflight_receipt_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert preflight["env"] == _environment_binding(
        profile,
        request["env"],
    )


def test_busy_gpu_fails_before_any_tmux_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    runner = MainRunner(compute_pids="1234\n5678\n")

    with pytest.raises(dispatch.DispatchError) as raised:
        dispatch.start_request(
            _request(profile, cwd=cwd, binary=binary),
            profile=profile,
            runner=runner,
        )

    assert raised.value.stage == "gpu_busy"
    assert raised.value.launched_job_count == 0
    assert _commands(runner)[-1] == [
        "nvidia-smi",
        "--id=0",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    attempt_root = Path(profile["state_root"]) / "attempt-001"
    assert (attempt_root / "request.json").is_file()
    failure = json.loads(
        (attempt_root / "failure.json").read_text(encoding="utf-8")
    )
    assert failure["stage"] == "gpu_busy"
    assert failure["terminal"] is True
    assert not any(
        command[:2] in (["tmux", "new-window"], ["tmux", "new-session"])
        for command in _commands(runner)
    )


def test_identical_attempt_is_idempotent_and_request_is_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    runner = MainRunner()

    first = dispatch.start_request(request, profile=profile, runner=runner)
    request_path = (
        Path(profile["state_root"]) / "attempt-001" / "request.json"
    )
    original_bytes = request_path.read_bytes()
    second = dispatch.start_request(request, profile=profile, runner=runner)

    assert first["status"] == "dispatched"
    assert second["status"] == "already_dispatched"
    assert request_path.read_bytes() == original_bytes
    assert sum(
        command[:2] == ["tmux", "new-window"]
        for command in _commands(runner)
    ) == 1
    assert sum(command[0] == "nvidia-smi" for command in _commands(runner)) == 1


def test_same_attempt_id_with_changed_request_is_a_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    runner = MainRunner()
    dispatch.start_request(request, profile=profile, runner=runner)
    changed = copy.deepcopy(request)
    changed["env"]["OMP_NUM_THREADS"] = "2"

    with pytest.raises(dispatch.DispatchError) as raised:
        dispatch.start_request(changed, profile=profile, runner=runner)

    assert raised.value.stage == "attempt_collision"
    assert raised.value.launched_job_count == 0
    assert sum(
        command[:2] == ["tmux", "new-window"]
        for command in _commands(runner)
    ) == 1


def test_trex_dispatch_is_exactly_one_bounded_batchmode_ssh_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="trex",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    monkeypatch.setenv("LD_PRELOAD", "/ambient/loader.so")
    monkeypatch.setenv("PYTHONPATH", "/ambient/python")
    monkeypatch.setenv("PYTHONHOME", "/ambient/home")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        argv = list(command)
        calls.append((argv, dict(kwargs)))
        receipt = {
            "schema_version": dispatch.DISPATCH_SCHEMA,
            "status": "dispatched",
            "attempt_id": request["attempt_id"],
            "target": "trex",
            "launched_job_count": 1,
        }
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(receipt) + "\n",
            stderr="",
        )

    receipt = dispatch.start_request(
        request,
        profile=profile,
        runner=runner,
    )

    assert receipt["status"] == "dispatched"
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[0] == "ssh"
    assert ["-o", "BatchMode=yes"] == command[1:3]
    assert ["-o", "ConnectTimeout=7"] == command[3:5]
    assert command[5] == "dispatch-test@trex"
    assert len(command) == 7
    assert "remote-dispatch" in command[6]
    assert str(profile["profile_path"]) in command[6]
    remote_tail = _assert_isolated_shell_command(
        command[6],
        environment=profile["base_env"],
    )
    assert remote_tail == [
        str(profile["python_path"]),
        str(profile["helper_path"]),
        "remote-dispatch",
        "--profile",
        str(profile["profile_path"]),
    ]
    assert kwargs["timeout"] == 19
    assert kwargs["text"] is True
    assert kwargs["capture_output"] is True
    assert json.loads(kwargs["input"]) == request
    assert not any(item.startswith("tmux") for item in command)


@pytest.mark.parametrize("failure_mode", ["timeout", "nonzero"])
def test_ssh_failure_is_ambiguous_and_never_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="trex",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    calls: list[list[str]] = []

    def runner(
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        argv = list(command)
        calls.append(argv)
        if failure_mode == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(
            argv,
            255,
            stdout="",
            stderr="connection lost",
        )

    with pytest.raises(dispatch.DispatchError) as raised:
        dispatch.start_request(
            request,
            profile=profile,
            runner=runner,
        )

    assert raised.value.stage == "ssh_dispatch_ambiguous"
    assert raised.value.launched_job_count is None
    assert calls and len(calls) == 1
    assert calls[0][0] == "ssh"


def test_not_found_status_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, _cwd, _binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    state_root = Path(profile["state_root"])
    assert not state_root.exists()

    result = dispatch.status_on_target(
        profile=profile,
        attempt_id="attempt-404",
    )

    assert result == {
        "status": "not_found",
        "attempt_id": "attempt-404",
        "target": "main",
    }
    assert not state_root.exists()


@pytest.mark.parametrize(
    "live_snapshot",
    [
        {},
        {4242: (1, "S", "101")},
    ],
    ids=["dead-worker", "reused-pid"],
)
def test_status_marks_dead_or_reused_worker_stale_and_reserves_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_snapshot: dict[int, tuple[int, str, str]],
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    attempt_root = Path(profile["state_root"]) / request["attempt_id"]
    request_path = attempt_root / "request.json"
    dispatch._atomic_create(request_path, request)
    running_path = attempt_root / "running.json"
    running = {
        "schema_version": dispatch.RUNNING_SCHEMA,
        "status": "running",
        "attempt_id": request["attempt_id"],
        "experiment_id": request["experiment_id"],
        "target": request["target"],
        "request_sha256": _sha256(request_path),
        "started_at_utc": "2026-07-28T00:00:00+00:00",
        "worker_pid": 4242,
        "worker_starttime": "100",
    }
    dispatch._atomic_create(running_path, running)
    before = {
        path.relative_to(attempt_root): (
            path.read_bytes(),
            path.stat().st_mode,
        )
        for path in attempt_root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        dispatch,
        "_proc_snapshot",
        lambda: live_snapshot,
    )

    result = dispatch.status_on_target(
        profile=profile,
        attempt_id=request["attempt_id"],
    )

    assert result == {
        **running,
        "status": "stale_running",
        "terminal": False,
        "lane_reserved": True,
        "worker_identity_live": False,
        "smallest_next_action": (
            "Inspect the immutable attempt and target read-only; do not "
            "launch another attempt until quiescence is proven."
        ),
    }
    after = {
        path.relative_to(attempt_root): (
            path.read_bytes(),
            path.stat().st_mode,
        )
        for path in attempt_root.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not (Path(profile["state_root"]) / ".dispatch.lock").exists()


def test_request_validation_rejects_unsafe_identifiers_and_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    valid = _request(profile, cwd=cwd, binary=binary)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_binary = outside / "python"
    outside_binary.write_text("", encoding="utf-8")
    invalid_requests: list[dict[str, Any]] = []

    invalid_attempt = copy.deepcopy(valid)
    invalid_attempt["attempt_id"] = "../../escape"
    invalid_requests.append(invalid_attempt)

    invalid_cwd = copy.deepcopy(valid)
    invalid_cwd["cwd"] = str(outside)
    invalid_requests.append(invalid_cwd)

    invalid_executable = copy.deepcopy(valid)
    invalid_executable["argv"][0] = str(outside_binary)
    invalid_requests.append(invalid_executable)

    invalid_env = copy.deepcopy(valid)
    invalid_env["env"] = {"LD_PRELOAD;touch": "bad"}
    invalid_requests.append(invalid_env)

    short_run = copy.deepcopy(valid)
    short_run["expected_duration_seconds"] = 599
    invalid_requests.append(short_run)

    for value in invalid_requests:
        with pytest.raises(ValueError):
            dispatch.validate_request(value, profile=profile)


def test_post_launch_readback_failure_is_terminal_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    runner = MainRunner(readback_returncode=1)

    with pytest.raises(dispatch.DispatchError) as raised:
        dispatch.start_request(
            _request(profile, cwd=cwd, binary=binary),
            profile=profile,
            runner=runner,
        )

    assert raised.value.stage == "post_dispatch_readback"
    assert raised.value.launched_job_count == 1
    assert raised.value.launch_id == "%22"
    commands = _commands(runner)
    assert sum(
        command[:2] == ["tmux", "new-window"] for command in commands
    ) == 1
    assert sum(
        command[:2] == ["tmux", "display-message"] for command in commands
    ) == 1
    assert all("send-keys" not in command for command in commands)
    assert all(
        command[:2] not in (["tmux", "kill-window"], ["tmux", "new-session"])
        for command in commands
    )
    failure = json.loads(
        (
            Path(profile["state_root"])
            / "attempt-001"
            / "failure.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert failure["terminal"] is True
    assert failure["stage"] == "post_dispatch_readback"
    assert failure["launched_job_count"] == 1
    assert failure["launch_id"] == "%22"


def test_concurrent_identical_dispatches_launch_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    runner = MainRunner()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _index: dispatch.start_request(
                    request,
                    profile=profile,
                    runner=runner,
                ),
                range(2),
            )
        )

    assert sorted(result["status"] for result in results) == [
        "already_dispatched",
        "dispatched",
    ]
    assert sum(
        command[:2] == ["tmux", "new-window"]
        for command in _commands(runner)
    ) == 1
    assert sum(command[0] == "nvidia-smi" for command in _commands(runner)) == 1


def test_trex_status_is_exactly_one_bounded_batchmode_ssh_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, _cwd, _binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="trex",
    )
    monkeypatch.setenv("LD_PRELOAD", "/ambient/loader.so")
    monkeypatch.setenv("PYTHONPATH", "/ambient/python")
    monkeypatch.setenv("PYTHONHOME", "/ambient/home")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        argv = list(command)
        calls.append((argv, dict(kwargs)))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                {
                    "status": "not_found",
                    "attempt_id": "attempt-404",
                    "target": "trex",
                }
            )
            + "\n",
            stderr="",
        )

    result = dispatch.status_request(
        profile=profile,
        attempt_id="attempt-404",
        runner=runner,
    )

    assert result["status"] == "not_found"
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:6] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=7",
        "dispatch-test@trex",
    ]
    assert len(command) == 7
    assert "remote-status" in command[6]
    assert "--attempt-id attempt-404" in command[6]
    assert str(profile["profile_path"]) in command[6]
    remote_tail = _assert_isolated_shell_command(
        command[6],
        environment=profile["base_env"],
    )
    assert remote_tail == [
        str(profile["python_path"]),
        str(profile["helper_path"]),
        "remote-status",
        "--profile",
        str(profile["profile_path"]),
        "--attempt-id",
        "attempt-404",
    ]
    assert kwargs["timeout"] == 19
    assert kwargs["input"] is None


def test_structured_remote_dispatch_failure_is_propagated_not_ambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="trex",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    calls: list[list[str]] = []
    remote_error = dispatch.DispatchError(
        "gpu_busy",
        "Expected target GPU 0 to have no compute processes.",
        launched_job_count=0,
        attempt_id=request["attempt_id"],
        experiment_id=request["experiment_id"],
        target="trex",
        completed_stages=["request_persisted", "launch_authorization"],
        last_valid_artifacts={"request_path": "/remote/request.json"},
    )

    def runner(
        command: Sequence[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        argv = list(command)
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv,
            2,
            stdout=json.dumps(remote_error.as_dict()) + "\n",
            stderr="",
        )

    with pytest.raises(dispatch.DispatchError) as raised:
        dispatch.start_request(
            request,
            profile=profile,
            runner=runner,
        )

    assert len(calls) == 1
    assert raised.value.stage == "gpu_busy"
    assert raised.value.launched_job_count == 0
    assert raised.value.attempt_id == request["attempt_id"]
    assert raised.value.experiment_id == request["experiment_id"]
    assert raised.value.target == "trex"
    assert raised.value.completed_stages == [
        "request_persisted",
        "launch_authorization",
    ]
    assert raised.value.last_valid_artifacts == {
        "request_path": "/remote/request.json"
    }


def test_existing_request_only_crash_state_becomes_terminal_without_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    attempt_root = Path(profile["state_root"]) / request["attempt_id"]
    request_path = attempt_root / "request.json"
    dispatch._atomic_create(request_path, request)
    calls: list[list[str]] = []

    def runner(
        command: Sequence[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        raise AssertionError("a request-only attempt must never launch")

    with pytest.raises(dispatch.DispatchError) as first:
        dispatch.dispatch_on_target(
            request,
            profile=profile,
            runner=runner,
        )

    assert first.value.stage == "dispatch_outcome_unknown"
    assert first.value.launched_job_count is None
    failure_path = attempt_root / "failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["terminal"] is True
    assert failure["retry_authorized"] is False
    assert failure["stage"] == "dispatch_outcome_unknown"
    assert failure["last_valid_artifacts"] == {
        "request_path": str(request_path),
        "request_sha256": _sha256(request_path),
    }

    with pytest.raises(dispatch.DispatchError) as second:
        dispatch.dispatch_on_target(
            request,
            profile=profile,
            runner=runner,
        )

    assert second.value.stage == "dispatch_outcome_unknown"
    assert failure_path.read_text(encoding="utf-8") == json.dumps(
        failure,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    assert calls == []


@pytest.mark.parametrize(
    "readback_stdout",
    [
        "malformed\n",
        "main|@99|%22|333|0\n",
        "main|@11|%22|333|1\n",
    ],
    ids=["malformed", "mismatched", "dead"],
)
def test_invalid_tmux_readback_persists_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    readback_stdout: str,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    runner = MainRunner(readback_stdout=readback_stdout)

    with pytest.raises(dispatch.DispatchError) as raised:
        dispatch.start_request(
            _request(profile, cwd=cwd, binary=binary),
            profile=profile,
            runner=runner,
        )

    assert raised.value.stage == "post_dispatch_readback"
    assert raised.value.launched_job_count == 1
    assert raised.value.launch_id == "%22"
    attempt_root = Path(profile["state_root"]) / "attempt-001"
    failure = json.loads(
        (attempt_root / "failure.json").read_text(encoding="utf-8")
    )
    assert failure["terminal"] is True
    assert failure["stage"] == "post_dispatch_readback"
    assert failure["launched_job_count"] == 1
    assert not (attempt_root / "dispatch.json").exists()
    assert sum(
        command[:2] == ["tmux", "new-window"]
        for command in _commands(runner)
    ) == 1
    assert sum(
        command[:2] == ["tmux", "display-message"]
        for command in _commands(runner)
    ) == 1


@pytest.mark.parametrize("failure_mode", ["stale_gate", "changed_tracker"])
def test_invalid_tracker_authorization_blocks_before_gpu_or_tmux(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    gate_path = Path(request["tracker_gate_receipt_path"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if failure_mode == "stale_gate":
        gate["generated_at_utc"] = (
            datetime.now(timezone.utc)
            - timedelta(
                seconds=dispatch.MAX_TRACKER_GATE_AGE_SECONDS + 30
            )
        ).isoformat()
        gate_path.write_text(
            json.dumps(gate, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        request["tracker_gate_receipt_sha256"] = _sha256(gate_path)
    else:
        tracker_path = Path(gate["tracker_path"])
        tracker_path.write_text(
            "changed after tracker gate publication\n",
            encoding="utf-8",
        )
    runner = MainRunner()

    with pytest.raises(dispatch.DispatchError) as raised:
        dispatch.start_request(
            request,
            profile=profile,
            runner=runner,
        )

    assert raised.value.stage == "launch_authorization"
    commands = _commands(runner)
    assert not any(command[0] == "nvidia-smi" for command in commands)
    assert not any(command[0] == "tmux" for command in commands)
    failure = json.loads(
        (
            Path(profile["state_root"])
            / request["attempt_id"]
            / "failure.json"
        ).read_text(encoding="utf-8")
    )
    assert failure["stage"] == "launch_authorization"
    assert failure["launched_job_count"] == 0


def test_dispatch_lock_contention_has_a_bounded_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    state_root = Path(profile["state_root"])
    state_root.mkdir(parents=True)
    lock_path = state_root / ".dispatch.lock"
    calls: list[list[str]] = []

    def runner(
        command: Sequence[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        raise AssertionError("lock timeout must precede external commands")

    monkeypatch.setattr(dispatch, "LOCK_TIMEOUT_SECONDS", 0.1)
    with lock_path.open("a+", encoding="utf-8") as held_lock:
        fcntl.flock(held_lock.fileno(), fcntl.LOCK_EX)
        started = time.monotonic()
        with pytest.raises(dispatch.DispatchError) as raised:
            dispatch.dispatch_on_target(
                request,
                profile=profile,
                runner=runner,
            )
        elapsed = time.monotonic() - started

    assert raised.value.stage == "dispatch_lock_timeout"
    assert raised.value.launched_job_count == 0
    assert raised.value.attempt_id == request["attempt_id"]
    assert raised.value.terminal is True
    assert raised.value.as_dict()["retry_authorized"] is False
    assert 0.08 <= elapsed < 1
    assert calls == []
    attempt_root = state_root / request["attempt_id"]
    assert (attempt_root / "request.json").is_file()
    assert (attempt_root / "failure.json").is_file()


def test_execute_worker_popen_failure_writes_terminal_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    attempt_root = Path(profile["state_root"]) / request["attempt_id"]
    request_path = attempt_root / "request.json"
    dispatch._atomic_create(request_path, request)
    request_sha256 = _sha256(request_path)
    binary.unlink()

    returncode = dispatch._execute_payload(
        request_path,
        request_sha256=request_sha256,
        profile=profile,
    )

    assert returncode == 1
    assert (attempt_root / "running.json").is_file()
    assert not (attempt_root / "completion.json").exists()
    failure = json.loads(
        (attempt_root / "failure.json").read_text(encoding="utf-8")
    )
    assert failure["schema_version"] == dispatch.FAILURE_SCHEMA
    assert failure["status"] == "failed"
    assert failure["terminal"] is True
    assert failure["stage"] == "payload_execution"
    assert failure["attempt_id"] == request["attempt_id"]
    assert failure["experiment_id"] == request["experiment_id"]
    assert failure["target"] == "main"
    assert "FileNotFoundError" in failure["error"]


def _pid_is_live(pid: int) -> bool:
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text(
            encoding="utf-8"
        )
    except OSError:
        return False
    return stat[stat.rfind(")") + 2 :].split()[0] != "Z"


@pytest.mark.skipif(
    not Path("/proc").is_dir(),
    reason="requires Linux process inspection",
)
def test_execute_hard_timeout_kills_term_ignoring_detached_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    descendant_pid_path = tmp_path / "descendant.pid"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "from pathlib import Path\n"
        "import signal\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "child_code = ("
        "\"import signal,time;\""
        "\"signal.signal(signal.SIGTERM,signal.SIG_IGN);\""
        "\"time.sleep(60)\""
        ")\n"
        "child = subprocess.Popen("
        "[sys.executable, '-c', child_code], start_new_session=True"
        ")\n"
        f"Path({str(descendant_pid_path)!r}).write_text("
        "str(child.pid), encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    attempt_root = Path(profile["state_root"]) / request["attempt_id"]
    request_path = attempt_root / "request.json"
    dispatch._atomic_create(request_path, request)
    request_sha256 = _sha256(request_path)
    real_validate = dispatch.validate_request

    def validate_with_short_deadline(
        raw: Mapping[str, Any],
        *,
        profile: Mapping[str, Any],
    ) -> dict[str, Any]:
        value = real_validate(raw, profile=profile)
        value["hard_deadline_seconds"] = 0.75
        return value

    monkeypatch.setattr(
        dispatch,
        "validate_request",
        validate_with_short_deadline,
    )
    descendant_pid: int | None = None
    try:
        returncode = dispatch._execute_payload(
            request_path,
            request_sha256=request_sha256,
            profile=profile,
        )
        assert descendant_pid_path.is_file()
        descendant_pid = int(
            descendant_pid_path.read_text(encoding="utf-8")
        )
        assert returncode != 0
        assert not _pid_is_live(descendant_pid)
        failure = json.loads(
            (attempt_root / "failure.json").read_text(encoding="utf-8")
        )
        assert failure["stage"] == "payload_execution"
        assert failure["terminal"] is True
        assert "timed_out=True" in failure["error"]
        assert not (attempt_root / "completion.json").exists()
    finally:
        if descendant_pid is not None and _pid_is_live(descendant_pid):
            os.kill(descendant_pid, signal.SIGKILL)


def test_approved_launcher_rejects_an_unapproved_argv_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    valid = _request(profile, cwd=cwd, binary=binary)

    with pytest.raises(ValueError, match="approved plan launcher"):
        dispatch.build_request(
            profile=profile,
            attempt_id="attempt-suffix",
            experiment_id=valid["experiment_id"],
            cwd=str(cwd),
            argv=[*valid["argv"], "--unapproved"],
            env={"OMP_NUM_THREADS": "1"},
            source_id=valid["source_id"],
            expected_duration_seconds=600,
            hard_deadline_seconds=900,
            approved_plan_path=valid["approved_plan_path"],
            tracker_gate_receipt_path=valid[
                "tracker_gate_receipt_path"
            ],
        )


def test_approved_launcher_rejects_same_basename_at_another_allowed_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    valid = _request(profile, cwd=cwd, binary=binary)
    other_python = cwd / "other-bin" / "python"
    other_python.parent.mkdir()
    other_python.write_text("#!/bin/sh\n", encoding="utf-8")
    other_python.chmod(0o755)

    with pytest.raises(ValueError, match="exactly match"):
        dispatch.build_request(
            profile=profile,
            attempt_id="attempt-other-python",
            experiment_id=valid["experiment_id"],
            cwd=str(cwd),
            argv=[str(other_python), *valid["argv"][1:]],
            env={"OMP_NUM_THREADS": "1"},
            source_id=valid["source_id"],
            expected_duration_seconds=600,
            hard_deadline_seconds=900,
            approved_plan_path=valid["approved_plan_path"],
            tracker_gate_receipt_path=valid[
                "tracker_gate_receipt_path"
            ],
        )


def test_literal_python_launcher_maps_only_to_profile_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    plan, _gate = _write_launch_authority(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id="literal-python",
        experiment_id="experiment-001",
        build_envelope=False,
    )
    contract = _read_plan_contract(plan)
    contract["execution"]["launcher"][0] = "python"
    _write_plan_contract(plan, contract)
    smoke_receipt = (
        cwd
        / "results"
        / "experiment-001"
        / "preflight"
        / "smoke"
        / "receipt.json"
    )

    built = dispatch.build_preflight(
        **_preflight_arguments(
            profile=profile,
            cwd=cwd,
            binary=binary,
            plan=plan,
        ),
        smoke_receipt_path=str(smoke_receipt),
    )

    assert built["status"] == "built"
    other_python = cwd / "other-python" / "python"
    other_python.parent.mkdir()
    other_python.write_text("#!/bin/sh\n", encoding="utf-8")
    other_python.chmod(0o755)
    arguments = _preflight_arguments(
        profile=profile,
        cwd=cwd,
        binary=other_python,
        plan=plan,
    )
    with pytest.raises(ValueError, match="exact production argv"):
        dispatch.build_preflight(
            **arguments,
            smoke_receipt_path=str(smoke_receipt),
        )


@pytest.mark.parametrize("unsafe_name", ["PYTHONPATH", "LD_PRELOAD"])
def test_request_rejects_unsafe_environment_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_name: str,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    request["env"][unsafe_name] = str(tmp_path / "injected")

    with pytest.raises(ValueError, match="Expected env names"):
        dispatch.validate_request(request, profile=profile)


@pytest.mark.parametrize(
    "failure_mode",
    ["wrong_source", "required_check_failed", "wrong_env_sha"],
)
def test_preflight_envelope_mismatch_blocks_before_gpu_or_tmux(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    preflight_path = Path(request["preflight_receipt_path"])
    preflight = json.loads(
        preflight_path.read_text(encoding="utf-8")
    )
    if failure_mode == "wrong_source":
        preflight["source_id"] = "b" * 40
    elif failure_mode == "required_check_failed":
        preflight["checks"]["smoke"]["status"] = "failed"
    else:
        preflight["env"]["sha256"] = "0" * 64
    preflight_path.chmod(0o600)
    preflight_path.write_text(
        json.dumps(preflight, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    request["preflight_receipt_sha256"] = _sha256(preflight_path)
    runner = MainRunner()

    with pytest.raises(dispatch.DispatchError) as raised:
        dispatch.start_request(
            request,
            profile=profile,
            runner=runner,
        )

    assert raised.value.stage == "launch_authorization"
    assert raised.value.terminal is True
    assert not any(
        command[0] in {"nvidia-smi", "tmux"}
        for command in _commands(runner)
    )
    failure = json.loads(
        (
            Path(profile["state_root"])
            / request["attempt_id"]
            / "failure.json"
        ).read_text(encoding="utf-8")
    )
    assert failure["stage"] == "launch_authorization"
    assert failure["retry_authorized"] is False


def test_untracked_source_blocks_before_gpu_or_tmux(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    runner = MainRunner(git_status_stdout="?? stray-output.npz\n")

    with pytest.raises(dispatch.DispatchError) as raised:
        dispatch.start_request(
            request,
            profile=profile,
            runner=runner,
        )

    assert raised.value.stage == "source_identity"
    assert "untracked files" in str(raised.value)
    commands = _commands(runner)
    assert any(
        "status" in command and "--untracked-files=all" in command
        for command in commands
    )
    assert not any(command[0] == "nvidia-smi" for command in commands)
    assert not any(command[0] == "tmux" for command in commands)


def test_bound_gate_receipt_rejects_a_symlinked_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    preflight_path = Path(request["preflight_receipt_path"])
    preflight = json.loads(
        preflight_path.read_text(encoding="utf-8")
    )
    smoke_path = Path(
        preflight["checks"]["smoke"]["receipt_path"]
    )
    linked_parent = cwd / "linked-gate-parent"
    linked_parent.symlink_to(smoke_path.parent, target_is_directory=True)
    preflight["checks"]["smoke"]["receipt_path"] = str(
        linked_parent / smoke_path.name
    )
    preflight_path.chmod(0o600)
    preflight_path.write_text(
        json.dumps(preflight, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    request["preflight_receipt_sha256"] = _sha256(preflight_path)
    runner = MainRunner()

    with pytest.raises(dispatch.DispatchError) as raised:
        dispatch.start_request(
            request,
            profile=profile,
            runner=runner,
        )

    assert raised.value.terminal is True
    assert "symlink components" in str(raised.value)
    assert not any(
        command[0] in {"nvidia-smi", "tmux"}
        for command in _commands(runner)
    )
    failure = json.loads(
        (
            Path(profile["state_root"])
            / request["attempt_id"]
            / "failure.json"
        ).read_text(encoding="utf-8")
    )
    assert failure["terminal"] is True
    assert "symlink components" in failure["error"]


def test_ordinary_trex_status_failure_is_nonterminal_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, _cwd, _binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="trex",
    )
    calls: list[list[str]] = []

    def runner(
        command: Sequence[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        argv = list(command)
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv,
            255,
            stdout="",
            stderr="temporary transport failure",
        )

    with pytest.raises(dispatch.DispatchError) as raised:
        dispatch.status_request(
            profile=profile,
            attempt_id="attempt-status",
            runner=runner,
        )

    error = raised.value
    assert len(calls) == 1
    assert error.stage == "ssh_status"
    assert error.terminal is False
    assert error.launched_job_count is None
    assert error.as_dict()["retry_authorized"] is True
    assert "retry once" in error.smallest_next_action
    assert not Path(profile["state_root"]).exists()


def test_unexpected_post_arm_exception_persists_one_terminal_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    ordinary = MainRunner()

    def crashing_runner(
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "nvidia-smi":
            raise LookupError("injected unexpected failure")
        return ordinary(command, **kwargs)

    with pytest.raises(dispatch.DispatchError) as first:
        dispatch.start_request(
            request,
            profile=profile,
            runner=crashing_runner,
        )

    assert first.value.stage == "gpu_probe"
    assert first.value.terminal is True
    assert first.value.launched_job_count == 0
    assert "LookupError: injected unexpected failure" in str(first.value)
    attempt_root = Path(profile["state_root"]) / request["attempt_id"]
    failure_path = attempt_root / "failure.json"
    first_bytes = failure_path.read_bytes()
    failure = json.loads(first_bytes)
    assert failure["terminal"] is True
    assert failure["retry_authorized"] is False
    assert failure["stage"] == "gpu_probe"
    assert failure["last_valid_artifacts"]["request_path"] == str(
        attempt_root / "request.json"
    )

    def forbidden_runner(
        command: Sequence[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"terminal attempt reran command: {command!r}")

    with pytest.raises(dispatch.DispatchError) as second:
        dispatch.start_request(
            request,
            profile=profile,
            runner=forbidden_runner,
        )

    assert second.value.stage == "gpu_probe"
    assert failure_path.read_bytes() == first_bytes


@pytest.mark.skipif(
    not Path("/proc").is_dir(),
    reason="requires Linux child-subreaper support",
)
def test_successful_payload_with_double_fork_descendant_is_killed_and_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    descendant_pid_path = tmp_path / "double-fork-descendant.pid"
    descendant_code = (
        "import os,signal,time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(descendant_pid_path)!r}).write_text("
        "str(os.getpid()), encoding='utf-8')\n"
        "time.sleep(60)\n"
    )
    middle_code = (
        "import subprocess,sys\n"
        "subprocess.Popen("
        f"[sys.executable, '-c', {descendant_code!r}], "
        "start_new_session=True)\n"
    )
    binary.write_text(
        f"#!{sys.executable}\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "middle = subprocess.Popen("
        f"[sys.executable, '-c', {middle_code!r}], "
        "start_new_session=True)\n"
        "middle.wait()\n"
        "time.sleep(0.4)\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    attempt_root = Path(profile["state_root"]) / request["attempt_id"]
    request_path = attempt_root / "request.json"
    dispatch._atomic_create(request_path, request)
    descendant_pid: int | None = None
    try:
        returncode = dispatch._execute_payload(
            request_path,
            request_sha256=_sha256(request_path),
            profile=profile,
        )
        assert descendant_pid_path.is_file()
        descendant_pid = int(
            descendant_pid_path.read_text(encoding="utf-8")
        )
        assert returncode == 124
        assert not _pid_is_live(descendant_pid)
        assert not (attempt_root / "completion.json").exists()
        failure = json.loads(
            (attempt_root / "failure.json").read_text(encoding="utf-8")
        )
        assert failure["stage"] == "payload_execution"
        assert failure["terminal"] is True
        assert "returncode=0" in failure["error"]
        assert "timed_out=False" in failure["error"]
        assert "lingering_descendants=[" in failure["error"]
    finally:
        if descendant_pid is not None and _pid_is_live(descendant_pid):
            os.kill(descendant_pid, signal.SIGKILL)


def _wait_for_path(
    path: Path,
    *,
    timeout_seconds: float,
    process: subprocess.Popen[str] | None = None,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            return True
        if process is not None and process.poll() is not None:
            return False
        time.sleep(0.05)
    return path.is_file()


def _kill_test_process(pid: int) -> None:
    if not _pid_is_live(pid):
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.skipif(
    not Path("/proc").is_dir(),
    reason="requires Linux process groups and child-subreaper support",
)
def test_execute_worker_sigterm_cleans_payload_and_writes_bound_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(profile, cwd=cwd, binary=binary)
    payload_pid_path = tmp_path / "signal-payload.pid"
    child_pid_path = tmp_path / "signal-child.pid"
    ready_path = tmp_path / "signal-ready"
    child_code = (
        "import os,signal,time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(child_pid_path)!r}).write_text("
        "str(os.getpid()), encoding='utf-8')\n"
        "time.sleep(60)\n"
    )
    binary.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "from pathlib import Path\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        f"Path({str(payload_pid_path)!r}).write_text("
        "str(os.getpid()), encoding='utf-8')\n"
        "child = subprocess.Popen("
        f"[sys.executable, '-c', {child_code!r}], "
        "start_new_session=True)\n"
        f"ready = Path({str(ready_path)!r})\n"
        f"child_pid = Path({str(child_pid_path)!r})\n"
        "deadline = time.monotonic() + 5\n"
        "while not child_pid.is_file() and time.monotonic() < deadline:\n"
        "    time.sleep(0.02)\n"
        "ready.write_text('ready\\n', encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    attempt_root = Path(profile["state_root"]) / request["attempt_id"]
    request_path = attempt_root / "request.json"
    dispatch._atomic_create(request_path, request)
    request_sha256 = _sha256(request_path)
    worker: subprocess.Popen[str] | None = None
    payload_pid: int | None = None
    child_pid: int | None = None
    try:
        worker = subprocess.Popen(
            [
                sys.executable,
                str(REPO_ROOT / "experiments" / "local_dispatch.py"),
                "execute",
                "--profile",
                str(profile["_source_path"]),
                "--request",
                str(request_path),
                "--request-sha256",
                request_sha256,
            ],
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert _wait_for_path(
            ready_path,
            timeout_seconds=5,
            process=worker,
        )
        payload_pid = int(
            payload_pid_path.read_text(encoding="utf-8")
        )
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))

        worker.send_signal(signal.SIGTERM)
        stdout, stderr = worker.communicate(timeout=15)

        assert worker.returncode is not None and worker.returncode != 0, (
            stdout,
            stderr,
        )
        assert not _pid_is_live(payload_pid)
        assert not _pid_is_live(child_pid)
        failure_path = attempt_root / "failure.json"
        assert failure_path.is_file()
        failure_bytes = failure_path.read_bytes()
        failure = json.loads(failure_bytes)
        assert failure["schema_version"] == dispatch.FAILURE_SCHEMA
        assert failure["terminal"] is True
        assert failure["retry_authorized"] is False
        assert failure["attempt_id"] == request["attempt_id"]
        assert failure["experiment_id"] == request["experiment_id"]
        assert failure["target"] == request["target"]
        assert failure["launched_job_count"] == 1
        assert failure["last_valid_artifacts"]["request_path"] == str(
            request_path
        )
        assert failure["last_valid_artifacts"]["request_sha256"] == (
            request_sha256
        )
        assert failure_path.stat().st_mode & 0o222 == 0
        time.sleep(0.05)
        assert failure_path.read_bytes() == failure_bytes
    finally:
        if worker is not None and worker.poll() is None:
            _kill_test_process(worker.pid)
            try:
                worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        for pid in (payload_pid, child_pid):
            if pid is not None:
                _kill_test_process(pid)


@pytest.mark.skipif(
    not Path("/proc").is_dir(),
    reason="requires Linux child-subreaper support",
)
def test_cleanup_kills_descendant_spawned_by_term_handler_during_grace(
    tmp_path: Path,
) -> None:
    ready_path = tmp_path / "term-handler-ready"
    late_pid_path = tmp_path / "late-descendant.pid"
    child_code = (
        "import os,signal,time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(late_pid_path)!r}).write_text("
        "str(os.getpid()), encoding='utf-8')\n"
        "time.sleep(60)\n"
    )
    root_script = tmp_path / "term-handler-root.py"
    root_script.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "from pathlib import Path\n"
        "import signal\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "def stop(_signum, _frame):\n"
        "    subprocess.Popen("
        f"[sys.executable, '-c', {child_code!r}], "
        "start_new_session=True)\n"
        "    time.sleep(0.15)\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        f"Path({str(ready_path)!r}).write_text("
        "'ready\\n', encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    root_script.chmod(0o755)
    try:
        previous_subreaper = dispatch._set_child_subreaper(True)
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"child subreaper unavailable: {exc}")
    baseline = dispatch._descendant_identities(os.getpid())
    process: subprocess.Popen[bytes] | None = None
    late_pid: int | None = None
    try:
        process = subprocess.Popen(
            [str(root_script)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        assert _wait_for_path(ready_path, timeout_seconds=5)
        assert not late_pid_path.exists()
        started = time.monotonic()

        returncode = dispatch._terminate_payload(
            process,
            identities={},
            baseline_descendants=baseline,
        )

        elapsed = time.monotonic() - started
        assert returncode == 0
        assert late_pid_path.is_file()
        late_pid = int(late_pid_path.read_text(encoding="utf-8"))
        assert not _pid_is_live(late_pid)
        assert elapsed < 3
    finally:
        if process is not None and process.poll() is None:
            _kill_test_process(process.pid)
        if late_pid_path.is_file():
            late_pid = int(late_pid_path.read_text(encoding="utf-8"))
        if late_pid is not None:
            _kill_test_process(late_pid)
            try:
                os.waitpid(late_pid, os.WNOHANG)
            except (ChildProcessError, ProcessLookupError):
                pass
        dispatch._set_child_subreaper(previous_subreaper)


def test_failure_receipts_reserve_or_release_target_lease_by_quiescence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    state_root = Path(profile["state_root"])

    def persist_failure(
        attempt_id: str,
        *,
        launched_job_count: int | None,
        stage: str,
    ) -> None:
        request = _request(
            profile,
            cwd=cwd,
            binary=binary,
            attempt_id=attempt_id,
        )
        request_path = state_root / attempt_id / "request.json"
        dispatch._atomic_create(request_path, request)
        error = dispatch.DispatchError(
            stage,
            f"terminal {stage} fixture",
            launched_job_count=launched_job_count,
        ).bind_request(
            request,
            last_valid_artifacts={
                "request_path": str(request_path),
                "request_sha256": _sha256(request_path),
            },
        )
        dispatch._atomic_create(
            request_path.parent / "failure.json",
            error.as_dict(),
        )

    persist_failure(
        "lease-unknown",
        launched_job_count=None,
        stage="tmux_dispatch",
    )
    persist_failure(
        "lease-launched",
        launched_job_count=1,
        stage="post_dispatch_readback",
    )
    persist_failure(
        "lease-prelaunch",
        launched_job_count=0,
        stage="gpu_busy",
    )

    worker_request = _request(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id="lease-worker-failed",
    )
    binary.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
    binary.chmod(0o755)
    worker_request_path = (
        state_root / "lease-worker-failed" / "request.json"
    )
    dispatch._atomic_create(worker_request_path, worker_request)
    assert dispatch._execute_payload(
        worker_request_path,
        request_sha256=_sha256(worker_request_path),
        profile=profile,
    ) == 7
    worker_failure = json.loads(
        (
            worker_request_path.parent / "failure.json"
        ).read_text(encoding="utf-8")
    )
    assert worker_failure["stage"] == "payload_execution"

    assert dispatch._active_attempts(
        state_root,
        exclude="new-attempt",
    ) == ["lease-launched", "lease-unknown"]


def test_public_start_lock_timeout_is_terminal_and_persists_bound_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    request = _request(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id="lock-after-start",
    )
    state_root = Path(profile["state_root"])
    state_root.mkdir(parents=True)
    lock_path = state_root / ".dispatch.lock"
    monkeypatch.setattr(dispatch, "LOCK_TIMEOUT_SECONDS", 0.1)

    with lock_path.open("a+", encoding="utf-8") as held_lock:
        fcntl.flock(held_lock.fileno(), fcntl.LOCK_EX)
        with pytest.raises(dispatch.DispatchError) as raised:
            dispatch.start_request(
                request,
                profile=profile,
                runner=lambda *_args, **_kwargs: pytest.fail(
                    "lock timeout must precede external commands"
                ),
            )

    assert raised.value.stage == "dispatch_lock_timeout"
    assert raised.value.terminal is True
    assert raised.value.launched_job_count == 0
    attempt_root = state_root / request["attempt_id"]
    request_path = attempt_root / "request.json"
    failure_path = attempt_root / "failure.json"
    assert request_path.is_file()
    assert failure_path.is_file()
    first_bytes = failure_path.read_bytes()
    failure = json.loads(first_bytes)
    assert failure["terminal"] is True
    assert failure["retry_authorized"] is False
    assert failure["last_valid_artifacts"]["request_path"] == str(
        request_path
    )
    assert failure["last_valid_artifacts"]["request_sha256"] == _sha256(
        request_path
    )

    with pytest.raises(dispatch.DispatchError) as repeated:
        dispatch.start_request(
            request,
            profile=profile,
            runner=lambda *_args, **_kwargs: pytest.fail(
                "terminal lock failure must not run commands"
            ),
        )

    assert repeated.value.stage == "dispatch_lock_timeout"
    assert failure_path.read_bytes() == first_bytes


def test_build_and_verify_preflight_v2_have_zero_launch_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    plan, _gate = _write_launch_authority(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id="preflight-build",
        experiment_id="experiment-001",
        build_envelope=False,
    )
    smoke_receipt = (
        cwd
        / "results"
        / "experiment-001"
        / "preflight"
        / "smoke"
        / "receipt.json"
    )
    state_root = Path(profile["state_root"])
    assert not state_root.exists()

    built = dispatch.build_preflight(
        **_preflight_arguments(
            profile=profile,
            cwd=cwd,
            binary=binary,
            plan=plan,
        ),
        smoke_receipt_path=str(smoke_receipt),
    )

    assert built["status"] == "built"
    assert built["side_effects"] == 1
    assert built["envelope"]["schema_version"] == dispatch.PREFLIGHT_SCHEMA
    assert built["envelope"]["env"] == _environment_binding(
        profile,
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
        },
    )
    smoke = built["envelope"]["checks"]["smoke"]
    assert smoke["schema_version"] == (
        "mnist-conv-perfectdiode-preflight-receipt/v1"
    )
    assert smoke["validator_id"] == "perfectdiode-hparam-smoke/v1"
    assert smoke["claims"]["study_id"] == "experiment-001"
    assert smoke["claims"]["host"] == "akib"
    assert smoke["claims"]["architecture"] == "conv1"
    for role in ("tk_reference", "scheduled_run_preflight"):
        assert built["envelope"]["checks"][role] == {
            "required": False,
            "status": "not_required",
            "schema_version": None,
            "validator_id": None,
            "receipt_path": None,
            "receipt_sha256": None,
            "claims": None,
        }
    envelope_path = Path(built["preflight_receipt_path"])
    before = envelope_path.read_bytes()

    verified = dispatch.verify_preflight(
        **_preflight_arguments(
            profile=profile,
            cwd=cwd,
            binary=binary,
            plan=plan,
        )
    )

    assert verified["status"] == "verified"
    assert verified["side_effects"] == 0
    assert envelope_path.read_bytes() == before
    assert envelope_path.stat().st_mode & 0o222 == 0
    assert not state_root.exists()
    assert not list(cwd.rglob("dispatch.json"))
    assert not list(cwd.rglob("running.json"))


def test_scheduled_preflight_accepts_the_official_shell_producer_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    plan, _gate = _write_launch_authority(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id="scheduled-producer",
        experiment_id="experiment-001",
        build_envelope=False,
    )
    contract = _read_plan_contract(plan)
    preflight = contract["execution"]["preflight"]
    output_root = cwd / "results" / "experiment-001"
    scheduled_path = output_root / "preflight" / "scheduled" / "receipt.json"
    preflight["scheduled_run_preflight_required"] = True
    preflight["receipt_schemas"]["scheduled_run_preflight"] = (
        "scheduled-run-preflight-receipt/v1"
    )
    preflight["receipt_bindings"]["scheduled_run_preflight"] = {
        "receipt_path": str(scheduled_path.relative_to(cwd)),
        "subject_id": None,
        "producer_source_id": None,
    }
    _write_plan_contract(plan, contract)
    scheduled_receipt, runner = _write_scheduled_preflight(
        output_root=output_root,
        cwd=cwd,
    )
    smoke_receipt = output_root / "preflight" / "smoke" / "receipt.json"

    built = dispatch.build_preflight(
        **_preflight_arguments(
            profile=profile,
            cwd=cwd,
            binary=binary,
            plan=plan,
        ),
        smoke_receipt_path=str(smoke_receipt),
        scheduled_run_preflight_receipt_path=str(scheduled_receipt),
    )

    scheduled = built["envelope"]["checks"]["scheduled_run_preflight"]
    assert scheduled["status"] == "passed"
    assert scheduled["claims"]["experiment_id"] == "experiment-001"
    assert scheduled["claims"]["source_id"] == "a" * 40
    assert scheduled["claims"]["target"] == "main"
    assert scheduled["claims"]["output_root"] == str(output_root)
    assert scheduled["claims"]["runner"] == str(runner)
    assert scheduled["claims"]["preflight_command"] == [
        "/bin/bash",
        "-euo",
        "pipefail",
        str(runner),
        "--preflight",
    ]
    before = Path(built["preflight_receipt_path"]).read_bytes()
    verified = dispatch.verify_preflight(
        **_preflight_arguments(
            profile=profile,
            cwd=cwd,
            binary=binary,
            plan=plan,
        )
    )
    assert verified["status"] == "verified"
    assert verified["side_effects"] == 0
    assert Path(built["preflight_receipt_path"]).read_bytes() == before


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("extra_argument", "official producer command"),
        ("stale", "fresh scheduled-run preflight"),
        ("outside_output", "approved-plan binding"),
    ],
)
def test_scheduled_preflight_rejects_unbound_or_stale_producer_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    message: str,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    plan, _gate = _write_launch_authority(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id=f"scheduled-{drift}",
        experiment_id="experiment-001",
        build_envelope=False,
    )
    contract = _read_plan_contract(plan)
    preflight = contract["execution"]["preflight"]
    output_root = cwd / "results" / "experiment-001"
    scheduled_path = output_root / "preflight" / "scheduled" / "receipt.json"
    preflight["scheduled_run_preflight_required"] = True
    preflight["receipt_schemas"]["scheduled_run_preflight"] = (
        "scheduled-run-preflight-receipt/v1"
    )
    preflight["receipt_bindings"]["scheduled_run_preflight"] = {
        "receipt_path": str(scheduled_path.relative_to(cwd)),
        "subject_id": None,
        "producer_source_id": None,
    }
    _write_plan_contract(plan, contract)
    receipt_root = (
        cwd / "unbound-scheduled"
        if drift == "outside_output"
        else None
    )
    scheduled_receipt, _runner = _write_scheduled_preflight(
        output_root=output_root,
        cwd=cwd,
        receipt_root=receipt_root,
    )
    if drift in {"extra_argument", "stale"}:
        value = json.loads(scheduled_receipt.read_text(encoding="utf-8"))
        if drift == "extra_argument":
            value["preflight_command"].append("--extra")
        else:
            value["timestamp_utc"] = (
                datetime.now(timezone.utc) - timedelta(minutes=11)
            ).isoformat()
        scheduled_receipt.write_text(
            json.dumps(value, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    smoke_receipt = output_root / "preflight" / "smoke" / "receipt.json"

    with pytest.raises(ValueError, match=message):
        dispatch.build_preflight(
            **_preflight_arguments(
                profile=profile,
                cwd=cwd,
                binary=binary,
                plan=plan,
            ),
            smoke_receipt_path=str(smoke_receipt),
            scheduled_run_preflight_receipt_path=str(scheduled_receipt),
        )

    assert not (output_root / "preflight" / "receipt.json").exists()


def test_tk_preflight_rederivation_is_read_only_and_never_calls_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    plan, _gate = _write_launch_authority(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id="tk-read-only",
        experiment_id="experiment-001",
        build_envelope=False,
    )
    contract = _read_plan_contract(plan)
    preflight = contract["execution"]["preflight"]
    output_root = cwd / "results" / "experiment-001"
    upstream_study_id = "tkstudy-upstream-001"
    upstream_source_id = "b" * 40
    selection_path = output_root / "preflight" / "tk" / "selection.json"
    preflight["tk_reference_required"] = True
    preflight["receipt_schemas"]["tk_reference"] = (
        "mnist-conv-perfectdiode-tk-selection/v1"
    )
    preflight["receipt_bindings"]["tk_reference"] = {
        "receipt_path": str(selection_path.relative_to(cwd)),
        "subject_id": upstream_study_id,
        "producer_source_id": upstream_source_id,
    }
    _write_plan_contract(plan, contract)
    selection = _write_producer_tk_selection(
        monkeypatch,
        output_root=output_root,
        study_id=upstream_study_id,
        source_id=upstream_source_id,
    )
    smoke_receipt = output_root / "preflight" / "smoke" / "receipt.json"

    built = dispatch.build_preflight(
        **_preflight_arguments(
            profile=profile,
            cwd=cwd,
            binary=binary,
            plan=plan,
        ),
        smoke_receipt_path=str(smoke_receipt),
        tk_reference_receipt_path=str(selection),
    )

    tk_check = built["envelope"]["checks"]["tk_reference"]
    assert tk_check["status"] == "passed"
    assert tk_check["claims"]["study_id"] == upstream_study_id
    assert tk_check["claims"]["study_id"] != "experiment-001"
    assert tk_check["claims"]["producer_source_id"] == upstream_source_id
    assert tk_check["claims"]["producer_source_id"] != "a" * 40
    assert tk_check["claims"]["row_count"] == 3
    envelope_path = Path(built["preflight_receipt_path"])
    before = {
        str(path.relative_to(output_root)): (
            path.stat().st_mode,
            path.read_bytes(),
        )
        for path in output_root.rglob("*")
        if path.is_file()
    }
    verified = dispatch.verify_preflight(
        **_preflight_arguments(
            profile=profile,
            cwd=cwd,
            binary=binary,
            plan=plan,
        )
    )
    after = {
        str(path.relative_to(output_root)): (
            path.stat().st_mode,
            path.read_bytes(),
        )
        for path in output_root.rglob("*")
        if path.is_file()
    }
    assert verified["status"] == "verified"
    assert verified["side_effects"] == 0
    assert envelope_path.stat().st_mode & 0o222 == 0
    assert after == before


@pytest.mark.parametrize(
    "drift",
    ["receipt_path", "subject_id", "producer_source_id"],
)
def test_tk_preflight_rejects_approved_path_subject_or_source_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    plan, _gate = _write_launch_authority(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id=f"tk-binding-{drift}",
        experiment_id="experiment-001",
        build_envelope=False,
    )
    output_root = cwd / "results" / "experiment-001"
    upstream_study_id = "tkstudy-upstream-001"
    upstream_source_id = "b" * 40
    selection = _write_producer_tk_selection(
        monkeypatch,
        output_root=output_root,
        study_id=upstream_study_id,
        source_id=upstream_source_id,
    )
    contract = _read_plan_contract(plan)
    preflight = contract["execution"]["preflight"]
    preflight["tk_reference_required"] = True
    preflight["receipt_schemas"]["tk_reference"] = (
        "mnist-conv-perfectdiode-tk-selection/v1"
    )
    preflight["receipt_bindings"]["tk_reference"] = {
        "receipt_path": str(selection.relative_to(cwd)),
        "subject_id": upstream_study_id,
        "producer_source_id": upstream_source_id,
    }
    if drift == "receipt_path":
        preflight["receipt_bindings"]["tk_reference"]["receipt_path"] = (
            "results/experiment-001/preflight/tk/other-selection.json"
        )
    else:
        if drift == "subject_id":
            preflight["receipt_bindings"]["tk_reference"]["subject_id"] = (
                "tkstudy-wrong-001"
            )
        else:
            preflight["receipt_bindings"]["tk_reference"][
                "producer_source_id"
            ] = "c" * 40
    _write_plan_contract(plan, contract)
    smoke_receipt = output_root / "preflight" / "smoke" / "receipt.json"

    with pytest.raises(ValueError) as raised:
        dispatch.build_preflight(
            **_preflight_arguments(
                profile=profile,
                cwd=cwd,
                binary=binary,
                plan=plan,
            ),
            smoke_receipt_path=str(smoke_receipt),
            tk_reference_receipt_path=str(selection),
        )

    if drift == "receipt_path":
        assert "approved-plan binding" in str(raised.value)
    else:
        assert "fully revalidated selected T/K reference" in str(
            raised.value
        )
    assert not (output_root / "preflight" / "receipt.json").exists()


def test_smoke_producer_source_must_equal_downstream_launch_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    plan, _gate = _write_launch_authority(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id="smoke-source-drift",
        experiment_id="experiment-001",
        build_envelope=False,
    )
    contract = _read_plan_contract(plan)
    contract["execution"]["preflight"]["receipt_bindings"]["smoke"][
        "producer_source_id"
    ] = "b" * 40
    _write_plan_contract(plan, contract)
    smoke_receipt = (
        cwd
        / "results"
        / "experiment-001"
        / "preflight"
        / "smoke"
        / "receipt.json"
    )

    with pytest.raises(ValueError, match="downstream launch source_id"):
        dispatch.build_preflight(
            **_preflight_arguments(
                profile=profile,
                cwd=cwd,
                binary=binary,
                plan=plan,
            ),
            smoke_receipt_path=str(smoke_receipt),
        )


def test_legacy_plan_without_receipt_bindings_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    plan, _gate = _write_launch_authority(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id="legacy-receipt-bindings",
        experiment_id="experiment-001",
        build_envelope=False,
    )
    contract = _read_plan_contract(plan)
    del contract["execution"]["preflight"]["receipt_bindings"]
    _write_plan_contract(plan, contract)
    smoke_receipt = (
        cwd
        / "results"
        / "experiment-001"
        / "preflight"
        / "smoke"
        / "receipt.json"
    )

    with pytest.raises(ValueError, match="receipt_bindings"):
        dispatch.build_preflight(
            **_preflight_arguments(
                profile=profile,
                cwd=cwd,
                binary=binary,
                plan=plan,
            ),
            smoke_receipt_path=str(smoke_receipt),
        )


def test_canonical_plan_validator_requires_exact_receipt_bindings() -> None:
    validator_path = (
        REPO_ROOT
        / "skills"
        / "run-experiment-pipeline"
        / "scripts"
        / "validate_experiment_plan.py"
    )
    namespace = runpy.run_path(str(validator_path))
    validate = namespace["validate_preflight_contract"]
    plan_error = namespace["PlanError"]
    preflight = {
        "smoke_required": True,
        "tk_reference_required": True,
        "scheduled_run_preflight_required": False,
        "receipt_path": "results/downstream/preflight/receipt.json",
        "receipt_schemas": {
            "smoke": "producer-smoke/v1",
            "tk_reference": "producer-tk/v1",
            "scheduled_run_preflight": None,
        },
        "receipt_bindings": {
            "smoke": {
                "receipt_path": (
                    "results/downstream/preflight/smoke/receipt.json"
                ),
                "subject_id": "smoke-upstream-001",
                "producer_source_id": "a" * 40,
            },
            "tk_reference": {
                "receipt_path": (
                    "results/downstream/preflight/tk/selection.json"
                ),
                "subject_id": "tkstudy-upstream-001",
                "producer_source_id": "b" * 40,
            },
            "scheduled_run_preflight": None,
        },
    }

    outer, bindings = validate(copy.deepcopy(preflight))

    assert outer == Path("results/downstream/preflight/receipt.json")
    assert bindings["tk_reference"]["subject_id"] == "tkstudy-upstream-001"
    assert bindings["tk_reference"]["producer_source_id"] == "b" * 40
    legacy = copy.deepcopy(preflight)
    del legacy["receipt_bindings"]
    with pytest.raises(plan_error, match="receipt_bindings"):
        validate(legacy)
    unsafe = copy.deepcopy(preflight)
    unsafe["receipt_bindings"]["tk_reference"]["subject_id"] = "../other"
    with pytest.raises(plan_error, match="filesystem-safe"):
        validate(unsafe)
    wrong_source = copy.deepcopy(preflight)
    wrong_source["receipt_bindings"]["tk_reference"][
        "producer_source_id"
    ] = "b" * 39
    with pytest.raises(plan_error, match="40-character"):
        validate(wrong_source)


def test_preflight_plan_rejects_unknown_receipt_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    plan, _gate = _write_launch_authority(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id="unknown-schema",
        experiment_id="experiment-001",
        build_envelope=False,
    )
    contract = _read_plan_contract(plan)
    contract["execution"]["preflight"]["receipt_schemas"]["smoke"] = (
        "unknown-smoke/v99"
    )
    _write_plan_contract(plan, contract)
    smoke_receipt = (
        cwd
        / "results"
        / "experiment-001"
        / "preflight"
        / "smoke"
        / "receipt.json"
    )

    with pytest.raises(ValueError, match="supported schema"):
        dispatch.build_preflight(
            **_preflight_arguments(
                profile=profile,
                cwd=cwd,
                binary=binary,
                plan=plan,
            ),
            smoke_receipt_path=str(smoke_receipt),
        )

    assert not Path(profile["state_root"]).exists()


def test_preflight_rejects_receipt_schema_mismatched_to_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    plan, _gate = _write_launch_authority(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id="mismatched-schema",
        experiment_id="experiment-001",
        build_envelope=False,
    )
    smoke_receipt = (
        cwd
        / "results"
        / "experiment-001"
        / "preflight"
        / "smoke"
        / "receipt.json"
    )
    smoke = json.loads(smoke_receipt.read_text(encoding="utf-8"))
    smoke["schema_version"] = "test-smoke/v1"
    smoke_receipt.write_text(
        json.dumps(smoke, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="receipt schema"):
        dispatch.build_preflight(
            **_preflight_arguments(
                profile=profile,
                cwd=cwd,
                binary=binary,
                plan=plan,
            ),
            smoke_receipt_path=str(smoke_receipt),
        )

    assert not Path(profile["state_root"]).exists()


@pytest.mark.parametrize(
    ("required", "schema", "message"),
    [
        (True, None, "required preflight role"),
        (
            False,
            "mnist-conv-perfectdiode-tk-selection/v1",
            "nonrequired preflight role",
        ),
    ],
)
def test_preflight_required_schema_null_matrix_rejects_invalid_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    required: bool,
    schema: str | None,
    message: str,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    plan, _gate = _write_launch_authority(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id="schema-matrix",
        experiment_id="experiment-001",
        build_envelope=False,
    )
    contract = _read_plan_contract(plan)
    preflight = contract["execution"]["preflight"]
    preflight["tk_reference_required"] = required
    preflight["receipt_schemas"]["tk_reference"] = schema
    _write_plan_contract(plan, contract)
    smoke_receipt = (
        cwd
        / "results"
        / "experiment-001"
        / "preflight"
        / "smoke"
        / "receipt.json"
    )

    with pytest.raises(ValueError, match=message):
        dispatch.build_preflight(
            **_preflight_arguments(
                profile=profile,
                cwd=cwd,
                binary=binary,
                plan=plan,
            ),
            smoke_receipt_path=str(smoke_receipt),
        )


def test_preflight_rejects_receipt_for_nonrequired_null_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    plan, _gate = _write_launch_authority(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id="unexpected-role",
        experiment_id="experiment-001",
        build_envelope=False,
    )
    smoke_receipt = (
        cwd
        / "results"
        / "experiment-001"
        / "preflight"
        / "smoke"
        / "receipt.json"
    )

    with pytest.raises(ValueError, match="no receipt for nonrequired role"):
        dispatch.build_preflight(
            **_preflight_arguments(
                profile=profile,
                cwd=cwd,
                binary=binary,
                plan=plan,
            ),
            smoke_receipt_path=str(smoke_receipt),
            tk_reference_receipt_path=str(smoke_receipt),
        )


@pytest.mark.parametrize("drift", ["claims", "output_artifact"])
def test_verify_preflight_revalidates_exact_claims_and_live_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    plan, _gate = _write_launch_authority(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id="claims-drift",
        experiment_id="experiment-001",
    )
    envelope_path = (
        cwd / "results" / "experiment-001" / "preflight" / "receipt.json"
    )
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    if drift == "claims":
        envelope["checks"]["smoke"]["claims"]["completed_steps"] = 999
        envelope_path.chmod(0o600)
        envelope_path.write_text(
            json.dumps(envelope, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        artifact_path = Path(
            envelope["checks"]["smoke"]["claims"]["output_artifacts"][0]
        )
        artifact_path.write_text("drifted\n", encoding="utf-8")

    with pytest.raises(ValueError):
        dispatch.verify_preflight(
            **_preflight_arguments(
                profile=profile,
                cwd=cwd,
                binary=binary,
                plan=plan,
            )
        )

    assert not Path(profile["state_root"]).exists()


def test_preflight_first_write_is_idempotent_and_conflicts_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, cwd, binary = _write_profile(
        tmp_path,
        monkeypatch,
        target="main",
    )
    plan, _gate = _write_launch_authority(
        profile,
        cwd=cwd,
        binary=binary,
        attempt_id="preflight-idempotent",
        experiment_id="experiment-001",
        build_envelope=False,
    )
    output_root = cwd / "results" / "experiment-001"
    smoke_receipt = output_root / "preflight" / "smoke" / "receipt.json"
    arguments = _preflight_arguments(
        profile=profile,
        cwd=cwd,
        binary=binary,
        plan=plan,
    )

    first = dispatch.build_preflight(
        **arguments,
        smoke_receipt_path=str(smoke_receipt),
    )
    envelope_path = Path(first["preflight_receipt_path"])
    first_bytes = envelope_path.read_bytes()
    second = dispatch.build_preflight(
        **arguments,
        smoke_receipt_path=str(smoke_receipt),
    )

    assert first["status"] == "built"
    assert first["side_effects"] == 1
    assert second["status"] == "already_built"
    assert second["side_effects"] == 0
    assert envelope_path.read_bytes() == first_bytes
    assert envelope_path.stat().st_mode & 0o222 == 0

    alternate_receipt, _artifact = _write_perfectdiode_smoke(
        output_root=output_root,
        experiment_id="experiment-001",
        target="main",
        source_id="a" * 40,
        fixture_name="alternate-smoke",
    )
    with pytest.raises(ValueError, match="approved-plan binding"):
        dispatch.build_preflight(
            **arguments,
            smoke_receipt_path=str(alternate_receipt),
        )

    assert envelope_path.read_bytes() == first_bytes
    assert not Path(profile["state_root"]).exists()
