#!/usr/bin/env python3
"""Run one bounded fail-closed perfect-diode successor reconciliation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mnist_conv.io import atomic_write_json, read_json
from experiments.mnist_conv.identity import canonical_json_bytes
from experiments.submit_mnist_conv_perfectdiode_successor_confirmation_jeanzay import (
    ACCOUNT,
    CONSTRAINT,
    DEFAULT_CANARY_PACK_INDEX,
    LOGICAL_ENTRY_COUNT,
    PARTITION,
    PRODUCTION_TASK_COUNT,
    PYTHON_EXECUTABLE,
    QOS,
    _append_exports,
    _canonical_json_sha256,
    _error,
    _parse_result_marker,
    _submitted_job_id,
    _validate_live_allocation,
    append_launch_template_exports,
    canary_gate_binding,
    ensure_preflight_receipt,
    ensure_scheduled_preflight_receipt,
    immutable_input_binding,
    launch_contract_value,
    materialize_canary_gate_command,
    materialize_launch_command,
    preflight_receipt_value,
    prepare_submission,
    run_wrapper_preflight,
    sha256_file,
    validate_launch_authorization_receipt,
    verify_login_umg_v100_allocation,
    verify_submitted_job_contract,
)
from experiments.verify_mnist_conv_perfectdiode_successor_canary import (
    OFFICIAL_CANARY_VERIFIER,
    validate_canary_gate_receipt,
    verify_canary,
)


STATE_SCHEMA_VERSION = (
    "perfectdiode-successor-jeanzay-supervisor/v1"
)
FAILURE_REPORT_SCHEMA_VERSION = (
    "perfectdiode-successor-jeanzay-failure-report/v1"
)
FAILURE_JSON_MARKER = "PD_SUCCESSOR_FAILURE_JSON="
EXPERIMENT_ID = (
    "perfectdiode-conv12-best-observed-confirmation-20260727-v1"
)
TRACKER_VALIDATOR_RELATIVE = Path(
    "skills/run-experiment-pipeline/scripts/"
    "validate_current_experiments.py"
)
TRACKER_RELATIVE = Path("docs/current_experiments.md")
TRACKER_GATE_SCHEMA_VERSION = "current-experiments-launch-gate/v1"
TRACKER_VALIDATOR_SHA256 = (
    "4d82bbb61792476c085942822190b43542aebd3f655b008cc992faafd04f5215"
)
TRACKER_GATE_MAX_AGE_SECONDS = 15 * 60
MAX_ARMED_COMMAND_TIMEOUT_SECONDS = 60.0
DEFAULT_COMMAND_TIMEOUT_SECONDS = MAX_ARMED_COMMAND_TIMEOUT_SECONDS
DEFAULT_MAX_FOLLOW_SECONDS = 3600.0
LONG_RUN_EXPECTED_MAX_SECONDS = 8 * 60 * 60
LONG_RUN_HARD_DEADLINE_SECONDS = 8 * 60 * 60
PROCESS_GROUP_TERMINATE_SECONDS = 2.0
STATUS_SCHEMA_VERSION = "mnist-conv-perfectdiode-successor-status/v1"
COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-successor-entry-completion/v1"
)
SLURM_ACTIVE_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESIZING",
    "RUNNING",
    "SIGNALING",
    "STAGE_OUT",
    "SUSPENDED",
}
SLURM_FAILED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "STOPPED",
    "TIMEOUT",
}


def _strict_json_bytes(payload: bytes, *, label: str) -> Any:
    """Parse exactly the bytes that were hash-checked."""

    def reject_constant(value: str) -> None:
        raise ValueError(
            f"Expected {label} to use strict JSON without NaN or Infinity. "
            f"Provided value: {value!r}."
        )

    def unique_object(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(
                    f"Expected {label} JSON keys to be unique. Provided "
                    f"duplicate key: {key!r}."
                )
            value[key] = item
        return value

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Expected {label} to be UTF-8 JSON. Provided value: {exc}."
        ) from exc
    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _read_regular_file_once(path: Path, *, label: str) -> bytes:
    """Open without following symlinks and return one immutable snapshot."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(
            f"Expected {label} to be an existing non-symlink file. "
            f"Provided value: {path!r}; error={exc!r}."
        ) from exc
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"Expected {label} to be a regular file. Provided value: "
                f"{path!r}."
            )
        return handle.read()


def _positive_timeout(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 1 <= float(value) <= 24 * 60 * 60
    ):
        raise _error(f"{label} to be a number from 1 through 86400", value)
    return float(value)


def _bounded_follow_timeout(value: Any) -> float:
    timeout = _positive_timeout(value, label="max-follow-seconds")
    if timeout > LONG_RUN_HARD_DEADLINE_SECONDS:
        raise _error(
            "max-follow-seconds to be no greater than the declared "
            f"{LONG_RUN_HARD_DEADLINE_SECONDS}-second long-run hard deadline",
            value,
        )
    return timeout


def _bounded_command_timeout(value: Any) -> float:
    timeout = _positive_timeout(value, label="command-timeout-seconds")
    if timeout > MAX_ARMED_COMMAND_TIMEOUT_SECONDS:
        raise _error(
            "command-timeout-seconds to be no greater than the armed "
            f"{MAX_ARMED_COMMAND_TIMEOUT_SECONDS:g}-second observability "
            "interval",
            value,
        )
    return timeout


def _run_process_group(
    command: Sequence[str],
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Run one command in a new session and reap its whole group on error."""

    timeout = kwargs.pop("timeout", None)
    check = bool(kwargs.pop("check", False))
    input_value = kwargs.pop("input", None)
    capture_output = bool(kwargs.pop("capture_output", False))
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError(
                "Expected capture_output not to be combined with stdout or "
                f"stderr. Provided value: {kwargs!r}."
            )
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    kwargs["start_new_session"] = True
    process = subprocess.Popen(list(command), **kwargs)

    def signal_group(value: int) -> str | None:
        try:
            os.killpg(process.pid, value)
        except ProcessLookupError:
            return None
        except OSError as exc:
            return f"{signal.Signals(value).name}: {exc!r}"
        return None

    def group_is_live() -> bool:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            # Permission errors still mean the process group exists.
            return True
        return True

    def proc_snapshot() -> dict[int, tuple[int, str, str]]:
        """Return pid -> (ppid, state, starttime) for identity-safe cleanup."""

        snapshot: dict[int, tuple[int, str, str]] = {}
        proc_root = Path("/proc")
        if not proc_root.is_dir():
            return snapshot
        for candidate in proc_root.iterdir():
            if not candidate.name.isdigit():
                continue
            try:
                value = (candidate / "stat").read_text(
                    encoding="utf-8"
                )
                fields = value[value.rfind(")") + 2 :].split()
                if len(fields) < 20:
                    continue
                snapshot[int(candidate.name)] = (
                    int(fields[1]),
                    fields[0],
                    fields[19],
                )
            except (FileNotFoundError, PermissionError, ValueError):
                continue
        return snapshot

    def descendant_identities() -> dict[int, str]:
        snapshot = proc_snapshot()
        descendants: dict[int, str] = {}
        pending = [process.pid]
        while pending:
            parent = pending.pop()
            children = [
                pid
                for pid, (ppid, _state, _starttime) in snapshot.items()
                if ppid == parent and pid not in descendants
            ]
            for child in children:
                descendants[child] = snapshot[child][2]
                pending.append(child)
        return descendants

    def signal_descendants(
        identities: Mapping[int, str],
        value: int,
    ) -> list[str]:
        errors: list[str] = []
        snapshot = proc_snapshot()
        for pid, starttime in identities.items():
            current = snapshot.get(pid)
            if current is None or current[2] != starttime or current[1] == "Z":
                continue
            try:
                os.kill(pid, value)
            except ProcessLookupError:
                continue
            except OSError as exc:
                errors.append(
                    f"{signal.Signals(value).name} pid {pid}: {exc!r}"
                )
        return errors

    def stop_group() -> tuple[Any, Any, list[str]]:
        cleanup_errors: list[str] = []
        descendants = descendant_identities()
        term_error = signal_group(signal.SIGTERM)
        if term_error is not None:
            cleanup_errors.append(term_error)
        cleanup_errors.extend(
            signal_descendants(descendants, signal.SIGTERM)
        )
        try:
            output = process.communicate(
                timeout=PROCESS_GROUP_TERMINATE_SECONDS
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.output, exc.stderr)
        # Always send SIGKILL: the leader may have exited while a descendant
        # ignored SIGTERM and retained the process group.
        kill_error = signal_group(signal.SIGKILL)
        if kill_error is not None:
            cleanup_errors.append(kill_error)
        cleanup_errors.extend(
            signal_descendants(descendants, signal.SIGKILL)
        )
        try:
            final = process.communicate(
                timeout=PROCESS_GROUP_TERMINATE_SECONDS
            )
        except subprocess.TimeoutExpired as exc:
            final = (exc.output, exc.stderr)
            # An escaped descendant can retain inherited pipe descriptors
            # after both the leader and its original process group are dead.
            # Close our pipe ends and make only one final *bounded* leader
            # wait; never turn cleanup into another infinite supervision loop.
            for label, stream in (
                ("stdout", process.stdout),
                ("stderr", process.stderr),
            ):
                if stream is None:
                    continue
                try:
                    stream.close()
                except OSError as close_error:
                    cleanup_errors.append(
                        f"close {label}: {close_error!r}"
                    )
            try:
                process.wait(timeout=PROCESS_GROUP_TERMINATE_SECONDS)
            except subprocess.TimeoutExpired as wait_error:
                cleanup_errors.append(
                    "leader did not exit after SIGKILL within "
                    f"{PROCESS_GROUP_TERMINATE_SECONDS:g}s: {wait_error!r}"
                )
        final_snapshot = proc_snapshot()
        remaining = [
            pid
            for pid, starttime in descendants.items()
            if (
                (current := final_snapshot.get(pid)) is not None
                and current[2] == starttime
                and current[1] != "Z"
            )
        ]
        if remaining:
            cleanup_errors.append(
                f"descendants remained live after SIGKILL: {remaining!r}"
            )
        return (
            final[0] if final[0] is not None else output[0],
            final[1] if final[1] is not None else output[1],
            cleanup_errors,
        )

    try:
        stdout, stderr = process.communicate(
            input=input_value,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout, stderr, cleanup_errors = stop_group()
        timeout_error = subprocess.TimeoutExpired(
            list(command),
            timeout,
            output=stdout if stdout is not None else exc.output,
            stderr=stderr if stderr is not None else exc.stderr,
        )
        if cleanup_errors:
            timeout_error.add_note(
                "Process-group cleanup diagnostics: "
                + "; ".join(cleanup_errors)
            )
        raise timeout_error from exc
    except BaseException as exc:
        _stdout, _stderr, cleanup_errors = stop_group()
        if cleanup_errors:
            exc.add_note(
                "Process-group cleanup diagnostics: "
                + "; ".join(cleanup_errors)
            )
        raise
    completed: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        list(command),
        process.returncode,
        stdout,
        stderr,
    )
    if check and completed.returncode != 0:
        _stdout, _stderr, cleanup_errors = stop_group()
        if cleanup_errors:
            completed_error = subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )
            completed_error.add_note(
                "Process-group cleanup diagnostics: "
                + "; ".join(cleanup_errors)
            )
            raise completed_error
        completed.check_returncode()
    if group_is_live():
        _stdout, _stderr, cleanup_errors = stop_group()
        detail = (
            "; cleanup diagnostics: " + "; ".join(cleanup_errors)
            if cleanup_errors
            else ""
        )
        raise RuntimeError(
            "Expected the completed command not to leave background "
            f"descendants in process group {process.pid}. Provided command: "
            f"{list(command)!r}{detail}."
        )
    return completed


def _exception_record(error: BaseException) -> dict[str, Any]:
    def command_value(value: Any) -> list[str] | str:
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return str(value)

    def stream_value(value: Any) -> str | None:
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return repr(value)

    value: dict[str, Any] = {
        "error_type": type(error).__name__,
        "message": str(error),
    }
    if isinstance(error, subprocess.CalledProcessError):
        value.update(
            {
                "command": command_value(error.cmd),
                "returncode": error.returncode,
                "stdout": stream_value(error.stdout),
                "stderr": stream_value(error.stderr),
            }
        )
    elif isinstance(error, subprocess.TimeoutExpired):
        value.update(
            {
                "command": command_value(error.cmd),
                "timeout_seconds": error.timeout,
                "stdout": stream_value(error.stdout),
                "stderr": stream_value(error.stderr),
            }
        )
    return value


def _active_stage(state: Mapping[str, Any] | None) -> str:
    if not isinstance(state, Mapping):
        return "initialization"
    stages = state.get("stages")
    if not isinstance(stages, Mapping):
        return "initialization"
    for stage in ("canary", "production"):
        record = stages.get(stage)
        if isinstance(record, Mapping) and record.get("status") != "complete":
            return stage
    return "finalization"


def _read_existing_failure_report(path: Path) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise _error(
            "failure report to be a non-symlink regular JSON file",
            path,
        )
    existing = read_json(path)
    if not isinstance(existing, dict):
        raise _error("failure report to be a JSON object", existing)
    return existing


def _atomic_create_failure_report(
    destination: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish one report without ever replacing a winner."""

    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        atomic_write_json(temporary, dict(report), canonical=True)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = _read_existing_failure_report(destination)
            if existing is None:
                raise RuntimeError(
                    "Expected an atomic failure-report winner after an "
                    f"EEXIST race. Provided value: {destination}."
                )
            return existing
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return dict(report)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_failure_report(
    path: str | Path,
    *,
    error: BaseException,
    state_path: str | Path,
    state: Mapping[str, Any] | None,
    scheduler_readback: Mapping[str, Any] | None = None,
    diagnostic_evidence: Mapping[str, Any] | None = None,
    failure_operation: Mapping[str, Any] | None = None,
    observed_attempt_paths: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the first terminal report and never overwrite it."""

    destination = Path(
        os.path.abspath(Path(path).expanduser())
    )
    if not destination.parent.is_dir():
        raise _error(
            "failure report parent to be an existing directory",
            destination.parent,
        )
    existing = _read_existing_failure_report(destination)
    if existing is not None:
        return existing

    stages = (
        state.get("stages")
        if isinstance(state, Mapping)
        and isinstance(state.get("stages"), Mapping)
        else {}
    )
    launched_jobs = []
    completed_stages = []
    for stage in ("canary", "production"):
        record = stages.get(stage) if isinstance(stages, Mapping) else None
        if not isinstance(record, Mapping):
            continue
        if record.get("status") == "complete":
            completed_stages.append(stage)
        job_id = record.get("job_id")
        if isinstance(job_id, str) and job_id.isdigit():
            live = None
            if isinstance(scheduler_readback, Mapping):
                jobs = scheduler_readback.get("jobs")
                if isinstance(jobs, Mapping):
                    live = jobs.get(stage)
            launched_jobs.append(
                {
                    "stage": stage,
                    "job_id": job_id,
                    "status": record.get("status"),
                    "cached_array_state": record.get("last_array_state"),
                    "failure_readback": live,
                }
            )

    last_valid_artifacts: dict[str, Any] = {}
    if isinstance(state, Mapping):
        binding = state.get("binding")
        if isinstance(binding, Mapping):
            for key, value in binding.items():
                if key.endswith(("_path", "_sha256")):
                    last_valid_artifacts[key] = value
        tracker_gate = state.get("last_tracker_gate")
        if isinstance(tracker_gate, Mapping):
            for key, value in tracker_gate.items():
                if key.endswith(("_path", "_sha256")):
                    last_valid_artifacts[
                        f"last_tracker_gate.{key}"
                    ] = value
        for stage_name in ("canary", "production"):
            record = stages.get(stage_name)
            if not isinstance(record, Mapping):
                continue
            for key, value in record.items():
                if key.endswith(("_path", "_sha256")):
                    last_valid_artifacts[f"{stage_name}.{key}"] = value
            for gate_name in (
                "last_tracker_gate",
                "live_submission_tracker_gate",
            ):
                gate = record.get(gate_name)
                if not isinstance(gate, Mapping):
                    continue
                for key, value in gate.items():
                    if key.endswith(("_path", "_sha256")):
                        last_valid_artifacts[
                            f"{stage_name}.{gate_name}.{key}"
                        ] = value

    stage = _active_stage(state)
    stage_record = (
        stages.get(stage)
        if isinstance(stages, Mapping)
        and isinstance(stages.get(stage), Mapping)
        else {}
    )
    failure = _exception_record(error)
    operation = (
        dict(failure_operation)
        if isinstance(failure_operation, Mapping)
        else (
            dict(stage_record["failure_operation"])
            if isinstance(stage_record, Mapping)
            and isinstance(stage_record.get("failure_operation"), Mapping)
            else {
                "name": "unclassified",
                "stage": stage,
                "command": failure.get("command"),
            }
        )
    )
    if "command" not in failure and operation.get("command") is not None:
        failure["command"] = operation["command"]
    ambiguous_submissions = (
        scheduler_readback.get("ambiguous_submissions", [])
        if isinstance(scheduler_readback, Mapping)
        else []
    )
    launched_job_count: int | None = (
        None if ambiguous_submissions else len(launched_jobs)
    )
    report = {
        "schema_version": FAILURE_REPORT_SCHEMA_VERSION,
        "status": "failed",
        "terminal": True,
        "retry_authorized": False,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "failure_operation": operation,
        "failure": failure,
        "completed_stages": completed_stages,
        "launched_job_count": launched_job_count,
        "launched_job_count_status": (
            "unknown_due_to_ambiguous_submission"
            if ambiguous_submissions
            else "exact"
        ),
        "launched_jobs": launched_jobs,
        "last_valid_artifacts": last_valid_artifacts,
        "scheduler_readback": (
            dict(scheduler_readback)
            if isinstance(scheduler_readback, Mapping)
            else {
                "attempted": False,
                "reason": "no scheduler readback was available",
            }
        ),
        "diagnostic_evidence": (
            dict(diagnostic_evidence)
            if isinstance(diagnostic_evidence, Mapping)
            else {}
        ),
        "state_path": str(Path(state_path).expanduser().resolve()),
        "state_changed": isinstance(state, Mapping),
        "files_or_external_state_changed": {
            "state_file": (
                str(Path(state_path).expanduser().resolve())
                if isinstance(state, Mapping)
                else None
            ),
            "scheduler_jobs": [
                item["job_id"] for item in launched_jobs
            ],
            "observed_attempt_paths": (
                dict(observed_attempt_paths)
                if isinstance(observed_attempt_paths, Mapping)
                else {}
            ),
        },
        "smallest_next_action": (
            "Report this failure and wait for a new user message before "
            "diagnosis, repair, cancellation, or a new immutable attempt."
        ),
    }
    # Materialize strict JSON before publishing so serialization failures
    # cannot create a partial terminal sentinel.
    canonical_json_bytes(report)
    return _atomic_create_failure_report(destination, report)


def require_login_node_context(
    environment: Mapping[str, str] | None = None,
) -> None:
    values = os.environ if environment is None else environment
    observed = {
        key: values.get(key)
        for key in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_STEP_ID",
            "SLURM_STEP_NODELIST",
        )
        if values.get(key)
    }
    if observed:
        raise RuntimeError(
            "Expected the successor supervisor to run on a Jean Zay login "
            f"node outside every allocation. Provided value: {observed!r}."
        )


def current_canary_smoke_receipt(
    *,
    output_root: str | Path,
    bundle_id: str,
    array_job_id: str,
) -> Path:
    if not array_job_id.isdigit():
        raise _error(
            "canary array job id to contain only decimal digits",
            array_job_id,
        )
    jobs_root = (
        Path(output_root).expanduser().resolve()
        / "smoke"
        / bundle_id
        / "jobs"
    )
    if not jobs_root.is_dir():
        raise _error(
            "current canary job output directory to exist", jobs_root
        )
    matches = []
    pattern = re.compile(
        rf"{re.escape(array_job_id)}_([0-9]+)_0\Z"
    )
    for candidate in jobs_root.iterdir():
        if (
            candidate.is_symlink()
            or not candidate.is_dir()
            or pattern.fullmatch(candidate.name) is None
        ):
            continue
        receipt = candidate / "receipt.json"
        if receipt.is_file() and not receipt.is_symlink():
            matches.append(receipt.resolve())
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one job-scoped smoke receipt for the "
            f"just-submitted canary. Provided value: root={jobs_root}, "
            f"array_job_id={array_job_id!r}, matches="
            f"{[str(item) for item in matches]!r}."
        )
    return matches[0]


def _normalized_state(value: str, *, job_id: str) -> str:
    state = value.strip().split()[0].rstrip("+") if value.strip() else ""
    if (
        state != "COMPLETED"
        and state not in SLURM_ACTIVE_STATES
        and state not in SLURM_FAILED_STATES
    ):
        raise RuntimeError(
            "Expected a recognized Slurm allocation state. "
            f"Provided value: job_id={job_id!r}, state={state!r}."
        )
    return state


def _validate_accounting_contract(row: Mapping[str, str]) -> None:
    expected = {
        "account": ACCOUNT,
        "partition": PARTITION,
        "qos": QOS,
        "constraint": CONSTRAINT,
    }
    mismatch = {
        key: {"expected": value, "provided": row.get(key)}
        for key, value in expected.items()
        if row.get(key) != value
    }
    if mismatch:
        raise RuntimeError(
            "Expected every successor array accounting row to match the exact "
            f"UMG V100 scheduler contract. Provided value: "
            f"job_id={row.get('job_id')!r}, mismatches={mismatch!r}."
        )


def _query_scontrol_array_state(
    job_id: str,
    expected_task_count: int,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    command = ["scontrol", "show", "job", job_id, "-o"]
    completed = runner(command, check=True, text=True, capture_output=True)
    task_states: dict[int, str] = {}
    task_exit_codes: dict[int, str] = {}
    compressed_rows: list[str] = []
    parent_states: list[str] = []
    for raw_line in completed.stdout.splitlines():
        fields: dict[str, str] = {}
        for token in raw_line.split():
            if "=" in token:
                key, value = token.split("=", maxsplit=1)
                fields[key] = value
        if not fields:
            continue
        row = {
            "job_id": fields.get("JobId", job_id),
            "state": _normalized_state(
                fields.get("JobState", ""),
                job_id=fields.get("JobId", job_id),
            ),
            "exit_code": fields.get("ExitCode", "0:0"),
            "account": fields.get("Account", ""),
            "partition": fields.get("Partition", ""),
            "qos": fields.get("QOS", ""),
            "constraint": fields.get("Features", ""),
        }
        if CONSTRAINT in row["constraint"]:
            row["constraint"] = CONSTRAINT
        _validate_accounting_contract(row)
        if row["state"] in SLURM_FAILED_STATES or (
            row["state"] == "COMPLETED" and row["exit_code"] != "0:0"
        ):
            raise RuntimeError(
                "Expected scontrol fallback rows to avoid terminal failure. "
                f"Provided value: {row!r}."
            )
        task_spec = fields.get("ArrayTaskId")
        if task_spec is None:
            parent_states.append(row["state"])
            continue
        throttle_free = task_spec.split("%", maxsplit=1)[0]
        if throttle_free.isdigit():
            index = int(throttle_free)
            if index in task_states:
                raise RuntimeError(
                    "Expected unique task indices in scontrol fallback. "
                    f"Provided duplicate: {index!r}."
                )
            task_states[index] = row["state"]
            task_exit_codes[index] = row["exit_code"]
        else:
            compressed_rows.append(task_spec)
    unexpected = sorted(
        set(task_states) - set(range(expected_task_count))
    )
    if unexpected:
        raise RuntimeError(
            "Expected only exact packed successor array tasks in "
            f"scontrol fallback. Provided value: {unexpected!r}."
        )
    if not task_states and not compressed_rows and not parent_states:
        raise RuntimeError(
            "Expected sacct or scontrol to expose the submitted array. "
            f"Provided value: {completed.stdout!r}."
        )
    missing = sorted(set(range(expected_task_count)) - set(task_states))
    complete = (
        not missing
        and all(state == "COMPLETED" for state in task_states.values())
        and all(code == "0:0" for code in task_exit_codes.values())
    )
    return {
        "status": "complete" if complete else "waiting",
        "parent_state": (
            "COMPLETED"
            if complete
            else (
                "RUNNING"
                if "RUNNING" in set(task_states.values()) | set(parent_states)
                else "SCONTROL_PENDING"
            )
        ),
        "expected_task_count": expected_task_count,
        "observed_task_count": len(task_states),
        "completed_task_count": sum(
            state == "COMPLETED" for state in task_states.values()
        ),
        "missing_task_indices": missing,
        "task_states": {
            str(index): state
            for index, state in sorted(task_states.items())
        },
        "task_exit_codes": {
            str(index): code
            for index, code in sorted(task_exit_codes.items())
        },
        "compressed_rows": compressed_rows,
        "completion_basis": (
            "all_indexed_tasks_from_scontrol" if complete else None
        ),
        "accounting_source": "scontrol_fallback",
        "command": command,
    }


def query_successor_array_state(
    job_id: str,
    expected_task_count: int,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Strictly account for one exact Slurm array and no other task indices."""

    if not isinstance(job_id, str) or not job_id.isdigit():
        raise _error("a numeric Slurm array job id", job_id)
    if type(expected_task_count) is not int or expected_task_count < 1:
        raise _error(
            "expected_task_count to be an integer >= 1",
            expected_task_count,
        )
    command = [
        "sacct",
        "-n",
        "-X",
        "-j",
        job_id,
        "--format=JobIDRaw,JobID,State,ExitCode,Account,Partition,QOS,Constraints",
        "--parsable2",
    ]
    try:
        completed = runner(
            command, check=True, text=True, capture_output=True
        )
    except subprocess.CalledProcessError:
        return _query_scontrol_array_state(
            job_id,
            expected_task_count,
            runner=runner,
        )
    parent: dict[str, str] | None = None
    tasks: dict[int, dict[str, str]] = {}
    compressed: list[dict[str, str]] = []
    pattern = re.compile(rf"{re.escape(job_id)}_([0-9]+)\Z")
    for raw_line in completed.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split("|")]
        if len(fields) < 8:
            continue
        row = {
            "job_id_raw": fields[0],
            "job_id": fields[1],
            "state": _normalized_state(fields[2], job_id=fields[1]),
            "exit_code": fields[3],
            "account": fields[4],
            "partition": fields[5],
            "qos": fields[6],
            "constraint": fields[7],
        }
        display = row["job_id"]
        if display == job_id:
            if parent is not None:
                raise RuntimeError(
                    "Expected at most one successor array parent accounting "
                    f"row. Provided value: {[parent, row]!r}."
                )
            _validate_accounting_contract(row)
            parent = row
            continue
        match = pattern.fullmatch(display)
        if match is not None:
            index = int(match.group(1))
            if index in tasks:
                raise RuntimeError(
                    "Expected one accounting row per successor array task. "
                    f"Provided duplicate index: {index!r}."
                )
            _validate_accounting_contract(row)
            tasks[index] = row
            continue
        if display.startswith(f"{job_id}_["):
            _validate_accounting_contract(row)
            compressed.append(row)
    expected_indices = set(range(expected_task_count))
    if parent is None and not tasks and not compressed:
        return _query_scontrol_array_state(
            job_id,
            expected_task_count,
            runner=runner,
        )
    observed_indices = set(tasks)
    unexpected = sorted(observed_indices - expected_indices)
    missing = sorted(expected_indices - observed_indices)
    if unexpected:
        raise RuntimeError(
            "Expected only exact packed successor array tasks. "
            f"Provided unexpected indices: {unexpected!r}."
        )
    failed = {
        index: row["state"]
        for index, row in sorted(tasks.items())
        if row["state"] in SLURM_FAILED_STATES
    }
    if parent is not None and parent["state"] in SLURM_FAILED_STATES:
        raise RuntimeError(
            "Expected the successor array parent to avoid terminal failure. "
            f"Provided value: job_id={job_id!r}, state={parent['state']!r}."
        )
    if failed:
        raise RuntimeError(
            "Expected every successor array task to avoid terminal failure. "
            f"Provided value: {failed!r}."
        )
    compressed_failed = {
        row["job_id"]: {
            "state": row["state"],
            "exit_code": row["exit_code"],
        }
        for row in compressed
        if row["state"] in SLURM_FAILED_STATES
        or (
            row["state"] == "COMPLETED"
            and row["exit_code"] != "0:0"
        )
    }
    if compressed_failed:
        raise RuntimeError(
            "Expected every compressed successor array row to avoid terminal "
            f"failure. Provided value: {compressed_failed!r}."
        )
    bad_completed = {
        index: row["exit_code"]
        for index, row in sorted(tasks.items())
        if row["state"] == "COMPLETED" and row["exit_code"] != "0:0"
    }
    if bad_completed:
        raise RuntimeError(
            "Expected every completed successor task to exit 0:0. "
            f"Provided value: {bad_completed!r}."
        )
    if (
        parent is not None
        and parent["state"] == "COMPLETED"
        and parent["exit_code"] != "0:0"
    ):
        raise RuntimeError(
            "Expected a completed successor array parent to exit 0:0. "
            f"Provided value: {parent['exit_code']!r}."
        )
    non_complete = {
        index: row["state"]
        for index, row in sorted(tasks.items())
        if row["state"] != "COMPLETED"
    }
    completion_basis: str | None = None
    if parent is not None and parent["state"] == "COMPLETED":
        if missing:
            raise RuntimeError(
                "Expected a completed successor parent to retain every exact "
                f"task row. Provided missing indices: {missing!r}."
            )
        if non_complete:
            raise RuntimeError(
                "Expected all tasks under a completed successor parent to be "
                f"COMPLETED. Provided value: {non_complete!r}."
            )
        status = "complete"
        completion_basis = "explicit_parent_and_all_indexed_tasks"
    elif not missing and not non_complete:
        status = "complete"
        completion_basis = "all_indexed_tasks_without_parent_row"
    else:
        status = "waiting"
    if compressed:
        compressed_states = {row["state"] for row in compressed}
        if len(compressed_states) != 1:
            raise RuntimeError(
                "Expected consistent states across compressed successor array "
                f"rows. Provided value: {sorted(compressed_states)!r}."
            )
    parent_state = (
        parent["state"]
        if parent is not None
        else (
            "COMPLETED"
            if status == "complete"
            else (
                compressed[0]["state"]
                if compressed
                else (
                    "RUNNING"
                    if any(
                        row["state"] == "RUNNING" for row in tasks.values()
                    )
                    else "ACCOUNTING_PENDING"
                )
            )
        )
    )
    return {
        "status": status,
        "parent_state": parent_state,
        "expected_task_count": expected_task_count,
        "observed_task_count": len(tasks),
        "completed_task_count": sum(
            row["state"] == "COMPLETED" for row in tasks.values()
        ),
        "missing_task_indices": missing,
        "task_states": {
            str(index): row["state"]
            for index, row in sorted(tasks.items())
        },
        "task_exit_codes": {
            str(index): row["exit_code"]
            for index, row in sorted(tasks.items())
        },
        "compressed_rows": [row["job_id"] for row in compressed],
        "completion_basis": completion_basis,
        "accounting_source": "sacct",
        "command": command,
    }


def _artifact_under_root(raw_path: Any, root: Path, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise _error(f"{label} to be a non-empty path string", raw_path)
    supplied = Path(raw_path)
    path = (
        supplied.resolve()
        if supplied.is_absolute()
        else (root / supplied).resolve()
    )
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise _error(f"{label} to be under {root}", path) from exc
    if not path.is_file():
        raise _error(f"{label} to be an existing file", path)
    return path


def _validate_entry_completion(
    path: Path,
    *,
    output_root: Path,
    status_entry: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise _error("entry completion marker to be a JSON object", value)
    expected_fields = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "official_test_read": False,
        "bundle_id": expected_identity["bundle_id"],
        "bundle_manifest_sha256": expected_identity[
            "bundle_manifest_sha256"
        ],
        "config_sha256": expected_identity["config_sha256"],
        "config_file_sha256": expected_identity["config_file_sha256"],
        "entry_index": status_entry["entry_index"],
        "entry_id": status_entry["entry_id"],
    }
    mismatch = {
        key: {"expected": expected, "provided": value.get(key)}
        for key, expected in expected_fields.items()
        if value.get(key) != expected
    }
    if mismatch:
        raise RuntimeError(
            "Expected each production completion marker to bind the exact "
            f"immutable entry and prohibit test reads. Provided value: "
            f"path={path}, mismatches={mismatch!r}."
        )
    for field in ("entry_payload_sha256",):
        if not isinstance(value.get(field), str) or not re.fullmatch(
            r"[0-9a-f]{64}", value[field]
        ):
            raise _error(
                f"completion {field} to be a lowercase SHA-256",
                value.get(field),
            )
    if not isinstance(value.get("execution_source"), dict):
        raise _error(
            "completion execution_source to be a JSON object",
            value.get("execution_source"),
        )

    def live_receipt_payload(
        *,
        path_value: Any,
        sha_value: Any,
        label: str,
    ) -> tuple[Path, dict[str, Any]]:
        if not isinstance(path_value, str) or not path_value:
            raise _error(f"{label} path to be non-empty", path_value)
        receipt_path = Path(path_value).expanduser().resolve()
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise _error(
                f"{label} to be an existing non-symlink file",
                receipt_path,
            )
        if (
            not isinstance(sha_value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", sha_value)
            or sha256_file(receipt_path) != sha_value
        ):
            raise RuntimeError(
                f"Expected {label} file SHA-256 to match the supervisor "
                f"binding. Provided value: path={receipt_path}, "
                f"expected_sha256={sha_value!r}."
            )
        payload = read_json(receipt_path)
        if not isinstance(payload, dict):
            raise _error(f"{label} payload to be a JSON object", payload)
        return receipt_path, payload

    preflight_path, preflight_payload = live_receipt_payload(
        path_value=expected_identity["preflight_receipt_path"],
        sha_value=expected_identity["preflight_receipt_sha256"],
        label="runtime preflight receipt",
    )
    authorization_path, authorization_payload = live_receipt_payload(
        path_value=expected_identity["launch_authorization_receipt_path"],
        sha_value=expected_identity[
            "launch_authorization_receipt_sha256"
        ],
        label="launch-authorization receipt",
    )
    expected_preflight = {
        "source_path": str(preflight_path),
        "source_file_sha256": expected_identity[
            "preflight_receipt_sha256"
        ],
        "payload_sha256": _canonical_json_sha256(preflight_payload),
        "payload": preflight_payload,
        "launch_authorization": {
            "source_path": str(authorization_path),
            "source_file_sha256": expected_identity[
                "launch_authorization_receipt_sha256"
            ],
            "payload_sha256": _canonical_json_sha256(
                authorization_payload
            ),
            "payload": authorization_payload,
        },
    }
    if value.get("preflight_receipt") != expected_preflight:
        raise RuntimeError(
            "Expected every production completion marker to bind the exact "
            f"embedded runtime preflight receipt. Provided value: "
            f"expected={expected_preflight!r}, "
            f"observed={value.get('preflight_receipt')!r}."
        )
    canary_path, canary_payload = live_receipt_payload(
        path_value=expected_identity["canary_gate_receipt_path"],
        sha_value=expected_identity["canary_gate_receipt_sha256"],
        label="canary-gate receipt",
    )
    expected_canary = {
        "source_path": str(canary_path),
        "source_file_sha256": expected_identity[
            "canary_gate_receipt_sha256"
        ],
        "payload_sha256": _canonical_json_sha256(canary_payload),
        "payload": canary_payload,
    }
    if value.get("canary_gate_receipt") != expected_canary:
        raise RuntimeError(
            "Expected every production completion marker to bind the exact "
            "verified canary-gate receipt. Provided value: "
            f"expected={expected_canary!r}, "
            f"observed={value.get('canary_gate_receipt')!r}."
        )
    outputs = value.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise _error("completion outputs to be a non-empty list", outputs)
    paths = [item.get("path") if isinstance(item, dict) else None for item in outputs]
    if paths != sorted(paths):
        raise RuntimeError(
            "Expected completion outputs to be sorted by relative path. "
            f"Provided value: {paths!r}."
        )
    validated: list[dict[str, Any]] = []
    for index, artifact in enumerate(outputs):
        if not isinstance(artifact, dict) or set(artifact) != {
            "path",
            "sha256",
            "bytes",
        }:
            raise _error(
                "each completion output to contain path, sha256, and bytes",
                artifact,
            )
        artifact_path = _artifact_under_root(
            artifact["path"],
            output_root,
            f"completion outputs[{index}].path",
        )
        size = artifact["bytes"]
        digest = artifact["sha256"]
        if (
            type(size) is not int
            or size < 0
            or artifact_path.stat().st_size != size
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or sha256_file(artifact_path) != digest
        ):
            raise RuntimeError(
                "Expected every completion output size/hash to match. "
                f"Provided value: artifact={artifact!r}, path={artifact_path}."
            )
        validated.append(
            {
                "path": str(artifact_path),
                "sha256": digest,
                "bytes": size,
            }
        )
    return {
        "completion_path": str(path),
        "completion_sha256": sha256_file(path),
        "outputs": validated,
    }


def validate_production_status(
    value: Any,
    *,
    output_root: str | Path,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    if not isinstance(value, dict):
        raise _error("production status to be a JSON object", value)
    expected = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": "complete",
        "official_test_read": False,
        "bundle_id": expected_identity["bundle_id"],
        "bundle_manifest_sha256": expected_identity[
            "bundle_manifest_sha256"
        ],
        "config_sha256": expected_identity["config_sha256"],
        "config_file_sha256": expected_identity["config_file_sha256"],
        "expected_entry_count": LOGICAL_ENTRY_COUNT,
        "completed_entry_count": LOGICAL_ENTRY_COUNT,
        "pending_entry_count": 0,
        "errors": [],
    }
    mismatches = {
        key: {"expected": expected_value, "provided": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            "Expected the full 12-entry successor status to be complete and "
            f"valid. Provided mismatches: {mismatches!r}."
        )
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != LOGICAL_ENTRY_COUNT:
        raise _error(
            "production status entries to contain exactly 12 rows", entries
        )
    completions: list[dict[str, Any]] = []
    for expected_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise _error(f"status entry {expected_index} to be an object", entry)
        expected_steps = 34380 if expected_index < 6 else 68760
        checks = {
            "entry_index": expected_index,
            "status": "complete",
            "mode": "successor_confirmation",
            "optimizer_steps_completed": expected_steps,
        }
        bad = {
            key: {"expected": expected_value, "provided": entry.get(key)}
            for key, expected_value in checks.items()
            if entry.get(key) != expected_value
        }
        if bad:
            raise RuntimeError(
                "Expected each ordered successor status entry to complete its "
                f"exact budget. Provided value: index={expected_index}, "
                f"mismatches={bad!r}."
            )
        if not isinstance(entry.get("entry_id"), str) or not entry["entry_id"]:
            raise _error("status entry_id to be non-empty", entry.get("entry_id"))
        completion_path = _artifact_under_root(
            entry.get("completion_path"),
            root,
            f"entries[{expected_index}].completion_path",
        )
        _artifact_under_root(
            entry.get("result_path"),
            root,
            f"entries[{expected_index}].result_path",
        )
        checkpoints = entry.get("checkpoint_paths")
        if not isinstance(checkpoints, list) or len(checkpoints) != 2:
            raise _error(
                "each status entry checkpoint_paths to contain best and final",
                checkpoints,
            )
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            _artifact_under_root(
                checkpoint,
                root,
                f"entries[{expected_index}].checkpoint_paths[{checkpoint_index}]",
            )
        completions.append(
            {
                "entry_index": expected_index,
                "entry_id": entry["entry_id"],
                **_validate_entry_completion(
                    completion_path,
                    output_root=root,
                    status_entry=entry,
                    expected_identity=expected_identity,
                ),
            }
        )
    return {
        "status": value,
        "completion_markers": completions,
    }


def validate_partial_production_status(
    value: Any,
    *,
    output_root: str | Path,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every artifact already published while an array is active."""

    root = Path(output_root).expanduser().resolve()
    if not isinstance(value, dict):
        raise _error("partial production status to be a JSON object", value)
    fixed = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "official_test_read": False,
        "bundle_id": expected_identity["bundle_id"],
        "bundle_manifest_sha256": expected_identity[
            "bundle_manifest_sha256"
        ],
        "config_sha256": expected_identity["config_sha256"],
        "config_file_sha256": expected_identity["config_file_sha256"],
        "expected_entry_count": LOGICAL_ENTRY_COUNT,
        "errors": [],
    }
    mismatches = {
        key: {"expected": expected, "provided": value.get(key)}
        for key, expected in fixed.items()
        if value.get(key) != expected
    }
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != LOGICAL_ENTRY_COUNT:
        mismatches["entries"] = {
            "expected": "exactly 12 ordered rows",
            "provided": entries,
        }
    if mismatches:
        raise RuntimeError(
            "Expected active production status to retain exact immutable "
            f"identity with no invalid outputs. Provided mismatches: "
            f"{mismatches!r}."
        )
    completions: list[dict[str, Any]] = []
    pending_count = 0
    for expected_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise _error(
                f"partial status entry {expected_index} to be an object",
                entry,
            )
        state = entry.get("status")
        common = {
            "entry_index": expected_index,
            "mode": "successor_confirmation",
        }
        bad = {
            key: {"expected": expected, "provided": entry.get(key)}
            for key, expected in common.items()
            if entry.get(key) != expected
        }
        if not isinstance(entry.get("entry_id"), str) or not entry["entry_id"]:
            bad["entry_id"] = {
                "expected": "a non-empty string",
                "provided": entry.get("entry_id"),
            }
        expected_steps = 34380 if expected_index < 6 else 68760
        if state == "pending":
            pending_count += 1
            if entry.get("optimizer_steps_completed") != 0:
                bad["optimizer_steps_completed"] = {
                    "expected": 0,
                    "provided": entry.get("optimizer_steps_completed"),
                }
        elif state == "complete":
            if entry.get("optimizer_steps_completed") != expected_steps:
                bad["optimizer_steps_completed"] = {
                    "expected": expected_steps,
                    "provided": entry.get("optimizer_steps_completed"),
                }
        else:
            bad["status"] = {
                "expected": "pending or complete",
                "provided": state,
            }
        if bad:
            raise RuntimeError(
                "Expected each active successor status entry to be exact and "
                f"valid. Provided value: index={expected_index}, "
                f"mismatches={bad!r}."
            )
        if state != "complete":
            continue
        completion_path = _artifact_under_root(
            entry.get("completion_path"),
            root,
            f"entries[{expected_index}].completion_path",
        )
        _artifact_under_root(
            entry.get("result_path"),
            root,
            f"entries[{expected_index}].result_path",
        )
        checkpoints = entry.get("checkpoint_paths")
        if not isinstance(checkpoints, list) or len(checkpoints) != 2:
            raise _error(
                "each completed partial-status checkpoint_paths to contain "
                "best and final",
                checkpoints,
            )
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            _artifact_under_root(
                checkpoint,
                root,
                f"entries[{expected_index}].checkpoint_paths"
                f"[{checkpoint_index}]",
            )
        completions.append(
            {
                "entry_index": expected_index,
                "entry_id": entry["entry_id"],
                **_validate_entry_completion(
                    completion_path,
                    output_root=root,
                    status_entry=entry,
                    expected_identity=expected_identity,
                ),
            }
        )
    completed_count = len(completions)
    expected_status = (
        "complete"
        if completed_count == LOGICAL_ENTRY_COUNT
        else "partial"
        if completed_count
        else "pending"
    )
    counts = {
        "status": expected_status,
        "completed_entry_count": completed_count,
        "pending_entry_count": pending_count,
    }
    count_mismatches = {
        key: {"expected": expected, "provided": value.get(key)}
        for key, expected in counts.items()
        if value.get(key) != expected
    }
    if pending_count + completed_count != LOGICAL_ENTRY_COUNT:
        count_mismatches["entry_state_count"] = {
            "expected": LOGICAL_ENTRY_COUNT,
            "provided": pending_count + completed_count,
        }
    if count_mismatches:
        raise RuntimeError(
            "Expected active production status counts to match the exact "
            f"validated rows. Provided mismatches: {count_mismatches!r}."
        )
    return {
        "status": value,
        "completion_markers": completions,
    }


class SuccessorJeanZaySupervisor:
    """Two-stage canary/production state machine with immutable resumption."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        allocation_verifier: Callable[..., dict[str, Any]] = (
            verify_login_umg_v100_allocation
        ),
        tracker_validator: Callable[[], dict[str, Any]] | None = None,
    ):
        self.args = args
        self.raw_runner = (
            _run_process_group if runner is subprocess.run else runner
        )
        self.command_timeout_seconds = _bounded_command_timeout(
            getattr(
                args,
                "command_timeout_seconds",
                DEFAULT_COMMAND_TIMEOUT_SECONDS,
            )
        )
        self.runner = self._run_command
        self.sleeper = sleeper
        self.clock = clock
        self.allocation_verifier = allocation_verifier
        self.tracker_validator = tracker_validator
        self.state_path = Path(args.state).expanduser().resolve()
        self.failure_report_path = self.state_path.with_name(
            f"{self.state_path.stem}-failure.json"
        )
        self.state: dict[str, Any] | None = None
        self.canary_command: list[str] | None = None
        self.production_command: list[str] | None = None
        self.gate_binding: dict[str, Any] | None = None
        self.runtime_preflight: dict[str, Any] | None = None
        self.official_verifier: str | None = None
        self.failure_operation: dict[str, Any] | None = None
        self.follow_deadline: float | None = None

    def _run_command(
        self,
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        timeout = float(
            kwargs.get("timeout", self.command_timeout_seconds)
        )
        if self.follow_deadline is not None:
            remaining = self.follow_deadline - self.clock()
            if remaining <= 0:
                raise TimeoutError(
                    "Expected the supervisor command to start before the "
                    "current follow deadline. Provided remaining seconds: "
                    f"{remaining:.6f}."
                )
            timeout = min(timeout, remaining)
        kwargs["timeout"] = timeout
        return self.raw_runner(list(command), **kwargs)

    def _reject_terminal_attempt(self) -> None:
        existing_failure = _read_existing_failure_report(
            self.failure_report_path
        )
        if existing_failure is not None:
            raise RuntimeError(
                "Expected a new immutable attempt path after a terminal "
                f"failure report. Provided value: {self.failure_report_path}."
            )
        if not self.state_path.is_file():
            return
        existing = read_json(self.state_path)
        if (
            isinstance(existing, dict)
            and existing.get("status") == "failed"
        ):
            raise RuntimeError(
                "Expected a terminal failed supervisor state never to resume. "
                f"Provided value: {self.state_path}."
            )

    def _require_fresh_canary_receipt_paths(self) -> None:
        for raw_path, label in (
            (
                self.args.official_canary_receipt,
                "official canary receipt",
            ),
            (
                self.args.canary_receipt,
                "successor semantic canary receipt",
            ),
        ):
            candidate = Path(
                os.path.abspath(Path(raw_path).expanduser())
            )
            if candidate.exists() or candidate.is_symlink():
                raise _error(
                    f"new-attempt {label} path not to exist",
                    candidate,
                )

    def record_failure(self, error: BaseException) -> dict[str, Any]:
        if self.state is None and self.state_path.is_file():
            recovered = read_json(self.state_path)
            if isinstance(recovered, dict):
                self.state = recovered
        scheduler_readback = self._capture_failure_scheduler_readback()
        diagnostic_evidence = self._collect_failure_log_evidence()
        observed_attempt_paths: dict[str, Any] = {}
        for argument, label in (
            ("tracker_gate_receipt", "tracker_gate_receipt"),
            (
                "launch_authorization_receipt",
                "launch_authorization_receipt",
            ),
            ("scheduled_preflight_receipt", "scheduled_preflight_receipt"),
            ("preflight_receipt", "runtime_preflight_receipt"),
            ("official_canary_receipt", "official_canary_receipt"),
            ("canary_receipt", "successor_canary_receipt"),
            ("output_root", "output_root"),
        ):
            raw_path = getattr(self.args, argument, None)
            if not isinstance(raw_path, (str, os.PathLike)) or not str(
                raw_path
            ):
                continue
            candidate = Path(
                os.path.abspath(Path(raw_path).expanduser())
            )
            exists = candidate.exists() or candidate.is_symlink()
            item: dict[str, Any] = {
                "path": str(candidate),
                "observed": exists,
                "symlink": candidate.is_symlink(),
                "change_status": (
                    "may_have_been_created_or_modified_by_attempt"
                    if exists
                    else "not_observed"
                ),
            }
            if exists and candidate.is_file() and not candidate.is_symlink():
                try:
                    item["sha256"] = sha256_file(candidate)
                except OSError as hash_error:
                    item["sha256_error"] = repr(hash_error)
            observed_attempt_paths[label] = item
        output_root = getattr(self.args, "output_root", None)
        if isinstance(output_root, (str, os.PathLike)) and str(output_root):
            slurm_dir = (
                Path(os.path.abspath(Path(output_root).expanduser()))
                / "slurm"
            )
            observed_attempt_paths["slurm_log_directory"] = {
                "path": str(slurm_dir),
                "observed": slurm_dir.exists() or slurm_dir.is_symlink(),
                "symlink": slurm_dir.is_symlink(),
                "change_status": (
                    "may_have_been_created_by_attempt"
                    if slurm_dir.exists() or slurm_dir.is_symlink()
                    else "not_observed"
                ),
            }
        report = write_failure_report(
            self.failure_report_path,
            error=error,
            state_path=self.state_path,
            state=self.state,
            scheduler_readback=scheduler_readback,
            diagnostic_evidence=diagnostic_evidence,
            failure_operation=self.failure_operation,
            observed_attempt_paths=observed_attempt_paths,
        )
        if isinstance(self.state, dict):
            self.state["status"] = "failed"
            self.state["failure_report_path"] = str(
                self.failure_report_path
            )
            self.state["failure_report_sha256"] = sha256_file(
                self.failure_report_path
            )
            atomic_write_json(
                self.state_path,
                self.state,
                canonical=True,
            )
        return report

    def _note_failure_operation(
        self,
        *,
        name: str,
        stage: str,
        command: Sequence[str] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        value: dict[str, Any] = {
            "name": name,
            "stage": stage,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        if command is not None:
            value["command"] = [str(item) for item in command]
        if details is not None:
            value["details"] = dict(details)
        self.failure_operation = value
        if isinstance(self.state, dict):
            stages = self.state.get("stages")
            if isinstance(stages, dict):
                record = stages.get(stage)
                if isinstance(record, dict):
                    record["failure_operation"] = value

    def _capture_failure_scheduler_readback(self) -> dict[str, Any]:
        captured_at = datetime.now(timezone.utc).isoformat()
        jobs: dict[str, Any] = {}
        ambiguous: list[dict[str, Any]] = []
        if not isinstance(self.state, Mapping):
            return {
                "attempted": False,
                "captured_at": captured_at,
                "jobs": jobs,
                "ambiguous_submissions": ambiguous,
                "reason": "supervisor state was unavailable",
            }
        stages = self.state.get("stages")
        if not isinstance(stages, Mapping):
            return {
                "attempted": False,
                "captured_at": captured_at,
                "jobs": jobs,
                "ambiguous_submissions": ambiguous,
                "reason": "supervisor stages were unavailable",
            }
        for stage in ("canary", "production"):
            record = stages.get(stage)
            if not isinstance(record, Mapping):
                continue
            job_id = record.get("job_id")
            if not isinstance(job_id, str) or not job_id.isdigit():
                if record.get("submission_status") in {
                    "started",
                    "ambiguous",
                }:
                    ambiguous.append(
                        {
                            "stage": stage,
                            "submission_status": record.get(
                                "submission_status"
                            ),
                            "command": record.get("command"),
                            "reason": "sbatch may have created an unknown job",
                        }
                    )
                continue
            command = [
                "sacct",
                "-n",
                "-X",
                "-j",
                job_id,
                "--format=JobIDRaw,JobID,State,ExitCode",
                "--parsable2",
            ]
            try:
                completed = self.raw_runner(
                    command,
                    check=True,
                    text=True,
                    capture_output=True,
                    timeout=min(self.command_timeout_seconds, 30.0),
                )
                rows = []
                for raw_line in completed.stdout.splitlines():
                    fields = [
                        field.strip() for field in raw_line.split("|")
                    ]
                    if len(fields) < 4:
                        continue
                    rows.append(
                        {
                            "job_id_raw": fields[0],
                            "job_id": fields[1],
                            "state": fields[2],
                            "exit_code": fields[3],
                        }
                    )
                jobs[stage] = {
                    "attempted": True,
                    "command": command,
                    "rows": rows,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            except BaseException as readback_error:
                jobs[stage] = {
                    "attempted": True,
                    "command": command,
                    "error": _exception_record(readback_error),
                }
        return {
            "attempted": bool(jobs),
            "captured_at": captured_at,
            "jobs": jobs,
            "ambiguous_submissions": ambiguous,
            "reason": (
                None
                if jobs
                else "no numeric scheduler job ID was recorded"
            ),
        }

    @staticmethod
    def _bounded_log_tail(path: Path, *, limit: int = 16 * 1024) -> dict[str, Any]:
        value: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists() or path.is_symlink(),
        }
        if not value["exists"]:
            return value
        if path.is_symlink() or not path.is_file():
            value["error"] = (
                "expected a non-symlink regular Slurm diagnostic log"
            )
            return value
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(size - limit, 0))
            tail = handle.read(limit)
        value.update(
            {
                "size_bytes": size,
                "tail_truncated": size > limit,
                "tail": tail.decode("utf-8", errors="replace"),
            }
        )
        return value

    def _collect_failure_log_evidence(self) -> dict[str, Any]:
        evidence: dict[str, Any] = {"slurm_logs": []}
        if not isinstance(self.state, Mapping):
            evidence["reason"] = "supervisor state was unavailable"
            return evidence
        stages = self.state.get("stages")
        if not isinstance(stages, Mapping):
            evidence["reason"] = "supervisor stages were unavailable"
            return evidence
        log_root = (
            Path(self.args.output_root).expanduser().resolve() / "slurm"
        )
        for stage, task_count in (
            ("canary", 1),
            ("production", PRODUCTION_TASK_COUNT),
        ):
            record = stages.get(stage)
            if not isinstance(record, Mapping):
                continue
            job_id = record.get("job_id")
            if not isinstance(job_id, str) or not job_id.isdigit():
                continue
            for task_index in range(task_count):
                for suffix in ("out", "err"):
                    path = (
                        log_root
                        / (
                            f"pd-successor-{stage}-{job_id}_"
                            f"{task_index}.{suffix}"
                        )
                    )
                    try:
                        evidence["slurm_logs"].append(
                            self._bounded_log_tail(path)
                        )
                    except BaseException as log_error:
                        evidence["slurm_logs"].append(
                            {
                                "path": str(path),
                                "error": _exception_record(log_error),
                            }
                        )
        return evidence

    def _validate_tracker(
        self,
        *,
        required_status: str | None,
    ) -> dict[str, Any]:
        if required_status not in {None, "preflighting", "launch-ready"}:
            raise _error(
                "required tracker status to be preflighting, launch-ready, "
                "or None",
                required_status,
            )
        if self.tracker_validator is not None:
            value = self.tracker_validator()
            if (
                not isinstance(value, dict)
                or value.get("experiment_id") != EXPERIMENT_ID
                or (
                    required_status is not None
                    and value.get("status") != required_status
                )
            ):
                raise _error(
                    "injected tracker validation to bind the exact experiment"
                    + (
                        f" in {required_status} state"
                        if required_status is not None
                        else ""
                    ),
                    value,
                )
            return dict(value)

        raw_path = getattr(self.args, "tracker_gate_receipt", None)
        raw_sha256 = getattr(
            self.args,
            "tracker_gate_receipt_sha256",
            None,
        )
        if not isinstance(raw_path, str) or not raw_path:
            raise _error(
                "--tracker-gate-receipt to be a non-empty path string",
                raw_path,
            )
        if (
            not isinstance(raw_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", raw_sha256) is None
        ):
            raise _error(
                "--tracker-gate-receipt-sha256 to be a lowercase SHA-256",
                raw_sha256,
            )
        receipt_path = Path(
            os.path.abspath(Path(raw_path).expanduser())
        )
        receipt_payload = _read_regular_file_once(
            receipt_path,
            label="tracker gate receipt",
        )
        observed_receipt_sha256 = hashlib.sha256(
            receipt_payload
        ).hexdigest()
        if observed_receipt_sha256 != raw_sha256:
            raise RuntimeError(
                "Expected the tracker gate receipt SHA-256 to match its "
                f"explicit argument. Provided value: expected={raw_sha256!r}, "
                f"observed={observed_receipt_sha256!r}."
            )
        receipt = _strict_json_bytes(
            receipt_payload,
            label="tracker gate receipt",
        )
        expected_keys = {
            "schema_version",
            "generated_at_utc",
            "experiment_id",
            "gate_stage",
            "status",
            "state_path",
            "output_root",
            "tracker_path",
            "tracker_sha256",
            "validator_path",
            "validator_sha256",
        }
        if not isinstance(receipt, dict) or set(receipt) != expected_keys:
            raise _error(
                f"tracker gate receipt keys to be exactly "
                f"{sorted(expected_keys)!r}",
                receipt,
            )
        expected_values = {
            "schema_version": TRACKER_GATE_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "validator_sha256": TRACKER_VALIDATOR_SHA256,
            "state_path": str(self.state_path),
            "output_root": str(
                Path(self.args.output_root).expanduser().resolve()
            ),
        }
        mismatches = {
            key: {
                "expected": expected,
                "provided": receipt.get(key),
            }
            for key, expected in expected_values.items()
            if receipt.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                "Expected the tracker gate receipt to come from the reviewed "
                f"canonical validator. Provided mismatches: {mismatches!r}."
            )
        observed_status = receipt["status"]
        if (receipt["gate_stage"], observed_status) not in {
            ("canary", "preflighting"),
            ("production", "launch-ready"),
        }:
            raise RuntimeError(
                "Expected a stage-appropriate tracker gate receipt. "
                f"Provided value: stage={receipt['gate_stage']!r}, "
                f"status={observed_status!r}."
            )
        if (
            required_status is not None
            and observed_status != required_status
        ):
            raise RuntimeError(
                f"Expected tracker experiment {EXPERIMENT_ID!r} to have "
                f"Status {required_status!r}. Provided value: "
                f"{observed_status!r}."
            )
        expected_stage = {
            "preflighting": "canary",
            "launch-ready": "production",
        }.get(required_status)
        if expected_stage is not None and receipt["gate_stage"] != expected_stage:
            raise RuntimeError(
                "Expected the tracker gate receipt stage to match the next "
                f"scheduler side effect. Provided value: "
                f"expected={expected_stage!r}, "
                f"observed={receipt['gate_stage']!r}."
            )
        for key in ("tracker_sha256", "validator_sha256"):
            if (
                not isinstance(receipt[key], str)
                or re.fullmatch(r"[0-9a-f]{64}", receipt[key]) is None
            ):
                raise _error(
                    f"tracker gate receipt {key} to be a lowercase SHA-256",
                    receipt[key],
                )
        generated_raw = receipt["generated_at_utc"]
        try:
            generated_at = datetime.fromisoformat(generated_raw)
        except (TypeError, ValueError) as exc:
            raise _error(
                "tracker gate generated_at_utc to be an ISO-8601 timestamp",
                generated_raw,
            ) from exc
        if generated_at.tzinfo is None:
            raise _error(
                "tracker gate generated_at_utc to include a timezone",
                generated_raw,
            )
        age_seconds = (
            datetime.now(timezone.utc)
            - generated_at.astimezone(timezone.utc)
        ).total_seconds()
        if required_status is not None and not (
            -60 <= age_seconds <= TRACKER_GATE_MAX_AGE_SECONDS
        ):
            raise RuntimeError(
                "Expected the tracker gate receipt to be fresh immediately "
                f"before submission. Provided age_seconds={age_seconds:.3f}, "
                f"maximum={TRACKER_GATE_MAX_AGE_SECONDS}."
            )
        return {
            "status": observed_status,
            "experiment_id": EXPERIMENT_ID,
            "required_status": required_status,
            "gate_stage": receipt["gate_stage"],
            "tracker_path": receipt["tracker_path"],
            "tracker_sha256": receipt["tracker_sha256"],
            "validator_path": receipt["validator_path"],
            "validator_sha256": receipt["validator_sha256"],
            "receipt_path": str(receipt_path),
            "receipt_sha256": observed_receipt_sha256,
            "age_seconds": age_seconds,
        }

    def _submission_args(
        self, kind: str, output_root: str
    ) -> argparse.Namespace:
        return argparse.Namespace(
            kind=kind,
            bundle_dir=self.args.bundle_dir,
            output_root=output_root,
            data_root=self.args.data_root,
            source_archive=self.args.source_archive,
            environment_contract=self.args.environment_contract,
            launch_authorization_receipt=(
                self.args.launch_authorization_receipt
            ),
            launch_authorization_receipt_sha256=(
                self.args.launch_authorization_receipt_sha256
            ),
            repo_root=self.args.repo_root,
            python=self.args.python,
            official_verifier=self.args.official_verifier,
            remote_user=self.args.remote_user,
            canary_pack_index=self.args.canary_pack_index,
            submit=False,
            test_only=False,
            preflight_receipt=self.args.preflight_receipt,
            scheduled_preflight_receipt=(
                self.args.scheduled_preflight_receipt
            ),
            canary_receipt=self.args.canary_receipt,
            canary_receipt_sha256=None,
            official_canary_receipt=self.args.official_canary_receipt,
            supervisor_state=self.args.state,
        )

    def initialize(self) -> None:
        self._note_failure_operation(
            name="supervisor_initialization",
            stage="initialization",
        )
        require_login_node_context()
        existing_state: dict[str, Any] | None = None
        if self.state_path.is_file():
            loaded = read_json(self.state_path)
            if not isinstance(loaded, dict):
                raise _error("supervisor state to be a JSON object", loaded)
            existing_state = loaded
            # Recover the persisted state before any fallible revalidation so
            # a preflight timeout can still report known jobs and mark the
            # attempt terminal.
            self.state = existing_state
        self._reject_terminal_attempt()
        if not self.state_path.parent.is_dir():
            raise _error(
                "supervisor state parent to be an existing directory",
                self.state_path.parent,
            )
        resuming = existing_state is not None
        if not resuming:
            self._require_fresh_canary_receipt_paths()
        tracker_gate = self._validate_tracker(
            required_status=None if resuming else "preflighting",
        )
        output_root = Path(self.args.output_root).expanduser().resolve()
        canary_command, canary_metadata = prepare_submission(
            self._submission_args("canary", str(output_root))
        )
        production_command, production_metadata = prepare_submission(
            self._submission_args("production", str(output_root))
        )
        preflight = run_wrapper_preflight(
            canary_metadata,
            runner=self.runner,
        )
        production_preflight = run_wrapper_preflight(
            production_metadata,
            runner=self.runner,
        )
        if preflight["runtime_preflight"] != production_preflight[
            "runtime_preflight"
        ]:
            raise RuntimeError(
                "Expected canary and production runtime preflight output to be "
                "identical. Provided distinct values."
            )
        scheduled_path, scheduled_hash = (
            ensure_scheduled_preflight_receipt(
                self.args.scheduled_preflight_receipt,
                metadata=canary_metadata,
                runtime_preflight=preflight["runtime_preflight"],
                create=True,
                runner=self.runner,
            )
        )
        authorization = validate_launch_authorization_receipt(
            self.args.launch_authorization_receipt,
            expected_sha256=self.args.launch_authorization_receipt_sha256,
            metadata=canary_metadata,
            runtime_preflight=preflight["runtime_preflight"],
            runner=self.runner,
        )
        append_launch_template_exports(
            canary_command,
            authorization=authorization,
            scheduled_preflight_receipt_path=scheduled_path,
            scheduled_preflight_receipt_sha256=scheduled_hash,
            include_canary_gate=False,
        )
        append_launch_template_exports(
            production_command,
            authorization=authorization,
            scheduled_preflight_receipt_path=scheduled_path,
            scheduled_preflight_receipt_sha256=scheduled_hash,
            include_canary_gate=True,
        )
        launch_contract = launch_contract_value(
            canary_command_template=canary_command,
            canary_metadata=canary_metadata,
            production_command_template=production_command,
            production_metadata=production_metadata,
            authorization=authorization,
            scheduled_preflight_receipt_path=scheduled_path,
            scheduled_preflight_receipt_sha256=scheduled_hash,
        )
        input_binding = immutable_input_binding(
            canary_metadata,
            preflight["runtime_preflight"],
            scheduled_preflight_receipt_path=scheduled_path,
            scheduled_preflight_receipt_sha256=scheduled_hash,
            launch_authorization_receipt_path=authorization[
                "receipt_path"
            ],
            launch_authorization_receipt_sha256=authorization[
                "receipt_sha256"
            ],
            approved_plan=authorization["approved_plan"],
            launch_contract=launch_contract,
        )
        production_input = immutable_input_binding(
            production_metadata,
            production_preflight["runtime_preflight"],
            scheduled_preflight_receipt_path=scheduled_path,
            scheduled_preflight_receipt_sha256=scheduled_hash,
            launch_authorization_receipt_path=authorization[
                "receipt_path"
            ],
            launch_authorization_receipt_sha256=authorization[
                "receipt_sha256"
            ],
            approved_plan=authorization["approved_plan"],
            launch_contract=launch_contract,
        )
        if production_input != input_binding:
            raise RuntimeError(
                "Expected canary and production to retain identical immutable "
                f"scientific/source inputs. Provided value: "
                f"canary={input_binding!r}, production={production_input!r}."
            )
        preflight_value = preflight_receipt_value(
            input_binding=input_binding,
            runtime_preflight=preflight["runtime_preflight"],
        )
        preflight_path, preflight_hash = ensure_preflight_receipt(
            self.args.preflight_receipt,
            expected_value=preflight_value,
            create=True,
        )
        gate_binding = canary_gate_binding(
            input_binding=input_binding,
            preflight_receipt_path=preflight_path,
            preflight_receipt_sha256=preflight_hash,
        )
        canary_command = materialize_launch_command(
            canary_command,
            preflight_receipt_path=preflight_path,
            preflight_receipt_sha256=preflight_hash,
        )
        production_command = materialize_launch_command(
            production_command,
            preflight_receipt_path=preflight_path,
            preflight_receipt_sha256=preflight_hash,
        )
        log_dir = output_root / "slurm"
        if log_dir.is_symlink():
            raise _error(
                "shared successor Slurm log directory not to be a symlink",
                log_dir,
            )
        log_dir.mkdir(parents=True, exist_ok=True)
        if not log_dir.is_dir() or log_dir.resolve() != log_dir:
            raise _error(
                "shared successor Slurm log directory to be a real directory",
                log_dir,
            )
        allocation = _validate_live_allocation(
            self.allocation_verifier(runner=self.runner)
        )
        binding = {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "gate_binding": gate_binding,
            "gate_binding_sha256": _canonical_json_sha256(gate_binding),
            "output_root": str(output_root),
            "canary_command": canary_command,
            "production_command": production_command,
            "canary_receipt_path": str(
                Path(self.args.canary_receipt).expanduser().resolve()
            ),
            "official_canary_receipt_path": str(
                Path(self.args.official_canary_receipt).expanduser().resolve()
            ),
            "preflight_receipt_path": str(preflight_path),
            "preflight_receipt_sha256": preflight_hash,
            "scheduled_preflight_receipt_path": str(scheduled_path),
            "scheduled_preflight_receipt_sha256": scheduled_hash,
            "launch_authorization_receipt_path": authorization[
                "receipt_path"
            ],
            "launch_authorization_receipt_sha256": authorization[
                "receipt_sha256"
            ],
            "official_verifier_path": str(
                Path(
                    str(canary_metadata["official_canary_verifier"])
                ).expanduser().resolve()
            ),
        }
        if existing_state is not None:
            state = existing_state
            if state.get("schema_version") != STATE_SCHEMA_VERSION:
                raise _error(
                    f"supervisor schema_version to be {STATE_SCHEMA_VERSION!r}",
                    state.get("schema_version"),
                )
            if state.get("binding") != binding:
                raise RuntimeError(
                    "Expected a resumed successor supervisor to retain exact "
                    f"commands, paths, and hashes. Provided value: "
                    f"expected={binding!r}, observed={state.get('binding')!r}."
                )
            state["live_allocation_audit"] = allocation
        else:
            smoke_jobs = (
                output_root
                / "smoke"
                / str(gate_binding["bundle_id"])
                / "jobs"
            )
            if smoke_jobs.exists() or (output_root / "entries").exists():
                raise RuntimeError(
                    "Expected a new supervisor state before any canary or "
                    "production output exists; refusing ambiguous adoption. "
                    f"Provided value: smoke_jobs={smoke_jobs}, "
                    f"entries={output_root / 'entries'}."
                )
            state = {
                "schema_version": STATE_SCHEMA_VERSION,
                "status": "active",
                "binding": binding,
                "live_allocation_audit": allocation,
                "stages": {
                    "canary": {"status": "ready"},
                    "production": {"status": "blocked_on_canary"},
                },
            }
        state["last_tracker_gate"] = tracker_gate
        self.state = state
        self.canary_command = canary_command
        self.production_command = production_command
        self.gate_binding = gate_binding
        self.runtime_preflight = dict(preflight["runtime_preflight"])
        self.official_verifier = str(
            canary_metadata["official_canary_verifier"]
        )
        self._save()

    def _require_initialized(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.state is None or self.gate_binding is None:
            raise RuntimeError("Expected the successor supervisor to initialize.")
        return self.state, self.gate_binding

    def _save(self) -> None:
        state, _gate = self._require_initialized()
        atomic_write_json(self.state_path, state, canonical=True)

    def _post_submit_readback(
        self,
        *,
        stage: str,
        job_id: str,
    ) -> dict[str, Any]:
        _state, gate = self._require_initialized()
        launch_contract = gate.get("launch_contract")
        if not isinstance(launch_contract, Mapping):
            raise _error(
                "gate binding launch_contract to be a JSON object",
                launch_contract,
            )
        return verify_submitted_job_contract(
            job_id=job_id,
            kind=stage,
            metadata={
                "resources": launch_contract.get("resources"),
                "repo_root": gate.get("repo_root"),
                "wrapper": gate.get("wrapper"),
                "output_root": launch_contract.get("output_root"),
            },
            runner=self.runner,
        )

    def _submit_once(
        self,
        stage: str,
        command: Sequence[str],
    ) -> str:
        state, _gate = self._require_initialized()
        if state.get("status") == "failed":
            raise RuntimeError(
                "Expected a terminal failed attempt never to submit or resume."
            )
        record = state["stages"][stage]
        if str(record.get("status", "")).startswith("failed"):
            raise RuntimeError(
                "Expected a failed stage to be terminal for this attempt. "
                f"Provided value: stage={stage!r}, status="
                f"{record.get('status')!r}."
            )
        if record.get("job_id") is not None:
            if record.get("submission_status") in {
                "submitted_contract_mismatch",
                "ambiguous",
            }:
                raise RuntimeError(
                    "Expected authoritative reconciliation before resuming a "
                    f"non-verified submission. Provided value: {record!r}."
                )
            job_id = record["job_id"]
            if not isinstance(job_id, str) or not job_id.isdigit():
                raise _error("recorded Slurm job id to be numeric", job_id)
            submission_status = record.get("submission_status")
            if submission_status == "submitted_unverified":
                self._note_failure_operation(
                    name="post_submission_contract_readback",
                    stage=stage,
                    command=[
                        "scontrol",
                        "show",
                        "job",
                        job_id,
                        "-o",
                    ],
                )
                try:
                    readback = self._post_submit_readback(
                        stage=stage,
                        job_id=job_id,
                    )
                except BaseException as exc:
                    record.update(
                        {
                            "submission_status": (
                                "submitted_contract_mismatch"
                            ),
                            "post_submit_error": repr(exc),
                        }
                    )
                    self._save()
                    raise
                record.update(
                    {
                        "submission_status": "submitted",
                        "post_submit_readback": readback,
                    }
                )
                self._save()
                return job_id
            if submission_status != "submitted":
                raise RuntimeError(
                    "Expected a recorded job id to be either submitted or "
                    "submitted_unverified for immediate reconciliation. "
                    f"Provided value: stage={stage!r}, record={record!r}."
                )
            return job_id
        previous = record.get("submission_status")
        if previous in {"started", "ambiguous"}:
            raise RuntimeError(
                "Expected no ambiguous prior sbatch call before resumption; "
                f"refusing duplicate submission. Provided value: "
                f"stage={stage!r}, status={previous!r}."
            )
        if not self.args.enable_submit:
            raise RuntimeError(
                "Expected --enable-submit before the supervisor may call "
                f"sbatch. Provided pending stage: {stage!r}."
            )
        self._note_failure_operation(
            name="pre_submission_tracker_gate",
            stage=stage,
        )
        tracker_gate = self._validate_tracker(
            required_status=(
                "preflighting" if stage == "canary" else "launch-ready"
            ),
        )
        record["last_tracker_gate"] = tracker_gate
        self._save()
        self._note_failure_operation(
            name="live_allocation_audit",
            stage=stage,
        )
        live_allocation = _validate_live_allocation(
            self.allocation_verifier(runner=self.runner)
        )
        test_command = [command[0], "--test-only", *command[1:]]
        self._note_failure_operation(
            name="scheduler_test_only",
            stage=stage,
            command=test_command,
        )
        try:
            tested = self.runner(
                test_command,
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            record.update(
                {
                    "status": "failed_sbatch_test_only",
                    "test_only_command": test_command,
                    "test_only_error": str(exc),
                    "test_only_allocation_audit": live_allocation,
                }
            )
            self._save()
            raise
        test_record = {
            "command": test_command,
            "stdout": tested.stdout,
            "stderr": tested.stderr,
            "allocation_audit": live_allocation,
            "binding_sha256": self.state["binding"][
                "gate_binding_sha256"
            ],
        }
        record.setdefault("test_only_attempts", []).append(test_record)
        record["last_test_only"] = test_record
        self._save()
        required_tracker_status = (
            "preflighting" if stage == "canary" else "launch-ready"
        )
        self._note_failure_operation(
            name="immediate_live_submission_tracker_gate",
            stage=stage,
        )
        final_tracker_gate = self._validate_tracker(
            required_status=required_tracker_status,
        )
        record["live_submission_tracker_gate"] = final_tracker_gate
        self._save()
        record.update(
            {
                "submission_status": "started",
                "command": list(command),
            }
        )
        self._save()
        self._note_failure_operation(
            name="scheduler_submission",
            stage=stage,
            command=command,
        )
        try:
            completed = self.runner(
                list(command),
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            record.update(
                {
                    "submission_status": "ambiguous",
                    "submission_error": str(exc),
                    "submission_returncode": exc.returncode,
                    "submission_stdout": exc.stdout,
                    "submission_stderr": exc.stderr,
                }
            )
            self._save()
            raise
        except BaseException as exc:
            record["submission_status"] = "ambiguous"
            record["submission_error"] = repr(exc)
            self._save()
            raise
        try:
            job_id = _submitted_job_id(completed)
        except BaseException as exc:
            record.update(
                {
                    "submission_status": "ambiguous",
                    "submission_stdout": completed.stdout,
                    "submission_stderr": completed.stderr,
                    "submission_error": repr(exc),
                }
            )
            self._save()
            raise
        record.update(
            {
                "job_id": job_id,
                "submission_status": "submitted_unverified",
                "status": "waiting",
                "submission_stdout": completed.stdout,
                "submission_stderr": completed.stderr,
            }
        )
        self._save()
        self._note_failure_operation(
            name="post_submission_contract_readback",
            stage=stage,
            command=["scontrol", "show", "job", job_id, "-o"],
        )
        try:
            readback = self._post_submit_readback(
                stage=stage,
                job_id=job_id,
            )
        except BaseException as exc:
            record.update(
                {
                    "submission_status": "submitted_contract_mismatch",
                    "post_submit_error": repr(exc),
                }
            )
            self._save()
            raise
        record.update(
            {
                "submission_status": "submitted",
                "post_submit_readback": readback,
            }
        )
        self._save()
        return job_id

    def _poll_stage(
        self,
        stage: str,
        *,
        job_id: str,
        expected_tasks: int,
    ) -> dict[str, Any]:
        state, _gate = self._require_initialized()
        accounting_command = [
            "sacct",
            "-n",
            "-X",
            "-j",
            job_id,
            "--format=JobIDRaw,JobID,State,ExitCode,Account,Partition,QOS,"
            "Constraints",
            "--parsable2",
        ]
        self._note_failure_operation(
            name="scheduler_accounting",
            stage=stage,
            command=accounting_command,
        )
        try:
            value = query_successor_array_state(
                job_id,
                expected_tasks,
                runner=self.runner,
            )
        except RuntimeError as exc:
            state["stages"][stage]["status"] = "failed_accounting_validation"
            state["stages"][stage]["accounting_error"] = str(exc)
            self._save()
            raise
        state["stages"][stage]["last_array_state"] = value
        self._save()
        return value

    def _revalidate_before_production(self) -> None:
        state, gate = self._require_initialized()
        self._note_failure_operation(
            name="production_immutable_input_revalidation",
            stage="production",
        )
        canary_receipt = state["stages"]["canary"].get("gate_receipt_path")
        expected_canary_hash = state["stages"]["canary"].get(
            "gate_receipt_sha256"
        )
        if not isinstance(canary_receipt, str):
            raise RuntimeError(
                "Expected a verified canary receipt before production."
            )
        if (
            not isinstance(expected_canary_hash, str)
            or sha256_file(canary_receipt) != expected_canary_hash
        ):
            raise RuntimeError(
                "Expected the verified canary receipt hash to remain unchanged "
                f"before production. Provided value: path={canary_receipt!r}, "
                f"recorded_sha256={expected_canary_hash!r}."
            )
        validate_canary_gate_receipt(
            canary_receipt,
            expected_gate_binding=gate,
        )
        command, metadata = prepare_submission(
            self._submission_args(
                "production", self.args.output_root
            )
        )
        canary_command, canary_metadata = prepare_submission(
            self._submission_args(
                "canary", self.args.output_root
            )
        )
        preflight = run_wrapper_preflight(metadata, runner=self.runner)
        authorization = validate_launch_authorization_receipt(
            gate["launch_authorization_receipt_path"],
            expected_sha256=gate[
                "launch_authorization_receipt_sha256"
            ],
            metadata=metadata,
            runtime_preflight=preflight["runtime_preflight"],
            runner=self.runner,
        )
        scheduled = ensure_scheduled_preflight_receipt(
            gate["scheduled_preflight_receipt_path"],
            metadata=metadata,
            runtime_preflight=preflight["runtime_preflight"],
            create=False,
            runner=self.runner,
        )
        if scheduled[1] != gate["scheduled_preflight_receipt_sha256"]:
            raise RuntimeError(
                "Expected the official scheduled-run preflight receipt to "
                f"remain unchanged after canary. Provided value: {scheduled!r}."
            )
        append_launch_template_exports(
            canary_command,
            authorization=authorization,
            scheduled_preflight_receipt_path=scheduled[0],
            scheduled_preflight_receipt_sha256=scheduled[1],
            include_canary_gate=False,
        )
        append_launch_template_exports(
            command,
            authorization=authorization,
            scheduled_preflight_receipt_path=scheduled[0],
            scheduled_preflight_receipt_sha256=scheduled[1],
            include_canary_gate=True,
        )
        launch_contract = launch_contract_value(
            canary_command_template=canary_command,
            canary_metadata=canary_metadata,
            production_command_template=command,
            production_metadata=metadata,
            authorization=authorization,
            scheduled_preflight_receipt_path=scheduled[0],
            scheduled_preflight_receipt_sha256=scheduled[1],
        )
        current_input = immutable_input_binding(
            metadata,
            preflight["runtime_preflight"],
            scheduled_preflight_receipt_path=scheduled[0],
            scheduled_preflight_receipt_sha256=scheduled[1],
            launch_authorization_receipt_path=authorization[
                "receipt_path"
            ],
            launch_authorization_receipt_sha256=authorization[
                "receipt_sha256"
            ],
            approved_plan=authorization["approved_plan"],
            launch_contract=launch_contract,
        )
        expected_input = {
            key: value
            for key, value in gate.items()
            if key
            not in {
                "preflight_receipt_path",
                "preflight_receipt_sha256",
            }
        }
        if current_input != expected_input:
            raise RuntimeError(
                "Expected source, bundle, runtime, wrapper, environment, and "
                "preflight identities to remain unchanged after canary. "
                f"Provided value: expected={expected_input!r}, "
                f"observed={current_input!r}."
            )
        receipt = ensure_preflight_receipt(
            gate["preflight_receipt_path"],
            expected_value=preflight_receipt_value(
                input_binding=current_input,
                runtime_preflight=preflight["runtime_preflight"],
            ),
            create=False,
        )
        if receipt[1] != gate["preflight_receipt_sha256"]:
            raise RuntimeError(
                "Expected the preflight receipt to remain unchanged after "
                f"canary. Provided value: {receipt!r}."
            )
        command = materialize_launch_command(
            command,
            preflight_receipt_path=receipt[0],
            preflight_receipt_sha256=receipt[1],
        )
        expected_production_command = materialize_canary_gate_command(
            self.production_command or [],
            canary_receipt_path=canary_receipt,
            canary_receipt_sha256=expected_canary_hash,
        )
        command = materialize_canary_gate_command(
            command,
            canary_receipt_path=canary_receipt,
            canary_receipt_sha256=expected_canary_hash,
        )
        if command != expected_production_command:
            raise RuntimeError(
                "Expected the production sbatch command to remain byte-for-byte "
                f"unchanged after canary. Provided value: {command!r}."
            )
        self.production_command = command

    def _advance_canary(self) -> dict[str, Any]:
        state, gate = self._require_initialized()
        record = state["stages"]["canary"]
        if record.get("status") == "complete":
            return {"status": "stage_complete", "stage": "canary"}
        if record.get("job_id") is None and not self.args.enable_submit:
            return {
                "status": "submission_required",
                "stage": "canary",
                "command": self.canary_command,
            }
        job_id = self._submit_once("canary", self.canary_command or [])
        array = self._poll_stage(
            "canary",
            job_id=job_id,
            expected_tasks=1,
        )
        if array["status"] == "waiting":
            return {
                "status": "waiting",
                "stage": "canary",
                "job_id": job_id,
                "array": array,
            }
        smoke_receipt = current_canary_smoke_receipt(
            output_root=self.args.output_root,
            bundle_id=str(gate["bundle_id"]),
            array_job_id=job_id,
        )
        canary_log_root = (
            Path(self.args.output_root).expanduser().resolve()
            / "slurm"
        )
        canary_stdout = (
            canary_log_root
            / f"pd-successor-canary-{job_id}_0.out"
        )
        canary_stderr = (
            canary_log_root
            / f"pd-successor-canary-{job_id}_0.err"
        )
        self._note_failure_operation(
            name="canary_semantic_verification",
            stage="canary",
            details={
                "smoke_receipt": str(smoke_receipt),
                "stdout_log": str(canary_stdout),
                "stderr_log": str(canary_stderr),
                "official_verifier": str(self.official_verifier),
            },
        )
        result = verify_canary(
            job_id=job_id,
            smoke_receipt=smoke_receipt,
            output_root=self.args.output_root,
            gate_binding=gate,
            official_receipt=self.args.official_canary_receipt,
            receipt=self.args.canary_receipt,
            canary_stdout_log=canary_stdout,
            canary_stderr_log=canary_stderr,
            official_verifier=(
                self.official_verifier
                or self.args.official_verifier
            ),
            runner=self.runner,
        )
        record.update(
            {
                "status": "complete",
                "gate_receipt_path": result["receipt_path"],
                "gate_receipt_sha256": result["receipt_sha256"],
                "official_gate_receipt_path": result["official_gate"][
                    "receipt_path"
                ],
                "smoke_receipt_path": str(smoke_receipt),
                "smoke_receipt_sha256": result["successor_semantics"][
                    "smoke_receipt_sha256"
                ],
            }
        )
        state["stages"]["production"]["status"] = "ready"
        self._save()
        return {
            "status": "stage_complete",
            "stage": "canary",
            "job_id": job_id,
            "gate_receipt": result["receipt_path"],
        }

    def _current_completion_identity(self) -> dict[str, Any]:
        state, gate = self._require_initialized()
        canary_path_value = state["stages"]["canary"].get(
            "gate_receipt_path"
        )
        canary_sha = state["stages"]["canary"].get(
            "gate_receipt_sha256"
        )
        if not isinstance(canary_path_value, str):
            raise RuntimeError(
                "Expected the verified canary-gate receipt path before "
                "validating production completion markers."
            )
        canary_path = Path(canary_path_value).expanduser().resolve()
        if (
            not isinstance(canary_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", canary_sha)
            or not canary_path.is_file()
            or canary_path.is_symlink()
            or sha256_file(canary_path) != canary_sha
        ):
            raise RuntimeError(
                "Expected the verified canary-gate receipt to remain an "
                "unchanged non-symlink file before production status "
                f"validation. Provided value: path={canary_path}, "
                f"sha256={canary_sha!r}."
            )
        validate_canary_gate_receipt(
            canary_path,
            expected_gate_binding=gate,
        )
        return {
            **gate,
            "canary_gate_receipt_path": str(canary_path),
            "canary_gate_receipt_sha256": canary_sha,
        }

    def _fetch_production_status(self) -> dict[str, Any]:
        command = [
            self.args.python,
            str(
                Path(self.args.repo_root).expanduser().resolve()
                / "experiments"
                / "run_mnist_conv_perfectdiode_successor_confirmation.py"
            ),
            "status",
            "--bundle-dir",
            str(Path(self.args.bundle_dir).expanduser().resolve()),
            "--output-root",
            str(
                Path(self.args.output_root).expanduser().resolve()
            ),
        ]
        self._note_failure_operation(
            name="production_status_snapshot",
            stage="production",
            command=command,
        )
        completed = self.runner(
            command,
            check=True,
            text=True,
            capture_output=True,
        )
        value = _parse_result_marker(
            completed.stdout,
            label="successor production status",
        )
        return value

    def _inspect_first_production_artifact(self) -> dict[str, Any] | None:
        state, _gate = self._require_initialized()
        record = state["stages"]["production"]
        existing = record.get("first_production_artifact_verified")
        if isinstance(existing, dict):
            return existing
        entries_root = (
            Path(self.args.output_root).expanduser().resolve() / "entries"
        )
        published = (
            []
            if not entries_root.is_dir()
            else [
                candidate
                for candidate in entries_root.glob("*/completion.json")
                if candidate.exists() or candidate.is_symlink()
            ]
        )
        if not published:
            return None
        value = self._fetch_production_status()
        validated = validate_partial_production_status(
            value,
            output_root=self.args.output_root,
            expected_identity=self._current_completion_identity(),
        )
        completions = validated["completion_markers"]
        if not completions:
            raise RuntimeError(
                "Expected a status snapshot taken after observing a "
                "completion.json to validate at least one published entry. "
                f"Provided value: {[str(path) for path in published]!r}."
            )
        return self._persist_first_production_artifact(
            value=value,
            completions=completions,
        )

    def _persist_first_production_artifact(
        self,
        *,
        value: Mapping[str, Any],
        completions: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        state, _gate = self._require_initialized()
        record = state["stages"]["production"]
        existing = record.get("first_production_artifact_verified")
        if isinstance(existing, dict):
            return existing
        if not completions:
            raise RuntimeError(
                "Expected at least one validated completion before recording "
                "first production-artifact evidence."
            )
        first = completions[0]
        status_path = (
            self.state_path.parent
            / "first_production_artifact_status.json"
        )
        atomic_write_json(status_path, value, canonical=True)
        evidence = {
            "entry_index": first["entry_index"],
            "entry_id": first["entry_id"],
            "completion_path": first["completion_path"],
            "completion_sha256": first["completion_sha256"],
            "outputs": first["outputs"],
            "status_path": str(status_path),
            "status_sha256": sha256_file(status_path),
            "validated_completed_entry_count": len(completions),
        }
        record["first_production_artifact_verified"] = evidence
        self._save()
        return evidence

    def _run_production_status(self) -> dict[str, Any]:
        state, _gate = self._require_initialized()
        value = self._fetch_production_status()
        validated = validate_production_status(
            value,
            output_root=self.args.output_root,
            expected_identity=self._current_completion_identity(),
        )
        self._persist_first_production_artifact(
            value=value,
            completions=validated["completion_markers"],
        )
        status_path = self.state_path.parent / "production_status.json"
        atomic_write_json(status_path, value, canonical=True)
        state["stages"]["production"].update(
            {
                "status_path": str(status_path),
                "status_sha256": sha256_file(status_path),
                "validated_completion_markers": validated[
                    "completion_markers"
                ],
            }
        )
        self._save()
        return value

    def _advance_production(self) -> dict[str, Any]:
        state, _gate = self._require_initialized()
        record = state["stages"]["production"]
        if record.get("status") == "complete":
            return {"status": "stage_complete", "stage": "production"}
        if record.get("status") == "failed_first_artifact_validation":
            raise RuntimeError(
                "Expected authoritative reconciliation after a failed first "
                "production-artifact validation. Provided value: "
                f"{record.get('first_artifact_validation_error')!r}."
            )
        self._revalidate_before_production()
        if record.get("job_id") is None and not self.args.enable_submit:
            return {
                "status": "submission_required",
                "stage": "production",
                "command": self.production_command,
            }
        job_id = self._submit_once(
            "production", self.production_command or []
        )
        array = self._poll_stage(
            "production",
            job_id=job_id,
            expected_tasks=PRODUCTION_TASK_COUNT,
        )
        if array["status"] == "waiting":
            try:
                self._note_failure_operation(
                    name="first_production_artifact_validation",
                    stage="production",
                )
                first_artifact = self._inspect_first_production_artifact()
            except BaseException as exc:
                record.update(
                    {
                        "status": "failed_first_artifact_validation",
                        "first_artifact_validation_error": repr(exc),
                    }
                )
                self._save()
                raise
            return {
                "status": "waiting",
                "stage": "production",
                "job_id": job_id,
                "array": array,
                "first_production_artifact_verified": first_artifact,
            }
        self._run_production_status()
        record["status"] = "complete"
        self._save()
        return {
            "status": "stage_complete",
            "stage": "production",
            "job_id": job_id,
        }

    def advance_once(self) -> dict[str, Any]:
        state, _gate = self._require_initialized()
        if state.get("status") == "failed":
            raise RuntimeError(
                "Expected a terminal failed attempt never to advance."
            )
        if state["stages"]["canary"].get("status") != "complete":
            return self._advance_canary()
        if state["stages"]["production"].get("status") != "complete":
            return self._advance_production()
        state["status"] = "complete"
        self._save()
        return {
            "status": "complete",
            "state_path": str(self.state_path),
            "canary_job_id": state["stages"]["canary"]["job_id"],
            "production_job_id": state["stages"]["production"]["job_id"],
        }

    def run(
        self,
        *,
        poll_seconds: float,
        once: bool,
        max_follow_seconds: float = DEFAULT_MAX_FOLLOW_SECONDS,
    ) -> dict[str, Any]:
        if (
            isinstance(poll_seconds, bool)
            or not isinstance(poll_seconds, (int, float))
            or not 1 <= float(poll_seconds) <= 60
        ):
            raise _error(
                "poll-seconds to be a number from 1 through 60",
                poll_seconds,
            )
        follow_limit = _bounded_follow_timeout(max_follow_seconds)
        follow_started = self.clock()
        previous_deadline = self.follow_deadline
        self.follow_deadline = follow_started + follow_limit
        try:
            while True:
                if self.clock() >= self.follow_deadline:
                    raise TimeoutError(
                        "Expected explicit supervision to make terminal "
                        f"progress within {follow_limit:g} seconds before "
                        "the next reconciliation pass."
                    )
                result = self.advance_once()
                print(
                    json.dumps(result, sort_keys=True, allow_nan=False),
                    flush=True,
                )
                elapsed = max(self.clock() - follow_started, 0.0)
                if elapsed >= follow_limit:
                    raise TimeoutError(
                        "Expected explicit supervision to make terminal "
                        f"progress within {follow_limit:g} seconds. "
                        f"Provided elapsed={elapsed:.3f}, "
                        f"last_result={result!r}."
                    )
                if (
                    once
                    or result["status"]
                    in {
                        "complete",
                        "submission_required",
                        "stage_complete",
                    }
                ):
                    return result
                self.sleeper(
                    min(float(poll_seconds), follow_limit - elapsed)
                )
        finally:
            self.follow_deadline = previous_deadline


def _parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--environment-contract", required=True)
    parser.add_argument("--launch-authorization-receipt", required=True)
    parser.add_argument(
        "--launch-authorization-receipt-sha256",
        required=True,
    )
    parser.add_argument("--scheduled-preflight-receipt", required=True)
    parser.add_argument("--preflight-receipt", required=True)
    parser.add_argument("--tracker-gate-receipt", required=True)
    parser.add_argument(
        "--tracker-gate-receipt-sha256",
        required=True,
    )
    parser.add_argument("--official-canary-receipt", required=True)
    parser.add_argument("--canary-receipt", required=True)
    parser.add_argument("--repo-root", default=str(repo))
    parser.add_argument("--python", default=PYTHON_EXECUTABLE)
    parser.add_argument("--remote-user", default=os.environ.get("USER", ""))
    parser.add_argument(
        "--canary-pack-index",
        type=int,
        default=DEFAULT_CANARY_PACK_INDEX,
    )
    parser.add_argument(
        "--official-verifier",
        default=None,
        help=(
            "Reviewed staged verifier. By default derive the environment-"
            "contract launch_tools path from --remote-user."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    monitor_mode = parser.add_mutually_exclusive_group()
    monitor_mode.add_argument(
        "--once",
        dest="once",
        action="store_true",
        help="Perform one reconciliation pass (the default).",
    )
    monitor_mode.add_argument(
        "--follow",
        dest="once",
        action="store_false",
        help=(
            "Explicitly follow the current stage until its boundary or the "
            "hard deadline; never cross canary into production."
        ),
    )
    parser.set_defaults(once=True)
    parser.add_argument(
        "--max-follow-seconds",
        type=float,
        default=DEFAULT_MAX_FOLLOW_SECONDS,
    )
    parser.add_argument(
        "--command-timeout-seconds",
        type=float,
        default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--enable-submit",
        action="store_true",
        help="Allow this state machine to call sbatch.",
    )
    parser.add_argument(
        "--arm-long-run",
        action="store_true",
        help=(
            "Explicitly arm this long simulation attempt. Without this flag "
            "the supervisor performs no state write, validation, monitoring, "
            "or scheduler action."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.arm_long_run:
        raise _error(
            "--arm-long-run before entering this dedicated long-simulation "
            "supervisor",
            args.arm_long_run,
        )
    state = Path(args.state).expanduser().resolve()
    if not state.parent.is_dir():
        raise _error(
            "supervisor state parent to be an existing directory",
            state.parent,
        )
    follow_limit = _bounded_follow_timeout(args.max_follow_seconds)
    lock_path = state.with_name(state.name + ".lock")
    supervisor = SuccessorJeanZaySupervisor(args=args)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            # A contender does not own the attempt and therefore must never
            # poison the lock owner's state or failure sentinel.
            raise RuntimeError(
                "Expected exactly one active successor supervisor. "
                f"Provided locked path: {lock_path}."
            ) from exc
        # Reject an already terminal attempt or an unusable immutable failure
        # destination before declaring this long-run attempt armed.  Keep the
        # matching check in initialize() to defend against later state changes
        # while the lock is held.
        supervisor._reject_terminal_attempt()
        print(
            "LONG-RUN ATTEMPT ARMED "
            f"experiment_id={EXPERIMENT_ID} "
            "target=jean-zay "
            f"expected_duration_seconds<={LONG_RUN_EXPECTED_MAX_SECONDS} "
            f"hard_deadline_seconds={follow_limit:g} "
            f"state_path={state} "
            f"failure_report_path={supervisor.failure_report_path} "
            f"tracker_gate_receipt={args.tracker_gate_receipt}",
            file=sys.stderr,
            flush=True,
        )
        try:
            supervisor.initialize()
            supervisor.run(
                poll_seconds=args.poll_seconds,
                once=args.once,
                max_follow_seconds=follow_limit,
            )
        except BaseException as exc:
            try:
                report = supervisor.record_failure(exc)
            except BaseException as report_error:
                print(
                    "Failed to persist fail-stop report: "
                    f"{report_error!r}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    FAILURE_JSON_MARKER
                    + json.dumps(
                        report,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
