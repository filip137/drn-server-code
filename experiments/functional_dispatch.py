#!/usr/bin/env python3
"""Bounded, receipt-backed execution of a functional executor attempt.

One target-side supervisor consumes one executor attempt.  It launches the
declared batches without reconstructing commands, verifies the adapter's
compact receipt, writes one attempt result, and never retries a failed job.
Production failures are terminal for that immutable attempt.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

# The scheduled-run checker executes this file by absolute path.  Preserve the
# same module implementation while making the repository namespace importable.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import experiment_executor as executor
from experiments.mnist_conv.io import (
    atomic_create_json,
    atomic_write_json,
    read_json,
)
from experiments.mnist_conv.perfectdiode_functional_smoke import (
    validate_functional_preflight,
)
from experiments.run_mnist_conv_perfectdiode_job import (
    validate_job_completion,
)


DISPATCH_REQUEST_SCHEMA = "functional-dispatch-request/v1"
DISPATCH_RECEIPT_SCHEMA = "functional-dispatch-receipt/v1"
DISPATCH_FAILURE_SCHEMA = "functional-dispatch-failure/v1"
PROGRESS_SCHEMA = "functional-dispatch-progress/v1"
VALIDATION_MAX_SECONDS = 10 * 60
PROGRESS_INTERVAL_SECONDS = 60

Runner = Callable[..., subprocess.CompletedProcess[str]]
PopenFactory = Callable[..., subprocess.Popen[Any]]


class FunctionalDispatchError(RuntimeError):
    """An immutable functional attempt failed before or after launch."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        launched_job_count: int | None = 0,
        live_jobs: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.launched_job_count = launched_job_count
        self.live_jobs = [dict(item) for item in live_jobs]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Expected {label} to be a mapping. "
            f"Provided value: {type(value).__name__}."
        )
    return dict(value)


def _safe_id(value: Any, label: str) -> str:
    text = str(value)
    if executor.ID_RE.fullmatch(text) is None:
        raise ValueError(
            f"Expected {label} to be a lowercase filesystem-safe identifier. "
            f"Provided value: {value!r}."
        )
    return text


def _absolute(value: Any, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"Expected {label} to be an absolute path without '..'. "
            f"Provided value: {value!r}."
        )
    return path


def validate_dispatch_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = _strict_mapping(value, "dispatch request")
    if request.get("schema_version") != DISPATCH_REQUEST_SCHEMA:
        raise ValueError(
            f"Expected schema_version {DISPATCH_REQUEST_SCHEMA!r}. "
            f"Provided value: {request.get('schema_version')!r}."
        )
    run_class = request.get("run_class")
    if run_class not in executor.RUN_CLASSES:
        raise ValueError(
            f"Expected run_class to be one of {sorted(executor.RUN_CLASSES)!r}. "
            f"Provided value: {run_class!r}."
        )
    launch_backend = request.get("launch_backend")
    if launch_backend not in {"tmux", "detached_process"}:
        raise ValueError(
            "Expected launch_backend to be 'tmux' or 'detached_process'. "
            f"Provided value: {launch_backend!r}."
        )
    for key in ("study_id", "proposal_id", "attempt_id", "target_id"):
        _safe_id(request.get(key), key)
    _absolute(request.get("workspace_root"), "workspace_root")
    result_root = _absolute(request.get("result_root"), "result_root")
    state_dir = _absolute(request.get("state_dir"), "state_dir")
    try:
        state_dir.relative_to(result_root)
    except ValueError as exc:
        raise ValueError(
            "Expected state_dir to stay within result_root. "
            f"Provided value: state_dir={state_dir}, result_root={result_root}."
        ) from exc
    expected = request.get("expected_duration_seconds")
    deadline = request.get("hard_deadline_seconds")
    if (
        not isinstance(expected, int)
        or isinstance(expected, bool)
        or expected <= 0
        or not isinstance(deadline, int)
        or isinstance(deadline, bool)
        or deadline < expected
    ):
        raise ValueError(
            "Expected positive integer durations with hard deadline >= expected "
            f"duration. Provided value: expected={expected!r}, "
            f"deadline={deadline!r}."
        )
    if run_class == "validation" and deadline > VALIDATION_MAX_SECONDS:
        raise ValueError(
            f"Expected a validation deadline <= {VALIDATION_MAX_SECONDS}. "
            f"Provided value: {deadline!r}."
        )
    if run_class == "production" and (
        expected < 600 or request.get("authorized") is not True
    ):
        raise ValueError(
            "Expected production duration >=600 seconds and authorized=true. "
            f"Provided value: expected={expected!r}, "
            f"authorized={request.get('authorized')!r}."
        )
    jobs = request.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(
            "Expected jobs to be a non-empty list. "
            f"Provided value: {jobs!r}."
        )
    job_ids: list[str] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(jobs):
        job = _strict_mapping(raw, f"jobs[{index}]")
        job_id = _safe_id(job.get("job_id"), f"jobs[{index}].job_id")
        argv = job.get("worker_argv")
        if not (
            isinstance(argv, list)
            and argv
            and all(isinstance(item, str) and item for item in argv)
        ):
            raise ValueError(
                f"Expected jobs[{index}].worker_argv to be non-empty argv. "
                f"Provided value: {argv!r}."
            )
        _absolute(job.get("output_dir"), f"jobs[{index}].output_dir")
        job_ids.append(job_id)
        by_id[job_id] = job
    if len(job_ids) != len(set(job_ids)):
        raise ValueError(
            "Expected unique job IDs. "
            f"Provided value: {job_ids!r}."
        )
    batches = request.get("launch_batches")
    if not (
        isinstance(batches, list)
        and batches
        and all(
            isinstance(batch, list)
            and batch
            and all(isinstance(item, str) for item in batch)
            for batch in batches
        )
    ):
        raise ValueError(
            "Expected launch_batches to be non-empty job-ID batches. "
            f"Provided value: {batches!r}."
        )
    flattened = [job_id for batch in batches for job_id in batch]
    if flattened != job_ids:
        raise ValueError(
            "Expected launch_batches to cover jobs exactly once in job order. "
            f"Provided value: batches={flattened!r}, jobs={job_ids!r}."
        )
    return request


def build_dispatch_requests(
    plan: Mapping[str, Any],
    *,
    expected_duration_seconds: int,
    hard_deadline_seconds: int,
    production_preflight: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Project a generic plan into one immutable supervisor request per target."""

    if plan.get("schema_version") != executor.EXECUTION_PLAN_SCHEMA:
        raise ValueError(
            f"Expected an {executor.EXECUTION_PLAN_SCHEMA!r} plan. "
            f"Provided value: {plan.get('schema_version')!r}."
        )
    run_class = plan.get("run_class")
    if run_class == "production":
        if (
            not isinstance(production_preflight, Mapping)
            or production_preflight.get("schema_version")
            != executor.PREFLIGHT_RESULT_SCHEMA
            or production_preflight.get("status") != "passed"
            or production_preflight.get("proposal_id") != plan.get("proposal_id")
        ):
            raise ValueError(
                "Expected a passing production functional-preflight result "
                "bound to this proposal. "
                f"Provided value: {production_preflight!r}."
            )
    requests: list[dict[str, Any]] = []
    for attempt in plan.get("attempts", []):
        result_root = _absolute(attempt["result_root"], "attempt.result_root")
        state_dir = result_root / "_dispatch" / attempt["attempt_id"]
        request = {
            "schema_version": DISPATCH_REQUEST_SCHEMA,
            "study_id": plan["study_id"],
            "proposal_id": plan["proposal_id"],
            "attempt_id": attempt["attempt_id"],
            "target_id": attempt["target_id"],
            "run_class": run_class,
            "launch_backend": (
                "detached_process"
                if attempt.get("target_kind") == "ssh_process"
                else "tmux"
            ),
            "workspace_root": attempt["workspace_root"],
            "result_root": attempt["result_root"],
            "state_dir": str(state_dir),
            "tmux_session": attempt["tmux_session"],
            "expected_duration_seconds": expected_duration_seconds,
            "hard_deadline_seconds": hard_deadline_seconds,
            "authorized": (
                run_class == "validation"
                or (
                    plan.get("launch_after_checks") is True
                    and production_preflight is not None
                    and production_preflight.get("status") == "passed"
                )
            ),
            "jobs": [
                {
                    "job_id": job["job_id"],
                    "worker_argv": list(job["worker_argv"]),
                    "output_dir": job["output_dir"],
                }
                for job in attempt["jobs"]
            ],
            "launch_batches": copy_batches(attempt["launch_batches"]),
            "preflight_summary": (
                None
                if production_preflight is None
                else {
                    "schema_version": production_preflight["schema_version"],
                    "status": production_preflight["status"],
                    "proposal_id": production_preflight["proposal_id"],
                }
            ),
        }
        requests.append(validate_dispatch_request(request))
    return requests


def copy_batches(value: Sequence[Sequence[str]]) -> list[list[str]]:
    return [list(batch) for batch in value]


def host_facts_from_smoke(
    receipt: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert one real numerical smoke into executor capability facts."""

    value = validate_functional_preflight(receipt)
    target = executor.validate_target_profile(profile)
    if value.get("target") != target["target_id"]:
        raise ValueError(
            f"Expected smoke target {target['target_id']!r}. "
            f"Provided value: {value.get('target')!r}."
        )
    environment = _strict_mapping(value.get("environment"), "smoke.environment")
    devices = environment.get("cuda_devices")
    devices = devices if isinstance(devices, list) else []
    rendered_devices = [
        {
            "index": int(item["index"]),
            "name": str(item["name"]),
            "free_memory_mb": int(item.get("free_memory_bytes", 0)) // (1024 * 1024),
        }
        for item in devices
        if isinstance(item, Mapping)
        and isinstance(item.get("index"), int)
        and isinstance(item.get("name"), str)
    ]
    smoke_value = _strict_mapping(value["smoke"], "smoke")
    benchmark = smoke_value.get("benchmark")
    benchmark = benchmark if isinstance(benchmark, Mapping) else {}
    peak = benchmark.get("cuda_peak_memory_reserved_bytes")
    free = max(
        (
            int(item.get("free_memory_bytes", 0))
            for item in devices
            if isinstance(item, Mapping)
        ),
        default=0,
    )
    estimated = target["default_concurrency"]
    if isinstance(peak, int) and peak > 0 and free > 0:
        estimated = max(
            1,
            min(
                target["default_concurrency"],
                int((0.8 * free) // peak),
            ),
        )
    scientific = _strict_mapping(
        value.get("scientific_binding"),
        "smoke.scientific_binding",
    )
    imports = {
        name: {
            "available": True,
            "version": (
                environment.get("pytorch_version")
                if name == "torch"
                else "imported"
            ),
        }
        for name in target["required_imports"]
    }
    return {
        "schema_version": executor.HOST_FACTS_SCHEMA,
        "target_id": target["target_id"],
        "reachable": True,
        "python": {
            "available": True,
            "executable": environment.get("python_executable", "python"),
            "version": environment.get("python_version", "unknown"),
        },
        "imports": imports,
        "cuda": {
            "available": environment.get("cuda_available") is True,
            "device_count": int(environment.get("cuda_device_count", 0)),
            "devices": rendered_devices,
            "runtime": environment.get("cuda_runtime"),
        },
        "dataset": {
            "readable": value.get("capabilities", {}).get("dataset_readable")
            is True,
            "identity": scientific.get("train_indices_sha256", "resolved-dataset"),
        },
        "output": {
            "writable": value.get("capabilities", {}).get("output_writable")
            is True,
            "path": target["result_root"],
        },
        "smoke": {
            "status": "passed",
            "completed_optimizer_steps": int(smoke_value["completed_steps"]),
            "finite": True,
            "artifact_written": True,
        },
        "tk_reference": {
            "status": "reused",
            "evidence": "unchanged continuation operating point",
        },
        "max_safe_concurrency": estimated,
        "observed_at": _now(),
    }


def build_production_preflight(
    production_plan: Mapping[str, Any],
    *,
    smoke_receipts: Mapping[str, Mapping[str, Any]],
    target_profiles: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build the one aggregate receipt from one smoke per target."""

    facts = {
        target_id: host_facts_from_smoke(receipt, target_profiles[target_id])
        for target_id, receipt in smoke_receipts.items()
    }
    result = executor.functional_preflight(
        production_plan,
        host_facts=facts,
        target_profiles=target_profiles,
    )
    return result, facts


def _paths(request: Mapping[str, Any]) -> dict[str, Path]:
    root = _absolute(request["state_dir"], "state_dir")
    return {
        "root": root,
        "worker_logs": root / "worker_logs",
        "request": root / "request.json",
        "dispatch": root / "dispatch.json",
        "progress": root / "progress.json",
        "attempt": root / "attempt_result.json",
        "failure": root / "failure.json",
        "controller_log": root / "controller.log",
    }


def _failure_record(
    request: Mapping[str, Any],
    error: FunctionalDispatchError,
    *,
    completed_job_ids: Sequence[str],
    commands: Sequence[Sequence[str]],
) -> dict[str, Any]:
    return {
        "schema_version": DISPATCH_FAILURE_SCHEMA,
        "status": "failed",
        "terminal": request["run_class"] == "production",
        "stage": error.stage,
        "error": str(error),
        "study_id": request["study_id"],
        "proposal_id": request["proposal_id"],
        "attempt_id": request["attempt_id"],
        "target_id": request["target_id"],
        "run_class": request["run_class"],
        "completed_job_ids": list(completed_job_ids),
        "completed_stages": (
            ["request_validated"]
            if not completed_job_ids
            else ["request_validated", "jobs_partially_completed"]
        ),
        "commands": [list(command) for command in commands],
        "launched_job_count": error.launched_job_count,
        "live_jobs": [dict(item) for item in error.live_jobs],
        "last_valid_artifacts": [
            str(_paths(request)["request"]),
            *[
                str(
                    _absolute(job["output_dir"], "job.output_dir")
                    / (
                        "functional_preflight.json"
                        if request["run_class"] == "validation"
                        else "job_completion.json"
                    )
                )
                for job in request["jobs"]
                if job["job_id"] in completed_job_ids
            ],
        ],
        "files_or_external_state_changed": [
            str(_paths(request)["root"]),
            *[
                str(job["output_dir"])
                for job in request["jobs"]
                if job["job_id"] in completed_job_ids
            ],
        ],
        "smallest_next_action": (
            "Report this immutable production-attempt failure and wait for a "
            "new user message before diagnosis, cancellation, or a new attempt."
            if request["run_class"] == "production"
            else "Diagnose this bounded validation failure within the pre-arm "
            "budget; if the budget is exhausted, ask the user how to proceed."
        ),
        "failed_at_utc": _now(),
    }


def _adapter_receipt(
    *,
    run_class: str,
    output_dir: Path,
    attempt_id: str,
    job_id: str,
) -> tuple[dict[str, Any], list[str]]:
    if run_class == "validation":
        path = output_dir / "functional_preflight.json"
        value = validate_functional_preflight(read_json(path))
        checkpoint = output_dir / value["smoke"]["checkpoint"]["path"]
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise ValueError(
                "Expected the validation smoke disposable checkpoint to exist. "
                f"Provided value: {checkpoint}."
            )
        return value, [str(path), str(checkpoint)]
    path = output_dir / "job_completion.json"
    value = validate_job_completion(
        read_json(path),
        expected_attempt_id=attempt_id,
        expected_job_id=job_id,
    )
    return value, [
        str(path),
        *[
            str(output_dir / item["path"])
            for item in value["artifacts"]
        ],
    ]


def _progress(
    request: Mapping[str, Any],
    *,
    completed: Sequence[str],
    live: Sequence[Mapping[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    value = {
        "schema_version": PROGRESS_SCHEMA,
        "status": "running",
        "attempt_id": request["attempt_id"],
        "target_id": request["target_id"],
        "run_class": request["run_class"],
        "started_at_utc": started_at,
        "observed_at_utc": _now(),
        "completed_job_ids": list(completed),
        "live_jobs": [dict(item) for item in live],
    }
    atomic_write_json(_paths(request)["progress"], value, canonical=True)
    return value


def execute_request(
    value: Mapping[str, Any],
    *,
    popen_factory: PopenFactory = subprocess.Popen,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run one target attempt.  This function never retries a process."""

    request = validate_dispatch_request(value)
    paths = _paths(request)
    if paths["attempt"].is_file():
        return read_json(paths["attempt"])
    if paths["failure"].is_file():
        return read_json(paths["failure"])
    workspace = _absolute(request["workspace_root"], "workspace_root")
    if not workspace.is_dir():
        raise FunctionalDispatchError(
            "workspace_check",
            f"Expected workspace_root to exist. Provided value: {workspace}.",
        )
    jobs_by_id = {job["job_id"]: job for job in request["jobs"]}
    started_at = _now()
    started_clock = monotonic()
    next_progress = started_clock
    launched = 0
    completed: list[str] = []
    job_results: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    try:
        paths["worker_logs"].mkdir(parents=True, exist_ok=True)
        for batch in request["launch_batches"]:
            running: dict[str, dict[str, Any]] = {}
            for job_id in batch:
                job = jobs_by_id[job_id]
                _absolute(job["output_dir"], "job.output_dir")
                log_path = paths["worker_logs"] / f"{job_id}.log"
                log_handle = log_path.open("xb", buffering=0)
                command = list(job["worker_argv"])
                commands.append(command)
                try:
                    process = popen_factory(
                        command,
                        cwd=workspace,
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except Exception:
                    log_handle.close()
                    raise
                launched += 1
                running[job_id] = {
                    "process": process,
                    "log_handle": log_handle,
                    "log_path": str(log_path),
                    "command": command,
                }

            while running:
                now = monotonic()
                if now - started_clock >= request["hard_deadline_seconds"]:
                    live = [
                        {
                            "job_id": job_id,
                            "pid": record["process"].pid,
                            "state": "running",
                            "log": record["log_path"],
                        }
                        for job_id, record in running.items()
                        if record["process"].poll() is None
                    ]
                    for record in running.values():
                        if record["process"].poll() is None:
                            record["process"].terminate()
                    raise FunctionalDispatchError(
                        "hard_deadline",
                        "The explicit hard deadline expired.",
                        launched_job_count=launched,
                        live_jobs=live,
                    )
                if now >= next_progress:
                    live = [
                        {
                            "job_id": job_id,
                            "pid": record["process"].pid,
                            "state": "running",
                            "log": record["log_path"],
                        }
                        for job_id, record in running.items()
                        if record["process"].poll() is None
                    ]
                    _progress(
                        request,
                        completed=completed,
                        live=live,
                        started_at=started_at,
                    )
                    next_progress = now + PROGRESS_INTERVAL_SECONDS

                finished = []
                for job_id, record in running.items():
                    return_code = record["process"].poll()
                    if return_code is None:
                        continue
                    record["log_handle"].close()
                    if return_code != 0:
                        live = [
                            {
                                "job_id": other_id,
                                "pid": other["process"].pid,
                                "state": "running",
                                "log": other["log_path"],
                            }
                            for other_id, other in running.items()
                            if other_id != job_id
                            and other["process"].poll() is None
                        ]
                        raise FunctionalDispatchError(
                            "payload_execution",
                            f"Command {record['command']!r} exited with "
                            f"status {return_code}; log={record['log_path']}.",
                            launched_job_count=launched,
                            live_jobs=live,
                        )
                    output = _absolute(
                        jobs_by_id[job_id]["output_dir"],
                        "job.output_dir",
                    )
                    try:
                        receipt, artifacts = _adapter_receipt(
                            run_class=request["run_class"],
                            output_dir=output,
                            attempt_id=request["attempt_id"],
                            job_id=job_id,
                        )
                    except (OSError, TypeError, ValueError) as exc:
                        raise FunctionalDispatchError(
                            "adapter_receipt_validation",
                            f"Expected valid adapter output for {job_id!r}. "
                            f"Provided value: {exc!r}.",
                            launched_job_count=launched,
                        ) from exc
                    job_results.append(
                        executor.make_job_result(
                            plan={
                                "schema_version": executor.EXECUTION_PLAN_SCHEMA,
                                "study_id": request["study_id"],
                                "proposal_id": request["proposal_id"],
                                "run_class": request["run_class"],
                            },
                            attempt_id=request["attempt_id"],
                            target_id=request["target_id"],
                            job_id=job_id,
                            status="complete",
                            exit_code=0,
                            artifacts=artifacts,
                            provenance=receipt.get(
                                "environment",
                                receipt.get("adapter_provenance", {}),
                            ),
                        )
                    )
                    completed.append(job_id)
                    finished.append(job_id)
                for job_id in finished:
                    running.pop(job_id)
                if running:
                    sleep(1.0)

        ordered = {
            result["job_id"]: result for result in job_results
        }
        attempt_result = {
            "schema_version": executor.ATTEMPT_RESULT_SCHEMA,
            "status": "complete",
            "study_id": request["study_id"],
            "proposal_id": request["proposal_id"],
            "attempt_id": request["attempt_id"],
            "target_id": request["target_id"],
            "run_class": request["run_class"],
            "started_at_utc": started_at,
            "completed_at_utc": _now(),
            "job_results": [
                ordered[job["job_id"]] for job in request["jobs"]
            ],
        }
        atomic_create_json(paths["attempt"], attempt_result, canonical=True)
        return attempt_result
    except FunctionalDispatchError as error:
        failure = _failure_record(
            request,
            error,
            completed_job_ids=completed,
            commands=commands,
        )
        if not paths["failure"].exists():
            atomic_create_json(paths["failure"], failure, canonical=True)
        return failure
    except Exception as exc:
        error = FunctionalDispatchError(
            "controller_exception",
            repr(exc),
            launched_job_count=launched,
        )
        failure = _failure_record(
            request,
            error,
            completed_job_ids=completed,
            commands=commands,
        )
        if not paths["failure"].exists():
            atomic_create_json(paths["failure"], failure, canonical=True)
        return failure


def start_request(
    value: Mapping[str, Any],
    *,
    runner: Runner = subprocess.run,
    popen_factory: PopenFactory = subprocess.Popen,
) -> dict[str, Any]:
    """Persist and detach one attempt, with immediate tmux readback."""

    request = validate_dispatch_request(value)
    paths = _paths(request)
    paths["root"].mkdir(parents=True, exist_ok=True)
    if paths["request"].exists():
        if read_json(paths["request"]) != request:
            raise FileExistsError(
                "Expected an existing attempt request to match exactly. "
                f"Provided value: {paths['request']}."
            )
    else:
        atomic_create_json(paths["request"], request, canonical=True)
    for terminal in ("attempt", "failure", "dispatch"):
        if paths[terminal].is_file():
            return read_json(paths[terminal])

    if request["run_class"] == "production":
        print(
            "LONG-RUN ATTEMPT ARMED "
            f"attempt_id={request['attempt_id']} "
            f"target={request['target_id']} "
            f"expected_duration_seconds={request['expected_duration_seconds']} "
            f"hard_deadline_seconds={request['hard_deadline_seconds']} "
            f"state_path={paths['root']} "
            f"receipt_path={paths['attempt']}",
            flush=True,
        )
    worker_command = [
        sys.executable,
        "-m",
        "experiments.functional_dispatch",
        "execute",
        "--request",
        str(paths["request"]),
    ]
    if request["launch_backend"] == "tmux":
        command = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            request["tmux_session"],
            "-c",
            request["workspace_root"],
            *worker_command,
        ]
        launched = runner(
            command,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        if launched.returncode != 0:
            raise FunctionalDispatchError(
                "tmux_launch",
                f"Expected tmux launch to succeed. Provided value: "
                f"returncode={launched.returncode}, "
                f"stderr={launched.stderr!r}.",
                launched_job_count=0,
            )
        readback = runner(
            [
                "tmux",
                "list-panes",
                "-t",
                request["tmux_session"],
                "-F",
                "#{pane_pid} #{pane_current_command}",
            ],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        if readback.returncode != 0 or not readback.stdout.strip():
            raise FunctionalDispatchError(
                "immediate_readback",
                "Expected a live tmux pane immediately after launch. "
                f"Provided value: returncode={readback.returncode}, "
                f"stdout={readback.stdout!r}, stderr={readback.stderr!r}.",
                launched_job_count=None,
            )
        immediate_readback = readback.stdout.strip()
    else:
        command = worker_command
        log_handle = paths["controller_log"].open("ab", buffering=0)
        try:
            process = popen_factory(
                command,
                cwd=request["workspace_root"],
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_handle.close()
        return_code = process.poll()
        if return_code is not None:
            raise FunctionalDispatchError(
                "detached_process_launch",
                "Expected the detached supervisor to remain live after "
                f"launch. Provided value: pid={process.pid}, "
                f"returncode={return_code}, "
                f"log={paths['controller_log']}.",
                launched_job_count=0,
            )
        immediate_readback = f"pid={process.pid} state=running"
    receipt = {
        "schema_version": DISPATCH_RECEIPT_SCHEMA,
        "status": "dispatched",
        "study_id": request["study_id"],
        "proposal_id": request["proposal_id"],
        "attempt_id": request["attempt_id"],
        "target_id": request["target_id"],
        "run_class": request["run_class"],
        "launch_backend": request["launch_backend"],
        "tmux_session": (
            request["tmux_session"]
            if request["launch_backend"] == "tmux"
            else None
        ),
        "launch_command": command,
        "immediate_readback": immediate_readback,
        "launched_job_count": 0,
        "dispatched_at_utc": _now(),
    }
    atomic_create_json(paths["dispatch"], receipt, canonical=True)
    return receipt


def status_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = validate_dispatch_request(value)
    paths = _paths(request)
    # Terminal receipts win; while an attempt is active, its latest progress
    # is authoritative over the initial dispatch acknowledgement.
    for role in ("failure", "attempt", "progress", "dispatch", "request"):
        if paths[role].is_file():
            return read_json(paths[role])
    return {
        "schema_version": "functional-dispatch-status/v1",
        "status": "not_found",
        "attempt_id": request["attempt_id"],
        "target_id": request["target_id"],
    }


def runner_preflight() -> dict[str, Any]:
    """Side-effect-free scheduled-runner contract check."""

    repo_root = Path(__file__).resolve().parents[1]
    profile_root = repo_root / "configs" / "dispatch"
    profiles = {}
    for target_id in ("local", "trex", "akib"):
        path = profile_root / f"functional_{target_id}.json"
        profiles[target_id] = executor.load_target_profile(path)[
            "schema_version"
        ]
    return {
        "schema_version": "functional-dispatch-runner-preflight/v1",
        "status": "passed",
        "runner": str(Path(__file__).resolve()),
        "profiles": profiles,
        "validation_max_seconds": VALIDATION_MAX_SECONDS,
        "progress_interval_seconds": PROGRESS_INTERVAL_SECONDS,
        "side_effects_performed": False,
    }


def _ssh_transport_command(
    request: Mapping[str, Any],
    profile: Mapping[str, Any],
    remote_argv: Sequence[str],
) -> list[str]:
    remote_script = (
        f"cd {shlex.quote(str(request['workspace_root']))} && "
        f"exec {shlex.join(remote_argv)}"
    )
    command = ["ssh"]
    ssh_config = profile["transport"].get("ssh_config")
    if ssh_config is not None:
        command.extend(["-F", ssh_config])
    command.extend(
        [
            profile["transport"]["ssh_target"],
            shlex.join(["/bin/bash", "-lc", remote_script]),
        ]
    )
    return command


def submit_request(
    value: Mapping[str, Any],
    profile_value: Mapping[str, Any],
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Start through the declared local or one-SSH transport without retry."""

    request = validate_dispatch_request(value)
    profile = executor.validate_target_profile(profile_value)
    if request["target_id"] != profile["target_id"]:
        raise ValueError(
            f"Expected request target_id {profile['target_id']!r}. "
            f"Provided value: {request['target_id']!r}."
        )
    if profile["transport"]["kind"] == "local":
        return start_request(request, runner=runner)

    remote_argv = [
        profile["python_candidates"][0],
        "-m",
        "experiments.functional_dispatch",
        "remote-start",
    ]
    command = _ssh_transport_command(request, profile, remote_argv)
    if request["run_class"] == "production":
        paths = _paths(request)
        print(
            "LONG-RUN ATTEMPT ARMED "
            f"attempt_id={request['attempt_id']} "
            f"target={request['target_id']} "
            f"expected_duration_seconds={request['expected_duration_seconds']} "
            f"hard_deadline_seconds={request['hard_deadline_seconds']} "
            f"state_path={paths['root']} "
            f"receipt_path={paths['attempt']}",
            flush=True,
        )
    submitted = runner(
        command,
        input=json.dumps(
            request,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if submitted.returncode != 0:
        raise FunctionalDispatchError(
            "remote_start",
            "Expected the one-SSH remote start to succeed. "
            f"Provided value: returncode={submitted.returncode}, "
            f"stderr={submitted.stderr!r}.",
            launched_job_count=None,
        )
    output = submitted.stdout
    json_start = output.find("{")
    if json_start < 0:
        raise FunctionalDispatchError(
            "remote_readback",
            "Expected remote start to return a JSON dispatch receipt. "
            f"Provided value: {output!r}.",
            launched_job_count=None,
        )
    try:
        result = json.loads(output[json_start:])
    except json.JSONDecodeError as exc:
        raise FunctionalDispatchError(
            "remote_readback",
            "Expected remote start to return one valid JSON dispatch receipt. "
            f"Provided value: {output!r}.",
            launched_job_count=None,
        ) from exc
    if not isinstance(result, Mapping):
        raise FunctionalDispatchError(
            "remote_readback",
            "Expected the remote dispatch receipt to be a mapping. "
            f"Provided value: {result!r}.",
            launched_job_count=None,
        )
    return dict(result)


def query_request(
    value: Mapping[str, Any],
    profile_value: Mapping[str, Any],
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Read one target attempt through its declared transport."""

    request = validate_dispatch_request(value)
    profile = executor.validate_target_profile(profile_value)
    if request["target_id"] != profile["target_id"]:
        raise ValueError(
            f"Expected request target_id {profile['target_id']!r}. "
            f"Provided value: {request['target_id']!r}."
        )
    if profile["transport"]["kind"] == "local":
        return status_request(request)
    remote_argv = [
        profile["python_candidates"][0],
        "-m",
        "experiments.functional_dispatch",
        "remote-status",
    ]
    queried = runner(
        _ssh_transport_command(request, profile, remote_argv),
        input=json.dumps(
            request,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if queried.returncode not in {0, 2}:
        raise FunctionalDispatchError(
            "remote_status",
            "Expected the one-SSH remote status query to succeed. "
            f"Provided value: returncode={queried.returncode}, "
            f"stderr={queried.stderr!r}.",
            launched_job_count=None,
        )
    try:
        result = json.loads(queried.stdout)
    except json.JSONDecodeError as exc:
        raise FunctionalDispatchError(
            "remote_status_readback",
            "Expected remote status to return one valid JSON document. "
            f"Provided value: {queried.stdout!r}.",
            launched_job_count=None,
        ) from exc
    if not isinstance(result, Mapping):
        raise FunctionalDispatchError(
            "remote_status_readback",
            "Expected remote status to return a mapping. "
            f"Provided value: {result!r}.",
            launched_job_count=None,
        )
    return dict(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one immutable functional executor attempt."
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate the static runner/profile contract without side effects",
    )
    commands = parser.add_subparsers(dest="command")
    for name in ("start", "execute"):
        child = commands.add_parser(name)
        child.add_argument("--request", required=True)
    status = commands.add_parser("status")
    status.add_argument("--request", required=True)
    status.add_argument(
        "--profile",
        help=(
            "optional target profile; when supplied, query the target through "
            "its declared local or SSH transport"
        ),
    )
    submit = commands.add_parser("submit")
    submit.add_argument("--request", required=True)
    submit.add_argument("--profile", required=True)
    query = commands.add_parser("query")
    query.add_argument("--request", required=True)
    query.add_argument("--profile", required=True)
    for name in ("remote-start", "remote-status"):
        remote = commands.add_parser(name)
        remote.add_argument(
            "--max-request-bytes", type=int, default=1024 * 1024
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.preflight:
        if args.command is not None:
            parser.error("--preflight cannot be combined with a command")
        print(json.dumps(runner_preflight(), indent=2, sort_keys=True))
        return 0
    if args.command is None:
        parser.error("a command is required unless --preflight is used")
    if args.command in {"remote-start", "remote-status"}:
        payload = sys.stdin.buffer.read(args.max_request_bytes + 1)
        if len(payload) > args.max_request_bytes:
            raise ValueError(
                f"Expected remote request <= {args.max_request_bytes} bytes. "
                f"Provided value: {len(payload)}."
            )
        request = json.loads(payload)
        result = (
            start_request(request)
            if args.command == "remote-start"
            else status_request(request)
        )
    else:
        request = read_json(args.request)
        if args.command == "submit":
            result = submit_request(request, read_json(args.profile))
        elif args.command == "query":
            result = query_request(request, read_json(args.profile))
        elif args.command == "status" and args.profile is not None:
            result = query_request(request, read_json(args.profile))
        elif args.command == "start":
            result = start_request(request)
        elif args.command == "execute":
            result = execute_request(request)
        else:
            result = status_request(request)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DISPATCH_FAILURE_SCHEMA",
    "DISPATCH_RECEIPT_SCHEMA",
    "DISPATCH_REQUEST_SCHEMA",
    "FunctionalDispatchError",
    "build_dispatch_requests",
    "build_production_preflight",
    "execute_request",
    "host_facts_from_smoke",
    "query_request",
    "runner_preflight",
    "start_request",
    "status_request",
    "submit_request",
    "validate_dispatch_request",
]
