"""Measured multi-process throughput benchmark for the v6 Conv2 sweep."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from .identity import code_provenance
from .io import atomic_write_json, read_json
from .lr_artifacts import load_stage_manifest
from .lr_stages import execute_v6_shadow_benchmark_entry
from .lr_study import _require_current_source, load_study
from .lr_v6_packing import (
    CONCURRENCY_LEVELS,
    MEASURED_STEPS,
    WARMUP_STEPS,
    build_benchmark_report,
    find_safe_baseline_entry,
    validate_benchmark_report,
)


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def _contained(path: Path, root: Path, label: str) -> Path:
    supplied = path.expanduser().absolute()
    if supplied.is_symlink():
        raise _error(f"{label} not to be a symlink", supplied)
    resolved = supplied.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise _error(f"{label} to be contained by the v6 study", resolved) from exc
    return resolved


def _safe_payload(
    stage_manifest: Mapping[str, Any], entry_index: int
) -> Mapping[str, Any]:
    expected = find_safe_baseline_entry(stage_manifest)
    if entry_index != expected:
        raise _error(
            "benchmark entry_index to identify rho_conv=rho_dense=0.003",
            entry_index,
        )
    payload = stage_manifest["entries"][entry_index]["payload"]
    required = {"row_id", "architecture", "scheme", "learning_rates_by_parameter"}
    required.update(
        {
            "candidate_role",
            "candidate_stage",
            "median_units_by_weight",
            "peak_learning_rate",
        }
    )
    if not isinstance(payload, Mapping) or not required.issubset(payload):
        raise _error(
            f"safe baseline payload to contain {sorted(required)!r}", payload
        )
    if (
        payload["architecture"] != "conv2"
        or payload["scheme"] != "baseline"
        or not isinstance(payload["learning_rates_by_parameter"], Mapping)
    ):
        raise _error("safe benchmark payload to be the Conv2 baseline", payload)
    return payload


def run_shadow_child(
    *,
    study_dir: str | Path,
    candidate_manifest_path: str | Path,
    entry_index: int,
    data_root: str | Path,
    ready_path: str | Path,
    start_path: str | Path,
    output_path: str | Path,
    child_index: int,
    device: str = "cuda",
    warmup_steps: int = WARMUP_STEPS,
    measured_steps: int = MEASURED_STEPS,
    start_timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    """Run one noncanonical child and publish only timing evidence."""

    root, spec = load_study(study_dir)
    if spec.data.get("schema_version") != "mnist-conv-lr-study/v6":
        raise _error("study schema_version to be mnist-conv-lr-study/v6", spec.data)
    if spec.data.get("dataset", {}).get("official_test") != {
        "enabled": False,
        "read_allowed": False,
    }:
        raise _error(
            "v6 official MNIST test reads to be disabled and prohibited",
            spec.data.get("dataset", {}).get("official_test"),
        )
    if warmup_steps != WARMUP_STEPS or measured_steps != MEASURED_STEPS:
        raise _error(
            f"shadow steps to be {WARMUP_STEPS} warm-up plus {MEASURED_STEPS} measured",
            {"warmup": warmup_steps, "measured": measured_steps},
        )
    if type(child_index) is not int or child_index < 0:
        raise _error("child_index to be a non-negative integer", child_index)
    manifest_path = _contained(
        Path(candidate_manifest_path), root, "baseline-candidate manifest"
    )
    output = Path(output_path).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    ready = Path(ready_path).expanduser().absolute()
    ready.parent.mkdir(parents=True, exist_ok=True)
    start = Path(start_path).expanduser().absolute()
    manifest = load_stage_manifest(
        manifest_path,
        study_dir=root,
        expected_study_id=spec.study_id,
        expected_stage_name="baseline_candidates",
    )
    _require_current_source(manifest["code_provenance"], code_provenance())
    payload = _safe_payload(manifest, entry_index)
    row_id = payload["row_id"]
    report = execute_v6_shadow_benchmark_entry(
        spec.data,
        root,
        str(row_id),
        str(payload["candidate_role"]),
        candidate_payload=payload,
        data_root=data_root,
        download=False,
        device=device,
        ready_path=ready,
        start_path=start,
        scratch_output_dir=output.parent / "production_shadow_artifacts",
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        start_timeout_seconds=start_timeout_seconds,
    )
    result = {**report, "child_index": child_index, "pid": os.getpid()}
    atomic_write_json(output, result, canonical=True)
    return result


def _host_rss_mib(pids: list[int]) -> float:
    total_kib = 0
    for pid in pids:
        try:
            lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) >= 2:
                    total_kib += int(fields[1])
                break
    return total_kib / 1024.0


def _gpu_sample(visible_device: str) -> tuple[float, float]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={visible_device}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    records = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(records) != 1:
        raise RuntimeError(
            "Expected nvidia-smi to return exactly one GPU sample. "
            f"Provided value: {records!r}."
        )
    fields = [field.strip() for field in records[0].split(",")]
    if len(fields) != 2:
        raise RuntimeError(
            "Expected nvidia-smi memory/utilization fields. "
            f"Provided value: {records[0]!r}."
        )
    return float(fields[0]), float(fields[1])


def _device_metadata(device: str) -> dict[str, Any]:
    import torch

    requested = torch.device(device)
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise _error("an available CUDA device", device)
    properties = torch.cuda.get_device_properties(requested)
    return {
        "name": properties.name,
        "total_memory_mib": float(properties.total_memory) / 1_048_576.0,
    }


def _run_level(
    *,
    concurrency: int,
    study_dir: Path,
    candidate_manifest_path: Path,
    safe_entry_index: int,
    data_root: Path,
    scratch_root: Path,
    device: str,
    visible_device: str,
    poll_seconds: float,
    ready_timeout_seconds: float,
) -> dict[str, Any]:
    level_root = scratch_root / f"concurrency_{concurrency:02d}"
    if level_root.exists():
        shutil.rmtree(level_root)
    level_root.mkdir(parents=True)
    start_path = level_root / "start"
    processes: list[subprocess.Popen[str]] = []
    streams: list[tuple[Any, Any]] = []
    launched = time.time()
    for child_index in range(concurrency):
        child_root = level_root / f"child_{child_index:02d}"
        child_root.mkdir()
        stdout_stream = (child_root / "stdout.log").open("w")
        stderr_stream = (child_root / "stderr.log").open("w")
        command = [
            sys.executable,
            "-m",
            "experiments.mnist_conv",
            "lr-study",
            "--stage",
            "baseline_candidates",
            "--study",
            str(study_dir),
            "--manifest",
            str(candidate_manifest_path),
            "--entry-index",
            str(safe_entry_index),
            "--data-root",
            str(data_root),
            "--device",
            device,
            "--shadow-benchmark",
            "--shadow-ready",
            str(child_root / "ready.json"),
            "--shadow-start",
            str(start_path),
            "--shadow-scratch-output",
            str(child_root / "production_shadow_artifacts"),
            "--shadow-start-timeout-seconds",
            str(ready_timeout_seconds),
        ]
        processes.append(
            subprocess.Popen(
                command,
                stdout=stdout_stream,
                stderr=stderr_stream,
                text=True,
            )
        )
        streams.append((stdout_stream, stderr_stream))

    gpu_memory_samples: list[float] = []
    utilization_samples: list[float] = []
    rss_samples: list[float] = []

    def sample() -> None:
        memory, utilization = _gpu_sample(visible_device)
        gpu_memory_samples.append(memory)
        utilization_samples.append(utilization)
        rss_samples.append(_host_rss_mib([process.pid for process in processes]))

    readiness_deadline = time.monotonic() + ready_timeout_seconds
    readiness_failed = False
    try:
        while True:
            sample()
            ready_count = sum(
                (level_root / f"child_{index:02d}" / "ready.json").is_file()
                for index in range(concurrency)
            )
            if ready_count == concurrency:
                break
            if any(process.poll() is not None for process in processes):
                readiness_failed = True
                break
            if time.monotonic() >= readiness_deadline:
                readiness_failed = True
                break
            time.sleep(poll_seconds)
        if not readiness_failed:
            start_path.touch(exist_ok=False)
            while any(process.poll() is None for process in processes):
                sample()
                time.sleep(poll_seconds)
        else:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        sample()
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for stdout_stream, stderr_stream in streams:
            stdout_stream.close()
            stderr_stream.close()

    finished = time.time()
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for child_index, process in enumerate(processes):
        child_root = level_root / f"child_{child_index:02d}"
        result_path = child_root / "result.json"
        stdout_path = child_root / "stdout.log"
        report = None
        if process.returncode == 0 and stdout_path.is_file():
            for line in reversed(stdout_path.read_text().splitlines()):
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    report = candidate
                    break
        if report is not None:
            if (
                report.get("status") == "complete"
                and report.get("warmup_steps") == WARMUP_STEPS
                and report.get("measured_steps") == MEASURED_STEPS
                and report.get("official_test_read") is False
                and report.get("production_loop") is True
                and report.get("attempted_steps")
                == WARMUP_STEPS + MEASURED_STEPS
                and report.get("completed_steps")
                == WARMUP_STEPS + MEASURED_STEPS
                and isinstance(report.get("transition_row_count"), int)
                and report["transition_row_count"] > 0
                and report.get("canonical_completion_published") is False
            ):
                report = {
                    **report,
                    "child_index": child_index,
                    "pid": processes[child_index].pid,
                }
                atomic_write_json(result_path, report, canonical=True)
                reports.append(report)
                continue
        stderr_path = child_root / "stderr.log"
        stderr_tail = ""
        if stderr_path.is_file():
            stderr_tail = "\n".join(stderr_path.read_text().splitlines()[-20:])
        failures.append(
            {
                "child_index": child_index,
                "returncode": process.returncode,
                "stderr_tail": stderr_tail,
                "readiness_failed": readiness_failed,
            }
        )
    if reports:
        measured_span = max(
            float(report["measured_finished_unix_s"]) for report in reports
        ) - min(float(report["measured_started_unix_s"]) for report in reports)
        throughput = (
            len(reports) * MEASURED_STEPS / measured_span
            if measured_span > 0.0
            else 0.0
        )
    else:
        throughput = 0.0
    return {
        "concurrency": concurrency,
        "completed_children": len(reports),
        "failed_children": len(failures),
        "combined_peak_gpu_memory_mib": max(gpu_memory_samples, default=0.0),
        "peak_host_rss_mib": max(rss_samples, default=0.0),
        "mean_gpu_utilization_percent": (
            sum(utilization_samples) / len(utilization_samples)
            if utilization_samples
            else 0.0
        ),
        "aggregate_successful_steps_per_second": throughput,
        "successful_measured_steps": len(reports) * MEASURED_STEPS,
        "wall_seconds": max(finished - launched, 1e-12),
        "child_failures": failures,
    }


def run_concurrency_benchmark(
    *,
    study_dir: str | Path,
    candidate_manifest_path: str | Path,
    data_root: str | Path,
    output_path: str | Path,
    scratch_root: str | Path,
    device: str = "cuda",
    poll_seconds: float = 0.10,
    ready_timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    root, spec = load_study(study_dir)
    if spec.data.get("schema_version") != "mnist-conv-lr-study/v6":
        raise _error("study schema_version to be mnist-conv-lr-study/v6", spec.data)
    manifest_path = _contained(
        Path(candidate_manifest_path), root, "baseline-candidate manifest"
    )
    output = _contained(Path(output_path), root, "benchmark output")
    stage_bytes = manifest_path.read_bytes()
    stage_sha256 = hashlib.sha256(stage_bytes).hexdigest()
    stage_manifest = load_stage_manifest(
        manifest_path,
        study_dir=root,
        expected_study_id=spec.study_id,
        expected_stage_name="baseline_candidates",
    )
    if output.exists():
        existing = validate_benchmark_report(
            read_json(output), expected_study_id=spec.study_id
        )
        if existing["stage_manifest_sha256"] != stage_sha256:
            raise _error(
                "existing benchmark to bind the current candidate manifest",
                existing["stage_manifest_sha256"],
            )
        return existing
    scratch = Path(scratch_root).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    visible = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    if len(visible) != 1:
        raise _error("CUDA_VISIBLE_DEVICES to contain exactly one GPU", visible)
    if poll_seconds <= 0.0 or ready_timeout_seconds <= 0.0:
        raise _error(
            "positive benchmark polling and readiness timeouts",
            {"poll_seconds": poll_seconds, "ready_timeout": ready_timeout_seconds},
        )
    safe_entry_index = find_safe_baseline_entry(stage_manifest)
    levels = [
        _run_level(
            concurrency=concurrency,
            study_dir=root,
            candidate_manifest_path=manifest_path,
            safe_entry_index=safe_entry_index,
            data_root=Path(data_root).expanduser().resolve(),
            scratch_root=scratch,
            device=device,
            visible_device=visible[0],
            poll_seconds=poll_seconds,
            ready_timeout_seconds=ready_timeout_seconds,
        )
        for concurrency in CONCURRENCY_LEVELS
    ]
    report = build_benchmark_report(
        stage_manifest,
        stage_manifest_sha256=stage_sha256,
        levels=levels,
        device=_device_metadata(device),
    )
    atomic_write_json(output, report, canonical=True)
    return report


__all__ = ["run_concurrency_benchmark", "run_shadow_child"]
