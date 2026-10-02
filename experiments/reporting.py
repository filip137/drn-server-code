#!/usr/bin/env python3
"""Write and validate lightweight experiment reporting bundles."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any, Iterable, Mapping, Sequence


MANIFEST_SCHEMA = "experiment-run-manifest/v1"
STATUS_SCHEMA = "experiment-run-status/v1"
METRIC_SCHEMA = "experiment-run-metric/v1"
RESULT_SCHEMA = "experiment-run-result/v1"

ACTIVE_BEGIN = "<!-- BEGIN GENERATED ACTIVE RUNS -->"
ACTIVE_END = "<!-- END GENERATED ACTIVE RUNS -->"

RETIRED_DOCUMENTS = {
    Path(__file__).resolve().parents[1] / "docs" / name
    for name in ("current_simulations.md", "experimental_manifest.md")
}

CANONICAL_FILES = {
    "manifest.json",
    "status.json",
    "metrics.jsonl",
    "result.json",
}
CHECKPOINT_NAMES = {
    "best_model.pt",
    "final_model.pt",
    "weights_best.npz",
    "weights_final.npz",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON file on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_metric(path: Path, value: Mapping[str, Any]) -> None:
    """Append one finite JSON measurement and flush it to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    payload.setdefault("schema_version", METRIC_SCHEMA)
    payload.setdefault("recorded_at", utc_now())
    line = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    # json.loads accepts NaN/Infinity and can overflow a JSON number to inf.
    _json_bytes(value)
    return value


def iter_metrics(path: Path) -> Iterable[dict[str, Any]]:
    """Read finite metric objects, reporting the location of malformed records."""
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield _json_object(line)
            except ValueError as error:
                raise ValueError(f"metrics.jsonl line {line_number}: {error}") from error


def _manifest_comparable(value: Mapping[str, Any]) -> dict[str, Any]:
    comparable = dict(value)
    comparable.pop("created_at", None)
    return comparable


def start_run(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Create an immutable manifest and transition a run to running."""
    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "metrics.jsonl").touch(exist_ok=True)

    manifest_payload = dict(manifest)
    manifest_payload.setdefault("schema_version", MANIFEST_SCHEMA)
    manifest_payload.setdefault("created_at", utc_now())
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_comparable(existing) != _manifest_comparable(manifest_payload):
            raise RuntimeError(
                "Existing manifest.json does not match the requested run. "
                f"Path: {manifest_path}."
            )
        manifest_payload = existing
    else:
        atomic_write_json(manifest_path, manifest_payload)

    result_path = run_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(
            "A successful result.json already exists for this run. "
            f"Use a new run directory instead of overwriting {result_path}."
        )
    status_path = run_dir / "status.json"
    if status_path.exists():
        existing_status = json.loads(status_path.read_text(encoding="utf-8"))
        raise FileExistsError(
            "A reporting status already exists for this run "
            f"(state={existing_status.get('state')!r}). Use a new run directory "
            f"instead of restarting {run_dir}."
        )

    now = utc_now()
    status = {
        "schema_version": STATUS_SCHEMA,
        "study_id": manifest_payload["study_id"],
        "run_id": manifest_payload["run_id"],
        "arm_id": manifest_payload["arm_id"],
        "state": "running",
        "started_at": now,
        "updated_at": now,
        "heartbeat_at": now,
        "progress": {"stage": "starting"},
        "runtime": dict(manifest_payload.get("runtime", {})),
    }
    atomic_write_json(status_path, status)
    return status


def update_status_progress(run_dir: Path, progress: Mapping[str, Any]) -> None:
    status_path = Path(run_dir).expanduser().resolve() / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "running":
        raise RuntimeError(
            "Progress can only be recorded for a running experiment. "
            f"Provided state: {status.get('state')!r}."
        )
    now = utc_now()
    status["updated_at"] = now
    status["heartbeat_at"] = now
    status["progress"] = dict(progress)
    atomic_write_json(status_path, status)


def is_paused(status: Mapping[str, Any]) -> bool:
    """Recognize canonical pauses and both historical runner encodings."""
    progress = status.get("progress")
    error = status.get("error")
    return (
        status.get("state") == "paused"
        or (status.get("state") == "running"
            and isinstance(progress, Mapping) and progress.get("stage") == "checkpointed_pause")
        or (status.get("state") == "failed"
            and isinstance(error, Mapping) and error.get("type") == "PlannedPause"
            and error.get("returncode") == 75)
    )


def pause_run(
    run_dir: Path, *, checkpoint: Path, progress: Mapping[str, Any], reason: str,
) -> dict[str, Any]:
    """Record a planned, checkpointed pause without claiming success or failure."""
    run_dir = Path(run_dir).resolve()
    status_path = run_dir / "status.json"
    status = _json_object(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "running" or (run_dir / "result.json").exists():
        raise RuntimeError("Only a running experiment without a result can pause")
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise ValueError("A planned pause requires a non-empty checkpoint")
    pause = {"checkpoint": checkpoint.relative_to(run_dir).as_posix(),
             "checkpoint_sha256": sha256_file(checkpoint), "reason": reason}
    now = utc_now()
    status.update(state="paused", updated_at=now, heartbeat_at=now, paused_at=now,
                  progress={**progress, "stage": "checkpointed_pause"}, pause=pause)
    atomic_write_json(status_path, status)
    return status


def resume_run(run_dir: Path, *, runtime: Mapping[str, Any]) -> None:
    """Resume reporting after the runner has checked its continuation contract."""
    run_dir = Path(run_dir).resolve()
    status_path = run_dir / "status.json"
    status = _json_object(status_path.read_text(encoding="utf-8"))
    if not is_paused(status) or (run_dir / "result.json").exists():
        raise RuntimeError("Only a checkpointed pause can resume")
    if status.get("state") == "paused":
        pause = status["pause"]
        if sha256_file(run_dir / pause["checkpoint"]) != pause["checkpoint_sha256"]:
            raise ValueError("Paused checkpoint hash mismatch")
    now = utc_now()
    status.update(state="running", updated_at=now, heartbeat_at=now, resumed_at=now,
                  runtime=dict(runtime), progress={**status.get("progress", {}), "stage": "resuming"})
    status.pop("error", None)
    atomic_write_json(status_path, status)


def _relative_symlink(source: Path, destination: Path) -> None:
    target = Path(os.path.relpath(source, start=destination.parent))
    if destination.is_symlink():
        if Path(os.readlink(destination)) == target:
            return
        raise RuntimeError(f"Unexpected existing artifact link: {destination}.")
    if destination.exists():
        raise RuntimeError(f"Unexpected existing artifact path: {destination}.")
    destination.symlink_to(target)


def materialize_artifact_views(run_dir: Path) -> None:
    """Expose legacy flat trainer outputs through canonical artifact folders."""
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"
    artifact_dir = run_dir / "artifacts"
    checkpoint_dir.mkdir(exist_ok=True)
    artifact_dir.mkdir(exist_ok=True)

    for source in sorted(run_dir.iterdir()):
        if (
            source.name in CANONICAL_FILES
            or source.name in {"checkpoints", "artifacts"}
            or not source.is_file()
        ):
            continue
        destination_dir = (
            checkpoint_dir if source.name in CHECKPOINT_NAMES else artifact_dir
        )
        _relative_symlink(source, destination_dir / source.name)


def _artifact_records(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for directory, kind in (
        (run_dir / "checkpoints", "checkpoint"),
        (run_dir / "artifacts", "artifact"),
    ):
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            records.append(
                {
                    "kind": kind,
                    "path": path.relative_to(run_dir).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return records


def complete_run(
    run_dir: Path,
    *,
    terminal_metrics: Mapping[str, Any],
    completion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write result.json, then make the complete transition visible."""
    run_dir = Path(run_dir).expanduser().resolve()
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "running":
        raise RuntimeError(
            "Only a running experiment can complete. "
            f"Provided state: {status.get('state')!r}."
        )

    completion = dict({"criteria_met": True} if completion is None else completion)
    if not isinstance(completion.get("criteria_met"), bool):
        raise ValueError("Completion requires a boolean criteria_met")
    # Diagnostics may legitimately have no epoch records, but malformed or
    # nonfinite measurements must never be sealed into a successful bundle.
    for _ in iter_metrics(run_dir / "metrics.jsonl"):
        pass
    materialize_artifact_views(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics_path = run_dir / "metrics.jsonl"
    completed_at = utc_now()
    result = {
        "schema_version": RESULT_SCHEMA,
        "study_id": status["study_id"],
        "run_id": status["run_id"],
        "arm_id": status["arm_id"],
        "evidence_class": manifest.get("evidence_class"),
        "dataset": manifest.get("dataset"),
        "smoke": bool(manifest.get("smoke", False)),
        "completed_at": completed_at,
        "terminal_metrics": dict(terminal_metrics),
        "completion": completion,
        "manifest_sha256": sha256_file(manifest_path),
        "metrics_sha256": sha256_file(metrics_path),
        "artifacts": _artifact_records(run_dir),
    }
    result_path = run_dir / "result.json"
    atomic_write_json(result_path, result)

    status["state"] = "complete"
    status["updated_at"] = completed_at
    status["heartbeat_at"] = completed_at
    status["completed_at"] = completed_at
    status["progress"] = {"stage": "complete"}
    status["result_sha256"] = sha256_file(result_path)
    atomic_write_json(status_path, status)
    return result


def fail_run(
    run_dir: Path,
    *,
    error: BaseException | str,
    returncode: int | None = None,
) -> dict[str, Any]:
    """Retain a structured error and intentionally omit result.json."""
    run_dir = Path(run_dir).expanduser().resolve()
    result_path = run_dir / "result.json"
    if result_path.exists():
        raise RuntimeError(
            "Refusing to mark a run failed while result.json exists. "
            f"Path: {result_path}."
        )
    materialize_artifact_views(run_dir)
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    now = utc_now()
    status["state"] = "failed"
    status["updated_at"] = now
    status["heartbeat_at"] = now
    status["failed_at"] = now
    status["progress"] = {"stage": "failed"}
    status["error"] = {
        "type": type(error).__name__ if isinstance(error, BaseException) else "Error",
        "message": str(error),
    }
    if returncode is not None:
        status["error"]["returncode"] = int(returncode)
    atomic_write_json(status_path, status)
    return status


def validate_run(run_dir: Path) -> list[str]:
    """Return validation errors for one canonical run bundle."""
    run_dir = Path(run_dir).expanduser().resolve()
    errors = []
    manifest_path = run_dir / "manifest.json"
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.jsonl"
    for path in (manifest_path, status_path, metrics_path):
        if not path.exists():
            errors.append(f"missing {path.name}")
    if errors:
        return errors

    try:
        manifest = _json_object(manifest_path.read_text(encoding="utf-8"))
        status = _json_object(status_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        return [f"invalid reporting JSON: {error}"]

    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append("unexpected manifest schema")
    if status.get("schema_version") != STATUS_SCHEMA:
        errors.append("unexpected status schema")
    for key in ("study_id", "run_id", "arm_id"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            errors.append(f"missing or invalid manifest {key}")
        if status.get(key) != manifest.get(key):
            errors.append(f"status {key} does not match manifest")
    try:
        for _ in iter_metrics(metrics_path):
            pass
    except (ValueError, OSError) as error:
        errors.append(str(error))

    result_path = run_dir / "result.json"
    state = status.get("state")
    if state == "complete" and not result_path.exists():
        errors.append("complete status without result.json")
    if (state == "failed" or is_paused(status)) and result_path.exists():
        errors.append("failed or paused status must not have result.json")
    if state not in ("running", "paused", "complete", "failed"):
        errors.append(f"unsupported state {state!r}")
    if state == "paused":
        pause = status.get("pause", {})
        try:
            checkpoint = run_dir / pause["checkpoint"]
            if (not checkpoint.is_file() or not checkpoint.stat().st_size
                    or sha256_file(checkpoint) != pause["checkpoint_sha256"]):
                errors.append("paused checkpoint missing or hash mismatch")
        except (KeyError, TypeError, OSError):
            errors.append("invalid paused checkpoint record")

    if result_path.exists():
        try:
            result = _json_object(result_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            errors.append(f"invalid result.json: {error}")
            return errors
        if result.get("schema_version") != RESULT_SCHEMA:
            errors.append("unexpected result schema")
        if state == "complete" and status.get("result_sha256") != sha256_file(result_path):
            errors.append("result hash mismatch")
        completion = result.get("completion")
        if not isinstance(completion, dict) or not isinstance(completion.get("criteria_met"), bool):
            errors.append("missing or invalid completion criteria")
        if not isinstance(result.get("terminal_metrics"), dict):
            errors.append("missing or invalid terminal metrics")
        if result.get("manifest_sha256") != sha256_file(manifest_path):
            errors.append("manifest hash mismatch")
        if result.get("metrics_sha256") != sha256_file(metrics_path):
            errors.append("metrics hash mismatch")
        for key in ("study_id", "run_id", "arm_id"):
            if result.get(key) != manifest.get(key):
                errors.append(f"result {key} does not match manifest")
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, list):
            errors.append("missing or invalid artifact index")
        else:
            for artifact in artifacts:
                try:
                    path = run_dir / artifact["path"]
                    if not path.is_file():
                        errors.append(f"missing indexed artifact {artifact['path']}")
                    elif sha256_file(path) != artifact["sha256"]:
                        errors.append(f"artifact hash mismatch {artifact['path']}")
                except (KeyError, TypeError, OSError):
                    errors.append("invalid indexed artifact")
    return errors


def _display_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def _observable_statuses(results_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    # Prune staging/archive trees before descending; never rewrite their evidence.
    excluded = {"archive", "archives", "archived", "stopped-production"}
    for directory, children, files in os.walk(results_root):
        children[:] = sorted(name for name in children
                             if not name.startswith(".") and name not in excluded)
        if "status.json" not in files:
            continue
        path = Path(directory) / "status.json"
        try:
            status = _json_object(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if (
            status.get("schema_version") == STATUS_SCHEMA
            and (status.get("state") == "running" or is_paused(status))
        ):
            rows.append((path, status))
    return rows


def observe_status(
    status: Mapping[str, Any], *, now: datetime | None = None,
    stale_after_minutes: float = 45,
) -> dict[str, Any]:
    """Classify recorded evidence; a fresh heartbeat is not a process liveness check."""
    now = now or datetime.now(timezone.utc)
    age = None
    try:
        heartbeat = datetime.fromisoformat(status.get("heartbeat_at") or status["updated_at"])
        if heartbeat.tzinfo is not None:
            age = (now - heartbeat).total_seconds()
    except (KeyError, TypeError, ValueError):
        pass
    state = status.get("state")
    alerts = []
    if is_paused(status):
        observation = "paused"
    elif state == "running":
        if age is None or age < -60:
            observation = "unknown"
            alerts.append("Running status has a missing or invalid heartbeat time")
        elif age > stale_after_minutes * 60:
            observation = "stale"
            alerts.append("Heartbeat is stale; check launcher, logs and remote synchronization")
        else:
            observation = "running"
    elif state in ("complete", "failed"):
        observation = state
        if state == "failed":
            alerts.append("Run reports failure; inspect the structured error and log tail")
    else:
        observation = "unknown"
        alerts.append(f"Unsupported recorded state: {state!r}")
    return {"observation": observation, "heartbeat_age_seconds": round(age, 1) if age is not None else None,
            "alerts": alerts}


def inspect_runs(run_dirs: Sequence[Path], *, stale_after_minutes: float = 45) -> list[dict[str, Any]]:
    """Return small observations of explicit runs without scanning or hashing payloads."""
    rows = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir).expanduser().resolve()
        row: dict[str, Any] = {"run_dir": str(run_dir)}
        try:
            status = _json_object((run_dir / "status.json").read_text(encoding="utf-8"))
            if status.get("schema_version") != STATUS_SCHEMA:
                raise ValueError("unexpected status schema")
            for key in ("progress", "runtime"):
                if not isinstance(status.get(key, {}), dict):
                    raise ValueError(f"invalid status {key}")
            row.update(observe_status(status, stale_after_minutes=stale_after_minutes))
            row.update(state=status.get("state"), progress=status.get("progress", {}),
                       runtime=status.get("runtime", {}))
            if status.get("error"):
                row["error"] = status["error"]
            if status.get("pause"):
                row["pause"] = status["pause"]
            if status.get("state") == "paused":
                pause = status.get("pause")
                if not isinstance(pause, dict) or not isinstance(pause.get("checkpoint"), str):
                    row["alerts"].append("Paused status has no checkpoint reference")
                elif not (run_dir / pause["checkpoint"]).is_file():
                    row["alerts"].append("Paused checkpoint is missing")
            metrics = run_dir / "metrics.jsonl"
            row["metrics_bytes"] = metrics.stat().st_size if metrics.exists() else None
            row["result_present"] = (run_dir / "result.json").is_file()
            if status.get("state") == "complete" and not row["result_present"]:
                row["alerts"].append("Complete status without result.json")
            if (is_paused(status) or status.get("state") == "failed") and row["result_present"]:
                row["alerts"].append("Paused or failed run has result.json")
        except (ValueError, OSError) as error:
            row.update(observation="unknown", alerts=[str(error)])
        rows.append(row)
    return rows


def render_active_block(
    rows: Iterable[tuple[Path, Mapping[str, Any]]],
    *,
    results_root: Path,
    stale_after_minutes: float = 45,
    now: datetime | None = None,
) -> str:
    grouped: dict[str, list[tuple[Path, Mapping[str, Any]]]] = defaultdict(list)
    for path, status in rows:
        grouped[str(status["study_id"])].append((path, status))
    if not grouped:
        return "_No reporting-contract runs are currently marked `running` or `paused`._"

    lines = []
    for study_id in sorted(grouped):
        lines.extend(
            [
                f"### `{study_id}`",
                "",
                "| Run | Arm | Observation | Progress | Target | Updated | Status file |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for path, status in sorted(
            grouped[study_id], key=lambda item: str(item[1]["run_id"])
        ):
            observation = observe_status(status, now=now, stale_after_minutes=stale_after_minutes)["observation"]
            progress = status.get("progress", {})
            stage = str(progress.get("stage", "running"))
            if observation == "paused" and stage == "failed":
                stage = "checkpointed_pause (legacy encoding)"
            if progress.get("epoch") is not None:
                stage += f" epoch {progress['epoch']}"
                if progress.get("epochs") is not None:
                    stage += f"/{progress['epochs']}"
            runtime = status.get("runtime", {})
            target = (
                runtime.get("target")
                or runtime.get("slurm_job_id")
                or runtime.get("host")
                or "--"
            )
            lines.append(
                "| `{run}` | `{arm}` | {observation} | {progress} | `{target}` | `{updated}` | "
                "`{path}` |".format(
                    run=status["run_id"],
                    arm=status["arm_id"],
                    observation=observation,
                    progress=stage,
                    target=target,
                    updated=status.get("updated_at", "--"),
                    path=_display_path(path, Path(results_root).parent),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def refresh_active_document(results_root: Path, document: Path, *, stale_after_minutes: float = 45) -> None:
    """Replace the generated Active block in an explicitly chosen, non-retired document."""
    results_root = Path(results_root).expanduser().resolve()
    document = Path(document).expanduser().resolve()
    if document in RETIRED_DOCUMENTS:
        raise ValueError(f"{document} is a retired historical snapshot; choose another document.")
    text = document.read_text(encoding="utf-8")
    if text.count(ACTIVE_BEGIN) != 1 or text.count(ACTIVE_END) != 1:
        raise RuntimeError(
            f"Expected exactly one generated Active block in {document}."
        )
    prefix, remainder = text.split(ACTIVE_BEGIN, 1)
    _, suffix = remainder.split(ACTIVE_END, 1)
    body = render_active_block(
        _observable_statuses(results_root),
        results_root=results_root,
        stale_after_minutes=stale_after_minutes,
    )
    updated = f"{prefix}{ACTIVE_BEGIN}\n{body}\n{ACTIVE_END}{suffix}"
    if updated != text:
        temporary = document.with_name(f".{document.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(updated, encoding="utf-8")
            os.replace(temporary, document)
        finally:
            if temporary.exists():
                temporary.unlink()


def runtime_context(
    *,
    target: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    environment = os.environ if environ is None else environ
    return {
        "target": target,
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "conda_default_env": environment.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": environment.get("SLURM_JOB_ID"),
        "slurm_array_task_id": environment.get("SLURM_ARRAY_TASK_ID"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    refresh = subparsers.add_parser(
        "refresh-active", help="Regenerate running, paused and stale observations"
    )
    refresh.add_argument("--results-root", type=Path, default=Path("results"))
    refresh.add_argument(
        "--document", type=Path, required=True,
        help="Explicit output document with Active block markers; retired documents are read-only",
    )
    refresh.add_argument("--stale-after-minutes", type=float, default=45)

    inspect = subparsers.add_parser("inspect-runs", help="Read compact status evidence for explicit runs")
    inspect.add_argument("run_dirs", nargs="+", type=Path)
    inspect.add_argument("--stale-after-minutes", type=float, default=45)

    validate = subparsers.add_parser(
        "validate-run", help="Validate one canonical run bundle"
    )
    validate.add_argument("run_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(args, "stale_after_minutes") and not 0 < args.stale_after_minutes < float("inf"):
        raise ValueError("stale-after-minutes must be positive and finite")
    if args.command == "inspect-runs":
        rows = inspect_runs(args.run_dirs, stale_after_minutes=args.stale_after_minutes)
        print(json.dumps({"runs": rows}, indent=2, allow_nan=False))
        return int(any(row["alerts"] for row in rows))
    if args.command == "refresh-active":
        refresh_active_document(args.results_root, args.document, stale_after_minutes=args.stale_after_minutes)
        return 0
    errors = validate_run(args.run_dir)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"Valid reporting bundle: {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
