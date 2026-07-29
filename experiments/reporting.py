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
    _refresh_active_for_run(run_dir)
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
    _refresh_active_for_run(status_path.parent)


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
        "completion": dict(completion or {"criteria_met": True}),
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
    _refresh_active_for_run(run_dir)
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
    _refresh_active_for_run(run_dir)
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
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return [f"invalid reporting JSON: {error}"]

    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append("unexpected manifest schema")
    if status.get("schema_version") != STATUS_SCHEMA:
        errors.append("unexpected status schema")
    if status.get("study_id") != manifest.get("study_id"):
        errors.append("status study_id does not match manifest")
    if status.get("run_id") != manifest.get("run_id"):
        errors.append("status run_id does not match manifest")

    result_path = run_dir / "result.json"
    state = status.get("state")
    if state == "complete" and not result_path.exists():
        errors.append("complete status without result.json")
    if state == "failed" and result_path.exists():
        errors.append("failed status must not have result.json")
    if state not in {"running", "complete", "failed"}:
        errors.append(f"unsupported state {state!r}")

    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            errors.append(f"invalid result.json: {error}")
            return errors
        if result.get("manifest_sha256") != sha256_file(manifest_path):
            errors.append("manifest hash mismatch")
        if result.get("metrics_sha256") != sha256_file(metrics_path):
            errors.append("metrics hash mismatch")
        if result.get("study_id") != manifest.get("study_id"):
            errors.append("result study_id does not match manifest")
        if result.get("run_id") != manifest.get("run_id"):
            errors.append("result run_id does not match manifest")
        for artifact in result.get("artifacts", []):
            path = run_dir / artifact["path"]
            if not path.is_file():
                errors.append(f"missing indexed artifact {artifact['path']}")
            elif sha256_file(path) != artifact["sha256"]:
                errors.append(f"artifact hash mismatch {artifact['path']}")
    return errors


def _find_results_root(run_dir: Path) -> tuple[Path, Path] | None:
    for parent in (run_dir, *run_dir.parents):
        if parent.name != "results":
            continue
        document = parent.parent / "docs" / "current_simulations.md"
        if document.exists():
            return parent, document
    return None


def _refresh_active_for_run(run_dir: Path) -> None:
    discovered = _find_results_root(run_dir)
    if discovered is None:
        return
    results_root, document = discovered
    refresh_active_document(results_root, document)


def _display_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def _running_statuses(results_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    for path in sorted(Path(results_root).rglob("status.json")):
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if (
            status.get("schema_version") == STATUS_SCHEMA
            and status.get("state") == "running"
        ):
            rows.append((path, status))
    return rows


def render_active_block(
    rows: Iterable[tuple[Path, Mapping[str, Any]]],
    *,
    results_root: Path,
) -> str:
    grouped: dict[str, list[tuple[Path, Mapping[str, Any]]]] = defaultdict(list)
    for path, status in rows:
        grouped[str(status["study_id"])].append((path, status))
    if not grouped:
        return "_No reporting-contract runs are currently marked `running`._"

    lines = []
    for study_id in sorted(grouped):
        lines.extend(
            [
                f"### `{study_id}`",
                "",
                "| Run | Arm | Progress | Target | Updated | Status file |",
                "|---|---|---|---|---|---|",
            ]
        )
        for path, status in sorted(
            grouped[study_id], key=lambda item: str(item[1]["run_id"])
        ):
            progress = status.get("progress", {})
            stage = str(progress.get("stage", "running"))
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
                "| `{run}` | `{arm}` | {progress} | `{target}` | `{updated}` | "
                "`{path}` |".format(
                    run=status["run_id"],
                    arm=status["arm_id"],
                    progress=stage,
                    target=target,
                    updated=status.get("updated_at", "--"),
                    path=_display_path(path, Path(results_root).parent),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def refresh_active_document(results_root: Path, document: Path) -> None:
    """Replace only the generated Active block in current_simulations.md."""
    results_root = Path(results_root).expanduser().resolve()
    document = Path(document).expanduser().resolve()
    text = document.read_text(encoding="utf-8")
    if text.count(ACTIVE_BEGIN) != 1 or text.count(ACTIVE_END) != 1:
        raise RuntimeError(
            f"Expected exactly one generated Active block in {document}."
        )
    prefix, remainder = text.split(ACTIVE_BEGIN, 1)
    _, suffix = remainder.split(ACTIVE_END, 1)
    body = render_active_block(
        _running_statuses(results_root),
        results_root=results_root,
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
        "refresh-active", help="Regenerate the running-only Markdown block"
    )
    refresh.add_argument("--results-root", type=Path, default=Path("results"))
    refresh.add_argument(
        "--document", type=Path, default=Path("docs/current_simulations.md")
    )

    validate = subparsers.add_parser(
        "validate-run", help="Validate one canonical run bundle"
    )
    validate.add_argument("run_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "refresh-active":
        refresh_active_document(args.results_root, args.document)
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
