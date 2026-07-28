#!/usr/bin/env python3
"""Resumably copy a remote result bundle into local staging and write a receipt."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


SCHEMA_VERSION = "remote-result-sync-receipt/v1"
HOST_TARGETS = {
    "akib": "akib",
    "trex": "filip@trex",
    "jean-zay": "jean-zay",
}
METADATA_PATTERNS = (
    "*.json",
    "*.jsonl",
    "*.csv",
    "*.tsv",
    "*.yaml",
    "*.yml",
    "*.toml",
    "*.md",
    "*.sha256",
)
METADATA_EXCLUDES = (
    "**/step_log.csv",
)
BROAD_REMOTE_ROOTS = {
    "akib": {
        "/home/filiposana/results",
        "/home/filiposana/server_code/results",
    },
    "trex": {
        "/home/filip/results",
        "/home/filip/server_code/results",
    },
    "jean-zay": {
        "/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results",
    },
}
SAFE_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9._/@+=,:%-]+(?:/[A-Za-z0-9._@+=,:%-]+)*$")
SAFE_RELATIVE_PATH = re.compile(r"^[A-Za-z0-9._/@+=,:%-]+$")
SAFE_RELATIVE_PATTERN = re.compile(r"^[A-Za-z0-9._/@+=,:?*%{}\[\]-]+$")


def _error(expected: str, provided: object, field: str) -> ValueError:
    return ValueError(
        f"Expected {field} to {expected}. Provided value: {provided!r}."
    )


def _validate_remote_path(value: str) -> str:
    if not SAFE_REMOTE_PATH.fullmatch(value):
        raise _error(
            "be a normalized absolute POSIX path without whitespace, '..', or shell syntax",
            value,
            "--remote-path",
        )
    if ".." in PurePosixPath(value).parts:
        raise _error("contain no '..' component", value, "--remote-path")
    return value.rstrip("/") or "/"


def _validate_relative_pattern(value: str, field: str) -> str:
    if (
        not value
        or value.startswith("/")
        or not SAFE_RELATIVE_PATTERN.fullmatch(value)
        or ".." in PurePosixPath(value).parts
    ):
        raise _error(
            "be a safe bundle-relative path or glob without '..'",
            value,
            field,
        )
    return value


def _validate_relative_path(value: str, field: str) -> str:
    if (
        not value
        or value.startswith("/")
        or not SAFE_RELATIVE_PATH.fullmatch(value)
        or ".." in PurePosixPath(value).parts
    ):
        raise _error(
            "be a safe exact bundle-relative path without globs or '..'",
            value,
            field,
        )
    return value


def _absolute_local_path(value: str, field: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not Path(value).expanduser().is_absolute():
        raise _error("be an absolute local path", value, field)
    return path


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _rsync_filters(
    mode: str,
    completion_markers: Sequence[str],
    extra_includes: Sequence[str],
) -> list[str]:
    if mode == "full":
        return []
    filters = ["--prune-empty-dirs", "--include=*/"]
    for pattern in METADATA_EXCLUDES:
        filters.append(f"--exclude={pattern}")
    for pattern in METADATA_PATTERNS:
        filters.append(f"--include={pattern}")
    for value in (*completion_markers, *extra_includes):
        filters.append(f"--include=/{value}")
    filters.append("--exclude=*")
    return filters


def _rsync_command(
    *,
    target: str,
    remote_path: str,
    local_stage: Path,
    mode: str,
    completion_markers: Sequence[str],
    extra_includes: Sequence[str],
    verify: bool,
    full_checksum: bool,
) -> list[str]:
    command = [
        "rsync",
        "--archive",
        "--partial",
        "--partial-dir=.rsync-partial",
        "--protect-args",
        "--itemize-changes",
        "--out-format=%i %n%L",
    ]
    command.extend(_rsync_filters(mode, completion_markers, extra_includes))
    if verify:
        command.append("--dry-run")
        if mode == "metadata" or full_checksum:
            command.append("--checksum")
    command.extend(
        [
            f"{target}:{remote_path}/",
            f"{local_stage}/",
        ]
    )
    return command


def _completion_check_command(
    target: str, remote_path: str, marker: str
) -> list[str]:
    marker_path = f"{remote_path}/{marker}"
    # POSIX ``test`` does not specify ``--`` and some remote /bin/sh
    # implementations parse it as an operand.  Both path components have
    # already passed the strict shell-syntax validators above, so quoting the
    # resulting absolute path is sufficient and portable.
    remote_command = f"test -f {shlex.quote(marker_path)}"
    return ["ssh", "-o", "BatchMode=yes", target, remote_command]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metadata_path_selected(
    relative: str,
    *,
    completion_markers: Sequence[str],
    extra_includes: Sequence[str],
) -> bool:
    if relative.endswith("/step_log.csv") or relative == "step_log.csv":
        return False
    if relative in completion_markers:
        return True
    if any(fnmatch.fnmatchcase(relative, pattern) for pattern in extra_includes):
        return True
    name = PurePosixPath(relative).name
    return any(
        fnmatch.fnmatchcase(name, pattern) for pattern in METADATA_PATTERNS
    )


def _local_inventory(
    root: Path,
    *,
    content_hashes: bool,
    metadata_only: bool,
    completion_markers: Sequence[str],
    extra_includes: Sequence[str],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    symlink_count = 0
    for path in sorted(root.rglob("*")):
        if ".rsync-partial" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            symlink_count += 1
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if metadata_only and not _metadata_path_selected(
            relative,
            completion_markers=completion_markers,
            extra_includes=extra_includes,
        ):
            continue
        stat = path.stat()
        record: dict[str, Any] = {
            "path": relative,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if content_hashes:
            record["sha256"] = _sha256_file(path)
        records.append(record)
    return {
        "file_count": len(records),
        "byte_count": sum(record["bytes"] for record in records),
        "symlink_count": symlink_count,
        "inventory_sha256": _sha256_json(records),
        "content_hashed": content_hashes,
    }


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _run_command(
    command: Sequence[str], *, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        capture_output=capture_output,
        text=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a run-specific result bundle from Akib, Trex, or Jean Zay "
            "into local staging, verify the transfer, and write a receipt."
        )
    )
    parser.add_argument("--host", choices=tuple(HOST_TARGETS), required=True)
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--local-stage", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument(
        "--mode", choices=("metadata", "full"), default="metadata"
    )
    parser.add_argument(
        "--completion-marker",
        action="append",
        default=[],
        help=(
            "Bundle-relative marker that must exist remotely; repeat to "
            "require more than one marker."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Copy a crashed or partial bundle and record it as incomplete.",
    )
    parser.add_argument(
        "--include-relative",
        action="append",
        default=[],
        help=(
            "Additional bundle-relative rsync include for metadata mode, "
            "for example 'slurm/*.err'."
        ),
    )
    parser.add_argument(
        "--full-checksum",
        action="store_true",
        help=(
            "Checksum every file during full-mode verification. Use for "
            "legacy bundles without trustworthy artifact manifests."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the exact checks and rsync commands without executing them.",
    )
    return parser


def _normalized_arguments(args: argparse.Namespace) -> dict[str, Any]:
    remote_path = _validate_remote_path(args.remote_path)
    local_stage = _absolute_local_path(args.local_stage, "--local-stage")
    receipt = _absolute_local_path(args.receipt, "--receipt")
    if remote_path in BROAD_REMOTE_ROOTS[args.host]:
        raise _error(
            "name one run, study, or shard below the host result root",
            remote_path,
            "--remote-path",
        )
    if ".incoming" not in local_stage.parts:
        raise _error(
            "be inside a '.incoming' staging directory",
            str(local_stage),
            "--local-stage",
        )
    if _is_within(receipt, local_stage):
        raise _error(
            "be outside the staged result payload",
            str(receipt),
            "--receipt",
        )
    markers = [
        _validate_relative_path(value, "--completion-marker")
        for value in args.completion_marker
    ]
    if not markers and not args.allow_incomplete:
        raise _error(
            "provide at least one marker or be paired with --allow-incomplete",
            markers,
            "--completion-marker",
        )
    extras = [
        _validate_relative_pattern(value, "--include-relative")
        for value in args.include_relative
    ]
    if args.mode == "full" and extras:
        raise _error(
            "be omitted in full mode because every file is included",
            extras,
            "--include-relative",
        )
    if args.mode == "metadata" and args.full_checksum:
        raise _error(
            "be omitted in metadata mode, which already uses checksums",
            True,
            "--full-checksum",
        )
    return {
        "host": args.host,
        "target": HOST_TARGETS[args.host],
        "remote_path": remote_path,
        "local_stage": local_stage,
        "receipt": receipt,
        "mode": args.mode,
        "markers": markers,
        "allow_incomplete": args.allow_incomplete,
        "extra_includes": extras,
        "full_checksum": args.full_checksum,
        "plan_only": args.plan_only,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    values = _normalized_arguments(args)
    checks = [
        _completion_check_command(
            values["target"], values["remote_path"], marker
        )
        for marker in values["markers"]
    ]
    sync = _rsync_command(
        target=values["target"],
        remote_path=values["remote_path"],
        local_stage=values["local_stage"],
        mode=values["mode"],
        completion_markers=values["markers"],
        extra_includes=values["extra_includes"],
        verify=False,
        full_checksum=values["full_checksum"],
    )
    verify = _rsync_command(
        target=values["target"],
        remote_path=values["remote_path"],
        local_stage=values["local_stage"],
        mode=values["mode"],
        completion_markers=values["markers"],
        extra_includes=values["extra_includes"],
        verify=True,
        full_checksum=values["full_checksum"],
    )
    plan = {
        "schema_version": SCHEMA_VERSION,
        "status": "planned",
        "host": values["host"],
        "ssh_target": values["target"],
        "remote_path": values["remote_path"],
        "local_stage": str(values["local_stage"]),
        "receipt": str(values["receipt"]),
        "mode": values["mode"],
        "completion_state": (
            "incomplete_with_valid_markers"
            if values["allow_incomplete"] and values["markers"]
            else (
                "incomplete_without_marker"
                if values["allow_incomplete"]
                else "markers_required"
            )
        ),
        "completion_markers": values["markers"],
        "extra_includes": values["extra_includes"],
        "commands": {
            "completion_checks": checks,
            "sync": sync,
            "verify": verify,
        },
    }
    if values["plan_only"]:
        return plan
    if values["receipt"].exists():
        raise _error(
            "not already exist; use a new receipt path for each attempt",
            str(values["receipt"]),
            "--receipt",
        )
    values["local_stage"].mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    for command in checks:
        _run_command(command)
    _run_command(sync)
    verification = _run_command(verify, capture_output=True)
    changed_items = [
        line for line in verification.stdout.splitlines() if line.strip()
    ]
    if changed_items:
        preview = changed_items[:10]
        raise RuntimeError(
            "Expected verification rsync to report no remaining changes. "
            f"Provided changed items: {preview!r}."
        )
    inventory = _local_inventory(
        values["local_stage"],
        content_hashes=values["mode"] == "metadata",
        metadata_only=values["mode"] == "metadata",
        completion_markers=values["markers"],
        extra_includes=values["extra_includes"],
    )
    receipt = {
        **plan,
        "status": "synced",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "verification": {
            "method": (
                "rsync_checksum_dry_run"
                if values["mode"] == "metadata" or values["full_checksum"]
                else "rsync_quick_check_dry_run"
            ),
            "changed_item_count": 0,
            "scientific_validation": "pending",
        },
        "local_inventory": inventory,
    }
    _write_json_atomic(values["receipt"], receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = execute(_parser().parse_args(argv))
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
