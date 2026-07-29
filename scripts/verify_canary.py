#!/usr/bin/env python3
"""Verify a singleton Jean Zay canary and write a reproducible gate receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


FIELDS = (
    "JobIDRaw",
    "State",
    "ExitCode",
    "Account",
    "Partition",
    "QOS",
    "Constraints",
    "AllocTRES",
)


def parse_sacct(text: str) -> list[dict[str, str]]:
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        values = line.split("|")
        if len(values) < len(FIELDS):
            raise ValueError(f"Malformed sacct row: {raw_line!r}.")
        rows.append(dict(zip(FIELDS, values[: len(FIELDS)])))
    return rows


def task_rows(job_id: str, rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    tasks = []
    for row in rows:
        raw = row["JobIDRaw"]
        if "." in raw:
            continue
        if raw == job_id or raw.startswith(f"{job_id}_"):
            tasks.append(row)
    return tasks


def require_semantic_json(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Required semantic completion file is missing or empty: {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(
            f"Expected one-row canary summary JSON at {path}, got {type(payload).__name__}."
        )
    row = payload[0]
    if not isinstance(row, dict) or row.get("status") != "complete":
        raise RuntimeError(f"Canary summary is not complete: {row!r}.")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "row": row,
    }


def validate_task_rows(
    job_id: str,
    tasks: Sequence[dict[str, str]],
    *,
    expected_tasks: int,
    expected_account: str,
    expected_partition: str,
    expected_qos: str,
    expected_constraint: str,
) -> None:
    if len(tasks) != expected_tasks:
        raise RuntimeError(
            f"Expected {expected_tasks} task row(s), found {len(tasks)}: {tasks!r}."
        )
    raw_ids = [row["JobIDRaw"] for row in tasks]
    if expected_tasks == 1:
        allowed = {job_id, f"{job_id}_0"}
        if raw_ids[0] not in allowed:
            raise RuntimeError(
                f"Unexpected singleton canary task row {raw_ids[0]!r}; "
                f"expected one of {sorted(allowed)!r}."
            )
    elif sorted(raw_ids) != [f"{job_id}_{index}" for index in range(expected_tasks)]:
        raise RuntimeError(
            "Unexpected, missing, or duplicate array task rows: "
            f"{raw_ids!r}."
        )

    for row in tasks:
        expected = {
            "State": "COMPLETED",
            "ExitCode": "0:0",
            "Account": expected_account,
            "Partition": expected_partition,
            "QOS": expected_qos,
            "Constraints": expected_constraint,
        }
        mismatches = {
            key: {"expected": value, "actual": row.get(key)}
            for key, value in expected.items()
            if row.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Canary scheduler contract mismatch: {mismatches!r}.")
        if "gres/gpu=1" not in row.get("AllocTRES", ""):
            raise RuntimeError(f"Expected one allocated GPU, got {row.get('AllocTRES')!r}.")


def verify(args: argparse.Namespace) -> dict[str, object]:
    command = [
        "sacct",
        "-j",
        args.job_id,
        "--noheader",
        "--parsable2",
        f"--format={','.join(FIELDS)}",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = parse_sacct(completed.stdout)
    tasks = task_rows(args.job_id, rows)
    validate_task_rows(
        args.job_id,
        tasks,
        expected_tasks=args.expected_tasks,
        expected_account=args.expected_account,
        expected_partition=args.expected_partition,
        expected_qos=args.expected_qos,
        expected_constraint=args.expected_constraint,
    )

    semantic = require_semantic_json(Path(args.require_file))
    receipt = {
        "schema_version": "jean-zay-canary-gate/v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "job_id": args.job_id,
        "expected": {
            "account": args.expected_account,
            "partition": args.expected_partition,
            "qos": args.expected_qos,
            "constraint": args.expected_constraint,
            "tasks": args.expected_tasks,
        },
        "task_rows": tasks,
        "semantic_completion": semantic,
        "status": "passed",
    }
    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-account", required=True)
    parser.add_argument("--expected-partition", required=True)
    parser.add_argument("--expected-qos", required=True)
    parser.add_argument("--expected-constraint", required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--require-file", required=True)
    parser.add_argument("--receipt", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    receipt = verify(parse_args(argv))
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
