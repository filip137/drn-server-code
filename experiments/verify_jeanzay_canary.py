#!/usr/bin/env python3
"""Verify a completed Jean Zay Slurm canary and emit a gate receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "jean-zay-pre-submit-gate/v1"


def fail(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def normalized_state(value: str) -> str:
    return value.strip().split()[0].rstrip("+") if value.strip() else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_receipt_exclusive(path: Path, receipt: dict[str, Any]) -> None:
    """Atomically publish one immutable receipt without overwriting evidence."""

    destination = Path(os.path.abspath(path.expanduser()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise fail(
            "receipt path not to exist before verification",
            destination,
        )
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    payload = (
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(destination.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_text(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command),
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def verify(args: argparse.Namespace) -> dict[str, Any]:
    if not args.job_id.isdigit():
        raise fail("job-id to contain only decimal digits", args.job_id)
    if args.expected_tasks < 1:
        raise fail("expected-tasks to be at least 1", args.expected_tasks)

    fields = (
        "JobIDRaw,JobID,State,ExitCode,Account,Partition,QOS,Constraints"
    )
    output = run_text(
        [
            "sacct",
            "-n",
            "-j",
            args.job_id,
            f"--format={fields}",
            "--parsable2",
        ]
    )
    parent: dict[str, str] | None = None
    tasks: dict[int, dict[str, str]] = {}
    for raw_line in output.splitlines():
        values = raw_line.split("|")
        if len(values) < 8:
            continue
        (
            job_id_raw,
            job_id,
            state,
            exit_code,
            account,
            partition,
            qos,
            constraint,
        ) = (
            value.strip() for value in values[:8]
        )
        row = {
            "job_id_raw": job_id_raw,
            "job_id": job_id,
            "state": normalized_state(state),
            "exit_code": exit_code,
            "account": account,
            "partition": partition,
            "qos": qos,
            "constraint": constraint,
        }
        if job_id == args.job_id:
            if parent is not None:
                raise fail("one parent accounting row", [parent, row])
            parent = row
            continue
        prefix = f"{args.job_id}_"
        if not job_id.startswith(prefix):
            continue
        suffix = job_id[len(prefix) :]
        if not suffix.isdigit():
            continue
        index = int(suffix)
        if index in tasks:
            raise fail("one accounting row per array task", job_id)
        tasks[index] = row

    if parent is not None and (
        parent["state"] != "COMPLETED" or parent["exit_code"] != "0:0"
    ):
        raise fail(
            "any separate canary parent row to be COMPLETED with exit 0:0",
            {"state": parent["state"], "exit_code": parent["exit_code"]},
        )

    expected_contract = {
        "account": args.expected_account,
        "partition": args.expected_partition,
        "qos": args.expected_qos,
        "constraint": args.expected_constraint,
    }
    expected_indices = set(range(args.expected_tasks))
    observed_indices = set(tasks)
    if observed_indices != expected_indices:
        raise fail(
            f"exact array task indices {sorted(expected_indices)!r}",
            sorted(observed_indices),
        )
    bad_tasks = {
        index: {
            "state": row["state"],
            "exit_code": row["exit_code"],
        }
        for index, row in sorted(tasks.items())
        if row["state"] != "COMPLETED" or row["exit_code"] != "0:0"
    }
    if bad_tasks:
        raise fail("every canary task to complete with exit 0:0", bad_tasks)
    contract_rows = [
        *(tasks[index] for index in sorted(tasks)),
        *([] if parent is None else [parent]),
    ]
    bad_contracts = {
        row["job_id"]: {
            key: row[key] for key in expected_contract
        }
        for row in contract_rows
        if {key: row[key] for key in expected_contract}
        != expected_contract
    }
    if bad_contracts:
        raise fail(
            f"every allocation row to match scheduler contract {expected_contract!r}",
            bad_contracts,
        )
    singleton_jobidraw_is_parent = (
        args.expected_tasks == 1
        and tasks[0]["job_id_raw"] == args.job_id
    )

    artifacts = []
    for raw_path in args.require_file:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise fail("each required artifact to be an existing file", path)
        artifacts.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": args.job_id,
        "expected_tasks": args.expected_tasks,
        "singleton_jobidraw_is_parent": singleton_jobidraw_is_parent,
        "scheduler_contract": expected_contract,
        "separate_parent_row": (
            None
            if parent is None
            else {
                "job_id_raw": parent["job_id_raw"],
                "state": parent["state"],
                "exit_code": parent["exit_code"],
            }
        ),
        "tasks": {
            str(index): {
                "job_id_raw": row["job_id_raw"],
                "job_id": row["job_id"],
                "state": row["state"],
                "exit_code": row["exit_code"],
            }
            for index, row in sorted(tasks.items())
        },
        "required_artifacts": artifacts,
        "gate_passed": True,
    }
    if args.receipt is not None:
        write_receipt_exclusive(
            Path(args.receipt),
            receipt,
        )
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--job-id", required=True)
    value.add_argument("--expected-account", required=True)
    value.add_argument("--expected-partition", required=True)
    value.add_argument("--expected-qos", required=True)
    value.add_argument("--expected-constraint", required=True)
    value.add_argument("--expected-tasks", type=int, default=1)
    value.add_argument("--require-file", action="append", default=[])
    value.add_argument("--receipt")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(
        json.dumps(
            verify(args),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
