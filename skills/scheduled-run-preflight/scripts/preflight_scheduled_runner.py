#!/usr/bin/env python3
"""Fail-closed preflight for a scheduled runner's ``--preflight`` contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import NoReturn


def fail(expected: str, provided: object) -> NoReturn:
    raise SystemExit(f"Expected {expected}. Provided value: {provided!r}.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runner_command(runner: Path) -> list[str]:
    if runner.suffix == ".sh":
        return [
            "/bin/bash",
            "-euo",
            "pipefail",
            str(runner),
            "--preflight",
        ]
    if runner.suffix == ".py":
        return [sys.executable, str(runner), "--preflight"]
    if not os.access(runner, os.X_OK):
        fail("runner to be executable or have a .sh/.py suffix", str(runner))
    return [str(runner), "--preflight"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate syntax and execute a scheduled runner's side-effect-free "
            "--preflight path before queueing."
        )
    )
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def write_receipt(path: Path, receipt: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def main() -> int:
    args = parse_args()
    runner = args.runner.expanduser().resolve()
    cwd = args.cwd.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()

    if not runner.is_file():
        fail("runner to be an existing regular file", str(runner))
    if not cwd.is_dir():
        fail("cwd to be an existing directory", str(cwd))
    if args.timeout_seconds <= 0:
        fail("timeout-seconds to be a positive number", args.timeout_seconds)
    if receipt_path == runner:
        fail("receipt path to differ from runner path", str(receipt_path))
    if receipt_path.exists():
        fail("receipt path not to exist before preflight", str(receipt_path))

    initial_sha256 = sha256_file(runner)

    if runner.suffix == ".sh":
        syntax = subprocess.run(
            ["/bin/bash", "-n", str(runner)],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if syntax.returncode != 0:
            fail(
                "shell runner syntax check to exit with code 0",
                {
                    "exit_code": syntax.returncode,
                    "stdout": syntax.stdout,
                    "stderr": syntax.stderr,
                },
            )

    command = runner_command(runner)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=args.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        fail(
            f"runner preflight to finish within {args.timeout_seconds} seconds",
            {
                "command": command,
                "stdout": exc.stdout,
                "stderr": exc.stderr,
            },
        )

    if completed.returncode != 0:
        fail(
            "runner --preflight to exit with code 0",
            {
                "command": command,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )

    final_sha256 = sha256_file(runner)
    if final_sha256 != initial_sha256:
        fail(
            "runner SHA-256 to remain unchanged during preflight",
            {"before": initial_sha256, "after": final_sha256},
        )

    receipt = {
        "schema_version": "scheduled-run-preflight-receipt/v1",
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runner": str(runner),
        "runner_sha256": final_sha256,
        "cwd": str(cwd),
        "preflight_command": command,
        "timeout_seconds": args.timeout_seconds,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    write_receipt(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
