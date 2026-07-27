#!/usr/bin/env python3
"""Public, non-launching CLI for Conv3 perfect-diode LR-v2 collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mnist_conv.perfectdiode_hparam_v2_orchestration import (
    collector_plan,
    finalize_host_shard,
    finalize_study,
    host_status,
    import_host_shard,
    study_status,
    validate_study,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and collect terminal Conv3 perfect-diode LR-v2 shards. "
            "This command never launches jobs."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def root_argument(command: argparse.ArgumentParser) -> None:
        command.add_argument("--root", type=Path, required=True)

    plan = subparsers.add_parser("plan", help="print the immutable collector plan")
    root_argument(plan)

    status = subparsers.add_parser("status", help="read-only validated status")
    root_argument(status)
    status.add_argument("--host", choices=("main", "akibscomputer"))

    finalize_host = subparsers.add_parser(
        "finalize-host", help="write one terminal host marker"
    )
    root_argument(finalize_host)
    finalize_host.add_argument("--host", choices=("main", "akibscomputer"), required=True)

    import_shard = subparsers.add_parser(
        "import-shard", help="atomically publish a validated incoming host shard"
    )
    root_argument(import_shard)
    import_shard.add_argument("--host", choices=("main", "akibscomputer"), required=True)
    import_shard.add_argument("--incoming", type=Path, required=True)

    finalize = subparsers.add_parser(
        "finalize-study", help="write the terminal six-surface aggregate"
    )
    root_argument(finalize)

    validate = subparsers.add_parser(
        "validate-study", help="read-only terminal study validation"
    )
    root_argument(validate)
    return parser


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.command == "plan":
        return collector_plan(arguments.root)
    if arguments.command == "status":
        if arguments.host:
            return host_status(arguments.root, arguments.host)
        return study_status(arguments.root)
    if arguments.command == "finalize-host":
        return finalize_host_shard(arguments.root, arguments.host)
    if arguments.command == "import-shard":
        return import_host_shard(arguments.root, arguments.incoming, arguments.host)
    if arguments.command == "finalize-study":
        return finalize_study(arguments.root)
    if arguments.command == "validate-study":
        return validate_study(arguments.root)
    raise AssertionError(f"Unhandled command: {arguments.command}")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = _run(arguments)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
