#!/usr/bin/env python3
"""CLI for the immutable bounded-initialization perfect-diode rho comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mnist_conv.identity import sha256_file
from experiments.mnist_conv.perfectdiode_init_rho_comparison import (
    DEFAULT_CONFIG,
    build_manifest,
    collect,
    load_config,
    materialize,
    run_gate,
    run_surface,
    status,
    validate_root,
)


RESULT_MARKER = "PD_INIT_RHO_RESULT_JSON="


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate, materialize, execute, and collect the fresh "
            "bounded-uniform versus bounded-Kaiming perfect-diode rho study."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    plan.add_argument("--source-commit", required=True)
    plan.add_argument("--source-archive-sha256", required=True)
    plan.add_argument("--environment-contract-sha256", required=True)

    create = subparsers.add_parser("materialize")
    create.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--source-commit", required=True)
    create.add_argument("--source-archive-sha256", required=True)
    create.add_argument("--environment-contract", type=Path, required=True)

    for name in ("validate-root", "status", "collect"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--root", type=Path, required=True)
    preflight.add_argument("--source-archive", type=Path, required=True)
    preflight.add_argument("--environment-contract", type=Path, required=True)

    for name in ("run-gate", "run-surface"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--index", type=int, required=True)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--download", action="store_true")
    return parser


def _print(value: Any) -> None:
    print(
        RESULT_MARKER
        + json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )


def _preflight(
    root: Path,
    source_archive: Path,
    environment_contract: Path,
) -> dict[str, Any]:
    destination, manifest, _config = validate_root(root)
    observed_archive = sha256_file(source_archive.expanduser().resolve())
    if observed_archive != manifest["source_archive_sha256"]:
        raise ValueError(
            "Expected source archive SHA-256 to be "
            f"{manifest['source_archive_sha256']}. "
            f"Provided value: {observed_archive!r}."
        )
    observed_environment = sha256_file(
        environment_contract.expanduser().resolve()
    )
    if observed_environment != manifest["environment_contract_sha256"]:
        raise ValueError(
            "Expected environment contract SHA-256 to be "
            f"{manifest['environment_contract_sha256']}. "
            f"Provided value: {observed_environment!r}."
        )
    return {
        "schema_version": (
            "mnist-conv-perfectdiode-initialization-rho-preflight/v1"
        ),
        "status": "passed",
        "root": str(destination),
        "manifest_id": manifest["manifest_id"],
        "source_commit": manifest["source_commit"],
        "source_archive_sha256": observed_archive,
        "environment_contract_sha256": observed_environment,
        "official_test_read": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-config":
        source, config, _spec = load_config(args.config)
        manifest = build_manifest(
            source,
            source_commit="0" * 40,
            source_archive_sha256="0" * 64,
            environment_contract_sha256="0" * 64,
        )
        _print(
            {
                "status": "valid",
                "config": str(source),
                "experiment_id": config["experiment_id"],
                "counts": manifest["counts"],
            }
        )
        return 0
    if args.command == "plan":
        _print(
            build_manifest(
                args.config,
                source_commit=args.source_commit,
                source_archive_sha256=args.source_archive_sha256,
                environment_contract_sha256=(
                    args.environment_contract_sha256
                ),
            )
        )
        return 0
    if args.command == "materialize":
        _print(
            materialize(
                args.config,
                args.root,
                source_commit=args.source_commit,
                source_archive_sha256=args.source_archive_sha256,
                environment_contract=args.environment_contract,
            )
        )
        return 0
    if args.command == "validate-root":
        destination, manifest, _config = validate_root(args.root)
        _print(
            {
                "status": "valid",
                "root": str(destination),
                "manifest_id": manifest["manifest_id"],
                "counts": manifest["counts"],
            }
        )
        return 0
    if args.command == "preflight":
        _print(
            _preflight(
                args.root,
                args.source_archive,
                args.environment_contract,
            )
        )
        return 0
    if args.command == "run-gate":
        result = run_gate(
            args.root,
            args.index,
            data_root=args.data_root,
            device=args.device,
            download=args.download,
        )
        _print(result)
        return 0
    if args.command == "run-surface":
        result = run_surface(
            args.root,
            args.index,
            data_root=args.data_root,
            device=args.device,
            download=args.download,
        )
        _print(result)
        return 0
    if args.command == "status":
        result = status(args.root)
        _print(result)
        return 2 if result["state"] == "invalid" else 0
    if args.command == "collect":
        _print(collect(args.root))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
