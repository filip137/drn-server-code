#!/usr/bin/env python3
"""Public CLI for the immutable perfect-diode 10/20-epoch successor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mnist_conv.perfectdiode_successor_confirmation import (
    CANARY_PACK_INDEX,
    RESULT_JSON_MARKER,
    build_input_bundle,
    load_config,
    output_status,
    plan_config,
    preflight_bundle,
    run_canary_entry,
    run_indexed_entry,
    run_pack,
    run_smoke,
    run_smoke_pack,
    validate_input_bundle,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "conv"
    / "perfectdiode_conv12_best_observed_confirmation_20260727_v1.json"
)
RESULT_MARKER = RESULT_JSON_MARKER


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate, bundle, preflight, smoke, and execute the immutable "
            "twelve-entry perfect-diode best-observed successor."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_config_parser = subparsers.add_parser("validate-config")
    validate_config_parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG
    )
    validate_config_parser.add_argument(
        "--verify-authority-files", action="store_true"
    )
    validate_config_parser.add_argument(
        "--require-production-approval", action="store_true"
    )

    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    build = subparsers.add_parser("build-bundle")
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build.add_argument("--bundle-dir", type=Path, required=True)
    build.add_argument("--source-archive", type=Path, required=True)
    build.add_argument("--source-commit", required=True)
    build.add_argument("--effective-code-fingerprint", required=True)
    build.add_argument("--environment-contract", type=Path, required=True)

    validate_bundle = subparsers.add_parser("validate-bundle")
    validate_bundle.add_argument("--bundle-dir", type=Path, required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--bundle-dir", type=Path, required=True)
    preflight.add_argument("--source-archive", type=Path, required=True)
    preflight.add_argument("--environment-contract", type=Path, required=True)

    for name in ("smoke", "run-entry"):
        command = subparsers.add_parser(name)
        command.add_argument("--bundle-dir", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--source-archive", type=Path, required=True)
        command.add_argument("--environment-contract", type=Path, required=True)
        command.add_argument(
            "--entry-index",
            type=int,
            default=9 if name == "smoke" else None,
            required=name == "run-entry",
        )
        command.add_argument("--device", default="cuda")
        command.add_argument("--download", action="store_true")

    for name in ("smoke-pack", "run-pack", "run-canary-entry"):
        command = subparsers.add_parser(name)
        command.add_argument("--bundle-dir", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--source-archive", type=Path, required=True)
        command.add_argument("--environment-contract", type=Path, required=True)
        if name in {"smoke-pack", "run-pack"}:
            command.add_argument(
                "--pack-index",
                type=int,
                required=name == "run-pack",
                default=CANARY_PACK_INDEX if name == "smoke-pack" else None,
            )
        else:
            command.add_argument("--entry-index", type=int, required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--download", action="store_true")

    status = subparsers.add_parser("status")
    status.add_argument("--bundle-dir", type=Path, required=True)
    status.add_argument("--output-root", type=Path, required=True)
    return parser


def _print(value: Any) -> None:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    print(f"{RESULT_MARKER}{payload}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-config":
        config = load_config(
            args.config,
            verify_authority_files=args.verify_authority_files,
            require_production_approval=args.require_production_approval,
        )
        _print(
            {
                "status": "valid",
                "config": str(args.config.expanduser().resolve()),
                "experiment_id": config["experiment_id"],
                "entry_count": len(config["entries"]),
            }
        )
        return 0
    if args.command == "plan":
        _print(plan_config(args.config))
        return 0
    if args.command == "build-bundle":
        manifest = build_input_bundle(
            config_path=args.config,
            bundle_dir=args.bundle_dir,
            source_archive=args.source_archive,
            source_commit=args.source_commit,
            effective_code_fingerprint=args.effective_code_fingerprint,
            environment_contract=args.environment_contract,
        )
        _print(
            {
                "status": "built",
                "bundle_dir": str(args.bundle_dir.expanduser().resolve()),
                "bundle_id": manifest["bundle_id"],
                "entry_count": manifest["entry_count"],
                "production_ready": manifest["production_ready"],
            }
        )
        return 0
    if args.command == "validate-bundle":
        manifest = validate_input_bundle(args.bundle_dir)
        _print(
            {
                "status": "valid",
                "bundle_dir": str(args.bundle_dir.expanduser().resolve()),
                "bundle_id": manifest["bundle_id"],
                "entry_count": manifest["entry_count"],
                "production_ready": manifest["production_ready"],
            }
        )
        return 0
    if args.command == "preflight":
        result = preflight_bundle(
            bundle_dir=args.bundle_dir,
            source_archive=args.source_archive,
            environment_contract=args.environment_contract,
        )
        _print(result)
        return 0 if result["passed"] else 2
    if args.command == "smoke":
        _print(
            run_smoke(
                bundle_dir=args.bundle_dir,
                output_root=args.output_root,
                data_root=args.data_root,
                source_archive=args.source_archive,
                environment_contract=args.environment_contract,
                entry_index=args.entry_index,
                device=args.device,
                download=args.download,
            )
        )
        return 0
    if args.command == "run-entry":
        _print(
            run_indexed_entry(
                bundle_dir=args.bundle_dir,
                entry_index=args.entry_index,
                output_root=args.output_root,
                data_root=args.data_root,
                source_archive=args.source_archive,
                environment_contract=args.environment_contract,
                device=args.device,
                download=args.download,
            )
        )
        return 0
    if args.command == "run-canary-entry":
        _print(
            run_canary_entry(
                bundle_dir=args.bundle_dir,
                entry_index=args.entry_index,
                output_root=args.output_root,
                data_root=args.data_root,
                source_archive=args.source_archive,
                environment_contract=args.environment_contract,
                device=args.device,
                download=args.download,
            )
        )
        return 0
    if args.command == "smoke-pack":
        _print(
            run_smoke_pack(
                bundle_dir=args.bundle_dir,
                output_root=args.output_root,
                data_root=args.data_root,
                source_archive=args.source_archive,
                environment_contract=args.environment_contract,
                pack_index=args.pack_index,
                device=args.device,
                download=args.download,
            )
        )
        return 0
    if args.command == "run-pack":
        _print(
            run_pack(
                bundle_dir=args.bundle_dir,
                pack_index=args.pack_index,
                output_root=args.output_root,
                data_root=args.data_root,
                source_archive=args.source_archive,
                environment_contract=args.environment_contract,
                device=args.device,
                download=args.download,
            )
        )
        return 0
    if args.command == "status":
        result = output_status(
            bundle_dir=args.bundle_dir,
            output_root=args.output_root,
        )
        _print(result)
        return 2 if result["status"] == "invalid" else 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
