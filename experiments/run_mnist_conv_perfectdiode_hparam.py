#!/usr/bin/env python3
"""CLI for the host-sharded perfect-diode Conv1/Conv2 LR screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mnist_conv.identity import code_provenance
from experiments.mnist_conv.perfectdiode_hparam import (
    HOST_ARCHITECTURES,
    create_host_shard,
    environment_fingerprint,
    execute_screen_manifest_entry,
    merge_screen_shards,
    plan_host_screen,
    run_aggregate_confirmations,
    run_host_screen,
    run_representative_preflight_canary,
    shard_status,
)
from experiments.mnist_conv.perfectdiode_hparam_spec import (
    PerfectDiodeHparamStudySpec,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "conv"
    / "perfectdiode_conv12_sgd_adam_hparam_ordinary_mnist_v1.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, execute, resume, and merge the ordinary-MNIST "
            "perfect-diode Conv1/Conv2 LR screen."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    plan.add_argument("--host", choices=tuple(HOST_ARCHITECTURES), required=True)

    initialize = subparsers.add_parser("init-shard")
    initialize.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    initialize.add_argument("--results-root", type=Path, required=True)
    initialize.add_argument(
        "--host", choices=tuple(HOST_ARCHITECTURES), required=True
    )
    initialize.add_argument("--source-commit", required=True)
    initialize.add_argument("--source-archive-sha256", required=True)

    run_host = subparsers.add_parser("run-host")
    run_host.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_host.add_argument("--results-root", type=Path, required=True)
    run_host.add_argument(
        "--host", choices=tuple(HOST_ARCHITECTURES), required=True
    )
    run_host.add_argument("--data-root", type=Path, required=True)
    run_host.add_argument("--device", default="cuda")
    run_host.add_argument("--download", action="store_true")
    run_host.add_argument("--source-commit", required=True)
    run_host.add_argument("--source-archive-sha256", required=True)
    run_host.add_argument(
        "--stop-after",
        choices=(
            "audit",
            "assets",
            "fixed_tk_gradient_security",
            "optimizer_probe",
            "rho_canary_core",
            "rho_core_candidates",
            "select_core",
            "rho_canary_extension",
            "rho_extension_candidates",
            "select_final",
        ),
        default="select_final",
    )

    entry = subparsers.add_parser("run-entry")
    entry.add_argument("--shard", type=Path, required=True)
    entry.add_argument("--manifest", type=Path, required=True)
    entry.add_argument("--entry-index", type=int, required=True)
    entry.add_argument("--data-root", type=Path, required=True)
    entry.add_argument("--device", default="cuda")
    entry.add_argument("--download", action="store_true")

    preflight = subparsers.add_parser("preflight-canary")
    preflight.add_argument("--shard", type=Path, required=True)
    preflight.add_argument("--data-root", type=Path, required=True)
    preflight.add_argument("--device", default="cuda")
    preflight.add_argument("--download", action="store_true")

    merge = subparsers.add_parser("merge")
    merge.add_argument("--destination", type=Path, required=True)
    merge.add_argument("--akib-shard", type=Path, required=True)
    merge.add_argument("--trex-shard", type=Path, required=True)

    aggregate = subparsers.add_parser("run-aggregate")
    aggregate.add_argument("--aggregate", type=Path, required=True)
    aggregate.add_argument("--data-root", type=Path, required=True)
    aggregate.add_argument("--device", default="cuda")
    aggregate.add_argument("--download", action="store_true")

    status = subparsers.add_parser("status")
    status.add_argument("--shard", type=Path, required=True)
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        spec = PerfectDiodeHparamStudySpec.from_path(args.config)
        _print(plan_host_screen(spec, args.host))
        return 0
    if args.command == "init-shard":
        spec = PerfectDiodeHparamStudySpec.from_path(args.config)
        _root, shard = create_host_shard(
            spec,
            args.results_root,
            args.host,
            provenance=code_provenance(),
            source_commit=args.source_commit,
            source_archive_sha256=args.source_archive_sha256,
            environment=environment_fingerprint(),
        )
        _print(
            {
                "status": "initialized",
                "study_id": spec.study_id,
                "host": args.host,
                "architecture": HOST_ARCHITECTURES[args.host],
                "shard_dir": str(shard),
                "launched_jobs": 0,
            }
        )
        return 0
    if args.command == "run-host":
        spec = PerfectDiodeHparamStudySpec.from_path(args.config)
        _print(
            run_host_screen(
                spec=spec,
                results_root=args.results_root,
                host=args.host,
                data_root=args.data_root,
                device=args.device,
                download=args.download,
                provenance=code_provenance(),
                source_commit=args.source_commit,
                source_archive_sha256=args.source_archive_sha256,
                environment=environment_fingerprint(),
                stop_after=args.stop_after,
            )
        )
        return 0
    if args.command == "run-entry":
        _print(
            execute_screen_manifest_entry(
                shard_dir=args.shard,
                manifest_path=args.manifest,
                entry_index=args.entry_index,
                data_root=args.data_root,
                device=args.device,
                download=args.download,
            )
        )
        return 0
    if args.command == "preflight-canary":
        _print(
            run_representative_preflight_canary(
                shard_dir=args.shard,
                data_root=args.data_root,
                device=args.device,
                download=args.download,
            )
        )
        return 0
    if args.command == "merge":
        aggregate, receipt = merge_screen_shards(
            args.destination,
            akib_shard=args.akib_shard,
            trex_shard=args.trex_shard,
        )
        _print(
            {
                "status": "merged",
                "aggregate_dir": str(aggregate),
                "receipt": receipt,
            }
        )
        return 0
    if args.command == "run-aggregate":
        _print(
            run_aggregate_confirmations(
                aggregate_dir=args.aggregate,
                data_root=args.data_root,
                device=args.device,
                download=args.download,
            )
        )
        return 0
    if args.command == "status":
        _print(shard_status(args.shard))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
