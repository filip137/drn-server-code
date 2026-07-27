#!/usr/bin/env python3
"""Run one manifest-bound Conv3 perfect-diode LR-v2 surface stage.

This is an execution worker, not a launcher.  It never creates tmux sessions
or schedules jobs; the caller must invoke it inside the surface's manifest
route (``main`` or ``akibscomputer``) and supply the observed immutable source
and environment identities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mnist_conv.perfectdiode_conv3_hparam_v2_spec import (
    ALLOWED_HOSTS,
    PUBLIC_STAGE_SEQUENCE,
)
from experiments.mnist_conv.perfectdiode_hparam_v2_execution import (
    execute_representative_preflight_canary,
    execute_surface_stage,
    load_execution_context,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute one hash-bound numerical or conditional stage for one "
            "Conv3 perfect-diode scheme x optimizer surface."
        )
    )
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--surface-manifest", type=Path, required=True)
    parser.add_argument("--surface-id", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--stage", choices=PUBLIC_STAGE_SEQUENCE)
    action.add_argument(
        "--preflight-canary",
        action="store_true",
        help=(
            "Run only the manifest-selected isolated 640-step Adam smoke "
            "canary; write outside all public stage output directories."
        ),
    )
    parser.add_argument("--host", choices=ALLOWED_HOSTS, required=True)
    parser.add_argument(
        "--source-archive",
        type=Path,
        required=True,
        help="Exact source archive whose bytes are bound by the manifest.",
    )
    parser.add_argument(
        "--environment-contract",
        type=Path,
        required=True,
        help="Shared environment-contract receipt bound by the manifest.",
    )
    parser.add_argument(
        "--execution-environment",
        type=Path,
        required=True,
        help="Actual host environment receipt bound to this surface route.",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        help="Imported complete Conv3 T/K shared-asset directory.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Ordinary-MNIST data root used to regenerate bound loaders.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    context = load_execution_context(
        study_path=args.study,
        manifest_path=args.surface_manifest,
        surface_id=args.surface_id,
        host=args.host,
        source_archive_path=args.source_archive,
        environment_contract_path=args.environment_contract,
        execution_environment_path=args.execution_environment,
    )
    if args.preflight_canary:
        result = execute_representative_preflight_canary(
            context,
            asset_dir=args.assets_dir,
            data_root=args.data_root,
            download=args.download,
            device=args.device,
        )
    else:
        result = execute_surface_stage(
            context,
            args.stage,
            asset_dir=args.assets_dir,
            data_root=args.data_root,
            download=args.download,
            device=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return (
        0
        if not args.preflight_canary or result.get("passed") is True
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
