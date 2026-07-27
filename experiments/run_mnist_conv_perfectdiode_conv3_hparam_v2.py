#!/usr/bin/env python3
"""Materialize and plan the hash-bound Conv3 perfect-diode LR-v2 study.

Numerical stage execution is provided by the separate v2 execution worker.
This controller owns only immutable configuration, route, manifest, and
preflight-plan publication.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mnist_conv.io import atomic_write_json
from experiments.mnist_conv.identity import sha256_file
from experiments.mnist_conv.perfectdiode_conv3_hparam_v2_spec import (
    DEFAULT_TEMPLATE,
    PerfectDiodeConv3HparamStudySpec,
    build_surface_manifest,
    materialize_study,
    representative_preflight_plan,
)
from experiments.mnist_conv.perfectdiode_hparam_v2_execution import (
    build_environment_contract,
    derive_shared_asset_hashes,
    observe_execution_environment,
    validate_environment_contract,
    validate_environment_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize, validate, and plan the ordinary-MNIST Conv3 "
            "perfect-diode SGD/Adam LR-v2 study."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    materialize.add_argument("--tk-selection", type=Path, required=True)
    materialize.add_argument("--tk-selection-sha256", required=True)
    materialize.add_argument(
        "--tk-shared-assets",
        type=Path,
        required=True,
        help=(
            "Complete producer assets directory containing assets.json, "
            "completion.json, and initialization.pt."
        ),
    )
    materialize.add_argument(
        "--tk-source-archive",
        type=Path,
        required=True,
        help=(
            "Exact T/K producer source archive matching manifest.staged_source."
        ),
    )
    materialize.add_argument("--output-dir", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--study", type=Path, required=True)
    validate.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--study", type=Path, required=True)
    plan.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)

    shared_assets = subparsers.add_parser("derive-shared-assets")
    shared_assets.add_argument("--study", type=Path, required=True)
    shared_assets.add_argument("--data-root", type=Path, required=True)
    shared_assets.add_argument("--download", action="store_true")
    shared_assets.add_argument("--output", type=Path, required=True)

    capture_environment = subparsers.add_parser("capture-environment")
    capture_environment.add_argument(
        "--host",
        choices=("main", "akibscomputer"),
        required=True,
    )
    capture_environment.add_argument("--output", type=Path, required=True)

    environment_contract = subparsers.add_parser(
        "build-environment-contract"
    )
    environment_contract.add_argument(
        "--main-receipt", type=Path, required=True
    )
    environment_contract.add_argument(
        "--akibscomputer-receipt", type=Path, required=True
    )
    environment_contract.add_argument("--output", type=Path, required=True)

    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument("--study", type=Path, required=True)
    manifest.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    manifest.add_argument(
        "--shared-assets",
        type=Path,
        required=True,
        help=(
            "JSON object containing the exact split, initialization, T/K "
            "cohort, and three epoch-order SHA-256 values."
        ),
    )
    manifest.add_argument(
        "--routes",
        type=Path,
        help=(
            "Optional JSON object mapping all six surface IDs to main or "
            "akibscomputer. Routes must equal each scheme's T/K host."
        ),
    )
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--source-commit", required=True)
    manifest.add_argument("--source-archive-sha256", required=True)
    manifest.add_argument("--effective-code-fingerprint", required=True)
    manifest.add_argument("--worker-launcher", type=Path, required=True)
    manifest.add_argument(
        "--environment-contract", type=Path, required=True
    )
    manifest.add_argument("--main-environment", type=Path, required=True)
    manifest.add_argument(
        "--akibscomputer-environment", type=Path, required=True
    )

    preflight = subparsers.add_parser("preflight-plan")
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument(
        "--host",
        choices=("main", "akibscomputer"),
        required=True,
    )
    return parser


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Expected {label} to be a readable JSON object. "
            f"Provided value: {path}."
        ) from exc
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Expected {label} to be a JSON object. Provided value: {value!r}."
        )
    return dict(value)


def _print(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _publish_canonical(
    path: Path,
    value: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    target = path.expanduser().resolve()
    if target.exists():
        existing = _read_mapping(target, f"existing {label}")
        if existing != value:
            raise ValueError(
                f"Expected an existing {label} to be canonically identical. "
                f"Provided value: {target}."
            )
    else:
        atomic_write_json(target, value, canonical=True)
    return target


def _plan(spec: PerfectDiodeConv3HparamStudySpec) -> dict[str, Any]:
    rows = []
    for row in spec.rows:
        rows.append(
            {
                "row_id": row["row_id"],
                "scheme": row["scheme"],
                "lr_eligible": row["lr_eligible"],
                "zero_work_reason": row["zero_work_reason"],
                "T": row["inference_iterations"],
                "K": row["training_iterations"],
                "input_gain": row["input_gain"],
                "tk_execution_host": row["upstream_tk"]["execution_host"],
                "operating_point_audit_sha256": row["upstream_tk"][
                    "operating_point_audit_sha256"
                ],
            }
        )
    eligible_rows = sum(int(row["lr_eligible"]) for row in spec.rows)
    return {
        "status": "planned",
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "architecture": "conv3",
        "input_gain": 360.0,
        "training_batch_size": 16,
        "validation_batch_size": 64,
        "candidate_epochs": 3,
        "candidate_total_steps": 10_314,
        "eligible_rows": eligible_rows,
        "eligible_surfaces": eligible_rows * 2,
        "zero_work_surfaces": (3 - eligible_rows) * 2,
        "surface_count": 6,
        "hosts": ["main", "akibscomputer"],
        "shared_asset_manifest_required_before_execution": True,
        "long_confirmation": False,
        "official_test_read": False,
        "rows": rows,
        "launched_jobs": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        spec = materialize_study(
            template_path=args.template,
            tk_selection_path=args.tk_selection,
            tk_selection_sha256=args.tk_selection_sha256,
            tk_shared_assets_path=args.tk_shared_assets,
            tk_source_archive_path=args.tk_source_archive,
            output_dir=args.output_dir,
        )
        _print(
            {
                "status": "materialized",
                "study_id": spec.study_id,
                "config_sha256": spec.config_sha256,
                "study": str(
                    args.output_dir.expanduser().resolve()
                    / "study.resolved.json"
                ),
                "eligible_rows": len(spec.eligible_rows),
                "launched_jobs": 0,
            }
        )
        return 0
    if args.command == "validate":
        spec = PerfectDiodeConv3HparamStudySpec.from_path(
            args.study, template_path=args.template
        )
        _print(
            {
                "status": "valid",
                "study_id": spec.study_id,
                "config_sha256": spec.config_sha256,
                "eligible_rows": len(spec.eligible_rows),
                "launched_jobs": 0,
            }
        )
        return 0
    if args.command == "plan":
        spec = PerfectDiodeConv3HparamStudySpec.from_path(
            args.study, template_path=args.template
        )
        _print(_plan(spec))
        return 0
    if args.command == "derive-shared-assets":
        value = derive_shared_asset_hashes(
            study_path=args.study,
            data_root=args.data_root,
            download=args.download,
        )
        target = _publish_canonical(
            args.output,
            value,
            label="shared-asset hash mapping",
        )
        _print(
            {
                "status": "shared_assets_derived",
                "path": str(target),
                "sha256": sha256_file(target),
                "hash_count": len(value),
                "official_test_read": False,
                "launched_jobs": 0,
            }
        )
        return 0
    if args.command == "capture-environment":
        value = observe_execution_environment(args.host)
        target = _publish_canonical(
            args.output,
            value,
            label=f"{args.host} environment receipt",
        )
        _print(
            {
                "status": "environment_captured",
                "host": args.host,
                "path": str(target),
                "sha256": sha256_file(target),
                "launched_jobs": 0,
            }
        )
        return 0
    if args.command == "build-environment-contract":
        main_receipt = _read_mapping(
            args.main_receipt, "main environment receipt"
        )
        akib_receipt = _read_mapping(
            args.akibscomputer_receipt,
            "akibscomputer environment receipt",
        )
        value = build_environment_contract(main_receipt, akib_receipt)
        target = _publish_canonical(
            args.output,
            value,
            label="environment contract",
        )
        _print(
            {
                "status": "environment_contract_built",
                "path": str(target),
                "sha256": sha256_file(target),
                "hosts": list(value["hosts"]),
                "launched_jobs": 0,
            }
        )
        return 0
    if args.command == "build-manifest":
        study_path = args.study.expanduser().resolve()
        target = args.output.expanduser().resolve()
        if study_path.name != "study.resolved.json":
            raise ValueError(
                "Expected --study to be named study.resolved.json. "
                f"Provided value: {study_path}."
            )
        if target.parent != study_path.parent:
            raise ValueError(
                "Expected --output to be inside the materialized study root. "
                f"Provided study={study_path}, output={target}."
            )
        launcher = args.worker_launcher.expanduser().resolve()
        if not launcher.is_file():
            raise ValueError(
                "Expected --worker-launcher to be an existing regular file. "
                f"Provided value: {launcher}."
            )
        spec = PerfectDiodeConv3HparamStudySpec.from_path(
            study_path, template_path=args.template
        )
        shared = _read_mapping(args.shared_assets, "shared asset hashes")
        routes = (
            None
            if args.routes is None
            else _read_mapping(args.routes, "surface routes")
        )
        environment_contract = validate_environment_contract(
            _read_mapping(
                args.environment_contract, "environment contract"
            )
        )
        main_environment = validate_environment_receipt(
            _read_mapping(args.main_environment, "main environment receipt"),
            expected_host="main",
        )
        akibscomputer_environment = validate_environment_receipt(
            _read_mapping(
                args.akibscomputer_environment,
                "akibscomputer environment receipt",
            ),
            expected_host="akibscomputer",
        )
        if (
            environment_contract["host_receipts"]["main"]
            != main_environment
            or environment_contract["host_receipts"]["akibscomputer"]
            != akibscomputer_environment
        ):
            raise ValueError(
                "Expected --environment-contract to embed the exact "
                "--main-environment and --akibscomputer-environment receipts. "
                "Provided values do not match."
            )
        value = build_surface_manifest(
            spec,
            shared_asset_hashes=shared,
            execution_source={
                "source_commit": args.source_commit,
                "source_archive_sha256": args.source_archive_sha256,
                "effective_code_fingerprint": args.effective_code_fingerprint,
                "worker_launcher_sha256": sha256_file(launcher),
                "environment_contract_sha256": sha256_file(
                    args.environment_contract
                ),
                "host_environment_sha256s": {
                    "main": sha256_file(args.main_environment),
                    "akibscomputer": sha256_file(
                        args.akibscomputer_environment
                    ),
                },
                "resolved_study_path": "study.resolved.json",
                "resolved_study_sha256": sha256_file(study_path),
                "output_root": "surfaces",
            },
            routes=routes,
        )
        if target.exists():
            existing = _read_mapping(target, "existing surface manifest")
            if existing != value:
                raise ValueError(
                    "Expected an existing surface manifest to be canonically "
                    f"identical. Provided value: {target}."
                )
        else:
            atomic_write_json(target, value, canonical=True)
        _print(
            {
                "status": "manifest_published",
                "manifest_id": value["manifest_id"],
                "path": str(target),
                "eligible_surfaces": value["eligible_surface_count"],
                "zero_work_surfaces": value["zero_work_surface_count"],
                "launched_jobs": 0,
            }
        )
        return 0
    if args.command == "preflight-plan":
        manifest_path = args.manifest.expanduser().resolve()
        manifest = _read_mapping(manifest_path, "surface manifest")
        source = manifest.get("execution_source")
        if not isinstance(source, Mapping):
            raise ValueError(
                "Expected surface manifest.execution_source to be a JSON "
                f"object. Provided value: {source!r}."
            )
        relative_study = Path(str(source.get("resolved_study_path")))
        if (
            relative_study != Path("study.resolved.json")
            or relative_study.is_absolute()
            or ".." in relative_study.parts
        ):
            raise ValueError(
                "Expected execution_source.resolved_study_path to be exactly "
                "'study.resolved.json'. "
                f"Provided value: {source.get('resolved_study_path')!r}."
            )
        study_path = manifest_path.parent / relative_study
        if (
            not study_path.is_file()
            or sha256_file(study_path)
            != source.get("resolved_study_sha256")
        ):
            raise ValueError(
                "Expected the manifest-bound resolved study file SHA-256 to "
                f"verify. Provided value: {study_path}."
            )
        spec = PerfectDiodeConv3HparamStudySpec.from_path(study_path)
        _print(
            representative_preflight_plan(
                manifest,
                spec=spec,
                host=args.host,
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
