"""Canonical command-line entry point for MNIST resistive Conv experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .backend import ExecutionContext, TrainingBackend
from .collection import collect_sweep
from .identity import code_provenance, normalize_code_provenance, run_fingerprint
from .layout import ResultLayout
from .manifest import load_manifest, manifest_entry, publish_manifest
from .runner import ExecutionPolicy, execute_run
from .specs import RunSpec, SweepSpec, require_generic_run_schema


DEFAULT_DATA_ROOT = Path("~/datasets/mnist").expanduser()
STALE_CLAIM_SECONDS = 6 * 60 * 60


def _infrastructure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--retry-failed", "--retry", dest="retry_failed", action="store_true")
    parser.add_argument("--recover-stale", "--recovery", dest="recover_stale", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="plan or execute one complete RunSpec")
    run.add_argument("--config", required=True)
    _infrastructure(run)
    run.add_argument("--plan-only", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="alias for --plan-only")

    sweep = subparsers.add_parser("sweep", help="plan or execute one deterministic SweepSpec")
    source = sweep.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--manifest", help=argparse.SUPPRESS)
    _infrastructure(sweep)
    sweep.add_argument("--plan-only", action="store_true")
    sweep.add_argument("--executor", choices=("local", "slurm"))
    sweep.add_argument("--workers", type=int, default=1)
    sweep.add_argument("--profile")
    sweep.add_argument("--dry-run", action="store_true")
    sweep.add_argument("--job-index", type=int, help=argparse.SUPPRESS)
    sweep.add_argument("--shard-count", type=int, default=1)
    sweep.add_argument("--shard-index", type=int, default=0)

    collect = subparsers.add_parser("collect", help="collect one canonical sweep directory")
    collect.add_argument("--sweep", required=True)
    collect.add_argument("--allow-incomplete", action="store_true")

    lr_study = subparsers.add_parser(
        "lr-study", help="plan or execute one staged Conv learning-rate study"
    )
    lr_study.add_argument(
        "--stage",
        required=True,
        choices=(
            "audit",
            "probe",
            "range",
            "candidates",
            "select",
            "baseline_candidates",
            "select_baseline",
            "confirmations",
            "preflight",
            "core_candidates",
            "select_core",
            "extension_candidates",
            "finalize",
        ),
    )
    lr_study.add_argument("--config")
    lr_study.add_argument("--study")
    lr_study.add_argument("--results-root")
    lr_study.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    lr_study.add_argument("--device", default="cuda")
    lr_study.add_argument("--download", action="store_true")
    lr_study.add_argument("--plan-only", action="store_true")
    lr_study.add_argument("--executor", choices=("local", "slurm"), default="local")
    lr_study.add_argument("--workers", type=int, default=1)
    lr_study.add_argument("--profile")
    lr_study.add_argument("--dry-run", action="store_true")
    lr_study.add_argument("--manifest", help=argparse.SUPPRESS)
    lr_study.add_argument("--entry-index", type=int, help=argparse.SUPPRESS)
    lr_study.add_argument("--finalize-stage", action="store_true", help=argparse.SUPPRESS)
    lr_study.add_argument("--shadow-benchmark", action="store_true", help=argparse.SUPPRESS)
    lr_study.add_argument("--shadow-ready", help=argparse.SUPPRESS)
    lr_study.add_argument("--shadow-start", help=argparse.SUPPRESS)
    lr_study.add_argument("--shadow-scratch-output", help=argparse.SUPPRESS)
    lr_study.add_argument(
        "--shadow-start-timeout-seconds",
        type=float,
        default=900.0,
        help=argparse.SUPPRESS,
    )
    return parser


def _policy(args: argparse.Namespace) -> ExecutionPolicy:
    return ExecutionPolicy(
        max_attempts=3,
        retry_failed=bool(args.retry_failed),
        retry_pruned=False,
        reclaim_stale_after_seconds=STALE_CLAIM_SECONDS if args.recover_stale else None,
    )


def _context(args: argparse.Namespace) -> ExecutionContext:
    return ExecutionContext(dataset_root=args.data_root, device=args.device)


def _require_current_manifest_source(
    manifest: dict[str, Any],
    current_provenance: dict[str, Any],
) -> None:
    planned = normalize_code_provenance(manifest["code_provenance"])
    current = normalize_code_provenance(current_provenance)
    if planned != current:
        raise RuntimeError(
            "Expected worker source provenance to exactly match the immutable sweep plan. "
            f"Provided planned={planned!r}, current={current!r}."
        )


def _run_entry(
    entry: dict[str, Any],
    *,
    args: argparse.Namespace,
    layout: ResultLayout,
    provenance: dict[str, Any],
    backend: TrainingBackend | None,
) -> dict[str, Any]:
    result = execute_run(
        RunSpec.from_dict(entry["run_spec"]),
        _context(args),
        layout,
        provenance,
        backend=backend,
        policy=_policy(args),
    )
    return {
        "job_index": entry["job_index"], "run_id": result.run_id,
        "status": result.status,
        "run_dir": str(result.run_dir) if result.run_dir is not None else None,
        "attempt_id": result.attempt_id,
    }


def _run_command(
    args: argparse.Namespace,
    *,
    provenance: dict[str, Any],
    backend: TrainingBackend | None,
) -> dict[str, Any]:
    spec, layout = RunSpec.from_path(args.config), ResultLayout(args.results_root)
    require_generic_run_schema(spec.data["schema_version"], surface="run")
    run_id = run_fingerprint(spec, provenance)
    path = layout.run_dir(spec.data["label"], run_id)
    if args.plan_only or args.dry_run:
        return {"planned": True, "run_id": run_id, "run_dir": str(path)}
    result = execute_run(
        spec, _context(args), layout, provenance, backend=backend, policy=_policy(args)
    )
    return {
        "planned": False, "run_id": result.run_id, "status": result.status,
        "run_dir": str(result.run_dir) if result.run_dir is not None else None,
        "attempt_id": result.attempt_id,
    }


def _sweep_command(
    args: argparse.Namespace,
    *,
    provenance: dict[str, Any],
    backend: TrainingBackend | None,
    slurm_submitter: Callable[..., dict[str, Any]] | None,
) -> dict[str, Any]:
    layout = ResultLayout(args.results_root)
    if args.manifest:
        if args.shard_count != 1 or args.shard_index != 0:
            raise ValueError(
                f"Expected internal worker mode without shard flags. Provided shard_count={args.shard_count!r}, shard_index={args.shard_index!r}."
            )
        if args.job_index is None:
            raise ValueError(
                f"Expected --job-index with internal --manifest worker mode. Provided value: {args.job_index!r}."
            )
        manifest_path = Path(args.manifest).expanduser().resolve()
        manifest = load_manifest(manifest_path)
        _require_current_manifest_source(manifest, provenance)
        entry = manifest_entry(manifest, args.job_index)
        require_generic_run_schema(
            entry["run_spec"]["schema_version"],
            surface="sweep",
        )
        return _run_entry(
            entry, args=args, layout=layout,
            provenance=provenance, backend=backend,
        )

    spec = SweepSpec.from_path(args.config)
    require_generic_run_schema(
        spec.data["base_run"]["schema_version"],
        surface="sweep",
    )
    manifest, manifest_path = publish_manifest(spec, layout, provenance)
    sweep_dir = manifest_path.parent
    planned = {
        "planned": True, "sweep_id": manifest["sweep_id"],
        "sweep_dir": str(sweep_dir), "manifest_path": str(manifest_path),
        "job_count": len(manifest["entries"]),
    }
    if args.plan_only:
        if args.executor is not None or args.profile is not None or args.dry_run or args.shard_count != 1 or args.shard_index != 0:
            raise ValueError("Expected --plan-only without executor, profile, or dry-run submission options. Provided conflicting options.")
        return planned
    if args.executor is None:
        raise ValueError(f"Expected --executor local or slurm when not planning only. Provided value: {args.executor!r}.")
    if args.executor == "local":
        if args.profile is not None or args.dry_run:
            raise ValueError("Expected local executor without --profile or --dry-run. Provided conflicting options.")
        if isinstance(args.workers, bool) or args.workers < 1:
            raise ValueError(f"Expected --workers to be an integer >= 1. Provided value: {args.workers!r}.")
        if isinstance(args.shard_count, bool) or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise ValueError(
                f"Expected 0 <= shard_index < shard_count with shard_count >= 1. Provided shard_count={args.shard_count!r}, shard_index={args.shard_index!r}."
            )
        selected_entries = [
            entry for entry in manifest["entries"]
            if entry["job_index"] % args.shard_count == args.shard_index
        ]

        def execute_local(entry: dict[str, Any]) -> dict[str, Any]:
            if backend is not None:
                return _run_entry(
                    entry, args=args, layout=layout, provenance=provenance, backend=backend
                )
            command = [
                sys.executable, "-m", "experiments.mnist_conv", "sweep",
                "--manifest", str(manifest_path), "--job-index", str(entry["job_index"]),
                "--results-root", str(layout.root), "--data-root", str(Path(args.data_root).expanduser()),
                "--device", args.device,
            ]
            if args.retry_failed:
                command.append("--retry-failed")
            if args.recover_stale:
                command.append("--recover-stale")
            completed = subprocess.run(command, text=True, capture_output=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    "Expected isolated local worker to exit successfully. "
                    f"Provided job_index={entry['job_index']}, exit={completed.returncode}, stderr={completed.stderr.strip()!r}."
                )
            try:
                return json.loads(completed.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Expected isolated local worker to emit result JSON. Provided stdout: {completed.stdout!r}."
                ) from exc
        if backend is not None:
            # Injected backends are a test seam and execute in-process; the
            # runner's stdout capture is process-global, so keep this seam
            # serial. Production workers below are isolated subprocesses.
            executions = [execute_local(entry) for entry in selected_entries]
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(execute_local, entry) for entry in selected_entries]
                executions = [future.result() for future in futures]
        return {
            **planned, "planned": False, "executor": "local",
            "shard_count": args.shard_count, "shard_index": args.shard_index,
            "selected_job_indices": [entry["job_index"] for entry in selected_entries],
            "executions": executions,
        }

    if args.profile is None:
        raise ValueError(f"Expected --profile for Slurm executor. Provided value: {args.profile!r}.")
    if args.workers != 1:
        raise ValueError(f"Expected --workers only with local executor. Provided value: {args.workers!r}.")
    if args.shard_count != 1 or args.shard_index != 0:
        raise ValueError(
            f"Expected shard flags only with local executor. Provided shard_count={args.shard_count!r}, shard_index={args.shard_index!r}."
        )
    if slurm_submitter is None:
        from experiments.submit_mnist_conv_slurm import submit_sweep
        slurm_submitter = submit_sweep
    submission = slurm_submitter(
        sweep_dir=sweep_dir, profile=args.profile, results_root=layout.root,
        dataset_root=Path(args.data_root).expanduser(), device=args.device,
        dry_run=args.dry_run, retry_failed=args.retry_failed,
        recover_stale=args.recover_stale,
    )
    return {**planned, "planned": False, "executor": "slurm", "submission": submission}


def _collect_command(args: argparse.Namespace) -> dict[str, Any]:
    sweep_dir = Path(args.sweep).expanduser().resolve()
    manifest_path = sweep_dir / "manifest.json"
    if sweep_dir.parent.name != "sweeps":
        raise ValueError(
            f"Expected --sweep to be <results-root>/sweeps/<name>--<sweep-id>. Provided value: {sweep_dir}."
        )
    layout = ResultLayout(sweep_dir.parent.parent)
    return collect_sweep(manifest_path, layout, allow_incomplete=args.allow_incomplete)


def _lr_study_command(
    args: argparse.Namespace,
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    from .lr_stages import (
        execute_v6_shadow_benchmark_entry,
        prepare_study_assets,
    )
    from .lr_study import (
        _require_current_source,
        create_study,
        execute_local_stage,
        execute_manifest_entry,
        finalize_stage,
        load_study,
        plan_only_result,
        publish_lr_stage_manifest,
        submit_slurm_stage,
    )
    from .lr_study_spec import LRStudySpec

    if args.shadow_benchmark:
        from .lr_artifacts import load_stage_manifest

        supplied = {
            "study": args.study,
            "manifest": args.manifest,
            "entry_index": args.entry_index,
            "shadow_ready": args.shadow_ready,
            "shadow_start": args.shadow_start,
            "shadow_scratch_output": args.shadow_scratch_output,
        }
        missing = sorted(name for name, value in supplied.items() if value is None)
        if missing:
            raise ValueError(
                "Expected internal shadow benchmark mode to receive study, manifest, "
                "entry-index, ready/start paths, and a scratch output directory. "
                f"Provided missing values: {missing!r}."
            )
        if (
            args.stage != "baseline_candidates"
            or args.config is not None
            or args.results_root is not None
            or args.download
            or args.plan_only
            or args.executor != "local"
            or args.workers != 1
            or args.profile is not None
            or args.dry_run
            or args.finalize_stage
        ):
            raise ValueError(
                "Expected internal shadow benchmark mode to be noncanonical baseline "
                "candidate execution with no download, planning, profile, finalizer, "
                "or worker fan-out. "
                f"Provided value: {vars(args)!r}."
            )
        root, spec = load_study(args.study)
        if spec.data.get("schema_version") != "mnist-conv-lr-study/v6":
            raise ValueError(
                "Expected shadow benchmark study schema_version to be "
                "mnist-conv-lr-study/v6. "
                f"Provided value: {spec.data.get('schema_version')!r}."
            )
        manifest = load_stage_manifest(
            args.manifest,
            study_dir=root,
            expected_study_id=spec.study_id,
            expected_stage_name="baseline_candidates",
        )
        _require_current_source(manifest["code_provenance"], provenance)
        if not 0 <= args.entry_index < len(manifest["entries"]):
            raise ValueError(
                f"Expected entry_index in [0, {len(manifest['entries']) - 1}]. "
                f"Provided value: {args.entry_index!r}."
            )
        payload = manifest["entries"][args.entry_index]["payload"]
        return execute_v6_shadow_benchmark_entry(
            spec.data,
            root,
            payload["row_id"],
            payload["candidate_role"],
            candidate_payload=payload,
            data_root=args.data_root,
            download=False,
            device=args.device,
            ready_path=args.shadow_ready,
            start_path=args.shadow_start,
            scratch_output_dir=args.shadow_scratch_output,
            warmup_steps=32,
            measured_steps=256,
            start_timeout_seconds=args.shadow_start_timeout_seconds,
        )

    if args.manifest is not None:
        if args.study is None or args.config is not None or args.results_root is not None:
            raise ValueError(
                "Expected internal --manifest mode with --study and without --config or "
                f"--results-root. Provided value: study={args.study!r}, "
                f"config={args.config!r}, results_root={args.results_root!r}."
            )
        if args.finalize_stage:
            if args.entry_index is not None:
                raise ValueError(
                    "Expected --finalize-stage without --entry-index. "
                    f"Provided value: {args.entry_index!r}."
                )
            return finalize_stage(args.study, args.manifest)
        if args.entry_index is None:
            raise ValueError(
                "Expected --entry-index in internal --manifest worker mode. "
                f"Provided value: {args.entry_index!r}."
            )
        return execute_manifest_entry(
            study_dir=args.study,
            manifest_path=args.manifest,
            entry_index=args.entry_index,
            data_root=args.data_root,
            download=args.download,
            device=args.device,
            current_provenance=provenance,
        )

    if (
        args.entry_index is not None
        or args.finalize_stage
        or args.shadow_ready is not None
        or args.shadow_start is not None
        or args.shadow_scratch_output is not None
    ):
        raise ValueError(
            "Expected internal worker/shadow flags only with their internal mode. "
            f"Provided value: entry_index={args.entry_index!r}, "
            f"finalize_stage={args.finalize_stage!r}."
        )
    if args.stage in {"audit", "probe"}:
        if args.config is None or args.study is not None or args.results_root is None:
            raise ValueError(
                f"Expected {args.stage} stage with --config and --results-root, "
                "without --study. "
                f"Provided value: config={args.config!r}, study={args.study!r}, "
                f"results_root={args.results_root!r}."
            )
        spec = LRStudySpec.from_path(args.config)
        layout = ResultLayout(args.results_root)
        study_dir = layout.lr_study_dir(spec.data["name"], spec.study_id)
        if args.plan_only:
            if args.profile is not None or args.dry_run:
                raise ValueError(
                    "Expected --plan-only without --profile or --dry-run. "
                    f"Provided value: profile={args.profile!r}, dry_run={args.dry_run!r}."
                )
            return plan_only_result(spec, study_dir, args.stage)
        if (
            spec.data.get("schema_version") == "mnist-conv-lr-study/v6"
            and args.stage
            in {"audit", "probe", "baseline_candidates", "confirmations"}
        ):
            raise ValueError(
                "Expected non-plan v6 stages to use only the dedicated Jean Zay R3 "
                "wrapper and its internal manifest workers; top-level local and "
                "generic Slurm execution are prohibited. "
                f"Provided stage: {args.stage!r}."
            )
        study_dir, _ = create_study(spec, args.results_root)
        # V6 performs verified v5 reuse (or deterministic fallback rebuilds)
        # inside its Slurm audit entry.  Preparing those assets here would run
        # model/dataset work on a Jean Zay login node before submission.
        if spec.data.get("schema_version") not in {
            "mnist-conv-lr-study/v6",
            "mnist-conv-lr-study/v7",
        }:
            prepare_study_assets(
                spec.data,
                study_dir,
                data_root=args.data_root,
                download=args.download,
                device=args.device,
            )
    else:
        if args.study is None or args.config is not None or args.results_root is not None:
            raise ValueError(
                f"Expected {args.stage} stage with --study, without --config or --results-root. "
                f"Provided value: study={args.study!r}, config={args.config!r}, "
                f"results_root={args.results_root!r}."
            )
        study_dir, spec = load_study(args.study)
        if args.plan_only:
            if args.profile is not None or args.dry_run:
                raise ValueError(
                    "Expected --plan-only without --profile or --dry-run. "
                    f"Provided value: profile={args.profile!r}, dry_run={args.dry_run!r}."
                )
            return plan_only_result(spec, study_dir, args.stage)
        if (
            spec.data.get("schema_version") == "mnist-conv-lr-study/v6"
            and args.stage
            in {"audit", "probe", "baseline_candidates", "confirmations"}
        ):
            raise ValueError(
                "Expected non-plan v6 stages to use only the dedicated Jean Zay R3 "
                "wrapper and its internal manifest workers; top-level local and "
                "generic Slurm execution are prohibited. "
                f"Provided stage: {args.stage!r}."
            )

    manifest, manifest_path = publish_lr_stage_manifest(
        study_dir, spec, args.stage, provenance
    )
    planned = {
        "planned": True,
        "study_id": spec.study_id,
        "study_dir": str(study_dir),
        "stage": args.stage,
        "manifest_path": str(manifest_path),
        "entry_count": len(manifest["entries"]),
    }
    if args.executor == "local":
        if args.profile is not None or args.dry_run:
            raise ValueError(
                "Expected local LR-study execution without --profile or --dry-run. "
                f"Provided value: profile={args.profile!r}, dry_run={args.dry_run!r}."
            )
        result = execute_local_stage(
            study_dir=study_dir,
            manifest_path=manifest_path,
            data_root=args.data_root,
            download=args.download,
            device=args.device,
            workers=args.workers,
            provenance=provenance,
        )
        return {**planned, "planned": False, **result}
    if args.profile is None:
        raise ValueError(
            f"Expected --profile for Slurm execution. Provided value: {args.profile!r}."
        )
    if args.workers != 1:
        raise ValueError(
            f"Expected --workers only for local execution. Provided value: {args.workers!r}."
        )
    submission = submit_slurm_stage(
        study_dir=study_dir,
        manifest_path=manifest_path,
        data_root=args.data_root,
        device=args.device,
        profile_path=args.profile,
        dry_run=args.dry_run,
    )
    return {**planned, "planned": False, "executor": "slurm", "submission": submission}


def main(
    argv: list[str] | None = None,
    *,
    backend: TrainingBackend | None = None,
    source_provenance: dict[str, Any] | None = None,
    slurm_submitter: Callable[..., dict[str, Any]] | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            result = _collect_command(args)
        else:
            provenance = normalize_code_provenance(source_provenance or code_provenance())
            if args.command == "run":
                result = _run_command(args, provenance=provenance, backend=backend)
            elif args.command == "lr-study":
                result = _lr_study_command(args, provenance=provenance)
            else:
                result = _sweep_command(
                    args, provenance=provenance, backend=backend,
                    slurm_submitter=slurm_submitter,
                )
    except Exception as exc:
        message = str(exc)
        if not message.startswith("Expected"):
            message = f"Expected command to complete successfully. Provided error: {message}."
        parser.exit(2, message + "\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0
