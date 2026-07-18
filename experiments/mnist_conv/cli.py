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
from .specs import RunSpec, SweepSpec


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
        return _run_entry(
            manifest_entry(manifest, args.job_index), args=args, layout=layout,
            provenance=provenance, backend=backend,
        )

    spec = SweepSpec.from_path(args.config)
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
