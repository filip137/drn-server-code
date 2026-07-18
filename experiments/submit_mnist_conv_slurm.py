#!/usr/bin/env python3
"""Submit one canonical MNIST Conv manifest and an ``afterany`` collector."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


PROFILE_SCHEMA_VERSION = "mnist-conv-slurm-profile/v1"
RESOURCE_KEYS = {
    "account",
    "partition",
    "qos",
    "constraint",
    "time_limit",
    "cpus_per_task",
    "gpus_per_task",
    "module",
    "idrenv_project",
}


def _fail(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def _read_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise _fail(f"{description} at {path} to be valid JSON", str(exc)) from exc
    if not isinstance(value, dict):
        raise _fail(f"{description} at {path} to be a JSON object", value)
    return value


def _validate_resources(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != RESOURCE_KEYS:
        raise _fail(f"{path} to contain exactly {sorted(RESOURCE_KEYS)}", value)
    for key in ("account", "partition", "time_limit", "idrenv_project"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise _fail(f"{path}.{key} to be a non-empty string", value[key])
    for key in ("qos", "constraint", "module"):
        if value[key] is not None and (not isinstance(value[key], str) or not value[key].strip()):
            raise _fail(f"{path}.{key} to be null or a non-empty string", value[key])
    for key in ("cpus_per_task", "gpus_per_task"):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0:
            raise _fail(f"{path}.{key} to be a non-negative integer", value[key])
    if value["cpus_per_task"] < 1:
        raise _fail(f"{path}.cpus_per_task to be at least 1", value["cpus_per_task"])
    return dict(value)


def load_profile(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = _read_object(source, "executor profile")
    expected = {"schema_version", "name", "executor", "concurrency", "worker", "collector"}
    if set(value) != expected:
        raise _fail(f"executor profile to contain exactly {sorted(expected)}", value)
    if value["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise _fail(f"executor profile schema to be {PROFILE_SCHEMA_VERSION!r}", value["schema_version"])
    if value["executor"] != "slurm":
        raise _fail("executor profile executor to be 'slurm'", value["executor"])
    if not isinstance(value["name"], str) or not value["name"].strip():
        raise _fail("executor profile name to be a non-empty string", value["name"])
    if isinstance(value["concurrency"], bool) or not isinstance(value["concurrency"], int) or value["concurrency"] < 1:
        raise _fail("executor profile concurrency to be a positive integer", value["concurrency"])
    value["worker"] = _validate_resources(value["worker"], "executor profile worker")
    value["collector"] = _validate_resources(value["collector"], "executor profile collector")
    return value


def load_manifest_size(path: str | Path) -> tuple[int, str]:
    source = Path(path).expanduser().resolve()
    from experiments.mnist_conv.manifest import load_manifest

    value = load_manifest(source)
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise _fail("manifest.entries to be a non-empty list", entries)
    for expected_index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("job_index") != expected_index:
            raise _fail("manifest job indices to be contiguous from zero", entry)
    sweep_id = value.get("sweep_id")
    if not isinstance(sweep_id, str) or not sweep_id:
        raise _fail("manifest.sweep_id to be a non-empty string", sweep_id)
    return len(entries), sweep_id


def _resource_args(resources: dict[str, Any]) -> list[str]:
    args = [
        f"--account={resources['account']}",
        f"--partition={resources['partition']}",
        f"--time={resources['time_limit']}",
        f"--cpus-per-task={resources['cpus_per_task']}",
    ]
    if resources["qos"] is not None:
        args.append(f"--qos={resources['qos']}")
    if resources["constraint"] is not None:
        args.append(f"--constraint={resources['constraint']}")
    if resources["gpus_per_task"]:
        args.append(f"--gres=gpu:{resources['gpus_per_task']}")
    return args


def _export_arg(values: dict[str, str]) -> str:
    for key, value in values.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise _fail("Slurm export keys to use uppercase shell-variable syntax", key)
        if "," in value or "\n" in value or "\r" in value:
            raise _fail("Slurm export values not to contain commas or newlines", {key: value})
    return "--export=ALL," + ",".join(f"{key}={value}" for key, value in values.items())


def build_submission_commands(args: argparse.Namespace) -> tuple[list[str], list[str], str]:
    manifest = Path(args.manifest).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    profile = load_profile(args.profile)
    job_count, sweep_id = load_manifest_size(manifest)

    array_script = Path(args.array_script).expanduser().resolve()
    collector_script = Path(args.collector_script).expanduser().resolve()
    for script in (array_script, collector_script):
        if not script.is_file():
            raise _fail("a Slurm wrapper file", str(script))

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", profile["name"]).strip("-") or "mnist-conv"
    log_dir = results_root / "slurm" / sweep_id
    common_export = {
        "MNIST_CONV_REPO_ROOT": str(repo_root),
        "MNIST_CONV_PYTHON": args.python,
        "MNIST_CONV_MANIFEST": str(manifest),
    }
    worker_export = {
        **common_export,
        "MNIST_CONV_RESULTS_ROOT": str(results_root),
        "MNIST_CONV_DATASET_ROOT": str(dataset_root),
        "MNIST_CONV_DEVICE": args.device,
        "MNIST_CONV_RETRY_FAILED": "1" if args.retry_failed else "0",
        "MNIST_CONV_RECOVER_STALE": "1" if args.recover_stale else "0",
        "MNIST_CONV_MODULE": profile["worker"]["module"] or "",
        "MNIST_CONV_IDRENV_PROJECT": profile["worker"]["idrenv_project"],
    }
    collector_export = {
        **common_export,
        "MNIST_CONV_MODULE": profile["collector"]["module"] or "",
        "MNIST_CONV_IDRENV_PROJECT": profile["collector"]["idrenv_project"],
    }
    worker = [
        "sbatch",
        "--parsable",
        f"--job-name=mnist-conv-{safe_name}",
        f"--array=0-{job_count - 1}%{profile['concurrency']}",
        f"--chdir={repo_root}",
        f"--output={log_dir}/%x-%A_%a.out",
        f"--error={log_dir}/%x-%A_%a.err",
        *_resource_args(profile["worker"]),
        _export_arg(worker_export),
        str(array_script),
    ]
    collector_prefix = [
        "sbatch",
        "--parsable",
        f"--job-name=mnist-conv-collect-{safe_name}",
        f"--chdir={repo_root}",
        f"--output={log_dir}/%x-%j.out",
        f"--error={log_dir}/%x-%j.err",
        *_resource_args(profile["collector"]),
        _export_arg(collector_export),
    ]
    return worker, collector_prefix + ["--dependency=afterany:{worker_job_id}", str(collector_script)], sweep_id


def _submitted_job_id(completed: subprocess.CompletedProcess[str]) -> str:
    value = completed.stdout.strip().split(";", maxsplit=1)[0]
    if not re.fullmatch(r"[0-9]+", value):
        raise RuntimeError(f"Expected sbatch --parsable to return a numeric job id. Provided value: {completed.stdout!r}.")
    return value


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--data-root", dest="data_root", default="~/datasets/mnist")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument("--python", default="python")
    parser.add_argument("--array-script", default=str(repo_root / "experiments" / "run_mnist_conv_slurm_array.sh"))
    parser.add_argument("--collector-script", default=str(repo_root / "experiments" / "collect_mnist_conv_slurm.sh"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--recover-stale", action="store_true")
    return parser.parse_args()


def submit_sweep(
    *,
    sweep_dir: str | Path,
    profile: str | Path,
    results_root: str | Path,
    dataset_root: str | Path,
    device: str,
    dry_run: bool = False,
    retry_failed: bool = False,
    recover_stale: bool = False,
    repo_root: str | Path | None = None,
    python: str = "python",
) -> dict[str, Any]:
    """Submit a published sweep manifest and its mandatory afterany collector.

    This callable is the stable bridge used by the canonical
    ``sweep --executor slurm`` command.  All scientific settings are already
    sealed in ``sweep_dir/manifest.json``.
    """

    source_root = Path(repo_root).expanduser().resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    args = argparse.Namespace(
        manifest=str(Path(sweep_dir).expanduser().resolve() / "manifest.json"),
        results_root=str(results_root),
        dataset_root=str(dataset_root),
        profile=str(profile),
        device=device,
        repo_root=str(source_root),
        python=python,
        array_script=str(source_root / "experiments" / "run_mnist_conv_slurm_array.sh"),
        collector_script=str(source_root / "experiments" / "collect_mnist_conv_slurm.sh"),
        retry_failed=retry_failed,
        recover_stale=recover_stale,
    )
    worker, collector_template, sweep_id = build_submission_commands(args)
    if dry_run:
        return {"sweep_id": sweep_id, "submitted": False, "worker": worker, "collector": collector_template}

    log_dir = Path(results_root).expanduser().resolve() / "slurm" / sweep_id
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_result = subprocess.run(worker, check=True, text=True, capture_output=True)
    worker_job_id = _submitted_job_id(worker_result)
    collector = [item.format(worker_job_id=worker_job_id) for item in collector_template]
    collector_result = subprocess.run(collector, check=True, text=True, capture_output=True)
    collector_job_id = _submitted_job_id(collector_result)
    return {
        "sweep_id": sweep_id,
        "submitted": True,
        "worker_job_id": worker_job_id,
        "collector_job_id": collector_job_id,
    }


def main() -> int:
    args = parse_args()
    try:
        result = submit_sweep(
            sweep_dir=Path(args.manifest).expanduser().resolve().parent,
            profile=args.profile,
            results_root=args.results_root,
            dataset_root=args.data_root,
            device=args.device,
            dry_run=args.dry_run,
            retry_failed=args.retry_failed,
            recover_stale=args.recover_stale,
            repo_root=args.repo_root,
            python=args.python,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), flush=True)
        return 2
    print(json.dumps(result, indent=2 if args.dry_run else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
