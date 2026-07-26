"""Publish and submit the v6 concurrency benchmark from a Jean Zay login node."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from experiments.mnist_conv.lr_v6_jeanzay import (
    frozen_exports,
    frozen_sbatch_resources,
    login_audit_commands,
    publish_v6_stage_from_login,
    verify_login_r3_allocation,
)
from experiments.submit_mnist_conv_slurm import _export_arg, _submitted_job_id


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def _prepare(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    repo = Path(args.repo_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    if not repo.is_dir():
        raise _error("repo root to be an existing directory", repo)
    if not data_root.is_dir():
        raise _error("MNIST train-data root to be an existing directory", data_root)
    study, manifest_path, manifest = publish_v6_stage_from_login(
        stage="baseline_candidates",
        config_path=args.config,
        results_root=args.results_root,
        study_dir=args.study,
    )
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else manifest_path.parent / "concurrency_benchmark.v1.json"
    )
    try:
        output.relative_to(study)
    except ValueError as exc:
        raise _error("benchmark output to be contained by the v6 study", output) from exc
    wrapper = repo / "experiments/run_mnist_conv_lr_v6_benchmark_jeanzay.slurm"
    if not wrapper.is_file():
        raise _error("v6 benchmark Slurm wrapper to exist", wrapper)
    log_dir = study / "slurm" / "concurrency-benchmark-v6"
    exports = {
        **frozen_exports(),
        "MNIST_CONV_REPO_ROOT": str(repo),
        "MNIST_CONV_PYTHON": args.python,
        "MNIST_CONV_LR_STUDY": str(study),
        "MNIST_CONV_LR_MANIFEST": str(manifest_path),
        "MNIST_CONV_LR_BENCHMARK": str(output),
        "MNIST_CONV_DATASET_ROOT": str(data_root),
        "MNIST_CONV_DEVICE": "cuda",
        "MNIST_CONV_MODULE": args.module,
    }
    command = [
        "sbatch",
        "--parsable",
        "--job-name=conv-lr-v6-benchmark",
        *frozen_sbatch_resources(),
        f"--chdir={repo}",
        f"--output={log_dir}/%x-%j.out",
        f"--error={log_dir}/%x-%j.err",
        _export_arg(exports),
        str(wrapper),
    ]
    next_step = {
        "execution_host": "jean_zay_login_node",
        "run_after": "benchmark_job_terminal_and_successful",
        "working_directory": str(repo),
        "command": [
            args.python,
            "-m",
            "experiments.create_mnist_conv_lr_v6_pack_manifest",
            "--stage-manifest",
            str(manifest_path),
            "--benchmark",
            str(output),
            "--output",
            str(manifest_path.parent / "pack_manifest.v2.json"),
        ],
    }
    metadata = {
        "study_id": manifest["study_id"],
        "study": str(study),
        "candidate_manifest": str(manifest_path),
        "benchmark_output": str(output),
        "log_dir": str(log_dir),
        "required_login_audit": login_audit_commands(),
        "next_login_step": next_step,
    }
    return command, metadata


def submit(args: argparse.Namespace) -> dict[str, Any]:
    command, metadata = _prepare(args)
    if not args.submit:
        return {
            "status": "dry_run",
            "submitted": False,
            "worker": command,
            **metadata,
        }
    audit = verify_login_r3_allocation()
    Path(metadata["log_dir"]).mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return {
        "status": "complete",
        "submitted": True,
        "worker_job_id": _submitted_job_id(completed),
        "login_audit": audit,
        **metadata,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--study")
    parser.add_argument("--results-root")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--module", default="pytorch-gpu/py3/2.5.0")
    parser.add_argument("--output")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if args.config is not None and args.results_root is None:
        parser.error("--config requires --results-root")
    if args.study is not None and args.results_root is not None:
        parser.error("--results-root is only valid with --config")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = submit(args)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), flush=True)
        return 2
    print(json.dumps(result, indent=2 if not args.submit else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
