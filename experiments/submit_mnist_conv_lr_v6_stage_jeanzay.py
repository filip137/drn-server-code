"""Publish and submit one packed v6 stage from a Jean Zay login node."""

from __future__ import annotations

import argparse
import os
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
from experiments.mnist_conv.lr_v6_packing import create_v6_pack_manifest
from experiments.submit_mnist_conv_slurm import _export_arg, _submitted_job_id


V5_STUDY_ID = (
    "lrstudy_5da3a7452c00d42325bfe930d80f037791125d3ce7b5ddd79be13ff8b96b49e2"
)
V5_STUDY_DIRECTORY = (
    "conv1-conv2-hard-sigmoid-ordinary-mnist-architecture-relative-median-study--"
    + V5_STUDY_ID
)
SUPPORTED_STAGES = ("audit", "probe", "baseline_candidates", "confirmations")


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def frozen_v5_parent_study(user: str | None = None) -> Path:
    selected_user = user or os.environ.get("USER") or Path.home().name
    if not selected_user or "/" in selected_user:
        raise _error("current Jean Zay user to be a single path component", selected_user)
    return (
        Path("/lustre/fsn1/projects/rech/fmu")
        / selected_user
        / "server_code/results"
        / V5_STUDY_ID
        / "lr_studies"
        / V5_STUDY_DIRECTORY
    )


def _validate_v5_parent(args: argparse.Namespace) -> str:
    if args.stage == "audit":
        if args.v5_study is None:
            raise _error("--v5-study for the audit stage", args.v5_study)
        supplied = Path(args.v5_study).expanduser().absolute()
        expected = frozen_v5_parent_study()
        if supplied != expected:
            raise _error(
                "--v5-study to identify the frozen content-addressed v5 R3 parent",
                supplied,
            )
        return str(supplied)
    if args.v5_study is not None:
        raise _error("--v5-study only for the audit stage", args.v5_study)
    return ""


def _prepare(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    repo = Path(args.repo_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    if not repo.is_dir():
        raise _error("repo root to be an existing directory", repo)
    if not data_root.is_dir():
        raise _error("MNIST train-data root to be an existing directory", data_root)
    v5_parent = _validate_v5_parent(args)
    study, manifest_path, manifest = publish_v6_stage_from_login(
        stage=args.stage,
        config_path=args.config,
        results_root=args.results_root,
        study_dir=args.study,
    )
    benchmark = None
    if args.benchmark is not None:
        benchmark = Path(args.benchmark).expanduser().resolve()
        try:
            benchmark.relative_to(study)
        except ValueError as exc:
            raise _error("benchmark to be contained by the v6 study", benchmark) from exc
        if not benchmark.is_file():
            raise _error("benchmark to be an existing file", benchmark)
    if args.stage in {"baseline_candidates", "confirmations"} and benchmark is None:
        raise _error(f"--benchmark for stage {args.stage}", args.benchmark)
    if args.stage in {"audit", "probe"} and benchmark is not None:
        raise _error(f"no --benchmark for stage {args.stage}", benchmark)
    pack_path = (
        Path(args.pack_manifest).expanduser().resolve()
        if args.pack_manifest
        else manifest_path.parent / "pack_manifest.v2.json"
    )
    try:
        pack_path.relative_to(study)
    except ValueError as exc:
        raise _error("pack manifest to be contained by the v6 study", pack_path) from exc
    pack = create_v6_pack_manifest(
        manifest_path,
        pack_path,
        benchmark_path=benchmark,
    )
    wrapper = repo / "experiments/run_mnist_conv_lr_v6_pack_jeanzay.slurm"
    if not wrapper.is_file():
        raise _error("v6 packed Slurm wrapper to exist", wrapper)
    log_dir = study / "slurm" / f"{args.stage}-v6-pack"
    exports = {
        **frozen_exports(),
        "MNIST_CONV_REPO_ROOT": str(repo),
        "MNIST_CONV_PYTHON": args.python,
        "MNIST_CONV_LR_STUDY": str(study),
        "MNIST_CONV_LR_MANIFEST": str(manifest_path),
        "MNIST_CONV_LR_PACK_MANIFEST": str(pack_path),
        "MNIST_CONV_LR_BENCHMARK": "" if benchmark is None else str(benchmark),
        "MNIST_CONV_LR_STAGE": args.stage,
        "MNIST_CONV_LR_V5_STUDY": v5_parent,
        "MNIST_CONV_DATASET_ROOT": str(data_root),
        "MNIST_CONV_DEVICE": "cuda",
        "MNIST_CONV_MODULE": args.module,
    }
    command = [
        "sbatch",
        "--parsable",
        f"--job-name=conv-lr-v6-{args.stage}",
        *frozen_sbatch_resources(),
        f"--chdir={repo}",
        f"--output={log_dir}/%x-%j.out",
        f"--error={log_dir}/%x-%j.err",
        _export_arg(exports),
        str(wrapper),
    ]
    login_finalizer = {
        "execution_host": "jean_zay_login_node",
        "run_after": "worker_job_terminal_and_all_waves_successful",
        "working_directory": str(repo),
        "command": [
            args.python,
            "-m",
            "experiments.mnist_conv",
            "lr-study",
            "--stage",
            args.stage,
            "--study",
            str(study),
            "--manifest",
            str(manifest_path),
            "--finalize-stage",
            "--data-root",
            str(data_root),
            "--device",
            "cpu",
        ],
    }
    metadata = {
        "study_id": manifest["study_id"],
        "study": str(study),
        "stage": args.stage,
        "stage_manifest": str(manifest_path),
        "pack_manifest": str(pack_path),
        "wave_count": len(pack["waves"]),
        "selected_concurrency": pack["execution_contract"][
            "selected_concurrency"
        ],
        "log_dir": str(log_dir),
        "required_login_audit": login_audit_commands(),
        "login_finalizer": login_finalizer,
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
    v5_parent = frozen_v5_parent_study()
    if args.stage == "audit" and not v5_parent.is_dir():
        raise _error("the frozen v5 parent study to exist", v5_parent)
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
    parser.add_argument("--stage", required=True, choices=SUPPORTED_STAGES)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--study")
    parser.add_argument("--results-root")
    parser.add_argument("--v5-study")
    parser.add_argument("--benchmark")
    parser.add_argument("--pack-manifest")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--module", default="pytorch-gpu/py3/2.5.0")
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
