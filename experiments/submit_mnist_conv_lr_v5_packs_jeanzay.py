"""Submit immutable v5 training packs from a Jean Zay login node.

Only the GPU training array is submitted to Slurm.  The returned login-node
finalization command is intentionally not submitted as a CPU job: the v5 study
contract reserves Slurm for training and performs collection on a login node.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from experiments.mnist_conv.identity import sha256_file
from experiments.mnist_conv.io import read_json
from experiments.submit_mnist_conv_slurm import _export_arg, _submitted_job_id


def _submission_error(exc: BaseException) -> dict[str, Any]:
    result: dict[str, Any] = {
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    if isinstance(exc, subprocess.CalledProcessError):
        result["returncode"] = exc.returncode
        stdout = exc.stdout if exc.stdout is not None else exc.output
        if stdout:
            result["stdout"] = str(stdout).strip()
        if exc.stderr:
            result["stderr"] = str(exc.stderr).strip()
    return result


def _run_sbatch(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def _commands(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    repo = Path(args.repo_root).expanduser().resolve()
    study = Path(args.study).expanduser().resolve()
    candidate_manifest = Path(args.candidate_manifest).expanduser().resolve()
    pack_manifest = Path(args.pack_manifest).expanduser().resolve()
    pack = read_json(pack_manifest)
    stage = read_json(candidate_manifest)
    if pack.get("schema_version") != "mnist-conv-lr-pack-manifest/v1":
        raise ValueError(
            "Expected pack manifest schema_version to be "
            f"'mnist-conv-lr-pack-manifest/v1'. Provided value: {pack.get('schema_version')!r}."
        )
    if pack.get("stage_manifest_sha256") != sha256_file(candidate_manifest):
        raise ValueError(
            "Expected pack manifest to bind the candidate-stage manifest. "
            f"Provided value: {pack.get('stage_manifest_sha256')!r}."
        )
    if pack.get("study_id") != stage.get("study_id"):
        raise ValueError(
            "Expected pack and candidate manifests to have the same study_id. "
            f"Provided value: pack={pack.get('study_id')!r}, stage={stage.get('study_id')!r}."
        )
    packs = pack.get("packs")
    if not isinstance(packs, list) or not packs:
        raise ValueError(f"Expected a non-empty pack list. Provided value: {packs!r}.")
    worker_script = repo / "experiments/run_mnist_conv_lr_v5_pack_jeanzay.slurm"
    if not worker_script.is_file():
        raise FileNotFoundError(
            f"Expected Jean Zay v5 wrapper to exist. Provided value: {worker_script}."
        )
    log_dir = study / "slurm" / "candidates-v5-packs"
    exports = {
        "MNIST_CONV_REPO_ROOT": str(repo),
        "MNIST_CONV_PYTHON": args.python,
        "MNIST_CONV_LR_STUDY": str(study),
        "MNIST_CONV_LR_MANIFEST": str(candidate_manifest),
        "MNIST_CONV_LR_PACK_MANIFEST": str(pack_manifest),
        "MNIST_CONV_LR_STAGE": "candidates",
        "MNIST_CONV_DATASET_ROOT": str(Path(args.data_root).expanduser().resolve()),
        "MNIST_CONV_DEVICE": "cuda",
        "MNIST_CONV_MODULE": args.module,
        "MNIST_CONV_IDRENV_PROJECT": "fmu",
        "MNIST_CONV_SLURM_ACCOUNT": "fmu@v100",
        "MNIST_CONV_SLURM_CONSTRAINT": "v100-32g",
        "MNIST_CONV_SLURM_GPUS": "1",
    }
    worker = [
        "sbatch",
        "--parsable",
        "--account=fmu@v100",
        "--partition=gpu_p13",
        "--qos=qos_gpu-t3",
        "--constraint=v100-32g",
        "--gres=gpu:1",
        f"--array=0-{len(packs) - 1}%{args.max_concurrent_packs}",
        f"--chdir={repo}",
        f"--output={log_dir}/%x-%A_%a.out",
        f"--error={log_dir}/%x-%A_%a.err",
        _export_arg(exports),
        str(worker_script),
    ]
    login_finalizer = {
        "execution_host": "jean_zay_login_node",
        "run_after": "worker_array_terminal",
        "idrenv_project": "fmu",
        "module": args.module,
        "working_directory": str(repo),
        "command": [
            args.python,
            "-m",
            "experiments.mnist_conv",
            "lr-study",
            "--stage",
            "candidates",
            "--study",
            str(study),
            "--manifest",
            str(candidate_manifest),
            "--finalize-stage",
            "--data-root",
            str(Path(args.data_root).expanduser().resolve()),
            "--device",
            "cpu",
        ],
    }
    return worker, login_finalizer


def submit(args: argparse.Namespace) -> dict[str, Any]:
    worker, login_finalizer = _commands(args)
    if args.dry_run:
        return {
            "mode": "training_array_with_login_collection",
            "status": "dry_run",
            "submitted": False,
            "worker": worker,
            "login_finalizer": login_finalizer,
        }
    log_dir = Path(args.study).expanduser().resolve() / "slurm/candidates-v5-packs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        worker_result = _run_sbatch(worker)
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "mode": "training_array_with_login_collection",
            "status": "failed",
            "submitted": False,
            "worker_submitted": False,
            "worker_error": _submission_error(exc),
            "login_finalizer": login_finalizer,
        }
    try:
        worker_id = _submitted_job_id(worker_result)
    except RuntimeError as exc:
        return {
            "mode": "training_array_with_login_collection",
            "status": "submission_unconfirmed",
            "submitted": False,
            "worker_submitted": None,
            "worker_error": _submission_error(exc),
            "login_finalizer": login_finalizer,
        }
    return {
        "mode": "training_array_with_login_collection",
        "status": "complete",
        "submitted": True,
        "worker_submitted": True,
        "worker_job_id": worker_id,
        "login_finalizer": login_finalizer,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--pack-manifest", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--module", default="")
    parser.add_argument("--max-concurrent-packs", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_concurrent_packs < 1:
        parser.error("--max-concurrent-packs must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = submit(args)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), flush=True)
        return 2
    print(json.dumps(result, indent=2 if args.dry_run else None, sort_keys=True))
    if result.get("status") in {"failed", "submission_unconfirmed"}:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
