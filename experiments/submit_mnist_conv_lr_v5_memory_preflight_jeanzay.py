"""Safely generate or submit the v5 memory-preflight job on Jean Zay."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from experiments.mnist_conv.io import read_json
from experiments.submit_mnist_conv_slurm import _export_arg, _submitted_job_id


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def _command(args: argparse.Namespace) -> list[str]:
    repo = Path(args.repo_root).expanduser().resolve()
    study = Path(args.study).expanduser().resolve()
    candidate_supplied = Path(args.candidate_manifest).expanduser().absolute()
    output_supplied = Path(args.output).expanduser().absolute()
    for label, path in (
        ("candidate-stage manifest", candidate_supplied),
        ("memory-preflight output", output_supplied),
    ):
        if path.is_symlink():
            raise _error(f"{label} not to be a symlink", path)
    candidate_manifest = candidate_supplied.resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output = output_supplied.resolve()
    if not repo.is_dir():
        raise _error("repo root to be an existing directory", repo)
    if not study.is_dir():
        raise _error("v5 study to be an existing directory", study)
    if not candidate_manifest.is_file():
        raise _error(
            "candidate-stage manifest to be an existing file", candidate_manifest
        )
    if not data_root.is_dir():
        raise _error("MNIST train-data root to be an existing directory", data_root)
    for label, path in (
        ("candidate-stage manifest", candidate_manifest),
        ("memory-preflight output", output),
    ):
        try:
            path.relative_to(study)
        except ValueError as exc:
            raise _error(f"{label} to be contained by the v5 study", path) from exc
    if output == candidate_manifest:
        raise _error(
            "memory-preflight output to differ from the candidate manifest", output
        )
    if type(args.steps_per_runtime) is not int or args.steps_per_runtime <= 0:
        raise _error("steps-per-runtime to be a positive integer", args.steps_per_runtime)

    stage = read_json(candidate_manifest)
    if not isinstance(stage, dict) or stage.get("stage_name") != "candidates":
        raise _error(
            "candidate manifest stage_name to be 'candidates'",
            None if not isinstance(stage, dict) else stage.get("stage_name"),
        )
    entries = stage.get("entries")
    if not isinstance(entries, list) or len(entries) != 36:
        raise _error("v5 candidate manifest to contain exactly 36 entries", entries)
    if [entry.get("entry_index") for entry in entries if isinstance(entry, dict)] != list(
        range(36)
    ):
        raise _error(
            "v5 candidate entry indices to be contiguous from zero",
            [
                None if not isinstance(entry, dict) else entry.get("entry_index")
                for entry in entries
            ],
        )

    wrapper = repo / "experiments/run_mnist_conv_lr_v5_memory_preflight_jeanzay.slurm"
    if not wrapper.is_file():
        raise _error("Jean Zay v5 memory-preflight wrapper to exist", wrapper)
    log_dir = study / "slurm" / "memory-preflight-v5"
    exports = {
        "MNIST_CONV_REPO_ROOT": str(repo),
        "MNIST_CONV_PYTHON": args.python,
        "MNIST_CONV_LR_STUDY": str(study),
        "MNIST_CONV_LR_MANIFEST": str(candidate_manifest),
        "MNIST_CONV_LR_MEMORY_PREFLIGHT": str(output),
        "MNIST_CONV_LR_MEMORY_PREFLIGHT_STEPS": str(args.steps_per_runtime),
        "MNIST_CONV_LR_STAGE": "candidates",
        "MNIST_CONV_DATASET_ROOT": str(data_root),
        "MNIST_CONV_DEVICE": "cuda",
        "MNIST_CONV_MODULE": args.module,
        "MNIST_CONV_IDRENV_PROJECT": "fmu",
        "MNIST_CONV_SLURM_ACCOUNT": "fmu@v100",
        "MNIST_CONV_SLURM_CONSTRAINT": "v100-32g",
        "MNIST_CONV_SLURM_GPUS": "1",
    }
    return [
        "sbatch",
        "--parsable",
        "--account=fmu@v100",
        "--partition=gpu_p13",
        "--qos=qos_gpu-t3",
        "--constraint=v100-32g",
        "--gres=gpu:1",
        f"--chdir={repo}",
        f"--output={log_dir}/%x-%j.out",
        f"--error={log_dir}/%x-%j.err",
        _export_arg(exports),
        str(wrapper),
    ]


def submit(args: argparse.Namespace) -> dict[str, Any]:
    command = _command(args)
    if not args.submit:
        return {"submitted": False, "command": command}
    log_dir = Path(args.study).expanduser().resolve() / "slurm/memory-preflight-v5"
    log_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return {"submitted": True, "job_id": _submitted_job_id(completed)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--module", default="")
    parser.add_argument("--steps-per-runtime", type=int, default=3)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit with sbatch; without this flag, print a dry-run command only.",
    )
    args = parser.parse_args()
    try:
        result = submit(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), flush=True)
        return 2
    print(json.dumps(result, indent=2 if not args.submit else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
