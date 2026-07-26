"""Frozen Jean Zay submission contracts for the Conv2 optimizer diagnostic."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping


ALLOCATION_ID = "AD010913993R3"
SLURM_ACCOUNT = "fmu@v100"
PARTITION = "gpu_p13"
QOS = "qos_gpu-t3"
CONSTRAINT = "v100-32g"
AUTHORIZATION = (
    "AD010913993R3:fmu@v100:gpu_p13:qos_gpu-t3:"
    "v100-32g:gpu1:conv2-optimizer-boundary-v1"
)
CPUS = 16
WALLTIME = "20:00:00"


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def frozen_sbatch_resources() -> list[str]:
    return [
        f"--account={SLURM_ACCOUNT}",
        f"--partition={PARTITION}",
        f"--qos={QOS}",
        f"--constraint={CONSTRAINT}",
        "--gres=gpu:1",
        f"--cpus-per-task={CPUS}",
        "--hint=nomultithread",
        f"--time={WALLTIME}",
    ]


def pack_exports(
    *,
    repo_root: str | Path,
    python: str | Path,
    study_config: str | Path,
    study_dir: str | Path,
    manifest: str | Path,
    benchmark: str | Path,
    reuse_root: str | Path,
    data_root: str | Path,
    scheme: str,
    stage: str,
    concurrency: int,
    log_dir: str | Path,
    output: str | Path,
    module: str = "pytorch-gpu/py3/2.5.0",
) -> dict[str, str]:
    if scheme not in {"baseline", "ours", "legacy"}:
        raise _error("scheme to be baseline, ours, or legacy", scheme)
    if stage not in {
        "main_grid",
        "upper_sentinel",
        "bias_capped_confirmation",
    }:
        raise _error("a supported optimizer-boundary stage", stage)
    if concurrency not in {1, 2, 4, 8, 12, 16}:
        raise _error("concurrency to be one of 1,2,4,8,12,16", concurrency)
    return {
        "MNIST_CONV_BOUNDARY_REPO_ROOT": str(Path(repo_root).expanduser().resolve()),
        "MNIST_CONV_BOUNDARY_PYTHON": str(Path(python).expanduser().resolve()),
        "MNIST_CONV_BOUNDARY_STUDY_CONFIG": str(
            Path(study_config).expanduser().resolve()
        ),
        "MNIST_CONV_BOUNDARY_STUDY_DIR": str(
            Path(study_dir).expanduser().resolve()
        ),
        "MNIST_CONV_BOUNDARY_MANIFEST": str(
            Path(manifest).expanduser().resolve()
        ),
        "MNIST_CONV_BOUNDARY_BENCHMARK": str(
            Path(benchmark).expanduser().resolve()
        ),
        "MNIST_CONV_BOUNDARY_REUSE_ROOT": str(
            Path(reuse_root).expanduser().resolve()
        ),
        "MNIST_CONV_BOUNDARY_DATA_ROOT": str(
            Path(data_root).expanduser().resolve()
        ),
        "MNIST_CONV_BOUNDARY_SCHEME": scheme,
        "MNIST_CONV_BOUNDARY_STAGE": stage,
        "MNIST_CONV_BOUNDARY_CONCURRENCY": str(concurrency),
        "MNIST_CONV_BOUNDARY_LOG_DIR": str(Path(log_dir).expanduser().resolve()),
        "MNIST_CONV_BOUNDARY_PACK_OUTPUT": str(
            Path(output).expanduser().resolve()
        ),
        "MNIST_CONV_BOUNDARY_MODULE": module,
        "MNIST_CONV_BOUNDARY_AUTHORIZATION": AUTHORIZATION,
    }


def benchmark_exports(
    *,
    repo_root: str | Path,
    python: str | Path,
    study_config: str | Path,
    study_dir: str | Path,
    manifest: str | Path,
    reuse_root: str | Path,
    data_root: str | Path,
    output: str | Path,
    module: str = "pytorch-gpu/py3/2.5.0",
) -> dict[str, str]:
    return {
        "MNIST_CONV_BOUNDARY_REPO_ROOT": str(
            Path(repo_root).expanduser().resolve()
        ),
        "MNIST_CONV_BOUNDARY_PYTHON": str(Path(python).expanduser().resolve()),
        "MNIST_CONV_BOUNDARY_STUDY_CONFIG": str(
            Path(study_config).expanduser().resolve()
        ),
        "MNIST_CONV_BOUNDARY_STUDY_DIR": str(
            Path(study_dir).expanduser().resolve()
        ),
        "MNIST_CONV_BOUNDARY_MANIFEST": str(
            Path(manifest).expanduser().resolve()
        ),
        "MNIST_CONV_BOUNDARY_REUSE_ROOT": str(
            Path(reuse_root).expanduser().resolve()
        ),
        "MNIST_CONV_BOUNDARY_DATA_ROOT": str(
            Path(data_root).expanduser().resolve()
        ),
        "MNIST_CONV_BOUNDARY_BENCHMARK_OUTPUT": str(
            Path(output).expanduser().resolve()
        ),
        "MNIST_CONV_BOUNDARY_MODULE": module,
        "MNIST_CONV_BOUNDARY_AUTHORIZATION": AUTHORIZATION,
    }


def sbatch_command(
    *,
    wrapper: str | Path,
    exports: Mapping[str, str],
    job_name: str,
    dependency: str | None = None,
) -> list[str]:
    if not job_name or any(character.isspace() for character in job_name):
        raise _error("a non-empty job name without whitespace", job_name)
    if any("," in key or "," in value for key, value in exports.items()):
        raise _error("export keys and values without commas", exports)
    command = [
        "sbatch",
        "--parsable",
        *frozen_sbatch_resources(),
        f"--job-name={job_name}",
        "--export=ALL,"
        + ",".join(f"{key}={value}" for key, value in sorted(exports.items())),
    ]
    if dependency is not None:
        if not dependency.isdigit():
            raise _error("a numeric Slurm dependency job id", dependency)
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(Path(wrapper).expanduser().resolve()))
    return command


def submit(
    *,
    wrapper: str | Path,
    exports: Mapping[str, str],
    job_name: str,
    dependency: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    command = sbatch_command(
        wrapper=wrapper,
        exports=exports,
        job_name=job_name,
        dependency=dependency,
    )
    completed = runner(command, check=True, text=True, capture_output=True)
    job_id = completed.stdout.strip().split(";", maxsplit=1)[0]
    if not job_id.isdigit():
        raise RuntimeError(
            "Expected sbatch --parsable to return a numeric job id. "
            f"Provided value: {completed.stdout!r}."
        )
    return job_id


__all__ = [
    "ALLOCATION_ID",
    "AUTHORIZATION",
    "CONSTRAINT",
    "CPUS",
    "PARTITION",
    "QOS",
    "SLURM_ACCOUNT",
    "WALLTIME",
    "benchmark_exports",
    "frozen_sbatch_resources",
    "pack_exports",
    "sbatch_command",
    "submit",
]
