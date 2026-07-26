"""Login-node preparation and frozen Jean Zay R3 submission contracts for v6."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

from .identity import code_provenance
from .lr_study import create_study, load_study, publish_lr_stage_manifest
from .lr_study_spec import LRStudySpec
from .lr_v6_packing import (
    GPU_CONSTRAINT,
    IDRENV_PROJECT,
    PARTITION,
    QOS,
    R3_ALLOCATION_ID,
    R3_AUTHORIZATION_TOKEN,
    SLURM_ACCOUNT,
)


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def publish_v6_stage_from_login(
    *,
    stage: str,
    config_path: str | Path | None = None,
    results_root: str | Path | None = None,
    study_dir: str | Path | None = None,
) -> tuple[Path, Path, Mapping[str, Any]]:
    """Publish a v6 stage manifest without executing model/dataset work."""

    if config_path is not None:
        if study_dir is not None or results_root is None:
            raise _error(
                "--config with --results-root and without --study",
                {
                    "config": str(config_path),
                    "results_root": None if results_root is None else str(results_root),
                    "study": None if study_dir is None else str(study_dir),
                },
            )
        spec = LRStudySpec.from_path(config_path)
        resolved_results_root = Path(results_root).expanduser().resolve()
        if resolved_results_root.name != spec.study_id:
            raise _error(
                "v6 results-root to end with the content-addressed study_id wrapper",
                resolved_results_root,
            )
        root, _contract = create_study(spec, results_root)
    else:
        if study_dir is None or results_root is not None:
            raise _error(
                "--study without --config or --results-root",
                {
                    "config": config_path,
                    "results_root": None if results_root is None else str(results_root),
                    "study": None if study_dir is None else str(study_dir),
                },
            )
        root, spec = load_study(study_dir)
    if spec.data.get("schema_version") != "mnist-conv-lr-study/v6":
        raise _error("study schema_version to be mnist-conv-lr-study/v6", spec.data)
    expected_wrapper = root.parent.parent
    if expected_wrapper.name != spec.study_id:
        raise _error(
            "v6 study to be nested under <results>/{study_id}/lr_studies",
            root,
        )
    manifest, manifest_path = publish_lr_stage_manifest(
        root, spec, stage, code_provenance()
    )
    return root, manifest_path, manifest


def frozen_sbatch_resources() -> list[str]:
    return [
        f"--account={SLURM_ACCOUNT}",
        f"--partition={PARTITION}",
        f"--qos={QOS}",
        f"--constraint={GPU_CONSTRAINT}",
        "--gres=gpu:1",
        "--cpus-per-task=16",
        "--hint=nomultithread",
        "--time=20:00:00",
    ]


def frozen_exports() -> dict[str, str]:
    return {
        "MNIST_CONV_R3_ALLOCATION": R3_ALLOCATION_ID,
        "MNIST_CONV_V6_R3_AUTHORIZATION": R3_AUTHORIZATION_TOKEN,
        "MNIST_CONV_IDRENV_PROJECT": IDRENV_PROJECT,
        "MNIST_CONV_SLURM_ACCOUNT": SLURM_ACCOUNT,
        "MNIST_CONV_SLURM_PARTITION": PARTITION,
        "MNIST_CONV_SLURM_QOS": QOS,
        "MNIST_CONV_SLURM_CONSTRAINT": GPU_CONSTRAINT,
        "MNIST_CONV_SLURM_GPUS": "1",
    }


def login_audit_commands() -> list[list[str]]:
    user = os.environ.get("USER", "")
    return [
        ["/gpfslocalsup/bin/idrproj"],
        ["/gpfslocalsup/bin/idrenv", "-d", IDRENV_PROJECT],
        [
            "sacctmgr",
            "show",
            "assoc",
            "where",
            f"user={user}",
            "format=Account,Partition,QOS%50",
            "-n",
            "-P",
        ],
        ["scontrol", "show", "partition", PARTITION, "-o"],
        ["sinfo", "-h", "-p", PARTITION, "-o", "%P|%f"],
    ]


def verify_login_r3_allocation(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Verify allocation/project/account on a Jean Zay login node."""

    commands = login_audit_commands()
    results = [
        runner(command, check=True, text=True, capture_output=True)
        for command in commands
    ]
    idrproj_output = results[0].stdout
    project_lines = [
        line.strip()
        for line in idrproj_output.splitlines()
        if re.search(r"\bfmu\b", line, flags=re.IGNORECASE)
    ]
    matching_project_lines = [
        line for line in project_lines if R3_ALLOCATION_ID in line
    ]
    other_fmu_allocations = sorted(
        {
            allocation
            for line in project_lines
            for allocation in re.findall(r"AD[0-9]+R[0-9]+", line)
            if allocation != R3_ALLOCATION_ID
        }
    )
    if len(matching_project_lines) != 1 or other_fmu_allocations:
        raise RuntimeError(
            "Expected idrproj to map project 'fmu' on one line to exactly "
            f"{R3_ALLOCATION_ID!r}. Provided fmu lines: {project_lines!r}."
        )
    account_output = results[2].stdout
    association_lines = []
    for raw_line in account_output.splitlines():
        fields = [field.strip() for field in raw_line.split("|")]
        if len(fields) != 3 or fields[0] != SLURM_ACCOUNT:
            continue
        partition_field = fields[1]
        qos_values = {value.strip() for value in fields[2].split(",")}
        if partition_field not in {"", PARTITION} or QOS not in qos_values:
            continue
        association_lines.append(raw_line.strip())
    if len(association_lines) != 1:
        raise RuntimeError(
            "Expected one live Slurm association line to contain account "
            f"{SLURM_ACCOUNT!r}, QOS {QOS!r}, and either an unscoped or "
            f"{PARTITION!r} partition field. "
            f"Provided value: {account_output!r}."
        )
    partition_output = results[3].stdout
    expected_partition_tokens = {
        f"PartitionName={PARTITION}",
        "State=UP",
    }
    if not expected_partition_tokens.issubset(set(partition_output.split())):
        raise RuntimeError(
            f"Expected live partition {PARTITION!r} to exist in State=UP. "
            f"Provided value: {partition_output!r}."
        )
    feature_output = results[4].stdout
    feature_lines = []
    for raw_line in feature_output.splitlines():
        fields = [field.strip() for field in raw_line.split("|", maxsplit=1)]
        if len(fields) != 2 or fields[0].rstrip("*") != PARTITION:
            continue
        features = {value.strip().lower() for value in fields[1].split(",")}
        if {"v100", GPU_CONSTRAINT}.issubset(features):
            feature_lines.append(raw_line.strip())
    if not feature_lines:
        raise RuntimeError(
            f"Expected partition {PARTITION!r} to advertise both V100 and "
            f"{GPU_CONSTRAINT!r} nodes. Provided value: {feature_output!r}."
        )
    return {
        "allocation_id": R3_ALLOCATION_ID,
        "allocation_suffix": "R3",
        "idrenv_project": IDRENV_PROJECT,
        "slurm_account": SLURM_ACCOUNT,
        "partition": PARTITION,
        "qos": QOS,
        "constraint": GPU_CONSTRAINT,
        "commands": commands,
        "verified": True,
    }


__all__ = [
    "frozen_exports",
    "frozen_sbatch_resources",
    "login_audit_commands",
    "publish_v6_stage_from_login",
    "verify_login_r3_allocation",
]
