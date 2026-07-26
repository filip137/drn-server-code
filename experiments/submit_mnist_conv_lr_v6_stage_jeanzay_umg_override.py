"""Submit an existing v6 training stage through the user-approved UMG V100 account.

This is an operational override only.  It preserves the content-addressed v6
study, manifests, benchmark, pack, and fmu result location while recording that
the actual Slurm allocation is AD011016471R1 / umg@v100.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

import experiments.submit_mnist_conv_lr_v6_stage_jeanzay as frozen_submit
from experiments.submit_mnist_conv_slurm import _submitted_job_id


UMG_ALLOCATION_ID = "AD011016471R1"
UMG_PROJECT = "umg"
UMG_ACCOUNT = "umg@v100"
PARTITION = "gpu_p13"
QOS = "qos_gpu-t3"
GPU_CONSTRAINT = "v100-32g"
OVERRIDE_MODE = "user-approved-umg-v100-20260721"
OVERRIDE_AUTHORIZATION = (
    "AD011016471R1:umg@v100:gpu_p13:qos_gpu-t3:"
    "v100-32g:gpu1:user-approved-20260721"
)
SUPPORTED_STAGES = ("baseline_candidates", "confirmations")


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def umg_login_audit_commands() -> list[list[str]]:
    user = os.environ.get("USER", "")
    return [
        ["/gpfslocalsup/bin/idrproj"],
        ["/gpfslocalsup/bin/idrenv", "-d", UMG_PROJECT],
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


def verify_login_umg_v100_allocation(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    commands = umg_login_audit_commands()
    results = [
        runner(command, check=True, text=True, capture_output=True)
        for command in commands
    ]
    project_lines = [
        line.strip()
        for line in results[0].stdout.splitlines()
        if re.search(r"\bumg\b", line, flags=re.IGNORECASE)
    ]
    matching = [line for line in project_lines if UMG_ALLOCATION_ID in line]
    if len(matching) != 1:
        raise RuntimeError(
            "Expected idrproj to map project 'umg' on one line to exactly "
            f"{UMG_ALLOCATION_ID!r}. Provided umg lines: {project_lines!r}."
        )
    if UMG_PROJECT not in results[1].stdout.lower():
        raise RuntimeError(
            "Expected idrenv -d umg to initialize project umg. "
            f"Provided value: {results[1].stdout!r}."
        )
    association_lines = []
    for raw_line in results[2].stdout.splitlines():
        fields = [field.strip() for field in raw_line.split("|")]
        if len(fields) != 3 or fields[0] != UMG_ACCOUNT:
            continue
        qos_values = {value.strip() for value in fields[2].split(",")}
        if fields[1] in {"", PARTITION} and QOS in qos_values:
            association_lines.append(raw_line.strip())
    if len(association_lines) != 1:
        raise RuntimeError(
            "Expected one live UMG V100 association with qos_gpu-t3. "
            f"Provided value: {results[2].stdout!r}."
        )
    partition_tokens = set(results[3].stdout.split())
    if not {f"PartitionName={PARTITION}", "State=UP"}.issubset(
        partition_tokens
    ):
        raise RuntimeError(
            f"Expected live partition {PARTITION!r} in State=UP. "
            f"Provided value: {results[3].stdout!r}."
        )
    feature_lines = []
    for raw_line in results[4].stdout.splitlines():
        fields = [field.strip() for field in raw_line.split("|", maxsplit=1)]
        if len(fields) != 2 or fields[0].rstrip("*") != PARTITION:
            continue
        features = {value.strip().lower() for value in fields[1].split(",")}
        if {"v100", GPU_CONSTRAINT}.issubset(features):
            feature_lines.append(raw_line.strip())
    if not feature_lines:
        raise RuntimeError(
            f"Expected {PARTITION!r} to advertise V100 and {GPU_CONSTRAINT!r}. "
            f"Provided value: {results[4].stdout!r}."
        )
    return {
        "allocation_id": UMG_ALLOCATION_ID,
        "idrenv_project": UMG_PROJECT,
        "slurm_account": UMG_ACCOUNT,
        "partition": PARTITION,
        "qos": QOS,
        "constraint": GPU_CONSTRAINT,
        "commands": commands,
        "user_authorized_override": True,
        "verified": True,
    }


def _append_exports(command: list[str], values: dict[str, str]) -> None:
    export_indices = [
        index for index, item in enumerate(command) if item.startswith("--export=ALL,")
    ]
    if len(export_indices) != 1:
        raise _error("one Slurm --export argument", export_indices)
    for key, value in values.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise _error("override export keys to use shell-variable syntax", key)
        if any(character in value for character in ",\r\n"):
            raise _error("override export values not to contain commas or newlines", value)
    suffix = ",".join(f"{key}={value}" for key, value in values.items())
    command[export_indices[0]] += "," + suffix


def _prepare(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    command, metadata = frozen_submit._prepare(args)
    command = list(command)
    try:
        account_index = command.index("--account=fmu@v100")
    except ValueError as exc:
        raise _error("the frozen submitter command to contain fmu@v100", command) from exc
    command[account_index] = f"--account={UMG_ACCOUNT}"
    repo = Path(args.repo_root).expanduser().resolve()
    wrapper = repo / "experiments/run_mnist_conv_lr_v6_pack_jeanzay.slurm"
    if command[-1] != str(wrapper):
        raise _error("the frozen v6 pack wrapper to be the final command item", command[-1])
    _append_exports(
        command,
        {
            "MNIST_CONV_EXECUTION_OVERRIDE_MODE": OVERRIDE_MODE,
            "MNIST_CONV_EXECUTION_OVERRIDE_AUTHORIZATION": OVERRIDE_AUTHORIZATION,
            "MNIST_CONV_EXECUTION_ALLOCATION_OVERRIDE": UMG_ALLOCATION_ID,
            "MNIST_CONV_EXECUTION_PROJECT_OVERRIDE": UMG_PROJECT,
            "MNIST_CONV_EXECUTION_ACCOUNT_OVERRIDE": UMG_ACCOUNT,
        },
    )
    override = {
        "schema_version": "mnist-conv-lr-execution-override/v1",
        "user_authorized": True,
        "authorization_date": "2026-07-21",
        "frozen_study_execution": {
            "allocation": "AD010913993R3",
            "project": "fmu",
            "account": "fmu@v100",
        },
        "actual_execution": {
            "allocation": UMG_ALLOCATION_ID,
            "project": UMG_PROJECT,
            "account": UMG_ACCOUNT,
            "partition": PARTITION,
            "qos": QOS,
            "constraint": GPU_CONSTRAINT,
            "gpus": 1,
        },
    }
    return command, {
        **metadata,
        "required_login_audit": umg_login_audit_commands(),
        "execution_override": override,
    }


def submit(args: argparse.Namespace) -> dict[str, Any]:
    command, metadata = _prepare(args)
    if not args.submit:
        return {
            "status": "dry_run",
            "submitted": False,
            "worker": command,
            **metadata,
        }
    audit = verify_login_umg_v100_allocation()
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
    parser.add_argument("--study", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--pack-manifest")
    parser.add_argument(
        "--repo-root", default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument("--python", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--module", default="pytorch-gpu/py3/2.5.0")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    args.config = None
    args.results_root = None
    args.v5_study = None
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
