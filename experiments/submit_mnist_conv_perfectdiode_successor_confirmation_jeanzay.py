#!/usr/bin/env python3
"""Prepare or submit guarded perfect-diode successor arrays on Jean Zay.

The default action is a read-only dry run.  ``--test-only`` performs the live
UMG allocation audit and asks Slurm to validate the request without creating a
job.  ``--submit`` is the only mode that can create a job.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mnist_conv.io import atomic_write_json, read_json


ENVIRONMENT_SCHEMA_VERSION = (
    "perfectdiode-successor-jeanzay-environment/v1"
)
ALLOCATION_ID = "AD011016471R1"
PROJECT = "umg"
ACCOUNT = "umg@v100"
PARTITION = "gpu_p13"
QOS = "qos_gpu-t3"
CONSTRAINT = "v100-32g"
MODULE = "pytorch-gpu/py3/2.5.0"
PYTHON_EXECUTABLE = (
    "/lustre/fshomisc/sup/hpe/pub/miniforge/24.9.0/envs/"
    "pytorch-gpu-2.5.0+py3.12.7/bin/python"
)
CPUS_PER_TASK = 16
GPUS_PER_TASK = 1
HOST_MEMORY_POLICY = "jean_zay_site_managed_no_explicit_slurm_request"
CANARY_ARRAY = "0-0"
PRODUCTION_ARRAY = "0-5%6"
CANARY_TASK_COUNT = 1
PRODUCTION_TASK_COUNT = 6
LOGICAL_ENTRY_COUNT = 12
RUNS_PER_GPU = 2
DEFAULT_CANARY_PACK_INDEX = 4
DEFAULT_CANARY_ENTRY_INDICES = (8, 9)
FIXED_ENTRY_PACKS = tuple(
    (index, index + 1)
    for index in range(0, LOGICAL_ENTRY_COUNT, RUNS_PER_GPU)
)
ALLOCATION_AUTHORIZATION = (
    "AD011016471R1:umg@v100:gpu_p13:qos_gpu-t3:"
    "v100-32g:gpu1:cpu16:hostmem-site-managed:nomultithread:"
    "runs2:concurrent-processes:"
    "allocation-user-approved-20260727"
)
WRAPPER_NAME = (
    "run_mnist_conv_perfectdiode_successor_confirmation_jeanzay.slurm"
)
RUNTIME_CLI_NAME = (
    "run_mnist_conv_perfectdiode_successor_confirmation.py"
)
SUBMITTER_NAME = (
    "submit_mnist_conv_perfectdiode_successor_confirmation_jeanzay.py"
)
SUPERVISOR_NAME = (
    "supervise_mnist_conv_perfectdiode_successor_confirmation_jeanzay.py"
)
SEMANTIC_CANARY_VERIFIER_NAME = (
    "verify_mnist_conv_perfectdiode_successor_canary.py"
)
SCHEDULED_PREFLIGHT_RELATIVE = Path(
    "skills/scheduled-run-preflight/scripts/preflight_scheduled_runner.py"
)
EXPERIMENT_PLAN_VALIDATOR_RELATIVE = Path(
    "skills/run-experiment-pipeline/scripts/validate_experiment_plan.py"
)
PREFLIGHT_PATH_PLACEHOLDER = (
    "__PD_SUCCESSOR_PREFLIGHT_RECEIPT_PATH__"
)
PREFLIGHT_SHA256_PLACEHOLDER = (
    "__PD_SUCCESSOR_PREFLIGHT_RECEIPT_SHA256__"
)
LAUNCH_AUTHORIZATION_SHA256_PLACEHOLDER = (
    "__PD_SUCCESSOR_LAUNCH_AUTHORIZATION_SHA256__"
)
TRACKER_GATE_RECEIPT_PATH_PLACEHOLDER = (
    "__PD_SUCCESSOR_TRACKER_GATE_RECEIPT_PATH__"
)
TRACKER_GATE_RECEIPT_SHA256_PLACEHOLDER = (
    "__PD_SUCCESSOR_TRACKER_GATE_RECEIPT_SHA256__"
)
CANARY_RECEIPT_PATH_PLACEHOLDER = (
    "__PD_SUCCESSOR_CANARY_RECEIPT_PATH__"
)
CANARY_RECEIPT_SHA256_PLACEHOLDER = (
    "__PD_SUCCESSOR_CANARY_RECEIPT_SHA256__"
)
RESULT_JSON_MARKER = "PD_SUCCESSOR_RESULT_JSON="
DEFAULT_COMMAND_TIMEOUT_SECONDS = 300.0


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def _export_arg(values: Mapping[str, str]) -> str:
    for key, value in values.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise _error(
                "Slurm export keys to use uppercase shell-variable syntax",
                key,
            )
        if any(character in value for character in ",\r\n"):
            raise _error(
                "Slurm export values not to contain commas/newlines",
                {key: value},
            )
    return "--export=ALL," + ",".join(
        f"{key}={value}" for key, value in values.items()
    )


def _submitted_job_id(
    completed: subprocess.CompletedProcess[str],
) -> str:
    value = completed.stdout.strip().split(";", maxsplit=1)[0]
    if not re.fullmatch(r"[0-9]+", value):
        raise RuntimeError(
            "Expected sbatch --parsable to return a numeric job id. "
            f"Provided value: {completed.stdout!r}."
        )
    return value


def umg_login_audit_commands() -> list[list[str]]:
    return [
        ["/gpfslocalsup/bin/idrproj"],
        ["/gpfslocalsup/bin/idrenv", "-d", PROJECT],
        [
            "sacctmgr",
            "show",
            "assoc",
            "where",
            f"user={os.environ.get('USER', '')}",
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
    matching = [line for line in project_lines if ALLOCATION_ID in line]
    if len(matching) != 1:
        raise RuntimeError(
            "Expected /gpfslocalsup/bin/idrproj to map project 'umg' on "
            f"one line to {ALLOCATION_ID!r}. Provided value: "
            f"{project_lines!r}."
        )
    if PROJECT not in results[1].stdout.lower():
        raise RuntimeError(
            "Expected /gpfslocalsup/bin/idrenv -d umg to initialize umg. "
            f"Provided value: {results[1].stdout!r}."
        )
    association_lines = []
    for raw_line in results[2].stdout.splitlines():
        fields = [field.strip() for field in raw_line.split("|")]
        if len(fields) != 3 or fields[0] != ACCOUNT:
            continue
        qos_values = {item.strip() for item in fields[2].split(",")}
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
        features = {item.strip().lower() for item in fields[1].split(",")}
        if {"v100", CONSTRAINT}.issubset(features):
            feature_lines.append(raw_line.strip())
    if not feature_lines:
        raise RuntimeError(
            f"Expected {PARTITION!r} to advertise V100 and {CONSTRAINT!r}. "
            f"Provided value: {results[4].stdout!r}."
        )
    return {
        "allocation_id": ALLOCATION_ID,
        "dossier_mapping": matching[0],
        "idrenv_project": PROJECT,
        "slurm_account": ACCOUNT,
        "partition": PARTITION,
        "qos": QOS,
        "constraint": CONSTRAINT,
        "commands": commands,
        "user_authorized_override": True,
        "verified": True,
    }


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory_tree(path: str | Path) -> str:
    """Hash every regular file and relative path in an immutable bundle."""

    root = Path(path).resolve()
    if not root.is_dir():
        raise _error("bundle tree root to be an existing directory", root)
    digest = hashlib.sha256()
    files: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise _error(
                "the immutable bundle tree not to contain symlinks",
                candidate,
            )
        if candidate.is_file():
            files.append(candidate)
        elif not candidate.is_dir():
            raise _error(
                "the immutable bundle tree to contain only files/directories",
                candidate,
            )
    for candidate in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        file_digest = bytes.fromhex(sha256_file(candidate))
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(candidate.stat().st_size.to_bytes(8, "big"))
        digest.update(file_digest)
    return digest.hexdigest()


def validate_source_archive(
    archive: str | Path,
    *,
    staged_files: Mapping[str, Path],
) -> dict[str, str]:
    """Reject arbitrary/unsafe archives and bind staged executable sources."""

    source = Path(archive).resolve()
    if not (
        source.name.endswith(".tar.gz") or source.name.endswith(".tgz")
    ):
        raise _error(
            "source archive filename to end in .tar.gz or .tgz",
            source.name,
        )
    try:
        handle = tarfile.open(source, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise _error(
            "source archive to be a readable gzip tar archive", str(exc)
        ) from exc
    observed: dict[str, str] = {}
    with handle:
        members = handle.getmembers()
        for member in members:
            member_path = Path(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise _error(
                    "source archive to contain only safe regular files/directories",
                    member.name,
                )
            normalized = member_path.as_posix().lstrip("./")
            if (
                normalized.endswith(
                    "docs/experiment_plans/"
                    "perfectdiode-conv12-best-observed-confirmation-20260727-v1.md"
                )
                or "launch-authorization" in normalized
            ):
                raise RuntimeError(
                    "Expected source archive to exclude the final plan and "
                    f"launch authorization to avoid a hash cycle. "
                    f"Provided member: {member.name!r}."
                )
        for relative, staged in staged_files.items():
            candidates = [
                member
                for member in members
                if member.isfile()
                and (
                    Path(member.name).as_posix().lstrip("./") == relative
                    or Path(member.name)
                    .as_posix()
                    .lstrip("./")
                    .endswith("/" + relative)
                )
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    "Expected source archive to contain exactly one staged "
                    f"{relative!r}. Provided matches: "
                    f"{[item.name for item in candidates]!r}."
                )
            stream = handle.extractfile(candidates[0])
            if stream is None:
                raise RuntimeError(
                    f"Expected source member {candidates[0].name!r} to be readable."
                )
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            archive_hash = digest.hexdigest()
            staged_hash = sha256_file(staged)
            if archive_hash != staged_hash:
                raise RuntimeError(
                    "Expected staged executable source to match its immutable "
                    f"archive member. Provided value: relative={relative!r}, "
                    f"archive={archive_hash!r}, staged={staged_hash!r}."
                )
            observed[relative] = archive_hash
    return observed


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _error(f"{label} to be readable valid JSON", str(exc)) from exc
    if not isinstance(value, dict):
        raise _error(f"{label} to be a JSON object", value)
    return value


def load_environment_contract(path: str | Path) -> dict[str, Any]:
    """Load the immutable UMG V100 contract with exact-value validation."""

    source = Path(path).expanduser().resolve()
    value = _read_json_object(source, "successor environment contract")
    expected_top = {
        "schema_version",
        "allocation",
        "resources",
        "arrays",
        "storage",
        "verification",
        "walltime",
    }
    if set(value) != expected_top:
        raise _error(
            f"environment contract keys to be exactly {sorted(expected_top)!r}",
            sorted(value),
        )
    if value["schema_version"] != ENVIRONMENT_SCHEMA_VERSION:
        raise _error(
            f"environment schema_version to be {ENVIRONMENT_SCHEMA_VERSION!r}",
            value["schema_version"],
        )
    expected_allocation = {
        "dossier": ALLOCATION_ID,
        "project": PROJECT,
        "account": ACCOUNT,
        "partition": PARTITION,
        "qos": QOS,
        "constraint": CONSTRAINT,
    }
    if value["allocation"] != expected_allocation:
        raise _error(
            "the exact user-approved AD011016471R1 UMG V100 allocation",
            value["allocation"],
        )
    resources = value["resources"]
    if not isinstance(resources, dict):
        raise _error(
            "environment resources to be a JSON object", resources
        )
    python_executable = resources.get("python_executable")
    if (
        not isinstance(python_executable, str)
        or not python_executable
        or not Path(python_executable).is_absolute()
    ):
        raise _error(
            "resources.python_executable to be an absolute path",
            python_executable,
        )
    expected_resources = {
        "nodes": 1,
        "tasks": 1,
        "gpus_per_task": GPUS_PER_TASK,
        "cpus_per_task": CPUS_PER_TASK,
        "host_memory_policy": HOST_MEMORY_POLICY,
        "hint": "nomultithread",
        "module": MODULE,
        "python_executable": python_executable,
        "device": "cuda",
        "thread_limits": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    }
    if value["resources"] != expected_resources:
        raise _error(
            "the exact one-V100, 16-physical-CPU runtime resources",
            value["resources"],
        )
    expected_arrays = {
        "canary": CANARY_ARRAY,
        "canary_task_count": CANARY_TASK_COUNT,
        "canary_pack_index": DEFAULT_CANARY_PACK_INDEX,
        "canary_entry_indices": list(DEFAULT_CANARY_ENTRY_INDICES),
        "production": PRODUCTION_ARRAY,
        "production_task_count": PRODUCTION_TASK_COUNT,
        "logical_entry_count": LOGICAL_ENTRY_COUNT,
        "runs_per_gpu": RUNS_PER_GPU,
        "concurrent_within_pack": True,
        "fixed_entry_packs": [
            list(entry_indices) for entry_indices in FIXED_ENTRY_PACKS
        ],
    }
    if value["arrays"] != expected_arrays:
        raise _error(
            "the exact paired canary pack 4 and production 0-5%6 arrays",
            value["arrays"],
        )
    expected_storage = {
        "source_root_template": (
            "/lustre/fswork/projects/rech/umg/{user}/server_code"
        ),
        "data_root_template": (
            "/lustre/fsn1/projects/rech/umg/{user}/datasets/mnist"
        ),
        "result_root_template": (
            "/lustre/fsn1/projects/rech/umg/{user}/server_code/results"
        ),
    }
    if value["storage"] != expected_storage:
        raise _error(
            "the frozen UMG source, MNIST-data, and result root templates",
            value["storage"],
        )
    expected_verification = {
        "official_canary_verifier_path_template": (
            "/lustre/fswork/projects/rech/umg/{user}/server_code/"
            "launch_tools/verify_canary.py"
        ),
        "official_canary_verifier_sha256": (
            "032979ace766382941a769db205df1c7fd2576e846e59873645cfe7ad894b5b0"
        ),
    }
    if value["verification"] != expected_verification:
        raise _error(
            "the exact staged official canary verifier path/hash",
            value["verification"],
        )
    expected_walltime = {
        "canary": "02:00:00",
        "production": "08:00:00",
    }
    if value["walltime"] != expected_walltime:
        raise _error(
            "canary walltime 02:00:00 and production walltime 08:00:00",
            value["walltime"],
        )
    return value


def expected_official_canary_verifier_path(
    contract: Mapping[str, Any],
    *,
    remote_user: str,
) -> Path:
    """Resolve the one verifier location authorized by the environment."""

    user = remote_user.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", user):
        raise _error("remote-user to be a safe non-empty login name", user)
    try:
        template = contract["verification"][
            "official_canary_verifier_path_template"
        ]
    except (KeyError, TypeError) as exc:
        raise _error(
            "environment verification to define "
            "official_canary_verifier_path_template",
            contract.get("verification")
            if isinstance(contract, Mapping)
            else contract,
        ) from exc
    if not isinstance(template, str) or "{user}" not in template:
        raise _error(
            "official canary verifier path template to contain '{user}'",
            template,
        )
    resolved = Path(template.format(user=user))
    if not resolved.is_absolute():
        raise _error(
            "official canary verifier path template to resolve absolutely",
            resolved,
        )
    return resolved


def _resolve_existing(
    raw_path: str | Path,
    *,
    label: str,
    kind: str,
) -> Path:
    supplied = Path(raw_path).expanduser().absolute()
    if supplied.is_symlink():
        raise _error(f"{label} not to be a symlink", supplied)
    resolved = supplied.resolve()
    if kind == "file" and not resolved.is_file():
        raise _error(f"{label} to be an existing file", resolved)
    if kind == "directory" and not resolved.is_dir():
        raise _error(f"{label} to be an existing directory", resolved)
    return resolved


def _require_relative(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise _error(f"{label} to be under {root}", path) from exc


def _optional_absolute_path(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise _error("an optional path to be a non-empty path string", value)
    return str(Path(value).expanduser().absolute())


def _parse_result_marker(stdout: str, *, label: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    marked = [
        line[len(RESULT_JSON_MARKER) :]
        for line in lines
        if line.startswith(RESULT_JSON_MARKER)
    ]
    if (
        len(marked) != 1
        or not lines
        or not lines[-1].startswith(RESULT_JSON_MARKER)
    ):
        raise RuntimeError(
            f"Expected {label} to contain exactly one terminal "
            f"{RESULT_JSON_MARKER!r} marker. Provided value: {stdout!r}."
        )
    try:
        value = json.loads(marked[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Expected the terminal {label} marker to contain one JSON "
            f"object. Provided value: {marked[0]!r}."
        ) from exc
    if not isinstance(value, dict):
        raise _error(f"{label} output to be a JSON object", value)
    return value


def _parse_runtime_preflight(stdout: str) -> dict[str, Any]:
    value = _parse_result_marker(stdout, label="successor runtime preflight")
    if (
        value.get("schema_version")
        != "mnist-conv-perfectdiode-successor-preflight/v1"
    ):
        raise _error(
            "runtime preflight schema_version to be "
            "'mnist-conv-perfectdiode-successor-preflight/v1'",
            value.get("schema_version"),
        )
    status = value.get("status")
    if status not in {"complete", "passed", "ready"}:
        raise RuntimeError(
            "Expected the successor runtime preflight status to be complete, "
            f"passed, or ready. Provided value: {status!r}."
        )
    if value.get("passed") is not True or value.get(
        "official_test_read"
    ) is not False:
        raise RuntimeError(
            "Expected runtime preflight to pass while prohibiting official "
            f"test reads. Provided value: passed={value.get('passed')!r}, "
            f"official_test_read={value.get('official_test_read')!r}."
        )
    checks = value.get("checks")
    if not isinstance(checks, dict):
        raise _error("runtime preflight checks to be a JSON object", checks)
    required_true = (
        "approved_plan",
        "launch_authorized",
        "selection_override_approved",
        "config_valid",
        "manifest_valid",
        "official_test_prohibited",
        "expected_entry_count",
        "source_archive_valid",
        "environment_contract_valid",
        "worker_launcher_valid",
        "runtime_module_valid",
        "implementation_ready",
    )
    failures = [
        field for field in required_true if checks.get(field) is not True
    ]
    if value.get("expected_entry_count") != LOGICAL_ENTRY_COUNT:
        failures.append("expected_entry_count")
    for field in (
        "bundle_manifest_sha256",
        "config_sha256",
        "config_file_sha256",
    ):
        field_value = value.get(field)
        if (
            not isinstance(field_value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", field_value)
        ):
            failures.append(field)
    if not isinstance(value.get("bundle_id"), str) or not value["bundle_id"]:
        failures.append("bundle_id")
    if failures:
        raise RuntimeError(
            "Expected runtime preflight to prove approved-plan, selection "
            "override, immutable config/manifest, official-test prohibition, "
            "and the exact 12-entry launch. "
            f"Provided failing fields: {sorted(set(failures))!r}; "
            f"preflight={value!r}."
        )
    return value


def _validate_live_allocation(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "verified": True,
        "allocation_id": ALLOCATION_ID,
        "idrenv_project": PROJECT,
        "slurm_account": ACCOUNT,
        "partition": PARTITION,
        "qos": QOS,
        "constraint": CONSTRAINT,
    }
    mismatches = {
        key: {"expected": expected_value, "provided": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            "Expected the exact live AD011016471R1 UMG V100 allocation. "
            f"Provided mismatches: {mismatches!r}."
        )
    return dict(value)


def _scontrol_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", maxsplit=1)
        if key in fields:
            raise RuntimeError(
                "Expected unique fields in scontrol job output. "
                f"Provided duplicate: {key!r}."
            )
        fields[key] = value
    return fields


def _array_task_spec(value: str) -> tuple[set[int], int | None]:
    indices: set[int] = set()
    throttles: set[int] = set()
    for part in value.split(","):
        match = re.fullmatch(
            r"([0-9]+)(?:-([0-9]+)(?::([0-9]+))?)?(?:%([0-9]+))?",
            part,
        )
        if match is None:
            raise _error(
                "scontrol ArrayTaskId to be a decimal/range specification",
                value,
            )
        start = int(match.group(1))
        end = int(match.group(2) or start)
        step = int(match.group(3) or 1)
        if end < start or step < 1:
            raise _error("a valid ascending ArrayTaskId range", part)
        indices.update(range(start, end + 1, step))
        if match.group(4) is not None:
            throttles.add(int(match.group(4)))
    if len(throttles) > 1:
        raise _error(
            "one consistent scontrol array throttle", sorted(throttles)
        )
    return indices, next(iter(throttles), None)


def verify_submitted_job_contract(
    *,
    job_id: str,
    kind: str,
    metadata: Mapping[str, Any],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Immediately re-read the accepted array and fail on scheduler drift."""

    if not job_id.isdigit():
        raise _error("submitted job id to be decimal", job_id)
    if kind not in {"canary", "production"}:
        raise _error("submitted job kind to be canary or production", kind)
    resources = metadata.get("resources")
    if not isinstance(resources, Mapping):
        raise _error(
            "submitted-job metadata.resources to be a JSON object",
            resources,
        )
    expected_repo = metadata.get("repo_root")
    expected_wrapper = metadata.get("wrapper")
    expected_output = metadata.get("output_root")
    for label, value in (
        ("repo_root", expected_repo),
        ("wrapper", expected_wrapper),
        ("output_root", expected_output),
    ):
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise _error(
                f"submitted-job metadata.{label} to be an absolute path",
                value,
            )
    expected_time = "02:00:00" if kind == "canary" else "08:00:00"
    expected_job_name = f"pd-successor-{kind}"
    expected_log_dir = Path(expected_output) / "slurm"

    def log_path_matches(value: str, *, suffix: str) -> bool:
        raw = expected_log_dir / f"%x-%A_%a.{suffix}"
        if value == str(raw):
            return True
        expanded = re.fullmatch(
            re.escape(
                str(
                    expected_log_dir
                    / f"{expected_job_name}-{job_id}_"
                )
            )
            + r"(?:%a|4294967294|[0-9]+)"
            + re.escape(f".{suffix}"),
            value,
        )
        if expanded is None:
            return False
        task_token = value.rsplit("_", maxsplit=1)[-1].split(".", maxsplit=1)[0]
        if task_token in {"%a", "4294967294"}:
            return True
        task_index = int(task_token)
        return (
            task_index == 0
            if kind == "canary"
            else 0 <= task_index < PRODUCTION_TASK_COUNT
        )

    command = ["scontrol", "show", "job", job_id, "-o"]
    completed = runner(command, check=True, text=True, capture_output=True)
    rows = [
        _scontrol_fields(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(
            "Expected immediate scontrol readback for the submitted array. "
            f"Provided value: {completed.stdout!r}."
        )
    observed_indices: set[int] = set()
    observed_throttles: set[int] = set()
    for row in rows:
        expected_fields = {
            "Account": ACCOUNT,
            "Partition": PARTITION,
            "QOS": QOS,
            "NumTasks": "1",
            "TimeLimit": expected_time,
            "WorkDir": expected_repo,
            "Command": expected_wrapper,
        }
        mismatches = {
            key: {"expected": expected, "provided": row.get(key)}
            for key, expected in expected_fields.items()
            if row.get(key) != expected
        }
        if row.get("NumNodes") not in {"1", "1-1"}:
            mismatches["NumNodes"] = {
                "expected": "1 or the pending-range spelling 1-1",
                "provided": row.get("NumNodes"),
            }
        if row.get("NumCPUs") not in {
            str(CPUS_PER_TASK),
            str(2 * CPUS_PER_TASK),
        }:
            mismatches["NumCPUs"] = {
                "expected": (
                    "16 while pending or 32 after Jean Zay expands "
                    "16 physical CPUs to logical CPUs"
                ),
                "provided": row.get("NumCPUs"),
            }
        for field, suffix in (("StdOut", "out"), ("StdErr", "err")):
            value = row.get(field)
            if value in {None, "", "(null)", "N/A"} or not log_path_matches(
                str(value),
                suffix=suffix,
            ):
                mismatches[field] = {
                    "expected": (
                        f"{expected_log_dir}/{expected_job_name}-"
                        f"{job_id}_<array-task>.{suffix}"
                    ),
                    "provided": value,
                }
        features = row.get("Features", "")
        if CONSTRAINT not in features.split("&") and CONSTRAINT not in features:
            mismatches["Features"] = {
                "expected": CONSTRAINT,
                "provided": features,
            }
        thread_value = row.get("ThreadsPerCore")
        if thread_value is None:
            thread_spec = row.get("ReqB:S:C:T", "")
            thread_value = (
                thread_spec.rsplit(":", maxsplit=1)[-1]
                if ":" in thread_spec
                else None
            )
        if thread_value != "1":
            mismatches["threads_per_core"] = {
                "expected": "1",
                "provided": thread_value,
            }
        req_tres = row.get("ReqTRES", "")
        tres_per_node = row.get("TresPerNode", "")
        if (
            "gres/gpu=1" not in req_tres.split(",")
            and "gres/gpu:1" not in tres_per_node.split(",")
        ):
            mismatches["gpu"] = {
                "expected": "one GPU",
                "provided": {
                    "ReqTRES": req_tres,
                    "TresPerNode": tres_per_node,
                },
            }
        cpus_per_task = row.get("CPUs/Task")
        if cpus_per_task != str(CPUS_PER_TASK):
            mismatches["cpus_per_task"] = {
                "expected": str(CPUS_PER_TASK),
                "provided": cpus_per_task,
            }
        requested_cpu_match = re.search(
            r"(?:^|,)cpu=([0-9]+)(?:,|$)", req_tres
        )
        requested_cpus = (
            None
            if requested_cpu_match is None
            else requested_cpu_match.group(1)
        )
        if requested_cpus != str(CPUS_PER_TASK):
            mismatches["requested_cpus"] = {
                "expected": str(CPUS_PER_TASK),
                "provided": requested_cpus,
            }
        if row.get("ArrayJobId") != job_id:
            mismatches["ArrayJobId"] = {
                "expected": job_id,
                "provided": row.get("ArrayJobId"),
            }
        task_spec = row.get("ArrayTaskId")
        if task_spec is None:
            mismatches["ArrayTaskId"] = {
                "expected": "an exact array task/range",
                "provided": None,
            }
        else:
            indices, throttle = _array_task_spec(task_spec)
            observed_indices.update(indices)
            if throttle is not None:
                observed_throttles.add(throttle)
        field_throttle = row.get("ArrayTaskThrottle")
        if field_throttle not in {None, "N/A", ""}:
            if not field_throttle.isdigit():
                mismatches["ArrayTaskThrottle"] = {
                    "expected": "a decimal throttle",
                    "provided": field_throttle,
                }
            else:
                observed_throttles.add(int(field_throttle))
        if mismatches:
            raise RuntimeError(
                "Expected immediate scontrol readback to match the exact "
                f"successor allocation. Provided mismatches: {mismatches!r}; "
                f"row={row!r}."
            )
    expected_indices = (
        {0} if kind == "canary" else set(range(PRODUCTION_TASK_COUNT))
    )
    if observed_indices != expected_indices:
        raise RuntimeError(
            "Expected immediate scontrol readback to cover the exact array "
            f"indices. Provided value: kind={kind!r}, "
            f"observed={sorted(observed_indices)!r}."
        )
    if kind == "production" and observed_throttles != {
        PRODUCTION_TASK_COUNT
    }:
        raise RuntimeError(
            "Expected production readback to retain exact %6 throttle. "
            f"Provided value: {sorted(observed_throttles)!r}."
        )
    if kind == "canary" and any(value != 1 for value in observed_throttles):
        raise RuntimeError(
            "Expected canary readback throttle, when present, to be one. "
            f"Provided value: {sorted(observed_throttles)!r}."
        )
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "job_id": job_id,
        "kind": kind,
        "array_indices": sorted(observed_indices),
        "array_throttle": (
            None
            if not observed_throttles
            else next(iter(observed_throttles))
        ),
        "resources": dict(resources),
        "time_limit": expected_time,
        "work_dir": expected_repo,
        "command_path": expected_wrapper,
        "log_dir": str(expected_log_dir),
    }


def prepare_submission(
    args: argparse.Namespace,
    *,
    enforce_storage_roots: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Build one immutable canary or production sbatch request."""

    if args.kind not in {"canary", "production"}:
        raise _error("kind to be 'canary' or 'production'", args.kind)
    if (
        type(args.canary_pack_index) is not int
        or not 0 <= args.canary_pack_index < PRODUCTION_TASK_COUNT
    ):
        raise _error(
            "canary-pack-index to be an integer from 0 through 5",
            args.canary_pack_index,
        )
    if args.canary_pack_index != DEFAULT_CANARY_PACK_INDEX:
        raise _error(
            "canary-pack-index to remain the frozen worst-case Conv2 pack 4",
            args.canary_pack_index,
        )
    repo = _resolve_existing(
        args.repo_root, label="staged repository root", kind="directory"
    )
    bundle = _resolve_existing(
        args.bundle_dir, label="successor bundle directory", kind="directory"
    )
    output = _resolve_existing(
        args.output_root, label="successor output root", kind="directory"
    )
    data = _resolve_existing(
        args.data_root, label="ordinary-MNIST data root", kind="directory"
    )
    source_archive = _resolve_existing(
        args.source_archive, label="immutable source archive", kind="file"
    )
    environment = _resolve_existing(
        args.environment_contract,
        label="successor environment contract",
        kind="file",
    )
    contract = load_environment_contract(environment)
    python_executable = contract["resources"]["python_executable"]
    if args.python != python_executable or not Path(args.python).is_absolute():
        raise _error(
            "python to be the exact absolute interpreter frozen in the "
            "environment contract",
            args.python,
        )
    if enforce_storage_roots and python_executable != PYTHON_EXECUTABLE:
        raise _error(
            f"the reviewed Jean Zay module interpreter {PYTHON_EXECUTABLE}",
            python_executable,
        )
    wrapper = _resolve_existing(
        repo / "experiments" / WRAPPER_NAME,
        label="successor Jean Zay wrapper",
        kind="file",
    )
    runtime_cli = _resolve_existing(
        repo / "experiments" / RUNTIME_CLI_NAME,
        label="successor runtime CLI",
        kind="file",
    )
    runtime_module = _resolve_existing(
        repo
        / "experiments"
        / "mnist_conv"
        / "perfectdiode_successor_confirmation.py",
        label="successor runtime module",
        kind="file",
    )
    submitter = _resolve_existing(
        repo / "experiments" / SUBMITTER_NAME,
        label="successor Jean Zay submitter",
        kind="file",
    )
    supervisor = _resolve_existing(
        repo / "experiments" / SUPERVISOR_NAME,
        label="successor persistent Jean Zay supervisor",
        kind="file",
    )
    semantic_canary_verifier = _resolve_existing(
        repo / "experiments" / SEMANTIC_CANARY_VERIFIER_NAME,
        label="successor semantic canary verifier",
        kind="file",
    )
    validate_source_archive(
        source_archive,
        staged_files={
            f"experiments/{RUNTIME_CLI_NAME}": runtime_cli,
            (
                "experiments/mnist_conv/"
                "perfectdiode_successor_confirmation.py"
            ): runtime_module,
        },
    )
    scheduled_preflight_script = _resolve_existing(
        repo / SCHEDULED_PREFLIGHT_RELATIVE,
        label="repo scheduled-run preflight script",
        kind="file",
    )
    experiment_plan_validator = _resolve_existing(
        repo / EXPERIMENT_PLAN_VALIDATOR_RELATIVE,
        label="official experiment-plan validator",
        kind="file",
    )
    remote_user = args.remote_user.strip()
    expected_verifier_path = expected_official_canary_verifier_path(
        contract,
        remote_user=remote_user,
    )
    supplied_verifier = getattr(args, "official_verifier", None)
    official_verifier = _resolve_existing(
        (
            expected_verifier_path
            if supplied_verifier in {None, ""}
            else supplied_verifier
        ),
        label="staged official Jean Zay canary verifier",
        kind="file",
    )
    expected_verifier_hash = contract["verification"][
        "official_canary_verifier_sha256"
    ]
    observed_verifier_hash = sha256_file(official_verifier)
    if observed_verifier_hash != expected_verifier_hash:
        raise RuntimeError(
            "Expected the exact reviewed official Jean Zay canary verifier. "
            f"Provided value: expected={expected_verifier_hash!r}, "
            f"observed={observed_verifier_hash!r}, path={official_verifier}."
        )
    if bundle == output:
        raise _error(
            "successor input bundle and output root to be distinct",
            output,
        )
    storage = contract["storage"]
    source_root = Path(
        storage["source_root_template"].format(user=remote_user)
    )
    data_root = Path(
        storage["data_root_template"].format(user=remote_user)
    )
    result_root = Path(
        storage["result_root_template"].format(user=remote_user)
    )
    if enforce_storage_roots:
        for path, label in (
            (repo, "staged repository root"),
            (source_archive, "source archive"),
            (environment, "environment contract"),
        ):
            _require_relative(path, source_root, label)
        for path, label in (
            (bundle, "successor input bundle"),
            (output, "successor output root"),
        ):
            _require_relative(path, result_root, label)
        if data != data_root:
            raise _error(
                f"ordinary-MNIST data root to be exactly {data_root}",
                data,
            )
        if official_verifier != expected_verifier_path:
            raise _error(
                "staged official canary verifier to be exactly "
                f"{expected_verifier_path}",
                official_verifier,
            )

    array = (
        contract["arrays"]["canary"]
        if args.kind == "canary"
        else contract["arrays"]["production"]
    )
    task_count = (
        CANARY_TASK_COUNT
        if args.kind == "canary"
        else PRODUCTION_TASK_COUNT
    )
    mode = "smoke-pack" if args.kind == "canary" else "run-pack"
    walltime = contract["walltime"][args.kind]
    hashes = {
        "bundle_sha256": sha256_directory_tree(bundle),
        "source_archive_sha256": sha256_file(source_archive),
        "environment_contract_sha256": sha256_file(environment),
        "runtime_cli_sha256": sha256_file(runtime_cli),
        "runtime_module_sha256": sha256_file(runtime_module),
        "submitter_sha256": sha256_file(submitter),
        "supervisor_sha256": sha256_file(supervisor),
        "semantic_canary_verifier_sha256": sha256_file(
            semantic_canary_verifier
        ),
        "experiment_plan_validator_sha256": sha256_file(
            experiment_plan_validator
        ),
        "wrapper_sha256": sha256_file(wrapper),
        "scheduled_preflight_script_sha256": sha256_file(
            scheduled_preflight_script
        ),
        "official_canary_verifier_sha256": observed_verifier_hash,
    }
    exports = {
        "PD_SUCCESSOR_REPO_ROOT": str(repo),
        "PD_SUCCESSOR_PYTHON": args.python,
        "PD_SUCCESSOR_BUNDLE_DIR": str(bundle),
        "PD_SUCCESSOR_OUTPUT_ROOT": str(output),
        "PD_SUCCESSOR_DATA_ROOT": str(data),
        "PD_SUCCESSOR_SOURCE_ARCHIVE": str(source_archive),
        "PD_SUCCESSOR_ENVIRONMENT_CONTRACT": str(environment),
        "PD_SUCCESSOR_BUNDLE_SHA256": hashes["bundle_sha256"],
        "PD_SUCCESSOR_SOURCE_ARCHIVE_SHA256": hashes[
            "source_archive_sha256"
        ],
        "PD_SUCCESSOR_ENVIRONMENT_CONTRACT_SHA256": hashes[
            "environment_contract_sha256"
        ],
        "PD_SUCCESSOR_RUNTIME_CLI_SHA256": hashes[
            "runtime_cli_sha256"
        ],
        "PD_SUCCESSOR_RUNTIME_MODULE_SHA256": hashes[
            "runtime_module_sha256"
        ],
        "PD_SUCCESSOR_WRAPPER_SHA256": hashes["wrapper_sha256"],
        "PD_SUCCESSOR_MODE": mode,
        "PD_SUCCESSOR_ARRAY_KIND": args.kind,
        "PD_SUCCESSOR_CANARY_PACK_INDEX": str(args.canary_pack_index),
        "PD_SUCCESSOR_PRODUCTION_PACK_COUNT": str(PRODUCTION_TASK_COUNT),
        "PD_SUCCESSOR_LOGICAL_ENTRY_COUNT": str(LOGICAL_ENTRY_COUNT),
        "PD_SUCCESSOR_RUNS_PER_GPU": str(RUNS_PER_GPU),
        "PD_SUCCESSOR_CONCURRENT_WITHIN_PACK": "true",
        "PD_SUCCESSOR_ALLOCATION": ALLOCATION_ID,
        "PD_SUCCESSOR_PROJECT": PROJECT,
        "PD_SUCCESSOR_ACCOUNT": ACCOUNT,
        "PD_SUCCESSOR_PARTITION": PARTITION,
        "PD_SUCCESSOR_QOS": QOS,
        "PD_SUCCESSOR_CONSTRAINT": CONSTRAINT,
        "PD_SUCCESSOR_MODULE": MODULE,
        "PD_SUCCESSOR_AUTHORIZATION": ALLOCATION_AUTHORIZATION,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    log_dir = output / "slurm"
    command = [
        "sbatch",
        "--parsable",
        f"--job-name=pd-successor-{args.kind}",
        f"--array={array}",
        f"--chdir={repo}",
        f"--output={log_dir}/%x-%A_%a.out",
        f"--error={log_dir}/%x-%A_%a.err",
        "--nodes=1",
        "--ntasks=1",
        f"--time={walltime}",
        f"--account={ACCOUNT}",
        f"--partition={PARTITION}",
        f"--qos={QOS}",
        f"--constraint={CONSTRAINT}",
        f"--gres=gpu:{GPUS_PER_TASK}",
        f"--cpus-per-task={CPUS_PER_TASK}",
        "--hint=nomultithread",
        _export_arg(exports),
        str(wrapper),
    ]
    binding = {
        "schema_version": "perfectdiode-successor-jeanzay-submission/v1",
        "kind": args.kind,
        "array": array,
        "task_count": task_count,
        "canary_pack_index": args.canary_pack_index,
        "canary_entry_indices": list(DEFAULT_CANARY_ENTRY_INDICES),
        "fixed_entry_packs": [
            list(entry_indices) for entry_indices in FIXED_ENTRY_PACKS
        ],
        "logical_entry_count": LOGICAL_ENTRY_COUNT,
        "runs_per_gpu": RUNS_PER_GPU,
        "concurrent_within_pack": True,
        "remote_user": remote_user,
        "repo_root": str(repo),
        "bundle_dir": str(bundle),
        "output_root": str(output),
        "data_root": str(data),
        "source_archive": str(source_archive),
        "environment_contract": str(environment),
        "runtime_cli": str(runtime_cli),
        "runtime_module": str(runtime_module),
        "submitter": str(submitter),
        "supervisor": str(supervisor),
        "semantic_canary_verifier": str(semantic_canary_verifier),
        "experiment_plan_validator": str(experiment_plan_validator),
        "wrapper": str(wrapper),
        "scheduled_preflight_script": str(scheduled_preflight_script),
        "official_canary_verifier": str(official_verifier),
        "supervisor_state": _optional_absolute_path(
            getattr(args, "supervisor_state", None)
        ),
        "official_canary_receipt": _optional_absolute_path(
            getattr(args, "official_canary_receipt", None)
        ),
        "successor_canary_receipt": _optional_absolute_path(
            getattr(args, "canary_receipt", None)
        ),
        "launch_authorization_receipt": _optional_absolute_path(
            getattr(args, "launch_authorization_receipt", None)
        ),
        "scheduled_preflight_receipt": _optional_absolute_path(
            getattr(args, "scheduled_preflight_receipt", None)
        ),
        "preflight_receipt": _optional_absolute_path(
            getattr(args, "preflight_receipt", None)
        ),
        "log_dir": str(log_dir),
        "hashes": hashes,
        "resources": {
            "allocation_id": ALLOCATION_ID,
            "project": PROJECT,
            "account": ACCOUNT,
            "partition": PARTITION,
            "qos": QOS,
            "constraint": CONSTRAINT,
            "nodes": 1,
            "tasks": 1,
            "gpus_per_task": GPUS_PER_TASK,
            "cpus_per_task": CPUS_PER_TASK,
            "host_memory_policy": HOST_MEMORY_POLICY,
            "hint": "nomultithread",
            "module": MODULE,
            "python_executable": python_executable,
            "walltime": walltime,
        },
    }
    metadata = {
        **binding,
        "binding_sha256": _canonical_json_sha256(binding),
        "wrapper_preflight_command": ["bash", str(wrapper), "--preflight"],
        "wrapper_preflight_environment": exports,
    }
    return command, metadata


def _append_exports(command: list[str], values: Mapping[str, str]) -> None:
    indices = [
        index
        for index, item in enumerate(command)
        if item.startswith("--export=ALL,")
    ]
    if len(indices) != 1:
        raise _error("one Slurm --export=ALL argument", indices)
    for key, value in values.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise _error("additional export keys to use shell syntax", key)
        if any(character in value for character in ",\r\n"):
            raise _error(
                "additional export values not to contain commas/newlines",
                {key: value},
            )
    command[indices[0]] += "," + ",".join(
        f"{key}={value}" for key, value in values.items()
    )


def _launch_pair_args(
    args: argparse.Namespace,
) -> tuple[argparse.Namespace, argparse.Namespace]:
    output_root = getattr(args, "output_root", None)
    if not output_root:
        raise RuntimeError(
            "Expected --output-root to bind the shared canary/production "
            "experiment output."
        )
    resolved_output = Path(output_root).expanduser().resolve()
    common = vars(args).copy()
    canary = argparse.Namespace(
        **{
            **common,
            "kind": "canary",
            "output_root": str(resolved_output),
        }
    )
    production = argparse.Namespace(
        **{
            **common,
            "kind": "production",
            "output_root": str(resolved_output),
        }
    )
    return canary, production


def append_launch_template_exports(
    command: list[str],
    *,
    authorization: Mapping[str, Any],
    scheduled_preflight_receipt_path: str | Path,
    scheduled_preflight_receipt_sha256: str,
    include_canary_gate: bool,
) -> None:
    exports = {
            "PD_SUCCESSOR_LAUNCH_AUTHORIZATION_RECEIPT": str(
                authorization["receipt_path"]
            ),
            "PD_SUCCESSOR_LAUNCH_AUTHORIZATION_RECEIPT_SHA256": str(
                authorization["receipt_sha256"]
            ),
            "PD_SUCCESSOR_SCHEDULED_PREFLIGHT_RECEIPT": str(
                Path(scheduled_preflight_receipt_path)
                .expanduser()
                .resolve()
            ),
            "PD_SUCCESSOR_SCHEDULED_PREFLIGHT_RECEIPT_SHA256": (
                scheduled_preflight_receipt_sha256
            ),
            "PD_SUCCESSOR_PREFLIGHT_RECEIPT": (
                PREFLIGHT_PATH_PLACEHOLDER
            ),
            "PD_SUCCESSOR_PREFLIGHT_RECEIPT_SHA256": (
                PREFLIGHT_SHA256_PLACEHOLDER
            ),
        }
    if include_canary_gate:
        exports.update(
            {
                "PD_SUCCESSOR_CANARY_RECEIPT": (
                    CANARY_RECEIPT_PATH_PLACEHOLDER
                ),
                "PD_SUCCESSOR_CANARY_RECEIPT_SHA256": (
                    CANARY_RECEIPT_SHA256_PLACEHOLDER
                ),
            }
        )
    _append_exports(command, exports)


def materialize_launch_command(
    command_template: Sequence[str],
    *,
    preflight_receipt_path: str | Path,
    preflight_receipt_sha256: str,
) -> list[str]:
    path = str(Path(preflight_receipt_path).expanduser().resolve())
    if not re.fullmatch(r"[0-9a-f]{64}", preflight_receipt_sha256):
        raise _error(
            "preflight receipt SHA-256 to be lowercase hex",
            preflight_receipt_sha256,
        )
    result = [
        item.replace(PREFLIGHT_PATH_PLACEHOLDER, path).replace(
            PREFLIGHT_SHA256_PLACEHOLDER,
            preflight_receipt_sha256,
        )
        for item in command_template
    ]
    if any(
        PREFLIGHT_PATH_PLACEHOLDER in item
        or PREFLIGHT_SHA256_PLACEHOLDER in item
        for item in result
    ):
        raise RuntimeError(
            "Expected every command self-reference placeholder to be "
            "materialized exactly once."
        )
    return result


def materialize_canary_gate_command(
    command_template: Sequence[str],
    *,
    canary_receipt_path: str | Path,
    canary_receipt_sha256: str,
) -> list[str]:
    path = str(Path(canary_receipt_path).expanduser().resolve())
    if not re.fullmatch(r"[0-9a-f]{64}", canary_receipt_sha256):
        raise _error(
            "canary receipt SHA-256 to be lowercase hex",
            canary_receipt_sha256,
        )
    result = [
        item.replace(CANARY_RECEIPT_PATH_PLACEHOLDER, path).replace(
            CANARY_RECEIPT_SHA256_PLACEHOLDER,
            canary_receipt_sha256,
        )
        for item in command_template
    ]
    if any(
        CANARY_RECEIPT_PATH_PLACEHOLDER in item
        or CANARY_RECEIPT_SHA256_PLACEHOLDER in item
        for item in result
    ):
        raise RuntimeError(
            "Expected every production canary-gate placeholder to be "
            "materialized exactly once."
        )
    return result


def launch_contract_value(
    *,
    canary_command_template: Sequence[str],
    canary_metadata: Mapping[str, Any],
    production_command_template: Sequence[str],
    production_metadata: Mapping[str, Any],
    authorization: Mapping[str, Any],
    scheduled_preflight_receipt_path: str | Path,
    scheduled_preflight_receipt_sha256: str,
) -> dict[str, Any]:
    for label, command in (
        ("canary", canary_command_template),
        ("production", production_command_template),
    ):
        if not isinstance(command, Sequence) or isinstance(command, str) or not all(
            isinstance(item, str) for item in command
        ):
            raise _error(f"{label} command template to be a string list", command)
        placeholders = [
            PREFLIGHT_PATH_PLACEHOLDER,
            PREFLIGHT_SHA256_PLACEHOLDER,
            *(
                []
                if label == "canary"
                else [
                    CANARY_RECEIPT_PATH_PLACEHOLDER,
                    CANARY_RECEIPT_SHA256_PLACEHOLDER,
                ]
            ),
        ]
        placeholder_counts = {
            placeholder: sum(item.count(placeholder) for item in command)
            for placeholder in placeholders
        }
        if any(count != 1 for count in placeholder_counts.values()):
            raise _error(
                f"{label} command template to contain every required "
                "self-reference placeholder exactly once",
                placeholder_counts,
            )
    common_fields = (
        "repo_root",
        "bundle_dir",
        "data_root",
        "source_archive",
        "environment_contract",
        "runtime_cli",
        "runtime_module",
        "submitter",
        "supervisor",
        "semantic_canary_verifier",
        "experiment_plan_validator",
        "wrapper",
        "scheduled_preflight_script",
        "official_canary_verifier",
        "supervisor_state",
        "official_canary_receipt",
        "successor_canary_receipt",
        "preflight_receipt",
        "hashes",
        "canary_pack_index",
        "canary_entry_indices",
        "fixed_entry_packs",
        "logical_entry_count",
        "runs_per_gpu",
        "concurrent_within_pack",
        "remote_user",
    )
    common_mismatches = {
        key: {
            "canary": canary_metadata.get(key),
            "production": production_metadata.get(key),
        }
        for key in common_fields
        if canary_metadata.get(key) != production_metadata.get(key)
    }
    if common_mismatches:
        raise RuntimeError(
            "Expected canary and production to share immutable launch inputs. "
            f"Provided mismatches: {common_mismatches!r}."
        )
    resources = dict(canary_metadata["resources"])
    production_resources = dict(production_metadata["resources"])
    resources.pop("walltime", None)
    production_resources.pop("walltime", None)
    if resources != production_resources:
        raise RuntimeError(
            "Expected canary and production to share exact non-walltime "
            f"resources. Provided value: canary={resources!r}, "
            f"production={production_resources!r}."
        )
    if canary_metadata["output_root"] != production_metadata["output_root"]:
        raise RuntimeError(
            "Expected canary smoke/ and production entries/ to share one "
            f"experiment output root. Provided value: "
            f"canary={canary_metadata['output_root']!r}, "
            f"production={production_metadata['output_root']!r}."
        )
    scheduled_path = str(
        Path(scheduled_preflight_receipt_path).expanduser().resolve()
    )
    canary_template = list(canary_command_template)
    production_template = list(production_command_template)
    return {
        "schema_version": "perfectdiode-successor-launch-contract/v1",
        "self_reference_placeholders": {
            "preflight_receipt_path": PREFLIGHT_PATH_PLACEHOLDER,
            "preflight_receipt_sha256": PREFLIGHT_SHA256_PLACEHOLDER,
            "canary_receipt_path": CANARY_RECEIPT_PATH_PLACEHOLDER,
            "canary_receipt_sha256": CANARY_RECEIPT_SHA256_PLACEHOLDER,
        },
        "python_executable": canary_metadata["resources"][
            "python_executable"
        ],
        "output_root": canary_metadata["output_root"],
        "common_inputs": {
            key: canary_metadata[key] for key in common_fields
        }
        | {
            "launch_authorization_receipt_path": authorization[
                "receipt_path"
            ],
            "launch_authorization_receipt_sha256": authorization[
                "receipt_sha256"
            ],
            "scheduled_preflight_receipt_path": scheduled_path,
            "scheduled_preflight_receipt_sha256": (
                scheduled_preflight_receipt_sha256
            ),
        },
        "resources": resources,
        "commands": {
            "canary": {
                "kind": "canary",
                "array": CANARY_ARRAY,
                "task_count": CANARY_TASK_COUNT,
                "walltime": canary_metadata["resources"]["walltime"],
                "output_root": canary_metadata["output_root"],
                "template": canary_template,
                "template_sha256": _canonical_json_sha256(
                    canary_template
                ),
            },
            "production": {
                "kind": "production",
                "array": PRODUCTION_ARRAY,
                "task_count": PRODUCTION_TASK_COUNT,
                "walltime": production_metadata["resources"]["walltime"],
                "output_root": production_metadata["output_root"],
                "template": production_template,
                "template_sha256": _canonical_json_sha256(
                    production_template
                ),
            },
        },
    }


def immutable_input_binding(
    metadata: Mapping[str, Any],
    runtime_preflight: Mapping[str, Any],
    *,
    scheduled_preflight_receipt_path: str | Path,
    scheduled_preflight_receipt_sha256: str,
    launch_authorization_receipt_path: str | Path,
    launch_authorization_receipt_sha256: str,
    approved_plan: Mapping[str, Any],
    launch_contract: Mapping[str, Any],
) -> dict[str, Any]:
    hashes = metadata.get("hashes")
    if not isinstance(hashes, dict):
        raise _error("submission metadata hashes to be a JSON object", hashes)
    return {
        "schema_version": "perfectdiode-successor-gate-input/v1",
        "bundle_id": runtime_preflight["bundle_id"],
        "bundle_manifest_sha256": runtime_preflight[
            "bundle_manifest_sha256"
        ],
        "config_sha256": runtime_preflight["config_sha256"],
        "config_file_sha256": runtime_preflight["config_file_sha256"],
        "repo_root": metadata["repo_root"],
        "bundle_dir": metadata["bundle_dir"],
        "data_root": metadata["data_root"],
        "source_archive": metadata["source_archive"],
        "environment_contract": metadata["environment_contract"],
        "runtime_cli": metadata["runtime_cli"],
        "runtime_module": metadata["runtime_module"],
        "submitter": metadata["submitter"],
        "supervisor": metadata["supervisor"],
        "semantic_canary_verifier": metadata[
            "semantic_canary_verifier"
        ],
        "experiment_plan_validator": metadata[
            "experiment_plan_validator"
        ],
        "wrapper": metadata["wrapper"],
        "scheduled_preflight_script": metadata[
            "scheduled_preflight_script"
        ],
        "official_canary_verifier": metadata[
            "official_canary_verifier"
        ],
        "canary_pack_index": metadata["canary_pack_index"],
        "canary_entry_indices": list(metadata["canary_entry_indices"]),
        "fixed_entry_packs": [
            list(entry_indices)
            for entry_indices in metadata["fixed_entry_packs"]
        ],
        "logical_entry_count": metadata["logical_entry_count"],
        "runs_per_gpu": metadata["runs_per_gpu"],
        "concurrent_within_pack": metadata["concurrent_within_pack"],
        "hashes": dict(hashes),
        "runtime_preflight_sha256": _canonical_json_sha256(
            runtime_preflight
        ),
        "scheduled_preflight_receipt_path": str(
            Path(scheduled_preflight_receipt_path).expanduser().resolve()
        ),
        "scheduled_preflight_receipt_sha256": (
            scheduled_preflight_receipt_sha256
        ),
        "launch_authorization_receipt_path": str(
            Path(launch_authorization_receipt_path).expanduser().resolve()
        ),
        "launch_authorization_receipt_sha256": (
            launch_authorization_receipt_sha256
        ),
        "approved_plan": dict(approved_plan),
        "launch_contract": dict(launch_contract),
        "launch_contract_sha256": _canonical_json_sha256(
            launch_contract
        ),
    }


def _expected_plan_supervisor_launcher(
    metadata: Mapping[str, Any],
) -> list[str]:
    required = (
        "supervisor_state",
        "bundle_dir",
        "output_root",
        "data_root",
        "source_archive",
        "environment_contract",
        "launch_authorization_receipt",
        "scheduled_preflight_receipt",
        "preflight_receipt",
        "official_canary_receipt",
        "successor_canary_receipt",
        "repo_root",
        "official_canary_verifier",
    )
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise RuntimeError(
            "Expected complete paired launch paths before validating the "
            f"approved plan. Provided missing fields: {missing!r}."
        )
    return [
        str(metadata["resources"]["python_executable"]),
        str(metadata["supervisor"]),
        "--state",
        str(metadata["supervisor_state"]),
        "--bundle-dir",
        str(metadata["bundle_dir"]),
        "--output-root",
        str(metadata["output_root"]),
        "--data-root",
        str(metadata["data_root"]),
        "--source-archive",
        str(metadata["source_archive"]),
        "--environment-contract",
        str(metadata["environment_contract"]),
        "--launch-authorization-receipt",
        str(metadata["launch_authorization_receipt"]),
        "--launch-authorization-receipt-sha256",
        LAUNCH_AUTHORIZATION_SHA256_PLACEHOLDER,
        "--scheduled-preflight-receipt",
        str(metadata["scheduled_preflight_receipt"]),
        "--preflight-receipt",
        str(metadata["preflight_receipt"]),
        "--tracker-gate-receipt",
        TRACKER_GATE_RECEIPT_PATH_PLACEHOLDER,
        "--tracker-gate-receipt-sha256",
        TRACKER_GATE_RECEIPT_SHA256_PLACEHOLDER,
        "--official-canary-receipt",
        str(metadata["official_canary_receipt"]),
        "--canary-receipt",
        str(metadata["successor_canary_receipt"]),
        "--repo-root",
        str(metadata["repo_root"]),
        "--remote-user",
        str(metadata["remote_user"]),
        "--official-verifier",
        str(metadata["official_canary_verifier"]),
        "--canary-pack-index",
        str(metadata["canary_pack_index"]),
        "--poll-seconds",
        "60",
        "--arm-long-run",
        "--enable-submit",
    ]


def validate_launch_authorization_receipt(
    path: str | Path,
    *,
    expected_sha256: str,
    metadata: Mapping[str, Any],
    runtime_preflight: Mapping[str, Any],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    source = Path(path).expanduser().absolute()
    if source.is_symlink():
        raise _error("launch-authorization receipt not to be a symlink", source)
    source = source.resolve()
    if not source.is_file():
        raise _error(
            "launch-authorization receipt to be an existing file", source
        )
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""):
        raise _error(
            "launch-authorization receipt SHA-256 to be lowercase hex",
            expected_sha256,
        )
    observed_receipt_hash = sha256_file(source)
    if observed_receipt_hash != expected_sha256:
        raise RuntimeError(
            "Expected the immutable launch-authorization receipt SHA-256 to "
            f"match. Provided value: expected={expected_sha256!r}, "
            f"observed={observed_receipt_hash!r}."
        )
    value = read_json(source)
    if not isinstance(value, dict):
        raise _error("launch-authorization receipt to be a JSON object", value)
    expected_top = {
        "schema_version",
        "status",
        "launch_authorized",
        "approved_plan",
        "bundle",
        "execution",
    }
    if set(value) != expected_top:
        raise _error(
            f"launch-authorization keys to be exactly {sorted(expected_top)!r}",
            sorted(value),
        )
    if (
        value.get("schema_version")
        != "perfectdiode-successor-launch-authorization/v1"
        or value.get("status") != "approved"
        or value.get("launch_authorized") is not True
    ):
        raise RuntimeError(
            "Expected an explicitly approved successor launch authorization. "
            f"Provided value: {value!r}."
        )
    plan = value.get("approved_plan")
    if not isinstance(plan, dict) or set(plan) != {"path", "sha256"}:
        raise _error(
            "launch authorization approved_plan to contain path and sha256",
            plan,
        )
    plan_path = Path(str(plan["path"])).expanduser().absolute()
    if plan_path.is_symlink():
        raise _error("approved plan not to be a symlink", plan_path)
    plan_path = plan_path.resolve()
    if (
        not plan_path.is_file()
        or not re.fullmatch(r"[0-9a-f]{64}", str(plan["sha256"]))
        or sha256_file(plan_path) != plan["sha256"]
    ):
        raise RuntimeError(
            "Expected launch authorization to bind an existing immutable "
            f"approved plan. Provided value: {plan!r}."
        )
    repo_root = Path(str(metadata["repo_root"])).resolve()
    expected_plan_path = (
        repo_root
        / "docs"
        / "experiment_plans"
        / "perfectdiode-conv12-best-observed-confirmation-20260727-v1.md"
    ).resolve()
    if plan_path != expected_plan_path:
        raise _error(
            f"approved plan path to be exactly {expected_plan_path}",
            plan_path,
        )
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_blocks = re.findall(
        r"```json[ \t]*\n(.*?)\n```",
        plan_text,
        flags=re.DOTALL,
    )
    if len(plan_blocks) != 1:
        raise _error(
            "approved plan to contain exactly one fenced JSON contract",
            len(plan_blocks),
        )
    try:
        plan_contract = json.loads(plan_blocks[0])
    except json.JSONDecodeError as exc:
        raise _error(
            "approved plan fenced block to contain valid JSON", str(exc)
        ) from exc
    if not isinstance(plan_contract, dict):
        raise _error("approved plan contract to be a JSON object", plan_contract)
    if (
        plan_contract.get("schema_version") != "experiment-run-plan/v1"
        or plan_contract.get("experiment_id")
        != "perfectdiode-conv12-best-observed-confirmation-20260727-v1"
    ):
        raise RuntimeError(
            "Expected the official successor experiment-run-plan/v1 identity. "
            f"Provided value: schema={plan_contract.get('schema_version')!r}, "
            f"experiment_id={plan_contract.get('experiment_id')!r}."
        )
    approval = plan_contract.get("approval")
    if (
        not isinstance(approval, dict)
        or approval.get("status") != "approved"
        or approval.get("approved_by") != "Filip"
        or not isinstance(approval.get("approved_at"), str)
    ):
        raise RuntimeError(
            "Expected explicit approved status/by/at in the official plan. "
            f"Provided value: {approval!r}."
        )
    try:
        approved_at = datetime.fromisoformat(approval["approved_at"])
    except ValueError as exc:
        raise _error(
            "approved plan approved_at to be ISO-8601",
            approval["approved_at"],
        ) from exc
    if approved_at.tzinfo is None:
        raise _error(
            "approved plan approved_at to be timezone-aware",
            approval["approved_at"],
        )
    execution_plan = plan_contract.get("execution")
    if not isinstance(execution_plan, dict):
        raise _error(
            "approved plan execution to be a JSON object",
            execution_plan,
        )
    expected_launcher = _expected_plan_supervisor_launcher(metadata)
    if execution_plan.get("launcher") != expected_launcher:
        raise RuntimeError(
            "Expected the approved plan launcher to encode the exact paired "
            "supervisor paths and absolute module Python. "
            f"Provided value: expected={expected_launcher!r}, "
            f"observed={execution_plan.get('launcher')!r}."
        )
    storage_plan = plan_contract.get("storage")
    remote_staging = (
        storage_plan.get("remote_staging")
        if isinstance(storage_plan, dict)
        else None
    )
    if not isinstance(remote_staging, list) or not all(
        isinstance(item, dict) and isinstance(item.get("path"), str)
        for item in remote_staging
    ):
        raise _error(
            "approved plan storage.remote_staging to be a list of path "
            "objects",
            remote_staging,
        )
    staged_paths = {item["path"] for item in remote_staging}
    expected_output_paths = {metadata["output_root"]}
    if not expected_output_paths.issubset(staged_paths):
        raise RuntimeError(
            "Expected approved plan storage.remote_staging to bind both "
            f"output roots. Provided value: expected={sorted(expected_output_paths)!r}, "
            f"observed={sorted(staged_paths)!r}."
        )
    sweep = plan_contract.get("sweep")
    if not isinstance(sweep, dict):
        raise _error("approved plan sweep to be a JSON object", sweep)
    expected_plan_identities = {
        "config_sha256": runtime_preflight["config_file_sha256"],
        "manifest_sha256": runtime_preflight["bundle_manifest_sha256"],
        "expected_initial_job_count": 12,
        "maximum_total_job_count": 12,
    }
    plan_identity_mismatches = {
        key: {"expected": expected, "provided": sweep.get(key)}
        for key, expected in expected_plan_identities.items()
        if sweep.get(key) != expected
    }
    if plan_identity_mismatches:
        raise RuntimeError(
            "Expected the approved plan to bind the exact 12-entry "
            f"config/manifest identities. Provided mismatches: "
            f"{plan_identity_mismatches!r}."
        )
    plan_validator = (
        repo_root
        / "skills"
        / "run-experiment-pipeline"
        / "scripts"
        / "validate_experiment_plan.py"
    )
    if not plan_validator.is_file():
        raise _error(
            "repo experiment-plan validator to be an existing file",
            plan_validator,
        )
    validated_plan = runner(
        [
            sys.executable,
            str(plan_validator),
            str(plan_path),
            "--require-approved",
            "--verify-files",
        ],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    if validated_plan.returncode != 0:
        raise RuntimeError(
            "Expected the official experiment-plan validator to accept the "
            f"final approved plan. Provided value: "
            f"stdout={validated_plan.stdout!r}, "
            f"stderr={validated_plan.stderr!r}."
        )
    hashes = metadata.get("hashes")
    if not isinstance(hashes, dict):
        raise _error("submission metadata hashes to be a JSON object", hashes)
    expected_bundle = {
        "bundle_id": runtime_preflight["bundle_id"],
        "manifest_sha256": runtime_preflight["bundle_manifest_sha256"],
        "config_sha256": runtime_preflight["config_sha256"],
        "config_file_sha256": runtime_preflight["config_file_sha256"],
        "bundle_tree_sha256": hashes["bundle_sha256"],
    }
    expected_execution = {
        "source_archive_sha256": hashes["source_archive_sha256"],
        "environment_contract_sha256": hashes[
            "environment_contract_sha256"
        ],
        "runtime_cli_sha256": hashes["runtime_cli_sha256"],
        "runtime_module_sha256": hashes["runtime_module_sha256"],
        "submitter_sha256": hashes["submitter_sha256"],
        "supervisor_sha256": hashes["supervisor_sha256"],
        "semantic_canary_verifier_sha256": hashes[
            "semantic_canary_verifier_sha256"
        ],
        "experiment_plan_validator_sha256": hashes[
            "experiment_plan_validator_sha256"
        ],
        "wrapper_sha256": hashes["wrapper_sha256"],
        "scheduled_preflight_script_sha256": hashes[
            "scheduled_preflight_script_sha256"
        ],
        "official_canary_verifier_sha256": hashes[
            "official_canary_verifier_sha256"
        ],
    }
    mismatches = {}
    if value.get("bundle") != expected_bundle:
        mismatches["bundle"] = {
            "expected": expected_bundle,
            "provided": value.get("bundle"),
        }
    if value.get("execution") != expected_execution:
        mismatches["execution"] = {
            "expected": expected_execution,
            "provided": value.get("execution"),
        }
    if mismatches:
        raise RuntimeError(
            "Expected launch authorization to bind the exact current bundle, "
            f"source, environment, runtime, and wrapper. Provided "
            f"mismatches: {mismatches!r}."
        )
    return {
        "receipt_path": str(source),
        "receipt_sha256": observed_receipt_hash,
        "approved_plan": {
            "path": str(plan_path),
            "sha256": plan["sha256"],
        },
        "receipt": value,
    }


def validate_scheduled_preflight_receipt(
    path: str | Path,
    *,
    metadata: Mapping[str, Any],
    runtime_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = read_json(source)
    if not isinstance(value, dict):
        raise _error(
            "scheduled-run preflight receipt to be a JSON object", value
        )
    expected = {
        "schema_version": "scheduled-run-preflight-receipt/v1",
        "status": "passed",
        "runner": metadata["wrapper"],
        "runner_sha256": metadata["hashes"]["wrapper_sha256"],
        "cwd": metadata["repo_root"],
        "exit_code": 0,
    }
    mismatches = {
        key: {"expected": expected_value, "provided": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    expected_command = [metadata["wrapper"], "--preflight"]
    if value.get("preflight_command") != expected_command:
        mismatches["preflight_command"] = {
            "expected": expected_command,
            "provided": value.get("preflight_command"),
        }
    receipt_preflight = _parse_result_marker(
        value.get("stdout", ""),
        label="scheduled-run runtime preflight stdout",
    )
    if receipt_preflight != dict(runtime_preflight):
        mismatches["stdout"] = {
            "expected_runtime_preflight": dict(runtime_preflight),
            "provided": receipt_preflight,
        }
    if mismatches:
        raise RuntimeError(
            "Expected the official repo scheduled-run preflight receipt to "
            f"bind the exact staged wrapper and runtime preflight. "
            f"Provided mismatches: {mismatches!r}."
        )
    return value


def ensure_scheduled_preflight_receipt(
    path: str | Path,
    *,
    metadata: Mapping[str, Any],
    runtime_preflight: Mapping[str, Any],
    create: bool,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[Path, str]:
    supplied = Path(path).expanduser().absolute()
    if supplied.is_symlink():
        raise _error(
            "scheduled-run preflight receipt not to be a symlink", supplied
        )
    source = supplied.resolve()
    if not source.is_file():
        if not create:
            raise _error(
                "scheduled-run preflight receipt to be an existing file",
                source,
            )
        if not source.parent.is_dir():
            raise _error(
                "scheduled-run preflight receipt parent to exist",
                source.parent,
            )
        script = Path(metadata["scheduled_preflight_script"])
        command = [
            sys.executable,
            str(script),
            "--runner",
            metadata["wrapper"],
            "--cwd",
            metadata["repo_root"],
            "--receipt",
            str(source),
            "--timeout-seconds",
            "120",
        ]
        exports = metadata.get("wrapper_preflight_environment")
        if not isinstance(exports, dict):
            raise _error(
                "wrapper preflight environment to be a JSON object", exports
            )
        runner(
            command,
            check=True,
            text=True,
            capture_output=True,
            env={**os.environ, **exports},
        )
    validate_scheduled_preflight_receipt(
        source,
        metadata=metadata,
        runtime_preflight=runtime_preflight,
    )
    return source, sha256_file(source)


def preflight_receipt_value(
    *,
    input_binding: Mapping[str, Any],
    runtime_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    scheduled_path = Path(
        str(input_binding["scheduled_preflight_receipt_path"])
    ).expanduser().resolve()
    scheduled = read_json(scheduled_path)
    if not isinstance(scheduled, dict):
        raise _error(
            "scheduled-run preflight receipt to be a JSON object", scheduled
        )
    return {
        "schema_version": "perfectdiode-successor-preflight-receipt/v1",
        "status": "passed",
        "passed": True,
        "runtime_preflight": dict(runtime_preflight),
        "runtime_preflight_sha256": _canonical_json_sha256(
            runtime_preflight
        ),
        "input_binding": dict(input_binding),
        "input_binding_sha256": _canonical_json_sha256(input_binding),
        "scheduled_run_preflight": {
            "path": str(scheduled_path),
            "sha256": input_binding[
                "scheduled_preflight_receipt_sha256"
            ],
            "schema_version": scheduled.get("schema_version"),
            "status": scheduled.get("status"),
            "runner": scheduled.get("runner"),
            "runner_sha256": scheduled.get("runner_sha256"),
        },
    }


def ensure_preflight_receipt(
    path: str | Path,
    *,
    expected_value: Mapping[str, Any],
    create: bool,
) -> tuple[Path, str]:
    supplied = Path(path).expanduser().absolute()
    if supplied.is_symlink():
        raise _error("preflight receipt not to be a symlink", supplied)
    source = supplied.resolve()
    if source.is_file():
        observed = read_json(source)
        if observed != dict(expected_value):
            raise RuntimeError(
                "Expected an existing preflight receipt to retain the exact "
                f"current launch binding. Provided value: path={source}."
            )
    elif create:
        if not source.parent.is_dir():
            raise _error(
                "preflight receipt parent to be an existing directory",
                source.parent,
            )
        atomic_write_json(source, dict(expected_value), canonical=True)
    else:
        raise _error("preflight receipt to be an existing file", source)
    return source, sha256_file(source)


def canary_gate_binding(
    *,
    input_binding: Mapping[str, Any],
    preflight_receipt_path: str | Path,
    preflight_receipt_sha256: str,
) -> dict[str, Any]:
    return {
        **dict(input_binding),
        "preflight_receipt_path": str(
            Path(preflight_receipt_path).expanduser().resolve()
        ),
        "preflight_receipt_sha256": preflight_receipt_sha256,
    }


def run_wrapper_preflight(
    metadata: Mapping[str, Any],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    command = metadata.get("wrapper_preflight_command")
    exports = metadata.get("wrapper_preflight_environment")
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        raise _error("wrapper preflight command to be a string list", command)
    if not isinstance(exports, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in exports.items()
    ):
        raise _error("wrapper preflight environment to be a string map", exports)
    completed = runner(
        command,
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, **exports},
    )
    value = _parse_runtime_preflight(completed.stdout)
    return {
        "command": command,
        "runtime_preflight": value,
        "runtime_preflight_sha256": _canonical_json_sha256(value),
    }


def dispatch(
    args: argparse.Namespace,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    allocation_verifier: Callable[..., dict[str, Any]] = (
        verify_login_umg_v100_allocation
    ),
    enforce_storage_roots: bool = True,
) -> dict[str, Any]:
    timeout_value = getattr(
        args,
        "command_timeout_seconds",
        DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )
    if (
        isinstance(timeout_value, bool)
        or not isinstance(timeout_value, (int, float))
        or not 1 <= float(timeout_value) <= 24 * 60 * 60
    ):
        raise _error(
            "command-timeout-seconds to be a number from 1 through 86400",
            timeout_value,
        )
    raw_runner = runner

    def bounded_runner(
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        kwargs.setdefault("timeout", float(timeout_value))
        return raw_runner(list(command), **kwargs)

    runner = bounded_runner
    if args.submit:
        raise RuntimeError(
            "Expected real sbatch submission to occur only through the "
            "flock-protected persistent successor supervisor. Standalone "
            "submitter --submit is disabled; use dry-run or --test-only."
        )
    canary_args, production_args = _launch_pair_args(args)
    canary_command, canary_metadata = prepare_submission(
        canary_args,
        enforce_storage_roots=enforce_storage_roots,
    )
    production_command, production_metadata = prepare_submission(
        production_args,
        enforce_storage_roots=enforce_storage_roots,
    )
    command = (
        canary_command if args.kind == "canary" else production_command
    )
    metadata = (
        canary_metadata if args.kind == "canary" else production_metadata
    )
    preflight = run_wrapper_preflight(canary_metadata, runner=runner)
    production_preflight = run_wrapper_preflight(
        production_metadata, runner=runner
    )
    if (
        production_preflight["runtime_preflight"]
        != preflight["runtime_preflight"]
    ):
        raise RuntimeError(
            "Expected canary and production to use one identical runtime "
            "preflight under the same module/interpreter."
        )
    runtime_preflight = preflight["runtime_preflight"]
    public_metadata = {
        key: value
        for key, value in metadata.items()
        if key != "wrapper_preflight_environment"
    }
    blockers: list[str] = []
    if not args.launch_authorization_receipt:
        blockers.append("missing --launch-authorization-receipt")
    if not args.launch_authorization_receipt_sha256:
        blockers.append("missing --launch-authorization-receipt-sha256")
    if not args.scheduled_preflight_receipt:
        blockers.append("missing --scheduled-preflight-receipt")
    if not args.preflight_receipt:
        blockers.append("missing --preflight-receipt")
    if not getattr(args, "supervisor_state", None):
        blockers.append("missing --supervisor-state")
    if not getattr(args, "official_canary_receipt", None):
        blockers.append("missing --official-canary-receipt")
    if not args.canary_receipt:
        blockers.append("missing --canary-receipt")
    if args.kind == "production" and not args.canary_receipt_sha256:
        blockers.append("missing --canary-receipt-sha256")
    live_operation = args.submit or args.test_only
    if live_operation and blockers:
        raise RuntimeError(
            "Expected all fail-closed gate inputs before a live scheduler "
            f"operation. Provided blockers: {blockers!r}."
        )

    authorization: dict[str, Any] | None = None
    if (
        args.launch_authorization_receipt
        and args.launch_authorization_receipt_sha256
    ):
        authorization = validate_launch_authorization_receipt(
            args.launch_authorization_receipt,
            expected_sha256=args.launch_authorization_receipt_sha256,
            metadata=metadata,
            runtime_preflight=runtime_preflight,
            runner=runner,
        )
    scheduled: tuple[Path, str] | None = None
    if args.scheduled_preflight_receipt:
        scheduled_source = Path(
            args.scheduled_preflight_receipt
        ).expanduser().resolve()
        if scheduled_source.is_file() or (
            live_operation and args.kind == "canary"
        ):
            scheduled = ensure_scheduled_preflight_receipt(
                scheduled_source,
                metadata=metadata,
                runtime_preflight=runtime_preflight,
                create=live_operation and args.kind == "canary",
                runner=runner,
            )
        else:
            blockers.append(
                "scheduled-run preflight receipt does not exist"
            )
    input_binding: dict[str, Any] | None = None
    launch_contract: dict[str, Any] | None = None
    preflight_value: dict[str, Any] | None = None
    if scheduled is not None and authorization is not None:
        append_launch_template_exports(
            canary_command,
            authorization=authorization,
            scheduled_preflight_receipt_path=scheduled[0],
            scheduled_preflight_receipt_sha256=scheduled[1],
            include_canary_gate=False,
        )
        append_launch_template_exports(
            production_command,
            authorization=authorization,
            scheduled_preflight_receipt_path=scheduled[0],
            scheduled_preflight_receipt_sha256=scheduled[1],
            include_canary_gate=True,
        )
        launch_contract = launch_contract_value(
            canary_command_template=canary_command,
            canary_metadata=canary_metadata,
            production_command_template=production_command,
            production_metadata=production_metadata,
            authorization=authorization,
            scheduled_preflight_receipt_path=scheduled[0],
            scheduled_preflight_receipt_sha256=scheduled[1],
        )
        input_binding = immutable_input_binding(
            metadata,
            runtime_preflight,
            scheduled_preflight_receipt_path=scheduled[0],
            scheduled_preflight_receipt_sha256=scheduled[1],
            launch_authorization_receipt_path=authorization[
                "receipt_path"
            ],
            launch_authorization_receipt_sha256=authorization[
                "receipt_sha256"
            ],
            approved_plan=authorization["approved_plan"],
            launch_contract=launch_contract,
        )
        preflight_value = preflight_receipt_value(
            input_binding=input_binding,
            runtime_preflight=runtime_preflight,
        )
    if not args.submit and not args.test_only:
        return {
            "status": "dry_run" if not blockers else "dry_run_blocked",
            "submitted": False,
            "test_only": False,
            "command": command,
            "launch_contract": launch_contract,
            "blockers": blockers,
            "preflight": preflight,
            "preflight_receipt_candidate": preflight_value,
            "input_binding": input_binding,
            "scheduled_preflight_receipt": (
                None
                if scheduled is None
                else {"path": str(scheduled[0]), "sha256": scheduled[1]}
            ),
            "launch_authorization": authorization,
            **public_metadata,
        }

    if blockers:
        raise RuntimeError(
            "Expected all fail-closed gate inputs before a live scheduler "
            f"operation. Provided blockers: {blockers!r}."
        )
    if (
        scheduled is None
        or authorization is None
        or input_binding is None
        or launch_contract is None
        or preflight_value is None
    ):
        raise RuntimeError(
            "Expected both official scheduled-run and successor runtime "
            "preflight bindings before a live scheduler operation."
        )
    preflight_path, preflight_hash = ensure_preflight_receipt(
        args.preflight_receipt,
        expected_value=preflight_value,
        create=True,
    )
    gate_binding = canary_gate_binding(
        input_binding=input_binding,
        preflight_receipt_path=preflight_path,
        preflight_receipt_sha256=preflight_hash,
    )
    canary_command = materialize_launch_command(
        canary_command,
        preflight_receipt_path=preflight_path,
        preflight_receipt_sha256=preflight_hash,
    )
    production_command = materialize_launch_command(
        production_command,
        preflight_receipt_path=preflight_path,
        preflight_receipt_sha256=preflight_hash,
    )
    command = (
        canary_command if args.kind == "canary" else production_command
    )
    verified_canary: dict[str, Any] | None = None
    if args.kind == "production":
        from experiments.verify_mnist_conv_perfectdiode_successor_canary import (
            validate_canary_gate_receipt,
        )

        observed_canary_hash = sha256_file(args.canary_receipt)
        if observed_canary_hash != args.canary_receipt_sha256:
            raise RuntimeError(
                "Expected the immutable canary receipt SHA-256 to match before "
                f"production. Provided value: expected="
                f"{args.canary_receipt_sha256!r}, "
                f"observed={observed_canary_hash!r}."
            )
        verified_canary = validate_canary_gate_receipt(
            args.canary_receipt,
            expected_gate_binding=gate_binding,
        )
        production_command = materialize_canary_gate_command(
            production_command,
            canary_receipt_path=args.canary_receipt,
            canary_receipt_sha256=args.canary_receipt_sha256,
        )
        command = production_command
    allocation = _validate_live_allocation(
        allocation_verifier(runner=runner)
    )
    if args.test_only:
        test_command = [command[0], "--test-only", *command[1:]]
        completed = runner(
            test_command,
            check=True,
            text=True,
            capture_output=True,
        )
        return {
            "status": "test_only_passed",
            "submitted": False,
            "test_only": True,
            "command": test_command,
            "slurm_stdout": completed.stdout,
            "slurm_stderr": completed.stderr,
            "login_allocation_audit": allocation,
            "preflight": preflight,
            "preflight_receipt": {
                "path": str(preflight_path),
                "sha256": preflight_hash,
            },
            "scheduled_preflight_receipt": {
                "path": str(scheduled[0]),
                "sha256": scheduled[1],
            },
            "launch_authorization": authorization,
            "gate_binding": gate_binding,
            "canary_gate": verified_canary,
            **public_metadata,
        }

    log_dir = Path(metadata["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    submission_test_command = [command[0], "--test-only", *command[1:]]
    submission_test = runner(
        submission_test_command,
        check=True,
        text=True,
        capture_output=True,
    )
    completed = runner(
        command,
        check=True,
        text=True,
        capture_output=True,
    )
    submitted_job_id = _submitted_job_id(completed)
    submitted_job_audit = verify_submitted_job_contract(
        job_id=submitted_job_id,
        kind=args.kind,
        metadata=metadata,
        runner=runner,
    )
    return {
        "status": "submitted",
        "submitted": True,
        "test_only": False,
        "job_id": submitted_job_id,
        "command": command,
        "submission_test_only": {
            "command": submission_test_command,
            "stdout": submission_test.stdout,
            "stderr": submission_test.stderr,
        },
        "login_allocation_audit": allocation,
        "submitted_job_audit": submitted_job_audit,
        "preflight": preflight,
        "preflight_receipt": {
            "path": str(preflight_path),
            "sha256": preflight_hash,
        },
        "scheduled_preflight_receipt": {
            "path": str(scheduled[0]),
            "sha256": scheduled[1],
        },
        "launch_authorization": authorization,
        "gate_binding": gate_binding,
        "canary_gate": verified_canary,
        **public_metadata,
    }


def _parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("canary", "production"))
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--environment-contract", required=True)
    parser.add_argument(
        "--launch-authorization-receipt",
        help=(
            "External immutable authorization created only after the final "
            "plan binds the already-materialized bundle."
        ),
    )
    parser.add_argument(
        "--launch-authorization-receipt-sha256",
        help="Expected SHA-256 of --launch-authorization-receipt.",
    )
    parser.add_argument(
        "--scheduled-preflight-receipt",
        help=(
            "Required repo skill scheduled-run-preflight-receipt/v1. "
            "For a canary live operation, the exact skill runner creates it "
            "atomically if absent; production requires the existing receipt."
        ),
    )
    parser.add_argument(
        "--preflight-receipt",
        help=(
            "Shared immutable preflight receipt. Required for --test-only "
            "and --submit; created atomically if absent."
        ),
    )
    parser.add_argument(
        "--canary-receipt",
        help=(
            "Passing successor semantic + official gate receipt. Required "
            "for production --test-only and --submit."
        ),
    )
    parser.add_argument(
        "--official-canary-receipt",
        help="Path reserved for the official Jean Zay canary receipt.",
    )
    parser.add_argument(
        "--supervisor-state",
        help="Path reserved for the duplicate-safe persistent supervisor state.",
    )
    parser.add_argument(
        "--canary-receipt-sha256",
        help=(
            "Immutable SHA-256 emitted by the passing canary verifier. "
            "Required with production --test-only and --submit."
        ),
    )
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument("--python", default=PYTHON_EXECUTABLE)
    parser.add_argument(
        "--official-verifier",
        default=None,
        help=(
            "Exact verifier staged on Jean Zay. By default, derive the "
            "reviewed launch_tools path from the environment contract and "
            "--remote-user."
        ),
    )
    parser.add_argument(
        "--remote-user",
        default=os.environ.get("USER", ""),
        help="Jean Zay login used when expanding immutable storage roots.",
    )
    parser.add_argument(
        "--canary-pack-index",
        type=int,
        default=DEFAULT_CANARY_PACK_INDEX,
    )
    parser.add_argument(
        "--command-timeout-seconds",
        type=float,
        default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
        help="Hard timeout for each external validation command.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--test-only",
        action="store_true",
        help="Run the live allocation audit and `sbatch --test-only`.",
    )
    action.add_argument(
        "--submit",
        action="store_true",
        help="Submit one array after all read-only checks pass.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(str(exc), flush=True)
        return 2
    print(
        json.dumps(
            result,
            indent=None if args.submit else 2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
