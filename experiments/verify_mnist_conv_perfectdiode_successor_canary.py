#!/usr/bin/env python3
"""Verify the successor smoke semantics and official Jean Zay canary gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mnist_conv.io import atomic_create_json, read_json
from experiments.submit_mnist_conv_perfectdiode_successor_confirmation_jeanzay import (
    ACCOUNT,
    CONSTRAINT,
    DEFAULT_CANARY_ENTRY_INDICES,
    PARTITION,
    QOS,
    RESULT_JSON_MARKER,
    _canonical_json_sha256,
    _error,
    sha256_file,
)


OFFICIAL_CANARY_VERIFIER = (
    Path(__file__).resolve().with_name("verify_jeanzay_canary.py")
)
OFFICIAL_CANARY_VERIFIER_SHA256 = (
    "032979ace766382941a769db205df1c7fd2576e846e59873645cfe7ad894b5b0"
)
SMOKE_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-successor-smoke-receipt/v1"
)
CANARY_GATE_SCHEMA_VERSION = "perfectdiode-successor-canary-gate/v1"


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise _error(f"{label} to be a lowercase SHA-256 hex string", value)
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise _error(
            f"{label} to be a lowercase SHA-256 hex string", value
        ) from exc
    if value.lower() != value:
        raise _error(f"{label} to be lowercase", value)
    return value


def _artifact_path(
    artifact: Mapping[str, Any],
    *,
    base: Path,
    label: str,
    require_under_base: bool,
) -> Path:
    expected_keys = {"path", "sha256", "bytes"}
    if set(artifact) != expected_keys:
        raise _error(
            f"{label} keys to be exactly {sorted(expected_keys)!r}",
            sorted(artifact),
        )
    raw_path = artifact["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise _error(f"{label}.path to be a non-empty string", raw_path)
    supplied = Path(raw_path).expanduser()
    path = (
        supplied.resolve()
        if supplied.is_absolute()
        else (base / supplied).resolve()
    )
    if require_under_base:
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise _error(f"{label}.path to be under {base}", path) from exc
    if not path.is_file():
        raise _error(f"{label}.path to be an existing file", path)
    size = artifact["bytes"]
    if type(size) is not int or size < 0:
        raise _error(f"{label}.bytes to be a non-negative integer", size)
    if path.stat().st_size != size:
        raise RuntimeError(
            f"Expected {label}.bytes to match the artifact. "
            f"Provided value: recorded={size!r}, observed={path.stat().st_size!r}, "
            f"path={path}."
        )
    expected_hash = _require_sha256(artifact["sha256"], f"{label}.sha256")
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"Expected {label}.sha256 to match the artifact. "
            f"Provided value: recorded={expected_hash!r}, "
            f"observed={observed_hash!r}, path={path}."
        )
    return path


def _artifact_from_value(
    value: Any,
    *,
    base: Path,
    label: str,
    require_under_base: bool,
) -> Path:
    if not isinstance(value, dict):
        raise _error(f"{label} to be an artifact JSON object", value)
    return _artifact_path(
        value,
        base=base,
        label=label,
        require_under_base=require_under_base,
    )


def _validate_lr_vector(value: Any) -> None:
    if isinstance(value, dict):
        values = list(value.values())
    elif isinstance(value, list):
        values = value
    else:
        raise _error("smoke optimizer_step LR vector to be a map or list", value)
    if not values:
        raise _error("smoke optimizer_step LR vector to be non-empty", value)
    for item in values:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) <= 0.0
        ):
            raise _error(
                "every smoke optimizer_step LR to be finite and positive",
                item,
            )


def _validate_live_gate_files(gate_binding: Mapping[str, Any]) -> list[Path]:
    hashes = gate_binding.get("hashes")
    if not isinstance(hashes, dict):
        raise _error("gate binding hashes to be a JSON object", hashes)
    path_hash_pairs = {
        "source_archive": "source_archive_sha256",
        "environment_contract": "environment_contract_sha256",
        "runtime_cli": "runtime_cli_sha256",
        "runtime_module": "runtime_module_sha256",
        "submitter": "submitter_sha256",
        "supervisor": "supervisor_sha256",
        "semantic_canary_verifier": (
            "semantic_canary_verifier_sha256"
        ),
        "experiment_plan_validator": (
            "experiment_plan_validator_sha256"
        ),
        "wrapper": "wrapper_sha256",
        "scheduled_preflight_script": (
            "scheduled_preflight_script_sha256"
        ),
        "official_canary_verifier": (
            "official_canary_verifier_sha256"
        ),
    }
    validated: list[Path] = []
    for path_key, hash_key in path_hash_pairs.items():
        raw_path = gate_binding.get(path_key)
        if not isinstance(raw_path, str) or not raw_path:
            raise _error(
                f"gate binding {path_key} to be a non-empty path",
                raw_path,
            )
        supplied = Path(raw_path).expanduser().absolute()
        if supplied.is_symlink():
            raise _error(
                f"gate binding {path_key} not to be a symlink", supplied
            )
        path = supplied.resolve()
        expected = _require_sha256(
            hashes.get(hash_key), f"gate binding hashes.{hash_key}"
        )
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(
                "Expected every live launch-control file to retain its "
                f"canary-bound hash. Provided value: key={path_key!r}, "
                f"path={path}, expected_sha256={expected!r}."
            )
        validated.append(path)
    plan = gate_binding.get("approved_plan")
    if not isinstance(plan, dict) or set(plan) != {"path", "sha256"}:
        raise _error(
            "gate binding approved_plan to contain path and sha256", plan
        )
    plan_path = Path(str(plan["path"])).expanduser().absolute()
    if (
        plan_path.is_symlink()
        or not plan_path.resolve().is_file()
        or sha256_file(plan_path.resolve())
        != _require_sha256(plan["sha256"], "approved plan sha256")
    ):
        raise RuntimeError(
            "Expected the approved plan file to retain its gate-bound hash. "
            f"Provided value: {plan!r}."
        )
    validated.append(plan_path.resolve())
    return validated


def _validate_embedded_preflight(
    value: Any,
    *,
    gate_binding: Mapping[str, Any],
) -> tuple[Path, Path]:
    if not isinstance(value, dict) or set(value) != {
        "source_path",
        "source_file_sha256",
        "payload_sha256",
        "payload",
        "launch_authorization",
    }:
        raise _error(
            "smoke preflight_receipt to contain the exact embedded receipt "
            "and launch-authorization binding",
            value,
        )
    expected_path = str(
        Path(str(gate_binding["preflight_receipt_path"]))
        .expanduser()
        .resolve()
    )
    expected_sha = _require_sha256(
        gate_binding.get("preflight_receipt_sha256"),
        "gate binding preflight_receipt_sha256",
    )
    if (
        value.get("source_path") != expected_path
        or value.get("source_file_sha256") != expected_sha
    ):
        raise RuntimeError(
            "Expected the embedded preflight source identity to match the "
            f"canary gate. Provided value: {value!r}."
        )
    payload = value.get("payload")
    if not isinstance(payload, dict):
        raise _error("embedded preflight payload to be a JSON object", payload)
    if value.get("payload_sha256") != _canonical_json_sha256(payload):
        raise RuntimeError(
            "Expected the embedded preflight payload hash to match its "
            f"payload. Provided value: {value.get('payload_sha256')!r}."
        )
    source = Path(expected_path).expanduser().absolute()
    if (
        source.is_symlink()
        or not source.resolve().is_file()
        or sha256_file(source.resolve()) != expected_sha
        or read_json(source.resolve()) != payload
    ):
        raise RuntimeError(
            "Expected the live preflight receipt file to equal its embedded "
            f"hash-bound payload. Provided value: {source}."
        )
    expected_input = {
        key: item
        for key, item in gate_binding.items()
        if key
        not in {"preflight_receipt_path", "preflight_receipt_sha256"}
    }
    if (
        payload.get("input_binding") != expected_input
        or payload.get("input_binding_sha256")
        != _canonical_json_sha256(expected_input)
    ):
        raise RuntimeError(
            "Expected the embedded preflight payload to retain the exact "
            "current launch-pair gate binding."
        )
    authorization = value.get("launch_authorization")
    if not isinstance(authorization, dict) or set(authorization) != {
        "source_path",
        "source_file_sha256",
        "payload_sha256",
        "payload",
    }:
        raise _error(
            "embedded launch_authorization to contain exact source and "
            "payload identities",
            authorization,
        )
    auth_path = str(
        Path(str(gate_binding["launch_authorization_receipt_path"]))
        .expanduser()
        .resolve()
    )
    auth_sha = _require_sha256(
        gate_binding.get("launch_authorization_receipt_sha256"),
        "gate binding launch_authorization_receipt_sha256",
    )
    auth_payload = authorization.get("payload")
    if (
        authorization.get("source_path") != auth_path
        or authorization.get("source_file_sha256") != auth_sha
        or not isinstance(auth_payload, dict)
        or authorization.get("payload_sha256")
        != _canonical_json_sha256(auth_payload)
    ):
        raise RuntimeError(
            "Expected the embedded launch authorization to retain the exact "
            f"canary gate identity. Provided value: {authorization!r}."
        )
    live_auth = Path(auth_path).expanduser().absolute()
    if (
        live_auth.is_symlink()
        or not live_auth.resolve().is_file()
        or sha256_file(live_auth.resolve()) != auth_sha
        or read_json(live_auth.resolve()) != auth_payload
    ):
        raise RuntimeError(
            "Expected the live launch authorization to equal its embedded "
            f"hash-bound payload. Provided value: {live_auth}."
        )
    return source.resolve(), live_auth.resolve()


def validate_smoke_receipt(
    receipt_path: str | Path,
    *,
    output_root: str | Path,
    gate_binding: Mapping[str, Any],
    expected_array_job_id: str,
    expected_entry_indices: Sequence[int] = DEFAULT_CANARY_ENTRY_INDICES,
) -> dict[str, Any]:
    """Validate all successor-specific scientific and artifact semantics."""

    source = Path(receipt_path).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        raise _error("canary output root to be an existing directory", root)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise _error("smoke receipt to be under the canary output root", source) from exc
    value = read_json(source)
    if not isinstance(value, dict):
        raise _error("smoke receipt to be a JSON object", value)
    if value.get("schema_version") != SMOKE_SCHEMA_VERSION:
        raise _error(
            f"smoke schema_version to be {SMOKE_SCHEMA_VERSION!r}",
            value.get("schema_version"),
        )
    expected_top_keys = {
        "schema_version",
        "status",
        "passed",
        "official_test_read",
        "bundle_id",
        "bundle_manifest_sha256",
        "config_sha256",
        "config_file_sha256",
        "expected_tk_security_count",
        "slurm_identity",
        "pack_index",
        "entry_indices",
        "runs_per_gpu",
        "concurrent",
        "tk_security",
        "training_canaries",
        "child_processes",
        "timing_memory_evidence",
        "execution_source",
        "preflight_receipt",
        "input_artifacts",
        "output_artifacts",
        "smoke_id",
    }
    if set(value) != expected_top_keys:
        raise _error(
            f"smoke receipt keys to be exactly {sorted(expected_top_keys)!r}",
            sorted(value),
        )
    checks = {
        "status": (value.get("status"), "passed"),
        "passed": (value.get("passed"), True),
        "official_test_read": (value.get("official_test_read"), False),
        "expected_tk_security_count": (
            value.get("expected_tk_security_count"),
            6,
        ),
        "pack_index": (value.get("pack_index"), 4),
        "entry_indices": (value.get("entry_indices"), [8, 9]),
        "runs_per_gpu": (value.get("runs_per_gpu"), 2),
        "concurrent": (value.get("concurrent"), True),
    }
    mismatches = {
        key: {"provided": provided, "expected": expected}
        for key, (provided, expected) in checks.items()
        if provided != expected
    }
    if mismatches:
        raise RuntimeError(
            "Expected the smoke receipt to pass the selected real-step canary. "
            f"Provided mismatches: {mismatches!r}."
        )
    if not isinstance(value.get("bundle_id"), str) or not value["bundle_id"]:
        raise _error(
            "smoke bundle_id to be a non-empty string", value.get("bundle_id")
        )
    body = {key: item for key, item in value.items() if key != "smoke_id"}
    expected_smoke_id = "pdconfirmsmoke_" + _canonical_json_sha256(body)
    if value.get("smoke_id") != expected_smoke_id:
        raise RuntimeError(
            "Expected smoke_id to bind the complete semantic receipt body. "
            f"Provided value: expected={expected_smoke_id!r}, "
            f"observed={value.get('smoke_id')!r}."
        )
    for key in (
        "bundle_manifest_sha256",
        "config_sha256",
        "config_file_sha256",
    ):
        _require_sha256(value.get(key), f"smoke {key}")

    binding_checks = {
        "bundle_id": value.get("bundle_id"),
        "bundle_manifest_sha256": value.get("bundle_manifest_sha256"),
        "config_sha256": value.get("config_sha256"),
        "config_file_sha256": value.get("config_file_sha256"),
    }
    for key, observed in binding_checks.items():
        if gate_binding.get(key) != observed:
            raise RuntimeError(
                "Expected smoke identity to match the immutable preflight "
                f"binding. Provided value: key={key!r}, "
                f"preflight={gate_binding.get(key)!r}, smoke={observed!r}."
            )
    if not isinstance(expected_array_job_id, str) or not (
        expected_array_job_id.isdigit()
    ):
        raise _error(
            "expected canary array job id to contain only decimal digits",
            expected_array_job_id,
        )
    slurm_identity = value.get("slurm_identity")
    expected_identity_keys = {
        "job_id",
        "array_job_id",
        "array_task_id",
        "account",
        "partition",
        "qos",
        "constraint",
    }
    if (
        not isinstance(slurm_identity, dict)
        or set(slurm_identity) != expected_identity_keys
    ):
        raise _error(
            "smoke slurm_identity to contain the exact seven live fields",
            slurm_identity,
        )
    expected_identity = {
        "array_job_id": expected_array_job_id,
        "array_task_id": "0",
        "account": ACCOUNT,
        "partition": PARTITION,
        "qos": QOS,
        "constraint": CONSTRAINT,
    }
    identity_mismatches = {
        key: {"expected": expected, "provided": slurm_identity.get(key)}
        for key, expected in expected_identity.items()
        if slurm_identity.get(key) != expected
    }
    task_job_id = slurm_identity.get("job_id")
    if not isinstance(task_job_id, str) or not task_job_id.isdigit():
        identity_mismatches["job_id"] = {
            "expected": "a decimal live task job id",
            "provided": task_job_id,
        }
    if identity_mismatches:
        raise RuntimeError(
            "Expected smoke Slurm identity to match the just-submitted "
            f"canary. Provided mismatches: {identity_mismatches!r}."
        )
    expected_source = (
        root
        / "smoke"
        / str(value["bundle_id"])
        / "jobs"
        / f"{expected_array_job_id}_{task_job_id}_0"
        / "receipt.json"
    ).resolve()
    if source != expected_source:
        raise RuntimeError(
            "Expected the smoke receipt to occupy the exact current-job "
            f"path. Provided value: expected={expected_source}, "
            f"observed={source}."
        )
    preflight_path, authorization_path = _validate_embedded_preflight(
        value.get("preflight_receipt"),
        gate_binding=gate_binding,
    )

    tk_security = value.get("tk_security")
    if not isinstance(tk_security, list) or len(tk_security) != 6:
        raise _error(
            "smoke tk_security to contain exactly six fresh records",
            tk_security,
        )
    row_ids: set[str] = set()
    tk_pairs: list[tuple[int, int]] = []
    smoke_root = source.parent
    bundle_root = Path(str(gate_binding["bundle_dir"])).expanduser().resolve()
    semantic_artifacts: list[Path] = [
        source,
        preflight_path,
        authorization_path,
        *_validate_live_gate_files(gate_binding),
    ]
    for index, row in enumerate(tk_security):
        if not isinstance(row, dict):
            raise _error(f"tk_security[{index}] to be a JSON object", row)
        required = {
            "row_id",
            "architecture",
            "scheme",
            "T",
            "K",
            "reference_T",
            "reference_K",
            "security_passed",
            "fresh_replay",
            "result_artifact",
        }
        if set(row) != required:
            raise _error(
                f"tk_security[{index}] keys to be exactly {sorted(required)!r}",
                sorted(row),
            )
        row_id = row["row_id"]
        if not isinstance(row_id, str) or not row_id or row_id in row_ids:
            raise _error(
                "each tk_security row_id to be unique and non-empty", row_id
            )
        row_ids.add(row_id)
        pair = (row["T"], row["K"])
        if pair not in {(4, 4), (6, 6)}:
            raise _error(
                "each T/K security row to use 4/4 or 6/6", pair
            )
        tk_pairs.append(pair)
        if (
            row["reference_T"] != 64
            or row["reference_K"] != 64
            or row["security_passed"] is not True
            or row["fresh_replay"] is not True
        ):
            raise RuntimeError(
                "Expected every fresh T/K row to pass against 64/64. "
                f"Provided value: {row!r}."
            )
        semantic_artifacts.append(
            _artifact_from_value(
                row["result_artifact"],
                base=smoke_root,
                label=f"tk_security[{index}].result_artifact",
                require_under_base=True,
            )
        )
    if sorted(tk_pairs) != [(4, 4)] * 3 + [(6, 6)] * 3:
        raise RuntimeError(
            "Expected three Conv1 4/4 and three Conv2 6/6 T/K rows. "
            f"Provided value: {tk_pairs!r}."
        )

    if list(expected_entry_indices) != list(DEFAULT_CANARY_ENTRY_INDICES):
        raise _error(
            "expected canary entry indices to remain the frozen pair [8, 9]",
            list(expected_entry_indices),
        )
    expected_training_identities = (
        {
            "entry_index": 8,
            "entry_id": "pdconfirm_conv2_ours_sgd_seed0",
            "architecture": "conv2",
            "scheme": "ours",
            "optimizer": "sgd",
        },
        {
            "entry_index": 9,
            "entry_id": "pdconfirm_conv2_ours_adam_seed0",
            "architecture": "conv2",
            "scheme": "ours",
            "optimizer": "adam",
        },
    )
    required_training = {
        "entry_index",
        "entry_id",
        "architecture",
        "scheme",
        "optimizer",
        "mode",
        "status",
        "training_completed",
        "completed_steps",
        "expected_steps",
        "epochs_completed",
        "expected_epochs",
        "official_test_read",
        "fresh_restart_from_shared_initialization",
        "raw_learning_rates_by_parameter",
        "benchmark",
        "output_artifacts",
    }
    required_benchmark = {
        "elapsed_seconds",
        "successful_steps_per_second",
        "cuda_peak_memory_allocated_bytes",
        "cuda_peak_memory_reserved_bytes",
    }
    expected_names = {
        "best_validation.pt",
        "final.pt",
        "result.json",
        "run_spec.json",
        "step_log.csv",
        "validation.json",
    }

    def positive_finite(raw: Any, label: str) -> float:
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) <= 0.0
        ):
            raise _error(f"{label} to be finite and positive", raw)
        return float(raw)

    trainings = value.get("training_canaries")
    if not isinstance(trainings, list) or len(trainings) != 2:
        raise _error(
            "smoke training_canaries to contain exactly entries 8 and 9",
            trainings,
        )
    training_by_index: dict[int, dict[str, Any]] = {}
    for position, (training, identity) in enumerate(
        zip(trainings, expected_training_identities, strict=True)
    ):
        if not isinstance(training, dict) or set(training) != required_training:
            raise _error(
                f"smoke training_canaries[{position}] keys to be exactly "
                f"{sorted(required_training)!r}",
                sorted(training) if isinstance(training, dict) else training,
            )
        training_checks = {
            **identity,
            "mode": "successor_canary",
            "status": "complete",
            "training_completed": True,
            "completed_steps": 3438,
            "expected_steps": 3438,
            "epochs_completed": 1,
            "expected_epochs": 1,
            "official_test_read": False,
            "fresh_restart_from_shared_initialization": True,
        }
        training_mismatches = {
            key: {"expected": expected, "provided": training.get(key)}
            for key, expected in training_checks.items()
            if training.get(key) != expected
        }
        if training_mismatches:
            raise RuntimeError(
                "Expected both real paired smoke children to complete one "
                f"full 3,438-step epoch. Provided value: position={position}, "
                f"mismatches={training_mismatches!r}."
            )
        _validate_lr_vector(training["raw_learning_rates_by_parameter"])
        benchmark = training["benchmark"]
        if not isinstance(benchmark, dict) or set(benchmark) != required_benchmark:
            raise _error(
                f"training_canaries[{position}].benchmark keys to be exactly "
                f"{sorted(required_benchmark)!r}",
                benchmark,
            )
        elapsed = positive_finite(
            benchmark["elapsed_seconds"],
            f"training_canaries[{position}].benchmark.elapsed_seconds",
        )
        throughput = positive_finite(
            benchmark["successful_steps_per_second"],
            "training_canaries"
            f"[{position}].benchmark.successful_steps_per_second",
        )
        if not math.isclose(
            throughput,
            3438.0 / elapsed,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "Expected each child benchmark throughput to equal successful "
                f"steps divided by elapsed time. Provided value: {benchmark!r}."
            )
        allocated = benchmark["cuda_peak_memory_allocated_bytes"]
        reserved = benchmark["cuda_peak_memory_reserved_bytes"]
        if (
            type(allocated) is not int
            or allocated <= 0
            or (
                reserved is not None
                and (type(reserved) is not int or reserved < allocated)
            )
        ):
            raise _error(
                "child CUDA peak memory to have positive allocated bytes and "
                "null or at-least-allocated reserved bytes",
                benchmark,
            )
        training_artifacts = training["output_artifacts"]
        if (
            not isinstance(training_artifacts, list)
            or len(training_artifacts) != 6
        ):
            raise _error(
                f"training_canaries[{position}].output_artifacts to contain "
                "exactly six production-path artifacts",
                training_artifacts,
            )
        training_paths = [
            artifact.get("path") if isinstance(artifact, dict) else None
            for artifact in training_artifacts
        ]
        if training_paths != sorted(training_paths):
            raise RuntimeError(
                "Expected each training canary output artifact list to be "
                f"path-sorted. Provided value: {training_paths!r}."
            )
        observed_names: set[str] = set()
        artifact_parents: set[Path] = set()
        for artifact_index, artifact in enumerate(training_artifacts):
            artifact_path = _artifact_from_value(
                artifact,
                base=smoke_root,
                label=(
                    f"training_canaries[{position}].output_artifacts"
                    f"[{artifact_index}]"
                ),
                require_under_base=True,
            )
            semantic_artifacts.append(artifact_path)
            observed_names.add(artifact_path.name)
            artifact_parents.add(artifact_path.parent)
        if observed_names != expected_names or len(artifact_parents) != 1:
            raise RuntimeError(
                "Expected each paired training child to publish the exact six "
                "co-located production-path artifacts. Provided value: "
                f"names={sorted(observed_names)!r}, "
                f"parents={sorted(str(item) for item in artifact_parents)!r}."
            )
        training_by_index[identity["entry_index"]] = training

    required_child = {
        "label",
        "entry_index",
        "entry_id",
        "command",
        "returncode",
        "process_elapsed_seconds",
        "terminal_result",
        "stdout_log",
        "stderr_log",
        "receipt_artifact",
    }
    required_terminal = {
        "status",
        "entry_index",
        "entry_id",
        "receipt_path",
        "receipt_sha256",
        "completed_steps",
        "benchmark",
    }
    children = value.get("child_processes")
    if not isinstance(children, list) or len(children) != 2:
        raise _error(
            "smoke child_processes to contain exactly entries 8 and 9",
            children,
        )
    child_labels: set[str] = set()
    for position, (child, identity) in enumerate(
        zip(children, expected_training_identities, strict=True)
    ):
        if not isinstance(child, dict) or set(child) != required_child:
            raise _error(
                f"child_processes[{position}] keys to be exactly "
                f"{sorted(required_child)!r}",
                sorted(child) if isinstance(child, dict) else child,
            )
        mismatches = {
            key: {"expected": expected, "provided": child.get(key)}
            for key, expected in identity.items()
            if key in {"entry_index", "entry_id"}
            and child.get(key) != expected
        }
        if child.get("returncode") != 0:
            mismatches["returncode"] = {
                "expected": 0,
                "provided": child.get("returncode"),
            }
        label = child.get("label")
        if (
            not isinstance(label, str)
            or not label
            or label in child_labels
        ):
            mismatches["label"] = {
                "expected": "a unique non-empty child label",
                "provided": label,
            }
        else:
            child_labels.add(label)
        positive_finite(
            child.get("process_elapsed_seconds"),
            f"child_processes[{position}].process_elapsed_seconds",
        )
        command = child.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
            or "run-canary-entry" not in command
            or "--entry-index" not in command
            or str(identity["entry_index"]) not in command
        ):
            mismatches["command"] = {
                "expected": (
                    "a non-empty string list invoking run-canary-entry for "
                    f"entry {identity['entry_index']}"
                ),
                "provided": command,
            }
        if mismatches:
            raise RuntimeError(
                "Expected each canary child process to bind the exact paired "
                f"entry and exit successfully. Provided value: {mismatches!r}."
            )
        receipt_path = _artifact_from_value(
            child["receipt_artifact"],
            base=smoke_root,
            label=f"child_processes[{position}].receipt_artifact",
            require_under_base=True,
        )
        stdout_path = _artifact_from_value(
            child["stdout_log"],
            base=smoke_root,
            label=f"child_processes[{position}].stdout_log",
            require_under_base=True,
        )
        stderr_path = _artifact_from_value(
            child["stderr_log"],
            base=smoke_root,
            label=f"child_processes[{position}].stderr_log",
            require_under_base=True,
        )
        semantic_artifacts.extend((receipt_path, stdout_path, stderr_path))
        stderr_text = stderr_path.read_text(
            encoding="utf-8", errors="replace"
        )
        if re.search(
            r"(?:^|\n)\s*(?:Traceback|RuntimeError|Exception|Error:)|"
            r"out of memory|oom[-_ ]?kill|segmentation fault|killed",
            stderr_text,
            flags=re.IGNORECASE,
        ):
            raise RuntimeError(
                "Expected each paired child stderr to contain no fatal "
                f"diagnostic. Provided value: path={stderr_path}, "
                f"tail={stderr_text[-4000:]!r}."
            )
        stdout_lines = [
            line
            for line in stdout_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if line.strip()
        ]
        marked = [
            line[len(RESULT_JSON_MARKER) :]
            for line in stdout_lines
            if line.startswith(RESULT_JSON_MARKER)
        ]
        if (
            len(marked) != 1
            or not stdout_lines
            or not stdout_lines[-1].startswith(RESULT_JSON_MARKER)
        ):
            raise RuntimeError(
                "Expected each child stdout to contain exactly one terminal "
                f"{RESULT_JSON_MARKER!r} marker. Provided value: "
                f"path={stdout_path}."
            )
        try:
            logged_terminal = json.loads(marked[0])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Expected each child terminal marker to contain valid JSON. "
                f"Provided value: {marked[0]!r}."
            ) from exc
        terminal = child.get("terminal_result")
        if (
            not isinstance(terminal, dict)
            or set(terminal) != required_terminal
            or logged_terminal != terminal
        ):
            raise _error(
                f"child_processes[{position}].terminal_result to equal the "
                "exact logged terminal object",
                terminal,
            )
        terminal_expected = {
            "entry_index": identity["entry_index"],
            "entry_id": identity["entry_id"],
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "completed_steps": 3438,
            "benchmark": training_by_index[
                identity["entry_index"]
            ]["benchmark"],
        }
        terminal_mismatches = {
            key: {"expected": expected, "provided": terminal.get(key)}
            for key, expected in terminal_expected.items()
            if terminal.get(key) != expected
        }
        if terminal.get("status") not in {"passed", "already_passed"}:
            terminal_mismatches["status"] = {
                "expected": "passed or already_passed",
                "provided": terminal.get("status"),
            }
        if terminal_mismatches:
            raise RuntimeError(
                "Expected each child terminal result to bind its current "
                f"receipt, benchmark, and completed epoch. Provided value: "
                f"{terminal_mismatches!r}."
            )

    required_thresholds = {
        "gpu_memory_capacity_bytes": 34_359_738_368,
        "maximum_combined_peak_fraction": 0.8,
        "maximum_combined_peak_reserved_bytes": 27_487_790_694,
        "maximum_projected_child_duration_seconds": 28_800,
    }
    required_timing_keys = {
        "thresholds",
        "children",
        "combined_peak_memory_allocated_bytes",
        "combined_peak_memory_reserved_bytes",
        "combined_peak_memory_fraction",
        "concurrent_training_wall_elapsed_seconds",
        "aggregate_completed_steps",
        "aggregate_successful_steps_per_second",
        "maximum_projected_20_epoch_duration_seconds",
        "checks",
        "passed",
    }
    required_child_evidence = {
        "entry_index",
        "entry_id",
        "completed_steps",
        "elapsed_seconds",
        "successful_steps_per_second",
        "cuda_peak_memory_allocated_bytes",
        "cuda_peak_memory_reserved_bytes",
        "projected_20_epoch_duration_seconds",
    }
    timing = value.get("timing_memory_evidence")
    if not isinstance(timing, dict) or set(timing) != required_timing_keys:
        raise _error(
            "timing_memory_evidence keys to be exactly "
            f"{sorted(required_timing_keys)!r}",
            timing,
        )
    if timing.get("thresholds") != required_thresholds:
        raise RuntimeError(
            "Expected timing/memory evidence to retain the exact V100 and "
            f"eight-hour safety thresholds. Provided value: "
            f"{timing.get('thresholds')!r}."
        )
    timing_children = timing.get("children")
    if not isinstance(timing_children, list) or len(timing_children) != 2:
        raise _error(
            "timing_memory_evidence.children to contain entries 8 and 9",
            timing_children,
        )
    child_peaks: list[int] = []
    child_reserved_peaks: list[int] = []
    projected_durations: list[float] = []
    child_elapsed: list[float] = []
    for position, (evidence, identity) in enumerate(
        zip(timing_children, expected_training_identities, strict=True)
    ):
        if (
            not isinstance(evidence, dict)
            or set(evidence) != required_child_evidence
        ):
            raise _error(
                f"timing children[{position}] keys to be exactly "
                f"{sorted(required_child_evidence)!r}",
                evidence,
            )
        benchmark = training_by_index[identity["entry_index"]]["benchmark"]
        expected_fields = {
            "entry_index": identity["entry_index"],
            "entry_id": identity["entry_id"],
            "completed_steps": 3438,
            "elapsed_seconds": benchmark["elapsed_seconds"],
            "successful_steps_per_second": benchmark[
                "successful_steps_per_second"
            ],
            "cuda_peak_memory_allocated_bytes": benchmark[
                "cuda_peak_memory_allocated_bytes"
            ],
            "cuda_peak_memory_reserved_bytes": benchmark[
                "cuda_peak_memory_reserved_bytes"
            ],
        }
        evidence_mismatches = {
            key: {"expected": expected, "provided": evidence.get(key)}
            for key, expected in expected_fields.items()
            if evidence.get(key) != expected
        }
        elapsed = positive_finite(
            evidence.get("elapsed_seconds"),
            f"timing children[{position}].elapsed_seconds",
        )
        positive_finite(
            evidence.get("successful_steps_per_second"),
            f"timing children[{position}].successful_steps_per_second",
        )
        projected = positive_finite(
            evidence.get("projected_20_epoch_duration_seconds"),
            "timing children"
            f"[{position}].projected_20_epoch_duration_seconds",
        )
        if not math.isclose(
            projected,
            elapsed * 20.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            evidence_mismatches[
                "projected_20_epoch_duration_seconds"
            ] = {
                "expected": elapsed * 20.0,
                "provided": projected,
            }
        if evidence_mismatches:
            raise RuntimeError(
                "Expected per-child timing/memory evidence to bind the "
                f"validated benchmark. Provided value: "
                f"{evidence_mismatches!r}."
            )
        child_elapsed.append(elapsed)
        child_peaks.append(
            int(evidence["cuda_peak_memory_allocated_bytes"])
        )
        child_reserved_peaks.append(
            int(evidence["cuda_peak_memory_reserved_bytes"])
        )
        projected_durations.append(projected)

    combined_peak = sum(child_peaks)
    combined_reserved_peak = sum(child_reserved_peaks)
    capacity = required_thresholds["gpu_memory_capacity_bytes"]
    combined_fraction = combined_reserved_peak / capacity
    wall_elapsed = positive_finite(
        timing.get("concurrent_training_wall_elapsed_seconds"),
        "timing_memory_evidence.concurrent_training_wall_elapsed_seconds",
    )
    aggregate_throughput = positive_finite(
        timing.get("aggregate_successful_steps_per_second"),
        "timing_memory_evidence.aggregate_successful_steps_per_second",
    )
    maximum_projected = max(projected_durations)
    timing_mismatches: dict[str, Any] = {}
    exact_timing = {
        "combined_peak_memory_allocated_bytes": combined_peak,
        "combined_peak_memory_reserved_bytes": combined_reserved_peak,
        "aggregate_completed_steps": 6876,
    }
    timing_mismatches.update(
        {
            key: {"expected": expected, "provided": timing.get(key)}
            for key, expected in exact_timing.items()
            if timing.get(key) != expected
        }
    )
    for key, observed, expected in (
        (
            "combined_peak_memory_fraction",
            timing.get("combined_peak_memory_fraction"),
            combined_fraction,
        ),
        (
            "aggregate_successful_steps_per_second",
            aggregate_throughput,
            6876.0 / wall_elapsed,
        ),
        (
            "maximum_projected_20_epoch_duration_seconds",
            timing.get("maximum_projected_20_epoch_duration_seconds"),
            maximum_projected,
        ),
    ):
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or not math.isclose(
                float(observed),
                float(expected),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            timing_mismatches[key] = {
                "expected": expected,
                "provided": observed,
            }
    if wall_elapsed < max(child_elapsed):
        timing_mismatches["concurrent_training_wall_elapsed_seconds"] = {
            "expected": (
                "at least the longest validated concurrent child duration"
            ),
            "provided": wall_elapsed,
        }
    required_checks = {
        "child_evidence_positive": True,
        "combined_peak_within_limit": True,
        "projected_duration_below_limit": True,
    }
    if timing.get("checks") != required_checks:
        timing_mismatches["checks"] = {
            "expected": required_checks,
            "provided": timing.get("checks"),
        }
    if timing.get("passed") is not True:
        timing_mismatches["passed"] = {
            "expected": True,
            "provided": timing.get("passed"),
        }
    if (
        combined_reserved_peak
        >= capacity
        * required_thresholds["maximum_combined_peak_fraction"]
        or combined_reserved_peak
        > required_thresholds["maximum_combined_peak_reserved_bytes"]
    ):
        timing_mismatches["combined_peak_safety"] = {
            "expected": (
                "summed reserved peaks strictly below 80% of a 32-GiB V100"
            ),
            "provided": combined_reserved_peak,
        }
    if maximum_projected >= required_thresholds[
        "maximum_projected_child_duration_seconds"
    ]:
        timing_mismatches["projected_duration_safety"] = {
            "expected": "strictly below 28,800 seconds",
            "provided": maximum_projected,
        }
    if timing_mismatches:
        raise RuntimeError(
            "Expected paired canary timing/memory evidence to prove both "
            "ordinary processes fit concurrently and project below the "
            f"eight-hour bound. Provided mismatches: {timing_mismatches!r}."
        )

    if not isinstance(value.get("execution_source"), dict):
        raise _error(
            "smoke execution_source to be a JSON object",
            value.get("execution_source"),
        )

    for collection, base, under_root in (
        ("input_artifacts", bundle_root, True),
        ("output_artifacts", smoke_root, True),
    ):
        artifacts = value.get(collection)
        if not isinstance(artifacts, list) or not artifacts:
            raise _error(f"smoke {collection} to be a non-empty list", artifacts)
        for index, artifact in enumerate(artifacts):
            semantic_artifacts.append(
                _artifact_from_value(
                    artifact,
                    base=base,
                    label=f"{collection}[{index}]",
                    require_under_base=under_root,
                )
            )
    unique_artifacts = sorted({path.resolve() for path in semantic_artifacts})
    return {
        "smoke_receipt": value,
        "smoke_receipt_path": str(source),
        "smoke_receipt_sha256": sha256_file(source),
        "required_artifacts": [str(path) for path in unique_artifacts],
        "tk_row_ids": sorted(row_ids),
        "tk_pairs": [list(pair) for pair in sorted(tk_pairs)],
        "training_canary_entry_indices": [8, 9],
        "training_canary_child_count": 2,
        "training_canary_total_steps_completed": 6876,
        "timing_memory_thresholds": required_thresholds,
        "timing_memory_evidence": timing,
    }


def _run_official_verifier(
    *,
    job_id: str,
    required_artifacts: Sequence[str],
    official_verifier: Path,
    official_receipt: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    if not job_id.isdigit():
        raise _error("canary job id to contain only decimal digits", job_id)
    if not official_verifier.is_file():
        raise _error(
            "the official jean-zay-pre-submit-gate verifier to exist",
            official_verifier,
        )
    command = [
        sys.executable,
        str(official_verifier),
        "--job-id",
        job_id,
        "--expected-account",
        ACCOUNT,
        "--expected-partition",
        PARTITION,
        "--expected-qos",
        QOS,
        "--expected-constraint",
        CONSTRAINT,
        "--expected-tasks",
        "1",
    ]
    for path in required_artifacts:
        command.extend(["--require-file", path])
    command.extend(["--receipt", str(official_receipt)])
    completed = runner(
        command,
        check=True,
        text=True,
        capture_output=True,
    )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Expected the official canary verifier to print one JSON object. "
            f"Provided value: {completed.stdout!r}."
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "jean-zay-pre-submit-gate/v1"
        or value.get("gate_passed") is not True
    ):
        raise RuntimeError(
            "Expected the official Jean Zay canary gate to pass. "
            f"Provided value: {value!r}."
        )
    persisted = read_json(official_receipt)
    if persisted != value:
        raise RuntimeError(
            "Expected the official canary receipt file to equal verifier "
            f"stdout. Provided value: path={official_receipt}."
        )
    return {
        "command": command,
        "receipt": value,
        "receipt_path": str(official_receipt),
        "receipt_sha256": sha256_file(official_receipt),
    }


def inspect_canary_logs(
    *,
    stdout_log: str | Path,
    stderr_log: str | Path,
    smoke_receipt: str | Path,
) -> dict[str, Any]:
    expected_receipt = str(Path(smoke_receipt).expanduser().resolve())
    records: dict[str, dict[str, Any]] = {}
    texts: dict[str, str] = {}
    for label, raw_path in (
        ("stdout", stdout_log),
        ("stderr", stderr_log),
    ):
        supplied = Path(raw_path).expanduser().absolute()
        if supplied.is_symlink():
            raise _error(f"canary {label} log not to be a symlink", supplied)
        path = supplied.resolve()
        if not path.is_file():
            raise _error(
                f"canary {label} log to be an existing file", path
            )
        text = path.read_text(encoding="utf-8", errors="replace")
        texts[label] = text
        records[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    fatal_pattern = re.compile(
        r"(?:^|\n)\s*(?:Traceback|RuntimeError|Exception|Error:)|"
        r"out of memory|oom[-_ ]?kill|segmentation fault|killed",
        flags=re.IGNORECASE,
    )
    if fatal_pattern.search(texts["stderr"]):
        raise RuntimeError(
            "Expected the completed canary stderr to contain no fatal "
            f"diagnostic. Provided value: path={records['stderr']['path']!r}, "
            f"tail={texts['stderr'][-4000:]!r}."
        )
    stdout_lines = [
        line for line in texts["stdout"].splitlines() if line.strip()
    ]
    marked = [
        line[len(RESULT_JSON_MARKER) :]
        for line in stdout_lines
        if line.startswith(RESULT_JSON_MARKER)
    ]
    if (
        len(marked) != 1
        or not stdout_lines
        or not stdout_lines[-1].startswith(RESULT_JSON_MARKER)
    ):
        raise RuntimeError(
            "Expected canary stdout to contain exactly one terminal "
            f"{RESULT_JSON_MARKER!r} marker. "
            f"Provided value: path={records['stdout']['path']!r}."
        )
    try:
        terminal = json.loads(marked[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Expected the canary stdout result marker to contain valid JSON. "
            f"Provided value: {marked[0]!r}."
        ) from exc
    if not isinstance(terminal, dict):
        raise _error(
            "canary stdout terminal result to be a JSON object", terminal
        )
    if (
        terminal.get("status") not in {"passed", "already_passed"}
        or terminal.get("receipt_path") != expected_receipt
        or terminal.get("receipt_sha256")
        != sha256_file(expected_receipt)
    ):
        raise RuntimeError(
            "Expected canary stdout terminal JSON to bind the current "
            f"job-scoped receipt. Provided value: {terminal!r}."
        )
    return {
        "stdout": records["stdout"],
        "stderr": records["stderr"],
        "terminal_result": terminal,
    }


def _validate_canary_log_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "stdout",
        "stderr",
        "terminal_result",
    }:
        raise _error(
            "canary_logs to contain stdout, stderr, and terminal_result",
            value,
        )
    terminal = value.get("terminal_result")
    if not isinstance(terminal, dict):
        raise _error("canary terminal_result to be a JSON object", terminal)
    for label in ("stdout", "stderr"):
        artifact = value.get(label)
        if not isinstance(artifact, dict):
            raise _error(f"canary_logs.{label} to be an artifact", artifact)
        _artifact_path(
            artifact,
            base=Path("/"),
            label=f"canary_logs.{label}",
            require_under_base=False,
        )
    return dict(value)


def verify_canary(
    *,
    job_id: str,
    smoke_receipt: str | Path,
    output_root: str | Path,
    gate_binding: Mapping[str, Any],
    official_receipt: str | Path,
    receipt: str | Path,
    canary_stdout_log: str | Path,
    canary_stderr_log: str | Path,
    official_verifier: str | Path = OFFICIAL_CANARY_VERIFIER,
    expected_official_verifier_sha256: str = (
        OFFICIAL_CANARY_VERIFIER_SHA256
    ),
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    semantic = validate_smoke_receipt(
        smoke_receipt,
        output_root=output_root,
        gate_binding=gate_binding,
        expected_array_job_id=job_id,
    )
    canary_logs = inspect_canary_logs(
        stdout_log=canary_stdout_log,
        stderr_log=canary_stderr_log,
        smoke_receipt=smoke_receipt,
    )
    verifier = Path(official_verifier).expanduser().resolve()
    observed_verifier_hash = sha256_file(verifier)
    if observed_verifier_hash != expected_official_verifier_sha256:
        raise RuntimeError(
            "Expected the exact reviewed jean-zay-pre-submit-gate verifier. "
            f"Provided value: expected={expected_official_verifier_sha256!r}, "
            f"observed={observed_verifier_hash!r}, path={verifier}."
        )
    official_path = Path(
        os.path.abspath(Path(official_receipt).expanduser())
    )
    final_path = Path(os.path.abspath(Path(receipt).expanduser()))
    if official_path == final_path:
        raise _error(
            "official and successor semantic receipt paths to differ",
            final_path,
        )
    for path, label in (
        (official_path, "official canary receipt"),
        (final_path, "successor semantic receipt"),
    ):
        if path.exists() or path.is_symlink():
            raise _error(
                f"{label} path not to exist before verification",
                path,
            )
    official = _run_official_verifier(
        job_id=job_id,
        required_artifacts=[
            *semantic["required_artifacts"],
            canary_logs["stdout"]["path"],
            canary_logs["stderr"]["path"],
        ],
        official_verifier=verifier,
        official_receipt=official_path,
        runner=runner,
    )
    result = {
        "schema_version": CANARY_GATE_SCHEMA_VERSION,
        "status": "passed",
        "gate_passed": True,
        "job_id": job_id,
        "gate_binding": dict(gate_binding),
        "gate_binding_sha256": _canonical_json_sha256(gate_binding),
        "official_verifier": {
            "path": str(verifier),
            "sha256": observed_verifier_hash,
        },
        "official_gate": official,
        "successor_semantics": semantic,
        "canary_logs": canary_logs,
    }
    atomic_create_json(final_path, result, canonical=True)
    return {
        **result,
        "receipt_path": str(final_path),
        "receipt_sha256": sha256_file(final_path),
    }


def validate_canary_gate_receipt(
    path: str | Path,
    *,
    expected_gate_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed before any production scheduler call."""

    source = Path(os.path.abspath(Path(path).expanduser()))
    if source.is_symlink():
        raise _error(
            "successor canary gate receipt not to be a symlink",
            source,
        )
    source = source.resolve()
    value = read_json(source)
    if not isinstance(value, dict):
        raise _error("successor canary gate receipt to be a JSON object", value)
    if (
        value.get("schema_version") != CANARY_GATE_SCHEMA_VERSION
        or value.get("status") != "passed"
        or value.get("gate_passed") is not True
    ):
        raise RuntimeError(
            "Expected a passing successor canary gate receipt before "
            f"production. Provided value: {value!r}."
        )
    if value.get("gate_binding") != dict(expected_gate_binding):
        raise RuntimeError(
            "Expected the canary receipt to bind the exact current production "
            f"inputs. Provided value: expected={dict(expected_gate_binding)!r}, "
            f"observed={value.get('gate_binding')!r}."
        )
    if value.get("gate_binding_sha256") != _canonical_json_sha256(
        expected_gate_binding
    ):
        raise RuntimeError(
            "Expected the canary gate-binding digest to match current inputs. "
            f"Provided value: {value.get('gate_binding_sha256')!r}."
        )
    official = value.get("official_gate")
    if (
        not isinstance(official, dict)
        or not isinstance(official.get("receipt"), dict)
        or official["receipt"].get("gate_passed") is not True
    ):
        raise RuntimeError(
            "Expected the nested official Jean Zay gate to pass. "
            f"Provided value: {official!r}."
        )
    semantic = value.get("successor_semantics")
    if (
        not isinstance(semantic, dict)
        or semantic.get("training_canary_entry_indices") != [8, 9]
        or semantic.get("training_canary_child_count") != 2
        or semantic.get("training_canary_total_steps_completed") != 6876
        or len(semantic.get("tk_row_ids", [])) != 6
        or semantic.get("timing_memory_thresholds")
        != {
            "gpu_memory_capacity_bytes": 34_359_738_368,
            "maximum_combined_peak_fraction": 0.8,
            "maximum_combined_peak_reserved_bytes": 27_487_790_694,
            "maximum_projected_child_duration_seconds": 28_800,
        }
        or not isinstance(semantic.get("timing_memory_evidence"), dict)
        or semantic["timing_memory_evidence"].get("passed") is not True
    ):
        raise RuntimeError(
            "Expected the nested successor semantic gate to contain six T/K "
            "passes, both full paired training-canary epochs, and passing "
            "timing/memory safety evidence. "
            f"Provided value: {semantic!r}."
        )
    _validate_live_gate_files(expected_gate_binding)
    _validate_canary_log_binding(value.get("canary_logs"))
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--smoke-receipt", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gate-binding", required=True)
    parser.add_argument("--official-receipt", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--canary-stdout-log", required=True)
    parser.add_argument("--canary-stderr-log", required=True)
    parser.add_argument(
        "--official-verifier",
        default=str(OFFICIAL_CANARY_VERIFIER),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        gate_binding = read_json(Path(args.gate_binding).expanduser().resolve())
        if not isinstance(gate_binding, dict):
            raise _error("gate binding to be a JSON object", gate_binding)
        result = verify_canary(
            job_id=args.job_id,
            smoke_receipt=args.smoke_receipt,
            output_root=args.output_root,
            gate_binding=gate_binding,
            official_receipt=args.official_receipt,
            receipt=args.receipt,
            canary_stdout_log=args.canary_stdout_log,
            canary_stderr_log=args.canary_stderr_log,
            official_verifier=args.official_verifier,
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(str(exc), flush=True)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
