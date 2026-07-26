#!/usr/bin/env python3
"""Fail-closed scientific audit for the frozen MNIST Conv LR v3 study.

This tool is deliberately read-only and lives outside the Conv execution-code
fingerprint.  It validates both the immutable artifact chain and the scientific
claims derived from the candidate and selection outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mnist_conv.identity import sha256_file
from experiments.mnist_conv.io import read_json
from experiments.mnist_conv.lr_artifacts import validate_stage_completion
from experiments.mnist_conv.lr_protocol import (
    CandidateRunResult,
    candidate_learning_rate_at_step,
    linear_quantile,
    select_final_candidate,
)
from experiments.mnist_conv.lr_study_spec import (
    LR_RELATIVE_RHO_STUDY_SCHEMA_VERSION,
    LRStudySpec,
)
from experiments.mnist_conv.specs import RunSpec
from labs.datasets import stable_batch_order_hash, stable_index_sequence_hash


EXPECTED_STUDY_ID = (
    "lrstudy_b2cffbb9c9976c58338141fc179fba2ff4f37b3d02009ef9255e3d5a21d650d2"
)
EXPECTED_PROTOCOL_ID = "conv-hardsigmoid-lr-sgd-bs16-relative-rho-v3"
EXPECTED_CODE_FINGERPRINT = (
    "2efe660e439d9a4a3fd1abbe6c8fb60488a11160dbf5814d79b57ae07ac1f55a"
)
ROLES = ("fast", "middle", "high")
STAGES = ("probe", "range", "candidates", "select")
TOTAL_STEPS = 17_190
STEPS_PER_EPOCH = 3_438
WARMUP_STEPS = 860
WARMUP_OBSERVATION_START = 829
EPOCHS = 5
SHA256_RE = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_TEST_METRIC_KEYS = {
    "best_test_accuracy",
    "final_test_accuracy",
    "test_accuracy",
    "best_test_loss",
    "final_test_loss",
    "test_loss",
}
STEP_LOG_FIELDS = (
    "step",
    "epoch",
    "batch_in_epoch",
    "learning_rate",
    "scheduled_rho",
    "scheduled_rho_relative",
    "scheduled_rho_span",
    "loss",
    "loss_ema",
    "accuracy",
    "sample_count",
    "status",
    "failure_reason",
)
PARAMETER_DIAGNOSTIC_FIELDS = (
    "step",
    "learning_rate",
    "scheduled_rho",
    "scheduled_rho_relative",
    "scheduled_rho_span",
    "parameter",
    "parameter_kind",
    "bounded_gate",
    "report_only",
    "gradient_rms",
    "proposed_update_rms",
    "normalized_update",
    "span_normalized_update",
    "initial_parameter_rms",
    "parameter_relative_update",
    "proposed_bound_crossing_fraction",
    "lower_bound_occupancy",
    "upper_bound_occupancy",
    "combined_bound_occupancy",
    "projection_efficiency",
    "proposal_is_numerically_zero",
    "projection_gate_eligible",
)
SELECTION_CSV_FIELDS = (
    "row_id",
    "architecture",
    "scheme",
    "input_gain",
    "inference_iterations",
    "training_iterations",
    "status",
    "reason",
    "selected_candidate_role",
    "selected_peak_learning_rate",
    "selected_rho_target",
    "selected_rho_target_relative",
    "observed_peak_rho",
    "observed_peak_rho_relative",
    "observed_peak_rho_span",
    "final_validation_loss",
    "final_validation_accuracy",
    "median_projection_efficiency",
    "plateau_roles",
    "minimum_final_validation_loss",
)
CALIBRATION_LAYER_SATURATIONS = {
    "conv1_ours_v4_c1": [0.30000024912308676],
    "conv1_legacy_v4_c0p25": [0.29999993771922834],
    "conv2_ours_v4_c1": [0.29999993771922834, 0.04516103316326531],
    "conv2_legacy_v4_c0p25": [0.29999993771922834, 0.5926881128427933],
}


class AuditError(RuntimeError):
    """A scientific or provenance invariant failed."""


def _fail(path: str, expected: str, provided: Any) -> None:
    raise AuditError(
        f"Expected {path} to be {expected}. Provided value: {provided!r}."
    )


def _require(condition: bool, path: str, expected: str, provided: Any) -> None:
    if not condition:
        _fail(path, expected, provided)


def _require_file(path: Path, label: str) -> None:
    _require(
        path.is_file() and not path.is_symlink(),
        label,
        "an existing regular file",
        str(path),
    )


def _finite(value: Any, path: str, *, nonnegative: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(
            f"Expected {path} to be a finite number. Provided value: {value!r}."
        ) from exc
    _require(math.isfinite(number), path, "a finite number", value)
    if nonnegative:
        _require(number >= 0.0, path, "a non-negative finite number", number)
    return number


def _near(observed: Any, expected: Any, path: str) -> None:
    left = _finite(observed, path)
    right = _finite(expected, f"{path}.expected")
    _require(
        math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15),
        path,
        f"numerically equal to {right!r}",
        left,
    )


def _csv_bool(value: str, path: str) -> bool:
    _require(value in {"True", "False"}, path, "'True' or 'False'", value)
    return value == "True"


def _require_sha(value: Any, path: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        path,
        "a lowercase SHA-256 digest",
        value,
    )
    return str(value)


def _checkpoint_tensor_digest(path: Path, label: str) -> str:
    """Decode a versioned function checkpoint and hash its finite tensors."""

    import torch

    _require_file(path, label)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise AuditError(
            f"Expected {label} to be a safely decodable versioned function "
            f"checkpoint. Provided value: {type(exc).__name__}: {exc}."
        ) from exc
    _require(
        isinstance(payload, dict)
        and set(payload) == {"format", "version", "schema", "states"},
        f"{label} payload",
        "an object with exactly format, version, schema, and states",
        None if not isinstance(payload, dict) else sorted(payload),
    )
    assert isinstance(payload, dict)
    _require(
        payload["format"] == "drn.function.parameters"
        and payload["version"] == 1,
        f"{label} format/version",
        "drn.function.parameters version 1",
        {"format": payload["format"], "version": payload["version"]},
    )
    schema = payload["schema"]
    states = payload["states"]
    _require(
        isinstance(schema, list)
        and isinstance(states, list)
        and len(schema) == len(states)
        and len(states) > 0,
        f"{label} schema/state alignment",
        "equally sized non-empty lists",
        {
            "schema_type": type(schema).__name__,
            "state_type": type(states).__name__,
            "schema_count": len(schema) if isinstance(schema, list) else None,
            "state_count": len(states) if isinstance(states, list) else None,
        },
    )
    assert isinstance(schema, list) and isinstance(states, list)
    digest = hashlib.sha256()
    digest.update(b"mnist-conv-parameter-tensors/v1\0")
    digest.update(len(states).to_bytes(8, byteorder="big", signed=False))
    seen_names: set[str] = set()
    for index, (record, state) in enumerate(zip(schema, states)):
        prefix = f"{label}.parameters[{index}]"
        _require(
            isinstance(record, dict)
            and set(record) == {"name", "type", "shape", "dtype"},
            f"{prefix}.schema",
            "an object with exactly name, type, shape, and dtype",
            record,
        )
        assert isinstance(record, dict)
        name = str(record["name"]).strip()
        _require(
            bool(name) and name not in seen_names,
            f"{prefix}.name",
            "a unique non-empty canonical parameter name",
            name,
        )
        seen_names.add(name)
        _require(
            torch.is_tensor(state),
            f"{prefix}.state",
            "a torch.Tensor",
            type(state).__name__,
        )
        assert torch.is_tensor(state)
        expected_shape = [int(value) for value in state.shape]
        _require(
            record["shape"] == expected_shape,
            f"{prefix}.shape",
            repr(expected_shape),
            record["shape"],
        )
        _require(
            record["dtype"] == str(state.dtype),
            f"{prefix}.dtype",
            repr(str(state.dtype)),
            record["dtype"],
        )
        finite = bool(torch.isfinite(state).all().item())
        _require(finite, f"{prefix}.state", "entirely finite", "non-finite")
        if name.startswith(("ConvWeight_", "DenseWeight_")):
            minimum = float(state.min().item())
            maximum = float(state.max().item())
            _require(
                minimum >= 0.0 and maximum <= 100.0,
                f"{prefix}.conductance bounds",
                "all values in [0, 100]",
                {"minimum": minimum, "maximum": maximum},
            )
        tensor = state.detach().cpu().contiguous()
        encoded = (
            name.encode("utf-8"),
            str(record["type"]).encode("utf-8"),
            ",".join(str(int(value)) for value in tensor.shape).encode("ascii"),
            str(tensor.dtype).encode("ascii"),
        )
        for item in encoded:
            digest.update(len(item).to_bytes(8, byteorder="big", signed=False))
            digest.update(item)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_parameter_diagnostics(
    path: Path,
    label: str,
) -> dict[str, dict[str, Any]]:
    """Recompute the frozen scales and occupancies directly from a checkpoint.

    The v3 LR coordinate is defined by the initial tensor RMS, so validating a
    checkpoint digest while trusting the adjacent metadata's RMS would leave
    the central scientific normalization unaudited.  The checkpoint schema has
    already been validated by :func:`_checkpoint_tensor_digest` at each call
    site; this helper independently recomputes the numerical claims.
    """

    import torch

    _require_file(path, label)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise AuditError(
            f"Expected {label} to be a safely decodable versioned function "
            f"checkpoint. Provided value: {type(exc).__name__}: {exc}."
        ) from exc
    schema = payload["schema"]
    states = payload["states"]
    result: dict[str, dict[str, Any]] = {}
    for record, state in zip(schema, states):
        name = str(record["name"]).strip()
        tensor = state.detach().cpu()
        bounded = name.startswith(("ConvWeight_", "DenseWeight_"))
        diagnostic: dict[str, Any] = {
            "name": name,
            "parameter_type": str(record["type"]).rsplit(".", 1)[-1],
            "shape": [int(value) for value in tensor.shape],
            "rms": float(
                torch.sqrt(torch.mean(tensor.to(torch.float64) ** 2)).item()
            ),
            "std": float(tensor.to(torch.float64).std(unbiased=False).item()),
            "bounded_gate": bounded,
            "report_only": not bounded,
        }
        if bounded:
            lower_mask = tensor <= 0.0
            upper_mask = tensor >= 100.0
            diagnostic.update(
                {
                    "lower_bound": 0.0,
                    "upper_bound": 100.0,
                    "lower_bound_occupancy": float(
                        lower_mask.to(torch.float64).mean().item()
                    ),
                    "upper_bound_occupancy": float(
                        upper_mask.to(torch.float64).mean().item()
                    ),
                    "combined_bound_occupancy": float(
                        (lower_mask | upper_mask).to(torch.float64).mean().item()
                    ),
                }
            )
        result[name] = diagnostic
    return result


def _audit_initialization_parameter_diagnostics(
    checkpoint: Path,
    metadata_diagnostics: Any,
    *,
    label: str,
) -> None:
    """Require metadata scales to match values recomputed from checkpoint bytes."""

    _require(
        isinstance(metadata_diagnostics, dict) and metadata_diagnostics,
        f"{label} metadata parameter diagnostics",
        "a non-empty object",
        metadata_diagnostics,
    )
    assert isinstance(metadata_diagnostics, dict)
    observed = _checkpoint_parameter_diagnostics(checkpoint, label)
    _require(
        set(metadata_diagnostics) == set(observed),
        f"{label} metadata parameter names",
        repr(sorted(observed)),
        sorted(metadata_diagnostics),
    )
    for name, recomputed in observed.items():
        provided = metadata_diagnostics[name]
        _require(
            isinstance(provided, dict) and set(provided) == set(recomputed),
            f"{label} metadata diagnostics for {name}",
            f"an object with exactly keys {sorted(recomputed)!r}",
            provided,
        )
        assert isinstance(provided, dict)
        for field in (
            "name",
            "parameter_type",
            "shape",
            "bounded_gate",
            "report_only",
        ):
            _require(
                provided[field] == recomputed[field],
                f"{label} metadata {name}.{field}",
                repr(recomputed[field]),
                provided[field],
            )
        for field in set(recomputed) - {
            "name",
            "parameter_type",
            "shape",
            "bounded_gate",
            "report_only",
        }:
            _near(
                provided[field],
                recomputed[field],
                f"{label} metadata {name}.{field}",
            )


def _walk_mapping_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_mapping_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_mapping_keys(item)


def _require_no_test_metrics(value: Any, path: str) -> None:
    found = sorted(FORBIDDEN_TEST_METRIC_KEYS & set(_walk_mapping_keys(value)))
    _require(found == [], path, "free of official-test metric keys", found)


def _candidate_dir(study_dir: Path, entry_id: str) -> Path:
    return study_dir / "stages" / "candidates" / "entries" / entry_id


def _row_stage_dir(study_dir: Path, stage: str, row_id: str) -> Path:
    return study_dir / "stages" / stage / "entries" / row_id


def _aggregate_observed_rho(
    relative_by_parameter: Mapping[str, Sequence[float]],
    span_values: Sequence[float],
    *,
    path: str,
) -> dict[str, Any]:
    _require(
        bool(relative_by_parameter)
        and all(len(values) == 32 for values in relative_by_parameter.values()),
        f"{path} relative-rho observation window",
        "32 values for every bounded parameter",
        {name: len(values) for name, values in relative_by_parameter.items()},
    )
    expected_span_count = 32 * len(relative_by_parameter)
    _require(
        len(span_values) == expected_span_count,
        f"{path} span-rho observation window",
        f"exactly {expected_span_count} bounded tensor-step values",
        len(span_values),
    )
    per_parameter = {
        name: linear_quantile(values, 0.9)
        for name, values in sorted(relative_by_parameter.items())
    }
    return {
        "observed_relative": max(per_parameter.values()),
        "observed_relative_by_parameter": per_parameter,
        "observed_span": linear_quantile(span_values, 0.9),
    }


def _audit_stage_chain(study_dir: Path) -> tuple[dict[str, Any], str]:
    manifests: dict[str, Any] = {}
    fingerprints: set[str] = set()
    for stage in STAGES:
        manifest_path = study_dir / "stages" / stage / "manifest.json"
        validate_stage_completion(
            study_dir=study_dir,
            manifest_path=manifest_path,
        )
        manifest = read_json(manifest_path)
        _require(
            manifest.get("study_id") == EXPECTED_STUDY_ID,
            f"{stage} manifest study id",
            EXPECTED_STUDY_ID,
            manifest.get("study_id"),
        )
        _require(
            manifest.get("stage_name") == stage,
            f"{stage} manifest stage name",
            repr(stage),
            manifest.get("stage_name"),
        )
        manifests[stage] = manifest
        fingerprints.add(manifest["code_provenance"]["effective_code_fingerprint"])
    _require(
        len(fingerprints) == 1,
        "stage manifests effective code fingerprints",
        "one common fingerprint",
        sorted(fingerprints),
    )
    fingerprint = next(iter(fingerprints))
    _require(
        fingerprint == EXPECTED_CODE_FINGERPRINT,
        "stage manifests effective code fingerprint",
        EXPECTED_CODE_FINGERPRINT,
        fingerprint,
    )
    return manifests, fingerprint


def _audit_assets(
    study_dir: Path,
    study: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    frozen = study["rescue"]["frozen_provenance"]
    indices_path = study_dir / "split" / "indices.json"
    provenance_path = study_dir / "split" / "provenance.json"
    receipt_path = study_dir / "rescue" / "import_receipt.json"
    for path, label in (
        (indices_path, "split indices"),
        (provenance_path, "split provenance"),
        (receipt_path, "parent import receipt"),
    ):
        _require_file(path, label)

    indices = read_json(indices_path)
    provenance = read_json(provenance_path)
    receipt = read_json(receipt_path)
    _require(
        indices.get("official_test_read") is False,
        "split indices official_test_read",
        "false",
        indices.get("official_test_read"),
    )
    _require(
        provenance.get("official_test_read") is False,
        "split provenance official_test_read",
        "false",
        provenance.get("official_test_read"),
    )
    _require(
        receipt.get("official_test_read") is False,
        "import receipt official_test_read",
        "false",
        receipt.get("official_test_read"),
    )
    _require(
        receipt.get("mode") == "reused_parent_assets_recomputed_probe"
        and receipt.get("probe_recomputed_under_relative_rho") is True,
        "import receipt mode",
        "reused parent assets with a freshly recomputed relative-rho probe",
        {
            "mode": receipt.get("mode"),
            "probe_recomputed_under_relative_rho": receipt.get(
                "probe_recomputed_under_relative_rho"
            ),
        },
    )
    _require(
        sha256_file(indices_path) == frozen["split_indices_sha256"],
        "split indices SHA-256",
        frozen["split_indices_sha256"],
        sha256_file(indices_path),
    )
    _require(
        sha256_file(provenance_path) == frozen["split_provenance_sha256"],
        "split provenance SHA-256",
        frozen["split_provenance_sha256"],
        sha256_file(provenance_path),
    )

    train_indices = indices.get("train_indices")
    validation_indices = indices.get("validation_indices")
    _require(
        isinstance(train_indices, list) and len(train_indices) == 55_000,
        "split train_indices",
        "a 55,000-element list",
        None if not isinstance(train_indices, list) else len(train_indices),
    )
    _require(
        isinstance(validation_indices, list) and len(validation_indices) == 5_000,
        "split validation_indices",
        "a 5,000-element list",
        None
        if not isinstance(validation_indices, list)
        else len(validation_indices),
    )
    assert isinstance(train_indices, list) and isinstance(validation_indices, list)
    train_set = set(train_indices)
    validation_set = set(validation_indices)
    _require(
        len(train_set) == 55_000,
        "split train index uniqueness",
        "55,000 unique indices",
        len(train_set),
    )
    _require(
        len(validation_set) == 5_000,
        "split validation index uniqueness",
        "5,000 unique indices",
        len(validation_set),
    )
    _require(
        not train_set & validation_set,
        "train/validation overlap",
        "an empty set",
        sorted(train_set & validation_set)[:20],
    )
    _require(
        train_set | validation_set == set(range(60_000)),
        "train/validation union",
        "exactly MNIST train indices 0..59999",
        {
            "size": len(train_set | validation_set),
            "minimum": min(train_set | validation_set),
            "maximum": max(train_set | validation_set),
        },
    )
    train_hash = stable_index_sequence_hash(train_indices)
    validation_hash = stable_index_sequence_hash(validation_indices)
    for key, observed in (
        ("train_indices_sha256", train_hash),
        ("validation_indices_sha256", validation_hash),
    ):
        _require(
            observed == frozen[key] == provenance.get(key),
            f"split {key}",
            frozen[key],
            {"computed": observed, "provenance": provenance.get(key)},
        )
    _require(
        provenance.get("indices_file_sha256") == frozen["split_indices_sha256"],
        "split provenance indices_file_sha256",
        frozen["split_indices_sha256"],
        provenance.get("indices_file_sha256"),
    )
    epoch_hashes = provenance.get("train_batch_order_sha256")
    _require(
        isinstance(epoch_hashes, list)
        and len(epoch_hashes) == EPOCHS
        and all(SHA256_RE.fullmatch(str(item)) for item in epoch_hashes),
        "split provenance train_batch_order_sha256",
        "five lowercase SHA-256 digests",
        epoch_hashes,
    )

    initialization: dict[str, dict[str, Any]] = {}
    for architecture in ("conv1", "conv2"):
        checkpoint = study_dir / "initialization" / f"{architecture}.pt"
        metadata_path = study_dir / "initialization" / f"{architecture}.json"
        _require_file(checkpoint, f"{architecture} initialization checkpoint")
        _require_file(metadata_path, f"{architecture} initialization metadata")
        metadata = read_json(metadata_path)
        frozen_architecture = frozen["architectures"][architecture]
        observed_checkpoint_sha = sha256_file(checkpoint)
        observed_metadata_sha = sha256_file(metadata_path)
        _require(
            observed_checkpoint_sha
            == metadata.get("checkpoint_sha256")
            == frozen_architecture["checkpoint_sha256"],
            f"{architecture} initialization checkpoint SHA-256",
            frozen_architecture["checkpoint_sha256"],
            {
                "computed": observed_checkpoint_sha,
                "metadata": metadata.get("checkpoint_sha256"),
            },
        )
        _require(
            observed_metadata_sha == frozen_architecture["metadata_sha256"],
            f"{architecture} initialization metadata SHA-256",
            frozen_architecture["metadata_sha256"],
            observed_metadata_sha,
        )
        _require(
            metadata.get("parameter_tensor_sha256")
            == frozen_architecture["parameter_tensor_sha256"],
            f"{architecture} initialization tensor SHA-256",
            frozen_architecture["parameter_tensor_sha256"],
            metadata.get("parameter_tensor_sha256"),
        )
        decoded_tensor_sha = _checkpoint_tensor_digest(
            checkpoint,
            f"{architecture} initialization checkpoint",
        )
        _require(
            decoded_tensor_sha == metadata["parameter_tensor_sha256"],
            f"{architecture} decoded initialization tensor SHA-256",
            metadata["parameter_tensor_sha256"],
            decoded_tensor_sha,
        )
        _require(
            metadata.get("architecture") == architecture
            and metadata.get("model_seed") == 0,
            f"{architecture} initialization identity",
            f"architecture={architecture!r}, model_seed=0",
            {
                "architecture": metadata.get("architecture"),
                "model_seed": metadata.get("model_seed"),
            },
        )
        for field in (
            "natural_scheme_tensor_sha256",
            "loaded_scheme_tensor_sha256",
        ):
            scheme_digests = metadata.get(field)
            _require(
                isinstance(scheme_digests, dict)
                and set(scheme_digests) == {"baseline", "ours", "legacy"}
                and set(scheme_digests.values())
                == {metadata["parameter_tensor_sha256"]},
                f"{architecture} {field}",
                "the shared initialization tensor digest for all three schemes",
                scheme_digests,
            )
        diagnostics = metadata.get("parameter_diagnostics")
        _audit_initialization_parameter_diagnostics(
            checkpoint,
            diagnostics,
            label=f"{architecture} initialization checkpoint",
        )
        assert isinstance(diagnostics, dict)
        initialization[architecture] = metadata

    copied = {
        record.get("destination_path"): record.get("sha256")
        for record in receipt.get("copied_artifacts", [])
        if isinstance(record, dict)
    }
    expected_copies = {
        "split/indices.json": sha256_file(indices_path),
        "split/provenance.json": sha256_file(provenance_path),
        **{
            f"initialization/{architecture}.pt": metadata["checkpoint_sha256"]
            for architecture, metadata in initialization.items()
        },
        **{
            f"initialization/{architecture}.json": sha256_file(
                study_dir / "initialization" / f"{architecture}.json"
            )
            for architecture in initialization
        },
    }
    _require(
        copied == expected_copies,
        "import receipt copied artifact hashes",
        repr(expected_copies),
        copied,
    )

    return indices, provenance, initialization


def _audit_step_log(path: Path, peak_lr: float, entry_id: str) -> list[float]:
    _require_file(path, f"{entry_id} step log")
    learning_rates = [math.nan] * (TOTAL_STEPS + 1)
    observed_rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(
            tuple(reader.fieldnames or ()) == STEP_LOG_FIELDS,
            f"{entry_id} step-log columns",
            repr(STEP_LOG_FIELDS),
            tuple(reader.fieldnames or ()),
        )
        for observed_rows, row in enumerate(reader, start=1):
            path_prefix = f"{entry_id}.step_log[{observed_rows}]"
            step = int(row["step"])
            _require(
                step == observed_rows,
                f"{path_prefix}.step",
                str(observed_rows),
                step,
            )
            expected_epoch = (step - 1) // STEPS_PER_EPOCH + 1
            expected_batch = (step - 1) % STEPS_PER_EPOCH + 1
            _require(
                int(row["epoch"]) == expected_epoch,
                f"{path_prefix}.epoch",
                str(expected_epoch),
                row["epoch"],
            )
            _require(
                int(row["batch_in_epoch"]) == expected_batch,
                f"{path_prefix}.batch_in_epoch",
                str(expected_batch),
                row["batch_in_epoch"],
            )
            expected_count = 8 if expected_batch == STEPS_PER_EPOCH else 16
            _require(
                int(row["sample_count"]) == expected_count,
                f"{path_prefix}.sample_count",
                str(expected_count),
                row["sample_count"],
            )
            _require(
                row["status"] == "complete" and row["failure_reason"] == "",
                f"{path_prefix}.status",
                "complete without a failure reason",
                {"status": row["status"], "failure_reason": row["failure_reason"]},
            )
            for field in ("scheduled_rho", "scheduled_rho_relative", "scheduled_rho_span"):
                _require(
                    row[field] == "",
                    f"{path_prefix}.{field}",
                    "blank for candidate training",
                    row[field],
                )
            learning_rate = _finite(
                row["learning_rate"], f"{path_prefix}.learning_rate", nonnegative=True
            )
            expected_lr = candidate_learning_rate_at_step(peak_lr, step)
            _near(learning_rate, expected_lr, f"{path_prefix}.learning_rate")
            learning_rates[step] = learning_rate
            _finite(row["loss"], f"{path_prefix}.loss", nonnegative=True)
            _finite(row["loss_ema"], f"{path_prefix}.loss_ema", nonnegative=True)
            accuracy = _finite(row["accuracy"], f"{path_prefix}.accuracy")
            _require(
                0.0 <= accuracy <= 1.0,
                f"{path_prefix}.accuracy",
                "a fraction in [0, 1]",
                accuracy,
            )
    _require(
        observed_rows == TOTAL_STEPS,
        f"{entry_id} step-log row count",
        f"exactly {TOTAL_STEPS}",
        observed_rows,
    )
    _near(learning_rates[1], peak_lr / WARMUP_STEPS, f"{entry_id}.lr.step1")
    _near(learning_rates[WARMUP_STEPS], peak_lr, f"{entry_id}.lr.step860")
    _near(learning_rates[TOTAL_STEPS], 0.0, f"{entry_id}.lr.final")
    return learning_rates


def _audit_minibatches(
    path: Path,
    *,
    entry_id: str,
    train_indices: Sequence[int],
    provenance: Mapping[str, Any],
) -> str:
    _require_file(path, f"{entry_id} minibatches")
    payload = read_json(path)
    _require(
        payload.get("schema_version") == "mnist-conv-lr-minibatches/v1"
        and payload.get("row_id") == entry_id.rsplit("--", 1)[0]
        and payload.get("candidate_role") == entry_id.rsplit("--", 1)[1]
        and payload.get("stage") == "candidates",
        f"{entry_id} minibatch identity",
        "v1 candidates artifact with matching row id and role",
        {
            key: payload.get(key)
            for key in (
                "schema_version",
                "row_id",
                "candidate_role",
                "stage",
            )
        },
    )
    batches = payload.get("batches")
    _require(
        isinstance(batches, list) and len(batches) == TOTAL_STEPS,
        f"{entry_id} minibatches",
        f"a {TOTAL_STEPS}-batch list",
        None if not isinstance(batches, list) else len(batches),
    )
    assert isinstance(batches, list)
    expected_epoch_hashes = provenance["train_batch_order_sha256"]
    expected_sorted_train = sorted(int(value) for value in train_indices)
    for epoch in range(EPOCHS):
        start = epoch * STEPS_PER_EPOCH
        stop = start + STEPS_PER_EPOCH
        epoch_batches = batches[start:stop]
        sizes = [len(batch) for batch in epoch_batches]
        _require(
            sizes == [16] * (STEPS_PER_EPOCH - 1) + [8],
            f"{entry_id} epoch {epoch + 1} batch sizes",
            "3,437 batches of 16 followed by one batch of 8",
            Counter(sizes),
        )
        epoch_hash = stable_batch_order_hash(epoch_batches)
        _require(
            epoch_hash == expected_epoch_hashes[epoch],
            f"{entry_id} epoch {epoch + 1} batch-order SHA-256",
            expected_epoch_hashes[epoch],
            epoch_hash,
        )
        flattened = [int(value) for batch in epoch_batches for value in batch]
        _require(
            sorted(flattened) == expected_sorted_train,
            f"{entry_id} epoch {epoch + 1} train-index coverage",
            "every frozen train index exactly once",
            {
                "count": len(flattened),
                "unique": len(set(flattened)),
                "sorted_sequence_sha256": stable_index_sequence_hash(
                    sorted(flattened)
                ),
            },
        )
    digest = stable_batch_order_hash(batches)
    _require(
        payload.get("batch_order_sha256") == digest,
        f"{entry_id} minibatch payload SHA-256",
        digest,
        payload.get("batch_order_sha256"),
    )
    return digest


def _audit_parameter_diagnostics(
    path: Path,
    *,
    entry_id: str,
    learning_rates: Sequence[float],
    initialization_diagnostics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _require_file(path, f"{entry_id} parameter diagnostics")
    parameter_names = set(initialization_diagnostics)
    bounded_names = {
        name
        for name, record in initialization_diagnostics.items()
        if record["bounded_gate"] is True
    }
    _require(
        bool(bounded_names),
        f"{entry_id} bounded parameter set",
        "non-empty",
        sorted(bounded_names),
    )
    counts: Counter[str] = Counter()
    warmup_relative: dict[str, list[float]] = defaultdict(list)
    warmup_span: list[float] = []
    eligible_projection_values: list[float] = []
    early_streak = {name: 0 for name in bounded_names}
    current_step = 0
    current_parameters: set[str] = set()
    row_count = 0

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(
            tuple(reader.fieldnames or ()) == PARAMETER_DIAGNOSTIC_FIELDS,
            f"{entry_id} parameter-diagnostic columns",
            repr(PARAMETER_DIAGNOSTIC_FIELDS),
            tuple(reader.fieldnames or ()),
        )
        for row_count, row in enumerate(reader, start=1):
            prefix = f"{entry_id}.parameter_diagnostics[{row_count}]"
            step = int(row["step"])
            if step != current_step:
                if current_step:
                    _require(
                        current_parameters == parameter_names,
                        f"{entry_id} diagnostic parameters at step {current_step}",
                        f"exactly {sorted(parameter_names)!r}",
                        sorted(current_parameters),
                    )
                _require(
                    step == current_step + 1,
                    f"{prefix}.step",
                    str(current_step + 1),
                    step,
                )
                current_step = step
                current_parameters = set()
            name = row["parameter"]
            _require(
                name in parameter_names and name not in current_parameters,
                f"{prefix}.parameter",
                "one unique frozen parameter name at this step",
                name,
            )
            current_parameters.add(name)
            counts[name] += 1
            metadata = initialization_diagnostics[name]
            bounded = _csv_bool(row["bounded_gate"], f"{prefix}.bounded_gate")
            report_only = _csv_bool(row["report_only"], f"{prefix}.report_only")
            proposal_zero = _csv_bool(
                row["proposal_is_numerically_zero"],
                f"{prefix}.proposal_is_numerically_zero",
            )
            gate_eligible = _csv_bool(
                row["projection_gate_eligible"],
                f"{prefix}.projection_gate_eligible",
            )
            _require(
                bounded is bool(metadata["bounded_gate"]),
                f"{prefix}.bounded_gate",
                str(bool(metadata["bounded_gate"])),
                bounded,
            )
            _require(
                report_only is (not bounded),
                f"{prefix}.report_only",
                str(not bounded),
                report_only,
            )
            _require(
                gate_eligible is (bounded and not proposal_zero),
                f"{prefix}.projection_gate_eligible",
                str(bounded and not proposal_zero),
                gate_eligible,
            )
            _near(row["learning_rate"], learning_rates[step], f"{prefix}.learning_rate")
            for field in ("scheduled_rho", "scheduled_rho_relative", "scheduled_rho_span"):
                _require(
                    row[field] == "",
                    f"{prefix}.{field}",
                    "blank for candidate training",
                    row[field],
                )
            gradient_rms = _finite(
                row["gradient_rms"], f"{prefix}.gradient_rms", nonnegative=True
            )
            proposal_rms = _finite(
                row["proposed_update_rms"],
                f"{prefix}.proposed_update_rms",
                nonnegative=True,
            )
            normalized = _finite(
                row["normalized_update"],
                f"{prefix}.normalized_update",
                nonnegative=True,
            )
            efficiency = _finite(
                row["projection_efficiency"],
                f"{prefix}.projection_efficiency",
                nonnegative=True,
            )
            del gradient_rms
            if bounded:
                span = _finite(
                    row["span_normalized_update"],
                    f"{prefix}.span_normalized_update",
                    nonnegative=True,
                )
                initial_rms = _finite(
                    row["initial_parameter_rms"],
                    f"{prefix}.initial_parameter_rms",
                    nonnegative=True,
                )
                _require(
                    initial_rms > 0.0,
                    f"{prefix}.initial_parameter_rms",
                    "positive",
                    initial_rms,
                )
                relative = _finite(
                    row["parameter_relative_update"],
                    f"{prefix}.parameter_relative_update",
                    nonnegative=True,
                )
                _near(initial_rms, metadata["rms"], f"{prefix}.initial_parameter_rms")
                _near(span, proposal_rms / 100.0, f"{prefix}.span_normalized_update")
                _near(normalized, span, f"{prefix}.normalized_update")
                _near(
                    relative,
                    proposal_rms / initial_rms,
                    f"{prefix}.parameter_relative_update",
                )
                fractions = {
                    field: _finite(row[field], f"{prefix}.{field}")
                    for field in (
                        "proposed_bound_crossing_fraction",
                        "lower_bound_occupancy",
                        "upper_bound_occupancy",
                        "combined_bound_occupancy",
                    )
                }
                for field, value in fractions.items():
                    _require(
                        0.0 <= value <= 1.0,
                        f"{prefix}.{field}",
                        "a fraction in [0, 1]",
                        value,
                    )
                _near(
                    fractions["combined_bound_occupancy"],
                    fractions["lower_bound_occupancy"]
                    + fractions["upper_bound_occupancy"],
                    f"{prefix}.combined_bound_occupancy",
                )
                if gate_eligible:
                    eligible_projection_values.append(efficiency)
                    if step <= int(0.20 * TOTAL_STEPS):
                        if efficiency < 0.50:
                            early_streak[name] += 1
                            _require(
                                early_streak[name] < 16,
                                f"{entry_id} early projection streak for {name}",
                                "fewer than 16 consecutive eligible steps below 0.50",
                                {"step": step, "streak": early_streak[name]},
                            )
                        else:
                            early_streak[name] = 0
                if WARMUP_OBSERVATION_START <= step <= WARMUP_STEPS:
                    warmup_relative[name].append(relative)
                    warmup_span.append(span)
            else:
                for field in (
                    "span_normalized_update",
                    "initial_parameter_rms",
                    "parameter_relative_update",
                    "proposed_bound_crossing_fraction",
                    "lower_bound_occupancy",
                    "upper_bound_occupancy",
                    "combined_bound_occupancy",
                ):
                    _require(
                        row[field] == "",
                        f"{prefix}.{field}",
                        "blank for report-only parameters",
                        row[field],
                    )

    if current_step:
        _require(
            current_parameters == parameter_names,
            f"{entry_id} diagnostic parameters at step {current_step}",
            f"exactly {sorted(parameter_names)!r}",
            sorted(current_parameters),
        )
    _require(
        current_step == TOTAL_STEPS,
        f"{entry_id} final diagnostic step",
        str(TOTAL_STEPS),
        current_step,
    )
    expected_rows = TOTAL_STEPS * len(parameter_names)
    _require(
        row_count == expected_rows,
        f"{entry_id} parameter diagnostic row count",
        str(expected_rows),
        row_count,
    )
    _require(
        counts == Counter({name: TOTAL_STEPS for name in parameter_names}),
        f"{entry_id} parameter diagnostic counts",
        f"{TOTAL_STEPS} rows per parameter",
        dict(counts),
    )
    _require(
        set(warmup_relative) == bounded_names,
        f"{entry_id} relative-rho bounded parameter set",
        repr(sorted(bounded_names)),
        sorted(warmup_relative),
    )
    observed = _aggregate_observed_rho(
        warmup_relative,
        warmup_span,
        path=entry_id,
    )
    return {
        **observed,
        "median_projection_efficiency": linear_quantile(
            eligible_projection_values, 0.5
        ),
        "row_count": row_count,
    }


def _audit_validation_metrics(
    validation: Any,
    *,
    entry_id: str,
) -> None:
    _require(
        isinstance(validation, list) and len(validation) == EPOCHS,
        f"{entry_id} validation metrics",
        "five epoch records",
        validation,
    )
    assert isinstance(validation, list)
    for epoch, metric in enumerate(validation, start=1):
        prefix = f"{entry_id}.validation[{epoch}]"
        _require(metric.get("epoch") == epoch, f"{prefix}.epoch", str(epoch), metric.get("epoch"))
        _require(
            metric.get("step") == epoch * STEPS_PER_EPOCH,
            f"{prefix}.step",
            str(epoch * STEPS_PER_EPOCH),
            metric.get("step"),
        )
        _require(
            metric.get("sample_count") == 5_000,
            f"{prefix}.sample_count",
            "5000",
            metric.get("sample_count"),
        )
        for field in ("loss", "train_loss"):
            _finite(metric.get(field), f"{prefix}.{field}", nonnegative=True)
        for field in ("accuracy", "train_accuracy"):
            value = _finite(metric.get(field), f"{prefix}.{field}")
            _require(
                0.0 <= value <= 1.0,
                f"{prefix}.{field}",
                "a fraction in [0, 1]",
                value,
            )


def _audit_candidate(
    study_dir: Path,
    study: Mapping[str, Any],
    manifest_entry: Mapping[str, Any],
    row: Mapping[str, Any],
    role: str,
    *,
    train_indices: Sequence[int],
    split_provenance: Mapping[str, Any],
    initialization: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    entry_id = f"{row['row_id']}--{role}"
    directory = _candidate_dir(study_dir, entry_id)
    summary = read_json(directory / "summary.json")
    validation = read_json(directory / "validation.json")
    run_spec_path = directory / "run_spec.v2.json"
    run_spec = RunSpec.from_path(run_spec_path).data
    payload = manifest_entry["payload"]

    _require(summary.get("schema_version") == "mnist-conv-lr-candidate-result/v2", f"{entry_id} schema", "'mnist-conv-lr-candidate-result/v2'", summary.get("schema_version"))
    _require(
        summary.get("status") == "complete"
        and summary.get("admissible") is True,
        f"{entry_id} candidate status",
        "complete and admissible",
        {"status": summary.get("status"), "admissible": summary.get("admissible")},
    )
    _require(
        summary.get("inadmissible_reason") is None
        and summary.get("numerical_failure_detail") is None,
        f"{entry_id} failure fields",
        "null",
        {
            "inadmissible_reason": summary.get("inadmissible_reason"),
            "numerical_failure_detail": summary.get("numerical_failure_detail"),
        },
    )
    _require(
        summary.get("attempted_steps") == TOTAL_STEPS
        and summary.get("completed_steps") == TOTAL_STEPS,
        f"{entry_id} step counts",
        f"attempted=completed={TOTAL_STEPS}",
        {
            "attempted": summary.get("attempted_steps"),
            "completed": summary.get("completed_steps"),
        },
    )
    _require(
        summary.get("rho_definition") == "initial_parameter_rms",
        f"{entry_id} rho definition",
        "'initial_parameter_rms'",
        summary.get("rho_definition"),
    )
    peak_lr = _finite(summary.get("peak_learning_rate"), f"{entry_id}.peak_learning_rate")
    _require(peak_lr > 0.0, f"{entry_id}.peak_learning_rate", "positive", peak_lr)
    _near(payload["peak_learning_rate"], peak_lr, f"{entry_id}.manifest peak LR")
    _near(payload["rho_target"], summary["rho_target"], f"{entry_id}.manifest rho target")
    _near(summary["rho_target_relative"], summary["rho_target"], f"{entry_id}.rho_target_relative")
    _require(summary.get("candidate_role") == role, f"{entry_id}.candidate_role", role, summary.get("candidate_role"))
    _require(summary.get("row") == row, f"{entry_id}.row", repr(dict(row)), summary.get("row"))
    _require(
        payload.get("row_id") == row["row_id"]
        and payload.get("candidate_role") == role,
        f"{entry_id} manifest row/role",
        f"{row['row_id']}/{role}",
        {
            "row_id": payload.get("row_id"),
            "candidate_role": payload.get("candidate_role"),
        },
    )

    probe_summary_path = (
        _row_stage_dir(study_dir, "probe", row["row_id"]) / "summary.json"
    )
    range_summary_path = (
        _row_stage_dir(study_dir, "range", row["row_id"]) / "summary.json"
    )
    probe_summary_value = read_json(probe_summary_path)
    range_summary_value = read_json(range_summary_path)
    range_candidate = range_summary_value.get("candidates", {}).get(role)
    _require(
        range_summary_value.get("candidates", {}).get("status") == "resolved"
        and isinstance(range_candidate, dict),
        f"{entry_id} range candidate",
        "a resolved candidate object",
        range_summary_value.get("candidates"),
    )
    assert isinstance(range_candidate, dict)
    _near(range_candidate["rho_target"], summary["rho_target"], f"{entry_id} range rho target")
    _near(range_candidate["learning_rate"], peak_lr, f"{entry_id} range peak LR")
    _near(range_summary_value["rho_unit"], probe_summary_value["rho_unit"], f"{entry_id} probe/range rho unit")
    _near(
        peak_lr,
        float(summary["rho_target"]) / float(range_summary_value["rho_unit"]),
        f"{entry_id} rho-to-LR conversion",
    )

    learning_rates = _audit_step_log(directory / "step_log.csv", peak_lr, entry_id)
    batch_hash = _audit_minibatches(
        directory / "minibatches.json",
        entry_id=entry_id,
        train_indices=train_indices,
        provenance=split_provenance,
    )
    diagnostics = _audit_parameter_diagnostics(
        directory / "parameter_diagnostics.csv",
        entry_id=entry_id,
        learning_rates=learning_rates,
        initialization_diagnostics=initialization["parameter_diagnostics"],
    )

    _require(validation == summary.get("validation_metrics"), f"{entry_id} validation JSON", "identical to summary.validation_metrics", validation)
    _audit_validation_metrics(validation, entry_id=entry_id)
    best_epoch = summary.get("best_validation_epoch")
    _require(type(best_epoch) is int and 1 <= best_epoch <= EPOCHS, f"{entry_id}.best_validation_epoch", "an integer in [1, 5]", best_epoch)
    expected_best_epoch = min(
        validation,
        key=lambda metric: (float(metric["loss"]), int(metric["epoch"])),
    )["epoch"]
    _require(
        best_epoch == expected_best_epoch,
        f"{entry_id}.best_validation_epoch",
        str(expected_best_epoch),
        best_epoch,
    )
    final_metric = validation[-1]
    _near(summary["final_validation_loss"], final_metric["loss"], f"{entry_id}.final_validation_loss")
    _near(summary["final_validation_accuracy"], final_metric["accuracy"], f"{entry_id}.final_validation_accuracy")

    for field, expected in diagnostics["observed_relative_by_parameter"].items():
        provided = summary.get("observed_peak_rho_relative_by_parameter", {}).get(field)
        _near(provided, expected, f"{entry_id}.observed_peak_rho_relative_by_parameter[{field}]")
    _require(
        set(summary.get("observed_peak_rho_relative_by_parameter", {}))
        == set(diagnostics["observed_relative_by_parameter"]),
        f"{entry_id} observed relative-rho parameter set",
        repr(sorted(diagnostics["observed_relative_by_parameter"])),
        sorted(summary.get("observed_peak_rho_relative_by_parameter", {})),
    )
    for key in ("observed_peak_rho", "observed_peak_rho_relative"):
        _near(summary.get(key), diagnostics["observed_relative"], f"{entry_id}.{key}")
    _near(summary.get("observed_peak_rho_span"), diagnostics["observed_span"], f"{entry_id}.observed_peak_rho_span")
    _near(summary.get("median_projection_efficiency"), diagnostics["median_projection_efficiency"], f"{entry_id}.median_projection_efficiency")
    bounded_scales = {
        name: record["rms"]
        for name, record in initialization["parameter_diagnostics"].items()
        if record["bounded_gate"] is True
    }
    _require(summary.get("initial_bounded_parameter_rms") == bounded_scales, f"{entry_id}.initial_bounded_parameter_rms", repr(bounded_scales), summary.get("initial_bounded_parameter_rms"))

    architecture = row["architecture"]
    checkpoint_path = study_dir / "initialization" / f"{architecture}.pt"
    _require(summary.get("checkpoint_sha256") == sha256_file(checkpoint_path) == initialization["checkpoint_sha256"], f"{entry_id} initialization checkpoint SHA-256", initialization["checkpoint_sha256"], summary.get("checkpoint_sha256"))
    _require(summary.get("initial_parameter_tensor_sha256") == initialization["parameter_tensor_sha256"], f"{entry_id} initial tensor SHA-256", initialization["parameter_tensor_sha256"], summary.get("initial_parameter_tensor_sha256"))
    _require(summary.get("train_indices_sha256") == split_provenance["train_indices_sha256"], f"{entry_id} train split SHA-256", split_provenance["train_indices_sha256"], summary.get("train_indices_sha256"))
    _require(summary.get("validation_indices_sha256") == split_provenance["validation_indices_sha256"], f"{entry_id} validation split SHA-256", split_provenance["validation_indices_sha256"], summary.get("validation_indices_sha256"))
    _require(summary.get("minibatch_order_sha256") == batch_hash, f"{entry_id} summary minibatch SHA-256", batch_hash, summary.get("minibatch_order_sha256"))
    for filename, field in (("best_validation.pt", "best_checkpoint_sha256"), ("final.pt", "final_checkpoint_sha256"), ("run_spec.v2.json", "run_spec_v2_sha256")):
        file_path = directory / filename
        _require_file(file_path, f"{entry_id} {filename}")
        observed = sha256_file(file_path)
        _require(summary.get(field) == observed, f"{entry_id}.{field}", observed, summary.get(field))
    _checkpoint_tensor_digest(
        directory / "best_validation.pt",
        f"{entry_id} best-validation checkpoint",
    )
    decoded_final_tensor_sha = _checkpoint_tensor_digest(
        directory / "final.pt",
        f"{entry_id} final checkpoint",
    )
    _require(
        summary.get("final_parameter_tensor_sha256") == decoded_final_tensor_sha,
        f"{entry_id}.final_parameter_tensor_sha256",
        decoded_final_tensor_sha,
        summary.get("final_parameter_tensor_sha256"),
    )

    expected_label = f"{row['row_id']}-{role}-seed0-lr-screen"
    _require(
        {
            key: run_spec.get(key)
            for key in (
                "schema_version",
                "label",
                "seed",
                "replicate_id",
                "protocol_id",
                "category",
            )
        }
        == {
            "schema_version": "mnist-conv-run/v2",
            "label": expected_label,
            "seed": 0,
            "replicate_id": None,
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "category": "diagnostic",
        },
        f"{entry_id} RunSpec identity",
        "the exact seed-0 v3 diagnostic identity",
        {
            key: run_spec.get(key)
            for key in (
                "schema_version",
                "label",
                "seed",
                "replicate_id",
                "protocol_id",
                "category",
            )
        },
    )
    run = run_spec["run"]
    expected_checkpoint_path = checkpoint_path.relative_to(
        study_dir.parent.parent
    ).as_posix()
    affine = dict(study["dataset"]["affine"])
    affine.pop("deterministic_by_original_index")
    architecture_contract = study["model"]["architectures"][architecture]
    layer_saturations = CALIBRATION_LAYER_SATURATIONS[row["row_id"]]
    expected_sections = {
        "dataset": {
            "name": "mnist",
            "input_shape": [2, 28, 28],
            "batch_size": 16,
            "normalization": dict(study["dataset"]["normalization"]),
            "affine": affine,
            "max_batches": None,
            "train_shuffle_seed": 0,
            "validation": {
                "source": "mnist_train",
                "size": 5_000,
                "samples_per_class": 500,
                "split_seed": 0,
                "batch_size": 128,
                "stratified": True,
                "official_test_enabled": False,
                "indices_sha256": split_provenance[
                    "validation_indices_sha256"
                ],
            },
        },
        "architecture": {
            "profile": architecture,
            "channels": list(architecture_contract["channels"]),
            "kernel_sizes": list(architecture_contract["kernel_sizes"]),
            "strides": list(architecture_contract["strides"]),
            "paddings": list(architecture_contract["paddings"]),
            "output_dim": 20,
            "pooling": {"mode": "none"},
        },
        "model": {
            "type": "resistive_conv",
            "non_linearity": "hard_sigmoid",
            "voltage_amp": row["voltage_amp"],
            "current_amp": row["current_amp"],
            "input_gain": row["input_gain"],
            "weight_gains": [1.0]
            * (len(architecture_contract["channels"]) + 1),
            "weight_min": 0.0,
            "weight_max": 100.0,
            "weight_init_mode": "kaiming_uniform",
            "quadratic_diode_param": {},
            "exponential_diode_param": {},
            "hard_sigmoid_param": dict(study["model"]["hard_sigmoid"]),
            "trainable_parameters": {
                "weights": True,
                "biases": True,
                "amplification": False,
                "hard_sigmoid_v_off": False,
            },
            "amplification_min": 1e-6,
            "amplification_max": None,
        },
        "solver": {
            "inference_iterations": row["inference_iterations"],
            "training_iterations": row["training_iterations"],
            "energy_mode": study["solver"]["energy_mode"],
            "minimizer": dict(study["solver"]["minimizer"]),
        },
        "training": {
            "algorithm": "BP",
            "epochs": EPOCHS,
            "optimizer": {
                "name": "SGD",
                "momentum": 0.0,
                "weight_decay": 0.0,
            },
            "peak_learning_rate": peak_lr,
            "schedule": {
                "name": "linear_warmup_cosine",
                "interval": "optimizer_step",
                "warmup_fraction": 0.05,
                "warmup_steps": WARMUP_STEPS,
                "total_steps": TOTAL_STEPS,
                "min_lr_factor": 0.0,
            },
            "beta": 0.1,
            "checkpoint_rule": "min_validation_loss",
            "pruning": {
                "enabled": False,
                "after_epoch": None,
                "min_best_validation_accuracy": None,
            },
            "batch_state_policy": "reset_each_batch",
            "diagnostics": {
                "enabled": True,
                "bounded_parameter_classes": ["ConvWeight", "DenseWeight"],
                "bias_global_gate": False,
                "range_epsilon": 1e-8,
                "projection_epsilon": 1e-12,
                "early_fraction": 0.2,
                "projection_efficiency_threshold": 0.5,
                "projection_persistence": 16,
                "occupancy_delta_threshold": 0.2,
                "occupancy_persistence": 16,
            },
        },
        "initialization": {
            "checkpoint": {
                "path": expected_checkpoint_path,
                "sha256": initialization["checkpoint_sha256"],
                "format": "drn.function.parameters/v1",
                "source_run_id": None,
                "role": "initialization",
            }
        },
        "calibration": {
            "kind": "hard_sigmoid_saturation",
            "calibration_id": (
                "conv-hardsigmoid-sat30-20260718-" + row["row_id"]
            ),
            "scope": "first_hidden_layer",
            "sample_count": 256,
            "batch_size": 64,
            "model_seed": 0,
            "affine_seed": 1729,
            "settling_iterations": 64,
            "adaptive_equilibrium": False,
            "v_off": 4.0,
            "g_on": 100.0,
            "g_off": 0.0,
            "target_initial_saturation": 0.3,
            "measured_initial_saturation": layer_saturations[0],
            "layer_measurements": [
                {"layer_index": index, "measured_saturation": value}
                for index, value in enumerate(layer_saturations, start=1)
            ],
        },
    }
    for section, expected_section in expected_sections.items():
        _require(
            run[section] == expected_section,
            f"{entry_id} RunSpec {section}",
            repr(expected_section),
            run[section],
        )
    provenance = run["lr_provenance"]
    _require(
        provenance["study_id"] == EXPECTED_STUDY_ID,
        f"{entry_id} RunSpec LR study id",
        EXPECTED_STUDY_ID,
        provenance["study_id"],
    )
    _require(provenance["row_id"] == row["row_id"] and provenance["candidate_role"] == role, f"{entry_id} RunSpec LR row/role", f"{row['row_id']}/{role}", {"row_id": provenance["row_id"], "role": provenance["candidate_role"]})
    _near(provenance["rho_target"], summary["rho_target"], f"{entry_id} RunSpec rho target")
    _near(provenance["rho_unit"], range_summary_value["rho_unit"], f"{entry_id} RunSpec rho unit")
    _require(provenance["probe_sha256"] == sha256_file(probe_summary_path), f"{entry_id} RunSpec probe SHA-256", sha256_file(probe_summary_path), provenance["probe_sha256"])
    _require(provenance["range_sha256"] == sha256_file(range_summary_path), f"{entry_id} RunSpec range SHA-256", sha256_file(range_summary_path), provenance["range_sha256"])
    _require(provenance["split_sha256"] == split_provenance["validation_indices_sha256"], f"{entry_id} RunSpec split SHA-256", split_provenance["validation_indices_sha256"], provenance["split_sha256"])
    _require(provenance["batch_order_sha256"] == batch_hash, f"{entry_id} RunSpec batch-order SHA-256", batch_hash, provenance["batch_order_sha256"])
    _require(provenance["initialization_tensor_sha256"] == initialization["parameter_tensor_sha256"], f"{entry_id} RunSpec initialization tensor SHA-256", initialization["parameter_tensor_sha256"], provenance["initialization_tensor_sha256"])
    _require_no_test_metrics(summary, f"{entry_id} summary")
    _require_no_test_metrics(validation, f"{entry_id} validation")
    return summary, batch_hash


def _candidate_result(role: str, summary: Mapping[str, Any]) -> CandidateRunResult:
    return CandidateRunResult(
        candidate_id=role,
        peak_learning_rate=float(summary["peak_learning_rate"]),
        admissible=bool(summary["admissible"]),
        final_validation_loss=summary["final_validation_loss"],
        final_validation_accuracy=summary["final_validation_accuracy"],
        median_projection_efficiency=summary["median_projection_efficiency"],
        inadmissible_reason=summary["inadmissible_reason"],
    )


def _audit_selection(
    study_dir: Path,
    study: Mapping[str, Any],
    candidate_summaries: Mapping[str, Mapping[str, Any]],
) -> str:
    selection_path = study_dir / "stages" / "select" / "entries" / "selection" / "selection.json"
    selection = read_json(selection_path)
    _require(selection.get("schema_version") == "mnist-conv-lr-selection/v2", "selection schema", "'mnist-conv-lr-selection/v2'", selection.get("schema_version"))
    _require(selection.get("status") == "frozen_seed0_screen", "selection status", "'frozen_seed0_screen'", selection.get("status"))
    _require(selection.get("rho_definition") == "initial_parameter_rms", "selection rho definition", "'initial_parameter_rms'", selection.get("rho_definition"))
    _require(selection.get("final_paper_training_authorized") is False, "selection final_paper_training_authorized", "false", selection.get("final_paper_training_authorized"))
    rows = selection.get("rows")
    _require(isinstance(rows, list) and len(rows) == len(study["rows"]), "selection rows", f"exactly {len(study['rows'])} rows", rows)
    assert isinstance(rows, list)
    by_row = {row["row_id"]: row for row in rows}
    _require(len(by_row) == len(rows), "selection row ids", "unique", [row["row_id"] for row in rows])
    for frozen_row in study["rows"]:
        row_id = frozen_row["row_id"]
        observed = by_row.get(row_id)
        _require(observed is not None, f"selection row {row_id}", "present", observed)
        summaries = {
            role: candidate_summaries[f"{row_id}--{role}"] for role in ROLES
        }
        expected = select_final_candidate(
            tuple(_candidate_result(role, summaries[role]) for role in ROLES),
            plateau_relative_tolerance=float(
                study["selection"]["plateau_relative_to_minimum"]
            ),
        )
        _require(expected.selected is not None and expected.status == "frozen_seed0_screen", f"recomputed selection for {row_id}", "frozen_seed0_screen with a selected candidate", expected)
        chosen = expected.selected
        chosen_summary = summaries[chosen.candidate_id]
        _require(observed["status"] == expected.status and observed["reason"] == expected.reason, f"selection status for {row_id}", repr((expected.status, expected.reason)), (observed["status"], observed["reason"]))
        _require(observed["selected_candidate_role"] == chosen.candidate_id, f"selection role for {row_id}", chosen.candidate_id, observed["selected_candidate_role"])
        _require(observed["plateau_roles"] == [item.candidate_id for item in expected.plateau], f"selection plateau for {row_id}", repr([item.candidate_id for item in expected.plateau]), observed["plateau_roles"])
        for field, value in (
            ("selected_peak_learning_rate", chosen.peak_learning_rate),
            ("selected_rho_target", chosen_summary["rho_target"]),
            ("selected_rho_target_relative", chosen_summary["rho_target"]),
            ("observed_peak_rho", chosen_summary["observed_peak_rho"]),
            ("observed_peak_rho_relative", chosen_summary["observed_peak_rho_relative"]),
            ("observed_peak_rho_span", chosen_summary["observed_peak_rho_span"]),
            ("final_validation_loss", chosen.final_validation_loss),
            ("final_validation_accuracy", chosen.final_validation_accuracy),
            ("median_projection_efficiency", chosen.median_projection_efficiency),
            ("minimum_final_validation_loss", expected.minimum_final_validation_loss),
        ):
            _near(observed[field], value, f"selection {row_id}.{field}")
        for field in ("architecture", "scheme", "input_gain", "inference_iterations", "training_iterations"):
            _require(observed[field] == frozen_row[field], f"selection {row_id}.{field}", repr(frozen_row[field]), observed[field])

    csv_path = selection_path.with_suffix(".csv")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(
            tuple(reader.fieldnames or ()) == SELECTION_CSV_FIELDS,
            "selection CSV columns",
            repr(SELECTION_CSV_FIELDS),
            tuple(reader.fieldnames or ()),
        )
        csv_rows = list(reader)
    _require(len(csv_rows) == len(study["rows"]), "selection CSV row count", str(len(study["rows"])), len(csv_rows))
    expected_row_order = [row["row_id"] for row in study["rows"]]
    _require(
        [row["row_id"] for row in rows] == expected_row_order,
        "selection JSON row order",
        repr(expected_row_order),
        [row["row_id"] for row in rows],
    )
    _require(
        [row["row_id"] for row in csv_rows] == expected_row_order,
        "selection CSV row order",
        repr(expected_row_order),
        [row["row_id"] for row in csv_rows],
    )
    for index, (json_row, csv_row) in enumerate(zip(rows, csv_rows)):
        expected_csv_row = {}
        for field in SELECTION_CSV_FIELDS:
            value = json_row[field]
            if value is None:
                expected_csv_row[field] = ""
            elif field == "plateau_roles":
                expected_csv_row[field] = ",".join(value)
            else:
                expected_csv_row[field] = str(value)
        _require(
            csv_row == expected_csv_row,
            f"selection CSV row {index}",
            "an exact serialization of selection.json",
            {"expected": expected_csv_row, "provided": csv_row},
        )
    _require_no_test_metrics(selection, "selection artifact")

    selection_sha = sha256_file(selection_path)
    marker = read_json(selection_path.parent / "complete.json")
    output_record = next(
        (
            record
            for record in marker["outputs"]
            if record["path"].endswith("/selection.json")
        ),
        None,
    )
    _require(output_record is not None, "selection entry completion record", "a selection.json record", output_record)
    _require(output_record["sha256"] == selection_sha, "selection JSON completion SHA-256", selection_sha, output_record["sha256"])
    return selection_sha


def audit_study(study_dir: str | Path) -> dict[str, Any]:
    root = Path(study_dir).expanduser().resolve()
    _require(root.is_dir() and not root.is_symlink(), "--study", "an existing regular directory", str(root))
    resolved_path = root / "study.resolved.json"
    _require_file(resolved_path, "resolved study contract")
    spec = LRStudySpec.from_path(resolved_path)
    study = spec.data
    _require(spec.study_id == EXPECTED_STUDY_ID, "study id", EXPECTED_STUDY_ID, spec.study_id)
    _require(root.name.endswith(f"--{spec.study_id}"), "study directory name", f"a name ending in --{spec.study_id}", root.name)
    _require(study["schema_version"] == LR_RELATIVE_RHO_STUDY_SCHEMA_VERSION, "study schema", LR_RELATIVE_RHO_STUDY_SCHEMA_VERSION, study["schema_version"])
    _require(study["protocol_id"] == EXPECTED_PROTOCOL_ID, "study protocol id", EXPECTED_PROTOCOL_ID, study["protocol_id"])
    _require(study["dataset"]["official_test"] == {"enabled": False, "read_allowed": False}, "study official-test contract", "disabled and unreadable", study["dataset"]["official_test"])
    _require(study["candidate_training"]["official_test_evaluation"] is False, "candidate official_test_evaluation", "false", study["candidate_training"]["official_test_evaluation"])

    manifests, code_fingerprint = _audit_stage_chain(root)
    indices, split_provenance, initialization = _audit_assets(root, study)
    expected_entry_ids = [
        f"{row['row_id']}--{role}" for row in study["rows"] for role in ROLES
    ]
    row_ids = [row["row_id"] for row in study["rows"]]
    for stage in ("probe", "range"):
        observed_ids = [
            entry["entry_id"] for entry in manifests[stage]["entries"]
        ]
        _require(
            observed_ids == row_ids,
            f"{stage} manifest entries",
            repr(row_ids),
            observed_ids,
        )
    select_ids = [
        entry["entry_id"] for entry in manifests["select"]["entries"]
    ]
    _require(
        select_ids == ["selection"],
        "select manifest entries",
        "['selection']",
        select_ids,
    )
    candidate_entries = manifests["candidates"]["entries"]
    observed_entry_ids = [entry["entry_id"] for entry in candidate_entries]
    _require(observed_entry_ids == expected_entry_ids, "candidate manifest entries", repr(expected_entry_ids), observed_entry_ids)
    _require(len(candidate_entries) == 12, "candidate manifest entry count", "12", len(candidate_entries))
    entry_by_id = {entry["entry_id"]: entry for entry in candidate_entries}

    candidate_summaries: dict[str, dict[str, Any]] = {}
    batch_hashes: set[str] = set()
    for row in study["rows"]:
        metadata = initialization[row["architecture"]]
        for role in ROLES:
            entry_id = f"{row['row_id']}--{role}"
            summary, batch_hash = _audit_candidate(
                root,
                study,
                entry_by_id[entry_id],
                row,
                role,
                train_indices=indices["train_indices"],
                split_provenance=split_provenance,
                initialization=metadata,
            )
            candidate_summaries[entry_id] = summary
            batch_hashes.add(batch_hash)
    _require(len(batch_hashes) == 1, "candidate minibatch-order hashes", "one identical hash across all 12 candidates", sorted(batch_hashes))
    selection_sha = _audit_selection(root, study, candidate_summaries)
    return {
        "status": "validated",
        "study_id": spec.study_id,
        "candidate_count": len(candidate_summaries),
        "all_candidates_admissible": True,
        "candidate_minibatch_order_sha256": next(iter(batch_hashes)),
        "effective_code_fingerprint": code_fingerprint,
        "selection_sha256": selection_sha,
        "official_test_read": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        required=True,
        help="Canonical v3 study directory containing study.resolved.json.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = audit_study(args.study)
    except Exception as exc:
        message = str(exc)
        if not (
            message.startswith("Expected ") and "Provided value:" in message
        ):
            message = (
                "Expected the v3 scientific audit to complete without an "
                f"unexpected exception. Provided value: {type(exc).__name__}: "
                f"{message}"
            )
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": message,
        }
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
