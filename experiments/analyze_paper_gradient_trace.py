#!/usr/bin/env python3
"""Summarize the limited paper Conv gradient/update trace experiment.

The input is the top-level study directory produced by
``run_paper_gradient_trace_jeanzay.slurm``.  The analyzer is intentionally
strict about production coverage, trace schemas, and the matched training
cohort.  Scientific irregularities are preserved in a machine-readable
anomaly report rather than silently filtered.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.gradient_trace import (
    GRADIENT_TRACE_METADATA_SCHEMA,
    GRADIENT_TRACE_SCHEMA,
)
from experiments.paper_gradient_trace_contract import validate_frozen_manifest


SUMMARY_SCHEMA = "paper-conv-gradient-trace-analysis/v1"
ANOMALY_SCHEMA = "paper-conv-gradient-trace-anomalies/v1"
EXPECTED_ARCHITECTURE_ARMS = {
    "conv1": {
        "conv1_baseline_sgd_seed0",
        "conv1_baseline_adam_seed0",
        "conv1_ours_sgd_seed0",
        "conv1_ours_adam_seed0",
        "conv1_legacy_sgd_seed0",
        "conv1_legacy_adam_seed0",
    },
    "conv2": {
        "conv2_baseline_sgd_seed0",
        "conv2_ours_sgd_seed0",
        "conv2_legacy_sgd_seed0",
    },
    "conv3": {
        "conv3_baseline_sgd_seed0",
        "conv3_ours_sgd_seed0",
        "conv3_legacy_sgd_seed0",
    },
}
ARM_PATTERN = re.compile(
    r"^(?P<architecture>conv[123])_"
    r"(?P<scheme>baseline|ours|legacy)_"
    r"(?P<optimizer>sgd|adam)_seed(?P<seed>\d+)$"
)
SUMMARY_METRICS = (
    "loss",
    "gradient_rms",
    "gradient_zero_fraction",
    "raw_optimizer_update_rms",
    "applied_update_rms",
    "raw_optimizer_over_fresh_shadow_rms",
    "applied_over_fresh_shadow_rms",
    "raw_optimizer_fresh_shadow_cosine",
    "applied_fresh_shadow_cosine",
    "applied_update_over_parameter_rms",
    "applied_update_over_initial_parameter_rms",
    "projection_efficiency_l2",
    "projection_changed_fraction",
    "descent_cosine_raw_optimizer_update",
    "descent_cosine_applied_update",
    "lower_bound_fraction_before",
    "lower_bound_fraction_after",
    "upper_bound_fraction_before",
    "upper_bound_fraction_after",
    "gradient_momentum_cosine",
    "gradient_momentum_sign_disagreement_fraction",
    "adam_exp_avg_rms",
    "adam_bias_corrected_exp_avg_rms",
    "adam_exp_avg_sq_mean",
)
FRACTION_METRICS = {
    "gradient_zero_fraction",
    "projection_changed_fraction",
    "lower_bound_fraction_before",
    "lower_bound_fraction_after",
    "upper_bound_fraction_before",
    "upper_bound_fraction_after",
    "gradient_momentum_sign_disagreement_fraction",
}
COSINE_METRICS = {
    "raw_optimizer_fresh_shadow_cosine",
    "applied_fresh_shadow_cosine",
    "descent_cosine_raw_optimizer_update",
    "descent_cosine_applied_update",
    "gradient_momentum_cosine",
}
NONNEGATIVE_METRICS = {
    "loss",
    "gradient_rms",
    "raw_optimizer_update_rms",
    "applied_update_rms",
    "raw_optimizer_over_fresh_shadow_rms",
    "applied_over_fresh_shadow_rms",
    "applied_update_over_parameter_rms",
    "applied_update_over_initial_parameter_rms",
    "projection_efficiency_l2",
    "adam_exp_avg_rms",
    "adam_bias_corrected_exp_avg_rms",
    "adam_exp_avg_sq_mean",
}
SCHEME_COLORS = {
    "baseline": "#4d4d4d",
    "ours": "#1770b8",
    "legacy": "#d97904",
}
OPTIMIZER_LINESTYLES = {"SGD": "-", "Adam": "--"}
ANOMALY_THRESHOLDS = {
    "sgd_raw_over_fresh_absolute_tolerance": 5.0e-3,
    "adam_raw_over_fresh_low": 0.5,
    "adam_raw_over_fresh_high": 2.0,
    "adam_raw_fresh_cosine_low": 0.5,
    "adam_momentum_sign_disagreement_high": 0.30,
    "gradient_zero_fraction_high": 0.95,
    "gradient_rms_collapse_ratio": 1.0e-4,
    "projection_efficiency_low": 0.80,
    "projection_changed_fraction_high": 0.25,
    "bound_occupancy_high": 0.50,
    "upper_bound_occupancy_nontrivial": 1.0e-3,
    "applied_descent_cosine_low": 0.50,
    "relative_applied_update_high": 0.20,
    "relative_applied_update_low": 1.0e-10,
}


@dataclass(frozen=True)
class RunTrace:
    run_dir: Path
    arm_id: str
    architecture: str
    scheme: str
    optimizer: str
    seed: int
    manifest: dict[str, Any]
    config: dict[str, Any]
    metadata: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    frozen_provenance: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing required JSON file: {path}.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read valid JSON from {path}: {error}.") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_index_sequence_hash(indices: Sequence[int]) -> str:
    values = tuple(int(index) for index in indices)
    digest = hashlib.sha256()
    digest.update(b"mnist-index-sequence/v1\0")
    digest.update(len(values).to_bytes(8, byteorder="big", signed=False))
    for value in values:
        if value < 0:
            raise ValueError(f"Expected non-negative source indices; found {value}.")
        digest.update(value.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Expected numeric {label}; found {value!r}.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected numeric {label}; found {value!r}.") from error
    if not math.isfinite(result):
        raise ValueError(f"Expected finite {label}; found {value!r}.")
    return result


def _optional_finite_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, label)


def _quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        return {
            "count": 0,
            "median": None,
            "q10": None,
            "q90": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": int(len(array)),
        "median": float(np.quantile(array, 0.50)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _discover_run_dirs(input_root: Path) -> list[tuple[str, Path]]:
    runs_root = input_root / "runs"
    if not runs_root.is_dir():
        raise ValueError(f"Missing production runs directory: {runs_root}.")
    discovered: list[tuple[str, Path]] = []
    for architecture, expected_arms in EXPECTED_ARCHITECTURE_ARMS.items():
        architecture_dir = runs_root / architecture
        if not architecture_dir.is_dir():
            raise ValueError(f"Missing architecture directory: {architecture_dir}.")
        candidates = sorted(
            path
            for path in architecture_dir.iterdir()
            if path.is_dir() and path.name[:1].isdigit()
        )
        if len(candidates) != len(expected_arms):
            raise ValueError(
                f"Expected {len(expected_arms)} direct production bundles under "
                f"{architecture_dir}; found {len(candidates)}."
            )
        discovered.extend((architecture, path) for path in candidates)
    return discovered


def _parse_arm(arm_id: str) -> tuple[str, str, str, int]:
    match = ARM_PATTERN.fullmatch(arm_id)
    if match is None:
        raise ValueError(f"Unexpected gradient-trace arm id: {arm_id!r}.")
    optimizer = {"sgd": "SGD", "adam": "Adam"}[match.group("optimizer")]
    return (
        match.group("architecture"),
        match.group("scheme"),
        optimizer,
        int(match.group("seed")),
    )


def _required_mapping(value: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"Expected {label}.{key} to be an object.")
    return result


def _load_trace_rows(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Missing or empty gradient trace: {path}.")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {error}."
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            rows.append(value)
    if not rows:
        raise ValueError(f"Gradient trace contains no rows: {path}.")
    return tuple(rows)


def _validate_trace_row(
    row: Mapping[str, Any],
    *,
    run_dir: Path,
    optimizer: str,
    parameter_names: set[str],
) -> None:
    if row.get("schema_version") != GRADIENT_TRACE_SCHEMA:
        raise ValueError(f"Unexpected trace schema in {run_dir}: {row.get('schema_version')!r}.")
    row_optimizer = str(row.get("optimizer"))
    if row_optimizer != optimizer:
        raise ValueError(
            f"Optimizer mismatch in {run_dir}: expected={optimizer}, actual={row_optimizer}."
        )
    name = str(row.get("parameter_name", "")).strip()
    if name not in parameter_names:
        raise ValueError(f"Unexpected traced parameter {name!r} in {run_dir}.")
    epoch = int(row.get("epoch", -1))
    batch = int(row.get("batch", -1))
    total_batches = int(row.get("total_batches", -1))
    global_step = int(row.get("global_step", -1))
    if not 1 <= epoch <= 10 or batch <= 0 or total_batches <= 0:
        raise ValueError(f"Invalid trace coordinates in {run_dir}: {(epoch, batch, total_batches)}.")
    if global_step != (epoch - 1) * total_batches + batch:
        raise ValueError(f"Invalid global step in {run_dir}: {global_step}.")
    source_hash = row.get("source_indices_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError(f"Invalid source-index hash in {run_dir}: {source_hash!r}.")
    source_indices = row.get("source_indices")
    if not isinstance(source_indices, list) or not source_indices:
        raise ValueError(f"Missing source indices in {run_dir} at {(epoch, batch)}.")
    if _stable_index_sequence_hash(source_indices) != source_hash:
        raise ValueError(
            f"Source-index SHA-256 mismatch in {run_dir} at {(epoch, batch)}."
        )
    for metric in SUMMARY_METRICS:
        if metric not in row:
            raise ValueError(f"Missing trace metric {metric!r} in {run_dir}.")
        value = _optional_finite_float(row.get(metric), f"{metric} in {run_dir}")
        if value is None:
            continue
        if metric in FRACTION_METRICS and not -1.0e-9 <= value <= 1.0 + 1.0e-9:
            raise ValueError(f"Out-of-range fraction {metric}={value} in {run_dir}.")
        if metric in COSINE_METRICS and not -1.0 - 1.0e-9 <= value <= 1.0 + 1.0e-9:
            raise ValueError(f"Out-of-range cosine {metric}={value} in {run_dir}.")
        if metric in NONNEGATIVE_METRICS and value < 0.0:
            raise ValueError(f"Negative non-negative metric {metric}={value} in {run_dir}.")
    if optimizer == "SGD":
        if row.get("fresh_shadow_proposal_kind") != "fresh_sgd":
            raise ValueError(f"Unexpected SGD shadow proposal in {run_dir}.")
    else:
        if row.get("fresh_shadow_proposal_kind") != "fresh_adam":
            raise ValueError(f"Unexpected Adam shadow proposal in {run_dir}.")
        for metric in (
            "optimizer_step",
            "adam_exp_avg_rms",
            "adam_exp_avg_sq_mean",
            "gradient_momentum_cosine",
            "gradient_momentum_sign_disagreement_fraction",
        ):
            if _optional_finite_float(row.get(metric), f"{metric} in {run_dir}") is None:
                raise ValueError(f"Missing Adam diagnostic {metric!r} in {run_dir}.")


def _load_run(expected_architecture: str, run_dir: Path) -> RunTrace:
    manifest = _read_json(run_dir / "manifest.json")
    status = _read_json(run_dir / "status.json")
    result = _read_json(run_dir / "result.json")
    config = _read_json(run_dir / "config.json")
    metadata = _read_json(run_dir / "gradient_trace_metadata.json")
    trace_path = run_dir / "gradient_trace.jsonl"

    if bool(manifest.get("smoke")) or bool(result.get("smoke")):
        raise ValueError(f"Refusing to analyze smoke bundle as production: {run_dir}.")
    if status.get("state") != "complete":
        raise ValueError(f"Incomplete run status in {run_dir}: {status.get('state')!r}.")
    completion = _required_mapping(result, "completion", "result")
    if not bool(completion.get("criteria_met")):
        raise ValueError(f"Completion criteria were not met in {run_dir}.")
    if metadata.get("schema_version") != GRADIENT_TRACE_METADATA_SCHEMA:
        raise ValueError(
            f"Unexpected trace metadata schema in {run_dir}: "
            f"{metadata.get('schema_version')!r}."
        )
    if metadata.get("status") != "complete":
        raise ValueError(f"Incomplete gradient trace metadata in {run_dir}.")
    if int(metadata.get("epoch_count", -1)) != 10:
        raise ValueError(f"Expected ten traced epochs in {run_dir}.")
    if int(metadata.get("recorded_steps", -1)) != 50:
        raise ValueError(f"Expected fifty traced optimizer steps in {run_dir}.")
    if int(metadata.get("samples_per_epoch", -1)) != 5:
        raise ValueError(f"Expected five requested trace samples per epoch in {run_dir}.")
    effective_batches = int(metadata.get("effective_batches_per_epoch", -1))
    if effective_batches <= 0:
        raise ValueError(f"Invalid effective batch count in {run_dir}.")
    sampled_positions = metadata.get("sampled_batch_positions")
    if not isinstance(sampled_positions, list) or len(sampled_positions) != 5:
        raise ValueError(f"Expected five sampled batch positions in {run_dir}.")
    if sampled_positions != sorted(set(int(value) for value in sampled_positions)):
        raise ValueError(f"Invalid sampled batch positions in {run_dir}.")
    if min(sampled_positions) < 1 or max(sampled_positions) > effective_batches:
        raise ValueError(f"Sampled batch position outside the epoch in {run_dir}.")

    arm_id = str(manifest.get("arm_id", ""))
    architecture, scheme, optimizer, seed = _parse_arm(arm_id)
    frozen_provenance = validate_frozen_manifest(manifest, arm_id=arm_id)
    if architecture != expected_architecture:
        raise ValueError(
            f"Architecture path/arm mismatch in {run_dir}: "
            f"{expected_architecture!r} versus {architecture!r}."
        )
    if seed != 0:
        raise ValueError(f"Expected seed zero in {run_dir}; found {seed}.")
    for document, label in ((status, "status"), (result, "result")):
        if document.get("arm_id") != arm_id:
            raise ValueError(f"Arm id mismatch between manifest and {label} in {run_dir}.")
    resolved = _required_mapping(
        _required_mapping(manifest, "configuration", "manifest"),
        "resolved",
        "manifest.configuration",
    )
    resolved_optimizer = _required_mapping(resolved, "optimizer", "resolved")
    config_optimizer = _required_mapping(config, "optimizer", "config")
    if str(resolved_optimizer.get("name")) != optimizer:
        raise ValueError(f"Manifest optimizer does not match arm id in {run_dir}.")
    if str(config_optimizer.get("name")) != optimizer:
        raise ValueError(f"Resolved config optimizer does not match arm id in {run_dir}.")
    training = _required_mapping(config, "training", "config")
    if int(training.get("epochs", -1)) != 10:
        raise ValueError(f"Expected the ten-epoch diagnostic budget in {run_dir}.")

    parameter_names_raw = metadata.get("parameter_names")
    if not isinstance(parameter_names_raw, list) or not parameter_names_raw:
        raise ValueError(f"Missing parameter names in {run_dir} trace metadata.")
    parameter_names = {str(value).strip() for value in parameter_names_raw}
    if len(parameter_names) != len(parameter_names_raw) or "" in parameter_names:
        raise ValueError(f"Invalid parameter-name inventory in {run_dir}.")
    rows = _load_trace_rows(trace_path)
    if _sha256_file(trace_path) != metadata.get("trace_sha256"):
        raise ValueError(f"Gradient trace SHA-256 mismatch in {run_dir}.")
    if trace_path.stat().st_size != int(metadata.get("trace_size_bytes", -1)):
        raise ValueError(f"Gradient trace size mismatch in {run_dir}.")
    if len(rows) != int(metadata.get("recorded_rows", -1)):
        raise ValueError(f"Gradient trace row-count mismatch in {run_dir}.")
    for row in rows:
        _validate_trace_row(
            row,
            run_dir=run_dir,
            optimizer=optimizer,
            parameter_names=parameter_names,
        )
        if int(row["total_batches"]) != effective_batches:
            raise ValueError(f"Trace batch-count drift in {run_dir}.")

    expected_batches = {int(value) for value in sampled_positions}
    expected_keys = {
        (epoch, batch, parameter)
        for epoch in range(1, 11)
        for batch in expected_batches
        for parameter in parameter_names
    }
    actual_keys = {
        (int(row["epoch"]), int(row["batch"]), str(row["parameter_name"]).strip())
        for row in rows
    }
    if len(actual_keys) != len(rows) or actual_keys != expected_keys:
        raise ValueError(f"Missing or duplicate trace coordinates in {run_dir}.")

    sampled_batches_raw = metadata.get("sampled_source_batches")
    if not isinstance(sampled_batches_raw, list):
        raise ValueError(f"Missing sampled-source metadata in {run_dir}.")
    sampled_batches = {
        (int(item["epoch"]), int(item["batch"])): (
            str(item["source_indices_sha256"]),
            tuple(int(value) for value in item["source_indices"]),
        )
        for item in sampled_batches_raw
    }
    if len(sampled_batches) != 50:
        raise ValueError(f"Expected fifty unique sampled batches in {run_dir}.")
    for key, (source_hash, source_indices) in sampled_batches.items():
        if _stable_index_sequence_hash(source_indices) != source_hash:
            raise ValueError(
                f"Sampled-source SHA-256 mismatch in {run_dir} at {key}."
            )
    for row in rows:
        key = (int(row["epoch"]), int(row["batch"]))
        expected_source = sampled_batches.get(key)
        actual_source = (
            str(row["source_indices_sha256"]),
            tuple(int(value) for value in row["source_indices"]),
        )
        if expected_source != actual_source:
            raise ValueError(f"Trace/source-index metadata mismatch in {run_dir} at {key}.")

    return RunTrace(
        run_dir=run_dir.resolve(),
        arm_id=arm_id,
        architecture=architecture,
        scheme=scheme,
        optimizer=optimizer,
        seed=seed,
        manifest=manifest,
        config=config,
        metadata=metadata,
        rows=rows,
        frozen_provenance=frozen_provenance,
    )


def _load_and_validate_runs(input_root: Path) -> tuple[list[RunTrace], dict[str, Any]]:
    runs = [
        _load_run(architecture, run_dir)
        for architecture, run_dir in _discover_run_dirs(input_root)
    ]
    actual_by_architecture = {
        architecture: {
            run.arm_id for run in runs if run.architecture == architecture
        }
        for architecture in EXPECTED_ARCHITECTURE_ARMS
    }
    if actual_by_architecture != EXPECTED_ARCHITECTURE_ARMS:
        raise ValueError(
            "Production arm coverage does not match the frozen twelve-arm surface: "
            f"{actual_by_architecture!r}."
        )

    reference = runs[0]
    reference_provenance = reference.metadata.get("dataset_provenance")
    reference_batches = reference.metadata.get("sampled_source_batches")
    if not isinstance(reference_provenance, dict):
        raise ValueError(f"Missing dataset provenance in {reference.run_dir}.")
    order_hashes = reference_provenance.get("train_batch_order_sha256")
    if not isinstance(order_hashes, list) or len(order_hashes) != 10:
        raise ValueError("Expected ten epoch-order hashes in gradient-trace provenance.")
    for run in runs[1:]:
        if run.metadata.get("dataset_provenance") != reference_provenance:
            raise ValueError(
                "Cross-arm cohort or epoch-order mismatch: "
                f"reference={reference.arm_id}, differing={run.arm_id}."
            )
        if run.metadata.get("sampled_source_batches") != reference_batches:
            raise ValueError(
                "Cross-arm sampled-source mismatch: "
                f"reference={reference.arm_id}, differing={run.arm_id}."
            )
    cohort_audit = {
        "status": "pass",
        "reference_arm_id": reference.arm_id,
        "matched_arm_count": len(runs),
        "dataset_provenance_sha256": _canonical_hash(reference_provenance),
        "sampled_source_batches_sha256": _canonical_hash(reference_batches),
        "train_indices_sha256": reference_provenance.get("train_indices_sha256"),
        "validation_indices_sha256": reference_provenance.get(
            "validation_indices_sha256"
        ),
        "first_epoch_batch_order_sha256": reference_provenance.get(
            "first_epoch_batch_order_sha256"
        ),
        "train_batch_order_sha256": list(order_hashes),
    }
    return sorted(runs, key=lambda run: run.arm_id), cohort_audit


def _parameter_sort_key(name: str) -> tuple[int, int, str]:
    match = re.match(r"^(ConvWeight|DenseWeight|Bias)_(\d+)$", name)
    if match is None:
        return (99, 99, name)
    kind_order = {"ConvWeight": 0, "DenseWeight": 1, "Bias": 2}
    return (kind_order[match.group(1)], int(match.group(2)), name)


def _summarize(runs: Sequence[RunTrace]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for run in runs:
        parameter_names = sorted(
            {str(row["parameter_name"]).strip() for row in run.rows},
            key=_parameter_sort_key,
        )
        for epoch in range(1, 11):
            for parameter_name in parameter_names:
                samples = [
                    row
                    for row in run.rows
                    if int(row["epoch"]) == epoch
                    and str(row["parameter_name"]).strip() == parameter_name
                ]
                if len(samples) != 5:
                    raise ValueError(
                        f"Expected five samples for {run.arm_id}, epoch={epoch}, "
                        f"parameter={parameter_name}; found {len(samples)}."
                    )
                result: dict[str, Any] = {
                    "arm_id": run.arm_id,
                    "architecture": run.architecture,
                    "scheme": run.scheme,
                    "optimizer": run.optimizer,
                    "seed": run.seed,
                    "epoch": epoch,
                    "parameter_name": parameter_name,
                    "sample_count": len(samples),
                    "learning_rate": _finite_float(
                        samples[0].get("learning_rate"), "learning rate"
                    ),
                }
                for metric in SUMMARY_METRICS:
                    values = [
                        value
                        for sample in samples
                        if (
                            value := _optional_finite_float(
                                sample.get(metric),
                                f"{metric} for {run.arm_id}/{parameter_name}",
                            )
                        )
                        is not None
                    ]
                    stats = _quantiles(values)
                    result[f"{metric}_count"] = stats["count"]
                    for suffix in ("median", "q10", "q90", "minimum", "maximum"):
                        result[f"{metric}_{suffix}"] = stats[suffix]
                summary_rows.append(result)
    return summary_rows


def _audit_optimizers(runs: Sequence[RunTrace]) -> dict[str, Any]:
    sgd_rows = [row for run in runs if run.optimizer == "SGD" for row in run.rows]
    sgd_ratios = [
        _finite_float(row["raw_optimizer_over_fresh_shadow_rms"], "SGD ratio")
        for row in sgd_rows
        if row.get("raw_optimizer_over_fresh_shadow_rms") is not None
    ]
    tolerance = ANOMALY_THRESHOLDS["sgd_raw_over_fresh_absolute_tolerance"]
    maximum_deviation = max((abs(value - 1.0) for value in sgd_ratios), default=None)
    sgd_cosines = [
        _finite_float(row["raw_optimizer_fresh_shadow_cosine"], "SGD cosine")
        for row in sgd_rows
        if row.get("raw_optimizer_fresh_shadow_cosine") is not None
    ]
    maximum_cosine_deviation = max(
        (abs(value - 1.0) for value in sgd_cosines), default=None
    )
    sgd_status = (
        "pass"
        if sgd_ratios
        and len(sgd_ratios) == len(sgd_rows)
        and len(sgd_cosines) == len(sgd_rows)
        and maximum_deviation is not None
        and maximum_cosine_deviation is not None
        and maximum_deviation <= tolerance
        and maximum_cosine_deviation <= tolerance
        else "fail"
    )

    adam_rows = [row for run in runs if run.optimizer == "Adam" for row in run.rows]
    adam_metrics = {}
    for metric in (
        "raw_optimizer_over_fresh_shadow_rms",
        "raw_optimizer_fresh_shadow_cosine",
        "gradient_momentum_cosine",
        "gradient_momentum_sign_disagreement_fraction",
    ):
        adam_metrics[metric] = _quantiles(
            _finite_float(row[metric], f"Adam {metric}")
            for row in adam_rows
            if row.get(metric) is not None
        )
    return {
        "sgd_fresh_shadow_verification": {
            "status": sgd_status,
            "checked_row_count": len(sgd_ratios),
            "expected_row_count": len(sgd_rows),
            "absolute_tolerance": tolerance,
            "maximum_absolute_deviation_from_one": maximum_deviation,
            "ratio_distribution": _quantiles(sgd_ratios),
            "maximum_cosine_deviation_from_one": maximum_cosine_deviation,
            "cosine_distribution": _quantiles(sgd_cosines),
        },
        "adam_actual_update_and_momentum": {
            "status": "measured" if adam_rows else "not_present",
            "checked_row_count": len(adam_rows),
            "distributions": adam_metrics,
        },
    }


def _anomaly(
    *,
    kind: str,
    severity: str,
    row: Mapping[str, Any],
    metric: str,
    observed: float,
    expectation: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "severity": severity,
        "arm_id": row["arm_id"],
        "architecture": row["architecture"],
        "scheme": row["scheme"],
        "optimizer": row["optimizer"],
        "epoch": int(row["epoch"]),
        "parameter_name": row["parameter_name"],
        "metric": metric,
        "observed_epoch_median": observed,
        "expectation": expectation,
    }


def _detect_anomalies(
    summary_rows: Sequence[Mapping[str, Any]], optimizer_audit: Mapping[str, Any]
) -> dict[str, Any]:
    anomalies: list[dict[str, Any]] = []
    first_gradient = {
        (row["arm_id"], row["parameter_name"]): row["gradient_rms_median"]
        for row in summary_rows
        if int(row["epoch"]) == 1
    }
    for row in summary_rows:
        optimizer = str(row["optimizer"])
        gradient_rms = row["gradient_rms_median"]
        zero_fraction = row["gradient_zero_fraction_median"]
        projection_efficiency = row["projection_efficiency_l2_median"]
        projection_changed = row["projection_changed_fraction_median"]
        lower_after = row["lower_bound_fraction_after_median"]
        upper_after = row["upper_bound_fraction_after_median"]
        descent_cosine = row["descent_cosine_applied_update_median"]
        relative_update = row["applied_update_over_parameter_rms_median"]
        raw_over_fresh = row["raw_optimizer_over_fresh_shadow_rms_median"]
        raw_fresh_cosine = row["raw_optimizer_fresh_shadow_cosine_median"]

        if zero_fraction is not None and zero_fraction >= ANOMALY_THRESHOLDS[
            "gradient_zero_fraction_high"
        ]:
            anomalies.append(
                _anomaly(
                    kind="high_gradient_sparsity",
                    severity="warning",
                    row=row,
                    metric="gradient_zero_fraction",
                    observed=zero_fraction,
                    expectation="epoch median below 0.95",
                )
            )
        initial_gradient = first_gradient[(row["arm_id"], row["parameter_name"])]
        if (
            gradient_rms is not None
            and initial_gradient is not None
            and initial_gradient > 0.0
            and gradient_rms / initial_gradient
            < ANOMALY_THRESHOLDS["gradient_rms_collapse_ratio"]
        ):
            anomalies.append(
                _anomaly(
                    kind="gradient_rms_collapse",
                    severity="warning",
                    row=row,
                    metric="gradient_rms_epoch1_ratio",
                    observed=gradient_rms / initial_gradient,
                    expectation="epoch median at least 1e-4 of epoch-1 median",
                )
            )
        if (
            projection_efficiency is not None
            and projection_efficiency
            < ANOMALY_THRESHOLDS["projection_efficiency_low"]
        ):
            anomalies.append(
                _anomaly(
                    kind="low_projection_efficiency",
                    severity="warning",
                    row=row,
                    metric="projection_efficiency_l2",
                    observed=projection_efficiency,
                    expectation="epoch median at least 0.80",
                )
            )
        if (
            projection_changed is not None
            and projection_changed
            > ANOMALY_THRESHOLDS["projection_changed_fraction_high"]
        ):
            anomalies.append(
                _anomaly(
                    kind="frequent_projection",
                    severity="warning",
                    row=row,
                    metric="projection_changed_fraction",
                    observed=projection_changed,
                    expectation="epoch median at most 0.25",
                )
            )
        if (
            lower_after is not None
            and lower_after >= ANOMALY_THRESHOLDS["bound_occupancy_high"]
        ):
            anomalies.append(
                _anomaly(
                    kind="high_lower_bound_occupancy",
                    severity="warning",
                    row=row,
                    metric="lower_bound_fraction_after",
                    observed=lower_after,
                    expectation="epoch median below 0.50",
                )
            )
        if (
            upper_after is not None
            and upper_after
            > ANOMALY_THRESHOLDS["upper_bound_occupancy_nontrivial"]
        ):
            anomalies.append(
                _anomaly(
                    kind="upper_bound_occupancy",
                    severity="warning",
                    row=row,
                    metric="upper_bound_fraction_after",
                    observed=upper_after,
                    expectation="epoch median at most 1e-3",
                )
            )
        if (
            descent_cosine is not None
            and descent_cosine < ANOMALY_THRESHOLDS["applied_descent_cosine_low"]
        ):
            anomalies.append(
                _anomaly(
                    kind="poor_applied_descent_alignment",
                    severity="critical" if descent_cosine < 0.0 else "warning",
                    row=row,
                    metric="descent_cosine_applied_update",
                    observed=descent_cosine,
                    expectation="epoch median at least 0.50",
                )
            )
        if relative_update is not None:
            if relative_update > ANOMALY_THRESHOLDS["relative_applied_update_high"]:
                anomalies.append(
                    _anomaly(
                        kind="large_relative_applied_update",
                        severity="warning",
                        row=row,
                        metric="applied_update_over_parameter_rms",
                        observed=relative_update,
                        expectation="epoch median at most 0.20",
                    )
                )
            elif (
                gradient_rms is not None
                and gradient_rms > 1.0e-12
                and relative_update
                < ANOMALY_THRESHOLDS["relative_applied_update_low"]
            ):
                anomalies.append(
                    _anomaly(
                        kind="negligible_relative_applied_update",
                        severity="warning",
                        row=row,
                        metric="applied_update_over_parameter_rms",
                        observed=relative_update,
                        expectation="epoch median at least 1e-10 for nonzero gradients",
                    )
                )

        if optimizer == "SGD" and raw_over_fresh is not None:
            deviation = abs(raw_over_fresh - 1.0)
            if deviation > ANOMALY_THRESHOLDS[
                "sgd_raw_over_fresh_absolute_tolerance"
            ]:
                anomalies.append(
                    _anomaly(
                        kind="sgd_fresh_shadow_mismatch",
                        severity="critical",
                        row=row,
                        metric="raw_optimizer_over_fresh_shadow_rms",
                        observed=raw_over_fresh,
                        expectation="epoch median within 0.005 of one",
                    )
                )
            if (
                raw_fresh_cosine is not None
                and abs(raw_fresh_cosine - 1.0)
                > ANOMALY_THRESHOLDS[
                    "sgd_raw_over_fresh_absolute_tolerance"
                ]
            ):
                anomalies.append(
                    _anomaly(
                        kind="sgd_fresh_shadow_direction_mismatch",
                        severity="critical",
                        row=row,
                        metric="raw_optimizer_fresh_shadow_cosine",
                        observed=raw_fresh_cosine,
                        expectation="epoch median within 0.005 of one",
                    )
                )
        if optimizer == "Adam":
            if raw_over_fresh is not None and not (
                ANOMALY_THRESHOLDS["adam_raw_over_fresh_low"]
                <= raw_over_fresh
                <= ANOMALY_THRESHOLDS["adam_raw_over_fresh_high"]
            ):
                anomalies.append(
                    _anomaly(
                        kind="adam_actual_fresh_scale_divergence",
                        severity="warning",
                        row=row,
                        metric="raw_optimizer_over_fresh_shadow_rms",
                        observed=raw_over_fresh,
                        expectation="epoch median in [0.5, 2.0]",
                    )
                )
            if (
                raw_fresh_cosine is not None
                and raw_fresh_cosine
                < ANOMALY_THRESHOLDS["adam_raw_fresh_cosine_low"]
            ):
                anomalies.append(
                    _anomaly(
                        kind="adam_actual_fresh_direction_divergence",
                        severity="critical" if raw_fresh_cosine < 0.0 else "warning",
                        row=row,
                        metric="raw_optimizer_fresh_shadow_cosine",
                        observed=raw_fresh_cosine,
                        expectation="epoch median at least 0.50",
                    )
                )
            sign_disagreement = row[
                "gradient_momentum_sign_disagreement_fraction_median"
            ]
            if (
                sign_disagreement is not None
                and sign_disagreement
                > ANOMALY_THRESHOLDS[
                    "adam_momentum_sign_disagreement_high"
                ]
            ):
                anomalies.append(
                    _anomaly(
                        kind="adam_gradient_momentum_sign_disagreement",
                        severity="warning",
                        row=row,
                        metric="gradient_momentum_sign_disagreement_fraction",
                        observed=sign_disagreement,
                        expectation="epoch median at most 0.30",
                    )
                )

    if (
        optimizer_audit["sgd_fresh_shadow_verification"]["status"] != "pass"
        and not any(item["kind"] == "sgd_fresh_shadow_mismatch" for item in anomalies)
    ):
        anomalies.append(
            {
                "kind": "sgd_fresh_shadow_audit_failure",
                "severity": "critical",
                "metric": "raw_optimizer_over_fresh_shadow_rms",
                "observed": optimizer_audit["sgd_fresh_shadow_verification"],
                "expectation": "every finite SGD trace row within 0.005 of one",
            }
        )
    anomalies.sort(
        key=lambda item: (
            item.get("severity", ""),
            item.get("kind", ""),
            item.get("arm_id", ""),
            int(item.get("epoch", -1)),
            item.get("parameter_name", ""),
        )
    )
    counts_by_kind: dict[str, int] = {}
    counts_by_severity: dict[str, int] = {}
    for item in anomalies:
        counts_by_kind[item["kind"]] = counts_by_kind.get(item["kind"], 0) + 1
        counts_by_severity[item["severity"]] = (
            counts_by_severity.get(item["severity"], 0) + 1
        )
    return {
        "schema_version": ANOMALY_SCHEMA,
        "status": "anomalies_found" if anomalies else "none",
        "thresholds": dict(ANOMALY_THRESHOLDS),
        "counts_by_kind": counts_by_kind,
        "counts_by_severity": counts_by_severity,
        "anomalies": anomalies,
    }


def _series(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    arm_id: str,
    parameter_name: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = sorted(
        (
            row
            for row in summary_rows
            if row["arm_id"] == arm_id and row["parameter_name"] == parameter_name
        ),
        key=lambda row: int(row["epoch"]),
    )
    epochs = np.asarray([int(row["epoch"]) for row in rows], dtype=np.int64)
    median = np.asarray([row[f"{metric}_median"] for row in rows], dtype=np.float64)
    q10 = np.asarray([row[f"{metric}_q10"] for row in rows], dtype=np.float64)
    q90 = np.asarray([row[f"{metric}_q90"] for row in rows], dtype=np.float64)
    return epochs, median, q10, q90


def _plot_metric_grid(
    *,
    architecture: str,
    summary_rows: Sequence[Mapping[str, Any]],
    metric_specs: Sequence[tuple[str, str, str]],
    output_path: Path,
) -> None:
    architecture_rows = [
        row for row in summary_rows if row["architecture"] == architecture
    ]
    parameters = sorted(
        {str(row["parameter_name"]) for row in architecture_rows},
        key=_parameter_sort_key,
    )
    arms = sorted(
        {
            (str(row["arm_id"]), str(row["scheme"]), str(row["optimizer"]))
            for row in architecture_rows
        },
        key=lambda value: (value[1], value[2]),
    )
    figure, axes = plt.subplots(
        len(parameters),
        len(metric_specs),
        figsize=(5.0 * len(metric_specs), max(3.0, 2.7 * len(parameters))),
        squeeze=False,
        sharex=True,
    )
    for row_index, parameter in enumerate(parameters):
        for column_index, (metric, title, scale) in enumerate(metric_specs):
            axis = axes[row_index][column_index]
            for arm_id, scheme, optimizer in arms:
                epochs, median, q10, q90 = _series(
                    architecture_rows,
                    arm_id=arm_id,
                    parameter_name=parameter,
                    metric=metric,
                )
                if not len(epochs) or np.all(np.isnan(median)):
                    continue
                label = f"{scheme} {optimizer}"
                axis.plot(
                    epochs,
                    median,
                    label=label,
                    color=SCHEME_COLORS[scheme],
                    linestyle=OPTIMIZER_LINESTYLES[optimizer],
                    marker="o",
                    markersize=2.8,
                    linewidth=1.5,
                )
                axis.fill_between(
                    epochs,
                    q10,
                    q90,
                    color=SCHEME_COLORS[scheme],
                    alpha=0.10,
                    linewidth=0,
                )
            if scale == "log":
                axis.set_yscale("log")
            elif scale == "fraction":
                axis.set_ylim(-0.03, 1.03)
            elif scale == "cosine":
                axis.set_ylim(-1.03, 1.03)
                axis.axhline(0.0, color="#aaaaaa", linewidth=0.7)
            if metric == "raw_optimizer_over_fresh_shadow_rms":
                axis.axhline(1.0, color="#999999", linewidth=0.8, linestyle=":")
            if metric == "projection_efficiency_l2":
                axis.axhline(1.0, color="#999999", linewidth=0.8, linestyle=":")
            if row_index == 0:
                axis.set_title(title)
            if column_index == 0:
                axis.set_ylabel(parameter)
            if row_index == len(parameters) - 1:
                axis.set_xlabel("epoch")
            axis.set_xticks(range(1, 11))
            axis.grid(True, alpha=0.22)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=min(6, len(handles)))
    figure.suptitle(f"{architecture.upper()} sampled training gradients and updates", y=1.002)
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _make_plots(
    summary_rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> list[str]:
    plot_paths: list[str] = []
    for architecture in EXPECTED_ARCHITECTURE_ARMS:
        gradient_path = output_dir / f"{architecture}_gradient_scale_and_sparsity.png"
        _plot_metric_grid(
            architecture=architecture,
            summary_rows=summary_rows,
            metric_specs=(
                ("gradient_rms", "gradient RMS (median, q10-q90)", "log"),
                ("gradient_zero_fraction", "near-zero gradient fraction", "fraction"),
            ),
            output_path=gradient_path,
        )
        plot_paths.append(gradient_path.name)

        update_path = output_dir / f"{architecture}_update_and_projection.png"
        _plot_metric_grid(
            architecture=architecture,
            summary_rows=summary_rows,
            metric_specs=(
                (
                    "applied_update_over_parameter_rms",
                    "applied update / parameter RMS",
                    "log",
                ),
                (
                    "raw_optimizer_over_fresh_shadow_rms",
                    "actual raw / fresh-shadow RMS",
                    "log",
                ),
                ("projection_efficiency_l2", "projection efficiency", "fraction"),
                (
                    "lower_bound_fraction_after",
                    "post-step lower-bound occupancy",
                    "fraction",
                ),
            ),
            output_path=update_path,
        )
        plot_paths.append(update_path.name)

    adam_path = output_dir / "conv1_adam_momentum_diagnostics.png"
    adam_rows = [row for row in summary_rows if row["optimizer"] == "Adam"]
    _plot_metric_grid(
        architecture="conv1",
        summary_rows=adam_rows,
        metric_specs=(
            (
                "raw_optimizer_over_fresh_shadow_rms",
                "actual raw / fresh-shadow RMS",
                "log",
            ),
            (
                "raw_optimizer_fresh_shadow_cosine",
                "actual/fresh update cosine",
                "cosine",
            ),
            (
                "gradient_momentum_cosine",
                "gradient/momentum cosine",
                "cosine",
            ),
            (
                "gradient_momentum_sign_disagreement_fraction",
                "gradient/momentum sign disagreement",
                "fraction",
            ),
        ),
        output_path=adam_path,
    )
    plot_paths.append(adam_path.name)
    return plot_paths


def analyze(input_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    input_path = Path(input_root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    runs, cohort_audit = _load_and_validate_runs(input_path)
    summary_rows = _summarize(runs)
    optimizer_audit = _audit_optimizers(runs)
    anomaly_report = _detect_anomalies(summary_rows, optimizer_audit)
    output_path.mkdir(parents=True, exist_ok=True)

    inventory_rows = [
        {
            "arm_id": run.arm_id,
            "architecture": run.architecture,
            "scheme": run.scheme,
            "optimizer": run.optimizer,
            "seed": run.seed,
            "run_dir": str(run.run_dir),
            "trace_rows": len(run.rows),
            "trace_steps": int(run.metadata["recorded_steps"]),
            "trace_sha256": run.metadata["trace_sha256"],
            "source_config_sha256": run.frozen_provenance["source_config_sha256"],
            "source_commit": run.frozen_provenance["source_commit"],
            "source_archive_sha256": run.frozen_provenance[
                "source_archive_sha256"
            ],
        }
        for run in runs
    ]
    _write_csv(output_path / "run_inventory.csv", inventory_rows)
    _write_csv(
        output_path / "gradient_trace_epoch_parameter_summary.csv", summary_rows
    )
    _write_json(output_path / "gradient_trace_anomalies.json", anomaly_report)
    plot_paths = _make_plots(summary_rows, output_path)

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "input_root": str(input_path),
        "output_dir": str(output_path),
        "run_count": len(runs),
        "arm_ids": [run.arm_id for run in runs],
        "frozen_provenance_audit": {
            "status": "pass",
            "runs": [run.frozen_provenance for run in runs],
        },
        "cohort_and_epoch_order_audit": cohort_audit,
        "optimizer_audit": optimizer_audit,
        "anomaly_report": {
            "path": "gradient_trace_anomalies.json",
            "status": anomaly_report["status"],
            "counts_by_kind": anomaly_report["counts_by_kind"],
            "counts_by_severity": anomaly_report["counts_by_severity"],
        },
        "outputs": {
            "run_inventory_csv": "run_inventory.csv",
            "epoch_parameter_summary_csv": "gradient_trace_epoch_parameter_summary.csv",
            "plots": plot_paths,
        },
        "epoch_parameter_summaries": summary_rows,
    }
    _write_json(output_path / "gradient_trace_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Top-level gradient-trace study directory containing runs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for CSV, JSON, and PNG analysis outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = analyze(args.input_root, args.output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "run_count": summary["run_count"],
                "cohort_audit": summary["cohort_and_epoch_order_audit"]["status"],
                "sgd_shadow_audit": summary["optimizer_audit"][
                    "sgd_fresh_shadow_verification"
                ]["status"],
                "anomalies": summary["anomaly_report"]["counts_by_severity"],
                "output_dir": summary["output_dir"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
