#!/usr/bin/env python3
"""Analyze bounded-uniform, zero-bias Conv LR-search shards read-only.

The analyzer treats terminal transport receipts and canonical candidate bundles
as evidence, not as hints.  It can also snapshot an in-flight study: missing,
probe-only, running, unresolved, and rejected rows remain visible and incomplete
coverage is never promoted to a complete study.  Ordinary-MNIST accuracy is a
selection diagnostic; no LR winner is inferred below the configured accuracy
gate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from experiments.reporting import (
        MANIFEST_SCHEMA,
        RESULT_SCHEMA,
        STATUS_SCHEMA,
        atomic_write_json,
        sha256_file,
        validate_run,
    )
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from reporting import (  # type: ignore[no-redef]
        MANIFEST_SCHEMA,
        RESULT_SCHEMA,
        STATUS_SCHEMA,
        atomic_write_json,
        sha256_file,
        validate_run,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_CONFIG = (
    REPO_ROOT
    / "configs/conv/"
    "perfectdiode_bounded_uniform_zero_bias_lr_search_conv123_seed0_20260810_v1.json"
)
ANALYSIS_SCHEMA = (
    "perfectdiode-bounded-uniform-zero-bias-lr-search-analysis/v1"
)
STUDY_SCHEMA = "perfectdiode-conv123-bounded-uniform-zero-bias-rho-study/v1"
RESOLVED_SCHEMA = (
    "perfectdiode-conv123-bounded-uniform-zero-bias-rho-resolved/v1"
)
SELECTION_SCHEMA = (
    "perfectdiode-conv123-bounded-uniform-zero-bias-rho-surface/v1"
)
RECEIPT_SCHEMA = (
    "perfectdiode-conv123-bounded-uniform-zero-bias-rho-transport-receipt/v1"
)
BOUNDED_WEIGHT_PREFIXES = ("ConvWeight_", "DenseWeight_")
COMPLETED_CELL_STATUSES = {"complete", "candidate_rejected_post_training_tk"}
TERMINAL_REJECTION_PREFIXES = ("canary_rejected_", "candidate_rejected_")
SCHEME_COLORS = {
    "baseline": "#4c78a8",
    "ours": "#f58518",
    "legacy": "#54a24b",
}
OPTIMIZER_MARKERS = {"SGD": "o", "Adam": "s"}
PARAMETER_OCCUPANCY_FIELDS = (
    "study_id",
    "surface_index",
    "surface_id",
    "source_shard_root",
    "target",
    "initializer",
    "architecture",
    "scheme",
    "optimizer",
    "cell_index",
    "cell_id",
    "rho_conv",
    "rho_dense",
    "status",
    "surface_status",
    "receipt_status",
    "final_validation_accuracy",
    "final_validation_accuracy_percent",
    "learning_rate",
    "learning_rates_by_parameter_json",
    "checkpoint_path",
    "checkpoint_sha256",
    "parameter_index",
    "parameter_name",
    "parameter_kind",
    "parameter_dtype",
    "parameter_shape_json",
    "num_weights",
    "exact_lower_bound_count",
    "exact_lower_bound_percent",
    "exact_upper_bound_count",
    "exact_upper_bound_percent",
    "exact_either_bound_count",
    "exact_either_bound_percent",
    "outside_bound_count",
    "outside_bound_percent",
)
CORRELATION_FIELDS = (
    "surface_index",
    "surface_id",
    "target",
    "initializer",
    "architecture",
    "scheme",
    "optimizer",
    "terminal_status",
    "receipt_status",
    "canonical_numeric_candidate_count",
    "minimum_candidate_count",
    "pearson_r_accuracy_vs_exact_either_bound_percent",
    "spearman_r_accuracy_vs_exact_either_bound_percent",
    "status",
    "candidate_cell_ids_json",
    "descriptive_non_causal",
    "interpretation",
)


class IncompleteCoverageError(RuntimeError):
    """Raised after partial outputs are written when full coverage was required."""

    def __init__(self, report: Mapping[str, Any]):
        coverage = report["coverage"]
        super().__init__(
            "LR-search snapshot is incomplete: "
            f"terminal={coverage['terminal_surface_count']}/"
            f"{coverage['expected_surface_count']}, "
            f"valid_receipts={coverage['valid_receipt_count']}, "
            f"invalid_completed_bundles="
            f"{coverage['invalid_completed_bundle_count']}."
        )
        self.report = report


@dataclass(frozen=True)
class StudyContract:
    path: Path
    payload: dict[str, Any]
    sha256: str
    study_id: str
    evidence_class: str
    surfaces: tuple[dict[str, Any], ...]
    weight_min: float
    weight_max: float
    accuracy_gate: float


@dataclass(frozen=True)
class Shard:
    root: Path
    target: str
    resolved: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Expected finite {label}, got {value!r}.")
    return float(value)


def _finite_accuracy(value: Any, *, label: str) -> float:
    number = _finite_float(value, label=label)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"Expected {label} in [0,1], got {value!r}.")
    return number


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _study_contract(path: Path) -> StudyContract:
    path = Path(path).expanduser().resolve()
    study = _load_json(path)
    if study.get("schema_version") != STUDY_SCHEMA:
        raise ValueError(f"Unexpected LR-search study schema in {path}.")
    study_id = study.get("study_id")
    if not isinstance(study_id, str) or not study_id:
        raise ValueError(f"Missing study_id in {path}.")
    evidence_class = study.get("evidence_class")
    if evidence_class != "ordinary_mnist_selection":
        raise ValueError(
            f"Expected ordinary_mnist_selection evidence in {path}, "
            f"got {evidence_class!r}."
        )

    scope = study.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError(f"Missing scope in {path}.")
    axes: dict[str, list[str]] = {}
    for key in ("initializers", "architectures", "schemes", "optimizers"):
        values = scope.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise ValueError(f"Invalid scope.{key} in {path}.")
        axes[key] = list(values)
    if axes["initializers"] != ["bounded_uniform"]:
        raise ValueError(
            "This analyzer requires the bounded_uniform initializer-only study."
        )

    bias = study.get("bias_contract")
    if not isinstance(bias, Mapping) or bias.get("learning_rate") != 0.0:
        raise ValueError("The analyzed study must declare exact zero bias LR.")
    model = study.get("model")
    if not isinstance(model, Mapping):
        raise ValueError(f"Missing model contract in {path}.")
    weight_min = _finite_float(model.get("weight_min"), label="weight_min")
    weight_max = _finite_float(model.get("weight_max"), label="weight_max")
    if not 0.0 < weight_min < weight_max:
        raise ValueError(f"Invalid weight interval {weight_min} .. {weight_max}.")
    search = study.get("rho_search")
    if not isinstance(search, Mapping):
        raise ValueError(f"Missing rho_search in {path}.")
    accuracy_gate = _finite_accuracy(
        search.get("minimum_validation_accuracy"),
        label="minimum_validation_accuracy",
    )

    surfaces: list[dict[str, Any]] = []
    for initializer in axes["initializers"]:
        for architecture in axes["architectures"]:
            for scheme in axes["schemes"]:
                for optimizer in axes["optimizers"]:
                    index = len(surfaces)
                    surfaces.append(
                        {
                            "index": index,
                            "initializer": initializer,
                            "architecture": architecture,
                            "scheme": scheme,
                            "optimizer": optimizer,
                            "surface_id": (
                                f"{initializer}__{architecture}__{scheme}__"
                                f"{optimizer.lower()}"
                            ),
                        }
                    )
    return StudyContract(
        path=path,
        payload=study,
        sha256=sha256_file(path),
        study_id=study_id,
        evidence_class=evidence_class,
        surfaces=tuple(surfaces),
        weight_min=weight_min,
        weight_max=weight_max,
        accuracy_gate=accuracy_gate,
    )


def discover_shard_roots(
    *, study_root: Path | None, shard_roots: Sequence[Path]
) -> list[Path]:
    """Return unique authoritative local shard roots in deterministic order."""
    discovered: list[Path] = []
    if study_root is not None:
        resolved_study_root = Path(study_root).expanduser().resolve()
        shard_parent = resolved_study_root / "shards"
        if not shard_parent.is_dir():
            raise ValueError(f"Study root has no shards directory: {shard_parent}.")
        discovered.extend(
            path.parent.resolve()
            for path in sorted(shard_parent.rglob("study.resolved.json"))
        )
    discovered.extend(Path(path).expanduser().resolve() for path in shard_roots)
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in discovered:
        if root in seen:
            continue
        if not root.is_dir():
            raise ValueError(f"Shard root is not a directory: {root}.")
        if not (root / "study.resolved.json").is_file():
            raise ValueError(f"Shard root lacks study.resolved.json: {root}.")
        unique.append(root)
        seen.add(root)
    if not unique:
        raise ValueError("No authoritative local shard roots were supplied or discovered.")
    return unique


def _load_shard(root: Path, contract: StudyContract) -> Shard:
    resolved = _load_json(root / "study.resolved.json")
    if resolved.get("schema_version") != RESOLVED_SCHEMA:
        raise ValueError(f"Unexpected resolved-study schema in {root}.")
    if resolved.get("study_id") != contract.study_id:
        raise ValueError(f"Shard study_id does not match the configured study: {root}.")
    if resolved.get("study_config_sha256") != contract.sha256:
        raise ValueError(f"Shard study-config digest mismatch: {root}.")
    if resolved.get("official_test_read") is not False:
        raise ValueError(f"Shard does not declare official_test_read=false: {root}.")
    target = resolved.get("target")
    if not isinstance(target, str) or not target:
        raise ValueError(f"Shard target is missing: {root}.")
    if resolved.get("surface_count") != len(contract.surfaces):
        raise ValueError(f"Shard surface_count mismatch: {root}.")
    recorded_surfaces = resolved.get("surfaces")
    if recorded_surfaces != list(contract.surfaces):
        raise ValueError(f"Shard surface ordering/identity mismatch: {root}.")
    code = resolved.get("code")
    if not isinstance(code, Mapping):
        raise ValueError(f"Shard code provenance is missing: {root}.")
    for key in ("commit", "source_archive_sha256"):
        if not isinstance(code.get(key), str) or not code[key]:
            raise ValueError(f"Shard code.{key} is missing: {root}.")
    return Shard(root=root, target=target, resolved=resolved)


def _derived_receipt_roots(
    *, study_root: Path | None, shards: Sequence[Shard]
) -> list[Path]:
    roots: list[Path] = []
    if study_root is not None:
        roots.append(Path(study_root).expanduser().resolve() / "transport_receipts")
    for shard in shards:
        for parent in (shard.root.parent, *shard.root.parents):
            if parent.name == "shards":
                roots.append(parent.parent / "transport_receipts")
                break
    return roots


def _receipt_inventory(
    roots: Sequence[Path], contract: StudyContract
) -> tuple[dict[tuple[str, int], list[tuple[Path, dict[str, Any]]]], list[str]]:
    inventory: dict[tuple[str, int], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    issues: list[str] = []
    seen_paths: set[Path] = set()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if root in seen_paths or not root.is_dir():
            continue
        seen_paths.add(root)
        for path in sorted(root.rglob("surface_*.json")):
            try:
                receipt = _load_json(path)
            except ValueError as error:
                issues.append(str(error))
                continue
            if receipt.get("schema_version") != RECEIPT_SCHEMA:
                continue
            if receipt.get("study_id") != contract.study_id:
                continue
            target = receipt.get("target")
            index = receipt.get("surface_index")
            if not isinstance(target, str) or isinstance(index, bool) or not isinstance(index, int):
                issues.append(f"Receipt has invalid target/index: {path}.")
                continue
            inventory[(target, index)].append((path.resolve(), receipt))
    return inventory, issues


def _path_has_suffix(value: Any, suffix: Sequence[str]) -> bool:
    if not isinstance(value, str) or not value:
        return False
    parts = Path(value).parts
    return len(parts) >= len(suffix) and tuple(parts[-len(suffix) :]) == tuple(suffix)


def _validate_selection(
    selection: Mapping[str, Any], spec: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    expected = {
        "index": spec["index"],
        "initializer": spec["initializer"],
        "architecture": spec["architecture"],
        "scheme": spec["scheme"],
        "optimizer": spec["optimizer"],
        "surface_id": spec["surface_id"],
    }
    if selection.get("schema_version") != SELECTION_SCHEMA:
        errors.append("unexpected selection schema")
    for key, value in expected.items():
        if selection.get(key) != value:
            errors.append(f"selection {key} mismatch")
    if selection.get("official_test_read") is not False:
        errors.append("selection official_test_read is not false")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list):
        errors.append("selection candidates is not a list")
    else:
        cell_ids = [row.get("cell_id") for row in candidates if isinstance(row, Mapping)]
        if len(cell_ids) != len(candidates) or any(
            not isinstance(value, str) or not value for value in cell_ids
        ):
            errors.append("selection contains an invalid candidate cell_id")
        elif len(cell_ids) != len(set(cell_ids)):
            errors.append("selection contains duplicate candidate cell_id values")
    status = selection.get("status")
    if not isinstance(status, str) or not status or status in {"pending", "running"}:
        errors.append(f"selection is not terminal: {status!r}")
    return errors


def _validate_receipt(
    *,
    path: Path,
    receipt: Mapping[str, Any],
    selection: Mapping[str, Any],
    spec: Mapping[str, Any],
    shard: Shard,
    contract: StudyContract,
) -> list[str]:
    errors: list[str] = []
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        errors.append("unexpected receipt schema")
    if receipt.get("semantic_status") != "pass":
        errors.append("receipt semantic_status is not pass")
    if receipt.get("study_id") != contract.study_id:
        errors.append("receipt study_id mismatch")
    if receipt.get("surface_index") != spec["index"]:
        errors.append("receipt surface_index mismatch")
    if receipt.get("target") != shard.target:
        errors.append("receipt target mismatch")
    if receipt.get("official_test_read") is not False:
        errors.append("receipt official_test_read is not false")
    if receipt.get("selection_status") != selection.get("status"):
        errors.append("receipt selection_status mismatch")
    surface = receipt.get("surface")
    expected_surface = {
        key: spec[key]
        for key in ("initializer", "architecture", "scheme", "optimizer", "surface_id")
    }
    if surface != expected_surface:
        errors.append("receipt surface identity mismatch")
    if not _path_has_suffix(
        receipt.get("selection_path"),
        ("surfaces", str(spec["surface_id"]), "selection.json"),
    ):
        errors.append("receipt selection_path suffix mismatch")

    code = shard.resolved["code"]
    if receipt.get("source_commit") != code.get("commit"):
        errors.append("receipt source_commit mismatch")
    if receipt.get("source_archive_sha256") != code.get("source_archive_sha256"):
        errors.append("receipt source_archive_sha256 mismatch")
    if receipt.get("study_config_sha256") != contract.sha256:
        errors.append("receipt study_config_sha256 mismatch")
    environment_id = receipt.get("environment_id")
    if not isinstance(environment_id, str) or not environment_id:
        errors.append("receipt environment_id is missing")

    init_path = (
        shard.root
        / "assets"
        / str(spec["initializer"])
        / str(spec["architecture"])
        / "final_model.pt"
    )
    if not init_path.is_file():
        errors.append(f"shared initialization checkpoint is missing: {init_path}")
    else:
        if receipt.get("shared_initialization_checkpoint_sha256") != sha256_file(init_path):
            errors.append("shared initialization checkpoint digest mismatch")
    if not _path_has_suffix(
        receipt.get("shared_initialization_checkpoint"),
        (
            "assets",
            str(spec["initializer"]),
            str(spec["architecture"]),
            "final_model.pt",
        ),
    ):
        errors.append("shared initialization checkpoint path suffix mismatch")

    selected = selection.get("selected")
    selected_run_dir = receipt.get("selected_run_dir")
    if selected is None:
        if selected_run_dir is not None:
            errors.append("receipt names a selected run for an unresolved selection")
    elif isinstance(selected, Mapping):
        cell_id = selected.get("cell_id")
        if not isinstance(cell_id, str) or not _path_has_suffix(
            selected_run_dir,
            ("surfaces", str(spec["surface_id"]), "rho", "cells", cell_id),
        ):
            errors.append("receipt selected_run_dir mismatch")
    else:
        errors.append("selection selected field is invalid")
    if path.name != f"surface_{spec['index']}.json":
        errors.append("receipt filename does not match surface index")
    return errors


def _decode_parameter_name(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("param_names contains invalid UTF-8 bytes") from error
    return str(value)


def _indexed_final_checkpoint(run_dir: Path, result: Mapping[str, Any]) -> tuple[Path, str]:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("result artifacts must be a list")
    records = [
        record
        for record in artifacts
        if isinstance(record, Mapping)
        and record.get("kind") == "checkpoint"
        and Path(str(record.get("path", ""))).name == "weights_final.npz"
    ]
    if len(records) != 1:
        raise ValueError(
            "expected exactly one indexed checkpoint named weights_final.npz, "
            f"found {len(records)}"
        )
    record = records[0]
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute():
        raise ValueError("indexed final checkpoint path must be relative")
    checkpoint = (run_dir / relative).resolve()
    try:
        checkpoint.relative_to(run_dir.resolve())
    except ValueError as error:
        raise ValueError("indexed final checkpoint escapes its run directory") from error
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("indexed final checkpoint digest is invalid")
    if not checkpoint.is_file():
        raise ValueError(f"indexed final checkpoint is missing: {checkpoint}")
    if sha256_file(checkpoint) != digest:
        raise ValueError("indexed final checkpoint digest mismatch")
    return checkpoint, digest


def _exact_endpoint_occupancy(
    checkpoint: Path, *, weight_min: float, weight_max: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lower_count = 0
    upper_count = 0
    either_count = 0
    outside_count = 0
    total = 0
    parameter_count = 0
    parameter_rows: list[dict[str, Any]] = []
    with np.load(checkpoint, allow_pickle=False) as payload:
        if "param_names" not in payload.files:
            raise ValueError(f"Checkpoint lacks param_names: {checkpoint}.")
        raw_names = np.asarray(payload["param_names"])
        if raw_names.ndim != 1:
            raise ValueError(f"param_names must be one-dimensional: {checkpoint}.")
        names = [_decode_parameter_name(value) for value in raw_names.tolist()]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate param_names in {checkpoint}.")
        bounded_names = [name for name in names if name.startswith(BOUNDED_WEIGHT_PREFIXES)]
        if not bounded_names:
            raise ValueError(f"No bounded weight arrays in {checkpoint}.")
        for parameter_index, name in enumerate(bounded_names):
            if name not in payload.files:
                raise ValueError(f"param_names references absent array {name!r}.")
            values = np.asarray(payload[name])
            if not np.issubdtype(values.dtype, np.floating):
                raise ValueError(f"{name} must have floating dtype, got {values.dtype}.")
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} is empty or contains non-finite values.")
            lower = np.asarray(weight_min, dtype=values.dtype)[()]
            upper = np.asarray(weight_max, dtype=values.dtype)[()]
            if not bool(lower < upper):
                raise ValueError(f"Configured interval collapses in {name} dtype.")
            at_lower = values == lower
            at_upper = values == upper
            parameter_lower_count = int(np.count_nonzero(at_lower))
            parameter_upper_count = int(np.count_nonzero(at_upper))
            parameter_either_count = int(
                np.count_nonzero(np.logical_or(at_lower, at_upper))
            )
            parameter_outside_count = int(
                np.count_nonzero(np.logical_or(values < lower, values > upper))
            )
            parameter_total = int(values.size)
            lower_count += parameter_lower_count
            upper_count += parameter_upper_count
            either_count += parameter_either_count
            outside_count += parameter_outside_count
            total += parameter_total
            parameter_count += 1
            parameter_rows.append(
                {
                    "parameter_index": parameter_index,
                    "parameter_name": name,
                    "parameter_kind": (
                        "conv_weight" if name.startswith("ConvWeight_") else "dense_weight"
                    ),
                    "parameter_dtype": str(values.dtype),
                    "parameter_shape_json": _stable_json(list(values.shape)),
                    "num_weights": parameter_total,
                    "exact_lower_bound_count": parameter_lower_count,
                    "exact_lower_bound_percent": (
                        100.0 * parameter_lower_count / parameter_total
                    ),
                    "exact_upper_bound_count": parameter_upper_count,
                    "exact_upper_bound_percent": (
                        100.0 * parameter_upper_count / parameter_total
                    ),
                    "exact_either_bound_count": parameter_either_count,
                    "exact_either_bound_percent": (
                        100.0 * parameter_either_count / parameter_total
                    ),
                    "outside_bound_count": parameter_outside_count,
                    "outside_bound_percent": (
                        100.0 * parameter_outside_count / parameter_total
                    ),
                }
            )
    if total <= 0:
        raise ValueError(f"No bounded weights found in {checkpoint}.")
    if outside_count:
        raise ValueError(
            f"Final checkpoint has {outside_count}/{total} bounded weights outside "
            "the configured interval."
        )
    return (
        {
            "bounded_parameter_count": parameter_count,
            "num_bounded_weights": total,
            "exact_lower_bound_count": lower_count,
            "exact_lower_bound_percent": 100.0 * lower_count / total,
            "exact_upper_bound_count": upper_count,
            "exact_upper_bound_percent": 100.0 * upper_count / total,
            "exact_either_bound_count": either_count,
            "exact_either_bound_percent": 100.0 * either_count / total,
            "outside_bound_count": outside_count,
            "outside_bound_percent": 100.0 * outside_count / total,
        },
        parameter_rows,
    )


def _candidate_learning_rates(
    cell: Mapping[str, Any], *, require: bool
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    raw_vector = cell.get("learning_rate_vector")
    raw_mapping = cell.get("learning_rates_by_parameter")
    vector: list[float] | None = None
    mapping: dict[str, float] | None = None
    if isinstance(raw_vector, list):
        try:
            vector = [
                _finite_float(value, label=f"learning_rate_vector[{index}]")
                for index, value in enumerate(raw_vector)
            ]
        except ValueError as error:
            errors.append(str(error))
    elif require:
        errors.append("candidate is missing learning_rate_vector")
    if isinstance(raw_mapping, Mapping):
        mapping = {}
        for name, value in raw_mapping.items():
            if not isinstance(name, str) or not name:
                errors.append("learning_rates_by_parameter has an invalid name")
                continue
            try:
                mapping[name] = _finite_float(value, label=f"LR for {name}")
            except ValueError as error:
                errors.append(str(error))
    elif require:
        errors.append("candidate is missing learning_rates_by_parameter")

    conv = {key: value for key, value in (mapping or {}).items() if key.startswith("ConvWeight_")}
    dense = {key: value for key, value in (mapping or {}).items() if key.startswith("DenseWeight_")}
    bias = {key: value for key, value in (mapping or {}).items() if key.startswith("Bias_")}
    all_bias_zero: bool | None = None
    if mapping is not None:
        if not bias:
            errors.append("learning_rates_by_parameter has no Bias_* entry")
            all_bias_zero = False
        else:
            all_bias_zero = all(value == 0.0 for value in bias.values())
            if not all_bias_zero:
                errors.append("candidate has a nonzero Bias_* learning rate")
        if not conv or not dense:
            errors.append("learning_rates_by_parameter lacks ConvWeight_* or DenseWeight_*")
        if vector is not None and len(vector) != len(mapping):
            errors.append("learning_rate_vector length does not match parameter mapping")
    return (
        {
            "learning_rate_vector_json": _stable_json(vector) if vector is not None else None,
            "learning_rates_by_parameter_json": (
                _stable_json(mapping) if mapping is not None else None
            ),
            "conv_learning_rates_json": _stable_json(conv) if mapping is not None else None,
            "dense_learning_rates_json": _stable_json(dense) if mapping is not None else None,
            "bias_learning_rates_json": _stable_json(bias) if mapping is not None else None,
            "all_bias_learning_rates_zero": all_bias_zero,
        },
        errors,
    )


def _bundle_evidence(
    *,
    run_dir: Path,
    cell: Mapping[str, Any],
    spec: Mapping[str, Any],
    shard: Shard,
    contract: StudyContract,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    evidence = {
        "bundle_state": None,
        "bundle_valid": None,
        "result_sha256": None,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "bounded_parameter_count": None,
        "num_bounded_weights": None,
        "exact_lower_bound_count": None,
        "exact_lower_bound_percent": None,
        "exact_upper_bound_count": None,
        "exact_upper_bound_percent": None,
        "exact_either_bound_count": None,
        "exact_either_bound_percent": None,
        "outside_bound_count": None,
        "outside_bound_percent": None,
        "canonical_final_validation_accuracy": None,
        "canonical_final_validation_loss": None,
    }
    errors: list[str] = []
    parameter_occupancy: list[dict[str, Any]] = []
    manifest_path = run_dir / "manifest.json"
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not any(path.exists() for path in (manifest_path, status_path, metrics_path)):
        if str(cell.get("status")) in COMPLETED_CELL_STATUSES:
            errors.append("completed candidate lacks a canonical run bundle")
            evidence["bundle_valid"] = False
        return evidence, errors, parameter_occupancy

    try:
        canonical_errors = validate_run(run_dir)
    except Exception as error:  # Malformed evidence is reported, never trusted.
        canonical_errors = [f"canonical bundle validation failed: {error}"]
    errors.extend(canonical_errors)
    try:
        manifest = _load_json(manifest_path)
        status = _load_json(status_path)
    except ValueError as error:
        errors.append(str(error))
        evidence["bundle_valid"] = False
        return evidence, sorted(set(errors)), parameter_occupancy

    state = status.get("state")
    evidence["bundle_state"] = state
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append("unexpected candidate manifest schema")
    if status.get("schema_version") != STATUS_SCHEMA:
        errors.append("unexpected candidate status schema")
    if manifest.get("study_id") != contract.study_id:
        errors.append("candidate manifest study_id mismatch")
    if manifest.get("evidence_class") != contract.evidence_class:
        errors.append("candidate evidence_class mismatch")
    if manifest.get("smoke") is not False:
        errors.append("candidate is marked smoke")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("official_test_read") is not False:
        errors.append("candidate manifest does not declare official_test_read=false")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("target") != shard.target:
        errors.append("candidate runtime target mismatch")
    command = manifest.get("command")
    if not isinstance(command, Mapping) or command.get("surface_index") != spec["index"]:
        errors.append("candidate command surface_index mismatch")
    if isinstance(command, Mapping) and command.get("cell_index") != cell.get("index"):
        errors.append("candidate command cell_index mismatch")
    configuration = manifest.get("configuration")
    resolved = configuration.get("resolved") if isinstance(configuration, Mapping) else None
    if not isinstance(resolved, Mapping):
        errors.append("candidate resolved configuration is missing")
    else:
        optimizer = resolved.get("optimizer")
        if not isinstance(optimizer, Mapping) or optimizer.get("name") != spec["optimizer"]:
            errors.append("candidate optimizer mismatch")
        model = resolved.get("model_base")
        if not isinstance(model, Mapping):
            errors.append("candidate model_base is missing")
        else:
            if model.get("weight_init_mode") != spec["initializer"]:
                errors.append("candidate initializer mismatch")
            if model.get("weight_min") != contract.weight_min:
                errors.append("candidate weight_min mismatch")
            if model.get("weight_max") != contract.weight_max:
                errors.append("candidate weight_max mismatch")
        bias = resolved.get("bias_contract")
        if not isinstance(bias, Mapping) or bias.get("learning_rate") != 0.0:
            errors.append("candidate resolved bias LR is not exact zero")
    code = shard.resolved["code"]
    git = manifest.get("git")
    if not isinstance(git, Mapping) or git.get("commit") != code.get("commit"):
        errors.append("candidate source commit mismatch")
    if not isinstance(git, Mapping) or git.get("source_archive_sha256") != code.get(
        "source_archive_sha256"
    ):
        errors.append("candidate source archive mismatch")

    cell_status = str(cell.get("status"))
    if cell_status in COMPLETED_CELL_STATUSES and state != "complete":
        errors.append(f"completed cell has canonical state {state!r}")
    if cell_status.startswith(TERMINAL_REJECTION_PREFIXES) and state not in {
        "complete",
        "failed",
    }:
        errors.append(f"rejected cell has canonical state {state!r}")

    if state == "complete":
        result_path = run_dir / "result.json"
        try:
            result = _load_json(result_path)
        except ValueError as error:
            errors.append(str(error))
        else:
            result_sha256 = sha256_file(result_path)
            evidence["result_sha256"] = result_sha256
            if status.get("result_sha256") != result_sha256:
                errors.append("candidate status result_sha256 mismatch")
            if result.get("schema_version") != RESULT_SCHEMA:
                errors.append("unexpected candidate result schema")
            if result.get("study_id") != contract.study_id:
                errors.append("candidate result study_id mismatch")
            if result.get("arm_id") != manifest.get("arm_id"):
                errors.append("candidate result arm_id mismatch")
            if result.get("evidence_class") != contract.evidence_class:
                errors.append("candidate result evidence_class mismatch")
            if result.get("smoke") is not False:
                errors.append("candidate result is marked smoke")
            result_dataset = result.get("dataset")
            if not isinstance(result_dataset, Mapping) or result_dataset.get(
                "official_test_read"
            ) is not False:
                errors.append("candidate result does not declare official_test_read=false")
            completion = result.get("completion")
            if not isinstance(completion, Mapping) or completion.get(
                "rho_cell_complete"
            ) is not True:
                errors.append("candidate result lacks rho_cell_complete=true")
            if not isinstance(completion, Mapping) or completion.get(
                "official_test_read"
            ) is not False:
                errors.append("candidate completion does not declare official_test_read=false")
            terminal = result.get("terminal_metrics")
            validation = terminal.get("validation") if isinstance(terminal, Mapping) else None
            if not isinstance(validation, Mapping):
                errors.append("candidate terminal validation metrics are missing")
            else:
                try:
                    evidence["canonical_final_validation_accuracy"] = _finite_accuracy(
                        validation.get("final_accuracy"),
                        label="canonical final validation accuracy",
                    )
                    evidence["canonical_final_validation_loss"] = _finite_float(
                        validation.get("final_loss"),
                        label="canonical final validation loss",
                    )
                except ValueError as error:
                    errors.append(str(error))
            try:
                checkpoint, checkpoint_sha = _indexed_final_checkpoint(run_dir, result)
                occupancy, parameter_occupancy = _exact_endpoint_occupancy(
                    checkpoint,
                    weight_min=contract.weight_min,
                    weight_max=contract.weight_max,
                )
            except (OSError, ValueError) as error:
                errors.append(str(error))
            else:
                evidence.update(occupancy)
                evidence["checkpoint_path"] = str(checkpoint)
                evidence["checkpoint_sha256"] = checkpoint_sha

    evidence["bundle_valid"] = not errors
    return evidence, sorted(set(errors)), parameter_occupancy


def _blank_row(spec: Mapping[str, Any], *, shard: Shard | None) -> dict[str, Any]:
    return {
        "row_kind": None,
        "study_id": None,
        "surface_index": spec["index"],
        "surface_id": spec["surface_id"],
        "source_shard_root": str(shard.root) if shard else None,
        "target": shard.target if shard else None,
        "initializer": spec["initializer"],
        "architecture": spec["architecture"],
        "scheme": spec["scheme"],
        "optimizer": spec["optimizer"],
        "cell_index": None,
        "cell_id": None,
        "rho_conv": None,
        "rho_dense": None,
        "status": None,
        "surface_status": None,
        "receipt_status": None,
        "selection_eligible": None,
        "meets_accuracy_gate": None,
        "runner_selected": False,
        "reportable_selected": False,
        "final_validation_accuracy": None,
        "final_validation_accuracy_percent": None,
        "final_validation_loss": None,
        "learning_rate_vector_json": None,
        "learning_rates_by_parameter_json": None,
        "conv_learning_rates_json": None,
        "dense_learning_rates_json": None,
        "bias_learning_rates_json": None,
        "all_bias_learning_rates_zero": None,
        "bundle_state": None,
        "bundle_valid": None,
        "result_sha256": None,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "bounded_parameter_count": None,
        "num_bounded_weights": None,
        "exact_lower_bound_count": None,
        "exact_lower_bound_percent": None,
        "exact_upper_bound_count": None,
        "exact_upper_bound_percent": None,
        "exact_either_bound_count": None,
        "exact_either_bound_percent": None,
        "outside_bound_count": None,
        "outside_bound_percent": None,
        "canonical_final_validation_accuracy": None,
        "canonical_final_validation_loss": None,
        "record_path": None,
        "validation_errors_json": "[]",
    }


def _candidate_accuracy(
    *,
    cell: Mapping[str, Any],
    selection_record: Mapping[str, Any] | None,
    run_dir: Path,
    bundle: Mapping[str, Any],
) -> tuple[float | None, float | None, list[str]]:
    errors: list[str] = []
    accuracy = bundle.get("canonical_final_validation_accuracy")
    loss = bundle.get("canonical_final_validation_loss")
    if accuracy is None and selection_record is not None:
        accuracy = selection_record.get("final_validation_accuracy")
        loss = selection_record.get("final_validation_loss")
    metrics_path = run_dir / "metrics.json"
    if accuracy is None and metrics_path.is_file():
        try:
            metrics = _load_json(metrics_path)
            accuracy = metrics.get("final_validation_accuracy", metrics.get("final_test_accuracy"))
            loss = metrics.get("final_validation_loss", metrics.get("final_test_loss"))
        except ValueError as error:
            errors.append(str(error))
    try:
        parsed_accuracy = (
            _finite_accuracy(accuracy, label="final validation accuracy")
            if accuracy is not None
            else None
        )
        parsed_loss = (
            _finite_float(loss, label="final validation loss")
            if loss is not None
            else None
        )
    except ValueError as error:
        errors.append(str(error))
        return None, None, errors
    if selection_record is not None and parsed_accuracy is not None:
        selected_accuracy = selection_record.get("final_validation_accuracy")
        if selected_accuracy is not None:
            try:
                recorded_accuracy = _finite_accuracy(
                    selected_accuracy, label="selection candidate accuracy"
                )
            except ValueError as error:
                errors.append(str(error))
            else:
                if not math.isclose(
                    parsed_accuracy,
                    recorded_accuracy,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    errors.append("canonical and selection candidate accuracy mismatch")
    return parsed_accuracy, parsed_loss, errors


def _candidate_row(
    *,
    cell_path: Path,
    cell: Mapping[str, Any],
    selection_record: Mapping[str, Any] | None,
    selected_cell_id: str | None,
    selection_status: str | None,
    receipt_status: str,
    spec: Mapping[str, Any],
    shard: Shard,
    contract: StudyContract,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    row = _blank_row(spec, shard=shard)
    errors: list[str] = []
    cell_id = cell_path.parent.name
    status = cell.get("status")
    row.update(
        row_kind="candidate",
        study_id=contract.study_id,
        cell_id=cell_id,
        status=status,
        surface_status=selection_status,
        receipt_status=receipt_status,
        record_path=str(cell_path),
    )
    try:
        row["cell_index"] = int(cell["index"])
        row["rho_conv"] = _finite_float(cell["rho_conv"], label="rho_conv")
        row["rho_dense"] = _finite_float(cell["rho_dense"], label="rho_dense")
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"invalid cell identity/coordinates: {error}")
    if not isinstance(status, str) or not status:
        errors.append("candidate status is missing")
    if cell.get("optimizer") != spec["optimizer"]:
        errors.append("candidate optimizer does not match surface")
    row["selection_eligible"] = (
        bool(cell["selection_eligible"])
        if isinstance(cell.get("selection_eligible"), bool)
        else None
    )
    require_lrs = isinstance(status, str) and (
        status in COMPLETED_CELL_STATUSES
        or status.startswith(TERMINAL_REJECTION_PREFIXES)
    )
    lr_fields, lr_errors = _candidate_learning_rates(cell, require=require_lrs)
    row.update(lr_fields)
    errors.extend(lr_errors)

    bundle, bundle_errors, parameter_occupancy = _bundle_evidence(
        run_dir=cell_path.parent,
        cell=cell,
        spec=spec,
        shard=shard,
        contract=contract,
    )
    row.update(bundle)
    errors.extend(bundle_errors)
    accuracy, loss, metric_errors = _candidate_accuracy(
        cell=cell,
        selection_record=selection_record,
        run_dir=cell_path.parent,
        bundle=bundle,
    )
    errors.extend(metric_errors)
    row["final_validation_accuracy"] = accuracy
    row["final_validation_accuracy_percent"] = (
        100.0 * accuracy if accuracy is not None else None
    )
    row["final_validation_loss"] = loss
    row["meets_accuracy_gate"] = (
        accuracy >= contract.accuracy_gate if accuracy is not None else None
    )
    row["runner_selected"] = cell_id == selected_cell_id
    row["reportable_selected"] = bool(
        row["runner_selected"]
        and row["meets_accuracy_gate"] is True
        and row["bundle_state"] == "complete"
        and row["bundle_valid"] is True
    )
    if row["runner_selected"] and row["meets_accuracy_gate"] is not True:
        errors.append("runner selection is below the configured accuracy gate")
    elif row["runner_selected"] and not row["reportable_selected"]:
        errors.append("runner selection lacks a valid canonical completed bundle")
    row["validation_errors_json"] = _stable_json(sorted(set(errors)))
    raw_lrs = cell.get("learning_rates_by_parameter")
    learning_rates = raw_lrs if isinstance(raw_lrs, Mapping) else {}
    enriched_parameter_occupancy = [
        {
            "study_id": contract.study_id,
            "surface_index": spec["index"],
            "surface_id": spec["surface_id"],
            "source_shard_root": str(shard.root),
            "target": shard.target,
            "initializer": spec["initializer"],
            "architecture": spec["architecture"],
            "scheme": spec["scheme"],
            "optimizer": spec["optimizer"],
            "cell_index": row["cell_index"],
            "cell_id": cell_id,
            "rho_conv": row["rho_conv"],
            "rho_dense": row["rho_dense"],
            "status": status,
            "surface_status": selection_status,
            "receipt_status": receipt_status,
            "final_validation_accuracy": row["final_validation_accuracy"],
            "final_validation_accuracy_percent": row[
                "final_validation_accuracy_percent"
            ],
            "learning_rate": learning_rates.get(parameter["parameter_name"]),
            "learning_rates_by_parameter_json": row[
                "learning_rates_by_parameter_json"
            ],
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            **parameter,
        }
        for parameter in parameter_occupancy
        if row["bundle_valid"] is True and row["validation_errors_json"] == "[]"
    ]
    return row, sorted(set(errors)), enriched_parameter_occupancy


def _selection_candidate_map(selection: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if selection is None or not isinstance(selection.get("candidates"), list):
        return {}
    return {
        str(record["cell_id"]): dict(record)
        for record in selection["candidates"]
        if isinstance(record, Mapping) and isinstance(record.get("cell_id"), str)
    }


def _canonical_numeric_candidates(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("row_kind") == "candidate"
        and row.get("bundle_state") == "complete"
        and row.get("bundle_valid") is True
        and row.get("final_validation_accuracy") is not None
        and row.get("exact_either_bound_percent") is not None
        and row.get("validation_errors_json") == "[]"
    ]


def _candidate_report_detail(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cell_index": row.get("cell_index"),
        "cell_id": row.get("cell_id"),
        "rho_conv": row.get("rho_conv"),
        "rho_dense": row.get("rho_dense"),
        "learning_rates_by_parameter_json": row.get(
            "learning_rates_by_parameter_json"
        ),
        "final_validation_accuracy": row.get("final_validation_accuracy"),
        "final_validation_accuracy_percent": row.get(
            "final_validation_accuracy_percent"
        ),
        "exact_lower_bound_percent": row.get("exact_lower_bound_percent"),
        "exact_upper_bound_percent": row.get("exact_upper_bound_percent"),
        "exact_either_bound_percent": row.get("exact_either_bound_percent"),
    }


def _surface_source(
    *,
    spec: Mapping[str, Any],
    shard: Shard,
    surface_dir: Path,
    receipts: Mapping[tuple[str, int], list[tuple[Path, dict[str, Any]]]],
    contract: StudyContract,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[str],
]:
    rows: list[dict[str, Any]] = []
    parameter_occupancy_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    selection_path = surface_dir / "selection.json"
    selection: dict[str, Any] | None = None
    selection_errors: list[str] = []
    if selection_path.is_file():
        try:
            selection = _load_json(selection_path)
            selection_errors = _validate_selection(selection, spec)
        except ValueError as error:
            selection_errors = [str(error)]
        errors.extend(f"{spec['surface_id']}: {error}" for error in selection_errors)

    receipt_records = receipts.get((shard.target, int(spec["index"])), [])
    receipt_status = "not_applicable"
    receipt_path: Path | None = None
    receipt_errors: list[str] = []
    if selection is not None:
        if len(receipt_records) == 0:
            receipt_status = "missing"
            receipt_errors.append("terminal selection has no transport receipt")
        elif len(receipt_records) > 1:
            receipt_status = "ambiguous"
            receipt_errors.append("terminal selection has multiple transport receipts")
        else:
            receipt_path, receipt = receipt_records[0]
            receipt_errors = _validate_receipt(
                path=receipt_path,
                receipt=receipt,
                selection=selection,
                spec=spec,
                shard=shard,
                contract=contract,
            )
            receipt_status = "valid" if not receipt_errors else "invalid"
    elif receipt_records:
        receipt_status = "orphaned"
        receipt_errors.append("transport receipt exists without terminal selection")
    errors.extend(f"{spec['surface_id']}: {error}" for error in receipt_errors)

    selection_status = str(selection.get("status")) if selection is not None else None
    surface_row = _blank_row(spec, shard=shard)
    surface_row.update(
        row_kind="surface",
        study_id=contract.study_id,
        status=(
            selection_status
            if selection_status is not None
            else "in_progress"
            if (surface_dir / "rho").exists()
            else "not_started"
        ),
        surface_status=selection_status,
        receipt_status=receipt_status,
        record_path=str(selection_path) if selection_path.is_file() else str(surface_dir),
        validation_errors_json=_stable_json(selection_errors + receipt_errors),
    )
    rows.append(surface_row)

    probe_path = surface_dir / "rho" / "probe.json"
    probe_status: str | None = None
    if probe_path.is_file():
        try:
            probe = _load_json(probe_path)
            probe_status = str(probe.get("status", "unknown"))
            probe_errors = []
            if probe.get("official_test_read") is not False:
                probe_errors.append("probe official_test_read is not false")
        except ValueError as error:
            probe_status = "invalid"
            probe_errors = [str(error)]
        probe_row = _blank_row(spec, shard=shard)
        probe_row.update(
            row_kind="probe",
            study_id=contract.study_id,
            status=probe_status,
            surface_status=selection_status,
            receipt_status=receipt_status,
            record_path=str(probe_path),
            validation_errors_json=_stable_json(probe_errors),
        )
        rows.append(probe_row)
        errors.extend(f"{spec['surface_id']}: {error}" for error in probe_errors)

    selected = selection.get("selected") if selection is not None else None
    selected_cell_id = (
        str(selected.get("cell_id"))
        if isinstance(selected, Mapping) and selected.get("cell_id") is not None
        else None
    )
    selected_records = _selection_candidate_map(selection)
    discovered_ids: set[str] = set()
    cells_root = surface_dir / "rho" / "cells"
    for cell_path in sorted(cells_root.glob("*/cell.json")) if cells_root.is_dir() else []:
        try:
            cell = _load_json(cell_path)
        except ValueError as error:
            errors.append(f"{spec['surface_id']}: {error}")
            continue
        cell_id = cell_path.parent.name
        discovered_ids.add(cell_id)
        row, row_errors, candidate_parameter_occupancy = _candidate_row(
            cell_path=cell_path,
            cell=cell,
            selection_record=selected_records.get(cell_id),
            selected_cell_id=selected_cell_id,
            selection_status=selection_status,
            receipt_status=receipt_status,
            spec=spec,
            shard=shard,
            contract=contract,
        )
        rows.append(row)
        parameter_occupancy_rows.extend(candidate_parameter_occupancy)
        errors.extend(f"{spec['surface_id']}/{cell_id}: {error}" for error in row_errors)

    missing_cell_ids = sorted(set(selected_records) - discovered_ids)
    for cell_id in missing_cell_ids:
        record = selected_records[cell_id]
        placeholder = _blank_row(spec, shard=shard)
        placeholder.update(
            row_kind="candidate",
            study_id=contract.study_id,
            cell_id=cell_id,
            cell_index=record.get("index"),
            rho_conv=record.get("rho_conv"),
            rho_dense=record.get("rho_dense"),
            status=record.get("status"),
            surface_status=selection_status,
            receipt_status=receipt_status,
            selection_eligible=record.get("selection_eligible"),
            final_validation_accuracy=record.get("final_validation_accuracy"),
            final_validation_accuracy_percent=(
                100.0 * float(record["final_validation_accuracy"])
                if record.get("final_validation_accuracy") is not None
                else None
            ),
            final_validation_loss=record.get("final_validation_loss"),
            runner_selected=cell_id == selected_cell_id,
            reportable_selected=False,
            record_path=str(selection_path),
            validation_errors_json=_stable_json(["selection candidate cell.json is missing"]),
        )
        rows.append(placeholder)
        errors.append(f"{spec['surface_id']}/{cell_id}: selection candidate cell.json is missing")

    if selection is not None:
        unlisted = sorted(discovered_ids - set(selected_records))
        if unlisted:
            errors.append(
                f"{spec['surface_id']}: terminal selection omits cells {unlisted!r}"
            )

    candidates = [row for row in rows if row["row_kind"] == "candidate"]
    completed = [row for row in candidates if row["bundle_state"] == "complete"]
    rejected = [
        row
        for row in candidates
        if isinstance(row["status"], str)
        and row["status"].startswith(TERMINAL_REJECTION_PREFIXES)
    ]
    canonical_numeric = _canonical_numeric_candidates(candidates)
    highest_observed = (
        sorted(
            canonical_numeric,
            key=lambda row: (
                -float(row["final_validation_accuracy"]),
                (
                    int(row["cell_index"])
                    if isinstance(row.get("cell_index"), int)
                    else sys.maxsize
                ),
                str(row["cell_id"]),
            ),
        )[0]
        if canonical_numeric
        else None
    )
    gate_qualified = [
        row for row in candidates if row["meets_accuracy_gate"] is True
    ]
    reportable_selected = [row for row in candidates if row["reportable_selected"]]
    summary = {
        **dict(spec),
        "target": shard.target,
        "source_shard_root": str(shard.root),
        "discovery_status": "terminal" if selection is not None else "in_progress",
        "terminal_status": selection_status,
        "selection_path": str(selection_path) if selection_path.is_file() else None,
        "selection_valid": selection is not None and not selection_errors,
        "receipt_status": receipt_status,
        "receipt_path": str(receipt_path) if receipt_path else None,
        "candidate_count": len(candidates),
        "canonical_complete_candidate_count": len(completed),
        "rejected_candidate_count": len(rejected),
        "gate_qualified_candidate_count": len(gate_qualified),
        "highest_observed_accuracy": (
            highest_observed["final_validation_accuracy"]
            if highest_observed is not None
            else None
        ),
        "highest_observed_row": (
            _candidate_report_detail(highest_observed)
            if highest_observed is not None
            else None
        ),
        "runner_selected_cell_id": selected_cell_id,
        "reportable_selected_cell_id": (
            reportable_selected[0]["cell_id"] if len(reportable_selected) == 1 else None
        ),
        "reportable_selected_row": (
            _candidate_report_detail(reportable_selected[0])
            if len(reportable_selected) == 1
            else None
        ),
        "winner_inferred": False,
        "probe_status": probe_status,
        "validation_errors": sorted(set(selection_errors + receipt_errors)),
    }
    return rows, parameter_occupancy_rows, summary, sorted(set(errors))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def _pearson(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    if x.size < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def _accuracy_occupancy_correlations(
    *,
    rows: Sequence[Mapping[str, Any]],
    surfaces: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    correlations: list[dict[str, Any]] = []
    for surface in surfaces:
        if not (
            surface.get("discovery_status") == "terminal"
            and surface.get("selection_valid") is True
        ):
            continue
        candidates = [
            row
            for row in _canonical_numeric_candidates(rows)
            if row.get("surface_id") == surface.get("surface_id")
        ]
        candidates.sort(
            key=lambda row: (
                int(row["cell_index"])
                if isinstance(row.get("cell_index"), int)
                else sys.maxsize,
                str(row.get("cell_id")),
            )
        )
        occupancy = [float(row["exact_either_bound_percent"]) for row in candidates]
        accuracy = [float(row["final_validation_accuracy_percent"]) for row in candidates]
        if len(candidates) < 3:
            status = "insufficient_candidates"
            pearson_r = None
            spearman_r = None
        else:
            pearson_r = _pearson(occupancy, accuracy)
            spearman_r = _pearson(
                _average_ranks(occupancy),
                _average_ranks(accuracy),
            )
            status = (
                "computed"
                if pearson_r is not None and spearman_r is not None
                else "undefined_constant_values"
            )
        correlations.append(
            {
                "surface_index": surface["index"],
                "surface_id": surface["surface_id"],
                "target": surface.get("target"),
                "initializer": surface["initializer"],
                "architecture": surface["architecture"],
                "scheme": surface["scheme"],
                "optimizer": surface["optimizer"],
                "terminal_status": surface.get("terminal_status"),
                "receipt_status": surface.get("receipt_status"),
                "canonical_numeric_candidate_count": len(candidates),
                "minimum_candidate_count": 3,
                "pearson_r_accuracy_vs_exact_either_bound_percent": pearson_r,
                "spearman_r_accuracy_vs_exact_either_bound_percent": spearman_r,
                "status": status,
                "candidate_cell_ids_json": _stable_json(
                    [row["cell_id"] for row in candidates]
                ),
                "descriptive_non_causal": True,
                "interpretation": (
                    "Descriptive within-surface association only; no causal or "
                    "learning-rate-adequacy inference."
                ),
            }
        )
    return correlations


def _plot_accuracy(rows: Sequence[Mapping[str, Any]], path: Path, gate: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    architectures = ("conv1", "conv2", "conv3")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True)
    candidate_rows = [
        row
        for row in rows
        if row["row_kind"] == "candidate"
        and row["final_validation_accuracy_percent"] is not None
    ]
    categories = [
        (scheme, optimizer)
        for scheme in ("baseline", "ours", "legacy")
        for optimizer in ("SGD", "Adam")
    ]
    category_index = {key: index for index, key in enumerate(categories)}
    for axis, architecture in zip(axes, architectures, strict=True):
        selected = [row for row in candidate_rows if row["architecture"] == architecture]
        for point_index, row in enumerate(selected):
            base = category_index[(str(row["scheme"]), str(row["optimizer"]))]
            # A deterministic small spread exposes multiple rho candidates without
            # suggesting an ordering or winner.
            jitter = ((point_index % 9) - 4) * 0.035
            axis.scatter(
                base + jitter,
                float(row["final_validation_accuracy_percent"]),
                color=SCHEME_COLORS[str(row["scheme"])],
                marker=OPTIMIZER_MARKERS[str(row["optimizer"])],
                s=26,
                alpha=0.65,
            )
        axis.axhline(100.0 * gate, color="#d62728", linestyle="--", linewidth=1.2)
        axis.set_title(architecture.upper())
        axis.set_xticks(range(len(categories)))
        axis.set_xticklabels(
            [f"{scheme}\n{optimizer}" for scheme, optimizer in categories],
            rotation=35,
            ha="right",
        )
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Final validation accuracy (%)")
    fig.suptitle(
        "Observed ordinary-MNIST candidates (90% gate; no inferred winner)",
        fontsize=12,
    )
    legend = [
        Line2D([0], [0], color=color, marker="o", linestyle="", label=scheme)
        for scheme, color in SCHEME_COLORS.items()
    ] + [
        Line2D([0], [0], color="black", marker=marker, linestyle="", label=optimizer)
        for optimizer, marker in OPTIMIZER_MARKERS.items()
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_occupancy(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    architectures = ("conv1", "conv2", "conv3")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True)
    occupied = [
        row
        for row in rows
        if row["row_kind"] == "candidate"
        and row["bundle_valid"] is True
        and row["exact_either_bound_percent"] is not None
    ]
    categories = [
        (scheme, optimizer)
        for scheme in ("baseline", "ours", "legacy")
        for optimizer in ("SGD", "Adam")
    ]
    category_index = {key: index for index, key in enumerate(categories)}
    endpoint_styles = (
        ("exact_lower_bound_percent", "v", "lower"),
        ("exact_upper_bound_percent", "^", "upper"),
        ("exact_either_bound_percent", "x", "either"),
    )
    for axis, architecture in zip(axes, architectures, strict=True):
        selected = [row for row in occupied if row["architecture"] == architecture]
        for point_index, row in enumerate(selected):
            base = category_index[(str(row["scheme"]), str(row["optimizer"]))]
            jitter = ((point_index % 9) - 4) * 0.035
            for field, marker, _ in endpoint_styles:
                axis.scatter(
                    base + jitter,
                    float(row[field]),
                    color=SCHEME_COLORS[str(row["scheme"])],
                    marker=marker,
                    s=27,
                    alpha=0.65,
                )
        axis.set_title(architecture.upper())
        axis.set_xticks(range(len(categories)))
        axis.set_xticklabels(
            [f"{scheme}\n{optimizer}" for scheme, optimizer in categories],
            rotation=35,
            ha="right",
        )
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Exact final endpoint occupancy (%)")
    fig.suptitle("Exact lower / upper / either-bound occupancy", fontsize=12)
    legend = [
        Line2D([0], [0], color="black", marker=marker, linestyle="", label=label)
        for _, marker, label in endpoint_styles
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_accuracy_vs_occupancy(
    rows: Sequence[Mapping[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
    path: Path,
    gate: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    architectures = ("conv1", "conv2", "conv3")
    categories = [
        (scheme, optimizer)
        for scheme in ("baseline", "ours", "legacy")
        for optimizer in ("SGD", "Adam")
    ]
    correlation_by_surface = {
        str(row["surface_id"]): row for row in correlations
    }
    candidate_rows = [
        row
        for row in _canonical_numeric_candidates(rows)
        if str(row.get("surface_id")) in correlation_by_surface
    ]
    fig, axes = plt.subplots(
        len(architectures),
        len(categories),
        figsize=(18.0, 9.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, architecture in enumerate(architectures):
        for column_index, (scheme, optimizer) in enumerate(categories):
            axis = axes[row_index, column_index]
            selected = [
                row
                for row in candidate_rows
                if row.get("architecture") == architecture
                and row.get("scheme") == scheme
                and row.get("optimizer") == optimizer
            ]
            axis.scatter(
                [float(row["exact_either_bound_percent"]) for row in selected],
                [float(row["final_validation_accuracy_percent"]) for row in selected],
                color=SCHEME_COLORS[scheme],
                marker=OPTIMIZER_MARKERS[optimizer],
                s=27,
                alpha=0.72,
            )
            axis.axhline(
                100.0 * gate,
                color="#d62728",
                linestyle="--",
                linewidth=0.9,
                alpha=0.75,
            )
            surface_id = (
                f"bounded_uniform__{architecture}__{scheme}__{optimizer.lower()}"
            )
            correlation = correlation_by_surface.get(surface_id)
            if correlation is not None and correlation.get("status") == "computed":
                annotation = (
                    f"n={len(selected)}\n"
                    f"r={float(correlation['pearson_r_accuracy_vs_exact_either_bound_percent']):.2f}, "
                    f"rho={float(correlation['spearman_r_accuracy_vs_exact_either_bound_percent']):.2f}"
                )
            else:
                annotation = f"n={len(selected)}"
            axis.text(
                0.03,
                0.05,
                annotation,
                transform=axis.transAxes,
                fontsize=7.5,
                va="bottom",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
            )
            axis.grid(True, alpha=0.2)
            axis.set_xlim(-2.0, 102.0)
            axis.set_ylim(-2.0, 102.0)
            if row_index == 0:
                axis.set_title(f"{scheme}\n{optimizer}", fontsize=10)
            if column_index == 0:
                axis.set_ylabel(f"{architecture.upper()}\nAccuracy (%)")
            if row_index == len(architectures) - 1:
                axis.set_xlabel("Either-bound occupancy (%)")
    fig.suptitle(
        "Accuracy vs exact either-bound occupancy by terminal surface\n"
        "Descriptive within-surface associations only; no causal inference",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _format_percent(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def _format_occupancy_percent(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2f}%"


def _format_rho(value: Any) -> str:
    return "—" if value is None else f"{float(value):.8g}"


def _format_correlation(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _report_markdown(report: Mapping[str, Any]) -> str:
    coverage = report["coverage"]
    lines = [
        "# Bounded-uniform zero-bias LR-search snapshot",
        "",
        (
            f"Status: **{report['snapshot_status']}**; terminal surfaces "
            f"{coverage['terminal_surface_count']}/{coverage['expected_surface_count']}, "
            f"valid receipts {coverage['valid_receipt_count']}."
        ),
        "",
        (
            "This is ordinary-MNIST selection evidence, not paper accuracy. "
            f"The handoff gate is {100.0 * report['accuracy_gate']:.1f}%. "
            "Observed maxima below that gate are retained but never labeled winners."
        ),
        "",
        "## Surface snapshot",
        "",
        "| Surface | Target | State | Receipt | Candidates | Complete | Rejected | "
        "Max accuracy | >= gate | Reportable selection |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for surface in report["surfaces"]:
        lines.append(
            "| {surface_id} | {target} | {state} | {receipt} | {candidates} | "
            "{complete} | {rejected} | {accuracy} | {qualified} | {selected} |".format(
                surface_id=surface["surface_id"],
                target=surface.get("target") or "—",
                state=surface.get("terminal_status") or surface["discovery_status"],
                receipt=surface.get("receipt_status") or "—",
                candidates=surface.get("candidate_count", 0),
                complete=surface.get("canonical_complete_candidate_count", 0),
                rejected=surface.get("rejected_candidate_count", 0),
                accuracy=_format_percent(surface.get("highest_observed_accuracy")),
                qualified=surface.get("gate_qualified_candidate_count", 0),
                selected=surface.get("reportable_selected_cell_id") or "—",
            )
        )
    lines.extend(
        [
            "",
            "## Observed rows and reportable runner selections",
            "",
            (
                "Highest rows are descriptive observations from valid canonical bundles. "
                "They are not inferred winners. A runner-selected row is shown only when "
                "it meets the configured accuracy gate and has a valid canonical bundle."
            ),
            "",
            "| Surface | Role | Cell | rho conv | rho dense | Actual per-parameter LR JSON | "
            "Accuracy | Lower bound | Upper bound | Either bound |",
            "|---|---|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for surface in report["surfaces"]:
        highest = surface.get("highest_observed_row")
        detail_rows: list[tuple[str, Mapping[str, Any] | None]] = []
        if isinstance(highest, Mapping):
            role = (
                "highest observed (below gate; observation, not winner)"
                if float(highest["final_validation_accuracy"])
                < float(report["accuracy_gate"])
                else "highest observed (meets gate; observation, not inferred winner)"
            )
            detail_rows.append((role, highest))
        else:
            detail_rows.append(("highest observed (unavailable)", None))
        selected = surface.get("reportable_selected_row")
        if isinstance(selected, Mapping):
            detail_rows.append(("runner-selected (meets gate)", selected))
        for role, detail in detail_rows:
            detail = detail or {}
            lr_json = detail.get("learning_rates_by_parameter_json")
            lines.append(
                "| {surface_id} | {role} | {cell} | {rho_conv} | {rho_dense} | "
                "{lrs} | {accuracy} | {lower} | {upper} | {either} |".format(
                    surface_id=surface["surface_id"],
                    role=role,
                    cell=detail.get("cell_id") or "—",
                    rho_conv=_format_rho(detail.get("rho_conv")),
                    rho_dense=_format_rho(detail.get("rho_dense")),
                    lrs=f"`{lr_json}`" if lr_json is not None else "—",
                    accuracy=_format_percent(detail.get("final_validation_accuracy")),
                    lower=_format_occupancy_percent(
                        detail.get("exact_lower_bound_percent")
                    ),
                    upper=_format_occupancy_percent(
                        detail.get("exact_upper_bound_percent")
                    ),
                    either=_format_occupancy_percent(
                        detail.get("exact_either_bound_percent")
                    ),
                )
            )
    lines.extend(
        [
            "",
            "## Endpoint occupancy",
            "",
            (
                f"Exact final occupancy was computed for "
                f"{coverage['occupancy_candidate_count']} canonical completed candidates "
                "from the SHA-256-indexed `weights_final.npz`. Bias arrays are excluded."
            ),
            "",
            (
                f"The per-parameter artifact contains "
                f"{coverage['parameter_occupancy_row_count']} Conv/Dense parameter rows "
                "with separate exact lower, upper, and either-bound counts and percentages."
            ),
            "",
            "The candidate and per-parameter evidence preserves the actual LR vector and "
            "per-parameter mapping; every recorded `Bias_*` LR is checked for exact zero.",
            "",
            "## Accuracy–occupancy correlations",
            "",
            (
                "Pearson and Spearman coefficients are computed only within a terminal "
                "surface with at least three valid canonical numeric candidates. These "
                "are descriptive associations, not causal evidence and not a test of "
                "learning-rate adequacy."
            ),
            "",
            "| Surface | n | Status | Pearson r | Spearman rho |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for correlation in report["accuracy_occupancy_correlations"]:
        lines.append(
            "| {surface_id} | {count} | {status} | {pearson} | {spearman} |".format(
                surface_id=correlation["surface_id"],
                count=correlation["canonical_numeric_candidate_count"],
                status=correlation["status"],
                pearson=_format_correlation(
                    correlation[
                        "pearson_r_accuracy_vs_exact_either_bound_percent"
                    ]
                ),
                spearman=_format_correlation(
                    correlation[
                        "spearman_r_accuracy_vs_exact_either_bound_percent"
                    ]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Limitations and validation",
            "",
        ]
    )
    if report["issues"]:
        lines.extend(f"- {issue}" for issue in report["issues"])
    else:
        lines.append("- No evidence-integrity errors were found in the discovered snapshot.")
    if not coverage["complete"]:
        lines.append(
            "- Coverage is partial; absent or in-progress surfaces are not treated as failed "
            "or complete."
        )
    if report["unresolved_surface_ids"]:
        lines.append(
            "- Terminal but unresolved surfaces: "
            + ", ".join(f"`{value}`" for value in report["unresolved_surface_ids"])
            + "."
        )
    lines.append("")
    return "\n".join(lines)


def analyze(
    *,
    study_config_path: Path = DEFAULT_STUDY_CONFIG,
    study_root: Path | None = None,
    shard_roots: Sequence[Path] = (),
    receipt_roots: Sequence[Path] = (),
    output_dir: Path,
    require_complete: bool = False,
) -> dict[str, Any]:
    contract = _study_contract(study_config_path)
    resolved_shard_roots = discover_shard_roots(
        study_root=study_root,
        shard_roots=shard_roots,
    )
    shards = [_load_shard(root, contract) for root in resolved_shard_roots]

    all_receipt_roots = [Path(path).expanduser().resolve() for path in receipt_roots]
    all_receipt_roots.extend(_derived_receipt_roots(study_root=study_root, shards=shards))
    receipt_inventory, issues = _receipt_inventory(all_receipt_roots, contract)

    locations: dict[str, list[tuple[Shard, Path]]] = defaultdict(list)
    for shard in shards:
        for spec in contract.surfaces:
            surface_dir = shard.root / "surfaces" / str(spec["surface_id"])
            if surface_dir.is_dir():
                locations[str(spec["surface_id"])].append((shard, surface_dir))
    selected_targets = {shard.target for shard in shards}
    located_receipt_keys = {
        (shard.target, int(spec["index"]))
        for spec in contract.surfaces
        for shard, _ in locations.get(str(spec["surface_id"]), [])
    }
    for (target, index), records in sorted(receipt_inventory.items()):
        if target in selected_targets and (target, index) not in located_receipt_keys:
            issues.append(
                f"Orphaned receipt for target={target!r}, surface_index={index}: "
                + ", ".join(str(path) for path, _ in records)
            )

    rows: list[dict[str, Any]] = []
    parameter_occupancy_rows: list[dict[str, Any]] = []
    surface_summaries: list[dict[str, Any]] = []
    for spec in contract.surfaces:
        surface_id = str(spec["surface_id"])
        candidates = locations.get(surface_id, [])
        if not candidates:
            missing = _blank_row(spec, shard=None)
            missing.update(
                row_kind="surface",
                study_id=contract.study_id,
                status="not_discovered",
                receipt_status="not_applicable",
                record_path=None,
            )
            rows.append(missing)
            surface_summaries.append(
                {
                    **dict(spec),
                    "target": None,
                    "source_shard_root": None,
                    "discovery_status": "missing",
                    "terminal_status": None,
                    "selection_path": None,
                    "selection_valid": False,
                    "receipt_status": "not_applicable",
                    "receipt_path": None,
                    "candidate_count": 0,
                    "canonical_complete_candidate_count": 0,
                    "rejected_candidate_count": 0,
                    "gate_qualified_candidate_count": 0,
                    "highest_observed_accuracy": None,
                    "highest_observed_row": None,
                    "runner_selected_cell_id": None,
                    "reportable_selected_cell_id": None,
                    "reportable_selected_row": None,
                    "winner_inferred": False,
                    "probe_status": None,
                    "validation_errors": [],
                }
            )
            continue
        if len(candidates) > 1:
            issues.append(
                f"Surface {surface_id} occurs in multiple authoritative shards: "
                + ", ".join(str(shard.root) for shard, _ in candidates)
            )
        source_summaries: list[dict[str, Any]] = []
        for shard, surface_dir in candidates:
            (
                source_rows,
                source_parameter_occupancy,
                summary,
                source_issues,
            ) = _surface_source(
                spec=spec,
                shard=shard,
                surface_dir=surface_dir,
                receipts=receipt_inventory,
                contract=contract,
            )
            rows.extend(source_rows)
            parameter_occupancy_rows.extend(source_parameter_occupancy)
            source_summaries.append(summary)
            issues.extend(source_issues)
        if len(source_summaries) == 1:
            surface_summaries.append(source_summaries[0])
        else:
            surface_summaries.append(
                {
                    **dict(spec),
                    "target": None,
                    "source_shard_root": None,
                    "discovery_status": "ambiguous",
                    "terminal_status": None,
                    "selection_path": None,
                    "selection_valid": False,
                    "receipt_status": "ambiguous",
                    "receipt_path": None,
                    "candidate_count": sum(row["candidate_count"] for row in source_summaries),
                    "canonical_complete_candidate_count": sum(
                        row["canonical_complete_candidate_count"] for row in source_summaries
                    ),
                    "rejected_candidate_count": sum(
                        row["rejected_candidate_count"] for row in source_summaries
                    ),
                    "gate_qualified_candidate_count": sum(
                        row["gate_qualified_candidate_count"] for row in source_summaries
                    ),
                    "highest_observed_accuracy": None,
                    "highest_observed_row": None,
                    "runner_selected_cell_id": None,
                    "reportable_selected_cell_id": None,
                    "reportable_selected_row": None,
                    "winner_inferred": False,
                    "probe_status": None,
                    "validation_errors": ["duplicate authoritative surface"],
                }
            )

    candidate_rows = [row for row in rows if row["row_kind"] == "candidate"]
    invalid_completed = [
        row
        for row in candidate_rows
        if row["status"] in COMPLETED_CELL_STATUSES and row["bundle_valid"] is not True
    ]
    occupancy_rows = [
        row
        for row in candidate_rows
        if row["bundle_valid"] is True and row["exact_either_bound_percent"] is not None
    ]
    terminal_surfaces = [
        surface
        for surface in surface_summaries
        if surface["discovery_status"] == "terminal" and surface["selection_valid"]
    ]
    valid_receipts = [
        surface for surface in terminal_surfaces if surface["receipt_status"] == "valid"
    ]
    duplicate_surface_count = sum(
        surface["discovery_status"] == "ambiguous" for surface in surface_summaries
    )
    complete_coverage = bool(
        len(terminal_surfaces) == len(contract.surfaces)
        and len(valid_receipts) == len(contract.surfaces)
        and not invalid_completed
        and not issues
        and duplicate_surface_count == 0
    )
    unresolved_surface_ids = [
        str(surface["surface_id"])
        for surface in surface_summaries
        if isinstance(surface.get("terminal_status"), str)
        and str(surface["terminal_status"]).startswith("unresolved")
    ]
    reportable_handoffs = [
        surface
        for surface in surface_summaries
        if surface.get("reportable_selected_cell_id") is not None
    ]
    scientific_handoff_complete = bool(
        complete_coverage and len(reportable_handoffs) == len(contract.surfaces)
    )
    correlations = _accuracy_occupancy_correlations(
        rows=rows,
        surfaces=surface_summaries,
    )
    status_counts = Counter(str(row["status"]) for row in candidate_rows)
    coverage = {
        "complete": complete_coverage,
        "snapshot_is_partial": not complete_coverage,
        "expected_surface_count": len(contract.surfaces),
        "discovered_surface_count": sum(
            surface["discovery_status"] != "missing" for surface in surface_summaries
        ),
        "terminal_surface_count": len(terminal_surfaces),
        "valid_receipt_count": len(valid_receipts),
        "duplicate_surface_count": duplicate_surface_count,
        "candidate_count": len(candidate_rows),
        "candidate_status_counts": dict(sorted(status_counts.items())),
        "canonical_complete_candidate_count": sum(
            row["bundle_state"] == "complete" for row in candidate_rows
        ),
        "valid_completed_bundle_count": sum(
            row["bundle_state"] == "complete" and row["bundle_valid"] is True
            for row in candidate_rows
        ),
        "invalid_completed_bundle_count": len(invalid_completed),
        "occupancy_candidate_count": len(occupancy_rows),
        "parameter_occupancy_row_count": len(parameter_occupancy_rows),
        "reportable_correlation_count": sum(
            row["status"] == "computed" for row in correlations
        ),
        "missing_surface_ids": [
            str(surface["surface_id"])
            for surface in surface_summaries
            if surface["discovery_status"] == "missing"
        ],
        "in_progress_surface_ids": [
            str(surface["surface_id"])
            for surface in surface_summaries
            if surface["discovery_status"] == "in_progress"
        ],
    }
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "candidate_csv": "candidate_rows.csv",
        "surface_csv": "surface_summary.csv",
        "parameter_occupancy_csv": "parameter_endpoint_occupancy.csv",
        "parameter_occupancy_json": "parameter_endpoint_occupancy.json",
        "correlation_csv": "accuracy_occupancy_correlations.csv",
        "summary_json": "summary.json",
        "report_markdown": "report.md",
        "accuracy_plot": "plots/validation_accuracy.png",
        "occupancy_plot": "plots/final_bound_occupancy.png",
        "accuracy_occupancy_plot": (
            "plots/accuracy_vs_either_bound_occupancy.png"
        ),
    }
    report: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA,
        "study_id": contract.study_id,
        "evidence_class": contract.evidence_class,
        "official_test_read": False,
        "study_config": str(contract.path),
        "study_config_sha256": contract.sha256,
        "authoritative_shard_roots": [str(shard.root) for shard in shards],
        "accuracy_gate": contract.accuracy_gate,
        "weight_min": contract.weight_min,
        "weight_max": contract.weight_max,
        "snapshot_status": "complete_coverage" if complete_coverage else "partial",
        "scientific_handoff_complete": scientific_handoff_complete,
        "winner_inference_policy": "none_below_accuracy_gate",
        "winner_inferred": False,
        "coverage": coverage,
        "unresolved_surface_ids": unresolved_surface_ids,
        "surfaces": surface_summaries,
        "candidate_rows": rows,
        "parameter_occupancy_rows": parameter_occupancy_rows,
        "accuracy_occupancy_correlations": correlations,
        "issues": sorted(set(issues)),
        "artifacts": artifacts,
    }
    row_fields = list(_blank_row(contract.surfaces[0], shard=None))
    _write_csv(output_dir / artifacts["candidate_csv"], rows, row_fields)
    _write_csv(
        output_dir / artifacts["parameter_occupancy_csv"],
        parameter_occupancy_rows,
        PARAMETER_OCCUPANCY_FIELDS,
    )
    atomic_write_json(
        output_dir / artifacts["parameter_occupancy_json"],
        {
            "schema_version": (
                "perfectdiode-bounded-uniform-zero-bias-parameter-occupancy/v1"
            ),
            "study_id": contract.study_id,
            "weight_min": contract.weight_min,
            "weight_max": contract.weight_max,
            "bias_arrays_excluded": True,
            "rows": parameter_occupancy_rows,
        },
    )
    _write_csv(
        output_dir / artifacts["correlation_csv"],
        correlations,
        CORRELATION_FIELDS,
    )
    surface_fields = list(surface_summaries[0])
    surface_csv_rows = []
    for surface in surface_summaries:
        serialized = dict(surface)
        for field in (
            "highest_observed_row",
            "reportable_selected_row",
            "validation_errors",
        ):
            serialized[field] = _stable_json(surface[field])
        surface_csv_rows.append(serialized)
    _write_csv(output_dir / artifacts["surface_csv"], surface_csv_rows, surface_fields)
    _plot_accuracy(rows, output_dir / artifacts["accuracy_plot"], contract.accuracy_gate)
    _plot_occupancy(rows, output_dir / artifacts["occupancy_plot"])
    _plot_accuracy_vs_occupancy(
        rows,
        correlations,
        output_dir / artifacts["accuracy_occupancy_plot"],
        contract.accuracy_gate,
    )
    (output_dir / artifacts["report_markdown"]).write_text(
        _report_markdown(report), encoding="utf-8"
    )
    atomic_write_json(output_dir / artifacts["summary_json"], report)
    if require_complete and not complete_coverage:
        raise IncompleteCoverageError(report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STUDY_CONFIG)
    parser.add_argument(
        "--study-root",
        type=Path,
        help="Discover shard and receipt roots below one authoritative local study root.",
    )
    parser.add_argument(
        "--shard-root",
        type=Path,
        action="append",
        default=[],
        help="Authoritative local shard root containing study.resolved.json (repeatable).",
    )
    parser.add_argument(
        "--receipt-root",
        type=Path,
        action="append",
        default=[],
        help="Additional local transport-receipt search root (repeatable).",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Write the snapshot, then fail unless all expected surfaces are terminal and valid.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir is None:
        if args.study_root is None:
            raise SystemExit("--output-dir is required when --study-root is omitted")
        output_dir = Path(args.study_root) / "analysis" / "lr_search_snapshot"
    else:
        output_dir = args.output_dir
    try:
        report = analyze(
            study_config_path=args.study_config,
            study_root=args.study_root,
            shard_roots=args.shard_root,
            receipt_roots=args.receipt_root,
            output_dir=output_dir,
            require_complete=args.require_complete,
        )
    except IncompleteCoverageError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "snapshot_status": report["snapshot_status"],
                "coverage": report["coverage"],
                "output_dir": str(Path(output_dir).expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
