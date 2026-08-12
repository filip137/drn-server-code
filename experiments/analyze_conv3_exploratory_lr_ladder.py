#!/usr/bin/env python3
"""Analyze the Conv3 exploratory low-to-high learning-rate ladder.

This analyzer is intentionally fail-closed.  It accepts one collected shard
for each of baseline/ours x SGD/Adam, validates all 64 declared cells per
surface, and writes a descriptive snapshot.  Shards may use the original Jean
Zay receipt layout or the local Akib/Trex receipt written inside each shard.
The snapshot is explicitly exploratory and noncanonical: it is neither an LR
handoff nor paper evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from experiments.reporting import atomic_write_json, sha256_file, validate_run
    from experiments.rho_search import prepare_training_config
    from experiments.run_conv3_exploratory_lr_ladder import (
        DEFAULT_STUDY,
        EXPECTED_INITIALIZER_SHA256,
        RESOLVED_SCHEMA,
        STUDY_SCHEMA,
        SURFACE_SUMMARY_SCHEMA,
    )
    from experiments.run_conv12_bounded_rho import build_source_config
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from reporting import atomic_write_json, sha256_file, validate_run  # type: ignore[no-redef]
    from rho_search import prepare_training_config  # type: ignore[no-redef]
    from run_conv3_exploratory_lr_ladder import (  # type: ignore[no-redef]
        DEFAULT_STUDY,
        EXPECTED_INITIALIZER_SHA256,
        RESOLVED_SCHEMA,
        STUDY_SCHEMA,
        SURFACE_SUMMARY_SCHEMA,
    )
    from run_conv12_bounded_rho import build_source_config  # type: ignore[no-redef]


ANALYSIS_SCHEMA = "perfectdiode-conv3-exploratory-lr-ladder-analysis/v1"
CANONICAL_STUDY_ID = (
    "perfectdiode-conv3-exploratory-lr-ladder-seed0-20260811-v1"
)
RECEIPT_SCHEMA = "perfectdiode-conv3-exploratory-lr-ladder-transport/v1"
LOCAL_RECEIPT_SCHEMA = (
    "perfectdiode-conv3-exploratory-lr-ladder-local-transport-receipt/v1"
)
ACCEPTED_TARGETS = frozenset({"jean-zay", "akib", "trex"})
EXPLORATORY_LABEL = (
    "EXPLORATORY / NONCANONICAL ordinary-MNIST diagnostic: canaries, "
    "scientific safety rejections, and post-training T/K checks were disabled. "
    "These observations are not an LR handoff and are not paper evidence."
)
CORRELATION_CAVEAT = (
    "Descriptive and noncausal: rho_conv and rho_dense jointly vary across "
    "the grid, so these within-surface correlations do not isolate an effect "
    "of clipping or projection efficiency on accuracy."
)
EXPECTED_CELL_COUNT = 64
WEIGHT_NAMES = (
    "ConvWeight_0",
    "ConvWeight_1",
    "ConvWeight_2",
    "DenseWeight_0",
)
BIAS_NAMES = ("Bias_0", "Bias_1", "Bias_2")
NUMERIC_STATUS = "complete"
NONFINITE_STATUSES = {
    "candidate_rejected_nonfinite": "nonfinite_training_value",
    "candidate_rejected_safety": "nonfinite_diagnostic",
}
AXIAL_DIRECTIONS = (
    ("lower_rho_conv", "rho_conv", -1, 0),
    ("upper_rho_conv", "rho_conv", 1, 0),
    ("lower_rho_dense", "rho_dense", 0, -1),
    ("upper_rho_dense", "rho_dense", 0, 1),
)
SURFACE_FIELDS = (
    "surface_index",
    "surface_id",
    "scheme",
    "optimizer",
    "target",
    "execution_environment_json",
    "baseline_vs_ours_within_target",
    "baseline_vs_ours_same_host_runtime_contract",
    "baseline_vs_ours_host_runtime_confounded",
    "baseline_vs_ours_execution_environments_json",
    "cross_optimizer_host_runtime_confounded",
    "cross_optimizer_causal_interpretation_allowed",
    "cross_optimizer_comparison_statement",
    "numeric_cell_count",
    "nonfinite_cell_count",
    "best_observed_cell_index",
    "best_observed_rho_conv",
    "best_observed_rho_dense",
    "best_observed_lr_conv_weight_0",
    "best_observed_lr_conv_weight_1",
    "best_observed_lr_conv_weight_2",
    "best_observed_lr_dense_weight_0",
    "best_observed_weight_learning_rates_json",
    "best_observed_bias_learning_rates_json",
    "best_observed_bias_lrs_zero",
    "best_observed_conv_axis_index",
    "best_observed_dense_axis_index",
    "maximum_accuracy_co_maxima_count",
    "maximum_accuracy_co_maxima_json",
    "best_observed_validation_accuracy",
    "best_observed_validation_accuracy_percent",
    "best_observed_validation_loss",
    "best_observed_exact_bound_occupancy_percent",
    "best_observed_exact_bound_occupancy_by_parameter_percent_json",
    "best_observed_median_projection_efficiency",
    "median_validation_accuracy",
    "median_validation_loss",
    "median_exact_bound_occupancy_percent",
    "median_projection_efficiency",
    "accuracy_vs_exact_bound_occupancy_sample_count",
    "accuracy_vs_exact_bound_occupancy_pearson",
    "accuracy_vs_exact_bound_occupancy_spearman",
    "accuracy_vs_exact_bound_occupancy_by_parameter_json",
    "accuracy_vs_median_projection_efficiency_sample_count",
    "accuracy_vs_median_projection_efficiency_pearson",
    "accuracy_vs_median_projection_efficiency_spearman",
    "range_status",
    "range_status_statement",
    "locally_bracketed_by_accuracy",
    "open_directions_json",
    "nonnumeric_axial_directions_json",
    "improving_or_tied_directions_json",
    "axial_neighbors_json",
    "correlations_noncausal",
    "exploratory_noncanonical",
)
CELL_FIELDS = (
    "study_id",
    "source_shard_root",
    "surface_index",
    "surface_id",
    "scheme",
    "optimizer",
    "target",
    "cell_index",
    "execution_rank",
    "conv_axis_index",
    "dense_axis_index",
    "rho_conv",
    "rho_dense",
    "canonical_outcome",
    "raw_status",
    "nonfinite_kind",
    "nonfinite_parameter",
    "nonfinite_confirmed_step",
    "completed_steps",
    "final_validation_accuracy",
    "final_validation_accuracy_percent",
    "final_validation_loss",
    "exact_lower_bound_count",
    "exact_upper_bound_count",
    "exact_either_bound_count",
    "bounded_weight_count",
    "exact_bound_occupancy_fraction",
    "exact_bound_occupancy_percent",
    "exact_bound_occupancy_by_parameter_percent_json",
    "diagnostic_endpoint_bound_occupancy_mean",
    "diagnostic_endpoint_bound_occupancy_max",
    "final_bound_occupancy_by_parameter_json",
    "median_projection_efficiency",
    "median_projection_efficiency_by_parameter_json",
    "median_proposed_update_rms_by_parameter_json",
    "median_applied_update_rms_by_parameter_json",
    "learning_rates_by_parameter_json",
    "manifest_json_sha256",
    "status_json_sha256",
    "metrics_jsonl_sha256",
    "result_json_sha256",
    "initializer_checkpoint_sha256",
    "probe_initial_parameter_sha256",
    "bias_lr_zero",
    "final_biases_exact_zero",
    "official_test_read",
    "canonical_bundle_valid",
    "exploratory_noncanonical",
    "run_dir",
)


@dataclass(frozen=True)
class StudyContract:
    path: Path
    payload: dict[str, Any]
    sha256: str
    study_id: str
    initializer_sha256: str
    rho_conv: tuple[float, ...]
    rho_dense: tuple[float, ...]
    surfaces: tuple[dict[str, Any], ...]
    weight_min: float
    weight_max: float


@dataclass(frozen=True)
class Shard:
    root: Path
    surface: dict[str, Any]
    resolved: dict[str, Any]
    target: str


class IncompleteExploratoryCoverageError(RuntimeError):
    """A valid terminal scientific outcome does not provide the requested grid."""

    def __init__(self, surface_id: str, summary_status: str):
        if summary_status == "unresolved_probe":
            detail = "the optimizer proposal probe did not resolve"
        else:
            detail = f"the terminal surface status is {summary_status!r}"
        super().__init__(
            f"Cannot analyze the complete 4x64 exploratory grid: surface "
            f"{surface_id!r} is a legitimate scientific terminal, but {detail}. "
            "This is not a transport failure; the surface has zero validated "
            "candidate bundles."
        )
        self.surface_id = surface_id
        self.summary_status = summary_status


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON object {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _finite_float(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Expected finite {label}, got {value!r}.")
    return float(value)


def _optional_finite_float(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, label=label)


def _finite_fraction(value: Any, *, label: str) -> float:
    number = _finite_float(value, label=label)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"Expected {label} in [0,1], got {number!r}.")
    return number


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Expected lowercase SHA-256 for {label}, got {value!r}.")
    return value


def _expected_axes() -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        tuple(0.009 / (3.0**power) for power in range(7, -1, -1)),
        tuple(0.03 / (3.0**power) for power in range(7, -1, -1)),
    )


def _exact_float_axis(value: Any, expected: Sequence[float], *, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != len(expected):
        raise ValueError(f"Expected eight values for {label}.")
    observed = tuple(_finite_float(item, label=label) for item in value)
    if any(
        not math.isclose(left, right, rel_tol=1e-15, abs_tol=0.0)
        for left, right in zip(observed, expected)
    ):
        raise ValueError(f"Unexpected {label}: {observed!r}.")
    return observed


def _surface_specs() -> tuple[dict[str, Any], ...]:
    rows = []
    for scheme in ("baseline", "ours"):
        for optimizer in ("SGD", "Adam"):
            rows.append(
                {
                    "index": len(rows),
                    "initializer": "bounded_uniform",
                    "architecture": "conv3",
                    "scheme": scheme,
                    "optimizer": optimizer,
                    "surface_id": f"bounded_uniform__conv3__{scheme}__{optimizer.lower()}",
                }
            )
    return tuple(rows)


def load_contract(path: Path = DEFAULT_STUDY) -> StudyContract:
    source = Path(path).expanduser().resolve()
    study = _load_json(source)
    if study.get("schema_version") != STUDY_SCHEMA:
        raise ValueError(f"Unexpected exploratory study schema in {source}.")
    if study.get("evidence_class") != "ordinary_mnist_exploratory":
        raise ValueError("The analyzer only accepts ordinary-MNIST exploratory evidence.")
    expected_scope = {
        "architectures": ["conv3"],
        "schemes": ["baseline", "ours"],
        "optimizers": ["SGD", "Adam"],
        "initializers": ["bounded_uniform"],
        "excluded": ["legacy", "conv1", "conv2"],
    }
    if study.get("scope") != expected_scope:
        raise ValueError("The study scope is not exactly Conv3 baseline/ours x SGD/Adam.")
    study_id = study.get("study_id")
    if not isinstance(study_id, str) or not study_id:
        raise ValueError("The exploratory study_id is missing.")
    parent = study.get("parent")
    if not isinstance(parent, Mapping):
        raise ValueError("The exploratory parent record is missing.")
    initializer_sha256 = _require_sha256(
        parent.get("initializer_checkpoint_sha256"),
        label="parent.initializer_checkpoint_sha256",
    )
    if source == Path(DEFAULT_STUDY).resolve() and initializer_sha256 != EXPECTED_INITIALIZER_SHA256:
        raise ValueError("The default study does not pin the expected initializer hash.")
    search = study.get("rho_search")
    if not isinstance(search, Mapping):
        raise ValueError("The exploratory rho_search contract is missing.")
    expected_conv, expected_dense = _expected_axes()
    rho_conv = _exact_float_axis(search.get("rho_conv"), expected_conv, label="rho_conv")
    rho_dense = _exact_float_axis(search.get("rho_dense"), expected_dense, label="rho_dense")
    required = {
        "candidate_epochs": 3,
        "expected_candidate_steps": 10314,
        "minimum_validation_accuracy": 0.0,
        "skip_canary": True,
        "canary_steps": 0,
        "disable_safety_rejections": True,
        "post_candidate_tk": False,
        "bias_policy": "zero",
    }
    for key, expected in required.items():
        if search.get(key) != expected:
            raise ValueError(f"Expected rho_search.{key}={expected!r}.")
    return StudyContract(
        path=source,
        payload=study,
        sha256=sha256_file(source),
        study_id=study_id,
        initializer_sha256=initializer_sha256,
        rho_conv=rho_conv,
        rho_dense=rho_dense,
        surfaces=_surface_specs(),
        weight_min=1e-5,
        weight_max=1e-4,
    )


def discover_shard_roots(
    *, study_root: Path | None, shard_roots: Sequence[Path]
) -> list[Path]:
    discovered: list[Path] = []
    if study_root is not None:
        root = Path(study_root).expanduser().resolve()
        shard_parent = root / "shards"
        if not shard_parent.is_dir():
            raise ValueError(f"Study root has no shards directory: {shard_parent}.")
        discovered.extend(path.parent for path in sorted(shard_parent.rglob("study.resolved.json")))
    discovered.extend(Path(path).expanduser().resolve() for path in shard_roots)
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in discovered:
        root = root.resolve()
        if root in seen:
            continue
        if not root.is_dir() or not (root / "study.resolved.json").is_file():
            raise ValueError(f"Invalid collected shard root: {root}.")
        unique.append(root)
        seen.add(root)
    if not unique:
        raise ValueError("No collected shard roots were supplied or discovered.")
    return unique


def _load_shards(roots: Sequence[Path], contract: StudyContract) -> list[Shard]:
    expected_by_id = {row["surface_id"]: row for row in contract.surfaces}
    shards: list[Shard] = []
    observed_ids: list[str] = []
    for root in roots:
        resolved = _load_json(root / "study.resolved.json")
        if resolved.get("schema_version") != RESOLVED_SCHEMA:
            raise ValueError(f"Unexpected resolved-study schema in {root}.")
        required_equal = {
            "study_id": contract.study_id,
            "evidence_class": "ordinary_mnist_exploratory",
            "study_config_sha256": contract.sha256,
            "initializer_checkpoint_sha256": contract.initializer_sha256,
            "surface_count": 4,
            "cells_per_surface": EXPECTED_CELL_COUNT,
            "surfaces": list(contract.surfaces),
            "official_test_read": False,
            "restart_interrupted_cells": True,
        }
        for key, expected in required_equal.items():
            if resolved.get(key) != expected:
                raise ValueError(f"Resolved shard {root} has invalid {key}.")
        target = resolved.get("target")
        if target not in ACCEPTED_TARGETS:
            raise ValueError(
                f"Resolved shard {root} has invalid target {target!r}; expected one "
                f"of {sorted(ACCEPTED_TARGETS)!r}."
            )
        code = resolved.get("code")
        if not isinstance(code, Mapping):
            raise ValueError(f"Resolved shard {root} has no source identity.")
        commit = code.get("commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise ValueError(f"Resolved shard {root} has invalid source commit.")
        source_archive_sha256 = _require_sha256(
            code.get("source_archive_sha256"),
            label=f"{root} resolved source archive",
        )
        if (
            code.get("dirty") is not False
            or code.get("source_kind") != "frozen_archive"
            or code.get("working_tree_sha256") != source_archive_sha256
        ):
            raise ValueError(
                f"Resolved shard {root} does not prove one clean frozen source archive."
            )
        surface_parent = root / "surfaces"
        present = sorted(
            path.name
            for path in surface_parent.iterdir()
            if path.is_dir() and path.name in expected_by_id
        ) if surface_parent.is_dir() else []
        unexpected = sorted(
            path.name
            for path in surface_parent.iterdir()
            if path.is_dir() and path.name not in expected_by_id
        ) if surface_parent.is_dir() else []
        if len(present) != 1 or unexpected:
            raise ValueError(
                f"Expected exactly one known surface in shard {root}; "
                f"known={present!r}, unexpected={unexpected!r}."
            )
        surface = expected_by_id[present[0]]
        shards.append(
            Shard(root=root, surface=surface, resolved=resolved, target=str(target))
        )
        observed_ids.append(present[0])
    expected_ids = sorted(expected_by_id)
    if len(roots) != 4 or sorted(observed_ids) != expected_ids:
        raise ValueError(
            "Expected exactly four unique shard surfaces (baseline/ours x SGD/Adam); "
            f"observed={sorted(observed_ids)!r}."
        )
    return sorted(shards, key=lambda shard: int(shard.surface["index"]))


def _receipt_roots(shards: Sequence[Shard], extra_roots: Sequence[Path]) -> list[Path]:
    roots = [Path(path).expanduser().resolve() for path in extra_roots]
    for shard in shards:
        if shard.root.parent.name == "shards":
            roots.append(shard.root.parent.parent / "transport_receipts")
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root not in seen:
            unique.append(root)
            seen.add(root)
    return unique


def _receipt_paths(shards: Sequence[Shard], extra_roots: Sequence[Path]) -> list[Path]:
    """Return unique local and Jean Zay receipt candidates.

    Local receipts are authoritative only at ``SHARD/transport_receipt.json``;
    explicit search roots are still scanned so duplicate or misplaced receipts
    fail closed instead of being silently ignored.
    """

    candidates = [shard.root / "transport_receipt.json" for shard in shards]
    for root in _receipt_roots(shards, extra_roots):
        if root.is_file():
            candidates.append(root)
        elif root.is_dir():
            candidates.extend(root.rglob("jean-zay-task_*.json"))
            candidates.extend(root.rglob("transport_receipt.json"))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        path = path.expanduser().resolve()
        if not path.is_file() or path in seen:
            continue
        unique.append(path)
        seen.add(path)
    return unique


def _path_suffix_matches(value: Any, suffix: Sequence[str]) -> bool:
    if not isinstance(value, str) or not value:
        return False
    parts = Path(value).parts
    return len(parts) >= len(suffix) and tuple(parts[-len(suffix) :]) == tuple(suffix)


def _validate_receipts(
    shards: Sequence[Shard],
    contract: StudyContract,
    *,
    receipt_roots: Sequence[Path],
) -> list[dict[str, Any]]:
    expected_by_id = {
        str(surface["surface_id"]): surface for surface in contract.surfaces
    }
    local_path_to_shard = {
        (shard.root / "transport_receipt.json").resolve(): shard for shard in shards
    }
    inventory: dict[int, list[tuple[Path, dict[str, Any]]]] = {
        int(surface["index"]): [] for surface in contract.surfaces
    }
    for path in _receipt_paths(shards, receipt_roots):
        receipt = _load_json(path)
        schema = receipt.get("schema_version")
        if schema not in {RECEIPT_SCHEMA, LOCAL_RECEIPT_SCHEMA}:
            continue
        if receipt.get("study_id") != contract.study_id:
            continue
        if schema == RECEIPT_SCHEMA:
            task_id = receipt.get("task_id")
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id not in inventory:
                raise ValueError(f"Invalid exploratory receipt task_id in {path}.")
            inventory[task_id].append((path.resolve(), receipt))
            continue
        surface_id = receipt.get("surface_id")
        if not isinstance(surface_id, str) or surface_id not in expected_by_id:
            raise ValueError(f"Invalid exploratory local receipt surface_id in {path}.")
        containing_shard = local_path_to_shard.get(path.resolve())
        if (
            containing_shard is not None
            and surface_id != containing_shard.surface["surface_id"]
        ):
            raise ValueError(
                f"Local transport receipt {path} does not bind its containing "
                "shard surface."
            )
        index = int(expected_by_id[surface_id]["index"])
        inventory[index].append((path.resolve(), receipt))
    validated: list[dict[str, Any]] = []
    for shard in shards:
        index = int(shard.surface["index"])
        records = inventory[index]
        if len(records) != 1:
            raise ValueError(
                f"Expected exactly one terminal transport receipt for surface "
                f"{shard.surface['surface_id']!r}; found {len(records)}."
            )
        path, receipt = records[0]
        summary_path = (
            shard.root / "surfaces" / str(shard.surface["surface_id"]) / "summary.json"
        )
        summary = _load_json(summary_path)
        schema = receipt.get("schema_version")
        if schema == RECEIPT_SCHEMA:
            required = {
                "schema_version": RECEIPT_SCHEMA,
                "study_id": contract.study_id,
                "surface": shard.surface,
                "task_id": index,
                "target": shard.target,
                "study_config_sha256": contract.sha256,
                "initializer_checkpoint_sha256": contract.initializer_sha256,
                "expected_cells": 64,
                "official_test_read": False,
                "semantic_pass": True,
            }
            for key, expected in required.items():
                if receipt.get(key) != expected:
                    raise ValueError(f"Transport receipt {path} has invalid {key}.")
            if shard.target != "jean-zay":
                raise ValueError(
                    f"Jean Zay receipt {path} cannot bind target {shard.target!r}."
                )
            if path.name != f"jean-zay-task_{index}.json":
                raise ValueError(f"Jean Zay receipt {path} has an invalid filename.")
            if not isinstance(receipt.get("environment_id"), str) or not receipt[
                "environment_id"
            ]:
                raise ValueError(f"Transport receipt {path} is missing environment_id.")
        elif schema == LOCAL_RECEIPT_SCHEMA:
            required = {
                "schema_version": LOCAL_RECEIPT_SCHEMA,
                "study_id": contract.study_id,
                "surface_id": shard.surface["surface_id"],
                "target": shard.target,
                "initializer_checkpoint_sha256": contract.initializer_sha256,
                "official_test_read": False,
            }
            for key, expected in required.items():
                if receipt.get(key) != expected:
                    raise ValueError(f"Local transport receipt {path} has invalid {key}.")
            if shard.target not in {"akib", "trex"}:
                raise ValueError(
                    f"Local transport receipt {path} cannot bind target "
                    f"{shard.target!r}."
                )
            expected_path = (shard.root / "transport_receipt.json").resolve()
            if path != expected_path:
                raise ValueError(
                    f"Local transport receipt for {shard.surface['surface_id']!r} "
                    f"must live inside its shard at {expected_path}; found {path}."
                )
            for key in ("host", "python", "torch", "cuda_device", "recorded_at"):
                if not isinstance(receipt.get(key), str) or not receipt[key]:
                    raise ValueError(f"Local transport receipt {path} is missing {key}.")
            try:
                recorded_at = datetime.fromisoformat(str(receipt["recorded_at"]))
            except ValueError as error:
                raise ValueError(
                    f"Local transport receipt {path} has invalid recorded_at."
                ) from error
            if recorded_at.tzinfo is None:
                raise ValueError(
                    f"Local transport receipt {path} has timezone-naive recorded_at."
                )
        else:  # Inventory construction admits only the two schemas above.
            raise AssertionError(f"Unexpected receipt schema after inventory: {schema!r}")
        source_commit = receipt.get("source_commit")
        if (
            not isinstance(source_commit, str)
            or len(source_commit) != 40
            or any(character not in "0123456789abcdef" for character in source_commit)
        ):
            raise ValueError(f"Transport receipt {path} has invalid source_commit.")
        source_archive_sha256 = _require_sha256(
            receipt.get("source_archive_sha256"), label=f"{path} source archive"
        )
        code = shard.resolved["code"]
        if (
            source_commit != code["commit"]
            or source_archive_sha256 != code["source_archive_sha256"]
        ):
            raise ValueError(
                f"Transport receipt {path} does not bind the shard's resolved source."
            )
        if schema == RECEIPT_SCHEMA:
            if (
                not _path_suffix_matches(
                    receipt.get("summary_path"),
                    ("surfaces", str(shard.surface["surface_id"]), "summary.json"),
                )
                or receipt.get("summary_sha256") != sha256_file(summary_path)
            ):
                raise ValueError(f"Transport receipt {path} does not bind the local summary.")
            summary_status = receipt.get("summary_status")
        else:
            summary_status = receipt.get("surface_status")
        if summary_status == "unresolved_probe":
            if (
                receipt.get("validated_cell_bundles") != 0
                or summary.get("status") != "unresolved_probe"
                or summary.get("terminal_cells") != 0
                or summary.get("candidates") not in ([], None)
            ):
                raise ValueError(f"Malformed unresolved-probe receipt {path}.")
            if schema == RECEIPT_SCHEMA and receipt.get("terminal_cells") != 0:
                raise ValueError(f"Malformed unresolved-probe receipt {path}.")
            raise IncompleteExploratoryCoverageError(
                str(shard.surface["surface_id"]), "unresolved_probe"
            )
        if summary_status != "complete":
            raise ValueError(f"Unsupported terminal summary status in receipt {path}.")
        if (
            receipt.get("validated_cell_bundles") != 64
            or summary.get("status") != "complete"
            or summary.get("complete") is not True
            or summary.get("expected_cells") != 64
            or summary.get("terminal_cells") != 64
        ):
            raise ValueError(
                f"Receipt {path} does not prove a complete, validated 64-cell surface."
            )
        if schema == RECEIPT_SCHEMA and (
            receipt.get("terminal_cells") != 64
            or receipt.get("status_counts") != summary.get("status_counts")
        ):
            raise ValueError(
                f"Receipt {path} does not prove a complete, validated 64-cell surface."
            )
        validated.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "schema_version": schema,
                "target": shard.target,
                "surface_id": shard.surface["surface_id"],
                "summary_status": "complete",
                "validated_cell_bundles": 64,
                "source_commit": source_commit,
                "source_archive_sha256": source_archive_sha256,
                "execution_environment": {
                    "target": shard.target,
                    "host": receipt.get("host"),
                    "python": receipt.get("python"),
                    "torch": receipt.get("torch"),
                    "cuda_device": receipt.get("cuda_device"),
                    "environment_id": receipt.get("environment_id"),
                },
            }
        )
    source_identities = {
        (row["source_commit"], row["source_archive_sha256"]) for row in validated
    }
    if len(source_identities) != 1:
        raise ValueError(
            "Transport receipts do not share one source commit/archive identity."
        )
    return validated


def _target_assignment(
    contract: StudyContract, receipts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Derive the scientific target assignment from validated receipts."""

    expected_ids = {str(surface["surface_id"]) for surface in contract.surfaces}
    targets_by_surface: dict[str, str] = {}
    environments_by_surface: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        surface_id = str(receipt.get("surface_id"))
        target = receipt.get("target")
        if surface_id in targets_by_surface:
            raise ValueError(
                f"Duplicate validated target assignment for surface {surface_id!r}."
            )
        if surface_id not in expected_ids or target not in ACCEPTED_TARGETS:
            raise ValueError("Validated receipts contain an invalid target assignment.")
        targets_by_surface[surface_id] = str(target)
        environment = receipt.get("execution_environment")
        if not isinstance(environment, Mapping):
            raise ValueError(
                f"Validated receipt for {surface_id!r} lacks execution environment."
            )
        environments_by_surface[surface_id] = dict(environment)
    if set(targets_by_surface) != expected_ids:
        raise ValueError("Validated receipts do not cover every surface target assignment.")

    surfaces = [
        {
            "surface_index": int(surface["index"]),
            "surface_id": str(surface["surface_id"]),
            "scheme": str(surface["scheme"]),
            "optimizer": str(surface["optimizer"]),
            "target": targets_by_surface[str(surface["surface_id"])],
            "execution_environment": environments_by_surface[
                str(surface["surface_id"])
            ],
        }
        for surface in contract.surfaces
    ]
    within_optimizer: list[dict[str, Any]] = []
    targets_by_optimizer: dict[str, str] = {}
    environments_by_optimizer: dict[str, list[dict[str, Any]]] = {}
    for optimizer in ("SGD", "Adam"):
        by_scheme = {
            row["scheme"]: row
            for row in surfaces
            if row["optimizer"] == optimizer
        }
        if set(by_scheme) != {"baseline", "ours"}:
            raise ValueError(
                f"Target assignment does not contain baseline and ours for {optimizer}."
            )
        baseline_target = str(by_scheme["baseline"]["target"])
        ours_target = str(by_scheme["ours"]["target"])
        if baseline_target != ours_target:
            raise ValueError(
                f"Matched baseline-vs-ours {optimizer} surfaces are split across "
                f"targets ({baseline_target!r} versus {ours_target!r}); refusing "
                "a host-confounded scheme comparison."
            )
        targets_by_optimizer[optimizer] = baseline_target
        baseline_environment = dict(by_scheme["baseline"]["execution_environment"])
        ours_environment = dict(by_scheme["ours"]["execution_environment"])
        same_runtime = baseline_environment == ours_environment
        environments_by_optimizer[optimizer] = [
            baseline_environment,
            ours_environment,
        ]
        if same_runtime:
            statement = (
                f"Baseline-vs-ours for {optimizer} is a within-host/runtime "
                f"comparison on {baseline_target}."
            )
        else:
            statement = (
                f"Baseline-vs-ours for {optimizer} shares target label "
                f"{baseline_target}, but its validated host/runtime tuples differ; "
                "scheme differences are host/runtime-confounded."
            )
        within_optimizer.append(
            {
                "optimizer": optimizer,
                "target": baseline_target,
                "baseline_surface_id": by_scheme["baseline"]["surface_id"],
                "ours_surface_id": by_scheme["ours"]["surface_id"],
                "same_target": True,
                "same_host_runtime_contract": same_runtime,
                "host_runtime_confounded": not same_runtime,
                "host_target_confounded": not same_runtime,
                "causal_scheme_interpretation_allowed": same_runtime,
                "baseline_execution_environment": baseline_environment,
                "ours_execution_environment": ours_environment,
                "statement": statement,
            }
        )

    optimizer_targets_differ = (
        targets_by_optimizer["SGD"] != targets_by_optimizer["Adam"]
    )
    optimizer_environments_differ = (
        environments_by_optimizer["SGD"] != environments_by_optimizer["Adam"]
    )
    cross_optimizer_runtime_confounded = (
        optimizer_targets_differ or optimizer_environments_differ
    )
    if cross_optimizer_runtime_confounded:
        cross_optimizer_statement = (
            f"SGD ran on {targets_by_optimizer['SGD']} while Adam ran on "
            f"{targets_by_optimizer['Adam']}; their validated host/runtime "
            "contracts differ, so SGD-vs-Adam differences must not be interpreted "
            "causally as optimizer effects."
        )
    else:
        cross_optimizer_statement = (
            f"SGD and Adam both ran on {targets_by_optimizer['SGD']}; this target "
            "assignment adds no cross-optimizer host difference, but the "
            "exploratory design still does not support causal claims."
        )
    return {
        "derived_from_validated_transport_receipts": True,
        "surfaces": surfaces,
        "within_optimizer_scheme_comparisons": within_optimizer,
        "targets_by_optimizer": targets_by_optimizer,
        "execution_environments_by_optimizer": environments_by_optimizer,
        "cross_optimizer_comparison": {
            "same_target": not optimizer_targets_differ,
            "same_host_runtime_contract": not optimizer_environments_differ,
            "host_runtime_confounded": cross_optimizer_runtime_confounded,
            "causal_interpretation_allowed": False,
            "statement": cross_optimizer_statement,
        },
    }


def _verify_initializer(shard: Shard, contract: StudyContract) -> None:
    asset_root = shard.root / "assets" / "bounded_uniform" / "conv3"
    checkpoint = asset_root / "final_model.pt"
    asset = _load_json(asset_root / "asset.json")
    expected = contract.initializer_sha256
    required = {
        "architecture": "conv3",
        "initializer": "bounded_uniform",
        "checkpoint_sha256": expected,
        "expected_checkpoint_sha256": expected,
        "official_test_read": False,
    }
    for key, value in required.items():
        if asset.get(key) != value:
            raise ValueError(f"Initializer asset in {shard.root} has invalid {key}.")
    checkpoint_suffix = (
        shard.root.name,
        "assets",
        "bounded_uniform",
        "conv3",
        "final_model.pt",
    )
    if not _path_suffix_matches(asset.get("checkpoint"), checkpoint_suffix):
        raise ValueError(
            f"Initializer asset in {shard.root} does not identify this shard's "
            "copied checkpoint."
        )
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected:
        raise ValueError(f"Initializer checkpoint bytes do not match in {shard.root}.")
    zero_bias = asset.get("zero_bias_checkpoint_verification")
    if (
        not isinstance(zero_bias, Mapping)
        or zero_bias.get("all_exact_zero") is not True
        or zero_bias.get("nonzero_bias_element_count") != 0
        or sorted(zero_bias.get("expected_bias_names", [])) != list(BIAS_NAMES)
    ):
        raise ValueError(f"Initializer zero-bias verification failed in {shard.root}.")


def _governing_parent_config(contract: StudyContract) -> dict[str, Any]:
    """Load the hash-pinned parent used to construct every source config."""

    parent_record = contract.payload.get("parent")
    if not isinstance(parent_record, Mapping):
        raise ValueError("The exploratory study is missing its parent contract.")
    parent_reference = parent_record.get("study_config")
    parent_sha256 = _require_sha256(
        parent_record.get("study_config_sha256"),
        label="parent.study_config_sha256",
    )
    if not isinstance(parent_reference, str) or not parent_reference:
        raise ValueError("The exploratory parent study_config path is missing.")
    path = Path(parent_reference).expanduser()
    if not path.is_absolute():
        repository_root = Path(__file__).resolve().parents[1]
        path = repository_root / path
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != parent_sha256:
        raise ValueError("The hash-pinned governing parent config is unavailable or changed.")
    parent = _load_json(path)
    if parent.get("study_id") != parent_record.get("study_id"):
        raise ValueError("The governing parent config has the wrong study identity.")
    return parent


def _verify_source_config(shard: Shard, contract: StudyContract) -> tuple[dict[str, Any], str]:
    surface_dir = shard.root / "surfaces" / str(shard.surface["surface_id"])
    source_path = surface_dir / "source_config.json"
    source = _load_json(source_path)
    init_path = source.get("init_checkpoint_path")
    checkpoint_suffix = (
        shard.root.name,
        "assets",
        "bounded_uniform",
        "conv3",
        "final_model.pt",
    )
    if not _path_suffix_matches(init_path, checkpoint_suffix):
        raise ValueError(
            f"Source config {source_path} does not bind this shard's pinned "
            "initializer asset."
        )
    datasets = source.get("datasets")
    mnist = datasets.get("mnist") if isinstance(datasets, Mapping) else None
    params = mnist.get("params") if isinstance(mnist, Mapping) else None
    dataset_root = params.get("root") if isinstance(params, Mapping) else None
    if not isinstance(dataset_root, str) or not dataset_root:
        raise ValueError(f"Source config {source_path} has no MNIST dataset root.")
    parent = _governing_parent_config(contract)
    expected = build_source_config(
        parent,
        initializer="bounded_uniform",
        architecture="conv3",
        scheme=str(shard.surface["scheme"]),
        optimizer=str(shard.surface["optimizer"]),
        init_checkpoint_path=Path(str(init_path)),
        dataset_root=Path(dataset_root),
    )
    if source != expected:
        raise ValueError(
            f"Source config {source_path} does not match the complete governed "
            "Conv3 BP training contract."
        )
    return source, sha256_file(source_path)


def _verify_probe(
    shard: Shard,
    contract: StudyContract,
    *,
    source_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    surface_dir = shard.root / "surfaces" / str(shard.surface["surface_id"])
    resolved_path = surface_dir / "rho" / "resolved.json"
    probe_path = surface_dir / "rho" / "probe.json"
    resolved = _load_json(resolved_path)
    probe = _load_json(probe_path)
    search = probe.get("search_signature")
    if not isinstance(search, Mapping):
        raise ValueError(f"Optimizer probe lacks its search signature: {probe_path}.")
    if resolved.get("schema_version") != "conv-rho-search/v1" or search != resolved:
        raise ValueError(
            "Optimizer probe search signature does not exactly match "
            f"rho/resolved.json: {probe_path}."
        )
    expected_search = {
        "schema_version": "conv-rho-search/v1",
        "source_config_sha256": source_sha256,
        "code": shard.resolved["code"],
        "optimizer": shard.surface["optimizer"],
        "evidence_class": "ordinary_mnist_exploratory",
        "rho_conv": list(contract.rho_conv),
        "rho_dense": list(contract.rho_dense),
        "bias_policy": "zero",
        "probe_stability_scope": "weights_only",
        "probe_batches": [128],
        "stability_tolerance": 0.5,
        "epochs": 3,
        "max_batches": None,
        "max_validation_batches": None,
        "split_seed": 0,
        "shuffle_seed": 0,
        "validation_batch_size": 64,
        "device": "cuda",
        "canary_steps": 0,
        "skip_canary": True,
        "restart_interrupted_cells": True,
        "expected_candidate_steps": 10314,
        "minimum_validation_accuracy": 0.0,
        "target": shard.target,
    }
    for key, expected in expected_search.items():
        if search.get(key) != expected:
            raise ValueError(
                f"Optimizer probe search signature has invalid {key}: {probe_path}."
            )
    search_safety = search.get("safety")
    if (
        not isinstance(search_safety, Mapping)
        or search_safety.get("rejections_enabled") is not False
        or search_safety.get("bound_occupancy") != "report_only"
        or search_safety.get("projection_efficiency") != "report_only"
        or search_safety.get("bound_occupancy_increase_maximum") is not None
        or search_safety.get("projection_efficiency_minimum") is not None
    ):
        raise ValueError(
            f"Optimizer probe search safety is not report-only: {probe_path}."
        )
    if (
        probe.get("schema_version") != "conv-rho-probe/v1"
        or probe.get("status") != "complete"
        or probe.get("probe_stable") is not True
        or probe.get("proposal_units_valid") is not True
        or probe.get("parameter_names") != list(WEIGHT_NAMES + BIAS_NAMES)
        or probe.get("weight_names") != list(WEIGHT_NAMES)
    ):
        raise ValueError(f"Surface optimizer probe is not resolved: {probe_path}.")
    units = probe.get("normalization_unit_by_weight")
    if not isinstance(units, Mapping) or set(units) != set(WEIGHT_NAMES):
        raise ValueError(f"Optimizer probe has invalid weight normalization units: {probe_path}.")
    for name in WEIGHT_NAMES:
        if _finite_float(units[name], label=f"probe unit {name}") <= 0.0:
            raise ValueError(f"Optimizer probe has nonpositive unit for {name}: {probe_path}.")
    probe_zero = probe.get("zero_bias_checkpoint_verification")
    if not isinstance(probe_zero, Mapping) or probe_zero.get("all_exact_zero") is not True:
        raise ValueError(f"Probe zero-bias verification failed: {probe_path}.")
    _require_sha256(
        probe.get("initial_parameter_sha256"),
        label=f"{probe_path} initial parameter identity",
    )
    semantic_sha256 = hashlib.sha256(
        json.dumps(probe, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return probe, semantic_sha256, sha256_file(probe_path)


def classify_cell_status(
    raw_status: Any,
    safety_failure: Any,
    *,
    reporting_state: Any,
    result_present: bool,
) -> tuple[str, str | None]:
    """Return the canonical exploratory outcome or reject an alien status."""

    if raw_status == NUMERIC_STATUS:
        if reporting_state != "complete" or not result_present:
            raise ValueError("A numeric cell must be a canonical complete bundle.")
        return "numeric_complete", None
    expected_kind = NONFINITE_STATUSES.get(raw_status)
    if expected_kind is None:
        raise ValueError(f"Unexpected exploratory cell status: {raw_status!r}.")
    if (
        not isinstance(safety_failure, Mapping)
        or safety_failure.get("kind") != expected_kind
    ):
        raise ValueError(
            f"Status {raw_status!r} requires non-finite kind {expected_kind!r}."
        )
    if reporting_state != "failed" or result_present:
        raise ValueError("A non-finite cell must be a canonical failed bundle.")
    return "nonfinite", expected_kind


def _verify_official_test_unread(
    manifest: Mapping[str, Any], result: Mapping[str, Any] | None, metrics: Mapping[str, Any] | None
) -> None:
    dataset = manifest.get("dataset")
    if (
        manifest.get("evidence_class") != "ordinary_mnist_exploratory"
        or manifest.get("smoke") is not False
        or not isinstance(dataset, Mapping)
        or dataset.get("evaluation_split") != "validation"
        or dataset.get("official_test_read") is not False
    ):
        raise ValueError("Cell manifest does not prove validation-only ordinary MNIST use.")
    if result is not None:
        result_dataset = result.get("dataset")
        completion = result.get("completion")
        if (
            result.get("evidence_class") != "ordinary_mnist_exploratory"
            or not isinstance(result_dataset, Mapping)
            or result_dataset.get("official_test_read") is not False
            or not isinstance(completion, Mapping)
            or completion.get("official_test_read") is not False
        ):
            raise ValueError("Cell result does not declare official_test_read=false.")
    if metrics is not None and (
        metrics.get("official_test_accuracy") is not None
        or metrics.get("official_test_loss") is not None
        or metrics.get("official_test_examples") != 0
        or metrics.get("official_test_evaluations") != 0
    ):
        raise ValueError("Cell metrics contain an official-test evaluation.")


def _finite_vector(value: Any, *, label: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"Expected {label} to contain {length} values.")
    return [_finite_float(item, label=label) for item in value]


def _same_float(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-15, abs_tol=0.0)


def _same_float_vector(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        _same_float(a, b) for a, b in zip(left, right)
    )


def _manifest_input(
    manifest: Mapping[str, Any], role: str, *, run_dir: Path
) -> Mapping[str, Any]:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError(f"Manifest inputs are missing in {run_dir}.")
    records = [
        item
        for item in inputs
        if isinstance(item, Mapping) and item.get("role") == role
    ]
    if len(records) != 1:
        raise ValueError(
            f"Expected exactly one manifest input role {role!r} in {run_dir}."
        )
    return records[0]


def _verify_cell_scientific_identity(
    *,
    shard: Shard,
    cell_dir: Path,
    cell: Mapping[str, Any],
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
    expected_index: int,
    expected_rho_conv: float,
    expected_rho_dense: float,
    source_sha256: str,
    source_config: Mapping[str, Any],
    probe: Mapping[str, Any],
    probe_semantic_sha256: str,
    probe_file_sha256: str,
    receipt_environment: Mapping[str, Any],
) -> tuple[dict[str, float], list[float]]:
    signature = cell.get("signature")
    if not isinstance(signature, Mapping):
        raise ValueError(f"Missing cell signature in {cell_dir}.")
    for label, value in (
        ("cell.rho_conv", cell.get("rho_conv")),
        ("signature.rho_conv", signature.get("rho_conv")),
    ):
        if not _same_float(
            _finite_float(value, label=label), expected_rho_conv
        ):
            raise ValueError(f"Cell rho_conv identity mismatch in {cell_dir}.")
    for label, value in (
        ("cell.rho_dense", cell.get("rho_dense")),
        ("signature.rho_dense", signature.get("rho_dense")),
    ):
        if not _same_float(
            _finite_float(value, label=label), expected_rho_dense
        ):
            raise ValueError(f"Cell rho_dense identity mismatch in {cell_dir}.")
    if (
        signature.get("source_config_sha256") != source_sha256
        or signature.get("probe_sha256") != probe_semantic_sha256
        or signature.get("code") != shard.resolved["code"]
        or signature.get("safety")
        != probe.get("search_signature", {}).get("safety")
    ):
        raise ValueError(
            f"Cell source/probe/code signature identity mismatch in {cell_dir}."
        )

    parameter_names = probe.get("parameter_names")
    if parameter_names != list(WEIGHT_NAMES + BIAS_NAMES):
        raise ValueError(f"Probe parameter order is invalid for {cell_dir}.")
    raw_rates = cell.get("learning_rates_by_parameter")
    if not isinstance(raw_rates, Mapping) or set(raw_rates) != set(parameter_names):
        raise ValueError(f"Unexpected learning-rate map in {cell_dir}.")
    rates = {
        name: _finite_float(raw_rates[name], label=f"LR {name}")
        for name in parameter_names
    }
    vector = _finite_vector(
        cell.get("learning_rate_vector"),
        label="cell.learning_rate_vector",
        length=len(parameter_names),
    )
    named_vector = [rates[name] for name in parameter_names]
    if not _same_float_vector(vector, named_vector):
        raise ValueError(
            f"Named learning-rate map does not match ordered vector in {cell_dir}."
        )
    units = probe["normalization_unit_by_weight"]
    expected_rates = {
        name: (
            expected_rho_conv / float(units[name])
            if name.startswith("ConvWeight_")
            else expected_rho_dense / float(units[name])
        )
        for name in WEIGHT_NAMES
    }
    expected_rates.update({name: 0.0 for name in BIAS_NAMES})
    if any(not _same_float(rates[name], expected_rates[name]) for name in parameter_names):
        raise ValueError(
            f"Learning-rate map does not match rho/probe-derived rates in {cell_dir}."
        )

    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError(f"Manifest scientific configuration is missing in {cell_dir}.")
    for label, value, expected in (
        ("manifest rho_conv", configuration.get("rho_conv"), expected_rho_conv),
        ("manifest rho_dense", configuration.get("rho_dense"), expected_rho_dense),
    ):
        if not _same_float(_finite_float(value, label=label), expected):
            raise ValueError(f"Manifest rho identity mismatch in {cell_dir}.")
    manifest_rates = _finite_vector(
        configuration.get("learning_rates"),
        label="manifest configuration learning_rates",
        length=len(parameter_names),
    )
    if not _same_float_vector(vector, manifest_rates):
        raise ValueError(
            f"Manifest learning-rate vector mismatch in {cell_dir}."
        )
    candidate_path = cell_dir / "source_config.json"
    candidate = _load_json(candidate_path)
    candidate_sha256 = sha256_file(candidate_path)
    expected_candidate = prepare_training_config(
        source_config,
        str(shard.surface["optimizer"]),
        vector,
        epochs=3,
        max_batches=None,
        max_validation_batches=None,
        split_seed=0,
        shuffle_seed=0,
        validation_batch_size=64,
    )
    if (
        candidate != expected_candidate
        or
        configuration.get("sha256") != candidate_sha256
        or configuration.get("resolved") != candidate
        or configuration.get("optimizer") != shard.surface["optimizer"]
        or manifest.get("git") != shard.resolved["code"]
    ):
        raise ValueError(
            f"Manifest resolved scientific configuration mismatch in {cell_dir}."
        )
    used_path = cell_dir / "config.used.json"
    if not used_path.is_file() or _load_json(used_path) != candidate:
        raise ValueError(
            f"Executed config.used.json does not match candidate source_config.json in {cell_dir}."
        )
    candidate_optimizer = candidate.get("optimizer")
    if (
        candidate.get("lr") != vector
        or not isinstance(candidate_optimizer, Mapping)
        or candidate_optimizer.get("name") != shard.surface["optimizer"]
        or candidate_optimizer.get("learning_rate") != vector
        or configuration.get("resolved", {}).get("lr") != vector
        or configuration.get("resolved", {}).get("optimizer", {}).get(
            "learning_rate"
        )
        != vector
    ):
        raise ValueError(
            f"Resolved optimizer learning-rate vector mismatch in {cell_dir}."
        )
    source_input = _manifest_input(manifest, "source_config", run_dir=cell_dir)
    probe_input = _manifest_input(manifest, "optimizer_probe", run_dir=cell_dir)
    if (
        source_input.get("sha256") != source_sha256
        or probe_input.get("sha256") != probe_file_sha256
    ):
        raise ValueError(
            f"Manifest source/probe input hash mismatch in {cell_dir}."
        )
    recovery = cell.get("recovery")
    inputs = manifest.get("inputs")
    interrupted_inputs = [
        item
        for item in inputs
        if isinstance(item, Mapping)
        and item.get("role") == "interrupted_attempt_receipt"
    ]
    if recovery is None:
        if configuration.get("recovery") is not None or interrupted_inputs:
            raise ValueError(
                f"Manifest unexpectedly declares recovery provenance in {cell_dir}."
            )
    else:
        if (
            not isinstance(recovery, Mapping)
            or configuration.get("recovery") != recovery
            or len(interrupted_inputs) != 1
        ):
            raise ValueError(
                f"Manifest does not bind current cell recovery provenance in {cell_dir}."
            )
        receipt_relative = recovery.get("receipt")
        attempt = recovery.get("attempt")
        path_relative = recovery.get("path")
        if (
            isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt < 1
            or recovery.get("restart_mode") != "clean_from_exact_initializer"
            or path_relative != f"recovery_attempts/attempt_{attempt:03d}"
            or receipt_relative != f"{path_relative}/recovery.json"
        ):
            raise ValueError(f"Cell recovery binding is noncanonical in {cell_dir}.")
        receipt_path = (cell_dir / str(receipt_relative)).resolve()
        attempt_path = (cell_dir / str(path_relative)).resolve()
        if receipt_path != attempt_path / "recovery.json" or not receipt_path.is_file():
            raise ValueError(f"Cell recovery receipt path is invalid in {cell_dir}.")
        recovery_receipt = _load_json(receipt_path)
        receipt_sha256 = sha256_file(receipt_path)
        if (
            recovery.get("receipt_sha256") != receipt_sha256
            or interrupted_inputs[0].get("path") != receipt_relative
            or interrupted_inputs[0].get("sha256") != receipt_sha256
            or recovery_receipt.get("schema_version")
            != "conv-rho-interrupted-attempt/v1"
            or recovery_receipt.get("attempt") != attempt
            or recovery_receipt.get("restart_mode")
            != "clean_from_exact_initializer"
        ):
            raise ValueError(
                f"Manifest interrupted-attempt receipt identity mismatch in {cell_dir}."
            )
    runtime = manifest.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("target") != shard.target
        or status.get("runtime") != runtime
    ):
        raise ValueError(f"Manifest/status runtime identity mismatch in {cell_dir}.")
    for key in ("target", "host"):
        expected = receipt_environment.get(key)
        if expected is not None and runtime.get(key) != expected:
            raise ValueError(
                f"Canonical cell runtime does not match its transport receipt "
                f"for {key} in {cell_dir}."
            )
    receipt_python = receipt_environment.get("python")
    runtime_python = runtime.get("python")
    if receipt_python is not None and (
        not isinstance(receipt_python, str)
        or not isinstance(runtime_python, str)
        or receipt_python.split()[0] != runtime_python
    ):
        raise ValueError(
            f"Canonical cell runtime does not match its transport receipt for "
            f"python in {cell_dir}."
        )
    command = manifest.get("command")
    if (
        not isinstance(command, Mapping)
        or command.get("surface_index") != shard.surface["index"]
        or command.get("cell_index") != expected_index
    ):
        raise ValueError(f"Manifest command cell identity mismatch in {cell_dir}.")
    return rates, vector


def _parameter_map(
    value: Any, *, label: str, required: bool
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        if required:
            raise ValueError(f"Expected {label} to be a parameter map.")
        return {}
    result: dict[str, float] = {}
    for name, raw in value.items():
        if name not in WEIGHT_NAMES:
            raise ValueError(f"Unexpected parameter {name!r} in {label}.")
        result[str(name)] = _finite_float(raw, label=f"{label}.{name}")
    if required and set(result) != set(WEIGHT_NAMES):
        raise ValueError(f"Expected all Conv3 weights in {label}.")
    return result


def _optional_parameter_map(
    value: Any, *, label: str, required: bool
) -> dict[str, float | None]:
    """Validate a diagnostic map whose individual measurements may be absent."""

    if not isinstance(value, Mapping):
        if required:
            raise ValueError(f"Expected {label} to be a parameter map.")
        return {}
    result: dict[str, float | None] = {}
    for name, raw in value.items():
        if name not in WEIGHT_NAMES:
            raise ValueError(f"Unexpected parameter {name!r} in {label}.")
        result[str(name)] = (
            None if raw is None else _finite_float(raw, label=f"{label}.{name}")
        )
    if required and set(result) != set(WEIGHT_NAMES):
        raise ValueError(f"Expected all Conv3 weights in {label}.")
    return result


def _indexed_final_weights(run_dir: Path, result: Mapping[str, Any]) -> Path:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"Result artifacts are missing in {run_dir}.")
    records = [
        row
        for row in artifacts
        if isinstance(row, Mapping)
        and row.get("kind") == "checkpoint"
        and Path(str(row.get("path", ""))).name == "weights_final.npz"
    ]
    if len(records) != 1:
        raise ValueError(f"Expected one indexed weights_final.npz in {run_dir}.")
    relative = Path(str(records[0]["path"]))
    if relative.is_absolute():
        raise ValueError("The indexed final-weight path must be relative.")
    checkpoint = (run_dir / relative).resolve()
    try:
        checkpoint.relative_to(run_dir.resolve())
    except ValueError as error:
        raise ValueError("The indexed final-weight path escapes its bundle.") from error
    if not checkpoint.is_file():
        raise ValueError(f"Missing indexed final weights: {checkpoint}.")
    return checkpoint


def _exact_weight_diagnostics(
    checkpoint: Path,
    reported_occupancy: Mapping[str, float],
    contract: StudyContract,
) -> dict[str, Any]:
    lower_count = upper_count = either_count = total = 0
    by_parameter: dict[str, float] = {}
    biases_zero = True
    with np.load(checkpoint, allow_pickle=False) as payload:
        for name in WEIGHT_NAMES:
            if name not in payload.files:
                raise ValueError(f"Final weights omit {name}: {checkpoint}.")
            values = np.asarray(payload[name])
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError(f"Final {name} is empty or non-finite: {checkpoint}.")
            outside = (values < contract.weight_min) | (values > contract.weight_max)
            if np.any(outside):
                raise ValueError(f"Final {name} leaves the bounded interval: {checkpoint}.")
            lower = int(np.count_nonzero(values <= contract.weight_min))
            upper = int(np.count_nonzero(values >= contract.weight_max))
            either = int(np.count_nonzero(
                (values <= contract.weight_min) | (values >= contract.weight_max)
            ))
            lower_count += lower
            upper_count += upper
            either_count += either
            total += int(values.size)
            occupancy = either / int(values.size)
            by_parameter[name] = occupancy
            if not math.isclose(
                occupancy,
                reported_occupancy[name],
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    f"Endpoint occupancy mismatch for {name} in {checkpoint}."
                )
        for name in BIAS_NAMES:
            if name not in payload.files or not np.all(np.asarray(payload[name]) == 0):
                biases_zero = False
    return {
        "exact_lower_bound_count": lower_count,
        "exact_upper_bound_count": upper_count,
        "exact_either_bound_count": either_count,
        "bounded_weight_count": total,
        "exact_bound_occupancy_fraction": either_count / total,
        "exact_bound_occupancy_percent": 100.0 * either_count / total,
        "exact_bound_occupancy_by_parameter": by_parameter,
        "final_biases_exact_zero": biases_zero,
    }


def _execution_ranks(contract: StudyContract) -> dict[int, int]:
    cells = []
    for conv_index in range(8):
        for dense_index in range(8):
            index = 8 * conv_index + dense_index
            cells.append((max(conv_index, dense_index), conv_index + dense_index, conv_index, dense_index, index))
    return {row[-1]: rank for rank, row in enumerate(sorted(cells))}


def _verify_zero_bias_record(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping) or value.get("all_exact_zero") is not True:
        raise ValueError(f"Missing exact zero-bias verification in {label}.")
    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(f"Missing checkpoint zero-bias records in {label}.")
    for checkpoint in checkpoints:
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("all_exact_zero") is not True
            or checkpoint.get("nonzero_bias_element_count") != 0
            or sorted(checkpoint.get("expected_bias_names", [])) != list(BIAS_NAMES)
        ):
            raise ValueError(f"Invalid checkpoint zero-bias record in {label}.")


def _read_cell(
    shard: Shard,
    contract: StudyContract,
    *,
    cell_dir: Path,
    expected_index: int,
    source_sha256: str,
    source_config: Mapping[str, Any],
    probe: Mapping[str, Any],
    probe_semantic_sha256: str,
    probe_file_sha256: str,
    execution_rank: int,
    receipt_environment: Mapping[str, Any],
) -> dict[str, Any]:
    cell_path = cell_dir / "cell.json"
    status_path = cell_dir / "status.json"
    manifest_path = cell_dir / "manifest.json"
    metrics_jsonl_path = cell_dir / "metrics.jsonl"
    cell = _load_json(cell_path)
    status = _load_json(status_path)
    manifest = _load_json(manifest_path)
    result_path = cell_dir / "result.json"
    result = _load_json(result_path) if result_path.is_file() else None
    result_sha256 = sha256_file(result_path) if result is not None else None
    validation_errors = validate_run(cell_dir)
    if validation_errors:
        raise ValueError(f"Invalid canonical bundle {cell_dir}: {validation_errors!r}.")
    if manifest.get("study_id") != contract.study_id or manifest.get("run_id") != cell_dir.name:
        raise ValueError(f"Manifest identity mismatch in {cell_dir}.")
    if status.get("study_id") != contract.study_id or status.get("run_id") != cell_dir.name:
        raise ValueError(f"Status identity mismatch in {cell_dir}.")
    outcome, nonfinite_kind = classify_cell_status(
        cell.get("status"),
        cell.get("safety_failure"),
        reporting_state=status.get("state"),
        result_present=result is not None,
    )
    if outcome == "numeric_complete":
        if status.get("result_sha256") != result_sha256:
            raise ValueError(
                f"Complete status does not bind result.json in {cell_dir}."
            )
    elif status.get("result_sha256") is not None:
        raise ValueError(
            f"Failed non-finite status unexpectedly binds result.json in {cell_dir}."
        )
    conv_index, dense_index = divmod(expected_index, 8)
    expected_rho_conv = contract.rho_conv[conv_index]
    expected_rho_dense = contract.rho_dense[dense_index]
    if (
        cell.get("index") != expected_index
        or cell.get("optimizer") != shard.surface["optimizer"]
        or cell.get("bias_policy") != "zero"
        or not math.isclose(_finite_float(cell.get("rho_conv"), label="rho_conv"), expected_rho_conv, rel_tol=1e-15)
        or not math.isclose(_finite_float(cell.get("rho_dense"), label="rho_dense"), expected_rho_dense, rel_tol=1e-15)
    ):
        raise ValueError(f"Cell identity/grid mismatch in {cell_dir}.")
    signature = cell.get("signature")
    if not isinstance(signature, Mapping):
        raise ValueError(f"Missing cell signature in {cell_dir}.")
    required_signature = {
        "optimizer": shard.surface["optimizer"],
        "evidence_class": "ordinary_mnist_exploratory",
        "bias_policy": "zero",
        "epochs": 3,
        "expected_candidate_steps": 10314,
        "max_batches": None,
        "max_validation_batches": None,
        "minimum_validation_accuracy": 0.0,
        "skip_canary": True,
        "restart_interrupted_cells": True,
        "canary_steps": 0,
        "source_config_sha256": source_sha256,
        "split_seed": 0,
        "shuffle_seed": 0,
    }
    for key, expected in required_signature.items():
        if signature.get(key) != expected:
            raise ValueError(f"Cell signature {key} mismatch in {cell_dir}.")
    if cell.get("evidence_class") != "ordinary_mnist_exploratory":
        raise ValueError(f"Cell evidence class mismatch in {cell_dir}.")
    safety_signature = signature.get("safety")
    if (
        not isinstance(safety_signature, Mapping)
        or safety_signature.get("rejections_enabled") is not False
        or safety_signature.get("bound_occupancy") != "report_only"
        or safety_signature.get("projection_efficiency") != "report_only"
        or safety_signature.get("bound_occupancy_increase_maximum") is not None
        or safety_signature.get("projection_efficiency_minimum") is not None
    ):
        raise ValueError(f"Cell signature does not retain report-only diagnostics: {cell_dir}.")
    learning_rates, vector = _verify_cell_scientific_identity(
        shard=shard,
        cell_dir=cell_dir,
        cell=cell,
        manifest=manifest,
        status=status,
        expected_index=expected_index,
        expected_rho_conv=expected_rho_conv,
        expected_rho_dense=expected_rho_dense,
        source_sha256=source_sha256,
        source_config=source_config,
        probe=probe,
        probe_semantic_sha256=probe_semantic_sha256,
        probe_file_sha256=probe_file_sha256,
        receipt_environment=receipt_environment,
    )
    if any(value < 0.0 for value in vector) or vector[-3:] != [0.0, 0.0, 0.0]:
        raise ValueError(f"Invalid zero-bias learning-rate vector in {cell_dir}.")
    diagnostics = _load_json(cell_dir / "safety_diagnostics.json")
    if diagnostics.get("schema_version") != "conv-rho-training-safety/v1":
        raise ValueError(f"Unexpected safety diagnostic schema in {cell_dir}.")
    report_only = diagnostics.get("report_only_diagnostics")
    gates = diagnostics.get("terminal_gates")
    if (
        report_only != {
            "bound_occupancy": True,
            "projection_efficiency": True,
            "used_for_rejection": False,
        }
        or not isinstance(gates, Mapping)
        or gates.get("scientific_rejections_enabled") is not False
        or gates.get("bound_occupancy_increase_maximum") is not None
        or gates.get("projection_efficiency_minimum") is not None
    ):
        raise ValueError(f"Safety diagnostics are not report-only in {cell_dir}.")
    required_diagnostics = outcome == "numeric_complete"
    occupancy = _parameter_map(
        diagnostics.get("final_bound_occupancy_by_parameter"),
        label="final_bound_occupancy_by_parameter",
        required=required_diagnostics,
    )
    for name, value in occupancy.items():
        _finite_fraction(value, label=f"endpoint occupancy {name}")
    projection = _optional_parameter_map(
        diagnostics.get("median_projection_efficiency_by_parameter"),
        label="median_projection_efficiency_by_parameter",
        required=required_diagnostics,
    )
    for name, value in projection.items():
        if value is not None and value < 0.0:
            raise ValueError(f"Negative projection efficiency for {name} in {cell_dir}.")
    proposed = _parameter_map(
        diagnostics.get("median_proposed_update_rms_by_parameter"),
        label="median_proposed_update_rms_by_parameter",
        required=required_diagnostics,
    )
    applied = _parameter_map(
        diagnostics.get("median_applied_update_rms_by_parameter"),
        label="median_applied_update_rms_by_parameter",
        required=required_diagnostics,
    )
    median_projection = _optional_finite_float(
        diagnostics.get("median_projection_efficiency"),
        label="median_projection_efficiency",
    )
    if median_projection is not None and median_projection < 0.0:
        raise ValueError(f"Negative overall projection efficiency in {cell_dir}.")
    metrics: dict[str, Any] | None = None
    exact = {
        "exact_lower_bound_count": None,
        "exact_upper_bound_count": None,
        "exact_either_bound_count": None,
        "bounded_weight_count": None,
        "exact_bound_occupancy_fraction": None,
        "exact_bound_occupancy_percent": None,
        "exact_bound_occupancy_by_parameter": {},
        "final_biases_exact_zero": None,
    }
    final_accuracy = final_loss = None
    if outcome == "numeric_complete":
        if cell.get("completed_steps") != 10314 or diagnostics.get("processed_steps") != 10314:
            raise ValueError(f"Numeric cell did not complete 10,314 steps: {cell_dir}.")
        if cell.get("selection_eligible") is not True:
            raise ValueError(f"Numeric exploratory cell is not marked eligible: {cell_dir}.")
        _verify_zero_bias_record(
            cell.get("zero_bias_checkpoint_verification"), label=str(cell_dir / "cell.json")
        )
        _verify_zero_bias_record(
            diagnostics.get("zero_bias_checkpoint_verification"),
            label=str(cell_dir / "safety_diagnostics.json"),
        )
        metrics = _load_json(cell_dir / "metrics.json")
        final_accuracy = _finite_fraction(
            metrics.get("final_validation_accuracy", metrics.get("final_test_accuracy")),
            label="final validation accuracy",
        )
        final_loss = _finite_float(
            metrics.get("final_validation_loss", metrics.get("final_test_loss")),
            label="final validation loss",
        )
        if final_loss < 0.0:
            raise ValueError(f"Negative validation loss in {cell_dir}.")
        if result is None:
            raise AssertionError("Numeric result disappeared after status classification.")
        terminal = result.get("terminal_metrics")
        validation = terminal.get("validation") if isinstance(terminal, Mapping) else None
        completion = result.get("completion")
        if (
            result.get("schema_version") != "experiment-run-result/v1"
            or not isinstance(validation, Mapping)
            or not math.isclose(_finite_float(validation.get("final_accuracy"), label="result accuracy"), final_accuracy, rel_tol=1e-12, abs_tol=1e-15)
            or not math.isclose(_finite_float(validation.get("final_loss"), label="result loss"), final_loss, rel_tol=1e-12, abs_tol=1e-15)
            or not isinstance(completion, Mapping)
            or completion.get("criteria_met") is not True
            or completion.get("rho_cell_complete") is not True
        ):
            raise ValueError(f"Canonical result metrics/completion mismatch in {cell_dir}.")
        final_weights = _indexed_final_weights(cell_dir, result)
        exact = _exact_weight_diagnostics(final_weights, occupancy, contract)
        if exact["final_biases_exact_zero"] is not True:
            raise ValueError(f"Final checkpoint has nonzero biases: {cell_dir}.")
    else:
        if diagnostics.get("safety_failure") != cell.get("safety_failure"):
            raise ValueError(f"Non-finite safety records disagree in {cell_dir}.")
    _verify_official_test_unread(manifest, result, metrics)
    failure = cell.get("safety_failure") if isinstance(cell.get("safety_failure"), Mapping) else {}
    diagnostic_mean = float(np.mean(list(occupancy.values()))) if occupancy else None
    diagnostic_max = max(occupancy.values()) if occupancy else None
    exact_occupancy_by_parameter_percent = {
        name: 100.0 * value
        for name, value in exact["exact_bound_occupancy_by_parameter"].items()
    }
    row = {
        "study_id": contract.study_id,
        "source_shard_root": str(shard.root),
        "surface_index": shard.surface["index"],
        "surface_id": shard.surface["surface_id"],
        "scheme": shard.surface["scheme"],
        "optimizer": shard.surface["optimizer"],
        "target": shard.target,
        "cell_index": expected_index,
        "execution_rank": execution_rank,
        "conv_axis_index": conv_index,
        "dense_axis_index": dense_index,
        "rho_conv": expected_rho_conv,
        "rho_dense": expected_rho_dense,
        "canonical_outcome": outcome,
        "raw_status": cell["status"],
        "nonfinite_kind": nonfinite_kind,
        "nonfinite_parameter": failure.get("parameter"),
        "nonfinite_confirmed_step": failure.get("confirmed_step"),
        "completed_steps": cell.get("completed_steps", diagnostics.get("processed_steps")),
        "final_validation_accuracy": final_accuracy,
        "final_validation_accuracy_percent": None if final_accuracy is None else 100.0 * final_accuracy,
        "final_validation_loss": final_loss,
        **exact,
        "exact_bound_occupancy_by_parameter_percent": (
            exact_occupancy_by_parameter_percent
        ),
        "exact_bound_occupancy_by_parameter_percent_json": _stable_json(
            exact_occupancy_by_parameter_percent
        ),
        "diagnostic_endpoint_bound_occupancy_mean": diagnostic_mean,
        "diagnostic_endpoint_bound_occupancy_max": diagnostic_max,
        "final_bound_occupancy_by_parameter": occupancy,
        "final_bound_occupancy_by_parameter_json": _stable_json(occupancy),
        "median_projection_efficiency": median_projection,
        "median_projection_efficiency_by_parameter": projection,
        "median_projection_efficiency_by_parameter_json": _stable_json(projection),
        "median_proposed_update_rms_by_parameter_json": _stable_json(proposed),
        "median_applied_update_rms_by_parameter_json": _stable_json(applied),
        "learning_rates_by_parameter": dict(learning_rates),
        "learning_rates_by_parameter_json": _stable_json(learning_rates),
        "manifest_json_sha256": sha256_file(manifest_path),
        "status_json_sha256": sha256_file(status_path),
        "metrics_jsonl_sha256": sha256_file(metrics_jsonl_path),
        "result_json_sha256": result_sha256,
        "initializer_checkpoint_sha256": contract.initializer_sha256,
        "probe_initial_parameter_sha256": probe["initial_parameter_sha256"],
        "bias_lr_zero": True,
        "official_test_read": False,
        "canonical_bundle_valid": True,
        "exploratory_noncanonical": True,
        "run_dir": str(cell_dir),
    }
    return row


def _read_surface(
    shard: Shard,
    contract: StudyContract,
    *,
    receipt_environment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _verify_initializer(shard, contract)
    source, source_sha256 = _verify_source_config(shard, contract)
    probe, probe_semantic_sha256, probe_file_sha256 = _verify_probe(
        shard, contract, source_sha256=source_sha256
    )
    surface_dir = shard.root / "surfaces" / str(shard.surface["surface_id"])
    summary = _load_json(surface_dir / "summary.json")
    required_summary = {
        "schema_version": SURFACE_SUMMARY_SCHEMA,
        "study_id": contract.study_id,
        "surface": shard.surface,
        "status": "complete",
        "complete": True,
        "expected_cells": EXPECTED_CELL_COUNT,
        "terminal_cells": EXPECTED_CELL_COUNT,
        "official_test_read": False,
    }
    for key, expected in required_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"Surface summary has invalid {key}: {surface_dir}.")
    cells_root = surface_dir / "rho" / "cells"
    directories = sorted(path for path in cells_root.iterdir() if path.is_dir()) if cells_root.is_dir() else []
    if len(directories) != EXPECTED_CELL_COUNT:
        raise ValueError(f"Expected 64 cell directories in {cells_root}, found {len(directories)}.")
    by_index: dict[int, Path] = {}
    for directory in directories:
        cell = _load_json(directory / "cell.json")
        index = cell.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 64:
            raise ValueError(f"Invalid cell index in {directory}.")
        if index in by_index:
            raise ValueError(f"Duplicate cell index {index} in {cells_root}.")
        by_index[index] = directory
    if set(by_index) != set(range(64)):
        raise ValueError(f"Cell-index coverage is incomplete in {cells_root}.")
    ranks = _execution_ranks(contract)
    rows = [
        _read_cell(
            shard,
            contract,
            cell_dir=by_index[index],
            expected_index=index,
            source_sha256=source_sha256,
            source_config=source,
            probe=probe,
            probe_semantic_sha256=probe_semantic_sha256,
            probe_file_sha256=probe_file_sha256,
            execution_rank=ranks[index],
            receipt_environment=receipt_environment,
        )
        for index in range(64)
    ]
    observed_counts = dict(Counter(row["raw_status"] for row in rows))
    if summary.get("status_counts") != observed_counts:
        raise ValueError(f"Surface status counts disagree with cell bundles: {surface_dir}.")
    candidates = summary.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 64:
        raise ValueError(f"Surface summary lacks 64 candidate records: {surface_dir}.")
    summary_by_index = {row.get("index"): row for row in candidates if isinstance(row, Mapping)}
    if set(summary_by_index) != set(range(64)):
        raise ValueError(f"Surface summary candidate indices are invalid: {surface_dir}.")
    for row in rows:
        candidate = summary_by_index[row["cell_index"]]
        if candidate.get("status") != row["raw_status"]:
            raise ValueError(f"Surface summary candidate status mismatch: {surface_dir}.")
    return rows


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _pearson(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    if len(values_x) != len(values_y):
        raise ValueError("Correlation vectors have different lengths.")
    if len(values_x) < 2:
        return None
    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Correlation vectors must be finite.")
    centered_x = x - np.mean(x)
    centered_y = y - np.mean(y)
    denominator = math.sqrt(
        float(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y))
    )
    if denominator == 0.0:
        return None
    value = float(np.dot(centered_x, centered_y) / denominator)
    return min(1.0, max(-1.0, value))


def _correlation_statistics(
    values_x: Sequence[float], values_y: Sequence[float]
) -> dict[str, Any]:
    """Return descriptive Pearson/Spearman coefficients, null if undefined."""

    if len(values_x) != len(values_y):
        raise ValueError("Correlation vectors have different lengths.")
    pearson = _pearson(values_x, values_y)
    spearman = (
        None
        if len(values_x) < 2
        else _pearson(_average_ranks(values_x), _average_ranks(values_y))
    )
    return {
        "sample_count": len(values_x),
        "pearson": pearson,
        "spearman": spearman,
    }


def _diagnostic_correlation(
    rows: Sequence[Mapping[str, Any]], diagnostic_field: str
) -> dict[str, Any]:
    paired = [
        (float(row[diagnostic_field]), float(row["final_validation_accuracy"]))
        for row in rows
        if row.get("canonical_outcome") == "numeric_complete"
        and row.get(diagnostic_field) is not None
        and row.get("final_validation_accuracy") is not None
    ]
    return _correlation_statistics(
        [diagnostic for diagnostic, _accuracy in paired],
        [accuracy for _diagnostic, accuracy in paired],
    )


def _parameter_diagnostic_correlations(
    rows: Sequence[Mapping[str, Any]], diagnostic_field: str
) -> dict[str, dict[str, Any]]:
    correlations: dict[str, dict[str, Any]] = {}
    for name in WEIGHT_NAMES:
        paired: list[tuple[float, float]] = []
        for row in rows:
            diagnostics = row.get(diagnostic_field)
            if (
                row.get("canonical_outcome") != "numeric_complete"
                or not isinstance(diagnostics, Mapping)
                or diagnostics.get(name) is None
                or row.get("final_validation_accuracy") is None
            ):
                continue
            paired.append(
                (
                    float(diagnostics[name]),
                    float(row["final_validation_accuracy"]),
                )
            )
        correlations[name] = _correlation_statistics(
            [diagnostic for diagnostic, _accuracy in paired],
            [accuracy for _diagnostic, accuracy in paired],
        )
    return correlations


def _range_evidence(
    best: Mapping[str, Any] | None,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe the four direct axial neighbors of the highest-accuracy cell.

    This is deliberately an accuracy-local diagnostic, not the rho selector
    from the governing protocol.  A point is called locally bracketed only
    when it is interior and all four direct axial neighbors are numeric with
    strictly lower final validation accuracy.
    """

    if best is None:
        return {
            "range_status": "no_numeric_observation",
            "range_status_statement": (
                "No numeric-complete cell exists, so local range geometry is "
                "unavailable."
            ),
            "locally_bracketed_by_accuracy": False,
            "open_directions": [],
            "nonnumeric_axial_directions": [],
            "improving_or_tied_directions": [],
            "axial_neighbors": {},
        }

    best_conv = int(best["conv_axis_index"])
    best_dense = int(best["dense_axis_index"])
    best_accuracy = _finite_float(
        best.get("final_validation_accuracy"), label="best validation accuracy"
    )
    by_coordinate: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        if "conv_axis_index" not in row or "dense_axis_index" not in row:
            continue
        coordinate = (int(row["conv_axis_index"]), int(row["dense_axis_index"]))
        if coordinate in by_coordinate:
            raise ValueError(f"Duplicate grid coordinate in surface rows: {coordinate!r}.")
        by_coordinate[coordinate] = row

    neighbors: dict[str, dict[str, Any]] = {}
    open_directions: list[str] = []
    nonnumeric_directions: list[str] = []
    improving_or_tied: list[str] = []
    for direction, axis, conv_delta, dense_delta in AXIAL_DIRECTIONS:
        conv_index = best_conv + conv_delta
        dense_index = best_dense + dense_delta
        if not (0 <= conv_index < 8 and 0 <= dense_index < 8):
            open_directions.append(direction)
            neighbors[direction] = {
                "direction": direction,
                "axis": axis,
                "grid_state": "outside_tested_range",
                "cell_index": None,
                "conv_axis_index": conv_index,
                "dense_axis_index": dense_index,
                "rho_conv": None,
                "rho_dense": None,
                "canonical_outcome": None,
                "final_validation_accuracy": None,
                "final_validation_accuracy_percent": None,
                "accuracy_relation_to_best": "untested",
            }
            continue

        neighbor = by_coordinate.get((conv_index, dense_index))
        if neighbor is None:
            raise ValueError(
                "Cannot classify the highest-accuracy range geometry because "
                f"axial cell ({conv_index},{dense_index}) is missing."
            )
        outcome = str(neighbor.get("canonical_outcome"))
        neighbor_accuracy = neighbor.get("final_validation_accuracy")
        if outcome != "numeric_complete" or neighbor_accuracy is None:
            relation = "nonnumeric"
            nonnumeric_directions.append(direction)
            numeric_accuracy = None
        else:
            numeric_accuracy = _finite_float(
                neighbor_accuracy, label=f"{direction} validation accuracy"
            )
            if numeric_accuracy < best_accuracy:
                relation = "strictly_lower"
            elif numeric_accuracy > best_accuracy:
                relation = "improving"
                improving_or_tied.append(direction)
            else:
                relation = "tied"
                improving_or_tied.append(direction)
        neighbors[direction] = {
            "direction": direction,
            "axis": axis,
            "grid_state": "observed",
            "cell_index": int(neighbor["cell_index"]),
            "conv_axis_index": conv_index,
            "dense_axis_index": dense_index,
            "rho_conv": _finite_float(neighbor.get("rho_conv"), label="neighbor rho_conv"),
            "rho_dense": _finite_float(neighbor.get("rho_dense"), label="neighbor rho_dense"),
            "canonical_outcome": outcome,
            "final_validation_accuracy": numeric_accuracy,
            "final_validation_accuracy_percent": (
                None if numeric_accuracy is None else 100.0 * numeric_accuracy
            ),
            "accuracy_relation_to_best": relation,
        }

    if open_directions:
        status = "open_boundary"
        statement = (
            "The highest-accuracy cell is on the tested boundary; untested "
            f"direction(s): {', '.join(open_directions)}."
        )
    elif nonnumeric_directions:
        status = "unresolved_nonnumeric_axial_neighbor"
        statement = (
            "The highest-accuracy cell is interior, but local bracketing is "
            "unresolved because direct axial neighbor(s) are nonnumeric: "
            f"{', '.join(nonnumeric_directions)}."
        )
    elif not improving_or_tied:
        status = "locally_bracketed_by_accuracy"
        statement = (
            "The highest-accuracy cell is interior and all four direct axial "
            "neighbors are numeric with strictly lower accuracy."
        )
    else:
        status = "not_locally_bracketed"
        statement = (
            "The highest-accuracy cell is interior but has improving or tied "
            f"direct axial direction(s): {', '.join(improving_or_tied)}."
        )
    return {
        "range_status": status,
        "range_status_statement": statement,
        "locally_bracketed_by_accuracy": status == "locally_bracketed_by_accuracy",
        "open_directions": open_directions,
        "nonnumeric_axial_directions": nonnumeric_directions,
        "improving_or_tied_directions": improving_or_tied,
        "axial_neighbors": neighbors,
    }


def _co_maximum_range_evidence(
    *,
    selected_best: Mapping[str, Any] | None,
    co_maxima: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if selected_best is None:
        evidence = _range_evidence(None, rows)
        return {
            **evidence,
            "selected_best_range_evidence": evidence,
            "maximum_accuracy_co_maxima": [],
        }
    details = []
    for maximum in sorted(co_maxima, key=lambda row: int(row["cell_index"])):
        evidence = _range_evidence(maximum, rows)
        details.append(
            {
                "cell_index": int(maximum["cell_index"]),
                "conv_axis_index": int(maximum["conv_axis_index"]),
                "dense_axis_index": int(maximum["dense_axis_index"]),
                "rho_conv": float(maximum["rho_conv"]),
                "rho_dense": float(maximum["rho_dense"]),
                "final_validation_accuracy": float(
                    maximum["final_validation_accuracy"]
                ),
                "final_validation_accuracy_percent": float(
                    maximum["final_validation_accuracy_percent"]
                ),
                "selected_by_loss_rank_tiebreak": (
                    int(maximum["cell_index"])
                    == int(selected_best["cell_index"])
                ),
                "range_evidence": evidence,
            }
        )
    statuses = [row["range_evidence"]["range_status"] for row in details]
    if "open_boundary" in statuses:
        status = "open_boundary"
        statement = (
            f"At least one of {len(details)} maximum-accuracy co-maxima lies on "
            "the tested boundary, so the maximum-accuracy set is open."
        )
    elif "unresolved_nonnumeric_axial_neighbor" in statuses:
        status = "unresolved_nonnumeric_axial_neighbor"
        statement = (
            f"At least one of {len(details)} maximum-accuracy co-maxima has a "
            "nonnumeric direct axial neighbor, so conservative local bracketing "
            "is unresolved."
        )
    elif statuses and all(value == "locally_bracketed_by_accuracy" for value in statuses):
        status = "locally_bracketed_by_accuracy"
        statement = (
            f"All {len(details)} maximum-accuracy co-maxima are interior and "
            "have four numeric direct axial neighbors with strictly lower accuracy."
        )
    else:
        status = "not_locally_bracketed"
        statement = (
            f"At least one of {len(details)} maximum-accuracy co-maxima has a "
            "direct axial neighbor with tied or improving accuracy."
        )
    selected = next(
        row["range_evidence"]
        for row in details
        if row["selected_by_loss_rank_tiebreak"]
    )

    def union(field: str) -> list[str]:
        return sorted(
            {
                value
                for row in details
                for value in row["range_evidence"][field]
            }
        )

    return {
        "range_status": status,
        "range_status_statement": statement,
        "locally_bracketed_by_accuracy": (
            status == "locally_bracketed_by_accuracy"
        ),
        "open_directions": union("open_directions"),
        "nonnumeric_axial_directions": union("nonnumeric_axial_directions"),
        "improving_or_tied_directions": union(
            "improving_or_tied_directions"
        ),
        "axial_neighbors": selected["axial_neighbors"],
        "selected_best_range_evidence": selected,
        "maximum_accuracy_co_maxima": details,
    }


def _surface_summary(surface: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    targets = {str(row["target"]) for row in rows}
    if len(targets) != 1:
        raise ValueError(
            f"Surface {surface['surface_id']!r} spans multiple targets: "
            f"{sorted(targets)!r}."
        )
    target = next(iter(targets))
    numeric = [row for row in rows if row["canonical_outcome"] == "numeric_complete"]
    best = max(
        numeric,
        key=lambda row: (
            row["final_validation_accuracy"],
            -row["final_validation_loss"],
            -row["execution_rank"],
        ),
    ) if numeric else None
    maximum_accuracy = (
        None
        if best is None
        else float(best["final_validation_accuracy"])
    )
    co_maxima = (
        []
        if maximum_accuracy is None
        else [
            row
            for row in numeric
            if float(row["final_validation_accuracy"]) == maximum_accuracy
        ]
    )
    best_weight_learning_rates = None if best is None else {
        name: float(best["learning_rates_by_parameter"][name])
        for name in WEIGHT_NAMES
    }
    best_bias_learning_rates = None if best is None else {
        name: float(best["learning_rates_by_parameter"][name])
        for name in BIAS_NAMES
    }
    if best_bias_learning_rates is not None and any(
        value != 0.0 for value in best_bias_learning_rates.values()
    ):
        raise ValueError(
            f"Highest-accuracy cell for {surface['surface_id']!r} has a nonzero "
            "bias learning rate."
        )
    occupancy_correlation = _diagnostic_correlation(
        numeric, "exact_bound_occupancy_percent"
    )
    occupancy_by_parameter_correlations = _parameter_diagnostic_correlations(
        numeric, "exact_bound_occupancy_by_parameter_percent"
    )
    projection_correlation = _diagnostic_correlation(
        numeric, "median_projection_efficiency"
    )
    range_evidence = _co_maximum_range_evidence(
        selected_best=best, co_maxima=co_maxima, rows=rows
    )
    summary = {
        "surface_index": surface["index"],
        "surface_id": surface["surface_id"],
        "scheme": surface["scheme"],
        "optimizer": surface["optimizer"],
        "target": target,
        "numeric_cell_count": len(numeric),
        "nonfinite_cell_count": len(rows) - len(numeric),
        "best_observed_cell_index": None if best is None else best["cell_index"],
        "best_observed_rho_conv": None if best is None else best["rho_conv"],
        "best_observed_rho_dense": None if best is None else best["rho_dense"],
        "best_observed_lr_conv_weight_0": (
            None
            if best_weight_learning_rates is None
            else best_weight_learning_rates["ConvWeight_0"]
        ),
        "best_observed_lr_conv_weight_1": (
            None
            if best_weight_learning_rates is None
            else best_weight_learning_rates["ConvWeight_1"]
        ),
        "best_observed_lr_conv_weight_2": (
            None
            if best_weight_learning_rates is None
            else best_weight_learning_rates["ConvWeight_2"]
        ),
        "best_observed_lr_dense_weight_0": (
            None
            if best_weight_learning_rates is None
            else best_weight_learning_rates["DenseWeight_0"]
        ),
        "best_observed_weight_learning_rates": best_weight_learning_rates,
        "best_observed_weight_learning_rates_json": _stable_json(
            best_weight_learning_rates
        ),
        "best_observed_bias_learning_rates": best_bias_learning_rates,
        "best_observed_bias_learning_rates_json": _stable_json(
            best_bias_learning_rates
        ),
        "best_observed_bias_lrs_zero": (
            None if best_bias_learning_rates is None else True
        ),
        "best_observed_conv_axis_index": (
            None if best is None else best["conv_axis_index"]
        ),
        "best_observed_dense_axis_index": (
            None if best is None else best["dense_axis_index"]
        ),
        "maximum_accuracy_co_maxima_count": len(co_maxima),
        "maximum_accuracy_co_maxima": range_evidence[
            "maximum_accuracy_co_maxima"
        ],
        "maximum_accuracy_co_maxima_json": _stable_json(
            range_evidence["maximum_accuracy_co_maxima"]
        ),
        "best_observed_validation_accuracy": None if best is None else best["final_validation_accuracy"],
        "best_observed_validation_accuracy_percent": None if best is None else best["final_validation_accuracy_percent"],
        "best_observed_validation_loss": None if best is None else best["final_validation_loss"],
        "best_observed_exact_bound_occupancy_percent": None if best is None else best["exact_bound_occupancy_percent"],
        "best_observed_exact_bound_occupancy_by_parameter_percent": (
            {} if best is None else best["exact_bound_occupancy_by_parameter_percent"]
        ),
        "best_observed_exact_bound_occupancy_by_parameter_percent_json": (
            _stable_json(
                {}
                if best is None
                else best["exact_bound_occupancy_by_parameter_percent"]
            )
        ),
        "best_observed_median_projection_efficiency": None if best is None else best["median_projection_efficiency"],
        "median_validation_accuracy": _median([row["final_validation_accuracy"] for row in numeric]),
        "median_validation_loss": _median([row["final_validation_loss"] for row in numeric]),
        "median_exact_bound_occupancy_percent": _median([row["exact_bound_occupancy_percent"] for row in numeric]),
        "median_projection_efficiency": _median([row["median_projection_efficiency"] for row in numeric if row["median_projection_efficiency"] is not None]),
        "accuracy_vs_exact_bound_occupancy_sample_count": occupancy_correlation[
            "sample_count"
        ],
        "accuracy_vs_exact_bound_occupancy_pearson": occupancy_correlation[
            "pearson"
        ],
        "accuracy_vs_exact_bound_occupancy_spearman": occupancy_correlation[
            "spearman"
        ],
        "accuracy_vs_exact_bound_occupancy_by_parameter_json": _stable_json(
            occupancy_by_parameter_correlations
        ),
        "accuracy_vs_median_projection_efficiency_sample_count": projection_correlation[
            "sample_count"
        ],
        "accuracy_vs_median_projection_efficiency_pearson": projection_correlation[
            "pearson"
        ],
        "accuracy_vs_median_projection_efficiency_spearman": projection_correlation[
            "spearman"
        ],
        **range_evidence,
        "open_directions_json": _stable_json(range_evidence["open_directions"]),
        "nonnumeric_axial_directions_json": _stable_json(
            range_evidence["nonnumeric_axial_directions"]
        ),
        "improving_or_tied_directions_json": _stable_json(
            range_evidence["improving_or_tied_directions"]
        ),
        "axial_neighbors_json": _stable_json(range_evidence["axial_neighbors"]),
        "descriptive_correlations": {
            "accuracy_vs_exact_bound_occupancy": occupancy_correlation,
            "accuracy_vs_exact_bound_occupancy_by_parameter": (
                occupancy_by_parameter_correlations
            ),
            "accuracy_vs_median_projection_efficiency": projection_correlation,
            "noncausal": True,
            "caveat": CORRELATION_CAVEAT,
        },
        "correlations_noncausal": True,
        "exploratory_noncanonical": True,
    }
    summary["highest_observed_not_selected"] = None if best is None else {
        key: best[key]
        for key in (
            "cell_index",
            "rho_conv",
            "rho_dense",
            "final_validation_accuracy",
            "final_validation_loss",
            "exact_bound_occupancy_percent",
            "median_projection_efficiency",
        )
    }
    return summary


def _infer_study_evidence_root(
    *, study_root: Path | None, shards: Sequence[Shard]
) -> Path | None:
    if study_root is not None:
        return Path(study_root).expanduser().resolve()
    candidates = {
        shard.root.parent.parent.resolve()
        for shard in shards
        if shard.root.parent.name == "shards"
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _file_hash_record(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    return {
        "path": str(source),
        "sha256": sha256_file(source),
    }


def _optional_bundle_hashes(run_dir: Path) -> dict[str, str | None]:
    records: dict[str, str | None] = {}
    for name in ("manifest.json", "status.json", "metrics.jsonl"):
        path = run_dir / name
        records[name.replace(".", "_") + "_sha256"] = (
            sha256_file(path) if path.is_file() else None
        )
    return records


def _canonical_run_inventory(
    rows: Sequence[Mapping[str, Any]], contract: StudyContract
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    expected = {
        (str(surface["surface_id"]), index)
        for surface in contract.surfaces
        for index in range(EXPECTED_CELL_COUNT)
    }
    observed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["surface_id"]), int(row["cell_index"]))
        if key in observed:
            raise ValueError(f"Terminal inventory contains duplicate cell {key!r}.")
        observed[key] = row
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        unexpected = sorted(set(observed) - expected)
        raise ValueError(
            "Terminal inventory cannot reconcile the 256 declared cells: "
            f"missing={missing!r}, unexpected={unexpected!r}."
        )

    included: list[dict[str, Any]] = []
    numeric_exclusions: list[dict[str, Any]] = []
    for key in sorted(observed):
        row = observed[key]
        run_dir = Path(str(row["run_dir"])).expanduser().resolve()
        numeric = row["canonical_outcome"] == "numeric_complete"
        result_sha256 = row.get("result_json_sha256")
        if numeric != (result_sha256 is not None):
            raise ValueError(
                "Terminal inventory result.json reconciliation failed for "
                f"{key!r}: outcome={row['canonical_outcome']!r}, "
                f"result_sha256={result_sha256!r}."
            )
        record = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "surface_id": key[0],
            "surface_index": int(row["surface_index"]),
            "scheme": str(row["scheme"]),
            "optimizer": str(row["optimizer"]),
            "target": str(row["target"]),
            "cell_index": key[1],
            "canonical_outcome": str(row["canonical_outcome"]),
            "coverage_included": True,
            "scientific_authority": "canonical_terminal_cell",
            "numeric_summary_included": numeric,
            "manifest_json_sha256": str(row["manifest_json_sha256"]),
            "status_json_sha256": str(row["status_json_sha256"]),
            "metrics_jsonl_sha256": str(row["metrics_jsonl_sha256"]),
            "result_json_path": (
                str(run_dir / "result.json") if numeric else None
            ),
            "result_json_sha256": result_sha256,
        }
        included.append(record)
        if not numeric:
            numeric_exclusions.append(
                {
                    **record,
                    "numeric_summary_exclusion_reason": (
                        "Retained terminal non-finite scientific outcome; it "
                        "counts toward declared coverage but has no finite "
                        "accuracy or result.json."
                    ),
                }
            )

    numeric_count = sum(row["numeric_summary_included"] for row in included)
    result_count = sum(row["result_json_sha256"] is not None for row in included)
    if (
        len(included) != 256
        or numeric_count + len(numeric_exclusions) != 256
        or result_count != numeric_count
    ):
        raise ValueError(
            "Terminal inventory cannot reconcile 256 declared cells with "
            "numeric-complete result.json files and retained non-finite outcomes."
        )
    reconciliation = {
        "declared_cell_count": 256,
        "canonical_terminal_cell_count": len(included),
        "numeric_summary_included_count": numeric_count,
        "numeric_summary_excluded_terminal_count": len(numeric_exclusions),
        "source_result_json_count": result_count,
        "result_json_expected_absent_count": len(numeric_exclusions),
        "unique_surface_cell_keys": len(observed),
        "reconciled": True,
    }
    return included, numeric_exclusions, reconciliation


def _archived_attempt_inventory(
    included_runs: Sequence[Mapping[str, Any]],
    *,
    evidence_root: Path | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    runs_by_dir = {
        Path(str(run["run_dir"])).resolve(): run for run in included_runs
    }
    attempt_roots = {
        run_dir / "recovery_attempts"
        for run_dir in runs_by_dir
        if (run_dir / "recovery_attempts").exists()
    }
    if evidence_root is not None:
        attempt_roots.update(
            path.resolve() for path in evidence_root.rglob("recovery_attempts")
        )
    for attempts_root in sorted(attempt_roots):
        if not attempts_root.is_dir():
            raise ValueError(f"Recovery-attempt path is not a directory: {attempts_root}.")
        run_dir = attempts_root.parent.resolve()
        run = runs_by_dir.get(run_dir)
        entries = sorted(attempts_root.iterdir())
        for attempt_dir in entries:
            name = attempt_dir.name
            if (
                not attempt_dir.is_dir()
                or not name.startswith("attempt_")
                or len(name) != len("attempt_001")
                or not name.removeprefix("attempt_").isdigit()
            ):
                raise ValueError(
                    f"Misnamed or non-directory recovery attempt entry: {attempt_dir}."
                )
            receipt_path = attempt_dir / "recovery.json"
            if not receipt_path.is_file():
                raise ValueError(
                    f"Recovery attempt directory is missing recovery.json: {attempt_dir}."
                )
            recovery_like = sorted(attempt_dir.glob("recovery*.json"))
            if recovery_like != [receipt_path]:
                raise ValueError(
                    f"Recovery attempt contains a misnamed or duplicate receipt: {attempt_dir}."
                )
            receipt = _load_json(receipt_path)
            attempt_index = int(name.removeprefix("attempt_"))
            if (
                receipt.get("schema_version") != "conv-rho-interrupted-attempt/v1"
                or receipt.get("attempt") != attempt_index
                or receipt.get("restart_mode") != "clean_from_exact_initializer"
                or not isinstance(receipt.get("archived_entries"), list)
            ):
                raise ValueError(
                    f"Malformed archived interrupted-attempt receipt {receipt_path}."
                )
            result_paths = sorted(attempt_dir.rglob("result.json"))
            if result_paths:
                raise ValueError(
                    "An archived interrupted attempt unexpectedly contains "
                    f"result.json: {result_paths!r}."
                )
            for name in receipt["archived_entries"]:
                if not isinstance(name, str) or not (attempt_dir / name).exists():
                    raise ValueError(
                        f"Archived attempt receipt does not bind entry {name!r}: "
                        f"{receipt_path}."
                    )
            records.append(
                {
                    "evidence_kind": "archived_interrupted_attempt",
                    "run_id": run["run_id"] if run is not None else run_dir.name,
                    "surface_id": (
                        run["surface_id"] if run is not None else None
                    ),
                    "cell_index": run["cell_index"] if run is not None else None,
                    "path": str(attempt_dir.resolve()),
                    "authoritative": False,
                    "coverage_included": False,
                    "numeric_summary_included": False,
                    "excluded_reason": (
                        "Interrupted or failed predecessor preserved before a "
                        "clean restart from the exact initializer."
                    ),
                    "recovery_receipt": _file_hash_record(receipt_path),
                    "previous_cell_status": receipt.get("previous_cell_status"),
                    "previous_reporting_state": receipt.get(
                        "previous_reporting_state"
                    ),
                    "source_result_jsons": [],
                    **_optional_bundle_hashes(attempt_dir),
                }
            )
    for run_dir in sorted(runs_by_dir):
        cell = _load_json(run_dir / "cell.json")
        recovery = cell.get("recovery")
        attempts_root = run_dir / "recovery_attempts"
        attempt_dirs = (
            sorted(path.resolve() for path in attempts_root.iterdir())
            if attempts_root.is_dir()
            else []
        )
        if attempt_dirs and recovery is None:
            raise ValueError(
                f"Current cell omits recovery binding for archived attempt(s) in {run_dir}."
            )
        if recovery is not None:
            if not isinstance(recovery, Mapping):
                raise ValueError(f"Malformed current recovery binding in {run_dir}.")
            receipt_relative = recovery.get("receipt")
            path_relative = recovery.get("path")
            attempt = recovery.get("attempt")
            if (
                not isinstance(receipt_relative, str)
                or not isinstance(path_relative, str)
                or isinstance(attempt, bool)
                or not isinstance(attempt, int)
                or attempt < 1
                or recovery.get("restart_mode") != "clean_from_exact_initializer"
                or path_relative != f"recovery_attempts/attempt_{attempt:03d}"
                or receipt_relative != f"{path_relative}/recovery.json"
            ):
                raise ValueError(f"Malformed current recovery paths in {run_dir}.")
            receipt_path = (run_dir / receipt_relative).resolve()
            attempt_path = (run_dir / path_relative).resolve()
            receipt_payload = _load_json(receipt_path) if receipt_path.is_file() else {}
            if (
                not receipt_path.is_file()
                or receipt_path != attempt_path / "recovery.json"
                or attempt_path.parent != run_dir / "recovery_attempts"
                or not attempt_dirs
                or attempt_path != attempt_dirs[-1]
                or recovery.get("receipt_sha256") != sha256_file(receipt_path)
                or receipt_payload.get("schema_version")
                != "conv-rho-interrupted-attempt/v1"
                or receipt_payload.get("attempt") != attempt
                or receipt_payload.get("restart_mode")
                != "clean_from_exact_initializer"
            ):
                raise ValueError(f"Current cell does not bind its recovery receipt in {run_dir}.")
    return records


def _supporting_result_inventory(
    evidence_root: Path | None,
    included_runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Inventory every noncanonical reporting bundle under the study root.

    Operational failures often stop after writing ``manifest.json`` and
    ``status.json``.  Looking only for ``result.json`` therefore hides exactly
    the attempts most useful for diagnosing a target gate.  Discover bundle
    directories from the union of all three reporting files, and then add any
    standalone result summaries that are not themselves reporting bundles.
    """

    if evidence_root is None:
        return []
    authoritative_dirs = {
        Path(str(row["run_dir"])).resolve() for row in included_runs
    }

    def classify(path: Path) -> tuple[str, str]:
        relative_parts = path.relative_to(evidence_root).parts
        # A gate path normally also contains ``smoke``.  Gate provenance is
        # more specific and must win over the generic smoke classification.
        if "gates" in relative_parts:
            return (
                "target_gate",
                "Target-gate evidence is operational and excluded from the "
                "256-cell scientific analysis.",
            )
        if "smoke" in relative_parts:
            return (
                "smoke",
                "Smoke evidence is operational and excluded from the 256-cell "
                "scientific analysis.",
            )
        return (
            "noncanonical_result_bundle",
            "Reporting evidence lies outside the four receipt-bound canonical "
            "shards and is excluded as non-authoritative execution evidence.",
        )

    records: list[dict[str, Any]] = []
    bundle_dirs = {
        path.parent.resolve()
        for name in ("manifest.json", "status.json")
        for path in evidence_root.rglob(name)
    }
    for run_dir in sorted(bundle_dirs):
        if run_dir in authoritative_dirs:
            continue
        relative_parts = run_dir.relative_to(evidence_root).parts
        if "recovery_attempts" in relative_parts:
            # Recovery contents are exhaustively checked by
            # _archived_attempt_inventory; accepting a second reporting-bundle
            # route here would permit unreceipted attempts to escape that gate.
            continue
        kind, reason = classify(run_dir)
        manifest_path = run_dir / "manifest.json"
        status_path = run_dir / "status.json"
        result_path = run_dir / "result.json"
        manifest = _load_json(manifest_path) if manifest_path.is_file() else None
        status = _load_json(status_path) if status_path.is_file() else None
        result = _load_json(result_path) if result_path.is_file() else None
        identities = [
            payload.get("run_id")
            for payload in (manifest, status, result)
            if isinstance(payload, Mapping) and payload.get("run_id") is not None
        ]
        if len(set(identities)) > 1:
            raise ValueError(
                f"Operational bundle has inconsistent run identities: {run_dir}."
            )
        validation_errors = (
            validate_run(run_dir)
            if manifest_path.is_file() and status_path.is_file()
            else ["incomplete reporting bundle: manifest.json and status.json are both required"]
        )
        records.append(
            {
                "evidence_kind": kind,
                "run_id": identities[0] if identities else run_dir.name,
                "path": str(run_dir),
                "authoritative": False,
                "coverage_included": False,
                "numeric_summary_included": False,
                "excluded_reason": reason,
                "reporting_files_present": [
                    name
                    for name in ("manifest.json", "status.json", "result.json")
                    if (run_dir / name).is_file()
                ],
                "reporting_state": (
                    status.get("state") if isinstance(status, Mapping) else None
                ),
                "bundle_validation_errors": validation_errors,
                "source_result_jsons": (
                    [_file_hash_record(result_path)]
                    if result_path.is_file()
                    else []
                ),
                **_optional_bundle_hashes(run_dir),
            }
        )

    bundled_result_paths = {
        (run_dir / "result.json").resolve() for run_dir in bundle_dirs
    }
    authoritative_results = {
        (run_dir / "result.json").resolve() for run_dir in authoritative_dirs
    }
    for result_path in sorted(evidence_root.rglob("result.json")):
        resolved = result_path.resolve()
        if resolved in bundled_result_paths or resolved in authoritative_results:
            continue
        relative_parts = result_path.relative_to(evidence_root).parts
        if "recovery_attempts" in relative_parts:
            raise ValueError(
                "Archived interrupted execution evidence contains result.json: "
                f"{result_path}."
            )
        kind, reason = classify(result_path.parent.resolve())
        result = _load_json(result_path)
        records.append(
            {
                "evidence_kind": kind,
                "run_id": result.get("run_id", result_path.parent.name),
                "path": str(result_path.parent.resolve()),
                "authoritative": False,
                "coverage_included": False,
                "numeric_summary_included": False,
                "excluded_reason": reason,
                "reporting_files_present": ["result.json"],
                "reporting_state": None,
                "bundle_validation_errors": [
                    "standalone result.json without manifest.json/status.json bundle"
                ],
                "source_result_jsons": [_file_hash_record(result_path)],
                **_optional_bundle_hashes(result_path.parent),
            }
        )
    records.sort(key=lambda row: (str(row["path"]), str(row["evidence_kind"])))
    return records


def _documented_operational_attempt_inventory(
    evidence_root: Path | None,
    existing_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain provenance-documented attempts that never reached a run bundle."""

    if evidence_root is None:
        return []
    receipt_path = evidence_root / "provenance" / "local_fallback_submission.json"
    if not receipt_path.is_file():
        return []
    receipt = _load_json(receipt_path)
    if receipt.get("study_id") != CANONICAL_STUDY_ID:
        return []
    existing_paths = {str(record.get("path")) for record in existing_records}
    target_gates = receipt.get("target_gates")
    if not isinstance(target_gates, Mapping):
        return []
    documented: list[tuple[str, str, Mapping[str, Any]]] = []
    for target, payload in target_gates.items():
        if not isinstance(payload, Mapping):
            continue
        if isinstance(payload.get("path"), str):
            documented.append((str(target), "passing_attempt", payload))
        for label in ("failed_attempt", "passing_attempt"):
            attempt = payload.get(label)
            if isinstance(attempt, Mapping) and isinstance(attempt.get("path"), str):
                documented.append((str(target), label, attempt))
    records: list[dict[str, Any]] = []
    for target, label, attempt in documented:
        raw_path = str(attempt["path"])
        documented_path = Path(raw_path).expanduser()
        # Passing-attempt provenance points at result.json; inventory the gate
        # root rather than inventing a missing result-bundle path locally.
        gate_name = next(
            (
                part
                for part in reversed(documented_path.parts)
                if part.endswith("target-gate") or "target-gate-" in part
            ),
            documented_path.parent.name,
        )
        collected_path = (evidence_root / "gates" / gate_name).resolve()
        if any(
            Path(existing).resolve() == collected_path
            or collected_path in Path(existing).resolve().parents
            for existing in existing_paths
            if existing not in {"None", ""}
        ):
            continue
        collected_files = (
            [
                _file_hash_record(path)
                for path in sorted(collected_path.rglob("*"))
                if path.is_file()
            ]
            if collected_path.is_dir()
            else []
        )
        passed = label == "passing_attempt"
        records.append(
            {
                "evidence_kind": "target_gate",
                "run_id": gate_name,
                "target": target,
                "path": str(collected_path),
                "documented_remote_path": raw_path,
                "authoritative": False,
                "coverage_included": False,
                "numeric_summary_included": False,
                "excluded_reason": (
                    "Target gate documented by local_fallback_submission.json; "
                    "it is operational evidence outside the scientific grid."
                ),
                "failure": attempt.get("failure"),
                "diagnosis": attempt.get("diagnosis"),
                "documented_status": attempt.get("status"),
                "reporting_files_present": [],
                "reporting_state": (
                    "passed_documented_remote_gate"
                    if passed
                    else "failed_before_reporting_bundle"
                ),
                "bundle_validation_errors": (
                    ["remote gate bundle was not collected locally"]
                    if passed
                    else ["no reporting bundle was produced"]
                ),
                "source_result_jsons": [],
                "collected_files": collected_files,
                "provenance_sources": [_file_hash_record(receipt_path)],
                **_optional_bundle_hashes(collected_path),
            }
        )
    return records


def _provenance_sources(
    provenance_root: Path, names: Sequence[str]
) -> list[dict[str, Any]]:
    return [
        _file_hash_record(provenance_root / name)
        for name in names
        if (provenance_root / name).is_file()
    ]


def _canceled_job_inventory(
    evidence_root: Path | None, contract: StudyContract
) -> list[dict[str, Any]]:
    if contract.study_id != CANONICAL_STUDY_ID:
        return []
    if evidence_root is None:
        raise ValueError(
            "The canonical terminal inventory cannot locate the study root "
            "needed to reconcile canceled Jean Zay jobs 844032 and 844396."
        )
    provenance = evidence_root / "provenance"
    provenance_specs = {
        "launch_plan.json": (
            "perfectdiode-conv3-exploratory-lr-ladder-launch-plan/v1"
        ),
        "jean_zay_submission.json": (
            "perfectdiode-conv3-exploratory-lr-ladder-submission/v1"
        ),
        "replacement_plan.json": (
            "perfectdiode-conv3-exploratory-lr-ladder-replacement-plan/v1"
        ),
        "recovery_submission.json": (
            "perfectdiode-conv3-exploratory-lr-ladder-recovery-submission/v1"
        ),
        "local_fallback_plan.json": (
            "perfectdiode-conv3-exploratory-lr-ladder-local-fallback-plan/v1"
        ),
        "local_fallback_submission.json": (
            "perfectdiode-conv3-exploratory-lr-ladder-local-fallback-submission/v1"
        ),
        "jean_zay_terminal_accounting.json": (
            "perfectdiode-conv3-exploratory-lr-ladder-jean-zay-accounting/v1"
        ),
    }
    payloads: dict[str, dict[str, Any]] = {}
    for name, schema in provenance_specs.items():
        path = provenance / name
        if not path.is_file():
            raise ValueError(
                "The canonical terminal inventory is missing scheduler "
                f"provenance required to reconcile jobs 844032/844396: {path}."
            )
        payload = _load_json(path)
        if (
            payload.get("schema_version") != schema
            or payload.get("study_id") != contract.study_id
        ):
            raise ValueError(
                f"Scheduler provenance has invalid schema/study identity: {path}."
            )
        payloads[name] = payload
    submission = payloads["jean_zay_submission.json"]
    replacement_plan = payloads["replacement_plan.json"]
    recovery = payloads["recovery_submission.json"]
    fallback_plan = payloads["local_fallback_plan.json"]
    fallback = payloads["local_fallback_submission.json"]
    accounting = payloads["jean_zay_terminal_accounting.json"]
    supersedes = recovery.get("supersedes")
    replacement = recovery.get("replacement")
    replacement_reason = replacement_plan.get("reason")
    fallback_reason = fallback_plan.get("reason")
    fallback_job = fallback.get("superseded_jean_zay_job")
    accounting_jobs = accounting.get("jobs")
    if not isinstance(accounting_jobs, list):
        raise ValueError("Jean Zay terminal accounting has no jobs inventory.")
    accounting_by_id = {
        str(job.get("job_id")): job
        for job in accounting_jobs
        if isinstance(job, Mapping)
    }
    if len(accounting_jobs) != 2 or set(accounting_by_id) != {"844032", "844396"}:
        raise ValueError(
            "Jean Zay terminal accounting does not contain exactly jobs "
            "844032 and 844396."
        )
    expected_job_names = {
        "844032": "pd-c3-lr-ladder",
        "844396": "pd-c3-lr2",
    }
    for job_id, accounting_job in accounting_by_id.items():
        state = accounting_job.get("state")
        if (
            accounting_job.get("job_name") != expected_job_names[job_id]
            or not isinstance(state, str)
            or not state.startswith("CANCELLED")
            or accounting_job.get("exit_code") != "0:0"
            or accounting_job.get("elapsed") != "00:00:00"
            or accounting_job.get("start") is not None
            or not isinstance(accounting_job.get("end"), str)
        ):
            raise ValueError(
                "Jean Zay terminal accounting does not prove zero-runtime "
                f"pre-allocation cancellation for job {job_id}."
            )
    if (
        not isinstance(supersedes, Mapping)
        or supersedes.get("job_id") != "844032"
        or supersedes.get("state") != "CANCELLED before allocation"
        or supersedes.get("elapsed") != "00:00:00"
        or supersedes.get("artifacts") != 0
        or submission.get("job_id") != "844032"
        or not isinstance(replacement_reason, Mapping)
        or replacement_reason.get("superseded_job_id") != "844032"
        or replacement_reason.get("superseded_job_state")
        != "CANCELLED before allocation"
        or replacement_reason.get("superseded_job_elapsed") != "00:00:00"
        or replacement_reason.get("scientific_artifacts_produced") != 0
    ):
        raise ValueError(
            "Scheduler provenance does not prove that Jean Zay job 844032 was "
            "canceled before allocation with zero artifacts."
        )
    if (
        not isinstance(replacement, Mapping)
        or replacement.get("job_id") != "844396"
        or not isinstance(fallback_reason, Mapping)
        or fallback_reason.get("jean_zay_job_id") != "844396"
        or fallback_reason.get("state")
        != "four array elements pending with QOSGrpCpuLimit and no advertised start"
        or fallback_reason.get("scientific_artifacts_produced") != 0
        or not isinstance(fallback_job, Mapping)
        or fallback_job.get("job_id") != "844396"
        or fallback_job.get("state_before_cancellation")
        != "four PENDING array elements; QOSGrpCpuLimit; no advertised start"
        or fallback_job.get("terminal_state") != "CANCELLED by owner"
        or fallback_job.get("elapsed") != "00:00:00"
        or fallback_job.get("scientific_artifacts") != 0
    ):
        raise ValueError(
            "Scheduler provenance does not prove that Jean Zay job 844396 was "
            "canceled before allocation with zero artifacts."
        )
    return [
        {
            "evidence_kind": "canceled_scheduler_job",
            "job_id": "844032",
            "target": "jean-zay",
            "authoritative": False,
            "coverage_included": False,
            "numeric_summary_included": False,
            "terminal_state": supersedes["state"],
            "elapsed": supersedes["elapsed"],
            "scientific_artifacts": 0,
            "excluded_reason": supersedes.get("reason"),
            "source_result_jsons": [],
            "provenance_sources": _provenance_sources(
                provenance,
                (
                    "launch_plan.json",
                    "jean_zay_submission.json",
                    "recovery_submission.json",
                    "replacement_plan.json",
                    "jean_zay_terminal_accounting.json",
                ),
            ),
        },
        {
            "evidence_kind": "canceled_scheduler_job",
            "job_id": "844396",
            "target": "jean-zay",
            "authoritative": False,
            "coverage_included": False,
            "numeric_summary_included": False,
            "terminal_state": fallback_job["terminal_state"],
            "elapsed": fallback_job["elapsed"],
            "scientific_artifacts": 0,
            "excluded_reason": fallback_job.get("reason"),
            "source_result_jsons": [],
            "provenance_sources": _provenance_sources(
                provenance,
                (
                    "recovery_submission.json",
                    "local_fallback_plan.json",
                    "local_fallback_submission.json",
                    "jean_zay_terminal_accounting.json",
                ),
            ),
        },
    ]


def _run_inventory(
    rows: Sequence[Mapping[str, Any]],
    contract: StudyContract,
    *,
    evidence_root: Path | None,
) -> dict[str, Any]:
    included, numeric_exclusions, reconciliation = _canonical_run_inventory(
        rows, contract
    )
    archived_attempts = _archived_attempt_inventory(
        included, evidence_root=evidence_root
    )
    supporting_results = _supporting_result_inventory(evidence_root, included)
    documented_attempts = _documented_operational_attempt_inventory(
        evidence_root, supporting_results
    )
    canceled_jobs = _canceled_job_inventory(evidence_root, contract)
    excluded_runs = [*supporting_results, *documented_attempts, *archived_attempts]
    source_results = [
        {
            "authority": "canonical_terminal_cell",
            "run_id": run["run_id"],
            "path": run["result_json_path"],
            "sha256": run["result_json_sha256"],
        }
        for run in included
        if run["result_json_sha256"] is not None
    ]
    for record in excluded_runs:
        for source in record["source_result_jsons"]:
            source_results.append(
                {
                    "authority": record["evidence_kind"],
                    "run_id": record.get("run_id"),
                    **source,
                }
            )
    reconciliation = {
        **reconciliation,
        "excluded_non_authoritative_run_count": len(excluded_runs),
        "discovered_operational_bundle_count": sum(
            bool(record.get("manifest_json_sha256"))
            or bool(record.get("status_json_sha256"))
            for record in [*supporting_results, *documented_attempts]
        ),
        "discovered_standalone_result_count": sum(
            record.get("reporting_files_present") == ["result.json"]
            for record in [*supporting_results, *documented_attempts]
        ),
        "discovered_target_gate_count": sum(
            record.get("evidence_kind") == "target_gate"
            for record in [*supporting_results, *documented_attempts]
        ),
        "discovered_smoke_count": sum(
            record.get("evidence_kind") == "smoke"
            for record in [*supporting_results, *documented_attempts]
        ),
        "canceled_zero_artifact_execution_count": len(canceled_jobs),
        "all_discovered_source_result_json_count": len(source_results),
    }
    return {
        "scope": (
            "Four receipt-bound canonical scientific shards; other locally "
            "discoverable study result bundles, smokes, gates, archived "
            "attempts, and canceled scheduler jobs are non-authoritative."
        ),
        "study_evidence_root": None if evidence_root is None else str(evidence_root),
        "included_runs": included,
        "numeric_summary_excluded_terminal_runs": numeric_exclusions,
        "excluded_runs": excluded_runs,
        "excluded_execution_attempts": canceled_jobs,
        "source_result_jsons": source_results,
        "reconciliation": reconciliation,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _surface_axes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(17, 13), sharex=True, sharey=True)
    return plt, figure, axes


def _plot_heatmaps(
    rows: Sequence[Mapping[str, Any]],
    surfaces: Sequence[Mapping[str, Any]],
    contract: StudyContract,
    output: Path,
    *,
    field: str,
    colorbar_label: str,
    annotation_format: str,
    vmin: float,
    vmax: float,
    cmap_name: str,
) -> None:
    plt, figure, axes = _surface_axes()
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#d9d9d9")
    image = None
    for axis, surface in zip(axes.flat, surfaces):
        matrix = np.full((8, 8), np.nan)
        surface_rows = [row for row in rows if row["surface_id"] == surface["surface_id"]]
        for row in surface_rows:
            value = row[field]
            if value is not None:
                matrix[row["conv_axis_index"], row["dense_axis_index"]] = float(value)
        image = axis.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        for conv_index in range(8):
            for dense_index in range(8):
                value = matrix[conv_index, dense_index]
                text = "NF" if np.isnan(value) else format(value, annotation_format)
                axis.text(dense_index, conv_index, text, ha="center", va="center", fontsize=6)
        axis.set_title(f"{surface['scheme']} / {surface['optimizer']} (exploratory)")
        axis.set_xticks(range(8), [f"{value:.2g}" for value in contract.rho_dense], rotation=45, ha="right")
        axis.set_yticks(range(8), [f"{value:.2g}" for value in contract.rho_conv])
        axis.set_xlabel(r"$\rho_{dense}$")
        axis.set_ylabel(r"$\rho_{conv}$")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label=colorbar_label, shrink=0.85)
    figure.suptitle(EXPLORATORY_LABEL, fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_scatter(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#4c78a8", "ours": "#f58518"}
    markers = {"SGD": "o", "Adam": "s"}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for surface_id in sorted({str(row["surface_id"]) for row in rows}):
        surface_rows = [
            row for row in rows
            if row["surface_id"] == surface_id and row["canonical_outcome"] == "numeric_complete"
        ]
        if not surface_rows:
            continue
        scheme = str(surface_rows[0]["scheme"])
        optimizer = str(surface_rows[0]["optimizer"])
        label = f"{scheme} / {optimizer}"
        axes[0].scatter(
            [row["exact_bound_occupancy_percent"] for row in surface_rows],
            [row["final_validation_accuracy_percent"] for row in surface_rows],
            color=colors[scheme], marker=markers[optimizer], alpha=0.75, label=label,
        )
        projection_rows = [row for row in surface_rows if row["median_projection_efficiency"] is not None]
        axes[1].scatter(
            [row["median_projection_efficiency"] for row in projection_rows],
            [row["final_validation_accuracy_percent"] for row in projection_rows],
            color=colors[scheme], marker=markers[optimizer], alpha=0.75, label=label,
        )
    axes[0].set_xlabel("Final weights exactly at either bound (%)")
    axes[0].set_ylabel("Final validation accuracy (%)")
    axes[1].set_xlabel("Median applied/proposed update RMS")
    axes[1].set_ylabel("Final validation accuracy (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(EXPLORATORY_LABEL, fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown(report: Mapping[str, Any]) -> str:
    targets = sorted(
        {str(shard["target"]) for shard in report["source_shards"]}
    )

    def correlation_text(value: Any) -> str:
        return "undefined" if value is None else f"{float(value):.4f}"

    def number_text(value: Any, format_spec: str) -> str:
        return "unavailable" if value is None else format(float(value), format_spec)

    lines = [
        "# Conv3 exploratory LR ladder — NONCANONICAL",
        "",
        f"> **{EXPLORATORY_LABEL}**",
        "",
        "## Coverage",
        "",
        f"Validated **{report['coverage']['cell_count']} / 256** cells across "
        f"**{report['coverage']['surface_count']} / 4** collected shards "
        f"(targets: **{', '.join(targets)}**). "
        f"Numeric: **{report['coverage']['numeric_cell_count']}**; non-finite: "
        f"**{report['coverage']['nonfinite_cell_count']}**.",
        "",
        f"Analysis generated at `{report['analysis_generated_at']}` with analyzer "
        f"source SHA-256 `{report['analyzer_source_sha256']}`.",
        "",
        "### Run and execution-evidence inventory",
        "",
        f"The terminal inventory reconciles **{report['run_inventory']['reconciliation']['canonical_terminal_cell_count']} / 256** "
        "canonical scientific cells. "
        f"**{report['run_inventory']['reconciliation']['numeric_summary_included_count']}** "
        "numeric-complete cells have source `result.json` hashes; "
        f"**{report['run_inventory']['reconciliation']['numeric_summary_excluded_terminal_count']}** "
        "retained terminal non-finite cells count toward coverage but are excluded "
        "from finite-metric summaries.",
        "",
        f"Excluded non-authoritative run evidence: **{len(report['excluded_runs'])}**; "
        f"canceled zero-artifact scheduler attempts: **{len(report['excluded_execution_attempts'])}**.",
        "",
    ]
    if report["excluded_runs"]:
        for record in report["excluded_runs"]:
            lines.append(
                f"- `{record['evidence_kind']}` at `{record['path']}`: "
                f"{record['excluded_reason']}"
            )
    else:
        lines.append("- No non-authoritative result bundles or archived attempts were discovered.")
    for attempt in report["excluded_execution_attempts"]:
        lines.append(
            f"- Jean Zay job `{attempt['job_id']}`: `{attempt['terminal_state']}`, "
            f"elapsed `{attempt['elapsed']}`, zero scientific artifacts; "
            f"{attempt['excluded_reason']}"
        )
    lines.extend(
        [
        "",
        "## Target assignment and comparison limits",
        "",
        "The assignment below is derived from the validated terminal transport receipts.",
        "",
        "| Scheme | Optimizer | Target | Validated execution environment |",
        "|---|---|---|---|",
        ]
    )
    assignment = report["target_assignment"]
    for row in assignment["surfaces"]:
        lines.append(
            f"| {row['scheme']} | {row['optimizer']} | {row['target']} | "
            f"`{_stable_json(row['execution_environment'])}` |"
        )
    lines.append("")
    for comparison in assignment["within_optimizer_scheme_comparisons"]:
        lines.append(f"- {comparison['statement']}")
    lines.extend(
        [
            "",
            "> **Cross-optimizer comparison limit.** "
            + assignment["cross_optimizer_comparison"]["statement"],
            "",
            "## Highest observed accuracy per surface",
            "",
            "These are descriptive maxima, not selected learning rates.",
            "",
            "| Scheme | Optimizer | Target | Numeric / 64 | rho conv | rho dense | Accuracy | Loss | At either bound | Projection efficiency | Range status |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["surfaces"]:
        if row["best_observed_cell_index"] is None:
            values = ("—",) * 6
        else:
            values = (
                f"{row['best_observed_rho_conv']:.6g}",
                f"{row['best_observed_rho_dense']:.6g}",
                f"{row['best_observed_validation_accuracy_percent']:.2f}%",
                f"{row['best_observed_validation_loss']:.6g}",
                f"{row['best_observed_exact_bound_occupancy_percent']:.2f}%",
                number_text(
                    row["best_observed_median_projection_efficiency"], ".4g"
                ),
            )
        lines.append(
            f"| {row['scheme']} | {row['optimizer']} | {row['target']} | "
            f"{row['numeric_cell_count']} / 64 | {' | '.join(values)} | "
            f"`{row['range_status']}` |"
        )
    lines.extend(
        [
            "",
            "### Four-direction axial range evidence",
            "",
            "This local accuracy geometry is descriptive only. `outside_tested_range` "
            "marks an open direction; a local bracket requires four observed numeric "
            "neighbors with strictly lower accuracy.",
            "",
            "Maximum-accuracy co-maxima are enumerated conservatively: a boundary "
            "or unresolved co-maximum prevents the aggregate surface from being "
            "called locally bracketed even when the loss/rank-tiebroken displayed "
            "point is interior.",
            "",
            "| Scheme | Optimizer | Co-maximum cell | rho conv | rho dense | Accuracy | Selected detail row | Individual range status |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in report["surfaces"]:
        for maximum in row["maximum_accuracy_co_maxima"]:
            lines.append(
                f"| {row['scheme']} | {row['optimizer']} | "
                f"{maximum['cell_index']} | {maximum['rho_conv']:.8g} | "
                f"{maximum['rho_dense']:.8g} | "
                f"{maximum['final_validation_accuracy_percent']:.2f}% | "
                f"{maximum['selected_by_loss_rank_tiebreak']} | "
                f"`{maximum['range_evidence']['range_status']}` |"
            )
    lines.extend(
        [
            "",
            "| Scheme | Optimizer | Direction | Grid state | Cell | Outcome | Accuracy | Relation to best |",
            "|---|---|---|---|---:|---|---:|---|",
        ]
    )
    for row in report["surfaces"]:
        if not row["axial_neighbors"]:
            lines.append(
                f"| {row['scheme']} | {row['optimizer']} | unavailable | "
                "unavailable | — | unavailable | unavailable | unavailable |"
            )
            continue
        for direction, neighbor in row["axial_neighbors"].items():
            lines.append(
                f"| {row['scheme']} | {row['optimizer']} | {direction} | "
                f"{neighbor['grid_state']} | "
                f"{number_text(neighbor['cell_index'], '.0f')} | "
                f"{neighbor['canonical_outcome'] or 'unavailable'} | "
                f"{number_text(neighbor['final_validation_accuracy_percent'], '.2f')} | "
                f"{neighbor['accuracy_relation_to_best']} |"
            )
    lines.extend(
        [
            "",
        ]
    )
    for row in report["surfaces"]:
        lines.append(
            f"- **{row['scheme']} / {row['optimizer']}:** "
            f"`{row['range_status']}` — {row['range_status_statement']}"
        )
    lines.extend(
        [
            "",
            "### Actual raw learning-rate vector at each highest-accuracy point",
            "",
            "Weight LRs are read from the validated winning `cell.json`; rho "
            "coordinates are retained separately and are not substituted for "
            "the actual optimizer LR vector.",
            "",
            "| Scheme | Optimizer | Target | Cell | rho conv | rho dense | ConvWeight_0 LR | ConvWeight_1 LR | ConvWeight_2 LR | DenseWeight_0 LR | Bias LRs |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["surfaces"]:
        weight_lrs = row["best_observed_weight_learning_rates"]
        if weight_lrs is None:
            lr_values = ("unavailable",) * 4
        else:
            lr_values = tuple(
                number_text(weight_lrs[name], ".8g") for name in WEIGHT_NAMES
            )
        bias_text = (
            "`Bias_0=Bias_1=Bias_2=0`"
            if row["best_observed_bias_lrs_zero"] is True
            else "unavailable"
        )
        lines.append(
            f"| {row['scheme']} | {row['optimizer']} | {row['target']} | "
            f"{number_text(row['best_observed_cell_index'], '.0f')} | "
            f"{number_text(row['best_observed_rho_conv'], '.8g')} | "
            f"{number_text(row['best_observed_rho_dense'], '.8g')} | "
            f"{' | '.join(lr_values)} | {bias_text} |"
        )
    lines.extend(
        [
            "",
            "### Endpoint occupancy at each surface's highest-accuracy point",
            "",
            "Percentages are recomputed exactly from each final checkpoint.",
            "",
            "| Scheme | Optimizer | ConvWeight_0 | ConvWeight_1 | ConvWeight_2 | DenseWeight_0 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["surfaces"]:
        by_parameter = row[
            "best_observed_exact_bound_occupancy_by_parameter_percent"
        ]
        values = [
            number_text(by_parameter.get(name), ".2f") + "%"
            if by_parameter.get(name) is not None
            else "unavailable"
            for name in WEIGHT_NAMES
        ]
        lines.append(
            f"| {row['scheme']} | {row['optimizer']} | {' | '.join(values)} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive diagnostic correlations",
            "",
            f"> **Noncausal.** {CORRELATION_CAVEAT}",
            "",
            "Coefficients use numeric-complete cells within each surface. "
            "`undefined` means fewer than two pairs or zero variance in at "
            "least one variable.",
            "",
            "| Scheme | Optimizer | Diagnostic | n | Pearson r | Spearman rho |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in report["surfaces"]:
        correlations = row["descriptive_correlations"]
        diagnostic_rows = [
            (
                "Exact endpoint occupancy — all weights",
                correlations["accuracy_vs_exact_bound_occupancy"],
            ),
            *[
                (
                    f"Exact endpoint occupancy — {name}",
                    correlations[
                        "accuracy_vs_exact_bound_occupancy_by_parameter"
                    ][name],
                )
                for name in WEIGHT_NAMES
            ],
            (
                "Median projection efficiency",
                correlations["accuracy_vs_median_projection_efficiency"],
            ),
        ]
        for label, correlation in diagnostic_rows:
            lines.append(
                f"| {row['scheme']} | {row['optimizer']} | {label} | "
                f"{correlation['sample_count']} | "
                f"{correlation_text(correlation['pearson'])} | "
                f"{correlation_text(correlation['spearman'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- All cells use ordinary MNIST; no official test examples were read.",
            "- Every cell restarts from the same hash-verified `Uniform[1e-5,1e-4)` Conv3 initializer and keeps bias LR exactly zero.",
            "- Canaries, scientific safety rejection gates, and post-training T/K checks were intentionally disabled.",
            "- A high observed accuracy or favorable clipping statistic here does not create a canonical LR handoff. It identifies neighborhoods for a later protocol-compliant confirmation.",
            "- Range status uses final validation accuracy and direct axial neighbors only; it is not the protocol loss-plateau selector.",
            "",
            "## Artifacts",
            "",
            "- `cells.csv`: all numeric and non-finite terminal cells.",
            "- `summary.csv`: conventional descriptive per-surface summary.",
            "- `surfaces.csv`: descriptive per-surface aggregates.",
            "- `summary.json`: validated provenance, source-result hashes, included/excluded inventories, range evidence, coverage, rows, and aggregates.",
            "- `plots/`: accuracy, endpoint clipping, projection heatmaps, and diagnostic scatters.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    *,
    study_config_path: Path = DEFAULT_STUDY,
    study_root: Path | None = None,
    shard_roots: Sequence[Path] = (),
    receipt_roots: Sequence[Path] = (),
    output_dir: Path,
) -> dict[str, Any]:
    contract = load_contract(study_config_path)
    roots = discover_shard_roots(study_root=study_root, shard_roots=shard_roots)
    shards = _load_shards(roots, contract)
    receipts = _validate_receipts(shards, contract, receipt_roots=receipt_roots)
    target_assignment = _target_assignment(contract, receipts)
    environments_by_surface = {
        str(receipt["surface_id"]): dict(receipt["execution_environment"])
        for receipt in receipts
    }
    rows: list[dict[str, Any]] = []
    for shard in shards:
        rows.extend(
            _read_surface(
                shard,
                contract,
                receipt_environment=environments_by_surface[
                    str(shard.surface["surface_id"])
                ],
            )
        )
    if len(rows) != 256:
        raise AssertionError(f"Expected 256 validated rows, got {len(rows)}.")
    initial_parameter_identities = {
        str(row["probe_initial_parameter_sha256"]) for row in rows
    }
    if len(initial_parameter_identities) != 1:
        raise ValueError(
            "Surface probes do not share one loaded initial-parameter identity."
        )
    probe_initial_parameter_sha256 = next(iter(initial_parameter_identities))
    evidence_root = _infer_study_evidence_root(
        study_root=study_root, shards=shards
    )
    inventory = _run_inventory(rows, contract, evidence_root=evidence_root)
    surfaces = [
        _surface_summary(
            spec,
            [row for row in rows if row["surface_id"] == spec["surface_id"]],
        )
        for spec in contract.surfaces
    ]
    within_optimizer = {
        str(row["optimizer"]): row
        for row in target_assignment["within_optimizer_scheme_comparisons"]
    }
    assignment_by_surface = {
        str(row["surface_id"]): row for row in target_assignment["surfaces"]
    }
    cross_optimizer = target_assignment["cross_optimizer_comparison"]
    for surface in surfaces:
        matched = within_optimizer[str(surface["optimizer"])]
        environment = assignment_by_surface[str(surface["surface_id"])][
            "execution_environment"
        ]
        surface.update(
            {
                "execution_environment": environment,
                "execution_environment_json": _stable_json(environment),
                "baseline_vs_ours_within_target": matched["same_target"],
                "baseline_vs_ours_same_host_runtime_contract": matched[
                    "same_host_runtime_contract"
                ],
                "baseline_vs_ours_host_runtime_confounded": matched[
                    "host_target_confounded"
                ],
                "baseline_vs_ours_execution_environments_json": _stable_json(
                    {
                        "baseline": matched["baseline_execution_environment"],
                        "ours": matched["ours_execution_environment"],
                    }
                ),
                "cross_optimizer_host_runtime_confounded": cross_optimizer[
                    "host_runtime_confounded"
                ],
                "cross_optimizer_causal_interpretation_allowed": cross_optimizer[
                    "causal_interpretation_allowed"
                ],
                "cross_optimizer_comparison_statement": cross_optimizer[
                    "statement"
                ],
            }
        )
    numeric_count = sum(row["canonical_outcome"] == "numeric_complete" for row in rows)
    output = Path(output_dir).expanduser().resolve()
    analyzer_source = Path(__file__).resolve()
    artifacts = {
        "cells_csv": "cells.csv",
        "summary_csv": "summary.csv",
        "surfaces_csv": "surfaces.csv",
        "summary_json": "summary.json",
        "report_markdown": "report.md",
        "accuracy_heatmaps": "plots/final_validation_accuracy_heatmaps.png",
        "occupancy_heatmaps": "plots/endpoint_bound_occupancy_heatmaps.png",
        "projection_heatmaps": "plots/projection_efficiency_heatmaps.png",
        "diagnostic_scatter": "plots/accuracy_vs_diagnostics.png",
    }
    report = {
        "schema_version": ANALYSIS_SCHEMA,
        "analysis_generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "analyzer_source": str(analyzer_source),
        "analyzer_source_sha256": sha256_file(analyzer_source),
        "study_id": contract.study_id,
        "evidence_label": EXPLORATORY_LABEL,
        "evidence_class": "ordinary_mnist_exploratory",
        "exploratory_noncanonical": True,
        "canonical_lr_handoff": False,
        "paper_evidence": False,
        "official_test_read": False,
        "disabled_scientific_checks": [
            "canary",
            "scientific_safety_rejections",
            "post_training_tk",
        ],
        "study_config": str(contract.path),
        "study_config_sha256": contract.sha256,
        "initializer_checkpoint_sha256": contract.initializer_sha256,
        "probe_initial_parameter_sha256": probe_initial_parameter_sha256,
        "source_shards": [
            {
                "root": str(shard.root),
                "target": shard.target,
                "surface_id": shard.surface["surface_id"],
                "resolved_sha256": sha256_file(shard.root / "study.resolved.json"),
            }
            for shard in shards
        ],
        "transport_receipts": receipts,
        "run_inventory": {
            "scope": inventory["scope"],
            "study_evidence_root": inventory["study_evidence_root"],
            "source_result_jsons": inventory["source_result_jsons"],
            "reconciliation": inventory["reconciliation"],
        },
        "included_runs": inventory["included_runs"],
        "numeric_summary_excluded_terminal_runs": inventory[
            "numeric_summary_excluded_terminal_runs"
        ],
        "excluded_runs": inventory["excluded_runs"],
        "excluded_execution_attempts": inventory[
            "excluded_execution_attempts"
        ],
        "target_assignment": target_assignment,
        "correlation_interpretation": {
            "causal": False,
            "scope": "within_surface_numeric_complete_cells",
            "caveat": CORRELATION_CAVEAT,
        },
        "coverage": {
            "surface_count": len(shards),
            "expected_surface_count": 4,
            "cell_count": len(rows),
            "expected_cell_count": 256,
            "cells_per_surface": 64,
            "numeric_cell_count": numeric_count,
            "nonfinite_cell_count": len(rows) - numeric_count,
            "canonical_bundle_count": len(rows),
        },
        "surfaces": surfaces,
        "cells": rows,
        "artifacts": artifacts,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / artifacts["cells_csv"], rows, CELL_FIELDS)
    _write_csv(output / artifacts["summary_csv"], surfaces, SURFACE_FIELDS)
    _write_csv(output / artifacts["surfaces_csv"], surfaces, SURFACE_FIELDS)
    _plot_heatmaps(
        rows,
        contract.surfaces,
        contract,
        output / artifacts["accuracy_heatmaps"],
        field="final_validation_accuracy_percent",
        colorbar_label="Final validation accuracy (%)",
        annotation_format=".1f",
        vmin=0.0,
        vmax=100.0,
        cmap_name="viridis",
    )
    _plot_heatmaps(
        rows,
        contract.surfaces,
        contract,
        output / artifacts["occupancy_heatmaps"],
        field="exact_bound_occupancy_percent",
        colorbar_label="Weights exactly at either bound (%)",
        annotation_format=".1f",
        vmin=0.0,
        vmax=100.0,
        cmap_name="magma",
    )
    projection_values = [
        float(row["median_projection_efficiency"])
        for row in rows
        if row["median_projection_efficiency"] is not None
    ]
    _plot_heatmaps(
        rows,
        contract.surfaces,
        contract,
        output / artifacts["projection_heatmaps"],
        field="median_projection_efficiency",
        colorbar_label="Median applied/proposed update RMS",
        annotation_format=".2f",
        vmin=0.0,
        vmax=max([1.0, *projection_values]),
        cmap_name="cividis",
    )
    _plot_scatter(rows, output / artifacts["diagnostic_scatter"])
    (output / artifacts["report_markdown"]).write_text(
        _markdown(report), encoding="utf-8"
    )
    atomic_write_json(output / artifacts["summary_json"], report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STUDY)
    parser.add_argument(
        "--study-root",
        type=Path,
        help="Discover collected task roots below STUDY_ROOT/shards.",
    )
    parser.add_argument(
        "--receipt-root",
        type=Path,
        action="append",
        default=[],
        help="Additional directory containing terminal transport receipts.",
    )
    parser.add_argument(
        "--shard-root",
        type=Path,
        action="append",
        default=[],
        help="Collected task root; repeat exactly once per surface.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir is None:
        if args.study_root is None:
            raise SystemExit("--output-dir is required when --study-root is omitted")
        output_dir = args.study_root / "analysis" / "exploratory_lr_ladder"
    else:
        output_dir = args.output_dir
    try:
        report = analyze(
            study_config_path=args.study_config,
            study_root=args.study_root,
            shard_roots=args.shard_root,
            receipt_roots=args.receipt_root,
            output_dir=output_dir,
        )
    except IncompleteExploratoryCoverageError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "label": EXPLORATORY_LABEL,
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
