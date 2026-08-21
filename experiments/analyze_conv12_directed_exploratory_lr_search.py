#!/usr/bin/env python3
"""Fail-closed analysis of the directed Conv1/Conv2 exploratory LR search.

The study is a sparse, non-duplicate extension of an earlier LR search.  This
module therefore derives every expected coordinate from ``surface_cells``;
directory numbering is never interpreted as a dense grid.  It can write an
honestly labelled partial snapshot while jobs are live, or reject anything
short of the complete 12-surface/257-cell contract with ``--require-complete``.

This is ordinary-MNIST exploratory evidence.  It is not a learning-rate
handoff and it is not paper-facing evidence.
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
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from experiments.reporting import atomic_write_json, sha256_file, validate_run
    from experiments.rho_search import prepare_training_config
    from experiments.run_conv12_bounded_rho import build_source_config
    from experiments.run_conv12_directed_exploratory_lr_search import (
        DEFAULT_STUDY,
        EXPECTED_INITIALIZER_SHA256,
        EXPECTED_TK,
        RESOLVED_SCHEMA,
        STUDY_SCHEMA,
        SURFACE_STATUS_SCHEMA,
        SURFACE_SUMMARY_SCHEMA,
        load_study,
        surface_cells,
        surface_specs,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from reporting import atomic_write_json, sha256_file, validate_run  # type: ignore[no-redef]
    from rho_search import prepare_training_config  # type: ignore[no-redef]
    from run_conv12_bounded_rho import build_source_config  # type: ignore[no-redef]
    from run_conv12_directed_exploratory_lr_search import (  # type: ignore[no-redef]
        DEFAULT_STUDY,
        EXPECTED_INITIALIZER_SHA256,
        EXPECTED_TK,
        RESOLVED_SCHEMA,
        STUDY_SCHEMA,
        SURFACE_STATUS_SCHEMA,
        SURFACE_SUMMARY_SCHEMA,
        load_study,
        surface_cells,
        surface_specs,
    )


ANALYSIS_SCHEMA = "perfectdiode-conv12-directed-exploratory-lr-analysis/v1"
RECEIPT_SCHEMA = "perfectdiode-conv12-directed-exploratory-lr-transport/v1"
EXPECTED_SURFACE_COUNT = 12
EXPECTED_CELL_COUNT = 257
EXPECTED_CELL_COUNTS = (39, 30, 18, 16, 20, 6, 30, 40, 17, 17, 14, 10)
WEIGHT_MIN = 1e-5
WEIGHT_MAX = 1e-4
EXPLORATORY_LABEL = (
    "EXPLORATORY / NONCANONICAL ordinary-MNIST diagnostic; safety rejection, "
    "per-cell canaries, and post-training T/K checks were disabled."
)
TERMINAL_SURFACE_STATES = frozenset({"complete", "unresolved_probe", "failed"})
TERMINAL_CELL_STATUSES = frozenset(
    {"complete", "candidate_rejected_nonfinite", "candidate_rejected_safety"}
)
TASK_ROOT_PATTERN = re.compile(r".+-task_(?P<task>[0-9]|1[01])$")

CELL_FIELDS = (
    "study_id",
    "surface_index",
    "surface_id",
    "architecture",
    "scheme",
    "optimizer",
    "target",
    "cell_index",
    "execution_rank",
    "rho_conv_ladder_index",
    "rho_dense_ladder_index",
    "rho_conv",
    "rho_dense",
    "raw_status",
    "canonical_outcome",
    "transport_validated",
    "final_validation_accuracy",
    "final_validation_accuracy_percent",
    "best_validation_accuracy",
    "best_validation_accuracy_percent",
    "final_validation_loss",
    "best_epoch",
    "final_lower_count",
    "final_upper_count",
    "final_either_count",
    "final_weight_count",
    "final_either_fraction",
    "final_either_percent",
    "final_conv_lower_count",
    "final_conv_upper_count",
    "final_conv_either_count",
    "final_conv_weight_count",
    "final_conv_either_fraction",
    "final_conv_either_percent",
    "final_dense_lower_count",
    "final_dense_upper_count",
    "final_dense_either_count",
    "final_dense_weight_count",
    "final_dense_either_fraction",
    "final_dense_either_percent",
    "final_by_parameter_json",
    "best_lower_count",
    "best_upper_count",
    "best_either_count",
    "best_weight_count",
    "best_either_fraction",
    "best_either_percent",
    "best_conv_lower_count",
    "best_conv_upper_count",
    "best_conv_either_count",
    "best_conv_weight_count",
    "best_conv_either_fraction",
    "best_conv_either_percent",
    "best_dense_lower_count",
    "best_dense_upper_count",
    "best_dense_either_count",
    "best_dense_weight_count",
    "best_dense_either_fraction",
    "best_dense_either_percent",
    "best_by_parameter_json",
    "learning_rates_by_parameter_json",
    "canonical_bundle_valid",
    "official_test_read",
    "run_dir",
)

SURFACE_FIELDS = (
    "surface_index",
    "surface_id",
    "architecture",
    "scheme",
    "optimizer",
    "declared_target",
    "observed_target",
    "surface_state",
    "receipt_state",
    "expected_cell_count",
    "observed_terminal_cell_count",
    "numeric_cell_count",
    "nonfinite_cell_count",
    "missing_cell_count",
    "best_tested_cell_index",
    "best_tested_rho_conv",
    "best_tested_rho_dense",
    "best_tested_validation_accuracy",
    "best_tested_validation_accuracy_percent",
    "best_tested_final_clipping_percent",
    "best_tested_best_clipping_percent",
    "range_status",
    "locally_bracketed_by_accuracy",
    "open_directions_json",
    "unresolved_directions_json",
    "improving_or_tied_directions_json",
    "neighbors_json",
)

CORRELATION_FIELDS = (
    "surface_index",
    "surface_id",
    "architecture",
    "scheme",
    "optimizer",
    "endpoint",
    "scope",
    "parameter",
    "accuracy_field",
    "clipping_field",
    "sample_count",
    "pearson",
    "spearman",
)


@dataclass(frozen=True)
class StudyContract:
    path: Path
    payload: dict[str, Any]
    parent_path: Path
    parent: dict[str, Any]
    sha256: str
    study_id: str
    surfaces: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Shard:
    root: Path
    surface: dict[str, Any]
    resolved: dict[str, Any]
    target: str


class IncompleteCoverageError(RuntimeError):
    """Raised when complete coverage was requested but is not present."""

    def __init__(self, message: str, *, report: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.report = None if report is None else dict(report)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite_float(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Expected finite {label}, got {value!r}.")
    return float(value)


def _fraction(value: Any, *, label: str) -> float:
    result = _finite_float(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"Expected {label} in [0,1], got {result!r}.")
    return result


def _same_float(left: Any, right: Any) -> bool:
    return math.isclose(
        _finite_float(left, label="floating value"),
        _finite_float(right, label="floating value"),
        rel_tol=1e-15,
        abs_tol=0.0,
    )


def _path_has_suffix(value: Any, suffix: Sequence[str]) -> bool:
    if not isinstance(value, str) or not value:
        return False
    parts = Path(value).parts
    return len(parts) >= len(suffix) and tuple(parts[-len(suffix) :]) == tuple(suffix)


def _weight_names(architecture: str) -> tuple[str, ...]:
    if architecture == "conv1":
        return ("ConvWeight_0", "DenseWeight_0")
    if architecture == "conv2":
        return ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0")
    raise ValueError(f"Unsupported directed architecture {architecture!r}.")


def _bias_names(architecture: str) -> tuple[str, ...]:
    if architecture == "conv1":
        return ("Bias_0",)
    if architecture == "conv2":
        return ("Bias_0", "Bias_1")
    raise ValueError(f"Unsupported directed architecture {architecture!r}.")


def load_contract(path: Path = DEFAULT_STUDY) -> StudyContract:
    """Use the scientific runner's exact loader, then pin analyzer invariants."""

    source, study, parent_path, parent = load_study(path)
    surfaces = tuple(surface_specs(study))
    if study.get("schema_version") != STUDY_SCHEMA:
        raise ValueError("Unexpected directed-study schema after canonical load.")
    if len(surfaces) != EXPECTED_SURFACE_COUNT:
        raise ValueError(f"Expected {EXPECTED_SURFACE_COUNT} directed surfaces.")
    counts = tuple(len(surface_cells(study, surface)) for surface in surfaces)
    expected_total = sum(counts)
    if counts != EXPECTED_CELL_COUNTS or expected_total != EXPECTED_CELL_COUNT:
        raise ValueError(f"Expected {EXPECTED_CELL_COUNT} directed cells, got {expected_total}.")
    if tuple(int(surface["index"]) for surface in surfaces) != tuple(
        range(EXPECTED_SURFACE_COUNT)
    ):
        raise ValueError("Directed surface indices are not canonical.")
    return StudyContract(
        path=source,
        payload=study,
        parent_path=parent_path,
        parent=parent,
        sha256=sha256_file(source),
        study_id=str(study["study_id"]),
        surfaces=surfaces,
    )


def discover_shard_roots(
    *, study_root: Path | None, shard_roots: Sequence[Path]
) -> list[Path]:
    """Discover task roots without descending into smoke or collection staging."""

    candidates: list[Path] = []
    if study_root is not None:
        parent = Path(study_root).expanduser().resolve() / "shards"
        if parent.is_dir():
            candidates.extend(path.parent for path in sorted(parent.rglob("study.resolved.json")))
    candidates.extend(Path(path).expanduser().resolve() for path in shard_roots)
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        root = candidate.resolve()
        if root in seen:
            continue
        if not root.is_dir() or not (root / "study.resolved.json").is_file():
            raise ValueError(f"Invalid directed task shard root: {root}.")
        result.append(root)
        seen.add(root)
    for index, left in enumerate(result):
        for right in result[index + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError(f"Shard roots are not disjoint: {left} and {right}.")
    return result


def _validate_resolved(root: Path, contract: StudyContract) -> Shard:
    resolved = _load_json(root / "study.resolved.json")
    expected = {
        "schema_version": RESOLVED_SCHEMA,
        "study_id": contract.study_id,
        "evidence_class": "ordinary_mnist_exploratory",
        "study_config_sha256": contract.sha256,
        "parent_study_config_sha256": sha256_file(contract.parent_path),
        "initializer_checkpoint_sha256_by_architecture": EXPECTED_INITIALIZER_SHA256,
        "surface_count": EXPECTED_SURFACE_COUNT,
        "total_cells": EXPECTED_CELL_COUNT,
        "surfaces": list(contract.surfaces),
        "official_test_read": False,
        "restart_interrupted_cells": True,
    }
    for key, value in expected.items():
        if resolved.get(key) != value:
            raise ValueError(f"Resolved shard {root} has invalid {key}.")
    if not _path_has_suffix(
        resolved.get("study_config"), ("configs", "conv", contract.path.name)
    ) or not _path_has_suffix(
        resolved.get("parent_study_config"),
        ("configs", "conv", contract.parent_path.name),
    ):
        raise ValueError(f"Resolved shard {root} has noncanonical config paths.")
    if resolved.get("device") != "cuda":
        raise ValueError(f"Resolved shard {root} is not a CUDA execution.")
    target = resolved.get("target")
    if target not in {"main", "akib", "jean-zay"}:
        raise ValueError(f"Resolved shard {root} has unsupported target {target!r}.")
    code = resolved.get("code")
    if not isinstance(code, Mapping):
        raise ValueError(f"Resolved shard {root} lacks source identity.")
    commit = code.get("commit")
    archive = code.get("source_archive_sha256")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
        or not isinstance(archive, str)
        or len(archive) != 64
        or any(char not in "0123456789abcdef" for char in archive)
        or code.get("source_kind") != "frozen_archive"
        or code.get("working_tree_sha256") != archive
        or code.get("dirty") is not False
    ):
        raise ValueError(f"Resolved shard {root} does not prove one frozen source archive.")

    expected_ids = {str(surface["surface_id"]): surface for surface in contract.surfaces}
    surfaces_root = root / "surfaces"
    directories = sorted(path for path in surfaces_root.iterdir() if path.is_dir()) if surfaces_root.is_dir() else []
    known = [path for path in directories if path.name in expected_ids]
    unknown = [path.name for path in directories if path.name not in expected_ids]
    if len(known) != 1 or unknown:
        raise ValueError(
            f"Expected exactly one declared surface in {root}; "
            f"known={[path.name for path in known]!r}, unknown={unknown!r}."
        )
    surface = expected_ids[known[0].name]
    task_match = TASK_ROOT_PATTERN.fullmatch(root.name)
    if task_match is not None and int(task_match.group("task")) != int(surface["index"]):
        raise ValueError(f"Task-root name does not match its surface under {root}.")
    declared_target = str(surface["target"])
    expected_runtime_target = (
        "jean-zay" if declared_target.startswith("jean-zay-") else declared_target
    )
    if target != expected_runtime_target:
        raise ValueError(
            f"Resolved shard {root} target {target!r} does not match declared "
            f"surface target {declared_target!r}."
        )
    return Shard(root=root, surface=surface, resolved=resolved, target=str(target))


def _load_shards(roots: Sequence[Path], contract: StudyContract) -> list[Shard]:
    shards = [_validate_resolved(root, contract) for root in roots]
    indices = [int(shard.surface["index"]) for shard in shards]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate directed surface authority: {indices!r}.")
    source_identities = {
        (
            shard.resolved["code"]["commit"],
            shard.resolved["code"]["source_archive_sha256"],
        )
        for shard in shards
    }
    if len(source_identities) > 1:
        raise ValueError("Collected shards do not share one frozen source identity.")
    return sorted(shards, key=lambda shard: int(shard.surface["index"]))


def _receipt_search_roots(
    *, study_root: Path | None, receipt_roots: Sequence[Path]
) -> list[Path]:
    roots = [Path(path).expanduser().resolve() for path in receipt_roots]
    if study_root is not None:
        roots.append(Path(study_root).expanduser().resolve() / "transport_receipts")
    result: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            result.append(root)
            seen.add(root)
    return result


def _receipt_inventory(
    contract: StudyContract,
    *,
    study_root: Path | None,
    receipt_roots: Sequence[Path],
) -> dict[int, list[tuple[Path, dict[str, Any]]]]:
    inventory = {int(surface["index"]): [] for surface in contract.surfaces}
    paths: list[Path] = []
    for root in _receipt_search_roots(
        study_root=study_root, receipt_roots=receipt_roots
    ):
        if root.is_file():
            paths.append(root)
        elif root.is_dir():
            paths.extend(sorted(root.rglob("*.json")))
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        if path in seen:
            continue
        seen.add(path)
        receipt = _load_json(path)
        if receipt.get("schema_version") != RECEIPT_SCHEMA:
            continue
        if receipt.get("study_id") != contract.study_id:
            continue
        task = receipt.get("task_id")
        if isinstance(task, bool) or not isinstance(task, int) or task not in inventory:
            raise ValueError(f"Transport receipt {path} has invalid task_id {task!r}.")
        inventory[task].append((path, receipt))
    for task, records in inventory.items():
        if len(records) > 1:
            raise ValueError(f"Duplicate transport receipts for task {task}: {records!r}.")
    return inventory


def _validate_receipt(
    shard: Shard,
    contract: StudyContract,
    *,
    summary: Mapping[str, Any],
    record: tuple[Path, dict[str, Any]],
) -> dict[str, Any]:
    path, receipt = record
    surface = shard.surface
    index = int(surface["index"])
    architecture = str(surface["architecture"])
    expected_cells = len(surface_cells(contract.payload, surface))
    required = {
        "schema_version": RECEIPT_SCHEMA,
        "study_id": contract.study_id,
        "surface": surface,
        "task_id": index,
        "target": shard.target,
        "study_config_sha256": contract.sha256,
        "initializer_checkpoint_sha256": EXPECTED_INITIALIZER_SHA256[architecture],
        "expected_cells": expected_cells,
        "official_test_read": False,
        "semantic_pass": True,
        "summary_status": summary.get("status"),
        "terminal_cells": summary.get("terminal_cells"),
        "status_counts": summary.get("status_counts"),
    }
    for key, value in required.items():
        if receipt.get(key) != value:
            raise ValueError(f"Transport receipt {path} has invalid {key}.")
    if path.name != f"{shard.target}-task_{index}.json":
        raise ValueError(f"Transport receipt {path} has a noncanonical filename.")
    code = shard.resolved["code"]
    if (
        receipt.get("source_commit") != code.get("commit")
        or receipt.get("source_archive_sha256") != code.get("source_archive_sha256")
    ):
        raise ValueError(f"Transport receipt {path} does not bind the shard source.")
    environment_id = receipt.get("environment_id")
    if not isinstance(environment_id, str) or not environment_id:
        raise ValueError(f"Transport receipt {path} lacks environment_id.")
    summary_path = shard.root / "surfaces" / str(surface["surface_id"]) / "summary.json"
    if (
        not _path_has_suffix(
            receipt.get("summary_path"),
            (
                "shards",
                shard.root.name,
                "surfaces",
                str(surface["surface_id"]),
                "summary.json",
            ),
        )
        or receipt.get("summary_sha256") != sha256_file(summary_path)
    ):
        raise ValueError(f"Transport receipt {path} does not bind the local summary.")
    if receipt.get("best_observation") != summary.get("best_observation"):
        raise ValueError(f"Transport receipt {path} has a mismatched best observation.")
    state = str(summary.get("status"))
    validated = receipt.get("validated_cell_bundles")
    if state == "complete":
        if validated != expected_cells:
            raise ValueError(f"Receipt {path} does not validate every directed cell.")
    elif state == "unresolved_probe":
        if validated != 0 or summary.get("terminal_cells") != 0:
            raise ValueError(f"Malformed unresolved-probe receipt {path}.")
    else:
        raise ValueError(f"Receipt {path} binds unsupported state {state!r}.")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "task_id": index,
        "surface_id": surface["surface_id"],
        "target": shard.target,
        "environment_id": environment_id,
        "source_commit": receipt["source_commit"],
        "source_archive_sha256": receipt["source_archive_sha256"],
        "summary_status": state,
        "validated_cell_bundles": validated,
    }


def _verify_initializer(shard: Shard) -> None:
    architecture = str(shard.surface["architecture"])
    expected = EXPECTED_INITIALIZER_SHA256[architecture]
    asset_root = shard.root / "assets" / "bounded_uniform" / architecture
    asset_path = asset_root / "asset.json"
    checkpoint = asset_root / "final_model.pt"
    asset = _load_json(asset_path)
    required = {
        "architecture": architecture,
        "initializer": "bounded_uniform",
        "checkpoint_sha256": expected,
        "official_test_read": False,
    }
    for key, value in required.items():
        if asset.get(key) != value:
            raise ValueError(f"Initializer asset {asset_path} has invalid {key}.")
    if not _path_has_suffix(
        asset.get("checkpoint"),
        (
            shard.root.name,
            "assets",
            "bounded_uniform",
            architecture,
            "final_model.pt",
        ),
    ):
        raise ValueError(f"Initializer asset path does not bind shard {shard.root}.")
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected:
        raise ValueError(f"Initializer checkpoint bytes do not match in {shard.root}.")
    zero = asset.get("zero_bias_checkpoint_verification")
    if (
        not isinstance(zero, Mapping)
        or zero.get("all_exact_zero") is not True
        or zero.get("nonzero_bias_element_count") != 0
        or sorted(zero.get("expected_bias_names", [])) != list(_bias_names(architecture))
    ):
        raise ValueError(f"Initializer zero-bias verification failed in {shard.root}.")


def _verify_source_config(shard: Shard, contract: StudyContract) -> tuple[dict[str, Any], str]:
    surface = shard.surface
    architecture = str(surface["architecture"])
    source_path = (
        shard.root / "surfaces" / str(surface["surface_id"]) / "source_config.json"
    )
    source = _load_json(source_path)
    init_path = source.get("init_checkpoint_path")
    if not _path_has_suffix(
        init_path,
        (
            shard.root.name,
            "assets",
            "bounded_uniform",
            architecture,
            "final_model.pt",
        ),
    ):
        raise ValueError(f"Source config {source_path} does not bind its initializer.")
    datasets = source.get("datasets")
    mnist = datasets.get("mnist") if isinstance(datasets, Mapping) else None
    params = mnist.get("params") if isinstance(mnist, Mapping) else None
    dataset_root = params.get("root") if isinstance(params, Mapping) else None
    if not isinstance(dataset_root, str) or not dataset_root:
        raise ValueError(f"Source config {source_path} lacks its MNIST root.")
    expected = build_source_config(
        contract.parent,
        initializer="bounded_uniform",
        architecture=architecture,
        scheme=str(surface["scheme"]),
        optimizer=str(surface["optimizer"]),
        init_checkpoint_path=Path(str(init_path)),
        dataset_root=Path(dataset_root),
    )
    if source != expected:
        raise ValueError(f"Source config {source_path} differs from its governed contract.")
    return source, sha256_file(source_path)


def _verify_probe(
    shard: Shard,
    contract: StudyContract,
    *,
    source_sha256: str,
    required_complete: bool,
) -> tuple[dict[str, Any], str, str] | None:
    surface_dir = shard.root / "surfaces" / str(shard.surface["surface_id"])
    resolved_path = surface_dir / "rho" / "resolved.json"
    probe_path = surface_dir / "rho" / "probe.json"
    if not resolved_path.is_file() and not probe_path.is_file() and not required_complete:
        return None
    if not resolved_path.is_file() or not probe_path.is_file():
        raise ValueError(f"Surface probe provenance is incomplete under {surface_dir}.")
    resolved = _load_json(resolved_path)
    probe = _load_json(probe_path)
    search = probe.get("search_signature")
    if not isinstance(search, Mapping) or search != resolved:
        raise ValueError(f"Probe/resolved search signatures disagree in {surface_dir}.")
    expected_cells = surface_cells(contract.payload, shard.surface)
    conv_axis = sorted({float(cell["rho_conv"]) for cell in expected_cells})
    dense_axis = sorted({float(cell["rho_dense"]) for cell in expected_cells})
    expected = {
        "schema_version": "conv-rho-search/v1",
        "source_config_sha256": source_sha256,
        "code": shard.resolved["code"],
        "optimizer": shard.surface["optimizer"],
        "evidence_class": "ordinary_mnist_exploratory",
        "rho_conv": conv_axis,
        "rho_dense": dense_axis,
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
    for key, value in expected.items():
        if search.get(key) != value:
            raise ValueError(f"Probe search signature has invalid {key}: {probe_path}.")
    safety = search.get("safety")
    if (
        not isinstance(safety, Mapping)
        or safety.get("rejections_enabled") is not False
        or safety.get("bound_occupancy") != "report_only"
        or safety.get("projection_efficiency") != "report_only"
        or safety.get("bound_occupancy_increase_maximum") is not None
        or safety.get("projection_efficiency_minimum") is not None
    ):
        raise ValueError(f"Probe safety contract is not report-only: {probe_path}.")
    architecture = str(shard.surface["architecture"])
    weights = _weight_names(architecture)
    biases = _bias_names(architecture)
    if probe.get("schema_version") != "conv-rho-probe/v1":
        raise ValueError(f"Unexpected optimizer-probe schema: {probe_path}.")
    if probe.get("official_test_read") is not False:
        raise ValueError(f"Optimizer probe does not prove official_test_read=false: {probe_path}.")
    if required_complete and (
        probe.get("status") != "complete"
        or probe.get("probe_stable") is not True
        or probe.get("proposal_units_valid") is not True
    ):
        raise ValueError(f"Optimizer probe is not resolved: {probe_path}.")
    if probe.get("status") == "complete":
        if (
            probe.get("parameter_names") != list(weights + biases)
            or probe.get("weight_names") != list(weights)
        ):
            raise ValueError(f"Optimizer probe parameter order is invalid: {probe_path}.")
        units = probe.get("normalization_unit_by_weight")
        if not isinstance(units, Mapping) or set(units) != set(weights):
            raise ValueError(f"Optimizer probe normalization units are invalid: {probe_path}.")
        if any(_finite_float(units[name], label=f"probe unit {name}") <= 0 for name in weights):
            raise ValueError(f"Optimizer probe contains a nonpositive unit: {probe_path}.")
        zero = probe.get("zero_bias_checkpoint_verification")
        _verify_zero_bias_record(
            zero,
            architecture=architecture,
            label=str(probe_path),
        )
        initial_sha = probe.get("initial_parameter_sha256")
        if (
            not isinstance(initial_sha, str)
            or len(initial_sha) != 64
            or any(character not in "0123456789abcdef" for character in initial_sha)
        ):
            raise ValueError(f"Optimizer probe has invalid initial-parameter SHA: {probe_path}.")
    semantic_sha256 = hashlib.sha256(
        json.dumps(probe, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return probe, semantic_sha256, sha256_file(probe_path)


def _indexed_checkpoint(run_dir: Path, result: Mapping[str, Any], name: str) -> Path:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"Result artifacts are missing in {run_dir}.")
    records = [
        record
        for record in artifacts
        if isinstance(record, Mapping)
        and record.get("kind") == "checkpoint"
        and Path(str(record.get("path", ""))).name == name
    ]
    if len(records) != 1:
        raise ValueError(f"Expected one indexed {name} in {run_dir}.")
    relative = Path(str(records[0].get("path", "")))
    if relative.is_absolute():
        raise ValueError(f"Indexed checkpoint path must be relative in {run_dir}.")
    checkpoint = (run_dir / relative).resolve()
    try:
        checkpoint.relative_to(run_dir.resolve())
    except ValueError as error:
        raise ValueError(f"Indexed checkpoint escapes bundle: {checkpoint}.") from error
    if not checkpoint.is_file():
        raise ValueError(f"Missing indexed checkpoint {checkpoint}.")
    return checkpoint


def _group_counts(lower: int, upper: int, either: int, count: int) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("Bounded weight group is empty.")
    return {
        "lower_count": lower,
        "upper_count": upper,
        "either_count": either,
        "weight_count": count,
        "either_fraction": either / count,
        "either_percent": 100.0 * either / count,
    }


def _checkpoint_endpoint_diagnostics(
    checkpoint: Path,
    *,
    architecture: str,
    weight_min: float = WEIGHT_MIN,
    weight_max: float = WEIGHT_MAX,
    reported_final_occupancy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute exact endpoint counts using bounds cast to each array dtype.

    Casting is essential: a float32 checkpoint's exact representation of
    ``1e-5`` is slightly below the Python float, and comparing it to the latter
    would incorrectly classify a valid lower-bound value as out of range.
    """

    weights = _weight_names(architecture)
    biases = _bias_names(architecture)
    totals = {scope: [0, 0, 0, 0] for scope in ("all", "conv", "dense")}
    per_parameter: dict[str, dict[str, Any]] = {}
    biases_exact_zero = True
    with np.load(checkpoint, allow_pickle=False) as payload:
        for name in weights:
            if name not in payload.files:
                raise ValueError(f"Checkpoint {checkpoint} omits {name}.")
            values = np.asarray(payload[name])
            if values.size == 0 or not np.issubdtype(values.dtype, np.floating):
                raise ValueError(f"Checkpoint weight {name} has invalid dtype/size.")
            if not np.isfinite(values).all():
                raise ValueError(f"Checkpoint weight {name} is non-finite.")
            lower_bound = np.asarray(weight_min, dtype=values.dtype).item()
            upper_bound = np.asarray(weight_max, dtype=values.dtype).item()
            if not lower_bound < upper_bound:
                raise ValueError(f"Bounds collapse in dtype {values.dtype} for {name}.")
            if np.any((values < lower_bound) | (values > upper_bound)):
                observed_min = float(np.min(values))
                observed_max = float(np.max(values))
                raise ValueError(
                    f"Checkpoint weight {name} leaves the dtype-cast interval: "
                    f"min={observed_min}, max={observed_max}."
                )
            lower = int(np.count_nonzero(values <= lower_bound))
            upper = int(np.count_nonzero(values >= upper_bound))
            either = int(np.count_nonzero(
                (values <= lower_bound) | (values >= upper_bound)
            ))
            count = int(values.size)
            record = _group_counts(lower, upper, either, count)
            record.update(
                {
                    "dtype": str(values.dtype),
                    "dtype_lower_bound": float(lower_bound),
                    "dtype_upper_bound": float(upper_bound),
                }
            )
            per_parameter[name] = record
            scope = "conv" if name.startswith("ConvWeight_") else "dense"
            for target in ("all", scope):
                totals[target][0] += lower
                totals[target][1] += upper
                totals[target][2] += either
                totals[target][3] += count
            if reported_final_occupancy is not None:
                if name not in reported_final_occupancy:
                    raise ValueError(f"Reported final occupancy omits {name}.")
                reported = _fraction(
                    reported_final_occupancy[name], label=f"reported occupancy {name}"
                )
                if not math.isclose(
                    reported, record["either_fraction"], rel_tol=1e-12, abs_tol=1e-15
                ):
                    raise ValueError(
                        f"Reported/recomputed endpoint occupancy differs for {name}."
                    )
        if reported_final_occupancy is not None and set(reported_final_occupancy) != set(weights):
            raise ValueError("Reported final occupancy has unexpected parameters.")
        for name in biases:
            if name not in payload.files or not np.all(np.asarray(payload[name]) == 0):
                biases_exact_zero = False
    result = {
        "all": _group_counts(*totals["all"]),
        "conv": _group_counts(*totals["conv"]),
        "dense": _group_counts(*totals["dense"]),
        "by_parameter": per_parameter,
        "biases_exact_zero": biases_exact_zero,
    }
    return result


def _flatten_endpoint(prefix: str, diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for source_scope, output_scope in (("all", ""), ("conv", "conv_"), ("dense", "dense_")):
        record = diagnostics[source_scope]
        for key in (
            "lower_count",
            "upper_count",
            "either_count",
            "weight_count",
            "either_fraction",
            "either_percent",
        ):
            row[f"{prefix}_{output_scope}{key}"] = record[key]
    row[f"{prefix}_by_parameter"] = diagnostics["by_parameter"]
    row[f"{prefix}_by_parameter_json"] = _stable_json(diagnostics["by_parameter"])
    row[f"{prefix}_biases_exact_zero"] = diagnostics["biases_exact_zero"]
    return row


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
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denominator == 0.0:
        return None
    return min(1.0, max(-1.0, float(np.dot(x, y) / denominator)))


def correlation_statistics(
    values_x: Sequence[float], values_y: Sequence[float]
) -> dict[str, Any]:
    """Return paired Pearson/Spearman statistics with tie-aware ranks."""

    if len(values_x) != len(values_y):
        raise ValueError("Correlation vectors have different lengths.")
    return {
        "sample_count": len(values_x),
        "pearson": _pearson(values_x, values_y),
        "spearman": (
            None
            if len(values_x) < 2
            else _pearson(_average_ranks(values_x), _average_ranks(values_y))
        ),
    }


def _classify_cell(
    raw_status: Any,
    safety_failure: Any,
    *,
    reporting_state: Any,
    result_present: bool,
) -> str:
    if raw_status == "running":
        if reporting_state != "running" or result_present:
            raise ValueError("A running cell must be a canonical running bundle.")
        return "running"
    if raw_status == "complete":
        if reporting_state != "complete" or not result_present:
            raise ValueError("A numeric cell must be a canonical complete bundle.")
        return "numeric_complete"
    expected_kind = {
        "candidate_rejected_nonfinite": {"nonfinite_training_value"},
        "candidate_rejected_safety": {
            "nonfinite_diagnostic",
            "nonfinite_training_value",
        },
    }.get(raw_status)
    kind = safety_failure.get("kind") if isinstance(safety_failure, Mapping) else None
    if expected_kind is None or kind not in expected_kind:
        raise ValueError(f"Unexpected terminal exploratory cell status {raw_status!r}.")
    if reporting_state != "failed" or result_present:
        raise ValueError("A non-finite cell must be a canonical failed bundle.")
    return "nonfinite"


def _verify_zero_bias_record(value: Any, *, architecture: str, label: str) -> None:
    if not isinstance(value, Mapping) or value.get("all_exact_zero") is not True:
        raise ValueError(f"Missing exact zero-bias verification in {label}.")
    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 2:
        raise ValueError(f"Expected best/final zero-bias checkpoint records in {label}.")
    expected = list(_bias_names(architecture))
    for checkpoint in checkpoints:
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("all_exact_zero") is not True
            or checkpoint.get("nonzero_bias_element_count") != 0
            or sorted(checkpoint.get("expected_bias_names", [])) != expected
        ):
            raise ValueError(f"Invalid zero-bias checkpoint record in {label}.")


def _verify_official_test_unread(
    manifest: Mapping[str, Any],
    result: Mapping[str, Any] | None,
    metrics: Mapping[str, Any] | None,
) -> None:
    dataset = manifest.get("dataset")
    if (
        manifest.get("evidence_class") != "ordinary_mnist_exploratory"
        or manifest.get("smoke") is not False
        or not isinstance(dataset, Mapping)
        or dataset.get("evaluation_split") != "validation"
        or dataset.get("official_test_read") is not False
    ):
        raise ValueError("Manifest does not prove validation-only ordinary MNIST use.")
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
            raise ValueError("Result does not prove official_test_read=false.")
    if metrics is not None and (
        metrics.get("official_test_accuracy") is not None
        or metrics.get("official_test_loss") is not None
        or metrics.get("official_test_examples") != 0
        or metrics.get("official_test_evaluations") != 0
    ):
        raise ValueError("Metrics contain an official-test evaluation.")


def _manifest_input(
    manifest: Mapping[str, Any], role: str, *, run_dir: Path
) -> Mapping[str, Any]:
    values = manifest.get("inputs")
    records = [
        value
        for value in values if isinstance(value, Mapping) and value.get("role") == role
    ] if isinstance(values, list) else []
    if len(records) != 1:
        raise ValueError(f"Expected one manifest input {role!r} in {run_dir}.")
    return records[0]


def _read_cell(
    shard: Shard,
    contract: StudyContract,
    *,
    cell_dir: Path,
    expected: Mapping[str, Any],
    source_config: Mapping[str, Any],
    source_sha256: str,
    probe_record: tuple[dict[str, Any], str, str],
    transport_validated: bool,
) -> dict[str, Any]:
    errors = validate_run(cell_dir)
    if errors:
        raise ValueError(f"Invalid canonical bundle {cell_dir}: {errors!r}.")
    cell = _load_json(cell_dir / "cell.json")
    manifest = _load_json(cell_dir / "manifest.json")
    status = _load_json(cell_dir / "status.json")
    result_path = cell_dir / "result.json"
    result = _load_json(result_path) if result_path.is_file() else None
    metrics_path = cell_dir / "metrics.json"
    metrics = _load_json(metrics_path) if metrics_path.is_file() else None
    diagnostics_path = cell_dir / "safety_diagnostics.json"
    diagnostics = _load_json(diagnostics_path) if diagnostics_path.is_file() else None
    outcome = _classify_cell(
        cell.get("status"),
        cell.get("safety_failure"),
        reporting_state=status.get("state"),
        result_present=result is not None,
    )
    if manifest.get("study_id") != contract.study_id or manifest.get("run_id") != cell_dir.name:
        raise ValueError(f"Manifest identity mismatch in {cell_dir}.")
    if status.get("study_id") != contract.study_id or status.get("run_id") != cell_dir.name:
        raise ValueError(f"Status identity mismatch in {cell_dir}.")
    if result is not None and status.get("result_sha256") != sha256_file(result_path):
        raise ValueError(f"Status does not bind result.json in {cell_dir}.")

    surface = shard.surface
    architecture = str(surface["architecture"])
    expected_index = int(expected["index"])
    if (
        cell.get("index") != expected_index
        or cell.get("optimizer") != surface["optimizer"]
        or cell.get("bias_policy") != "zero"
        or cell.get("evidence_class") != "ordinary_mnist_exploratory"
        or not _same_float(cell.get("rho_conv"), expected["rho_conv"])
        or not _same_float(cell.get("rho_dense"), expected["rho_dense"])
    ):
        raise ValueError(f"Sparse coordinate/cell identity mismatch in {cell_dir}.")
    signature = cell.get("signature")
    if not isinstance(signature, Mapping):
        raise ValueError(f"Missing cell signature in {cell_dir}.")
    required_signature = {
        "optimizer": surface["optimizer"],
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
        "code": shard.resolved["code"],
    }
    for key, value in required_signature.items():
        if signature.get(key) != value:
            raise ValueError(f"Cell signature has invalid {key} in {cell_dir}.")
    if (
        not _same_float(signature.get("rho_conv"), expected["rho_conv"])
        or not _same_float(signature.get("rho_dense"), expected["rho_dense"])
    ):
        raise ValueError(f"Cell signature sparse coordinate mismatch in {cell_dir}.")
    probe, probe_semantic_sha256, probe_file_sha256 = probe_record
    if (
        signature.get("probe_sha256") != probe_semantic_sha256
        or signature.get("safety") != probe.get("search_signature", {}).get("safety")
    ):
        raise ValueError(f"Cell does not bind its optimizer probe in {cell_dir}.")

    parameter_names = list(_weight_names(architecture) + _bias_names(architecture))
    if probe.get("parameter_names") != parameter_names:
        raise ValueError(f"Probe parameter order disagrees in {cell_dir}.")
    raw_rates = cell.get("learning_rates_by_parameter")
    vector = cell.get("learning_rate_vector")
    if (
        not isinstance(raw_rates, Mapping)
        or set(raw_rates) != set(parameter_names)
        or not isinstance(vector, list)
        or len(vector) != len(parameter_names)
    ):
        raise ValueError(f"Learning-rate record is malformed in {cell_dir}.")
    rates = {
        name: _finite_float(raw_rates[name], label=f"learning rate {name}")
        for name in parameter_names
    }
    finite_vector = [_finite_float(value, label="learning-rate vector") for value in vector]
    if any(not _same_float(rates[name], finite_vector[index]) for index, name in enumerate(parameter_names)):
        raise ValueError(f"Named/vector learning rates disagree in {cell_dir}.")
    units = probe.get("normalization_unit_by_weight")
    assert isinstance(units, Mapping)  # established by _verify_probe
    for name in _weight_names(architecture):
        rho = expected["rho_conv"] if name.startswith("ConvWeight_") else expected["rho_dense"]
        if not _same_float(rates[name], float(rho) / float(units[name])):
            raise ValueError(f"Rho-derived learning rate differs for {name} in {cell_dir}.")
    if any(rates[name] != 0.0 for name in _bias_names(architecture)):
        raise ValueError(f"Cell has a nonzero bias learning rate in {cell_dir}.")

    candidate_path = cell_dir / "source_config.json"
    candidate = _load_json(candidate_path)
    expected_candidate = prepare_training_config(
        source_config,
        str(surface["optimizer"]),
        finite_vector,
        epochs=3,
        max_batches=None,
        max_validation_batches=None,
        split_seed=0,
        shuffle_seed=0,
        validation_batch_size=64,
    )
    configuration = manifest.get("configuration")
    if (
        candidate != expected_candidate
        or _load_json(cell_dir / "config.used.json") != expected_candidate
        or not isinstance(configuration, Mapping)
        or configuration.get("sha256") != sha256_file(candidate_path)
        or configuration.get("resolved") != expected_candidate
        or configuration.get("optimizer") != surface["optimizer"]
        or configuration.get("learning_rates") != finite_vector
        or not _same_float(configuration.get("rho_conv"), expected["rho_conv"])
        or not _same_float(configuration.get("rho_dense"), expected["rho_dense"])
        or manifest.get("git") != shard.resolved["code"]
    ):
        raise ValueError(f"Manifest/candidate scientific configuration mismatch in {cell_dir}.")
    model = candidate.get("model_base")
    expected_t, expected_k = EXPECTED_TK[architecture]
    if (
        not isinstance(model, Mapping)
        or not _same_float(model.get("weight_min"), WEIGHT_MIN)
        or not _same_float(model.get("weight_max"), WEIGHT_MAX)
        or model.get("num_iterations_inference") != expected_t
        or model.get("num_iterations_training") != expected_k
        or candidate.get("training_algorithm") != "BP"
    ):
        raise ValueError(f"Cell model/T/K/bounds contract mismatch in {cell_dir}.")
    if (
        _manifest_input(manifest, "source_config", run_dir=cell_dir).get("sha256")
        != source_sha256
        or _manifest_input(manifest, "optimizer_probe", run_dir=cell_dir).get("sha256")
        != probe_file_sha256
    ):
        raise ValueError(f"Manifest source/probe hashes disagree in {cell_dir}.")
    runtime = manifest.get("runtime")
    command = manifest.get("command")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("target") != shard.target
        or status.get("runtime") != runtime
        or not isinstance(command, Mapping)
        or command.get("surface_index") != surface["index"]
        or command.get("cell_index") != expected_index
    ):
        raise ValueError(f"Cell runtime/command identity mismatch in {cell_dir}.")

    _verify_official_test_unread(manifest, result, metrics)
    row: dict[str, Any] = {
        "study_id": contract.study_id,
        "surface_index": surface["index"],
        "surface_id": surface["surface_id"],
        "architecture": architecture,
        "scheme": surface["scheme"],
        "optimizer": surface["optimizer"],
        "target": shard.target,
        "cell_index": expected_index,
        "execution_rank": expected["execution_rank"],
        "rho_conv_ladder_index": expected["rho_conv_ladder_index"],
        "rho_dense_ladder_index": expected["rho_dense_ladder_index"],
        "rho_conv": float(expected["rho_conv"]),
        "rho_dense": float(expected["rho_dense"]),
        "raw_status": cell["status"],
        "canonical_outcome": outcome,
        "transport_validated": transport_validated,
        "learning_rates_by_parameter": rates,
        "learning_rates_by_parameter_json": _stable_json(rates),
        "canonical_bundle_valid": True,
        "official_test_read": False,
        "run_dir": str(cell_dir),
        "final_validation_accuracy": None,
        "final_validation_accuracy_percent": None,
        "best_validation_accuracy": None,
        "best_validation_accuracy_percent": None,
        "final_validation_loss": None,
        "best_epoch": None,
    }
    for endpoint in ("final", "best"):
        row.update(
            {
                f"{endpoint}_lower_count": None,
                f"{endpoint}_upper_count": None,
                f"{endpoint}_either_count": None,
                f"{endpoint}_weight_count": None,
                f"{endpoint}_either_fraction": None,
                f"{endpoint}_either_percent": None,
                f"{endpoint}_conv_lower_count": None,
                f"{endpoint}_conv_upper_count": None,
                f"{endpoint}_conv_either_count": None,
                f"{endpoint}_conv_weight_count": None,
                f"{endpoint}_conv_either_fraction": None,
                f"{endpoint}_conv_either_percent": None,
                f"{endpoint}_dense_lower_count": None,
                f"{endpoint}_dense_upper_count": None,
                f"{endpoint}_dense_either_count": None,
                f"{endpoint}_dense_weight_count": None,
                f"{endpoint}_dense_either_fraction": None,
                f"{endpoint}_dense_either_percent": None,
                f"{endpoint}_by_parameter": {},
                f"{endpoint}_by_parameter_json": "{}",
            }
        )
    if outcome == "numeric_complete":
        if metrics is None or diagnostics is None or result is None:
            raise ValueError(f"Numeric cell lacks metrics/diagnostics/result: {cell_dir}.")
        if cell.get("completed_steps") != 10314 or diagnostics.get("processed_steps") != 10314:
            raise ValueError(f"Numeric cell did not complete 10,314 steps: {cell_dir}.")
        if cell.get("selection_eligible") is not True:
            raise ValueError(f"Numeric cell is not selection-eligible: {cell_dir}.")
        _verify_zero_bias_record(
            cell.get("zero_bias_checkpoint_verification"),
            architecture=architecture,
            label=str(cell_dir / "cell.json"),
        )
        _verify_zero_bias_record(
            diagnostics.get("zero_bias_checkpoint_verification"),
            architecture=architecture,
            label=str(diagnostics_path),
        )
        report_only = diagnostics.get("report_only_diagnostics")
        gates = diagnostics.get("terminal_gates")
        if (
            diagnostics.get("schema_version") != "conv-rho-training-safety/v1"
            or report_only
            != {
                "bound_occupancy": True,
                "projection_efficiency": True,
                "used_for_rejection": False,
            }
            or not isinstance(gates, Mapping)
            or gates.get("scientific_rejections_enabled") is not False
            or gates.get("bound_occupancy_increase_maximum") is not None
            or gates.get("projection_efficiency_minimum") is not None
        ):
            raise ValueError(f"Cell diagnostics are not report-only in {cell_dir}.")
        final_accuracy = _fraction(
            metrics.get("final_validation_accuracy", metrics.get("final_test_accuracy")),
            label="final validation accuracy",
        )
        best_accuracy = _fraction(
            metrics.get("best_validation_accuracy", metrics.get("best_test_accuracy")),
            label="best validation accuracy",
        )
        final_loss = _finite_float(
            metrics.get("final_validation_loss", metrics.get("final_test_loss")),
            label="final validation loss",
        )
        if final_loss < 0:
            raise ValueError(f"Negative validation loss in {cell_dir}.")
        terminal = result.get("terminal_metrics")
        validation = terminal.get("validation") if isinstance(terminal, Mapping) else None
        completion = result.get("completion")
        if (
            result.get("schema_version") != "experiment-run-result/v1"
            or not isinstance(validation, Mapping)
            or not _same_float(validation.get("final_accuracy"), final_accuracy)
            or not _same_float(validation.get("final_loss"), final_loss)
            or not isinstance(completion, Mapping)
            or completion.get("criteria_met") is not True
            or completion.get("rho_cell_complete") is not True
        ):
            raise ValueError(f"Result terminal metrics/completion mismatch in {cell_dir}.")
        occupancy = diagnostics.get("final_bound_occupancy_by_parameter")
        if not isinstance(occupancy, Mapping):
            raise ValueError(f"Final occupancy map is missing in {cell_dir}.")
        final_diagnostics = _checkpoint_endpoint_diagnostics(
            _indexed_checkpoint(cell_dir, result, "weights_final.npz"),
            architecture=architecture,
            reported_final_occupancy=occupancy,
        )
        best_diagnostics = _checkpoint_endpoint_diagnostics(
            _indexed_checkpoint(cell_dir, result, "weights_best.npz"),
            architecture=architecture,
        )
        if (
            final_diagnostics["biases_exact_zero"] is not True
            or best_diagnostics["biases_exact_zero"] is not True
        ):
            raise ValueError(f"Endpoint checkpoint has nonzero biases in {cell_dir}.")
        row.update(_flatten_endpoint("final", final_diagnostics))
        row.update(_flatten_endpoint("best", best_diagnostics))
        row.update(
            {
                "final_validation_accuracy": final_accuracy,
                "final_validation_accuracy_percent": 100.0 * final_accuracy,
                "best_validation_accuracy": best_accuracy,
                "best_validation_accuracy_percent": 100.0 * best_accuracy,
                "final_validation_loss": final_loss,
                "best_epoch": metrics.get("best_epoch"),
            }
        )
    elif diagnostics is not None and diagnostics.get("safety_failure") != cell.get("safety_failure"):
        raise ValueError(f"Non-finite safety records disagree in {cell_dir}.")
    return row


def _validate_observed_cell_identity(
    directory: Path,
    expected_by_index: Mapping[int, Mapping[str, Any]],
) -> tuple[int, dict[str, Any]]:
    cell = _load_json(directory / "cell.json")
    index = cell.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index not in expected_by_index:
        raise ValueError(f"Undeclared sparse cell index {index!r} in {directory}.")
    expected = expected_by_index[index]
    if (
        not _same_float(cell.get("rho_conv"), expected["rho_conv"])
        or not _same_float(cell.get("rho_dense"), expected["rho_dense"])
    ):
        raise ValueError(f"Sparse cell coordinate mismatch in {directory}.")
    return index, cell


def _read_surface(
    shard: Shard,
    contract: StudyContract,
    *,
    receipt_record: tuple[Path, dict[str, Any]] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    surface = shard.surface
    surface_dir = shard.root / "surfaces" / str(surface["surface_id"])
    status_path = surface_dir / "status.json"
    summary_path = surface_dir / "summary.json"
    status = _load_json(status_path)
    summary = _load_json(summary_path)
    expected_cells = surface_cells(contract.payload, surface)
    expected_by_index = {int(cell["index"]): cell for cell in expected_cells}
    expected_count = len(expected_cells)
    if len(expected_by_index) != expected_count:
        raise ValueError(f"Declared sparse coordinates collide for {surface['surface_id']}.")
    required_status = {
        "schema_version": SURFACE_STATUS_SCHEMA,
        "study_id": contract.study_id,
        "surface_id": surface["surface_id"],
        "official_test_read": False,
    }
    for key, value in required_status.items():
        if status.get(key) != value:
            raise ValueError(f"Surface status {status_path} has invalid {key}.")
    state = status.get("status")
    required_summary = {
        "schema_version": SURFACE_SUMMARY_SCHEMA,
        "study_id": contract.study_id,
        "surface": surface,
        "status": state,
        "expected_cells": expected_count,
        "official_test_read": False,
    }
    for key, value in required_summary.items():
        if summary.get(key) != value:
            raise ValueError(f"Surface summary {summary_path} has invalid {key}.")
    if state not in {"running", "complete", "unresolved_probe", "failed"}:
        raise ValueError(f"Unsupported surface state {state!r} in {surface_dir}.")
    if status.get("terminal") is not (state in TERMINAL_SURFACE_STATES):
        raise ValueError(f"Surface terminal flag disagrees in {status_path}.")
    candidates = summary.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"Surface summary candidates are malformed in {summary_path}.")
    candidate_by_index: dict[int, Mapping[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError(f"Malformed candidate in {summary_path}.")
        index = candidate.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index not in expected_by_index:
            raise ValueError(f"Summary contains undeclared sparse cell {index!r}.")
        if index in candidate_by_index:
            raise ValueError(f"Summary repeats sparse cell index {index}.")
        expected = expected_by_index[index]
        if (
            not _same_float(candidate.get("rho_conv"), expected["rho_conv"])
            or not _same_float(candidate.get("rho_dense"), expected["rho_dense"])
            or candidate.get("execution_rank") != expected["execution_rank"]
            or candidate.get("status") not in TERMINAL_CELL_STATUSES
        ):
            raise ValueError(f"Summary candidate sparse identity is invalid for index {index}.")
        candidate_by_index[index] = candidate
    observed_counts = dict(Counter(str(row.get("status")) for row in candidates))
    if summary.get("status_counts") != observed_counts:
        raise ValueError(f"Surface summary status counts disagree in {summary_path}.")
    if summary.get("terminal_cells") != len(candidates):
        raise ValueError(f"Surface summary terminal count disagrees in {summary_path}.")
    if state == "complete" and (
        summary.get("complete") is not True
        or len(candidates) != expected_count
        or set(candidate_by_index) != set(expected_by_index)
    ):
        raise ValueError(f"Complete surface lacks exact sparse coverage: {surface_dir}.")
    if state == "unresolved_probe" and (
        summary.get("complete") is not False or candidates or summary.get("terminal_cells") != 0
    ):
        raise ValueError(f"Malformed unresolved-probe surface: {surface_dir}.")
    if state != "complete" and summary.get("complete") is not False:
        raise ValueError(f"Non-complete surface claims complete in {summary_path}.")

    _verify_initializer(shard)
    source, source_sha256 = _verify_source_config(shard, contract)
    cells_root = surface_dir / "rho" / "cells"
    directories = sorted(path for path in cells_root.iterdir() if path.is_dir()) if cells_root.is_dir() else []
    directory_by_index: dict[int, Path] = {}
    for directory in directories:
        cell_path = directory / "cell.json"
        if not cell_path.is_file():
            if state == "complete":
                raise ValueError(f"Complete surface has an unformed cell directory {directory}.")
            continue
        index, _cell = _validate_observed_cell_identity(directory, expected_by_index)
        if index in directory_by_index:
            raise ValueError(f"Duplicate sparse cell index {index} in {cells_root}.")
        directory_by_index[index] = directory
    if state == "complete" and set(directory_by_index) != set(expected_by_index):
        raise ValueError(f"Complete surface directory coverage is not exact: {cells_root}.")
    for index, candidate in candidate_by_index.items():
        directory = directory_by_index.get(index)
        if directory is None:
            raise ValueError(f"Summary candidate {index} has no local bundle in {cells_root}.")
        if not _path_has_suffix(candidate.get("path"), ("cells", directory.name)):
            raise ValueError(f"Summary candidate path does not bind local cell {directory}.")
        cell = _load_json(directory / "cell.json")
        if cell.get("status") != candidate.get("status"):
            raise ValueError(f"Summary/cell status mismatch for {directory}.")

    receipt: dict[str, Any] | None = None
    if state in {"complete", "unresolved_probe"}:
        if receipt_record is None:
            raise ValueError(f"Terminal surface {surface['surface_id']} lacks a receipt.")
        receipt = _validate_receipt(
            shard, contract, summary=summary, record=receipt_record
        )
    elif receipt_record is not None:
        raise ValueError(f"Nonterminal/failed surface has a premature receipt: {surface_dir}.")
    probe_record = _verify_probe(
        shard,
        contract,
        source_sha256=source_sha256,
        required_complete=bool(candidates),
    )
    if candidates and probe_record is None:
        raise ValueError(f"Observed candidates lack a validated optimizer probe: {surface_dir}.")
    observed_rows = [
        _read_cell(
            shard,
            contract,
            cell_dir=directory_by_index[index],
            expected=expected_by_index[index],
            source_config=source,
            source_sha256=source_sha256,
            probe_record=probe_record,  # type: ignore[arg-type]
            transport_validated=receipt is not None and state == "complete",
        )
        for index in sorted(directory_by_index)
    ]
    rows = [
        row
        for row in observed_rows
        if row["canonical_outcome"] in {"numeric_complete", "nonfinite"}
    ]
    row_by_index = {int(row["cell_index"]): row for row in rows}
    for index, candidate in candidate_by_index.items():
        row = row_by_index.get(index)
        if row is None:
            raise ValueError(
                f"Summary candidate {index} is not a canonical terminal bundle in {surface_dir}."
            )
        if candidate.get("final_validation_accuracy") is None:
            if row["canonical_outcome"] == "numeric_complete":
                raise ValueError(f"Numeric summary candidate lacks accuracy in {surface_dir}.")
        elif not _same_float(
            candidate.get("final_validation_accuracy"), row["final_validation_accuracy"]
        ):
            raise ValueError(f"Summary/canonical accuracy mismatch in {surface_dir}.")
    if state == "complete" and set(row_by_index) != set(expected_by_index):
        raise ValueError(f"Complete surface lacks canonical terminal coverage: {surface_dir}.")
    snapshot = {
        "surface_index": surface["index"],
        "surface_id": surface["surface_id"],
        "architecture": surface["architecture"],
        "scheme": surface["scheme"],
        "optimizer": surface["optimizer"],
        "declared_target": surface["target"],
        "observed_target": shard.target,
        "surface_state": state,
        "receipt_state": "validated" if receipt is not None else "pending",
        "expected_cell_count": expected_count,
        "observed_terminal_cell_count": len(rows),
        "numeric_cell_count": sum(row["canonical_outcome"] == "numeric_complete" for row in rows),
        "nonfinite_cell_count": sum(row["canonical_outcome"] == "nonfinite" for row in rows),
        "missing_cell_count": expected_count - len(rows),
        "shard_root": str(shard.root),
        "resolved_sha256": sha256_file(shard.root / "study.resolved.json"),
        "summary_sha256": sha256_file(summary_path),
    }
    return snapshot, rows, receipt


def _range_evidence(
    best: Mapping[str, Any] | None,
    rows: Sequence[Mapping[str, Any]],
    contract: StudyContract,
) -> dict[str, Any]:
    """Classify global-ladder immediate neighbors of the best tested cell."""

    if best is None:
        return {
            "range_status": "no_numeric_observation",
            "locally_bracketed_by_accuracy": False,
            "open_directions": [],
            "unresolved_directions": [],
            "improving_or_tied_directions": [],
            "neighbors": {},
        }
    conv_count = len(contract.payload["rho_search"]["rho_conv_ladder"])
    dense_count = len(contract.payload["rho_search"]["rho_dense_ladder"])
    coordinate_rows: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        coordinate = (
            int(row["rho_conv_ladder_index"]),
            int(row["rho_dense_ladder_index"]),
        )
        if coordinate in coordinate_rows:
            raise ValueError(f"Duplicate global-ladder coordinate {coordinate!r}.")
        coordinate_rows[coordinate] = row
    best_coordinate = (
        int(best["rho_conv_ladder_index"]),
        int(best["rho_dense_ladder_index"]),
    )
    best_accuracy = float(best["final_validation_accuracy"])
    directions = (
        ("lower_rho_conv", -1, 0),
        ("upper_rho_conv", 1, 0),
        ("lower_rho_dense", 0, -1),
        ("upper_rho_dense", 0, 1),
    )
    neighbors: dict[str, dict[str, Any]] = {}
    open_directions: list[str] = []
    unresolved: list[str] = []
    improving_or_tied: list[str] = []
    for direction, conv_delta, dense_delta in directions:
        coordinate = (
            best_coordinate[0] + conv_delta,
            best_coordinate[1] + dense_delta,
        )
        record: dict[str, Any] = {
            "rho_conv_ladder_index": coordinate[0],
            "rho_dense_ladder_index": coordinate[1],
        }
        if not (0 <= coordinate[0] < conv_count and 0 <= coordinate[1] < dense_count):
            record.update({"state": "outside_global_ladder", "accuracy_relation": "open"})
            open_directions.append(direction)
        else:
            neighbor = coordinate_rows.get(coordinate)
            if neighbor is None:
                record.update({"state": "untested_in_directed_surface", "accuracy_relation": "unresolved"})
                unresolved.append(direction)
            elif (
                neighbor.get("canonical_outcome") != "numeric_complete"
                or neighbor.get("final_validation_accuracy") is None
            ):
                record.update(
                    {
                        "state": "observed_nonnumeric",
                        "cell_index": neighbor.get("cell_index"),
                        "accuracy_relation": "unresolved",
                    }
                )
                unresolved.append(direction)
            else:
                accuracy = float(neighbor["final_validation_accuracy"])
                relation = (
                    "strictly_lower"
                    if accuracy < best_accuracy
                    else "improving"
                    if accuracy > best_accuracy
                    else "tied"
                )
                record.update(
                    {
                        "state": "observed_numeric",
                        "cell_index": neighbor["cell_index"],
                        "rho_conv": neighbor["rho_conv"],
                        "rho_dense": neighbor["rho_dense"],
                        "final_validation_accuracy": accuracy,
                        "accuracy_relation": relation,
                    }
                )
                if relation in {"improving", "tied"}:
                    open_directions.append(direction)
                    improving_or_tied.append(direction)
        neighbors[direction] = record
    if open_directions:
        status = "open"
    elif unresolved:
        status = "unresolved"
    else:
        status = "bracketed"
    return {
        "range_status": status,
        "locally_bracketed_by_accuracy": status == "bracketed",
        "open_directions": open_directions,
        "unresolved_directions": unresolved,
        "improving_or_tied_directions": improving_or_tied,
        "neighbors": neighbors,
    }


def _surface_summary(
    snapshot: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    contract: StudyContract,
) -> dict[str, Any]:
    numeric = [row for row in rows if row["canonical_outcome"] == "numeric_complete"]
    best = max(
        numeric,
        key=lambda row: (
            float(row["final_validation_accuracy"]),
            -float(row["final_validation_loss"]),
            -int(row["execution_rank"]),
        ),
    ) if numeric else None
    maximum = None if best is None else float(best["final_validation_accuracy"])
    co_maxima = [
        {
            "cell_index": row["cell_index"],
            "execution_rank": row["execution_rank"],
            "rho_conv": row["rho_conv"],
            "rho_dense": row["rho_dense"],
            "rho_conv_ladder_index": row["rho_conv_ladder_index"],
            "rho_dense_ladder_index": row["rho_dense_ladder_index"],
            "final_validation_accuracy": row["final_validation_accuracy"],
        }
        for row in numeric
        if float(row["final_validation_accuracy"]) == maximum
    ] if maximum is not None else []
    evidence = _range_evidence(best, rows, contract)
    result = {
        **dict(snapshot),
        "best_tested_cell_index": None if best is None else best["cell_index"],
        "best_tested_rho_conv": None if best is None else best["rho_conv"],
        "best_tested_rho_dense": None if best is None else best["rho_dense"],
        "best_tested_validation_accuracy": maximum,
        "best_tested_validation_accuracy_percent": None if maximum is None else 100.0 * maximum,
        "best_tested_final_clipping_percent": None if best is None else best["final_either_percent"],
        "best_tested_best_clipping_percent": None if best is None else best["best_either_percent"],
        "best_tested_cells": co_maxima,
        **evidence,
        "open_directions_json": _stable_json(evidence["open_directions"]),
        "unresolved_directions_json": _stable_json(evidence["unresolved_directions"]),
        "improving_or_tied_directions_json": _stable_json(
            evidence["improving_or_tied_directions"]
        ),
        "neighbors_json": _stable_json(evidence["neighbors"]),
    }
    return result


def _pending_surface_summary(
    surface: Mapping[str, Any], contract: StudyContract
) -> dict[str, Any]:
    expected_count = len(surface_cells(contract.payload, surface))
    snapshot = {
        "surface_index": surface["index"],
        "surface_id": surface["surface_id"],
        "architecture": surface["architecture"],
        "scheme": surface["scheme"],
        "optimizer": surface["optimizer"],
        "declared_target": surface["target"],
        "observed_target": None,
        "surface_state": "pending",
        "receipt_state": "pending",
        "expected_cell_count": expected_count,
        "observed_terminal_cell_count": 0,
        "numeric_cell_count": 0,
        "nonfinite_cell_count": 0,
        "missing_cell_count": expected_count,
        "shard_root": None,
        "resolved_sha256": None,
        "summary_sha256": None,
    }
    return _surface_summary(snapshot, [], contract)


def _correlation_rows(
    surface: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    numeric = [row for row in rows if row["canonical_outcome"] == "numeric_complete"]
    architecture = str(surface["architecture"])
    result: list[dict[str, Any]] = []
    for endpoint, accuracy_field in (
        ("final", "final_validation_accuracy"),
        ("best", "best_validation_accuracy"),
    ):
        scopes: list[tuple[str, str | None, str]] = [
            ("all", None, f"{endpoint}_either_percent"),
            ("conv", None, f"{endpoint}_conv_either_percent"),
            ("dense", None, f"{endpoint}_dense_either_percent"),
        ]
        scopes.extend(("parameter", name, f"{endpoint}_by_parameter") for name in _weight_names(architecture))
        for scope, parameter, clipping_field in scopes:
            pairs: list[tuple[float, float]] = []
            for row in numeric:
                if parameter is None:
                    clipping = row.get(clipping_field)
                else:
                    values = row.get(clipping_field)
                    record = values.get(parameter) if isinstance(values, Mapping) else None
                    clipping = record.get("either_percent") if isinstance(record, Mapping) else None
                accuracy = row.get(accuracy_field)
                if clipping is not None and accuracy is not None:
                    pairs.append((float(clipping), float(accuracy)))
            stats = correlation_statistics(
                [clipping for clipping, _accuracy in pairs],
                [accuracy for _clipping, accuracy in pairs],
            )
            result.append(
                {
                    "surface_index": surface["surface_index"],
                    "surface_id": surface["surface_id"],
                    "architecture": architecture,
                    "scheme": surface["scheme"],
                    "optimizer": surface["optimizer"],
                    "endpoint": endpoint,
                    "scope": scope,
                    "parameter": parameter,
                    "accuracy_field": accuracy_field,
                    "clipping_field": (
                        clipping_field if parameter is None else f"{clipping_field}.{parameter}.either_percent"
                    ),
                    **stats,
                }
            )
    return result


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _plot_accuracy_vs_clipping(
    rows: Sequence[Mapping[str, Any]], output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    numeric = [row for row in rows if row["canonical_outcome"] == "numeric_complete"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = {"baseline": "#4c78a8", "ours": "#f58518", "legacy": "#54a24b"}
    markers = {"conv1": "o", "conv2": "s"}
    for axis, endpoint, accuracy_field in zip(
        axes,
        ("final", "best"),
        ("final_validation_accuracy_percent", "best_validation_accuracy_percent"),
    ):
        for scheme in ("baseline", "ours", "legacy"):
            for architecture in ("conv1", "conv2"):
                selected = [
                    row
                    for row in numeric
                    if row["scheme"] == scheme and row["architecture"] == architecture
                ]
                if not selected:
                    continue
                axis.scatter(
                    [float(row[f"{endpoint}_either_percent"]) for row in selected],
                    [float(row[accuracy_field]) for row in selected],
                    color=colors[scheme],
                    marker=markers[architecture],
                    alpha=0.75,
                    s=30,
                    label=f"{architecture} {scheme}",
                )
        axis.set_xlabel(f"{endpoint.capitalize()} checkpoint: weights at either bound (%)")
        axis.set_ylabel(
            "Final validation accuracy (%)"
            if endpoint == "final"
            else "Best validation accuracy (%)"
        )
        axis.set_title(f"{endpoint.capitalize()} checkpoint / corresponding accuracy")
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        figure.legend(
            by_label.values(), by_label.keys(), loc="outside lower center", ncol=3
        )
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "No numeric terminal cells", ha="center", va="center", transform=axis.transAxes)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _markdown(report: Mapping[str, Any]) -> str:
    coverage = report["coverage"]
    lines = [
        "# Directed Conv1/Conv2 exploratory LR snapshot",
        "",
        f"> {EXPLORATORY_LABEL}",
        "",
        f"Snapshot status: **{coverage['status']}**. Validated "
        f"{coverage['observed_terminal_cell_count']}/{coverage['expected_cell_count']} "
        f"declared sparse terminal cells across "
        f"{coverage['discovered_surface_count']}/{coverage['expected_surface_count']} shards.",
        "",
        "Final-checkpoint clipping is paired only with final validation accuracy; "
        "best-checkpoint clipping is paired only with best validation accuracy. "
        "Correlations are descriptive within each surface and are not causal.",
        "",
        "| Architecture | Scheme | Optimizer | State | Cells | Best final accuracy | Final clipping at best | Range |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for surface in report["surfaces"]:
        accuracy = surface.get("best_tested_validation_accuracy_percent")
        clipping = surface.get("best_tested_final_clipping_percent")
        lines.append(
            "| {architecture} | {scheme} | {optimizer} | {state} | {observed}/{expected} | "
            "{accuracy} | {clipping} | {range_status} |".format(
                architecture=surface["architecture"],
                scheme=surface["scheme"],
                optimizer=surface["optimizer"],
                state=surface["surface_state"],
                observed=surface["observed_terminal_cell_count"],
                expected=surface["expected_cell_count"],
                accuracy="—" if accuracy is None else f"{float(accuracy):.2f}%",
                clipping="—" if clipping is None else f"{float(clipping):.3f}%",
                range_status=surface["range_status"],
            )
        )
    lines.extend(
        [
            "",
            "The range label uses immediate neighbors on the full declared global "
            "rho ladders. Missing directed coordinates and nonnumeric cells are "
            "unresolved; an outside-ladder or improving/tied neighbor is open; all "
            "four numeric, strictly lower neighbors are bracketed.",
            "",
            "Artifacts: `cells.csv`, `surfaces.csv`, `correlations.csv`, "
            "`summary.json`, and `plots/accuracy_vs_clipping.png`.",
            "",
        ]
    )
    return "\n".join(lines)


def _complete_coverage(coverage: Mapping[str, Any]) -> bool:
    return bool(
        coverage.get("discovered_surface_count") == EXPECTED_SURFACE_COUNT
        and coverage.get("complete_surface_count") == EXPECTED_SURFACE_COUNT
        and coverage.get("validated_receipt_count") == EXPECTED_SURFACE_COUNT
        and coverage.get("observed_terminal_cell_count") == EXPECTED_CELL_COUNT
        and coverage.get("transport_validated_cell_count") == EXPECTED_CELL_COUNT
    )


def analyze(
    *,
    study_config_path: Path = DEFAULT_STUDY,
    study_root: Path | None = None,
    shard_roots: Sequence[Path] = (),
    receipt_roots: Sequence[Path] = (),
    output_dir: Path,
    require_complete: bool = False,
) -> dict[str, Any]:
    contract = load_contract(study_config_path)
    roots = discover_shard_roots(study_root=study_root, shard_roots=shard_roots)
    shards = _load_shards(roots, contract)
    inferred_receipt_roots = list(receipt_roots)
    for shard in shards:
        if shard.root.parent.name == "shards":
            inferred_receipt_roots.append(shard.root.parent.parent / "transport_receipts")
    receipt_inventory = _receipt_inventory(
        contract,
        study_root=study_root,
        receipt_roots=inferred_receipt_roots,
    )
    snapshots_by_index: dict[int, dict[str, Any]] = {}
    rows_by_index: dict[int, list[dict[str, Any]]] = {}
    receipts: list[dict[str, Any]] = []
    for shard in shards:
        index = int(shard.surface["index"])
        records = receipt_inventory[index]
        snapshot, rows, receipt = _read_surface(
            shard,
            contract,
            receipt_record=records[0] if records else None,
        )
        snapshots_by_index[index] = snapshot
        rows_by_index[index] = rows
        if receipt is not None:
            receipts.append(receipt)
    source_identities = {
        (receipt["source_commit"], receipt["source_archive_sha256"])
        for receipt in receipts
    }
    if len(source_identities) > 1:
        raise ValueError("Validated receipts do not share one frozen source identity.")
    all_rows = [
        row
        for index in sorted(rows_by_index)
        for row in rows_by_index[index]
    ]
    surfaces: list[dict[str, Any]] = []
    correlations: list[dict[str, Any]] = []
    for surface in contract.surfaces:
        index = int(surface["index"])
        if index in snapshots_by_index:
            summary = _surface_summary(
                snapshots_by_index[index], rows_by_index[index], contract
            )
        else:
            summary = _pending_surface_summary(surface, contract)
        surfaces.append(summary)
        correlations.extend(_correlation_rows(summary, rows_by_index.get(index, [])))
    state_counts = dict(Counter(surface["surface_state"] for surface in surfaces))
    coverage = {
        "status": "partial",
        "expected_surface_count": EXPECTED_SURFACE_COUNT,
        "discovered_surface_count": len(shards),
        "complete_surface_count": state_counts.get("complete", 0),
        "surface_state_counts": state_counts,
        "validated_receipt_count": len(receipts),
        "expected_cell_count": EXPECTED_CELL_COUNT,
        "observed_terminal_cell_count": len(all_rows),
        "numeric_cell_count": sum(row["canonical_outcome"] == "numeric_complete" for row in all_rows),
        "nonfinite_cell_count": sum(row["canonical_outcome"] == "nonfinite" for row in all_rows),
        "transport_validated_cell_count": sum(bool(row["transport_validated"]) for row in all_rows),
        "missing_cell_count": EXPECTED_CELL_COUNT - len(all_rows),
    }
    if _complete_coverage(coverage):
        coverage["status"] = "complete"
    artifacts = {
        "cells_csv": "cells.csv",
        "surfaces_csv": "surfaces.csv",
        "correlations_csv": "correlations.csv",
        "summary_json": "summary.json",
        "report_markdown": "report.md",
        "accuracy_vs_clipping_plot": "plots/accuracy_vs_clipping.png",
    }
    analyzer_path = Path(__file__).resolve()
    report = {
        "schema_version": ANALYSIS_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_label": EXPLORATORY_LABEL,
        "evidence_class": "ordinary_mnist_exploratory",
        "exploratory_noncanonical": True,
        "paper_evidence": False,
        "official_test_read": False,
        "study_id": contract.study_id,
        "study_config": str(contract.path),
        "study_config_sha256": contract.sha256,
        "analyzer_source": str(analyzer_path),
        "analyzer_source_sha256": sha256_file(analyzer_path),
        "weight_interval": [WEIGHT_MIN, WEIGHT_MAX],
        "correlation_interpretation": {
            "scope": "within_surface_numeric_terminal_cells",
            "causal": False,
            "pairing": {
                "final_checkpoint": "final_validation_accuracy",
                "best_checkpoint": "best_validation_accuracy",
            },
        },
        "coverage": coverage,
        "source_shards": [
            {
                "root": str(shard.root),
                "surface_index": shard.surface["index"],
                "surface_id": shard.surface["surface_id"],
                "target": shard.target,
                "resolved_sha256": sha256_file(shard.root / "study.resolved.json"),
            }
            for shard in shards
        ],
        "transport_receipts": receipts,
        "surfaces": surfaces,
        "correlations": correlations,
        "cells": all_rows,
        "artifacts": artifacts,
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / artifacts["cells_csv"], all_rows, CELL_FIELDS)
    _write_csv(output / artifacts["surfaces_csv"], surfaces, SURFACE_FIELDS)
    _write_csv(
        output / artifacts["correlations_csv"], correlations, CORRELATION_FIELDS
    )
    _plot_accuracy_vs_clipping(
        all_rows, output / artifacts["accuracy_vs_clipping_plot"]
    )
    (output / artifacts["report_markdown"]).write_text(
        _markdown(report), encoding="utf-8"
    )
    atomic_write_json(output / artifacts["summary_json"], report)
    if require_complete and not _complete_coverage(coverage):
        raise IncompleteCoverageError(
            "Directed Conv1/Conv2 coverage is partial; the partial snapshot was written.",
            report=report,
        )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STUDY)
    parser.add_argument(
        "--study-root", type=Path, help="Discover task roots below STUDY_ROOT/shards."
    )
    parser.add_argument(
        "--shard-root",
        type=Path,
        action="append",
        default=[],
        help="Collected task shard root; repeat for explicit roots.",
    )
    parser.add_argument(
        "--receipt-root",
        type=Path,
        action="append",
        default=[],
        help="Additional terminal receipt file or directory.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Write the snapshot, then exit 2 unless all 12/257 evidence is complete.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir is None:
        if args.study_root is None:
            raise SystemExit("--output-dir is required when --study-root is omitted")
        output_dir = args.study_root / "analysis" / "directed_exploratory_lr"
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
        report = error.report or {}
        print(
            json.dumps(
                {
                    "error": str(error),
                    "coverage": report.get("coverage"),
                    "output_dir": str(Path(output_dir).expanduser().resolve()),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
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
