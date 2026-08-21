#!/usr/bin/env python3
"""Validate collection of the directed Conv1/Conv2 exploratory LR study.

This is an evidence validator, not a scientific analyzer.  It validates every
collected authority that is present and reports an honest partial snapshot
while jobs or collection are incomplete.  ``--require-complete`` additionally
requires exactly twelve disjoint task authorities, twelve locally rebound
terminal receipts, and all 257 declared sparse cell bundles.

The study is ordinary-MNIST exploratory evidence.  Successful validation does
not create a learning-rate handoff or paper-facing result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

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
        SURFACE_STATUS_SCHEMA,
        SURFACE_SUMMARY_SCHEMA,
        load_study,
        surface_cells,
        surface_specs,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution fallback
    from reporting import atomic_write_json, sha256_file, validate_run  # type: ignore[no-redef]
    from rho_search import prepare_training_config  # type: ignore[no-redef]
    from run_conv12_bounded_rho import build_source_config  # type: ignore[no-redef]
    from run_conv12_directed_exploratory_lr_search import (  # type: ignore[no-redef]
        DEFAULT_STUDY,
        EXPECTED_INITIALIZER_SHA256,
        EXPECTED_TK,
        RESOLVED_SCHEMA,
        SURFACE_STATUS_SCHEMA,
        SURFACE_SUMMARY_SCHEMA,
        load_study,
        surface_cells,
        surface_specs,
    )


VALIDATION_SCHEMA = "perfectdiode-conv12-directed-exploratory-lr-collection-validation/v1"
RECEIPT_SCHEMA = "perfectdiode-conv12-directed-exploratory-lr-transport/v1"
EXPECTED_CELL_COUNTS = (39, 30, 18, 16, 20, 6, 30, 40, 17, 17, 14, 10)
EXPECTED_SURFACES = 12
EXPECTED_CELLS = 257
ALLOWED_TARGETS = frozenset({"main", "akib", "jean-zay"})
NONFINITE_KINDS = frozenset({"nonfinite_diagnostic", "nonfinite_training_value"})
TASK_ROOT_PATTERN = re.compile(r".+-task_(?P<task>[0-9]|1[01])$")


class IncompleteCollectionError(RuntimeError):
    """Raised when ``require_complete`` is requested for a partial collection."""

    def __init__(self, report: Mapping[str, Any]):
        coverage = report["coverage"]
        super().__init__(
            "Directed Conv1/Conv2 collection is incomplete: "
            f"authorities={coverage['authority_count']}/{EXPECTED_SURFACES}, "
            f"receipts={coverage['validated_receipt_count']}/{EXPECTED_SURFACES}, "
            f"terminal_cells={coverage['terminal_cell_count']}/{EXPECTED_CELLS}."
        )
        self.report = dict(report)


@dataclass(frozen=True)
class Contract:
    path: Path
    study: dict[str, Any]
    parent_path: Path
    parent: dict[str, Any]
    sha256: str
    study_id: str
    surfaces: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Authority:
    root: Path
    surface: dict[str, Any]
    resolved: dict[str, Any]
    target: str


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


def _same_float(left: Any, right: Any) -> bool:
    return math.isclose(
        _finite_float(left, label="floating value"),
        _finite_float(right, label="floating value"),
        rel_tol=1e-15,
        abs_tol=0.0,
    )


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Expected lowercase SHA-256 for {label}, got {value!r}.")
    return value


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
    raise ValueError(f"Unsupported architecture {architecture!r}.")


def _bias_names(architecture: str) -> tuple[str, ...]:
    if architecture == "conv1":
        return ("Bias_0",)
    if architecture == "conv2":
        return ("Bias_0", "Bias_1")
    raise ValueError(f"Unsupported architecture {architecture!r}.")


def _canonical_target(surface: Mapping[str, Any]) -> str:
    declared = str(surface["target"])
    return "jean-zay" if declared.startswith("jean-zay") else declared


def load_contract(path: Path = DEFAULT_STUDY) -> Contract:
    source, study, parent_path, parent = load_study(path)
    surfaces = tuple(surface_specs(study))
    counts = tuple(len(surface_cells(study, surface)) for surface in surfaces)
    if counts != EXPECTED_CELL_COUNTS:
        raise ValueError(f"Unexpected directed sparse-cell counts: {counts!r}.")
    if sum(counts) != EXPECTED_CELLS or len(surfaces) != EXPECTED_SURFACES:
        raise ValueError("Directed study is not the exact 12-surface/257-cell contract.")
    if tuple(int(surface["index"]) for surface in surfaces) != tuple(range(12)):
        raise ValueError("Directed surface indices are not canonical.")
    return Contract(
        path=source,
        study=study,
        parent_path=parent_path,
        parent=parent,
        sha256=sha256_file(source),
        study_id=str(study["study_id"]),
        surfaces=surfaces,
    )


def discover_shard_roots(
    *, study_root: Path | None, shard_roots: Sequence[Path]
) -> list[Path]:
    candidates: list[Path] = []
    if study_root is not None:
        shard_parent = Path(study_root).expanduser().resolve() / "shards"
        if shard_parent.is_dir():
            candidates.extend(
                path.parent for path in sorted(shard_parent.rglob("study.resolved.json"))
            )
        elif not shard_roots:
            # An uncollected study root is a legitimate partial snapshot.
            return []
    candidates.extend(Path(path).expanduser().resolve() for path in shard_roots)
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        root = candidate.resolve()
        if root in seen:
            continue
        if not root.is_dir() or not (root / "study.resolved.json").is_file():
            raise ValueError(f"Invalid collected task root: {root}.")
        unique.append(root)
        seen.add(root)
    for index, left in enumerate(unique):
        for right in unique[index + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError(f"Shard roots are not disjoint: {left} and {right}.")
    return unique


def _validate_resolved(root: Path, contract: Contract) -> Authority:
    resolved = _load_json(root / "study.resolved.json")
    required = {
        "schema_version": RESOLVED_SCHEMA,
        "study_id": contract.study_id,
        "evidence_class": "ordinary_mnist_exploratory",
        "study_config_sha256": contract.sha256,
        "parent_study_config_sha256": sha256_file(contract.parent_path),
        "initializer_checkpoint_sha256_by_architecture": EXPECTED_INITIALIZER_SHA256,
        "surface_count": EXPECTED_SURFACES,
        "total_cells": EXPECTED_CELLS,
        "surfaces": list(contract.surfaces),
        "official_test_read": False,
        "restart_interrupted_cells": True,
    }
    for key, expected in required.items():
        if resolved.get(key) != expected:
            raise ValueError(f"Resolved task {root} has invalid {key}.")
    if not _path_has_suffix(
        resolved.get("study_config"),
        ("configs", "conv", contract.path.name),
    ):
        raise ValueError(f"Resolved task {root} has an unexpected study_config path.")
    if not _path_has_suffix(
        resolved.get("parent_study_config"),
        ("configs", "conv", contract.parent_path.name),
    ):
        raise ValueError(f"Resolved task {root} has an unexpected parent config path.")
    target = resolved.get("target")
    if target not in ALLOWED_TARGETS:
        raise ValueError(f"Resolved task {root} has invalid target {target!r}.")
    code = resolved.get("code")
    if not isinstance(code, Mapping):
        raise ValueError(f"Resolved task {root} lacks frozen-source identity.")
    commit = code.get("commit")
    archive = _require_sha256(
        code.get("source_archive_sha256"), label=f"{root} source archive"
    )
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or code.get("dirty") is not False
        or code.get("source_kind") != "frozen_archive"
        or code.get("working_tree_sha256") != archive
    ):
        raise ValueError(f"Resolved task {root} does not prove one clean frozen source.")

    expected_by_id = {str(surface["surface_id"]): surface for surface in contract.surfaces}
    surfaces_root = root / "surfaces"
    directories = (
        sorted(path for path in surfaces_root.iterdir() if path.is_dir())
        if surfaces_root.is_dir()
        else []
    )
    known = [path for path in directories if path.name in expected_by_id]
    unknown = [path.name for path in directories if path.name not in expected_by_id]
    if len(known) != 1 or unknown:
        raise ValueError(
            f"Expected exactly one declared surface under {root}; "
            f"known={[path.name for path in known]!r}, unknown={unknown!r}."
        )
    surface = expected_by_id[known[0].name]
    match = TASK_ROOT_PATTERN.fullmatch(root.name)
    if match is not None and int(match.group("task")) != int(surface["index"]):
        raise ValueError(f"Task-root name does not match its surface under {root}.")
    expected_target = _canonical_target(surface)
    if target != expected_target:
        raise ValueError(
            f"Surface {surface['surface_id']} is on {target!r}, expected {expected_target!r}."
        )
    return Authority(root=root, surface=surface, resolved=resolved, target=str(target))


def _load_authorities(roots: Sequence[Path], contract: Contract) -> list[Authority]:
    authorities = [_validate_resolved(root, contract) for root in roots]
    indices = [int(authority.surface["index"]) for authority in authorities]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate task/surface authorities: {indices!r}.")
    identities = {
        (
            authority.resolved["code"]["commit"],
            authority.resolved["code"]["source_archive_sha256"],
        )
        for authority in authorities
    }
    if len(identities) > 1:
        raise ValueError("Collected authorities do not share one frozen source identity.")
    return sorted(authorities, key=lambda item: int(item.surface["index"]))


def _validate_zero_bias_record(
    value: Any, *, architecture: str, label: str, require_checkpoints: bool
) -> None:
    if not isinstance(value, Mapping) or value.get("all_exact_zero") is not True:
        raise ValueError(f"Missing exact-zero bias verification in {label}.")
    if value.get("nonzero_bias_element_count", 0) != 0:
        raise ValueError(f"Nonzero bias recorded in {label}.")
    expected = list(_bias_names(architecture))
    direct_names = value.get("expected_bias_names")
    if direct_names is not None and sorted(direct_names) != expected:
        raise ValueError(f"Unexpected bias names in {label}.")
    checkpoints = value.get("checkpoints")
    if require_checkpoints:
        if not isinstance(checkpoints, list) or len(checkpoints) != 2:
            raise ValueError(f"Expected best/final zero-bias records in {label}.")
        for checkpoint in checkpoints:
            if (
                not isinstance(checkpoint, Mapping)
                or checkpoint.get("all_exact_zero") is not True
                or checkpoint.get("nonzero_bias_element_count") != 0
                or sorted(checkpoint.get("expected_bias_names", [])) != expected
            ):
                raise ValueError(f"Invalid checkpoint zero-bias record in {label}.")


def _validate_initializer(authority: Authority) -> dict[str, Any]:
    architecture = str(authority.surface["architecture"])
    expected_sha = EXPECTED_INITIALIZER_SHA256[architecture]
    asset_root = authority.root / "assets" / "bounded_uniform" / architecture
    asset_path = asset_root / "asset.json"
    checkpoint = asset_root / "final_model.pt"
    asset = _load_json(asset_path)
    required = {
        "architecture": architecture,
        "initializer": "bounded_uniform",
        "checkpoint_sha256": expected_sha,
        "official_test_read": False,
    }
    for key, expected in required.items():
        if asset.get(key) != expected:
            raise ValueError(f"Initializer asset {asset_path} has invalid {key}.")
    if not _path_has_suffix(
        asset.get("checkpoint"),
        (
            authority.root.name,
            "assets",
            "bounded_uniform",
            architecture,
            "final_model.pt",
        ),
    ):
        raise ValueError(f"Initializer asset {asset_path} is not locally rebindable.")
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected_sha:
        raise ValueError(f"Initializer bytes do not match in {authority.root}.")
    _validate_zero_bias_record(
        asset.get("zero_bias_checkpoint_verification"),
        architecture=architecture,
        label=str(asset_path),
        require_checkpoints=False,
    )
    return {
        "path": str(asset_path),
        "sha256": sha256_file(asset_path),
        "checkpoint_sha256": expected_sha,
    }


def _validate_source_config(
    authority: Authority, contract: Contract
) -> tuple[dict[str, Any], str]:
    surface = authority.surface
    architecture = str(surface["architecture"])
    source_path = (
        authority.root / "surfaces" / str(surface["surface_id"]) / "source_config.json"
    )
    source = _load_json(source_path)
    init_path = source.get("init_checkpoint_path")
    if not _path_has_suffix(
        init_path,
        (
            authority.root.name,
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
        raise ValueError(f"Source config {source_path} lacks an MNIST root.")
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


def _validate_probe(
    authority: Authority,
    contract: Contract,
    *,
    source_sha256: str,
    require_complete: bool,
) -> tuple[dict[str, Any], str, str]:
    surface_dir = authority.root / "surfaces" / str(authority.surface["surface_id"])
    resolved_path = surface_dir / "rho" / "resolved.json"
    probe_path = surface_dir / "rho" / "probe.json"
    resolved = _load_json(resolved_path)
    probe = _load_json(probe_path)
    search = probe.get("search_signature")
    if not isinstance(search, Mapping) or search != resolved:
        raise ValueError(f"Probe/resolved signatures disagree under {surface_dir}.")
    cells = surface_cells(contract.study, authority.surface)
    expected = {
        "schema_version": "conv-rho-search/v1",
        "source_config_sha256": source_sha256,
        "code": authority.resolved["code"],
        "optimizer": authority.surface["optimizer"],
        "evidence_class": "ordinary_mnist_exploratory",
        "rho_conv": sorted({float(cell["rho_conv"]) for cell in cells}),
        "rho_dense": sorted({float(cell["rho_dense"]) for cell in cells}),
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
        "target": authority.target,
    }
    for key, expected_value in expected.items():
        if search.get(key) != expected_value:
            raise ValueError(f"Probe {probe_path} has invalid {key}.")
    safety = search.get("safety")
    if (
        not isinstance(safety, Mapping)
        or safety.get("rejections_enabled") is not False
        or safety.get("bound_occupancy") != "report_only"
        or safety.get("projection_efficiency") != "report_only"
        or safety.get("bound_occupancy_increase_maximum") is not None
        or safety.get("projection_efficiency_minimum") is not None
    ):
        raise ValueError(f"Probe {probe_path} does not retain report-only safety.")
    architecture = str(authority.surface["architecture"])
    weights = _weight_names(architecture)
    biases = _bias_names(architecture)
    if (
        probe.get("schema_version") != "conv-rho-probe/v1"
        or probe.get("official_test_read") is not False
    ):
        raise ValueError(f"Optimizer probe identity is invalid: {probe_path}.")
    if require_complete and (
        probe.get("status") != "complete"
        or probe.get("probe_stable") is not True
        or probe.get("proposal_units_valid") is not True
    ):
        raise ValueError(f"Optimizer probe is not valid and complete: {probe_path}.")
    if probe.get("status") == "complete":
        if (
            probe.get("parameter_names") != list(weights + biases)
            or probe.get("weight_names") != list(weights)
        ):
            raise ValueError(f"Optimizer probe parameter order is invalid: {probe_path}.")
        units = probe.get("normalization_unit_by_weight")
        if not isinstance(units, Mapping) or set(units) != set(weights):
            raise ValueError(f"Optimizer probe has invalid normalization units: {probe_path}.")
        if any(
            _finite_float(units[name], label=f"probe unit {name}") <= 0
            for name in weights
        ):
            raise ValueError(f"Optimizer probe has a nonpositive unit: {probe_path}.")
        _require_sha256(
            probe.get("initial_parameter_sha256"),
            label=f"{probe_path} initial parameters",
        )
        _validate_zero_bias_record(
            probe.get("zero_bias_checkpoint_verification"),
            architecture=architecture,
            label=str(probe_path),
            require_checkpoints=True,
        )
    semantic_sha256 = hashlib.sha256(
        json.dumps(probe, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return probe, semantic_sha256, sha256_file(probe_path)


def _manifest_input(manifest: Mapping[str, Any], role: str, run_dir: Path) -> Mapping[str, Any]:
    inputs = manifest.get("inputs")
    records = (
        [row for row in inputs if isinstance(row, Mapping) and row.get("role") == role]
        if isinstance(inputs, list)
        else []
    )
    if len(records) != 1:
        raise ValueError(f"Expected one manifest input role {role!r} in {run_dir}.")
    return records[0]


def _validate_npz_biases(
    run_dir: Path, result: Mapping[str, Any], architecture: str
) -> None:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"Result artifact index is missing in {run_dir}.")
    for checkpoint_name in ("weights_best.npz", "weights_final.npz"):
        records = [
            row
            for row in artifacts
            if isinstance(row, Mapping)
            and row.get("kind") == "checkpoint"
            and Path(str(row.get("path", ""))).name == checkpoint_name
        ]
        if len(records) != 1:
            raise ValueError(f"Expected one indexed {checkpoint_name} in {run_dir}.")
        relative = Path(str(records[0]["path"]))
        if relative.is_absolute():
            raise ValueError(f"Indexed checkpoint path is absolute in {run_dir}.")
        checkpoint = (run_dir / relative).resolve()
        try:
            checkpoint.relative_to(run_dir.resolve())
        except ValueError as error:
            raise ValueError(f"Indexed checkpoint escapes {run_dir}.") from error
        with np.load(checkpoint, allow_pickle=False) as payload:
            for name in _bias_names(architecture):
                if name not in payload.files or not np.all(np.asarray(payload[name]) == 0):
                    raise ValueError(f"Checkpoint {checkpoint} has nonzero/missing {name}.")


def classify_cell_outcome(
    raw_status: Any,
    safety_failure: Any,
    *,
    reporting_state: Any,
    result_present: bool,
) -> str:
    """Classify a cell without converting live work into terminal evidence."""

    if raw_status == "running":
        if reporting_state != "running" or result_present:
            raise ValueError("A running cell must be a canonical running bundle.")
        return "running"
    if raw_status == "complete":
        if reporting_state != "complete" or not result_present:
            raise ValueError("A numeric cell must be a canonical complete bundle.")
        return "numeric_complete"
    if raw_status in {"candidate_rejected_nonfinite", "candidate_rejected_safety"}:
        kind = (
            safety_failure.get("kind")
            if isinstance(safety_failure, Mapping)
            else None
        )
        if kind not in NONFINITE_KINDS:
            raise ValueError("Rejected cell is not a declared nonfinite outcome.")
        if reporting_state != "failed" or result_present:
            raise ValueError("A nonfinite cell must be a canonical failed bundle.")
        return "nonfinite"
    raise ValueError(f"Unsupported directed cell status {raw_status!r}.")


def _validate_cell(
    authority: Authority,
    contract: Contract,
    *,
    run_dir: Path,
    expected: Mapping[str, Any],
    source_config: Mapping[str, Any],
    source_sha256: str,
    probe: Mapping[str, Any],
    probe_semantic_sha256: str,
    probe_file_sha256: str,
) -> dict[str, Any]:
    errors = validate_run(run_dir)
    if errors:
        raise ValueError(f"Invalid canonical bundle {run_dir}: {errors!r}.")
    cell = _load_json(run_dir / "cell.json")
    manifest = _load_json(run_dir / "manifest.json")
    status = _load_json(run_dir / "status.json")
    result_path = run_dir / "result.json"
    result = _load_json(result_path) if result_path.is_file() else None
    index = int(expected["index"])
    architecture = str(authority.surface["architecture"])
    weights = _weight_names(architecture)
    biases = _bias_names(architecture)
    if (
        cell.get("index") != index
        or cell.get("optimizer") != authority.surface["optimizer"]
        or cell.get("evidence_class") != "ordinary_mnist_exploratory"
        or cell.get("bias_policy") != "zero"
        or not _same_float(cell.get("rho_conv"), expected["rho_conv"])
        or not _same_float(cell.get("rho_dense"), expected["rho_dense"])
    ):
        raise ValueError(f"Sparse cell identity mismatch in {run_dir}.")
    raw_status = cell.get("status")
    failure = cell.get("safety_failure")
    try:
        outcome = classify_cell_outcome(
            raw_status,
            failure,
            reporting_state=status.get("state"),
            result_present=result is not None,
        )
    except ValueError as error:
        raise ValueError(f"{error} Path: {run_dir}.") from error

    rates_raw = cell.get("learning_rates_by_parameter")
    names = weights + biases
    if not isinstance(rates_raw, Mapping) or set(rates_raw) != set(names):
        raise ValueError(f"Cell learning-rate map is invalid in {run_dir}.")
    rates = {name: _finite_float(rates_raw[name], label=f"LR {name}") for name in names}
    if any(rates[name] != 0.0 for name in biases):
        raise ValueError(f"Cell has a nonzero bias LR in {run_dir}.")
    vector = cell.get("learning_rate_vector")
    if not isinstance(vector, list) or len(vector) != len(names) or any(
        not _same_float(vector[position], rates[name]) for position, name in enumerate(names)
    ):
        raise ValueError(f"Cell LR vector/order mismatch in {run_dir}.")
    units = probe["normalization_unit_by_weight"]
    expected_rates = {
        name: (
            float(expected["rho_conv"]) / float(units[name])
            if name.startswith("ConvWeight_")
            else float(expected["rho_dense"]) / float(units[name])
        )
        for name in weights
    }
    expected_rates.update({name: 0.0 for name in biases})
    if any(not _same_float(rates[name], expected_rates[name]) for name in names):
        raise ValueError(f"Cell rates do not match rho/probe units in {run_dir}.")

    signature = cell.get("signature")
    if (
        not isinstance(signature, Mapping)
        or signature.get("source_config_sha256") != source_sha256
        or signature.get("probe_sha256") != probe_semantic_sha256
        or signature.get("code") != authority.resolved["code"]
        or signature.get("epochs") != 3
        or signature.get("expected_candidate_steps") != 10314
        or signature.get("skip_canary") is not True
        or signature.get("evidence_class") != "ordinary_mnist_exploratory"
        or signature.get("split_seed") != 0
        or signature.get("shuffle_seed") != 0
        or signature.get("safety", {}).get("rejections_enabled") is not False
    ):
        raise ValueError(f"Cell source/config signature mismatch in {run_dir}.")

    expected_candidate = prepare_training_config(
        source_config,
        str(authority.surface["optimizer"]),
        [rates[name] for name in names],
        epochs=3,
        max_batches=None,
        max_validation_batches=None,
        split_seed=0,
        shuffle_seed=0,
        validation_batch_size=64,
    )
    candidate_path = run_dir / "source_config.json"
    candidate = _load_json(candidate_path)
    if candidate != expected_candidate or _load_json(run_dir / "config.used.json") != candidate:
        raise ValueError(f"Executed cell config differs from its governed config: {run_dir}.")
    configuration = manifest.get("configuration")
    command = manifest.get("command")
    dataset = manifest.get("dataset")
    runtime = manifest.get("runtime")
    if (
        manifest.get("study_id") != contract.study_id
        or manifest.get("run_id") != run_dir.name
        or manifest.get("evidence_class") != "ordinary_mnist_exploratory"
        or manifest.get("smoke") is not False
        or not isinstance(dataset, Mapping)
        or dataset.get("evaluation_split") != "validation"
        or dataset.get("official_test_read") is not False
        or not isinstance(runtime, Mapping)
        or runtime.get("target") != authority.target
        or status.get("runtime") != runtime
        or not isinstance(command, Mapping)
        or command.get("surface_index") != authority.surface["index"]
        or command.get("cell_index") != index
        or not isinstance(configuration, Mapping)
        or configuration.get("resolved") != candidate
        or configuration.get("sha256") != sha256_file(candidate_path)
        or configuration.get("optimizer") != authority.surface["optimizer"]
    ):
        raise ValueError(f"Cell manifest identity/config mismatch in {run_dir}.")
    if (
        _manifest_input(manifest, "source_config", run_dir).get("sha256") != source_sha256
        or _manifest_input(manifest, "optimizer_probe", run_dir).get("sha256")
        != probe_file_sha256
    ):
        raise ValueError(f"Cell manifest input hashes mismatch in {run_dir}.")
    t_value, k_value = EXPECTED_TK[architecture]
    model = candidate.get("model_base")
    if (
        not isinstance(model, Mapping)
        or model.get("weight_init_mode") != "bounded_uniform"
        or not _same_float(model.get("weight_min"), 1e-5)
        or not _same_float(model.get("weight_max"), 1e-4)
        or model.get("num_iterations_inference") != t_value
        or model.get("num_iterations_training") != k_value
    ):
        raise ValueError(f"Cell initializer/bounds/TK mismatch in {run_dir}.")

    if outcome == "numeric_complete":
        if cell.get("completed_steps") != 10314 or cell.get("selection_eligible") is not True:
            raise ValueError(f"Numeric cell did not complete 10,314 steps: {run_dir}.")
        diagnostics = _load_json(run_dir / "safety_diagnostics.json")
        if (
            diagnostics.get("processed_steps") != 10314
            or diagnostics.get("report_only_diagnostics", {}).get("used_for_rejection")
            is not False
            or diagnostics.get("terminal_gates", {}).get("scientific_rejections_enabled")
            is not False
        ):
            raise ValueError(f"Numeric cell safety contract mismatch in {run_dir}.")
        _validate_zero_bias_record(
            cell.get("zero_bias_checkpoint_verification"),
            architecture=architecture,
            label=str(run_dir / "cell.json"),
            require_checkpoints=True,
        )
        _validate_zero_bias_record(
            diagnostics.get("zero_bias_checkpoint_verification"),
            architecture=architecture,
            label=str(run_dir / "safety_diagnostics.json"),
            require_checkpoints=True,
        )
        metrics = _load_json(run_dir / "metrics.json")
        if (
            metrics.get("official_test_accuracy") is not None
            or metrics.get("official_test_loss") is not None
            or metrics.get("official_test_examples") != 0
            or metrics.get("official_test_evaluations") != 0
        ):
            raise ValueError(f"Numeric cell contains official-test metrics: {run_dir}.")
        result_dataset = result.get("dataset") if isinstance(result, Mapping) else None
        completion = result.get("completion") if isinstance(result, Mapping) else None
        if (
            not isinstance(result_dataset, Mapping)
            or result_dataset.get("official_test_read") is not False
            or not isinstance(completion, Mapping)
            or completion.get("official_test_read") is not False
            or completion.get("criteria_met") is not True
            or completion.get("rho_cell_complete") is not True
        ):
            raise ValueError(f"Numeric result completion/test contract mismatch: {run_dir}.")
        _validate_zero_bias_record(
            completion.get("zero_bias_checkpoint_verification"),
            architecture=architecture,
            label=str(result_path),
            require_checkpoints=True,
        )
        _validate_npz_biases(run_dir, result, architecture)
    return {
        "index": index,
        "execution_rank": int(expected["execution_rank"]),
        "rho_conv_ladder_index": int(expected["rho_conv_ladder_index"]),
        "rho_dense_ladder_index": int(expected["rho_dense_ladder_index"]),
        "rho_conv": float(expected["rho_conv"]),
        "rho_dense": float(expected["rho_dense"]),
        "raw_status": raw_status,
        "canonical_outcome": outcome,
        "run_dir": str(run_dir),
        "result_sha256": None if result is None else sha256_file(result_path),
    }


def _cell_directories(surface_dir: Path) -> list[Path]:
    root = surface_dir / "rho" / "cells"
    return sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []


def _validate_authority(authority: Authority, contract: Contract) -> dict[str, Any]:
    surface = authority.surface
    index = int(surface["index"])
    expected_cells = surface_cells(contract.study, surface)
    expected_by_index = {int(cell["index"]): cell for cell in expected_cells}
    surface_dir = authority.root / "surfaces" / str(surface["surface_id"])
    status_path = surface_dir / "status.json"
    summary_path = surface_dir / "summary.json"
    status = _load_json(status_path)
    summary = _load_json(summary_path)
    if (
        status.get("schema_version") != SURFACE_STATUS_SCHEMA
        or status.get("study_id") != contract.study_id
        or status.get("surface_id") != surface["surface_id"]
        or status.get("official_test_read") is not False
        or summary.get("schema_version") != SURFACE_SUMMARY_SCHEMA
        or summary.get("study_id") != contract.study_id
        or summary.get("surface") != surface
        or summary.get("official_test_read") is not False
        or summary.get("expected_cells") != len(expected_cells)
    ):
        raise ValueError(f"Surface status/summary identity mismatch under {surface_dir}.")
    if summary.get("status") != status.get("status"):
        raise ValueError(f"Surface status and summary state disagree under {surface_dir}.")
    state = str(summary.get("status"))
    if state not in {"running", "complete", "failed", "unresolved_probe"}:
        raise ValueError(f"Unsupported surface state {state!r} under {surface_dir}.")
    initializer = _validate_initializer(authority)
    source, source_sha256 = _validate_source_config(authority, contract)
    probe, probe_semantic_sha256, probe_file_sha256 = _validate_probe(
        authority,
        contract,
        source_sha256=source_sha256,
        require_complete=state != "unresolved_probe",
    )

    by_index: dict[int, Path] = {}
    for directory in _cell_directories(surface_dir):
        cell_path = directory / "cell.json"
        if not cell_path.is_file():
            # A currently training cell is not a terminal bundle yet.
            continue
        cell = _load_json(cell_path)
        cell_index = cell.get("index")
        if isinstance(cell_index, bool) or not isinstance(cell_index, int):
            raise ValueError(f"Invalid cell index in {directory}.")
        if cell_index not in expected_by_index or cell_index in by_index:
            raise ValueError(f"Unexpected/duplicate sparse cell index {cell_index} in {surface_dir}.")
        by_index[cell_index] = directory
    observed_rows = [
        _validate_cell(
            authority,
            contract,
            run_dir=directory,
            expected=expected_by_index[cell_index],
            source_config=source,
            source_sha256=source_sha256,
            probe=probe,
            probe_semantic_sha256=probe_semantic_sha256,
            probe_file_sha256=probe_file_sha256,
        )
        for cell_index, directory in sorted(by_index.items())
    ]
    rows = [
        row
        for row in observed_rows
        if row["canonical_outcome"] in {"numeric_complete", "nonfinite"}
    ]
    running_rows = [
        row for row in observed_rows if row["canonical_outcome"] == "running"
    ]
    observed_counts = dict(Counter(str(row["raw_status"]) for row in rows))
    candidates = summary.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"Surface summary candidates are malformed under {surface_dir}.")
    if state == "complete":
        if (
            summary.get("complete") is not True
            or summary.get("terminal_cells") != len(expected_cells)
            or set(by_index) != set(expected_by_index)
            or len(candidates) != len(expected_cells)
            or summary.get("status_counts") != observed_counts
        ):
            raise ValueError(f"Complete surface coverage is inconsistent under {surface_dir}.")
    elif state == "unresolved_probe":
        if summary.get("terminal_cells") != 0 or candidates or observed_rows:
            raise ValueError(f"Malformed unresolved-probe surface under {surface_dir}.")
    elif len(rows) > len(expected_cells):
        raise ValueError(f"Partial surface exceeds declared coverage under {surface_dir}.")
    # During a live surface the summary may lag the just-completed cell by one
    # atomic update; canonical bundles are authoritative for the partial count.
    return {
        "task_id": index,
        "surface_id": surface["surface_id"],
        "architecture": surface["architecture"],
        "scheme": surface["scheme"],
        "optimizer": surface["optimizer"],
        "declared_target": surface["target"],
        "target": authority.target,
        "root": str(authority.root),
        "resolved_sha256": sha256_file(authority.root / "study.resolved.json"),
        "source_config_sha256": source_sha256,
        "probe_sha256": probe_file_sha256,
        "initializer": initializer,
        "state": state,
        "expected_cells": len(expected_cells),
        "terminal_cells": len(rows),
        "numeric_cells": sum(row["canonical_outcome"] == "numeric_complete" for row in rows),
        "nonfinite_cells": sum(row["canonical_outcome"] == "nonfinite" for row in rows),
        "running_cells": len(running_rows),
        "running_cell_indices": [int(row["index"]) for row in running_rows],
        "status_counts": observed_counts,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "cells": observed_rows,
    }


def _receipt_paths(
    *, study_root: Path | None, receipt_roots: Sequence[Path]
) -> list[Path]:
    roots = [Path(path).expanduser().resolve() for path in receipt_roots]
    if study_root is not None:
        roots.append(Path(study_root).expanduser().resolve() / "transport_receipts")
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        candidates = [root] if root.is_file() else sorted(root.rglob("*.json")) if root.is_dir() else []
        for candidate in candidates:
            path = candidate.resolve()
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def _receipt_inventory(
    contract: Contract,
    *,
    study_root: Path | None,
    receipt_roots: Sequence[Path],
) -> dict[int, tuple[Path, dict[str, Any]]]:
    inventory: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in _receipt_paths(study_root=study_root, receipt_roots=receipt_roots):
        receipt = _load_json(path)
        if receipt.get("schema_version") != RECEIPT_SCHEMA:
            continue
        if receipt.get("study_id") != contract.study_id:
            continue
        task = receipt.get("task_id")
        if isinstance(task, bool) or not isinstance(task, int) or not 0 <= task < 12:
            raise ValueError(f"Receipt {path} has invalid task_id {task!r}.")
        if task in inventory:
            raise ValueError(f"Duplicate terminal receipts for task {task}.")
        inventory[task] = (path, receipt)
    return inventory


def _validate_receipt(
    authority: Authority,
    contract: Contract,
    authority_report: Mapping[str, Any],
    record: tuple[Path, dict[str, Any]],
) -> dict[str, Any]:
    path, receipt = record
    surface = authority.surface
    task = int(surface["index"])
    state = authority_report["state"]
    expected_cells = len(surface_cells(contract.study, surface))
    required = {
        "schema_version": RECEIPT_SCHEMA,
        "study_id": contract.study_id,
        "surface": surface,
        "task_id": task,
        "target": authority.target,
        "source_commit": authority.resolved["code"]["commit"],
        "source_archive_sha256": authority.resolved["code"]["source_archive_sha256"],
        "study_config_sha256": contract.sha256,
        "initializer_checkpoint_sha256": EXPECTED_INITIALIZER_SHA256[
            str(surface["architecture"])
        ],
        "summary_status": state,
        "expected_cells": expected_cells,
        "terminal_cells": authority_report["terminal_cells"],
        "validated_cell_bundles": authority_report["terminal_cells"],
        "status_counts": authority_report["status_counts"],
        "official_test_read": False,
        "semantic_pass": True,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"Receipt {path} has invalid {key}.")
    if state == "complete":
        if authority_report["terminal_cells"] != expected_cells:
            raise ValueError(f"Receipt {path} does not bind complete cell coverage.")
    elif state == "unresolved_probe":
        if authority_report["terminal_cells"] != 0:
            raise ValueError(f"Unresolved-probe receipt {path} has terminal cells.")
    else:
        raise ValueError(f"Receipt {path} binds unsupported terminal state {state!r}.")
    environment_id = receipt.get("environment_id")
    if not isinstance(environment_id, str) or not environment_id:
        raise ValueError(f"Receipt {path} lacks environment_id.")
    summary_path = Path(str(authority_report["summary_path"]))
    if (
        not _path_has_suffix(
            receipt.get("summary_path"),
            (
                "shards",
                authority.root.name,
                "surfaces",
                str(surface["surface_id"]),
                "summary.json",
            ),
        )
        or receipt.get("summary_sha256") != sha256_file(summary_path)
    ):
        raise ValueError(f"Receipt {path} does not bind the locally collected summary.")
    local_summary = _load_json(summary_path)
    if receipt.get("best_observation") != local_summary.get("best_observation"):
        raise ValueError(f"Receipt {path} best observation differs from local summary.")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "task_id": task,
        "surface_id": surface["surface_id"],
        "target": authority.target,
        "environment_id": environment_id,
        "summary_sha256": receipt["summary_sha256"],
        "validated_cell_bundles": receipt["validated_cell_bundles"],
    }


def validate_collection(
    *,
    study_config_path: Path = DEFAULT_STUDY,
    study_root: Path | None = None,
    shard_roots: Sequence[Path] = (),
    receipt_roots: Sequence[Path] = (),
    require_complete: bool = False,
) -> dict[str, Any]:
    contract = load_contract(study_config_path)
    roots = discover_shard_roots(study_root=study_root, shard_roots=shard_roots)
    authorities = _load_authorities(roots, contract)
    reports = [_validate_authority(authority, contract) for authority in authorities]
    by_task = {int(report["task_id"]): report for report in reports}
    authority_by_task = {int(item.surface["index"]): item for item in authorities}
    receipts = _receipt_inventory(
        contract, study_root=study_root, receipt_roots=receipt_roots
    )
    orphan_receipts = sorted(set(receipts) - set(by_task))
    if orphan_receipts:
        raise ValueError(
            "Terminal receipts cannot be locally rebound because their authorities "
            f"are absent: {orphan_receipts!r}."
        )
    validated_receipts: list[dict[str, Any]] = []
    for task, report in sorted(by_task.items()):
        record = receipts.get(task)
        if record is None:
            report["receipt"] = None
            continue
        receipt = _validate_receipt(
            authority_by_task[task], contract, report, record
        )
        report["receipt"] = receipt
        validated_receipts.append(receipt)
    terminal_cells = sum(int(report["terminal_cells"]) for report in reports)
    numeric_cells = sum(int(report["numeric_cells"]) for report in reports)
    nonfinite_cells = sum(int(report["nonfinite_cells"]) for report in reports)
    complete_tasks = sorted(
        int(report["task_id"])
        for report in reports
        if report["state"] == "complete"
        and report["terminal_cells"] == report["expected_cells"]
        and report.get("receipt") is not None
    )
    missing_tasks = sorted(set(range(12)) - set(by_task))
    incomplete_tasks = sorted(set(by_task) - set(complete_tasks))
    complete = (
        len(reports) == EXPECTED_SURFACES
        and len(validated_receipts) == EXPECTED_SURFACES
        and terminal_cells == EXPECTED_CELLS
        and complete_tasks == list(range(12))
    )
    report = {
        "schema_version": VALIDATION_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study_id": contract.study_id,
        "study_config": str(contract.path),
        "study_config_sha256": contract.sha256,
        "parent_study_config": str(contract.parent_path),
        "parent_study_config_sha256": sha256_file(contract.parent_path),
        "evidence_class": "ordinary_mnist_exploratory",
        "exploratory_noncanonical": True,
        "canonical_lr_handoff": False,
        "paper_evidence": False,
        "official_test_read": False,
        "status": "complete" if complete else "partial",
        "complete": complete,
        "coverage": {
            "authority_count": len(reports),
            "expected_authority_count": EXPECTED_SURFACES,
            "validated_receipt_count": len(validated_receipts),
            "expected_receipt_count": EXPECTED_SURFACES,
            "terminal_cell_count": terminal_cells,
            "numeric_cell_count": numeric_cells,
            "nonfinite_cell_count": nonfinite_cells,
            "expected_cell_count": EXPECTED_CELLS,
            "complete_task_ids": complete_tasks,
            "missing_task_ids": missing_tasks,
            "incomplete_task_ids": incomplete_tasks,
            "expected_sparse_cell_counts": list(EXPECTED_CELL_COUNTS),
        },
        "authorities": reports,
        "receipts": validated_receipts,
    }
    if require_complete and not complete:
        raise IncompleteCollectionError(report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--study-root", type=Path)
    parser.add_argument("--shard-root", type=Path, action="append", default=[])
    parser.add_argument("--receipt-root", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate_collection(
            study_config_path=args.study_config,
            study_root=args.study_root,
            shard_roots=args.shard_root,
            receipt_roots=args.receipt_root,
            require_complete=args.require_complete,
        )
    except IncompleteCollectionError as error:
        if args.output_json is not None:
            atomic_write_json(args.output_json.expanduser().resolve(), error.report)
        print(str(error), file=sys.stderr)
        return 2
    if args.output_json is not None:
        atomic_write_json(args.output_json.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
