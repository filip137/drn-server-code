"""Fresh bounded-initialization rho comparison for perfect-diode Conv1/Conv2.

This workflow is deliberately separate from the completed seed-0 v1 study.
It materializes two new content-derived studies, reruns assets, fixed-T/K
security, optimizer probes, canaries, and three-epoch candidates, and never
loads a numerical artifact from either historical comparison card.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .identity import sha256_file, sha256_json
from .io import atomic_write_bytes, atomic_write_json, read_json
from .perfectdiode_hparam_runtime import (
    execute_perfectdiode_stage_entry,
    require_official_test_excluded,
    rho_cell_id,
)
from .perfectdiode_hparam_spec import (
    CANDIDATE_TOTAL_STEPS,
    PerfectDiodeHparamStudySpec,
    candidate_grid,
    select_surface_candidates,
)


CONFIG_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-initialization-rho-comparison/v1"
)
RESOLVED_STUDY_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-bounded-init-study/v1"
)
MANIFEST_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-initialization-rho-manifest/v1"
)
GATE_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-initialization-rho-gate-completion/v1"
)
ENTRY_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-initialization-rho-entry-completion/v1"
)
SURFACE_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-initialization-rho-surface-completion/v1"
)
SUMMARY_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-initialization-rho-summary/v1"
)
COMPARISON_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-initialization-rho-completion/v1"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv12_initialization_rho_comparison_20260727_v1.json"
)
BASE_CONFIG_FILE_SHA256 = (
    "647f92e01c8619951367d71cadd6320c6924c0223a2f5ed344f7ff1739bac5c1"
)
BASE_CONFIG_CANONICAL_SHA256 = (
    "699804e1f7c35f65f50656db4af0b120dd3a3d0f2fb95e3a17b65ebfe7b78535"
)
PARENT_STUDY_ID = (
    "lrstudy_5afe8bc7c9180cd1c677d1d87a497e1d6a89c1599cd015d7a6c127ebb18f2e46"
)
INITIALIZER_ORDER = (
    "bounded-uniform-1e-5-1e-4",
    "bounded-kaiming-uniform-1e-5-1e-4",
)
OPTIMIZER_ORDER = ("sgd", "adam")
EXPECTED_BOUNDS = [1e-5, 1e-4]


class InitRhoComparisonError(ValueError):
    """A config, manifest, execution entry, or result failed closed."""


def _error(path: str, expected: str, provided: Any) -> InitRhoComparisonError:
    return InitRhoComparisonError(
        f"Expected {path} to be {expected}. Provided value: {provided!r}."
    )


def _strict_json(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _error(str(path), "JSON objects with unique keys", key)
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise _error(str(path), "finite JSON numbers", value)

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except OSError as exc:
        raise _error(str(path), "a readable UTF-8 JSON file", str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise _error(str(path), "strict JSON", str(exc)) from exc
    if not isinstance(value, dict):
        raise _error(str(path), "a JSON object", type(value).__name__)
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    path: str,
) -> None:
    provided = set(value)
    if provided != expected:
        raise _error(
            path,
            f"exactly the keys {sorted(expected)!r}",
            {
                "missing": sorted(expected - provided),
                "unknown": sorted(provided - expected),
            },
        )


def _positive_axis(value: Any, path: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise _error(path, "a non-empty increasing JSON array", value)
    result: list[float] = []
    for index, item in enumerate(value):
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) <= 0.0
        ):
            raise _error(f"{path}[{index}]", "a positive finite number", item)
        result.append(float(item))
    if result != sorted(set(result)):
        raise _error(path, "strictly increasing unique values", result)
    return result


def _base_spec(config: Mapping[str, Any], config_path: Path) -> PerfectDiodeHparamStudySpec:
    base = config["base_contract"]
    base_path = config_path.parent / str(base["path"])
    if sha256_file(base_path) != BASE_CONFIG_FILE_SHA256:
        raise _error(
            "base_contract.file_sha256",
            BASE_CONFIG_FILE_SHA256,
            sha256_file(base_path),
        )
    spec = PerfectDiodeHparamStudySpec.from_path(base_path)
    if spec.config_sha256 != BASE_CONFIG_CANONICAL_SHA256:
        raise _error(
            "base_contract.canonical_sha256",
            BASE_CONFIG_CANONICAL_SHA256,
            spec.config_sha256,
        )
    return spec


def load_config(
    path: str | Path = DEFAULT_CONFIG,
) -> tuple[Path, dict[str, Any], PerfectDiodeHparamStudySpec]:
    """Load and validate the complete user-approved comparison design."""

    source = Path(path).expanduser().resolve()
    data = _strict_json(source)
    _exact_keys(
        data,
        {
            "schema_version",
            "experiment_id",
            "title",
            "protocol_id",
            "evidence_scope",
            "base_contract",
            "parent_lineage",
            "initializers",
            "surfaces",
            "decision",
            "execution",
            "approval",
        },
        "config",
    )
    if data["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise _error("schema_version", CONFIG_SCHEMA_VERSION, data["schema_version"])
    if data["evidence_scope"] != "diagnostic":
        raise _error("evidence_scope", "'diagnostic'", data["evidence_scope"])
    if data["approval"].get("status") != "approved":
        raise _error("approval.status", "'approved'", data["approval"].get("status"))
    base = data["base_contract"]
    if not isinstance(base, Mapping):
        raise _error("base_contract", "a JSON object", base)
    if (
        base.get("file_sha256") != BASE_CONFIG_FILE_SHA256
        or base.get("canonical_sha256") != BASE_CONFIG_CANONICAL_SHA256
    ):
        raise _error(
            "base_contract hashes",
            "the immutable completed v1 config hashes",
            base,
        )
    lineage = data["parent_lineage"]
    if (
        lineage.get("study_id") != PARENT_STUDY_ID
        or lineage.get("reuse_numerical_outputs") is not False
        or lineage.get("rerun_under_parent_identity") is not False
    ):
        raise _error(
            "parent_lineage",
            "comparison-only lineage with both numerical reuse flags false",
            lineage,
        )
    initializers = data["initializers"]
    if (
        not isinstance(initializers, list)
        or [item.get("initializer_id") for item in initializers]
        != list(INITIALIZER_ORDER)
    ):
        raise _error(
            "initializers",
            f"the exact order {INITIALIZER_ORDER!r}",
            initializers,
        )
    expected_modes = ("bounded_uniform", "bounded_kaiming_uniform")
    for index, (initializer, mode) in enumerate(zip(initializers, expected_modes)):
        if (
            initializer.get("weight_initialization") != mode
            or initializer.get("conductance_bounds") != EXPECTED_BOUNDS
            or initializer.get("weight_gain") != 1.0
        ):
            raise _error(
                f"initializers[{index}]",
                f"mode={mode!r}, bounds={EXPECTED_BOUNDS!r}, gain=1.0",
                initializer,
            )

    base_spec = _base_spec(data, source)
    row_by_id = {row["row_id"]: row for row in base_spec.rows}
    surfaces = data["surfaces"]
    if not isinstance(surfaces, list) or len(surfaces) != 11:
        raise _error("surfaces", "the eleven referenced surfaces", surfaces)
    seen: set[tuple[str, str]] = set()
    conv1_count = 0
    conv2_count = 0
    total_cells = 0
    for index, surface in enumerate(surfaces):
        if not isinstance(surface, Mapping):
            raise _error(f"surfaces[{index}]", "a JSON object", surface)
        _exact_keys(
            surface,
            {"row_id", "optimizer", "rho_conv", "rho_dense"},
            f"surfaces[{index}]",
        )
        row_id = surface["row_id"]
        optimizer = surface["optimizer"]
        if row_id not in row_by_id:
            raise _error(f"surfaces[{index}].row_id", "a frozen v1 row", row_id)
        if optimizer not in OPTIMIZER_ORDER:
            raise _error(
                f"surfaces[{index}].optimizer",
                f"one of {OPTIMIZER_ORDER!r}",
                optimizer,
            )
        key = (str(row_id), str(optimizer))
        if key in seen:
            raise _error("surfaces", "unique row/optimizer pairs", key)
        seen.add(key)
        conv = _positive_axis(surface["rho_conv"], f"surfaces[{index}].rho_conv")
        dense = _positive_axis(
            surface["rho_dense"], f"surfaces[{index}].rho_dense"
        )
        architecture = row_by_id[str(row_id)]["architecture"]
        if architecture == "conv1":
            conv1_count += 1
            if conv != [0.001, 0.003, 0.009] or dense != [
                0.0033333333333333335,
                0.01,
                0.03,
            ]:
                raise _error(
                    f"surfaces[{index}] rho grid",
                    "the exact referenced Conv1 3x3 core",
                    {"rho_conv": conv, "rho_dense": dense},
                )
        else:
            conv2_count += 1
            expected_conv = (
                [0.003, 0.009]
                if row_by_id[str(row_id)]["scheme"] == "legacy"
                else [0.009, 0.027]
            )
            if conv != expected_conv or dense != [0.03, 0.09]:
                raise _error(
                    f"surfaces[{index}] rho grid",
                    "the exact referenced Conv2 2x2 high-rho corner",
                    {"rho_conv": conv, "rho_dense": dense},
                )
        total_cells += len(conv) * len(dense)
    if (conv1_count, conv2_count, total_cells) != (6, 5, 74):
        raise _error(
            "surface coverage",
            "six Conv1 surfaces, five Conv2 surfaces, and 74 cells",
            (conv1_count, conv2_count, total_cells),
        )
    if data["decision"].get("candidate_steps") != CANDIDATE_TOTAL_STEPS:
        raise _error(
            "decision.candidate_steps",
            CANDIDATE_TOTAL_STEPS,
            data["decision"].get("candidate_steps"),
        )
    if data["execution"].get("official_test_read") is not False:
        raise _error(
            "execution.official_test_read",
            "false",
            data["execution"].get("official_test_read"),
        )
    return source, data, base_spec


def _resolved_study(
    config: Mapping[str, Any],
    base_spec: PerfectDiodeHparamStudySpec,
    initializer: Mapping[str, Any],
) -> PerfectDiodeHparamStudySpec:
    data = base_spec.data
    initializer_id = str(initializer["initializer_id"])
    data["schema_version"] = RESOLVED_STUDY_SCHEMA_VERSION
    data["run_schema_version"] = (
        "mnist-conv-perfectdiode-bounded-init-run/v1"
    )
    data["name"] = (
        "perfect-diode-conv1-conv2-initialization-rho-"
        f"{initializer_id}"
    )
    data["protocol_id"] = config["protocol_id"]
    data["status"] = {
        "measurements": "pending",
        "study_role": "ordinary_mnist_initialization_rho_diagnostic",
        "medium_affine_paper_handoff": "unresolved",
        "final_paper_training_authorized": False,
    }
    data["model"]["conductance_bounds"] = copy.deepcopy(
        initializer["conductance_bounds"]
    )
    data["model"]["weight_initialization"] = initializer[
        "weight_initialization"
    ]
    data["model"]["weight_gains"] = float(initializer["weight_gain"])
    data["model"]["shared_initialization"] = (
        "fresh_seed0_content_equal_checkpoint_per_initializer_architecture"
    )
    data["successor_contract"] = {
        "experiment_id": config["experiment_id"],
        "initializer_id": initializer_id,
        "initializer_definition": initializer["definition"],
        "parent_study_id": PARENT_STUDY_ID,
        "reuse_parent_numerical_outputs": False,
        "official_test_read": False,
    }
    config_sha256 = sha256_json(data)
    identity = {
        "identity_schema": (
            "mnist-conv-perfectdiode-bounded-init-study-id/v1"
        ),
        "resolved_study": data,
    }
    study_id = "lrinitstudy_" + sha256_json(identity)
    return PerfectDiodeHparamStudySpec(data, study_id, config_sha256)


def build_manifest(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    source_commit: str,
    source_archive_sha256: str,
    environment_contract_sha256: str,
) -> dict[str, Any]:
    source, config, base_spec = load_config(config_path)
    config_sha256 = sha256_json(config)
    comparison_id = "pdinitrho_" + sha256_json(
        {
            "identity_schema": (
                "mnist-conv-perfectdiode-initialization-rho-id/v1"
            ),
            "config": config,
        }
    )
    row_by_id = {row["row_id"]: row for row in base_spec.rows}
    studies: list[dict[str, Any]] = []
    gate_entries: list[dict[str, Any]] = []
    surface_entries: list[dict[str, Any]] = []
    all_entries: list[dict[str, Any]] = []
    initializers = {
        item["initializer_id"]: item for item in config["initializers"]
    }
    surface_specs = list(config["surfaces"])
    for initializer_id in INITIALIZER_ORDER:
        spec = _resolved_study(config, base_spec, initializers[initializer_id])
        studies.append(
            {
                "initializer_id": initializer_id,
                "study_id": spec.study_id,
                "config_sha256": spec.config_sha256,
                "resolved_config_path": (
                    f"studies/{spec.study_id}/resolved_config.json"
                ),
            }
        )
        required_rows = []
        for surface in surface_specs:
            if surface["row_id"] not in required_rows:
                required_rows.append(surface["row_id"])
        for row_id in required_rows:
            row = row_by_id[row_id]
            entry = {
                "stage": "gate",
                "entry_index": len(all_entries),
                "entry_id": f"{initializer_id}--{row_id}",
                "initializer_id": initializer_id,
                "study_id": spec.study_id,
                "row_id": row_id,
                "architecture": row["architecture"],
                "scheme": row["scheme"],
            }
            gate_entries.append(entry)
            all_entries.append(entry)
        for surface in surface_specs:
            row = row_by_id[surface["row_id"]]
            surface_id = f"{surface['row_id']}--{surface['optimizer']}"
            cells = []
            for rho_conv, rho_dense in candidate_grid(
                surface["rho_conv"], surface["rho_dense"]
            ):
                cells.append(
                    {
                        "cell_id": rho_cell_id(
                            study_id=spec.study_id,
                            surface_id=surface_id,
                            rho_conv=rho_conv,
                            rho_dense=rho_dense,
                        ),
                        "rho_conv": rho_conv,
                        "rho_dense": rho_dense,
                    }
                )
            entry = {
                "stage": "surface",
                "entry_index": len(all_entries),
                "surface_index": len(surface_entries),
                "entry_id": f"{initializer_id}--{surface_id}",
                "initializer_id": initializer_id,
                "study_id": spec.study_id,
                "row_id": surface["row_id"],
                "architecture": row["architecture"],
                "scheme": row["scheme"],
                "optimizer": surface["optimizer"],
                "surface_id": surface_id,
                "rho_conv": list(surface["rho_conv"]),
                "rho_dense": list(surface["rho_dense"]),
                "cells": cells,
            }
            surface_entries.append(entry)
            all_entries.append(entry)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "comparison_id": comparison_id,
        "experiment_id": config["experiment_id"],
        "config_path": source.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": config_sha256,
        "parent_study_id": PARENT_STUDY_ID,
        "reuse_parent_numerical_outputs": False,
        "source_commit": source_commit,
        "source_archive_sha256": source_archive_sha256,
        "environment_contract_sha256": environment_contract_sha256,
        "studies": studies,
        "gate_entries": gate_entries,
        "surface_entries": surface_entries,
        "entries": all_entries,
        "counts": {
            "initializers": 2,
            "gate_entries": 12,
            "surface_entries": 22,
            "rho_cells": 148,
            "maximum_promoted_candidates": 148,
        },
        "official_test_read": False,
    }
    manifest["manifest_id"] = "pdinitmanifest_" + sha256_json(manifest)
    return manifest


def materialize(
    config_path: str | Path,
    root: str | Path,
    *,
    source_commit: str,
    source_archive_sha256: str,
    environment_contract: str | Path,
) -> dict[str, Any]:
    """Write the immutable manifest and both fresh resolved study configs."""

    destination = Path(root).expanduser().resolve()
    environment_path = Path(environment_contract).expanduser().resolve()
    source, config, base_spec = load_config(config_path)
    manifest = build_manifest(
        source,
        source_commit=source_commit,
        source_archive_sha256=source_archive_sha256,
        environment_contract_sha256=sha256_file(environment_path),
    )
    destination.mkdir(parents=True, exist_ok=True)
    config_snapshot = destination / "config.snapshot.json"
    environment_snapshot = destination / "environment_contract.json"
    manifest_path = destination / "manifest.json"
    environment_value = _strict_json(environment_path)
    environment_bytes = environment_path.read_bytes()
    if environment_snapshot.exists():
        if (
            read_json(environment_snapshot) != environment_value
            or environment_snapshot.read_bytes() != environment_bytes
        ):
            raise _error(
                str(environment_snapshot),
                "the existing immutable byte-identical environment contract",
                environment_value,
            )
    else:
        atomic_write_bytes(environment_snapshot, environment_bytes)
    for path, value in (
        (config_snapshot, config),
        (manifest_path, manifest),
    ):
        if path.exists() and read_json(path) != value:
            raise _error(str(path), "the existing immutable content", value)
        if not path.exists():
            atomic_write_json(path, value, canonical=True)
    initializer_by_id = {
        item["initializer_id"]: item for item in config["initializers"]
    }
    for record in manifest["studies"]:
        spec = _resolved_study(
            config,
            base_spec,
            initializer_by_id[record["initializer_id"]],
        )
        study_path = destination / record["resolved_config_path"]
        study_path.parent.mkdir(parents=True, exist_ok=True)
        if study_path.exists() and read_json(study_path) != spec.data:
            raise _error(str(study_path), "the immutable resolved study", spec.data)
        if not study_path.exists():
            atomic_write_json(study_path, spec.data, canonical=True)
    validate_root(destination)
    return manifest


def validate_root(root: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    destination = Path(root).expanduser().resolve()
    manifest = read_json(destination / "manifest.json")
    config = read_json(destination / "config.snapshot.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise _error(
            "manifest.schema_version",
            MANIFEST_SCHEMA_VERSION,
            manifest.get("schema_version"),
        )
    if manifest.get("config_sha256") != sha256_json(config):
        raise _error(
            "manifest.config_sha256",
            sha256_json(config),
            manifest.get("config_sha256"),
        )
    if manifest.get("reuse_parent_numerical_outputs") is not False:
        raise _error(
            "manifest.reuse_parent_numerical_outputs",
            "false",
            manifest.get("reuse_parent_numerical_outputs"),
        )
    if manifest.get("counts") != {
        "initializers": 2,
        "gate_entries": 12,
        "surface_entries": 22,
        "rho_cells": 148,
        "maximum_promoted_candidates": 148,
    }:
        raise _error("manifest.counts", "the frozen workload counts", manifest.get("counts"))
    environment_path = destination / "environment_contract.json"
    if sha256_file(environment_path) != manifest["environment_contract_sha256"]:
        raise _error(
            "environment_contract SHA-256",
            manifest["environment_contract_sha256"],
            sha256_file(environment_path),
        )
    for record in manifest["studies"]:
        resolved_path = destination / record["resolved_config_path"]
        data = read_json(resolved_path)
        if sha256_json(data) != record["config_sha256"]:
            raise _error(
                f"{resolved_path} canonical SHA-256",
                record["config_sha256"],
                sha256_json(data),
            )
    return destination, manifest, config


def _runtime_spec(
    root: Path,
    manifest: Mapping[str, Any],
    study_id: str,
) -> PerfectDiodeHparamStudySpec:
    records = [record for record in manifest["studies"] if record["study_id"] == study_id]
    if len(records) != 1:
        raise _error("study_id", "one manifest study", study_id)
    record = records[0]
    data = read_json(root / record["resolved_config_path"])
    return PerfectDiodeHparamStudySpec(
        data,
        str(record["study_id"]),
        str(record["config_sha256"]),
    )


def _entry(
    manifest: Mapping[str, Any],
    stage: str,
    index: int,
) -> dict[str, Any]:
    entries = manifest[f"{stage}_entries"]
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(entries):
        raise _error(f"{stage} entry index", f"an integer in [0,{len(entries)-1}]", index)
    return copy.deepcopy(entries[index])


def _artifact_records(paths: Sequence[Path], root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def _files_below(path: Path, *, exclude: set[Path] | None = None) -> list[Path]:
    excluded = {item.resolve() for item in (exclude or set())}
    return [
        item
        for item in path.rglob("*")
        if item.is_file() and item.resolve() not in excluded
    ]


def _validate_completion(
    path: Path,
    root: Path,
    *,
    schema_version: str,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema_version") != schema_version:
        raise _error(
            f"{path}.schema_version",
            schema_version,
            value.get("schema_version"),
        )
    for key, expected in identity.items():
        if value.get(key) != expected:
            raise _error(f"{path}.{key}", repr(expected), value.get(key))
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise _error(f"{path}.artifacts", "a non-empty artifact list", artifacts)
    for record in artifacts:
        artifact = root / record["path"]
        if (
            not artifact.is_file()
            or artifact.stat().st_size != record["bytes"]
            or sha256_file(artifact) != record["sha256"]
        ):
            raise _error(
                f"{path} artifact",
                "an existing file with the recorded size and SHA-256",
                record,
            )
    return value


def _quarantine_partial(path: Path, root: Path) -> None:
    if not path.exists():
        return
    quarantine = root / "incomplete_attempts"
    quarantine.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    target = quarantine / f"{path.name}-{timestamp}"
    os.replace(path, target)


def _write_entry_completion(
    path: Path,
    root: Path,
    *,
    entry_kind: str,
    entry_id: str,
    study_id: str,
    files: Sequence[Path],
    status: str = "complete",
) -> dict[str, Any]:
    value = {
        "schema_version": ENTRY_COMPLETION_SCHEMA_VERSION,
        "entry_kind": entry_kind,
        "entry_id": entry_id,
        "study_id": study_id,
        "status": status,
        "official_test_read": False,
        "artifacts": _artifact_records(files, root),
    }
    atomic_write_json(path, value, canonical=True)
    return value


def run_gate(
    root: str | Path,
    index: int,
    *,
    data_root: str | Path,
    device: str,
    download: bool = False,
) -> dict[str, Any]:
    destination, manifest, _config = validate_root(root)
    entry = _entry(manifest, "gate", index)
    spec = _runtime_spec(destination, manifest, entry["study_id"])
    study_root = destination / "studies" / spec.study_id
    row_root = study_root / "rows" / entry["row_id"]
    completion_path = row_root / "gate_completion.json"
    identity = {
        "entry_id": entry["entry_id"],
        "study_id": spec.study_id,
        "row_id": entry["row_id"],
    }
    if completion_path.is_file():
        value = _validate_completion(
            completion_path,
            destination,
            schema_version=GATE_COMPLETION_SCHEMA_VERSION,
            identity=identity,
        )
        return {**value, "resume": "skipped_hash_verified_complete"}
    if row_root.exists():
        _quarantine_partial(row_root, study_root)
    row_root.mkdir(parents=True, exist_ok=True)
    asset = execute_perfectdiode_stage_entry(
        spec,
        row_root,
        "assets",
        {
            "entry_id": entry["architecture"],
            "architecture": entry["architecture"],
        },
        data_root=data_root,
        device=device,
        download=download,
    )
    security = execute_perfectdiode_stage_entry(
        spec,
        row_root,
        "fixed_tk_gradient_security",
        {
            "entry_id": entry["row_id"],
            "row_id": entry["row_id"],
            "asset_entry_id": entry["architecture"],
        },
        data_root=data_root,
        device=device,
        download=download,
    )
    if security.get("passed") is not True:
        raise InitRhoComparisonError(
            "Expected the fresh fixed-T/K security check to pass. "
            f"Provided value: {security!r}."
        )
    files = _files_below(row_root, exclude={completion_path})
    value = {
        "schema_version": GATE_COMPLETION_SCHEMA_VERSION,
        **identity,
        "architecture": entry["architecture"],
        "scheme": entry["scheme"],
        "initializer_id": entry["initializer_id"],
        "status": "complete",
        "security_passed": True,
        "initialization_checkpoint_sha256": asset[
            "initialization_checkpoint_sha256"
        ],
        "initialization_tensor_sha256": asset["initialization_tensor_sha256"],
        "official_test_read": False,
        "artifacts": _artifact_records(files, destination),
    }
    atomic_write_json(completion_path, value, canonical=True)
    return value


def _validated_gate(
    destination: Path,
    manifest: Mapping[str, Any],
    surface: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    matches = [
        entry
        for entry in manifest["gate_entries"]
        if entry["study_id"] == surface["study_id"]
        and entry["row_id"] == surface["row_id"]
    ]
    if len(matches) != 1:
        raise _error("surface gate", "one matching gate entry", matches)
    gate = matches[0]
    row_root = (
        destination
        / "studies"
        / surface["study_id"]
        / "rows"
        / surface["row_id"]
    )
    value = _validate_completion(
        row_root / "gate_completion.json",
        destination,
        schema_version=GATE_COMPLETION_SCHEMA_VERSION,
        identity={
            "entry_id": gate["entry_id"],
            "study_id": surface["study_id"],
            "row_id": surface["row_id"],
        },
    )
    if value.get("security_passed") is not True:
        raise _error("gate.security_passed", "true", value.get("security_passed"))
    return row_root, value


def _entry_completion(
    entry_dir: Path,
    destination: Path,
    *,
    entry_kind: str,
    entry_id: str,
    study_id: str,
) -> dict[str, Any] | None:
    completion = entry_dir / "completion.json"
    if not completion.is_file():
        return None
    return _validate_completion(
        completion,
        destination,
        schema_version=ENTRY_COMPLETION_SCHEMA_VERSION,
        identity={
            "entry_kind": entry_kind,
            "entry_id": entry_id,
            "study_id": study_id,
        },
    )


def _validate_surface_completion(
    path: Path,
    destination: Path,
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    value = _validate_completion(
        path,
        destination,
        schema_version=SURFACE_COMPLETION_SCHEMA_VERSION,
        identity=identity,
    )
    for artifact_record in value["artifacts"]:
        nested_path = destination / artifact_record["path"]
        nested = read_json(nested_path)
        _validate_completion(
            nested_path,
            destination,
            schema_version=ENTRY_COMPLETION_SCHEMA_VERSION,
            identity={
                "entry_kind": nested.get("entry_kind"),
                "entry_id": nested.get("entry_id"),
                "study_id": nested.get("study_id"),
            },
        )
    return value


def _run_probe(
    spec: PerfectDiodeHparamStudySpec,
    destination: Path,
    row_root: Path,
    surface: Mapping[str, Any],
    *,
    data_root: str | Path,
    device: str,
    download: bool,
) -> dict[str, Any]:
    entry_id = surface["surface_id"]
    entry_dir = row_root / "stages" / "optimizer_probe" / "entries" / entry_id
    completed = _entry_completion(
        entry_dir,
        destination,
        entry_kind="optimizer_probe",
        entry_id=entry_id,
        study_id=spec.study_id,
    )
    if completed is not None:
        return read_json(entry_dir / "result.json")
    if entry_dir.exists():
        _quarantine_partial(entry_dir, row_root)
    result = execute_perfectdiode_stage_entry(
        spec,
        row_root,
        "optimizer_probe",
        {
            "entry_id": entry_id,
            "row_id": surface["row_id"],
            "optimizer": surface["optimizer"],
            "asset_entry_id": surface["architecture"],
        },
        data_root=data_root,
        device=device,
        download=download,
    )
    if result.get("probe_stable") is not True:
        raise InitRhoComparisonError(
            "Expected the fresh optimizer probe to be stable. "
            f"Provided value: {result!r}."
        )
    _write_entry_completion(
        entry_dir / "completion.json",
        destination,
        entry_kind="optimizer_probe",
        entry_id=entry_id,
        study_id=spec.study_id,
        files=_files_below(entry_dir, exclude={entry_dir / "completion.json"}),
    )
    return result


def _run_cell(
    spec: PerfectDiodeHparamStudySpec,
    destination: Path,
    manifest: Mapping[str, Any],
    row_root: Path,
    surface: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    data_root: str | Path,
    device: str,
    download: bool,
) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    canary_dir = (
        row_root
        / "stages"
        / "rho_canary_extension"
        / "entries"
        / cell_id
    )
    canary_completion = _entry_completion(
        canary_dir,
        destination,
        entry_kind="rho_canary",
        entry_id=cell_id,
        study_id=spec.study_id,
    )
    if canary_completion is None:
        if canary_dir.exists():
            _quarantine_partial(canary_dir, row_root)
        canary = execute_perfectdiode_stage_entry(
            spec,
            row_root,
            "rho_canary_extension",
            {
                "entry_id": cell_id,
                "row_id": surface["row_id"],
                "optimizer": surface["optimizer"],
                "surface_id": surface["surface_id"],
                "asset_entry_id": surface["architecture"],
                "probe_entry_id": surface["surface_id"],
                "cells": [dict(cell)],
            },
            data_root=data_root,
            device=device,
            download=download,
        )
        _write_entry_completion(
            canary_dir / "completion.json",
            destination,
            entry_kind="rho_canary",
            entry_id=cell_id,
            study_id=spec.study_id,
            files=_files_below(
                canary_dir, exclude={canary_dir / "completion.json"}
            ),
        )
    else:
        canary = read_json(canary_dir / "result.json")
    if len(canary.get("cells", [])) != 1:
        raise _error(f"canary {cell_id}.cells", "one cell", canary.get("cells"))
    canary_cell = canary["cells"][0]

    candidate_dir = (
        row_root
        / "stages"
        / "rho_core_candidates"
        / "entries"
        / cell_id
    )
    candidate_completion = _entry_completion(
        candidate_dir,
        destination,
        entry_kind="rho_candidate",
        entry_id=cell_id,
        study_id=spec.study_id,
    )
    if candidate_completion is None:
        if candidate_dir.exists():
            _quarantine_partial(candidate_dir, row_root)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        if canary_cell.get("safety_clean") is True:
            candidate = execute_perfectdiode_stage_entry(
                spec,
                row_root,
                "rho_core_candidates",
                {
                    "entry_id": cell_id,
                    "cell_id": cell_id,
                    "row_id": surface["row_id"],
                    "optimizer": surface["optimizer"],
                    "surface_id": surface["surface_id"],
                    "asset_entry_id": surface["architecture"],
                    "probe_entry_id": surface["surface_id"],
                    "rho_conv": cell["rho_conv"],
                    "rho_dense": cell["rho_dense"],
                    "source_commit": manifest["source_commit"],
                    "source_archive_sha256": manifest[
                        "source_archive_sha256"
                    ],
                    "environment_sha256": manifest[
                        "environment_contract_sha256"
                    ],
                },
                data_root=data_root,
                device=device,
                download=download,
            )
            completion_status = "complete"
        else:
            candidate = {
                "schema_version": (
                    "mnist-conv-perfectdiode-zero-work-candidate/v1"
                ),
                "status": "zero_work_canary_rejected",
                "study_id": spec.study_id,
                "config_sha256": spec.config_sha256,
                "entry_id": cell_id,
                "cell_id": cell_id,
                "surface_id": surface["surface_id"],
                "row_id": surface["row_id"],
                "optimizer": surface["optimizer"],
                "rho_conv": cell["rho_conv"],
                "rho_dense": cell["rho_dense"],
                "canary_result_sha256": sha256_file(
                    canary_dir / "result.json"
                ),
                "canary_failure": canary_cell.get("safety_failure"),
                "training_completed": False,
                "completed_steps": 0,
                "official_test_read": False,
            }
            atomic_write_json(
                candidate_dir / "result.json", candidate, canonical=True
            )
            completion_status = "zero_work_canary_rejected"
        _write_entry_completion(
            candidate_dir / "completion.json",
            destination,
            entry_kind="rho_candidate",
            entry_id=cell_id,
            study_id=spec.study_id,
            files=_files_below(
                candidate_dir, exclude={candidate_dir / "completion.json"}
            ),
            status=completion_status,
        )
    else:
        candidate = read_json(candidate_dir / "result.json")
    require_official_test_excluded(candidate, require_result_marker=True)
    return {
        "cell_id": cell_id,
        "rho_conv": cell["rho_conv"],
        "rho_dense": cell["rho_dense"],
        "canary_safety_clean": canary_cell.get("safety_clean") is True,
        "canary_result_sha256": sha256_file(canary_dir / "result.json"),
        "candidate_status": candidate.get("status"),
        "candidate_result_sha256": sha256_file(candidate_dir / "result.json"),
    }


def run_surface(
    root: str | Path,
    index: int,
    *,
    data_root: str | Path,
    device: str,
    download: bool = False,
) -> dict[str, Any]:
    destination, manifest, _config = validate_root(root)
    surface = _entry(manifest, "surface", index)
    spec = _runtime_spec(destination, manifest, surface["study_id"])
    row_root, gate = _validated_gate(destination, manifest, surface)
    completion_dir = row_root / "surface_completions"
    completion_dir.mkdir(parents=True, exist_ok=True)
    completion_path = completion_dir / f"{surface['surface_id']}.json"
    identity = {
        "entry_id": surface["entry_id"],
        "study_id": spec.study_id,
        "surface_id": surface["surface_id"],
    }
    if completion_path.is_file():
        value = _validate_surface_completion(
            completion_path,
            destination,
            identity=identity,
        )
        return {**value, "resume": "skipped_hash_verified_complete"}
    probe = _run_probe(
        spec,
        destination,
        row_root,
        surface,
        data_root=data_root,
        device=device,
        download=download,
    )
    cells = [
        _run_cell(
            spec,
            destination,
            manifest,
            row_root,
            surface,
            cell,
            data_root=data_root,
            device=device,
            download=download,
        )
        for cell in surface["cells"]
    ]
    completion_files = [
        row_root
        / "stages"
        / "optimizer_probe"
        / "entries"
        / surface["surface_id"]
        / "completion.json"
    ]
    for cell in surface["cells"]:
        for stage in ("rho_canary_extension", "rho_core_candidates"):
            completion_files.append(
                row_root
                / "stages"
                / stage
                / "entries"
                / cell["cell_id"]
                / "completion.json"
            )
    value = {
        "schema_version": SURFACE_COMPLETION_SCHEMA_VERSION,
        **identity,
        "initializer_id": surface["initializer_id"],
        "row_id": surface["row_id"],
        "architecture": surface["architecture"],
        "scheme": surface["scheme"],
        "optimizer": surface["optimizer"],
        "status": "complete",
        "gate_completion_sha256": sha256_file(
            row_root / "gate_completion.json"
        ),
        "probe_result_sha256": sha256_file(
            row_root
            / "stages"
            / "optimizer_probe"
            / "entries"
            / surface["surface_id"]
            / "result.json"
        ),
        "probe_stable": probe.get("probe_stable") is True,
        "cells": cells,
        "official_test_read": False,
        "artifacts": _artifact_records(completion_files, destination),
    }
    atomic_write_json(completion_path, value, canonical=True)
    return value


def _candidate_record(
    row_root: Path,
    cell: Mapping[str, Any],
) -> dict[str, Any]:
    result = read_json(
        row_root
        / "stages"
        / "rho_core_candidates"
        / "entries"
        / cell["cell_id"]
        / "result.json"
    )
    if result.get("status") == "zero_work_canary_rejected":
        failure = result.get("canary_failure")
        return {
            "candidate_id": cell["cell_id"],
            "rho_conv": cell["rho_conv"],
            "rho_dense": cell["rho_dense"],
            "admissible": False,
            "inadmissible_reason": (
                str(failure) if failure is not None else "canary_rejected"
            ),
        }
    if result.get("safety_admissible") is not True:
        failure = result.get("safety_failure")
        return {
            "candidate_id": cell["cell_id"],
            "rho_conv": cell["rho_conv"],
            "rho_dense": cell["rho_dense"],
            "admissible": False,
            "inadmissible_reason": (
                str(failure) if failure is not None else "candidate_safety_failure"
            ),
        }
    return {
        "candidate_id": cell["cell_id"],
        "rho_conv": cell["rho_conv"],
        "rho_dense": cell["rho_dense"],
        "admissible": True,
        "completed_steps": result["completed_steps"],
        "final_validation_loss": result["final_validation_loss"],
        "final_validation_accuracy": result["final_validation_accuracy"],
        "median_projection_efficiency": result[
            "median_projection_efficiency"
        ],
    }


def _selection_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": value["status"],
        "reason": value["reason"],
        "minimum_final_validation_loss": value[
            "minimum_final_validation_loss"
        ],
        "diagnostic_best": value["diagnostic_best"],
        "plateau": value["plateau"],
        "expansion": asdict(value["expansion"]),
        "evaluated_candidates": value["evaluated_candidates"],
    }


def status(root: str | Path) -> dict[str, Any]:
    destination, manifest, _config = validate_root(root)
    gate_complete = 0
    surface_complete = 0
    errors: list[str] = []
    for entry in manifest["gate_entries"]:
        path = (
            destination
            / "studies"
            / entry["study_id"]
            / "rows"
            / entry["row_id"]
            / "gate_completion.json"
        )
        if not path.is_file():
            continue
        try:
            _validate_completion(
                path,
                destination,
                schema_version=GATE_COMPLETION_SCHEMA_VERSION,
                identity={
                    "entry_id": entry["entry_id"],
                    "study_id": entry["study_id"],
                    "row_id": entry["row_id"],
                },
            )
            gate_complete += 1
        except Exception as exc:  # status must report every invalid entry
            errors.append(str(exc))
    for entry in manifest["surface_entries"]:
        path = (
            destination
            / "studies"
            / entry["study_id"]
            / "rows"
            / entry["row_id"]
            / "surface_completions"
            / f"{entry['surface_id']}.json"
        )
        if not path.is_file():
            continue
        try:
            _validate_surface_completion(
                path,
                destination,
                identity={
                    "entry_id": entry["entry_id"],
                    "study_id": entry["study_id"],
                    "surface_id": entry["surface_id"],
                },
            )
            surface_complete += 1
        except Exception as exc:
            errors.append(str(exc))
    state = (
        "invalid"
        if errors
        else "complete"
        if gate_complete == 12 and surface_complete == 22
        else "in_progress"
        if gate_complete or surface_complete
        else "not_started"
    )
    return {
        "schema_version": (
            "mnist-conv-perfectdiode-initialization-rho-status/v1"
        ),
        "comparison_id": manifest["comparison_id"],
        "state": state,
        "gate_entries": {"complete": gate_complete, "expected": 12},
        "surface_entries": {"complete": surface_complete, "expected": 22},
        "errors": errors,
        "official_test_read": False,
    }


def collect(root: str | Path) -> dict[str, Any]:
    destination, manifest, config = validate_root(root)
    observed_status = status(destination)
    if observed_status["state"] != "complete":
        raise _error("execution status", "'complete'", observed_status)
    initialization_audit: list[dict[str, Any]] = []
    for initializer_id in INITIALIZER_ORDER:
        for architecture in ("conv1", "conv2"):
            gates = [
                entry
                for entry in manifest["gate_entries"]
                if entry["initializer_id"] == initializer_id
                and entry["architecture"] == architecture
            ]
            hashes = []
            for gate in gates:
                completion_path = (
                    destination
                    / "studies"
                    / gate["study_id"]
                    / "rows"
                    / gate["row_id"]
                    / "gate_completion.json"
                )
                hashes.append(
                    read_json(completion_path)[
                        "initialization_tensor_sha256"
                    ]
                )
            if len(gates) != 3 or len(set(hashes)) != 1:
                raise _error(
                    f"initialization audit for {initializer_id}/{architecture}",
                    "three independently materialized scheme gates with one "
                    "content-equal initialization tensor SHA-256",
                    {"gate_count": len(gates), "tensor_sha256": hashes},
                )
            initialization_audit.append(
                {
                    "initializer_id": initializer_id,
                    "architecture": architecture,
                    "scheme_gate_count": len(gates),
                    "content_equal_across_schemes": True,
                    "initialization_tensor_sha256": hashes[0],
                }
            )

    surfaces: list[dict[str, Any]] = []
    for entry in manifest["surface_entries"]:
        row_root = (
            destination
            / "studies"
            / entry["study_id"]
            / "rows"
            / entry["row_id"]
        )
        candidates = [_candidate_record(row_root, cell) for cell in entry["cells"]]
        selected = select_surface_candidates(
            candidates,
            rho_conv_values=entry["rho_conv"],
            rho_dense_values=entry["rho_dense"],
            expansion_available=False,
        )
        surfaces.append(
            {
                "initializer_id": entry["initializer_id"],
                "study_id": entry["study_id"],
                "surface_id": entry["surface_id"],
                "row_id": entry["row_id"],
                "architecture": entry["architecture"],
                "scheme": entry["scheme"],
                "optimizer": entry["optimizer"],
                "rho_conv": entry["rho_conv"],
                "rho_dense": entry["rho_dense"],
                "selection": _selection_json(selected),
            }
        )
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for surface in surfaces:
        by_key.setdefault(
            (surface["row_id"], surface["optimizer"]), []
        ).append(surface)
    comparisons = []
    for key, arms in by_key.items():
        if [arm["initializer_id"] for arm in arms] != list(INITIALIZER_ORDER):
            raise _error(
                f"initializer coverage for {key!r}",
                repr(INITIALIZER_ORDER),
                [arm["initializer_id"] for arm in arms],
            )
        first, second = arms
        first_best = first["selection"]["diagnostic_best"]
        second_best = second["selection"]["diagnostic_best"]
        first_plateau = {
            (item["rho_conv"], item["rho_dense"])
            for item in first["selection"]["plateau"]
        }
        second_plateau = {
            (item["rho_conv"], item["rho_dense"])
            for item in second["selection"]["plateau"]
        }
        if first_best is None or second_best is None:
            concordance = False
            reason = "one_or_both_initializers_have_no_diagnostic_best"
        else:
            first_pair = (
                first_best["rho_conv"],
                first_best["rho_dense"],
            )
            second_pair = (
                second_best["rho_conv"],
                second_best["rho_dense"],
            )
            concordance = (
                first_pair in second_plateau and second_pair in first_plateau
            )
            reason = (
                "mutual_inclusive_2pct_plateau_retention"
                if concordance
                else "mutual_plateau_retention_failed"
            )
        comparisons.append(
            {
                "row_id": key[0],
                "optimizer": key[1],
                "architecture": first["architecture"],
                "scheme": first["scheme"],
                "bounded_uniform_best": first_best,
                "bounded_kaiming_uniform_best": second_best,
                "common_plateau_pairs": [
                    {"rho_conv": pair[0], "rho_dense": pair[1]}
                    for pair in sorted(first_plateau & second_plateau)
                ],
                "initializer_concordance": concordance,
                "reason": reason,
            }
        )
    overall = all(item["initializer_concordance"] for item in comparisons)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "comparison_id": manifest["comparison_id"],
        "experiment_id": manifest["experiment_id"],
        "config_sha256": manifest["config_sha256"],
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": sha256_file(destination / "manifest.json"),
        "parent_study_id": PARENT_STUDY_ID,
        "reuse_parent_numerical_outputs": False,
        "decision_rule": config["decision"],
        "initialization_audit": initialization_audit,
        "surface_results": surfaces,
        "initializer_comparisons": comparisons,
        "overall_initializer_concordance": overall,
        "official_test_read": False,
    }
    summary_path = destination / "summary.json"
    atomic_write_json(summary_path, summary, canonical=True)
    completion = {
        "schema_version": COMPARISON_COMPLETION_SCHEMA_VERSION,
        "comparison_id": manifest["comparison_id"],
        "experiment_id": manifest["experiment_id"],
        "status": "complete",
        "gate_entries_complete": 12,
        "surface_entries_complete": 22,
        "rho_cells_complete": 148,
        "summary_sha256": sha256_file(summary_path),
        "official_test_read": False,
        "artifacts": _artifact_records(
            [destination / "manifest.json", summary_path], destination
        ),
    }
    atomic_write_json(destination / "completion.json", completion, canonical=True)
    return summary


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_CONFIG",
    "InitRhoComparisonError",
    "build_manifest",
    "collect",
    "load_config",
    "materialize",
    "run_gate",
    "run_surface",
    "status",
    "validate_root",
]
