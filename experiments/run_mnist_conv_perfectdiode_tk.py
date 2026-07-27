#!/usr/bin/env python3
"""Plan and execute the immutable Conv3 perfect-diode T/K diagnostic."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.mnist_conv.identity import (  # noqa: E402
    code_provenance,
    sha256_file,
)
from experiments.mnist_conv.io import (  # noqa: E402
    atomic_write_json,
    read_json,
)
from experiments.mnist_conv.perfectdiode_tk_runtime import (  # noqa: E402
    execute_tk_row,
    load_shared_assets,
    prepare_shared_assets,
    run_tk_smoke,
    validate_shared_assets,
)
from experiments.mnist_conv.perfectdiode_tk_spec import (  # noqa: E402
    SCHEME_ORDER,
    TK_ENTRY_COMPLETION_SCHEMA_VERSION,
    TK_MANIFEST_SCHEMA_VERSION,
    TK_SELECTION_SCHEMA_VERSION,
    PerfectDiodeTKStudySpec,
    tk_entry_id,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "conv"
    / "perfectdiode_conv3_tk_ordinary_mnist_v1.json"
)
LAUNCHER_PATH = Path(__file__).resolve()
LANE_BY_SCHEME = {
    "baseline": "main",
    "ours": "akibscomputer",
    "legacy": "main",
}


class PerfectDiodeTKOrchestrationError(RuntimeError):
    """Raised when an immutable T/K orchestration artifact fails validation."""


def _publish_identical(path: Path, value: Mapping[str, Any]) -> Path:
    if path.exists():
        observed = read_json(path)
        if observed != dict(value):
            raise PerfectDiodeTKOrchestrationError(
                "Expected the immutable artifact to contain identical content. "
                f"Provided value: {path}."
            )
        return path
    return atomic_write_json(path, dict(value), canonical=True)


def _study_root(results_root: str | Path, spec: PerfectDiodeTKStudySpec) -> Path:
    name = str(spec.data["name"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise ValueError(
            "Expected study.name to contain only lowercase letters, digits, and "
            f"hyphens. Provided value: {name!r}."
        )
    return (
        Path(results_root).expanduser().resolve()
        / "perfectdiode_tk_studies"
        / f"{name}--{spec.study_id}"
    )


def _asset_binding(assets: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": str(assets["schema_version"]),
        "study_id": str(assets["study_id"]),
        "config_sha256": str(assets["config_sha256"]),
        "assets_sha256": str(assets["assets_sha256"]),
        "completion_sha256": str(assets["completion_sha256"]),
        "initialization_checkpoint_sha256": str(
            assets["initialization"]["sha256"]
        ),
        "initialization_tensor_sha256": str(
            assets["initialization"]["parameter_tensor_sha256"]
        ),
        "train_indices_sha256": str(assets["split"]["train_indices_sha256"]),
        "validation_indices_sha256": str(
            assets["split"]["validation_indices_sha256"]
        ),
        "t_cohort_indices_sha256": str(
            assets["cohorts"]["t"]["source_indices_sha256"]
        ),
        "k_cohort_indices_sha256": str(
            assets["cohorts"]["k"]["source_indices_sha256"]
        ),
    }


def plan_study(
    spec: PerfectDiodeTKStudySpec,
    *,
    results_root: str | Path,
    asset_dir: str | Path,
    source_commit: str,
    source_archive_sha256: str,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish the resolved config and three-row immutable manifest."""

    assets = validate_shared_assets(spec, asset_dir)
    root = _study_root(results_root, spec)
    root.mkdir(parents=True, exist_ok=True)
    resolved_path = root / "resolved_config.json"
    _publish_identical(resolved_path, spec.data)
    resolved_file_sha256 = sha256_file(resolved_path)
    source = copy.deepcopy(dict(provenance or code_provenance()))
    if not isinstance(source.get("effective_code_fingerprint"), str):
        raise PerfectDiodeTKOrchestrationError(
            "Expected code provenance to include an effective code fingerprint."
        )
    if (
        not isinstance(source_commit, str)
        or len(source_commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError(
            "Expected source_commit to be a lowercase 40- or 64-character commit "
            f"id. Provided value: {source_commit!r}."
        )
    if (
        not isinstance(source_archive_sha256, str)
        or len(source_archive_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_archive_sha256
        )
    ):
        raise ValueError(
            "Expected source_archive_sha256 to be a lowercase SHA-256 digest. "
            f"Provided value: {source_archive_sha256!r}."
        )
    if (
        source.get("git_revision") != source_commit
        or source.get("dirty_source_digest") is not None
    ):
        raise PerfectDiodeTKOrchestrationError(
            "Expected planning from a clean checkout at the exact staged source "
            f"commit. Provided value: commit={source_commit!r}, "
            f"provenance={source!r}."
        )
    entries = []
    for index, row in enumerate(spec.rows):
        entry_id = tk_entry_id(spec.study_id, row)
        entry_dir = f"entries/{entry_id}"
        entries.append(
            {
                "entry_index": index,
                "entry_id": entry_id,
                "row_id": row["row_id"],
                "scheme": row["scheme"],
                "architecture": "conv3",
                "execution": {
                    "backend": "tmux",
                    "lane": LANE_BY_SCHEME[row["scheme"]],
                    "whole_row_on_one_lane": True,
                },
                "output_dir": entry_dir,
                "completion_path": f"{entry_dir}/completion.json",
                "outputs": [
                    f"{entry_dir}/environment.json",
                    f"{entry_dir}/t_measurements.json",
                    f"{entry_dir}/k_measurements.json",
                    f"{entry_dir}/operating_point_audit.json",
                    f"{entry_dir}/result.json",
                ],
            }
        )
    manifest = {
        "schema_version": TK_MANIFEST_SCHEMA_VERSION,
        "study_id": spec.study_id,
        "config_path": "resolved_config.json",
        "config_sha256": spec.config_sha256,
        "resolved_config_file_sha256": resolved_file_sha256,
        "launcher_sha256": sha256_file(LAUNCHER_PATH),
        "code_provenance": source,
        "staged_source": {
            "commit": source_commit,
            "archive_sha256": source_archive_sha256,
            "effective_code_fingerprint": source[
                "effective_code_fingerprint"
            ],
        },
        "asset_binding": _asset_binding(assets),
        "entry_count": len(entries),
        "entries": entries,
        "selection_path": "selection.json",
    }
    manifest_path = root / "manifest.json"
    _publish_identical(manifest_path, manifest)
    return {
        "status": "planned",
        "study_id": spec.study_id,
        "study_root": str(root),
        "resolved_config": str(resolved_path),
        "resolved_config_file_sha256": resolved_file_sha256,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "asset_dir": str(Path(asset_dir).expanduser().resolve()),
        "asset_binding": manifest["asset_binding"],
        "staged_source": manifest["staged_source"],
        "entry_count": len(entries),
        "entries": entries,
        "launched_jobs": 0,
        "official_test_read": False,
    }


def _load_manifest(
    manifest_path: str | Path,
    *,
    enforce_code_identity: bool,
) -> tuple[Path, dict[str, Any], PerfectDiodeTKStudySpec]:
    path = Path(manifest_path).expanduser().resolve()
    root = path.parent
    manifest = read_json(path)
    if manifest.get("schema_version") != TK_MANIFEST_SCHEMA_VERSION:
        raise PerfectDiodeTKOrchestrationError(
            "Expected a Conv3 perfect-diode T/K v1 manifest. "
            f"Provided value: {manifest.get('schema_version')!r}."
        )
    config_relative = Path(str(manifest.get("config_path")))
    if config_relative != Path("resolved_config.json"):
        raise PerfectDiodeTKOrchestrationError(
            "Expected manifest.config_path to be exactly 'resolved_config.json'. "
            f"Provided value: {manifest.get('config_path')!r}."
        )
    config_path = root / config_relative
    if (
        not config_path.is_file()
        or sha256_file(config_path) != manifest.get("resolved_config_file_sha256")
    ):
        raise PerfectDiodeTKOrchestrationError(
            "Expected the resolved config file SHA-256 to verify."
        )
    spec = PerfectDiodeTKStudySpec.from_path(config_path)
    if (
        spec.study_id != manifest.get("study_id")
        or spec.config_sha256 != manifest.get("config_sha256")
    ):
        raise PerfectDiodeTKOrchestrationError(
            "Expected the manifest and resolved config identities to match."
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 3:
        raise PerfectDiodeTKOrchestrationError(
            f"Expected exactly three complete-row manifest entries. "
            f"Provided value: {entries!r}."
        )
    for index, (entry, row, scheme) in enumerate(
        zip(entries, spec.rows, SCHEME_ORDER)
    ):
        expected_id = tk_entry_id(spec.study_id, row)
        if (
            entry.get("entry_index") != index
            or entry.get("entry_id") != expected_id
            or entry.get("row_id") != row["row_id"]
            or entry.get("scheme") != scheme
            or entry.get("execution", {}).get("lane") != LANE_BY_SCHEME[scheme]
            or entry.get("execution", {}).get("whole_row_on_one_lane") is not True
        ):
            raise PerfectDiodeTKOrchestrationError(
                f"Expected manifest entry {index} to match the frozen whole-row "
                f"route. Provided value: {entry!r}."
            )
    if manifest.get("launcher_sha256") != sha256_file(LAUNCHER_PATH):
        raise PerfectDiodeTKOrchestrationError(
            "Expected the launcher SHA-256 to match the planned source."
        )
    if enforce_code_identity:
        staged = manifest.get("staged_source")
        if not isinstance(staged, Mapping):
            raise PerfectDiodeTKOrchestrationError(
                "Expected the manifest to bind an exact staged source archive."
            )
        current = code_provenance()
        if (
            current.get("git_revision") != staged.get("commit")
            or current.get("dirty_source_digest") is not None
            or current.get("effective_code_fingerprint")
            != staged.get("effective_code_fingerprint")
        ):
            raise PerfectDiodeTKOrchestrationError(
                "Expected a clean checkout at the exact staged commit and "
                "effective source fingerprint. Provided value: staged="
                f"{dict(staged)!r}, "
                f"current={current!r}."
            )
    return path, manifest, spec


def _validate_asset_binding(
    spec: PerfectDiodeTKStudySpec,
    manifest: Mapping[str, Any],
    asset_dir: str | Path,
) -> dict[str, Any]:
    assets = validate_shared_assets(spec, asset_dir)
    observed = _asset_binding(assets)
    if observed != manifest.get("asset_binding"):
        raise PerfectDiodeTKOrchestrationError(
            "Expected the imported asset bundle hashes to match the immutable "
            f"manifest. Provided value: {observed!r}."
        )
    return assets


def _entry_by_index(manifest: Mapping[str, Any], index: int) -> dict[str, Any]:
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(
            f"Expected entry-index to be an integer in [0, 2]. Provided value: {index!r}."
        )
    entries = manifest["entries"]
    if not 0 <= index < len(entries):
        raise ValueError(
            f"Expected entry-index to be an integer in [0, 2]. Provided value: {index!r}."
        )
    return copy.deepcopy(entries[index])


def _validate_entry_completion(
    root: Path,
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    completion_path = root / str(entry["completion_path"])
    if not completion_path.is_file():
        raise FileNotFoundError(completion_path)
    completion = read_json(completion_path)
    if (
        completion.get("schema_version")
        != TK_ENTRY_COMPLETION_SCHEMA_VERSION
        or completion.get("state") != "complete"
        or completion.get("study_id") != manifest["study_id"]
        or completion.get("entry_id") != entry["entry_id"]
        or completion.get("manifest_sha256") != manifest_sha256
    ):
        raise PerfectDiodeTKOrchestrationError(
            f"Expected a matching terminal completion marker: {completion_path}."
        )
    records = completion.get("outputs")
    expected_paths = list(entry["outputs"])
    if (
        not isinstance(records, list)
        or [record.get("path") for record in records] != expected_paths
    ):
        raise PerfectDiodeTKOrchestrationError(
            f"Expected completion outputs in manifest order: {expected_paths!r}."
        )
    for record in records:
        path = root / str(record["path"])
        if (
            not path.is_file()
            or sha256_file(path) != record.get("sha256")
            or path.stat().st_size != record.get("bytes")
        ):
            raise PerfectDiodeTKOrchestrationError(
                f"Expected completed output hash and size to verify: {path}."
            )
    result_path = root / expected_paths[-1]
    result = read_json(result_path)
    if (
        result.get("study_id") != manifest["study_id"]
        or result.get("config_sha256") != manifest["config_sha256"]
        or result.get("manifest_sha256") != manifest_sha256
        or result.get("entry_id") != entry["entry_id"]
        or result.get("row_id") != entry["row_id"]
        or result.get("scheme") != entry["scheme"]
        or result.get("architecture") != "conv3"
        or result.get("execution_host") != entry["execution"]["lane"]
        or result.get("execution_backend") != "tmux"
        or result.get("official_test_read") is not False
    ):
        raise PerfectDiodeTKOrchestrationError(
            f"Expected completed row result identity to verify: {result_path}."
        )
    audit_path = root / expected_paths[-2]
    audit = read_json(audit_path)
    if (
        result.get("operating_point_audit_path") != audit_path.name
        or result.get("operating_point_audit_sha256") != sha256_file(audit_path)
        or result.get("operating_point_audit_passed")
        is not (audit.get("passed") is True)
        or audit.get("study_id") != manifest["study_id"]
        or audit.get("entry_id") != entry["entry_id"]
        or audit.get("row_id") != entry["row_id"]
        or audit.get("execution_host") != entry["execution"]["lane"]
        or audit.get("fresh_replay_after_selection")
        is not (result.get("diagnostic_selection_status") == "selected")
        or audit.get("official_test_read") is not False
    ):
        raise PerfectDiodeTKOrchestrationError(
            f"Expected the independent operating-point audit to verify: {audit_path}."
        )
    if result.get("status") == "selected" and (
        audit.get("passed") is not True
        or result.get("selected_t") != audit.get("selected_t")
        or result.get("selected_k") != audit.get("selected_k")
    ):
        raise PerfectDiodeTKOrchestrationError(
            "Expected every selected row to have a passing, row-matched "
            "operating-point audit."
        )
    environment_path = root / expected_paths[0]
    environment = read_json(environment_path)
    if (
        result.get("execution_environment_sha256")
        != sha256_file(environment_path)
        or audit.get("execution_environment_sha256")
        != sha256_file(environment_path)
        or environment.get("execution_host") != entry["execution"]["lane"]
        or environment.get("execution_backend") != "tmux"
    ):
        raise PerfectDiodeTKOrchestrationError(
            f"Expected the row execution environment to verify: {environment_path}."
        )
    return completion, result


def _output_records(root: Path, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for relative in entry["outputs"]:
        path = root / str(relative)
        if not path.is_file():
            raise PerfectDiodeTKOrchestrationError(
                f"Expected row executor to publish declared output: {path}."
            )
        records.append(
            {
                "path": str(relative),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def execution_environment_fingerprint(
    *,
    device: str,
    execution_host: str,
) -> dict[str, Any]:
    """Return the immutable row-host Python/Torch/CUDA descriptor."""

    import torch

    requested = torch.device(device)
    cuda = requested.type == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError(
            f"Expected requested CUDA device to be available. Provided value: {device!r}."
        )
    index = (
        requested.index
        if requested.index is not None
        else torch.cuda.current_device()
        if cuda
        else None
    )
    properties = torch.cuda.get_device_properties(index) if cuda else None
    return {
        "schema_version": "mnist-conv-perfectdiode-tk-environment/v1",
        "execution_backend": "tmux",
        "execution_host": execution_host,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": str(Path(sys.executable).resolve()),
        "torch_version": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "requested_device": str(requested),
        "device_index": index,
        "device_name": properties.name if properties is not None else None,
        "device_capability": (
            list(torch.cuda.get_device_capability(index)) if cuda else None
        ),
        "device_total_memory_bytes": (
            int(properties.total_memory) if properties is not None else None
        ),
    }


def _publish_selection_if_complete(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> dict[str, Any] | None:
    lock_path = root / ".selection.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        completed: list[tuple[Mapping[str, Any], dict[str, Any], dict[str, Any]]] = []
        for entry in manifest["entries"]:
            try:
                completion, result = _validate_entry_completion(
                    root,
                    manifest,
                    entry,
                    manifest_sha256=manifest_sha256,
                )
            except FileNotFoundError:
                return None
            completed.append((entry, completion, result))
        rows = []
        row_hashes: dict[str, dict[str, str]] = {}
        for entry, _completion, result in completed:
            result_path = root / str(entry["outputs"][-1])
            completion_path = root / str(entry["completion_path"])
            result_hash = sha256_file(result_path)
            completion_hash = sha256_file(completion_path)
            rows.append(
                {
                    "row_id": result["row_id"],
                    "architecture": "conv3",
                    "scheme": result["scheme"],
                    "execution_backend": result["execution_backend"],
                    "execution_host": result["execution_host"],
                    "execution_environment_path": str(entry["outputs"][0]),
                    "execution_environment_sha256": result[
                        "execution_environment_sha256"
                    ],
                    "run_name": result["run_name"],
                    "voltage_amp": result["voltage_amp"],
                    "current_amp": result["current_amp"],
                    "input_gain": result["input_gain"],
                    "status": result["status"],
                    "selected_t": result["selected_t"],
                    "selected_k": result["selected_k"],
                    "t_reference": result["t_reference"],
                    "k_reference": result["k_reference"],
                    "t_extension_used": result["t_extension_used"],
                    "k128_sentinel_used": result["k128_sentinel_used"],
                    "result_path": str(entry["outputs"][-1]),
                    "result_sha256": result_hash,
                    "completion_path": str(entry["completion_path"]),
                    "completion_sha256": completion_hash,
                    "operating_point_audit_path": str(entry["outputs"][-2]),
                    "operating_point_audit_sha256": (
                        result["operating_point_audit_sha256"]
                    ),
                    "operating_point_audit_passed": (
                        result["operating_point_audit_passed"]
                    ),
                    "t_cohort_indices_sha256": result[
                        "t_cohort_indices_sha256"
                    ],
                    "k_cohort_indices_sha256": result[
                        "k_cohort_indices_sha256"
                    ],
                }
            )
            row_hashes[result["row_id"]] = {
                "result_sha256": result_hash,
                "completion_sha256": completion_hash,
                "operating_point_audit_sha256": result[
                    "operating_point_audit_sha256"
                ],
                "execution_environment_sha256": result[
                    "execution_environment_sha256"
                ],
            }
        selection = {
            "schema_version": TK_SELECTION_SCHEMA_VERSION,
            "study_id": manifest["study_id"],
            "config_sha256": manifest["config_sha256"],
            "manifest_sha256": manifest_sha256,
            "official_test_read": False,
            "status": (
                "selected"
                if all(row["status"] == "selected" for row in rows)
                else "unresolved"
            ),
            "rows": rows,
            "artifact_sha256s": {
                "asset_bundle": copy.deepcopy(manifest["asset_binding"]),
                "rows": row_hashes,
            },
        }
        selection_path = root / str(manifest["selection_path"])
        _publish_identical(selection_path, selection)
        return selection


def execute_manifest_entry(
    manifest_path: str | Path,
    *,
    entry_index: int,
    asset_dir: str | Path,
    data_root: str | Path,
    device: str,
    download: bool = False,
    row_executor: Callable[..., Mapping[str, Any]] = execute_tk_row,
    enforce_code_identity: bool = True,
) -> dict[str, Any]:
    """Run or resume one whole-scheme T/K manifest entry."""

    path, manifest, spec = _load_manifest(
        manifest_path, enforce_code_identity=enforce_code_identity
    )
    root = path.parent
    _validate_asset_binding(spec, manifest, asset_dir)
    entry = _entry_by_index(manifest, entry_index)
    manifest_sha256 = sha256_file(path)
    completion_path = root / str(entry["completion_path"])
    if completion_path.exists():
        completion, result = _validate_entry_completion(
            root,
            manifest,
            entry,
            manifest_sha256=manifest_sha256,
        )
        return {
            "status": "already_complete",
            "entry": entry,
            "completion": completion,
            "result": result,
            "selection": _publish_selection_if_complete(
                root, manifest, manifest_sha256=manifest_sha256
            ),
        }
    entry_dir = root / str(entry["output_dir"])
    if entry_dir.exists() and any(entry_dir.iterdir()):
        raise PerfectDiodeTKOrchestrationError(
            "Expected an incomplete entry directory to be empty before retry. "
            f"Provided value: {entry_dir}."
        )
    entry_dir.mkdir(parents=True, exist_ok=True)
    environment = execution_environment_fingerprint(
        device=device,
        execution_host=entry["execution"]["lane"],
    )
    environment_path = entry_dir / "environment.json"
    atomic_write_json(environment_path, environment, canonical=True)
    environment_sha256 = sha256_file(environment_path)
    assets = load_shared_assets(
        spec,
        asset_dir=asset_dir,
        data_root=data_root,
        download=download,
    )
    row = next(row for row in spec.rows if row["row_id"] == entry["row_id"])
    result = dict(
        row_executor(
            spec,
            row=row,
            assets=assets,
            output_dir=entry_dir,
            device=device,
            manifest_sha256=manifest_sha256,
            entry_id=entry["entry_id"],
            execution_host=entry["execution"]["lane"],
            execution_environment_sha256=environment_sha256,
        )
    )
    records = _output_records(root, entry)
    completion = {
        "schema_version": TK_ENTRY_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "study_id": spec.study_id,
        "entry_id": entry["entry_id"],
        "manifest_sha256": manifest_sha256,
        "official_test_read": False,
        "outputs": records,
    }
    atomic_write_json(completion_path, completion, canonical=True)
    _validate_entry_completion(
        root, manifest, entry, manifest_sha256=manifest_sha256
    )
    return {
        "status": "complete",
        "entry": entry,
        "completion": completion,
        "result": result,
        "selection": _publish_selection_if_complete(
            root, manifest, manifest_sha256=manifest_sha256
        ),
    }


def study_status(manifest_path: str | Path) -> dict[str, Any]:
    """Return a read-only, hash-validated study status."""

    path, manifest, spec = _load_manifest(
        manifest_path, enforce_code_identity=False
    )
    root = path.parent
    manifest_sha256 = sha256_file(path)
    rows = []
    for entry in manifest["entries"]:
        completion_path = root / str(entry["completion_path"])
        if not completion_path.exists():
            rows.append(
                {
                    "entry_index": entry["entry_index"],
                    "row_id": entry["row_id"],
                    "scheme": entry["scheme"],
                    "lane": entry["execution"]["lane"],
                    "state": "pending",
                    "status": None,
                    "selected_t": None,
                    "selected_k": None,
                }
            )
            continue
        _completion, result = _validate_entry_completion(
            root,
            manifest,
            entry,
            manifest_sha256=manifest_sha256,
        )
        rows.append(
            {
                "entry_index": entry["entry_index"],
                "row_id": entry["row_id"],
                "scheme": entry["scheme"],
                "lane": entry["execution"]["lane"],
                "state": "complete",
                "status": result["status"],
                "selected_t": result["selected_t"],
                "selected_k": result["selected_k"],
            }
        )
    selection_path = root / str(manifest["selection_path"])
    selection = read_json(selection_path) if selection_path.is_file() else None
    if selection is not None and (
        selection.get("schema_version") != TK_SELECTION_SCHEMA_VERSION
        or selection.get("study_id") != spec.study_id
        or selection.get("manifest_sha256") != manifest_sha256
    ):
        raise PerfectDiodeTKOrchestrationError(
            "Expected the aggregate selection identity to verify."
        )
    return {
        "study_id": spec.study_id,
        "manifest": str(path),
        "manifest_sha256": manifest_sha256,
        "completed_entries": sum(row["state"] == "complete" for row in rows),
        "entry_count": len(rows),
        "rows": rows,
        "selection": selection,
    }


def finalize_study(manifest_path: str | Path) -> dict[str, Any]:
    """Validate all copied row bundles and publish the aggregate selection."""

    path, manifest, spec = _load_manifest(
        manifest_path, enforce_code_identity=False
    )
    manifest_sha256 = sha256_file(path)
    selection = _publish_selection_if_complete(
        path.parent,
        manifest,
        manifest_sha256=manifest_sha256,
    )
    if selection is None:
        raise PerfectDiodeTKOrchestrationError(
            "Expected all three copied row completions before finalization. "
            f"Provided value: {path.parent}."
        )
    return {
        "status": selection["status"],
        "study_id": spec.study_id,
        "manifest_sha256": manifest_sha256,
        "selection": str(path.parent / str(manifest["selection_path"])),
        "selection_sha256": sha256_file(
            path.parent / str(manifest["selection_path"])
        ),
        "rows": selection["rows"],
        "official_test_read": False,
    }


def run_preflight_smoke(
    manifest_path: str | Path,
    *,
    entry_index: int,
    asset_dir: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    device: str,
    download: bool,
) -> dict[str, Any]:
    """Execute the real one-T-batch/one-K4-batch numerical smoke path."""

    path, manifest, spec = _load_manifest(
        manifest_path, enforce_code_identity=True
    )
    _validate_asset_binding(spec, manifest, asset_dir)
    entry = _entry_by_index(manifest, entry_index)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise PerfectDiodeTKOrchestrationError(
            "Expected the disposable smoke output directory to be empty. "
            f"Provided value: {destination}."
        )
    destination.mkdir(parents=True, exist_ok=True)
    environment = execution_environment_fingerprint(
        device=device,
        execution_host=entry["execution"]["lane"],
    )
    environment_path = destination / "environment.json"
    atomic_write_json(environment_path, environment, canonical=True)
    assets = load_shared_assets(
        spec,
        asset_dir=asset_dir,
        data_root=data_root,
        download=download,
    )
    row = next(row for row in spec.rows if row["row_id"] == entry["row_id"])
    result = run_tk_smoke(
        spec,
        row=row,
        assets=assets,
        output_dir=destination,
        device=device,
        manifest_sha256=sha256_file(path),
        entry_id=entry["entry_id"],
        execution_host=entry["execution"]["lane"],
        execution_environment_sha256=sha256_file(environment_path),
    )
    completion_path = destination / "completion.json"
    if not completion_path.is_file():
        raise PerfectDiodeTKOrchestrationError(
            "Expected the smoke completion receipt to be published last."
        )
    return {
        "status": "passed",
        "study_id": spec.study_id,
        "entry_id": entry["entry_id"],
        "output_dir": str(destination),
        "completion": str(completion_path),
        "completion_sha256": sha256_file(completion_path),
        "result": result,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create shared assets, plan, execute, and inspect the Conv3 "
            "perfect-diode ordinary-MNIST T/K study."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    assets = subparsers.add_parser("assets")
    assets.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    assets.add_argument("--assets-root", type=Path, required=True)
    assets.add_argument("--data-root", type=Path, required=True)
    assets.add_argument("--device", default="cuda")
    assets.add_argument("--download", action="store_true")

    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    plan.add_argument("--results-root", type=Path, required=True)
    plan.add_argument("--asset-dir", type=Path, required=True)
    plan.add_argument("--source-commit", required=True)
    plan.add_argument("--source-archive-sha256", required=True)

    entry = subparsers.add_parser("run-entry")
    entry.add_argument("--manifest", type=Path, required=True)
    entry.add_argument("--entry-index", type=int, required=True)
    entry.add_argument("--asset-dir", type=Path, required=True)
    entry.add_argument("--data-root", type=Path, required=True)
    entry.add_argument("--device", default="cuda")
    entry.add_argument("--download", action="store_true")

    status = subparsers.add_parser("status")
    status.add_argument("--manifest", type=Path, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--manifest", type=Path, required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--manifest", type=Path, required=True)
    smoke.add_argument("--entry-index", type=int, default=0)
    smoke.add_argument("--asset-dir", type=Path, required=True)
    smoke.add_argument("--data-root", type=Path, required=True)
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--device", default="cuda")
    smoke.add_argument("--download", action="store_true")
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "assets":
        spec = PerfectDiodeTKStudySpec.from_path(args.config)
        result = prepare_shared_assets(
            spec,
            assets_root=args.assets_root,
            data_root=args.data_root,
            device=args.device,
            download=args.download,
        )
        _print(
            {
                "status": "complete",
                "study_id": spec.study_id,
                "asset_dir": result["asset_dir"],
                "assets_sha256": result["assets_sha256"],
                "completion_sha256": result["completion_sha256"],
                "official_test_read": False,
            }
        )
        return 0
    if args.command == "plan":
        _print(
            plan_study(
                PerfectDiodeTKStudySpec.from_path(args.config),
                results_root=args.results_root,
                asset_dir=args.asset_dir,
                source_commit=args.source_commit,
                source_archive_sha256=args.source_archive_sha256,
            )
        )
        return 0
    if args.command == "run-entry":
        _print(
            execute_manifest_entry(
                args.manifest,
                entry_index=args.entry_index,
                asset_dir=args.asset_dir,
                data_root=args.data_root,
                device=args.device,
                download=args.download,
            )
        )
        return 0
    if args.command == "status":
        _print(study_status(args.manifest))
        return 0
    if args.command == "finalize":
        _print(finalize_study(args.manifest))
        return 0
    if args.command == "smoke":
        _print(
            run_preflight_smoke(
                args.manifest,
                entry_index=args.entry_index,
                asset_dir=args.asset_dir,
                data_root=args.data_root,
                output_dir=args.output_dir,
                device=args.device,
                download=args.download,
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
