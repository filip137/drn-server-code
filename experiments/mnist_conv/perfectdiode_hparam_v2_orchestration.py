"""Read-only collection and terminal publication for Conv3 LR-v2 shards.

This module deliberately has no job-launching capability.  It validates the
immutable six-surface manifest, validates every public stage recursively, and
publishes only terminal host/study markers or an already-terminal incoming
host shard.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .identity import sha256_file, sha256_json
from .io import atomic_write_json, read_json
from .perfectdiode_conv3_hparam_v2_spec import (
    ALLOWED_HOSTS,
    PUBLIC_STAGE_SEQUENCE,
    PerfectDiodeConv3HparamStudySpec,
    validate_surface_manifest,
)


STUDY_FILENAME = "study.resolved.json"
MANIFEST_FILENAME = "surface_manifest.json"

EXECUTION_STAGE_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-execution-stage/v2"
)
EXECUTION_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-execution-completion/v2"
)
CELL_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-cell-completion/v2"
)
SURFACE_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-surface-completion/v2"
)
HOST_TERMINAL_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-host-terminal/v2"
)
IMPORT_RECEIPT_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-host-import/v2"
)
STUDY_RESULT_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-study-result/v2"
)
STUDY_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-study-completion/v2"
)
STATUS_SCHEMA_VERSION = "mnist-conv-perfectdiode-hparam-status/v2"
PLAN_SCHEMA_VERSION = "mnist-conv-perfectdiode-hparam-collector-plan/v2"


class PerfectDiodeConv3OrchestrationError(RuntimeError):
    """Raised when a shard or study cannot be accepted safely."""


def _error(path: str, expected: str, provided: Any) -> PerfectDiodeConv3OrchestrationError:
    return PerfectDiodeConv3OrchestrationError(
        f"Expected {path} to be {expected}. Provided value: {provided!r}."
    )


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "a JSON object", value)
    return copy.deepcopy(dict(value))


def _relative(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(path, "a non-empty traversal-free relative path", value)
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise _error(path, "a non-empty traversal-free relative path", value)
    return candidate.as_posix()


def _require_fields(value: Mapping[str, Any], expected: Mapping[str, Any], path: str) -> None:
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise _error(f"{path}.{key}", f"exactly {wanted!r}", value.get(key))


def _require_official_test_false(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        if "official_test_read" in value and value["official_test_read"] is not False:
            raise _error(f"{path}.official_test_read", "exactly false", value["official_test_read"])
        for key, item in value.items():
            _require_official_test_false(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_official_test_false(item, f"{path}[{index}]")


def _regular_files(root: Path, *, exclude: Iterable[Path] = ()) -> list[Path]:
    if not root.is_dir():
        raise _error(str(root), "an existing directory", str(root))
    excluded = {item.resolve() for item in exclude}
    files: list[Path] = []
    for item in root.rglob("*"):
        if item.is_symlink():
            raise _error(str(item), "a non-symlink artifact", str(item))
        if item.is_file() and item.resolve() not in excluded:
            files.append(item)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _record(path: Path, *, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _records(root: Path, *, exclude: Iterable[Path] = ()) -> list[dict[str, Any]]:
    return [_record(path, base=root) for path in _regular_files(root, exclude=exclude)]


def _validate_records(
    root: Path,
    raw_records: Any,
    *,
    path: str,
    expected_files: Sequence[Path] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(raw_records, list):
        raise _error(path, "an array of artifact records", raw_records)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_records):
        value = _mapping(raw, f"{path}[{index}]")
        relative = _relative(value.get("path"), f"{path}[{index}].path")
        if relative in seen:
            raise _error(path, "unique artifact paths", relative)
        seen.add(relative)
        target = root / relative
        try:
            target.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise _error(f"{path}[{index}].path", "a path inside the artifact root", relative) from exc
        if target.is_symlink() or not target.is_file():
            raise _error(str(target), "a regular file", str(target))
        expected = {
            "path": relative,
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
        if value != expected:
            raise _error(f"{path}[{index}]", f"exactly {expected!r}", value)
        normalized.append(expected)
    if expected_files is not None:
        wanted = {
            item.relative_to(root).as_posix()
            for item in expected_files
        }
        if seen != wanted:
            raise _error(path, f"exact coverage of {sorted(wanted)!r}", sorted(seen))
    if [item["path"] for item in normalized] != sorted(seen):
        raise _error(path, "records sorted by path", [item["path"] for item in normalized])
    return normalized


@dataclass(frozen=True)
class _Authority:
    root: Path
    spec: PerfectDiodeConv3HparamStudySpec
    manifest: dict[str, Any]
    surfaces: dict[str, dict[str, Any]]

    @property
    def manifest_id(self) -> str:
        return str(self.manifest["manifest_id"])


def _load_authority(root: str | Path) -> _Authority:
    source = Path(root).expanduser().resolve()
    study_path = source / STUDY_FILENAME
    manifest_path = source / MANIFEST_FILENAME
    if not study_path.is_file() or not manifest_path.is_file():
        raise _error(
            str(source),
            f"a collector containing {STUDY_FILENAME!r} and {MANIFEST_FILENAME!r}",
            str(source),
        )
    spec = PerfectDiodeConv3HparamStudySpec.from_path(study_path)
    manifest = validate_surface_manifest(
        _mapping(read_json(manifest_path), "surface manifest"),
        spec=spec,
    )
    if manifest.get("public_stage_sequence") != list(PUBLIC_STAGE_SEQUENCE):
        raise _error(
            "surface manifest.public_stage_sequence",
            f"exactly {list(PUBLIC_STAGE_SEQUENCE)!r}",
            manifest.get("public_stage_sequence"),
        )
    surfaces: dict[str, dict[str, Any]] = {}
    for raw in manifest["surfaces"]:
        surface = _mapping(raw, "surface manifest surface")
        surface_id = str(surface.get("surface_id"))
        if surface_id in surfaces:
            raise _error("surface manifest.surfaces", "unique surface ids", surface_id)
        expected_output = f"surfaces/{surface_id}"
        _require_fields(
            surface,
            {
                "output_path": expected_output,
                "completion_path": f"{expected_output}/completion.json",
                "finalization_path": f"{expected_output}/stages/finalize_lr/result.json",
            },
            f"surface manifest surface {surface_id}",
        )
        surfaces[surface_id] = surface
    if len(surfaces) != 6:
        raise _error("surface manifest surfaces", "exactly six unique surfaces", sorted(surfaces))
    return _Authority(source, spec, manifest, surfaces)


def _surface_dirs(authority: _Authority) -> dict[str, Path]:
    root = authority.root / "surfaces"
    if not root.exists():
        return {}
    if root.is_symlink() or not root.is_dir():
        raise _error(str(root), "a regular directory", str(root))
    result: dict[str, Path] = {}
    for item in sorted(root.iterdir(), key=lambda value: value.name):
        if item.is_symlink() or not item.is_dir():
            raise _error(str(item), "a non-symlink surface directory", str(item))
        result[item.name] = item
    return result


def _validate_cell(
    authority: _Authority,
    surface: Mapping[str, Any],
    stage: str,
    cell_dir: Path,
) -> None:
    cell_id = cell_dir.name
    completion_path = cell_dir / "completion.json"
    result_path = cell_dir / "result.json"
    if not completion_path.is_file() or not result_path.is_file():
        raise _error(str(cell_dir), "a complete cell with result.json and completion.json", str(cell_dir))
    completion = _mapping(read_json(completion_path), f"{cell_id} completion")
    _require_fields(
        completion,
        {
            "schema_version": CELL_COMPLETION_SCHEMA_VERSION,
            "state": "complete",
            "manifest_id": authority.manifest_id,
            "surface_id": surface["surface_id"],
            "stage": stage,
            "cell_id": cell_id,
            "official_test_read": False,
        },
        f"{cell_id} completion",
    )
    expected_files = _regular_files(cell_dir, exclude=(completion_path,))
    _validate_records(
        cell_dir,
        completion.get("outputs"),
        path=f"{cell_id} completion.outputs",
        expected_files=expected_files,
    )
    result = _mapping(read_json(result_path), f"{cell_id} result")
    _require_fields(
        result,
        {
            "surface_id": surface["surface_id"],
            "cell_id": cell_id,
            "official_test_read": False,
        },
        f"{cell_id} result",
    )
    _require_official_test_false(result, f"{cell_id} result")


def _validate_stage(
    authority: _Authority,
    surface: Mapping[str, Any],
    stage: str,
    stage_dir: Path,
) -> dict[str, Any]:
    completion_path = stage_dir / "completion.json"
    result_path = stage_dir / "result.json"
    if not completion_path.is_file() or not result_path.is_file():
        raise _error(str(stage_dir), "a complete public stage", str(stage_dir))
    completion = _mapping(read_json(completion_path), f"{stage} completion")
    _require_fields(
        completion,
        {
            "schema_version": EXECUTION_COMPLETION_SCHEMA_VERSION,
            "state": "complete",
            "study_id": authority.spec.study_id,
            "config_sha256": authority.spec.config_sha256,
            "manifest_id": authority.manifest_id,
            "surface_id": surface["surface_id"],
            "stage": stage,
            "official_test_read": False,
        },
        f"{surface['surface_id']} {stage} completion",
    )
    expected_files = _regular_files(stage_dir, exclude=(completion_path,))
    _validate_records(
        stage_dir,
        completion.get("outputs"),
        path=f"{surface['surface_id']} {stage} completion.outputs",
        expected_files=expected_files,
    )
    result = _mapping(read_json(result_path), f"{stage} result")
    _require_fields(
        result,
        {
            "schema_version": EXECUTION_STAGE_SCHEMA_VERSION,
            "study_id": authority.spec.study_id,
            "config_sha256": authority.spec.config_sha256,
            "manifest_id": authority.manifest_id,
            "surface_id": surface["surface_id"],
            "stage": stage,
            "official_test_read": False,
        },
        f"{surface['surface_id']} {stage} result",
    )
    _require_official_test_false(result, f"{surface['surface_id']} {stage} result")
    cells_root = stage_dir / "cells"
    if cells_root.exists():
        if cells_root.is_symlink() or not cells_root.is_dir():
            raise _error(str(cells_root), "a non-symlink directory", str(cells_root))
        for cell_dir in sorted(cells_root.iterdir(), key=lambda value: value.name):
            if cell_dir.is_symlink() or not cell_dir.is_dir():
                raise _error(str(cell_dir), "a non-symlink cell directory", str(cell_dir))
            _validate_cell(authority, surface, stage, cell_dir)
    for json_path in _regular_files(stage_dir):
        if json_path.suffix == ".json":
            _require_official_test_false(read_json(json_path), str(json_path))
    return result


def _validate_surface(authority: _Authority, surface_id: str, surface_root: Path) -> dict[str, Any]:
    if surface_id not in authority.surfaces:
        raise _error("surface id", "one of the exact manifest surfaces", surface_id)
    surface = authority.surfaces[surface_id]
    allowed_root_entries = {"stages", "completion.json"}
    unexpected = {item.name for item in surface_root.iterdir()} - allowed_root_entries
    if unexpected:
        raise _error(str(surface_root), f"only {sorted(allowed_root_entries)!r}", sorted(unexpected))
    stages_root = surface_root / "stages"
    if not stages_root.is_dir() or stages_root.is_symlink():
        raise _error(str(stages_root), "a non-symlink stages directory", str(stages_root))
    stage_names = {
        item.name
        for item in stages_root.iterdir()
        if item.is_dir() and not item.is_symlink()
    }
    non_dirs = [item.name for item in stages_root.iterdir() if not item.is_dir() or item.is_symlink()]
    if non_dirs or stage_names != set(PUBLIC_STAGE_SEQUENCE):
        raise _error(
            str(stages_root),
            f"exactly the public stages {list(PUBLIC_STAGE_SEQUENCE)!r}",
            {"stage_names": sorted(stage_names), "other_entries": sorted(non_dirs)},
        )
    results = {
        stage: _validate_stage(authority, surface, stage, stages_root / stage)
        for stage in PUBLIC_STAGE_SEQUENCE
    }
    completion_path = surface_root / "completion.json"
    completion = _mapping(read_json(completion_path), f"{surface_id} completion")
    final = results["finalize_lr"]
    _require_fields(
        completion,
        {
            "schema_version": SURFACE_COMPLETION_SCHEMA_VERSION,
            "state": "complete",
            "study_id": authority.spec.study_id,
            "config_sha256": authority.spec.config_sha256,
            "manifest_id": authority.manifest_id,
            "surface_id": surface_id,
            "host": surface["host"],
            "status": final.get("status"),
            "zero_work": final.get("zero_work"),
            "official_test_read": False,
        },
        f"{surface_id} completion",
    )
    final_files = [
        stages_root / "finalize_lr" / "completion.json",
        stages_root / "finalize_lr" / "result.json",
    ]
    _validate_records(
        surface_root,
        completion.get("outputs"),
        path=f"{surface_id} completion.outputs",
        expected_files=final_files,
    )
    selected_status = authority.spec.data["selection"]["selected_status"]
    unresolved = set(authority.spec.data["selection"]["unresolved_statuses"])
    allowed_statuses = {selected_status, *unresolved}
    # The adaptive legacy surface can terminate before a 3x3 grid exists.
    # Execution propagates this exact status through finalize_lr; it predates
    # the two Conv3 additions to the resolved protocol's unresolved list.
    rho_policy = surface.get("rho_policy")
    if (
        isinstance(rho_policy, Mapping)
        and rho_policy.get("core_mode")
        == "adaptive_safe_center_factor_three_3x3"
    ):
        allowed_statuses.add("unresolved_no_safe_center")
    if surface.get("zero_work") is True:
        allowed_statuses.add("zero_work")
    if final.get("status") not in allowed_statuses:
        raise _error(
            f"{surface_id} finalize_lr.status",
            f"one of {sorted(allowed_statuses)!r}",
            final.get("status"),
        )
    if final.get("status") == selected_status:
        _require_fields(
            final,
            {
                "selected_t": surface["inference_iterations"],
                "selected_k": surface["training_iterations"],
                "zero_work": False,
            },
            f"{surface_id} selected final result",
        )
        selected_rho = final.get("selected_rho")
        if not isinstance(selected_rho, Mapping) or set(selected_rho) != {"rho_conv", "rho_dense"}:
            raise _error(
                f"{surface_id} selected_rho",
                "an object with exactly rho_conv and rho_dense",
                selected_rho,
            )
    return final


def _assigned_ids(authority: _Authority, host: str) -> list[str]:
    if host not in ALLOWED_HOSTS:
        raise _error("host", f"one of {ALLOWED_HOSTS!r}", host)
    return [
        surface_id
        for surface_id, surface in authority.surfaces.items()
        if surface["host"] == host
    ]


def _host_artifacts(authority: _Authority, host: str) -> list[dict[str, Any]]:
    files = [
        authority.root / STUDY_FILENAME,
        authority.root / MANIFEST_FILENAME,
    ]
    for surface_id in _assigned_ids(authority, host):
        files.extend(_regular_files(authority.root / "surfaces" / surface_id))
    files = sorted(files, key=lambda item: item.relative_to(authority.root).as_posix())
    return [_record(item, base=authority.root) for item in files]


def _host_terminal_payload(authority: _Authority, host: str) -> dict[str, Any]:
    artifacts = _host_artifacts(authority, host)
    return {
        "schema_version": HOST_TERMINAL_SCHEMA_VERSION,
        "state": "complete",
        "study_id": authority.spec.study_id,
        "config_sha256": authority.spec.config_sha256,
        "manifest_id": authority.manifest_id,
        "host": host,
        "surface_ids": _assigned_ids(authority, host),
        "surface_count": len(_assigned_ids(authority, host)),
        "official_test_read": False,
        "artifacts": artifacts,
        "artifact_tree_sha256": sha256_json(artifacts),
        "file_count": len(artifacts),
        "bytes": sum(item["bytes"] for item in artifacts),
        "launched_jobs": 0,
    }


def _validate_host_layout(authority: _Authority, host: str, *, require_complete: bool) -> dict[str, Any]:
    assigned = set(_assigned_ids(authority, host))
    present = _surface_dirs(authority)
    cross_host = sorted(set(present) - assigned)
    if cross_host:
        raise _error(
            str(authority.root / "surfaces"),
            f"only surfaces assigned to host {host!r}",
            cross_host,
        )
    surface_statuses: list[dict[str, Any]] = []
    for surface_id in _assigned_ids(authority, host):
        surface_root = authority.root / "surfaces" / surface_id
        if not surface_root.is_dir():
            surface_statuses.append(
                {"surface_id": surface_id, "state": "missing", "completed_stages": []}
            )
            continue
        stages_root = surface_root / "stages"
        completed: list[str] = []
        if stages_root.exists():
            if stages_root.is_symlink() or not stages_root.is_dir():
                raise _error(str(stages_root), "a non-symlink directory", str(stages_root))
            unknown = sorted(
                item.name for item in stages_root.iterdir()
                if item.name not in PUBLIC_STAGE_SEQUENCE
            )
            if unknown:
                raise _error(str(stages_root), "only public stage directories", unknown)
            for stage in PUBLIC_STAGE_SEQUENCE:
                stage_dir = stages_root / stage
                if not stage_dir.exists():
                    continue
                if stage_dir.is_symlink() or not stage_dir.is_dir():
                    raise _error(str(stage_dir), "a non-symlink stage directory", str(stage_dir))
                completion = stage_dir / "completion.json"
                if completion.exists():
                    _validate_stage(authority, authority.surfaces[surface_id], stage, stage_dir)
                    completed.append(stage)
                else:
                    cells = stage_dir / "cells"
                    if cells.is_dir():
                        for cell_dir in cells.iterdir():
                            if (cell_dir / "completion.json").exists():
                                _validate_cell(
                                    authority,
                                    authority.surfaces[surface_id],
                                    stage,
                                    cell_dir,
                                )
        if (surface_root / "completion.json").exists():
            _validate_surface(authority, surface_id, surface_root)
            state = "complete"
        else:
            state = "in_progress"
        surface_statuses.append(
            {"surface_id": surface_id, "state": state, "completed_stages": completed}
        )
    complete = (
        set(present) == assigned
        and all(item["state"] == "complete" for item in surface_statuses)
    )
    if require_complete and not complete:
        raise _error(
            f"host {host} shard",
            "every assigned surface complete",
            surface_statuses,
        )
    return {
        "host": host,
        "assigned_surface_ids": _assigned_ids(authority, host),
        "surfaces": surface_statuses,
        "complete": complete,
    }


def validate_host_terminal(root: str | Path, host: str) -> dict[str, Any]:
    """Validate an exact per-host terminal marker and all bound artifacts."""

    authority = _load_authority(root)
    _validate_host_layout(authority, host, require_complete=True)
    marker_path = authority.root / "host_terminals" / f"{host}.json"
    if not marker_path.is_file():
        raise _error(str(marker_path), "an existing host terminal marker", str(marker_path))
    marker = _mapping(read_json(marker_path), "host terminal")
    expected = _host_terminal_payload(authority, host)
    if marker != expected:
        raise _error("host terminal", f"exactly {expected!r}", marker)
    return marker


def finalize_host_shard(root: str | Path, host: str) -> dict[str, Any]:
    """Write a host terminal marker last after strict recursive validation."""

    authority = _load_authority(root)
    _validate_host_layout(authority, host, require_complete=True)
    payload = _host_terminal_payload(authority, host)
    target = authority.root / "host_terminals" / f"{host}.json"
    if target.exists():
        existing = _mapping(read_json(target), "host terminal")
        if existing != payload:
            raise _error("existing host terminal", f"exactly {payload!r}", existing)
    else:
        atomic_write_json(target, payload, canonical=True)
    return validate_host_terminal(authority.root, host)


def host_status(root: str | Path, host: str) -> dict[str, Any]:
    """Return validated host progress without writing or launching anything."""

    authority = _load_authority(root)
    detail = _validate_host_layout(authority, host, require_complete=False)
    marker_path = authority.root / "host_terminals" / f"{host}.json"
    terminal = False
    if marker_path.exists():
        validate_host_terminal(authority.root, host)
        terminal = True
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "scope": "host",
        "study_id": authority.spec.study_id,
        "manifest_id": authority.manifest_id,
        **detail,
        "terminal": terminal,
        "state": "terminal" if terminal else "ready_to_finalize" if detail["complete"] else "pending",
        "launched_jobs": 0,
    }


def collector_plan(root: str | Path) -> dict[str, Any]:
    """Return the immutable six-surface collection plan; never write."""

    authority = _load_authority(root)
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "study_id": authority.spec.study_id,
        "config_sha256": authority.spec.config_sha256,
        "manifest_id": authority.manifest_id,
        "surface_count": 6,
        "public_stage_sequence": list(PUBLIC_STAGE_SEQUENCE),
        "routes": [
            {
                key: surface[key]
                for key in ("surface_id", "scheme", "optimizer", "host", "output_path")
            }
            for surface in authority.manifest["surfaces"]
        ],
        "operations": [
            "status",
            "finalize-host",
            "import-shard",
            "finalize-study",
            "validate-study",
        ],
        "launch_capability": False,
        "launched_jobs": 0,
    }


def _same_authority(left: _Authority, right: _Authority) -> None:
    for filename in (STUDY_FILENAME, MANIFEST_FILENAME):
        left_hash = sha256_file(left.root / filename)
        right_hash = sha256_file(right.root / filename)
        if left_hash != right_hash:
            raise _error(
                f"incoming {filename}",
                f"the collector SHA-256 {left_hash}",
                right_hash,
            )


def _validate_import_receipt(collector: _Authority, host: str, shard: Path) -> dict[str, Any]:
    path = collector.root / "import_receipts" / f"{host}.json"
    receipt = _mapping(read_json(path), "import receipt")
    expected_prefix = {
        "schema_version": IMPORT_RECEIPT_SCHEMA_VERSION,
        "state": "published",
        "study_id": collector.spec.study_id,
        "config_sha256": collector.spec.config_sha256,
        "manifest_id": collector.manifest_id,
        "host": host,
        "destination": f"imported_shards/{host}",
        "publication_mode": "same_filesystem_atomic_rename",
        "validated_before_publication": True,
        "official_test_read": False,
        "launched_jobs": 0,
    }
    _require_fields(receipt, expected_prefix, "import receipt")
    artifacts = _records(shard)
    _require_fields(
        receipt,
        {
            "artifact_tree_sha256": sha256_json(artifacts),
            "file_count": len(artifacts),
            "bytes": sum(item["bytes"] for item in artifacts),
        },
        "import receipt",
    )
    if not isinstance(receipt.get("published_at_utc"), str):
        raise _error("import receipt.published_at_utc", "an ISO-8601 string", receipt.get("published_at_utc"))
    return receipt


def import_host_shard(root: str | Path, incoming: str | Path, host: str) -> dict[str, Any]:
    """Validate and atomically publish one terminal host shard.

    ``incoming`` must be ``ROOT/.incoming/<bundle>/shards/<host>``.  The
    directory itself is renamed into ``ROOT/imported_shards/<host>`` so a
    collector can never observe a partially copied shard.
    """

    collector = _load_authority(root)
    if host not in ALLOWED_HOSTS:
        raise _error("host", f"one of {ALLOWED_HOSTS!r}", host)
    source = Path(incoming).expanduser().resolve()
    incoming_root = (collector.root / ".incoming").resolve()
    try:
        source.relative_to(incoming_root)
    except ValueError as exc:
        raise _error("incoming shard", f"a path below {incoming_root}", str(source)) from exc
    if (
        source.name != host
        or source.parent.name != "shards"
        or source.parent.parent.parent != incoming_root
    ):
        raise _error(
            "incoming shard",
            f"ROOT/.incoming/<bundle>/shards/{host}",
            str(source),
        )
    incoming_authority = _load_authority(source)
    _same_authority(collector, incoming_authority)
    validate_host_terminal(source, host)
    existing_ids = set(_surface_dirs(collector))
    imported_root = collector.root / "imported_shards"
    if imported_root.exists():
        for child in imported_root.iterdir():
            if child.is_dir() and not child.is_symlink():
                existing_ids.update(_surface_dirs(_load_authority(child)))
    incoming_ids = set(_surface_dirs(incoming_authority))
    duplicate = sorted(existing_ids & incoming_ids)
    if duplicate:
        raise _error("incoming surfaces", "no duplicate logical surfaces", duplicate)
    destination = imported_root / host
    receipt_path = collector.root / "import_receipts" / f"{host}.json"
    if destination.exists() or receipt_path.exists():
        raise _error(
            "host import destination",
            "an unpublished host destination and receipt",
            {"destination_exists": destination.exists(), "receipt_exists": receipt_path.exists()},
        )
    imported_root.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != imported_root.stat().st_dev:
        raise _error(
            "incoming shard filesystem",
            "the same filesystem as imported_shards for atomic rename",
            {"incoming_st_dev": source.stat().st_dev, "destination_st_dev": imported_root.stat().st_dev},
        )
    before = _records(source)
    os.replace(source, destination)
    after = _records(destination)
    if before != after:
        raise _error("published shard", "byte-identical artifacts after atomic rename", after)
    receipt = {
        "schema_version": IMPORT_RECEIPT_SCHEMA_VERSION,
        "state": "published",
        "study_id": collector.spec.study_id,
        "config_sha256": collector.spec.config_sha256,
        "manifest_id": collector.manifest_id,
        "host": host,
        "incoming_path": str(source),
        "destination": f"imported_shards/{host}",
        "publication_mode": "same_filesystem_atomic_rename",
        "validated_before_publication": True,
        "artifact_tree_sha256": sha256_json(after),
        "file_count": len(after),
        "bytes": sum(item["bytes"] for item in after),
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_test_read": False,
        "launched_jobs": 0,
    }
    atomic_write_json(receipt_path, receipt, canonical=True)
    _validate_import_receipt(collector, host, destination)
    return receipt


def _study_sources(authority: _Authority, *, require_terminal: bool) -> dict[str, _Authority]:
    sources: dict[str, _Authority] = {}
    direct = _surface_dirs(authority)
    if direct:
        hosts = {
            authority.surfaces[surface_id]["host"]
            for surface_id in direct
            if surface_id in authority.surfaces
        }
        unknown = sorted(set(direct) - set(authority.surfaces))
        if unknown:
            raise _error("collector surfaces", "only manifest surface ids", unknown)
        if len(hosts) != 1:
            raise _error("collector direct surfaces", "surfaces from exactly one host", sorted(hosts))
        host = next(iter(hosts))
        sources[host] = authority
    imported_root = authority.root / "imported_shards"
    if imported_root.exists():
        if imported_root.is_symlink() or not imported_root.is_dir():
            raise _error(str(imported_root), "a non-symlink directory", str(imported_root))
        for shard in sorted(imported_root.iterdir(), key=lambda item: item.name):
            if shard.is_symlink() or not shard.is_dir() or shard.name not in ALLOWED_HOSTS:
                raise _error(str(shard), f"a shard directory named from {ALLOWED_HOSTS!r}", str(shard))
            if shard.name in sources:
                raise _error("study host sources", "one source per host", shard.name)
            shard_authority = _load_authority(shard)
            _same_authority(authority, shard_authority)
            _validate_import_receipt(authority, shard.name, shard)
            sources[shard.name] = shard_authority
    if require_terminal:
        if set(sources) != set(ALLOWED_HOSTS):
            raise _error("study host sources", f"exactly {sorted(ALLOWED_HOSTS)!r}", sorted(sources))
        seen: set[str] = set()
        for host, source in sources.items():
            validate_host_terminal(source.root, host)
            ids = set(_surface_dirs(source))
            duplicate = seen & ids
            if duplicate:
                raise _error("study surfaces", "unique logical surfaces", sorted(duplicate))
            seen.update(ids)
        if seen != set(authority.surfaces):
            raise _error("study surfaces", "exactly the six manifest surfaces", sorted(seen))
    return sources


def _study_payload(authority: _Authority) -> dict[str, Any]:
    sources = _study_sources(authority, require_terminal=True)
    rows: list[dict[str, Any]] = []
    selected_status = authority.spec.data["selection"]["selected_status"]
    for raw_surface in authority.manifest["surfaces"]:
        surface = _mapping(raw_surface, "manifest surface")
        source = sources[surface["host"]]
        surface_root = source.root / surface["output_path"]
        final = _validate_surface(source, surface["surface_id"], surface_root)
        final_path = surface_root / "stages" / "finalize_lr" / "result.json"
        completion_path = surface_root / "completion.json"
        row = {
            "surface_index": surface["surface_index"],
            "surface_id": surface["surface_id"],
            "row_id": surface["row_id"],
            "scheme": surface["scheme"],
            "optimizer": surface["optimizer"],
            "host": surface["host"],
            "artifact_root": surface_root.relative_to(authority.root).as_posix(),
            "status": final.get("status"),
            "zero_work": final.get("zero_work"),
            "reason": final.get("reason"),
            "final_result_sha256": sha256_file(final_path),
            "surface_completion_sha256": sha256_file(completion_path),
        }
        for key in (
            "selected_t",
            "selected_k",
            "selected_cell_id",
            "selected_rho",
            "final_validation_loss",
            "final_validation_accuracy",
            "median_projection_efficiency",
            "expansion_used",
        ):
            if key in final:
                row[key] = copy.deepcopy(final[key])
        rows.append(row)
    selected_count = sum(item["status"] == selected_status for item in rows)
    return {
        "schema_version": STUDY_RESULT_SCHEMA_VERSION,
        "state": "complete",
        "study_id": authority.spec.study_id,
        "config_sha256": authority.spec.config_sha256,
        "manifest_id": authority.manifest_id,
        "surface_count": 6,
        "selected_status": selected_status,
        "selected_surface_count": selected_count,
        "unresolved_surface_count": 6 - selected_count,
        "status": "complete" if selected_count == 6 else "complete_with_unresolved_surfaces",
        "surfaces": rows,
        "official_test_read": False,
        "launched_jobs": 0,
    }


def _study_completion_payload(authority: _Authority, result_path: Path) -> dict[str, Any]:
    return {
        "schema_version": STUDY_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "study_id": authority.spec.study_id,
        "config_sha256": authority.spec.config_sha256,
        "manifest_id": authority.manifest_id,
        "surface_count": 6,
        "official_test_read": False,
        "outputs": [_record(result_path, base=authority.root / "finalization")],
        "launched_jobs": 0,
    }


def finalize_study(root: str | Path) -> dict[str, Any]:
    """Publish the terminal aggregate only after all six surfaces validate."""

    authority = _load_authority(root)
    result = _study_payload(authority)
    finalization = authority.root / "finalization"
    result_path = finalization / "result.json"
    completion_path = finalization / "completion.json"
    if completion_path.exists():
        validate_study(authority.root)
        if read_json(result_path) != result:
            raise _error("existing study result", f"exactly {result!r}", read_json(result_path))
        return result
    if result_path.exists():
        raise _error(str(result_path), "absent when no completion marker exists", str(result_path))
    atomic_write_json(result_path, result, canonical=True)
    atomic_write_json(
        completion_path,
        _study_completion_payload(authority, result_path),
        canonical=True,
    )
    validate_study(authority.root)
    return result


def validate_study(root: str | Path) -> dict[str, Any]:
    """Read-only validation of a terminal six-surface study."""

    authority = _load_authority(root)
    expected_result = _study_payload(authority)
    finalization = authority.root / "finalization"
    result_path = finalization / "result.json"
    completion_path = finalization / "completion.json"
    if not result_path.is_file() or not completion_path.is_file():
        raise _error(str(finalization), "terminal result.json and completion.json", str(finalization))
    result = _mapping(read_json(result_path), "study result")
    if result != expected_result:
        raise _error("study result", f"exactly {expected_result!r}", result)
    completion = _mapping(read_json(completion_path), "study completion")
    expected_completion = _study_completion_payload(authority, result_path)
    if completion != expected_completion:
        raise _error("study completion", f"exactly {expected_completion!r}", completion)
    return result


def study_status(root: str | Path) -> dict[str, Any]:
    """Return validated collection progress without changing the study."""

    authority = _load_authority(root)
    sources = _study_sources(authority, require_terminal=False)
    hosts = []
    for host in ALLOWED_HOSTS:
        if host not in sources:
            hosts.append({"host": host, "state": "missing"})
        else:
            hosts.append(host_status(sources[host].root, host))
    terminal = (authority.root / "finalization" / "completion.json").exists()
    if terminal:
        validate_study(authority.root)
    ready = (
        set(sources) == set(ALLOWED_HOSTS)
        and all(item.get("terminal") is True for item in hosts)
    )
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "scope": "study",
        "study_id": authority.spec.study_id,
        "manifest_id": authority.manifest_id,
        "hosts": hosts,
        "surface_count": 6,
        "terminal": terminal,
        "state": "terminal" if terminal else "ready_to_finalize" if ready else "pending",
        "launched_jobs": 0,
    }


__all__ = [
    "HOST_TERMINAL_SCHEMA_VERSION",
    "IMPORT_RECEIPT_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "PLAN_SCHEMA_VERSION",
    "PerfectDiodeConv3OrchestrationError",
    "STATUS_SCHEMA_VERSION",
    "STUDY_COMPLETION_SCHEMA_VERSION",
    "STUDY_FILENAME",
    "STUDY_RESULT_SCHEMA_VERSION",
    "collector_plan",
    "finalize_host_shard",
    "finalize_study",
    "host_status",
    "import_host_shard",
    "study_status",
    "validate_host_terminal",
    "validate_study",
]
