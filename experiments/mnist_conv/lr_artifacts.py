"""Immutable, checksummed artifacts for staged Conv learning-rate studies.

The module deliberately contains no experiment policy.  It provides a small
filesystem contract shared by local and Slurm executors:

* a stage manifest fixes an ordered set of entries and their declared outputs;
* an entry completion marker is published only after every declared output is
  present and checksummed;
* a stage completion marker is published only after every entry marker and its
  outputs have been revalidated; and
* every read revalidates the study config and upstream artifact hashes.

Completion markers and manifests contain no wall-clock fields.  Their bytes
are therefore canonical scientific records rather than execution metadata.
"""

from __future__ import annotations

import copy
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .identity import canonical_json_bytes, normalize_code_provenance, sha256_file
from .io import atomic_write_json, read_json


LR_STAGE_MANIFEST_SCHEMA_VERSION = "mnist-conv-lr-stage-manifest/v1"
LR_ENTRY_COMPLETION_SCHEMA_VERSION = "mnist-conv-lr-entry-completion/v1"
LR_STAGE_COMPLETION_SCHEMA_VERSION = "mnist-conv-lr-stage-completion/v1"

_MANIFEST_FIELDS = {
    "schema_version",
    "study_id",
    "study_config_path",
    "study_config_sha256",
    "code_provenance",
    "stage_name",
    "upstream_artifacts",
    "entries",
}
_ENTRY_FIELDS = {
    "entry_index",
    "entry_id",
    "completion_path",
    "outputs",
    "payload",
}
_ARTIFACT_FIELDS = {"path", "sha256", "bytes"}
_ENTRY_COMPLETION_FIELDS = {
    "schema_version",
    "state",
    "study_id",
    "stage_name",
    "entry_id",
    "stage_manifest",
    "outputs",
}
_STAGE_COMPLETION_FIELDS = {
    "schema_version",
    "state",
    "study_id",
    "stage_name",
    "stage_manifest",
    "entries",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_STUDY_ID_RE = re.compile(r"lrstudy_[0-9a-f]{64}")
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class LRArtifactError(ValueError):
    """An LR artifact is missing, malformed, or no longer matches its hash."""


class LRArtifactConflictError(LRArtifactError):
    """An immutable artifact already exists with different content."""


def _error(expected: str, provided: Any, path: str) -> LRArtifactError:
    return LRArtifactError(
        f"Expected {path} to be {expected}. Provided value: {provided!r}."
    )


def _conflict(expected: str, provided: Any, path: str) -> LRArtifactConflictError:
    return LRArtifactConflictError(
        f"Expected {path} to be {expected}. Provided value: {provided!r}."
    )


def _root(study_dir: str | Path) -> Path:
    raw = Path(study_dir).expanduser().absolute()
    if raw.is_symlink():
        raise _error("a real study directory, not a symlink", str(raw), "study_dir")
    return raw.resolve()


def _portable_relative(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise _error("a non-empty portable relative path", value, path)
    relative = Path(value)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise _error("a non-empty portable relative path", value, path)
    canonical = relative.as_posix()
    if canonical != value:
        raise _error("a normalized portable relative path", value, path)
    return canonical


def _study_path(study_dir: str | Path, value: str | Path, path: str) -> tuple[Path, str]:
    root = _root(study_dir)
    supplied = Path(value).expanduser()
    candidate = supplied.absolute() if supplied.is_absolute() else root / supplied
    resolved = candidate.resolve(strict=False)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise _error("a path contained by the study directory", str(value), path) from exc
    _portable_relative(relative, path)

    # Reject symlinks even when they happen to point back inside the study.
    lexical = candidate.absolute()
    while lexical != root and lexical != lexical.parent:
        if lexical.is_symlink():
            raise _error("a real path without symlink components", str(value), path)
        lexical = lexical.parent
    return resolved, relative


def _validate_digest(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _error("a lowercase SHA-256 digest", value, path)
    return value


def _validate_artifact_record(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _ARTIFACT_FIELDS:
        raise _error(
            f"an object with exactly keys {sorted(_ARTIFACT_FIELDS)!r}", value, path
        )
    relative = _portable_relative(value["path"], f"{path}.path")
    digest = _validate_digest(value["sha256"], f"{path}.sha256")
    byte_count = value["bytes"]
    if type(byte_count) is not int or byte_count < 0:
        raise _error("a non-negative integer", byte_count, f"{path}.bytes")
    return {"path": relative, "sha256": digest, "bytes": byte_count}


def _validate_record_list(value: Any, path: str, *, allow_empty: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise _error(qualifier, value, path)
    records = [
        _validate_artifact_record(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    ]
    record_paths = [record["path"] for record in records]
    if record_paths != sorted(set(record_paths)):
        raise _error(
            "records sorted by unique portable paths", record_paths, path
        )
    return records


def artifact_record(study_dir: str | Path, artifact: str | Path) -> dict[str, Any]:
    """Return a portable SHA-256/size record for one regular study artifact."""

    target, relative = _study_path(study_dir, artifact, "artifact")
    if target.is_symlink() or not target.is_file():
        raise _error("an existing regular file", str(target), "artifact")
    return {
        "path": relative,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }


def artifact_records(
    study_dir: str | Path, artifacts: Iterable[str | Path]
) -> list[dict[str, Any]]:
    """Record artifacts in canonical path order and reject duplicate paths."""

    records = sorted(
        (artifact_record(study_dir, artifact) for artifact in artifacts),
        key=lambda record: record["path"],
    )
    paths = [record["path"] for record in records]
    if len(paths) != len(set(paths)):
        raise _error("a set of unique artifact paths", paths, "artifacts")
    return records


def validate_artifact_records(
    study_dir: str | Path, records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Recompute every record and reject missing, resized, or changed files."""

    normalized = _validate_record_list(list(records), "artifacts", allow_empty=True)
    observed = artifact_records(study_dir, [record["path"] for record in normalized])
    if observed != normalized:
        raise _error("the current artifact hashes and sizes", normalized, "artifacts")
    return copy.deepcopy(normalized)


def stage_entry(
    entry_index: int,
    entry_id: str,
    *,
    completion_path: str,
    outputs: Sequence[str],
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one explicit ordered stage-entry declaration."""

    value = {
        "entry_index": entry_index,
        "entry_id": entry_id,
        "completion_path": completion_path,
        "outputs": list(outputs),
        "payload": dict(payload or {}),
    }
    # Reuse the complete manifest validator's entry checks through this small
    # local validator so construction fails before anything reaches disk.
    return _validate_entry(value, entry_index, "entry")


def _validate_entry(value: Any, expected_index: int, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _ENTRY_FIELDS:
        raise _error(
            f"an object with exactly keys {sorted(_ENTRY_FIELDS)!r}", value, path
        )
    index = value["entry_index"]
    if type(index) is not int or index != expected_index:
        raise _error(
            f"the contiguous integer {expected_index}", index, f"{path}.entry_index"
        )
    entry_id = value["entry_id"]
    if not isinstance(entry_id, str) or _NAME_RE.fullmatch(entry_id) is None:
        raise _error(
            "a normalized identifier containing letters, digits, '.', '_', or '-'",
            entry_id,
            f"{path}.entry_id",
        )
    completion_path = _portable_relative(
        value["completion_path"], f"{path}.completion_path"
    )
    outputs_value = value["outputs"]
    if not isinstance(outputs_value, list) or not outputs_value:
        raise _error("a non-empty list", outputs_value, f"{path}.outputs")
    outputs = [
        _portable_relative(item, f"{path}.outputs[{output_index}]")
        for output_index, item in enumerate(outputs_value)
    ]
    if outputs != sorted(set(outputs)):
        raise _error("sorted unique output paths", outputs, f"{path}.outputs")
    if completion_path in outputs:
        raise _error(
            "a path distinct from declared outputs", completion_path, f"{path}.completion_path"
        )
    payload = value["payload"]
    if not isinstance(payload, dict):
        raise _error("a JSON object", payload, f"{path}.payload")
    try:
        canonical_json_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise _error("a finite canonical JSON object", payload, f"{path}.payload") from exc
    return {
        "entry_index": index,
        "entry_id": entry_id,
        "completion_path": completion_path,
        "outputs": outputs,
        "payload": copy.deepcopy(payload),
    }


def validate_stage_manifest(value: Any) -> dict[str, Any]:
    """Validate manifest structure without consulting the filesystem."""

    if not isinstance(value, dict) or set(value) != _MANIFEST_FIELDS:
        raise _error(
            f"an object with exactly keys {sorted(_MANIFEST_FIELDS)!r}",
            value,
            "stage_manifest",
        )
    if value["schema_version"] != LR_STAGE_MANIFEST_SCHEMA_VERSION:
        raise _error(
            f"exactly {LR_STAGE_MANIFEST_SCHEMA_VERSION!r}",
            value["schema_version"],
            "stage_manifest.schema_version",
        )
    study_id = value["study_id"]
    if not isinstance(study_id, str) or _STUDY_ID_RE.fullmatch(study_id) is None:
        raise _error("a canonical LR study id", study_id, "stage_manifest.study_id")
    config_path = _portable_relative(
        value["study_config_path"], "stage_manifest.study_config_path"
    )
    config_sha = _validate_digest(
        value["study_config_sha256"], "stage_manifest.study_config_sha256"
    )
    try:
        provenance = normalize_code_provenance(value["code_provenance"])
    except ValueError as exc:
        raise LRArtifactError(str(exc)) from exc
    stage_name = value["stage_name"]
    if not isinstance(stage_name, str) or _NAME_RE.fullmatch(stage_name) is None:
        raise _error(
            "a normalized stage identifier", stage_name, "stage_manifest.stage_name"
        )
    upstream = _validate_record_list(
        value["upstream_artifacts"],
        "stage_manifest.upstream_artifacts",
        allow_empty=True,
    )
    entries_value = value["entries"]
    if not isinstance(entries_value, list) or not entries_value:
        raise _error("a non-empty ordered list", entries_value, "stage_manifest.entries")
    entries = [
        _validate_entry(item, index, f"stage_manifest.entries[{index}]")
        for index, item in enumerate(entries_value)
    ]
    entry_ids = [entry["entry_id"] for entry in entries]
    if len(entry_ids) != len(set(entry_ids)):
        raise _error("unique entry ids", entry_ids, "stage_manifest.entries")

    owned_paths: list[str] = []
    for entry in entries:
        owned_paths.append(entry["completion_path"])
        owned_paths.extend(entry["outputs"])
    upstream_paths = [record["path"] for record in upstream]
    collisions = sorted(
        path
        for path in set(owned_paths)
        if owned_paths.count(path) > 1 or path in upstream_paths or path == config_path
    )
    if collisions:
        raise _error(
            "globally unique entry paths disjoint from upstream/config artifacts",
            collisions,
            "stage_manifest.entries",
        )
    return {
        "schema_version": LR_STAGE_MANIFEST_SCHEMA_VERSION,
        "study_id": study_id,
        "study_config_path": config_path,
        "study_config_sha256": config_sha,
        "code_provenance": provenance,
        "stage_name": stage_name,
        "upstream_artifacts": upstream,
        "entries": entries,
    }


def build_stage_manifest(
    *,
    study_dir: str | Path,
    study_id: str,
    study_config_path: str | Path,
    code_provenance: Mapping[str, Any],
    stage_name: str,
    entries: Sequence[Mapping[str, Any]],
    upstream_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Build a manifest from current config/upstream file hashes."""

    config_record = artifact_record(study_dir, study_config_path)
    value = {
        "schema_version": LR_STAGE_MANIFEST_SCHEMA_VERSION,
        "study_id": study_id,
        "study_config_path": config_record["path"],
        "study_config_sha256": config_record["sha256"],
        "code_provenance": dict(code_provenance),
        "stage_name": stage_name,
        "upstream_artifacts": artifact_records(study_dir, upstream_paths),
        "entries": [copy.deepcopy(dict(entry)) for entry in entries],
    }
    return validate_stage_manifest(value)


def _load_canonical_json(path: Path, artifact_name: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise _error(f"an existing regular {artifact_name}", str(path), artifact_name)
    try:
        value = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise _error("strict canonical JSON", str(path), artifact_name) from exc
    try:
        expected_bytes = canonical_json_bytes(value) + b"\n"
    except (TypeError, ValueError) as exc:
        raise _error("strict canonical JSON", value, artifact_name) from exc
    if path.read_bytes() != expected_bytes:
        raise _error("canonical JSON bytes", str(path), artifact_name)
    return value


def _publish_no_clobber(path: Path, value: Mapping[str, Any], artifact_name: str) -> Path:
    """Publish canonical JSON atomically without ever replacing an existing file."""

    expected = copy.deepcopy(dict(value))
    if path.exists() or path.is_symlink():
        observed = _load_canonical_json(path, artifact_name)
        if observed != expected:
            raise _conflict("identical immutable content", observed, artifact_name)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.candidate")
    try:
        atomic_write_json(candidate, expected, canonical=True)
        try:
            os.link(candidate, path)
        except FileExistsError:
            observed = _load_canonical_json(path, artifact_name)
            if observed != expected:
                raise _conflict("identical immutable content", observed, artifact_name)
    finally:
        if candidate.exists():
            candidate.unlink()
    return path


def _validate_manifest_files(study_dir: str | Path, manifest: dict[str, Any]) -> None:
    config = artifact_record(study_dir, manifest["study_config_path"])
    if config["sha256"] != manifest["study_config_sha256"]:
        raise _error(
            f"the recorded SHA-256 {manifest['study_config_sha256']!r}",
            config["sha256"],
            "study config SHA-256",
        )
    validate_artifact_records(study_dir, manifest["upstream_artifacts"])


def load_stage_manifest(
    manifest_path: str | Path,
    *,
    study_dir: str | Path,
    expected_study_id: str | None = None,
    expected_stage_name: str | None = None,
) -> dict[str, Any]:
    """Load a canonical manifest and revalidate config and upstream hashes."""

    path, _ = _study_path(study_dir, manifest_path, "stage_manifest")
    manifest = validate_stage_manifest(_load_canonical_json(path, "stage_manifest"))
    if expected_study_id is not None and manifest["study_id"] != expected_study_id:
        raise _error(expected_study_id, manifest["study_id"], "stage_manifest.study_id")
    if expected_stage_name is not None and manifest["stage_name"] != expected_stage_name:
        raise _error(
            expected_stage_name, manifest["stage_name"], "stage_manifest.stage_name"
        )
    _validate_manifest_files(study_dir, manifest)
    return manifest


def publish_stage_manifest(
    manifest_path: str | Path,
    manifest: Mapping[str, Any],
    *,
    study_dir: str | Path,
) -> Path:
    """Publish a new immutable manifest or verify an identical existing one."""

    path, relative = _study_path(study_dir, manifest_path, "stage_manifest")
    normalized = validate_stage_manifest(dict(manifest))
    if relative in {normalized["study_config_path"]} | {
        record["path"] for record in normalized["upstream_artifacts"]
    }:
        raise _error("a path distinct from config/upstream artifacts", relative, "stage_manifest")
    for entry in normalized["entries"]:
        if relative == entry["completion_path"] or relative in entry["outputs"]:
            raise _error("a path distinct from entry artifacts", relative, "stage_manifest")
    _validate_manifest_files(study_dir, normalized)
    return _publish_no_clobber(path, normalized, "stage_manifest")


def _entry_by_id(manifest: dict[str, Any], entry_id: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["entries"] if entry["entry_id"] == entry_id]
    if len(matches) != 1:
        raise _error("one entry declared by the stage manifest", entry_id, "entry_id")
    return matches[0]


def _expected_entry_completion(
    study_dir: str | Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": LR_ENTRY_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "study_id": manifest["study_id"],
        "stage_name": manifest["stage_name"],
        "entry_id": entry["entry_id"],
        "stage_manifest": artifact_record(study_dir, manifest_path),
        "outputs": artifact_records(study_dir, entry["outputs"]),
    }


def _validate_entry_completion_shape(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _ENTRY_COMPLETION_FIELDS:
        raise _error(
            f"an object with exactly keys {sorted(_ENTRY_COMPLETION_FIELDS)!r}",
            value,
            "entry_completion",
        )
    if value["schema_version"] != LR_ENTRY_COMPLETION_SCHEMA_VERSION:
        raise _error(
            f"exactly {LR_ENTRY_COMPLETION_SCHEMA_VERSION!r}",
            value["schema_version"],
            "entry_completion.schema_version",
        )
    if value["state"] != "complete":
        raise _error("exactly 'complete'", value["state"], "entry_completion.state")
    if not isinstance(value["study_id"], str) or _STUDY_ID_RE.fullmatch(value["study_id"]) is None:
        raise _error("a canonical LR study id", value["study_id"], "entry_completion.study_id")
    for name in ("stage_name", "entry_id"):
        item = value[name]
        if not isinstance(item, str) or _NAME_RE.fullmatch(item) is None:
            raise _error("a normalized identifier", item, f"entry_completion.{name}")
    stage_manifest = _validate_artifact_record(
        value["stage_manifest"], "entry_completion.stage_manifest"
    )
    outputs = _validate_record_list(
        value["outputs"], "entry_completion.outputs", allow_empty=False
    )
    return {
        "schema_version": LR_ENTRY_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "study_id": value["study_id"],
        "stage_name": value["stage_name"],
        "entry_id": value["entry_id"],
        "stage_manifest": stage_manifest,
        "outputs": outputs,
    }


def publish_entry_completion(
    *,
    study_dir: str | Path,
    manifest_path: str | Path,
    entry_id: str,
) -> Path:
    """Write an entry marker last, after all declared outputs are validated."""

    manifest_file, _ = _study_path(study_dir, manifest_path, "stage_manifest")
    manifest = load_stage_manifest(manifest_file, study_dir=study_dir)
    entry = _entry_by_id(manifest, entry_id)
    marker, _ = _study_path(study_dir, entry["completion_path"], "entry_completion")
    expected = _expected_entry_completion(study_dir, manifest_file, manifest, entry)
    return _publish_no_clobber(marker, expected, "entry_completion")


def validate_entry_completion(
    *,
    study_dir: str | Path,
    manifest_path: str | Path,
    entry_id: str,
) -> dict[str, Any]:
    """Validate one completion marker and every artifact it covers."""

    manifest_file, _ = _study_path(study_dir, manifest_path, "stage_manifest")
    manifest = load_stage_manifest(manifest_file, study_dir=study_dir)
    entry = _entry_by_id(manifest, entry_id)
    marker, _ = _study_path(study_dir, entry["completion_path"], "entry_completion")
    observed = _validate_entry_completion_shape(
        _load_canonical_json(marker, "entry_completion")
    )
    expected = _expected_entry_completion(study_dir, manifest_file, manifest, entry)
    if observed != expected:
        raise _error(
            "the current stage manifest and declared output hashes",
            observed,
            "entry_completion",
        )
    return observed


def entry_is_complete(
    *,
    study_dir: str | Path,
    manifest_path: str | Path,
    entry_id: str,
) -> bool:
    """Return false only for a missing marker; malformed completed work raises."""

    manifest_file, _ = _study_path(study_dir, manifest_path, "stage_manifest")
    manifest = load_stage_manifest(manifest_file, study_dir=study_dir)
    entry = _entry_by_id(manifest, entry_id)
    marker, _ = _study_path(study_dir, entry["completion_path"], "entry_completion")
    if not marker.exists() and not marker.is_symlink():
        return False
    validate_entry_completion(
        study_dir=study_dir, manifest_path=manifest_file, entry_id=entry_id
    )
    return True


def _expected_stage_completion(
    study_dir: str | Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    entries = []
    for entry in manifest["entries"]:
        validate_entry_completion(
            study_dir=study_dir,
            manifest_path=manifest_path,
            entry_id=entry["entry_id"],
        )
        entries.append(
            {
                "entry_index": entry["entry_index"],
                "entry_id": entry["entry_id"],
                "completion": artifact_record(study_dir, entry["completion_path"]),
            }
        )
    return {
        "schema_version": LR_STAGE_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "study_id": manifest["study_id"],
        "stage_name": manifest["stage_name"],
        "stage_manifest": artifact_record(study_dir, manifest_path),
        "entries": entries,
    }


def _validate_stage_completion_shape(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _STAGE_COMPLETION_FIELDS:
        raise _error(
            f"an object with exactly keys {sorted(_STAGE_COMPLETION_FIELDS)!r}",
            value,
            "stage_completion",
        )
    if value["schema_version"] != LR_STAGE_COMPLETION_SCHEMA_VERSION:
        raise _error(
            f"exactly {LR_STAGE_COMPLETION_SCHEMA_VERSION!r}",
            value["schema_version"],
            "stage_completion.schema_version",
        )
    if value["state"] != "complete":
        raise _error("exactly 'complete'", value["state"], "stage_completion.state")
    if not isinstance(value["study_id"], str) or _STUDY_ID_RE.fullmatch(value["study_id"]) is None:
        raise _error("a canonical LR study id", value["study_id"], "stage_completion.study_id")
    if not isinstance(value["stage_name"], str) or _NAME_RE.fullmatch(value["stage_name"]) is None:
        raise _error("a normalized identifier", value["stage_name"], "stage_completion.stage_name")
    stage_manifest = _validate_artifact_record(
        value["stage_manifest"], "stage_completion.stage_manifest"
    )
    entries_value = value["entries"]
    if not isinstance(entries_value, list) or not entries_value:
        raise _error("a non-empty ordered list", entries_value, "stage_completion.entries")
    entries = []
    for index, entry in enumerate(entries_value):
        fields = {"entry_index", "entry_id", "completion"}
        if not isinstance(entry, dict) or set(entry) != fields:
            raise _error(
                f"an object with exactly keys {sorted(fields)!r}",
                entry,
                f"stage_completion.entries[{index}]",
            )
        if type(entry["entry_index"]) is not int or entry["entry_index"] != index:
            raise _error(
                f"the contiguous integer {index}",
                entry["entry_index"],
                f"stage_completion.entries[{index}].entry_index",
            )
        if not isinstance(entry["entry_id"], str) or _NAME_RE.fullmatch(entry["entry_id"]) is None:
            raise _error(
                "a normalized identifier",
                entry["entry_id"],
                f"stage_completion.entries[{index}].entry_id",
            )
        entries.append(
            {
                "entry_index": index,
                "entry_id": entry["entry_id"],
                "completion": _validate_artifact_record(
                    entry["completion"],
                    f"stage_completion.entries[{index}].completion",
                ),
            }
        )
    return {
        "schema_version": LR_STAGE_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "study_id": value["study_id"],
        "stage_name": value["stage_name"],
        "stage_manifest": stage_manifest,
        "entries": entries,
    }


def publish_stage_completion(
    *,
    study_dir: str | Path,
    manifest_path: str | Path,
    completion_path: str | Path | None = None,
) -> Path:
    """Publish the aggregate marker last, after all entries revalidate."""

    manifest_file, _ = _study_path(study_dir, manifest_path, "stage_manifest")
    manifest = load_stage_manifest(manifest_file, study_dir=study_dir)
    if completion_path is None:
        marker = manifest_file.parent / "complete.json"
    else:
        marker, _ = _study_path(study_dir, completion_path, "stage_completion")
    marker, marker_relative = _study_path(study_dir, marker, "stage_completion")
    reserved = {
        manifest["study_config_path"],
        *(record["path"] for record in manifest["upstream_artifacts"]),
        *(entry["completion_path"] for entry in manifest["entries"]),
        *(output for entry in manifest["entries"] for output in entry["outputs"]),
    }
    _, manifest_relative = _study_path(study_dir, manifest_file, "stage_manifest")
    reserved.add(manifest_relative)
    if marker_relative in reserved:
        raise _error("a path distinct from all stage artifacts", marker_relative, "stage_completion")
    expected = _expected_stage_completion(study_dir, manifest_file, manifest)
    return _publish_no_clobber(marker, expected, "stage_completion")


def validate_stage_completion(
    *,
    study_dir: str | Path,
    manifest_path: str | Path,
    completion_path: str | Path | None = None,
) -> dict[str, Any]:
    """Revalidate an aggregate marker, every entry marker, and every output."""

    manifest_file, _ = _study_path(study_dir, manifest_path, "stage_manifest")
    manifest = load_stage_manifest(manifest_file, study_dir=study_dir)
    marker = manifest_file.parent / "complete.json" if completion_path is None else completion_path
    marker_file, _ = _study_path(study_dir, marker, "stage_completion")
    observed = _validate_stage_completion_shape(
        _load_canonical_json(marker_file, "stage_completion")
    )
    expected = _expected_stage_completion(study_dir, manifest_file, manifest)
    if observed != expected:
        raise _error(
            "the current stage manifest and entry completion hashes",
            observed,
            "stage_completion",
        )
    return observed


def stage_is_complete(
    *,
    study_dir: str | Path,
    manifest_path: str | Path,
    completion_path: str | Path | None = None,
) -> bool:
    """Return false only when the aggregate marker is absent; corruption raises."""

    manifest_file, _ = _study_path(study_dir, manifest_path, "stage_manifest")
    marker = manifest_file.parent / "complete.json" if completion_path is None else completion_path
    marker_file, _ = _study_path(study_dir, marker, "stage_completion")
    if not marker_file.exists() and not marker_file.is_symlink():
        return False
    validate_stage_completion(
        study_dir=study_dir,
        manifest_path=manifest_file,
        completion_path=marker_file,
    )
    return True
