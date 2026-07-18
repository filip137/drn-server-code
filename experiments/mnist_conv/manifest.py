"""Deterministic sweep planning and immutable manifest publication."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .identity import (
    canonical_json_bytes,
    code_identity,
    normalize_code_provenance,
    run_fingerprint,
    sweep_entries_fingerprint,
    sweep_fingerprint,
)
from .io import atomic_write_bytes, atomic_write_json, read_json, relative_posix
from .layout import ResultLayout, safe_label
from .specs import RunSpec, SpecValidationError, SweepSpec, pointer_get


MANIFEST_SCHEMA_VERSION = "mnist-conv-manifest/v1"
_MANIFEST_FIELDS = {
    "schema_version",
    "sweep_id",
    "sweep_name",
    "code_fingerprint",
    "code_identity",
    "code_provenance",
    "varying_fields",
    "collection",
    "entries",
}
_ENTRY_FIELDS = {
    "job_index",
    "logical_key",
    "case_id",
    "axes",
    "run_id",
    "run_spec",
    "run_relpath",
}
_COLLECTION_FIELDS = {"expected_seeds", "required_cases", "group_by"}


class ManifestConflictError(RuntimeError):
    pass


def _error(expected: str, provided: Any, path: str) -> SpecValidationError:
    return SpecValidationError(f"Expected {path} to be {expected}. Provided value: {provided!r}.")


def _json_pointer(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("/") and value != "/"


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            flattened.update(_flatten(value[key], f"{prefix}/{key}"))
        return flattened
    if isinstance(value, list):
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(_flatten(item, f"{prefix}/{index}"))
        return flattened
    return {prefix: value}


def _covers(declaration: str, leaf: str) -> bool:
    return leaf == declaration or leaf.startswith(declaration.rstrip("/") + "/")


def _plan_payload(
    manifest: dict[str, Any],
    *,
    include_name: bool,
    include_run_labels: bool,
) -> dict[str, Any]:
    """Return immutable manifest content without location-dependent paths."""

    payload = {
        "schema_version": manifest["schema_version"],
        "sweep_id": manifest["sweep_id"],
        "code_fingerprint": manifest["code_fingerprint"],
        "code_identity": manifest["code_identity"],
        "code_provenance": copy.deepcopy(manifest["code_provenance"]),
        "varying_fields": copy.deepcopy(manifest["varying_fields"]),
        "collection": copy.deepcopy(manifest["collection"]),
        "entries": [
            {key: copy.deepcopy(value) for key, value in entry.items() if key != "run_relpath"}
            for entry in manifest["entries"]
        ],
    }
    if include_name:
        payload["sweep_name"] = manifest["sweep_name"]
    if not include_run_labels:
        for entry in payload["entries"]:
            entry["run_spec"] = RunSpec.from_dict(entry["run_spec"]).identity_payload()
    return payload


def _with_run_paths(
    manifest: dict[str, Any],
    *,
    sweep_dir: Path,
    layout: ResultLayout,
) -> dict[str, Any]:
    result = copy.deepcopy(manifest)
    for entry in result["entries"]:
        entry["run_relpath"] = relative_posix(
            layout.run_dir(entry["run_spec"]["label"], entry["run_id"]),
            sweep_dir,
        )
    return result


def build_manifest(
    spec: SweepSpec,
    source_provenance: dict[str, Any],
) -> dict[str, Any]:
    source_provenance = normalize_code_provenance(source_provenance)
    expanded = spec.expand()
    run_ids = [run_fingerprint(item.spec, source_provenance) for item in expanded]
    duplicates: dict[str, list[str]] = {}
    for item, run_id in zip(expanded, run_ids):
        duplicates.setdefault(run_id, []).append(item.logical_key)
    repeated = {run_id: keys for run_id, keys in duplicates.items() if len(keys) > 1}
    if repeated:
        raise SpecValidationError(
            "Expected every expanded logical run to have a distinct run identity. "
            f"Provided duplicates: {repeated!r}."
        )
    sweep_id = sweep_fingerprint(spec, expanded, run_ids, source_provenance)
    entries = []
    for item, run_id in zip(expanded, run_ids):
        entries.append(
            {
                "job_index": item.job_index,
                "logical_key": item.logical_key,
                "case_id": item.case_id,
                "axes": copy.deepcopy(item.axes),
                "run_id": run_id,
                "run_spec": item.spec.to_dict(),
            }
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "sweep_id": sweep_id,
        "sweep_name": spec.data["name"],
        "code_fingerprint": source_provenance["effective_code_fingerprint"],
        "code_identity": code_identity(source_provenance),
        "code_provenance": copy.deepcopy(source_provenance),
        "varying_fields": copy.deepcopy(spec.data["varying_fields"]),
        "collection": copy.deepcopy(spec.data["collection"]),
        "entries": entries,
    }


def _validate_varying_and_collection(value: dict[str, Any]) -> None:
    varying = value["varying_fields"]
    if (
        not isinstance(varying, list)
        or any(not _json_pointer(pointer) for pointer in varying)
        or len(varying) != len(set(varying))
        or varying != sorted(varying)
    ):
        raise _error(
            "a sorted list of unique non-root JSON pointers",
            varying,
            "manifest.varying_fields",
        )

    collection = value["collection"]
    if not isinstance(collection, dict) or set(collection) != _COLLECTION_FIELDS:
        raise _error(
            f"an object with exactly keys {sorted(_COLLECTION_FIELDS)!r}",
            collection,
            "manifest.collection",
        )
    seeds = collection["expected_seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise _error(
            "a non-empty list of unique non-negative integers",
            seeds,
            "manifest.collection.expected_seeds",
        )
    cases = collection["required_cases"]
    if (
        not isinstance(cases, list)
        or not cases
        or any(not isinstance(case_id, str) or not case_id.strip() for case_id in cases)
        or len(cases) != len(set(cases))
    ):
        raise _error(
            "a non-empty list of unique non-empty case ids",
            cases,
            "manifest.collection.required_cases",
        )
    group_by = collection["group_by"]
    if (
        not isinstance(group_by, list)
        or any(not _json_pointer(pointer) for pointer in group_by)
        or len(group_by) != len(set(group_by))
        or set(group_by) & {"/seed", "/replicate_id"}
    ):
        raise _error(
            "a list of unique non-root JSON pointers excluding /seed and /replicate_id",
            group_by,
            "manifest.collection.group_by",
        )


def validate_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _error("a JSON object", value, "manifest")
    missing, extra = sorted(_MANIFEST_FIELDS - set(value)), sorted(set(value) - _MANIFEST_FIELDS)
    if missing or extra:
        raise _error(
            f"an object with exactly keys {sorted(_MANIFEST_FIELDS)!r}",
            {"missing": missing, "extra": extra},
            "manifest",
        )
    if value["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise _error(MANIFEST_SCHEMA_VERSION, value["schema_version"], "manifest.schema_version")
    if (
        not isinstance(value["sweep_name"], str)
        or not value["sweep_name"].strip()
        or value["sweep_name"] != value["sweep_name"].strip()
    ):
        raise _error("a normalized non-empty string", value["sweep_name"], "manifest.sweep_name")

    try:
        provenance = normalize_code_provenance(value["code_provenance"])
    except ValueError as exc:
        raise SpecValidationError(str(exc)) from exc
    if value["code_fingerprint"] != provenance["effective_code_fingerprint"]:
        raise _error(
            "code_provenance.effective_code_fingerprint",
            value["code_fingerprint"],
            "manifest.code_fingerprint",
        )
    expected_code_identity = code_identity(provenance)
    if value["code_identity"] != expected_code_identity:
        raise _error(expected_code_identity, value["code_identity"], "manifest.code_identity")
    _validate_varying_and_collection(value)

    entries = value["entries"]
    if not isinstance(entries, list) or not entries:
        raise _error("a non-empty list", entries, "manifest.entries")
    seen_run_ids: set[str] = set()
    cases_in_order: list[str] = []
    seeds_by_case: dict[str, set[int]] = {}
    specs: list[RunSpec] = []
    identity_entries: list[dict[str, Any]] = []
    for expected_index, entry in enumerate(entries):
        path = f"manifest.entries[{expected_index}]"
        if not isinstance(entry, dict) or set(entry) != _ENTRY_FIELDS:
            raise _error(
                f"an object with exactly keys {sorted(_ENTRY_FIELDS)!r}",
                entry,
                path,
            )
        job_index = entry["job_index"]
        if isinstance(job_index, bool) or not isinstance(job_index, int) or job_index != expected_index:
            raise _error(
                f"the contiguous integer index {expected_index}",
                job_index,
                f"{path}.job_index",
            )
        if not isinstance(entry["case_id"], str) or not entry["case_id"].strip():
            raise _error("a non-empty string", entry["case_id"], f"{path}.case_id")
        if not isinstance(entry["axes"], dict):
            raise _error("an object", entry["axes"], f"{path}.axes")

        spec = RunSpec.from_dict(entry["run_spec"])
        specs.append(spec)
        expected_run_id = run_fingerprint(spec, provenance)
        if entry["run_id"] != expected_run_id:
            raise _error(expected_run_id, entry["run_id"], f"{path}.run_id")
        if entry["run_id"] in seen_run_ids:
            raise _error("a unique run id", entry["run_id"], f"{path}.run_id")
        seen_run_ids.add(entry["run_id"])

        for pointer, axis_value in entry["axes"].items():
            if pointer not in value["varying_fields"]:
                raise _error(
                    "a path declared by manifest.varying_fields",
                    pointer,
                    f"{path}.axes",
                )
            resolved_value = pointer_get(spec.to_dict(), pointer)
            if resolved_value != axis_value:
                raise _error(
                    f"the resolved run value {resolved_value!r}",
                    axis_value,
                    f"{path}.axes[{pointer!r}]",
                )
        for pointer in value["varying_fields"] + value["collection"]["group_by"]:
            pointer_get(spec.to_dict(), pointer)

        case_id = entry["case_id"]
        if case_id not in seeds_by_case:
            cases_in_order.append(case_id)
            seeds_by_case[case_id] = set()
        seeds_by_case[case_id].add(spec.data["seed"])

        expected_name = f"{safe_label(spec.data['label'])}--{entry['run_id']}"
        relative_value = entry["run_relpath"]
        relative = Path(relative_value) if isinstance(relative_value, str) else None
        expected_parts = ("..", "..", "runs", expected_name)
        if (
            relative is None
            or "\\" in relative_value
            or relative.is_absolute()
            or relative.parts != expected_parts
        ):
            raise _error(
                f"the portable canonical path {'/'.join(expected_parts)!r}",
                relative_value,
                f"{path}.run_relpath",
            )
        identity_entries.append(
            {
                "logical_key": entry["logical_key"],
                "case_id": case_id,
                "axes": copy.deepcopy(entry["axes"]),
                "run_id": entry["run_id"],
            }
        )

    collection = value["collection"]
    if cases_in_order != collection["required_cases"]:
        raise _error(
            f"entries grouped in required-case order {collection['required_cases']!r}",
            cases_in_order,
            "manifest.entries",
        )
    expected_seeds = set(collection["expected_seeds"])
    if any(seeds != expected_seeds for seeds in seeds_by_case.values()):
        raise _error(
            f"every case to cover seeds {collection['expected_seeds']!r}",
            {case_id: sorted(seeds) for case_id, seeds in seeds_by_case.items()},
            "manifest.entries",
        )

    flattened = [_flatten(spec.identity_payload()) for spec in specs]
    leaves = sorted(set().union(*(item.keys() for item in flattened)))
    changed = {
        leaf
        for leaf in leaves
        if len(
            {
                canonical_json_bytes(item.get(leaf)).decode("utf-8")
                for item in flattened
            }
        )
        > 1
    }
    varying = value["varying_fields"]
    uncovered = sorted(leaf for leaf in changed if not any(_covers(pointer, leaf) for pointer in varying))
    # A manifest contains resolved jobs, but not the pre-override base run.  A
    # field changed once from that base can therefore be constant across every
    # manifest entry. SweepSpec validates both assignment coverage and unused
    # declarations against the base before publication; here we can still
    # prove that no observed inter-job variation is undeclared.
    if uncovered:
        raise _error(
            "coverage of all observed scientific variation",
            {"uncovered": uncovered},
            "manifest.varying_fields",
        )

    try:
        expected_sweep_id = sweep_entries_fingerprint(
            identity_entries,
            varying_fields=varying,
            collection=collection,
            source_provenance=provenance,
        )
    except ValueError as exc:
        raise SpecValidationError(str(exc)) from exc
    if value["sweep_id"] != expected_sweep_id:
        raise _error(expected_sweep_id, value["sweep_id"], "manifest.sweep_id")
    return copy.deepcopy(value)


def _jobs_bytes(manifest: dict[str, Any]) -> bytes:
    return b"".join(canonical_json_bytes(entry) + b"\n" for entry in manifest["entries"])


def _validate_companions(path: Path, manifest: dict[str, Any]) -> None:
    """Cross-check canonical planning files while permitting manifest-only copies."""

    resolved_path = path.parent / "sweep.resolved.json"
    jobs_path = path.parent / "jobs.jsonl"
    canonical_location = path.name == "manifest.json" and path.parent.parent.name == "sweeps"
    if canonical_location and not path.parent.name.endswith(f"--{manifest['sweep_id']}"):
        raise _error(
            f"a directory ending in '--{manifest['sweep_id']}'",
            path.parent.name,
            "manifest path",
        )
    if canonical_location and (not resolved_path.is_file() or not jobs_path.is_file()):
        raise _error(
            "canonical sweep.resolved.json and jobs.jsonl companions",
            {
                "sweep.resolved.json": resolved_path.is_file(),
                "jobs.jsonl": jobs_path.is_file(),
            },
            "manifest directory",
        )

    if resolved_path.is_file():
        try:
            resolved_spec = SweepSpec.from_path(resolved_path)
            expected = build_manifest(resolved_spec, manifest["code_provenance"])
        except (OSError, ValueError) as exc:
            raise SpecValidationError(
                "Expected sweep.resolved.json to be a valid canonical sweep matching manifest.json. "
                f"Provided path: {resolved_path}; error: {exc}."
            ) from exc
        if _plan_payload(
            expected,
            include_name=True,
            include_run_labels=True,
        ) != _plan_payload(
            manifest,
            include_name=True,
            include_run_labels=True,
        ):
            raise _error(
                "the same immutable plan as sweep.resolved.json",
                str(resolved_path),
                "manifest",
            )
    if jobs_path.is_file():
        expected_jobs = _jobs_bytes(manifest)
        if jobs_path.read_bytes() != expected_jobs:
            raise _error(
                "canonical jobs matching manifest.entries",
                str(jobs_path),
                "jobs.jsonl",
            )


def _read_existing_manifest(path: Path) -> dict[str, Any]:
    try:
        return load_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ManifestConflictError(
            "Expected an existing sweep-id directory to contain an intact immutable manifest. "
            f"Provided path: {path}; error: {exc}."
        ) from exc


def publish_manifest(
    spec: SweepSpec,
    layout: ResultLayout,
    source_provenance: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    candidate = build_manifest(spec, source_provenance)
    try:
        existing_dir = layout.find_sweep_dir(candidate["sweep_id"])
    except RuntimeError as exc:
        raise ManifestConflictError(str(exc)) from exc

    if existing_dir is not None and (existing_dir / "manifest.json").is_file():
        manifest_path = existing_dir / "manifest.json"
        existing = _read_existing_manifest(manifest_path)
        if _plan_payload(
            existing,
            include_name=False,
            include_run_labels=False,
        ) != _plan_payload(
            candidate,
            include_name=False,
            include_run_labels=False,
        ):
            raise ManifestConflictError(
                "Expected an existing sweep id to have identical immutable scientific content. "
                f"Provided path: {manifest_path}."
            )
        return existing, manifest_path

    desired_dir = layout.sweep_dir(candidate["sweep_name"], candidate["sweep_id"])
    if existing_dir is not None and existing_dir != desired_dir:
        raise ManifestConflictError(
            "Expected a partially published sweep-id directory to use the requested display name. "
            f"Provided path: {existing_dir}."
        )
    sweep_dir = existing_dir or desired_dir
    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest = _with_run_paths(candidate, sweep_dir=sweep_dir, layout=layout)
    manifest_path = sweep_dir / "manifest.json"
    resolved_path = sweep_dir / "sweep.resolved.json"
    jobs_path = sweep_dir / "jobs.jsonl"
    resolved_bytes = canonical_json_bytes(spec.to_dict()) + b"\n"
    jobs_bytes = _jobs_bytes(manifest)
    for path, expected in ((resolved_path, resolved_bytes), (jobs_path, jobs_bytes)):
        if path.exists() and path.read_bytes() != expected:
            raise ManifestConflictError(
                f"Expected existing immutable sweep planning file to match. Provided path: {path}."
            )
        if not path.exists():
            atomic_write_bytes(path, expected)
    if manifest_path.exists():
        existing = _read_existing_manifest(manifest_path)
        if existing != manifest:
            raise ManifestConflictError(
                "Expected an existing sweep id to have identical immutable manifest content. "
                f"Provided path: {manifest_path}."
            )
    else:
        atomic_write_json(manifest_path, manifest, canonical=True)
    return load_manifest(manifest_path), manifest_path


def load_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        manifest = validate_manifest(read_json(source))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecValidationError(
            f"Expected {source} to contain a valid canonical manifest. Provided error: {exc}."
        ) from exc
    _validate_companions(source, manifest)
    return manifest


def manifest_entry(manifest: dict[str, Any], job_index: int) -> dict[str, Any]:
    validate_manifest(manifest)
    if isinstance(job_index, bool) or not isinstance(job_index, int):
        raise _error("an integer", job_index, "job_index")
    entries = manifest["entries"]
    if not 0 <= job_index < len(entries):
        raise _error(f"an integer in [0, {len(entries) - 1}]", job_index, "job_index")
    return copy.deepcopy(entries[job_index])
