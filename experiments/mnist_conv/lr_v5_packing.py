"""Deterministic V100-32GB pack planning for the v5 candidate manifest."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import atomic_write_json, read_json


PACK_SCHEMA_VERSION = "mnist-conv-lr-pack-manifest/v1"
MEMORY_PREFLIGHT_SCHEMA_VERSION = "mnist-conv-lr-memory-preflight/v1"
GPU_CAPACITY_MIB = 32_768
HEADROOM_FRACTION = 0.10
PACK_CAPACITY_MIB = GPU_CAPACITY_MIB / (1.0 + HEADROOM_FRACTION)


def _error(expected: str, provided: Any) -> ValueError:
    return ValueError(f"Expected {expected}. Provided value: {provided!r}.")


def _positive_memory(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise _error(f"{path} to be a positive finite MiB measurement", value)
    return float(value)


def _stage_entries(stage_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(stage_manifest, Mapping):
        raise _error("candidate-stage manifest to be a JSON object", stage_manifest)
    if stage_manifest.get("stage_name") != "candidates":
        raise _error(
            "candidate-stage manifest stage_name to be 'candidates'",
            stage_manifest.get("stage_name"),
        )
    study_id = stage_manifest.get("study_id")
    if not isinstance(study_id, str) or not study_id.startswith("lrstudy_"):
        raise _error("candidate-stage manifest to contain a study_id", study_id)
    entries = stage_manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 36:
        raise _error("v5 candidate-stage manifest to contain exactly 36 entries", entries)
    normalized: list[dict[str, Any]] = []
    for expected_index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or entry.get("entry_index") != expected_index:
            raise _error(
                "candidate entry indices to be contiguous from zero",
                entry,
            )
        payload = entry.get("payload")
        if not isinstance(payload, Mapping):
            raise _error(f"entry {expected_index} payload to be an object", payload)
        architecture = payload.get("architecture")
        row_id = payload.get("row_id")
        if architecture not in {"conv1", "conv2"}:
            raise _error(
                f"entry {expected_index} architecture to be conv1 or conv2",
                architecture,
            )
        if not isinstance(row_id, str) or not row_id.startswith(architecture + "_"):
            raise _error(
                f"entry {expected_index} row_id to match {architecture}", row_id
            )
        if payload.get("arm") not in {"strict_equal", "historical_profile"}:
            raise _error(f"entry {expected_index} to declare a v5 arm", payload)
        if payload.get("alpha_role") not in {"lower", "center", "upper"}:
            raise _error(f"entry {expected_index} to declare a v5 alpha role", payload)
        normalized.append(dict(entry))
    return normalized


def validate_memory_preflight(
    memory_preflight: Mapping[str, Any],
    *,
    stage_manifest_sha256: str,
    entry_count: int = 36,
) -> dict[int, float]:
    """Validate one measured per-run peak for every immutable candidate entry."""

    if not isinstance(memory_preflight, Mapping):
        raise _error("memory preflight to be a JSON object", memory_preflight)
    if memory_preflight.get("schema_version") != MEMORY_PREFLIGHT_SCHEMA_VERSION:
        raise _error(
            f"memory preflight schema_version to be {MEMORY_PREFLIGHT_SCHEMA_VERSION!r}",
            memory_preflight.get("schema_version"),
        )
    if memory_preflight.get("stage_manifest_sha256") != stage_manifest_sha256:
        raise _error(
            "memory preflight stage_manifest_sha256 to match the candidate manifest",
            memory_preflight.get("stage_manifest_sha256"),
        )
    raw = memory_preflight.get("peak_gpu_memory_mib_by_entry_index")
    if not isinstance(raw, Mapping):
        raise _error(
            "peak_gpu_memory_mib_by_entry_index to be an object", raw
        )
    expected_keys = {str(index) for index in range(entry_count)}
    if set(raw) != expected_keys:
        raise _error(
            f"memory preflight to contain exactly entry indices 0..{entry_count - 1}",
            sorted(raw),
        )
    return {
        index: _positive_memory(raw[str(index)], f"entry {index} peak GPU memory")
        for index in range(entry_count)
    }


def _pack_architecture(
    entry_indices: Sequence[int], peak_memory_mib: Mapping[int, float]
) -> list[tuple[int, ...]]:
    """Find deterministic 2/3-run packs, preferring three-run packs."""

    ordered = tuple(
        sorted(entry_indices, key=lambda index: (-peak_memory_mib[index], index))
    )

    @lru_cache(maxsize=None)
    def solve(remaining: tuple[int, ...]) -> tuple[tuple[int, ...], ...] | None:
        if not remaining:
            return ()
        if len(remaining) == 1:
            return None
        first = remaining[0]
        tail = remaining[1:]
        for size in (3, 2):
            if len(remaining) < size:
                continue
            for partners in itertools.combinations(tail, size - 1):
                pack = (first, *partners)
                combined = sum(peak_memory_mib[index] for index in pack)
                if combined > PACK_CAPACITY_MIB:
                    continue
                partner_set = set(partners)
                next_remaining = tuple(
                    index for index in tail if index not in partner_set
                )
                suffix = solve(next_remaining)
                if suffix is not None:
                    return (tuple(sorted(pack)), *suffix)
        return None

    solution = solve(ordered)
    if solution is None:
        raise RuntimeError(
            "Expected every same-architecture run to fit a deterministic two- or "
            "three-run V100-32GB pack with 10% headroom. "
            f"Provided entries: {ordered!r}."
        )
    return list(solution)


def build_v5_pack_manifest(
    stage_manifest: Mapping[str, Any],
    *,
    stage_manifest_sha256: str,
    peak_gpu_memory_mib_by_entry_index: Mapping[int, float],
) -> dict[str, Any]:
    """Build the immutable execution-only pack manifest for all 36 runs."""

    entries = _stage_entries(stage_manifest)
    expected_indices = set(range(len(entries)))
    if set(peak_gpu_memory_mib_by_entry_index) != expected_indices:
        raise _error(
            "peak-memory measurements for every candidate entry",
            sorted(peak_gpu_memory_mib_by_entry_index),
        )
    peaks = {
        index: _positive_memory(value, f"entry {index} peak GPU memory")
        for index, value in peak_gpu_memory_mib_by_entry_index.items()
    }
    packs: list[dict[str, Any]] = []
    for architecture in ("conv1", "conv2"):
        indices = [
            entry["entry_index"]
            for entry in entries
            if entry["payload"]["architecture"] == architecture
        ]
        if len(indices) != 18:
            raise _error(
                f"exactly 18 {architecture} candidate entries", len(indices)
            )
        for pack in _pack_architecture(indices, peaks):
            combined = sum(peaks[index] for index in pack)
            packs.append(
                {
                    "pack_index": len(packs),
                    "architecture": architecture,
                    "entry_indices": list(pack),
                    "measured_combined_peak_gpu_memory_mib": combined,
                }
            )
    flattened = [index for pack in packs for index in pack["entry_indices"]]
    if len(flattened) != 36 or set(flattened) != expected_indices:
        raise RuntimeError(
            "Expected the v5 pack plan to cover every candidate exactly once. "
            f"Provided value: {flattened!r}."
        )
    return {
        "schema_version": PACK_SCHEMA_VERSION,
        "study_id": stage_manifest["study_id"],
        "stage_name": "candidates",
        "stage_manifest_sha256": stage_manifest_sha256,
        "gpu_contract": {
            "account": "fmu@v100",
            "constraint": "v100-32g",
            "gpus": 1,
            "capacity_mib": GPU_CAPACITY_MIB,
            "headroom_fraction": HEADROOM_FRACTION,
        },
        "packs": packs,
    }


def create_v5_pack_manifest(
    stage_manifest_path: str | Path,
    memory_preflight_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    stage_path = Path(stage_manifest_path).expanduser().resolve()
    memory_path = Path(memory_preflight_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    stage_bytes = stage_path.read_bytes()
    stage_sha256 = hashlib.sha256(stage_bytes).hexdigest()
    stage = json.loads(stage_bytes)
    preflight = read_json(memory_path)
    peaks = validate_memory_preflight(
        preflight,
        stage_manifest_sha256=stage_sha256,
        entry_count=36,
    )
    manifest = build_v5_pack_manifest(
        stage,
        stage_manifest_sha256=stage_sha256,
        peak_gpu_memory_mib_by_entry_index=peaks,
    )
    if output.exists():
        if read_json(output) != manifest:
            raise RuntimeError(
                "Expected an existing content-bound pack manifest to be identical. "
                f"Provided value: {output}."
            )
    else:
        atomic_write_json(output, manifest, canonical=True)
    return manifest


__all__ = [
    "GPU_CAPACITY_MIB",
    "HEADROOM_FRACTION",
    "MEMORY_PREFLIGHT_SCHEMA_VERSION",
    "PACK_SCHEMA_VERSION",
    "build_v5_pack_manifest",
    "create_v5_pack_manifest",
    "validate_memory_preflight",
]
