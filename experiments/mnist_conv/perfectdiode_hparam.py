"""Immutable host-sharded orchestration for the perfect-diode LR screen.

This module intentionally lives beside, rather than inside, the historical
``lr_study`` implementation.  The separate namespace keeps the v1--v7
hard-sigmoid studies byte-for-byte compatible while providing a small
orchestration surface for the ordinary-MNIST perfect-diode diagnostic.

The screen is executed in two immutable shards:

* ``akib`` owns every Conv1 surface;
* ``trex`` owns every Conv2 surface.

The shards stop after ``select_final``.  They are then copied into a new,
hash-verified aggregate.  ``long_confirm`` and ``finalize`` run sequentially
from that aggregate on Trex.  No two hosts ever write the same directory.
"""

from __future__ import annotations

import copy
import math
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .identity import (
    canonical_json_bytes,
    code_provenance,
    normalize_code_provenance,
    sha256_file,
    sha256_json,
)
from .io import atomic_write_bytes, atomic_write_csv, atomic_write_json, read_json
from .lr_artifacts import (
    artifact_records,
    build_stage_manifest,
    entry_is_complete,
    load_stage_manifest,
    publish_entry_completion,
    publish_stage_completion,
    publish_stage_manifest,
    stage_entry,
    validate_stage_completion,
)


PD_STUDY_SCHEMA_VERSION = "mnist-conv-perfectdiode-hparam-study/v1"
PD_RUN_SCHEMA_VERSION = "mnist-conv-perfectdiode-hparam-run/v1"
PD_SHARD_SCHEMA_VERSION = "mnist-conv-perfectdiode-hparam-shard/v1"
PD_MERGE_SCHEMA_VERSION = "mnist-conv-perfectdiode-hparam-merge/v1"
PD_CELL_ID_SCHEMA_VERSION = "mnist-conv-perfectdiode-rho-cell/v1"

PUBLIC_STAGE_SEQUENCE = (
    "audit",
    "assets",
    "fixed_tk_gradient_security",
    "optimizer_probe",
    "rho_canary_core",
    "rho_core_candidates",
    "select_core",
    "rho_canary_extension",
    "rho_extension_candidates",
    "select_final",
    "long_confirm",
    "finalize",
)
SCREEN_STAGE_SEQUENCE = PUBLIC_STAGE_SEQUENCE[:10]
AGGREGATE_STAGE_SEQUENCE = PUBLIC_STAGE_SEQUENCE[10:]

HOST_ARCHITECTURES = {"akib": "conv1", "trex": "conv2"}
ARCHITECTURE_HOSTS = {value: key for key, value in HOST_ARCHITECTURES.items()}
OPTIMIZER_ORDER = ("sgd", "adam")
SCHEME_ORDER = ("baseline", "ours", "legacy")

CANARY_STEPS = 640
CANDIDATE_STEPS = 10_314
LONG_STEPS = {"conv1": 34_380, "conv2": 103_140}

_RUNTIME_OUTPUTS = {
    "assets": (
        "initialization.json",
        "initialization.pt",
        "result.json",
        "split_indices.json",
        "split_provenance.json",
    ),
    "fixed_tk_gradient_security": (
        "parameter_diagnostics.csv",
        "result.json",
    ),
    "optimizer_probe": (
        "minibatches.json",
        "parameter_diagnostics.csv",
        "result.json",
    ),
    "rho_canary_core": ("result.json", "step_log.csv"),
    "rho_canary_extension": ("result.json", "step_log.csv"),
    "rho_core_candidates": (
        "best_validation.pt",
        "final.pt",
        "result.json",
        "run_spec.json",
        "step_log.csv",
        "validation.json",
    ),
    "rho_extension_candidates": (
        "best_validation.pt",
        "final.pt",
        "result.json",
        "run_spec.json",
        "step_log.csv",
        "validation.json",
    ),
    "long_confirm": (
        "best_validation.pt",
        "final.pt",
        "result.json",
        "run_spec.json",
        "step_log.csv",
        "validation.json",
    ),
}
_PREFLIGHT_OUTPUTS = ("result.json", "step_log.csv")


class PerfectDiodeOrchestrationError(ValueError):
    """A perfect-diode orchestration artifact violates its contract."""


def _error(expected: str, provided: Any, path: str) -> PerfectDiodeOrchestrationError:
    return PerfectDiodeOrchestrationError(
        f"Expected {path} to be {expected}. Provided value: {provided!r}."
    )


def _spec_data(spec: Any) -> dict[str, Any]:
    value = spec.data if hasattr(spec, "data") else spec
    if not isinstance(value, Mapping):
        raise _error("a study spec or mapping", value, "spec")
    result = copy.deepcopy(dict(value))
    if result.get("schema_version") != PD_STUDY_SCHEMA_VERSION:
        raise _error(
            f"exactly {PD_STUDY_SCHEMA_VERSION!r}",
            result.get("schema_version"),
            "spec.schema_version",
        )
    return result


def _spec_id(spec: Any) -> str:
    if hasattr(spec, "study_id"):
        value = spec.study_id
    else:
        from .perfectdiode_hparam_spec import PerfectDiodeHparamStudySpec

        value = PerfectDiodeHparamStudySpec.from_dict(
            _spec_data(spec)
        ).study_id
    if (
        not isinstance(value, str)
        or not value.startswith("lrstudy_")
        or len(value) != 72
        or any(character not in "0123456789abcdef" for character in value[8:])
    ):
        raise _error("a canonical lrstudy_ SHA-256 id", value, "spec.study_id")
    return value


def _require_exact_clean_staged_source(
    provenance: Mapping[str, Any],
    staged_source: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Require a clean Git checkout at the exact staged commit and content."""

    source = normalize_code_provenance(provenance)
    if not isinstance(staged_source, Mapping):
        raise _error("an object", staged_source, f"{context}.staged_source")
    commit = staged_source.get("commit")
    archive_sha256 = staged_source.get("archive_sha256")
    fingerprint = staged_source.get("effective_code_fingerprint")
    if (
        not isinstance(commit, str)
        or len(commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise _error(
            "a lowercase 40- or 64-character commit id",
            commit,
            f"{context}.staged_source.commit",
        )
    if (
        not isinstance(archive_sha256, str)
        or len(archive_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in archive_sha256
        )
    ):
        raise _error(
            "a lowercase SHA-256 digest",
            archive_sha256,
            f"{context}.staged_source.archive_sha256",
        )
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(
            character not in "0123456789abcdef"
            for character in fingerprint
        )
    ):
        raise _error(
            "a lowercase SHA-256 digest",
            fingerprint,
            f"{context}.staged_source.effective_code_fingerprint",
        )
    expected = {
        "git_revision": commit,
        "dirty_source_digest": None,
        "effective_code_fingerprint": fingerprint,
    }
    if source != expected:
        raise RuntimeError(
            f"Expected {context} to use a clean Git checkout at the exact "
            "staged commit and effective source fingerprint. "
            f"Provided staged={dict(staged_source)!r}, current={source!r}."
        )
    return source


def _rows(spec: Any) -> list[dict[str, Any]]:
    raw = spec.rows if hasattr(spec, "rows") else _spec_data(spec).get("rows")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise _error("a non-empty row sequence", raw, "spec.rows")
    rows = [copy.deepcopy(dict(row)) for row in raw]
    if len(rows) != 6:
        raise _error("exactly six Conv1/Conv2 rows", len(rows), "spec.rows")
    expected = [
        (architecture, scheme)
        for architecture in ("conv1", "conv2")
        for scheme in SCHEME_ORDER
    ]
    observed = [(row.get("architecture"), row.get("scheme")) for row in rows]
    if observed != expected:
        raise _error(
            f"rows in architecture/scheme order {expected!r}",
            observed,
            "spec.rows",
        )
    for index, row in enumerate(rows):
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id:
            raise _error("a non-empty string", row_id, f"spec.rows[{index}].row_id")
    return rows


def surface_id(row: Mapping[str, Any], optimizer: str) -> str:
    """Return the stable row-by-optimizer surface identifier."""

    if optimizer not in OPTIMIZER_ORDER:
        raise _error(f"one of {OPTIMIZER_ORDER!r}", optimizer, "optimizer")
    row_id = row.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise _error("a non-empty row id", row_id, "row.row_id")
    return f"{row_id}--{optimizer}"


def rho_cell_id(
    study_id: str,
    surface: str,
    rho_conv: float,
    rho_dense: float,
) -> str:
    """Content-address one target pair within one immutable surface."""

    for name, value in (("rho_conv", rho_conv), ("rho_dense", rho_dense)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise _error("a finite positive number", value, name)
    return "pdcell_" + sha256_json(
        {
            "schema_version": PD_CELL_ID_SCHEMA_VERSION,
            "study_id": study_id,
            "surface_id": surface,
            "rho_conv": float(rho_conv),
            "rho_dense": float(rho_dense),
        }
    )


def host_rows(spec: Any, host: str) -> list[dict[str, Any]]:
    if host not in HOST_ARCHITECTURES:
        raise _error(f"one of {tuple(HOST_ARCHITECTURES)!r}", host, "host")
    architecture = HOST_ARCHITECTURES[host]
    result = [row for row in _rows(spec) if row["architecture"] == architecture]
    if len(result) != 3:
        raise _error("exactly three routed rows", len(result), f"rows[{host}]")
    return result


def host_surfaces(spec: Any, host: str) -> list[dict[str, Any]]:
    return [
        {
            "surface_id": surface_id(row, optimizer),
            "row": row,
            "optimizer": optimizer,
            "host": host,
            "architecture": row["architecture"],
        }
        for row in host_rows(spec, host)
        for optimizer in OPTIMIZER_ORDER
    ]


def _study_root(results_root: str | Path, spec: Any) -> Path:
    data = _spec_data(spec)
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise _error("a non-empty string", name, "spec.name")
    return (
        Path(results_root).expanduser().resolve()
        / "perfectdiode_hparam_studies"
        / f"{name.strip()}--{_spec_id(spec)}"
    )


def _publish_identical(path: Path, value: Mapping[str, Any]) -> Path:
    if path.exists():
        if read_json(path) != dict(value):
            raise RuntimeError(
                "Expected the immutable artifact to contain identical content. "
                f"Provided value: {path}."
            )
        return path
    return atomic_write_json(path, dict(value), canonical=True)


def create_host_shard(
    spec: Any,
    results_root: str | Path,
    host: str,
    *,
    provenance: Mapping[str, Any] | None = None,
    source_commit: str | None = None,
    source_archive_sha256: str | None = None,
    environment: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Create one immutable architecture shard without launching any work."""

    data = _spec_data(spec)
    study_id = _spec_id(spec)
    source = normalize_code_provenance(provenance or code_provenance())
    resolved_commit = source_commit
    if (
        not isinstance(resolved_commit, str)
        or len(resolved_commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in resolved_commit)
    ):
        raise _error(
            "a lowercase 40- or 64-character commit id",
            resolved_commit,
            "source_commit",
        )
    if (
        not isinstance(source_archive_sha256, str)
        or len(source_archive_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_archive_sha256
        )
    ):
        raise _error(
            "a lowercase SHA-256 digest",
            source_archive_sha256,
            "source_archive_sha256",
        )
    staged_source = {
        "commit": resolved_commit,
        "archive_sha256": source_archive_sha256,
        "effective_code_fingerprint": source["effective_code_fingerprint"],
    }
    _require_exact_clean_staged_source(
        source, staged_source, context="host-shard creation"
    )
    environment_value = dict(environment or environment_fingerprint())
    root = _study_root(results_root, spec)
    shard = root / "shards" / host
    shard.mkdir(parents=True, exist_ok=True)
    _publish_identical(shard / "study.resolved.json", data)
    descriptor = {
        "schema_version": PD_SHARD_SCHEMA_VERSION,
        "study_id": study_id,
        "host": host,
        "architecture": HOST_ARCHITECTURES.get(host),
        "surface_ids": [
            item["surface_id"] for item in host_surfaces(spec, host)
        ],
        "code_provenance": source,
        "staged_source": staged_source,
        "environment": environment_value,
        "environment_sha256": sha256_json(environment_value),
        "config_sha256": sha256_json(data),
        "resolved_config_file_sha256": sha256_file(
            shard / "study.resolved.json"
        ),
        "dataset_contract_sha256": sha256_json(data.get("dataset")),
        "launched_jobs": 0,
    }
    if descriptor["architecture"] is None:
        raise _error(f"one of {tuple(HOST_ARCHITECTURES)!r}", host, "host")
    descriptor["shard_id"] = "pdshard_" + sha256_json(descriptor)
    _publish_identical(shard / "shard.resolved.json", descriptor)
    return root, shard


def environment_fingerprint() -> dict[str, Any]:
    """Return a deterministic runtime-environment descriptor for provenance."""

    result: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable_name": Path(sys.executable).name,
    }
    try:
        import torch
    except ImportError:
        result["torch"] = None
        result["cuda_runtime"] = None
    else:
        result["torch"] = torch.__version__
        result["cuda_runtime"] = torch.version.cuda
    return result


def _load_spec(path: Path) -> Any:
    from .perfectdiode_hparam_spec import PerfectDiodeHparamStudySpec

    return PerfectDiodeHparamStudySpec.from_path(path)


def load_host_shard(shard_dir: str | Path) -> tuple[Path, Any, dict[str, Any]]:
    root = Path(shard_dir).expanduser().resolve()
    spec = _load_spec(root / "study.resolved.json")
    descriptor = read_json(root / "shard.resolved.json")
    expected_host = descriptor.get("host")
    if expected_host not in HOST_ARCHITECTURES:
        raise _error(
            f"one of {tuple(HOST_ARCHITECTURES)!r}",
            expected_host,
            "shard.host",
        )
    expected_surfaces = [
        item["surface_id"] for item in host_surfaces(spec, expected_host)
    ]
    if descriptor.get("study_id") != _spec_id(spec):
        raise _error(_spec_id(spec), descriptor.get("study_id"), "shard.study_id")
    if descriptor.get("architecture") != HOST_ARCHITECTURES[expected_host]:
        raise _error(
            HOST_ARCHITECTURES[expected_host],
            descriptor.get("architecture"),
            "shard.architecture",
        )
    if descriptor.get("surface_ids") != expected_surfaces:
        raise _error(expected_surfaces, descriptor.get("surface_ids"), "shard.surface_ids")
    _require_exact_clean_staged_source(
        descriptor.get("code_provenance"),
        descriptor.get("staged_source"),
        context="immutable shard descriptor",
    )
    if descriptor.get("config_sha256") != spec.config_sha256:
        raise _error(
            spec.config_sha256,
            descriptor.get("config_sha256"),
            "shard.config_sha256",
        )
    resolved_config_sha256 = sha256_file(root / "study.resolved.json")
    if (
        descriptor.get("resolved_config_file_sha256")
        != resolved_config_sha256
    ):
        raise _error(
            resolved_config_sha256,
            descriptor.get("resolved_config_file_sha256"),
            "shard.resolved_config_file_sha256",
        )
    return root, spec, descriptor


def _entry_base(stage: str, entry_id: str) -> str:
    return f"stages/{stage}/entries/{entry_id}"


def _entry_outputs(stage: str, entry_id: str, *, no_op: bool = False) -> list[str]:
    base = _entry_base(stage, entry_id)
    if no_op or stage == "audit":
        names = ("result.json",)
    elif stage in _RUNTIME_OUTPUTS:
        names = _RUNTIME_OUTPUTS[stage]
    elif stage in {"select_core", "select_final"}:
        names = ("result.csv", "result.json")
    elif stage == "finalize":
        names = ("result.csv", "result.json")
    else:
        raise _error(f"one of {PUBLIC_STAGE_SEQUENCE!r}", stage, "stage")
    return sorted(f"{base}/{name}" for name in names)


def _completion_path(stage: str, entry_id: str) -> str:
    return f"{_entry_base(stage, entry_id)}/complete.json"


def _stage_manifest_path(root: Path, stage: str) -> Path:
    return root / "stages" / stage / "manifest.json"


def _stage_result(root: Path, stage: str, entry_id: str) -> dict[str, Any]:
    return read_json(root / _entry_base(stage, entry_id) / "result.json")


def _status_passed(value: Mapping[str, Any]) -> bool:
    return value.get("status") in {
        "complete",
        "passed",
        "resolved",
        "safe",
        "safety_clean",
        "selected",
    }


def _surface_payload(surface: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "surface_id": surface["surface_id"],
        "row": copy.deepcopy(surface["row"]),
        "row_id": surface["row"]["row_id"],
        "optimizer": surface["optimizer"],
        "host": surface["host"],
        "architecture": surface["architecture"],
        "asset_entry_id": f"{surface['architecture']}-assets",
        "probe_entry_id": surface["surface_id"],
    }


def _no_op_entry(
    stage: str,
    entry_id: str,
    *,
    index: int,
    payload: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    value = {**copy.deepcopy(dict(payload)), "no_op": True, "reason": reason}
    return stage_entry(
        index,
        entry_id,
        completion_path=_completion_path(stage, entry_id),
        outputs=_entry_outputs(stage, entry_id, no_op=True),
        payload=value,
    )


def _normal_entry(
    stage: str,
    entry_id: str,
    *,
    index: int,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return stage_entry(
        index,
        entry_id,
        completion_path=_completion_path(stage, entry_id),
        outputs=_entry_outputs(stage, entry_id),
        payload=payload,
    )


def _rho_search(data: Mapping[str, Any]) -> dict[str, Any]:
    value = data.get("rho_search", {})
    if not isinstance(value, Mapping):
        raise _error("an object", value, "study.rho_search")
    center = value.get(
        "initial_center",
        value.get("center", value.get("center_canary", {})),
    )
    if isinstance(center, Mapping):
        conv = center.get("rho_conv", center.get("initial_rho_conv", 3e-3))
        dense = center.get(
            "rho_dense", center.get("initial_rho_dense", 1e-2)
        )
    else:
        conv, dense = 3e-3, 1e-2
        center = {}
    return {
        "rho_conv": float(conv),
        "rho_dense": float(dense),
        "factor": float(
            value.get("factor", center.get("failure_scale_divisor", 3.0))
        ),
        "maximum_center_attempts": int(
            value.get(
                "maximum_center_attempts",
                center.get("maximum_attempts", 6),
            )
        ),
        "maximum_cells_per_surface": int(
            value.get(
                "maximum_cells_per_surface",
                value.get("expansion", {}).get(
                    "maximum_cells_per_surface", 16
                ),
            )
        ),
    }


def _core_cells(
    study_id: str,
    surface: str,
    canary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cells_raw = canary.get("cells")
    if not isinstance(cells_raw, Sequence) or isinstance(cells_raw, (str, bytes)):
        if canary.get("status") in {"unresolved", "failed", "no_safe_center"}:
            return []
        raise _error("a sequence", cells_raw, "core_canary.cells")
    cells: list[dict[str, Any]] = []
    seen_pairs: set[tuple[float, float]] = set()
    for index, raw in enumerate(cells_raw):
        if not isinstance(raw, Mapping):
            raise _error("an object", raw, f"core_canary.cells[{index}]")
        conv = float(raw["rho_conv"])
        dense = float(raw["rho_dense"])
        pair = (conv, dense)
        if pair in seen_pairs:
            raise _error("unique rho pairs", pair, "core_canary.cells")
        seen_pairs.add(pair)
        expected_id = rho_cell_id(study_id, surface, conv, dense)
        provided_id = raw.get("cell_id", expected_id)
        if provided_id != expected_id:
            raise _error(expected_id, provided_id, f"core_canary.cells[{index}].cell_id")
        cells.append({**copy.deepcopy(dict(raw)), "cell_id": expected_id})
    if cells and len(cells) != 9:
        raise _error("exactly nine core cells", len(cells), "core_canary.cells")
    safe_center = canary.get("safe_center")
    if cells and isinstance(safe_center, Mapping):
        center_id = rho_cell_id(
            study_id,
            surface,
            float(safe_center["rho_conv"]),
            float(safe_center["rho_dense"]),
        )
        if center_id not in {cell["cell_id"] for cell in cells}:
            raise _error(
                "the safe-center cell to occur in the core grid",
                center_id,
                "core_canary.safe_center",
            )
        if canary.get("center_reused_as_core") is not True:
            raise _error(
                "exactly true",
                canary.get("center_reused_as_core"),
                "core_canary.center_reused_as_core",
            )
    return cells


def _cell_is_clean(cell: Mapping[str, Any]) -> bool:
    explicit = cell.get("safety_clean")
    if isinstance(explicit, bool):
        return explicit
    return cell.get("status") in {"passed", "safe", "safety_clean", "complete"}


def _selection_expansion(value: Mapping[str, Any]) -> dict[str, str]:
    raw = value.get("expansion")
    if raw is None:
        axes = value.get("expansion_axes", [])
        if isinstance(axes, Sequence) and not isinstance(axes, (str, bytes)):
            raw = {str(axis): "upper" for axis in axes}
        else:
            raw = {}
    if not isinstance(raw, Mapping):
        raise _error("an axis-to-side object", raw, "selection.expansion")
    result = {str(axis): str(side) for axis, side in raw.items()}
    if not set(result) <= {"rho_conv", "rho_dense"}:
        raise _error(
            "only rho_conv/rho_dense axes",
            sorted(result),
            "selection.expansion",
        )
    if any(side not in {"lower", "upper"} for side in result.values()):
        raise _error(
            "lower or upper sides",
            result,
            "selection.expansion",
        )
    return result


def _expanded_cells(
    study_id: str,
    surface: str,
    core_cells: Sequence[Mapping[str, Any]],
    expansion: Mapping[str, str],
) -> list[dict[str, Any]]:
    conv_values = sorted({float(cell["rho_conv"]) for cell in core_cells})
    dense_values = sorted({float(cell["rho_dense"]) for cell in core_cells})
    if len(conv_values) != 3 or len(dense_values) != 3:
        raise _error("a complete 3x3 core grid", (conv_values, dense_values), "core_grid")
    if "rho_conv" in expansion:
        conv_values.append(
            min(conv_values) / 3.0
            if expansion["rho_conv"] == "lower"
            else max(conv_values) * 3.0
        )
    if "rho_dense" in expansion:
        dense_values.append(
            min(dense_values) / 3.0
            if expansion["rho_dense"] == "lower"
            else max(dense_values) * 3.0
        )
    conv_values = sorted(set(conv_values))
    dense_values = sorted(set(dense_values))
    existing = {
        (float(cell["rho_conv"]), float(cell["rho_dense"])) for cell in core_cells
    }
    result = []
    for rho_conv in conv_values:
        for rho_dense in dense_values:
            if (rho_conv, rho_dense) in existing:
                continue
            result.append(
                {
                    "rho_conv": rho_conv,
                    "rho_dense": rho_dense,
                    "cell_id": rho_cell_id(
                        study_id, surface, rho_conv, rho_dense
                    ),
                }
            )
    if len(existing) + len(result) > 16 or len(result) > 7:
        raise _error(
            "one symmetric expansion capped at 16 total cells",
            {"existing": len(existing), "new": len(result)},
            "expansion",
        )
    return result


def _entries_for_screen_stage(
    spec: Any,
    root: Path,
    descriptor: Mapping[str, Any],
    stage: str,
) -> list[dict[str, Any]]:
    host = descriptor["host"]
    surfaces = host_surfaces(spec, host)
    study_id = _spec_id(spec)

    def resolved_surface_payload(surface: Mapping[str, Any]) -> dict[str, Any]:
        staged = descriptor["staged_source"]
        return {
            **_surface_payload(surface),
            "source_commit": staged["commit"],
            "source_archive_sha256": staged["archive_sha256"],
            "environment_sha256": descriptor["environment_sha256"],
            "config_sha256": descriptor["config_sha256"],
        }
    if stage == "audit":
        entry_id = f"audit-{host}"
        return [
            _normal_entry(
                stage,
                entry_id,
                index=0,
                payload={
                    "host": host,
                    "architecture": descriptor["architecture"],
                    "study_id": study_id,
                    "surface_ids": [item["surface_id"] for item in surfaces],
                    "expected_launched_jobs_before_execution": 0,
                    "staged_source": descriptor["staged_source"],
                    "environment": descriptor["environment"],
                    "environment_sha256": descriptor["environment_sha256"],
                    "config_sha256": descriptor["config_sha256"],
                    "dataset_contract_sha256": descriptor[
                        "dataset_contract_sha256"
                    ],
                },
            )
        ]
    if stage == "assets":
        entry_id = f"{descriptor['architecture']}-assets"
        return [
            _normal_entry(
                stage,
                entry_id,
                index=0,
                payload={
                    "host": host,
                    "architecture": descriptor["architecture"],
                    "model_seed": 0,
                    "train_loader_seed": 0,
                    "official_test_read": False,
                },
            )
        ]
    if stage == "fixed_tk_gradient_security":
        return [
            _normal_entry(
                stage,
                row["row_id"],
                index=index,
                payload={
                    "row": row,
                    "row_id": row["row_id"],
                    "host": host,
                    "architecture": row["architecture"],
                    "asset_entry_id": f"{row['architecture']}-assets",
                    "source_commit": descriptor["staged_source"]["commit"],
                    "source_archive_sha256": descriptor["staged_source"][
                        "archive_sha256"
                    ],
                    "environment_sha256": descriptor["environment_sha256"],
                    "reference_inference_iterations": 64,
                    "reference_training_iterations": 64,
                    "norm_delta_max": 0.10,
                    "zero_fraction_delta_max": 0.02,
                    "cosine_min": 0.90,
                },
            )
            for index, row in enumerate(host_rows(spec, host))
        ]
    if stage == "optimizer_probe":
        entries = []
        for surface in surfaces:
            payload = resolved_surface_payload(surface)
            security = _stage_result(
                root,
                "fixed_tk_gradient_security",
                surface["row"]["row_id"],
            )
            if not _status_passed(security):
                entries.append(
                    _no_op_entry(
                        stage,
                        f"{surface['surface_id']}--security-unresolved",
                        index=len(entries),
                        payload=payload,
                        reason="unresolved_fixed_tk_gradient_mismatch",
                    )
                )
            else:
                entries.append(
                    _normal_entry(
                        stage,
                        surface["surface_id"],
                        index=len(entries),
                        payload={
                            **payload,
                            "initial_minibatches": 32,
                            "adaptive_minibatches": [64, 128],
                            "split_half_relative_tolerance": 0.10,
                        },
                    )
                )
        return entries
    if stage == "rho_canary_core":
        search = _rho_search(_spec_data(spec))
        entries = []
        for surface in surfaces:
            payload = resolved_surface_payload(surface)
            probe_id = surface["surface_id"]
            probe_path = root / _entry_base("optimizer_probe", probe_id) / "result.json"
            if not probe_path.is_file():
                entries.append(
                    _no_op_entry(
                        stage,
                        f"{probe_id}--probe-unresolved",
                        index=len(entries),
                        payload=payload,
                        reason="unresolved_probe",
                    )
                )
                continue
            probe = read_json(probe_path)
            if not _status_passed(probe):
                entries.append(
                    _no_op_entry(
                        stage,
                        f"{probe_id}--probe-unresolved",
                        index=len(entries),
                        payload=payload,
                        reason="unresolved_probe",
                    )
                )
                continue
            entries.append(
                _normal_entry(
                    stage,
                    surface["surface_id"],
                    index=len(entries),
                    payload={
                        **payload,
                        "initial_center": {
                            "rho_conv": search["rho_conv"],
                            "rho_dense": search["rho_dense"],
                        },
                        "factor": search["factor"],
                        "maximum_center_attempts": search[
                            "maximum_center_attempts"
                        ],
                        "canary_steps": CANARY_STEPS,
                        "probe_result": (
                            f"{_entry_base('optimizer_probe', probe_id)}/result.json"
                        ),
                        "cell_identity_schema": PD_CELL_ID_SCHEMA_VERSION,
                        "reuse_center_canary_as_identical_core_cell": True,
                    },
                )
            )
        return entries
    if stage == "rho_core_candidates":
        entries = []
        for surface in surfaces:
            payload = resolved_surface_payload(surface)
            canary_path = (
                root
                / _entry_base("rho_canary_core", surface["surface_id"])
                / "result.json"
            )
            if not canary_path.is_file():
                entries.append(
                    _no_op_entry(
                        stage,
                        f"{surface['surface_id']}--no-safe-core",
                        index=len(entries),
                        payload=payload,
                        reason="unresolved_no_safe_center",
                    )
                )
                continue
            canary = read_json(canary_path)
            cells = _core_cells(study_id, surface["surface_id"], canary)
            promoted = [cell for cell in cells if _cell_is_clean(cell)]
            if not promoted:
                entries.append(
                    _no_op_entry(
                        stage,
                        f"{surface['surface_id']}--no-promoted-core",
                        index=len(entries),
                        payload=payload,
                        reason="no_safety_clean_core_cell",
                    )
                )
                continue
            for cell in promoted:
                entries.append(
                    _normal_entry(
                        stage,
                        f"{surface['surface_id']}--{cell['cell_id']}",
                        index=len(entries),
                        payload={
                            **payload,
                            "cell": cell,
                            "cell_id": cell["cell_id"],
                            "rho_conv": float(cell["rho_conv"]),
                            "rho_dense": float(cell["rho_dense"]),
                            "candidate_steps": CANDIDATE_STEPS,
                            "candidate_epochs": 3,
                            "official_test_read": False,
                            "canary_result": (
                                f"{_entry_base('rho_canary_core', surface['surface_id'])}"
                                "/result.json"
                            ),
                        },
                    )
                )
        return entries
    if stage in {"select_core", "select_final"}:
        return [
            _normal_entry(
                stage,
                surface["surface_id"],
                index=index,
                payload={
                    **resolved_surface_payload(surface),
                    "minimum_accuracy": 0.90,
                    "inclusive_accuracy_gate": True,
                    "plateau_relative_tolerance": 0.02,
                    "maximum_cells_per_surface": 16,
                    "selection_scope": (
                        "core_only" if stage == "select_core" else "core_and_extension"
                    ),
                },
            )
            for index, surface in enumerate(surfaces)
        ]
    if stage == "rho_canary_extension":
        entries = []
        for surface in surfaces:
            payload = resolved_surface_payload(surface)
            selection = _stage_result(root, "select_core", surface["surface_id"])
            expansion = _selection_expansion(selection)
            if not expansion:
                entries.append(
                    _no_op_entry(
                        stage,
                        f"{surface['surface_id']}--no-expansion",
                        index=len(entries),
                        payload=payload,
                        reason=(
                            selection.get("reason")
                            or "no_surface_expansion_required"
                        ),
                    )
                )
                continue
            core = _stage_result(
                root, "rho_canary_core", surface["surface_id"]
            )
            cells = _expanded_cells(
                study_id,
                surface["surface_id"],
                _core_cells(study_id, surface["surface_id"], core),
                expansion,
            )
            for cell in cells:
                entries.append(
                    _normal_entry(
                        stage,
                        f"{surface['surface_id']}--{cell['cell_id']}",
                        index=len(entries),
                        payload={
                            **payload,
                            "cell": cell,
                            "cell_id": cell["cell_id"],
                            "rho_conv": cell["rho_conv"],
                            "rho_dense": cell["rho_dense"],
                            "expansion": expansion,
                            "canary_steps": CANARY_STEPS,
                        },
                    )
                )
        return entries
    if stage == "rho_extension_candidates":
        canary_manifest = load_stage_manifest(
            _stage_manifest_path(root, "rho_canary_extension"),
            study_dir=root,
            expected_study_id=study_id,
        )
        entries = []
        by_surface = {item["surface_id"]: [] for item in surfaces}
        for canary_entry in canary_manifest["entries"]:
            payload = canary_entry["payload"]
            surface = payload.get("surface_id")
            if surface in by_surface and payload.get("no_op") is not True:
                result = _stage_result(
                    root, "rho_canary_extension", canary_entry["entry_id"]
                )
                result_cells = result.get("cells")
                if (
                    not isinstance(result_cells, list)
                    or len(result_cells) != 1
                ):
                    raise _error(
                        "exactly one extension-canary cell result",
                        result_cells,
                        "extension_canary.cells",
                    )
                if _cell_is_clean(result_cells[0]):
                    by_surface[surface].append((payload, result_cells[0]))
        surfaces_by_id = {item["surface_id"]: item for item in surfaces}
        for surface_id_value, promoted in by_surface.items():
            payload = resolved_surface_payload(
                surfaces_by_id[surface_id_value]
            )
            if not promoted:
                entries.append(
                    _no_op_entry(
                        stage,
                        f"{surface_id_value}--no-promoted-extension",
                        index=len(entries),
                        payload=payload,
                        reason="no_safety_clean_extension_cell",
                    )
                )
                continue
            for canary_payload, result in promoted:
                cell = copy.deepcopy(canary_payload["cell"])
                entries.append(
                    _normal_entry(
                        stage,
                        f"{surface_id_value}--{cell['cell_id']}",
                        index=len(entries),
                        payload={
                            **payload,
                            "cell": cell,
                            "cell_id": cell["cell_id"],
                            "rho_conv": cell["rho_conv"],
                            "rho_dense": cell["rho_dense"],
                            "candidate_steps": CANDIDATE_STEPS,
                            "candidate_epochs": 3,
                            "official_test_read": False,
                            "canary_result": (
                                f"{_entry_base('rho_canary_extension', canary_payload['surface_id'] + '--' + cell['cell_id'])}"
                                "/result.json"
                            ),
                        },
                    )
                )
        return entries
    raise _error(f"one of {SCREEN_STAGE_SEQUENCE!r}", stage, "stage")


def _completed_stage_upstreams(root: Path, stage: str) -> list[Path]:
    manifest_path = _stage_manifest_path(root, stage)
    validate_stage_completion(study_dir=root, manifest_path=manifest_path)
    manifest = load_stage_manifest(manifest_path, study_dir=root)
    paths = [manifest_path, manifest_path.parent / "complete.json"]
    for entry in manifest["entries"]:
        paths.append(root / entry["completion_path"])
        paths.extend(root / value for value in entry["outputs"])
    return paths


def _screen_upstreams(root: Path, stage: str) -> list[Path]:
    paths = [root / "shard.resolved.json"]
    index = SCREEN_STAGE_SEQUENCE.index(stage)
    if index:
        paths.extend(_completed_stage_upstreams(root, SCREEN_STAGE_SEQUENCE[index - 1]))
    if stage == "rho_canary_core":
        receipt = root / "preflight" / "adam-center-ours" / "receipt.json"
        if not receipt.is_file():
            raise RuntimeError(
                "Expected the representative Adam center preflight receipt "
                "before publishing rho_canary_core."
            )
        value = read_json(receipt)
        if value.get("execution_passed") is not True:
            raise RuntimeError(
                "Expected the representative Adam center preflight execution "
                f"gate to pass. Provided value: {value!r}."
            )
        paths.append(receipt)
        paths.extend(root / record["path"] for record in value["outputs"])
    return paths


def publish_screen_stage_manifest(
    shard_dir: str | Path,
    stage: str,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Publish one host-stage manifest after its predecessor is complete."""

    if stage not in SCREEN_STAGE_SEQUENCE:
        raise _error(f"one of {SCREEN_STAGE_SEQUENCE!r}", stage, "stage")
    root, spec, descriptor = load_host_shard(shard_dir)
    source = _require_exact_clean_staged_source(
        provenance or code_provenance(),
        descriptor["staged_source"],
        context="screen-stage planning",
    )
    entries = _entries_for_screen_stage(spec, root, descriptor, stage)
    manifest = build_stage_manifest(
        study_dir=root,
        study_id=_spec_id(spec),
        study_config_path=root / "study.resolved.json",
        code_provenance=source,
        stage_name=stage,
        entries=entries,
        upstream_paths=_screen_upstreams(root, stage),
    )
    path = _stage_manifest_path(root, stage)
    publish_stage_manifest(path, manifest, study_dir=root)
    return manifest, path


def _candidate_results(
    root: Path,
    surface: str,
    *,
    include_extension: bool,
) -> list[dict[str, Any]]:
    # First declare every cell from canary artifacts.  A failed canary remains
    # an explicit inadmissible record; otherwise selection would infer false
    # grid edges from only the promoted cells.
    declared: dict[tuple[float, float], dict[str, Any]] = {}
    core_canary_path = (
        root / _entry_base("rho_canary_core", surface) / "result.json"
    )
    if core_canary_path.is_file():
        study_id = read_json(root / "shard.resolved.json")["study_id"]
        for cell in _core_cells(study_id, surface, read_json(core_canary_path)):
            pair = (float(cell["rho_conv"]), float(cell["rho_dense"]))
            declared[pair] = {
                "candidate_id": cell["cell_id"],
                "cell_id": cell["cell_id"],
                "rho_conv": pair[0],
                "rho_dense": pair[1],
                "admissible": False,
                "completed_steps": 0,
                "final_validation_accuracy": None,
                "final_validation_loss": None,
                "median_projection_efficiency": None,
                "inadmissible_reason": cell.get(
                    "safety_failure",
                    cell.get("reason", "core_canary_failed"),
                ),
                "result_path": None,
            }
    if include_extension:
        canary_manifest_path = _stage_manifest_path(
            root, "rho_canary_extension"
        )
        if canary_manifest_path.is_file():
            manifest = load_stage_manifest(canary_manifest_path, study_dir=root)
            for entry in manifest["entries"]:
                payload = entry["payload"]
                if (
                    payload.get("surface_id") != surface
                    or payload.get("no_op") is True
                ):
                    continue
                cell = payload["cell"]
                pair = (float(cell["rho_conv"]), float(cell["rho_dense"]))
                result = _stage_result(
                    root, "rho_canary_extension", entry["entry_id"]
                )
                result_cells = result.get("cells")
                cell_result = (
                    result_cells[0]
                    if isinstance(result_cells, list)
                    and len(result_cells) == 1
                    else {}
                )
                declared[pair] = {
                    "candidate_id": cell["cell_id"],
                    "cell_id": cell["cell_id"],
                    "rho_conv": pair[0],
                    "rho_dense": pair[1],
                    "admissible": False,
                    "completed_steps": 0,
                    "final_validation_accuracy": None,
                    "final_validation_loss": None,
                    "median_projection_efficiency": None,
                    "inadmissible_reason": cell_result.get(
                        "safety_failure",
                        cell_result.get("reason", "extension_canary_failed"),
                    ),
                    "result_path": None,
                }

    candidate_stages = ["rho_core_candidates"]
    if include_extension:
        candidate_stages.append("rho_extension_candidates")
    for stage in candidate_stages:
        manifest = load_stage_manifest(
            _stage_manifest_path(root, stage), study_dir=root
        )
        for entry in manifest["entries"]:
            if (
                entry["payload"].get("surface_id") != surface
                or entry["payload"].get("no_op") is True
            ):
                continue
            result = _stage_result(root, stage, entry["entry_id"])
            rho_conv = float(entry["payload"]["rho_conv"])
            rho_dense = float(entry["payload"]["rho_dense"])
            completed_steps = result.get("completed_steps")
            final_validation = result.get("final_validation", {})
            accuracy = result.get(
                "final_validation_accuracy",
                final_validation.get("accuracy")
                if isinstance(final_validation, Mapping)
                else None,
            )
            loss = result.get(
                "final_validation_loss",
                final_validation.get("loss")
                if isinstance(final_validation, Mapping)
                else None,
            )
            efficiency = result.get(
                "median_projection_efficiency",
                (
                    result.get("safety", {}).get(
                        "median_projection_efficiency"
                    )
                    if isinstance(result.get("safety"), Mapping)
                    else result.get("projection_efficiency")
                ),
            )
            finite_efficiency = (
                isinstance(efficiency, (int, float))
                and not isinstance(efficiency, bool)
                and math.isfinite(float(efficiency))
                and float(efficiency) >= 0.0
            )
            admissible = (
                result.get("admissible") is True
                or result.get("status") in {"passed", "complete", "selected"}
            ) and completed_steps == CANDIDATE_STEPS and finite_efficiency
            pair = (rho_conv, rho_dense)
            if pair not in declared:
                raise RuntimeError(
                    "Expected every trained candidate to correspond to a "
                    f"declared canary cell. Provided value: {pair!r}."
                )
            declared[pair] = {
                "candidate_id": entry["payload"]["cell_id"],
                "cell_id": entry["payload"]["cell_id"],
                "rho_conv": rho_conv,
                "rho_dense": rho_dense,
                "admissible": admissible,
                "completed_steps": completed_steps if admissible else 0,
                "final_validation_accuracy": accuracy if admissible else None,
                "final_validation_loss": loss if admissible else None,
                "median_projection_efficiency": (
                    float(efficiency) if admissible else None
                ),
                "inadmissible_reason": (
                    None
                    if admissible
                    else (
                        "missing_or_nonfinite_projection_efficiency"
                        if not finite_efficiency
                        else result.get(
                            "safety_failure",
                            result.get("reason", "candidate_inadmissible"),
                        )
                    )
                ),
                "result_path": (
                    f"{_entry_base(stage, entry['entry_id'])}/result.json"
                ),
            }
    return [declared[pair] for pair in sorted(declared)]


def _write_selection(
    root: Path,
    stage: str,
    entry_id: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    from .perfectdiode_hparam_spec import (
        BoundaryExpansionPlan,
        select_surface_candidates,
    )

    final = stage == "select_final"
    records = _candidate_results(
        root, payload["surface_id"], include_extension=final
    )
    if records:
        rho_conv_values = sorted(
            {float(record["rho_conv"]) for record in records}
        )
        rho_dense_values = sorted(
            {float(record["rho_dense"]) for record in records}
        )
        selection = select_surface_candidates(
            records,
            rho_conv_values=rho_conv_values,
            rho_dense_values=rho_dense_values,
            expansion_available=not final,
            minimum_accuracy=0.90,
            plateau_relative_tolerance=0.02,
            required_completed_steps=CANDIDATE_STEPS,
        )
        expansion_value = selection.get("expansion")
        if isinstance(expansion_value, BoundaryExpansionPlan):
            expansion = {
                axis: direction
                for axis, direction in expansion_value.directions
            }
            selection = {
                **selection,
                "expansion": expansion,
                "expansion_axes": list(expansion),
                "expanded_rho_conv_values": list(
                    expansion_value.rho_conv_values
                ),
                "expanded_rho_dense_values": list(
                    expansion_value.rho_dense_values
                ),
                "new_cells": [
                    {"rho_conv": conv, "rho_dense": dense}
                    for conv, dense in expansion_value.new_cells
                ],
            }
        selected = selection.get("selected")
        if isinstance(selected, Mapping):
            source = next(
                record
                for record in records
                if record["candidate_id"] == selected["candidate_id"]
            )
            selected = {
                **copy.deepcopy(dict(selected)),
                "cell_id": source["cell_id"],
                "result_path": source["result_path"],
            }
            selection["selected"] = selected
            selection["selected_cell_id"] = selected["cell_id"]
            selection["selected_rho_conv"] = selected["rho_conv"]
            selection["selected_rho_dense"] = selected["rho_dense"]
    else:
        selection = {
            "status": "unresolved_no_pass",
            "reason": "no_safe_center_or_declared_candidate_grid",
            "selected": None,
            "diagnostic_best": None,
            "minimum_final_validation_loss": None,
            "plateau": [],
            "expansion": {},
            "expansion_axes": [],
            "evaluated_candidates": [],
        }
    selection["eligible_for_long_confirm"] = (
        final and selection["status"] == "selected"
    )
    result = {
        "schema_version": (
            "mnist-conv-perfectdiode-surface-selection/v1"
        ),
        "surface_id": payload["surface_id"],
        "architecture": payload["architecture"],
        "scheme": payload["row"]["scheme"],
        "optimizer": payload["optimizer"],
        "selection_stage": stage,
        "minimum_accuracy": 0.90,
        "inclusive_accuracy_gate": True,
        "plateau_relative_tolerance": 0.02,
        "official_test_read": False,
        **selection,
    }
    entry_dir = root / _entry_base(stage, entry_id)
    atomic_write_json(entry_dir / "result.json", result, canonical=True)
    selected = result.get("selected")
    atomic_write_csv(
        entry_dir / "result.csv",
        [
            "surface_id",
            "status",
            "reason",
            "selected_cell_id",
            "rho_conv",
            "rho_dense",
            "final_validation_accuracy",
            "final_validation_loss",
        ],
        [
            {
                "surface_id": payload["surface_id"],
                "status": result["status"],
                "reason": result.get("reason"),
                "selected_cell_id": result.get("selected_cell_id"),
                "rho_conv": (
                    selected.get("rho_conv")
                    if isinstance(selected, Mapping)
                    else None
                ),
                "rho_dense": (
                    selected.get("rho_dense")
                    if isinstance(selected, Mapping)
                    else None
                ),
                "final_validation_accuracy": (
                    selected.get("final_validation_accuracy")
                    if isinstance(selected, Mapping)
                    else None
                ),
                "final_validation_loss": (
                    selected.get("final_validation_loss")
                    if isinstance(selected, Mapping)
                    else None
                ),
            }
        ],
    )
    return result


def _write_no_op(
    root: Path,
    stage: str,
    entry_id: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "schema_version": "mnist-conv-perfectdiode-zero-work/v1",
        "status": "zero_work_complete",
        "stage": stage,
        "entry_id": entry_id,
        "reason": payload["reason"],
        "surface_id": payload.get("surface_id"),
        "official_test_read": False,
    }
    atomic_write_json(
        root / _entry_base(stage, entry_id) / "result.json",
        result,
        canonical=True,
    )
    return result


def _write_audit(
    root: Path,
    entry_id: str,
    payload: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "schema_version": "mnist-conv-perfectdiode-audit/v1",
        "status": "passed",
        **copy.deepcopy(dict(payload)),
        "code_provenance": normalize_code_provenance(provenance),
        "official_test_read": False,
        "launched_jobs": 0,
        "stage_manifest_sha256": sha256_file(
            _stage_manifest_path(root, "audit")
        ),
    }
    atomic_write_json(
        root / _entry_base("audit", entry_id) / "result.json",
        result,
        canonical=True,
    )
    return result


RuntimeExecutor = Callable[..., Mapping[str, Any]]


def _default_runtime_executor(**kwargs: Any) -> Mapping[str, Any]:
    from .perfectdiode_hparam_runtime import execute_perfectdiode_stage_entry

    return execute_perfectdiode_stage_entry(**kwargs)


def execute_screen_manifest_entry(
    *,
    shard_dir: str | Path,
    manifest_path: str | Path,
    entry_index: int,
    data_root: str | Path,
    device: str,
    download: bool = False,
    provenance: Mapping[str, Any] | None = None,
    runtime_executor: RuntimeExecutor | None = None,
) -> dict[str, Any]:
    """Execute one manifest entry and publish completion only after hashing."""

    root, spec, descriptor = load_host_shard(shard_dir)
    manifest = load_stage_manifest(
        manifest_path, study_dir=root, expected_study_id=_spec_id(spec)
    )
    current = _require_exact_clean_staged_source(
        provenance or code_provenance(),
        descriptor["staged_source"],
        context="screen worker execution",
    )
    if normalize_code_provenance(manifest["code_provenance"]) != current:
        raise RuntimeError(
            "Expected worker clean Git provenance to match the manifest exactly. "
            f"Provided value: manifest={manifest['code_provenance']!r}, "
            f"worker={current!r}."
        )
    entries = manifest["entries"]
    if type(entry_index) is not int or not 0 <= entry_index < len(entries):
        raise _error(
            f"an integer in [0, {len(entries) - 1}]",
            entry_index,
            "entry_index",
        )
    entry = entries[entry_index]
    if entry_is_complete(
        study_dir=root,
        manifest_path=manifest_path,
        entry_id=entry["entry_id"],
    ):
        return {
            "entry_index": entry_index,
            "entry_id": entry["entry_id"],
            "status": "resumed_complete",
        }
    stage = manifest["stage_name"]
    payload = entry["payload"]
    if payload.get("no_op") is True:
        result = _write_no_op(root, stage, entry["entry_id"], payload)
    elif stage == "audit":
        result = _write_audit(root, entry["entry_id"], payload, current)
    elif stage in {"select_core", "select_final"}:
        result = _write_selection(root, stage, entry["entry_id"], payload)
    else:
        executor = runtime_executor or _default_runtime_executor
        result = executor(
            study=_spec_data(spec),
            shard_dir=root,
            stage=stage,
            payload={**copy.deepcopy(payload), "entry_id": entry["entry_id"]},
            data_root=Path(data_root).expanduser().resolve(),
            device=device,
            download=download,
        )
        if not isinstance(result, Mapping):
            raise _error("a result mapping", result, "runtime_executor result")
    publish_entry_completion(
        study_dir=root,
        manifest_path=manifest_path,
        entry_id=entry["entry_id"],
    )
    return {
        "entry_index": entry_index,
        "entry_id": entry["entry_id"],
        "status": "complete",
        "result_status": result.get("status"),
    }


def execute_screen_stage(
    *,
    shard_dir: str | Path,
    manifest_path: str | Path,
    data_root: str | Path,
    device: str,
    download: bool = False,
    provenance: Mapping[str, Any] | None = None,
    runtime_executor: RuntimeExecutor | None = None,
) -> dict[str, Any]:
    """Run pending entries sequentially on one GPU and close the stage."""

    root, spec, _descriptor = load_host_shard(shard_dir)
    manifest = load_stage_manifest(
        manifest_path, study_dir=root, expected_study_id=_spec_id(spec)
    )
    pending = [
        entry
        for entry in manifest["entries"]
        if not entry_is_complete(
            study_dir=root,
            manifest_path=manifest_path,
            entry_id=entry["entry_id"],
        )
    ]
    executions = [
        execute_screen_manifest_entry(
            shard_dir=root,
            manifest_path=manifest_path,
            entry_index=entry["entry_index"],
            data_root=data_root,
            device=device,
            download=download,
            provenance=provenance,
            runtime_executor=runtime_executor,
        )
        for entry in pending
    ]
    marker = publish_stage_completion(
        study_dir=root, manifest_path=manifest_path
    )
    return {
        "study_id": _spec_id(spec),
        "stage": manifest["stage_name"],
        "status": "complete",
        "entry_count": len(manifest["entries"]),
        "resumed_count": len(manifest["entries"]) - len(pending),
        "executions": executions,
        "completion_path": str(marker),
        "workers": 1,
    }


def run_representative_preflight_canary(
    *,
    shard_dir: str | Path,
    data_root: str | Path,
    device: str = "cuda",
    download: bool = False,
    provenance: Mapping[str, Any] | None = None,
    runtime_executor: RuntimeExecutor | None = None,
) -> dict[str, Any]:
    """Run the non-canonical ours/Adam initial-center canary exactly once.

    Assets, security, and optimizer probes are canonical prerequisites.  The
    preflight is an execution/memory/throughput check and is deliberately not
    a public scientific stage: a safety-clean false center is still a valid
    execution result because the canonical center search may move downward.
    """

    root, spec, descriptor = load_host_shard(shard_dir)
    validate_stage_completion(
        study_dir=root,
        manifest_path=_stage_manifest_path(root, "optimizer_probe"),
    )
    current = _require_exact_clean_staged_source(
        provenance or code_provenance(),
        descriptor["staged_source"],
        context="preflight execution",
    )
    row = next(
        row
        for row in host_rows(spec, descriptor["host"])
        if row["scheme"] == "ours"
    )
    surface = {
        "surface_id": surface_id(row, "adam"),
        "row": row,
        "optimizer": "adam",
        "host": descriptor["host"],
        "architecture": row["architecture"],
    }
    entry_id = f"{surface['surface_id']}--initial-center"
    staged = descriptor["staged_source"]
    payload = {
        **_surface_payload(surface),
        "entry_id": entry_id,
        "source_commit": staged["commit"],
        "source_archive_sha256": staged["archive_sha256"],
        "environment_sha256": descriptor["environment_sha256"],
        "rho_conv": 3e-3,
        "rho_dense": 1e-2,
        "canary_steps": CANARY_STEPS,
        "canonical_stage": False,
    }
    preflight_root = root / "preflight" / "adam-center-ours"
    request = {
        "schema_version": "mnist-conv-perfectdiode-preflight-request/v1",
        "study_id": _spec_id(spec),
        "host": descriptor["host"],
        "architecture": descriptor["architecture"],
        "payload": payload,
        "code_provenance": current,
        "optimizer_probe_completion_sha256": sha256_file(
            _stage_manifest_path(root, "optimizer_probe").parent
            / "complete.json"
        ),
    }
    _publish_identical(preflight_root / "request.json", request)
    receipt_path = preflight_root / "receipt.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        for record in receipt["outputs"]:
            path = root / record["path"]
            if (
                not path.is_file()
                or path.stat().st_size != record["bytes"]
                or sha256_file(path) != record["sha256"]
            ):
                raise RuntimeError(
                    "Expected every resumed preflight output to retain its "
                    f"recorded hash. Provided value: {path}."
                )
        return {**receipt, "status": "resumed_complete"}

    executor = runtime_executor or _default_runtime_executor
    result = executor(
        study=_spec_data(spec),
        shard_dir=root,
        stage="preflight_canary",
        payload=payload,
        data_root=Path(data_root).expanduser().resolve(),
        device=device,
        download=download,
    )
    output_dir = root / "stages" / "preflight_canary" / "entries" / entry_id
    outputs = artifact_records(
        root, [output_dir / name for name in _PREFLIGHT_OUTPUTS]
    )
    benchmark = result.get("benchmark")
    cell = result.get("cell")
    execution_passed = (
        result.get("canonical_stage") is False
        and isinstance(benchmark, Mapping)
        and isinstance(cell, Mapping)
        and cell.get("expected_steps") == CANARY_STEPS
        and isinstance(cell.get("completed_steps"), int)
        and 0 < cell["completed_steps"] <= CANARY_STEPS
        and result.get("official_test_read") is False
    )
    receipt = {
        "schema_version": "mnist-conv-perfectdiode-preflight-receipt/v1",
        "study_id": _spec_id(spec),
        "host": descriptor["host"],
        "architecture": descriptor["architecture"],
        "surface_id": surface["surface_id"],
        "entry_id": entry_id,
        "execution_passed": execution_passed,
        "scientific_center_safety_clean": result.get("preflight_passed"),
        "completed_steps": (
            cell.get("completed_steps") if isinstance(cell, Mapping) else None
        ),
        "expected_steps": CANARY_STEPS,
        "benchmark": benchmark,
        "request_sha256": sha256_file(preflight_root / "request.json"),
        "outputs": outputs,
        "official_test_read": False,
    }
    _publish_identical(receipt_path, receipt)
    if not execution_passed:
        raise RuntimeError(
            "Expected the representative Adam center preflight to complete its "
            f"execution/provenance gate. Provided value: {receipt!r}."
        )
    return {**receipt, "status": "complete"}


def run_host_screen(
    *,
    spec: Any,
    results_root: str | Path,
    host: str,
    data_root: str | Path,
    device: str = "cuda",
    download: bool = False,
    provenance: Mapping[str, Any] | None = None,
    source_commit: str | None = None,
    source_archive_sha256: str | None = None,
    environment: Mapping[str, Any] | None = None,
    runtime_executor: RuntimeExecutor | None = None,
    stop_after: str = "select_final",
) -> dict[str, Any]:
    """Resume a complete Conv1 or Conv2 screen through ``select_final``."""

    if stop_after not in SCREEN_STAGE_SEQUENCE:
        raise _error(f"one of {SCREEN_STAGE_SEQUENCE!r}", stop_after, "stop_after")
    if source_commit is None or source_archive_sha256 is None:
        raise RuntimeError(
            "Expected production host execution to provide the exact staged "
            "source commit and archive SHA-256. "
            f"Provided commit={source_commit!r}, "
            f"archive_sha256={source_archive_sha256!r}."
        )
    raw_source = normalize_code_provenance(provenance or code_provenance())
    source = _require_exact_clean_staged_source(
        raw_source,
        {
            "commit": source_commit,
            "archive_sha256": source_archive_sha256,
            "effective_code_fingerprint": raw_source[
                "effective_code_fingerprint"
            ],
        },
        context="production host execution",
    )
    _study, shard = create_host_shard(
        spec,
        results_root,
        host,
        provenance=source,
        source_commit=source_commit,
        source_archive_sha256=source_archive_sha256,
        environment=environment,
    )
    stage_results = []
    for stage in SCREEN_STAGE_SEQUENCE:
        if stage == "rho_canary_core":
            run_representative_preflight_canary(
                shard_dir=shard,
                data_root=data_root,
                device=device,
                download=download,
                provenance=source,
                runtime_executor=runtime_executor,
            )
        manifest_path = _stage_manifest_path(shard, stage)
        if manifest_path.is_file():
            manifest = load_stage_manifest(
                manifest_path,
                study_dir=shard,
                expected_study_id=_spec_id(spec),
            )
        else:
            manifest, manifest_path = publish_screen_stage_manifest(
                shard, stage, provenance=source
            )
        stage_results.append(
            execute_screen_stage(
                shard_dir=shard,
                manifest_path=manifest_path,
                data_root=data_root,
                device=device,
                download=download,
                provenance=source,
                runtime_executor=runtime_executor,
            )
        )
        if stage == stop_after:
            break
    return {
        "study_id": _spec_id(spec),
        "host": host,
        "architecture": HOST_ARCHITECTURES[host],
        "shard_dir": str(shard),
        "status": "complete_through_" + stop_after,
        "stages": stage_results,
    }


def plan_host_screen(spec: Any, host: str) -> dict[str, Any]:
    """Return the complete conditional plan without creating files or jobs."""

    surfaces = host_surfaces(spec, host)
    counts: dict[str, Any] = {
        "audit": 1,
        "assets": 1,
        "fixed_tk_gradient_security": 3,
        "optimizer_probe": 6,
        "rho_canary_core": 6,
        "rho_core_candidates": {"minimum": 6, "maximum": 54},
        "select_core": 6,
        "rho_canary_extension": {"minimum": 6, "maximum": 42},
        "rho_extension_candidates": {"minimum": 6, "maximum": 42},
        "select_final": 6,
        "long_confirm": 0,
        "finalize": 0,
    }
    return {
        "planned": True,
        "study_id": _spec_id(spec),
        "study_schema_version": PD_STUDY_SCHEMA_VERSION,
        "run_schema_version": PD_RUN_SCHEMA_VERSION,
        "host": host,
        "architecture": HOST_ARCHITECTURES[host],
        "surface_ids": [value["surface_id"] for value in surfaces],
        "stage_sequence": list(PUBLIC_STAGE_SEQUENCE),
        "entry_counts": counts,
        "routing": dict(HOST_ARCHITECTURES),
        "one_process_per_gpu": True,
        "launched_jobs": 0,
    }


def _iter_regular_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise _error("no symlinks", str(path), "shard")
        if path.is_file():
            yield path


def merge_screen_shards(
    destination: str | Path,
    *,
    akib_shard: str | Path,
    trex_shard: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Copy two closed shards into a new aggregate and verify every byte."""

    target = Path(destination).expanduser().resolve()
    sources = {"akib": Path(akib_shard).expanduser().resolve(), "trex": Path(trex_shard).expanduser().resolve()}
    loaded: dict[str, tuple[Path, Any, dict[str, Any]]] = {}
    config_bytes: bytes | None = None
    study_id: str | None = None
    common_source: dict[str, Any] | None = None
    for host, source in sources.items():
        root, spec, descriptor = load_host_shard(source)
        if descriptor["host"] != host:
            raise _error(host, descriptor["host"], f"shards.{host}.host")
        validate_stage_completion(
            study_dir=root,
            manifest_path=_stage_manifest_path(root, "select_final"),
        )
        current_config = (root / "study.resolved.json").read_bytes()
        if config_bytes is None:
            config_bytes = current_config
            study_id = _spec_id(spec)
        elif current_config != config_bytes or _spec_id(spec) != study_id:
            raise RuntimeError(
                "Expected both shards to contain the identical resolved study."
            )
        staged_source = descriptor.get("staged_source")
        if (
            not isinstance(staged_source, Mapping)
            or staged_source.get("commit") is None
            or staged_source.get("archive_sha256") is None
        ):
            raise RuntimeError(
                "Expected every production shard to record the explicit staged "
                f"source commit and archive SHA-256. Provided host={host!r}, "
                f"value={staged_source!r}."
            )
        normalized_source = copy.deepcopy(dict(staged_source))
        _require_exact_clean_staged_source(
            descriptor["code_provenance"],
            normalized_source,
            context=f"{host} merged shard",
        )
        if common_source is None:
            common_source = normalized_source
        elif normalized_source != common_source:
            raise RuntimeError(
                "Expected Akib and Trex to use the identical staged commit, "
                "archive, and effective code fingerprint. "
                f"Provided akib/trex values: {common_source!r}, "
                f"{normalized_source!r}."
            )
        if (
            descriptor["code_provenance"]["effective_code_fingerprint"]
            != normalized_source["effective_code_fingerprint"]
        ):
            raise RuntimeError(
                "Expected shard code provenance to match its staged-source "
                f"descriptor. Provided host={host!r}."
            )
        loaded[host] = (root, spec, descriptor)
    assert config_bytes is not None and study_id is not None and common_source is not None
    target.mkdir(parents=True, exist_ok=True)
    config_path = target / "study.resolved.json"
    if config_path.exists():
        if config_path.read_bytes() != config_bytes:
            raise RuntimeError(
                "Expected existing aggregate config to be byte-identical."
            )
    else:
        atomic_write_bytes(config_path, config_bytes)

    shard_receipts = []
    for host in ("akib", "trex"):
        source = loaded[host][0]
        records = [
            {
                "path": path.relative_to(source).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in _iter_regular_files(source)
        ]
        destination_root = target / "shards" / host
        for record in records:
            source_path = source / record["path"]
            destination_path = destination_root / record["path"]
            if destination_path.exists():
                if (
                    destination_path.stat().st_size != record["bytes"]
                    or sha256_file(destination_path) != record["sha256"]
                ):
                    raise RuntimeError(
                        "Expected an existing aggregate shard artifact to match "
                        f"its source. Provided value: {destination_path}."
                    )
            else:
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination_path)
            if sha256_file(destination_path) != record["sha256"]:
                raise RuntimeError(
                    f"Expected copied shard artifact hash to verify. Provided value: {destination_path}."
                )
        shard_receipts.append(
            {
                "host": host,
                "architecture": HOST_ARCHITECTURES[host],
                "shard_id": loaded[host][2]["shard_id"],
                "records": records,
                "records_sha256": sha256_json(records),
            }
        )
    receipt = {
        "schema_version": PD_MERGE_SCHEMA_VERSION,
        "study_id": study_id,
        "config_sha256": loaded["akib"][1].config_sha256,
        "resolved_config_file_sha256": sha256_file(config_path),
        "shards": shard_receipts,
        "aggregate_records_sha256": sha256_json(
            [item["records_sha256"] for item in shard_receipts]
        ),
        "screen_complete_through": "select_final",
        "long_confirm_execution_host": "trex",
        "staged_source": common_source,
    }
    _publish_identical(target / "merge_receipt.json", receipt)
    return target, receipt


def _aggregate_selections(root: Path) -> list[dict[str, Any]]:
    selections = []
    for host in ("akib", "trex"):
        shard = root / "shards" / host
        _root, spec, _descriptor = load_host_shard(shard)
        for surface in host_surfaces(spec, host):
            value = _stage_result(shard, "select_final", surface["surface_id"])
            selections.append(
                {
                    "host": host,
                    "shard": f"shards/{host}",
                    "surface": surface,
                    "selection": value,
                    "selection_path": (
                        f"shards/{host}/"
                        f"{_entry_base('select_final', surface['surface_id'])}/result.json"
                    ),
                }
            )
    return selections


def publish_aggregate_stage_manifest(
    aggregate_dir: str | Path,
    stage: str,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    if stage not in AGGREGATE_STAGE_SEQUENCE:
        raise _error(f"one of {AGGREGATE_STAGE_SEQUENCE!r}", stage, "stage")
    root = Path(aggregate_dir).expanduser().resolve()
    spec = _load_spec(root / "study.resolved.json")
    receipt = read_json(root / "merge_receipt.json")
    source = _require_exact_clean_staged_source(
        provenance or code_provenance(),
        receipt["staged_source"],
        context="aggregate planning on Trex",
    )
    selections = _aggregate_selections(root)
    entries = []
    if stage == "long_confirm":
        execution_descriptor = read_json(
            root / "shards" / "trex" / "shard.resolved.json"
        )
        for item in selections:
            surface = item["surface"]
            selection = item["selection"]
            source_descriptor = read_json(
                root / item["shard"] / "shard.resolved.json"
            )
            staged_source = source_descriptor["staged_source"]
            payload = {
                **_surface_payload(surface),
                "source_shard": item["shard"],
                "selection_result": item["selection_path"],
                "execution_host": "trex",
                "source_commit": staged_source["commit"],
                "source_archive_sha256": staged_source["archive_sha256"],
                "source_shard_environment_sha256": source_descriptor[
                    "environment_sha256"
                ],
                "execution_environment_sha256": execution_descriptor[
                    "environment_sha256"
                ],
                "environment_sha256": execution_descriptor[
                    "environment_sha256"
                ],
                "config_sha256": source_descriptor["config_sha256"],
                "probe_result_path": (
                    f"{item['shard']}/stages/optimizer_probe/entries/"
                    f"{surface['surface_id']}/result.json"
                ),
                "fresh_restart_from_shared_initialization": True,
                "epochs": 10 if surface["architecture"] == "conv1" else 30,
                "expected_steps": LONG_STEPS[surface["architecture"]],
                "official_test_read": False,
            }
            if (
                selection.get("status") != "selected"
                or selection.get("eligible_for_long_confirm") is not True
            ):
                entries.append(
                    _no_op_entry(
                        stage,
                        f"{surface['surface_id']}--ineligible",
                        index=len(entries),
                        payload=payload,
                        reason=selection.get(
                            "reason", "surface_not_eligible_for_long_confirm"
                        ),
                    )
                )
            else:
                selected = selection["selected"]
                entries.append(
                    _normal_entry(
                        stage,
                        surface["surface_id"],
                        index=len(entries),
                        payload={
                            **payload,
                            "selected_cell_id": selected["cell_id"],
                            "cell_id": selected["cell_id"],
                            "rho_conv": selected["rho_conv"],
                            "rho_dense": selected["rho_dense"],
                            "candidate_result": selected["result_path"],
                        },
                    )
                )
        upstream = [root / "merge_receipt.json"] + [
            root / item["selection_path"] for item in selections
        ]
    else:
        entry_id = "finalization"
        entries = [
            _normal_entry(
                stage,
                entry_id,
                index=0,
                payload={
                    "partial_handoff_allowed": True,
                    "surface_count": 12,
                    "ordinary_mnist_diagnostic": True,
                    "medium_affine_paper_protocol_resolved": False,
                    "official_test_read": False,
                },
            )
        ]
        upstream = [root / "merge_receipt.json"]
        upstream.extend(_completed_stage_upstreams(root, "long_confirm"))
    manifest = build_stage_manifest(
        study_dir=root,
        study_id=_spec_id(spec),
        study_config_path=root / "study.resolved.json",
        code_provenance=source,
        stage_name=stage,
        entries=entries,
        upstream_paths=upstream,
    )
    path = _stage_manifest_path(root, stage)
    publish_stage_manifest(path, manifest, study_dir=root)
    return manifest, path


def _read_required_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            f"Expected the selected {label} artifact to exist. Provided value: {path}."
        )
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise RuntimeError(
            f"Expected the selected {label} artifact to contain a mapping. "
            f"Provided value: {path}."
        )
    return copy.deepcopy(dict(value))


def _selected_candidate_handoff(
    root: Path,
    item: Mapping[str, Any],
) -> dict[str, Any]:
    """Hydrate one selection with its complete candidate and provenance record."""

    selection = item["selection"]
    selected = selection.get("selected")
    shard = root / item["shard"]
    descriptor_path = shard / "shard.resolved.json"
    descriptor = _read_required_mapping(
        descriptor_path, label="source-shard descriptor"
    )
    shard_spec = _load_spec(shard / "study.resolved.json")
    canonical_config_sha256 = shard_spec.config_sha256
    selection_path = root / item["selection_path"]
    selection_sha256 = sha256_file(selection_path)
    base_provenance = {
        "study_id": descriptor["study_id"],
        "config_sha256": canonical_config_sha256,
        "resolved_config_file_sha256": descriptor[
            "resolved_config_file_sha256"
        ],
        "source_commit": descriptor["staged_source"]["commit"],
        "source_archive_sha256": descriptor["staged_source"]["archive_sha256"],
        "source_shard_environment_sha256": descriptor["environment_sha256"],
        "dataset_contract_sha256": descriptor["dataset_contract_sha256"],
        "selection_result_sha256": selection_sha256,
        "selection_stage_manifest_sha256": sha256_file(
            shard / "stages" / "select_final" / "manifest.json"
        ),
        "merge_receipt_sha256": sha256_file(root / "merge_receipt.json"),
    }
    if not isinstance(selected, Mapping):
        return {
            "selected_rho_targets": None,
            "raw_learning_rates_by_parameter": None,
            "validation_metrics": None,
            "safety_diagnostics": None,
            "median_projection_efficiency": None,
            "achieved_updates_by_parameter": None,
            "checkpoints": None,
            "provenance": base_provenance,
        }

    result_relative = selected.get("result_path")
    if not isinstance(result_relative, str) or not result_relative:
        raise RuntimeError(
            "Expected every selected surface to record its candidate result path. "
            f"Provided value: {result_relative!r}."
        )
    candidate_path = (shard / result_relative).resolve()
    try:
        candidate_path.relative_to(shard.resolve())
    except ValueError as exc:
        raise RuntimeError(
            "Expected the selected candidate result to stay inside its source shard. "
            f"Provided value: {result_relative!r}."
        ) from exc
    candidate = _read_required_mapping(candidate_path, label="candidate result")
    run_spec_path = candidate_path.parent / "run_spec.json"
    validation_path = candidate_path.parent / "validation.json"
    run_spec = _read_required_mapping(run_spec_path, label="candidate run spec")
    validation = _read_required_mapping(
        validation_path, label="candidate validation history"
    )
    if (
        candidate.get("official_test_read") is not False
        or run_spec.get("official_test_read") is not False
        or validation.get("official_test_read") is not False
    ):
        raise RuntimeError(
            "Expected every selected candidate artifact to record "
            "official_test_read=false."
        )
    if (
        candidate.get("study_id") != descriptor["study_id"]
        or candidate.get("config_sha256") != canonical_config_sha256
        or run_spec.get("study_id") != descriptor["study_id"]
        or run_spec.get("config_sha256") != canonical_config_sha256
    ):
        raise RuntimeError(
            "Expected the selected candidate to match its source shard study/config."
        )
    if (
        float(candidate.get("rho_conv")) != float(selected["rho_conv"])
        or float(candidate.get("rho_dense")) != float(selected["rho_dense"])
        or candidate.get("cell_id") != selected.get("cell_id")
    ):
        raise RuntimeError(
            "Expected selected rho targets and cell identity to match the "
            "candidate result exactly."
        )
    run_provenance = run_spec.get("provenance")
    if not isinstance(run_provenance, Mapping):
        raise RuntimeError(
            "Expected the selected candidate run spec to contain provenance."
        )
    staged_source = descriptor["staged_source"]
    required_provenance = {
        "source_commit": staged_source["commit"],
        "source_archive_sha256": staged_source["archive_sha256"],
        "environment_sha256": descriptor["environment_sha256"],
        "initialization_checkpoint_sha256": candidate.get(
            "initialization_checkpoint_sha256"
        ),
        "initialization_tensor_sha256": candidate.get(
            "initialization_tensor_sha256"
        ),
        "train_indices_sha256": candidate.get("train_indices_sha256"),
        "validation_indices_sha256": candidate.get(
            "validation_indices_sha256"
        ),
        "probe_entry_id": candidate.get("probe_entry_id"),
    }
    for key, expected in required_provenance.items():
        if expected is None or run_provenance.get(key) != expected:
            raise RuntimeError(
                "Expected selected candidate provenance to match its immutable "
                f"source shard. Provided field={key!r}, "
                f"run_spec={run_provenance.get(key)!r}, expected={expected!r}."
            )
    rates = candidate.get("raw_learning_rates_by_parameter")
    safety = candidate.get("safety")
    achieved = candidate.get("achieved_updates_by_parameter")
    checkpoints = candidate.get("checkpoints")
    records = validation.get("records")
    if not isinstance(rates, Mapping) or not rates:
        raise RuntimeError(
            "Expected the selected candidate to contain the full raw LR vector."
        )
    if not isinstance(safety, Mapping):
        raise RuntimeError(
            "Expected the selected candidate to contain safety diagnostics."
        )
    if not isinstance(achieved, Mapping) or not achieved:
        raise RuntimeError(
            "Expected the selected candidate to contain achieved-update diagnostics."
        )
    if not isinstance(checkpoints, Mapping) or not checkpoints:
        raise RuntimeError(
            "Expected the selected candidate to contain checkpoint hashes."
        )
    if not isinstance(records, list) or not records:
        raise RuntimeError(
            "Expected the selected candidate to contain epochwise validation records."
        )
    probe_entry_id = str(candidate["probe_entry_id"])
    probe_path = (
        shard
        / "stages"
        / "optimizer_probe"
        / "entries"
        / probe_entry_id
        / "result.json"
    )
    _read_required_mapping(probe_path, label="optimizer probe")
    candidate_stage = candidate_path.relative_to(shard).parts[1]
    provenance = {
        **base_provenance,
        "train_indices_sha256": candidate["train_indices_sha256"],
        "validation_indices_sha256": candidate["validation_indices_sha256"],
        "initialization_checkpoint_sha256": candidate[
            "initialization_checkpoint_sha256"
        ],
        "initialization_tensor_sha256": candidate[
            "initialization_tensor_sha256"
        ],
        "probe_entry_id": probe_entry_id,
        "probe_result_sha256": sha256_file(probe_path),
        "candidate_result_sha256": sha256_file(candidate_path),
        "candidate_run_spec_sha256": sha256_file(run_spec_path),
        "candidate_validation_sha256": sha256_file(validation_path),
        "candidate_stage_manifest_sha256": sha256_file(
            shard / "stages" / candidate_stage / "manifest.json"
        ),
    }
    return {
        "selected_rho_targets": {
            "rho_conv": float(selected["rho_conv"]),
            "rho_dense": float(selected["rho_dense"]),
            "cell_id": selected["cell_id"],
        },
        "raw_learning_rates_by_parameter": copy.deepcopy(dict(rates)),
        "validation_metrics": {
            "final_loss": candidate.get("final_validation_loss"),
            "final_accuracy": candidate.get("final_validation_accuracy"),
            "best_loss": candidate.get("best_validation_loss"),
            "best_epoch": candidate.get("best_validation_epoch"),
            "inclusive_90_percent_accuracy_gate_passed": candidate.get(
                "inclusive_90_percent_accuracy_gate_passed"
            ),
            "records": copy.deepcopy(records),
        },
        "safety_diagnostics": copy.deepcopy(dict(safety)),
        "median_projection_efficiency": candidate.get(
            "median_projection_efficiency"
        ),
        "achieved_updates_by_parameter": copy.deepcopy(dict(achieved)),
        "checkpoints": copy.deepcopy(dict(checkpoints)),
        "provenance": provenance,
    }


def _long_confirmation_handoff(
    root: Path,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    entry_id = entry["entry_id"]
    entry_dir = root / _entry_base("long_confirm", entry_id)
    result_path = entry_dir / "result.json"
    result = _read_required_mapping(result_path, label="long-confirm result")
    value: dict[str, Any] = {
        "status": result.get("status"),
        "result": result,
        "result_sha256": sha256_file(result_path),
        "stage_manifest_sha256": sha256_file(
            root / "stages" / "long_confirm" / "manifest.json"
        ),
    }
    if entry["payload"].get("no_op") is True:
        value["zero_work"] = True
        return value
    run_spec_path = entry_dir / "run_spec.json"
    validation_path = entry_dir / "validation.json"
    run_spec = _read_required_mapping(run_spec_path, label="long-confirm run spec")
    validation = _read_required_mapping(
        validation_path, label="long-confirm validation history"
    )
    if (
        result.get("official_test_read") is not False
        or run_spec.get("official_test_read") is not False
        or validation.get("official_test_read") is not False
    ):
        raise RuntimeError(
            "Expected every long-confirm artifact to record official_test_read=false."
        )
    provenance = run_spec.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError(
            "Expected every long-confirm run spec to contain provenance."
        )
    for field in (
        "source_commit",
        "source_archive_sha256",
        "source_shard_environment_sha256",
        "execution_environment_sha256",
    ):
        if not provenance.get(field):
            raise RuntimeError(
                "Expected long-confirm provenance to contain "
                f"{field}. Provided value: {provenance.get(field)!r}."
            )
    value.update(
        {
            "zero_work": False,
            "validation_records": copy.deepcopy(validation.get("records")),
            "run_provenance": copy.deepcopy(dict(provenance)),
            "run_spec_sha256": sha256_file(run_spec_path),
            "validation_sha256": sha256_file(validation_path),
        }
    )
    return value


def _write_finalization(root: Path, entry_id: str) -> dict[str, Any]:
    selections = _aggregate_selections(root)
    long_manifest = load_stage_manifest(
        _stage_manifest_path(root, "long_confirm"), study_dir=root
    )
    long_by_surface: dict[str, dict[str, Any]] = {}
    for entry in long_manifest["entries"]:
        surface = entry["payload"].get("surface_id")
        if surface is not None:
            long_by_surface[surface] = _long_confirmation_handoff(root, entry)
    rows = []
    for item in selections:
        surface = item["surface"]
        selection = item["selection"]
        candidate_handoff = _selected_candidate_handoff(root, item)
        source_shard = root / item["shard"]
        core_selection = _stage_result(
            source_shard, "select_core", surface["surface_id"]
        )
        core_expansion_axes = copy.deepcopy(
            core_selection.get("expansion_axes", [])
        )
        core_expansion = copy.deepcopy(core_selection.get("expansion", {}))
        boundary_status = selection.get("reason")
        if boundary_status is None:
            boundary_status = (
                "resolved_after_single_expansion"
                if core_expansion_axes
                else "resolved_without_boundary_expansion"
            )
        rows.append(
            {
                "surface_id": surface["surface_id"],
                "architecture": surface["architecture"],
                "scheme": surface["row"]["scheme"],
                "optimizer": surface["optimizer"],
                "selection_status": selection["status"],
                "selection": selection.get("selected"),
                "selected_rho_targets": candidate_handoff[
                    "selected_rho_targets"
                ],
                "raw_learning_rates_by_parameter": candidate_handoff[
                    "raw_learning_rates_by_parameter"
                ],
                "validation_metrics": candidate_handoff["validation_metrics"],
                "safety_diagnostics": candidate_handoff[
                    "safety_diagnostics"
                ],
                "median_projection_efficiency": candidate_handoff[
                    "median_projection_efficiency"
                ],
                "achieved_updates_by_parameter": candidate_handoff[
                    "achieved_updates_by_parameter"
                ],
                "checkpoints": candidate_handoff["checkpoints"],
                "boundary_status": boundary_status,
                "boundary_diagnostics": {
                    "core_selection_status": core_selection.get("status"),
                    "core_reason": core_selection.get("reason"),
                    "core_expansion_axes": core_expansion_axes,
                    "core_expansion": core_expansion,
                    "selection_status": selection["status"],
                    "reason": selection.get("reason"),
                    "final_expansion_axes": copy.deepcopy(
                        selection.get("expansion_axes", [])
                    ),
                    "final_expansion": copy.deepcopy(
                        selection.get("expansion", {})
                    ),
                },
                "provenance": candidate_handoff["provenance"],
                "long_confirmation": long_by_surface.get(surface["surface_id"]),
                "official_test_read": False,
            }
        )
    result = {
        "schema_version": "mnist-conv-perfectdiode-hparam-finalization/v1",
        "status": "partial_complete"
        if any(row["selection_status"] != "selected" for row in rows)
        else "complete",
        "study_id": read_json(root / "merge_receipt.json")["study_id"],
        "surface_count": 12,
        "selected_surface_count": sum(
            row["selection_status"] == "selected" for row in rows
        ),
        "rows": rows,
        "ordinary_mnist_diagnostic": True,
        "medium_affine_paper_protocol_resolved": False,
        "official_test_read": False,
        "merge_receipt_sha256": sha256_file(root / "merge_receipt.json"),
    }
    entry_dir = root / _entry_base("finalize", entry_id)
    atomic_write_json(entry_dir / "result.json", result, canonical=True)
    atomic_write_csv(
        entry_dir / "result.csv",
        [
            "surface_id",
            "architecture",
            "scheme",
            "optimizer",
            "selection_status",
            "boundary_status",
        ],
        rows,
    )
    return result


def execute_aggregate_stage(
    *,
    aggregate_dir: str | Path,
    manifest_path: str | Path,
    data_root: str | Path,
    device: str = "cuda",
    download: bool = False,
    provenance: Mapping[str, Any] | None = None,
    runtime_executor: RuntimeExecutor | None = None,
) -> dict[str, Any]:
    """Run aggregate entries sequentially; all long runs therefore serialize."""

    root = Path(aggregate_dir).expanduser().resolve()
    spec = _load_spec(root / "study.resolved.json")
    manifest = load_stage_manifest(
        manifest_path, study_dir=root, expected_study_id=_spec_id(spec)
    )
    receipt = read_json(root / "merge_receipt.json")
    current = _require_exact_clean_staged_source(
        provenance or code_provenance(),
        receipt["staged_source"],
        context="aggregate execution on Trex",
    )
    if normalize_code_provenance(manifest["code_provenance"]) != current:
        raise RuntimeError(
            "Expected aggregate worker clean Git provenance to match its "
            "immutable manifest exactly."
        )
    pending = [
        entry
        for entry in manifest["entries"]
        if not entry_is_complete(
            study_dir=root,
            manifest_path=manifest_path,
            entry_id=entry["entry_id"],
        )
    ]
    executions = []
    for entry in pending:
        payload = entry["payload"]
        if payload.get("no_op") is True:
            result = _write_no_op(
                root, manifest["stage_name"], entry["entry_id"], payload
            )
        elif manifest["stage_name"] == "finalize":
            result = _write_finalization(root, entry["entry_id"])
        else:
            executor = runtime_executor or _default_runtime_executor
            result = executor(
                study=_spec_data(spec),
                shard_dir=root,
                stage=manifest["stage_name"],
                payload={**copy.deepcopy(payload), "entry_id": entry["entry_id"]},
                data_root=Path(data_root).expanduser().resolve(),
                device=device,
                download=download,
            )
        publish_entry_completion(
            study_dir=root,
            manifest_path=manifest_path,
            entry_id=entry["entry_id"],
        )
        executions.append(
            {
                "entry_id": entry["entry_id"],
                "status": "complete",
                "result_status": result.get("status"),
            }
        )
    marker = publish_stage_completion(
        study_dir=root, manifest_path=manifest_path
    )
    return {
        "study_id": _spec_id(spec),
        "stage": manifest["stage_name"],
        "status": "complete",
        "entry_count": len(manifest["entries"]),
        "resumed_count": len(manifest["entries"]) - len(pending),
        "executions": executions,
        "completion_path": str(marker),
        "workers": 1,
        "execution_host": "trex",
    }


def run_aggregate_confirmations(
    *,
    aggregate_dir: str | Path,
    data_root: str | Path,
    device: str = "cuda",
    download: bool = False,
    provenance: Mapping[str, Any] | None = None,
    runtime_executor: RuntimeExecutor | None = None,
) -> dict[str, Any]:
    source = normalize_code_provenance(provenance or code_provenance())
    results = []
    for stage in AGGREGATE_STAGE_SEQUENCE:
        path = _stage_manifest_path(
            Path(aggregate_dir).expanduser().resolve(), stage
        )
        if not path.is_file():
            _manifest, path = publish_aggregate_stage_manifest(
                aggregate_dir, stage, provenance=source
            )
        results.append(
            execute_aggregate_stage(
                aggregate_dir=aggregate_dir,
                manifest_path=path,
                data_root=data_root,
                device=device,
                download=download,
                provenance=source,
                runtime_executor=runtime_executor,
            )
        )
    return {
        "status": "complete",
        "execution_host": "trex",
        "stages": results,
    }


def shard_status(shard_dir: str | Path) -> dict[str, Any]:
    root, spec, descriptor = load_host_shard(shard_dir)
    stages = []
    for stage in SCREEN_STAGE_SEQUENCE:
        manifest = _stage_manifest_path(root, stage)
        if not manifest.is_file():
            state = "not_planned"
            completed = 0
            total = None
        else:
            value = load_stage_manifest(manifest, study_dir=root)
            total = len(value["entries"])
            completed = sum(
                entry_is_complete(
                    study_dir=root,
                    manifest_path=manifest,
                    entry_id=entry["entry_id"],
                )
                for entry in value["entries"]
            )
            try:
                validate_stage_completion(study_dir=root, manifest_path=manifest)
            except (FileNotFoundError, ValueError):
                state = "running_or_pending"
            else:
                state = "complete"
        stages.append(
            {
                "stage": stage,
                "state": state,
                "completed_entries": completed,
                "total_entries": total,
            }
        )
    return {
        "study_id": _spec_id(spec),
        "host": descriptor["host"],
        "architecture": descriptor["architecture"],
        "shard_dir": str(root),
        "stages": stages,
    }


__all__ = [
    "AGGREGATE_STAGE_SEQUENCE",
    "ARCHITECTURE_HOSTS",
    "CANARY_STEPS",
    "CANDIDATE_STEPS",
    "HOST_ARCHITECTURES",
    "LONG_STEPS",
    "PD_CELL_ID_SCHEMA_VERSION",
    "PD_RUN_SCHEMA_VERSION",
    "PD_STUDY_SCHEMA_VERSION",
    "PUBLIC_STAGE_SEQUENCE",
    "SCREEN_STAGE_SEQUENCE",
    "create_host_shard",
    "execute_aggregate_stage",
    "execute_screen_manifest_entry",
    "execute_screen_stage",
    "host_rows",
    "host_surfaces",
    "load_host_shard",
    "merge_screen_shards",
    "plan_host_screen",
    "publish_aggregate_stage_manifest",
    "publish_screen_stage_manifest",
    "rho_cell_id",
    "run_aggregate_confirmations",
    "run_host_screen",
    "run_representative_preflight_canary",
    "shard_status",
    "surface_id",
]
