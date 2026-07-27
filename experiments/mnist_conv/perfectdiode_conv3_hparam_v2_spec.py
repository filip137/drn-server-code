"""Strict Conv3 perfect-diode LR-v2 materialization and planning contract.

The checked-in v2 file is a design template, not an executable study.  An
executable study can be materialized only from a hash-verified Conv3 T/K
``selection.json``.  The resolved study expands the already-tested Conv1/2 v1
contract, changes only the explicitly declared Conv3 successor fields, and
binds the whole T/K selection plus every row result, completion, and
operating-point-audit digest.

This module does not modify or reinterpret the immutable Conv1/2 v1 study.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .identity import canonical_json_bytes, sha256_file, sha256_json
from .io import atomic_write_bytes, atomic_write_json
from .perfectdiode_hparam_spec import (
    PerfectDiodeHparamStudySpec,
    candidate_grid,
)
from .perfectdiode_tk_spec import (
    CONV3_CONV_WEIGHTS,
    K_BOUNDARY_SENTINEL,
    K_GRID,
    K_REFERENCE,
    T_CORE_GRID,
    T_EXTENSION_GRID,
    T_REFERENCE,
    TK_MANIFEST_SCHEMA_VERSION,
    TK_SELECTION_SCHEMA_VERSION,
    PerfectDiodeTKStudySpec,
    _candidate_k_pass,
    _candidate_t_pass,
    select_k_measurements,
    select_t_measurements,
    tk_entry_id,
)


PD_CONV3_TEMPLATE_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-template/v2"
)
PD_CONV3_STUDY_SCHEMA_VERSION = "mnist-conv-perfectdiode-hparam-study/v2"
PD_CONV3_RUN_SCHEMA_VERSION = "mnist-conv-perfectdiode-hparam-run/v2"
PD_CONV3_STUDY_ID_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-study-id/v2"
)
PD_CONV3_SURFACE_MANIFEST_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-surface-manifest/v2"
)
LR_OPERATING_POINT_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-lr-operating-point/v1"
)
TK_T_MEASUREMENTS_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-tk-t-measurements/v1"
)
TK_K_MEASUREMENTS_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-tk-k-measurements/v1"
)
TK_ROW_RESULT_SCHEMA_VERSION = "mnist-conv-perfectdiode-tk-row-result/v1"
TK_OPERATING_POINT_AUDIT_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-tk-operating-point-audit/v1"
)
TK_ENVIRONMENT_SCHEMA_VERSION = "mnist-conv-perfectdiode-tk-environment/v1"
TK_ASSET_SCHEMA_VERSION = "mnist-conv-perfectdiode-tk-assets/v1"
PUBLIC_STAGE_SEQUENCE = (
    "audit",
    "import_tk",
    "optimizer_probe",
    "rho_canary_core",
    "rho_core_candidates",
    "select_core",
    "rho_canary_expansion",
    "rho_expansion_candidates",
    "select_expanded",
    "post_training_tk",
    "finalize_lr",
)
OPTIMIZER_ORDER = ("sgd", "adam")
SCHEME_ORDER = ("baseline", "ours", "legacy")
ALLOWED_HOSTS = ("main", "akibscomputer")
SHARED_ASSET_HASH_KEYS = (
    "train_indices_sha256",
    "validation_indices_sha256",
    "initialization_checkpoint_sha256",
    "initialization_tensor_sha256",
    "t_cohort_indices_sha256",
    "k_cohort_indices_sha256",
    "train_batch_order_epoch_1_sha256",
    "train_batch_order_epoch_2_sha256",
    "train_batch_order_epoch_3_sha256",
)
EXECUTION_AUTHORITY_KEYS = (
    "source_commit",
    "source_archive_sha256",
    "effective_code_fingerprint",
    "worker_launcher_sha256",
    "environment_contract_sha256",
    "host_environment_sha256s",
    "resolved_study_path",
    "resolved_study_sha256",
    "output_root",
)
FIXED_HIGH_RHO_CONV = (0.009, 0.027, 0.081)
FIXED_HIGH_RHO_DENSE = (0.03, 0.09, 0.27)
FIXED_HIGH_GRID = candidate_grid(FIXED_HIGH_RHO_CONV, FIXED_HIGH_RHO_DENSE)
CANARY_STEPS = 640
CANDIDATE_TOTAL_STEPS = 10_314
MAXIMUM_CELLS_PER_SURFACE = 16

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_TK_CONFIG = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv3_tk_ordinary_mnist_v1.json"
)
FROZEN_TK_CONFIG_SHA256 = (
    "bd5cc81c63ed4362aea3506a38e98b2f4a600ed6e75fbdc1d9b63ceb2b8a955e"
)
FROZEN_TK_STUDY_ID = (
    "tkstudy_407b09b9a5bb6127224ece23361c1f350a05b2e3bcc950a41cffd5d51744ef30"
)
TK_HOST_BY_SCHEME = {
    "baseline": "main",
    "ours": "akibscomputer",
    "legacy": "main",
}
DEFAULT_TEMPLATE = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv3_sgd_adam_hparam_ordinary_mnist_v2.template.json"
)
DEFAULT_OPERATING_POINT_CONTRACT = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv3_lr_operating_point_t8_k8_20260727_v1.json"
)
_FROZEN_TEMPLATE_SHA256 = (
    "5f81de60986c4c54328a1df0c41ec46c94261adc415ff50215a4f98dfe5fa211"
)
_FROZEN_OPERATING_POINT_CONTRACT_SHA256 = (
    "e8426635c0c3d36b6d19412a97fad9ef8a631b79cb758000b70e8f66f8d64276"
)
_FROZEN_BASE_CONFIG_SHA256 = (
    "699804e1f7c35f65f50656db4af0b120dd3a3d0f2fb95e3a17b65ebfe7b78535"
)


class PerfectDiodeConv3HparamValidationError(ValueError):
    """A v2 template, T/K handoff, resolved study, or route is invalid."""


def _error(
    path: str, expected: str, provided: Any
) -> PerfectDiodeConv3HparamValidationError:
    return PerfectDiodeConv3HparamValidationError(
        f"Expected {path} to be {expected}. Provided value: {provided!r}."
    )


def _strict_json_load(path: Path) -> Any:
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
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except OSError as exc:
        raise PerfectDiodeConv3HparamValidationError(
            f"Expected JSON input to be readable. Provided value: {path}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise PerfectDiodeConv3HparamValidationError(
            f"Expected {path} to contain strict JSON. Provided error: {exc}."
        ) from exc


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "a JSON object", value)
    return copy.deepcopy(dict(value))


def _sequence(value: Any, path: str, *, length: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise _error(path, "a JSON array", value)
    if length is not None and len(value) != length:
        raise _error(path, f"an array of length {length}", value)
    return copy.deepcopy(value)


def _positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _error(path, "an integer >= 1", value)
    return int(value)


def _finite_number(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise _error(path, "a finite number", value)
    return float(value)


def _sha256(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _error(path, "a lowercase SHA-256 digest", value)
    return value


def _commit(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _error(path, "a lowercase 40- or 64-character commit id", value)
    return value


def _relative_path(value: Any, path: str, *, expected: str | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise _error(path, "a non-empty traversal-free relative path", value)
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise _error(path, "a traversal-free relative path", value)
    normalized = relative.as_posix()
    if expected is not None and normalized != expected:
        raise _error(path, f"exactly {expected!r}", value)
    return normalized


def _artifact_path(selection_path: Path, value: Any, path: str) -> Path:
    if not isinstance(value, str) or not value:
        raise _error(path, "a non-empty relative artifact path", value)
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise _error(path, "a traversal-free relative artifact path", value)
    resolved = (selection_path.parent / relative).resolve()
    try:
        resolved.relative_to(selection_path.parent.resolve())
    except ValueError as exc:
        raise _error(path, "a path inside the T/K study root", value) from exc
    return resolved


def _require_artifact_sha(
    selection_path: Path,
    row: Mapping[str, Any],
    *,
    path_key: str,
    sha_key: str,
    row_path: str,
    verify_files: bool,
) -> str:
    expected = _sha256(row.get(sha_key), f"{row_path}.{sha_key}")
    artifact = _artifact_path(
        selection_path,
        row.get(path_key),
        f"{row_path}.{path_key}",
    )
    if verify_files:
        if not artifact.is_file():
            raise _error(
                f"{row_path}.{path_key}",
                "an existing regular artifact",
                str(artifact),
            )
        observed = sha256_file(artifact)
        if observed != expected:
            raise _error(
                f"{row_path}.{sha_key}",
                f"the artifact SHA-256 {expected}",
                observed,
            )
    return expected


def _exact_fields(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    path: str,
) -> None:
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise _error(
                f"{path}.{key}",
                f"exactly {expected_value!r}",
                value.get(key),
            )


def _load_relative_json(
    root: Path,
    relative_value: Any,
    path: str,
) -> tuple[Path, dict[str, Any]]:
    anchor = root / "selection.json"
    artifact = _artifact_path(anchor, relative_value, path)
    if not artifact.is_file():
        raise _error(path, "an existing regular artifact", str(artifact))
    return artifact, _mapping(_strict_json_load(artifact), path)


def _iteration_measurements(
    value: Any,
    *,
    key: str,
    path: str,
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    records = _sequence(value, path)
    order: list[int] = []
    by_iteration: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(records):
        record = _mapping(raw, f"{path}[{index}]")
        iteration = record.get(key)
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration in by_iteration
        ):
            raise _error(
                f"{path}[{index}].{key}",
                "a unique integer iteration count",
                iteration,
            )
        order.append(iteration)
        by_iteration[iteration] = record
    return order, by_iteration


def _validate_result_artifact_records(
    *,
    root: Path,
    entry_dir: Path,
    result: Mapping[str, Any],
    expected_paths: Sequence[str],
    path: str,
) -> None:
    records = _sequence(result.get("artifacts"), f"{path}.artifacts")
    expected_names = [
        Path(relative).relative_to(entry_dir).as_posix()
        for relative in expected_paths
    ]
    observed_names = [
        record.get("path") if isinstance(record, Mapping) else None
        for record in records
    ]
    if observed_names != expected_names:
        raise _error(
            f"{path}.artifacts",
            f"records in exact order for {expected_names!r}",
            records,
        )
    for index, (raw, relative) in enumerate(zip(records, expected_paths)):
        record = _mapping(raw, f"{path}.artifacts[{index}]")
        artifact = root / relative
        expected = {
            "path": expected_names[index],
            "sha256": sha256_file(artifact),
            "bytes": artifact.stat().st_size,
        }
        if record != expected:
            raise _error(
                f"{path}.artifacts[{index}]",
                f"exactly {expected!r}",
                record,
            )


def _validate_selected_audit_measurements(
    *,
    audit: Mapping[str, Any],
    row_path: str,
    selected_t: int,
    selected_k: int,
    t_extension_used: bool,
) -> None:
    t_measurement = _mapping(
        audit.get("t_measurement"),
        f"{row_path}.operating_point_audit.t_measurement",
    )
    t_passed, t_reasons, t_complete = _candidate_t_pass(
        t_measurement,
        expected_iteration=selected_t,
    )
    if not (t_complete and t_passed):
        raise _error(
            f"{row_path}.operating_point_audit.t_measurement",
            "a complete passing replay at selected T",
            t_reasons,
        )
    t64 = _mapping(
        audit.get("t64_sentinel_measurement"),
        f"{row_path}.operating_point_audit.t64_sentinel_measurement",
    )
    t64_passed, t64_reasons, t64_complete = _candidate_t_pass(
        t64,
        expected_iteration=T_REFERENCE,
    )
    if (
        not t64_complete
        or audit.get("t64_sentinel_passed") is not t64_passed
    ):
        raise _error(
            f"{row_path}.operating_point_audit.t64_sentinel_measurement",
            "a complete replay whose pass flag is recomputed exactly",
            {
                "reasons": t64_reasons,
                "recorded_passed": audit.get("t64_sentinel_passed"),
            },
        )
    if t_extension_used:
        t256 = _mapping(
            audit.get("t256_extension_sentinel_measurement"),
            f"{row_path}.operating_point_audit."
            "t256_extension_sentinel_measurement",
        )
        sentinel_passed, sentinel_reasons, sentinel_complete = (
            _candidate_t_pass(
                t256,
                expected_iteration=T_EXTENSION_GRID[-1],
            )
        )
        if (
            not sentinel_complete
            or not sentinel_passed
            or audit.get("t256_extension_sentinel_passed") is not True
        ):
            raise _error(
                f"{row_path}.operating_point_audit."
                "t256_extension_sentinel_measurement",
                "a complete passing T=256 extension sentinel",
                {
                    "reasons": sentinel_reasons,
                    "recorded_passed": audit.get(
                        "t256_extension_sentinel_passed"
                    ),
                },
            )
    elif (
        audit.get("t256_extension_sentinel_measurement") is not None
        or audit.get("t256_extension_sentinel_passed") is not None
        or not t64_passed
    ):
        raise _error(
            f"{row_path}.operating_point_audit T sentinel",
            "a passing T=64 core sentinel and no T=256 extension replay",
            {
                "t64_passed": t64_passed,
                "t256_measurement": audit.get(
                    "t256_extension_sentinel_measurement"
                ),
                "t256_passed": audit.get(
                    "t256_extension_sentinel_passed"
                ),
            },
        )
    k_measurement = _mapping(
        audit.get("k_measurement"),
        f"{row_path}.operating_point_audit.k_measurement",
    )
    expected_reference = (
        K_BOUNDARY_SENTINEL if selected_k == K_REFERENCE else K_REFERENCE
    )
    if (
        k_measurement.get("candidate_k") != selected_k
        or k_measurement.get("reference_k") != expected_reference
        or audit.get("k_audit_reference") != expected_reference
    ):
        raise _error(
            f"{row_path}.operating_point_audit.k_measurement",
            "the selected K against its exact required reference",
            {
                "candidate_k": k_measurement.get("candidate_k"),
                "reference_k": k_measurement.get("reference_k"),
                "k_audit_reference": audit.get("k_audit_reference"),
            },
        )
    k_passed, k_reasons, k_complete = _candidate_k_pass(
        k_measurement,
        CONV3_CONV_WEIGHTS,
    )
    if not (k_complete and k_passed):
        raise _error(
            f"{row_path}.operating_point_audit.k_measurement",
            "a complete passing selected-K gradient replay",
            k_reasons,
        )


def _validate_selected_audit(
    *,
    selection_path: Path,
    row: Mapping[str, Any],
    row_path: str,
    selected_t: int,
    selected_k: int,
    execution_environment_sha256: str,
    t_cohort_sha256: str,
    k_cohort_sha256: str,
) -> None:
    audit_path = _artifact_path(
        selection_path,
        row.get("operating_point_audit_path"),
        f"{row_path}.operating_point_audit_path",
    )
    audit = _mapping(_strict_json_load(audit_path), "operating-point audit")
    exact = {
        "schema_version": TK_OPERATING_POINT_AUDIT_SCHEMA_VERSION,
        "study_id": FROZEN_TK_STUDY_ID,
        "config_sha256": FROZEN_TK_CONFIG_SHA256,
        "entry_id": tk_entry_id(
            FROZEN_TK_STUDY_ID,
            {
                "row_id": row["row_id"],
                "scheme": row["scheme"],
            },
        ),
        "row_id": row["row_id"],
        "architecture": "conv3",
        "scheme": row["scheme"],
        "execution_backend": "tmux",
        "execution_host": row["execution_host"],
        "execution_environment_sha256": execution_environment_sha256,
        "official_test_read": False,
        "selected_t": selected_t,
        "selected_k": selected_k,
        "status": "passed",
        "passed": True,
        "fresh_replay_after_selection": True,
        "t_extension_used": row["t_extension_used"],
    }
    for key, expected in exact.items():
        if audit.get(key) != expected:
            raise _error(
                f"{row_path}.operating_point_audit.{key}",
                f"exactly {expected!r}",
                audit.get(key),
            )
    t_measurement = _mapping(
        audit.get("t_measurement"),
        f"{row_path}.operating_point_audit.t_measurement",
    )
    k_measurement = _mapping(
        audit.get("k_measurement"),
        f"{row_path}.operating_point_audit.k_measurement",
    )
    if t_measurement.get("cohort_source_indices_sha256") != t_cohort_sha256:
        raise _error(
            f"{row_path}.operating_point_audit.t_measurement."
            "cohort_source_indices_sha256",
            f"the row T-cohort SHA-256 {t_cohort_sha256}",
            t_measurement.get("cohort_source_indices_sha256"),
        )
    if k_measurement.get("cohort_source_indices_sha256") != k_cohort_sha256:
        raise _error(
            f"{row_path}.operating_point_audit.k_measurement."
            "cohort_source_indices_sha256",
            f"the row K-cohort SHA-256 {k_cohort_sha256}",
            k_measurement.get("cohort_source_indices_sha256"),
        )
    _validate_selected_audit_measurements(
        audit=audit,
        row_path=row_path,
        selected_t=selected_t,
        selected_k=selected_k,
        t_extension_used=bool(row["t_extension_used"]),
    )


def _validate_tk_row_evidence(
    *,
    root: Path,
    manifest_sha256: str,
    entry: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    frozen_row: Mapping[str, Any],
    asset_binding: Mapping[str, Any],
) -> dict[str, Any]:
    row_path = f"producer row {selection_row.get('row_id')!r}"
    outputs = _sequence(entry.get("outputs"), f"{row_path}.manifest.outputs")
    entry_dir = Path(str(entry["output_dir"]))
    expected_outputs = [
        f"{entry_dir.as_posix()}/environment.json",
        f"{entry_dir.as_posix()}/t_measurements.json",
        f"{entry_dir.as_posix()}/k_measurements.json",
        f"{entry_dir.as_posix()}/operating_point_audit.json",
        f"{entry_dir.as_posix()}/result.json",
    ]
    if outputs != expected_outputs:
        raise _error(
            f"{row_path}.manifest.outputs",
            f"exactly {expected_outputs!r}",
            outputs,
        )
    completion_relative = f"{entry_dir.as_posix()}/completion.json"
    if entry.get("completion_path") != completion_relative:
        raise _error(
            f"{row_path}.manifest.completion_path",
            f"exactly {completion_relative!r}",
            entry.get("completion_path"),
        )
    environment_path, environment = _load_relative_json(
        root,
        outputs[0],
        f"{row_path}.environment",
    )
    _exact_fields(
        environment,
        {
            "schema_version": TK_ENVIRONMENT_SCHEMA_VERSION,
            "execution_backend": "tmux",
            "execution_host": TK_HOST_BY_SCHEME[frozen_row["scheme"]],
        },
        f"{row_path}.environment",
    )
    environment_sha = sha256_file(environment_path)

    t_path, t_artifact = _load_relative_json(
        root,
        outputs[1],
        f"{row_path}.t_measurements",
    )
    common = {
        "study_id": FROZEN_TK_STUDY_ID,
        "config_sha256": FROZEN_TK_CONFIG_SHA256,
        "entry_id": entry["entry_id"],
        "row_id": frozen_row["row_id"],
        "architecture": "conv3",
        "scheme": frozen_row["scheme"],
        "official_test_read": False,
    }
    _exact_fields(
        t_artifact,
        {"schema_version": TK_T_MEASUREMENTS_SCHEMA_VERSION, **common},
        f"{row_path}.t_measurements",
    )
    t_order, t_by_iteration = _iteration_measurements(
        t_artifact.get("measurements"),
        key="iteration_count",
        path=f"{row_path}.t_measurements.measurements",
    )
    allowed_t_orders = (
        list(T_CORE_GRID),
        list(T_CORE_GRID + T_EXTENSION_GRID),
    )
    if t_order not in allowed_t_orders:
        raise _error(
            f"{row_path}.t_measurements.measurements",
            "the exact ordered core grid, optionally followed by the complete "
            "extension grid",
            t_order,
        )
    core_t_decision = select_t_measurements(
        {count: t_by_iteration[count] for count in T_CORE_GRID}
    )
    extension_present = t_order == list(T_CORE_GRID + T_EXTENSION_GRID)
    if extension_present and core_t_decision.get("status") != "needs_extension":
        raise _error(
            f"{row_path}.t_measurements.measurements",
            "the extension grid only after the exact core selector returns "
            "'needs_extension'",
            {
                "core_selector_status": core_t_decision.get("status"),
                "extension_present": True,
            },
        )
    recomputed_t = select_t_measurements(t_by_iteration)
    if (
        t_artifact.get("selection") != recomputed_t
        or recomputed_t.get("status") == "needs_extension"
    ):
        raise _error(
            f"{row_path}.t_measurements.selection",
            "the terminal selector output recomputed from every exact-grid "
            "residual measurement",
            t_artifact.get("selection"),
        )
    _exact_fields(
        t_artifact,
        {
            "cohort_source_indices_sha256": asset_binding[
                "t_cohort_indices_sha256"
            ],
            "initialization_checkpoint_sha256": asset_binding[
                "initialization_checkpoint_sha256"
            ],
        },
        f"{row_path}.t_measurements",
    )

    k_path, k_artifact = _load_relative_json(
        root,
        outputs[2],
        f"{row_path}.k_measurements",
    )
    _exact_fields(
        k_artifact,
        {"schema_version": TK_K_MEASUREMENTS_SCHEMA_VERSION, **common},
        f"{row_path}.k_measurements",
    )
    k_order, k_by_iteration = _iteration_measurements(
        k_artifact.get("measurements"),
        key="candidate_k",
        path=f"{row_path}.k_measurements.measurements",
    )
    selected_t = recomputed_t.get("selected_t")
    if recomputed_t.get("status") == "selected":
        if k_order != list(K_GRID):
            raise _error(
                f"{row_path}.k_measurements.measurements",
                f"the exact ordered K grid {list(K_GRID)!r}",
                k_order,
            )
        k_sentinel = k_artifact.get("k128_sentinel")
        if k_sentinel is not None:
            k_sentinel = _mapping(
                k_sentinel,
                f"{row_path}.k_measurements.k128_sentinel",
            )
        core_k_decision = select_k_measurements(k_by_iteration)
        if (
            (k_sentinel is not None)
            != (core_k_decision.get("status") == "needs_k128_sentinel")
        ):
            raise _error(
                f"{row_path}.k_measurements.k128_sentinel",
                "present if and only if the exact K core selector returns "
                "'needs_k128_sentinel'",
                {
                    "core_selector_status": core_k_decision.get("status"),
                    "sentinel_present": k_sentinel is not None,
                },
            )
        recomputed_k = select_k_measurements(
            k_by_iteration,
            k128_sentinel=k_sentinel,
        )
    else:
        if k_order or k_artifact.get("k128_sentinel") is not None:
            raise _error(
                f"{row_path}.k_measurements",
                "zero K measurements and no sentinel when T is unresolved",
                {
                    "candidate_order": k_order,
                    "k128_sentinel": k_artifact.get("k128_sentinel"),
                },
            )
        recomputed_k = {
            "status": "not_run_unresolved_t",
            "selected_k": None,
            "k128_sentinel_used": False,
            "candidates": [],
        }
    if k_artifact.get("selection") != recomputed_k:
        raise _error(
            f"{row_path}.k_measurements.selection",
            "the selector output recomputed from the exact K grid and "
            "conditional K=128 sentinel",
            k_artifact.get("selection"),
        )
    _exact_fields(
        k_artifact,
        {
            "selected_t": selected_t,
            "cohort_source_indices_sha256": asset_binding[
                "k_cohort_indices_sha256"
            ],
            "initialization_checkpoint_sha256": asset_binding[
                "initialization_checkpoint_sha256"
            ],
        },
        f"{row_path}.k_measurements",
    )

    audit_path, _audit = _load_relative_json(
        root,
        outputs[3],
        f"{row_path}.operating_point_audit",
    )
    result_path, result = _load_relative_json(
        root,
        outputs[4],
        f"{row_path}.result",
    )
    diagnostic_status = (
        "selected"
        if recomputed_t["status"] == "selected"
        and recomputed_k["status"] == "selected"
        else (
            str(recomputed_t["status"])
            if recomputed_t["status"] != "selected"
            else str(recomputed_k["status"])
        )
    )
    terminal_status = (
        "selected"
        if diagnostic_status == "selected"
        and result.get("operating_point_audit_passed") is True
        else (
            "unresolved_operating_point_audit"
            if diagnostic_status == "selected"
            else diagnostic_status
        )
    )
    _exact_fields(
        result,
        {
            "schema_version": TK_ROW_RESULT_SCHEMA_VERSION,
            **common,
            "manifest_sha256": manifest_sha256,
            "execution_backend": "tmux",
            "execution_host": TK_HOST_BY_SCHEME[frozen_row["scheme"]],
            "execution_environment_sha256": environment_sha,
            "run_name": frozen_row["run_name"],
            "voltage_amp": frozen_row["voltage_amp"],
            "current_amp": frozen_row["current_amp"],
            "input_gain": frozen_row["input_gain"],
            "status": terminal_status,
            "diagnostic_selection_status": diagnostic_status,
            "diagnostic_selected_t": recomputed_t.get("selected_t"),
            "diagnostic_selected_k": recomputed_k.get("selected_k"),
            "selected_t": (
                recomputed_t.get("selected_t")
                if terminal_status == "selected"
                else None
            ),
            "selected_k": (
                recomputed_k.get("selected_k")
                if terminal_status == "selected"
                else None
            ),
            "t_reference": T_REFERENCE,
            "k_reference": K_REFERENCE,
            "t_extension_used": recomputed_t.get("extension_used"),
            "k128_sentinel_used": recomputed_k.get(
                "k128_sentinel_used"
            ),
            "initialization_checkpoint_sha256": asset_binding[
                "initialization_checkpoint_sha256"
            ],
            "initialization_tensor_sha256": asset_binding[
                "initialization_tensor_sha256"
            ],
            "train_indices_sha256": asset_binding[
                "train_indices_sha256"
            ],
            "validation_indices_sha256": asset_binding[
                "validation_indices_sha256"
            ],
            "t_cohort_indices_sha256": asset_binding[
                "t_cohort_indices_sha256"
            ],
            "k_cohort_indices_sha256": asset_binding[
                "k_cohort_indices_sha256"
            ],
            "operating_point_audit_path": "operating_point_audit.json",
            "operating_point_audit_sha256": sha256_file(audit_path),
            "t_selection": recomputed_t,
            "k_selection": recomputed_k,
        },
        f"{row_path}.result",
    )
    if (
        selection_row.get("status") != terminal_status
        or selection_row.get("selected_t") != result.get("selected_t")
        or selection_row.get("selected_k") != result.get("selected_k")
        or selection_row.get("t_extension_used")
        is not result.get("t_extension_used")
        or selection_row.get("k128_sentinel_used")
        is not result.get("k128_sentinel_used")
    ):
        raise _error(
            row_path,
            "selection fields reproduced exactly from residual and gradient "
            "candidate artifacts",
            {
                "selection_status": selection_row.get("status"),
                "result_status": terminal_status,
                "selection_t": selection_row.get("selected_t"),
                "result_t": result.get("selected_t"),
                "selection_k": selection_row.get("selected_k"),
                "result_k": result.get("selected_k"),
                "selection_t_extension": selection_row.get(
                    "t_extension_used"
                ),
                "result_t_extension": result.get("t_extension_used"),
                "selection_k128": selection_row.get(
                    "k128_sentinel_used"
                ),
                "result_k128": result.get("k128_sentinel_used"),
            },
        )
    _validate_result_artifact_records(
        root=root,
        entry_dir=entry_dir,
        result=result,
        expected_paths=outputs[1:4],
        path=f"{row_path}.result",
    )
    return {
        "execution_environment_path": outputs[0],
        "execution_environment_sha256": environment_sha,
        "t_measurements_path": outputs[1],
        "t_measurements_sha256": sha256_file(t_path),
        "k_measurements_path": outputs[2],
        "k_measurements_sha256": sha256_file(k_path),
        "operating_point_audit_path": outputs[3],
        "operating_point_audit_sha256": sha256_file(audit_path),
        "result_path": outputs[4],
        "result_sha256": sha256_file(result_path),
        "completion_path": completion_relative,
        "completion_sha256": sha256_file(root / completion_relative),
    }


def _validate_tk_producer_bundle(
    *,
    selection_path: Path,
    selection: Mapping[str, Any],
    template: Mapping[str, Any],
    shared_assets_path: Path | None,
    source_archive_path: Path | None,
) -> dict[str, Any]:
    """Validate one self-contained, terminal T/K producer study root."""

    root = selection_path.parent.resolve()
    if selection_path.name != "selection.json":
        raise _error(
            "upstream T/K selection path",
            "a file named 'selection.json' at the producer study root",
            str(selection_path),
        )
    manifest_path, manifest = _load_relative_json(
        root,
        "manifest.json",
        "producer manifest",
    )
    resolved_path, _resolved = _load_relative_json(
        root,
        "resolved_config.json",
        "producer resolved config",
    )
    frozen_spec = PerfectDiodeTKStudySpec.from_path(resolved_path)
    if (
        frozen_spec.study_id != FROZEN_TK_STUDY_ID
        or frozen_spec.config_sha256 != FROZEN_TK_CONFIG_SHA256
    ):
        raise _error(
            "producer resolved config identity",
            f"exactly study={FROZEN_TK_STUDY_ID!r}, "
            f"config_sha256={FROZEN_TK_CONFIG_SHA256!r}",
            {
                "study_id": frozen_spec.study_id,
                "config_sha256": frozen_spec.config_sha256,
            },
        )
    manifest_sha = sha256_file(manifest_path)
    _exact_fields(
        manifest,
        {
            "schema_version": TK_MANIFEST_SCHEMA_VERSION,
            "study_id": frozen_spec.study_id,
            "config_path": "resolved_config.json",
            "config_sha256": frozen_spec.config_sha256,
            "resolved_config_file_sha256": sha256_file(resolved_path),
            "entry_count": 3,
            "selection_path": "selection.json",
        },
        "producer manifest",
    )
    _sha256(
        manifest.get("launcher_sha256"),
        "producer manifest.launcher_sha256",
    )
    staged_source = _mapping(
        manifest.get("staged_source"),
        "producer manifest.staged_source",
    )
    if set(staged_source) != {
        "commit",
        "archive_sha256",
        "effective_code_fingerprint",
    }:
        raise _error(
            "producer manifest.staged_source",
            "exactly commit, archive_sha256, and effective_code_fingerprint",
            sorted(staged_source),
        )
    _commit(staged_source.get("commit"), "producer manifest.staged_source.commit")
    for key in ("archive_sha256", "effective_code_fingerprint"):
        _sha256(
            staged_source.get(key),
            f"producer manifest.staged_source.{key}",
        )
    source_archive = (
        (root / "source" / "source.tar").resolve()
        if source_archive_path is None
        else Path(source_archive_path).expanduser().resolve()
    )
    if (
        not source_archive.is_file()
        or sha256_file(source_archive) != staged_source["archive_sha256"]
    ):
        raise _error(
            "upstream T/K producer source archive",
            "an existing archive with the exact manifest-staged SHA-256 "
            f"{staged_source['archive_sha256']}",
            str(source_archive),
        )
    code_provenance = _mapping(
        manifest.get("code_provenance"),
        "producer manifest.code_provenance",
    )
    _exact_fields(
        code_provenance,
        {
            "git_revision": staged_source["commit"],
            "dirty_source_digest": None,
            "effective_code_fingerprint": staged_source[
                "effective_code_fingerprint"
            ],
        },
        "producer manifest.code_provenance",
    )
    if selection.get("manifest_sha256") != manifest_sha:
        raise _error(
            "selection.manifest_sha256",
            f"the self-contained producer manifest SHA-256 {manifest_sha}",
            selection.get("manifest_sha256"),
        )
    asset_binding = _mapping(
        manifest.get("asset_binding"),
        "producer manifest.asset_binding",
    )
    expected_asset_keys = {
        "schema_version",
        "study_id",
        "config_sha256",
        "assets_sha256",
        "completion_sha256",
        "initialization_checkpoint_sha256",
        "initialization_tensor_sha256",
        "train_indices_sha256",
        "validation_indices_sha256",
        "t_cohort_indices_sha256",
        "k_cohort_indices_sha256",
    }
    if set(asset_binding) != expected_asset_keys:
        raise _error(
            "producer manifest.asset_binding",
            f"exactly {sorted(expected_asset_keys)!r}",
            sorted(asset_binding),
        )
    _exact_fields(
        asset_binding,
        {
            "schema_version": TK_ASSET_SCHEMA_VERSION,
            "study_id": frozen_spec.study_id,
            "config_sha256": frozen_spec.config_sha256,
        },
        "producer manifest.asset_binding",
    )
    for key in expected_asset_keys - {
        "schema_version",
        "study_id",
        "config_sha256",
    }:
        _sha256(
            asset_binding.get(key),
            f"producer manifest.asset_binding.{key}",
        )
    if selection.get("artifact_sha256s", {}).get("asset_bundle") != asset_binding:
        raise _error(
            "selection.artifact_sha256s.asset_bundle",
            "the exact producer manifest asset binding",
            selection.get("artifact_sha256s", {}).get("asset_bundle"),
        )
    asset_root = (
        (root / "shared_assets").resolve()
        if shared_assets_path is None
        else Path(shared_assets_path).expanduser().resolve()
    )
    try:
        from .perfectdiode_tk_runtime import validate_shared_assets

        validated_assets = validate_shared_assets(frozen_spec, asset_root)
    except Exception as exc:
        raise _error(
            "upstream T/K shared assets",
            "a complete hash-verified producer asset bundle",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    observed_asset_binding = {
        "schema_version": validated_assets["schema_version"],
        "study_id": validated_assets["study_id"],
        "config_sha256": validated_assets["config_sha256"],
        "assets_sha256": validated_assets["assets_sha256"],
        "completion_sha256": validated_assets["completion_sha256"],
        "initialization_checkpoint_sha256": validated_assets[
            "initialization"
        ]["sha256"],
        "initialization_tensor_sha256": validated_assets[
            "initialization"
        ]["parameter_tensor_sha256"],
        "train_indices_sha256": validated_assets["split"][
            "train_indices_sha256"
        ],
        "validation_indices_sha256": validated_assets["split"][
            "validation_indices_sha256"
        ],
        "t_cohort_indices_sha256": validated_assets["cohorts"]["t"][
            "source_indices_sha256"
        ],
        "k_cohort_indices_sha256": validated_assets["cohorts"]["k"][
            "source_indices_sha256"
        ],
    }
    if observed_asset_binding != asset_binding:
        raise _error(
            "producer manifest.asset_binding",
            "the exact binding rederived from the full shared-asset bundle",
            {
                "manifest": asset_binding,
                "observed": observed_asset_binding,
            },
        )
    entries = _sequence(
        manifest.get("entries"),
        "producer manifest.entries",
        length=3,
    )
    selection_rows = _sequence(
        selection.get("rows"),
        "selection.rows",
        length=3,
    )
    row_evidence: dict[str, dict[str, Any]] = {}
    bundle_paths = {
        "selection.json",
        "manifest.json",
        "resolved_config.json",
    }
    for index, (raw_entry, row, raw_selection_row) in enumerate(
        zip(entries, frozen_spec.rows, selection_rows)
    ):
        entry = _mapping(raw_entry, f"producer manifest.entries[{index}]")
        selection_row = _mapping(
            raw_selection_row,
            f"selection.rows[{index}]",
        )
        expected_entry_id = tk_entry_id(frozen_spec.study_id, row)
        expected_lane = TK_HOST_BY_SCHEME[row["scheme"]]
        expected_entry_dir = f"entries/{expected_entry_id}"
        _exact_fields(
            entry,
            {
                "entry_index": index,
                "entry_id": expected_entry_id,
                "row_id": row["row_id"],
                "scheme": row["scheme"],
                "architecture": "conv3",
                "output_dir": expected_entry_dir,
                "completion_path": f"{expected_entry_dir}/completion.json",
            },
            f"producer manifest.entries[{index}]",
        )
        _exact_fields(
            _mapping(
                entry.get("execution"),
                f"producer manifest.entries[{index}].execution",
            ),
            {
                "backend": "tmux",
                "lane": expected_lane,
                "whole_row_on_one_lane": True,
            },
            f"producer manifest.entries[{index}].execution",
        )
        if selection_row.get("execution_host") != expected_lane:
            raise _error(
                f"selection.rows[{index}].execution_host",
                f"exactly {expected_lane!r} for {row['scheme']!r}",
                selection_row.get("execution_host"),
            )
        evidence = _validate_tk_row_evidence(
            root=root,
            manifest_sha256=manifest_sha,
            entry=entry,
            selection_row=selection_row,
            frozen_row=row,
            asset_binding=asset_binding,
        )
        expected_selection_paths = {
            "execution_environment_path": evidence[
                "execution_environment_path"
            ],
            "result_path": evidence["result_path"],
            "completion_path": evidence["completion_path"],
            "operating_point_audit_path": evidence[
                "operating_point_audit_path"
            ],
        }
        _exact_fields(
            selection_row,
            expected_selection_paths,
            f"selection.rows[{index}]",
        )
        for key in (
            "execution_environment_sha256",
            "result_sha256",
            "completion_sha256",
            "operating_point_audit_sha256",
        ):
            if selection_row.get(key) != evidence[key]:
                raise _error(
                    f"selection.rows[{index}].{key}",
                    f"the producer artifact SHA-256 {evidence[key]}",
                    selection_row.get(key),
                )
        row_evidence[row["row_id"]] = evidence
        bundle_paths.update(evidence[key] for key in (
            "execution_environment_path",
            "t_measurements_path",
            "k_measurements_path",
            "operating_point_audit_path",
            "result_path",
            "completion_path",
        ))

    # Reuse the producer's strict completion validator.  Call it only with
    # paths rooted under the copied producer study: unlike the producer's
    # top-level status helper, this does not compare against a launcher path
    # in the current checkout.
    try:
        from experiments.run_mnist_conv_perfectdiode_tk import (
            _validate_entry_completion,
        )

        producer_completions = [
            _validate_entry_completion(
                root,
                manifest,
                _mapping(
                    raw_entry,
                    f"producer manifest.entries[{index}]",
                ),
                manifest_sha256=manifest_sha,
            )
            for index, raw_entry in enumerate(entries)
        ]
    except Exception as exc:
        raise _error(
            "upstream T/K producer bundle",
            "a terminal bundle accepted by the T/K producer validator",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    if len(producer_completions) != 3:
        raise _error(
            "upstream T/K producer bundle",
            "exactly three producer-validated completion/result pairs",
            len(producer_completions),
        )
    expected_template_tk = _mapping(
        template.get("upstream_tk"),
        "template.upstream_tk",
    )
    _exact_fields(
        expected_template_tk,
        {
            "required_study_id": frozen_spec.study_id,
            "required_config_sha256": frozen_spec.config_sha256,
            "manifest_path": "upstream_tk/manifest.json",
            "resolved_config_path": "upstream_tk/resolved_config.json",
            "shared_assets_path": "upstream_tk/shared_assets",
            "producer_shared_assets_dirname": "shared_assets",
            "source_archive_path": "upstream_tk/source/source.tar",
            "require_self_contained_producer_bundle": True,
            "require_exact_candidate_grids_and_sentinels": True,
            "exact_execution_hosts": TK_HOST_BY_SCHEME,
        },
        "template.upstream_tk",
    )
    return {
        "manifest_sha256": manifest_sha,
        "resolved_config_file_sha256": sha256_file(resolved_path),
        "asset_binding": asset_binding,
        "row_evidence": row_evidence,
        "bundle_relative_paths": sorted(
            bundle_paths
            | {
                "shared_assets/initialization.pt",
                "shared_assets/assets.json",
                "shared_assets/completion.json",
                "source/source.tar",
            }
        ),
        "staged_source": staged_source,
    }


def load_frozen_template(
    path: str | Path = DEFAULT_TEMPLATE,
) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve()
    data = _mapping(_strict_json_load(source), "template")
    digest = sha256_json(data)
    if digest != _FROZEN_TEMPLATE_SHA256:
        raise _error(
            "template",
            "the immutable Conv3 perfect-diode v2 template with canonical "
            f"SHA-256 {_FROZEN_TEMPLATE_SHA256}",
            digest,
        )
    if data.get("schema_version") != PD_CONV3_TEMPLATE_SCHEMA_VERSION:
        raise _error(
            "template.schema_version",
            f"exactly {PD_CONV3_TEMPLATE_SCHEMA_VERSION!r}",
            data.get("schema_version"),
        )
    base = _mapping(data.get("base_contract"), "template.base_contract")
    if base.get("canonical_sha256") != _FROZEN_BASE_CONFIG_SHA256:
        raise _error(
            "template.base_contract.canonical_sha256",
            f"exactly {_FROZEN_BASE_CONFIG_SHA256}",
            base.get("canonical_sha256"),
        )
    return data, source


def load_frozen_operating_point_contract(
    path: str | Path = DEFAULT_OPERATING_POINT_CONTRACT,
) -> tuple[dict[str, Any], Path]:
    """Load the user-fixed LR operating point without rewriting T/K evidence."""

    source = Path(path).expanduser().resolve()
    data = _mapping(_strict_json_load(source), "operating-point contract")
    digest = sha256_json(data)
    if digest != _FROZEN_OPERATING_POINT_CONTRACT_SHA256:
        raise _error(
            "operating-point contract",
            "the immutable user-fixed T=8, K=8 contract with canonical "
            f"SHA-256 {_FROZEN_OPERATING_POINT_CONTRACT_SHA256}",
            digest,
        )
    expected = {
        "schema_version": LR_OPERATING_POINT_SCHEMA_VERSION,
        "mode": "user_fixed_after_residual_gradient_review",
        "inference_iterations": 8,
        "training_iterations": 8,
        "reference_inference_iterations": 64,
        "reference_training_iterations": 64,
        "shared_across_schemes": True,
        "retain_upstream_diagnostic_selection": True,
        "require_fresh_manifest_bound_preflight_gate": True,
        "provenance": "user_directed_2026-07-27",
    }
    if data != expected:
        raise _error(
            "operating-point contract",
            f"exactly {expected!r}",
            data,
        )
    return data, source


def _load_frozen_base(
    template: Mapping[str, Any], template_path: Path
) -> dict[str, Any]:
    base_contract = _mapping(
        template.get("base_contract"), "template.base_contract"
    )
    relative = Path(str(base_contract.get("path")))
    if relative.is_absolute() or ".." in relative.parts:
        raise _error(
            "template.base_contract.path",
            "a traversal-free path beside the template",
            str(relative),
        )
    source = (template_path.parent / relative).resolve()
    data = _mapping(_strict_json_load(source), "base config")
    digest = sha256_json(data)
    if digest != _FROZEN_BASE_CONFIG_SHA256:
        raise _error(
            "base config",
            f"canonical SHA-256 {_FROZEN_BASE_CONFIG_SHA256}",
            digest,
        )
    # The frozen v1 validator is an additional independent guard.
    PerfectDiodeHparamStudySpec.from_dict(data)
    return data


def validate_tk_selection(
    selection: Mapping[str, Any],
    *,
    selection_path: str | Path,
    expected_sha256: str,
    verify_artifact_files: bool,
    template: Mapping[str, Any],
    shared_assets_path: str | Path | None = None,
    source_archive_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the complete upstream handoff and return normalized bindings."""

    if verify_artifact_files is not True:
        raise _error(
            "verify_artifact_files",
            "exactly true for the fail-closed LR-v2 T/K consumer",
            verify_artifact_files,
        )
    source = Path(selection_path).expanduser().resolve()
    expected_selection_sha = _sha256(
        expected_sha256, "upstream T/K selection expected_sha256"
    )
    if not source.is_file():
        raise _error(
            "upstream T/K selection path",
            "an existing regular file",
            str(source),
        )
    observed_selection_sha = sha256_file(source)
    if observed_selection_sha != expected_selection_sha:
        raise _error(
            "upstream T/K selection SHA-256",
            expected_selection_sha,
            observed_selection_sha,
        )
    data = _mapping(selection, "upstream T/K selection")
    file_data = _mapping(
        _strict_json_load(source),
        "upstream T/K selection file",
    )
    if data != file_data:
        raise _error(
            "upstream T/K selection",
            "the exact object stored in selection_path",
            sha256_json(data),
        )
    if data.get("schema_version") != TK_SELECTION_SCHEMA_VERSION:
        raise _error(
            "selection.schema_version",
            f"exactly {TK_SELECTION_SCHEMA_VERSION!r}",
            data.get("schema_version"),
        )
    if data.get("official_test_read") is not False:
        raise _error(
            "selection.official_test_read", "exactly false", data.get("official_test_read")
        )
    frozen_tk_spec = PerfectDiodeTKStudySpec.from_path(FROZEN_TK_CONFIG)
    if (
        frozen_tk_spec.study_id != FROZEN_TK_STUDY_ID
        or frozen_tk_spec.config_sha256 != FROZEN_TK_CONFIG_SHA256
    ):
        raise _error(
            "frozen T/K authority",
            f"exactly study={FROZEN_TK_STUDY_ID!r}, "
            f"config_sha256={FROZEN_TK_CONFIG_SHA256!r}",
            {
                "study_id": frozen_tk_spec.study_id,
                "config_sha256": frozen_tk_spec.config_sha256,
            },
        )
    expected_upstream = _mapping(
        template.get("upstream_tk"),
        "template.upstream_tk",
    )
    _exact_fields(
        expected_upstream,
        {
            "required_schema_version": TK_SELECTION_SCHEMA_VERSION,
            "required_study_id": FROZEN_TK_STUDY_ID,
            "required_config_sha256": FROZEN_TK_CONFIG_SHA256,
            "selection_path": "upstream_tk/selection.json",
            "manifest_path": "upstream_tk/manifest.json",
            "resolved_config_path": "upstream_tk/resolved_config.json",
            "shared_assets_path": "upstream_tk/shared_assets",
            "producer_shared_assets_dirname": "shared_assets",
            "source_archive_path": "upstream_tk/source/source.tar",
            "require_self_contained_producer_bundle": True,
            "require_exact_candidate_grids_and_sentinels": True,
            "exact_execution_hosts": TK_HOST_BY_SCHEME,
        },
        "template.upstream_tk",
    )
    _exact_fields(
        data,
        {
            "study_id": FROZEN_TK_STUDY_ID,
            "config_sha256": FROZEN_TK_CONFIG_SHA256,
        },
        "selection",
    )
    _sha256(data.get("manifest_sha256"), "selection.manifest_sha256")
    study_id = FROZEN_TK_STUDY_ID

    templates = _sequence(
        template.get("row_templates"), "template.row_templates", length=3
    )
    rows = _sequence(data.get("rows"), "selection.rows", length=3)
    normalized_rows: list[dict[str, Any]] = []
    selected_count = 0
    for index, (raw_row, raw_expected) in enumerate(zip(rows, templates)):
        path = f"selection.rows[{index}]"
        row = _mapping(raw_row, path)
        expected = _mapping(raw_expected, f"template.row_templates[{index}]")
        for key in (
            "row_id",
            "architecture",
            "scheme",
            "run_name",
            "voltage_amp",
            "current_amp",
        ):
            if row.get(key) != expected.get(key):
                raise _error(
                    f"{path}.{key}",
                    f"exactly {expected.get(key)!r}",
                    row.get(key),
                )
        if _finite_number(row.get("input_gain"), f"{path}.input_gain") != 360.0:
            raise _error(f"{path}.input_gain", "exactly 360.0", row.get("input_gain"))
        if row.get("t_reference") != 64 or row.get("k_reference") != 64:
            raise _error(
                path,
                "T/K reference values exactly 64/64",
                {
                    "t_reference": row.get("t_reference"),
                    "k_reference": row.get("k_reference"),
                },
            )
        for flag in ("t_extension_used", "k128_sentinel_used"):
            if not isinstance(row.get(flag), bool):
                raise _error(
                    f"{path}.{flag}", "a boolean", row.get(flag)
                )
        execution_host = row.get("execution_host")
        expected_host = TK_HOST_BY_SCHEME[row["scheme"]]
        if execution_host != expected_host:
            raise _error(
                f"{path}.execution_host",
                f"exactly {expected_host!r} for {row['scheme']!r}",
                execution_host,
            )
        if row.get("execution_backend") != "tmux":
            raise _error(
                f"{path}.execution_backend",
                "exactly 'tmux'",
                row.get("execution_backend"),
            )
        tk_environment_sha = _sha256(
            row.get("execution_environment_sha256"),
            f"{path}.execution_environment_sha256",
        )
        t_cohort_sha = _sha256(
            row.get("t_cohort_indices_sha256"),
            f"{path}.t_cohort_indices_sha256",
        )
        k_cohort_sha = _sha256(
            row.get("k_cohort_indices_sha256"),
            f"{path}.k_cohort_indices_sha256",
        )
        result_sha = _require_artifact_sha(
            source,
            row,
            path_key="result_path",
            sha_key="result_sha256",
            row_path=path,
            verify_files=verify_artifact_files,
        )
        completion_sha = _require_artifact_sha(
            source,
            row,
            path_key="completion_path",
            sha_key="completion_sha256",
            row_path=path,
            verify_files=verify_artifact_files,
        )
        selected_t: int | None
        selected_k: int | None
        audit_sha: str | None
        zero_work_reason: str | None
        status = row.get("status")
        if status == "selected":
            selected_t = _positive_integer(row.get("selected_t"), f"{path}.selected_t")
            selected_k = _positive_integer(row.get("selected_k"), f"{path}.selected_k")
            if row.get("operating_point_audit_passed") is not True:
                raise _error(
                    f"{path}.operating_point_audit_passed",
                    "exactly true for a selected row",
                    row.get("operating_point_audit_passed"),
                )
            audit_sha = _require_artifact_sha(
                source,
                row,
                path_key="operating_point_audit_path",
                sha_key="operating_point_audit_sha256",
                row_path=path,
                verify_files=verify_artifact_files,
            )
            if verify_artifact_files:
                _validate_selected_audit(
                    selection_path=source,
                    row=row,
                    row_path=path,
                    selected_t=selected_t,
                    selected_k=selected_k,
                    execution_environment_sha256=tk_environment_sha,
                    t_cohort_sha256=t_cohort_sha,
                    k_cohort_sha256=k_cohort_sha,
                )
            eligible = True
            zero_work_reason = None
            selected_count += 1
        elif isinstance(status, str) and status.startswith("unresolved_"):
            if row.get("selected_t") is not None or row.get("selected_k") is not None:
                raise _error(
                    path,
                    "null selected_t/selected_k for an unresolved row",
                    {
                        "selected_t": row.get("selected_t"),
                        "selected_k": row.get("selected_k"),
                    },
                )
            selected_t = None
            selected_k = None
            audit_sha = (
                _sha256(
                    row.get("operating_point_audit_sha256"),
                    f"{path}.operating_point_audit_sha256",
                )
                if row.get("operating_point_audit_sha256") is not None
                else None
            )
            eligible = False
            zero_work_reason = f"upstream_tk_{status}"
        else:
            raise _error(
                f"{path}.status",
                "'selected' or an explicit 'unresolved_*' status",
                status,
            )
        normalized_rows.append(
            {
                "row_id": row["row_id"],
                "scheme": row["scheme"],
                "status": status,
                "selected_t": selected_t,
                "selected_k": selected_k,
                "t_reference": 64,
                "k_reference": 64,
                "t_extension_used": row["t_extension_used"],
                "k128_sentinel_used": row["k128_sentinel_used"],
                "execution_host": execution_host,
                "execution_backend": "tmux",
                "execution_environment_sha256": tk_environment_sha,
                "t_cohort_indices_sha256": t_cohort_sha,
                "k_cohort_indices_sha256": k_cohort_sha,
                "result_sha256": result_sha,
                "completion_sha256": completion_sha,
                "operating_point_audit_sha256": audit_sha,
                "operating_point_audit_passed": row.get(
                    "operating_point_audit_passed"
                )
                is True,
                "lr_eligible": eligible,
                "zero_work_reason": zero_work_reason,
            }
        )
    expected_global = "selected" if selected_count == 3 else "unresolved"
    if data.get("status") != expected_global:
        raise _error(
            "selection.status",
            f"exactly {expected_global!r} for {selected_count}/3 selected rows",
            data.get("status"),
        )
    artifact_sha256s = _mapping(
        data.get("artifact_sha256s"), "selection.artifact_sha256s"
    )
    if set(artifact_sha256s) != {"asset_bundle", "rows"}:
        raise _error(
            "selection.artifact_sha256s",
            "exactly {'asset_bundle','rows'}",
            sorted(artifact_sha256s),
        )
    _mapping(
        artifact_sha256s["asset_bundle"],
        "selection.artifact_sha256s.asset_bundle",
    )
    row_hashes = _mapping(
        artifact_sha256s["rows"],
        "selection.artifact_sha256s.rows",
    )
    expected_row_ids = {row["row_id"] for row in normalized_rows}
    if set(row_hashes) != expected_row_ids:
        raise _error(
            "selection.artifact_sha256s.rows",
            "exactly one record for every selection row",
            sorted(row_hashes),
        )
    for row in normalized_rows:
        hashes = _mapping(
            row_hashes[row["row_id"]],
            f"selection.artifact_sha256s.rows.{row['row_id']}",
        )
        expected_hashes = {
            "result_sha256": row["result_sha256"],
            "completion_sha256": row["completion_sha256"],
            "operating_point_audit_sha256": row[
                "operating_point_audit_sha256"
            ],
            "execution_environment_sha256": row[
                "execution_environment_sha256"
            ],
        }
        if hashes != expected_hashes:
            raise _error(
                f"selection.artifact_sha256s.rows.{row['row_id']}",
                f"exactly {expected_hashes!r}",
                hashes,
            )
    producer = _validate_tk_producer_bundle(
        selection_path=source,
        selection=data,
        template=template,
        shared_assets_path=(
            None
            if shared_assets_path is None
                else Path(shared_assets_path).expanduser().resolve()
        ),
        source_archive_path=(
            None
            if source_archive_path is None
            else Path(source_archive_path).expanduser().resolve()
        ),
    )
    for row in normalized_rows:
        evidence = producer["row_evidence"][row["row_id"]]
        for key in (
            "execution_environment_sha256",
            "result_sha256",
            "completion_sha256",
            "operating_point_audit_sha256",
        ):
            if row[key] != evidence[key]:
                raise _error(
                    f"selection row {row['row_id']}.{key}",
                    f"the strict producer evidence SHA-256 {evidence[key]}",
                    row[key],
                )
        row.update(copy.deepcopy(evidence))
    return {
        "schema_version": TK_SELECTION_SCHEMA_VERSION,
        "study_id": study_id,
        "config_sha256": data["config_sha256"],
        "manifest_sha256": producer["manifest_sha256"],
        "resolved_config_file_sha256": producer[
            "resolved_config_file_sha256"
        ],
        "selection_sha256": observed_selection_sha,
        "status": expected_global,
        "official_test_read": False,
        "rows": normalized_rows,
        "artifact_sha256s": {
            "asset_bundle": copy.deepcopy(producer["asset_binding"]),
            "rows": copy.deepcopy(artifact_sha256s["rows"]),
        },
        "producer_bundle_relative_paths": copy.deepcopy(
            producer["bundle_relative_paths"]
        ),
        "producer_staged_source": copy.deepcopy(
            producer["staged_source"]
        ),
    }


def _materialized_rows(
    template: Mapping[str, Any],
    binding: Mapping[str, Any],
    operating_point: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    by_row_id = {row["row_id"]: row for row in binding["rows"]}
    result = []
    for raw in template["row_templates"]:
        row = copy.deepcopy(raw)
        upstream = copy.deepcopy(by_row_id[row["row_id"]])
        eligible = bool(upstream["lr_eligible"])
        row.update(
            {
                "input_gain": 360.0,
                "inference_iterations": (
                    int(operating_point["inference_iterations"])
                    if operating_point is not None and eligible
                    else upstream["selected_t"]
                ),
                "training_iterations": (
                    int(operating_point["training_iterations"])
                    if operating_point is not None and eligible
                    else upstream["selected_k"]
                ),
                "reference_inference_iterations": 64,
                "reference_training_iterations": 64,
                "lr_eligible": eligible,
                "zero_work_reason": upstream["zero_work_reason"],
                "upstream_tk": upstream,
            }
        )
        result.append(row)
    return result


def materialized_study_data(
    *,
    template: Mapping[str, Any],
    base: Mapping[str, Any],
    tk_binding: Mapping[str, Any],
    operating_point: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Expand the v2 design and upstream T/K handoff into runtime-ready data."""

    data = copy.deepcopy(dict(base))
    contract = template["study_contract"]
    data.update(
        {
            "schema_version": contract["schema_version"],
            "run_schema_version": contract["run_schema_version"],
            "name": template["name"],
            "protocol_id": template["protocol_id"],
            "status": {
                "measurements": "pending",
                "study_role": contract["study_role"],
                "medium_affine_paper_handoff": contract[
                    "medium_affine_paper_handoff"
                ],
                "final_paper_training_authorized": False,
                "eligible_tk_rows": sum(
                    int(row["lr_eligible"]) for row in tk_binding["rows"]
                ),
                "total_tk_rows": 3,
            },
            "design_template_sha256": _FROZEN_TEMPLATE_SHA256,
            "base_contract_sha256": _FROZEN_BASE_CONFIG_SHA256,
        }
    )
    data["model"]["architectures"] = {
        "conv3": copy.deepcopy(template["architecture"])
    }
    data["model"]["input_gain"] = {
        "conv3": 360.0,
        "shared_across_schemes": True,
        "recalibrate": False,
        "provenance": template["input_gain"]["provenance"],
    }
    data["model"]["shared_initialization"] = (
        "one_seed0_conv3_checkpoint_shared_across_schemes_optimizers_"
        "probes_canaries_and_candidates_on_both_hosts_by_sha256"
    )
    data["rows"] = _materialized_rows(
        template,
        tk_binding,
        operating_point=operating_point,
    )
    if operating_point is not None:
        data["operating_point"] = copy.deepcopy(dict(operating_point))
        data["operating_point_contract_sha256"] = (
            _FROZEN_OPERATING_POINT_CONTRACT_SHA256
        )
    data.pop("fixed_tk_gradient_security", None)
    data["upstream_tk"] = {
        "selection_artifact": template["upstream_tk"]["selection_path"],
        "shared_assets_artifact": template["upstream_tk"][
            "shared_assets_path"
        ],
        "producer_source_archive_artifact": template["upstream_tk"][
            "source_archive_path"
        ],
        **copy.deepcopy(dict(tk_binding)),
    }
    data["rho_search"] = copy.deepcopy(template["rho_search"])
    data["long_confirm"] = copy.deepcopy(template["long_confirm"])
    data["training_data_contract"] = copy.deepcopy(
        template["training_data_contract"]
    )
    data["dataset"]["train"]["batch_size"] = 16
    data["dataset"]["validation"]["batch_size"] = 64
    data["candidate_training"]["batch_size"] = 16
    data["candidate_training"]["validation_batch_size"] = 64
    data["candidate_training"]["epochs"] = 3
    data["candidate_training"]["steps_per_epoch"] = 3_438
    data["candidate_training"]["total_steps"] = CANDIDATE_TOTAL_STEPS
    data["stages"] = copy.deepcopy(template["stages"])
    data["execution"] = copy.deepcopy(template["execution"])
    unresolved = list(data["selection"]["unresolved_statuses"])
    for value in (
        "unresolved_upstream_tk",
        "unresolved_post_training_tk",
    ):
        if value not in unresolved:
            unresolved.append(value)
    data["selection"]["unresolved_statuses"] = unresolved
    data["selection"]["selected_status"] = (
        "selected_seed0_ordinary_mnist_three_epoch_screen"
    )
    data["artifacts"].update(
        {
            "upstream_tk_selection_sha256": tk_binding["selection_sha256"],
            "upstream_tk_row_hashes_required": True,
            "long_confirmation_required": False,
        }
    )
    return data


@dataclass(frozen=True)
class PerfectDiodeConv3HparamStudySpec(PerfectDiodeHparamStudySpec):
    """A validated materialized Conv3 successor study.

    Subclassing the v1 spec is intentional: dependency-light numerical helpers
    that only require ``data``, ``study_id``, and ``config_sha256`` can be
    reused without altering the immutable v1 implementation.
    """

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        template_path: str | Path = DEFAULT_TEMPLATE,
    ) -> "PerfectDiodeConv3HparamStudySpec":
        source = Path(path).expanduser().resolve()
        data = _mapping(_strict_json_load(source), "resolved v2 study")
        template, template_source = load_frozen_template(template_path)
        base = _load_frozen_base(template, template_source)
        if data.get("schema_version") != PD_CONV3_STUDY_SCHEMA_VERSION:
            raise _error(
                "study.schema_version",
                f"exactly {PD_CONV3_STUDY_SCHEMA_VERSION!r}",
                data.get("schema_version"),
            )
        upstream = _mapping(data.get("upstream_tk"), "study.upstream_tk")
        relative = Path(str(upstream.get("selection_artifact")))
        if (
            relative.as_posix() != template["upstream_tk"]["selection_path"]
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise _error(
                "study.upstream_tk.selection_artifact",
                f"exactly {template['upstream_tk']['selection_path']!r}",
                upstream.get("selection_artifact"),
            )
        selection_path = (source.parent / relative).resolve()
        selection = _mapping(_strict_json_load(selection_path), "T/K selection")
        expected_selection_sha = _sha256(
            upstream.get("selection_sha256"),
            "study.upstream_tk.selection_sha256",
        )
        binding = validate_tk_selection(
            selection,
            selection_path=selection_path,
            expected_sha256=expected_selection_sha,
            verify_artifact_files=True,
            template=template,
        )
        operating_point: dict[str, Any] | None = None
        if (
            "operating_point" in data
            or "operating_point_contract_sha256" in data
        ):
            operating_point, _operating_point_source = (
                load_frozen_operating_point_contract()
            )
            if (
                data.get("operating_point") != operating_point
                or data.get("operating_point_contract_sha256")
                != _FROZEN_OPERATING_POINT_CONTRACT_SHA256
            ):
                raise _error(
                    "resolved v2 study operating-point amendment",
                    "the exact frozen T=8, K=8 contract and canonical digest",
                    {
                        "operating_point": data.get("operating_point"),
                        "operating_point_contract_sha256": data.get(
                            "operating_point_contract_sha256"
                        ),
                    },
                )
        expected = materialized_study_data(
            template=template,
            base=base,
            tk_binding=binding,
            operating_point=operating_point,
        )
        if data != expected:
            raise _error(
                "resolved v2 study",
                "the exact template-plus-T/K materialization",
                sha256_json(data),
            )
        config_sha = sha256_json(data)
        identity_payload = copy.deepcopy(data)
        identity_payload.pop("name", None)
        study_id = "lrstudy_" + sha256_json(
            {
                "identity_schema": PD_CONV3_STUDY_ID_SCHEMA_VERSION,
                "study": identity_payload,
            }
        )
        return cls(data, study_id, config_sha)

    @property
    def eligible_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            copy.deepcopy(row)
            for row in self._data["rows"]
            if row["lr_eligible"]
        )

    @property
    def surfaces(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (row["row_id"], row["scheme"], optimizer)
            for row in self._data["rows"]
            for optimizer in OPTIMIZER_ORDER
        )


def materialize_study(
    *,
    template_path: str | Path,
    tk_selection_path: str | Path,
    tk_selection_sha256: str,
    tk_shared_assets_path: str | Path | None = None,
    tk_source_archive_path: str | Path | None = None,
    operating_point_path: str | Path | None = None,
    output_dir: str | Path,
) -> PerfectDiodeConv3HparamStudySpec:
    """Verify and copy the T/K handoff, then publish an exact resolved study."""

    template, template_source = load_frozen_template(template_path)
    base = _load_frozen_base(template, template_source)
    selection_source = Path(tk_selection_path).expanduser().resolve()
    selection = _mapping(_strict_json_load(selection_source), "T/K selection")
    binding = validate_tk_selection(
        selection,
        selection_path=selection_source,
        expected_sha256=tk_selection_sha256,
        verify_artifact_files=True,
        template=template,
        shared_assets_path=tk_shared_assets_path,
        source_archive_path=tk_source_archive_path,
    )
    operating_point = None
    if operating_point_path is not None:
        operating_point, _operating_point_source = (
            load_frozen_operating_point_contract(operating_point_path)
        )
    destination = Path(output_dir).expanduser().resolve()
    copied_selection = destination / template["upstream_tk"]["selection_path"]
    copied_tk_root = copied_selection.parent
    copied_tk_root.mkdir(parents=True, exist_ok=True)
    source_shared_assets = (
        (selection_source.parent / "shared_assets").resolve()
        if tk_shared_assets_path is None
        else Path(tk_shared_assets_path).expanduser().resolve()
    )
    source_archive = (
        (selection_source.parent / "source" / "source.tar").resolve()
        if tk_source_archive_path is None
        else Path(tk_source_archive_path).expanduser().resolve()
    )
    for relative_value in binding["producer_bundle_relative_paths"]:
        relative = Path(relative_value)
        if relative.parts and relative.parts[0] == "shared_assets":
            source_artifact = (
                source_shared_assets / Path(*relative.parts[1:])
            ).resolve()
            source_root = source_shared_assets
        elif relative == Path("source/source.tar"):
            source_artifact = source_archive
            source_root = source_archive.parent
        else:
            source_artifact = (selection_source.parent / relative).resolve()
            source_root = selection_source.parent.resolve()
        copied_artifact = (copied_tk_root / relative).resolve()
        try:
            source_artifact.relative_to(source_root)
            copied_artifact.relative_to(copied_tk_root)
        except ValueError as exc:
            raise _error(
                "producer_bundle_relative_paths",
                "only traversal-free paths inside both T/K roots",
                relative_value,
            ) from exc
        source_bytes = source_artifact.read_bytes()
        copied_artifact.parent.mkdir(parents=True, exist_ok=True)
        if copied_artifact.exists():
            if copied_artifact.read_bytes() != source_bytes:
                raise PerfectDiodeConv3HparamValidationError(
                    "Expected every existing copied T/K producer artifact to "
                    "be byte-identical. "
                    f"Provided value: {copied_artifact}."
                )
        else:
            atomic_write_bytes(copied_artifact, source_bytes)
    copied_selection_data = _mapping(
        _strict_json_load(copied_selection),
        "copied T/K selection",
    )
    copied_binding = validate_tk_selection(
        copied_selection_data,
        selection_path=copied_selection,
        expected_sha256=tk_selection_sha256,
        verify_artifact_files=True,
        template=template,
        shared_assets_path=None,
        source_archive_path=None,
    )
    if copied_binding != binding:
        raise _error(
            "copied T/K producer bundle",
            "the same normalized binding as the verified source bundle",
            copied_binding,
        )
    data = materialized_study_data(
        template=template,
        base=base,
        tk_binding=copied_binding,
        operating_point=operating_point,
    )
    resolved = destination / "study.resolved.json"
    if resolved.exists():
        existing = _mapping(_strict_json_load(resolved), "existing resolved study")
        if existing != data:
            raise PerfectDiodeConv3HparamValidationError(
                "Expected an existing resolved study to be canonically identical. "
                f"Provided value: {resolved}."
            )
    else:
        atomic_write_json(resolved, data, canonical=True)
    return PerfectDiodeConv3HparamStudySpec.from_path(
        resolved, template_path=template_source
    )


def _expanded_surface_policy(
    policy_value: Mapping[str, Any],
    scheme: str,
) -> dict[str, Any]:
    policy = copy.deepcopy(dict(policy_value))
    if scheme in {"baseline", "ours"}:
        observed_conv = tuple(float(value) for value in policy["rho_conv"])
        observed_dense = tuple(float(value) for value in policy["rho_dense"])
        if (
            observed_conv != FIXED_HIGH_RHO_CONV
            or observed_dense != FIXED_HIGH_RHO_DENSE
            or policy.get("automatic_lower_center_search") is not False
        ):
            raise _error(
                f"rho policy {scheme}",
                "the exact fixed high 3x3 core without lower-center search",
                policy,
            )
        policy["core_cells"] = [
            {"rho_conv": conv, "rho_dense": dense}
            for conv, dense in FIXED_HIGH_GRID
        ]
    return policy


def surface_rho_policy(
    spec: PerfectDiodeConv3HparamStudySpec,
    scheme: str,
) -> dict[str, Any]:
    """Derive one exact rho policy from the frozen template and study."""

    if scheme not in SCHEME_ORDER:
        raise _error(
            "rho policy scheme",
            f"one of {SCHEME_ORDER!r}",
            scheme,
        )
    template, _source = load_frozen_template()
    expected = copy.deepcopy(
        template["rho_search"]["surface_policies"][scheme]
    )
    observed = copy.deepcopy(
        spec.data["rho_search"]["surface_policies"][scheme]
    )
    if observed != expected:
        raise _error(
            f"rho policy {scheme}",
            "the exact frozen-template policy",
            observed,
        )
    return _expanded_surface_policy(expected, scheme)


def _normalize_routes(
    spec: PerfectDiodeConv3HparamStudySpec,
    routes: Mapping[str, str] | None,
) -> dict[str, str]:
    defaults = copy.deepcopy(spec.data["execution"]["default_surface_routes"])
    provided = defaults if routes is None else dict(routes)
    expected = {
        f"{row['row_id']}--{optimizer}"
        for row in spec.rows
        for optimizer in OPTIMIZER_ORDER
    }
    if set(provided) != expected:
        raise _error(
            "surface routes",
            "exactly one route for every one of the six surfaces",
            {
                "missing": sorted(expected - set(provided)),
                "extra": sorted(set(provided) - expected),
            },
        )
    for surface_id, host in provided.items():
        if host not in ALLOWED_HOSTS:
            raise _error(
                f"surface route {surface_id!r}",
                f"one of {ALLOWED_HOSTS!r}",
                host,
            )
    tk_host_by_row = {
        row["row_id"]: row["upstream_tk"]["execution_host"]
        for row in spec.rows
    }
    for surface_id, host in provided.items():
        row_id, _separator, _optimizer = surface_id.rpartition("--")
        expected_host = tk_host_by_row[row_id]
        if host != expected_host:
            raise _error(
                f"surface route {surface_id!r}",
                f"the same-scheme T/K execution host {expected_host!r}",
                host,
            )
    return {surface_id: provided[surface_id] for surface_id in sorted(provided)}


def _normalize_shared_assets(value: Mapping[str, Any]) -> dict[str, str]:
    assets = _mapping(value, "shared asset hashes")
    if set(assets) != set(SHARED_ASSET_HASH_KEYS):
        raise _error(
            "shared asset hashes",
            f"exactly {SHARED_ASSET_HASH_KEYS!r}",
            sorted(assets),
        )
    return {key: _sha256(assets[key], f"shared_assets.{key}") for key in SHARED_ASSET_HASH_KEYS}


def _normalize_execution_source(value: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(value, "execution source")
    if set(source) != set(EXECUTION_AUTHORITY_KEYS):
        raise _error(
            "execution source",
            f"exactly {EXECUTION_AUTHORITY_KEYS!r}",
            {
                "missing": sorted(set(EXECUTION_AUTHORITY_KEYS) - set(source)),
                "extra": sorted(set(source) - set(EXECUTION_AUTHORITY_KEYS)),
            },
        )
    host_environments = _mapping(
        source["host_environment_sha256s"],
        "execution_source.host_environment_sha256s",
    )
    if set(host_environments) != set(ALLOWED_HOSTS):
        raise _error(
            "execution_source.host_environment_sha256s",
            f"exactly the hosts {ALLOWED_HOSTS!r}",
            sorted(host_environments),
        )
    return {
        "source_commit": _commit(
            source["source_commit"], "execution_source.source_commit"
        ),
        "source_archive_sha256": _sha256(
            source["source_archive_sha256"],
            "execution_source.source_archive_sha256",
        ),
        "effective_code_fingerprint": _sha256(
            source["effective_code_fingerprint"],
            "execution_source.effective_code_fingerprint",
        ),
        "worker_launcher_sha256": _sha256(
            source["worker_launcher_sha256"],
            "execution_source.worker_launcher_sha256",
        ),
        "environment_contract_sha256": _sha256(
            source["environment_contract_sha256"],
            "execution_source.environment_contract_sha256",
        ),
        "host_environment_sha256s": {
            host: _sha256(
                host_environments[host],
                f"execution_source.host_environment_sha256s.{host}",
            )
            for host in ALLOWED_HOSTS
        },
        "resolved_study_path": _relative_path(
            source["resolved_study_path"],
            "execution_source.resolved_study_path",
            expected="study.resolved.json",
        ),
        "resolved_study_sha256": _sha256(
            source["resolved_study_sha256"],
            "execution_source.resolved_study_sha256",
        ),
        "output_root": _relative_path(
            source["output_root"],
            "execution_source.output_root",
            expected="surfaces",
        ),
    }


def build_surface_manifest(
    spec: PerfectDiodeConv3HparamStudySpec,
    *,
    shared_asset_hashes: Mapping[str, Any],
    execution_source: Mapping[str, Any],
    routes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the six-surface immutable launch manifest.

    Each surface has exactly one host.  Every entry binds the same split,
    initialization, T/K-cohort, and epoch-order hashes while selecting the
    actual environment receipt for its host.  A surface cannot mix cells from
    the different local/Akib torch builds.
    """

    route_map = _normalize_routes(spec, routes)
    shared = _normalize_shared_assets(shared_asset_hashes)
    source = _normalize_execution_source(execution_source)
    expected_resolved_sha = hashlib.sha256(
        canonical_json_bytes(spec.data) + b"\n"
    ).hexdigest()
    if source["resolved_study_sha256"] != expected_resolved_sha:
        raise _error(
            "execution_source.resolved_study_sha256",
            f"the exact canonical resolved-study file SHA-256 {expected_resolved_sha}",
            source["resolved_study_sha256"],
        )
    upstream_asset_binding = spec.data["upstream_tk"]["artifact_sha256s"][
        "asset_bundle"
    ]
    for key in (
        "train_indices_sha256",
        "validation_indices_sha256",
        "initialization_checkpoint_sha256",
        "initialization_tensor_sha256",
        "t_cohort_indices_sha256",
        "k_cohort_indices_sha256",
    ):
        row_values = {
            row["upstream_tk"].get(key)
            for row in spec.rows
            if key in row["upstream_tk"]
        }
        if row_values and (
            len(row_values) != 1
            or next(iter(row_values)) != upstream_asset_binding[key]
        ):
            raise _error(
                f"upstream {key}",
                "the exact shared-asset digest across all three schemes",
                sorted(row_values),
            )
        expected = upstream_asset_binding[key]
        if shared[key] != expected:
            raise _error(
                f"shared_assets.{key}",
                f"the hash-bound upstream T/K cohort digest {expected}",
                shared[key],
            )
    surfaces: list[dict[str, Any]] = []
    for row in spec.rows:
        for optimizer in OPTIMIZER_ORDER:
            surface_id = f"{row['row_id']}--{optimizer}"
            eligible = bool(row["lr_eligible"])
            surfaces.append(
                {
                    "surface_index": len(surfaces),
                    "surface_id": surface_id,
                    "row_id": row["row_id"],
                    "architecture": "conv3",
                    "scheme": row["scheme"],
                    "optimizer": optimizer,
                    "host": route_map[surface_id],
                    "lr_eligible": eligible,
                    "zero_work": not eligible,
                    "zero_work_reason": row["zero_work_reason"],
                    "inference_iterations": row["inference_iterations"],
                    "training_iterations": row["training_iterations"],
                    "input_gain": 360.0,
                    "upstream_tk": copy.deepcopy(row["upstream_tk"]),
                    "shared_asset_hashes": copy.deepcopy(shared),
                    "execution_source": copy.deepcopy(source),
                    "execution_environment_sha256": source[
                        "host_environment_sha256s"
                    ][route_map[surface_id]],
                    "output_path": f"surfaces/{surface_id}",
                    "completion_path": (
                        f"surfaces/{surface_id}/completion.json"
                    ),
                    "finalization_path": (
                        f"surfaces/{surface_id}/stages/finalize_lr/result.json"
                    ),
                    "rho_policy": (
                        surface_rho_policy(spec, row["scheme"])
                        if eligible
                        else None
                    ),
                    "canary_steps_per_cell": CANARY_STEPS,
                    "candidate_total_steps": CANDIDATE_TOTAL_STEPS,
                    "maximum_cells": MAXIMUM_CELLS_PER_SURFACE,
                    "maximum_expansion_waves": 1,
                    "long_confirmation": False,
                }
            )
    payload = {
        "schema_version": PD_CONV3_SURFACE_MANIFEST_SCHEMA_VERSION,
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "upstream_tk_selection_sha256": spec.data["upstream_tk"][
            "selection_sha256"
        ],
        "official_test_read": False,
        "routing_mode": "explicit_one_host_per_complete_surface",
        "cell_mixing_across_hosts": False,
        "relative_paths_resolve_from": "surface_manifest_parent",
        "execution_source": source,
        "shared_asset_hashes": shared,
        "surface_count": 6,
        "eligible_surface_count": sum(
            int(surface["lr_eligible"]) for surface in surfaces
        ),
        "zero_work_surface_count": sum(
            int(surface["zero_work"]) for surface in surfaces
        ),
        "maximum_core_cells": sum(
            9 for surface in surfaces if surface["lr_eligible"]
        ),
        "maximum_candidate_cells_after_one_expansion": sum(
            MAXIMUM_CELLS_PER_SURFACE
            for surface in surfaces
            if surface["lr_eligible"]
        ),
        "surfaces": surfaces,
        "public_stage_sequence": list(PUBLIC_STAGE_SEQUENCE),
        "long_confirmation": False,
    }
    payload["manifest_id"] = "pdlrmanifest_" + sha256_json(payload)
    return payload


def representative_preflight_plan(
    manifest: Mapping[str, Any],
    *,
    spec: PerfectDiodeConv3HparamStudySpec,
    host: str,
) -> dict[str, Any]:
    """Return a real Adam 640-step minimal-core canary from the manifest."""

    value = validate_surface_manifest(manifest, spec=spec)
    if host not in ALLOWED_HOSTS:
        raise _error(
            "representative preflight host",
            f"one of {ALLOWED_HOSTS!r}",
            host,
        )
    surfaces = _sequence(value.get("surfaces"), "surface manifest.surfaces")
    eligible_adam = [
        surface
        for surface in surfaces
        if (
            surface.get("optimizer") == "adam"
            and surface.get("host") == host
            and surface.get("lr_eligible") is True
        )
    ]
    if not eligible_adam:
        return {
            "status": "zero_work",
            "reason": f"no_tk_eligible_adam_surface_on_{host}",
            "host": host,
            "official_test_read": False,
        }
    surface = eligible_adam[0]
    policy = surface["rho_policy"]
    if policy["core_mode"] == "fixed_high_3x3":
        rho_conv = min(float(value) for value in policy["rho_conv"])
        rho_dense = min(float(value) for value in policy["rho_dense"])
    else:
        rho_conv = float(policy["initial_rho_conv"])
        rho_dense = float(policy["initial_rho_dense"])
    return {
        "status": "required",
        "stage": "preflight_canary",
        "same_public_runner": True,
        "manifest_id": value["manifest_id"],
        "surface_id": surface["surface_id"],
        "row_id": surface["row_id"],
        "optimizer": "adam",
        "host": surface["host"],
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "steps": CANARY_STEPS,
        "restart_from_shared_initialization": True,
        "shared_asset_hashes": copy.deepcopy(surface["shared_asset_hashes"]),
        "execution_source": copy.deepcopy(surface["execution_source"]),
        "execution_environment_sha256": surface[
            "execution_environment_sha256"
        ],
        "official_test_read": False,
    }


def validate_surface_manifest(
    manifest: Mapping[str, Any],
    *,
    spec: PerfectDiodeConv3HparamStudySpec,
) -> dict[str, Any]:
    """Validate identity plus the exact template-derived six-surface contract."""

    value = _mapping(manifest, "surface manifest")
    identifier = value.pop("manifest_id", None)
    expected_id = "pdlrmanifest_" + sha256_json(value)
    if identifier != expected_id:
        raise _error(
            "surface manifest.manifest_id",
            f"the content-derived identity {expected_id!r}",
            identifier,
        )
    value["manifest_id"] = identifier
    if value.get("schema_version") != PD_CONV3_SURFACE_MANIFEST_SCHEMA_VERSION:
        raise _error(
            "surface manifest.schema_version",
            f"exactly {PD_CONV3_SURFACE_MANIFEST_SCHEMA_VERSION!r}",
            value.get("schema_version"),
        )
    template, _source = load_frozen_template()
    surfaces = _sequence(
        value.get("surfaces"),
        "surface manifest.surfaces",
        length=6,
    )
    if value.get("surface_count") != 6:
        raise _error(
            "surface manifest.surface_count",
            "exactly 6",
            value.get("surface_count"),
        )
    if (
        value.get("official_test_read") is not False
        or value.get("long_confirmation") is not False
        or value.get("cell_mixing_across_hosts") is not False
    ):
        raise _error(
            "surface manifest execution gates",
            "official_test_read=false, long_confirmation=false, and "
            "cell_mixing_across_hosts=false",
            {
                "official_test_read": value.get("official_test_read"),
                "long_confirmation": value.get("long_confirmation"),
                "cell_mixing_across_hosts": value.get(
                    "cell_mixing_across_hosts"
                ),
            },
        )
    _exact_fields(
        value,
        {
            "study_id": spec.study_id,
            "config_sha256": spec.config_sha256,
            "upstream_tk_selection_sha256": spec.data["upstream_tk"][
                "selection_sha256"
            ],
            "routing_mode": "explicit_one_host_per_complete_surface",
            "relative_paths_resolve_from": "surface_manifest_parent",
            "public_stage_sequence": list(PUBLIC_STAGE_SEQUENCE),
            "surface_count": 6,
        },
        "surface manifest",
    )
    normalized_shared = _normalize_shared_assets(
        _mapping(
            value.get("shared_asset_hashes"),
            "surface manifest.shared_asset_hashes",
        )
    )
    if normalized_shared != value.get("shared_asset_hashes"):
        raise _error(
            "surface manifest.shared_asset_hashes",
            "the exact normalized shared-asset mapping",
            value.get("shared_asset_hashes"),
        )
    normalized_source = _normalize_execution_source(
        _mapping(
            value.get("execution_source"),
            "surface manifest.execution_source",
        )
    )
    if normalized_source != value.get("execution_source"):
        raise _error(
            "surface manifest.execution_source",
            "the exact normalized execution authority",
            value.get("execution_source"),
        )
    expected_resolved_sha = hashlib.sha256(
        canonical_json_bytes(spec.data) + b"\n"
    ).hexdigest()
    if normalized_source["resolved_study_sha256"] != expected_resolved_sha:
        raise _error(
            "surface manifest.execution_source.resolved_study_sha256",
            f"the exact resolved-study file SHA-256 {expected_resolved_sha}",
            normalized_source["resolved_study_sha256"],
        )
    upstream_assets = spec.data["upstream_tk"]["artifact_sha256s"][
        "asset_bundle"
    ]
    for key in (
        "train_indices_sha256",
        "validation_indices_sha256",
        "initialization_checkpoint_sha256",
        "initialization_tensor_sha256",
        "t_cohort_indices_sha256",
        "k_cohort_indices_sha256",
    ):
        if normalized_shared[key] != upstream_assets[key]:
            raise _error(
                f"surface manifest.shared_asset_hashes.{key}",
                f"the resolved-study upstream digest {upstream_assets[key]}",
                normalized_shared[key],
            )
    expected_surfaces: list[tuple[dict[str, Any], str]] = []
    for raw_row in spec.rows:
        row = _mapping(raw_row, "resolved study row")
        for optimizer in OPTIMIZER_ORDER:
            expected_surfaces.append((row, optimizer))
    eligible_count = 0
    for index, (raw_surface, expected_surface) in enumerate(
        zip(surfaces, expected_surfaces)
    ):
        surface = _mapping(
            raw_surface,
            f"surface manifest.surfaces[{index}]",
        )
        row, optimizer = expected_surface
        surface_id = f"{row['row_id']}--{optimizer}"
        expected_host = row["upstream_tk"]["execution_host"]
        eligible = bool(row["lr_eligible"])
        expected_policy = (
            surface_rho_policy(spec, row["scheme"])
            if eligible
            else None
        )
        _exact_fields(
            surface,
            {
                "surface_index": index,
                "surface_id": surface_id,
                "row_id": row["row_id"],
                "architecture": "conv3",
                "scheme": row["scheme"],
                "optimizer": optimizer,
                "host": expected_host,
                "lr_eligible": eligible,
                "zero_work": not eligible,
                "zero_work_reason": row["zero_work_reason"],
                "inference_iterations": row["inference_iterations"],
                "training_iterations": row["training_iterations"],
                "input_gain": 360.0,
                "upstream_tk": row["upstream_tk"],
                "shared_asset_hashes": normalized_shared,
                "execution_source": normalized_source,
                "execution_environment_sha256": normalized_source[
                    "host_environment_sha256s"
                ][expected_host],
                "output_path": f"surfaces/{surface_id}",
                "completion_path": f"surfaces/{surface_id}/completion.json",
                "finalization_path": (
                    f"surfaces/{surface_id}/stages/finalize_lr/result.json"
                ),
                "canary_steps_per_cell": CANARY_STEPS,
                "candidate_total_steps": CANDIDATE_TOTAL_STEPS,
                "maximum_cells": MAXIMUM_CELLS_PER_SURFACE,
                "maximum_expansion_waves": 1,
                "long_confirmation": False,
                "rho_policy": expected_policy,
            },
            f"surface manifest.surfaces[{index}]",
        )
        if eligible:
            eligible_count += 1
    expected_zero = 6 - eligible_count
    _exact_fields(
        value,
        {
            "eligible_surface_count": eligible_count,
            "zero_work_surface_count": expected_zero,
            "maximum_core_cells": 9 * eligible_count,
            "maximum_candidate_cells_after_one_expansion": (
                MAXIMUM_CELLS_PER_SURFACE * eligible_count
            ),
        },
        "surface manifest",
    )
    return value


__all__ = [
    "ALLOWED_HOSTS",
    "CANARY_STEPS",
    "CANDIDATE_TOTAL_STEPS",
    "DEFAULT_OPERATING_POINT_CONTRACT",
    "DEFAULT_TEMPLATE",
    "EXECUTION_AUTHORITY_KEYS",
    "FIXED_HIGH_GRID",
    "FIXED_HIGH_RHO_CONV",
    "FIXED_HIGH_RHO_DENSE",
    "FROZEN_TK_CONFIG",
    "FROZEN_TK_CONFIG_SHA256",
    "FROZEN_TK_STUDY_ID",
    "MAXIMUM_CELLS_PER_SURFACE",
    "LR_OPERATING_POINT_SCHEMA_VERSION",
    "OPTIMIZER_ORDER",
    "PD_CONV3_RUN_SCHEMA_VERSION",
    "PD_CONV3_STUDY_SCHEMA_VERSION",
    "PD_CONV3_SURFACE_MANIFEST_SCHEMA_VERSION",
    "PD_CONV3_TEMPLATE_SCHEMA_VERSION",
    "PUBLIC_STAGE_SEQUENCE",
    "SHARED_ASSET_HASH_KEYS",
    "PerfectDiodeConv3HparamStudySpec",
    "PerfectDiodeConv3HparamValidationError",
    "build_surface_manifest",
    "load_frozen_template",
    "load_frozen_operating_point_contract",
    "materialize_study",
    "materialized_study_data",
    "representative_preflight_plan",
    "surface_rho_policy",
    "validate_surface_manifest",
    "validate_tk_selection",
]
