"""Separate low-rho rescue for the unresolved ordinary-MNIST Conv3 legacy row.

The rescue deliberately consumes v7 assets without adding cells to, mutating,
or finalizing the immutable v7 study.  Three 256-step safety diagnostics are
content addressed here; only safety-clean entries can enter a fresh three-
epoch promotion run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .identity import (
    code_provenance,
    normalize_code_provenance,
    sha256_file,
    sha256_json,
)
from .io import atomic_write_csv, atomic_write_json, read_json
from .lr_engine import (
    build_loader_bundle,
    build_model_runtime,
    evaluate_validation,
    parameter_state_diagnostics,
    parameter_tensor_digest,
)
from .lr_protocol import (
    layerwise_learning_rate_report,
    linear_quantile,
    two_rho_learning_rates,
)
from .lr_stages import (
    PROBE_DIAGNOSTIC_COLUMNS,
    STEP_COLUMNS,
    _CandidateStepLoop,
    _architecture_checkpoint,
    _batch_hash,
    _bounded_initial_rms_scales,
    execute_candidate_entry,
    execute_probe_entry,
    execute_v7_asset_audit,
)
from .lr_study import create_study
from .lr_study_spec import LRStudySpec


RESCUE_SCHEMA_VERSION = "mnist-conv-lr-conv3-legacy-rescue/v1"
RESCUE_ID_SCHEMA_VERSION = "mnist-conv-lr-conv3-legacy-rescue-id/v1"
RESCUE_MANIFEST_SCHEMA_VERSION = "mnist-conv-lr-conv3-legacy-rescue-manifest/v1"
RESCUE_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-lr-conv3-legacy-rescue-completion/v1"
)
RESCUE_PROTOCOL_ID = (
    "conv-hardsigmoid-lr-conv3-legacy-low-rho-rescue-constant-sgd-bs16-v1"
)
PARENT_V7_STUDY_ID = (
    "lrstudy_6e56b399cf3e522a2c3f39f66a51611147b6171dd3b9e223f20791bc6275caf0"
)
LEGACY_ROW_ID = "conv3_legacy_v4_c0p25"
DIAGNOSTIC_STAGE = "diagnostic"
PROMOTION_STAGE = "promotion"
PROMOTION_RETRY_STAGE = "promotion_retry_1"

_TOP_LEVEL_KEYS = {
    "schema_version",
    "name",
    "protocol_id",
    "parent",
    "diagnostic",
    "promotion",
    "fallback",
    "artifacts",
}

_EXPECTED_PARENT = {
    "study_schema_version": "mnist-conv-lr-study/v7",
    "study_id": PARENT_V7_STUDY_ID,
    "row_id": LEGACY_ROW_ID,
    "parent_row_status": "unresolved",
    "mutate_parent_study": False,
    "reuse": [
        "ordinary_mnist_split",
        "seed0_conv3_initialization",
        "legacy_32_minibatch_probe",
    ],
}

_EXPECTED_TARGETS = [
    {"role": "lower-10x", "rho_conv": 5e-5, "rho_dense": 3e-4},
    {"role": "lower-33x", "rho_conv": 1.5e-5, "rho_dense": 1e-4},
    {"role": "lower-100x", "rho_conv": 5e-6, "rho_dense": 3e-5},
]

_EXPECTED_DIAGNOSTIC = {
    "steps": 256,
    "batch_size": 16,
    "schedule": "constant_seven_parameter_vector",
    "restart_each_entry_from_shared_initialization": True,
    "reset_train_shuffle_each_entry": True,
    "full_validation_after_safety_clean": True,
    "validation_is_diagnostic_only": True,
    "safety_gates": "parent_v7_range_test_gates",
    "targets": _EXPECTED_TARGETS,
}

_EXPECTED_PROMOTION = {
    "eligibility": "diagnostic_safety_clean_only",
    "epochs": 3,
    "steps_per_epoch": 3438,
    "total_steps": 10314,
    "schedule": "constant_seven_parameter_vector",
    "minimum_final_validation_accuracy": 0.9,
    "accuracy_boundary": "greater_than_or_equal",
    "cross_scheme_transfer_allowed": False,
    "selected_status": "legacy_rescue_candidate_passed_seed0_ordinary_mnist",
    "does_not_freeze_parent_v7": True,
}

_EXPECTED_FALLBACK = {
    "trigger": "all_three_diagnostics_fail_safety",
    "action": "investigate_reused_gain_tk_and_tied_bias_policy",
    "automatic_additional_target_reduction": False,
    "warmup_diagnostic_requires_new_protocol": True,
}

_EXPECTED_ARTIFACTS = {
    "hash": "sha256",
    "immutable_manifests": True,
    "completion_marker_written_last": True,
    "resume_completed_work": True,
    "official_test_read": False,
}


def _error(expected: str, provided: Any, path: str) -> ValueError:
    return ValueError(
        f"Expected {path} to be {expected}. Provided value: {provided!r}."
    )


def _assert_exact(expected: Any, provided: Any, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(provided, Mapping) or set(provided) != set(expected):
            raise _error(
                f"an object with exactly keys {sorted(expected)!r}",
                provided,
                path,
            )
        for key, value in expected.items():
            _assert_exact(value, provided[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(provided, list) or len(provided) != len(expected):
            raise _error(f"a list of length {len(expected)}", provided, path)
        for index, (expected_item, provided_item) in enumerate(
            zip(expected, provided, strict=True)
        ):
            _assert_exact(expected_item, provided_item, f"{path}[{index}]")
        return
    if expected != provided:
        raise _error(f"exactly {expected!r}", provided, path)


def _reject_non_finite(value: Any, path: str = "rescue") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise _error("finite JSON", value, path)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


@dataclass(frozen=True)
class LegacyRescueSpec:
    """Validated immutable-by-copy rescue contract."""

    data: dict[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LegacyRescueSpec":
        if not isinstance(value, Mapping):
            raise _error("a JSON object", value, "rescue")
        data = copy.deepcopy(dict(value))
        _reject_non_finite(data)
        if set(data) != _TOP_LEVEL_KEYS:
            raise _error(
                f"an object with exactly keys {sorted(_TOP_LEVEL_KEYS)!r}",
                {
                    "missing": sorted(_TOP_LEVEL_KEYS - set(data)),
                    "extra": sorted(set(data) - _TOP_LEVEL_KEYS),
                },
                "rescue",
            )
        if data["schema_version"] != RESCUE_SCHEMA_VERSION:
            raise _error(
                f"exactly {RESCUE_SCHEMA_VERSION!r}",
                data["schema_version"],
                "rescue.schema_version",
            )
        name = data["name"]
        if not isinstance(name, str) or not name.strip():
            raise _error("a non-empty string", name, "rescue.name")
        data["name"] = name.strip()
        if data["protocol_id"] != RESCUE_PROTOCOL_ID:
            raise _error(
                f"exactly {RESCUE_PROTOCOL_ID!r}",
                data["protocol_id"],
                "rescue.protocol_id",
            )
        for key, expected in (
            ("parent", _EXPECTED_PARENT),
            ("diagnostic", _EXPECTED_DIAGNOSTIC),
            ("promotion", _EXPECTED_PROMOTION),
            ("fallback", _EXPECTED_FALLBACK),
            ("artifacts", _EXPECTED_ARTIFACTS),
        ):
            _assert_exact(expected, data[key], f"rescue.{key}")
        return cls(data)

    @classmethod
    def from_path(cls, path: str | Path) -> "LegacyRescueSpec":
        source = Path(path).expanduser().resolve()
        try:
            value = json.loads(source.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Expected rescue config to contain strict JSON. "
                f"Provided value: {source}: {exc}."
            ) from exc
        return cls.from_dict(value)

    def identity_payload(self) -> dict[str, Any]:
        value = copy.deepcopy(self.data)
        value.pop("name")
        return value

    @property
    def rescue_id(self) -> str:
        digest = sha256_json(
            {
                "schema_version": RESCUE_ID_SCHEMA_VERSION,
                "rescue_spec": self.identity_payload(),
            }
        )
        return f"lrrescue_{digest}"


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or read_json(path) != value:
            raise RuntimeError(
                "Expected existing immutable JSON to have identical content. "
                f"Provided value: {path}."
            )
        return
    atomic_write_json(path, value, canonical=True)


def _artifact_record(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(
            "Expected parent rescue input to be an existing regular file. "
            f"Provided value: {path}."
        )
    return {
        "path": relative_path,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _verify_artifact_records(root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        observed = _artifact_record(root, str(record["path"]))
        expected = {
            "path": str(record["path"]),
            "sha256": str(record["sha256"]),
            "bytes": int(record["bytes"]),
        }
        if observed != expected:
            raise RuntimeError(
                "Expected immutable rescue input artifact to match the manifest. "
                f"Provided value: expected={expected!r}, observed={observed!r}."
            )


def prepare_local_parent(
    *,
    rescue_config: str | Path,
    parent_config: str | Path,
    parent_results_root: str | Path,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    """Create local v7 assets and measure only the legacy 32-minibatch probe."""

    rescue = LegacyRescueSpec.from_path(rescue_config)
    parent = LRStudySpec.from_path(parent_config)
    if parent.study_id != rescue.data["parent"]["study_id"]:
        raise RuntimeError(
            "Expected parent v7 study ID to match the immutable rescue contract. "
            f"Provided value: {parent.study_id!r}."
        )
    parent_dir, _ = create_study(parent, parent_results_root)
    audit = execute_v7_asset_audit(
        parent.data,
        parent_dir,
        data_root=data_root,
        download=download,
        device=device,
    )
    probe = execute_probe_entry(
        parent.data,
        parent_dir,
        LEGACY_ROW_ID,
        data_root=data_root,
        download=download,
        device=device,
    )
    return {
        "status": "complete",
        "parent_study_id": parent.study_id,
        "parent_study_dir": str(parent_dir),
        "asset_audit_status": audit["status"],
        "probe_status": probe["status"],
        "probe_median_units_by_weight": probe["median_units_by_weight"],
        "official_test_read": False,
    }


def plan_rescue(
    *,
    rescue_config: str | Path,
    parent_study_dir: str | Path,
    results_root: str | Path,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish the immutable three-entry diagnostic manifest."""

    rescue = LegacyRescueSpec.from_path(rescue_config)
    parent_root = Path(parent_study_dir).expanduser().resolve()
    parent = LRStudySpec.from_path(parent_root / "study.resolved.json")
    if parent.study_id != rescue.data["parent"]["study_id"]:
        raise RuntimeError(
            "Expected parent study ID to match the rescue contract. "
            f"Provided value: {parent.study_id!r}."
        )
    probe_relative = f"stages/probe/entries/{LEGACY_ROW_ID}/summary.json"
    parent_artifacts = [
        _artifact_record(parent_root, relative)
        for relative in (
            "study.resolved.json",
            "split/indices.json",
            "split/provenance.json",
            "initialization/conv3.pt",
            "initialization/conv3.json",
            probe_relative,
        )
    ]
    probe = read_json(parent_root / probe_relative)
    if probe.get("row", {}).get("row_id") != LEGACY_ROW_ID:
        raise RuntimeError(
            "Expected the parent probe summary to belong to the legacy Conv3 row. "
            f"Provided value: {probe.get('row')!r}."
        )
    units = {
        str(name): float(value)
        for name, value in probe["median_units_by_weight"].items()
    }
    groups = {
        str(name): [str(member) for member in members]
        for name, members in probe["bias_weight_lr_groups"].items()
    }
    results = Path(results_root).expanduser().resolve()
    rescue_dir = (
        results
        / "lr_rescues"
        / f"{rescue.data['name']}--{rescue.rescue_id}"
    )
    rescue_dir.mkdir(parents=True, exist_ok=True)
    _write_immutable_json(rescue_dir / "rescue.resolved.json", rescue.data)
    entries: list[dict[str, Any]] = []
    thresholds = parent.data["artifacts"]["large_raw_lr_reporting"]["thresholds"]
    for index, target in enumerate(rescue.data["diagnostic"]["targets"]):
        rates = two_rho_learning_rates(
            units,
            rho_conv=float(target["rho_conv"]),
            rho_dense=float(target["rho_dense"]),
            bias_weight_lr_groups=groups,
        )
        report = layerwise_learning_rate_report(rates, thresholds=thresholds)
        entry_id = str(target["role"])
        entries.append(
            {
                "entry_index": index,
                "entry_id": entry_id,
                "output_dir": f"stages/{DIAGNOSTIC_STAGE}/entries/{entry_id}",
                "payload": {
                    "row_id": LEGACY_ROW_ID,
                    "role": entry_id,
                    "rho_conv": float(target["rho_conv"]),
                    "rho_dense": float(target["rho_dense"]),
                    "median_units_by_weight": units,
                    "bias_weight_lr_groups": groups,
                    "learning_rates_by_parameter": rates,
                    "learning_rates_by_weight": report["weight_learning_rates"],
                    "peak_learning_rate": report["maximum_weight_learning_rate"],
                },
            }
        )
    manifest = {
        "schema_version": RESCUE_MANIFEST_SCHEMA_VERSION,
        "stage": DIAGNOSTIC_STAGE,
        "rescue_id": rescue.rescue_id,
        "parent_study_id": parent.study_id,
        "parent_study_dir": str(parent_root),
        "parent_artifacts": parent_artifacts,
        "code_provenance": normalize_code_provenance(
            provenance or code_provenance()
        ),
        "entry_count": len(entries),
        "entries": entries,
        "official_test_read": False,
    }
    manifest_path = rescue_dir / f"stages/{DIAGNOSTIC_STAGE}/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable_json(manifest_path, manifest)
    return {
        "status": "planned",
        "rescue_id": rescue.rescue_id,
        "rescue_dir": str(rescue_dir),
        "manifest_path": str(manifest_path),
        "entry_count": len(entries),
        "entries": [
            {
                "entry_index": entry["entry_index"],
                "entry_id": entry["entry_id"],
                "rho_conv": entry["payload"]["rho_conv"],
                "rho_dense": entry["payload"]["rho_dense"],
                "learning_rates_by_parameter": entry["payload"][
                    "learning_rates_by_parameter"
                ],
            }
            for entry in entries
        ],
    }


def _load_rescue(
    rescue_dir: str | Path,
    stage: str,
    *,
    require_current_source: bool = True,
) -> tuple[Path, LegacyRescueSpec, dict[str, Any], Path]:
    root = Path(rescue_dir).expanduser().resolve()
    spec = LegacyRescueSpec.from_path(root / "rescue.resolved.json")
    if not root.name.endswith(f"--{spec.rescue_id}"):
        raise RuntimeError(
            "Expected rescue directory name to end with its content address. "
            f"Provided value: {root.name!r}."
        )
    manifest_path = root / "stages" / stage / "manifest.json"
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != RESCUE_MANIFEST_SCHEMA_VERSION
        or manifest.get("stage") != stage
        or manifest.get("rescue_id") != spec.rescue_id
    ):
        raise RuntimeError(
            "Expected a matching immutable rescue manifest. "
            f"Provided value: {manifest!r}."
        )
    planned = normalize_code_provenance(manifest["code_provenance"])
    current = normalize_code_provenance(code_provenance())
    if require_current_source and (
        planned["effective_code_fingerprint"]
        != current["effective_code_fingerprint"]
    ):
        raise RuntimeError(
            "Expected current source fingerprint to match the rescue manifest. "
            f"Provided value: planned={planned!r}, current={current!r}."
        )
    parent_root = Path(manifest["parent_study_dir"]).resolve()
    _verify_artifact_records(parent_root, manifest["parent_artifacts"])
    return root, spec, manifest, parent_root


def _completion_path(root: Path, stage: str, entry_id: str) -> Path:
    return root / "stages" / stage / "entries" / entry_id / "complete.json"


def _validate_entry_completion(
    root: Path, stage: str, entry_id: str
) -> dict[str, Any] | None:
    path = _completion_path(root, stage, entry_id)
    if not path.is_file() or path.is_symlink():
        return None
    completion = read_json(path)
    if (
        completion.get("schema_version") != RESCUE_COMPLETION_SCHEMA_VERSION
        or completion.get("stage") != stage
        or completion.get("entry_id") != entry_id
    ):
        return None
    entry_dir = path.parent
    for record in completion.get("outputs", []):
        output = entry_dir / str(record["name"])
        if (
            output.is_symlink()
            or not output.is_file()
            or output.stat().st_size != int(record["bytes"])
            or sha256_file(output) != str(record["sha256"])
        ):
            return None
    return completion


def _publish_entry_completion(
    root: Path,
    *,
    stage: str,
    entry_id: str,
    required_outputs: Sequence[str],
) -> dict[str, Any]:
    entry_dir = _completion_path(root, stage, entry_id).parent
    records = []
    for name in sorted(required_outputs):
        path = entry_dir / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                "Expected rescue output to exist before completion publication. "
                f"Provided value: {path}."
            )
        records.append(
            {"name": name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
    value = {
        "schema_version": RESCUE_COMPLETION_SCHEMA_VERSION,
        "stage": stage,
        "entry_id": entry_id,
        "outputs": records,
    }
    _write_immutable_json(_completion_path(root, stage, entry_id), value)
    return value


def _q90_by_parameter(values: Mapping[str, Sequence[float]]) -> dict[str, float]:
    return {
        name: linear_quantile(samples, 0.9)
        for name, samples in values.items()
        if samples
    }


def execute_diagnostic_entry(
    *,
    rescue_dir: str | Path,
    entry_index: int,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    """Run one 256-step legacy candidate from the shared initialization."""

    root, spec, manifest, parent_root = _load_rescue(
        rescue_dir, DIAGNOSTIC_STAGE
    )
    entries = manifest["entries"]
    if type(entry_index) is not int or not 0 <= entry_index < len(entries):
        raise ValueError(
            f"Expected entry_index in [0, {len(entries) - 1}]. "
            f"Provided value: {entry_index!r}."
        )
    entry = entries[entry_index]
    entry_id = str(entry["entry_id"])
    completed = _validate_entry_completion(root, DIAGNOSTIC_STAGE, entry_id)
    if completed is not None:
        return {
            "status": "resumed_complete",
            "entry_index": entry_index,
            "entry_id": entry_id,
            "summary": read_json(
                completed_path(root, DIAGNOSTIC_STAGE, entry_id)
            ),
        }

    parent = LRStudySpec.from_path(parent_root / "study.resolved.json")
    row = next(
        item for item in parent.rows if item["row_id"] == LEGACY_ROW_ID
    )
    payload = entry["payload"]
    rates = {
        str(name): float(value)
        for name, value in payload["learning_rates_by_parameter"].items()
    }
    runtime = build_model_runtime(
        parent.data,
        row,
        device=device,
        initialization_checkpoint=_architecture_checkpoint(parent_root, row),
        learning_rate=rates,
    )
    initial_tensor_digest = parameter_tensor_digest(runtime.parameters)
    initial_diagnostics = parameter_state_diagnostics(runtime.parameters)
    bounded_scales = _bounded_initial_rms_scales(initial_diagnostics)
    bundle = build_loader_bundle(
        parent.data,
        data_root=data_root,
        download=download,
        return_source_indices=True,
    )
    bundle.reset_train_shuffle()
    loop = _CandidateStepLoop(
        study=parent.data,
        runtime=runtime,
        peak_learning_rate=max(rates.values()),
        learning_rates_by_parameter=rates,
        bounded_initial_scales=bounded_scales,
        initial_diagnostics=initial_diagnostics,
        relative=True,
        layerwise=True,
        weight_median=True,
    )
    epoch_relative = {name: [] for name in bounded_scales}
    iterator = iter(bundle.train_loader)
    requested_steps = int(spec.data["diagnostic"]["steps"])
    for step in range(1, requested_steps + 1):
        result = loop.run_step(
            next(iterator),
            epoch=0,
            batch_in_epoch=step,
            epoch_relative_rhos=epoch_relative,
        )
        if result is None or loop.inadmissible_reason is not None:
            break
    safety_clean = (
        loop.successful_steps == requested_steps
        and loop.inadmissible_reason is None
    )
    validation_metric = (
        evaluate_validation(runtime, bundle.validation_loader)
        if safety_clean
        and spec.data["diagnostic"]["full_validation_after_safety_clean"]
        else None
    )
    if validation_metric is not None:
        validation_metric.pop("source_indices")
    entry_dir = root / str(entry["output_dir"])
    entry_dir.mkdir(parents=True, exist_ok=True)
    minibatches = {
        "schema_version": "mnist-conv-lr-minibatches/v1",
        "row_id": LEGACY_ROW_ID,
        "stage": DIAGNOSTIC_STAGE,
        "batches": [list(batch) for batch in loop.minibatches],
        "batch_order_sha256": _batch_hash(loop.minibatches),
    }
    run_spec = {
        "schema_version": "mnist-conv-lr-conv3-legacy-rescue-run/v1",
        "rescue_id": spec.rescue_id,
        "phase": DIAGNOSTIC_STAGE,
        "entry_id": entry_id,
        "parent_study_id": parent.study_id,
        "row": row,
        "rho_conv": float(payload["rho_conv"]),
        "rho_dense": float(payload["rho_dense"]),
        "learning_rates_by_parameter": rates,
        "steps": requested_steps,
        "schedule": "constant",
        "restart_from_shared_initialization": True,
        "checkpoint_sha256": sha256_file(
            _architecture_checkpoint(parent_root, row)
        ),
        "initial_parameter_tensor_sha256": initial_tensor_digest,
        "train_indices_sha256": bundle.train_indices_hash,
        "validation_indices_sha256": bundle.validation_indices_hash,
        "official_test_read": False,
    }
    losses = [
        float(item["loss"])
        for item in loop.step_rows
        if item.get("loss") is not None
    ]
    summary = {
        "schema_version": "mnist-conv-lr-conv3-legacy-rescue-diagnostic/v1",
        "rescue_id": spec.rescue_id,
        "entry_id": entry_id,
        "status": "safety_clean" if safety_clean else "failed",
        "safety_clean": safety_clean,
        "row": row,
        "rho_conv": float(payload["rho_conv"]),
        "rho_dense": float(payload["rho_dense"]),
        "median_units_by_weight": payload["median_units_by_weight"],
        "learning_rates_by_parameter": rates,
        "requested_steps": requested_steps,
        "attempted_steps": loop.total_step,
        "completed_steps": loop.successful_steps,
        "inadmissible_reason": loop.inadmissible_reason,
        "numerical_failure_detail": loop.numerical_failure,
        "safety_gate_failure": loop.safety_gate_failure,
        "first_loss": None if not losses else losses[0],
        "final_loss": None if not losses else losses[-1],
        "minimum_loss": None if not losses else min(losses),
        "final_loss_ema": loop.ema,
        "validation_metrics": validation_metric,
        "median_projection_efficiency": (
            None
            if not loop.projection_values
            else linear_quantile(loop.projection_values, 0.5)
        ),
        "median_projection_efficiency_by_parameter": {
            name: linear_quantile(values, 0.5)
            for name, values in loop.projection_values_by_parameter.items()
            if values
        },
        "final_bound_occupancy_by_parameter": (
            loop.final_bound_occupancy_by_parameter
        ),
        "maximum_bound_occupancy_by_parameter": (
            loop.maximum_bound_occupancy_by_parameter
        ),
        "observed_rho_relative_q90_by_parameter": _q90_by_parameter(
            loop.complete_relative_rhos
        ),
        "observed_rho_span_q90_by_parameter": _q90_by_parameter(
            loop.complete_span_rhos
        ),
        "checkpoint_sha256": run_spec["checkpoint_sha256"],
        "initial_parameter_tensor_sha256": initial_tensor_digest,
        "final_parameter_tensor_sha256": parameter_tensor_digest(
            runtime.parameters
        ),
        "train_indices_sha256": bundle.train_indices_hash,
        "validation_indices_sha256": bundle.validation_indices_hash,
        "minibatch_order_sha256": minibatches["batch_order_sha256"],
        "eligible_for_three_epoch_promotion": safety_clean,
        "parent_v7_mutated": False,
        "official_test_read": False,
    }
    atomic_write_csv(entry_dir / "step_log.csv", STEP_COLUMNS, loop.step_rows)
    atomic_write_csv(
        entry_dir / "parameter_diagnostics.csv",
        PROBE_DIAGNOSTIC_COLUMNS,
        loop.diagnostics,
    )
    atomic_write_json(entry_dir / "minibatches.json", minibatches, canonical=True)
    atomic_write_json(
        entry_dir / "validation.json",
        [] if validation_metric is None else [validation_metric],
        canonical=True,
    )
    atomic_write_json(
        entry_dir / "run_spec.rescue.json", run_spec, canonical=True
    )
    atomic_write_json(entry_dir / "summary.json", summary, canonical=True)
    _publish_entry_completion(
        root,
        stage=DIAGNOSTIC_STAGE,
        entry_id=entry_id,
        required_outputs=(
            "minibatches.json",
            "parameter_diagnostics.csv",
            "run_spec.rescue.json",
            "step_log.csv",
            "summary.json",
            "validation.json",
        ),
    )
    return {
        "status": "complete",
        "entry_index": entry_index,
        "entry_id": entry_id,
        "summary": summary,
    }


def completed_path(root: Path, stage: str, entry_id: str) -> Path:
    return root / "stages" / stage / "entries" / entry_id / "summary.json"


def _publish_stage_completion(
    root: Path,
    *,
    stage: str,
    entry_ids: Sequence[str],
) -> dict[str, Any]:
    entry_completions = []
    for entry_id in entry_ids:
        completion = _validate_entry_completion(root, stage, entry_id)
        if completion is None:
            raise RuntimeError(
                "Expected every rescue entry to complete before stage finalization. "
                f"Provided value: stage={stage!r}, entry_id={entry_id!r}."
            )
        path = _completion_path(root, stage, entry_id)
        entry_completions.append(
            {
                "entry_id": entry_id,
                "completion_sha256": sha256_file(path),
                "completion_bytes": path.stat().st_size,
            }
        )
    value = {
        "schema_version": "mnist-conv-lr-conv3-legacy-rescue-stage-completion/v1",
        "stage": stage,
        "entry_count": len(entry_ids),
        "entries": entry_completions,
    }
    path = root / "stages" / stage / "complete.json"
    _write_immutable_json(path, value)
    return value


def finalize_diagnostics(*, rescue_dir: str | Path) -> dict[str, Any]:
    """Close the diagnostic stage and publish only safety-clean promotions."""

    root, spec, manifest, _ = _load_rescue(rescue_dir, DIAGNOSTIC_STAGE)
    entry_ids = [str(entry["entry_id"]) for entry in manifest["entries"]]
    _publish_stage_completion(root, stage=DIAGNOSTIC_STAGE, entry_ids=entry_ids)
    summaries = [
        read_json(completed_path(root, DIAGNOSTIC_STAGE, entry_id))
        for entry_id in entry_ids
    ]
    eligible = {
        summary["entry_id"]: summary
        for summary in summaries
        if summary["safety_clean"] is True
    }
    promotion_entries = []
    by_id = {str(entry["entry_id"]): entry for entry in manifest["entries"]}
    for entry_id in entry_ids:
        if entry_id not in eligible:
            continue
        diagnostic = by_id[entry_id]
        role = f"legacy-rescue-{entry_id}"
        promotion_entry_id = f"{LEGACY_ROW_ID}--{role}"
        payload = copy.deepcopy(diagnostic["payload"])
        payload.update(
            {
                "role": role,
                "candidate_role": role,
                "candidate_stage": "legacy_rescue_promotion",
                "rescue_id": spec.rescue_id,
                "diagnostic_summary_sha256": sha256_file(
                    completed_path(root, DIAGNOSTIC_STAGE, entry_id)
                ),
            }
        )
        promotion_entries.append(
            {
                "entry_index": len(promotion_entries),
                "entry_id": promotion_entry_id,
                "source_diagnostic_entry_id": entry_id,
                "output_dir": (
                    f"stages/{PROMOTION_STAGE}/entries/{promotion_entry_id}"
                ),
                "payload": payload,
            }
        )
    promotion_manifest = {
        "schema_version": RESCUE_MANIFEST_SCHEMA_VERSION,
        "stage": PROMOTION_STAGE,
        "rescue_id": spec.rescue_id,
        "parent_study_id": manifest["parent_study_id"],
        "parent_study_dir": manifest["parent_study_dir"],
        "parent_artifacts": manifest["parent_artifacts"],
        "code_provenance": manifest["code_provenance"],
        "diagnostic_manifest_sha256": sha256_file(
            root / f"stages/{DIAGNOSTIC_STAGE}/manifest.json"
        ),
        "diagnostic_completion_sha256": sha256_file(
            root / f"stages/{DIAGNOSTIC_STAGE}/complete.json"
        ),
        "entry_count": len(promotion_entries),
        "entries": promotion_entries,
        "zero_work": not promotion_entries,
        "official_test_read": False,
    }
    path = root / f"stages/{PROMOTION_STAGE}/manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable_json(path, promotion_manifest)
    if not promotion_entries:
        _publish_stage_completion(root, stage=PROMOTION_STAGE, entry_ids=[])
    return {
        "status": "complete",
        "diagnostic_count": len(summaries),
        "safety_clean_count": len(promotion_entries),
        "promotion_manifest_path": str(path),
        "promotion_entry_ids": [
            entry["entry_id"] for entry in promotion_entries
        ],
        "fallback_required": not promotion_entries,
    }


def replan_promotions(
    *,
    rescue_dir: str | Path,
    retry_stage: str = PROMOTION_RETRY_STAGE,
) -> dict[str, Any]:
    """Preserve a failed promotion manifest and issue one code-fixed retry."""

    if retry_stage != PROMOTION_RETRY_STAGE:
        raise ValueError(
            f"Expected retry_stage to be {PROMOTION_RETRY_STAGE!r}. "
            f"Provided value: {retry_stage!r}."
        )
    root, spec, diagnostic_manifest, _ = _load_rescue(
        rescue_dir,
        DIAGNOSTIC_STAGE,
        require_current_source=False,
    )
    diagnostic_entry_ids = [
        str(entry["entry_id"]) for entry in diagnostic_manifest["entries"]
    ]
    for entry_id in diagnostic_entry_ids:
        if _validate_entry_completion(
            root, DIAGNOSTIC_STAGE, entry_id
        ) is None:
            raise RuntimeError(
                "Expected immutable diagnostic completions before promotion "
                f"retry planning. Provided value: {entry_id!r}."
            )
    diagnostic_completion = root / f"stages/{DIAGNOSTIC_STAGE}/complete.json"
    if not diagnostic_completion.is_file():
        raise RuntimeError(
            "Expected the diagnostic stage completion before promotion retry "
            f"planning. Provided value: {diagnostic_completion}."
        )
    failed_manifest_path = root / f"stages/{PROMOTION_STAGE}/manifest.json"
    failed_manifest = read_json(failed_manifest_path)
    if (
        failed_manifest.get("rescue_id") != spec.rescue_id
        or failed_manifest.get("stage") != PROMOTION_STAGE
    ):
        raise RuntimeError(
            "Expected the preserved first promotion manifest to match this "
            f"rescue. Provided value: {failed_manifest!r}."
        )
    entries = copy.deepcopy(failed_manifest["entries"])
    for entry in entries:
        entry["output_dir"] = (
            f"stages/{retry_stage}/entries/{entry['entry_id']}"
        )
    manifest = {
        "schema_version": RESCUE_MANIFEST_SCHEMA_VERSION,
        "stage": retry_stage,
        "rescue_id": spec.rescue_id,
        "parent_study_id": failed_manifest["parent_study_id"],
        "parent_study_dir": failed_manifest["parent_study_dir"],
        "parent_artifacts": failed_manifest["parent_artifacts"],
        "code_provenance": normalize_code_provenance(code_provenance()),
        "diagnostic_manifest_sha256": sha256_file(
            root / f"stages/{DIAGNOSTIC_STAGE}/manifest.json"
        ),
        "diagnostic_completion_sha256": sha256_file(
            diagnostic_completion
        ),
        "retry_of": {
            "stage": PROMOTION_STAGE,
            "manifest_sha256": sha256_file(failed_manifest_path),
            "failure_phase": "pre_training_run_spec_validation",
            "candidate_outputs_published": False,
        },
        "entry_count": len(entries),
        "entries": entries,
        "zero_work": not entries,
        "official_test_read": False,
    }
    path = root / "stages" / retry_stage / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable_json(path, manifest)
    return {
        "status": "planned",
        "stage": retry_stage,
        "manifest_path": str(path),
        "entry_count": len(entries),
        "entry_ids": [entry["entry_id"] for entry in entries],
        "preserved_failed_manifest_path": str(failed_manifest_path),
    }


def execute_promotion_entry(
    *,
    rescue_dir: str | Path,
    entry_index: int,
    data_root: str | Path,
    download: bool,
    device: str,
    stage: str = PROMOTION_STAGE,
) -> dict[str, Any]:
    """Run one eligible three-epoch candidate under a rescue-owned wrapper."""

    if stage not in {PROMOTION_STAGE, PROMOTION_RETRY_STAGE}:
        raise ValueError(
            "Expected promotion stage to be 'promotion' or "
            f"{PROMOTION_RETRY_STAGE!r}. Provided value: {stage!r}."
        )
    root, spec, manifest, parent_root = _load_rescue(rescue_dir, stage)
    entries = manifest["entries"]
    if type(entry_index) is not int or not 0 <= entry_index < len(entries):
        raise ValueError(
            f"Expected entry_index in [0, {len(entries) - 1}]. "
            f"Provided value: {entry_index!r}."
        )
    entry = entries[entry_index]
    entry_id = str(entry["entry_id"])
    completed = _validate_entry_completion(root, stage, entry_id)
    if completed is not None:
        return {
            "status": "resumed_complete",
            "entry_index": entry_index,
            "entry_id": entry_id,
            "summary": read_json(
                root
                / "stages"
                / stage
                / "entries"
                / entry_id
                / "rescue_summary.json"
            ),
        }
    payload = entry["payload"]
    source_summary = (
        root
        / "stages"
        / DIAGNOSTIC_STAGE
        / "entries"
        / entry["source_diagnostic_entry_id"]
        / "summary.json"
    )
    if (
        sha256_file(source_summary)
        != payload["diagnostic_summary_sha256"]
        or read_json(source_summary).get("safety_clean") is not True
    ):
        raise RuntimeError(
            "Expected promotion source diagnostic to remain safety-clean and "
            f"hash-identical. Provided value: {source_summary}."
        )
    parent = LRStudySpec.from_path(parent_root / "study.resolved.json")
    underlying = execute_candidate_entry(
        parent.data,
        parent_root,
        LEGACY_ROW_ID,
        str(payload["candidate_role"]),
        candidate_payload=payload,
        output_stage=stage,
        output_root=root,
        data_root=data_root,
        download=download,
        device=device,
    )
    entry_dir = root / str(entry["output_dir"])
    underlying_summary_path = entry_dir / "summary.json"
    underlying_summary = read_json(underlying_summary_path)
    wrapper_summary = {
        "schema_version": "mnist-conv-lr-conv3-legacy-rescue-promotion/v1",
        "rescue_id": spec.rescue_id,
        "entry_id": entry_id,
        "source_diagnostic_entry_id": entry["source_diagnostic_entry_id"],
        "source_diagnostic_summary_sha256": payload[
            "diagnostic_summary_sha256"
        ],
        "status": "passed" if underlying_summary["admissible"] else "failed",
        "selected_status": (
            spec.data["promotion"]["selected_status"]
            if underlying_summary["admissible"]
            else None
        ),
        "rho_conv": float(payload["rho_conv"]),
        "rho_dense": float(payload["rho_dense"]),
        "learning_rates_by_parameter": payload[
            "learning_rates_by_parameter"
        ],
        "underlying_parent_v7_runtime_summary_sha256": sha256_file(
            underlying_summary_path
        ),
        "training_completed": underlying_summary["training_completed"],
        "admissible": underlying_summary["admissible"],
        "inadmissible_reason": underlying_summary["inadmissible_reason"],
        "safety_gate_failure": underlying_summary["safety_gate_failure"],
        "completed_steps": underlying_summary["completed_steps"],
        "final_validation_loss": underlying_summary[
            "final_validation_loss"
        ],
        "final_validation_accuracy": underlying_summary[
            "final_validation_accuracy"
        ],
        "accuracy_gate_passed": underlying_summary["accuracy_gate_passed"],
        "does_not_freeze_parent_v7": True,
        "parent_v7_mutated": False,
        "official_test_read": False,
    }
    rescue_run_spec = {
        "schema_version": "mnist-conv-lr-conv3-legacy-rescue-run/v1",
        "rescue_id": spec.rescue_id,
        "phase": stage,
        "entry_id": entry_id,
        "parent_study_id": parent.study_id,
        "rho_conv": float(payload["rho_conv"]),
        "rho_dense": float(payload["rho_dense"]),
        "learning_rates_by_parameter": payload[
            "learning_rates_by_parameter"
        ],
        "epochs": int(spec.data["promotion"]["epochs"]),
        "total_steps": int(spec.data["promotion"]["total_steps"]),
        "minimum_final_validation_accuracy": float(
            spec.data["promotion"]["minimum_final_validation_accuracy"]
        ),
        "underlying_run_spec_sha256": sha256_file(
            entry_dir / "run_spec.v7.json"
        ),
        "official_test_read": False,
    }
    atomic_write_json(
        entry_dir / "rescue_run_spec.json", rescue_run_spec, canonical=True
    )
    atomic_write_json(
        entry_dir / "rescue_summary.json", wrapper_summary, canonical=True
    )
    _publish_entry_completion(
        root,
        stage=stage,
        entry_id=entry_id,
        required_outputs=(
            "best_validation.pt",
            "final.pt",
            "minibatches.json",
            "parameter_diagnostics.csv",
            "rescue_run_spec.json",
            "rescue_summary.json",
            "run_spec.v7.json",
            "step_log.csv",
            "summary.json",
            "validation.json",
        ),
    )
    return {
        "status": "complete",
        "entry_index": entry_index,
        "entry_id": entry_id,
        "underlying_status": underlying["status"],
        "summary": wrapper_summary,
    }


def finalize_promotions(
    *,
    rescue_dir: str | Path,
    stage: str = PROMOTION_STAGE,
) -> dict[str, Any]:
    if stage not in {PROMOTION_STAGE, PROMOTION_RETRY_STAGE}:
        raise ValueError(
            "Expected promotion stage to be 'promotion' or "
            f"{PROMOTION_RETRY_STAGE!r}. Provided value: {stage!r}."
        )
    root, _, manifest, _ = _load_rescue(rescue_dir, stage)
    entry_ids = [str(entry["entry_id"]) for entry in manifest["entries"]]
    completion = _publish_stage_completion(
        root, stage=stage, entry_ids=entry_ids
    )
    summaries = [
        read_json(
            root
            / "stages"
            / stage
            / "entries"
            / entry_id
            / "rescue_summary.json"
        )
        for entry_id in entry_ids
    ]
    return {
        "status": "complete",
        "entry_count": len(entry_ids),
        "passed_entry_ids": [
            summary["entry_id"]
            for summary in summaries
            if summary["status"] == "passed"
        ],
        "completion": completion,
    }


def rescue_status(*, rescue_dir: str | Path) -> dict[str, Any]:
    root = Path(rescue_dir).expanduser().resolve()
    spec = LegacyRescueSpec.from_path(root / "rescue.resolved.json")
    stages: dict[str, Any] = {}
    for stage in (
        DIAGNOSTIC_STAGE,
        PROMOTION_STAGE,
        PROMOTION_RETRY_STAGE,
    ):
        manifest_path = root / "stages" / stage / "manifest.json"
        if not manifest_path.is_file():
            stages[stage] = {"status": "not_planned", "entries": []}
            continue
        manifest = read_json(manifest_path)
        entry_states = []
        for entry in manifest["entries"]:
            entry_id = str(entry["entry_id"])
            summary_path = completed_path(root, stage, entry_id)
            completion = _validate_entry_completion(root, stage, entry_id)
            entry_states.append(
                {
                    "entry_id": entry_id,
                    "status": (
                        "complete"
                        if completion is not None
                        else "incomplete"
                    ),
                    "summary": (
                        read_json(
                            (
                                summary_path
                                if stage == DIAGNOSTIC_STAGE
                                else summary_path.parent
                                / "rescue_summary.json"
                            )
                        )
                        if completion is not None and summary_path.is_file()
                        else None
                    ),
                }
            )
        stages[stage] = {
            "status": (
                "complete"
                if (root / "stages" / stage / "complete.json").is_file()
                else "in_progress"
            ),
            "entry_count": len(entry_states),
            "entries": entry_states,
        }
    return {
        "rescue_id": spec.rescue_id,
        "rescue_dir": str(root),
        "stages": stages,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-parent")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--parent-config", required=True)
    prepare.add_argument("--parent-results-root", required=True)
    prepare.add_argument("--data-root", required=True)
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--download", action="store_true")

    plan = sub.add_parser("plan")
    plan.add_argument("--config", required=True)
    plan.add_argument("--parent-study", required=True)
    plan.add_argument("--results-root", required=True)

    diagnostic = sub.add_parser("run-diagnostic")
    diagnostic.add_argument("--rescue", required=True)
    diagnostic.add_argument("--entry-index", type=int, required=True)
    diagnostic.add_argument("--data-root", required=True)
    diagnostic.add_argument("--device", default="cuda")
    diagnostic.add_argument("--download", action="store_true")

    finalize_diagnostic = sub.add_parser("finalize-diagnostics")
    finalize_diagnostic.add_argument("--rescue", required=True)

    replan_promotion = sub.add_parser("replan-promotions")
    replan_promotion.add_argument("--rescue", required=True)

    promotion = sub.add_parser("run-promotion")
    promotion.add_argument("--rescue", required=True)
    promotion.add_argument("--entry-index", type=int, required=True)
    promotion.add_argument("--data-root", required=True)
    promotion.add_argument("--device", default="cuda")
    promotion.add_argument("--download", action="store_true")
    promotion.add_argument(
        "--stage",
        choices=(PROMOTION_STAGE, PROMOTION_RETRY_STAGE),
        default=PROMOTION_STAGE,
    )

    finalize_promotion = sub.add_parser("finalize-promotions")
    finalize_promotion.add_argument("--rescue", required=True)
    finalize_promotion.add_argument(
        "--stage",
        choices=(PROMOTION_STAGE, PROMOTION_RETRY_STAGE),
        default=PROMOTION_STAGE,
    )

    status = sub.add_parser("status")
    status.add_argument("--rescue", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-parent":
        result = prepare_local_parent(
            rescue_config=args.config,
            parent_config=args.parent_config,
            parent_results_root=args.parent_results_root,
            data_root=args.data_root,
            download=args.download,
            device=args.device,
        )
    elif args.command == "plan":
        result = plan_rescue(
            rescue_config=args.config,
            parent_study_dir=args.parent_study,
            results_root=args.results_root,
        )
    elif args.command == "run-diagnostic":
        result = execute_diagnostic_entry(
            rescue_dir=args.rescue,
            entry_index=args.entry_index,
            data_root=args.data_root,
            download=args.download,
            device=args.device,
        )
    elif args.command == "finalize-diagnostics":
        result = finalize_diagnostics(rescue_dir=args.rescue)
    elif args.command == "replan-promotions":
        result = replan_promotions(rescue_dir=args.rescue)
    elif args.command == "run-promotion":
        result = execute_promotion_entry(
            rescue_dir=args.rescue,
            entry_index=args.entry_index,
            data_root=args.data_root,
            download=args.download,
            device=args.device,
            stage=args.stage,
        )
    elif args.command == "finalize-promotions":
        result = finalize_promotions(
            rescue_dir=args.rescue,
            stage=args.stage,
        )
    else:
        result = rescue_status(rescue_dir=args.rescue)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LegacyRescueSpec",
    "execute_diagnostic_entry",
    "execute_promotion_entry",
    "finalize_diagnostics",
    "finalize_promotions",
    "plan_rescue",
    "prepare_local_parent",
    "replan_promotions",
    "rescue_status",
]
