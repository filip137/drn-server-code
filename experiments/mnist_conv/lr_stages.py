"""Scientific implementations of the four Conv LR-study stages."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .identity import sha256_file
from .io import atomic_write_csv, atomic_write_json, read_json
from .lr_artifacts import load_stage_manifest, validate_stage_completion
from .lr_engine import (
    LRStudyNumericalError,
    build_loader_bundle,
    build_model_runtime,
    evaluate_validation,
    parameter_state_diagnostics,
    parameter_tensor_digest,
    training_step,
    transition_rows,
)
from .lr_protocol import (
    CANDIDATE_TOTAL_STEPS,
    CANDIDATE_WARMUP_STEPS,
    CandidateRunResult,
    PROBE_BATCHES,
    RangeCandidateSet,
    RangeFailure,
    RangeStepRecord,
    RangeTensorRecord,
    analyze_range_stop_gates,
    alpha_candidate_grid,
    architecture_relative_learning_rates,
    candidate_learning_rate_at_step,
    extract_range_candidates,
    layerwise_learning_rate_report,
    layerwise_parameter_groups,
    layerwise_target_learning_rates,
    linear_quantile,
    probe_parameter_relative_rho_unit,
    probe_median_parameter_relative_units,
    probe_rho_unit,
    range_learning_rate_schedule,
    range_normalized_update_schedule,
    select_final_candidate,
    select_two_rho_candidates,
    select_v5_architecture,
    select_v6_conv2_baseline,
    target_learning_rates,
    two_rho_learning_rates,
    two_rho_upper_boundary_axes,
)
from .lr_study_spec import LRStudySpec
from .lr_v6_packing import R3_AUTHORIZATION_TOKEN
from .specs import RunSpec


PROBE_DIAGNOSTIC_COLUMNS = [
    "step",
    "learning_rate",
    "scheduled_rho",
    "scheduled_rho_relative",
    "scheduled_rho_span",
    "parameter",
    "parameter_kind",
    "bounded_gate",
    "report_only",
    "gradient_rms",
    "proposed_update_rms",
    "normalized_update",
    "span_normalized_update",
    "initial_parameter_rms",
    "parameter_relative_update",
    "proposed_bound_crossing_fraction",
    "lower_bound_occupancy",
    "upper_bound_occupancy",
    "combined_bound_occupancy",
    "projection_efficiency",
    "proposal_is_numerically_zero",
    "projection_gate_eligible",
]

STEP_COLUMNS = [
    "step",
    "epoch",
    "batch_in_epoch",
    "learning_rate",
    "scheduled_rho",
    "scheduled_rho_relative",
    "scheduled_rho_span",
    "loss",
    "loss_ema",
    "accuracy",
    "sample_count",
    "status",
    "failure_reason",
]

V6_R3_AUTHORIZATION_ENV = "MNIST_CONV_V6_R3_AUTHORIZATION"
V6_R3_AUTHORIZATION_TOKEN = R3_AUTHORIZATION_TOKEN


def require_v6_r3_authorization() -> None:
    provided = os.environ.get(V6_R3_AUTHORIZATION_ENV)
    if provided != V6_R3_AUTHORIZATION_TOKEN:
        raise RuntimeError(
            f"Expected {V6_R3_AUTHORIZATION_ENV} to exactly authorize the dedicated "
            f"Jean Zay R3 V100 path as {V6_R3_AUTHORIZATION_TOKEN!r}. "
            f"Provided value: {provided!r}."
        )


def _uses_parameter_relative_rho(study: Mapping[str, Any]) -> bool:
    return study["probe"].get("rho_definition") in {
        "initial_parameter_rms",
        "initial_parameter_rms_per_bounded_weight",
        "initial_parameter_rms_per_bounded_weight_median",
        "median_batch_rms_gradient_over_initial_weight_rms_per_bounded_weight",
    }


def _uses_layerwise_rho(study: Mapping[str, Any]) -> bool:
    return study["probe"].get("rho_definition") in {
        "initial_parameter_rms_per_bounded_weight",
        "initial_parameter_rms_per_bounded_weight_median",
        "median_batch_rms_gradient_over_initial_weight_rms_per_bounded_weight",
    }


def _uses_architecture_relative_median(study: Mapping[str, Any]) -> bool:
    return (
        study.get("schema_version") == "mnist-conv-lr-study/v5"
        and study["probe"].get("rho_definition")
        == "initial_parameter_rms_per_bounded_weight_median"
        and study["probe"].get("normalization_statistic") == "median"
        and study["target_policies"].get("alpha_scope") == "architecture"
    )


def _uses_two_rho(study: Mapping[str, Any]) -> bool:
    return (
        study.get("schema_version")
        in {"mnist-conv-lr-study/v6", "mnist-conv-lr-study/v7"}
        and study["probe"].get("rho_definition")
        == "median_batch_rms_gradient_over_initial_weight_rms_per_bounded_weight"
        and study["probe"].get("normalization_statistic") == "median"
    )


def _uses_v6_two_rho(study: Mapping[str, Any]) -> bool:
    return (
        study.get("schema_version") == "mnist-conv-lr-study/v6"
        and _uses_two_rho(study)
    )


def _uses_v7_two_rho(study: Mapping[str, Any]) -> bool:
    return (
        study.get("schema_version") == "mnist-conv-lr-study/v7"
        and _uses_two_rho(study)
    )


def _uses_weight_relative_median(study: Mapping[str, Any]) -> bool:
    return _uses_architecture_relative_median(study) or _uses_two_rho(study)


def _bounded_initial_rms_scales(
    diagnostics: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    scales = {
        name: float(record["rms"])
        for name, record in diagnostics.items()
        if record["bounded_gate"]
    }
    if not scales or any(not math.isfinite(value) or value <= 0.0 for value in scales.values()):
        raise RuntimeError(
            "Expected every bounded initialization tensor to have a positive finite "
            f"RMS scale. Provided value: {scales!r}."
        )
    return scales


def _row_by_id(study: Mapping[str, Any], row_id: str) -> dict[str, Any]:
    matches = [row for row in study["rows"] if row["row_id"] == row_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one frozen row for {row_id!r}. Provided value: {len(matches)}."
        )
    return dict(matches[0])


def _architecture_checkpoint(study_dir: Path, row: Mapping[str, Any]) -> Path:
    return study_dir / "initialization" / f"{row['architecture']}.pt"


def _stage_entry_dir(study_dir: Path, stage: str, entry_id: str) -> Path:
    return study_dir / "stages" / stage / "entries" / entry_id


def _batch_hash(batches: Sequence[Sequence[int]]) -> str:
    from labs.datasets import stable_batch_order_hash

    return stable_batch_order_hash(batches)


def _checkpoint_contract(
    study: Mapping[str, Any],
    architecture: str,
    *,
    checkpoint_path: Path,
    device: str,
) -> dict[str, Any]:
    rows = [row for row in study["rows"] if row["architecture"] == architecture]
    if len(rows) != 3:
        raise RuntimeError(
            f"Expected three schemes for architecture {architecture!r}. Provided value: {len(rows)}."
        )
    natural_digests: dict[str, str] = {}
    natural_diagnostics: dict[str, Any] = {}
    first_runtime = None
    for row in rows:
        runtime = build_model_runtime(study, row, device=device, learning_rate=1.0)
        digest = parameter_tensor_digest(runtime.parameters)
        natural_digests[row["scheme"]] = digest
        natural_diagnostics[row["scheme"]] = parameter_state_diagnostics(runtime.parameters)
        if first_runtime is None:
            first_runtime = runtime
    if len(set(natural_digests.values())) != 1:
        raise RuntimeError(
            "Expected all amplification schemes in one architecture to have identical "
            f"seed-0 initialization tensor digests. Provided value: {natural_digests!r}."
        )
    assert first_runtime is not None
    tensor_digest = next(iter(natural_digests.values()))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_owner = checkpoint_path.with_suffix(".json")

    def validate_checkpoint(path: Path) -> dict[str, str]:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                "Expected initialization checkpoint to be a regular file. "
                f"Provided value: {path}."
            )
        loaded: dict[str, str] = {}
        for candidate_row in rows:
            runtime = build_model_runtime(
                study,
                candidate_row,
                device=device,
                initialization_checkpoint=path,
                learning_rate=1.0,
            )
            loaded[candidate_row["scheme"]] = parameter_tensor_digest(
                runtime.parameters
            )
        if set(loaded.values()) != {tensor_digest}:
            raise RuntimeError(
                "Expected the shared initialization checkpoint to load the same "
                f"tensors in all schemes. Provided value: {loaded!r}."
            )
        return loaded

    if metadata_owner.exists() or metadata_owner.is_symlink():
        if metadata_owner.is_symlink() or not metadata_owner.is_file():
            raise RuntimeError(
                "Expected initialization metadata owner to be a regular file. "
                f"Provided value: {metadata_owner}."
            )
        loaded_digests = validate_checkpoint(checkpoint_path)
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{checkpoint_path.name}.save-",
            suffix=".tmp",
            dir=checkpoint_path.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            first_runtime.save(temporary)
            loaded_digests = validate_checkpoint(temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, checkpoint_path)
            directory_fd = os.open(checkpoint_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        loaded_digests = validate_checkpoint(checkpoint_path)
    result = {
        "schema_version": "mnist-conv-lr-initialization/v1",
        "architecture": architecture,
        "model_seed": 0,
        "checkpoint_path": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_tensor_sha256": tensor_digest,
        "natural_scheme_tensor_sha256": natural_digests,
        "loaded_scheme_tensor_sha256": loaded_digests,
        "parameter_diagnostics": natural_diagnostics[rows[0]["scheme"]],
    }
    return result


def _rescue_import_contract(study: Mapping[str, Any]) -> Mapping[str, Any] | None:
    rescue = study.get("rescue")
    if not isinstance(rescue, Mapping):
        return None
    if rescue.get("import_mode") not in {
        "copy_and_verify_parent_probe",
        "copy_and_verify_parent_assets",
    }:
        return None
    return rescue


def _require_file_sha256(path: Path, expected: str, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(
            f"Expected {label} to be an existing regular file. Provided value: {path}."
        )
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"Expected {label} SHA-256 to be {expected!r}. Provided value: "
            f"{{'path': {str(path)!r}, 'sha256': {observed!r}}}."
        )
    return {"sha256": observed, "bytes": path.stat().st_size}


def _copy_parent_artifact(
    *,
    parent_root: Path,
    study_root: Path,
    relative_path: str,
    expected_sha256: str,
) -> dict[str, Any]:
    source = parent_root / relative_path
    destination = study_root / relative_path
    source_record = _require_file_sha256(
        source, expected_sha256, f"parent artifact {relative_path!r}"
    )
    _atomic_verified_copy(
        source,
        destination,
        expected_sha256,
        f"imported artifact {relative_path!r}",
        immutable_owner_paths=(
            study_root / "reuse/import_receipt.json",
            study_root / "rescue/import_receipt.json",
        ),
    )
    return {
        "source_path": relative_path,
        "destination_path": relative_path,
        **source_record,
    }


def _prepare_rescue_import(
    study: Mapping[str, Any], root: Path, rescue: Mapping[str, Any]
) -> dict[str, Any]:
    """Copy immutable v1 inputs and probe outputs into a portable v2 study."""

    parent_contract = rescue["parent_study"]
    parent_name = f"{parent_contract['name']}--{parent_contract['study_id']}"
    parent_root = root.parent / parent_name
    if parent_root == root or not parent_root.is_dir() or parent_root.is_symlink():
        raise RuntimeError(
            "Expected the completed parent v1 study to be a real sibling directory. "
            f"Provided value: {parent_root}."
        )

    parent_config = parent_root / "study.resolved.json"
    _require_file_sha256(
        parent_config,
        parent_contract["resolved_config_sha256"],
        "parent resolved study config",
    )
    parent_spec = LRStudySpec.from_path(parent_config)
    if parent_spec.study_id != parent_contract["study_id"]:
        raise RuntimeError(
            f"Expected parent study ID to be {parent_contract['study_id']!r}. "
            f"Provided value: {parent_spec.study_id!r}."
        )

    stage_files: dict[str, dict[str, Any]] = {}
    for stage in ("probe", "range", "select"):
        manifest = parent_root / "stages" / stage / "manifest.json"
        validate_stage_completion(study_dir=parent_root, manifest_path=manifest)
        completion = manifest.parent / "complete.json"
        stage_files[stage] = {
            "manifest_path": f"stages/{stage}/manifest.json",
            "manifest_sha256": sha256_file(manifest),
            "completion_path": f"stages/{stage}/complete.json",
            "completion_sha256": sha256_file(completion),
        }

    frozen = rescue["frozen_provenance"]
    if stage_files["probe"]["manifest_sha256"] != frozen["parent_probe_manifest_sha256"]:
        raise RuntimeError(
            "Expected parent probe manifest SHA-256 to match the frozen rescue "
            f"contract. Provided value: {stage_files['probe']['manifest_sha256']!r}."
        )
    if stage_files["probe"]["completion_sha256"] != frozen["parent_probe_completion_sha256"]:
        raise RuntimeError(
            "Expected parent probe completion SHA-256 to match the frozen rescue "
            f"contract. Provided value: {stage_files['probe']['completion_sha256']!r}."
        )

    selection_path = parent_root / "stages/select/entries/selection/selection.json"
    _require_file_sha256(
        selection_path,
        parent_contract["selection_sha256"],
        "parent selection artifact",
    )
    parent_selection = read_json(selection_path)
    selection_rows = {row["row_id"]: row for row in parent_selection["rows"]}

    eligibility = rescue["eligibility"]
    expected_row_ids = [row["row_id"] for row in study["rows"]]
    if expected_row_ids != eligibility["included_row_ids"]:
        raise RuntimeError(
            "Expected rescue rows to equal the frozen eligible parent rows in order. "
            f"Provided value: {expected_row_ids!r}."
        )

    split_provenance_path = parent_root / "split/provenance.json"
    split_provenance = read_json(split_provenance_path)
    for key in ("train_indices_sha256", "validation_indices_sha256"):
        if split_provenance.get(key) != frozen[key]:
            raise RuntimeError(
                f"Expected parent split provenance {key} to be {frozen[key]!r}. "
                f"Provided value: {split_provenance.get(key)!r}."
            )
    if split_provenance.get("official_test_read") is not False:
        raise RuntimeError(
            "Expected parent split provenance official_test_read to be false. "
            f"Provided value: {split_provenance.get('official_test_read')!r}."
        )

    initialization_metadata = read_json(parent_root / "initialization/conv1.json")
    if initialization_metadata.get("parameter_tensor_sha256") != frozen[
        "conv1_parameter_tensor_sha256"
    ]:
        raise RuntimeError(
            "Expected parent Conv1 parameter tensor digest to match the rescue "
            f"contract. Provided value: {initialization_metadata.get('parameter_tensor_sha256')!r}."
        )

    copied: list[dict[str, Any]] = []
    fixed_copies = {
        "split/indices.json": frozen["split_indices_sha256"],
        "split/provenance.json": frozen["split_provenance_sha256"],
        "initialization/conv1.pt": frozen["conv1_checkpoint_sha256"],
        "initialization/conv1.json": frozen["conv1_metadata_sha256"],
    }
    for relative_path, digest in fixed_copies.items():
        copied.append(
            _copy_parent_artifact(
                parent_root=parent_root,
                study_root=root,
                relative_path=relative_path,
                expected_sha256=digest,
            )
        )

    verified_ranges: list[dict[str, Any]] = []
    for row in study["rows"]:
        row_id = row["row_id"]
        row_frozen = frozen["rows"][row_id]
        range_relative = f"stages/range/entries/{row_id}/summary.json"
        range_path = parent_root / range_relative
        range_record = _require_file_sha256(
            range_path,
            row_frozen["parent_range_summary_sha256"],
            f"parent range summary for {row_id!r}",
        )
        parent_range = read_json(range_path)
        parent_selected = selection_rows.get(row_id)
        failure = parent_range.get("failure") or {}
        observed_eligibility = {
            "selection_status": None if parent_selected is None else parent_selected.get("status"),
            "range_status": parent_range.get("status"),
            "unresolved_reason": parent_range.get("unresolved_reason"),
            "failure_kind": failure.get("kind"),
            "failure_onset_step": failure.get("onset_step"),
            "failure_confirmed_step": failure.get("confirmed_step"),
        }
        expected_eligibility = {
            "selection_status": eligibility["parent_row_status"],
            "range_status": eligibility["parent_row_status"],
            "unresolved_reason": eligibility["parent_unresolved_reason"],
            "failure_kind": eligibility["parent_failure_kind"],
            "failure_onset_step": eligibility["parent_failure_onset_step"],
            "failure_confirmed_step": eligibility["parent_failure_confirmed_step"],
        }
        if observed_eligibility != expected_eligibility:
            raise RuntimeError(
                f"Expected parent rescue eligibility for {row_id!r} to be "
                f"{expected_eligibility!r}. Provided value: {observed_eligibility!r}."
            )
        verified_ranges.append(
            {"path": range_relative, "row_id": row_id, **range_record}
        )

        for filename, digest in row_frozen["probe_outputs"].items():
            relative_path = f"stages/probe/entries/{row_id}/{filename}"
            copied.append(
                _copy_parent_artifact(
                    parent_root=parent_root,
                    study_root=root,
                    relative_path=relative_path,
                    expected_sha256=digest,
                )
            )

        imported_summary = read_json(
            root / f"stages/probe/entries/{row_id}/summary.json"
        )
        expected_probe_values = {
            "row": row,
            "status": "complete",
            "rho_unit": row_frozen["rho_unit"],
            "checkpoint_sha256": frozen["conv1_checkpoint_sha256"],
            "initial_parameter_tensor_sha256": frozen["conv1_parameter_tensor_sha256"],
            "final_parameter_tensor_sha256": frozen["conv1_parameter_tensor_sha256"],
            "train_indices_sha256": frozen["train_indices_sha256"],
            "validation_indices_sha256": frozen["validation_indices_sha256"],
            "minibatch_order_sha256": frozen["probe_minibatch_order_sha256"],
        }
        observed_probe_values = {
            key: imported_summary.get(key) for key in expected_probe_values
        }
        if observed_probe_values != expected_probe_values:
            raise RuntimeError(
                f"Expected imported probe values for {row_id!r} to be "
                f"{expected_probe_values!r}. Provided value: {observed_probe_values!r}."
            )

    copied.sort(key=lambda item: item["destination_path"])
    verified_ranges.sort(key=lambda item: item["path"])
    receipt = {
        "schema_version": "mnist-conv-lr-parent-import/v1",
        "mode": "reused_parent_artifact",
        "parent_study": {
            "directory_name": parent_name,
            "study_id": parent_contract["study_id"],
            "resolved_config_sha256": parent_contract["resolved_config_sha256"],
            "selection_sha256": parent_contract["selection_sha256"],
        },
        "validated_parent_stages": stage_files,
        "verified_parent_range_summaries": verified_ranges,
        "copied_artifacts": copied,
        "official_test_read": False,
    }
    receipt_path = root / "rescue/import_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        if receipt_path.is_symlink() or read_json(receipt_path) != receipt:
            raise RuntimeError(
                "Expected existing rescue import receipt to have identical immutable "
                f"content. Provided value: {receipt_path}."
            )
    else:
        atomic_write_json(receipt_path, receipt, canonical=True)
    return {
        "indices_path": root / "split/indices.json",
        "split_provenance_path": root / "split/provenance.json",
        "checkpoint_paths": {"conv1": root / "initialization/conv1.pt"},
        "checkpoint_metadata_paths": {"conv1": root / "initialization/conv1.json"},
        "split_provenance": split_provenance,
        "checkpoints": {"conv1": initialization_metadata},
        "import_receipt_path": receipt_path,
        "import_receipt": receipt,
    }


def _prepare_relative_rho_asset_import(
    study: Mapping[str, Any], root: Path, rescue: Mapping[str, Any]
) -> dict[str, Any]:
    """Copy immutable v1 split/checkpoints for a fresh v3 relative-rho study."""

    parent_contract = rescue["parent_study"]
    parent_name = f"{parent_contract['name']}--{parent_contract['study_id']}"
    parent_root = root.parent / parent_name
    if parent_root == root or not parent_root.is_dir() or parent_root.is_symlink():
        raise RuntimeError(
            "Expected the completed parent v1 study to be a real sibling directory. "
            f"Provided value: {parent_root}."
        )
    parent_config = parent_root / "study.resolved.json"
    _require_file_sha256(
        parent_config,
        parent_contract["resolved_config_sha256"],
        "parent resolved study config",
    )
    parent_spec = LRStudySpec.from_path(parent_config)
    if parent_spec.study_id != parent_contract["study_id"]:
        raise RuntimeError(
            f"Expected parent study ID to be {parent_contract['study_id']!r}. "
            f"Provided value: {parent_spec.study_id!r}."
        )

    frozen = rescue["frozen_provenance"]
    validated_stages: dict[str, dict[str, Any]] = {}
    for stage in ("probe", "range", "select"):
        manifest = parent_root / "stages" / stage / "manifest.json"
        validate_stage_completion(study_dir=parent_root, manifest_path=manifest)
        completion = manifest.parent / "complete.json"
        manifest_digest = sha256_file(manifest)
        completion_digest = sha256_file(completion)
        expected_manifest = frozen.get(f"parent_{stage}_manifest_sha256")
        expected_completion = frozen.get(f"parent_{stage}_completion_sha256")
        if expected_manifest is not None and manifest_digest != expected_manifest:
            raise RuntimeError(
                f"Expected parent {stage} manifest SHA-256 to be "
                f"{expected_manifest!r}. Provided value: {manifest_digest!r}."
            )
        if expected_completion is not None and completion_digest != expected_completion:
            raise RuntimeError(
                f"Expected parent {stage} completion SHA-256 to be "
                f"{expected_completion!r}. Provided value: {completion_digest!r}."
            )
        validated_stages[stage] = {
            "manifest_sha256": manifest_digest,
            "completion_sha256": completion_digest,
        }

    selection_path = parent_root / "stages/select/entries/selection/selection.json"
    _require_file_sha256(
        selection_path,
        parent_contract["selection_sha256"],
        "parent selection artifact",
    )
    selection_rows = {
        row["row_id"]: row for row in read_json(selection_path)["rows"]
    }
    eligibility = rescue["eligibility"]
    expected_row_ids = [row["row_id"] for row in study["rows"]]
    if expected_row_ids != eligibility["included_row_ids"]:
        raise RuntimeError(
            "Expected relative-rho rows to equal the frozen eligible parent rows in "
            f"order. Provided value: {expected_row_ids!r}."
        )
    verified_ranges: list[dict[str, Any]] = []
    for row_id in expected_row_ids:
        row_frozen = frozen["rows"][row_id]
        relative = f"stages/range/entries/{row_id}/summary.json"
        path = parent_root / relative
        record = _require_file_sha256(
            path,
            row_frozen["parent_range_summary_sha256"],
            f"parent range summary for {row_id!r}",
        )
        summary = read_json(path)
        failure = summary.get("failure") or {}
        selected = selection_rows.get(row_id) or {}
        observed = {
            "selection_status": selected.get("status"),
            "range_status": summary.get("status"),
            "unresolved_reason": summary.get("unresolved_reason"),
            "failure_kind": failure.get("kind"),
            "failure_onset_step": failure.get("onset_step"),
            "failure_confirmed_step": failure.get("confirmed_step"),
        }
        expected = {
            "selection_status": eligibility["parent_row_status"],
            "range_status": eligibility["parent_row_status"],
            "unresolved_reason": eligibility["parent_unresolved_reason"],
            "failure_kind": eligibility["parent_failure_kind"],
            "failure_onset_step": eligibility["parent_failure_onset_step"],
            "failure_confirmed_step": eligibility["parent_failure_confirmed_step"],
        }
        if observed != expected:
            raise RuntimeError(
                f"Expected parent eligibility for {row_id!r} to be {expected!r}. "
                f"Provided value: {observed!r}."
            )
        verified_ranges.append({"path": relative, "row_id": row_id, **record})

    split_provenance = read_json(parent_root / "split/provenance.json")
    for key in ("train_indices_sha256", "validation_indices_sha256"):
        if split_provenance.get(key) != frozen[key]:
            raise RuntimeError(
                f"Expected parent split provenance {key} to be {frozen[key]!r}. "
                f"Provided value: {split_provenance.get(key)!r}."
            )
    if split_provenance.get("official_test_read") is not False:
        raise RuntimeError(
            "Expected parent split provenance official_test_read to be false. "
            f"Provided value: {split_provenance.get('official_test_read')!r}."
        )

    copied: list[dict[str, Any]] = []
    fixed_copies = {
        "split/indices.json": frozen["split_indices_sha256"],
        "split/provenance.json": frozen["split_provenance_sha256"],
    }
    initialization_metadata: dict[str, Any] = {}
    for architecture, values in frozen["architectures"].items():
        fixed_copies[f"initialization/{architecture}.pt"] = values[
            "checkpoint_sha256"
        ]
        fixed_copies[f"initialization/{architecture}.json"] = values[
            "metadata_sha256"
        ]
        metadata = read_json(parent_root / f"initialization/{architecture}.json")
        if metadata.get("parameter_tensor_sha256") != values[
            "parameter_tensor_sha256"
        ]:
            raise RuntimeError(
                f"Expected parent {architecture} parameter tensor digest to be "
                f"{values['parameter_tensor_sha256']!r}. Provided value: "
                f"{metadata.get('parameter_tensor_sha256')!r}."
            )
        initialization_metadata[architecture] = metadata
    for relative_path, digest in fixed_copies.items():
        copied.append(
            _copy_parent_artifact(
                parent_root=parent_root,
                study_root=root,
                relative_path=relative_path,
                expected_sha256=digest,
            )
        )

    evidence_contract = rescue.get("evidence_study")
    verified_evidence: dict[str, Any] | None = None
    if isinstance(evidence_contract, Mapping):
        evidence_name = (
            f"{evidence_contract['name']}--{evidence_contract['study_id']}"
        )
        evidence_root = root.parent / evidence_name
        evidence_config = evidence_root / "study.resolved.json"
        _require_file_sha256(
            evidence_config,
            evidence_contract["resolved_config_sha256"],
            "v2 evidence resolved study config",
        )
        evidence_spec = LRStudySpec.from_path(evidence_config)
        if evidence_spec.study_id != evidence_contract["study_id"]:
            raise RuntimeError(
                f"Expected v2 evidence study ID to be {evidence_contract['study_id']!r}. "
                f"Provided value: {evidence_spec.study_id!r}."
            )
        evidence_selection = (
            evidence_root / "stages/select/entries/selection/selection.json"
        )
        _require_file_sha256(
            evidence_selection,
            evidence_contract["selection_sha256"],
            "v2 evidence selection artifact",
        )
        evidence_rows = []
        for row_id, values in evidence_contract["rows"].items():
            relative = f"stages/range/entries/{row_id}/summary.json"
            evidence_rows.append(
                {
                    "row_id": row_id,
                    **_require_file_sha256(
                        evidence_root / relative,
                        values["range_summary_sha256"],
                        f"v2 evidence range summary for {row_id!r}",
                    ),
                }
            )
        verified_evidence = {
            "directory_name": evidence_name,
            "study_id": evidence_contract["study_id"],
            "rows": evidence_rows,
        }

    copied.sort(key=lambda item: item["destination_path"])
    verified_ranges.sort(key=lambda item: item["path"])
    receipt = {
        "schema_version": "mnist-conv-lr-parent-import/v2",
        "mode": "reused_parent_assets_recomputed_probe",
        "parent_study": {
            "directory_name": parent_name,
            "study_id": parent_contract["study_id"],
            "resolved_config_sha256": parent_contract["resolved_config_sha256"],
            "selection_sha256": parent_contract["selection_sha256"],
        },
        "validated_parent_stages": validated_stages,
        "verified_parent_range_summaries": verified_ranges,
        "verified_evidence_study": verified_evidence,
        "copied_artifacts": copied,
        "probe_recomputed_under_relative_rho": True,
        "official_test_read": False,
    }
    receipt_path = root / "rescue/import_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        if receipt_path.is_symlink() or read_json(receipt_path) != receipt:
            raise RuntimeError(
                "Expected existing v3 import receipt to have identical immutable "
                f"content. Provided value: {receipt_path}."
            )
    else:
        atomic_write_json(receipt_path, receipt, canonical=True)
    architectures = tuple(frozen["architectures"])
    return {
        "indices_path": root / "split/indices.json",
        "split_provenance_path": root / "split/provenance.json",
        "checkpoint_paths": {
            architecture: root / "initialization" / f"{architecture}.pt"
            for architecture in architectures
        },
        "checkpoint_metadata_paths": {
            architecture: root / "initialization" / f"{architecture}.json"
            for architecture in architectures
        },
        "split_provenance": split_provenance,
        "checkpoints": initialization_metadata,
        "import_receipt_path": receipt_path,
        "import_receipt": receipt,
    }


def _v6_parent_study_root(root: Path, source_study_id: str) -> Path | None:
    """Locate the content-addressed v5 wrapper without trusting its location."""

    candidates: list[Path] = []
    override = os.environ.get("MNIST_CONV_LR_V5_STUDY")
    if override:
        candidates.append(Path(override).expanduser())
    # Canonical Jean Zay/local layout:
    # <results>/<study-id>/lr_studies/<name>--<study-id>.
    if len(root.parents) >= 3:
        wrapper = root.parents[2] / source_study_id / "lr_studies"
        if wrapper.is_dir() and not wrapper.is_symlink():
            candidates.extend(wrapper.glob(f"*--{source_study_id}"))
    # Compact test/development layout with sibling study directories.
    candidates.extend(root.parent.glob(f"*--{source_study_id}"))
    unique: dict[Path, Path] = {}
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved != root and resolved.is_dir() and not candidate.is_symlink():
            unique[resolved] = resolved
    if len(unique) == 1:
        return next(iter(unique.values()))
    return None


def _v6_artifact_check(path: Path, expected: str, label: str) -> str | None:
    if path.is_symlink() or not path.is_file():
        return f"{label}:missing_or_not_regular"
    observed = sha256_file(path)
    if observed != expected:
        return f"{label}:sha256_mismatch:{observed}"
    return None


def _copy_verified_artifact_to(
    source: Path,
    destination: Path,
    expected_sha256: str,
    label: str,
    *,
    immutable_owner_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    failure = _v6_artifact_check(source, expected_sha256, label)
    if failure is not None:
        raise RuntimeError(
            f"Expected {label} to match its verified digest. Provided value: {failure}."
        )
    _atomic_verified_copy(
        source,
        destination,
        expected_sha256,
        label,
        immutable_owner_paths=immutable_owner_paths,
    )
    return {
        "path": destination.as_posix(),
        "sha256": expected_sha256,
        "bytes": destination.stat().st_size,
    }


def _atomic_verified_copy(
    source: Path,
    destination: Path,
    expected_sha256: str,
    label: str,
    *,
    immutable_owner_paths: Sequence[Path] = (),
) -> None:
    """Verify, copy to a same-directory temporary, then atomically publish."""

    _require_file_sha256(source, expected_sha256, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        try:
            _require_file_sha256(destination, expected_sha256, label)
        except RuntimeError:
            owners = [
                path
                for path in immutable_owner_paths
                if path.exists() or path.is_symlink()
            ]
            if owners:
                raise RuntimeError(
                    f"Expected {label} SHA-256 to remain {expected_sha256!r}; "
                    "immutable owners prohibit repair. "
                    f"Provided owners: {[str(path) for path in owners]!r}."
                )
            if destination.is_dir() and not destination.is_symlink():
                raise RuntimeError(
                    f"Expected repairable {label} to be a file or symlink. "
                    f"Provided value: {destination}."
                )
        else:
            return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.copy-",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        _require_file_sha256(temporary, expected_sha256, f"temporary {label}")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _require_file_sha256(destination, expected_sha256, label)
    finally:
        temporary.unlink(missing_ok=True)


def _v6_materialized_asset_result(
    root: Path, receipt: Mapping[str, Any]
) -> dict[str, Any]:
    for record in receipt["materialized_artifacts"]:
        _require_file_sha256(
            root / record["path"], record["sha256"], record["path"]
        )
    split_provenance = read_json(root / "split/provenance.json")
    initialization = read_json(root / "initialization/conv2.json")
    return {
        "indices_path": root / "split/indices.json",
        "split_provenance_path": root / "split/provenance.json",
        "checkpoint_paths": {"conv2": root / "initialization/conv2.pt"},
        "checkpoint_metadata_paths": {
            "conv2": root / "initialization/conv2.json"
        },
        "split_provenance": split_provenance,
        "checkpoints": {"conv2": initialization},
        "import_receipt_path": root / "reuse/import_receipt.json",
        "import_receipt": dict(receipt),
    }


def _prepare_v6_assets(
    study: Mapping[str, Any],
    root: Path,
    *,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    """Verify v5 inputs, or regenerate their complete dependency closure."""

    receipt_path = root / "reuse/import_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        if receipt_path.is_symlink():
            raise RuntimeError(
                f"Expected the v6 reuse receipt to be a regular file. Provided value: {receipt_path}."
            )
        receipt = read_json(receipt_path)
        if receipt.get("schema_version") != "mnist-conv-lr-v6-reuse/v1":
            raise RuntimeError(
                "Expected the existing v6 reuse receipt schema to be "
                "'mnist-conv-lr-v6-reuse/v1'. Provided value: "
                f"{receipt.get('schema_version')!r}."
            )
        return _v6_materialized_asset_result(root, receipt)

    reuse = study["reuse"]
    expected = reuse["expected_assets"]
    source_id = reuse["source_study_id"]
    source_root = _v6_parent_study_root(root, source_id)
    source_failures: list[str] = []
    source_spec: LRStudySpec | None = None
    if source_root is None:
        source_failures.append("source_study:not_found_or_ambiguous")
    else:
        source_config = source_root / "study.resolved.json"
        failure = _v6_artifact_check(
            source_config,
            reuse["source_resolved_config_sha256"],
            "source_resolved_config",
        )
        if failure is not None:
            source_failures.append(failure)
        else:
            try:
                source_spec = LRStudySpec.from_path(source_config)
            except Exception as exc:
                source_failures.append(
                    f"source_resolved_config:contract_invalid:{type(exc).__name__}"
                )
            if source_spec is not None and (
                source_spec.study_id != source_id
                or source_spec.data.get("schema_version")
                != reuse["source_study_schema_version"]
            ):
                source_failures.append("source_resolved_config:identity_mismatch")

    split_failures = list(source_failures)
    initialization_failures = list(source_failures)
    if source_root is not None and not source_failures:
        for relative, digest, label in (
            (
                "split/indices.json",
                expected["split_indices_sha256"],
                "split_indices",
            ),
            (
                "split/provenance.json",
                expected["split_provenance_sha256"],
                "split_provenance",
            ),
        ):
            failure = _v6_artifact_check(source_root / relative, digest, label)
            if failure is not None:
                split_failures.append(failure)
        if not split_failures:
            provenance = read_json(source_root / "split/provenance.json")
            if provenance.get("train_indices_sha256") != expected[
                "train_indices_sha256"
            ]:
                split_failures.append("split_provenance:train_indices_mismatch")
            if provenance.get("validation_indices_sha256") != expected[
                "validation_indices_sha256"
            ]:
                split_failures.append(
                    "split_provenance:validation_indices_mismatch"
                )
            if provenance.get("official_test_read") is not False:
                split_failures.append("split_provenance:official_test_read")

        for relative, digest, label in (
            (
                "initialization/conv2.pt",
                expected["conv2_checkpoint_sha256"],
                "conv2_checkpoint",
            ),
            (
                "initialization/conv2.json",
                expected["conv2_initialization_record_sha256"],
                "conv2_initialization_record",
            ),
        ):
            failure = _v6_artifact_check(source_root / relative, digest, label)
            if failure is not None:
                initialization_failures.append(failure)
        if not initialization_failures:
            initialization = read_json(source_root / "initialization/conv2.json")
            if initialization.get("parameter_tensor_sha256") != expected[
                "conv2_initialization_tensor_sha256"
            ]:
                initialization_failures.append(
                    "conv2_initialization_record:tensor_digest_mismatch"
                )

    split_reused = not split_failures
    initialization_reused = not initialization_failures
    split_dir = root / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    if split_reused:
        assert source_root is not None
        _copy_parent_artifact(
            parent_root=source_root,
            study_root=root,
            relative_path="split/indices.json",
            expected_sha256=expected["split_indices_sha256"],
        )
        _copy_parent_artifact(
            parent_root=source_root,
            study_root=root,
            relative_path="split/provenance.json",
            expected_sha256=expected["split_provenance_sha256"],
        )
        split_provenance = read_json(root / "split/provenance.json")
    else:
        bundle = build_loader_bundle(
            study,
            data_root=data_root,
            download=download,
            return_source_indices=True,
        )
        index_payload = {
            "schema_version": "mnist-conv-lr-split-indices/v1",
            "source_split": "mnist_train",
            "official_test_read": False,
            "train_indices": list(bundle.train_indices),
            "validation_indices": list(bundle.validation_indices),
        }
        indices_path = root / "split/indices.json"
        if indices_path.exists():
            if read_json(indices_path) != index_payload:
                raise RuntimeError(
                    "Expected an existing regenerated v6 split to equal the "
                    f"deterministic split. Provided value: {indices_path}."
                )
        else:
            atomic_write_json(indices_path, index_payload, canonical=True)
        split_provenance = bundle.provenance(num_epochs=5)
        split_provenance.pop("train_indices")
        split_provenance.pop("validation_indices")
        split_provenance.update(
            {
                "official_test_read": False,
                "indices_path": "indices.json",
                "indices_file_sha256": sha256_file(indices_path),
            }
        )
        provenance_path = root / "split/provenance.json"
        if provenance_path.exists():
            if read_json(provenance_path) != split_provenance:
                raise RuntimeError(
                    "Expected an existing regenerated v6 split provenance to "
                    f"match current indices. Provided value: {provenance_path}."
                )
        else:
            atomic_write_json(provenance_path, split_provenance, canonical=True)
    if split_provenance.get("official_test_read") is not False:
        raise RuntimeError(
            "Expected the v6 train/validation split to prohibit official-test reads. "
            f"Provided value: {split_provenance.get('official_test_read')!r}."
        )

    initialization_dir = root / "initialization"
    initialization_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = initialization_dir / "conv2.pt"
    metadata_path = initialization_dir / "conv2.json"
    if initialization_reused:
        assert source_root is not None
        _copy_parent_artifact(
            parent_root=source_root,
            study_root=root,
            relative_path="initialization/conv2.pt",
            expected_sha256=expected["conv2_checkpoint_sha256"],
        )
        _copy_parent_artifact(
            parent_root=source_root,
            study_root=root,
            relative_path="initialization/conv2.json",
            expected_sha256=expected["conv2_initialization_record_sha256"],
        )
        initialization = read_json(metadata_path)
    elif metadata_path.exists():
        initialization = read_json(metadata_path)
        _require_file_sha256(
            checkpoint,
            initialization["checkpoint_sha256"],
            "regenerated Conv2 initialization checkpoint",
        )
    else:
        initialization = _checkpoint_contract(
            study,
            "conv2",
            checkpoint_path=checkpoint,
            device=device,
        )
        atomic_write_json(metadata_path, initialization, canonical=True)

    probe_common_failures = list(source_failures)
    if not split_reused:
        probe_common_failures.append("dependency:split_regenerated")
    if not initialization_reused:
        probe_common_failures.append("dependency:conv2_initialization_regenerated")
    probe_outputs_by_row: dict[str, list[dict[str, Any]]] = {}
    probe_failures_by_row: dict[str, list[str]] = {
        row["row_id"]: list(probe_common_failures) for row in study["rows"]
    }
    if source_root is not None and not probe_common_failures:
        probe_manifest = source_root / "stages/probe/manifest.json"
        probe_completion = source_root / "stages/probe/complete.json"
        for path, digest, label in (
            (probe_manifest, expected["probe_manifest_sha256"], "probe_manifest"),
            (
                probe_completion,
                expected["probe_completion_sha256"],
                "probe_completion",
            ),
        ):
            failure = _v6_artifact_check(path, digest, label)
            if failure is not None:
                for failures in probe_failures_by_row.values():
                    failures.append(failure)
        if not any(probe_failures_by_row.values()):
            try:
                source_manifest = load_stage_manifest(
                    probe_manifest, study_dir=source_root
                )
                stage_completion = read_json(probe_completion)
            except Exception as exc:
                for failures in probe_failures_by_row.values():
                    failures.append(
                        f"probe_stage:integrity_failure:{type(exc).__name__}"
                    )
            else:
                entries = {
                    entry["entry_id"]: entry for entry in source_manifest["entries"]
                }
                completion_records = {
                    entry["entry_id"]: entry["completion"]
                    for entry in stage_completion["entries"]
                }
                for row in study["rows"]:
                    row_id = row["row_id"]
                    entry = entries.get(row_id)
                    if entry is None:
                        probe_failures_by_row[row_id].append(
                            "probe_stage:row_entry_missing"
                        )
                        continue
                    completion_record = completion_records.get(row_id)
                    if completion_record is None or _v6_artifact_check(
                        source_root / entry["completion_path"],
                        completion_record["sha256"],
                        f"probe completion for {row_id!r}",
                    ) is not None:
                        probe_failures_by_row[row_id].append(
                            "probe_stage:row_completion_invalid"
                        )
                        continue
                    completion = read_json(source_root / entry["completion_path"])
                    outputs = list(completion["outputs"])
                    records_by_name = {
                        Path(record["path"]).name: dict(record)
                        for record in outputs
                    }
                    required_names = {
                        "minibatches.json",
                        "parameter_diagnostics.csv",
                        "step_log.csv",
                        "summary.json",
                    }
                    if set(records_by_name) != required_names:
                        probe_failures_by_row[row_id].append(
                            "probe_stage:unexpected_output_set"
                        )
                        continue
                    invalid_output = next(
                        (
                            filename
                            for filename, record in records_by_name.items()
                            if _v6_artifact_check(
                                source_root / record["path"],
                                record["sha256"],
                                f"probe output {filename!r} for {row_id!r}",
                            )
                            is not None
                            or (source_root / record["path"]).stat().st_size
                            != int(record["bytes"])
                        ),
                        None,
                    )
                    if invalid_output is not None:
                        probe_failures_by_row[row_id].append(
                            f"probe_stage:row_output_invalid:{invalid_output}"
                        )
                        continue
                    summary_record = records_by_name["summary.json"]
                    if summary_record["sha256"] != expected[
                        "probe_summary_sha256_by_row"
                    ][row_id]:
                        probe_failures_by_row[row_id].append(
                            "probe_summary:declared_digest_mismatch"
                        )
                        continue
                    source_summary = read_json(source_root / summary_record["path"])
                    source_minibatches = read_json(
                        source_root / records_by_name["minibatches.json"]["path"]
                    )
                    if (
                        source_summary.get("minibatch_order_sha256")
                        != expected["probe_minibatch_order_sha256"]
                        or source_minibatches.get("batch_order_sha256")
                        != expected["probe_minibatch_order_sha256"]
                    ):
                        probe_failures_by_row[row_id].append(
                            "probe_minibatches:order_digest_mismatch"
                        )
                        continue
                    probe_outputs_by_row[row_id] = outputs

    cached_probe_outputs_by_row: dict[str, list[dict[str, Any]]] = {}
    if source_root is not None:
        rows_by_id = {row["row_id"]: row for row in study["rows"]}
        for row_id, source_outputs in probe_outputs_by_row.items():
            if probe_failures_by_row[row_id]:
                continue
            try:
                source_by_name = {
                    Path(record["path"]).name: record for record in source_outputs
                }
                cached: list[dict[str, Any]] = []
                cache_dir = root / "reuse/probes" / row_id
                for filename in (
                    "minibatches.json",
                    "parameter_diagnostics.csv",
                    "step_log.csv",
                ):
                    source_record = source_by_name[filename]
                    destination = cache_dir / filename
                    copied = _copy_verified_artifact_to(
                        source_root / source_record["path"],
                        destination,
                        source_record["sha256"],
                        f"cached source probe {filename!r} for {row_id!r}",
                        immutable_owner_paths=(
                            root / "reuse/import_receipt.json",
                        ),
                    )
                    copied["path"] = destination.relative_to(root).as_posix()
                    cached.append(copied)
                source_summary_record = source_by_name["summary.json"]
                _require_file_sha256(
                    source_root / source_summary_record["path"],
                    source_summary_record["sha256"],
                    f"source probe summary for {row_id!r}",
                )
                neutral_summary = _v6_probe_summary(
                    study,
                    rows_by_id[row_id],
                    read_json(source_root / source_summary_record["path"]),
                    source_measurement_mode="verified_v5_measurement",
                )
                cached_summary = cache_dir / "summary.json"
                if cached_summary.exists():
                    if read_json(cached_summary) != neutral_summary:
                        raise RuntimeError(
                            "Expected an existing cached v6 probe summary to "
                            f"match the verified measurement. Provided value: {cached_summary}."
                        )
                else:
                    atomic_write_json(cached_summary, neutral_summary, canonical=True)
                cached.append(
                    {
                        "path": cached_summary.relative_to(root).as_posix(),
                        "sha256": sha256_file(cached_summary),
                        "bytes": cached_summary.stat().st_size,
                    }
                )
                cached_probe_outputs_by_row[row_id] = sorted(
                    cached, key=lambda item: item["path"]
                )
            except Exception as exc:
                probe_failures_by_row[row_id].append(
                    f"probe_cache:materialization_failure:{type(exc).__name__}"
                )

    materialized = [
        {
            "path": relative,
            "sha256": sha256_file(root / relative),
            "bytes": (root / relative).stat().st_size,
        }
        for relative in (
            "split/indices.json",
            "split/provenance.json",
            "initialization/conv2.pt",
            "initialization/conv2.json",
        )
    ]
    receipt = {
        "schema_version": "mnist-conv-lr-v6-reuse/v1",
        "status": "complete",
        "source_study_id": source_id,
        "source_study_found": source_root is not None,
        "source_study_directory": None if source_root is None else source_root.name,
        "split": {
            "mode": "reused" if split_reused else "regenerated",
            "fallback_reasons": split_failures,
        },
        "conv2_initialization": {
            "mode": "reused" if initialization_reused else "regenerated",
            "fallback_reasons": initialization_failures,
            "parameter_tensor_sha256": initialization["parameter_tensor_sha256"],
        },
        "probes": {
            row["row_id"]: {
                "mode": (
                    "reused_measurement"
                    if not probe_failures_by_row[row["row_id"]]
                    else "regenerate"
                ),
                "fallback_reasons": probe_failures_by_row[row["row_id"]],
                "source_outputs": probe_outputs_by_row.get(row["row_id"], []),
                "cached_outputs": cached_probe_outputs_by_row.get(
                    row["row_id"], []
                ),
            }
            for row in study["rows"]
        },
        "materialized_artifacts": sorted(materialized, key=lambda item: item["path"]),
        "official_test_read": False,
    }
    atomic_write_json(receipt_path, receipt, canonical=True)
    return _v6_materialized_asset_result(root, receipt)


def prepare_study_assets(
    study: Mapping[str, Any],
    study_dir: str | Path,
    *,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    """Materialize and validate split indices and one checkpoint per architecture."""

    root = Path(study_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if _uses_v6_two_rho(study):
        return _prepare_v6_assets(
            study,
            root,
            data_root=data_root,
            download=download,
            device=device,
        )
    rescue = _rescue_import_contract(study)
    if rescue is not None:
        if rescue["import_mode"] == "copy_and_verify_parent_probe":
            return _prepare_rescue_import(study, root, rescue)
        return _prepare_relative_rho_asset_import(study, root, rescue)
    split_dir = root / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_loader_bundle(
        study,
        data_root=data_root,
        download=download,
        return_source_indices=True,
    )
    index_payload = {
        "schema_version": "mnist-conv-lr-split-indices/v1",
        "source_split": "mnist_train",
        "official_test_read": False,
        "train_indices": list(bundle.train_indices),
        "validation_indices": list(bundle.validation_indices),
    }
    indices_path = split_dir / "indices.json"
    if indices_path.exists():
        if read_json(indices_path) != index_payload:
            raise RuntimeError(
                f"Expected existing split indices to equal the deterministic seed-0 split. Provided value: {indices_path}."
            )
    else:
        atomic_write_json(indices_path, index_payload, canonical=True)
    provenance = bundle.provenance(
        num_epochs=int(study["candidate_training"]["epochs"])
    )
    provenance.pop("train_indices")
    provenance.pop("validation_indices")
    provenance.update(
        {
            "official_test_read": False,
            "indices_path": "indices.json",
            "indices_file_sha256": sha256_file(indices_path),
        }
    )
    provenance_path = split_dir / "provenance.json"
    if provenance_path.exists():
        if read_json(provenance_path) != provenance:
            raise RuntimeError(
                f"Expected existing split provenance to match current indices. Provided value: {provenance_path}."
            )
    else:
        atomic_write_json(provenance_path, provenance, canonical=True)

    initialization_dir = root / "initialization"
    initialization_dir.mkdir(parents=True, exist_ok=True)
    architectures = tuple(
        dict.fromkeys(str(row["architecture"]) for row in study["rows"])
    )
    checkpoint_records: dict[str, Any] = {}
    for architecture in architectures:
        checkpoint = initialization_dir / f"{architecture}.pt"
        metadata_path = initialization_dir / f"{architecture}.json"
        if metadata_path.exists():
            record = read_json(metadata_path)
            expected_metadata = {
                "schema_version": "mnist-conv-lr-initialization/v1",
                "architecture": architecture,
                "model_seed": int(study["model"]["model_seed"]),
                "checkpoint_path": checkpoint.name,
            }
            observed_metadata = {
                key: record.get(key) for key in expected_metadata
            }
            if observed_metadata != expected_metadata:
                raise RuntimeError(
                    "Expected existing initialization metadata to match the frozen "
                    f"architecture contract. Provided value: {metadata_path}."
                )
            _require_file_sha256(
                checkpoint,
                record.get("checkpoint_sha256"),
                f"{architecture} shared initialization checkpoint",
            )
        else:
            record = _checkpoint_contract(
                study,
                architecture,
                checkpoint_path=checkpoint,
                device=device,
            )
            atomic_write_json(metadata_path, record, canonical=True)
        checkpoint_records[architecture] = record
    return {
        "indices_path": indices_path,
        "split_provenance_path": provenance_path,
        "checkpoint_paths": {
            architecture: initialization_dir / f"{architecture}.pt"
            for architecture in architectures
        },
        "checkpoint_metadata_paths": {
            architecture: initialization_dir / f"{architecture}.json"
            for architecture in architectures
        },
        "split_provenance": provenance,
        "checkpoints": checkpoint_records,
    }


def execute_v6_reuse_audit(
    study: Mapping[str, Any],
    study_dir: str | Path,
    *,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    """Materialize v6 assets and publish reuse/fallback provenance."""

    require_v6_r3_authorization()
    if not _uses_v6_two_rho(study):
        raise ValueError(
            "Expected an mnist-conv-lr-study/v6 two-rho study. "
            f"Provided value: {study.get('schema_version')!r}."
        )
    root = Path(study_dir).resolve()
    prepared = prepare_study_assets(
        study,
        root,
        data_root=data_root,
        download=download,
        device=device,
    )
    receipt = prepared["import_receipt"]
    output_dir = _stage_entry_dir(root, "audit", "reuse-audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "artifact": "split",
            "row_id": None,
            "mode": receipt["split"]["mode"],
            "fallback_reasons": json.dumps(
                receipt["split"]["fallback_reasons"], separators=(",", ":")
            ),
        },
        {
            "artifact": "conv2_initialization",
            "row_id": None,
            "mode": receipt["conv2_initialization"]["mode"],
            "fallback_reasons": json.dumps(
                receipt["conv2_initialization"]["fallback_reasons"],
                separators=(",", ":"),
            ),
        },
        *(
            {
                "artifact": "probe",
                "row_id": row_id,
                "mode": value["mode"],
                "fallback_reasons": json.dumps(
                    value["fallback_reasons"], separators=(",", ":")
                ),
            }
            for row_id, value in receipt["probes"].items()
        ),
    ]
    summary = {
        "schema_version": "mnist-conv-lr-v6-reuse-audit/v1",
        "status": "complete",
        "study_role": "ordinary_mnist_optimization_diagnostic",
        "source_study_id": receipt["source_study_id"],
        "split_mode": receipt["split"]["mode"],
        "conv2_initialization_mode": receipt["conv2_initialization"]["mode"],
        "probe_modes_by_row": {
            row_id: value["mode"] for row_id, value in receipt["probes"].items()
        },
        "regeneration_runs_on_compute_node": True,
        "official_test_read": False,
        "final_paper_training_authorized": False,
    }
    atomic_write_csv(
        output_dir / "reuse_audit.csv",
        ["artifact", "row_id", "mode", "fallback_reasons"],
        records,
    )
    atomic_write_json(output_dir / "summary.json", summary, canonical=True)
    return summary


def execute_v7_asset_audit(
    study: Mapping[str, Any],
    study_dir: str | Path,
    *,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    """Create and verify the v7 split and shared Conv3 initialization."""

    if not _uses_v7_two_rho(study):
        raise ValueError(
            "Expected an mnist-conv-lr-study/v7 Conv3 two-rho study. "
            f"Provided value: {study.get('schema_version')!r}."
        )
    root = Path(study_dir).resolve()
    prepared = prepare_study_assets(
        study,
        root,
        data_root=data_root,
        download=download,
        device=device,
    )
    output_dir = _stage_entry_dir(root, "audit", "asset-audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = Path(prepared["indices_path"])
    split_provenance_path = Path(prepared["split_provenance_path"])
    checkpoint_path = Path(prepared["checkpoint_paths"]["conv3"])
    checkpoint_metadata_path = Path(
        prepared["checkpoint_metadata_paths"]["conv3"]
    )
    records = [
        {
            "artifact": "split_indices",
            "path": str(split_path.relative_to(root)),
            "sha256": sha256_file(split_path),
            "status": "verified",
        },
        {
            "artifact": "split_provenance",
            "path": str(split_provenance_path.relative_to(root)),
            "sha256": sha256_file(split_provenance_path),
            "status": "verified",
        },
        {
            "artifact": "conv3_initialization_checkpoint",
            "path": str(checkpoint_path.relative_to(root)),
            "sha256": sha256_file(checkpoint_path),
            "status": "verified",
        },
        {
            "artifact": "conv3_initialization_record",
            "path": str(checkpoint_metadata_path.relative_to(root)),
            "sha256": sha256_file(checkpoint_metadata_path),
            "status": "verified",
        },
    ]
    summary = {
        "schema_version": "mnist-conv-lr-v7-asset-audit/v1",
        "status": "complete",
        "study_role": "ordinary_mnist_conv3_optimization_diagnostic",
        "architecture": "conv3",
        "model_seed": 0,
        "shared_initialization": True,
        "gain_calibration_dataset": "deterministic_medium_affine_mnist",
        "optimization_dataset": "ordinary_mnist",
        "gain_recalibrated": False,
        "artifacts": records,
        "official_test_read": False,
        "final_paper_training_authorized": False,
        "medium_affine_conv3_handoff_replaced": False,
    }
    atomic_write_csv(
        output_dir / "asset_audit.csv",
        ["artifact", "path", "sha256", "status"],
        records,
    )
    atomic_write_json(output_dir / "summary.json", summary, canonical=True)
    return summary


def execute_anchor_audit(
    study: Mapping[str, Any], study_dir: str | Path
) -> dict[str, Any]:
    """Publish the immutable historical-anchor decision that precedes v5 probes.

    The legacy roots are provenance only: their architectures and initialization
    bytes cannot be made identical to the current frozen model, and reading
    their reported test metrics is prohibited.  The source contract therefore
    predeclares rounded fallback centers; this stage verifies and materializes
    that decision without touching official MNIST test data.
    """

    if not _uses_architecture_relative_median(study):
        raise ValueError(
            "Expected an mnist-conv-lr-study/v5 architecture-relative study. "
            f"Provided value: {study.get('schema_version')!r}."
        )
    root = Path(study_dir).resolve()
    output_dir = _stage_entry_dir(root, "audit", "anchor-audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = study["anchor_audit"]
    policy = study["target_policies"]
    centers = contract["fallback_centers_by_architecture_and_arm"]
    if centers != policy["centers_by_architecture_and_arm"]:
        raise RuntimeError(
            "Expected the immutable anchor-audit centers to equal the v5 target-policy "
            f"centers. Provided value: audit={centers!r}, policy="
            f"{policy['centers_by_architecture_and_arm']!r}."
        )
    grids = {
        architecture: {
            arm: list(alpha_candidate_grid(float(center)))
            for arm, center in arms.items()
        }
        for architecture, arms in centers.items()
    }
    expected_grids = {
        architecture: {
            arm: [
                float(center) * float(factor)
                for factor in policy["candidate_factors"]
            ]
            for arm, center in arms.items()
        }
        for architecture, arms in centers.items()
    }
    for architecture in grids:
        for arm in grids[architecture]:
            if any(
                not math.isclose(a, b, rel_tol=1e-15, abs_tol=0.0)
                for a, b in zip(
                    grids[architecture][arm],
                    expected_grids[architecture][arm],
                    strict=True,
                )
            ):
                raise RuntimeError(
                    "Expected the configured candidate factors to be exactly "
                    "center/3, center, 3*center. "
                    f"Provided value: {expected_grids[architecture][arm]!r}."
                )
    anchors = [dict(value) for value in contract["anchors"]]
    summary = {
        "schema_version": "mnist-conv-lr-v5-anchor-audit/v1",
        "status": "complete",
        "derivation_status": contract["derivation_status"],
        "fallback_reason": contract["fallback_reason"],
        "formula": contract["formula"],
        "normalization_statistic": "median",
        "probe_batches": 32,
        "batch_size": 16,
        "weights_only": True,
        "official_test_read": False,
        "historical_test_metrics_used": False,
        "anchors": anchors,
        "centers_by_architecture_and_arm": centers,
        "candidate_grids_by_architecture_and_arm": grids,
        "warnings": list(contract["warnings"]),
    }
    anchor_fields = list(anchors[0]) if anchors else []
    atomic_write_json(output_dir / "summary.json", summary, canonical=True)
    atomic_write_csv(output_dir / "anchors.csv", anchor_fields, anchors)
    return summary


def _v6_probe_summary(
    study: Mapping[str, Any],
    row: Mapping[str, Any],
    base: Mapping[str, Any],
    *,
    source_measurement_mode: str,
) -> dict[str, Any]:
    units = base.get("median_units_by_weight")
    if units is None:
        units = base.get("rho_unit_relative_by_parameter")
    if not isinstance(units, Mapping):
        raise RuntimeError(
            "Expected a per-weight median measurement in the Conv2 probe summary. "
            f"Provided value: {units!r}."
        )
    normalized_units = {name: float(value) for name, value in units.items()}
    depth = len(
        study["model"]["architectures"][row["architecture"]]["channels"]
    )
    expected_weights = {
        *(f"ConvWeight_{index}" for index in range(depth)),
        "DenseWeight_0",
    }
    if set(normalized_units) != expected_weights:
        raise RuntimeError(
            f"Expected {row['architecture']} median measurements for every "
            "convolutional weight and the dense weight. "
            f"Provided value: {normalized_units!r}."
        )
    grid = study["rho_grid"]
    if study.get("schema_version") == "mnist-conv-lr-study/v7":
        rho_conv_grid = grid["core"]["rho_conv"]
        rho_dense_grid = grid["core"]["rho_dense"]
        schema_version = "mnist-conv-lr-probe-result/v6"
    else:
        rho_conv_grid = grid["rho_conv"]
        rho_dense_grid = grid["rho_dense"]
        schema_version = "mnist-conv-lr-probe-result/v5"
    result = {
        "schema_version": schema_version,
        "row": dict(row),
        "status": "complete",
        "failure_reason": None,
        "source_measurement_mode": source_measurement_mode,
        "checkpoint_sha256": base["checkpoint_sha256"],
        "initial_parameter_tensor_sha256": base[
            "initial_parameter_tensor_sha256"
        ],
        "final_parameter_tensor_sha256": base["final_parameter_tensor_sha256"],
        "train_indices_sha256": base["train_indices_sha256"],
        "validation_indices_sha256": base["validation_indices_sha256"],
        "minibatch_order_sha256": base["minibatch_order_sha256"],
        "probe_batches": int(base["probe_batches"]),
        "bounded_tensor_batch_value_count": int(
            base["bounded_tensor_batch_value_count"]
        ),
        "rho_definition": study["probe"]["rho_definition"],
        "normalization_statistic": "median",
        "median_units_by_weight": normalized_units,
        "relative_q90": base.get("rho_unit_relative_q90"),
        "relative_q90_by_weight": dict(
            base.get("rho_unit_relative_q90_by_parameter") or {}
        ),
        "relative_median_by_weight": dict(
            base.get("rho_unit_relative_q50_by_parameter") or normalized_units
        ),
        "span_q90": base.get("rho_span_unit_diagnostic"),
        "span_q90_by_weight": dict(
            base.get("rho_span_unit_diagnostic_by_parameter") or {}
        ),
        "bias_weight_lr_groups": {
            name: list(members)
            for name, members in base["bias_weight_lr_groups"].items()
        },
        "initial_bounded_parameter_rms": dict(
            base["initial_bounded_parameter_rms"]
        ),
        "rho_conv_grid": [float(value) for value in rho_conv_grid],
        "rho_dense_grid": [
            float(value) for value in rho_dense_grid
        ],
        "weights_only_for_normalization": True,
        "official_test_read": False,
    }
    if study.get("schema_version") == "mnist-conv-lr-study/v7":
        result.update(
            {
                "gain_calibration_dataset": "deterministic_medium_affine_mnist",
                "optimization_dataset": "ordinary_mnist",
            }
        )
    return result


def _reuse_v6_probe_if_available(
    study: Mapping[str, Any], root: Path, row: Mapping[str, Any]
) -> dict[str, Any] | None:
    receipt = read_json(root / "reuse/import_receipt.json")
    probe_record = receipt["probes"][row["row_id"]]
    if probe_record["mode"] != "reused_measurement":
        return None
    output_dir = _stage_entry_dir(root, "probe", row["row_id"])
    cached = {
        Path(record["path"]).name: record
        for record in probe_record.get("cached_outputs", [])
    }
    required = {
        "minibatches.json",
        "parameter_diagnostics.csv",
        "step_log.csv",
        "summary.json",
    }
    if set(cached) != required:
        return None
    for filename, record in cached.items():
        if _v6_artifact_check(
            root / record["path"],
            record["sha256"],
            f"cached probe {filename!r} for {row['row_id']!r}",
        ) is not None:
            # Cache corruption is never accepted as reuse.  The caller falls
            # through to a fresh measurement on this compute node.
            return None
    for filename, record in cached.items():
        _copy_verified_artifact_to(
            root / record["path"],
            output_dir / filename,
            record["sha256"],
            f"v6-owned cached probe {filename!r} for {row['row_id']!r}",
            immutable_owner_paths=(output_dir / "complete.json",),
        )
    return read_json(output_dir / "summary.json")


def execute_probe_entry(
    study: Mapping[str, Any],
    study_dir: str | Path,
    row_id: str,
    *,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    if _uses_v6_two_rho(study):
        require_v6_r3_authorization()
    row = _row_by_id(study, row_id)
    root = Path(study_dir).resolve()
    output_dir = _stage_entry_dir(root, "probe", row_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    v6_source_measurement_mode = (
        "fresh_v7_measurement"
        if _uses_v7_two_rho(study)
        else "fresh_v6_measurement"
    )
    if _uses_v6_two_rho(study):
        reused = _reuse_v6_probe_if_available(study, root, row)
        if reused is not None:
            return reused
        receipt = read_json(root / "reuse/import_receipt.json")
        if receipt["probes"][row_id]["mode"] == "reused_measurement":
            v6_source_measurement_mode = (
                "fresh_v6_measurement_after_reuse_cache_rejected"
            )
    rescue = _rescue_import_contract(study)
    if (
        rescue is not None
        and rescue["import_mode"] == "copy_and_verify_parent_probe"
    ):
        expected = rescue["frozen_provenance"]["rows"][row_id]["probe_outputs"]
        for filename, digest in expected.items():
            _require_file_sha256(
                output_dir / filename,
                digest,
                f"imported probe output {filename!r} for {row_id!r}",
            )
        summary = read_json(output_dir / "summary.json")
        if summary.get("rho_unit") != rescue["frozen_provenance"]["rows"][row_id][
            "rho_unit"
        ]:
            raise RuntimeError(
                f"Expected imported probe rho_unit for {row_id!r} to be "
                f"{rescue['frozen_provenance']['rows'][row_id]['rho_unit']!r}. "
                f"Provided value: {summary.get('rho_unit')!r}."
            )
        return {
            "status": "complete",
            "mode": "reused_parent_artifact",
            "row_id": row_id,
            "rho_unit": summary["rho_unit"],
        }
    checkpoint = _architecture_checkpoint(root, row)
    runtime = build_model_runtime(
        study,
        row,
        device=device,
        initialization_checkpoint=checkpoint,
        learning_rate=float(study["probe"]["nominal_lr"]),
    )
    initial_tensor_digest = parameter_tensor_digest(runtime.parameters)
    initial_diagnostics = parameter_state_diagnostics(runtime.parameters)
    bounded_initial_scales = _bounded_initial_rms_scales(initial_diagnostics)
    bundle = build_loader_bundle(
        study,
        data_root=data_root,
        download=download,
        return_source_indices=True,
    )
    bundle.reset_train_shuffle()
    diagnostics: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    minibatches: list[tuple[int, ...]] = []
    bounded_span_rhos: list[float] = []
    bounded_span_rhos_by_parameter: dict[str, list[float]] = {
        name: [] for name in bounded_initial_scales
    }
    bounded_relative_rhos: dict[str, list[float]] = {
        name: [] for name in bounded_initial_scales
    }
    nominal_lr = float(study["probe"]["nominal_lr"])
    iterator = iter(bundle.train_loader)
    for step in range(1, PROBE_BATCHES + 1):
        result = training_step(
            runtime,
            next(iterator),
            learning_rate=nominal_lr,
            restore=True,
            zero_proposal_epsilon=float(
                study["range_test"]["gates"]["projection_efficiency"]["zero_update_epsilon"]
            ),
        )
        minibatches.append(result.source_indices)
        transition_records = transition_rows(
            result.transition,
            step=step,
            learning_rate=nominal_lr,
            rho_schedule=None,
            bounded_initial_rms_scales=bounded_initial_scales,
        )
        diagnostics.extend(transition_records)
        bounded_span_rhos.extend(
            record["normalized_update"]
            for record in transition_records
            if record["bounded_gate"]
        )
        for record in transition_records:
            if record["bounded_gate"]:
                bounded_span_rhos_by_parameter[record["parameter"]].append(
                    float(record["normalized_update"])
                )
                bounded_relative_rhos[record["parameter"]].append(
                    float(record["parameter_relative_update"])
                )
        step_rows.append(
            {
                "step": step,
                "epoch": 0,
                "batch_in_epoch": step,
                "learning_rate": nominal_lr,
                "scheduled_rho": None,
                "loss": result.loss,
                "loss_ema": None,
                "accuracy": result.accuracy,
                "sample_count": result.sample_count,
                "status": "complete",
                "failure_reason": None,
            }
        )
        observed_digest = parameter_tensor_digest(runtime.parameters)
        if observed_digest != initial_tensor_digest:
            raise RuntimeError(
                "Expected shadow SGD proposal to restore parameters exactly. "
                f"Provided value at step {step}: {observed_digest}."
            )
    span_unit = probe_rho_unit(bounded_span_rhos)
    relative = _uses_parameter_relative_rho(study)
    layerwise = _uses_layerwise_rho(study)
    architecture_median = _uses_architecture_relative_median(study)
    two_rho = _uses_two_rho(study)
    weight_median = architecture_median or two_rho
    relative_q90_unit = None
    relative_q90_by_parameter: dict[str, float] = {}
    if relative:
        relative_q90_unit, relative_q90_by_parameter = probe_parameter_relative_rho_unit(
            bounded_relative_rhos
        )
        if weight_median:
            unit_by_parameter = probe_median_parameter_relative_units(
                bounded_relative_rhos
            )
            unit = max(unit_by_parameter.values())
        else:
            unit = relative_q90_unit
            unit_by_parameter = relative_q90_by_parameter
    else:
        unit = span_unit
        unit_by_parameter = {}
    raw_rates = (
        {}
        if weight_median
        else target_learning_rates(unit, study["probe"]["rho_targets"])
    )
    parameter_names = [item.name for item in runtime.parameters]
    parameter_groups = (
        layerwise_parameter_groups(parameter_names) if layerwise else {}
    )
    relative_q50_by_parameter = (
        {
            name: linear_quantile(values, 0.5)
            for name, values in bounded_relative_rhos.items()
        }
        if relative
        else {}
    )
    span_q90_by_parameter = {
        name: linear_quantile(values, 0.9)
        for name, values in bounded_span_rhos_by_parameter.items()
    }
    if two_rho:
        raw_learning_rates = []
    elif architecture_median:
        raw_learning_rates = []
        thresholds = study["artifacts"]["large_raw_lr_reporting"]["thresholds"]
        architecture = row["architecture"]
        for arm, arm_contract in study["target_policies"]["arms"].items():
            multipliers = {
                name: (
                    float(arm_contract["dense_multiplier_by_architecture"][architecture])
                    if name.startswith("DenseWeight_")
                    else float(arm_contract["conv_multiplier"])
                )
                for name in unit_by_parameter
            }
            center = float(
                study["target_policies"]["centers_by_architecture_and_arm"]
                [architecture][arm]
            )
            for alpha_role, factor in zip(
                study["target_policies"]["candidate_roles"],
                study["target_policies"]["candidate_factors"],
                strict=True,
            ):
                alpha = center * float(factor)
                vector = architecture_relative_learning_rates(
                    unit_by_parameter,
                    alpha,
                    parameter_names,
                    target_multipliers=multipliers,
                )
                report = layerwise_learning_rate_report(
                    vector, thresholds=thresholds
                )
                raw_learning_rates.append(
                    {
                        "arm": arm,
                        "alpha_role": alpha_role,
                        "alpha": alpha,
                        "target_multipliers_by_weight": multipliers,
                        "learning_rate": report["maximum_weight_learning_rate"],
                        "learning_rates_by_parameter": vector,
                        "learning_rates_by_weight": report["weight_learning_rates"],
                        "large_learning_rate_report": report,
                    }
                )
    elif layerwise:
        raw_learning_rates = []
        thresholds = study["artifacts"]["large_raw_lr_reporting"]["thresholds"]
        for raw_target in study["probe"]["rho_targets"]:
            target = float(raw_target)
            vector = layerwise_target_learning_rates(
                unit_by_parameter, target, parameter_names
            )
            report = layerwise_learning_rate_report(
                vector, thresholds=thresholds
            )
            raw_learning_rates.append(
                {
                    "rho_target": target,
                    "learning_rate": report["maximum_weight_learning_rate"],
                    "learning_rates_by_parameter": vector,
                    "learning_rates_by_weight": report["weight_learning_rates"],
                    "large_learning_rate_report": report,
                }
            )
    else:
        raw_learning_rates = [
            {"rho_target": target, "learning_rate": raw_rates[float(target)]}
            for target in study["probe"]["rho_targets"]
        ]
    minibatch_payload = {
        "schema_version": "mnist-conv-lr-minibatches/v1",
        "row_id": row_id,
        "stage": "probe",
        "batches": [list(batch) for batch in minibatches],
        "batch_order_sha256": _batch_hash(minibatches),
    }
    summary = {
        "schema_version": (
            "mnist-conv-lr-probe-result/v4"
            if weight_median
            else "mnist-conv-lr-probe-result/v3"
            if layerwise
            else "mnist-conv-lr-probe-result/v2"
            if relative
            else "mnist-conv-lr-probe-result/v1"
        ),
        "row": row,
        "status": "complete",
        "failure_reason": None,
        "checkpoint_sha256": sha256_file(checkpoint),
        "initial_parameter_tensor_sha256": initial_tensor_digest,
        "final_parameter_tensor_sha256": parameter_tensor_digest(runtime.parameters),
        "train_indices_sha256": bundle.train_indices_hash,
        "validation_indices_sha256": bundle.validation_indices_hash,
        "minibatch_order_sha256": minibatch_payload["batch_order_sha256"],
        "probe_batches": PROBE_BATCHES,
        "bounded_tensor_batch_value_count": len(bounded_span_rhos),
        "rho_unit": unit,
        "rho_definition": (
            "initial_parameter_rms_per_bounded_weight_median"
            if architecture_median
            else "initial_parameter_rms_per_bounded_weight"
            if layerwise
            else "initial_parameter_rms"
            if relative
            else "conductance_bound_span"
        ),
        "rho_unit_relative": unit if relative else None,
        "rho_unit_relative_by_parameter": unit_by_parameter,
        "rho_unit_relative_q50_by_parameter": relative_q50_by_parameter,
        "rho_unit_relative_q90": relative_q90_unit,
        "rho_unit_relative_q90_by_parameter": relative_q90_by_parameter,
        "rho_unit_relative_limiting_parameter": (
            max(unit_by_parameter, key=unit_by_parameter.get)
            if unit_by_parameter
            else None
        ),
        "rho_span_unit_diagnostic": span_unit,
        "rho_span_unit_diagnostic_by_parameter": span_q90_by_parameter,
        "rho_unit_is_lr_denominator": not layerwise,
        "normalization_statistic": "median" if architecture_median else "q90",
        "alpha_scope": "architecture" if architecture_median else None,
        "bias_weight_lr_groups": parameter_groups,
        "initial_bounded_parameter_rms": bounded_initial_scales,
        "rho_targets": (
            []
            if weight_median
            else [float(value) for value in study["probe"]["rho_targets"]]
        ),
        "raw_learning_rates": raw_learning_rates,
    }
    atomic_write_csv(output_dir / "step_log.csv", STEP_COLUMNS, step_rows)
    atomic_write_csv(
        output_dir / "parameter_diagnostics.csv", PROBE_DIAGNOSTIC_COLUMNS, diagnostics
    )
    atomic_write_json(output_dir / "minibatches.json", minibatch_payload, canonical=True)
    if two_rho:
        summary = _v6_probe_summary(
            study,
            row,
            summary,
            source_measurement_mode=v6_source_measurement_mode,
        )
    atomic_write_json(output_dir / "summary.json", summary, canonical=True)
    return summary


def _range_tensor_records(result: Any) -> tuple[RangeTensorRecord, ...]:
    return tuple(
        RangeTensorRecord(
            name=item.name,
            bounded=item.bounded_gate,
            gradient_rms=item.gradient_rms,
            proposed_update_rms=item.proposed_update_rms,
            normalized_update=item.normalized_update,
            lower_bound_occupancy=(item.lower_bound_occupancy or 0.0),
            upper_bound_occupancy=(item.upper_bound_occupancy or 0.0),
            projection_efficiency=item.projection_efficiency,
            proposed_bound_crossing_fraction=(
                item.proposed_bound_crossing_fraction or 0.0
            ),
        )
        for item in result.transition.parameters
    )


def _range_candidate_payload(
    candidates: RangeCandidateSet,
    rho_unit: float,
    *,
    relative: bool = False,
    layerwise_unit_by_parameter: Mapping[str, float] | None = None,
    parameter_names: Sequence[str] = (),
    large_lr_thresholds: Sequence[float] = (1.0, 10.0),
) -> dict[str, Any]:
    def one(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        record = asdict(value)
        if layerwise_unit_by_parameter is not None:
            vector = layerwise_target_learning_rates(
                layerwise_unit_by_parameter,
                value.rho_target,
                parameter_names,
            )
            report = layerwise_learning_rate_report(
                vector, thresholds=large_lr_thresholds
            )
            record["learning_rate"] = report["maximum_weight_learning_rate"]
            record["learning_rates_by_parameter"] = vector
            record["learning_rates_by_weight"] = report["weight_learning_rates"]
            record["large_learning_rate_report"] = report
        else:
            record["learning_rate"] = value.rho_target / rho_unit
        if relative:
            record["rho_target_relative"] = value.rho_target
        return record

    return {
        "status": candidates.status,
        "reason": candidates.reason,
        "fast": one(candidates.fast),
        "middle": one(candidates.middle),
        "high": one(candidates.high),
        "stable_targets": [one(value) for value in candidates.stable_targets],
    }


_CALIBRATION_LAYER_SATURATIONS = {
    "conv1_baseline_v1_c1": [0.30000024912308676],
    "conv1_ours_v4_c1": [0.30000024912308676],
    "conv1_legacy_v4_c0p25": [0.29999993771922834],
    "conv2_baseline_v1_c1": [0.29999993771922834, 0.0],
    "conv2_ours_v4_c1": [0.29999993771922834, 0.04516103316326531],
    "conv2_legacy_v4_c0p25": [0.29999993771922834, 0.5926881128427933],
    "conv3_baseline_v1_c1": [0.29999993771922834, 0.0, 0.0],
    "conv3_ours_v4_c1": [0.30000024912308676, 0.0, 0.0],
    "conv3_legacy_v4_c0p25": [
        0.29999993771922834,
        0.0,
        0.1768219616948342,
    ],
}


def candidate_run_spec_v2(
    study: Mapping[str, Any],
    study_dir: str | Path,
    row: Mapping[str, Any],
    role: str,
    *,
    rho_target: float,
    rho_unit: float,
    peak_learning_rate: float,
    bundle: Any,
) -> RunSpec:
    """Build and validate the portable v2 RunSpec recorded by one candidate."""

    root = Path(study_dir).resolve()
    checkpoint = _architecture_checkpoint(root, row)
    results_root = root.parent.parent
    checkpoint_relative = checkpoint.relative_to(results_root).as_posix()
    probe_summary = _stage_entry_dir(root, "probe", row["row_id"]) / "summary.json"
    range_summary = _stage_entry_dir(root, "range", row["row_id"]) / "summary.json"
    # V5 deliberately has no increasing-LR range stage.  This path is used only
    # while constructing the common metadata base before its v4 provenance is
    # substituted below; bind it to the immutable probe in that case.
    if _uses_weight_relative_median(study):
        range_summary = probe_summary
    init_metadata = read_json(root / "initialization" / f"{row['architecture']}.json")
    architecture = study["model"]["architectures"][row["architecture"]]
    affine = dict(study["dataset"]["affine"])
    affine.pop("deterministic_by_original_index")
    all_batches = [
        batch
        for epoch in bundle.train_batch_indices(num_epochs=5)
        for batch in epoch
    ]
    layer_saturations = _CALIBRATION_LAYER_SATURATIONS[row["row_id"]]
    value = {
        "schema_version": "mnist-conv-run/v2",
        "label": f"{row['row_id']}-{role}-seed0-lr-screen",
        "seed": 0,
        "replicate_id": None,
        "protocol_id": study["protocol_id"],
        "category": "diagnostic",
        "run": {
            "dataset": {
                "name": "mnist",
                "input_shape": [2, 28, 28],
                "batch_size": 16,
                "normalization": dict(study["dataset"]["normalization"]),
                "affine": affine,
                "max_batches": None,
                "train_shuffle_seed": 0,
                "validation": {
                    "source": "mnist_train",
                    "size": 5000,
                    "samples_per_class": 500,
                    "split_seed": 0,
                    "batch_size": 128,
                    "stratified": True,
                    "official_test_enabled": False,
                    "indices_sha256": bundle.validation_indices_hash,
                },
            },
            "architecture": {
                "profile": row["architecture"],
                "channels": list(architecture["channels"]),
                "kernel_sizes": list(architecture["kernel_sizes"]),
                "strides": list(architecture["strides"]),
                "paddings": list(architecture["paddings"]),
                "output_dim": 20,
                "pooling": {"mode": "none"},
            },
            "model": {
                "type": "resistive_conv",
                "non_linearity": "hard_sigmoid",
                "voltage_amp": row["voltage_amp"],
                "current_amp": row["current_amp"],
                "input_gain": row["input_gain"],
                "weight_gains": [1.0] * (len(architecture["channels"]) + 1),
                "weight_min": 0.0,
                "weight_max": 100.0,
                "weight_init_mode": "kaiming_uniform",
                "quadratic_diode_param": {},
                "exponential_diode_param": {},
                "hard_sigmoid_param": dict(study["model"]["hard_sigmoid"]),
                "trainable_parameters": {
                    "weights": True,
                    "biases": True,
                    "amplification": False,
                    "hard_sigmoid_v_off": False,
                },
                "amplification_min": 1e-6,
                "amplification_max": None,
            },
            "solver": {
                "inference_iterations": row["inference_iterations"],
                "training_iterations": row["training_iterations"],
                "energy_mode": study["solver"]["energy_mode"],
                "minimizer": dict(study["solver"]["minimizer"]),
            },
            "training": {
                "algorithm": "BP",
                "epochs": 5,
                "optimizer": {"name": "SGD", "momentum": 0.0, "weight_decay": 0.0},
                "peak_learning_rate": peak_learning_rate,
                "schedule": {
                    "name": "linear_warmup_cosine",
                    "interval": "optimizer_step",
                    "warmup_fraction": 0.05,
                    "warmup_steps": 860,
                    "total_steps": 17190,
                    "min_lr_factor": 0.0,
                },
                "beta": 0.1,
                "checkpoint_rule": "min_validation_loss",
                "pruning": {
                    "enabled": False,
                    "after_epoch": None,
                    "min_best_validation_accuracy": None,
                },
                "batch_state_policy": "reset_each_batch",
                "diagnostics": {
                    "enabled": True,
                    "bounded_parameter_classes": ["ConvWeight", "DenseWeight"],
                    "bias_global_gate": False,
                    "range_epsilon": 1e-8,
                    "projection_epsilon": 1e-12,
                    "early_fraction": 0.2,
                    "projection_efficiency_threshold": 0.5,
                    "projection_persistence": 16,
                    "occupancy_delta_threshold": 0.2,
                    "occupancy_persistence": 16,
                },
            },
            "initialization": {
                "checkpoint": {
                    "path": checkpoint_relative,
                    "sha256": sha256_file(checkpoint),
                    "format": "drn.function.parameters/v1",
                    "source_run_id": None,
                    "role": "initialization",
                }
            },
            "calibration": {
                "kind": "hard_sigmoid_saturation",
                "calibration_id": f"conv-hardsigmoid-sat30-20260718-{row['row_id']}",
                "scope": "first_hidden_layer",
                "sample_count": 256,
                "batch_size": 64,
                "model_seed": 0,
                "affine_seed": 1729,
                "settling_iterations": 64,
                "adaptive_equilibrium": False,
                "v_off": 4.0,
                "g_on": 100.0,
                "g_off": 0.0,
                "target_initial_saturation": 0.3,
                "measured_initial_saturation": layer_saturations[0],
                "layer_measurements": [
                    {"layer_index": index, "measured_saturation": value}
                    for index, value in enumerate(layer_saturations, start=1)
                ],
            },
            "lr_provenance": {
                "study_id": LRStudySpec.from_dict(study).study_id,
                "row_id": row["row_id"],
                "candidate_role": role,
                "rho_target": rho_target,
                "rho_unit": rho_unit,
                "probe_sha256": sha256_file(probe_summary),
                "range_sha256": sha256_file(range_summary),
                "split_sha256": bundle.validation_indices_hash,
                "batch_order_sha256": _batch_hash(all_batches),
                "initialization_tensor_sha256": init_metadata[
                    "parameter_tensor_sha256"
                ],
            },
        },
    }
    return RunSpec.from_dict(value)


def candidate_run_spec_v3(
    study: Mapping[str, Any],
    study_dir: str | Path,
    row: Mapping[str, Any],
    role: str,
    *,
    rho_target: float,
    rho_unit_by_weight: Mapping[str, float],
    learning_rates_by_parameter: Mapping[str, float],
    bundle: Any,
) -> RunSpec:
    """Build the constant layer-wise v3 RunSpec without duplicating metadata."""

    report = layerwise_learning_rate_report(
        learning_rates_by_parameter,
        thresholds=study["artifacts"]["large_raw_lr_reporting"]["thresholds"],
    )
    base = candidate_run_spec_v2(
        study,
        study_dir,
        row,
        role,
        rho_target=rho_target,
        rho_unit=max(float(value) for value in rho_unit_by_weight.values()),
        peak_learning_rate=float(report["maximum_weight_learning_rate"]),
        bundle=bundle,
    ).to_dict()
    base["schema_version"] = "mnist-conv-run/v3"
    training = base["run"]["training"]
    training.pop("peak_learning_rate")
    training["learning_rates_by_parameter"] = {
        name: float(value) for name, value in learning_rates_by_parameter.items()
    }
    training["schedule"] = {
        "name": "constant",
        "interval": "optimizer_step",
        "total_steps": CANDIDATE_TOTAL_STEPS,
        "scheduler_enabled": False,
    }
    provenance = base["run"]["lr_provenance"]
    provenance.pop("rho_unit")
    provenance["rho_unit_by_weight"] = {
        name: float(value) for name, value in rho_unit_by_weight.items()
    }
    groups = layerwise_parameter_groups(learning_rates_by_parameter)
    provenance["bias_weight_mapping"] = {
        member: weight
        for weight, members in groups.items()
        for member in members
        if member.startswith("Bias_")
    }
    return RunSpec.from_dict(base)


def candidate_run_spec_v4(
    study: Mapping[str, Any],
    study_dir: str | Path,
    row: Mapping[str, Any],
    role: str,
    *,
    arm: str,
    alpha_role: str,
    alpha: float,
    median_units_by_weight: Mapping[str, float],
    target_multipliers_by_weight: Mapping[str, float],
    learning_rates_by_parameter: Mapping[str, float],
    bundle: Any,
) -> RunSpec:
    """Build the v5 portable run bundle with explicit architecture-alpha data."""

    # Reuse the thoroughly validated common v3 metadata shape.  Effective
    # units make its temporary equal-target identity hold even for the profile
    # arm; the returned v4 provenance replaces them with the true medians and
    # target multipliers.
    effective_units = {
        name: float(median_units_by_weight[name])
        / float(target_multipliers_by_weight[name])
        for name in median_units_by_weight
    }
    base = candidate_run_spec_v3(
        study,
        study_dir,
        row,
        "fast",
        rho_target=float(alpha),
        rho_unit_by_weight=effective_units,
        learning_rates_by_parameter=learning_rates_by_parameter,
        bundle=bundle,
    ).to_dict()
    root = Path(study_dir).resolve()
    init_metadata = read_json(
        root / "initialization" / f"{row['architecture']}.json"
    )
    probe_summary = _stage_entry_dir(
        root, "probe", row["row_id"]
    ) / "summary.json"
    anchor_audit = _stage_entry_dir(
        root, "audit", "anchor-audit"
    ) / "summary.json"
    all_batches = [
        batch
        for epoch in bundle.train_batch_indices(num_epochs=5)
        for batch in epoch
    ]
    groups = layerwise_parameter_groups(learning_rates_by_parameter)
    base["schema_version"] = "mnist-conv-run/v4"
    base["label"] = f"{row['row_id']}--{role}-seed0-v5-diagnostic"
    base["run"]["training"]["diagnostics"]["early_fraction"] = 1.0
    base["run"]["lr_provenance"] = {
        "study_id": LRStudySpec.from_dict(study).study_id,
        "row_id": row["row_id"],
        "candidate_arm": arm,
        "alpha_role": alpha_role,
        "alpha_arch": float(alpha),
        "median_unit_by_weight": {
            name: float(value) for name, value in median_units_by_weight.items()
        },
        "target_multipliers_by_weight": {
            name: float(value)
            for name, value in target_multipliers_by_weight.items()
        },
        "bias_weight_mapping": {
            member: weight
            for weight, members in groups.items()
            for member in members
            if member.startswith("Bias_")
        },
        "probe_sha256": sha256_file(probe_summary),
        "anchor_audit_sha256": sha256_file(anchor_audit),
        "split_sha256": bundle.validation_indices_hash,
        "batch_order_sha256": _batch_hash(all_batches),
        "initialization_tensor_sha256": init_metadata[
            "parameter_tensor_sha256"
        ],
    }
    return RunSpec.from_dict(base)


def candidate_run_spec_v5(
    study: Mapping[str, Any],
    study_dir: str | Path,
    row: Mapping[str, Any],
    role: str,
    *,
    candidate_stage: str,
    rho_conv: float,
    rho_dense: float,
    median_units_by_weight: Mapping[str, float],
    learning_rates_by_parameter: Mapping[str, float],
    bundle: Any,
) -> RunSpec:
    """Build the direct two-target portable run bundle used by v6."""

    # Construct the shared layer-wise metadata through v3 using effective
    # units, then replace the provenance with the exact v5 direct-target
    # contract before final validation.
    temporary_target = max(float(rho_conv), float(rho_dense))
    effective_units = {
        name: temporary_target / float(learning_rates_by_parameter[name])
        for name in median_units_by_weight
    }
    base = candidate_run_spec_v3(
        study,
        study_dir,
        row,
        "fast",
        rho_target=temporary_target,
        rho_unit_by_weight=effective_units,
        learning_rates_by_parameter=learning_rates_by_parameter,
        bundle=bundle,
    ).to_dict()
    root = Path(study_dir).resolve()
    checkpoint = _architecture_checkpoint(root, row)
    init_metadata = read_json(root / "initialization/conv2.json")
    probe_summary = _stage_entry_dir(
        root, "probe", row["row_id"]
    ) / "summary.json"
    all_batches = [
        batch
        for epoch in bundle.train_batch_indices(num_epochs=5)
        for batch in epoch
    ]
    groups = layerwise_parameter_groups(learning_rates_by_parameter)
    base["schema_version"] = "mnist-conv-run/v5"
    base["label"] = f"{row['row_id']}--{role}-seed0-v6-diagnostic"
    base["run"]["training"]["diagnostics"]["early_fraction"] = 1.0
    base["run"]["lr_provenance"] = {
        "study_id": LRStudySpec.from_dict(study).study_id,
        "row_id": row["row_id"],
        "candidate_stage": candidate_stage,
        "rho_conv": float(rho_conv),
        "rho_dense": float(rho_dense),
        "median_unit_by_weight": {
            name: float(value) for name, value in median_units_by_weight.items()
        },
        "bias_weight_mapping": {
            member: weight
            for weight, members in groups.items()
            for member in members
            if member.startswith("Bias_")
        },
        "probe_sha256": sha256_file(probe_summary),
        "split_sha256": bundle.validation_indices_hash,
        "batch_order_sha256": _batch_hash(all_batches),
        "initialization_checkpoint_sha256": sha256_file(checkpoint),
        "initialization_tensor_sha256": init_metadata[
            "parameter_tensor_sha256"
        ],
    }
    return RunSpec.from_dict(base)


def candidate_run_spec_v6(
    study: Mapping[str, Any],
    study_dir: str | Path,
    row: Mapping[str, Any],
    role: str,
    *,
    rho_conv: float,
    rho_dense: float,
    median_units_by_weight: Mapping[str, float],
    learning_rates_by_parameter: Mapping[str, float],
    bundle: Any,
) -> RunSpec:
    """Build a direct two-target amplified-scheme run bundle."""

    temporary_target = max(float(rho_conv), float(rho_dense))
    effective_units = {
        name: temporary_target / float(learning_rates_by_parameter[name])
        for name in median_units_by_weight
    }
    base = candidate_run_spec_v3(
        study,
        study_dir,
        row,
        "fast",
        rho_target=temporary_target,
        rho_unit_by_weight=effective_units,
        learning_rates_by_parameter=learning_rates_by_parameter,
        bundle=bundle,
    ).to_dict()
    root = Path(study_dir).resolve()
    checkpoint = _architecture_checkpoint(root, row)
    init_metadata = read_json(
        root / "initialization" / f"{row['architecture']}.json"
    )
    probe_summary = _stage_entry_dir(
        root, "probe", row["row_id"]
    ) / "summary.json"
    all_batches = [
        batch
        for epoch in bundle.train_batch_indices(num_epochs=5)
        for batch in epoch
    ]
    groups = layerwise_parameter_groups(learning_rates_by_parameter)
    base["schema_version"] = "mnist-conv-run/v6"
    architecture = str(row["architecture"])
    base["protocol_id"] = (
        f"conv-hardsigmoid-lr-{architecture}-amplified-scheme-two-rho-"
        "median-constant-sgd-bs16"
    )
    base["label"] = f"{row['row_id']}--{role}-seed0-scheme-rho-diagnostic"
    base["run"]["training"]["diagnostics"]["early_fraction"] = 1.0
    base["run"]["lr_provenance"] = {
        "study_id": LRStudySpec.from_dict(study).study_id,
        "row_id": row["row_id"],
        "candidate_stage": "scheme_grid",
        "rho_conv": float(rho_conv),
        "rho_dense": float(rho_dense),
        "median_unit_by_weight": {
            name: float(value) for name, value in median_units_by_weight.items()
        },
        "bias_weight_mapping": {
            member: weight
            for weight, members in groups.items()
            for member in members
            if member.startswith("Bias_")
        },
        "probe_sha256": sha256_file(probe_summary),
        "split_sha256": bundle.validation_indices_hash,
        "batch_order_sha256": _batch_hash(all_batches),
        "initialization_checkpoint_sha256": sha256_file(checkpoint),
        "initialization_tensor_sha256": init_metadata[
            "parameter_tensor_sha256"
        ],
    }
    return RunSpec.from_dict(base)


def candidate_run_spec_v7(
    study: Mapping[str, Any],
    study_dir: str | Path,
    row: Mapping[str, Any],
    role: str,
    *,
    candidate_stage: str,
    rho_conv: float,
    rho_dense: float,
    median_units_by_weight: Mapping[str, float],
    learning_rates_by_parameter: Mapping[str, float],
    bundle: Any,
) -> RunSpec:
    """Build one immutable three-epoch ordinary-MNIST Conv3 v7 run."""

    temporary_target = max(float(rho_conv), float(rho_dense))
    effective_units = {
        name: temporary_target / float(learning_rates_by_parameter[name])
        for name in median_units_by_weight
    }
    base = candidate_run_spec_v3(
        study,
        study_dir,
        row,
        "fast",
        rho_target=temporary_target,
        rho_unit_by_weight=effective_units,
        learning_rates_by_parameter=learning_rates_by_parameter,
        bundle=bundle,
    ).to_dict()
    root = Path(study_dir).resolve()
    checkpoint = _architecture_checkpoint(root, row)
    init_metadata = read_json(root / "initialization/conv3.json")
    probe_summary = (
        _stage_entry_dir(root, "probe", row["row_id"]) / "summary.json"
    )
    all_batches = [
        batch
        for epoch in bundle.train_batch_indices(num_epochs=3)
        for batch in epoch
    ]
    groups = layerwise_parameter_groups(learning_rates_by_parameter)
    base["schema_version"] = "mnist-conv-run/v7"
    base["protocol_id"] = study["protocol_id"]
    base["label"] = f"{row['row_id']}--{role}-seed0-v7-ordinary-mnist"
    base["run"]["dataset"]["validation"]["batch_size"] = 64
    training = base["run"]["training"]
    training["epochs"] = 3
    training["schedule"]["total_steps"] = 10314
    training["diagnostics"]["early_fraction"] = 1.0
    base["run"]["lr_provenance"] = {
        "study_id": LRStudySpec.from_dict(study).study_id,
        "row_id": row["row_id"],
        "candidate_stage": candidate_stage,
        "rho_conv": float(rho_conv),
        "rho_dense": float(rho_dense),
        "median_unit_by_weight": {
            name: float(value) for name, value in median_units_by_weight.items()
        },
        "bias_weight_mapping": {
            member: weight
            for weight, members in groups.items()
            for member in members
            if member.startswith("Bias_")
        },
        "probe_sha256": sha256_file(probe_summary),
        "split_sha256": bundle.validation_indices_hash,
        "batch_order_sha256": _batch_hash(all_batches),
        "initialization_checkpoint_sha256": sha256_file(checkpoint),
        "initialization_tensor_sha256": init_metadata[
            "parameter_tensor_sha256"
        ],
        "gain_calibration_dataset": "deterministic_medium_affine_mnist",
        "optimization_dataset": "ordinary_mnist",
        "official_test_read": False,
    }
    return RunSpec.from_dict(base)


def execute_range_entry(
    study: Mapping[str, Any],
    study_dir: str | Path,
    row_id: str,
    *,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    row = _row_by_id(study, row_id)
    root = Path(study_dir).resolve()
    output_dir = _stage_entry_dir(root, "range", row_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_summary_path = _stage_entry_dir(root, "probe", row_id) / "summary.json"
    probe = read_json(probe_summary_path)
    rho_unit = float(probe["rho_unit"])
    relative = _uses_parameter_relative_rho(study)
    layerwise = _uses_layerwise_rho(study)
    layerwise_units = (
        {
            name: float(value)
            for name, value in probe["rho_unit_relative_by_parameter"].items()
        }
        if layerwise
        else None
    )
    parameter_names = tuple(
        name for members in probe.get("bias_weight_lr_groups", {}).values() for name in members
    )
    if layerwise:
        # Preserve model parameter order in runtime validation while deriving
        # all rates from canonical names.
        parameter_names = tuple(
            name
            for group in probe["bias_weight_lr_groups"].values()
            for name in group
        )
    span_rho_unit = float(
        probe.get("rho_span_unit_diagnostic", probe["rho_unit"])
    )
    range_contract = study["range_test"]
    main_steps = int(range_contract["steps"])
    schedule_kwargs: dict[str, Any] = {
        "main_steps": main_steps,
        "start_rho": float(range_contract["start_rho_target"]),
        "end_rho": float(range_contract["end_rho_target"]),
        "extension_steps": int(range_contract["extension"]["steps"]),
        "extension_start_rho": float(
            range_contract["extension"]["start_rho_target"]
        ),
        "extension_end_rho": float(
            range_contract["extension"]["end_rho_target"]
        ),
    }
    if range_contract["schedule"] == "geometric_warm_in_then_v1_main":
        anchor_step = int(range_contract["anchor_step"])
        schedule_kwargs.update(
            {
                "exact_tail_start_rho": float(
                    range_contract["anchor_rho_target"]
                ),
                "exact_tail_steps": main_steps - anchor_step + 1,
            }
        )
    main_rhos = range_normalized_update_schedule(
        include_extension=False, **schedule_kwargs
    )
    if layerwise:
        assert layerwise_units is not None
        main_rates: Sequence[Any] = tuple(
            layerwise_target_learning_rates(
                layerwise_units, rho, parameter_names
            )
            for rho in main_rhos
        )
    else:
        main_rates = range_learning_rate_schedule(
            rho_unit, include_extension=False, **schedule_kwargs
        )
    checkpoint = _architecture_checkpoint(root, row)
    runtime = build_model_runtime(
        study,
        row,
        device=device,
        initialization_checkpoint=checkpoint,
        learning_rate=main_rates[0],
    )
    initial_diagnostics = parameter_state_diagnostics(runtime.parameters)
    bounded_initial_scales = (
        _bounded_initial_rms_scales(initial_diagnostics) if relative else None
    )
    initial_occupancy = {
        name: float(record["combined_bound_occupancy"])
        for name, record in initial_diagnostics.items()
        if record["bounded_gate"]
    }
    bundle = build_loader_bundle(
        study,
        data_root=data_root,
        download=download,
        return_source_indices=True,
    )
    bundle.reset_train_shuffle()
    iterator = iter(bundle.train_loader)
    minibatches: list[tuple[int, ...]] = []
    records: list[RangeStepRecord] = []
    step_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    validation = [{"step": 0, **evaluate_validation(runtime, bundle.validation_loader)}]
    validation[0].pop("source_indices")
    ema: float | None = None
    failure: RangeFailure | None = None
    failure_detail: str | None = None
    zero_epsilon = float(
        study["range_test"]["gates"]["projection_efficiency"]["zero_update_epsilon"]
    )

    def run_schedule(rhos: Sequence[float], rates: Sequence[Any], first_step: int) -> bool:
        nonlocal ema, failure, failure_detail, iterator
        for offset, (scheduled_rho, learning_rate) in enumerate(zip(rhos, rates)):
            step = first_step + offset
            if isinstance(learning_rate, Mapping):
                weight_rates = {
                    name: float(learning_rate[name])
                    for name in layerwise_units or {}
                }
                logged_learning_rate = max(weight_rates.values())
                span_by_parameter = probe[
                    "rho_span_unit_diagnostic_by_parameter"
                ]
                scheduled_span_rho = max(
                    weight_rates[name] * float(span_by_parameter[name])
                    for name in weight_rates
                )
            else:
                logged_learning_rate = float(learning_rate)
                scheduled_span_rho = logged_learning_rate * span_rho_unit
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(bundle.train_loader)
                batch = next(iterator)
            try:
                result = training_step(
                    runtime,
                    batch,
                    learning_rate=learning_rate,
                    restore=False,
                    zero_proposal_epsilon=zero_epsilon,
                )
            except (LRStudyNumericalError, FloatingPointError) as exc:
                failure = RangeFailure(
                    kind="non_finite",
                    onset_step=step,
                    confirmed_step=step,
                    detail=str(exc),
                )
                failure_detail = str(exc)
                step_rows.append(
                    {
                        "step": step,
                        "epoch": 0,
                        "batch_in_epoch": step,
                        "learning_rate": logged_learning_rate,
                        "scheduled_rho": scheduled_rho,
                        "scheduled_rho_relative": (
                            scheduled_rho if relative else None
                        ),
                        "scheduled_rho_span": scheduled_span_rho,
                        "loss": None,
                        "loss_ema": ema,
                        "accuracy": None,
                        "sample_count": None,
                        "status": "failed",
                        "failure_reason": str(exc),
                    }
                )
                return False
            minibatches.append(result.source_indices)
            ema_decay = float(range_contract["loss_ema_decay"])
            ema = (
                result.loss
                if ema is None
                else ema_decay * ema + (1.0 - ema_decay) * result.loss
            )
            range_record = RangeStepRecord(
                step=step,
                learning_rate=logged_learning_rate,
                rho=float(scheduled_rho),
                loss=result.loss,
                tensors=_range_tensor_records(result),
            )
            records.append(range_record)
            step_rows.append(
                {
                    "step": step,
                    "epoch": 0,
                    "batch_in_epoch": step,
                    "learning_rate": logged_learning_rate,
                    "scheduled_rho": scheduled_rho,
                    "scheduled_rho_relative": (
                        scheduled_rho if relative else None
                    ),
                    "scheduled_rho_span": scheduled_span_rho,
                    "loss": result.loss,
                    "loss_ema": ema,
                    "accuracy": result.accuracy,
                    "sample_count": result.sample_count,
                    "status": "complete",
                    "failure_reason": None,
                }
            )
            diagnostics.extend(
                transition_rows(
                    result.transition,
                    step=step,
                    learning_rate=learning_rate,
                    rho_schedule=scheduled_rho,
                    bounded_initial_rms_scales=bounded_initial_scales,
                    rho_schedule_relative=(scheduled_rho if relative else None),
                    rho_schedule_span=scheduled_span_rho,
                )
            )
            if step in study["range_test"]["validation_steps"]:
                metric = {"step": step, **evaluate_validation(runtime, bundle.validation_loader)}
                metric.pop("source_indices")
                validation.append(metric)
            if len(records) >= int(study["range_test"]["reference_steps"]):
                gate = analyze_range_stop_gates(
                    records,
                    initial_occupancy,
                    reference_steps=int(range_contract["reference_steps"]),
                    main_range_steps=main_steps,
                    ema_decay=float(range_contract["loss_ema_decay"]),
                    zero_update_epsilon=zero_epsilon,
                )
                if gate.failure is not None:
                    failure = gate.failure
                    failure_detail = gate.failure.detail
                    return False
        return True

    main_safe = run_schedule(main_rhos, main_rates, 1)
    extension_ran = False
    if main_safe and len(records) == main_steps:
        gate_at_main = analyze_range_stop_gates(
            records,
            initial_occupancy,
            reference_steps=int(range_contract["reference_steps"]),
            main_range_steps=main_steps,
            ema_decay=float(range_contract["loss_ema_decay"]),
            zero_update_epsilon=zero_epsilon,
        )
        if gate_at_main.extension_allowed:
            extension_rhos = range_normalized_update_schedule(
                include_extension=True, **schedule_kwargs
            )[main_steps:]
            if layerwise:
                assert layerwise_units is not None
                extension_rates = tuple(
                    layerwise_target_learning_rates(
                        layerwise_units, rho, parameter_names
                    )
                    for rho in extension_rhos
                )
            else:
                extension_rates = range_learning_rate_schedule(
                    rho_unit, include_extension=True, **schedule_kwargs
                )[main_steps:]
            extension_ran = True
            run_schedule(extension_rhos, extension_rates, main_steps + 1)

    if failure is not None:
        failure_onset = failure.onset_step
        maximum_step = max(0, failure_onset - 1)
    else:
        failure_onset = None
        maximum_step = records[-1].step if records else 0
    maximum_record = records[maximum_step - 1] if maximum_step else None
    candidates = extract_range_candidates(
        records,
        failure_onset_step=failure_onset,
        targets=study["probe"]["rho_targets"],
        reference_steps=int(range_contract["reference_steps"]),
        summary_steps=int(
            range_contract["candidate_extraction"]["summary_steps_per_target"]
        ),
        ema_decay=float(range_contract["loss_ema_decay"]),
    )
    minibatch_payload = {
        "schema_version": "mnist-conv-lr-minibatches/v1",
        "row_id": row_id,
        "stage": "range",
        "batches": [list(batch) for batch in minibatches],
        "batch_order_sha256": _batch_hash(minibatches),
    }
    summary = {
        "schema_version": (
            "mnist-conv-lr-range-result/v3"
            if layerwise
            else "mnist-conv-lr-range-result/v2"
            if relative
            else "mnist-conv-lr-range-result/v1"
        ),
        "row": row,
        "status": "resolved" if candidates.status == "resolved" else "unresolved",
        "failure": None if failure is None else asdict(failure),
        "failure_detail": failure_detail,
        "extension_ran": extension_ran,
        "completed_steps": len(records),
        "maximum_admissible_step": maximum_step,
        "maximum_admissible_learning_rate": (
            None if maximum_record is None else maximum_record.learning_rate
        ),
        "maximum_admissible_learning_rates_by_parameter": (
            None
            if not layerwise or maximum_record is None
            else layerwise_target_learning_rates(
                layerwise_units or {}, maximum_record.rho, parameter_names
            )
        ),
        "maximum_admissible_rho": None if maximum_record is None else maximum_record.rho,
        "rho_unit": rho_unit,
        "rho_definition": (
            "initial_parameter_rms_per_bounded_weight"
            if layerwise
            else "initial_parameter_rms"
            if relative
            else "conductance_bound_span"
        ),
        "rho_unit_relative": rho_unit if relative else None,
        "rho_unit_relative_by_parameter": (
            layerwise_units if layerwise else probe.get("rho_unit_relative_by_parameter", {})
        ),
        "rho_span_unit_diagnostic": span_rho_unit,
        "maximum_admissible_rho_relative": (
            None
            if not relative or maximum_record is None
            else maximum_record.rho
        ),
        "maximum_admissible_rho_span": (
            None
            if maximum_record is None
            else max(
                layerwise_target_learning_rates(
                    layerwise_units or {}, maximum_record.rho, parameter_names
                )[name]
                * float(
                    probe["rho_span_unit_diagnostic_by_parameter"][name]
                )
                for name in (layerwise_units or {})
            )
            if layerwise
            else maximum_record.learning_rate * span_rho_unit
        ),
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_indices_sha256": bundle.train_indices_hash,
        "validation_indices_sha256": bundle.validation_indices_hash,
        "minibatch_order_sha256": minibatch_payload["batch_order_sha256"],
        "initial_parameter_diagnostics": initial_diagnostics,
        "validation_metrics": validation,
        "candidates": _range_candidate_payload(
            candidates,
            rho_unit,
            relative=relative,
            layerwise_unit_by_parameter=layerwise_units,
            parameter_names=parameter_names,
            large_lr_thresholds=tuple(
                study["artifacts"].get("large_raw_lr_reporting", {}).get(
                    "thresholds", (1.0, 10.0)
                )
            ),
        ),
        "unresolved_reason": candidates.reason,
    }
    if range_contract["schedule"] == "geometric_warm_in_then_v1_main":
        summary["range_schedule"] = {
            "name": range_contract["schedule"],
            "main_steps": main_steps,
            "start_rho": main_rhos[0],
            "anchor_step": int(range_contract["anchor_step"]),
            "anchor_rho": main_rhos[int(range_contract["anchor_step"]) - 1],
            "end_rho": main_rhos[-1],
            "extension_steps": int(range_contract["extension"]["steps"]),
            "extension_start_step": main_steps + 1,
            "extension_end_step": main_steps
            + int(range_contract["extension"]["steps"]),
        }
    elif relative:
        summary["range_schedule"] = {
            "name": range_contract["schedule"],
            "main_steps": main_steps,
            "start_rho": main_rhos[0],
            "end_rho": main_rhos[-1],
            "extension_steps": int(range_contract["extension"]["steps"]),
            "extension_start_step": main_steps + 1,
            "extension_end_step": main_steps
            + int(range_contract["extension"]["steps"]),
        }
    atomic_write_csv(output_dir / "step_log.csv", STEP_COLUMNS, step_rows)
    atomic_write_csv(
        output_dir / "parameter_diagnostics.csv", PROBE_DIAGNOSTIC_COLUMNS, diagnostics
    )
    atomic_write_json(output_dir / "minibatches.json", minibatch_payload, canonical=True)
    atomic_write_json(output_dir / "summary.json", summary, canonical=True)
    return summary


class _CandidateStepLoop:
    """One production candidate step, shared by canonical and shadow runs."""

    def __init__(
        self,
        *,
        study: Mapping[str, Any],
        runtime: Any,
        peak_learning_rate: float,
        learning_rates_by_parameter: Mapping[str, float] | None,
        bounded_initial_scales: Mapping[str, float] | None,
        initial_diagnostics: Mapping[str, Mapping[str, Any]],
        relative: bool,
        layerwise: bool,
        weight_median: bool,
    ) -> None:
        self.study = study
        self.runtime = runtime
        self.peak_learning_rate = float(peak_learning_rate)
        self.learning_rates_by_parameter = (
            None
            if learning_rates_by_parameter is None
            else {
                str(name): float(value)
                for name, value in learning_rates_by_parameter.items()
            }
        )
        self.bounded_initial_scales = (
            None
            if bounded_initial_scales is None
            else {
                str(name): float(value)
                for name, value in bounded_initial_scales.items()
            }
        )
        self.relative = bool(relative)
        self.layerwise = bool(layerwise)
        self.weight_median = bool(weight_median)
        if self.layerwise and self.learning_rates_by_parameter is None:
            raise RuntimeError(
                "Expected layerwise candidate steps to receive parameter learning "
                "rates. Provided value: None."
            )

        bounded_initial = {
            name: record
            for name, record in initial_diagnostics.items()
            if record["bounded_gate"]
        }
        self.step_rows: list[dict[str, Any]] = []
        self.diagnostics: list[dict[str, Any]] = []
        self.minibatches: list[tuple[int, ...]] = []
        self.projection_values: list[float] = []
        self.projection_values_by_parameter: dict[str, list[float]] = {
            name: [] for name in (self.bounded_initial_scales or {})
        }
        self.final_bound_occupancy_by_parameter: dict[str, float] = {}
        self.maximum_bound_occupancy_by_parameter: dict[str, float] = {
            name: 0.0 for name in (self.bounded_initial_scales or {})
        }
        self.warmup_peak_rhos: list[float] = []
        self.warmup_peak_relative_rhos: dict[str, list[float]] = {
            name: [] for name in (self.bounded_initial_scales or {})
        }
        self.complete_relative_rhos: dict[str, list[float]] = {
            name: [] for name in (self.bounded_initial_scales or {})
        }
        self.complete_span_rhos: dict[str, list[float]] = {
            name: [] for name in (self.bounded_initial_scales or {})
        }
        self.record_complete_span = _uses_v7_two_rho(study)
        self.projection_streaks = {name: 0 for name in bounded_initial}
        self.initial_occupancy = {
            name: float(record["combined_bound_occupancy"])
            for name, record in bounded_initial.items()
        }
        self.gradient_reference_samples: dict[str, list[float]] = {
            name: [] for name in bounded_initial
        }
        self.gradient_reference: dict[str, float] = {}
        self.gradient_streaks = {name: 0 for name in bounded_initial}
        self.occupancy_streaks = {name: 0 for name in bounded_initial}
        self.loss_gate_streak = 0
        self.prior_loss_ema_minimum = math.inf
        self.safety_gate_failure: dict[str, Any] | None = None
        self.inadmissible_reason: str | None = None
        self.numerical_failure: str | None = None
        self.total_step = 0
        self.successful_steps = 0
        self.ema: float | None = None
        self.zero_epsilon = float(
            study["range_test"]["gates"]["projection_efficiency"][
                "zero_update_epsilon"
            ]
        )
        self.total_steps = int(study["candidate_training"]["total_steps"])
        self.early_steps = int(
            float(
                study["candidate_training"]["inadmissibility"][
                    "projection_domination"
                ]["training_fraction"]
            )
            * self.total_steps
        )
        if self.weight_median:
            self.early_steps = self.total_steps

    def run_step(
        self,
        batch: Any,
        *,
        epoch: int,
        batch_in_epoch: int,
        epoch_relative_rhos: dict[str, list[float]],
    ) -> Any | None:
        """Execute and account for one step; ``None`` means a gated failure."""

        self.total_step += 1
        learning_rate: Any = (
            self.learning_rates_by_parameter
            if self.layerwise
            else candidate_learning_rate_at_step(
                self.peak_learning_rate, self.total_step
            )
        )
        logged_learning_rate = (
            self.peak_learning_rate
            if self.layerwise
            else float(learning_rate)
        )
        try:
            result = training_step(
                self.runtime,
                batch,
                learning_rate=learning_rate,
                restore=False,
                zero_proposal_epsilon=self.zero_epsilon,
            )
        except (LRStudyNumericalError, FloatingPointError) as exc:
            self.numerical_failure = str(exc)
            self.inadmissible_reason = "numerical_failure"
            self.step_rows.append(
                {
                    "step": self.total_step,
                    "epoch": epoch,
                    "batch_in_epoch": batch_in_epoch,
                    "learning_rate": logged_learning_rate,
                    "scheduled_rho": None,
                    "loss": None,
                    "loss_ema": self.ema,
                    "accuracy": None,
                    "sample_count": None,
                    "status": "failed",
                    "failure_reason": self.numerical_failure,
                }
            )
            return None

        self.minibatches.append(result.source_indices)
        self.successful_steps += 1
        self.ema = (
            result.loss
            if self.ema is None
            else 0.98 * self.ema + 0.02 * result.loss
        )
        self.step_rows.append(
            {
                "step": self.total_step,
                "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
                "learning_rate": logged_learning_rate,
                "scheduled_rho": None,
                "loss": result.loss,
                "loss_ema": self.ema,
                "accuracy": result.accuracy,
                "sample_count": result.sample_count,
                "status": "complete",
                "failure_reason": None,
            }
        )
        flat = transition_rows(
            result.transition,
            step=self.total_step,
            learning_rate=learning_rate,
            rho_schedule=None,
            bounded_initial_rms_scales=self.bounded_initial_scales,
        )
        self.diagnostics.extend(flat)
        bounded_flat = {
            record["parameter"]: record
            for record in flat
            if record["bounded_gate"]
        }
        if self.weight_median:
            if self.total_step <= PROBE_BATCHES:
                for name, record in bounded_flat.items():
                    self.gradient_reference_samples[name].append(
                        float(record["gradient_rms"])
                    )
                if self.total_step == PROBE_BATCHES:
                    self.gradient_reference = {
                        name: linear_quantile(values, 0.5)
                        for name, values in self.gradient_reference_samples.items()
                    }
                self.prior_loss_ema_minimum = min(
                    self.prior_loss_ema_minimum, float(self.ema)
                )
            else:
                gate_failures: list[dict[str, Any]] = []
                gates = self.study["range_test"]["gates"]
                loss_gate = gates["loss_ema"]
                if float(self.ema) > float(
                    loss_gate["factor_over_prior_minimum"]
                ) * self.prior_loss_ema_minimum:
                    self.loss_gate_streak += 1
                    if self.loss_gate_streak == int(loss_gate["persistence"]):
                        gate_failures.append(
                            {
                                "kind": "loss_ema_explosion",
                                "parameter": None,
                                "onset_step": self.total_step
                                - self.loss_gate_streak
                                + 1,
                                "confirmed_step": self.total_step,
                            }
                        )
                else:
                    self.loss_gate_streak = 0
                self.prior_loss_ema_minimum = min(
                    self.prior_loss_ema_minimum, float(self.ema)
                )
                gradient_gate = gates["bounded_gradient_rms"]
                occupancy_gate = gates["combined_bound_occupancy"]
                for name, record in bounded_flat.items():
                    if float(record["gradient_rms"]) > float(
                        gradient_gate["factor_over_first_32_median"]
                    ) * self.gradient_reference[name]:
                        self.gradient_streaks[name] += 1
                        if self.gradient_streaks[name] == int(
                            gradient_gate["persistence"]
                        ):
                            gate_failures.append(
                                {
                                    "kind": "gradient_rms_explosion",
                                    "parameter": name,
                                    "onset_step": self.total_step
                                    - self.gradient_streaks[name]
                                    + 1,
                                    "confirmed_step": self.total_step,
                                }
                            )
                    else:
                        self.gradient_streaks[name] = 0
                    if float(record["combined_bound_occupancy"]) > (
                        self.initial_occupancy[name]
                        + float(occupancy_gate["increase_over_initial"])
                    ):
                        self.occupancy_streaks[name] += 1
                        if self.occupancy_streaks[name] == int(
                            occupancy_gate["persistence"]
                        ):
                            gate_failures.append(
                                {
                                    "kind": "bound_occupancy_increase",
                                    "parameter": name,
                                    "onset_step": self.total_step
                                    - self.occupancy_streaks[name]
                                    + 1,
                                    "confirmed_step": self.total_step,
                                }
                            )
                    else:
                        self.occupancy_streaks[name] = 0
                if gate_failures:
                    self.safety_gate_failure = min(
                        gate_failures,
                        key=lambda item: (
                            item["onset_step"],
                            item["kind"],
                            item["parameter"] or "",
                        ),
                    )
                    self.inadmissible_reason = (
                        f"{self.safety_gate_failure['kind']}:"
                        f"{self.safety_gate_failure['parameter'] or 'global'}:"
                        f"onset={self.safety_gate_failure['onset_step']}"
                    )

        for record in flat:
            if not record["bounded_gate"]:
                continue
            if self.layerwise:
                relative_value = float(record["parameter_relative_update"])
                self.complete_relative_rhos[record["parameter"]].append(
                    relative_value
                )
                if self.record_complete_span:
                    self.complete_span_rhos[record["parameter"]].append(
                        float(record["span_normalized_update"])
                    )
                epoch_relative_rhos[record["parameter"]].append(relative_value)
            if record["projection_gate_eligible"]:
                self.projection_values.append(record["projection_efficiency"])
                self.projection_values_by_parameter.setdefault(
                    record["parameter"], []
                ).append(float(record["projection_efficiency"]))
                if self.total_step <= self.early_steps:
                    name = record["parameter"]
                    if record["projection_efficiency"] < 0.50:
                        self.projection_streaks[name] += 1
                    else:
                        self.projection_streaks[name] = 0
                    if self.projection_streaks[name] >= 16:
                        self.inadmissible_reason = (
                            f"early_projection_domination:{name}:onset="
                            f"{self.total_step - 15}"
                        )
                        self.safety_gate_failure = {
                            "kind": "projection_efficiency",
                            "parameter": name,
                            "onset_step": self.total_step - 15,
                            "confirmed_step": self.total_step,
                        }
            occupancy = float(record["combined_bound_occupancy"])
            self.final_bound_occupancy_by_parameter[record["parameter"]] = occupancy
            self.maximum_bound_occupancy_by_parameter[record["parameter"]] = max(
                self.maximum_bound_occupancy_by_parameter.get(
                    record["parameter"], 0.0
                ),
                occupancy,
            )
            observation_window = (
                self.total_step
                <= int(
                    self.study["candidate_training"]["observed_rho"].get(
                        "initial_steps", 32
                    )
                )
                if self.layerwise
                else CANDIDATE_WARMUP_STEPS - 31
                <= self.total_step
                <= CANDIDATE_WARMUP_STEPS
            )
            if observation_window:
                self.warmup_peak_rhos.append(record["normalized_update"])
                if self.relative:
                    self.warmup_peak_relative_rhos[record["parameter"]].append(
                        float(record["parameter_relative_update"])
                    )
        return result


def execute_candidate_entry(
    study: Mapping[str, Any],
    study_dir: str | Path,
    row_id: str,
    role: str,
    *,
    candidate_payload: Mapping[str, Any] | None = None,
    output_stage: str = "candidates",
    output_root: str | Path | None = None,
    data_root: str | Path,
    download: bool,
    device: str,
    controlled_runtime_row: Mapping[str, Any] | None = None,
    controlled_run_spec: Mapping[str, Any] | None = None,
    controlled_summary_schema: str | None = None,
    controlled_bias_learning_rate_overrides: Mapping[str, float] | None = None,
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    row = _row_by_id(study, row_id)
    controlled_variant = controlled_runtime_row is not None
    if controlled_variant != (controlled_run_spec is not None):
        raise ValueError(
            "Expected controlled_runtime_row and controlled_run_spec to be "
            "provided together. Provided value: "
            f"runtime_row={controlled_runtime_row!r}, "
            f"run_spec={controlled_run_spec!r}."
        )
    if progress_path is not None and not controlled_variant:
        raise ValueError(
            "Expected progress_path only for a controlled candidate with its "
            "own immutable wrapper. Provided value: "
            f"progress_path={progress_path!r}, controlled_variant={controlled_variant!r}."
        )
    if (
        controlled_bias_learning_rate_overrides is not None
        and not controlled_variant
    ):
        raise ValueError(
            "Expected controlled_bias_learning_rate_overrides only for a "
            "controlled candidate with its own immutable wrapper. Provided "
            f"value: overrides={controlled_bias_learning_rate_overrides!r}, "
            f"controlled_variant={controlled_variant!r}."
        )
    if controlled_bias_learning_rate_overrides is not None:
        if (
            not isinstance(controlled_bias_learning_rate_overrides, Mapping)
            or not controlled_bias_learning_rate_overrides
        ):
            raise ValueError(
                "Expected controlled_bias_learning_rate_overrides to be a "
                "non-empty mapping. Provided value: "
                f"{controlled_bias_learning_rate_overrides!r}."
            )
        payload_rates = (
            candidate_payload.get("learning_rates_by_parameter")
            if isinstance(candidate_payload, Mapping)
            else None
        )
        run_spec_rates = (
            controlled_run_spec.get("learning_rates_by_parameter")
            if isinstance(controlled_run_spec, Mapping)
            else None
        )
        if (
            not isinstance(payload_rates, Mapping)
            or not isinstance(run_spec_rates, Mapping)
            or dict(run_spec_rates) != dict(payload_rates)
        ):
            raise ValueError(
                "Expected a controlled bias-LR override to be bound to the "
                "same complete learning_rates_by_parameter mapping in both "
                "the candidate payload and controlled run spec. Provided "
                f"value: payload_rates={payload_rates!r}, "
                f"run_spec_rates={run_spec_rates!r}."
            )
        if any(
            name not in run_spec_rates or run_spec_rates[name] != value
            for name, value in controlled_bias_learning_rate_overrides.items()
        ):
            raise ValueError(
                "Expected every controlled bias-LR override to equal its "
                "declared controlled run-spec rate. Provided value: "
                f"overrides={controlled_bias_learning_rate_overrides!r}, "
                f"run_spec_rates={run_spec_rates!r}."
            )
    progress_target = (
        None
        if progress_path is None
        else Path(progress_path).expanduser().resolve()
    )
    runtime_row = row
    if controlled_runtime_row is not None:
        runtime_row = dict(controlled_runtime_row)
        if set(runtime_row) != set(row):
            raise ValueError(
                "Expected a controlled runtime row with exactly the frozen row "
                f"keys. Provided value: {runtime_row!r}."
            )
        changed = {
            key
            for key in row
            if runtime_row[key] != row[key]
        }
        input_gain = runtime_row.get("input_gain")
        if (
            changed != {"input_gain"}
            or not isinstance(input_gain, (int, float))
            or isinstance(input_gain, bool)
            or not math.isfinite(float(input_gain))
            or float(input_gain) <= 0.0
        ):
            raise ValueError(
                "Expected a controlled runtime row to differ only by one positive "
                f"finite input_gain. Provided value: changed={sorted(changed)!r}, "
                f"input_gain={input_gain!r}."
            )
        if not isinstance(controlled_summary_schema, str) or not (
            controlled_summary_schema.strip()
        ):
            raise ValueError(
                "Expected controlled_summary_schema to be a non-empty string. "
                f"Provided value: {controlled_summary_schema!r}."
            )
    root = Path(study_dir).resolve()
    entry_id = f"{row_id}--{role}"
    result_root = root if output_root is None else Path(output_root).resolve()
    output_dir = _stage_entry_dir(result_root, output_stage, entry_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    architecture_median = _uses_architecture_relative_median(study)
    canonical_v6_two_rho = _uses_v6_two_rho(study)
    canonical_v7_two_rho = _uses_v7_two_rho(study)
    canonical_two_rho = canonical_v6_two_rho or canonical_v7_two_rho
    scheme_two_rho = (
        isinstance(candidate_payload, Mapping)
        and candidate_payload.get("candidate_coordinate")
        in {"conv1_scheme_direct_two_rho", "conv2_scheme_direct_two_rho"}
    )
    two_rho = canonical_two_rho or scheme_two_rho
    if (
        controlled_bias_learning_rate_overrides is not None
        and not (two_rho or architecture_median)
    ):
        raise ValueError(
            "Expected controlled_bias_learning_rate_overrides only on an "
            "executor branch that constructs and records a layer-wise LR "
            f"report. Provided value: two_rho={two_rho!r}, "
            f"architecture_median={architecture_median!r}."
        )
    if canonical_v6_two_rho:
        require_v6_r3_authorization()
    weight_median = architecture_median or two_rho
    if two_rho:
        if not isinstance(candidate_payload, Mapping):
            raise RuntimeError(
                "Expected a v6 two-rho candidate manifest payload. "
                f"Provided value: {candidate_payload!r}."
            )
        allowed_stages = (
            {"scheme_grid"}
            if scheme_two_rho
            else {"core_grid", "extension_grid"}
            if canonical_v7_two_rho
            else {"baseline_grid", "confirmation"}
        )
        if (
            canonical_v7_two_rho
            and output_root is not None
            and output_stage in {"promotion", "promotion_retry_1"}
            and candidate_payload.get("rescue_id")
        ):
            # A separate, content-addressed rescue may reuse the exact v7
            # runtime contract without adding a cell to the immutable v7
            # manifests or writing under the v7 study directory.
            allowed_stages = allowed_stages | {"legacy_rescue_promotion"}
        if controlled_variant:
            allowed_stages = allowed_stages | {"legacy_gain_sensitivity"}
        if candidate_payload.get("candidate_stage") not in allowed_stages:
            raise RuntimeError(
                f"Expected direct two-rho candidate_stage to be one of "
                f"{sorted(allowed_stages)!r}. Provided value: "
                f"{candidate_payload.get('candidate_stage')!r}."
            )
        candidate = {
            "rho_target": max(
                float(candidate_payload["rho_conv"]),
                float(candidate_payload["rho_dense"]),
            ),
            "learning_rate": float(candidate_payload["peak_learning_rate"]),
            "learning_rates_by_parameter": dict(
                candidate_payload["learning_rates_by_parameter"]
            ),
            "learning_rates_by_weight": dict(
                candidate_payload["learning_rates_by_weight"]
            ),
            "large_learning_rate_report": layerwise_learning_rate_report(
                candidate_payload["learning_rates_by_parameter"],
                thresholds=study["artifacts"]["large_raw_lr_reporting"][
                    "thresholds"
                ],
                bias_learning_rate_overrides=(
                    controlled_bias_learning_rate_overrides
                ),
            ),
        }
        range_summary = {
            "rho_unit_relative_by_parameter": dict(
                candidate_payload["median_units_by_weight"]
            )
        }
    elif architecture_median:
        if not isinstance(candidate_payload, Mapping):
            raise RuntimeError(
                "Expected a v5 candidate manifest payload. "
                f"Provided value: {candidate_payload!r}."
            )
        probe_summary = read_json(
            _stage_entry_dir(root, "probe", row_id) / "summary.json"
        )
        candidate = {
            "rho_target": float(candidate_payload["alpha"]),
            "learning_rate": float(candidate_payload["peak_learning_rate"]),
            "learning_rates_by_parameter": dict(
                candidate_payload["learning_rates_by_parameter"]
            ),
            "learning_rates_by_weight": dict(
                candidate_payload["learning_rates_by_weight"]
            ),
            "large_learning_rate_report": layerwise_learning_rate_report(
                candidate_payload["learning_rates_by_parameter"],
                thresholds=study["artifacts"]["large_raw_lr_reporting"][
                    "thresholds"
                ],
                bias_learning_rate_overrides=(
                    controlled_bias_learning_rate_overrides
                ),
            ),
        }
        range_summary = {
            "rho_unit_relative_by_parameter": dict(
                candidate_payload["median_units_by_weight"]
            )
        }
    else:
        range_summary = read_json(
            _stage_entry_dir(root, "range", row_id) / "summary.json"
        )
        candidate = range_summary["candidates"].get(role)
        if range_summary["candidates"]["status"] != "resolved" or candidate is None:
            raise RuntimeError(
                f"Expected row {row_id!r} to have a resolved {role!r} candidate. "
                f"Provided value: {range_summary['candidates']!r}."
            )
    rho_target = float(candidate["rho_target"])
    peak_lr = float(candidate["learning_rate"])
    layerwise = _uses_layerwise_rho(study)
    learning_rates_by_parameter = (
        {
            name: float(value)
            for name, value in candidate["learning_rates_by_parameter"].items()
        }
        if layerwise
        else None
    )
    candidate_learning_rate: Any = (
        learning_rates_by_parameter
        if layerwise
        else candidate_learning_rate_at_step(peak_lr, 1)
    )
    checkpoint = _architecture_checkpoint(root, row)
    runtime = build_model_runtime(
        study,
        runtime_row,
        device=device,
        initialization_checkpoint=checkpoint,
        learning_rate=candidate_learning_rate,
    )
    relative = _uses_parameter_relative_rho(study)
    initial_tensor_digest = parameter_tensor_digest(runtime.parameters)
    initial_diagnostics = parameter_state_diagnostics(runtime.parameters)
    bounded_initial_scales = (
        _bounded_initial_rms_scales(initial_diagnostics) if relative else None
    )
    bundle = build_loader_bundle(
        study,
        data_root=data_root,
        download=download,
        return_source_indices=True,
    )
    bundle.reset_train_shuffle()
    if controlled_variant:
        assert controlled_run_spec is not None
        run_spec_path = output_dir / "run_spec.gain_sensitivity.json"
        atomic_write_json(
            run_spec_path,
            dict(controlled_run_spec),
            canonical=True,
        )
        run_spec = None
    elif two_rho:
        assert candidate_payload is not None
        if canonical_v7_two_rho:
            run_spec = candidate_run_spec_v7(
                study,
                root,
                row,
                role,
                candidate_stage=str(candidate_payload["candidate_stage"]),
                rho_conv=float(candidate_payload["rho_conv"]),
                rho_dense=float(candidate_payload["rho_dense"]),
                median_units_by_weight=candidate_payload["median_units_by_weight"],
                learning_rates_by_parameter=learning_rates_by_parameter or {},
                bundle=bundle,
            )
            run_spec_path = output_dir / "run_spec.v7.json"
        elif scheme_two_rho:
            run_spec = candidate_run_spec_v6(
                study,
                root,
                row,
                role,
                rho_conv=float(candidate_payload["rho_conv"]),
                rho_dense=float(candidate_payload["rho_dense"]),
                median_units_by_weight=candidate_payload["median_units_by_weight"],
                learning_rates_by_parameter=learning_rates_by_parameter or {},
                bundle=bundle,
            )
            run_spec_path = output_dir / "run_spec.v6.json"
        else:
            run_spec = candidate_run_spec_v5(
                study,
                root,
                row,
                role,
                candidate_stage=str(candidate_payload["candidate_stage"]),
                rho_conv=float(candidate_payload["rho_conv"]),
                rho_dense=float(candidate_payload["rho_dense"]),
                median_units_by_weight=candidate_payload["median_units_by_weight"],
                learning_rates_by_parameter=learning_rates_by_parameter or {},
                bundle=bundle,
            )
            run_spec_path = output_dir / "run_spec.v5.json"
    elif architecture_median:
        assert candidate_payload is not None
        run_spec = candidate_run_spec_v4(
            study,
            root,
            row,
            role,
            arm=str(candidate_payload["arm"]),
            alpha_role=str(candidate_payload["alpha_role"]),
            alpha=float(candidate_payload["alpha"]),
            median_units_by_weight=candidate_payload["median_units_by_weight"],
            target_multipliers_by_weight=candidate_payload[
                "target_multipliers_by_weight"
            ],
            learning_rates_by_parameter=learning_rates_by_parameter or {},
            bundle=bundle,
        )
        run_spec_path = output_dir / "run_spec.v4.json"
    elif layerwise:
        run_spec = candidate_run_spec_v3(
            study,
            root,
            row,
            role,
            rho_target=rho_target,
            rho_unit_by_weight=range_summary[
                "rho_unit_relative_by_parameter"
            ],
            learning_rates_by_parameter=learning_rates_by_parameter or {},
            bundle=bundle,
        )
        run_spec_path = output_dir / "run_spec.v3.json"
    else:
        run_spec = candidate_run_spec_v2(
            study,
            root,
            row,
            role,
            rho_target=rho_target,
            rho_unit=float(range_summary["rho_unit"]),
            peak_learning_rate=peak_lr,
            bundle=bundle,
        )
        run_spec_path = output_dir / "run_spec.v2.json"
    if run_spec is not None:
        atomic_write_json(run_spec_path, run_spec.to_dict(), canonical=True)
    validation: list[dict[str, Any]] = []
    epoch_relative_rho_summaries: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    best_epoch: int | None = None
    best_checkpoint = output_dir / "best_validation.pt"
    final_checkpoint = output_dir / "final.pt"
    candidate_loop = _CandidateStepLoop(
        study=study,
        runtime=runtime,
        peak_learning_rate=peak_lr,
        learning_rates_by_parameter=learning_rates_by_parameter,
        bounded_initial_scales=bounded_initial_scales,
        initial_diagnostics=initial_diagnostics,
        relative=relative,
        layerwise=layerwise,
        weight_median=weight_median,
    )

    for epoch in range(1, int(study["candidate_training"]["epochs"]) + 1):
        epoch_relative_rhos: dict[str, list[float]] = {
            name: [] for name in (bounded_initial_scales or {})
        }
        epoch_seen = 0
        epoch_loss = 0.0
        epoch_correct = 0.0
        batch_count = 0
        for batch_count, batch in enumerate(bundle.train_loader, start=1):
            result = candidate_loop.run_step(
                batch,
                epoch=epoch,
                batch_in_epoch=batch_count,
                epoch_relative_rhos=epoch_relative_rhos,
            )
            if result is None:
                break
            epoch_seen += result.sample_count
            epoch_loss += result.loss * result.sample_count
            epoch_correct += result.accuracy * result.sample_count
            if candidate_loop.inadmissible_reason is not None:
                break
        if candidate_loop.inadmissible_reason is not None:
            break
        if batch_count != int(study["candidate_training"]["steps_per_epoch"]):
            raise RuntimeError(
                f"Expected 3438 batches in epoch {epoch}. Provided value: {batch_count}."
            )
        if epoch_seen != int(study["candidate_training"]["training_examples"]):
            raise RuntimeError(
                f"Expected 55000 training examples in epoch {epoch}. Provided value: {epoch_seen}."
            )
        metric = {
            "epoch": epoch,
            "step": candidate_loop.total_step,
            **evaluate_validation(runtime, bundle.validation_loader),
        }
        metric.pop("source_indices")
        metric["train_loss"] = epoch_loss / epoch_seen
        metric["train_accuracy"] = epoch_correct / epoch_seen
        validation.append(metric)
        if progress_target is not None:
            progress_target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                progress_target,
                {
                    "schema_version": "mnist-conv-lr-controlled-progress/v1",
                    "row_id": row_id,
                    "candidate_role": role,
                    "completed_epochs": epoch,
                    "completed_steps": candidate_loop.total_step,
                    "validation_metrics": validation,
                    "official_test_read": False,
                },
                canonical=True,
            )
        if layerwise:
            epoch_relative_rho_summaries.append(
                {
                    "epoch": epoch,
                    "q90_relative_by_parameter": {
                        name: linear_quantile(values, 0.9)
                        for name, values in epoch_relative_rhos.items()
                    },
                }
            )
        if float(metric["loss"]) < best_validation_loss:
            best_validation_loss = float(metric["loss"])
            best_epoch = epoch
            runtime.save(best_checkpoint)

    step_rows = candidate_loop.step_rows
    diagnostics = candidate_loop.diagnostics
    minibatches = candidate_loop.minibatches
    projection_values = candidate_loop.projection_values
    projection_values_by_parameter = candidate_loop.projection_values_by_parameter
    final_bound_occupancy_by_parameter = (
        candidate_loop.final_bound_occupancy_by_parameter
    )
    maximum_bound_occupancy_by_parameter = (
        candidate_loop.maximum_bound_occupancy_by_parameter
    )
    warmup_peak_rhos = candidate_loop.warmup_peak_rhos
    warmup_peak_relative_rhos = candidate_loop.warmup_peak_relative_rhos
    complete_relative_rhos = candidate_loop.complete_relative_rhos
    complete_span_rhos = candidate_loop.complete_span_rhos
    safety_gate_failure = candidate_loop.safety_gate_failure
    inadmissible_reason = candidate_loop.inadmissible_reason
    numerical_failure = candidate_loop.numerical_failure
    total_step = candidate_loop.total_step
    successful_steps = candidate_loop.successful_steps

    runtime.save(final_checkpoint)
    candidate_total_steps = int(study["candidate_training"]["total_steps"])
    completed = (
        successful_steps == candidate_total_steps
        and inadmissible_reason is None
    )
    if completed and not validation:
        raise RuntimeError(
            f"Expected epoch validation metrics for completed candidate. Provided value: {validation!r}."
        )
    # An incomplete entry may be retried in place.  If this attempt failed
    # before its first validation, replace any checkpoint left by an earlier
    # attempt instead of accidentally hashing stale state.
    if best_epoch is None:
        runtime.save(best_checkpoint)
    minibatch_payload = {
        "schema_version": "mnist-conv-lr-minibatches/v1",
        "row_id": row_id,
        "candidate_role": role,
        "stage": output_stage,
        "batches": [list(batch) for batch in minibatches],
        "batch_order_sha256": _batch_hash(minibatches),
    }
    final_metric = validation[-1] if completed else None
    minimum_final_accuracy = float(
        study["candidate_training"].get("minimum_final_validation_accuracy", 0.0)
    )
    accuracy_passed = (
        completed
        and final_metric is not None
        and float(final_metric["accuracy"]) >= minimum_final_accuracy
    )
    if weight_median and completed and not accuracy_passed:
        inadmissible_reason = (
            f"final_validation_accuracy_below_{minimum_final_accuracy:g}"
        )
    admissible = completed and (accuracy_passed or not weight_median)
    observed_span_rho = (
        None
        if not warmup_peak_rhos
        else linear_quantile(warmup_peak_rhos, 0.9)
    )
    if any(warmup_peak_relative_rhos.values()):
        observed_relative_rho, observed_relative_by_parameter = (
            probe_parameter_relative_rho_unit(warmup_peak_relative_rhos)
        )
    else:
        observed_relative_rho = None
        observed_relative_by_parameter = {}
    complete_relative_by_parameter = (
        {
            name: linear_quantile(values, 0.9)
            for name, values in complete_relative_rhos.items()
            if values
        }
        if layerwise
        else {}
    )
    complete_span_by_parameter = (
        {
            name: linear_quantile(values, 0.9)
            for name, values in complete_span_rhos.items()
            if values
        }
        if layerwise
        else {}
    )
    summary = {
        "schema_version": (
            controlled_summary_schema
            if controlled_variant
            else
            "mnist-conv-lr-candidate-result/v7"
            if canonical_v7_two_rho
            else "mnist-conv-lr-candidate-result/v6"
            if scheme_two_rho
            else "mnist-conv-lr-candidate-result/v5"
            if canonical_two_rho
            else "mnist-conv-lr-candidate-result/v4"
            if architecture_median
            else "mnist-conv-lr-candidate-result/v3"
            if layerwise
            else "mnist-conv-lr-candidate-result/v2"
            if relative
            else "mnist-conv-lr-candidate-result/v1"
        ),
        "row": runtime_row,
        "candidate_role": role,
        "candidate_arm": (
            None if candidate_payload is None else candidate_payload.get("arm")
        ),
        "alpha_role": (
            None if candidate_payload is None else candidate_payload.get("alpha_role")
        ),
        "alpha_arch": (
            None if candidate_payload is None else candidate_payload.get("alpha")
        ),
        "target_multipliers_by_weight": (
            None
            if candidate_payload is None
            else candidate_payload.get("target_multipliers_by_weight")
        ),
        "normalization_statistic": "median" if weight_median else "q90",
        "rho_target": rho_target,
        "rho_target_relative": rho_target if relative else None,
        "rho_definition": (
            "initial_parameter_rms_per_bounded_weight_median"
            if weight_median
            else "initial_parameter_rms_per_bounded_weight"
            if layerwise
            else "initial_parameter_rms"
            if relative
            else "conductance_bound_span"
        ),
        "peak_learning_rate": peak_lr,
        "learning_rates_by_parameter": (
            learning_rates_by_parameter if layerwise else None
        ),
        "learning_rates_by_weight": (
            candidate.get("learning_rates_by_weight") if layerwise else None
        ),
        "large_learning_rate_report": (
            candidate.get("large_learning_rate_report") if layerwise else None
        ),
        "learning_rate_schedule": (
            {
                "type": "constant_layerwise",
                "scheduler_enabled": False,
                "first_step": 1,
                "final_step": candidate_total_steps,
            }
            if layerwise
            else {
                "type": "linear_warmup_cosine_decay",
                "warmup_steps": CANDIDATE_WARMUP_STEPS,
                "final_step": candidate_total_steps,
            }
        ),
        "status": "complete" if admissible else "failed",
        "training_completed": completed,
        "minimum_final_validation_accuracy": (
            minimum_final_accuracy if weight_median else None
        ),
        "accuracy_gate_passed": accuracy_passed if weight_median else None,
        "admissible": admissible,
        "inadmissible_reason": inadmissible_reason,
        "numerical_failure_detail": numerical_failure,
        "safety_gate_failure": safety_gate_failure,
        "attempted_steps": total_step,
        "completed_steps": successful_steps,
        "checkpoint_sha256": sha256_file(checkpoint),
        "initial_parameter_tensor_sha256": initial_tensor_digest,
        "train_indices_sha256": bundle.train_indices_hash,
        "validation_indices_sha256": bundle.validation_indices_hash,
        "minibatch_order_sha256": minibatch_payload["batch_order_sha256"],
        "validation_metrics": validation,
        "best_validation_epoch": best_epoch,
        "final_validation_loss": None if final_metric is None else final_metric["loss"],
        "final_validation_accuracy": None if final_metric is None else final_metric["accuracy"],
        "median_projection_efficiency": (
            None if not projection_values else linear_quantile(projection_values, 0.5)
        ),
        "median_projection_efficiency_by_parameter": {
            name: linear_quantile(values, 0.5)
            for name, values in projection_values_by_parameter.items()
            if values
        },
        "final_bound_occupancy_by_parameter": final_bound_occupancy_by_parameter,
        "maximum_bound_occupancy_by_parameter": maximum_bound_occupancy_by_parameter,
        "observed_peak_rho": (
            observed_relative_rho if relative else observed_span_rho
        ),
        "observed_peak_rho_relative": (
            observed_relative_rho if relative else None
        ),
        "observed_peak_rho_relative_by_parameter": (
            observed_relative_by_parameter if relative else {}
        ),
        "observed_rho_window": (
            {"name": "first_constant_lr_steps", "steps": 32}
            if layerwise
            else {
                "name": "final_warmup_steps",
                "first_step": CANDIDATE_WARMUP_STEPS - 31,
                "last_step": CANDIDATE_WARMUP_STEPS,
            }
        ),
        "observed_rho_relative_q90_by_epoch": (
            epoch_relative_rho_summaries if layerwise else []
        ),
        "observed_rho_relative_q90_complete_run_by_parameter": (
            complete_relative_by_parameter if layerwise else {}
        ),
        "observed_peak_rho_span": observed_span_rho,
        "initial_bounded_parameter_rms": (
            bounded_initial_scales if relative else None
        ),
        "best_checkpoint_sha256": sha256_file(best_checkpoint),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "final_parameter_tensor_sha256": parameter_tensor_digest(runtime.parameters),
        (
            "run_spec_gain_sensitivity_sha256"
            if controlled_variant
            else
            "run_spec_v7_sha256"
            if canonical_v7_two_rho
            else "run_spec_v6_sha256"
            if scheme_two_rho
            else "run_spec_v5_sha256"
            if canonical_two_rho
            else "run_spec_v4_sha256"
            if architecture_median
            else "run_spec_v3_sha256"
            if layerwise
            else "run_spec_v2_sha256"
        ): sha256_file(run_spec_path),
    }
    if two_rho:
        assert candidate_payload is not None
        for retired_key in (
            "candidate_arm",
            "alpha_role",
            "alpha_arch",
            "target_multipliers_by_weight",
            "rho_target",
            "rho_target_relative",
        ):
            summary.pop(retired_key, None)
        summary.update(
            {
                "candidate_stage": candidate_payload["candidate_stage"],
                "rho_conv": float(candidate_payload["rho_conv"]),
                "rho_dense": float(candidate_payload["rho_dense"]),
                "median_units_by_weight": {
                    name: float(value)
                    for name, value in candidate_payload[
                        "median_units_by_weight"
                    ].items()
                },
                "rho_definition": (
                    "median_batch_rms_gradient_over_initial_weight_rms_"
                    "per_bounded_weight"
                    if scheme_two_rho
                    else study["probe"]["rho_definition"]
                ),
            }
        )
    if canonical_v7_two_rho:
        summary[
            "observed_rho_span_q90_complete_run_by_parameter"
        ] = complete_span_by_parameter
    atomic_write_csv(output_dir / "step_log.csv", STEP_COLUMNS, step_rows)
    atomic_write_csv(
        output_dir / "parameter_diagnostics.csv", PROBE_DIAGNOSTIC_COLUMNS, diagnostics
    )
    atomic_write_json(output_dir / "minibatches.json", minibatch_payload, canonical=True)
    atomic_write_json(output_dir / "validation.json", validation, canonical=True)
    atomic_write_json(output_dir / "summary.json", summary, canonical=True)
    return summary


def execute_v7_preflight_entry(
    study: Mapping[str, Any],
    study_dir: str | Path,
    *,
    candidate_payload: Mapping[str, Any],
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    """Measure the frozen worst-cost v7 row before any core candidate runs."""

    if not _uses_v7_two_rho(study):
        raise ValueError(
            "Expected an mnist-conv-lr-study/v7 Conv3 two-rho study. "
            f"Provided value: {study.get('schema_version')!r}."
        )
    contract = study["preflight"]
    row_id = str(contract["row_id"])
    expected = {
        "row_id": row_id,
        "rho_conv": float(contract["rho_conv"]),
        "rho_dense": float(contract["rho_dense"]),
    }
    observed = {
        "row_id": candidate_payload.get("row_id"),
        "rho_conv": float(candidate_payload.get("rho_conv")),
        "rho_dense": float(candidate_payload.get("rho_dense")),
    }
    if observed != expected:
        raise ValueError(
            "Expected the frozen v7 preflight row and rho pair. "
            f"Provided value: {observed!r}."
        )
    if download:
        raise ValueError(
            "Expected the V100 preflight to use already materialized MNIST assets. "
            f"Provided value: download={download!r}."
        )

    import torch

    requested = torch.device(device)
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Expected the required v7 preflight on an available V100 CUDA device. "
            f"Provided value: {device!r}."
        )
    properties = torch.cuda.get_device_properties(requested)
    device_name = str(properties.name)
    is_v100 = "V100" in device_name.upper()
    root = Path(study_dir).resolve()
    output_dir = _stage_entry_dir(root, "preflight", "ours-worst-cost")
    output_dir.mkdir(parents=True, exist_ok=True)
    row = _row_by_id(study, row_id)
    rates = {
        str(name): float(value)
        for name, value in candidate_payload[
            "learning_rates_by_parameter"
        ].items()
    }
    runtime = build_model_runtime(
        study,
        row,
        device=device,
        initialization_checkpoint=_architecture_checkpoint(root, row),
        learning_rate=rates,
    )
    initial_diagnostics = parameter_state_diagnostics(runtime.parameters)
    bounded_scales = _bounded_initial_rms_scales(initial_diagnostics)
    bundle = build_loader_bundle(
        study,
        data_root=data_root,
        download=False,
        return_source_indices=True,
    )
    bundle.reset_train_shuffle()
    loop = _CandidateStepLoop(
        study=study,
        runtime=runtime,
        peak_learning_rate=max(rates.values()),
        learning_rates_by_parameter=rates,
        bounded_initial_scales=bounded_scales,
        initial_diagnostics=initial_diagnostics,
        relative=True,
        layerwise=True,
        weight_median=True,
    )
    iterator = iter(bundle.train_loader)
    epoch_relative_rhos = {name: [] for name in bounded_scales}
    warmup_steps = int(contract["warmup_steps"])
    measured_steps = int(contract["measured_steps"])
    torch.cuda.reset_peak_memory_stats(requested)
    failure: str | None = None
    for batch_index in range(1, warmup_steps + 1):
        result = loop.run_step(
            next(iterator),
            epoch=0,
            batch_in_epoch=batch_index,
            epoch_relative_rhos=epoch_relative_rhos,
        )
        if result is None or loop.inadmissible_reason is not None:
            failure = loop.inadmissible_reason or "warmup_failed"
            break
    torch.cuda.synchronize(requested)
    measured_started = time.perf_counter()
    if failure is None:
        for offset in range(1, measured_steps + 1):
            result = loop.run_step(
                next(iterator),
                epoch=0,
                batch_in_epoch=warmup_steps + offset,
                epoch_relative_rhos=epoch_relative_rhos,
            )
            if result is None or loop.inadmissible_reason is not None:
                failure = loop.inadmissible_reason or "measured_step_failed"
                break
    torch.cuda.synchronize(requested)
    measured_seconds = time.perf_counter() - measured_started
    validation_started = time.perf_counter()
    validation_metric = (
        evaluate_validation(runtime, bundle.validation_loader)
        if failure is None and bool(contract["full_validation_pass"])
        else None
    )
    torch.cuda.synchronize(requested)
    validation_seconds = time.perf_counter() - validation_started
    if validation_metric is not None:
        validation_metric.pop("source_indices")

    peak_allocated = int(torch.cuda.max_memory_allocated(requested))
    peak_reserved = int(torch.cuda.max_memory_reserved(requested))
    total_memory = int(properties.total_memory)
    required_total_memory = int(contract["gpu_memory_mib"]) * 1024 * 1024
    # CUDA commonly reports slightly less than the marketed framebuffer
    # capacity. A five-percent allowance admits a real 16-GB V100 while the
    # measured headroom gate decides whether this workload actually fits.
    device_capacity_passed = (
        total_memory >= math.floor(0.95 * required_total_memory)
    )
    headroom_fraction = (
        0.0
        if total_memory <= 0
        else max(0.0, (total_memory - peak_reserved) / total_memory)
    )
    completed_measured = max(0, loop.successful_steps - warmup_steps)
    projected_training_seconds = (
        math.inf
        if completed_measured != measured_steps or measured_seconds <= 0.0
        else measured_seconds
        * int(study["candidate_training"]["total_steps"])
        / measured_steps
    )
    projected_candidate_seconds = projected_training_seconds + (
        validation_seconds * int(study["candidate_training"]["epochs"])
    )
    projected_candidate_hours = projected_candidate_seconds / 3600.0
    memory_passed = headroom_fraction >= float(
        contract["memory_headroom_fraction"]
    )
    runtime_passed = projected_candidate_hours <= float(
        contract["maximum_projected_candidate_hours"]
    )
    passed = (
        failure is None
        and is_v100
        and device_capacity_passed
        and memory_passed
        and runtime_passed
        and completed_measured == measured_steps
        and validation_metric is not None
    )
    if passed:
        failure_reason = None
    elif failure is not None:
        failure_reason = failure
    elif not is_v100:
        failure_reason = "device_is_not_v100"
    elif not device_capacity_passed:
        failure_reason = "device_is_below_minimum_v100_memory"
    elif not memory_passed:
        failure_reason = "memory_headroom_below_10_percent"
    elif not runtime_passed:
        failure_reason = "projected_runtime_exceeds_worker_limit"
    else:
        failure_reason = "incomplete_measurement"
    minibatch_payload = {
        "schema_version": "mnist-conv-lr-minibatches/v1",
        "row_id": row_id,
        "stage": "preflight",
        "batches": [list(batch) for batch in loop.minibatches],
        "batch_order_sha256": _batch_hash(loop.minibatches),
    }
    summary = {
        "schema_version": "mnist-conv-lr-v7-preflight/v1",
        "status": "passed" if passed else "failed",
        "failure_reason": failure_reason,
        "row_id": row_id,
        "rho_conv": expected["rho_conv"],
        "rho_dense": expected["rho_dense"],
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "completed_steps": loop.successful_steps,
        "completed_measured_steps": completed_measured,
        "device_name": device_name,
        "device_is_v100": is_v100,
        "total_memory_bytes": total_memory,
        "required_total_memory_bytes": required_total_memory,
        "device_capacity_passed": device_capacity_passed,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "memory_headroom_fraction": headroom_fraction,
        "required_memory_headroom_fraction": float(
            contract["memory_headroom_fraction"]
        ),
        "memory_passed": memory_passed,
        "measured_seconds": measured_seconds,
        "validation_seconds": validation_seconds,
        "projected_candidate_hours": projected_candidate_hours,
        "maximum_projected_candidate_hours": float(
            contract["maximum_projected_candidate_hours"]
        ),
        "runtime_passed": runtime_passed,
        "validation_metrics": validation_metric,
        "safety_gate_failure": loop.safety_gate_failure,
        "learning_rates_by_parameter": rates,
        "minibatch_order_sha256": minibatch_payload["batch_order_sha256"],
        "official_test_read": False,
        "canonical_candidate_completion_published": False,
    }
    atomic_write_csv(output_dir / "step_log.csv", STEP_COLUMNS, loop.step_rows)
    atomic_write_csv(
        output_dir / "parameter_diagnostics.csv",
        PROBE_DIAGNOSTIC_COLUMNS,
        loop.diagnostics,
    )
    atomic_write_json(
        output_dir / "minibatches.json", minibatch_payload, canonical=True
    )
    atomic_write_json(
        output_dir / "validation.json",
        [] if validation_metric is None else [validation_metric],
        canonical=True,
    )
    atomic_write_json(output_dir / "summary.json", summary, canonical=True)
    return summary


def execute_v6_shadow_benchmark_entry(
    study: Mapping[str, Any],
    study_dir: str | Path,
    row_id: str,
    role: str,
    *,
    candidate_payload: Mapping[str, Any],
    data_root: str | Path,
    download: bool,
    device: str,
    ready_path: str | Path,
    start_path: str | Path,
    scratch_output_dir: str | Path,
    warmup_steps: int = 32,
    measured_steps: int = 256,
    start_timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    """Run a synchronized, noncanonical benchmark through the production loop."""

    if not _uses_v6_two_rho(study):
        raise ValueError(
            "Expected a mnist-conv-lr-study/v6 two-rho study. "
            f"Provided value: {study.get('schema_version')!r}."
        )
    require_v6_r3_authorization()
    if download:
        raise ValueError(
            "Expected v6 shadow benchmark download to be disabled. "
            f"Provided value: {download!r}."
        )
    if study.get("dataset", {}).get("official_test") != {
        "enabled": False,
        "read_allowed": False,
    }:
        raise ValueError(
            "Expected official MNIST test reads to be disabled and prohibited. "
            f"Provided value: {study.get('dataset', {}).get('official_test')!r}."
        )
    if warmup_steps != PROBE_BATCHES or measured_steps != 256:
        raise ValueError(
            f"Expected shadow steps to be {PROBE_BATCHES} warm-up plus 256 measured. "
            f"Provided value: warmup={warmup_steps}, measured={measured_steps}."
        )
    if not math.isfinite(float(start_timeout_seconds)) or float(
        start_timeout_seconds
    ) <= 0.0:
        raise ValueError(
            "Expected start_timeout_seconds to be positive and finite. "
            f"Provided value: {start_timeout_seconds!r}."
        )
    if candidate_payload.get("row_id") != row_id:
        raise ValueError(
            f"Expected candidate payload row_id to be {row_id!r}. "
            f"Provided value: {candidate_payload.get('row_id')!r}."
        )
    if candidate_payload.get("candidate_role") != role:
        raise ValueError(
            f"Expected candidate payload role to be {role!r}. "
            f"Provided value: {candidate_payload.get('candidate_role')!r}."
        )
    if candidate_payload.get("candidate_stage") != "baseline_grid":
        raise ValueError(
            "Expected the shadow benchmark to use a baseline_grid candidate. "
            f"Provided value: {candidate_payload.get('candidate_stage')!r}."
        )
    row = _row_by_id(study, row_id)
    if row.get("architecture") != "conv2" or row.get("scheme") != "baseline":
        raise ValueError(
            "Expected the shadow benchmark row to be the Conv2 baseline. "
            f"Provided value: {row!r}."
        )

    root = Path(study_dir).resolve()
    scratch = Path(scratch_output_dir).expanduser().absolute()
    if scratch.is_symlink():
        raise ValueError(
            "Expected scratch_output_dir not to be a symlink. "
            f"Provided value: {scratch}."
        )
    scratch = scratch.resolve()
    canonical_stage_root = (root / "stages").resolve()
    try:
        scratch.relative_to(canonical_stage_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "Expected shadow scratch output outside the canonical stages tree. "
            f"Provided value: {scratch}."
        )
    scratch.mkdir(parents=True, exist_ok=True)
    ready = Path(ready_path).expanduser().absolute()
    start = Path(start_path).expanduser().absolute()
    ready.parent.mkdir(parents=True, exist_ok=True)

    rates = {
        str(name): float(value)
        for name, value in candidate_payload[
            "learning_rates_by_parameter"
        ].items()
    }
    checkpoint = _architecture_checkpoint(root, row)
    runtime = build_model_runtime(
        study,
        row,
        device=device,
        initialization_checkpoint=checkpoint,
        learning_rate=rates,
    )
    initial_diagnostics = parameter_state_diagnostics(runtime.parameters)
    bounded_initial_scales = _bounded_initial_rms_scales(initial_diagnostics)
    bundle = build_loader_bundle(
        study,
        data_root=data_root,
        download=False,
        return_source_indices=True,
    )
    bundle.reset_train_shuffle()
    run_spec = candidate_run_spec_v5(
        study,
        root,
        row,
        role,
        candidate_stage="baseline_grid",
        rho_conv=float(candidate_payload["rho_conv"]),
        rho_dense=float(candidate_payload["rho_dense"]),
        median_units_by_weight=candidate_payload["median_units_by_weight"],
        learning_rates_by_parameter=rates,
        bundle=bundle,
    )
    atomic_write_json(
        scratch / "run_spec.v5.json", run_spec.to_dict(), canonical=True
    )
    loop = _CandidateStepLoop(
        study=study,
        runtime=runtime,
        peak_learning_rate=float(candidate_payload["peak_learning_rate"]),
        learning_rates_by_parameter=rates,
        bounded_initial_scales=bounded_initial_scales,
        initial_diagnostics=initial_diagnostics,
        relative=True,
        layerwise=True,
        weight_median=True,
    )
    iterator = iter(bundle.train_loader)
    atomic_write_json(
        ready,
        {
            "pid": os.getpid(),
            "status": "ready",
            "production_loop": True,
            "scratch_output_dir": str(scratch),
        },
        canonical=True,
    )
    deadline = time.monotonic() + float(start_timeout_seconds)
    while not start.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Expected the synchronized shadow start barrier before timeout. "
                f"Provided path: {start}."
            )
        time.sleep(0.02)

    measured_started: float | None = None
    measured_finished: float | None = None
    epoch_relative_rhos = {
        name: [] for name in bounded_initial_scales
    }

    def synchronize() -> None:
        import torch

        requested = torch.device(device)
        if requested.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Expected an available CUDA device. Provided value: {device!r}."
                )
            torch.cuda.synchronize(requested)

    failure: BaseException | None = None
    try:
        for batch_in_epoch in range(1, warmup_steps + 1):
            result = loop.run_step(
                next(iterator),
                epoch=1,
                batch_in_epoch=batch_in_epoch,
                epoch_relative_rhos=epoch_relative_rhos,
            )
            if result is None or loop.inadmissible_reason is not None:
                raise RuntimeError(
                    "Expected the shadow warm-up to pass every production safety "
                    f"gate. Provided failure: {loop.inadmissible_reason!r}."
                )
        synchronize()
        measured_started = time.time()
        for offset in range(1, measured_steps + 1):
            result = loop.run_step(
                next(iterator),
                epoch=1,
                batch_in_epoch=warmup_steps + offset,
                epoch_relative_rhos=epoch_relative_rhos,
            )
            if result is None or loop.inadmissible_reason is not None:
                raise RuntimeError(
                    "Expected the measured shadow interval to pass every production "
                    f"safety gate. Provided failure: {loop.inadmissible_reason!r}."
                )
        synchronize()
        measured_finished = time.time()
    except BaseException as exc:
        failure = exc
    finally:
        runtime.save(scratch / "final.pt")
        minibatch_payload = {
            "schema_version": "mnist-conv-lr-minibatches/v1",
            "row_id": row_id,
            "candidate_role": role,
            "stage": "shadow_benchmark",
            "batches": [list(batch) for batch in loop.minibatches],
            "batch_order_sha256": _batch_hash(loop.minibatches),
        }
        report = {
            "schema_version": "mnist-conv-lr-v6-shadow-result/v1",
            "status": "complete" if failure is None else "failed",
            "row_id": row_id,
            "candidate_role": role,
            "rho_conv": float(candidate_payload["rho_conv"]),
            "rho_dense": float(candidate_payload["rho_dense"]),
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "attempted_steps": loop.total_step,
            "completed_steps": loop.successful_steps,
            "measured_started_unix_s": measured_started,
            "measured_finished_unix_s": measured_finished,
            "measured_seconds": (
                None
                if measured_started is None or measured_finished is None
                else measured_finished - measured_started
            ),
            "official_test_read": False,
            "production_loop": True,
            "diagnostic_row_count": len(loop.diagnostics),
            "transition_row_count": len(loop.diagnostics),
            "minibatch_count": len(loop.minibatches),
            "safety_gate_failure": loop.safety_gate_failure,
            "inadmissible_reason": loop.inadmissible_reason,
            "numerical_failure_detail": loop.numerical_failure,
            "failure_detail": None if failure is None else str(failure),
            "canonical_completion_published": False,
        }
        atomic_write_csv(scratch / "step_log.csv", STEP_COLUMNS, loop.step_rows)
        atomic_write_csv(
            scratch / "parameter_diagnostics.csv",
            PROBE_DIAGNOSTIC_COLUMNS,
            loop.diagnostics,
        )
        atomic_write_json(
            scratch / "minibatches.json", minibatch_payload, canonical=True
        )
        atomic_write_json(
            scratch / "shadow_summary.json", report, canonical=True
        )
        del iterator
        del runtime
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    if failure is not None:
        raise failure
    return report


def execute_no_candidate_entry(
    study: Mapping[str, Any],
    study_dir: str | Path,
    entry_id: str = "no-resolved-candidates",
) -> dict[str, Any]:
    """Publish an explicit zero-work candidate result when every row is unresolved."""

    root = Path(study_dir).resolve()
    output_dir = _stage_entry_dir(root, "candidates", entry_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row in study["rows"]:
        path = _stage_entry_dir(root, "range", row["row_id"]) / "summary.json"
        summary = read_json(path)
        candidates = summary["candidates"]
        if candidates["status"] == "resolved":
            raise RuntimeError(
                "Expected the no-candidate entry only when every range row is "
                f"unresolved. Provided resolved row: {row['row_id']!r}."
            )
        rows.append(
            {
                "row_id": row["row_id"],
                "range_summary_sha256": sha256_file(path),
                "status": "unresolved",
                "reason": candidates["reason"],
            }
        )
    result = {
        "schema_version": "mnist-conv-lr-no-candidates/v1",
        "status": "complete",
        "candidate_count": 0,
        "reason": "all_range_rows_unresolved",
        "rows": rows,
    }
    atomic_write_json(output_dir / "summary.json", result, canonical=True)
    return result


def _plot_selection(rows: list[dict[str, Any]], path: Path, *, field: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#1f77b4", "ours": "#ff7f0e", "legacy": "#2ca02c"}
    architectures = tuple(dict.fromkeys(row["architecture"] for row in rows))
    if not architectures:
        raise ValueError("Expected at least one selection row. Provided value: [].")
    fig, axes_grid = plt.subplots(
        1,
        len(architectures),
        figsize=(4.75 * len(architectures), 4.2),
        sharey=False,
        squeeze=False,
    )
    axes = axes_grid[0]
    for axis, architecture in zip(axes, architectures):
        for scheme in ("baseline", "ours", "legacy"):
            points = [
                row
                for row in rows
                if row["architecture"] == architecture
                and row["scheme"] == scheme
                and row.get(field) is not None
            ]
            if points:
                axis.scatter(
                    [point["input_gain"] for point in points],
                    [point[field] for point in points],
                    color=colors[scheme],
                    label=scheme,
                    s=55,
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(architecture.capitalize())
        axis.set_xlabel("Frozen input_gain")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(ylabel)
    legend: dict[str, Any] = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend.setdefault(label, handle)
    if legend:
        fig.legend(
            list(legend.values()),
            list(legend),
            loc="upper center",
            ncol=3,
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_v5_final_metrics(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"strict_equal": "#1f77b4", "historical_profile": "#d62728"}
    markers = {"baseline": "o", "ours": "s", "legacy": "^"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    fig.suptitle(
        "Ordinary-MNIST optimization diagnostic — not paper training",
        y=0.995,
    )
    for row_index, architecture in enumerate(("conv1", "conv2")):
        subset = [item for item in records if item["architecture"] == architecture]
        for column, (field, label) in enumerate(
            (
                ("final_validation_accuracy", "Epoch-5 validation accuracy"),
                ("final_validation_loss", "Epoch-5 validation loss"),
            )
        ):
            axis = axes[row_index][column]
            for arm in ("strict_equal", "historical_profile"):
                for scheme in ("baseline", "ours", "legacy"):
                    series = sorted(
                        (
                            item
                            for item in subset
                            if item["arm"] == arm and item["scheme"] == scheme
                        ),
                        key=lambda item: float(item["alpha"]),
                    )
                    axis.plot(
                        [float(item["alpha"]) for item in series],
                        [item[field] for item in series],
                        color=colors[arm],
                        marker=markers[scheme],
                        linestyle="-" if arm == "strict_equal" else "--",
                        label=f"{arm} / {scheme}",
                    )
            axis.set_xscale("log")
            axis.set_xlabel("Architecture-level alpha")
            axis.set_ylabel(label)
            axis.set_title(architecture.upper())
            axis.grid(True, alpha=0.25)
            if column == 0:
                axis.axhline(0.90, color="black", linewidth=1, linestyle=":")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_v5_parameter_mapping(
    records: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    field: str,
    ylabel: str,
    log_y: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arm_colors = {"strict_equal": "#1f77b4", "historical_profile": "#d62728"}
    scheme_markers = {"baseline": "o", "ours": "s", "legacy": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
    fig.suptitle(
        "Ordinary-MNIST optimization diagnostic — not paper training",
        y=0.995,
    )
    for column, architecture in enumerate(("conv1", "conv2")):
        axis = axes[0][column]
        subset = [item for item in records if item["architecture"] == architecture]
        parameter_names = sorted(
            {name for item in subset for name in (item.get(field) or {})}
        )
        line_styles = ["-", "--", ":"]
        style_by_parameter = {
            name: line_styles[index % len(line_styles)]
            for index, name in enumerate(parameter_names)
        }
        for arm in ("strict_equal", "historical_profile"):
            for scheme in ("baseline", "ours", "legacy"):
                series = sorted(
                    (
                        item
                        for item in subset
                        if item["arm"] == arm and item["scheme"] == scheme
                    ),
                    key=lambda item: float(item["alpha"]),
                )
                for parameter in parameter_names:
                    points = [
                        (float(item["alpha"]), (item.get(field) or {}).get(parameter))
                        for item in series
                    ]
                    points = [(x, y) for x, y in points if y is not None]
                    if not points:
                        continue
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        color=arm_colors[arm],
                        marker=scheme_markers[scheme],
                        linestyle=style_by_parameter[parameter],
                        alpha=0.85,
                        label=f"{arm}/{scheme}/{parameter}",
                    )
        axis.set_xscale("log")
        if log_y:
            axis.set_yscale("log")
        axis.set_xlabel("Architecture-level alpha")
        axis.set_ylabel(ylabel)
        axis.set_title(architecture.upper())
        axis.grid(True, alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _execute_v5_selection(
    study: Mapping[str, Any], root: Path, output_dir: Path
) -> dict[str, Any]:
    policy = study["target_policies"]
    records: list[dict[str, Any]] = []
    architecture_results: dict[str, Any] = {}
    for architecture in ("conv1", "conv2"):
        architecture_candidates: list[dict[str, Any]] = []
        rows = [row for row in study["rows"] if row["architecture"] == architecture]
        for arm in policy["arm_order"]:
            for alpha_role, factor in zip(
                policy["candidate_roles"],
                policy["candidate_factors"],
                strict=True,
            ):
                alpha = float(
                    policy["centers_by_architecture_and_arm"][architecture][arm]
                ) * float(factor)
                selection_rows: list[dict[str, Any]] = []
                for row in rows:
                    role = f"{arm}--{alpha_role}"
                    entry_id = f"{row['row_id']}--{role}"
                    summary = read_json(
                        _stage_entry_dir(root, "candidates", entry_id)
                        / "summary.json"
                    )
                    admissible = bool(summary["admissible"])
                    selection_rows.append(
                        {
                            "scheme": row["scheme"],
                            "admissible": admissible,
                            "final_validation_loss": summary["final_validation_loss"] if admissible else None,
                            "final_validation_accuracy": summary["final_validation_accuracy"] if admissible else None,
                            "median_projection_efficiency": summary["median_projection_efficiency"] if admissible else None,
                            "inadmissible_reason": summary["inadmissible_reason"],
                        }
                    )
                    records.append(
                        {
                            "study_role": "ordinary_mnist_optimization_diagnostic",
                            "final_paper_training_authorized": False,
                            "medium_affine_handoff_replaced": False,
                            "architecture": architecture,
                            "row_id": row["row_id"],
                            "scheme": row["scheme"],
                            "arm": arm,
                            "alpha_role": alpha_role,
                            "alpha": alpha,
                            "admissible": admissible,
                            "inadmissible_reason": summary["inadmissible_reason"],
                            "final_validation_loss": summary["final_validation_loss"],
                            "final_validation_accuracy": summary["final_validation_accuracy"],
                            "median_projection_efficiency": summary["median_projection_efficiency"],
                            "learning_rates_by_weight": summary["learning_rates_by_weight"],
                            "achieved_relative_updates_by_weight": summary[
                                "observed_rho_relative_q90_complete_run_by_parameter"
                            ],
                            "maximum_bound_occupancy_by_parameter": summary["maximum_bound_occupancy_by_parameter"],
                            "median_projection_efficiency_by_parameter": summary["median_projection_efficiency_by_parameter"],
                        }
                    )
                architecture_candidates.append(
                    {"arm": arm, "alpha": alpha, "rows": selection_rows}
                )
        architecture_results[architecture] = select_v5_architecture(
            architecture_candidates,
            minimum_accuracy=float(study["selection"]["minimum_final_accuracy"]),
            plateau_relative_tolerance=float(study["selection"]["plateau_relative_to_minimum"]),
        )

    for record in records:
        selected = architecture_results[record["architecture"]]
        record["selected_arm"] = selected["selected_arm"]
        record["selected_alpha"] = selected["selected_alpha"]
        record["selected_candidate"] = (
            selected["status"] == "selected"
            and record["arm"] == selected["selected_arm"]
            and math.isclose(
                float(record["alpha"]),
                float(selected["selected_alpha"]),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )

    overall_status = (
        "selected"
        if all(value["status"] == "selected" for value in architecture_results.values())
        else "failed"
    )
    summary = {
        "schema_version": "mnist-conv-lr-selection/v4",
        "status": overall_status,
        "study_role": "ordinary_mnist_optimization_diagnostic",
        "normalization_statistic": "median",
        "alpha_scope": "architecture",
        "minimum_final_validation_accuracy": float(study["selection"]["minimum_final_accuracy"]),
        "final_paper_training_authorized": False,
        "medium_affine_handoff_replaced": False,
        "architectures": architecture_results,
        "candidates": records,
    }
    csv_fields = [
        "study_role", "final_paper_training_authorized",
        "medium_affine_handoff_replaced", "architecture", "row_id", "scheme",
        "arm", "alpha_role", "alpha",
        "admissible", "inadmissible_reason", "final_validation_loss",
        "final_validation_accuracy", "median_projection_efficiency",
        "selected_arm", "selected_alpha", "selected_candidate",
        "learning_rates_by_weight", "achieved_relative_updates_by_weight",
        "maximum_bound_occupancy_by_parameter",
        "median_projection_efficiency_by_parameter",
    ]
    csv_records: list[dict[str, Any]] = []
    for record in records:
        value = dict(record)
        for field in (
            "learning_rates_by_weight",
            "achieved_relative_updates_by_weight",
            "maximum_bound_occupancy_by_parameter",
            "median_projection_efficiency_by_parameter",
        ):
            value[field] = json.dumps(value[field], sort_keys=True, separators=(",", ":"))
        csv_records.append(value)
    atomic_write_json(output_dir / "selection.json", summary, canonical=True)
    atomic_write_csv(output_dir / "selection.csv", csv_fields, csv_records)
    _plot_v5_final_metrics(records, output_dir / "epoch5_accuracy_loss.png")
    for filename, field, ylabel, log_y in (
        ("raw_learning_rates.png", "learning_rates_by_weight", "Constant raw learning rate", True),
        ("achieved_relative_updates.png", "achieved_relative_updates_by_weight", "Q90 achieved update / RMS(W0)", True),
        ("occupancy.png", "maximum_bound_occupancy_by_parameter", "Maximum conductance-bound occupancy", False),
        ("projection_efficiency.png", "median_projection_efficiency_by_parameter", "Median projection efficiency", False),
    ):
        _plot_v5_parameter_mapping(
            records,
            output_dir / filename,
            field=field,
            ylabel=ylabel,
            log_y=log_y,
        )
    return summary


def _v6_heatmap_matrix(
    records: Sequence[Mapping[str, Any]],
    conv_values: Sequence[float],
    dense_values: Sequence[float],
    value,
) -> list[list[float]]:
    by_pair = {
        (float(record["rho_conv"]), float(record["rho_dense"])): record
        for record in records
    }
    matrix: list[list[float]] = []
    for rho_conv in conv_values:
        row: list[float] = []
        for rho_dense in dense_values:
            raw = value(by_pair[(float(rho_conv), float(rho_dense))])
            row.append(math.nan if raw is None else float(raw))
        matrix.append(row)
    return matrix


def _v6_format_target(value: float) -> str:
    return f"{float(value):g}"


def _v6_draw_heatmap(
    axis: Any,
    matrix: Sequence[Sequence[float]],
    conv_values: Sequence[float],
    dense_values: Sequence[float],
    *,
    title: str,
    colorbar_label: str,
) -> Any:
    image = axis.imshow(matrix, origin="lower", aspect="auto")
    axis.set_xticks(range(len(dense_values)))
    axis.set_xticklabels([_v6_format_target(value) for value in dense_values])
    axis.set_yticks(range(len(conv_values)))
    axis.set_yticklabels([_v6_format_target(value) for value in conv_values])
    axis.set_xlabel("Dense relative-update target")
    axis.set_ylabel("Convolution relative-update target")
    axis.set_title(title)
    image.colorbar_label = colorbar_label
    return image


def _plot_v6_final_metrics(
    records: Sequence[Mapping[str, Any]],
    path: Path,
    conv_values: Sequence[float],
    dense_values: Sequence[float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    for axis, field, title, label in (
        (
            axes[0][0],
            "final_validation_accuracy",
            "Epoch-5 validation accuracy",
            "Accuracy",
        ),
        (
            axes[0][1],
            "final_validation_loss",
            "Epoch-5 validation loss",
            "Loss",
        ),
    ):
        image = _v6_draw_heatmap(
            axis,
            _v6_heatmap_matrix(
                records, conv_values, dense_values, lambda record: record.get(field)
            ),
            conv_values,
            dense_values,
            title=title,
            colorbar_label=label,
        )
        fig.colorbar(image, ax=axis, label=label)
    fig.suptitle("Ordinary-MNIST Conv2 optimization diagnostic")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_v6_parameter_heatmaps(
    records: Sequence[Mapping[str, Any]],
    path: Path,
    conv_values: Sequence[float],
    dense_values: Sequence[float],
    *,
    field: str,
    title: str,
    colorbar_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parameters = ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    for axis, parameter in zip(axes[0], parameters, strict=True):
        image = _v6_draw_heatmap(
            axis,
            _v6_heatmap_matrix(
                records,
                conv_values,
                dense_values,
                lambda record, parameter=parameter: (
                    record.get(field) or {}
                ).get(parameter),
            ),
            conv_values,
            dense_values,
            title=parameter,
            colorbar_label=colorbar_label,
        )
        fig.colorbar(image, ax=axis, label=colorbar_label)
    fig.suptitle(f"{title} — ordinary-MNIST Conv2 diagnostic")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def execute_v6_baseline_selection(
    study: Mapping[str, Any], study_dir: str | Path
) -> dict[str, Any]:
    """Select a bracketed pair from the complete 4x4 baseline grid."""

    if not _uses_v6_two_rho(study):
        raise ValueError(
            "Expected an mnist-conv-lr-study/v6 two-rho study. "
            f"Provided value: {study.get('schema_version')!r}."
        )
    root = Path(study_dir).resolve()
    output_dir = _stage_entry_dir(root, "select_baseline", "selection")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_stage_manifest(
        root / "stages/baseline_candidates/manifest.json", study_dir=root
    )
    records: list[dict[str, Any]] = []
    selector_inputs: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        payload = entry["payload"]
        summary = read_json(
            _stage_entry_dir(root, "baseline_candidates", entry["entry_id"])
            / "summary.json"
        )
        selector_inputs.append(
            {
                "candidate_id": entry["entry_id"],
                "rho_conv": payload["rho_conv"],
                "rho_dense": payload["rho_dense"],
                "admissible": bool(summary["admissible"]),
                "inadmissible_reason": summary["inadmissible_reason"],
                "final_validation_loss": summary["final_validation_loss"],
                "final_validation_accuracy": summary["final_validation_accuracy"],
                "median_projection_efficiency": summary[
                    "median_projection_efficiency"
                ],
            }
        )
        records.append(
            {
                "study_role": "ordinary_mnist_optimization_diagnostic",
                "row_id": payload["row_id"],
                "scheme": payload["scheme"],
                "entry_id": entry["entry_id"],
                "rho_conv": float(payload["rho_conv"]),
                "rho_dense": float(payload["rho_dense"]),
                "admissible": bool(summary["admissible"]),
                "inadmissible_reason": summary["inadmissible_reason"],
                "final_validation_loss": summary["final_validation_loss"],
                "final_validation_accuracy": summary["final_validation_accuracy"],
                "median_projection_efficiency": summary[
                    "median_projection_efficiency"
                ],
                "learning_rates_by_weight": summary["learning_rates_by_weight"],
                "achieved_relative_updates_by_weight": summary[
                    "observed_rho_relative_q90_complete_run_by_parameter"
                ],
                "maximum_bound_occupancy_by_parameter": summary[
                    "maximum_bound_occupancy_by_parameter"
                ],
                "median_projection_efficiency_by_parameter": summary[
                    "median_projection_efficiency_by_parameter"
                ],
            }
        )
    grid = study["rho_grid"]
    selected = select_v6_conv2_baseline(
        selector_inputs,
        rho_conv_values=grid["rho_conv"],
        rho_dense_values=grid["rho_dense"],
        minimum_accuracy=float(study["selection"]["minimum_final_accuracy"]),
        plateau_relative_tolerance=float(
            study["selection"]["plateau_relative_to_minimum"]
        ),
    )
    chosen = selected["diagnostic_best"]
    chosen_id = None if chosen is None else chosen["candidate_id"]
    plateau_ids = {item["candidate_id"] for item in selected["plateau"]}
    for record in records:
        record["in_loss_plateau"] = record["entry_id"] in plateau_ids
        record["diagnostic_best"] = record["entry_id"] == chosen_id
        record["selected_for_confirmation"] = (
            selected["status"] == "selected" and record["entry_id"] == chosen_id
        )
    summary = {
        "schema_version": "mnist-conv-lr-v6-baseline-selection/v1",
        "status": selected["status"],
        "reason": selected["reason"],
        "study_role": "ordinary_mnist_optimization_diagnostic",
        "minimum_final_validation_accuracy": float(
            study["selection"]["minimum_final_accuracy"]
        ),
        "minimum_final_validation_loss": selected[
            "minimum_final_validation_loss"
        ],
        "selected_rho_conv": selected["selected_rho_conv"],
        "selected_rho_dense": selected["selected_rho_dense"],
        "selected_entry_id": (
            None
            if selected["selected"] is None
            else selected["selected"]["candidate_id"]
        ),
        "diagnostic_best_entry_id": chosen_id,
        "plateau_entry_ids": [item["candidate_id"] for item in selected["plateau"]],
        "candidates": records,
        "final_paper_training_authorized": False,
        "medium_affine_handoff_replaced": False,
    }
    csv_fields = list(records[0])
    csv_records: list[dict[str, Any]] = []
    mapping_fields = (
        "learning_rates_by_weight",
        "achieved_relative_updates_by_weight",
        "maximum_bound_occupancy_by_parameter",
        "median_projection_efficiency_by_parameter",
    )
    for record in records:
        value = dict(record)
        for field in mapping_fields:
            value[field] = json.dumps(
                value[field], sort_keys=True, separators=(",", ":")
            )
        csv_records.append(value)
    atomic_write_json(output_dir / "selection.json", summary, canonical=True)
    atomic_write_csv(output_dir / "selection.csv", csv_fields, csv_records)
    conv_values = [float(value) for value in grid["rho_conv"]]
    dense_values = [float(value) for value in grid["rho_dense"]]
    _plot_v6_final_metrics(
        records,
        output_dir / "epoch5_accuracy_loss_heatmap.png",
        conv_values,
        dense_values,
    )
    for filename, field, title, label in (
        (
            "raw_learning_rates_heatmap.png",
            "learning_rates_by_weight",
            "Raw learning rates",
            "Learning rate",
        ),
        (
            "achieved_relative_updates_heatmap.png",
            "achieved_relative_updates_by_weight",
            "Achieved Q90 relative updates",
            "Q90 update / RMS(initial weight)",
        ),
        (
            "occupancy_heatmap.png",
            "maximum_bound_occupancy_by_parameter",
            "Maximum conductance-bound occupancy",
            "Occupancy",
        ),
        (
            "projection_efficiency_heatmap.png",
            "median_projection_efficiency_by_parameter",
            "Median projection efficiency",
            "Projection efficiency",
        ),
    ):
        _plot_v6_parameter_heatmaps(
            records,
            output_dir / filename,
            conv_values,
            dense_values,
            field=field,
            title=title,
            colorbar_label=label,
        )
    return summary


def execute_v6_finalization(
    study: Mapping[str, Any], study_dir: str | Path
) -> dict[str, Any]:
    """Freeze only a bracketed pair whose three schemes all pass."""

    if not _uses_v6_two_rho(study):
        raise ValueError(
            "Expected an mnist-conv-lr-study/v6 two-rho study. "
            f"Provided value: {study.get('schema_version')!r}."
        )
    root = Path(study_dir).resolve()
    output_dir = _stage_entry_dir(root, "finalize", "finalization")
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = read_json(
        _stage_entry_dir(root, "select_baseline", "selection") / "selection.json"
    )
    records: list[dict[str, Any]] = []
    if selection["status"] == "selected":
        def exact_float(left: Any, right: Any) -> bool:
            try:
                return float(left) == float(right)
            except (TypeError, ValueError):
                return False

        selected_entry = selection["selected_entry_id"]
        selected_rho_conv = selection["selected_rho_conv"]
        selected_rho_dense = selection["selected_rho_dense"]
        if (
            not isinstance(selected_entry, str)
            or not selected_entry
            or selected_rho_conv is None
            or selected_rho_dense is None
        ):
            raise RuntimeError(
                "Expected a selected baseline entry and complete selected rho pair. "
                f"Provided value: {selection!r}."
            )
        selected_candidates = [
            candidate
            for candidate in selection.get("candidates", [])
            if candidate.get("entry_id") == selected_entry
        ]
        if len(selected_candidates) != 1:
            raise RuntimeError(
                "Expected selected_entry_id to identify exactly one baseline selection "
                f"candidate. Provided value: {selected_candidates!r}."
            )
        selected_candidate = selected_candidates[0]
        baseline_row_id = study["rho_grid"]["baseline_row_id"]
        if (
            selected_candidate.get("selected_for_confirmation") is not True
            or selected_candidate.get("row_id") != baseline_row_id
            or selected_candidate.get("scheme") != "baseline"
            or not exact_float(
                selected_candidate.get("rho_conv"), selected_rho_conv
            )
            or not exact_float(
                selected_candidate.get("rho_dense"), selected_rho_dense
            )
        ):
            raise RuntimeError(
                "Expected selected baseline metadata to identify the configured baseline "
                "at the exact selected rho pair. "
                f"Provided value: {selected_candidate!r}."
            )
        baseline = read_json(
            _stage_entry_dir(root, "baseline_candidates", selected_entry)
            / "summary.json"
        )
        confirmation_row_ids = tuple(
            study["rho_grid"]["confirmation_row_ids"]
        )
        if len(confirmation_row_ids) != 2 or len(set(confirmation_row_ids)) != 2:
            raise RuntimeError(
                "Expected exactly two unique configured confirmation row IDs. "
                f"Provided value: {confirmation_row_ids!r}."
            )
        rows_by_id = {row["row_id"]: row for row in study["rows"]}
        expected_rows = (baseline_row_id, *confirmation_row_ids)
        if len(set(expected_rows)) != 3 or any(
            row_id not in rows_by_id for row_id in expected_rows
        ):
            raise RuntimeError(
                "Expected baseline and confirmation configuration to identify three "
                f"unique frozen rows. Provided value: {expected_rows!r}."
            )
        summaries = [baseline]
        summaries.extend(
            read_json(
                _stage_entry_dir(root, "confirmations", f"{row_id}--confirmation")
                / "summary.json"
            )
            for row_id in confirmation_row_ids
        )
        expected_stages = ("baseline_grid", "confirmation", "confirmation")
        for summary_value, expected_row_id, expected_stage in zip(
            summaries, expected_rows, expected_stages, strict=True
        ):
            expected_scheme = rows_by_id[expected_row_id]["scheme"]
            observed_contract = {
                "row_id": summary_value.get("row", {}).get("row_id"),
                "scheme": summary_value.get("row", {}).get("scheme"),
                "candidate_stage": summary_value.get("candidate_stage"),
                "rho_conv": summary_value.get("rho_conv"),
                "rho_dense": summary_value.get("rho_dense"),
            }
            if (
                observed_contract["row_id"] != expected_row_id
                or observed_contract["scheme"] != expected_scheme
                or observed_contract["candidate_stage"] != expected_stage
                or not exact_float(
                    observed_contract["rho_conv"], selected_rho_conv
                )
                or not exact_float(
                    observed_contract["rho_dense"], selected_rho_dense
                )
            ):
                raise RuntimeError(
                    "Expected every finalization input to match its configured row, "
                    "scheme, candidate stage, and exact selected rho pair. "
                    f"Provided value: expected_row={expected_row_id!r}, "
                    f"expected_scheme={expected_scheme!r}, "
                    f"expected_stage={expected_stage!r}, "
                    f"observed={observed_contract!r}."
                )
        observed_rows = [summary["row"]["row_id"] for summary in summaries]
        if len(set(observed_rows)) != 3:
            raise RuntimeError(
                "Expected finalization to consume three unique rows. "
                f"Provided value: {observed_rows!r}."
            )
        minimum_accuracy = float(study["selection"]["minimum_final_accuracy"])
        for summary in summaries:
            passed = bool(summary["admissible"]) and (
                summary["final_validation_accuracy"] is not None
                and float(summary["final_validation_accuracy"]) >= minimum_accuracy
            )
            records.append(
                {
                    "row_id": summary["row"]["row_id"],
                    "scheme": summary["row"]["scheme"],
                    "candidate_stage": summary["candidate_stage"],
                    "rho_conv": summary["rho_conv"],
                    "rho_dense": summary["rho_dense"],
                    "passed": passed,
                    "inadmissible_reason": summary["inadmissible_reason"],
                    "final_validation_loss": summary["final_validation_loss"],
                    "final_validation_accuracy": summary[
                        "final_validation_accuracy"
                    ],
                    "median_projection_efficiency": summary[
                        "median_projection_efficiency"
                    ],
                }
            )
        frozen = all(record["passed"] for record in records) and len(records) == 3
        reason = None if frozen else "one_or_more_confirmation_rows_failed"
    else:
        frozen = False
        reason = selection["reason"]
    summary = {
        "schema_version": "mnist-conv-lr-v6-finalization/v1",
        "status": (
            study["selection"]["selected_status"] if frozen else "protocol_failed"
        ),
        "reason": reason,
        "frozen": frozen,
        "rho_conv": selection["selected_rho_conv"] if frozen else None,
        "rho_dense": selection["selected_rho_dense"] if frozen else None,
        "baseline_selection_status": selection["status"],
        "rows": records,
        "canonical_training_run_count": 16 + (2 if records else 0),
        "study_role": "ordinary_mnist_optimization_diagnostic",
        "official_test_read": False,
        "final_paper_training_authorized": False,
        "medium_affine_handoff_replaced": False,
    }
    fields = list(records[0]) if records else [
        "row_id",
        "scheme",
        "candidate_stage",
        "rho_conv",
        "rho_dense",
        "passed",
        "inadmissible_reason",
        "final_validation_loss",
        "final_validation_accuracy",
        "median_projection_efficiency",
    ]
    atomic_write_json(output_dir / "finalization.json", summary, canonical=True)
    atomic_write_csv(output_dir / "finalization.csv", fields, records)
    return summary


def _v7_candidate_records(
    root: Path,
    *,
    stage: str,
    row_id: str | None = None,
) -> list[dict[str, Any]]:
    manifest = load_stage_manifest(
        root / "stages" / stage / "manifest.json",
        study_dir=root,
    )
    records: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        payload = entry["payload"]
        if payload.get("no_op") is True:
            continue
        if row_id is not None and payload["row_id"] != row_id:
            continue
        summary_path = (
            _stage_entry_dir(root, stage, entry["entry_id"]) / "summary.json"
        )
        summary = read_json(summary_path)
        records.append(
            {
                "stage": stage,
                "entry_id": entry["entry_id"],
                "row_id": payload["row_id"],
                "scheme": payload["scheme"],
                "rho_conv": float(payload["rho_conv"]),
                "rho_dense": float(payload["rho_dense"]),
                "admissible": bool(summary["admissible"]),
                "inadmissible_reason": summary["inadmissible_reason"],
                "final_validation_loss": summary["final_validation_loss"],
                "final_validation_accuracy": summary[
                    "final_validation_accuracy"
                ],
                "median_projection_efficiency": summary[
                    "median_projection_efficiency"
                ],
                "learning_rates_by_weight": summary[
                    "learning_rates_by_weight"
                ],
                "learning_rates_by_parameter": summary[
                    "learning_rates_by_parameter"
                ],
                "achieved_relative_updates_by_weight": summary[
                    "observed_rho_relative_q90_complete_run_by_parameter"
                ],
                "achieved_span_updates_by_weight": summary[
                    "observed_rho_span_q90_complete_run_by_parameter"
                ],
                "maximum_bound_occupancy_by_parameter": summary[
                    "maximum_bound_occupancy_by_parameter"
                ],
                "median_projection_efficiency_by_parameter": summary[
                    "median_projection_efficiency_by_parameter"
                ],
                "summary_path": str(summary_path.relative_to(root)),
                "summary_sha256": sha256_file(summary_path),
            }
        )
    return records


def _v7_selector_input(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": record["entry_id"],
        "rho_conv": record["rho_conv"],
        "rho_dense": record["rho_dense"],
        "admissible": record["admissible"],
        "inadmissible_reason": record["inadmissible_reason"],
        "final_validation_loss": record["final_validation_loss"],
        "final_validation_accuracy": record["final_validation_accuracy"],
        "median_projection_efficiency": record[
            "median_projection_efficiency"
        ],
    }


def _plot_v7_final_metrics(
    records: Sequence[Mapping[str, Any]],
    path: Path,
    conv_values: Sequence[float],
    dense_values: Sequence[float],
    *,
    grid_by_row_id: Mapping[
        str, tuple[Sequence[float], Sequence[float]]
    ]
    | None = None,
    scope_title: str = "core grid",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = tuple(dict.fromkeys(str(record["row_id"]) for record in records))
    fig, axes = plt.subplots(
        len(rows), 2, figsize=(11, 4.1 * len(rows)), squeeze=False
    )
    for row_index, row_id in enumerate(rows):
        subset = [record for record in records if record["row_id"] == row_id]
        scheme = subset[0]["scheme"]
        row_conv_values, row_dense_values = (
            (conv_values, dense_values)
            if grid_by_row_id is None
            else grid_by_row_id[row_id]
        )
        for column, (field, title, label) in enumerate(
            (
                (
                    "final_validation_accuracy",
                    "Epoch-3 validation accuracy",
                    "Accuracy",
                ),
                (
                    "final_validation_loss",
                    "Epoch-3 validation loss",
                    "Loss",
                ),
            )
        ):
            image = _v6_draw_heatmap(
                axes[row_index][column],
                _v6_heatmap_matrix(
                    subset,
                    row_conv_values,
                    row_dense_values,
                    lambda record, field=field: record.get(field),
                ),
                row_conv_values,
                row_dense_values,
                title=f"{scheme}: {title}",
                colorbar_label=label,
            )
            fig.colorbar(image, ax=axes[row_index][column], label=label)
    fig.suptitle(f"Ordinary-MNIST Conv3 v7 {scope_title}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_v7_parameter_heatmaps(
    records: Sequence[Mapping[str, Any]],
    path: Path,
    conv_values: Sequence[float],
    dense_values: Sequence[float],
    *,
    field: str,
    title: str,
    colorbar_label: str,
    grid_by_row_id: Mapping[
        str, tuple[Sequence[float], Sequence[float]]
    ]
    | None = None,
    scope_title: str = "core grid",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = tuple(dict.fromkeys(str(record["row_id"]) for record in records))
    parameters = (
        "ConvWeight_0",
        "ConvWeight_1",
        "ConvWeight_2",
        "DenseWeight_0",
    )
    fig, axes = plt.subplots(
        len(rows),
        len(parameters),
        figsize=(19, 4.0 * len(rows)),
        squeeze=False,
    )
    for row_index, row_id in enumerate(rows):
        subset = [record for record in records if record["row_id"] == row_id]
        scheme = subset[0]["scheme"]
        row_conv_values, row_dense_values = (
            (conv_values, dense_values)
            if grid_by_row_id is None
            else grid_by_row_id[row_id]
        )
        for column, parameter in enumerate(parameters):
            image = _v6_draw_heatmap(
                axes[row_index][column],
                _v6_heatmap_matrix(
                    subset,
                    row_conv_values,
                    row_dense_values,
                    lambda record, parameter=parameter: (
                        record.get(field) or {}
                    ).get(parameter),
                ),
                row_conv_values,
                row_dense_values,
                title=f"{scheme}: {parameter}",
                colorbar_label=colorbar_label,
            )
            fig.colorbar(
                image, ax=axes[row_index][column], label=colorbar_label
            )
    fig.suptitle(f"{title} — ordinary-MNIST Conv3 v7 {scope_title}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def execute_v7_core_selection(
    study: Mapping[str, Any], study_dir: str | Path
) -> dict[str, Any]:
    """Select each scheme independently or request its sole expansion."""

    if not _uses_v7_two_rho(study):
        raise ValueError(
            "Expected an mnist-conv-lr-study/v7 Conv3 two-rho study. "
            f"Provided value: {study.get('schema_version')!r}."
        )
    root = Path(study_dir).resolve()
    output_dir = _stage_entry_dir(root, "select_core", "selection")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _v7_candidate_records(root, stage="core_candidates")
    core = study["rho_grid"]["core"]
    conv_values = [float(value) for value in core["rho_conv"]]
    dense_values = [float(value) for value in core["rho_dense"]]
    row_results: list[dict[str, Any]] = []
    for row in study["rows"]:
        row_records = [
            record for record in records if record["row_id"] == row["row_id"]
        ]
        selected = select_two_rho_candidates(
            [_v7_selector_input(record) for record in row_records],
            rho_conv_values=conv_values,
            rho_dense_values=dense_values,
            minimum_accuracy=float(study["selection"]["minimum_final_accuracy"]),
            plateau_relative_tolerance=float(
                study["selection"]["plateau_relative_to_minimum"]
            ),
        )
        axes = (
            two_rho_upper_boundary_axes(
                selected["plateau"],
                rho_conv_values=conv_values,
                rho_dense_values=dense_values,
            )
            if selected["status"] == "unbracketed"
            else ()
        )
        status = (
            "selected_core"
            if selected["status"] == "selected"
            else "expansion_required"
            if axes
            else "unresolved"
        )
        row_results.append(
            {
                "row_id": row["row_id"],
                "scheme": row["scheme"],
                "status": status,
                "reason": selected["reason"],
                "expansion_axes": list(axes),
                "selected_entry_id": (
                    None
                    if selected["selected"] is None
                    else selected["selected"]["candidate_id"]
                ),
                "selected_rho_conv": selected["selected_rho_conv"],
                "selected_rho_dense": selected["selected_rho_dense"],
                "diagnostic_best_entry_id": (
                    None
                    if selected["diagnostic_best"] is None
                    else selected["diagnostic_best"]["candidate_id"]
                ),
                "minimum_final_validation_loss": selected[
                    "minimum_final_validation_loss"
                ],
                "plateau_entry_ids": [
                    item["candidate_id"] for item in selected["plateau"]
                ],
            }
        )
    expansion_count = sum(
        bool(row["expansion_axes"]) for row in row_results
    )
    summary = {
        "schema_version": "mnist-conv-lr-v7-core-selection/v1",
        "status": "complete",
        "rows": row_results,
        "schemes_requesting_expansion": expansion_count,
        "one_expansion_wave_only": True,
        "cross_scheme_transfer_used": False,
        "minimum_final_validation_accuracy": float(
            study["selection"]["minimum_final_accuracy"]
        ),
        "candidates": records,
        "official_test_read": False,
        "final_paper_training_authorized": False,
        "medium_affine_conv3_handoff_replaced": False,
    }
    mapping_fields = (
        "learning_rates_by_weight",
        "learning_rates_by_parameter",
        "achieved_relative_updates_by_weight",
        "achieved_span_updates_by_weight",
        "maximum_bound_occupancy_by_parameter",
        "median_projection_efficiency_by_parameter",
    )
    csv_records: list[dict[str, Any]] = []
    for record in records:
        value = dict(record)
        for field in mapping_fields:
            value[field] = json.dumps(
                value[field], sort_keys=True, separators=(",", ":")
            )
        csv_records.append(value)
    atomic_write_json(output_dir / "selection.json", summary, canonical=True)
    atomic_write_csv(
        output_dir / "selection.csv", list(csv_records[0]), csv_records
    )
    _plot_v7_final_metrics(
        records,
        output_dir / "epoch3_accuracy_loss_heatmap.png",
        conv_values,
        dense_values,
    )
    for filename, field, title, label in (
        (
            "raw_learning_rates_heatmap.png",
            "learning_rates_by_weight",
            "Raw learning rates",
            "Learning rate",
        ),
        (
            "achieved_relative_updates_heatmap.png",
            "achieved_relative_updates_by_weight",
            "Achieved Q90 relative updates",
            "Q90 update / RMS(initial weight)",
        ),
        (
            "occupancy_heatmap.png",
            "maximum_bound_occupancy_by_parameter",
            "Maximum conductance-bound occupancy",
            "Occupancy",
        ),
        (
            "projection_efficiency_heatmap.png",
            "median_projection_efficiency_by_parameter",
            "Median projection efficiency",
            "Projection efficiency",
        ),
    ):
        _plot_v7_parameter_heatmaps(
            records,
            output_dir / filename,
            conv_values,
            dense_values,
            field=field,
            title=title,
            colorbar_label=label,
        )
    return summary


def execute_v7_no_extension_entry(
    study: Mapping[str, Any],
    study_dir: str | Path,
    entry_id: str = "no-expansion-required",
) -> dict[str, Any]:
    """Publish the required explicit zero-work extension completion."""

    if not _uses_v7_two_rho(study):
        raise ValueError(
            "Expected an mnist-conv-lr-study/v7 Conv3 two-rho study. "
            f"Provided value: {study.get('schema_version')!r}."
        )
    root = Path(study_dir).resolve()
    selection = read_json(
        _stage_entry_dir(root, "select_core", "selection") / "selection.json"
    )
    if any(row["expansion_axes"] for row in selection["rows"]):
        raise RuntimeError(
            "Expected zero-work extension only when no scheme requests expansion."
        )
    output_dir = _stage_entry_dir(root, "extension_candidates", entry_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "mnist-conv-lr-v7-zero-work/v1",
        "status": "complete",
        "candidate_count": 0,
        "reason": "no_scheme_requested_expansion",
        "official_test_read": False,
    }
    atomic_write_json(output_dir / "summary.json", summary, canonical=True)
    return summary


def execute_v7_finalization(
    study: Mapping[str, Any], study_dir: str | Path
) -> dict[str, Any]:
    """Freeze selected ordinary-MNIST rows and retain unresolved rows."""

    if not _uses_v7_two_rho(study):
        raise ValueError(
            "Expected an mnist-conv-lr-study/v7 Conv3 two-rho study. "
            f"Provided value: {study.get('schema_version')!r}."
        )
    root = Path(study_dir).resolve()
    output_dir = _stage_entry_dir(root, "finalize", "finalization")
    output_dir.mkdir(parents=True, exist_ok=True)
    study_contract_path = root / "study.resolved.json"
    core_selection_path = (
        _stage_entry_dir(root, "select_core", "selection")
        / "selection.json"
    )
    core_selection = read_json(core_selection_path)
    core_records = _v7_candidate_records(root, stage="core_candidates")
    extension_records = _v7_candidate_records(
        root, stage="extension_candidates"
    )
    core_grid = study["rho_grid"]["core"]
    expansion = study["rho_grid"]["expansion"]
    candidate_count = len(core_records) + len(extension_records)
    maximum_candidate_count = int(expansion["maximum_total_training_runs"])
    if candidate_count > maximum_candidate_count:
        raise RuntimeError(
            "Expected the v7 study to stay within its immutable candidate-run cap. "
            f"Provided value: {candidate_count}."
        )
    expected_core_count = int(core_grid["total_candidate_count"])
    if len(core_records) != expected_core_count:
        raise RuntimeError(
            "Expected finalization to consume the complete v7 core grid. "
            f"Provided value: {len(core_records)}."
        )
    core_by_row = {
        row["row_id"]: row for row in core_selection["rows"]
    }
    expected_row_ids = [row["row_id"] for row in study["rows"]]
    if (
        [row.get("row_id") for row in core_selection.get("rows", [])]
        != expected_row_ids
        or len(core_by_row) != len(expected_row_ids)
    ):
        raise RuntimeError(
            "Expected finalization to consume one core selection for every v7 "
            f"row in frozen order. Provided value: {core_selection.get('rows')!r}."
        )
    final_rows: list[dict[str, Any]] = []
    adaptive_grid_by_row_id: dict[
        str, tuple[Sequence[float], Sequence[float]]
    ] = {}
    for row in study["rows"]:
        row_id = row["row_id"]
        core_result = core_by_row[row_id]
        row_records = [
            record
            for record in (*core_records, *extension_records)
            if record["row_id"] == row_id
        ]
        target_pairs = [
            (float(record["rho_conv"]), float(record["rho_dense"]))
            for record in row_records
        ]
        if len(target_pairs) != len(set(target_pairs)):
            raise RuntimeError(
                "Expected v7 extension candidates to reuse every existing core "
                f"cell rather than rerun it. Provided value: row={row_id!r}, "
                f"target_pairs={target_pairs!r}."
            )
        if len(row_records) > int(expansion["maximum_candidates_per_row"]):
            raise RuntimeError(
                "Expected each v7 scheme to stay within its immutable candidate "
                f"cap. Provided value: row={row_id!r}, count={len(row_records)}."
            )
        axes = tuple(core_result["expansion_axes"])
        conv_values = [float(value) for value in core_grid["rho_conv"]]
        dense_values = [float(value) for value in core_grid["rho_dense"]]
        if "rho_conv" in axes:
            conv_values.append(float(expansion["rho_conv"]))
        if "rho_dense" in axes:
            dense_values.append(float(expansion["rho_dense"]))
        adaptive_grid_by_row_id[row_id] = (conv_values, dense_values)
        if axes:
            selected = select_two_rho_candidates(
                [_v7_selector_input(record) for record in row_records],
                rho_conv_values=conv_values,
                rho_dense_values=dense_values,
                minimum_accuracy=float(
                    study["selection"]["minimum_final_accuracy"]
                ),
                plateau_relative_tolerance=float(
                    study["selection"]["plateau_relative_to_minimum"]
                ),
            )
            chosen_id = (
                None
                if selected["selected"] is None
                else selected["selected"]["candidate_id"]
            )
            reason = (
                "post_expansion_plateau_on_boundary"
                if selected["status"] == "unbracketed"
                else selected["reason"]
            )
            remaining_boundary_axes = (
                two_rho_upper_boundary_axes(
                    selected["plateau"],
                    rho_conv_values=conv_values,
                    rho_dense_values=dense_values,
                )
                if selected["status"] == "unbracketed"
                else ()
            )
            decision_fields = {
                "selection_stage": "post_expansion",
                "expansion_axes": list(axes),
                "selection_status": selected["status"],
                "selection_reason": reason,
                "plateau_entry_ids": [
                    candidate["candidate_id"]
                    for candidate in selected["plateau"]
                ],
                "diagnostic_best_entry_id": (
                    None
                    if selected["diagnostic_best"] is None
                    else selected["diagnostic_best"]["candidate_id"]
                ),
                "minimum_final_validation_loss": selected[
                    "minimum_final_validation_loss"
                ],
                "remaining_upper_boundary_axes": list(
                    remaining_boundary_axes
                ),
                "candidate_count_for_scheme": len(row_records),
            }
        else:
            chosen_id = core_result["selected_entry_id"]
            reason = core_result["reason"]
            decision_fields = {
                "selection_stage": "core",
                "expansion_axes": [],
                "selection_status": core_result["status"],
                "selection_reason": reason,
                "plateau_entry_ids": list(
                    core_result.get("plateau_entry_ids", [])
                ),
                "diagnostic_best_entry_id": core_result.get(
                    "diagnostic_best_entry_id"
                ),
                "minimum_final_validation_loss": core_result.get(
                    "minimum_final_validation_loss"
                ),
                "remaining_upper_boundary_axes": [],
                "candidate_count_for_scheme": len(row_records),
            }
        chosen_record = next(
            (
                record
                for record in row_records
                if record["entry_id"] == chosen_id
            ),
            None,
        )
        selection_has_winner = decision_fields["selection_status"] in {
            "selected",
            "selected_core",
        }
        if selection_has_winner != (chosen_record is not None):
            raise RuntimeError(
                "Expected the persisted v7 selection decision and selected "
                "candidate artifact to agree. "
                f"Provided value: row={row_id!r}, "
                f"selection_status={decision_fields['selection_status']!r}, "
                f"selected_entry_id={chosen_id!r}."
            )
        if chosen_record is None:
            final_rows.append(
                {
                    "row_id": row_id,
                    "scheme": row["scheme"],
                    "status": study["selection"]["unresolved_status"],
                    "reason": reason,
                    **decision_fields,
                    "selected_from_stage": None,
                    "rho_conv": None,
                    "rho_dense": None,
                    "four_weight_learning_rates": None,
                    "seven_parameter_learning_rate_vector": None,
                    "final_validation_loss": None,
                    "final_validation_accuracy": None,
                    "validation_metrics": None,
                    "achieved_relative_updates_by_weight": None,
                    "achieved_span_updates_by_weight": None,
                    "maximum_bound_occupancy_by_parameter": None,
                    "median_projection_efficiency": None,
                    "median_projection_efficiency_by_parameter": None,
                    "safety_diagnostics": None,
                    "provenance_hashes": None,
                }
            )
            continue
        candidate_dir = _stage_entry_dir(
            root, chosen_record["stage"], chosen_record["entry_id"]
        )
        candidate_summary_path = candidate_dir / "summary.json"
        candidate_summary = read_json(candidate_summary_path)
        run_spec_path = candidate_dir / "run_spec.v7.json"
        probe_path = (
            _stage_entry_dir(root, "probe", row_id) / "summary.json"
        )
        final_rows.append(
            {
                "row_id": row_id,
                "scheme": row["scheme"],
                "status": study["selection"]["selected_status"],
                "reason": None,
                **decision_fields,
                "selected_from_stage": chosen_record["stage"],
                "rho_conv": chosen_record["rho_conv"],
                "rho_dense": chosen_record["rho_dense"],
                "four_weight_learning_rates": chosen_record[
                    "learning_rates_by_weight"
                ],
                "seven_parameter_learning_rate_vector": chosen_record[
                    "learning_rates_by_parameter"
                ],
                "final_validation_loss": chosen_record[
                    "final_validation_loss"
                ],
                "final_validation_accuracy": chosen_record[
                    "final_validation_accuracy"
                ],
                "validation_metrics": candidate_summary["validation_metrics"],
                "achieved_relative_updates_by_weight": chosen_record[
                    "achieved_relative_updates_by_weight"
                ],
                "achieved_span_updates_by_weight": chosen_record[
                    "achieved_span_updates_by_weight"
                ],
                "maximum_bound_occupancy_by_parameter": chosen_record[
                    "maximum_bound_occupancy_by_parameter"
                ],
                "median_projection_efficiency": chosen_record[
                    "median_projection_efficiency"
                ],
                "median_projection_efficiency_by_parameter": chosen_record[
                    "median_projection_efficiency_by_parameter"
                ],
                "safety_diagnostics": {
                    "inadmissible_reason": candidate_summary[
                        "inadmissible_reason"
                    ],
                    "safety_gate_failure": candidate_summary[
                        "safety_gate_failure"
                    ],
                    "accuracy_gate_passed": candidate_summary[
                        "accuracy_gate_passed"
                    ],
                },
                "provenance_hashes": {
                    "study_contract_sha256": sha256_file(
                        study_contract_path
                    ),
                    "core_selection_sha256": sha256_file(
                        core_selection_path
                    ),
                    "probe_sha256": sha256_file(probe_path),
                    "candidate_summary_sha256": sha256_file(
                        candidate_summary_path
                    ),
                    "run_spec_v7_sha256": sha256_file(run_spec_path),
                    "initialization_checkpoint_sha256": candidate_summary[
                        "checkpoint_sha256"
                    ],
                    "initialization_tensor_sha256": candidate_summary[
                        "initial_parameter_tensor_sha256"
                    ],
                    "split_sha256": candidate_summary[
                        "validation_indices_sha256"
                    ],
                    "training_batch_order_sha256": candidate_summary[
                        "minibatch_order_sha256"
                    ],
                },
            }
        )
    frozen_count = sum(
        row["status"] == study["selection"]["selected_status"]
        for row in final_rows
    )
    overall_status = (
        study["selection"]["selected_status"]
        if frozen_count == len(final_rows)
        else "partial_seed0_ordinary_mnist_layerwise_screen"
        if frozen_count
        else "unresolved"
    )
    summary = {
        "schema_version": "mnist-conv-lr-v7-finalization/v1",
        "status": overall_status,
        "frozen_row_count": frozen_count,
        "unresolved_row_count": len(final_rows) - frozen_count,
        "partial_handoff": 0 < frozen_count < len(final_rows),
        "rows": final_rows,
        "candidate_training_run_count": candidate_count,
        "maximum_candidate_training_runs": maximum_candidate_count,
        "candidate_epochs": 3 * candidate_count,
        "study_contract_sha256": sha256_file(study_contract_path),
        "core_selection_sha256": sha256_file(core_selection_path),
        "ordinary_mnist_only": True,
        "gain_calibration_dataset": "deterministic_medium_affine_mnist",
        "optimization_dataset": "ordinary_mnist",
        "gain_recalibrated": False,
        "official_test_read": False,
        "final_paper_training_authorized": False,
        "medium_affine_conv3_handoff_replaced": False,
        "heatmap_scope": "per_scheme_final_adaptive_grid",
    }
    fields = list(final_rows[0])
    mapping_fields = {
        "four_weight_learning_rates",
        "seven_parameter_learning_rate_vector",
        "validation_metrics",
        "achieved_relative_updates_by_weight",
        "achieved_span_updates_by_weight",
        "maximum_bound_occupancy_by_parameter",
        "median_projection_efficiency_by_parameter",
        "safety_diagnostics",
        "provenance_hashes",
        "expansion_axes",
        "plateau_entry_ids",
        "remaining_upper_boundary_axes",
    }
    csv_rows: list[dict[str, Any]] = []
    for row in final_rows:
        value = dict(row)
        for field in mapping_fields:
            if value[field] is not None:
                value[field] = json.dumps(
                    value[field], sort_keys=True, separators=(",", ":")
                )
        csv_rows.append(value)
    all_records = [*core_records, *extension_records]
    base_conv_values = [float(value) for value in core_grid["rho_conv"]]
    base_dense_values = [float(value) for value in core_grid["rho_dense"]]
    _plot_v7_final_metrics(
        all_records,
        output_dir / "epoch3_accuracy_loss_heatmap.png",
        base_conv_values,
        base_dense_values,
        grid_by_row_id=adaptive_grid_by_row_id,
        scope_title="final adaptive grids",
    )
    for filename, field, title, label in (
        (
            "raw_learning_rates_heatmap.png",
            "learning_rates_by_weight",
            "Raw learning rates",
            "Learning rate",
        ),
        (
            "achieved_relative_updates_heatmap.png",
            "achieved_relative_updates_by_weight",
            "Achieved Q90 relative updates",
            "Q90 update / RMS(initial weight)",
        ),
        (
            "occupancy_heatmap.png",
            "maximum_bound_occupancy_by_parameter",
            "Maximum conductance-bound occupancy",
            "Occupancy",
        ),
        (
            "projection_efficiency_heatmap.png",
            "median_projection_efficiency_by_parameter",
            "Median projection efficiency",
            "Projection efficiency",
        ),
    ):
        _plot_v7_parameter_heatmaps(
            all_records,
            output_dir / filename,
            base_conv_values,
            base_dense_values,
            field=field,
            title=title,
            colorbar_label=label,
            grid_by_row_id=adaptive_grid_by_row_id,
            scope_title="final adaptive grids",
        )
    atomic_write_json(
        output_dir / "finalization.json", summary, canonical=True
    )
    atomic_write_csv(output_dir / "finalization.csv", fields, csv_rows)
    return summary


def execute_selection(
    study: Mapping[str, Any],
    study_dir: str | Path,
) -> dict[str, Any]:
    root = Path(study_dir).resolve()
    output_dir = _stage_entry_dir(root, "select", "selection")
    output_dir.mkdir(parents=True, exist_ok=True)
    if _uses_architecture_relative_median(study):
        return _execute_v5_selection(study, root, output_dir)
    relative = _uses_parameter_relative_rho(study)
    layerwise = _uses_layerwise_rho(study)
    selected_rows: list[dict[str, Any]] = []
    for row in study["rows"]:
        range_summary = read_json(
            _stage_entry_dir(root, "range", row["row_id"]) / "summary.json"
        )
        if range_summary["candidates"]["status"] != "resolved":
            selected_rows.append(
                {
                    "row_id": row["row_id"],
                    "architecture": row["architecture"],
                    "scheme": row["scheme"],
                    "input_gain": row["input_gain"],
                    "inference_iterations": row["inference_iterations"],
                    "training_iterations": row["training_iterations"],
                    "status": "unresolved",
                    "reason": range_summary["candidates"]["reason"],
                    "selected_candidate_role": None,
                    "selected_peak_learning_rate": None,
                    "selected_learning_rates_by_parameter": None,
                    "selected_learning_rates_by_weight": None,
                    "selected_max_weight_learning_rate": None,
                    "large_learning_rate_report": None,
                    "selected_rho_target": None,
                    "selected_rho_target_relative": None,
                    "observed_peak_rho": None,
                    "observed_peak_rho_relative": None,
                    "observed_peak_rho_span": None,
                    "observed_peak_rho_relative_by_parameter": None,
                    "final_validation_loss": None,
                    "final_validation_accuracy": None,
                    "median_projection_efficiency": None,
                    "plateau_roles": [],
                    "minimum_final_validation_loss": None,
                }
            )
            continue
        results: list[CandidateRunResult] = []
        summaries: dict[str, dict[str, Any]] = {}
        for role in study["candidate_training"]["candidate_roles"]:
            entry_id = f"{row['row_id']}--{role}"
            summary = read_json(
                _stage_entry_dir(root, "candidates", entry_id) / "summary.json"
            )
            summaries[role] = summary
            results.append(
                CandidateRunResult(
                    candidate_id=role,
                    peak_learning_rate=float(summary["peak_learning_rate"]),
                    admissible=bool(summary["admissible"]),
                    final_validation_loss=summary["final_validation_loss"],
                    final_validation_accuracy=summary["final_validation_accuracy"],
                    median_projection_efficiency=summary["median_projection_efficiency"],
                    inadmissible_reason=summary["inadmissible_reason"],
                    selection_coordinate=(
                        float(summary["rho_target"]) if layerwise else None
                    ),
                )
            )
        selection = select_final_candidate(
            results,
            plateau_relative_tolerance=float(
                study["selection"]["plateau_relative_to_minimum"]
            ),
        )
        chosen = selection.selected
        chosen_summary = None if chosen is None else summaries[chosen.candidate_id]
        selected_status = (
            "frozen_seed0_layerwise_diagnostic"
            if layerwise and chosen is not None
            else selection.status
        )
        selected_rows.append(
            {
                "row_id": row["row_id"],
                "architecture": row["architecture"],
                "scheme": row["scheme"],
                "input_gain": row["input_gain"],
                "inference_iterations": row["inference_iterations"],
                "training_iterations": row["training_iterations"],
                "status": selected_status,
                "reason": selection.reason,
                "selected_candidate_role": None if chosen is None else chosen.candidate_id,
                "selected_peak_learning_rate": (
                    None if chosen is None else chosen.peak_learning_rate
                ),
                "selected_learning_rates_by_parameter": (
                    None
                    if chosen_summary is None
                    else chosen_summary.get("learning_rates_by_parameter")
                ),
                "selected_learning_rates_by_weight": (
                    None
                    if chosen_summary is None
                    else chosen_summary.get("learning_rates_by_weight")
                ),
                "selected_max_weight_learning_rate": (
                    None if chosen is None else chosen.peak_learning_rate
                ),
                "large_learning_rate_report": (
                    None
                    if chosen_summary is None
                    else chosen_summary.get("large_learning_rate_report")
                ),
                "selected_rho_target": (
                    None if chosen_summary is None else chosen_summary["rho_target"]
                ),
                "selected_rho_target_relative": (
                    None
                    if not relative or chosen_summary is None
                    else chosen_summary["rho_target"]
                ),
                "observed_peak_rho": (
                    None if chosen_summary is None else chosen_summary["observed_peak_rho"]
                ),
                "observed_peak_rho_relative": (
                    None
                    if chosen_summary is None
                    else chosen_summary.get("observed_peak_rho_relative")
                ),
                "observed_peak_rho_span": (
                    None
                    if chosen_summary is None
                    else chosen_summary.get("observed_peak_rho_span")
                ),
                "observed_peak_rho_relative_by_parameter": (
                    None
                    if chosen_summary is None
                    else chosen_summary.get(
                        "observed_peak_rho_relative_by_parameter"
                    )
                ),
                "final_validation_loss": (
                    None if chosen is None else chosen.final_validation_loss
                ),
                "final_validation_accuracy": (
                    None if chosen is None else chosen.final_validation_accuracy
                ),
                "median_projection_efficiency": (
                    None if chosen is None else chosen.median_projection_efficiency
                ),
                "plateau_roles": [item.candidate_id for item in selection.plateau],
                "minimum_final_validation_loss": selection.minimum_final_validation_loss,
            }
        )
    overall_status = (
        "frozen_seed0_layerwise_diagnostic"
        if layerwise
        and all(
            row["status"] == "frozen_seed0_layerwise_diagnostic"
            for row in selected_rows
        )
        else "frozen_seed0_screen"
        if all(row["status"] == "frozen_seed0_screen" for row in selected_rows)
        else "unresolved"
    )
    summary = {
        "schema_version": (
            "mnist-conv-lr-selection/v3"
            if layerwise
            else "mnist-conv-lr-selection/v2"
            if relative
            else "mnist-conv-lr-selection/v1"
        ),
        "status": overall_status,
        "rho_definition": (
            "initial_parameter_rms_per_bounded_weight"
            if layerwise
            else "initial_parameter_rms"
            if relative
            else "conductance_bound_span"
        ),
        "final_paper_training_authorized": False,
        "rows": selected_rows,
    }
    fields = list(selected_rows[0])
    csv_rows = []
    for row in selected_rows:
        value = dict(row)
        value["plateau_roles"] = ",".join(row["plateau_roles"])
        for field in (
            "selected_learning_rates_by_parameter",
            "selected_learning_rates_by_weight",
            "large_learning_rate_report",
            "observed_peak_rho_relative_by_parameter",
        ):
            if value.get(field) is not None:
                value[field] = json.dumps(
                    value[field], sort_keys=True, separators=(",", ":")
                )
        csv_rows.append(value)
    atomic_write_json(output_dir / "selection.json", summary, canonical=True)
    atomic_write_csv(output_dir / "selection.csv", fields, csv_rows)
    _plot_selection(
        selected_rows,
        output_dir / "selected_raw_peak_lr_vs_input_gain.png",
        field=(
            "selected_max_weight_learning_rate"
            if layerwise
            else "selected_peak_learning_rate"
        ),
        ylabel=(
            "Selected maximum raw layer learning rate"
            if layerwise
            else "Selected raw peak learning rate"
        ),
    )
    _plot_selection(
        selected_rows,
        output_dir / "observed_normalized_peak_update_vs_input_gain.png",
        field="observed_peak_rho",
        ylabel=(
            "Observed peak update / initial parameter RMS"
            if relative
            else "Observed normalized peak update"
        ),
    )
    return summary


__all__ = [
    "PROBE_DIAGNOSTIC_COLUMNS",
    "STEP_COLUMNS",
    "execute_candidate_entry",
    "execute_anchor_audit",
    "execute_no_candidate_entry",
    "execute_probe_entry",
    "execute_range_entry",
    "execute_selection",
    "execute_v6_baseline_selection",
    "execute_v6_finalization",
    "execute_v6_reuse_audit",
    "execute_v7_asset_audit",
    "execute_v7_core_selection",
    "execute_v7_finalization",
    "execute_v7_no_extension_entry",
    "execute_v7_preflight_entry",
    "candidate_run_spec_v2",
    "candidate_run_spec_v3",
    "candidate_run_spec_v4",
    "candidate_run_spec_v5",
    "candidate_run_spec_v7",
    "prepare_study_assets",
    "two_rho_learning_rates",
]
