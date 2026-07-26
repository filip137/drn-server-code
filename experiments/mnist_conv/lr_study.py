"""Plan and execute immutable stages of the Conv1/Conv2 LR study."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from .identity import code_provenance, normalize_code_provenance
from .io import atomic_write_json, read_json
from .layout import ResultLayout
from .lr_artifacts import (
    build_stage_manifest,
    entry_is_complete,
    load_stage_manifest,
    publish_entry_completion,
    publish_stage_completion,
    publish_stage_manifest,
    stage_entry,
    validate_stage_completion,
)
from .lr_stages import (
    execute_anchor_audit,
    execute_candidate_entry,
    execute_no_candidate_entry,
    execute_probe_entry,
    execute_range_entry,
    execute_selection,
    execute_v6_baseline_selection,
    execute_v6_finalization,
    execute_v6_reuse_audit,
    execute_v7_asset_audit,
    execute_v7_core_selection,
    execute_v7_finalization,
    execute_v7_no_extension_entry,
    execute_v7_preflight_entry,
    prepare_study_assets,
    require_v6_r3_authorization,
    two_rho_learning_rates,
)
from .lr_protocol import (
    architecture_relative_learning_rates,
    layerwise_learning_rate_report,
)
from .lr_study_spec import LRStudySpec


LEGACY_STAGES = ("audit", "probe", "range", "candidates", "select")
V6_STAGES = (
    "audit",
    "probe",
    "baseline_candidates",
    "select_baseline",
    "confirmations",
    "finalize",
)
V7_STAGES = (
    "audit",
    "probe",
    "preflight",
    "core_candidates",
    "select_core",
    "extension_candidates",
    "finalize",
)
STAGES = tuple(dict.fromkeys((*LEGACY_STAGES, *V6_STAGES, *V7_STAGES)))


def _is_v5(spec: LRStudySpec) -> bool:
    return spec.data.get("schema_version") == "mnist-conv-lr-study/v5"


def _is_v6(spec: LRStudySpec) -> bool:
    return spec.data.get("schema_version") == "mnist-conv-lr-study/v6"


def _is_v7(spec: LRStudySpec) -> bool:
    return spec.data.get("schema_version") == "mnist-conv-lr-study/v7"


def _contract_path(study_dir: Path) -> Path:
    return study_dir / "study.resolved.json"


def _manifest_path(study_dir: Path, stage: str) -> Path:
    return study_dir / "stages" / stage / "manifest.json"


def _entry_dir(stage: str, entry_id: str) -> str:
    return f"stages/{stage}/entries/{entry_id}"


def _publish_contract(spec: LRStudySpec, study_dir: Path) -> Path:
    target = _contract_path(study_dir)
    value = spec.data
    if target.exists():
        observed = read_json(target)
        if observed != value:
            raise RuntimeError(
                "Expected the content-addressed study directory to contain the identical "
                f"resolved contract. Provided value: {target}."
            )
    else:
        atomic_write_json(target, value, canonical=True)
    return target


def create_study(
    spec: LRStudySpec,
    results_root: str | Path,
) -> tuple[Path, Path]:
    layout = ResultLayout(results_root)
    layout.ensure_roots()
    study_dir = layout.lr_study_dir(spec.data["name"], spec.study_id)
    study_dir.mkdir(parents=True, exist_ok=True)
    return study_dir, _publish_contract(spec, study_dir)


def load_study(study_dir: str | Path) -> tuple[Path, LRStudySpec]:
    root = Path(study_dir).expanduser().resolve()
    if root.parent.name != "lr_studies":
        raise ValueError(
            "Expected --study to be <results-root>/lr_studies/<name>--<lrstudy-id>. "
            f"Provided value: {root}."
        )
    spec = LRStudySpec.from_path(_contract_path(root))
    expected_suffix = f"--{spec.study_id}"
    if not root.name.endswith(expected_suffix):
        raise ValueError(
            f"Expected study directory name to end with {expected_suffix!r}. "
            f"Provided value: {root.name!r}."
        )
    return root, spec


def _outputs(
    stage: str,
    entry_id: str,
    *,
    candidate_run_spec_version: int = 2,
    architecture_v5: bool = False,
    two_rho_v6: bool = False,
    conv3_v7: bool = False,
) -> list[str]:
    base = _entry_dir(stage, entry_id)
    if stage == "audit":
        if conv3_v7:
            return sorted(
                [f"{base}/asset_audit.csv", f"{base}/summary.json"]
            )
        if two_rho_v6:
            return sorted(
                [
                    f"{base}/reuse_audit.csv",
                    f"{base}/summary.json",
                    "reuse/import_receipt.json",
                ]
            )
        names = ["anchors.csv", "summary.json"]
    elif stage in {"probe", "range"}:
        names = [
            "minibatches.json",
            "parameter_diagnostics.csv",
            "step_log.csv",
            "summary.json",
        ]
    elif stage == "preflight" and conv3_v7:
        names = [
            "minibatches.json",
            "parameter_diagnostics.csv",
            "step_log.csv",
            "summary.json",
            "validation.json",
        ]
    elif stage in {
        "candidates",
        "baseline_candidates",
        "confirmations",
        "core_candidates",
        "extension_candidates",
    }:
        names = [
            "best_validation.pt",
            "final.pt",
            "minibatches.json",
            "parameter_diagnostics.csv",
            f"run_spec.v{candidate_run_spec_version}.json",
            "step_log.csv",
            "summary.json",
            "validation.json",
        ]
    elif stage == "select_core" and conv3_v7:
        names = [
            "achieved_relative_updates_heatmap.png",
            "epoch3_accuracy_loss_heatmap.png",
            "occupancy_heatmap.png",
            "projection_efficiency_heatmap.png",
            "raw_learning_rates_heatmap.png",
            "selection.csv",
            "selection.json",
        ]
    elif stage in {"select", "select_baseline"}:
        names = (
            [
                "achieved_relative_updates_heatmap.png",
                "epoch5_accuracy_loss_heatmap.png",
                "occupancy_heatmap.png",
                "projection_efficiency_heatmap.png",
                "raw_learning_rates_heatmap.png",
                "selection.csv",
                "selection.json",
            ]
            if two_rho_v6
            else
            [
                "achieved_relative_updates.png",
                "epoch5_accuracy_loss.png",
                "occupancy.png",
                "projection_efficiency.png",
                "raw_learning_rates.png",
                "selection.csv",
                "selection.json",
            ]
            if architecture_v5
            else [
                "observed_normalized_peak_update_vs_input_gain.png",
                "selected_raw_peak_lr_vs_input_gain.png",
                "selection.csv",
                "selection.json",
            ]
        )
    elif stage == "finalize" and (two_rho_v6 or conv3_v7):
        names = (
            [
                "achieved_relative_updates_heatmap.png",
                "epoch3_accuracy_loss_heatmap.png",
                "finalization.csv",
                "finalization.json",
                "occupancy_heatmap.png",
                "projection_efficiency_heatmap.png",
                "raw_learning_rates_heatmap.png",
            ]
            if conv3_v7
            else ["finalization.csv", "finalization.json"]
        )
    else:
        raise ValueError(f"Expected stage in {STAGES!r}. Provided value: {stage!r}.")
    return sorted(f"{base}/{name}" for name in names)


def _completion_path(stage: str, entry_id: str) -> str:
    return f"{_entry_dir(stage, entry_id)}/complete.json"


def _configured_architectures(spec: LRStudySpec) -> tuple[str, ...]:
    """Return architectures represented by rows, preserving contract order."""

    return tuple(dict.fromkeys(row["architecture"] for row in spec.rows))


def _probe_upstreams(study_dir: Path, spec: LRStudySpec) -> list[Path]:
    paths = [
        study_dir / "split" / "indices.json",
        study_dir / "split" / "provenance.json",
    ]
    for architecture in _configured_architectures(spec):
        paths.extend(
            [
                study_dir / "initialization" / f"{architecture}.pt",
                study_dir / "initialization" / f"{architecture}.json",
            ]
        )
    rescue = spec.data.get("rescue")
    if (
        isinstance(rescue, dict)
        and rescue.get("import_mode")
        in {"copy_and_verify_parent_probe", "copy_and_verify_parent_assets"}
    ):
        paths.append(study_dir / "rescue" / "import_receipt.json")
    # V6 declares the reuse receipt as an output of its audit entry. Every
    # post-audit v6 stage binds all completed audit outputs, so adding it here
    # as well would create a duplicate upstream path and make manifest
    # publication fail closed in ``artifact_records``.
    return paths


def _v6_completed_stage_upstreams(
    study_dir: Path,
    stage: str,
    *,
    include_entry_summaries: bool = True,
) -> list[Path]:
    manifest = _manifest_path(study_dir, stage)
    validate_stage_completion(study_dir=study_dir, manifest_path=manifest)
    value = load_stage_manifest(manifest, study_dir=study_dir)
    paths = [manifest, manifest.parent / "complete.json"]
    if include_entry_summaries:
        for entry in value["entries"]:
            paths.append(study_dir / entry["completion_path"])
            paths.extend(study_dir / output for output in entry["outputs"])
    return paths


def _v6_stage_upstreams(
    study_dir: Path, spec: LRStudySpec, stage: str
) -> list[Path]:
    if stage == "audit":
        # The audit itself owns import/fallback materialization.  Requiring
        # split/init here would make the fallback impossible to schedule on a
        # compute node.
        return []
    paths = _probe_upstreams(study_dir, spec)
    paths.extend(_v6_completed_stage_upstreams(study_dir, "audit"))
    if stage == "probe":
        return paths
    paths.extend(_v6_completed_stage_upstreams(study_dir, "probe"))
    if stage == "baseline_candidates":
        return paths
    paths.extend(
        _v6_completed_stage_upstreams(study_dir, "baseline_candidates")
    )
    if stage == "select_baseline":
        return paths
    paths.extend(_v6_completed_stage_upstreams(study_dir, "select_baseline"))
    if stage == "confirmations":
        return paths
    if stage == "finalize":
        selection = read_json(
            study_dir
            / _entry_dir("select_baseline", "selection")
            / "selection.json"
        )
        if selection["status"] == "selected":
            paths.extend(_v6_completed_stage_upstreams(study_dir, "confirmations"))
        return paths
    raise ValueError(
        f"Expected v6 stage in {V6_STAGES!r}. Provided value: {stage!r}."
    )


def _v7_stage_upstreams(
    study_dir: Path, spec: LRStudySpec, stage: str
) -> list[Path]:
    if stage == "audit":
        return []
    paths = _probe_upstreams(study_dir, spec)
    paths.extend(_v6_completed_stage_upstreams(study_dir, "audit"))
    if stage == "probe":
        return paths
    paths.extend(_v6_completed_stage_upstreams(study_dir, "probe"))
    if stage == "preflight":
        return paths
    paths.extend(_v6_completed_stage_upstreams(study_dir, "preflight"))
    preflight = read_json(
        study_dir
        / _entry_dir("preflight", "ours-worst-cost")
        / "summary.json"
    )
    if preflight.get("status") != "passed":
        raise RuntimeError(
            "Expected the required v7 V100 preflight to pass before core "
            f"candidates. Provided value: {preflight.get('status')!r}."
        )
    if stage == "core_candidates":
        return paths
    paths.extend(_v6_completed_stage_upstreams(study_dir, "core_candidates"))
    if stage == "select_core":
        return paths
    paths.extend(_v6_completed_stage_upstreams(study_dir, "select_core"))
    if stage == "extension_candidates":
        return paths
    paths.extend(
        _v6_completed_stage_upstreams(study_dir, "extension_candidates")
    )
    if stage == "finalize":
        return paths
    raise ValueError(
        f"Expected v7 stage in {V7_STAGES!r}. Provided value: {stage!r}."
    )


def _stage_upstreams(study_dir: Path, spec: LRStudySpec, stage: str) -> list[Path]:
    if _is_v7(spec):
        return _v7_stage_upstreams(study_dir, spec, stage)
    if _is_v6(spec):
        return _v6_stage_upstreams(study_dir, spec, stage)
    paths = _probe_upstreams(study_dir, spec)
    if stage == "audit":
        if not _is_v5(spec):
            raise ValueError(
                "Expected the anchor-audit stage only for mnist-conv-lr-study/v5. "
                f"Provided schema: {spec.data.get('schema_version')!r}."
            )
        return paths
    if stage == "probe":
        if _is_v5(spec):
            audit_manifest = _manifest_path(study_dir, "audit")
            validate_stage_completion(
                study_dir=study_dir, manifest_path=audit_manifest
            )
            paths.extend(
                [
                    audit_manifest,
                    audit_manifest.parent / "complete.json",
                    study_dir
                    / _entry_dir("audit", "anchor-audit")
                    / "summary.json",
                ]
            )
        return paths
    probe_manifest = _manifest_path(study_dir, "probe")
    validate_stage_completion(study_dir=study_dir, manifest_path=probe_manifest)
    paths.extend(
        [
            probe_manifest,
            probe_manifest.parent / "complete.json",
            *(
                study_dir / _entry_dir("probe", row["row_id"]) / "summary.json"
                for row in spec.rows
            ),
            *(
                study_dir / _entry_dir("probe", row["row_id"]) / "minibatches.json"
                for row in spec.rows
            ),
        ]
    )
    if stage == "range":
        if _is_v5(spec):
            raise ValueError(
                "Expected v5 to skip the increasing-LR range stage. "
                "Provided stage: 'range'."
            )
        return paths
    if _is_v5(spec):
        if stage == "candidates":
            return paths
        if stage == "select":
            candidate_manifest = _manifest_path(study_dir, "candidates")
            validate_stage_completion(
                study_dir=study_dir, manifest_path=candidate_manifest
            )
            candidate_manifest_value = load_stage_manifest(
                candidate_manifest, study_dir=study_dir
            )
            paths.extend(
                [
                    candidate_manifest,
                    candidate_manifest.parent / "complete.json",
                    *(
                        study_dir
                        / _entry_dir("candidates", entry["entry_id"])
                        / "summary.json"
                        for entry in candidate_manifest_value["entries"]
                    ),
                ]
            )
            return paths
    range_manifest = _manifest_path(study_dir, "range")
    validate_stage_completion(study_dir=study_dir, manifest_path=range_manifest)
    paths.extend(
        [
            range_manifest,
            range_manifest.parent / "complete.json",
            *(
                study_dir / _entry_dir("range", row["row_id"]) / "summary.json"
                for row in spec.rows
            ),
        ]
    )
    if stage == "candidates":
        return paths
    candidate_manifest = _manifest_path(study_dir, "candidates")
    validate_stage_completion(study_dir=study_dir, manifest_path=candidate_manifest)
    candidate_manifest_value = load_stage_manifest(
        candidate_manifest, study_dir=study_dir
    )
    paths.extend(
        [
            candidate_manifest,
            candidate_manifest.parent / "complete.json",
            *(
                study_dir / _entry_dir("candidates", entry["entry_id"]) / "summary.json"
                for entry in candidate_manifest_value["entries"]
            ),
        ]
    )
    return paths


def _v6_probe_payload(study_dir: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    probe = read_json(
        study_dir / _entry_dir("probe", row["row_id"]) / "summary.json"
    )
    units = {
        name: float(value)
        for name, value in probe["median_units_by_weight"].items()
    }
    groups = {
        name: list(members)
        for name, members in probe["bias_weight_lr_groups"].items()
    }
    return {"median_units_by_weight": units, "bias_weight_lr_groups": groups}


def _v6_candidate_entry(
    spec: LRStudySpec,
    study_dir: Path,
    *,
    stage: str,
    row: Mapping[str, Any],
    role: str,
    candidate_stage: str,
    rho_conv: float,
    rho_dense: float,
    index: int,
) -> dict[str, Any]:
    probe = _v6_probe_payload(study_dir, row)
    rates = two_rho_learning_rates(
        probe["median_units_by_weight"],
        rho_conv=rho_conv,
        rho_dense=rho_dense,
        bias_weight_lr_groups=probe["bias_weight_lr_groups"],
    )
    report = layerwise_learning_rate_report(
        rates,
        thresholds=spec.data["artifacts"]["large_raw_lr_reporting"]["thresholds"],
    )
    entry_id = f"{row['row_id']}--{role}"
    return stage_entry(
        index,
        entry_id,
        completion_path=_completion_path(stage, entry_id),
        outputs=_outputs(
            stage,
            entry_id,
            candidate_run_spec_version=5,
            two_rho_v6=True,
        ),
        payload={
            "row_id": row["row_id"],
            "architecture": row["architecture"],
            "scheme": row["scheme"],
            "candidate_role": role,
            "candidate_stage": candidate_stage,
            "rho_conv": float(rho_conv),
            "rho_dense": float(rho_dense),
            "median_units_by_weight": probe["median_units_by_weight"],
            "learning_rates_by_parameter": rates,
            "learning_rates_by_weight": report["weight_learning_rates"],
            "peak_learning_rate": report["maximum_weight_learning_rate"],
        },
    )


def _v6_entries(
    spec: LRStudySpec, study_dir: Path, stage: str
) -> list[dict[str, Any]]:
    grid = spec.data["rho_grid"]
    if stage == "audit":
        entry_id = "reuse-audit"
        return [
            stage_entry(
                0,
                entry_id,
                completion_path=_completion_path(stage, entry_id),
                outputs=_outputs(stage, entry_id, two_rho_v6=True),
                payload={
                    "reuse_mode": spec.data["reuse"]["mode"],
                    "source_study_id": spec.data["reuse"]["source_study_id"],
                },
            )
        ]
    if stage == "probe":
        return [
            stage_entry(
                index,
                row["row_id"],
                completion_path=_completion_path(stage, row["row_id"]),
                outputs=_outputs(stage, row["row_id"], two_rho_v6=True),
                payload={"row": row},
            )
            for index, row in enumerate(spec.rows)
        ]
    if stage == "baseline_candidates":
        baseline = next(
            row for row in spec.rows if row["row_id"] == grid["baseline_row_id"]
        )
        entries: list[dict[str, Any]] = []
        for conv_index, rho_conv in enumerate(grid["rho_conv"]):
            for dense_index, rho_dense in enumerate(grid["rho_dense"]):
                entries.append(
                    _v6_candidate_entry(
                        spec,
                        study_dir,
                        stage=stage,
                        row=baseline,
                        role=f"grid-c{conv_index:02d}-d{dense_index:02d}",
                        candidate_stage="baseline_grid",
                        rho_conv=float(rho_conv),
                        rho_dense=float(rho_dense),
                        index=len(entries),
                    )
                )
        if len(entries) != int(grid["baseline_candidate_count"]):
            raise RuntimeError(
                "Expected the v6 baseline grid to contain exactly "
                f"{grid['baseline_candidate_count']} candidates. Provided value: "
                f"{len(entries)}."
            )
        return entries
    if stage == "select_baseline":
        entry_id = "selection"
        return [
            stage_entry(
                0,
                entry_id,
                completion_path=_completion_path(stage, entry_id),
                outputs=_outputs(stage, entry_id, two_rho_v6=True),
                payload={"baseline_candidate_count": grid["baseline_candidate_count"]},
            )
        ]
    if stage == "confirmations":
        selection = read_json(
            study_dir
            / _entry_dir("select_baseline", "selection")
            / "selection.json"
        )
        if selection["status"] != "selected":
            raise RuntimeError(
                "Expected a bracketed v6 baseline selection before creating "
                "confirmation entries. Provided value: "
                f"{selection['status']!r}."
            )
        rho_conv = float(selection["selected_rho_conv"])
        rho_dense = float(selection["selected_rho_dense"])
        rows_by_id = {row["row_id"]: row for row in spec.rows}
        return [
            _v6_candidate_entry(
                spec,
                study_dir,
                stage=stage,
                row=rows_by_id[row_id],
                role="confirmation",
                candidate_stage="confirmation",
                rho_conv=rho_conv,
                rho_dense=rho_dense,
                index=index,
            )
            for index, row_id in enumerate(grid["confirmation_row_ids"])
        ]
    if stage == "finalize":
        entry_id = "finalization"
        return [
            stage_entry(
                0,
                entry_id,
                completion_path=_completion_path(stage, entry_id),
                outputs=_outputs(stage, entry_id, two_rho_v6=True),
                payload={"maximum_canonical_training_runs": 18},
            )
        ]
    raise ValueError(
        f"Expected v6 stage in {V6_STAGES!r}. Provided value: {stage!r}."
    )


def _v7_candidate_entry(
    spec: LRStudySpec,
    study_dir: Path,
    *,
    stage: str,
    row: Mapping[str, Any],
    role: str,
    candidate_stage: str,
    rho_conv: float,
    rho_dense: float,
    index: int,
) -> dict[str, Any]:
    probe = _v6_probe_payload(study_dir, row)
    rates = two_rho_learning_rates(
        probe["median_units_by_weight"],
        rho_conv=rho_conv,
        rho_dense=rho_dense,
        bias_weight_lr_groups=probe["bias_weight_lr_groups"],
    )
    report = layerwise_learning_rate_report(
        rates,
        thresholds=spec.data["artifacts"]["large_raw_lr_reporting"][
            "thresholds"
        ],
    )
    entry_id = f"{row['row_id']}--{role}"
    return stage_entry(
        index,
        entry_id,
        completion_path=_completion_path(stage, entry_id),
        outputs=_outputs(
            stage,
            entry_id,
            candidate_run_spec_version=7,
            conv3_v7=True,
        ),
        payload={
            "row_id": row["row_id"],
            "architecture": row["architecture"],
            "scheme": row["scheme"],
            "candidate_role": role,
            "candidate_stage": candidate_stage,
            "rho_conv": float(rho_conv),
            "rho_dense": float(rho_dense),
            "median_units_by_weight": probe["median_units_by_weight"],
            "learning_rates_by_parameter": rates,
            "learning_rates_by_weight": report["weight_learning_rates"],
            "peak_learning_rate": report["maximum_weight_learning_rate"],
        },
    )


def _v7_entries(
    spec: LRStudySpec, study_dir: Path, stage: str
) -> list[dict[str, Any]]:
    grid = spec.data["rho_grid"]
    core = grid["core"]
    if stage == "audit":
        entry_id = "asset-audit"
        return [
            stage_entry(
                0,
                entry_id,
                completion_path=_completion_path(stage, entry_id),
                outputs=_outputs(stage, entry_id, conv3_v7=True),
                payload={"architecture": "conv3", "model_seed": 0},
            )
        ]
    if stage == "probe":
        return [
            stage_entry(
                index,
                row["row_id"],
                completion_path=_completion_path(stage, row["row_id"]),
                outputs=_outputs(stage, row["row_id"], conv3_v7=True),
                payload={"row": row},
            )
            for index, row in enumerate(spec.rows)
        ]
    if stage == "preflight":
        contract = spec.data["preflight"]
        row = next(
            row
            for row in spec.rows
            if row["row_id"] == contract["row_id"]
        )
        return [
            _v7_candidate_entry(
                spec,
                study_dir,
                stage=stage,
                row=row,
                role="ours-worst-cost",
                candidate_stage="preflight",
                rho_conv=float(contract["rho_conv"]),
                rho_dense=float(contract["rho_dense"]),
                index=0,
            )
            | {
                "entry_id": "ours-worst-cost",
                "completion_path": _completion_path(
                    stage, "ours-worst-cost"
                ),
                "outputs": _outputs(
                    stage, "ours-worst-cost", conv3_v7=True
                ),
            }
        ]
    if stage == "core_candidates":
        entries: list[dict[str, Any]] = []
        for row in spec.rows:
            for conv_index, rho_conv in enumerate(core["rho_conv"]):
                for dense_index, rho_dense in enumerate(core["rho_dense"]):
                    entries.append(
                        _v7_candidate_entry(
                            spec,
                            study_dir,
                            stage=stage,
                            row=row,
                            role=(
                                f"core-c{conv_index:02d}-d{dense_index:02d}"
                            ),
                            candidate_stage="core_grid",
                            rho_conv=float(rho_conv),
                            rho_dense=float(rho_dense),
                            index=len(entries),
                        )
                    )
        if len(entries) != int(core["total_candidate_count"]):
            raise RuntimeError(
                "Expected exactly 27 v7 core candidates. "
                f"Provided value: {len(entries)}."
            )
        return entries
    if stage == "select_core":
        entry_id = "selection"
        return [
            stage_entry(
                0,
                entry_id,
                completion_path=_completion_path(stage, entry_id),
                outputs=_outputs(stage, entry_id, conv3_v7=True),
                payload={"candidate_count": int(core["total_candidate_count"])},
            )
        ]
    if stage == "extension_candidates":
        selection = read_json(
            study_dir
            / _entry_dir("select_core", "selection")
            / "selection.json"
        )
        rows_by_id = {row["row_id"]: row for row in spec.rows}
        selected_rows = selection.get("rows")
        expected_row_ids = list(rows_by_id)
        if (
            not isinstance(selected_rows, list)
            or [row.get("row_id") for row in selected_rows]
            != expected_row_ids
        ):
            raise RuntimeError(
                "Expected the v7 core selection to contain every configured row "
                f"once in frozen order. Provided value: {selected_rows!r}."
            )
        entries = []
        maximum_new_per_row = int(
            grid["expansion"]["maximum_new_candidates_per_row"]
        )
        allowed_axes = {
            (),
            ("rho_conv",),
            ("rho_dense",),
            ("rho_conv", "rho_dense"),
        }
        for selected_row in selected_rows:
            axes = tuple(selected_row["expansion_axes"])
            if axes not in allowed_axes:
                raise RuntimeError(
                    "Expected each v7 row to request no expansion or one frozen "
                    "Conv/Dense expansion wave. "
                    f"Provided value: row={selected_row['row_id']!r}, "
                    f"axes={axes!r}."
                )
            if not axes:
                continue
            row_entry_start = len(entries)
            conv_values = [float(value) for value in core["rho_conv"]]
            dense_values = [float(value) for value in core["rho_dense"]]
            if "rho_conv" in axes:
                conv_values.append(float(grid["expansion"]["rho_conv"]))
            if "rho_dense" in axes:
                dense_values.append(float(grid["expansion"]["rho_dense"]))
            for conv_index, rho_conv in enumerate(conv_values):
                for dense_index, rho_dense in enumerate(dense_values):
                    if (
                        rho_conv in core["rho_conv"]
                        and rho_dense in core["rho_dense"]
                    ):
                        continue
                    entries.append(
                        _v7_candidate_entry(
                            spec,
                            study_dir,
                            stage=stage,
                            row=rows_by_id[selected_row["row_id"]],
                            role=(
                                "extension-"
                                f"c{conv_index:02d}-d{dense_index:02d}"
                            ),
                            candidate_stage="extension_grid",
                            rho_conv=rho_conv,
                            rho_dense=rho_dense,
                            index=len(entries),
                        )
                    )
            row_entry_count = len(entries) - row_entry_start
            if row_entry_count > maximum_new_per_row:
                raise RuntimeError(
                    "Expected at most the configured number of new v7 candidates "
                    f"for one scheme. Provided value: row={selected_row['row_id']!r}, "
                    f"count={row_entry_count}."
                )
        maximum_extension_count = (
            int(grid["expansion"]["maximum_total_training_runs"])
            - int(core["total_candidate_count"])
        )
        if len(entries) > maximum_extension_count:
            raise RuntimeError(
                "Expected the v7 extension wave to preserve the frozen total run cap. "
                f"Provided value: {len(entries)}."
            )
        if entries:
            return entries
        entry_id = "no-expansion-required"
        return [
            stage_entry(
                0,
                entry_id,
                completion_path=_completion_path(stage, entry_id),
                outputs=[f"{_entry_dir(stage, entry_id)}/summary.json"],
                payload={
                    "no_op": True,
                    "reason": "no_scheme_requested_expansion",
                },
            )
        ]
    if stage == "finalize":
        entry_id = "finalization"
        return [
            stage_entry(
                0,
                entry_id,
                completion_path=_completion_path(stage, entry_id),
                outputs=_outputs(stage, entry_id, conv3_v7=True),
                payload={
                    "maximum_candidate_training_runs": int(
                        grid["expansion"]["maximum_total_training_runs"]
                    ),
                    "partial_handoff_allowed": True,
                },
            )
        ]
    raise ValueError(
        f"Expected v7 stage in {V7_STAGES!r}. Provided value: {stage!r}."
    )


def _entries(spec: LRStudySpec, study_dir: Path, stage: str) -> list[dict[str, Any]]:
    if _is_v7(spec):
        return _v7_entries(spec, study_dir, stage)
    if _is_v6(spec):
        return _v6_entries(spec, study_dir, stage)
    if stage == "audit":
        if not _is_v5(spec):
            raise ValueError(
                "Expected the anchor-audit stage only for mnist-conv-lr-study/v5. "
                f"Provided schema: {spec.data.get('schema_version')!r}."
            )
        return [
            stage_entry(
                0,
                "anchor-audit",
                completion_path=_completion_path(stage, "anchor-audit"),
                outputs=_outputs(stage, "anchor-audit"),
                payload={"derivation_status": spec.data["anchor_audit"]["derivation_status"]},
            )
        ]
    if stage == "range" and _is_v5(spec):
        raise ValueError(
            "Expected v5 to run candidates directly after its median probe. "
            "Provided stage: 'range'."
        )
    if stage in {"probe", "range"}:
        reused_probe = (
            stage == "probe"
            and spec.data.get("rescue", {}).get("import_mode")
            == "copy_and_verify_parent_probe"
        )
        return [
            stage_entry(
                index,
                row["row_id"],
                completion_path=_completion_path(stage, row["row_id"]),
                outputs=_outputs(stage, row["row_id"]),
                payload={
                    "row": row,
                    **(
                        {"execution_mode": "reused_parent_artifact"}
                        if reused_probe
                        else {}
                    ),
                },
            )
            for index, row in enumerate(spec.rows)
        ]
    if stage == "candidates":
        if _is_v5(spec):
            values: list[dict[str, Any]] = []
            policy = spec.data["target_policies"]
            for architecture in _configured_architectures(spec):
                architecture_rows = [
                    row for row in spec.rows if row["architecture"] == architecture
                ]
                for arm in policy["arm_order"]:
                    arm_contract = policy["arms"][arm]
                    for alpha_role, factor in zip(
                        policy["candidate_roles"],
                        policy["candidate_factors"],
                        strict=True,
                    ):
                        alpha = float(
                            policy["centers_by_architecture_and_arm"][architecture][arm]
                        ) * float(factor)
                        for row in architecture_rows:
                            probe = read_json(
                                study_dir
                                / _entry_dir("probe", row["row_id"])
                                / "summary.json"
                            )
                            units = {
                                name: float(value)
                                for name, value in probe[
                                    "rho_unit_relative_by_parameter"
                                ].items()
                            }
                            groups = probe["bias_weight_lr_groups"]
                            parameter_names = [
                                member
                                for members in groups.values()
                                for member in members
                            ]
                            multipliers = {
                                name: (
                                    float(
                                        arm_contract[
                                            "dense_multiplier_by_architecture"
                                        ][architecture]
                                    )
                                    if name.startswith("DenseWeight_")
                                    else float(arm_contract["conv_multiplier"])
                                )
                                for name in units
                            }
                            rates = architecture_relative_learning_rates(
                                units,
                                alpha,
                                parameter_names,
                                target_multipliers=multipliers,
                            )
                            report = layerwise_learning_rate_report(
                                rates,
                                thresholds=spec.data["artifacts"][
                                    "large_raw_lr_reporting"
                                ]["thresholds"],
                            )
                            candidate_role = f"{arm}--{alpha_role}"
                            entry_id = f"{row['row_id']}--{candidate_role}"
                            values.append(
                                stage_entry(
                                    len(values),
                                    entry_id,
                                    completion_path=_completion_path(stage, entry_id),
                                    outputs=_outputs(
                                        stage,
                                        entry_id,
                                        candidate_run_spec_version=4,
                                    ),
                                    payload={
                                        "row_id": row["row_id"],
                                        "architecture": architecture,
                                        "scheme": row["scheme"],
                                        "candidate_role": candidate_role,
                                        "arm": arm,
                                        "alpha_role": alpha_role,
                                        "alpha": alpha,
                                        "target_multipliers_by_weight": multipliers,
                                        "median_units_by_weight": units,
                                        "learning_rates_by_parameter": rates,
                                        "learning_rates_by_weight": report[
                                            "weight_learning_rates"
                                        ],
                                        "peak_learning_rate": report[
                                            "maximum_weight_learning_rate"
                                        ],
                                    },
                                )
                            )
            if len(values) != 36:
                raise RuntimeError(
                    "Expected v5 to create exactly 36 five-epoch candidates. "
                    f"Provided value: {len(values)}."
                )
            return values
        values: list[dict[str, Any]] = []
        layerwise = (
            spec.data["probe"].get("rho_definition")
            == "initial_parameter_rms_per_bounded_weight"
        )
        for row in spec.rows:
            summary = read_json(
                study_dir / _entry_dir("range", row["row_id"]) / "summary.json"
            )
            candidate_set = summary["candidates"]
            if candidate_set["status"] != "resolved":
                continue
            for role in spec.data["candidate_training"]["candidate_roles"]:
                candidate = candidate_set[role]
                entry_id = f"{row['row_id']}--{role}"
                values.append(
                    stage_entry(
                        len(values),
                        entry_id,
                        completion_path=_completion_path(stage, entry_id),
                        outputs=_outputs(
                            stage,
                            entry_id,
                            candidate_run_spec_version=3 if layerwise else 2,
                        ),
                        payload={
                            "row_id": row["row_id"],
                            "candidate_role": role,
                            "rho_target": candidate["rho_target"],
                            "peak_learning_rate": candidate["learning_rate"],
                            **(
                                {
                                    "learning_rates_by_parameter": candidate[
                                        "learning_rates_by_parameter"
                                    ],
                                    "learning_rates_by_weight": candidate[
                                        "learning_rates_by_weight"
                                    ],
                                }
                                if layerwise
                                else {}
                            ),
                        },
                    )
                )
        if not values:
            entry_id = "no-resolved-candidates"
            return [
                stage_entry(
                    0,
                    entry_id,
                    completion_path=_completion_path(stage, entry_id),
                    outputs=[f"{_entry_dir(stage, entry_id)}/summary.json"],
                    payload={
                        "no_op": True,
                        "reason": "all_range_rows_unresolved",
                    },
                )
            ]
        return values
    if stage == "select":
        return [
            stage_entry(
                0,
                "selection",
                completion_path=_completion_path(stage, "selection"),
                outputs=_outputs(
                    stage, "selection", architecture_v5=_is_v5(spec)
                ),
                payload={"row_count": len(spec.rows)},
            )
        ]
    raise ValueError(f"Expected stage in {STAGES!r}. Provided value: {stage!r}.")


def publish_lr_stage_manifest(
    study_dir: Path,
    spec: LRStudySpec,
    stage: str,
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    manifest_path = _manifest_path(study_dir, stage)
    entries = _entries(spec, study_dir, stage)
    manifest = build_stage_manifest(
        study_dir=study_dir,
        study_id=spec.study_id,
        study_config_path=_contract_path(study_dir),
        code_provenance=provenance,
        stage_name=stage,
        entries=entries,
        upstream_paths=_stage_upstreams(study_dir, spec, stage),
    )
    publish_stage_manifest(manifest_path, manifest, study_dir=study_dir)
    return manifest, manifest_path


def _require_current_source(
    planned: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    expected = normalize_code_provenance(dict(planned))
    observed = normalize_code_provenance(dict(current))
    # Git metadata is useful provenance, but copied worktrees and source
    # archives legitimately lack the originating repository object database.
    # The effective fingerprint hashes every executable Conv source file and
    # is therefore the portable worker-identity gate across local, SSH, and
    # Slurm executors.
    if (
        expected["effective_code_fingerprint"]
        != observed["effective_code_fingerprint"]
    ):
        raise RuntimeError(
            "Expected worker effective source fingerprint to match the immutable "
            "stage manifest. "
            f"Provided value: planned={expected!r}, current={observed!r}."
        )


def execute_manifest_entry(
    *,
    study_dir: str | Path,
    manifest_path: str | Path,
    entry_index: int,
    data_root: str | Path,
    download: bool,
    device: str,
    current_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root, spec = load_study(study_dir)
    manifest = load_stage_manifest(
        manifest_path,
        study_dir=root,
        expected_study_id=spec.study_id,
    )
    _require_current_source(
        manifest["code_provenance"],
        current_provenance or code_provenance(),
    )
    if type(entry_index) is not int or not 0 <= entry_index < len(manifest["entries"]):
        raise ValueError(
            f"Expected entry_index in [0, {len(manifest['entries']) - 1}]. "
            f"Provided value: {entry_index!r}."
        )
    entry = manifest["entries"][entry_index]
    stage = manifest["stage_name"]
    if _is_v6(spec) and stage in {
        "audit",
        "probe",
        "baseline_candidates",
        "confirmations",
    }:
        require_v6_r3_authorization()
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
    payload = entry["payload"]
    if stage == "audit":
        if _is_v7(spec):
            result = execute_v7_asset_audit(
                spec.data,
                root,
                data_root=data_root,
                download=download,
                device=device,
            )
        elif _is_v6(spec):
            result = execute_v6_reuse_audit(
                spec.data,
                root,
                data_root=data_root,
                download=download,
                device=device,
            )
        else:
            result = execute_anchor_audit(spec.data, root)
    elif stage == "probe":
        result = execute_probe_entry(
            spec.data,
            root,
            payload["row"]["row_id"],
            data_root=data_root,
            download=download,
            device=device,
        )
    elif stage == "range":
        result = execute_range_entry(
            spec.data,
            root,
            payload["row"]["row_id"],
            data_root=data_root,
            download=download,
            device=device,
        )
    elif stage == "preflight" and _is_v7(spec):
        result = execute_v7_preflight_entry(
            spec.data,
            root,
            candidate_payload=payload,
            data_root=data_root,
            download=download,
            device=device,
        )
    elif stage in {
        "candidates",
        "baseline_candidates",
        "confirmations",
        "core_candidates",
        "extension_candidates",
    }:
        if payload.get("no_op") is True:
            result = (
                execute_v7_no_extension_entry(
                    spec.data, root, entry["entry_id"]
                )
                if _is_v7(spec)
                else execute_no_candidate_entry(
                    spec.data, root, entry["entry_id"]
                )
            )
        else:
            result = execute_candidate_entry(
                spec.data,
                root,
                payload["row_id"],
                payload["candidate_role"],
                candidate_payload=(
                    payload
                    if (_is_v5(spec) or _is_v6(spec) or _is_v7(spec))
                    else None
                ),
                output_stage=stage,
                data_root=data_root,
                download=download,
                device=device,
            )
    elif stage == "select":
        result = execute_selection(spec.data, root)
    elif stage == "select_baseline" and _is_v6(spec):
        result = execute_v6_baseline_selection(spec.data, root)
    elif stage == "select_core" and _is_v7(spec):
        result = execute_v7_core_selection(spec.data, root)
    elif stage == "finalize" and _is_v6(spec):
        result = execute_v6_finalization(spec.data, root)
    elif stage == "finalize" and _is_v7(spec):
        result = execute_v7_finalization(spec.data, root)
    else:
        raise ValueError(f"Expected stage in {STAGES!r}. Provided value: {stage!r}.")
    publish_entry_completion(
        study_dir=root,
        manifest_path=manifest_path,
        entry_id=entry["entry_id"],
    )
    return {
        "entry_index": entry_index,
        "entry_id": entry["entry_id"],
        "status": "complete",
        "result_status": result.get("status") if isinstance(result, dict) else None,
    }


def finalize_stage(study_dir: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    root, spec = load_study(study_dir)
    manifest = load_stage_manifest(
        manifest_path, study_dir=root, expected_study_id=spec.study_id
    )
    marker = publish_stage_completion(
        study_dir=root,
        manifest_path=manifest_path,
    )
    return {
        "study_id": spec.study_id,
        "stage": manifest["stage_name"],
        "status": "complete",
        "completion_path": str(marker),
    }


def execute_local_stage(
    *,
    study_dir: Path,
    manifest_path: Path,
    data_root: str | Path,
    download: bool,
    device: str,
    workers: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = load_stage_manifest(manifest_path, study_dir=study_dir)
    pending = [
        entry
        for entry in manifest["entries"]
        if not entry_is_complete(
            study_dir=study_dir,
            manifest_path=manifest_path,
            entry_id=entry["entry_id"],
        )
    ]
    if type(workers) is not int or workers < 1:
        raise ValueError(
            f"Expected workers to be an integer >= 1. Provided value: {workers!r}."
        )

    def one(entry: dict[str, Any]) -> dict[str, Any]:
        if workers == 1:
            return execute_manifest_entry(
                study_dir=study_dir,
                manifest_path=manifest_path,
                entry_index=entry["entry_index"],
                data_root=data_root,
                download=download,
                device=device,
                current_provenance=provenance,
            )
        command = [
            sys.executable,
            "-m",
            "experiments.mnist_conv",
            "lr-study",
            "--stage",
            manifest["stage_name"],
            "--study",
            str(study_dir),
            "--manifest",
            str(manifest_path),
            "--entry-index",
            str(entry["entry_index"]),
            "--data-root",
            str(Path(data_root).expanduser()),
            "--device",
            device,
        ]
        if download:
            command.append("--download")
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "Expected isolated LR-stage worker to exit successfully. "
                f"Provided entry={entry['entry_id']!r}, exit={completed.returncode}, "
                f"stderr={completed.stderr.strip()!r}."
            )
        try:
            return json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Expected isolated LR-stage worker to emit JSON. "
                f"Provided value: {completed.stdout!r}."
            ) from exc

    if workers == 1:
        executions = [one(entry) for entry in pending]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            executions = [future.result() for future in [pool.submit(one, entry) for entry in pending]]
    final = finalize_stage(study_dir, manifest_path)
    return {
        **final,
        "executor": "local",
        "entry_count": len(manifest["entries"]),
        "resumed_count": len(manifest["entries"]) - len(pending),
        "executions": executions,
    }


def submit_slurm_stage(
    *,
    study_dir: Path,
    manifest_path: Path,
    data_root: str | Path,
    device: str,
    profile_path: str | Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Submit one Slurm array and an afterany marker-last finalizer."""

    from experiments.submit_mnist_conv_slurm import (
        _export_arg,
        _resource_args,
        _submitted_job_id,
        load_profile,
    )

    _root, spec = load_study(study_dir)
    if _is_v5(spec) or _is_v6(spec):
        raise ValueError(
            "Expected mnist-conv-lr-study/v5 or v6 Jean Zay execution to use the "
            "dedicated measured-memory pack launcher and login-node collection; "
            "generic Slurm arrays and Slurm finalizers are prohibited. "
            f"Provided study: {spec.study_id!r}."
        )
    manifest = load_stage_manifest(manifest_path, study_dir=study_dir)
    profile = load_profile(profile_path)
    if _is_v7(spec):
        expected_worker = {
            "account": "fmu@v100",
            "partition": "gpu_p13",
            "qos": "qos_gpu-t3",
            "constraint": "v100",
            "time_limit": "20:00:00",
            "cpus_per_task": 16,
            "gpus_per_task": 1,
            "idrenv_project": "fmu",
        }
        observed_worker = {
            key: profile["worker"].get(key) for key in expected_worker
        }
        if observed_worker != expected_worker:
            raise ValueError(
                "Expected the frozen Conv3 v7 Jean Zay R3 one-process-per-V100 "
                f"profile {expected_worker!r}. Provided value: "
                f"{observed_worker!r}."
            )
    repo_root = Path(__file__).resolve().parents[2]
    log_dir = study_dir / "slurm" / manifest["stage_name"]
    worker_script = repo_root / "experiments" / "run_mnist_conv_lr_stage_slurm.sh"
    finalizer_script = repo_root / "experiments" / "finalize_mnist_conv_lr_stage_slurm.sh"
    for script in (worker_script, finalizer_script):
        if not script.is_file():
            raise FileNotFoundError(
                f"Expected LR-study Slurm wrapper to exist. Provided value: {script}."
            )
    exports = {
        "MNIST_CONV_REPO_ROOT": str(repo_root),
        "MNIST_CONV_PYTHON": sys.executable,
        "MNIST_CONV_LR_STUDY": str(study_dir),
        "MNIST_CONV_LR_MANIFEST": str(manifest_path),
        "MNIST_CONV_LR_STAGE": manifest["stage_name"],
        "MNIST_CONV_DATASET_ROOT": str(Path(data_root).expanduser().resolve()),
        "MNIST_CONV_DEVICE": device,
        "MNIST_CONV_MODULE": profile["worker"]["module"] or "",
        "MNIST_CONV_IDRENV_PROJECT": profile["worker"]["idrenv_project"],
    }
    if _is_v7(spec):
        exports.update(
            {
                "MNIST_CONV_R3_ALLOCATION": "AD010913993R3",
                "MNIST_CONV_EXPECTED_SLURM_ACCOUNT": "fmu@v100",
                "MNIST_CONV_EXPECTED_SLURM_PARTITION": "gpu_p13",
                "MNIST_CONV_EXPECTED_SLURM_QOS": "qos_gpu-t3",
                "MNIST_CONV_EXPECTED_SLURM_CONSTRAINT": "v100",
                "MNIST_CONV_THREADS_PER_PROCESS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
    safe_name = f"conv-lr-{manifest['stage_name']}"
    worker_resources = _resource_args(profile["worker"])
    if _is_v7(spec):
        worker_resources.append("--hint=nomultithread")
    worker = [
        "sbatch",
        "--parsable",
        f"--job-name={safe_name}",
        f"--array=0-{len(manifest['entries']) - 1}%{profile['concurrency']}",
        f"--chdir={repo_root}",
        f"--output={log_dir}/%x-%A_%a.out",
        f"--error={log_dir}/%x-%A_%a.err",
        *worker_resources,
        _export_arg(exports),
        str(worker_script),
    ]
    final_exports = {
        **exports,
        "MNIST_CONV_MODULE": profile["collector"]["module"] or "",
        "MNIST_CONV_IDRENV_PROJECT": profile["collector"]["idrenv_project"],
    }
    finalizer_template = [
        "sbatch",
        "--parsable",
        f"--job-name={safe_name}-finalize",
        f"--chdir={repo_root}",
        f"--output={log_dir}/%x-%j.out",
        f"--error={log_dir}/%x-%j.err",
        *_resource_args(profile["collector"]),
        _export_arg(final_exports),
        "--dependency=afterany:{worker_job_id}",
        str(finalizer_script),
    ]
    if dry_run:
        return {
            "submitted": False,
            "worker": worker,
            "finalizer": finalizer_template,
        }
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_result = subprocess.run(worker, check=True, text=True, capture_output=True)
    worker_job_id = _submitted_job_id(worker_result)
    finalizer = [item.format(worker_job_id=worker_job_id) for item in finalizer_template]
    finalizer_result = subprocess.run(finalizer, check=True, text=True, capture_output=True)
    return {
        "submitted": True,
        "worker_job_id": worker_job_id,
        "finalizer_job_id": _submitted_job_id(finalizer_result),
    }


def plan_only_result(spec: LRStudySpec, study_dir: Path, stage: str) -> dict[str, Any]:
    if _is_v7(spec):
        core = spec.data["rho_grid"]["core"]
        if stage == "audit":
            entry_ids = ["asset-audit"]
        elif stage == "probe":
            entry_ids = [row["row_id"] for row in spec.rows]
        elif stage == "preflight":
            entry_ids = ["ours-worst-cost"]
        elif stage == "core_candidates":
            entry_ids = [
                f"{row['row_id']}--core-c{conv_index:02d}-d{dense_index:02d}"
                for row in spec.rows
                for conv_index, _rho_conv in enumerate(core["rho_conv"])
                for dense_index, _rho_dense in enumerate(core["rho_dense"])
            ]
        elif stage == "select_core":
            entry_ids = ["selection"]
        elif stage == "extension_candidates":
            selection_path = (
                study_dir
                / _entry_dir("select_core", "selection")
                / "selection.json"
            )
            if not selection_path.is_file():
                return {
                    "planned": True,
                    "study_id": spec.study_id,
                    "study_dir": str(study_dir),
                    "stage": stage,
                    "entry_count": None,
                    "entry_count_range": [0, 21],
                    "conditional_zero_work_completion": True,
                    "entry_ids": [],
                }
            entry_ids = [
                entry["entry_id"]
                for entry in _v7_entries(spec, study_dir, stage)
            ]
        elif stage == "finalize":
            entry_ids = ["finalization"]
        else:
            raise ValueError(
                f"Expected v7 stage in {V7_STAGES!r}. Provided value: {stage!r}."
            )
        return {
            "planned": True,
            "study_id": spec.study_id,
            "study_dir": str(study_dir),
            "stage": stage,
            "entry_count": len(entry_ids),
            "entry_ids": entry_ids,
        }
    if _is_v6(spec):
        grid = spec.data["rho_grid"]
        if stage == "audit":
            entry_ids = ["reuse-audit"]
        elif stage == "probe":
            entry_ids = [row["row_id"] for row in spec.rows]
        elif stage == "baseline_candidates":
            entry_ids = [
                f"{grid['baseline_row_id']}--grid-c{conv_index:02d}-d{dense_index:02d}"
                for conv_index, _rho_conv in enumerate(grid["rho_conv"])
                for dense_index, _rho_dense in enumerate(grid["rho_dense"])
            ]
        elif stage == "select_baseline":
            entry_ids = ["selection"]
        elif stage == "confirmations":
            entry_ids = [
                f"{row_id}--confirmation"
                for row_id in grid["confirmation_row_ids"]
            ]
        elif stage == "finalize":
            entry_ids = ["finalization"]
        else:
            raise ValueError(
                f"Expected v6 stage in {V6_STAGES!r}. Provided value: {stage!r}."
            )
        return {
            "planned": True,
            "study_id": spec.study_id,
            "study_dir": str(study_dir),
            "stage": stage,
            "entry_count": len(entry_ids),
            "entry_ids": entry_ids,
        }
    if _is_v5(spec) and stage == "range":
        raise ValueError(
            "Expected v5 to skip the increasing-LR range stage. "
            "Provided stage: 'range'."
        )
    if stage == "audit":
        if not _is_v5(spec):
            raise ValueError(
                "Expected the anchor-audit stage only for mnist-conv-lr-study/v5. "
                f"Provided schema: {spec.data.get('schema_version')!r}."
            )
        entry_ids = ["anchor-audit"]
    elif stage in {"probe", "range"}:
        entry_ids = [row["row_id"] for row in spec.rows]
    elif stage == "candidates":
        if _is_v5(spec):
            entry_ids = [
                f"{row['row_id']}--{arm}--{role}"
                for architecture in _configured_architectures(spec)
                for arm in spec.data["target_policies"]["arm_order"]
                for role in spec.data["target_policies"]["candidate_roles"]
                for row in spec.rows
                if row["architecture"] == architecture
            ]
        else:
            entry_ids = [
                f"{row['row_id']}--{role}"
                for row in spec.rows
                for role in spec.data["candidate_training"]["candidate_roles"]
            ]
    else:
        entry_ids = ["selection"]
    return {
        "planned": True,
        "study_id": spec.study_id,
        "study_dir": str(study_dir),
        "stage": stage,
        "entry_count": len(entry_ids),
        "entry_ids": entry_ids,
    }


__all__ = [
    "STAGES",
    "V6_STAGES",
    "V7_STAGES",
    "create_study",
    "execute_local_stage",
    "execute_manifest_entry",
    "finalize_stage",
    "load_study",
    "plan_only_result",
    "publish_lr_stage_manifest",
    "submit_slurm_stage",
]
