"""Fail-closed post-run analysis for local IBM OM reference compensation.

The analyzer does not trust the scientific summary as a source of physical
facts.  It reloads each sampled population, reconstructs the two fixed-address
baseline plans and every persisted ``G = B + d`` mapping from the frozen
logical weights, and then checks exact counts and aggregates.  A study is
accepted only when an independent native replay is semantically identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.artifacts import atomic_write_json, content_hash, sha256_file
from experiments.mnist_relu_drn.analyze_ibm_om_four_reference_balance import (
    _accuracy_summary,
    _artifact_by_kind,
    _artifact_replay,
    _digest,
    _finite_numbers,
    _load_npz,
    _normalize_paths,
    _one_run,
    _population_map,
    _read_json,
    _registered_receipt,
    _replay_provenance,
    _same_recomputed_report,
    _scalar,
    _source_logical_weights,
    _tensor_sha256,
    _validated_metric,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _binding_slices,
    _native_to_unit,
)
from experiments.mnist_relu_drn.ibm_om_ideal_mapping_scheme_screen import (
    _quad_contrast,
)
from experiments.mnist_relu_drn.ibm_om_local_reference_compensation import (
    CONTROL_POLICY,
    MAPPING_SCHEMA,
    MAPPING_SCHEMA_VERSION,
    PLAN_SCHEMA,
    PLAN_SCHEMA_VERSION,
    POLICIES,
    TREATMENT_POLICY,
    _quad_stack,
    build_local_baseline_plan,
    build_matched_continuous_targets,
)
from experiments.mnist_relu_drn.ibm_om_local_reference_compensation_runtime import (
    SUMMARY_SCHEMA,
    SUMMARY_SCHEMA_VERSION,
    _compact_mapping,
)
from experiments.study_workflow import (
    StudyWorkflowError,
    load_study_plan,
    load_study_record,
)


ANALYSIS_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_local_reference_compensation_analysis"
)
ANALYSIS_SCHEMA_VERSION = 1
EXPECTED_ASSIGNMENTS = (87001, 87002, 87003)
EXPECTED_DEVELOPMENT = 86001
EXPECTED_STUDY_ID = (
    "mnist-ibm-om-local-reference-compensation-ideal-init-20260828-v1"
)
EXPECTED_EVIDENCE_CLASS = "model_based_aihwkit_preset"
EXPECTED_EXPERIMENT_ID = "ibm_om_local_reference_compensation.v1"
EXPECTED_METRIC_DEFINITION = (
    "ibm_om.local_reference_compensated_continuous_init.v1"
)
EXPECTED_ARM_IDS = ("main", "replay")
_PLAN_COMPONENT_FIELDS = (
    "raw_intrinsic_reference",
    "intrinsic_reference",
    "lower_bound",
    "upper_bound",
    "reachable_control_baseline",
    "baseline",
    "correction",
    "correction_from_reachable_control",
    "baseline_contrast",
    "feasible_minimum_contrast",
    "feasible_maximum_contrast",
    "projection_multiplier",
    "bound_active_quad",
    "at_lower",
    "at_upper",
)
_MAPPING_FIELDS = (
    "baseline",
    "offset",
    "conductance",
    "baseline_contrast",
    "offset_contrast",
    "realized_contrast",
    "loading",
    "active_mask",
    "common_headroom",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run", type=Path, required=True)
    parser.add_argument("--replay-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _validate_study_context(
    run_dir: Path,
    manifest: Mapping[str, Any],
    *,
    expected_arm_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Require an exact prepared-study arm and tracked source-plan identity."""

    if expected_arm_id not in EXPECTED_ARM_IDS:
        raise ValueError("Expected the canonical main or replay arm ID.")
    if run_dir.parent.name != expected_arm_id or run_dir.parent.parent.name != "runs":
        raise ValueError(
            f"Expected the {expected_arm_id!r} run below runs/{expected_arm_id}/."
        )
    study_root = run_dir.parents[2]
    study_path = study_root / "study.json"
    try:
        study = load_study_record(study_root)
    except StudyWorkflowError as error:
        raise ValueError("Prepared study contract/schema mismatch.") from error
    if (
        not isinstance(study, Mapping)
        or study.get("schema") != "ebl.study"
        or study.get("schema_version") != 1
        or study.get("study_id") != EXPECTED_STUDY_ID
        or study.get("evidence_class") != EXPECTED_EVIDENCE_CLASS
    ):
        raise ValueError("Prepared study contract/schema mismatch.")
    source_plan = study.get("source_plan")
    if not isinstance(source_plan, Mapping):
        raise ValueError("Expected prepared source-plan provenance.")
    source_plan_sha = _digest(
        source_plan.get("sha256"), label="study source-plan hash"
    )
    source_plan_path = Path(str(source_plan.get("path"))).expanduser().resolve()
    if not source_plan_path.is_file() or sha256_file(source_plan_path) != source_plan_sha:
        raise ValueError("Tracked source-plan bytes do not match study.json.")
    try:
        declared_plan = load_study_plan(source_plan_path)
    except StudyWorkflowError as error:
        raise ValueError("Tracked source plan is not a valid study contract.") from error
    if any(study.get(name) != value for name, value in declared_plan.items()):
        raise ValueError("Prepared study does not exactly materialize its source plan.")
    arms_raw = study.get("arms")
    if not isinstance(arms_raw, list) or len(arms_raw) != 2:
        raise ValueError("Expected exactly the declared main and replay study arms.")
    arms = {
        item.get("arm_id"): item for item in arms_raw if isinstance(item, Mapping)
    }
    if set(arms) != set(EXPECTED_ARM_IDS):
        raise ValueError("Prepared study arm coverage mismatch.")
    for arm_id in EXPECTED_ARM_IDS:
        arm = arms[arm_id]
        configs = arm.get("configs")
        if (
            arm.get("experiment_id") != EXPECTED_EXPERIMENT_ID
            or arm.get("mode") != "validate"
            or not isinstance(configs, list)
            or len(configs) != 1
            or not isinstance(configs[0], Mapping)
        ):
            raise ValueError("Prepared study arm contract mismatch.")
        _digest(configs[0].get("sha256"), label=f"{arm_id} source-config hash")
    if arms["main"]["configs"][0]["sha256"] != arms["replay"]["configs"][0]["sha256"]:
        raise ValueError("Main/replay source-config provenance differs.")

    expected_context = {
        "study_id": EXPECTED_STUDY_ID,
        "arm_id": expected_arm_id,
        "evidence_class": EXPECTED_EVIDENCE_CLASS,
        "study_sha256": sha256_file(study_path),
        "source_plan_sha256": source_plan_sha,
        "source_config_sha256": arms[expected_arm_id]["configs"][0]["sha256"],
    }
    context = manifest.get("study")
    if not isinstance(context, Mapping) or dict(context) != expected_context:
        raise ValueError("Manifest study provenance does not exactly match study.json.")
    return dict(context), study


def _validate_bundle(
    run_dir: Path,
    *,
    expected_arm_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    manifest = _read_json(run_dir / "manifest.json")
    status = _read_json(run_dir / "status.json")
    result = _read_json(run_dir / "result.json")
    config = _read_json(run_dir / "config.resolved.json")
    if (
        manifest.get("schema") != "ebl.run"
        or manifest.get("schema_version") != 1
        or manifest.get("run_id") != run_dir.name
        or manifest.get("experiment_id") != EXPECTED_EXPERIMENT_ID
        or status.get("schema") != "ebl.run"
        or status.get("schema_version") != 1
        or status.get("run_id") != manifest.get("run_id")
        or status.get("status") != "complete"
        or result.get("schema") != "ebl.run"
        or result.get("schema_version") != 1
        or result.get("status") != "complete"
        or result.get("experiment_id") != manifest.get("experiment_id")
        or result.get("run_id") != manifest.get("run_id")
    ):
        raise ValueError("Expected one complete native local-compensation run bundle.")
    study_context, study = _validate_study_context(
        run_dir, manifest, expected_arm_id=expected_arm_id
    )
    if manifest.get("config", {}).get("sha256") != content_hash(config):
        raise ValueError("Resolved config hash mismatch.")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Expected registered native artifacts.")
    seen: set[str] = set()
    for record in artifacts:
        if not isinstance(record, Mapping):
            raise ValueError("Expected artifact records to be objects.")
        relative = record.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise ValueError("Expected unique relative artifact paths.")
        seen.add(relative)
        source = (run_dir / relative).resolve()
        try:
            source.relative_to(run_dir.resolve())
        except ValueError as error:
            raise ValueError("Artifact escaped the native run directory.") from error
        if (
            not source.is_file()
            or sha256_file(source) != record.get("sha256")
            or source.stat().st_size != record.get("size_bytes")
        ):
            raise ValueError(f"Registered artifact mismatch: {relative}.")
    summaries = [
        record for record in artifacts if record.get("kind") == "scientific_summary"
    ]
    if len(summaries) != 1:
        raise ValueError("Expected exactly one scientific summary artifact.")
    summary = _read_json(run_dir / summaries[0]["path"])
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("schema_version") != SUMMARY_SCHEMA_VERSION
        or summary.get("status") != "complete"
    ):
        raise ValueError("Scientific summary schema/status mismatch.")
    return {
        "manifest": manifest,
        "status": status,
        "result": result,
        "config": config,
        "artifacts": artifacts,
        "study_context": study_context,
        "study": study,
    }, summary


def _require_artifact_contract(by_kind: Mapping[str, Sequence[Path]]) -> None:
    expected = {
        "ibm_om_population": 4,
        "ibm_om_population_receipt": 4,
        "ibm_om_local_baseline_plan": 8,
        "ibm_om_local_baseline_plan_receipt": 8,
        "ibm_om_full_g_mapping": 14,
        "ibm_om_full_g_mapping_receipt": 14,
        "scientific_summary": 1,
    }
    observed = {name: len(paths) for name, paths in by_kind.items()}
    if observed != expected:
        raise ValueError(
            f"Registered artifact kind/count contract mismatch: {observed!r}."
        )


def _plan_summary_index(
    summary: Mapping[str, Any],
) -> Mapping[tuple[int, str], Mapping[str, Any]]:
    result: dict[tuple[int, str], Mapping[str, Any]] = {}
    development = summary.get("development_baseline_plans")
    if not isinstance(development, Mapping):
        raise ValueError("Expected development baseline-plan reports.")
    for policy in POLICIES:
        item = development.get(policy)
        if not isinstance(item, Mapping):
            raise ValueError("Expected one development plan per baseline policy.")
        result[(EXPECTED_DEVELOPMENT, policy)] = item
    heldout = summary.get("heldout")
    if not isinstance(heldout, list):
        raise ValueError("Expected held-out baseline-plan reports.")
    for block in heldout:
        seed = block.get("assignment_seed")
        plans = block.get("baseline_plans")
        if not isinstance(plans, Mapping):
            raise ValueError("Expected held-out baseline-plan reports.")
        for policy in POLICIES:
            item = plans.get(policy)
            if not isinstance(item, Mapping) or (seed, policy) in result:
                raise ValueError("Duplicate or absent baseline-plan report.")
            result[(seed, policy)] = item
    expected = {
        (seed, policy)
        for seed in (EXPECTED_DEVELOPMENT, *EXPECTED_ASSIGNMENTS)
        for policy in POLICIES
    }
    if set(result) != expected:
        raise ValueError("Baseline-plan summary coverage mismatch.")
    return result


def _validate_plans(
    paths: Sequence[Path],
    receipt_paths: Sequence[Path],
    populations: Mapping[int, Any],
    summary: Mapping[str, Any],
) -> Mapping[tuple[int, str], Any]:
    """Reload and independently reconstruct every fixed-identity baseline plan."""

    if len(paths) != 8 or len(receipt_paths) != 8:
        raise ValueError("Expected eight baseline plans and eight receipts.")
    summary_index = _plan_summary_index(summary)
    expected_fields = {
        "schema",
        "schema_version",
        "policy",
        "assignment_seed",
        "population_fingerprint",
        "identity_order",
        "baseline_unit",
        *(
            f"layer_{layer}_{name}"
            for layer in range(2)
            for name in _PLAN_COMPONENT_FIELDS
        ),
    }
    result: dict[tuple[int, str], Any] = {}
    for path in paths:
        payload = _load_npz(path)
        if set(payload) != expected_fields:
            raise ValueError("Local baseline-plan NPZ field mismatch.")
        if (
            str(_scalar(payload, "schema")) != PLAN_SCHEMA
            or int(_scalar(payload, "schema_version")) != PLAN_SCHEMA_VERSION
            or payload["schema_version"].dtype != np.int64
            or payload["assignment_seed"].dtype != np.int64
        ):
            raise ValueError("Local baseline-plan scalar schema/dtype mismatch.")
        seed = int(_scalar(payload, "assignment_seed"))
        policy = str(_scalar(payload, "policy"))
        if seed not in populations or policy not in POLICIES:
            raise ValueError("Unexpected baseline-plan assignment or policy.")
        population = populations[seed][0]
        if str(_scalar(payload, "population_fingerprint")) != population.fingerprint:
            raise ValueError("Baseline-plan population fingerprint mismatch.")
        expected = build_local_baseline_plan(population, policy=policy)
        identity_order = payload["identity_order"]
        if (
            identity_order.dtype != np.int64
            or identity_order.shape != (population.size,)
            or not np.array_equal(identity_order, np.arange(population.size))
        ):
            raise ValueError("Baseline plan changed sampled identity order.")
        baseline_unit = torch.from_numpy(payload["baseline_unit"])
        if baseline_unit.dtype != torch.float64 or not torch.equal(
            baseline_unit, expected.baseline_unit
        ):
            raise ValueError("Stored baseline tensor does not reconstruct.")
        for layer, (shape, component) in enumerate(
            zip(population.binding_shapes, expected.components)
        ):
            for name in _PLAN_COMPONENT_FIELDS:
                stored = torch.from_numpy(payload[f"layer_{layer}_{name}"])
                if not torch.equal(stored, component[name]):
                    raise ValueError(
                        f"Stored baseline component {name!r} does not reconstruct."
                    )
                if name in {"bound_active_quad", "at_lower", "at_upper"}:
                    if stored.dtype != torch.bool:
                        raise ValueError("Expected boolean baseline-plan masks.")
                elif stored.dtype != torch.float64:
                    raise ValueError("Expected float64 baseline-plan tensors.")
            if payload[f"layer_{layer}_baseline"].shape != tuple(shape):
                raise ValueError("Baseline-plan physical layout mismatch.")
        receipt_path = _registered_receipt(path, receipt_paths)
        receipt = _read_json(receipt_path)
        metadata = {
            "artifact": path.name,
            "artifact_sha256": sha256_file(path),
            "binding_keys": list(population.binding_keys),
            "binding_shapes": [list(shape) for shape in population.binding_shapes],
        }
        if (
            set(receipt) != set(expected.report) | set(metadata)
            or any(receipt.get(name) != value for name, value in metadata.items())
            or not _same_recomputed_report(
                {name: receipt[name] for name in expected.report}, expected.report
            )
        ):
            raise ValueError("Baseline-plan receipt does not exactly reconstruct.")
        key = (seed, policy)
        if key in result:
            raise ValueError("Duplicate baseline-plan assignment/policy.")
        summary_item = summary_index.get(key)
        artifact_link = (
            summary_item.get("artifact")
            if isinstance(summary_item, Mapping)
            else None
        )
        if (
            not isinstance(summary_item, Mapping)
            or not _same_recomputed_report(summary_item.get("report"), expected.report)
            or not isinstance(artifact_link, Mapping)
            or Path(str(artifact_link.get("path"))).name != path.name
            or artifact_link.get("sha256") != sha256_file(path)
            or Path(str(artifact_link.get("receipt"))).name != receipt_path.name
            or artifact_link.get("receipt_sha256") != sha256_file(receipt_path)
            or artifact_link.get("policy") != policy
            or artifact_link.get("identity_order_sha256")
            != expected.report["destination_to_source_sha256"]
            or artifact_link.get("baseline_sha256")
            != expected.report["baseline_sha256"]
        ):
            raise ValueError("Summary baseline plan is not linked to its artifacts.")
        result[key] = expected
    if set(result) != set(summary_index) or len(result) != 8:
        raise ValueError("Baseline-plan assignment/policy coverage mismatch.")
    return result


def _selected_candidate_index(candidates: Sequence[Mapping[str, Any]]) -> int:
    if len(candidates) != 16:
        raise ValueError("Expected exactly sixteen development candidates.")
    return min(
        range(len(candidates)),
        key=lambda index: (
            -float(candidates[index]["calibration"]["student_accuracy"]),
            float(candidates[index]["calibration"]["calibrated_kl"]),
            tuple(float(value) for value in candidates[index]["scale_fractions"]),
        ),
    )


def _validate_development(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    populations: Mapping[int, Any],
    plans: Mapping[tuple[int, str], Any],
    logical_weights: Sequence[torch.Tensor],
) -> Mapping[str, Mapping[str, Any]]:
    """Rebuild all scale-pair mappings and recompute both policy selections."""

    development = summary.get("development")
    grid = config.get("student", {}).get("mapping", {}).get("scale_fractions")
    if not isinstance(development, Mapping) or not isinstance(grid, list):
        raise ValueError("Expected development calibration and resolved scale grid.")
    expected_pairs = [
        (float(left), float(right)) for left in grid for right in grid
    ]
    if len(expected_pairs) != 16 or len(set(expected_pairs)) != 16:
        raise ValueError("Expected the frozen four-by-four scale grid.")
    population = populations[EXPECTED_DEVELOPMENT][0]
    policy_plans = {
        policy: plans[(EXPECTED_DEVELOPMENT, policy)] for policy in POLICIES
    }
    student = config.get("student", {})
    model = student.get("model", {}) if isinstance(student, Mapping) else {}
    conductance_min = float(model.get("conductance_min"))
    conductance_max = float(model.get("conductance_max"))

    reconstructed: dict[tuple[str, tuple[float, float]], Mapping[str, Any]] = {}
    result: dict[str, Mapping[str, Any]] = {}
    for policy in POLICIES:
        record = development.get(policy)
        if not isinstance(record, Mapping) or record.get("baseline_policy") != policy:
            raise ValueError("Development baseline-policy report mismatch.")
        candidates = record.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 16:
            raise ValueError("Expected exactly sixteen development candidates per policy.")
        observed_pairs: list[tuple[float, float]] = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping) or not _finite_numbers(candidate):
                raise ValueError("Expected finite development candidate metrics.")
            pair_raw = candidate.get("scale_fractions")
            calibration = candidate.get("calibration")
            target_hashes = candidate.get("target_hashes")
            offset_hashes = candidate.get("offset_hashes")
            contrast_rms = candidate.get("baseline_contrast_rms")
            if (
                not isinstance(pair_raw, list)
                or len(pair_raw) != 2
                or not isinstance(calibration, Mapping)
                or float(calibration.get("gain", 0.0)) <= 0.0
                or not 0.0 <= float(calibration.get("student_accuracy", -1.0)) <= 1.0
                or float(calibration.get("calibrated_kl", -1.0)) < 0.0
                or calibration.get("gain_source")
                != "per_baseline_continuous_development_fit"
                or not isinstance(target_hashes, list)
                or len(target_hashes) != 2
                or not isinstance(offset_hashes, list)
                or len(offset_hashes) != 2
                or not isinstance(contrast_rms, list)
                or len(contrast_rms) != 2
            ):
                raise ValueError("Development candidate calibration/hash contract mismatch.")
            for digest in (*target_hashes, *offset_hashes):
                _digest(digest, label="development mapping hash")
            pair = (float(pair_raw[0]), float(pair_raw[1]))
            observed_pairs.append(pair)
            if pair != expected_pairs[index]:
                raise ValueError("Development candidate scale-pair order drifted.")
            targets, mapping, _components = build_matched_continuous_targets(
                logical_weights,
                population,
                policy_plans,
                policy=policy,
                scale_fractions=pair,
                conductance_min=conductance_min,
                conductance_max=conductance_max,
            )
            expected_target_hashes = [_tensor_sha256(value) for value in targets]
            expected_offset_hashes = [
                layer["hashes"]["offset"] for layer in mapping["layers"]
            ]
            expected_contrast_rms = [
                layer["baseline_contrast"]["rms"] for layer in mapping["layers"]
            ]
            if (
                target_hashes != expected_target_hashes
                or offset_hashes != expected_offset_hashes
                or not _same_recomputed_report(contrast_rms, expected_contrast_rms)
            ):
                raise ValueError("Development candidate mapping does not reconstruct.")
            reconstructed[(policy, pair)] = mapping
        if observed_pairs != expected_pairs:
            raise ValueError("Development candidates are not the exact scale grid.")
        selected_index = _selected_candidate_index(candidates)
        if (
            record.get("selected_index") != selected_index
            or record.get("selected") != candidates[selected_index]
            or record.get("selection_domain")
            != "development_assignment_86001_calibration_subset"
            or record.get("selection_metric")
            != "mapped_accuracy_then_calibrated_kl_then_scale_pair"
        ):
            raise ValueError("Stored development selection does not recompute exactly.")
        selected_mapping = record.get("selected_mapping")
        expected_mapping = reconstructed[(policy, expected_pairs[selected_index])]
        if (
            not isinstance(selected_mapping, Mapping)
            or selected_mapping != _compact_mapping(expected_mapping)
        ):
            raise ValueError("Selected development mapping does not reconstruct.")
        result[policy] = record

    # The common-headroom construction must make d bit-identical for every
    # scale pair, not only for whichever pair each policy selected.
    for pair in expected_pairs:
        left = reconstructed[(CONTROL_POLICY, pair)]
        right = reconstructed[(TREATMENT_POLICY, pair)]
        left_hashes = tuple(layer["hashes"]["offset"] for layer in left["layers"])
        right_hashes = tuple(layer["hashes"]["offset"] for layer in right["layers"])
        if left_hashes != right_hashes:
            raise ValueError("Development policy arms do not use identical d tensors.")
    return result


def _summary_mapping_index(
    summary: Mapping[str, Any],
    development: Mapping[str, Mapping[str, Any]],
) -> Mapping[tuple[int, str, str], Mapping[str, Any]]:
    result: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for policy in POLICIES:
        selected = development[policy]["selected"]
        result[(EXPECTED_DEVELOPMENT, policy, policy)] = {
            "scale_fractions": selected["scale_fractions"],
            "fixed_logit_gain": selected["calibration"]["gain"],
            "mapping": development[policy]["selected_mapping"],
            "artifact": development[policy]["selected_mapping_artifact"],
        }
    heldout = summary.get("heldout")
    if not isinstance(heldout, list):
        raise ValueError("Expected held-out mapping reports.")
    for block in heldout:
        seed = block.get("assignment_seed")
        arms = block.get("arms")
        if not isinstance(arms, list):
            raise ValueError("Expected held-out mapping arms.")
        for arm in arms:
            if not isinstance(arm, Mapping):
                raise ValueError("Expected held-out arm objects.")
            baseline = arm.get("baseline_policy")
            calibration = arm.get("calibration_policy")
            key = (seed, baseline, calibration)
            if key in result or baseline not in POLICIES or calibration not in POLICIES:
                raise ValueError("Duplicate or unknown held-out mapping arm.")
            selected = development[calibration]["selected"]
            if (
                arm.get("assignment_seed") != seed
                or arm.get("scale_fractions") != selected["scale_fractions"]
                or arm.get("fixed_logit_gain") != selected["calibration"]["gain"]
                or arm.get("calibration_source")
                != "frozen_development_assignment_86001"
            ):
                raise ValueError("Held-out mapping refit or calibration drift detected.")
            result[key] = {
                "scale_fractions": arm["scale_fractions"],
                "fixed_logit_gain": arm["fixed_logit_gain"],
                "mapping": arm.get("mapping"),
                "artifact": arm.get("mapping_artifact"),
            }
    return result


def _validate_mappings(
    paths: Sequence[Path],
    receipt_paths: Sequence[Path],
    populations: Mapping[int, Any],
    plans: Mapping[tuple[int, str], Any],
    logical_weights: Sequence[torch.Tensor],
    summary_index: Mapping[tuple[int, str, str], Mapping[str, Any]],
    *,
    conductance_min: float,
    conductance_max: float,
) -> Mapping[tuple[int, str, str], Mapping[str, Any]]:
    """Reconstruct all selected physical B, d, G, contrast, and loading tensors."""

    if len(paths) != 14 or len(receipt_paths) != 14:
        raise ValueError("Expected fourteen mappings and fourteen receipts.")
    expected_fields = {
        "schema",
        "schema_version",
        "assignment_seed",
        "baseline_policy",
        "calibration_policy",
        *(
            f"layer_{layer}_{name}"
            for layer in range(2)
            for name in _MAPPING_FIELDS
        ),
    }
    result: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for path in paths:
        payload = _load_npz(path)
        if set(payload) != expected_fields:
            raise ValueError("Continuous mapping NPZ field mismatch.")
        if (
            str(_scalar(payload, "schema")) != MAPPING_SCHEMA
            or int(_scalar(payload, "schema_version")) != MAPPING_SCHEMA_VERSION
            or payload["schema_version"].dtype != np.int64
            or payload["assignment_seed"].dtype != np.int64
        ):
            raise ValueError("Continuous mapping scalar schema/dtype mismatch.")
        seed = int(_scalar(payload, "assignment_seed"))
        baseline_policy = str(_scalar(payload, "baseline_policy"))
        calibration_policy = str(_scalar(payload, "calibration_policy"))
        key = (seed, baseline_policy, calibration_policy)
        if (
            seed not in populations
            or baseline_policy not in POLICIES
            or calibration_policy not in POLICIES
            or key not in summary_index
        ):
            raise ValueError("Unexpected mapping assignment or policy.")
        summary_item = summary_index[key]
        pair_raw = summary_item.get("scale_fractions")
        if not isinstance(pair_raw, list) or len(pair_raw) != 2:
            raise ValueError("Expected two frozen selected scale fractions.")
        pair = (float(pair_raw[0]), float(pair_raw[1]))
        population = populations[seed][0]
        policy_plans = {policy: plans[(seed, policy)] for policy in POLICIES}
        expected_targets, expected_mapping, expected_components = (
            build_matched_continuous_targets(
                logical_weights,
                population,
                policy_plans,
                policy=baseline_policy,
                scale_fractions=pair,
                conductance_min=conductance_min,
                conductance_max=conductance_max,
            )
        )
        receipt_path = _registered_receipt(path, receipt_paths)
        receipt = _read_json(receipt_path)
        metadata = {
            "schema": MAPPING_SCHEMA,
            "schema_version": MAPPING_SCHEMA_VERSION,
            "assignment_seed": seed,
            "baseline_policy": baseline_policy,
            "calibration_policy": calibration_policy,
            "artifact": path.name,
            "artifact_sha256": sha256_file(path),
        }
        if (
            set(receipt) != set(metadata) | {"mapping"}
            or any(receipt.get(name) != value for name, value in metadata.items())
            or not _same_recomputed_report(receipt.get("mapping"), expected_mapping)
        ):
            raise ValueError("Mapping receipt does not exactly reconstruct.")
        if summary_item.get("mapping") != _compact_mapping(receipt["mapping"]):
            raise ValueError("Summary mapping report does not reconstruct.")
        link = summary_item.get("artifact")
        if (
            not isinstance(link, Mapping)
            or Path(str(link.get("path"))).name != path.name
            or link.get("sha256") != sha256_file(path)
            or Path(str(link.get("receipt"))).name != receipt_path.name
            or link.get("receipt_sha256") != sha256_file(receipt_path)
            or link.get("assignment_seed") != seed
            or link.get("baseline_policy") != baseline_policy
            or link.get("calibration_policy") != calibration_policy
        ):
            raise ValueError("Summary mapping artifact/receipt link mismatch.")
        lower_unit = _native_to_unit(
            population.min_bound.detach().cpu().to(torch.float64)
        )
        upper_unit = _native_to_unit(
            population.max_bound.detach().cpu().to(torch.float64)
        )
        slices = _binding_slices(population)
        span = conductance_max - conductance_min
        for layer, (binding_key, shape, layout) in enumerate(
            zip(population.binding_keys, population.binding_shapes, ("halves", "paired"))
        ):
            values = {
                name: torch.from_numpy(payload[f"layer_{layer}_{name}"])
                for name in _MAPPING_FIELDS
            }
            for name in _MAPPING_FIELDS:
                if not torch.equal(values[name], expected_components[layer][name]):
                    raise ValueError(
                        f"Stored mapping tensor {name!r} does not reconstruct."
                    )
            if not torch.equal(values["conductance"], expected_targets[layer]):
                raise ValueError("Stored G is not the reconstructed circuit target.")
            for name in ("baseline", "offset", "conductance", "common_headroom"):
                if values[name].dtype != torch.float32 or values[name].shape != tuple(shape):
                    raise ValueError("Mapping branch tensor dtype/shape mismatch.")
            if (
                values["active_mask"].dtype != torch.bool
                or values["active_mask"].shape != tuple(shape)
            ):
                raise ValueError("Mapping active-mask dtype/shape mismatch.")
            for name in (
                "baseline_contrast",
                "offset_contrast",
                "realized_contrast",
                "loading",
            ):
                if values[name].dtype != torch.float32:
                    raise ValueError("Mapping derived tensor dtype mismatch.")
            if not all(
                bool(torch.isfinite(value).all())
                for name, value in values.items()
                if name != "active_mask"
            ):
                raise ValueError("Nonfinite mapping tensor.")
            if not torch.equal(values["conductance"], values["baseline"] + values["offset"]):
                raise ValueError("Stored physical G is not bit-exact B+d.")
            if bool(torch.any(values["conductance"] < 0.0)) or bool(
                torch.any(values["offset"] < 0.0)
            ):
                raise ValueError("Expected nonnegative physical G and d.")
            if bool(torch.any(values["offset"][~values["active_mask"]] != 0.0)):
                raise ValueError("Inactive rail branches must have zero offset.")
            expected_baseline_contrast = _quad_contrast(
                values["baseline"], layout=layout
            ) / 2.0
            expected_offset_contrast = _quad_contrast(
                values["offset"], layout=layout
            ) / 2.0
            expected_realized_contrast = _quad_contrast(
                values["conductance"], layout=layout
            ) / 2.0
            expected_loading = _quad_stack(
                values["conductance"], layout=layout
            ).sum(dim=-1)
            if not (
                torch.equal(values["baseline_contrast"], expected_baseline_contrast)
                and torch.equal(values["offset_contrast"], expected_offset_contrast)
                and torch.equal(values["realized_contrast"], expected_realized_contrast)
                and torch.equal(values["loading"], expected_loading)
            ):
                raise ValueError("Stored full-G contrast/loading does not reproduce.")
            binding_slice = slices[binding_key]
            physical_lower = conductance_min + span * lower_unit[binding_slice].reshape(shape)
            physical_upper = conductance_min + span * upper_unit[binding_slice].reshape(shape)
            if bool(
                torch.any(
                    (values["conductance"].to(torch.float64) < physical_lower)
                    | (values["conductance"].to(torch.float64) > physical_upper)
                )
            ):
                raise ValueError("Mapped full conductance lies outside sampled bounds.")
            hashes = expected_mapping["layers"][layer]["hashes"]
            hash_values = {
                "baseline": values["baseline"],
                "offset": values["offset"],
                "conductance": values["conductance"],
                "baseline_contrast": values["baseline_contrast"],
                "realized_contrast": values["realized_contrast"],
                "loading": values["loading"],
                "common_headroom": values["common_headroom"],
            }
            if any(
                _tensor_sha256(value) != hashes[name]
                for name, value in hash_values.items()
            ):
                raise ValueError("Mapping tensor hash does not match its report.")
        if key in result:
            raise ValueError("Duplicate mapping assignment/policy key.")
        result[key] = {
            "path": path,
            "receipt_path": receipt_path,
            "mapping": expected_mapping,
        }

    expected_keys = {
        (EXPECTED_DEVELOPMENT, policy, policy) for policy in POLICIES
    } | {
        (seed, baseline, calibration)
        for seed in EXPECTED_ASSIGNMENTS
        for baseline in POLICIES
        for calibration in POLICIES
    }
    if set(result) != expected_keys or set(summary_index) != expected_keys:
        raise ValueError("Mapping assignment/policy coverage mismatch.")
    for seed in EXPECTED_ASSIGNMENTS:
        for calibration in POLICIES:
            left = result[(seed, CONTROL_POLICY, calibration)]["mapping"]
            right = result[(seed, TREATMENT_POLICY, calibration)]["mapping"]
            left_hashes = tuple(layer["hashes"]["offset"] for layer in left["layers"])
            right_hashes = tuple(layer["hashes"]["offset"] for layer in right["layers"])
            if left_hashes != right_hashes:
                raise ValueError("Held-out matched arms do not use identical d tensors.")
    return result


def _arm_index(
    summary: Mapping[str, Any],
) -> Mapping[tuple[int, str, str], Mapping[str, Any]]:
    heldout = summary.get("heldout")
    if not isinstance(heldout, list) or tuple(
        block.get("assignment_seed") for block in heldout
    ) != EXPECTED_ASSIGNMENTS:
        raise ValueError("Held-out assignment coverage/order mismatch.")
    result: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for block in heldout:
        arms = block.get("arms")
        if not isinstance(arms, list) or len(arms) != 4:
            raise ValueError("Expected four baseline/calibration arms per assignment.")
        for arm in arms:
            if not isinstance(arm, Mapping):
                raise ValueError("Expected held-out arm objects.")
            key = (
                block["assignment_seed"],
                arm.get("baseline_policy"),
                arm.get("calibration_policy"),
            )
            if key in result or key[1] not in POLICIES or key[2] not in POLICIES:
                raise ValueError("Held-out arm key coverage mismatch.")
            _validated_metric(arm.get("ideal_bounded_continuous_test"))
            result[key] = arm
    expected = {
        (seed, baseline, calibration)
        for seed in EXPECTED_ASSIGNMENTS
        for baseline in POLICIES
        for calibration in POLICIES
    }
    if set(result) != expected:
        raise ValueError("Held-out arm key coverage mismatch.")
    return result


def _metrics(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recompute exact test counts, both comparison families, and both gates."""

    if (
        summary.get("metric_definition_id") != EXPECTED_METRIC_DEFINITION
        or summary.get("exclusions")
        != {
            "optimizer_updates": 0,
            "quantization_enabled": False,
            "program_verify_enabled": False,
            "hwa_enabled": False,
            "deployment_write_noise": 0.0,
            "inference_read_noise": 0.0,
            "retention_or_drift": False,
            "identity_reassignment": False,
        }
    ):
        raise ValueError("Scientific definition/exclusion contract mismatch.")
    by_key = _arm_index(summary)
    recomputed: dict[str, Any] = {
        "scheme_optimized": {},
        "fixed_calibration": {},
    }
    for policy in POLICIES:
        per_assignment = []
        for seed in EXPECTED_ASSIGNMENTS:
            metric = by_key[(seed, policy, policy)][
                "ideal_bounded_continuous_test"
            ]
            per_assignment.append(
                {
                    "assignment_seed": seed,
                    "correct": metric["student_correct"],
                    "examples": metric["examples"],
                    "accuracy": metric["student_accuracy"],
                    "prediction_sha256": metric["prediction_sha256"],
                    "teacher_agreement": metric["teacher_agreement"],
                    "kl_teacher_student": metric["kl_teacher_student"],
                }
            )
        values = [item["accuracy"] for item in per_assignment]
        recomputed["scheme_optimized"][policy] = {
            "per_assignment": per_assignment,
            "values": values,
            **_accuracy_summary(values),
        }

    for calibration in POLICIES:
        per_assignment = []
        for seed in EXPECTED_ASSIGNMENTS:
            control_arm = by_key[(seed, CONTROL_POLICY, calibration)]
            treatment_arm = by_key[(seed, TREATMENT_POLICY, calibration)]
            control = control_arm["ideal_bounded_continuous_test"]
            treatment = treatment_arm["ideal_bounded_continuous_test"]
            paired = next(
                block["paired_baseline_effect"][calibration]
                for block in summary["heldout"]
                if block["assignment_seed"] == seed
            )
            examples = int(control["examples"])
            correct_delta = int(treatment["student_correct"]) - int(
                control["student_correct"]
            )
            delta = correct_delta / examples
            offset_hashes = control_arm.get("mapping", {}).get("offset_hashes")
            if (
                not isinstance(paired, Mapping)
                or set(paired)
                != {
                    "treatment_minus_control_correct",
                    "examples",
                    "treatment_minus_control_accuracy",
                    "prediction_flip_count",
                    "prediction_flip_fraction",
                    "identical_offset_hashes",
                }
                or treatment["examples"] != examples
                or paired.get("treatment_minus_control_correct") != correct_delta
                or paired.get("examples") != examples
                or paired.get("treatment_minus_control_accuracy") != delta
                or isinstance(paired.get("prediction_flip_count"), bool)
                or not isinstance(paired.get("prediction_flip_count"), int)
                or not 0 <= paired["prediction_flip_count"] <= examples
                or paired.get("prediction_flip_fraction")
                != paired["prediction_flip_count"] / examples
                or not isinstance(offset_hashes, list)
                or treatment_arm.get("mapping", {}).get("offset_hashes")
                != offset_hashes
                or paired.get("identical_offset_hashes") != offset_hashes
            ):
                raise ValueError("Stored paired fixed-calibration effect mismatch.")
            for digest in offset_hashes:
                _digest(digest, label="paired offset hash")
            per_assignment.append(
                {
                    "assignment_seed": seed,
                    "control_correct": control["student_correct"],
                    "treatment_correct": treatment["student_correct"],
                    "examples": examples,
                    "control_accuracy": control["student_accuracy"],
                    "treatment_accuracy": treatment["student_accuracy"],
                    "treatment_minus_control_correct": correct_delta,
                    "treatment_minus_control_accuracy": delta,
                    "prediction_flip_count": paired["prediction_flip_count"],
                    "prediction_flip_fraction": paired["prediction_flip_fraction"],
                    "identical_offset_hashes": offset_hashes,
                }
            )
        deltas = [item["treatment_minus_control_accuracy"] for item in per_assignment]
        recomputed["fixed_calibration"][calibration] = {
            "per_assignment": per_assignment,
            "values": deltas,
            **_accuracy_summary(deltas),
        }

    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError("Expected scientific aggregate block.")
    for policy in POLICIES:
        expected_accuracy = {
            name: recomputed["scheme_optimized"][policy][name]
            for name in ("mean", "minimum", "maximum", "sample_standard_deviation")
        }
        if aggregate.get("scheme_optimized", {}).get(policy) != expected_accuracy:
            raise ValueError("Stored scheme-optimized aggregate mismatch.")
    for calibration in POLICIES:
        expected_effect = {
            name: recomputed["fixed_calibration"][calibration][name]
            for name in ("mean", "minimum", "maximum", "sample_standard_deviation")
        }
        stored_effects = [
            next(
                block["paired_baseline_effect"][calibration]
                for block in summary["heldout"]
                if block["assignment_seed"] == seed
            )
            for seed in EXPECTED_ASSIGNMENTS
        ]
        if aggregate.get("fixed_calibration", {}).get(calibration) != {
            "treatment_minus_control_accuracy": expected_effect,
            "per_assignment": stored_effects,
        }:
            raise ValueError("Stored fixed-calibration aggregate mismatch.")

    primary = recomputed["fixed_calibration"][CONTROL_POLICY]
    expected_primary = {
        "calibration_policy": CONTROL_POLICY,
        "treatment_minus_control_accuracy": {
            name: primary[name]
            for name in ("mean", "minimum", "maximum", "sample_standard_deviation")
        },
        "total_treatment_minus_control_correct": sum(
            item["treatment_minus_control_correct"]
            for item in primary["per_assignment"]
        ),
        "total_examples": sum(item["examples"] for item in primary["per_assignment"]),
    }
    if aggregate.get("primary_fixed_control_calibration") != expected_primary:
        raise ValueError("Stored primary fixed-calibration aggregate mismatch.")
    recomputed["primary_fixed_control_calibration"] = {
        **expected_primary,
        "per_assignment": primary["per_assignment"],
    }

    secondary_per_assignment = []
    for seed in EXPECTED_ASSIGNMENTS:
        control = by_key[(seed, CONTROL_POLICY, CONTROL_POLICY)][
            "ideal_bounded_continuous_test"
        ]
        treatment = by_key[(seed, TREATMENT_POLICY, TREATMENT_POLICY)][
            "ideal_bounded_continuous_test"
        ]
        examples = int(control["examples"])
        correct_delta = int(treatment["student_correct"]) - int(
            control["student_correct"]
        )
        secondary_per_assignment.append(
            {
                "assignment_seed": seed,
                "control_correct": control["student_correct"],
                "treatment_correct": treatment["student_correct"],
                "examples": examples,
                "control_accuracy": control["student_accuracy"],
                "treatment_accuracy": treatment["student_accuracy"],
                "treatment_minus_control_correct": correct_delta,
                "treatment_minus_control_accuracy": correct_delta / examples,
            }
        )
    secondary_summary = _accuracy_summary(
        [item["treatment_minus_control_accuracy"] for item in secondary_per_assignment]
    )
    expected_secondary = {
        "control_accuracy": aggregate["scheme_optimized"][CONTROL_POLICY],
        "treatment_accuracy": aggregate["scheme_optimized"][TREATMENT_POLICY],
        "treatment_minus_control_accuracy": secondary_summary,
        "total_treatment_minus_control_correct": sum(
            item["treatment_minus_control_correct"]
            for item in secondary_per_assignment
        ),
        "total_examples": sum(
            item["examples"] for item in secondary_per_assignment
        ),
    }
    if aggregate.get("secondary_scheme_optimized") != expected_secondary:
        raise ValueError("Stored secondary scheme-refitted aggregate mismatch.")
    recomputed["secondary_scheme_optimized"] = {
        **expected_secondary,
        "per_assignment": secondary_per_assignment,
    }

    primary_gate = {
        "threshold": 0.05,
        "observed_mean": primary["mean"],
        "passed": primary["mean"] >= 0.05,
    }
    treatment_gate = {
        "threshold": 0.90,
        "observed_mean": recomputed["scheme_optimized"][TREATMENT_POLICY]["mean"],
        "passed": recomputed["scheme_optimized"][TREATMENT_POLICY]["mean"] >= 0.90,
    }
    if aggregate.get("primary_effect_gate") != primary_gate:
        raise ValueError("Stored five-percentage-point primary gate mismatch.")
    if aggregate.get("treatment_accuracy_gate") != treatment_gate:
        raise ValueError("Stored 90-percent treatment gate mismatch.")
    recomputed["gates"] = {
        "primary_effect": primary_gate,
        "treatment_accuracy": treatment_gate,
    }
    return recomputed


def _expected_result_metrics(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    compact = []
    for block in summary["heldout"]:
        arms = block["arms"]
        compact.append(
            {
                "assignment_seed": block["assignment_seed"],
                "scheme_optimized": {
                    policy: next(
                        arm["ideal_bounded_continuous_test"]
                        for arm in arms
                        if arm["baseline_policy"] == policy
                        and arm["calibration_policy"] == policy
                    )
                    for policy in POLICIES
                },
                "paired_baseline_effect": block["paired_baseline_effect"],
            }
        )
    return {
        "metric_definition_id": summary["metric_definition_id"],
        "evidence_class": summary["contract"]["device"]["evidence_class"],
        "initialization": "pre_bptt_continuous_bounded_local_baseline",
        "heldout": compact,
        "aggregate": summary["aggregate"],
        "artifact_contract": summary["artifact_contract"],
        "coverage_valid": True,
        "optimizer_updates": 0,
    }


def _validate_result_metrics(bundle: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    if bundle["result"].get("metrics") != _expected_result_metrics(summary):
        raise ValueError("result.json metrics and scientific summary differ.")


def _mechanism(
    summary: Mapping[str, Any],
    plans: Mapping[tuple[int, str], Any],
    mappings: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Collect independently checked baseline, loading, sign, and voltage facts."""

    arms = _arm_index(summary)
    assignments = []
    for seed in EXPECTED_ASSIGNMENTS:
        policy_records: dict[str, Any] = {}
        for policy in POLICIES:
            plan = plans[(seed, policy)]
            mapping = mappings[(seed, policy, policy)]["mapping"]
            metric = arms[(seed, policy, policy)]["ideal_bounded_continuous_test"]
            layers = []
            for layer_index, (plan_layer, map_layer, shared_layer) in enumerate(
                zip(
                    plan.report["layers"],
                    mapping["layers"],
                    mapping["shared_matching"]["layers"],
                )
            ):
                layers.append(
                    {
                        "layer": layer_index,
                        "quad_count": plan_layer["quad_count"],
                        "intrinsic_reference_contrast": plan_layer[
                            "intrinsic_reference_contrast"
                        ],
                        "reachable_control_contrast": plan_layer[
                            "reachable_control_contrast"
                        ],
                        "ideal_unit_baseline_contrast": plan_layer[
                            "baseline_contrast"
                        ],
                        "exact_zero_feasible_count": plan_layer[
                            "exact_zero_feasible_count"
                        ],
                        "exact_zero_max_abs_residual": plan_layer[
                            "exact_zero_max_abs_residual"
                        ],
                        "reference_outside_active_bounds_count": plan_layer[
                            "reference_outside_active_bounds_count"
                        ],
                        "analytic_unconstrained_in_bounds_count": plan_layer[
                            "analytic_unconstrained_in_bounds_count"
                        ],
                        "bound_active_projection_quad_count": plan_layer[
                            "bound_active_projection_quad_count"
                        ],
                        "baseline_at_lower_count": plan_layer[
                            "baseline_at_lower_count"
                        ],
                        "baseline_at_upper_count": plan_layer[
                            "baseline_at_upper_count"
                        ],
                        "correction_from_intrinsic_reference": plan_layer[
                            "correction_from_intrinsic_reference"
                        ],
                        "correction_from_reachable_control": plan_layer[
                            "correction_from_reachable_control"
                        ],
                        "loading_change_from_intrinsic_reference": plan_layer[
                            "loading_change_from_intrinsic_reference"
                        ],
                        "loading_change_from_reachable_control": plan_layer[
                            "loading_change_from_reachable_control"
                        ],
                        "physical_baseline_contrast": map_layer[
                            "baseline_contrast"
                        ],
                        "physical_full_g_loading": map_layer["quad_loading"],
                        "physical_baseline_loading": map_layer[
                            "baseline_loading"
                        ],
                        "common_matched_headroom": shared_layer[
                            "common_matched_headroom"
                        ],
                        "zero_common_headroom_count": shared_layer[
                            "zero_common_headroom_count"
                        ],
                        "logical_sign_flip_count": map_layer[
                            "logical_sign_flip_count"
                        ],
                        "logical_sign_flip_fraction_nonzero": map_layer[
                            "logical_sign_flip_fraction_nonzero"
                        ],
                        "full_g_decomposition_max_abs_residual": map_layer[
                            "full_g_decomposition_max_abs_residual"
                        ],
                        "bound_violation_count": map_layer[
                            "bound_violation_count"
                        ],
                    }
                )
            policy_records[policy] = {
                "layers": layers,
                "network_voltage": metric["voltage"],
            }
        assignments.append(
            {
                "assignment_seed": seed,
                "policies": policy_records,
            }
        )
    return {
        "identity_binding": "sampled_order_no_permutation",
        "intrinsic_reference_is_immutable": True,
        "decomposition": "every physical branch G=B+d",
        "circuit_accounting": "full G enters numerator and denominator",
        "assignments": assignments,
    }


def analyze(main_run: Path, replay_run: Path) -> Mapping[str, Any]:
    main_dir = _one_run(main_run)
    replay_dir = _one_run(replay_run)
    main_bundle, main_summary = _validate_bundle(main_dir, expected_arm_id="main")
    replay_bundle, replay_summary = _validate_bundle(
        replay_dir, expected_arm_id="replay"
    )
    replay_attestation = _replay_provenance(
        main_dir, main_bundle, replay_dir, replay_bundle
    )
    main_plans = None
    main_mappings = None
    for run_dir, bundle, summary in (
        (main_dir, main_bundle, main_summary),
        (replay_dir, replay_bundle, replay_summary),
    ):
        by_kind = _artifact_by_kind(bundle, run_dir)
        _require_artifact_contract(by_kind)
        if summary.get("artifact_contract") != {
            "population_artifacts": 4,
            "baseline_plan_artifacts": 8,
            "mapping_artifacts": 14,
            "identity_reassignment_artifacts": 0,
            "scientific_summary_artifacts": 1,
        }:
            raise ValueError("Scientific artifact cardinality report mismatch.")
        populations = _population_map(
            by_kind["ibm_om_population"],
            by_kind["ibm_om_population_receipt"],
            summary,
        )
        logical_weights = _source_logical_weights(bundle, summary, populations)
        plans = _validate_plans(
            by_kind["ibm_om_local_baseline_plan"],
            by_kind["ibm_om_local_baseline_plan_receipt"],
            populations,
            summary,
        )
        development = _validate_development(
            summary, bundle["config"], populations, plans, logical_weights
        )
        mapping_index = _summary_mapping_index(summary, development)
        student = bundle["config"]["student"]
        mappings = _validate_mappings(
            by_kind["ibm_om_full_g_mapping"],
            by_kind["ibm_om_full_g_mapping_receipt"],
            populations,
            plans,
            logical_weights,
            mapping_index,
            conductance_min=float(student["model"]["conductance_min"]),
            conductance_max=float(student["model"]["conductance_max"]),
        )
        _metrics(summary)
        _validate_result_metrics(bundle, summary)
        if run_dir == main_dir:
            main_plans = plans
            main_mappings = mappings
    if _normalize_paths(main_summary) != _normalize_paths(replay_summary):
        raise ValueError("Main/replay scientific summaries differ semantically.")
    _artifact_replay(main_bundle, main_dir, replay_bundle, replay_dir)
    if main_plans is None or main_mappings is None:
        raise RuntimeError("Main-run reconstruction was not retained.")
    metrics = _metrics(main_summary)
    mechanism = _mechanism(main_summary, main_plans, main_mappings)
    return {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "complete_semantic_replay_verified",
        "main_run": str(main_dir),
        "replay_run": str(replay_dir),
        "replay_attestation": replay_attestation,
        "metric_definition_id": main_summary["metric_definition_id"],
        "accuracy": metrics,
        "mechanism": mechanism,
        "gates": metrics["gates"],
        "claim_boundary": main_summary["claim_boundary"],
    }


def _report(analysis: Mapping[str, Any]) -> str:
    accuracy = analysis["accuracy"]
    optimized = accuracy["scheme_optimized"]
    primary = accuracy["primary_fixed_control_calibration"]
    secondary = accuracy["secondary_scheme_optimized"]
    lines = [
        "# Local reference-compensation post-run analysis",
        "",
        f"Status: `{analysis['status']}`",
        "",
        "## Ideal bounded continuous initialization",
        "",
        "| Baseline policy | Mean | Range | Exact held-out results |",
        "|---|---:|---:|---|",
    ]
    for policy in POLICIES:
        item = optimized[policy]
        exact = ", ".join(
            f"{row['assignment_seed']}: {row['correct']}/{row['examples']} "
            f"({100.0 * row['accuracy']:.2f}%)"
            for row in item["per_assignment"]
        )
        lines.append(
            f"| `{policy}` | {100.0 * item['mean']:.2f}% | "
            f"{100.0 * item['minimum']:.2f}–{100.0 * item['maximum']:.2f}% | "
            f"{exact} |"
        )
    lines.extend(
        [
            "",
            "## Matched baseline effects",
            "",
            "| Comparison | Mean treatment−control | Range | Per assignment |",
            "|---|---:|---:|---|",
        ]
    )
    for label, item in (
        ("Primary: frozen control calibration", primary),
        ("Secondary: each scheme refitted", secondary),
    ):
        effect = item["treatment_minus_control_accuracy"]
        exact = ", ".join(
            f"{row['assignment_seed']}: {row['treatment_correct']}−"
            f"{row['control_correct']} "
            f"({100.0 * row['treatment_minus_control_accuracy']:+.2f} pp)"
            for row in item["per_assignment"]
        )
        lines.append(
            f"| {label} | {100.0 * effect['mean']:+.2f} pp | "
            f"{100.0 * effect['minimum']:+.2f} to "
            f"{100.0 * effect['maximum']:+.2f} pp | {exact} |"
        )
    lines.extend(
        [
            "",
            "## Local baseline and full-G diagnostics",
            "",
            (
                "| Assignment | Layer | Policy | Intrinsic contrast RMS | "
                "Ideal B contrast RMS | Physical B contrast RMS | Correction "
                "RMS / p90 / max | Bound-active quads | B lower / upper | "
                "Headroom mean | Zero-headroom | Sign flips | Full-G loading mean |"
            ),
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for assignment in analysis["mechanism"]["assignments"]:
        for policy in POLICIES:
            for item in assignment["policies"][policy]["layers"]:
                correction = item["correction_from_intrinsic_reference"]["absolute"]
                lines.append(
                    f"| {assignment['assignment_seed']} | {item['layer']} | "
                    f"`{policy}` | "
                    f"{item['intrinsic_reference_contrast']['rms']:.8g} | "
                    f"{item['ideal_unit_baseline_contrast']['rms']:.8g} | "
                    f"{item['physical_baseline_contrast']['rms']:.8g} | "
                    f"{correction['rms']:.8g} / {correction['p90']:.8g} / "
                    f"{correction['maximum']:.8g} | "
                    f"{item['bound_active_projection_quad_count']} | "
                    f"{item['baseline_at_lower_count']} / "
                    f"{item['baseline_at_upper_count']} | "
                    f"{item['common_matched_headroom']['mean']:.8g} | "
                    f"{item['zero_common_headroom_count']} | "
                    f"{item['logical_sign_flip_count']} | "
                    f"{item['physical_full_g_loading']['mean']:.8g} |"
                )
    lines.extend(
        [
            "",
            "## Row, column, and total loading change from reachable control",
            "",
            "| Assignment | Layer | Policy | Row RMS | Column RMS | Total RMS |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for assignment in analysis["mechanism"]["assignments"]:
        for policy in POLICIES:
            for item in assignment["policies"][policy]["layers"]:
                loading = item["loading_change_from_reachable_control"]
                lines.append(
                    f"| {assignment['assignment_seed']} | {item['layer']} | "
                    f"`{policy}` | {loading['row']['rms']:.8g} | "
                    f"{loading['column']['rms']:.8g} | "
                    f"{loading['total']['rms']:.8g} |"
                )
    lines.extend(
        [
            "",
            "## Network voltage",
            "",
            "| Assignment | Policy | Network layer | Values | RMS | Mean | Std | Range |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for assignment in analysis["mechanism"]["assignments"]:
        for policy in POLICIES:
            for voltage in assignment["policies"][policy]["network_voltage"]:
                lines.append(
                    f"| {assignment['assignment_seed']} | `{policy}` | "
                    f"{voltage['layer']} | {voltage['values']} | "
                    f"{voltage['rms']:.8g} | {voltage['mean']:.8g} | "
                    f"{voltage['standard_deviation']:.8g} | "
                    f"{voltage['minimum']:.8g} to {voltage['maximum']:.8g} |"
                )
    primary_gate = analysis["gates"]["primary_effect"]
    accuracy_gate = analysis["gates"]["treatment_accuracy"]
    lines.extend(
        [
            "",
            "## Predeclared gates",
            "",
            f"Primary fixed-calibration improvement >= 5 pp: "
            f"**{primary_gate['passed']}** "
            f"(observed {100.0 * primary_gate['observed_mean']:+.2f} pp).",
            "",
            f"Compensated scheme-refitted mean >= 90%: "
            f"**{accuracy_gate['passed']}** "
            f"(observed {100.0 * accuracy_gate['observed_mean']:.2f}%).",
            "",
            "Every registered population, fixed identity order, baseline projection, "
            "selected B/d/G mapping, exact accuracy count, aggregate, and main/replay "
            "semantic value passed reconstruction. The full G was retained in both "
            "contrast and loading; no cancellation was assumed.",
            "",
            f"Claim boundary: {analysis['claim_boundary']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parser().parse_args()
    result = analyze(args.main_run, args.replay_run)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "analysis.json", result)
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    print(json.dumps(result["accuracy"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
