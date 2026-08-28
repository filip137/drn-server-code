"""Fail-closed post-run analysis for the four-reference balance study."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.artifacts import atomic_write_json, content_hash, sha256_file
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _binding_slices,
    _native_to_unit,
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_four_reference_balance import (
    BALANCED_POLICY,
    MAPPING_SCHEMA,
    MAPPING_SCHEMA_VERSION,
    PLAN_SCHEMA,
    PLAN_SCHEMA_VERSION,
    POLICIES,
    RANDOM_POLICY,
    build_continuous_reference_targets,
    _quad_stack,
    build_reference_binding_plan,
)
from experiments.mnist_relu_drn.ibm_om_four_reference_balance_runtime import (
    SUMMARY_SCHEMA,
    SUMMARY_SCHEMA_VERSION,
)
from experiments.mnist_relu_drn.ibm_om_ideal_mapping_scheme_screen import (
    _normalized,
    _quad_contrast,
)
from experiments.study_workflow import (
    StudyWorkflowError,
    load_study_plan,
    load_study_record,
)
from training.ibm_reram_hwa import load_om_array_population


ANALYSIS_SCHEMA = "ebl.mnist_relu_drn.ibm_om_four_reference_balance_analysis"
ANALYSIS_SCHEMA_VERSION = 1
EXPECTED_ASSIGNMENTS = (87001, 87002, 87003)
EXPECTED_DEVELOPMENT = 86001
EXPECTED_STUDY_ID = "mnist-ibm-om-four-reference-balance-ideal-init-20260828-v1"
EXPECTED_EVIDENCE_CLASS = "model_based_aihwkit_preset"
EXPECTED_EXPERIMENT_ID = "ibm_om_four_reference_balance.v1"
EXPECTED_ARM_IDS = ("main", "replay")
_SHA256 = re.compile(r"[0-9a-f]{64}")
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


def _digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"Expected {label} to be a lowercase SHA-256 digest.")
    return value


def _float32_inward_bounds(
    lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independently reproduce the mapper's representable in-bound endpoints."""

    negative_infinity = torch.full_like(lower.to(torch.float32), float("-inf"))
    positive_infinity = torch.full_like(lower.to(torch.float32), float("inf"))
    lower_inward = lower.to(torch.float32)
    lower_inward = torch.where(
        lower_inward.to(torch.float64) < lower,
        torch.nextafter(lower_inward, positive_infinity),
        lower_inward,
    )
    upper_inward = upper.to(torch.float32)
    upper_inward = torch.where(
        upper_inward.to(torch.float64) > upper,
        torch.nextafter(upper_inward, negative_infinity),
        upper_inward,
    )
    if bool(torch.any(lower_inward > upper_inward)):
        raise ValueError("No float32 conductance exists inside an active bound interval.")
    return lower_inward, upper_inward


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run", type=Path, required=True)
    parser.add_argument("--replay-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _one_run(path: Path) -> Path:
    root = path.expanduser().resolve()
    if (root / "result.json").is_file():
        return root
    candidates = tuple(item.parent for item in root.glob("*/result.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one completed native run below {root}; found {len(candidates)}."
        )
    return candidates[0]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Expected readable JSON: {path}.") from error


def _validate_study_context(
    run_dir: Path,
    manifest: Mapping[str, Any],
    *,
    expected_arm_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
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

    context = manifest.get("study")
    expected_context = {
        "study_id": EXPECTED_STUDY_ID,
        "arm_id": expected_arm_id,
        "evidence_class": EXPECTED_EVIDENCE_CLASS,
        "study_sha256": sha256_file(study_path),
        "source_plan_sha256": source_plan_sha,
        "source_config_sha256": arms[expected_arm_id]["configs"][0]["sha256"],
    }
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
        raise ValueError("Expected one complete native four-reference run bundle.")
    study_context, study = _validate_study_context(
        run_dir, manifest, expected_arm_id=expected_arm_id
    )
    if manifest.get("config", {}).get("sha256") != content_hash(config):
        raise ValueError("Resolved config hash mismatch.")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Expected registered native artifacts.")
    seen = set()
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
    summary_records = [
        record for record in artifacts if record.get("kind") == "scientific_summary"
    ]
    if len(summary_records) != 1:
        raise ValueError("Expected exactly one scientific summary artifact.")
    summary = _read_json(run_dir / summary_records[0]["path"])
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


def _artifact_by_kind(bundle: Mapping[str, Any], run_dir: Path):
    result: dict[str, list[Path]] = defaultdict(list)
    for record in bundle["artifacts"]:
        result[str(record["kind"])].append(run_dir / str(record["path"]))
    return result


def _require_artifact_contract(by_kind: Mapping[str, Sequence[Path]]) -> None:
    expected = {
        "ibm_om_population": 4,
        "ibm_om_population_receipt": 4,
        "ibm_om_reference_binding_plan": 8,
        "ibm_om_reference_binding_plan_receipt": 8,
        "ibm_om_full_g_mapping": 14,
        "ibm_om_full_g_mapping_receipt": 14,
        "scientific_summary": 1,
    }
    observed = {name: len(paths) for name, paths in by_kind.items()}
    if observed != expected:
        raise ValueError(
            f"Registered artifact kind/count contract mismatch: {observed!r}."
        )


def _registered_receipt(path: Path, receipt_paths: Sequence[Path]) -> Path:
    expected = path.with_suffix(".receipt.json").resolve()
    registered = {item.resolve() for item in receipt_paths}
    if expected not in registered:
        raise ValueError(f"Expected a registered receipt for {path.name}.")
    return expected


def _input_map(bundle: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    inputs = bundle["manifest"].get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 3:
        raise ValueError("Expected exactly weights, teacher, and AIHWKit inputs.")
    result: dict[str, Mapping[str, Any]] = {}
    for record in inputs:
        if not isinstance(record, Mapping) or set(record) != {"role", "path", "sha256"}:
            raise ValueError("Input artifact record fields mismatch.")
        role = record.get("role")
        if not isinstance(role, str) or role in result:
            raise ValueError("Input artifact roles must be unique strings.")
        source = Path(str(record.get("path"))).expanduser().resolve()
        digest = _digest(record.get("sha256"), label=f"{role} input hash")
        if not source.is_file() or sha256_file(source) != digest:
            raise ValueError(f"Input artifact is absent or changed: {role}.")
        result[role] = dict(record)
    if set(result) != {"weights", "teacher_weights", "aihwkit_python"}:
        raise ValueError("Input artifact role coverage mismatch.")
    return result


def _strict_named_weights(path: Path) -> tuple[Mapping[str, Any], Mapping[str, torch.Tensor]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError) as error:
        raise ValueError(f"Expected a safe named-weight checkpoint: {path}.") from error
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema", "schema_version", "catalog", "weights", "metadata"}
        or payload.get("schema") != "drn.named-weights"
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("catalog"), list)
        or not isinstance(payload.get("weights"), Mapping)
        or not isinstance(payload.get("metadata"), Mapping)
    ):
        raise ValueError("Named-weight checkpoint schema/fields mismatch.")
    weights = payload["weights"]
    catalog = payload["catalog"]
    if len(catalog) != len(weights):
        raise ValueError("Named-weight checkpoint catalog cardinality mismatch.")
    for index, (key, value) in enumerate(weights.items()):
        descriptor = catalog[index]
        if (
            not isinstance(value, torch.Tensor)
            or not isinstance(descriptor, Mapping)
            or descriptor.get("key") != key
            or tuple(descriptor.get("shape", ())) != tuple(value.shape)
            or descriptor.get("dtype") != str(value.dtype)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("Named-weight tensor/catalog descriptor mismatch.")
    return payload, weights


def _source_logical_weights(
    bundle: Mapping[str, Any],
    summary: Mapping[str, Any],
    populations: Mapping[int, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = _input_map(bundle)
    protocol = bundle["config"].get("protocol")
    if not isinstance(protocol, Mapping) or summary.get("contract") != protocol:
        raise ValueError("Resolved protocol and scientific-summary contract differ.")
    source_contract = protocol.get("source")
    if not isinstance(source_contract, Mapping):
        raise ValueError("Expected frozen source contract in resolved config.")
    weights_record = inputs["weights"]
    teacher_record = inputs["teacher_weights"]
    if (
        weights_record["sha256"] != source_contract.get("expected_weights_sha256")
        or teacher_record["sha256"] != source_contract.get("expected_teacher_sha256")
    ):
        raise ValueError("Manifest inputs do not match the frozen source hashes.")
    source_payload, source_weights = _strict_named_weights(Path(weights_record["path"]))
    population = populations[EXPECTED_DEVELOPMENT][0]
    expected_source_keys = ("base.dense_weight.0", "base.dense_weight.1")
    if tuple(source_weights) != expected_source_keys or tuple(
        tuple(source_weights[key].shape) for key in expected_source_keys
    ) != population.binding_shapes:
        raise ValueError("Frozen source checkpoint physical layout mismatch.")
    logical = tuple(
        _quad_contrast(source_weights[key], layout=layout) / 2.0
        for key, layout in zip(expected_source_keys, ("halves", "paired"))
    )

    teacher_payload, teacher_weights = _strict_named_weights(Path(teacher_record["path"]))
    expected_teacher_keys = ("teacher.dense_weight.0", "teacher.dense_weight.1")
    if tuple(teacher_weights) != expected_teacher_keys:
        raise ValueError("Frozen teacher checkpoint key order mismatch.")
    deviations = []
    for source, key in zip(logical, expected_teacher_keys):
        teacher = teacher_weights[key]
        if source.shape != teacher.shape:
            raise ValueError("Frozen source and teacher logical shapes differ.")
        deviations.append(
            float((_normalized(source) - _normalized(teacher)).abs().max().item())
        )
    if max(deviations) > 2e-6:
        raise ValueError("Frozen source no longer reproduces the named teacher.")
    expected_source = {
        "checkpoint_sha256": weights_record["sha256"],
        "checkpoint_metadata": source_payload["metadata"],
        "logical_weight_hashes": [_tensor_sha256(value) for value in logical],
        "normalized_teacher_max_abs_deviation": deviations,
        "extraction": "half_four_rail_contrast_then_layerwise_absmax_normalization",
    }
    if summary.get("source") != expected_source:
        raise ValueError("Scientific-summary source report does not reconstruct.")
    expected_teacher = summary.get("teacher")
    if (
        not isinstance(expected_teacher, Mapping)
        or expected_teacher.get("sha256") != teacher_record["sha256"]
        or expected_teacher.get("metadata") != teacher_payload["metadata"]
    ):
        raise ValueError("Scientific-summary teacher report mismatch.")
    return logical[0], logical[1]


def _load_npz(path: Path) -> Mapping[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as raw:
            return {name: raw[name].copy() for name in raw.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"Expected readable pickle-free NPZ: {path}.") from error


def _scalar(payload: Mapping[str, np.ndarray], name: str) -> Any:
    value = payload[name]
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"Expected scalar NPZ field {name!r}.")
    return value.item()


def _population_summary_index(summary: Mapping[str, Any]) -> Mapping[int, Mapping[str, Any]]:
    development = summary.get("development_population")
    heldout = summary.get("heldout")
    if not isinstance(development, Mapping) or not isinstance(heldout, list):
        raise ValueError("Expected development and held-out population reports.")
    result = {EXPECTED_DEVELOPMENT: development}
    for block in heldout:
        if not isinstance(block, Mapping) or not isinstance(block.get("population"), Mapping):
            raise ValueError("Expected held-out population report objects.")
        seed = block.get("assignment_seed")
        if seed in result:
            raise ValueError("Duplicate summary population assignment seed.")
        result[seed] = block["population"]
    if set(result) != {EXPECTED_DEVELOPMENT, *EXPECTED_ASSIGNMENTS}:
        raise ValueError("Summary population assignment coverage mismatch.")
    return result


def _population_map(
    paths: Sequence[Path],
    receipt_paths: Sequence[Path],
    summary: Mapping[str, Any],
):
    if len(paths) != 4:
        raise ValueError("Expected four population artifacts.")
    if len(receipt_paths) != 4:
        raise ValueError("Expected four registered population receipts.")
    summary_index = _population_summary_index(summary)
    result = {}
    for path in paths:
        population = load_om_array_population(path)
        if population.assignment_seed in result:
            raise ValueError("Duplicate population assignment seed.")
        receipt_path = _registered_receipt(path, receipt_paths)
        receipt = _read_json(receipt_path)
        expected_request = {
            "preset": "reram_array_om",
            "assignment_seed": population.assignment_seed,
            "corruption_policy": population.corruption_policy,
            "binding_keys": list(population.binding_keys),
            "binding_shapes": [list(shape) for shape in population.binding_shapes],
            "required_aihwkit_version": "1.1.0",
        }
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("schema") != "ebl.ibm_reram.om_array_population_receipt"
            or receipt.get("schema_version") != 1
            or receipt.get("backend") != "external_pinned_aihwkit_python"
            or receipt.get("aihwkit_version") != "1.1.0"
            or receipt.get("request") != expected_request
            or receipt.get("num_cells") != population.size
            or receipt.get("population_fingerprint") != population.fingerprint
            or receipt.get("population_sha256") != sha256_file(path)
        ):
            raise ValueError("Population receipt does not match its sampled population.")
        for name in ("sampler_source_sha256", "population_implementation_sha256"):
            _digest(receipt.get(name), label=f"population receipt {name}")
        report = summary_index[population.assignment_seed]
        if (
            report.get("assignment_seed") != population.assignment_seed
            or report.get("topology") != 4
            or Path(str(report.get("path"))).name != path.name
            or report.get("sha256") != sha256_file(path)
            or Path(str(report.get("receipt"))).name != receipt_path.name
            or report.get("receipt_sha256") != sha256_file(receipt_path)
            or report.get("population_fingerprint") != population.fingerprint
            or report.get("cells") != population.size
            or report.get("final_corrupt_cells") != int(population.corrupt.sum().item())
            or report.get("published_corrupt_cells_repaired")
            != int(population.published_corrupt.sum().item())
        ):
            raise ValueError("Summary population report is not linked to its artifacts.")
        result[population.assignment_seed] = (population, path, receipt_path)
    if set(result) != {EXPECTED_DEVELOPMENT, *EXPECTED_ASSIGNMENTS}:
        raise ValueError("Population assignment coverage mismatch.")
    return result


def _validate_plans(
    paths: Sequence[Path],
    receipt_paths: Sequence[Path],
    populations,
    summary: Mapping[str, Any],
) -> Mapping[tuple[int, str], Any]:
    if len(paths) != 8:
        raise ValueError("Expected two binding plans for each of four assignments.")
    if len(receipt_paths) != 8:
        raise ValueError("Expected eight registered binding-plan receipts.")
    summary_index: dict[tuple[int, str], Mapping[str, Any]] = {}
    development = summary.get("development")
    if not isinstance(development, Mapping):
        raise ValueError("Expected development binding-plan reports.")
    for policy in POLICIES:
        item = development.get(policy)
        if not isinstance(item, Mapping):
            raise ValueError("Expected one development report per binding policy.")
        summary_index[(EXPECTED_DEVELOPMENT, policy)] = {
            "report": item.get("binding_plan"),
            "artifact": item.get("binding_plan_artifact"),
        }
    heldout = summary.get("heldout")
    if not isinstance(heldout, list):
        raise ValueError("Expected held-out binding-plan reports.")
    for block in heldout:
        seed = block.get("assignment_seed")
        plan_reports = block.get("binding_plans")
        if not isinstance(plan_reports, Mapping):
            raise ValueError("Expected held-out binding-plan reports.")
        for policy in POLICIES:
            item = plan_reports.get(policy)
            if not isinstance(item, Mapping):
                raise ValueError("Expected every held-out binding policy report.")
            summary_index[(seed, policy)] = item
    result = {}
    expected_fields = {
        "schema",
        "schema_version",
        "policy",
        "assignment_seed",
        "population_fingerprint",
        "binding_keys_json",
        "binding_shapes_json",
        "destination_to_source",
    }
    for path in paths:
        payload = _load_npz(path)
        if set(payload) != expected_fields:
            raise ValueError("Binding-plan NPZ field mismatch.")
        if (
            str(_scalar(payload, "schema")) != PLAN_SCHEMA
            or int(_scalar(payload, "schema_version")) != PLAN_SCHEMA_VERSION
            or payload["schema_version"].dtype != np.int64
            or payload["assignment_seed"].dtype != np.int64
        ):
            raise ValueError("Binding-plan scalar schema/dtype mismatch.")
        seed = int(_scalar(payload, "assignment_seed"))
        policy = str(_scalar(payload, "policy"))
        if policy not in POLICIES or seed not in populations:
            raise ValueError("Unexpected binding-plan assignment or policy.")
        population = populations[seed][0]
        if str(_scalar(payload, "population_fingerprint")) != population.fingerprint:
            raise ValueError("Binding plan population fingerprint mismatch.")
        permutation_array = payload["destination_to_source"]
        if permutation_array.dtype != np.int64 or permutation_array.shape != (
            population.size,
        ):
            raise ValueError("Binding-plan permutation dtype/shape mismatch.")
        permutation = torch.from_numpy(permutation_array)
        expected = build_reference_binding_plan(population, policy=policy)
        if not torch.equal(permutation, expected.destination_to_source):
            raise ValueError("Stored binding plan does not reproduce from its contract.")
        if (
            json.loads(str(_scalar(payload, "binding_keys_json")))
            != list(population.binding_keys)
            or json.loads(str(_scalar(payload, "binding_shapes_json")))
            != [list(shape) for shape in population.binding_shapes]
        ):
            raise ValueError("Binding-plan stored layout mismatch.")
        receipt_path = _registered_receipt(path, receipt_paths)
        receipt = _read_json(receipt_path)
        receipt_metadata = {
            "artifact": path.name,
            "artifact_sha256": sha256_file(path),
            "binding_keys": list(population.binding_keys),
            "binding_shapes": [list(shape) for shape in population.binding_shapes],
        }
        report_keys = set(expected.report)
        if (
            set(receipt) != report_keys | set(receipt_metadata)
            or any(receipt.get(name) != value for name, value in receipt_metadata.items())
        ):
            raise ValueError("Binding-plan receipt mismatch.")
        stored_report = {name: receipt[name] for name in expected.report}
        if not _same_recomputed_report(stored_report, expected.report):
            raise ValueError("Binding-plan receipt does not reproduce numerically.")
        key = (seed, policy)
        if key in result:
            raise ValueError("Duplicate binding-plan assignment/policy.")
        summary_item = summary_index.get(key)
        artifact_link = (
            summary_item.get("artifact") if isinstance(summary_item, Mapping) else None
        )
        if (
            not isinstance(summary_item, Mapping)
            or summary_item.get("report") != stored_report
            or not isinstance(artifact_link, Mapping)
            or Path(str(artifact_link.get("path"))).name != path.name
            or artifact_link.get("sha256") != sha256_file(path)
            or Path(str(artifact_link.get("receipt"))).name != receipt_path.name
            or artifact_link.get("receipt_sha256") != sha256_file(receipt_path)
            or artifact_link.get("policy") != policy
            or artifact_link.get("destination_to_source_sha256")
            != expected.report["destination_to_source_sha256"]
        ):
            raise ValueError("Summary binding plan is not linked to its exact artifacts.")
        result[key] = expected
    if set(result) != set(summary_index) or len(result) != 8:
        raise ValueError("Binding-plan assignment/policy coverage mismatch.")
    return result


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _same_recomputed_report(stored: Any, expected: Any) -> bool:
    """Compare recomputed reports without trusting nondeterministic reduction ulps.

    Material tensors and their hashes are checked exactly elsewhere.  Report
    scalars produced by parallel CPU reductions can differ in the final few
    binary digits across otherwise identical processes, so keep the complete
    JSON structure and every non-floating value exact while allowing only a
    very small finite-float tolerance.
    """

    if isinstance(expected, Mapping):
        return (
            isinstance(stored, Mapping)
            and set(stored) == set(expected)
            and all(
                _same_recomputed_report(stored[key], expected[key])
                for key in expected
            )
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(stored, (list, tuple))
            and len(stored) == len(expected)
            and all(
                _same_recomputed_report(left, right)
                for left, right in zip(stored, expected)
            )
        )
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return type(stored) is type(expected) and stored == expected
    if isinstance(expected, int):
        return isinstance(stored, int) and not isinstance(stored, bool) and stored == expected
    if isinstance(expected, float):
        return (
            isinstance(stored, float)
            and math.isfinite(stored)
            and math.isfinite(expected)
            and math.isclose(stored, expected, rel_tol=1e-12, abs_tol=1e-15)
        )
    return type(stored) is type(expected) and stored == expected


def _validate_development(
    summary: Mapping[str, Any], config: Mapping[str, Any]
) -> Mapping[str, Mapping[str, Any]]:
    development = summary.get("development")
    grid = config.get("student", {}).get("mapping", {}).get("scale_fractions")
    if not isinstance(development, Mapping) or not isinstance(grid, list):
        raise ValueError("Expected development calibration and resolved scale grid.")
    expected_pairs = {
        (float(left), float(right)) for left in grid for right in grid
    }
    if len(expected_pairs) != 16:
        raise ValueError("Expected the frozen four-by-four scale grid.")
    result = {}
    for policy in POLICIES:
        record = development.get(policy)
        if not isinstance(record, Mapping) or record.get("binding_policy") != policy:
            raise ValueError("Development binding-policy report mismatch.")
        candidates = record.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 16:
            raise ValueError("Expected exactly sixteen development candidates per policy.")
        observed_pairs = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not _finite_numbers(candidate):
                raise ValueError("Expected finite development candidate metrics.")
            pair = candidate.get("scale_fractions")
            calibration = candidate.get("calibration")
            hashes = candidate.get("target_hashes")
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(calibration, Mapping)
                or float(calibration.get("gain", 0.0)) <= 0.0
                or not 0.0 <= float(calibration.get("student_accuracy", -1.0)) <= 1.0
                or float(calibration.get("calibrated_kl", -1.0)) < 0.0
                or not isinstance(hashes, list)
                or len(hashes) != 2
            ):
                raise ValueError("Development candidate calibration/hash contract mismatch.")
            for digest in hashes:
                _digest(digest, label="development target hash")
            observed_pairs.append((float(pair[0]), float(pair[1])))
        if len(set(observed_pairs)) != 16 or set(observed_pairs) != expected_pairs:
            raise ValueError("Development candidates are not the exact Cartesian scale grid.")
        expected_index = min(
            range(16),
            key=lambda index: (
                -float(candidates[index]["calibration"]["student_accuracy"]),
                float(candidates[index]["calibration"]["calibrated_kl"]),
                tuple(float(value) for value in candidates[index]["scale_fractions"]),
            ),
        )
        if (
            record.get("selected_index") != expected_index
            or record.get("selected") != candidates[expected_index]
            or record.get("selection_domain")
            != "development_assignment_86001_calibration_subset"
            or record.get("selection_metric")
            != "mapped_accuracy_then_calibrated_kl_then_scale_pair"
        ):
            raise ValueError("Stored development selection does not recompute exactly.")
        selected_mapping = record.get("selected_mapping")
        if (
            not isinstance(selected_mapping, Mapping)
            or selected_mapping.get("target_hashes")
            != candidates[expected_index]["target_hashes"]
            or selected_mapping.get("scale_fractions")
            != candidates[expected_index]["scale_fractions"]
        ):
            raise ValueError("Selected development mapping is not linked to its candidate.")
        result[policy] = record
    return result


def _compact_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "policy": mapping["policy"],
        "assignment_seed": mapping["assignment_seed"],
        "scale_fractions": mapping["scale_fractions"],
        "target_hashes": mapping["target_hashes"],
        "layers": [
            {
                "layer": layer["layer"],
                "baseline_contrast": layer["baseline_contrast"],
                "realized_contrast": layer["realized_contrast"],
                "quad_loading": layer["quad_loading"],
                "common_positive_headroom": layer["common_positive_headroom"],
                "zero_positive_headroom_count": layer["zero_positive_headroom_count"],
                "reference_outside_active_bounds_count": layer[
                    "reference_outside_active_bounds_count"
                ],
                "continuous_logical_sign_flip_count": layer[
                    "continuous_logical_sign_flip_count"
                ],
                "continuous_logical_sign_flip_fraction_nonzero": layer[
                    "continuous_logical_sign_flip_fraction_nonzero"
                ],
                "full_g_decomposition_max_abs_residual": layer[
                    "full_g_decomposition_max_abs_residual"
                ],
                "active_bound_violation_count": layer["active_bound_violation_count"],
                "float32_inward_baseline_adjustment_count": layer[
                    "float32_inward_baseline_adjustment_count"
                ],
                "float32_inward_conductance_adjustment_count": layer[
                    "float32_inward_conductance_adjustment_count"
                ],
                "float32_inward_conductance_max_abs_adjustment": layer[
                    "float32_inward_conductance_max_abs_adjustment"
                ],
                "hashes": layer["hashes"],
            }
            for layer in mapping["layers"]
        ],
    }


def _summary_mapping_index(
    summary: Mapping[str, Any],
    development: Mapping[str, Mapping[str, Any]],
) -> Mapping[tuple[int, str, str], Mapping[str, Any]]:
    result = {}
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
            binding = arm.get("binding_policy")
            calibration = arm.get("calibration_policy")
            key = (seed, binding, calibration)
            if key in result or binding not in POLICIES or calibration not in POLICIES:
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
    populations,
    plans,
    logical_weights: Sequence[torch.Tensor],
    summary_index: Mapping[tuple[int, str, str], Mapping[str, Any]],
    *,
    conductance_min: float,
    conductance_max: float,
) -> Mapping[tuple[int, str, str], Mapping[str, Any]]:
    if len(paths) != 14:
        raise ValueError("Expected two development and twelve held-out mapping artifacts.")
    if len(receipt_paths) != 14:
        raise ValueError("Expected fourteen registered mapping receipts.")
    expected_fields = {
        "schema",
        "schema_version",
        "assignment_seed",
        "binding_policy",
        "calibration_policy",
        *(
            f"layer_{layer}_{name}"
            for layer in range(2)
            for name in _MAPPING_FIELDS
        ),
    }
    span = conductance_max - conductance_min
    result = {}
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
        binding_policy = str(_scalar(payload, "binding_policy"))
        calibration_policy = str(_scalar(payload, "calibration_policy"))
        if (
            binding_policy not in POLICIES
            or calibration_policy not in POLICIES
            or seed not in populations
            or (seed, binding_policy) not in plans
        ):
            raise ValueError("Unexpected mapping policy.")
        mapping_key = (seed, binding_policy, calibration_policy)
        summary_item = summary_index.get(mapping_key)
        if not isinstance(summary_item, Mapping):
            raise ValueError("Mapping artifact has no exact summary arm.")
        pair = summary_item.get("scale_fractions")
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Expected two frozen selected scale fractions.")
        population = populations[seed][0]
        plan = plans[(seed, binding_policy)]
        expected_targets, expected_mapping, expected_components = (
            build_continuous_reference_targets(
                logical_weights,
                population,
                plan,
                scale_fractions=(float(pair[0]), float(pair[1])),
                conductance_min=conductance_min,
                conductance_max=conductance_max,
            )
        )
        receipt_path = _registered_receipt(path, receipt_paths)
        receipt = _read_json(receipt_path)
        expected_receipt_metadata = {
            "schema": MAPPING_SCHEMA,
            "schema_version": MAPPING_SCHEMA_VERSION,
            "assignment_seed": seed,
            "binding_policy": binding_policy,
            "calibration_policy": calibration_policy,
            "artifact": path.name,
            "artifact_sha256": sha256_file(path),
        }
        if (
            set(receipt) != set(expected_receipt_metadata) | {"mapping"}
            or any(
                receipt.get(name) != value
                for name, value in expected_receipt_metadata.items()
            )
            or not _same_recomputed_report(receipt.get("mapping"), expected_mapping)
        ):
            raise ValueError("Mapping receipt does not exactly reconstruct.")
        if summary_item.get("mapping") != _compact_mapping(receipt["mapping"]):
            raise ValueError("Summary mapping report does not exactly reconstruct.")
        artifact_link = summary_item.get("artifact")
        if (
            not isinstance(artifact_link, Mapping)
            or Path(str(artifact_link.get("path"))).name != path.name
            or artifact_link.get("sha256") != sha256_file(path)
            or Path(str(artifact_link.get("receipt"))).name != receipt_path.name
            or artifact_link.get("receipt_sha256") != sha256_file(receipt_path)
            or artifact_link.get("assignment_seed") != seed
            or artifact_link.get("binding_policy") != binding_policy
            or artifact_link.get("calibration_policy") != calibration_policy
        ):
            raise ValueError("Summary mapping artifact/receipt link mismatch.")
        slices = _binding_slices(population)
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
                raise ValueError("Stored conductance is not the reconstructed target.")
            for name in ("baseline", "offset", "conductance", "common_headroom"):
                if values[name].dtype != torch.float32 or values[name].shape != shape:
                    raise ValueError("Mapping branch tensor dtype/shape mismatch.")
            if values["active_mask"].dtype != torch.bool or values["active_mask"].shape != shape:
                raise ValueError("Mapping active-mask dtype/shape mismatch.")
            for name in ("baseline_contrast", "offset_contrast", "realized_contrast", "loading"):
                if values[name].dtype != torch.float32:
                    raise ValueError("Mapping derived tensor dtype mismatch.")
            if not all(
                bool(torch.isfinite(value).all())
                for name, value in values.items()
                if name != "active_mask"
            ):
                raise ValueError("Nonfinite mapping tensor.")
            if not torch.equal(values["conductance"], values["baseline"] + values["offset"]):
                raise ValueError("Stored physical conductance is not bit-exact B+d.")
            if bool(torch.any(values["conductance"] < 0.0)) or bool(
                torch.any(values["offset"] < 0.0)
            ):
                raise ValueError("Expected nonnegative physical G and d.")
            if bool(torch.any(values["offset"][~values["active_mask"]] != 0.0)):
                raise ValueError("Inactive reference branches must have zero offset.")
            expected_contrast = _quad_contrast(values["conductance"], layout=layout) / 2.0
            expected_baseline = _quad_contrast(values["baseline"], layout=layout) / 2.0
            expected_offset = _quad_contrast(values["offset"], layout=layout) / 2.0
            expected_loading = _quad_stack(values["conductance"], layout=layout).sum(dim=-1)
            if not (
                torch.equal(values["realized_contrast"], expected_contrast)
                and torch.equal(values["baseline_contrast"], expected_baseline)
                and torch.equal(values["offset_contrast"], expected_offset)
                and torch.equal(values["loading"], expected_loading)
            ):
                raise ValueError("Stored full-G contrast/loading does not reproduce.")
            layer_report = expected_mapping["layers"][layer]
            hashes = layer_report["hashes"]
            hash_values = {
                "baseline": values["baseline"],
                "offset": values["offset"],
                "conductance": values["conductance"],
                "baseline_contrast": values["baseline_contrast"],
                "realized_contrast": values["realized_contrast"],
                "loading": values["loading"],
                "active_mask": values["active_mask"],
                "common_headroom": values["common_headroom"],
            }
            if any(_tensor_sha256(value) != hashes[name] for name, value in hash_values.items()):
                raise ValueError("Mapping tensor hash does not match its report.")
            binding_slice = slices[binding_key]
            source_indices = plan.destination_to_source[binding_slice]
            lower = _native_to_unit(
                population.min_bound[source_indices].to(torch.float64)
            ).reshape(shape)
            upper = _native_to_unit(
                population.max_bound[source_indices].to(torch.float64)
            ).reshape(shape)
            reference = _native_to_unit(
                population.reference[source_indices].to(torch.float64)
            ).reshape(shape)
            active_zero = torch.minimum(torch.maximum(reference, lower), upper)
            expected_baseline_unit = torch.where(
                values["active_mask"], active_zero, reference
            )
            raw_baseline_physical = (
                conductance_min + span * expected_baseline_unit
            ).to(torch.float32)
            physical_lower = conductance_min + span * lower
            physical_upper = conductance_min + span * upper
            lower_inward, upper_inward = _float32_inward_bounds(
                physical_lower, physical_upper
            )
            active_baseline_physical = torch.minimum(
                torch.maximum(raw_baseline_physical, lower_inward), upper_inward
            )
            expected_baseline_physical = torch.where(
                values["active_mask"],
                active_baseline_physical,
                raw_baseline_physical,
            )
            if not torch.equal(values["baseline"], expected_baseline_physical):
                raise ValueError(
                    "Stored branch baseline does not match the assigned-reference rule."
                )
            conductance_unit = (values["conductance"] - conductance_min) / span
            active = values["active_mask"]
            if bool(
                torch.any(
                    active
                    & ((conductance_unit < lower - 2e-7) | (conductance_unit > upper + 2e-7))
                )
            ):
                raise ValueError("Active mapped conductance lies outside assigned bounds.")
        if mapping_key in result:
            raise ValueError("Duplicate mapping assignment/policy key.")
        result[mapping_key] = {
            "path": path,
            "receipt_path": receipt_path,
            "receipt": receipt,
            "mapping": expected_mapping,
        }
    expected_keys = {
        (EXPECTED_DEVELOPMENT, policy, policy) for policy in POLICIES
    } | {
        (seed, binding, calibration)
        for seed in EXPECTED_ASSIGNMENTS
        for binding in POLICIES
        for calibration in POLICIES
    }
    if set(result) != expected_keys or set(summary_index) != expected_keys:
        raise ValueError("Mapping assignment/policy coverage mismatch.")
    return result


def _accuracy_summary(values: Sequence[float]) -> Mapping[str, float]:
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
        "sample_standard_deviation": float(tensor.std(unbiased=True).item()),
    }


def _validated_metric(metric: Any) -> Mapping[str, Any]:
    expected_fields = {
        "examples",
        "student_correct",
        "teacher_correct",
        "teacher_agreement_count",
        "kl_teacher_student",
        "raw_kl_teacher_student",
        "student_accuracy",
        "teacher_accuracy",
        "teacher_agreement",
        "raw_score_rms",
        "calibrated_score_rms",
        "teacher_logit_rms",
        "fixed_logit_gain",
        "prediction_sha256",
        "voltage",
    }
    if not isinstance(metric, Mapping) or set(metric) != expected_fields:
        raise ValueError("Held-out metric field contract mismatch.")
    examples = metric["examples"]
    counts = (
        metric["student_correct"],
        metric["teacher_correct"],
        metric["teacher_agreement_count"],
    )
    if (
        isinstance(examples, bool)
        or examples != 10000
        or any(isinstance(value, bool) or not isinstance(value, int) for value in counts)
        or any(not 0 <= value <= examples for value in counts)
        or not _finite_numbers(metric)
        or metric["student_accuracy"] != counts[0] / examples
        or metric["teacher_accuracy"] != counts[1] / examples
        or metric["teacher_agreement"] != counts[2] / examples
        or not 0.0 <= metric["student_accuracy"] <= 1.0
        or not 0.0 <= metric["teacher_accuracy"] <= 1.0
        or not 0.0 <= metric["teacher_agreement"] <= 1.0
        or metric["kl_teacher_student"] < 0.0
        or metric["raw_kl_teacher_student"] < 0.0
        or metric["fixed_logit_gain"] <= 0.0
    ):
        raise ValueError("Held-out exact metric/count record mismatch.")
    _digest(metric["prediction_sha256"], label="prediction hash")
    voltage = metric["voltage"]
    expected_voltage_values = (15_680_000, 1_000_000, 200_000)
    voltage_fields = {
        "layer",
        "values",
        "minimum",
        "maximum",
        "mean",
        "standard_deviation",
        "rms",
    }
    if not isinstance(voltage, list) or len(voltage) != len(expected_voltage_values):
        raise ValueError("Expected input, hidden, and output voltage reports.")
    for index, (item, expected_values) in enumerate(
        zip(voltage, expected_voltage_values)
    ):
        if (
            not isinstance(item, Mapping)
            or set(item) != voltage_fields
            or item.get("layer") != index
            or item.get("values") != expected_values
            or not _finite_numbers(item)
            or float(item.get("rms", -1.0)) < 0.0
            or float(item.get("standard_deviation", -1.0)) < 0.0
            or not float(item.get("minimum", 1.0))
            <= float(item.get("mean", 0.0))
            <= float(item.get("maximum", -1.0))
            or not math.isclose(
                float(item["rms"]) ** 2,
                float(item["standard_deviation"]) ** 2
                + float(item["mean"]) ** 2,
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Voltage diagnostic record mismatch.")
    return metric


def _arm_index(summary: Mapping[str, Any]) -> Mapping[tuple[int, str, str], Mapping[str, Any]]:
    heldout = summary.get("heldout")
    if not isinstance(heldout, list) or tuple(
        block.get("assignment_seed") for block in heldout
    ) != EXPECTED_ASSIGNMENTS:
        raise ValueError("Held-out assignment coverage/order mismatch.")
    result = {}
    for block in heldout:
        arms = block.get("arms")
        if not isinstance(arms, list) or len(arms) != 4:
            raise ValueError("Expected four binding/calibration arms per assignment.")
        for arm in arms:
            if not isinstance(arm, Mapping):
                raise ValueError("Expected held-out arm objects.")
            key = (
                block["assignment_seed"],
                arm.get("binding_policy"),
                arm.get("calibration_policy"),
            )
            if key in result or key[1] not in POLICIES or key[2] not in POLICIES:
                raise ValueError("Held-out arm key coverage mismatch.")
            _validated_metric(arm.get("ideal_bounded_continuous_test"))
            result[key] = arm
    expected = {
        (seed, binding, calibration)
        for seed in EXPECTED_ASSIGNMENTS
        for binding in POLICIES
        for calibration in POLICIES
    }
    if set(result) != expected:
        raise ValueError("Held-out arm key coverage mismatch.")
    return result


def _metrics(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    if (
        summary.get("metric_definition_id")
        != "ibm_om.reference_balanced_continuous_init.v1"
        or summary.get("exclusions")
        != {
            "optimizer_updates": 0,
            "quantization_enabled": False,
            "program_verify_enabled": False,
            "hwa_enabled": False,
            "deployment_write_noise": 0.0,
            "inference_read_noise": 0.0,
            "retention_or_drift": False,
        }
    ):
        raise ValueError("Scientific definition/exclusion contract mismatch.")
    by_key = _arm_index(summary)
    recomputed = {"scheme_optimized": {}, "fixed_calibration": {}}
    for policy in POLICIES:
        per_assignment = []
        for seed in EXPECTED_ASSIGNMENTS:
            metric = by_key[(seed, policy, policy)]["ideal_bounded_continuous_test"]
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
            balanced_metric = by_key[(seed, BALANCED_POLICY, calibration)][
                "ideal_bounded_continuous_test"
            ]
            random_metric = by_key[(seed, RANDOM_POLICY, calibration)][
                "ideal_bounded_continuous_test"
            ]
            paired = next(
                block["paired_binding_effect"][calibration]
                for block in summary["heldout"]
                if block["assignment_seed"] == seed
            )
            delta = balanced_metric["student_accuracy"] - random_metric["student_accuracy"]
            if (
                not isinstance(paired, Mapping)
                or paired.get("balanced_minus_random_accuracy") != delta
                or isinstance(paired.get("prediction_flip_count"), bool)
                or not isinstance(paired.get("prediction_flip_count"), int)
                or not 0 <= paired["prediction_flip_count"] <= 10000
                or paired.get("prediction_flip_fraction")
                != paired["prediction_flip_count"] / 10000.0
            ):
                raise ValueError("Stored paired fixed-calibration effect mismatch.")
            per_assignment.append(
                {
                    "assignment_seed": seed,
                    "random_correct": random_metric["student_correct"],
                    "balanced_correct": balanced_metric["student_correct"],
                    "random_accuracy": random_metric["student_accuracy"],
                    "balanced_accuracy": balanced_metric["student_accuracy"],
                    "balanced_minus_random_accuracy": delta,
                    "prediction_flip_count": paired["prediction_flip_count"],
                    "prediction_flip_fraction": paired["prediction_flip_fraction"],
                }
            )
        deltas = [item["balanced_minus_random_accuracy"] for item in per_assignment]
        recomputed["fixed_calibration"][calibration] = {
            "per_assignment": per_assignment,
            "values": deltas,
            **_accuracy_summary(deltas),
        }
    aggregate = summary.get("aggregate", {})
    for policy in POLICIES:
        stored = aggregate.get("scheme_optimized", {}).get(policy, {})
        expected_aggregate = {
            name: recomputed["scheme_optimized"][policy][name]
            for name in ("mean", "minimum", "maximum", "sample_standard_deviation")
        }
        if stored != expected_aggregate:
            raise ValueError("Stored scheme-optimized aggregate mismatch.")
    for calibration in POLICIES:
        stored = aggregate.get("fixed_calibration", {}).get(calibration, {})
        expected_aggregate = {
            name: recomputed["fixed_calibration"][calibration][name]
            for name in ("mean", "minimum", "maximum", "sample_standard_deviation")
        }
        if stored != {
            "balanced_minus_random_accuracy": expected_aggregate,
            "per_assignment": recomputed["fixed_calibration"][calibration]["values"],
        }:
            raise ValueError("Stored fixed-calibration aggregate mismatch.")
    gate = aggregate.get("balanced_accuracy_gate", {})
    expected_gate = recomputed["scheme_optimized"][BALANCED_POLICY]["mean"] >= 0.90
    if gate != {
        "threshold": 0.90,
        "observed_mean": recomputed["scheme_optimized"][BALANCED_POLICY]["mean"],
        "passed": expected_gate,
    }:
        raise ValueError("Stored 90-percent gate mismatch.")
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
                        if arm["binding_policy"] == policy
                        and arm["calibration_policy"] == policy
                    )
                    for policy in POLICIES
                },
                "paired_binding_effect": block["paired_binding_effect"],
            }
        )
    return {
        "metric_definition_id": summary["metric_definition_id"],
        "evidence_class": summary["contract"]["device"]["evidence_class"],
        "initialization": "pre_bptt_continuous_bounded_fixed_r",
        "heldout": compact,
        "aggregate": summary["aggregate"],
        "coverage_valid": True,
        "optimizer_updates": 0,
    }


def _validate_result_metrics(bundle: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    if bundle["result"].get("metrics") != _expected_result_metrics(summary):
        raise ValueError("result.json metrics and scientific summary differ.")


def _normalize_paths(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        return {
            name: _normalize_paths(item, key=str(name))
            for name, item in sorted(value.items())
        }
    if isinstance(value, list):
        return [_normalize_paths(item, key=key) for item in value]
    if isinstance(value, str) and key in {"path", "receipt"}:
        return Path(value).name
    return value


def _artifact_replay(bundle_a, run_a: Path, bundle_b, run_b: Path) -> None:
    def keyed(bundle, run):
        material = {}
        receipts = {}
        for record in bundle["artifacts"]:
            if record["kind"] == "scientific_summary":
                continue
            key = (record["kind"], Path(record["path"]).name)
            if key in material or key in receipts:
                raise ValueError("Duplicate replay artifact kind/basename key.")
            path = run / record["path"]
            if str(record["kind"]).endswith("_receipt"):
                receipts[key] = _normalize_paths(_read_json(path))
            else:
                material[key] = (record["sha256"], record["size_bytes"])
        return material, receipts

    material_a, receipts_a = keyed(bundle_a, run_a)
    material_b, receipts_b = keyed(bundle_b, run_b)
    if material_a != material_b:
        raise ValueError("Main/replay material artifact bytes differ.")
    if receipts_a != receipts_b:
        raise ValueError("Main/replay receipt artifacts differ semantically.")


def _replay_provenance(main_dir, main_bundle, replay_dir, replay_bundle):
    if main_dir.resolve() == replay_dir.resolve():
        raise ValueError("Main and replay must be distinct native run directories.")
    if main_bundle["manifest"]["run_id"] == replay_bundle["manifest"]["run_id"]:
        raise ValueError("Main and replay must have distinct native run IDs.")
    main_context = dict(main_bundle["study_context"])
    replay_context = dict(replay_bundle["study_context"])
    if main_context.pop("arm_id") != "main" or replay_context.pop("arm_id") != "replay":
        raise ValueError("Main/replay canonical study arm IDs mismatch.")
    if main_context != replay_context:
        raise ValueError("Main/replay study, plan, or config provenance differs.")
    if main_bundle["config"] != replay_bundle["config"]:
        raise ValueError("Main/replay resolved configs differ.")
    if main_bundle["manifest"].get("source") != replay_bundle["manifest"].get("source"):
        raise ValueError("Main/replay git source identity differs.")
    if main_bundle["manifest"].get("runtime") != replay_bundle["manifest"].get("runtime"):
        raise ValueError("Main/replay deterministic runtime identity differs.")
    main_inputs = {
        role: record["sha256"] for role, record in _input_map(main_bundle).items()
    }
    replay_inputs = {
        role: record["sha256"] for role, record in _input_map(replay_bundle).items()
    }
    if main_inputs != replay_inputs:
        raise ValueError("Main/replay input content hashes differ.")
    return {
        **main_context,
        "main_run_id": main_bundle["manifest"]["run_id"],
        "replay_run_id": replay_bundle["manifest"]["run_id"],
        "distinct_run_directories": True,
        "distinct_run_ids": True,
        "source_identity_equal": True,
        "runtime_identity_equal": True,
        "input_sha256_by_role": main_inputs,
    }


def _mechanism(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    by_key = _arm_index(summary)
    assignments = []
    for block in summary["heldout"]:
        seed = block["assignment_seed"]
        policy_layers = {}
        for policy in POLICIES:
            arm = by_key[(seed, policy, policy)]
            plan_layers = block["binding_plans"][policy]["report"]["layers"]
            mapping_layers = arm["mapping"]["layers"]
            voltage = arm["ideal_bounded_continuous_test"]["voltage"]
            if not (len(plan_layers) == len(mapping_layers) == 2 and len(voltage) == 3):
                raise ValueError("Expected two mapping and three network-voltage layers.")
            policy_layers[policy] = {
                "mapping_layers": [
                    {
                        "layer": layer,
                        "r_only_zero_contrast": plan_layers[layer][
                            "baseline_contrast_after"
                        ],
                        "physical_baseline_contrast": mapping_layers[layer][
                            "baseline_contrast"
                        ],
                        "full_four_cell_loading": mapping_layers[layer][
                            "quad_loading"
                        ],
                        "common_positive_headroom": mapping_layers[layer][
                            "common_positive_headroom"
                        ],
                        "reference_outside_active_bounds_count": mapping_layers[layer][
                            "reference_outside_active_bounds_count"
                        ],
                        "continuous_logical_sign_flip_count": mapping_layers[layer][
                            "continuous_logical_sign_flip_count"
                        ],
                        "continuous_logical_sign_flip_fraction_nonzero": mapping_layers[
                            layer
                        ]["continuous_logical_sign_flip_fraction_nonzero"],
                        "float32_inward_baseline_adjustment_count": mapping_layers[
                            layer
                        ]["float32_inward_baseline_adjustment_count"],
                        "float32_inward_conductance_adjustment_count": mapping_layers[
                            layer
                        ]["float32_inward_conductance_adjustment_count"],
                        "float32_inward_conductance_max_abs_adjustment": mapping_layers[
                            layer
                        ]["float32_inward_conductance_max_abs_adjustment"],
                    }
                    for layer in range(2)
                ],
                "network_voltage": voltage,
            }
        layer_records = []
        for layer in range(2):
            random_rms = policy_layers[RANDOM_POLICY]["mapping_layers"][layer][
                "r_only_zero_contrast"
            ]["rms"]
            balanced_rms = policy_layers[BALANCED_POLICY]["mapping_layers"][layer][
                "r_only_zero_contrast"
            ]["rms"]
            layer_records.append(
                {
                    "layer": layer,
                    "balanced_over_random_r_only_zero_contrast_rms": (
                        balanced_rms / random_rms if random_rms > 0.0 else None
                    ),
                    "bindings": {
                        policy: policy_layers[policy]["mapping_layers"][layer]
                        for policy in POLICIES
                    },
                }
            )
        assignments.append(
            {
                "assignment_seed": seed,
                "layers": layer_records,
                "network_voltage_by_binding": {
                    policy: policy_layers[policy]["network_voltage"]
                    for policy in POLICIES
                },
            }
        )
    return {"assignments": assignments}


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
    for run_dir, bundle, summary in (
        (main_dir, main_bundle, main_summary),
        (replay_dir, replay_bundle, replay_summary),
    ):
        by_kind = _artifact_by_kind(bundle, run_dir)
        _require_artifact_contract(by_kind)
        populations = _population_map(
            by_kind["ibm_om_population"],
            by_kind["ibm_om_population_receipt"],
            summary,
        )
        logical_weights = _source_logical_weights(bundle, summary, populations)
        development = _validate_development(summary, bundle["config"])
        plans = _validate_plans(
            by_kind["ibm_om_reference_binding_plan"],
            by_kind["ibm_om_reference_binding_plan_receipt"],
            populations,
            summary,
        )
        mapping_index = _summary_mapping_index(summary, development)
        student = bundle["config"]["student"]
        _validate_mappings(
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
    if _normalize_paths(main_summary) != _normalize_paths(replay_summary):
        raise ValueError("Main/replay scientific summaries differ semantically.")
    _artifact_replay(main_bundle, main_dir, replay_bundle, replay_dir)
    metrics = _metrics(main_summary)
    mechanism = _mechanism(main_summary)
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
        "gate": main_summary["aggregate"]["balanced_accuracy_gate"],
        "claim_boundary": main_summary["claim_boundary"],
    }


def _report(analysis: Mapping[str, Any]) -> str:
    optimized = analysis["accuracy"]["scheme_optimized"]
    fixed = analysis["accuracy"]["fixed_calibration"]
    lines = [
        "# Four-reference balance post-run analysis",
        "",
        f"Status: `{analysis['status']}`",
        "",
        "## Ideal bounded continuous initialization",
        "",
        "| Binding | Mean | Range | Exact held-out results |",
        "|---|---:|---:|---|",
    ]
    for policy in POLICIES:
        item = optimized[policy]
        values = ", ".join(
            f"{record['assignment_seed']}: {record['correct']}/{record['examples']} "
            f"({100.0 * record['accuracy']:.2f}%)"
            for record in item["per_assignment"]
        )
        lines.append(
            f"| `{policy}` | {100.0 * item['mean']:.2f}% | "
            f"{100.0 * item['minimum']:.2f}–{100.0 * item['maximum']:.2f}% | {values} |"
        )
    lines.extend(
        [
            "",
            "## Fixed-calibration binding effect",
            "",
            "| Calibration | Mean balanced−random | Range | Per assignment |",
            "|---|---:|---:|---|",
        ]
    )
    for policy in POLICIES:
        item = fixed[policy]
        per_assignment = ", ".join(
            f"{record['assignment_seed']}: {record['balanced_correct']}−"
            f"{record['random_correct']} "
            f"({100.0 * record['balanced_minus_random_accuracy']:+.2f} pp; "
            f"{record['prediction_flip_count']} flips)"
            for record in item["per_assignment"]
        )
        lines.append(
            f"| `{policy}` | {100.0 * item['mean']:+.2f} pp | "
            f"{100.0 * item['minimum']:+.2f} to {100.0 * item['maximum']:+.2f} pp | "
            f"{per_assignment} |"
        )
    lines.extend(
        [
            "",
            "## Per-assignment physical mechanism",
            "",
            (
                "| Assignment | Layer | Binding | r-only contrast RMS | "
                "Physical B contrast RMS | Loading mean | Headroom mean | "
                "r outside bounds | Sign flips | Sign-flip fraction | B inward | "
                "G inward | Max G inward adjustment | Balanced/random r-RMS |"
            ),
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for assignment in analysis["mechanism"]["assignments"]:
        for layer in assignment["layers"]:
            ratio = layer["balanced_over_random_r_only_zero_contrast_rms"]
            for policy in POLICIES:
                item = layer["bindings"][policy]
                lines.append(
                    f"| {assignment['assignment_seed']} | {layer['layer']} | `{policy}` | "
                    f"{item['r_only_zero_contrast']['rms']:.8g} | "
                    f"{item['physical_baseline_contrast']['rms']:.8g} | "
                    f"{item['full_four_cell_loading']['mean']:.8g} | "
                    f"{item['common_positive_headroom']['mean']:.8g} | "
                    f"{item['reference_outside_active_bounds_count']} | "
                    f"{item['continuous_logical_sign_flip_count']} | "
                    f"{item['continuous_logical_sign_flip_fraction_nonzero']:.8g} | "
                    f"{item['float32_inward_baseline_adjustment_count']} | "
                    f"{item['float32_inward_conductance_adjustment_count']} | "
                    f"{item['float32_inward_conductance_max_abs_adjustment']:.8g} | "
                    f"{ratio:.8g} |"
                )
    lines.extend(
        [
            "",
            "## Per-assignment network voltage",
            "",
            (
                "| Assignment | Binding | Network layer | Values | RMS | Mean | "
                "Standard deviation | Range |"
            ),
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for assignment in analysis["mechanism"]["assignments"]:
        for policy in POLICIES:
            for voltage in assignment["network_voltage_by_binding"][policy]:
                lines.append(
                    f"| {assignment['assignment_seed']} | `{policy}` | "
                    f"{voltage['layer']} | {voltage['values']} | "
                    f"{voltage['rms']:.8g} | {voltage['mean']:.8g} | "
                    f"{voltage['standard_deviation']:.8g} | "
                    f"{voltage['minimum']:.8g} to {voltage['maximum']:.8g} |"
                )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"Balanced held-out mean >= 90%: **{analysis['gate']['passed']}** "
            f"(observed {100.0 * analysis['gate']['observed_mean']:.2f}%).",
            "",
            "Every registered artifact, binding permutation, B+d decomposition, "
            "full-G contrast/loading tensor, active bound, exact accuracy count, "
            "aggregate, and main/replay semantic value passed validation.",
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
