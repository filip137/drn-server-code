"""Scientific study contract independent of any execution attempt."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .common import (
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    load_json,
    reject_placeholders,
    require,
    require_bool,
    require_exact_keys,
    require_id,
    require_nonnegative_int,
    require_positive_int,
    require_sha256,
    require_string_list,
    require_text,
    require_timezone_timestamp,
    resolve_pointer,
    safe_relative_path,
)


STUDY_SCHEMA = "experiment-study/v1"
EVIDENCE_SCOPES = {"paper_facing", "diagnostic", "historical_replay"}
APPROVAL_STATES = {"draft", "review_required", "approved"}


def load_study(path: str | Path) -> dict[str, Any]:
    return load_json(Path(path).expanduser().resolve())


def study_spec_sha256(value: Mapping[str, Any]) -> str:
    study = value.get("study") if "study" in value else value
    return canonical_sha256(study)


def study_bindings_sha256(value: Mapping[str, Any]) -> str:
    bindings = value.get("bindings") if "bindings" in value else value
    return canonical_sha256(bindings)


def _hypothesis(value: Any) -> dict[str, Any]:
    stage = "study_validation"
    result = require_exact_keys(
        value,
        keys={
            "statement",
            "control",
            "treatments",
            "expected_direction",
            "decision_rule",
            "supports_if",
            "does_not_support_if",
            "inconclusive_if",
        },
        stage=stage,
        path="$.study.hypothesis",
    )
    for key in (
        "statement",
        "control",
        "expected_direction",
        "supports_if",
        "does_not_support_if",
        "inconclusive_if",
    ):
        require_text(
            result[key],
            stage=stage,
            path=f"$.study.hypothesis.{key}",
        )
    require_string_list(
        result["treatments"],
        stage=stage,
        path="$.study.hypothesis.treatments",
    )
    decision = require_exact_keys(
        result["decision_rule"],
        keys={
            "metric",
            "split",
            "checkpoint_role",
            "seed_aggregation",
            "comparison",
            "minimum_meaningful_effect",
            "effect_units",
        },
        stage=stage,
        path="$.study.hypothesis.decision_rule",
    )
    for key in (
        "metric",
        "split",
        "checkpoint_role",
        "seed_aggregation",
        "comparison",
        "effect_units",
    ):
        require_text(
            decision[key],
            stage=stage,
            path=f"$.study.hypothesis.decision_rule.{key}",
        )
    effect = decision["minimum_meaningful_effect"]
    require(
        isinstance(effect, (int, float))
        and not isinstance(effect, bool)
        and math.isfinite(float(effect))
        and effect >= 0,
        stage=stage,
        code="invalid_effect_threshold",
        path=(
            "$.study.hypothesis.decision_rule.minimum_meaningful_effect"
        ),
        expected="a non-negative numeric effect threshold",
        provided=effect,
    )
    return result


def _design(value: Any) -> dict[str, Any]:
    stage = "study_validation"
    result = require_exact_keys(
        value,
        keys={
            "artifacts",
            "axes",
            "case_ids",
            "seeds",
            "conditional_expansion",
            "expected_initial_job_count",
            "maximum_total_job_count",
            "protocols",
        },
        stage=stage,
        path="$.study.design",
    )
    artifacts = require_exact_keys(
        result["artifacts"],
        keys={
            "scientific_config_sha256",
            "study_manifest_sha256",
            "manifest_job_count_pointer",
        },
        stage=stage,
        path="$.study.design.artifacts",
    )
    require_sha256(
        artifacts["scientific_config_sha256"],
        stage=stage,
        path="$.study.design.artifacts.scientific_config_sha256",
    )
    require_sha256(
        artifacts["study_manifest_sha256"],
        stage=stage,
        path="$.study.design.artifacts.study_manifest_sha256",
    )
    pointer = require_text(
        artifacts["manifest_job_count_pointer"],
        stage=stage,
        path="$.study.design.artifacts.manifest_job_count_pointer",
    )
    require(
        pointer.startswith("/"),
        stage=stage,
        code="invalid_json_pointer",
        path="$.study.design.artifacts.manifest_job_count_pointer",
        expected="a JSON pointer beginning with '/'",
        provided=pointer,
    )

    axes = result["axes"]
    require(
        isinstance(axes, list) and bool(axes),
        stage=stage,
        code="missing_sweep_axis",
        path="$.study.design.axes",
        expected="at least one sweep axis",
        provided=axes,
    )
    axis_names: set[str] = set()
    for index, raw_axis in enumerate(axes):
        axis_path = f"$.study.design.axes[{index}]"
        axis = require_exact_keys(
            raw_axis,
            keys={"name", "config_path", "ordered_values"},
            stage=stage,
            path=axis_path,
        )
        name = require_id(
            axis["name"],
            stage=stage,
            path=f"{axis_path}.name",
        )
        require(
            name not in axis_names,
            stage=stage,
            code="duplicate_axis",
            path=f"{axis_path}.name",
            expected="a unique sweep-axis name",
            provided=name,
        )
        axis_names.add(name)
        config_pointer = require_text(
            axis["config_path"],
            stage=stage,
            path=f"{axis_path}.config_path",
        )
        require(
            config_pointer.startswith("/"),
            stage=stage,
            code="invalid_json_pointer",
            path=f"{axis_path}.config_path",
            expected="a JSON pointer beginning with '/'",
            provided=config_pointer,
        )
        values = axis["ordered_values"]
        require(
            isinstance(values, list) and bool(values),
            stage=stage,
            code="missing_axis_values",
            path=f"{axis_path}.ordered_values",
            expected="a non-empty ordered value list",
            provided=values,
        )
        canonical = [canonical_bytes(item) for item in values]
        require(
            len(canonical) == len(set(canonical)),
            stage=stage,
            code="duplicate_axis_value",
            path=f"{axis_path}.ordered_values",
            expected="unique ordered values",
            provided=values,
        )

    case_ids = require_string_list(
        result["case_ids"],
        stage=stage,
        path="$.study.design.case_ids",
    )
    for index, case_id in enumerate(case_ids):
        require_id(
            case_id,
            stage=stage,
            path=f"$.study.design.case_ids[{index}]",
        )
    seeds = result["seeds"]
    require(
        isinstance(seeds, list)
        and bool(seeds)
        and all(
            isinstance(seed, int)
            and not isinstance(seed, bool)
            and seed >= 0
            for seed in seeds
        )
        and len(seeds) == len(set(seeds)),
        stage=stage,
        code="invalid_seeds",
        path="$.study.design.seeds",
        expected="unique non-negative integer seeds",
        provided=seeds,
    )
    expansion = require_exact_keys(
        result["conditional_expansion"],
        keys={"enabled", "rule", "maximum_additional_jobs"},
        stage=stage,
        path="$.study.design.conditional_expansion",
    )
    enabled = require_bool(
        expansion["enabled"],
        stage=stage,
        path="$.study.design.conditional_expansion.enabled",
    )
    maximum_additional = require_nonnegative_int(
        expansion["maximum_additional_jobs"],
        stage=stage,
        path=(
            "$.study.design.conditional_expansion.maximum_additional_jobs"
        ),
    )
    if enabled:
        require_text(
            expansion["rule"],
            stage=stage,
            path="$.study.design.conditional_expansion.rule",
        )
    else:
        require(
            expansion["rule"] is None and maximum_additional == 0,
            stage=stage,
            code="disabled_expansion_not_null",
            path="$.study.design.conditional_expansion",
            expected="null rule and zero additional jobs when disabled",
            provided=expansion,
        )
    initial = require_positive_int(
        result["expected_initial_job_count"],
        stage=stage,
        path="$.study.design.expected_initial_job_count",
    )
    maximum = require_positive_int(
        result["maximum_total_job_count"],
        stage=stage,
        path="$.study.design.maximum_total_job_count",
    )
    require(
        maximum == initial + maximum_additional,
        stage=stage,
        code="job_cap_mismatch",
        path="$.study.design.maximum_total_job_count",
        expected=(
            "initial job count plus maximum approved additional jobs"
        ),
        provided=maximum,
    )

    protocols = result["protocols"]
    require(
        isinstance(protocols, list),
        stage=stage,
        code="wrong_type",
        path="$.study.design.protocols",
        expected="a protocol identity list",
        provided=protocols,
    )
    protocol_ids: set[str] = set()
    for index, raw_protocol in enumerate(protocols):
        protocol_path = f"$.study.design.protocols[{index}]"
        protocol = require_exact_keys(
            raw_protocol,
            keys={"authority_id", "sha256"},
            stage=stage,
            path=protocol_path,
        )
        authority_id = require_id(
            protocol["authority_id"],
            stage=stage,
            path=f"{protocol_path}.authority_id",
        )
        require(
            authority_id not in protocol_ids,
            stage=stage,
            code="duplicate_protocol_id",
            path=f"{protocol_path}.authority_id",
            expected="a unique protocol authority_id",
            provided=authority_id,
        )
        protocol_ids.add(authority_id)
        require_sha256(
            protocol["sha256"],
            stage=stage,
            path=f"{protocol_path}.sha256",
        )
    return result


def _reporting(value: Any) -> dict[str, Any]:
    stage = "study_validation"
    result = require_exact_keys(
        value,
        keys={"comparisons"},
        stage=stage,
        path="$.study.reporting",
    )
    comparisons = result["comparisons"]
    require(
        isinstance(comparisons, list) and bool(comparisons),
        stage=stage,
        code="missing_comparison",
        path="$.study.reporting.comparisons",
        expected="at least one planned comparison",
        provided=comparisons,
    )
    comparison_ids: set[str] = set()
    for index, raw_comparison in enumerate(comparisons):
        comparison_path = f"$.study.reporting.comparisons[{index}]"
        comparison = require_exact_keys(
            raw_comparison,
            keys={"comparison_id", "card_schema_version"},
            stage=stage,
            path=comparison_path,
        )
        comparison_id = require_id(
            comparison["comparison_id"],
            stage=stage,
            path=f"{comparison_path}.comparison_id",
        )
        require(
            comparison_id not in comparison_ids,
            stage=stage,
            code="duplicate_comparison_id",
            path=f"{comparison_path}.comparison_id",
            expected="a unique comparison_id",
            provided=comparison_id,
        )
        comparison_ids.add(comparison_id)
        require_text(
            comparison["card_schema_version"],
            stage=stage,
            path=f"{comparison_path}.card_schema_version",
        )
    return result


def _acceptance(value: Any) -> dict[str, Any]:
    stage = "study_validation"
    result = require_exact_keys(
        value,
        keys={
            "required_coverage",
            "allowed_exclusions",
            "failure_handling",
            "required_scientific_gates",
            "local_validation_required",
            "review_required",
        },
        stage=stage,
        path="$.study.acceptance",
    )
    require_text(
        result["required_coverage"],
        stage=stage,
        path="$.study.acceptance.required_coverage",
    )
    require_string_list(
        result["allowed_exclusions"],
        stage=stage,
        path="$.study.acceptance.allowed_exclusions",
        allow_empty=True,
    )
    require_text(
        result["failure_handling"],
        stage=stage,
        path="$.study.acceptance.failure_handling",
    )
    gates = require_string_list(
        result["required_scientific_gates"],
        stage=stage,
        path="$.study.acceptance.required_scientific_gates",
        allow_empty=True,
    )
    for index, gate in enumerate(gates):
        require_id(
            gate,
            stage=stage,
            path=f"$.study.acceptance.required_scientific_gates[{index}]",
        )
    for key in ("local_validation_required", "review_required"):
        require(
            require_bool(
                result[key],
                stage=stage,
                path=f"$.study.acceptance.{key}",
            )
            is True,
            stage=stage,
            code="required_true",
            path=f"$.study.acceptance.{key}",
            expected="true",
            provided=result[key],
        )
    return result


def _study(value: Any) -> dict[str, Any]:
    stage = "study_validation"
    result = require_exact_keys(
        value,
        keys={
            "study_id",
            "title",
            "evidence_scope",
            "hypothesis",
            "design",
            "acceptance",
            "reporting",
        },
        stage=stage,
        path="$.study",
    )
    require_id(result["study_id"], stage=stage, path="$.study.study_id")
    require_text(result["title"], stage=stage, path="$.study.title")
    require(
        result["evidence_scope"] in EVIDENCE_SCOPES,
        stage=stage,
        code="invalid_evidence_scope",
        path="$.study.evidence_scope",
        expected=f"one of {sorted(EVIDENCE_SCOPES)}",
        provided=result["evidence_scope"],
    )
    _hypothesis(result["hypothesis"])
    _design(result["design"])
    _acceptance(result["acceptance"])
    _reporting(result["reporting"])
    return result


def _bindings(
    value: Any,
    *,
    study: Mapping[str, Any],
) -> dict[str, Any]:
    stage = "study_binding_validation"
    result = require_exact_keys(
        value,
        keys={
            "scientific_config_path",
            "study_manifest_path",
            "protocols",
            "reporting",
        },
        stage=stage,
        path="$.bindings",
    )
    safe_relative_path(
        result["scientific_config_path"],
        stage=stage,
        path="$.bindings.scientific_config_path",
    )
    safe_relative_path(
        result["study_manifest_path"],
        stage=stage,
        path="$.bindings.study_manifest_path",
    )
    protocol_bindings = result["protocols"]
    require(
        isinstance(protocol_bindings, list),
        stage=stage,
        code="wrong_type",
        path="$.bindings.protocols",
        expected="a protocol binding list",
        provided=protocol_bindings,
    )
    observed_protocols: dict[str, str] = {}
    for index, raw_binding in enumerate(protocol_bindings):
        binding_path = f"$.bindings.protocols[{index}]"
        binding = require_exact_keys(
            raw_binding,
            keys={"authority_id", "path"},
            stage=stage,
            path=binding_path,
        )
        authority_id = require_id(
            binding["authority_id"],
            stage=stage,
            path=f"{binding_path}.authority_id",
        )
        require(
            authority_id not in observed_protocols,
            stage=stage,
            code="duplicate_protocol_binding",
            path=f"{binding_path}.authority_id",
            expected="a unique protocol binding",
            provided=authority_id,
        )
        protocol_path = safe_relative_path(
            binding["path"],
            stage=stage,
            path=f"{binding_path}.path",
        )
        observed_protocols[authority_id] = str(protocol_path)
    expected_protocols = {
        item["authority_id"] for item in study["design"]["protocols"]
    }
    require(
        set(observed_protocols) == expected_protocols,
        stage=stage,
        code="protocol_binding_set_mismatch",
        path="$.bindings.protocols",
        expected="one binding for every study protocol identity",
        provided={
            "missing": sorted(expected_protocols - set(observed_protocols)),
            "unknown": sorted(set(observed_protocols) - expected_protocols),
        },
    )

    reporting = require_exact_keys(
        result["reporting"],
        keys={"comparisons", "final_results_page"},
        stage=stage,
        path="$.bindings.reporting",
    )
    final_page = safe_relative_path(
        reporting["final_results_page"],
        stage=stage,
        path="$.bindings.reporting.final_results_page",
    )
    require(
        final_page == Path("docs/results/index.md"),
        stage=stage,
        code="noncanonical_results_page",
        path="$.bindings.reporting.final_results_page",
        expected="docs/results/index.md",
        provided=str(final_page),
    )
    comparisons = reporting["comparisons"]
    require(
        isinstance(comparisons, list),
        stage=stage,
        code="wrong_type",
        path="$.bindings.reporting.comparisons",
        expected="a comparison binding list",
        provided=comparisons,
    )
    bound_comparisons: set[str] = set()
    for index, raw_comparison in enumerate(comparisons):
        comparison_path = f"$.bindings.reporting.comparisons[{index}]"
        comparison = require_exact_keys(
            raw_comparison,
            keys={
                "comparison_id",
                "card_path",
                "review_path",
                "final_results_anchor",
            },
            stage=stage,
            path=comparison_path,
        )
        comparison_id = require_id(
            comparison["comparison_id"],
            stage=stage,
            path=f"{comparison_path}.comparison_id",
        )
        require(
            comparison_id not in bound_comparisons,
            stage=stage,
            code="duplicate_comparison_binding",
            path=f"{comparison_path}.comparison_id",
            expected="a unique comparison binding",
            provided=comparison_id,
        )
        bound_comparisons.add(comparison_id)
        card_path = safe_relative_path(
            comparison["card_path"],
            stage=stage,
            path=f"{comparison_path}.card_path",
        )
        review_path = safe_relative_path(
            comparison["review_path"],
            stage=stage,
            path=f"{comparison_path}.review_path",
        )
        require(
            card_path.parts[:2] == ("result_registry", "cards")
            and card_path.suffix == ".json",
            stage=stage,
            code="invalid_card_path",
            path=f"{comparison_path}.card_path",
            expected="a JSON path below result_registry/cards",
            provided=str(card_path),
        )
        require(
            review_path.parts[:2] == ("result_registry", "reviews")
            and review_path.suffix == ".json",
            stage=stage,
            code="invalid_review_path",
            path=f"{comparison_path}.review_path",
            expected="a JSON path below result_registry/reviews",
            provided=str(review_path),
        )
        require(
            comparison["final_results_anchor"]
            == f"docs/results/index.md#{comparison_id}",
            stage=stage,
            code="invalid_results_anchor",
            path=f"{comparison_path}.final_results_anchor",
            expected=f"docs/results/index.md#{comparison_id}",
            provided=comparison["final_results_anchor"],
        )
    expected_comparisons = {
        item["comparison_id"] for item in study["reporting"]["comparisons"]
    }
    require(
        bound_comparisons == expected_comparisons,
        stage=stage,
        code="comparison_binding_set_mismatch",
        path="$.bindings.reporting.comparisons",
        expected="one binding for every planned comparison",
        provided={
            "missing": sorted(expected_comparisons - bound_comparisons),
            "unknown": sorted(bound_comparisons - expected_comparisons),
        },
    )
    return result


def _approval(
    value: Any,
    *,
    expected_study_sha256: str,
) -> dict[str, Any]:
    stage = "approval_validation"
    result = require_exact_keys(
        value,
        keys={
            "status",
            "approved_by",
            "approved_at",
            "approved_study_sha256",
            "amendment_of",
        },
        stage=stage,
        path="$.approval",
    )
    status = result["status"]
    require(
        status in APPROVAL_STATES,
        stage=stage,
        code="invalid_approval_state",
        path="$.approval.status",
        expected=f"one of {sorted(APPROVAL_STATES)}",
        provided=status,
    )
    amendment = result["amendment_of"]
    if amendment is not None:
        require_id(
            amendment,
            stage=stage,
            path="$.approval.amendment_of",
        )
    if status == "approved":
        require_text(
            result["approved_by"],
            stage=stage,
            path="$.approval.approved_by",
        )
        require_timezone_timestamp(
            result["approved_at"],
            stage=stage,
            path="$.approval.approved_at",
        )
        approved_hash = require_sha256(
            result["approved_study_sha256"],
            stage=stage,
            path="$.approval.approved_study_sha256",
        )
        require(
            approved_hash == expected_study_sha256,
            stage=stage,
            code="approval_hash_mismatch",
            path="$.approval.approved_study_sha256",
            expected=(
                "the canonical SHA-256 of the scientific study object"
            ),
            provided=approved_hash,
        )
    else:
        for key in ("approved_by", "approved_at", "approved_study_sha256"):
            require(
                result[key] is None,
                stage=stage,
                code="unapproved_field_not_null",
                path=f"$.approval.{key}",
                expected=f"null while approval status is {status!r}",
                provided=result[key],
            )
    return result


def _provenance(value: Any) -> dict[str, Any]:
    stage = "study_validation"
    result = require_exact_keys(
        value,
        keys={"origin", "source_path", "source_sha256", "notes"},
        stage=stage,
        path="$.provenance",
    )
    require_id(
        result["origin"],
        stage=stage,
        path="$.provenance.origin",
    )
    source_path = result["source_path"]
    source_hash = result["source_sha256"]
    require(
        (source_path is None and source_hash is None)
        or (
            isinstance(source_path, str)
            and bool(source_path)
            and isinstance(source_hash, str)
            and len(source_hash) == 64
        ),
        stage=stage,
        code="invalid_provenance_source",
        path="$.provenance",
        expected="both source_path/source_sha256 or both null",
        provided={
            "source_path": source_path,
            "source_sha256": source_hash,
        },
    )
    if source_hash is not None:
        safe_relative_path(
            source_path,
            stage=stage,
            path="$.provenance.source_path",
        )
        require_sha256(
            source_hash,
            stage=stage,
            path="$.provenance.source_sha256",
        )
    require_string_list(
        result["notes"],
        stage=stage,
        path="$.provenance.notes",
        allow_empty=True,
    )
    return result


def _verify_bindings(
    document: Mapping[str, Any],
    *,
    repo_root: Path,
) -> None:
    stage = "binding_verification"
    study = document["study"]
    design = study["design"]
    artifacts = design["artifacts"]
    bindings = document["bindings"]
    for label, path_key, hash_key in (
        (
            "scientific config",
            "scientific_config_path",
            "scientific_config_sha256",
        ),
        (
            "study manifest",
            "study_manifest_path",
            "study_manifest_sha256",
        ),
    ):
        relative = safe_relative_path(
            bindings[path_key],
            stage=stage,
            path=f"$.bindings.{path_key}",
        )
        absolute = repo_root / relative
        require(
            absolute.is_file() and not absolute.is_symlink(),
            stage=stage,
            code="bound_file_missing",
            path=f"$.bindings.{path_key}",
            expected=f"an existing non-symlink {label} file",
            provided=str(relative),
        )
        observed = file_sha256(absolute)
        require(
            observed == artifacts[hash_key],
            stage=stage,
            code="bound_file_hash_mismatch",
            path=f"$.bindings.{path_key}",
            expected=artifacts[hash_key],
            provided=observed,
        )

    manifest_path = repo_root / Path(bindings["study_manifest_path"])
    manifest = load_json(manifest_path)
    count_value = resolve_pointer(
        manifest,
        artifacts["manifest_job_count_pointer"],
        path="$.study.design.artifacts.manifest_job_count_pointer",
    )
    if isinstance(count_value, list):
        observed_count = len(count_value)
    else:
        observed_count = require_nonnegative_int(
            count_value,
            stage=stage,
            path="manifest job-count pointer result",
        )
    require(
        observed_count == design["expected_initial_job_count"],
        stage=stage,
        code="manifest_job_count_mismatch",
        path="$.bindings.study_manifest_path",
        expected=(
            f"{design['expected_initial_job_count']} initial manifest jobs"
        ),
        provided=observed_count,
    )

    protocol_specs = {
        item["authority_id"]: item["sha256"]
        for item in design["protocols"]
    }
    for index, binding in enumerate(bindings["protocols"]):
        relative = Path(binding["path"])
        absolute = repo_root / relative
        require(
            absolute.is_file() and not absolute.is_symlink(),
            stage=stage,
            code="bound_protocol_missing",
            path=f"$.bindings.protocols[{index}].path",
            expected="an existing non-symlink protocol file",
            provided=str(relative),
        )
        observed = file_sha256(absolute)
        expected = protocol_specs[binding["authority_id"]]
        require(
            observed == expected,
            stage=stage,
            code="bound_protocol_hash_mismatch",
            path=f"$.bindings.protocols[{index}].path",
            expected=expected,
            provided=observed,
        )


def validate_study(
    document: Mapping[str, Any],
    *,
    repo_root: str | Path,
    verify_bindings: bool = False,
    require_approved: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    top = require_exact_keys(
        document,
        keys={"schema_version", "study", "bindings", "approval", "provenance"},
        stage="study_validation",
        path="$",
    )
    require(
        top["schema_version"] == STUDY_SCHEMA,
        stage="study_validation",
        code="unsupported_schema",
        path="$.schema_version",
        expected=STUDY_SCHEMA,
        provided=top["schema_version"],
    )
    study = _study(top["study"])
    bindings = _bindings(top["bindings"], study=study)
    spec_hash = study_spec_sha256(top)
    binding_hash = study_bindings_sha256(top)
    approval = _approval(
        top["approval"],
        expected_study_sha256=spec_hash,
    )
    _provenance(top["provenance"])
    if require_approved:
        require(
            approval["status"] == "approved",
            stage="approval_validation",
            code="study_not_approved",
            path="$.approval.status",
            expected="approved",
            provided=approval["status"],
        )
    if approval["status"] == "approved":
        reject_placeholders(top)
    if verify_bindings or require_approved:
        _verify_bindings(top, repo_root=root)
    return {
        "schema_version": STUDY_SCHEMA,
        "status": "passed",
        "study_id": study["study_id"],
        "approval_status": approval["status"],
        "study_spec_sha256": spec_hash,
        "study_bindings_sha256": binding_hash,
        "scientific_config_sha256": study["design"]["artifacts"][
            "scientific_config_sha256"
        ],
        "study_manifest_sha256": study["design"]["artifacts"][
            "study_manifest_sha256"
        ],
        "expected_initial_job_count": study["design"][
            "expected_initial_job_count"
        ],
    }
