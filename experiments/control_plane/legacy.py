"""Deterministic split of legacy mixed experiment plans.

The converter preserves the old execution material as explicitly
non-authoritative residue.  It never transfers legacy approval into the new
scientific-study approval field and never fabricates an execution attempt.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Mapping

from .common import (
    file_sha256,
    loads_json,
    require,
    require_bool,
    require_exact_keys,
    require_id,
    require_string_list,
    safe_relative_path,
    write_json_first,
)
from .study import STUDY_SCHEMA, study_spec_sha256, validate_study


LEGACY_PLAN_SCHEMA = "experiment-run-plan/v1"
LEGACY_RESIDUE_SCHEMA = "experiment-legacy-execution-residue/v1"
_JSON_FENCE_RE = re.compile(r"```json[ \t]*\r?\n(.*?)\r?\n```", re.DOTALL)


def load_legacy_plan(path: str | Path) -> dict[str, Any]:
    """Read the single JSON contract embedded in a legacy Markdown plan."""

    source = Path(path).expanduser().resolve()
    try:
        payload = source.read_text(encoding="utf-8")
    except OSError as exc:
        from .common import fail

        fail(
            stage="legacy_plan_loading",
            code="unreadable_legacy_plan",
            path=str(source),
            expected="a readable UTF-8 Markdown plan",
            provided=str(exc),
        )
    blocks = _JSON_FENCE_RE.findall(payload)
    require(
        len(blocks) == 1,
        stage="legacy_plan_loading",
        code="legacy_json_block_count",
        path=str(source),
        expected="exactly one fenced json contract",
        provided=len(blocks),
    )
    return loads_json(blocks[0], source=f"{source} fenced json")


def _validate_legacy_shape(plan: Any) -> dict[str, Any]:
    """Reject unknown legacy fields so migration cannot silently drop them."""

    stage = "legacy_plan_validation"
    top = require_exact_keys(
        plan,
        keys={
            "schema_version",
            "experiment_id",
            "title",
            "evidence_scope",
            "hypothesis",
            "sweep",
            "execution",
            "storage",
            "reporting",
            "completion",
            "approval",
        },
        stage=stage,
        path="$",
    )
    require(
        top["schema_version"] == LEGACY_PLAN_SCHEMA,
        stage=stage,
        code="unsupported_legacy_schema",
        path="$.schema_version",
        expected=LEGACY_PLAN_SCHEMA,
        provided=top["schema_version"],
    )
    require_id(
        top["experiment_id"],
        stage=stage,
        path="$.experiment_id",
    )
    require_exact_keys(
        top["hypothesis"],
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
        path="$.hypothesis",
    )
    require_exact_keys(
        top["hypothesis"]["decision_rule"],
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
        path="$.hypothesis.decision_rule",
    )
    sweep = require_exact_keys(
        top["sweep"],
        keys={
            "config_path",
            "config_sha256",
            "manifest_path",
            "manifest_sha256",
            "manifest_job_count_pointer",
            "axes",
            "case_ids",
            "seeds",
            "conditional_expansion",
            "expected_initial_job_count",
            "maximum_total_job_count",
            "undeclared_fields_fixed_by",
        },
        stage=stage,
        path="$.sweep",
    )
    require_string_list(
        sweep["undeclared_fields_fixed_by"],
        stage=stage,
        path="$.sweep.undeclared_fields_fixed_by",
        allow_empty=True,
    )
    execution = require_exact_keys(
        top["execution"],
        keys={"launcher", "collector", "validator", "preflight"},
        stage=stage,
        path="$.execution",
    )
    preflight = require_exact_keys(
        execution["preflight"],
        keys={
            "smoke_required",
            "tk_reference_required",
            "scheduled_run_preflight_required",
            "receipt_path",
        },
        stage=stage,
        path="$.execution.preflight",
    )
    for key in (
        "smoke_required",
        "tk_reference_required",
        "scheduled_run_preflight_required",
    ):
        require_bool(
            preflight[key],
            stage=stage,
            path=f"$.execution.preflight.{key}",
        )
    storage = require_exact_keys(
        top["storage"],
        keys={
            "root_alias",
            "local_results_root",
            "local_bundle_path",
            "remote_staging",
        },
        stage=stage,
        path="$.storage",
    )
    require(
        isinstance(storage["remote_staging"], list),
        stage=stage,
        code="wrong_type",
        path="$.storage.remote_staging",
        expected="a list",
        provided=storage["remote_staging"],
    )
    for index, entry in enumerate(storage["remote_staging"]):
        require_exact_keys(
            entry,
            keys={"host", "path", "intended_local_destination"},
            stage=stage,
            path=f"$.storage.remote_staging[{index}]",
        )
    reporting = require_exact_keys(
        top["reporting"],
        keys={"progress_tracker", "comparison_cards", "final_results_page"},
        stage=stage,
        path="$.reporting",
    )
    require(
        isinstance(reporting["comparison_cards"], list),
        stage=stage,
        code="wrong_type",
        path="$.reporting.comparison_cards",
        expected="a list",
        provided=reporting["comparison_cards"],
    )
    for index, card in enumerate(reporting["comparison_cards"]):
        require_exact_keys(
            card,
            keys={
                "comparison_id",
                "card_schema_version",
                "card_path",
                "review_path",
                "final_results_anchor",
            },
            stage=stage,
            path=f"$.reporting.comparison_cards[{index}]",
        )
    require_exact_keys(
        top["completion"],
        keys={
            "required_coverage",
            "allowed_exclusions",
            "failure_handling",
            "local_validation_required",
            "review_required",
        },
        stage=stage,
        path="$.completion",
    )
    approval = require_exact_keys(
        top["approval"],
        keys={"status", "approved_by", "approved_at", "amendment_of"},
        stage=stage,
        path="$.approval",
    )
    require(
        approval["status"] in {"draft", "approved"},
        stage=stage,
        code="invalid_legacy_approval_status",
        path="$.approval.status",
        expected="draft or approved",
        provided=approval["status"],
    )
    return top


def _authority_ids(paths: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    used: set[str] = set()
    for value in paths:
        stem = re.sub(r"[^a-z0-9._-]+", "-", Path(value).stem.lower())
        base = f"protocol-{stem}".strip("-")
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}-{suffix}"
            suffix += 1
        used.add(candidate)
        result[value] = candidate
    return result


def split_legacy_plan(
    path: str | Path,
    *,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a review-required study and non-authoritative residue."""

    root = Path(repo_root).expanduser().resolve()
    source = Path(path).expanduser().resolve()
    try:
        source_relative = source.relative_to(root)
    except ValueError:
        require(
            False,
            stage="legacy_plan_loading",
            code="legacy_plan_outside_repo",
            path=str(source),
            expected=f"a plan below repository root {root}",
            provided=str(source),
        )
        raise AssertionError("unreachable")
    safe_relative_path(
        str(source_relative),
        stage="legacy_plan_loading",
        path="legacy source path",
    )
    plan = _validate_legacy_shape(load_legacy_plan(source))
    sweep = plan["sweep"]
    reporting = plan["reporting"]
    completion = plan["completion"]
    preflight = plan["execution"]["preflight"]

    fixed_by = list(sweep["undeclared_fields_fixed_by"])
    scientific_protocol_paths = [
        value
        for value in fixed_by
        if Path(value).parts and Path(value).parts[0] == "docs"
    ]
    primary_artifacts = {
        sweep["config_path"],
        sweep["manifest_path"],
    }
    unclassified = [
        value
        for value in fixed_by
        if value not in scientific_protocol_paths
        and value not in primary_artifacts
    ]
    authority_ids = _authority_ids(scientific_protocol_paths)
    protocol_specs: list[dict[str, str]] = []
    protocol_bindings: list[dict[str, str]] = []
    for protocol_path in scientific_protocol_paths:
        relative = safe_relative_path(
            protocol_path,
            stage="legacy_plan_validation",
            path="$.sweep.undeclared_fields_fixed_by",
        )
        absolute = root / relative
        require(
            absolute.is_file() and not absolute.is_symlink(),
            stage="legacy_plan_validation",
            code="legacy_protocol_missing",
            path="$.sweep.undeclared_fields_fixed_by",
            expected="an existing non-symlink docs/ protocol file",
            provided=protocol_path,
        )
        authority_id = authority_ids[protocol_path]
        protocol_specs.append(
            {
                "authority_id": authority_id,
                "sha256": file_sha256(absolute),
            }
        )
        protocol_bindings.append(
            {"authority_id": authority_id, "path": protocol_path}
        )

    study_document: dict[str, Any] = {
        "schema_version": STUDY_SCHEMA,
        "study": {
            "study_id": plan["experiment_id"],
            "title": plan["title"],
            "evidence_scope": plan["evidence_scope"],
            "hypothesis": deepcopy(plan["hypothesis"]),
            "design": {
                "artifacts": {
                    "scientific_config_sha256": sweep["config_sha256"],
                    "study_manifest_sha256": sweep["manifest_sha256"],
                    "manifest_job_count_pointer": sweep[
                        "manifest_job_count_pointer"
                    ],
                },
                "axes": deepcopy(sweep["axes"]),
                "case_ids": deepcopy(sweep["case_ids"]),
                "seeds": deepcopy(sweep["seeds"]),
                "conditional_expansion": deepcopy(
                    sweep["conditional_expansion"]
                ),
                "expected_initial_job_count": sweep[
                    "expected_initial_job_count"
                ],
                "maximum_total_job_count": sweep[
                    "maximum_total_job_count"
                ],
                "protocols": protocol_specs,
            },
            "acceptance": {
                "required_coverage": completion["required_coverage"],
                "allowed_exclusions": deepcopy(
                    completion["allowed_exclusions"]
                ),
                "failure_handling": completion["failure_handling"],
                "required_scientific_gates": (
                    ["tk_reference"]
                    if preflight["tk_reference_required"]
                    else []
                ),
                "local_validation_required": completion[
                    "local_validation_required"
                ],
                "review_required": completion["review_required"],
            },
            "reporting": {
                "comparisons": [
                    {
                        "comparison_id": card["comparison_id"],
                        "card_schema_version": card["card_schema_version"],
                    }
                    for card in reporting["comparison_cards"]
                ]
            },
        },
        "bindings": {
            "scientific_config_path": sweep["config_path"],
            "study_manifest_path": sweep["manifest_path"],
            "protocols": protocol_bindings,
            "reporting": {
                "comparisons": [
                    {
                        "comparison_id": card["comparison_id"],
                        "card_path": card["card_path"],
                        "review_path": card["review_path"],
                        "final_results_anchor": card[
                            "final_results_anchor"
                        ],
                    }
                    for card in reporting["comparison_cards"]
                ],
                "final_results_page": reporting["final_results_page"],
            },
        },
        "approval": {
            "status": (
                "review_required"
                if plan["approval"]["status"] == "approved"
                else "draft"
            ),
            "approved_by": None,
            "approved_at": None,
            "approved_study_sha256": None,
            "amendment_of": None,
        },
        "provenance": {
            "origin": "legacy-plan-split",
            "source_path": str(source_relative),
            "source_sha256": file_sha256(source),
            "notes": [
                (
                    "Generated deterministically from an "
                    "experiment-run-plan/v1 contract."
                ),
                (
                    "Legacy approval is preserved only in the "
                    "non-authoritative residue; scientific approval was "
                    "not inferred."
                ),
                (
                    "Only docs/ fixed-by entries were classified as "
                    "scientific protocol authorities; all other auxiliary "
                    "entries remain review residue."
                ),
            ],
        },
    }

    residue_document: dict[str, Any] = {
        "schema_version": LEGACY_RESIDUE_SCHEMA,
        "status": "not_launch_authority",
        "study_id": plan["experiment_id"],
        "study_spec_sha256": study_spec_sha256(study_document),
        "source": {
            "path": str(source_relative),
            "sha256": file_sha256(source),
            "schema_version": plan["schema_version"],
        },
        "legacy_approval": deepcopy(plan["approval"]),
        "operational_residue": {
            "execution": deepcopy(plan["execution"]),
            "storage": deepcopy(plan["storage"]),
            "progress_tracker": reporting["progress_tracker"],
            "fixed_by": {
                "scientific_protocols": scientific_protocol_paths,
                "primary_scientific_artifacts": [
                    sweep["config_path"],
                    sweep["manifest_path"],
                ],
                "unclassified_auxiliary_or_operational": unclassified,
            },
        },
        "blocking_reasons": [
            (
                "The legacy approval was not cryptographically bound to the "
                "new scientific study object."
            ),
            (
                "A target profile, immutable source release, routed "
                "manifest, environment identity, run class, and hard "
                "deadline cannot be inferred safely from the mixed plan."
            ),
            (
                "Create and separately authorize an "
                "experiment-execution-attempt/v1 contract before any "
                "launch."
            ),
        ],
    }
    validate_study(
        study_document,
        repo_root=root,
        verify_bindings=True,
        require_approved=False,
    )
    return study_document, residue_document


def publish_legacy_split(
    *,
    study_document: Mapping[str, Any],
    residue_document: Mapping[str, Any],
    study_output: str | Path,
    residue_output: str | Path,
) -> None:
    """Publish both outputs first-write-wins, checking collisions up front."""

    study_path = Path(study_output).expanduser().resolve()
    residue_path = Path(residue_output).expanduser().resolve()
    require(
        study_path != residue_path,
        stage="contract_publication",
        code="output_path_collision",
        path="study_output/residue_output",
        expected="two distinct output paths",
        provided=str(study_path),
    )
    existing = [
        str(path) for path in (study_path, residue_path) if path.exists()
    ]
    require(
        not existing,
        stage="contract_publication",
        code="output_exists",
        path="study_output/residue_output",
        expected="two new output paths",
        provided=existing,
    )
    # Publish the explicitly non-authoritative residue first.  In the unlikely
    # event of a race on the second path, no launch-authoritative file is left.
    write_json_first(residue_path, residue_document)
    write_json_first(study_path, study_document)


__all__ = [
    "LEGACY_PLAN_SCHEMA",
    "LEGACY_RESIDUE_SCHEMA",
    "load_legacy_plan",
    "publish_legacy_split",
    "split_legacy_plan",
]
