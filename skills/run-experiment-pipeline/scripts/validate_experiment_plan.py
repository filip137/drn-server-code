#!/usr/bin/env python3
"""Validate an experiment-plan Markdown contract and its bound files."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
JSON_BLOCK_RE = re.compile(r"```json[ \t]*\n(.*?)\n```", re.DOTALL)
PLACEHOLDER_RE = re.compile(r"(?:REPLACE_ME|replace-me|TODO|TBD|<[^>]+>)")
SUPPORTED_CARD_SCHEMAS = {
    "drn-result-card/v1",
    "drn-optimizer-screen-card/v1",
    "drn-rho-corner-diagnostic-card/v1",
    "drn-initialization-rho-comparison-card/v1",
}


class PlanError(ValueError):
    """Raised when an experiment plan violates the contract."""


def require(condition: bool, expected: str, provided: Any) -> None:
    if not condition:
        raise PlanError(f"Expected {expected}; got {provided!r}")


def require_keys(
    value: Any,
    *,
    required: set[str],
    context: str,
) -> dict[str, Any]:
    require(isinstance(value, dict), f"an object for {context}", type(value).__name__)
    missing = required - set(value)
    unknown = set(value) - required
    require(not missing, f"all required keys in {context}", sorted(missing))
    require(not unknown, f"only documented keys in {context}", sorted(unknown))
    return value


def require_text(value: Any, context: str) -> str:
    require(isinstance(value, str) and value.strip(), f"non-empty text for {context}", value)
    return value


def require_string_list(value: Any, context: str, *, allow_empty: bool = False) -> list[str]:
    require(
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, str) and item.strip() for item in value),
        f"{'a' if allow_empty else 'a non-empty'} list of strings for {context}",
        value,
    )
    require(len(value) == len(set(value)), f"unique strings for {context}", value)
    return value


def relative_repo_path(value: Any, context: str) -> Path:
    text = require_text(value, context)
    path = Path(text)
    require(not path.is_absolute(), f"a repository-relative path for {context}", text)
    require(".." not in path.parts, f"a path without '..' for {context}", text)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_plan(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlanError(f"Expected a readable experiment plan at {path}; got {exc}") from exc
    blocks = JSON_BLOCK_RE.findall(text)
    require(len(blocks) == 1, "exactly one fenced JSON contract", len(blocks))
    try:
        value = json.loads(blocks[0])
    except json.JSONDecodeError as exc:
        raise PlanError(f"Expected valid JSON in {path}; got {exc}") from exc
    require(isinstance(value, dict), "a JSON object for the plan contract", type(value).__name__)
    return value


def resolve_pointer(value: Any, pointer: str) -> Any:
    require(
        isinstance(pointer, str) and pointer.startswith("/"),
        "a non-empty JSON pointer beginning with '/'",
        pointer,
    )
    current = value
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            require(part in current, f"JSON pointer component {part!r}", list(current))
            current = current[part]
        elif isinstance(current, list):
            require(part.isdigit(), f"a list index in JSON pointer for {part!r}", part)
            index = int(part)
            require(0 <= index < len(current), f"a valid list index for {part!r}", index)
            current = current[index]
        else:
            raise PlanError(
                f"Expected JSON pointer {pointer!r} to traverse a container; "
                f"got {type(current).__name__!r}"
            )
    return current


def reject_placeholders(value: Any, context: str = "plan") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            reject_placeholders(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_placeholders(item, f"{context}[{index}]")
    elif isinstance(value, str):
        require(
            PLACEHOLDER_RE.search(value) is None,
            f"no placeholder in {context}",
            value,
        )


def validate_plan(
    plan_path: Path,
    plan: dict[str, Any],
    *,
    require_approved: bool,
    verify_files: bool,
) -> None:
    require_keys(
        plan,
        required={
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
        context="experiment plan",
    )
    require(
        plan["schema_version"] == "experiment-run-plan/v1",
        "schema_version 'experiment-run-plan/v1'",
        plan["schema_version"],
    )
    experiment_id = require_text(plan["experiment_id"], "experiment_id")
    require(ID_RE.fullmatch(experiment_id) is not None, "a filesystem-safe experiment_id", experiment_id)
    require(plan_path.stem == experiment_id, "plan filename matching experiment_id", plan_path.stem)
    require_text(plan["title"], "title")
    require(
        plan["evidence_scope"] in {"paper_facing", "diagnostic", "historical_replay"},
        "evidence_scope paper_facing, diagnostic, or historical_replay",
        plan["evidence_scope"],
    )

    hypothesis = require_keys(
        plan["hypothesis"],
        required={
            "statement",
            "control",
            "treatments",
            "expected_direction",
            "decision_rule",
            "supports_if",
            "does_not_support_if",
            "inconclusive_if",
        },
        context="hypothesis",
    )
    for key in (
        "statement",
        "control",
        "expected_direction",
        "supports_if",
        "does_not_support_if",
        "inconclusive_if",
    ):
        require_text(hypothesis[key], f"hypothesis.{key}")
    require_string_list(hypothesis["treatments"], "hypothesis.treatments")
    decision = require_keys(
        hypothesis["decision_rule"],
        required={
            "metric",
            "split",
            "checkpoint_role",
            "seed_aggregation",
            "comparison",
            "minimum_meaningful_effect",
            "effect_units",
        },
        context="hypothesis.decision_rule",
    )
    for key in (
        "metric",
        "split",
        "checkpoint_role",
        "seed_aggregation",
        "comparison",
        "effect_units",
    ):
        require_text(decision[key], f"hypothesis.decision_rule.{key}")
    require(
        isinstance(decision["minimum_meaningful_effect"], (int, float))
        and not isinstance(decision["minimum_meaningful_effect"], bool)
        and decision["minimum_meaningful_effect"] >= 0,
        "a non-negative numeric minimum_meaningful_effect",
        decision["minimum_meaningful_effect"],
    )

    sweep = require_keys(
        plan["sweep"],
        required={
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
        context="sweep",
    )
    config_path = relative_repo_path(sweep["config_path"], "sweep.config_path")
    manifest_path = relative_repo_path(sweep["manifest_path"], "sweep.manifest_path")
    require_text(sweep["manifest_job_count_pointer"], "sweep.manifest_job_count_pointer")
    require(isinstance(sweep["axes"], list) and sweep["axes"], "at least one sweep axis", sweep["axes"])
    axis_names: set[str] = set()
    for axis in sweep["axes"]:
        axis = require_keys(
            axis,
            required={"name", "config_path", "ordered_values"},
            context="sweep axis",
        )
        name = require_text(axis["name"], "sweep axis name")
        require(name not in axis_names, "unique sweep axis names", name)
        axis_names.add(name)
        pointer = require_text(axis["config_path"], f"sweep axis {name} config_path")
        require(pointer.startswith("/"), f"a JSON pointer for sweep axis {name}", pointer)
        values = axis["ordered_values"]
        require(isinstance(values, list) and values, f"ordered values for sweep axis {name}", values)
        canonical = [json.dumps(value, sort_keys=True, separators=(",", ":")) for value in values]
        require(len(canonical) == len(set(canonical)), f"unique ordered values for sweep axis {name}", values)
    require_string_list(sweep["case_ids"], "sweep.case_ids")
    require(
        isinstance(sweep["seeds"], list)
        and sweep["seeds"]
        and all(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 for seed in sweep["seeds"]),
        "a non-empty list of non-negative integer seeds",
        sweep["seeds"],
    )
    require(len(sweep["seeds"]) == len(set(sweep["seeds"])), "unique sweep seeds", sweep["seeds"])
    expansion = require_keys(
        sweep["conditional_expansion"],
        required={"enabled", "rule", "maximum_additional_jobs"},
        context="sweep.conditional_expansion",
    )
    require(isinstance(expansion["enabled"], bool), "a boolean expansion.enabled", expansion["enabled"])
    require(
        isinstance(expansion["maximum_additional_jobs"], int)
        and not isinstance(expansion["maximum_additional_jobs"], bool)
        and expansion["maximum_additional_jobs"] >= 0,
        "a non-negative integer maximum_additional_jobs",
        expansion["maximum_additional_jobs"],
    )
    if expansion["enabled"]:
        require_text(expansion["rule"], "conditional expansion rule")
    else:
        require(expansion["rule"] is None, "null rule when expansion is disabled", expansion["rule"])
        require(expansion["maximum_additional_jobs"] == 0, "zero additional jobs when expansion is disabled", expansion["maximum_additional_jobs"])
    initial_count = sweep["expected_initial_job_count"]
    maximum_count = sweep["maximum_total_job_count"]
    require(
        isinstance(initial_count, int) and not isinstance(initial_count, bool) and initial_count > 0,
        "a positive integer expected_initial_job_count",
        initial_count,
    )
    require(
        maximum_count == initial_count + expansion["maximum_additional_jobs"],
        "maximum_total_job_count equal to initial plus approved expansion cap",
        maximum_count,
    )
    fixed_by = require_string_list(
        sweep["undeclared_fields_fixed_by"],
        "sweep.undeclared_fields_fixed_by",
    )
    for index, value in enumerate(fixed_by):
        relative_repo_path(value, f"sweep.undeclared_fields_fixed_by[{index}]")

    execution = require_keys(
        plan["execution"],
        required={"launcher", "collector", "validator", "preflight"},
        context="execution",
    )
    for key in ("launcher", "collector", "validator"):
        require_string_list(execution[key], f"execution.{key}")
    preflight = require_keys(
        execution["preflight"],
        required={
            "smoke_required",
            "tk_reference_required",
            "scheduled_run_preflight_required",
            "receipt_path",
        },
        context="execution.preflight",
    )
    require(preflight["smoke_required"] is True, "smoke_required=true", preflight["smoke_required"])
    for key in ("tk_reference_required", "scheduled_run_preflight_required"):
        require(isinstance(preflight[key], bool), f"a boolean preflight.{key}", preflight[key])
    receipt_path = relative_repo_path(preflight["receipt_path"], "execution.preflight.receipt_path")
    require(receipt_path.parts[0] == "results", "a preflight receipt below results/", str(receipt_path))

    storage = require_keys(
        plan["storage"],
        required={"root_alias", "local_results_root", "local_bundle_path", "remote_staging"},
        context="storage",
    )
    require(storage["root_alias"] == "REPO_ROOT", "storage.root_alias='REPO_ROOT'", storage["root_alias"])
    local_root = relative_repo_path(storage["local_results_root"], "storage.local_results_root")
    local_bundle = relative_repo_path(storage["local_bundle_path"], "storage.local_bundle_path")
    require(local_root == Path("results"), "canonical local results root 'results'", str(local_root))
    require(local_bundle.parts[0] == "results" and len(local_bundle.parts) > 1, "a run-specific local bundle below results/", str(local_bundle))
    require(isinstance(storage["remote_staging"], list), "a remote_staging list", storage["remote_staging"])
    for remote in storage["remote_staging"]:
        remote = require_keys(
            remote,
            required={"host", "path", "intended_local_destination"},
            context="remote staging entry",
        )
        require_text(remote["host"], "remote staging host")
        remote_path = Path(require_text(remote["path"], "remote staging path"))
        require(remote_path.is_absolute(), "an absolute remote staging path", str(remote_path))
        destination = relative_repo_path(
            remote["intended_local_destination"],
            "remote intended_local_destination",
        )
        require(destination == local_bundle, "remote destination matching local bundle", str(destination))

    reporting = require_keys(
        plan["reporting"],
        required={"progress_tracker", "comparison_cards", "final_results_page"},
        context="reporting",
    )
    require(
        reporting["progress_tracker"] == "docs/current_experiments.md",
        "progress_tracker 'docs/current_experiments.md'",
        reporting["progress_tracker"],
    )
    require(
        reporting["final_results_page"] == "docs/results/index.md",
        "final_results_page 'docs/results/index.md'",
        reporting["final_results_page"],
    )
    require(
        isinstance(reporting["comparison_cards"], list) and reporting["comparison_cards"],
        "at least one planned comparison card",
        reporting["comparison_cards"],
    )
    comparison_ids: set[str] = set()
    for comparison in reporting["comparison_cards"]:
        comparison = require_keys(
            comparison,
            required={
                "comparison_id",
                "card_schema_version",
                "card_path",
                "review_path",
                "final_results_anchor",
            },
            context="comparison card",
        )
        comparison_id = require_text(comparison["comparison_id"], "comparison_id")
        require(ID_RE.fullmatch(comparison_id) is not None, "a filesystem-safe comparison_id", comparison_id)
        require(comparison_id not in comparison_ids, "unique comparison IDs", comparison_id)
        comparison_ids.add(comparison_id)
        require(
            comparison["card_schema_version"] in SUPPORTED_CARD_SCHEMAS,
            f"a currently supported result-card schema for {comparison_id}",
            comparison["card_schema_version"],
        )
        card_path = relative_repo_path(comparison["card_path"], f"{comparison_id} card_path")
        review_path = relative_repo_path(comparison["review_path"], f"{comparison_id} review_path")
        require(card_path.parts[:2] == ("result_registry", "cards") and card_path.suffix == ".json", f"a JSON result card below result_registry/cards for {comparison_id}", str(card_path))
        require(review_path.parts[:2] == ("result_registry", "reviews") and review_path.suffix == ".json", f"a JSON review below result_registry/reviews for {comparison_id}", str(review_path))
        require(
            comparison["final_results_anchor"] == f"docs/results/index.md#{comparison_id}",
            f"the canonical final index anchor for {comparison_id}",
            comparison["final_results_anchor"],
        )

    completion = require_keys(
        plan["completion"],
        required={
            "required_coverage",
            "allowed_exclusions",
            "failure_handling",
            "local_validation_required",
            "review_required",
        },
        context="completion",
    )
    require_text(completion["required_coverage"], "completion.required_coverage")
    require_string_list(completion["allowed_exclusions"], "completion.allowed_exclusions", allow_empty=True)
    require_text(completion["failure_handling"], "completion.failure_handling")
    require(completion["local_validation_required"] is True, "local_validation_required=true", completion["local_validation_required"])
    require(completion["review_required"] is True, "review_required=true", completion["review_required"])

    approval = require_keys(
        plan["approval"],
        required={"status", "approved_by", "approved_at", "amendment_of"},
        context="approval",
    )
    require(approval["status"] in {"draft", "approved"}, "approval status draft or approved", approval["status"])
    if approval["amendment_of"] is not None:
        amendment = require_text(approval["amendment_of"], "approval.amendment_of")
        require(ID_RE.fullmatch(amendment) is not None, "a filesystem-safe amendment_of", amendment)
    if approval["status"] == "approved":
        require_text(approval["approved_by"], "approval.approved_by")
        approved_at = require_text(approval["approved_at"], "approval.approved_at")
        try:
            parsed = datetime.fromisoformat(approved_at)
        except ValueError as exc:
            raise PlanError(f"Expected an ISO-8601 approval timestamp; got {approved_at!r}") from exc
        require(parsed.tzinfo is not None, "a timezone-aware approval timestamp", approved_at)
        reject_placeholders(plan)
    else:
        require(approval["approved_by"] is None, "null approved_by for a draft", approval["approved_by"])
        require(approval["approved_at"] is None, "null approved_at for a draft", approval["approved_at"])
    if require_approved:
        require(approval["status"] == "approved", "an approved plan", approval["status"])

    if verify_files or require_approved:
        for label, relative_path, expected_hash in (
            ("config", config_path, sweep["config_sha256"]),
            ("manifest", manifest_path, sweep["manifest_sha256"]),
        ):
            require(
                isinstance(expected_hash, str) and SHA256_RE.fullmatch(expected_hash) is not None,
                f"a lowercase SHA-256 for the {label}",
                expected_hash,
            )
            absolute_path = REPO_ROOT / relative_path
            require(absolute_path.is_file(), f"an existing {label} at {relative_path}", "missing")
            observed_hash = sha256(absolute_path)
            require(observed_hash == expected_hash, f"SHA-256 {expected_hash} for {relative_path}", observed_hash)
        try:
            manifest = json.loads((REPO_ROOT / manifest_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PlanError(f"Expected a readable JSON manifest at {manifest_path}; got {exc}") from exc
        count_value = resolve_pointer(manifest, sweep["manifest_job_count_pointer"])
        if isinstance(count_value, list):
            observed_count = len(count_value)
        else:
            require(
                isinstance(count_value, int) and not isinstance(count_value, bool),
                "the manifest job-count pointer to resolve to an integer or list",
                count_value,
            )
            observed_count = count_value
        require(
            observed_count == initial_count,
            f"manifest job count {initial_count}",
            observed_count,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--require-approved", action="store_true")
    parser.add_argument("--verify-files", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan_path = args.plan.expanduser().resolve()
    try:
        plan = load_plan(plan_path)
        validate_plan(
            plan_path,
            plan,
            require_approved=args.require_approved,
            verify_files=args.verify_files,
        )
        print(f"OK {plan_path}")
        return 0
    except PlanError as exc:
        print(f"Experiment plan validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
