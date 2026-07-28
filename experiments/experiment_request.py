#!/usr/bin/env python3
"""Parse and validate the user-facing experiment request Markdown.

This module intentionally stops at intake.  It does not resolve a catalog
entry, approve a study, mutate the tracker, run preflight, or launch work.
The public contract is:

``parse_request_text(markdown) -> normalized request mapping``
``validate_request(mapping) -> aggregated validation mapping``

The normalized request keeps user answers, agent-resolved material, and user
review in separate objects so that an agent-authored summary can never fill a
missing user-owned field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


REQUEST_SCHEMA_VERSION = "experiment-request/v1"
VALIDATION_SCHEMA_VERSION = "experiment-request-validation/v1"

USER_ANSWER_FIELDS = (
    "short_name",
    "request_type",
    "parent",
    "purpose",
    "cases_subset",
    "scientific_changes",
    "keep_fixed",
    "training_evaluation_budget",
    "seeds",
    "sweep_or_conditional_expansion",
    "decision_style",
    "expected_outcome",
    "negative_evidence",
    "inconclusive_outcome",
    "official_test_policy",
    "evidence_scope",
    "preferred_compute",
    "distribution_preference",
    "executor_adapt_operations",
    "executor_launch_after_checks",
)

COMPOSITE_ANSWER_FIELDS = (
    "budget_seeds_sweep",
    "evidence_official_test_policy",
    "compute_distribution",
    "executor_authority",
)

REVIEW_FIELDS = (
    "scientific_summary_approved",
    "execution_proposal_authorized",
    "corrections_or_constraints",
)

_INHERIT_VALUES = {
    "inherit",
    "inherit parent",
    "inherit from parent",
    "inherited from parent",
    "same as parent",
    "use parent",
}
_MISSING_VALUES = {
    "",
    "fill",
    "missing",
    "pending",
    "please fill",
    "tbd",
    "todo",
    "unknown",
    "unset",
}
_PLACEHOLDER_RE = re.compile(r"<[^>\n]+>")
_LABEL_RE = re.compile(
    r"(?ms)^[ \t]*-\s+\*\*(?P<label>.+?)(?::|\?)?\*\*[ \t]*(?P<body>.*?)"
    r"(?=^[ \t]*-\s+\*\*|^#{2,3}[ \t]+|\Z)"
)
_CHECKBOX_RE = re.compile(
    r"(?mi)^[ \t]*-\s+`?\[(?P<mark>[ xX])\]`?\s+(?P<label>.+)$"
)
_HEADING_RE = re.compile(r"(?m)^###\s+(?P<label>.+?)\s*$")


_FULL_LABEL_TO_FIELD = {
    "short name": "short_name",
    "request type": "request_type",
    "parent experiment or result": "parent",
    "purpose": "purpose",
    "cases or subset": "cases_subset",
    "scientific changes from the parent": "scientific_changes",
    "keep fixed from the parent": "keep_fixed",
    "training evaluation budget": "training_evaluation_budget",
    "seeds": "seeds",
    "sweep or conditional expansion": "sweep_or_conditional_expansion",
    "decision style": "decision_style",
    "expected outcome or direction": "expected_outcome",
    "what would count as failure or negative evidence": "negative_evidence",
    "what would make the result inconclusive": "inconclusive_outcome",
    "official test set use": "official_test_policy",
    "evidence scope": "evidence_scope",
    "preferred compute": "preferred_compute",
    "distribution preference": "distribution_preference",
    (
        "may the executor adapt operational code and environment after a "
        "passing functional smoke"
    ): "executor_adapt_operations",
    (
        "after scientific approval and passing functional checks may the "
        "executor launch automatically"
    ): "executor_launch_after_checks",
}

_COMPACT_LABEL_TO_FIELD = {
    "experiment": ("answer", "short_name"),
    "type": ("answer", "request_type"),
    "parent": ("answer", "parent"),
    "purpose": ("answer", "purpose"),
    "cases subset": ("answer", "cases_subset"),
    "scientific changes": ("answer", "scientific_changes"),
    "keep fixed": ("answer", "keep_fixed"),
    "budget seeds sweep": ("composite", "budget_seeds_sweep"),
    "decision style": ("answer", "decision_style"),
    (
        "evidence official test policy"
    ): ("composite", "evidence_official_test_policy"),
    "compute distribution": ("composite", "compute_distribution"),
    "executor": ("composite", "executor_authority"),
}

_REQUEST_TYPE_PATTERNS = (
    ("derived_comparison", ("changed comparison", "derived comparison")),
    ("continuation", ("continue", "continuation", "extend")),
    ("repeat", ("repeat", "replay")),
    ("new", ("define a new", "new study", "new")),
)
_DECISION_STYLE_PATTERNS = (
    ("standard_continuation", ("standard continuation", "continuation check")),
    ("standard_comparison", ("standard controlled comparison", "standard comparison")),
    ("custom", ("custom",)),
)
_OFFICIAL_TEST_PATTERNS = (
    ("forbidden", ("forbidden",)),
    (
        "parent_authorized_only",
        ("allowed only", "allowed by a parent", "parent protocol explicitly authorizes"),
    ),
)
_EVIDENCE_SCOPE_PATTERNS = (
    ("ordinary_diagnostic", ("ordinary diagnostic", "diagnostic")),
    ("paper_facing", ("paper-facing", "paper facing")),
    ("historical_replay", ("historical replay",)),
)


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _clean_text(value: str) -> str:
    lines = []
    for raw_line in value.strip().splitlines():
        line = raw_line.strip()
        if line.startswith("```") or not line:
            continue
        lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = _clean_text(value)
    if _key(cleaned) in _INHERIT_VALUES:
        return "inherit_from_parent"
    return cleaned or None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    cleaned = value.strip()
    if _key(cleaned) in _MISSING_VALUES:
        return True
    return bool(_PLACEHOLDER_RE.search(cleaned))


def _section(text: str, heading: str, next_headings: Sequence[str]) -> str | None:
    start_match = re.search(
        rf"(?mi)^##[ \t]+{re.escape(heading)}[ \t]*$",
        text,
    )
    if start_match is None:
        return None
    end = len(text)
    for candidate in next_headings:
        match = re.search(
            rf"(?mi)^##[ \t]+{re.escape(candidate)}[ \t]*$",
            text[start_match.end() :],
        )
        if match is not None:
            end = min(end, start_match.end() + match.start())
    return text[start_match.end() : end]


def _labeled_blocks(section: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    blocks: dict[str, str] = {}
    issues: list[dict[str, Any]] = []
    for match in _LABEL_RE.finditer(section):
        label = _key(_clean_text(match.group("label")))
        body = match.group("body").strip()
        if label in blocks:
            issues.append(
                {
                    "field": label,
                    "expected": "one answer for each user-owned field",
                    "provided": "duplicate field",
                }
            )
            continue
        blocks[label] = body
    return blocks, issues


def _selected_checkbox(body: str) -> tuple[str | None, list[str], int]:
    checkboxes = list(_CHECKBOX_RE.finditer(body))
    selected = [
        _clean_text(match.group("label"))
        for match in checkboxes
        if match.group("mark").casefold() == "x"
    ]
    return (
        selected[0] if len(selected) == 1 else None,
        selected,
        len(checkboxes),
    )


def _match_choice(
    value: str | None,
    patterns: Sequence[tuple[str, Sequence[str]]],
) -> str | None:
    normalized = _key(value or "")
    if normalized in _INHERIT_VALUES or normalized == "inherit_from_parent":
        return "inherit_from_parent"
    for canonical, aliases in patterns:
        if any(_key(alias) in normalized for alias in aliases):
            return canonical
    return _normalize_answer(value)


def _normalize_yes_no(value: str | None) -> str | None:
    normalized = _key(value or "")
    if normalized in _INHERIT_VALUES or normalized == "inherit_from_parent":
        return "inherit_from_parent"
    if normalized in {"yes", "y", "true"}:
        return "yes"
    if normalized in {"no", "n", "false"}:
        return "no"
    return _normalize_answer(value)


def _choice_from_block(
    *,
    field: str,
    body: str,
    patterns: Sequence[tuple[str, Sequence[str]]],
    issues: list[dict[str, Any]],
) -> str | None:
    selected, all_selected, checkbox_count = _selected_checkbox(body)
    if len(all_selected) > 1:
        issues.append(
            {
                "field": field,
                "expected": "exactly one selected option",
                "provided": all_selected,
            }
        )
        return None
    if selected is not None:
        return _match_choice(selected, patterns)
    if checkbox_count:
        return None
    inline = _clean_text(_CHECKBOX_RE.sub("", body))
    return _match_choice(inline, patterns) if inline else None


def _empty_request(source_format: str) -> dict[str, Any]:
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "source": {"format": source_format},
        "user_answers": {field: None for field in USER_ANSWER_FIELDS},
        "composite_answers": {
            field: None for field in COMPOSITE_ANSWER_FIELDS
        },
        "agent_resolved_summary": {},
        "user_review": {field: None for field in REVIEW_FIELDS},
        "parse_issues": [],
    }


def _parse_full_user_answers(section: str, request: dict[str, Any]) -> None:
    blocks, issues = _labeled_blocks(section)
    request["parse_issues"].extend(issues)
    answers = request["user_answers"]

    for label, body in blocks.items():
        field = _FULL_LABEL_TO_FIELD.get(label)
        if field is None:
            continue
        if field == "request_type":
            answers[field] = _choice_from_block(
                field=field,
                body=body,
                patterns=_REQUEST_TYPE_PATTERNS,
                issues=request["parse_issues"],
            )
        elif field == "decision_style":
            answers[field] = _choice_from_block(
                field=field,
                body=body,
                patterns=_DECISION_STYLE_PATTERNS,
                issues=request["parse_issues"],
            )
        elif field == "official_test_policy":
            answers[field] = _choice_from_block(
                field=field,
                body=body,
                patterns=_OFFICIAL_TEST_PATTERNS,
                issues=request["parse_issues"],
            )
        elif field == "evidence_scope":
            answers[field] = _choice_from_block(
                field=field,
                body=body,
                patterns=_EVIDENCE_SCOPE_PATTERNS,
                issues=request["parse_issues"],
            )
        elif field in {
            "executor_adapt_operations",
            "executor_launch_after_checks",
        }:
            answers[field] = _normalize_yes_no(_clean_text(body))
        else:
            answers[field] = _normalize_answer(body)


def _subsection_fields(section: str) -> dict[str, dict[str, str]]:
    headings = list(_HEADING_RE.finditer(section))
    parsed: dict[str, dict[str, str]] = {}
    if not headings:
        blocks, _ = _labeled_blocks(section)
        return {"summary": {_slug(key): _clean_text(value) for key, value in blocks.items()}}
    for index, match in enumerate(headings):
        start = match.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(section)
        blocks, _ = _labeled_blocks(section[start:end])
        parsed[_slug(match.group("label"))] = {
            _slug(key): _clean_text(value) for key, value in blocks.items()
        }
    return parsed


def _parse_review(section: str, request: dict[str, Any]) -> None:
    blocks, issues = _labeled_blocks(section)
    request["parse_issues"].extend(issues)
    review = request["user_review"]
    label_map = {
        "scientific summary approved": "scientific_summary_approved",
        "execution proposal authorized": "execution_proposal_authorized",
        "corrections or constraints": "corrections_or_constraints",
    }
    for label, body in blocks.items():
        field = label_map.get(label)
        if field is None:
            continue
        value = _clean_text(body)
        review[field] = (
            _normalize_yes_no(value)
            if field != "corrections_or_constraints"
            else _normalize_answer(value)
        )


def _parse_compact(text: str, request: dict[str, Any]) -> None:
    recognized = set(_COMPACT_LABEL_TO_FIELD)
    current: tuple[str, str] | None = None
    accumulated: list[str] = []

    def finish() -> None:
        nonlocal current, accumulated
        if current is None:
            return
        destination, field = current
        value = _normalize_answer("\n".join(accumulated))
        if destination == "answer":
            if field == "request_type":
                value = _match_choice(value, _REQUEST_TYPE_PATTERNS)
            elif field == "decision_style":
                value = _match_choice(value, _DECISION_STYLE_PATTERNS)
            request["user_answers"][field] = value
        else:
            request["composite_answers"][field] = value
        current = None
        accumulated = []

    line_re = re.compile(
        r"^\s*(?:[-*]\s+)?(?:\*\*)?(?P<label>[A-Za-z][^:\n]*?)"
        r"(?:\*\*)?:\s*(?P<value>.*)$"
    )
    for line in text.splitlines():
        match = line_re.match(line)
        label = _key(match.group("label")) if match else None
        if match and label in recognized:
            finish()
            current = _COMPACT_LABEL_TO_FIELD[label]
            accumulated = [match.group("value")]
        elif current is not None and not line.strip().startswith("```"):
            accumulated.append(line)
    finish()


def parse_request_text(text: str) -> dict[str, Any]:
    """Return a normalized, JSON-serializable experiment request mapping.

    Both the compact conversational form and the full Markdown intake form are
    accepted.  Missing answers remain ``None``; call :func:`validate_request`
    to obtain one aggregated report.
    """

    if not isinstance(text, str):
        raise TypeError(
            f"Expected Markdown text as a string; provided value: {type(text).__name__}"
        )

    user_section = _section(
        text,
        "User answers",
        ("Agent-resolved summary", "User review"),
    )
    if user_section is None:
        request = _empty_request("compact_conversational_markdown/v1")
        _parse_compact(text, request)
        return request

    request = _empty_request("full_markdown/v1")
    _parse_full_user_answers(user_section, request)

    summary_section = _section(text, "Agent-resolved summary", ("User review",))
    if summary_section is not None:
        request["agent_resolved_summary"] = _subsection_fields(summary_section)

    review_section = _section(text, "User review", ())
    if review_section is not None:
        _parse_review(review_section, request)
    return request


def load_request(path: Path) -> dict[str, Any]:
    """Read and parse one request document."""

    try:
        request = parse_request_text(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            f"Expected a readable experiment request; provided value: {path}: {exc}"
        ) from exc
    request["source"]["path"] = str(path)
    return request


def _answered(mapping: Mapping[str, Any], field: str) -> bool:
    return field in mapping and not _is_missing(mapping[field])


def _invalid_choice_issue(
    *,
    answers: Mapping[str, Any],
    field: str,
    allowed: set[str],
) -> dict[str, Any] | None:
    value = answers.get(field)
    if _is_missing(value) or value in allowed or value == "inherit_from_parent":
        return None
    return {
        "field": field,
        "expected": f"one of {sorted(allowed)} or 'inherit from parent'",
        "provided": value,
    }


def _inheritance_fields(request: Mapping[str, Any]) -> list[str]:
    fields = []
    for group_name in ("user_answers", "composite_answers"):
        group = request.get(group_name)
        if not isinstance(group, Mapping):
            continue
        for field, value in group.items():
            if value == "inherit_from_parent":
                fields.append(f"{group_name}.{field}")
    return sorted(fields)


def validate_request(
    request: Mapping[str, Any],
    *,
    require_review: bool = False,
) -> dict[str, Any]:
    """Validate a normalized request and aggregate every intake problem.

    ``inherit_from_parent`` is a complete user answer.  It is reported under
    ``unresolved_fields`` for the resolver, but never under ``missing_fields``.
    No hash, receipt, environment path, Git identity, or launch path is part
    of this validation contract.
    """

    invalid: list[dict[str, Any]] = []
    missing: list[str] = []

    if not isinstance(request, Mapping):
        return {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "status": "invalid",
            "missing_fields": [],
            "unresolved_fields": [],
            "invalid_fields": [
                {
                    "field": "$",
                    "expected": "a normalized experiment-request mapping",
                    "provided": type(request).__name__,
                }
            ],
            "review_required": require_review,
            "review_status": "not_checked",
        }

    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        invalid.append(
            {
                "field": "schema_version",
                "expected": REQUEST_SCHEMA_VERSION,
                "provided": request.get("schema_version"),
            }
        )

    answers = request.get("user_answers")
    composites = request.get("composite_answers")
    review = request.get("user_review")
    if not isinstance(answers, Mapping):
        invalid.append(
            {
                "field": "user_answers",
                "expected": "an object containing only user-provided answers",
                "provided": type(answers).__name__,
            }
        )
        answers = {}
    if not isinstance(composites, Mapping):
        invalid.append(
            {
                "field": "composite_answers",
                "expected": "an object containing compact combined answers",
                "provided": type(composites).__name__,
            }
        )
        composites = {}
    if not isinstance(review, Mapping):
        invalid.append(
            {
                "field": "user_review",
                "expected": "an object containing only user review answers",
                "provided": type(review).__name__,
            }
        )
        review = {}

    source = request.get("source")
    source_format = source.get("format") if isinstance(source, Mapping) else None
    is_compact = source_format == "compact_conversational_markdown/v1"

    for field in (
        "short_name",
        "request_type",
        "parent",
        "purpose",
        "cases_subset",
        "scientific_changes",
        "keep_fixed",
        "decision_style",
    ):
        if not _answered(answers, field):
            missing.append(field)

    grouped_requirements = (
        (
            "budget_seeds_sweep",
            (
                "training_evaluation_budget",
                "seeds",
                "sweep_or_conditional_expansion",
            ),
        ),
        (
            "evidence_official_test_policy",
            ("evidence_scope", "official_test_policy"),
        ),
        (
            "compute_distribution",
            ("preferred_compute", "distribution_preference"),
        ),
        (
            "executor_authority",
            ("executor_adapt_operations", "executor_launch_after_checks"),
        ),
    )
    for composite_field, detailed_fields in grouped_requirements:
        if _answered(composites, composite_field):
            continue
        absent = [field for field in detailed_fields if not _answered(answers, field)]
        if absent:
            missing.extend([composite_field] if is_compact else absent)

    # The detailed outcome fields are optional in the compact form when a
    # standard decision style is selected; the agent resolves its documented
    # defaults.  They remain user-owned in the full form and for custom rules.
    decision_style = answers.get("decision_style")
    if not is_compact or decision_style == "custom":
        for field in (
            "expected_outcome",
            "negative_evidence",
            "inconclusive_outcome",
        ):
            if not _answered(answers, field):
                missing.append(field)

    for issue in (
        _invalid_choice_issue(
            answers=answers,
            field="request_type",
            allowed={"continuation", "repeat", "derived_comparison", "new"},
        ),
        _invalid_choice_issue(
            answers=answers,
            field="decision_style",
            allowed={"standard_continuation", "standard_comparison", "custom"},
        ),
        _invalid_choice_issue(
            answers=answers,
            field="official_test_policy",
            allowed={"forbidden", "parent_authorized_only"},
        ),
        _invalid_choice_issue(
            answers=answers,
            field="evidence_scope",
            allowed={"ordinary_diagnostic", "paper_facing", "historical_replay"},
        ),
        _invalid_choice_issue(
            answers=answers,
            field="executor_adapt_operations",
            allowed={"yes", "no"},
        ),
        _invalid_choice_issue(
            answers=answers,
            field="executor_launch_after_checks",
            allowed={"yes", "no"},
        ),
    ):
        if issue is not None:
            invalid.append(issue)

    parse_issues = request.get("parse_issues", [])
    if isinstance(parse_issues, list):
        invalid.extend(issue for issue in parse_issues if isinstance(issue, dict))
    else:
        invalid.append(
            {
                "field": "parse_issues",
                "expected": "a list",
                "provided": type(parse_issues).__name__,
            }
        )

    review_status = "not_required"
    if require_review:
        scientific = review.get("scientific_summary_approved")
        execution = review.get("execution_proposal_authorized")
        if scientific == "yes" and execution == "yes":
            review_status = "approved"
        elif scientific == "no" or execution == "no":
            review_status = "not_approved"
            invalid.append(
                {
                    "field": "user_review",
                    "expected": "scientific and execution review both approved",
                    "provided": {
                        "scientific_summary_approved": scientific,
                        "execution_proposal_authorized": execution,
                    },
                }
            )
        else:
            review_status = "pending"
            if _is_missing(scientific):
                missing.append("user_review.scientific_summary_approved")
            if _is_missing(execution):
                missing.append("user_review.execution_proposal_authorized")

    # Stable order without duplicates makes a single conversational correction
    # round straightforward.
    missing = list(dict.fromkeys(missing))
    unresolved = _inheritance_fields(request)
    status = "invalid" if invalid else ("needs_user_input" if missing else "valid")
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": status,
        "missing_fields": missing,
        "unresolved_fields": unresolved,
        "invalid_fields": invalid,
        "review_required": require_review,
        "review_status": review_status,
    }


def parse_and_validate(
    text: str,
    *,
    require_review: bool = False,
) -> dict[str, Any]:
    """Return the normalized request and its aggregate validation envelope."""

    request = parse_request_text(text)
    return {
        "request": request,
        "validation": validate_request(request, require_review=require_review),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a user-facing Markdown experiment request.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate",
        help="parse the request and emit normalized JSON plus aggregate validation",
    )
    validate.add_argument("request", type=Path, help="Markdown request path")
    validate.add_argument(
        "--require-review",
        action="store_true",
        help="also require scientific approval and execution authorization",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        request = load_request(args.request)
        result = {
            "request": request,
            "validation": validate_request(
                request,
                require_review=args.require_review,
            ),
        }
    except (TypeError, ValueError) as exc:
        result = {
            "request": None,
            "validation": {
                "schema_version": VALIDATION_SCHEMA_VERSION,
                "status": "invalid",
                "missing_fields": [],
                "unresolved_fields": [],
                "invalid_fields": [
                    {
                        "field": "request",
                        "expected": "a readable Markdown experiment request",
                        "provided": str(exc),
                    }
                ],
                "review_required": bool(args.require_review),
                "review_status": "not_checked",
            },
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    status = result["validation"]["status"]
    return 0 if status == "valid" else (2 if status == "needs_user_input" else 3)


if __name__ == "__main__":
    raise SystemExit(main())
