"""Build a compact scientific study from a validated user request.

The user-facing validator checks intent.  This module resolves inherited
scientific values from the catalog parent, selects exact jobs, and generates
the study identity automatically.  Host, runner, environment, output, and
receipt details remain outside the scientific hash.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re
from typing import Any, Mapping

from experiments.control_plane.common import canonical_sha256
from experiments.experiment_catalog import (
    REPO_ROOT,
    resolve,
    resolve_structured_request,
)
from experiments.mnist_conv.identity import sha256_file


RESOLVED_STUDY_SCHEMA = "experiment-resolved-study/v1"
STUDY_VALIDATION_SCHEMA = "experiment-resolved-study-validation/v1"

SCIENTIFIC_CONTEXT_KEYS = (
    "dataset",
    "model",
    "solver",
    "optimizer_arms",
    "training_contracts",
    "safety",
    "evidence_scope",
    "selection_override",
    "prohibitions",
)
JOB_EVIDENCE_KEYS = {
    "boundary_status",
    "candidate_evidence",
    "probe_result",
    "runtime_rho_conv",
    "runtime_rho_dense",
    "security_result",
    "selection_basis",
    "source_candidate",
}
TK_RELEVANT_TERMS = {
    "architecture",
    "nonlinearity",
    "non-linearity",
    "amplification",
    "input gain",
    "input_gain",
    " t ",
    " k ",
    "t/k",
    "tk",
    "equilibrium",
    "gradient algorithm",
    "gradient estimator",
    "solver",
}
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
}


class StudyResolutionError(ValueError):
    """The validated request cannot yet resolve to exact scientific jobs."""


def _plain(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _plain(value).casefold()).strip()


def _answers(request: Mapping[str, Any]) -> Mapping[str, Any]:
    value = request.get("user_answers")
    if not isinstance(value, Mapping):
        raise StudyResolutionError(
            "Expected request.user_answers to be a mapping. "
            f"Provided value: {value!r}."
        )
    return value


def resolve_parent_id(
    parent_text: Any,
    catalog: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
) -> str:
    """Resolve an ID embedded in prose before falling back to the catalog."""

    text = _plain(parent_text)
    if not text:
        raise StudyResolutionError(
            "Expected a parent experiment ID or description. "
            f"Provided value: {parent_text!r}."
        )
    embedded = [
        entry["experiment_id"]
        for entry in catalog.get("entries", [])
        if isinstance(entry, Mapping)
        and isinstance(entry.get("experiment_id"), str)
        and entry["experiment_id"] in text
    ]
    if len(embedded) == 1:
        return embedded[0]
    if len(embedded) > 1:
        raise StudyResolutionError(
            "Expected the parent description to contain at most one catalog "
            f"experiment ID. Provided value: {embedded!r}."
        )
    outcome = resolve(catalog, text, repo_root)
    if outcome.get("status") != "resolved":
        raise StudyResolutionError(
            "Expected the parent description to resolve unambiguously. "
            f"Provided value: {outcome!r}."
        )
    return str(outcome["selected"]["experiment_id"])


def _summary_case_text(request: Mapping[str, Any]) -> str:
    summary = request.get("agent_resolved_summary")
    if not isinstance(summary, Mapping):
        return ""
    section = summary.get("resolved_scientific_study")
    if not isinstance(section, Mapping):
        return ""
    return _plain(section.get("treatments_cases_in_order"))


def _expected_count(text: str) -> int | None:
    matches: list[int] = []
    for match in re.finditer(
        r"\b(?:exactly|all)?\s*(\d+|"
        + "|".join(NUMBER_WORDS)
        + r")\s+(?:entries|jobs|runs|cases)\b",
        text.casefold(),
    ):
        token = match.group(1)
        matches.append(int(token) if token.isdigit() else NUMBER_WORDS[token])
    return matches[0] if matches and len(set(matches)) == 1 else None


def select_parent_jobs(
    request: Mapping[str, Any],
    parent_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resolve prose or explicit IDs to an exact ordered parent subset."""

    answers = _answers(request)
    entries = parent_config.get("entries")
    if not isinstance(entries, list) or not entries or any(
        not isinstance(item, Mapping) for item in entries
    ):
        raise StudyResolutionError(
            "Expected the parent scientific config to contain a non-empty "
            f"entries list. Provided value: {entries!r}."
        )
    jobs = [copy.deepcopy(dict(item)) for item in entries]
    identifiers = [job.get("entry_id") for job in jobs]
    if any(not isinstance(item, str) or not item for item in identifiers):
        raise StudyResolutionError(
            "Expected every parent entry to have a non-empty entry_id. "
            f"Provided value: {identifiers!r}."
        )
    if len(identifiers) != len(set(identifiers)):
        raise StudyResolutionError(
            "Expected unique parent entry IDs. "
            f"Provided value: {identifiers!r}."
        )

    cases_text = _plain(answers.get("cases_subset"))
    normalized_cases = _normalized(cases_text)
    if normalized_cases in {
        "all",
        "all cases",
        "all parent cases",
        "inherit",
        "inherit from parent",
        "inherit_from_parent",
    }:
        return jobs

    full_text = " ".join(
        (
            _plain(answers.get("short_name")),
            _plain(answers.get("purpose")),
            cases_text,
            _summary_case_text(request),
        )
    )
    explicit = [
        job
        for job in jobs
        if str(job["entry_id"]) in full_text
    ]
    if explicit:
        selected = explicit
    else:
        selected = jobs
        architectures = {
            architecture
            for architecture in ("conv1", "conv2", "conv3")
            if re.search(rf"\b{architecture}\b", full_text.casefold())
        }
        if architectures:
            selected = [
                job for job in selected if job.get("architecture") in architectures
            ]
        schemes = {
            scheme
            for scheme in ("baseline", "ours", "legacy")
            if re.search(rf"\b{scheme}\b", cases_text.casefold())
        }
        if schemes:
            selected = [
                job for job in selected if job.get("scheme") in schemes
            ]
        optimizers = {
            optimizer
            for optimizer in ("sgd", "adam")
            if re.search(rf"\b{optimizer}\b", cases_text.casefold())
        }
        if optimizers:
            selected = [
                job for job in selected if job.get("optimizer") in optimizers
            ]
        seed_matches = {
            int(value)
            for value in re.findall(
                r"\bseed(?:s)?\s*[-:=]?\s*(\d+)\b",
                full_text.casefold(),
            )
        }
        if seed_matches:
            inherited_seed = (
                parent_config.get("model", {}).get("model_seed")
                if isinstance(parent_config.get("model"), Mapping)
                else None
            )
            selected = [
                job
                for job in selected
                if job.get("model_seed", inherited_seed) in seed_matches
            ]

    if not selected:
        raise StudyResolutionError(
            "Expected cases/subset to select at least one parent job. "
            f"Provided value: {cases_text!r}."
        )
    expected = _expected_count(full_text)
    if expected is not None and len(selected) != expected:
        raise StudyResolutionError(
            "Expected cases/subset to resolve to the stated job count. "
            f"Provided value: stated={expected}, resolved={len(selected)}, "
            f"jobs={[item['entry_id'] for item in selected]!r}."
        )
    if selected == jobs and normalized_cases not in {
        "all",
        "all cases",
        "all parent cases",
    }:
        raise StudyResolutionError(
            "Expected a non-'all' cases/subset answer to narrow the parent or "
            "name every parent entry explicitly. "
            f"Provided value: {cases_text!r}."
        )
    return selected


def _scientific_change_description(value: Any) -> str | None:
    text = _plain(value)
    normalized = _normalized(text)
    if normalized in {"", "none", "no change", "no changes"}:
        return None
    # Selecting a parent subset is a derived study but not a change to any
    # selected job's operating point.
    if normalized.startswith("none ") and (
        "subset" in normalized or "selected" in normalized
    ):
        return None
    return text


def _validate_inherited_budget(
    answers: Mapping[str, Any],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    budget_text = _plain(answers.get("training_evaluation_budget"))
    epoch_values = {
        int(job["epochs"])
        for job in jobs
        if isinstance(job.get("epochs"), int)
        and not isinstance(job.get("epochs"), bool)
    }
    step_values = {
        int(job["expected_steps"])
        for job in jobs
        if isinstance(job.get("expected_steps"), int)
        and not isinstance(job.get("expected_steps"), bool)
    }
    stated_epochs = [
        int(value)
        for value in re.findall(r"\b(\d+)\s+epochs?\b", budget_text.casefold())
    ]
    if stated_epochs and (
        len(epoch_values) != 1 or stated_epochs[0] not in epoch_values
    ):
        raise StudyResolutionError(
            "Expected the requested epoch budget to match every selected "
            f"parent job. Provided value: requested={stated_epochs!r}, "
            f"parent={sorted(epoch_values)!r}."
        )
    return {
        "description": budget_text,
        "epochs_per_job": (
            next(iter(epoch_values)) if len(epoch_values) == 1 else None
        ),
        "steps_per_job": (
            next(iter(step_values)) if len(step_values) == 1 else None
        ),
        "job_count": len(jobs),
    }


def _scientific_job(
    entry: Mapping[str, Any],
    *,
    official_test_read: bool,
) -> dict[str, Any]:
    result = {
        key: copy.deepcopy(value)
        for key, value in entry.items()
        if key not in JOB_EVIDENCE_KEYS
    }
    result["job_id"] = result["entry_id"]
    result["official_test_read"] = official_test_read
    return result


def _tk_gate(
    *,
    request_type: str,
    scientific_change: str | None,
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_change = f" {_normalized(scientific_change)} "
    relevant = (
        scientific_change is not None
        and any(term in normalized_change for term in TK_RELEVANT_TERMS)
    )
    reusable_kind = request_type in {"continuation", "repeat"}
    fresh_required = relevant or not reusable_kind
    authorities = [
        {
            "job_id": job["entry_id"],
            "security_result": job.get("security_result"),
        }
        for job in jobs
        if job.get("security_result") is not None
    ]
    return {
        "fresh_required": fresh_required,
        "prior_result_reused": not fresh_required,
        "reason": (
            "A T/K-relevant scientific field changed."
            if relevant
            else (
                "The selected continuation/repeat keeps learning rates and "
                "the accepted operating point fixed."
                if reusable_kind
                else "A new or changed study requires a fresh T/K decision."
            )
        ),
        "prior_authorities": authorities if not fresh_required else [],
        "functional_smoke_required": True,
    }


def _catalog_request(
    request: Mapping[str, Any],
    *,
    parent_id: str,
    job_ids: list[str],
    scientific_change: str | None,
) -> dict[str, Any]:
    answers = _answers(request)
    operational = {
        "preferred_compute": answers.get("preferred_compute"),
        "distribution": answers.get("distribution_preference"),
        "adapt_environment": answers.get("executor_adapt_operations"),
        "launch_after_checks": answers.get("executor_launch_after_checks"),
    }
    return {
        "experiment": answers.get("short_name"),
        "type": answers.get("request_type"),
        "parent": parent_id,
        "cases_subset": job_ids,
        "scientific_changes": (
            "none" if scientific_change is None else scientific_change
        ),
        "keep_fixed": answers.get("keep_fixed"),
        "operational_changes": operational,
        "unresolved_scientific_choices": "none",
    }


def build_resolved_study(
    request: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Resolve one intake request into a user-reviewable scientific study."""

    answers = _answers(request)
    parent_id = resolve_parent_id(answers.get("parent"), catalog, repo_root)
    matches = [
        entry
        for entry in catalog.get("entries", [])
        if isinstance(entry, Mapping)
        and entry.get("experiment_id") == parent_id
    ]
    if len(matches) != 1:
        raise StudyResolutionError(
            f"Expected one catalog parent {parent_id!r}. "
            f"Provided value: {len(matches)} entries."
        )
    parent = copy.deepcopy(dict(matches[0]))
    config_value = parent.get("authority", {}).get("config")
    config_path = Path(str(config_value))
    if config_path.is_absolute() or ".." in config_path.parts:
        raise StudyResolutionError(
            "Expected the catalog parent config to be repository-relative. "
            f"Provided value: {config_value!r}."
        )
    resolved_config = repo_root / config_path
    try:
        parent_config = json.loads(resolved_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyResolutionError(
            f"Expected a readable parent config at {resolved_config}. "
            f"Provided value: {exc!r}."
        ) from exc
    if not isinstance(parent_config, Mapping):
        raise StudyResolutionError(
            "Expected the parent config to contain a JSON object. "
            f"Provided value: {type(parent_config).__name__}."
        )
    selected_entries = select_parent_jobs(request, parent_config)
    scientific_change = _scientific_change_description(
        answers.get("scientific_changes")
    )
    request_type = _plain(answers.get("request_type"))
    budget = _validate_inherited_budget(answers, selected_entries)
    context = {
        key: copy.deepcopy(parent_config[key])
        for key in SCIENTIFIC_CONTEXT_KEYS
        if key in parent_config
    }
    official_test = parent_config.get("dataset", {}).get("official_test")
    official_test_excluded = (
        isinstance(official_test, Mapping)
        and official_test.get("enabled") is False
        and official_test.get("read_allowed") is False
    )
    if not official_test_excluded:
        raise StudyResolutionError(
            "Expected the parent dataset to explicitly forbid official-test "
            f"reads. Provided value: {official_test!r}."
        )
    scientific = {
        "request_type": request_type,
        "parent_experiment_id": parent_id,
        "purpose": _plain(answers.get("purpose")),
        "scientific_changes": (
            [] if scientific_change is None else [scientific_change]
        ),
        "keep_fixed": _plain(answers.get("keep_fixed")),
        "decision": {
            "style": _plain(answers.get("decision_style")),
            "expected_outcome": _plain(answers.get("expected_outcome")),
            "negative_evidence": _plain(answers.get("negative_evidence")),
            "inconclusive_outcome": _plain(
                answers.get("inconclusive_outcome")
            ),
        },
        "evidence_scope": _plain(answers.get("evidence_scope")),
        "official_test_policy": _plain(
            answers.get("official_test_policy")
        ),
        "budget": budget,
        "seeds": _plain(answers.get("seeds")),
        "conditional_expansion": _plain(
            answers.get("sweep_or_conditional_expansion")
        ),
        "parent_scientific_context": context,
        "jobs": [
            _scientific_job(entry, official_test_read=False)
            for entry in selected_entries
        ],
        "tk_gate": _tk_gate(
            request_type=request_type,
            scientific_change=scientific_change,
            jobs=selected_entries,
        ),
    }
    digest = canonical_sha256(scientific)
    review = request.get("user_review")
    review = review if isinstance(review, Mapping) else {}
    approval = (
        "approved"
        if review.get("scientific_summary_approved") == "yes"
        else "pending"
    )
    catalog_resolution = resolve_structured_request(
        catalog,
        _catalog_request(
            request,
            parent_id=parent_id,
            job_ids=[entry["entry_id"] for entry in selected_entries],
            scientific_change=scientific_change,
        ),
        repo_root,
    )
    document = {
        "schema_version": RESOLVED_STUDY_SCHEMA,
        "study_id": f"study_{digest}",
        "study_spec_sha256": digest,
        "status": "approved" if approval == "approved" else "ready_for_review",
        "scientific_approval": {
            "status": approval,
            "approved_study_sha256": digest if approval == "approved" else None,
        },
        "study": scientific,
        "bindings_and_provenance": {
            "catalog_resolution": catalog_resolution,
            "parent_config_path": config_path.as_posix(),
            "parent_config_sha256": sha256_file(resolved_config),
            "parent_authority": copy.deepcopy(parent.get("authority")),
            "job_evidence": [
                {
                    "job_id": entry["entry_id"],
                    **{
                        key: copy.deepcopy(entry[key])
                        for key in JOB_EVIDENCE_KEYS
                        if key in entry
                    },
                }
                for entry in selected_entries
            ],
        },
    }
    validation = validate_resolved_study(document)
    if validation["status"] == "invalid":
        raise StudyResolutionError(
            "Expected the generated scientific study to validate. "
            f"Provided value: {validation!r}."
        )
    return document


def validate_resolved_study(value: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate generated-study problems without a validator error loop."""

    issues: list[dict[str, Any]] = []

    def issue(field: str, expected: Any, provided: Any) -> None:
        issues.append(
            {"field": field, "expected": expected, "provided": provided}
        )

    if not isinstance(value, Mapping):
        issue("$", "a resolved-study mapping", type(value).__name__)
        return {
            "schema_version": STUDY_VALIDATION_SCHEMA,
            "status": "invalid",
            "issues": issues,
        }
    if value.get("schema_version") != RESOLVED_STUDY_SCHEMA:
        issue(
            "schema_version",
            RESOLVED_STUDY_SCHEMA,
            value.get("schema_version"),
        )
    study = value.get("study")
    if not isinstance(study, Mapping):
        issue("study", "a mapping", study)
        study = {}
    jobs = study.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        issue("study.jobs", "a non-empty exact job list", jobs)
        jobs = []
    identifiers: list[Any] = []
    for index, job in enumerate(jobs):
        if not isinstance(job, Mapping):
            issue(f"study.jobs[{index}]", "a job mapping", job)
            continue
        job_id = job.get("job_id")
        identifiers.append(job_id)
        if not isinstance(job_id, str) or not job_id:
            issue(f"study.jobs[{index}].job_id", "non-empty text", job_id)
        rates = job.get("raw_learning_rates_by_parameter")
        if (
            not isinstance(rates, Mapping)
            or not rates
            or any(
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or float(rate) <= 0.0
                for rate in rates.values()
            )
        ):
            issue(
                f"study.jobs[{index}].raw_learning_rates_by_parameter",
                "a non-empty positive numeric mapping",
                rates,
            )
        if job.get("official_test_read") is not False:
            issue(
                f"study.jobs[{index}].official_test_read",
                False,
                job.get("official_test_read"),
            )
    if len(identifiers) != len(set(identifiers)):
        issue("study.jobs", "unique job IDs", identifiers)
    budget = study.get("budget")
    if (
        not isinstance(budget, Mapping)
        or budget.get("job_count") != len(jobs)
    ):
        issue(
            "study.budget.job_count",
            len(jobs),
            budget.get("job_count") if isinstance(budget, Mapping) else budget,
        )
    tk_gate = study.get("tk_gate")
    if (
        not isinstance(tk_gate, Mapping)
        or tk_gate.get("functional_smoke_required") is not True
        or not isinstance(tk_gate.get("fresh_required"), bool)
    ):
        issue(
            "study.tk_gate",
            "a boolean fresh decision and functional_smoke_required=true",
            tk_gate,
        )
    digest = canonical_sha256(study)
    if value.get("study_spec_sha256") != digest:
        issue(
            "study_spec_sha256",
            digest,
            value.get("study_spec_sha256"),
        )
    if value.get("study_id") != f"study_{digest}":
        issue("study_id", f"study_{digest}", value.get("study_id"))
    approval = value.get("scientific_approval")
    approved = (
        isinstance(approval, Mapping) and approval.get("status") == "approved"
    )
    if approved and approval.get("approved_study_sha256") != digest:
        issue(
            "scientific_approval.approved_study_sha256",
            digest,
            approval.get("approved_study_sha256"),
        )
    return {
        "schema_version": STUDY_VALIDATION_SCHEMA,
        "status": (
            "invalid"
            if issues
            else "valid"
            if approved
            else "needs_user_review"
        ),
        "study_id": value.get("study_id"),
        "study_spec_sha256": digest,
        "job_count": len(jobs),
        "issues": issues,
        "next_step": (
            "repair_generated_study"
            if issues
            else "build_execution_attempts"
            if approved
            else "ask_for_scientific_approval"
        ),
    }


__all__ = [
    "RESOLVED_STUDY_SCHEMA",
    "STUDY_VALIDATION_SCHEMA",
    "StudyResolutionError",
    "build_resolved_study",
    "resolve_parent_id",
    "select_parent_jobs",
    "validate_resolved_study",
]
