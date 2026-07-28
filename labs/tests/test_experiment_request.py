from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from experiments import experiment_request


REPO_ROOT = Path(__file__).resolve().parents[2]
FILLED_REQUEST = (
    REPO_ROOT
    / "docs"
    / "experiment_requests"
    / "perfectdiode-conv1-best-rho-continuation-20260728.md"
)
REQUEST_TEMPLATE = REPO_ROOT / "docs" / "experiment_request_template.md"


def test_filled_repository_request_normalizes_without_operational_plumbing() -> None:
    request = experiment_request.load_request(FILLED_REQUEST)
    validation = experiment_request.validate_request(request)

    assert request["schema_version"] == "experiment-request/v1"
    assert request["source"]["format"] == "full_markdown/v1"
    assert request["user_answers"]["request_type"] == "continuation"
    assert request["user_answers"]["scientific_changes"].startswith("none")
    assert "baseline, ours, and legacy" in request["user_answers"]["cases_subset"]
    assert request["user_answers"]["official_test_policy"] == "forbidden"
    assert request["user_answers"]["evidence_scope"] == "ordinary_diagnostic"
    assert request["user_answers"]["executor_adapt_operations"] == "yes"
    assert request["user_answers"]["executor_launch_after_checks"] == "yes"
    assert validation["status"] == "valid"
    assert validation["missing_fields"] == []
    assert validation["unresolved_fields"] == []

    serialized = json.dumps(request)
    for forbidden_user_requirement in (
        "receipt_schemas",
        "receipt_path",
        "environment_hash",
        "python_path",
        "cuda_version",
    ):
        assert forbidden_user_requirement not in serialized


def test_agent_summary_and_review_are_separate_from_user_answers() -> None:
    request = experiment_request.load_request(FILLED_REQUEST)

    assert "catalog_and_parent_resolution" in request["agent_resolved_summary"]
    assert (
        request["agent_resolved_summary"]["catalog_and_parent_resolution"][
            "resolution"
        ]
        == "derived from parent"
    )
    assert request["user_review"]["scientific_summary_approved"] == "pending"
    assert "purpose_hypothesis" not in request["user_answers"]

    request["user_answers"]["purpose"] = None
    validation = experiment_request.validate_request(request)
    assert validation["status"] == "needs_user_input"
    assert validation["missing_fields"] == ["purpose"]


def test_template_reports_every_missing_user_owned_field_together() -> None:
    request = experiment_request.load_request(REQUEST_TEMPLATE)
    validation = experiment_request.validate_request(request)

    assert validation["status"] == "needs_user_input"
    assert {
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
    } == set(validation["missing_fields"])
    assert not any(
        word in field
        for field in validation["missing_fields"]
        for word in ("hash", "receipt", "environment", "path")
    )


def test_valid_compact_conversational_form() -> None:
    text = """\
Experiment: Conv1 fixed-LR continuation
Type: continuation
Parent: pd-parent-v1
Purpose: Confirm that all selected jobs complete.
Cases/subset: baseline and ours, SGD and Adam
Scientific changes: none
Keep fixed: inherit from parent
Budget / seeds / sweep: 10 epochs; seed 0; no sweep
Decision style: standard continuation
Evidence / official-test policy: ordinary diagnostic; official test forbidden
Compute / distribution: local GPU; two jobs
Executor: adapt operations yes; launch after checks yes
"""
    request = experiment_request.parse_request_text(text)
    validation = experiment_request.validate_request(request)

    assert request["source"]["format"] == "compact_conversational_markdown/v1"
    assert request["user_answers"]["scientific_changes"] == "none"
    assert request["user_answers"]["cases_subset"] == (
        "baseline and ours, SGD and Adam"
    )
    assert request["user_answers"]["keep_fixed"] == "inherit_from_parent"
    assert (
        request["composite_answers"]["budget_seeds_sweep"]
        == "10 epochs; seed 0; no sweep"
    )
    assert validation["status"] == "valid"
    assert validation["missing_fields"] == []
    assert validation["unresolved_fields"] == ["user_answers.keep_fixed"]


def test_compact_form_aggregates_missing_groups_in_one_report() -> None:
    request = experiment_request.parse_request_text(
        """\
Experiment: test
Type:
Parent:
Purpose:
Cases/subset:
Scientific changes:
Keep fixed:
Budget / seeds / sweep:
Decision style:
Evidence / official-test policy:
Compute / distribution:
Executor:
"""
    )
    validation = experiment_request.validate_request(request)

    assert validation["status"] == "needs_user_input"
    assert validation["missing_fields"] == [
        "request_type",
        "parent",
        "purpose",
        "cases_subset",
        "scientific_changes",
        "keep_fixed",
        "decision_style",
        "budget_seeds_sweep",
        "evidence_official_test_policy",
        "compute_distribution",
        "executor_authority",
    ]


def test_inherit_from_parent_is_complete_but_flagged_for_resolution() -> None:
    text = """\
Experiment: inherited continuation
Type: inherit from parent
Parent: parent-v1
Purpose: inherit from parent
Cases/subset: inherit from parent
Scientific changes: none
Keep fixed: inherit from parent
Budget / seeds / sweep: inherit from parent
Decision style: inherit from parent
Evidence / official-test policy: inherit from parent
Compute / distribution: inherit from parent
Executor: inherit from parent
"""
    validation = experiment_request.validate_request(
        experiment_request.parse_request_text(text)
    )

    assert validation["status"] == "valid"
    assert validation["missing_fields"] == []
    assert validation["unresolved_fields"] == [
        "composite_answers.budget_seeds_sweep",
        "composite_answers.compute_distribution",
        "composite_answers.evidence_official_test_policy",
        "composite_answers.executor_authority",
        "user_answers.cases_subset",
        "user_answers.decision_style",
        "user_answers.keep_fixed",
        "user_answers.purpose",
        "user_answers.request_type",
    ]


def test_multiple_checked_options_is_invalid_not_silently_selected() -> None:
    text = FILLED_REQUEST.read_text(encoding="utf-8").replace(
        "- `[ ]` Repeat an existing study",
        "- `[x]` Repeat an existing study",
    )
    request = experiment_request.parse_request_text(text)
    validation = experiment_request.validate_request(request)

    assert validation["status"] == "invalid"
    assert any(
        issue["field"] == "request_type"
        and issue["expected"] == "exactly one selected option"
        for issue in validation["invalid_fields"]
    )


def test_unknown_enumerated_answer_reports_expected_then_provided() -> None:
    request = experiment_request.load_request(FILLED_REQUEST)
    request["user_answers"]["request_type"] = "surprise mode"
    validation = experiment_request.validate_request(request)

    assert validation["status"] == "invalid"
    issue = next(
        item
        for item in validation["invalid_fields"]
        if item["field"] == "request_type"
    )
    assert list(issue) == ["field", "expected", "provided"]
    assert issue["provided"] == "surprise mode"


def test_review_is_optional_for_intake_and_aggregated_when_required() -> None:
    request = experiment_request.load_request(FILLED_REQUEST)

    assert experiment_request.validate_request(request)["status"] == "valid"
    reviewed = experiment_request.validate_request(request, require_review=True)
    assert reviewed["status"] == "needs_user_input"
    assert reviewed["review_status"] == "pending"
    assert reviewed["missing_fields"] == [
        "user_review.scientific_summary_approved",
        "user_review.execution_proposal_authorized",
    ]


def test_cli_validate_emits_normalized_json_and_useful_exit_codes(
    tmp_path: Path,
    capsys,
) -> None:
    valid_code = experiment_request.main(["validate", str(FILLED_REQUEST)])
    valid_output = json.loads(capsys.readouterr().out)
    assert valid_code == 0
    assert valid_output["validation"]["status"] == "valid"
    assert valid_output["request"]["user_answers"]["scientific_changes"].startswith(
        "none"
    )

    incomplete = tmp_path / "incomplete.md"
    incomplete.write_text("Experiment: small test\n", encoding="utf-8")
    incomplete_code = experiment_request.main(["validate", str(incomplete)])
    incomplete_output = json.loads(capsys.readouterr().out)
    assert incomplete_code == 2
    assert incomplete_output["validation"]["status"] == "needs_user_input"
    assert len(incomplete_output["validation"]["missing_fields"]) > 1


def test_validate_request_rejects_wrong_schema_without_throwing() -> None:
    request = experiment_request.load_request(FILLED_REQUEST)
    malformed = deepcopy(request)
    malformed["schema_version"] = "experiment-request/v99"

    validation = experiment_request.validate_request(malformed)
    assert validation["status"] == "invalid"
    assert validation["invalid_fields"][0] == {
        "field": "schema_version",
        "expected": "experiment-request/v1",
        "provided": "experiment-request/v99",
    }
