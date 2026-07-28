from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = REPO_ROOT / "skills" / "run-experiment-pipeline"
VALIDATOR_PATH = SKILL_ROOT / "scripts" / "validate_experiment_plan.py"
TEMPLATE_PATH = SKILL_ROOT / "assets" / "experiment-plan-template.md"


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_experiment_plan",
        VALIDATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def approved_plan(tmp_path: Path):
    validator = load_validator()
    validator.REPO_ROOT = tmp_path

    config_path = tmp_path / "configs" / "test.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text('{"axis":[1,2]}\n', encoding="utf-8")

    manifest_path = tmp_path / "results" / "test-experiment" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"entries":[{"id":"a"},{"id":"b"}]}\n', encoding="utf-8")

    plan = validator.load_plan(TEMPLATE_PATH)
    plan["experiment_id"] = "test-experiment"
    plan["title"] = "Test experiment"
    plan["hypothesis"] = {
        "statement": "Treatment increases selected-checkpoint accuracy.",
        "control": "baseline",
        "treatments": ["ours"],
        "expected_direction": "ours greater than baseline",
        "decision_rule": {
            "metric": "selected_checkpoint_test_accuracy",
            "split": "ordinary_mnist_test",
            "checkpoint_role": "minimum_validation_loss",
            "seed_aggregation": "mean_and_population_std",
            "comparison": "ours_minus_baseline",
            "minimum_meaningful_effect": 0.01,
            "effect_units": "accuracy_fraction",
        },
        "supports_if": "The matched mean difference is at least 0.01.",
        "does_not_support_if": "The matched mean difference is below 0.01.",
        "inconclusive_if": "Required seed coverage is incomplete.",
    }
    plan["sweep"] = {
        "config_path": "configs/test.json",
        "config_sha256": sha256(config_path),
        "manifest_path": "results/test-experiment/manifest.json",
        "manifest_sha256": sha256(manifest_path),
        "manifest_job_count_pointer": "/entries",
        "axes": [
            {
                "name": "weight_max",
                "config_path": "/axis",
                "ordered_values": [0.0001, 0.001],
            }
        ],
        "case_ids": ["baseline", "ours"],
        "seeds": [0],
        "conditional_expansion": {
            "enabled": False,
            "rule": None,
            "maximum_additional_jobs": 0,
        },
        "expected_initial_job_count": 2,
        "maximum_total_job_count": 2,
        "undeclared_fields_fixed_by": ["configs/test.json"],
    }
    plan["execution"] = {
        "launcher": ["python", "-m", "experiments.mnist_conv", "sweep"],
        "collector": ["python", "-m", "experiments.mnist_conv", "collect"],
        "validator": ["python", "labs/tools/render_result_registry.py", "--check"],
        "preflight": {
            "smoke_required": True,
            "tk_reference_required": True,
            "scheduled_run_preflight_required": False,
            "receipt_path": "results/test-experiment/preflight/receipt.json",
            "receipt_schemas": {
                "smoke": "test-smoke-receipt/v1",
                "tk_reference": "test-tk-reference-receipt/v1",
                "scheduled_run_preflight": None,
            },
            "receipt_bindings": {
                "smoke": {
                    "receipt_path": (
                        "results/test-experiment/preflight/smoke/receipt.json"
                    ),
                    "subject_id": "test-smoke",
                    "producer_source_id": "a" * 40,
                },
                "tk_reference": {
                    "receipt_path": (
                        "results/test-experiment/preflight/tk/selection.json"
                    ),
                    "subject_id": "test-tk-reference",
                    "producer_source_id": "b" * 40,
                },
                "scheduled_run_preflight": None,
            },
        },
    }
    plan["storage"] = {
        "root_alias": "REPO_ROOT",
        "local_results_root": "results",
        "local_bundle_path": "results/test-experiment",
        "remote_staging": [],
    }
    plan["reporting"] = {
        "progress_tracker": "docs/current_experiments.md",
        "comparison_cards": [
            {
                "comparison_id": "test-comparison",
                "card_schema_version": "drn-result-card/v1",
                "card_path": "result_registry/cards/diagnostics/test-comparison.json",
                "review_path": "result_registry/reviews/test-comparison.json",
                "final_results_anchor": "docs/results/index.md#test-comparison",
            }
        ],
        "final_results_page": "docs/results/index.md",
    }
    plan["completion"] = {
        "required_coverage": "All planned cases and seeds.",
        "allowed_exclusions": [],
        "failure_handling": "Record failed or missing rows; do not replace them.",
        "local_validation_required": True,
        "review_required": True,
    }
    plan["approval"] = {
        "status": "approved",
        "approved_by": "Filip",
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "amendment_of": None,
    }

    plan_path = tmp_path / "docs" / "experiment_plans" / "test-experiment.md"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        "# Experiment Plan: Test experiment\n\n```json\n"
        + json.dumps(plan, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return validator, plan_path, plan


def validate(validator, plan_path: Path, plan: dict) -> None:
    validator.validate_plan(
        plan_path,
        plan,
        require_approved=True,
        verify_files=True,
    )


def test_approved_plan_binds_hypothesis_sweep_storage_and_reporting(tmp_path: Path) -> None:
    validator, plan_path, plan = approved_plan(tmp_path)
    validate(validator, plan_path, plan)


def test_legacy_receipt_contract_is_catalog_only_and_launch_blocked(
    tmp_path: Path,
) -> None:
    validator, plan_path, plan = approved_plan(tmp_path)
    legacy = deepcopy(plan)
    preflight = legacy["execution"]["preflight"]
    del preflight["receipt_schemas"]
    del preflight["receipt_bindings"]

    with pytest.raises(validator.PlanError, match="receipt_schemas"):
        validate(validator, plan_path, legacy)

    validator.validate_plan(
        plan_path,
        legacy,
        require_approved=True,
        verify_files=True,
        allow_legacy_receipt_contract_for_catalog=True,
    )


def test_catalog_legacy_compatibility_rejects_other_malformed_preflight(
    tmp_path: Path,
) -> None:
    validator, plan_path, plan = approved_plan(tmp_path)
    malformed = deepcopy(plan)
    del malformed["execution"]["preflight"]["receipt_bindings"]

    with pytest.raises(validator.PlanError, match="receipt_bindings"):
        validator.validate_plan(
            plan_path,
            malformed,
            require_approved=True,
            verify_files=True,
            allow_legacy_receipt_contract_for_catalog=True,
        )


def test_catalog_legacy_compatibility_fully_validates_old_shape(
    tmp_path: Path,
) -> None:
    validator, plan_path, plan = approved_plan(tmp_path)
    legacy = deepcopy(plan)
    preflight = legacy["execution"]["preflight"]
    del preflight["receipt_schemas"]
    del preflight["receipt_bindings"]
    preflight["receipt_path"] = "results/other-experiment/preflight/receipt.json"

    with pytest.raises(
        validator.PlanError,
        match=(
            r"execution\.preflight\.receipt_path below "
            r"storage\.local_bundle_path"
        ),
    ):
        validator.validate_plan(
            plan_path,
            legacy,
            require_approved=True,
            verify_files=True,
            allow_legacy_receipt_contract_for_catalog=True,
        )


def test_plan_accepts_rho_corner_diagnostic_card_schema(tmp_path: Path) -> None:
    validator, plan_path, plan = approved_plan(tmp_path)
    plan["reporting"]["comparison_cards"][0]["card_schema_version"] = (
        "drn-rho-corner-diagnostic-card/v1"
    )
    validate(validator, plan_path, plan)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda plan: plan["hypothesis"].update(statement=""),
            "non-empty text for hypothesis.statement",
        ),
        (
            lambda plan: plan["sweep"]["axes"][0].update(
                ordered_values=[0.0001, 0.0001]
            ),
            "unique ordered values",
        ),
        (
            lambda plan: plan["reporting"].update(progress_tracker="docs/current_state.md"),
            "progress_tracker 'docs/current_experiments.md'",
        ),
        (
            lambda plan: plan["reporting"].update(final_results_page="docs/my_notes.md"),
            "final_results_page 'docs/results/index.md'",
        ),
    ],
)
def test_plan_rejects_missing_or_ambiguous_contract_fields(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    validator, plan_path, plan = approved_plan(tmp_path)
    malformed = deepcopy(plan)
    mutation(malformed)

    with pytest.raises(validator.PlanError, match=message):
        validate(validator, plan_path, malformed)


def test_plan_rejects_manifest_job_count_mismatch(tmp_path: Path) -> None:
    validator, plan_path, plan = approved_plan(tmp_path)
    plan["sweep"]["expected_initial_job_count"] = 3
    plan["sweep"]["maximum_total_job_count"] = 3

    with pytest.raises(validator.PlanError, match="manifest job count 3"):
        validate(validator, plan_path, plan)


def test_plan_rejects_remote_staging_without_local_destination(tmp_path: Path) -> None:
    validator, plan_path, plan = approved_plan(tmp_path)
    plan["storage"]["remote_staging"] = [
        {
            "host": "jean-zay",
            "path": "/lustre/results/test-experiment",
        }
    ]

    with pytest.raises(validator.PlanError, match="all required keys"):
        validate(validator, plan_path, plan)


def test_plan_rejects_inner_receipt_outside_local_bundle(
    tmp_path: Path,
) -> None:
    validator, plan_path, plan = approved_plan(tmp_path)
    plan["execution"]["preflight"]["receipt_bindings"]["smoke"][
        "receipt_path"
    ] = "results/other-experiment/preflight/smoke/receipt.json"

    with pytest.raises(
        validator.PlanError,
        match=(
            r"receipt_bindings\.smoke\.receipt_path below "
            r"storage\.local_bundle_path"
        ),
    ):
        validate(validator, plan_path, plan)


def test_plan_rejects_placeholders_after_approval(tmp_path: Path) -> None:
    validator, plan_path, plan = approved_plan(tmp_path)
    plan["completion"]["failure_handling"] = "REPLACE_ME"

    with pytest.raises(validator.PlanError, match="no placeholder"):
        validate(validator, plan_path, plan)


def test_skill_delegates_scheduled_queue_validation() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert "../scheduled-run-preflight/SKILL.md" in text
    assert "plan-only command is not a payload smoke test" in text


def test_skill_requires_minimal_tracker_entry_and_exact_id_launch_gate() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert "<!-- experiment-id: <experiment-id> -->" in text
    assert "- **Testing:**" in text
    assert "- **Where:**" in text
    assert "- **Status:**" in text
    assert "validate_current_experiments.py" in text
    assert "--require-experiment-id <experiment-id>" in text
    assert "--require-launch-ready" in text
    assert "--write-gate-receipt" in text
    assert "execution.preflight.receipt_schemas" in text
    assert "execution.preflight.receipt_bindings" in text
    assert "first-write-wins structured failure" in text
    assert "observable progress at least every 60 seconds" in text
    assert text.index("## 3. Add the live tracker entry") < text.index(
        "## 4. Preflight and launch"
    )


def test_launch_request_requires_prefilled_user_intake_before_preparation() -> None:
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    template = (
        REPO_ROOT / "docs" / "experiment_request_template.md"
    ).read_text(encoding="utf-8")
    launch_request = (
        REPO_ROOT / "docs" / "experiment_launch_request.md"
    ).read_text(encoding="utf-8")
    normalized_skill = " ".join(skill.split())
    normalized_agents = " ".join(agents.split())
    normalized_template = " ".join(template.split())
    normalized_launch_request = " ".join(launch_request.split())

    assert "## Launch intake gate" in skill
    assert "../../docs/experiment_request_template.md" in skill
    assert (
        "use read-only catalog, protocol, config, and result inspection"
        in normalized_skill
    )
    assert (
        "present the fully prefilled form and ask for one confirmation"
        in normalized_skill
    )
    assert "stop before creating a plan or tracker entry" in normalized_skill

    assert "## Experiment Launch Intake" in agents
    assert "docs/experiment_request_template.md" in agents
    assert "do not continue launch preparation" in normalized_agents

    for field in (
        "Experiment:",
        "Type: continuation | repeat | derived comparison | new",
        "Parent:",
        "Cases/subset:",
        "Scientific changes:",
        "Budget / seeds / sweep:",
        "Compute / distribution:",
        "launch after checks yes|no",
    ):
        assert field in template
    assert "must not continue plan, tracker, preflight" in normalized_template

    assert "## Mandatory intake before launch preparation" in launch_request
    assert "compact conversational form" in normalized_launch_request
    assert (
        "one explicit `approved` response may approve both"
        in normalized_launch_request
    )


def test_skill_has_authoritative_fast_continuation_path() -> None:
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    normalized_skill = " ".join(skill.split())
    normalized_agents = " ".join(agents.split())

    assert "## Fast ordinary-diagnostic continuation path" in skill
    for module in (
        "experiments.experiment_request",
        "experiments.experiment_catalog",
        "experiments.experiment_study",
        "experiments.experiment_flow",
        "experiments.functional_dispatch",
    ):
        assert module in skill
    assert "one explicit `approved` response may set both review fields" in (
        normalized_skill
    )
    assert "The user never authors content hashes" in normalized_skill
    assert "fresh functional smoke, not a fresh scientific gate" in (
        normalized_skill
    )
    assert "ten-minute or two-failed-repair budget" in normalized_skill

    assert "ordinary diagnostic continuation or repeat" in normalized_agents
    assert "immutable authorized dispatch request" in normalized_agents
    assert "user is never asked to repair tracker syntax" in normalized_agents
