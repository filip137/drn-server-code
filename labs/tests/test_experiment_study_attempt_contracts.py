from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.control_plane.attempt import (
    ATTEMPT_SCHEMA,
    attempt_spec_sha256,
    validate_attempt,
)
from experiments.control_plane.common import ContractError
from experiments.control_plane.coverage import validate_attempt_set
from experiments.control_plane.legacy import (
    LEGACY_RESIDUE_SCHEMA,
    load_legacy_plan,
    publish_legacy_split,
    split_legacy_plan,
)
from experiments.control_plane.study import (
    STUDY_SCHEMA,
    study_bindings_sha256,
    study_spec_sha256,
    validate_study,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _scientific_fixture(root: Path) -> tuple[dict, Path]:
    config_path = root / "configs" / "study-a.json"
    manifest_path = root / "results" / "study-a" / "manifest.json"
    protocol_path = root / "docs" / "protocol-a.md"
    _write_json(
        config_path,
        {
            "architecture": "conv1",
            "dataset": "mnist",
            "epochs": 1,
            "seed": 0,
        },
    )
    _write_json(
        manifest_path,
        {"entries": [{"case_id": "case-a", "seed": 0}]},
    )
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text("# Protocol A\n", encoding="utf-8")

    study = {
        "schema_version": STUDY_SCHEMA,
        "study": {
            "study_id": "study-a",
            "title": "Study A",
            "evidence_scope": "diagnostic",
            "hypothesis": {
                "statement": "Treatment changes the validation metric.",
                "control": "Control A",
                "treatments": ["Treatment A"],
                "expected_direction": "Treatment A is lower.",
                "decision_rule": {
                    "metric": "validation_loss",
                    "split": "validation",
                    "checkpoint_role": "final",
                    "seed_aggregation": "single prespecified seed",
                    "comparison": "Treatment A minus Control A",
                    "minimum_meaningful_effect": 0.0,
                    "effect_units": "loss",
                },
                "supports_if": "The declared treatment is lower.",
                "does_not_support_if": "The declared treatment is not lower.",
                "inconclusive_if": "Any required artifact is missing.",
            },
            "design": {
                "artifacts": {
                    "scientific_config_sha256": _sha(config_path),
                    "study_manifest_sha256": _sha(manifest_path),
                    "manifest_job_count_pointer": "/entries",
                },
                "axes": [
                    {
                        "name": "case",
                        "config_path": "/cases",
                        "ordered_values": ["case-a"],
                    }
                ],
                "case_ids": ["case-a"],
                "seeds": [0],
                "conditional_expansion": {
                    "enabled": False,
                    "rule": None,
                    "maximum_additional_jobs": 0,
                },
                "expected_initial_job_count": 1,
                "maximum_total_job_count": 1,
                "protocols": [
                    {
                        "authority_id": "protocol-a",
                        "sha256": _sha(protocol_path),
                    }
                ],
            },
            "acceptance": {
                "required_coverage": "The one declared case.",
                "allowed_exclusions": [],
                "failure_handling": "Fail closed.",
                "required_scientific_gates": ["tk_reference"],
                "local_validation_required": True,
                "review_required": True,
            },
            "reporting": {
                "comparisons": [
                    {
                        "comparison_id": "comparison-a",
                        "card_schema_version": "drn-result-card/v1",
                    }
                ]
            },
        },
        "bindings": {
            "scientific_config_path": "configs/study-a.json",
            "study_manifest_path": "results/study-a/manifest.json",
            "protocols": [
                {
                    "authority_id": "protocol-a",
                    "path": "docs/protocol-a.md",
                }
            ],
            "reporting": {
                "comparisons": [
                    {
                        "comparison_id": "comparison-a",
                        "card_path": (
                            "result_registry/cards/diagnostics/"
                            "comparison-a.json"
                        ),
                        "review_path": (
                            "result_registry/reviews/comparison-a.json"
                        ),
                        "final_results_anchor": (
                            "docs/results/index.md#comparison-a"
                        ),
                    }
                ],
                "final_results_page": "docs/results/index.md",
            },
        },
        "approval": {
            "status": "approved",
            "approved_by": "Filip",
            "approved_at": "2026-07-28T12:00:00+02:00",
            "approved_study_sha256": None,
            "amendment_of": None,
        },
        "provenance": {
            "origin": "native-test",
            "source_path": None,
            "source_sha256": None,
            "notes": [],
        },
    }
    study["approval"]["approved_study_sha256"] = study_spec_sha256(study)
    study_path = root / "docs" / "experiment_studies" / "study-a.json"
    _write_json(study_path, study)
    return study, study_path


def _attempt_fixture(
    root: Path,
    study: dict,
    study_path: Path,
) -> dict:
    profile_path = root / "launch_tools" / "profiles" / "trex.json"
    routed_path = root / "results" / "attempt-a" / "staging" / "route.json"
    environment_path = root / "configs" / "environments" / "trex.json"
    release_path = (
        root / "experiments" / "control_plane" / "releases" / "release-a.json"
    )
    _write_json(
        profile_path,
        {"host": "trex", "tmux_session": "validation-a"},
    )
    _write_json(
        routed_path,
        {"entries": [{"case_id": "case-a", "host": "trex"}]},
    )
    _write_json(
        environment_path,
        {"python": "3.12", "torch": "2.5", "device": "cuda:0"},
    )
    _write_json(
        release_path,
        {
            "release_id": "control-plane-a",
            "study_schema": STUDY_SCHEMA,
            "attempt_schema": ATTEMPT_SCHEMA,
        },
    )
    bundle = "results/attempt-a"
    requirements = []
    for role in ("smoke", "tk_reference", "scheduled_run_preflight"):
        requirements.append(
            {
                "role": role,
                "required": True,
                "schema_version": f"{role}-receipt/v1",
                "receipt_path": f"{bundle}/preflight/{role}.json",
                "subject_id": f"{role}-subject",
                "producer_source_id": "1" * 40,
            }
        )
    attempt = {
        "schema_version": ATTEMPT_SCHEMA,
        "attempt": {
            "attempt_id": "attempt-a",
            "study_ref": {
                "path": str(study_path.relative_to(root)),
                "study_id": study["study"]["study_id"],
                "study_spec_sha256": study_spec_sha256(study),
                "study_bindings_sha256": study_bindings_sha256(study),
            },
            "supersedes_attempt_id": None,
            "source": {
                "git_commit": "2" * 40,
                "control_plane_release_id": "control-plane-a",
                "control_plane_release_path": str(
                    release_path.relative_to(root)
                ),
                "control_plane_release_sha256": _sha(release_path),
            },
            "target": {
                "kind": "ssh_tmux",
                "host": "trex",
                "device": "cuda:0",
                "profile_path": str(profile_path.relative_to(root)),
                "profile_sha256": _sha(profile_path),
            },
            "execution": {
                "launcher": [
                    "python",
                    "-m",
                    "experiments.validation",
                    "--attempt",
                    "attempt-a",
                ],
                "collector": [
                    "python",
                    "-m",
                    "experiments.validation",
                    "collect",
                ],
                "validator": [
                    "python",
                    "-m",
                    "experiments.validation",
                    "validate",
                ],
                "routed_manifest": {
                    "path": str(routed_path.relative_to(root)),
                    "sha256": _sha(routed_path),
                    "source_study_manifest_sha256": study["study"]["design"][
                        "artifacts"
                    ]["study_manifest_sha256"],
                    "job_count_pointer": "/entries",
                    "job_count": 1,
                    "selection": {
                        "mode": "all",
                        "job_id_key": "case_id",
                        "selected_study_job_ids": ["case-a"],
                    },
                },
                "environment_contract": {
                    "path": str(environment_path.relative_to(root)),
                    "sha256": _sha(environment_path),
                },
            },
            "preflight": {
                "outer_receipt_path": f"{bundle}/preflight/outer.json",
                "requirements": requirements,
            },
            "storage": {
                "local_bundle_path": bundle,
                "remote_staging": [
                    {
                        "host": "trex",
                        "path": "/home/filip/results/attempt-a",
                        "intended_local_destination": bundle,
                    }
                ],
            },
            "deadlines": {
                "run_class": "validation",
                "expected_duration_seconds": 120,
                "hard_deadline_seconds": 300,
                "scheduler_start_timeout_seconds": None,
            },
        },
        "authorization": {
            "status": "authorized",
            "authorized_by": "Filip",
            "authorized_at": "2026-07-28T12:05:00+02:00",
            "authorized_attempt_sha256": None,
        },
        "provenance": {"origin": "native-test", "notes": []},
    }
    attempt["authorization"]["authorized_attempt_sha256"] = (
        attempt_spec_sha256(attempt)
    )
    return attempt


def test_approved_study_and_authorized_trex_validation_attempt(
    tmp_path: Path,
) -> None:
    study, study_path = _scientific_fixture(tmp_path)
    study_summary = validate_study(
        study,
        repo_root=tmp_path,
        verify_bindings=True,
        require_approved=True,
    )
    attempt = _attempt_fixture(tmp_path, study, study_path)
    attempt_summary = validate_attempt(
        attempt,
        repo_root=tmp_path,
        verify_files=True,
        require_authorized=True,
    )
    assert study_summary["approval_status"] == "approved"
    assert attempt_summary["authorization_status"] == "authorized"
    assert attempt_summary["target_kind"] == "ssh_tmux"
    assert attempt_summary["run_class"] == "validation"


def test_binding_move_preserves_scientific_approval_but_requires_new_attempt(
    tmp_path: Path,
) -> None:
    study, study_path = _scientific_fixture(tmp_path)
    attempt = _attempt_fixture(tmp_path, study, study_path)
    old_spec_hash = study_spec_sha256(study)
    old_binding_hash = study_bindings_sha256(study)

    moved_path = tmp_path / "configs" / "moved" / "study-a.json"
    moved_path.parent.mkdir(parents=True)
    moved_path.write_bytes((tmp_path / "configs" / "study-a.json").read_bytes())
    moved_study = deepcopy(study)
    moved_study["bindings"]["scientific_config_path"] = (
        "configs/moved/study-a.json"
    )
    _write_json(study_path, moved_study)

    assert study_spec_sha256(moved_study) == old_spec_hash
    assert study_bindings_sha256(moved_study) != old_binding_hash
    validate_study(
        moved_study,
        repo_root=tmp_path,
        verify_bindings=True,
        require_approved=True,
    )
    with pytest.raises(ContractError) as mismatch:
        validate_attempt(attempt, repo_root=tmp_path)
    assert mismatch.value.code == "study_bindings_hash_mismatch"

    replacement = deepcopy(attempt)
    replacement["attempt"]["study_ref"]["study_bindings_sha256"] = (
        study_bindings_sha256(moved_study)
    )
    with pytest.raises(ContractError) as stale_authorization:
        validate_attempt(replacement, repo_root=tmp_path)
    assert stale_authorization.value.code == "authorization_hash_mismatch"

    replacement["authorization"]["authorized_attempt_sha256"] = (
        attempt_spec_sha256(replacement)
    )
    validate_attempt(
        replacement,
        repo_root=tmp_path,
        verify_files=True,
        require_authorized=True,
    )


def test_scientific_change_invalidates_scientific_approval(
    tmp_path: Path,
) -> None:
    study, _ = _scientific_fixture(tmp_path)
    changed = deepcopy(study)
    changed["study"]["design"]["seeds"] = [1]
    with pytest.raises(ContractError) as error:
        validate_study(changed, repo_root=tmp_path)
    assert error.value.code == "approval_hash_mismatch"


def test_operational_change_invalidates_only_attempt_authorization(
    tmp_path: Path,
) -> None:
    study, study_path = _scientific_fixture(tmp_path)
    attempt = _attempt_fixture(tmp_path, study, study_path)
    changed = deepcopy(attempt)
    changed["attempt"]["target"]["host"] = "trex-alternate"
    with pytest.raises(ContractError) as error:
        validate_attempt(changed, repo_root=tmp_path)
    assert error.value.code == "authorization_hash_mismatch"
    validate_study(
        study,
        repo_root=tmp_path,
        verify_bindings=True,
        require_approved=True,
    )


@pytest.mark.parametrize(
    ("run_class", "expected_seconds", "hard_seconds", "error_code"),
    [
        ("validation", 600, 600, "validation_duration_too_long"),
        ("production", 599, 600, "production_duration_too_short"),
    ],
)
def test_run_class_duration_boundary(
    tmp_path: Path,
    run_class: str,
    expected_seconds: int,
    hard_seconds: int,
    error_code: str,
) -> None:
    study, study_path = _scientific_fixture(tmp_path)
    attempt = _attempt_fixture(tmp_path, study, study_path)
    attempt["attempt"]["deadlines"].update(
        {
            "run_class": run_class,
            "expected_duration_seconds": expected_seconds,
            "hard_deadline_seconds": hard_seconds,
        }
    )
    attempt["authorization"]["status"] = "draft"
    attempt["authorization"]["authorized_by"] = None
    attempt["authorization"]["authorized_at"] = None
    attempt["authorization"]["authorized_attempt_sha256"] = None
    with pytest.raises(ContractError) as error:
        validate_attempt(attempt, repo_root=tmp_path)
    assert error.value.code == error_code


def test_study_required_scientific_gate_cannot_be_omitted(
    tmp_path: Path,
) -> None:
    study, study_path = _scientific_fixture(tmp_path)
    attempt = _attempt_fixture(tmp_path, study, study_path)
    attempt["attempt"]["preflight"]["requirements"] = [
        requirement
        for requirement in attempt["attempt"]["preflight"]["requirements"]
        if requirement["role"] != "tk_reference"
    ]
    attempt["authorization"]["status"] = "draft"
    attempt["authorization"]["authorized_by"] = None
    attempt["authorization"]["authorized_at"] = None
    attempt["authorization"]["authorized_attempt_sha256"] = None
    with pytest.raises(ContractError) as error:
        validate_attempt(attempt, repo_root=tmp_path)
    assert error.value.code == "scientific_gate_missing"


def test_exact_key_error_reports_missing_and_unknown_separately(
    tmp_path: Path,
) -> None:
    study, _ = _scientific_fixture(tmp_path)
    malformed = deepcopy(study)
    del malformed["provenance"]
    malformed["mystery"] = {}
    with pytest.raises(ContractError) as error:
        validate_study(malformed, repo_root=tmp_path)
    assert error.value.code == "object_keys_mismatch"
    assert error.value.provided == {
        "missing": ["provenance"],
        "unknown": ["mystery"],
    }
    assert error.value.as_dict()["launched_job_count"] == 0
    assert error.value.as_dict()["recovery_scope"] == "pre_arm_repairable"


def test_legacy_split_is_deterministic_and_never_transfers_approval() -> None:
    plan_path = (
        REPO_ROOT
        / "docs"
        / "experiment_plans"
        / "perfectdiode-conv12-best-observed-confirmation-20260727-v1.md"
    )
    legacy = load_legacy_plan(plan_path)
    first_study, first_residue = split_legacy_plan(
        plan_path,
        repo_root=REPO_ROOT,
    )
    second_study, second_residue = split_legacy_plan(
        plan_path,
        repo_root=REPO_ROOT,
    )
    assert first_study == second_study
    assert first_residue == second_residue
    assert first_study["approval"]["status"] == "review_required"
    assert first_study["approval"]["approved_by"] is None
    assert "execution" not in first_study["study"]
    assert "storage" not in first_study["study"]
    assert first_residue["schema_version"] == LEGACY_RESIDUE_SCHEMA
    assert first_residue["status"] == "not_launch_authority"
    assert first_residue["legacy_approval"] == legacy["approval"]
    assert first_study["study"]["hypothesis"] == legacy["hypothesis"]
    assert first_study["study"]["design"]["axes"] == legacy["sweep"]["axes"]
    assert first_study["study"]["design"]["case_ids"] == legacy["sweep"][
        "case_ids"
    ]
    assert first_study["study"]["design"]["seeds"] == legacy["sweep"]["seeds"]
    assert first_study["bindings"]["scientific_config_path"] == legacy[
        "sweep"
    ]["config_path"]
    assert first_study["bindings"]["study_manifest_path"] == legacy["sweep"][
        "manifest_path"
    ]
    assert (
        first_residue["operational_residue"]["execution"]
        == legacy["execution"]
    )
    assert (
        first_residue["operational_residue"]["storage"] == legacy["storage"]
    )


def test_every_legacy_plan_can_be_split_without_creating_authority() -> None:
    paths = sorted((REPO_ROOT / "docs" / "experiment_plans").glob("*.md"))
    assert len(paths) == 11
    for path in paths:
        study, residue = split_legacy_plan(path, repo_root=REPO_ROOT)
        assert study["approval"]["status"] in {"draft", "review_required"}
        assert residue["status"] == "not_launch_authority"
        assert residue["study_spec_sha256"] == study_spec_sha256(study)


def test_split_publication_checks_both_collisions_before_writing(
    tmp_path: Path,
) -> None:
    study_output = tmp_path / "study.json"
    residue_output = tmp_path / "residue.json"
    study_output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(ContractError) as error:
        publish_legacy_split(
            study_document={"study": "unused"},
            residue_document={"residue": "unused"},
            study_output=study_output,
            residue_output=residue_output,
        )
    assert error.value.code == "output_exists"
    assert not residue_output.exists()


def test_sharded_attempt_set_requires_exact_nonoverlapping_coverage(
    tmp_path: Path,
) -> None:
    study, study_path = _scientific_fixture(tmp_path)
    manifest_path = tmp_path / study["bindings"]["study_manifest_path"]
    _write_json(
        manifest_path,
        {
            "entries": [
                {"case_id": "case-a", "seed": 0},
                {"case_id": "case-b", "seed": 0},
            ]
        },
    )
    study["study"]["design"]["artifacts"]["study_manifest_sha256"] = _sha(
        manifest_path
    )
    study["study"]["design"]["axes"][0]["ordered_values"] = [
        "case-a",
        "case-b",
    ]
    study["study"]["design"]["case_ids"] = ["case-a", "case-b"]
    study["study"]["design"]["expected_initial_job_count"] = 2
    study["study"]["design"]["maximum_total_job_count"] = 2
    study["approval"]["approved_study_sha256"] = study_spec_sha256(study)
    _write_json(study_path, study)

    first = _attempt_fixture(tmp_path, study, study_path)
    first_route = first["attempt"]["execution"]["routed_manifest"]
    first_route["selection"]["mode"] = "subset"
    first["authorization"]["authorized_attempt_sha256"] = (
        attempt_spec_sha256(first)
    )

    def replace_attempt_id(value: object) -> object:
        if isinstance(value, str):
            return value.replace("attempt-a", "attempt-b")
        if isinstance(value, list):
            return [replace_attempt_id(item) for item in value]
        if isinstance(value, dict):
            return {
                key: replace_attempt_id(item) for key, item in value.items()
            }
        return value

    second = replace_attempt_id(deepcopy(first))
    assert isinstance(second, dict)
    second_route = second["attempt"]["execution"]["routed_manifest"]
    second_route_path = tmp_path / second_route["path"]
    _write_json(
        second_route_path,
        {"entries": [{"case_id": "case-b", "host": "trex"}]},
    )
    second_route["sha256"] = _sha(second_route_path)
    second_route["selection"]["selected_study_job_ids"] = ["case-b"]
    second["authorization"]["authorized_attempt_sha256"] = (
        attempt_spec_sha256(second)
    )

    first_path = tmp_path / "docs" / "execution_attempts" / "attempt-a.json"
    second_path = tmp_path / "docs" / "execution_attempts" / "attempt-b.json"
    _write_json(first_path, first)
    _write_json(second_path, second)
    summary = validate_attempt_set(
        study_path=study_path,
        attempt_paths=[first_path, second_path],
        repo_root=tmp_path,
        verify_files=True,
        require_authorized=True,
    )
    assert summary["study_job_count"] == 2
    assert summary["attempt_count"] == 2

    _write_json(
        second_route_path,
        {"entries": [{"case_id": "case-a", "host": "trex"}]},
    )
    second_route["sha256"] = _sha(second_route_path)
    second_route["selection"]["selected_study_job_ids"] = ["case-a"]
    second["authorization"]["authorized_attempt_sha256"] = (
        attempt_spec_sha256(second)
    )
    _write_json(second_path, second)
    with pytest.raises(ContractError) as duplicate:
        validate_attempt_set(
            study_path=study_path,
            attempt_paths=[first_path, second_path],
            repo_root=tmp_path,
            verify_files=True,
            require_authorized=True,
        )
    assert duplicate.value.code == "duplicate_study_job_route"


def test_native_study_template_is_structurally_valid() -> None:
    template_path = (
        REPO_ROOT
        / "skills"
        / "run-experiment-pipeline"
        / "assets"
        / "experiment-study-template-v1.json"
    )
    document = json.loads(template_path.read_text(encoding="utf-8"))
    summary = validate_study(document, repo_root=REPO_ROOT)
    assert summary["approval_status"] == "draft"
