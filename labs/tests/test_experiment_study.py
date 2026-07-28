from __future__ import annotations

import copy
from pathlib import Path

from experiments.experiment_catalog import load_catalog
from experiments.experiment_request import parse_request_text, validate_request
from experiments.experiment_study import (
    build_resolved_study,
    resolve_parent_id,
    validate_resolved_study,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUEST_PATH = (
    REPO_ROOT
    / "docs"
    / "experiment_requests"
    / "perfectdiode-conv1-best-rho-continuation-20260728.md"
)
CATALOG_PATH = REPO_ROOT / "experiments" / "experiment_catalog.json"


def _request() -> dict:
    value = parse_request_text(REQUEST_PATH.read_text())
    assert validate_request(value)["status"] == "valid"
    return value


def test_parent_id_embedded_in_prose_resolves_exactly() -> None:
    catalog = load_catalog(CATALOG_PATH)
    parent = _request()["user_answers"]["parent"]
    assert resolve_parent_id(parent, catalog, REPO_ROOT) == (
        "perfectdiode-conv12-best-observed-confirmation-20260727-v1"
    )


def test_filled_request_resolves_six_exact_conv1_jobs() -> None:
    study = build_resolved_study(
        _request(),
        load_catalog(CATALOG_PATH),
        repo_root=REPO_ROOT,
    )

    assert study["status"] == "ready_for_review"
    assert study["scientific_approval"]["status"] == "pending"
    assert [job["job_id"] for job in study["study"]["jobs"]] == [
        "pdconfirm_conv1_baseline_sgd_seed0",
        "pdconfirm_conv1_baseline_adam_seed0",
        "pdconfirm_conv1_ours_sgd_seed0",
        "pdconfirm_conv1_ours_adam_seed0",
        "pdconfirm_conv1_legacy_sgd_seed0",
        "pdconfirm_conv1_legacy_adam_seed0",
    ]
    assert study["study"]["budget"] == {
        "description": "10 epochs per job",
        "epochs_per_job": 10,
        "steps_per_job": 34380,
        "job_count": 6,
    }
    assert study["study"]["scientific_changes"] == []
    assert study["study"]["tk_gate"]["fresh_required"] is False
    assert study["study"]["tk_gate"]["prior_result_reused"] is True
    assert (
        study["bindings_and_provenance"]["catalog_resolution"]["status"]
        == "derived"
    )
    assert validate_resolved_study(study)["status"] == "needs_user_review"


def test_scientific_approval_is_one_user_field() -> None:
    request = _request()
    request["user_review"]["scientific_summary_approved"] = "yes"
    study = build_resolved_study(
        request,
        load_catalog(CATALOG_PATH),
        repo_root=REPO_ROOT,
    )
    assert study["status"] == "approved"
    assert study["scientific_approval"] == {
        "status": "approved",
        "approved_study_sha256": study["study_spec_sha256"],
    }
    assert validate_resolved_study(study)["status"] == "valid"


def test_operational_bindings_do_not_change_scientific_hash() -> None:
    study = build_resolved_study(
        _request(),
        load_catalog(CATALOG_PATH),
        repo_root=REPO_ROOT,
    )
    moved = copy.deepcopy(study)
    moved["bindings_and_provenance"]["parent_config_path"] = (
        "some/other/location.json"
    )
    moved["bindings_and_provenance"]["target"] = "trex"
    assert validate_resolved_study(moved)["issues"] == []
    assert moved["study_spec_sha256"] == study["study_spec_sha256"]


def test_scientific_learning_rate_change_changes_study_identity() -> None:
    study = build_resolved_study(
        _request(),
        load_catalog(CATALOG_PATH),
        repo_root=REPO_ROOT,
    )
    changed = copy.deepcopy(study)
    changed["study"]["jobs"][0]["raw_learning_rates_by_parameter"][
        "ConvWeight_0"
    ] *= 2
    validation = validate_resolved_study(changed)
    assert validation["status"] == "invalid"
    assert validation["study_spec_sha256"] != study["study_spec_sha256"]
