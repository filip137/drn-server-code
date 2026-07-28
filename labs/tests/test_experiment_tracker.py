from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from experiments.experiment_catalog import load_catalog
from experiments.experiment_flow import prepare_request
from experiments.experiment_request import parse_request_text
from experiments.experiment_tracker import (
    ExperimentTrackerError,
    tracker_entry_from_prepared,
    upsert_tracker,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUEST_PATH = (
    REPO_ROOT
    / "docs"
    / "experiment_requests"
    / "perfectdiode-conv1-best-rho-continuation-20260728.md"
)
CATALOG_PATH = REPO_ROOT / "experiments" / "experiment_catalog.json"
VALIDATOR_PATH = (
    REPO_ROOT
    / "skills"
    / "run-experiment-pipeline"
    / "scripts"
    / "validate_current_experiments.py"
)


def _prepared(*, approved: bool = True) -> dict:
    request = parse_request_text(REQUEST_PATH.read_text(encoding="utf-8"))
    if approved:
        request["user_review"]["scientific_summary_approved"] = "yes"
        request["user_review"]["execution_proposal_authorized"] = "yes"
        request["user_review"]["corrections_or_constraints"] = "none"
    else:
        request["user_review"]["scientific_summary_approved"] = "pending"
        request["user_review"]["execution_proposal_authorized"] = "pending"
    return prepare_request(
        request,
        catalog=load_catalog(CATALOG_PATH),
        repo_root=REPO_ROOT,
    )


def _validator():
    spec = importlib.util.spec_from_file_location(
        "_test_current_experiments_validator",
        VALIDATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pending_review_cannot_create_tracker_state() -> None:
    with pytest.raises(ExperimentTrackerError, match="approved prepared flow"):
        tracker_entry_from_prepared(
            _prepared(approved=False),
            status="preflighting",
            detail="Waiting for target smokes.",
        )


def test_executor_generates_valid_entry_and_updates_it_in_place(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    tracker = tmp_path / "current_experiments.md"
    tracker.write_text("# Current Experiments\n\nNone.\n", encoding="utf-8")
    preflight = tracker_entry_from_prepared(
        prepared,
        status="preflighting",
        detail="Three bounded target smokes are ready.",
        request_link="request",
    )

    assert upsert_tracker(tracker, preflight) == "updated"
    tracker.write_text(
        tracker.read_text(encoding="utf-8")
        + "\n## Another group\n\nUnrelated text stays here.\n",
        encoding="utf-8",
    )
    text = tracker.read_text(encoding="utf-8")
    validator = _validator()
    entries = validator.validate_tracker(text)
    study_id = prepared["resolved_study"]["study_id"]
    assert entries[study_id].status == "preflighting"
    assert "local:" in entries[study_id].where
    assert "trex:" in entries[study_id].where
    assert "akib:" in entries[study_id].where

    launch_ready = tracker_entry_from_prepared(
        prepared,
        status="launch-ready",
        detail="All three functional smokes passed.",
    )
    assert upsert_tracker(tracker, launch_ready) == "updated"
    updated = tracker.read_text(encoding="utf-8")
    assert upsert_tracker(tracker, launch_ready) == "unchanged"
    assert updated.count(f"<!-- experiment-id: {study_id} -->") == 1
    assert "## Another group\n\nUnrelated text stays here." in updated
    validator.validate_tracker(
        updated,
        required_experiment_id=study_id,
        require_launch_ready=True,
    )


def test_duplicate_existing_identity_is_rejected(tmp_path: Path) -> None:
    prepared = _prepared()
    entry = tracker_entry_from_prepared(
        prepared,
        status="preflighting",
        detail="Ready.",
    )
    marker = f"<!-- experiment-id: {entry['experiment_id']} -->"
    tracker = tmp_path / "current_experiments.md"
    tracker.write_text(
        "# Current Experiments\n\n"
        + marker
        + "\n### First\n\n"
        + "- **Testing:** First.\n- **Where:** local.\n"
        + "- **Status:** `preflighting`\n\n"
        + marker
        + "\n### Second\n\n"
        + "- **Testing:** Second.\n- **Where:** local.\n"
        + "- **Status:** `preflighting`\n",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentTrackerError, match="at most one"):
        upsert_tracker(tracker, entry)
