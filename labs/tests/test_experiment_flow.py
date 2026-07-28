from __future__ import annotations

import copy
from pathlib import Path

from experiments import experiment_executor as executor
from experiments.experiment_catalog import load_catalog
from experiments.experiment_flow import (
    apply_workspace_overrides,
    discover_parent_bundle,
    load_profiles,
    load_smoke_host_facts,
    prepare_request,
    publish_prepared_flow,
)
from experiments.experiment_request import parse_request_text


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUEST_PATH = (
    REPO_ROOT
    / "docs"
    / "experiment_requests"
    / "perfectdiode-conv1-best-rho-continuation-20260728.md"
)
CATALOG_PATH = REPO_ROOT / "experiments" / "experiment_catalog.json"


def _request(*, approved: bool) -> dict:
    request = parse_request_text(REQUEST_PATH.read_text())
    if approved:
        request["user_review"]["scientific_summary_approved"] = "yes"
        request["user_review"]["execution_proposal_authorized"] = "yes"
        request["user_review"]["corrections_or_constraints"] = "none"
    else:
        request["user_review"]["scientific_summary_approved"] = "pending"
        request["user_review"]["execution_proposal_authorized"] = "pending"
    return request


def _host_facts(target_id: str, concurrency: int) -> dict:
    return {
        "schema_version": executor.HOST_FACTS_SCHEMA,
        "target_id": target_id,
        "reachable": True,
        "python": {
            "available": True,
            "executable": f"/opt/{target_id}/python",
            "version": "compatible",
        },
        "imports": {"torch": {"available": True, "version": "compatible"}},
        "cuda": {
            "available": True,
            "device_count": 1,
            "devices": [
                {
                    "index": 0,
                    "name": f"{target_id}-gpu",
                    "free_memory_mb": 12000,
                }
            ],
        },
        "dataset": {"readable": True, "identity": "ordinary-mnist"},
        "output": {"writable": True, "path": f"/results/{target_id}"},
        "smoke": {
            "status": "passed",
            "completed_optimizer_steps": 1,
            "finite": True,
            "artifact_written": True,
        },
        "tk_reference": {
            "status": "reused",
            "evidence": "unchanged continuation operating point",
        },
        "max_safe_concurrency": concurrency,
        "observed_at": "2026-07-28T14:00:00+00:00",
    }


def test_pending_request_stops_before_attempt_planning() -> None:
    result = prepare_request(
        _request(approved=False),
        catalog=load_catalog(CATALOG_PATH),
        repo_root=REPO_ROOT,
    )

    assert result["status"] == "awaiting_user_review"
    assert result["launched_job_count"] == 0
    assert result["side_effects_performed"] is False
    assert result["validation_plan"] is None
    assert result["production_plan"] is None
    assert result["resolved_study"]["study"]["budget"]["job_count"] == 6


def test_approved_golden_request_plans_three_smokes_then_six_jobs() -> None:
    result = prepare_request(
        _request(approved=True),
        catalog=load_catalog(CATALOG_PATH),
        repo_root=REPO_ROOT,
    )

    assert result["status"] == "ready_for_functional_smokes"
    assert result["launched_job_count"] == 0
    assert result["validation_plan"]["planned_job_count"] == 3
    assert result["production_plan"]["planned_job_count"] == 6
    assert result["validation_plan"]["target_count"] == 3
    assert len(result["validation_dispatch_requests"]) == 3
    assert result["production_dispatch_requests"] is None
    assert {
        request["target_id"]
        for request in result["validation_dispatch_requests"]
    } == {"local", "trex", "akib"}
    assert all(
        request["run_class"] == "validation"
        and request["authorized"] is True
        and request["hard_deadline_seconds"] == 360
        and len(request["jobs"]) == 1
        for request in result["validation_dispatch_requests"]
    )
    assert [
        attempt["owned_job_ids"]
        for attempt in result["production_plan"]["attempts"]
    ] == [
        [
            "pdconfirm_conv1_baseline_sgd_seed0",
            "pdconfirm_conv1_baseline_adam_seed0",
        ],
        [
            "pdconfirm_conv1_ours_sgd_seed0",
            "pdconfirm_conv1_ours_adam_seed0",
        ],
        [
            "pdconfirm_conv1_legacy_sgd_seed0",
            "pdconfirm_conv1_legacy_adam_seed0",
        ],
    ]
    assert result["production_proposal"]["preflight"] == {
        "functional_smoke_required": True,
        "fresh_tk_gate_required": False,
    }
    workers = [
        job["worker_argv"]
        for attempt in result["validation_plan"]["attempts"]
        for job in attempt["jobs"]
    ]
    assert len(workers) == 3
    assert all(
        "experiments.run_mnist_conv_perfectdiode_job" in argv
        and "validation" in argv
        and "cuda:0" in argv
        for argv in workers
    )


def test_bundle_discovery_is_automatic_and_compatible() -> None:
    selected = [
        "pdconfirm_conv1_baseline_sgd_seed0",
        "pdconfirm_conv1_baseline_adam_seed0",
        "pdconfirm_conv1_ours_sgd_seed0",
        "pdconfirm_conv1_ours_adam_seed0",
        "pdconfirm_conv1_legacy_sgd_seed0",
        "pdconfirm_conv1_legacy_adam_seed0",
    ]
    path = discover_parent_bundle(
        repo_root=REPO_ROOT,
        parent_experiment_id=(
            "perfectdiode-conv12-best-observed-confirmation-20260727-v1"
        ),
        parent_config_path=(
            "configs/conv/"
            "perfectdiode_conv12_best_observed_confirmation_20260727_v1.json"
        ),
        selected_job_ids=selected,
    )
    assert path.name.startswith("pdconfirmbundle_")
    assert (REPO_ROOT / path / "manifest.json").is_file()


def test_profile_paths_are_resolved_per_target() -> None:
    result = prepare_request(
        _request(approved=True),
        catalog=load_catalog(CATALOG_PATH),
        repo_root=REPO_ROOT,
        target_profiles=load_profiles(),
    )
    workers = {
        attempt["target_id"]: attempt["jobs"][0]["worker_argv"]
        for attempt in result["validation_plan"]["attempts"]
    }
    assert "/home/filip/data" in workers["local"]
    assert "/home/filip/server_code/data" in workers["trex"]
    assert "/home/filiposana/data" in workers["akib"]
    assert any(
        value.startswith("/home/filiposana/server_code/")
        for value in workers["akib"]
    )


def test_workspace_overrides_support_isolated_remote_deployments() -> None:
    profiles = apply_workspace_overrides(
        load_profiles(),
        [
            "trex=/remote/trex/deployment-a",
            "akib=/remote/akib/deployment-a",
        ],
    )
    assert profiles["local"]["workspace_root"] == str(REPO_ROOT)
    assert profiles["trex"]["workspace_root"] == "/remote/trex/deployment-a"
    assert profiles["akib"]["workspace_root"] == "/remote/akib/deployment-a"


def test_workspace_overrides_reject_unknown_or_duplicate_targets() -> None:
    profiles = load_profiles()
    for overrides in (
        ["unknown=/remote/deployment"],
        ["trex=relative/path"],
        ["trex=/one", "trex=/two"],
    ):
        try:
            apply_workspace_overrides(profiles, overrides)
        except ValueError as exc:
            assert "Expected each workspace override once" in str(exc)
        else:
            raise AssertionError(f"override must fail: {overrides!r}")


def test_smoke_receipt_bindings_require_existing_absolute_target_paths(
    tmp_path,
) -> None:
    profiles = load_profiles()
    for binding in (
        "unknown=/tmp/smoke.json",
        "local=relative.json",
        f"local={tmp_path / 'missing.json'}",
    ):
        try:
            load_smoke_host_facts(profiles, [binding])
        except ValueError as exc:
            assert "Expected each smoke receipt once" in str(exc)
        else:
            raise AssertionError(f"binding must fail: {binding!r}")


def test_passing_target_smokes_generate_three_authorized_production_requests() -> None:
    facts = {
        "local": _host_facts("local", 2),
        "trex": _host_facts("trex", 2),
        "akib": _host_facts("akib", 1),
    }
    result = prepare_request(
        _request(approved=True),
        catalog=load_catalog(CATALOG_PATH),
        repo_root=REPO_ROOT,
        host_facts=facts,
    )

    assert result["status"] == "ready_for_production"
    assert result["functional_preflight"]["status"] == "passed"
    requests = result["production_dispatch_requests"]
    assert len(requests) == 3
    assert sum(len(request["jobs"]) for request in requests) == 6
    assert all(
        request["run_class"] == "production"
        and request["authorized"] is True
        and request["preflight_summary"]["status"] == "passed"
        for request in requests
    )
    akib = next(
        request for request in requests if request["target_id"] == "akib"
    )
    assert [len(batch) for batch in akib["launch_batches"]] == [1, 1]


def test_attempt_label_creates_fresh_operational_ids_only() -> None:
    first = prepare_request(
        _request(approved=True),
        catalog=load_catalog(CATALOG_PATH),
        repo_root=REPO_ROOT,
        attempt_label="repair-a",
    )
    second = prepare_request(
        _request(approved=True),
        catalog=load_catalog(CATALOG_PATH),
        repo_root=REPO_ROOT,
        attempt_label="repair-b",
    )
    assert first["resolved_study"] == second["resolved_study"]
    assert first["executor_study"] == second["executor_study"]
    assert (
        first["validation_proposal"]["proposal_id"]
        != second["validation_proposal"]["proposal_id"]
    )
    assert (
        first["validation_plan"]["attempts"][0]["attempt_id"]
        != second["validation_plan"]["attempts"][0]["attempt_id"]
    )


def test_publishing_is_authorized_and_idempotent(tmp_path) -> None:
    pending = prepare_request(
        _request(approved=False),
        catalog=load_catalog(CATALOG_PATH),
        repo_root=REPO_ROOT,
    )
    try:
        publish_prepared_flow(pending, tmp_path / "pending")
    except ValueError:
        pass
    else:
        raise AssertionError("pending request must not publish attempts")

    ready = prepare_request(
        _request(approved=True),
        catalog=load_catalog(CATALOG_PATH),
        repo_root=REPO_ROOT,
    )
    first = publish_prepared_flow(ready, tmp_path / "ready")
    second = publish_prepared_flow(copy.deepcopy(ready), tmp_path / "ready")
    assert first == second
    assert set(first) == {
        "prepared_flow.json",
        "resolved_study.json",
        "executor_study.json",
        "validation_proposal.json",
        "production_proposal.json",
        "validation_plan.json",
        "production_plan.json",
        "validation_dispatch_local.json",
        "validation_dispatch_trex.json",
        "validation_dispatch_akib.json",
    }
    assert Path(first["prepared_flow.json"]).is_file()
