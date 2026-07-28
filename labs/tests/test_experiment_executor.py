from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import experiment_executor as executor


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATHS = {
    target_id: REPO_ROOT / "configs" / "dispatch" / f"functional_{target_id}.json"
    for target_id in ("local", "trex", "akib")
}


def _profiles() -> dict[str, dict]:
    return {
        target_id: executor.load_target_profile(path)
        for target_id, path in PROFILE_PATHS.items()
    }


def _study() -> dict:
    jobs = []
    for scheme in ("baseline", "ours", "legacy"):
        for optimizer in ("sgd", "adam"):
            jobs.append(
                {
                    "job_id": f"conv1-{scheme}-{optimizer}-seed0",
                    "scheme": scheme,
                    "optimizer": optimizer,
                    "seed": 0,
                }
            )
    return {
        "schema_version": executor.RESOLVED_STUDY_SCHEMA,
        "study_id": "perfectdiode-conv1-six-run-v1",
        "jobs": jobs,
    }


def _proposal(run_class: str = "production") -> dict:
    return {
        "schema_version": executor.EXECUTION_PROPOSAL_SCHEMA,
        "proposal_id": "perfectdiode-conv1-six-run-local",
        "run_class": run_class,
        "launch_after_checks": True,
        "runner": {
            "adapter_id": "mnist-conv-one-batch-capable-v1",
            "argv_template": [
                "{python}",
                "-m",
                "experiments.fake_adapter",
                "--job-id",
                "{job_id}",
                "--data-root",
                "{data_root}",
                "--device",
                "{device}",
                "--scheme",
                "{job.scheme}",
                "--run-class",
                "{run_class}",
                "--output-dir",
                "{output_dir}",
            ],
        },
        "preflight": {
            "functional_smoke_required": True,
            "fresh_tk_gate_required": False,
        },
        "targets": [
            {
                "target_id": "local",
                "job_ids": [
                    "conv1-baseline-sgd-seed0",
                    "conv1-baseline-adam-seed0",
                ],
                "concurrency": 2,
                "validation_job_id": "conv1-baseline-sgd-seed0",
            },
            {
                "target_id": "trex",
                "job_ids": [
                    "conv1-ours-sgd-seed0",
                    "conv1-ours-adam-seed0",
                ],
                "concurrency": "auto",
                "validation_job_id": "conv1-ours-adam-seed0",
            },
            {
                "target_id": "akib",
                "job_ids": [
                    "conv1-legacy-sgd-seed0",
                    "conv1-legacy-adam-seed0",
                ],
                "concurrency": 2,
                "validation_job_id": "conv1-legacy-sgd-seed0",
            },
        ],
    }


def _facts(
    target_id: str,
    *,
    python_version: str,
    torch_version: str,
    smoke_status: str = "passed",
    max_safe_concurrency: int | None = 2,
) -> dict:
    completed_steps = 1 if smoke_status == "passed" else 0
    return {
        "schema_version": executor.HOST_FACTS_SCHEMA,
        "target_id": target_id,
        "reachable": True,
        "python": {
            "available": True,
            "executable": f"/opt/{target_id}/bin/python",
            "version": python_version,
        },
        "imports": {
            "torch": {
                "available": True,
                "version": torch_version,
            }
        },
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
        "dataset": {
            "readable": True,
            "identity": "ordinary-mnist",
        },
        "output": {
            "writable": True,
            "path": f"/results/{target_id}",
        },
        "smoke": {
            "status": smoke_status,
            "completed_optimizer_steps": completed_steps,
            "finite": smoke_status == "passed",
            "artifact_written": smoke_status == "passed",
        },
        "tk_reference": {
            "status": "reused",
            "evidence": "unchanged continuation operating point",
        },
        "max_safe_concurrency": max_safe_concurrency,
        "observed_at": "2026-07-28T14:00:00+00:00",
    }


def _all_facts(*, smoke_status: str = "passed") -> dict[str, dict]:
    return {
        "local": _facts(
            "local",
            python_version="3.10.14",
            torch_version="2.2.2",
            smoke_status=smoke_status,
            max_safe_concurrency=2,
        ),
        "trex": _facts(
            "trex",
            python_version="3.12.3",
            torch_version="2.6.0+cu128",
            smoke_status=smoke_status,
            max_safe_concurrency=2,
        ),
        "akib": _facts(
            "akib",
            python_version="3.13.1",
            torch_version="2.5.1+cu124",
            smoke_status=smoke_status,
            max_safe_concurrency=1,
        ),
    }


def test_repository_functional_profiles_are_capability_based() -> None:
    profiles = _profiles()

    assert set(profiles) == {"local", "trex", "akib"}
    assert profiles["local"]["transport"]["kind"] == "local"
    assert profiles["trex"]["transport"]["ssh_target"] == "filip@trex"
    assert profiles["akib"]["transport"]["ssh_target"] == "akib"
    for profile in profiles.values():
        assert profile["required_imports"] == ["torch"]
        assert profile["requires_cuda"] is True
        assert profile["device"] == "cuda:0"
        assert profile["data_root"].startswith("/home/")
        assert "python_version" not in profile
        assert "torch_version" not in profile
        assert "cuda_version" not in profile
        assert "sha256" not in json.dumps(profile)


def test_production_plan_has_exact_six_job_ownership_and_deterministic_argv() -> None:
    study = _study()
    proposal = _proposal()
    facts = _all_facts()

    first = executor.build_execution_plan(
        study,
        proposal,
        target_profiles=_profiles(),
        host_facts=facts,
    )
    second = executor.plan_attempts(
        study,
        proposal,
        target_profiles=_profiles(),
        host_facts=facts,
    )

    assert first == second
    assert first["side_effects_performed"] is False
    assert first["resolved_job_count"] == 6
    assert first["planned_job_count"] == 6
    assert [attempt["target_id"] for attempt in first["attempts"]] == [
        "local",
        "trex",
        "akib",
    ]
    assert [
        job_id
        for attempt in first["attempts"]
        for job_id in attempt["owned_job_ids"]
    ] == [job["job_id"] for job in study["jobs"]]

    local, trex, akib = first["attempts"]
    assert local["effective_concurrency"] == 2
    assert trex["effective_concurrency"] == 2
    assert akib["effective_concurrency"] == 1
    assert akib["warnings"] == [
        "reduced concurrency from 2 to observed safe limit 1"
    ]
    assert akib["launch_batches"] == [
        ["conv1-legacy-sgd-seed0"],
        ["conv1-legacy-adam-seed0"],
    ]

    local_command = local["jobs"][0]
    assert local_command["worker_argv"][0] == "/opt/local/bin/python"
    assert "--scheme" in local_command["worker_argv"]
    assert "baseline" in local_command["worker_argv"]
    assert "/home/filip/data" in local_command["worker_argv"]
    assert "cuda:0" in local_command["worker_argv"]
    assert local_command["dispatch_argv"][0] == "tmux"

    trex_command = trex["jobs"][0]["dispatch_argv"]
    assert trex_command[:4] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "filip@trex",
    ]
    assert "experiments.fake_adapter" in trex_command[-1]


def test_validation_plan_runs_one_representative_one_batch_job_per_target() -> None:
    proposal = _proposal("validation")
    facts = _all_facts(smoke_status="not_run")
    plan = executor.build_execution_plan(
        _study(),
        proposal,
        target_profiles=_profiles(),
        host_facts=facts,
    )

    assert plan["run_class"] == "validation"
    assert plan["resolved_job_count"] == 6
    assert plan["planned_job_count"] == 3
    assert [
        attempt["planned_job_ids"] for attempt in plan["attempts"]
    ] == [
        ["conv1-baseline-sgd-seed0"],
        ["conv1-ours-adam-seed0"],
        ["conv1-legacy-sgd-seed0"],
    ]
    assert all(
        attempt["effective_concurrency"] == 1
        for attempt in plan["attempts"]
    )
    assert all(
        "validation" in attempt["jobs"][0]["worker_argv"]
        for attempt in plan["attempts"]
    )

    receipt = executor.functional_preflight(
        plan,
        host_facts=facts,
        target_profiles=_profiles(),
    )
    assert receipt["status"] == "passed"
    smoke_checks = [
        next(
            check
            for check in target["checks"]
            if check["role"] == "functional_smoke"
        )
        for target in receipt["targets"]
    ]
    assert all(check["status"] == "planned" for check in smoke_checks)


def test_production_preflight_accepts_different_runtime_versions() -> None:
    facts = _all_facts()
    plan = executor.build_execution_plan(
        _study(),
        _proposal(),
        target_profiles=_profiles(),
        host_facts=facts,
    )
    receipt = executor.functional_preflight(
        plan,
        host_facts=facts,
        target_profiles=_profiles(),
    )

    assert receipt["status"] == "passed"
    assert receipt["versions_are_provenance_only"] is True
    assert [
        target["provenance"]["python"]["version"]
        for target in receipt["targets"]
    ] == ["3.10.14", "3.12.3", "3.13.1"]
    assert [
        target["provenance"]["imports"]["torch"]["version"]
        for target in receipt["targets"]
    ] == ["2.2.2", "2.6.0+cu128", "2.5.1+cu124"]
    assert all(
        next(
            check
            for check in target["checks"]
            if check["role"] == "tk_reference"
        )["status"]
        == "passed"
        for target in receipt["targets"]
    )


def test_capability_failure_is_a_compact_failed_receipt_not_an_exception() -> None:
    facts = _all_facts()
    facts["akib"]["dataset"]["readable"] = False
    plan = executor.build_execution_plan(
        _study(),
        _proposal(),
        target_profiles=_profiles(),
        host_facts=facts,
    )

    receipt = executor.functional_preflight(
        plan,
        host_facts=facts,
        target_profiles=_profiles(),
    )

    assert receipt["status"] == "failed"
    assert receipt["blocking_reasons"] == [
        {"target_id": "akib", "reasons": ["dataset"]}
    ]
    assert receipt["side_effects_performed"] is False


@pytest.mark.parametrize("mutation", ["missing", "unknown", "duplicate"])
def test_job_ownership_must_cover_the_study_exactly_once(mutation: str) -> None:
    proposal = _proposal()
    if mutation == "missing":
        proposal["targets"][2]["job_ids"].pop()
    elif mutation == "unknown":
        proposal["targets"][2]["job_ids"][-1] = "conv1-unknown-adam-seed0"
    else:
        proposal["targets"][2]["job_ids"][-1] = (
            "conv1-baseline-adam-seed0"
        )

    with pytest.raises(
        executor.ExecutorError,
        match=(
            r"^Expected every resolved study job owned by exactly one target "
            r"at execution_proposal.targets; provided value:"
        ),
    ) as caught:
        executor.build_execution_plan(
            _study(),
            proposal,
            target_profiles=_profiles(),
        )

    assert caught.value.code == "job_ownership_mismatch"


def test_missing_target_profile_fails_before_any_dispatch() -> None:
    profiles = _profiles()
    del profiles["akib"]

    with pytest.raises(executor.ExecutorError) as caught:
        executor.build_execution_plan(
            _study(),
            _proposal(),
            target_profiles=profiles,
        )

    assert caught.value.code == "target_profile_missing"
    assert caught.value.as_dict()["launched_job_count"] == 0


def test_fresh_tk_gate_is_only_required_when_proposal_requests_it() -> None:
    proposal = _proposal()
    proposal["preflight"]["fresh_tk_gate_required"] = True
    facts = _all_facts()
    facts["trex"]["tk_reference"] = {
        "status": "reused",
        "evidence": "old gate",
    }
    plan = executor.build_execution_plan(
        _study(),
        proposal,
        target_profiles=_profiles(),
        host_facts=facts,
    )

    receipt = executor.functional_preflight(
        plan,
        host_facts=facts,
        target_profiles=_profiles(),
    )
    assert receipt["status"] == "failed"
    assert {"target_id": "trex", "reasons": ["tk_reference"]} in receipt[
        "blocking_reasons"
    ]

    for target_facts in facts.values():
        target_facts["tk_reference"] = {
            "status": "passed",
            "evidence": "fresh gate receipt",
        }
    passing = executor.functional_preflight(
        plan,
        host_facts=facts,
        target_profiles=_profiles(),
    )
    assert passing["status"] == "passed"


def test_receipt_schemas_are_generated_and_result_helpers_check_coverage() -> None:
    schemas = executor.receipt_schema_definitions()
    assert schemas["preflight"]["$id"] == executor.PREFLIGHT_RESULT_SCHEMA
    assert schemas["job"]["$id"] == executor.JOB_RESULT_SCHEMA
    assert schemas["attempt"]["$id"] == executor.ATTEMPT_RESULT_SCHEMA

    plan = executor.build_execution_plan(
        _study(),
        _proposal("validation"),
        target_profiles=_profiles(),
    )
    attempt = plan["attempts"][0]
    job_id = attempt["planned_job_ids"][0]
    job_result = executor.make_job_result(
        plan=plan,
        attempt_id=attempt["attempt_id"],
        target_id=attempt["target_id"],
        job_id=job_id,
        status="complete",
        exit_code=0,
        artifacts=["checkpoint.pt", "result.json"],
        provenance={"python": "3.11", "torch": "2.4"},
    )
    result = executor.make_attempt_result(
        plan=plan,
        attempt_id=attempt["attempt_id"],
        target_id=attempt["target_id"],
        job_results=[job_result],
    )
    assert result["status"] == "complete"
    assert result["job_results"] == [job_result]

    with pytest.raises(executor.ExecutorError) as caught:
        executor.make_attempt_result(
            plan=plan,
            attempt_id=attempt["attempt_id"],
            target_id=attempt["target_id"],
            job_results=[],
        )
    assert caught.value.code == "job_result_coverage_mismatch"


def test_cli_plan_is_read_only_and_doctor_uses_supplied_facts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    study_path = tmp_path / "study.json"
    proposal_path = tmp_path / "proposal.json"
    facts_path = tmp_path / "facts.json"
    study_path.write_text(json.dumps(_study()), encoding="utf-8")
    proposal_path.write_text(json.dumps(_proposal()), encoding="utf-8")
    facts_path.write_text(
        json.dumps({"targets": list(_all_facts().values())}),
        encoding="utf-8",
    )
    profiles = [
        item
        for target_id in ("local", "trex", "akib")
        for item in ("--profile", str(PROFILE_PATHS[target_id]))
    ]

    assert (
        executor.main(
            [
                "plan",
                "--study",
                str(study_path),
                "--proposal",
                str(proposal_path),
                *profiles,
            ]
        )
        == 0
    )
    planned = json.loads(capsys.readouterr().out)
    assert planned["status"] == "planned"
    assert planned["side_effects_performed"] is False

    assert (
        executor.main(
            [
                "doctor",
                "--study",
                str(study_path),
                "--proposal",
                str(proposal_path),
                *profiles,
                "--facts",
                str(facts_path),
            ]
        )
        == 0
    )
    doctor = json.loads(capsys.readouterr().out)
    assert doctor["status"] == "passed"
    assert doctor["side_effects_performed"] is False


def test_adapter_template_cannot_omit_job_or_output_binding() -> None:
    proposal = _proposal()
    proposal["runner"]["argv_template"] = ["{python}", "worker.py"]

    with pytest.raises(executor.ExecutorError) as caught:
        executor.validate_execution_proposal(proposal)

    assert caught.value.code == "missing_argv_placeholder"


def test_job_fields_can_be_bound_without_hard_coded_manifest_indices() -> None:
    study = _study()
    for index, job in enumerate(study["jobs"]):
        job["scientific"] = {"learning_rate": 0.125 + index}
    proposal = _proposal("validation")
    proposal["runner"]["argv_template"].extend(
        ["--learning-rate", "{job.scientific.learning_rate}"]
    )

    plan = executor.build_execution_plan(
        study,
        proposal,
        target_profiles=_profiles(),
    )

    assert plan["attempts"][0]["jobs"][0]["worker_argv"][-2:] == [
        "--learning-rate",
        "0.125",
    ]
