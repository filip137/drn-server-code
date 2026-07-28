from __future__ import annotations

import json
from pathlib import Path
import subprocess

from experiments import jeanzay_validation as fastpath
from experiments.mnist_conv.io import read_json


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "experiments" / "run_jeanzay_validation.slurm"
PROFILE = REPO_ROOT / "configs" / "dispatch" / "functional_jeanzay.json"


def test_stable_runner_is_syntax_valid_and_remote_preflight_is_import_only() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    text = RUNNER.read_text(encoding="utf-8")

    assert len(fastpath.FOCUSED_TESTS) == 3
    assert "#SBATCH" not in text
    assert text.count('"${python_executable}" -c') == 1
    assert "import custom_minimizer" in text
    assert "from labs.custom_classes import FlexibleDeepResistiveEnergy" in text
    assert "experiments.jeanzay_validation_worker" in text
    assert "--run-class production" not in text
    assert "sbatch" not in text


def test_cached_profile_builds_one_exact_singleton_sbatch_command() -> None:
    profile = fastpath.load_profile(PROFILE)
    layout = {
        "remote_stage": "/remote/stage/experiment/attempt",
        "remote_result": "/remote/results/experiment/attempt",
        "remote_runner": (
            "/remote/stage/experiment/attempt/source/experiments/"
            "run_jeanzay_validation.slurm"
        ),
    }

    command = fastpath.sbatch_command(
        profile,
        layout,
        attempt_id="attempt-a",
        test_only=False,
    )
    test_only = fastpath.sbatch_command(
        profile,
        layout,
        attempt_id="attempt-a",
        test_only=True,
    )

    assert command[:2] == ["sbatch", "--parsable"]
    assert test_only[:2] == ["sbatch", "--test-only"]
    assert "--account=fmu@v100" in command
    assert "--partition=gpu_p13" in command
    assert "--qos=qos_gpu-dev" in command
    assert "--constraint=v100-16g" in command
    assert "--array=0-0" in command
    assert "--gres=gpu:1" in command
    assert command.count(layout["remote_runner"]) == 1
    assert "production" not in " ".join(command)


def test_approved_request_prepares_one_package_without_its_own_catalog_entry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    experiment_id = (
        "perfectdiode-conv1-jeanzay-functional-validation-20260728-v1"
    )
    entry_id = "pdconfirm_conv1_ours_sgd_seed0"
    catalog = {
        "schema_version": "experiment-catalog/v1",
        "entries": [{"experiment_id": "immutable-parent"}],
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    request_path = tmp_path / "approved-request.md"
    request_path.write_text("approved request\n", encoding="utf-8")
    bundle = tmp_path / "parent-bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"entries": [{"entry_id": entry_id}]}),
        encoding="utf-8",
    )

    request = {
        "user_answers": {
            "short_name": "Conv1 Jean Zay functional validation",
        }
    }
    study = {
        "study": {
            "parent_experiment_id": "immutable-parent",
            "purpose": "Validate one fixed continuation on Jean Zay.",
            "jobs": [
                {
                    "job_id": entry_id,
                    "official_test_read": False,
                }
            ],
            "tk_gate": {"fresh_required": False},
        },
        "bindings_and_provenance": {
            "parent_config_path": "configs/parent.json",
        },
    }
    monkeypatch.setattr(fastpath, "load_request", lambda _path: request)
    monkeypatch.setattr(
        fastpath,
        "validate_request",
        lambda _request, require_review: {"status": "valid"},
    )
    monkeypatch.setattr(fastpath, "load_catalog", lambda _path: catalog)

    def build_study(
        _request: object,
        observed_catalog: dict[str, object],
        *,
        repo_root: Path,
    ) -> dict[str, object]:
        assert all(
            entry["experiment_id"] != experiment_id
            for entry in observed_catalog["entries"]
        )
        assert repo_root == REPO_ROOT
        return study

    monkeypatch.setattr(fastpath, "build_resolved_study", build_study)
    monkeypatch.setattr(
        fastpath,
        "validate_resolved_study",
        lambda _study: {"status": "valid"},
    )

    local_attempt = tmp_path / "attempt"
    layout = {
        "remote_stage": "/remote/stage/experiment/attempt",
        "remote_result": "/remote/results/experiment/attempt",
        "remote_runner": (
            "/remote/stage/experiment/attempt/source/experiments/"
            "run_jeanzay_validation.slurm"
        ),
        "local_attempt": str(local_attempt),
        "local_package": str(local_attempt / "package"),
        "local_archive": str(local_attempt / "package.tar.gz"),
        "local_receipts": str(local_attempt / "receipts"),
        "local_collection": str(tmp_path / "incoming"),
        "transfer_root": str(tmp_path / "transfer"),
    }
    monkeypatch.setattr(fastpath, "_layout", lambda **kwargs: layout)

    def copy_runtime_source(_root: Path, destination: Path) -> None:
        runner = destination / fastpath.STABLE_RUNNER_RELATIVE
        runner.parent.mkdir(parents=True)
        runner.write_text("#!/bin/bash\n", encoding="utf-8")

    monkeypatch.setattr(
        fastpath,
        "_copy_runtime_source",
        copy_runtime_source,
    )

    prepared = fastpath.prepare_package(
        request_path=request_path,
        experiment_id=experiment_id,
        attempt_id="test-attempt",
        profile_path=PROFILE,
        catalog_path=catalog_path,
        repo_root=REPO_ROOT,
        bundle_override=bundle,
    )

    spec = read_json(local_attempt / "package" / "launch-spec.json")
    assert spec["entry_id"] == entry_id
    assert spec["run_class"] == "validation"
    assert spec["maximum_jobs"] == 1
    assert spec["official_test_read"] is False
    assert (local_attempt / "package.tar.gz").is_file()
    assert prepared["package_receipt"]["catalog_role"] == "parent_discovery_only"
    assert prepared["package_receipt"]["tracker_role"] == (
        "best_effort_observation_only"
    )
    assert prepared["package_receipt"]["broad_tests_run"] == []
