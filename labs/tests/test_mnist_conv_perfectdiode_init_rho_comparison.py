from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.mnist_conv.io import atomic_write_json
from experiments.mnist_conv.perfectdiode_init_rho_comparison import (
    INITIALIZER_ORDER,
    build_manifest,
    load_config,
    materialize,
    run_gate,
    run_surface,
    status,
    validate_root,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv12_initialization_rho_comparison_20260727_v1.json"
)


def _environment(tmp_path: Path) -> Path:
    path = tmp_path / "environment.json"
    atomic_write_json(
        path,
        {
            "schema_version": "test-environment/v1",
            "device": "cuda",
        },
        canonical=True,
    )
    return path


def _materialized(tmp_path: Path) -> Path:
    root = tmp_path / "result"
    materialize(
        CONFIG,
        root,
        source_commit="1" * 40,
        source_archive_sha256="2" * 64,
        environment_contract=_environment(tmp_path),
    )
    return root


def test_manifest_has_two_new_study_identities_and_exact_frozen_coverage():
    manifest = build_manifest(
        CONFIG,
        source_commit="1" * 40,
        source_archive_sha256="2" * 64,
        environment_contract_sha256="3" * 64,
    )

    assert manifest["parent_study_id"].startswith("lrstudy_")
    assert manifest["reuse_parent_numerical_outputs"] is False
    assert manifest["counts"] == {
        "initializers": 2,
        "gate_entries": 12,
        "surface_entries": 22,
        "rho_cells": 148,
        "maximum_promoted_candidates": 148,
    }
    assert [study["initializer_id"] for study in manifest["studies"]] == list(
        INITIALIZER_ORDER
    )
    assert len({study["study_id"] for study in manifest["studies"]}) == 2
    assert all(
        study["study_id"] != manifest["parent_study_id"]
        for study in manifest["studies"]
    )
    assert [
        entry["surface_index"] for entry in manifest["surface_entries"]
    ] == list(range(22))
    assert sum(
        len(entry["cells"]) for entry in manifest["surface_entries"]
    ) == 148


def test_manifest_identity_is_deterministic_and_binds_execution_inputs():
    arguments = {
        "source_commit": "1" * 40,
        "source_archive_sha256": "2" * 64,
        "environment_contract_sha256": "3" * 64,
    }
    first = build_manifest(CONFIG, **arguments)
    second = build_manifest(CONFIG, **arguments)
    assert first == second

    changed = build_manifest(
        CONFIG,
        source_commit="4" * 40,
        source_archive_sha256="2" * 64,
        environment_contract_sha256="3" * 64,
    )
    assert changed["manifest_id"] != first["manifest_id"]


def test_materialize_is_idempotent_and_rejects_environment_drift(tmp_path):
    root = _materialized(tmp_path)
    destination, manifest, config = validate_root(root)
    assert destination == root.resolve()
    assert manifest["config_sha256"]
    assert config["experiment_id"] == manifest["experiment_id"]

    materialize(
        CONFIG,
        root,
        source_commit="1" * 40,
        source_archive_sha256="2" * 64,
        environment_contract=tmp_path / "environment.json",
    )
    environment = json.loads(
        (tmp_path / "environment.json").read_text(encoding="utf-8")
    )
    environment["device"] = "cpu"
    atomic_write_json(
        tmp_path / "environment.json",
        environment,
        canonical=True,
    )
    with pytest.raises(ValueError, match="environment_contract.json"):
        materialize(
            CONFIG,
            root,
            source_commit="1" * 40,
            source_archive_sha256="2" * 64,
            environment_contract=tmp_path / "environment.json",
        )


def test_gate_and_surface_resume_only_hash_verified_completions(
    tmp_path,
    monkeypatch,
):
    root = _materialized(tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_execute(
        spec,
        row_root,
        stage,
        payload,
        *,
        data_root,
        device,
        download,
    ):
        del data_root, device, download
        entry_id = payload["entry_id"]
        entry_dir = (
            Path(row_root)
            / "stages"
            / stage
            / "entries"
            / entry_id
        )
        entry_dir.mkdir(parents=True, exist_ok=True)
        calls.append((stage, entry_id))
        if stage == "assets":
            result = {
                "initialization_checkpoint_sha256": "a" * 64,
                "initialization_tensor_sha256": "b" * 64,
                "official_test_read": False,
            }
        elif stage == "fixed_tk_gradient_security":
            result = {"passed": True, "official_test_read": False}
        elif stage == "optimizer_probe":
            result = {"probe_stable": True, "official_test_read": False}
        elif stage == "rho_canary_extension":
            result = {
                "cells": [
                    {
                        "cell_id": payload["cells"][0]["cell_id"],
                        "safety_clean": True,
                    }
                ],
                "official_test_read": False,
            }
        elif stage == "rho_core_candidates":
            result = {
                "status": "complete",
                "study_id": spec.study_id,
                "cell_id": payload["cell_id"],
                "safety_admissible": True,
                "completed_steps": 10314,
                "final_validation_loss": 0.1,
                "final_validation_accuracy": 0.91,
                "median_projection_efficiency": 0.8,
                "official_test_read": False,
            }
        else:
            raise AssertionError(stage)
        atomic_write_json(entry_dir / "result.json", result, canonical=True)
        return result

    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_init_rho_comparison."
        "execute_perfectdiode_stage_entry",
        fake_execute,
    )
    first_gate = run_gate(
        root,
        0,
        data_root=tmp_path / "mnist",
        device="cuda",
    )
    resumed_gate = run_gate(
        root,
        0,
        data_root=tmp_path / "mnist",
        device="cuda",
    )
    assert first_gate["security_passed"] is True
    assert resumed_gate["resume"] == "skipped_hash_verified_complete"

    first_surface = run_surface(
        root,
        0,
        data_root=tmp_path / "mnist",
        device="cuda",
    )
    calls_after_surface = list(calls)
    resumed_surface = run_surface(
        root,
        0,
        data_root=tmp_path / "mnist",
        device="cuda",
    )
    assert first_surface["status"] == "complete"
    assert len(first_surface["cells"]) == 9
    assert resumed_surface["resume"] == "skipped_hash_verified_complete"
    assert calls == calls_after_surface
    assert status(root)["surface_entries"] == {
        "complete": 1,
        "expected": 22,
    }

    candidate = (
        root
        / "studies"
        / first_surface["study_id"]
        / "rows"
        / first_surface["row_id"]
        / "stages"
        / "rho_core_candidates"
        / "entries"
        / first_surface["cells"][0]["cell_id"]
        / "result.json"
    )
    candidate.write_text("{}\n", encoding="utf-8")
    invalid = status(root)
    assert invalid["state"] == "invalid"
    assert invalid["errors"]


def test_config_loader_preserves_user_approved_sequential_24_gpu_cap():
    _source, config, _base = load_config(CONFIG)
    assert config["execution"]["slurm"]["array_concurrency"] == 24
    assert config["execution"]["official_test_read"] is False
