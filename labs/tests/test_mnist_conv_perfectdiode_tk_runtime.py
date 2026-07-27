from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from experiments.mnist_conv.identity import sha256_file, sha256_json
from experiments.mnist_conv.io import atomic_write_json, read_json
from experiments.mnist_conv.perfectdiode_tk_runtime import (
    compare_k_gradients,
    perfect_diode_clamped_occupancy,
    perfect_diode_projected_kkt_residual,
)
from experiments.mnist_conv.perfectdiode_tk_spec import (
    CONV3_CONV_WEIGHTS,
    PerfectDiodeTKStudySpec,
)
from experiments.run_mnist_conv_perfectdiode_tk import (
    PerfectDiodeTKOrchestrationError,
    finalize_study,
    plan_study,
    study_status,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv3_tk_ordinary_mnist_v1.json"
)


def test_projected_kkt_residual_has_correct_excitation_and_inhibition_signs() -> None:
    state = torch.tensor(
        [[0.0, 2.0, 0.0, -2.0]], dtype=torch.float64
    )
    gradient = torch.tensor(
        [[-3.0, -4.0, 5.0, 6.0]], dtype=torch.float64
    )
    residual = perfect_diode_projected_kkt_residual(
        state, gradient, epsilon=1.0e-8
    )
    # Excitation at zero violates only a negative gradient; inhibition at zero
    # violates only a positive gradient. Free units use the raw magnitude.
    assert torch.equal(
        residual, torch.tensor([[3.0, 4.0, 5.0, 6.0]], dtype=torch.float64)
    )
    occupancy = perfect_diode_clamped_occupancy(state)
    assert occupancy["fraction"] == pytest.approx(0.5)
    assert occupancy["excitation_fraction"] == pytest.approx(0.5)
    assert occupancy["inhibition_fraction"] == pytest.approx(0.5)


def _gradient_collection(values: list[float]) -> dict:
    return {
        "gradients": {
            name: tuple(
                torch.tensor([value], dtype=torch.float64) for value in values
            )
            for name in CONV3_CONV_WEIGHTS
        },
        "initial_weight_rms": {name: 1.0 for name in CONV3_CONV_WEIGHTS},
        "free_equilibrium_sha256_by_batch": ["a" * 64] * 8,
    }


def test_k_comparison_uses_mean_per_batch_not_concatenated_norm() -> None:
    candidate = _gradient_collection([1.0, 3.0] * 4)
    reference = _gradient_collection([2.0] * 8)
    result = compare_k_gradients(
        candidate, reference, candidate_k=4, reference_k=64
    )
    row = result["parameter_diagnostics"][0]
    assert row["gradient_l2"] == pytest.approx(2.0)
    assert row["reference_gradient_l2"] == pytest.approx(2.0)
    assert row["relative_gradient_l2_norm_delta"] == pytest.approx(0.0)
    assert row["gradient_vector_relative_error"] == pytest.approx(0.5)
    assert row["gradient_vector_cosine"] == pytest.approx(1.0)
    assert row["batch_count"] == 8
    assert len(row["batches"]) == 8


def _fake_assets(spec: PerfectDiodeTKStudySpec, root: Path) -> Path:
    asset_dir = root / spec.study_id
    asset_dir.mkdir(parents=True)
    checkpoint = asset_dir / "initialization.pt"
    checkpoint.write_bytes(b"shared-conv3-initialization\n")
    train_indices = list(range(55_000))
    validation_indices = list(range(55_000, 60_000))
    t_indices = train_indices[:1024]
    k_indices = train_indices[:256]
    assets = {
        "schema_version": "mnist-conv-perfectdiode-tk-assets/v1",
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "architecture": "conv3",
        "model_seed": 0,
        "official_test_read": False,
        "initialization": {
            "path": checkpoint.name,
            "sha256": sha256_file(checkpoint),
            "parameter_tensor_sha256": "1" * 64,
            "parameter_names": [
                "ConvWeight_0",
                "Bias_0",
                "ConvWeight_1",
                "Bias_1",
                "ConvWeight_2",
                "Bias_2",
                "DenseWeight_0",
            ],
            "shared_across_schemes": True,
        },
        "split": {
            "source": "mnist_train",
            "train_size": 55_000,
            "validation_size": 5_000,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "train_indices_sha256": "2" * 64,
            "validation_indices_sha256": "3" * 64,
        },
        "cohorts": {
            "base_batch_size": 16,
            "t": {
                "examples": 1024,
                "batch_size": 64,
                "source_indices": t_indices,
                "source_indices_sha256": sha256_json(t_indices),
            },
            "k": {
                "examples": 256,
                "batch_size": 32,
                "source_indices": k_indices,
                "source_indices_sha256": sha256_json(k_indices),
            },
        },
    }
    assets_path = asset_dir / "assets.json"
    atomic_write_json(assets_path, assets, canonical=True)
    completion = {
        "schema_version": "mnist-conv-perfectdiode-tk-assets-completion/v1",
        "state": "complete",
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "official_test_read": False,
        "outputs": [
            {
                "path": checkpoint.name,
                "sha256": sha256_file(checkpoint),
                "bytes": checkpoint.stat().st_size,
            },
            {
                "path": assets_path.name,
                "sha256": sha256_file(assets_path),
                "bytes": assets_path.stat().st_size,
            },
        ],
    }
    atomic_write_json(asset_dir / "completion.json", completion, canonical=True)
    return asset_dir


def _write_completed_entries(manifest_path: Path) -> None:
    manifest = read_json(manifest_path)
    root = manifest_path.parent
    manifest_sha = sha256_file(manifest_path)
    for entry in manifest["entries"]:
        entry_dir = root / entry["output_dir"]
        entry_dir.mkdir(parents=True)
        environment = {
            "execution_backend": "tmux",
            "execution_host": entry["execution"]["lane"],
        }
        atomic_write_json(entry_dir / "environment.json", environment, canonical=True)
        environment_sha = sha256_file(entry_dir / "environment.json")
        atomic_write_json(entry_dir / "t_measurements.json", {"ok": True}, canonical=True)
        atomic_write_json(entry_dir / "k_measurements.json", {"ok": True}, canonical=True)
        audit = {
            "study_id": manifest["study_id"],
            "entry_id": entry["entry_id"],
            "row_id": entry["row_id"],
            "execution_host": entry["execution"]["lane"],
            "execution_environment_sha256": environment_sha,
            "official_test_read": False,
            "fresh_replay_after_selection": True,
            "selected_t": 16,
            "selected_k": 8,
            "passed": True,
        }
        audit_path = entry_dir / "operating_point_audit.json"
        atomic_write_json(audit_path, audit, canonical=True)
        scheme_contract = {
            "baseline": ("mnist_bp_amp_v1_c1", 1.0, 1.0),
            "ours": ("mnist_bp_amp_v4_c1", 4.0, 1.0),
            "legacy": ("mnist_bp_amp_v4_c0p25", 4.0, 0.25),
        }
        run_name, voltage, current = scheme_contract[entry["scheme"]]
        result = {
            "study_id": manifest["study_id"],
            "config_sha256": manifest["config_sha256"],
            "manifest_sha256": manifest_sha,
            "entry_id": entry["entry_id"],
            "row_id": entry["row_id"],
            "architecture": "conv3",
            "scheme": entry["scheme"],
            "execution_backend": "tmux",
            "execution_host": entry["execution"]["lane"],
            "execution_environment_sha256": environment_sha,
            "run_name": run_name,
            "voltage_amp": voltage,
            "current_amp": current,
            "input_gain": 360.0,
            "official_test_read": False,
            "status": "selected",
            "diagnostic_selection_status": "selected",
            "selected_t": 16,
            "selected_k": 8,
            "t_reference": 64,
            "k_reference": 64,
            "t_extension_used": False,
            "k128_sentinel_used": False,
            "t_cohort_indices_sha256": "4" * 64,
            "k_cohort_indices_sha256": "5" * 64,
            "operating_point_audit_path": audit_path.name,
            "operating_point_audit_sha256": sha256_file(audit_path),
            "operating_point_audit_passed": True,
        }
        atomic_write_json(entry_dir / "result.json", result, canonical=True)
        records = []
        for relative in entry["outputs"]:
            path = root / relative
            records.append(
                {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        completion = {
            "schema_version": "mnist-conv-perfectdiode-tk-entry-completion/v1",
            "state": "complete",
            "study_id": manifest["study_id"],
            "entry_id": entry["entry_id"],
            "manifest_sha256": manifest_sha,
            "official_test_read": False,
            "outputs": records,
        }
        atomic_write_json(
            root / entry["completion_path"], completion, canonical=True
        )


def test_plan_routes_whole_rows_and_finalize_binds_audits_and_environments(
    tmp_path: Path,
) -> None:
    spec = PerfectDiodeTKStudySpec.from_path(CONFIG)
    asset_dir = _fake_assets(spec, tmp_path / "assets")
    commit = "a" * 40
    plan = plan_study(
        spec,
        results_root=tmp_path / "results",
        asset_dir=asset_dir,
        source_commit=commit,
        source_archive_sha256="b" * 64,
        provenance={
            "git_revision": commit,
            "dirty_source_digest": None,
            "effective_code_fingerprint": "c" * 64,
        },
    )
    assert [entry["execution"]["lane"] for entry in plan["entries"]] == [
        "main",
        "akibscomputer",
        "main",
    ]
    assert all(
        entry["execution"]["whole_row_on_one_lane"] for entry in plan["entries"]
    )
    manifest_path = Path(plan["manifest"])
    assert study_status(manifest_path)["completed_entries"] == 0
    _write_completed_entries(manifest_path)
    finalized = finalize_study(manifest_path)
    selection = read_json(finalized["selection"])
    assert selection["status"] == "selected"
    assert len(selection["rows"]) == 3
    for row in selection["rows"]:
        assert row["operating_point_audit_passed"] is True
        assert row["operating_point_audit_path"].startswith("entries/")
        assert row["execution_environment_path"].startswith("entries/")
        assert len(row["execution_environment_sha256"]) == 64


def test_finalize_rejects_tampered_completed_output(tmp_path: Path) -> None:
    spec = PerfectDiodeTKStudySpec.from_path(CONFIG)
    asset_dir = _fake_assets(spec, tmp_path / "assets")
    commit = "a" * 40
    plan = plan_study(
        spec,
        results_root=tmp_path / "results",
        asset_dir=asset_dir,
        source_commit=commit,
        source_archive_sha256="b" * 64,
        provenance={
            "git_revision": commit,
            "dirty_source_digest": None,
            "effective_code_fingerprint": "c" * 64,
        },
    )
    manifest_path = Path(plan["manifest"])
    _write_completed_entries(manifest_path)
    manifest = read_json(manifest_path)
    tampered = manifest_path.parent / manifest["entries"][0]["outputs"][-1]
    tampered.write_text("{}\n")
    with pytest.raises(PerfectDiodeTKOrchestrationError):
        finalize_study(manifest_path)
