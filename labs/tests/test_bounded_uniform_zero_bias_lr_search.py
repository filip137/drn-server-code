from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.conv3_operating_point_gate as conv3_gate
import experiments.run_conv12_bounded_rho as runner


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_PATH = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_bounded_uniform_zero_bias_lr_search_conv123_seed0_20260810_v1.json"
)


def test_study_declares_exact_eighteen_zero_bias_surfaces() -> None:
    _path, study = runner.load_study(STUDY_PATH)
    surfaces = runner.surface_specs(study)

    assert len(surfaces) == 18
    assert {row["architecture"] for row in surfaces} == {
        "conv1",
        "conv2",
        "conv3",
    }
    assert {row["scheme"] for row in surfaces} == {
        "baseline",
        "ours",
        "legacy",
    }
    assert {row["optimizer"] for row in surfaces} == {"SGD", "Adam"}
    assert {row["initializer"] for row in surfaces} == {"bounded_uniform"}
    assert study["bias_contract"] == {
        "initialization": "default_zero",
        "learning_rate": 0.0,
        "conductance_projection": False,
    }
    assert study["rho_search"]["bias_policy"] == "zero"
    assert study["rho_search"]["select_best_safe_below_accuracy"] is False


def test_surface_policies_keep_conv3_high_grid_and_hard_boundary_gates() -> None:
    _path, study = runner.load_study(STUDY_PATH)
    surfaces = runner.surface_specs(study)
    by_key = {
        (row["architecture"], row["scheme"], row["optimizer"]): row
        for row in surfaces
    }

    baseline = runner._surface_search(
        study, by_key[("conv3", "baseline", "SGD")]
    )
    legacy = runner._surface_search(study, by_key[("conv3", "legacy", "SGD")])
    conv2 = runner._surface_search(study, by_key[("conv2", "baseline", "SGD")])

    assert baseline["core_mode"] == "fixed_grid"
    assert baseline["fixed_core"] == {
        "rho_conv": [0.009, 0.027, 0.081],
        "rho_dense": [0.03, 0.09, 0.27],
    }
    assert baseline["safety"] == {
        **study["rho_search"]["safety"],
        "bound_occupancy": "reject_persistent_increase",
        "projection_efficiency": "reject_persistent_low_efficiency",
        "bound_occupancy_increase_maximum": 0.2,
        "projection_efficiency_minimum": 0.5,
        "boundary_persistence_steps": 16,
    }
    assert legacy["core_mode"] == "adaptive_safe_center"
    assert legacy["safety"] == baseline["safety"]
    assert conv2["core_mode"] == "adaptive_safe_center"
    assert "bound_occupancy_increase_maximum" not in conv2["safety"]


@pytest.mark.parametrize(
    ("architecture", "expected_rates"),
    (("conv1", [1.0, 1.0, 0.0]), ("conv2", [1.0, 1.0, 1.0, 0.0, 0.0])),
)
def test_source_config_starts_with_zero_bias_rates(
    tmp_path: Path,
    architecture: str,
    expected_rates: list[float],
) -> None:
    _path, study = runner.load_study(STUDY_PATH)
    config = runner.build_source_config(
        study,
        initializer="bounded_uniform",
        architecture=architecture,
        scheme="legacy",
        optimizer="Adam",
        init_checkpoint_path=tmp_path / "initial.pt",
        dataset_root=tmp_path / "mnist",
    )

    assert config["lr"] == expected_rates
    assert config["optimizer"]["learning_rate"] == expected_rates
    assert config["model_base"]["voltage_amp"] == 4.0
    assert config["model_base"]["current_amp"] == 0.25
    assert config["model_base"]["weight_init_mode"] == "bounded_uniform"
    assert config["model_base"]["weight_min"] == 1e-5
    assert config["model_base"]["weight_max"] == 1e-4


def test_conv3_gate_is_cached_across_optimizers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, study = runner.load_study(STUDY_PATH)
    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"shared-conv3-checkpoint")
    calls: list[tuple[Path, Path, bool]] = []

    def fake_gate(
        source_config_path,
        checkpoint_path,
        *,
        device,
        output_path,
        smoke,
    ):
        del device, output_path
        source = Path(source_config_path)
        checkpoint_value = Path(checkpoint_path)
        calls.append((source, checkpoint_value, smoke))
        return {
            "schema_version": "test-conv3-gate/v1",
            "status": "complete",
            "security_passed": True,
            "scientifically_complete": not smoke,
            "smoke": smoke,
            "source_config_sha256": runner._sha256_file(source),
            "checkpoint_sha256": runner._sha256_file(checkpoint_value),
            "official_test_read": False,
        }

    monkeypatch.setattr(conv3_gate, "run_conv3_operating_point_gate", fake_gate)
    surface = {
        "initializer": "bounded_uniform",
        "architecture": "conv3",
        "scheme": "ours",
        "optimizer": "SGD",
    }
    first = runner.run_conv3_tk_operating_point_gate(
        study,
        tmp_path / "results",
        surface,
        checkpoint,
        device="cpu",
        smoke=False,
        dataset_root=tmp_path / "mnist",
    )
    surface["optimizer"] = "Adam"
    second = runner.run_conv3_tk_operating_point_gate(
        study,
        tmp_path / "results",
        surface,
        checkpoint,
        device="cpu",
        smoke=False,
        dataset_root=tmp_path / "mnist",
    )

    assert first == second
    assert len(calls) == 1
    source_config = json.loads(calls[0][0].read_text(encoding="utf-8"))
    assert source_config["optimizer"]["name"] == "SGD"
    assert all(rate == 0.0 for rate in source_config["lr"])
