from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.mnist_conv.lr_study_spec import (
    FROZEN_CONV3_SCHEME_TWO_RHO_ROWS,
    LR_CONV3_SCHEME_TWO_RHO_STUDY_SCHEMA_VERSION,
    LRStudySpec,
)
from experiments.mnist_conv.specs import SpecValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs/conv"
V7_CONFIG = (
    CONFIG_ROOT
    / "hardsigmoid_lr_conv3_scheme_two_rho_constant_sgd_bs16_v7.json"
)
V7_STUDY_ID = (
    "lrstudy_6e56b399cf3e522a2c3f39f66a516111"
    "47b6171dd3b9e223f20791bc6275caf0"
)

HISTORICAL_STUDY_IDS = {
    "hardsigmoid_lr_study_sgd_bs16_v1.json": (
        "lrstudy_2103c850c10540d5628cd20cbebc0af5"
        "19bd871116062e8b80e50ad0c068a9a4"
    ),
    "hardsigmoid_lr_rescue_warmin_sgd_bs16_v2.json": (
        "lrstudy_aa0a5d954056594099cf76ed49a8ef15"
        "9f9e707d207b0784fc11f44212f9b0a6"
    ),
    "hardsigmoid_lr_relative_rho_sgd_bs16_v3.json": (
        "lrstudy_b2cffbb9c9976c58338141fc179fba2f"
        "f4f37b3d02009ef9255e3d5a21d650d2"
    ),
    "hardsigmoid_lr_layerwise_relative_rho_constant_sgd_bs16_v4.json": (
        "lrstudy_b345477de5ff64c828a9d31602314a00"
        "046bc9728039562d3e5011132805e14d"
    ),
    "hardsigmoid_lr_architecture_relative_rho_constant_sgd_bs16_v5.json": (
        "lrstudy_5da3a7452c00d42325bfe930d80f0377"
        "91125d3ce7b5ddd79be13ff8b96b49e2"
    ),
    "hardsigmoid_lr_conv2_two_rho_constant_sgd_bs16_v6.json": (
        "lrstudy_c735d2beead9fcbadda89a65ffe86257"
        "da53f15e6cbfab7b4f04b14b08ff6c2f"
    ),
}


def test_v7_is_a_strict_content_addressed_conv3_only_contract() -> None:
    study = LRStudySpec.from_path(V7_CONFIG)

    assert study.data["schema_version"] == (
        LR_CONV3_SCHEME_TWO_RHO_STUDY_SCHEMA_VERSION
    )
    assert study.data["protocol_id"] == (
        "conv-hardsigmoid-lr-conv3-scheme-two-rho-median-constant-sgd-bs16-v7"
    )
    assert study.study_id == V7_STUDY_ID
    assert tuple(study.rows) == FROZEN_CONV3_SCHEME_TWO_RHO_ROWS
    assert [row["row_id"] for row in study.rows] == [
        "conv3_baseline_v1_c1",
        "conv3_ours_v4_c1",
        "conv3_legacy_v4_c0p25",
    ]
    assert [
        (row["input_gain"], row["inference_iterations"], row["training_iterations"])
        for row in study.rows
    ] == [
        (251.306137085, 24, 8),
        (744.739074707, 32, 8),
        (665.0302124023, 8, 6),
    ]

    renamed = copy.deepcopy(study.data)
    renamed["name"] = "display-name-does-not-change-scientific-identity"
    assert LRStudySpec.from_dict(renamed).study_id == V7_STUDY_ID

    changed = copy.deepcopy(study.data)
    changed["rho_grid"]["core"]["rho_dense"][-1] = 0.04
    with pytest.raises(SpecValidationError, match="frozen rho_grid contract"):
        LRStudySpec.from_dict(changed)

    changed_row = copy.deepcopy(study.data)
    changed_row["rows"][1]["input_gain"] = 744.0
    with pytest.raises(SpecValidationError, match=r"study\.rows"):
        LRStudySpec.from_dict(changed_row)


def test_v7_freezes_ordinary_mnist_seed0_probe_and_three_epoch_schedule() -> None:
    data = LRStudySpec.from_path(V7_CONFIG).data

    assert data["dataset"]["variant"] == "ordinary"
    assert data["dataset"]["affine"]["enabled"] is False
    assert data["dataset"]["train"] == {
        "source": "mnist_train",
        "size": 55000,
        "batch_size": 16,
        "drop_last": False,
        "shuffle_method": "torch_randperm",
        "shuffle_seed": 0,
        "reset_shuffle_identically_before_each_candidate": True,
        "steps_per_epoch": 3438,
    }
    assert data["dataset"]["validation"]["size"] == 5000
    assert data["dataset"]["validation"]["batch_size"] == 64
    assert data["dataset"]["official_test"] == {
        "enabled": False,
        "read_allowed": False,
    }
    assert data["model"]["model_seed"] == 0
    assert data["model"]["input_gain_calibration"][
        "reuse_for_ordinary_mnist_is_intentional"
    ] is True
    assert data["model"]["reset_global_name_counters_before_build"] is True
    assert data["probe"]["minibatches"] == 32
    assert data["probe"]["shared_minibatch_sequence_across_rows"] is True
    assert data["probe"]["rho_definition"] == (
        "median_batch_rms_gradient_over_initial_weight_rms_per_bounded_weight"
    )
    assert data["probe"]["bounded_parameter_names"] == [
        "ConvWeight_0",
        "ConvWeight_1",
        "ConvWeight_2",
        "DenseWeight_0",
    ]
    assert data["optimizer"]["bias_weight_mapping"] == {
        "Bias_0": "ConvWeight_0",
        "Bias_1": "ConvWeight_1",
        "Bias_2": "ConvWeight_2",
    }

    training = data["candidate_training"]
    assert (training["epochs"], training["steps_per_epoch"], training["total_steps"]) == (
        3,
        3438,
        10314,
    )
    assert training["minimum_final_validation_accuracy"] == 0.9
    assert training["schedule"] == {
        "type": "constant_weight_vector",
        "scheduler_enabled": False,
        "first_step": 1,
        "final_step": 10314,
        "first_lr_factor": 1.0,
        "final_lr_factor": 1.0,
        "warmup_steps": 0,
        "decay": "none",
        "restarts": False,
    }
    assert data["selection"]["accuracy_boundary"] == "greater_than_or_equal"
    assert data["selection"]["selected_status"] == (
        "frozen_seed0_ordinary_mnist_layerwise_screen"
    )
    assert data["selection"]["final_paper_training_authorized"] is False


def test_v7_freezes_reduced_grid_single_expansion_and_stage_outputs() -> None:
    data = LRStudySpec.from_path(V7_CONFIG).data
    grid = data["rho_grid"]

    assert grid["scope"] == "each_scheme_independently"
    assert grid["core"] == {
        "rho_conv": [0.0005, 0.003, 0.01],
        "rho_dense": [0.003, 0.01, 0.03],
        "ordering": "row_then_rho_conv_then_rho_dense",
        "candidate_count_per_row": 9,
        "total_candidate_count": 27,
    }
    assert grid["expansion"]["rho_conv"] == 0.03
    assert grid["expansion"]["rho_dense"] == 0.1
    assert grid["expansion"]["maximum_waves"] == 1
    assert grid["expansion"]["maximum_new_candidates_per_row"] == 7
    assert grid["expansion"]["maximum_candidates_per_row"] == 16
    assert grid["expansion"]["maximum_total_training_runs"] == 48
    assert grid["expansion"]["reuse_existing_cells_by_hash"] is True
    assert grid["expansion"]["second_automatic_expansion_allowed"] is False

    stages = data["stages"]
    assert stages["candidate_run_schema_version"] == "mnist-conv-run/v7"
    assert stages["public_sequence"] == [
        "audit",
        "probe",
        "preflight",
        "core_candidates",
        "select_core",
        "extension_candidates",
        "finalize",
    ]
    assert stages["conditional_zero_work_completion"]["extension_candidates"] == (
        "required_when_no_scheme_requests_expansion"
    )
    assert stages["contracts"]["core_candidates"]["output"] == (
        "twenty_seven_candidate_completions"
    )
    assert stages["contracts"]["extension_candidates"]["output"] == (
        "zero_to_twenty_one_candidate_completions"
    )
    assert data["artifacts"]["final_handoff_fields"] == [
        "rho_conv",
        "rho_dense",
        "four_weight_learning_rates",
        "seven_parameter_learning_rate_vector",
        "achieved_relative_updates_by_weight",
        "achieved_span_updates_by_weight",
        "validation_metrics",
        "safety_diagnostics",
        "provenance_hashes",
    ]
    assert data["artifacts"]["ordinary_mnist_only_handoff"] is True
    assert data["artifacts"]["replaces_medium_affine_paper_handoff"] is False
    assert data["artifacts"]["authorizes_final_paper_training"] is False


def test_v7_freezes_worst_cost_preflight_and_execution_boundaries() -> None:
    data = LRStudySpec.from_path(V7_CONFIG).data

    assert data["preflight"] == {
        "required_before_core_candidates": True,
        "row_id": "conv3_ours_v4_c1",
        "rho_conv": 0.003,
        "rho_dense": 0.01,
        "inference_iterations": 32,
        "training_iterations": 8,
        "warmup_steps": 32,
        "measured_steps": 256,
        "full_validation_pass": True,
        "validation_batch_size": 64,
        "device": "v100-16g-or-larger",
        "gpu_memory_mib": 16384,
        "memory_headroom_fraction": 0.1,
        "maximum_projected_candidate_hours": 20,
        "failure_policy": "stop_and_revise_protocol",
        "batch_size_change_allowed": False,
    }
    execution = data["execution"]
    assert execution["availability_checks"] == ["main", "akibscomputer", "trex"]
    assert execution["conv3_dispatch_priority"][:2] == ["trex", "jean_zay_r3"]
    assert execution["allocation"] == "AD010913993R3"
    assert execution["project"] == "fmu"
    assert execution["slurm_account"] == "fmu@v100"
    assert execution["constraint"] == "v100"
    assert execution["gpu_memory_mib"] == 16384
    assert execution["default_candidate_processes_per_gpu"] == 1
    assert execution["packing_allowed_only_after_measured_preflight"] is True
    assert execution["packing_requires_memory_and_throughput_benchmark"] is True


@pytest.mark.parametrize(
    ("filename", "expected_study_id"),
    HISTORICAL_STUDY_IDS.items(),
)
def test_v7_addition_preserves_v1_through_v6_identities(
    filename: str,
    expected_study_id: str,
) -> None:
    assert LRStudySpec.from_path(CONFIG_ROOT / filename).study_id == expected_study_id
