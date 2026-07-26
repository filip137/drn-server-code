from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.mnist_conv.perfectdiode_hparam_spec import (
    CANDIDATE_TOTAL_STEPS,
    CANARY_STEPS,
    FROZEN_ROWS,
    PD_RUN_SCHEMA_VERSION,
    PD_STUDY_SCHEMA_VERSION,
    PUBLIC_STAGE_SEQUENCE,
    SURFACES,
    PerfectDiodeHparamStudySpec,
    PerfectDiodeHparamValidationError,
    boundary_expansion_plan,
    core_grid,
    probe_stability_decision,
    safe_center,
    select_surface_candidates,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv12_sgd_adam_hparam_ordinary_mnist_v1.json"
)
EXPECTED_CONFIG_SHA256 = (
    "699804e1f7c35f65f50656db4af0b120dd3a3d0f2fb95e3a17b65ebfe7b78535"
)
EXPECTED_STUDY_ID = (
    "lrstudy_5afe8bc7c9180cd1c677d1d87a497e1d6a89c1599cd015d7a6c127ebb18f2e46"
)


def _results(
    grid: tuple[tuple[float, float], ...],
    *,
    winner: tuple[float, float],
    winner_accuracy: float = 0.9,
    winner_loss: float = 0.2,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (rho_conv, rho_dense) in enumerate(grid):
        is_winner = (rho_conv, rho_dense) == winner
        rows.append(
            {
                "candidate_id": f"candidate-{index:02d}",
                "rho_conv": rho_conv,
                "rho_dense": rho_dense,
                "admissible": True,
                "completed_steps": CANDIDATE_TOTAL_STEPS,
                "final_validation_loss": winner_loss if is_winner else 1.0,
                "final_validation_accuracy": (
                    winner_accuracy if is_winner else 0.89
                ),
                "median_projection_efficiency": 0.8,
            }
        )
    return rows


def test_config_has_strict_identity_and_does_not_reuse_lr_v1_through_v7() -> None:
    study = PerfectDiodeHparamStudySpec.from_path(CONFIG)

    assert study.data["schema_version"] == PD_STUDY_SCHEMA_VERSION
    assert study.data["run_schema_version"] == PD_RUN_SCHEMA_VERSION
    assert study.config_sha256 == EXPECTED_CONFIG_SHA256
    assert study.study_id == EXPECTED_STUDY_ID
    assert study.study_id.startswith("lrstudy_")
    assert len(study.study_id) == len("pdstudy_") + 64

    changed = study.data
    changed["candidate_training"]["epochs"] = 5
    with pytest.raises(
        PerfectDiodeHparamValidationError,
        match="immutable perfect-diode Conv1/Conv2 v1 config",
    ):
        PerfectDiodeHparamStudySpec.from_dict(changed)


def test_rows_freeze_six_operating_points_and_twelve_independent_surfaces() -> None:
    study = PerfectDiodeHparamStudySpec.from_path(CONFIG)

    assert study.rows == FROZEN_ROWS
    assert len(study.rows) == 6
    assert len(study.surfaces) == 12
    assert study.surfaces == SURFACES
    assert [(row["architecture"], row["input_gain"]) for row in study.rows] == [
        ("conv1", 40.0),
        ("conv1", 40.0),
        ("conv1", 40.0),
        ("conv2", 100.0),
        ("conv2", 100.0),
        ("conv2", 100.0),
    ]
    assert {
        (
            row["architecture"],
            row["inference_iterations"],
            row["training_iterations"],
            row["reference_inference_iterations"],
            row["reference_training_iterations"],
        )
        for row in study.rows
    } == {
        ("conv1", 4, 4, 64, 64),
        ("conv2", 6, 6, 64, 64),
    }
    assert [row["scheme"] for row in study.rows] == [
        "baseline",
        "ours",
        "legacy",
        "baseline",
        "ours",
        "legacy",
    ]


def test_dataset_optimizer_and_public_stage_contracts_are_exact() -> None:
    data = PerfectDiodeHparamStudySpec.from_path(CONFIG).data

    assert data["dataset"]["variant"] == "ordinary"
    assert data["dataset"]["train"]["size"] == 55_000
    assert data["dataset"]["validation"]["size"] == 5_000
    assert data["dataset"]["train"]["batch_size"] == 16
    assert data["dataset"]["validation"]["batch_size"] == 64
    assert data["dataset"]["official_test"] == {
        "enabled": False,
        "read_allowed": False,
    }
    assert data["model"]["model_seed"] == 0
    assert data["model"]["non_linearity"] == "perfect_diode"
    assert data["model"]["quadratic_diode_param"] == {}
    assert data["model"]["exponential_diode_param"] == {}
    assert data["model"]["hard_sigmoid"] == {}
    assert data["optimizer_arms"]["sgd"] == {
        "name": "SGD",
        "momentum": 0.0,
        "weight_decay": 0.0,
    }
    assert data["optimizer_arms"]["adam"] == {
        "name": "Adam",
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "foreach": False,
        "fused": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
    }
    assert data["stages"]["public_sequence"] == list(PUBLIC_STAGE_SEQUENCE)


def test_security_probe_canary_candidate_and_confirmation_counts_are_frozen() -> None:
    data = PerfectDiodeHparamStudySpec.from_path(CONFIG).data
    security = data["fixed_tk_gradient_security"]

    assert security["comparisons"] == {
        "conv1": {"operational": [4, 4], "reference": [64, 64]},
        "conv2": {"operational": [6, 6], "reference": [64, 64]},
    }
    assert security["gates"] == {
        "relative_gradient_l2_norm_delta_maximum": 0.1,
        "absolute_zero_fraction_delta_maximum": 0.02,
        "gradient_vector_cosine_minimum": 0.9,
    }
    assert data["optimizer_probe"]["batch_counts"] == [32, 64, 128]
    assert data["optimizer_probe"]["split_half_relative_difference_threshold"] == 0.1
    assert data["optimizer_probe"]["bias_learning_rate_formula"] == (
        "min(attached_conv_weight_lr,rho_conv/u_bias_i_q90)"
    )
    assert data["rho_search"]["center_canary"]["steps"] == CANARY_STEPS == 640
    assert data["rho_search"]["core"]["candidate_count_per_surface"] == 9
    assert data["rho_search"]["expansion"]["maximum_cells_per_surface"] == 16
    assert data["candidate_training"] == {
        "epochs": 3,
        "steps_per_epoch": 3438,
        "total_steps": 10314,
        "batch_size": 16,
        "validation_batch_size": 64,
        "validate_after_every_epoch": True,
        "restart_from_shared_initialization": True,
        "continue_from_canary": False,
        "schedule": {
            "type": "constant_parameter_specific_lr_vector",
            "warmup_steps": 0,
            "scheduler_enabled": False,
            "decay": "none",
            "restarts": False,
        },
        "promotion": "every_safety_clean_canary_cell",
        "failed_canary_stops_before_candidate": True,
        "minimum_final_validation_accuracy": 0.9,
        "accuracy_boundary": "greater_than_or_equal",
        "official_test_evaluation": False,
    }
    assert data["long_confirm"]["architectures"] == {
        "conv1": {"epochs": 10, "steps_per_epoch": 3438, "total_steps": 34380},
        "conv2": {"epochs": 30, "steps_per_epoch": 3438, "total_steps": 103140},
    }
    assert data["long_confirm"]["maximum_run_count"] == 12


def test_adaptive_probe_uses_strict_ten_percent_boundary_and_caps_at_128() -> None:
    stable = probe_stability_decision(
        {"ConvWeight_0": 0.10, "DenseWeight_0": 0.01},
        used_batches=32,
    )
    assert stable.status == "stable"
    assert stable.next_batches is None

    first = probe_stability_decision(
        {"ConvWeight_0": 0.1000001, "DenseWeight_0": 0.01},
        used_batches=32,
    )
    assert (first.status, first.next_batches, first.unstable_parameters) == (
        "extend",
        64,
        ("ConvWeight_0",),
    )
    second = probe_stability_decision(
        {"ConvWeight_1": 0.2}, used_batches=64
    )
    assert (second.status, second.next_batches) == ("extend", 128)
    final = probe_stability_decision(
        {"ConvWeight_1": 0.2}, used_batches=128
    )
    assert (final.status, final.next_batches) == ("unresolved", None)


def test_restarted_center_and_core_grid_are_factor_three_and_capped() -> None:
    assert safe_center(0) == (0.003, 0.01)
    assert safe_center(1) == (0.001, 0.01 / 3.0)
    assert safe_center(5) == (0.003 / 243.0, 0.01 / 243.0)
    with pytest.raises(PerfectDiodeHparamValidationError, match="attempt"):
        safe_center(6)

    grid = core_grid(*safe_center(0))
    assert len(grid) == 9
    assert sorted({pair[0] for pair in grid}) == pytest.approx(
        [0.001, 0.003, 0.009]
    )
    assert sorted({pair[1] for pair in grid}) == pytest.approx([
        0.01 / 3.0,
        0.01,
        0.03,
    ])


def test_symmetric_expansion_adds_only_implicated_lower_or_upper_axis() -> None:
    grid = core_grid(0.003, 0.01)
    conv = tuple(sorted({pair[0] for pair in grid}))
    dense = tuple(sorted({pair[1] for pair in grid}))

    lower_conv = boundary_expansion_plan(
        [
            {"rho_conv": conv[0], "rho_dense": dense[0]},
            {"rho_conv": conv[0], "rho_dense": dense[1]},
        ],
        rho_conv_values=conv,
        rho_dense_values=dense,
    )
    assert lower_conv.directions == (("rho_conv", "lower"),)
    assert lower_conv.rho_conv_values[0] == conv[0] / 3.0
    assert len(lower_conv.new_cells) == 3

    upper_dense = boundary_expansion_plan(
        [
            {"rho_conv": conv[0], "rho_dense": dense[-1]},
            {"rho_conv": conv[1], "rho_dense": dense[-1]},
        ],
        rho_conv_values=conv,
        rho_dense_values=dense,
    )
    assert upper_dense.directions == (("rho_dense", "upper"),)
    assert upper_dense.rho_dense_values[-1] == dense[-1] * 3.0
    assert len(upper_dense.new_cells) == 3


def test_corner_expansion_adds_both_axes_and_never_exceeds_sixteen_cells() -> None:
    grid = core_grid(0.003, 0.01)
    conv = tuple(sorted({pair[0] for pair in grid}))
    dense = tuple(sorted({pair[1] for pair in grid}))
    plan = boundary_expansion_plan(
        [{"rho_conv": conv[-1], "rho_dense": dense[0]}],
        rho_conv_values=conv,
        rho_dense_values=dense,
    )

    assert plan.directions == (
        ("rho_conv", "upper"),
        ("rho_dense", "lower"),
    )
    assert len(plan.rho_conv_values) == 4
    assert len(plan.rho_dense_values) == 4
    assert len(plan.new_cells) == 7
    assert len(core_grid(0.003, 0.01)) + len(plan.new_cells) == 16


def test_l_shaped_plateau_implicates_both_incident_edges() -> None:
    grid = core_grid(0.003, 0.01)
    conv = tuple(sorted({pair[0] for pair in grid}))
    dense = tuple(sorted({pair[1] for pair in grid}))
    plan = boundary_expansion_plan(
        [
            {"rho_conv": conv[-1], "rho_dense": dense[1]},
            {"rho_conv": conv[1], "rho_dense": dense[-1]},
        ],
        rho_conv_values=conv,
        rho_dense_values=dense,
    )

    assert plan.directions == (
        ("rho_conv", "upper"),
        ("rho_dense", "upper"),
    )
    assert len(plan.new_cells) == 7


def test_selection_uses_inclusive_ninety_percent_gate_and_bracketed_center() -> None:
    grid = core_grid(0.003, 0.01)
    conv = tuple(sorted({pair[0] for pair in grid}))
    dense = tuple(sorted({pair[1] for pair in grid}))
    center = (conv[1], dense[1])
    selection = select_surface_candidates(
        _results(grid, winner=center, winner_accuracy=0.9),
        rho_conv_values=conv,
        rho_dense_values=dense,
        expansion_available=True,
    )

    assert selection["status"] == "selected"
    assert selection["selected"]["final_validation_accuracy"] == 0.9
    assert (
        selection["selected"]["rho_conv"],
        selection["selected"]["rho_dense"],
    ) == center


def test_core_boundary_requests_one_expansion_then_final_boundary_is_unresolved() -> None:
    grid = core_grid(0.003, 0.01)
    conv = tuple(sorted({pair[0] for pair in grid}))
    dense = tuple(sorted({pair[1] for pair in grid}))
    upper = (conv[-1], dense[1])
    core_selection = select_surface_candidates(
        _results(grid, winner=upper),
        rho_conv_values=conv,
        rho_dense_values=dense,
        expansion_available=True,
    )

    assert core_selection["status"] == "needs_expansion"
    assert core_selection["selected"] is None
    plan = core_selection["expansion"]
    assert plan.directions == (("rho_conv", "upper"),)
    assert len(plan.new_cells) == 3

    expanded_grid = tuple(grid) + tuple(plan.new_cells)
    outer = (plan.rho_conv_values[-1], dense[1])
    final_selection = select_surface_candidates(
        _results(expanded_grid, winner=outer),
        rho_conv_values=plan.rho_conv_values,
        rho_dense_values=plan.rho_dense_values,
        expansion_available=False,
    )
    assert final_selection["status"] == "unresolved_boundary"
    assert final_selection["selected"] is None


def test_no_passing_candidate_does_not_request_expansion() -> None:
    grid = core_grid(0.003, 0.01)
    conv = tuple(sorted({pair[0] for pair in grid}))
    dense = tuple(sorted({pair[1] for pair in grid}))
    candidates = _results(grid, winner=(conv[-1], dense[-1]), winner_accuracy=0.8999)

    selection = select_surface_candidates(
        candidates,
        rho_conv_values=conv,
        rho_dense_values=dense,
        expansion_available=True,
    )

    assert selection["status"] == "unresolved_no_pass"
    assert selection["expansion"].required is False


def test_candidate_requires_exact_three_epoch_step_count() -> None:
    grid = core_grid(0.003, 0.01)
    conv = tuple(sorted({pair[0] for pair in grid}))
    dense = tuple(sorted({pair[1] for pair in grid}))
    candidates = _results(grid, winner=(conv[1], dense[1]))
    candidates[0] = copy.deepcopy(candidates[0])
    candidates[0]["completed_steps"] = CANDIDATE_TOTAL_STEPS - 1

    with pytest.raises(
        PerfectDiodeHparamValidationError,
        match="exactly 10314",
    ):
        select_surface_candidates(
            candidates,
            rho_conv_values=conv,
            rho_dense_values=dense,
            expansion_available=True,
        )
