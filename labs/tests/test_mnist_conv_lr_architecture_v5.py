from __future__ import annotations

import pytest

from experiments.mnist_conv.lr_protocol import (
    alpha_candidate_grid,
    architecture_relative_learning_rates,
    probe_median_parameter_relative_units,
    select_v5_architecture,
)


SCHEMES = ("baseline", "ours", "legacy")


def _rows(
    *,
    losses: tuple[float, float, float],
    accuracies: tuple[float, float, float],
    efficiencies: tuple[float, float, float] = (0.8, 0.8, 0.8),
    admissible: tuple[bool, bool, bool] = (True, True, True),
) -> list[dict[str, object]]:
    return [
        {
            "scheme": scheme,
            "admissible": is_admissible,
            "final_validation_loss": loss if is_admissible else None,
            "final_validation_accuracy": accuracy if is_admissible else None,
            "median_projection_efficiency": efficiency if is_admissible else None,
            "inadmissible_reason": None if is_admissible else "projection_safety_gate",
        }
        for scheme, loss, accuracy, efficiency, is_admissible in zip(
            SCHEMES,
            losses,
            accuracies,
            efficiencies,
            admissible,
            strict=True,
        )
    ]


def _candidate(
    arm: str,
    alpha: float,
    *,
    losses: tuple[float, float, float],
    accuracies: tuple[float, float, float],
    efficiencies: tuple[float, float, float] = (0.8, 0.8, 0.8),
    admissible: tuple[bool, bool, bool] = (True, True, True),
) -> dict[str, object]:
    return {
        "arm": arm,
        "alpha": alpha,
        "rows": _rows(
            losses=losses,
            accuracies=accuracies,
            efficiencies=efficiencies,
            admissible=admissible,
        ),
    }


def test_probe_uses_per_weight_batch_median_not_q90_or_cross_weight_max() -> None:
    units = probe_median_parameter_relative_units(
        {
            "ConvWeight_0": (0.1, 0.4, 0.2, 0.3),
            "ConvWeight_1": (4.0, 1.0, 3.0, 2.0),
            "DenseWeight_0": (0.8, 0.2, 0.6, 0.4),
        }
    )

    assert units == pytest.approx(
        {
            "ConvWeight_0": 0.25,
            "ConvWeight_1": 2.5,
            "DenseWeight_0": 0.5,
        }
    )
    assert units["ConvWeight_0"] != pytest.approx(0.37)

    with pytest.raises(ValueError, match="bounded weights"):
        probe_median_parameter_relative_units(
            {"ConvWeight_0": (0.1, 0.2), "Bias_0": (0.3, 0.4)}
        )


def test_strict_and_historical_profiles_hit_targets_and_tie_biases() -> None:
    units = {
        "ConvWeight_0": 0.2,
        "ConvWeight_1": 0.4,
        "DenseWeight_0": 2.0,
    }
    parameter_names = (
        "DenseWeight_0",
        "Bias_1",
        "ConvWeight_0",
        "Bias_0",
        "ConvWeight_1",
    )

    strict = architecture_relative_learning_rates(
        units,
        0.01,
        parameter_names,
        target_multipliers={name: 1.0 for name in units},
    )
    assert strict == pytest.approx(
        {
            "ConvWeight_0": 0.05,
            "Bias_0": 0.05,
            "ConvWeight_1": 0.025,
            "Bias_1": 0.025,
            "DenseWeight_0": 0.005,
        }
    )

    conv2_profile = architecture_relative_learning_rates(
        units,
        5e-4,
        parameter_names,
        target_multipliers={
            "ConvWeight_0": 1.0,
            "ConvWeight_1": 1.0,
            "DenseWeight_0": 20.0,
        },
    )
    assert conv2_profile == pytest.approx(
        {
            "ConvWeight_0": 0.0025,
            "Bias_0": 0.0025,
            "ConvWeight_1": 0.00125,
            "Bias_1": 0.00125,
            "DenseWeight_0": 0.005,
        }
    )
    for weight, multiplier in {
        "ConvWeight_0": 1.0,
        "ConvWeight_1": 1.0,
        "DenseWeight_0": 20.0,
    }.items():
        assert conv2_profile[weight] * units[weight] == pytest.approx(
            5e-4 * multiplier
        )

    conv1_profile = architecture_relative_learning_rates(
        {"ConvWeight_0": 0.2, "DenseWeight_0": 2.0},
        1e-3,
        ("Bias_0", "DenseWeight_0", "ConvWeight_0"),
        target_multipliers={"ConvWeight_0": 1.0, "DenseWeight_0": 30.0},
    )
    assert conv1_profile["DenseWeight_0"] == pytest.approx(0.015)
    assert conv1_profile["Bias_0"] == conv1_profile["ConvWeight_0"]


def test_alpha_grid_is_architecture_level_and_not_scheme_specific() -> None:
    centers = {
        "conv1": {"strict_equal": 1e-2, "historical_profile": 1e-3},
        "conv2": {"strict_equal": 1e-3, "historical_profile": 5e-4},
    }
    grids = {
        architecture: {
            arm: alpha_candidate_grid(center) for arm, center in arms.items()
        }
        for architecture, arms in centers.items()
    }

    assert grids["conv1"]["strict_equal"] == pytest.approx(
        (1e-2 / 3.0, 1e-2, 3e-2)
    )
    assert grids["conv2"]["strict_equal"] == pytest.approx(
        (1e-3 / 3.0, 1e-3, 3e-3)
    )
    for architecture, arms in grids.items():
        for arm, alpha_grid in arms.items():
            assigned_by_scheme = {scheme: alpha_grid for scheme in SCHEMES}
            assert len({assigned_by_scheme[scheme] for scheme in SCHEMES}) == 1, (
                architecture,
                arm,
            )


def test_architecture_selection_applies_joint_90_percent_gate_and_plateaus() -> None:
    candidates = [
        _candidate(
            "strict_equal",
            1e-3,
            losses=(0.48, 0.50, 0.49),
            accuracies=(0.90, 0.90, 0.90),
            efficiencies=(0.7, 0.7, 0.7),
        ),
        _candidate(
            "strict_equal",
            3e-3,
            losses=(0.20, 0.20, 0.20),
            accuracies=(0.95, 0.8999, 0.95),
            efficiencies=(0.9, 0.9, 0.9),
        ),
        _candidate(
            "strict_equal",
            9e-3,
            losses=(0.49, 0.505, 0.50),
            accuracies=(0.95, 0.95, 0.95),
            efficiencies=(0.8, 0.8, 0.8),
        ),
        _candidate(
            "historical_profile",
            1e-4,
            losses=(0.49, 0.50, 0.50),
            accuracies=(0.99, 0.99, 0.99),
            efficiencies=(0.99, 0.99, 0.99),
        ),
    ]

    selected = select_v5_architecture(candidates)

    assert selected["status"] == "selected"
    assert selected["arm_selections"]["strict_equal"]["viable_alphas"] == [
        1e-3,
        9e-3,
    ]
    assert selected["arm_selections"]["strict_equal"]["selected_alpha"] == 9e-3
    assert selected["selected_arm"] == "strict_equal"
    assert selected["selected_alpha"] == 9e-3


def test_profile_wins_only_when_its_worst_loss_is_outside_strict_plateau() -> None:
    selected = select_v5_architecture(
        [
            _candidate(
                "strict_equal",
                1e-3,
                losses=(0.50, 0.51, 0.50),
                accuracies=(0.96, 0.96, 0.96),
            ),
            _candidate(
                "historical_profile",
                5e-4,
                losses=(0.48, 0.49, 0.48),
                accuracies=(0.91, 0.91, 0.91),
            ),
        ]
    )

    assert selected["selected_arm"] == "historical_profile"
    assert selected["selected_alpha"] == 5e-4


def test_alpha_plateau_uses_efficiency_then_lower_alpha_tie_breakers() -> None:
    selected = select_v5_architecture(
        [
            _candidate(
                "strict_equal",
                1e-3,
                losses=(0.50, 0.50, 0.50),
                accuracies=(0.95, 0.95, 0.95),
                efficiencies=(0.7, 0.7, 0.7),
            ),
            _candidate(
                "strict_equal",
                3e-3,
                losses=(0.505, 0.505, 0.505),
                accuracies=(0.95, 0.95, 0.95),
                efficiencies=(0.8, 0.8, 0.8),
            ),
            _candidate(
                "strict_equal",
                9e-3,
                losses=(0.505, 0.505, 0.505),
                accuracies=(0.95, 0.95, 0.95),
                efficiencies=(0.8, 0.8, 0.8),
            ),
        ]
    )

    assert selected["selected_arm"] == "strict_equal"
    assert selected["selected_alpha"] == 3e-3


def test_architecture_is_failed_when_no_alpha_passes_for_all_three_schemes() -> None:
    selected = select_v5_architecture(
        [
            _candidate(
                "strict_equal",
                1e-3,
                losses=(0.4, 0.4, 0.4),
                accuracies=(0.95, 0.95, 0.89),
            ),
            _candidate(
                "historical_profile",
                5e-4,
                losses=(0.4, 0.4, 0.4),
                accuracies=(0.95, 0.95, 0.95),
                admissible=(True, False, True),
            ),
        ]
    )

    assert selected["status"] == "failed"
    assert selected["reason"] == "no_common_passing_alpha"
    assert selected["selected_arm"] is None
    assert selected["selected_alpha"] is None
