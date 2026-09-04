from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_apparent_pv_adam_recovery import (
    EXTENDED_FIGURE6_LEARNING_RATE,
    EXTENDED_FIGURE6_PULSE_CAP_PER_CELL,
    EXTENDED_FIGURE6_TRAINING_EPOCHS,
    EXTENDED_FIGURE6_TRAINING_EXAMPLES,
    LEARNING_RATE_GRID,
    MAIN_TRAINING_EXAMPLES,
    MODEL_FIGURE6,
    MODEL_WINSORIZED,
    PULSE_CAP_PER_CELL,
    main_arms,
    select_common_learning_rate,
    winsorized_apparent_projection,
    winsorized_full_conductance,
)
from training.ibm_reram_program_verify import make_buffered_normal_draws
from experiments.mnist_relu_drn import (
    ibm_om_apparent_pv_adam_recovery_experiment as runtime,
)


def test_main_arm_matrix_is_exactly_two_by_two_by_two() -> None:
    arms = main_arms()
    assert len(arms) == 8
    assert len({arm.slug for arm in arms}) == 8


def test_figure6_corrupt_extension_scales_epoch_and_cap_budget() -> None:
    assert EXTENDED_FIGURE6_LEARNING_RATE == 3.0e-5
    assert EXTENDED_FIGURE6_TRAINING_EPOCHS == 3
    assert EXTENDED_FIGURE6_TRAINING_EXAMPLES == 3 * MAIN_TRAINING_EXAMPLES
    assert EXTENDED_FIGURE6_PULSE_CAP_PER_CELL == 3 * PULSE_CAP_PER_CELL


def test_figure6_corrupt_extension_cli_has_only_the_target_axis() -> None:
    args = runtime._parser().parse_args(
        ["extended-figure6-corrupt-arm", "--target", "hwa"]
    )
    assert args.target == "hwa"
    assert not hasattr(args, "model")
    assert not hasattr(args, "corruption")


def test_old_apparent_mapping_projects_but_persistent_mapping_is_strict() -> None:
    raw_a = torch.tensor([-1.2, -1.0, 0.25, 1.0, 1.3], dtype=torch.float32)
    (full_g,) = winsorized_full_conductance(
        raw_a,
        ((5, 1),),
        apparent=True,
    )
    assert torch.equal(
        full_g.reshape(-1),
        torch.tensor([0.0, 0.0, 1.25, 2.0, 2.0]),
    )
    projection = winsorized_apparent_projection(raw_a)
    assert projection["below_Gmin"] == 1
    assert projection["above_Gmax"] == 1
    assert projection["projected_cells"] == 2
    with pytest.raises(RuntimeError, match="left G=\\[0,2\\]"):
        winsorized_full_conductance(raw_a, ((5, 1),), apparent=False)


def _candidate(
    learning_rate: float,
    *,
    accuracy: float,
    kl: float,
    pulses: int,
) -> dict:
    return {
        "learning_rate": learning_rate,
        "anchors": {
            model: {
                "after_validation": {
                    "apparent": {
                        "student_accuracy": accuracy,
                        "kl_teacher_student": kl,
                    }
                },
                "pulses": {"applied": pulses},
            }
            for model in (MODEL_WINSORIZED, MODEL_FIGURE6)
        },
    }


def test_common_rate_selection_uses_apparent_accuracy_then_frozen_ties() -> None:
    candidates = [
        _candidate(rate, accuracy=0.8, kl=0.4, pulses=100)
        for rate in LEARNING_RATE_GRID
    ]
    candidates[1] = _candidate(
        LEARNING_RATE_GRID[1], accuracy=0.9, kl=0.5, pulses=120
    )
    candidates[2] = _candidate(
        LEARNING_RATE_GRID[2], accuracy=0.9, kl=0.3, pulses=140
    )
    candidates[3] = _candidate(
        LEARNING_RATE_GRID[3], accuracy=0.9, kl=0.3, pulses=130
    )
    candidates[4] = _candidate(
        LEARNING_RATE_GRID[4], accuracy=0.9, kl=0.3, pulses=130
    )
    selected = select_common_learning_rate(candidates)
    assert selected["learning_rate"] == LEARNING_RATE_GRID[3]


def test_common_rate_selection_rejects_reordered_grid() -> None:
    candidates = [
        _candidate(rate, accuracy=0.8, kl=0.4, pulses=100)
        for rate in reversed(LEARNING_RATE_GRID)
    ]
    with pytest.raises(ValueError, match="grid order"):
        select_common_learning_rate(candidates)


def test_extending_buffered_per_cell_rng_preserves_saved_stream_prefix() -> None:
    seeds = (17, 23, 41)
    original = make_buffered_normal_draws(
        seeds,
        maximum_random_draws=7,
        device="cpu",
    )
    extended = make_buffered_normal_draws(
        seeds,
        maximum_random_draws=11,
        device="cpu",
    )
    assert torch.equal(original, extended[:, :7])


def test_gradient_producing_forward_is_wired_to_held_apparent_state() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(runtime._run_recovery)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_one_apparent_forward_gradient"
    ]
    assert len(calls) == 1
    full_g = next(
        keyword.value for keyword in calls[0].keywords if keyword.arg == "full_g"
    )
    assert isinstance(full_g, ast.Call)
    assert isinstance(full_g.func, ast.Attribute)
    assert full_g.func.attr == "apparent_full_g"
