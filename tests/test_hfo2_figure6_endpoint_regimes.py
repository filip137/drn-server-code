import math

import pytest
import torch

from experiments.mnist_relu_drn.hfo2_figure6_endpoint_regimes import (
    FIGURE6_RESET_TO_SET_STD_RATIO,
    HFO2_FIGURE6_RESET_STATE_STD,
    HFO2_SET_STATE_STD,
    INDEPENDENT_ENDPOINTS,
    RANK_MATCHED_ENDPOINTS,
    RESET_STATE_FLOOR,
    full_tile_RESET,
    full_tile_SET,
    full_tile_reset,
    full_tile_set,
    sample_hfo2_figure6_endpoint_population,
)


def test_reset_standard_deviation_uses_the_figure6_ratio_before_clipping() -> None:
    assert FIGURE6_RESET_TO_SET_STD_RATIO == pytest.approx(6.4411 / 32.5241)
    assert HFO2_FIGURE6_RESET_STATE_STD == pytest.approx(
        HFO2_SET_STATE_STD * FIGURE6_RESET_TO_SET_STD_RATIO
    )
    assert HFO2_FIGURE6_RESET_STATE_STD == pytest.approx(0.08505853966750809)


def test_full_tile_reset_clamps_only_the_lower_tail_and_retains_every_device() -> None:
    z = torch.tensor([-10.0, 0.0, 1.0], dtype=torch.float64)
    realized, raw, clipped = full_tile_reset(z)

    assert raw[1].item() == pytest.approx(0.1)
    assert realized.tolist() == pytest.approx(
        [RESET_STATE_FLOOR, 0.1, 0.1 + HFO2_FIGURE6_RESET_STATE_STD]
    )
    assert clipped.tolist() == [True, False, False]
    assert realized.numel() == z.numel()


def test_full_tile_set_is_the_unclipped_gaussian_affine_law() -> None:
    z = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)
    state = full_tile_set(z)

    assert torch.equal(state, 2.0 + HFO2_SET_STATE_STD * z)
    assert state[0].item() < 2.0
    assert state[-1].item() > 2.0
    assert full_tile_RESET is full_tile_reset
    assert full_tile_SET is full_tile_set


@pytest.mark.parametrize(
    "regime", [INDEPENDENT_ENDPOINTS, RANK_MATCHED_ENDPOINTS]
)
def test_population_draw_is_reproducible_and_records_the_endpoint_contract(
    regime: str,
) -> None:
    first = sample_hfo2_figure6_endpoint_population(
        devices=200,
        assignment_seed=20260903,
        regime=regime,
    )
    replay = sample_hfo2_figure6_endpoint_population(
        devices=200,
        assignment_seed=20260903,
        regime=regime,
    )

    assert torch.equal(first.reset_state, replay.reset_state)
    assert torch.equal(first.set_state, replay.set_state)
    assert torch.all(first.reset_state >= RESET_STATE_FLOOR)
    assert torch.all(first.set_state > first.reset_state)
    report = first.report()
    assert report["stock_aihwkit_preset_modified"] is False
    assert report["full_tile_RESET"]["lower_clip"] == RESET_STATE_FLOOR
    assert report["full_tile_SET"]["lower_clip"] is None
    assert report["full_tile_SET"]["upper_clip"] is None
    assert report["reset_state_sha256"] == replay.report()["reset_state_sha256"]
    assert report["set_state_sha256"] == replay.report()["set_state_sha256"]


def test_pairing_regimes_change_only_the_joint_endpoint_assumption() -> None:
    independent = sample_hfo2_figure6_endpoint_population(
        devices=200,
        assignment_seed=17,
        regime=INDEPENDENT_ENDPOINTS,
    )
    rank_matched = sample_hfo2_figure6_endpoint_population(
        devices=200,
        assignment_seed=17,
        regime=RANK_MATCHED_ENDPOINTS,
    )

    assert not torch.equal(
        independent.reset_standard_normal,
        independent.set_standard_normal,
    )
    assert torch.equal(
        independent.reset_standard_normal,
        rank_matched.reset_standard_normal,
    )
    assert torch.equal(
        torch.sort(independent.set_standard_normal).values,
        torch.sort(rank_matched.set_standard_normal).values,
    )
    assert torch.equal(
        torch.argsort(rank_matched.reset_standard_normal),
        torch.argsort(rank_matched.set_standard_normal),
    )
    assert math.isfinite(
        rank_matched.report()["dynamic_range_SET_over_RESET"]["median"]
    )
