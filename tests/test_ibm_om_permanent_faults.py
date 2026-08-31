from __future__ import annotations

import math

import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_permanent_faults import (
    DEFAULT_CORRUPT_PROBABILITY,
    DEFAULT_CORRUPT_RAW_A_RANGE,
    FULL_CONDUCTANCE_COORDINATE,
    RAW_A_COORDINATE,
    sample_aihwkit_compatible_permanent_fault_overlay,
)
from training.ibm_reram_hwa import IbmReramArrayPopulation
from training.ibm_reram_program_verify import IbmReramRawActivePlant


def _repaired_population(
    shapes: tuple[tuple[int, int], ...],
    *,
    minimum: float = -0.8,
    maximum: float = 0.8,
) -> IbmReramArrayPopulation:
    size = sum(math.prod(shape) for shape in shapes)
    return IbmReramArrayPopulation(
        assignment_seed=87004,
        corruption_policy="counterfactual_repaired",
        binding_keys=tuple(f"w{index + 1}" for index in range(len(shapes))),
        binding_shapes=shapes,
        binding_sampling_seeds=tuple(101 + index for index in range(len(shapes))),
        donor_sampling_seeds=tuple(201 + index for index in range(len(shapes))),
        nominal_dw_min=0.05,
        dw_min_std=0.2,
        write_noise_std=0.3,
        max_bound=torch.full((size,), maximum, dtype=torch.float32),
        min_bound=torch.full((size,), minimum, dtype=torch.float32),
        dwmin_up=torch.full((size,), 0.05, dtype=torch.float32),
        dwmin_down=torch.full((size,), 0.04, dtype=torch.float32),
        reference=torch.zeros(size, dtype=torch.float32),
        corrupt=torch.zeros(size, dtype=torch.bool),
        published_corrupt=torch.zeros(size, dtype=torch.bool),
        fingerprint="repaired-test-population",
        aihwkit_version="1.1.0",
    )


def test_full_drn_masks_91501_through_91505_replay_with_realized_counts() -> None:
    population = _repaired_population(((1568, 100), (100, 20)))
    expected = {
        # These exact values seal the named CPU generator stream, flattened
        # binding order, and independently realized (not forced) layer counts.
        91501: (
            (20975, 278),
            21253,
            "0c7e4a0f0bcf30675c3e6f976090e8da22ce44c2df4f4ee9c1d9351be98f8964",
        ),
        91502: (
            (21142, 302),
            21444,
            "f5c419af14ab47dd2cde0312d3a150eed69608156ae0e64738b8ce398bb5bd65",
        ),
        91503: (
            (21081, 272),
            21353,
            "c1f8f55aedb1cf7b9d138aa8d4b7c8ed61c6ec46cba6f176edb60df6e3a7ca80",
        ),
        91504: (
            (21045, 284),
            21329,
            "1788be0489beb0ce50c4ec4d65294ba66570515992edd33f3ae1daa1e5436ac1",
        ),
        91505: (
            (21334, 250),
            21584,
            "9b112a6c04534e650f8d03d45fc52116cdc21227ac5ee2a5379bcd80e7f87fc7",
        ),
    }
    observed_hashes = set()
    for seed, expected_value in expected.items():
        first = sample_aihwkit_compatible_permanent_fault_overlay(
            population,
            mask_seed=seed,
        )
        replay = sample_aihwkit_compatible_permanent_fault_overlay(
            population,
            mask_seed=seed,
        )
        assert torch.equal(first.fault_mask, replay.fault_mask)
        assert torch.equal(first.raw_a_stuck, replay.raw_a_stuck)
        assert first.receipt == replay.receipt
        observed_hashes.add(first.receipt.mask_sha256)
        count_by_binding, total, mask_sha256 = expected_value
        assert first.receipt.realized_fault_count_by_binding == count_by_binding
        assert first.receipt.realized_fault_count == total
        assert first.receipt.mask_sha256 == mask_sha256
        assert abs(
            first.receipt.realized_fault_count / first.size
            - DEFAULT_CORRUPT_PROBABILITY
        ) < 0.003
    assert len(observed_hashes) == len(expected)


def test_overlay_collapses_selected_raw_a_support_and_preserves_healthy_cells() -> None:
    population = _repaired_population(((400, 100), (100, 20)))
    overlay = sample_aihwkit_compatible_permanent_fault_overlay(
        population,
        mask_seed=91501,
    )
    pulse = overlay.pulse_population
    mask = overlay.fault_mask
    healthy = ~mask
    selected_raw_a = overlay.raw_a_stuck[mask]
    selected_g = overlay.full_conductance_stuck[mask]

    assert mask.any() and healthy.any()
    assert torch.all(selected_raw_a >= -DEFAULT_CORRUPT_RAW_A_RANGE)
    assert torch.all(selected_raw_a <= DEFAULT_CORRUPT_RAW_A_RANGE)
    assert torch.all(selected_g >= 0.99)
    assert torch.all(selected_g <= 1.01)
    assert torch.all(selected_g >= 0.0)
    assert abs(float(selected_raw_a.mean().item())) < 2.5e-4
    assert abs(
        float(selected_raw_a.std(unbiased=False).item())
        - DEFAULT_CORRUPT_RAW_A_RANGE / math.sqrt(3.0)
    ) < 2.5e-4
    assert torch.equal(pulse.min_bound[mask], selected_raw_a)
    assert torch.equal(pulse.max_bound[mask], selected_raw_a)
    assert torch.count_nonzero(pulse.dwmin_up[mask]) == 0
    assert torch.count_nonzero(pulse.dwmin_down[mask]) == 0
    assert torch.equal(pulse.corrupt, mask)
    assert pulse.dw_min_std == 0.0
    assert pulse.write_noise_std == 0.0

    torch.testing.assert_close(pulse.min_bound[healthy], population.min_bound[healthy])
    torch.testing.assert_close(pulse.max_bound[healthy], population.max_bound[healthy])
    torch.testing.assert_close(pulse.dwmin_up[healthy], population.dwmin_up[healthy])
    torch.testing.assert_close(pulse.dwmin_down[healthy], population.dwmin_down[healthy])
    receipt = overlay.receipt.as_dict()
    assert receipt["coordinates"]["device_state"] == RAW_A_COORDINATE
    assert (
        receipt["coordinates"]["circuit_facing_conductance"]
        == FULL_CONDUCTANCE_COORDINATE
    )
    assert receipt["coordinates"]["negative_full_conductance_stuck_count"] == 0
    assert receipt["pulse_response_policy"] == {
        "heterogeneous_sampled_bounds_and_steps_retained": True,
        "source_dw_min_std": 0.2,
        "source_write_noise_std": 0.3,
        "overlay_dw_min_std": 0.0,
        "overlay_write_noise_std": 0.0,
        "cycle_to_cycle_noise_enabled": False,
        "apparent_write_noise_enabled": False,
        "only_pulse_selection_is_stochastic_during_recovery": True,
    }
    assert receipt["realized_fault_count"] == sum(
        receipt["realized_fault_count_by_binding"].values()
    )

    alternate_values = sample_aihwkit_compatible_permanent_fault_overlay(
        population,
        mask_seed=91501,
        raw_a_stuck_seed=314159,
    )
    assert torch.equal(alternate_values.fault_mask, overlay.fault_mask)
    assert not torch.equal(
        alternate_values.raw_a_stuck[mask],
        overlay.raw_a_stuck[mask],
    )


def test_support_intersection_truncates_or_fails_closed() -> None:
    truncated = _repaired_population(((64, 1),), minimum=-0.004, maximum=0.006)
    overlay = sample_aihwkit_compatible_permanent_fault_overlay(
        truncated,
        mask_seed=17,
        corrupt_probability=1.0,
    )
    stuck = overlay.raw_a_stuck
    assert torch.all(stuck >= -0.004)
    assert torch.all(stuck <= 0.006)
    assert overlay.receipt.device_truncated_intersection_count == overlay.size
    assert overlay.receipt.empty_support_intersection_count == 0

    disjoint = _repaired_population(((8, 1),), minimum=0.02, maximum=0.03)
    with pytest.raises(ValueError, match="empty intersection"):
        sample_aihwkit_compatible_permanent_fault_overlay(
            disjoint,
            mask_seed=17,
            corrupt_probability=1.0,
        )


def test_continuous_projection_reclamps_faults_in_raw_a_and_full_g() -> None:
    population = _repaired_population(((32, 1),))
    overlay = sample_aihwkit_compatible_permanent_fault_overlay(
        population,
        mask_seed=23,
        corrupt_probability=0.5,
    )
    mask = overlay.fault_mask
    assert mask.any() and (~mask).any()

    raw_a = torch.full((overlay.size,), 2.0, dtype=torch.float32)
    overlay.clamp_raw_a_(raw_a)
    assert torch.equal(raw_a[mask], overlay.raw_a_stuck[mask])
    assert torch.all(raw_a[~mask] == 0.8)

    conductance_g = torch.full((overlay.size,), -5.0, dtype=torch.float32)
    overlay.clamp_full_conductance_(conductance_g)
    assert torch.equal(
        conductance_g[mask],
        overlay.full_conductance_stuck[mask],
    )
    torch.testing.assert_close(
        conductance_g[~mask],
        torch.full_like(conductance_g[~mask], 0.2),
    )
    assert torch.all(conductance_g >= 0.0)


def test_write_only_port_suppresses_fault_attempts_and_healthy_cells_move() -> None:
    population = _repaired_population(((32, 1),))
    overlay = sample_aihwkit_compatible_permanent_fault_overlay(
        population,
        mask_seed=23,
        corrupt_probability=0.5,
    )
    plant = IbmReramRawActivePlant(
        overlay.pulse_population,
        seeds=tuple(range(1, overlay.size + 1)),
        device="cpu",
    )
    overlay.clamp_raw_a_(plant.persistent)
    plant.apparent.copy_(plant.persistent)
    before = plant.persistent.clone()
    port, ledger = overlay.make_write_port(plant.controller_port())

    assert port.size == overlay.size
    for forbidden in (
        "mask",
        "fault_mask",
        "bounds",
        "min_bound",
        "max_bound",
        "persistent",
        "population",
        "receipt",
        "success",
        "verify",
    ):
        assert not hasattr(port, forbidden)

    direction = torch.ones(overlay.size, dtype=torch.int8)
    count = torch.ones(overlay.size, dtype=torch.int64)
    port.apply_identical_pulses(direction, count)
    after_set = plant.persistent.clone()
    port.apply_identical_pulses(-direction, count)
    after_reset = plant.persistent.clone()

    mask = overlay.fault_mask
    healthy = ~mask
    assert torch.equal(after_set[mask], before[mask])
    assert torch.equal(after_reset[mask], before[mask])
    assert torch.any(after_set[healthy] != before[healthy])
    transition = overlay.transition_report(
        before,
        after_set,
        coordinate="raw_a",
    )
    assert transition["fault_moved_count"] == 0
    assert transition["fault_maximum_absolute_movement"] == 0.0
    assert transition["healthy_moved_count"] == int(healthy.sum().item())

    report = ledger.report()
    assert report["attempted_pulse_requests"] == 2 * overlay.size
    assert report["suppressed_fault_pulse_requests"] == 2 * int(mask.sum().item())
    assert report["forwarded_healthy_pulse_requests"] == 2 * int(healthy.sum().item())
    assert report["verify_reads"] == 0
