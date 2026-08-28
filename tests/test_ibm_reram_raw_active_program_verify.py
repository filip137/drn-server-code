from dataclasses import replace

import pytest
import torch

from training.ibm_reram_hwa import IbmReramArrayPopulation
from training.ibm_reram_program_verify import (
    ControllerSettings,
    IbmReramRawActivePlant,
    run_program_verify,
)
from training.ibm_reram_raw_active_program_verify import (
    RAW_ACTIVE_COORDINATE,
    array_population_as_pulse_population,
    classify_persistent_uniform_codes,
    matched_trajectory_seeds,
    project_raw_active_unit_to_full_conductance,
    raw_active_x_to_full_conductance,
    raw_active_unit_to_full_conductance,
    required_cuda_random_draws,
    run_raw_active_program_verify,
)


def _array_population(
    *,
    reference: tuple[float, float] = (0.25, -0.35),
    write_noise_std: float = 0.0,
) -> IbmReramArrayPopulation:
    return IbmReramArrayPopulation(
        assignment_seed=87001,
        corruption_policy="counterfactual_repaired",
        binding_keys=("base.dense_weight.0",),
        binding_shapes=((1, 2),),
        binding_sampling_seeds=(12345,),
        donor_sampling_seeds=(54321,),
        nominal_dw_min=0.1,
        dw_min_std=0.0,
        write_noise_std=write_noise_std,
        max_bound=torch.tensor([1.0, 0.8], dtype=torch.float32),
        min_bound=torch.tensor([-1.0, -0.8], dtype=torch.float32),
        dwmin_up=torch.tensor([0.2, 0.2], dtype=torch.float32),
        dwmin_down=torch.tensor([0.2, 0.2], dtype=torch.float32),
        reference=torch.tensor(reference, dtype=torch.float32),
        corrupt=torch.tensor([False, False]),
        published_corrupt=torch.tensor([False, False]),
        fingerprint="raw-active-test-population",
        aihwkit_version="1.1.0",
    )


def test_raw_active_plant_ignores_reference_exactly() -> None:
    first_array = _array_population(reference=(0.25, -0.35), write_noise_std=0.4)
    second_array = replace(
        first_array,
        reference=torch.tensor([-0.7, 0.6], dtype=torch.float32),
        fingerprint="raw-active-test-population-reference-counterfactual",
    )
    targets = torch.tensor([0.65, 0.55], dtype=torch.float32)

    first = run_raw_active_program_verify(
        first_array,
        targets_unit=targets,
        endpoint_seed=89001,
        tolerance_unit=0.025,
        maximum_program_pulses=32,
        device="cpu",
    )
    second = run_raw_active_program_verify(
        second_array,
        targets_unit=targets,
        endpoint_seed=89001,
        tolerance_unit=0.025,
        maximum_program_pulses=32,
        device="cpu",
    )

    assert first.coordinate == RAW_ACTIVE_COORDINATE
    assert torch.equal(first.initial_persistent_unit, second.initial_persistent_unit)
    assert torch.equal(first.initial_apparent_unit, second.initial_apparent_unit)
    assert torch.equal(first.apparent_endpoint_unit, second.apparent_endpoint_unit)
    assert torch.equal(first.persistent_endpoint_unit, second.persistent_endpoint_unit)
    assert torch.equal(first.programming.total_pulses, second.programming.total_pulses)
    assert first.inference_read_noise_enabled is False


def test_apparent_acceptance_does_not_imply_persistent_correctness() -> None:
    pulse_population = array_population_as_pulse_population(_array_population())
    plant = IbmReramRawActivePlant(pulse_population, seeds=(11, 12))
    plant.initialize_at_sampled_lower()
    # The capability-limited controller sees apparent raw x=0.5, while the
    # persistent raw state remains at each sampled lower bound.
    plant.apparent.zero_()
    result = run_program_verify(
        plant.controller_port(),
        targets=torch.tensor([0.5, 0.5]),
        tolerance=1e-3,
        maximum_pulses=8,
        settings=ControllerSettings(kind="one_pulse"),
    )
    persistent = (plant.persistent + 1.0) / 2.0

    assert torch.equal(result.accepted, torch.tensor([True, True]))
    assert torch.equal(result.total_pulses, torch.tensor([0, 0]))
    assert not bool(torch.any(torch.abs(persistent - 0.5) <= 1e-3))


def test_raw_active_continuation_is_exact_and_coordinate_tagged() -> None:
    population = array_population_as_pulse_population(
        _array_population(write_noise_std=0.4)
    )
    first = IbmReramRawActivePlant(population, seeds=(21, 22))
    first.initialize_at_sampled_lower()
    first.pulse(torch.tensor([1, 1], dtype=torch.int8))
    state = first.state_dict()

    restored = IbmReramRawActivePlant(population, seeds=(999, 1000))
    restored.load_state_dict(state)
    direction = torch.tensor([1, -1], dtype=torch.int8)
    first.pulse(direction)
    restored.pulse(direction)

    assert state["state_coordinate"] == "native_raw_active_a"
    assert torch.equal(first.persistent, restored.persistent)
    assert torch.equal(first.apparent, restored.apparent)


def test_support_preflight_and_full_conductance_conversion() -> None:
    population = _array_population()
    with pytest.raises(ValueError, match="exact persistent support"):
        run_raw_active_program_verify(
            population,
            targets_unit=torch.tensor([1.1, 0.5]),
            endpoint_seed=89001,
            tolerance_unit=0.025,
            maximum_program_pulses=16,
            device="cpu",
        )

    endpoint = torch.tensor([0.25, 0.75])
    full = raw_active_unit_to_full_conductance(
        endpoint,
        conductance_min=10.0,
        conductance_max=110.0,
    )
    assert torch.equal(full, torch.tensor([35.0, 85.0]))

    with pytest.raises(ValueError, match="explicit public-coordinate projection"):
        raw_active_unit_to_full_conductance(
            torch.tensor([-0.1, 1.1]),
            conductance_min=10.0,
            conductance_max=110.0,
        )


def test_public_projection_preserves_raw_endpoint_and_records_masks() -> None:
    raw = torch.tensor([-0.2, 0.0, 0.4, 1.0, 1.3])
    projected = project_raw_active_unit_to_full_conductance(
        raw,
        conductance_min=10.0,
        conductance_max=110.0,
    )

    assert torch.equal(projected.raw_endpoint_unit, raw)
    assert torch.equal(
        projected.applied_endpoint_unit,
        torch.tensor([0.0, 0.0, 0.4, 1.0, 1.0]),
    )
    assert torch.equal(
        projected.below_public_minimum,
        torch.tensor([True, False, False, False, False]),
    )
    assert torch.equal(
        projected.above_public_maximum,
        torch.tensor([False, False, False, False, True]),
    )
    assert torch.equal(
        projected.full_conductance,
        torch.tensor([10.0, 10.0, 50.0, 110.0, 110.0]),
    )
    assert projected.projected_count == 2
    report = projected.report()
    assert report["native_controller_state_changed"] is False
    assert report["persistent_continuation_state_changed"] is False
    assert report["below_public_minimum"] == 1
    assert report["above_public_maximum"] == 1


def test_affine_raw_x_handoff_is_unclipped_and_strictly_positive() -> None:
    origin = -1.3759248719940185
    slope = 1.10e-4
    raw_maximum = 1.8474750518798828
    ceiling = slope * (raw_maximum - origin)
    raw = torch.tensor(
        [-1.3759238719940186, 0.0, 1.0, raw_maximum],
        dtype=torch.float64,
    )

    full = raw_active_x_to_full_conductance(
        raw,
        raw_x_origin=origin,
        conductance_per_raw_x=slope,
        conductance_ceiling=ceiling,
    )

    assert torch.equal(raw, torch.tensor(raw.tolist(), dtype=torch.float64))
    assert full[0].item() == pytest.approx(1.10e-10, rel=1e-9)
    assert full[2].item() > 1.10e-4
    assert full[-1].item() == pytest.approx(ceiling, rel=1e-12)
    assert bool(torch.all(full > 0.0))


def test_affine_raw_x_handoff_fails_instead_of_clipping() -> None:
    origin = -1.5
    slope = 2.0
    ceiling = 7.0

    with pytest.raises(ValueError, match="strictly positive"):
        raw_active_x_to_full_conductance(
            torch.tensor([origin], dtype=torch.float64),
            raw_x_origin=origin,
            conductance_per_raw_x=slope,
            conductance_ceiling=ceiling,
        )
    with pytest.raises(ValueError, match="ceiling"):
        raw_active_x_to_full_conductance(
            torch.tensor([2.1], dtype=torch.float64),
            raw_x_origin=origin,
            conductance_per_raw_x=slope,
            conductance_ceiling=ceiling,
        )


def test_seed_matching_and_cuda_draw_budget_exclude_mapping_arm() -> None:
    population = _array_population(write_noise_std=0.4)
    first = matched_trajectory_seeds(population, endpoint_seed=89001)
    second = matched_trajectory_seeds(population, endpoint_seed=89001)
    other = matched_trajectory_seeds(population, endpoint_seed=89002)

    assert first == second
    assert first != other
    assert len(set(first)) == population.size
    assert required_cuda_random_draws(
        population, maximum_program_pulses=128
    ) == 257


def test_persistent_nearest_code_is_classified_separately() -> None:
    classified = classify_persistent_uniform_codes(
        torch.tensor([0.09, 0.31, 0.64, 0.95]),
        baseline_unit=torch.tensor([0.1, 0.1, 0.1, 0.1]),
        spacing_unit=0.2,
        requested_index=torch.tensor([0, 1, 2, 3]),
        maximum_index=torch.tensor([3, 3, 3, 3]),
    )

    assert torch.equal(
        classified.nearest_persistent_index,
        torch.tensor([0, 1, 3, 3]),
    )
    assert torch.equal(
        classified.requested_code_correct,
        torch.tensor([True, True, False, True]),
    )
