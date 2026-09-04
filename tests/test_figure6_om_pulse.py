from __future__ import annotations

import torch

from experiments.mnist_relu_drn.figure6_om_pulse import (
    OM_CYCLE_NOISE_STD,
    OM_NOMINAL_DW_MIN_RAW_A,
    OM_WRITE_NOISE_STD,
    Figure6OmPulsePlant,
    PersistentFigure6OmPulseAdam,
    corruption_constrained_progress,
    program_progress_with_verify,
    validate_paired_om_populations,
)
from experiments.mnist_relu_drn.hfo2_figure6_drn import EndpointField, flatten_physical
from training.ibm_reram_hwa import IbmReramArrayPopulation


SHAPES = ((4, 4), (2, 2))


def _field() -> EndpointField:
    return EndpointField(
        reset=(torch.full(SHAPES[0], 0.05), torch.full(SHAPES[1], 0.1)),
        set=(torch.full(SHAPES[0], 2.0), torch.full(SHAPES[1], 2.1)),
        shapes=SHAPES,
        layouts=("halves", "paired"),
        population_report={"test": True},
    )


def _population(policy: str) -> IbmReramArrayPopulation:
    size = sum(value.numel() for value in _field().reset)
    published_corrupt = torch.zeros(size, dtype=torch.bool)
    published_corrupt[[1, 10]] = True
    repaired = policy == "counterfactual_repaired"
    corrupt = torch.zeros_like(published_corrupt) if repaired else published_corrupt.clone()
    lower = torch.full((size,), -1.1, dtype=torch.float32)
    upper = torch.full((size,), 1.1, dtype=torch.float32)
    up = torch.full((size,), 0.1, dtype=torch.float32)
    down = torch.full((size,), 0.1, dtype=torch.float32)
    reference = torch.zeros(size, dtype=torch.float32)
    if repaired:
        lower[published_corrupt] = -0.9
        upper[published_corrupt] = 0.9
        up[published_corrupt] = 0.12
        down[published_corrupt] = 0.12
        reference[published_corrupt] = 0.02
    else:
        lower[published_corrupt] = 0.0
        upper[published_corrupt] = 0.0
        up[published_corrupt] = 0.0
        down[published_corrupt] = 0.0
    return IbmReramArrayPopulation(
        assignment_seed=123,
        corruption_policy=policy,
        binding_keys=("w1", "w2"),
        binding_shapes=SHAPES,
        binding_sampling_seeds=(11, 12),
        donor_sampling_seeds=(21, 22),
        nominal_dw_min=OM_NOMINAL_DW_MIN_RAW_A,
        dw_min_std=OM_CYCLE_NOISE_STD,
        write_noise_std=OM_WRITE_NOISE_STD,
        max_bound=upper,
        min_bound=lower,
        dwmin_up=up,
        dwmin_down=down,
        reference=reference,
        corrupt=corrupt,
        published_corrupt=published_corrupt,
        fingerprint=f"toy-{policy}",
        aihwkit_version="1.1.0",
    )


def _zeros() -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(torch.zeros(shape, dtype=torch.float32) for shape in SHAPES)  # type: ignore[return-value]


def test_pairing_repairs_only_published_corrupt_coordinates() -> None:
    report = validate_paired_om_populations(
        _population("published"),
        _population("counterfactual_repaired"),
    )
    assert report["published_corrupt_cells"] == 2
    assert report["healthy_cells_bitwise_identical"] is True
    assert report["published_stuck_progress_minimum"] == 0.5


def test_published_corrupt_cells_keep_native_stuck_progress_under_pulses() -> None:
    population = _population("published")
    plant = Figure6OmPulsePlant(
        _field(),
        _zeros(),
        population,
        pulse_noise_seed=99,
    )
    corrupt = population.corrupt
    before = plant.raw_a.clone()
    changed, capped = plant.pulse(torch.ones(plant.size), pulse_cap=3)
    assert capped == 0
    assert changed == plant.size - int(corrupt.sum().item())
    assert torch.equal(plant.raw_a[corrupt], before[corrupt])
    assert torch.equal(plant.pulse_count, torch.ones_like(plant.pulse_count))


def test_program_verify_preaccepts_only_healthy_exact_reset_targets() -> None:
    field = _field()
    published = _population("published")
    plant, result = program_progress_with_verify(
        field,
        _zeros(),
        published,
        pulse_noise_seed=101,
        tolerance_progress=0.01,
        maximum_pulses=3,
    )
    assert bool(torch.all(result.total_pulses[~published.corrupt] == 0))
    assert bool(torch.all(result.accepted[~published.corrupt]))
    assert bool(torch.all(result.total_pulses[published.corrupt] == 3))
    assert bool(torch.all(result.budget_exhausted[published.corrupt]))
    assert torch.equal(plant.raw_a[published.corrupt], torch.zeros(2))

    repaired = _population("counterfactual_repaired")
    _, repaired_result = program_progress_with_verify(
        field,
        _zeros(),
        repaired,
        pulse_noise_seed=101,
        tolerance_progress=0.01,
        maximum_pulses=3,
    )
    assert bool(torch.all(repaired_result.accepted))
    assert int(repaired_result.total_pulses.sum().item()) == 0


def test_corruption_constrained_target_changes_only_corrupt_cells() -> None:
    field = _field()
    population = _population("published")
    target = tuple(torch.full(shape, 0.25) for shape in SHAPES)
    constrained = flatten_physical(
        corruption_constrained_progress(target, field, population)
    )
    corrupt = population.corrupt
    assert torch.equal(constrained[~corrupt], torch.full_like(constrained[~corrupt], 0.25))
    assert torch.equal(constrained[corrupt], torch.full_like(constrained[corrupt], 0.5))


def test_clone_starts_a_fresh_phase_without_moving_persistent_state() -> None:
    population = _population("published")
    plant = Figure6OmPulsePlant(
        _field(),
        _zeros(),
        population,
        pulse_noise_seed=7,
    )
    plant.pulse(torch.ones(plant.size), pulse_cap=4)
    clone = plant.clone_for_new_pulse_phase(pulse_noise_seed=8)
    assert torch.equal(clone.raw_a, plant.raw_a)
    assert torch.equal(clone.apparent_raw_a, plant.apparent_raw_a)
    assert int(clone.pulse_count.sum().item()) == 0
    assert torch.equal(clone.raw_a[population.corrupt], plant.raw_a[population.corrupt])


def test_pulse_adam_records_probability_clipping_events() -> None:
    plant = Figure6OmPulsePlant(
        _field(),
        _zeros(),
        _population("counterfactual_repaired"),
        pulse_noise_seed=77,
    )
    optimizer = PersistentFigure6OmPulseAdam(
        plant,
        learning_rate_progress=1.0,
        pulse_cap=2,
        pulse_selection_seed=78,
    )
    optimizer.step(tuple(torch.ones(shape) for shape in SHAPES))
    state = optimizer.state_dict()
    assert state["total_probability_clipped_cells"] == plant.size


def test_post_pv_reset_stuck_fault_changes_only_mask_and_refreshes_on_attempt() -> None:
    population = _population("counterfactual_repaired")
    plant = Figure6OmPulsePlant(
        _field(),
        tuple(torch.full(shape, 0.4) for shape in SHAPES),
        population,
        pulse_noise_seed=501,
    )
    plant.sample_apparent()
    persistent_before = plant.raw_a.clone()
    apparent_before = plant.apparent_raw_a.clone()
    receipt = plant.inject_post_pv_reset_stuck_faults(
        population.published_corrupt,
        observation_seed=502,
    )
    mask = population.published_corrupt

    assert receipt["intervention"] == "post_pv_reset_stuck_fault"
    assert torch.equal(plant.raw_a[mask], torch.full_like(plant.raw_a[mask], -1.0))
    assert torch.equal(plant.raw_a[~mask], persistent_before[~mask])
    assert torch.equal(plant.apparent_raw_a[~mask], apparent_before[~mask])
    fault_apparent = plant.apparent_raw_a[mask].clone()

    direction = torch.zeros(plant.size, dtype=torch.int8)
    direction[mask] = 1
    changed, capped = plant.pulse(direction, pulse_cap=4)
    assert changed == 0
    assert capped == 0
    assert torch.equal(plant.raw_a[mask], torch.full_like(plant.raw_a[mask], -1.0))
    assert not torch.equal(plant.apparent_raw_a[mask], fault_apparent)
    assert torch.equal(plant.pulse_count[mask], torch.ones_like(plant.pulse_count[mask]))


def test_post_pv_fault_clone_and_state_round_trip_are_exact() -> None:
    population = _population("counterfactual_repaired")
    plant = Figure6OmPulsePlant(
        _field(),
        tuple(torch.full(shape, 0.6) for shape in SHAPES),
        population,
        pulse_noise_seed=601,
    )
    plant.inject_post_pv_reset_stuck_faults(
        population.published_corrupt,
        observation_seed=602,
    )
    clone = plant.clone_for_new_pulse_phase(pulse_noise_seed=603)
    assert torch.equal(clone.raw_a, plant.raw_a)
    assert torch.equal(clone.apparent_raw_a, plant.apparent_raw_a)
    assert torch.equal(clone.corrupt, plant.corrupt)
    assert clone.fault_intervention == "post_pv_reset_stuck_fault"

    state = clone.state_dict()
    restored = Figure6OmPulsePlant(
        _field(),
        _zeros(),
        population,
        pulse_noise_seed=603,
    )
    restored.load_state_dict(state)
    assert torch.equal(restored.raw_a, clone.raw_a)
    assert torch.equal(restored.apparent_raw_a, clone.apparent_raw_a)
    assert torch.equal(restored.corrupt, clone.corrupt)
    assert restored.fault_observation_seed == 602
