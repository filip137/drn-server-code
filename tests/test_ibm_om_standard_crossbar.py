from __future__ import annotations

from dataclasses import replace

import torch

from training.ibm_om_standard_crossbar import (
    IbmOmEffectiveCrossbarPlant,
    PulseAdam,
    apply_population_bound_policy,
    build_crossbar_layout,
    build_deterministic_effective_codebook,
    effective_state_to_logical_weights,
    map_logical_weights,
    project_to_nearest_effective_code,
    project_to_nearest_effective_code_device,
    standard_crossbar_logits,
)
from training.ibm_reram_hwa import IbmReramArrayPopulation


def _population(
    layout,
    *,
    minimum: float = -1.0,
    maximum: float = 1.0,
    reference: float = 0.0,
    noise: float = 0.0,
) -> IbmReramArrayPopulation:
    size = sum(tile.cells for tile in layout)
    return IbmReramArrayPopulation(
        assignment_seed=87004,
        corruption_policy="counterfactual_repaired",
        binding_keys=tuple(tile.key for tile in layout),
        binding_shapes=tuple(tile.shape for tile in layout),
        binding_sampling_seeds=tuple(range(11, 11 + len(layout))),
        donor_sampling_seeds=tuple(range(21, 21 + len(layout))),
        nominal_dw_min=0.1,
        dw_min_std=noise,
        write_noise_std=noise,
        max_bound=torch.full((size,), maximum),
        min_bound=torch.full((size,), minimum),
        dwmin_up=torch.full((size,), 0.1),
        dwmin_down=torch.full((size,), 0.1),
        reference=torch.full((size,), reference),
        corrupt=torch.zeros(size, dtype=torch.bool),
        published_corrupt=torch.zeros(size, dtype=torch.bool),
        fingerprint="crossbar-fixture",
        aihwkit_version="1.1.0",
    )


def _published_population(layout) -> IbmReramArrayPopulation:
    population = _population(layout)
    corrupt = torch.zeros(population.size, dtype=torch.bool)
    corrupt[0] = True
    minimum = population.min_bound.clone()
    maximum = population.max_bound.clone()
    upward = population.dwmin_up.clone()
    downward = population.dwmin_down.clone()
    minimum[corrupt] = 0.05
    maximum[corrupt] = 0.05
    upward[corrupt] = 0.0
    downward[corrupt] = 0.0
    return replace(
        population,
        corruption_policy="published",
        min_bound=minimum,
        max_bound=maximum,
        dwmin_up=upward,
        dwmin_down=downward,
        corrupt=corrupt,
        published_corrupt=corrupt.clone(),
        fingerprint="published-crossbar-fixture",
    )


def test_layout_matches_aihwkit_balanced_512_input_tiling() -> None:
    layout = build_crossbar_layout((784, 50, 10), maximum_input_size=512)

    assert [(tile.key, tile.shape) for tile in layout] == [
        ("crossbar.layer0.tile0", (392, 50)),
        ("crossbar.layer0.tile1", (392, 50)),
        ("crossbar.layer1.tile0", (50, 10)),
    ]
    assert sum(tile.cells for tile in layout) == 39_700


def test_mapping_round_trip_and_forward_match_digital_relu() -> None:
    generator = torch.Generator().manual_seed(7)
    layout = build_crossbar_layout((8, 3, 2), maximum_input_size=5)
    weights = (
        torch.randn((8, 3), generator=generator),
        torch.randn((3, 2), generator=generator),
    )
    inputs = torch.randn((4, 8), generator=generator)

    state, scales = map_logical_weights(
        weights,
        layout,
        weight_scaling_omega=(0.8, 0.9),
    )
    reconstructed = effective_state_to_logical_weights(
        state,
        layout,
        digital_scales=scales,
    )

    torch.testing.assert_close(reconstructed[0], weights[0])
    torch.testing.assert_close(reconstructed[1], weights[1])
    torch.testing.assert_close(
        standard_crossbar_logits(inputs, state, layout, digital_scales=scales),
        torch.relu(inputs @ weights[0]) @ weights[1],
    )


def test_bound_policy_retains_identity_and_matches_drn_winsorization() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout)
    population = replace(
        population,
        min_bound=torch.tensor([-1.4, -0.8, -1.2, -0.5, -2.0, -0.1]),
        max_bound=torch.tensor([1.3, 0.7, 1.2, 0.5, 2.0, 0.1]),
        reference=torch.tensor([0.1, -0.1, 0.0, 0.0, 0.2, 0.0]),
    )

    native, native_report = apply_population_bound_policy(
        population,
        policy="native_sampled_bounds",
    )
    matched, matched_report = apply_population_bound_policy(
        population,
        policy="winsorize_raw_active_a_to_unit_interval",
    )

    assert native.fingerprint == population.fingerprint
    assert torch.equal(native.min_bound, population.min_bound)
    assert torch.equal(native.max_bound, population.max_bound)
    assert native_report["any_bound_changed"] == 0
    torch.testing.assert_close(
        matched.min_bound,
        torch.maximum(population.min_bound, torch.full_like(population.min_bound, -1.0)),
    )
    torch.testing.assert_close(
        matched.max_bound,
        torch.minimum(population.max_bound, torch.full_like(population.max_bound, 1.0)),
    )
    assert matched.binding_sampling_seeds == population.binding_sampling_seeds
    assert torch.equal(matched.reference, population.reference)
    assert matched_report["identity_resampled"] is False
    assert matched_report["any_bound_changed"] == 3


def test_deterministic_codebook_uses_effective_q_and_low_pulse_ties() -> None:
    layout = build_crossbar_layout((1, 1, 1), maximum_input_size=1)
    population = _population(layout, minimum=-1.0, maximum=1.0, reference=0.2)
    codebook = build_deterministic_effective_codebook(population, maximum_pulses=2)

    torch.testing.assert_close(codebook.values[0], torch.full((2,), -1.2))
    torch.testing.assert_close(codebook.values[1], torch.full((2,), -1.0))
    midpoint = (codebook.values[0] + codebook.values[1]) / 2.0
    realized, indices = project_to_nearest_effective_code(codebook, midpoint)

    assert torch.equal(indices, torch.zeros(2, dtype=torch.int64))
    torch.testing.assert_close(realized, codebook.values[0])

    device_realized, device_indices = project_to_nearest_effective_code_device(
        codebook.values,
        midpoint,
    )
    assert torch.equal(device_indices, indices)
    torch.testing.assert_close(device_realized, realized)
    cached_realized, cached_indices = project_to_nearest_effective_code_device(
        codebook.values.transpose(0, 1).contiguous(),
        midpoint,
        rowwise=True,
        validate=False,
    )
    assert torch.equal(cached_indices, indices)
    torch.testing.assert_close(cached_realized, realized)


def test_published_corrupt_cell_has_one_immutable_code_and_persistent_state() -> None:
    layout = build_crossbar_layout((1, 1, 1), maximum_input_size=1)
    population = _published_population(layout)
    codebook = build_deterministic_effective_codebook(
        population,
        maximum_pulses=4,
    )

    assert codebook.effective_level_counts.tolist() == [1, 5]
    torch.testing.assert_close(
        codebook.values[:, 0],
        codebook.values[0, 0].expand(5),
    )
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(17),
        device="cpu",
    )
    initial_corrupt = plant.persistent[0].clone()
    for _ in range(3):
        plant.pulse(torch.ones(population.size, dtype=torch.int8))

    torch.testing.assert_close(plant.persistent[0], initial_corrupt)
    assert plant.persistent[1] > population.logical_min[1]
    statistics = plant.pulse_statistics()
    assert statistics["final_corrupt_cells"] == 1
    assert statistics["published_corrupt_cells"] == 1
    assert statistics["commanded_final_corrupt_cells"] == 1
    assert statistics["pulses_to_final_corrupt_cells"] == 3
    assert statistics["persistent_moved_from_reset_cells"] == 1


def test_plant_checkpoint_replays_cycle_and_write_noise_exactly() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout, noise=0.2)
    generator = torch.Generator().manual_seed(123)
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=generator,
        device="cpu",
    )
    plant.pulse(torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.int8))
    checkpoint = plant.state_dict()
    commands = torch.tensor([-1, 1, 0, 1, 0, -1], dtype=torch.int8)
    plant.pulse(commands)
    expected = plant.state_dict()

    replay_generator = torch.Generator().manual_seed(999)
    replay = IbmOmEffectiveCrossbarPlant(
        population,
        generator=replay_generator,
        device="cpu",
    )
    replay.load_state_dict(checkpoint)
    replay.pulse(commands)
    actual = replay.state_dict()

    for name in (
        "persistent",
        "apparent",
        "upward_pulses",
        "downward_pulses",
        "generator_state",
    ):
        assert torch.equal(actual[name], expected[name])


def test_pulse_adam_layer_scope_and_recovery_cap_are_physical() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout)
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(3),
        device="cpu",
    )
    optimizer = PulseAdam(
        size=population.size,
        layout=layout,
        learning_rates=(1.0, 1.0),
        betas=(0.0, 0.0),
        epsilon=1e-8,
        layer_scope="input_only",
        nominal_dw_min=0.1,
        pulse_cap_per_cell=2,
        generator=torch.Generator().manual_seed(4),
        device="cpu",
    )

    for _ in range(3):
        optimizer.step(-torch.ones(population.size), plant)
    report = optimizer.report()

    assert report["applied_pulses_by_layer"] == [8, 0]
    assert report["applied_pulses"] == 8
    assert report["cells_at_cap"] == 4
    assert report["maximum_pulses_per_cell"] == 2
    assert report["blocked_at_cap"] == 4
    assert report["enabled_cells"] == 4
