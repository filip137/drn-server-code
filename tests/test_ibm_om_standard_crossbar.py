from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from training.ibm_om_standard_crossbar import (
    IbmOmEffectiveCrossbarPlant,
    PulseAdam,
    PulseSGD,
    apply_population_bound_policy,
    build_crossbar_layout,
    build_deterministic_effective_codebook,
    effective_state_to_logical_weights,
    flatten_logical_crossbar_matrices,
    local_star_crossbar_step,
    map_logical_weights,
    project_to_nearest_effective_code,
    project_to_nearest_effective_code_device,
    standard_crossbar_logits,
    standard_crossbar_forward_states,
    tensor_sha256,
)
from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    om_array_population_fingerprint,
)


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


def _published_population(
    layout,
    *,
    noise: float = 0.0,
) -> IbmReramArrayPopulation:
    population = _population(layout, noise=noise)
    corrupt = torch.zeros(population.size, dtype=torch.bool)
    corrupt[0] = True
    minimum = population.min_bound.clone()
    maximum = population.max_bound.clone()
    upward = population.dwmin_up.clone()
    downward = population.dwmin_down.clone()
    minimum[corrupt] = 0.005
    maximum[corrupt] = 0.005
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
    states = standard_crossbar_forward_states(
        inputs, state, layout, digital_scales=scales
    )
    torch.testing.assert_close(states.hidden_preactivation, inputs @ weights[0])
    torch.testing.assert_close(states.hidden_post_relu, torch.relu(inputs @ weights[0]))
    torch.testing.assert_close(states.output_logits, torch.relu(inputs @ weights[0]) @ weights[1])


def test_local_star_gradients_match_two_stopped_local_objectives() -> None:
    generator = torch.Generator().manual_seed(91)
    layout = build_crossbar_layout((3, 2, 2), maximum_input_size=2)
    logical = (
        torch.randn((3, 2), generator=generator),
        torch.randn((2, 2), generator=generator),
    )
    state, scales = map_logical_weights(
        logical, layout, weight_scaling_omega=(0.8, 0.9)
    )
    inputs = torch.randn((4, 3), generator=generator)
    labels = torch.tensor([0, 1, 0, 1])
    hidden_targets = torch.randn((2, 2), generator=generator)
    output_targets = torch.randn((2, 2), generator=generator)
    step = local_star_crossbar_step(
        inputs,
        labels,
        state,
        layout,
        digital_scales=scales,
        hidden_targets=hidden_targets,
        output_targets=output_targets,
        hidden_gain=0.3,
        output_gain=0.7,
    )

    weight_0 = logical[0].detach().clone().requires_grad_(True)
    hidden = torch.relu(inputs @ weight_0)
    hidden_loss = 0.5 * 0.3 * (hidden - hidden_targets[labels]).square().sum(dim=1).mean()
    gradient_0 = torch.autograd.grad(hidden_loss, weight_0)[0]
    weight_1 = logical[1].detach().clone().requires_grad_(True)
    output = hidden.detach() @ weight_1
    output_loss = 0.5 * 0.7 * (output - output_targets[labels]).square().sum(dim=1).mean()
    gradient_1 = torch.autograd.grad(output_loss, weight_1)[0]
    expected = flatten_logical_crossbar_matrices(
        (scales[0] * gradient_0, scales[1] * gradient_1), layout
    )

    torch.testing.assert_close(step.gradient_q, expected)
    slices = (
        slice(0, logical[0].numel()),
        slice(logical[0].numel(), state.numel()),
    )
    changed_output_state = state.clone()
    changed_output_state[slices[1]] *= -3.0
    changed = local_star_crossbar_step(
        inputs,
        labels,
        changed_output_state,
        layout,
        digital_scales=scales,
        hidden_targets=hidden_targets,
        output_targets=output_targets,
        hidden_gain=0.3,
        output_gain=0.7,
    )
    torch.testing.assert_close(step.gradient_q[slices[0]], changed.gradient_q[slices[0]])


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
    assert matched.fingerprint == om_array_population_fingerprint(
        assignment_seed=matched.assignment_seed,
        corruption_policy=matched.corruption_policy,
        keys=matched.binding_keys,
        shapes=matched.binding_shapes,
        binding_sampling_seeds=matched.binding_sampling_seeds,
        donor_sampling_seeds=matched.donor_sampling_seeds,
        scalar_parameters={
            "aihwkit_version": matched.aihwkit_version,
            "nominal_dw_min": matched.nominal_dw_min,
            "dw_min_std": matched.dw_min_std,
            "write_noise_std": matched.write_noise_std,
        },
        tensors=matched.tensor_state(),
    )
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


def test_aihwkit_corrupt_transition_keeps_persistent_stuck_but_resamples_apparent() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    published = _published_population(layout, noise=0.2)
    population = replace(
        _population(layout, noise=0.2),
        published_corrupt=published.corrupt.clone(),
    )
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(1234),
        device="cpu",
    )
    plant.pulse(torch.ones(population.size, dtype=torch.int8))
    pre_fault = plant.state_dict()
    healthy_persistent = plant.persistent.clone()
    mask = published.corrupt
    stuck = published.logical_min
    report = plant.apply_stuck_at_fault_transition(
        mask=mask,
        stuck_persistent_q=stuck,
        transition_id="fault-1",
        source_population_fingerprint=published.fingerprint,
    )
    post = plant.state_dict()
    post_fault_apparent = plant.apparent[mask].clone()
    plant.pulse(torch.ones(population.size, dtype=torch.int8))

    torch.testing.assert_close(plant.persistent[mask], stuck[mask])
    assert not torch.equal(plant.apparent[mask], stuck[mask])
    assert not torch.equal(plant.apparent[mask], post_fault_apparent)
    assert torch.any(plant.persistent[~mask] != healthy_persistent[~mask])
    assert report["faulted_cells"] == 1
    assert report["non_fault_persistent_unchanged"] is True
    assert report["non_fault_apparent_unchanged"] is True
    assert report["persistent_fault_state"] == (
        "sampled_collapsed_bound_with_zero_pulse_increments"
    )
    assert report["apparent_fault_state"] == (
        "write_noise_resampled_at_transition_and_each_write_attempt"
    )
    assert plant.pulse_statistics()["pulses_to_post_deployment_fault_cells"] == 1
    with pytest.raises(RuntimeError, match="exactly one"):
        plant.apply_stuck_at_fault_transition(
            mask=mask,
            stuck_persistent_q=stuck,
            transition_id="fault-2",
            source_population_fingerprint=published.fingerprint,
        )

    replay = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(99),
        device="cpu",
    )
    with pytest.raises(ValueError, match="published companion population"):
        replay.load_state_dict(post)
    replay.load_state_dict(
        post,
        expected_fault_source_population=published,
        expected_pre_fault_state=pre_fault,
    )
    replay.pulse(torch.ones(population.size, dtype=torch.int8))
    torch.testing.assert_close(replay.persistent, plant.persistent)
    torch.testing.assert_close(replay.apparent, plant.apparent)


def test_faulted_plant_checkpoint_rejects_corrupt_transition_and_state() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    published = _published_population(layout, noise=0.2)
    population = replace(
        _population(layout, noise=0.2),
        published_corrupt=published.corrupt.clone(),
    )
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(1234),
        device="cpu",
    )
    plant.pulse(torch.ones(population.size, dtype=torch.int8))
    pre_fault = plant.state_dict()
    mask = published.corrupt
    stuck = published.logical_min
    plant.apply_stuck_at_fault_transition(
        mask=mask,
        stuck_persistent_q=stuck,
        transition_id="fault-1",
        source_population_fingerprint=published.fingerprint,
    )
    checkpoint = plant.state_dict()
    corrupted = []

    bad_transition = deepcopy(checkpoint)
    bad_transition["fault_transition"]["fault_mask_sha256"] = "0" * 64
    corrupted.append(bad_transition)

    missing_transition_field = deepcopy(checkpoint)
    del missing_transition_field["fault_transition"]["fault_fraction"]
    corrupted.append(missing_transition_field)

    boolean_transition_version = deepcopy(checkpoint)
    boolean_transition_version["fault_transition"]["schema_version"] = True
    corrupted.append(boolean_transition_version)

    floating_plant_version = deepcopy(checkpoint)
    floating_plant_version["schema_version"] = 2.0
    corrupted.append(floating_plant_version)

    nonfinite = deepcopy(checkpoint)
    nonfinite["persistent"][1] = torch.nan
    corrupted.append(nonfinite)

    negative_pulses = deepcopy(checkpoint)
    negative_pulses["upward_pulses"][1] = -1
    corrupted.append(negative_pulses)

    impossible_baseline = deepcopy(checkpoint)
    impossible_baseline["post_deployment_fault_pulse_baseline"].fill_(10**9)
    corrupted.append(impossible_baseline)

    coherent_wrong_stuck = deepcopy(checkpoint)
    coherent_wrong_stuck["post_deployment_stuck_persistent_q"][mask] += 100.0
    coherent_wrong_stuck["persistent"][mask] += 100.0
    coherent_wrong_stuck["fault_transition"]["stuck_persistent_q_sha256"] = (
        tensor_sha256(
            coherent_wrong_stuck["post_deployment_stuck_persistent_q"][mask]
        )
    )
    coherent_wrong_stuck["fault_transition"]["post_fault_persistent_sha256"] = (
        tensor_sha256(coherent_wrong_stuck["persistent"])
    )
    corrupted.append(coherent_wrong_stuck)

    wrong_pre_hash = deepcopy(checkpoint)
    wrong_pre_hash["fault_transition"]["pre_fault_persistent_sha256"] = "0" * 64
    corrupted.append(wrong_pre_hash)

    wrong_source = deepcopy(checkpoint)
    wrong_source["fault_transition"]["source_population_fingerprint"] = (
        "forged-published-population"
    )
    corrupted.append(wrong_source)

    for state in corrupted:
        replay = IbmOmEffectiveCrossbarPlant(
            population,
            generator=torch.Generator().manual_seed(99),
            device="cpu",
        )
        with pytest.raises(ValueError):
            replay.load_state_dict(
                state,
                expected_fault_source_population=published,
                expected_pre_fault_state=pre_fault,
            )

    outside_aihwkit_range = replace(
        published,
        min_bound=published.min_bound.clone(),
        max_bound=published.max_bound.clone(),
    )
    outside_aihwkit_range.min_bound[mask] = 0.05
    outside_aihwkit_range.max_bound[mask] = 0.05
    replay = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(99),
        device="cpu",
    )
    with pytest.raises(ValueError, match="configured corrupt range"):
        replay.load_state_dict(
            checkpoint,
            expected_fault_source_population=outside_aihwkit_range,
            expected_pre_fault_state=pre_fault,
        )


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


def test_pulse_sgd_is_moment_free_and_obeys_layer_scope_and_cap() -> None:
    layout = build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    population = _population(layout)
    plant = IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(3),
        device="cpu",
    )
    optimizer = PulseSGD(
        size=population.size,
        layout=layout,
        learning_rates=(0.1, 0.1),
        layer_scope="output_only",
        nominal_dw_min=0.1,
        pulse_cap_per_cell=1,
        pulse_rule="stochastic_pulse_sign_sgd",
        generator=torch.Generator().manual_seed(4),
        device="cpu",
    )

    optimizer.step(torch.ones(population.size), plant)
    optimizer.step(torch.ones(population.size), plant)
    report = optimizer.report()

    assert report["commanded_pulses_by_layer"] == [0, 2]
    assert report["commanded_pulses"] == 2
    assert report["commanded_cells"] == 2
    assert report["blocked_at_cap"] == 2
    assert report["digital_first_moment_values"] == 0
    assert report["digital_second_moment_values"] == 0
    assert report["fp32_shadow_weight_values"] == 0
