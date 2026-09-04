import torch

from experiments.mnist_relu_drn.hfo2_figure6_drn import (
    ENDPOINT_LAYOUTS,
    EndpointField,
    HfO2Figure6PulsePlant,
    PersistentHfO2PulseAdam,
    endpoint_field_from_population,
    lift_endpoint_gradients,
    map_masters_to_conductance,
    master_to_progress,
    program_masters_with_verify,
    program_progress_with_verify,
    rotate_endpoint_field,
)
from experiments.mnist_relu_drn.hfo2_figure6_endpoint_regimes import (
    INDEPENDENT_ENDPOINTS,
    sample_hfo2_figure6_endpoint_population,
)


SHAPES = ((4, 4), (2, 2))


def _field() -> EndpointField:
    reset = (
        torch.linspace(0.03, 0.12, 16, dtype=torch.float32).reshape(4, 4),
        torch.linspace(0.04, 0.10, 4, dtype=torch.float32).reshape(2, 2),
    )
    set_state = (
        torch.linspace(1.4, 2.4, 16, dtype=torch.float32).reshape(4, 4),
        torch.linspace(1.6, 2.2, 4, dtype=torch.float32).reshape(2, 2),
    )
    return EndpointField(
        reset=reset,
        set=set_state,
        shapes=SHAPES,
        layouts=ENDPOINT_LAYOUTS,
        population_report={"regime": INDEPENDENT_ENDPOINTS},
    )


def _masters(*, requires_grad: bool = False):
    first = torch.tensor(
        [[0.75, -0.25], [-0.5, 1.0]],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    second = torch.tensor(
        [[-0.6]], dtype=torch.float32, requires_grad=requires_grad
    )
    return first, second


def test_endpoint_population_is_placed_in_canonical_binding_order() -> None:
    population = sample_hfo2_figure6_endpoint_population(
        devices=20,
        assignment_seed=17,
        regime=INDEPENDENT_ENDPOINTS,
    )
    field = endpoint_field_from_population(
        population, shapes=SHAPES, dtype=torch.float64
    )

    assert field.devices == 20
    assert torch.equal(
        torch.cat(tuple(value.reshape(-1).double() for value in field.reset)),
        population.reset_state,
    )
    assert torch.equal(
        torch.cat(tuple(value.reshape(-1).double() for value in field.set)),
        population.set_state,
    )


def test_direct_mapping_uses_reset_at_zero_and_set_at_unit_progress() -> None:
    field = _field()
    progress = master_to_progress(_masters(), field)
    conductance = map_masters_to_conductance(_masters(), field)

    for value in progress:
        assert torch.all((value >= 0.0) & (value <= 1.0))
    for value, lower, upper in zip(
        conductance, field.reset, field.set, strict=True
    ):
        assert torch.all(value >= lower)
        assert torch.all(value <= upper)
        assert torch.equal(value[value == lower], lower[value == lower])
    assert any(bool(torch.any(value == upper)) for value, upper in zip(conductance, field.set))


def test_endpoint_gradient_lift_matches_autograd_chain_rule() -> None:
    field = _field()
    masters = _masters(requires_grad=True)
    physical = map_masters_to_conductance(masters, field)
    physical_gradients = (
        torch.linspace(-0.8, 0.9, 16).reshape(4, 4),
        torch.tensor([[0.2, -0.3], [0.5, -0.7]]),
    )
    loss = sum(
        (value * gradient).sum()
        for value, gradient in zip(physical, physical_gradients, strict=True)
    )
    loss.backward()
    expected = tuple(master.grad.detach().clone() for master in masters)
    observed = lift_endpoint_gradients(masters, physical_gradients, field)

    for actual, reference in zip(observed, expected, strict=True):
        assert torch.allclose(actual, reference, rtol=0.0, atol=1e-6)


def test_assignment_rotation_preserves_endpoint_pairs_and_marginals() -> None:
    field = _field()
    rotated = rotate_endpoint_field(field, 7)
    original_reset = torch.cat(tuple(value.reshape(-1) for value in field.reset))
    original_set = torch.cat(tuple(value.reshape(-1) for value in field.set))
    rotated_reset = torch.cat(tuple(value.reshape(-1) for value in rotated.reset))
    rotated_set = torch.cat(tuple(value.reshape(-1) for value in rotated.set))

    assert torch.equal(torch.sort(rotated_reset).values, torch.sort(original_reset).values)
    assert torch.equal(torch.sort(rotated_set).values, torch.sort(original_set).values)
    assert torch.equal(
        rotated_set - rotated_reset,
        torch.roll(original_set - original_reset, shifts=7),
    )


def test_pulse_plant_mutates_only_requested_cells_and_stays_in_bounds() -> None:
    field = _field()
    progress = master_to_progress(_masters(), field)
    plant = HfO2Figure6PulsePlant(
        field,
        progress,
        pulse_parameter_seed=101,
        pulse_noise_seed=102,
    )
    before = plant.raw_a.clone()
    direction = torch.zeros(plant.size, dtype=torch.int8)
    direction[0] = 1
    direction[3] = -1
    changed, capped = plant.pulse(direction, pulse_cap=2)

    assert changed <= 2
    assert capped == 0
    assert torch.equal(plant.raw_a[direction == 0], before[direction == 0])
    assert torch.all((plant.raw_a >= -1.0) & (plant.raw_a <= 1.0))
    for value, lower, upper in zip(
        plant.full_conductance, field.reset, field.set, strict=True
    ):
        assert torch.all(value >= lower)
        assert torch.all(value <= upper)


def test_pulse_adam_has_no_authoritative_weight_shadow_and_restores_exactly() -> None:
    field = _field()
    progress = master_to_progress(_masters(), field)
    plant = HfO2Figure6PulsePlant(
        field,
        progress,
        pulse_parameter_seed=201,
        pulse_noise_seed=202,
    )
    optimizer = PersistentHfO2PulseAdam(
        plant,
        learning_rate_progress=1.0,
        pulse_cap=4,
        pulse_selection_seed=203,
    )
    gradients = tuple(torch.ones(shape) for shape in SHAPES)
    report = optimizer.step(gradients)
    plant_state = plant.state_dict()
    optimizer_state = optimizer.state_dict()

    assert report.pulsed_cells > 0
    assert optimizer_state["authoritative_weight_shadow"] is None
    saved_raw_a = plant.raw_a.clone()
    saved_first_moment = optimizer.first_moment.clone()
    optimizer.step(tuple(-value for value in gradients))
    plant.load_state_dict(plant_state)
    optimizer.load_state_dict(optimizer_state)
    assert torch.equal(plant.raw_a, saved_raw_a)
    assert torch.equal(optimizer.first_moment, saved_first_moment)


def test_program_verify_starts_at_full_reset_and_accepts_zero_without_pulses() -> None:
    field = _field()
    targets = tuple(torch.zeros(shape) for shape in SHAPES)
    plant, result = program_progress_with_verify(
        field,
        targets,
        pulse_parameter_seed=301,
        pulse_noise_seed=302,
        tolerance_progress=0.1,
        maximum_pulses=8,
        noisy_initial_reset_verify=True,
    )

    assert torch.all(result.accepted)
    assert not torch.any(result.nonfinite)
    assert not torch.any(result.budget_exhausted)
    assert torch.all(result.total_pulses == 0)
    assert torch.all(result.verify_count == 1)
    assert torch.all(plant.raw_a == -1.0)
    for conductance, reset in zip(
        plant.full_conductance, field.reset, strict=True
    ):
        assert torch.equal(conductance, reset)


def test_program_verify_uses_stochastic_pulses_and_keeps_terminal_accounting() -> None:
    field = _field()
    plant, result = program_masters_with_verify(
        _masters(),
        field,
        pulse_parameter_seed=401,
        pulse_noise_seed=402,
        tolerance_progress=0.1,
        maximum_pulses=12,
        noisy_initial_reset_verify=False,
    )

    assert torch.any(result.total_pulses > 0)
    assert torch.equal(result.total_pulses, plant.pulse_count)
    assert torch.equal(result.set_count + result.reset_count, result.total_pulses)
    assert torch.all(result.accepted | result.nonfinite | result.budget_exhausted)
    assert torch.all(result.total_pulses <= 12)
    assert torch.equal(
        result.apparent_endpoint,
        torch.cat(
            tuple(value.reshape(-1) for value in plant.apparent_progress_unprojected)
        ),
    )
    assert all(
        torch.all((value >= lower) & (value <= upper))
        for value, lower, upper in zip(
            plant.full_conductance, field.reset, field.set, strict=True
        )
    )


def test_apparent_projection_is_explicit_and_recovery_clone_is_exact() -> None:
    field = _field()
    plant = HfO2Figure6PulsePlant(
        field,
        master_to_progress(_masters(), field),
        pulse_parameter_seed=501,
        pulse_noise_seed=502,
    )
    plant.apparent_raw_a[0] = -1.5
    plant.apparent_raw_a[1] = 1.5
    projected = torch.cat(
        tuple(value.reshape(-1) for value in plant.apparent_progress_projected)
    )
    assert projected[0] == 0.0
    assert projected[1] == 1.0
    for conductance, lower, upper in zip(
        plant.apparent_full_conductance, field.reset, field.set, strict=True
    ):
        assert torch.all(conductance >= lower)
        assert torch.all(conductance <= upper)

    clone = plant.clone_for_new_pulse_phase(pulse_noise_seed=503)
    assert torch.equal(clone.raw_a, plant.raw_a)
    assert torch.equal(clone.apparent_raw_a, plant.apparent_raw_a)
    assert torch.equal(clone.dwmin_up_raw_a, plant.dwmin_up_raw_a)
    assert torch.equal(clone.dwmin_down_raw_a, plant.dwmin_down_raw_a)
    assert torch.all(clone.pulse_count == 0)
