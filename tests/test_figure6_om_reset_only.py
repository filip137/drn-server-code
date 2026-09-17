from __future__ import annotations

import torch

from experiments.mnist_relu_drn.figure6_om_pulse import (
    OM_CYCLE_NOISE_STD,
    OM_NOMINAL_DW_MIN_RAW_A,
    OM_WRITE_NOISE_STD,
)
from experiments.mnist_relu_drn.figure6_om_reset_only import (
    apparent_conductance_unprojected,
    lift_reset_only_gradients,
    map_masters_from_reset_only,
    program_conductance_with_verify,
)
from experiments.mnist_relu_drn.hfo2_figure6_drn import EndpointField
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    quad_stack,
    scatter_quads,
)
from training.ibm_reram_hwa import IbmReramArrayPopulation


SHAPES = ((4, 4), (2, 2))
LAYOUTS = ("halves", "paired")


def _physical_from_quads(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        scatter_quads(first, shape=SHAPES[0], layout=LAYOUTS[0]),
        scatter_quads(second, shape=SHAPES[1], layout=LAYOUTS[1]),
    )


def _field(*, low_set: bool = False) -> EndpointField:
    reset = _physical_from_quads(
        torch.tensor(
            [
                [[0.10, 0.20, 0.30, 0.40], [0.05, 0.07, 0.06, 0.08]],
                [[0.11, 0.12, 0.13, 0.14], [0.21, 0.22, 0.23, 0.24]],
            ],
            dtype=torch.float32,
        ),
        torch.tensor([[[0.10, 0.20, 0.30, 0.40]]], dtype=torch.float32),
    )
    set_quads_0 = torch.full((2, 2, 4), 2.2, dtype=torch.float32)
    set_quads_1 = torch.full((1, 1, 4), 2.2, dtype=torch.float32)
    if low_set:
        set_quads_0[0, 0, 0] = 0.60
        set_quads_0[0, 0, 3] = 0.80
        set_quads_1[0, 0, 0] = 0.55
    set_state = _physical_from_quads(set_quads_0, set_quads_1)
    return EndpointField(
        reset=reset,
        set=set_state,
        shapes=SHAPES,
        layouts=LAYOUTS,
        population_report={"test": True},
    )


def _masters() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[0.5, -0.25], [0.0, 1.0]], dtype=torch.float32),
        torch.tensor([[0.5]], dtype=torch.float32),
    )


def _population() -> IbmReramArrayPopulation:
    size = sum(shape[0] * shape[1] for shape in SHAPES)
    return IbmReramArrayPopulation(
        assignment_seed=123,
        corruption_policy="counterfactual_repaired",
        binding_keys=("w1", "w2"),
        binding_shapes=SHAPES,
        binding_sampling_seeds=(11, 12),
        donor_sampling_seeds=(21, 22),
        nominal_dw_min=OM_NOMINAL_DW_MIN_RAW_A,
        dw_min_std=OM_CYCLE_NOISE_STD,
        write_noise_std=OM_WRITE_NOISE_STD,
        max_bound=torch.full((size,), 1.1, dtype=torch.float32),
        min_bound=torch.full((size,), -1.1, dtype=torch.float32),
        dwmin_up=torch.full((size,), 0.1, dtype=torch.float32),
        dwmin_down=torch.full((size,), 0.1, dtype=torch.float32),
        reference=torch.zeros(size, dtype=torch.float32),
        corrupt=torch.zeros(size, dtype=torch.bool),
        published_corrupt=torch.zeros(size, dtype=torch.bool),
        fingerprint="toy-reset-only",
        aihwkit_version="1.1.0",
    )


def test_reset_only_requests_exact_physical_pair_equalities() -> None:
    field = _field()
    mapping = map_masters_from_reset_only(
        _masters(), field, nominal_set=2.0
    )
    for target, reset, layout in zip(
        mapping.requested_g, field.reset, LAYOUTS, strict=True
    ):
        target_quad = quad_stack(target, layout=layout)
        reset_quad = quad_stack(reset, layout=layout)
        baseline = reset_quad.max(dim=-1).values
        assert torch.equal(target_quad[..., 0], target_quad[..., 3])
        assert torch.equal(target_quad[..., 1], target_quad[..., 2])
        assert torch.equal(mapping.baselines[LAYOUTS.index(layout)], baseline)
        assert bool(torch.all(target_quad >= baseline.unsqueeze(-1)))


def test_true_set_changes_realization_but_not_reset_only_request() -> None:
    high = map_masters_from_reset_only(_masters(), _field(), nominal_set=2.0)
    low = map_masters_from_reset_only(
        _masters(), _field(low_set=True), nominal_set=2.0
    )
    for high_request, low_request in zip(
        high.requested_g, low.requested_g, strict=True
    ):
        assert torch.equal(high_request, low_request)
    assert any(
        bool(torch.any(left != right))
        for left, right in zip(
            high.plant_limited_g, low.plant_limited_g, strict=True
        )
    )
    assert sum(
        int(mask.sum().item()) for mask in low.requested_above_true_set
    ) > 0


def test_reset_only_gradient_does_not_consume_true_set_or_saturation_mask() -> None:
    gradients = tuple(torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape) for shape in SHAPES)
    high = lift_reset_only_gradients(
        _masters(), gradients, _field(), nominal_set=2.0
    )
    low = lift_reset_only_gradients(
        _masters(), gradients, _field(low_set=True), nominal_set=2.0
    )
    assert all(torch.equal(left, right) for left, right in zip(high, low, strict=True))


def test_absolute_g_program_verify_does_not_normalize_by_true_set() -> None:
    field = _field(low_set=True)
    requested = tuple(value.clone() for value in field.reset)
    requested[0][0, 0] = 1.5
    plant, result = program_conductance_with_verify(
        field,
        requested,
        _population(),
        pulse_noise_seed=991,
        tolerance_conductance=0.001,
        maximum_pulses=2,
    )
    assert int(result.budget_exhausted.sum().item()) == 1
    assert int(result.total_pulses.sum().item()) == 2
    assert torch.equal(
        result.apparent_endpoint, apparent_conductance_unprojected(plant)
    )
    assert bool(torch.all(result.accepted[1:]))

