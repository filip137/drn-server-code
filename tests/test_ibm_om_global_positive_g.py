from __future__ import annotations

import torch

from experiments.mnist_relu_drn.ibm_om_baseline_selection import quad_stack
from experiments.mnist_relu_drn.ibm_om_global_positive_g import (
    lift_global_positive_g_gradients,
    map_global_positive_g,
)


def _masters() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[-1.0, -0.25], [0.0, 0.75]]),
        torch.tensor([[0.2, -0.6]]),
    )


def test_global_mapping_uses_four_positive_cells_in_g_zero_to_two() -> None:
    view = map_global_positive_g(_masters())

    expected = (
        torch.tensor(
            [
                [[0.0, 2.0, 2.0, 0.0], [0.0, 0.5, 0.5, 0.0]],
                [[0.0, 0.0, 0.0, 0.0], [1.5, 0.0, 0.0, 1.5]],
            ]
        ),
        torch.tensor([[[0.4, 0.0, 0.0, 0.4], [0.0, 1.2, 1.2, 0.0]]]),
    )
    for physical, quads, layout in zip(
        view.full_conductance, expected, ("halves", "paired"), strict=True
    ):
        assert torch.equal(quad_stack(physical, layout=layout), quads)
        assert bool(torch.all(physical >= 0.0))
        assert bool(torch.all(physical <= 2.0))
    assert view.mapping_report["array_specific_information_consumed"] is False
    assert view.mapping_report["coordinate"] == "x=G/2"


def test_global_quantizer_is_optional_and_uses_one_shared_spacing() -> None:
    view = map_global_positive_g(_masters(), quantization_spacing_x=0.25)

    assert torch.equal(
        view.realized_logical_master[0],
        torch.tensor([[-1.0, -0.25], [0.0, 0.75]]),
    )
    assert torch.equal(
        view.realized_logical_master[1], torch.tensor([[0.25, -0.5]])
    )
    assert view.quantization_spacing_x == 0.25
    assert view.requested_level is not None


def test_global_positive_g_gradient_lift_uses_sign_selected_active_cells() -> None:
    masters = _masters()
    view = map_global_positive_g(masters)
    physical = tuple(
        torch.arange(value.numel()).reshape(value.shape).float()
        for value in view.full_conductance
    )

    lifted = lift_global_positive_g_gradients(masters, physical)

    for master, gradient, physical_gradient, layout in zip(
        masters, lifted, physical, ("halves", "paired"), strict=True
    ):
        quad = quad_stack(physical_gradient, layout=layout)
        expected = torch.where(
            master >= 0.0,
            2.0 * (quad[..., 0] + quad[..., 3]),
            -2.0 * (quad[..., 1] + quad[..., 2]),
        )
        assert torch.equal(gradient, expected)


def test_global_positive_g_gradient_lift_matches_autograd_away_from_zero() -> None:
    masters = (
        torch.tensor([[-0.8, 0.3], [0.2, -0.4]], requires_grad=True),
        torch.tensor([[0.7, -0.6]], requires_grad=True),
    )
    view = map_global_positive_g(masters)
    coefficients = tuple(
        torch.linspace(-0.7, 1.3, value.numel()).reshape(value.shape)
        for value in view.full_conductance
    )
    objective = sum(
        (value * coefficient).sum()
        for value, coefficient in zip(
            view.full_conductance,
            coefficients,
            strict=True,
        )
    )
    automatic = torch.autograd.grad(objective, masters)
    lifted = lift_global_positive_g_gradients(masters, coefficients)

    for actual, expected in zip(lifted, automatic, strict=True):
        assert torch.equal(actual, expected)
