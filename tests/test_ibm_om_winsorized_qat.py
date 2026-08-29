import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    quad_stack,
    scatter_quads,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat import (
    WinsorizedQatLayerTemplate,
    logical_master_gradients,
    quantize_winsorized_logical_master,
)


_LOGICAL_SHAPE = (2, 3)
_SPACING_RAW_X = 0.125


def _template(
    *,
    layer_index: int,
    layout: str,
    initial: torch.Tensor,
) -> WinsorizedQatLayerTemplate:
    plus_baseline = torch.full(_LOGICAL_SHAPE, 0.25, dtype=torch.float32)
    minus_baseline = torch.full(_LOGICAL_SHAPE, 0.375, dtype=torch.float32)
    baseline_quad = torch.stack(
        (plus_baseline, minus_baseline, plus_baseline, minus_baseline), dim=-1
    )
    upper_quad = baseline_quad + torch.tensor(
        [0.5, 0.25, 0.25, 0.5], dtype=torch.float32
    )
    physical_shape = (2 * _LOGICAL_SHAPE[0], 2 * _LOGICAL_SHAPE[1])
    baseline_raw_x = scatter_quads(
        baseline_quad, shape=physical_shape, layout=layout
    )
    return WinsorizedQatLayerTemplate(
        layer_index=layer_index,
        layout=layout,
        baseline_full_g=2.0 * baseline_raw_x,
        baseline_raw_x=baseline_raw_x,
        cell_upper_raw_x=scatter_quads(
            upper_quad, shape=physical_shape, layout=layout
        ),
        positive_headroom_raw_x=torch.full(_LOGICAL_SHAPE, 0.5),
        negative_headroom_raw_x=torch.full(_LOGICAL_SHAPE, 0.25),
        positive_capacity=torch.full(_LOGICAL_SHAPE, 4, dtype=torch.int64),
        negative_capacity=torch.full(_LOGICAL_SHAPE, 2, dtype=torch.int64),
        initial_normalized_weight=initial.clone(),
        level_spacing_raw_x=_SPACING_RAW_X,
    )


def _templates_and_masters():
    # The first two values are exact half-level ties on the positive and
    # negative envelopes.  The +/-1 values exercise both capacity ceilings.
    first = torch.tensor(
        [[0.125, -0.25, 0.0], [1.0, -1.0, 0.49]], dtype=torch.float32
    )
    second = torch.tensor(
        [[-0.25, 0.125, 0.0], [-1.0, 1.0, -0.74]], dtype=torch.float32
    )
    templates = (
        _template(layer_index=0, layout="halves", initial=first),
        _template(layer_index=1, layout="paired", initial=second),
    )
    return templates, (first.clone(), second.clone())


def _expected_logical_levels(master: torch.Tensor) -> torch.Tensor:
    positive = master >= 0.0
    headroom = torch.where(positive, 0.5, 0.25)
    capacity = torch.where(
        positive,
        torch.full_like(master, 4, dtype=torch.int64),
        torch.full_like(master, 2, dtype=torch.int64),
    )
    rounded = torch.floor(headroom * master.abs() / _SPACING_RAW_X + 0.5).to(
        torch.int64
    )
    return torch.minimum(rounded, capacity)


def _expected_physical_levels(master: torch.Tensor, *, layout: str) -> torch.Tensor:
    level = _expected_logical_levels(master)
    positive = master >= 0.0
    zero = torch.zeros_like(level)
    quad = torch.stack(
        (
            torch.where(positive, level, zero),
            torch.where(positive, zero, level),
            torch.where(positive, zero, level),
            torch.where(positive, level, zero),
        ),
        dim=-1,
    )
    return scatter_quads(
        quad,
        shape=(2 * master.shape[0], 2 * master.shape[1]),
        layout=layout,
    )


def test_synthetic_epoch_zero_parity_sign_capacity_and_half_ties() -> None:
    templates, masters = _templates_and_masters()
    view = quantize_winsorized_logical_master(
        masters, templates, include_report=False
    )

    assert view.report == {}
    for master, template, actual_index, target in zip(
        masters,
        templates,
        view.requested_index,
        view.full_conductance_targets,
    ):
        expected_index = _expected_physical_levels(
            master, layout=template.layout
        )
        expected_target = template.baseline_full_g + expected_index.to(
            torch.float32
        ) * (2.0 * _SPACING_RAW_X)
        assert torch.equal(actual_index, expected_index)
        assert torch.equal(target, expected_target)

        selected = _expected_logical_levels(master)
        # floor(z+0.5) makes both exact +/- half-level ties program level one.
        assert selected[0, 0].item() == 1
        assert selected[0, 1].item() == 1
        assert selected[0, 2].item() == 0
        # The two endpoints reach their sign-specific capacity, 4 versus 2.
        assert selected[1, 0].item() in (2, 4)
        assert selected[1, 1].item() in (2, 4)
        assert {selected[1, 0].item(), selected[1, 1].item()} == {2, 4}


def test_quantizer_keeps_full_baseline_and_exact_G_equals_two_x() -> None:
    templates, masters = _templates_and_masters()
    view = quantize_winsorized_logical_master(masters, templates)

    assert len(view.report["layers"]) == 2
    for master, template, target_g, target_x, index in zip(
        masters,
        templates,
        view.full_conductance_targets,
        view.raw_x_targets,
        view.requested_index,
    ):
        assert torch.equal(target_g, 2.0 * target_x)
        assert bool(torch.all(target_g >= template.baseline_full_g))
        assert bool(torch.all(target_g <= 2.0))
        assert torch.equal(
            target_g,
            template.baseline_full_g
            + index.to(torch.float32) * (2.0 * _SPACING_RAW_X),
        )

        zero_location = (0, 2)
        assert master[zero_location].item() == 0.0
        target_quad = quad_stack(target_g, layout=template.layout)
        baseline_quad = quad_stack(
            template.baseline_full_g, layout=template.layout
        )
        # Logical zero is the complete, nonzero shared baseline on all rails;
        # it is not represented by a subtracted or all-zero conductance quad.
        assert torch.equal(target_quad[zero_location], baseline_quad[zero_location])
        assert bool(torch.all(target_quad[zero_location] > 0.0))


@pytest.mark.parametrize("layout", ["halves", "paired"])
def test_manual_ste_gradient_matches_full_g_continuous_lift(layout: str) -> None:
    templates, masters = _templates_and_masters()
    selected_templates = tuple(
        _template(layer_index=index, layout=layout, initial=master)
        for index, master in enumerate(masters)
    )
    gradient_quad = torch.tensor([1.0, 2.0, 3.0, 4.0]).expand(
        *_LOGICAL_SHAPE, 4
    )
    physical_gradients = tuple(
        scatter_quads(
            gradient_quad,
            shape=template.baseline_full_g.shape,
            layout=layout,
        )
        for template in selected_templates
    )

    actual = logical_master_gradients(
        masters, physical_gradients, selected_templates
    )
    for master, gradient in zip(masters, actual):
        # Positive (including the exact zero tie):
        #   2*H+*(dL/dG++ + dL/dG--) = 2*.5*(1+4) = 5.
        # Negative:
        #  -2*H-*(dL/dG+- + dL/dG-+) = -2*.25*(2+3) = -2.5.
        expected = torch.where(
            master >= 0.0,
            torch.full_like(master, 5.0),
            torch.full_like(master, -2.5),
        )
        assert torch.equal(gradient, expected)
        assert bool(torch.all(torch.isfinite(gradient)))


def test_manual_ste_rejects_nonfinite_physical_gradient() -> None:
    templates, masters = _templates_and_masters()
    physical_gradients = [torch.ones_like(value.baseline_full_g) for value in templates]
    physical_gradients[0][0, 0] = torch.inf
    with pytest.raises(FloatingPointError, match="became non-finite"):
        logical_master_gradients(masters, physical_gradients, templates)
