from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import pytest
import torch

from experiments.mnist_conv import lr_step


@dataclass
class _Parameter:
    state: torch.Tensor
    name: str
    min_cond: float | None
    max_cond: float | None


def _parameter(values, name, *, lower=None, upper=None):
    return _Parameter(
        state=torch.tensor(values, dtype=torch.float64, requires_grad=True),
        name=name,
        min_cond=lower,
        max_cond=upper,
    )


def _optimizer(params, *, lr=1.0, momentum=0.0, weight_decay=0.0):
    return torch.optim.SGD(
        [param.state for param in params],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def test_step_exposes_all_transitions_and_per_parameter_diagnostics():
    weight = _parameter(
        [0.0, 50.0, 100.0, 90.0],
        "ConvWeight_0 ",
        lower=0.0,
        upper=100.0,
    )
    bias = _parameter([0.0, 2.0], "Bias_0")
    weight.state.grad = torch.tensor([1.0, -10.0, -1.0, -20.0], dtype=torch.float64)
    bias.state.grad = torch.tensor([1.0, -2.0], dtype=torch.float64)

    result = lr_step.sgd_step_with_diagnostics(
        [weight, bias],
        _optimizer([weight, bias]),
    )

    by_name = result.by_name
    transition = by_name["ConvWeight_0"]
    assert result.restored is False
    assert transition.parameter_kind == "bounded_weight"
    assert transition.bounded_gate is True
    assert transition.report_only is False
    assert torch.equal(
        transition.pre_update,
        torch.tensor([0.0, 50.0, 100.0, 90.0], dtype=torch.float64),
    )
    assert torch.equal(
        transition.post_sgd_pre_projection,
        torch.tensor([-1.0, 60.0, 101.0, 110.0], dtype=torch.float64),
    )
    assert torch.equal(
        transition.post_projection,
        torch.tensor([0.0, 60.0, 100.0, 100.0], dtype=torch.float64),
    )
    expected_rms = math.sqrt(502.0 / 4.0)
    assert transition.gradient_rms == pytest.approx(expected_rms)
    assert transition.proposed_update_rms == pytest.approx(expected_rms)
    assert transition.normalized_update == pytest.approx(expected_rms / 100.0)
    assert transition.proposed_bound_crossing_fraction == 0.75
    assert transition.lower_bound_occupancy == 0.25
    assert transition.upper_bound_occupancy == 0.5
    assert transition.combined_bound_occupancy == 0.75
    assert transition.projection_efficiency == pytest.approx(
        math.sqrt(200.0) / (math.sqrt(502.0) + 1e-12)
    )
    assert transition.proposal_is_numerically_zero is False
    assert transition.projection_gate_eligible is True
    assert torch.equal(weight.state.detach(), transition.post_projection)

    bias_transition = by_name["Bias_0"]
    assert bias_transition.parameter_kind == "bias"
    assert bias_transition.bounded_gate is False
    assert bias_transition.report_only is True
    assert torch.equal(
        bias_transition.post_sgd_pre_projection,
        torch.tensor([-1.0, 4.0], dtype=torch.float64),
    )
    assert torch.equal(
        bias_transition.post_projection,
        bias_transition.post_sgd_pre_projection,
    )
    assert bias_transition.proposed_update_rms == pytest.approx(math.sqrt(2.5))
    assert bias_transition.normalized_update == pytest.approx(
        math.sqrt(2.5) / math.sqrt(2.0)
    )
    assert bias_transition.proposed_bound_crossing_fraction is None
    assert bias_transition.lower_bound_occupancy is None
    assert bias_transition.upper_bound_occupancy is None
    assert bias_transition.combined_bound_occupancy is None
    assert bias_transition.projection_efficiency == pytest.approx(1.0)
    assert bias_transition.projection_gate_eligible is False


def test_shadow_step_restores_parameter_and_optimizer_state_exactly():
    weight = _parameter(
        [0.0, 50.0, 100.0],
        "DenseWeight_0",
        lower=0.0,
        upper=100.0,
    )
    weight.state.grad = torch.tensor([2.0, -3.0, -4.0], dtype=torch.float64)
    optimizer = _optimizer([weight], lr=2.0)
    optimizer.state[weight.state]["sentinel"] = torch.tensor(
        [1.25], dtype=torch.float64
    )
    original_tensor = weight.state
    original_value = weight.state.detach().clone()
    original_gradient = weight.state.grad.detach().clone()
    original_optimizer_state = copy.deepcopy(optimizer.state_dict())

    result = lr_step.shadow_sgd_step_with_diagnostics([weight], optimizer)

    assert result.restored is True
    assert weight.state is original_tensor
    assert torch.equal(weight.state.detach(), original_value)
    assert torch.equal(weight.state.grad, original_gradient)
    assert optimizer.param_groups[0]["lr"] == original_optimizer_state["param_groups"][0]["lr"]
    assert torch.equal(
        optimizer.state[weight.state]["sentinel"],
        original_optimizer_state["state"][0]["sentinel"],
    )
    transition = result.by_name["DenseWeight_0"]
    assert not torch.equal(transition.post_sgd_pre_projection, original_value)
    assert torch.equal(
        transition.post_projection,
        torch.tensor([0.0, 56.0, 100.0], dtype=torch.float64),
    )


def test_crossings_are_strict_while_projection_occupancy_includes_bounds():
    weight = _parameter(
        [1.0, 99.0, 50.0],
        "DenseWeight_2",
        lower=0.0,
        upper=100.0,
    )
    weight.state.grad = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)

    transition = lr_step.sgd_step_with_diagnostics(
        [weight], _optimizer([weight])
    ).parameters[0]

    assert torch.equal(
        transition.post_sgd_pre_projection,
        torch.tensor([0.0, 100.0, 50.0], dtype=torch.float64),
    )
    assert transition.proposed_bound_crossing_fraction == 0.0
    assert transition.lower_bound_occupancy == pytest.approx(1.0 / 3.0)
    assert transition.upper_bound_occupancy == pytest.approx(1.0 / 3.0)
    assert transition.combined_bound_occupancy == pytest.approx(2.0 / 3.0)
    assert transition.projection_efficiency == pytest.approx(1.0)


def test_zero_proposals_have_formula_efficiency_but_are_excluded_from_gates():
    weight = _parameter(
        [0.0, 25.0],
        "ConvWeight_4",
        lower=0.0,
        upper=100.0,
    )
    bias = _parameter([0.0, 0.0], "Bias_4")
    weight.state.grad = torch.zeros_like(weight.state)
    bias.state.grad = torch.tensor([-1e-9, 0.0], dtype=torch.float64)

    result = lr_step.sgd_step_with_diagnostics(
        [weight, bias], _optimizer([weight, bias])
    )

    weight_transition = result.by_name["ConvWeight_4"]
    assert weight_transition.proposed_update_rms == 0.0
    assert weight_transition.projection_efficiency == 0.0
    assert weight_transition.proposal_is_numerically_zero is True
    assert weight_transition.projection_gate_eligible is False

    bias_transition = result.by_name["Bias_4"]
    assert bias_transition.normalized_update == pytest.approx(
        (1e-9 / math.sqrt(2.0)) / 1e-8
    )
    assert bias_transition.proposal_is_numerically_zero is False
    assert bias_transition.projection_gate_eligible is False


def test_normalized_weight_update_uses_each_tensor_bound_width():
    weight = _parameter(
        [0.0, 1.0],
        "DenseWeight_7",
        lower=-2.0,
        upper=2.0,
    )
    weight.state.grad = torch.tensor([4.0, 4.0], dtype=torch.float64)

    transition = lr_step.sgd_step_with_diagnostics(
        [weight], _optimizer([weight])
    ).parameters[0]

    assert transition.proposed_update_rms == pytest.approx(4.0)
    assert transition.normalized_update == pytest.approx(1.0)


def test_unbounded_normalization_can_be_frozen_to_initial_scale():
    bias = _parameter([3.0, 4.0], "Bias_8")
    scales = lr_step.initial_unbounded_parameter_scales([bias])
    assert scales == {"Bias_8": pytest.approx(math.sqrt(12.5))}
    bias.state.data.fill_(20.0)
    bias.state.grad = torch.tensor([2.0, -2.0], dtype=torch.float64)

    transition = lr_step.sgd_step_with_diagnostics(
        [bias],
        _optimizer([bias]),
        unbounded_normalization_scales=scales,
    ).parameters[0]

    assert transition.proposed_update_rms == pytest.approx(2.0)
    assert transition.normalized_update == pytest.approx(2.0 / math.sqrt(12.5))


@pytest.mark.parametrize(
    ("momentum", "weight_decay", "message"),
    [
        (0.1, 0.0, "momentum to be 0"),
        (0.0, 0.1, "weight_decay to be 0"),
    ],
)
def test_optimizer_contract_rejects_momentum_and_weight_decay_without_mutation(
    momentum, weight_decay, message
):
    weight = _parameter(
        [25.0],
        "ConvWeight_0",
        lower=0.0,
        upper=100.0,
    )
    weight.state.grad = torch.tensor([1.0], dtype=torch.float64)
    before = weight.state.detach().clone()

    with pytest.raises(ValueError, match=message):
        lr_step.sgd_step_with_diagnostics(
            [weight],
            _optimizer(
                [weight],
                momentum=momentum,
                weight_decay=weight_decay,
            ),
        )

    assert torch.equal(weight.state.detach(), before)
