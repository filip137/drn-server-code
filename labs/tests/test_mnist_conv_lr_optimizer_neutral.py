from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch

from experiments.mnist_conv.lr_engine import (
    FROZEN_ADAM_OPTIMIZER,
    FROZEN_SGD_OPTIMIZER,
    build_model_runtime,
    validate_explicit_optimizer_contract,
)
from experiments.mnist_conv.lr_step import (
    OptimizerParameterTransition,
    OptimizerStateNumericalError,
    optimizer_step_with_diagnostics,
    shadow_optimizer_step_with_diagnostics,
)
from experiments.mnist_conv.lr_study_spec import LRStudySpec


ROOT = Path(__file__).resolve().parents[2]
V6_CONFIG = (
    ROOT
    / "configs"
    / "conv"
    / "hardsigmoid_lr_conv2_two_rho_constant_sgd_bs16_v6.json"
)


@dataclass
class _Parameter:
    state: torch.Tensor
    name: str
    min_cond: float | None
    max_cond: float | None


def _parameter(
    values: list[float],
    name: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> _Parameter:
    return _Parameter(
        state=torch.tensor(values, dtype=torch.float64, requires_grad=True),
        name=name,
        min_cond=lower,
        max_cond=upper,
    )


def _adam(
    params: list[_Parameter],
    *,
    learning_rates: list[float] | None = None,
    amsgrad: bool = False,
) -> torch.optim.Adam:
    rates = learning_rates or [0.01] * len(params)
    return torch.optim.Adam(
        [
            {"params": [param.state], "lr": rate}
            for param, rate in zip(params, rates)
        ],
        lr=1.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=amsgrad,
        foreach=False,
        fused=False,
        maximize=False,
        capturable=False,
        differentiable=False,
    )


def _assert_nested_exact(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        assert torch.equal(actual, expected)
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_exact(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_exact(actual_item, expected_item)
    else:
        assert actual == expected


def test_exact_first_adam_step_and_optimizer_neutral_field_name() -> None:
    weight = _parameter(
        [25.0, 75.0],
        "ConvWeight_0",
        lower=0.0,
        upper=100.0,
    )
    gradient = torch.tensor([2.0, -4.0], dtype=torch.float64)
    weight.state.grad = gradient.clone()
    optimizer = _adam([weight], learning_rates=[0.01])

    result = optimizer_step_with_diagnostics([weight], optimizer)

    expected = torch.tensor([25.0, 75.0], dtype=torch.float64) - (
        0.01 * gradient / (gradient.abs() + 1e-8)
    )
    assert result.optimizer_name == "Adam"
    assert result.restored is False
    assert torch.allclose(weight.state.detach(), expected, rtol=0.0, atol=1e-14)
    transition = result.by_name["ConvWeight_0"]
    assert torch.allclose(
        transition.post_optimizer_pre_projection,
        expected,
        rtol=0.0,
        atol=1e-14,
    )
    assert "post_optimizer_pre_projection" in {
        field.name for field in fields(OptimizerParameterTransition)
    }
    assert "post_sgd_pre_projection" not in {
        field.name for field in fields(OptimizerParameterTransition)
    }
    state = optimizer.state[weight.state]
    assert float(state["step"].item()) == 1.0
    assert torch.allclose(state["exp_avg"], 0.1 * gradient)
    assert torch.allclose(state["exp_avg_sq"], 0.001 * gradient.square())


def test_adam_proposal_is_projected_after_the_optimizer_transition() -> None:
    weight = _parameter(
        [0.005, 99.995],
        "DenseWeight_0",
        lower=0.0,
        upper=100.0,
    )
    weight.state.grad = torch.tensor([1.0, -1.0], dtype=torch.float64)

    transition = optimizer_step_with_diagnostics(
        [weight],
        _adam([weight], learning_rates=[0.01]),
    ).parameters[0]

    assert transition.post_optimizer_pre_projection[0] < 0.0
    assert transition.post_optimizer_pre_projection[1] > 100.0
    assert torch.equal(
        transition.post_projection,
        torch.tensor([0.0, 100.0], dtype=torch.float64),
    )
    assert transition.proposed_bound_crossing_fraction == 1.0
    assert transition.combined_bound_occupancy == 1.0


def test_adam_shadow_restores_parameters_gradients_and_complete_state() -> None:
    weight = _parameter(
        [25.0, 75.0],
        "ConvWeight_0",
        lower=0.0,
        upper=100.0,
    )
    optimizer = _adam([weight], learning_rates=[0.02])
    weight.state.grad = torch.tensor([1.0, -3.0], dtype=torch.float64)
    optimizer_step_with_diagnostics([weight], optimizer)

    weight.state.grad = torch.tensor([-7.0, 11.0], dtype=torch.float64)
    parameter_before = weight.state.detach().clone()
    gradient_before = weight.state.grad.detach().clone()
    optimizer_before = copy.deepcopy(optimizer.state_dict())

    result = shadow_optimizer_step_with_diagnostics([weight], optimizer)

    assert result.restored is True
    assert not torch.equal(
        result.parameters[0].post_optimizer_pre_projection,
        parameter_before,
    )
    assert torch.equal(weight.state.detach(), parameter_before)
    assert torch.equal(weight.state.grad, gradient_before)
    _assert_nested_exact(optimizer.state_dict(), optimizer_before)
    assert float(optimizer.state[weight.state]["step"].item()) == 1.0


def test_adam_real_steps_preserve_and_advance_moments() -> None:
    bias = _parameter([1.0, -1.0], "Bias_0")
    optimizer = _adam([bias], learning_rates=[0.005])
    bias.state.grad = torch.tensor([2.0, -3.0], dtype=torch.float64)
    optimizer_step_with_diagnostics([bias], optimizer)
    first_average = optimizer.state[bias.state]["exp_avg"].detach().clone()

    bias.state.grad = torch.tensor([-4.0, 5.0], dtype=torch.float64)
    optimizer_step_with_diagnostics([bias], optimizer)

    state = optimizer.state[bias.state]
    assert float(state["step"].item()) == 2.0
    assert torch.allclose(
        state["exp_avg"],
        0.9 * first_average
        + 0.1 * torch.tensor([-4.0, 5.0], dtype=torch.float64),
    )


def test_nonfinite_adam_state_fails_before_parameter_mutation() -> None:
    bias = _parameter([1.0, -1.0], "Bias_0")
    optimizer = _adam([bias])
    bias.state.grad = torch.tensor([2.0, -3.0], dtype=torch.float64)
    optimizer_step_with_diagnostics([bias], optimizer)
    optimizer.state[bias.state]["exp_avg"][0] = float("nan")
    bias.state.grad = torch.tensor([-4.0, 5.0], dtype=torch.float64)
    before = bias.state.detach().clone()

    with pytest.raises(OptimizerStateNumericalError, match="non-finite"):
        optimizer_step_with_diagnostics([bias], optimizer)

    assert torch.equal(bias.state.detach(), before)


def test_adam_settings_and_one_group_per_parameter_are_strict() -> None:
    weight = _parameter(
        [25.0],
        "ConvWeight_0",
        lower=0.0,
        upper=100.0,
    )
    weight.state.grad = torch.tensor([1.0], dtype=torch.float64)
    before = weight.state.detach().clone()

    with pytest.raises(ValueError, match="amsgrad"):
        optimizer_step_with_diagnostics(
            [weight],
            _adam([weight], amsgrad=True),
        )
    assert torch.equal(weight.state.detach(), before)

    shared_group = torch.optim.Adam(
        [{"params": [weight.state, weight.state.detach().clone().requires_grad_()]}],
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        fused=False,
        maximize=False,
        capturable=False,
        differentiable=False,
    )
    with pytest.raises(ValueError, match="one optimizer parameter group"):
        optimizer_step_with_diagnostics([weight], shared_group)


def test_explicit_optimizer_contract_and_real_adam_runtime() -> None:
    assert validate_explicit_optimizer_contract(FROZEN_SGD_OPTIMIZER) == (
        FROZEN_SGD_OPTIMIZER
    )
    invalid_sgd = dict(FROZEN_SGD_OPTIMIZER)
    invalid_sgd["momentum"] = 0.1
    with pytest.raises(ValueError, match="exactly 0.0"):
        validate_explicit_optimizer_contract(invalid_sgd)

    assert validate_explicit_optimizer_contract(FROZEN_ADAM_OPTIMIZER) == (
        FROZEN_ADAM_OPTIMIZER
    )
    invalid = dict(FROZEN_ADAM_OPTIMIZER)
    invalid["betas"] = (0.9, 0.999)
    with pytest.raises(ValueError, match=r"exactly \[0.9, 0.999\]"):
        validate_explicit_optimizer_contract(invalid)

    study = LRStudySpec.from_path(V6_CONFIG)
    baseline = next(row for row in study.rows if row["scheme"] == "baseline")
    learning_rates = {
        "ConvWeight_0": 0.001,
        "Bias_0": 0.001,
        "ConvWeight_1": 0.002,
        "Bias_1": 0.002,
        "DenseWeight_0": 0.003,
    }
    runtime = build_model_runtime(
        study.data,
        baseline,
        device="cpu",
        learning_rate=learning_rates,
        optimizer_contract=FROZEN_ADAM_OPTIMIZER,
    )

    assert runtime.optimizer_name == "Adam"
    assert runtime.optimizer_neutral_transitions is True
    assert isinstance(runtime.optimizer, torch.optim.Adam)
    assert len(runtime.optimizer.param_groups) == len(runtime.parameters) == 5
    observed = {}
    for parameter, group in zip(runtime.parameters, runtime.optimizer.param_groups):
        assert len(group["params"]) == 1
        assert group["params"][0] is parameter.state
        observed[parameter.name.strip()] = group["lr"]
        for key in (
            "amsgrad",
            "foreach",
            "fused",
            "maximize",
            "capturable",
            "differentiable",
        ):
            assert group[key] is False
    assert observed == learning_rates
