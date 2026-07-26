"""Optimizer transitions and projection diagnostics for Conv LR studies.

The original SGD-facing dataclasses and wrappers remain intentionally intact:
v1--v6 artifacts serialize ``post_sgd_pre_projection`` exactly as before.
Optimizer-neutral callers use the parallel ``Optimizer*Transition`` types,
whose corresponding state is named ``post_optimizer_pre_projection``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import torch


BIAS_NORMALIZATION_EPSILON = 1e-8
PROJECTION_EFFICIENCY_EPSILON = 1e-12
ZERO_PROPOSAL_EPSILON = 1e-30


@dataclass(frozen=True)
class ParameterTransition:
    """Legacy v1--v6 SGD transition for one scientific parameter."""

    name: str
    parameter_kind: str
    bounded_gate: bool
    report_only: bool
    pre_update: torch.Tensor
    post_sgd_pre_projection: torch.Tensor
    post_projection: torch.Tensor
    gradient_rms: float
    proposed_update_rms: float
    normalized_update: float
    proposed_bound_crossing_fraction: float | None
    lower_bound_occupancy: float | None
    upper_bound_occupancy: float | None
    combined_bound_occupancy: float | None
    projection_efficiency: float
    proposal_is_numerically_zero: bool
    projection_gate_eligible: bool


@dataclass(frozen=True)
class SGDTransition:
    """Legacy result of one SGD step, projection, and optional restoration."""

    parameters: tuple[ParameterTransition, ...]
    restored: bool

    @property
    def by_name(self) -> dict[str, ParameterTransition]:
        return {transition.name: transition for transition in self.parameters}


@dataclass(frozen=True)
class OptimizerParameterTransition:
    """Optimizer-neutral transition for one scientific parameter."""

    name: str
    parameter_kind: str
    bounded_gate: bool
    report_only: bool
    pre_update: torch.Tensor
    post_optimizer_pre_projection: torch.Tensor
    post_projection: torch.Tensor
    gradient_rms: float
    proposed_update_rms: float
    normalized_update: float
    proposed_bound_crossing_fraction: float | None
    lower_bound_occupancy: float | None
    upper_bound_occupancy: float | None
    combined_bound_occupancy: float | None
    projection_efficiency: float
    proposal_is_numerically_zero: bool
    projection_gate_eligible: bool


@dataclass(frozen=True)
class OptimizerTransition:
    """Result of one optimizer step, projection, and optional restoration."""

    parameters: tuple[OptimizerParameterTransition, ...]
    restored: bool
    optimizer_name: str

    @property
    def by_name(self) -> dict[str, OptimizerParameterTransition]:
        return {transition.name: transition for transition in self.parameters}


@dataclass(frozen=True)
class ParameterOptimizerSnapshot:
    """Complete mutable state needed to undo one shadow optimizer proposal."""

    parameter_names: tuple[str, ...]
    parameter_states: tuple[torch.Tensor, ...]
    parameter_gradients: tuple[torch.Tensor | None, ...]
    optimizer_state_dict: dict[str, Any]


class OptimizerStateNumericalError(FloatingPointError):
    """Raised as soon as a parameter or optimizer state becomes non-finite."""


def _parameter_kind(name: str) -> str:
    if name.startswith(("ConvWeight_", "DenseWeight_")):
        return "bounded_weight"
    if name.startswith("Bias_"):
        return "bias"
    return "report_only"


def _rms(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        raise ValueError(
            f"Expected a non-empty parameter tensor, got shape {tuple(tensor.shape)}."
        )
    values = tensor.detach().to(dtype=torch.float64)
    return float(torch.sqrt(torch.mean(values * values)).item())


def _population_std(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        raise ValueError(
            f"Expected a non-empty parameter tensor, got shape {tuple(tensor.shape)}."
        )
    return float(tensor.detach().to(dtype=torch.float64).std(unbiased=False).item())


def _l2_norm(tensor: torch.Tensor) -> float:
    values = tensor.detach().to(dtype=torch.float64)
    return float(torch.linalg.vector_norm(values).item())


def _finite_bound(value: Any, *, name: str, side: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected {name}.{side} to be a finite bound, got {value!r}."
        ) from exc
    if not math.isfinite(result):
        raise ValueError(
            f"Expected {name}.{side} to be a finite bound, got {value!r}."
        )
    return result


def _validate_parameter_coverage(
    params: tuple[Any, ...],
    optimizer: torch.optim.Optimizer,
    *,
    optimizer_label: str,
    one_group_per_parameter: bool,
) -> None:
    optimizer_tensors = []
    for group_index, group in enumerate(optimizer.param_groups):
        tensors = list(group.get("params", ()))
        if one_group_per_parameter and len(tensors) != 1:
            raise ValueError(
                "Expected one optimizer parameter group per scientific parameter "
                f"with one tensor in every group; {optimizer_label} group "
                f"{group_index} contains {len(tensors)} tensors."
            )
        optimizer_tensors.extend(tensors)

    parameter_tensors = [param.state for param in params]
    optimizer_ids = [id(tensor) for tensor in optimizer_tensors]
    parameter_ids = [id(tensor) for tensor in parameter_tensors]
    if len(set(optimizer_ids)) != len(optimizer_ids):
        raise ValueError(
            f"Expected each tensor to occur once in {optimizer_label} parameter "
            f"groups, got tensor ids {optimizer_ids}."
        )
    if set(optimizer_ids) != set(parameter_ids) or len(optimizer_ids) != len(
        parameter_ids
    ):
        raise ValueError(
            f"Expected {optimizer_label} parameter groups to contain exactly the "
            f"supplied parameter states, got {len(optimizer_ids)} optimizer "
            f"tensors for {len(parameter_ids)} supplied parameters."
        )
    if one_group_per_parameter and len(optimizer.param_groups) != len(params):
        raise ValueError(
            "Expected one optimizer parameter group per scientific parameter, "
            f"got {len(optimizer.param_groups)} groups for {len(params)} parameters."
        )


def _validate_optimizer(
    params: tuple[Any, ...], optimizer: torch.optim.Optimizer
) -> None:
    """Validate the historical SGD contract without tightening group layout."""

    if not isinstance(optimizer, torch.optim.SGD):
        raise TypeError(
            "Expected optimizer to be torch.optim.SGD with momentum=0 and "
            f"weight_decay=0, got {type(optimizer).__name__}."
        )

    for group_index, group in enumerate(optimizer.param_groups):
        momentum = float(group.get("momentum", 0.0))
        weight_decay = float(group.get("weight_decay", 0.0))
        if momentum != 0.0:
            raise ValueError(
                f"Expected optimizer param group {group_index} momentum to be 0, "
                f"got {momentum}."
            )
        if weight_decay != 0.0:
            raise ValueError(
                "Expected optimizer param group "
                f"{group_index} weight_decay to be 0, got {weight_decay}."
            )
    _validate_parameter_coverage(
        params,
        optimizer,
        optimizer_label="SGD",
        one_group_per_parameter=False,
    )


def _require_exact_group_value(
    group: Mapping[str, Any],
    *,
    group_index: int,
    key: str,
    expected: Any,
) -> None:
    provided = group.get(key)
    if isinstance(expected, bool):
        matches = provided is expected
    elif key == "betas":
        matches = (
            isinstance(provided, tuple)
            and len(provided) == 2
            and not any(isinstance(value, bool) for value in provided)
            and tuple(float(value) for value in provided) == expected
        )
    else:
        matches = (
            not isinstance(provided, bool)
            and isinstance(provided, (int, float))
            and math.isfinite(float(provided))
            and float(provided) == float(expected)
        )
    if not matches:
        raise ValueError(
            f"Expected Adam param group {group_index} {key} to be exactly "
            f"{expected!r}, got {provided!r}."
        )


def _validate_adam_optimizer(
    params: tuple[Any, ...], optimizer: torch.optim.Optimizer
) -> None:
    if not isinstance(optimizer, torch.optim.Adam) or isinstance(
        optimizer, torch.optim.AdamW
    ):
        raise TypeError(
            "Expected optimizer to be torch.optim.Adam with the frozen Conv2 "
            f"diagnostic settings, got {type(optimizer).__name__}."
        )
    expected = {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "foreach": False,
        "fused": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
    }
    for group_index, group in enumerate(optimizer.param_groups):
        for key, value in expected.items():
            _require_exact_group_value(
                group,
                group_index=group_index,
                key=key,
                expected=value,
            )
        learning_rate = group.get("lr")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) < 0.0
        ):
            raise ValueError(
                "Expected every Adam parameter group to have an explicit "
                f"non-negative finite learning rate, got group {group_index} "
                f"value {learning_rate!r}."
            )
    _validate_parameter_coverage(
        params,
        optimizer,
        optimizer_label="Adam",
        one_group_per_parameter=True,
    )


def validate_optimizer(
    params: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    *,
    optimizer_neutral: bool = True,
) -> str:
    """Validate a supported frozen optimizer and return its canonical name."""

    values = _validate_params(params)
    if isinstance(optimizer, torch.optim.Adam) and not isinstance(
        optimizer, torch.optim.AdamW
    ):
        _validate_adam_optimizer(values, optimizer)
        return "Adam"
    if isinstance(optimizer, torch.optim.SGD):
        _validate_optimizer(values, optimizer)
        if optimizer_neutral:
            _validate_parameter_coverage(
                values,
                optimizer,
                optimizer_label="SGD",
                one_group_per_parameter=True,
            )
            for group_index, group in enumerate(optimizer.param_groups):
                learning_rate = group.get("lr")
                if (
                    isinstance(learning_rate, bool)
                    or not isinstance(learning_rate, (int, float))
                    or not math.isfinite(float(learning_rate))
                    or float(learning_rate) < 0.0
                ):
                    raise ValueError(
                        "Expected every SGD parameter group to have an explicit "
                        "non-negative finite learning rate, got group "
                        f"{group_index} value {learning_rate!r}."
                    )
        return "SGD"
    raise TypeError(
        "Expected optimizer to be frozen plain SGD or Adam, "
        f"got {type(optimizer).__name__}."
    )


def _validate_params(params: Iterable[Any]) -> tuple[Any, ...]:
    result = tuple(params)
    if not result:
        raise ValueError("Expected at least one optimizer parameter, got an empty list.")

    names = []
    for index, param in enumerate(result):
        missing = [
            attribute
            for attribute in ("state", "name", "min_cond", "max_cond")
            if not hasattr(param, attribute)
        ]
        if missing:
            raise TypeError(
                "Expected each parameter to expose state, name, min_cond, and "
                f"max_cond; parameter {index} is missing {missing}."
            )
        if not isinstance(param.state, torch.Tensor):
            raise TypeError(
                f"Expected parameter {index}.state to be a torch.Tensor, "
                f"got {type(param.state).__name__}."
            )
        name = str(param.name).strip()
        if not name:
            raise ValueError(f"Expected parameter {index}.name to be non-empty, got {param.name!r}.")
        names.append(name)

        if _parameter_kind(name) == "bounded_weight":
            lower = _finite_bound(param.min_cond, name=name, side="min_cond")
            upper = _finite_bound(param.max_cond, name=name, side="max_cond")
            if not lower < upper:
                raise ValueError(
                    f"Expected {name} bounds to satisfy min_cond < max_cond, "
                    f"got [{lower}, {upper}]."
                )

    if len(set(names)) != len(names):
        raise ValueError(f"Expected unique parameter names, got {names}.")
    return result


def _clone_state(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone(memory_format=torch.preserve_format)


def initial_unbounded_parameter_scales(
    params: Iterable[Any],
    *,
    epsilon: float = BIAS_NORMALIZATION_EPSILON,
) -> dict[str, float]:
    """Freeze the report-only normalization scale from initial parameter state."""

    values = _validate_params(params)
    try:
        floor = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected epsilon to be a positive finite number, got {epsilon!r}."
        ) from exc
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError(
            f"Expected epsilon to be a positive finite number, got {epsilon!r}."
        )
    return {
        str(param.name).strip(): max(
            _rms(param.state),
            _population_std(param.state),
            floor,
        )
        for param in values
        if _parameter_kind(str(param.name).strip()) != "bounded_weight"
    }


def _clone_gradient(tensor: torch.Tensor) -> torch.Tensor | None:
    if tensor.grad is None:
        return None
    return tensor.grad.detach().clone(memory_format=torch.preserve_format)


def capture_parameter_optimizer_state(
    params: Iterable[Any],
    optimizer: torch.optim.Optimizer,
) -> ParameterOptimizerSnapshot:
    """Capture parameter values, gradients, and the complete optimizer state."""

    values = _validate_params(params)
    _validate_parameter_coverage(
        values,
        optimizer,
        optimizer_label=type(optimizer).__name__,
        one_group_per_parameter=False,
    )
    return ParameterOptimizerSnapshot(
        parameter_names=tuple(str(param.name).strip() for param in values),
        parameter_states=tuple(_clone_state(param.state) for param in values),
        parameter_gradients=tuple(_clone_gradient(param.state) for param in values),
        optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
    )


def restore_parameter_optimizer_state(
    params: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    snapshot: ParameterOptimizerSnapshot,
) -> None:
    """Restore a snapshot without replacing any live parameter tensor object."""

    values = _validate_params(params)
    names = tuple(str(param.name).strip() for param in values)
    if names != snapshot.parameter_names:
        raise ValueError(
            "Expected snapshot parameter names and order to match live parameters, "
            f"got snapshot={snapshot.parameter_names!r}, live={names!r}."
        )
    if not (
        len(values)
        == len(snapshot.parameter_states)
        == len(snapshot.parameter_gradients)
    ):
        raise ValueError(
            "Expected snapshot tensor counts to match live parameters, "
            f"got parameters={len(values)}, states={len(snapshot.parameter_states)}, "
            f"gradients={len(snapshot.parameter_gradients)}."
        )
    _validate_parameter_coverage(
        values,
        optimizer,
        optimizer_label=type(optimizer).__name__,
        one_group_per_parameter=False,
    )
    with torch.no_grad():
        for param, state, gradient in zip(
            values,
            snapshot.parameter_states,
            snapshot.parameter_gradients,
        ):
            if state.shape != param.state.shape:
                raise ValueError(
                    f"Expected snapshot state for {str(param.name).strip()!r} "
                    f"to have shape {tuple(param.state.shape)}, got "
                    f"{tuple(state.shape)}."
                )
            param.state.copy_(state)
            if gradient is None:
                param.state.grad = None
            elif param.state.grad is None:
                param.state.grad = gradient.detach().clone(
                    memory_format=torch.preserve_format
                )
            else:
                if gradient.shape != param.state.grad.shape:
                    raise ValueError(
                        f"Expected snapshot gradient for "
                        f"{str(param.name).strip()!r} to have shape "
                        f"{tuple(param.state.grad.shape)}, got "
                        f"{tuple(gradient.shape)}."
                    )
                param.state.grad.copy_(gradient)
    optimizer.load_state_dict(copy.deepcopy(snapshot.optimizer_state_dict))


def _iter_state_values(value: Any, path: str):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_state_values(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _iter_state_values(child, f"{path}[{index}]")
    else:
        yield path, value


def require_finite_optimizer_state(
    optimizer: torch.optim.Optimizer,
    *,
    context: str = "optimizer state",
) -> None:
    """Fail immediately on a non-finite tensor/scalar in optimizer state."""

    for path, value in _iter_state_values(optimizer.state_dict(), context):
        if isinstance(value, torch.Tensor):
            finite = torch.isfinite(value)
            if not bool(finite.all().item()):
                count = int((~finite).sum().item())
                raise OptimizerStateNumericalError(
                    f"Expected {path} to contain only finite values, got "
                    f"{count} non-finite element(s)."
                )
        elif isinstance(value, bool) or value is None or isinstance(value, str):
            continue
        elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise OptimizerStateNumericalError(
                f"Expected {path} to be finite, got {value!r}."
            )


def _require_finite_parameter_state(
    params: tuple[Any, ...],
    *,
    context: str,
) -> None:
    for param in params:
        state = param.state
        finite = torch.isfinite(state)
        if not bool(finite.all().item()):
            count = int((~finite).sum().item())
            raise OptimizerStateNumericalError(
                f"Expected {context} for {str(param.name).strip()!r} to contain "
                f"only finite values, got {count} non-finite element(s)."
            )


def _optimizer_step_with_diagnostics(
    params: tuple[Any, ...],
    optimizer: torch.optim.Optimizer,
    *,
    optimizer_name: str,
    restore: bool = False,
    zero_proposal_epsilon: float = ZERO_PROPOSAL_EPSILON,
    unbounded_normalization_scales: Mapping[str, float] | None = None,
) -> OptimizerTransition:
    unbounded_names = {
        str(param.name).strip()
        for param in params
        if _parameter_kind(str(param.name).strip()) != "bounded_weight"
    }
    if unbounded_normalization_scales is None:
        normalization_scales = initial_unbounded_parameter_scales(params)
    else:
        if set(unbounded_normalization_scales) != unbounded_names:
            raise ValueError(
                "Expected unbounded_normalization_scales to contain exactly "
                f"{sorted(unbounded_names)!r}, got "
                f"{sorted(unbounded_normalization_scales)!r}."
            )
        normalization_scales = {}
        for name, raw_scale in unbounded_normalization_scales.items():
            try:
                scale = float(raw_scale)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Expected normalization scale for {name!r} to be positive "
                    f"and finite, got {raw_scale!r}."
                ) from exc
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    f"Expected normalization scale for {name!r} to be positive "
                    f"and finite, got {raw_scale!r}."
                )
            normalization_scales[name] = scale
    try:
        zero_proposal_epsilon = float(zero_proposal_epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Expected zero_proposal_epsilon to be a non-negative finite number, "
            f"got {zero_proposal_epsilon!r}."
        ) from exc
    if not math.isfinite(zero_proposal_epsilon) or zero_proposal_epsilon < 0.0:
        raise ValueError(
            "Expected zero_proposal_epsilon to be a non-negative finite number, "
            f"got {zero_proposal_epsilon!r}."
        )

    pre_update = tuple(_clone_state(param.state) for param in params)
    gradients = tuple(
        torch.zeros_like(param.state)
        if param.state.grad is None
        else _clone_state(param.state.grad)
        for param in params
    )
    for param, gradient in zip(params, gradients):
        if gradient.shape != param.state.shape:
            raise ValueError(
                f"Expected {str(param.name).strip()} gradient shape "
                f"{tuple(param.state.shape)}, got {tuple(gradient.shape)}."
            )

    snapshot = (
        capture_parameter_optimizer_state(params, optimizer) if restore else None
    )
    try:
        _require_finite_parameter_state(params, context="pre-optimizer state")
        require_finite_optimizer_state(
            optimizer, context=f"{optimizer_name} pre-step state"
        )
        optimizer.step()
        _require_finite_parameter_state(
            params, context="post-optimizer/pre-projection state"
        )
        require_finite_optimizer_state(
            optimizer, context=f"{optimizer_name} post-step state"
        )
        post_optimizer = tuple(_clone_state(param.state) for param in params)

        with torch.no_grad():
            for param in params:
                name = str(param.name).strip()
                if _parameter_kind(name) != "bounded_weight":
                    continue
                lower = float(param.min_cond)
                upper = float(param.max_cond)
                param.state.clamp_(min=lower, max=upper)
        _require_finite_parameter_state(params, context="post-projection state")
        post_projection = tuple(_clone_state(param.state) for param in params)

        transitions = []
        for param, gradient, before, proposed_state, projected_state in zip(
            params,
            gradients,
            pre_update,
            post_optimizer,
            post_projection,
        ):
            name = str(param.name).strip()
            kind = _parameter_kind(name)
            bounded = kind == "bounded_weight"
            proposed_delta = proposed_state - before
            projected_delta = projected_state - before
            gradient_rms = _rms(gradient)
            proposed_update_rms = _rms(proposed_delta)
            if bounded:
                lower = float(param.min_cond)
                upper = float(param.max_cond)
                normalized_update = proposed_update_rms / (upper - lower)
                proposed_bound_crossing_fraction = float(
                    ((proposed_state < lower) | (proposed_state > upper))
                    .to(dtype=torch.float64)
                    .mean()
                    .item()
                )
                lower_occupancy_mask = projected_state <= lower
                upper_occupancy_mask = projected_state >= upper
                lower_bound_occupancy = float(
                    lower_occupancy_mask.to(dtype=torch.float64).mean().item()
                )
                upper_bound_occupancy = float(
                    upper_occupancy_mask.to(dtype=torch.float64).mean().item()
                )
                combined_bound_occupancy = float(
                    (lower_occupancy_mask | upper_occupancy_mask)
                    .to(dtype=torch.float64)
                    .mean()
                    .item()
                )
            else:
                state_scale = normalization_scales[name]
                normalized_update = proposed_update_rms / state_scale
                proposed_bound_crossing_fraction = None
                lower_bound_occupancy = None
                upper_bound_occupancy = None
                combined_bound_occupancy = None

            proposed_norm = _l2_norm(proposed_delta)
            projected_norm = _l2_norm(projected_delta)
            projection_efficiency = projected_norm / (
                proposed_norm + PROJECTION_EFFICIENCY_EPSILON
            )
            proposal_is_zero = proposed_norm <= zero_proposal_epsilon
            transitions.append(
                OptimizerParameterTransition(
                    name=name,
                    parameter_kind=kind,
                    bounded_gate=bounded,
                    report_only=not bounded,
                    pre_update=before,
                    post_optimizer_pre_projection=proposed_state,
                    post_projection=projected_state,
                    gradient_rms=gradient_rms,
                    proposed_update_rms=proposed_update_rms,
                    normalized_update=normalized_update,
                    proposed_bound_crossing_fraction=(
                        proposed_bound_crossing_fraction
                    ),
                    lower_bound_occupancy=lower_bound_occupancy,
                    upper_bound_occupancy=upper_bound_occupancy,
                    combined_bound_occupancy=combined_bound_occupancy,
                    projection_efficiency=projection_efficiency,
                    proposal_is_numerically_zero=proposal_is_zero,
                    projection_gate_eligible=bounded and not proposal_is_zero,
                )
            )
    except Exception:
        if snapshot is not None:
            restore_parameter_optimizer_state(params, optimizer, snapshot)
        raise

    if snapshot is not None:
        restore_parameter_optimizer_state(params, optimizer, snapshot)
    return OptimizerTransition(
        parameters=tuple(transitions),
        restored=bool(restore),
        optimizer_name=optimizer_name,
    )


def optimizer_step_with_diagnostics(
    params: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    *,
    restore: bool = False,
    zero_proposal_epsilon: float = ZERO_PROPOSAL_EPSILON,
    unbounded_normalization_scales: Mapping[str, float] | None = None,
) -> OptimizerTransition:
    """Run a frozen SGD/Adam proposal, project, and capture generic diagnostics.

    Every scientific parameter must have its own optimizer group.  Gradients
    must already be assigned to ``param.state.grad``.  A restored shadow step
    returns detached transition tensors while putting parameter values,
    gradients, Adam moments/variances/step counters, and group settings back
    exactly.
    """

    values = _validate_params(params)
    optimizer_name = validate_optimizer(
        values,
        optimizer,
        optimizer_neutral=True,
    )
    return _optimizer_step_with_diagnostics(
        values,
        optimizer,
        optimizer_name=optimizer_name,
        restore=restore,
        zero_proposal_epsilon=zero_proposal_epsilon,
        unbounded_normalization_scales=unbounded_normalization_scales,
    )


def sgd_step_with_diagnostics(
    params: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    *,
    restore: bool = False,
    zero_proposal_epsilon: float = ZERO_PROPOSAL_EPSILON,
    unbounded_normalization_scales: Mapping[str, float] | None = None,
) -> SGDTransition:
    """Run the historical SGD transition and preserve its serialized fields."""

    values = _validate_params(params)
    _validate_optimizer(values, optimizer)
    generic = _optimizer_step_with_diagnostics(
        values,
        optimizer,
        optimizer_name="SGD",
        restore=restore,
        zero_proposal_epsilon=zero_proposal_epsilon,
        unbounded_normalization_scales=unbounded_normalization_scales,
    )
    return SGDTransition(
        parameters=tuple(
            ParameterTransition(
                name=item.name,
                parameter_kind=item.parameter_kind,
                bounded_gate=item.bounded_gate,
                report_only=item.report_only,
                pre_update=item.pre_update,
                post_sgd_pre_projection=item.post_optimizer_pre_projection,
                post_projection=item.post_projection,
                gradient_rms=item.gradient_rms,
                proposed_update_rms=item.proposed_update_rms,
                normalized_update=item.normalized_update,
                proposed_bound_crossing_fraction=(
                    item.proposed_bound_crossing_fraction
                ),
                lower_bound_occupancy=item.lower_bound_occupancy,
                upper_bound_occupancy=item.upper_bound_occupancy,
                combined_bound_occupancy=item.combined_bound_occupancy,
                projection_efficiency=item.projection_efficiency,
                proposal_is_numerically_zero=item.proposal_is_numerically_zero,
                projection_gate_eligible=item.projection_gate_eligible,
            )
            for item in generic.parameters
        ),
        restored=generic.restored,
    )


def shadow_sgd_step_with_diagnostics(
    params: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    *,
    zero_proposal_epsilon: float = ZERO_PROPOSAL_EPSILON,
    unbounded_normalization_scales: Mapping[str, float] | None = None,
) -> SGDTransition:
    """Run ``sgd_step_with_diagnostics`` and restore live training state."""

    return sgd_step_with_diagnostics(
        params,
        optimizer,
        restore=True,
        zero_proposal_epsilon=zero_proposal_epsilon,
        unbounded_normalization_scales=unbounded_normalization_scales,
    )


def shadow_optimizer_step_with_diagnostics(
    params: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    *,
    zero_proposal_epsilon: float = ZERO_PROPOSAL_EPSILON,
    unbounded_normalization_scales: Mapping[str, float] | None = None,
) -> OptimizerTransition:
    """Run an optimizer-neutral proposal and restore all mutable training state."""

    return optimizer_step_with_diagnostics(
        params,
        optimizer,
        restore=True,
        zero_proposal_epsilon=zero_proposal_epsilon,
        unbounded_normalization_scales=unbounded_normalization_scales,
    )


__all__ = [
    "BIAS_NORMALIZATION_EPSILON",
    "OptimizerParameterTransition",
    "OptimizerStateNumericalError",
    "OptimizerTransition",
    "PROJECTION_EFFICIENCY_EPSILON",
    "ParameterOptimizerSnapshot",
    "ParameterTransition",
    "SGDTransition",
    "ZERO_PROPOSAL_EPSILON",
    "capture_parameter_optimizer_state",
    "initial_unbounded_parameter_scales",
    "optimizer_step_with_diagnostics",
    "require_finite_optimizer_state",
    "restore_parameter_optimizer_state",
    "sgd_step_with_diagnostics",
    "shadow_optimizer_step_with_diagnostics",
    "shadow_sgd_step_with_diagnostics",
    "validate_optimizer",
]
