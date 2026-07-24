"""Tiki-Taka gradient accumulation for EqProp/DRN parameters.

The DRN differentiators produce a complete gradient tensor for every model
parameter.  This module treats those tensors as writes to a hidden, signed
gradient crossbar and periodically transfers slices of that crossbar to the
visible model weights.

This is an ideal tensor implementation of AIHWKit's ``TransferCompound``
update pipeline.  It intentionally does not simulate pulse generation,
device-to-device variation, read noise, or asymmetric device response.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from math import isfinite, prod
from numbers import Integral, Real
from typing import Optional, Union

import torch

from model.variable.parameter import Bias, ConvWeight, DenseWeight, PoolWeight


@dataclass(frozen=True)
class TikiTakaConfig:
    """Configuration for the hidden-gradient-crossbar update pipeline.

    ``transfer_every`` counts optimizer steps, which correspond to minibatches
    in the existing DRN trainers.  The auxiliary array stores the negative
    gradient/update direction.  Consequently, transfers add selected
    auxiliary slices to the visible weights.
    """

    fast_lr: float = 1.0
    transfer_every: int = 1
    units_in_mbatch: bool = True
    n_reads_per_transfer: int = 1
    gamma: float = 0.0
    transfer_lr: float = 1.0
    scale_transfer_lr: bool = True
    transfer_columns: bool = True
    with_reset_prob: float = 0.0
    random_selection: bool = False
    fast_weight_min: Optional[float] = None
    fast_weight_max: Optional[float] = None
    accumulate_biases: bool = False

    def __post_init__(self) -> None:
        _require_real("fast_lr", self.fast_lr, lower=0.0)
        _require_integer("transfer_every", self.transfer_every, lower=0)
        _require_bool("units_in_mbatch", self.units_in_mbatch)
        _require_integer("n_reads_per_transfer", self.n_reads_per_transfer, lower=1)
        _require_real("gamma", self.gamma)
        _require_real("transfer_lr", self.transfer_lr, lower=0.0)
        _require_bool("scale_transfer_lr", self.scale_transfer_lr)
        _require_bool("transfer_columns", self.transfer_columns)
        _require_real(
            "with_reset_prob",
            self.with_reset_prob,
            lower=0.0,
            upper=1.0,
        )
        _require_bool("random_selection", self.random_selection)
        _require_optional_real("fast_weight_min", self.fast_weight_min)
        _require_optional_real("fast_weight_max", self.fast_weight_max)
        _require_bool("accumulate_biases", self.accumulate_biases)
        if not self.units_in_mbatch:
            raise ValueError(
                "Expected update_pipeline.units_in_mbatch to be true because "
                "server_code supplies one averaged EqProp gradient per minibatch. "
                f"Provided value: {self.units_in_mbatch!r}."
            )
        if self.gamma != 0.0:
            raise ValueError(
                "Expected update_pipeline.gamma to be 0.0 so the main DRN "
                "crossbar is the only array visible to forward and inference. "
                f"Provided value: {self.gamma!r}."
            )
        if not self.transfer_columns and self.with_reset_prob > 0.0:
            raise ValueError(
                "Expected update_pipeline.with_reset_prob to be 0.0 when "
                "transfer_columns is false; AIHWKit resets transferred columns "
                "only. "
                f"Provided value: transfer_columns={self.transfer_columns!r}, "
                f"with_reset_prob={self.with_reset_prob!r}."
            )
        if (
            self.fast_weight_min is not None
            and self.fast_weight_max is not None
            and self.fast_weight_min >= self.fast_weight_max
        ):
            raise ValueError(
                "Expected fast_weight_min to be smaller than fast_weight_max. "
                f"Provided value: min={self.fast_weight_min!r}, "
                f"max={self.fast_weight_max!r}."
            )


def _require_bool(name: str, value) -> None:
    if not isinstance(value, bool):
        raise ValueError(
            f"Expected update_pipeline.{name} to be a boolean. "
            f"Provided value: {value!r}."
        )


def _require_integer(name: str, value, *, lower: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
        raise ValueError(
            f"Expected update_pipeline.{name} to be an integer >= {lower}. "
            f"Provided value: {value!r}."
        )


def _require_real(
    name: str,
    value,
    *,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    lower_inclusive: bool = True,
) -> None:
    valid = (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and isfinite(float(value))
    )
    if valid and lower is not None:
        valid = value >= lower if lower_inclusive else value > lower
    if valid and upper is not None:
        valid = value <= upper
    if valid:
        return

    if lower is not None and upper is not None:
        expectation = f"a number in [{lower}, {upper}]"
    elif lower is not None and lower_inclusive:
        expectation = f"a number >= {lower}"
    elif lower is not None:
        expectation = f"a number > {lower}"
    else:
        expectation = "a number"
    raise ValueError(
        f"Expected update_pipeline.{name} to be {expectation}. "
        f"Provided value: {value!r}."
    )


def _require_optional_real(name: str, value) -> None:
    valid = (
        value is None
        or (
            not isinstance(value, bool)
            and isinstance(value, Real)
            and isfinite(float(value))
        )
    )
    if not valid:
        raise ValueError(
            f"Expected update_pipeline.{name} to be a number or null. "
            f"Provided value: {value!r}."
        )


def parse_update_pipeline(
    value: Optional[Union[TikiTakaConfig, Mapping]],
) -> Optional[TikiTakaConfig]:
    """Parse the optional ``update_pipeline`` configuration block."""

    if value is None or isinstance(value, TikiTakaConfig):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(
            "Expected update_pipeline to be an object with "
            "'type': 'direct' or 'type': 'tiki_taka'. "
            f"Provided value: {value!r}."
        )

    pipeline_type = value.get("type")
    if pipeline_type == "direct":
        if len(value) != 1:
            raise ValueError(
                "Expected update_pipeline with type='direct' to contain no "
                "additional keys. "
                f"Provided value: {value!r}."
            )
        return None
    if pipeline_type != "tiki_taka":
        raise ValueError(
            "Expected update_pipeline.type to be 'direct' or 'tiki_taka'. "
            f"Provided value: {pipeline_type!r}."
        )

    config_values = {key: item for key, item in value.items() if key != "type"}
    allowed = {field.name for field in fields(TikiTakaConfig)}
    unknown = sorted(set(config_values) - allowed)
    if unknown:
        raise ValueError(
            "Expected update_pipeline keys to be drawn from: "
            f"{', '.join(sorted(allowed | {'type'}))}. "
            f"Provided value: unknown keys {unknown!r} in {value!r}."
        )
    return TikiTakaConfig(**config_values)


def _crossbar_layout(parameter) -> tuple[str, Optional[int]]:
    if isinstance(parameter, DenseWeight):
        return "dense", prod(parameter._layer_pre_shape)
    if isinstance(parameter, ConvWeight):
        return "conv", None
    if isinstance(parameter, Bias):
        return "bias", None
    return "flat", None


def _as_crossbar_matrix(
    tensor: torch.Tensor,
    layout: str,
    dense_input_size: Optional[int],
) -> torch.Tensor:
    """Return a logical ``[out, in]`` view of a parameter tensor."""

    if layout == "dense":
        return tensor.reshape(dense_input_size, -1).transpose(0, 1)
    if layout == "conv":
        return tensor.reshape(tensor.shape[0], -1)
    if layout == "bias":
        return tensor.reshape(-1, 1)
    return tensor.reshape(1, -1)


class TikiTakaOptimizer(torch.optim.Optimizer):
    """Optimizer implementing a two-crossbar Tiki-Taka update pipeline.

    Dense and convolutional weights use the auxiliary crossbar.  Biases use
    direct digital SGD unless ``accumulate_biases`` is enabled.  Pooling
    weights remain frozen, matching :class:`training.monitor.Optimizer`.

    The auxiliary state and its per-parameter transfer counters/cursors live in
    the standard PyTorch optimizer state, so ``state_dict()`` preserves them.
    """

    def __init__(
        self,
        energy_fn,
        cost_fn,
        learning_rates,
        config: Union[TikiTakaConfig, Mapping],
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        provided_config = config
        config = parse_update_pipeline(config)
        if config is None:
            raise ValueError(
                "Expected a TikiTakaConfig or Tiki-Taka update_pipeline object. "
                f"Provided value: {provided_config!r}."
            )
        if momentum != 0.0 or weight_decay != 0.0:
            raise ValueError(
                "Expected momentum=0 and weight_decay=0 for Tiki-Taka updates; "
                "their placement in the two-crossbar pipeline is undefined. "
                f"Provided value: momentum={momentum!r}, "
                f"weight_decay={weight_decay!r}."
            )

        parameters = [
            parameter
            for parameter in energy_fn.params() + cost_fn.params()
            if not isinstance(parameter, PoolWeight)
        ]
        if len(learning_rates) != len(parameters):
            raise ValueError(
                f"Expected {len(parameters)} learning rates after filtering "
                "PoolWeight parameters. "
                f"Provided value: {len(learning_rates)} learning rates."
            )

        parameter_groups = []
        for parameter, learning_rate in zip(parameters, learning_rates):
            _require_real("learning_rate", learning_rate, lower=0.0)
            layout, dense_input_size = _crossbar_layout(parameter)
            use_auxiliary = not isinstance(parameter, Bias) or config.accumulate_biases
            parameter_groups.append(
                {
                    "params": [parameter.state],
                    "lr": float(learning_rate),
                    "use_tiki_taka": use_auxiliary,
                    "crossbar_layout": layout,
                    "dense_input_size": dense_input_size,
                }
            )

        defaults = {"lr": 0.1}
        super().__init__(parameter_groups, defaults)
        self._learning_rates = list(learning_rates)
        self._config = config

    @property
    def config(self) -> TikiTakaConfig:
        """Return the immutable Tiki-Taka configuration."""

        return self._config

    def auxiliary_state(self, parameter: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the live auxiliary array for a parameter tensor, if created."""

        return self.state[parameter].get("auxiliary")

    def state_dict(self) -> dict:
        """Serialize auxiliary arrays, transfer progress, and pipeline config."""

        state_dict = super().state_dict()
        state_dict["tiki_taka_config"] = asdict(self._config)
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore state while guarding against a different transfer pipeline."""

        saved_config = state_dict.get("tiki_taka_config")
        current_config = asdict(self._config)
        if saved_config is not None and saved_config != current_config:
            raise ValueError(
                "Expected optimizer state with the same Tiki-Taka configuration "
                "as the current optimizer. "
                f"Provided value: saved={saved_config!r}, "
                f"current={current_config!r}."
            )
        optimizer_state = {
            key: value
            for key, value in state_dict.items()
            if key != "tiki_taka_config"
        }
        return super().load_state_dict(optimizer_state)

    def _initialize_auxiliary_state(self, parameter: torch.Tensor) -> dict:
        state = self.state[parameter]
        if "auxiliary" not in state:
            state["auxiliary"] = torch.zeros_like(
                parameter,
                memory_format=torch.preserve_format,
            )
            state["step"] = 0
            state["transfer_index"] = 0
            state["transfer_count"] = 0
        return state

    def _transfer_slices(self, parameter: torch.Tensor, group: dict, state: dict) -> None:
        config = self._config
        visible = _as_crossbar_matrix(
            parameter,
            group["crossbar_layout"],
            group["dense_input_size"],
        )
        auxiliary = _as_crossbar_matrix(
            state["auxiliary"],
            group["crossbar_layout"],
            group["dense_input_size"],
        )

        axis = 1 if config.transfer_columns else 0
        axis_size = visible.shape[axis]
        n_reads = min(config.n_reads_per_transfer, axis_size)
        if config.random_selection:
            start = int(torch.randint(axis_size, (), device=parameter.device).item())
        else:
            start = int(state["transfer_index"])
        indices = [(start + offset) % axis_size for offset in range(n_reads)]

        transfer_scale = config.transfer_lr
        if config.scale_transfer_lr:
            transfer_scale *= group["lr"]

        if transfer_scale != 0.0:
            for index in indices:
                if config.transfer_columns:
                    visible[:, index].add_(auxiliary[:, index], alpha=transfer_scale)
                else:
                    visible[index, :].add_(auxiliary[index, :], alpha=transfer_scale)

                if config.with_reset_prob > 0.0:
                    should_reset = config.with_reset_prob == 1.0 or bool(
                        torch.rand((), device=parameter.device) < config.with_reset_prob
                    )
                    if should_reset:
                        if config.transfer_columns:
                            auxiliary[:, index].zero_()
                        else:
                            auxiliary[index, :].zero_()

        state["transfer_index"] = (start + n_reads) % axis_size
        state["transfer_count"] += 1

    @torch.no_grad()
    def step(self, closure=None):
        """Write gradients to auxiliary arrays and perform due transfers."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("TikiTakaOptimizer does not support sparse gradients")

                if not group["use_tiki_taka"]:
                    parameter.add_(gradient, alpha=-group["lr"])
                    continue

                state = self._initialize_auxiliary_state(parameter)
                auxiliary = state["auxiliary"]
                fast_learning_rate = self._config.fast_lr
                if fast_learning_rate == 0.0:
                    fast_learning_rate = group["lr"]
                auxiliary.add_(gradient, alpha=-fast_learning_rate)
                if (
                    self._config.fast_weight_min is not None
                    or self._config.fast_weight_max is not None
                ):
                    auxiliary.clamp_(
                        min=self._config.fast_weight_min,
                        max=self._config.fast_weight_max,
                    )

                state["step"] += 1
                transfer_every = self._config.transfer_every
                if transfer_every > 0 and state["step"] % transfer_every == 0:
                    self._transfer_slices(parameter, group, state)

        return loss

    def __str__(self) -> str:
        config = self._config
        direction = "columns" if config.transfer_columns else "rows"
        return (
            "Tiki-Taka -- initial learning rates = "
            f"{self._learning_rates}, fast_lr={config.fast_lr}, "
            f"transfer_every={config.transfer_every}, "
            f"n_reads_per_transfer={config.n_reads_per_transfer}, "
            f"transfer_lr={config.transfer_lr}, direction={direction}"
        )


def build_optimizer(
    energy_fn,
    cost_fn,
    learning_rates,
    *,
    update_pipeline=None,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
):
    """Build the existing SGD optimizer or the optional Tiki-Taka optimizer."""

    config = parse_update_pipeline(update_pipeline)
    if config is None:
        # Import lazily to avoid coupling Monitor's TensorBoard setup to users
        # that import only the Tiki-Taka configuration helpers.
        from training.monitor import Optimizer

        return Optimizer(
            energy_fn,
            cost_fn,
            learning_rates,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    return TikiTakaOptimizer(
        energy_fn,
        cost_fn,
        learning_rates,
        config=config,
        momentum=momentum,
        weight_decay=weight_decay,
    )
