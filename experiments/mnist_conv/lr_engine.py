"""Numerical engine used by the staged Conv learning-rate study.

The module is intentionally separate from the generic v1 training backend.  It
keeps the official MNIST test split out of reach, uses one resettable training
stream, and exposes the optimizer transition before and after conductance
projection on every step.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .lr_step import (
    OptimizerTransition,
    SGDTransition,
    initial_unbounded_parameter_scales,
    optimizer_step_with_diagnostics,
    require_finite_optimizer_state,
    sgd_step_with_diagnostics,
    validate_optimizer,
)


class LRStudyNumericalError(FloatingPointError):
    """Raised when a study step contains a non-finite scientific value."""


LearningRateValue = float | Mapping[str, float]
TransitionValue = SGDTransition | OptimizerTransition


FROZEN_SGD_OPTIMIZER = {
    "name": "SGD",
    "momentum": 0.0,
    "weight_decay": 0.0,
}

FROZEN_ADAM_OPTIMIZER = {
    "name": "Adam",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
    "foreach": False,
    "fused": False,
    "maximize": False,
    "capturable": False,
    "differentiable": False,
}


def _finite_scalar(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LRStudyNumericalError(
            f"Expected {label} to be finite. Provided value: {value!r}."
        ) from exc
    if not math.isfinite(result):
        raise LRStudyNumericalError(
            f"Expected {label} to be finite. Provided value: {value!r}."
        )
    return result


def _require_finite_tensor(tensor: Any, label: str) -> None:
    import torch

    if not torch.is_tensor(tensor):
        raise TypeError(
            f"Expected {label} to be a torch.Tensor. "
            f"Provided value: {type(tensor).__name__}."
        )
    if not bool(torch.isfinite(tensor).all().item()):
        count = int((~torch.isfinite(tensor)).sum().item())
        raise LRStudyNumericalError(
            f"Expected {label} to contain only finite values. "
            f"Provided value: {count} non-finite element(s)."
        )


def _strict_zero(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Expected {label} to be exactly 0.0. Provided value: {value!r}."
        )
    result = float(value)
    if not math.isfinite(result) or result != 0.0:
        raise ValueError(
            f"Expected {label} to be exactly 0.0. Provided value: {value!r}."
        )
    return result


def validate_explicit_optimizer_contract(
    optimizer_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonicalize the strict v7 SGD/Adam tagged union."""

    if not isinstance(optimizer_contract, Mapping):
        raise TypeError(
            "Expected optimizer_contract to be an SGD or Adam mapping. "
            f"Provided value: {type(optimizer_contract).__name__}."
        )
    provided = dict(optimizer_contract)
    name = provided.get("name")
    if name == "SGD":
        expected_keys = set(FROZEN_SGD_OPTIMIZER)
        if set(provided) != expected_keys:
            raise ValueError(
                "Expected explicit SGD optimizer keys to be exactly "
                f"{sorted(expected_keys)!r}. Provided value: "
                f"{sorted(provided)!r}."
            )
        _strict_zero(provided["momentum"], "optimizer_contract.momentum")
        _strict_zero(provided["weight_decay"], "optimizer_contract.weight_decay")
        return dict(FROZEN_SGD_OPTIMIZER)
    if name == "Adam":
        expected_keys = set(FROZEN_ADAM_OPTIMIZER)
        if set(provided) != expected_keys:
            raise ValueError(
                "Expected explicit Adam optimizer keys to be exactly "
                f"{sorted(expected_keys)!r}. Provided value: "
                f"{sorted(provided)!r}."
            )
        betas = provided["betas"]
        if (
            not isinstance(betas, list)
            or len(betas) != 2
            or any(isinstance(value, bool) for value in betas)
            or not all(isinstance(value, (int, float)) for value in betas)
            or [float(value) for value in betas] != [0.9, 0.999]
        ):
            raise ValueError(
                "Expected optimizer_contract.betas to be exactly [0.9, 0.999]. "
                f"Provided value: {betas!r}."
            )
        eps = provided["eps"]
        if (
            isinstance(eps, bool)
            or not isinstance(eps, (int, float))
            or not math.isfinite(float(eps))
            or float(eps) != 1e-8
        ):
            raise ValueError(
                "Expected optimizer_contract.eps to be exactly 1e-8. "
                f"Provided value: {eps!r}."
            )
        _strict_zero(provided["weight_decay"], "optimizer_contract.weight_decay")
        for key in (
            "amsgrad",
            "foreach",
            "fused",
            "maximize",
            "capturable",
            "differentiable",
        ):
            if provided[key] is not False:
                raise ValueError(
                    f"Expected optimizer_contract.{key} to be exactly False. "
                    f"Provided value: {provided[key]!r}."
                )
        return {
            "name": "Adam",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
        }
    raise ValueError(
        "Expected optimizer_contract.name to be exactly 'SGD' or 'Adam'. "
        f"Provided value: {name!r}."
    )


def canonical_parameter_name(parameter: Any) -> str:
    name = str(getattr(parameter, "name", "")).strip()
    if not name:
        raise ValueError(
            f"Expected every parameter to have a non-empty name. Provided value: {name!r}."
        )
    return name


def parameter_tensor_digest(parameters: Iterable[Any]) -> str:
    """Hash ordered parameter names, types, shapes, dtypes, and exact bytes."""

    digest = hashlib.sha256()
    digest.update(b"mnist-conv-parameter-tensors/v1\0")
    values = tuple(parameters)
    digest.update(len(values).to_bytes(8, byteorder="big", signed=False))
    for parameter in values:
        tensor = parameter.state.detach().cpu().contiguous()
        name = canonical_parameter_name(parameter).encode("utf-8")
        type_name = (
            f"{type(parameter).__module__}.{type(parameter).__qualname__}"
        ).encode("utf-8")
        shape = ",".join(str(int(item)) for item in tensor.shape).encode("ascii")
        dtype = str(tensor.dtype).encode("ascii")
        for item in (name, type_name, shape, dtype):
            digest.update(len(item).to_bytes(8, byteorder="big", signed=False))
            digest.update(item)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def parameter_state_diagnostics(parameters: Iterable[Any]) -> dict[str, dict[str, Any]]:
    """Return checkpoint occupancy and scale diagnostics by parameter name."""

    import torch

    result: dict[str, dict[str, Any]] = {}
    for parameter in parameters:
        name = canonical_parameter_name(parameter)
        state = parameter.state.detach()
        _require_finite_tensor(state, f"initial parameter {name!r}")
        rms = float(torch.sqrt(torch.mean(state.to(torch.float64) ** 2)).item())
        std = float(state.to(torch.float64).std(unbiased=False).item())
        bounded = name.startswith(("ConvWeight_", "DenseWeight_"))
        record: dict[str, Any] = {
            "name": name,
            "parameter_type": type(parameter).__name__,
            "shape": [int(item) for item in state.shape],
            "rms": rms,
            "std": std,
            "bounded_gate": bounded,
            "report_only": not bounded,
        }
        if bounded:
            lower = float(parameter.min_cond)
            upper = float(parameter.max_cond)
            lower_mask = state <= lower
            upper_mask = state >= upper
            record.update(
                {
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "lower_bound_occupancy": float(
                        lower_mask.to(torch.float64).mean().item()
                    ),
                    "upper_bound_occupancy": float(
                        upper_mask.to(torch.float64).mean().item()
                    ),
                    "combined_bound_occupancy": float(
                        (lower_mask | upper_mask).to(torch.float64).mean().item()
                    ),
                }
            )
        result[name] = record
    return result


def _architecture_geometry(
    architecture: Mapping[str, Any],
) -> tuple[list[tuple[int, ...]], list[dict[str, Any]]]:
    channels = [int(value) for value in architecture["channels"]]
    kernels = [int(value) for value in architecture["kernel_sizes"]]
    strides = [int(value) for value in architecture["strides"]]
    paddings = [int(value) for value in architecture["paddings"]]
    if not (
        len(channels) == len(kernels) == len(strides) == len(paddings)
        and len(channels) in (1, 2, 3)
    ):
        raise ValueError(
            "Expected one, two, or three aligned convolution stages. "
            f"Provided value: channels={channels!r}, kernels={kernels!r}, "
            f"strides={strides!r}, paddings={paddings!r}."
        )
    height = width = 28
    layer_shapes: list[tuple[int, ...]] = [(2, 28, 28)]
    pipeline: list[dict[str, Any]] = []
    previous_channels = 2
    for output_channels, kernel, stride, padding in zip(
        channels, kernels, strides, paddings
    ):
        height = (height + 2 * padding - kernel) // stride + 1
        width = (width + 2 * padding - kernel) // stride + 1
        if height <= 0 or width <= 0:
            raise ValueError(
                "Expected convolution geometry to stay positive. "
                f"Provided value: height={height}, width={width}."
            )
        layer_shapes.append((output_channels, height, width))
        pipeline.append(
            {
                "kernel": [kernel, kernel],
                "stride": stride,
                "padding": padding,
                "mode": "convolution",
                "input_channels": previous_channels,
                "output_channels": output_channels,
            }
        )
        previous_channels = output_channels
    layer_shapes.append((20,))
    for item in pipeline:
        item.pop("input_channels")
        item.pop("output_channels")
    return layer_shapes, pipeline


@dataclass
class LRModelRuntime:
    row: Mapping[str, Any]
    device: Any
    energy_fn: Any
    network: Any
    free_layers: list[Any]
    cost_fn: Any
    minimizer_inference: Any
    minimizer_training: Any
    estimator: Any
    parameters: list[Any]
    optimizer: Any
    unbounded_normalization_scales: dict[str, float]
    optimizer_name: str = "SGD"
    optimizer_neutral_transitions: bool = False

    def set_learning_rate(self, learning_rate: LearningRateValue) -> None:
        """Set either one scalar rate or one explicit rate per parameter.

        The mapping form is used by the layer-wise study.  It is deliberately
        keyed by canonical scientific parameter names rather than optimizer
        group positions so that a missing, extra, or mis-associated bias fails
        loudly.
        """

        names = [canonical_parameter_name(parameter) for parameter in self.parameters]
        if isinstance(learning_rate, Mapping):
            provided = {str(name): value for name, value in learning_rate.items()}
            if set(provided) != set(names):
                raise ValueError(
                    "Expected layer-wise learning rates to contain exactly "
                    f"{sorted(names)!r}. Provided value: {sorted(provided)!r}."
                )
            values = []
            for name in names:
                value = _finite_scalar(
                    provided[name], f"learning rate for parameter {name!r}"
                )
                if value < 0.0:
                    raise ValueError(
                        "Expected every layer-wise learning rate to be non-negative. "
                        f"Provided value for {name!r}: {value!r}."
                    )
                values.append(value)
        else:
            value = _finite_scalar(learning_rate, "learning rate")
            if value < 0.0:
                raise ValueError(
                    f"Expected learning rate to be non-negative. Provided value: {value!r}."
                )
            values = [value] * len(names)
        if len(self.optimizer.param_groups) != len(values):
            raise RuntimeError(
                "Expected one optimizer parameter group per scientific parameter. "
                f"Provided value: groups={len(self.optimizer.param_groups)}, "
                f"parameters={len(values)}."
            )
        value_by_tensor_id = {
            id(parameter.state): value
            for parameter, value in zip(self.parameters, values)
        }
        seen_tensor_ids: set[int] = set()
        for group_index, group in enumerate(self.optimizer.param_groups):
            tensors = list(group.get("params", ()))
            if len(tensors) != 1 or id(tensors[0]) not in value_by_tensor_id:
                raise RuntimeError(
                    "Expected each optimizer group to contain exactly one known "
                    f"scientific parameter tensor. Provided value for group "
                    f"{group_index}: {len(tensors)} tensor(s)."
                )
            tensor_id = id(tensors[0])
            if tensor_id in seen_tensor_ids:
                raise RuntimeError(
                    "Expected every scientific parameter tensor in exactly one "
                    f"optimizer group. Provided duplicate group: {group_index}."
                )
            seen_tensor_ids.add(tensor_id)
            group["lr"] = value_by_tensor_id[tensor_id]
        if seen_tensor_ids != set(value_by_tensor_id):
            raise RuntimeError(
                "Expected optimizer groups to cover every scientific parameter tensor."
            )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.energy_fn.save(target)
        return target


def build_model_runtime(
    study: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    device: str,
    initialization_checkpoint: str | Path | None = None,
    learning_rate: LearningRateValue = 1.0,
    optimizer_contract: Mapping[str, Any] | None = None,
) -> LRModelRuntime:
    """Build one reset-counter, seed-0 Conv runtime under a frozen contract.

    Omitting ``optimizer_contract`` takes the untouched v1--v6 SGD path from
    ``study["optimizer"]``.  V7 callers pass an exact tagged-union arm
    explicitly, so optimizer choice cannot alter any older study identity.
    """

    import torch

    from labs.custom_classes import FlexibleDeepResistiveEnergy
    from labs.mnist_train import _build_tracking_minimizer, _reset_name_counters, _set_seed
    from model.function.cost import SquaredErrorPairedOutputs
    from model.function.network import Network
    from training.monitor import Optimizer
    from training.sgd import Backprop

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Expected requested CUDA device to be available. Provided value: {device!r}."
        )
    model = study["model"]
    explicit_optimizer_contract = optimizer_contract is not None
    selected_optimizer_contract = (
        validate_explicit_optimizer_contract(optimizer_contract)
        if explicit_optimizer_contract
        else study["optimizer"]
    )
    optimizer_contract = selected_optimizer_contract
    scalar_contract = (
        optimizer_contract.get("name") == "SGD"
        and float(optimizer_contract.get("momentum")) == 0.0
        and float(optimizer_contract.get("weight_decay")) == 0.0
        and optimizer_contract.get("scalar_lr_for_all_trainable_parameters") is True
        and optimizer_contract.get("bounded_weights_set_gates") is True
        and optimizer_contract.get("biases_reported_separately") is True
    )
    layerwise_contract = (
        optimizer_contract.get("name") == "SGD"
        and float(optimizer_contract.get("momentum")) == 0.0
        and float(optimizer_contract.get("weight_decay")) == 0.0
        and optimizer_contract.get("scalar_lr_for_all_trainable_parameters") is False
        and optimizer_contract.get("layerwise_lr_for_bounded_weights") is True
        and optimizer_contract.get("bias_lr_policy")
        == "match_associated_conv_weight"
        and optimizer_contract.get("biases_excluded_from_lr_tuning") is True
        and optimizer_contract.get("bounded_weights_set_gates") is True
        and optimizer_contract.get("biases_reported_separately") is True
    )
    architecture_relative_contract = (
        study.get("schema_version") == "mnist-conv-lr-study/v5"
        and optimizer_contract.get("name") == "SGD"
        and float(optimizer_contract.get("momentum")) == 0.0
        and float(optimizer_contract.get("weight_decay")) == 0.0
        and optimizer_contract.get("alpha_scope") == "architecture"
        and optimizer_contract.get("normalization_statistic") == "median"
        and optimizer_contract.get("scalar_lr_for_all_trainable_parameters") is False
        and optimizer_contract.get("weight_specific_learning_rates") is True
        and optimizer_contract.get("bias_lr_policy")
        == "match_associated_conv_weight"
        and optimizer_contract.get("biases_excluded_from_lr_tuning") is True
        and optimizer_contract.get("bounded_weights_set_gates") is True
        and optimizer_contract.get("biases_reported_separately") is True
    )
    two_rho_contract = (
        study.get("schema_version")
        in {"mnist-conv-lr-study/v6", "mnist-conv-lr-study/v7"}
        and optimizer_contract.get("name") == "SGD"
        and float(optimizer_contract.get("momentum")) == 0.0
        and float(optimizer_contract.get("weight_decay")) == 0.0
        and optimizer_contract.get("target_scope") == "conv_and_dense"
        and optimizer_contract.get("normalization_statistic") == "median"
        and optimizer_contract.get("scalar_lr_for_all_trainable_parameters") is False
        and optimizer_contract.get("weight_specific_learning_rates") is True
        and optimizer_contract.get("bias_lr_policy")
        == "match_associated_conv_weight"
        and optimizer_contract.get("biases_excluded_from_lr_tuning") is True
        and optimizer_contract.get("bounded_weights_set_gates") is True
        and optimizer_contract.get("biases_reported_separately") is True
    )
    explicit_contract = (
        explicit_optimizer_contract
        and optimizer_contract.get("name") in {"SGD", "Adam"}
    )
    if not (
        explicit_contract
        or scalar_contract
        or layerwise_contract
        or architecture_relative_contract
        or two_rho_contract
    ):
        raise ValueError(
            "Expected the frozen SGD optimizer contract. "
            f"Provided value: {optimizer_contract!r}."
        )

    _set_seed(int(model["model_seed"]))
    _reset_name_counters()
    architecture = model["architectures"][row["architecture"]]
    layer_shapes, pipeline = _architecture_geometry(architecture)
    solver = study["solver"]
    bounds = model["conductance_bounds"]
    number_of_weights = len(architecture["channels"]) + 1
    model_cfg = {
        "non_linearity": model["non_linearity"],
        "quadratic_diode_param": dict(model["quadratic_diode_param"]),
        "exponential_diode_param": dict(model["exponential_diode_param"]),
        "hard_sigmoid_param": dict(model["hard_sigmoid"]),
        "minimizer": dict(solver["minimizer"]),
    }
    energy_fn = FlexibleDeepResistiveEnergy(
        layer_shapes=layer_shapes,
        conv_pipeline=pipeline,
        pooling_mode=None,
        weight_gains=[float(model["weight_gains"])] * number_of_weights,
        input_gain=float(row["input_gain"]),
        non_linearity=model["non_linearity"],
        exponential_diode_param=model_cfg["exponential_diode_param"],
        quadratic_diode_param=model_cfg["quadratic_diode_param"],
        hard_sigmoid_param=model_cfg["hard_sigmoid_param"],
        voltage_amp=float(row["voltage_amp"]),
        current_amp=float(row["current_amp"]),
        weight_min=float(bounds[0]),
        weight_max=float(bounds[1]),
        weight_init_mode=model["weight_initialization"],
        input_mode="train",
        trainable_amplification=False,
        amplification_min=1e-6,
        amplification_max=None,
    )
    energy_fn.set_device(requested)
    if initialization_checkpoint is not None:
        checkpoint_path = Path(initialization_checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Expected initialization checkpoint to exist. "
                f"Provided value: {checkpoint_path}."
            )
        energy_fn.load(checkpoint_path)

    network = Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]
    cost_fn = SquaredErrorPairedOutputs(output_layer, num_classes=10)
    energy_mode = solver["energy_mode"]
    minimizer_inference = _build_tracking_minimizer(
        energy_fn,
        free_layers,
        model_cfg,
        energy_mode,
        num_iterations=int(row["inference_iterations"]),
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )
    minimizer_training = _build_tracking_minimizer(
        energy_fn,
        free_layers,
        model_cfg,
        energy_mode,
        num_iterations=int(row["training_iterations"]),
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )
    parameters = list(energy_fn.params())
    estimator = Backprop(
        parameters, free_layers, cost_fn, minimizer_training
    )
    if isinstance(learning_rate, Mapping):
        names = [canonical_parameter_name(parameter) for parameter in parameters]
        provided = {str(name): value for name, value in learning_rate.items()}
        if set(provided) != set(names):
            raise ValueError(
                "Expected initial layer-wise learning rates to contain exactly "
                f"{sorted(names)!r}. Provided value: {sorted(provided)!r}."
            )
        rates = [
            _finite_scalar(provided[name], f"learning rate for parameter {name!r}")
            for name in names
        ]
        if any(value < 0.0 for value in rates):
            raise ValueError(
                "Expected initial layer-wise learning rates to be non-negative. "
                f"Provided value: {provided!r}."
            )
    else:
        rate = _finite_scalar(learning_rate, "learning rate")
        if rate < 0.0:
            raise ValueError(
                f"Expected learning rate to be non-negative. Provided value: {rate!r}."
            )
        rates = [rate] * len(parameters)
    optimizer_name = str(optimizer_contract["name"])
    if optimizer_name == "SGD":
        optimizer = Optimizer(
            energy_fn,
            cost_fn,
            rates,
            momentum=0.0,
            weight_decay=0.0,
        )
    else:
        optimizer = torch.optim.Adam(
            [
                {"params": [parameter.state], "lr": rate}
                for parameter, rate in zip(parameters, rates)
            ],
            lr=1.0,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
            foreach=False,
            maximize=False,
            capturable=False,
            differentiable=False,
            fused=False,
        )
    validate_optimizer(parameters, optimizer, optimizer_neutral=True)
    runtime = LRModelRuntime(
        row=row,
        device=requested,
        energy_fn=energy_fn,
        network=network,
        free_layers=free_layers,
        cost_fn=cost_fn,
        minimizer_inference=minimizer_inference,
        minimizer_training=minimizer_training,
        estimator=estimator,
        parameters=parameters,
        optimizer=optimizer,
        unbounded_normalization_scales=initial_unbounded_parameter_scales(
            parameters
        ),
        optimizer_name=optimizer_name,
        optimizer_neutral_transitions=explicit_optimizer_contract,
    )
    runtime.set_learning_rate(learning_rate)
    for parameter in parameters:
        _require_finite_tensor(
            parameter.state, f"initial parameter {canonical_parameter_name(parameter)!r}"
        )
    require_finite_optimizer_state(
        runtime.optimizer,
        context=f"{optimizer_name} initial state",
    )
    return runtime


def build_loader_bundle(
    study: Mapping[str, Any],
    *,
    data_root: str | Path,
    download: bool,
    return_source_indices: bool = True,
):
    """Build only MNIST-train loaders; the official test split is never instantiated."""

    from labs.datasets import build_mnist_train_validation_loaders

    dataset = study["dataset"]
    validation = dataset["validation"]
    train = dataset["train"]
    affine_config = dict(dataset["affine"])
    if affine_config.get("enabled") is False:
        affine_config = None
    return build_mnist_train_validation_loaders(
        batch_size=int(train["batch_size"]),
        root=str(Path(data_root).expanduser()),
        download=bool(download),
        normalize=True,
        normalize_mean=float(dataset["normalization"]["mean"]),
        normalize_std=float(dataset["normalization"]["std"]),
        normalize_scale=float(dataset["normalization"]["scale"]),
        affine_config=affine_config,
        split_seed=int(validation["split_seed"]),
        shuffle_seed=int(train["shuffle_seed"]),
        validation_batch_size=int(validation["batch_size"]),
        return_source_indices=return_source_indices,
        num_workers=0,
        pin_memory=False,
    )


@dataclass(frozen=True)
class BatchResult:
    loss: float
    accuracy: float
    sample_count: int
    source_indices: tuple[int, ...]
    transition: TransitionValue


def _unpack_batch(batch: Any) -> tuple[Any, Any, tuple[int, ...]]:
    if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
        raise ValueError(
            "Expected a training batch as (images, labels[, source_indices]). "
            f"Provided value: {type(batch).__name__}."
        )
    images, labels = batch[0], batch[1]
    source_indices = (
        tuple(int(value) for value in batch[2].reshape(-1).tolist())
        if len(batch) == 3
        else ()
    )
    return images, labels, source_indices


def training_step(
    runtime: LRModelRuntime,
    batch: Any,
    *,
    learning_rate: LearningRateValue,
    restore: bool = False,
    zero_proposal_epsilon: float = 1e-12,
) -> BatchResult:
    """Compute a BP gradient and one fully captured optimizer transition."""

    import torch

    images, labels, source_indices = _unpack_batch(batch)
    images = images.to(runtime.device)
    labels = labels.to(runtime.device)
    _require_finite_tensor(images, "training input")
    runtime.optimizer.zero_grad()
    runtime.network.set_input(images, reset=True)
    runtime.minimizer_inference.compute_equilibrium()
    for layer in runtime.free_layers:
        _require_finite_tensor(layer.state, f"free layer {getattr(layer, 'name', '?')!r}")
    runtime.cost_fn.set_target(labels)
    cost = runtime.cost_fn.eval()
    _require_finite_tensor(cost, "training cost")
    loss = _finite_scalar(cost.mean().item(), "training batch loss")
    errors = runtime.cost_fn.error_fn()
    correct = int((~errors).sum().item())

    gradients = runtime.estimator.compute_gradient()
    if len(gradients) < len(runtime.parameters):
        raise RuntimeError(
            f"Expected at least {len(runtime.parameters)} gradients. "
            f"Provided value: {len(gradients)}."
        )
    for parameter, gradient in zip(runtime.parameters, gradients[: len(runtime.parameters)]):
        name = canonical_parameter_name(parameter)
        _require_finite_tensor(gradient, f"gradient {name!r}")
        if tuple(gradient.shape) != tuple(parameter.state.shape):
            raise RuntimeError(
                f"Expected gradient {name!r} to have shape {tuple(parameter.state.shape)}. "
                f"Provided value: {tuple(gradient.shape)}."
            )
        parameter.state.grad = gradient
    runtime.set_learning_rate(learning_rate)
    optimizer_neutral = bool(
        getattr(runtime, "optimizer_neutral_transitions", False)
    ) or isinstance(runtime.optimizer, torch.optim.Adam)
    step_fn = (
        optimizer_step_with_diagnostics
        if optimizer_neutral
        else sgd_step_with_diagnostics
    )
    transition = step_fn(
        runtime.parameters,
        runtime.optimizer,
        restore=restore,
        zero_proposal_epsilon=zero_proposal_epsilon,
        unbounded_normalization_scales=runtime.unbounded_normalization_scales,
    )
    for item in transition.parameters:
        _require_finite_tensor(item.pre_update, f"pre-update state {item.name!r}")
        post_optimizer = (
            item.post_optimizer_pre_projection
            if hasattr(item, "post_optimizer_pre_projection")
            else item.post_sgd_pre_projection
        )
        _require_finite_tensor(
            post_optimizer,
            f"post-optimizer/pre-projection state {item.name!r}",
        )
        _require_finite_tensor(item.post_projection, f"post-projection state {item.name!r}")
        for field in (
            "gradient_rms",
            "proposed_update_rms",
            "normalized_update",
            "projection_efficiency",
        ):
            _finite_scalar(getattr(item, field), f"{field} for {item.name!r}")
    return BatchResult(
        loss=loss,
        accuracy=correct / int(images.shape[0]),
        sample_count=int(images.shape[0]),
        source_indices=source_indices,
        transition=transition,
    )


def evaluate_validation(runtime: LRModelRuntime, validation_loader: Any) -> dict[str, Any]:
    """Evaluate the complete held-out MNIST-train validation cohort."""

    running_loss = 0.0
    running_correct = 0
    seen = 0
    ordered_indices: list[int] = []
    runtime.optimizer.zero_grad()
    for batch in validation_loader:
        images, labels, source_indices = _unpack_batch(batch)
        images = images.to(runtime.device)
        labels = labels.to(runtime.device)
        _require_finite_tensor(images, "validation input")
        runtime.network.set_input(images, reset=True)
        runtime.minimizer_inference.compute_equilibrium()
        for layer in runtime.free_layers:
            _require_finite_tensor(
                layer.state, f"validation free layer {getattr(layer, 'name', '?')!r}"
            )
        runtime.cost_fn.set_target(labels)
        cost = runtime.cost_fn.eval()
        _require_finite_tensor(cost, "validation cost")
        batch_size = int(images.shape[0])
        running_loss += _finite_scalar(cost.mean().item(), "validation batch loss") * batch_size
        running_correct += int((~runtime.cost_fn.error_fn()).sum().item())
        seen += batch_size
        ordered_indices.extend(source_indices)
    if seen != 5_000:
        raise RuntimeError(
            f"Expected complete validation split of 5000 examples. Provided value: {seen}."
        )
    return {
        "sample_count": seen,
        "loss": _finite_scalar(running_loss / seen, "validation loss"),
        "accuracy": _finite_scalar(running_correct / seen, "validation accuracy"),
        "source_indices": ordered_indices,
    }


def transition_rows(
    transition: TransitionValue,
    *,
    step: int,
    learning_rate: LearningRateValue,
    rho_schedule: float | None,
    bounded_initial_rms_scales: Mapping[str, float] | None = None,
    rho_schedule_relative: float | None = None,
    rho_schedule_span: float | None = None,
) -> list[dict[str, Any]]:
    """Flatten a transition into JSON/CSV-safe per-parameter records."""

    bounded_names = {
        item.name for item in transition.parameters if item.bounded_gate
    }
    scales: dict[str, float] | None = None
    if bounded_initial_rms_scales is not None:
        if set(bounded_initial_rms_scales) != bounded_names:
            raise ValueError(
                "Expected bounded_initial_rms_scales to contain exactly "
                f"{sorted(bounded_names)!r}. Provided value: "
                f"{sorted(bounded_initial_rms_scales)!r}."
            )
        scales = {}
        for name, raw_scale in bounded_initial_rms_scales.items():
            scale = _finite_scalar(raw_scale, f"initial RMS scale for {name!r}")
            if scale <= 0.0:
                raise ValueError(
                    f"Expected initial RMS scale for {name!r} to be positive. "
                    f"Provided value: {raw_scale!r}."
                )
            scales[name] = scale

    if isinstance(learning_rate, Mapping):
        per_parameter_learning_rate = {
            str(name): _finite_scalar(value, f"learning rate for parameter {name!r}")
            for name, value in learning_rate.items()
        }
    else:
        scalar_learning_rate = _finite_scalar(learning_rate, "learning rate")
        per_parameter_learning_rate = {
            item.name: scalar_learning_rate for item in transition.parameters
        }
    expected_names = {item.name for item in transition.parameters}
    if set(per_parameter_learning_rate) != expected_names:
        raise ValueError(
            "Expected transition learning rates to contain exactly "
            f"{sorted(expected_names)!r}. Provided value: "
            f"{sorted(per_parameter_learning_rate)!r}."
        )

    rows = []
    for item in transition.parameters:
        initial_scale = (
            None
            if scales is None or not item.bounded_gate
            else scales[item.name]
        )
        rows.append(
            {
                "step": int(step),
                "learning_rate": per_parameter_learning_rate[item.name],
                "scheduled_rho": None if rho_schedule is None else float(rho_schedule),
                "scheduled_rho_relative": (
                    None
                    if rho_schedule_relative is None
                    else float(rho_schedule_relative)
                ),
                "scheduled_rho_span": (
                    None if rho_schedule_span is None else float(rho_schedule_span)
                ),
                "parameter": item.name,
                "parameter_kind": item.parameter_kind,
                "bounded_gate": item.bounded_gate,
                "report_only": item.report_only,
                "gradient_rms": item.gradient_rms,
                "proposed_update_rms": item.proposed_update_rms,
                # ``normalized_update`` is retained byte-for-byte in meaning
                # for v1/v2: bounded proposals divided by their conductance
                # span, and report-only biases divided by their frozen scale.
                "normalized_update": item.normalized_update,
                "span_normalized_update": (
                    item.normalized_update if item.bounded_gate else None
                ),
                "initial_parameter_rms": initial_scale,
                "parameter_relative_update": (
                    None
                    if initial_scale is None
                    else item.proposed_update_rms / initial_scale
                ),
                "proposed_bound_crossing_fraction": item.proposed_bound_crossing_fraction,
                "lower_bound_occupancy": item.lower_bound_occupancy,
                "upper_bound_occupancy": item.upper_bound_occupancy,
                "combined_bound_occupancy": item.combined_bound_occupancy,
                "projection_efficiency": item.projection_efficiency,
                "proposal_is_numerically_zero": item.proposal_is_numerically_zero,
                "projection_gate_eligible": item.projection_gate_eligible,
            }
        )
    return rows


__all__ = [
    "BatchResult",
    "FROZEN_ADAM_OPTIMIZER",
    "FROZEN_SGD_OPTIMIZER",
    "LRModelRuntime",
    "LRStudyNumericalError",
    "LearningRateValue",
    "TransitionValue",
    "build_loader_bundle",
    "build_model_runtime",
    "canonical_parameter_name",
    "evaluate_validation",
    "parameter_state_diagnostics",
    "parameter_tensor_digest",
    "training_step",
    "transition_rows",
    "validate_explicit_optimizer_contract",
]
