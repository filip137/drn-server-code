"""Numerical runtime for the ordinary-MNIST perfect-diode LR screen.

This module is deliberately independent of the historical hard-sigmoid v1--v7
stage implementations.  It provides the numerical operations shared by the
Conv1 and Conv2 perfect-diode study:

* one seed-0 initialization checkpoint per architecture;
* the fixed ``T/K`` versus ``64/64`` gradient-security comparison;
* optimizer-specific, exactly restored 32/64/128 shadow probes;
* independent Conv/Dense rho-to-LR conversion with the bias-Q90 cap;
* restarted 640-step canaries with the frozen online safety gates; and
* exact three-epoch candidates and fresh 10/30-epoch confirmations.

Only the MNIST training split is instantiated.  The official test split is
never read.
"""

from __future__ import annotations

import copy
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .identity import sha256_file, sha256_json
from .io import atomic_write_csv, atomic_write_json, read_json
from .lr_engine import (
    LRStudyNumericalError,
    _unpack_batch,
    build_loader_bundle,
    build_model_runtime,
    canonical_parameter_name,
    evaluate_validation,
    parameter_state_diagnostics,
    parameter_tensor_digest,
    training_step,
)
from .lr_step import (
    OptimizerStateNumericalError,
    require_finite_optimizer_state,
)
from .perfectdiode_hparam_spec import (
    CANDIDATE_EPOCHS,
    CANDIDATE_TOTAL_STEPS,
    CANARY_STEPS,
    CONV1_CONFIRM_EPOCHS,
    CONV1_CONFIRM_STEPS,
    CONV2_CONFIRM_EPOCHS,
    CONV2_CONFIRM_STEPS,
    MAX_CENTER_ATTEMPTS,
    PD_RUN_SCHEMA_VERSION,
    PROBE_BATCH_COUNTS,
    STEPS_PER_EPOCH,
    PerfectDiodeHparamStudySpec,
    core_grid,
    probe_stability_decision,
    safe_center,
)


SECURITY_EXAMPLES = 256
SECURITY_BATCH_SIZE = 32
SECURITY_BATCHES = SECURITY_EXAMPLES // SECURITY_BATCH_SIZE
GRADIENT_ZERO_EPSILON = 1.0e-12
NORM_DELTA_EPSILON = 1.0e-30
SECURITY_NORM_DELTA_MAXIMUM = 0.10
SECURITY_ZERO_FRACTION_DELTA_MAXIMUM = 0.02
SECURITY_COSINE_MINIMUM = 0.90
PROBE_NOMINAL_LEARNING_RATE = 1.0


class PerfectDiodeRuntimeError(RuntimeError):
    """Raised when a fail-closed perfect-diode runtime contract is violated."""


def _finite(
    value: Any,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        qualifier = (
            "positive finite"
            if positive
            else "non-negative finite"
            if nonnegative
            else "finite"
        )
        raise ValueError(
            f"Expected {label} to be a {qualifier} number. "
            f"Provided value: {value!r}."
        )
    result = float(value)
    if positive and result <= 0.0:
        raise ValueError(
            f"Expected {label} to be a positive finite number. "
            f"Provided value: {value!r}."
        )
    if nonnegative and result < 0.0:
        raise ValueError(
            f"Expected {label} to be a non-negative finite number. "
            f"Provided value: {value!r}."
        )
    return result


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"Expected {label} to be an integer >= 1. Provided value: {value!r}."
        )
    return int(value)


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(_finite(value, "quantile sample") for value in values)
    if not ordered:
        raise ValueError(
            "Expected quantile samples to be non-empty. Provided value: []."
        )
    q = _finite(probability, "quantile probability")
    if not 0.0 <= q <= 1.0:
        raise ValueError(
            "Expected quantile probability to be in [0, 1]. "
            f"Provided value: {probability!r}."
        )
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _rms(tensor: Any) -> float:
    import torch

    values = tensor.detach().to(dtype=torch.float64)
    if values.numel() == 0:
        raise ValueError(
            f"Expected a non-empty tensor. Provided value: shape={tuple(values.shape)!r}."
        )
    return _finite(
        float(torch.sqrt(torch.mean(values * values)).item()),
        "tensor RMS",
        nonnegative=True,
    )


def _relative_difference(left: float, right: float) -> float:
    left_value = _finite(left, "left relative-difference value", nonnegative=True)
    right_value = _finite(right, "right relative-difference value", nonnegative=True)
    return abs(left_value - right_value) / max(
        abs(left_value), abs(right_value), NORM_DELTA_EPSILON
    )


def _state_values_equal(left: Any, right: Any) -> bool:
    """Return exact equality for nested optimizer state dictionaries."""

    import torch

    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.dtype == right.dtype
            and left.device == right.device
            and tuple(left.shape) == tuple(right.shape)
            and bool(torch.equal(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_state_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                _state_values_equal(left_value, right_value)
                for left_value, right_value in zip(left, right)
            )
        )
    return type(left) is type(right) and left == right


def _parameter_by_name(parameters: Iterable[Any]) -> dict[str, Any]:
    values = list(parameters)
    result = {
        canonical_parameter_name(parameter): parameter for parameter in values
    }
    if len(result) != len(values):
        raise PerfectDiodeRuntimeError(
            "Expected every scientific parameter name to be unique."
        )
    return result


def _infer_bias_to_weight(parameter_names: Iterable[str]) -> dict[str, str]:
    names = tuple(str(name) for name in parameter_names)
    mapping: dict[str, str] = {}
    for name in names:
        if not name.startswith("Bias_"):
            continue
        suffix = name.removeprefix("Bias_")
        weight = f"ConvWeight_{suffix}"
        if weight not in names:
            raise ValueError(
                "Expected every perfect-diode bias to have an attached Conv weight. "
                f"Provided value: bias={name!r}, expected_weight={weight!r}."
            )
        mapping[name] = weight
    return mapping


def _conv_weight_names(parameter_names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(
        sorted(
            (str(name) for name in parameter_names if str(name).startswith("ConvWeight_")),
            key=lambda value: int(value.removeprefix("ConvWeight_")),
        )
    )
    if not names:
        raise ValueError(
            "Expected at least one ConvWeight_* parameter. Provided value: []."
        )
    return names


def _weight_names(parameter_names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in parameter_names)
    conv = _conv_weight_names(names)
    dense = tuple(name for name in names if name.startswith("DenseWeight_"))
    if dense != ("DenseWeight_0",):
        raise ValueError(
            "Expected exactly DenseWeight_0 after the Conv weights. "
            f"Provided value: {dense!r}."
        )
    return (*conv, "DenseWeight_0")


def compare_fixed_tk_gradient_security(
    operational_by_parameter: Mapping[str, Sequence[Any]],
    reference_by_parameter: Mapping[str, Sequence[Any]],
    *,
    operational_t: int,
    operational_k: int,
    reference_t: int = 64,
    reference_k: int = 64,
    expected_batch_count: int | None = SECURITY_BATCHES,
    zero_epsilon: float = GRADIENT_ZERO_EPSILON,
    norm_delta_maximum: float = SECURITY_NORM_DELTA_MAXIMUM,
    zero_fraction_delta_maximum: float = SECURITY_ZERO_FRACTION_DELTA_MAXIMUM,
    cosine_minimum: float = SECURITY_COSINE_MINIMUM,
) -> dict[str, Any]:
    """Apply the fixed-``T/K`` security gate to every Conv gradient.

    The norm and zero-fraction gates use the mean per-batch statistics that
    established the earlier Conv T/K protocol.  Cosine is likewise averaged
    across identical batches.  Missing, non-finite, shape-mismatched, or dead
    vectors fail closed.
    """

    import torch

    for value, label in (
        (operational_t, "operational_t"),
        (operational_k, "operational_k"),
        (reference_t, "reference_t"),
        (reference_k, "reference_k"),
    ):
        _positive_integer(value, label)
    zero = _finite(zero_epsilon, "zero_epsilon", positive=True)
    norm_gate = _finite(
        norm_delta_maximum, "norm_delta_maximum", nonnegative=True
    )
    zero_gate = _finite(
        zero_fraction_delta_maximum,
        "zero_fraction_delta_maximum",
        nonnegative=True,
    )
    cosine_gate = _finite(cosine_minimum, "cosine_minimum")
    operational_names = set(operational_by_parameter)
    reference_names = set(reference_by_parameter)
    if operational_names != reference_names:
        raise ValueError(
            "Expected operational and reference gradients for identical "
            f"ConvWeight names. Provided value: operational="
            f"{sorted(operational_names)!r}, reference={sorted(reference_names)!r}."
        )
    names = _conv_weight_names(operational_names)
    if set(names) != operational_names:
        raise ValueError(
            "Expected fixed-T/K security inputs to contain only ConvWeight_* "
            f"parameters. Provided value: {sorted(operational_names)!r}."
        )

    records: list[dict[str, Any]] = []
    for name in names:
        operational = tuple(operational_by_parameter[name])
        reference = tuple(reference_by_parameter[name])
        if len(operational) != len(reference) or not operational:
            raise ValueError(
                f"Expected matched non-empty gradient batches for {name!r}. "
                f"Provided value: operational={len(operational)}, "
                f"reference={len(reference)}."
            )
        if expected_batch_count is not None and len(operational) != int(
            expected_batch_count
        ):
            raise ValueError(
                f"Expected exactly {int(expected_batch_count)} security batches "
                f"for {name!r}. Provided value: {len(operational)}."
            )
        batch_records: list[dict[str, Any]] = []
        for batch_index, (current_tensor, reference_tensor) in enumerate(
            zip(operational, reference)
        ):
            current = (
                current_tensor.detach()
                .reshape(-1)
                .to(dtype=torch.float64, device="cpu")
            )
            target = (
                reference_tensor.detach()
                .reshape(-1)
                .to(dtype=torch.float64, device="cpu")
            )
            if tuple(current.shape) != tuple(target.shape):
                raise ValueError(
                    f"Expected gradient shapes for {name!r} batch {batch_index} "
                    f"to match. Provided value: operational={tuple(current.shape)!r}, "
                    f"reference={tuple(target.shape)!r}."
                )
            if not bool(torch.isfinite(current).all()) or not bool(
                torch.isfinite(target).all()
            ):
                raise ValueError(
                    f"Expected finite gradients for {name!r} batch {batch_index}."
                )
            norm = float(torch.linalg.vector_norm(current).item())
            reference_norm = float(torch.linalg.vector_norm(target).item())
            norm_delta = _relative_difference(norm, reference_norm)
            current_zero = float((current.abs() <= zero).to(torch.float64).mean())
            reference_zero = float((target.abs() <= zero).to(torch.float64).mean())
            zero_delta = abs(current_zero - reference_zero)
            denominator = norm * reference_norm
            cosine = (
                float(torch.dot(current, target).item() / denominator)
                if denominator > 0.0
                else math.nan
            )
            batch_records.append(
                {
                    "batch_index": batch_index,
                    "gradient_l2": norm,
                    "reference_gradient_l2": reference_norm,
                    "relative_gradient_l2_norm_delta": norm_delta,
                    "gradient_zero_fraction": current_zero,
                    "reference_gradient_zero_fraction": reference_zero,
                    "absolute_zero_fraction_delta": zero_delta,
                    "gradient_vector_cosine": cosine,
                }
            )

        mean_norm = sum(item["gradient_l2"] for item in batch_records) / len(
            batch_records
        )
        mean_reference_norm = sum(
            item["reference_gradient_l2"] for item in batch_records
        ) / len(batch_records)
        mean_zero = sum(
            item["gradient_zero_fraction"] for item in batch_records
        ) / len(batch_records)
        mean_reference_zero = sum(
            item["reference_gradient_zero_fraction"] for item in batch_records
        ) / len(batch_records)
        cosines = [item["gradient_vector_cosine"] for item in batch_records]
        mean_cosine = (
            sum(cosines) / len(cosines)
            if all(math.isfinite(value) for value in cosines)
            else math.nan
        )
        norm_delta = _relative_difference(mean_norm, mean_reference_norm)
        zero_delta = abs(mean_zero - mean_reference_zero)
        passed = (
            math.isfinite(mean_cosine)
            and norm_delta <= norm_gate
            and zero_delta <= zero_gate
            and mean_cosine >= cosine_gate
        )
        records.append(
            {
                "parameter": name,
                "batch_count": len(batch_records),
                "gradient_l2_mean": mean_norm,
                "reference_gradient_l2_mean": mean_reference_norm,
                "relative_gradient_l2_norm_delta": norm_delta,
                "gradient_zero_fraction_mean": mean_zero,
                "reference_gradient_zero_fraction_mean": mean_reference_zero,
                "absolute_zero_fraction_delta": zero_delta,
                "gradient_vector_cosine_mean": mean_cosine,
                "maximum_batch_norm_delta": max(
                    item["relative_gradient_l2_norm_delta"]
                    for item in batch_records
                ),
                "maximum_batch_zero_fraction_delta": max(
                    item["absolute_zero_fraction_delta"]
                    for item in batch_records
                ),
                "minimum_batch_cosine": (
                    min(cosines)
                    if all(math.isfinite(value) for value in cosines)
                    else None
                ),
                "passed": passed,
                "batches": batch_records,
            }
        )

    passed = all(record["passed"] for record in records)
    return {
        "schema_version": "mnist-conv-perfectdiode-fixed-tk-security/v1",
        "status": "complete" if passed else "unresolved_fixed_tk_gradient_mismatch",
        "security_passed": passed,
        "official_test_read": False,
        "operational": {"T": int(operational_t), "K": int(operational_k)},
        "reference": {"T": int(reference_t), "K": int(reference_k)},
        "gates": {
            "relative_gradient_l2_norm_delta_maximum": norm_gate,
            "absolute_zero_fraction_delta_maximum": zero_gate,
            "gradient_vector_cosine_minimum": cosine_gate,
            "gradient_zero_epsilon": zero,
        },
        "parameter_diagnostics": records,
        "worst_relative_gradient_l2_norm_delta": max(
            record["relative_gradient_l2_norm_delta"] for record in records
        ),
        "worst_absolute_zero_fraction_delta": max(
            record["absolute_zero_fraction_delta"] for record in records
        ),
        "minimum_gradient_vector_cosine": (
            min(record["gradient_vector_cosine_mean"] for record in records)
            if all(
                math.isfinite(record["gradient_vector_cosine_mean"])
                for record in records
            )
            else None
        ),
    }


def collect_runtime_gradients(
    runtime: Any,
    batches: Sequence[Any],
) -> dict[str, tuple[Any, ...]]:
    """Collect read-only BP gradients for every Conv weight on frozen batches."""

    import torch

    if not batches:
        raise ValueError(
            "Expected at least one gradient-security batch. Provided value: []."
        )
    names = _conv_weight_names(
        canonical_parameter_name(parameter) for parameter in runtime.parameters
    )
    before = parameter_tensor_digest(runtime.parameters)
    collected: dict[str, list[Any]] = {name: [] for name in names}
    for batch_index, batch in enumerate(batches):
        images, labels, _source_indices = _unpack_batch(batch)
        images = images.to(runtime.device)
        labels = labels.to(runtime.device)
        if not bool(torch.isfinite(images).all()):
            raise LRStudyNumericalError(
                f"Expected finite security inputs at batch {batch_index}."
            )
        runtime.optimizer.zero_grad()
        runtime.network.set_input(images, reset=True)
        runtime.minimizer_inference.compute_equilibrium()
        runtime.cost_fn.set_target(labels)
        cost = runtime.cost_fn.eval()
        if not bool(torch.isfinite(cost).all()):
            raise LRStudyNumericalError(
                f"Expected finite security cost at batch {batch_index}."
            )
        gradients = runtime.estimator.compute_gradient()
        if len(gradients) < len(runtime.parameters):
            raise PerfectDiodeRuntimeError(
                f"Expected {len(runtime.parameters)} gradients. "
                f"Provided value: {len(gradients)}."
            )
        by_name = {
            canonical_parameter_name(parameter): gradient
            for parameter, gradient in zip(runtime.parameters, gradients)
        }
        for name in names:
            gradient = by_name[name]
            if not bool(torch.isfinite(gradient).all()):
                raise LRStudyNumericalError(
                    f"Expected finite security gradient for {name!r} "
                    f"at batch {batch_index}."
                )
            collected[name].append(
                gradient.detach().to(dtype=torch.float64, device="cpu").clone()
            )
    if parameter_tensor_digest(runtime.parameters) != before:
        raise PerfectDiodeRuntimeError(
            "Expected gradient-security replay not to mutate parameter tensors."
        )
    return {name: tuple(values) for name, values in collected.items()}


def run_fixed_tk_gradient_security(
    operational_runtime: Any,
    reference_runtime: Any,
    batches: Sequence[Any],
    *,
    operational_t: int,
    operational_k: int,
    reference_t: int = 64,
    reference_k: int = 64,
) -> dict[str, Any]:
    """Collect and compare operational and ``64/64`` Conv gradients."""

    operational = collect_runtime_gradients(operational_runtime, batches)
    reference = collect_runtime_gradients(reference_runtime, batches)
    return compare_fixed_tk_gradient_security(
        operational,
        reference,
        operational_t=operational_t,
        operational_k=operational_k,
        reference_t=reference_t,
        reference_k=reference_k,
        expected_batch_count=len(batches),
    )


def _probe_statistic(name: str, values: Sequence[float]) -> float:
    return _linear_quantile(values, 0.9 if name.startswith("Bias_") else 0.5)


def measure_adaptive_optimizer_probe(
    runtime: Any,
    batches: Sequence[Any],
    *,
    bias_to_weight: Mapping[str, str] | None = None,
    stability_tolerance: float = 0.10,
) -> dict[str, Any]:
    """Measure a fresh optimizer's nominal-LR-one proposal units.

    Every proposal is a shadow step restored by :func:`training_step`.  The
    parameter bytes and complete optimizer state dictionary are checked after
    every batch.  Adam therefore begins every shadow proposal with empty
    moments, variances, and step counter.
    """

    parameters = _parameter_by_name(runtime.parameters)
    initial_optimizer_state = runtime.optimizer.state_dict()
    mutable_state = initial_optimizer_state.get("state")
    if not isinstance(mutable_state, Mapping) or mutable_state:
        raise PerfectDiodeRuntimeError(
            "Expected an optimizer-specific probe to start from a fresh empty "
            f"{runtime.optimizer_name} mutable state. Provided value: "
            f"{len(mutable_state) if isinstance(mutable_state, Mapping) else mutable_state!r}."
        )
    parameter_names = tuple(parameters)
    weight_names = _weight_names(parameter_names)
    conv_names = _conv_weight_names(parameter_names)
    mapping = (
        _infer_bias_to_weight(parameter_names)
        if bias_to_weight is None
        else {str(key): str(value) for key, value in bias_to_weight.items()}
    )
    bias_names = tuple(name for name in parameter_names if name.startswith("Bias_"))
    if set(mapping) != set(bias_names) or any(
        mapping[bias] not in conv_names for bias in mapping
    ):
        raise ValueError(
            "Expected bias_to_weight to map every Bias_i exactly to a ConvWeight_i. "
            f"Provided value: {mapping!r}."
        )
    if len(batches) < PROBE_BATCH_COUNTS[0]:
        raise ValueError(
            f"Expected at least {PROBE_BATCH_COUNTS[0]} probe batches. "
            f"Provided value: {len(batches)}."
        )
    initial_weight_rms = {
        name: _rms(parameters[name].state) for name in weight_names
    }
    if any(value <= 0.0 for value in initial_weight_rms.values()):
        raise PerfectDiodeRuntimeError(
            "Expected every initial bounded-weight RMS to be positive."
        )
    rates = {name: PROBE_NOMINAL_LEARNING_RATE for name in parameter_names}
    runtime.set_learning_rate(rates)
    parameter_digest = parameter_tensor_digest(runtime.parameters)
    optimizer_state = copy.deepcopy(runtime.optimizer.state_dict())
    require_finite_optimizer_state(
        runtime.optimizer, context="adaptive probe initial optimizer state"
    )
    unit_samples: dict[str, list[float]] = {
        name: [] for name in parameter_names
    }
    records: list[dict[str, Any]] = []
    final_decision = None
    used_batches = 0

    for target_count in PROBE_BATCH_COUNTS:
        if len(batches) < target_count:
            raise ValueError(
                f"Expected at least {target_count} batches for an extended probe. "
                f"Provided value: {len(batches)}."
            )
        for batch_index in range(used_batches, target_count):
            before_parameter = parameter_tensor_digest(runtime.parameters)
            before_optimizer = copy.deepcopy(runtime.optimizer.state_dict())
            result = training_step(
                runtime,
                batches[batch_index],
                learning_rate=rates,
                restore=True,
            )
            if not bool(getattr(result.transition, "restored", False)):
                raise PerfectDiodeRuntimeError(
                    "Expected every optimizer probe proposal to report exact restoration."
                )
            if parameter_tensor_digest(runtime.parameters) != before_parameter:
                raise PerfectDiodeRuntimeError(
                    "Expected every optimizer probe proposal to restore all "
                    f"parameter bytes. Provided batch index: {batch_index}."
                )
            if not _state_values_equal(
                runtime.optimizer.state_dict(), before_optimizer
            ):
                raise PerfectDiodeRuntimeError(
                    "Expected every optimizer probe proposal to restore the complete "
                    f"optimizer state. Provided batch index: {batch_index}."
                )
            transition = result.transition.by_name
            if set(transition) != set(parameter_names):
                raise PerfectDiodeRuntimeError(
                    "Expected each optimizer probe transition to cover every "
                    f"scientific parameter. Provided value: {sorted(transition)!r}."
                )
            per_parameter: dict[str, Any] = {}
            for name in parameter_names:
                attached_weight = mapping.get(name, name)
                denominator = initial_weight_rms[attached_weight]
                unit = _finite(
                    float(transition[name].proposed_update_rms) / denominator,
                    f"proposal unit for {name!r}",
                    nonnegative=True,
                )
                unit_samples[name].append(unit)
                per_parameter[name] = {
                    "proposal_unit": unit,
                    "gradient_rms": float(transition[name].gradient_rms),
                    "proposed_update_rms": float(
                        transition[name].proposed_update_rms
                    ),
                    "projection_efficiency": float(
                        transition[name].projection_efficiency
                    ),
                }
            records.append(
                {
                    "batch_index": batch_index,
                    "source_indices": list(result.source_indices),
                    "loss": result.loss,
                    "accuracy": result.accuracy,
                    "parameters": per_parameter,
                }
            )
        used_batches = target_count
        half = target_count // 2
        split_differences: dict[str, float] = {}
        split_statistics: dict[str, dict[str, float]] = {}
        for name, samples in unit_samples.items():
            first = _probe_statistic(name, samples[:half])
            second = _probe_statistic(name, samples[half:target_count])
            difference = _relative_difference(first, second)
            split_statistics[name] = {
                "first_half": first,
                "second_half": second,
                "relative_difference": difference,
            }
            split_differences[name] = difference
        final_decision = probe_stability_decision(
            split_differences,
            used_batches=target_count,
            tolerance=stability_tolerance,
        )
        if final_decision.status != "extend":
            break

    assert final_decision is not None
    if parameter_tensor_digest(runtime.parameters) != parameter_digest:
        raise PerfectDiodeRuntimeError(
            "Expected all adaptive probe proposals together to leave parameter "
            "tensors unchanged."
        )
    if not _state_values_equal(runtime.optimizer.state_dict(), optimizer_state):
        raise PerfectDiodeRuntimeError(
            "Expected all adaptive probe proposals together to leave optimizer "
            "state unchanged."
        )
    weight_units = {
        name: _probe_statistic(name, unit_samples[name][:used_batches])
        for name in weight_names
    }
    bias_units = {
        name: _probe_statistic(name, unit_samples[name][:used_batches])
        for name in bias_names
    }
    invalid_weights = [
        name
        for name, value in weight_units.items()
        if not math.isfinite(value) or value <= 0.0
    ]
    status = final_decision.status
    if invalid_weights:
        status = "unresolved"
    return {
        "schema_version": "mnist-conv-perfectdiode-optimizer-probe/v1",
        "status": "complete" if status == "stable" else "unresolved_probe",
        "probe_stable": status == "stable",
        "official_test_read": False,
        "optimizer_name": str(runtime.optimizer_name),
        "parameter_names": list(parameter_names),
        "weight_names": list(weight_names),
        "conv_weight_names": list(conv_names),
        "bias_names": list(bias_names),
        "bias_to_weight": mapping,
        "nominal_learning_rate_by_parameter": rates,
        "used_batches": used_batches,
        "maximum_batches": PROBE_BATCH_COUNTS[-1],
        "stability_tolerance": float(stability_tolerance),
        "unstable_parameters": list(final_decision.unstable_parameters),
        "invalid_weight_units": invalid_weights,
        "split_half_statistics": split_statistics,
        "weight_unit_statistic": "median",
        "normalization_unit_by_weight": weight_units,
        "bias_unit_statistic": "q90",
        "bias_q90_unit_by_parameter": bias_units,
        "initial_weight_rms_by_parameter": initial_weight_rms,
        "proposal_unit_samples_by_parameter": {
            name: values[:used_batches] for name, values in unit_samples.items()
        },
        "records": records,
        "restoration": {
            "parameter_tensor_sha256": parameter_digest,
            "optimizer_state_exactly_restored_after_every_proposal": True,
            "fresh_optimizer_state_per_shadow_proposal": True,
            "initial_mutable_optimizer_state_empty": True,
        },
    }


def derive_raw_learning_rates(
    probe: Mapping[str, Any],
    *,
    rho_conv: float,
    rho_dense: float,
    bias_to_weight: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Derive the complete per-parameter LR vector from one stable probe."""

    conv_target = _finite(rho_conv, "rho_conv", positive=True)
    dense_target = _finite(rho_dense, "rho_dense", positive=True)
    if probe.get("probe_stable") is not True:
        raise ValueError(
            "Expected a stable optimizer probe before LR derivation. "
            f"Provided value: {probe.get('status')!r}."
        )
    parameter_names = tuple(str(name) for name in probe["parameter_names"])
    weight_names = _weight_names(parameter_names)
    conv_names = _conv_weight_names(parameter_names)
    units = probe.get("normalization_unit_by_weight")
    if not isinstance(units, Mapping) or set(units) != set(weight_names):
        raise ValueError(
            "Expected normalization_unit_by_weight for every bounded weight. "
            f"Provided value: {units!r}."
        )
    rates: dict[str, float] = {}
    for name in weight_names:
        target = conv_target if name in conv_names else dense_target
        unit = _finite(units[name], f"weight unit for {name!r}", positive=True)
        rates[name] = target / unit

    mapping = (
        {str(key): str(value) for key, value in probe["bias_to_weight"].items()}
        if bias_to_weight is None
        else {str(key): str(value) for key, value in bias_to_weight.items()}
    )
    bias_units = probe.get("bias_q90_unit_by_parameter")
    if not isinstance(bias_units, Mapping) or set(bias_units) != set(mapping):
        raise ValueError(
            "Expected one bias Q90 unit for every attached Conv bias. "
            f"Provided value: {bias_units!r}."
        )
    for bias, attached_weight in mapping.items():
        if attached_weight not in conv_names:
            raise ValueError(
                f"Expected {bias!r} to attach to a ConvWeight. "
                f"Provided value: {attached_weight!r}."
            )
        unit = _finite(
            bias_units[bias], f"bias Q90 unit for {bias!r}", nonnegative=True
        )
        attached_rate = rates[attached_weight]
        rates[bias] = (
            attached_rate
            if unit == 0.0
            else min(attached_rate, conv_target / unit)
        )
    if set(rates) != set(parameter_names):
        raise PerfectDiodeRuntimeError(
            "Expected LR derivation to cover every scientific parameter. "
            f"Provided value: rates={sorted(rates)!r}, "
            f"parameters={sorted(parameter_names)!r}."
        )
    return {name: float(rates[name]) for name in parameter_names}


@dataclass(frozen=True)
class TrainingContract:
    """Exact step/validation/checkpoint contract for one training mode."""

    mode: str
    architecture: str
    epochs: int
    steps_per_epoch: int
    total_steps: int
    restart_from_shared_initialization: bool
    validate_after_every_epoch: bool
    save_best_validation_loss: bool
    save_final: bool


def training_contract(
    *,
    architecture: str,
    mode: str,
) -> TrainingContract:
    """Return the immutable candidate or long-confirmation contract."""

    if architecture not in {"conv1", "conv2"}:
        raise ValueError(
            "Expected architecture to be 'conv1' or 'conv2'. "
            f"Provided value: {architecture!r}."
        )
    if mode == "candidate":
        epochs = CANDIDATE_EPOCHS
        total_steps = CANDIDATE_TOTAL_STEPS
    elif mode == "long_confirmation":
        if architecture == "conv1":
            epochs = CONV1_CONFIRM_EPOCHS
            total_steps = CONV1_CONFIRM_STEPS
        else:
            epochs = CONV2_CONFIRM_EPOCHS
            total_steps = CONV2_CONFIRM_STEPS
    else:
        raise ValueError(
            "Expected mode to be 'candidate' or 'long_confirmation'. "
            f"Provided value: {mode!r}."
        )
    if total_steps != epochs * STEPS_PER_EPOCH:
        raise PerfectDiodeRuntimeError(
            "Expected exact epoch and step contracts to be internally consistent."
        )
    return TrainingContract(
        mode=mode,
        architecture=architecture,
        epochs=epochs,
        steps_per_epoch=STEPS_PER_EPOCH,
        total_steps=total_steps,
        restart_from_shared_initialization=True,
        validate_after_every_epoch=True,
        save_best_validation_loss=True,
        save_final=True,
    )


def epoch_validation_checkpoint_plan(
    *,
    architecture: str,
    mode: str,
) -> tuple[dict[str, Any], ...]:
    """Return the exact epoch-end validation and checkpoint actions."""

    contract = training_contract(architecture=architecture, mode=mode)
    return tuple(
        {
            "epoch": epoch,
            "step": epoch * contract.steps_per_epoch,
            "full_validation": True,
            "update_best_validation_loss_checkpoint": True,
            "save_final_checkpoint": epoch == contract.epochs,
        }
        for epoch in range(1, contract.epochs + 1)
    )


class SafetyMonitor:
    """Online implementation of the frozen perfect-diode safety gates."""

    def __init__(
        self,
        initial_bound_occupancy: Mapping[str, float],
        *,
        ema_decay: float = 0.98,
        zero_update_epsilon: float = 1.0e-12,
    ) -> None:
        if not initial_bound_occupancy:
            raise ValueError(
                "Expected initial_bound_occupancy for at least one bounded weight. "
                "Provided value: {}."
            )
        self.initial_occupancy = {
            str(name): _finite(
                value, f"initial bound occupancy for {name!r}", nonnegative=True
            )
            for name, value in initial_bound_occupancy.items()
        }
        if any(value > 1.0 for value in self.initial_occupancy.values()):
            raise ValueError(
                "Expected every initial bound occupancy to lie in [0, 1]. "
                f"Provided value: {self.initial_occupancy!r}."
            )
        self.ema_decay = _finite(ema_decay, "ema_decay", nonnegative=True)
        if self.ema_decay >= 1.0:
            raise ValueError(
                f"Expected ema_decay in [0, 1). Provided value: {ema_decay!r}."
            )
        self.zero_update_epsilon = _finite(
            zero_update_epsilon,
            "zero_update_epsilon",
            nonnegative=True,
        )
        self.step = 0
        self.ema: float | None = None
        self.prior_ema_minimum = math.inf
        self.gradient_samples = {
            name: [] for name in self.initial_occupancy
        }
        self.gradient_reference: dict[str, float] = {}
        self.loss_streak = 0
        self.gradient_streak = {name: 0 for name in self.initial_occupancy}
        self.occupancy_streak = {name: 0 for name in self.initial_occupancy}
        self.projection_streak = {name: 0 for name in self.initial_occupancy}
        self.failure: dict[str, Any] | None = None
        self.projection_efficiencies: list[float] = []
        self.maximum_occupancy = dict(self.initial_occupancy)

    def _failure_record(
        self,
        *,
        kind: str,
        persistence: int,
        parameter: str | None,
    ) -> dict[str, Any]:
        return {
            "kind": kind,
            "parameter": parameter,
            "onset_step": self.step - persistence + 1,
            "confirmed_step": self.step,
        }

    def observe(self, *, loss: float, transition: Any) -> dict[str, Any] | None:
        """Consume one successful optimizer transition and return any failure."""

        if self.failure is not None:
            raise PerfectDiodeRuntimeError(
                "Expected SafetyMonitor.observe not to run after a confirmed failure."
            )
        self.step += 1
        batch_loss = _finite(loss, "training loss")
        self.ema = (
            batch_loss
            if self.ema is None
            else self.ema_decay * self.ema
            + (1.0 - self.ema_decay) * batch_loss
        )
        by_name = transition.by_name
        if not set(self.initial_occupancy).issubset(by_name):
            raise PerfectDiodeRuntimeError(
                "Expected every bounded weight in each safety transition. "
                f"Provided value: {sorted(by_name)!r}."
            )
        for name in self.initial_occupancy:
            item = by_name[name]
            for field in (
                "gradient_rms",
                "proposed_update_rms",
                "projection_efficiency",
                "combined_bound_occupancy",
            ):
                _finite(
                    getattr(item, field),
                    f"{field} for safety parameter {name!r}",
                    nonnegative=True,
                )

        early_failures: list[dict[str, Any]] = []
        for name in self.initial_occupancy:
            item = by_name[name]
            occupancy = float(item.combined_bound_occupancy)
            if occupancy > self.initial_occupancy[name] + 0.20:
                self.occupancy_streak[name] += 1
                if self.occupancy_streak[name] == 16:
                    early_failures.append(
                        self._failure_record(
                            kind="bound_occupancy_increase",
                            persistence=16,
                            parameter=name,
                        )
                    )
            else:
                self.occupancy_streak[name] = 0
            proposal_is_zero = bool(
                getattr(item, "proposal_is_numerically_zero", False)
            ) or float(item.proposed_update_rms) <= self.zero_update_epsilon
            if not proposal_is_zero:
                if float(item.projection_efficiency) < 0.50:
                    self.projection_streak[name] += 1
                    if self.projection_streak[name] == 16:
                        early_failures.append(
                            self._failure_record(
                                kind="projection_efficiency",
                                persistence=16,
                                parameter=name,
                            )
                        )
                else:
                    self.projection_streak[name] = 0

        if self.step <= 32:
            for name in self.initial_occupancy:
                self.gradient_samples[name].append(
                    float(by_name[name].gradient_rms)
                )
            if self.step == 32:
                self.gradient_reference = {
                    name: _linear_quantile(values, 0.5)
                    for name, values in self.gradient_samples.items()
                }
            self.prior_ema_minimum = min(self.prior_ema_minimum, self.ema)
        else:
            failures: list[dict[str, Any]] = []
            if self.ema > 4.0 * self.prior_ema_minimum:
                self.loss_streak += 1
                if self.loss_streak == 8:
                    failures.append(
                        self._failure_record(
                            kind="loss_ema_explosion",
                            persistence=8,
                            parameter=None,
                        )
                    )
            else:
                self.loss_streak = 0
            self.prior_ema_minimum = min(self.prior_ema_minimum, self.ema)

            for name in self.initial_occupancy:
                item = by_name[name]
                if float(item.gradient_rms) > 100.0 * self.gradient_reference[name]:
                    self.gradient_streak[name] += 1
                    if self.gradient_streak[name] == 8:
                        failures.append(
                            self._failure_record(
                                kind="gradient_rms_explosion",
                                persistence=8,
                                parameter=name,
                            )
                        )
                else:
                    self.gradient_streak[name] = 0

            failures.extend(early_failures)
            if failures:
                self.failure = min(
                    failures,
                    key=lambda item: (
                        item["onset_step"],
                        item["kind"],
                        item["parameter"] or "",
                    ),
                )
        if self.failure is None and early_failures:
            self.failure = min(
                early_failures,
                key=lambda item: (
                    item["onset_step"],
                    item["kind"],
                    item["parameter"] or "",
                ),
            )

        for name in self.initial_occupancy:
            item = by_name[name]
            occupancy = float(item.combined_bound_occupancy)
            self.maximum_occupancy[name] = max(
                self.maximum_occupancy[name], occupancy
            )
            if not bool(
                getattr(item, "proposal_is_numerically_zero", False)
            ):
                self.projection_efficiencies.append(
                    float(item.projection_efficiency)
                )
        return self.failure

    def summary(self) -> dict[str, Any]:
        return {
            "processed_steps": self.step,
            "loss_ema": self.ema,
            "prior_loss_ema_minimum": (
                None
                if not math.isfinite(self.prior_ema_minimum)
                else self.prior_ema_minimum
            ),
            "first_32_gradient_rms_median_by_parameter": dict(
                self.gradient_reference
            ),
            "initial_bound_occupancy_by_parameter": dict(
                self.initial_occupancy
            ),
            "maximum_bound_occupancy_by_parameter": dict(
                self.maximum_occupancy
            ),
            "median_projection_efficiency": (
                _linear_quantile(self.projection_efficiencies, 0.5)
                if self.projection_efficiencies
                else None
            ),
            "safety_failure": self.failure,
        }


def _bounded_initial_occupancy(runtime: Any) -> dict[str, float]:
    diagnostics = parameter_state_diagnostics(runtime.parameters)
    result = {
        name: float(record["combined_bound_occupancy"])
        for name, record in diagnostics.items()
        if record["bounded_gate"]
    }
    expected = set(
        _weight_names(canonical_parameter_name(item) for item in runtime.parameters)
    )
    if set(result) != expected:
        raise PerfectDiodeRuntimeError(
            "Expected initial occupancy for every bounded weight. "
            f"Provided value: {sorted(result)!r}."
        )
    return result


def _median_transition_projection(transition: Any) -> float | None:
    values = [
        float(item.projection_efficiency)
        for item in transition.parameters
        if bool(item.bounded_gate)
        and not bool(getattr(item, "proposal_is_numerically_zero", False))
    ]
    return _linear_quantile(values, 0.5) if values else None


def _bounded_update_scales(runtime: Any) -> tuple[dict[str, float], dict[str, float]]:
    parameters = _parameter_by_name(runtime.parameters)
    names = _weight_names(parameters)
    initial_rms = {name: _rms(parameters[name].state) for name in names}
    spans = {
        name: _finite(
            float(parameters[name].max_cond) - float(parameters[name].min_cond),
            f"conductance span for {name!r}",
            positive=True,
        )
        for name in names
    }
    if any(value <= 0.0 for value in initial_rms.values()):
        raise PerfectDiodeRuntimeError(
            "Expected every initial bounded-weight RMS to be positive."
        )
    return initial_rms, spans


def _transition_achieved_updates(
    transition: Any,
    *,
    initial_rms: Mapping[str, float],
    spans: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for item in transition.parameters:
        if not bool(item.bounded_gate):
            continue
        achieved_rms = _rms(item.post_projection - item.pre_update)
        result[item.name] = {
            "achieved_update_rms": achieved_rms,
            "achieved_relative_update": achieved_rms / initial_rms[item.name],
            "achieved_span_update": achieved_rms / spans[item.name],
        }
    if set(result) != set(initial_rms):
        raise PerfectDiodeRuntimeError(
            "Expected achieved-update diagnostics for every bounded weight. "
            f"Provided value: {sorted(result)!r}."
        )
    return result


def _summarize_achieved_updates(
    values: Mapping[str, Sequence[Mapping[str, float]]],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for name, records in values.items():
        if not records:
            continue
        relative = [
            _finite(
                record["achieved_relative_update"],
                f"achieved relative update for {name!r}",
                nonnegative=True,
            )
            for record in records
        ]
        span = [
            _finite(
                record["achieved_span_update"],
                f"achieved span update for {name!r}",
                nonnegative=True,
            )
            for record in records
        ]
        raw = [
            _finite(
                record["achieved_update_rms"],
                f"achieved update RMS for {name!r}",
                nonnegative=True,
            )
            for record in records
        ]
        summary[name] = {
            "step_count": len(records),
            "achieved_update_rms_median": _linear_quantile(raw, 0.5),
            "achieved_update_rms_q90": _linear_quantile(raw, 0.9),
            "achieved_relative_update_median": _linear_quantile(relative, 0.5),
            "achieved_relative_update_q90": _linear_quantile(relative, 0.9),
            "achieved_span_update_median": _linear_quantile(span, 0.5),
            "achieved_span_update_q90": _linear_quantile(span, 0.9),
        }
    return summary


def run_canary(
    runtime: Any,
    train_loader: Iterable[Any],
    *,
    learning_rates_by_parameter: Mapping[str, float],
    expected_source_indices: Sequence[Sequence[int]] | None = None,
    step_function: Any = training_step,
) -> dict[str, Any]:
    """Run one restarted constant-LR canary for at most exactly 640 steps."""

    rates = {
        str(name): _finite(value, f"learning rate for {name!r}", positive=True)
        for name, value in learning_rates_by_parameter.items()
    }
    initial_occupancy = _bounded_initial_occupancy(runtime)
    initial_rms, spans = _bounded_update_scales(runtime)
    monitor = SafetyMonitor(initial_occupancy)
    step_rows: list[dict[str, Any]] = []
    completed_steps = 0
    sample_count = 0
    last_loss: float | None = None
    failure: dict[str, Any] | None = None
    achieved_values: dict[str, list[dict[str, float]]] = {
        name: [] for name in initial_rms
    }
    iterator = iter(train_loader)
    for attempted_step in range(1, CANARY_STEPS + 1):
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise PerfectDiodeRuntimeError(
                f"Expected at least {CANARY_STEPS} canary minibatches. "
                f"Provided value: {completed_steps}."
            ) from exc
        step_succeeded = False
        try:
            result = step_function(
                runtime,
                batch,
                learning_rate=rates,
                restore=False,
                zero_proposal_epsilon=1.0e-12,
            )
            require_finite_optimizer_state(
                runtime.optimizer,
                context=f"{runtime.optimizer_name} canary state after "
                f"step {attempted_step}",
            )
            if expected_source_indices is not None:
                expected = tuple(
                    int(value)
                    for value in expected_source_indices[attempted_step - 1]
                )
                if tuple(result.source_indices) != expected:
                    raise PerfectDiodeRuntimeError(
                        "Expected canary minibatches to match the deterministic "
                        f"training stream at step {attempted_step}."
                    )
            failure = monitor.observe(
                loss=result.loss, transition=result.transition
            )
            step_succeeded = True
        except (
            LRStudyNumericalError,
            OptimizerStateNumericalError,
            FloatingPointError,
        ) as exc:
            failure = {
                "kind": "non_finite_optimizer_transition",
                "parameter": None,
                "onset_step": attempted_step,
                "confirmed_step": attempted_step,
                "detail": str(exc),
            }
        if failure is not None and monitor.failure is None:
            monitor.failure = failure
        if step_succeeded:
            completed_steps += 1
            sample_count += int(result.sample_count)
            last_loss = float(result.loss)
            achieved = _transition_achieved_updates(
                result.transition,
                initial_rms=initial_rms,
                spans=spans,
            )
            for name, record in achieved.items():
                achieved_values[name].append(record)
        if failure is not None:
            step_rows.append(
                {
                    "step": attempted_step,
                    "status": "safety_failure",
                    "loss": result.loss if step_succeeded else None,
                    "accuracy": result.accuracy if step_succeeded else None,
                    "sample_count": (
                        result.sample_count if step_succeeded else None
                    ),
                    "loss_ema": monitor.ema,
                    "median_projection_efficiency": (
                        _median_transition_projection(result.transition)
                        if step_succeeded
                        else None
                    ),
                    "failure_kind": failure["kind"],
                }
            )
            break
        step_rows.append(
            {
                "step": attempted_step,
                "status": "complete",
                "loss": result.loss,
                "accuracy": result.accuracy,
                "sample_count": result.sample_count,
                "loss_ema": monitor.ema,
                "median_projection_efficiency": _median_transition_projection(
                    result.transition
                ),
                "failure_kind": None,
            }
        )

    safety_clean = failure is None and completed_steps == CANARY_STEPS
    if failure is None and not safety_clean:
        raise PerfectDiodeRuntimeError(
            f"Expected a clean canary to stop at exactly {CANARY_STEPS} steps. "
            f"Provided value: {completed_steps}."
        )
    safety_summary = monitor.summary()
    return {
        "schema_version": "mnist-conv-perfectdiode-canary-result/v1",
        "status": "complete" if safety_clean else "safety_failure",
        "safety_clean": safety_clean,
        "official_test_read": False,
        "completed_steps": completed_steps,
        "expected_steps": CANARY_STEPS,
        "sample_count": sample_count,
        "last_loss": last_loss,
        "safety_failure": failure,
        "safety": safety_summary,
        "median_projection_efficiency": safety_summary[
            "median_projection_efficiency"
        ],
        "achieved_updates_by_parameter": _summarize_achieved_updates(
            achieved_values
        ),
        "raw_learning_rates_by_parameter": rates,
        "step_rows": step_rows,
    }


def _save_checkpoint_atomic(runtime: Any, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        runtime.save(temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def run_epoch_training(
    runtime: Any,
    bundle: Any,
    *,
    architecture: str,
    mode: str,
    learning_rates_by_parameter: Mapping[str, float],
    output_dir: str | Path,
    step_function: Any = training_step,
    validation_function: Any = evaluate_validation,
) -> dict[str, Any]:
    """Run an exact candidate or long confirmation from shared initialization."""

    contract = training_contract(architecture=architecture, mode=mode)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rates = {
        str(name): _finite(value, f"learning rate for {name!r}", positive=True)
        for name, value in learning_rates_by_parameter.items()
    }
    initial_occupancy = _bounded_initial_occupancy(runtime)
    initial_rms, spans = _bounded_update_scales(runtime)
    monitor = SafetyMonitor(initial_occupancy)
    expected_epochs = bundle.train_batch_indices(num_epochs=contract.epochs)
    bundle.reset_train_shuffle()
    validations: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    completed_steps = 0
    completed_examples = 0
    failure: dict[str, Any] | None = None
    best_validation_loss = math.inf
    best_validation_epoch: int | None = None
    best_checkpoint = destination / "best_validation.pt"
    final_checkpoint = destination / "final.pt"
    achieved_values: dict[str, list[dict[str, float]]] = {
        name: [] for name in initial_rms
    }
    achieved_by_epoch: list[dict[str, Any]] = []

    for epoch in range(1, contract.epochs + 1):
        epoch_steps = 0
        epoch_examples = 0
        epoch_loss = 0.0
        epoch_correct = 0.0
        epoch_achieved: dict[str, list[dict[str, float]]] = {
            name: [] for name in initial_rms
        }
        for batch_in_epoch, batch in enumerate(bundle.train_loader, start=1):
            attempted_step = completed_steps + 1
            step_succeeded = False
            try:
                result = step_function(
                    runtime,
                    batch,
                    learning_rate=rates,
                    restore=False,
                    zero_proposal_epsilon=1.0e-12,
                )
                require_finite_optimizer_state(
                    runtime.optimizer,
                    context=f"{runtime.optimizer_name} {mode} state after "
                    f"step {attempted_step}",
                )
                expected_indices = tuple(
                    int(value)
                    for value in expected_epochs[epoch - 1][batch_in_epoch - 1]
                )
                if tuple(result.source_indices) != expected_indices:
                    raise PerfectDiodeRuntimeError(
                        "Expected candidate minibatches to match the deterministic "
                        f"stream at epoch {epoch}, batch {batch_in_epoch}."
                    )
                failure = monitor.observe(
                    loss=result.loss, transition=result.transition
                )
                step_succeeded = True
            except (
                LRStudyNumericalError,
                OptimizerStateNumericalError,
                FloatingPointError,
            ) as exc:
                failure = {
                    "kind": "non_finite_optimizer_transition",
                    "parameter": None,
                    "onset_step": attempted_step,
                    "confirmed_step": attempted_step,
                    "detail": str(exc),
                }
            if failure is not None and monitor.failure is None:
                monitor.failure = failure
            if step_succeeded:
                completed_steps += 1
                epoch_steps += 1
                completed_examples += int(result.sample_count)
                epoch_examples += int(result.sample_count)
                epoch_loss += float(result.loss) * int(result.sample_count)
                epoch_correct += float(result.accuracy) * int(result.sample_count)
                achieved = _transition_achieved_updates(
                    result.transition,
                    initial_rms=initial_rms,
                    spans=spans,
                )
                for name, achieved_record in achieved.items():
                    achieved_values[name].append(achieved_record)
                    epoch_achieved[name].append(achieved_record)
            if failure is not None:
                step_rows.append(
                    {
                        "step": attempted_step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_in_epoch,
                        "status": "safety_failure",
                        "loss": result.loss if step_succeeded else None,
                        "accuracy": result.accuracy if step_succeeded else None,
                        "sample_count": (
                            result.sample_count if step_succeeded else None
                        ),
                        "loss_ema": monitor.ema,
                        "median_projection_efficiency": (
                            _median_transition_projection(result.transition)
                            if step_succeeded
                            else None
                        ),
                        "failure_kind": failure["kind"],
                    }
                )
                break

            step_rows.append(
                {
                    "step": completed_steps,
                    "epoch": epoch,
                    "batch_in_epoch": batch_in_epoch,
                    "status": "complete",
                    "loss": result.loss,
                    "accuracy": result.accuracy,
                    "sample_count": result.sample_count,
                    "loss_ema": monitor.ema,
                    "median_projection_efficiency": (
                        _median_transition_projection(result.transition)
                    ),
                    "failure_kind": None,
                }
            )
        if failure is not None:
            break
        if epoch_steps != contract.steps_per_epoch or epoch_examples != 55_000:
            raise PerfectDiodeRuntimeError(
                f"Expected epoch {epoch} to contain exactly "
                f"{contract.steps_per_epoch} steps and 55,000 examples. "
                f"Provided value: steps={epoch_steps}, examples={epoch_examples}."
            )
        validation = validation_function(runtime, bundle.validation_loader)
        source_indices = tuple(int(value) for value in validation["source_indices"])
        if source_indices != tuple(int(value) for value in bundle.validation_indices):
            raise PerfectDiodeRuntimeError(
                f"Expected epoch {epoch} validation to read the exact held-out "
                "MNIST-train indices in deterministic order."
            )
        record = {
            "epoch": epoch,
            "step": completed_steps,
            "sample_count": int(validation["sample_count"]),
            "loss": _finite(validation["loss"], "validation loss"),
            "accuracy": _finite(validation["accuracy"], "validation accuracy"),
            "train_loss": epoch_loss / epoch_examples,
            "train_accuracy": epoch_correct / epoch_examples,
            "validation_indices_sha256": bundle.validation_indices_hash,
        }
        validations.append(record)
        achieved_by_epoch.append(
            {
                "epoch": epoch,
                "step": completed_steps,
                "parameters": _summarize_achieved_updates(epoch_achieved),
            }
        )
        if record["loss"] < best_validation_loss:
            best_validation_loss = record["loss"]
            best_validation_epoch = epoch
            _save_checkpoint_atomic(runtime, best_checkpoint)

    _save_checkpoint_atomic(runtime, final_checkpoint)
    if best_validation_epoch is None:
        _save_checkpoint_atomic(runtime, best_checkpoint)
    completed = failure is None and completed_steps == contract.total_steps
    if failure is None and not completed:
        raise PerfectDiodeRuntimeError(
            f"Expected successful {mode} to complete exactly "
            f"{contract.total_steps} steps. Provided value: {completed_steps}."
        )
    if completed and len(validations) != contract.epochs:
        raise PerfectDiodeRuntimeError(
            f"Expected {contract.epochs} epoch validations. "
            f"Provided value: {len(validations)}."
        )
    final_validation = validations[-1] if completed else None
    accuracy_gate = (
        completed
        and final_validation is not None
        and float(final_validation["accuracy"]) >= 0.90
    )
    safety_summary = monitor.summary()
    result = {
        "schema_version": "mnist-conv-perfectdiode-training-result/v1",
        "mode": mode,
        "architecture": architecture,
        "status": "complete" if completed else "safety_failure",
        "training_completed": completed,
        "safety_admissible": completed,
        "official_test_read": False,
        "completed_steps": completed_steps,
        "expected_steps": contract.total_steps,
        "completed_examples": completed_examples,
        "epochs_completed": len(validations),
        "expected_epochs": contract.epochs,
        "raw_learning_rates_by_parameter": rates,
        "safety_failure": failure,
        "safety": safety_summary,
        "median_projection_efficiency": safety_summary[
            "median_projection_efficiency"
        ],
        "achieved_updates_by_parameter": _summarize_achieved_updates(
            achieved_values
        ),
        "achieved_updates_by_epoch": achieved_by_epoch,
        "final_validation_loss": (
            None if final_validation is None else final_validation["loss"]
        ),
        "final_validation_accuracy": (
            None if final_validation is None else final_validation["accuracy"]
        ),
        "inclusive_90_percent_accuracy_gate_passed": accuracy_gate,
        "passing_candidate": bool(mode == "candidate" and accuracy_gate),
        "best_validation_loss": (
            None if best_validation_epoch is None else best_validation_loss
        ),
        "best_validation_epoch": best_validation_epoch,
        "checkpoints": {
            "best_validation": {
                "path": best_checkpoint.name,
                "sha256": sha256_file(best_checkpoint),
            },
            "final": {
                "path": final_checkpoint.name,
                "sha256": sha256_file(final_checkpoint),
            },
        },
        "validations": validations,
        "step_rows": step_rows,
    }
    return result


def require_official_test_excluded(
    value: Mapping[str, Any],
    *,
    require_result_marker: bool = False,
) -> None:
    """Fail if a study or result permits or claims an official-test read."""

    if not isinstance(value, Mapping):
        raise TypeError(
            "Expected official-test exclusion input to be a mapping. "
            f"Provided value: {type(value).__name__}."
        )
    dataset = value.get("dataset")
    if isinstance(dataset, Mapping):
        official = dataset.get("official_test")
        if not isinstance(official, Mapping):
            raise ValueError(
                "Expected dataset.official_test to be an explicit mapping. "
                f"Provided value: {official!r}."
            )
        if official.get("enabled") is not False or official.get("read_allowed") is not False:
            raise ValueError(
                "Expected dataset.official_test.enabled and read_allowed to be false. "
                f"Provided value: {official!r}."
            )
    if "official_test_read" in value and value["official_test_read"] is not False:
        raise ValueError(
            "Expected official_test_read to be false. "
            f"Provided value: {value['official_test_read']!r}."
        )
    if require_result_marker and value.get("official_test_read") is not False:
        raise ValueError(
            "Expected a result to record official_test_read=false. "
            f"Provided value: {value.get('official_test_read')!r}."
        )


def rho_cell_id(
    *,
    study_id: str,
    surface_id: str,
    rho_conv: float,
    rho_dense: float,
) -> str:
    """Return the shared content-derived identity of one rho cell."""

    payload = {
        "schema_version": "mnist-conv-perfectdiode-rho-cell/v1",
        "study_id": str(study_id),
        "surface_id": str(surface_id),
        "rho_conv": _finite(rho_conv, "rho_conv", positive=True),
        "rho_dense": _finite(rho_dense, "rho_dense", positive=True),
    }
    return "pdcell_" + sha256_json(payload)


def _study_and_spec(
    study: Mapping[str, Any] | PerfectDiodeHparamStudySpec,
) -> tuple[dict[str, Any], PerfectDiodeHparamStudySpec]:
    spec = (
        study
        if isinstance(study, PerfectDiodeHparamStudySpec)
        else PerfectDiodeHparamStudySpec.from_dict(study)
    )
    data = spec.data
    require_official_test_excluded(data)
    return data, spec


def _row_by_id(study: Mapping[str, Any], row_id: str) -> dict[str, Any]:
    rows = [row for row in study["rows"] if row["row_id"] == row_id]
    if len(rows) != 1:
        raise ValueError(
            "Expected row_id to identify exactly one perfect-diode row. "
            f"Provided value: {row_id!r}."
        )
    return copy.deepcopy(rows[0])


def _optimizer_contract(
    study: Mapping[str, Any], optimizer_name: str
) -> dict[str, Any]:
    key = str(optimizer_name).lower()
    arms = study.get("optimizer_arms")
    if not isinstance(arms, Mapping) or key not in arms:
        raise ValueError(
            "Expected optimizer_name to identify an explicit optimizer arm. "
            f"Provided value: {optimizer_name!r}."
        )
    return copy.deepcopy(dict(arms[key]))


def _atomic_model_save(runtime: Any, path: Path) -> Path:
    return _save_checkpoint_atomic(runtime, path)


def _artifact_record(path: Path, *, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _combine_indexed_batches(batches: Sequence[Any]) -> tuple[Any, Any, Any]:
    import torch

    if not batches:
        raise ValueError(
            "Expected at least one indexed batch to combine. Provided value: []."
        )
    unpacked = [_unpack_batch(batch) for batch in batches]
    return (
        torch.cat([value[0] for value in unpacked], dim=0),
        torch.cat([value[1] for value in unpacked], dim=0),
        torch.as_tensor(
            [index for value in unpacked for index in value[2]],
            dtype=torch.int64,
        ),
    )


def _materialize_train_batches(
    bundle: Any,
    *,
    count: int,
    combine: int = 1,
) -> tuple[Any, ...]:
    required = _positive_integer(count, "batch count")
    group_size = _positive_integer(combine, "combine")
    bundle.reset_train_shuffle()
    iterator = iter(bundle.train_loader)
    result: list[Any] = []
    for output_index in range(required):
        pieces: list[Any] = []
        for _ in range(group_size):
            try:
                pieces.append(next(iterator))
            except StopIteration as exc:
                raise PerfectDiodeRuntimeError(
                    f"Expected {required * group_size} deterministic training "
                    f"minibatches. Provided value: "
                    f"{output_index * group_size + len(pieces)}."
                ) from exc
        result.append(
            pieces[0] if group_size == 1 else _combine_indexed_batches(pieces)
        )
    return tuple(result)


def _release_runtime(runtime: Any) -> None:
    import torch

    del runtime
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_architecture_assets(
    study: Mapping[str, Any] | PerfectDiodeHparamStudySpec,
    *,
    output_dir: str | Path,
    architecture: str,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    """Publish one ordinary-MNIST split and shared architecture checkpoint."""

    data, spec = _study_and_spec(study)
    if architecture not in {"conv1", "conv2"}:
        raise ValueError(
            "Expected architecture to be 'conv1' or 'conv2'. "
            f"Provided value: {architecture!r}."
        )
    rows = [
        copy.deepcopy(row)
        for row in data["rows"]
        if row["architecture"] == architecture
    ]
    if len(rows) != 3 or {row["scheme"] for row in rows} != {
        "baseline",
        "ours",
        "legacy",
    }:
        raise PerfectDiodeRuntimeError(
            f"Expected exactly three scheme rows for {architecture!r}."
        )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    bundle = build_loader_bundle(
        data,
        data_root=data_root,
        download=download,
        return_source_indices=True,
    )
    indices_path = destination / "split_indices.json"
    indices = {
        "schema_version": "mnist-conv-perfectdiode-split-indices/v1",
        "source_split": "mnist_train",
        "official_test_read": False,
        "train_indices": list(bundle.train_indices),
        "validation_indices": list(bundle.validation_indices),
        "train_indices_sha256": bundle.train_indices_hash,
        "validation_indices_sha256": bundle.validation_indices_hash,
    }
    atomic_write_json(indices_path, indices, canonical=True)
    split_provenance = bundle.provenance(num_epochs=CONV2_CONFIRM_EPOCHS)
    split_provenance.pop("train_indices")
    split_provenance.pop("validation_indices")
    split_provenance.update(
        {
            "schema_version": "mnist-conv-perfectdiode-split-provenance/v1",
            "official_test_read": False,
            "split_indices_file": indices_path.name,
            "split_indices_sha256": sha256_file(indices_path),
        }
    )
    provenance_path = destination / "split_provenance.json"
    atomic_write_json(provenance_path, split_provenance, canonical=True)

    optimizer_contract = _optimizer_contract(data, "sgd")
    natural_digests: dict[str, str] = {}
    checkpoint_path = destination / "initialization.pt"
    representative_runtime = None
    for index, row in enumerate(rows):
        runtime = build_model_runtime(
            data,
            row,
            device=device,
            learning_rate=1.0,
            optimizer_contract=optimizer_contract,
        )
        digest = parameter_tensor_digest(runtime.parameters)
        natural_digests[row["row_id"]] = digest
        if index == 0:
            representative_runtime = runtime
            _atomic_model_save(runtime, checkpoint_path)
        else:
            runtime = None
            _release_runtime(None)
    if len(set(natural_digests.values())) != 1:
        raise PerfectDiodeRuntimeError(
            "Expected reset-counter seed-0 initialization tensors to match "
            f"across all {architecture} schemes. Provided value: {natural_digests!r}."
        )
    assert representative_runtime is not None
    parameter_names = [
        canonical_parameter_name(parameter)
        for parameter in representative_runtime.parameters
    ]
    initial_diagnostics = parameter_state_diagnostics(
        representative_runtime.parameters
    )
    representative_runtime = None
    runtime = None
    _release_runtime(None)

    checkpoint_sha = sha256_file(checkpoint_path)
    initialization = {
        "schema_version": "mnist-conv-perfectdiode-initialization/v1",
        "architecture": architecture,
        "model_seed": int(data["model"]["model_seed"]),
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha,
        "parameter_tensor_sha256": next(iter(natural_digests.values())),
        "parameter_names": parameter_names,
        "parameter_diagnostics": initial_diagnostics,
        "natural_parameter_tensor_sha256_by_row": natural_digests,
        "shared_across_schemes_and_optimizers": True,
        "official_test_read": False,
    }
    initialization_path = destination / "initialization.json"
    atomic_write_json(initialization_path, initialization, canonical=True)

    result = {
        "schema_version": "mnist-conv-perfectdiode-assets-result/v1",
        "status": "complete",
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "architecture": architecture,
        "official_test_read": False,
        "train_size": len(bundle.train_indices),
        "validation_size": len(bundle.validation_indices),
        "train_indices_sha256": bundle.train_indices_hash,
        "validation_indices_sha256": bundle.validation_indices_hash,
        "initialization_checkpoint_sha256": checkpoint_sha,
        "initialization_tensor_sha256": initialization[
            "parameter_tensor_sha256"
        ],
        "artifacts": [
            _artifact_record(path, base=destination)
            for path in (
                indices_path,
                provenance_path,
                checkpoint_path,
                initialization_path,
            )
        ],
    }
    result_path = destination / "result.json"
    atomic_write_json(result_path, result, canonical=True)
    return result


@dataclass(frozen=True)
class LoadedPerfectDiodeAssets:
    architecture: str
    entry_dir: Path
    checkpoint_path: Path
    initialization: dict[str, Any]
    bundle: Any


def _load_architecture_assets(
    study: Mapping[str, Any],
    spec: PerfectDiodeHparamStudySpec,
    *,
    shard_dir: Path,
    architecture: str,
    asset_entry_id: str,
    data_root: str | Path,
    download: bool,
) -> LoadedPerfectDiodeAssets:
    entry_dir = (
        shard_dir / "stages" / "assets" / "entries" / str(asset_entry_id)
    )
    result = read_json(entry_dir / "result.json")
    require_official_test_excluded(result, require_result_marker=True)
    if (
        result.get("study_id") != spec.study_id
        or result.get("config_sha256") != spec.config_sha256
        or result.get("architecture") != architecture
    ):
        raise PerfectDiodeRuntimeError(
            "Expected downstream assets to match the exact study and architecture."
        )
    checkpoint = entry_dir / "initialization.pt"
    initialization = read_json(entry_dir / "initialization.json")
    if sha256_file(checkpoint) != initialization["checkpoint_sha256"]:
        raise PerfectDiodeRuntimeError(
            f"Expected initialization checkpoint hash to verify: {checkpoint}."
        )
    bundle = build_loader_bundle(
        study,
        data_root=data_root,
        download=download,
        return_source_indices=True,
    )
    indices = read_json(entry_dir / "split_indices.json")
    if (
        tuple(indices["train_indices"]) != tuple(bundle.train_indices)
        or tuple(indices["validation_indices"]) != tuple(bundle.validation_indices)
        or indices["train_indices_sha256"] != bundle.train_indices_hash
        or indices["validation_indices_sha256"] != bundle.validation_indices_hash
    ):
        raise PerfectDiodeRuntimeError(
            "Expected regenerated ordinary-MNIST split to match staged assets exactly."
        )
    return LoadedPerfectDiodeAssets(
        architecture=architecture,
        entry_dir=entry_dir,
        checkpoint_path=checkpoint,
        initialization=initialization,
        bundle=bundle,
    )


def _entry_output_dir(
    shard_dir: Path, stage: str, entry_id: str
) -> Path:
    output = shard_dir / "stages" / stage / "entries" / str(entry_id)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _source_shard_root(
    shard_dir: Path, payload: Mapping[str, Any]
) -> Path:
    raw = payload.get("source_shard")
    if raw is None:
        return shard_dir
    relative = Path(str(raw))
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] != "shards"
        or relative.parts[1] in {"", ".", ".."}
    ):
        raise ValueError(
            "Expected payload.source_shard to match 'shards/<host>' without "
            f"traversal. Provided value: {raw!r}."
        )
    resolved = (shard_dir / relative).resolve()
    try:
        resolved.relative_to(shard_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            "Expected payload.source_shard to stay inside the aggregate root. "
            f"Provided value: {raw!r}."
        ) from exc
    return resolved


def _probe_path(
    shard_dir: Path,
    payload: Mapping[str, Any],
) -> Path:
    explicit = payload.get("probe_result_path")
    if explicit is not None:
        path = Path(str(explicit))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "Expected payload.probe_result_path to be a traversal-free "
                f"relative path. Provided value: {explicit!r}."
            )
        resolved = (shard_dir / path).resolve()
        try:
            resolved.relative_to(shard_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                "Expected payload.probe_result_path to stay inside the shard root. "
                f"Provided value: {explicit!r}."
            ) from exc
        return resolved
    probe_entry_id = payload.get("probe_entry_id")
    if not isinstance(probe_entry_id, str) or not probe_entry_id:
        raise ValueError(
            "Expected payload.probe_entry_id or probe_result_path for a "
            f"rho-stage entry. Provided value: {probe_entry_id!r}."
        )
    return (
        _source_shard_root(shard_dir, payload)
        / "stages"
        / "optimizer_probe"
        / "entries"
        / probe_entry_id
        / "result.json"
    )


def _load_stable_probe(
    shard_dir: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    path = _probe_path(shard_dir, payload)
    probe = read_json(path)
    require_official_test_excluded(probe, require_result_marker=True)
    if probe.get("probe_stable") is not True:
        raise PerfectDiodeRuntimeError(
            f"Expected a stable optimizer probe. Provided value: {path}."
        )
    return probe


def _build_fresh_runtime(
    study: Mapping[str, Any],
    row: Mapping[str, Any],
    assets: LoadedPerfectDiodeAssets,
    *,
    optimizer_name: str,
    rates: Mapping[str, float],
    device: str,
) -> Any:
    runtime = build_model_runtime(
        study,
        row,
        device=device,
        initialization_checkpoint=assets.checkpoint_path,
        learning_rate=rates,
        optimizer_contract=_optimizer_contract(study, optimizer_name),
    )
    if parameter_tensor_digest(runtime.parameters) != assets.initialization[
        "parameter_tensor_sha256"
    ]:
        raise PerfectDiodeRuntimeError(
            "Expected every run to restart from the shared seed-0 initialization."
        )
    return runtime


def _write_step_log(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    fields = [
        "cell_id",
        "phase",
        "step",
        "epoch",
        "batch_in_epoch",
        "status",
        "loss",
        "accuracy",
        "sample_count",
        "loss_ema",
        "median_projection_efficiency",
        "failure_kind",
    ]
    return atomic_write_csv(
        path,
        fields,
        (
            {
                field: (
                    ""
                    if row.get(field) is None
                    else row.get(field)
                )
                for field in fields
            }
            for row in rows
        ),
    )


def _runtime_benchmark_start(device: str) -> tuple[float, bool]:
    import torch

    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    return time.monotonic(), cuda


def _runtime_benchmark_finish(
    started: float, cuda: bool, completed_steps: int
) -> dict[str, Any]:
    import torch

    if cuda:
        torch.cuda.synchronize()
    elapsed = max(time.monotonic() - started, 1.0e-12)
    return {
        "elapsed_seconds": elapsed,
        "successful_steps_per_second": completed_steps / elapsed,
        "cuda_peak_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if cuda else None
        ),
        "cuda_peak_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if cuda else None
        ),
    }


def _common_loaded_assets(
    study: Mapping[str, Any],
    spec: PerfectDiodeHparamStudySpec,
    shard_dir: Path,
    payload: Mapping[str, Any],
    *,
    data_root: str | Path,
    download: bool,
) -> tuple[dict[str, Any], LoadedPerfectDiodeAssets]:
    row = _row_by_id(study, str(payload.get("row_id")))
    asset_entry_id = str(payload.get("asset_entry_id") or row["architecture"])
    asset_root = _source_shard_root(shard_dir, payload)
    assets = _load_architecture_assets(
        study,
        spec,
        shard_dir=asset_root,
        architecture=row["architecture"],
        asset_entry_id=asset_entry_id,
        data_root=data_root,
        download=download,
    )
    return row, assets


def _execute_security_entry(
    study: Mapping[str, Any],
    spec: PerfectDiodeHparamStudySpec,
    shard_dir: Path,
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    row, assets = _common_loaded_assets(
        study,
        spec,
        shard_dir,
        payload,
        data_root=data_root,
        download=download,
    )
    batches = _materialize_train_batches(
        assets.bundle, count=SECURITY_BATCHES, combine=2
    )
    if any(int(batch[0].shape[0]) != SECURITY_BATCH_SIZE for batch in batches):
        raise PerfectDiodeRuntimeError(
            f"Expected every security batch to contain {SECURITY_BATCH_SIZE} examples."
        )
    operational_runtime = build_model_runtime(
        study,
        row,
        device=device,
        initialization_checkpoint=assets.checkpoint_path,
        learning_rate=1.0,
        optimizer_contract=_optimizer_contract(study, "sgd"),
    )
    operational = collect_runtime_gradients(operational_runtime, batches)
    del operational_runtime
    reference_row = copy.deepcopy(row)
    reference_row["inference_iterations"] = int(
        row["reference_inference_iterations"]
    )
    reference_row["training_iterations"] = int(
        row["reference_training_iterations"]
    )
    reference_runtime = build_model_runtime(
        study,
        reference_row,
        device=device,
        initialization_checkpoint=assets.checkpoint_path,
        learning_rate=1.0,
        optimizer_contract=_optimizer_contract(study, "sgd"),
    )
    reference = collect_runtime_gradients(reference_runtime, batches)
    del reference_runtime
    result = compare_fixed_tk_gradient_security(
        operational,
        reference,
        operational_t=int(row["inference_iterations"]),
        operational_k=int(row["training_iterations"]),
        reference_t=int(row["reference_inference_iterations"]),
        reference_k=int(row["reference_training_iterations"]),
    )
    result.update(
        {
            "study_id": spec.study_id,
            "config_sha256": spec.config_sha256,
            "entry_id": str(payload["entry_id"]),
            "row_id": row["row_id"],
            "architecture": row["architecture"],
            "scheme": row["scheme"],
            "cohort_examples": SECURITY_EXAMPLES,
            "batch_size": SECURITY_BATCH_SIZE,
            "initialization_checkpoint_sha256": assets.initialization[
                "checkpoint_sha256"
            ],
            "initialization_tensor_sha256": assets.initialization[
                "parameter_tensor_sha256"
            ],
            "train_indices_sha256": assets.bundle.train_indices_hash,
        }
    )
    csv_rows = []
    json_diagnostics = []
    for record in result["parameter_diagnostics"]:
        csv_rows.append(
            {
                key: value
                for key, value in record.items()
                if key != "batches"
            }
        )
        json_diagnostics.append(
            {
                key: value
                for key, value in record.items()
                if key != "batches"
            }
        )
    fields = [
        "parameter",
        "batch_count",
        "gradient_l2_mean",
        "reference_gradient_l2_mean",
        "relative_gradient_l2_norm_delta",
        "gradient_zero_fraction_mean",
        "reference_gradient_zero_fraction_mean",
        "absolute_zero_fraction_delta",
        "gradient_vector_cosine_mean",
        "maximum_batch_norm_delta",
        "maximum_batch_zero_fraction_delta",
        "minimum_batch_cosine",
        "passed",
    ]
    diagnostics_path = output_dir / "parameter_diagnostics.csv"
    atomic_write_csv(diagnostics_path, fields, csv_rows)
    result["parameter_diagnostics"] = json_diagnostics
    result["artifacts"] = [
        _artifact_record(diagnostics_path, base=output_dir)
    ]
    require_official_test_excluded(result, require_result_marker=True)
    atomic_write_json(output_dir / "result.json", result, canonical=True)
    return result


def _execute_probe_entry(
    study: Mapping[str, Any],
    spec: PerfectDiodeHparamStudySpec,
    shard_dir: Path,
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    row, assets = _common_loaded_assets(
        study,
        spec,
        shard_dir,
        payload,
        data_root=data_root,
        download=download,
    )
    optimizer_name = str(payload["optimizer"]).lower()
    batches = _materialize_train_batches(
        assets.bundle, count=PROBE_BATCH_COUNTS[-1]
    )
    runtime = build_model_runtime(
        study,
        row,
        device=device,
        initialization_checkpoint=assets.checkpoint_path,
        learning_rate=1.0,
        optimizer_contract=_optimizer_contract(study, optimizer_name),
    )
    started, cuda = _runtime_benchmark_start(device)
    measured = measure_adaptive_optimizer_probe(runtime, batches)
    benchmark = _runtime_benchmark_finish(
        started, cuda, int(measured["used_batches"])
    )
    del runtime
    raw_records = measured.pop("records")
    minibatches = {
        "schema_version": "mnist-conv-perfectdiode-probe-minibatches/v1",
        "official_test_read": False,
        "row_id": row["row_id"],
        "optimizer": optimizer_name,
        "used_batches": measured["used_batches"],
        "train_indices_sha256": assets.bundle.train_indices_hash,
        "source_indices_by_batch": [
            record["source_indices"] for record in raw_records
        ],
    }
    minibatches["batch_order_sha256"] = sha256_json(
        minibatches["source_indices_by_batch"]
    )
    minibatches_path = output_dir / "minibatches.json"
    atomic_write_json(minibatches_path, minibatches, canonical=True)
    diagnostic_rows = []
    for name in measured["parameter_names"]:
        split = measured["split_half_statistics"][name]
        diagnostic_rows.append(
            {
                "parameter": name,
                "unit_statistic": (
                    "q90" if name.startswith("Bias_") else "median"
                ),
                "unit": (
                    measured["bias_q90_unit_by_parameter"][name]
                    if name.startswith("Bias_")
                    else measured["normalization_unit_by_weight"][name]
                ),
                "first_half_unit": split["first_half"],
                "second_half_unit": split["second_half"],
                "split_half_relative_difference": split[
                    "relative_difference"
                ],
                "stable": name not in measured["unstable_parameters"],
            }
        )
    diagnostics_path = output_dir / "parameter_diagnostics.csv"
    atomic_write_csv(
        diagnostics_path,
        [
            "parameter",
            "unit_statistic",
            "unit",
            "first_half_unit",
            "second_half_unit",
            "split_half_relative_difference",
            "stable",
        ],
        diagnostic_rows,
    )
    measured.update(
        {
            "study_id": spec.study_id,
            "config_sha256": spec.config_sha256,
            "entry_id": str(payload["entry_id"]),
            "row_id": row["row_id"],
            "architecture": row["architecture"],
            "scheme": row["scheme"],
            "optimizer": optimizer_name,
            "initialization_checkpoint_sha256": assets.initialization[
                "checkpoint_sha256"
            ],
            "initialization_tensor_sha256": assets.initialization[
                "parameter_tensor_sha256"
            ],
            "train_indices_sha256": assets.bundle.train_indices_hash,
            "probe_minibatch_order_sha256": minibatches[
                "batch_order_sha256"
            ],
            "benchmark": benchmark,
            "artifacts": [
                _artifact_record(path, base=output_dir)
                for path in (minibatches_path, diagnostics_path)
            ],
        }
    )
    require_official_test_excluded(measured, require_result_marker=True)
    atomic_write_json(output_dir / "result.json", measured, canonical=True)
    return measured


def _resolve_rates(
    probe: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    rho_conv: float,
    rho_dense: float,
) -> dict[str, float]:
    derived = derive_raw_learning_rates(
        probe, rho_conv=rho_conv, rho_dense=rho_dense
    )
    provided = payload.get("raw_learning_rates_by_parameter")
    if provided is None:
        return derived
    if not isinstance(provided, Mapping) or set(provided) != set(derived):
        raise ValueError(
            "Expected payload raw LR vector for every scientific parameter. "
            f"Provided value: {provided!r}."
        )
    normalized = {
        str(name): _finite(
            value, f"provided learning rate for {name!r}", positive=True
        )
        for name, value in provided.items()
    }
    for name, expected in derived.items():
        if not math.isclose(
            normalized[name], expected, rel_tol=1.0e-12, abs_tol=0.0
        ):
            raise PerfectDiodeRuntimeError(
                f"Expected provided LR for {name!r} to match the probe-derived "
                f"value exactly. Provided value: {normalized[name]!r}; "
                f"derived value: {expected!r}."
            )
    return normalized


def _run_canary_cell(
    study: Mapping[str, Any],
    spec: PerfectDiodeHparamStudySpec,
    row: Mapping[str, Any],
    assets: LoadedPerfectDiodeAssets,
    probe: Mapping[str, Any],
    *,
    surface_id: str,
    optimizer_name: str,
    rho_conv: float,
    rho_dense: float,
    device: str,
    cell_id: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identifier = cell_id or rho_cell_id(
        study_id=spec.study_id,
        surface_id=surface_id,
        rho_conv=rho_conv,
        rho_dense=rho_dense,
    )
    rates = derive_raw_learning_rates(
        probe, rho_conv=rho_conv, rho_dense=rho_dense
    )
    runtime = _build_fresh_runtime(
        study,
        row,
        assets,
        optimizer_name=optimizer_name,
        rates=rates,
        device=device,
    )
    assets.bundle.reset_train_shuffle()
    expected_batches = assets.bundle.train_batch_indices(num_epochs=1)[0][
        :CANARY_STEPS
    ]
    started, cuda = _runtime_benchmark_start(device)
    raw = run_canary(
        runtime,
        assets.bundle.train_loader,
        learning_rates_by_parameter=rates,
        expected_source_indices=expected_batches,
    )
    benchmark = _runtime_benchmark_finish(
        started, cuda, int(raw["completed_steps"])
    )
    del runtime
    rows = raw.pop("step_rows")
    raw.update(
        {
            "cell_id": identifier,
            "surface_id": surface_id,
            "row_id": row["row_id"],
            "architecture": row["architecture"],
            "scheme": row["scheme"],
            "optimizer": optimizer_name,
            "rho_conv": float(rho_conv),
            "rho_dense": float(rho_dense),
            "study_id": spec.study_id,
            "config_sha256": spec.config_sha256,
            "probe_entry_id": probe.get("entry_id"),
            "initialization_checkpoint_sha256": assets.initialization[
                "checkpoint_sha256"
            ],
            "initialization_tensor_sha256": assets.initialization[
                "parameter_tensor_sha256"
            ],
            "benchmark": benchmark,
        }
    )
    return raw, rows


def _execute_core_canary_entry(
    study: Mapping[str, Any],
    spec: PerfectDiodeHparamStudySpec,
    shard_dir: Path,
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    row, assets = _common_loaded_assets(
        study,
        spec,
        shard_dir,
        payload,
        data_root=data_root,
        download=download,
    )
    optimizer_name = str(payload["optimizer"]).lower()
    surface_id = str(
        payload.get("surface_id")
        or f"{row['row_id']}--{optimizer_name}"
    )
    probe = _load_stable_probe(shard_dir, payload)
    attempts: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    safe_result: dict[str, Any] | None = None
    for attempt in range(MAX_CENTER_ATTEMPTS):
        rho_conv, rho_dense = safe_center(attempt)
        cell, rows = _run_canary_cell(
            study,
            spec,
            row,
            assets,
            probe,
            surface_id=surface_id,
            optimizer_name=optimizer_name,
            rho_conv=rho_conv,
            rho_dense=rho_dense,
            device=device,
        )
        attempts.append(
            {
                "attempt": attempt,
                **cell,
            }
        )
        all_rows.extend(
            {**step, "cell_id": cell["cell_id"], "phase": "center_attempt"}
            for step in rows
        )
        if cell["safety_clean"]:
            safe_result = cell
            break

    cells: list[dict[str, Any]] = []
    if safe_result is not None:
        center_pair = (
            float(safe_result["rho_conv"]),
            float(safe_result["rho_dense"]),
        )
        attempts_by_cell_id = {
            attempt["cell_id"]: attempt for attempt in attempts
        }
        for rho_conv, rho_dense in core_grid(*center_pair):
            grid_cell_id = rho_cell_id(
                study_id=spec.study_id,
                surface_id=surface_id,
                rho_conv=rho_conv,
                rho_dense=rho_dense,
            )
            if grid_cell_id in attempts_by_cell_id:
                reused = copy.deepcopy(attempts_by_cell_id[grid_cell_id])
                reused.pop("attempt", None)
                reused["reused_from_center_attempt"] = True
                cells.append(reused)
                continue
            cell, rows = _run_canary_cell(
                study,
                spec,
                row,
                assets,
                probe,
                surface_id=surface_id,
                optimizer_name=optimizer_name,
                rho_conv=rho_conv,
                rho_dense=rho_dense,
                device=device,
            )
            cell["reused_from_center_attempt"] = False
            cells.append(cell)
            all_rows.extend(
                {**step, "cell_id": cell["cell_id"], "phase": "core_grid"}
                for step in rows
            )
    result = {
        "schema_version": "mnist-conv-perfectdiode-core-canary-result/v1",
        "status": "complete" if safe_result is not None else "unresolved_no_safe_center",
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "entry_id": str(payload["entry_id"]),
        "surface_id": surface_id,
        "row_id": row["row_id"],
        "architecture": row["architecture"],
        "scheme": row["scheme"],
        "optimizer": optimizer_name,
        "official_test_read": False,
        "safe_center": (
            None
            if safe_result is None
            else {
                "rho_conv": safe_result["rho_conv"],
                "rho_dense": safe_result["rho_dense"],
                "cell_id": safe_result["cell_id"],
            }
        ),
        "center_attempts": attempts,
        "cells": cells,
        "center_reused_as_core": safe_result is not None,
        "initialization_checkpoint_sha256": assets.initialization[
            "checkpoint_sha256"
        ],
        "initialization_tensor_sha256": assets.initialization[
            "parameter_tensor_sha256"
        ],
    }
    step_path = _write_step_log(output_dir / "step_log.csv", all_rows)
    result["artifacts"] = [_artifact_record(step_path, base=output_dir)]
    atomic_write_json(output_dir / "result.json", result, canonical=True)
    return result


def _execute_preflight_canary_entry(
    study: Mapping[str, Any],
    spec: PerfectDiodeHparamStudySpec,
    shard_dir: Path,
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    """Run only the initial Adam center as a non-canonical host preflight."""

    row, assets = _common_loaded_assets(
        study,
        spec,
        shard_dir,
        payload,
        data_root=data_root,
        download=download,
    )
    optimizer_name = str(payload.get("optimizer", "adam")).lower()
    if optimizer_name != "adam":
        raise ValueError(
            "Expected preflight_canary optimizer to be exactly 'adam'. "
            f"Provided value: {optimizer_name!r}."
        )
    surface_id = str(
        payload.get("surface_id")
        or f"{row['row_id']}--{optimizer_name}"
    )
    probe = _load_stable_probe(shard_dir, payload)
    rho_conv, rho_dense = safe_center(0)
    cell, rows = _run_canary_cell(
        study,
        spec,
        row,
        assets,
        probe,
        surface_id=surface_id,
        optimizer_name=optimizer_name,
        rho_conv=rho_conv,
        rho_dense=rho_dense,
        device=device,
    )
    result = {
        "schema_version": "mnist-conv-perfectdiode-preflight-canary/v1",
        "status": "complete" if cell["safety_clean"] else "preflight_failed",
        "preflight_passed": bool(cell["safety_clean"]),
        "canonical_stage": False,
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "entry_id": str(payload["entry_id"]),
        "surface_id": surface_id,
        "row_id": row["row_id"],
        "architecture": row["architecture"],
        "scheme": row["scheme"],
        "optimizer": optimizer_name,
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "official_test_read": False,
        "cell": cell,
        "benchmark": cell["benchmark"],
    }
    step_path = _write_step_log(
        output_dir / "step_log.csv",
        (
            {**step, "cell_id": cell["cell_id"], "phase": "preflight_center"}
            for step in rows
        ),
    )
    result["artifacts"] = [_artifact_record(step_path, base=output_dir)]
    atomic_write_json(output_dir / "result.json", result, canonical=True)
    return result


def _extension_cells(
    payload: Mapping[str, Any],
    *,
    study_id: str,
    surface_id: str,
) -> list[dict[str, Any]]:
    raw_cells = payload.get("cells")
    if raw_cells is None:
        raw_cells = [
            {
                "rho_conv": payload.get("rho_conv"),
                "rho_dense": payload.get("rho_dense"),
                "cell_id": payload.get("cell_id"),
            }
        ]
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError(
            "Expected extension payload.cells to be a non-empty list. "
            f"Provided value: {raw_cells!r}."
        )
    cells = []
    for index, cell in enumerate(raw_cells):
        if not isinstance(cell, Mapping):
            raise ValueError(
                f"Expected payload.cells[{index}] to be a mapping. "
                f"Provided value: {cell!r}."
            )
        rho_conv = _finite(
            cell.get("rho_conv"), f"payload.cells[{index}].rho_conv", positive=True
        )
        rho_dense = _finite(
            cell.get("rho_dense"),
            f"payload.cells[{index}].rho_dense",
            positive=True,
        )
        expected_id = rho_cell_id(
            study_id=study_id,
            surface_id=surface_id,
            rho_conv=rho_conv,
            rho_dense=rho_dense,
        )
        provided_id = cell.get("cell_id")
        if provided_id is not None and provided_id != expected_id:
            raise ValueError(
                f"Expected payload.cells[{index}].cell_id to match rho identity. "
                f"Provided value: {provided_id!r}."
            )
        cells.append(
            {
                "rho_conv": rho_conv,
                "rho_dense": rho_dense,
                "cell_id": expected_id,
            }
        )
    return cells


def _execute_extension_canary_entry(
    study: Mapping[str, Any],
    spec: PerfectDiodeHparamStudySpec,
    shard_dir: Path,
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    row, assets = _common_loaded_assets(
        study,
        spec,
        shard_dir,
        payload,
        data_root=data_root,
        download=download,
    )
    optimizer_name = str(payload["optimizer"]).lower()
    surface_id = str(
        payload.get("surface_id")
        or f"{row['row_id']}--{optimizer_name}"
    )
    probe = _load_stable_probe(shard_dir, payload)
    cells = []
    all_rows = []
    for contract in _extension_cells(
        payload, study_id=spec.study_id, surface_id=surface_id
    ):
        cell, rows = _run_canary_cell(
            study,
            spec,
            row,
            assets,
            probe,
            surface_id=surface_id,
            optimizer_name=optimizer_name,
            rho_conv=contract["rho_conv"],
            rho_dense=contract["rho_dense"],
            device=device,
            cell_id=contract["cell_id"],
        )
        cells.append(cell)
        all_rows.extend(
            {**step, "cell_id": cell["cell_id"], "phase": "extension"}
            for step in rows
        )
    result = {
        "schema_version": "mnist-conv-perfectdiode-extension-canary-result/v1",
        "status": "complete",
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "entry_id": str(payload["entry_id"]),
        "surface_id": surface_id,
        "row_id": row["row_id"],
        "architecture": row["architecture"],
        "scheme": row["scheme"],
        "optimizer": optimizer_name,
        "official_test_read": False,
        "cells": cells,
    }
    step_path = _write_step_log(output_dir / "step_log.csv", all_rows)
    result["artifacts"] = [_artifact_record(step_path, base=output_dir)]
    atomic_write_json(output_dir / "result.json", result, canonical=True)
    return result


def _training_run_spec(
    spec: PerfectDiodeHparamStudySpec,
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    optimizer_name: str,
    mode: str,
    rho_conv: float,
    rho_dense: float,
    rates: Mapping[str, float],
    assets: LoadedPerfectDiodeAssets,
) -> dict[str, Any]:
    contract = training_contract(
        architecture=str(row["architecture"]), mode=mode
    )
    return {
        "schema_version": PD_RUN_SCHEMA_VERSION,
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "entry_id": str(payload["entry_id"]),
        "cell_id": str(payload["cell_id"]),
        "surface_id": str(
            payload.get("surface_id")
            or f"{row['row_id']}--{optimizer_name}"
        ),
        "category": "ordinary_mnist_optimization_diagnostic",
        "row": copy.deepcopy(dict(row)),
        "optimizer": optimizer_name,
        "optimizer_contract": _optimizer_contract(spec.data, optimizer_name),
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "raw_learning_rates_by_parameter": dict(rates),
        "training": {
            "mode": mode,
            "epochs": contract.epochs,
            "steps_per_epoch": contract.steps_per_epoch,
            "total_steps": contract.total_steps,
            "constant_lr_vector": True,
            "restart_from_shared_initialization": True,
            "continue_from_canary_or_candidate": False,
        },
        "provenance": {
            "initialization_checkpoint_sha256": assets.initialization[
                "checkpoint_sha256"
            ],
            "initialization_tensor_sha256": assets.initialization[
                "parameter_tensor_sha256"
            ],
            "train_indices_sha256": assets.bundle.train_indices_hash,
            "validation_indices_sha256": assets.bundle.validation_indices_hash,
            "probe_entry_id": payload.get("probe_entry_id"),
            "source_commit": payload.get("source_commit"),
            "source_archive_sha256": payload.get("source_archive_sha256"),
            "environment_sha256": payload.get("environment_sha256"),
            "source_shard_environment_sha256": payload.get(
                "source_shard_environment_sha256",
                payload.get("environment_sha256"),
            ),
            "execution_environment_sha256": payload.get(
                "execution_environment_sha256",
                payload.get("environment_sha256"),
            ),
        },
        "official_test_read": False,
    }


def _prepare_training_entry(
    study: Mapping[str, Any],
    spec: PerfectDiodeHparamStudySpec,
    shard_dir: Path,
    payload: Mapping[str, Any],
    *,
    data_root: str | Path,
    download: bool,
) -> dict[str, Any]:
    """Resolve the production preamble from scientific fields.

    ``cell_id`` is derived provenance.  It is not an admission field because
    a continuation may carry an identifier produced under an older parent
    study while retaining the same row, optimizer, rho values, and learning
    rates.
    """

    row, assets = _common_loaded_assets(
        study,
        spec,
        shard_dir,
        payload,
        data_root=data_root,
        download=download,
    )
    optimizer_name = str(payload["optimizer"]).lower()
    surface_id = str(
        payload.get("surface_id")
        or f"{row['row_id']}--{optimizer_name}"
    )
    rho_conv = _finite(
        payload.get("rho_conv"), "payload.rho_conv", positive=True
    )
    rho_dense = _finite(
        payload.get("rho_dense"), "payload.rho_dense", positive=True
    )
    resolved_payload = copy.deepcopy(dict(payload))
    resolved_payload["cell_id"] = rho_cell_id(
        study_id=spec.study_id,
        surface_id=surface_id,
        rho_conv=rho_conv,
        rho_dense=rho_dense,
    )
    probe = _load_stable_probe(shard_dir, resolved_payload)
    rates = _resolve_rates(
        probe,
        resolved_payload,
        rho_conv=rho_conv,
        rho_dense=rho_dense,
    )
    return {
        "row": row,
        "assets": assets,
        "payload": resolved_payload,
        "optimizer_name": optimizer_name,
        "surface_id": surface_id,
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "rates": rates,
    }


def _execute_training_entry(
    study: Mapping[str, Any],
    spec: PerfectDiodeHparamStudySpec,
    shard_dir: Path,
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    data_root: str | Path,
    download: bool,
    device: str,
    mode: str,
) -> dict[str, Any]:
    prepared = _prepare_training_entry(
        study,
        spec,
        shard_dir,
        payload,
        data_root=data_root,
        download=download,
    )
    row = prepared["row"]
    assets = prepared["assets"]
    payload = prepared["payload"]
    optimizer_name = prepared["optimizer_name"]
    surface_id = prepared["surface_id"]
    rho_conv = prepared["rho_conv"]
    rho_dense = prepared["rho_dense"]
    rates = prepared["rates"]
    expected_cell_id = payload["cell_id"]
    runtime = _build_fresh_runtime(
        study,
        row,
        assets,
        optimizer_name=optimizer_name,
        rates=rates,
        device=device,
    )
    run_spec = _training_run_spec(
        spec,
        row,
        payload,
        optimizer_name=optimizer_name,
        mode=mode,
        rho_conv=rho_conv,
        rho_dense=rho_dense,
        rates=rates,
        assets=assets,
    )
    run_spec_path = output_dir / "run_spec.json"
    atomic_write_json(run_spec_path, run_spec, canonical=True)
    started, cuda = _runtime_benchmark_start(device)
    raw = run_epoch_training(
        runtime,
        assets.bundle,
        architecture=row["architecture"],
        mode=mode,
        learning_rates_by_parameter=rates,
        output_dir=output_dir,
    )
    benchmark = _runtime_benchmark_finish(
        started, cuda, int(raw["completed_steps"])
    )
    del runtime
    rows = raw.pop("step_rows")
    validations = raw.pop("validations")
    step_path = _write_step_log(output_dir / "step_log.csv", rows)
    validation_payload = {
        "schema_version": "mnist-conv-perfectdiode-validation-history/v1",
        "entry_id": str(payload["entry_id"]),
        "cell_id": expected_cell_id,
        "official_test_read": False,
        "validation_indices_sha256": assets.bundle.validation_indices_hash,
        "records": validations,
    }
    validation_path = output_dir / "validation.json"
    atomic_write_json(validation_path, validation_payload, canonical=True)
    raw.update(
        {
            "study_id": spec.study_id,
            "config_sha256": spec.config_sha256,
            "entry_id": str(payload["entry_id"]),
            "cell_id": expected_cell_id,
            "surface_id": surface_id,
            "row_id": row["row_id"],
            "scheme": row["scheme"],
            "optimizer": optimizer_name,
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "initialization_checkpoint_sha256": assets.initialization[
                "checkpoint_sha256"
            ],
            "initialization_tensor_sha256": assets.initialization[
                "parameter_tensor_sha256"
            ],
            "train_indices_sha256": assets.bundle.train_indices_hash,
            "validation_indices_sha256": assets.bundle.validation_indices_hash,
            "probe_entry_id": payload.get("probe_entry_id"),
            "benchmark": benchmark,
        }
    )
    artifacts = [
        run_spec_path,
        validation_path,
        step_path,
        output_dir / "best_validation.pt",
        output_dir / "final.pt",
    ]
    raw["artifacts"] = [
        _artifact_record(path, base=output_dir) for path in artifacts
    ]
    require_official_test_excluded(raw, require_result_marker=True)
    atomic_write_json(output_dir / "result.json", raw, canonical=True)
    return raw


def execute_perfectdiode_stage_entry(
    study: Mapping[str, Any] | PerfectDiodeHparamStudySpec,
    shard_dir: str | Path,
    stage: str,
    payload: Mapping[str, Any],
    *,
    data_root: str | Path,
    device: str,
    download: bool = False,
    entry_id: str | None = None,
) -> dict[str, Any]:
    """Execute one self-contained GPU stage entry and publish ``result.json``.

    Orchestration owns manifests, selection, finalization, and resume.  This
    dispatcher owns all numerical entries and writes only inside
    ``stages/<stage>/entries/<entry_id>``.
    """

    data, spec = _study_and_spec(study)
    if not isinstance(payload, Mapping):
        raise TypeError(
            "Expected payload to be a mapping. "
            f"Provided value: {type(payload).__name__}."
        )
    normalized_payload = dict(payload)
    payload_entry_id = normalized_payload.get("entry_id")
    if entry_id is not None:
        if payload_entry_id is not None and payload_entry_id != entry_id:
            raise ValueError(
                "Expected entry_id keyword to match payload.entry_id. "
                f"Provided value: keyword={entry_id!r}, "
                f"payload={payload_entry_id!r}."
            )
        normalized_payload["entry_id"] = entry_id
    entry_id = normalized_payload.get("entry_id")
    if not isinstance(entry_id, str) or not entry_id:
        raise ValueError(
            "Expected payload.entry_id to be a non-empty string. "
            f"Provided value: {entry_id!r}."
        )
    root = Path(shard_dir).expanduser().resolve()
    output = _entry_output_dir(root, stage, entry_id)
    handlers = {
        "fixed_tk_gradient_security": _execute_security_entry,
        "optimizer_probe": _execute_probe_entry,
        "rho_canary_core": _execute_core_canary_entry,
        "rho_canary_extension": _execute_extension_canary_entry,
        "preflight_canary": _execute_preflight_canary_entry,
    }
    if stage == "assets":
        architecture = str(normalized_payload.get("architecture") or entry_id)
        return prepare_architecture_assets(
            spec,
            output_dir=output,
            architecture=architecture,
            data_root=data_root,
            download=download,
            device=device,
        )
    if stage in {"rho_core_candidates", "rho_extension_candidates"}:
        return _execute_training_entry(
            data,
            spec,
            root,
            output,
            normalized_payload,
            data_root=data_root,
            download=download,
            device=device,
            mode="candidate",
        )
    if stage == "long_confirm":
        return _execute_training_entry(
            data,
            spec,
            root,
            output,
            normalized_payload,
            data_root=data_root,
            download=download,
            device=device,
            mode="long_confirmation",
        )
    handler = handlers.get(stage)
    if handler is None:
        raise ValueError(
            "Expected a numerical perfect-diode stage in "
            "{'assets','preflight_canary','fixed_tk_gradient_security','optimizer_probe',"
            "'rho_canary_core','rho_canary_extension','rho_core_candidates',"
            "'rho_extension_candidates','long_confirm'}. "
            f"Provided value: {stage!r}."
        )
    return handler(
        data,
        spec,
        root,
        output,
        normalized_payload,
        data_root=data_root,
        download=download,
        device=device,
    )


__all__ = [
    "CANDIDATE_EPOCHS",
    "CANDIDATE_TOTAL_STEPS",
    "CANARY_STEPS",
    "CONV1_CONFIRM_EPOCHS",
    "CONV1_CONFIRM_STEPS",
    "CONV2_CONFIRM_EPOCHS",
    "CONV2_CONFIRM_STEPS",
    "PerfectDiodeRuntimeError",
    "SafetyMonitor",
    "TrainingContract",
    "collect_runtime_gradients",
    "compare_fixed_tk_gradient_security",
    "derive_raw_learning_rates",
    "epoch_validation_checkpoint_plan",
    "execute_perfectdiode_stage_entry",
    "measure_adaptive_optimizer_probe",
    "prepare_architecture_assets",
    "require_official_test_excluded",
    "rho_cell_id",
    "run_canary",
    "run_epoch_training",
    "run_fixed_tk_gradient_security",
    "training_contract",
]
