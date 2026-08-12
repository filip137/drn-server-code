#!/usr/bin/env python3
"""Positive one-sided EqProp/BPTT diagnostic for matched dense controls.

The runner reconstructs FC1/FC2/FC3 initializations with the exact signed,
amplification-scaled-bias runtime used by the Conv checkpoint replay.  It reads
the same deterministic ordinary-MNIST validation cohort, never constructs an
optimizer, and scores only DenseWeight gradients.  Biases remain active in the
energy dynamics but their gradients are excluded.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/conv/perfectdiode_fc123_initialization_positive_eqprop_small_beta_seed0_20260810_v1.json"
)
CONV_ANALYZER = (
    REPOSITORY_ROOT
    / "experiments/analyze_conv_eqprop_bptt_beta_tk_displacement.py"
)


def _load_conv_analyzer() -> Any:
    """Load the shared exact-runtime and EqProp diagnostic helpers by path."""

    spec = importlib.util.spec_from_file_location(
        "_fc_control_conv_eqprop_helpers", CONV_ANALYZER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {CONV_ANALYZER}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


conv = _load_conv_analyzer()
base = conv.base
np = base.np
torch = base.torch
plt = base.plt


SCHEMA = "perfectdiode-fc-positive-eqprop-small-beta/v1"
CONFIG_SCHEMA = "perfectdiode-fc-positive-eqprop-small-beta-replay/v1"
EQPROP_VARIANT = "positive_one_sided"
CHECKPOINT_ROLE = "reconstructed_initialization"
FLOAT32_EPSILON = float(torch.finfo(torch.float32).eps)


def _stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _architecture_map(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [dict(value) for value in config["architecture_controls"]]
    output = {str(row["architecture"]): row for row in rows}
    if set(output) != {"fc1", "fc2", "fc3"} or len(output) != len(rows):
        raise ValueError("Expected exactly one FC1, FC2, and FC3 control.")
    return output


def _scheme_map(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [dict(value) for value in config["amplification_schemes"]]
    output = {str(row["scheme"]): row for row in rows}
    expected = {
        "baseline": (1.0, 1.0),
        "ours": (4.0, 1.0),
        "legacy": (4.0, 0.25),
    }
    observed = {
        name: (
            float(row["voltage_amplification"]),
            float(row["current_amplification"]),
        )
        for name, row in output.items()
    }
    if observed != expected:
        raise ValueError(f"Unexpected amplification contract: {observed}.")
    return output


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("Unexpected FC control config schema.")
    architectures = _architecture_map(config)
    schemes = _scheme_map(config)
    model = config["model_contract"]
    if list(model["input_shape"]) != [2, 28, 28]:
        raise ValueError("The FC control requires the signed 2x28x28 input.")
    if list(model["output_shape"]) != [20] or model["conv_pipeline"] != []:
        raise ValueError("The FC control requires paired 20 outputs and no Conv stage.")
    if model["non_linearity"] != "perfect_diode":
        raise ValueError("The FC control requires perfect diodes.")
    for key in (
        "quadratic_diode_param",
        "hard_sigmoid_param",
        "exponential_diode_param",
    ):
        if not isinstance(model.get(key), dict) or not model[key]:
            raise ValueError(f"The FC control requires an explicit {key} dictionary.")
    expected_hidden = {
        "fc1": [[64]],
        "fc2": [[64], [128]],
        "fc3": [[64], [128], [256]],
    }
    for architecture, row in architectures.items():
        if row["hidden_layer_shapes"] != expected_hidden[architecture]:
            raise ValueError(f"Unexpected hidden shapes for {architecture}.")
        if int(row.get("nudging_scale_power", -1)) != len(
            row["hidden_layer_shapes"]
        ):
            raise ValueError(
                f"{architecture} must explicitly declare its output bias-current row exponent."
            )
        if set(row["native_tk_by_scheme"]) != set(schemes):
            raise ValueError(f"Incomplete native T/K map for {architecture}.")
        for coordinate in [*row["native_tk_by_scheme"].values(), row["long_tk"]]:
            if len(coordinate) != 2 or min(int(value) for value in coordinate) <= 0:
                raise ValueError(f"Invalid T/K coordinate for {architecture}: {coordinate}.")
    gradient = config["gradient_contract"]
    if gradient["eqprop_variant_primary"] != EQPROP_VARIANT or gradient[
        "eqprop_variants_reported"
    ] != [EQPROP_VARIANT]:
        raise ValueError("The FC control reports only positive one-sided EqProp.")
    if (
        gradient.get("amplification_exponent_convention")
        != "output_bias_current_row"
    ):
        raise ValueError(
            "FC beta normalization must use the output bias-current row exponent."
        )
    if str(gradient.get("nudging_mode", "cost")) not in {"cost", "current"}:
        raise ValueError("FC nudging_mode must be 'cost' or 'current'.")
    if str(
        gradient.get("normalized_beta_parameterization", "common_effective")
    ) not in {"common_effective", "common_base"}:
        raise ValueError("Unsupported FC normalized beta parameterization.")
    normalized = [float(value) for value in gradient["normalized_betas_requested"]]
    if (
        not normalized
        or normalized != sorted(normalized)
        or len(set(normalized)) != len(normalized)
        or any(not math.isfinite(value) or value <= 0.0 for value in normalized)
    ):
        raise ValueError("Normalized beta requests must be unique positive values.")


def _source_config(
    config: Mapping[str, Any], *, architecture: str, scheme: str
) -> dict[str, Any]:
    """Build the minimal exact-runtime config for one synthetic FC case."""

    architecture_row = _architecture_map(config)[architecture]
    scheme_row = _scheme_map(config)[scheme]
    model = config["model_contract"]
    dataset = config["dataset"]
    native_t, native_k = [
        int(value) for value in architecture_row["native_tk_by_scheme"][scheme]
    ]
    hidden = [list(shape) for shape in architecture_row["hidden_layer_shapes"]]
    layer_shapes = [list(model["input_shape"]), *hidden, list(model["output_shape"])]
    minimizer = json.loads(json.dumps(model["energy_minimizer"]))
    mode = str(minimizer.pop("mode"))
    source_dataset = dataset["source_config"]
    return {
        "schema_version": "synthetic-fc-initialization-replay-source/v1",
        "study_id": str(config["study_id"]),
        "arm_id": f"{architecture}_{scheme}_initialization_control",
        "training_algorithm": "BP",
        "batch_state_policy": "reset_each_batch",
        "seed": int(model["seed"]),
        "input_mode": str(model["input_mode"]),
        "lab": {
            "model_key": "mnist_bp_fc_control",
            "dataset_key": "mnist",
            "epochs": 0,
        },
        "datasets": {
            "mnist": {
                "factory": str(source_dataset["factory"]),
                "params": {
                    "name": "mnist",
                    "batch_size": int(dataset["batch_size"]),
                    "validation_batch_size": int(dataset["batch_size"]),
                    "root": str(dataset["dataset_root"]),
                    "train": True,
                    "download": False,
                    "normalize": bool(source_dataset["normalize"]),
                    "normalize_mean": float(source_dataset["normalize_mean"]),
                    "normalize_std": float(source_dataset["normalize_std"]),
                    "normalize_scale": float(source_dataset["normalize_scale"]),
                    "shuffle_seed": int(source_dataset["shuffle_seed"]),
                    "split_seed": int(source_dataset["split_seed"]),
                    "num_workers": int(source_dataset["num_workers"]),
                    "pin_memory": bool(source_dataset["pin_memory"]),
                    "affine_config": None,
                },
            }
        },
        "energy_minimizer": {"mode": mode},
        "model_base": {
            "weight_min": float(model["weight_min"]),
            "weight_max": float(model["weight_max"]),
            "weight_init_mode": str(model["weight_init_mode"]),
            "input_gain": float(architecture_row["input_gain"]),
            "voltage_amp": float(scheme_row["voltage_amplification"]),
            "current_amp": float(scheme_row["current_amplification"]),
            "non_linearity": str(model["non_linearity"]),
            "quadratic_diode_param": dict(model["quadratic_diode_param"]),
            "hard_sigmoid_param": dict(model["hard_sigmoid_param"]),
            "exponential_diode_param": dict(model["exponential_diode_param"]),
            "num_iterations_inference": native_t,
            "num_iterations_training": native_k,
            "trainable_amplification": bool(model["trainable_amplification"]),
            "amplification_min": float(model["amplification_min"]),
            "amplification_max": model["amplification_max"],
            "minimizer": minimizer,
        },
        "model_overrides": {
            "mnist_bp_fc_control": {
                "layer_shapes": layer_shapes,
                "conv_pipeline": [],
                "weight_gains": [1.0] * (len(layer_shapes) - 1),
            }
        },
    }


def _tk_coordinates(
    config: Mapping[str, Any], *, architecture: str, scheme: str
) -> list[tuple[int, int, str]]:
    row = _architecture_map(config)[architecture]
    native = tuple(int(value) for value in row["native_tk_by_scheme"][scheme])
    long = tuple(int(value) for value in row["long_tk"])
    values: list[tuple[int, int, str]] = [(*native, "native")]
    if long != native:
        values.append((*long, "long_64"))
    return values


def _output_curvature_scale(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Measure b0=2*median(a_out) directly on the constructed energy."""

    output_layer = runtime["free_layers"][-1]
    coefficients = (
        runtime["energy_fn"]
        .a_coef_fn(output_layer)()
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
    base._finite(coefficients, "FC output energy quadratic coefficient")
    if coefficients.numel() == 0 or bool((coefficients <= 0.0).any()):
        raise ValueError("FC output energy quadratic coefficients must be positive.")
    digest = hashlib.sha256()
    digest.update(str(tuple(coefficients.shape)).encode("ascii"))
    digest.update(coefficients.numpy().tobytes())
    median = float(torch.quantile(coefficients.flatten(), 0.5))
    return {
        "a_out_element_count": int(coefficients.numel()),
        "a_out_minimum": float(coefficients.min()),
        "a_out_median": median,
        "a_out_maximum": float(coefficients.max()),
        "b0_scheme_init": 2.0 * median,
        "a_out_sha256": digest.hexdigest(),
        "median_implementation": "torch.quantile(float64 flattened coefficient, 0.5; standard even-sample median)",
    }


def _derive_beta_points(
    *,
    normalized_requests: Sequence[float],
    b0_baseline_init: float,
    b0_scheme_init: float,
    voltage_amplification: float,
    current_amplification: float,
    amplification_exponent: int,
    actual_beta_cap: float,
    normalized_beta_parameterization: str = "common_effective",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map requested normalized nudges to deduplicated physical beta values."""

    b0_baseline_init = float(b0_baseline_init)
    b0_scheme_init = float(b0_scheme_init)
    voltage_amplification = float(voltage_amplification)
    current_amplification = float(current_amplification)
    amplification_exponent = int(amplification_exponent)
    cap = float(actual_beta_cap)
    if min(
        b0_baseline_init,
        b0_scheme_init,
        voltage_amplification,
        current_amplification,
        cap,
        amplification_exponent,
    ) <= 0.0:
        raise ValueError("Beta normalization inputs must be positive.")
    effective_multiplier = (
        voltage_amplification / current_amplification
    ) ** amplification_exponent
    parameterization = str(normalized_beta_parameterization)
    if parameterization not in {"common_effective", "common_base"}:
        raise ValueError(
            f"Unsupported normalized beta parameterization {parameterization!r}."
        )
    physical_multiplier = (
        1.0 / effective_multiplier
        if parameterization == "common_effective"
        else 1.0
    )
    all_rows: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    actual_to_request: dict[float, int] = {}
    for request_index, requested in enumerate(normalized_requests):
        beta_hat_requested = float(requested)
        uncapped = beta_hat_requested * b0_baseline_init * physical_multiplier
        if parameterization == "common_base":
            beta = min(uncapped, cap / effective_multiplier)
            beta_was_capped = bool(uncapped * effective_multiplier > cap)
            cap_semantics = "injected_beta"
        else:
            beta = min(uncapped, cap)
            beta_was_capped = bool(uncapped > cap)
            cap_semantics = "actual_beta"
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("Derived physical beta must be finite and positive.")
        retained_request = actual_to_request.get(beta)
        is_retained = retained_request is None
        if is_retained:
            actual_to_request[beta] = request_index
        effective_beta = beta * effective_multiplier
        row = {
            "request_index": request_index,
            "beta_hat_requested": beta_hat_requested,
            "actual_beta_uncapped": uncapped,
            "beta": beta,
            "actual_beta_cap": cap,
            "beta_was_capped": beta_was_capped,
            "retained_after_actual_beta_deduplication": is_retained,
            "deduplicated_to_request_index": (
                request_index if is_retained else int(retained_request)
            ),
            "amplification_depth_L": amplification_exponent,
            "amplification_exponent_convention": "output_bias_current_row",
            "nudging_effective_multiplier": effective_multiplier,
            "nudging_physical_multiplier": physical_multiplier,
            "effective_beta": effective_beta,
            "beta_hat_effective": effective_beta / b0_baseline_init,
            "beta_hat_base": beta / b0_baseline_init,
            "normalized_beta_parameterization": parameterization,
            "cap_semantics": cap_semantics,
            "b0_baseline_init": b0_baseline_init,
            "b0_scheme_init": b0_scheme_init,
            "actual_beta_over_b0_scheme_init": beta / b0_scheme_init,
        }
        all_rows.append(row)
        if is_retained:
            retained.append(row)
    return all_rows, retained


class _MeanAccumulator:
    def __init__(self) -> None:
        self.sums: dict[tuple[Any, ...], torch.Tensor] = {}
        self.examples: defaultdict[tuple[Any, ...], int] = defaultdict(int)

    def add(self, key: tuple[Any, ...], value: torch.Tensor, count: int) -> None:
        weighted = value.detach().cpu().to(torch.float64) * int(count)
        if key in self.sums:
            self.sums[key] += weighted
        else:
            self.sums[key] = weighted.clone()
        self.examples[key] += int(count)

    def mean(self, key: tuple[Any, ...]) -> torch.Tensor:
        if key not in self.sums or self.examples[key] <= 0:
            raise KeyError(key)
        return self.sums[key] / self.examples[key]


def _precision_metrics(
    zero_gradient: torch.Tensor,
    positive_gradient: torch.Tensor,
    *,
    minimum_snr: float,
) -> dict[str, Any]:
    zero = zero_gradient.detach().cpu().to(torch.float64)
    positive = positive_gradient.detach().cpu().to(torch.float64)
    numerator = positive - zero
    numerator_l2 = float(torch.linalg.vector_norm(numerator))
    zero_l2 = float(torch.linalg.vector_norm(zero))
    positive_l2 = float(torch.linalg.vector_norm(positive))
    denominator = FLOAT32_EPSILON * (positive_l2 + zero_l2)
    if denominator > 0.0:
        snr = numerator_l2 / denominator
    elif numerator_l2 > 0.0:
        snr = math.inf
    else:
        snr = 0.0
    finite = bool(math.isfinite(numerator_l2) and math.isfinite(denominator))
    passed = bool(
        finite
        and numerator_l2 > 0.0
        and math.isfinite(snr)
        and snr >= float(minimum_snr)
    )
    return {
        "finite_difference_numerator_l2": numerator_l2,
        "zero_energy_gradient_l2": zero_l2,
        "positive_energy_gradient_l2": positive_l2,
        "float32_roundoff_denominator": denominator,
        "precision_snr": snr if math.isfinite(snr) else None,
        "precision_snr_minimum": float(minimum_snr),
        "precision_gate_passed": passed,
        "finite_difference_exact_zero_fraction": float(
            (numerator == 0.0).to(torch.float64).mean()
        ),
        "finite_difference_numerator_sha256": base._tensor_sha256(numerator),
    }


def _adjacent_metrics(
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    cosine_minimum: float,
    symmetric_norm_delta_maximum: float,
) -> dict[str, Any]:
    lower64 = lower.detach().cpu().to(torch.float64)
    upper64 = upper.detach().cpu().to(torch.float64)
    metrics = base._vector_metrics(lower64, upper64, zero_epsilon=1.0e-30)
    lower_norm = float(torch.linalg.vector_norm(lower64))
    upper_norm = float(torch.linalg.vector_norm(upper64))
    norm_sum = lower_norm + upper_norm
    symmetric_delta = (
        2.0 * abs(upper_norm - lower_norm) / norm_sum
        if norm_sum > 0.0
        else None
    )
    passed = bool(
        metrics["cosine"] is not None
        and float(metrics["cosine"]) >= float(cosine_minimum)
        and symmetric_delta is not None
        and symmetric_delta <= float(symmetric_norm_delta_maximum)
    )
    return {
        "adjacent_eqprop_cosine": metrics["cosine"],
        "adjacent_symmetric_norm_delta": symmetric_delta,
        "adjacent_cosine_minimum": float(cosine_minimum),
        "adjacent_symmetric_norm_delta_maximum": float(
            symmetric_norm_delta_maximum
        ),
        "adjacent_consistency_passed": passed,
    }


def _parameter_rows_and_selection(
    accumulator: _MeanAccumulator,
    *,
    context: Mapping[str, Any],
    points: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    types: Mapping[str, str],
    expected_examples: int,
    zero_epsilon: float,
    residual_gates: Mapping[float, Mapping[str, Any]],
    minimum_snr: float,
    sentinel_max_beta_hat: float,
    adjacent_cosine_minimum: float,
    adjacent_norm_delta_maximum: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_lookup: dict[tuple[float, str], dict[str, Any]] = {}
    means: dict[tuple[float, str], torch.Tensor] = {}
    for point in points:
        beta = float(point["beta"])
        residual_gate = residual_gates[beta]
        for name in names:
            bptt_key = ("bptt", name)
            eqprop_key = ("eqprop", beta, name)
            zero_key = ("zero", name)
            plus_key = ("positive", beta, name)
            observed = int(accumulator.examples.get(eqprop_key, 0))
            coverage_complete = bool(
                observed == int(expected_examples)
                and accumulator.examples.get(bptt_key, 0) == int(expected_examples)
                and accumulator.examples.get(zero_key, 0) == int(expected_examples)
                and accumulator.examples.get(plus_key, 0) == int(expected_examples)
            )
            if coverage_complete:
                bptt = accumulator.mean(bptt_key)
                eqprop = accumulator.mean(eqprop_key)
                zero = accumulator.mean(zero_key)
                positive = accumulator.mean(plus_key)
                gradient_metrics = base._vector_metrics(
                    bptt, eqprop, zero_epsilon=zero_epsilon
                )
                precision = _precision_metrics(
                    zero, positive, minimum_snr=minimum_snr
                )
                means[(beta, name)] = eqprop
            else:
                gradient_metrics = {
                    "element_count": None,
                    "dot_product": None,
                    "bptt_l2": None,
                    "eqprop_l2": None,
                    "cosine": None,
                    "relative_l2_error": None,
                    "norm_ratio_eqprop_over_bptt": None,
                    "bptt_rms": None,
                    "eqprop_rms": None,
                    "bptt_zero_fraction": None,
                    "eqprop_zero_fraction": None,
                    "bptt_exact_zero_fraction": None,
                    "eqprop_exact_zero_fraction": None,
                }
                precision = {
                    "finite_difference_numerator_l2": None,
                    "zero_energy_gradient_l2": None,
                    "positive_energy_gradient_l2": None,
                    "float32_roundoff_denominator": None,
                    "precision_snr": None,
                    "precision_snr_minimum": float(minimum_snr),
                    "precision_gate_passed": False,
                    "finite_difference_exact_zero_fraction": None,
                    "finite_difference_numerator_sha256": None,
                }
            base_selectable = bool(
                coverage_complete
                and gradient_metrics["cosine"] is not None
                and residual_gate["passed"]
                and precision["precision_gate_passed"]
            )
            row = {
                "schema": SCHEMA,
                **context,
                **point,
                "eqprop_variant": EQPROP_VARIANT,
                "parameter_name": name,
                "parameter_type": types[name],
                "bias_excluded": True,
                "example_count": observed,
                "expected_example_count": int(expected_examples),
                "coverage_complete": coverage_complete,
                "residual_gate_passed": bool(residual_gate["passed"]),
                "residual_failed_layer_phases": "|".join(
                    residual_gate["failed_layer_phases"]
                ),
                "low_beta_float32_cancellation_sentinel": bool(
                    float(point["beta_hat_requested"])
                    <= float(sentinel_max_beta_hat)
                ),
                "base_selectable": base_selectable,
                "adjacent_lower_beta": None,
                "adjacent_lower_beta_hat_requested": None,
                "adjacent_pair_consecutive_uncapped_requests": None,
                "adjacent_eqprop_cosine": None,
                "adjacent_symmetric_norm_delta": None,
                "adjacent_cosine_minimum": float(adjacent_cosine_minimum),
                "adjacent_symmetric_norm_delta_maximum": float(
                    adjacent_norm_delta_maximum
                ),
                "adjacent_consistency_passed": None,
                "selected_by_adjacent_convergence_rule": False,
                "oracle_selected_by_bptt_cosine": False,
                "outcome": (
                    "incomplete_coverage"
                    if not coverage_complete
                    else (
                        "residual_gate_failed"
                        if not residual_gate["passed"]
                        else (
                            "precision_gate_failed"
                            if not precision["precision_gate_passed"]
                            else (
                                "ok"
                                if gradient_metrics["cosine"] is not None
                                else "dead_gradient"
                            )
                        )
                    )
                ),
                **gradient_metrics,
                **precision,
            }
            rows.append(row)
            row_lookup[(beta, name)] = row

    pair_candidates: list[dict[str, Any]] = []
    sorted_points = sorted(points, key=lambda row: int(row["request_index"]))
    for lower_point, upper_point in zip(sorted_points[:-1], sorted_points[1:]):
        lower_beta = float(lower_point["beta"])
        upper_beta = float(upper_point["beta"])
        consecutive_uncapped_requests = bool(
            not bool(lower_point["beta_was_capped"])
            and not bool(upper_point["beta_was_capped"])
            and int(upper_point["request_index"])
            == int(lower_point["request_index"]) + 1
        )
        layer_passes: list[bool] = []
        for name in names:
            upper_row = row_lookup[(upper_beta, name)]
            lower_row = row_lookup[(lower_beta, name)]
            upper_row["adjacent_lower_beta"] = lower_beta
            upper_row["adjacent_lower_beta_hat_requested"] = float(
                lower_point["beta_hat_requested"]
            )
            upper_row["adjacent_pair_consecutive_uncapped_requests"] = (
                consecutive_uncapped_requests
            )
            if (lower_beta, name) in means and (upper_beta, name) in means:
                adjacent = _adjacent_metrics(
                    means[(lower_beta, name)],
                    means[(upper_beta, name)],
                    cosine_minimum=adjacent_cosine_minimum,
                    symmetric_norm_delta_maximum=adjacent_norm_delta_maximum,
                )
            else:
                adjacent = {
                    "adjacent_eqprop_cosine": None,
                    "adjacent_symmetric_norm_delta": None,
                    "adjacent_cosine_minimum": float(adjacent_cosine_minimum),
                    "adjacent_symmetric_norm_delta_maximum": float(
                        adjacent_norm_delta_maximum
                    ),
                    "adjacent_consistency_passed": False,
                }
            upper_row.update(adjacent)
            layer_passes.append(
                bool(
                    lower_row["base_selectable"]
                    and upper_row["base_selectable"]
                    and adjacent["adjacent_consistency_passed"]
                    and consecutive_uncapped_requests
                )
            )
        pair_candidates.append(
            {
                "lower_point": lower_point,
                "upper_point": upper_point,
                "consecutive_uncapped_requests": consecutive_uncapped_requests,
                "all_layers_passed": bool(
                    consecutive_uncapped_requests
                    and layer_passes
                    and all(layer_passes)
                ),
            }
        )
    selected_pair = next(
        (candidate for candidate in pair_candidates if candidate["all_layers_passed"]),
        None,
    )
    selected_point = None if selected_pair is None else selected_pair["upper_point"]
    if selected_point is not None:
        selected_beta = float(selected_point["beta"])
        for name in names:
            row_lookup[(selected_beta, name)][
                "selected_by_adjacent_convergence_rule"
            ] = True

    oracle_candidates: list[dict[str, Any]] = []
    for point in points:
        beta = float(point["beta"])
        layer_rows = [row_lookup[(beta, name)] for name in names]
        if all(bool(row["base_selectable"]) for row in layer_rows):
            cosines = [float(row["cosine"]) for row in layer_rows]
            oracle_candidates.append(
                {
                    "point": point,
                    "worst": min(cosines),
                    "mean": float(np.mean(cosines)),
                    "worst_layer": min(
                        layer_rows, key=lambda row: float(row["cosine"])
                    )["parameter_name"],
                }
            )
    oracle = (
        sorted(
            oracle_candidates,
            key=lambda value: (
                -float(value["worst"]),
                int(value["point"]["request_index"]),
            ),
        )[0]
        if oracle_candidates
        else None
    )
    if oracle is not None:
        oracle_beta = float(oracle["point"]["beta"])
        for name in names:
            row_lookup[(oracle_beta, name)]["oracle_selected_by_bptt_cosine"] = True

    context_row = {
        "schema": SCHEMA,
        **context,
        "eqprop_variant": EQPROP_VARIANT,
        "tested_beta_count": len(points),
        "adjacent_pair_count": len(pair_candidates),
        "passing_adjacent_pair_count": sum(
            bool(value["all_layers_passed"]) for value in pair_candidates
        ),
        "selected_beta": None if selected_point is None else selected_point["beta"],
        "selected_beta_hat_requested": (
            None if selected_point is None else selected_point["beta_hat_requested"]
        ),
        "selected_beta_hat_effective": (
            None if selected_point is None else selected_point["beta_hat_effective"]
        ),
        "selection_rule": "larger_beta_of_lowest_all_layer_passing_adjacent_pair",
        "selection_independent_of_bptt_cosine": True,
        "selection_outcome": (
            "selected" if selected_point is not None else "no_converged_adjacent_pair"
        ),
        "oracle_beta": None if oracle is None else oracle["point"]["beta"],
        "oracle_beta_hat_requested": (
            None if oracle is None else oracle["point"]["beta_hat_requested"]
        ),
        "oracle_worst_layer_bptt_cosine": (
            None if oracle is None else oracle["worst"]
        ),
        "oracle_mean_layer_bptt_cosine": (
            None if oracle is None else oracle["mean"]
        ),
        "oracle_worst_layer": None if oracle is None else oracle["worst_layer"],
        "oracle_is_descriptive_not_selection": True,
    }
    return rows, context_row


def _plot_layerwise_cosines(
    parameter_rows: Sequence[Mapping[str, Any]], *, architecture: str, path: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for row_index, coordinate_kind in enumerate(("native", "long_64")):
        for column_index, scheme in enumerate(("baseline", "ours", "legacy")):
            axis = axes[row_index, column_index]
            panel = [
                row
                for row in parameter_rows
                if str(row["architecture"]) == architecture
                and str(row["scheme"]) == scheme
                and str(row["coordinate_kind"]) == coordinate_kind
            ]
            for name in dict.fromkeys(str(row["parameter_name"]) for row in panel):
                values = sorted(
                    [row for row in panel if str(row["parameter_name"]) == name],
                    key=lambda row: int(row["request_index"]),
                )
                axis.plot(
                    [float(row["beta_hat_requested"]) for row in values],
                    [
                        np.nan if row["cosine"] is None else float(row["cosine"])
                        for row in values
                    ],
                    marker="o",
                    linewidth=1.4,
                    markersize=3.5,
                    label=name.replace("DenseWeight_", "dense "),
                )
            selected = next(
                (
                    row
                    for row in panel
                    if bool(row["selected_by_adjacent_convergence_rule"])
                ),
                None,
            )
            if selected is not None:
                axis.axvline(
                    float(selected["beta_hat_requested"]),
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    label="selected",
                )
            axis.axhline(0.0, color="0.6", linewidth=0.8)
            axis.axhline(0.9, color="0.75", linewidth=0.8, linestyle=":")
            axis.set_xscale("log")
            axis.set_ylim(-1.05, 1.05)
            axis.grid(alpha=0.25)
            axis.set_title(f"{scheme} | {coordinate_kind}")
            if column_index == 0:
                axis.set_ylabel("EqProp--BPTT cosine")
            if row_index == 1:
                axis.set_xlabel("requested normalized beta")
            axis.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{architecture.upper()} positive one-sided EqProp by dense layer")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_displacements(
    state_rows: Sequence[Mapping[str, Any]], *, architecture: str, path: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for row_index, reference_kind in enumerate(("post_T_free", "matched_zero_K")):
        for column_index, scheme in enumerate(("baseline", "ours", "legacy")):
            axis = axes[row_index, column_index]
            for coordinate_kind, marker in (("native", "o"), ("long_64", "s")):
                panel = sorted(
                    [
                        row
                        for row in state_rows
                        if str(row["architecture"]) == architecture
                        and str(row["scheme"]) == scheme
                        and str(row["coordinate_kind"]) == coordinate_kind
                        and str(row["reference_kind"]) == reference_kind
                        and str(row["state_layer_name"]) == "__all__"
                        and str(row["phase"]) == "positive"
                    ],
                    key=lambda row: int(row["request_index"]),
                )
                axis.plot(
                    [float(row["beta_hat_requested"]) for row in panel],
                    [
                        np.nan
                        if row["pooled_relative_displacement"] is None
                        or float(row["pooled_relative_displacement"]) <= 0.0
                        else float(row["pooled_relative_displacement"])
                        for row in panel
                    ],
                    marker=marker,
                    linewidth=1.4,
                    markersize=3.5,
                    label=coordinate_kind,
                )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(alpha=0.25, which="both")
            axis.set_title(f"{scheme} | {reference_kind}")
            if column_index == 0:
                axis.set_ylabel("pooled relative displacement")
            if row_index == 1:
                axis.set_xlabel("requested normalized beta")
            axis.legend(fontsize=8)
    fig.suptitle(f"{architecture.upper()} free-to-positive-nudged displacement")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    config: Mapping[str, Any],
    context_rows: Sequence[Mapping[str, Any]],
    cohort: Mapping[str, Any],
    smoke: bool,
) -> None:
    table = []
    for row in context_rows:
        table.append(
            (
                row["architecture"],
                row["scheme"],
                row["coordinate_kind"],
                f"{row['T']}/{row['K']}",
                "--"
                if row["selected_beta_hat_requested"] is None
                else f"{float(row['selected_beta_hat_requested']):.3g}",
                "--"
                if row["oracle_worst_layer_bptt_cosine"] is None
                else f"{float(row['oracle_worst_layer_bptt_cosine']):.6f}",
            )
        )
    lines = [
        "# Dense initialization control: positive one-sided EqProp",
        "",
        f"Evidence class: `{config['evidence_class']}`; smoke: `{str(smoke).lower()}`.",
        "",
        "This is an ordinary-MNIST initialization diagnostic, not paper-facing accuracy evidence. It uses paired 20 outputs and the exact signed, amplification-scaled-bias runtime. Biases are active in every state but only DenseWeight gradients are scored. No optimizer is constructed and the official test split is not read.",
        "",
        "The selected beta is the larger member of the lowest adjacent pair for which every dense layer passes the projected-residual gate, float32 finite-difference SNR gate, and adjacent-estimate consistency gate. Selection never consults BPTT cosine. The BPTT-cosine oracle is reported separately as a descriptive diagnostic.",
        "",
        base._markdown_table(
            (
                "Architecture",
                "Scheme",
                "Coordinate",
                "T/K",
                "Selected beta-hat request",
                "Oracle worst-layer cosine",
            ),
            table,
        ),
        "",
        f"Fixed cohort: `{cohort['example_count']}` examples in `{cohort['batch_count']}` batches; cohort SHA `{cohort['cohort_sha256']}`.",
        "",
        "Artifacts include `parameter_summary.csv`, `context_beta_selection.csv`, `curvature_scales.csv`, `beta_mapping.csv`, exact per-layer equilibrium residual summaries and per-sample archives, phase diagnostics, and aggregate plus per-state-layer displacement tables.",
        "",
        "Interpretation limit: these dense networks are reconstructed initializations, because no protocol-matched trained FC checkpoints exist. They test whether convolution is necessary for the observed disagreement at initialization; they do not answer how a matched FC model behaves after training.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolved_run_dir(
    output_root: Path, *, smoke: bool, run_id: str | None
) -> tuple[str, Path]:
    resolved = run_id or (
        "fc-smoke-attempt-01" if smoke else "fc-initialization-control"
    )
    if not resolved or "/" in resolved or resolved in {".", ".."}:
        raise ValueError(f"Invalid run id {resolved!r}.")
    return resolved, output_root.expanduser().resolve() / resolved


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    config = base._read_json(config_path)
    _validate_config(config)
    if args.runtime_source_root.expanduser().resolve() != base.RUNTIME_SOURCE_ROOT:
        raise ValueError(
            "--runtime-source-root must be supplied at process bootstrap and match the exact source contract."
        )
    runtime_source = base._validate_runtime_source(config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    architectures = _architecture_map(config)
    schemes = _scheme_map(config)
    nudging_mode = str(config["gradient_contract"].get("nudging_mode", "cost"))
    source_configs = {
        (architecture, scheme): _source_config(
            config, architecture=architecture, scheme=scheme
        )
        for architecture in architectures
        for scheme in schemes
    }
    dataset_cfg = config["dataset"]
    batch_size = int(dataset_cfg["batch_size"])
    requested_examples = batch_size if args.smoke else int(dataset_cfg["cohort_examples"])
    batches, cohort = base._build_validation_cohort(
        source_configs[("fc1", "baseline")],
        data_root=args.dataset_root.expanduser().resolve(),
        batch_size=batch_size,
        example_count=requested_examples,
    )
    if cohort["validation_indices_sha256"] != dataset_cfg["validation_indices_sha256"]:
        raise ValueError("Materialized validation split differs from the matched Conv split.")
    observed_guards = [
        {
            "batch_index": int(batch["batch_index"]),
            "source_indices_sha256": batch["source_indices_sha256"],
            "payload_sha256": batch["payload_sha256"],
        }
        for batch in batches
    ]
    if observed_guards != dataset_cfg["expected_batch_guards"][: len(batches)]:
        raise ValueError("Fixed FC cohort batch guards differ from the matched Conv prefix.")
    first_signature = base._dataset_signature(
        source_configs[("fc1", "baseline")],
        data_root=args.dataset_root.expanduser().resolve(),
        batch_size=batch_size,
    )
    for source_config in source_configs.values():
        if base._dataset_signature(
            source_config,
            data_root=args.dataset_root.expanduser().resolve(),
            batch_size=batch_size,
        ) != first_signature:
            raise ValueError("Synthetic FC cases do not share one dataset contract.")

    output_root = args.output_root.expanduser().resolve()
    run_id, run_dir = _resolved_run_dir(
        output_root, smoke=bool(args.smoke), run_id=args.run_id
    )
    total_contexts = sum(
        len(_tk_coordinates(config, architecture=architecture, scheme=scheme))
        for architecture in architectures
        for scheme in schemes
    )
    manifest = {
        "study_id": str(config["study_id"]),
        "run_id": run_id,
        "arm_id": str(config["arm_id"]),
        "evidence_class": str(config["evidence_class"]),
        "dataset": str(dataset_cfg["name"]),
        "smoke": bool(args.smoke),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "configuration": {
            "path": str(config_path),
            "sha256": base.sha256_file(config_path),
            "resolved": config,
        },
        "runtime": {
            **base.runtime_context(target=args.target),
            "device": str(device),
            "hostname": socket.gethostname(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "exact_runtime_source": runtime_source,
        },
        "replay": {
            "checkpoint_role": CHECKPOINT_ROLE,
            "source_checkpoints_read": 0,
            "synthetic_initialization_cases": len(source_configs),
            "tk_contexts": total_contexts,
            "cohort_examples": requested_examples,
            "batch_size": batch_size,
            "official_test_read": False,
            "optimizer_constructed": False,
            "optimizer_steps_applied": False,
            "biases_active_in_dynamics": True,
            "bias_gradients_excluded": True,
            "nudging_mode": nudging_mode,
            "current_scale": "auto" if nudging_mode == "current" else 1.0,
            "finite_difference_denominator": (
                "beta_actual * output_row_current_scale"
                if nudging_mode == "current"
                else "beta_actual"
            ),
        },
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.sha256_file(Path(__file__).resolve()),
            "shared_conv_analyzer_path": str(CONV_ANALYZER),
            "shared_conv_analyzer_sha256": base.sha256_file(CONV_ANALYZER),
            "repository_root": str(REPOSITORY_ROOT),
            "repository_head": base._git_output(REPOSITORY_ROOT, "rev-parse", "HEAD"),
        },
    }
    base.start_run(run_dir, manifest)
    base._write_json(run_dir / "config.resolved.json", config)
    base._write_json(run_dir / "cohort.json", cohort)
    base._write_json(
        run_dir / "synthetic_source_configs.json",
        {
            f"{architecture}/{scheme}": {
                "sha256": _stable_json_sha256(source_config),
                "config": source_config,
            }
            for (architecture, scheme), source_config in source_configs.items()
        },
    )

    try:
        curvature_rows: list[dict[str, Any]] = []
        initialization_hashes: dict[tuple[str, str], str] = {}
        curvature_by_case: dict[tuple[str, str], float] = {}
        expected_initialization = config[
            "expected_reconstructed_initialization_guards"
        ]
        for architecture in architectures:
            for scheme in schemes:
                source_config = source_configs[(architecture, scheme)]
                native_k = int(
                    architectures[architecture]["native_tk_by_scheme"][scheme][1]
                )
                runtime = base._build_runtime(
                    source_config,
                    device=device,
                    gradient_iterations=native_k,
                    nudging_mode=nudging_mode,
                    current_scale="auto",
                )
                if any(
                    runtime["parameters"][index].__class__.__name__ != "DenseWeight"
                    for index in runtime["weight_indices"]
                ):
                    raise RuntimeError("A non-dense parameter entered the FC score set.")
                if any(
                    parameter.__class__.__name__ == "Bias"
                    for parameter in (
                        runtime["parameters"][index]
                        for index in runtime["weight_indices"]
                    )
                ):
                    raise RuntimeError("Bias leaked into the FC weight score set.")
                output_layer = runtime["cost_fn"].layers()[-1]
                output_bias_exponent = max(
                    conv.exact_runtime_layer_index(output_layer) - 1, 0
                )
                expected_output_bias_exponent = int(
                    architectures[architecture]["nudging_scale_power"]
                )
                if output_bias_exponent != expected_output_bias_exponent:
                    raise RuntimeError(
                        "FC output bias-current exponent mismatch for "
                        f"{architecture}/{scheme}: {output_bias_exponent} != "
                        f"{expected_output_bias_exponent}."
                    )
                exact_row_factor = float(
                    conv._amplified_layer_row_scale(
                        runtime["energy_fn"], output_layer
                    )
                )
                declared_row_factor = (
                    float(schemes[scheme]["voltage_amplification"])
                    / float(schemes[scheme]["current_amplification"])
                ) ** expected_output_bias_exponent
                if not math.isclose(
                    exact_row_factor,
                    declared_row_factor,
                    rel_tol=1.0e-12,
                    abs_tol=0.0,
                ):
                    raise RuntimeError(
                        "FC declared nudge multiplier does not match the exact "
                        f"runtime output-row current scale for {architecture}/{scheme}."
                    )
                expected_runtime_scale = (
                    exact_row_factor if nudging_mode == "current" else 1.0
                )
                if not math.isclose(
                    float(runtime["nudging_current_scale"]),
                    expected_runtime_scale,
                    rel_tol=1.0e-12,
                    abs_tol=0.0,
                ):
                    raise RuntimeError(
                        "FC runtime nudge scale does not implement the declared mode "
                        f"for {architecture}/{scheme}."
                    )
                init_hash = base._historical_initialization_sha256(
                    runtime["parameters"]
                )
                expected_hash = str(
                    expected_initialization[architecture]["tensor_sha256"]
                )
                if init_hash != expected_hash:
                    raise RuntimeError(
                        f"Initialization hash mismatch for {architecture}/{scheme}."
                    )
                initialization_hashes[(architecture, scheme)] = init_hash
                curvature = _output_curvature_scale(runtime)
                expected_b0 = float(
                    expected_initialization[architecture]["b0_by_scheme"][scheme]
                )
                # The declared guard values were measured on CPU.  CUDA may
                # change the final float32 reduction by one ulp, while the
                # reconstructed parameter hash remains exact.  Match the Conv
                # replay guard and admit only float32-scale device variation.
                if not math.isclose(
                    float(curvature["b0_scheme_init"]),
                    expected_b0,
                    rel_tol=1.0e-6,
                    abs_tol=0.0,
                ):
                    raise RuntimeError(
                        f"Output curvature mismatch for {architecture}/{scheme}: "
                        f"{curvature['b0_scheme_init']} != {expected_b0}."
                    )
                curvature_by_case[(architecture, scheme)] = float(
                    curvature["b0_scheme_init"]
                )
                curvature_rows.append(
                    {
                        "schema": SCHEMA,
                        "architecture": architecture,
                        "scheme": scheme,
                        "checkpoint_role": CHECKPOINT_ROLE,
                        "voltage_amplification": float(
                            schemes[scheme]["voltage_amplification"]
                        ),
                        "current_amplification": float(
                            schemes[scheme]["current_amplification"]
                        ),
                        "dense_interaction_count": len(
                            architectures[architecture]["hidden_layer_shapes"]
                        )
                        + 1,
                        "output_bias_current_row_exponent_L": expected_output_bias_exponent,
                        "output_row_current_scale": exact_row_factor,
                        "nudging_mode": nudging_mode,
                        "runtime_nudging_current_scale": float(
                            runtime["nudging_current_scale"]
                        ),
                        "initialization_tensor_sha256": init_hash,
                        "expected_initialization_tensor_sha256": expected_hash,
                        "initialization_hash_guard_passed": True,
                        "expected_b0_scheme_init": expected_b0,
                        "b0_guard_passed": True,
                        "source_config_sha256": _stable_json_sha256(source_config),
                        **curvature,
                    }
                )
        cross_scheme_initialization_guards = {
            architecture: len(
                {
                    initialization_hashes[(architecture, scheme)]
                    for scheme in schemes
                }
            )
            == 1
            for architecture in architectures
        }
        if not all(cross_scheme_initialization_guards.values()):
            raise RuntimeError("FC initialization differs across amplification schemes.")

        gradient_cfg = config["gradient_contract"]
        normalized_requests = [
            float(value) for value in gradient_cfg["normalized_betas_requested"]
        ]
        cap = float(gradient_cfg["actual_beta_cap"])
        beta_mapping_rows: list[dict[str, Any]] = []
        retained_points: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for architecture in architectures:
            baseline_b0 = curvature_by_case[(architecture, "baseline")]
            amplification_exponent = int(
                architectures[architecture]["nudging_scale_power"]
            )
            for scheme in schemes:
                voltage = float(schemes[scheme]["voltage_amplification"])
                current = float(schemes[scheme]["current_amplification"])
                all_points, retained = _derive_beta_points(
                    normalized_requests=normalized_requests,
                    b0_baseline_init=baseline_b0,
                    b0_scheme_init=curvature_by_case[(architecture, scheme)],
                    voltage_amplification=voltage,
                    current_amplification=current,
                    amplification_exponent=amplification_exponent,
                    actual_beta_cap=cap,
                    normalized_beta_parameterization=str(
                        gradient_cfg.get(
                            "normalized_beta_parameterization",
                            "common_effective",
                        )
                    ),
                )
                retained_points[(architecture, scheme)] = retained
                for point in all_points:
                    beta_mapping_rows.append(
                        {
                            "schema": SCHEMA,
                            "architecture": architecture,
                            "scheme": scheme,
                            "checkpoint_role": CHECKPOINT_ROLE,
                            **point,
                        }
                    )

        parameter_rows: list[dict[str, Any]] = []
        raw_rows: list[dict[str, Any]] = []
        context_rows: list[dict[str, Any]] = []
        phase_rows: list[dict[str, Any]] = []
        state_batch_rows: list[dict[str, Any]] = []
        residual_rows: list[dict[str, Any]] = []
        residual_archive_records: list[tuple[dict[str, Any], np.ndarray]] = []
        parameter_guards: list[dict[str, Any]] = []
        common_state_failures: list[dict[str, Any]] = []
        b0_recheck_failures: list[dict[str, Any]] = []
        completed_contexts = 0
        outcome_counts: defaultdict[str, int] = defaultdict(int)
        precision_cfg = gradient_cfg["precision_gate"]
        adjacent_cfg = gradient_cfg["adjacent_beta_consistency_gate"]
        residual_cfg = config["equilibrium_residual_contract"]

        for architecture in architectures:
            for scheme in schemes:
                source_config = source_configs[(architecture, scheme)]
                points = retained_points[(architecture, scheme)]
                betas = [float(point["beta"]) for point in points]
                point_by_beta = {float(point["beta"]): point for point in points}
                native_t, native_k = [
                    int(value)
                    for value in architectures[architecture]["native_tk_by_scheme"][
                        scheme
                    ]
                ]
                for t_value, k_value, coordinate_kind in _tk_coordinates(
                    config, architecture=architecture, scheme=scheme
                ):
                    runtime = base._build_runtime(
                        source_config,
                        device=device,
                        gradient_iterations=int(k_value),
                        nudging_mode=nudging_mode,
                        current_scale="auto",
                    )
                    init_hash = base._historical_initialization_sha256(
                        runtime["parameters"]
                    )
                    if init_hash != initialization_hashes[(architecture, scheme)]:
                        raise RuntimeError(
                            f"Initialization reconstruction changed for {architecture}/{scheme}."
                        )
                    observed_b0 = float(_output_curvature_scale(runtime)["b0_scheme_init"])
                    expected_b0 = curvature_by_case[(architecture, scheme)]
                    if observed_b0 != expected_b0:
                        b0_recheck_failures.append(
                            {
                                "architecture": architecture,
                                "scheme": scheme,
                                "T": t_value,
                                "K": k_value,
                                "expected_b0": expected_b0,
                                "observed_b0": observed_b0,
                            }
                        )
                    runtime["minimizer_inference"].num_iterations = int(t_value)
                    runtime["inference_iterations"] = int(t_value)
                    runtime["gradient_iterations"] = int(k_value)
                    names = list(runtime["weight_names"])
                    types = dict(zip(names, runtime["weight_types"], strict=True))
                    if len(names) != len(architectures[architecture]["hidden_layer_shapes"]) + 1:
                        raise RuntimeError(f"Unexpected dense weight count for {architecture}.")
                    before_hash = base._parameter_state_sha256(runtime["parameters"])
                    context = {
                        "architecture": architecture,
                        "matched_conv_architecture": architectures[architecture][
                            "matched_conv_architecture"
                        ],
                        "scheme": scheme,
                        "checkpoint_role": CHECKPOINT_ROLE,
                        "coordinate_kind": coordinate_kind,
                        "T": int(t_value),
                        "K": int(k_value),
                        "native_T": native_t,
                        "native_K": native_k,
                        "native_context": coordinate_kind == "native",
                        "shared_anchor_context": coordinate_kind == "native",
                        "voltage_amplification": float(
                            schemes[scheme]["voltage_amplification"]
                        ),
                        "current_amplification": float(
                            schemes[scheme]["current_amplification"]
                        ),
                        "nudging_mode": nudging_mode,
                        "nudging_current_scale": float(
                            runtime["nudging_current_scale"]
                        ),
                        "finite_difference_denominator_scale": (
                            float(runtime["nudging_current_scale"])
                            if nudging_mode == "current"
                            else 1.0
                        ),
                        "b0_baseline_init": curvature_by_case[
                            (architecture, "baseline")
                        ],
                        "b0_scheme_init": expected_b0,
                    }
                    accumulator = _MeanAccumulator()
                    residual_accumulator = conv._ResidualAccumulator()
                    for batch in batches:
                        batch_index = int(batch["batch_index"])
                        images = batch["images"].to(device)
                        labels = batch["labels"].to(device)
                        runtime["network"].set_input(images, reset=True)
                        runtime["minimizer_inference"].compute_equilibrium()
                        runtime["cost_fn"].set_target(labels)
                        free_states = base._clone_states(runtime["free_layers"])
                        free_hash = base._layer_state_sha256(runtime["free_layers"])
                        residual_accumulator.add_endpoint(
                            runtime,
                            gradient_source=runtime["energy_fn"],
                            phase="post_T_free",
                            beta=0.0,
                            source_indices=batch["source_indices"],
                        )
                        (
                            bptt,
                            estimates,
                            diagnostics,
                            displacements,
                            outcomes,
                            precision_endpoints,
                        ) = conv._run_gradient_phases(
                                runtime,
                                free_states=free_states,
                                betas=betas,
                                zero_epsilon=float(gradient_cfg["zero_epsilon"]),
                                state_reference_norm_epsilon=float(
                                    config["state_displacement_contract"]["zero_epsilon"]
                                ),
                                zero_endpoint_relative_tolerance=float(
                                    gradient_cfg[
                                        "zero_endpoint_equivalence_relative_l2_tolerance"
                                    ]
                                ),
                                eqprop_variant=EQPROP_VARIANT,
                                residual_accumulator=residual_accumulator,
                                source_indices=batch["source_indices"],
                            )
                        if any(
                            row["phase_start_state_sha256"] != free_hash
                            for row in diagnostics
                        ):
                            common_state_failures.append(
                                {**context, "batch_index": batch_index}
                            )
                        batch_size_actual = int(images.shape[0])
                        for name in names:
                            accumulator.add(
                                ("bptt", name), bptt[name], batch_size_actual
                            )
                            available_betas = [
                                beta for beta in betas if outcomes[beta] == "ok"
                            ]
                            if available_betas:
                                zero_reference = precision_endpoints[
                                    available_betas[0]
                                ][name]["zero"]
                                if any(
                                    not torch.equal(
                                        zero_reference,
                                        precision_endpoints[beta][name]["zero"],
                                    )
                                    for beta in available_betas[1:]
                                ):
                                    raise RuntimeError(
                                        "The returned matched beta=0 endpoint gradient changed across beta."
                                    )
                                accumulator.add(
                                    ("zero", name),
                                    zero_reference,
                                    batch_size_actual,
                                )
                            for beta in betas:
                                outcome = outcomes[beta]
                                if outcome != "ok":
                                    raw_rows.append(
                                        {
                                            "schema": SCHEMA,
                                            **context,
                                            **point_by_beta[beta],
                                            "batch_index": batch_index,
                                            "batch_payload_sha256": batch[
                                                "payload_sha256"
                                            ],
                                            "parameter_name": name,
                                            "parameter_type": types[name],
                                            "bias_excluded": True,
                                            "outcome": outcome,
                                        }
                                    )
                                    continue
                                estimate = estimates[beta][name]
                                zero_endpoint = precision_endpoints[beta][name][
                                    "zero"
                                ]
                                positive = precision_endpoints[beta][name][
                                    "positive"
                                ]
                                accumulator.add(
                                    ("eqprop", beta, name),
                                    estimate,
                                    batch_size_actual,
                                )
                                accumulator.add(
                                    ("positive", beta, name),
                                    positive,
                                    batch_size_actual,
                                )
                                gradient_metrics = base._vector_metrics(
                                    bptt[name],
                                    estimate,
                                    zero_epsilon=float(
                                        gradient_cfg["zero_epsilon"]
                                    ),
                                )
                                precision = _precision_metrics(
                                    zero_endpoint,
                                    positive,
                                    minimum_snr=float(
                                        precision_cfg["minimum_snr"]
                                    ),
                                )
                                raw_rows.append(
                                    {
                                        "schema": SCHEMA,
                                        **context,
                                        **point_by_beta[beta],
                                        "batch_index": batch_index,
                                        "batch_payload_sha256": batch[
                                            "payload_sha256"
                                        ],
                                        "parameter_name": name,
                                        "parameter_type": types[name],
                                        "bias_excluded": True,
                                        "outcome": (
                                            "ok"
                                            if gradient_metrics["cosine"] is not None
                                            else "dead_gradient"
                                        ),
                                        **gradient_metrics,
                                        **precision,
                                    }
                                )
                        for beta, outcome in outcomes.items():
                            outcome_counts[outcome] += 1
                        for diagnostic in diagnostics:
                            beta = float(diagnostic["beta"])
                            point = point_by_beta.get(beta, {})
                            phase_rows.append(
                                {
                                    "schema": SCHEMA,
                                    **context,
                                    **point,
                                    "eqprop_variant": EQPROP_VARIANT,
                                    "batch_index": batch_index,
                                    "batch_payload_sha256": batch[
                                        "payload_sha256"
                                    ],
                                    **diagnostic,
                                }
                            )
                        for displacement in displacements:
                            beta = float(displacement["beta"])
                            point = point_by_beta.get(beta, {})
                            state_batch_rows.append(
                                {
                                    "schema": SCHEMA,
                                    **context,
                                    **point,
                                    "eqprop_variant": EQPROP_VARIANT,
                                    "batch_index": batch_index,
                                    "batch_payload_sha256": batch[
                                        "payload_sha256"
                                    ],
                                    **displacement,
                                }
                            )

                    context_residual_rows = residual_accumulator.summary_rows(
                        context={**context, "eqprop_variant": EQPROP_VARIANT},
                        expected_examples=requested_examples,
                        p90_threshold=float(residual_cfg["p90_threshold"]),
                    )
                    for row in context_residual_rows:
                        maximum = row["selected_maximum"]
                        row["uniform_residual_threshold"] = float(
                            residual_cfg["uniform_flag_threshold"]
                        )
                        row["uniform_residual_passed"] = bool(
                            maximum is not None
                            and float(maximum)
                            < float(residual_cfg["uniform_flag_threshold"])
                        )
                        row["hard_failure_threshold"] = float(
                            residual_cfg["hard_failure_threshold"]
                        )
                        row["hard_failure_triggered"] = bool(
                            maximum is not None
                            and float(maximum)
                            >= float(residual_cfg["hard_failure_threshold"])
                        )
                        point = point_by_beta.get(float(row["beta"]), {})
                        row.update(point)
                    residual_rows.extend(context_residual_rows)
                    residual_archive_records.extend(
                        residual_accumulator.archive_records(
                            context={**context, "eqprop_variant": EQPROP_VARIANT}
                        )
                    )
                    residual_gates = conv._residual_gate_by_beta(
                        context_residual_rows,
                        betas=betas,
                        eqprop_variant=EQPROP_VARIANT,
                    )
                    context_parameter_rows, context_selection = (
                        _parameter_rows_and_selection(
                            accumulator,
                            context=context,
                            points=points,
                            names=names,
                            types=types,
                            expected_examples=requested_examples,
                            zero_epsilon=float(gradient_cfg["zero_epsilon"]),
                            residual_gates=residual_gates,
                            minimum_snr=float(precision_cfg["minimum_snr"]),
                            sentinel_max_beta_hat=float(
                                precision_cfg[
                                    "low_beta_sentinel_max_beta_hat_requested"
                                ]
                            ),
                            adjacent_cosine_minimum=float(
                                adjacent_cfg["eqprop_cosine_minimum"]
                            ),
                            adjacent_norm_delta_maximum=float(
                                adjacent_cfg["symmetric_norm_delta_maximum"]
                            ),
                        )
                    )
                    parameter_rows.extend(context_parameter_rows)
                    context_rows.append(context_selection)

                    gradient_arrays: dict[str, Any] = {
                        "metadata_json": np.asarray(
                            json.dumps(
                                {
                                    **context,
                                    "cohort_sha256": cohort["cohort_sha256"],
                                    "example_count": requested_examples,
                                },
                                sort_keys=True,
                                allow_nan=False,
                            )
                        )
                    }
                    for name in names:
                        gradient_arrays[f"bptt__{name}"] = accumulator.mean(
                            ("bptt", name)
                        ).numpy()
                        for point in points:
                            beta = float(point["beta"])
                            if accumulator.examples.get(("eqprop", beta, name), 0) != requested_examples:
                                continue
                            key = f"request_{int(point['request_index']):02d}"
                            gradient_arrays[f"eqprop__{key}__{name}"] = accumulator.mean(
                                ("eqprop", beta, name)
                            ).numpy()
                    archive_path = (
                        run_dir
                        / "artifacts/mean_gradients"
                        / f"{architecture}__{scheme}__{coordinate_kind}__T{t_value}__K{k_value}.npz"
                    )
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(archive_path, **gradient_arrays)

                    after_hash = base._parameter_state_sha256(runtime["parameters"])
                    unchanged = before_hash == after_hash
                    parameter_guards.append(
                        {
                            **context,
                            "parameter_state_sha256_before": before_hash,
                            "parameter_state_sha256_after": after_hash,
                            "unchanged": unchanged,
                            "optimizer_constructed": False,
                            "optimizer_steps_applied": False,
                            "biases_active_in_dynamics": True,
                            "bias_gradients_excluded": True,
                        }
                    )
                    if not unchanged:
                        raise RuntimeError(
                            f"Parameters changed in {architecture}/{scheme}/{coordinate_kind}."
                        )
                    completed_contexts += 1
                    base.append_metric(
                        run_dir / "metrics.jsonl",
                        {
                            "stage": "fc_context_complete",
                            "split": "validation",
                            **context,
                            "examples": requested_examples,
                            "retained_beta_count": len(points),
                            "selection_outcome": context_selection[
                                "selection_outcome"
                            ],
                        },
                    )
                    base.update_status_progress(
                        run_dir,
                        {
                            "stage": "fc_initialization_replay",
                            "completed_contexts": completed_contexts,
                            "total_contexts": total_contexts,
                            **context,
                        },
                    )

        if completed_contexts != total_contexts:
            raise RuntimeError(
                f"Expected {total_contexts} FC contexts, completed {completed_contexts}."
            )
        if common_state_failures:
            raise RuntimeError("At least one FC phase missed the common post-T state.")
        if b0_recheck_failures:
            raise RuntimeError("An FC output curvature scale changed across reconstruction.")

        state_summary = conv._summarize_state_rows(
            state_batch_rows, expected_batch_count=len(batches)
        )
        coordinate_kind_by_context = {
            (architecture, scheme, int(t_value), int(k_value)): coordinate_kind
            for architecture in architectures
            for scheme in schemes
            for t_value, k_value, coordinate_kind in _tk_coordinates(
                config, architecture=architecture, scheme=scheme
            )
        }
        mapping_by_case_beta = {
            (row["architecture"], row["scheme"], float(row["beta"])): row
            for row in beta_mapping_rows
            if bool(row["retained_after_actual_beta_deduplication"])
        }
        for row in state_summary:
            context_key = (
                str(row["architecture"]),
                str(row["scheme"]),
                int(row["T"]),
                int(row["K"]),
            )
            coordinate_kind = coordinate_kind_by_context.get(context_key)
            if coordinate_kind is None:
                raise RuntimeError(
                    f"No FC coordinate label for summarized state context {context_key}."
                )
            # The shared Conv summarizer intentionally keys on T/K and does
            # not preserve FC's native-versus-long display label.
            row["schema"] = SCHEMA
            row["coordinate_kind"] = coordinate_kind
            point = mapping_by_case_beta.get(
                (str(row["architecture"]), str(row["scheme"]), float(row["beta"]))
            )
            if point is not None:
                row.update(
                    {
                        key: point[key]
                        for key in (
                            "request_index",
                            "beta_hat_requested",
                            "effective_beta",
                            "beta_hat_effective",
                            "b0_baseline_init",
                            "b0_scheme_init",
                            "actual_beta_over_b0_scheme_init",
                        )
                    }
                )

        hard_residual_failures = [
            row for row in residual_rows if bool(row["hard_failure_triggered"])
        ]
        expected_parameter_rows = sum(
            len(retained_points[(architecture, scheme)])
            * (len(architectures[architecture]["hidden_layer_shapes"]) + 1)
            * len(_tk_coordinates(config, architecture=architecture, scheme=scheme))
            for architecture in architectures
            for scheme in schemes
        )
        exact_coverage = {
            "expected_context_count": total_contexts,
            "observed_context_count": len(context_rows),
            "contexts_exact": len(context_rows) == total_contexts,
            "expected_parameter_row_count": expected_parameter_rows,
            "observed_parameter_row_count": len(parameter_rows),
            "parameter_rows_exact": len(parameter_rows) == expected_parameter_rows,
            "all_parameter_rows_complete": all(
                bool(row["coverage_complete"]) for row in parameter_rows
            ),
            "all_state_rows_complete": all(
                bool(row["coverage_complete"])
                for row in state_summary
                if str(row["phase"]) == "positive"
            ),
            "all_residual_rows_complete": all(
                bool(row["coverage_complete"]) for row in residual_rows
            ),
            "low_beta_precision_sentinel_row_count": sum(
                bool(row["low_beta_float32_cancellation_sentinel"])
                for row in parameter_rows
            ),
            "precision_gate_pass_row_count": sum(
                bool(row["precision_gate_passed"]) for row in parameter_rows
            ),
            "precision_gate_fail_row_count": sum(
                not bool(row["precision_gate_passed"]) for row in parameter_rows
            ),
            "residual_p90_gate_pass_row_count": sum(
                bool(row["gate_passed"]) for row in residual_rows
            ),
            "uniform_residual_pass_row_count": sum(
                bool(row["uniform_residual_passed"]) for row in residual_rows
            ),
            "hard_residual_failure_row_count": len(hard_residual_failures),
        }
        guards = {
            "cross_scheme_initialization_identical": cross_scheme_initialization_guards,
            "all_cross_scheme_initializations_identical": all(
                cross_scheme_initialization_guards.values()
            ),
            "parameter_guards": parameter_guards,
            "all_parameter_tensors_unchanged": all(
                bool(row["unchanged"]) for row in parameter_guards
            ),
            "common_post_t_state_failures": common_state_failures,
            "all_phase_starts_match_common_post_t_state": not common_state_failures,
            "b0_recheck_failures": b0_recheck_failures,
            "all_b0_rechecks_exact": not b0_recheck_failures,
            "exact_coverage": exact_coverage,
            "biases_active_in_dynamics": True,
            "bias_gradients_excluded": True,
            "optimizer_constructed": False,
            "optimizer_steps_applied": False,
            "official_test_read": False,
        }

        base._write_csv(run_dir / "curvature_scales.csv", curvature_rows)
        base._write_csv(run_dir / "beta_mapping.csv", beta_mapping_rows)
        base._write_csv(run_dir / "raw_batch_metrics.csv", raw_rows)
        base._write_csv(run_dir / "parameter_summary.csv", parameter_rows)
        base._write_csv(run_dir / "context_beta_selection.csv", context_rows)
        base._write_csv(run_dir / "phase_diagnostics.csv", phase_rows)
        base._write_csv(
            run_dir / "state_displacement_batches.csv", state_batch_rows
        )
        base._write_csv(run_dir / "state_displacement_summary.csv", state_summary)
        base._write_csv(
            run_dir / "equilibrium_residual_summary.csv", residual_rows
        )
        conv._write_residual_archive(
            run_dir / "artifacts/equilibrium_residual_per_sample.npz",
            residual_archive_records,
        )
        base._write_json(run_dir / "read_only_guards.json", guards)

        plots_dir = run_dir / "artifacts/plots"
        for architecture in architectures:
            _plot_layerwise_cosines(
                parameter_rows,
                architecture=architecture,
                path=plots_dir / f"layerwise_cosine_beta_{architecture}.png",
            )
            _plot_displacements(
                state_summary,
                architecture=architecture,
                path=plots_dir / f"displacement_beta_{architecture}.png",
            )
        _write_report(
            run_dir / "report.md",
            config=config,
            context_rows=context_rows,
            cohort=cohort,
            smoke=bool(args.smoke),
        )

        if hard_residual_failures:
            sample = hard_residual_failures[0]
            raise RuntimeError(
                "Hard equilibrium residual failure: "
                f"{len(hard_residual_failures)} layer/phase rows have maximum >= "
                f"{residual_cfg['hard_failure_threshold']}; first="
                f"{sample['architecture']}/{sample['scheme']}/T{sample['T']}/K{sample['K']}/"
                f"{sample['phase']}/{sample['state_layer_name']} max={sample['selected_maximum']}."
            )

        completion = {
            "criteria_met": bool(
                exact_coverage["contexts_exact"]
                and exact_coverage["parameter_rows_exact"]
                and exact_coverage["all_parameter_rows_complete"]
                and exact_coverage["all_state_rows_complete"]
                and exact_coverage["all_residual_rows_complete"]
                and exact_coverage["low_beta_precision_sentinel_row_count"] > 0
                and not hard_residual_failures
                and guards["all_cross_scheme_initializations_identical"]
                and guards["all_parameter_tensors_unchanged"]
                and guards["all_phase_starts_match_common_post_t_state"]
                and guards["all_b0_rechecks_exact"]
            ),
            "exact_context_coverage": exact_coverage["contexts_exact"],
            "exact_parameter_coverage": exact_coverage["parameter_rows_exact"],
            "complete_parameter_coverage": exact_coverage[
                "all_parameter_rows_complete"
            ],
            "complete_state_coverage": exact_coverage["all_state_rows_complete"],
            "complete_residual_coverage": exact_coverage[
                "all_residual_rows_complete"
            ],
            "low_beta_precision_sentinels_reported": exact_coverage[
                "low_beta_precision_sentinel_row_count"
            ]
            > 0,
            "no_hard_residual_failure": not hard_residual_failures,
            "cross_scheme_initializations_identical": guards[
                "all_cross_scheme_initializations_identical"
            ],
            "parameter_tensors_unchanged": guards[
                "all_parameter_tensors_unchanged"
            ],
            "all_phase_starts_match_common_post_t_state": guards[
                "all_phase_starts_match_common_post_t_state"
            ],
            "b0_rechecks_exact": guards["all_b0_rechecks_exact"],
            "bias_gradients_excluded": True,
            "optimizer_steps_applied": False,
            "official_test_read": False,
        }
        if not completion["criteria_met"]:
            raise RuntimeError(f"FC completion criteria failed: {completion}.")
        result = base.complete_run(
            run_dir,
            terminal_metrics={
                "synthetic_initialization_cases": len(source_configs),
                "tk_contexts": len(context_rows),
                "cohort_examples": requested_examples,
                "parameter_rows": len(parameter_rows),
                "state_summary_rows": len(state_summary),
                "equilibrium_residual_summary_rows": len(residual_rows),
                "selected_context_count": sum(
                    row["selection_outcome"] == "selected" for row in context_rows
                ),
                "no_converged_adjacent_pair_context_count": sum(
                    row["selection_outcome"] == "no_converged_adjacent_pair"
                    for row in context_rows
                ),
                "positive_phase_outcomes": dict(outcome_counts),
                "precision_gate_pass_rows": exact_coverage[
                    "precision_gate_pass_row_count"
                ],
                "precision_gate_fail_rows": exact_coverage[
                    "precision_gate_fail_row_count"
                ],
                "bias_gradients_excluded": True,
                "optimizer_steps_applied": False,
                "official_test_read": False,
            },
            completion=completion,
        )
        errors = base.validate_run(run_dir)
        if errors:
            raise RuntimeError(
                f"Completed FC reporting bundle failed validation: {'; '.join(errors)}"
            )
        return result
    except BaseException as error:
        status_path = run_dir / "status.json"
        if status_path.is_file():
            status = base._read_json(status_path)
            if status.get("state") == "running" and not (run_dir / "result.json").exists():
                base.fail_run(run_dir, error=error)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--runtime-source-root", type=Path, default=base.RUNTIME_SOURCE_ROOT
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1"
        ),
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/filip/datasets/mnist")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--target", default="local")
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_analysis(args)
    run_id = args.run_id or (
        "fc-smoke-attempt-01" if args.smoke else "fc-initialization-control"
    )
    print(
        json.dumps(
            {
                "state": "complete",
                "result_sha256": base.sha256_file(
                    args.output_root.expanduser().resolve() / run_id / "result.json"
                ),
                "terminal_metrics": result["terminal_metrics"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
