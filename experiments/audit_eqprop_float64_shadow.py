#!/usr/bin/env python3
"""Audit one-sided or centered EqProp and BPTT in real float32/float64 dynamics.

The input is a completed native-T/K Conv beta-selection run.  For each
selected case this runner reconstructs the same source checkpoint, fixed
validation batch, T, K, and *actual* beta twice.  One runtime stays float32;
the other is converted in full to float64 immediately after checkpoint load
and before the batch state reset.  The finite difference is evaluated in the
runtime dtype, so this audit cannot hide float32 cancellation by casting two
already-rounded endpoints before subtraction.

This is a read-only ordinary-MNIST diagnostic.  Biases remain active in the
dynamics, but only ConvWeight and DenseWeight EqProp and BPTT gradients are
compared.  No optimizer is constructed and the official test split is never
read.
"""

from __future__ import annotations

import argparse
import csv
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
    / "configs/conv/perfectdiode_conv123_positive_onesided_small_beta_native_seed0_20260810_v1.json"
)
EXTENDED_RUNNER = (
    REPOSITORY_ROOT / "experiments/analyze_conv_eqprop_bptt_beta_tk_displacement.py"
)
SCHEMA = "perfectdiode-conv-current-eqprop-true-dtype-audit/v3"
PRECISIONS = ("float32", "float64")
CHECKPOINT_ROLES = ("reconstructed_initialization", "best_validation")
ARCHITECTURES = ("conv1", "conv2", "conv3")
SCHEMES = ("baseline", "ours", "legacy")
COSINE_MINIMUM = 0.99
SYMMETRIC_NORM_DELTA_MAXIMUM = 0.10
NORM_EPSILON = 1.0e-30


def _load_extended_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_eqprop_beta_tk_for_float64_shadow", EXTENDED_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {EXTENDED_RUNNER}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


extended = _load_extended_runner()
base = extended.base
np = base.np
torch = base.torch


def _json_semantic_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "none", "null", "nan"}:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite float, got {value!r}.")
    return result


def _parse_case_spec(value: str) -> dict[str, Any]:
    """Parse ARCH:SCHEME:ROLE[:ACTUAL_BETA]."""

    pieces = value.split(":")
    if len(pieces) not in {3, 4}:
        raise ValueError(
            "--case must use ARCH:SCHEME:ROLE[:ACTUAL_BETA], for example "
            "conv3:legacy:best_validation:1e-8."
        )
    architecture, scheme, role = pieces[:3]
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}.")
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme {scheme!r}.")
    if role not in CHECKPOINT_ROLES:
        raise ValueError(f"Unknown checkpoint role {role!r}.")
    beta = None if len(pieces) == 3 else float(pieces[3])
    if beta is not None and (not math.isfinite(beta) or beta <= 0.0):
        raise ValueError("An explicit actual beta must be finite and positive.")
    return {
        "architecture": architecture,
        "scheme": scheme,
        "checkpoint_role": role,
        "explicit_beta": beta,
    }


def _selection_csv(source_run: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = (
        "stage_beta_selection.csv",
        "adjacent_beta_selection.csv",
        "context_beta_selection.csv",
    )
    for name in candidates:
        path = source_run / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No beta-selection CSV was found in {source_run}; tried {candidates}."
    )


def _actual_beta_from_selection(row: Mapping[str, Any]) -> float | None:
    for field in (
        "selected_actual_beta",
        "selected_beta_actual",
        "selected_beta",
        "actual_beta",
    ):
        if field in row:
            return _optional_float(row[field])
    return None


def _native_selection_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in rows:
        architecture = str(raw.get("architecture", ""))
        scheme = str(raw.get("scheme", ""))
        role = str(raw.get("checkpoint_role", ""))
        if architecture not in ARCHITECTURES or scheme not in SCHEMES or role not in CHECKPOINT_ROLES:
            continue
        if "native_context" in raw:
            native = _parse_bool(raw["native_context"])
        else:
            native = (
                int(raw["T"]) == int(raw.get("native_T", raw["T"]))
                and int(raw["K"]) == int(raw.get("native_K", raw["K"]))
            )
        if not native:
            continue
        key = (architecture, scheme, role)
        if key in output:
            raise ValueError(f"Multiple native beta-selection rows for {key}.")
        output[key] = dict(raw)
    return output


def _requested_case_specs(
    explicit_specs: Sequence[str], *, all_18: bool
) -> list[dict[str, Any]]:
    if explicit_specs and all_18:
        raise ValueError("Use either repeated --case or --all-18, not both.")
    if all_18:
        specs = [
            {
                "architecture": architecture,
                "scheme": scheme,
                "checkpoint_role": role,
                "explicit_beta": None,
            }
            for architecture in ARCHITECTURES
            for scheme in SCHEMES
            for role in CHECKPOINT_ROLES
        ]
    elif explicit_specs:
        specs = [_parse_case_spec(value) for value in explicit_specs]
    else:
        # Default precision shadow: both deeper Conv architectures, all three
        # amplification schemes, at their trained checkpoint.
        specs = [
            {
                "architecture": architecture,
                "scheme": scheme,
                "checkpoint_role": "best_validation",
                "explicit_beta": None,
            }
            for architecture in ("conv2", "conv3")
            for scheme in SCHEMES
        ]
    keys = [
        (spec["architecture"], spec["scheme"], spec["checkpoint_role"])
        for spec in specs
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate --case selectors are not allowed.")
    return specs


def _resolve_selected_cases(
    selection_rows: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
    *,
    explicit_beta_label: str = "explicit_cli",
) -> list[dict[str, Any]]:
    native = _native_selection_rows(selection_rows)
    resolved: list[dict[str, Any]] = []
    for spec in specs:
        key = (
            str(spec["architecture"]),
            str(spec["scheme"]),
            str(spec["checkpoint_role"]),
        )
        if key not in native:
            raise ValueError(f"Missing native beta-selection row for {key}.")
        row = native[key]
        beta = spec.get("explicit_beta")
        explicit_beta = beta is not None
        source = explicit_beta_label if explicit_beta else "stage_selected"
        if beta is None:
            beta = _actual_beta_from_selection(row)
        if beta is None or not math.isfinite(float(beta)) or float(beta) <= 0.0:
            raise ValueError(
                f"No finite positive selected actual beta for {key}; pass an explicit --case beta."
            )
        if not explicit_beta and row.get("selection_rule") != (
            "lowest_uncapped_adjacent_consistent_pair_choose_larger"
        ):
            raise ValueError(
                f"Native row for {key} is not the frozen adjacent-consistency stage selection."
            )
        resolved.append(
            {
                "architecture": key[0],
                "scheme": key[1],
                "checkpoint_role": key[2],
                "T": int(row["T"]),
                "K": int(row["K"]),
                "actual_beta": float(beta),
                "beta_source": source,
                "beta_explicit_cli": explicit_beta,
                "source_selection_row": row,
                "source_selection_row_sha256": _json_semantic_sha256(row),
            }
        )
    return resolved


def _apply_tk_override(
    selected_cases: Sequence[dict[str, Any]],
    inventory_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    override: Sequence[int] | None,
) -> None:
    """Resolve replay T/K while retaining the source-native coordinate."""

    if override is not None:
        if len(override) != 2:
            raise ValueError("--tk-override requires exactly T and K.")
        replay_t, replay_k = (int(value) for value in override)
        if replay_t <= 0 or replay_k <= 0:
            raise ValueError("--tk-override T and K must be positive integers.")
    else:
        replay_t = replay_k = -1

    for selected in selected_cases:
        key = (str(selected["architecture"]), str(selected["scheme"]))
        case = inventory_by_key[key]
        native_t = int(case["native_T"])
        native_k = int(case["native_K"])
        if (int(selected["T"]), int(selected["K"])) != (native_t, native_k):
            raise ValueError(
                f"Source selection for {key} does not match inventory-native T/K."
            )
        selected["source_native_T"] = native_t
        selected["source_native_K"] = native_k
        if override is None:
            selected["tk_source"] = "source_native"
        else:
            if not bool(selected.get("beta_explicit_cli")):
                raise ValueError(
                    "--tk-override requires an explicit beta in every --case."
                )
            selected["T"] = replay_t
            selected["K"] = replay_k
            selected["tk_source"] = "explicit_cli_override"
        selected["native_context"] = bool(
            int(selected["T"]) == native_t and int(selected["K"]) == native_k
        )


def _resolve_beta_row(
    *,
    config: Mapping[str, Any],
    case: Mapping[str, Any],
    requested_actual_beta: float,
    allow_out_of_grid: bool,
) -> dict[str, Any]:
    """Resolve an exact source-grid beta or a declared uncapped extension."""

    beta_grid = extended._beta_grid_for_case(config["gradient_contract"], case)
    matched = [
        row
        for row in beta_grid
        if math.isclose(
            float(requested_actual_beta),
            float(row["beta"]),
            rel_tol=1.0e-8,
            abs_tol=1.0e-15,
        )
    ]
    if len(matched) == 1:
        return {
            **dict(matched[0]),
            "source_grid_member": True,
            "source_grid_cap_bypassed": False,
        }
    if len(matched) > 1:
        raise ValueError("Source beta grid contains duplicate resolved beta values.")
    if not allow_out_of_grid:
        raise ValueError(
            f"Actual beta {requested_actual_beta} is absent from the resolved source grid for "
            f"{case['architecture']}/{case['scheme']}."
        )
    if not math.isfinite(float(requested_actual_beta)) or requested_actual_beta <= 0.0:
        raise ValueError("Out-of-grid actual beta must be finite and positive.")

    contract = config["gradient_contract"]
    architecture = str(case["architecture"])
    exponent_by_architecture = contract.get("nudging_scale_power_by_architecture")
    if not isinstance(exponent_by_architecture, Mapping):
        raise ValueError("Out-of-grid beta requires explicit nudging-scale powers.")
    depth = int(exponent_by_architecture[architecture])
    voltage = float(case.get("voltage_amp", case["voltage_amplification"]))
    current = float(case.get("current_amp", case["current_amplification"]))
    factor = (voltage / current) ** depth
    effective = float(requested_actual_beta) * factor
    if not math.isfinite(effective) or effective <= 0.0:
        raise ValueError("Out-of-grid injected beta must be finite and positive.")
    baseline_b0 = float(
        contract["baseline_init_output_curvature_scale_b0_by_architecture"][architecture]
    )
    source_cap = _optional_float(contract.get("actual_beta_cap"))
    return {
        "beta": float(requested_actual_beta),
        "beta_effective": effective,
        "beta_hat": effective / baseline_b0,
        "beta_hat_requested": float(requested_actual_beta) / baseline_b0,
        "capped": False,
        "actual_beta_cap": source_cap,
        "source_grid_member": False,
        "source_grid_cap_bypassed": bool(
            source_cap is not None and effective > source_cap
        ),
    }


def _all_runtime_parameters(runtime: Mapping[str, Any]) -> list[Any]:
    energy = runtime["energy_fn"]
    values = list(getattr(energy, "_all_params", runtime["parameters"]))
    unique: list[Any] = []
    seen: set[int] = set()
    for value in values:
        if id(value) not in seen:
            seen.add(id(value))
            unique.append(value)
    return unique


def _variable_dtype_rows(runtime: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in runtime["energy_fn"].layers():
        rows.append(
            {
                "variable_kind": "layer",
                "variable_name": str(layer.name),
                "variable_type": layer.__class__.__name__,
                "dtype": str(layer.state.dtype),
                "device": str(layer.state.device),
                "shape": list(layer.state.shape),
            }
        )
    for parameter in _all_runtime_parameters(runtime):
        rows.append(
            {
                "variable_kind": "parameter",
                "variable_name": str(parameter.name).strip(),
                "variable_type": parameter.__class__.__name__,
                "dtype": str(parameter.state.dtype),
                "device": str(parameter.state.device),
                "shape": list(parameter.state.shape),
            }
        )
    return rows


def _convert_runtime_dtype(runtime: Mapping[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    """Convert every model layer and parameter state, preserving values exactly."""

    before = _variable_dtype_rows(runtime)
    variables = [*runtime["energy_fn"].layers(), *_all_runtime_parameters(runtime)]
    for variable in variables:
        requires_grad = bool(variable.state.requires_grad)
        converted = variable.state.detach().to(dtype=dtype).clone()
        converted.requires_grad_(requires_grad)
        variable.state = converted
    after = _variable_dtype_rows(runtime)
    target = str(dtype)
    if not after or any(row["dtype"] != target for row in after):
        raise RuntimeError(f"Not every runtime variable converted to {target}.")
    return {
        "before": before,
        "after_conversion_before_state_reset": after,
        "target_dtype": target,
        "all_runtime_variables_target_dtype": True,
    }


def _set_input_and_reset(
    runtime: Mapping[str, Any], images: torch.Tensor, *, dtype: torch.dtype
) -> dict[str, Any]:
    """Set a dtype-matched input and perform an explicit dtype-safe zero reset."""

    if images.dtype != dtype:
        raise ValueError(f"Input dtype {images.dtype} differs from target {dtype}.")
    layers = runtime["energy_fn"].layers()
    input_layer = layers[0]
    input_layer.set_input(images)
    batch_size = int(images.shape[0])
    for layer in runtime["free_layers"]:
        layer.state = torch.zeros(
            (batch_size, *tuple(layer.shape)),
            dtype=dtype,
            device=images.device,
            requires_grad=False,
        )
    rows = _variable_dtype_rows(runtime)
    if any(row["dtype"] != str(dtype) for row in rows):
        raise RuntimeError("The batch reset reintroduced a different dtype.")
    return {
        "raw_input_dtype": str(images.dtype),
        "network_input_dtype": str(input_layer.state.dtype),
        "network_input_sha256": base._tensor_sha256(input_layer.state),
        "after_input_and_state_reset": rows,
        "all_runtime_variables_target_dtype_after_reset": True,
    }


def _numeric_parameter_sha256(parameters: Sequence[Any]) -> str:
    """Hash parameter values after canonical float64 conversion, ignoring source dtype."""

    digest = hashlib.sha256()
    digest.update(b"drn-ordered-parameter-values-as-float64/v1\0")
    for parameter in parameters:
        tensor = parameter.state.detach().cpu().to(torch.float64).contiguous()
        digest.update(str(parameter.name).strip().encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _native_eqprop_estimate(
    *,
    eqprop_variant: str,
    beta: float,
    denominator_scale: float = 1.0,
    zero: torch.Tensor,
    negative: torch.Tensor | None = None,
    positive: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract and divide in the endpoint dtype, then return numerator/estimate."""

    if zero.shape != positive.shape or zero.dtype != positive.dtype:
        raise ValueError("Matched endpoints must have identical shapes and dtypes.")
    if zero.device != positive.device:
        raise ValueError("Matched endpoints must be on the same device.")
    if not zero.dtype.is_floating_point:
        raise ValueError("EqProp endpoints must be floating point tensors.")
    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("Actual beta must be finite and positive.")
    denominator_scale = float(denominator_scale)
    if not math.isfinite(denominator_scale) or denominator_scale <= 0.0:
        raise ValueError("EqProp denominator scale must be finite and positive.")
    if eqprop_variant == "positive_one_sided":
        numerator = positive - zero
        denominator_multiplier = 1.0
    elif eqprop_variant == "centered":
        if negative is None:
            raise ValueError("Centered EqProp requires a negative endpoint.")
        if (
            negative.shape != positive.shape
            or negative.dtype != positive.dtype
            or negative.device != positive.device
        ):
            raise ValueError(
                "Centered negative and positive endpoints must have identical "
                "shapes, dtypes, and devices."
            )
        numerator = positive - negative
        denominator_multiplier = 2.0
    else:
        raise ValueError(f"Unsupported EqProp variant {eqprop_variant!r}.")
    estimate = numerator / torch.as_tensor(
        denominator_multiplier * beta * denominator_scale,
        dtype=numerator.dtype,
        device=numerator.device,
    )
    base._finite(numerator, f"{eqprop_variant} finite-difference numerator")
    base._finite(estimate, f"{eqprop_variant} EqProp estimate")
    if numerator.dtype != zero.dtype or estimate.dtype != zero.dtype:
        raise RuntimeError("Finite-difference arithmetic changed dtype.")
    return numerator, estimate


def _native_one_sided_estimate(
    *,
    beta: float,
    denominator_scale: float = 1.0,
    zero: torch.Tensor,
    positive: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible wrapper for the original v2 helper contract."""

    return _native_eqprop_estimate(
        eqprop_variant="positive_one_sided",
        beta=beta,
        denominator_scale=denominator_scale,
        zero=zero,
        positive=positive,
    )


def _precision_metrics(float32_gradient: torch.Tensor, float64_gradient: torch.Tensor) -> dict[str, Any]:
    left = float32_gradient.detach().cpu().to(torch.float64).reshape(-1)
    right = float64_gradient.detach().cpu().to(torch.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Precision gradient shape mismatch: {left.shape} != {right.shape}.")
    base._finite(left, "float32 shadow gradient")
    base._finite(right, "float64 shadow gradient")
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    dot = float(torch.dot(left, right))
    denominator = left_norm * right_norm
    cosine = dot / denominator if denominator > NORM_EPSILON else None
    symmetric_norm_delta = 2.0 * abs(left_norm - right_norm) / max(
        left_norm + right_norm, NORM_EPSILON
    )
    difference_norm = float(torch.linalg.vector_norm(left - right))
    gate = bool(
        cosine is not None
        and cosine >= COSINE_MINIMUM
        and symmetric_norm_delta <= SYMMETRIC_NORM_DELTA_MAXIMUM
    )
    return {
        "element_count": int(left.numel()),
        "float32_l2": left_norm,
        "float64_l2": right_norm,
        "dot_product": dot,
        "cosine": cosine,
        "difference_l2": difference_norm,
        "relative_l2_difference_over_float64": (
            difference_norm / right_norm if right_norm > NORM_EPSILON else None
        ),
        "symmetric_norm_delta": symmetric_norm_delta,
        "cosine_minimum": COSINE_MINIMUM,
        "symmetric_norm_delta_maximum": SYMMETRIC_NORM_DELTA_MAXIMUM,
        "precision_gate_passed": gate,
    }


def _eqprop_bptt_metrics(
    *, eqprop_gradient: torch.Tensor, bptt_gradient: torch.Tensor
) -> dict[str, Any]:
    """Compare one native-dtype EqProp estimate with its same-dtype BPTT reference."""

    eqprop = eqprop_gradient.detach().cpu().to(torch.float64).reshape(-1)
    bptt = bptt_gradient.detach().cpu().to(torch.float64).reshape(-1)
    if eqprop.shape != bptt.shape:
        raise ValueError(
            f"EqProp/BPTT gradient shape mismatch: {eqprop.shape} != {bptt.shape}."
        )
    base._finite(eqprop, "EqProp gradient for BPTT comparison")
    base._finite(bptt, "BPTT reference gradient")
    eqprop_norm = float(torch.linalg.vector_norm(eqprop))
    bptt_norm = float(torch.linalg.vector_norm(bptt))
    dot = float(torch.dot(eqprop, bptt))
    denominator = eqprop_norm * bptt_norm
    difference_norm = float(torch.linalg.vector_norm(eqprop - bptt))
    return {
        "element_count": int(eqprop.numel()),
        "eqprop_l2": eqprop_norm,
        "bptt_l2": bptt_norm,
        "dot_product": dot,
        "cosine": dot / denominator if denominator > NORM_EPSILON else None,
        "difference_l2": difference_norm,
        "relative_l2_difference_over_bptt": (
            difference_norm / bptt_norm if bptt_norm > NORM_EPSILON else None
        ),
        "eqprop_over_bptt_norm_ratio": (
            eqprop_norm / bptt_norm if bptt_norm > NORM_EPSILON else None
        ),
        "symmetric_norm_delta": 2.0
        * abs(eqprop_norm - bptt_norm)
        / max(eqprop_norm + bptt_norm, NORM_EPSILON),
    }


def _native_bptt_weight_gradients(
    runtime: Mapping[str, Any],
    *,
    post_t_states: Sequence[torch.Tensor],
    post_t_hash: str,
    expected_k: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Run BPTT from post-T and retain only scored weights in the native dtype."""

    runtime_k = int(runtime["gradient_iterations"])
    minimizer_k = int(runtime["minimizer_gradient"].num_iterations)
    if runtime_k != int(expected_k) or minimizer_k != int(expected_k):
        raise RuntimeError(
            "BPTT runtime K differs from the selected EqProp K: "
            f"runtime={runtime_k}, minimizer={minimizer_k}, selected={expected_k}."
        )
    base._restore_states(runtime["free_layers"], post_t_states)
    start_hash = base._layer_state_sha256(runtime["free_layers"])
    if start_hash != post_t_hash:
        raise RuntimeError("BPTT did not start from the common post-T state.")

    all_gradients = runtime["backprop"].compute_gradient()
    if len(all_gradients) != len(runtime["parameters"]):
        raise RuntimeError("BPTT gradient/parameter count mismatch.")
    for parameter, gradient in zip(
        runtime["parameters"], all_gradients, strict=True
    ):
        name = str(parameter.name).strip()
        base._finite(gradient, f"BPTT gradient {name}")
        if gradient.shape != parameter.state.shape:
            raise RuntimeError(f"BPTT gradient shape mismatch for {name}.")

    gradients: dict[str, torch.Tensor] = {}
    scored_types: list[str] = []
    for index in runtime["weight_indices"]:
        parameter = runtime["parameters"][index]
        name = str(parameter.name).strip()
        gradient = all_gradients[index]
        gradients[name] = gradient.detach().cpu().clone()
        scored_types.append(parameter.__class__.__name__)
    if not gradients or any(value == "Bias" for value in scored_types):
        raise RuntimeError("BPTT weight-only inclusion contract failed.")
    if list(gradients) != list(runtime["weight_names"]):
        raise RuntimeError("BPTT scored-weight order differs from the runtime contract.")
    return gradients, {
        "bptt_start_state_sha256": start_hash,
        "bptt_end_state_sha256": base._layer_state_sha256(runtime["free_layers"]),
        "bptt_gradient_iterations_K": int(expected_k),
        "bptt_minimizer_iterations_K": minimizer_k,
        "bptt_all_parameter_gradient_count": len(all_gradients),
        "bptt_scored_weight_gradient_count": len(gradients),
        "bptt_bias_gradients_excluded": True,
    }


def _scaling_metadata(
    *,
    config: Mapping[str, Any],
    case: Mapping[str, Any],
    actual_beta: float,
    checkpoint_b0: float,
    scored_weight_count: int,
) -> dict[str, Any]:
    contract = config["gradient_contract"]
    architecture = str(case["architecture"])
    voltage = float(case.get("voltage_amp", case["voltage_amplification"]))
    current = float(case.get("current_amp", case["current_amplification"]))
    scored_count = int(case.get("scored_interaction_count_L", scored_weight_count))
    if scored_count != int(scored_weight_count):
        raise ValueError(
            f"Declared scored weight count={scored_count} differs from observed count={scored_weight_count}."
        )
    convention = str(
        contract.get(
            "amplification_exponent_convention", "scored_weight_interactions"
        )
    )
    if convention == "output_bias_current_row":
        declared_by_architecture = contract.get(
            "nudging_scale_power_by_architecture"
        )
        if not isinstance(declared_by_architecture, Mapping):
            raise ValueError("Missing explicit output-row nudging-scale powers.")
        depth = int(declared_by_architecture[architecture])
    elif convention == "scored_weight_interactions":
        depth = scored_count
    else:
        raise ValueError(
            f"Unsupported amplification exponent convention {convention!r}."
        )
    if voltage <= 0.0 or current <= 0.0 or depth <= 0:
        raise ValueError("Amplification values and depth must be positive.")
    factor = (voltage / current) ** depth
    effective = float(actual_beta) * factor
    baseline_map = contract.get(
        "baseline_init_output_curvature_scale_b0_by_architecture", {}
    )
    baseline_b0 = _optional_float(baseline_map.get(architecture))
    case_init_b0 = _optional_float(case.get("init_output_curvature_scale_b0"))
    reference_best_b0 = _optional_float(
        case.get("best_output_curvature_scale_b0_reference")
    )
    if baseline_b0 is None or baseline_b0 <= 0.0:
        raise ValueError(f"Missing positive baseline initialization b0 for {architecture}.")
    if case_init_b0 is None or case_init_b0 <= 0.0:
        raise ValueError("Missing positive per-case initialization b0.")
    if not math.isfinite(checkpoint_b0) or checkpoint_b0 <= 0.0:
        raise ValueError("Measured checkpoint b0 must be finite and positive.")
    return {
        "actual_beta": float(actual_beta),
        "amplification_depth_L": depth,
        "amplification_exponent_convention": convention,
        "scored_interaction_count": scored_count,
        "voltage_amplification": voltage,
        "current_amplification": current,
        "amplification_factor": factor,
        "effective_beta": effective,
        "beta_hat": effective / baseline_b0,
        "baseline_init_output_curvature_scale_b0": baseline_b0,
        "case_init_output_curvature_scale_b0": case_init_b0,
        "measured_checkpoint_output_curvature_scale_b0": checkpoint_b0,
        "reference_best_output_curvature_scale_b0": reference_best_b0,
        "actual_beta_over_case_init_b0": float(actual_beta) / case_init_b0,
        "actual_beta_over_checkpoint_b0": float(actual_beta) / checkpoint_b0,
        "scaling_formula": "effective_beta=actual_beta*(voltage_amp/current_amp)^L; beta_hat=effective_beta/b0_baseline_init",
    }


def _checkpoint_runtime(
    case: Mapping[str, Any],
    *,
    role: str,
    device: torch.device,
    gradient_iterations: int,
    nudging_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = base._build_runtime(
        case["source_config"],
        device=device,
        gradient_iterations=int(gradient_iterations),
        nudging_mode=nudging_mode,
        current_scale="auto",
    )
    initialization_hash = base._historical_initialization_sha256(
        runtime["parameters"]
    )
    if initialization_hash != case["reconstructed_initialization_tensor_sha256"]:
        raise ValueError(
            f"Initialization reconstruction mismatch for {case['scheme']}/{case['architecture']}."
        )
    if role == "best_validation":
        runtime["energy_fn"].load(case["best_checkpoint_path"])
        base._verify_npz_checkpoint(runtime, case["weights_best_path"])
    elif role != "reconstructed_initialization":
        raise ValueError(role)
    native_hash = base._parameter_state_sha256(runtime["parameters"])
    if any(parameter.state.dtype != torch.float32 for parameter in runtime["parameters"]):
        raise RuntimeError("Checkpoint must first be materialized in native float32.")
    return runtime, {
        "reconstructed_initialization_tensor_sha256": initialization_hash,
        "native_float32_parameter_state_sha256_before_cast": native_hash,
        "native_float32_parameter_numeric_sha256_before_cast": _numeric_parameter_sha256(
            runtime["parameters"]
        ),
        "checkpoint_role": role,
        "best_checkpoint_path": (
            str(case["best_checkpoint_path"]) if role == "best_validation" else None
        ),
        "best_checkpoint_sha256": (
            case["best_checkpoint_sha256"] if role == "best_validation" else None
        ),
    }


def _phase_diagnostic(
    runtime: Mapping[str, Any],
    *,
    phase: str,
    beta: float,
    states: Sequence[torch.Tensor],
    reference: Sequence[torch.Tensor],
) -> dict[str, Any]:
    base._restore_states(runtime["free_layers"], states)
    return {
        "phase": phase,
        "actual_beta": float(beta),
        "state_sha256": base._layer_state_sha256(runtime["free_layers"]),
        "all_states_finite": all(
            bool(torch.isfinite(layer.state).all()) for layer in runtime["free_layers"]
        ),
        **base._state_delta_metrics(runtime["free_layers"], reference),
        **base._phase_residual(runtime),
    }


def _iteration_contract(
    runtime: Mapping[str, Any], selected: Mapping[str, Any]
) -> dict[str, Any]:
    expected_t = int(selected["T"])
    expected_k = int(selected["K"])
    observed = {
        "runtime_inference_iterations_T": int(runtime["inference_iterations"]),
        "inference_minimizer_iterations_T": int(
            runtime["minimizer_inference"].num_iterations
        ),
        "runtime_gradient_iterations_K": int(runtime["gradient_iterations"]),
        "gradient_minimizer_iterations_K": int(
            runtime["minimizer_gradient"].num_iterations
        ),
        "augmented_minimizer_iterations_K": int(
            runtime["minimizer_augmented"].num_iterations
        ),
    }
    if any(
        value != expected_t
        for key, value in observed.items()
        if key.endswith("_T")
    ) or any(
        value != expected_k
        for key, value in observed.items()
        if key.endswith("_K")
    ):
        raise RuntimeError(
            f"Runtime iteration contract differs from T/K={expected_t}/{expected_k}: {observed}."
        )
    return {
        "expected_inference_iterations_T": expected_t,
        "expected_gradient_iterations_K": expected_k,
        **observed,
        "iteration_contract_passed": True,
    }


def _run_precision(
    *,
    case: Mapping[str, Any],
    selected: Mapping[str, Any],
    batch: Mapping[str, Any],
    precision: str,
    device: torch.device,
    residual_threshold: float,
    amplification_exponent: int,
    nudging_mode: str,
    eqprop_variant: str,
    capture_endpoint_states: bool = False,
) -> dict[str, Any]:
    dtype = {"float32": torch.float32, "float64": torch.float64}[precision]
    runtime, guard = _checkpoint_runtime(
        case,
        role=str(selected["checkpoint_role"]),
        device=device,
        gradient_iterations=int(selected["K"]),
        nudging_mode=nudging_mode,
    )
    output_layer = runtime["cost_fn"].layers()[-1]
    exact_output_power = max(
        extended.exact_runtime_layer_index(output_layer) - 1, 0
    )
    declared_output_power = int(amplification_exponent)
    if declared_output_power != exact_output_power:
        raise RuntimeError(
            "Float64 shadow amplification exponent does not match the exact "
            f"runtime output bias-current row: {declared_output_power} != {exact_output_power}."
        )
    voltage = float(case.get("voltage_amp", case["voltage_amplification"]))
    current = float(case.get("current_amp", case["current_amplification"]))
    exact_row_factor = float(
        extended._amplified_layer_row_scale(runtime["energy_fn"], output_layer)
    )
    if not math.isclose(
        exact_row_factor,
        (voltage / current) ** exact_output_power,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise RuntimeError("Exact runtime output-row current scale guard failed.")
    expected_runtime_scale = exact_row_factor if nudging_mode == "current" else 1.0
    if not math.isclose(
        float(runtime["nudging_current_scale"]),
        expected_runtime_scale,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise RuntimeError("Runtime nudge scale does not implement the declared mode.")
    guard.update(
        {
            "output_bias_current_row_exponent_L": exact_output_power,
            "output_row_current_scale": exact_row_factor,
            "nudging_mode": nudging_mode,
            "runtime_nudging_current_scale": float(
                runtime["nudging_current_scale"]
            ),
        }
    )
    dtype_proof = _convert_runtime_dtype(runtime, dtype)
    post_cast_hash = base._parameter_state_sha256(runtime["parameters"])
    post_cast_numeric_hash = _numeric_parameter_sha256(runtime["parameters"])
    conversion_preserved_values = bool(
        post_cast_numeric_hash
        == guard["native_float32_parameter_numeric_sha256_before_cast"]
    )
    native_bytes_preserved = bool(
        precision != "float32"
        or post_cast_hash
        == guard["native_float32_parameter_state_sha256_before_cast"]
    )
    if not conversion_preserved_values or not native_bytes_preserved:
        raise RuntimeError("Runtime dtype conversion changed checkpoint parameter values.")
    images = batch["images"].to(device=device, dtype=dtype)
    labels = batch["labels"].to(device=device)
    dtype_proof.update(_set_input_and_reset(runtime, images, dtype=dtype))
    runtime["minimizer_inference"].num_iterations = int(selected["T"])
    runtime["inference_iterations"] = int(selected["T"])
    runtime["gradient_iterations"] = int(selected["K"])
    guard.update(_iteration_contract(runtime, selected))
    runtime["minimizer_inference"].compute_equilibrium()
    runtime["cost_fn"].set_target(labels)
    post_t_states = base._clone_states(runtime["free_layers"])
    post_t_hash = base._layer_state_sha256(runtime["free_layers"])
    checkpoint_curvature = extended._output_curvature_diagnostic(runtime)

    residuals = extended._ResidualAccumulator()
    residuals.add_endpoint(
        runtime,
        gradient_source=runtime["energy_fn"],
        phase="post_T_free",
        beta=0.0,
        source_indices=batch["source_indices"],
    )

    eqprop_minimizer_k = int(runtime["minimizer_augmented"].num_iterations)
    if eqprop_minimizer_k != int(selected["K"]):
        raise RuntimeError(
            "EqProp minimizer K differs from the selected BPTT K: "
            f"{eqprop_minimizer_k} != {selected['K']}."
        )
    bptt_gradients, bptt_guard = _native_bptt_weight_gradients(
        runtime,
        post_t_states=post_t_states,
        post_t_hash=post_t_hash,
        expected_k=int(selected["K"]),
    )
    bptt_guard["eqprop_minimizer_iterations_K"] = eqprop_minimizer_k

    # Freeze the current-mode output force exactly once at the common post-T
    # state.  Every zero/negative/positive phase below reuses these exact
    # bytes, so a centered estimate cannot accidentally use two different
    # linearized objectives.
    base._restore_states(runtime["free_layers"], post_t_states)
    expected_force = None
    if nudging_mode == "current":
        expected_force = (
            -runtime["cost_fn"]._grad(output_layer, mean=False).detach().clone()
        )
    runtime["augmented_fn"].prepare_nudging()
    frozen_force = getattr(runtime["augmented_fn"]._nudging, "_force", None)
    if nudging_mode == "current":
        if frozen_force is None or expected_force is None:
            raise RuntimeError("Current nudging did not materialize a frozen force.")
        if not torch.equal(frozen_force, expected_force):
            raise RuntimeError("Frozen current force differs from -dC/dy at post-T.")
        frozen_force_sha256 = base._tensor_sha256(frozen_force)
    else:
        frozen_force_sha256 = None

    base._restore_states(runtime["free_layers"], post_t_states)
    zero_start_hash = base._layer_state_sha256(runtime["free_layers"])
    if zero_start_hash != post_t_hash:
        raise RuntimeError("Zero phase did not restore the common post-T state.")
    runtime["augmented_fn"].nudging = 0.0
    runtime["minimizer_augmented"].compute_equilibrium()
    zero_gradients = base._energy_gradients(runtime)
    zero_states = base._clone_states(runtime["free_layers"])
    zero_hash = base._layer_state_sha256(runtime["free_layers"])
    residuals.add_endpoint(
        runtime,
        gradient_source=runtime["augmented_fn"],
        phase="zero",
        beta=0.0,
        source_indices=batch["source_indices"],
    )
    zero_phase = _phase_diagnostic(
        runtime,
        phase="zero",
        beta=0.0,
        states=zero_states,
        reference=post_t_states,
    )

    actual_beta = float(selected["actual_beta"])
    native_beta_scalar = torch.as_tensor(actual_beta, dtype=dtype, device=device)
    dtype_proof.update(
        {
            "nudging_scalar_dtype": str(native_beta_scalar.dtype),
            "nudging_scalar_value_in_runtime_dtype": float(
                native_beta_scalar.detach().cpu()
            ),
            "declared_actual_beta": actual_beta,
        }
    )
    phase_gradients: dict[str, list[torch.Tensor]] = {}
    phase_states: dict[str, list[torch.Tensor]] = {}
    phase_hashes: dict[str, str] = {}
    phase_start_hashes: dict[str, str] = {}
    nudged_phase_rows: list[dict[str, Any]] = []
    for phase, multiplier in extended._nudged_phase_plan(eqprop_variant):
        base._restore_states(runtime["free_layers"], post_t_states)
        start_hash = base._layer_state_sha256(runtime["free_layers"])
        if start_hash != post_t_hash:
            raise RuntimeError(
                f"EqProp {phase} phase did not restore the common post-T state."
            )
        signed_beta = native_beta_scalar * torch.as_tensor(
            multiplier, dtype=dtype, device=device
        )
        runtime["augmented_fn"].nudging = signed_beta
        runtime["minimizer_augmented"].compute_equilibrium()
        endpoint_gradients = base._energy_gradients(runtime)
        endpoint_states = base._clone_states(runtime["free_layers"])
        if not all(bool(torch.isfinite(value).all()) for value in endpoint_states):
            raise FloatingPointError(f"Non-finite state in EqProp {phase} phase.")
        endpoint_hash = base._layer_state_sha256(runtime["free_layers"])
        residuals.add_endpoint(
            runtime,
            gradient_source=runtime["augmented_fn"],
            phase=phase,
            beta=float(multiplier) * actual_beta,
            source_indices=batch["source_indices"],
        )
        phase_gradients[phase] = endpoint_gradients
        phase_states[phase] = endpoint_states
        phase_hashes[phase] = endpoint_hash
        phase_start_hashes[phase] = start_hash
        nudged_phase_rows.append(
            _phase_diagnostic(
                runtime,
                phase=phase,
                beta=float(multiplier) * actual_beta,
                states=endpoint_states,
                reference=post_t_states,
            )
        )

    positive_gradients = phase_gradients["positive"]
    positive_states = phase_states["positive"]
    negative_gradients = phase_gradients.get("negative")
    negative_states = phase_states.get("negative")
    if nudging_mode == "current":
        final_force = getattr(runtime["augmented_fn"]._nudging, "_force", None)
        if final_force is None or base._tensor_sha256(final_force) != frozen_force_sha256:
            raise RuntimeError("The frozen current force changed across EqProp phases.")

    gradients: dict[str, torch.Tensor] = {}
    numerators: dict[str, torch.Tensor] = {}
    gradient_dtype_rows: list[dict[str, Any]] = []
    denominator_scale = (
        float(runtime["nudging_current_scale"])
        if nudging_mode == "current"
        else 1.0
    )
    for index in runtime["weight_indices"]:
        parameter = runtime["parameters"][index]
        name = str(parameter.name).strip()
        numerator, estimate = _native_eqprop_estimate(
            eqprop_variant=eqprop_variant,
            beta=actual_beta,
            denominator_scale=denominator_scale,
            zero=zero_gradients[index],
            negative=(
                negative_gradients[index]
                if negative_gradients is not None
                else None
            ),
            positive=positive_gradients[index],
        )
        numerators[name] = numerator.detach().cpu().clone()
        gradients[name] = estimate.detach().cpu().clone()
        bptt_gradient = bptt_gradients[name]
        gradient_dtype_rows.append(
            {
                "parameter_name": name,
                "parameter_type": parameter.__class__.__name__,
                "bptt_gradient_dtype": str(bptt_gradient.dtype),
                "zero_endpoint_gradient_dtype": str(zero_gradients[index].dtype),
                "negative_endpoint_gradient_dtype": (
                    str(negative_gradients[index].dtype)
                    if negative_gradients is not None
                    else None
                ),
                "positive_endpoint_gradient_dtype": str(positive_gradients[index].dtype),
                "finite_difference_numerator_dtype": str(numerator.dtype),
                "eqprop_gradient_dtype": str(estimate.dtype),
                "finite_difference_denominator_scale": denominator_scale,
                "finite_difference_denominator_multiplier": (
                    2 if eqprop_variant == "centered" else 1
                ),
                "finite_difference_denominator": actual_beta
                * denominator_scale
                * (2 if eqprop_variant == "centered" else 1),
                "bptt_l2": float(
                    torch.linalg.vector_norm(bptt_gradient.detach().to(torch.float64))
                ),
                "numerator_l2": float(
                    torch.linalg.vector_norm(numerator.detach().to(torch.float64))
                ),
                "zero_endpoint_l2": float(
                    torch.linalg.vector_norm(zero_gradients[index].detach().to(torch.float64))
                ),
                "negative_endpoint_l2": (
                    float(
                        torch.linalg.vector_norm(
                            negative_gradients[index].detach().to(torch.float64)
                        )
                    )
                    if negative_gradients is not None
                    else None
                ),
                "positive_endpoint_l2": float(
                    torch.linalg.vector_norm(positive_gradients[index].detach().to(torch.float64))
                ),
                "eqprop_l2": float(
                    torch.linalg.vector_norm(estimate.detach().to(torch.float64))
                ),
                "bptt_exact_zero_fraction": float(
                    (bptt_gradient == 0.0).to(torch.float64).mean()
                ),
                "exact_zero_fraction": float((estimate == 0.0).to(torch.float64).mean()),
            }
        )

    context = {
        "architecture": selected["architecture"],
        "scheme": selected["scheme"],
        "checkpoint_role": selected["checkpoint_role"],
        "T": int(selected["T"]),
        "K": int(selected["K"]),
        "source_native_T": int(selected["source_native_T"]),
        "source_native_K": int(selected["source_native_K"]),
        "native_context": bool(selected["native_context"]),
        "tk_source": str(selected["tk_source"]),
        "precision": precision,
        "eqprop_variant": eqprop_variant,
        "batch_index": int(batch["batch_index"]),
        "batch_payload_sha256": batch["payload_sha256"],
        "batch_source_indices_sha256": batch["source_indices_sha256"],
    }
    residual_rows = residuals.summary_rows(
        context=context,
        expected_examples=int(images.shape[0]),
        p90_threshold=float(residual_threshold),
    )
    residual_archive_records = residuals.archive_records(context=context)
    displacement_rows: list[dict[str, Any]] = []
    for phase, states in phase_states.items():
        for reference_kind, reference in (
            ("post_T_free", post_t_states),
            ("matched_zero_K", zero_states),
        ):
            for row in extended._state_displacement_rows(
                runtime["free_layers"],
                states,
                reference,
                reference_kind=reference_kind,
                reference_norm_epsilon=1.0e-30,
            ):
                displacement_rows.append(
                    {
                        **context,
                        "phase": phase,
                        "actual_beta": (
                            -actual_beta if phase == "negative" else actual_beta
                        ),
                        **row,
                    }
                )
    if negative_states is not None:
        for row in extended._state_displacement_rows(
            runtime["free_layers"],
            positive_states,
            negative_states,
            reference_kind="negative_phase",
            reference_norm_epsilon=1.0e-30,
        ):
            displacement_rows.append(
                {
                    **context,
                    "phase": "positive_minus_negative",
                    "actual_beta": actual_beta,
                    **row,
                }
            )

    final_hash = base._parameter_state_sha256(runtime["parameters"])
    final_numeric_hash = _numeric_parameter_sha256(runtime["parameters"])
    target_dtype = str(dtype)
    all_gradient_dtypes = all(
        row["bptt_gradient_dtype"] == target_dtype
        and row["zero_endpoint_gradient_dtype"] == target_dtype
        and (
            row["negative_endpoint_gradient_dtype"] == target_dtype
            if eqprop_variant == "centered"
            else row["negative_endpoint_gradient_dtype"] is None
        )
        and row["positive_endpoint_gradient_dtype"] == target_dtype
        and row["finite_difference_numerator_dtype"] == target_dtype
        and row["eqprop_gradient_dtype"] == target_dtype
        for row in gradient_dtype_rows
    )
    guard.update(
        {
            "architecture": selected["architecture"],
            "scheme": selected["scheme"],
            "checkpoint_role": selected["checkpoint_role"],
            "precision": precision,
            "target_dtype": target_dtype,
            "parameter_state_sha256_after_cast": post_cast_hash,
            "parameter_numeric_sha256_after_cast": post_cast_numeric_hash,
            "dtype_conversion_preserved_parameter_values": conversion_preserved_values,
            "float32_conversion_preserved_native_parameter_bytes": native_bytes_preserved,
            "parameter_state_sha256_after_phases": final_hash,
            "parameter_numeric_sha256_after_phases": final_numeric_hash,
            "parameter_tensors_unchanged_after_cast": final_hash == post_cast_hash,
            "parameter_values_unchanged_after_cast": final_numeric_hash
            == post_cast_numeric_hash,
            "all_gradient_arithmetic_target_dtype": all_gradient_dtypes,
            "common_post_T_state_sha256": post_t_hash,
            "zero_start_state_sha256": zero_start_hash,
            "negative_start_state_sha256": phase_start_hashes.get("negative"),
            "positive_start_state_sha256": phase_start_hashes["positive"],
            "zero_state_sha256": zero_hash,
            "negative_state_sha256": phase_hashes.get("negative"),
            "positive_state_sha256": phase_hashes["positive"],
            "eqprop_variant": eqprop_variant,
            "frozen_current_force_sha256": frozen_force_sha256,
            "frozen_current_force_matches_post_T_cost_gradient": (
                True if nudging_mode == "current" else None
            ),
            "frozen_current_force_unchanged_across_phases": (
                True if nudging_mode == "current" else None
            ),
            **bptt_guard,
        }
    )
    if not guard["parameter_tensors_unchanged_after_cast"]:
        raise RuntimeError("Parameter tensors changed during the read-only precision replay.")
    if not all_gradient_dtypes:
        raise RuntimeError("EqProp arithmetic did not remain in the requested runtime dtype.")
    result = {
        "gradients": gradients,
        "bptt_gradients": bptt_gradients,
        "numerators": numerators,
        "endpoint_gradients": {
            "zero": {
                str(runtime["parameters"][index].name).strip():
                zero_gradients[index].detach().cpu().clone()
                for index in runtime["weight_indices"]
            },
            **{
                phase: {
                    str(runtime["parameters"][index].name).strip():
                    values[index].detach().cpu().clone()
                    for index in runtime["weight_indices"]
                }
                for phase, values in phase_gradients.items()
            },
        },
        "gradient_dtype_rows": gradient_dtype_rows,
        "dtype_proof": dtype_proof,
        "guard": guard,
        "residual_rows": residual_rows,
        "residual_archive_records": residual_archive_records,
        "displacement_rows": displacement_rows,
        "phase_rows": [
            {**context, **zero_phase},
            *({**context, **row} for row in nudged_phase_rows),
        ],
        "checkpoint_curvature": checkpoint_curvature,
    }
    if capture_endpoint_states:
        result.update(
            {
                "captured_runtime": runtime,
                "captured_input_state": runtime["energy_fn"]
                .layers()[0]
                .state.detach()
                .cpu()
                .clone(),
                "captured_post_t_states": [
                    value.detach().cpu().clone() for value in post_t_states
                ],
                "captured_zero_states": [
                    value.detach().cpu().clone() for value in zero_states
                ],
                "captured_phase_states": {
                    phase: [value.detach().cpu().clone() for value in states]
                    for phase, states in phase_states.items()
                },
            }
        )
    return result


def _write_gradient_archive(
    path: Path,
    *,
    selected: Mapping[str, Any],
    precision_outputs: Mapping[str, Mapping[str, Any]],
    scaling: Mapping[str, Any],
) -> None:
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(
            json.dumps(
                {
                    **{
                        key: value
                        for key, value in selected.items()
                        if key not in {"source_selection_row"}
                    },
                    **dict(scaling),
                    "finite_difference_arithmetic": "native_runtime_dtype",
                    "bptt_reference": "same_post_T_state_exactly_K_iterations",
                    "scored_gradients": "weights_only_biases_excluded",
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
    }
    for precision, output in precision_outputs.items():
        for name, gradient in output["gradients"].items():
            arrays[f"eqprop_{precision}__{name}"] = gradient.numpy()
            arrays[f"bptt_{precision}__{name}"] = output["bptt_gradients"][
                name
            ].numpy()
            arrays[f"numerator_{precision}__{name}"] = output["numerators"][name].numpy()
        for phase, gradients in output.get("endpoint_gradients", {}).items():
            for name, gradient in gradients.items():
                arrays[f"endpoint_{phase}_{precision}__{name}"] = gradient.numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _completion_criteria_met(completion: Mapping[str, Any]) -> bool:
    required_false = ("optimizer_steps_applied", "official_test_read")
    return bool(
        all(
            bool(value)
            for key, value in completion.items()
            if key not in {"criteria_met", *required_false}
        )
        and all(completion.get(key) is False for key in required_false)
    )


def _write_report(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    selected_cases: Sequence[Mapping[str, Any]],
    source_run: Path,
    batch: Mapping[str, Any],
    eqprop_variant: str,
) -> None:
    summaries: list[tuple[Any, ...]] = []
    for selected in selected_cases:
        key = (
            str(selected["architecture"]),
            str(selected["scheme"]),
            str(selected["checkpoint_role"]),
        )
        layer_rows = [
            row
            for row in rows
            if (
                str(row["architecture"]),
                str(row["scheme"]),
                str(row["checkpoint_role"]),
            )
            == key
        ]
        float32_bptt_cosines = [
            float(row["float32_eqprop_vs_bptt_cosine"])
            for row in layer_rows
            if row["float32_eqprop_vs_bptt_cosine"] is not None
        ]
        float64_bptt_cosines = [
            float(row["float64_eqprop_vs_bptt_cosine"])
            for row in layer_rows
            if row["float64_eqprop_vs_bptt_cosine"] is not None
        ]
        summaries.append(
            (
                *key,
                f"{selected['source_native_T']}/{selected['source_native_K']}",
                f"{selected['T']}/{selected['K']}",
                str(selected["tk_source"]),
                str(selected["beta_source"]),
                f"{float(selected['actual_beta']):.6g}",
                f"{float(layer_rows[0]['effective_beta']):.6g}",
                f"{float(layer_rows[0]['beta_hat']):.6g}",
                (
                    "source_grid"
                    if bool(layer_rows[0]["source_grid_beta"])
                    else (
                        "cap_bypassed"
                        if bool(layer_rows[0]["source_grid_cap_bypassed"])
                        else "out_of_grid"
                    )
                ),
                str(layer_rows[0]["residual_limit_label"]),
                f"{sum(bool(row['precision_gate_passed']) for row in layer_rows)}/{len(layer_rows)}",
                "—"
                if any(row["cosine"] is None for row in layer_rows)
                else f"{min(float(row['cosine']) for row in layer_rows):.6f}",
                f"{max(float(row['symmetric_norm_delta']) for row in layer_rows):.6f}",
                (
                    f"{min(float32_bptt_cosines):.6f}"
                    if float32_bptt_cosines
                    else "—"
                ),
                f"{max(float(row['float32_eqprop_vs_bptt_symmetric_norm_delta']) for row in layer_rows):.6f}",
                (
                    f"{min(float64_bptt_cosines):.6f}"
                    if float64_bptt_cosines
                    else "—"
                ),
                f"{max(float(row['float64_eqprop_vs_bptt_symmetric_norm_delta']) for row in layer_rows):.6f}",
            )
        )
    lines = [
        f"# {extended._variant_title(eqprop_variant).title()} EqProp true-dtype audit",
        "",
        "Evidence class: `ordinary_mnist_learning_algorithm_gradient_diagnostic` (not paper-facing).",
        "",
        "The exact selected actual beta is replayed in independent real float32 and float64 runtimes. In each runtime, BPTT and every EqProp phase start from the identical hashed post-T state and use the same selected K. Current mode freezes one output force at post-T and reuses its exact bytes for zero and all signed phases. Every model layer, parameter, input, state reset, phase endpoint, BPTT gradient, and finite-difference subtraction uses the declared runtime dtype. Casting occurs only after each native-dtype gradient has been formed for comparison and reporting.",
        "",
        base._markdown_table(
            (
                "Architecture",
                "Scheme",
                "Checkpoint",
                "Source T/K",
                "Replay T/K",
                "T/K provenance",
                "Beta provenance",
                "Actual beta",
                "Injected beta",
                "Beta hat",
                "Beta contract",
                "Residual status",
                "Cross-precision passing layers",
                "Minimum cross-precision EqProp cosine",
                "Maximum cross-precision EqProp symmetric norm delta",
                "Minimum float32 EqProp/BPTT cosine",
                "Maximum float32 EqProp/BPTT symmetric norm delta",
                "Minimum float64 EqProp/BPTT cosine",
                "Maximum float64 EqProp/BPTT symmetric norm delta",
            ),
            summaries,
        ),
        "",
        f"The scientific precision gate is layerwise cosine >= {COSINE_MINIMUM:.2f} and symmetric norm delta <= {SYMMETRIC_NORM_DELTA_MAXIMUM:.2f}. A failed gate is a measured scientific outcome, not an operationally failed bundle.",
        "",
        "Beta scaling is recorded per case as `effective_beta = actual_beta * (voltage_amp/current_amp)^L`, with `L=max(layer_index(output)-1,0)`, exactly matching the output bias-current row exponent, and `beta_hat = effective_beta / b0_baseline_init`. The tables also report empirical `actual_beta/b0_case` at initialization and the replayed checkpoint.",
        "",
        f"Source run: `{source_run}`. Fixed batch index `{batch['batch_index']}`, payload SHA `{batch['payload_sha256']}`, source-index SHA `{batch['source_indices_sha256']}`.",
        "",
        "Artifacts include layerwise cross-precision EqProp metrics, separate float32 and float64 EqProp/BPTT cosine and norm metrics, exact projected-KKT residual summaries, per-layer and aggregate signed state displacement, phase diagnostics, dtype proofs, read-only source/parameter/force guards, and native-dtype endpoint/EqProp/BPTT gradient arrays.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    config = base._read_json(config_path)
    if config.get("schema_version") != "perfectdiode-conv-eqprop-bptt-beta-tk-displacement-replay/v1":
        raise ValueError("Unexpected source beta-run config schema.")
    source_eqprop_variant = extended._eqprop_variant(config["gradient_contract"])
    if source_eqprop_variant != "positive_one_sided":
        raise ValueError("The float64 shadow requires a positive one-sided EqProp source run.")
    eqprop_variant = (
        str(args.eqprop_variant)
        if args.eqprop_variant is not None
        else source_eqprop_variant
    )
    if eqprop_variant not in extended.EQPROP_VARIANTS:
        raise ValueError(f"Unsupported EqProp variant {eqprop_variant!r}.")
    nudging_mode = str(config["gradient_contract"].get("nudging_mode", "cost"))
    if nudging_mode not in {"cost", "current"}:
        raise ValueError(f"Unsupported nudging mode {nudging_mode!r}.")
    source_run = args.source_run.expanduser().resolve()
    validation_errors = base.validate_run(source_run)
    if validation_errors:
        raise ValueError(
            "Source run is not a valid completed canonical bundle: "
            + "; ".join(validation_errors)
        )
    source_status = base._read_json(source_run / "status.json")
    source_result = base._read_json(source_run / "result.json")
    source_manifest = base._read_json(source_run / "manifest.json")
    if source_status.get("state") != "complete" or bool(source_manifest.get("smoke")):
        raise ValueError("Source must be a completed non-smoke native-beta run.")
    source_resolved_config = base._read_json(source_run / "config.resolved.json")
    config_exact = _json_semantic_sha256(config) == _json_semantic_sha256(
        source_resolved_config
    )
    if not config_exact:
        raise ValueError(
            "--config is not semantically identical to the completed source run's config.resolved.json."
        )

    runtime_source = base._validate_runtime_source(config)
    inventory, source_hashes_before = extended._source_inventory(config)
    inventory_by_key = {
        (str(case["architecture"]), str(case["scheme"])): case
        for case in inventory
    }
    selection_path = _selection_csv(source_run, args.selection_csv)
    selection_rows = _read_csv(selection_path)
    specs = _requested_case_specs(args.case, all_18=bool(args.all_18))
    explicit_beta_label = str(args.explicit_beta_label).strip()
    if (
        not explicit_beta_label
        or "/" in explicit_beta_label
        or any(character.isspace() for character in explicit_beta_label)
    ):
        raise ValueError(
            "--explicit-beta-label must be a non-empty whitespace-free label without '/'."
        )
    selected_cases = _resolve_selected_cases(
        selection_rows,
        specs,
        explicit_beta_label=explicit_beta_label,
    )
    if eqprop_variant != source_eqprop_variant and any(
        not bool(row["beta_explicit_cli"]) for row in selected_cases
    ):
        raise ValueError(
            "An EqProp variant override requires an explicit beta for every case; "
            "the source one-sided beta selection is not a centered selection."
        )
    if eqprop_variant == "centered" and nudging_mode != "current":
        raise ValueError("This centered true-dtype extension is restricted to current nudging.")
    _apply_tk_override(selected_cases, inventory_by_key, args.tk_override)
    for selected in selected_cases:
        case = inventory_by_key[(selected["architecture"], selected["scheme"])]
        requested_actual_beta = float(selected["actual_beta"])
        resolved_beta_row = _resolve_beta_row(
            config=config,
            case=case,
            requested_actual_beta=requested_actual_beta,
            allow_out_of_grid=bool(args.allow_out_of_grid_beta),
        )
        selected["requested_actual_beta"] = requested_actual_beta
        selected["actual_beta"] = float(resolved_beta_row["beta"])
        selected["actual_beta_canonicalized_to_source_grid"] = bool(
            selected["actual_beta"] != requested_actual_beta
        )
        selected["resolved_beta_row"] = resolved_beta_row

    source_cohort = base._read_json(source_run / "cohort.json")
    batch_size = int(source_cohort["batch_size"])
    batches, materialized_cohort = base._build_validation_cohort(
        inventory[0]["source_config"],
        data_root=args.dataset_root.expanduser().resolve(),
        batch_size=batch_size,
        example_count=int(source_cohort["example_count"]),
    )
    expected_batch_guards = [
        {
            "batch_index": int(row["batch_index"]),
            "source_indices_sha256": row["source_indices_sha256"],
            "payload_sha256": row["payload_sha256"],
        }
        for row in source_cohort["batches"]
    ]
    observed_batch_guards = [
        {
            "batch_index": int(row["batch_index"]),
            "source_indices_sha256": row["source_indices_sha256"],
            "payload_sha256": row["payload_sha256"],
        }
        for row in batches
    ]
    cohort_exact = bool(
        observed_batch_guards == expected_batch_guards
        and materialized_cohort["validation_indices_sha256"]
        == source_cohort["validation_indices_sha256"]
        and materialized_cohort["cohort_sha256"] == source_cohort["cohort_sha256"]
        and int(materialized_cohort["example_count"])
        == int(source_cohort["example_count"])
        and int(materialized_cohort["batch_count"])
        == int(source_cohort["batch_count"])
    )
    if not cohort_exact:
        raise ValueError("Materialized validation cohort differs from the completed source run.")
    if args.batch_index < 0 or args.batch_index >= len(batches):
        raise ValueError(f"--batch-index must be in [0,{len(batches) - 1}].")
    batch = batches[int(args.batch_index)]
    residual_threshold = float(
        config.get("equilibrium_residual_contract", {}).get(
            "per_layer_sample_max_p90_threshold",
            config.get("equilibrium_residual_contract", {}).get("p90_threshold", 1.0e-2),
        )
    )
    if not math.isfinite(residual_threshold) or residual_threshold <= 0.0:
        raise ValueError("Residual p90 threshold must be finite and positive.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    output_root = args.output_root.expanduser().resolve()
    run_id = str(args.run_id)
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise ValueError(f"Invalid run id {run_id!r}.")
    run_dir = output_root / run_id
    source_files = {
        "result.json": base.sha256_file(source_run / "result.json"),
        "manifest.json": base.sha256_file(source_run / "manifest.json"),
        "config.resolved.json": base.sha256_file(source_run / "config.resolved.json"),
        "cohort.json": base.sha256_file(source_run / "cohort.json"),
        selection_path.name: base.sha256_file(selection_path),
    }
    if (source_run / "parameter_summary.csv").is_file():
        source_files["parameter_summary.csv"] = base.sha256_file(
            source_run / "parameter_summary.csv"
        )
    manifest = {
        "study_id": f"{config['study_id']}--float64-shadow",
        "run_id": run_id,
        "arm_id": f"conv_{eqprop_variant}_eqprop_float32_float64_shadow",
        "evidence_class": "ordinary_mnist_learning_algorithm_gradient_diagnostic",
        "dataset": config["dataset"]["name"],
        "smoke": bool(args.smoke),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "configuration": {
            "path": str(config_path),
            "sha256": base.sha256_file(config_path),
            "resolved": {
                "source_beta_config": config,
                "source_eqprop_variant": source_eqprop_variant,
                "eqprop_variant": eqprop_variant,
                "selected_cases": [
                    {key: value for key, value in row.items() if key != "source_selection_row"}
                    for row in selected_cases
                ],
                "batch_index": int(args.batch_index),
                "tk_override": (
                    [int(value) for value in args.tk_override]
                    if args.tk_override is not None
                    else None
                ),
                "allow_out_of_grid_beta": bool(args.allow_out_of_grid_beta),
                "source_injected_beta_cap": _optional_float(
                    config["gradient_contract"].get("actual_beta_cap")
                ),
                "precision_gate": {
                    "cosine_minimum": COSINE_MINIMUM,
                    "symmetric_norm_delta_maximum": SYMMETRIC_NORM_DELTA_MAXIMUM,
                    "symmetric_norm_delta_definition": "2*abs(norm32-norm64)/(norm32+norm64)",
                },
            },
        },
        "runtime": {
            **base.runtime_context(target=args.target),
            "device": str(device),
            "hostname": socket.gethostname(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "runtime_source": runtime_source,
        },
        "source_native_beta_run": {
            "path": str(source_run),
            "study_id": source_result["study_id"],
            "run_id": source_result["run_id"],
            "result_sha256": source_files["result.json"],
            "selection_path": str(selection_path),
            "selection_sha256": source_files[selection_path.name],
            "source_files": source_files,
        },
        "replay": {
            "precisions": list(PRECISIONS),
            "finite_difference_arithmetic": "native_runtime_dtype_before_diagnostic_cast",
            "bptt_reference": "same_post_T_state_exactly_selected_K_iterations",
            "source_eqprop_variant": source_eqprop_variant,
            "eqprop_variant": eqprop_variant,
            "eqprop_phase_initialization": (
                "restore_identical_post_T_before_zero_negative_and_positive"
                if eqprop_variant == "centered"
                else "restore_identical_post_T_before_zero_and_positive"
            ),
            "frozen_output_force": "prepared_once_at_post_T_and_reused_exactly",
            "fixed_batch": expected_batch_guards[int(args.batch_index)],
            "checkpoint_case_count": len(selected_cases),
            "official_test_read": False,
            "optimizer_constructed": False,
            "optimizer_steps_applied": False,
            "biases_active_in_dynamics": True,
            "bias_gradients_excluded": True,
            "nudging_mode": nudging_mode,
            "tk_source": (
                "explicit_cli_override"
                if args.tk_override is not None
                else "source_native"
            ),
            "out_of_grid_beta_allowed": bool(args.allow_out_of_grid_beta),
            "source_grid_cap_may_be_bypassed": bool(args.allow_out_of_grid_beta),
            "current_scale": "auto" if nudging_mode == "current" else 1.0,
            "finite_difference_denominator": (
                (
                    "2 * beta_actual * output_row_current_scale"
                    if eqprop_variant == "centered"
                    else "beta_actual * output_row_current_scale"
                )
                if nudging_mode == "current"
                else (
                    "2 * beta_actual"
                    if eqprop_variant == "centered"
                    else "beta_actual"
                )
            ),
        },
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.sha256_file(Path(__file__).resolve()),
            "extended_runner_path": str(EXTENDED_RUNNER),
            "extended_runner_sha256": base.sha256_file(EXTENDED_RUNNER),
            "repository_head": base._git_output(REPOSITORY_ROOT, "rev-parse", "HEAD"),
        },
    }
    base.start_run(run_dir, manifest)
    base._write_json(run_dir / "config.resolved.json", manifest["configuration"]["resolved"])
    base._write_json(run_dir / "source_cohort.json", source_cohort)
    base._write_json(run_dir / "materialized_cohort.json", materialized_cohort)
    base._write_json(
        run_dir / "selected_cases.json",
        {
            "selection_path": str(selection_path),
            "selection_sha256": source_files[selection_path.name],
            "cases": selected_cases,
        },
    )

    comparison_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    residual_archive_records: list[tuple[dict[str, Any], np.ndarray]] = []
    displacement_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    dtype_proofs: list[dict[str, Any]] = []
    parameter_guards: list[dict[str, Any]] = []
    scaling_rows: list[dict[str, Any]] = []
    completed_cases = 0
    bptt_tensor_count = 0
    try:
        for selected in selected_cases:
            key = (selected["architecture"], selected["scheme"])
            case = inventory_by_key[key]
            exponent_convention = str(
                config["gradient_contract"].get(
                    "amplification_exponent_convention",
                    "scored_weight_interactions",
                )
            )
            if exponent_convention == "output_bias_current_row":
                amplification_exponent = int(
                    config["gradient_contract"][
                        "nudging_scale_power_by_architecture"
                    ][selected["architecture"]]
                )
            elif exponent_convention == "scored_weight_interactions":
                amplification_exponent = int(case["scored_interaction_count_L"])
            else:
                raise ValueError(
                    f"Unsupported amplification exponent convention {exponent_convention!r}."
                )
            outputs: dict[str, dict[str, Any]] = {}
            for precision in PRECISIONS:
                outputs[precision] = _run_precision(
                    case=case,
                    selected=selected,
                    batch=batch,
                    precision=precision,
                    device=device,
                    residual_threshold=residual_threshold,
                    amplification_exponent=amplification_exponent,
                    nudging_mode=nudging_mode,
                    eqprop_variant=eqprop_variant,
                )
                residual_rows.extend(outputs[precision]["residual_rows"])
                residual_archive_records.extend(
                    outputs[precision]["residual_archive_records"]
                )
                displacement_rows.extend(outputs[precision]["displacement_rows"])
                phase_rows.extend(outputs[precision]["phase_rows"])
                dtype_proofs.append(
                    {
                        "architecture": selected["architecture"],
                        "scheme": selected["scheme"],
                        "checkpoint_role": selected["checkpoint_role"],
                        "eqprop_variant": eqprop_variant,
                        "T": int(selected["T"]),
                        "K": int(selected["K"]),
                        "source_native_T": int(selected["source_native_T"]),
                        "source_native_K": int(selected["source_native_K"]),
                        "native_context": bool(selected["native_context"]),
                        "tk_source": str(selected["tk_source"]),
                        "precision": precision,
                        **outputs[precision]["dtype_proof"],
                        "gradient_arithmetic": outputs[precision]["gradient_dtype_rows"],
                    }
                )
                parameter_guards.append(outputs[precision]["guard"])

            names32 = list(outputs["float32"]["gradients"])
            names64 = list(outputs["float64"]["gradients"])
            if names32 != names64 or not names32:
                raise RuntimeError("Float32 and float64 scored weight schemas differ.")
            for precision in PRECISIONS:
                bptt_names = list(outputs[precision]["bptt_gradients"])
                if bptt_names != names32:
                    raise RuntimeError(
                        f"{precision} BPTT and EqProp scored weight schemas differ."
                    )
                bptt_tensor_count += len(bptt_names)
            native_source_hashes = {
                outputs[precision]["guard"][
                    "native_float32_parameter_state_sha256_before_cast"
                ]
                for precision in PRECISIONS
            }
            native_numeric_hashes = {
                outputs[precision]["guard"][
                    "native_float32_parameter_numeric_sha256_before_cast"
                ]
                for precision in PRECISIONS
            }
            if len(native_source_hashes) != 1 or len(native_numeric_hashes) != 1:
                raise RuntimeError("Independent precision runtimes did not load identical source parameters.")
            residual_pass_by_precision = {
                precision: bool(
                    outputs[precision]["residual_rows"]
                    and all(
                        bool(row["gate_passed"])
                        for row in outputs[precision]["residual_rows"]
                    )
                )
                for precision in PRECISIONS
            }
            residual_limited = not all(residual_pass_by_precision.values())
            residual_limit_label = (
                "residual_limited" if residual_limited else "residual_valid"
            )
            checkpoint_b0_32 = float(
                outputs["float32"]["checkpoint_curvature"]["output_curvature_scale_b0"]
            )
            checkpoint_b0_64 = float(
                outputs["float64"]["checkpoint_curvature"]["output_curvature_scale_b0"]
            )
            b0_relative_delta = abs(checkpoint_b0_32 - checkpoint_b0_64) / max(
                abs(checkpoint_b0_32), abs(checkpoint_b0_64), NORM_EPSILON
            )
            if b0_relative_delta > 1.0e-6:
                raise RuntimeError(
                    f"Output curvature scale changed materially across precision: {b0_relative_delta}."
                )
            scaling = _scaling_metadata(
                config=config,
                case=case,
                actual_beta=float(selected["actual_beta"]),
                checkpoint_b0=checkpoint_b0_64,
                scored_weight_count=len(names32),
            )
            resolved_beta_row = selected["resolved_beta_row"]
            if not math.isclose(
                float(resolved_beta_row["beta_effective"]),
                float(scaling["effective_beta"]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ) or not math.isclose(
                float(resolved_beta_row["beta_hat"]),
                float(scaling["beta_hat"]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise ValueError("Resolved beta grid disagrees with shadow beta scaling.")
            scaling.update(
                {
                    "architecture": selected["architecture"],
                    "scheme": selected["scheme"],
                    "checkpoint_role": selected["checkpoint_role"],
                    "eqprop_variant": eqprop_variant,
                    "T": int(selected["T"]),
                    "K": int(selected["K"]),
                    "source_native_T": int(selected["source_native_T"]),
                    "source_native_K": int(selected["source_native_K"]),
                    "native_context": bool(selected["native_context"]),
                    "tk_source": str(selected["tk_source"]),
                    "beta_source": selected["beta_source"],
                    "requested_actual_beta": selected[
                        "requested_actual_beta"
                    ],
                    "actual_beta_canonicalized_to_source_grid": selected[
                        "actual_beta_canonicalized_to_source_grid"
                    ],
                    "float32_checkpoint_b0": checkpoint_b0_32,
                    "float64_checkpoint_b0": checkpoint_b0_64,
                    "checkpoint_b0_relative_delta": b0_relative_delta,
                    "beta_hat_requested": float(
                        resolved_beta_row["beta_hat_requested"]
                    ),
                    "beta_capped": bool(resolved_beta_row["capped"]),
                    "actual_beta_cap": resolved_beta_row.get("actual_beta_cap"),
                    "source_grid_beta": bool(
                        resolved_beta_row["source_grid_member"]
                    ),
                    "source_grid_cap_bypassed": bool(
                        resolved_beta_row["source_grid_cap_bypassed"]
                    ),
                    "float32_all_endpoint_residual_gates_passed": residual_pass_by_precision[
                        "float32"
                    ],
                    "float64_all_endpoint_residual_gates_passed": residual_pass_by_precision[
                        "float64"
                    ],
                    "residual_limited": residual_limited,
                    "residual_limit_label": residual_limit_label,
                }
            )
            source_row = selected["source_selection_row"]
            for source_field, observed_field in (
                ("selected_beta_hat_requested", "beta_hat_requested"),
                ("selected_beta_hat", "beta_hat"),
                ("selected_beta_effective", "effective_beta"),
                (
                    "selected_beta_over_case_init_b0",
                    "actual_beta_over_case_init_b0",
                ),
                (
                    "selected_beta_over_checkpoint_b0",
                    "actual_beta_over_checkpoint_b0",
                ),
            ):
                declared = _optional_float(source_row.get(source_field))
                scaling[f"source_{source_field}"] = declared
                comparison_rel_tol = (
                    1.0e-6
                    if source_field == "selected_beta_over_checkpoint_b0"
                    else 1.0e-9
                )
                if (
                    selected["beta_source"] == "stage_selected"
                    and declared is not None
                    and not math.isclose(
                        declared,
                        float(scaling[observed_field]),
                        rel_tol=comparison_rel_tol,
                        abs_tol=1.0e-15,
                    )
                ):
                    raise ValueError(
                        f"Source {source_field} disagrees with recomputed amplification scaling."
                    )
            scaling_rows.append(scaling)

            for name in names32:
                metrics = _precision_metrics(
                    outputs["float32"]["gradients"][name],
                    outputs["float64"]["gradients"][name],
                )
                eqprop_bptt32 = _eqprop_bptt_metrics(
                    eqprop_gradient=outputs["float32"]["gradients"][name],
                    bptt_gradient=outputs["float32"]["bptt_gradients"][name],
                )
                eqprop_bptt64 = _eqprop_bptt_metrics(
                    eqprop_gradient=outputs["float64"]["gradients"][name],
                    bptt_gradient=outputs["float64"]["bptt_gradients"][name],
                )
                dtype32 = next(
                    row
                    for row in outputs["float32"]["gradient_dtype_rows"]
                    if row["parameter_name"] == name
                )
                dtype64 = next(
                    row
                    for row in outputs["float64"]["gradient_dtype_rows"]
                    if row["parameter_name"] == name
                )
                comparison_outcome = (
                    "residual_limited"
                    if residual_limited
                    else (
                        "precision_agreement_passed"
                        if metrics["precision_gate_passed"]
                        else "precision_agreement_failed"
                    )
                )
                comparison_rows.append(
                    {
                        "schema": SCHEMA,
                        "architecture": selected["architecture"],
                        "scheme": selected["scheme"],
                        "checkpoint_role": selected["checkpoint_role"],
                        "eqprop_variant": eqprop_variant,
                        "T": int(selected["T"]),
                        "K": int(selected["K"]),
                        "source_native_T": int(selected["source_native_T"]),
                        "source_native_K": int(selected["source_native_K"]),
                        "native_context": bool(selected["native_context"]),
                        "tk_source": str(selected["tk_source"]),
                        "batch_index": int(batch["batch_index"]),
                        "batch_payload_sha256": batch["payload_sha256"],
                        "parameter_name": name,
                        "parameter_type": next(
                            row["parameter_type"]
                            for row in outputs["float32"]["gradient_dtype_rows"]
                            if row["parameter_name"] == name
                        ),
                        "bias_excluded": True,
                        "actual_beta": scaling["actual_beta"],
                        "requested_actual_beta": scaling[
                            "requested_actual_beta"
                        ],
                        "effective_beta": scaling["effective_beta"],
                        "beta_hat": scaling["beta_hat"],
                        "beta_hat_requested": scaling["beta_hat_requested"],
                        "beta_capped": scaling["beta_capped"],
                        "source_grid_beta": scaling["source_grid_beta"],
                        "source_grid_cap_bypassed": scaling[
                            "source_grid_cap_bypassed"
                        ],
                        "beta_source": scaling["beta_source"],
                        "residual_limited": scaling["residual_limited"],
                        "residual_limit_label": scaling[
                            "residual_limit_label"
                        ],
                        "precision_gate_interpretable": not scaling[
                            "residual_limited"
                        ],
                        "outcome": comparison_outcome,
                        "amplification_depth_L": scaling["amplification_depth_L"],
                        "amplification_factor": scaling["amplification_factor"],
                        "actual_beta_over_case_init_b0": scaling[
                            "actual_beta_over_case_init_b0"
                        ],
                        "actual_beta_over_checkpoint_b0": scaling[
                            "actual_beta_over_checkpoint_b0"
                        ],
                        "float32_eqprop_gradient_dtype": dtype32[
                            "eqprop_gradient_dtype"
                        ],
                        "float64_eqprop_gradient_dtype": dtype64[
                            "eqprop_gradient_dtype"
                        ],
                        "float32_bptt_gradient_dtype": dtype32[
                            "bptt_gradient_dtype"
                        ],
                        "float64_bptt_gradient_dtype": dtype64[
                            "bptt_gradient_dtype"
                        ],
                        "float32_numerator_l2": dtype32["numerator_l2"],
                        "float64_numerator_l2": dtype64["numerator_l2"],
                        "float32_exact_zero_fraction": dtype32[
                            "exact_zero_fraction"
                        ],
                        "float64_exact_zero_fraction": dtype64[
                            "exact_zero_fraction"
                        ],
                        "float32_bptt_exact_zero_fraction": dtype32[
                            "bptt_exact_zero_fraction"
                        ],
                        "float64_bptt_exact_zero_fraction": dtype64[
                            "bptt_exact_zero_fraction"
                        ],
                        "float32_eqprop_vs_bptt_element_count": eqprop_bptt32[
                            "element_count"
                        ],
                        "float32_bptt_l2": eqprop_bptt32["bptt_l2"],
                        "float32_eqprop_l2": eqprop_bptt32["eqprop_l2"],
                        "float32_eqprop_vs_bptt_dot_product": eqprop_bptt32[
                            "dot_product"
                        ],
                        "float32_eqprop_vs_bptt_cosine": eqprop_bptt32[
                            "cosine"
                        ],
                        "float32_eqprop_vs_bptt_difference_l2": eqprop_bptt32[
                            "difference_l2"
                        ],
                        "float32_eqprop_vs_bptt_relative_l2_difference_over_bptt": eqprop_bptt32[
                            "relative_l2_difference_over_bptt"
                        ],
                        "float32_eqprop_over_bptt_norm_ratio": eqprop_bptt32[
                            "eqprop_over_bptt_norm_ratio"
                        ],
                        "float32_eqprop_vs_bptt_symmetric_norm_delta": eqprop_bptt32[
                            "symmetric_norm_delta"
                        ],
                        "float64_eqprop_vs_bptt_element_count": eqprop_bptt64[
                            "element_count"
                        ],
                        "float64_bptt_l2": eqprop_bptt64["bptt_l2"],
                        "float64_eqprop_l2": eqprop_bptt64["eqprop_l2"],
                        "float64_eqprop_vs_bptt_dot_product": eqprop_bptt64[
                            "dot_product"
                        ],
                        "float64_eqprop_vs_bptt_cosine": eqprop_bptt64[
                            "cosine"
                        ],
                        "float64_eqprop_vs_bptt_difference_l2": eqprop_bptt64[
                            "difference_l2"
                        ],
                        "float64_eqprop_vs_bptt_relative_l2_difference_over_bptt": eqprop_bptt64[
                            "relative_l2_difference_over_bptt"
                        ],
                        "float64_eqprop_over_bptt_norm_ratio": eqprop_bptt64[
                            "eqprop_over_bptt_norm_ratio"
                        ],
                        "float64_eqprop_vs_bptt_symmetric_norm_delta": eqprop_bptt64[
                            "symmetric_norm_delta"
                        ],
                        **metrics,
                    }
                )
            archive_name = "__".join(
                (
                    str(selected["architecture"]),
                    str(selected["scheme"]),
                    str(selected["checkpoint_role"]),
                )
            ) + ".npz"
            _write_gradient_archive(
                run_dir / "artifacts/eqprop_gradients" / archive_name,
                selected=selected,
                precision_outputs=outputs,
                scaling=scaling,
            )
            completed_cases += 1
            case_comparisons = [
                row
                for row in comparison_rows
                if row["architecture"] == selected["architecture"]
                and row["scheme"] == selected["scheme"]
                and row["checkpoint_role"] == selected["checkpoint_role"]
            ]
            float32_bptt_cosines = [
                float(row["float32_eqprop_vs_bptt_cosine"])
                for row in case_comparisons
                if row["float32_eqprop_vs_bptt_cosine"] is not None
            ]
            float64_bptt_cosines = [
                float(row["float64_eqprop_vs_bptt_cosine"])
                for row in case_comparisons
                if row["float64_eqprop_vs_bptt_cosine"] is not None
            ]
            base.append_metric(
                run_dir / "metrics.jsonl",
                {
                    "stage": "precision_case_complete",
                    "split": "validation",
                    "architecture": selected["architecture"],
                    "scheme": selected["scheme"],
                    "checkpoint_role": selected["checkpoint_role"],
                    "eqprop_variant": eqprop_variant,
                    "actual_beta": float(selected["actual_beta"]),
                    "scored_weight_layers": len(names32),
                    "passing_weight_layers": sum(
                        bool(row["precision_gate_passed"])
                        for row in case_comparisons
                    ),
                    "minimum_float32_eqprop_vs_bptt_cosine": (
                        min(float32_bptt_cosines)
                        if float32_bptt_cosines
                        else None
                    ),
                    "maximum_float32_eqprop_vs_bptt_symmetric_norm_delta": max(
                        float(
                            row[
                                "float32_eqprop_vs_bptt_symmetric_norm_delta"
                            ]
                        )
                        for row in case_comparisons
                    ),
                    "minimum_float64_eqprop_vs_bptt_cosine": (
                        min(float64_bptt_cosines)
                        if float64_bptt_cosines
                        else None
                    ),
                    "maximum_float64_eqprop_vs_bptt_symmetric_norm_delta": max(
                        float(
                            row[
                                "float64_eqprop_vs_bptt_symmetric_norm_delta"
                            ]
                        )
                        for row in case_comparisons
                    ),
                },
            )
            base.update_status_progress(
                run_dir,
                {
                    "stage": "float64_shadow_replay",
                    "completed_cases": completed_cases,
                    "total_cases": len(selected_cases),
                    "architecture": selected["architecture"],
                    "scheme": selected["scheme"],
                    "checkpoint_role": selected["checkpoint_role"],
                },
            )

        source_hashes_after = extended._verify_source_hashes(
            inventory, source_hashes_before
        )
        source_bundle_files_after = {
            name: base.sha256_file(source_run / name)
            for name in (
                "result.json",
                "manifest.json",
                "config.resolved.json",
                "cohort.json",
            )
        }
        if "parameter_summary.csv" in source_files:
            source_bundle_files_after["parameter_summary.csv"] = base.sha256_file(
                source_run / "parameter_summary.csv"
            )
        source_bundle_files_after[selection_path.name] = base.sha256_file(selection_path)
        source_bundle_unchanged = source_bundle_files_after == source_files
        all_parameter_guards = all(
            bool(row["parameter_tensors_unchanged_after_cast"])
            and bool(row["parameter_values_unchanged_after_cast"])
            and bool(row["dtype_conversion_preserved_parameter_values"])
            and bool(row["float32_conversion_preserved_native_parameter_bytes"])
            and bool(row["all_gradient_arithmetic_target_dtype"])
            for row in parameter_guards
        )
        all_bptt_phase_guards = all(
            row["bptt_start_state_sha256"]
            == row["common_post_T_state_sha256"]
            and row["zero_start_state_sha256"]
            == row["common_post_T_state_sha256"]
            and row["positive_start_state_sha256"]
            == row["common_post_T_state_sha256"]
            and (
                row["negative_start_state_sha256"]
                == row["common_post_T_state_sha256"]
                if eqprop_variant == "centered"
                else row["negative_start_state_sha256"] is None
            )
            and (
                bool(row["frozen_current_force_matches_post_T_cost_gradient"])
                and bool(row["frozen_current_force_unchanged_across_phases"])
                if nudging_mode == "current"
                else True
            )
            and int(row["bptt_gradient_iterations_K"]) > 0
            and int(row["bptt_minimizer_iterations_K"])
            == int(row["bptt_gradient_iterations_K"])
            and int(row["eqprop_minimizer_iterations_K"])
            == int(row["bptt_gradient_iterations_K"])
            and int(row["bptt_scored_weight_gradient_count"]) > 0
            and bool(row["bptt_bias_gradients_excluded"])
            for row in parameter_guards
        )
        all_iteration_guards = all(
            bool(row["iteration_contract_passed"])
            and int(row["runtime_inference_iterations_T"])
            == int(row["expected_inference_iterations_T"])
            and int(row["inference_minimizer_iterations_T"])
            == int(row["expected_inference_iterations_T"])
            and int(row["runtime_gradient_iterations_K"])
            == int(row["expected_gradient_iterations_K"])
            and int(row["gradient_minimizer_iterations_K"])
            == int(row["expected_gradient_iterations_K"])
            and int(row["augmented_minimizer_iterations_K"])
            == int(row["expected_gradient_iterations_K"])
            for row in parameter_guards
        )
        all_dtype_proofs = all(
            bool(row["all_runtime_variables_target_dtype"])
            and bool(row["all_runtime_variables_target_dtype_after_reset"])
            for row in dtype_proofs
        )
        scientific_all_passed = all(
            bool(row["precision_gate_passed"]) for row in comparison_rows
        )
        guards = {
            "source_config_semantically_exact": config_exact,
            "source_bundle_validation_errors": validation_errors,
            "source_bundle_files_before": source_files,
            "source_bundle_files_after": source_bundle_files_after,
            "source_bundle_files_unchanged": source_bundle_unchanged,
            "source_checkpoint_hashes_before": {
                "/".join(key): value for key, value in source_hashes_before.items()
            },
            "source_checkpoint_hashes_after": source_hashes_after,
            "source_checkpoint_bytes_unchanged": all(
                source_hashes_before[key] == source_hashes_after["/".join(key)]
                for key in source_hashes_before
            ),
            "cohort_exact": cohort_exact,
            "fixed_batch_guard": expected_batch_guards[int(args.batch_index)],
            "parameter_guards": parameter_guards,
            "all_parameter_and_arithmetic_guards_passed": all_parameter_guards,
            "all_iteration_contracts_passed": all_iteration_guards,
            "dtype_proofs": dtype_proofs,
            "all_dtype_proofs_passed": all_dtype_proofs,
            "bptt_replay": {
                "all_bptt_and_configured_eqprop_phases_started_from_common_post_T": all_bptt_phase_guards,
                "native_bptt_weight_tensor_count": bptt_tensor_count,
                "bias_gradients_excluded": True,
            },
            "precision_gate": {
                "cosine_minimum": COSINE_MINIMUM,
                "symmetric_norm_delta_maximum": SYMMETRIC_NORM_DELTA_MAXIMUM,
                "comparison_count": len(comparison_rows),
                "pass_count": sum(
                    bool(row["precision_gate_passed"]) for row in comparison_rows
                ),
                "fail_count": sum(
                    not bool(row["precision_gate_passed"]) for row in comparison_rows
                ),
                "all_passed": scientific_all_passed,
            },
            "biases_active_in_dynamics": True,
            "bias_gradients_excluded": True,
            "optimizer_constructed": False,
            "optimizer_steps_applied": False,
            "official_test_read": False,
        }
        base._write_csv(run_dir / "parameter_precision_comparison.csv", comparison_rows)
        base._write_csv(run_dir / "beta_scaling.csv", scaling_rows)
        base._write_csv(run_dir / "equilibrium_residual_summary.csv", residual_rows)
        extended._write_residual_archive(
            run_dir / "artifacts/equilibrium_residual_per_sample.npz",
            residual_archive_records,
        )
        base._write_csv(run_dir / "state_displacement.csv", displacement_rows)
        base._write_csv(run_dir / "phase_diagnostics.csv", phase_rows)
        base._write_json(run_dir / "dtype_proofs.json", {"cases": dtype_proofs})
        base._write_json(run_dir / "read_only_guards.json", guards)
        _write_report(
            run_dir / "report.md",
            rows=comparison_rows,
            selected_cases=selected_cases,
            source_run=source_run,
            batch=batch,
            eqprop_variant=eqprop_variant,
        )
        float32_bptt_cosines_all = [
            float(row["float32_eqprop_vs_bptt_cosine"])
            for row in comparison_rows
            if row["float32_eqprop_vs_bptt_cosine"] is not None
        ]
        float64_bptt_cosines_all = [
            float(row["float64_eqprop_vs_bptt_cosine"])
            for row in comparison_rows
            if row["float64_eqprop_vs_bptt_cosine"] is not None
        ]
        base.append_metric(
            run_dir / "metrics.jsonl",
            {
                "stage": "analysis_complete",
                "split": "validation",
                "checkpoint_cases": len(selected_cases),
                "eqprop_variant": eqprop_variant,
                "precision_comparisons": len(comparison_rows),
                "precision_gate_pass_count": guards["precision_gate"]["pass_count"],
                "precision_gate_fail_count": guards["precision_gate"]["fail_count"],
                "minimum_float32_eqprop_vs_bptt_cosine": (
                    min(float32_bptt_cosines_all)
                    if float32_bptt_cosines_all
                    else None
                ),
                "maximum_float32_eqprop_vs_bptt_symmetric_norm_delta": max(
                    float(row["float32_eqprop_vs_bptt_symmetric_norm_delta"])
                    for row in comparison_rows
                ),
                "minimum_float64_eqprop_vs_bptt_cosine": (
                    min(float64_bptt_cosines_all)
                    if float64_bptt_cosines_all
                    else None
                ),
                "maximum_float64_eqprop_vs_bptt_symmetric_norm_delta": max(
                    float(row["float64_eqprop_vs_bptt_symmetric_norm_delta"])
                    for row in comparison_rows
                ),
            },
        )
        expected_comparisons = sum(
            int(case["scored_interaction_count_L"])
            for selected in selected_cases
            for case in [
                inventory_by_key[(selected["architecture"], selected["scheme"])]
            ]
        )
        expected_bptt_tensors = expected_comparisons * len(PRECISIONS)
        nudged_phase_count = len(extended._nudged_phase_plan(eqprop_variant))
        expected_residual_rows = sum(
            len(PRECISIONS)
            * (2 + nudged_phase_count)
            * int(case["scored_interaction_count_L"])
            for selected in selected_cases
            for case in [
                inventory_by_key[(selected["architecture"], selected["scheme"])]
            ]
        )
        displacement_reference_count = (
            2 * nudged_phase_count + (1 if eqprop_variant == "centered" else 0)
        )
        expected_displacement_rows = sum(
            len(PRECISIONS)
            * displacement_reference_count
            * (int(case["scored_interaction_count_L"]) + 1)
            for selected in selected_cases
            for case in [
                inventory_by_key[(selected["architecture"], selected["scheme"])]
            ]
        )
        expected_phase_rows = (
            len(selected_cases) * len(PRECISIONS) * (1 + nudged_phase_count)
        )
        completion = {
            "criteria_met": True,
            "completed_all_selected_cases": completed_cases == len(selected_cases),
            "all_weight_layer_precision_comparisons_present": len(comparison_rows)
            == expected_comparisons,
            "scientific_precision_gate_outcome_recorded": len(comparison_rows) > 0,
            "all_runtime_dtypes_proven": all_dtype_proofs,
            "all_iteration_contracts_passed": all_iteration_guards,
            "all_finite_differences_computed_in_native_dtype": all(
                row["float32_eqprop_gradient_dtype"] == "torch.float32"
                and row["float64_eqprop_gradient_dtype"] == "torch.float64"
                for row in comparison_rows
            ),
            "all_bptt_gradients_computed_in_native_dtype": all(
                row["float32_bptt_gradient_dtype"] == "torch.float32"
                and row["float64_bptt_gradient_dtype"] == "torch.float64"
                for row in comparison_rows
            ),
            "all_scored_bptt_weight_tensors_archived": bptt_tensor_count
            == expected_bptt_tensors,
            "all_eqprop_vs_bptt_layer_metrics_present": all(
                int(row["float32_eqprop_vs_bptt_element_count"])
                == int(row["element_count"])
                and int(row["float64_eqprop_vs_bptt_element_count"])
                == int(row["element_count"])
                for row in comparison_rows
            ),
            "bptt_and_configured_eqprop_phases_started_from_identical_post_T": all_bptt_phase_guards,
            "amplification_scaled_beta_metadata_recorded": len(scaling_rows)
            == len(selected_cases),
            "exact_residuals_recorded_for_both_precisions": len(residual_rows)
            == expected_residual_rows
            and all(bool(row["coverage_complete"]) for row in residual_rows),
            "displacement_recorded_for_both_precisions": len(displacement_rows)
            == expected_displacement_rows
            and all(str(row["outcome"]) == "ok" for row in displacement_rows),
            "phase_diagnostics_recorded_for_both_precisions": len(phase_rows)
            == expected_phase_rows,
            "source_bundle_files_unchanged": source_bundle_unchanged,
            "source_checkpoint_bytes_unchanged": guards[
                "source_checkpoint_bytes_unchanged"
            ],
            "parameter_tensors_unchanged": all_parameter_guards,
            "fixed_cohort_and_batch_reproduced": cohort_exact,
            "bias_gradients_excluded": True,
            "optimizer_steps_applied": False,
            "official_test_read": False,
        }
        completion["criteria_met"] = _completion_criteria_met(completion)
        if not completion["criteria_met"]:
            raise RuntimeError(f"Completion criteria failed: {completion}")
        result = base.complete_run(
            run_dir,
            terminal_metrics={
                "checkpoint_cases": len(selected_cases),
                "eqprop_variant": eqprop_variant,
                "batch_index": int(args.batch_index),
                "batch_examples": int(batch["images"].shape[0]),
                "precision_comparisons": len(comparison_rows),
                "precision_gate_pass_count": guards["precision_gate"]["pass_count"],
                "precision_gate_fail_count": guards["precision_gate"]["fail_count"],
                "precision_gate_all_passed": scientific_all_passed,
                "residual_limited_case_count": sum(
                    bool(row["residual_limited"]) for row in scaling_rows
                ),
                "minimum_precision_cosine": (
                    min(
                        float(row["cosine"])
                        for row in comparison_rows
                        if row["cosine"] is not None
                    )
                    if any(row["cosine"] is not None for row in comparison_rows)
                    else None
                ),
                "maximum_symmetric_norm_delta": max(
                    float(row["symmetric_norm_delta"]) for row in comparison_rows
                ),
                "minimum_float32_eqprop_vs_bptt_cosine": (
                    min(float32_bptt_cosines_all)
                    if float32_bptt_cosines_all
                    else None
                ),
                "maximum_float32_eqprop_vs_bptt_symmetric_norm_delta": max(
                    float(row["float32_eqprop_vs_bptt_symmetric_norm_delta"])
                    for row in comparison_rows
                ),
                "minimum_float64_eqprop_vs_bptt_cosine": (
                    min(float64_bptt_cosines_all)
                    if float64_bptt_cosines_all
                    else None
                ),
                "maximum_float64_eqprop_vs_bptt_symmetric_norm_delta": max(
                    float(row["float64_eqprop_vs_bptt_symmetric_norm_delta"])
                    for row in comparison_rows
                ),
                "native_bptt_weight_tensor_count": bptt_tensor_count,
                "bias_gradients_excluded": True,
                "optimizer_steps_applied": False,
                "official_test_read": False,
            },
            completion=completion,
        )
        errors = base.validate_run(run_dir)
        if errors:
            raise RuntimeError(
                "Completed reporting bundle failed validation: " + "; ".join(errors)
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
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="ARCH:SCHEME:ROLE[:ACTUAL_BETA]",
        help="Repeat for explicit cases; omit beta to use the source run's stage selection.",
    )
    parser.add_argument("--all-18", action="store_true")
    parser.add_argument(
        "--eqprop-variant",
        choices=extended.EQPROP_VARIANTS,
        help=(
            "Estimator replayed by this audit. Omit to preserve the source run's "
            "positive one-sided estimator; centered overrides require explicit betas."
        ),
    )
    parser.add_argument(
        "--explicit-beta-label",
        default="explicit_cli",
        help="Provenance label applied to cases with an explicit actual beta.",
    )
    parser.add_argument(
        "--tk-override",
        nargs=2,
        type=int,
        metavar=("T", "K"),
        help=(
            "Replay every explicitly-beta-selected Conv case at this positive T/K "
            "instead of its source-native coordinate."
        ),
    )
    parser.add_argument(
        "--allow-out-of-grid-beta",
        action="store_true",
        help=(
            "Allow an explicit Conv beta outside the completed source grid, without "
            "applying the source injected-current cap; recorded as a protocol extension."
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", default="float64-shadow")
    parser.add_argument(
        "--runtime-source-root", type=Path, default=base.RUNTIME_SOURCE_ROOT
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/filip/datasets/mnist")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--target", default="local:RTX3090")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runtime_source_root.expanduser().resolve() != base.RUNTIME_SOURCE_ROOT:
        raise ValueError(
            "--runtime-source-root must be supplied at process bootstrap and match the exact source contract."
        )
    result = run_audit(args)
    print(
        json.dumps(
            {
                "state": "complete",
                "result_sha256": base.sha256_file(
                    args.output_root.expanduser().resolve()
                    / str(args.run_id)
                    / "result.json"
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
