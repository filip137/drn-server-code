#!/usr/bin/env python3
"""Compare centered or positive one-sided EqProp with BPTT.

This is a deterministic, read-only checkpoint replay.  The trained source
parameters and signed amplification-scaled biases remain active in the energy
dynamics, but only ConvWeight and DenseWeight gradients are scored.  No
optimizer is constructed and the official MNIST test split is never read.
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
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs/conv/perfectdiode_conv123_all_amplification_adam_eqprop_bptt_beta_tk_displacement_seed0_20260810_v1.json"
)
IMMUTABLE_BASE_RUNNER = (
    REPOSITORY_ROOT / "experiments/analyze_conv_eqprop_bptt_checkpoint_gradients.py"
)


def _load_base_runner() -> Any:
    """Load the validated legacy analyzer after its exact-source bootstrap."""

    spec = importlib.util.spec_from_file_location(
        "_eqprop_checkpoint_gradient_base", IMMUTABLE_BASE_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {IMMUTABLE_BASE_RUNNER}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_runner()
np = base.np
torch = base.torch
plt = base.plt
from model.variable.layer import layer_index as exact_runtime_layer_index  # noqa: E402
from training.sgd import _amplified_layer_row_scale  # noqa: E402


SCHEMA = "perfectdiode-conv-eqprop-bptt-beta-tk-displacement/v1"
WEIGHT_TYPES = base.WEIGHT_TYPES
NORM_EPSILON = base.NORM_EPSILON
STATE_BOUND_TOLERANCE = base.STATE_BOUND_TOLERANCE
ROLES = ("reconstructed_initialization", "best_validation")
CENTERED_PHASES = ("negative", "positive")
POSITIVE_ONE_SIDED_PHASES = ("positive",)
# Retain the historical public constant for helper-level callers.
PHASES = CENTERED_PHASES
EQPROP_VARIANTS = ("centered", "positive_one_sided")


def _eqprop_variant(gradient_contract: Mapping[str, Any]) -> str:
    """Resolve the single estimator implemented by one replay invocation."""

    variant = str(gradient_contract.get("eqprop_variant_primary", "centered"))
    if variant not in EQPROP_VARIANTS:
        raise ValueError(
            f"Unsupported EqProp variant {variant!r}; expected one of {EQPROP_VARIANTS}."
        )
    declared = [
        str(value)
        for value in gradient_contract.get("eqprop_variants_reported", [variant])
    ]
    if declared != [variant]:
        raise ValueError(
            "This analyzer reports exactly one EqProp variant per run; "
            f"declared={declared}, primary={variant!r}."
        )
    return variant


def _nudged_phase_plan(eqprop_variant: str) -> tuple[tuple[str, float], ...]:
    """Return phase labels and beta multipliers for an estimator."""

    if eqprop_variant == "centered":
        return (("negative", -1.0), ("positive", 1.0))
    if eqprop_variant == "positive_one_sided":
        return (("positive", 1.0),)
    raise ValueError(f"Unsupported EqProp variant {eqprop_variant!r}.")


def _nudged_phase_names(eqprop_variant: str) -> tuple[str, ...]:
    return tuple(label for label, _multiplier in _nudged_phase_plan(eqprop_variant))


def _variant_title(eqprop_variant: str) -> str:
    return {
        "centered": "centered",
        "positive_one_sided": "positive one-sided",
    }[eqprop_variant]


def _eqprop_estimate(
    *,
    eqprop_variant: str,
    beta: float,
    denominator_scale: float = 1.0,
    zero_gradient: torch.Tensor,
    endpoint_gradients: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Form the configured EqProp finite difference in float64."""

    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("EqProp beta must be finite and positive.")
    denominator_scale = float(denominator_scale)
    if not math.isfinite(denominator_scale) or denominator_scale <= 0.0:
        raise ValueError("EqProp denominator scale must be finite and positive.")
    denominator = beta * denominator_scale
    zero = zero_gradient.detach().cpu().to(torch.float64)
    if eqprop_variant == "centered":
        negative = endpoint_gradients["negative"].detach().cpu().to(torch.float64)
        positive = endpoint_gradients["positive"].detach().cpu().to(torch.float64)
        return (positive - negative) / (2.0 * denominator)
    if eqprop_variant == "positive_one_sided":
        positive = endpoint_gradients["positive"].detach().cpu().to(torch.float64)
        return (positive - zero) / denominator
    raise ValueError(f"Unsupported EqProp variant {eqprop_variant!r}.")


def _case_for_base(case: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the all-scheme case schema to the immutable runner schema."""

    translated = dict(case)
    translated["inference_iterations"] = int(
        case["native_T"]
        if "native_T" in case
        else case["native_inference_iterations"]
    )
    translated["gradient_iterations"] = int(
        case["native_K"]
        if "native_K" in case
        else case["native_gradient_iterations"]
    )
    translated["best_checkpoint_sha256"] = (
        case["best_model_pt_sha256"]
        if "best_model_pt_sha256" in case
        else case["best_checkpoint_sha256"]
    )
    translated["source_manifest_sha256"] = (
        case["manifest_sha256"]
        if "manifest_sha256" in case
        else case["source_manifest_sha256"]
    )
    translated["reconstructed_initialization_tensor_sha256"] = (
        case["reconstructed_init_tensor_sha256"]
        if "reconstructed_init_tensor_sha256" in case
        else case["reconstructed_initialization_tensor_sha256"]
    )
    return translated


def _source_inventory(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, str]]]:
    """Validate all nine source bundles without architecture-key collisions."""

    inventory: list[dict[str, Any]] = []
    hashes: dict[tuple[str, str], dict[str, str]] = {}
    schemes = [str(value["scheme"]) for value in config["amplification_schemes"]]
    for scheme in schemes:
        cases = [
            _case_for_base(case)
            for case in config["cases"]
            if str(case["scheme"]) == scheme
        ]
        if len(cases) != 3:
            raise ValueError(f"Expected three source cases for scheme={scheme}.")
        subconfig = dict(config)
        subconfig["cases"] = cases
        scheme_inventory, scheme_hashes = base._source_run_inventory(subconfig)
        for case in scheme_inventory:
            architecture = str(case["architecture"])
            source_model = base._model_config(case["source_config"])
            declared = next(
                item
                for item in config["cases"]
                if str(item["scheme"]) == scheme
                and str(item["architecture"]) == architecture
            )
            observed_amp = (
                float(source_model["voltage_amp"]),
                float(source_model["current_amp"]),
            )
            expected_amp = (
                float(declared["voltage_amplification"]),
                float(declared["current_amplification"]),
            )
            if observed_amp != expected_amp:
                raise ValueError(
                    f"Amplification mismatch for {scheme}/{architecture}: "
                    f"{observed_amp} != {expected_amp}."
                )
            case.update(
                {
                    "scheme": scheme,
                    "native_T": int(declared["native_inference_iterations"]),
                    "native_K": int(declared["native_gradient_iterations"]),
                    "voltage_amp": expected_amp[0],
                    "current_amp": expected_amp[1],
                }
            )
            inventory.append(case)
            hashes[(scheme, architecture)] = scheme_hashes[architecture]
    expected_pairs = {
        (scheme, architecture)
        for scheme in schemes
        for architecture in ("conv1", "conv2", "conv3")
    }
    observed_pairs = {
        (str(case["scheme"]), str(case["architecture"])) for case in inventory
    }
    if observed_pairs != expected_pairs:
        raise ValueError("The source inventory does not cover the declared 3x3 grid.")
    return inventory, hashes


def _verify_source_hashes(
    inventory: Sequence[Mapping[str, Any]],
    expected: Mapping[tuple[str, str], Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    observed: dict[str, dict[str, str]] = {}
    for case in inventory:
        key = (str(case["scheme"]), str(case["architecture"]))
        run_dir = Path(case["run_dir"])
        current = {
            name: base.sha256_file(run_dir / name) for name in expected[key]
        }
        if current != dict(expected[key]):
            raise RuntimeError(f"Source bytes changed during replay for {key}.")
        observed["/".join(key)] = current
    return observed


def _validate_matched_model_contracts(
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Prove schemes differ only on declared amplification/native iterations."""

    hashes: dict[str, str] = {}
    for architecture in ("conv1", "conv2", "conv3"):
        normalized: list[tuple[str, dict[str, Any]]] = []
        for case in inventory:
            if str(case["architecture"]) != architecture:
                continue
            model = json.loads(json.dumps(base._model_config(case["source_config"])))
            for key in (
                "voltage_amp",
                "current_amp",
                "num_iterations_inference",
                "num_iterations_training",
            ):
                model.pop(key, None)
            normalized.append((str(case["scheme"]), model))
        if len(normalized) != 3:
            raise ValueError(f"Expected three model contracts for {architecture}.")
        reference = normalized[0][1]
        if any(model != reference for _, model in normalized[1:]):
            raise ValueError(
                f"Matched-model contract differs across schemes for {architecture}."
            )
        payload = json.dumps(reference, sort_keys=True, separators=(",", ":"))
        hashes[architecture] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return hashes


def _grid_for_case(
    config: Mapping[str, Any], case: Mapping[str, Any], *, smoke: bool
) -> list[tuple[int, int]]:
    architecture = str(case["architecture"])
    grid = config["tk_factorial"][architecture]
    native = (int(case["native_T"]), int(case["native_K"]))
    explicit = case.get("tk_coordinates", grid.get("coordinates"))
    if explicit is not None:
        coordinates = [
            (int(value[0]), int(value[1])) for value in explicit
        ]
        if not coordinates or any(min(value) <= 0 for value in coordinates):
            raise ValueError(f"Invalid explicit T/K coordinates for {architecture}.")
        coordinates = list(dict.fromkeys(coordinates))
        if native not in coordinates:
            raise ValueError(
                f"Native T/K {native} is absent from explicit {architecture} grid."
            )
        return coordinates[:1] if smoke else coordinates
    coordinate_mode = str(
        case.get(
            "tk_coordinate_mode",
            grid.get(
                "coordinate_mode",
                config.get("tk_coordinate_mode", "factorial"),
            ),
        )
    )
    if coordinate_mode == "native_only":
        return [native]
    if coordinate_mode not in {"factorial", "native_cross"}:
        raise ValueError(f"Unsupported T/K coordinate mode {coordinate_mode!r}.")
    t_values = [int(value) for value in grid["inference_iterations"]]
    k_values = [int(value) for value in grid["gradient_iterations"]]
    if not t_values or not k_values or min(t_values + k_values) <= 0:
        raise ValueError(f"Invalid T/K grid for {architecture}.")
    if coordinate_mode == "factorial":
        coordinates = [(t, k) for t in t_values for k in k_values]
    else:
        native_t, native_k = native
        coordinates = list(
            dict.fromkeys(
                [
                    *((native_t, value) for value in k_values),
                    *((value, native_k) for value in t_values),
                    (64, 64),
                ]
            )
        )
    if native not in coordinates:
        raise ValueError(f"Native T/K {native} is absent from {architecture} grid.")
    if not smoke:
        return coordinates
    # Exercise native and matched-scheme anchors plus a non-native long-phase
    # corner. Conv3 baseline is the sole case whose native and shared anchors
    # differ (12/8 versus 8/8).
    return list(
        dict.fromkeys(
            (native, _shared_anchor(architecture), (max(t_values), max(k_values)))
        )
    )


def _beta_grid_for_case(
    gradient_contract: Mapping[str, Any], case: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Resolve legacy absolute betas or the normalized per-case beta grid."""

    normalized = gradient_contract.get("normalized_betas")
    absolute = gradient_contract.get("betas")
    if normalized is not None:
        if absolute is not None:
            raise ValueError(
                "Declare normalized_betas or legacy betas, not both."
            )
        requested = [float(value) for value in normalized]
        architecture = str(case["architecture"])
        baseline_b0_by_architecture = gradient_contract[
            "baseline_init_output_curvature_scale_b0_by_architecture"
        ]
        baseline_b0 = float(baseline_b0_by_architecture[architecture])
        case_b0 = float(case["init_output_curvature_scale_b0"])
        cap = float(gradient_contract["actual_beta_cap"])
        if not math.isfinite(baseline_b0) or baseline_b0 <= 0.0:
            raise ValueError("Baseline initialization output-curvature scale must be positive.")
        if not math.isfinite(case_b0) or case_b0 <= 0.0:
            raise ValueError("init_output_curvature_scale_b0 must be positive.")
        if not math.isfinite(cap) or cap <= 0.0:
            raise ValueError("actual_beta_cap must be positive.")
        if not requested or len(set(requested)) != len(requested) or any(
            not math.isfinite(value) or value <= 0.0 for value in requested
        ):
            raise ValueError("Normalized betas must be unique, finite, and positive.")
        rows: list[dict[str, Any]] = []
        observed_actual: set[float] = set()
        scored_interaction_count = int(case["scored_interaction_count_L"])
        if scored_interaction_count <= 0:
            raise ValueError("scored_interaction_count_L must be positive.")
        exponent_convention = str(
            gradient_contract.get(
                "amplification_exponent_convention", "scored_weight_interactions"
            )
        )
        if exponent_convention == "output_bias_current_row":
            declared_by_architecture = gradient_contract.get(
                "nudging_scale_power_by_architecture"
            )
            if not isinstance(declared_by_architecture, Mapping):
                raise ValueError(
                    "output_bias_current_row scaling requires an explicit "
                    "nudging_scale_power_by_architecture mapping."
                )
            depth = int(declared_by_architecture[architecture])
        elif exponent_convention == "scored_weight_interactions":
            depth = scored_interaction_count
        else:
            raise ValueError(
                f"Unsupported amplification exponent convention {exponent_convention!r}."
            )
        if depth <= 0:
            raise ValueError("Amplification exponent L must be positive.")
        voltage_amp = float(case["voltage_amp"] if "voltage_amp" in case else case["voltage_amplification"])
        current_amp = float(case["current_amp"] if "current_amp" in case else case["current_amplification"])
        if voltage_amp <= 0.0 or current_amp <= 0.0:
            raise ValueError("Amplification values must be positive.")
        amplification_factor = (voltage_amp / current_amp) ** depth
        parameterization = str(
            gradient_contract.get(
                "normalized_beta_parameterization", "common_effective"
            )
        )
        if parameterization not in {"common_effective", "common_base"}:
            raise ValueError(
                f"Unsupported normalized beta parameterization {parameterization!r}."
            )
        for beta_hat_requested in requested:
            if parameterization == "common_effective":
                uncapped = (
                    beta_hat_requested * baseline_b0 / amplification_factor
                )
                beta = min(uncapped, cap)
                capped = bool(uncapped >= cap)
                cap_semantics = "actual_beta"
            else:
                uncapped = beta_hat_requested * baseline_b0
                beta = min(uncapped, cap / amplification_factor)
                capped = bool(uncapped * amplification_factor >= cap)
                cap_semantics = "injected_beta"
            if beta in observed_actual:
                continue
            observed_actual.add(beta)
            rows.append(
                {
                    "beta_hat_requested": beta_hat_requested,
                    "beta": beta,
                    "beta_effective": beta * amplification_factor,
                    "beta_hat": beta * amplification_factor / baseline_b0,
                    "beta_hat_base": beta / baseline_b0,
                    "capped": capped,
                    "baseline_init_output_curvature_scale_b0": baseline_b0,
                    "init_output_curvature_scale_b0": case_b0,
                    "amplification_depth_L": depth,
                    "amplification_exponent_convention": exponent_convention,
                    "scored_interaction_count": scored_interaction_count,
                    "amplification_factor": amplification_factor,
                    "actual_beta_cap": cap,
                    "cap_semantics": cap_semantics,
                    "normalized_beta_parameterization": parameterization,
                    "beta_mode": (
                        "common_base_with_explicit_current_scaling"
                        if parameterization == "common_base"
                        else "common_effective_baseline_output_curvature"
                    ),
                }
            )
        return rows
    betas = [float(value) for value in (absolute or [])]
    if not betas or len(set(betas)) != len(betas) or any(
        not math.isfinite(value) or value <= 0.0 for value in betas
    ):
        raise ValueError("Betas must be unique, finite, and positive.")
    return [
        {
            "beta_hat_requested": beta,
            "beta": beta,
            "beta_effective": beta,
            "beta_hat": beta,
            "capped": False,
            "baseline_init_output_curvature_scale_b0": None,
            "init_output_curvature_scale_b0": None,
            "amplification_depth_L": None,
            "amplification_factor": 1.0,
            "actual_beta_cap": None,
            "beta_mode": "legacy_absolute",
        }
        for beta in betas
    ]


def _beta_metadata_by_actual(
    beta_grid: Sequence[Mapping[str, Any]],
) -> dict[float, Mapping[str, Any]]:
    output = {float(row["beta"]): row for row in beta_grid}
    if len(output) != len(beta_grid):
        raise ValueError("Resolved actual betas must be unique.")
    return output


def _beta_reporting_fields(
    beta: float,
    metadata_by_actual: Mapping[float, Mapping[str, Any]],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    beta = float(beta)
    if beta == 0.0:
        return {
            "beta_hat_requested": 0.0,
            "beta_hat": 0.0,
            "beta_effective": 0.0,
            "beta_hat_base": 0.0,
            "beta_capped": False,
            "beta_over_case_init_b0": 0.0,
            "beta_over_checkpoint_b0": 0.0,
        }
    metadata = metadata_by_actual[beta]
    init_b0 = None if context is None else context.get(
        "init_output_curvature_scale_b0"
    )
    checkpoint_b0 = None if context is None else context.get(
        "checkpoint_output_curvature_scale_b0"
    )
    return {
        "beta_hat_requested": float(metadata["beta_hat_requested"]),
        "beta_hat": float(metadata["beta_hat"]),
        "beta_effective": float(metadata["beta_effective"]),
        "beta_hat_base": float(metadata.get("beta_hat_base", metadata["beta_hat"])),
        "beta_capped": bool(metadata["capped"]),
        "beta_mode": str(metadata["beta_mode"]),
        "baseline_init_output_curvature_scale_b0": metadata.get(
            "baseline_init_output_curvature_scale_b0"
        ),
        "amplification_depth_L": metadata.get("amplification_depth_L"),
        "amplification_exponent_convention": metadata.get(
            "amplification_exponent_convention"
        ),
        "scored_interaction_count": metadata.get("scored_interaction_count"),
        "amplification_factor": metadata.get("amplification_factor"),
        "beta_over_case_init_b0": (
            beta / float(init_b0) if init_b0 is not None else None
        ),
        "beta_over_checkpoint_b0": (
            beta / float(checkpoint_b0) if checkpoint_b0 is not None else None
        ),
    }


def _output_curvature_diagnostic(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Measure b0=2*standard median(a_out) without changing runtime state.

    ``torch.quantile(..., 0.5)`` averages the two middle values for an even
    output width, matching NumPy's standard median and the declared contract.
    """

    output_layer = runtime["free_layers"][-1]
    coefficients = (
        runtime["energy_fn"]
        .a_coef_fn(output_layer)()
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )
    base._finite(coefficients, "output energy quadratic coefficient")
    if coefficients.numel() == 0 or bool((coefficients <= 0.0).any()):
        raise ValueError("Output energy quadratic coefficients must be positive.")
    digest = hashlib.sha256()
    digest.update(str(tuple(coefficients.shape)).encode("ascii"))
    digest.update(coefficients.numpy().tobytes())
    median = float(torch.quantile(coefficients.flatten(), 0.5))
    return {
        "output_curvature_element_count": int(coefficients.numel()),
        "output_curvature_minimum": float(coefficients.min()),
        "output_curvature_median": median,
        "output_curvature_maximum": float(coefficients.max()),
        "output_curvature_tensor_sha256": digest.hexdigest(),
        "output_curvature_scale_b0": 2.0 * median,
    }


def _state_component_metrics(
    layer: Any,
    current: torch.Tensor,
    reference: torch.Tensor,
    *,
    reference_norm_epsilon: float,
) -> dict[str, Any]:
    current64 = current.detach().to(torch.float64)
    reference64 = reference.detach().to(torch.float64)
    delta = current64 - reference64
    delta_squared_sum = float(delta.square().sum().cpu())
    reference_squared_sum = float(reference64.square().sum().cpu())
    element_count = int(reference.numel())
    row: dict[str, Any] = {
        "state_layer_name": str(layer.name),
        "state_layer_type": layer.__class__.__name__,
        "state_shape": "x".join(str(value) for value in reference.shape[1:]),
        "element_count": element_count,
        "delta_squared_sum": delta_squared_sum,
        "reference_squared_sum": reference_squared_sum,
        "current_sum": float(current64.sum().cpu()),
        "current_squared_sum": float(current64.square().sum().cpu()),
        "reference_sum": float(reference64.sum().cpu()),
        "delta_sum": float(delta.sum().cpu()),
        "current_min": float(current64.min().cpu()),
        "current_max": float(current64.max().cpu()),
        "reference_min": float(reference64.min().cpu()),
        "reference_max": float(reference64.max().cpu()),
        "displacement_l2": math.sqrt(delta_squared_sum),
        "reference_l2": math.sqrt(reference_squared_sum),
        "reference_norm_epsilon": float(reference_norm_epsilon),
        "relative_displacement": math.sqrt(delta_squared_sum) / max(
            math.sqrt(reference_squared_sum), float(reference_norm_epsilon)
        ),
        "delta_rms": math.sqrt(delta_squared_sum / max(element_count, 1)),
        "reference_rms": math.sqrt(
            reference_squared_sum / max(element_count, 1)
        ),
        "max_abs_delta": float(delta.abs().max().cpu()),
        "active_set_transition_count": 0,
        "constrained_element_count": 0,
        "active_set_transition_fraction": 0.0,
    }
    if getattr(layer, "non_linearity", None) == "perfect_diode":
        split = int(current.shape[1]) // 2
        current_active = torch.cat(
            (
                current[:, :split] <= STATE_BOUND_TOLERANCE,
                current[:, split:] >= -STATE_BOUND_TOLERANCE,
            ),
            dim=1,
        )
        reference_active = torch.cat(
            (
                reference[:, :split] <= STATE_BOUND_TOLERANCE,
                reference[:, split:] >= -STATE_BOUND_TOLERANCE,
            ),
            dim=1,
        )
        transitions = int((current_active != reference_active).sum())
        constrained = int(current.numel())
        row.update(
            {
                "active_set_transition_count": transitions,
                "constrained_element_count": constrained,
                "active_set_transition_fraction": transitions / constrained,
            }
        )
    return row


def _state_displacement_rows(
    layers: Sequence[Any],
    current_states: Sequence[torch.Tensor],
    reference_states: Sequence[torch.Tensor],
    *,
    reference_kind: str,
    reference_norm_epsilon: float,
) -> list[dict[str, Any]]:
    rows = [
        {
            "reference_kind": reference_kind,
            "outcome": "ok",
            **_state_component_metrics(
                layer,
                current,
                reference,
                reference_norm_epsilon=reference_norm_epsilon,
            ),
        }
        for layer, current, reference in zip(
            layers, current_states, reference_states, strict=True
        )
    ]
    delta_squared_sum = sum(float(row["delta_squared_sum"]) for row in rows)
    reference_squared_sum = sum(
        float(row["reference_squared_sum"]) for row in rows
    )
    element_count = sum(int(row["element_count"]) for row in rows)
    transitions = sum(int(row["active_set_transition_count"]) for row in rows)
    constrained = sum(int(row["constrained_element_count"]) for row in rows)
    rows.append(
        {
            "reference_kind": reference_kind,
            "outcome": "ok",
            "state_layer_name": "__all__",
            "state_layer_type": "aggregate",
            "state_shape": "all_free_layers",
            "element_count": element_count,
            "delta_squared_sum": delta_squared_sum,
            "reference_squared_sum": reference_squared_sum,
            "displacement_l2": math.sqrt(delta_squared_sum),
            "reference_l2": math.sqrt(reference_squared_sum),
            "reference_norm_epsilon": float(reference_norm_epsilon),
            "relative_displacement": math.sqrt(delta_squared_sum) / max(
                math.sqrt(reference_squared_sum), float(reference_norm_epsilon)
            ),
            "delta_rms": math.sqrt(delta_squared_sum / max(element_count, 1)),
            "reference_rms": math.sqrt(
                reference_squared_sum / max(element_count, 1)
            ),
            "max_abs_delta": max(float(row["max_abs_delta"]) for row in rows),
            "active_set_transition_count": transitions,
            "constrained_element_count": constrained,
            "active_set_transition_fraction": (
                transitions / constrained if constrained else 0.0
            ),
        }
    )
    return rows


def _unavailable_state_displacement_rows(
    layers: Sequence[Any],
    reference_states: Sequence[torch.Tensor],
    *,
    reference_kind: str,
    reference_norm_epsilon: float,
    outcome: str,
) -> list[dict[str, Any]]:
    """Materialize structured rows when a nudged endpoint is non-finite."""

    rows: list[dict[str, Any]] = []
    for layer, reference in zip(layers, reference_states, strict=True):
        reference_squared_sum = float(
            reference.detach().to(torch.float64).square().sum().cpu()
        )
        constrained = (
            int(reference.numel())
            if getattr(layer, "non_linearity", None) == "perfect_diode"
            else 0
        )
        rows.append(
            {
                "reference_kind": reference_kind,
                "outcome": outcome,
                "state_layer_name": str(layer.name),
                "state_layer_type": layer.__class__.__name__,
                "state_shape": "x".join(
                    str(value) for value in reference.shape[1:]
                ),
                "element_count": int(reference.numel()),
                "delta_squared_sum": None,
                "reference_squared_sum": reference_squared_sum,
                "displacement_l2": None,
                "reference_l2": math.sqrt(reference_squared_sum),
                "reference_norm_epsilon": float(reference_norm_epsilon),
                "relative_displacement": None,
                "delta_rms": None,
                "reference_rms": math.sqrt(
                    reference_squared_sum / max(int(reference.numel()), 1)
                ),
                "max_abs_delta": None,
                "active_set_transition_count": None,
                "constrained_element_count": constrained,
                "active_set_transition_fraction": None,
            }
        )
    rows.append(
        {
            "reference_kind": reference_kind,
            "outcome": outcome,
            "state_layer_name": "__all__",
            "state_layer_type": "aggregate",
            "state_shape": "all_free_layers",
            "element_count": sum(int(row["element_count"]) for row in rows),
            "delta_squared_sum": None,
            "reference_squared_sum": sum(
                float(row["reference_squared_sum"]) for row in rows
            ),
            "displacement_l2": None,
            "reference_l2": math.sqrt(
                sum(float(row["reference_squared_sum"]) for row in rows)
            ),
            "reference_norm_epsilon": float(reference_norm_epsilon),
            "relative_displacement": None,
            "delta_rms": None,
            "reference_rms": math.sqrt(
                sum(float(row["reference_squared_sum"]) for row in rows)
                / max(sum(int(row["element_count"]) for row in rows), 1)
            ),
            "max_abs_delta": None,
            "active_set_transition_count": None,
            "constrained_element_count": sum(
                int(row["constrained_element_count"]) for row in rows
            ),
            "active_set_transition_fraction": None,
        }
    )
    return rows


def _projected_perfect_diode_residual(
    state: torch.Tensor,
    gradient: torch.Tensor,
    *,
    epsilon: float = STATE_BOUND_TOLERANCE,
) -> torch.Tensor:
    """Return the protocol's elementwise projected-KKT residual."""

    if state.shape != gradient.shape:
        raise ValueError(
            f"State/gradient shape mismatch: {tuple(state.shape)} != {tuple(gradient.shape)}."
        )
    if state.ndim < 2 or int(state.shape[1]) % 2:
        raise ValueError(
            "A perfect-diode state needs an even excitatory/inhibitory axis."
        )
    split = int(state.shape[1]) // 2
    residual = torch.empty_like(gradient)
    excitatory_state = state[:, :split]
    excitatory_gradient = gradient[:, :split]
    residual[:, :split] = torch.where(
        excitatory_state > float(epsilon),
        excitatory_gradient.abs(),
        torch.relu(-excitatory_gradient),
    )
    inhibitory_state = state[:, split:]
    inhibitory_gradient = gradient[:, split:]
    residual[:, split:] = torch.where(
        inhibitory_state < -float(epsilon),
        inhibitory_gradient.abs(),
        torch.relu(inhibitory_gradient),
    )
    return residual


def _residual_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "p99": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    if not bool(np.isfinite(array).all()):
        raise FloatingPointError("Non-finite per-sample equilibrium residual.")
    return {
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
    }


class _ResidualAccumulator:
    """Collect exact layerwise maxima before forming cohort quantiles."""

    def __init__(self) -> None:
        self.records: dict[tuple[Any, ...], dict[str, Any]] = {}

    @staticmethod
    def _key(
        *,
        phase: str,
        beta: float,
        layer_index: int,
        layer: Any,
        layer_count: int,
    ) -> tuple[Any, ...]:
        layer_role = "output" if layer_index == layer_count - 1 else f"hidden_{layer_index}"
        residual_mode = (
            "projected_kkt"
            if getattr(layer, "non_linearity", None) == "perfect_diode"
            else "raw"
        )
        return (
            str(phase),
            float(beta),
            int(layer_index),
            str(layer.name),
            layer.__class__.__name__,
            layer_role,
            residual_mode,
        )

    def add_endpoint(
        self,
        runtime: Mapping[str, Any],
        *,
        gradient_source: Any,
        phase: str,
        beta: float,
        source_indices: Sequence[int],
    ) -> None:
        layers = runtime["free_layers"]
        for layer_index, layer in enumerate(layers):
            gradient = gradient_source.grad_layer_fn(layer)().detach()
            state = layer.state.detach()
            base._finite(gradient, f"per-sample phase residual {phase}/{layer.name}")
            raw = gradient.abs()
            if getattr(layer, "non_linearity", None) == "perfect_diode":
                selected = _projected_perfect_diode_residual(state, gradient)
                split = int(state.shape[1]) // 2
                clamped = torch.cat(
                    (
                        state[:, :split] <= STATE_BOUND_TOLERANCE,
                        state[:, split:] >= -STATE_BOUND_TOLERANCE,
                    ),
                    dim=1,
                )
                occupancy = (
                    clamped.reshape(clamped.shape[0], -1)
                    .to(torch.float64)
                    .mean(dim=1)
                    .cpu()
                    .tolist()
                )
            else:
                selected = raw
                occupancy = [None] * int(state.shape[0])
            selected_max = (
                selected.reshape(selected.shape[0], -1)
                .amax(dim=1)
                .to(device="cpu", dtype=torch.float64)
                .tolist()
            )
            raw_max = (
                raw.reshape(raw.shape[0], -1)
                .amax(dim=1)
                .to(device="cpu", dtype=torch.float64)
                .tolist()
            )
            if not (
                len(source_indices)
                == len(selected_max)
                == len(raw_max)
                == len(occupancy)
            ):
                raise RuntimeError("Residual sample-count mismatch.")
            key = self._key(
                phase=phase,
                beta=beta,
                layer_index=layer_index,
                layer=layer,
                layer_count=len(layers),
            )
            record = self.records.setdefault(
                key,
                {
                    "source_indices": [],
                    "selected_max": [],
                    "raw_max": [],
                    "clamp_occupancy": [],
                    "outcome": "ok",
                },
            )
            record["source_indices"].extend(int(value) for value in source_indices)
            record["selected_max"].extend(float(value) for value in selected_max)
            record["raw_max"].extend(float(value) for value in raw_max)
            record["clamp_occupancy"].extend(occupancy)

    def mark_unavailable(
        self,
        runtime: Mapping[str, Any],
        *,
        phase: str,
        beta: float,
        outcome: str,
    ) -> None:
        layers = runtime["free_layers"]
        for layer_index, layer in enumerate(layers):
            key = self._key(
                phase=phase,
                beta=beta,
                layer_index=layer_index,
                layer=layer,
                layer_count=len(layers),
            )
            self.records.setdefault(
                key,
                {
                    "source_indices": [],
                    "selected_max": [],
                    "raw_max": [],
                    "clamp_occupancy": [],
                    "outcome": outcome,
                },
            )["outcome"] = outcome

    def summary_rows(
        self,
        *,
        context: Mapping[str, Any],
        expected_examples: int,
        p90_threshold: float,
        beta_metadata: Mapping[float, Mapping[str, Any]] | None = None,
        uniform_maximum_threshold: float | None = None,
        hard_maximum_threshold: float = 1.0e-1,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if uniform_maximum_threshold is None:
            uniform_maximum_threshold = float(p90_threshold)
        for key, record in sorted(self.records.items(), key=lambda item: item[0]):
            (
                phase,
                beta,
                layer_index,
                layer_name,
                layer_type,
                layer_role,
                residual_mode,
            ) = key
            selected = record["selected_max"]
            raw = record["raw_max"]
            source_indices = record["source_indices"]
            occupancy = [
                float(value)
                for value in record["clamp_occupancy"]
                if value is not None
            ]
            selected_stats = _residual_stats(selected)
            raw_stats = _residual_stats(raw)
            occupancy_stats = _residual_stats(occupancy)
            coverage_complete = bool(
                str(record["outcome"]) == "ok"
                and len(selected) == int(expected_examples)
                and len(source_indices) == int(expected_examples)
                and len(set(source_indices)) == int(expected_examples)
            )
            uniform_equilibrium = bool(
                coverage_complete
                and selected_stats["maximum"] is not None
                and float(selected_stats["maximum"])
                < float(uniform_maximum_threshold)
            )
            hard_failure = bool(
                not coverage_complete
                or selected_stats["maximum"] is None
                or float(selected_stats["maximum"])
                >= float(hard_maximum_threshold)
            )
            gate_passed = bool(
                coverage_complete
                and selected_stats["p90"] is not None
                and float(selected_stats["p90"]) < float(p90_threshold)
                and not hard_failure
            )
            digest = hashlib.sha256()
            digest.update(np.asarray(source_indices, dtype=np.int64).tobytes())
            digest.update(np.asarray(selected, dtype=np.float64).tobytes())
            rows.append(
                {
                    "schema": SCHEMA,
                    **context,
                    "phase": phase,
                    "beta": float(beta),
                    **(
                        _beta_reporting_fields(
                            float(beta), beta_metadata, context=context
                        )
                        if beta_metadata is not None
                        else {}
                    ),
                    "layer_index": int(layer_index),
                    "state_layer_name": layer_name,
                    "state_layer_type": layer_type,
                    "layer_role": layer_role,
                    "residual_mode": residual_mode,
                    "example_count": len(selected),
                    "expected_example_count": int(expected_examples),
                    "unique_source_index_count": len(set(source_indices)),
                    "coverage_complete": coverage_complete,
                    "outcome": str(record["outcome"]),
                    "per_sample_selected_max_sha256": digest.hexdigest(),
                    "selected_max_mean": selected_stats["mean"],
                    "selected_max_median": selected_stats["median"],
                    "selected_max_p90": selected_stats["p90"],
                    "selected_max_p99": selected_stats["p99"],
                    "selected_maximum": selected_stats["maximum"],
                    "raw_max_mean": raw_stats["mean"],
                    "raw_max_median": raw_stats["median"],
                    "raw_max_p90": raw_stats["p90"],
                    "raw_max_p99": raw_stats["p99"],
                    "raw_maximum": raw_stats["maximum"],
                    "clamp_occupancy_mean": occupancy_stats["mean"],
                    "clamp_occupancy_median": occupancy_stats["median"],
                    "clamp_occupancy_p90": occupancy_stats["p90"],
                    "clamp_occupancy_p99": occupancy_stats["p99"],
                    "clamp_occupancy_maximum": occupancy_stats["maximum"],
                    "gate_threshold": float(p90_threshold),
                    "gate_comparison": "strictly_less_than",
                    "gate_passed": gate_passed,
                    "uniform_equilibrium_max_threshold": float(
                        uniform_maximum_threshold
                    ),
                    "uniform_equilibrium": uniform_equilibrium,
                    "hard_failure_maximum_threshold": float(
                        hard_maximum_threshold
                    ),
                    "hard_failure": hard_failure,
                }
            )
        return rows

    def archive_records(
        self,
        *,
        context: Mapping[str, Any],
        beta_metadata: Mapping[float, Mapping[str, Any]] | None = None,
    ) -> list[tuple[dict[str, Any], np.ndarray]]:
        output: list[tuple[dict[str, Any], np.ndarray]] = []
        dtype = np.dtype(
            [
                ("source_index", "<i8"),
                ("selected_max", "<f8"),
                ("raw_max", "<f8"),
                ("clamp_occupancy", "<f8"),
            ]
        )
        for key, record in sorted(self.records.items(), key=lambda item: item[0]):
            (
                phase,
                beta,
                layer_index,
                layer_name,
                layer_type,
                layer_role,
                residual_mode,
            ) = key
            count = len(record["source_indices"])
            values = np.empty(count, dtype=dtype)
            values["source_index"] = np.asarray(
                record["source_indices"], dtype=np.int64
            )
            values["selected_max"] = np.asarray(
                record["selected_max"], dtype=np.float64
            )
            values["raw_max"] = np.asarray(record["raw_max"], dtype=np.float64)
            values["clamp_occupancy"] = np.asarray(
                [np.nan if value is None else float(value) for value in record["clamp_occupancy"]],
                dtype=np.float64,
            )
            output.append(
                (
                    {
                        **context,
                        "phase": phase,
                        "beta": float(beta),
                        **(
                            _beta_reporting_fields(
                                float(beta), beta_metadata, context=context
                            )
                            if beta_metadata is not None
                            else {}
                        ),
                        "layer_index": int(layer_index),
                        "state_layer_name": layer_name,
                        "state_layer_type": layer_type,
                        "layer_role": layer_role,
                        "residual_mode": residual_mode,
                        "outcome": str(record["outcome"]),
                    },
                    values,
                )
            )
        return output


def _write_residual_archive(
    path: Path, records: Sequence[tuple[Mapping[str, Any], np.ndarray]]
) -> None:
    arrays: dict[str, Any] = {}
    metadata: list[dict[str, Any]] = []
    for index, (record_metadata, values) in enumerate(records):
        key = f"residual_{index:06d}"
        arrays[key] = values
        metadata.append({"array_key": key, **dict(record_metadata)})
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, allow_nan=False)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _residual_gate_by_beta(
    rows: Sequence[Mapping[str, Any]],
    *,
    betas: Sequence[float],
    eqprop_variant: str,
) -> dict[float, dict[str, Any]]:
    """Require post-T, matched-zero, and estimator endpoint layer gates."""

    phases = _nudged_phase_names(eqprop_variant)
    base_rows = [row for row in rows if str(row["phase"]) in {"post_T_free", "zero"}]
    output: dict[float, dict[str, Any]] = {}
    for beta in betas:
        nudged_rows = [
            row
            for row in rows
            if str(row["phase"]) in phases and float(row["beta"]) == float(beta)
        ]
        required = [*base_rows, *nudged_rows]
        observed_phases = {str(row["phase"]) for row in required}
        expected_phases = {"post_T_free", "zero", *phases}
        passed = bool(
            required
            and observed_phases == expected_phases
            and all(bool(row["gate_passed"]) for row in required)
        )
        output[float(beta)] = {
            "passed": passed,
            "required_layer_phase_count": len(required),
            "failed_layer_phases": [
                f"{row['phase']}:{row['state_layer_name']}"
                for row in required
                if not bool(row["gate_passed"])
            ],
        }
    return output


def _run_gradient_phases(
    runtime: Mapping[str, Any],
    *,
    free_states: Sequence[torch.Tensor],
    betas: Sequence[float],
    zero_epsilon: float,
    state_reference_norm_epsilon: float,
    zero_endpoint_relative_tolerance: float,
    eqprop_variant: str = "centered",
    residual_accumulator: _ResidualAccumulator | None = None,
    source_indices: Sequence[int] = (),
) -> tuple[
    dict[str, torch.Tensor],
    dict[float, dict[str, torch.Tensor]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[float, str],
    dict[float, dict[str, dict[str, torch.Tensor]]],
]:
    """Run BPTT and the configured EqProp estimator from one post-T state."""

    phase_plan = _nudged_phase_plan(eqprop_variant)
    nudging_mode = str(runtime.get("nudging_mode", "cost"))
    nudging_current_scale = float(runtime.get("nudging_current_scale", 1.0))
    denominator_scale = nudging_current_scale if nudging_mode == "current" else 1.0

    free_hash = base._tensor_sequence_sha256(
        (
            (str(layer.name), state)
            for layer, state in zip(
                runtime["free_layers"], free_states, strict=True
            )
        ),
        schema=b"drn-ordered-free-layer-state/v1",
    )
    base._restore_states(runtime["free_layers"], free_states)
    if base._layer_state_sha256(runtime["free_layers"]) != free_hash:
        raise RuntimeError("BPTT did not start from the common post-T state.")
    bptt_all = runtime["backprop"].compute_gradient()
    if len(bptt_all) != len(runtime["parameters"]):
        raise RuntimeError("BPTT gradient/parameter count mismatch.")
    for gradient in bptt_all:
        base._finite(gradient, "BPTT gradient")
    bptt = base._weight_gradient_dict(runtime, bptt_all)
    bptt_zero_energy = base._energy_gradients(runtime)
    bptt_zero_states = base._clone_states(runtime["free_layers"])
    bptt_zero_hash = base._layer_state_sha256(runtime["free_layers"])

    # Execute beta=0 explicitly.  It defines the matched K-step relaxation
    # reference and must reproduce BPTT's forward endpoint.
    base._restore_states(runtime["free_layers"], free_states)
    zero_start_hash = base._layer_state_sha256(runtime["free_layers"])
    runtime["augmented_fn"].prepare_nudging()
    runtime["augmented_fn"].nudging = 0.0
    runtime["minimizer_augmented"].compute_equilibrium()
    zero_energy = base._energy_gradients(runtime)
    zero_states = base._clone_states(runtime["free_layers"])
    zero_hash = base._layer_state_sha256(runtime["free_layers"])
    if residual_accumulator is not None:
        residual_accumulator.add_endpoint(
            runtime,
            gradient_source=runtime["augmented_fn"],
            phase="zero",
            beta=0.0,
            source_indices=source_indices,
        )

    gradient_delta_squared = sum(
        float(
            (left.detach().to(torch.float64) - right.detach().to(torch.float64))
            .square()
            .sum()
            .cpu()
        )
        for left, right in zip(bptt_zero_energy, zero_energy, strict=True)
    )
    gradient_reference_squared = sum(
        float(value.detach().to(torch.float64).square().sum().cpu())
        for value in bptt_zero_energy
    )
    state_delta_squared = sum(
        float(
            (left.detach().to(torch.float64) - right.detach().to(torch.float64))
            .square()
            .sum()
            .cpu()
        )
        for left, right in zip(bptt_zero_states, zero_states, strict=True)
    )
    state_reference_squared = sum(
        float(value.detach().to(torch.float64).square().sum().cpu())
        for value in bptt_zero_states
    )
    gradient_relative_delta = math.sqrt(gradient_delta_squared) / max(
        math.sqrt(gradient_reference_squared), NORM_EPSILON
    )
    state_relative_delta = math.sqrt(state_delta_squared) / max(
        math.sqrt(state_reference_squared), NORM_EPSILON
    )
    equivalent = bool(
        gradient_relative_delta <= zero_endpoint_relative_tolerance
        and state_relative_delta <= zero_endpoint_relative_tolerance
    )
    if not equivalent:
        raise RuntimeError(
            "Explicit beta=0 endpoint differs from BPTT forward endpoint: "
            f"gradient={gradient_relative_delta}, state={state_relative_delta}."
        )

    phase_rows: list[dict[str, Any]] = [
        {
            "beta": 0.0,
            "phase": "zero",
            "nudging": 0.0,
            "eqprop_variant": eqprop_variant,
            "estimator_reference_phase": "zero",
            "common_post_t_state_sha256": free_hash,
            "phase_start_state_sha256": zero_start_hash,
            "phase_end_state_sha256": zero_hash,
            "bptt_zero_endpoint_sha256": bptt_zero_hash,
            "bptt_zero_gradient_relative_l2_delta": gradient_relative_delta,
            "bptt_zero_state_relative_l2_delta": state_relative_delta,
            "bptt_zero_endpoint_equivalent": equivalent,
            "outcome": "ok",
            **base._state_delta_metrics(runtime["free_layers"], free_states),
            **base._phase_residual(runtime),
        }
    ]
    state_rows: list[dict[str, Any]] = []
    for row in _state_displacement_rows(
        runtime["free_layers"],
        zero_states,
        free_states,
        reference_kind="post_T_free",
        reference_norm_epsilon=state_reference_norm_epsilon,
    ):
        state_rows.append({"beta": 0.0, "phase": "zero", **row})

    estimates: dict[float, dict[str, torch.Tensor]] = {}
    precision_endpoints: dict[
        float, dict[str, dict[str, torch.Tensor]]
    ] = {}
    outcomes: dict[float, str] = {}
    for beta in betas:
        endpoints: dict[str, list[torch.Tensor]] = {}
        endpoint_gradients: dict[str, list[torch.Tensor]] = {}
        phase_indices: list[int] = []
        for label, beta_multiplier in phase_plan:
            nudging = float(beta_multiplier) * float(beta)
            base._restore_states(runtime["free_layers"], free_states)
            start_hash = base._layer_state_sha256(runtime["free_layers"])
            if start_hash != free_hash:
                raise RuntimeError(
                    f"EqProp {label} phase did not start from the common post-T state."
                )
            runtime["augmented_fn"].prepare_nudging()
            runtime["augmented_fn"].nudging = nudging
            row: dict[str, Any] = {
                "beta": float(beta),
                "phase": label,
                "nudging": nudging,
                "nudging_mode": nudging_mode,
                "nudging_current_scale": nudging_current_scale,
                "effective_nudging": nudging * denominator_scale,
                "finite_difference_denominator": float(beta) * denominator_scale,
                "eqprop_variant": eqprop_variant,
                "estimator_reference_phase": (
                    "negative_and_positive"
                    if eqprop_variant == "centered"
                    else "zero_and_positive"
                ),
                "common_post_t_state_sha256": free_hash,
                "matched_zero_k_state_sha256": zero_hash,
                "phase_start_state_sha256": start_hash,
            }
            try:
                runtime["minimizer_augmented"].compute_equilibrium()
                candidate_gradients = base._energy_gradients(runtime)
                candidate_states = base._clone_states(runtime["free_layers"])
                if not all(
                    bool(torch.isfinite(value).all()) for value in candidate_states
                ):
                    raise FloatingPointError(
                        f"Non-finite layer state in EqProp {label} phase."
                    )
                row.update(
                    {
                        "phase_end_state_sha256": base._layer_state_sha256(
                            runtime["free_layers"]
                        ),
                        "phase_state_all_finite": True,
                        "outcome": "ok",
                        **base._state_delta_metrics(
                            runtime["free_layers"], free_states
                        ),
                        **base._phase_residual(runtime),
                    }
                )
                endpoints[label] = candidate_states
                endpoint_gradients[label] = candidate_gradients
                if residual_accumulator is not None:
                    residual_accumulator.add_endpoint(
                        runtime,
                        gradient_source=runtime["augmented_fn"],
                        phase=label,
                        beta=float(beta),
                        source_indices=source_indices,
                    )
                for reference_kind, reference_states in (
                    ("post_T_free", free_states),
                    ("matched_zero_K", zero_states),
                ):
                    for state_row in _state_displacement_rows(
                        runtime["free_layers"],
                        candidate_states,
                        reference_states,
                        reference_kind=reference_kind,
                        reference_norm_epsilon=state_reference_norm_epsilon,
                    ):
                        state_rows.append(
                            {
                                "beta": float(beta),
                                "phase": label,
                                **state_row,
                            }
                        )
            except FloatingPointError as error:
                unstable_outcome = f"{label}_phase_unstable"
                row.update(
                    {
                        "phase_end_state_sha256": base._layer_state_sha256(
                            runtime["free_layers"]
                        ),
                        "phase_state_all_finite": all(
                            bool(torch.isfinite(layer.state).all())
                            for layer in runtime["free_layers"]
                        ),
                        "outcome": unstable_outcome,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                for reference_kind, reference_states in (
                    ("post_T_free", free_states),
                    ("matched_zero_K", zero_states),
                ):
                    for state_row in _unavailable_state_displacement_rows(
                        runtime["free_layers"],
                        reference_states,
                        reference_kind=reference_kind,
                        reference_norm_epsilon=state_reference_norm_epsilon,
                        outcome=unstable_outcome,
                    ):
                        state_rows.append(
                            {
                                "beta": float(beta),
                                "phase": label,
                                **state_row,
                            }
                        )
                if residual_accumulator is not None:
                    residual_accumulator.mark_unavailable(
                        runtime,
                        phase=label,
                        beta=float(beta),
                        outcome=unstable_outcome,
                    )
            phase_rows.append(row)
            phase_indices.append(len(phase_rows) - 1)

        required_phases = set(_nudged_phase_names(eqprop_variant))
        available_phases = set(endpoint_gradients)
        if available_phases >= required_phases:
            estimates[float(beta)] = {}
            precision_endpoints[float(beta)] = {}
            for index in runtime["weight_indices"]:
                name = str(runtime["parameters"][index].name).strip()
                estimate = _eqprop_estimate(
                    eqprop_variant=eqprop_variant,
                    beta=float(beta),
                    denominator_scale=denominator_scale,
                    zero_gradient=zero_energy[index],
                    endpoint_gradients={
                        phase: endpoint_gradients[phase][index]
                        for phase in required_phases
                    },
                )
                base._finite(
                    estimate,
                    f"{eqprop_variant} EqProp {name} beta={beta}",
                )
                estimates[float(beta)][name] = estimate
                precision_endpoints[float(beta)][name] = {
                    "zero": zero_energy[index]
                    .detach()
                    .cpu()
                    .to(torch.float64),
                    "positive": endpoint_gradients["positive"][index]
                    .detach()
                    .cpu()
                    .to(torch.float64),
                }
            outcome = "ok"
            if eqprop_variant == "centered":
                symmetry_post_t = base._symmetry_defect(
                    free_states, endpoints["negative"], endpoints["positive"]
                )
                symmetry_zero_k = base._symmetry_defect(
                    zero_states, endpoints["negative"], endpoints["positive"]
                )
            else:
                symmetry_post_t = None
                symmetry_zero_k = None
        else:
            missing = sorted(required_phases - available_phases)
            outcome = (
                "both_nudged_phases_unstable"
                if eqprop_variant == "centered" and len(missing) == 2
                else f"{missing[0]}_phase_unstable"
            )
            symmetry_post_t = None
            symmetry_zero_k = None
        outcomes[float(beta)] = outcome
        for index in phase_indices:
            phase_rows[index]["state_symmetry_defect_post_T_free"] = (
                symmetry_post_t
            )
            phase_rows[index]["state_symmetry_defect_matched_zero_K"] = (
                symmetry_zero_k
            )
            phase_rows[index]["zero_epsilon"] = float(zero_epsilon)
    return bptt, estimates, phase_rows, state_rows, outcomes, precision_endpoints


class _ContextAccumulator:
    """Hold only one T/K context's cohort-mean gradient tensors."""

    def __init__(self) -> None:
        self.sums: dict[tuple[Any, ...], torch.Tensor] = {}
        self.examples: defaultdict[tuple[Any, ...], int] = defaultdict(int)
        self.batch_cosines: defaultdict[tuple[Any, ...], list[float]] = defaultdict(
            list
        )

    def add(
        self,
        key: tuple[Any, ...],
        value: torch.Tensor,
        *,
        batch_size: int,
        batch_cosine: float | None = None,
    ) -> None:
        weighted = value.detach().cpu().to(torch.float64) * int(batch_size)
        if key not in self.sums:
            self.sums[key] = weighted.clone()
        else:
            self.sums[key] += weighted
        self.examples[key] += int(batch_size)
        if batch_cosine is not None:
            self.batch_cosines[key].append(float(batch_cosine))

    def mean(self, key: tuple[Any, ...]) -> torch.Tensor:
        return self.sums[key] / self.examples[key]


def _finite_difference_precision_gates(
    accumulator: _ContextAccumulator,
    *,
    names: Sequence[str],
    betas: Sequence[float],
    expected_examples: int,
    ratio_threshold: float,
) -> dict[tuple[float, str], dict[str, Any]]:
    """Screen cohort-mean one-sided numerators against float32 roundoff."""

    epsilon = float(torch.finfo(torch.float32).eps)
    output: dict[tuple[float, str], dict[str, Any]] = {}
    for beta in betas:
        for name in names:
            zero_key = ("eqprop_zero_endpoint", float(beta), name)
            positive_key = ("eqprop_positive_endpoint", float(beta), name)
            coverage_complete = bool(
                accumulator.examples.get(zero_key, 0) == expected_examples
                and accumulator.examples.get(positive_key, 0) == expected_examples
            )
            if coverage_complete:
                zero = accumulator.mean(zero_key)
                positive = accumulator.mean(positive_key)
                numerator_l2 = float(torch.linalg.vector_norm(positive - zero))
                positive_l2 = float(torch.linalg.vector_norm(positive))
                zero_l2 = float(torch.linalg.vector_norm(zero))
                denominator = epsilon * (positive_l2 + zero_l2)
                ratio = (
                    numerator_l2 / denominator if denominator > 0.0 else None
                )
            else:
                numerator_l2 = None
                positive_l2 = None
                zero_l2 = None
                denominator = None
                ratio = None
            passed = bool(
                coverage_complete
                and numerator_l2 is not None
                and math.isfinite(numerator_l2)
                and numerator_l2 > 0.0
                and positive_l2 is not None
                and math.isfinite(positive_l2)
                and zero_l2 is not None
                and math.isfinite(zero_l2)
                and ratio is not None
                and math.isfinite(ratio)
                and ratio >= float(ratio_threshold)
            )
            output[(float(beta), name)] = {
                "coverage_complete": coverage_complete,
                "numerator_l2": numerator_l2,
                "positive_l2": positive_l2,
                "zero_l2": zero_l2,
                "denominator": denominator,
                "epsilon": epsilon,
                "ratio": ratio,
                "threshold": float(ratio_threshold),
                "passed": passed,
            }
    return output


def _attach_adjacent_beta_consistency(
    rows: Sequence[dict[str, Any]],
    accumulator: _ContextAccumulator,
    *,
    names: Sequence[str],
    beta_grid: Sequence[Mapping[str, Any]],
    expected_examples: int,
    cosine_threshold: float = 0.99,
    symmetric_norm_delta_threshold: float = 0.10,
) -> None:
    """Attach cohort-gradient consistency to the larger member of each pair."""

    by_key = {
        (float(row["beta"]), str(row["parameter_name"])): row for row in rows
    }
    for row in rows:
        row.update(
            {
                "adjacent_lower_beta": None,
                "adjacent_lower_beta_hat": None,
                "adjacent_pair_uncapped_consecutive": False,
                "adjacent_eqprop_cosine": None,
                "adjacent_symmetric_norm_delta": None,
                "adjacent_cosine_threshold": float(cosine_threshold),
                "adjacent_symmetric_norm_delta_threshold": float(
                    symmetric_norm_delta_threshold
                ),
                "adjacent_consistency_passed": False,
            }
        )
    for lower, upper in zip(beta_grid, beta_grid[1:]):
        lower_beta = float(lower["beta"])
        upper_beta = float(upper["beta"])
        eligible_pair = bool(not lower["capped"] and not upper["capped"])
        for name in names:
            row = by_key[(upper_beta, name)]
            lower_key = ("eqprop", lower_beta, name)
            upper_key = ("eqprop", upper_beta, name)
            coverage_complete = bool(
                accumulator.examples.get(lower_key, 0) == expected_examples
                and accumulator.examples.get(upper_key, 0) == expected_examples
            )
            if coverage_complete:
                lower_gradient = accumulator.mean(lower_key)
                upper_gradient = accumulator.mean(upper_key)
                lower_norm = float(torch.linalg.vector_norm(lower_gradient))
                upper_norm = float(torch.linalg.vector_norm(upper_gradient))
                denominator = lower_norm + upper_norm
                symmetric_delta = (
                    2.0 * abs(lower_norm - upper_norm) / denominator
                    if denominator > 0.0
                    else None
                )
                metrics = base._vector_metrics(
                    lower_gradient,
                    upper_gradient,
                    zero_epsilon=0.0,
                )
                cosine = metrics["cosine"]
            else:
                symmetric_delta = None
                cosine = None
            passed = bool(
                eligible_pair
                and coverage_complete
                and cosine is not None
                and math.isfinite(float(cosine))
                and float(cosine) >= float(cosine_threshold)
                and symmetric_delta is not None
                and math.isfinite(float(symmetric_delta))
                and float(symmetric_delta)
                <= float(symmetric_norm_delta_threshold)
            )
            row.update(
                {
                    "adjacent_lower_beta": lower_beta,
                    "adjacent_lower_beta_hat": float(lower["beta_hat"]),
                    "adjacent_pair_uncapped_consecutive": eligible_pair,
                    "adjacent_eqprop_cosine": cosine,
                    "adjacent_symmetric_norm_delta": symmetric_delta,
                    "adjacent_consistency_passed": passed,
                }
            )


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _parameter_rows_for_context(
    accumulator: _ContextAccumulator,
    *,
    context: Mapping[str, Any],
    names: Sequence[str],
    types: Mapping[str, str],
    betas: Sequence[float],
    expected_examples: int,
    zero_epsilon: float,
    eqprop_variant: str = "centered",
    residual_gates: Mapping[float, Mapping[str, Any]] | None = None,
    residual_gate_required: bool = False,
    beta_metadata: Mapping[float, Mapping[str, Any]] | None = None,
    precision_gates: Mapping[tuple[float, str], Mapping[str, Any]] | None = None,
    precision_gate_required: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for beta in betas:
        for name in names:
            bptt_key = ("bptt", name)
            ep_key = ("eqprop", float(beta), name)
            observed = accumulator.examples.get(ep_key, 0)
            coverage_complete = observed == expected_examples
            if ep_key in accumulator.sums:
                metrics = base._vector_metrics(
                    accumulator.mean(bptt_key),
                    accumulator.mean(ep_key),
                    zero_epsilon=zero_epsilon,
                )
            else:
                bptt_mean = accumulator.mean(bptt_key)
                bptt_norm = float(torch.linalg.vector_norm(bptt_mean))
                metrics = {
                    "element_count": int(bptt_mean.numel()),
                    "dot_product": None,
                    "bptt_l2": bptt_norm,
                    "eqprop_l2": None,
                    "cosine": None,
                    "relative_l2_error": None,
                    "norm_ratio_eqprop_over_bptt": None,
                    "bptt_rms": bptt_norm / math.sqrt(bptt_mean.numel()),
                    "eqprop_rms": None,
                    "bptt_zero_fraction": float(
                        (bptt_mean.abs() <= zero_epsilon).double().mean()
                    ),
                    "eqprop_zero_fraction": None,
                    "bptt_exact_zero_fraction": float(
                        (bptt_mean == 0.0).double().mean()
                    ),
                    "eqprop_exact_zero_fraction": None,
                }
            cosines = accumulator.batch_cosines.get(ep_key, [])
            residual_gate = (
                None
                if residual_gates is None
                else residual_gates.get(float(beta))
            )
            residual_gate_passed = (
                None if residual_gate is None else bool(residual_gate["passed"])
            )
            precision_gate = (
                None
                if precision_gates is None
                else precision_gates.get((float(beta), name))
            )
            precision_gate_passed = (
                None
                if precision_gate is None
                else bool(precision_gate["passed"])
            )
            rows.append(
                {
                    "schema": SCHEMA,
                    **context,
                    "beta": float(beta),
                    **(
                        _beta_reporting_fields(
                            float(beta), beta_metadata, context=context
                        )
                        if beta_metadata is not None
                        else {}
                    ),
                    "eqprop_variant": eqprop_variant,
                    "parameter_name": name,
                    "parameter_type": types[name],
                    "bias_excluded": True,
                    "example_count": observed,
                    "expected_example_count": expected_examples,
                    "coverage_complete": coverage_complete,
                    "residual_gate_required_for_selection": bool(
                        residual_gate_required
                    ),
                    "residual_gate_passed": residual_gate_passed,
                    "residual_failed_layer_phases": (
                        None
                        if residual_gate is None
                        else "|".join(residual_gate["failed_layer_phases"])
                    ),
                    "precision_gate_required_for_selection": bool(
                        precision_gate_required
                    ),
                    "fd_numerator_l2": (
                        None if precision_gate is None else precision_gate["numerator_l2"]
                    ),
                    "fd_positive_endpoint_l2": (
                        None if precision_gate is None else precision_gate["positive_l2"]
                    ),
                    "fd_zero_endpoint_l2": (
                        None if precision_gate is None else precision_gate["zero_l2"]
                    ),
                    "fd_roundoff_epsilon": (
                        None if precision_gate is None else precision_gate["epsilon"]
                    ),
                    "fd_roundoff_ratio": (
                        None if precision_gate is None else precision_gate["ratio"]
                    ),
                    "fd_roundoff_ratio_threshold": (
                        None if precision_gate is None else precision_gate["threshold"]
                    ),
                    "precision_gate_passed": precision_gate_passed,
                    "batch_count": len(cosines),
                    "outcome": (
                        "incomplete_phase_coverage"
                        if not coverage_complete
                        else (
                            "residual_gate_failed"
                            if residual_gate_required
                            and residual_gate_passed is not True
                            else (
                                "precision_gate_failed"
                                if precision_gate_required
                                and precision_gate_passed is not True
                                else (
                                    "ok"
                                    if metrics["cosine"] is not None
                                    else "dead_gradient"
                                )
                            )
                        )
                    ),
                    **metrics,
                    "batch_cosine_median": _quantile(cosines, 0.5),
                    "batch_cosine_q10": _quantile(cosines, 0.1),
                    "batch_cosine_q90": _quantile(cosines, 0.9),
                }
            )
    return rows


def _select_context_beta(
    rows: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    *,
    eqprop_variant: str = "centered",
) -> dict[str, Any]:
    grouped: defaultdict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["beta"])].append(row)
    normalized_rule = bool(
        eqprop_variant == "positive_one_sided"
        and any(
            str(row.get("beta_mode", "legacy_absolute"))
            == "amplification_aware_baseline_output_curvature"
            for row in rows
        )
    )

    def _endpoint_valid(beta_rows: Sequence[Mapping[str, Any]]) -> bool:
        return bool(
            beta_rows
            and all(
                bool(row["coverage_complete"])
                and (
                    not bool(
                        row.get("residual_gate_required_for_selection", False)
                    )
                    or row.get("residual_gate_passed") is True
                )
                and (
                    not bool(
                        row.get("precision_gate_required_for_selection", False)
                    )
                    or row.get("precision_gate_passed") is True
                )
                for row in beta_rows
            )
        )

    candidates: list[dict[str, Any]] = []
    for beta, beta_rows in grouped.items():
        first = beta_rows[0]
        lower_beta = first.get("adjacent_lower_beta")
        lower_rows = (
            [] if lower_beta is None else grouped.get(float(lower_beta), [])
        )
        adjacent_passed = bool(
            lower_rows
            and all(
                bool(row.get("adjacent_consistency_passed", False))
                for row in beta_rows
            )
        )
        endpoint_valid = _endpoint_valid(beta_rows)
        lower_endpoint_valid = _endpoint_valid(lower_rows)
        if normalized_rule:
            selectable = bool(
                endpoint_valid and lower_endpoint_valid and adjacent_passed
            )
        else:
            selectable = bool(
                endpoint_valid
                and all(row["cosine"] is not None for row in beta_rows)
            )
        cosines = [
            float(row["cosine"])
            for row in beta_rows
            if row["cosine"] is not None
        ]
        oracle_selectable = bool(
            endpoint_valid
            and len(cosines) == len(beta_rows)
        )
        candidates.append(
            {
                "beta": beta,
                "beta_hat": float(first.get("beta_hat", beta)),
                "beta_hat_requested": float(
                    first.get("beta_hat_requested", beta)
                ),
                "beta_effective": float(first.get("beta_effective", beta)),
                "beta_capped": bool(first.get("beta_capped", False)),
                "selectable": selectable,
                "endpoint_gates_passed": endpoint_valid,
                "lower_endpoint_gates_passed": lower_endpoint_valid,
                "adjacent_consistency_passed": adjacent_passed,
                "adjacent_lower_beta": lower_beta,
                "adjacent_lower_beta_hat": first.get("adjacent_lower_beta_hat"),
                "oracle_selectable": oracle_selectable,
                "worst_layer_eqprop_cosine": (
                    min(cosines) if len(cosines) == len(beta_rows) else None
                ),
                "mean_layer_eqprop_cosine": (
                    float(np.mean(cosines))
                    if len(cosines) == len(beta_rows)
                    else None
                ),
                "worst_layer": (
                    min(beta_rows, key=lambda row: float(row["cosine"]))[
                        "parameter_name"
                    ]
                    if len(cosines) == len(beta_rows)
                    else None
                ),
            }
        )
    selectable = [candidate for candidate in candidates if candidate["selectable"]]
    if selectable:
        if normalized_rule:
            selected = sorted(
                selectable,
                key=lambda value: (
                    float(value["beta_hat"]),
                    float(value["beta"]),
                ),
            )[0]
        else:
            selected = sorted(
                selectable,
                key=lambda value: (
                    -float(value["worst_layer_eqprop_cosine"]),
                    float(value["beta"]),
                ),
            )[0]
    else:
        selected = {
            "beta": None,
            "beta_hat": None,
            "beta_hat_requested": None,
            "beta_effective": None,
            "beta_capped": None,
            "selectable": False,
            "worst_layer_eqprop_cosine": None,
            "mean_layer_eqprop_cosine": None,
            "worst_layer": None,
            "adjacent_lower_beta": None,
            "adjacent_lower_beta_hat": None,
        }
    oracle_candidates = [
        candidate for candidate in candidates if candidate["oracle_selectable"]
    ]
    oracle = (
        sorted(
            oracle_candidates,
            key=lambda value: (
                -float(value["worst_layer_eqprop_cosine"]),
                float(value["beta_hat"]),
                float(value["beta"]),
            ),
        )[0]
        if oracle_candidates
        else None
    )
    return {
        "schema": SCHEMA,
        **context,
        "eqprop_variant": eqprop_variant,
        "selected_beta": selected["beta"],
        "selected_beta_hat": selected["beta_hat"],
        "selected_beta_hat_requested": selected["beta_hat_requested"],
        "selected_beta_effective": selected["beta_effective"],
        "selected_beta_capped": selected["beta_capped"],
        "selected_beta_over_case_init_b0": (
            None
            if selected["beta"] is None
            else float(selected["beta"])
            / float(context["init_output_curvature_scale_b0"])
        )
        if context.get("init_output_curvature_scale_b0") is not None
        else None,
        "selected_beta_over_checkpoint_b0": (
            None
            if selected["beta"] is None
            else float(selected["beta"])
            / float(context["checkpoint_output_curvature_scale_b0"])
        )
        if context.get("checkpoint_output_curvature_scale_b0") is not None
        else None,
        "selected_pair_lower_beta": selected.get("adjacent_lower_beta"),
        "selected_pair_lower_beta_hat": selected.get("adjacent_lower_beta_hat"),
        "worst_layer_eqprop_cosine": selected["worst_layer_eqprop_cosine"],
        "mean_layer_eqprop_cosine": selected["mean_layer_eqprop_cosine"],
        "worst_layer_centered_cosine": (
            selected["worst_layer_eqprop_cosine"]
            if eqprop_variant == "centered"
            else None
        ),
        "mean_layer_centered_cosine": (
            selected["mean_layer_eqprop_cosine"]
            if eqprop_variant == "centered"
            else None
        ),
        "worst_layer_positive_one_sided_cosine": (
            selected["worst_layer_eqprop_cosine"]
            if eqprop_variant == "positive_one_sided"
            else None
        ),
        "mean_layer_positive_one_sided_cosine": (
            selected["mean_layer_eqprop_cosine"]
            if eqprop_variant == "positive_one_sided"
            else None
        ),
        "worst_layer": selected["worst_layer"],
        "selectable_beta_count": len(selectable),
        "tested_beta_count": len(candidates),
        "selection_rule": (
            "lowest_uncapped_adjacent_consistent_pair_choose_larger"
            if normalized_rule
            else "maximize_worst_layer_then_smaller_beta"
        ),
        "oracle_selected_beta": None if oracle is None else oracle["beta"],
        "oracle_selected_beta_hat": None if oracle is None else oracle["beta_hat"],
        "oracle_selected_beta_hat_requested": (
            None if oracle is None else oracle["beta_hat_requested"]
        ),
        "oracle_selected_beta_effective": (
            None if oracle is None else oracle["beta_effective"]
        ),
        "oracle_beta_over_case_init_b0": (
            None
            if oracle is None
            or context.get("init_output_curvature_scale_b0") is None
            else float(oracle["beta"])
            / float(context["init_output_curvature_scale_b0"])
        ),
        "oracle_beta_over_checkpoint_b0": (
            None
            if oracle is None
            or context.get("checkpoint_output_curvature_scale_b0") is None
            else float(oracle["beta"])
            / float(context["checkpoint_output_curvature_scale_b0"])
        ),
        "oracle_worst_layer_bptt_cosine": (
            None if oracle is None else oracle["worst_layer_eqprop_cosine"]
        ),
        "outcome": (
            "selected"
            if selectable
            else (
                "no_passing_adjacent_beta_pair"
                if normalized_rule
                else "no_finite_all_layer_beta"
            )
        ),
    }


def _native_gradient_archive(
    path: Path,
    accumulator: _ContextAccumulator,
    *,
    names: Sequence[str],
    betas: Sequence[float],
    expected_examples: int,
    metadata: Mapping[str, Any],
    eqprop_variant: str = "centered",
) -> None:
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(
            json.dumps(dict(metadata), sort_keys=True, allow_nan=False)
        )
    }
    for name in names:
        arrays[f"bptt__{name}"] = accumulator.mean(("bptt", name)).numpy()
        for beta in betas:
            key = ("eqprop", float(beta), name)
            if accumulator.examples.get(key, 0) != expected_examples:
                continue
            beta_key = format(float(beta), ".12g").replace(".", "p")
            arrays[f"eqprop_{eqprop_variant}_beta_{beta_key}__{name}"] = accumulator.mean(
                key
            ).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _summarize_state_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_batch_count: int | None = None,
) -> list[dict[str, Any]]:
    key_fields = (
        "architecture",
        "scheme",
        "checkpoint_role",
        "T",
        "K",
        "native_T",
        "native_K",
        "native_context",
        "shared_anchor_context",
        "beta",
        "phase",
        "reference_kind",
        "state_layer_name",
        "state_layer_type",
        "state_shape",
    )
    grouped: defaultdict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(row)
    summaries: list[dict[str, Any]] = []
    for key, values in grouped.items():
        finite_values = [
            row
            for row in values
            if row.get("relative_displacement") is not None
            and row.get("delta_squared_sum") is not None
        ]
        relative = [
            float(row["relative_displacement"])
            for row in finite_values
        ]
        delta_squared = sum(
            float(row["delta_squared_sum"]) for row in finite_values
        )
        reference_squared = sum(
            float(row["reference_squared_sum"]) for row in finite_values
        )
        transitions = sum(
            int(row["active_set_transition_count"])
            for row in finite_values
            if row.get("active_set_transition_count") is not None
        )
        constrained = sum(
            int(row["constrained_element_count"]) for row in finite_values
        )
        declared_expected = (
            int(expected_batch_count)
            if expected_batch_count is not None
            else len(values)
        )
        coverage_complete = bool(
            len(values) == declared_expected
            and len(finite_values) == declared_expected
            and all(str(row.get("outcome")) == "ok" for row in values)
        )
        epsilon = max(
            float(row.get("reference_norm_epsilon", 0.0)) for row in values
        )
        pooled_displacement_l2 = math.sqrt(delta_squared) if coverage_complete else None
        pooled_reference_l2 = (
            math.sqrt(reference_squared) if coverage_complete else None
        )
        summaries.append(
            {
                "schema": SCHEMA,
                **dict(zip(key_fields, key, strict=True)),
                "eqprop_variant": str(values[0].get("eqprop_variant", "centered")),
                "beta_hat_requested": values[0].get("beta_hat_requested"),
                "beta_hat": values[0].get("beta_hat"),
                "beta_effective": values[0].get("beta_effective"),
                "beta_capped": values[0].get("beta_capped"),
                "beta_over_case_init_b0": values[0].get(
                    "beta_over_case_init_b0"
                ),
                "beta_over_checkpoint_b0": values[0].get(
                    "beta_over_checkpoint_b0"
                ),
                "batch_count": len(values),
                "expected_batch_count": declared_expected,
                "finite_relative_count": len(relative),
                "coverage_complete": coverage_complete,
                "outcome": "ok" if coverage_complete else "incomplete_or_unstable",
                "phase_outcomes": "|".join(
                    sorted({str(row.get("outcome")) for row in values})
                ),
                "pooled_delta_squared_sum": (
                    delta_squared if coverage_complete else None
                ),
                "pooled_reference_squared_sum": (
                    reference_squared if coverage_complete else None
                ),
                "pooled_displacement_l2": pooled_displacement_l2,
                "pooled_reference_l2": pooled_reference_l2,
                "reference_norm_epsilon": epsilon,
                "pooled_relative_displacement": (
                    pooled_displacement_l2 / max(pooled_reference_l2, epsilon)
                    if coverage_complete
                    and pooled_displacement_l2 is not None
                    and pooled_reference_l2 is not None
                    else None
                ),
                "batch_relative_median": _quantile(relative, 0.5),
                "batch_relative_q10": _quantile(relative, 0.1),
                "batch_relative_q90": _quantile(relative, 0.9),
                "batch_relative_min": min(relative) if relative else None,
                "batch_relative_max": max(relative) if relative else None,
                "pooled_active_set_transition_fraction": (
                    transitions / constrained
                    if coverage_complete and constrained
                    else (0.0 if coverage_complete else None)
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda row: tuple(str(row[field]) for field in key_fields),
    )


def _shared_anchor(architecture: str) -> tuple[int, int]:
    return {"conv1": (4, 4), "conv2": (6, 6), "conv3": (8, 8)}[
        architecture
    ]


def _native_or_shared_rows(
    rows: Sequence[Mapping[str, Any]], *, anchor: str
) -> list[Mapping[str, Any]]:
    if anchor == "native":
        return [row for row in rows if bool(row["native_context"])]
    if anchor == "shared":
        return [row for row in rows if bool(row["shared_anchor_context"])]
    raise ValueError(anchor)


def _plot_cosine_beta(
    rows: Sequence[Mapping[str, Any]],
    *,
    architecture: str,
    anchor: str,
    path: Path,
    eqprop_variant: str = "centered",
) -> None:
    selected = [
        row
        for row in _native_or_shared_rows(rows, anchor=anchor)
        if str(row["architecture"]) == architecture
    ]
    schemes = ("baseline", "ours", "legacy")
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 12), sharex=True, sharey=True)
    for row_index, scheme in enumerate(schemes):
        for column_index, role in enumerate(ROLES):
            axis = axes[row_index, column_index]
            panel = [
                row
                for row in selected
                if str(row["scheme"]) == scheme
                and str(row["checkpoint_role"]) == role
            ]
            names = list(dict.fromkeys(str(row["parameter_name"]) for row in panel))
            for name in names:
                layer_rows = sorted(
                    [row for row in panel if str(row["parameter_name"]) == name],
                    key=lambda row: float(row.get("beta_hat", row["beta"])),
                )
                axis.plot(
                    [
                        float(row.get("beta_hat", row["beta"]))
                        for row in layer_rows
                    ],
                    [
                        float(row["cosine"])
                        if row["cosine"] is not None
                        else np.nan
                        for row in layer_rows
                    ],
                    marker="o",
                    linewidth=1.5,
                    markersize=4,
                    label=name.replace("Weight_", ""),
                )
            axis.axhline(0.0, color="0.55", linewidth=0.8)
            axis.axhline(0.9, color="0.75", linewidth=0.8, linestyle="--")
            axis.set_xscale("log")
            axis.set_ylim(-1.05, 1.05)
            axis.grid(alpha=0.25)
            axis.set_title(f"{scheme} | {role.replace('_', ' ')}")
            if column_index == 0:
                axis.set_ylabel("EqProp–BPTT cosine")
            if row_index == 2:
                axis.set_xlabel(r"effective normalized $\hat{\beta}$")
            axis.legend(fontsize=8, loc="lower left")
    fig.suptitle(
        f"{architecture.upper()}: {_variant_title(eqprop_variant)} EqProp by beta "
        f"({anchor} T/K anchor)",
        y=0.995,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _matrix_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
) -> tuple[list[int], list[int], np.ndarray]:
    t_values = sorted({int(row["T"]) for row in rows})
    k_values = sorted({int(row["K"]) for row in rows})
    matrix = np.full((len(t_values), len(k_values)), np.nan, dtype=np.float64)
    for row in rows:
        value = row.get(value_field)
        if value is None:
            continue
        matrix[t_values.index(int(row["T"])), k_values.index(int(row["K"]))] = float(
            value
        )
    return t_values, k_values, matrix


def _draw_heatmap(
    axis: Any,
    matrix: np.ndarray,
    t_values: Sequence[int],
    k_values: Sequence[int],
    *,
    title: str,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> Any:
    image = axis.imshow(matrix, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="coolwarm")
    axis.set_xticks(range(len(k_values)), [str(value) for value in k_values])
    axis.set_yticks(range(len(t_values)), [str(value) for value in t_values])
    axis.set_xlabel("K")
    axis.set_ylabel("T")
    axis.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if math.isfinite(float(matrix[i, j])):
                axis.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if abs(matrix[i, j]) < 0.75 else "white",
                )
    return image


def _plot_tk_maximin(
    context_rows: Sequence[Mapping[str, Any]],
    *,
    architecture: str,
    role: str,
    path: Path,
    eqprop_variant: str = "centered",
) -> None:
    schemes = ("baseline", "ours", "legacy")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    image = None
    for index, scheme in enumerate(schemes):
        panel = [
            row
            for row in context_rows
            if str(row["architecture"]) == architecture
            and str(row["checkpoint_role"]) == role
            and str(row["scheme"]) == scheme
        ]
        t_values, k_values, matrix = _matrix_from_rows(
            panel, value_field="worst_layer_eqprop_cosine"
        )
        image = _draw_heatmap(
            axes[0, index],
            matrix,
            t_values,
            k_values,
            title=(
                f"{scheme}: best-beta maximin"
                if eqprop_variant == "centered"
                else f"{scheme}: selected-pair beta"
            ),
        )
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), label="worst weight-layer cosine", shrink=0.8)
    selection_label = (
        "centered cosine-maximin"
        if eqprop_variant == "centered"
        else "adjacent-precision selection (color shows BPTT cosine)"
    )
    fig.suptitle(
        f"{architecture.upper()} | {role.replace('_', ' ')}: "
        f"T/K dependence, {selection_label}"
    )
    fig.subplots_adjust(left=0.06, right=0.92, top=0.84, bottom=0.13, wspace=0.28)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fixed_beta_layer_rows(
    parameter_rows: Sequence[Mapping[str, Any]],
    context_rows: Sequence[Mapping[str, Any]],
    *,
    architecture: str,
    scheme: str,
    role: str,
) -> tuple[float | None, list[Mapping[str, Any]]]:
    shared = next(
        (
            row
            for row in context_rows
            if str(row["architecture"]) == architecture
            and str(row["scheme"]) == scheme
            and str(row["checkpoint_role"]) == role
            and bool(row["shared_anchor_context"])
        ),
        None,
    )
    if shared is None:
        shared = next(
            (
                row
                for row in context_rows
                if str(row["architecture"]) == architecture
                and str(row["scheme"]) == scheme
                and str(row["checkpoint_role"]) == role
                and bool(row["native_context"])
            ),
            None,
        )
    beta = None if shared is None else shared["selected_beta"]
    if beta is None:
        # A dedicated Stage-B config may intentionally declare exactly one
        # beta and perform no adjacent-beta selection.  In that case the sole
        # measured beta is the fixed scientific anchor by construction.
        candidates = {
            float(row["beta"])
            for row in parameter_rows
            if str(row["architecture"]) == architecture
            and str(row["scheme"]) == scheme
            and str(row["checkpoint_role"]) == role
        }
        if len(candidates) == 1:
            beta = next(iter(candidates))
    if beta is None:
        return None, []
    return float(beta), [
        row
        for row in parameter_rows
        if str(row["architecture"]) == architecture
        and str(row["scheme"]) == scheme
        and str(row["checkpoint_role"]) == role
        and float(row["beta"]) == float(beta)
    ]


def _plot_tk_by_layer_fixed_beta(
    parameter_rows: Sequence[Mapping[str, Any]],
    context_rows: Sequence[Mapping[str, Any]],
    *,
    architecture: str,
    role: str,
    path: Path,
    eqprop_variant: str = "centered",
) -> None:
    schemes = ("baseline", "ours", "legacy")
    names = list(
        dict.fromkeys(
            str(row["parameter_name"])
            for row in parameter_rows
            if str(row["architecture"]) == architecture
        )
    )
    fig, axes = plt.subplots(
        len(schemes), len(names), figsize=(4.1 * len(names), 4.0 * len(schemes)), squeeze=False
    )
    image = None
    for row_index, scheme in enumerate(schemes):
        beta, fixed_rows = _fixed_beta_layer_rows(
            parameter_rows,
            context_rows,
            architecture=architecture,
            scheme=scheme,
            role=role,
        )
        for column_index, name in enumerate(names):
            panel = [row for row in fixed_rows if str(row["parameter_name"]) == name]
            if panel:
                t_values, k_values, matrix = _matrix_from_rows(panel, value_field="cosine")
                image = _draw_heatmap(
                    axes[row_index, column_index],
                    matrix,
                    t_values,
                    k_values,
                    title=f"{scheme} | {name.replace('Weight_', '')}\nβ={beta:g}",
                )
            else:
                axes[row_index, column_index].axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="EqProp–BPTT cosine", shrink=0.65)
    fig.suptitle(
        f"{architecture.upper()} | {role.replace('_', ' ')}: "
        f"{_variant_title(eqprop_variant)} EqProp T/K at fixed beta",
        y=0.995,
    )
    fig.subplots_adjust(left=0.06, right=0.93, top=0.92, bottom=0.06, hspace=0.38, wspace=0.30)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_displacement_beta(
    state_summary: Sequence[Mapping[str, Any]],
    *,
    architecture: str,
    anchor: str,
    reference_kind: str,
    state_layer_name: str,
    path: Path,
    eqprop_variant: str = "centered",
) -> None:
    phases = _nudged_phase_names(eqprop_variant)
    selected = [
        row
        for row in _native_or_shared_rows(state_summary, anchor=anchor)
        if str(row["architecture"]) == architecture
        and str(row["reference_kind"]) == reference_kind
        and str(row["state_layer_name"]) == state_layer_name
        and str(row["phase"]) in phases
    ]
    schemes = ("baseline", "ours", "legacy")
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 12), sharex=True)
    for row_index, scheme in enumerate(schemes):
        for column_index, role in enumerate(ROLES):
            axis = axes[row_index, column_index]
            for phase, marker in (("negative", "o"), ("positive", "s")):
                if phase not in phases:
                    continue
                panel = sorted(
                    [
                        row
                        for row in selected
                        if str(row["scheme"]) == scheme
                        and str(row["checkpoint_role"]) == role
                        and str(row["phase"]) == phase
                    ],
                    key=lambda row: float(row.get("beta_hat", row["beta"])),
                )
                if not panel:
                    continue
                axis.plot(
                    [
                        float(row.get("beta_hat", row["beta"]))
                        for row in panel
                    ],
                    [
                        float(row["pooled_relative_displacement"])
                        if bool(row.get("coverage_complete"))
                        and row.get("pooled_relative_displacement") is not None
                        and float(row["pooled_relative_displacement"]) > 0.0
                        else np.nan
                        for row in panel
                    ],
                    marker=marker,
                    linewidth=1.5,
                    label=phase,
                )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(alpha=0.25, which="both")
            axis.set_title(f"{scheme} | {role.replace('_', ' ')}")
            if column_index == 0:
                axis.set_ylabel("pooled relative displacement")
            if row_index == 2:
                axis.set_xlabel(r"effective normalized $\hat{\beta}$")
            axis.legend(fontsize=8)
    fig.suptitle(
        f"{architecture.upper()}: {_variant_title(eqprop_variant)} EqProp "
        f"{state_layer_name} displacement vs beta\n"
        f"anchor={anchor}, reference={reference_kind}",
        y=0.995,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fixed_beta_context_rows(
    parameter_rows: Sequence[Mapping[str, Any]],
    context_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = {
        (
            str(row["architecture"]),
            str(row["scheme"]),
            str(row["checkpoint_role"]),
        )
        for row in context_rows
    }
    for architecture, scheme, role in sorted(groups):
        group_contexts = [
            row
            for row in context_rows
            if str(row["architecture"]) == architecture
            and str(row["scheme"]) == scheme
            and str(row["checkpoint_role"]) == role
        ]
        shared = next(
            (row for row in group_contexts if bool(row["shared_anchor_context"])),
            next(row for row in group_contexts if bool(row["native_context"])),
        )
        beta_source = (
            "shared_anchor_beta_selection"
            if bool(shared["shared_anchor_context"])
            else "native_selection_no_shared_anchor_context"
        )
        beta = shared["selected_beta"]
        fixed_metadata: Mapping[str, Any] = shared
        if beta is None:
            candidates = [
                row
                for row in parameter_rows
                if str(row["architecture"]) == architecture
                and str(row["scheme"]) == scheme
                and str(row["checkpoint_role"]) == role
            ]
            distinct_betas = {float(row["beta"]) for row in candidates}
            if len(distinct_betas) == 1 and candidates:
                beta = next(iter(distinct_betas))
                fixed_metadata = candidates[0]
                beta_source = "single_declared_fixed_beta"
        eqprop_variant = str(shared.get("eqprop_variant", "centered"))
        coordinates = {
            (int(row["T"]), int(row["K"]))
            for row in context_rows
            if str(row["architecture"]) == architecture
            and str(row["scheme"]) == scheme
            and str(row["checkpoint_role"]) == role
        }
        for t_value, k_value in sorted(coordinates):
            rows = [
                row
                for row in parameter_rows
                if str(row["architecture"]) == architecture
                and str(row["scheme"]) == scheme
                and str(row["checkpoint_role"]) == role
                and int(row["T"]) == t_value
                and int(row["K"]) == k_value
                and beta is not None
                and float(row["beta"]) == float(beta)
            ]
            valid = bool(rows) and all(
                bool(row["coverage_complete"]) and row["cosine"] is not None
                for row in rows
            )
            cosines = [float(row["cosine"]) for row in rows if row["cosine"] is not None]
            native = bool(rows[0]["native_context"]) if rows else False
            shared_anchor = bool(rows[0]["shared_anchor_context"]) if rows else False
            worst_cosine = min(cosines) if valid else None
            mean_cosine = float(np.mean(cosines)) if valid else None
            output.append(
                {
                    "schema": SCHEMA,
                    "architecture": architecture,
                    "scheme": scheme,
                    "checkpoint_role": role,
                    "eqprop_variant": eqprop_variant,
                    "T": t_value,
                    "K": k_value,
                    "fixed_beta": beta,
                    "fixed_beta_hat": (
                        shared.get("selected_beta_hat")
                        if shared.get("selected_beta_hat") is not None
                        else fixed_metadata.get("beta_hat")
                    ),
                    "fixed_beta_hat_requested": (
                        shared.get("selected_beta_hat_requested")
                        if shared.get("selected_beta_hat_requested") is not None
                        else fixed_metadata.get("beta_hat_requested")
                    ),
                    "fixed_beta_source": beta_source,
                    "native_context": native,
                    "shared_anchor_context": shared_anchor,
                    "layer_count": len(rows),
                    "coverage_complete": valid,
                    "worst_layer_eqprop_cosine": worst_cosine,
                    "mean_layer_eqprop_cosine": mean_cosine,
                    "worst_layer_centered_cosine": (
                        worst_cosine if eqprop_variant == "centered" else None
                    ),
                    "mean_layer_centered_cosine": (
                        mean_cosine if eqprop_variant == "centered" else None
                    ),
                    "worst_layer_positive_one_sided_cosine": (
                        worst_cosine
                        if eqprop_variant == "positive_one_sided"
                        else None
                    ),
                    "mean_layer_positive_one_sided_cosine": (
                        mean_cosine
                        if eqprop_variant == "positive_one_sided"
                        else None
                    ),
                    "worst_layer": (
                        min(rows, key=lambda row: float(row["cosine"]))[
                            "parameter_name"
                        ]
                        if valid
                        else None
                    ),
                    "outcome": "ok" if valid else "incomplete_or_dead",
                }
            )
    return output


def _worst_eqprop_cosine(row: Mapping[str, Any]) -> Any:
    if "worst_layer_eqprop_cosine" in row:
        return row["worst_layer_eqprop_cosine"]
    return row.get("worst_layer_centered_cosine")


def _study_summary_rows(
    context_rows: Sequence[Mapping[str, Any]],
    fixed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups = {
        (
            str(row["architecture"]),
            str(row["scheme"]),
            str(row["checkpoint_role"]),
        )
        for row in context_rows
    }
    output: list[dict[str, Any]] = []
    for architecture, scheme, role in sorted(groups):
        selected = [
            row
            for row in context_rows
            if str(row["architecture"]) == architecture
            and str(row["scheme"]) == scheme
            and str(row["checkpoint_role"]) == role
        ]
        native = next(row for row in selected if bool(row["native_context"]))
        shared = next(
            (row for row in selected if bool(row["shared_anchor_context"])),
            native,
        )
        eqprop_variant = str(shared.get("eqprop_variant", "centered"))
        eligible = [
            row
            for row in selected
            if _worst_eqprop_cosine(row) is not None
            and row.get("selected_beta") is not None
        ]
        best = (
            sorted(
                eligible,
                key=lambda row: (
                    -float(_worst_eqprop_cosine(row)),
                    int(row["T"]),
                    int(row["K"]),
                    float(row["selected_beta"]),
                ),
            )[0]
            if eligible
            else None
        )
        fixed_all = [
            row
            for row in fixed_rows
            if str(row["architecture"]) == architecture
            and str(row["scheme"]) == scheme
            and str(row["checkpoint_role"]) == role
        ]
        fixed_anchor = next(
            (row for row in fixed_all if bool(row["shared_anchor_context"])),
            next((row for row in fixed_all if bool(row["native_context"])), None),
        )
        shared_t, shared_k = _shared_anchor(architecture)
        vary_t = [row for row in fixed_all if int(row["K"]) == shared_k]
        vary_k = [row for row in fixed_all if int(row["T"]) == shared_t]

        def _slice_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            valid = [
                row
                for row in rows
                if bool(row["coverage_complete"])
                and _worst_eqprop_cosine(row) is not None
            ]
            values = [
                float(_worst_eqprop_cosine(row)) for row in valid
            ]
            complete = bool(rows) and len(valid) == len(rows)
            return {
                "total": len(rows),
                "valid": len(valid),
                "complete": complete,
                "range": max(values) - min(values) if complete else None,
            }

        t_stats = _slice_stats(vary_t)
        k_stats = _slice_stats(vary_k)
        full_stats = _slice_stats(fixed_all)

        output.append(
            {
                "schema": SCHEMA,
                "architecture": architecture,
                "scheme": scheme,
                "checkpoint_role": role,
                "eqprop_variant": eqprop_variant,
                "native_T": int(native["T"]),
                "native_K": int(native["K"]),
                "native_selected_beta": native["selected_beta"],
                "native_selected_beta_hat": native.get("selected_beta_hat"),
                "native_worst_layer_cosine": _worst_eqprop_cosine(native),
                "shared_T": int(shared["T"]),
                "shared_K": int(shared["K"]),
                "shared_selected_beta": shared["selected_beta"],
                "shared_selected_beta_hat": shared.get("selected_beta_hat"),
                "shared_anchor_available": any(
                    bool(row["shared_anchor_context"]) for row in selected
                ),
                "shared_worst_layer_cosine": _worst_eqprop_cosine(shared),
                "best_grid_T": int(best["T"]) if best else None,
                "best_grid_K": int(best["K"]) if best else None,
                "best_grid_beta": best["selected_beta"] if best else None,
                "best_grid_worst_layer_cosine": (
                    _worst_eqprop_cosine(best) if best else None
                ),
                "fixed_beta": (
                    fixed_anchor.get("fixed_beta")
                    if fixed_anchor is not None
                    else shared["selected_beta"]
                ),
                "fixed_beta_hat": (
                    fixed_anchor.get("fixed_beta_hat")
                    if fixed_anchor is not None
                    else shared.get("selected_beta_hat")
                ),
                "fixed_beta_vary_T_valid_coordinates": t_stats["valid"],
                "fixed_beta_vary_T_total_coordinates": t_stats["total"],
                "fixed_beta_vary_T_complete": t_stats["complete"],
                "fixed_beta_vary_T_worst_cosine_range": t_stats["range"],
                "fixed_beta_vary_K_valid_coordinates": k_stats["valid"],
                "fixed_beta_vary_K_total_coordinates": k_stats["total"],
                "fixed_beta_vary_K_complete": k_stats["complete"],
                "fixed_beta_vary_K_worst_cosine_range": k_stats["range"],
                "fixed_beta_full_grid_valid_coordinates": full_stats["valid"],
                "fixed_beta_full_grid_total_coordinates": full_stats["total"],
                "fixed_beta_full_grid_complete": full_stats["complete"],
                "fixed_beta_full_grid_worst_cosine_range": full_stats["range"],
            }
        )
    return output


def _exact_coverage_guards(
    *,
    context_rows: Sequence[Mapping[str, Any]],
    parameter_rows: Sequence[Mapping[str, Any]],
    phase_rows: Sequence[Mapping[str, Any]],
    state_summary: Sequence[Mapping[str, Any]],
    residual_summary: Sequence[Mapping[str, Any]],
    beta_grids_by_case: Mapping[
        tuple[str, str], Sequence[Mapping[str, Any]]
    ],
    weight_names_by_architecture: Mapping[str, Sequence[str]],
    state_layer_names_by_architecture: Mapping[str, Sequence[str]],
    batch_count: int,
    eqprop_variant: str = "centered",
) -> dict[str, Any]:
    phases = _nudged_phase_names(eqprop_variant)
    context_fields = ("architecture", "scheme", "checkpoint_role", "T", "K")
    contexts = {
        tuple(row[field] for field in context_fields) for row in context_rows
    }
    def _context_betas(context: Sequence[Any]) -> list[float]:
        architecture = str(context[0])
        scheme = str(context[1])
        return [
            float(row["beta"])
            for row in beta_grids_by_case[(scheme, architecture)]
        ]

    expected_parameter = {
        (*context, float(beta), name)
        for context in contexts
        for beta in _context_betas(context)
        for name in weight_names_by_architecture[str(context[0])]
    }
    observed_parameter = {
        (
            *(row[field] for field in context_fields),
            float(row["beta"]),
            str(row["parameter_name"]),
        )
        for row in parameter_rows
    }
    expected_phase = {
        (*context, batch_index, float(beta), phase)
        for context in contexts
        for batch_index in range(batch_count)
        for beta in _context_betas(context)
        for phase in phases
    }
    observed_phase = {
        (
            *(row[field] for field in context_fields),
            int(row["batch_index"]),
            float(row["beta"]),
            str(row["phase"]),
        )
        for row in phase_rows
        if str(row["phase"]) in phases
    }
    expected_state = {
        (*context, float(beta), phase, reference_kind, layer_name)
        for context in contexts
        for beta in _context_betas(context)
        for phase in phases
        for reference_kind in ("post_T_free", "matched_zero_K")
        for layer_name in state_layer_names_by_architecture[str(context[0])]
    }
    observed_state = {
        (
            *(row[field] for field in context_fields),
            float(row["beta"]),
            str(row["phase"]),
            str(row["reference_kind"]),
            str(row["state_layer_name"]),
        )
        for row in state_summary
        if str(row["phase"]) in phases
    }
    expected_residual = {
        (*context, phase, float(beta), layer_name)
        for context in contexts
        for phase, beta in (
            [("post_T_free", 0.0), ("zero", 0.0)]
            + [
                (phase, float(beta))
                for beta in _context_betas(context)
                for phase in phases
            ]
        )
        for layer_name in state_layer_names_by_architecture[str(context[0])]
        if layer_name != "__all__"
    }
    observed_residual = {
        (
            *(row[field] for field in context_fields),
            str(row["phase"]),
            float(row["beta"]),
            str(row["state_layer_name"]),
        )
        for row in residual_summary
    }
    return {
        "expected_parameter_key_count": len(expected_parameter),
        "observed_parameter_key_count": len(observed_parameter),
        "parameter_keys_exact": observed_parameter == expected_parameter,
        "missing_parameter_key_count": len(expected_parameter - observed_parameter),
        "extra_parameter_key_count": len(observed_parameter - expected_parameter),
        "expected_phase_key_count": len(expected_phase),
        "observed_phase_key_count": len(observed_phase),
        "phase_keys_exact": observed_phase == expected_phase,
        "missing_phase_key_count": len(expected_phase - observed_phase),
        "extra_phase_key_count": len(observed_phase - expected_phase),
        "expected_state_key_count": len(expected_state),
        "observed_state_key_count": len(observed_state),
        "state_keys_exact": observed_state == expected_state,
        "missing_state_key_count": len(expected_state - observed_state),
        "extra_state_key_count": len(observed_state - expected_state),
        "state_rows_have_expected_batch_count": all(
            int(row["batch_count"]) == batch_count
            for row in state_summary
            if str(row["phase"]) in phases
        ),
        "expected_residual_key_count": len(expected_residual),
        "observed_residual_key_count": len(observed_residual),
        "residual_keys_exact": observed_residual == expected_residual,
        "missing_residual_key_count": len(expected_residual - observed_residual),
        "extra_residual_key_count": len(observed_residual - expected_residual),
        "residual_rows_with_complete_sample_coverage": sum(
            bool(row["coverage_complete"]) for row in residual_summary
        ),
        "residual_row_count": len(residual_summary),
        "residual_gate_pass_count": sum(
            bool(row["gate_passed"]) for row in residual_summary
        ),
        "residual_gate_fail_count": sum(
            not bool(row["gate_passed"]) for row in residual_summary
        ),
        "residual_uniform_equilibrium_count": sum(
            bool(row["uniform_equilibrium"]) for row in residual_summary
        ),
        "residual_hard_failure_count": sum(
            bool(row["hard_failure"]) for row in residual_summary
        ),
        "finite_difference_precision_diagnostics_complete": all(
            row.get("fd_roundoff_epsilon") is not None
            and row.get("fd_roundoff_ratio_threshold") is not None
            and row.get("precision_gate_passed") is not None
            for row in parameter_rows
        ),
        "amplification_scaled_beta_metadata_complete": all(
            row.get("beta_hat_requested") is not None
            and row.get("beta_hat") is not None
            and row.get("beta_effective") is not None
            and row.get("beta_capped") is not None
            for row in parameter_rows
        ),
        "declared_gradient_variants": [eqprop_variant],
        "reported_gradient_variants": sorted(
            {str(row["eqprop_variant"]) for row in parameter_rows}
        ),
    }


def _completion_criteria_met(completion: Mapping[str, Any]) -> bool:
    """Require positive guards while preserving intentional false sentinels."""

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
    config: Mapping[str, Any],
    cohort: Mapping[str, Any],
    study_rows: Sequence[Mapping[str, Any]],
    parameter_count: int,
    state_count: int,
    context_count: int,
    smoke: bool,
) -> None:
    eqprop_variant = _eqprop_variant(config["gradient_contract"])
    variant_title = _variant_title(eqprop_variant)
    phase_notation = "s_{±β,K}" if eqprop_variant == "centered" else "s_{+β,K}"
    phase_description = (
        "positive/negative"
        if eqprop_variant == "centered"
        else "positive-only"
    )
    anchor_table = []
    for row in study_rows:
        anchor_table.append(
            (
                str(row["architecture"]),
                str(row["scheme"]),
                str(row["checkpoint_role"]).replace("_", " "),
                f"{row['native_T']}/{row['native_K']}",
                "—"
                if row["native_selected_beta"] is None
                else f"{float(row['native_selected_beta']):g}",
                "—"
                if row.get("native_selected_beta_hat") is None
                else f"{float(row['native_selected_beta_hat']):g}",
                "—"
                if row["native_worst_layer_cosine"] is None
                else f"{float(row['native_worst_layer_cosine']):.6f}",
                "—"
                if row["best_grid_T"] is None
                else f"{row['best_grid_T']}/{row['best_grid_K']}",
                "—"
                if row["best_grid_worst_layer_cosine"] is None
                else f"{float(row['best_grid_worst_layer_cosine']):.6f}",
            )
        )
    lines = [
        "# EqProp/BPTT beta, T/K, and state-displacement replay",
        "",
        f"Evidence class: `{config['evidence_class']}`; smoke: `{str(smoke).lower()}`.",
        f"EqProp estimator: `{eqprop_variant}` ({variant_title}).",
        "",
        "This is a read-only ordinary-MNIST diagnostic. Biases remain active in the states but bias gradients are excluded. No optimizer step or official-test read occurs.",
        "",
        (
            "## Native and best-grid centered maximin summaries"
            if eqprop_variant == "centered"
            else "## Native adjacent-precision selections and separate BPTT-cosine diagnostics"
        ),
        "",
        base._markdown_table(
            (
                "Architecture",
                "Scheme",
                "Checkpoint",
                "Native T/K",
                "Native beta",
                "Native effective beta-hat",
                "Native worst cosine",
                "Best grid T/K",
                "Best grid worst cosine",
            ),
            anchor_table,
        ),
        "",
        f"Beta selection rule: {config['gradient_contract']['primary_beta_rule']} `fixed_beta_tk_summary.csv` additionally freezes the selected beta before varying T/K; the BPTT-cosine oracle is reported separately and does not control positive one-sided selection.",
        "",
        "## State displacement definitions",
        "",
        f"- `post_T_free`: `||{phase_notation}-s_T||/||s_T||`; this includes residual zero-nudge K-step relaxation.",
        f"- `matched_zero_K`: `||{phase_notation}-s_{{0,K}}||/||s_{{0,K}}||`; this isolates the effect of nudging at matched K.",
        "- Pooled cohort displacement is computed from summed squared numerators and denominators, not by averaging batch ratios.",
        "",
        "## Coverage and artifacts",
        "",
        f"- Fixed cohort: `{cohort['example_count']}` examples in `{cohort['batch_count']}` batches; cohort SHA `{cohort['cohort_sha256']}`.",
        f"- T/K contexts: `{context_count}`; parameter rows: `{parameter_count}`; state summary rows: `{state_count}`.",
        "- `parameter_summary.csv`: every beta x T x K x weight-layer cohort-mean cosine and norm diagnostic.",
        "- Parameter rows also contain the finite-difference SNR>=100 screen and adjacent-beta gradient-consistency diagnostics used by the positive one-sided selector.",
        (
            "- `context_beta_selection.csv`: centered cosine-maximin beta at every T/K coordinate."
            if eqprop_variant == "centered"
            else "- `context_beta_selection.csv`: adjacent-precision-selected beta plus a separate BPTT-cosine oracle at every T/K coordinate."
        ),
        "- `fixed_beta_tk_summary.csv`: T/K dependence at beta fixed from the shared anchor.",
        f"- `state_displacement_summary.csv`: aggregate and per-state-layer {phase_description} displacement against both references.",
        "- `phase_diagnostics.csv` and `state_displacement_batches.csv`: batch-level phase and state evidence.",
        "- `equilibrium_residual_summary.csv`: exact layerwise cohort statistics and strict p90 gates over per-sample projected-KKT maxima at post-T, matched beta=0, and nudged endpoints.",
        "- `equilibrium_residual_per_sample.npz`: the underlying ordered per-sample layer maxima, raw maxima, clamp occupancy, and source indices.",
        f"- `native_mean_gradients/`: raw BPTT and {variant_title} EqProp cohort-mean weight-gradient arrays at each source-native T/K.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolved_run_dir(
    output_root: Path, *, smoke: bool, run_id: str | None
) -> tuple[str, Path]:
    if run_id is None:
        run_id = "smoke-attempt-01" if smoke else "analysis"
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise ValueError(f"Invalid run id {run_id!r}.")
    return run_id, output_root.expanduser().resolve() / run_id


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    config = base._read_json(config_path)
    if config.get("schema_version") != "perfectdiode-conv-eqprop-bptt-beta-tk-displacement-replay/v1":
        raise ValueError("Unexpected analysis config schema.")
    gradient_contract = config["gradient_contract"]
    eqprop_variant = _eqprop_variant(gradient_contract)
    nudging_mode = str(gradient_contract.get("nudging_mode", "cost"))
    if nudging_mode not in {"cost", "current"}:
        raise ValueError(f"Unsupported nudging mode {nudging_mode!r}.")
    residual_contract = config.get("equilibrium_residual_contract", {})
    residual_p90_threshold = float(
        residual_contract.get("per_layer_sample_max_p90_threshold", 1.0e-2)
    )
    residual_uniform_threshold = float(
        residual_contract.get(
            "uniform_endpoint_maximum_threshold", residual_p90_threshold
        )
    )
    residual_hard_threshold = float(
        residual_contract.get("hard_failure_maximum_threshold", 1.0e-1)
    )
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (
            residual_p90_threshold,
            residual_uniform_threshold,
            residual_hard_threshold,
        )
    ):
        raise ValueError("Residual thresholds must be finite and positive.")
    if residual_hard_threshold < residual_p90_threshold:
        raise ValueError("Residual hard-failure threshold cannot be below the p90 gate.")
    residual_gate_required = bool(
        residual_contract.get(
            "require_for_beta_selection",
            eqprop_variant == "positive_one_sided",
        )
    )
    transport_overrides: dict[str, str] = {}
    contract = config["source_contract"]
    if args.runtime_source_root is not None:
        contract["runtime_source_root"] = str(
            args.runtime_source_root.expanduser().resolve()
        )
        transport_overrides["runtime_source_root"] = contract[
            "runtime_source_root"
        ]
    if args.source_study_root is not None:
        contract["source_study_root"] = str(
            args.source_study_root.expanduser().resolve()
        )
        transport_overrides["source_study_root"] = contract[
            "source_study_root"
        ]
    if args.mechanism_initialization_hash_artifact is not None:
        contract["mechanism_initialization_hash_artifact"] = str(
            args.mechanism_initialization_hash_artifact.expanduser().resolve()
        )
        transport_overrides["mechanism_initialization_hash_artifact"] = contract[
            "mechanism_initialization_hash_artifact"
        ]
    runtime_source = base._validate_runtime_source(config)
    inventory, source_hashes_before = _source_inventory(config)
    matched_model_contract_sha256 = _validate_matched_model_contracts(inventory)
    beta_grids_by_case = {
        (str(case["scheme"]), str(case["architecture"])): _beta_grid_for_case(
            gradient_contract, case
        )
        for case in inventory
    }
    dataset_cfg = config["dataset"]
    data_root = args.dataset_root.expanduser().resolve()
    batch_size = int(dataset_cfg["batch_size"])
    requested_examples = (
        batch_size if args.smoke else int(dataset_cfg["cohort_examples"])
    )
    batches, cohort = base._build_validation_cohort(
        inventory[0]["source_config"],
        data_root=data_root,
        batch_size=batch_size,
        example_count=requested_examples,
    )
    if cohort["validation_indices_sha256"] != dataset_cfg[
        "validation_indices_sha256"
    ]:
        raise ValueError("Materialized validation split hash differs from config.")
    expected_guards = dataset_cfg["expected_batch_guards"][: len(batches)]
    observed_guards = [
        {
            "batch_index": int(batch["batch_index"]),
            "source_indices_sha256": batch["source_indices_sha256"],
            "payload_sha256": batch["payload_sha256"],
        }
        for batch in batches
    ]
    if observed_guards != expected_guards:
        raise ValueError("Fixed cohort batch guards differ from the declared prefix.")
    first_signature = base._dataset_signature(
        inventory[0]["source_config"], data_root=data_root, batch_size=batch_size
    )
    for case in inventory[1:]:
        if base._dataset_signature(
            case["source_config"], data_root=data_root, batch_size=batch_size
        ) != first_signature:
            raise ValueError("Source cases do not share the fixed dataset contract.")

    zero_epsilon = float(gradient_contract["zero_epsilon"])
    precision_ratio_threshold = float(
        gradient_contract.get("fd_roundoff_ratio_minimum", 100.0)
    )
    if (
        not math.isfinite(precision_ratio_threshold)
        or precision_ratio_threshold < 100.0
    ):
        raise ValueError("fd_roundoff_ratio_minimum must be finite and at least 100.")
    state_reference_norm_epsilon = float(
        config["state_displacement_contract"]["zero_epsilon"]
    )
    if (
        not math.isfinite(state_reference_norm_epsilon)
        or state_reference_norm_epsilon <= 0.0
    ):
        raise ValueError("State displacement epsilon must be finite and positive.")
    zero_tolerance = float(
        gradient_contract[
            "zero_endpoint_equivalence_relative_l2_tolerance"
        ]
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    output_root = args.output_root.expanduser().resolve()
    run_id, run_dir = _resolved_run_dir(
        output_root, smoke=bool(args.smoke), run_id=args.run_id
    )
    total_contexts = sum(
        len(_grid_for_case(config, case, smoke=bool(args.smoke))) * len(ROLES)
        for case in inventory
    )
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    manifest = {
        "study_id": config["study_id"],
        "run_id": run_id,
        "arm_id": (
            "conv123_all_amplification_adam_positive_one_sided_eqprop_native_beta_stage"
            if eqprop_variant == "positive_one_sided"
            and str(config.get("tk_coordinate_mode", "factorial"))
            == "native_only"
            else "conv123_all_amplification_adam_eqprop_beta_tk_displacement"
        ),
        "evidence_class": config["evidence_class"],
        "dataset": dataset_cfg["name"],
        "smoke": bool(args.smoke),
        "command": command,
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
            "runtime_source": runtime_source,
        },
        "replay": {
            "cohort_examples": requested_examples,
            "batch_size": batch_size,
            "source_cases": len(inventory),
            "checkpoint_cases": len(inventory) * len(ROLES),
            "tk_contexts": total_contexts,
            "matched_model_contract_sha256": matched_model_contract_sha256,
            "transport_overrides": transport_overrides,
            "official_test_read": False,
            "optimizer_constructed": False,
            "optimizer_steps_applied": False,
            "bias_gradients_excluded": True,
            "eqprop_variant": eqprop_variant,
            "nudging_mode": nudging_mode,
            "current_scale": "auto" if nudging_mode == "current" else 1.0,
            "finite_difference_denominator": (
                "beta_actual * output_row_current_scale"
                if nudging_mode == "current"
                else "beta_actual"
            ),
            "nudged_phases": list(_nudged_phase_names(eqprop_variant)),
            "beta_grids": {
                f"{scheme}/{architecture}": rows
                for (scheme, architecture), rows in beta_grids_by_case.items()
            },
            "finite_difference_precision": {
                "epsilon": float(torch.finfo(torch.float32).eps),
                "ratio_threshold": precision_ratio_threshold,
                "required_for_beta_selection": eqprop_variant
                == "positive_one_sided",
            },
            "equilibrium_residual": {
                "mode": "per_layer_per_sample_projected_kkt_max",
                "p90_threshold": residual_p90_threshold,
                "uniform_maximum_threshold": residual_uniform_threshold,
                "hard_failure_maximum_threshold": residual_hard_threshold,
                "comparison": "strictly_less_than",
                "require_for_beta_selection": residual_gate_required,
                "default_threshold_provenance": (
                    "config"
                    if "per_layer_sample_max_p90_threshold" in residual_contract
                    else "active_perfectdiode_conv3_protocol"
                ),
            },
        },
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.sha256_file(Path(__file__).resolve()),
            "immutable_base_runner_path": str(IMMUTABLE_BASE_RUNNER),
            "immutable_base_runner_sha256": base.sha256_file(
                IMMUTABLE_BASE_RUNNER
            ),
            "repository_root": str(REPOSITORY_ROOT),
            "repository_head": base._git_output(REPOSITORY_ROOT, "rev-parse", "HEAD"),
        },
    }
    base.start_run(run_dir, manifest)
    base._write_json(run_dir / "config.resolved.json", config)
    base._write_json(run_dir / "cohort.json", cohort)
    base._write_json(
        run_dir / "source_inventory.json",
        {
            "runtime_source": runtime_source,
            "cases": [
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in case.items()
                    if key != "source_config"
                }
                for case in inventory
            ],
        },
    )

    parameter_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    residual_summary_rows: list[dict[str, Any]] = []
    residual_archive_records: list[tuple[dict[str, Any], np.ndarray]] = []
    context_rows: list[dict[str, Any]] = []
    parameter_guards: list[dict[str, Any]] = []
    curvature_guards: list[dict[str, Any]] = []
    common_state_guard_failures: list[dict[str, Any]] = []
    cross_k_free_state_hashes: dict[tuple[str, str, str, int, int], str] = {}
    cross_k_free_state_failures: list[dict[str, Any]] = []
    weight_names_by_architecture: dict[str, list[str]] = {}
    state_layer_names_by_architecture: dict[str, list[str]] = {}
    outcome_counts: defaultdict[str, int] = defaultdict(int)
    completed_contexts = 0

    try:
        for case in inventory:
            architecture = str(case["architecture"])
            scheme = str(case["scheme"])
            beta_grid = beta_grids_by_case[(scheme, architecture)]
            betas = [float(row["beta"]) for row in beta_grid]
            beta_metadata = _beta_metadata_by_actual(beta_grid)
            source_config = case["source_config"]
            coordinates = _grid_for_case(config, case, smoke=bool(args.smoke))
            k_values = list(dict.fromkeys(k_value for _, k_value in coordinates))
            for role in ROLES:
                for k_value in k_values:
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
                    if init_hash != case[
                        "reconstructed_initialization_tensor_sha256"
                    ]:
                        raise ValueError(
                            f"Initialization reconstruction mismatch for {scheme}/{architecture}."
                        )
                    init_curvature = _output_curvature_diagnostic(runtime)
                    declared_init_b0 = case.get("init_output_curvature_scale_b0")
                    if declared_init_b0 is not None:
                        declared_init_b0 = float(declared_init_b0)
                        init_b0_matches = math.isclose(
                            float(init_curvature["output_curvature_scale_b0"]),
                            declared_init_b0,
                            rel_tol=1.0e-6,
                            abs_tol=0.0,
                        )
                        if not init_b0_matches:
                            raise ValueError(
                                f"Initialization b0 mismatch for {scheme}/{architecture}: "
                                f"{init_curvature['output_curvature_scale_b0']} != {declared_init_b0}."
                            )
                    else:
                        init_b0_matches = True
                    declared_depth = case.get("scored_interaction_count_L")
                    if declared_depth is not None and int(declared_depth) != len(
                        runtime["weight_names"]
                    ):
                        raise ValueError(
                            f"Scored interaction count mismatch for {scheme}/{architecture}."
                        )
                    if "normalized_betas" in gradient_contract:
                        resolved_depths = {
                            int(row["amplification_depth_L"]) for row in beta_grid
                        }
                        if len(resolved_depths) != 1:
                            raise ValueError(
                                f"Non-unique amplification exponent for {scheme}/{architecture}."
                            )
                        resolved_depth = next(iter(resolved_depths))
                        convention = str(
                            gradient_contract.get(
                                "amplification_exponent_convention",
                                "scored_weight_interactions",
                            )
                        )
                        if convention == "output_bias_current_row":
                            output_layer = runtime["cost_fn"].layers()[-1]
                            output_bias_exponent = max(
                                exact_runtime_layer_index(output_layer) - 1, 0
                            )
                            if resolved_depth != output_bias_exponent:
                                raise ValueError(
                                    "Amplification exponent does not match the output "
                                    f"bias-current row for {scheme}/{architecture}: "
                                    f"{resolved_depth} != {output_bias_exponent}."
                                )
                            exact_row_factor = float(
                                _amplified_layer_row_scale(
                                    runtime["energy_fn"], output_layer
                                )
                            )
                            declared_row_factor = (
                                float(case["voltage_amp"])
                                / float(case["current_amp"])
                            ) ** resolved_depth
                            if not math.isclose(
                                exact_row_factor,
                                declared_row_factor,
                                rel_tol=1.0e-12,
                                abs_tol=0.0,
                            ):
                                raise ValueError(
                                    "Declared nudge multiplier does not match the exact "
                                    f"runtime output-row current scale for {scheme}/{architecture}."
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
                                raise ValueError(
                                    "Runtime nudge scale does not implement the declared "
                                    f"nudging mode for {scheme}/{architecture}."
                                )
                    if "normalized_betas" in gradient_contract and scheme == "baseline":
                        declared_baseline_b0 = float(
                            gradient_contract[
                                "baseline_init_output_curvature_scale_b0_by_architecture"
                            ][architecture]
                        )
                        if not math.isclose(
                            float(init_curvature["output_curvature_scale_b0"]),
                            declared_baseline_b0,
                            rel_tol=1.0e-6,
                            abs_tol=0.0,
                        ):
                            raise ValueError(
                                f"Baseline b0 mapping mismatch for {architecture}."
                            )
                    if role == "best_validation":
                        runtime["energy_fn"].load(case["best_checkpoint_path"])
                        base._verify_npz_checkpoint(
                            runtime, case["weights_best_path"]
                        )
                    checkpoint_curvature = _output_curvature_diagnostic(runtime)
                    curvature_guards.append(
                        {
                            "architecture": architecture,
                            "scheme": scheme,
                            "checkpoint_role": role,
                            "K": int(k_value),
                            "declared_init_output_curvature_scale_b0": declared_init_b0,
                            "init_b0_matches": init_b0_matches,
                            **{
                                f"init_{key}": value
                                for key, value in init_curvature.items()
                            },
                            **{
                                f"checkpoint_{key}": value
                                for key, value in checkpoint_curvature.items()
                            },
                        }
                    )
                    names = list(runtime["weight_names"])
                    types = dict(zip(names, runtime["weight_types"], strict=True))
                    if not names or any(value not in WEIGHT_TYPES for value in types.values()):
                        raise RuntimeError("Weight-only inclusion contract failed.")
                    state_names = [
                        str(layer.name) for layer in runtime["free_layers"]
                    ] + ["__all__"]
                    if architecture in weight_names_by_architecture and weight_names_by_architecture[architecture] != names:
                        raise RuntimeError(
                            f"Weight schema changed within {architecture}."
                        )
                    if architecture in state_layer_names_by_architecture and state_layer_names_by_architecture[architecture] != state_names:
                        raise RuntimeError(
                            f"State-layer schema changed within {architecture}."
                        )
                    weight_names_by_architecture[architecture] = names
                    state_layer_names_by_architecture[architecture] = state_names
                    before_hash = base._parameter_state_sha256(runtime["parameters"])

                    for t_value, coordinate_k in coordinates:
                        if int(coordinate_k) != int(k_value):
                            continue
                        runtime["minimizer_inference"].num_iterations = int(t_value)
                        runtime["inference_iterations"] = int(t_value)
                        runtime["gradient_iterations"] = int(k_value)
                        shared_t, shared_k = _shared_anchor(architecture)
                        context = {
                            "architecture": architecture,
                            "scheme": scheme,
                            "checkpoint_role": role,
                            "eqprop_variant": eqprop_variant,
                            "nudging_mode": nudging_mode,
                            "nudging_current_scale": float(
                                runtime["nudging_current_scale"]
                            ),
                            "finite_difference_denominator_scale": (
                                float(runtime["nudging_current_scale"])
                                if nudging_mode == "current"
                                else 1.0
                            ),
                            "T": int(t_value),
                            "K": int(k_value),
                            "native_T": int(case["native_T"]),
                            "native_K": int(case["native_K"]),
                            "native_context": bool(
                                int(t_value) == int(case["native_T"])
                                and int(k_value) == int(case["native_K"])
                            ),
                            "shared_anchor_T": shared_t,
                            "shared_anchor_K": shared_k,
                            "shared_anchor_context": bool(
                                int(t_value) == shared_t and int(k_value) == shared_k
                            ),
                            "voltage_amplification": float(case["voltage_amp"]),
                            "current_amplification": float(case["current_amp"]),
                            "init_output_curvature_scale_b0": float(
                                init_curvature["output_curvature_scale_b0"]
                            ),
                            "checkpoint_output_curvature_scale_b0": float(
                                checkpoint_curvature["output_curvature_scale_b0"]
                            ),
                        }
                        accumulator = _ContextAccumulator()
                        residual_accumulator = _ResidualAccumulator()
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
                            cross_k_key = (
                                architecture,
                                scheme,
                                role,
                                int(t_value),
                                batch_index,
                            )
                            prior_free_hash = cross_k_free_state_hashes.setdefault(
                                cross_k_key, free_hash
                            )
                            if prior_free_hash != free_hash:
                                cross_k_free_state_failures.append(
                                    {
                                        **context,
                                        "batch_index": batch_index,
                                        "expected_post_T_state_sha256": prior_free_hash,
                                        "observed_post_T_state_sha256": free_hash,
                                    }
                                )
                            (
                                bptt,
                                estimates,
                                diagnostics,
                                state_diagnostics,
                                outcomes,
                                precision_endpoints,
                            ) = _run_gradient_phases(
                                runtime,
                                free_states=free_states,
                                betas=betas,
                                zero_epsilon=zero_epsilon,
                                state_reference_norm_epsilon=state_reference_norm_epsilon,
                                zero_endpoint_relative_tolerance=zero_tolerance,
                                eqprop_variant=eqprop_variant,
                                residual_accumulator=residual_accumulator,
                                source_indices=batch["source_indices"],
                            )
                            if any(
                                row["phase_start_state_sha256"] != free_hash
                                for row in diagnostics
                            ):
                                common_state_guard_failures.append(
                                    {
                                        **context,
                                        "batch_index": batch_index,
                                        "common_post_t_state_sha256": free_hash,
                                    }
                                )
                            for diagnostic in diagnostics:
                                phase_rows.append(
                                    {
                                        "schema": SCHEMA,
                                        **context,
                                        "batch_index": batch_index,
                                        "batch_payload_sha256": batch[
                                            "payload_sha256"
                                        ],
                                        **_beta_reporting_fields(
                                            float(diagnostic["beta"]),
                                            beta_metadata,
                                            context=context,
                                        ),
                                        **diagnostic,
                                    }
                                )
                            for diagnostic in state_diagnostics:
                                state_rows.append(
                                    {
                                        "schema": SCHEMA,
                                        **context,
                                        "batch_index": batch_index,
                                        "batch_payload_sha256": batch[
                                            "payload_sha256"
                                        ],
                                        **_beta_reporting_fields(
                                            float(diagnostic["beta"]),
                                            beta_metadata,
                                            context=context,
                                        ),
                                        **diagnostic,
                                    }
                                )
                            for beta in betas:
                                outcome_counts[outcomes[beta]] += 1
                            for name in names:
                                accumulator.add(
                                    ("bptt", name),
                                    bptt[name],
                                    batch_size=int(images.shape[0]),
                                )
                                for beta in betas:
                                    outcome = outcomes[beta]
                                    if outcome != "ok":
                                        raw_rows.append(
                                            {
                                                "schema": SCHEMA,
                                                **context,
                                                "batch_index": batch_index,
                                                "batch_size": int(images.shape[0]),
                                                "batch_payload_sha256": batch[
                                                    "payload_sha256"
                                                ],
                                                "beta": beta,
                                                **_beta_reporting_fields(
                                                    beta,
                                                    beta_metadata,
                                                    context=context,
                                                ),
                                                "eqprop_variant": eqprop_variant,
                                                "parameter_name": name,
                                                "parameter_type": types[name],
                                                "bias_excluded": True,
                                                "outcome": outcome,
                                            }
                                        )
                                        continue
                                    accumulator.add(
                                        ("eqprop_zero_endpoint", beta, name),
                                        precision_endpoints[beta][name]["zero"],
                                        batch_size=int(images.shape[0]),
                                    )
                                    accumulator.add(
                                        ("eqprop_positive_endpoint", beta, name),
                                        precision_endpoints[beta][name]["positive"],
                                        batch_size=int(images.shape[0]),
                                    )
                                    metrics = base._vector_metrics(
                                        bptt[name],
                                        estimates[beta][name],
                                        zero_epsilon=zero_epsilon,
                                    )
                                    accumulator.add(
                                        ("eqprop", beta, name),
                                        estimates[beta][name],
                                        batch_size=int(images.shape[0]),
                                        batch_cosine=metrics["cosine"],
                                    )
                                    raw_rows.append(
                                        {
                                            "schema": SCHEMA,
                                            **context,
                                            "batch_index": batch_index,
                                            "batch_size": int(images.shape[0]),
                                            "batch_payload_sha256": batch[
                                                "payload_sha256"
                                            ],
                                            "beta": beta,
                                            **_beta_reporting_fields(
                                                beta,
                                                beta_metadata,
                                                context=context,
                                            ),
                                            "eqprop_variant": eqprop_variant,
                                            "parameter_name": name,
                                            "parameter_type": types[name],
                                            "bias_excluded": True,
                                            "outcome": (
                                                "ok"
                                                if metrics["cosine"] is not None
                                                else "dead_gradient"
                                            ),
                                            **metrics,
                                        }
                                    )
                        context_residual_rows = residual_accumulator.summary_rows(
                            context=context,
                            expected_examples=requested_examples,
                            p90_threshold=residual_p90_threshold,
                            beta_metadata=beta_metadata,
                            uniform_maximum_threshold=residual_uniform_threshold,
                            hard_maximum_threshold=residual_hard_threshold,
                        )
                        residual_summary_rows.extend(context_residual_rows)
                        residual_archive_records.extend(
                            residual_accumulator.archive_records(
                                context=context,
                                beta_metadata=beta_metadata,
                            )
                        )
                        residual_gates = _residual_gate_by_beta(
                            context_residual_rows,
                            betas=betas,
                            eqprop_variant=eqprop_variant,
                        )
                        precision_gates = _finite_difference_precision_gates(
                            accumulator,
                            names=names,
                            betas=betas,
                            expected_examples=requested_examples,
                            ratio_threshold=precision_ratio_threshold,
                        )
                        context_parameter_rows = _parameter_rows_for_context(
                            accumulator,
                            context=context,
                            names=names,
                            types=types,
                            betas=betas,
                            expected_examples=requested_examples,
                            zero_epsilon=zero_epsilon,
                            eqprop_variant=eqprop_variant,
                            residual_gates=residual_gates,
                            residual_gate_required=residual_gate_required,
                            beta_metadata=beta_metadata,
                            precision_gates=precision_gates,
                            precision_gate_required=eqprop_variant
                            == "positive_one_sided",
                        )
                        _attach_adjacent_beta_consistency(
                            context_parameter_rows,
                            accumulator,
                            names=names,
                            beta_grid=beta_grid,
                            expected_examples=requested_examples,
                            cosine_threshold=float(
                                gradient_contract.get(
                                    "adjacent_precision_cosine_minimum", 0.99
                                )
                            ),
                            symmetric_norm_delta_threshold=float(
                                gradient_contract.get(
                                    "adjacent_precision_symmetric_norm_delta_maximum",
                                    0.10,
                                )
                            ),
                        )
                        parameter_rows.extend(context_parameter_rows)
                        context_rows.append(
                            _select_context_beta(
                                context_parameter_rows,
                                context,
                                eqprop_variant=eqprop_variant,
                            )
                        )
                        if context["native_context"]:
                            filename = (
                                f"{architecture}__{scheme}__{role}__T{t_value}__K{k_value}.npz"
                            )
                            _native_gradient_archive(
                                run_dir / "artifacts/native_mean_gradients" / filename,
                                accumulator,
                                names=names,
                                betas=betas,
                                expected_examples=requested_examples,
                                eqprop_variant=eqprop_variant,
                                metadata={
                                    **context,
                                    "cohort_sha256": cohort["cohort_sha256"],
                                    "example_count": requested_examples,
                                    "beta_grid": beta_grid,
                                },
                            )
                        completed_contexts += 1
                        base.update_status_progress(
                            run_dir,
                            {
                                "stage": "checkpoint_replay",
                                "completed_contexts": completed_contexts,
                                "total_contexts": total_contexts,
                                **context,
                            },
                        )
                    after_hash = base._parameter_state_sha256(runtime["parameters"])
                    unchanged = before_hash == after_hash
                    parameter_guards.append(
                        {
                            "architecture": architecture,
                            "scheme": scheme,
                            "checkpoint_role": role,
                            "K": int(k_value),
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
                            f"Parameter tensors changed for {scheme}/{architecture}/{role}/K={k_value}."
                        )
                base.append_metric(
                    run_dir / "metrics.jsonl",
                    {
                        "stage": "checkpoint_case_complete",
                        "split": "validation",
                        "architecture": architecture,
                        "scheme": scheme,
                        "checkpoint_role": role,
                        "contexts": len(coordinates),
                        "examples_per_context": requested_examples,
                    },
                )

        if completed_contexts != total_contexts:
            raise RuntimeError(
                f"Expected {total_contexts} contexts; completed {completed_contexts}."
            )
        if common_state_guard_failures:
            raise RuntimeError("At least one phase did not start at the common post-T state.")
        if cross_k_free_state_failures:
            raise RuntimeError(
                "Post-T free states changed across K at a matched T/batch."
            )
        state_summary = _summarize_state_rows(
            state_rows, expected_batch_count=len(batches)
        )
        fixed_rows = _fixed_beta_context_rows(parameter_rows, context_rows)
        study_rows = _study_summary_rows(context_rows, fixed_rows)
        coverage_guards = _exact_coverage_guards(
            context_rows=context_rows,
            parameter_rows=parameter_rows,
            phase_rows=phase_rows,
            state_summary=state_summary,
            residual_summary=residual_summary_rows,
            beta_grids_by_case=beta_grids_by_case,
            weight_names_by_architecture=weight_names_by_architecture,
            state_layer_names_by_architecture=state_layer_names_by_architecture,
            batch_count=len(batches),
            eqprop_variant=eqprop_variant,
        )
        source_hashes_after = _verify_source_hashes(
            inventory, source_hashes_before
        )
        guards = {
            "source_hashes_before": {
                "/".join(key): value for key, value in source_hashes_before.items()
            },
            "source_hashes_after": source_hashes_after,
            "source_bytes_unchanged": {
                "/".join(key): value == source_hashes_after["/".join(key)]
                for key, value in source_hashes_before.items()
            },
            "parameter_guards": parameter_guards,
            "output_curvature_guards": curvature_guards,
            "all_declared_initialization_output_curvatures_match": all(
                bool(row["init_b0_matches"]) for row in curvature_guards
            ),
            "all_parameter_tensors_unchanged": all(
                bool(row["unchanged"]) for row in parameter_guards
            ),
            "common_post_t_state_guard_failures": common_state_guard_failures,
            "all_phase_starts_match_common_post_t_state": not common_state_guard_failures,
            "cross_k_free_state_failures": cross_k_free_state_failures,
            "post_T_free_states_identical_across_K": not cross_k_free_state_failures,
            "matched_model_contract_sha256": matched_model_contract_sha256,
            "exact_coverage": coverage_guards,
            "equilibrium_residual_gate": {
                "p90_threshold": residual_p90_threshold,
                "comparison": "strictly_less_than",
                "require_for_beta_selection": residual_gate_required,
                "pass_count": coverage_guards["residual_gate_pass_count"],
                "fail_count": coverage_guards["residual_gate_fail_count"],
            },
            "biases_active_in_dynamics": True,
            "bias_gradients_excluded": True,
            "optimizer_constructed": False,
            "optimizer_steps_applied": False,
            "official_test_read": False,
        }
        base._write_csv(run_dir / "raw_batch_metrics.csv", raw_rows)
        base._write_csv(run_dir / "phase_diagnostics.csv", phase_rows)
        base._write_csv(run_dir / "state_displacement_batches.csv", state_rows)
        base._write_csv(run_dir / "parameter_summary.csv", parameter_rows)
        base._write_csv(run_dir / "context_beta_selection.csv", context_rows)
        base._write_csv(run_dir / "fixed_beta_tk_summary.csv", fixed_rows)
        base._write_csv(run_dir / "study_summary.csv", study_rows)
        base._write_csv(run_dir / "state_displacement_summary.csv", state_summary)
        base._write_csv(
            run_dir / "equilibrium_residual_summary.csv", residual_summary_rows
        )
        _write_residual_archive(
            run_dir / "artifacts/equilibrium_residual_per_sample.npz",
            residual_archive_records,
        )
        base._write_json(run_dir / "read_only_guards.json", guards)

        plots_dir = run_dir / "artifacts/plots"
        for architecture in ("conv1", "conv2", "conv3"):
            for anchor in ("native", "shared"):
                _plot_cosine_beta(
                    parameter_rows,
                    architecture=architecture,
                    anchor=anchor,
                    path=plots_dir / f"cosine_beta_{architecture}_{anchor}.png",
                    eqprop_variant=eqprop_variant,
                )
            for role in ROLES:
                _plot_tk_maximin(
                    context_rows,
                    architecture=architecture,
                    role=role,
                    path=plots_dir
                    / (
                        f"tk_maximin_{architecture}_{role}.png"
                        if eqprop_variant == "centered"
                        else f"tk_adjacent_precision_selection_{architecture}_{role}.png"
                    ),
                    eqprop_variant=eqprop_variant,
                )
                _plot_tk_by_layer_fixed_beta(
                    parameter_rows,
                    context_rows,
                    architecture=architecture,
                    role=role,
                    path=plots_dir / f"tk_layer_fixed_beta_{architecture}_{role}.png",
                    eqprop_variant=eqprop_variant,
                )
            for anchor in ("native", "shared"):
                for reference_kind in ("post_T_free", "matched_zero_K"):
                    _plot_displacement_beta(
                        state_summary,
                        architecture=architecture,
                        anchor=anchor,
                        reference_kind=reference_kind,
                        state_layer_name="__all__",
                        path=plots_dir
                        / f"displacement_beta_{architecture}_{anchor}_{reference_kind}.png",
                        eqprop_variant=eqprop_variant,
                    )
        _write_report(
            run_dir / "report.md",
            config=config,
            cohort=cohort,
            study_rows=study_rows,
            parameter_count=len(parameter_rows),
            state_count=len(state_summary),
            context_count=len(context_rows),
            smoke=bool(args.smoke),
        )
        base.append_metric(
            run_dir / "metrics.jsonl",
            {
                "stage": "analysis_complete",
                "split": "validation",
                "source_cases": len(inventory),
                "checkpoint_cases": len(inventory) * len(ROLES),
                "tk_contexts": len(context_rows),
                "parameter_rows": len(parameter_rows),
                "state_summary_rows": len(state_summary),
                "equilibrium_residual_summary_rows": len(residual_summary_rows),
            },
        )
        completion = {
            "criteria_met": True,
            "all_declared_source_cases_present": len(inventory) == 9,
            "all_declared_checkpoint_cases_present": len(inventory) * len(ROLES) == 18,
            "all_declared_tk_contexts_present": len(context_rows) == total_contexts,
            "all_parameter_tensors_unchanged": guards[
                "all_parameter_tensors_unchanged"
            ],
            "all_source_bytes_unchanged": all(
                guards["source_bytes_unchanged"].values()
            ),
            "all_phase_starts_match_common_post_t_state": guards[
                "all_phase_starts_match_common_post_t_state"
            ],
            "post_T_free_states_identical_across_K": guards[
                "post_T_free_states_identical_across_K"
            ],
            "aggregate_and_per_state_layer_displacement_reported": any(
                row["state_layer_name"] == "__all__" for row in state_summary
            )
            and any(row["state_layer_name"] != "__all__" for row in state_summary),
            "both_displacement_references_reported": {
                str(row["reference_kind"]) for row in state_summary
            }
            >= {"post_T_free", "matched_zero_K"},
            "parameter_keys_exact": coverage_guards["parameter_keys_exact"],
            "phase_keys_exact": coverage_guards["phase_keys_exact"],
            "state_keys_exact": coverage_guards["state_keys_exact"],
            "state_rows_have_expected_batch_count": coverage_guards[
                "state_rows_have_expected_batch_count"
            ],
            "residual_keys_exact": coverage_guards["residual_keys_exact"],
            "finite_difference_precision_diagnostics_complete": coverage_guards[
                "finite_difference_precision_diagnostics_complete"
            ],
            "amplification_scaled_beta_metadata_complete": coverage_guards[
                "amplification_scaled_beta_metadata_complete"
            ],
            "all_declared_initialization_output_curvatures_match": guards[
                "all_declared_initialization_output_curvatures_match"
            ],
            "declared_gradient_variants_reported": coverage_guards[
                "reported_gradient_variants"
            ]
            == config["gradient_contract"]["eqprop_variants_reported"],
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
                "source_cases": len(inventory),
                "checkpoint_cases": len(inventory) * len(ROLES),
                "tk_contexts": len(context_rows),
                "cohort_examples": requested_examples,
                "parameter_rows": len(parameter_rows),
                "state_summary_rows": len(state_summary),
                "equilibrium_residual_summary_rows": len(residual_summary_rows),
                "residual_hard_failure_rows": coverage_guards[
                    "residual_hard_failure_count"
                ],
                f"{eqprop_variant}_phase_outcomes": dict(outcome_counts),
                "bias_gradients_excluded": True,
                "optimizer_steps_applied": False,
                "official_test_read": False,
            },
            completion=completion,
        )
        errors = base.validate_run(run_dir)
        if errors:
            raise RuntimeError(
                f"Completed reporting bundle failed validation: {'; '.join(errors)}"
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
    parser.add_argument("--source-study-root", type=Path)
    parser.add_argument("--mechanism-initialization-hash-artifact", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results/perfectdiode-conv123-all-amplification-adam-eqprop-bptt-beta-tk-displacement-seed0-20260810-v1"
        ),
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/filip/datasets/mnist")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--target", default="main")
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runtime_source_root.expanduser().resolve() != base.RUNTIME_SOURCE_ROOT:
        raise ValueError(
            "--runtime-source-root must be supplied at process bootstrap and match the exact source contract."
        )
    result = run_analysis(args)
    run_id = args.run_id or ("smoke-attempt-01" if args.smoke else "analysis")
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
