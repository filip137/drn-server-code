#!/usr/bin/env python3
"""Fail-closed Conv3 bounded-uniform T/K operating-point audit.

This module implements the narrow audit required before a Conv3 rho probe.  It
does not select a new operating point: it verifies the accepted ``T=8, K=8``
point against the protocol sentinels on one shared initialization checkpoint.

The two comparisons are deliberately independent:

* residuals are measured at ``T=8`` and ``T=64``;
* gradients compare ``K=8`` with ``K=64`` while holding the free equilibrium
  fixed at ``T=8``.

The latter detail prevents a change in inference convergence from being
mistaken for a change in backward-unroll convergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from labs.mnist_train import (
    FlexibleDeepResistiveEnergy,
    _build_tracking_minimizer,
    _reset_name_counters,
    _resolve_callable,
    _resolve_dataset_config,
    _set_seed,
)
from model.function.cost import SquaredError, SquaredErrorPairedOutputs
from model.function.network import Network
from training.sgd import Backprop


SCHEMA_VERSION = "perfectdiode-conv3-bounded-uniform-operating-point-gate/v1"
EXPECTED_CONV_WEIGHTS = ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2")
DEFAULT_CONTRACT: dict[str, Any] = {
    "selected_T": 8,
    "reference_T": 64,
    "selected_K": 8,
    "reference_K": 64,
    "residual_examples": 1024,
    "residual_batch_size": 64,
    "residual_p90_maximum": 1.0e-2,
    "clamp_epsilon": 1.0e-8,
    "gradient_examples": 256,
    "gradient_batch_size": 32,
    "gradient_zero_epsilon": 1.0e-12,
    "relative_gradient_l2_norm_delta_maximum": 0.10,
    "absolute_zero_fraction_delta_maximum": 0.02,
    "gradient_vector_cosine_minimum": 0.90,
    "reference_median_gradient_rms_minimum": 1.0e-12,
    "reference_q90_zero_fraction_maximum": 0.99,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256_named_tensors(values: Sequence[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in values:
        array = tensor.detach().to(device="cpu").contiguous().numpy()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return math.nan
    return float(np.quantile(array, probability, method="linear"))


def _stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return {
            "mean": math.nan,
            "median": math.nan,
            "p90": math.nan,
            "p99": math.nan,
            "maximum": math.nan,
        }
    return {
        "mean": float(array.mean()),
        "median": _linear_quantile(array, 0.50),
        "p90": _linear_quantile(array, 0.90),
        "p99": _linear_quantile(array, 0.99),
        "maximum": float(array.max()),
    }


def _projected_perfect_diode_residual(
    state: torch.Tensor,
    gradient: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Return the projected KKT residual for one perfect-diode hidden layer."""

    if state.shape != gradient.shape:
        raise ValueError(
            f"State/gradient shape mismatch: {tuple(state.shape)} != {tuple(gradient.shape)}."
        )
    if state.ndim < 2 or state.shape[1] % 2:
        raise ValueError(
            "A perfect-diode hidden layer must have an even excitatory/inhibitory axis."
        )
    half = state.shape[1] // 2
    residual = torch.empty_like(gradient)

    positive_state = state[:, :half]
    positive_gradient = gradient[:, :half]
    residual[:, :half] = torch.where(
        positive_state > epsilon,
        positive_gradient.abs(),
        torch.relu(-positive_gradient),
    )

    negative_state = state[:, half:]
    negative_gradient = gradient[:, half:]
    residual[:, half:] = torch.where(
        negative_state < -epsilon,
        negative_gradient.abs(),
        torch.relu(negative_gradient),
    )
    return residual


def _clamp_occupancy_per_sample(
    state: torch.Tensor, *, epsilon: float
) -> torch.Tensor:
    if state.ndim < 2 or state.shape[1] % 2:
        raise ValueError("Expected an even excitatory/inhibitory hidden-layer axis.")
    half = state.shape[1] // 2
    clamped = torch.cat(
        (state[:, :half] <= epsilon, state[:, half:] >= -epsilon), dim=1
    )
    return clamped.reshape(clamped.shape[0], -1).to(torch.float64).mean(dim=1)


def _model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model_key = str(config.get("lab", {}).get("model_key", "mnist_bp_conv_amp"))
    overrides = config.get("model_overrides", {})
    if model_key not in overrides:
        raise KeyError(f"Missing model_overrides[{model_key!r}].")
    return {**dict(config["model_base"]), **dict(overrides[model_key])}


def _validate_source_contract(config: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> None:
    if model_cfg.get("non_linearity") != "perfect_diode":
        raise ValueError("The Conv3 operating-point gate requires perfect_diode.")
    if len(model_cfg.get("conv_pipeline") or []) != 3:
        raise ValueError("The operating-point gate requires exactly three Conv stages.")
    if model_cfg.get("weight_init_mode") != "bounded_uniform":
        raise ValueError("The new Conv3 gate requires bounded_uniform initialization.")
    bounds = [float(model_cfg["weight_min"]), float(model_cfg["weight_max"])]
    if bounds != [1.0e-5, 1.0e-4]:
        raise ValueError(f"Expected bounds [1e-5, 1e-4], got {bounds!r}.")
    if bool(model_cfg.get("minimizer", {}).get("adaptive_equilibrium", True)):
        raise ValueError("The operating-point audit requires fixed-step minimization.")
    if config.get("batch_state_policy") != "reset_each_batch":
        raise ValueError("The operating-point audit requires reset_each_batch.")
    configured_t = int(model_cfg.get("num_iterations_inference", -1))
    configured_k = int(model_cfg.get("num_iterations_training", -1))
    if (configured_t, configured_k) != (8, 8):
        raise ValueError(
            "The source config must bind the accepted Conv3 operating point T=8, K=8; "
            f"got T={configured_t}, K={configured_k}."
        )
    for key in (
        "quadratic_diode_param",
        "exponential_diode_param",
        "hard_sigmoid_param",
    ):
        if not isinstance(model_cfg.get(key), Mapping) or not model_cfg[key]:
            raise ValueError(f"Expected an explicit non-empty {key} dictionary.")


def _build_model_context(
    config: Mapping[str, Any],
    checkpoint: Path,
    *,
    device: torch.device,
    inference_iterations: int,
    training_iterations: int,
) -> dict[str, Any]:
    model_cfg = _model_config(config)
    _validate_source_contract(config, model_cfg)
    seed = int(config.get("seed", 0))
    _reset_name_counters()
    _set_seed(seed)

    layer_shapes = [
        tuple(shape) if isinstance(shape, (list, tuple)) else (int(shape),)
        for shape in model_cfg["layer_shapes"]
    ]
    energy_fn = FlexibleDeepResistiveEnergy(
        layer_shapes=layer_shapes,
        conv_pipeline=model_cfg.get("conv_pipeline") or [],
        pooling_mode=model_cfg.get("pooling_mode"),
        weight_gains=model_cfg["weight_gains"],
        input_gain=model_cfg["input_gain"],
        non_linearity=model_cfg["non_linearity"],
        exponential_diode_param=model_cfg["exponential_diode_param"],
        quadratic_diode_param=model_cfg["quadratic_diode_param"],
        hard_sigmoid_param=model_cfg["hard_sigmoid_param"],
        voltage_amp=model_cfg["voltage_amp"],
        current_amp=model_cfg["current_amp"],
        weight_min=model_cfg["weight_min"],
        weight_max=model_cfg["weight_max"],
        weight_init_mode=model_cfg["weight_init_mode"],
        input_mode=config.get("input_mode", "train"),
        trainable_amplification=bool(model_cfg.get("trainable_amplification", False)),
        amplification_min=model_cfg.get("amplification_min", 1.0e-6),
        amplification_max=model_cfg.get("amplification_max"),
    )
    energy_fn.set_device(device)
    energy_fn.load(checkpoint)
    network = Network(energy_fn)
    free_layers = tuple(network.free_layers())
    output_layer = energy_fn.layers()[-1]
    output_dim = int(output_layer.shape[0])
    if output_dim == 10:
        cost_fn = SquaredError(output_layer)
    elif output_dim == 20:
        cost_fn = SquaredErrorPairedOutputs(output_layer, num_classes=10)
    else:
        raise ValueError(f"Unsupported output dimension {output_dim}; expected 10 or 20.")

    minimizer_inference = _build_tracking_minimizer(
        energy_fn,
        free_layers,
        model_cfg,
        config["energy_minimizer"]["mode"],
        num_iterations=int(inference_iterations),
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
        adaptive_equilibrium=False,
    )
    minimizer_training = _build_tracking_minimizer(
        energy_fn,
        free_layers,
        model_cfg,
        config["energy_minimizer"]["mode"],
        num_iterations=int(training_iterations),
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
        adaptive_equilibrium=False,
    )
    parameters = tuple(energy_fn.params())
    parameter_names = tuple(str(getattr(value, "name", "")).strip() for value in parameters)
    conv_names = tuple(name for name in parameter_names if name.startswith("ConvWeight_"))
    if conv_names != EXPECTED_CONV_WEIGHTS:
        raise ValueError(
            f"Expected Conv parameter order {EXPECTED_CONV_WEIGHTS!r}, got {conv_names!r}."
        )
    configured_order = config.get("parameter_order")
    if configured_order is not None and tuple(configured_order) != parameter_names:
        raise ValueError(
            f"Configured/runtime parameter order mismatch: {configured_order!r} != {parameter_names!r}."
        )
    return {
        "energy_fn": energy_fn,
        "network": network,
        "free_layers": free_layers,
        "cost_fn": cost_fn,
        "parameters": parameters,
        "minimizer_inference": minimizer_inference,
        "minimizer_training": minimizer_training,
        "inference_iterations": int(inference_iterations),
        "training_iterations": int(training_iterations),
        "parameter_sha256": _sha256_named_tensors(
            [(name, parameter.state) for name, parameter in zip(parameter_names, parameters)]
        ),
    }


def _load_training_cohort(
    config: Mapping[str, Any],
    *,
    batch_size: int,
    examples: int,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    dataset_key = str(config.get("lab", {}).get("dataset_key", "mnist"))
    _resolved_key, dataset_cfg = _resolve_dataset_config(dict(config), dataset_key)
    params = dict(dataset_cfg["params"])
    params["batch_size"] = int(batch_size)
    params["device"] = torch.device("cpu")
    params["download"] = False
    if "root" in params:
        params["root"] = str(Path(str(params["root"])).expanduser())
    builder = _resolve_callable(dataset_cfg["factory"])(**params)
    bundle = builder.build()
    if not hasattr(bundle, "train_loader"):
        raise ValueError("The gate requires the deterministic train/validation loader bundle.")
    bundle.reset_train_shuffle()
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    seen = 0
    for images, labels in bundle.train_loader:
        if seen >= examples:
            break
        take = min(int(images.shape[0]), examples - seen)
        batches.append(
            (
                images[:take].detach().to(device="cpu").clone(),
                labels[:take].detach().to(device="cpu").clone(),
            )
        )
        seen += take
    if seen != examples:
        raise RuntimeError(f"Expected {examples} cohort examples, got {seen}.")

    original_batches = bundle.train_batch_indices(num_epochs=1)[0]
    original_indices = [
        int(index)
        for batch in original_batches
        for index in batch
    ][:examples]
    if len(original_indices) != examples:
        raise RuntimeError("Missing deterministic original indices for the gate cohort.")
    tensor_values: list[tuple[str, torch.Tensor]] = []
    for index, (images, labels) in enumerate(batches):
        tensor_values.extend(((f"images_{index}", images), (f"labels_{index}", labels)))
    provenance = {
        "source_split": "mnist_train_55000_subset",
        "examples": examples,
        "batch_size": batch_size,
        "split_seed": int(bundle.split_seed),
        "shuffle_seed": int(bundle.shuffle_seed),
        "train_indices_sha256": bundle.train_indices_hash,
        "validation_indices_sha256": bundle.validation_indices_hash,
        "cohort_original_indices_sha256": _sha256_json(original_indices),
        "cohort_original_indices": original_indices,
        "cohort_tensor_sha256": _sha256_named_tensors(tensor_values),
    }
    return batches, provenance


def _split_prefix_batches(
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    examples: int,
    batch_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    images = torch.cat([batch[0] for batch in batches], dim=0)[:examples]
    labels = torch.cat([batch[1] for batch in batches], dim=0)[:examples]
    if int(images.shape[0]) != examples:
        raise ValueError(f"Expected {examples} examples for the gradient cohort.")
    return [
        (images[start : start + batch_size].clone(), labels[start : start + batch_size].clone())
        for start in range(0, examples, batch_size)
    ]


def _residual_audit(
    context: Mapping[str, Any],
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    expected_examples: int,
    threshold: float,
    clamp_epsilon: float,
) -> dict[str, Any]:
    device = context["parameters"][0].state.device
    free_layers = context["free_layers"]
    observations = [
        {"selected": [], "raw": [], "clamp_occupancy": []}
        for _ in free_layers
    ]
    seen = 0
    for images_cpu, _labels_cpu in batches:
        images = images_cpu.to(device)
        context["network"].set_input(images, reset=True)
        with torch.no_grad():
            context["minimizer_inference"].compute_equilibrium()
            gradients = [
                context["energy_fn"].grad_layer_fn(layer)().detach()
                for layer in free_layers
            ]
        for index, (layer, gradient) in enumerate(zip(free_layers, gradients)):
            state = layer.state.detach()
            raw = gradient.abs()
            if index == len(free_layers) - 1:
                selected = raw
                occupancy = None
            else:
                selected = _projected_perfect_diode_residual(
                    state, gradient, epsilon=clamp_epsilon
                )
                occupancy = _clamp_occupancy_per_sample(
                    state, epsilon=clamp_epsilon
                )
            observations[index]["selected"].extend(
                selected.reshape(selected.shape[0], -1)
                .amax(dim=1)
                .to(device="cpu", dtype=torch.float64)
                .tolist()
            )
            observations[index]["raw"].extend(
                raw.reshape(raw.shape[0], -1)
                .amax(dim=1)
                .to(device="cpu", dtype=torch.float64)
                .tolist()
            )
            if occupancy is not None:
                observations[index]["clamp_occupancy"].extend(
                    occupancy.to(device="cpu").tolist()
                )
        seen += int(images.shape[0])

    records = []
    for index, (layer, values) in enumerate(zip(free_layers, observations)):
        role = "output" if index == len(free_layers) - 1 else f"hidden_{index}"
        selected_stats = _stats(values["selected"])
        complete = len(values["selected"]) == expected_examples
        passed = bool(
            complete
            and math.isfinite(selected_stats["p90"])
            and selected_stats["p90"] < threshold
        )
        records.append(
            {
                "layer": str(getattr(layer, "name", role)).strip(),
                "role": role,
                "residual_mode": "raw" if role == "output" else "projected_kkt",
                "example_count": len(values["selected"]),
                "selected_residual": selected_stats,
                "raw_residual": _stats(values["raw"]),
                "clamp_occupancy": (
                    None
                    if role == "output"
                    else _stats(values["clamp_occupancy"])
                ),
                "passed": passed,
            }
        )
    passed = bool(seen == expected_examples and records and all(row["passed"] for row in records))
    return {
        "T": int(context["inference_iterations"]),
        "example_count": seen,
        "threshold": threshold,
        "comparison": "strictly_less_than",
        "passed": passed,
        "layers": records,
    }


def _cache_free_equilibria(
    context: Mapping[str, Any],
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[list[tuple[torch.Tensor, ...]], str]:
    device = context["parameters"][0].state.device
    cached: list[tuple[torch.Tensor, ...]] = []
    tensors: list[tuple[str, torch.Tensor]] = []
    for batch_index, (images_cpu, _labels_cpu) in enumerate(batches):
        context["network"].set_input(images_cpu.to(device), reset=True)
        with torch.no_grad():
            context["minimizer_inference"].compute_equilibrium()
        states = tuple(
            layer.state.detach().to(device="cpu").clone()
            for layer in context["free_layers"]
        )
        cached.append(states)
        tensors.extend(
            (f"batch_{batch_index}_layer_{layer_index}", state)
            for layer_index, state in enumerate(states)
        )
    return cached, _sha256_named_tensors(tensors)


def _gradient_observations(
    context: Mapping[str, Any],
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    free_equilibria: Sequence[Sequence[torch.Tensor]],
    *,
    zero_epsilon: float,
) -> dict[str, Any]:
    if len(batches) != len(free_equilibria):
        raise ValueError("Gradient batches/free equilibria are not matched.")
    device = context["parameters"][0].state.device
    parameters = context["parameters"]
    names = [str(getattr(parameter, "name", "")).strip() for parameter in parameters]
    initial_states = {
        name: parameter.state.detach().to(device="cpu").clone()
        for name, parameter in zip(names, parameters)
        if name in EXPECTED_CONV_WEIGHTS
    }
    estimator = Backprop(
        parameters,
        context["free_layers"],
        context["cost_fn"],
        context["minimizer_training"],
    )
    by_parameter: dict[str, list[dict[str, Any]]] = {
        name: [] for name in EXPECTED_CONV_WEIGHTS
    }
    for batch_index, ((images_cpu, labels_cpu), cached_states) in enumerate(
        zip(batches, free_equilibria)
    ):
        context["network"].set_input(images_cpu.to(device), reset=True)
        for layer, state in zip(context["free_layers"], cached_states):
            layer.state = state.to(device).clone()
        context["cost_fn"].set_target(labels_cpu.to(device))
        gradients = estimator.compute_gradient()[: len(parameters)]
        for name, gradient in zip(names, gradients):
            if name not in by_parameter:
                continue
            vector = gradient.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
            if not torch.isfinite(vector).all():
                raise RuntimeError(f"Non-finite {name} gradient at batch {batch_index}.")
            rms = float(torch.sqrt(torch.mean(vector.square())).item())
            norm = float(torch.linalg.vector_norm(vector).item())
            zero_fraction = float((vector.abs() <= zero_epsilon).to(torch.float64).mean())
            weight = initial_states[name].to(dtype=torch.float64)
            weight_rms = float(torch.sqrt(torch.mean(weight.square())).item())
            proposal_unit = rms / weight_rms if weight_rms > 0.0 else math.nan
            by_parameter[name].append(
                {
                    "batch_index": batch_index,
                    "vector": vector,
                    "gradient_l2": norm,
                    "gradient_rms": rms,
                    "zero_fraction": zero_fraction,
                    "nominal_sgd_lr1_proposal_unit": proposal_unit,
                }
            )
    return {
        "K": int(context["training_iterations"]),
        "parameter_sha256": context["parameter_sha256"],
        "by_parameter": by_parameter,
    }


def assess_gradient_gate(
    operational: Mapping[str, Sequence[Mapping[str, Any]]],
    reference: Mapping[str, Sequence[Mapping[str, Any]]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Assess K convergence and K-reference viability from matched batch rows."""

    if tuple(sorted(operational)) != tuple(sorted(reference)) or not operational:
        raise ValueError("Expected identical non-empty gradient parameter sets.")
    records = []
    for name in sorted(reference):
        current_rows = tuple(operational[name])
        reference_rows = tuple(reference[name])
        if not current_rows or len(current_rows) != len(reference_rows):
            raise ValueError(f"Expected matched non-empty gradient rows for {name}.")
        batch_comparisons = []
        for current, target in zip(current_rows, reference_rows):
            left = current["vector"].detach().to(dtype=torch.float64, device="cpu")
            right = target["vector"].detach().to(dtype=torch.float64, device="cpu")
            if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise ValueError(f"Invalid matched gradient vectors for {name}.")
            left_norm = float(current["gradient_l2"])
            right_norm = float(target["gradient_l2"])
            denominator = left_norm * right_norm
            cosine = (
                float(torch.dot(left, right).item() / denominator)
                if denominator > 0.0
                else math.nan
            )
            batch_comparisons.append(
                {
                    "batch_index": int(current["batch_index"]),
                    "gradient_l2": left_norm,
                    "reference_gradient_l2": right_norm,
                    "gradient_zero_fraction": float(current["zero_fraction"]),
                    "reference_gradient_zero_fraction": float(target["zero_fraction"]),
                    "gradient_vector_cosine": cosine,
                }
            )

        mean_norm = float(np.mean([row["gradient_l2"] for row in batch_comparisons]))
        mean_reference_norm = float(
            np.mean([row["reference_gradient_l2"] for row in batch_comparisons])
        )
        relative_norm_delta = abs(mean_norm - mean_reference_norm) / max(
            abs(mean_norm), abs(mean_reference_norm), 1.0e-30
        )
        mean_zero = float(
            np.mean([row["gradient_zero_fraction"] for row in batch_comparisons])
        )
        mean_reference_zero = float(
            np.mean([row["reference_gradient_zero_fraction"] for row in batch_comparisons])
        )
        zero_delta = abs(mean_zero - mean_reference_zero)
        mean_cosine = float(
            np.mean([row["gradient_vector_cosine"] for row in batch_comparisons])
        )

        reference_rms = [float(row["gradient_rms"]) for row in reference_rows]
        reference_zero = [float(row["zero_fraction"]) for row in reference_rows]
        proposal_units = [
            float(row["nominal_sgd_lr1_proposal_unit"]) for row in reference_rows
        ]
        median_reference_rms = _linear_quantile(reference_rms, 0.50)
        q90_reference_zero = _linear_quantile(reference_zero, 0.90)
        median_proposal_unit = _linear_quantile(proposal_units, 0.50)
        reference_viable = bool(
            math.isfinite(median_reference_rms)
            and median_reference_rms
            > float(contract["reference_median_gradient_rms_minimum"])
            and math.isfinite(q90_reference_zero)
            and q90_reference_zero
            < float(contract["reference_q90_zero_fraction_maximum"])
            and all(math.isfinite(value) and value > 0.0 for value in proposal_units)
        )
        comparison_passed = bool(
            math.isfinite(relative_norm_delta)
            and relative_norm_delta
            <= float(contract["relative_gradient_l2_norm_delta_maximum"])
            and math.isfinite(zero_delta)
            and zero_delta
            <= float(contract["absolute_zero_fraction_delta_maximum"])
            and math.isfinite(mean_cosine)
            and mean_cosine >= float(contract["gradient_vector_cosine_minimum"])
        )
        records.append(
            {
                "parameter": name,
                "batch_count": len(batch_comparisons),
                "gradient_l2_mean": mean_norm,
                "reference_gradient_l2_mean": mean_reference_norm,
                "relative_gradient_l2_norm_delta": relative_norm_delta,
                "gradient_zero_fraction_mean": mean_zero,
                "reference_gradient_zero_fraction_mean": mean_reference_zero,
                "absolute_zero_fraction_delta": zero_delta,
                "gradient_vector_cosine_mean": mean_cosine,
                "reference_gradient_rms_median": median_reference_rms,
                "reference_gradient_zero_fraction_q90": q90_reference_zero,
                "reference_nominal_sgd_lr1_proposal_unit_median": median_proposal_unit,
                "reference_proposal_units_all_finite_positive": all(
                    math.isfinite(value) and value > 0.0 for value in proposal_units
                ),
                "reference_viable": reference_viable,
                "comparison_passed": comparison_passed,
                "passed": reference_viable and comparison_passed,
                "batches": batch_comparisons,
            }
        )
    return {
        "reference_viable": all(row["reference_viable"] for row in records),
        "comparison_passed": all(row["comparison_passed"] for row in records),
        "passed": all(row["passed"] for row in records),
        "parameters": records,
    }


def run_conv3_operating_point_gate(
    source_config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    output_path: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the accepted Conv3 T8/K8 audit on a shared bounded-uniform asset."""

    source = Path(source_config_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Missing source config or checkpoint: {source}, {checkpoint}.")
    config = json.loads(source.read_text(encoding="utf-8"))
    model_cfg = _model_config(config)
    _validate_source_contract(config, model_cfg)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested unavailable CUDA device {resolved_device}.")

    contract = dict(DEFAULT_CONTRACT)
    if smoke:
        contract["residual_examples"] = contract["residual_batch_size"]
        contract["gradient_examples"] = contract["gradient_batch_size"]
    residual_batches, cohort = _load_training_cohort(
        config,
        batch_size=int(contract["residual_batch_size"]),
        examples=int(contract["residual_examples"]),
    )
    gradient_batches = _split_prefix_batches(
        residual_batches,
        examples=int(contract["gradient_examples"]),
        batch_size=int(contract["gradient_batch_size"]),
    )
    cohort["gradient_prefix_examples"] = int(contract["gradient_examples"])
    cohort["gradient_batch_size"] = int(contract["gradient_batch_size"])
    cohort["gradient_prefix_original_indices_sha256"] = _sha256_json(
        cohort["cohort_original_indices"][: int(contract["gradient_examples"])]
    )

    operational = _build_model_context(
        config,
        checkpoint,
        device=resolved_device,
        inference_iterations=int(contract["selected_T"]),
        training_iterations=int(contract["selected_K"]),
    )
    reference_t = _build_model_context(
        config,
        checkpoint,
        device=resolved_device,
        inference_iterations=int(contract["reference_T"]),
        training_iterations=int(contract["selected_K"]),
    )
    if operational["parameter_sha256"] != reference_t["parameter_sha256"]:
        raise RuntimeError("T8 and T64 model contexts did not load identical parameters.")

    selected_residual = _residual_audit(
        operational,
        residual_batches,
        expected_examples=int(contract["residual_examples"]),
        threshold=float(contract["residual_p90_maximum"]),
        clamp_epsilon=float(contract["clamp_epsilon"]),
    )
    reference_residual = _residual_audit(
        reference_t,
        residual_batches,
        expected_examples=int(contract["residual_examples"]),
        threshold=float(contract["residual_p90_maximum"]),
        clamp_epsilon=float(contract["clamp_epsilon"]),
    )
    residual_passed = bool(selected_residual["passed"] and reference_residual["passed"])

    gradient_result = None
    if residual_passed:
        free_equilibria, free_sha256 = _cache_free_equilibria(
            operational, gradient_batches
        )
        operational_gradients = _gradient_observations(
            operational,
            gradient_batches,
            free_equilibria,
            zero_epsilon=float(contract["gradient_zero_epsilon"]),
        )
        reference_k = _build_model_context(
            config,
            checkpoint,
            device=resolved_device,
            inference_iterations=int(contract["selected_T"]),
            training_iterations=int(contract["reference_K"]),
        )
        if operational["parameter_sha256"] != reference_k["parameter_sha256"]:
            raise RuntimeError("K8 and K64 model contexts did not load identical parameters.")
        reference_gradients = _gradient_observations(
            reference_k,
            gradient_batches,
            free_equilibria,
            zero_epsilon=float(contract["gradient_zero_epsilon"]),
        )
        assessment = assess_gradient_gate(
            operational_gradients["by_parameter"],
            reference_gradients["by_parameter"],
            contract,
        )
        gradient_result = {
            "operational": {"T": int(contract["selected_T"]), "K": int(contract["selected_K"])},
            "reference": {"T": int(contract["selected_T"]), "K": int(contract["reference_K"])},
            "shared_free_equilibria_sha256": free_sha256,
            "same_free_equilibria_reused": True,
            **assessment,
        }

    reference_viable = bool(
        gradient_result is not None and gradient_result["reference_viable"]
    )
    gradient_passed = bool(gradient_result is not None and gradient_result["passed"])
    security_passed = bool(residual_passed and gradient_passed)
    if not residual_passed:
        status = "unresolved_tk_residual"
    elif not reference_viable:
        status = "unresolved_tk_gradient_viability"
    elif not gradient_passed:
        status = "unresolved_fixed_tk_gradient_mismatch"
    else:
        status = "complete"

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "security_passed": security_passed,
        "scientifically_complete": not smoke,
        "smoke": smoke,
        "contract": contract,
        "source_config": str(source),
        "source_config_sha256": _sha256_file(source),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "parameter_sha256": operational["parameter_sha256"],
        "device": str(resolved_device),
        "dataset_cohort": cohort,
        "residual": {
            "selected": selected_residual,
            "reference_sentinel": reference_residual,
            "passed": residual_passed,
        },
        "gradient": gradient_result,
        "official_test_read": False,
    }
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_conv3_operating_point_gate(
        args.config,
        args.checkpoint,
        device=args.device,
        output_path=args.output,
        smoke=args.smoke,
    )
    print(
        "CONV3_OPERATING_POINT_GATE "
        f"status={result['status']} security_passed={result['security_passed']} "
        f"smoke={result['smoke']} output={Path(args.output).expanduser().resolve()}",
        flush=True,
    )
    if result["security_passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
