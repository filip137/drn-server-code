#!/usr/bin/env python3
"""Replay active Conv checkpoints under persistent conductance write noise.

This is a validation-only deployment diagnostic.  It builds the exact source
runtime, loads the best-validation checkpoint, writes each conductance array
once, and holds that programmed realization fixed throughout evaluation.  The
official MNIST test split is never constructed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LABS_ROOT = REPOSITORY_ROOT / "labs"
for import_root in (REPOSITORY_ROOT, LABS_ROOT):
    value = str(import_root)
    if value not in sys.path:
        sys.path.insert(0, value)

import model  # noqa: E402,F401 - anchor the repository-local package.
from custom_classes import FlexibleDeepResistiveEnergy  # noqa: E402
from experiments.reporting import (  # noqa: E402
    append_metric,
    atomic_write_json,
    complete_run,
    fail_run,
    runtime_context,
    sha256_file,
    start_run,
    update_status_progress,
    validate_run,
)
from experiments.weight_write_noise import (  # noqa: E402
    MODEL_VERSION as WRITE_NOISE_MODEL,
    apply_weight_write_noise,
)
from labs.mnist_train import (  # noqa: E402
    _build_tracking_minimizer,
    _convert_energy_runtime_dtype,
    _require_minimizer_config,
    _reset_name_counters,
    _resolve_callable,
    _resolve_dataset_config,
    _runtime_dtype_from_config,
    _set_seed,
    _validate_diode_param_config,
)
from model.function.cost import SquaredError, SquaredErrorPairedOutputs  # noqa: E402
from model.function.network import Network  # noqa: E402


EVIDENCE_CLASS = "ordinary_mnist_weight_write_noise_diagnostic"
RUNNER_SCHEMA = "conv-weight-write-noise-validation-replay/v1"
SUMMARY_SCHEMA = "conv-weight-write-noise-study-summary/v1"
WEIGHT_NAMES = ("ConvWeight_", "DenseWeight_")


def _implementation_hashes() -> dict[str, str]:
    return {
        "evaluation_runner_sha256": sha256_file(Path(__file__).resolve()),
        "write_noise_implementation_sha256": sha256_file(
            REPOSITORY_ROOT / "experiments" / "weight_write_noise.py"
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _git_state() -> dict[str, Any]:
    status = _git_output("status", "--porcelain")
    return {
        "commit": _git_output("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _canonical_name(parameter: Any) -> str:
    return str(getattr(parameter, "name", "")).strip()


def _model_config(source_config: Mapping[str, Any]) -> dict[str, Any]:
    lab = source_config.get("lab")
    if not isinstance(lab, Mapping) or not isinstance(lab.get("model_key"), str):
        raise ValueError("Source config must declare lab.model_key.")
    model_key = str(lab["model_key"])
    base = source_config.get("model_base")
    overrides = source_config.get("model_overrides")
    if not isinstance(base, Mapping) or not isinstance(overrides, Mapping):
        raise ValueError("Source config must declare model_base and model_overrides.")
    selected = overrides.get(model_key)
    if not isinstance(selected, Mapping):
        raise ValueError(f"Missing model_overrides[{model_key!r}].")
    model_config = {**dict(base), **dict(selected)}
    _validate_diode_param_config(model_config)
    _require_minimizer_config(model_config)
    return model_config


def _indexed_checkpoint_hash(result: Mapping[str, Any], name: str) -> str:
    matches = [
        artifact
        for artifact in result.get("artifacts", [])
        if artifact.get("kind") == "checkpoint"
        and Path(str(artifact.get("path", ""))).name == name
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
        raise ValueError(f"Expected exactly one indexed checkpoint named {name!r}.")
    return str(matches[0]["sha256"])


def _infer_labels(source_config: Mapping[str, Any], model_config: Mapping[str, Any]) -> dict[str, str]:
    depth = sum(
        1
        for stage in model_config.get("conv_pipeline", []) or []
        if stage.get("mode") == "convolution"
    )
    voltage = float(model_config["voltage_amp"])
    current = float(model_config["current_amp"])
    if (voltage, current) == (1.0, 1.0):
        scheme = "baseline"
    elif (voltage, current) == (4.0, 1.0):
        scheme = "ours"
    elif (voltage, current) == (4.0, 0.25):
        scheme = "legacy"
    else:
        scheme = f"v{voltage:g}_c{current:g}"
    lower = float(model_config["weight_min"])
    upper = float(model_config["weight_max"])
    if (lower, upper) == (0.0, 100.0):
        weight_contract = "wide_0_100"
    elif math.isclose(lower, 1.0e-5) and math.isclose(upper, 1.0e-4):
        weight_contract = "bounded_1e-5_1e-4"
    else:
        weight_contract = f"custom_{lower:g}_{upper:g}"
    optimizer = source_config.get("optimizer", {})
    optimizer_name = (
        str(optimizer.get("name"))
        if isinstance(optimizer, Mapping) and optimizer.get("name") is not None
        else "unknown"
    )
    return {
        "architecture": f"conv{depth}",
        "scheme": scheme,
        "optimizer": optimizer_name,
        "weight_contract": weight_contract,
        "training_algorithm": str(source_config.get("training_algorithm", "unknown")),
    }


def _validate_active_source_scope(
    *,
    result: Mapping[str, Any],
    source_dataset: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> None:
    """Reject historical/custom sources that this active evaluator cannot label."""

    if result.get("smoke") is not False:
        raise ValueError("Write-noise replay requires a complete non-smoke source run.")
    expected_dataset = {
        "key": "mnist",
        "variant": "ordinary",
        "factory": "labs.datasets.MnistTrainValidationDataset",
        "evaluation_split": "validation",
    }
    mismatches = {
        key: {"expected": expected, "observed": source_dataset.get(key)}
        for key, expected in expected_dataset.items()
        if source_dataset.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Active write-noise replay requires ordinary-MNIST train/validation "
            f"sources; mismatches={mismatches!r}."
        )
    if model_config.get("non_linearity") != "perfect_diode":
        raise ValueError(
            "Active write-noise replay requires non_linearity='perfect_diode'."
        )
    depth = sum(
        1
        for stage in model_config.get("conv_pipeline", []) or []
        if stage.get("mode") == "convolution"
    )
    if depth not in {1, 2, 3}:
        raise ValueError(
            "Active write-noise replay supports Conv1/Conv2/Conv3 sources; "
            f"observed convolution depth {depth}."
        )


def _source_inventory(source_run: Path) -> dict[str, Any]:
    source_run = Path(source_run).expanduser().resolve()
    errors = validate_run(source_run)
    if errors:
        raise ValueError(
            f"Invalid source reporting bundle {source_run}: {'; '.join(errors)}"
        )
    required = {
        "manifest": source_run / "manifest.json",
        "result": source_run / "result.json",
        "metrics": source_run / "metrics.json",
        "config": source_run / "config.used.json",
        "checkpoint": source_run / "best_model.pt",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Source run {source_run} lacks required files: {', '.join(missing)}."
        )
    manifest = _read_json(required["manifest"])
    result = _read_json(required["result"])
    metrics = _read_json(required["metrics"])
    source_config = _read_json(required["config"])
    source_dataset = result.get("dataset")
    if not isinstance(source_dataset, Mapping) or source_dataset.get("official_test_read") is not False:
        raise ValueError("Source result must prove official_test_read=false.")
    terminal = result.get("terminal_metrics")
    if not isinstance(terminal, Mapping) or not isinstance(terminal.get("validation"), Mapping):
        raise ValueError("Source result must contain best-validation metrics.")
    try:
        source_best_validation_accuracy = float(
            terminal["validation"]["best_accuracy"]
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError(
            "Source result must contain a finite validation.best_accuracy."
        ) from None
    if not math.isfinite(source_best_validation_accuracy) or not (
        0.0 <= source_best_validation_accuracy <= 1.0
    ):
        raise ValueError(
            "Source result must contain validation.best_accuracy in [0, 1]."
        )
    evaluation = source_config.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("checkpoint_selection") != (
        "maximum_validation_accuracy"
    ):
        raise ValueError("Source config must select the maximum-validation checkpoint.")
    model_config = _model_config(source_config)
    _validate_active_source_scope(
        result=result,
        source_dataset=source_dataset,
        model_config=model_config,
    )
    lower = model_config.get("weight_min")
    upper = model_config.get("weight_max")
    try:
        lower_value = float(lower)
        upper_value = float(upper)
    except (TypeError, ValueError):
        raise ValueError("Write-noise replay requires explicit finite weight bounds.") from None
    if not math.isfinite(lower_value) or not math.isfinite(upper_value) or lower_value >= upper_value:
        raise ValueError("Write-noise replay requires finite weight_min < weight_max.")
    checkpoint_hash = sha256_file(required["checkpoint"])
    expected_checkpoint_hash = _indexed_checkpoint_hash(result, "best_model.pt")
    if checkpoint_hash != expected_checkpoint_hash:
        raise ValueError("Best checkpoint hash differs from its source result index.")
    hashes = {name: sha256_file(path) for name, path in required.items()}
    return {
        "run_dir": source_run,
        "paths": required,
        "hashes": hashes,
        "manifest": manifest,
        "result": result,
        "metrics": metrics,
        "source_config": source_config,
        "model_config": model_config,
        "labels": _infer_labels(source_config, model_config),
        "source_dataset": dict(source_dataset),
        "source_checkpoint_sha256": checkpoint_hash,
        "source_best_validation_accuracy": source_best_validation_accuracy,
    }


def _verify_source_unchanged(source: Mapping[str, Any]) -> None:
    current = {
        name: sha256_file(path)
        for name, path in source["paths"].items()
    }
    if current != source["hashes"]:
        raise RuntimeError(f"Source run changed during replay: {source['run_dir']}.")


def _build_runtime(
    source: Mapping[str, Any],
    *,
    device: torch.device,
    dataset_root: Path | None,
) -> dict[str, Any]:
    source_config = source["source_config"]
    model_config = source["model_config"]
    _set_seed(int(source_config.get("seed", 0)))
    _reset_name_counters()
    energy_fn = FlexibleDeepResistiveEnergy(
        layer_shapes=[tuple(shape) for shape in model_config["layer_shapes"]],
        conv_pipeline=model_config.get("conv_pipeline") or [],
        pooling_mode=model_config.get("pooling_mode"),
        weight_gains=model_config["weight_gains"],
        input_gain=model_config["input_gain"],
        non_linearity=model_config["non_linearity"],
        exponential_diode_param=model_config["exponential_diode_param"],
        quadratic_diode_param=model_config["quadratic_diode_param"],
        hard_sigmoid_param=model_config["hard_sigmoid_param"],
        voltage_amp=model_config["voltage_amp"],
        current_amp=model_config["current_amp"],
        weight_min=model_config["weight_min"],
        weight_max=model_config["weight_max"],
        weight_init_mode=model_config.get("weight_init_mode", "kaiming_uniform"),
        input_mode=source_config.get("input_mode", "train"),
        trainable_amplification=bool(model_config.get("trainable_amplification", False)),
        amplification_min=model_config.get("amplification_min", 1.0e-6),
        amplification_max=model_config.get("amplification_max"),
    )
    energy_fn.set_device(device)
    runtime_dtype_name, runtime_dtype = _runtime_dtype_from_config(source_config)
    _convert_energy_runtime_dtype(energy_fn, runtime_dtype)
    energy_fn.load(source["paths"]["checkpoint"])
    parameters = list(energy_fn.params())
    parameter_names = [_canonical_name(parameter) for parameter in parameters]
    configured_order = source_config.get("parameter_order")
    if configured_order is not None and list(configured_order) != parameter_names:
        raise ValueError(
            f"Runtime parameter order differs from the source config: "
            f"runtime={parameter_names!r}, configured={configured_order!r}."
        )
    selected_weight_names = [
        name for name in parameter_names if name.startswith(WEIGHT_NAMES)
    ]
    if not selected_weight_names:
        raise ValueError("Runtime has no ConvWeight_* or DenseWeight_* parameters.")

    network = Network(energy_fn)
    free_layers = list(network.free_layers())
    output_layer = energy_fn.layers()[-1]
    output_dim = int(output_layer.shape[0])
    if output_dim == 10:
        cost_fn = SquaredError(output_layer)
    elif output_dim == 20:
        cost_fn = SquaredErrorPairedOutputs(output_layer, num_classes=10)
    else:
        raise ValueError(f"Unsupported output dimension {output_dim}; expected 10 or 20.")
    minimizer = _build_tracking_minimizer(
        energy_fn,
        free_layers,
        model_config,
        source_config["energy_minimizer"]["mode"],
        num_iterations=int(model_config["num_iterations_inference"]),
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
        adaptive_equilibrium=False,
    )

    dataset_key = str(source_config["lab"]["dataset_key"])
    _, dataset_config = _resolve_dataset_config(source_config, dataset_key)
    dataset_params = _json_clone(dataset_config["params"])
    dataset_params["device"] = device
    dataset_params["download"] = False
    if dataset_root is not None:
        dataset_params["root"] = str(dataset_root.expanduser().resolve())
    elif "root" in dataset_params:
        dataset_params["root"] = os.path.expanduser(str(dataset_params["root"]))
    dataset_builder = _resolve_callable(dataset_config["factory"])(**dataset_params)
    loaders = dataset_builder.build()
    if not hasattr(loaders, "validation_loader"):
        raise ValueError(
            "Active write-noise replay requires a train/validation-only dataset "
            "builder with validation_loader; tuple/test loaders are rejected."
        )
    validation_hash = getattr(loaders, "validation_indices_hash", None)
    source_provenance = source["metrics"].get("dataset_provenance", {})
    expected_validation_hash = source_provenance.get("validation_indices_sha256")
    if expected_validation_hash is not None and validation_hash != expected_validation_hash:
        raise ValueError("Validation split hash differs from the source training run.")

    return {
        "energy_fn": energy_fn,
        "network": network,
        "free_layers": free_layers,
        "output_layer": output_layer,
        "output_dim": output_dim,
        "cost_fn": cost_fn,
        "minimizer": minimizer,
        "parameters": parameters,
        "base_states": [parameter.state.detach().clone() for parameter in parameters],
        "validation_loader": loaders.validation_loader,
        "validation_indices_sha256": validation_hash,
        "runtime_dtype": runtime_dtype_name,
        "input_dtype": runtime_dtype,
        "dataset_factory": dataset_config["factory"],
        "dataset_params": dataset_params,
        "inference_iterations": int(model_config["num_iterations_inference"]),
    }


def _restore_parameters(runtime: Mapping[str, Any]) -> None:
    parameters = runtime["parameters"]
    base_states = runtime["base_states"]
    if len(parameters) != len(base_states):
        raise ValueError("Parameter/base-state count mismatch.")
    for parameter, base in zip(parameters, base_states, strict=True):
        parameter.state = base.detach().clone()


def _scores(output: torch.Tensor) -> torch.Tensor:
    if output.shape[1] == 10:
        return output
    if output.shape[1] == 20:
        paired = output.reshape(output.shape[0], 10, 2)
        return paired[..., 0] - paired[..., 1]
    raise ValueError(f"Unsupported output shape {tuple(output.shape)}.")


def _margin_sum(scores: torch.Tensor, labels: torch.Tensor) -> float:
    true_score = scores.gather(1, labels[:, None]).squeeze(1)
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask.scatter_(1, labels[:, None], False)
    other = scores.masked_fill(~mask, float("-inf")).max(dim=1).values
    return float((true_score - other).sum().detach().cpu())


def _perfect_diode_layer_residual(
    layer: Any,
    gradient: torch.Tensor,
    *,
    output_layer: Any,
    epsilon: float,
) -> torch.Tensor:
    state = layer.state.detach()
    gradient = gradient.detach()
    if layer is output_layer:
        return gradient.abs().reshape(gradient.shape[0], -1).amax(dim=1)
    if state.ndim < 2 or state.shape[1] % 2:
        raise ValueError(f"Expected a paired perfect-diode layer, got {tuple(state.shape)}.")
    half = state.shape[1] // 2
    positive_state = state[:, :half]
    negative_state = state[:, half:]
    positive_gradient = gradient[:, :half]
    negative_gradient = gradient[:, half:]
    positive_residual = torch.where(
        positive_state > epsilon,
        positive_gradient.abs(),
        torch.clamp(-positive_gradient, min=0.0),
    )
    negative_residual = torch.where(
        negative_state < -epsilon,
        negative_gradient.abs(),
        torch.clamp(negative_gradient, min=0.0),
    )
    merged = torch.cat(
        [
            positive_residual.reshape(gradient.shape[0], -1),
            negative_residual.reshape(gradient.shape[0], -1),
        ],
        dim=1,
    )
    return merged.amax(dim=1)


def _saturation_counts(
    layers: Sequence[Any],
    *,
    output_layer: Any,
    epsilon: float,
) -> tuple[int, int]:
    saturated = 0
    total = 0
    for layer in layers:
        if layer is output_layer:
            continue
        state = layer.state.detach()
        if state.ndim < 2 or state.shape[1] % 2:
            continue
        half = state.shape[1] // 2
        saturated += int((state[:, :half] <= epsilon).sum().item())
        saturated += int((state[:, half:] >= -epsilon).sum().item())
        total += int(state.numel())
    return saturated, total


def _evaluate_validation(
    runtime: Mapping[str, Any],
    *,
    max_batches: int | None,
    residual_epsilon: float = 1.0e-8,
) -> dict[str, Any]:
    running_loss = 0.0
    running_correct = 0
    running_margin = 0.0
    seen = 0
    saturated = 0
    saturation_total = 0
    residuals: dict[str, list[np.ndarray]] = {
        str(layer.name): [] for layer in runtime["free_layers"]
    }
    device = runtime["base_states"][0].device
    # The equilibrium diagnostic below calls energy_fn.grad_layer_fn(), whose
    # generic implementation uses autograd even during evaluation.  Keep
    # gradient recording enabled for that residual calculation; no parameter
    # optimizer is present in this read-only replay.
    with torch.enable_grad():
        for batch_index, (images, labels) in enumerate(runtime["validation_loader"]):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device=device, dtype=runtime["input_dtype"])
            labels = labels.to(device=device)
            if not bool(torch.isfinite(images).all()):
                raise FloatingPointError("Non-finite validation input.")
            runtime["network"].set_input(images, reset=True)
            runtime["minimizer"].compute_equilibrium()
            for layer in runtime["free_layers"]:
                if not bool(torch.isfinite(layer.state).all()):
                    raise FloatingPointError(f"Non-finite equilibrium state in {layer.name}.")
            runtime["cost_fn"].set_target(labels)
            cost = runtime["cost_fn"].eval()
            if not bool(torch.isfinite(cost).all()):
                raise FloatingPointError("Non-finite validation cost.")
            scores = _scores(runtime["output_layer"].state.detach())
            if not bool(torch.isfinite(scores).all()):
                raise FloatingPointError("Non-finite validation scores.")
            batch_size = int(images.shape[0])
            running_loss += float(cost.sum().detach().cpu())
            running_correct += int((~runtime["cost_fn"].error_fn()).sum().item())
            running_margin += _margin_sum(scores, labels)
            seen += batch_size
            batch_saturated, batch_saturation_total = _saturation_counts(
                runtime["free_layers"],
                output_layer=runtime["output_layer"],
                epsilon=residual_epsilon,
            )
            saturated += batch_saturated
            saturation_total += batch_saturation_total
            for layer in runtime["free_layers"]:
                gradient = runtime["energy_fn"].grad_layer_fn(layer)()
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError(f"Non-finite energy residual in {layer.name}.")
                maxima = _perfect_diode_layer_residual(
                    layer,
                    gradient,
                    output_layer=runtime["output_layer"],
                    epsilon=residual_epsilon,
                )
                residuals[str(layer.name)].append(
                    maxima.detach().cpu().to(torch.float64).numpy()
                )
    if seen == 0:
        raise RuntimeError("Validation loader yielded zero examples.")
    layer_residuals = {}
    for name, chunks in residuals.items():
        values = np.concatenate(chunks)
        layer_residuals[name] = {
            "mean": float(np.mean(values)),
            "p90": float(np.quantile(values, 0.90)),
            "maximum": float(np.max(values)),
        }
    metrics = {
        "split": "validation",
        "official_test_read": False,
        "examples": seen,
        "batches": sum(len(chunks) for chunks in residuals.values()) // max(len(residuals), 1),
        "loss": running_loss / seen,
        "accuracy": running_correct / seen,
        "mean_output_margin": running_margin / seen,
        "hidden_saturation_fraction": saturated / saturation_total if saturation_total else 0.0,
        "inference_iterations": int(runtime["inference_iterations"]),
        "projected_kkt_residual_by_layer": layer_residuals,
    }
    if not all(
        math.isfinite(float(metrics[key]))
        for key in ("loss", "accuracy", "mean_output_margin", "hidden_saturation_fraction")
    ):
        raise FloatingPointError("Non-finite aggregate validation metrics.")
    return metrics


def _evaluate_validation_outcome(
    runtime: Mapping[str, Any],
    *,
    max_batches: int | None,
) -> dict[str, Any]:
    """Return finite metrics or retain non-finiteness as a scientific outcome."""

    try:
        metrics = _evaluate_validation(runtime, max_batches=max_batches)
    except FloatingPointError as error:
        return {
            "outcome": "nonfinite",
            "split": "validation",
            "official_test_read": False,
            "inference_iterations": int(runtime["inference_iterations"]),
            "max_validation_batches": max_batches,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    return {"outcome": "finite", **metrics}


def _verify_clean_reference(
    source: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    max_validation_batches: int | None,
) -> None:
    if validation.get("outcome") != "finite":
        raise RuntimeError(
            "The eta=0 clean reference did not produce finite validation metrics; "
            "refusing to interpret noisy cases against a broken replay baseline."
        )
    if max_validation_batches is not None:
        return
    observed = float(validation["accuracy"])
    expected = float(source["source_best_validation_accuracy"])
    if observed != expected:
        raise RuntimeError(
            "The eta=0 full-validation accuracy does not reproduce the source "
            f"best checkpoint: observed={observed!r}, expected={expected!r}."
        )


def _unique_values(values: Sequence[Any], *, cast) -> list[Any]:
    output = []
    seen = set()
    for value in values:
        converted = cast(value)
        if converted not in seen:
            seen.add(converted)
            output.append(converted)
    return output


def noise_cases(
    std_fractions: Sequence[float],
    programming_seeds: Sequence[int],
) -> list[tuple[float, int]]:
    fractions = _unique_values(std_fractions, cast=float)
    seeds = _unique_values(programming_seeds, cast=int)
    if not fractions:
        raise ValueError("At least one write-noise std fraction is required.")
    if not seeds:
        raise ValueError("At least one programming seed is required.")
    for fraction in fractions:
        if not math.isfinite(fraction) or fraction < 0.0:
            raise ValueError("Write-noise std fractions must be finite and nonnegative.")
    if 0.0 not in fractions:
        raise ValueError(
            "Every write-noise replay must include eta=0 as its clean reference."
        )
    for seed in seeds:
        if seed < 0 or seed > (1 << 63) - 1:
            raise ValueError(
                "Programming seeds must be integers in [0, 2^63-1]."
            )
    fractions = [0.0, *(fraction for fraction in fractions if fraction != 0.0)]
    cases = []
    for fraction in fractions:
        selected_seeds = seeds[:1] if fraction == 0.0 else seeds
        cases.extend((fraction, seed) for seed in selected_seeds)
    return cases


def _fraction_label(value: float) -> str:
    text = format(float(value), ".12g").lower()
    text = text.replace("-", "m").replace("+", "").replace(".", "p")
    return re.sub(r"[^a-z0-9_]", "_", text)


def _case_contract_sha256(
    source: Mapping[str, Any],
    std_fraction: float,
    seed: int,
    *,
    max_validation_batches: int | None,
    evaluation_device: str,
) -> str:
    payload = {
        "runner_schema": RUNNER_SCHEMA,
        "write_noise_model": WRITE_NOISE_MODEL,
        "implementation_hashes": _implementation_hashes(),
        "source_hashes": dict(sorted(source["hashes"].items())),
        "std_fraction": float(std_fraction),
        "programming_seed": int(seed),
        "max_validation_batches": max_validation_batches,
        "evaluation_device": str(evaluation_device),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _case_identity(
    source: Mapping[str, Any],
    std_fraction: float,
    seed: int,
    *,
    max_validation_batches: int | None,
    evaluation_device: str,
) -> tuple[str, str]:
    source_arm = str(source["manifest"]["arm_id"])
    contract_sha256 = _case_contract_sha256(
        source,
        std_fraction,
        seed,
        max_validation_batches=max_validation_batches,
        evaluation_device=evaluation_device,
    )
    noise = f"eta_{_fraction_label(std_fraction)}_write_seed_{seed}"
    budget = (
        "full_validation"
        if max_validation_batches is None
        else f"smoke_{max_validation_batches}_validation_batches"
    )
    arm_id = f"{source_arm}__{noise}__{budget}"
    source_run_name = Path(source["run_dir"]).name
    run_id = f"{source_run_name}__{noise}__{budget}__{contract_sha256[:12]}"
    return arm_id, run_id


def _require_matching_case_contract(
    existing_manifest: Mapping[str, Any],
    requested_manifest: Mapping[str, Any],
) -> None:
    existing = existing_manifest.get("configuration", {}).get(
        "case_contract_sha256"
    )
    requested = requested_manifest.get("configuration", {}).get(
        "case_contract_sha256"
    )
    if not isinstance(existing, str) or existing != requested:
        raise RuntimeError(
            "Existing complete run does not match the requested case contract; "
            f"existing={existing!r}, requested={requested!r}."
        )


def _case_manifest(
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    study_id: str,
    arm_id: str,
    run_id: str,
    std_fraction: float,
    programming_seed: int,
    command: Sequence[str],
    target: str,
    max_validation_batches: int | None,
) -> dict[str, Any]:
    return {
        "study_id": study_id,
        "run_id": run_id,
        "arm_id": arm_id,
        "evidence_class": EVIDENCE_CLASS,
        "smoke": max_validation_batches is not None,
        "configuration": {
            "schema_version": RUNNER_SCHEMA,
            "implementation_hashes": _implementation_hashes(),
            "case_contract_sha256": _case_contract_sha256(
                source,
                std_fraction,
                programming_seed,
                max_validation_batches=max_validation_batches,
                evaluation_device=str(runtime["base_states"][0].device),
            ),
            "source_run": str(source["run_dir"]),
            "source_arm_id": source["manifest"]["arm_id"],
            "source_checkpoint_role": "best_validation",
            "source_configuration": source["source_config"],
            "labels": source["labels"],
            "weight_write_noise": {
                "model": WRITE_NOISE_MODEL,
                "formula": (
                    "G_written=clip(G_target+eta*(G_max-G_min)*Z,G_min,G_max); "
                    "Z~Normal(0,1)"
                ),
                "std_fraction": float(std_fraction),
                "programming_seed": int(programming_seed),
                "persistence": "program_once_hold_for_complete_validation_replay",
                "boundary": "hard_clip_to_each_weight_parameter_bounds",
                "parameters": ["ConvWeight_*", "DenseWeight_*"],
                "biases_included": False,
                "common_random_numbers_across_noise_levels": True,
                "common_random_numbers_across_schemes": (
                    "for identical canonical parameter names and shapes"
                ),
            },
            "max_validation_batches": max_validation_batches,
            "evaluation_device": str(runtime["base_states"][0].device),
        },
        "inputs": {
            "source_files": {
                name: {"path": str(source["paths"][name]), "sha256": digest}
                for name, digest in source["hashes"].items()
            }
        },
        "dataset": {
            "key": source["source_dataset"].get("key", "mnist"),
            "factory": runtime["dataset_factory"],
            "evaluation_split": "validation",
            "validation_indices_sha256": runtime["validation_indices_sha256"],
            "official_test_read": False,
        },
        "command": list(command),
        "git": _git_state(),
        "runtime": runtime_context(target=target),
    }


def _summary_row(
    source: Mapping[str, Any],
    *,
    run_id: str,
    std_fraction: float,
    programming_seed: int,
    terminal: Mapping[str, Any] | None,
    state: str,
) -> dict[str, Any]:
    validation = {} if terminal is None else terminal.get("validation", {})
    noise = {} if terminal is None else terminal.get("weight_write_noise", {})
    return {
        **source["labels"],
        "source_run": str(source["run_dir"]),
        "source_checkpoint_sha256": source["source_checkpoint_sha256"],
        "run_id": run_id,
        "state": state,
        "std_fraction": float(std_fraction),
        "programming_seed": int(programming_seed),
        "validation_outcome": validation.get("outcome"),
        "validation_examples": validation.get("examples"),
        "validation_loss": validation.get("loss"),
        "validation_accuracy": validation.get("accuracy"),
        "mean_output_margin": validation.get("mean_output_margin"),
        "hidden_saturation_fraction": validation.get("hidden_saturation_fraction"),
        "realized_error_rms_fraction": noise.get("rms_error_fraction"),
        "lower_clipping_fraction": noise.get("lower_clipping_fraction"),
        "upper_clipping_fraction": noise.get("upper_clipping_fraction"),
        "official_test_read": False,
    }


def _write_summary(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = {
        "schema_version": SUMMARY_SCHEMA,
        "study_id": output_root.name,
        "write_noise_model": WRITE_NOISE_MODEL,
        "official_test_read": False,
        "case_count": len(rows),
        "complete_count": sum(row["state"] == "complete" for row in rows),
        "failed_count": sum(row["state"] == "failed" for row in rows),
        "cases": list(rows),
    }
    atomic_write_json(output_root / "summary.json", payload)
    if not rows:
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _run_source(
    source: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    device: torch.device,
    study_id: str,
    command: Sequence[str],
) -> tuple[list[dict[str, Any]], bool]:
    runtime = _build_runtime(
        source,
        device=device,
        dataset_root=args.dataset_root,
    )
    rows = []
    failed = False
    for std_fraction, programming_seed in noise_cases(
        args.std_fractions, args.programming_seeds
    ):
        arm_id, run_id = _case_identity(
            source,
            std_fraction,
            programming_seed,
            max_validation_batches=args.max_validation_batches,
            evaluation_device=str(device),
        )
        run_dir = args.output_root / "runs" / run_id
        manifest = _case_manifest(
            source,
            runtime,
            study_id=study_id,
            arm_id=arm_id,
            run_id=run_id,
            std_fraction=std_fraction,
            programming_seed=programming_seed,
            command=command,
            target=args.target,
            max_validation_batches=args.max_validation_batches,
        )
        if (run_dir / "result.json").is_file() and validate_run(run_dir) == []:
            existing_manifest = _read_json(run_dir / "manifest.json")
            _require_matching_case_contract(existing_manifest, manifest)
            result = _read_json(run_dir / "result.json")
            rows.append(
                _summary_row(
                    source,
                    run_id=run_id,
                    std_fraction=std_fraction,
                    programming_seed=programming_seed,
                    terminal=result.get("terminal_metrics", {}),
                    state="complete",
                )
            )
            print(f"[skip] complete {run_id}", flush=True)
            continue
        start_run(run_dir, manifest)
        try:
            _restore_parameters(runtime)
            update_status_progress(
                run_dir,
                {
                    "stage": "programming_weights",
                    "std_fraction": float(std_fraction),
                    "programming_seed": int(programming_seed),
                },
            )
            receipt = apply_weight_write_noise(
                runtime["parameters"],
                runtime["base_states"],
                eta=float(std_fraction),
                programming_seed=int(programming_seed),
            )
            receipt = {
                **receipt,
                "source_run": str(source["run_dir"]),
                "source_checkpoint_sha256": source["source_checkpoint_sha256"],
                "official_test_read": False,
            }
            atomic_write_json(run_dir / "noise_receipt.json", receipt)
            update_status_progress(
                run_dir,
                {
                    "stage": "validation_replay",
                    "std_fraction": float(std_fraction),
                    "programming_seed": int(programming_seed),
                },
            )
            validation = _evaluate_validation_outcome(
                runtime,
                max_batches=args.max_validation_batches,
            )
            if std_fraction == 0.0:
                _verify_clean_reference(
                    source,
                    validation,
                    max_validation_batches=args.max_validation_batches,
                )
            noise_summary = {
                key: value for key, value in receipt.items() if key != "parameters"
            }
            append_metric(
                run_dir / "metrics.jsonl",
                {
                    "stage": "weight_write_noise_validation",
                    "split": "validation",
                    "official_test_read": False,
                    "std_fraction": float(std_fraction),
                    "programming_seed": int(programming_seed),
                    "validation": validation,
                    "weight_write_noise": noise_summary,
                },
            )
            _verify_source_unchanged(source)
            terminal = {
                "validation": validation,
                "weight_write_noise": noise_summary,
                "source_checkpoint_sha256": source["source_checkpoint_sha256"],
                "official_test_read": False,
            }
            complete_run(
                run_dir,
                terminal_metrics=terminal,
                completion={
                    "criteria_met": True,
                    "validation_outcome": validation["outcome"],
                    "source_unchanged": True,
                    "noise_receipt_written": True,
                    "official_test_read": False,
                },
            )
            rows.append(
                _summary_row(
                    source,
                    run_id=run_id,
                    std_fraction=std_fraction,
                    programming_seed=programming_seed,
                    terminal=terminal,
                    state="complete",
                )
            )
            if validation["outcome"] == "finite":
                print(
                    f"[complete] {run_id} "
                    f"validation_accuracy={validation['accuracy']:.6f}",
                    flush=True,
                )
            else:
                print(
                    f"[complete-scientific-outcome] {run_id} "
                    f"validation_outcome={validation['outcome']} "
                    f"failure={validation['failure']}",
                    flush=True,
                )
        except Exception as error:
            failed = True
            fail_run(run_dir, error=error)
            rows.append(
                _summary_row(
                    source,
                    run_id=run_id,
                    std_fraction=std_fraction,
                    programming_seed=programming_seed,
                    terminal=None,
                    state="failed",
                )
            )
            print(f"[failed] {run_id}: {type(error).__name__}: {error}", file=sys.stderr)
            if std_fraction == 0.0:
                print(
                    "[abort-source] clean-reference replay failed; noisy cases "
                    "were not evaluated.",
                    file=sys.stderr,
                )
                break
        finally:
            _restore_parameters(runtime)
    _verify_source_unchanged(source)
    return rows, failed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        type=Path,
        action="append",
        required=True,
        help="Canonical active Conv run directory; repeat for matched schemes.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--std-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.0025, 0.005, 0.01, 0.02, 0.05],
    )
    parser.add_argument(
        "--programming-seeds",
        type=int,
        nargs="+",
        default=list(range(30)),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--target", default="local")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.source_run = [path.expanduser().resolve() for path in args.source_run]
    if len(args.source_run) != len(set(args.source_run)):
        parser.error("Duplicate --source-run paths are not allowed.")
    args.output_root = args.output_root.expanduser().resolve()
    if args.dataset_root is not None:
        args.dataset_root = args.dataset_root.expanduser().resolve()
    if args.max_validation_batches is not None and args.max_validation_batches <= 0:
        parser.error("--max-validation-batches must be positive.")
    try:
        noise_cases(args.std_fractions, args.programming_seeds)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sources = [_source_inventory(path) for path in args.source_run]
    cases = noise_cases(args.std_fractions, args.programming_seeds)
    print(
        f"[plan] sources={len(sources)} cases_per_source={len(cases)} "
        f"total_cases={len(sources) * len(cases)} model={WRITE_NOISE_MODEL}",
        flush=True,
    )
    for source in sources:
        print(
            f"[source] {source['labels']} run={source['run_dir']} "
            f"checkpoint_sha256={source['source_checkpoint_sha256']}",
            flush=True,
        )
    if args.dry_run:
        planned_device = str(
            torch.device(
                args.device
                if args.device is not None
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        )
        for source in sources:
            for std_fraction, programming_seed in cases:
                _arm_id, run_id = _case_identity(
                    source,
                    std_fraction,
                    programming_seed,
                    max_validation_batches=args.max_validation_batches,
                    evaluation_device=planned_device,
                )
                print(
                    f"[dry-run] {run_id} eta={std_fraction:g} "
                    f"programming_seed={programming_seed}",
                    flush=True,
                )
        return 0

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    study_id = args.output_root.name
    command = [sys.executable, "-m", "experiments.evaluate_conv_weight_write_noise", *sys.argv[1:]]
    summary_rows = []
    any_failed = False
    for source in sources:
        rows, failed = _run_source(
            source,
            args=args,
            device=device,
            study_id=study_id,
            command=command,
        )
        summary_rows.extend(rows)
        any_failed = any_failed or failed
        _write_summary(args.output_root, summary_rows)
    _write_summary(args.output_root, summary_rows)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
