#!/usr/bin/env python3
"""Diagnose gradients and hidden voltages for Conv MNIST saturation sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = REPO_ROOT / "labs"
for path in (REPO_ROOT, LABS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import model  # noqa: F401,E402 - anchor repo-local package before labs imports.
from custom_classes import FlexibleDeepResistiveEnergy  # noqa: E402
from evaluate_mnist_bp_write_noise_sweep import _build_eval_context  # noqa: E402
from labs.mnist_train import (  # noqa: E402
    _build_tracking_minimizer,
    _json_sanitize,
    _reset_name_counters,
    _resolve_callable,
    _resolve_dataset_config,
    _set_seed,
    load_config,
)
from model.function.cost import SquaredError, SquaredErrorPairedOutputs  # noqa: E402
from model.function.network import Network  # noqa: E402
from training.sgd import Backprop  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mnist_bp_conv_saturation_gradient_diagnostics"

RAW_COLUMNS = [
    "source_label",
    "run_dir",
    "checkpoint_kind",
    "checkpoint_path",
    "conv_depth",
    "v_off",
    "target_label",
    "target_saturation",
    "input_gain",
    "lr_initial",
    "lr_final",
    "batch_index",
    "split",
    "loss",
    "accuracy",
    "first_hidden_saturation",
    "all_hidden_saturation",
    "first_hidden_mean",
    "first_hidden_std",
    "first_hidden_abs_mean",
    "first_hidden_abs_p50",
    "first_hidden_abs_p90",
    "first_hidden_abs_p99",
    "first_hidden_abs_max",
    "first_hidden_rms",
    "first_hidden_move_rms",
    "param_index",
    "param_name",
    "param_class",
    "param_l2",
    "param_abs_mean",
    "param_min",
    "param_max",
    "param_lower_bound_fraction",
    "param_upper_bound_fraction",
    "grad_l2",
    "grad_abs_mean",
    "grad_abs_max",
    "grad_zero_fraction",
    "relative_grad_l2",
    "relative_step_l2_initial_lr",
    "relative_step_l2_final_lr",
    "log_weight_grad_l2",
    "log_weight_step_l2_initial_lr",
    "log_weight_step_l2_final_lr",
]

PARAM_SUMMARY_COLUMNS = [
    "source_label",
    "run_dir",
    "checkpoint_kind",
    "conv_depth",
    "v_off",
    "target_label",
    "target_saturation",
    "input_gain",
    "lr_initial",
    "lr_final",
    "param_index",
    "param_name",
    "param_class",
    "num_batches",
    "param_l2_mean",
    "grad_l2_mean",
    "grad_l2_p50",
    "grad_abs_max_mean",
    "grad_zero_fraction_mean",
    "relative_grad_l2_mean",
    "relative_step_l2_initial_lr_mean",
    "relative_step_l2_final_lr_mean",
    "log_weight_grad_l2_mean",
    "log_weight_step_l2_initial_lr_mean",
    "log_weight_step_l2_final_lr_mean",
    "param_lower_bound_fraction_mean",
    "param_upper_bound_fraction_mean",
]

RUN_SUMMARY_COLUMNS = [
    "source_label",
    "run_dir",
    "checkpoint_kind",
    "checkpoint_path",
    "complete",
    "conv_depth",
    "padding",
    "num_iterations_inference",
    "num_iterations_training",
    "v_off",
    "target_label",
    "target_saturation",
    "input_gain",
    "lr_initial",
    "lr_final",
    "epochs_recorded",
    "metrics_best_epoch",
    "metrics_best_test_accuracy",
    "metrics_final_test_accuracy",
    "history_best_test_accuracy",
    "history_final_test_accuracy",
    "test_accuracy_last_delta",
    "test_accuracy_last3_slope",
    "test_loss_last3_slope",
    "train_accuracy_last3_slope",
    "diagnostic_batches",
    "diagnostic_loss_mean",
    "diagnostic_accuracy_mean",
    "first_hidden_saturation_mean",
    "all_hidden_saturation_mean",
    "first_hidden_abs_mean_mean",
    "first_hidden_abs_p90_mean",
    "first_hidden_abs_p99_mean",
    "first_hidden_abs_max_mean",
    "first_hidden_move_rms_mean",
    "global_param_l2_mean",
    "global_grad_l2_mean",
    "global_relative_step_initial_lr_mean",
    "global_relative_step_final_lr_mean",
    "first_weight_name",
    "first_weight_grad_l2_mean",
    "first_weight_relative_step_initial_lr_mean",
    "first_weight_relative_step_final_lr_mean",
    "max_param_relative_step_initial_lr_mean",
    "max_param_relative_step_final_lr_mean",
]


@dataclass(frozen=True)
class DiagnosticRun:
    source_label: str
    run_dir: Path
    checkpoint_kind: str
    checkpoint_path: Path
    non_linearity: str
    run_name: str
    seed: int
    voltage_amp: float
    current_amp: float
    clean_accuracy: float


def _write_rows(path: Path, columns: list[str], rows: list[dict], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and append
    mode = "a" if append else "w"
    with path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: _json_sanitize(row.get(column, "")) for column in columns})


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_sanitize(payload), indent=2, sort_keys=True))


def _finite(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _mean(values: list[float]) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if arr.size else math.nan


def _percentile(values: list[float], q: float) -> float:
    arr = _finite(values)
    return float(np.percentile(arr, q)) if arr.size else math.nan


def _path_part(run_dir: Path, prefix: str) -> str:
    for part in reversed(run_dir.parts):
        if part.startswith(prefix):
            return part[len(prefix) :]
    return ""


def _label_to_float(value: str) -> float:
    if not value:
        return math.nan
    try:
        return float(value.replace("p", ".").replace("m", "-"))
    except ValueError:
        return math.nan


def _target_from_label(label: str) -> float:
    if label.startswith("sat"):
        try:
            return int(label[3:]) / 100.0
        except ValueError:
            return math.nan
    return math.nan


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _history_array(run_dir: Path, name: str) -> np.ndarray:
    path = run_dir / name
    if not path.exists():
        return np.asarray([], dtype=np.float64)
    return np.load(path).astype(np.float64)


def _last_slope(values: np.ndarray, window: int = 3) -> float:
    if values.size < 2:
        return math.nan
    k = min(int(window), int(values.size))
    if k < 2:
        return math.nan
    return float((values[-1] - values[-k]) / (k - 1))


def _checkpoint_for(run_dir: Path, checkpoint_kind: str) -> tuple[str, Path] | None:
    final_path = run_dir / "final_model.pt"
    best_path = run_dir / "best_model.pt"
    if checkpoint_kind == "init":
        source_config_path = run_dir / "source_config.json"
        return ("init", source_config_path) if source_config_path.exists() else None
    if checkpoint_kind == "final":
        return ("final", final_path) if final_path.exists() else None
    if checkpoint_kind == "best":
        return ("best", best_path) if best_path.exists() else None
    if checkpoint_kind == "auto":
        if final_path.exists():
            return "final", final_path
        if best_path.exists():
            return "best", best_path
        return None
    raise ValueError(f"Expected checkpoint_kind final/best/auto, got {checkpoint_kind!r}.")


def _discover_runs(input_roots: list[Path], checkpoint_kind: str) -> list[DiagnosticRun]:
    runs: list[DiagnosticRun] = []
    seen: set[Path] = set()
    for input_root in input_roots:
        input_root = input_root.expanduser().resolve()
        source_paths = sorted(input_root.glob("**/source_config.json"))
        for source_config_path in source_paths:
            run_dir = source_config_path.parent.resolve()
            if run_dir in seen:
                continue
            seen.add(run_dir)
            checkpoint = _checkpoint_for(run_dir, checkpoint_kind)
            if checkpoint is None:
                continue
            selected_kind, checkpoint_path = checkpoint
            source_config = _load_json(source_config_path)
            model_cfg = {
                **source_config.get("model_base", {}),
                **source_config.get("model_overrides", {}).get(
                    source_config.get("lab", {}).get("model_key", "mnist_bp_conv_amp"),
                    {},
                ),
            }
            metrics = _load_json(run_dir / "metrics.json")
            seed = int(source_config.get("seed", metrics.get("seed", 0)))
            clean_key = "final_test_accuracy" if selected_kind == "final" else "best_test_accuracy"
            clean_accuracy = math.nan if selected_kind == "init" else float(metrics.get(clean_key, math.nan))
            runs.append(
                DiagnosticRun(
                    source_label=input_root.name,
                    run_dir=run_dir,
                    checkpoint_kind=selected_kind,
                    checkpoint_path=checkpoint_path,
                    non_linearity=run_dir.parent.parent.name if run_dir.parent.parent else "",
                    run_name=run_dir.parent.name,
                    seed=seed,
                    voltage_amp=float(model_cfg.get("voltage_amp", math.nan)),
                    current_amp=float(model_cfg.get("current_amp", math.nan)),
                    clean_accuracy=clean_accuracy,
                )
            )
    return sorted(runs, key=lambda run: str(run.run_dir))


def _build_init_context(
    run: DiagnosticRun,
    *,
    device: torch.device,
    eval_batch_size: int | None,
    no_download: bool,
    inference_iterations_override: int | None,
    dataset_root_override: str | None = None,
) -> dict:
    source_config_path = run.run_dir / "source_config.json"
    if not source_config_path.exists():
        raise FileNotFoundError(f"Expected source_config.json at {source_config_path}.")

    config = load_config(source_config_path)
    _reset_name_counters()
    _set_seed(run.seed)

    model_key = config.get("lab", {}).get("model_key", "mnist_bp_amp")
    model_cfg = {
        **config["model_base"],
        **config["model_overrides"][model_key],
    }
    trained_input_gain = float(model_cfg["input_gain"])
    trained_hard_sigmoid_param = dict(model_cfg.get("hard_sigmoid_param", {}))
    dataset_key, dataset_cfg = _resolve_dataset_config(
        config,
        config.get("lab", {}).get("dataset_key", "mnist"),
    )
    del dataset_key

    layer_shapes = [
        tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
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
        hard_sigmoid_param=model_cfg.get("hard_sigmoid_param", {}),
        voltage_amp=model_cfg["voltage_amp"],
        current_amp=model_cfg["current_amp"],
        weight_min=model_cfg["weight_min"],
        weight_max=model_cfg["weight_max"],
        weight_init_mode=model_cfg.get("weight_init_mode", "kaiming_uniform"),
        input_mode=config.get("input_mode", "train"),
    )
    energy_fn.set_device(device)

    network = Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]
    output_dim = output_layer.shape[0]
    num_classes = 10
    if output_dim == num_classes:
        cost_fn = SquaredError(output_layer)
    elif output_dim == 2 * num_classes:
        cost_fn = SquaredErrorPairedOutputs(output_layer, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported output dimension {output_dim}.")

    source_inference_iterations = int(model_cfg["num_iterations_inference"])
    inference_iterations = (
        int(inference_iterations_override)
        if inference_iterations_override is not None
        else source_inference_iterations
    )
    minimizer = _build_tracking_minimizer(
        energy_fn,
        free_layers,
        model_cfg,
        config["energy_minimizer"]["mode"],
        num_iterations=inference_iterations,
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )

    dataset_factory = _resolve_callable(dataset_cfg["factory"])
    dataset_params = dict(dataset_cfg["params"])
    dataset_params["device"] = device
    if eval_batch_size is not None:
        dataset_params["batch_size"] = int(eval_batch_size)
    if dataset_root_override is not None:
        dataset_params["root"] = os.path.expanduser(str(dataset_root_override))
    if "root" in dataset_params:
        dataset_params["root"] = os.path.expanduser(str(dataset_params["root"]))
    if no_download:
        dataset_params["download"] = False
    loader_result = dataset_factory(**dataset_params).build()
    if not isinstance(loader_result, tuple):
        raise ValueError("Expected MNIST dataset factory to return train and test loaders.")
    train_loader, test_loader = loader_result

    params = energy_fn.params()
    base_states = [param.state.detach().clone() for param in params]
    return {
        "config": config,
        "model_cfg": model_cfg,
        "trained_input_gain": trained_input_gain,
        "eval_input_gain": float(model_cfg["input_gain"]),
        "trained_hard_sigmoid_param": trained_hard_sigmoid_param,
        "eval_hard_sigmoid_param": dict(model_cfg.get("hard_sigmoid_param", {})),
        "energy_fn": energy_fn,
        "network": network,
        "free_layers": free_layers,
        "output_layer": output_layer,
        "cost_fn": cost_fn,
        "minimizer": minimizer,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "params": params,
        "base_states": base_states,
        "noisable_indices": [
            index
            for index, param in enumerate(params)
            if "Weight" in param.__class__.__name__
        ],
        "inference_iterations": inference_iterations,
        "source_inference_iterations": source_inference_iterations,
        "dataset_params": dataset_params,
        "num_classes": num_classes,
    }


def _learning_rates(context: dict, metrics: dict) -> tuple[list[float], list[float]]:
    params = context["params"]
    config_lrs = context["config"].get("lr", [])
    initial = [float(value) for value in config_lrs]
    if not initial:
        initial = [math.nan]
    if len(initial) < len(params):
        initial.extend([initial[-1]] * (len(params) - len(initial)))
    initial = initial[: len(params)]

    final = metrics.get("final_learning_rate", initial)
    if isinstance(final, (int, float)):
        final_values = [float(final)]
    else:
        final_values = [float(value) for value in final]
    if not final_values:
        final_values = initial
    if len(final_values) < len(params):
        final_values.extend([final_values[-1]] * (len(params) - len(final_values)))
    final_values = final_values[: len(params)]
    return initial, final_values


def _param_name(param: object, index: int) -> str:
    return str(getattr(param, "name", f"param_{index}")).strip()


def _param_bounds(param: object) -> tuple[float | None, float | None]:
    lower = getattr(param, "min_cond", None)
    upper = getattr(param, "max_cond", None)
    if getattr(param, "_non_negative", False) and lower is None:
        lower = 0.0
    return lower, upper


def _param_stats(param: object, grad: torch.Tensor, lr_initial: float, lr_final: float) -> dict:
    state = param.state.detach()
    grad = grad.detach()
    param_l2 = float(torch.linalg.vector_norm(state).item())
    grad_l2 = float(torch.linalg.vector_norm(grad).item())
    lower, upper = _param_bounds(param)
    tol = 1.0e-8
    lower_fraction = math.nan
    upper_fraction = math.nan
    if lower is not None and math.isfinite(float(lower)):
        lower_fraction = float((state <= float(lower) + tol).float().mean().item())
    if upper is not None and math.isfinite(float(upper)):
        upper_fraction = float((state >= float(upper) - tol).float().mean().item())
    log_weight_grad = state * grad
    log_weight_grad_l2 = float(torch.linalg.vector_norm(log_weight_grad).item())
    return {
        "param_l2": param_l2,
        "param_abs_mean": float(torch.mean(torch.abs(state)).item()),
        "param_min": float(torch.min(state).item()),
        "param_max": float(torch.max(state).item()),
        "param_lower_bound_fraction": lower_fraction,
        "param_upper_bound_fraction": upper_fraction,
        "grad_l2": grad_l2,
        "grad_abs_mean": float(torch.mean(torch.abs(grad)).item()),
        "grad_abs_max": float(torch.max(torch.abs(grad)).item()),
        "grad_zero_fraction": float((grad.abs() <= 1.0e-12).float().mean().item()),
        "relative_grad_l2": grad_l2 / param_l2 if param_l2 > 0 else math.nan,
        "relative_step_l2_initial_lr": (
            lr_initial * grad_l2 / param_l2 if param_l2 > 0 and math.isfinite(lr_initial) else math.nan
        ),
        "relative_step_l2_final_lr": (
            lr_final * grad_l2 / param_l2 if param_l2 > 0 and math.isfinite(lr_final) else math.nan
        ),
        "log_weight_grad_l2": log_weight_grad_l2,
        "log_weight_step_l2_initial_lr": (
            lr_initial * log_weight_grad_l2 if math.isfinite(lr_initial) else math.nan
        ),
        "log_weight_step_l2_final_lr": (
            lr_final * log_weight_grad_l2 if math.isfinite(lr_final) else math.nan
        ),
    }


def _tensor_quantiles(tensor: torch.Tensor) -> dict:
    flat_abs = tensor.detach().abs().flatten()
    if flat_abs.numel() == 0:
        return {
            "abs_mean": math.nan,
            "abs_p50": math.nan,
            "abs_p90": math.nan,
            "abs_p99": math.nan,
            "abs_max": math.nan,
            "rms": math.nan,
        }
    quantiles = torch.quantile(flat_abs, torch.tensor([0.5, 0.9, 0.99], device=flat_abs.device))
    return {
        "abs_mean": float(flat_abs.mean().item()),
        "abs_p50": float(quantiles[0].item()),
        "abs_p90": float(quantiles[1].item()),
        "abs_p99": float(quantiles[2].item()),
        "abs_max": float(flat_abs.max().item()),
        "rms": float(torch.sqrt(torch.mean(tensor.detach() ** 2)).item()),
    }


def _layer_saturation(layer, v_off: float) -> tuple[float, int]:
    state = layer.state.detach()
    if not math.isfinite(v_off):
        return math.nan, int(state.numel())
    return float((state.abs() > float(v_off)).float().mean().item()), int(state.numel())


def _hidden_stats(context: dict, v_off: float) -> dict:
    layers = context["energy_fn"].layers()
    hidden_layers = layers[1:-1]
    first_hidden = layers[1]
    first_sat, _ = _layer_saturation(first_hidden, v_off)
    saturated = 0.0
    total = 0
    for layer in hidden_layers:
        sat, count = _layer_saturation(layer, v_off)
        if math.isfinite(sat):
            saturated += sat * count
            total += count
    first_state = first_hidden.state.detach()
    q = _tensor_quantiles(first_state)
    return {
        "first_hidden_saturation": first_sat,
        "all_hidden_saturation": saturated / total if total else math.nan,
        "first_hidden_mean": float(first_state.mean().item()),
        "first_hidden_std": float(first_state.std(unbiased=False).item()),
        "first_hidden_abs_mean": q["abs_mean"],
        "first_hidden_abs_p50": q["abs_p50"],
        "first_hidden_abs_p90": q["abs_p90"],
        "first_hidden_abs_p99": q["abs_p99"],
        "first_hidden_abs_max": q["abs_max"],
        "first_hidden_rms": q["rms"],
        "first_hidden_move_rms": q["rms"],
    }


def _run_metadata(run: DiagnosticRun) -> dict:
    source_config = _load_json(run.run_dir / "source_config.json")
    model_key = source_config.get("lab", {}).get("model_key", "mnist_bp_conv_amp")
    model_cfg = {
        **source_config.get("model_base", {}),
        **source_config.get("model_overrides", {}).get(model_key, {}),
    }
    hard_sigmoid = model_cfg.get("hard_sigmoid_param", {})
    metrics = _load_json(run.run_dir / "metrics.json")
    acc_test = _history_array(run.run_dir, "accuracy_test.npy")
    loss_test = _history_array(run.run_dir, "loss_test.npy")
    acc_train = _history_array(run.run_dir, "accuracy_train.npy")
    conv_pipeline = model_cfg.get("conv_pipeline") or []
    padding = ",".join(str(conf.get("padding", "")) for conf in conv_pipeline)
    return {
        "source_config": source_config,
        "model_cfg": model_cfg,
        "metrics": metrics,
        "conv_depth": len(conv_pipeline),
        "padding": padding,
        "num_iterations_inference": int(model_cfg.get("num_iterations_inference", 0)),
        "num_iterations_training": int(
            model_cfg.get("num_iterations_training", model_cfg.get("num_iterations_inference", 0))
        ),
        "v_off": abs(float(hard_sigmoid.get("v_max", _label_to_float(_path_part(run.run_dir, "v_off_"))))),
        "target_label": _path_part(run.run_dir, "target_"),
        "target_saturation": _target_from_label(_path_part(run.run_dir, "target_")),
        "input_gain": float(model_cfg.get("input_gain", _label_to_float(_path_part(run.run_dir, "input_gain_")))),
        "epochs_recorded": int(acc_test.size),
        "metrics_best_epoch": metrics.get("best_epoch"),
        "metrics_best_test_accuracy": metrics.get("best_test_accuracy"),
        "metrics_final_test_accuracy": metrics.get("final_test_accuracy"),
        "history_best_test_accuracy": float(np.max(acc_test)) if acc_test.size else math.nan,
        "history_final_test_accuracy": float(acc_test[-1]) if acc_test.size else math.nan,
        "test_accuracy_last_delta": float(acc_test[-1] - acc_test[-2]) if acc_test.size >= 2 else math.nan,
        "test_accuracy_last3_slope": _last_slope(acc_test, window=3),
        "test_loss_last3_slope": _last_slope(loss_test, window=3),
        "train_accuracy_last3_slope": _last_slope(acc_train, window=3),
        "complete": bool((run.run_dir / "final_model.pt").exists() and (run.run_dir / "metrics.json").exists()),
    }


def _diagnose_one(
    run: DiagnosticRun,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict]:
    metadata = _run_metadata(run)
    if run.checkpoint_kind == "init":
        context = _build_init_context(
            run,
            device=device,
            eval_batch_size=args.batch_size,
            no_download=args.no_download,
            inference_iterations_override=args.inference_iterations,
            dataset_root_override=args.dataset_root,
        )
    else:
        context = _build_eval_context(
            run,
            device=device,
            eval_batch_size=args.batch_size,
            no_download=args.no_download,
            inference_iterations_override=args.inference_iterations,
            dataset_root_override=args.dataset_root,
        )
    model_cfg = context["model_cfg"]
    training_iterations = (
        int(args.training_iterations)
        if args.training_iterations is not None
        else int(model_cfg.get("num_iterations_training", model_cfg["num_iterations_inference"]))
    )
    minimizer_training = _build_tracking_minimizer(
        context["energy_fn"],
        context["free_layers"],
        model_cfg,
        context["config"]["energy_minimizer"]["mode"],
        num_iterations=training_iterations,
        voltage_amp=context["energy_fn"]._voltage_amp,
        current_amp=context["energy_fn"]._current_amp,
    )
    minimizer_inference = context["minimizer"]
    estimator = Backprop(context["params"], context["free_layers"], context["cost_fn"], minimizer_training)
    loader = context["train_loader"] if args.split == "train" else context["test_loader"]
    lr_initial, lr_final = _learning_rates(context, metadata["metrics"])
    metadata["lr_initial"] = lr_initial[0] if lr_initial else math.nan
    metadata["lr_final"] = lr_final[0] if lr_final else math.nan
    metadata["num_iterations_training"] = training_iterations

    rows: list[dict] = []
    batch_losses: list[float] = []
    batch_accs: list[float] = []
    first_hidden_saturation: list[float] = []
    all_hidden_saturation: list[float] = []
    first_hidden_abs_mean: list[float] = []
    first_hidden_abs_p90: list[float] = []
    first_hidden_abs_p99: list[float] = []
    first_hidden_abs_max: list[float] = []
    first_hidden_move_rms: list[float] = []
    global_param_l2: list[float] = []
    global_grad_l2: list[float] = []
    global_step_initial: list[float] = []
    global_step_final: list[float] = []
    first_weight_grad_l2: list[float] = []
    first_weight_step_initial: list[float] = []
    first_weight_step_final: list[float] = []
    max_step_initial: list[float] = []
    max_step_final: list[float] = []
    first_weight_name = ""

    for batch_index, (images, labels) in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        for param, base in zip(context["params"], context["base_states"]):
            param.state = base.detach().clone()

        images = images.to(device)
        labels = labels.to(device)
        context["network"].set_input(images, reset=True)
        minimizer_inference.compute_equilibrium()
        context["cost_fn"].set_target(labels)
        with torch.no_grad():
            loss = float(context["cost_fn"].eval().mean().item())
            errors = context["cost_fn"].error_fn()
            accuracy = float((~errors).float().mean().item())
            hidden = _hidden_stats(context, metadata["v_off"])

        grads = estimator.compute_gradient()[: len(context["params"])]
        batch_losses.append(loss)
        batch_accs.append(accuracy)
        first_hidden_saturation.append(hidden["first_hidden_saturation"])
        all_hidden_saturation.append(hidden["all_hidden_saturation"])
        first_hidden_abs_mean.append(hidden["first_hidden_abs_mean"])
        first_hidden_abs_p90.append(hidden["first_hidden_abs_p90"])
        first_hidden_abs_p99.append(hidden["first_hidden_abs_p99"])
        first_hidden_abs_max.append(hidden["first_hidden_abs_max"])
        first_hidden_move_rms.append(hidden["first_hidden_move_rms"])

        param_l2_sq = 0.0
        grad_l2_sq = 0.0
        per_param_step_initial: list[float] = []
        per_param_step_final: list[float] = []
        for index, (param, grad) in enumerate(zip(context["params"], grads)):
            lr_i = lr_initial[index]
            lr_f = lr_final[index]
            stats = _param_stats(param, grad, lr_i, lr_f)
            param_l2_sq += stats["param_l2"] ** 2
            grad_l2_sq += stats["grad_l2"] ** 2
            per_param_step_initial.append(stats["relative_step_l2_initial_lr"])
            per_param_step_final.append(stats["relative_step_l2_final_lr"])
            if not first_weight_name and "Weight" in param.__class__.__name__:
                first_weight_name = _param_name(param, index)
            if _param_name(param, index) == first_weight_name:
                first_weight_grad_l2.append(stats["grad_l2"])
                first_weight_step_initial.append(stats["relative_step_l2_initial_lr"])
                first_weight_step_final.append(stats["relative_step_l2_final_lr"])
            rows.append(
                {
                    "source_label": run.source_label,
                    "run_dir": str(run.run_dir),
                    "checkpoint_kind": run.checkpoint_kind,
                    "checkpoint_path": str(run.checkpoint_path),
                    "conv_depth": metadata["conv_depth"],
                    "v_off": metadata["v_off"],
                    "target_label": metadata["target_label"],
                    "target_saturation": metadata["target_saturation"],
                    "input_gain": metadata["input_gain"],
                    "lr_initial": lr_i,
                    "lr_final": lr_f,
                    "batch_index": batch_index,
                    "split": args.split,
                    "loss": loss,
                    "accuracy": accuracy,
                    **hidden,
                    "param_index": index,
                    "param_name": _param_name(param, index),
                    "param_class": param.__class__.__name__,
                    **stats,
                }
            )
        gp = math.sqrt(param_l2_sq)
        gg = math.sqrt(grad_l2_sq)
        global_param_l2.append(gp)
        global_grad_l2.append(gg)
        global_step_initial.append(metadata["lr_initial"] * gg / gp if gp > 0 else math.nan)
        global_step_final.append(metadata["lr_final"] * gg / gp if gp > 0 else math.nan)
        max_step_initial.append(float(np.nanmax(np.asarray(per_param_step_initial, dtype=np.float64))))
        max_step_final.append(float(np.nanmax(np.asarray(per_param_step_final, dtype=np.float64))))

        del grads
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run_summary = {
        "source_label": run.source_label,
        "run_dir": str(run.run_dir),
        "checkpoint_kind": run.checkpoint_kind,
        "checkpoint_path": str(run.checkpoint_path),
        **{key: metadata.get(key) for key in RUN_SUMMARY_COLUMNS if key in metadata},
        "diagnostic_batches": len(batch_losses),
        "diagnostic_loss_mean": _mean(batch_losses),
        "diagnostic_accuracy_mean": _mean(batch_accs),
        "first_hidden_saturation_mean": _mean(first_hidden_saturation),
        "all_hidden_saturation_mean": _mean(all_hidden_saturation),
        "first_hidden_abs_mean_mean": _mean(first_hidden_abs_mean),
        "first_hidden_abs_p90_mean": _mean(first_hidden_abs_p90),
        "first_hidden_abs_p99_mean": _mean(first_hidden_abs_p99),
        "first_hidden_abs_max_mean": _mean(first_hidden_abs_max),
        "first_hidden_move_rms_mean": _mean(first_hidden_move_rms),
        "global_param_l2_mean": _mean(global_param_l2),
        "global_grad_l2_mean": _mean(global_grad_l2),
        "global_relative_step_initial_lr_mean": _mean(global_step_initial),
        "global_relative_step_final_lr_mean": _mean(global_step_final),
        "first_weight_name": first_weight_name,
        "first_weight_grad_l2_mean": _mean(first_weight_grad_l2),
        "first_weight_relative_step_initial_lr_mean": _mean(first_weight_step_initial),
        "first_weight_relative_step_final_lr_mean": _mean(first_weight_step_final),
        "max_param_relative_step_initial_lr_mean": _mean(max_step_initial),
        "max_param_relative_step_final_lr_mean": _mean(max_step_final),
    }
    return rows, run_summary


def _summarize_params(raw_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in raw_rows:
        key = (row["run_dir"], int(row["param_index"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict] = []
    for (_run_dir, _param_index), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        first = rows[0]
        summaries.append(
            {
                "source_label": first["source_label"],
                "run_dir": first["run_dir"],
                "checkpoint_kind": first["checkpoint_kind"],
                "conv_depth": first["conv_depth"],
                "v_off": first["v_off"],
                "target_label": first["target_label"],
                "target_saturation": first["target_saturation"],
                "input_gain": first["input_gain"],
                "lr_initial": first["lr_initial"],
                "lr_final": first["lr_final"],
                "param_index": first["param_index"],
                "param_name": first["param_name"],
                "param_class": first["param_class"],
                "num_batches": len(rows),
                "param_l2_mean": _mean([float(row["param_l2"]) for row in rows]),
                "grad_l2_mean": _mean([float(row["grad_l2"]) for row in rows]),
                "grad_l2_p50": _percentile([float(row["grad_l2"]) for row in rows], 50),
                "grad_abs_max_mean": _mean([float(row["grad_abs_max"]) for row in rows]),
                "grad_zero_fraction_mean": _mean([float(row["grad_zero_fraction"]) for row in rows]),
                "relative_grad_l2_mean": _mean([float(row["relative_grad_l2"]) for row in rows]),
                "relative_step_l2_initial_lr_mean": _mean(
                    [float(row["relative_step_l2_initial_lr"]) for row in rows]
                ),
                "relative_step_l2_final_lr_mean": _mean(
                    [float(row["relative_step_l2_final_lr"]) for row in rows]
                ),
                "log_weight_grad_l2_mean": _mean([float(row["log_weight_grad_l2"]) for row in rows]),
                "log_weight_step_l2_initial_lr_mean": _mean(
                    [float(row["log_weight_step_l2_initial_lr"]) for row in rows]
                ),
                "log_weight_step_l2_final_lr_mean": _mean(
                    [float(row["log_weight_step_l2_final_lr"]) for row in rows]
                ),
                "param_lower_bound_fraction_mean": _mean(
                    [float(row["param_lower_bound_fraction"]) for row in rows]
                ),
                "param_upper_bound_fraction_mean": _mean(
                    [float(row["param_upper_bound_fraction"]) for row in rows]
                ),
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-kind", choices=("auto", "best", "final", "init"), default="auto")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--inference-iterations", type=int, default=None)
    parser.add_argument("--training-iterations", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    input_roots = [Path(value).expanduser().resolve() for value in args.input_root]
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {args.device!r}, but CUDA is unavailable.")

    runs = _discover_runs(input_roots, args.checkpoint_kind)
    if args.limit is not None:
        runs = runs[: int(args.limit)]
    _write_json(
        output_root / "diagnostic_config.json",
        {
            "input_roots": [str(path) for path in input_roots],
            "output_root": str(output_root),
            "dataset_root": args.dataset_root,
            "device": str(device),
            "checkpoint_kind": args.checkpoint_kind,
            "split": args.split,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "inference_iterations": args.inference_iterations,
            "training_iterations": args.training_iterations,
            "num_runs": len(runs),
            "runs": [str(run.run_dir) for run in runs],
        },
    )
    print(f"[diagnose] runs={len(runs)} output={output_root} device={device}")
    if args.dry_run:
        for run in runs:
            print(f"[dry-run] {run.checkpoint_kind} {run.checkpoint_path}")
        return

    raw_rows: list[dict] = []
    run_summaries: list[dict] = []
    for index, run in enumerate(runs, start=1):
        print(f"[run {index}/{len(runs)}] {run.checkpoint_kind} {run.run_dir}", flush=True)
        rows, summary = _diagnose_one(run, args=args, device=device)
        raw_rows.extend(rows)
        run_summaries.append(summary)
        _write_rows(output_root / "raw_param_gradients.csv", RAW_COLUMNS, rows, append=True)
        _write_rows(output_root / "run_summary.csv", RUN_SUMMARY_COLUMNS, run_summaries, append=False)
        _write_rows(
            output_root / "param_summary.csv",
            PARAM_SUMMARY_COLUMNS,
            _summarize_params(raw_rows),
            append=False,
        )
    print(f"[done] wrote {len(raw_rows)} raw rows under {output_root}")


if __name__ == "__main__":
    main()
