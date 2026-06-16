#!/usr/bin/env python3
"""Evaluate MNIST DRN Hessian metrics with an LPW perfect-diode approximation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = REPO_ROOT / "labs"
TOOLS_DIR = LABS_DIR / "tools"
for path in (REPO_ROOT, LABS_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from custom_classes import FlexibleDeepResistiveEnergy  # noqa: E402
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
from model.variable.parameter import DenseWeight  # noqa: E402

import plot_conditioning_vs_runtime as pcr  # noqa: E402


DEFAULT_INPUT_ROOT = (
    REPO_ROOT / "results" / "mnist_bp_amplification_sweep_drn_xs_10epoch_legacy_preproc"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results" / "mnist_bp_hessian_lpw_amplification_sweep_legacy_drnxs"
)
RUN_ORDER = [
    "mnist_bp_amp_v1_c1",
    "mnist_bp_amp_v2_c1",
    "mnist_bp_amp_v4_c1",
    "mnist_bp_amp_v1_c2",
    "mnist_bp_amp_v1_c4",
]
SAMPLE_COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "sample_index",
    "label",
    "prediction",
    "correct",
    "lpw_active_hidden_count",
    "lpw_active_hidden_fraction",
    "rho_lpw",
    "inv_neg_log_rho_lpw",
    "inv_one_minus_rho_lpw",
    "sigma_lpw",
    "L_full_lpw",
    "kappa_full_lpw",
    "L_odd_lpw",
    "L_even_lpw",
]
SUMMARY_COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "checkpoint_path",
    "source_config_path",
    "resolved_config_path",
    "sample_count",
    "unique_lpw_masks",
    "test_accuracy_recomputed",
    "rho_linear",
    "sigma_linear",
    "L_full_linear",
    "kappa_full_linear",
    "L_odd_linear",
    "L_even_linear",
    "rho_lpw_mean_diag",
    "sigma_lpw_mean_diag",
    "L_full_lpw_mean_diag",
    "kappa_full_lpw_mean_diag",
    "L_odd_lpw_mean_diag",
    "L_even_lpw_mean_diag",
    "rho_lpw_mean",
    "rho_lpw_std",
    "rho_lpw_p10",
    "rho_lpw_p50",
    "rho_lpw_p90",
    "kappa_full_lpw_mean",
    "kappa_full_lpw_std",
    "kappa_full_lpw_p10",
    "kappa_full_lpw_p50",
    "kappa_full_lpw_p90",
    "sigma_lpw_mean",
    "sigma_lpw_std",
    "sigma_lpw_p10",
    "sigma_lpw_p50",
    "sigma_lpw_p90",
    "lpw_active_hidden_fraction_mean",
    "lpw_active_hidden_fraction_std",
    "lpw_active_hidden_fraction_p10",
    "lpw_active_hidden_fraction_p50",
    "lpw_active_hidden_fraction_p90",
    "sample_metrics_path",
    "hessian_linear_path",
    "hessian_lpw_mean_path",
    "lpw_masks_path",
    "summary_path",
]
BY_AMP_COLUMNS = [
    "run_name",
    "voltage_amp",
    "current_amp",
    "num_seeds",
    "sample_count",
    "test_accuracy_mean",
    "test_accuracy_std",
    "rho_lpw_mean",
    "rho_lpw_std",
    "rho_lpw_p10",
    "rho_lpw_p50",
    "rho_lpw_p90",
    "kappa_full_lpw_mean",
    "kappa_full_lpw_std",
    "kappa_full_lpw_p10",
    "kappa_full_lpw_p50",
    "kappa_full_lpw_p90",
    "sigma_lpw_mean",
    "sigma_lpw_std",
    "sigma_lpw_p10",
    "sigma_lpw_p50",
    "sigma_lpw_p90",
    "lpw_active_hidden_fraction_mean",
    "lpw_active_hidden_fraction_std",
    "lpw_active_hidden_fraction_p10",
    "lpw_active_hidden_fraction_p50",
    "lpw_active_hidden_fraction_p90",
]


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    seed: int
    voltage_amp: float
    current_amp: float
    run_dir: Path
    checkpoint_path: Path
    source_config_path: Path
    resolved_config_path: Path


def finite_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_sort_key(row: dict[str, str]) -> tuple[int, int]:
    run_name = row.get("run_name", "")
    try:
        run_idx = RUN_ORDER.index(run_name)
    except ValueError:
        run_idx = len(RUN_ORDER)
    return run_idx, int(row.get("seed", 0))


def read_run_specs(
    input_root: Path,
    *,
    checkpoint: str,
    run_names: set[str] | None,
    seeds: set[int] | None,
) -> list[RunSpec]:
    summary_path = input_root / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary.csv at {summary_path}.")

    rows = sorted(read_csv(summary_path), key=_summary_sort_key)
    specs: list[RunSpec] = []
    for row in rows:
        run_name = row["run_name"]
        seed = int(row["seed"])
        if run_names is not None and run_name not in run_names:
            continue
        if seeds is not None and seed not in seeds:
            continue

        weights_path = Path(row["weights_best_path" if checkpoint == "best" else "weights_final_path"])
        checkpoint_name = "best_model.pt" if checkpoint == "best" else "final_model.pt"
        checkpoint_path = weights_path.with_name(checkpoint_name)
        run_dir = checkpoint_path.parent
        source_config_path = run_dir / "source_config.json"
        resolved_config_path = run_dir / "config.json"
        specs.append(
            RunSpec(
                run_name=run_name,
                seed=seed,
                voltage_amp=float(row["voltage_amp"]),
                current_amp=float(row["current_amp"]),
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                source_config_path=source_config_path,
                resolved_config_path=resolved_config_path,
            )
        )

    if not specs:
        raise ValueError("No run specs matched the requested filters.")
    return specs


def run_output_dir(output_root: Path, spec: RunSpec) -> Path:
    return output_root / spec.run_name / f"seed_{spec.seed}"


def run_complete(output_root: Path, spec: RunSpec) -> bool:
    run_dir = run_output_dir(output_root, spec)
    required = [
        "hessian_linear.npz",
        "hessian_lpw_mean.npz",
        "lpw_conductance_masks.npz",
        "sample_metrics.csv",
        "summary.json",
        "config.json",
    ]
    return all((run_dir / name).exists() for name in required)


def load_source_and_resolved_config(spec: RunSpec) -> tuple[dict, dict]:
    if not spec.source_config_path.exists():
        raise FileNotFoundError(f"Expected source_config.json at {spec.source_config_path}.")
    if not spec.resolved_config_path.exists():
        raise FileNotFoundError(f"Expected config.json at {spec.resolved_config_path}.")
    return load_config(spec.source_config_path), json.loads(spec.resolved_config_path.read_text())


def build_eval_context(
    spec: RunSpec,
    *,
    device: torch.device,
    batch_size: int | None,
    dataset_root: str | None,
    no_download: bool,
) -> dict:
    if not spec.checkpoint_path.exists():
        raise FileNotFoundError(f"Expected checkpoint at {spec.checkpoint_path}.")

    source_config, resolved_config = load_source_and_resolved_config(spec)
    _reset_name_counters()
    _set_seed(spec.seed)

    model_key = source_config.get("lab", {}).get("model_key", "mnist_bp_amp")
    model_overrides = source_config["model_overrides"]
    if model_key not in model_overrides and len(model_overrides) == 1:
        model_key = next(iter(model_overrides))
    model_cfg = {**source_config["model_base"], **model_overrides[model_key]}
    dataset_key, dataset_cfg = _resolve_dataset_config(
        source_config, source_config.get("lab", {}).get("dataset_key", "mnist")
    )

    if str(model_cfg.get("non_linearity")) != "perfect_diode":
        raise ValueError(
            f"Expected source checkpoint non_linearity='perfect_diode', got {model_cfg.get('non_linearity')!r}."
        )
    if not math.isclose(float(model_cfg["voltage_amp"]), spec.voltage_amp):
        raise ValueError("voltage_amp mismatch between summary.csv and source_config.json.")
    if not math.isclose(float(model_cfg["current_amp"]), spec.current_amp):
        raise ValueError("current_amp mismatch between summary.csv and source_config.json.")

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
        input_mode=source_config.get("input_mode", "train"),
    )
    energy_fn.set_device(device)
    energy_fn.load(spec.checkpoint_path)

    network = Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]
    if output_layer.shape[0] == 10:
        cost_fn = SquaredError(output_layer)
    elif output_layer.shape[0] == 20:
        cost_fn = SquaredErrorPairedOutputs(output_layer, num_classes=10)
    else:
        raise ValueError(f"Unsupported output dimension {output_layer.shape[0]}.")

    minimizer = _build_tracking_minimizer(
        energy_fn,
        free_layers,
        model_cfg,
        source_config["energy_minimizer"]["mode"],
        num_iterations=int(model_cfg["num_iterations_inference"]),
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )

    dataset_factory = _resolve_callable(dataset_cfg["factory"])
    dataset_params = dict(dataset_cfg["params"])
    dataset_params["device"] = device
    if batch_size is not None:
        dataset_params["batch_size"] = int(batch_size)
    if dataset_root is not None:
        dataset_params["root"] = os.path.expanduser(str(dataset_root))
    elif "root" in dataset_params:
        dataset_params["root"] = os.path.expanduser(str(dataset_params["root"]))
    if no_download:
        dataset_params["download"] = False
    loader_result = dataset_factory(**dataset_params).build()
    if not isinstance(loader_result, tuple):
        raise ValueError("Expected MNIST dataset factory to return train and test loaders.")
    _, test_loader = loader_result

    return {
        "source_config": source_config,
        "resolved_config": resolved_config,
        "model_cfg": model_cfg,
        "dataset_key": dataset_key,
        "dataset_params": dataset_params,
        "energy_fn": energy_fn,
        "network": network,
        "free_layers": free_layers,
        "output_layer": output_layer,
        "cost_fn": cost_fn,
        "minimizer": minimizer,
        "test_loader": test_loader,
    }


def dense_weight_mats(energy_fn: FlexibleDeepResistiveEnergy) -> list[np.ndarray]:
    mats: list[np.ndarray] = []
    for param in getattr(energy_fn, "_all_params", energy_fn.params()):
        if not isinstance(param, DenseWeight):
            continue
        array = param.state.detach().cpu().numpy().astype(np.float64)
        if array.ndim < 2:
            raise ValueError(f"DenseWeight must have at least 2 dimensions, got {array.shape}.")
        mats.append(array.reshape(int(np.prod(array.shape[:-1])), array.shape[-1]))
    if len(mats) != 2:
        raise ValueError(f"Expected exactly two dense weight tensors for DRN-XS, got {len(mats)}.")
    return mats


def raw_output_predictions(output: torch.Tensor) -> torch.Tensor:
    if output.shape[1] == 10:
        return torch.argmax(output, dim=1)
    if output.shape[1] == 20:
        first, second = output[:, :10], output[:, 10:]
        return torch.argmax(first - second, dim=1)
    raise ValueError(f"Unsupported output shape {tuple(output.shape)}.")


def lpw_hidden_mask(hidden: np.ndarray) -> np.ndarray:
    if hidden.ndim != 2:
        raise ValueError(f"Expected hidden states with shape (samples, units), got {hidden.shape}.")
    if hidden.shape[1] % 2 != 0:
        raise ValueError(f"Expected even hidden dimension for excitatory/inhibitory split, got {hidden.shape[1]}.")
    half = hidden.shape[1] // 2
    mask = np.zeros(hidden.shape, dtype=bool)
    mask[:, :half] = hidden[:, :half] > 0.0
    mask[:, half:] = hidden[:, half:] < 0.0
    return mask


def safe_hessian_metrics(hessian: np.ndarray, free_dims: list[int], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        metrics = pcr.block_curvature_metrics(hessian, free_dims)
    except Exception:
        metrics = {
            "L_odd": math.nan,
            "L_even": math.nan,
            "sigma": math.nan,
            "L_full": math.nan,
            "kappa_full": math.nan,
        }
    try:
        rho = pcr.odd_even_spectral_radius(hessian, free_dims)
    except Exception:
        rho = math.nan
    inv_neg_log_rho, inv_one_minus_rho = pcr.stopping_time_proxies(rho)
    out[f"rho_{prefix}"] = rho
    out[f"inv_neg_log_rho_{prefix}"] = inv_neg_log_rho
    out[f"inv_one_minus_rho_{prefix}"] = inv_one_minus_rho
    out[f"sigma_{prefix}"] = float(metrics.get("sigma", math.nan))
    out[f"L_full_{prefix}"] = float(metrics.get("L_full", math.nan))
    out[f"kappa_full_{prefix}"] = float(metrics.get("kappa_full", math.nan))
    out[f"L_odd_{prefix}"] = float(metrics.get("L_odd", math.nan))
    out[f"L_even_{prefix}"] = float(metrics.get("L_even", math.nan))
    return out


def stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_p10": math.nan,
            f"{prefix}_p50": math.nan,
            f"{prefix}_p90": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_p10": float(np.percentile(finite, 10)),
        f"{prefix}_p50": float(np.percentile(finite, 50)),
        f"{prefix}_p90": float(np.percentile(finite, 90)),
    }


def collect_test_states(context: dict, max_test_samples: int | None) -> dict[str, np.ndarray]:
    network = context["network"]
    minimizer = context["minimizer"]
    output_layer = context["output_layer"]
    cost_fn = context["cost_fn"]
    test_loader = context["test_loader"]
    free_layers = context["free_layers"]

    labels_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    correct_all: list[np.ndarray] = []
    hidden_all: list[np.ndarray] = []
    losses: list[float] = []
    seen = 0

    for images, labels in test_loader:
        if max_test_samples is not None and seen >= max_test_samples:
            break
        remaining = None if max_test_samples is None else max_test_samples - seen
        if remaining is not None and images.shape[0] > remaining:
            images = images[:remaining]
            labels = labels[:remaining]
        images = images.to(context["energy_fn"]._device)
        labels = labels.to(context["energy_fn"]._device)

        network.set_input(images, reset=True)
        minimizer.compute_equilibrium()
        cost_fn.set_target(labels)

        output = output_layer.state.detach()
        preds = raw_output_predictions(output)
        correct = preds.eq(labels)
        batch_loss = cost_fn.eval().detach().mean().item()
        hidden = free_layers[0].state.detach().cpu().numpy().astype(np.float64)

        labels_all.append(labels.detach().cpu().numpy().astype(np.int64))
        preds_all.append(preds.detach().cpu().numpy().astype(np.int64))
        correct_all.append(correct.detach().cpu().numpy().astype(bool))
        hidden_all.append(hidden)
        losses.append(float(batch_loss))
        seen += int(images.shape[0])

    if seen == 0:
        raise ValueError("No test samples were evaluated.")

    return {
        "labels": np.concatenate(labels_all),
        "predictions": np.concatenate(preds_all),
        "correct": np.concatenate(correct_all),
        "hidden": np.concatenate(hidden_all),
        "mean_loss": np.asarray(losses, dtype=np.float64),
    }


def per_unique_mask_metrics(
    hessian_linear: np.ndarray,
    free_dims: list[int],
    unique_masks: np.ndarray,
    *,
    lpw_layer_conductance: float,
) -> list[dict[str, float]]:
    diag_idx = np.diag_indices_from(hessian_linear)
    output_zeros = np.zeros(free_dims[-1], dtype=np.float64)
    metrics: list[dict[str, float]] = []
    for mask in unique_masks:
        diag = np.concatenate([mask.astype(np.float64) * lpw_layer_conductance, output_zeros])
        hessian = hessian_linear.copy()
        hessian[diag_idx] += diag
        metrics.append(safe_hessian_metrics(hessian, free_dims, "lpw"))
    return metrics


def analyze_run(
    spec: RunSpec,
    args: argparse.Namespace,
    *,
    output_root: Path,
    device: torch.device,
) -> dict:
    out_dir = run_output_dir(output_root, spec)
    if run_complete(output_root, spec) and not args.rerun_complete:
        print(f"[skip] {spec.run_name} seed={spec.seed} already complete at {out_dir}")
        return json.loads((out_dir / "summary.json").read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    context = build_eval_context(
        spec,
        device=device,
        batch_size=args.batch_size,
        dataset_root=args.dataset_root,
        no_download=args.no_download,
    )
    mats = dense_weight_mats(context["energy_fn"])
    hessian_linear, free_dims = pcr.build_linear_hessian(mats, spec.voltage_amp, spec.current_amp)
    if hessian_linear.shape != (110, 110):
        raise ValueError(f"Expected DRN-XS H_linear shape 110x110, got {hessian_linear.shape}.")
    if not np.allclose(hessian_linear, hessian_linear.T, rtol=1e-9, atol=1e-12):
        raise ValueError("H_linear is not symmetric.")

    states = collect_test_states(context, args.max_test_samples)
    hidden_mask = lpw_hidden_mask(states["hidden"])
    if hidden_mask.shape[1] != free_dims[0]:
        raise ValueError(f"Hidden mask width {hidden_mask.shape[1]} does not match free_dims[0]={free_dims[0]}.")
    lpw_layer_conductance = float(args.lpw_conductance) * (spec.current_amp / spec.voltage_amp) ** 0

    unique_masks, inverse = np.unique(hidden_mask, axis=0, return_inverse=True)
    unique_metrics = per_unique_mask_metrics(
        hessian_linear,
        free_dims,
        unique_masks,
        lpw_layer_conductance=lpw_layer_conductance,
    )

    sample_count = hidden_mask.shape[0]
    sample_rows: list[dict] = []
    rho_vals = np.empty(sample_count, dtype=np.float64)
    kappa_vals = np.empty(sample_count, dtype=np.float64)
    sigma_vals = np.empty(sample_count, dtype=np.float64)
    active_fraction = hidden_mask.mean(axis=1).astype(np.float64)
    active_count = hidden_mask.sum(axis=1).astype(np.int64)
    for sample_index in range(sample_count):
        metric = unique_metrics[int(inverse[sample_index])]
        rho_vals[sample_index] = metric["rho_lpw"]
        kappa_vals[sample_index] = metric["kappa_full_lpw"]
        sigma_vals[sample_index] = metric["sigma_lpw"]
        sample_rows.append(
            {
                "run_name": spec.run_name,
                "seed": spec.seed,
                "voltage_amp": spec.voltage_amp,
                "current_amp": spec.current_amp,
                "sample_index": sample_index,
                "label": int(states["labels"][sample_index]),
                "prediction": int(states["predictions"][sample_index]),
                "correct": int(states["correct"][sample_index]),
                "lpw_active_hidden_count": int(active_count[sample_index]),
                "lpw_active_hidden_fraction": float(active_fraction[sample_index]),
                "rho_lpw": metric["rho_lpw"],
                "inv_neg_log_rho_lpw": metric["inv_neg_log_rho_lpw"],
                "inv_one_minus_rho_lpw": metric["inv_one_minus_rho_lpw"],
                "sigma_lpw": metric["sigma_lpw"],
                "L_full_lpw": metric["L_full_lpw"],
                "kappa_full_lpw": metric["kappa_full_lpw"],
                "L_odd_lpw": metric["L_odd_lpw"],
                "L_even_lpw": metric["L_even_lpw"],
            }
        )

    linear_metrics = safe_hessian_metrics(hessian_linear, free_dims, "linear")
    mean_diag_hidden = hidden_mask.mean(axis=0).astype(np.float64) * lpw_layer_conductance
    mean_diag = np.concatenate([mean_diag_hidden, np.zeros(free_dims[-1], dtype=np.float64)])
    hessian_lpw_mean = hessian_linear.copy()
    hessian_lpw_mean[np.diag_indices_from(hessian_lpw_mean)] += mean_diag
    mean_metrics = safe_hessian_metrics(hessian_lpw_mean, free_dims, "lpw_mean_diag")

    hessian_linear_path = out_dir / "hessian_linear.npz"
    hessian_lpw_mean_path = out_dir / "hessian_lpw_mean.npz"
    masks_path = out_dir / "lpw_conductance_masks.npz"
    sample_metrics_path = out_dir / "sample_metrics.csv"
    config_path = out_dir / "config.json"
    summary_path = out_dir / "summary.json"

    metadata = {
        "run_name": spec.run_name,
        "seed": spec.seed,
        "voltage_amp": spec.voltage_amp,
        "current_amp": spec.current_amp,
        "checkpoint_path": str(spec.checkpoint_path),
        "source_config_path": str(spec.source_config_path),
        "resolved_config_path": str(spec.resolved_config_path),
        "free_dims": free_dims,
        "lpw_conductance": float(args.lpw_conductance),
        "lpw_layer_conductance": lpw_layer_conductance,
        "lpw_rule": "excitatory v>0, inhibitory v<0, output conductance 0",
        "checkpoint_nonlinearity": "perfect_diode",
    }
    np.savez_compressed(
        hessian_linear_path,
        hessian=hessian_linear,
        free_dims=np.asarray(free_dims, dtype=np.int64),
        metrics_json=np.asarray(json.dumps(linear_metrics, sort_keys=True)),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    np.savez_compressed(
        hessian_lpw_mean_path,
        hessian=hessian_lpw_mean,
        free_dims=np.asarray(free_dims, dtype=np.int64),
        mean_lpw_diag=mean_diag,
        metrics_json=np.asarray(json.dumps(mean_metrics, sort_keys=True)),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    np.savez_compressed(
        masks_path,
        hidden_active_mask=hidden_mask,
        hidden_lpw_conductance=hidden_mask.astype(np.float32) * np.float32(lpw_layer_conductance),
        labels=states["labels"],
        predictions=states["predictions"],
        correct=states["correct"],
        unique_hidden_active_mask=unique_masks,
        inverse_unique_mask=inverse.astype(np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    write_csv(sample_metrics_path, sample_rows, SAMPLE_COLUMNS)

    evaluator_config = {
        "input_root": str(args.input_root),
        "output_root": str(output_root),
        "checkpoint": args.checkpoint,
        "lpw_conductance": float(args.lpw_conductance),
        "lpw_layer_conductance": lpw_layer_conductance,
        "device": str(device),
        "batch_size": args.batch_size,
        "max_test_samples": args.max_test_samples,
        "dataset_root_override": args.dataset_root,
        "source_config_path": str(spec.source_config_path),
        "resolved_config_path": str(spec.resolved_config_path),
        "checkpoint_path": str(spec.checkpoint_path),
        "dataset_params": context["dataset_params"],
        "source_preprocessing": context["resolved_config"].get("dataset", {}).get("preprocessing", {}),
    }
    config_path.write_text(json.dumps(_json_sanitize(evaluator_config), indent=2, sort_keys=True))

    summary = {
        "run_name": spec.run_name,
        "seed": spec.seed,
        "voltage_amp": spec.voltage_amp,
        "current_amp": spec.current_amp,
        "checkpoint_path": str(spec.checkpoint_path),
        "source_config_path": str(spec.source_config_path),
        "resolved_config_path": str(spec.resolved_config_path),
        "sample_count": int(sample_count),
        "unique_lpw_masks": int(unique_masks.shape[0]),
        "test_accuracy_recomputed": float(np.mean(states["correct"])),
        "rho_linear": linear_metrics["rho_linear"],
        "sigma_linear": linear_metrics["sigma_linear"],
        "L_full_linear": linear_metrics["L_full_linear"],
        "kappa_full_linear": linear_metrics["kappa_full_linear"],
        "L_odd_linear": linear_metrics["L_odd_linear"],
        "L_even_linear": linear_metrics["L_even_linear"],
        "rho_lpw_mean_diag": mean_metrics["rho_lpw_mean_diag"],
        "sigma_lpw_mean_diag": mean_metrics["sigma_lpw_mean_diag"],
        "L_full_lpw_mean_diag": mean_metrics["L_full_lpw_mean_diag"],
        "kappa_full_lpw_mean_diag": mean_metrics["kappa_full_lpw_mean_diag"],
        "L_odd_lpw_mean_diag": mean_metrics["L_odd_lpw_mean_diag"],
        "L_even_lpw_mean_diag": mean_metrics["L_even_lpw_mean_diag"],
        **stats(rho_vals, "rho_lpw"),
        **stats(kappa_vals, "kappa_full_lpw"),
        **stats(sigma_vals, "sigma_lpw"),
        **stats(active_fraction, "lpw_active_hidden_fraction"),
        "sample_metrics_path": str(sample_metrics_path),
        "hessian_linear_path": str(hessian_linear_path),
        "hessian_lpw_mean_path": str(hessian_lpw_mean_path),
        "lpw_masks_path": str(masks_path),
        "summary_path": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"[run] {spec.run_name} seed={spec.seed} samples={sample_count} "
        f"unique_masks={unique_masks.shape[0]} acc={summary['test_accuracy_recomputed']:.4f}"
    )
    return summary


def collect_run_summaries(output_root: Path, specs: list[RunSpec]) -> list[dict]:
    rows: list[dict] = []
    for spec in specs:
        path = run_output_dir(output_root, spec) / "summary.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    return rows


def aggregate_by_amp(summary_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for run_name in RUN_ORDER:
        rows = [row for row in summary_rows if row["run_name"] == run_name]
        if not rows:
            continue
        sample_rows: list[dict[str, str]] = []
        for row in rows:
            sample_path = Path(row["sample_metrics_path"])
            if sample_path.exists():
                sample_rows.extend(read_csv(sample_path))
        def sample_values(key: str) -> np.ndarray:
            return np.asarray([finite_float(row[key]) for row in sample_rows], dtype=np.float64)

        acc = np.asarray([finite_float(row["test_accuracy_recomputed"]) for row in rows], dtype=np.float64)
        result = {
            "run_name": run_name,
            "voltage_amp": rows[0]["voltage_amp"],
            "current_amp": rows[0]["current_amp"],
            "num_seeds": len(rows),
            "sample_count": len(sample_rows),
            "test_accuracy_mean": float(np.nanmean(acc)) if acc.size else math.nan,
            "test_accuracy_std": float(np.nanstd(acc)) if acc.size else math.nan,
            **stats(sample_values("rho_lpw"), "rho_lpw"),
            **stats(sample_values("kappa_full_lpw"), "kappa_full_lpw"),
            **stats(sample_values("sigma_lpw"), "sigma_lpw"),
            **stats(sample_values("lpw_active_hidden_fraction"), "lpw_active_hidden_fraction"),
        }
        out.append(result)
    return out


def make_by_amp_plot(rows: list[dict], key: str, ylabel: str, output_path: Path, *, log_y: bool = False) -> None:
    if not rows:
        return
    labels = [row["run_name"].replace("mnist_bp_amp_", "") for row in rows]
    means = np.asarray([finite_float(row[f"{key}_mean"]) for row in rows], dtype=np.float64)
    stds = np.asarray([finite_float(row[f"{key}_std"]) for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    x = np.arange(len(rows))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4c78a8", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    if log_y:
        ax.set_yscale("log")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_top_level_outputs(output_root: Path, summary_rows: list[dict]) -> None:
    summary_rows = sorted(summary_rows, key=lambda row: _summary_sort_key({k: str(v) for k, v in row.items()}))
    write_csv(output_root / "summary.csv", summary_rows, SUMMARY_COLUMNS)
    by_amp = aggregate_by_amp(summary_rows)
    write_csv(output_root / "summary_by_amp.csv", by_amp, BY_AMP_COLUMNS)
    make_by_amp_plot(by_amp, "rho_lpw", "LPW Hessian rho(T)", output_root / "rho_lpw_by_amp.png")
    make_by_amp_plot(
        by_amp,
        "kappa_full_lpw",
        "LPW Hessian kappa_full",
        output_root / "kappa_lpw_by_amp.png",
        log_y=True,
    )
    make_by_amp_plot(
        by_amp,
        "sigma_lpw",
        "LPW Hessian sigma",
        output_root / "sigma_lpw_by_amp.png",
        log_y=True,
    )
    make_by_amp_plot(
        by_amp,
        "lpw_active_hidden_fraction",
        "LPW active hidden fraction",
        output_root / "lpw_active_fraction_by_amp.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MNIST DRN-XS amplification Hessians using an LPW perfect-diode approximation."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--lpw-conductance", type=float, default=100.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--run-name", action="append", choices=RUN_ORDER)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--no-summary-after-run",
        action="store_true",
        help="Only write per-run outputs; useful for parallel launchers that collect once at the end.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device}, but CUDA is not available.")

    specs = read_run_specs(
        input_root,
        checkpoint=args.checkpoint,
        run_names=set(args.run_name) if args.run_name else None,
        seeds=set(args.seeds) if args.seeds else None,
    )

    if args.summary_only:
        all_specs = read_run_specs(
            input_root,
            checkpoint=args.checkpoint,
            run_names=None,
            seeds=set(args.seeds) if args.seeds else None,
        )
        summary_rows = collect_run_summaries(output_root, all_specs)
        write_top_level_outputs(output_root, summary_rows)
        print(f"[summary] wrote {output_root / 'summary.csv'}")
        print(f"[summary] wrote {output_root / 'summary_by_amp.csv'}")
        return

    print(f"[hessian] input_root={input_root}")
    print(f"[hessian] output_root={output_root}")
    print(f"[hessian] selected_runs={len(specs)} device={device}")
    for spec in specs:
        analyze_run(spec, args, output_root=output_root, device=device)

    if args.no_summary_after_run:
        print("[done] per-run outputs written; top-level summary skipped by request")
        return

    all_specs = read_run_specs(
        input_root,
        checkpoint=args.checkpoint,
        run_names=None,
        seeds=set(args.seeds) if args.seeds else None,
    )
    summary_rows = collect_run_summaries(output_root, all_specs)
    write_top_level_outputs(output_root, summary_rows)
    print(f"[done] summary={output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
