#!/usr/bin/env python3
"""Evaluate MNIST DRN Hessian metrics with the trained hard-sigmoid curvature."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = REPO_ROOT / "labs"
TOOLS_DIR = LABS_DIR / "tools"
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, LABS_DIR, TOOLS_DIR, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_mnist_bp_hessian_lpw_amplification_sweep as hess_eval  # noqa: E402
import plot_conditioning_vs_runtime as pcr  # noqa: E402


DEFAULT_INPUT_ROOT = (
    REPO_ROOT / "results" / "mnist_bp_amplification_sweep_hardsigmoid_voff15_legacy"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results" / "mnist_bp_hessian_hardsigmoid_voff15_legacy_drnxs"
)
RUN_ORDER = hess_eval.RUN_ORDER

SAMPLE_COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "sample_index",
    "label",
    "prediction",
    "correct",
    "hardsigmoid_active_hidden_count",
    "hardsigmoid_active_hidden_fraction",
    "rho_hardsigmoid",
    "inv_neg_log_rho_hardsigmoid",
    "inv_one_minus_rho_hardsigmoid",
    "sigma_hardsigmoid",
    "L_full_hardsigmoid",
    "kappa_full_hardsigmoid",
    "L_odd_hardsigmoid",
    "L_even_hardsigmoid",
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
    "unique_hardsigmoid_conductance_masks",
    "test_accuracy_recomputed",
    "rho_linear",
    "sigma_linear",
    "L_full_linear",
    "kappa_full_linear",
    "L_odd_linear",
    "L_even_linear",
    "rho_hardsigmoid_mean_diag",
    "sigma_hardsigmoid_mean_diag",
    "L_full_hardsigmoid_mean_diag",
    "kappa_full_hardsigmoid_mean_diag",
    "L_odd_hardsigmoid_mean_diag",
    "L_even_hardsigmoid_mean_diag",
    "rho_hardsigmoid_mean",
    "rho_hardsigmoid_std",
    "rho_hardsigmoid_p10",
    "rho_hardsigmoid_p50",
    "rho_hardsigmoid_p90",
    "kappa_full_hardsigmoid_mean",
    "kappa_full_hardsigmoid_std",
    "kappa_full_hardsigmoid_p10",
    "kappa_full_hardsigmoid_p50",
    "kappa_full_hardsigmoid_p90",
    "sigma_hardsigmoid_mean",
    "sigma_hardsigmoid_std",
    "sigma_hardsigmoid_p10",
    "sigma_hardsigmoid_p50",
    "sigma_hardsigmoid_p90",
    "hardsigmoid_active_hidden_fraction_mean",
    "hardsigmoid_active_hidden_fraction_std",
    "hardsigmoid_active_hidden_fraction_p10",
    "hardsigmoid_active_hidden_fraction_p50",
    "hardsigmoid_active_hidden_fraction_p90",
    "sample_metrics_path",
    "hessian_linear_path",
    "hessian_hardsigmoid_mean_path",
    "hardsigmoid_masks_path",
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
    "rho_hardsigmoid_mean",
    "rho_hardsigmoid_std",
    "rho_hardsigmoid_p10",
    "rho_hardsigmoid_p50",
    "rho_hardsigmoid_p90",
    "kappa_full_hardsigmoid_mean",
    "kappa_full_hardsigmoid_std",
    "kappa_full_hardsigmoid_p10",
    "kappa_full_hardsigmoid_p50",
    "kappa_full_hardsigmoid_p90",
    "sigma_hardsigmoid_mean",
    "sigma_hardsigmoid_std",
    "sigma_hardsigmoid_p10",
    "sigma_hardsigmoid_p50",
    "sigma_hardsigmoid_p90",
    "hardsigmoid_active_hidden_fraction_mean",
    "hardsigmoid_active_hidden_fraction_std",
    "hardsigmoid_active_hidden_fraction_p10",
    "hardsigmoid_active_hidden_fraction_p50",
    "hardsigmoid_active_hidden_fraction_p90",
]


def run_output_dir(output_root: Path, spec: hess_eval.RunSpec) -> Path:
    return output_root / spec.run_name / f"seed_{spec.seed}"


def run_complete(output_root: Path, spec: hess_eval.RunSpec) -> bool:
    run_dir = run_output_dir(output_root, spec)
    required = [
        "hessian_linear.npz",
        "hessian_hardsigmoid_mean.npz",
        "hardsigmoid_conductance_masks.npz",
        "sample_metrics.csv",
        "summary.json",
        "config.json",
    ]
    return all((run_dir / name).exists() for name in required)


def build_eval_context(
    spec: hess_eval.RunSpec,
    *,
    device: torch.device,
    batch_size: int | None,
    dataset_root: str | None,
    no_download: bool,
) -> dict:
    if not spec.checkpoint_path.exists():
        raise FileNotFoundError(f"Expected checkpoint at {spec.checkpoint_path}.")

    source_config, resolved_config = hess_eval.load_source_and_resolved_config(spec)
    hess_eval._reset_name_counters()
    hess_eval._set_seed(spec.seed)

    model_key = source_config.get("lab", {}).get("model_key", "mnist_bp_amp")
    model_overrides = source_config["model_overrides"]
    if model_key not in model_overrides and len(model_overrides) == 1:
        model_key = next(iter(model_overrides))
    model_cfg = {**source_config["model_base"], **model_overrides[model_key]}
    dataset_key, dataset_cfg = hess_eval._resolve_dataset_config(
        source_config, source_config.get("lab", {}).get("dataset_key", "mnist")
    )

    if str(model_cfg.get("non_linearity")) != "hard_sigmoid":
        raise ValueError(
            "Expected source checkpoint non_linearity='hard_sigmoid', "
            f"got {model_cfg.get('non_linearity')!r}."
        )
    if not math.isclose(float(model_cfg["voltage_amp"]), spec.voltage_amp):
        raise ValueError("voltage_amp mismatch between summary.csv and source_config.json.")
    if not math.isclose(float(model_cfg["current_amp"]), spec.current_amp):
        raise ValueError("current_amp mismatch between summary.csv and source_config.json.")

    layer_shapes = [
        tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        for shape in model_cfg["layer_shapes"]
    ]
    energy_fn = hess_eval.FlexibleDeepResistiveEnergy(
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

    network = hess_eval.Network(energy_fn)
    free_layers = network.free_layers()
    output_layer = energy_fn.layers()[-1]
    if output_layer.shape[0] == 10:
        cost_fn = hess_eval.SquaredError(output_layer)
    elif output_layer.shape[0] == 20:
        cost_fn = hess_eval.SquaredErrorPairedOutputs(output_layer, num_classes=10)
    else:
        raise ValueError(f"Unsupported output dimension {output_layer.shape[0]}.")

    minimizer = hess_eval._build_tracking_minimizer(
        energy_fn,
        free_layers,
        model_cfg,
        source_config["energy_minimizer"]["mode"],
        num_iterations=int(model_cfg["num_iterations_inference"]),
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )

    dataset_factory = hess_eval._resolve_callable(dataset_cfg["factory"])
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


def hard_sigmoid_hidden_conductance(
    hidden: np.ndarray,
    *,
    params: dict,
    voltage_amp: float,
    current_amp: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    if hidden.ndim != 2:
        raise ValueError(f"Expected hidden states with shape (samples, units), got {hidden.shape}.")
    v_min = float(params["v_min"])
    v_max = float(params["v_max"])
    g_on = float(params["g_on"])
    g_off = float(params["g_off"])
    layer_scale = float((current_amp / voltage_amp) ** 0)
    active = (hidden < v_min) | (hidden > v_max)
    conductance = np.where(active, g_on * layer_scale, g_off * layer_scale).astype(np.float64)
    return active, conductance, g_on, g_off, v_min, v_max, layer_scale


def per_unique_conductance_metrics(
    hessian_linear: np.ndarray,
    free_dims: list[int],
    unique_hidden_conductance: np.ndarray,
) -> list[dict[str, float]]:
    diag_idx = np.diag_indices_from(hessian_linear)
    output_zeros = np.zeros(free_dims[-1], dtype=np.float64)
    metrics: list[dict[str, float]] = []
    for hidden_diag in unique_hidden_conductance:
        diag = np.concatenate([hidden_diag.astype(np.float64), output_zeros])
        hessian = hessian_linear.copy()
        hessian[diag_idx] += diag
        metrics.append(hess_eval.safe_hessian_metrics(hessian, free_dims, "hardsigmoid"))
    return metrics


def analyze_run(
    spec: hess_eval.RunSpec,
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
    mats = hess_eval.dense_weight_mats(context["energy_fn"])
    hessian_linear, free_dims = pcr.build_linear_hessian(mats, spec.voltage_amp, spec.current_amp)
    if hessian_linear.shape != (110, 110):
        raise ValueError(f"Expected DRN-XS H_linear shape 110x110, got {hessian_linear.shape}.")
    if not np.allclose(hessian_linear, hessian_linear.T, rtol=1e-9, atol=1e-12):
        raise ValueError("H_linear is not symmetric.")

    states = hess_eval.collect_test_states(context, args.max_test_samples)
    hs_params = context["model_cfg"].get("hard_sigmoid_param", {})
    active_mask, hidden_conductance, g_on, g_off, v_min, v_max, layer_scale = (
        hard_sigmoid_hidden_conductance(
            states["hidden"],
            params=hs_params,
            voltage_amp=spec.voltage_amp,
            current_amp=spec.current_amp,
        )
    )
    if hidden_conductance.shape[1] != free_dims[0]:
        raise ValueError(
            f"Hidden conductance width {hidden_conductance.shape[1]} does not match free_dims[0]={free_dims[0]}."
        )

    unique_conductance, inverse = np.unique(hidden_conductance, axis=0, return_inverse=True)
    unique_metrics = per_unique_conductance_metrics(hessian_linear, free_dims, unique_conductance)

    sample_count = hidden_conductance.shape[0]
    sample_rows: list[dict] = []
    rho_vals = np.empty(sample_count, dtype=np.float64)
    kappa_vals = np.empty(sample_count, dtype=np.float64)
    sigma_vals = np.empty(sample_count, dtype=np.float64)
    active_fraction = active_mask.mean(axis=1).astype(np.float64)
    active_count = active_mask.sum(axis=1).astype(np.int64)
    for sample_index in range(sample_count):
        metric = unique_metrics[int(inverse[sample_index])]
        rho_vals[sample_index] = metric["rho_hardsigmoid"]
        kappa_vals[sample_index] = metric["kappa_full_hardsigmoid"]
        sigma_vals[sample_index] = metric["sigma_hardsigmoid"]
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
                "hardsigmoid_active_hidden_count": int(active_count[sample_index]),
                "hardsigmoid_active_hidden_fraction": float(active_fraction[sample_index]),
                "rho_hardsigmoid": metric["rho_hardsigmoid"],
                "inv_neg_log_rho_hardsigmoid": metric["inv_neg_log_rho_hardsigmoid"],
                "inv_one_minus_rho_hardsigmoid": metric["inv_one_minus_rho_hardsigmoid"],
                "sigma_hardsigmoid": metric["sigma_hardsigmoid"],
                "L_full_hardsigmoid": metric["L_full_hardsigmoid"],
                "kappa_full_hardsigmoid": metric["kappa_full_hardsigmoid"],
                "L_odd_hardsigmoid": metric["L_odd_hardsigmoid"],
                "L_even_hardsigmoid": metric["L_even_hardsigmoid"],
            }
        )

    linear_metrics = hess_eval.safe_hessian_metrics(hessian_linear, free_dims, "linear")
    mean_diag_hidden = hidden_conductance.mean(axis=0).astype(np.float64)
    mean_diag = np.concatenate([mean_diag_hidden, np.zeros(free_dims[-1], dtype=np.float64)])
    hessian_hardsigmoid_mean = hessian_linear.copy()
    hessian_hardsigmoid_mean[np.diag_indices_from(hessian_hardsigmoid_mean)] += mean_diag
    mean_metrics = hess_eval.safe_hessian_metrics(
        hessian_hardsigmoid_mean, free_dims, "hardsigmoid_mean_diag"
    )

    hessian_linear_path = out_dir / "hessian_linear.npz"
    hessian_hardsigmoid_mean_path = out_dir / "hessian_hardsigmoid_mean.npz"
    masks_path = out_dir / "hardsigmoid_conductance_masks.npz"
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
        "hard_sigmoid_g_on": g_on,
        "hard_sigmoid_g_off": g_off,
        "hard_sigmoid_v_min": v_min,
        "hard_sigmoid_v_max": v_max,
        "hard_sigmoid_layer_scale": layer_scale,
        "hard_sigmoid_rule": "hidden conductance is g_off inside [v_min,v_max], g_on outside; output conductance 0",
        "checkpoint_nonlinearity": "hard_sigmoid",
    }
    np.savez_compressed(
        hessian_linear_path,
        hessian=hessian_linear,
        free_dims=np.asarray(free_dims, dtype=np.int64),
        metrics_json=np.asarray(json.dumps(linear_metrics, sort_keys=True)),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    np.savez_compressed(
        hessian_hardsigmoid_mean_path,
        hessian=hessian_hardsigmoid_mean,
        free_dims=np.asarray(free_dims, dtype=np.int64),
        mean_hardsigmoid_diag=mean_diag,
        metrics_json=np.asarray(json.dumps(mean_metrics, sort_keys=True)),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    np.savez_compressed(
        masks_path,
        hidden_active_mask=active_mask,
        hidden_hardsigmoid_conductance=hidden_conductance.astype(np.float32),
        labels=states["labels"],
        predictions=states["predictions"],
        correct=states["correct"],
        unique_hidden_hardsigmoid_conductance=unique_conductance.astype(np.float32),
        inverse_unique_conductance=inverse.astype(np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    hess_eval.write_csv(sample_metrics_path, sample_rows, SAMPLE_COLUMNS)

    evaluator_config = {
        "input_root": str(args.input_root),
        "output_root": str(output_root),
        "checkpoint": args.checkpoint,
        "device": str(device),
        "batch_size": args.batch_size,
        "max_test_samples": args.max_test_samples,
        "dataset_root_override": args.dataset_root,
        "source_config_path": str(spec.source_config_path),
        "resolved_config_path": str(spec.resolved_config_path),
        "checkpoint_path": str(spec.checkpoint_path),
        "dataset_params": context["dataset_params"],
        "source_preprocessing": context["resolved_config"].get("dataset", {}).get("preprocessing", {}),
        "hard_sigmoid_param": hs_params,
    }
    config_path.write_text(json.dumps(hess_eval._json_sanitize(evaluator_config), indent=2, sort_keys=True))

    summary = {
        "run_name": spec.run_name,
        "seed": spec.seed,
        "voltage_amp": spec.voltage_amp,
        "current_amp": spec.current_amp,
        "checkpoint_path": str(spec.checkpoint_path),
        "source_config_path": str(spec.source_config_path),
        "resolved_config_path": str(spec.resolved_config_path),
        "sample_count": int(sample_count),
        "unique_hardsigmoid_conductance_masks": int(unique_conductance.shape[0]),
        "test_accuracy_recomputed": float(np.mean(states["correct"])),
        "rho_linear": linear_metrics["rho_linear"],
        "sigma_linear": linear_metrics["sigma_linear"],
        "L_full_linear": linear_metrics["L_full_linear"],
        "kappa_full_linear": linear_metrics["kappa_full_linear"],
        "L_odd_linear": linear_metrics["L_odd_linear"],
        "L_even_linear": linear_metrics["L_even_linear"],
        "rho_hardsigmoid_mean_diag": mean_metrics["rho_hardsigmoid_mean_diag"],
        "sigma_hardsigmoid_mean_diag": mean_metrics["sigma_hardsigmoid_mean_diag"],
        "L_full_hardsigmoid_mean_diag": mean_metrics["L_full_hardsigmoid_mean_diag"],
        "kappa_full_hardsigmoid_mean_diag": mean_metrics["kappa_full_hardsigmoid_mean_diag"],
        "L_odd_hardsigmoid_mean_diag": mean_metrics["L_odd_hardsigmoid_mean_diag"],
        "L_even_hardsigmoid_mean_diag": mean_metrics["L_even_hardsigmoid_mean_diag"],
        **hess_eval.stats(rho_vals, "rho_hardsigmoid"),
        **hess_eval.stats(kappa_vals, "kappa_full_hardsigmoid"),
        **hess_eval.stats(sigma_vals, "sigma_hardsigmoid"),
        **hess_eval.stats(active_fraction, "hardsigmoid_active_hidden_fraction"),
        "sample_metrics_path": str(sample_metrics_path),
        "hessian_linear_path": str(hessian_linear_path),
        "hessian_hardsigmoid_mean_path": str(hessian_hardsigmoid_mean_path),
        "hardsigmoid_masks_path": str(masks_path),
        "summary_path": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"[run] {spec.run_name} seed={spec.seed} samples={sample_count} "
        f"unique_conductance_masks={unique_conductance.shape[0]} acc={summary['test_accuracy_recomputed']:.4f}"
    )
    return summary


def collect_run_summaries(output_root: Path, specs: list[hess_eval.RunSpec]) -> list[dict]:
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
                sample_rows.extend(hess_eval.read_csv(sample_path))

        def sample_values(key: str) -> np.ndarray:
            return np.asarray([hess_eval.finite_float(row[key]) for row in sample_rows], dtype=np.float64)

        acc = np.asarray([hess_eval.finite_float(row["test_accuracy_recomputed"]) for row in rows], dtype=np.float64)
        result = {
            "run_name": run_name,
            "voltage_amp": rows[0]["voltage_amp"],
            "current_amp": rows[0]["current_amp"],
            "num_seeds": len(rows),
            "sample_count": len(sample_rows),
            "test_accuracy_mean": float(np.nanmean(acc)) if acc.size else math.nan,
            "test_accuracy_std": float(np.nanstd(acc)) if acc.size else math.nan,
            **hess_eval.stats(sample_values("rho_hardsigmoid"), "rho_hardsigmoid"),
            **hess_eval.stats(sample_values("kappa_full_hardsigmoid"), "kappa_full_hardsigmoid"),
            **hess_eval.stats(sample_values("sigma_hardsigmoid"), "sigma_hardsigmoid"),
            **hess_eval.stats(
                sample_values("hardsigmoid_active_hidden_fraction"),
                "hardsigmoid_active_hidden_fraction",
            ),
        }
        out.append(result)
    return out


def write_top_level_outputs(output_root: Path, summary_rows: list[dict]) -> None:
    summary_rows = sorted(
        summary_rows,
        key=lambda row: hess_eval._summary_sort_key({key: str(value) for key, value in row.items()}),
    )
    hess_eval.write_csv(output_root / "summary.csv", summary_rows, SUMMARY_COLUMNS)
    by_amp = aggregate_by_amp(summary_rows)
    hess_eval.write_csv(output_root / "summary_by_amp.csv", by_amp, BY_AMP_COLUMNS)
    hess_eval.make_by_amp_plot(
        by_amp,
        "rho_hardsigmoid",
        "Hard-sigmoid Hessian rho(T)",
        output_root / "rho_hardsigmoid_by_amp.png",
    )
    hess_eval.make_by_amp_plot(
        by_amp,
        "kappa_full_hardsigmoid",
        "Hard-sigmoid Hessian kappa_full",
        output_root / "kappa_hardsigmoid_by_amp.png",
        log_y=True,
    )
    hess_eval.make_by_amp_plot(
        by_amp,
        "sigma_hardsigmoid",
        "Hard-sigmoid Hessian sigma",
        output_root / "sigma_hardsigmoid_by_amp.png",
        log_y=True,
    )
    hess_eval.make_by_amp_plot(
        by_amp,
        "hardsigmoid_active_hidden_fraction",
        "Hard-sigmoid active hidden fraction",
        output_root / "hardsigmoid_active_fraction_by_amp.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MNIST DRN-XS amplification Hessians using actual hard-sigmoid curvature."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
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

    specs = hess_eval.read_run_specs(
        input_root,
        checkpoint=args.checkpoint,
        run_names=set(args.run_name) if args.run_name else None,
        seeds=set(args.seeds) if args.seeds else None,
    )

    if args.summary_only:
        all_specs = hess_eval.read_run_specs(
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

    all_specs = hess_eval.read_run_specs(
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
