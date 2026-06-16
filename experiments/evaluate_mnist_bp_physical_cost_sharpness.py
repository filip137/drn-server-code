#!/usr/bin/env python3
"""Evaluate physical-cost sharpness for MNIST DRN amplification checkpoints.

For log-conductance write noise G_noisy = G exp(sigma xi), the local
Gauss-Newton prediction is

    E[Delta loss] ~= 0.5 * sigma^2 * ||P H_phys^{-1} A_phi||_F^2

where A_phi = (dr/dG) diag(G).  This script computes that Frobenius norm
for the trained DRN-XS checkpoints without explicitly materializing A_phi.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = REPO_ROOT / "labs"
TOOLS_DIR = LABS_DIR / "tools"
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, LABS_DIR, TOOLS_DIR, EXPERIMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_mnist_bp_hessian_hardsigmoid_amplification_sweep as hs_eval  # noqa: E402
import evaluate_mnist_bp_hessian_lpw_amplification_sweep as hess_eval  # noqa: E402
import plot_conditioning_vs_runtime as pcr  # noqa: E402


DEFAULT_INPUT_ROOT = (
    REPO_ROOT / "results" / "mnist_bp_amplification_sweep_hardsigmoid_voff15_legacy"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results" / "mnist_bp_physical_cost_sharpness_hardsigmoid_voff15_iter16"
)
RUN_ORDER = hs_eval.RUN_ORDER

SAMPLE_COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "sample_index",
    "label",
    "prediction",
    "correct",
    "loss",
    "hardsigmoid_active_hidden_count",
    "hardsigmoid_active_hidden_fraction",
    "s_write",
    "s_write_weight0",
    "s_write_weight1",
    "s_write_per_conductance",
    "s_write_weight0_per_conductance",
    "s_write_weight1_per_conductance",
    "pred_delta_mse_sigma_0p1",
    "pred_delta_mse_sigma_0p2",
    "pred_delta_mse_sigma_0p5",
    "pred_delta_mse_sigma_1p0",
    "linear_solve_method",
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
    "test_accuracy_recomputed",
    "test_loss_recomputed",
    "num_conductances_weight0",
    "num_conductances_weight1",
    "num_conductances_total",
    "unique_hardsigmoid_conductance_masks",
    "inference_iterations",
    "s_write_mean",
    "s_write_std",
    "s_write_p10",
    "s_write_p50",
    "s_write_p90",
    "s_write_weight0_mean",
    "s_write_weight0_std",
    "s_write_weight0_p10",
    "s_write_weight0_p50",
    "s_write_weight0_p90",
    "s_write_weight1_mean",
    "s_write_weight1_std",
    "s_write_weight1_p10",
    "s_write_weight1_p50",
    "s_write_weight1_p90",
    "s_write_per_conductance_mean",
    "s_write_per_conductance_std",
    "s_write_per_conductance_p10",
    "s_write_per_conductance_p50",
    "s_write_per_conductance_p90",
    "pred_delta_mse_sigma_0p1_mean",
    "pred_delta_mse_sigma_0p1_std",
    "pred_delta_mse_sigma_0p1_p10",
    "pred_delta_mse_sigma_0p1_p50",
    "pred_delta_mse_sigma_0p1_p90",
    "pred_delta_mse_sigma_0p2_mean",
    "pred_delta_mse_sigma_0p2_std",
    "pred_delta_mse_sigma_0p2_p10",
    "pred_delta_mse_sigma_0p2_p50",
    "pred_delta_mse_sigma_0p2_p90",
    "pred_delta_mse_sigma_0p5_mean",
    "pred_delta_mse_sigma_0p5_std",
    "pred_delta_mse_sigma_0p5_p10",
    "pred_delta_mse_sigma_0p5_p50",
    "pred_delta_mse_sigma_0p5_p90",
    "pred_delta_mse_sigma_1p0_mean",
    "pred_delta_mse_sigma_1p0_std",
    "pred_delta_mse_sigma_1p0_p10",
    "pred_delta_mse_sigma_1p0_p50",
    "pred_delta_mse_sigma_1p0_p90",
    "hardsigmoid_active_hidden_fraction_mean",
    "hardsigmoid_active_hidden_fraction_std",
    "hardsigmoid_active_hidden_fraction_p10",
    "hardsigmoid_active_hidden_fraction_p50",
    "hardsigmoid_active_hidden_fraction_p90",
    "sample_metrics_path",
    "arrays_path",
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
    "s_write_mean",
    "s_write_std",
    "s_write_p10",
    "s_write_p50",
    "s_write_p90",
    "s_write_weight0_mean",
    "s_write_weight0_std",
    "s_write_weight0_p10",
    "s_write_weight0_p50",
    "s_write_weight0_p90",
    "s_write_weight1_mean",
    "s_write_weight1_std",
    "s_write_weight1_p10",
    "s_write_weight1_p50",
    "s_write_weight1_p90",
    "s_write_per_conductance_mean",
    "s_write_per_conductance_std",
    "s_write_per_conductance_p10",
    "s_write_per_conductance_p50",
    "s_write_per_conductance_p90",
    "pred_delta_mse_sigma_0p2_mean",
    "pred_delta_mse_sigma_0p2_std",
    "pred_delta_mse_sigma_0p2_p10",
    "pred_delta_mse_sigma_0p2_p50",
    "pred_delta_mse_sigma_0p2_p90",
    "pred_delta_mse_sigma_0p5_mean",
    "pred_delta_mse_sigma_0p5_std",
    "pred_delta_mse_sigma_0p5_p10",
    "pred_delta_mse_sigma_0p5_p50",
    "pred_delta_mse_sigma_0p5_p90",
]

PREDICTION_SIGMAS = [0.1, 0.2, 0.5, 1.0]


def sigma_key(sigma: float) -> str:
    return str(sigma).replace(".", "p")


def run_output_dir(output_root: Path, spec: hess_eval.RunSpec) -> Path:
    return output_root / spec.run_name / f"seed_{spec.seed}"


def run_complete(output_root: Path, spec: hess_eval.RunSpec) -> bool:
    run_dir = run_output_dir(output_root, spec)
    required = ["sample_sharpness.csv", "sharpness_arrays.npz", "summary.json", "config.json"]
    return all((run_dir / name).exists() for name in required)


def override_minimizer_iterations(context: dict, iterations: int | None) -> int:
    source_iterations = int(context["model_cfg"]["num_iterations_inference"])
    if iterations is None:
        return source_iterations
    context["minimizer"] = hess_eval._build_tracking_minimizer(
        context["energy_fn"],
        context["free_layers"],
        context["model_cfg"],
        context["source_config"]["energy_minimizer"]["mode"],
        num_iterations=int(iterations),
        voltage_amp=context["energy_fn"]._voltage_amp,
        current_amp=context["energy_fn"]._current_amp,
    )
    return int(iterations)


def output_projection(output_dim: int, num_classes: int = 10) -> np.ndarray:
    if output_dim == num_classes:
        return np.eye(num_classes, dtype=np.float64)
    if output_dim == 2 * num_classes:
        return np.concatenate(
            [np.eye(num_classes, dtype=np.float64), -np.eye(num_classes, dtype=np.float64)],
            axis=1,
        )
    raise ValueError(f"Unsupported output dimension {output_dim}.")


def solve_output_response(
    hessian: np.ndarray,
    free_dims: list[int],
    *,
    num_classes: int = 10,
    pinv_rcond: float = 1e-10,
) -> tuple[np.ndarray, str]:
    output_dim = free_dims[-1]
    total_dim = int(sum(free_dims))
    proj_out = output_projection(output_dim, num_classes=num_classes)
    rhs = np.zeros((total_dim, num_classes), dtype=np.float64)
    rhs[total_dim - output_dim : total_dim, :] = proj_out.T
    try:
        return np.linalg.solve(hessian, rhs), "solve"
    except np.linalg.LinAlgError:
        return np.linalg.pinv(hessian, rcond=pinv_rcond) @ rhs, "pinv"


def collect_samples(context: dict, max_test_samples: int | None) -> dict[str, np.ndarray]:
    network = context["network"]
    minimizer = context["minimizer"]
    input_layer = context["energy_fn"].layers()[0]
    output_layer = context["output_layer"]
    cost_fn = context["cost_fn"]
    free_layers = context["free_layers"]
    test_loader = context["test_loader"]
    device = context["energy_fn"]._device

    inputs_all: list[np.ndarray] = []
    hidden_all: list[np.ndarray] = []
    output_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    correct_all: list[np.ndarray] = []
    losses_all: list[np.ndarray] = []
    seen = 0

    with torch.no_grad():
        for images, labels in test_loader:
            if max_test_samples is not None and seen >= max_test_samples:
                break
            remaining = None if max_test_samples is None else max_test_samples - seen
            if remaining is not None and images.shape[0] > remaining:
                images = images[:remaining]
                labels = labels[:remaining]

            images = images.to(device)
            labels = labels.to(device)
            network.set_input(images, reset=True)
            physical_input = input_layer.state.detach()
            minimizer.compute_equilibrium()
            cost_fn.set_target(labels)

            output = output_layer.state.detach()
            preds = hess_eval.raw_output_predictions(output)
            correct = preds.eq(labels)
            losses = cost_fn.eval().detach()
            hidden = free_layers[0].state.detach()

            inputs_all.append(
                physical_input.cpu().numpy().reshape(physical_input.shape[0], -1).astype(np.float64)
            )
            hidden_all.append(hidden.cpu().numpy().astype(np.float64))
            output_all.append(output.cpu().numpy().astype(np.float64))
            labels_all.append(labels.detach().cpu().numpy().astype(np.int64))
            preds_all.append(preds.detach().cpu().numpy().astype(np.int64))
            correct_all.append(correct.detach().cpu().numpy().astype(bool))
            losses_all.append(losses.cpu().numpy().astype(np.float64))
            seen += int(images.shape[0])

    if seen == 0:
        raise ValueError("No test samples were evaluated.")
    return {
        "inputs": np.concatenate(inputs_all, axis=0),
        "hidden": np.concatenate(hidden_all, axis=0),
        "output": np.concatenate(output_all, axis=0),
        "labels": np.concatenate(labels_all, axis=0),
        "predictions": np.concatenate(preds_all, axis=0),
        "correct": np.concatenate(correct_all, axis=0),
        "loss": np.concatenate(losses_all, axis=0),
    }


def physical_cost_sharpness(
    *,
    x_flat: np.ndarray,
    hidden: np.ndarray,
    output: np.ndarray,
    weight0: np.ndarray,
    weight1: np.ndarray,
    z_response: np.ndarray,
    voltage_amp: float,
    current_amp: float,
) -> tuple[float, float, float]:
    hidden_dim = hidden.shape[0]
    output_dim = output.shape[0]
    z_hidden = z_response[:hidden_dim, :]
    z_output = z_response[hidden_dim : hidden_dim + output_dim, :]

    z_hidden_norm2 = np.einsum("hc,hc->h", z_hidden, z_hidden)
    scalar0 = weight0 * current_amp * (current_amp * hidden[None, :] - x_flat[:, None])
    s0 = float(np.sum((scalar0 * scalar0) * z_hidden_norm2[None, :]))

    edge_delta = voltage_amp * hidden[:, None] - current_amp * output[None, :]
    response1 = (
        current_amp * z_hidden[:, None, :]
        - (current_amp * current_amp / voltage_amp) * z_output[None, :, :]
    )
    weighted_delta = weight1 * edge_delta
    s1 = float(np.sum((weighted_delta[:, :, None] * response1) ** 2))
    return s0 + s1, s0, s1


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

    context = hs_eval.build_eval_context(
        spec,
        device=device,
        batch_size=args.batch_size,
        dataset_root=args.dataset_root,
        no_download=args.no_download,
    )
    inference_iterations = override_minimizer_iterations(context, args.inference_iterations)
    weights = hess_eval.dense_weight_mats(context["energy_fn"])
    weight0, weight1 = weights
    if np.any(weight0 < 0.0) or np.any(weight1 < 0.0):
        raise ValueError("Log-conductance sharpness expects nonnegative physical conductances.")

    hessian_linear, free_dims = pcr.build_linear_hessian(weights, spec.voltage_amp, spec.current_amp)
    if len(free_dims) != 2:
        raise ValueError(f"Expected one hidden layer and one output layer, got free_dims={free_dims}.")
    samples = collect_samples(context, args.max_test_samples)
    active_mask, hidden_conductance, *_ = hs_eval.hard_sigmoid_hidden_conductance(
        samples["hidden"],
        params=context["model_cfg"].get("hard_sigmoid_param", {}),
        voltage_amp=spec.voltage_amp,
        current_amp=spec.current_amp,
    )

    hidden_dim, output_dim = free_dims
    total_dim = hidden_dim + output_dim
    if hessian_linear.shape != (total_dim, total_dim):
        raise ValueError(f"Unexpected H_phys shape {hessian_linear.shape}, expected {(total_dim, total_dim)}.")
    if samples["inputs"].shape[1] != weight0.shape[0]:
        raise ValueError(
            f"Input width {samples['inputs'].shape[1]} does not match DenseWeight_0 rows {weight0.shape[0]}."
        )

    unique_conductance, inverse = np.unique(hidden_conductance, axis=0, return_inverse=True)
    z_cache: dict[int, tuple[np.ndarray, str]] = {}
    diag_idx = np.diag_indices_from(hessian_linear)
    for idx, hidden_diag in enumerate(unique_conductance):
        hessian = hessian_linear.copy()
        hessian[diag_idx] += np.concatenate([hidden_diag, np.zeros(output_dim, dtype=np.float64)])
        if args.ridge:
            hessian[diag_idx] += float(args.ridge)
        z_cache[idx] = solve_output_response(
            hessian,
            free_dims,
            num_classes=args.num_classes,
            pinv_rcond=args.pinv_rcond,
        )

    sample_rows: list[dict] = []
    s_write = np.empty(samples["hidden"].shape[0], dtype=np.float64)
    s_write0 = np.empty_like(s_write)
    s_write1 = np.empty_like(s_write)
    solve_methods: list[str] = []
    active_fraction = active_mask.mean(axis=1).astype(np.float64)
    active_count = active_mask.sum(axis=1).astype(np.int64)
    n0 = int(weight0.size)
    n1 = int(weight1.size)
    ntotal = n0 + n1

    for sample_index in range(samples["hidden"].shape[0]):
        z_response, solve_method = z_cache[int(inverse[sample_index])]
        total, layer0, layer1 = physical_cost_sharpness(
            x_flat=samples["inputs"][sample_index],
            hidden=samples["hidden"][sample_index],
            output=samples["output"][sample_index],
            weight0=weight0,
            weight1=weight1,
            z_response=z_response,
            voltage_amp=spec.voltage_amp,
            current_amp=spec.current_amp,
        )
        s_write[sample_index] = total
        s_write0[sample_index] = layer0
        s_write1[sample_index] = layer1
        solve_methods.append(solve_method)
        row = {
            "run_name": spec.run_name,
            "seed": spec.seed,
            "voltage_amp": spec.voltage_amp,
            "current_amp": spec.current_amp,
            "sample_index": sample_index,
            "label": int(samples["labels"][sample_index]),
            "prediction": int(samples["predictions"][sample_index]),
            "correct": int(samples["correct"][sample_index]),
            "loss": float(samples["loss"][sample_index]),
            "hardsigmoid_active_hidden_count": int(active_count[sample_index]),
            "hardsigmoid_active_hidden_fraction": float(active_fraction[sample_index]),
            "s_write": float(total),
            "s_write_weight0": float(layer0),
            "s_write_weight1": float(layer1),
            "s_write_per_conductance": float(total / ntotal),
            "s_write_weight0_per_conductance": float(layer0 / n0),
            "s_write_weight1_per_conductance": float(layer1 / n1),
            "linear_solve_method": solve_method,
        }
        for sigma in PREDICTION_SIGMAS:
            row[f"pred_delta_mse_sigma_{sigma_key(sigma)}"] = 0.5 * sigma * sigma * total
        sample_rows.append(row)

    arrays_path = out_dir / "sharpness_arrays.npz"
    sample_metrics_path = out_dir / "sample_sharpness.csv"
    summary_path = out_dir / "summary.json"
    config_path = out_dir / "config.json"

    metadata = {
        "run_name": spec.run_name,
        "seed": spec.seed,
        "voltage_amp": spec.voltage_amp,
        "current_amp": spec.current_amp,
        "checkpoint_path": str(spec.checkpoint_path),
        "source_config_path": str(spec.source_config_path),
        "resolved_config_path": str(spec.resolved_config_path),
        "free_dims": free_dims,
        "num_conductances_weight0": n0,
        "num_conductances_weight1": n1,
        "inference_iterations": inference_iterations,
        "metric": "S_write = ||P H_phys^{-1} A_phi||_F^2",
        "prediction": "E[Delta MSE] ~= 0.5 * sigma^2 * S_write",
    }
    np.savez_compressed(
        arrays_path,
        s_write=s_write,
        s_write_weight0=s_write0,
        s_write_weight1=s_write1,
        hardsigmoid_active_mask=active_mask,
        hidden_hardsigmoid_conductance=hidden_conductance.astype(np.float32),
        labels=samples["labels"],
        predictions=samples["predictions"],
        correct=samples["correct"],
        losses=samples["loss"],
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    hess_eval.write_csv(sample_metrics_path, sample_rows, SAMPLE_COLUMNS)

    config_payload = {
        "input_root": str(args.input_root),
        "output_root": str(output_root),
        "checkpoint": args.checkpoint,
        "device": str(device),
        "batch_size": args.batch_size,
        "max_test_samples": args.max_test_samples,
        "dataset_root_override": args.dataset_root,
        "inference_iterations": inference_iterations,
        "ridge": args.ridge,
        "pinv_rcond": args.pinv_rcond,
        "dataset_params": context["dataset_params"],
        "source_preprocessing": context["resolved_config"].get("dataset", {}).get("preprocessing", {}),
        "formula": metadata["metric"],
    }
    config_path.write_text(json.dumps(hess_eval._json_sanitize(config_payload), indent=2, sort_keys=True))

    summary = {
        "run_name": spec.run_name,
        "seed": spec.seed,
        "voltage_amp": spec.voltage_amp,
        "current_amp": spec.current_amp,
        "checkpoint_path": str(spec.checkpoint_path),
        "source_config_path": str(spec.source_config_path),
        "resolved_config_path": str(spec.resolved_config_path),
        "sample_count": int(samples["hidden"].shape[0]),
        "test_accuracy_recomputed": float(np.mean(samples["correct"])),
        "test_loss_recomputed": float(np.mean(samples["loss"])),
        "num_conductances_weight0": n0,
        "num_conductances_weight1": n1,
        "num_conductances_total": ntotal,
        "unique_hardsigmoid_conductance_masks": int(unique_conductance.shape[0]),
        "inference_iterations": inference_iterations,
        **hess_eval.stats(s_write, "s_write"),
        **hess_eval.stats(s_write0, "s_write_weight0"),
        **hess_eval.stats(s_write1, "s_write_weight1"),
        **hess_eval.stats(s_write / ntotal, "s_write_per_conductance"),
        **hess_eval.stats(active_fraction, "hardsigmoid_active_hidden_fraction"),
        "sample_metrics_path": str(sample_metrics_path),
        "arrays_path": str(arrays_path),
        "summary_path": str(summary_path),
    }
    for sigma in PREDICTION_SIGMAS:
        summary.update(hess_eval.stats(0.5 * sigma * sigma * s_write, f"pred_delta_mse_sigma_{sigma_key(sigma)}"))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"[sharpness] {spec.run_name} seed={spec.seed} samples={summary['sample_count']} "
        f"S_write_mean={summary['s_write_mean']:.6g} acc={summary['test_accuracy_recomputed']:.4f} "
        f"solve_methods={sorted(set(solve_methods))}"
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

        def values(key: str) -> np.ndarray:
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
            **hess_eval.stats(values("s_write"), "s_write"),
            **hess_eval.stats(values("s_write_weight0"), "s_write_weight0"),
            **hess_eval.stats(values("s_write_weight1"), "s_write_weight1"),
            **hess_eval.stats(values("s_write_per_conductance"), "s_write_per_conductance"),
            **hess_eval.stats(values("pred_delta_mse_sigma_0p2"), "pred_delta_mse_sigma_0p2"),
            **hess_eval.stats(values("pred_delta_mse_sigma_0p5"), "pred_delta_mse_sigma_0p5"),
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
        "s_write",
        "Physical-cost sharpness S_write",
        output_root / "s_write_by_amp.png",
        log_y=True,
    )
    hess_eval.make_by_amp_plot(
        by_amp,
        "s_write_weight0",
        "S_write contribution from DenseWeight_0",
        output_root / "s_write_weight0_by_amp.png",
        log_y=True,
    )
    hess_eval.make_by_amp_plot(
        by_amp,
        "s_write_weight1",
        "S_write contribution from DenseWeight_1",
        output_root / "s_write_weight1_by_amp.png",
        log_y=True,
    )
    hess_eval.make_by_amp_plot(
        by_amp,
        "pred_delta_mse_sigma_0p5",
        "Predicted MSE increase at sigma=0.5",
        output_root / "pred_delta_mse_sigma_0p5_by_amp.png",
        log_y=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--run-name", action="append", choices=RUN_ORDER)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--inference-iterations", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--pinv-rcond", type=float, default=1e-10)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--no-summary-after-run", action="store_true")
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
        summary_rows = collect_run_summaries(output_root, specs)
        write_top_level_outputs(output_root, summary_rows)
        print(f"[summary] wrote {output_root / 'summary.csv'}")
        print(f"[summary] wrote {output_root / 'summary_by_amp.csv'}")
        return

    print(f"[sharpness] input_root={input_root}")
    print(f"[sharpness] output_root={output_root}")
    print(f"[sharpness] selected_runs={len(specs)} device={device}")
    for spec in specs:
        analyze_run(spec, args, output_root=output_root, device=device)

    if args.no_summary_after_run:
        print("[done] per-run outputs written; top-level summary skipped by request")
        return
    summary_rows = collect_run_summaries(output_root, specs)
    write_top_level_outputs(output_root, summary_rows)
    print(f"[done] summary={output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
