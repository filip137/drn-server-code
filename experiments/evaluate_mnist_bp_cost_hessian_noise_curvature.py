#!/usr/bin/env python3
"""Estimate MNIST BP test-cost Hessian curvature along write-noise directions.

This evaluates finite-difference quadratic forms of the test loss,

    q(d) = (L(theta + eps d) - 2 L(theta) + L(theta - eps d)) / eps**2

where d = theta * xi for Weight parameters. Averaging q(d) over Gaussian xi
estimates Tr(H diag(theta**2)), the local second-order loss sensitivity to
multiplicative lognormal conductance/write noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_mnist_bp_write_noise_sweep import (
    DEFAULT_INPUT_ROOT,
    RunRow,
    _build_eval_context,
    _evaluate,
    _json_sanitize,
    _read_summary_rows,
    _reset_params,
    _select_shard,
    _write_json,
)
from training.sgd import Backprop


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mnist_bp_cost_hessian_noise_curvature"
DEFAULT_NOISE_ROOT = (
    REPO_ROOT
    / "results"
    / "mnist_bp_write_noise_sweep_hardsigmoid_voff15_sigma_response_iter16_30x"
)

RUN_ORDER = [
    "mnist_bp_amp_v1_c1",
    "mnist_bp_amp_v2_c1",
    "mnist_bp_amp_v4_c1",
    "mnist_bp_amp_v1_c2",
    "mnist_bp_amp_v1_c4",
]
RUN_LABELS = {
    "mnist_bp_amp_v1_c1": "v1/c1",
    "mnist_bp_amp_v2_c1": "v2/c1",
    "mnist_bp_amp_v4_c1": "v4/c1",
    "mnist_bp_amp_v1_c2": "v1/c2",
    "mnist_bp_amp_v1_c4": "v1/c4",
}
RUN_COLORS = {
    "mnist_bp_amp_v1_c1": "#4c78a8",
    "mnist_bp_amp_v2_c1": "#f58518",
    "mnist_bp_amp_v4_c1": "#e45756",
    "mnist_bp_amp_v1_c2": "#54a24b",
    "mnist_bp_amp_v1_c4": "#b279a2",
}

RAW_COLUMNS = [
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "checkpoint_path",
    "epsilon",
    "direction_index",
    "direction_seed",
    "target",
    "clean_loss",
    "clean_accuracy",
    "loss_plus",
    "loss_minus",
    "accuracy_plus",
    "accuracy_minus",
    "directional_curvature",
    "direction_l2_sq",
    "normalized_curvature",
    "predicted_loss_increase_sigma_0p1",
    "predicted_loss_increase_sigma_0p2",
]
SUMMARY_COLUMNS = [
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "epsilon",
    "target",
    "num_directions",
    "clean_loss",
    "clean_accuracy",
    "directional_curvature_mean",
    "directional_curvature_std",
    "directional_curvature_p10",
    "directional_curvature_p50",
    "directional_curvature_p90",
    "normalized_curvature_mean",
    "direction_l2_sq_mean",
    "predicted_loss_increase_sigma_0p1",
    "predicted_loss_increase_sigma_0p2",
    "noise_sigma_1pct",
    "noise_sigma_5pct",
    "noise_sigma_10pct",
    "noise_rauc_accuracy",
    "noise_accuracy_sigma_0p2",
    "noise_accuracy_sigma_0p5",
    "noise_accuracy_sigma_1p0",
]
AMP_COLUMNS = [
    "non_linearity",
    "run_name",
    "voltage_amp",
    "current_amp",
    "epsilon",
    "target",
    "num_models",
    "directional_curvature_mean",
    "directional_curvature_std_across_models",
    "directional_curvature_p50_across_models",
    "normalized_curvature_mean",
    "predicted_loss_increase_sigma_0p1",
    "predicted_loss_increase_sigma_0p2",
    "noise_sigma_1pct_mean",
    "noise_sigma_5pct_mean",
    "noise_sigma_10pct_mean",
    "noise_rauc_accuracy_mean",
    "noise_accuracy_sigma_1p0_mean",
]
GRADIENT_RAW_COLUMNS = [
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "checkpoint_path",
    "epsilon",
    "target",
    "clean_loss",
    "clean_accuracy",
    "loss_plus",
    "loss_minus",
    "accuracy_plus",
    "accuracy_minus",
    "gradient_num_examples",
    "gradient_l2_sq",
    "direction_l2_sq",
    "gradient_directional_curvature",
    "gradient_hessian_quadratic",
    "rho_gradient",
]
GRADIENT_SUMMARY_COLUMNS = [
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "epsilon",
    "target",
    "num_rows",
    "clean_loss",
    "clean_accuracy",
    "gradient_num_examples",
    "gradient_l2_sq_mean",
    "gradient_hessian_quadratic_mean",
    "gradient_directional_curvature_mean",
    "gradient_directional_curvature_p50",
    "rho_gradient_mean",
    "rho_gradient_p50",
]
GRADIENT_AMP_COLUMNS = [
    "non_linearity",
    "run_name",
    "voltage_amp",
    "current_amp",
    "epsilon",
    "target",
    "num_models",
    "gradient_l2_sq_mean",
    "gradient_hessian_quadratic_mean",
    "gradient_directional_curvature_mean",
    "gradient_directional_curvature_p50_across_models",
    "rho_gradient_mean",
    "rho_gradient_p50_across_models",
]


def _write_rows(path: Path, columns: list[str], rows: list[dict], *, append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and append
    mode = "a" if append else "w"
    with path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: _json_sanitize(row.get(column, "")) for column in columns})


def _finite_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": math.nan, "std": math.nan, "p10": math.nan, "p50": math.nan, "p90": math.nan}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _param_name(param: object, index: int) -> str:
    return str(getattr(param, "name", f"param_{index}"))


def _selected_param_indices(context: dict, *, include_biases: bool) -> list[int]:
    if include_biases:
        return [
            index
            for index, param in enumerate(context["params"])
            if "Weight" in param.__class__.__name__ or "Bias" in param.__class__.__name__
        ]
    return list(context["noisable_indices"])


def _make_direction(
    context: dict,
    *,
    generator: torch.Generator,
    selected_indices: set[int],
    target_index: int | None,
) -> tuple[list[torch.Tensor | None], float]:
    directions: list[torch.Tensor | None] = []
    l2_sq = 0.0
    for index, base in enumerate(context["base_states"]):
        if index not in selected_indices or (target_index is not None and index != target_index):
            directions.append(None)
            continue
        xi = torch.randn(base.shape, generator=generator, device=base.device, dtype=base.dtype)
        direction = base * xi
        directions.append(direction)
        l2_sq += float(torch.sum(direction.detach() ** 2).item())
    return directions, l2_sq


def _apply_direction(
    context: dict,
    directions: list[torch.Tensor | None],
    *,
    epsilon: float,
    sign: float,
    clamp_perturbed: bool,
) -> None:
    for param, base, direction in zip(context["params"], context["base_states"], directions):
        if direction is None:
            param.state = base.detach().clone()
            continue
        param.state = (base + sign * float(epsilon) * direction).detach().clone()
        if clamp_perturbed:
            param.clamp_()


def _mean_cost_gradient(
    context: dict,
    *,
    selected_indices: list[int],
    max_eval_batches: int | None,
) -> tuple[list[torch.Tensor | None], int]:
    params = context["params"]
    network = context["network"]
    cost_fn = context["cost_fn"]
    estimator = Backprop(params, context["free_layers"], cost_fn, context["minimizer"])
    test_loader = context["test_loader"]
    device = context["base_states"][0].device

    selected_set = set(selected_indices)
    gradient_sums: list[torch.Tensor | None] = [
        torch.zeros_like(base) if index in selected_set else None
        for index, base in enumerate(context["base_states"])
    ]
    total_seen = 0
    for batch_index, (images, labels) in enumerate(test_loader):
        if max_eval_batches is not None and batch_index >= max_eval_batches:
            break
        _reset_params(context)
        images = images.to(device)
        labels = labels.to(device)
        network.set_input(images, reset=True)
        cost_fn.set_target(labels)
        grads = estimator.compute_gradient()[: len(params)]
        batch_size = int(images.size(0))
        for index in selected_indices:
            gradient_sums[index].add_(grads[index].detach(), alpha=batch_size)
        total_seen += batch_size

    if total_seen == 0:
        raise ValueError("Gradient evaluation saw zero examples.")
    for index in selected_indices:
        gradient_sums[index] = gradient_sums[index] / float(total_seen)
    _reset_params(context)
    return gradient_sums, total_seen


def _make_normalized_gradient_direction(
    context: dict,
    gradients: list[torch.Tensor | None],
    *,
    selected_indices: set[int],
    target_index: int | None,
) -> tuple[list[torch.Tensor | None], float, float]:
    directions: list[torch.Tensor | None] = []
    gradient_l2_sq = 0.0
    for index, base in enumerate(context["base_states"]):
        if index not in selected_indices or (target_index is not None and index != target_index):
            directions.append(None)
            continue
        grad = gradients[index]
        if grad is None:
            directions.append(None)
            continue
        direction = grad.detach()
        directions.append(direction)
        gradient_l2_sq += float(torch.sum(direction**2).item())

    if gradient_l2_sq <= 0.0:
        return directions, gradient_l2_sq, 0.0

    scale = 1.0 / math.sqrt(gradient_l2_sq)
    direction_l2_sq = 0.0
    normalized: list[torch.Tensor | None] = []
    for direction in directions:
        if direction is None:
            normalized.append(None)
            continue
        normalized_direction = direction * scale
        normalized.append(normalized_direction)
        direction_l2_sq += float(torch.sum(normalized_direction.detach() ** 2).item())
    return normalized, gradient_l2_sq, direction_l2_sq


def _write_gradient_direction_rows(
    run: RunRow,
    *,
    args: argparse.Namespace,
    context: dict,
    targets: list[tuple[str, int | None]],
    selected_indices: list[int],
    selected_set: set[int],
    clean_metrics: dict,
    raw_path: Path,
) -> None:
    gradients, gradient_num_examples = _mean_cost_gradient(
        context,
        selected_indices=selected_indices,
        max_eval_batches=args.max_eval_batches,
    )
    raw_rows: list[dict] = []
    epsilons = args.gradient_epsilons if args.gradient_epsilons is not None else args.epsilons
    for target_name, target_index in targets:
        directions, gradient_l2_sq, direction_l2_sq = _make_normalized_gradient_direction(
            context,
            gradients,
            selected_indices=selected_set,
            target_index=target_index,
        )
        if gradient_l2_sq <= 0.0 or direction_l2_sq <= 0.0:
            for epsilon in epsilons:
                raw_rows.append(
                    {
                        "non_linearity": run.non_linearity,
                        "run_name": run.run_name,
                        "seed": run.seed,
                        "voltage_amp": run.voltage_amp,
                        "current_amp": run.current_amp,
                        "checkpoint_path": str(run.checkpoint_path),
                        "epsilon": float(epsilon),
                        "target": target_name,
                        "clean_loss": clean_metrics["loss"],
                        "clean_accuracy": clean_metrics["accuracy"],
                        "gradient_num_examples": gradient_num_examples,
                        "gradient_l2_sq": gradient_l2_sq,
                        "direction_l2_sq": direction_l2_sq,
                        "gradient_directional_curvature": math.nan,
                        "gradient_hessian_quadratic": math.nan,
                        "rho_gradient": math.nan,
                    }
                )
            continue
        for epsilon in epsilons:
            _apply_direction(
                context,
                directions,
                epsilon=float(epsilon),
                sign=1.0,
                clamp_perturbed=args.clamp_perturbed,
            )
            plus_metrics = _evaluate(context, max_eval_batches=args.max_eval_batches)
            _apply_direction(
                context,
                directions,
                epsilon=float(epsilon),
                sign=-1.0,
                clamp_perturbed=args.clamp_perturbed,
            )
            minus_metrics = _evaluate(context, max_eval_batches=args.max_eval_batches)
            _reset_params(context)
            curvature = (
                plus_metrics["loss"]
                - 2.0 * clean_metrics["loss"]
                + minus_metrics["loss"]
            ) / (float(epsilon) ** 2)
            gradient_directional_curvature = (
                curvature / direction_l2_sq if direction_l2_sq > 0.0 else math.nan
            )
            gradient_hessian_quadratic = gradient_directional_curvature * gradient_l2_sq
            rho_gradient = (
                gradient_l2_sq / gradient_hessian_quadratic
                if gradient_hessian_quadratic != 0.0
                else math.nan
            )
            raw_rows.append(
                {
                    "non_linearity": run.non_linearity,
                    "run_name": run.run_name,
                    "seed": run.seed,
                    "voltage_amp": run.voltage_amp,
                    "current_amp": run.current_amp,
                    "checkpoint_path": str(run.checkpoint_path),
                    "epsilon": float(epsilon),
                    "target": target_name,
                    "clean_loss": clean_metrics["loss"],
                    "clean_accuracy": clean_metrics["accuracy"],
                    "loss_plus": plus_metrics["loss"],
                    "loss_minus": minus_metrics["loss"],
                    "accuracy_plus": plus_metrics["accuracy"],
                    "accuracy_minus": minus_metrics["accuracy"],
                    "gradient_num_examples": gradient_num_examples,
                    "gradient_l2_sq": gradient_l2_sq,
                    "direction_l2_sq": direction_l2_sq,
                    "gradient_directional_curvature": gradient_directional_curvature,
                    "gradient_hessian_quadratic": gradient_hessian_quadratic,
                    "rho_gradient": rho_gradient,
                }
            )
    if raw_rows:
        _write_rows(raw_path, GRADIENT_RAW_COLUMNS, raw_rows)


def _run_single(
    run: RunRow,
    *,
    args: argparse.Namespace,
    device: torch.device,
    raw_path: Path,
    gradient_raw_path: Path | None,
) -> None:
    context = _build_eval_context(
        run,
        device=device,
        eval_batch_size=args.eval_batch_size,
        no_download=args.no_download,
        inference_iterations_override=args.inference_iterations,
    )
    selected_indices = _selected_param_indices(context, include_biases=args.include_biases)
    selected_set = set(selected_indices)
    targets: list[tuple[str, int | None]] = [("all", None)]
    if args.layerwise:
        targets.extend((_param_name(context["params"][index], index), index) for index in selected_indices)

    _reset_params(context)
    clean_metrics = _evaluate(context, max_eval_batches=args.max_eval_batches)
    if args.include_gradient_direction:
        if gradient_raw_path is None:
            raise ValueError("gradient_raw_path is required when --include-gradient-direction is set.")
        _write_gradient_direction_rows(
            run,
            args=args,
            context=context,
            targets=targets,
            selected_indices=selected_indices,
            selected_set=selected_set,
            clean_metrics=clean_metrics,
            raw_path=gradient_raw_path,
        )
    raw_rows: list[dict] = []
    for direction_index in range(args.num_directions):
        direction_seed = int(args.direction_seed_offset + 100000 * run.seed + direction_index)
        for target_name, target_index in targets:
            generator = torch.Generator(device=str(context["base_states"][0].device))
            generator.manual_seed(direction_seed)
            directions, direction_l2_sq = _make_direction(
                context,
                generator=generator,
                selected_indices=selected_set,
                target_index=target_index,
            )
            for epsilon in args.epsilons:
                _apply_direction(
                    context,
                    directions,
                    epsilon=float(epsilon),
                    sign=1.0,
                    clamp_perturbed=args.clamp_perturbed,
                )
                plus_metrics = _evaluate(context, max_eval_batches=args.max_eval_batches)
                _apply_direction(
                    context,
                    directions,
                    epsilon=float(epsilon),
                    sign=-1.0,
                    clamp_perturbed=args.clamp_perturbed,
                )
                minus_metrics = _evaluate(context, max_eval_batches=args.max_eval_batches)
                _reset_params(context)
                curvature = (
                    plus_metrics["loss"]
                    - 2.0 * clean_metrics["loss"]
                    + minus_metrics["loss"]
                ) / (float(epsilon) ** 2)
                normalized = curvature / direction_l2_sq if direction_l2_sq > 0 else float("nan")
                raw_rows.append(
                    {
                        "non_linearity": run.non_linearity,
                        "run_name": run.run_name,
                        "seed": run.seed,
                        "voltage_amp": run.voltage_amp,
                        "current_amp": run.current_amp,
                        "checkpoint_path": str(run.checkpoint_path),
                        "epsilon": float(epsilon),
                        "direction_index": direction_index,
                        "direction_seed": direction_seed,
                        "target": target_name,
                        "clean_loss": clean_metrics["loss"],
                        "clean_accuracy": clean_metrics["accuracy"],
                        "loss_plus": plus_metrics["loss"],
                        "loss_minus": minus_metrics["loss"],
                        "accuracy_plus": plus_metrics["accuracy"],
                        "accuracy_minus": minus_metrics["accuracy"],
                        "directional_curvature": curvature,
                        "direction_l2_sq": direction_l2_sq,
                        "normalized_curvature": normalized,
                        "predicted_loss_increase_sigma_0p1": 0.5 * (0.1**2) * curvature,
                        "predicted_loss_increase_sigma_0p2": 0.5 * (0.2**2) * curvature,
                    }
                )
                if len(raw_rows) >= 16:
                    _write_rows(raw_path, RAW_COLUMNS, raw_rows)
                    raw_rows = []
    if raw_rows:
        _write_rows(raw_path, RAW_COLUMNS, raw_rows)


def _interpolated_drop_threshold(points: list[tuple[float, float]], threshold: float) -> float:
    points = sorted(points)
    if not points:
        return math.nan
    if points[0][1] >= threshold:
        return float(points[0][0])
    for index in range(1, len(points)):
        x0, y0 = points[index - 1]
        x1, y1 = points[index]
        if y1 >= threshold:
            if y1 == y0:
                return float(x1)
            return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))
    return math.nan


def _noise_metrics(noise_root: Path | None) -> dict[tuple[str, str, int], dict[str, float]]:
    if noise_root is None:
        return {}
    summary_path = noise_root / "summary_by_model_sigma.csv"
    if not summary_path.exists():
        return {}
    rows = list(csv.DictReader(summary_path.open()))
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("non_linearity", ""),
                row["run_name"],
                int(row["training_seed"]),
            )
        ].append(row)
    out: dict[tuple[str, str, int], dict[str, float]] = {}
    for key, values in grouped.items():
        points = [
            (float(row["sigma"]), float(row["mean_accuracy_drop"]))
            for row in values
        ]
        sigmas = np.asarray([float(row["sigma"]) for row in values], dtype=np.float64)
        acc = np.asarray([float(row["mean_noisy_accuracy"]) for row in values], dtype=np.float64)
        order = np.argsort(sigmas)
        sigmas = sigmas[order]
        acc = acc[order]
        by_sigma = {float(row["sigma"]): float(row["mean_noisy_accuracy"]) for row in values}
        out[key] = {
            "noise_sigma_1pct": _interpolated_drop_threshold(points, 0.01),
            "noise_sigma_5pct": _interpolated_drop_threshold(points, 0.05),
            "noise_sigma_10pct": _interpolated_drop_threshold(points, 0.10),
            "noise_rauc_accuracy": float(np.trapz(acc, sigmas)),
            "noise_accuracy_sigma_0p2": by_sigma.get(0.2, math.nan),
            "noise_accuracy_sigma_0p5": by_sigma.get(0.5, math.nan),
            "noise_accuracy_sigma_1p0": by_sigma.get(1.0, math.nan),
        }
    return out


def write_summaries_and_plots(output_root: Path, *, noise_root: Path | None) -> None:
    raw_rows: list[dict] = []
    for path in sorted(output_root.glob("raw_curvature*.csv")):
        raw_rows.extend(csv.DictReader(path.open()))
    if not raw_rows:
        _write_gradient_summaries(output_root)
        return
    noise_by_model = _noise_metrics(noise_root)

    grouped: dict[tuple[str, str, int, str, str], list[dict]] = defaultdict(list)
    for row in raw_rows:
        grouped[
            (
                row.get("non_linearity", ""),
                row["run_name"],
                int(row["seed"]),
                row["epsilon"],
                row["target"],
            )
        ].append(row)

    summary_rows: list[dict] = []
    for (non_linearity, run_name, seed, epsilon, target), values in sorted(grouped.items()):
        curvatures = [float(row["directional_curvature"]) for row in values]
        normalized = [float(row["normalized_curvature"]) for row in values]
        direction_norms = [float(row["direction_l2_sq"]) for row in values]
        stats = _finite_stats(curvatures)
        norm_stats = _finite_stats(normalized)
        direction_stats = _finite_stats(direction_norms)
        noise = noise_by_model.get((non_linearity, run_name, seed), {})
        if not noise:
            noise = noise_by_model.get(("", run_name, seed), {})
        first = values[0]
        summary_rows.append(
            {
                "non_linearity": non_linearity,
                "run_name": run_name,
                "seed": seed,
                "voltage_amp": first["voltage_amp"],
                "current_amp": first["current_amp"],
                "epsilon": float(epsilon),
                "target": target,
                "num_directions": len(values),
                "clean_loss": first["clean_loss"],
                "clean_accuracy": first["clean_accuracy"],
                "directional_curvature_mean": stats["mean"],
                "directional_curvature_std": stats["std"],
                "directional_curvature_p10": stats["p10"],
                "directional_curvature_p50": stats["p50"],
                "directional_curvature_p90": stats["p90"],
                "normalized_curvature_mean": norm_stats["mean"],
                "direction_l2_sq_mean": direction_stats["mean"],
                "predicted_loss_increase_sigma_0p1": 0.5 * (0.1**2) * stats["mean"],
                "predicted_loss_increase_sigma_0p2": 0.5 * (0.2**2) * stats["mean"],
                "noise_sigma_1pct": noise.get("noise_sigma_1pct", math.nan),
                "noise_sigma_5pct": noise.get("noise_sigma_5pct", math.nan),
                "noise_sigma_10pct": noise.get("noise_sigma_10pct", math.nan),
                "noise_rauc_accuracy": noise.get("noise_rauc_accuracy", math.nan),
                "noise_accuracy_sigma_0p2": noise.get("noise_accuracy_sigma_0p2", math.nan),
                "noise_accuracy_sigma_0p5": noise.get("noise_accuracy_sigma_0p5", math.nan),
                "noise_accuracy_sigma_1p0": noise.get("noise_accuracy_sigma_1p0", math.nan),
            }
        )
    _write_rows(output_root / "summary_by_model.csv", SUMMARY_COLUMNS, summary_rows, append=False)

    amp_grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in summary_rows:
        amp_grouped[
            (
                row.get("non_linearity", ""),
                row["run_name"],
                str(row["epsilon"]),
                row["target"],
            )
        ].append(row)

    amp_rows: list[dict] = []
    for (non_linearity, run_name, epsilon, target), values in sorted(amp_grouped.items()):
        curv = [float(row["directional_curvature_mean"]) for row in values]
        norm = [float(row["normalized_curvature_mean"]) for row in values]
        pred01 = [float(row["predicted_loss_increase_sigma_0p1"]) for row in values]
        pred02 = [float(row["predicted_loss_increase_sigma_0p2"]) for row in values]
        first = values[0]
        curv_stats = _finite_stats(curv)
        amp_rows.append(
            {
                "non_linearity": non_linearity,
                "run_name": run_name,
                "voltage_amp": first["voltage_amp"],
                "current_amp": first["current_amp"],
                "epsilon": float(epsilon),
                "target": target,
                "num_models": len(values),
                "directional_curvature_mean": curv_stats["mean"],
                "directional_curvature_std_across_models": curv_stats["std"],
                "directional_curvature_p50_across_models": curv_stats["p50"],
                "normalized_curvature_mean": _finite_stats(norm)["mean"],
                "predicted_loss_increase_sigma_0p1": _finite_stats(pred01)["mean"],
                "predicted_loss_increase_sigma_0p2": _finite_stats(pred02)["mean"],
                "noise_sigma_1pct_mean": _finite_stats([float(row["noise_sigma_1pct"]) for row in values])["mean"],
                "noise_sigma_5pct_mean": _finite_stats([float(row["noise_sigma_5pct"]) for row in values])["mean"],
                "noise_sigma_10pct_mean": _finite_stats([float(row["noise_sigma_10pct"]) for row in values])["mean"],
                "noise_rauc_accuracy_mean": _finite_stats([float(row["noise_rauc_accuracy"]) for row in values])["mean"],
                "noise_accuracy_sigma_1p0_mean": _finite_stats([float(row["noise_accuracy_sigma_1p0"]) for row in values])["mean"],
            }
        )
    _write_rows(output_root / "summary_by_amp.csv", AMP_COLUMNS, amp_rows, append=False)
    _write_correlations(output_root, summary_rows, amp_rows)
    _plot_outputs(output_root, summary_rows, amp_rows)
    _write_gradient_summaries(output_root)


def _write_gradient_summaries(output_root: Path) -> None:
    raw_rows: list[dict] = []
    for path in sorted(output_root.glob("raw_gradient_curvature*.csv")):
        raw_rows.extend(csv.DictReader(path.open()))
    if not raw_rows:
        return

    grouped: dict[tuple[str, str, int, str, str], list[dict]] = defaultdict(list)
    for row in raw_rows:
        grouped[
            (
                row.get("non_linearity", ""),
                row["run_name"],
                int(row["seed"]),
                row["epsilon"],
                row["target"],
            )
        ].append(row)

    summary_rows: list[dict] = []
    for (non_linearity, run_name, seed, epsilon, target), values in sorted(grouped.items()):
        first = values[0]
        grad_l2 = _finite_stats([float(row["gradient_l2_sq"]) for row in values])
        g_h_g = _finite_stats([float(row["gradient_hessian_quadratic"]) for row in values])
        q_grad = _finite_stats([float(row["gradient_directional_curvature"]) for row in values])
        rho_grad = _finite_stats([float(row["rho_gradient"]) for row in values])
        num_examples = _finite_stats([float(row["gradient_num_examples"]) for row in values])
        summary_rows.append(
            {
                "non_linearity": non_linearity,
                "run_name": run_name,
                "seed": seed,
                "voltage_amp": first["voltage_amp"],
                "current_amp": first["current_amp"],
                "epsilon": float(epsilon),
                "target": target,
                "num_rows": len(values),
                "clean_loss": first["clean_loss"],
                "clean_accuracy": first["clean_accuracy"],
                "gradient_num_examples": num_examples["mean"],
                "gradient_l2_sq_mean": grad_l2["mean"],
                "gradient_hessian_quadratic_mean": g_h_g["mean"],
                "gradient_directional_curvature_mean": q_grad["mean"],
                "gradient_directional_curvature_p50": q_grad["p50"],
                "rho_gradient_mean": rho_grad["mean"],
                "rho_gradient_p50": rho_grad["p50"],
            }
        )
    _write_rows(
        output_root / "summary_gradient_by_model.csv",
        GRADIENT_SUMMARY_COLUMNS,
        summary_rows,
        append=False,
    )

    amp_grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in summary_rows:
        amp_grouped[
            (
                row.get("non_linearity", ""),
                row["run_name"],
                str(row["epsilon"]),
                row["target"],
            )
        ].append(row)

    amp_rows: list[dict] = []
    for (non_linearity, run_name, epsilon, target), values in sorted(amp_grouped.items()):
        first = values[0]
        grad_l2 = _finite_stats([float(row["gradient_l2_sq_mean"]) for row in values])
        g_h_g = _finite_stats([float(row["gradient_hessian_quadratic_mean"]) for row in values])
        q_grad = _finite_stats([float(row["gradient_directional_curvature_mean"]) for row in values])
        rho_grad = _finite_stats([float(row["rho_gradient_mean"]) for row in values])
        q_grad_p50 = _finite_stats([float(row["gradient_directional_curvature_p50"]) for row in values])
        rho_grad_p50 = _finite_stats([float(row["rho_gradient_p50"]) for row in values])
        amp_rows.append(
            {
                "non_linearity": non_linearity,
                "run_name": run_name,
                "voltage_amp": first["voltage_amp"],
                "current_amp": first["current_amp"],
                "epsilon": float(epsilon),
                "target": target,
                "num_models": len(values),
                "gradient_l2_sq_mean": grad_l2["mean"],
                "gradient_hessian_quadratic_mean": g_h_g["mean"],
                "gradient_directional_curvature_mean": q_grad["mean"],
                "gradient_directional_curvature_p50_across_models": q_grad_p50["p50"],
                "rho_gradient_mean": rho_grad["mean"],
                "rho_gradient_p50_across_models": rho_grad_p50["p50"],
            }
        )
    _write_rows(
        output_root / "summary_gradient_by_amp.csv",
        GRADIENT_AMP_COLUMNS,
        amp_rows,
        append=False,
    )
    _plot_gradient_metric_by_amp(
        output_root / "gradient_directional_curvature_by_amp.png",
        amp_rows,
        key="gradient_directional_curvature_mean",
        ylabel="g^T H g / g^T g",
    )
    _plot_gradient_metric_by_amp(
        output_root / "rho_gradient_by_amp.png",
        amp_rows,
        key="rho_gradient_mean",
        ylabel="g^T g / g^T H g",
    )


def _pearson(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return math.nan
    x_rank = np.argsort(np.argsort(x)).astype(np.float64)
    y_rank = np.argsort(np.argsort(y)).astype(np.float64)
    return _pearson(x_rank.tolist(), y_rank.tolist())


def _write_correlations(output_root: Path, summary_rows: list[dict], amp_rows: list[dict]) -> None:
    rows: list[dict] = []
    for level, source_rows in (("model", summary_rows), ("amp", amp_rows)):
        filtered = [row for row in source_rows if row["target"] == "all"]
        for epsilon in sorted({float(row["epsilon"]) for row in filtered}):
            values = [row for row in filtered if float(row["epsilon"]) == epsilon]
            x = [float(row["directional_curvature_mean"]) for row in values]
            for metric in (
                "noise_sigma_1pct",
                "noise_sigma_5pct",
                "noise_sigma_10pct",
                "noise_rauc_accuracy",
                "noise_accuracy_sigma_1p0",
            ):
                metric_name = metric if level == "model" else f"{metric}_mean"
                y = [float(row[metric_name]) for row in values if metric_name in row]
                x_for_y = [float(row["directional_curvature_mean"]) for row in values if metric_name in row]
                rows.append(
                    {
                        "level": level,
                        "epsilon": epsilon,
                        "target": "all",
                        "metric": metric_name,
                        "pearson": _pearson(x_for_y, y),
                        "spearman": _spearman(x_for_y, y),
                        "num_points": len(y),
                    }
                )
    columns = ["level", "epsilon", "target", "metric", "pearson", "spearman", "num_points"]
    _write_rows(output_root / "curvature_noise_correlations.csv", columns, rows, append=False)


def _plot_outputs(output_root: Path, summary_rows: list[dict], amp_rows: list[dict]) -> None:
    all_amp = [
        row
        for row in amp_rows
        if row["target"] == "all"
        and float(row["epsilon"]) == min(float(candidate["epsilon"]) for candidate in amp_rows)
    ]
    all_model = [
        row
        for row in summary_rows
        if row["target"] == "all"
        and float(row["epsilon"]) == min(float(candidate["epsilon"]) for candidate in summary_rows)
    ]
    _plot_curvature_by_amp(output_root / "noise_weighted_cost_curvature_by_amp.png", all_amp)
    _plot_layer_curvature(output_root / "layerwise_cost_curvature_by_amp.png", amp_rows)
    _plot_curvature_vs_resilience(
        output_root / "cost_curvature_vs_sigma_1pct.png",
        all_model,
        y_key="noise_sigma_1pct",
        y_label="write-noise sigma at 1% accuracy drop",
    )
    _plot_curvature_vs_resilience(
        output_root / "cost_curvature_vs_accuracy_sigma_1p0.png",
        all_model,
        y_key="noise_accuracy_sigma_1p0",
        y_label="test accuracy at sigma=1.0",
    )


def _plot_curvature_by_amp(path: Path, rows: list[dict]) -> None:
    rows = sorted(
        rows,
        key=lambda row: (
            row.get("non_linearity", ""),
            RUN_ORDER.index(row["run_name"]) if row["run_name"] in RUN_ORDER else 999,
        ),
    )
    fig, ax = plt.subplots(figsize=(max(7.0, 0.75 * len(rows)), 4.2), constrained_layout=True)
    x = np.arange(len(rows))
    y = np.asarray([float(row["directional_curvature_mean"]) for row in rows], dtype=np.float64)
    yerr = np.asarray([float(row["directional_curvature_std_across_models"]) for row in rows], dtype=np.float64)
    colors = [RUN_COLORS.get(row["run_name"], "#666666") for row in rows]
    ax.bar(x, y, yerr=yerr, color=colors, alpha=0.85, capsize=4)
    ax.set_xticks(x)
    multiple_nonlinearities = len({row.get("non_linearity", "") for row in rows}) > 1
    labels = []
    for row in rows:
        amp_label = RUN_LABELS.get(row["run_name"], row["run_name"])
        if multiple_nonlinearities:
            prefix = {"hard_sigmoid": "hs", "perfect_diode": "pd"}.get(
                row.get("non_linearity", ""),
                row.get("non_linearity", ""),
            )
            labels.append(f"{prefix}\n{amp_label}")
        else:
            labels.append(amp_label)
    ax.set_xticklabels(labels)
    ax.set_ylabel("E[d^T H_cost d]")
    ax.set_title("Noise-weighted test-cost curvature")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_layer_curvature(path: Path, rows: list[dict]) -> None:
    epsilon = min(float(row["epsilon"]) for row in rows)
    layer_rows = [
        row
        for row in rows
        if float(row["epsilon"]) == epsilon and row["target"] != "all"
    ]
    targets = sorted({row["target"] for row in layer_rows})
    run_keys = sorted(
        {(row.get("non_linearity", ""), row["run_name"]) for row in layer_rows},
        key=lambda item: (item[0], RUN_ORDER.index(item[1]) if item[1] in RUN_ORDER else 999),
    )
    fig, ax = plt.subplots(figsize=(max(8.0, 0.75 * len(run_keys)), 4.5), constrained_layout=True)
    x = np.arange(len(run_keys))
    width = 0.8 / max(len(targets), 1)
    for idx, target in enumerate(targets):
        values = []
        for non_linearity, run_name in run_keys:
            matches = [
                row
                for row in layer_rows
                if row.get("non_linearity", "") == non_linearity
                and row["run_name"] == run_name
                and row["target"] == target
            ]
            values.append(float(matches[0]["directional_curvature_mean"]) if matches else math.nan)
        ax.bar(x + (idx - (len(targets) - 1) / 2) * width, values, width=width, label=target)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{ {'hard_sigmoid': 'hs', 'perfect_diode': 'pd'}.get(non_linearity, non_linearity) }\n"
            f"{RUN_LABELS.get(run_name, run_name)}"
            for non_linearity, run_name in run_keys
        ]
    )
    ax.set_ylabel("E[d^T H_cost d]")
    ax.set_title("Layerwise noise-weighted curvature")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles, labels, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_curvature_vs_resilience(path: Path, rows: list[dict], *, y_key: str, y_label: str) -> None:
    rows = [
        row
        for row in rows
        if math.isfinite(float(row.get("directional_curvature_mean", math.nan)))
        and math.isfinite(float(row.get(y_key, math.nan)))
    ]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    for row in rows:
        x = float(row["directional_curvature_mean"])
        y = float(row[y_key])
        color = RUN_COLORS.get(row["run_name"], "#666666")
        prefix = {"hard_sigmoid": "hs", "perfect_diode": "pd"}.get(
            row.get("non_linearity", ""),
            row.get("non_linearity", ""),
        )
        label = RUN_LABELS.get(row["run_name"], row["run_name"])
        label = f"{prefix}/{label}" if prefix else label
        ax.scatter(x, y, color=color, s=52)
        ax.annotate(f"{label}/s{row['seed']}", (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    if all(float(row["directional_curvature_mean"]) > 0 for row in rows):
        ax.set_xscale("log")
    ax.set_xlabel("E[d^T H_cost d]")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_gradient_metric_by_amp(path: Path, rows: list[dict], *, key: str, ylabel: str) -> None:
    rows = [
        row
        for row in rows
        if row["target"] == "all" and math.isfinite(float(row.get(key, math.nan)))
    ]
    if not rows:
        return
    min_epsilon = min(float(row["epsilon"]) for row in rows)
    rows = [
        row
        for row in rows
        if float(row["epsilon"]) == min_epsilon
    ]
    rows = sorted(
        rows,
        key=lambda row: (
            row.get("non_linearity", ""),
            RUN_ORDER.index(row["run_name"]) if row["run_name"] in RUN_ORDER else 999,
        ),
    )
    fig, ax = plt.subplots(figsize=(max(7.0, 0.75 * len(rows)), 4.2), constrained_layout=True)
    x = np.arange(len(rows))
    y = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    colors = [RUN_COLORS.get(row["run_name"], "#666666") for row in rows]
    ax.bar(x, y, color=colors, alpha=0.85)
    ax.set_xticks(x)
    multiple_nonlinearities = len({row.get("non_linearity", "") for row in rows}) > 1
    labels = []
    for row in rows:
        amp_label = RUN_LABELS.get(row["run_name"], row["run_name"])
        if multiple_nonlinearities:
            prefix = {"hard_sigmoid": "hs", "perfect_diode": "pd"}.get(
                row.get("non_linearity", ""),
                row.get("non_linearity", ""),
            )
            labels.append(f"{prefix}\n{amp_label}")
        else:
            labels.append(amp_label)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--noise-root", default=str(DEFAULT_NOISE_ROOT))
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-kind", choices=("best", "final"), default="best")
    parser.add_argument("--run-name", action="append")
    parser.add_argument("--training-seeds", type=int, nargs="+")
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--inference-iterations", type=int, default=16)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--num-directions", type=int, default=16)
    parser.add_argument("--direction-seed-offset", type=int, default=12345)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.02])
    parser.add_argument("--include-gradient-direction", action="store_true")
    parser.add_argument("--gradient-epsilons", type=float, nargs="+")
    parser.add_argument("--include-biases", action="store_true")
    parser.add_argument("--layerwise", action="store_true")
    parser.add_argument("--clamp-perturbed", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--raw-results-name", default=None)
    parser.add_argument("--gradient-raw-results-name", default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    noise_root = Path(args.noise_root).expanduser().resolve() if args.noise_root else None
    output_root.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        write_summaries_and_plots(output_root, noise_root=noise_root)
        print(f"[summary] wrote cost-Hessian summaries under {output_root}")
        return

    if args.device is not None:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {args.device!r}, but CUDA is unavailable.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = _read_summary_rows(
        input_root,
        checkpoint_kind=args.checkpoint_kind,
        run_names=set(args.run_name) if args.run_name else None,
        training_seeds=set(args.training_seeds) if args.training_seeds else None,
    )
    selected = _select_shard(rows, args.num_shards, args.shard_index)
    raw_name = args.raw_results_name or f"raw_curvature_shard_{args.shard_index}.csv"
    raw_path = output_root / raw_name
    gradient_raw_name = (
        args.gradient_raw_results_name
        or f"raw_gradient_curvature_shard_{args.shard_index}.csv"
    )
    gradient_raw_path = output_root / gradient_raw_name if args.include_gradient_direction else None
    config_payload = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "noise_root": str(noise_root) if noise_root is not None else None,
        "checkpoint_kind": args.checkpoint_kind,
        "device": str(device),
        "eval_batch_size": args.eval_batch_size,
        "inference_iterations": args.inference_iterations,
        "max_eval_batches": args.max_eval_batches,
        "num_directions": args.num_directions,
        "direction_seed_offset": args.direction_seed_offset,
        "epsilons": [float(value) for value in args.epsilons],
        "include_gradient_direction": bool(args.include_gradient_direction),
        "gradient_epsilons": (
            [float(value) for value in args.gradient_epsilons]
            if args.gradient_epsilons is not None
            else None
        ),
        "gradient_raw_results_name": gradient_raw_name if args.include_gradient_direction else None,
        "include_biases": bool(args.include_biases),
        "layerwise": bool(args.layerwise),
        "clamp_perturbed": bool(args.clamp_perturbed),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selected_runs": [
            f"{row.non_linearity}/{row.run_name}/seed_{row.seed}" for row in selected
        ],
        "estimator": (
            "finite_difference_directional_cost_hessian_for_d=theta*xi"
            "+optional_normalized_gradient_direction"
        ),
    }
    _write_json(output_root / f"config_shard_{args.shard_index}.json", config_payload)

    print(f"[cost-hessian] input_root={input_root}")
    print(f"[cost-hessian] output_root={output_root}")
    print(f"[cost-hessian] selected_runs={len(selected)}/{len(rows)} shard={args.shard_index}/{args.num_shards}")
    if args.dry_run:
        for row in selected:
            print(
                f"[dry-run] {row.non_linearity} {row.run_name} "
                f"seed={row.seed} checkpoint={row.checkpoint_path}"
            )
        return
    for run in selected:
        print(
            f"[run] {run.non_linearity} {run.run_name} "
            f"seed={run.seed} checkpoint={run.checkpoint_path}"
        )
        _run_single(
            run,
            args=args,
            device=device,
            raw_path=raw_path,
            gradient_raw_path=gradient_raw_path,
        )
    write_summaries_and_plots(output_root, noise_root=noise_root)
    print(f"[done] output={output_root}")


if __name__ == "__main__":
    main()
