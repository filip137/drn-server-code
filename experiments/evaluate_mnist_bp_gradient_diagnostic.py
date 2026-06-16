#!/usr/bin/env python3
"""Gradient diagnostics for MNIST DRN amplification checkpoints.

The main quantity for physical multiplicative conductance noise is the
log-conductance gradient, dL/dlog(G) = G * dL/dG.  This script evaluates the
ordinary BP gradient used by training and summarizes both software-gradient and
log-conductance-gradient norms by amplification setting.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from evaluate_mnist_bp_write_noise_sweep import (  # noqa: E402
    DEFAULT_INPUT_ROOT,
    RunRow,
    _build_eval_context,
    _json_sanitize,
    _mean_margin,
    _read_summary_rows,
    _reset_params,
    _scores,
)
from training.sgd import Backprop  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mnist_bp_gradient_diagnostic"
RUN_ORDER = [
    "mnist_bp_amp_v1_c1",
    "mnist_bp_amp_v2_c1",
    "mnist_bp_amp_v4_c1",
    "mnist_bp_amp_v1_c2",
    "mnist_bp_amp_v1_c4",
]

PARAM_COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "batch_index",
    "num_examples",
    "loss",
    "accuracy",
    "mean_margin",
    "kcl_inf_mean",
    "kcl_inf_max",
    "param_index",
    "param_name",
    "param_class",
    "param_numel",
    "param_l2",
    "param_abs_mean",
    "grad_l2",
    "grad_abs_mean",
    "grad_rms",
    "grad_abs_max",
    "relative_grad_l2",
    "logg_grad_l2",
    "logg_grad_abs_mean",
    "logg_grad_rms",
    "logg_grad_abs_max",
]

MODEL_COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "num_batches",
    "num_examples",
    "loss_mean",
    "accuracy_mean",
    "mean_margin_mean",
    "kcl_inf_mean",
    "kcl_inf_max",
    "grad_l2_total_mean",
    "grad_l2_weight_mean",
    "grad_l2_bias_mean",
    "logg_grad_l2_weight_mean",
    "relative_logg_grad_l2_weight_mean",
    "weight0_grad_l2_mean",
    "weight1_grad_l2_mean",
    "weight0_logg_grad_l2_mean",
    "weight1_logg_grad_l2_mean",
    "weight1_over_weight0_logg_grad_l2_mean",
]

AMP_COLUMNS = [
    "run_name",
    "voltage_amp",
    "current_amp",
    "num_models",
    "num_batches",
    "loss_mean",
    "accuracy_mean",
    "grad_l2_total_mean",
    "grad_l2_weight_mean",
    "logg_grad_l2_weight_mean",
    "relative_logg_grad_l2_weight_mean",
    "weight0_logg_grad_l2_mean",
    "weight1_logg_grad_l2_mean",
    "weight1_over_weight0_logg_grad_l2_mean",
]


def _parse_csv_set(value: str | None, cast=str) -> set | None:
    if value is None or value.strip() == "":
        return None
    return {cast(part.strip()) for part in value.split(",") if part.strip()}


def _write_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _json_sanitize(row.get(column, "")) for column in columns})


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return float(arr.mean()) if arr.size else math.nan


def _safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or den == 0.0:
        return math.nan
    return float(num / den)


def _kcl_stats(context: dict) -> tuple[float, float]:
    energy_fn = context["energy_fn"]
    free_layers = context["network"].free_layers()
    values = []
    for layer in free_layers:
        residual = energy_fn.grad_layer_fn(layer)().detach()
        values.append(residual.flatten(start_dim=1).abs().amax(dim=1))
    if not values:
        return math.nan, math.nan
    merged = torch.cat(values)
    return float(merged.mean().item()), float(merged.max().item())


def _param_gradient_rows(
    run: RunRow,
    context: dict,
    *,
    max_eval_batches: int | None,
) -> tuple[list[dict], list[dict]]:
    network = context["network"]
    minimizer = context["minimizer"]
    cost_fn = context["cost_fn"]
    output_layer = context["output_layer"]
    test_loader = context["test_loader"]
    device = context["base_states"][0].device
    num_classes = context["num_classes"]
    params = context["params"]
    estimator = Backprop(params, network.free_layers(), cost_fn, minimizer)

    param_rows: list[dict] = []
    model_rows: list[dict] = []

    for batch_index, (images, labels) in enumerate(test_loader):
        if max_eval_batches is not None and batch_index >= max_eval_batches:
            break

        _reset_params(context)
        images = images.to(device)
        labels = labels.to(device)
        network.set_input(images, reset=True)
        cost_fn.set_target(labels)

        grads = estimator.compute_gradient()[: len(params)]
        loss = float(cost_fn.eval().mean().item())
        errors = cost_fn.error_fn()
        accuracy = float((~errors).float().mean().item())
        scores = _scores(output_layer.state.detach(), num_classes)
        mean_margin = float(_mean_margin(scores, labels).mean().item())
        kcl_inf_mean, kcl_inf_max = _kcl_stats(context)

        total_grad_sq = 0.0
        weight_grad_sq = 0.0
        bias_grad_sq = 0.0
        weight_logg_sq = 0.0
        weight_param_sq = 0.0
        weight0_grad_l2 = math.nan
        weight1_grad_l2 = math.nan
        weight0_logg_grad_l2 = math.nan
        weight1_logg_grad_l2 = math.nan

        for param_index, (param, grad) in enumerate(zip(params, grads)):
            state = param.state.detach()
            grad = grad.detach()
            name = getattr(param, "name", f"param_{param_index}")
            cls_name = param.__class__.__name__
            is_weight = "Weight" in cls_name or "Weight" in name
            is_bias = "Bias" in cls_name or "Bias" in name

            param_l2 = float(torch.linalg.vector_norm(state).item())
            grad_l2 = float(torch.linalg.vector_norm(grad).item())
            logg_grad = state * grad if is_weight else torch.zeros_like(grad)
            logg_grad_l2 = float(torch.linalg.vector_norm(logg_grad).item()) if is_weight else math.nan
            param_numel = int(state.numel())
            relative_grad_l2 = _safe_ratio(grad_l2, param_l2)

            total_grad_sq += grad_l2 * grad_l2
            if is_weight:
                weight_grad_sq += grad_l2 * grad_l2
                weight_logg_sq += logg_grad_l2 * logg_grad_l2
                weight_param_sq += param_l2 * param_l2
                if name.endswith("_0"):
                    weight0_grad_l2 = grad_l2
                    weight0_logg_grad_l2 = logg_grad_l2
                elif name.endswith("_1"):
                    weight1_grad_l2 = grad_l2
                    weight1_logg_grad_l2 = logg_grad_l2
            elif is_bias:
                bias_grad_sq += grad_l2 * grad_l2

            param_rows.append(
                {
                    "run_name": run.run_name,
                    "seed": run.seed,
                    "voltage_amp": run.voltage_amp,
                    "current_amp": run.current_amp,
                    "batch_index": batch_index,
                    "num_examples": int(images.shape[0]),
                    "loss": loss,
                    "accuracy": accuracy,
                    "mean_margin": mean_margin,
                    "kcl_inf_mean": kcl_inf_mean,
                    "kcl_inf_max": kcl_inf_max,
                    "param_index": param_index,
                    "param_name": name,
                    "param_class": cls_name,
                    "param_numel": param_numel,
                    "param_l2": param_l2,
                    "param_abs_mean": float(state.abs().mean().item()),
                    "grad_l2": grad_l2,
                    "grad_abs_mean": float(grad.abs().mean().item()),
                    "grad_rms": float(torch.sqrt(torch.mean(grad * grad)).item()),
                    "grad_abs_max": float(grad.abs().max().item()),
                    "relative_grad_l2": relative_grad_l2,
                    "logg_grad_l2": logg_grad_l2,
                    "logg_grad_abs_mean": float(logg_grad.abs().mean().item()) if is_weight else math.nan,
                    "logg_grad_rms": float(torch.sqrt(torch.mean(logg_grad * logg_grad)).item()) if is_weight else math.nan,
                    "logg_grad_abs_max": float(logg_grad.abs().max().item()) if is_weight else math.nan,
                }
            )

        model_rows.append(
            {
                "run_name": run.run_name,
                "seed": run.seed,
                "voltage_amp": run.voltage_amp,
                "current_amp": run.current_amp,
                "num_batches": 1,
                "num_examples": int(images.shape[0]),
                "loss_mean": loss,
                "accuracy_mean": accuracy,
                "mean_margin_mean": mean_margin,
                "kcl_inf_mean": kcl_inf_mean,
                "kcl_inf_max": kcl_inf_max,
                "grad_l2_total_mean": math.sqrt(total_grad_sq),
                "grad_l2_weight_mean": math.sqrt(weight_grad_sq),
                "grad_l2_bias_mean": math.sqrt(bias_grad_sq),
                "logg_grad_l2_weight_mean": math.sqrt(weight_logg_sq),
                "relative_logg_grad_l2_weight_mean": _safe_ratio(
                    math.sqrt(weight_logg_sq), math.sqrt(weight_param_sq)
                ),
                "weight0_grad_l2_mean": weight0_grad_l2,
                "weight1_grad_l2_mean": weight1_grad_l2,
                "weight0_logg_grad_l2_mean": weight0_logg_grad_l2,
                "weight1_logg_grad_l2_mean": weight1_logg_grad_l2,
                "weight1_over_weight0_logg_grad_l2_mean": _safe_ratio(
                    weight1_logg_grad_l2, weight0_logg_grad_l2
                ),
            }
        )

    return param_rows, model_rows


def _collapse_model_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["run_name"], int(row["seed"]))].append(row)

    out = []
    for (_, _), items in grouped.items():
        first = items[0]
        collapsed = {
            "run_name": first["run_name"],
            "seed": first["seed"],
            "voltage_amp": first["voltage_amp"],
            "current_amp": first["current_amp"],
            "num_batches": len(items),
            "num_examples": sum(int(item["num_examples"]) for item in items),
        }
        for column in MODEL_COLUMNS:
            if column in collapsed:
                continue
            collapsed[column] = _finite_mean([float(item.get(column, math.nan)) for item in items])
        out.append(collapsed)
    out.sort(key=lambda row: (RUN_ORDER.index(row["run_name"]) if row["run_name"] in RUN_ORDER else 999, row["seed"]))
    return out


def _summary_by_amp(model_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in model_rows:
        grouped[row["run_name"]].append(row)

    out = []
    for run_name, items in grouped.items():
        first = items[0]
        row = {
            "run_name": run_name,
            "voltage_amp": first["voltage_amp"],
            "current_amp": first["current_amp"],
            "num_models": len({int(item["seed"]) for item in items}),
            "num_batches": sum(int(item["num_batches"]) for item in items),
        }
        for column in AMP_COLUMNS:
            if column in row:
                continue
            row[column] = _finite_mean([float(item.get(column, math.nan)) for item in items])
        out.append(row)
    out.sort(key=lambda row: RUN_ORDER.index(row["run_name"]) if row["run_name"] in RUN_ORDER else 999)
    return out


def _plot_amp_summary(output_root: Path, rows: list[dict]) -> None:
    if not rows:
        return
    labels = [row["run_name"].replace("mnist_bp_amp_", "").replace("_", "/") for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    axes[0].plot(x, [float(row["grad_l2_weight_mean"]) for row in rows], marker="o", label="software W")
    axes[0].plot(x, [float(row["logg_grad_l2_weight_mean"]) for row in rows], marker="o", label="log-G W")
    axes[0].set_title("Gradient Norms")
    axes[0].set_ylabel("L2 norm")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].legend(frameon=False)

    axes[1].plot(x, [float(row["weight0_logg_grad_l2_mean"]) for row in rows], marker="o", label="W0 input-hidden")
    axes[1].plot(x, [float(row["weight1_logg_grad_l2_mean"]) for row in rows], marker="o", label="W1 hidden-output")
    axes[1].set_title("Log-G Gradient By Layer")
    axes[1].set_ylabel("L2 norm")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].legend(frameon=False)

    fig.savefig(output_root / "gradient_norms_by_amp.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    parser.add_argument("--run-names", default=",".join(RUN_ORDER))
    parser.add_argument("--training-seeds", default="0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-eval-batches", type=int, default=2)
    parser.add_argument("--inference-iterations", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_names = _parse_csv_set(args.run_names, str)
    seeds = _parse_csv_set(args.training_seeds, int)
    runs = _read_summary_rows(
        args.input_root,
        checkpoint_kind=args.checkpoint,
        run_names=run_names,
        training_seeds=seeds,
    )
    runs.sort(key=lambda run: (RUN_ORDER.index(run.run_name) if run.run_name in RUN_ORDER else 999, run.seed))

    all_param_rows: list[dict] = []
    all_model_batch_rows: list[dict] = []
    for run in runs:
        print(f"[grad] {run.run_name} seed={run.seed}")
        context = _build_eval_context(
            run,
            device=device,
            eval_batch_size=args.eval_batch_size,
            no_download=args.no_download,
            inference_iterations_override=args.inference_iterations,
        )
        param_rows, model_rows = _param_gradient_rows(
            run,
            context,
            max_eval_batches=args.max_eval_batches,
        )
        all_param_rows.extend(param_rows)
        all_model_batch_rows.extend(model_rows)

    model_summary = _collapse_model_rows(all_model_batch_rows)
    amp_summary = _summary_by_amp(model_summary)

    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_root / "param_gradients.csv", PARAM_COLUMNS, all_param_rows)
    _write_rows(args.output_root / "model_summary.csv", MODEL_COLUMNS, model_summary)
    _write_rows(args.output_root / "summary_by_amp.csv", AMP_COLUMNS, amp_summary)
    _plot_amp_summary(args.output_root, amp_summary)
    print(f"[done] wrote {args.output_root}")


if __name__ == "__main__":
    main()
