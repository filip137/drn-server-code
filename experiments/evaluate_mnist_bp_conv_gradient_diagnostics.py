#!/usr/bin/env python3
"""Evaluate BP gradient diagnostics for conv MNIST amplification sweeps."""

from __future__ import annotations

import argparse
import csv
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
    _build_eval_context,
    _json_sanitize,
    _read_summary_rows,
    _reset_params,
    _select_shard,
    _write_json,
)
from training.sgd import Backprop


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mnist_bp_conv_gradient_diagnostics"

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

RAW_COLUMNS = [
    "model_label",
    "split",
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "batch_index",
    "param_index",
    "param_name",
    "param_class",
    "loss",
    "accuracy",
    "param_l2",
    "param_abs_mean",
    "grad_l2",
    "grad_abs_mean",
    "grad_abs_max",
    "log_weight_grad_l2",
    "relative_grad_l2",
    "lr",
    "relative_step_l2",
    "log_weight_step_l2",
]
SUMMARY_COLUMNS = [
    "model_label",
    "split",
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "param_name",
    "param_class",
    "num_batches",
    "loss_mean",
    "accuracy_mean",
    "param_l2_mean",
    "grad_l2_mean",
    "grad_l2_p50",
    "log_weight_grad_l2_mean",
    "log_weight_grad_l2_p50",
    "relative_grad_l2_mean",
    "relative_step_l2_mean",
    "log_weight_step_l2_mean",
]
AMP_COLUMNS = [
    "model_label",
    "split",
    "non_linearity",
    "run_name",
    "voltage_amp",
    "current_amp",
    "param_name",
    "param_class",
    "num_models",
    "num_batches",
    "loss_mean",
    "accuracy_mean",
    "param_l2_mean",
    "grad_l2_mean",
    "log_weight_grad_l2_mean",
    "relative_grad_l2_mean",
    "relative_step_l2_mean",
    "log_weight_step_l2_mean",
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


def _finite(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _stats(values: list[float]) -> dict[str, float]:
    arr = _finite(values)
    if arr.size == 0:
        return {"mean": math.nan, "p50": math.nan}
    return {"mean": float(np.mean(arr)), "p50": float(np.percentile(arr, 50))}


def _param_name(param: object, index: int) -> str:
    return str(getattr(param, "name", f"param_{index}")).strip()


def _learning_rates(context: dict) -> list[float]:
    lr = context["config"].get("lr", [])
    values = [float(value) for value in lr]
    if not values:
        return [math.nan] * len(context["params"])
    if len(values) < len(context["params"]):
        values.extend([values[-1]] * (len(context["params"]) - len(values)))
    return values[: len(context["params"])]


def _batch_metrics(context: dict, images: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, list[torch.Tensor]]:
    network = context["network"]
    cost_fn = context["cost_fn"]
    estimator = Backprop(context["params"], context["free_layers"], cost_fn, context["minimizer"])

    network.set_input(images, reset=True)
    cost_fn.set_target(labels)
    grads = estimator.compute_gradient()[: len(context["params"])]
    with torch.no_grad():
        loss = float(cost_fn.eval().mean().item())
        errors = cost_fn.error_fn()
        accuracy = float((~errors).float().mean().item())
    return loss, accuracy, grads


def _run_single(run, *, args: argparse.Namespace, device: torch.device, raw_path: Path) -> None:
    context = _build_eval_context(
        run,
        device=device,
        eval_batch_size=args.batch_size,
        no_download=args.no_download,
        inference_iterations_override=args.inference_iterations,
    )
    loader = context["train_loader"] if args.split == "train" else context["test_loader"]
    lrs = _learning_rates(context)
    raw_rows: list[dict] = []
    for batch_index, (images, labels) in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        _reset_params(context)
        images = images.to(device)
        labels = labels.to(device)
        loss, accuracy, grads = _batch_metrics(context, images, labels)
        for index, (param, grad) in enumerate(zip(context["params"], grads)):
            if args.weights_only and "Weight" not in param.__class__.__name__:
                continue
            state = param.state.detach()
            grad = grad.detach()
            param_l2 = float(torch.linalg.vector_norm(state).item())
            grad_l2 = float(torch.linalg.vector_norm(grad).item())
            log_grad = state * grad
            log_grad_l2 = float(torch.linalg.vector_norm(log_grad).item())
            lr = float(lrs[index])
            raw_rows.append(
                {
                    "model_label": args.model_label,
                    "split": args.split,
                    "non_linearity": run.non_linearity,
                    "run_name": run.run_name,
                    "seed": run.seed,
                    "voltage_amp": run.voltage_amp,
                    "current_amp": run.current_amp,
                    "batch_index": batch_index,
                    "param_index": index,
                    "param_name": _param_name(param, index),
                    "param_class": param.__class__.__name__,
                    "loss": loss,
                    "accuracy": accuracy,
                    "param_l2": param_l2,
                    "param_abs_mean": float(torch.mean(torch.abs(state)).item()),
                    "grad_l2": grad_l2,
                    "grad_abs_mean": float(torch.mean(torch.abs(grad)).item()),
                    "grad_abs_max": float(torch.max(torch.abs(grad)).item()),
                    "log_weight_grad_l2": log_grad_l2,
                    "relative_grad_l2": grad_l2 / param_l2 if param_l2 > 0 else math.nan,
                    "lr": lr,
                    "relative_step_l2": lr * grad_l2 / param_l2 if param_l2 > 0 else math.nan,
                    "log_weight_step_l2": lr * log_grad_l2 if math.isfinite(lr) else math.nan,
                }
            )
        if len(raw_rows) >= 64:
            _write_rows(raw_path, RAW_COLUMNS, raw_rows)
            raw_rows = []
    if raw_rows:
        _write_rows(raw_path, RAW_COLUMNS, raw_rows)


def _read_raw_rows(output_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(output_root.glob("raw_gradients*.csv")):
        rows.extend(csv.DictReader(path.open()))
    return rows


def write_summaries_and_plots(output_root: Path) -> None:
    rows = _read_raw_rows(output_root)
    if not rows:
        return
    grouped: dict[tuple[str, str, str, str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("model_label", ""),
                row["split"],
                row.get("non_linearity", ""),
                row["run_name"],
                int(row["seed"]),
                row["param_name"],
            )
        ].append(row)
    summary_rows: list[dict] = []
    for (model_label, split, non_linearity, run_name, seed, param_name), values in sorted(grouped.items()):
        first = values[0]
        loss = _stats([float(row["loss"]) for row in values])
        acc = _stats([float(row["accuracy"]) for row in values])
        param_l2 = _stats([float(row["param_l2"]) for row in values])
        grad_l2 = _stats([float(row["grad_l2"]) for row in values])
        log_grad = _stats([float(row["log_weight_grad_l2"]) for row in values])
        rel_grad = _stats([float(row["relative_grad_l2"]) for row in values])
        rel_step = _stats([float(row["relative_step_l2"]) for row in values])
        log_step = _stats([float(row["log_weight_step_l2"]) for row in values])
        summary_rows.append(
            {
                "model_label": model_label,
                "split": split,
                "non_linearity": non_linearity,
                "run_name": run_name,
                "seed": seed,
                "voltage_amp": first["voltage_amp"],
                "current_amp": first["current_amp"],
                "param_name": param_name,
                "param_class": first["param_class"],
                "num_batches": len(values),
                "loss_mean": loss["mean"],
                "accuracy_mean": acc["mean"],
                "param_l2_mean": param_l2["mean"],
                "grad_l2_mean": grad_l2["mean"],
                "grad_l2_p50": grad_l2["p50"],
                "log_weight_grad_l2_mean": log_grad["mean"],
                "log_weight_grad_l2_p50": log_grad["p50"],
                "relative_grad_l2_mean": rel_grad["mean"],
                "relative_step_l2_mean": rel_step["mean"],
                "log_weight_step_l2_mean": log_step["mean"],
            }
        )
    _write_rows(output_root / "summary_by_model_param.csv", SUMMARY_COLUMNS, summary_rows, append=False)

    amp_grouped: dict[tuple[str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in summary_rows:
        amp_grouped[
            (
                row["model_label"],
                row["split"],
                row["non_linearity"],
                row["run_name"],
                row["param_name"],
            )
        ].append(row)
    amp_rows: list[dict] = []
    for (model_label, split, non_linearity, run_name, param_name), values in sorted(amp_grouped.items()):
        first = values[0]
        amp_rows.append(
            {
                "model_label": model_label,
                "split": split,
                "non_linearity": non_linearity,
                "run_name": run_name,
                "voltage_amp": first["voltage_amp"],
                "current_amp": first["current_amp"],
                "param_name": param_name,
                "param_class": first["param_class"],
                "num_models": len(values),
                "num_batches": sum(int(row["num_batches"]) for row in values),
                "loss_mean": _stats([float(row["loss_mean"]) for row in values])["mean"],
                "accuracy_mean": _stats([float(row["accuracy_mean"]) for row in values])["mean"],
                "param_l2_mean": _stats([float(row["param_l2_mean"]) for row in values])["mean"],
                "grad_l2_mean": _stats([float(row["grad_l2_mean"]) for row in values])["mean"],
                "log_weight_grad_l2_mean": _stats([float(row["log_weight_grad_l2_mean"]) for row in values])["mean"],
                "relative_grad_l2_mean": _stats([float(row["relative_grad_l2_mean"]) for row in values])["mean"],
                "relative_step_l2_mean": _stats([float(row["relative_step_l2_mean"]) for row in values])["mean"],
                "log_weight_step_l2_mean": _stats([float(row["log_weight_step_l2_mean"]) for row in values])["mean"],
            }
        )
    _write_rows(output_root / "summary_by_amp_param.csv", AMP_COLUMNS, amp_rows, append=False)
    _plot_metric(output_root / "log_weight_grad_l2_by_amp.png", amp_rows, "log_weight_grad_l2_mean", "mean ||theta * grad||")
    _plot_metric(output_root / "relative_step_l2_by_amp.png", amp_rows, "relative_step_l2_mean", "mean lr ||grad|| / ||theta||")


def _plot_metric(path: Path, rows: list[dict], key: str, ylabel: str) -> None:
    all_rows = [row for row in rows if "Weight" in row["param_class"]]
    if not all_rows:
        return
    param_names = sorted({row["param_name"] for row in all_rows})
    run_keys = sorted(
        {(row["non_linearity"], row["run_name"]) for row in all_rows},
        key=lambda item: (item[0], RUN_ORDER.index(item[1]) if item[1] in RUN_ORDER else 999),
    )
    fig, ax = plt.subplots(figsize=(max(8.0, 0.8 * len(run_keys)), 4.5), constrained_layout=True)
    x = np.arange(len(run_keys))
    width = 0.8 / max(len(param_names), 1)
    for idx, param_name in enumerate(param_names):
        values = []
        for non_linearity, run_name in run_keys:
            match = [
                row
                for row in all_rows
                if row["non_linearity"] == non_linearity
                and row["run_name"] == run_name
                and row["param_name"] == param_name
            ]
            values.append(float(match[0][key]) if match else math.nan)
        ax.bar(x + (idx - (len(param_names) - 1) / 2) * width, values, width=width, label=param_name)
    labels = []
    for non_linearity, run_name in run_keys:
        prefix = {"hard_sigmoid": "hs", "perfect_diode": "pd"}.get(non_linearity, non_linearity)
        labels.append(f"{prefix}\n{RUN_LABELS.get(run_name, run_name)}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-label", default="")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-kind", choices=("best", "final"), default="best")
    parser.add_argument("--run-name", action="append")
    parser.add_argument("--training-seeds", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--inference-iterations", type=int)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--raw-results-name", default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        write_summaries_and_plots(output_root)
        print(f"[summary] wrote gradient diagnostics under {output_root}")
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
    raw_name = args.raw_results_name or f"raw_gradients_shard_{args.shard_index}.csv"
    raw_path = output_root / raw_name
    _write_json(
        output_root / f"config_shard_{args.shard_index}.json",
        {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "model_label": args.model_label,
            "split": args.split,
            "checkpoint_kind": args.checkpoint_kind,
            "device": str(device),
            "batch_size": args.batch_size,
            "inference_iterations": args.inference_iterations,
            "max_batches": args.max_batches,
            "weights_only": args.weights_only,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "selected_runs": [
                f"{row.non_linearity}/{row.run_name}/seed_{row.seed}" for row in selected
            ],
            "gradient_coordinate": "raw_grad_and_log_weight_grad_theta_times_grad",
        },
    )
    print(f"[gradients] input_root={input_root}")
    print(f"[gradients] output_root={output_root}")
    print(f"[gradients] selected_runs={len(selected)}/{len(rows)} shard={args.shard_index}/{args.num_shards}")
    if args.dry_run:
        for row in selected:
            print(f"[dry-run] {row.non_linearity} {row.run_name} seed={row.seed} checkpoint={row.checkpoint_path}")
        return
    for run in selected:
        print(f"[run] {run.non_linearity} {run.run_name} seed={run.seed} checkpoint={run.checkpoint_path}")
        _run_single(run, args=args, device=device, raw_path=raw_path)
    write_summaries_and_plots(output_root)
    print(f"[done] output={output_root}")


if __name__ == "__main__":
    main()
