#!/usr/bin/env python3
"""Compare centered-EP and BP gradients for conv MNIST DRN checkpoints."""

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
    _build_eval_context,
    _json_sanitize,
    _read_summary_rows,
    _reset_params,
    _select_shard,
    _write_json,
)
from labs.mnist_train import _build_tracking_minimizer
from training.sgd import AugmentedFunction, Backprop, EquilibriumProp


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "mnist_bp_conv2_best_lr_50epoch_64_128ch_s2_valid_iter6_seed0"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mnist_bp_conv_ep_bp_cosine_vs_iterations"

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
    "checkpoint_kind",
    "clean_accuracy",
    "iteration_count",
    "beta",
    "batch_index",
    "scope",
    "param_index",
    "param_name",
    "param_class",
    "param_group",
    "num_values",
    "loss",
    "accuracy",
    "bp_grad_l2",
    "ep_grad_l2",
    "grad_dot",
    "cosine",
    "relative_l2_error",
]
SUMMARY_COLUMNS = [
    "model_label",
    "split",
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "checkpoint_kind",
    "iteration_count",
    "beta",
    "scope",
    "param_name",
    "param_class",
    "param_group",
    "num_batches",
    "num_values",
    "loss_mean",
    "accuracy_mean",
    "cosine_mean",
    "cosine_std",
    "cosine_p10",
    "cosine_p50",
    "cosine_p90",
    "bp_grad_l2_mean",
    "ep_grad_l2_mean",
    "relative_l2_error_mean",
]
AMP_COLUMNS = [
    "model_label",
    "split",
    "non_linearity",
    "run_name",
    "voltage_amp",
    "current_amp",
    "checkpoint_kind",
    "iteration_count",
    "beta",
    "scope",
    "param_name",
    "param_class",
    "param_group",
    "num_models",
    "num_batches",
    "num_values",
    "loss_mean",
    "accuracy_mean",
    "cosine_mean",
    "cosine_std",
    "cosine_p10",
    "cosine_p50",
    "cosine_p90",
    "bp_grad_l2_mean",
    "ep_grad_l2_mean",
    "relative_l2_error_mean",
]
SELECTED_COLUMNS = [
    "model_label",
    "split",
    "non_linearity",
    "run_name",
    "voltage_amp",
    "current_amp",
    "checkpoint_kind",
    "beta",
    "selected_iteration_count",
    "selected_cosine_mean",
    "reference_cosine_mean",
    "highest_iteration_count",
    "highest_iteration_cosine_mean",
    "num_models",
    "num_batches",
    "selection_rule",
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
        return {
            "mean": math.nan,
            "std": math.nan,
            "p10": math.nan,
            "p50": math.nan,
            "p90": math.nan,
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _float(row: dict, key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _param_name(param: object, index: int) -> str:
    return str(getattr(param, "name", f"param_{index}")).strip()


def _param_group(param: object) -> str:
    class_name = param.__class__.__name__
    if "Weight" in class_name:
        return "weights"
    if "Bias" in class_name:
        return "biases"
    return "other"


def _detach_state(context: dict) -> None:
    for param in context["params"]:
        param.state = param.state.detach()
        param.state.requires_grad = False
    for layer in context["free_layers"]:
        layer.state = layer.state.detach()


def _make_augmented_minimizer(context: dict, iteration_count: int):
    energy_fn = context["energy_fn"]
    cost_fn = context["cost_fn"]
    augmented_fn = AugmentedFunction(energy_fn, cost_fn)
    minimizer = _build_tracking_minimizer(
        augmented_fn,
        context["free_layers"],
        context["model_cfg"],
        context["config"]["energy_minimizer"]["mode"],
        num_iterations=int(iteration_count),
        voltage_amp=energy_fn._voltage_amp,
        current_amp=energy_fn._current_amp,
    )
    return augmented_fn, minimizer


def _collect_batches(context: dict, split: str, max_batches: int | None) -> list[tuple[torch.Tensor, torch.Tensor]]:
    loader = context["train_loader"] if split == "train" else context["test_loader"]
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batches.append((images.detach().clone(), labels.detach().clone()))
    if not batches:
        raise ValueError(f"No batches collected for split {split!r}.")
    return batches


def _compute_bp_gradient(
    context: dict,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float, list[torch.Tensor]]:
    network = context["network"]
    cost_fn = context["cost_fn"]
    estimator = Backprop(context["params"], context["free_layers"], cost_fn, context["minimizer"])

    _reset_params(context)
    network.set_input(images, reset=True)
    cost_fn.set_target(labels)
    grads = estimator.compute_gradient()[: len(context["params"])]
    with torch.no_grad():
        loss = float(cost_fn.eval().mean().item())
        errors = cost_fn.error_fn()
        accuracy = float((~errors).float().mean().item())
    grads = [grad.detach().clone() for grad in grads]
    _detach_state(context)
    return loss, accuracy, grads


def _compute_ep_gradient(
    context: dict,
    augmented_fn,
    augmented_minimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
) -> list[torch.Tensor]:
    network = context["network"]
    cost_fn = context["cost_fn"]
    free_minimizer = context["minimizer"]

    _reset_params(context)
    network.set_input(images, reset=True)
    cost_fn.set_target(labels)
    free_minimizer.compute_equilibrium()
    estimator = EquilibriumProp(
        context["params"],
        context["free_layers"],
        augmented_fn,
        cost_fn,
        augmented_minimizer,
        variant="centered",
        nudging=float(beta),
    )
    grads = estimator.compute_gradient()[: len(context["params"])]
    grads = [grad.detach().clone() for grad in grads]
    augmented_fn.nudging = 0.0
    _detach_state(context)
    return grads


def _vector_stats(bp_parts: list[torch.Tensor], ep_parts: list[torch.Tensor]) -> dict[str, float]:
    if not bp_parts:
        return {
            "num_values": 0,
            "bp_grad_l2": math.nan,
            "ep_grad_l2": math.nan,
            "grad_dot": math.nan,
            "cosine": math.nan,
            "relative_l2_error": math.nan,
        }
    bp_vec = torch.cat([part.detach().reshape(-1).float() for part in bp_parts])
    ep_vec = torch.cat([part.detach().reshape(-1).float() for part in ep_parts])
    dot = torch.dot(bp_vec, ep_vec)
    bp_norm = torch.linalg.vector_norm(bp_vec)
    ep_norm = torch.linalg.vector_norm(ep_vec)
    denom = bp_norm * ep_norm
    cosine = dot / denom if float(denom.item()) > 0.0 else torch.tensor(math.nan, device=bp_vec.device)
    diff_norm = torch.linalg.vector_norm(ep_vec - bp_vec)
    rel_error = diff_norm / bp_norm if float(bp_norm.item()) > 0.0 else torch.tensor(math.nan, device=bp_vec.device)
    return {
        "num_values": int(bp_vec.numel()),
        "bp_grad_l2": float(bp_norm.item()),
        "ep_grad_l2": float(ep_norm.item()),
        "grad_dot": float(dot.item()),
        "cosine": float(cosine.item()),
        "relative_l2_error": float(rel_error.item()),
    }


def _scope_rows(
    *,
    base_row: dict,
    params: list,
    bp_grads: list[torch.Tensor],
    ep_grads: list[torch.Tensor],
) -> list[dict]:
    rows: list[dict] = []

    def add_scope(scope: str, name: str, indices: list[int], param_class: str, param_group: str) -> None:
        if not indices:
            return
        stats = _vector_stats([bp_grads[index] for index in indices], [ep_grads[index] for index in indices])
        row = dict(base_row)
        row.update(
            {
                "scope": scope,
                "param_index": "" if scope != "param" else indices[0],
                "param_name": name,
                "param_class": param_class,
                "param_group": param_group,
                **stats,
            }
        )
        rows.append(row)

    all_indices = list(range(len(params)))
    weight_indices = [index for index, param in enumerate(params) if _param_group(param) == "weights"]
    bias_indices = [index for index, param in enumerate(params) if _param_group(param) == "biases"]
    other_indices = [index for index, param in enumerate(params) if _param_group(param) == "other"]
    add_scope("aggregate", "all_params", all_indices, "all", "all_params")
    add_scope("aggregate", "weights", weight_indices, "Weight", "weights")
    add_scope("aggregate", "biases", bias_indices, "Bias", "biases")
    add_scope("aggregate", "other", other_indices, "Other", "other")

    for index, param in enumerate(params):
        add_scope(
            "param",
            _param_name(param, index),
            [index],
            param.__class__.__name__,
            _param_group(param),
        )
    return rows


def _run_single(run, *, args: argparse.Namespace, device: torch.device, raw_path: Path) -> None:
    max_k = max(int(value) for value in args.iteration_counts)
    context = _build_eval_context(
        run,
        device=device,
        eval_batch_size=args.batch_size,
        no_download=args.no_download,
        inference_iterations_override=max_k,
    )
    batches = _collect_batches(context, args.split, args.max_batches)
    params = context["params"]

    for iteration_count in args.iteration_counts:
        context["minimizer"] = _build_tracking_minimizer(
            context["energy_fn"],
            context["free_layers"],
            context["model_cfg"],
            context["config"]["energy_minimizer"]["mode"],
            num_iterations=int(iteration_count),
            voltage_amp=context["energy_fn"]._voltage_amp,
            current_amp=context["energy_fn"]._current_amp,
        )
        augmented_fn, augmented_minimizer = _make_augmented_minimizer(context, int(iteration_count))
        buffered_rows: list[dict] = []
        for batch_index, (images, labels) in enumerate(batches):
            images = images.to(device)
            labels = labels.to(device)
            loss, accuracy, bp_grads = _compute_bp_gradient(context, images, labels)
            for beta in args.betas:
                ep_grads = _compute_ep_gradient(
                    context,
                    augmented_fn,
                    augmented_minimizer,
                    images,
                    labels,
                    float(beta),
                )
                base_row = {
                    "model_label": args.model_label,
                    "split": args.split,
                    "non_linearity": run.non_linearity,
                    "run_name": run.run_name,
                    "seed": run.seed,
                    "voltage_amp": run.voltage_amp,
                    "current_amp": run.current_amp,
                    "checkpoint_kind": args.checkpoint_kind,
                    "clean_accuracy": run.clean_accuracy,
                    "iteration_count": int(iteration_count),
                    "beta": float(beta),
                    "batch_index": batch_index,
                    "loss": loss,
                    "accuracy": accuracy,
                }
                buffered_rows.extend(
                    _scope_rows(
                        base_row=base_row,
                        params=params,
                        bp_grads=bp_grads,
                        ep_grads=ep_grads,
                    )
                )
            if len(buffered_rows) >= 128:
                _write_rows(raw_path, RAW_COLUMNS, buffered_rows)
                buffered_rows = []
        if buffered_rows:
            _write_rows(raw_path, RAW_COLUMNS, buffered_rows)


def _read_raw_rows(output_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(output_root.glob("raw_cosine*.csv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _summarize_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("model_label", ""),
                row.get("split", ""),
                row.get("non_linearity", ""),
                row.get("run_name", ""),
                int(float(row.get("seed", 0))),
                row.get("checkpoint_kind", ""),
                int(float(row.get("iteration_count", 0))),
                float(row.get("beta", 0.0)),
                row.get("scope", ""),
                row.get("param_name", ""),
            )
        ].append(row)

    summary_rows: list[dict] = []
    for key, values in sorted(grouped.items()):
        (
            model_label,
            split,
            non_linearity,
            run_name,
            seed,
            checkpoint_kind,
            iteration_count,
            beta,
            scope,
            param_name,
        ) = key
        first = values[0]
        cosine = _stats([_float(row, "cosine") for row in values])
        row = {
            "model_label": model_label,
            "split": split,
            "non_linearity": non_linearity,
            "run_name": run_name,
            "seed": seed,
            "voltage_amp": first.get("voltage_amp", ""),
            "current_amp": first.get("current_amp", ""),
            "checkpoint_kind": checkpoint_kind,
            "iteration_count": iteration_count,
            "beta": beta,
            "scope": scope,
            "param_name": param_name,
            "param_class": first.get("param_class", ""),
            "param_group": first.get("param_group", ""),
            "num_batches": len(values),
            "num_values": first.get("num_values", ""),
            "loss_mean": _stats([_float(value, "loss") for value in values])["mean"],
            "accuracy_mean": _stats([_float(value, "accuracy") for value in values])["mean"],
            "cosine_mean": cosine["mean"],
            "cosine_std": cosine["std"],
            "cosine_p10": cosine["p10"],
            "cosine_p50": cosine["p50"],
            "cosine_p90": cosine["p90"],
            "bp_grad_l2_mean": _stats([_float(value, "bp_grad_l2") for value in values])["mean"],
            "ep_grad_l2_mean": _stats([_float(value, "ep_grad_l2") for value in values])["mean"],
            "relative_l2_error_mean": _stats([_float(value, "relative_l2_error") for value in values])["mean"],
        }
        summary_rows.append(row)

    amp_grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in summary_rows:
        amp_grouped[
            (
                row["model_label"],
                row["split"],
                row["non_linearity"],
                row["run_name"],
                row["checkpoint_kind"],
                row["iteration_count"],
                row["beta"],
                row["scope"],
                row["param_name"],
            )
        ].append(row)

    amp_rows: list[dict] = []
    for key, values in sorted(amp_grouped.items()):
        (
            model_label,
            split,
            non_linearity,
            run_name,
            checkpoint_kind,
            iteration_count,
            beta,
            scope,
            param_name,
        ) = key
        first = values[0]
        cosine = _stats([_float(row, "cosine_mean") for row in values])
        amp_rows.append(
            {
                "model_label": model_label,
                "split": split,
                "non_linearity": non_linearity,
                "run_name": run_name,
                "voltage_amp": first.get("voltage_amp", ""),
                "current_amp": first.get("current_amp", ""),
                "checkpoint_kind": checkpoint_kind,
                "iteration_count": iteration_count,
                "beta": beta,
                "scope": scope,
                "param_name": param_name,
                "param_class": first.get("param_class", ""),
                "param_group": first.get("param_group", ""),
                "num_models": len(values),
                "num_batches": sum(int(row["num_batches"]) for row in values),
                "num_values": first.get("num_values", ""),
                "loss_mean": _stats([_float(row, "loss_mean") for row in values])["mean"],
                "accuracy_mean": _stats([_float(row, "accuracy_mean") for row in values])["mean"],
                "cosine_mean": cosine["mean"],
                "cosine_std": cosine["std"],
                "cosine_p10": cosine["p10"],
                "cosine_p50": cosine["p50"],
                "cosine_p90": cosine["p90"],
                "bp_grad_l2_mean": _stats([_float(row, "bp_grad_l2_mean") for row in values])["mean"],
                "ep_grad_l2_mean": _stats([_float(row, "ep_grad_l2_mean") for row in values])["mean"],
                "relative_l2_error_mean": _stats(
                    [_float(row, "relative_l2_error_mean") for row in values]
                )["mean"],
            }
        )
    return summary_rows, amp_rows


def _select_k(
    amp_rows: list[dict],
    *,
    primary_beta: float,
    plateau_cosine_tolerance: float,
) -> list[dict]:
    selected_rows: list[dict] = []
    candidates = [
        row
        for row in amp_rows
        if row.get("scope") == "aggregate"
        and row.get("param_name") == "all_params"
        and abs(_float(row, "beta") - primary_beta) < 1e-12
    ]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in candidates:
        grouped[
            (
                row.get("model_label", ""),
                row.get("split", ""),
                row.get("non_linearity", ""),
                row.get("run_name", ""),
                row.get("checkpoint_kind", ""),
            )
        ].append(row)

    for key, values in sorted(grouped.items()):
        values = sorted(values, key=lambda row: int(row["iteration_count"]))
        finite_values = [row for row in values if math.isfinite(_float(row, "cosine_mean"))]
        if not finite_values:
            continue
        reference = max(_float(row, "cosine_mean") for row in finite_values)
        highest = finite_values[-1]
        selected = None
        for row in finite_values:
            if _float(row, "cosine_mean") >= reference - plateau_cosine_tolerance:
                selected = row
                break
        if selected is None:
            selected = highest
        first = selected
        selected_rows.append(
            {
                "model_label": key[0],
                "split": key[1],
                "non_linearity": key[2],
                "run_name": key[3],
                "voltage_amp": first.get("voltage_amp", ""),
                "current_amp": first.get("current_amp", ""),
                "checkpoint_kind": key[4],
                "beta": primary_beta,
                "selected_iteration_count": int(first["iteration_count"]),
                "selected_cosine_mean": _float(first, "cosine_mean"),
                "reference_cosine_mean": reference,
                "highest_iteration_count": int(highest["iteration_count"]),
                "highest_iteration_cosine_mean": _float(highest, "cosine_mean"),
                "num_models": first.get("num_models", ""),
                "num_batches": first.get("num_batches", ""),
                "selection_rule": (
                    f"smallest K with all-parameter cosine within "
                    f"{plateau_cosine_tolerance:g} absolute cosine of best K at beta={primary_beta:g}"
                ),
            }
        )
    return selected_rows


def _beta_label(beta: float) -> str:
    return f"{float(beta):g}".replace("-", "m").replace(".", "p")


def _plot_cosine_by_amp(output_root: Path, amp_rows: list[dict], beta: float) -> None:
    rows = [
        row
        for row in amp_rows
        if row.get("scope") == "aggregate"
        and row.get("param_name") == "all_params"
        and abs(_float(row, "beta") - beta) < 1e-12
    ]
    if not rows:
        return
    nonlinearities = sorted({row.get("non_linearity", "") for row in rows})
    if not nonlinearities:
        return

    fig, axes = plt.subplots(
        1,
        len(nonlinearities),
        figsize=(max(6.0, 5.0 * len(nonlinearities)), 4.2),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, non_linearity in zip(axes[0], nonlinearities):
        subrows = [row for row in rows if row.get("non_linearity", "") == non_linearity]
        run_names = sorted(
            {row["run_name"] for row in subrows},
            key=lambda name: RUN_ORDER.index(name) if name in RUN_ORDER else 999,
        )
        for run_name in run_names:
            series = sorted(
                [row for row in subrows if row["run_name"] == run_name],
                key=lambda row: int(row["iteration_count"]),
            )
            x = [int(row["iteration_count"]) for row in series]
            y = [_float(row, "cosine_mean") for row in series]
            ax.plot(x, y, marker="o", linewidth=1.8, label=RUN_LABELS.get(run_name, run_name))
        ax.set_title(non_linearity or "unknown")
        ax.set_xlabel("iterations K")
        ax.set_ylabel("EP/BP cosine")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_root / f"cosine_all_params_by_amp_beta{_beta_label(beta)}.png", dpi=200)
    plt.close(fig)


def write_summaries_and_plots(
    output_root: Path,
    *,
    primary_beta: float = 0.25,
    plateau_cosine_tolerance: float = 0.01,
) -> None:
    rows = _read_raw_rows(output_root)
    if not rows:
        return
    summary_rows, amp_rows = _summarize_rows(rows)
    _write_rows(output_root / "summary_by_model_k_beta.csv", SUMMARY_COLUMNS, summary_rows, append=False)
    _write_rows(output_root / "summary_by_amp_k_beta.csv", AMP_COLUMNS, amp_rows, append=False)
    selected_rows = _select_k(
        amp_rows,
        primary_beta=primary_beta,
        plateau_cosine_tolerance=plateau_cosine_tolerance,
    )
    selected_name = f"selected_k_by_amp_beta{_beta_label(primary_beta)}.csv"
    _write_rows(output_root / selected_name, SELECTED_COLUMNS, selected_rows, append=False)

    betas = sorted({_float(row, "beta") for row in amp_rows if math.isfinite(_float(row, "beta"))})
    for beta in betas:
        _plot_cosine_by_amp(output_root, amp_rows, beta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-label", default="")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-kind", choices=("best", "final"), default="best")
    parser.add_argument("--run-name", action="append")
    parser.add_argument("--non-linearity", nargs="+")
    parser.add_argument("--training-seeds", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--iteration-counts", type=int, nargs="+", default=[4, 6, 8, 12, 16, 24, 32])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--primary-beta", type=float, default=0.25)
    parser.add_argument("--plateau-cosine-tolerance", type=float, default=0.01)
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
        write_summaries_and_plots(
            output_root,
            primary_beta=float(args.primary_beta),
            plateau_cosine_tolerance=float(args.plateau_cosine_tolerance),
        )
        print(f"[summary] wrote EP/BP cosine diagnostics under {output_root}")
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
    if args.non_linearity:
        nonlinearities = {value.lower() for value in args.non_linearity}
        rows = [row for row in rows if row.non_linearity.lower() in nonlinearities]
        if not rows:
            raise ValueError(f"No runs matched --non-linearity values {sorted(nonlinearities)}.")
    selected = _select_shard(rows, args.num_shards, args.shard_index)
    raw_name = args.raw_results_name or f"raw_cosine_shard_{args.shard_index}.csv"
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
            "max_batches": args.max_batches,
            "iteration_counts": [int(value) for value in args.iteration_counts],
            "betas": [float(value) for value in args.betas],
            "primary_beta": float(args.primary_beta),
            "plateau_cosine_tolerance": float(args.plateau_cosine_tolerance),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "selected_runs": [
                f"{row.non_linearity}/{row.run_name}/seed_{row.seed}" for row in selected
            ],
            "gradient_coordinate": "raw DRN parameters; aggregate rows include all params, weights only, and biases only",
            "ep_variant": "centered",
        },
    )
    print(f"[ep-bp-cosine] input_root={input_root}")
    print(f"[ep-bp-cosine] output_root={output_root}")
    print(f"[ep-bp-cosine] selected_runs={len(selected)}/{len(rows)} shard={args.shard_index}/{args.num_shards}")
    print(f"[ep-bp-cosine] K={args.iteration_counts} betas={args.betas} primary_beta={args.primary_beta}")
    if args.dry_run:
        for row in selected:
            print(f"[dry-run] {row.non_linearity} {row.run_name} seed={row.seed} checkpoint={row.checkpoint_path}")
        return

    for run in selected:
        print(f"[run] {run.non_linearity} {run.run_name} seed={run.seed} checkpoint={run.checkpoint_path}", flush=True)
        _run_single(run, args=args, device=device, raw_path=raw_path)
    write_summaries_and_plots(
        output_root,
        primary_beta=float(args.primary_beta),
        plateau_cosine_tolerance=float(args.plateau_cosine_tolerance),
    )
    print(f"[done] output={output_root}")


if __name__ == "__main__":
    main()
