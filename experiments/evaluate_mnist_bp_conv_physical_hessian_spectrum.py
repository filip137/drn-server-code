#!/usr/bin/env python3
"""Estimate physical free-state Hessian spectra for conv MNIST BP DRNs.

The conv networks have too many state variables to materialize the Hessian.
This script evaluates equilibria sample-by-sample and exposes the Hessian via
autograd Hessian-vector products. It then estimates spectrum/trace metrics that
are comparable across amplification schemes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_mnist_bp_write_noise_sweep import (
    DEFAULT_INPUT_ROOT,
    RunRow,
    _build_eval_context,
    _json_sanitize,
    _read_summary_rows,
    _reset_params,
    _scores,
    _select_shard,
    _write_json,
)
from model.variable.layer import layer_index


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "mnist_bp_conv_physical_hessian_spectrum"

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
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "checkpoint_path",
    "sample_index",
    "label",
    "prediction",
    "correct",
    "loss",
    "free_dim",
    "lambda_max",
    "lambda_min",
    "condition_est",
    "trace_est",
    "trace_per_dim",
    "rayleigh_mean",
    "rayleigh_std",
    "active_fraction",
    "active_count",
    "nonlinear_conductance_mean",
    "nonlinear_conductance_p90",
    "method",
    "num_hutchinson",
]
SUMMARY_COLUMNS = [
    "model_label",
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "num_samples",
    "accuracy",
    "loss_mean",
    "free_dim",
    "lambda_max_mean",
    "lambda_max_p50",
    "lambda_max_p90",
    "lambda_min_mean",
    "lambda_min_p10",
    "lambda_min_p50",
    "condition_est_mean",
    "condition_est_p50",
    "condition_est_p90",
    "trace_est_mean",
    "trace_per_dim_mean",
    "rayleigh_mean",
    "active_fraction_mean",
    "active_fraction_p50",
    "nonlinear_conductance_mean",
    "nonlinear_conductance_p90",
]
AMP_COLUMNS = [
    "model_label",
    "non_linearity",
    "run_name",
    "voltage_amp",
    "current_amp",
    "num_models",
    "num_samples",
    "accuracy_mean",
    "loss_mean",
    "lambda_max_mean",
    "lambda_max_std_across_models",
    "lambda_min_mean",
    "condition_est_mean",
    "trace_per_dim_mean",
    "rayleigh_mean",
    "active_fraction_mean",
    "nonlinear_conductance_mean",
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
        return {"mean": math.nan, "std": math.nan, "p10": math.nan, "p50": math.nan, "p90": math.nan}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _unflatten_like(flat: torch.Tensor, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    pieces: list[torch.Tensor] = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        pieces.append(flat[offset : offset + numel].view_as(tensor))
        offset += numel
    if offset != flat.numel():
        raise ValueError("Flat vector length does not match free-layer states.")
    return pieces


def _nonlinear_conductance_diag(
    context: dict,
    free_states: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    model_cfg = context["model_cfg"]
    non_linearity = model_cfg.get("non_linearity", "")
    voltage_amp = float(model_cfg["voltage_amp"])
    current_amp = float(model_cfg["current_amp"])
    output_layer = context["output_layer"]
    pieces: list[torch.Tensor] = []
    eligible_pieces: list[torch.Tensor] = []
    for layer, state in zip(context["free_layers"], free_states):
        diag = torch.zeros_like(state)
        eligible = torch.zeros_like(state, dtype=torch.bool)
        if layer is not output_layer:
            scale = (current_amp / voltage_amp) ** (layer_index(layer) - 1)
            if non_linearity == "hard_sigmoid":
                eligible = torch.ones_like(state, dtype=torch.bool)
                params = model_cfg.get("hard_sigmoid_param", {})
                v_min = float(params.get("v_min", -1.5))
                v_max = float(params.get("v_max", 1.5))
                g_on = float(params.get("g_on", 100.0)) * scale
                g_off = float(params.get("g_off", 0.0)) * scale
                off = (state >= v_min) & (state <= v_max)
                diag = torch.where(
                    off,
                    torch.full_like(state, g_off),
                    torch.full_like(state, g_on),
                )
            elif non_linearity == "perfect_diode":
                g = float(context["lpw_conductance"]) * scale
                if state.ndim >= 2 and state.shape[1] % 2 == 0:
                    eligible = torch.ones_like(state, dtype=torch.bool)
                    half = state.shape[1] // 2
                    exc = state[:, :half] > 0
                    inh = state[:, half:] < 0
                    diag[:, :half] = torch.where(exc, torch.full_like(diag[:, :half], g), diag[:, :half])
                    diag[:, half:] = torch.where(inh, torch.full_like(diag[:, half:], g), diag[:, half:])
        pieces.append(diag.reshape(-1))
        eligible_pieces.append(eligible.reshape(-1))
    return torch.cat(pieces), torch.cat(eligible_pieces)


def _prepare_free_state_hvp(context: dict) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor, dict]:
    free_layers = context["free_layers"]
    free_states: list[torch.Tensor] = []
    for layer in free_layers:
        layer.state = layer.state.detach().clone().requires_grad_(True)
        free_states.append(layer.state)

    nonlinear_diag, nonlinear_eligible = _nonlinear_conductance_diag(context, free_states)
    energy = context["energy_fn"].eval().sum()
    grads = torch.autograd.grad(energy, free_states, create_graph=True, retain_graph=True)
    flat_grad = _flatten(list(grads))
    add_diag = context["model_cfg"].get("non_linearity") == "perfect_diode"

    def hvp(vector: torch.Tensor) -> torch.Tensor:
        vector = vector.to(device=flat_grad.device, dtype=flat_grad.dtype)
        hessian_vector = torch.autograd.grad(
            flat_grad,
            free_states,
            grad_outputs=vector,
            retain_graph=True,
        )
        out = _flatten(list(hessian_vector))
        if add_diag:
            out = out + nonlinear_diag * vector
        return out

    eligible_diag = nonlinear_diag[nonlinear_eligible]
    active = eligible_diag > 0
    stats = {
        "free_dim": int(nonlinear_diag.numel()),
        "active_count": int(active.sum().item()),
        "active_fraction": float(active.float().mean().item()) if eligible_diag.numel() else math.nan,
        "nonlinear_conductance_mean": float(eligible_diag.mean().item()) if eligible_diag.numel() else math.nan,
        "nonlinear_conductance_p90": (
            float(torch.quantile(eligible_diag.detach(), 0.90).item())
            if eligible_diag.numel()
            else math.nan
        ),
    }
    return hvp, nonlinear_diag.detach(), stats


def _estimate_power_spectrum(
    hvp: Callable[[torch.Tensor], torch.Tensor],
    *,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    num_power_iters: int,
    num_hutchinson: int,
    seed: int,
) -> dict:
    generator = torch.Generator(device=str(device))
    generator.manual_seed(int(seed))
    vector = torch.randn(dim, generator=generator, device=device, dtype=dtype)
    vector = vector / torch.linalg.vector_norm(vector).clamp_min(1e-12)
    lambda_max = math.nan
    for _ in range(max(num_power_iters, 1)):
        out = hvp(vector)
        norm = torch.linalg.vector_norm(out).clamp_min(1e-12)
        vector = out / norm
        lambda_max = float(torch.dot(vector, hvp(vector)).detach().item())

    rayleigh: list[float] = []
    trace_values: list[float] = []
    for _ in range(max(num_hutchinson, 1)):
        z = torch.randint(0, 2, (dim,), generator=generator, device=device, dtype=torch.int64)
        z = (2 * z - 1).to(dtype=dtype)
        hz = hvp(z)
        value = float(torch.dot(z, hz).detach().item())
        trace_values.append(value)
        rayleigh.append(value / dim)
    return {
        "lambda_max": lambda_max,
        "lambda_min": math.nan,
        "condition_est": math.nan,
        "trace_est": float(np.mean(trace_values)),
        "trace_per_dim": float(np.mean(trace_values) / dim),
        "rayleigh_mean": float(np.mean(rayleigh)),
        "rayleigh_std": float(np.std(rayleigh, ddof=1)) if len(rayleigh) > 1 else 0.0,
    }


def _estimate_eigsh_spectrum(
    hvp: Callable[[torch.Tensor], torch.Tensor],
    *,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    num_hutchinson: int,
    eigsh_tol: float,
    eigsh_maxiter: int,
    seed: int,
) -> dict:
    try:
        from scipy.sparse.linalg import LinearOperator, eigsh
    except Exception:
        return _estimate_power_spectrum(
            hvp,
            dim=dim,
            device=device,
            dtype=dtype,
            num_power_iters=max(20, eigsh_maxiter // 4),
            num_hutchinson=num_hutchinson,
            seed=seed,
        )

    def matvec(array: np.ndarray) -> np.ndarray:
        vector = torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device=device, dtype=dtype)
        out = hvp(vector).detach().cpu().numpy()
        return np.asarray(out, dtype=np.float64)

    operator = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    try:
        lambda_max = float(eigsh(operator, k=1, which="LA", tol=eigsh_tol, maxiter=eigsh_maxiter, return_eigenvectors=False)[0])
    except Exception:
        lambda_max = math.nan
    try:
        lambda_min = float(eigsh(operator, k=1, which="SA", tol=eigsh_tol, maxiter=eigsh_maxiter, return_eigenvectors=False)[0])
    except Exception:
        lambda_min = math.nan
    condition = (
        float(lambda_max / lambda_min)
        if math.isfinite(lambda_max) and math.isfinite(lambda_min) and lambda_min > 0
        else math.nan
    )
    trace_stats = _estimate_power_spectrum(
        hvp,
        dim=dim,
        device=device,
        dtype=dtype,
        num_power_iters=1,
        num_hutchinson=num_hutchinson,
        seed=seed,
    )
    return {
        "lambda_max": lambda_max,
        "lambda_min": lambda_min,
        "condition_est": condition,
        "trace_est": trace_stats["trace_est"],
        "trace_per_dim": trace_stats["trace_per_dim"],
        "rayleigh_mean": trace_stats["rayleigh_mean"],
        "rayleigh_std": trace_stats["rayleigh_std"],
    }


def _run_single(run: RunRow, *, args: argparse.Namespace, device: torch.device, raw_path: Path) -> None:
    context = _build_eval_context(
        run,
        device=device,
        eval_batch_size=1,
        no_download=args.no_download,
        inference_iterations_override=args.inference_iterations,
    )
    context["lpw_conductance"] = float(args.lpw_conductance)
    network = context["network"]
    minimizer = context["minimizer"]
    cost_fn = context["cost_fn"]
    output_layer = context["output_layer"]
    num_classes = context["num_classes"]
    raw_rows: list[dict] = []
    seen = 0
    for batch_index, (images, labels) in enumerate(context["test_loader"]):
        if args.max_test_samples is not None and seen >= args.max_test_samples:
            break
        images = images.to(device)
        labels = labels.to(device)
        for local_index in range(images.shape[0]):
            if args.max_test_samples is not None and seen >= args.max_test_samples:
                break
            _reset_params(context)
            image = images[local_index : local_index + 1]
            label = labels[local_index : local_index + 1]
            with torch.no_grad():
                network.set_input(image, reset=True)
                minimizer.compute_equilibrium()
                cost_fn.set_target(label)
                loss = float(cost_fn.eval().mean().item())
                scores = _scores(output_layer.state.detach(), num_classes)
                prediction = int(torch.argmax(scores, dim=1).item())
                correct = int(prediction == int(label.item()))

            hvp, _diag, diag_stats = _prepare_free_state_hvp(context)
            dim = int(diag_stats["free_dim"])
            first_state = context["free_layers"][0].state
            seed = int(args.spectrum_seed_offset + 100000 * run.seed + seen)
            if args.spectrum_method == "eigsh":
                spectrum = _estimate_eigsh_spectrum(
                    hvp,
                    dim=dim,
                    device=first_state.device,
                    dtype=first_state.dtype,
                    num_hutchinson=args.num_hutchinson,
                    eigsh_tol=args.eigsh_tol,
                    eigsh_maxiter=args.eigsh_maxiter,
                    seed=seed,
                )
            else:
                spectrum = _estimate_power_spectrum(
                    hvp,
                    dim=dim,
                    device=first_state.device,
                    dtype=first_state.dtype,
                    num_power_iters=args.power_iters,
                    num_hutchinson=args.num_hutchinson,
                    seed=seed,
                )

            raw_rows.append(
                {
                    "model_label": args.model_label,
                    "non_linearity": run.non_linearity,
                    "run_name": run.run_name,
                    "seed": run.seed,
                    "voltage_amp": run.voltage_amp,
                    "current_amp": run.current_amp,
                    "checkpoint_path": str(run.checkpoint_path),
                    "sample_index": seen,
                    "label": int(label.item()),
                    "prediction": prediction,
                    "correct": correct,
                    "loss": loss,
                    **diag_stats,
                    **spectrum,
                    "method": args.spectrum_method,
                    "num_hutchinson": args.num_hutchinson,
                }
            )
            seen += 1
            if len(raw_rows) >= 8:
                _write_rows(raw_path, RAW_COLUMNS, raw_rows)
                raw_rows = []
    if raw_rows:
        _write_rows(raw_path, RAW_COLUMNS, raw_rows)


def _read_raw_rows(output_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(output_root.glob("raw_physical_hessian_samples*.csv")):
        rows.extend(csv.DictReader(path.open()))
    return rows


def write_summaries_and_plots(output_root: Path) -> None:
    rows = _read_raw_rows(output_root)
    if not rows:
        return
    grouped: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("model_label", ""),
                row.get("non_linearity", ""),
                row["run_name"],
                int(row["seed"]),
            )
        ].append(row)

    summary_rows: list[dict] = []
    for (model_label, non_linearity, run_name, seed), values in sorted(grouped.items()):
        first = values[0]
        lam_max = _stats([float(row["lambda_max"]) for row in values])
        lam_min = _stats([float(row["lambda_min"]) for row in values])
        cond = _stats([float(row["condition_est"]) for row in values])
        trace = _stats([float(row["trace_est"]) for row in values])
        trace_per_dim = _stats([float(row["trace_per_dim"]) for row in values])
        rayleigh = _stats([float(row["rayleigh_mean"]) for row in values])
        active = _stats([float(row["active_fraction"]) for row in values])
        conductance_mean = _stats([float(row["nonlinear_conductance_mean"]) for row in values])
        conductance_p90 = _stats([float(row["nonlinear_conductance_p90"]) for row in values])
        losses = _stats([float(row["loss"]) for row in values])
        correct = [int(row["correct"]) for row in values]
        summary_rows.append(
            {
                "model_label": model_label,
                "non_linearity": non_linearity,
                "run_name": run_name,
                "seed": seed,
                "voltage_amp": first["voltage_amp"],
                "current_amp": first["current_amp"],
                "num_samples": len(values),
                "accuracy": float(np.mean(correct)) if correct else math.nan,
                "loss_mean": losses["mean"],
                "free_dim": first["free_dim"],
                "lambda_max_mean": lam_max["mean"],
                "lambda_max_p50": lam_max["p50"],
                "lambda_max_p90": lam_max["p90"],
                "lambda_min_mean": lam_min["mean"],
                "lambda_min_p10": lam_min["p10"],
                "lambda_min_p50": lam_min["p50"],
                "condition_est_mean": cond["mean"],
                "condition_est_p50": cond["p50"],
                "condition_est_p90": cond["p90"],
                "trace_est_mean": trace["mean"],
                "trace_per_dim_mean": trace_per_dim["mean"],
                "rayleigh_mean": rayleigh["mean"],
                "active_fraction_mean": active["mean"],
                "active_fraction_p50": active["p50"],
                "nonlinear_conductance_mean": conductance_mean["mean"],
                "nonlinear_conductance_p90": conductance_p90["mean"],
            }
        )
    _write_rows(output_root / "summary_by_model.csv", SUMMARY_COLUMNS, summary_rows, append=False)

    amp_grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in summary_rows:
        amp_grouped[(row["model_label"], row["non_linearity"], row["run_name"])].append(row)
    amp_rows: list[dict] = []
    for (model_label, non_linearity, run_name), values in sorted(amp_grouped.items()):
        first = values[0]
        amp_rows.append(
            {
                "model_label": model_label,
                "non_linearity": non_linearity,
                "run_name": run_name,
                "voltage_amp": first["voltage_amp"],
                "current_amp": first["current_amp"],
                "num_models": len(values),
                "num_samples": sum(int(row["num_samples"]) for row in values),
                "accuracy_mean": _stats([float(row["accuracy"]) for row in values])["mean"],
                "loss_mean": _stats([float(row["loss_mean"]) for row in values])["mean"],
                "lambda_max_mean": _stats([float(row["lambda_max_mean"]) for row in values])["mean"],
                "lambda_max_std_across_models": _stats([float(row["lambda_max_mean"]) for row in values])["std"],
                "lambda_min_mean": _stats([float(row["lambda_min_mean"]) for row in values])["mean"],
                "condition_est_mean": _stats([float(row["condition_est_mean"]) for row in values])["mean"],
                "trace_per_dim_mean": _stats([float(row["trace_per_dim_mean"]) for row in values])["mean"],
                "rayleigh_mean": _stats([float(row["rayleigh_mean"]) for row in values])["mean"],
                "active_fraction_mean": _stats([float(row["active_fraction_mean"]) for row in values])["mean"],
                "nonlinear_conductance_mean": _stats([float(row["nonlinear_conductance_mean"]) for row in values])["mean"],
            }
        )
    _write_rows(output_root / "summary_by_nonlinearity_amp.csv", AMP_COLUMNS, amp_rows, append=False)
    _plot_amp_metric(output_root / "lambda_max_by_amp.png", amp_rows, "lambda_max_mean", "mean largest eigenvalue")
    _plot_amp_metric(output_root / "condition_est_by_amp.png", amp_rows, "condition_est_mean", "estimated condition")
    _plot_amp_metric(output_root / "trace_per_dim_by_amp.png", amp_rows, "trace_per_dim_mean", "trace per free dimension")
    _plot_amp_metric(output_root / "active_fraction_by_amp.png", amp_rows, "active_fraction_mean", "active nonlinear fraction")


def _plot_amp_metric(path: Path, rows: list[dict], key: str, ylabel: str) -> None:
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
    ax.bar(x, y, color="#4c78a8", alpha=0.85)
    multiple = len({row.get("non_linearity", "") for row in rows}) > 1
    labels = []
    for row in rows:
        amp = RUN_LABELS.get(row["run_name"], row["run_name"])
        if multiple:
            prefix = {"hard_sigmoid": "hs", "perfect_diode": "pd"}.get(
                row.get("non_linearity", ""),
                row.get("non_linearity", ""),
            )
            labels.append(f"{prefix}\n{amp}")
        else:
            labels.append(amp)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    if np.all(np.isfinite(y)) and np.nanmin(y) > 0 and np.nanmax(y) / np.nanmin(y) > 20:
        ax.set_yscale("log")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-label", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-kind", choices=("best", "final"), default="best")
    parser.add_argument("--run-name", action="append")
    parser.add_argument("--training-seeds", type=int, nargs="+")
    parser.add_argument("--inference-iterations", type=int)
    parser.add_argument("--max-test-samples", type=int, default=16)
    parser.add_argument("--spectrum-method", choices=("eigsh", "power"), default="eigsh")
    parser.add_argument("--power-iters", type=int, default=30)
    parser.add_argument("--num-hutchinson", type=int, default=8)
    parser.add_argument("--eigsh-tol", type=float, default=1e-3)
    parser.add_argument("--eigsh-maxiter", type=int, default=80)
    parser.add_argument("--lpw-conductance", type=float, default=100.0)
    parser.add_argument("--spectrum-seed-offset", type=int, default=98765)
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
        print(f"[summary] wrote physical Hessian summaries under {output_root}")
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
    raw_name = args.raw_results_name or f"raw_physical_hessian_samples_shard_{args.shard_index}.csv"
    raw_path = output_root / raw_name
    config_payload = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "model_label": args.model_label,
        "checkpoint_kind": args.checkpoint_kind,
        "device": str(device),
        "inference_iterations": args.inference_iterations,
        "max_test_samples": args.max_test_samples,
        "spectrum_method": args.spectrum_method,
        "power_iters": args.power_iters,
        "num_hutchinson": args.num_hutchinson,
        "eigsh_tol": args.eigsh_tol,
        "eigsh_maxiter": args.eigsh_maxiter,
        "lpw_conductance": args.lpw_conductance,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selected_runs": [
            f"{row.non_linearity}/{row.run_name}/seed_{row.seed}" for row in selected
        ],
    }
    _write_json(output_root / f"config_shard_{args.shard_index}.json", config_payload)
    print(f"[physical-hessian] input_root={input_root}")
    print(f"[physical-hessian] output_root={output_root}")
    print(f"[physical-hessian] selected_runs={len(selected)}/{len(rows)} shard={args.shard_index}/{args.num_shards}")
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
        _run_single(run, args=args, device=device, raw_path=raw_path)
    write_summaries_and_plots(output_root)
    print(f"[done] output={output_root}")


if __name__ == "__main__":
    main()
