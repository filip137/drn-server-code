#!/usr/bin/env python3
"""Analyze hidden/output Hessian block directionality for MNIST DRN-XS runs.

This is an offline follow-up to
``evaluate_mnist_bp_hessian_lpw_amplification_sweep.py``.  It consumes the
saved linear Hessians and LPW conductance masks, then measures whether the
hidden-output coupling is feedforward- or feedback-dominated after normalizing
by the local hidden and output block stiffnesses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HESSIAN_ROOT = (
    REPO_ROOT / "results" / "mnist_bp_hessian_lpw_amplification_sweep_legacy_drnxs"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_HESSIAN_ROOT / "block_directional_analysis"
RUN_ORDER = [
    "mnist_bp_amp_v1_c1",
    "mnist_bp_amp_v2_c1",
    "mnist_bp_amp_v4_c1",
    "mnist_bp_amp_v1_c2",
    "mnist_bp_amp_v1_c4",
]
EPS = 1e-30

BLOCK_METRIC_KEYS = [
    "F_raw",
    "B_raw",
    "raw_ff_ratio",
    "raw_asymmetry_relerr",
    "F_transfer",
    "B_transfer",
    "feedforwardness",
    "feedback_fraction",
    "loop_rho",
    "loop_norm2",
    "schur_correction_norm",
    "loop_correction_fraction",
    "H_hh_norm",
    "H_oo_norm",
    "H_cross_norm",
    "H_hh_cond",
    "H_oo_cond",
]
SAMPLE_COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "v_over_i",
    "sample_index",
    "label",
    "prediction",
    "correct",
    "lpw_active_hidden_count",
    "lpw_active_hidden_fraction",
    *BLOCK_METRIC_KEYS,
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
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


def finite_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_p10": math.nan,
            f"{prefix}_p50": math.nan,
            f"{prefix}_p90": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def _summary_sort_key(row: dict[str, object]) -> tuple[int, int]:
    run_name = str(row.get("run_name", ""))
    try:
        run_idx = RUN_ORDER.index(run_name)
    except ValueError:
        run_idx = len(RUN_ORDER)
    return run_idx, int(row.get("seed", 0))


def selected_summary_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    summary_path = args.hessian_root / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected Hessian summary CSV at {summary_path}.")
    rows = sorted(read_csv(summary_path), key=_summary_sort_key)
    run_names = set(args.run_name) if args.run_name else None
    seeds = set(args.seeds) if args.seeds else None
    out = []
    for row in rows:
        if run_names is not None and row["run_name"] not in run_names:
            continue
        if seeds is not None and int(row["seed"]) not in seeds:
            continue
        out.append(row)
    if not out:
        raise ValueError("No Hessian runs matched the requested filters.")
    return out


def load_npz_json(npz: np.lib.npyio.NpzFile, key: str) -> dict:
    if key not in npz.files:
        return {}
    value = npz[key]
    return json.loads(str(value.item() if value.shape == () else value))


def is_diagonal(matrix: np.ndarray, *, rtol: float = 1e-10, atol: float = 1e-12) -> bool:
    diag = np.diag(np.diag(matrix))
    return bool(np.allclose(matrix, diag, rtol=rtol, atol=atol))


def solve_left(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve matrix @ x = rhs, using the diagonal fast path common in DRN-XS."""
    diag = np.diag(matrix)
    if is_diagonal(matrix) and np.all(np.abs(diag) > EPS):
        return rhs / diag[:, None]
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix) @ rhs


def condition_number(matrix: np.ndarray) -> float:
    diag = np.diag(matrix)
    if is_diagonal(matrix) and np.all(np.abs(diag) > EPS):
        abs_diag = np.abs(diag)
        return float(np.max(abs_diag) / np.min(abs_diag))
    try:
        return float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return math.nan


def block_direction_metrics(hessian: np.ndarray, free_dims: np.ndarray | list[int]) -> dict[str, float]:
    free_dims = list(map(int, free_dims))
    if len(free_dims) != 2:
        raise ValueError(f"Expected one hidden layer plus output layer; got free_dims={free_dims}.")
    hidden_dim, output_dim = free_dims
    if hessian.shape != (hidden_dim + output_dim, hidden_dim + output_dim):
        raise ValueError(
            f"Expected Hessian shape {(hidden_dim + output_dim, hidden_dim + output_dim)}, "
            f"got {hessian.shape}."
        )

    h_hh = hessian[:hidden_dim, :hidden_dim]
    h_ho = hessian[:hidden_dim, hidden_dim:]
    h_oh = hessian[hidden_dim:, :hidden_dim]
    h_oo = hessian[hidden_dim:, hidden_dim:]

    f_raw = float(np.linalg.norm(h_oh, ord="fro"))
    b_raw = float(np.linalg.norm(h_ho, ord="fro"))
    raw_scale = max(f_raw, b_raw, EPS)
    raw_asymmetry = float(np.linalg.norm(h_oh - h_ho.T, ord="fro") / raw_scale)

    forward_transfer = solve_left(h_oo, h_oh)
    feedback_transfer = solve_left(h_hh, h_ho)
    f_transfer = float(np.linalg.norm(forward_transfer, ord="fro"))
    b_transfer = float(np.linalg.norm(feedback_transfer, ord="fro"))

    # Non-zero eigenvalues match the larger hidden-loop product, but this 10x10
    # output-loop form is cheaper and numerically cleaner for DRN-XS.
    loop_output = forward_transfer @ feedback_transfer
    loop_eigvals = np.linalg.eigvals(loop_output)
    schur_correction = h_oh @ feedback_transfer
    h_oo_norm = float(np.linalg.norm(h_oo, ord="fro"))

    return {
        "F_raw": f_raw,
        "B_raw": b_raw,
        "raw_ff_ratio": float(f_raw / (b_raw + EPS)),
        "raw_asymmetry_relerr": raw_asymmetry,
        "F_transfer": f_transfer,
        "B_transfer": b_transfer,
        "feedforwardness": float(f_transfer / (b_transfer + EPS)),
        "feedback_fraction": float(b_transfer / (f_transfer + b_transfer + EPS)),
        "loop_rho": float(np.max(np.abs(loop_eigvals))) if loop_eigvals.size else math.nan,
        "loop_norm2": float(np.linalg.norm(loop_output, ord=2)),
        "schur_correction_norm": float(np.linalg.norm(schur_correction, ord="fro")),
        "loop_correction_fraction": float(np.linalg.norm(schur_correction, ord="fro") / (h_oo_norm + EPS)),
        "H_hh_norm": float(np.linalg.norm(h_hh, ord="fro")),
        "H_oo_norm": h_oo_norm,
        "H_cross_norm": f_raw,
        "H_hh_cond": condition_number(h_hh),
        "H_oo_cond": condition_number(h_oo),
    }


def with_prefix(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def hessian_with_hidden_diag(
    hessian_linear: np.ndarray,
    hidden_diag: np.ndarray,
    output_dim: int,
) -> np.ndarray:
    hessian = hessian_linear.copy()
    diag = np.concatenate([hidden_diag.astype(np.float64), np.zeros(output_dim, dtype=np.float64)])
    idx = np.diag_indices_from(hessian)
    hessian[idx] += diag
    return hessian


def analyze_run(row: dict[str, str], args: argparse.Namespace) -> dict:
    run_name = row["run_name"]
    seed = int(row["seed"])
    out_dir = args.output_root / run_name / f"seed_{seed}"
    sample_path = out_dir / "block_sample_metrics.csv"
    summary_path = out_dir / "block_summary.json"
    if summary_path.exists() and sample_path.exists() and not args.rerun_complete:
        print(f"[skip] {run_name} seed={seed} already analyzed")
        return json.loads(summary_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    hessian_linear_npz = np.load(row["hessian_linear_path"], allow_pickle=True)
    hessian_mean_npz = np.load(row["hessian_lpw_mean_path"], allow_pickle=True)
    masks_npz = np.load(row["lpw_masks_path"], allow_pickle=True)

    hessian_linear = hessian_linear_npz["hessian"].astype(np.float64)
    hessian_lpw_mean = hessian_mean_npz["hessian"].astype(np.float64)
    free_dims = hessian_linear_npz["free_dims"].astype(np.int64)
    hidden_dim, output_dim = map(int, free_dims)
    if hidden_dim != 100 or output_dim != 10:
        raise ValueError(f"Expected DRN-XS free_dims [100,10], got {free_dims.tolist()}.")

    metadata = load_npz_json(masks_npz, "metadata_json")
    lpw_layer_conductance = float(metadata.get("lpw_layer_conductance", metadata.get("lpw_conductance", 100.0)))
    unique_masks = masks_npz["unique_hidden_active_mask"].astype(bool)
    inverse = masks_npz["inverse_unique_mask"].astype(np.int64)
    labels = masks_npz["labels"].astype(np.int64)
    predictions = masks_npz["predictions"].astype(np.int64)
    correct = masks_npz["correct"].astype(bool)

    sample_count = int(inverse.shape[0])
    if args.max_samples is not None:
        sample_count = min(sample_count, int(args.max_samples))
        inverse = inverse[:sample_count]
        labels = labels[:sample_count]
        predictions = predictions[:sample_count]
        correct = correct[:sample_count]

    unique_metric_cache: dict[int, dict[str, float]] = {}
    for unique_index in np.unique(inverse):
        hidden_diag = unique_masks[int(unique_index)].astype(np.float64) * lpw_layer_conductance
        hessian_sample = hessian_with_hidden_diag(hessian_linear, hidden_diag, output_dim)
        unique_metric_cache[int(unique_index)] = block_direction_metrics(hessian_sample, free_dims)

    sample_rows: list[dict] = []
    sample_values = {key: np.empty(sample_count, dtype=np.float64) for key in BLOCK_METRIC_KEYS}
    active_counts = np.empty(sample_count, dtype=np.int64)
    active_fractions = np.empty(sample_count, dtype=np.float64)
    voltage_amp = float(row["voltage_amp"])
    current_amp = float(row["current_amp"])
    v_over_i = voltage_amp / current_amp

    for sample_index, unique_index in enumerate(inverse):
        mask = unique_masks[int(unique_index)]
        active_count = int(mask.sum())
        active_fraction = float(active_count / hidden_dim)
        metrics = unique_metric_cache[int(unique_index)]
        for key in BLOCK_METRIC_KEYS:
            sample_values[key][sample_index] = metrics[key]
        active_counts[sample_index] = active_count
        active_fractions[sample_index] = active_fraction
        sample_rows.append(
            {
                "run_name": run_name,
                "seed": seed,
                "voltage_amp": voltage_amp,
                "current_amp": current_amp,
                "v_over_i": v_over_i,
                "sample_index": sample_index,
                "label": int(labels[sample_index]),
                "prediction": int(predictions[sample_index]),
                "correct": int(correct[sample_index]),
                "lpw_active_hidden_count": active_count,
                "lpw_active_hidden_fraction": active_fraction,
                **metrics,
            }
        )

    write_csv(sample_path, sample_rows, SAMPLE_COLUMNS)

    linear_metrics = block_direction_metrics(hessian_linear, free_dims)
    lpw_mean_metrics = block_direction_metrics(hessian_lpw_mean, free_dims)
    run_summary: dict[str, object] = {
        "run_name": run_name,
        "seed": seed,
        "voltage_amp": voltage_amp,
        "current_amp": current_amp,
        "v_over_i": v_over_i,
        "sample_count": sample_count,
        "unique_lpw_masks_used": int(len(unique_metric_cache)),
        "test_accuracy_recomputed": finite_float(row.get("test_accuracy_recomputed")),
        "lpw_layer_conductance": lpw_layer_conductance,
        "hessian_linear_path": row["hessian_linear_path"],
        "hessian_lpw_mean_path": row["hessian_lpw_mean_path"],
        "lpw_masks_path": row["lpw_masks_path"],
        "sample_block_metrics_path": str(sample_path),
        "block_summary_path": str(summary_path),
        **with_prefix(linear_metrics, "linear"),
        **with_prefix(lpw_mean_metrics, "lpw_mean"),
        **stats(active_fractions, "lpw_sample_active_hidden_fraction"),
    }
    for key in BLOCK_METRIC_KEYS:
        run_summary.update(stats(sample_values[key], f"lpw_sample_{key}"))

    summary_path.write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n")
    print(
        f"[run] {run_name} seed={seed} samples={sample_count} "
        f"ff={run_summary['lpw_sample_feedforwardness_mean']:.4g} "
        f"fb={run_summary['lpw_sample_feedback_fraction_mean']:.4g} "
        f"loop_rho={run_summary['lpw_sample_loop_rho_mean']:.4g}"
    )
    return run_summary


def aggregate_by_amp(run_rows: list[dict]) -> list[dict]:
    by_amp: list[dict] = []
    for run_name in RUN_ORDER:
        rows = [row for row in run_rows if row["run_name"] == run_name]
        if not rows:
            continue
        sample_rows: list[dict[str, str]] = []
        for row in rows:
            sample_path = Path(str(row["sample_block_metrics_path"]))
            if sample_path.exists():
                sample_rows.extend(read_csv(sample_path))

        result: dict[str, object] = {
            "run_name": run_name,
            "voltage_amp": rows[0]["voltage_amp"],
            "current_amp": rows[0]["current_amp"],
            "v_over_i": rows[0]["v_over_i"],
            "num_seeds": len(rows),
            "sample_count": len(sample_rows),
            "test_accuracy_mean": float(np.mean([finite_float(row["test_accuracy_recomputed"]) for row in rows])),
            "test_accuracy_std": float(np.std([finite_float(row["test_accuracy_recomputed"]) for row in rows])),
        }

        for key in BLOCK_METRIC_KEYS + ["lpw_active_hidden_fraction"]:
            values = np.asarray([finite_float(row[key]) for row in sample_rows], dtype=np.float64)
            result.update(stats(values, key))

        for key in BLOCK_METRIC_KEYS:
            result[f"linear_{key}_mean_over_seeds"] = float(
                np.mean([finite_float(row[f"linear_{key}"]) for row in rows])
            )
            result[f"lpw_mean_{key}_mean_over_seeds"] = float(
                np.mean([finite_float(row[f"lpw_mean_{key}"]) for row in rows])
            )
        by_amp.append(result)
    return by_amp


def make_bar_plot(
    rows: list[dict],
    key: str,
    ylabel: str,
    output_path: Path,
    *,
    log_y: bool = False,
    color: str = "#4c78a8",
) -> None:
    if not rows:
        return
    labels = [str(row["run_name"]).replace("mnist_bp_amp_", "") for row in rows]
    means = np.asarray([finite_float(row[f"{key}_mean"]) for row in rows], dtype=np.float64)
    stds = np.asarray([finite_float(row[f"{key}_std"]) for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    x = np.arange(len(rows))
    ax.bar(x, means, yerr=stds, capsize=4, color=color, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    if log_y:
        ax.set_yscale("log")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_transfer_plot(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    labels = [str(row["run_name"]).replace("mnist_bp_amp_", "") for row in rows]
    forward = np.asarray([finite_float(row["F_transfer_mean"]) for row in rows], dtype=np.float64)
    feedback = np.asarray([finite_float(row["B_transfer_mean"]) for row in rows], dtype=np.float64)
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, forward, width=width, label="hidden to output", color="#4c78a8")
    ax.bar(x + width / 2, feedback, width=width, label="output to hidden", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("normalized transfer Frobenius norm")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_v_over_i_plot(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    x = np.asarray([finite_float(row["v_over_i"]) for row in rows], dtype=np.float64)
    y = np.asarray([finite_float(row["feedforwardness_mean"]) for row in rows], dtype=np.float64)
    labels = [str(row["run_name"]).replace("mnist_bp_amp_", "") for row in rows]
    order = np.argsort(x)
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    ax.plot(x[order], y[order], color="#4c78a8", alpha=0.5)
    ax.scatter(x, y, s=54, color="#4c78a8")
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("voltage_amp / current_amp")
    ax.set_ylabel("feedforwardness")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_top_level_outputs(output_root: Path, run_rows: list[dict]) -> None:
    run_rows = sorted(run_rows, key=_summary_sort_key)
    write_csv(output_root / "block_directional_summary.csv", run_rows)
    by_amp = aggregate_by_amp(run_rows)
    write_csv(output_root / "block_directional_by_amp.csv", by_amp)
    make_bar_plot(
        by_amp,
        "feedforwardness",
        "LPW sample feedforwardness",
        output_root / "feedforwardness_by_amp.png",
        color="#4c78a8",
    )
    make_bar_plot(
        by_amp,
        "feedback_fraction",
        "LPW sample feedback fraction",
        output_root / "feedback_fraction_by_amp.png",
        color="#f58518",
    )
    make_bar_plot(
        by_amp,
        "loop_rho",
        "LPW sample closed-loop spectral radius",
        output_root / "loop_rho_by_amp.png",
        color="#54a24b",
    )
    make_bar_plot(
        by_amp,
        "loop_correction_fraction",
        "Schur loop correction / output block",
        output_root / "loop_correction_fraction_by_amp.png",
        color="#b279a2",
    )
    make_transfer_plot(by_amp, output_root / "forward_backward_transfer_by_amp.png")
    make_v_over_i_plot(by_amp, output_root / "feedforwardness_vs_v_over_i.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze hidden-output Hessian block directionality for MNIST DRN-XS amplification runs."
    )
    parser.add_argument("--hessian-root", type=Path, default=DEFAULT_HESSIAN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", action="append", choices=RUN_ORDER)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rerun-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.hessian_root = args.hessian_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = selected_summary_rows(args)
    print(f"[block] hessian_root={args.hessian_root}")
    print(f"[block] output_root={args.output_root}")
    print(f"[block] selected_runs={len(rows)}")
    run_summaries = [analyze_run(row, args) for row in rows]
    write_top_level_outputs(args.output_root, run_summaries)
    print(f"[done] wrote {args.output_root / 'block_directional_by_amp.csv'}")


if __name__ == "__main__":
    main()
