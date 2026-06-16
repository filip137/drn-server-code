#!/usr/bin/env python3
"""Compare MNIST BP write-noise sigma responses across inference iteration counts."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


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
ITER_COLORS = {
    "4": "#4c78a8",
    "8": "#f58518",
    "16": "#54a24b",
}


def finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def parse_labeled_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected ITER=PATH for --input-root, got {value!r}."
        )
    label, root = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(
            f"Expected non-empty ITER label for --input-root, got {value!r}."
        )
    return label, Path(root).expanduser().resolve()


def read_summary(label: str, root: Path) -> list[dict[str, object]]:
    path = root / "accuracy_vs_sigma_by_amp.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected accuracy summary at {path}.")
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out: dict[str, object] = dict(row)
            out["iteration_label"] = label
            out["source_root"] = str(root)
            out["sigma"] = finite_float(row["sigma"])
            out["clean_accuracy_mean"] = finite_float(row["clean_accuracy_mean"])
            out["noisy_accuracy_mean"] = finite_float(row["noisy_accuracy_mean"])
            out["noisy_accuracy_std"] = finite_float(row["noisy_accuracy_std"])
            out["accuracy_drop_mean"] = finite_float(row["accuracy_drop_mean"])
            out["accuracy_drop_std"] = finite_float(row["accuracy_drop_std"])
            rows.append(out)
    return rows


def write_combined_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "iteration_label",
        "run_name",
        "voltage_amp",
        "current_amp",
        "sigma",
        "num_rows",
        "num_training_seeds",
        "num_noise_seeds",
        "clean_accuracy_mean",
        "noisy_accuracy_mean",
        "noisy_accuracy_std",
        "accuracy_drop_mean",
        "accuracy_drop_std",
        "source_root",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sorted_iteration_labels(rows: list[dict[str, object]]) -> list[str]:
    labels = sorted({str(row["iteration_label"]) for row in rows}, key=lambda x: (finite_float(x), x))
    return labels


def plot_grid(
    rows: list[dict[str, object]],
    *,
    y_key: str,
    y_std_key: str,
    ylabel: str,
    output_path: Path,
) -> None:
    labels = _sorted_iteration_labels(rows)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), constrained_layout=True, sharex=True)
    axes_flat = list(axes.ravel())
    for axis, run_name in zip(axes_flat, RUN_ORDER):
        run_rows = [row for row in rows if row["run_name"] == run_name]
        for label in labels:
            series = [row for row in run_rows if str(row["iteration_label"]) == label]
            if not series:
                continue
            x = np.asarray([float(row["sigma"]) for row in series], dtype=np.float64)
            y = np.asarray([float(row[y_key]) for row in series], dtype=np.float64)
            y_std = np.asarray([float(row[y_std_key]) for row in series], dtype=np.float64)
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            y_std = y_std[order]
            color = ITER_COLORS.get(label)
            axis.plot(x, 100.0 * y, marker="o", linewidth=1.8, markersize=3.5, label=f"{label} iter", color=color)
            axis.fill_between(
                x,
                100.0 * (y - y_std),
                100.0 * (y + y_std),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        axis.set_title(RUN_LABELS.get(run_name, run_name))
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("lognormal write-noise sigma")
        axis.set_ylabel(ylabel)
    for axis in axes_flat[len(RUN_ORDER) :]:
        axis.axis("off")
    axes_flat[0].legend(frameon=False, title="Inference")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def interpolated_threshold(sigmas: np.ndarray, drops: np.ndarray, threshold: float) -> float:
    order = np.argsort(sigmas)
    sigmas = sigmas[order]
    drops = drops[order]
    if drops.size == 0 or not np.any(np.isfinite(drops)):
        return math.nan
    finite = np.isfinite(sigmas) & np.isfinite(drops)
    sigmas = sigmas[finite]
    drops = drops[finite]
    if sigmas.size == 0:
        return math.nan
    if drops[0] >= threshold:
        return float(sigmas[0])
    for idx in range(1, sigmas.size):
        if drops[idx] >= threshold:
            x0, x1 = sigmas[idx - 1], sigmas[idx]
            y0, y1 = drops[idx - 1], drops[idx]
            if y1 == y0:
                return float(x1)
            return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))
    return math.nan


def write_thresholds(path: Path, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    labels = _sorted_iteration_labels(rows)
    out: list[dict[str, object]] = []
    for run_name in RUN_ORDER:
        for label in labels:
            series = [row for row in rows if row["run_name"] == run_name and str(row["iteration_label"]) == label]
            sigmas = np.asarray([float(row["sigma"]) for row in series], dtype=np.float64)
            drops = np.asarray([float(row["accuracy_drop_mean"]) for row in series], dtype=np.float64)
            out.append(
                {
                    "iteration_label": label,
                    "run_name": run_name,
                    "sigma_1pct": interpolated_threshold(sigmas, drops, 0.01),
                    "sigma_5pct": interpolated_threshold(sigmas, drops, 0.05),
                    "sigma_10pct": interpolated_threshold(sigmas, drops, 0.10),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration_label", "run_name", "sigma_1pct", "sigma_5pct", "sigma_10pct"])
        writer.writeheader()
        writer.writerows(out)
    return out


def plot_thresholds(rows: list[dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for run_name in RUN_ORDER:
        series = [row for row in rows if row["run_name"] == run_name]
        x = np.asarray([finite_float(row["iteration_label"]) for row in series], dtype=np.float64)
        y = np.asarray([finite_float(row["sigma_1pct"]) for row in series], dtype=np.float64)
        order = np.argsort(x)
        ax.plot(x[order], y[order], marker="o", linewidth=2.0, label=RUN_LABELS.get(run_name, run_name))
    ax.set_xlabel("inference iterations")
    ax.set_ylabel("sigma at 1% accuracy drop")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Amplification", frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        action="append",
        type=parse_labeled_root,
        required=True,
        help="Labeled summary root in ITER=PATH format. Repeat for 4=..., 8=..., 16=....",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    rows: list[dict[str, object]] = []
    for label, root in args.input_root:
        rows.extend(read_summary(label, root))
    rows.sort(
        key=lambda row: (
            finite_float(row["iteration_label"]),
            RUN_ORDER.index(row["run_name"]) if row["run_name"] in RUN_ORDER else len(RUN_ORDER),
            float(row["sigma"]),
        )
    )
    write_combined_csv(output_dir / "accuracy_vs_sigma_by_amp_iteration_comparison.csv", rows)
    plot_grid(
        rows,
        y_key="noisy_accuracy_mean",
        y_std_key="noisy_accuracy_std",
        ylabel="test accuracy (%)",
        output_path=output_dir / "accuracy_vs_sigma_by_amp_iteration_grid.png",
    )
    plot_grid(
        rows,
        y_key="accuracy_drop_mean",
        y_std_key="accuracy_drop_std",
        ylabel="accuracy drop from clean (%)",
        output_path=output_dir / "accuracy_drop_vs_sigma_by_amp_iteration_grid.png",
    )
    thresholds = write_thresholds(output_dir / "sigma_drop_thresholds_by_iteration_amp.csv", rows)
    plot_thresholds(thresholds, output_dir / "sigma_1pct_vs_iterations_by_amp.png")
    print(f"[iteration-comparison] wrote outputs under {output_dir}")


if __name__ == "__main__":
    main()
