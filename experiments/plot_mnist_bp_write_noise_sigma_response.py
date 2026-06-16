#!/usr/bin/env python3
"""Plot MNIST BP write-noise accuracy as a function of lognormal sigma."""

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
SUMMARY_COLUMNS = [
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
    "noisy_accuracy_p10",
    "noisy_accuracy_p50",
    "noisy_accuracy_p90",
    "accuracy_drop_mean",
    "accuracy_drop_std",
    "test_loss_mean",
    "test_loss_delta_mean",
    "test_loss_delta_std",
    "mean_logit_margin",
    "mean_fraction_saturated_nodes",
]


def finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def read_raw_rows(input_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(input_root.glob("raw_results*.csv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"Expected raw_results*.csv files under {input_root}.")
    return rows


def stats(values: list[float]) -> dict[str, float]:
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


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    clean_accuracy_by_model: dict[tuple[str, str], float] = {}
    clean_loss_by_model: dict[tuple[str, str], float] = {}
    sigma_zero_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if finite_float(row["sigma"]) == 0.0:
            sigma_zero_groups[(row["run_name"], row["training_seed"])].append(row)
    for key, group in sigma_zero_groups.items():
        clean_accuracy_by_model[key] = stats([finite_float(row["noisy_accuracy"]) for row in group])["mean"]
        clean_loss_by_model[key] = stats([finite_float(row["test_loss"]) for row in group])["mean"]

    groups: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["run_name"], finite_float(row["sigma"]))].append(row)

    out: list[dict[str, object]] = []
    for (run_name, sigma), group in sorted(
        groups.items(),
        key=lambda item: (
            RUN_ORDER.index(item[0][0]) if item[0][0] in RUN_ORDER else len(RUN_ORDER),
            item[0][1],
        ),
    ):
        noisy = [finite_float(row["noisy_accuracy"]) for row in group]
        clean = [
            clean_accuracy_by_model.get(
                (row["run_name"], row["training_seed"]),
                finite_float(row["clean_accuracy"]),
            )
            for row in group
        ]
        drop = [clean_i - noisy_i for clean_i, noisy_i in zip(clean, noisy)]
        loss = [finite_float(row["test_loss"]) for row in group]
        clean_loss = [
            clean_loss_by_model.get(
                (row["run_name"], row["training_seed"]),
                finite_float(row["test_loss"]),
            )
            for row in group
        ]
        loss_delta = [loss_i - clean_i for clean_i, loss_i in zip(clean_loss, loss)]
        margin = [finite_float(row["mean_logit_margin"]) for row in group]
        saturation = [finite_float(row["fraction_saturated_nodes"]) for row in group]
        noisy_stats = stats(noisy)
        drop_stats = stats(drop)
        loss_stats = stats(loss)
        loss_delta_stats = stats(loss_delta)
        margin_stats = stats(margin)
        saturation_stats = stats(saturation)
        out.append(
            {
                "run_name": run_name,
                "voltage_amp": group[0]["voltage_amp"],
                "current_amp": group[0]["current_amp"],
                "sigma": sigma,
                "num_rows": len(group),
                "num_training_seeds": len({row["training_seed"] for row in group}),
                "num_noise_seeds": len({row["noise_seed"] for row in group}),
                "clean_accuracy_mean": stats(clean)["mean"],
                "noisy_accuracy_mean": noisy_stats["mean"],
                "noisy_accuracy_std": noisy_stats["std"],
                "noisy_accuracy_p10": noisy_stats["p10"],
                "noisy_accuracy_p50": noisy_stats["p50"],
                "noisy_accuracy_p90": noisy_stats["p90"],
                "accuracy_drop_mean": drop_stats["mean"],
                "accuracy_drop_std": drop_stats["std"],
                "test_loss_mean": loss_stats["mean"],
                "test_loss_delta_mean": loss_delta_stats["mean"],
                "test_loss_delta_std": loss_delta_stats["std"],
                "mean_logit_margin": margin_stats["mean"],
                "mean_fraction_saturated_nodes": saturation_stats["mean"],
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    rows: list[dict[str, object]],
    *,
    y_key: str,
    y_std_key: str | None,
    ylabel: str,
    output_path: Path,
    y_scale: float = 100.0,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for run_name in RUN_ORDER:
        run_rows = [row for row in rows if row["run_name"] == run_name]
        if not run_rows:
            continue
        x = np.asarray([float(row["sigma"]) for row in run_rows], dtype=np.float64)
        y = np.asarray([float(row[y_key]) for row in run_rows], dtype=np.float64)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        label = RUN_LABELS.get(run_name, run_name)
        color = RUN_COLORS.get(run_name)
        ax.plot(x, y_scale * y, marker="o", linewidth=2.0, markersize=4.0, label=label, color=color)
        if y_std_key is not None:
            y_std = np.asarray([float(row[y_std_key]) for row in run_rows], dtype=np.float64)[order]
            ax.fill_between(x, y_scale * (y - y_std), y_scale * (y + y_std), color=color, alpha=0.14, linewidth=0)
    ax.set_xlabel("lognormal write-noise sigma")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Amplification", frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else input_root
    rows = aggregate(read_raw_rows(input_root))
    summary_path = output_dir / "accuracy_vs_sigma_by_amp.csv"
    write_csv(summary_path, rows)
    plot_metric(
        rows,
        y_key="noisy_accuracy_mean",
        y_std_key="noisy_accuracy_std",
        ylabel="test accuracy (%)",
        output_path=output_dir / "accuracy_vs_sigma.png",
    )
    plot_metric(
        rows,
        y_key="accuracy_drop_mean",
        y_std_key="accuracy_drop_std",
        ylabel="accuracy drop from clean (%)",
        output_path=output_dir / "accuracy_drop_vs_sigma.png",
    )
    plot_metric(
        rows,
        y_key="test_loss_delta_mean",
        y_std_key="test_loss_delta_std",
        ylabel="test loss increase from clean",
        output_path=output_dir / "loss_delta_vs_sigma.png",
        y_scale=1.0,
    )
    print(f"[sigma-response] wrote {summary_path}")
    print(f"[sigma-response] wrote {output_dir / 'accuracy_vs_sigma.png'}")
    print(f"[sigma-response] wrote {output_dir / 'accuracy_drop_vs_sigma.png'}")
    print(f"[sigma-response] wrote {output_dir / 'loss_delta_vs_sigma.png'}")


if __name__ == "__main__":
    main()
