#!/usr/bin/env python3
"""Plot selected 100-epoch MNIST diode training comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path("/home/filip/server_code")
MNIST_ROOT = PROJECT_ROOT / "simulation_results" / "experiments_labs" / "paper_mnist_exp"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "labs" / "cases" / "presentation_mnist_diode_100epoch"
DEFAULT_STEM = "mnist_ideal10_ideal20_double_shockley20_100epoch_accuracy"


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    label: str
    color: str
    paths: tuple[Path, ...]


SERIES = (
    SeriesSpec(
        key="ideal_10",
        label="Ideal diode, 10 outputs",
        color="#4c78a8",
        paths=(
            MNIST_ROOT
            / "perfect_diode_output_dim_probe_20260514"
            / "runs"
            / "output_dim_10"
            / "seed_0"
            / "20260514-202826_train-stats"
            / "time_series.pkl",
            MNIST_ROOT
            / "perfect_diode_output_dim_probe_20260514"
            / "runs"
            / "output_dim_10"
            / "seed_1"
            / "20260521-041053_train-stats"
            / "time_series.pkl",
            MNIST_ROOT
            / "perfect_diode_output_dim_probe_20260514"
            / "runs"
            / "output_dim_10"
            / "seed_2"
            / "20260521-041053_train-stats"
            / "time_series.pkl",
            MNIST_ROOT
            / "perfect_diode_output_dim_probe_20260514"
            / "runs"
            / "output_dim_10"
            / "seed_3"
            / "20260521-041053_train-stats"
            / "time_series.pkl",
        ),
    ),
    SeriesSpec(
        key="ideal_20",
        label="Ideal diode, 20 outputs",
        color="#59a14f",
        paths=(
            MNIST_ROOT
            / "perfect_diode_output_dim_probe_20260514"
            / "runs"
            / "output_dim_20"
            / "seed_0"
            / "20260514-204813_train-stats"
            / "time_series.pkl",
            MNIST_ROOT
            / "perfect_diode_output_dim_probe_20260514"
            / "runs"
            / "output_dim_20"
            / "seed_1"
            / "20260514-222006_train-stats"
            / "time_series.pkl",
            MNIST_ROOT
            / "perfect_diode_output_dim_probe_20260514"
            / "runs"
            / "output_dim_20"
            / "seed_2"
            / "20260514-222006_train-stats"
            / "time_series.pkl",
            MNIST_ROOT
            / "perfect_diode_output_dim_probe_20260514"
            / "runs"
            / "output_dim_20"
            / "seed_3"
            / "20260514-222006_train-stats"
            / "time_series.pkl",
        ),
    ),
    SeriesSpec(
        key="double_shockley_20",
        label="Double Shockley diode, 20 outputs",
        color="#e15759",
        paths=(
            MNIST_ROOT
            / "voltage_amp1_current_amp1"
            / "runs"
            / "20260325-154905_train-stats"
            / "time_series.pkl",
            MNIST_ROOT
            / "voltage_amp1_current_amp1"
            / "runs"
            / "20260325-192519_train-stats"
            / "time_series.pkl",
            MNIST_ROOT
            / "voltage_amp1_current_amp1"
            / "runs"
            / "20260326-021631_train-stats"
            / "time_series.pkl",
            MNIST_ROOT
            / "voltage_amp1_current_amp1"
            / "runs"
            / "20260325-225016_train-stats"
            / "time_series.pkl",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MNIST 100-epoch accuracy for selected diode families."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help=f"Output filename stem. Default: {DEFAULT_STEM}",
    )
    parser.add_argument(
        "--title",
        default="MNIST 100-epoch training",
        help="Plot title.",
    )
    parser.add_argument("--dpi", type=int, default=240, help="PNG DPI.")
    return parser.parse_args()


def load_test_accuracy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Expected time_series.pkl. Provided value: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected time_series.pkl dict. Provided value type: {type(payload)!r}")
    if "Error/test" not in payload:
        raise KeyError(f"Expected Error/test in time_series.pkl. Provided value: {path}")
    error = np.asarray(payload["Error/test"], dtype=float)
    if error.ndim != 1 or error.size == 0:
        raise ValueError(f"Expected non-empty 1D Error/test. Provided value: {path}")
    return 100.0 - error


def aggregate(spec: SeriesSpec) -> dict[str, object]:
    curves = [load_test_accuracy(path) for path in spec.paths]
    min_len = min(curve.size for curve in curves)
    if min_len < 100:
        raise ValueError(f"Expected 100 epochs for {spec.key}. Provided value: {min_len}")
    stacked = np.vstack([curve[:100] for curve in curves])
    return {
        "spec": spec,
        "epochs": np.arange(1, 101),
        "mean": stacked.mean(axis=0),
        "min": stacked.min(axis=0),
        "max": stacked.max(axis=0),
        "stacked": stacked,
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "series",
                "label",
                "epoch",
                "mean_accuracy_percent",
                "min_accuracy_percent",
                "max_accuracy_percent",
            ],
        )
        writer.writeheader()
        for row in rows:
            spec = row["spec"]
            assert isinstance(spec, SeriesSpec)
            epochs = row["epochs"]
            mean = row["mean"]
            min_values = row["min"]
            max_values = row["max"]
            assert isinstance(epochs, np.ndarray)
            assert isinstance(mean, np.ndarray)
            assert isinstance(min_values, np.ndarray)
            assert isinstance(max_values, np.ndarray)
            for epoch, mean_value, min_value, max_value in zip(epochs, mean, min_values, max_values):
                writer.writerow(
                    {
                        "series": spec.key,
                        "label": spec.label,
                        "epoch": int(epoch),
                        "mean_accuracy_percent": float(mean_value),
                        "min_accuracy_percent": float(min_value),
                        "max_accuracy_percent": float(max_value),
                    }
                )


def write_summary(rows: list[dict[str, object]], output_paths: list[Path], csv_path: Path, path: Path) -> None:
    payload = {
        "title": "MNIST 100-epoch selected diode comparison",
        "metric": "test accuracy (%)",
        "outputs": [str(path) for path in output_paths],
        "csv": str(csv_path),
        "series": [],
    }
    for row in rows:
        spec = row["spec"]
        mean = row["mean"]
        min_values = row["min"]
        max_values = row["max"]
        assert isinstance(spec, SeriesSpec)
        assert isinstance(mean, np.ndarray)
        assert isinstance(min_values, np.ndarray)
        assert isinstance(max_values, np.ndarray)
        payload["series"].append(
            {
                "key": spec.key,
                "label": spec.label,
                "run_count": len(spec.paths),
                "epochs": int(mean.size),
                "final_mean_accuracy_percent": float(mean[-1]),
                "best_mean_accuracy_percent": float(mean.max()),
                "best_mean_epoch": int(mean.argmax() + 1),
                "final_min_accuracy_percent": float(min_values[-1]),
                "final_max_accuracy_percent": float(max_values[-1]),
                "source_paths": [str(source) for source in spec.paths],
            }
        )
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_plot(rows: list[dict[str, object]], path: Path, title: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    for row in rows:
        spec = row["spec"]
        epochs = row["epochs"]
        mean = row["mean"]
        min_values = row["min"]
        max_values = row["max"]
        assert isinstance(spec, SeriesSpec)
        assert isinstance(epochs, np.ndarray)
        assert isinstance(mean, np.ndarray)
        assert isinstance(min_values, np.ndarray)
        assert isinstance(max_values, np.ndarray)
        ax.plot(epochs, mean, color=spec.color, linewidth=2.2, label=spec.label)
        ax.fill_between(epochs, min_values, max_values, color=spec.color, alpha=0.16, linewidth=0)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(1, 100)
    ax.set_ylim(94.0, 98.0)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [aggregate(spec) for spec in SERIES]

    csv_path = args.output_dir / f"{args.stem}.csv"
    json_path = args.output_dir / f"{args.stem}.json"
    png_path = args.output_dir / f"{args.stem}.png"
    pdf_path = args.output_dir / f"{args.stem}.pdf"
    svg_path = args.output_dir / f"{args.stem}.svg"

    write_csv(rows, csv_path)
    for output_path in (png_path, pdf_path, svg_path):
        write_plot(rows, output_path, args.title, args.dpi)
    write_summary(rows, [png_path, pdf_path, svg_path], csv_path, json_path)

    print(f"Wrote PNG: {png_path}")
    print(f"Wrote PDF: {pdf_path}")
    print(f"Wrote SVG: {svg_path}")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    for row in rows:
        spec = row["spec"]
        mean = row["mean"]
        assert isinstance(spec, SeriesSpec)
        assert isinstance(mean, np.ndarray)
        print(
            f"{spec.label}: final mean={float(mean[-1]):.4g}%, "
            f"best mean={float(mean.max()):.4g}% at epoch {int(mean.argmax() + 1)}"
        )


if __name__ == "__main__":
    main()
