#!/usr/bin/env python3
"""Plot dense time curves for analog-crossbar and DRN MNIST drift."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "analog_drn_neck_to_neck"
DEFAULT_ANALOG_SUMMARY = DEFAULT_RESULTS_DIR / "analog_784_50_10_512_dense_times" / "summary.csv"
DEFAULT_DRN_SUMMARY = DEFAULT_RESULTS_DIR / "drn_perfect_v4c1_t0_1000_512_dense_times" / "summary_by_model_time.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "dense_time_plots"

SECONDS_PER_DAY = 86_400.0
KL_FLOOR = 1e-8

SERIES_STYLE = {
    "analog_none": {
        "label": "Analog 784-50-10, no comp",
        "color": "#c2410c",
        "marker": "o",
    },
    "analog_per_layer": {
        "label": "Analog 784-50-10, per-layer",
        "color": "#2563eb",
        "marker": "s",
    },
    "drn": {
        "label": "DRN 784 x 100 x 10, no comp",
        "color": "#059669",
        "marker": "^",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analog-summary", type=Path, default=DEFAULT_ANALOG_SUMMARY)
    parser.add_argument("--drn-summary", type=Path, default=DEFAULT_DRN_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--drn-t0-s", type=float, default=1000.0)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def load_series(analog_summary: Path, drn_summary: Path, drn_t0_s: float) -> list[dict]:
    series = []
    analog_rows = _read_csv(analog_summary)
    for key, mode in (("analog_none", "none"), ("analog_per_layer", "per_layer")):
        rows = sorted(
            [row for row in analog_rows if row["compensation"] == mode],
            key=lambda row: _float(row, "time_s"),
        )
        style = SERIES_STYLE[key]
        series.append(
            {
                "key": key,
                "label": style["label"],
                "color": style["color"],
                "marker": style["marker"],
                "time_s": np.asarray([_float(row, "time_s") for row in rows], dtype=float),
                "accuracy": np.asarray([100.0 * _float(row, "accuracy_mean") for row in rows], dtype=float),
                "accuracy_std": np.asarray([100.0 * _float(row, "accuracy_std") for row in rows], dtype=float),
                "kl": np.asarray([_float(row, "kl_mean") for row in rows], dtype=float),
            }
        )

    drn_rows = sorted(_read_csv(drn_summary), key=lambda row: _float(row, "t_inference"))
    style = SERIES_STYLE["drn"]
    series.append(
        {
            "key": "drn",
            "label": style["label"],
            "color": style["color"],
            "marker": style["marker"],
            "time_s": np.asarray([_float(row, "t_inference") + drn_t0_s for row in drn_rows], dtype=float),
            "accuracy": np.asarray([100.0 * _float(row, "mean_drifted_accuracy") for row in drn_rows], dtype=float),
            "accuracy_std": np.asarray([100.0 * _float(row, "std_drifted_accuracy") for row in drn_rows], dtype=float),
            "kl": np.asarray([_float(row, "mean_kl_clean_to_drifted") for row in drn_rows], dtype=float),
        }
    )
    return series


def write_combined_csv(series: list[dict], output_dir: Path) -> Path:
    output_path = output_dir / "analog_drn_dense_time_curves.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["series", "time_s", "time_days", "accuracy_pct", "accuracy_std_pp", "kl"],
        )
        writer.writeheader()
        for item in series:
            for time_s, acc, acc_std, kl in zip(
                item["time_s"],
                item["accuracy"],
                item["accuracy_std"],
                item["kl"],
            ):
                writer.writerow(
                    {
                        "series": item["label"],
                        "time_s": float(time_s),
                        "time_days": float(time_s / SECONDS_PER_DAY),
                        "accuracy_pct": float(acc),
                        "accuracy_std_pp": float(acc_std),
                        "kl": float(kl),
                    }
                )
    return output_path


def _set_time_axis(ax: plt.Axes, min_time_s: float, max_time_s: float) -> None:
    ticks = [
        (1_000.0, "1000s"),
        (86_400.0, "1d"),
        (604_800.0, "1w"),
        (2_592_000.0, "30d"),
        (7_776_000.0, "90d"),
        (15_552_000.0, "180d"),
        (31_536_000.0, "1y"),
    ]
    selected = [(time_s / SECONDS_PER_DAY, label) for time_s, label in ticks if min_time_s <= time_s <= max_time_s]
    ax.set_xscale("log")
    ax.set_xticks([value for value, _label in selected])
    ax.set_xticklabels([label for _value, label in selected])
    ax.set_xlabel("Time since programming (log scale)")


def plot_accuracy(series: list[dict], output_dir: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    all_time = np.concatenate([item["time_s"] for item in series])
    all_acc = np.concatenate([item["accuracy"] for item in series])
    for item in series:
        x = item["time_s"] / SECONDS_PER_DAY
        y = item["accuracy"]
        ax.plot(
            x,
            y,
            color=item["color"],
            marker=item["marker"],
            linewidth=2.0,
            markersize=5.0,
            label=item["label"],
        )

    _set_time_axis(ax, float(all_time.min()), float(all_time.max()))
    pad = 0.12
    ax.set_ylim(float(all_acc.min()) - pad, float(all_acc.max()) + pad)
    ax.set_ylabel("Drifted accuracy (%)")
    ax.set_title("MNIST drift: accuracy vs time")
    ax.grid(True, which="both", axis="both", color="#d1d5db", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    png = output_dir / "analog_drn_accuracy_vs_time.png"
    svg = output_dir / "analog_drn_accuracy_vs_time.svg"
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def plot_kl(series: list[dict], output_dir: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    all_time = np.concatenate([item["time_s"] for item in series])
    all_kl = np.concatenate([np.maximum(item["kl"], KL_FLOOR) for item in series])
    for item in series:
        x = item["time_s"] / SECONDS_PER_DAY
        y = np.maximum(item["kl"], KL_FLOOR)
        ax.plot(
            x,
            y,
            color=item["color"],
            marker=item["marker"],
            linewidth=2.0,
            markersize=5.0,
            label=item["label"],
        )

    _set_time_axis(ax, float(all_time.min()), float(all_time.max()))
    ax.set_yscale("log")
    ax.set_ylim(max(KL_FLOOR, float(all_kl.min()) * 0.6), float(all_kl.max()) * 1.8)
    ax.set_ylabel("KL(clean || drifted)")
    ax.set_title("MNIST drift: KL divergence vs time")
    ax.grid(True, which="both", axis="both", color="#d1d5db", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    png = output_dir / "analog_drn_kl_vs_time.png"
    svg = output_dir / "analog_drn_kl_vs_time.svg"
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series = load_series(args.analog_summary, args.drn_summary, args.drn_t0_s)
    combined = write_combined_csv(series, args.output_dir)
    acc_png, acc_svg = plot_accuracy(series, args.output_dir)
    kl_png, kl_svg = plot_kl(series, args.output_dir)
    print(f"Wrote {combined}")
    print(f"Wrote {acc_png}")
    print(f"Wrote {acc_svg}")
    print(f"Wrote {kl_png}")
    print(f"Wrote {kl_svg}")


if __name__ == "__main__":
    main()
