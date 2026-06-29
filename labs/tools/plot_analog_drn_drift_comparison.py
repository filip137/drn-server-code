#!/usr/bin/env python3
"""Plot the MNIST analog-crossbar vs DRN drift comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "results" / "analog_drn_neck_to_neck" / "analog50_vs_drn_summary.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "analog_drn_neck_to_neck" / "analog_drn_drift_comparison.png"

TIME_LABELS = {
    2_592_000.0: "30 days",
    31_536_000.0: "1 year",
}

SERIES = [
    {
        "key": "analog_none",
        "label": "Analog 784-50-10\nno comp",
        "family": "Analog crossbar + digital ReLU",
        "mode": "none",
        "color": "#c2410c",
    },
    {
        "key": "analog_per_layer",
        "label": "Analog 784-50-10\nper-layer",
        "family": "Analog crossbar + digital ReLU",
        "mode": "per_layer",
        "color": "#2563eb",
    },
    {
        "key": "drn",
        "label": "DRN 784 x 100 x 10\nno comp",
        "family": "DRN",
        "mode": "perfect_diode, no explicit compensation",
        "color": "#059669",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--svg",
        type=Path,
        default=None,
        help="Optional SVG output path. Defaults to output with .svg suffix.",
    )
    return parser.parse_args()


def _time_key(value: str) -> float:
    return float(value)


def load_points(input_csv: Path) -> dict[tuple[str, float], dict[str, float]]:
    points: dict[tuple[str, float], dict[str, float]] = {}
    with input_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            time_s = _time_key(row["time_s"])
            for series in SERIES:
                if row["family"] == series["family"] and row["mode"] == series["mode"]:
                    points[(series["key"], time_s)] = {
                        "accuracy_pct": 100.0 * float(row["drifted_accuracy"]),
                        "drop_pp": 100.0 * float(row["accuracy_drop"]),
                        "kl": float(row["kl_clean_to_drifted"]),
                    }
    expected = {
        (series["key"], time_s)
        for series in SERIES
        for time_s in TIME_LABELS
    }
    missing = sorted(expected - set(points))
    if missing:
        missing_text = ", ".join(f"{series_key}@{time_s:g}s" for series_key, time_s in missing)
        raise ValueError(f"Expected rows for all plotted series/times; missing {missing_text}")
    return points


def annotate_bars(ax: plt.Axes, bars, values: list[float], metric: str) -> None:
    for bar, value in zip(bars, values):
        if metric == "kl":
            label = f"{value:.3g}"
            offset = 1.10
            y = value * offset
            va = "bottom"
        elif metric == "accuracy":
            label = f"{value:.2f}"
            y = value + 0.015
            va = "bottom"
        else:
            label = f"{value:.2f}"
            y = value + 0.015
            va = "bottom"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va=va,
            fontsize=8,
            rotation=0,
        )


def plot(points: dict[tuple[str, float], dict[str, float]], output: Path, svg: Path) -> None:
    time_values = list(TIME_LABELS)
    group_x = list(range(len(time_values)))
    width = 0.24
    offsets = [-width, 0.0, width]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), constrained_layout=False)
    fig.suptitle("MNIST drift comparison: analog crossbar ReLU vs DRN", fontsize=14, fontweight="bold")

    panels = [
        ("accuracy_pct", "Drifted accuracy (%)", "accuracy"),
        ("drop_pp", "Accuracy drop (pp)", "drop"),
        ("kl", "KL(clean || drifted)", "kl"),
    ]

    for ax, (field, ylabel, metric) in zip(axes, panels):
        for offset, series in zip(offsets, SERIES):
            values = [points[(series["key"], time_s)][field] for time_s in time_values]
            bars = ax.bar(
                [x + offset for x in group_x],
                values,
                width=width,
                color=series["color"],
                label=series["label"],
                edgecolor="#111827",
                linewidth=0.4,
            )
            annotate_bars(ax, bars, values, metric)

        ax.set_xticks(group_x)
        ax.set_xticklabels([TIME_LABELS[time_s] for time_s in time_values])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylim(95.8, 97.0)
    axes[1].set_ylim(0.0, 0.75)
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-4, 8e-1)

    axes[0].legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01, 3.15, 0.2),
        mode="expand",
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.text(
        0.01,
        0.01,
        "Evaluation: 512 MNIST test examples, 5 drift seeds. KL axis is logarithmic.",
        fontsize=9,
        color="#374151",
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.78, wspace=0.34)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    fig.savefig(svg)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    svg = args.svg if args.svg is not None else args.output.with_suffix(".svg")
    points = load_points(args.input_csv)
    plot(points, args.output, svg)
    print(f"Wrote {args.output}")
    print(f"Wrote {svg}")


if __name__ == "__main__":
    main()
