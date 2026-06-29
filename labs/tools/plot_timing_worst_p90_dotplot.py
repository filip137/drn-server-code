#!/usr/bin/env python3
"""Nine-point worst-case CD-vs-SPICE P90 dot plot for timing runs."""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogFormatterMathtext  # noqa: E402


PROJECT_ROOT = Path("/home/filip/server_code")
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "labs"
    / "figures_for_paper_digits"
    / "timings"
    / "selected_for_paper"
    / "tables"
    / "table1_accuracy.csv"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "labs"
    / "figures_for_paper_digits"
    / "timings"
    / "selected_for_paper"
    / "figure_inputs"
)
DEFAULT_STEM = "cd_spice_worst_p90_by_family_depth"

FAMILY_ORDER = [
    "single_diode_exponential",
    "double_diode_exponential",
    "experimental",
]
FAMILY_LABELS = {
    "single_diode_exponential": "Single Shockley",
    "double_diode_exponential": "Double Shockley",
    "experimental": "PWL",
}


@dataclass(frozen=True)
class AccuracyRow:
    nonlinearity: str
    hidden_layers: int
    hidden_size: int
    p90: float


@dataclass(frozen=True)
class WorstCaseRow:
    nonlinearity: str
    nonlinearity_label: str
    hidden_layers: int
    worst_p90: float
    worst_hidden_size: int
    matched_widths: tuple[int, ...]
    threshold: float

    @property
    def matched_width_count(self) -> int:
        return len(self.matched_widths)

    @property
    def passes_threshold(self) -> bool:
        return self.worst_p90 <= self.threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the largest P90 relative L1 voltage error across matched hidden "
            "widths for each nonlinearity and hidden-layer depth."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Timing-run accuracy table. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUT_DIR / f"{DEFAULT_STEM}.csv",
        help="Nine-row source CSV written by this script.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DEFAULT_OUT_DIR / f"{DEFAULT_STEM}.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DEFAULT_OUT_DIR / f"{DEFAULT_STEM}.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.1e-4,
        help="Claim threshold for the horizontal reference line.",
    )
    parser.add_argument(
        "--title",
        default="Worst case P90 Error",
        help="Figure title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="PNG DPI.",
    )
    return parser.parse_args()


def as_float(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def read_rows(path: Path) -> list[AccuracyRow]:
    if not path.exists():
        raise FileNotFoundError(f"Expected timing-run accuracy CSV. Provided value: {path}")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "nonlinearity",
            "hidden_layers",
            "hidden_size",
            "p90_relative_error",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(
                f"Expected columns {sorted(required)}. Missing from {path}: {missing_cols}"
            )

        rows: list[AccuracyRow] = []
        for raw in reader:
            p90 = as_float(raw["p90_relative_error"])
            if p90 is None:
                continue
            rows.append(
                AccuracyRow(
                    nonlinearity=raw["nonlinearity"],
                    hidden_layers=int(raw["hidden_layers"]),
                    hidden_size=int(raw["hidden_size"]),
                    p90=p90,
                )
            )

    if not rows:
        raise ValueError(f"Expected at least one finite p90_relative_error. Provided value: {path}")
    return rows


def summarize_worst_cases(rows: Iterable[AccuracyRow], threshold: float) -> list[WorstCaseRow]:
    grouped: dict[tuple[str, int], list[AccuracyRow]] = {}
    for row in rows:
        grouped.setdefault((row.nonlinearity, row.hidden_layers), []).append(row)

    summary: list[WorstCaseRow] = []
    for family in FAMILY_ORDER:
        for depth in (1, 2, 3):
            group = sorted(grouped.get((family, depth), []), key=lambda row: row.hidden_size)
            if not group:
                raise ValueError(
                    "Expected at least one matched width for "
                    f"{family} hidden_layers={depth}. Provided value: none"
                )
            worst = max(group, key=lambda row: row.p90)
            summary.append(
                WorstCaseRow(
                    nonlinearity=family,
                    nonlinearity_label=FAMILY_LABELS[family],
                    hidden_layers=depth,
                    worst_p90=worst.p90,
                    worst_hidden_size=worst.hidden_size,
                    matched_widths=tuple(row.hidden_size for row in group),
                    threshold=threshold,
                )
            )
    return summary


def write_summary_csv(rows: list[WorstCaseRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "nonlinearity",
                "nonlinearity_label",
                "hidden_layers",
                "worst_p90_relative_error",
                "worst_hidden_size",
                "matched_width_count",
                "matched_widths",
                "threshold",
                "passes_threshold",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "nonlinearity": row.nonlinearity,
                    "nonlinearity_label": row.nonlinearity_label,
                    "hidden_layers": row.hidden_layers,
                    "worst_p90_relative_error": row.worst_p90,
                    "worst_hidden_size": row.worst_hidden_size,
                    "matched_width_count": row.matched_width_count,
                    "matched_widths": " ".join(str(width) for width in row.matched_widths),
                    "threshold": row.threshold,
                    "passes_threshold": row.passes_threshold,
                }
            )


def write_plot(rows: list[WorstCaseRow], png_path: Path, pdf_path: Path | None, title: str, dpi: int) -> None:
    x_positions: list[int] = []
    values: list[float] = []
    colors: list[str] = []
    depth_labels: list[str] = []
    color_by_family = {
        "single_diode_exponential": "#1f77b4",
        "double_diode_exponential": "#d62728",
        "experimental": "#2ca02c",
    }

    for family_index, family in enumerate(FAMILY_ORDER):
        base = family_index * 4
        family_rows = [row for row in rows if row.nonlinearity == family]
        for row in sorted(family_rows, key=lambda item: item.hidden_layers):
            x_positions.append(base + row.hidden_layers)
            values.append(row.worst_p90)
            colors.append(color_by_family[family])
            depth_labels.append(f"{row.hidden_layers}H")

    max_value = max(max(values), rows[0].threshold)
    y_max = 10 ** math.ceil(math.log10(max_value * 1.25))
    y_min = 1e-6

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ax.scatter(
        x_positions,
        values,
        s=72,
        c=colors,
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )

    threshold = rows[0].threshold
    ax.axhline(threshold, color="#404040", linewidth=1.0, linestyle="--", zorder=1)
    ax.text(
        max(x_positions) + 0.25,
        threshold * 1.02,
        r"P90 error $\leq 1.1\times10^{-4}$",
        ha="right",
        va="bottom",
        fontsize=11.5,
        color="#303030",
    )

    for separator in (3.5, 7.5):
        ax.axvline(separator, color="#bbbbbb", linewidth=0.8, zorder=0)

    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_yticks([1e-6, 1e-5, 1e-4])
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_ylabel(r"P90 relative $L_1$ voltage error", fontsize=13)
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(depth_labels)
    ax.set_xlim(0.35, 11.65)
    ax.grid(True, axis="y", which="major", linestyle=":", linewidth=0.9, alpha=0.65)
    ax.grid(True, axis="y", which="minor", linestyle=":", linewidth=0.5, alpha=0.25)
    ax.tick_params(axis="both", labelsize=12)

    for family_index, family in enumerate(FAMILY_ORDER):
        center = family_index * 4 + 2
        ax.text(
            center,
            -0.18,
            FAMILY_LABELS[family],
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=12,
        )

    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.26)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi)
    if pdf_path is not None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.threshold <= 0:
        raise ValueError(f"Expected positive threshold. Provided value: {args.threshold}")

    rows = read_rows(args.input_csv)
    summary = summarize_worst_cases(rows, args.threshold)
    write_summary_csv(summary, args.output_csv)

    output_pdf = args.output_pdf if str(args.output_pdf).strip() else None
    write_plot(summary, args.output_png, output_pdf, args.title, args.dpi)

    max_row = max(summary, key=lambda row: row.worst_p90)
    print(f"Wrote source CSV: {args.output_csv}")
    print(f"Wrote PNG: {args.output_png}")
    if output_pdf is not None:
        print(f"Wrote PDF: {output_pdf}")
    print(
        "Max worst-case P90: "
        f"{max_row.worst_p90:.6g} "
        f"({max_row.nonlinearity_label}, {max_row.hidden_layers}H, "
        f"width {max_row.worst_hidden_size})"
    )


if __name__ == "__main__":
    main()
