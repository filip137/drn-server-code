#!/usr/bin/env python3
"""Log-log timing plot for SPICE vs coordinate descent across hidden sizes.

Expected input is the CSV produced by:
.../timings/double_diode_exponential/extracted_timings/combined_latest_by_hidden.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a log-log plot of SPICE timings vs coordinate descent timings "
            "against hidden size, with all hidden-layer counts in one figure."
        )
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        required=True,
        help=(
            "Path to combined_latest_by_hidden.csv "
            "(contains depth,width,coord_user_time_seconds,spice_*_seconds)."
        ),
    )
    parser.add_argument(
        "--spice-time-field",
        choices=("total", "simulation", "netlist"),
        default="total",
        help="Which SPICE timing field to plot from the combined CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spice_vs_coordinate_descent_loglog.png"),
        help="Output image path (PNG/SVG/PDF, based on extension).",
    )
    parser.add_argument(
        "--title",
        default="SPICE vs Coordinate Descent (log-log)",
        help="Plot title.",
    )
    parser.add_argument(
        "--x-label",
        default="Hidden size",
        help="X-axis label.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively after saving.",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Optional fixed lower Y limit (must be > 0 for log scale).",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional fixed upper Y limit (must be > 0 for log scale).",
    )
    return parser.parse_args()


def _as_float(value: str) -> Optional[float]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def load_rows(
    combined_csv: Path, spice_column: str
) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]]]:
    if not combined_csv.exists():
        raise FileNotFoundError(f"Combined CSV not found: {combined_csv}")

    coord_by_depth: Dict[int, List[Tuple[int, float]]] = {}
    spice_by_depth: Dict[int, List[Tuple[int, float]]] = {}

    with combined_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"depth", "width", "coord_user_time_seconds", spice_column}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(
                f"Missing required column(s) in {combined_csv}: {missing_cols}"
            )

        for row in reader:
            try:
                depth = int(row["depth"])
                width = int(row["width"])
            except (TypeError, ValueError):
                continue

            coord = _as_float(row["coord_user_time_seconds"])
            spice = _as_float(row[spice_column])

            if coord is not None and coord > 0:
                coord_by_depth.setdefault(depth, []).append((width, coord))
            if spice is not None and spice > 0:
                spice_by_depth.setdefault(depth, []).append((width, spice))

    for depth in coord_by_depth:
        coord_by_depth[depth].sort(key=lambda item: item[0])
    for depth in spice_by_depth:
        spice_by_depth[depth].sort(key=lambda item: item[0])

    return coord_by_depth, spice_by_depth


def main() -> None:
    args = parse_args()
    if args.y_min is not None and args.y_min <= 0:
        raise ValueError(f"--y-min must be > 0 for log scale; got {args.y_min}")
    if args.y_max is not None and args.y_max <= 0:
        raise ValueError(f"--y-max must be > 0 for log scale; got {args.y_max}")
    if (
        args.y_min is not None
        and args.y_max is not None
        and args.y_min >= args.y_max
    ):
        raise ValueError(
            f"--y-min must be < --y-max; got y_min={args.y_min}, y_max={args.y_max}"
        )

    spice_column = f"spice_{args.spice_time_field}_seconds"
    coord_by_depth, spice_by_depth = load_rows(args.combined_csv, spice_column)

    all_depths = sorted(set(coord_by_depth) | set(spice_by_depth))
    if not all_depths:
        raise ValueError("No plottable timing values found (all missing or non-positive).")

    plt.figure(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")

    plotted_any = False
    for idx, depth in enumerate(all_depths):
        color = cmap(idx % 10)

        coord_points = coord_by_depth.get(depth, [])
        if coord_points:
            x = [p[0] for p in coord_points]
            y = [p[1] for p in coord_points]
            plt.plot(
                x,
                y,
                marker="o",
                linestyle="-",
                color=color,
                linewidth=2,
                markersize=6,
                label=f"{depth} hidden layer(s) - coordinate descent",
            )
            plotted_any = True

        spice_points = spice_by_depth.get(depth, [])
        if spice_points:
            x = [p[0] for p in spice_points]
            y = [p[1] for p in spice_points]
            plt.plot(
                x,
                y,
                marker="s",
                linestyle="--",
                color=color,
                linewidth=2,
                markersize=6,
                label=f"{depth} hidden layer(s) - SPICE ({args.spice_time_field})",
            )
            plotted_any = True

    if not plotted_any:
        raise ValueError("No valid timing points were plotted.")

    plt.xscale("log")
    plt.yscale("log")
    if args.y_min is not None or args.y_max is not None:
        plt.ylim(bottom=args.y_min, top=args.y_max)
    plt.xlabel(args.x_label)
    plt.ylabel("Time (seconds)")
    plt.title(args.title)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=180)

    print(f"Saved plot: {args.output}")
    print(f"Depth groups in plot: {', '.join(str(d) for d in all_depths)}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
