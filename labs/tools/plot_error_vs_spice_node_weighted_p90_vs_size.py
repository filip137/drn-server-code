#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HIDDEN_DIR_RE = re.compile(r"hidden_(\d+)$")
DEFAULT_STEM = "error_vs_spice_node_weighted_p90_vs_size_hidden1_hidden2_hidden3"


@dataclass(frozen=True)
class ErrorRow:
    num_hidden_layers: int
    hidden_size: int
    p90: float
    run_npz: str
    spice_npz: str
    summary_json: str


def _hidden_size_from_summary(summary_path: Path, hidden_root: Path) -> int:
    rel = summary_path.relative_to(hidden_root)
    if not rel.parts:
        raise ValueError(f"Cannot parse hidden size from summary path: {summary_path}")
    match = HIDDEN_DIR_RE.match(rel.parts[0])
    if not match:
        raise ValueError(f"First folder is not hidden_<size>: {summary_path}")
    return int(match.group(1))


def _latest_summary_per_size(hidden_root: Path) -> dict[int, Path]:
    by_size: dict[int, Path] = {}
    for summary in hidden_root.glob("hidden_*/**/cross_layer_rel_l1_percentiles_node_weighted.json"):
        try:
            hidden_size = _hidden_size_from_summary(summary, hidden_root)
        except ValueError:
            continue
        prev = by_size.get(hidden_size)
        if prev is None or summary.stat().st_mtime > prev.stat().st_mtime:
            by_size[hidden_size] = summary
    return by_size


def collect_rows(root: Path, hidden_layers: list[int]) -> list[ErrorRow]:
    rows: list[ErrorRow] = []
    for depth in hidden_layers:
        hidden_root = root / f"hidden_{depth}"
        if not hidden_root.exists():
            continue
        latest = _latest_summary_per_size(hidden_root)
        for hidden_size, summary_path in sorted(latest.items()):
            payload = json.loads(summary_path.read_text())
            node_weighted = payload.get("node_weighted_rel_l1_percentiles", {})
            if "p90" not in node_weighted:
                raise KeyError(f"Missing node_weighted_rel_l1_percentiles.p90 in {summary_path}")
            rows.append(
                ErrorRow(
                    num_hidden_layers=depth,
                    hidden_size=hidden_size,
                    p90=float(node_weighted["p90"]),
                    run_npz=str(payload.get("cd_npz", "")),
                    spice_npz=str(payload.get("spice_npz", "")),
                    summary_json=str(summary_path),
                )
            )
    rows.sort(key=lambda row: (row.num_hidden_layers, row.hidden_size))
    return rows


def _read_rows_from_csv(csv_path: Path) -> list[tuple[int, int, float]]:
    rows: list[tuple[int, int, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"num_hidden_layers", "hidden_size", "p90"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns {sorted(missing)} in {csv_path}")
        for row in reader:
            rows.append(
                (
                    int(row["num_hidden_layers"]),
                    int(row["hidden_size"]),
                    float(row["p90"]),
                )
            )
    return rows


def infer_axis_limits_from_csv(
    csv_path: Path, *, x_log_base: float | None, y_log: bool
) -> tuple[tuple[float, float], tuple[float, float]]:
    rows = _read_rows_from_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows in reference CSV: {csv_path}")

    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for depth, hidden_size, p90 in rows:
        grouped[depth].append((hidden_size, p90))
    for depth in grouped:
        grouped[depth].sort(key=lambda item: item[0])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    for depth in sorted(grouped):
        xs = [item[0] for item in grouped[depth]]
        ys = [item[1] for item in grouped[depth]]
        ax.plot(xs, ys, marker="o")
    if x_log_base is not None:
        ax.set_xscale("log", base=x_log_base)
    if y_log:
        ax.set_yscale("log")
    ax.relim()
    ax.autoscale_view()
    x_lim = tuple(float(v) for v in ax.get_xlim())
    y_lim = tuple(float(v) for v in ax.get_ylim())
    plt.close(fig)
    return x_lim, y_lim


def write_csv(rows: list[ErrorRow], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "num_hidden_layers",
                "hidden_size",
                "p90",
                "run_npz",
                "spice_npz",
                "summary_json",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.num_hidden_layers,
                    row.hidden_size,
                    row.p90,
                    row.run_npz,
                    row.spice_npz,
                    row.summary_json,
                ]
            )


def write_plot(
    rows: list[ErrorRow],
    png_path: Path,
    *,
    title: str,
    x_log_base: float | None,
    log_y: bool,
    x_lim: tuple[float, float] | None,
    y_lim: tuple[float, float] | None,
    x_ticks: list[float] | None,
    width: float,
    height: float,
    dpi: int,
    title_font_size: float,
    label_font_size: float,
    tick_font_size: float,
    legend_font_size: float,
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(width, height))
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    for depth in sorted({row.num_hidden_layers for row in rows}):
        group = [row for row in rows if row.num_hidden_layers == depth]
        xs = [row.hidden_size for row in group]
        ys = [row.p90 for row in group]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.0,
            markersize=7.0,
            color=colors.get(depth),
            label=f"hidden_{depth}",
        )
    if x_log_base is not None:
        ax.set_xscale("log", base=x_log_base)
    if log_y:
        ax.set_yscale("log")
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        if x_log_base is None:
            ax.set_xticklabels([str(int(tick)) if float(tick).is_integer() else str(tick) for tick in x_ticks])
    ax.set_xlabel("Hidden size", fontsize=label_font_size)
    ax.set_ylabel("Node-weighted rel-L1 p90 error vs SPICE", fontsize=label_font_size)
    ax.set_title(title, fontsize=title_font_size)
    ax.tick_params(axis="both", labelsize=tick_font_size)
    ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(frameon=False, fontsize=legend_font_size)
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build node-weighted p90 error-vs-size CSV and PNG from "
            "cross_layer_rel_l1_percentiles_node_weighted.json summaries."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/filip/server_code/labs/figures_for_paper_digits/timings/experimental"),
        help="Root directory containing hidden_1/hidden_2/hidden_3 folders.",
    )
    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Hidden-layer groups to include (default: 1 2 3).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <root>/%(default)s.csv).",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Output PNG path (default: <root>/%(default)s.png).",
    )
    parser.add_argument(
        "--title",
        default="Error vs SPICE (node-weighted p90) vs hidden size",
        help="Plot title.",
    )
    parser.add_argument(
        "--x-log2",
        action="store_true",
        help="Use log base-2 scale on x-axis.",
    )
    parser.add_argument(
        "--x-log10",
        action="store_true",
        help="Use log base-10 scale on x-axis.",
    )
    parser.add_argument("--width", type=float, default=18.7, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=11.44, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=100, help="PNG DPI.")
    parser.add_argument(
        "--x-ticks",
        nargs="+",
        type=float,
        default=None,
        help="Optional explicit x-axis tick positions.",
    )
    parser.add_argument("--x-min", type=float, default=None, help="Optional lower x-axis limit.")
    parser.add_argument("--x-max", type=float, default=None, help="Optional upper x-axis limit.")
    parser.add_argument("--y-min", type=float, default=None, help="Optional lower y-axis limit.")
    parser.add_argument("--y-max", type=float, default=None, help="Optional upper y-axis limit.")
    parser.add_argument("--title-font-size", type=float, default=22, help="Plot title font size.")
    parser.add_argument("--label-font-size", type=float, default=18, help="Axis label font size.")
    parser.add_argument("--tick-font-size", type=float, default=16, help="Tick label font size.")
    parser.add_argument("--legend-font-size", type=float, default=16, help="Legend font size.")
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use linear scale for y-axis (default is log).",
    )
    parser.add_argument(
        "--match-axis-from-csv",
        type=Path,
        default=None,
        help="Reference CSV to copy x/y limits from (requires num_hidden_layers,hidden_size,p90 columns).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.x_log2 and args.x_log10:
        raise SystemExit("Use at most one of --x-log2 or --x-log10.")
    x_log_base: float | None = None
    if args.x_log2:
        x_log_base = 2.0
    elif args.x_log10:
        x_log_base = 10.0

    out_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else (root / f"{DEFAULT_STEM}.csv")
    )
    out_png = (
        args.output_png.expanduser().resolve()
        if args.output_png is not None
        else (root / f"{DEFAULT_STEM}.png")
    )

    rows = collect_rows(root, args.hidden_layers)
    if not rows:
        raise SystemExit(f"No node-weighted summary JSONs found under {root}")

    write_csv(rows, out_csv)

    x_lim: tuple[float, float] | None = None
    y_lim: tuple[float, float] | None = None
    if args.match_axis_from_csv is not None:
        reference_csv = args.match_axis_from_csv.expanduser().resolve()
        x_lim, y_lim = infer_axis_limits_from_csv(
            reference_csv,
            x_log_base=x_log_base,
            y_log=not args.linear_y,
        )
        print(f"Matched axis limits from {reference_csv}")
        print(f"xlim={x_lim}")
        print(f"ylim={y_lim}")

    if args.x_min is not None or args.x_max is not None:
        current_x_min = x_lim[0] if x_lim is not None else None
        current_x_max = x_lim[1] if x_lim is not None else None
        x_lim = (
            args.x_min if args.x_min is not None else current_x_min,
            args.x_max if args.x_max is not None else current_x_max,
        )
    if args.y_min is not None or args.y_max is not None:
        current_y_min = y_lim[0] if y_lim is not None else None
        current_y_max = y_lim[1] if y_lim is not None else None
        y_lim = (
            args.y_min if args.y_min is not None else current_y_min,
            args.y_max if args.y_max is not None else current_y_max,
        )

    if x_lim is not None and (x_lim[0] is None or x_lim[1] is None):
        raise SystemExit("Both x-axis limits must be set when overriding x_lim without a reference CSV.")
    if y_lim is not None and (y_lim[0] is None or y_lim[1] is None):
        raise SystemExit("Both y-axis limits must be set when overriding y_lim without a reference CSV.")
    if x_log_base is not None and x_lim is not None and (x_lim[0] <= 0 or x_lim[1] <= 0):
        raise SystemExit("x-axis limits must be > 0 when using log x-axis.")
    if not args.linear_y and y_lim is not None and (y_lim[0] <= 0 or y_lim[1] <= 0):
        raise SystemExit("y-axis limits must be > 0 when using log y-axis.")

    write_plot(
        rows,
        out_png,
        title=args.title,
        x_log_base=x_log_base,
        log_y=not args.linear_y,
        x_lim=x_lim,
        y_lim=y_lim,
        x_ticks=args.x_ticks,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        title_font_size=args.title_font_size,
        label_font_size=args.label_font_size,
        tick_font_size=args.tick_font_size,
        legend_font_size=args.legend_font_size,
    )

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote PNG: {out_png}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
