#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _load_series(path: Path, *, exclude_dims: set[int] | None = None) -> dict[int, list[tuple[int, float]]]:
    data = json.loads(path.read_text())
    series = {1: [], 2: [], 3: []}
    for row in data:
        layers = int(row["hidden_layers"])
        dim = int(row["hidden_dim"])
        if exclude_dims and dim in exclude_dims:
            continue
        elapsed = float(row["elapsed_seconds"])
        if layers in series:
            series[layers].append((dim, elapsed))
    for layers in series:
        series[layers] = sorted(series[layers], key=lambda x: x[0])
    return series


def plot_timing_compare(
    cd_path: Path,
    spice_path: Path,
    output_path: Path,
    *,
    title: str = "Runtime CD/SPICE",
    xlabel: str = "Hidden size",
    ylabel: str = "Runtime (s)",
    log_x: bool = False,
    log_y: bool = False,
    exclude_dims: set[int] | None = None,
) -> None:
    labels = {1: "1H", 2: "2H", 3: "3H"}
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}

    cd_series = _load_series(cd_path, exclude_dims=exclude_dims)
    spice_series = _load_series(spice_path, exclude_dims=exclude_dims)

    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    for layers, points in cd_series.items():
        if not points:
            continue
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=6,
            color=colors[layers],
            label=f"{labels[layers]} (CD)",
        )

    for layers, points in spice_series.items():
        if not points:
            continue
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        ax.plot(
            x,
            y,
            marker="s",
            linewidth=2.0,
            markersize=6,
            color=colors[layers],
            linestyle="--",
            label=f"{labels[layers]} (SPICE)",
        )

    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)
    layer_handles = [
        Line2D([0], [0], color=colors[1], lw=2, marker="o", label="1H"),
        Line2D([0], [0], color=colors[2], lw=2, marker="o", label="2H"),
        Line2D([0], [0], color=colors[3], lw=2, marker="o", label="3H"),
    ]
    method_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle="--", marker="s", label="CD"),
        Line2D([0], [0], color="black", lw=2, linestyle="-", marker="o", label="SPICE"),
    ]
    ax.legend(
        handles=layer_handles + method_handles,
        title="Legend",
        frameon=False,
        fontsize=11,
        loc="upper left",
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot runtime comparison for coordinate descent vs spice simulations.",
    )
    parser.add_argument("--cd", required=True, help="Path to coordinate descent timing JSON.")
    parser.add_argument("--spice", required=True, help="Path to spice timing JSON.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--title", default="Runtime CD/SPICE")
    parser.add_argument("--xlabel", default="Hidden size")
    parser.add_argument("--ylabel", default="Runtime (s)")
    parser.add_argument("--log-x", action="store_true", help="Use log scale on x-axis.")
    parser.add_argument("--log-y", action="store_true", help="Use log scale on y-axis.")
    parser.add_argument(
        "--exclude-dims",
        nargs="+",
        type=int,
        default=None,
        help="Hidden sizes to exclude from the plot.",
    )
    args = parser.parse_args(argv)

    plot_timing_compare(
        Path(args.cd).expanduser().resolve(),
        Path(args.spice).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        log_x=args.log_x,
        log_y=args.log_y,
        exclude_dims=set(args.exclude_dims) if args.exclude_dims else None,
    )
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
