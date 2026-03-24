#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DISABLE_SHM", "1")
os.environ.setdefault("KMP_SHM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

TITLE_MAP = {
    "single_diode_exponential": "Single diode",
    "double_diode_exponential": "Double diode",
    "experimental": "Experimental I-V",
}


def hidden_label(hidden_key: str) -> str:
    if hidden_key.startswith("hidden_"):
        return f"Hidden {hidden_key.split('_', 1)[1]}"
    return hidden_key.replace("_", " ").title()


def filtered_points(rows, percentile, max_iter):
    pts = []
    key = f"p{percentile}"
    for row in rows:
        it = int(row["iterations"])
        if it <= max_iter and key in row:
            pts.append((it, float(row[key])))
    pts.sort(key=lambda x: x[0])
    return pts


def build_title(non_linearity: str, percentiles: list[int]) -> str:
    base = TITLE_MAP.get(non_linearity, non_linearity.replace("_", " ").title())
    if len(percentiles) == 1:
        return f"{base}: p{percentiles[0]} relative error vs iterations"
    joined = "/".join(f"p{p}" for p in percentiles)
    return f"{base}: {joined} relative error vs iterations"


def plot_summary(summary_path: Path, percentiles: list[int], output: Path, max_iter: int):
    data = json.loads(summary_path.read_text())
    hidden = data["hidden"]
    title = build_title(data.get("non_linearity", summary_path.parent.name), percentiles)

    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    colors = plt.get_cmap("tab10").colors
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "D", "^"]

    for idx, hidden_key in enumerate(sorted(hidden.keys())):
        color = colors[idx % len(colors)]
        label_base = hidden_label(hidden_key)
        for p_idx, percentile in enumerate(percentiles):
            pts = filtered_points(hidden[hidden_key], percentile, max_iter)
            if not pts:
                continue
            xs = [x for x, _ in pts]
            ys = [max(y, 1e-16) for _, y in pts]
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=line_styles[p_idx % len(line_styles)],
                marker=markers[p_idx % len(markers)],
                linewidth=2.0,
                markersize=6,
                label=f"{label_base} p{percentile}",
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = [t for t in [4, 8, 16, 32, 64, 128] if t <= max_iter]
    ax.set_xlim(min(ticks), max(ticks))
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(x)}" if any(abs(x - t) < 1e-9 for t in ticks) else "")
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Number of iterations", fontsize=13)
    ax.set_ylabel("Node-weighted relative L1 error", fontsize=13)
    ax.set_title(title, fontsize=15, pad=10)
    ax.grid(True, which="major", linestyle=":", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot digits error-vs-iterations summary JSON.")
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("--percentiles", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=128)
    args = parser.parse_args()
    plot_summary(args.summary_json, args.percentiles, args.output, args.max_iter)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
