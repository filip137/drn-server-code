#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_series(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {
        "iterations": [int(x) for x in data.get("iterations", [])],
        "mean_rel_l1": [float(x) for x in data.get("metrics", {}).get("mean_rel_l1", [])],
    }


def plot_mean_rel_error(
    paths: dict[str, Path],
    output_path: Path,
    log_x: bool = True,
    log_y: bool = True,
    width: float = 7.6,
    height: float = 5.4,
) -> None:
    fig, ax = plt.subplots(figsize=(width, height))

    for label, path in paths.items():
        series = _load_series(path)
        iters = series["iterations"]
        vals = [max(v, 1e-16) for v in series["mean_rel_l1"]]
        ax.plot(iters, vals, marker="o", linewidth=2.0, markersize=6, label=label)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_title("Mean relative error", fontsize=16, pad=10)
    ax.set_xlabel("Number of iterations", fontsize=13)
    ax.set_ylabel("Mean relative error", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=12)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot mean_rel_l1 vs iterations for multiple hidden counts.",
    )
    parser.add_argument("--hidden1", required=True, help="Path to hidden_1 mean_error_vs_iter.json")
    parser.add_argument("--hidden2", required=True, help="Path to hidden_2 mean_error_vs_iter.json")
    parser.add_argument("--hidden3", required=True, help="Path to hidden_3 mean_error_vs_iter.json")
    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument("--width", type=float, default=7.6, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=5.4, help="Figure height in inches.")
    args = parser.parse_args(argv)

    paths = {
        "1H": Path(args.hidden1).expanduser().resolve(),
        "2H": Path(args.hidden2).expanduser().resolve(),
        "3H": Path(args.hidden3).expanduser().resolve(),
    }
    plot_mean_rel_error(
        paths,
        Path(args.output).expanduser().resolve(),
        width=args.width,
        height=args.height,
    )
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
