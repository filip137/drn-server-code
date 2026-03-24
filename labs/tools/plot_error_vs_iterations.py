#!/usr/bin/env python3
import argparse
import json
import shlex
import sys
from pathlib import Path


def _title_from_non_linearity(value: str) -> str:
    return value.replace("_", " ").title()


def _load_error_series(path: Path, mode: str, percentiles: list[int]):
    payload = json.loads(path.read_text())
    series = payload.get("series", [])
    points = {p: [] for p in percentiles}
    for entry in series:
        iters = entry.get("iteration")
        if iters is None:
            continue
        source = entry.get("node_weighted") if mode == "node_weighted" else entry.get("average")
        if not isinstance(source, dict):
            continue
        for p in percentiles:
            key = f"p{p}"
            if key not in source:
                continue
            points[p].append((int(iters), float(source[key])))
    for p in points:
        points[p] = sorted(points[p], key=lambda item: item[0])
    return points, payload.get("hidden"), payload.get("non_linearity")


def _default_label(path: Path, hidden: int | None) -> str:
    if hidden is not None:
        return f"Hidden {hidden}"
    for part in path.parts[::-1]:
        if part.startswith("hidden_"):
            return f"Hidden {part.replace('hidden_', '')}"
    return path.stem


def _plot_series(
    inputs: list[Path],
    labels: list[str],
    percentiles: list[int],
    mode: str,
    output: Path,
    title: str | None,
    log_x: bool,
    log_y: bool,
    dpi: int,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    colors = plt.get_cmap("tab10").colors
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "D", "^", "v", "P", "X"]

    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    for idx, path in enumerate(inputs):
        points, _, _ = _load_error_series(path, mode, percentiles)
        color = colors[idx % len(colors)]
        for p_idx, p in enumerate(percentiles):
            xs = [it for it, _ in points.get(p, [])]
            ys = [val for _, val in points.get(p, [])]
            if log_y:
                ys = [max(v, 1e-16) for v in ys]
            label = f"{labels[idx]} p{p}"
            ax.plot(
                xs,
                ys,
                marker=markers[p_idx % len(markers)],
                linestyle=line_styles[p_idx % len(line_styles)],
                linewidth=2.0,
                markersize=6,
                color=color,
                label=label,
            )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("Number of iterations", fontsize=13)
    if mode == "node_weighted":
        ax.set_ylabel("Node-weighted rel L1 error", fontsize=13)
    else:
        ax.set_ylabel("Average rel L1 error across layers", fontsize=13)

    if title:
        ax.set_title(title, fontsize=16, pad=10)

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot percentile error vs iterations for multiple hidden configurations.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Paths to error_vs_iterations.json files (hidden_1/2/3, etc).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels matching the input order.",
    )
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=int,
        default=[50, 90],
        help="Percentiles to plot (default: 50 90).",
    )
    parser.add_argument(
        "--mode",
        choices=["node_weighted", "average"],
        default="node_weighted",
        help="Use node_weighted or average percentiles (default: node_weighted).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output image path (png or pdf).",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional output path for plot_metadata.json (default: alongside output).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        default=True,
        help="Use log scale on the x-axis (default: enabled).",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        default=True,
        help="Use log scale on the y-axis (default: enabled).",
    )
    parser.add_argument(
        "--linear-x",
        action="store_false",
        dest="log_x",
        help="Disable log scale on the x-axis.",
    )
    parser.add_argument(
        "--linear-y",
        action="store_false",
        dest="log_y",
        help="Disable log scale on the y-axis.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output DPI (default: 200).",
    )
    args = parser.parse_args(argv)

    inputs = [Path(p).expanduser().resolve() for p in args.inputs]
    for path in inputs:
        if not path.exists():
            raise SystemExit(f"Input not found: {path}")

    labels = []
    non_linearities = []
    for path in inputs:
        _, hidden, non_linearity = _load_error_series(path, args.mode, args.percentiles)
        labels.append(_default_label(path, hidden))
        if non_linearity:
            non_linearities.append(non_linearity)

    if args.labels is not None:
        if len(args.labels) != len(inputs):
            raise SystemExit("If provided, --labels must match number of inputs.")
        labels = args.labels

    title = args.title
    if title is None and non_linearities:
        if len(set(non_linearities)) == 1:
            title = _title_from_non_linearity(non_linearities[0])

    output = Path(args.output).expanduser().resolve()
    _plot_series(
        inputs=inputs,
        labels=labels,
        percentiles=args.percentiles,
        mode=args.mode,
        output=output,
        title=title,
        log_x=args.log_x,
        log_y=args.log_y,
        dpi=args.dpi,
    )

    metadata_path = (
        Path(args.metadata).expanduser().resolve()
        if args.metadata
        else output.parent / "plot_metadata.json"
    )
    metadata = {
        "cli": shlex.join(sys.argv),
        "inputs": [str(p) for p in inputs],
        "labels": labels,
        "percentiles": args.percentiles,
        "mode": args.mode,
        "log_x": bool(args.log_x),
        "log_y": bool(args.log_y),
        "title": title,
        "output": str(output),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved {output}")
    print(f"Saved {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
