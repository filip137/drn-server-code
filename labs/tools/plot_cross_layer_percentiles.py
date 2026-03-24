#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

PERCENTILE_KEYS = ("p50", "p60", "p70", "p80", "p90", "p95", "p99")


def _default_label(path: Path) -> str:
    # Expected layout: .../<run_dir>/<error_dir>/cross_layer_rel_l1_percentiles.json
    run_dir = path.parent.parent.name if path.parent.parent.name else path.parent.name
    return run_dir


def plot_cross_layer_percentiles(
    summary_paths,
    *,
    output_dir,
    output_stem,
    labels=None,
    title=None,
    y_log=False,
    node_weighted=False,
    output_pdf=False,
):
    paths = [Path(p).expanduser().resolve() for p in summary_paths]
    if not paths:
        raise ValueError("No summary paths provided.")

    if labels is not None and len(labels) != len(paths):
        raise ValueError("If provided, --labels must match number of input files.")

    runs = []
    for idx, path in enumerate(paths):
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        if node_weighted:
            avg = payload.get("node_weighted_rel_l1_percentiles")
            if not isinstance(avg, dict):
                raise KeyError(f"Missing node_weighted_rel_l1_percentiles in {path}")
        else:
            avg = payload.get("average_rel_l1_percentiles_across_layers")
            if not isinstance(avg, dict):
                raise KeyError(f"Missing average_rel_l1_percentiles_across_layers in {path}")
        missing = [k for k in PERCENTILE_KEYS if k not in avg]
        if missing:
            raise KeyError(f"Missing percentile keys {missing} in {path}")
        label = labels[idx] if labels is not None else _default_label(path)
        runs.append(
            {
                "label": label,
                "source": str(path),
                "num_layers": payload.get("num_layers"),
                "percentiles": {k: float(avg[k]) for k in PERCENTILE_KEYS},
            }
        )

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "percentile_order": list(PERCENTILE_KEYS),
        "runs": {run["label"]: run for run in runs},
        "y_log": bool(y_log),
        "title": title,
        "node_weighted": bool(node_weighted),
    }
    out_json = output_dir / f"{output_stem}.json"
    out_json.write_text(json.dumps(combined, indent=2))

    x = [int(k[1:]) for k in PERCENTILE_KEYS]
    plt.figure(figsize=(8, 4.8))
    for run in runs:
        y = [run["percentiles"][k] for k in PERCENTILE_KEYS]
        plt.plot(x, y, marker="o", linewidth=2.0, label=run["label"])
    if y_log:
        plt.yscale("log")
    plt.xlabel("Percentile")
    if node_weighted:
        plt.ylabel("Node-Weighted Relative L1 Error")
    else:
        plt.ylabel("Avg Relative L1 Error Across Layers")
    if title is not None:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png = output_dir / f"{output_stem}.png"
    plt.savefig(out_png, dpi=220)
    out_pdf = None
    if output_pdf:
        out_pdf = output_dir / f"{output_stem}.pdf"
        plt.savefig(out_pdf)
    plt.close()

    return out_json, out_png, out_pdf


def main() -> int:
    p = argparse.ArgumentParser(
        description="Combine and plot cross_layer_rel_l1_percentiles.json files.",
    )
    p.add_argument(
        "summary_paths",
        nargs="+",
        help="Paths to cross_layer_rel_l1_percentiles.json files.",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels matching the input order.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output JSON and PNG plot.",
    )
    p.add_argument(
        "--output-stem",
        default="cross_layer_rel_l1_percentiles_comparison",
        help="Base filename (without extension).",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )
    p.add_argument(
        "--y-log",
        action="store_true",
        help="Use log scale on the y-axis.",
    )
    p.add_argument(
        "--node-weighted",
        action="store_true",
        help="Read node_weighted_rel_l1_percentiles instead of average_rel_l1_percentiles_across_layers.",
    )
    p.add_argument(
        "--output-pdf",
        action="store_true",
        help="Also save a PDF alongside the PNG.",
    )
    args = p.parse_args()

    out_json, out_png, out_pdf = plot_cross_layer_percentiles(
        args.summary_paths,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        labels=args.labels,
        title=args.title,
        y_log=args.y_log,
        node_weighted=args.node_weighted,
        output_pdf=args.output_pdf,
    )
    print(out_json)
    print(out_png)
    if out_pdf is not None:
        print(out_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
