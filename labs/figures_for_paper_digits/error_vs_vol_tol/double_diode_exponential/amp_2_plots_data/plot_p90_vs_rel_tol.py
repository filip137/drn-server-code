#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


def _parse_rel_tol(item):
    if "rel_tol_value" in item and item["rel_tol_value"] is not None:
        return float(item["rel_tol_value"])
    val = item.get("rel_tol")
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        return float(val)
    raise ValueError(f"Unsupported rel_tol value: {val!r}")


def plot_p90_vs_rel_tol(summary_path, hidden_key, output_path, output_svg=None):
    summary_path = Path(summary_path)
    output_path = Path(output_path)
    if output_svg is not None:
        output_svg = Path(output_svg)

    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    hidden = data.get("hidden", {})
    if hidden_key not in hidden:
        raise KeyError(f"hidden key {hidden_key!r} not found. Available: {sorted(hidden.keys())}")

    rows = hidden[hidden_key]
    rel_tols = []
    p90s = []
    for row in rows:
        rel_tols.append(_parse_rel_tol(row))
        p90s.append(float(row["p90"]))

    # Sort by rel_tol ascending
    pairs = sorted(zip(rel_tols, p90s), key=lambda x: x[0])
    rel_tols = [p[0] for p in pairs]
    p90s = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rel_tols, p90s, marker="o", linewidth=1.6, color="#1f77b4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rel_tol")
    ax.set_ylabel("p90 error")
    ax.set_title(f"p90 vs rel_tol ({hidden_key})")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if output_svg is not None:
        fig.savefig(output_svg)
    plt.close(fig)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_summary": str(summary_path),
        "hidden_key": hidden_key,
        "rel_tol": rel_tols,
        "p90": p90s,
        "output_png": str(output_path),
        "output_svg": str(output_svg) if output_svg is not None else None,
        "matplotlib_version": matplotlib.__version__,
    }
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Plot p90 vs rel_tol from a summary JSON.")
    parser.add_argument("summary_json", help="Path to summary JSON")
    parser.add_argument("--hidden", dest="hidden_key", default=None, help="Hidden key to plot (e.g. hidden_2)")
    parser.add_argument("--out", dest="out_png", default=None, help="Output PNG path")
    parser.add_argument("--out-svg", dest="out_svg", default=None, help="Optional output SVG path")
    parser.add_argument("--metadata", dest="metadata_path", default=None, help="Metadata JSON output path")
    args = parser.parse_args()

    with open(args.summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    hidden = data.get("hidden", {})
    if args.hidden_key is None:
        if "hidden_2" in hidden:
            hidden_key = "hidden_2"
        elif len(hidden) == 1:
            hidden_key = next(iter(hidden))
        else:
            raise ValueError("Multiple hidden keys found. Please specify --hidden.")
    else:
        hidden_key = args.hidden_key

    summary_path = Path(args.summary_json)
    out_png = Path(args.out_png) if args.out_png else summary_path.with_name("p90_vs_rel_tol.png")
    out_svg = Path(args.out_svg) if args.out_svg else summary_path.with_name("p90_vs_rel_tol.svg")
    metadata_path = Path(args.metadata_path) if args.metadata_path else summary_path.with_name("p90_vs_rel_tol_metadata.json")

    metadata = plot_p90_vs_rel_tol(summary_path, hidden_key, out_png, out_svg)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
