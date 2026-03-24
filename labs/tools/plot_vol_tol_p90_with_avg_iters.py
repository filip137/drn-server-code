#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FAMILY_TITLE = {
    "single_diode_exponential": "Single diode",
    "double_diode_exponential": "Double diode",
    "experimental": "Experimental I-V",
}


def parse_rel_tol(item: dict) -> float:
    if "rel_tol_value" in item and item["rel_tol_value"] is not None:
        return float(item["rel_tol_value"])
    val = item.get("rel_tol")
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        return float(val)
    raise ValueError(f"Unsupported rel_tol value: {val!r}")


def display_label(hidden_key: str) -> str:
    if hidden_key.startswith("hidden_"):
        parts = hidden_key.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return f"Hidden {parts[1]}"
    return hidden_key.replace("_", " ").title()


def load_avg_iterations(row: dict) -> float:
    source = Path(row["source"])
    run_dir = source.parent.parent
    meta_path = run_dir / "validation_metadata.json"
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    eq = metadata.get("equilibrium_iterations", {})
    if "avg_iterations" not in eq:
        raise KeyError(f"avg_iterations missing in {meta_path}")
    return float(eq["avg_iterations"])


def build_series(summary_json: Path) -> tuple[dict, str]:
    with summary_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    hidden = data.get("hidden", {})
    if not hidden:
        raise ValueError(f"No hidden data found in {summary_json}")

    family = data.get("non_linearity", summary_json.parent.parent.name)
    title = FAMILY_TITLE.get(family, family)
    metadata_hidden = {}
    for hidden_key in sorted(hidden.keys()):
        rows = hidden[hidden_key]
        triples = []
        for row in rows:
            rel_tol = parse_rel_tol(row)
            p90 = float(row["p90"])
            avg_iters = load_avg_iterations(row)
            triples.append((rel_tol, p90, avg_iters))
        triples.sort(key=lambda x: x[0])
        metadata_hidden[hidden_key] = {
            "label": display_label(hidden_key),
            "rel_tol": [t[0] for t in triples],
            "p90": [t[1] for t in triples],
            "avg_outer_iterations": [t[2] for t in triples],
        }
    return metadata_hidden, title


def plot_summary(summary_json: Path, output_png: Path, output_svg: Path | None = None, metadata_json: Path | None = None) -> dict:
    series, title = build_series(summary_json)
    plot_title = f"{title}: p90 relative error and average outer sweeps vs voltage tolerance"

    fig, ax1 = plt.subplots(figsize=(7.4, 5.4))
    ax2 = ax1.twinx()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    hidden_handles = []
    hidden_labels = []
    for idx, hidden_key in enumerate(sorted(series.keys())):
        color = colors[idx % len(colors)]
        item = series[hidden_key]
        rel_tol = item["rel_tol"]
        p90 = item["p90"]
        avg_iters = item["avg_outer_iterations"]
        label = item["label"]

        line_p90, = ax1.plot(
            rel_tol,
            p90,
            marker="o",
            markersize=5,
            linewidth=1.8,
            color=color,
        )
        ax2.plot(
            rel_tol,
            avg_iters,
            marker="s",
            markersize=4.5,
            linewidth=1.5,
            linestyle="--",
            color=color,
            alpha=0.95,
        )
        hidden_handles.append(line_p90)
        hidden_labels.append(label)

    ax1.set_xscale("log")
    ax1.invert_xaxis()
    ax1.set_yscale("log")
    ax1.set_xlabel("Relative voltage tolerance")
    ax1.set_ylabel("P90 relative error")
    ax2.set_ylabel("Average outer sweeps")
    ax1.set_title(plot_title)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)

    hidden_legend = fig.legend(
        hidden_handles,
        hidden_labels,
        title="Configuration",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=max(1, len(hidden_labels)),
    )
    fig.add_artist(hidden_legend)
    metric_handles = [
        Line2D([0], [0], color="black", marker="o", linewidth=1.8, linestyle="-", label="P90 error"),
        Line2D([0], [0], color="black", marker="s", linewidth=1.5, linestyle="--", label="Avg. outer sweeps"),
    ]
    fig.legend(
        handles=metric_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
    )

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.92))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    if output_svg is not None:
        fig.savefig(output_svg)
    plt.close(fig)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary_json": str(summary_json),
        "output_png": str(output_png),
        "output_svg": str(output_svg) if output_svg is not None else None,
        "title": plot_title,
        "family_title": title,
        "hidden": series,
        "matplotlib_version": matplotlib.__version__,
    }
    if metadata_json is not None:
        with metadata_json.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot P90 error vs voltage tolerance with average outer sweeps on a secondary axis.")
    parser.add_argument("summary_json", type=Path, help="Path to the voltage-tolerance summary JSON.")
    parser.add_argument("--out", dest="output_png", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--out-svg", dest="output_svg", type=Path, default=None, help="Optional output SVG path.")
    parser.add_argument("--metadata", dest="metadata_json", type=Path, default=None, help="Optional output metadata JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_json = args.summary_json
    output_png = args.output_png or summary_json.with_name("p90_vs_rel_tol_all_hidden.png")
    output_svg = args.output_svg or summary_json.with_name("p90_vs_rel_tol_all_hidden.svg")
    metadata_json = args.metadata_json or summary_json.with_name("p90_vs_rel_tol_all_hidden_metadata.json")
    plot_summary(summary_json, output_png, output_svg, metadata_json)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_svg}")
    print(f"Wrote {metadata_json}")


if __name__ == "__main__":
    main()
