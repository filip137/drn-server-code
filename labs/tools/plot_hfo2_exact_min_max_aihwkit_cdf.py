#!/usr/bin/env python3
"""Plot exact sampled HfO2 limits in AIHWKit's native normalized coordinate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_hfo2_figure6_style_cdf import (
    CURVE_COLOR,
    PRESET,
    PROBABILITY_TICKS,
    _file_sha256,
    _load_source,
    _plotting_positions,
    _probit,
    _statistics,
)


def plot(sample_path: Path, output_dir: Path) -> None:
    config, receipt, samples = _load_source(sample_path)
    reference = samples["reference"]

    # Native SoftBoundsReferenceDevice bounds are stored in the raw active
    # coordinate a. AIHWKit exposes the reference-subtracted signed state
    # w = a - reference, so its exact per-device limits are transformed here.
    reset_w_min = samples["min_bound"] - reference
    set_w_max = samples["max_bound"] - reference
    if not np.all(set_w_max > reset_w_min):
        raise ValueError("Expected every sampled SET limit above its RESET limit.")

    _, probability_y = _plotting_positions(reset_w_min.size)
    reset_sorted = np.sort(reset_w_min)
    set_sorted = np.sort(set_w_max)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 20,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 23,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axis = plt.subplots(figsize=(10.5, 9), facecolor="white")
    axis.plot(
        set_sorted,
        probability_y,
        linestyle="none",
        marker="o",
        markersize=6.0,
        markerfacecolor="white",
        markeredgecolor=CURVE_COLOR,
        markeredgewidth=1.2,
        label=r"Maximum SET ($w_{\max,i}$)",
        zorder=3,
    )
    axis.plot(
        reset_sorted,
        probability_y,
        linestyle="none",
        marker="o",
        markersize=6.0,
        markerfacecolor=CURVE_COLOR,
        markeredgecolor=CURVE_COLOR,
        markeredgewidth=1.0,
        label=r"Minimum RESET ($w_{\min,i}$)",
        zorder=3,
    )

    tick_positions = np.asarray(
        _probit(PROBABILITY_TICKS / 100.0), dtype=np.float64
    )
    axis.set_yticks(
        tick_positions,
        [f"{value:g}" for value in PROBABILITY_TICKS],
    )
    axis.set_ylim(float(_probit(0.002)), float(_probit(0.998)))
    combined = np.concatenate((reset_w_min, set_w_max))
    padding = 0.035 * float(np.ptp(combined))
    axis.set_xlim(float(combined.min() - padding), float(combined.max() + padding))
    axis.set_xlabel(r"AIHWKit normalized signed state, $w$", labelpad=12)
    axis.set_ylabel("Cumulative probability (%)", labelpad=12)
    axis.tick_params(axis="both", length=6, width=1.2, pad=8)
    axis.grid(axis="y", color="#D9DCE1", linewidth=0.75, alpha=0.8)
    axis.legend(
        loc="upper left",
        frameon=False,
        handlelength=0.8,
        handletextpad=0.5,
        borderpad=0.2,
        borderaxespad=0.3,
        markerscale=1.25,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)

    figure.suptitle(
        "AIHWKit HfO₂:\nminimum RESET and maximum SET states",
        fontsize=24,
        weight="semibold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.035,
        f"{PRESET} · AIHWKit {receipt['aihwkit_version']}\n"
        r"$w_{\min,i}=a_{\min,i}-r_i$ and $w_{\max,i}=a_{\max,i}-r_i$",
        ha="center",
        va="bottom",
        fontsize=13,
        color="#4B5563",
    )
    figure.subplots_adjust(left=0.17, right=0.97, bottom=0.20, top=0.83)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "hfo2_200_device_exact_min_max_aihwkit_cdf"
    figure.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(figure)

    csv_path = stem.with_name(stem.name + "_values").with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("device_index", "exact_reset_w_min", "exact_set_w_max"))
        for index, (reset, set_state) in enumerate(
            zip(reset_w_min, set_w_max, strict=True)
        ):
            writer.writerow((index, reset, set_state))

    summary = {
        "schema": "ebl.lab.hfo2_exact_min_max_aihwkit_cdf.summary",
        "schema_version": 1,
        "source": {
            "sample_path": str(sample_path),
            "sample_sha256": _file_sha256(sample_path),
            "preset": PRESET,
            "aihwkit_version": receipt["aihwkit_version"],
            "construction_seed": receipt["construction_seed"],
            "num_devices": config["population"]["num_devices"],
            "corrupt_devices_enabled": config["source"][
                "enable_published_corruption"
            ],
        },
        "coordinate": {
            "name": "AIHWKit reference-subtracted signed state w",
            "reset_formula": "w_min_i = min_bound_i - reference_i",
            "set_formula": "w_max_i = max_bound_i - reference_i",
            "nominal_reset": -1.0,
            "nominal_set": 1.0,
            "additional_rescaling": False,
        },
        "exact_reset_w_min": _statistics(reset_w_min),
        "exact_set_w_max": _statistics(set_w_max),
        "all_device_bounds_strictly_ordered": True,
        "minimum_paired_state_window": float(np.min(set_w_max - reset_w_min)),
        "values_csv": str(csv_path),
        "presentation": {
            "size_inches": [10.5, 9],
            "font_sizes": {
                "title": 24,
                "axis_labels": 24,
                "ticks": 20,
                "legend": 23,
                "source_note": 13,
            },
            "png_dpi": 300,
        },
        "outputs": {
            "png": str(stem.with_suffix(".png")),
            "pdf": str(stem.with_suffix(".pdf")),
            "svg": str(stem.with_suffix(".svg")),
        },
    }
    stem.with_name(stem.name + "_stats").with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    plot(arguments.samples.resolve(), arguments.output_dir.resolve())


if __name__ == "__main__":
    main()
