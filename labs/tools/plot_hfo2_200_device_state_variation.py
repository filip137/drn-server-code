#!/usr/bin/env python3
"""Plot RESET/SET endpoint variation for a native AIHWKit HfO2 population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESET_COLOR = "#2878B5"
SET_COLOR = "#D95F0E"
SAMPLE_SCHEMA = "ebl.lab.hfo2_native_tile_reset_figure6.samples"


def _load_samples(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["schema"].item()) != SAMPLE_SCHEMA:
            raise ValueError("Unexpected native HfO2 sample schema.")
        if int(payload["schema_version"].item()) != 1:
            raise ValueError("Unexpected native HfO2 sample schema version.")
        samples = {
            name: payload[name].astype(np.float64, copy=True)
            for name in payload.files
            if name not in {"schema", "schema_version"}
        }

    required = {
        "reference",
        "reset_200_persistent_w",
        "set_200_persistent_w",
        "reset_200_apparent_w",
        "set_200_apparent_w",
    }
    missing = required - set(samples)
    if missing:
        raise ValueError(f"Missing sample arrays: {sorted(missing)!r}.")
    shapes = {samples[name].shape for name in required}
    if len(shapes) != 1:
        raise ValueError(f"Expected all sample arrays to share one shape, got {shapes!r}.")
    if not all(np.isfinite(samples[name]).all() for name in required):
        raise ValueError("Expected finite native HfO2 samples.")
    return samples


def _statistics(values: np.ndarray) -> dict[str, float]:
    q05, median, q95 = np.quantile(values, (0.05, 0.5, 0.95))
    return {
        "minimum": float(values.min()),
        "q05": float(q05),
        "median": float(median),
        "mean": float(values.mean()),
        "q95": float(q95),
        "maximum": float(values.max()),
        "standard_deviation": float(values.std()),
    }


def _endpoint_panel(
    axis: plt.Axes,
    reset: np.ndarray,
    set_state: np.ndarray,
    order: np.ndarray,
    *,
    title: str,
) -> None:
    x = np.arange(1, reset.size + 1)
    reset_ordered = reset[order]
    set_ordered = set_state[order]
    axis.vlines(
        x,
        reset_ordered,
        set_ordered,
        color="#A8ADB3",
        alpha=0.34,
        linewidth=0.65,
        zorder=1,
    )
    axis.scatter(
        x,
        reset_ordered,
        s=13,
        color=RESET_COLOR,
        edgecolors="white",
        linewidths=0.25,
        label=(
            f"RESET  median {np.median(reset):.2f}, "
            f"SD {np.std(reset):.2f}"
        ),
        zorder=3,
    )
    axis.scatter(
        x,
        set_ordered,
        s=13,
        color=SET_COLOR,
        edgecolors="white",
        linewidths=0.25,
        label=(
            f"SET  median {np.median(set_state):.2f}, "
            f"SD {np.std(set_state):.2f}"
        ),
        zorder=3,
    )
    axis.axhline(0.0, color="#4B5563", linewidth=0.8, alpha=0.55)
    axis.set_xlim(0, reset.size + 1)
    axis.set_title(title, loc="left", fontsize=11, weight="semibold")
    axis.set_ylabel("Normalized active state, a")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(loc="upper left", frameon=False, fontsize=8.4, ncol=2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _density_panel(
    axis: plt.Axes,
    reset: np.ndarray,
    set_state: np.ndarray,
    *,
    title: str,
) -> None:
    combined = np.concatenate((reset, set_state))
    bins = np.linspace(float(combined.min()), float(combined.max()), 31)
    axis.hist(
        reset,
        bins=bins,
        density=True,
        orientation="horizontal",
        histtype="stepfilled",
        alpha=0.34,
        color=RESET_COLOR,
        edgecolor=RESET_COLOR,
        linewidth=1.0,
    )
    axis.hist(
        set_state,
        bins=bins,
        density=True,
        orientation="horizontal",
        histtype="stepfilled",
        alpha=0.28,
        color=SET_COLOR,
        edgecolor=SET_COLOR,
        linewidth=1.0,
    )
    axis.axhline(0.0, color="#4B5563", linewidth=0.8, alpha=0.55)
    axis.set_title(title, loc="left", fontsize=10)
    axis.set_xlabel("Density")
    axis.grid(axis="x", alpha=0.22)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _summary(
    *,
    sample_path: Path,
    persistent_reset: np.ndarray,
    persistent_set: np.ndarray,
    apparent_reset: np.ndarray,
    apparent_set: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema": "ebl.lab.hfo2_200_device_state_variation.summary",
        "schema_version": 1,
        "source": {
            "sample_path": str(sample_path),
            "preset": "ReRamArrayHfO2PresetDevice",
            "aihwkit_version": "1.1.0",
            "pulses_per_endpoint": 200,
            "corrupt_devices_enabled": False,
            "coordinate": "normalized raw active state a = w + reference",
        },
        "num_devices": int(persistent_reset.size),
        "persistent": {
            "reset": _statistics(persistent_reset),
            "set": _statistics(persistent_set),
            "reset_at_or_above_paired_set_fraction": float(
                np.mean(persistent_reset >= persistent_set)
            ),
        },
        "apparent_after_write_noise": {
            "reset": _statistics(apparent_reset),
            "set": _statistics(apparent_set),
            "reset_at_or_above_paired_set_fraction": float(
                np.mean(apparent_reset >= apparent_set)
            ),
        },
    }


def plot(sample_path: Path, output_dir: Path) -> None:
    samples = _load_samples(sample_path)
    reference = samples["reference"]

    # SoftBoundsReferenceDevice stores the offset-subtracted state w. Adding
    # the fixed per-device reference reconstructs the common raw active-state
    # coordinate in which the lower/upper device endpoints are defined.
    persistent_reset = samples["reset_200_persistent_w"] + reference
    persistent_set = samples["set_200_persistent_w"] + reference
    apparent_reset = samples["reset_200_apparent_w"] + reference
    apparent_set = samples["set_200_apparent_w"] + reference

    if persistent_reset.size != 200:
        raise ValueError(
            f"This figure requires exactly 200 devices, got {persistent_reset.size}."
        )

    order = np.argsort(persistent_reset, kind="stable")
    all_values = np.concatenate(
        (persistent_reset, persistent_set, apparent_reset, apparent_set)
    )
    padding = 0.06 * float(np.ptp(all_values))
    y_limits = (float(all_values.min() - padding), float(all_values.max() + padding))

    figure = plt.figure(figsize=(12.0, 7.8), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(3.25, 1.0),
        left=0.075,
        right=0.975,
        bottom=0.13,
        top=0.82,
        hspace=0.38,
        wspace=0.08,
    )
    persistent_axis = figure.add_subplot(grid[0, 0])
    persistent_density = figure.add_subplot(
        grid[0, 1], sharey=persistent_axis
    )
    apparent_axis = figure.add_subplot(
        grid[1, 0], sharex=persistent_axis, sharey=persistent_axis
    )
    apparent_density = figure.add_subplot(
        grid[1, 1], sharey=persistent_axis
    )

    _endpoint_panel(
        persistent_axis,
        persistent_reset,
        persistent_set,
        order,
        title="(a) Persistent device states after 200 directional pulses",
    )
    _density_panel(
        persistent_density,
        persistent_reset,
        persistent_set,
        title="Persistent distribution",
    )
    _endpoint_panel(
        apparent_axis,
        apparent_reset,
        apparent_set,
        order,
        title="(b) Apparent states after post-write noise",
    )
    _density_panel(
        apparent_density,
        apparent_reset,
        apparent_set,
        title="Apparent distribution",
    )
    persistent_axis.set_ylim(*y_limits)
    persistent_axis.tick_params(labelbottom=False)
    apparent_axis.set_xlabel("Device rank (ordered by persistent RESET state)")
    persistent_density.tick_params(labelleft=False)
    apparent_density.tick_params(labelleft=False)

    figure.suptitle(
        "Device-to-device variation of AIHWKit baseline HfO₂ RESET/SET states",
        fontsize=15,
        weight="semibold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.905,
        "200 devices · ReRamArrayHfO2PresetDevice · native AIHWKit 1.1.0 CPU tile · "
        "default non-corrupt preset",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#374151",
    )
    figure.text(
        0.5,
        0.04,
        "Each vertical segment links RESET and SET for the same device. Values use the "
        "preset’s normalized active-state coordinate; AIHWKit supplies no native µS mapping.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4B5563",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "hfo2_200_device_set_reset_variation"
    figure.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)

    summary = _summary(
        sample_path=sample_path,
        persistent_reset=persistent_reset,
        persistent_set=persistent_set,
        apparent_reset=apparent_reset,
        apparent_set=apparent_set,
    )
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
