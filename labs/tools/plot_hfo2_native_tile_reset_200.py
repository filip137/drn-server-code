#!/usr/bin/env python3
"""Analyze and plot native HfO2 tile.reset -> 200-pulse endpoint samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SAMPLE_SCHEMA = "ebl.lab.hfo2_native_tile_reset_figure6.samples"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _statistics(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0))
    return {
        "minimum": float(quantiles[0]),
        "q01": float(quantiles[1]),
        "q05": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "q99": float(quantiles[5]),
        "maximum": float(quantiles[6]),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
    }


def _histogram_overlap(first: np.ndarray, second: np.ndarray) -> float:
    lower = float(min(first.min(), second.min()))
    upper = float(max(first.max(), second.max()))
    if lower == upper:
        return 1.0
    counts_first, edges = np.histogram(first, bins=240, range=(lower, upper))
    counts_second, _ = np.histogram(second, bins=edges)
    probability_first = counts_first / counts_first.sum()
    probability_second = counts_second / counts_second.sum()
    return float(np.minimum(probability_first, probability_second).sum())


def _comparison(
    reset: np.ndarray,
    set_state: np.ndarray,
) -> dict[str, Any]:
    reset_stats = _statistics(reset)
    set_stats = _statistics(set_state)
    return {
        "reset": reset_stats,
        "set": set_stats,
        "full_support_overlaps": bool(
            reset_stats["maximum"] >= set_stats["minimum"]
        ),
        "central_98_percent_support_overlaps": bool(
            reset_stats["q99"] >= set_stats["q01"]
        ),
        "paired_reset_at_or_above_set_fraction": float(np.mean(reset >= set_state)),
        "reset_at_or_above_set_minimum_fraction": float(
            np.mean(reset >= set_stats["minimum"])
        ),
        "set_at_or_below_reset_maximum_fraction": float(
            np.mean(set_state <= reset_stats["maximum"])
        ),
        "histogram_overlap_mass": _histogram_overlap(reset, set_state),
    }


def _target_range_coverage(
    reset: np.ndarray,
    set_state: np.ndarray,
    target: dict[str, Any],
) -> dict[str, Any]:
    devices = int(reset.size)
    reset_inside = np.logical_and(
        reset >= float(target["reset"]["minimum"]),
        reset <= float(target["reset"]["maximum"]),
    )
    set_inside = np.logical_and(
        set_state >= float(target["set"]["minimum"]),
        set_state <= float(target["set"]["maximum"]),
    )
    positive = np.logical_and(reset > 0.0, set_state > 0.0)
    ordered = set_state > reset
    ratio = np.full(reset.shape, np.nan, dtype=np.float64)
    np.divide(set_state, reset, out=ratio, where=positive)
    dynamic_range_ge_5 = np.logical_and.reduce(
        (positive, ordered, ratio >= 5.0)
    )

    def count_and_fraction(mask: np.ndarray) -> dict[str, float | int]:
        count = int(np.count_nonzero(mask))
        return {"count": count, "fraction": float(count / devices)}

    return {
        "devices": devices,
        "reset_inside_reported_min_max": count_and_fraction(reset_inside),
        "set_inside_reported_min_max": count_and_fraction(set_inside),
        "both_inside_reported_min_max": count_and_fraction(
            np.logical_and(reset_inside, set_inside)
        ),
        "strictly_positive_RESET_and_SET": count_and_fraction(positive),
        "SET_above_RESET": count_and_fraction(ordered),
        "positive_ordered_dynamic_range_at_least_5": count_and_fraction(
            dynamic_range_ge_5
        ),
        "device_filtering": "none",
        "endpoint_clipping_or_folding": "none",
    }


def _load(run_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    samples_path = run_dir / "native_hfo2_reset_200_samples.npz"
    with np.load(samples_path, allow_pickle=False) as payload:
        if str(payload["schema"].item()) != SAMPLE_SCHEMA:
            raise ValueError("Unexpected native HfO2 sample schema.")
        if int(payload["schema_version"].item()) != 1:
            raise ValueError("Unexpected native HfO2 sample schema version.")
        samples = {
            name: payload[name].astype(np.float64, copy=True)
            for name in payload.files
            if name not in {"schema", "schema_version"}
        }
    lengths = {array.shape for array in samples.values()}
    expected = (int(config["population"]["num_devices"]),)
    if lengths != {expected}:
        raise ValueError(f"Expected every sample vector to have shape {expected!r}.")
    if not all(np.isfinite(array).all() for array in samples.values()):
        raise ValueError("Expected finite native HfO2 samples.")
    return config, samples, samples_path


def _calibration(
    reset_persistent_a: np.ndarray,
    set_persistent_a: np.ndarray,
    target: dict[str, Any],
) -> tuple[float, float]:
    reset_median = float(np.median(reset_persistent_a))
    set_median = float(np.median(set_persistent_a))
    if not set_median > reset_median:
        raise ValueError("Expected the simulated SET median above RESET.")
    experimental_reset = float(target["reset"]["median"])
    experimental_set = float(target["set"]["median"])
    scale = (experimental_set - experimental_reset) / (set_median - reset_median)
    offset = experimental_reset - scale * reset_median
    return scale, offset


def _mapped(values: np.ndarray, scale: float, offset: float) -> np.ndarray:
    return scale * values + offset


def _plot_histogram(
    axis: plt.Axes,
    reset: np.ndarray,
    set_state: np.ndarray,
    tile_reset_start: np.ndarray,
    *,
    xlabel: str,
    title: str,
) -> None:
    combined = np.concatenate((reset, set_state))
    bin_count = max(16, min(80, int(np.ceil(2.0 * np.sqrt(combined.size)))))
    bins = np.linspace(
        float(combined.min()), float(combined.max()), bin_count + 1
    )
    axis.hist(
        reset,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.32,
        color="#2784c7",
        edgecolor="#17649a",
        linewidth=1.0,
        label="200 RESET pulses",
    )
    axis.hist(
        set_state,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.28,
        color="#e56a2e",
        edgecolor="#ae4315",
        linewidth=1.0,
        label="200 SET pulses",
    )
    start_q05, start_median, start_q95 = np.quantile(
        tile_reset_start, (0.05, 0.5, 0.95)
    )
    axis.axvspan(
        start_q05,
        start_q95,
        color="#4B5563",
        alpha=0.10,
        label="tile.reset() central 90%",
    )
    axis.axvline(start_median, color="#4B5563", linewidth=1.5)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Density")
    axis.grid(axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def analyze(run_dir: Path) -> None:
    config, sample, samples_path = _load(run_dir)
    reference = sample["reference"]
    values = {
        "reset_start_persistent_w": sample["reset_start_persistent_w"],
        "reset_start_apparent_w": sample["reset_start_apparent_w"],
        "set_start_persistent_w": sample["set_start_persistent_w"],
        "set_start_apparent_w": sample["set_start_apparent_w"],
        "reset_200_persistent_w": sample["reset_200_persistent_w"],
        "reset_200_apparent_w": sample["reset_200_apparent_w"],
        "set_200_persistent_w": sample["set_200_persistent_w"],
        "set_200_apparent_w": sample["set_200_apparent_w"],
    }
    for name in tuple(values):
        values[name.replace("_w", "_a")] = values[name] + reference

    target = config["figure6_target_user_supplied_us"]
    scale, offset = _calibration(
        values["reset_200_persistent_a"],
        values["set_200_persistent_a"],
        target,
    )
    calibrated = {
        name + "_calibrated_us": _mapped(array, scale, offset)
        for name, array in values.items()
        if name.endswith("_a")
    }

    persistent_native = _comparison(
        values["reset_200_persistent_a"], values["set_200_persistent_a"]
    )
    apparent_native = _comparison(
        values["reset_200_apparent_a"], values["set_200_apparent_a"]
    )
    persistent_us = _comparison(
        calibrated["reset_200_persistent_a_calibrated_us"],
        calibrated["set_200_persistent_a_calibrated_us"],
    )
    apparent_us = _comparison(
        calibrated["reset_200_apparent_a_calibrated_us"],
        calibrated["set_200_apparent_a_calibrated_us"],
    )
    target_range_coverage = _target_range_coverage(
        calibrated["reset_200_persistent_a_calibrated_us"],
        calibrated["set_200_persistent_a_calibrated_us"],
        target,
    )
    endpoint_deltas = {
        state: {
            key: float(persistent_us[state][key] - target[state][key])
            for key in ("minimum", "median", "maximum")
        }
        for state in ("reset", "set")
    }
    at_lower = np.isclose(
        values["reset_200_persistent_a"], sample["min_bound"], atol=1e-6, rtol=0.0
    )
    at_upper = np.isclose(
        values["set_200_persistent_a"], sample["max_bound"], atol=1e-6, rtol=0.0
    )
    reset_gap = values["reset_200_persistent_a"] - sample["min_bound"]
    set_gap = sample["max_bound"] - values["set_200_persistent_a"]

    summary = {
        "schema": "ebl.lab.hfo2_native_tile_reset_figure6.summary",
        "schema_version": 1,
        "probe_id": config["probe_id"],
        "evidence_class": config["evidence_class"],
        "sample_artifact_sha256": _sha256(samples_path),
        "num_devices": int(config["population"]["num_devices"]),
        "sequence": (
            "same native tile and sampled population; one independent "
            "tile.reset per arm, followed by 200 RESET or 200 SET pulses"
        ),
        "native_coordinates": {
            "reset_arm_start_persistent_raw_active_a": _statistics(
                values["reset_start_persistent_a"]
            ),
            "reset_arm_start_apparent_raw_active_a": _statistics(
                values["reset_start_apparent_a"]
            ),
            "set_arm_start_persistent_raw_active_a": _statistics(
                values["set_start_persistent_a"]
            ),
            "set_arm_start_apparent_raw_active_a": _statistics(
                values["set_start_apparent_a"]
            ),
            "persistent_endpoints": persistent_native,
            "apparent_endpoints": apparent_native,
            "reset_persistent_at_sampled_lower_bound_fraction": float(np.mean(at_lower)),
            "set_persistent_at_sampled_upper_bound_fraction": float(np.mean(at_upper)),
            "boundary_proximity": {
                "reset_endpoint_minus_lower_bound": _statistics(reset_gap),
                "upper_bound_minus_set_endpoint": _statistics(set_gap),
                "reset_within_0_001_of_lower_bound_fraction": float(
                    np.mean(reset_gap <= 0.001)
                ),
                "set_within_0_001_of_upper_bound_fraction": float(
                    np.mean(set_gap <= 0.001)
                ),
                "reset_endpoint_lower_bound_correlation": float(
                    np.corrcoef(
                        values["reset_200_persistent_a"], sample["min_bound"]
                    )[0, 1]
                ),
                "set_endpoint_upper_bound_correlation": float(
                    np.corrcoef(
                        values["set_200_persistent_a"], sample["max_bound"]
                    )[0, 1]
                ),
            },
        },
        "shared_affine_calibration_to_user_supplied_figure6_medians": {
            "formula": "G_us = scale_us_per_a * a + offset_us",
            "scale_us_per_a": scale,
            "offset_us": offset,
            "fit_points": {
                "reset_median_us": float(target["reset"]["median"]),
                "set_median_us": float(target["set"]["median"]),
            },
            "persistent_endpoints_us": persistent_us,
            "apparent_endpoints_us": apparent_us,
            "persistent_min_median_max_error_us": endpoint_deltas,
            "persistent_figure6_range_coverage": target_range_coverage,
        },
        "figure6_target_user_supplied_us": target,
        "limitations": [
            "AIHWKit's normalized HfO2 preset has no native microSiemens calibration.",
            "The shared affine mapping consumes both experimental medians; "
            "only spread and overlap remain out-of-fit checks.",
            "Only user-supplied min/median/max summaries are available, not "
            "the experimental samples or full Figure 6 density.",
            "The median-fitted affine surrogate gives nonpositive persistent "
            f"RESET conductance to {target_range_coverage['devices'] - target_range_coverage['strictly_positive_RESET_and_SET']['count']} "
            f"of {target_range_coverage['devices']} devices, so it is not a "
            "valid absolute positive-conductance calibration.",
            "AIHWKit CPU does not expose the operational RealWorldRNG state "
            "for exact reset/update replay; the realized arrays are preserved.",
        ],
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=False)
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.13,
        top=0.79,
        hspace=0.42,
        wspace=0.18,
    )
    _plot_histogram(
        axes[0, 0],
        values["reset_200_persistent_a"],
        values["set_200_persistent_a"],
        np.concatenate(
            (
                values["reset_start_persistent_a"],
                values["set_start_persistent_a"],
            )
        ),
        xlabel="Raw active state a (normalized)",
        title="(a) Persistent endpoints",
    )
    _plot_histogram(
        axes[0, 1],
        values["reset_200_apparent_a"],
        values["set_200_apparent_a"],
        np.concatenate(
            (
                values["reset_start_apparent_a"],
                values["set_start_apparent_a"],
            )
        ),
        xlabel="Raw apparent active state a (normalized)",
        title="(b) Apparent endpoints after write noise",
    )
    _plot_histogram(
        axes[1, 0],
        calibrated["reset_200_persistent_a_calibrated_us"],
        calibrated["set_200_persistent_a_calibrated_us"],
        np.concatenate(
            (
                calibrated["reset_start_persistent_a_calibrated_us"],
                calibrated["set_start_persistent_a_calibrated_us"],
            )
        ),
        xlabel="Conductance surrogate (µS)",
        title="(c) Persistent endpoints under shared median calibration",
    )
    _plot_histogram(
        axes[1, 1],
        calibrated["reset_200_apparent_a_calibrated_us"],
        calibrated["set_200_apparent_a_calibrated_us"],
        np.concatenate(
            (
                calibrated["reset_start_apparent_a_calibrated_us"],
                calibrated["set_start_apparent_a_calibrated_us"],
            )
        ),
        xlabel="Apparent conductance surrogate (µS)",
        title="(d) Apparent endpoints under the same calibration",
    )
    for axis in axes[1]:
        top = axis.get_ylim()[1]
        levels = {"reset": 0.92 * top, "set": 0.82 * top}
        colors = {"reset": "#17649a", "set": "#ae4315"}
        for state in ("reset", "set"):
            record = target[state]
            axis.hlines(
                levels[state],
                float(record["minimum"]),
                float(record["maximum"]),
                color=colors[state],
                linewidth=4.0,
                label=f"Figure 6 {state.upper()} min–max",
            )
            axis.plot(
                float(record["median"]),
                levels[state],
                marker="|",
                markersize=14,
                markeredgewidth=2.0,
                color=colors[state],
            )
        axis.set_ylim(0.0, top)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    figure.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=5,
        frameon=False,
    )
    figure.suptitle(
        "AIHWKit HfO₂: native tile.reset followed by 200 directional pulses\n"
        f"{config['population']['num_devices']:,} cells · default non-corrupt "
        "preset · Figure 6 bars use user-supplied summaries",
        fontsize=13,
        y=0.985,
    )
    figure.text(
        0.5,
        0.025,
        "Shared affine axis matches the two experimental medians only; negative "
        "conductance-surrogate values are unphysical and expose a spread mismatch.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4B5563",
    )
    figure.savefig(run_dir / "figure6_comparison.png", dpi=220, facecolor="white")
    figure.savefig(run_dir / "figure6_comparison.svg", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    analyze(arguments.run_dir.resolve())


if __name__ == "__main__":
    main()
