#!/usr/bin/env python3
"""Plot native HfO2 RESET/SET endpoints in the style of Gong et al. Fig. 6."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SAMPLE_SCHEMA = "ebl.lab.hfo2_native_tile_reset_figure6.samples"
PRESET = "ReRamArrayHfO2PresetDevice"
CURVE_COLOR = "#E31A1C"
PROBABILITY_TICKS = np.asarray(
    (0.5, 2.0, 10.0, 30.0, 50.0, 70.0, 90.0, 98.0, 99.5),
    dtype=np.float64,
)
NORMAL = NormalDist()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probit(probability: np.ndarray | float) -> np.ndarray | float:
    values = np.asarray(probability, dtype=np.float64)
    transformed = np.asarray(
        [NORMAL.inv_cdf(float(value)) for value in values.reshape(-1)],
        dtype=np.float64,
    ).reshape(values.shape)
    if np.isscalar(probability):
        return float(transformed.item())
    return transformed


def _plotting_positions(count: int) -> tuple[np.ndarray, np.ndarray]:
    probability = (np.arange(count, dtype=np.float64) + 0.5) / count
    return probability, np.asarray(_probit(probability), dtype=np.float64)


def _statistics(values: np.ndarray) -> dict[str, float]:
    q05, q50, q95 = np.quantile(values, (0.05, 0.5, 0.95))
    return {
        "minimum": float(values.min()),
        "q05": float(q05),
        "median": float(q50),
        "mean": float(values.mean()),
        "q95": float(q95),
        "maximum": float(values.max()),
        "standard_deviation": float(values.std()),
    }


def _load_source(
    sample_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    config_path = sample_path.parent / "config.json"
    receipt_path = sample_path.parent / "sampling_receipt.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    if config["source"]["preset"] != PRESET:
        raise ValueError(f"Expected {PRESET}, got {config['source']['preset']!r}.")
    if int(config["population"]["num_devices"]) != 200:
        raise ValueError("Expected a population of exactly 200 devices.")
    if int(config["update"]["pulses_per_arm"]) != 200:
        raise ValueError("Expected 200 directional pulses per endpoint.")
    if bool(config["source"]["enable_published_corruption"]):
        raise ValueError("Expected the default non-corrupt HfO2 preset population.")
    if receipt["samples_sha256"] != _file_sha256(sample_path):
        raise ValueError("Sample digest does not match the native sampling receipt.")

    with np.load(sample_path, allow_pickle=False) as payload:
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
    if {samples[name].shape for name in required} != {(200,)}:
        raise ValueError("Expected every required sample array to have shape (200,).")
    if not all(np.isfinite(samples[name]).all() for name in required):
        raise ValueError("Expected finite endpoint values.")
    return config, receipt, samples


def _normalize_pair(
    reset: np.ndarray, set_state: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    lower = float(min(reset.min(), set_state.min()))
    upper = float(max(reset.max(), set_state.max()))
    span = upper - lower
    if span <= 0.0:
        raise ValueError("Expected nonzero shared endpoint span.")
    return (
        (reset - lower) / span,
        (set_state - lower) / span,
        {
            "formula": "g = (a - A_min) / (A_max - A_min)",
            "A_min": lower,
            "A_max": upper,
            "span": span,
            "scope": "one shared affine map over RESET and SET endpoints",
            "per_device_normalization": False,
        },
    )


def _render(
    *,
    reset: np.ndarray,
    set_state: np.ndarray,
    state_kind: str,
    aihwkit_version: str,
    output_path: Path,
) -> dict[str, Any]:
    reset_g, set_g, normalization = _normalize_pair(reset, set_state)
    _, y = _plotting_positions(reset.size)
    reset_sorted = np.sort(reset_g)
    set_sorted = np.sort(set_g)
    reset_median = float(np.median(reset_g))
    set_median = float(np.median(set_g))
    median_separation = set_median - reset_median

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
        }
    )
    figure, axis = plt.subplots(figsize=(8.2, 7.2), facecolor="white")
    axis.plot(
        set_sorted,
        y,
        linestyle="none",
        marker="o",
        markersize=5.0,
        markerfacecolor="white",
        markeredgecolor=CURVE_COLOR,
        markeredgewidth=1.0,
        label=r"Fully SET ($G_{\max}$)",
        zorder=3,
    )
    axis.plot(
        reset_sorted,
        y,
        linestyle="none",
        marker="o",
        markersize=5.0,
        markerfacecolor=CURVE_COLOR,
        markeredgecolor=CURVE_COLOR,
        markeredgewidth=0.8,
        label=r"Fully RESET ($G_{\min}$)",
        zorder=3,
    )

    median_y = float(_probit(0.5))
    axis.annotate(
        "",
        xy=(set_median, median_y),
        xytext=(reset_median, median_y),
        arrowprops={
            "arrowstyle": "<->",
            "color": CURVE_COLOR,
            "linewidth": 1.7,
            "shrinkA": 2,
            "shrinkB": 2,
        },
        zorder=4,
    )
    axis.text(
        0.5 * (reset_median + set_median),
        float(_probit(0.68)),
        rf"median separation $\Delta g={median_separation:.3f}$",
        ha="center",
        va="bottom",
        color=CURVE_COLOR,
        fontsize=10.5,
        weight="semibold",
    )

    probability_tick_positions = np.asarray(
        _probit(PROBABILITY_TICKS / 100.0), dtype=np.float64
    )
    axis.set_yticks(
        probability_tick_positions,
        [f"{value:g}" for value in PROBABILITY_TICKS],
    )
    axis.set_ylim(float(_probit(0.002)), float(_probit(0.998)))
    axis.set_xlim(-0.03, 1.03)
    axis.set_xticks(np.linspace(0.0, 1.0, 6))
    axis.set_xlabel(
        r"Normalized conductance, $g=(a-A_{\min})/(A_{\max}-A_{\min})$"
    )
    axis.set_ylabel("Cumulative probability (%)")
    axis.grid(axis="y", color="#D9DCE1", linewidth=0.75, alpha=0.8)
    axis.legend(loc="upper left", frameon=False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)

    kind_label = (
        "apparent endpoint after post-write noise"
        if state_kind == "apparent"
        else "persistent endpoint before post-write noise"
    )
    figure.suptitle(
        "AIHWKit HfO₂: fully RESET and fully SET state distributions",
        fontsize=15,
        weight="semibold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.915,
        f"200 devices · 200 directional pulses per endpoint · {kind_label}",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#374151",
    )
    figure.text(
        0.5,
        0.035,
        f"{PRESET} · AIHWKit {aihwkit_version} · default non-corrupt preset · "
        "shared population normalization",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#4B5563",
    )
    figure.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.85)
    figure.savefig(output_path.with_suffix(".png"), dpi=240, facecolor="white")
    figure.savefig(output_path.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)

    return {
        "state_kind": state_kind,
        "normalization": normalization,
        "reset_raw_active_a": _statistics(reset),
        "set_raw_active_a": _statistics(set_state),
        "reset_normalized_conductance": _statistics(reset_g),
        "set_normalized_conductance": _statistics(set_g),
        "normalized_median_separation": median_separation,
        "paired_set_above_reset_fraction": float(np.mean(set_state > reset)),
        "outputs": {
            "png": str(output_path.with_suffix(".png")),
            "pdf": str(output_path.with_suffix(".pdf")),
        },
    }


def _write_values_csv(
    output_path: Path,
    *,
    persistent_reset: np.ndarray,
    persistent_set: np.ndarray,
    apparent_reset: np.ndarray,
    apparent_set: np.ndarray,
) -> None:
    persistent_reset_g, persistent_set_g, _ = _normalize_pair(
        persistent_reset, persistent_set
    )
    apparent_reset_g, apparent_set_g, _ = _normalize_pair(
        apparent_reset, apparent_set
    )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "device_index",
                "persistent_reset_a",
                "persistent_set_a",
                "persistent_reset_g",
                "persistent_set_g",
                "apparent_reset_a",
                "apparent_set_a",
                "apparent_reset_g",
                "apparent_set_g",
            )
        )
        for index in range(persistent_reset.size):
            writer.writerow(
                (
                    index,
                    persistent_reset[index],
                    persistent_set[index],
                    persistent_reset_g[index],
                    persistent_set_g[index],
                    apparent_reset[index],
                    apparent_set[index],
                    apparent_reset_g[index],
                    apparent_set_g[index],
                )
            )


def plot(sample_path: Path, output_dir: Path) -> None:
    config, receipt, samples = _load_source(sample_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = samples["reference"]
    persistent_reset = samples["reset_200_persistent_w"] + reference
    persistent_set = samples["set_200_persistent_w"] + reference
    apparent_reset = samples["reset_200_apparent_w"] + reference
    apparent_set = samples["set_200_apparent_w"] + reference

    records = {}
    for kind, reset, set_state in (
        ("apparent", apparent_reset, apparent_set),
        ("persistent", persistent_reset, persistent_set),
    ):
        stem = output_dir / f"hfo2_200_device_figure6_style_{kind}_cdf"
        records[kind] = _render(
            reset=reset,
            set_state=set_state,
            state_kind=kind,
            aihwkit_version=str(receipt["aihwkit_version"]),
            output_path=stem,
        )

    values_path = output_dir / "hfo2_200_device_figure6_style_values.csv"
    _write_values_csv(
        values_path,
        persistent_reset=persistent_reset,
        persistent_set=persistent_set,
        apparent_reset=apparent_reset,
        apparent_set=apparent_set,
    )
    summary = {
        "schema": "ebl.lab.hfo2_200_device_figure6_style_cdf.summary",
        "schema_version": 1,
        "source": {
            "sample_path": str(sample_path),
            "sample_sha256": _file_sha256(sample_path),
            "preset": PRESET,
            "aihwkit_version": receipt["aihwkit_version"],
            "construction_seed": receipt["construction_seed"],
            "num_devices": config["population"]["num_devices"],
            "pulses_per_endpoint": config["update"]["pulses_per_arm"],
            "corrupt_devices_enabled": config["source"][
                "enable_published_corruption"
            ],
        },
        "figure_convention": {
            "reference": "Gong et al., IEDM 2022, Figure 6",
            "x_axis": "shared normalized conductance",
            "y_axis": "empirical cumulative probability on a probit scale",
            "filled_markers": "fully RESET",
            "open_markers": "fully SET",
        },
        "endpoint_records": records,
        "values_csv": str(values_path),
    }
    (output_dir / "hfo2_200_device_figure6_style_stats.json").write_text(
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
