#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_plot_style import (
    PAPER_AXIS_LABEL_FONTSIZE,
    PAPER_LEGEND_FONTSIZE,
    PAPER_TICK_LABEL_FONTSIZE,
    apply_paper_plot_style,
)


@dataclass
class AggregateSeries:
    label: str
    color: str
    steps: np.ndarray
    mean: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    source_paths: list[Path]
    missing_paths: list[Path]

    @property
    def run_count(self) -> int:
        return len(self.source_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot MNIST test accuracy for the selected double-diode runs "
            "together with perfect-diode baseline runs."
        ),
    )
    parser.add_argument(
        "--selected-json",
        required=True,
        type=Path,
        help="Existing aggregate JSON from plot_mean_test_accuracy_from_events.py.",
    )
    parser.add_argument(
        "--perfect-run",
        action="append",
        default=[],
        type=Path,
        help="Perfect-diode run directory or time_series.pkl file. May be repeated.",
    )
    parser.add_argument(
        "--perfect-label",
        type=str,
        default="Perfect diode",
        help="Legend label for the perfect-diode aggregate.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Record missing perfect-diode paths in metadata instead of failing.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output PNG path.")
    parser.add_argument("--output-svg", type=Path, default=None, help="Optional SVG path.")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional JSON summary.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _as_float_array(values: list[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"Expected non-empty 1D values for {name}, got shape {array.shape}")
    return array


def load_selected_aggregate(path: Path) -> AggregateSeries:
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = _as_float_array(payload["steps"], "steps").astype(int)
    mean = _as_float_array(payload["mean_accuracy_percent"], "mean_accuracy_percent")
    minimum = _as_float_array(payload["min_accuracy_percent"], "min_accuracy_percent")
    maximum = _as_float_array(payload["max_accuracy_percent"], "max_accuracy_percent")

    if not (steps.size == mean.size == minimum.size == maximum.size):
        raise ValueError(
            "Expected selected JSON arrays to have equal length, got: "
            f"steps={steps.size}, mean={mean.size}, min={minimum.size}, max={maximum.size}"
        )

    run_paths = [
        Path(run["input_path"])
        for run in payload.get("runs", [])
        if isinstance(run, dict) and "input_path" in run
    ]
    return AggregateSeries(
        label="Double Shockley diode",
        color="tab:blue",
        steps=steps,
        mean=mean,
        minimum=minimum,
        maximum=maximum,
        source_paths=run_paths,
        missing_paths=[],
    )


def load_time_series_accuracy_percent(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected time_series.pkl to contain a dict, got {type(payload)!r}: {path}")

    if "Accuracy/test" in payload:
        values = _as_float_array(payload["Accuracy/test"], "Accuracy/test")
        return values * 100.0 if np.nanmax(np.abs(values)) <= 1.0 + 1e-6 else values

    if "Error/test" in payload:
        values = _as_float_array(payload["Error/test"], "Error/test")
        return (1.0 - values) * 100.0 if np.nanmax(np.abs(values)) <= 1.0 + 1e-6 else 100.0 - values

    expected = "Accuracy/test or Error/test"
    provided = sorted(payload.keys())
    raise KeyError(f"Expected {expected} in {path}, got: {provided}")


def resolve_time_series_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_dir():
        return expanded / "time_series.pkl"
    return expanded


def load_perfect_diode(paths: list[Path], allow_missing: bool, label: str) -> AggregateSeries:
    loaded_paths: list[Path] = []
    missing_paths: list[Path] = []
    series: list[np.ndarray] = []

    for path in paths:
        resolved = resolve_time_series_path(path)
        if not resolved.is_file():
            if allow_missing:
                missing_paths.append(resolved)
                continue
            raise FileNotFoundError(f"Expected perfect-diode time_series.pkl file, got: {resolved}")
        loaded_paths.append(resolved.resolve())
        series.append(load_time_series_accuracy_percent(resolved))

    if not series:
        raise FileNotFoundError("Expected at least one existing perfect-diode time_series.pkl file.")

    max_len = max(item.size for item in series)
    matrix = np.full((len(series), max_len), np.nan, dtype=float)
    for row, values in enumerate(series):
        matrix[row, : values.size] = values

    return AggregateSeries(
        label=label,
        color="tab:orange",
        steps=np.arange(1, max_len + 1, dtype=int),
        mean=np.nanmean(matrix, axis=0),
        minimum=np.nanmin(matrix, axis=0),
        maximum=np.nanmax(matrix, axis=0),
        source_paths=loaded_paths,
        missing_paths=missing_paths,
    )


def plot_accuracy(series_list: list[AggregateSeries], output: Path, output_svg: Path | None, dpi: int) -> None:
    apply_paper_plot_style()
    fig, ax = plt.subplots(figsize=(5.8, 4.1))

    ymin = np.inf
    ymax = -np.inf
    for series in series_list:
        label = f"{series.label} (n={series.run_count})"
        ax.fill_between(
            series.steps,
            series.minimum,
            series.maximum,
            color=series.color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(series.steps, series.mean, color=series.color, linewidth=2.0, label=label)
        ymin = min(ymin, float(np.nanmin(series.minimum)))
        ymax = max(ymax, float(np.nanmax(series.maximum)))

    ax.set_xlabel("Epoch", fontsize=PAPER_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Test accuracy (%)", fontsize=PAPER_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PAPER_TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=PAPER_LEGEND_FONTSIZE, loc="lower right")

    all_steps = np.concatenate([series.steps for series in series_list])
    ax.set_xlim(int(np.nanmin(all_steps)), int(np.nanmax(all_steps)))
    pad = max(0.2, 0.05 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if output_svg is not None:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def write_metadata(
    path: Path,
    series_list: list[AggregateSeries],
    output: Path,
    output_svg: Path | None,
) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "metric": "test accuracy (%)",
        "output_png": str(output.resolve()),
        "output_svg": str(output_svg.resolve()) if output_svg is not None else None,
        "series": [
            {
                "label": series.label,
                "run_count": series.run_count,
                "steps": series.steps.astype(int).tolist(),
                "mean_accuracy_percent": series.mean.tolist(),
                "min_accuracy_percent": series.minimum.tolist(),
                "max_accuracy_percent": series.maximum.tolist(),
                "final_mean_accuracy_percent": float(series.mean[-1]),
                "best_mean_accuracy_percent": float(np.nanmax(series.mean)),
                "source_paths": [str(item) for item in series.source_paths],
                "missing_paths": [str(item) for item in series.missing_paths],
            }
            for series in series_list
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    selected = load_selected_aggregate(args.selected_json.expanduser().resolve())
    perfect = load_perfect_diode(
        args.perfect_run,
        allow_missing=args.allow_missing,
        label=args.perfect_label,
    )
    output = args.output.expanduser().resolve()
    output_svg = args.output_svg.expanduser().resolve() if args.output_svg else None

    plot_accuracy([selected, perfect], output=output, output_svg=output_svg, dpi=args.dpi)
    print(f"Wrote {output}")
    if output_svg is not None:
        print(f"Wrote {output_svg}")

    if args.metadata is not None:
        metadata = args.metadata.expanduser().resolve()
        write_metadata(metadata, [selected, perfect], output=output, output_svg=output_svg)
        print(f"Wrote {metadata}")

    for missing in perfect.missing_paths:
        print(f"Missing perfect-diode run: {missing}")


if __name__ == "__main__":
    main()
