"""Plot IBM OM RESET-to-SET headroom under shared-baseline grouping.

The plot is a descriptive characterization of the jointly repaired physical
mappings saved by the completed baseline-selection study.  It does not model
pulse-to-pulse programming trajectories or claim persistent reachability.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_ROOT = (
    REPOSITORY_ROOT
    / "results/mnist-ibm-om-four-device-baseline-selection-20260828-v1"
)
DEFAULT_OUTPUT_PREFIX = (
    REPOSITORY_ROOT / "docs/figures/ibm_om_headroom_distribution"
)
EXPECTED_HELDOUT_ASSIGNMENTS = (87001, 87002, 87003)
NOMINAL_DW_MIN = 0.0949
DELTA_X = NOMINAL_DW_MIN / 2.0


@dataclass(frozen=True)
class HeadroomSeries:
    label: str
    values: np.ndarray
    color: str
    linestyle: str


def _quad_stack(matrix: np.ndarray, layout: str) -> np.ndarray:
    """Return canonical (++,+-,-+,--) cells for each logical weight."""

    if matrix.ndim != 2 or matrix.shape[0] % 2 or matrix.shape[1] % 2:
        raise ValueError(f"Expected an even two-dimensional matrix, got {matrix.shape}")
    rows = matrix.shape[0] // 2
    logical_columns = matrix.shape[1] // 2
    if layout == "halves":
        plus = np.arange(logical_columns)
        minus = plus + logical_columns
    elif layout == "paired":
        plus = 2 * np.arange(logical_columns)
        minus = plus + 1
    else:
        raise ValueError(f"Unknown logical-column layout: {layout}")
    return np.stack(
        (
            matrix[:rows, plus],
            matrix[:rows, minus],
            matrix[rows:, plus],
            matrix[rows:, minus],
        ),
        axis=-1,
    )


def _heldout_mapping_paths(study_root: Path) -> dict[int, Path]:
    run_root = study_root / "runs/shared-destination-columns-reset-max"
    paths: dict[int, Path] = {}
    for path in sorted(run_root.glob("*/artifacts/heldout/physical_mapping.npz")):
        with np.load(path, allow_pickle=False) as archive:
            assignment = int(archive["assignment_seed"])
        if assignment in paths:
            raise RuntimeError(f"Duplicate mapping for assignment {assignment}")
        paths[assignment] = path
    expected = set(EXPECTED_HELDOUT_ASSIGNMENTS)
    if set(paths) != expected:
        raise RuntimeError(
            "Expected held-out assignments "
            f"{sorted(expected)}, found {sorted(paths)} below {run_root}"
        )
    return paths


def _load_headroom(study_root: Path) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {
        "cell": [],
        "pair": [],
        "quad": [],
        "worst_sign": [],
    }
    for assignment, path in sorted(_heldout_mapping_paths(study_root).items()):
        with np.load(path, allow_pickle=False) as archive:
            for layer, layout in ((0, "halves"), (1, "paired")):
                upper = _quad_stack(
                    archive[f"layer_{layer}_cell_upper_unit"].astype(np.float64),
                    layout,
                )
                reset = _quad_stack(
                    archive[f"layer_{layer}_reset_baseline_unit"].astype(
                        np.float64
                    ),
                    layout,
                )
                baseline = _quad_stack(
                    archive[f"layer_{layer}_baseline_unit"].astype(np.float64),
                    layout,
                )
                saved_positive = archive[
                    f"layer_{layer}_positive_headroom_unit"
                ].astype(np.float64)
                saved_negative = archive[
                    f"layer_{layer}_negative_headroom_unit"
                ].astype(np.float64)

                pair_lower = np.stack(
                    (
                        np.maximum(reset[..., 0], reset[..., 2]),
                        np.maximum(reset[..., 1], reset[..., 3]),
                    ),
                    axis=-1,
                )
                pair_upper = np.stack(
                    (
                        np.minimum(upper[..., 0], upper[..., 2]),
                        np.minimum(upper[..., 1], upper[..., 3]),
                    ),
                    axis=-1,
                )
                expected_baseline = np.stack(
                    (
                        pair_lower[..., 0],
                        pair_lower[..., 1],
                        pair_lower[..., 0],
                        pair_lower[..., 1],
                    ),
                    axis=-1,
                )
                if not np.array_equal(baseline, expected_baseline):
                    raise RuntimeError(
                        "Saved shared-destination baseline does not match the "
                        f"RESET-max construction for assignment {assignment}, "
                        f"layer {layer}"
                    )

                cell = upper - reset
                pair = pair_upper - pair_lower
                quad = np.min(upper, axis=-1) - np.max(reset, axis=-1)
                from_pair_baseline = upper - baseline
                positive = np.minimum(
                    from_pair_baseline[..., 0], from_pair_baseline[..., 3]
                )
                negative = np.minimum(
                    from_pair_baseline[..., 1], from_pair_baseline[..., 2]
                )
                if not np.allclose(positive, saved_positive, rtol=0.0, atol=1e-7):
                    raise RuntimeError("Saved positive-headroom parity check failed")
                if not np.allclose(negative, saved_negative, rtol=0.0, atol=1e-7):
                    raise RuntimeError("Saved negative-headroom parity check failed")
                worst_sign = np.minimum(positive, negative)
                if not np.allclose(
                    worst_sign,
                    np.min(pair, axis=-1),
                    rtol=0.0,
                    atol=1e-7,
                ):
                    raise RuntimeError("Worst-sign versus pair-window parity failed")

                for values in (cell, pair, quad, worst_sign):
                    if not np.isfinite(values).all() or np.min(values) < -1e-7:
                        raise RuntimeError("Headroom must be finite and nonnegative")
                collected["cell"].append(cell.reshape(-1))
                collected["pair"].append(pair.reshape(-1))
                collected["quad"].append(quad.reshape(-1))
                collected["worst_sign"].append(worst_sign.reshape(-1))

    return {name: np.concatenate(values) for name, values in collected.items()}


def _coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    if mean <= 0.0:
        raise ValueError("Coefficient of variation requires a positive mean")
    return float(np.std(values) / mean)


def _secondary_delta_axis(axis: plt.Axes) -> None:
    secondary = axis.secondary_xaxis(
        "top",
        functions=(lambda value: value / DELTA_X, lambda value: value * DELTA_X),
    )
    secondary.set_xlabel(r"Headroom in nominal $\delta_x$ increments", labelpad=5)
    secondary.set_xticks((0, 5, 10, 15, 20))


def _plot(series: list[HeadroomSeries], output_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "font.size": 8.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 240,
        }
    )
    figure, (cdf_axis, summary_axis) = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.25),
        gridspec_kw={"width_ratios": (1.12, 1.0), "wspace": 0.43},
    )

    probability = np.linspace(0.0, 1.0, 2001)
    for item in series:
        quantiles = np.quantile(item.values, probability)
        cdf_axis.plot(
            quantiles,
            probability,
            color=item.color,
            linestyle=item.linestyle,
            linewidth=2.0,
            label=item.label,
        )
    cdf_axis.axvline(
        4.0 * DELTA_X,
        color="#6B7280",
        linestyle=(0, (2, 2)),
        linewidth=1.1,
        zorder=0,
    )
    cdf_axis.text(
        4.0 * DELTA_X + 0.012,
        0.965,
        r"$4\delta_x$",
        color="#4B5563",
        ha="left",
        va="top",
    )
    cdf_axis.set_xlim(0.0, 1.02)
    cdf_axis.set_ylim(0.0, 1.0)
    cdf_axis.set_xticks(np.linspace(0.0, 1.0, 6))
    cdf_axis.set_yticks(np.linspace(0.0, 1.0, 5))
    cdf_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    cdf_axis.set_xlabel("Usable headroom (normalized conductance coordinate)")
    cdf_axis.set_ylabel("Cumulative fraction")
    cdf_axis.set_title("(a) Empirical distributions", loc="left", pad=10)
    cdf_axis.grid(axis="both", color="#D1D5DB", linewidth=0.6, alpha=0.65)
    cdf_axis.legend(loc="lower right", frameon=False)

    y_positions = np.arange(len(series))[::-1]
    for y_position, item in zip(y_positions, series, strict=True):
        p10, median, p90 = np.quantile(item.values, (0.10, 0.50, 0.90))
        cv = _coefficient_of_variation(item.values)
        summary_axis.hlines(
            y_position,
            p10,
            p90,
            color=item.color,
            linewidth=4.0,
            alpha=0.82,
        )
        summary_axis.scatter(
            (p10, p90),
            (y_position, y_position),
            color=item.color,
            marker="|",
            s=120,
            linewidths=1.6,
            zorder=3,
        )
        summary_axis.scatter(
            (median,),
            (y_position,),
            color=item.color,
            marker="D",
            s=34,
            zorder=4,
        )
        summary_axis.text(
            p10,
            y_position - 0.22,
            f"{p10:.3f}",
            color="#374151",
            ha="center",
            va="top",
            fontsize=7.2,
        )
        summary_axis.text(
            median,
            y_position + 0.20,
            f"{median:.3f}",
            color="#111827",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )
        summary_axis.text(
            p90,
            y_position - 0.22,
            f"{p90:.3f}",
            color="#374151",
            ha="center",
            va="top",
            fontsize=7.2,
        )
        summary_axis.text(
            1.055,
            y_position,
            f"{100.0 * cv:.1f}%",
            color="#111827",
            ha="center",
            va="center",
            fontsize=7.7,
        )

    summary_axis.text(
        1.055,
        len(series) - 0.35,
        "CV",
        color="#4B5563",
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
    )
    summary_axis.set_xlim(0.20, 1.12)
    summary_axis.set_ylim(-0.55, len(series) - 0.45)
    summary_axis.set_xticks((0.2, 0.4, 0.6, 0.8, 1.0))
    summary_axis.set_yticks(y_positions, [item.label for item in series])
    for tick, item in zip(summary_axis.get_yticklabels(), series, strict=True):
        tick.set_color(item.color)
    summary_axis.set_xlabel("Usable headroom (normalized conductance coordinate)")
    summary_axis.set_title(
        "(b) p10--p90 interval, median and relative spread",
        loc="left",
        pad=10,
    )
    summary_axis.grid(axis="x", color="#D1D5DB", linewidth=0.6, alpha=0.65)
    _secondary_delta_axis(summary_axis)

    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.14, top=0.82)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    svg_path = output_prefix.with_suffix(".svg")
    figure.savefig(svg_path, bbox_inches="tight")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(figure)


def _print_summary(series: list[HeadroomSeries]) -> None:
    print(f"delta_x={DELTA_X:.6f}")
    print("series,n,p10,median,p90,cv,median_delta_x")
    for item in series:
        p10, median, p90 = np.quantile(item.values, (0.10, 0.50, 0.90))
        cv = _coefficient_of_variation(item.values)
        print(
            f'"{item.label}",{item.values.size},{p10:.6f},{median:.6f},'
            f"{p90:.6f},{cv:.6f},{median / DELTA_X:.3f}"
        )
    threshold = 4.0 * DELTA_X
    print(f"four_delta_x={threshold:.6f}")
    for item in series:
        print(
            f'fraction_below_4delta_x["{item.label}"]='
            f"{np.mean(item.values < threshold):.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        default=DEFAULT_STUDY_ROOT,
        help="Completed baseline-selection study root.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output path without an extension; PNG, PDF, and SVG are written.",
    )
    args = parser.parse_args()

    values = _load_headroom(args.study_root.resolve())
    series = [
        HeadroomSeries("Individual cell", values["cell"], "#0072B2", "-"),
        HeadroomSeries(
            "Two-cell common window", values["pair"], "#009E73", "--"
        ),
        HeadroomSeries("Four-cell common window", values["quad"], "#D55E00", "-."),
        HeadroomSeries(
            "Two-cell policy, worst-sign quad",
            values["worst_sign"],
            "#CC79A7",
            ":",
        ),
    ]
    _print_summary(series)
    _plot(series, args.output_prefix.resolve())


if __name__ == "__main__":
    main()
