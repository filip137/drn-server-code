#!/usr/bin/env python3
"""Plot widest common IBM OM intervals against the nominal step reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = ("p50", "p90", "p99")
WIDTHS = (0.254579408194968, 0.099045060048180, 0.031538705117495)
NOMINAL_DW_MIN = 0.0949
ARRAY_WIDE_SPAN = 5.994294881820679
NOMINAL_MINIMUM_STEP_G = NOMINAL_DW_MIN / ARRAY_WIDE_SPAN
DELTA = 0.5 * NOMINAL_MINIMUM_STEP_G
FOUR_DELTA = 4.0 * DELTA


def _default_output() -> Path:
    repository = Path(__file__).resolve().parents[2]
    return repository / "docs" / "ibm_om_common_intervals_p50_p90_p99.png"


def plot(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    positions = tuple(range(len(LABELS)))
    colors = ("#69aff0", "#339cff", "#69aff0")
    bars = axis.bar(positions, WIDTHS, width=0.56, color=colors, edgecolor="none")

    axis.axhline(FOUR_DELTA, color="#e25507", linewidth=1.8, zorder=3)
    axis.text(
        -0.53,
        FOUR_DELTA + 0.004,
        rf"$4\delta={FOUR_DELTA:.5f}$",
        color="#b74405",
        fontsize=9,
        ha="left",
        va="bottom",
    )

    for bar, width in zip(bars, WIDTHS, strict=True):
        ratio = width / FOUR_DELTA
        ratio_text = f"{ratio:.3f}" if ratio < 1.01 else f"{ratio:.2f}"
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(width, FOUR_DELTA) + 0.007,
            f"{width:.5f}  ({ratio_text}×)",
            fontsize=9,
            ha="center",
            va="bottom",
        )

    axis.text(
        0.98,
        0.96,
        (
            rf"$\delta=\frac{{1}}{{2}}\,\Delta g_{{\mathrm{{min,nom}}}}"
            rf"={DELTA:.6f}$"
            "\n(nominal minimum-step scale, not measured error)"
        ),
        transform=axis.transAxes,
        fontsize=8.5,
        ha="right",
        va="top",
        color="#4a4a4a",
    )

    axis.set_title("Widest common interval by required device coverage", pad=12)
    axis.set_xlabel("Required device coverage")
    axis.set_ylabel("Common interval width (global $g$ coordinate)")
    axis.set_xticks(positions, LABELS)
    axis.set_xlim(-0.62, 2.62)
    axis.set_ylim(0.0, 0.29)
    axis.set_yticks((0.0, 0.08, 0.16, 0.24))
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.65)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_default_output())
    arguments = parser.parse_args()
    plot(arguments.output.resolve())


if __name__ == "__main__":
    main()
