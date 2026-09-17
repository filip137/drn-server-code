"""Plot held-apparent accuracy and KL for the Figure-6/IBM-OM ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARM_ORDER = (
    "direct_pv",
    "direct_pv_reset_stuck",
    "hwa_pv",
    "hwa_pv_reset_stuck",
    "hwa_pv_adam",
    "hwa_pv_reset_stuck_adam",
    "scratch_pv_adam",
    "scratch_pv_reset_stuck_adam",
)

ARM_LABELS = {
    "direct_pv": "Direct P&V",
    "direct_pv_reset_stuck": "Direct P&V + RESET-stuck",
    "hwa_pv": "HWA + P&V",
    "hwa_pv_reset_stuck": "HWA + P&V + RESET-stuck",
    "hwa_pv_adam": "HWA + P&V + pulse Adam",
    "hwa_pv_reset_stuck_adam": "HWA + P&V + RESET-stuck + pulse Adam",
    "scratch_pv_adam": "Scratch + pulse Adam",
    "scratch_pv_reset_stuck_adam": "Scratch + RESET-stuck + pulse Adam",
}

FAMILY_COLORS = {
    "direct_pv": "#4c78a8",
    "direct_pv_reset_stuck": "#4c78a8",
    "hwa_pv": "#f58518",
    "hwa_pv_reset_stuck": "#f58518",
    "hwa_pv_adam": "#54a24b",
    "hwa_pv_reset_stuck_adam": "#54a24b",
    "scratch_pv_adam": "#b279a2",
    "scratch_pv_reset_stuck_adam": "#b279a2",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected {label} to be a mapping.")
    return value


def _load_summary(path: Path) -> Mapping[str, Any]:
    summary = _mapping(json.loads(path.read_text(encoding="utf-8")), "summary")
    if (
        summary.get("status") != "complete"
        or summary.get("primary_state") != "held_apparent"
        or int(summary.get("records", 0)) != 72
    ):
        raise ValueError("Expected the complete 72-record held-apparent summary.")
    arms = _mapping(summary.get("arms"), "summary arms")
    if set(arms) != set(ARM_ORDER):
        raise ValueError("Summary arms differ from the declared eight-arm ladder.")
    return summary


def _series(
    arms: Mapping[str, Any], metric: str, *, scale: float
) -> tuple[list[float], list[float], list[float]]:
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for arm in ARM_ORDER:
        value = _mapping(_mapping(arms[arm], arm)[metric], f"{arm}.{metric}")
        limits = value.get("full_range_across_array_means")
        if not isinstance(limits, Sequence) or len(limits) != 2:
            raise ValueError(f"Expected a two-value range for {arm}.{metric}.")
        means.append(scale * float(value["mean_across_arrays"]))
        lows.append(scale * float(limits[0]))
        highs.append(scale * float(limits[1]))
    return means, lows, highs


def _padded_domain(lows: Sequence[float], highs: Sequence[float]) -> tuple[float, float]:
    lower = min(lows)
    upper = max(highs)
    span = upper - lower
    padding = max(span * 0.16, 0.02)
    return lower - padding, upper + padding


def _draw_panel(
    axis: Any,
    *,
    means: Sequence[float],
    lows: Sequence[float],
    highs: Sequence[float],
    xlabel: str,
    value_format: str,
) -> None:
    y_values = list(range(len(ARM_ORDER)))
    for index, arm in enumerate(ARM_ORDER):
        color = FAMILY_COLORS[arm]
        axis.hlines(index, lows[index], highs[index], color=color, linewidth=3.0)
        axis.vlines(
            (lows[index], highs[index]),
            index - 0.09,
            index + 0.09,
            color=color,
            linewidth=1.4,
        )
        axis.scatter(
            means[index],
            index,
            color=color,
            marker="X" if "reset_stuck" in arm else "o",
            s=62,
            zorder=3,
        )
        axis.annotate(
            format(means[index], value_format),
            (means[index], index),
            xytext=(7, -10),
            textcoords="offset points",
            fontsize=8.5,
        )
    axis.set_xlim(*_padded_domain(lows, highs))
    axis.set_yticks(y_values, [ARM_LABELS[arm] for arm in ARM_ORDER])
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", alpha=0.24)
    axis.set_axisbelow(True)


def plot(summary: Mapping[str, Any], output_prefix: Path) -> None:
    arms = _mapping(summary["arms"], "summary arms")
    accuracy = _series(arms, "apparent_student_accuracy", scale=100.0)
    divergence = _series(arms, "apparent_kl_teacher_student", scale=1.0)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 7.0),
        constrained_layout=True,
        sharey=True,
    )
    _draw_panel(
        axes[0],
        means=accuracy[0],
        lows=accuracy[1],
        highs=accuracy[2],
        xlabel="MNIST test accuracy (%) — higher is better",
        value_format=".2f",
    )
    _draw_panel(
        axes[1],
        means=divergence[0],
        lows=divergence[1],
        highs=divergence[2],
        xlabel="KL(teacher || student) — lower is better",
        value_format=".4f",
    )
    axes[0].invert_yaxis()
    axes[0].set_title("Held-apparent accuracy")
    axes[1].set_title("Held-apparent KL divergence")
    figure.suptitle(
        "784-256-10 Figure-6 / IBM-OM deployment ladder",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.01,
        "Points are means across three held-out arrays after averaging three writes per array; "
        "whiskers show the full range of array means. X markers denote post-P&V "
        "RESET-stuck faults.",
        ha="center",
        fontsize=9,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    svg_path = output_prefix.with_suffix(".svg")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)

    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    arguments = parser.parse_args()
    plot(_load_summary(arguments.summary), arguments.output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
