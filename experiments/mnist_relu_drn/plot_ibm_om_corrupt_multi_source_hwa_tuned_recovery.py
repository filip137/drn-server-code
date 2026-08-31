"""Plot the completed multi-source IBM-OM HWA and recovery ladder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    REPOSITORY_ROOT
    / "results/mnist-ibm-om-corrupt-multi-source-hwa-tuned-recovery-"
    "exploratory-20260831-v1/20260831T182040.075995Z-7f054bc5-3e515245/"
    "artifacts/scientific_summary.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "docs/figures/ibm_om_corrupt_multi_source_hwa_tuned_recovery.svg"
)
EXPECTED_SCHEMA = "ebl.mnist_relu_drn.ibm_om_corrupt_multi_source_hwa_tuned_recovery"

BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#D55E00"
AMBER = "#E69F00"
PURPLE = "#7A3E9D"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"
PALE_PURPLE = "#F3EDF7"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if summary.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"Unexpected summary schema: {summary.get('schema')!r}")
    if summary.get("status") != "complete":
        raise ValueError("The scientific summary is not complete")
    if summary.get("evidence_tier") != "exploratory_noncanonical":
        raise ValueError("Expected exploratory_noncanonical evidence")
    return summary


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _validate(summary: dict[str, Any]) -> None:
    development = summary["source_and_development"]
    target = summary["target_94004"]
    expected_arms = {
        "single_w2_x1",
        "single_w2_x30",
        "multi_w2_x1",
        "multi_w2_x30",
    }
    if set(development["arms"]) != expected_arms:
        raise ValueError("Unexpected HWA arm set")
    if development["joint_selection"]["winner_arm"] != "multi_w2_x30":
        raise ValueError("Unexpected joint winner")
    if development["joint_selection"]["winner_epoch"] != 6:
        raise ValueError("Unexpected winning epoch")

    expected_target_seeds = summary["protocol"]["target"]["endpoint_seeds"]
    paired = target["program_verify"]["paired_hwa_gate"]["per_endpoint"]
    paired_seeds = [item["endpoint_seed"] for item in paired]
    if paired_seeds != expected_target_seeds or len(paired) != 5:
        raise ValueError("Expected five paired endpoints on target 94004")
    for state in ("epoch0", "winner"):
        block = target["program_verify"][state]
        accuracies = [item["test"]["student_accuracy"] for item in block["repeats"]]
        if not math.isclose(_mean(accuracies), block["mean"], abs_tol=1e-12):
            raise ValueError(f"Stored target {state} mean does not match repeats")

    recovery = target["output_kl_open_loop_adam"]
    if target["p0"]["endpoint_seed"] != 94301 or recovery["selected_epoch"] != 3:
        raise ValueError("Unexpected P0 or selected Adam epoch")


def _percent(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}%"


def _style() -> None:
    plt.rcParams.update(
        {
            "axes.edgecolor": "#374151",
            "axes.labelcolor": "#1F2937",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titlelocation": "left",
            "axes.titlesize": 10.5,
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "legend.fontsize": 7.5,
            "svg.fonttype": "none",
            "svg.hashsalt": "ibm-om-corrupt-multi-source-hwa-tuned-recovery",
            "text.color": "#111827",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
        }
    )


def _finish_axis(axis: plt.Axes, *, y_min: float = 0.0) -> None:
    axis.set_ylim(y_min, 1.0)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.grid(axis="y")
    axis.set_axisbelow(True)


def _plot(summary: dict[str, Any], output: Path) -> None:
    _style()
    development = summary["source_and_development"]
    target = summary["target_94004"]
    winner_arm = development["joint_selection"]["winner_arm"]
    winner_epoch = development["joint_selection"]["winner_epoch"]

    figure = plt.figure(figsize=(14.2, 5.15))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.23, 1.0, 1.42), wspace=0.34)
    dev_axis = figure.add_subplot(grid[0, 0])
    ladder_axis = figure.add_subplot(grid[0, 1])
    endpoint_axis = figure.add_subplot(grid[0, 2])
    figure.suptitle(
        "Multi-source HWA improves a sealed IBM-OM target; one realized P0 remains recoverable",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=14,
        fontweight="semibold",
    )

    # (a) Four pre-target development trajectories. Epoch zero is common.
    arm_style = {
        "single_w2_x1": ("Single source · W2 ×1", SKY, (0, (3, 2)), "o"),
        "single_w2_x30": ("Single source · W2 ×30", BLUE, "-", "o"),
        "multi_w2_x1": ("Two sources · W2 ×1", AMBER, (0, (3, 2)), "s"),
        "multi_w2_x30": ("Two sources · W2 ×30", ORANGE, "-", "s"),
    }
    common_epoch0 = development["development_epoch0"]["persistent_mean_accuracy"]
    arm_order = ("single_w2_x1", "single_w2_x30", "multi_w2_x1", "multi_w2_x30")
    for arm_id in arm_order:
        records = development["arms"][arm_id]
        label, color, linestyle, marker = arm_style[arm_id]
        epochs = [0] + [record["epoch"] for record in records]
        accuracy = [common_epoch0] + [
            record["development"]["persistent_mean_accuracy"] for record in records
        ]
        dev_axis.plot(
            epochs,
            accuracy,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4.3,
            linewidth=2.25 if arm_id == winner_arm else 1.45,
            label=label,
            zorder=3 if arm_id == winner_arm else 2,
        )
    winner_value = development["arms"][winner_arm][-1]["development"][
        "persistent_mean_accuracy"
    ]
    dev_axis.scatter(
        [winner_epoch], [winner_value], marker="*", s=115, color=ORANGE, zorder=5
    )
    dev_axis.annotate(
        f"winner  {_percent(winner_value, 3)}",
        (winner_epoch, winner_value),
        xytext=(-4, 9),
        textcoords="offset points",
        ha="right",
        color=ORANGE,
        fontweight="semibold",
    )
    dev_axis.set_xticks(range(7))
    dev_axis.set_xlabel("HWA epoch")
    dev_axis.set_ylabel("Development persistent P&V mean")
    dev_axis.set_title("(a) Development on identity 94003 (4 writes per point)")
    dev_axis.legend(loc="upper left", frameon=False, ncols=2, columnspacing=0.9)
    _finish_axis(dev_axis, y_min=0.25)

    # (b) Matched accuracy ladder on the sealed target.
    stage_x = [0, 1, 2]
    stage_labels = ["Ideal requested\n(no write)", "Native-corrupt\nfault clamp", "Persistent P&V\nmean"]
    ladder = {
        "Epoch 0": [
            target["ideal_requested_no_write"]["epoch0"]["metrics"]["student_accuracy"],
            target["deterministic_fault_only"]["epoch0"]["evaluation"]["metrics"]["student_accuracy"],
            target["program_verify"]["epoch0"]["mean"],
        ],
        f"HWA winner (epoch {winner_epoch})": [
            target["ideal_requested_no_write"]["winner"]["metrics"]["student_accuracy"],
            target["deterministic_fault_only"]["winner"]["evaluation"]["metrics"]["student_accuracy"],
            target["program_verify"]["winner"]["mean"],
        ],
    }
    for label, values in ladder.items():
        is_winner = label.startswith("HWA")
        color = ORANGE if is_winner else BLUE
        marker = "s" if is_winner else "o"
        ladder_axis.plot(
            stage_x,
            values,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2.2,
            label=label,
            zorder=3,
        )
        for x_value, value in zip(stage_x, values, strict=True):
            if x_value == 0:
                offset = (-9, 7) if is_winner else (9, -12)
                horizontal_alignment = "right" if is_winner else "left"
            else:
                offset = (0, 7 if is_winner else -12)
                horizontal_alignment = "center"
            ladder_axis.annotate(
                _percent(value),
                (x_value, value),
                xytext=offset,
                textcoords="offset points",
                ha=horizontal_alignment,
                va="bottom" if is_winner else "top",
                color=color,
                fontweight="semibold" if x_value == 2 else "normal",
            )
    ladder_axis.set_xticks(stage_x, stage_labels)
    ladder_axis.set_xlim(-0.22, 2.22)
    ladder_axis.set_ylabel("Target test accuracy")
    ladder_axis.set_title("(b) Sealed target identity 94004")
    ladder_axis.legend(loc="upper right", frameon=False)
    _finish_axis(ladder_axis)

    # (c) Five paired target writes, then a separately scoped one-P0 recovery.
    paired = target["program_verify"]["paired_hwa_gate"]["per_endpoint"]
    for item in paired:
        values = [item["epoch0_accuracy"], item["selected_hwa_accuracy"]]
        endpoint_axis.plot([0, 1], values, color=LIGHT_GRAY, linewidth=1.1, zorder=1)
        endpoint_axis.scatter([0], [values[0]], color=BLUE, marker="o", s=25, zorder=3)
        endpoint_axis.scatter([1], [values[1]], color=ORANGE, marker="s", s=25, zorder=3)
    paired_means = [
        target["program_verify"]["epoch0"]["mean"],
        target["program_verify"]["winner"]["mean"],
    ]
    endpoint_axis.plot(
        [0, 1], paired_means, color="#111827", marker="D", markersize=5.5, linewidth=2.0, zorder=4
    )
    endpoint_axis.text(
        0.5,
        0.68,
        f"5/5 improve · mean +{100.0 * (paired_means[1] - paired_means[0]):.3f} pp",
        ha="center",
        color="#374151",
    )

    endpoint_axis.axvspan(1.55, 5.25, color=PALE_PURPLE, zorder=0)
    recovery = target["output_kl_open_loop_adam"]
    recovery_epochs = [0] + [item["epoch"] for item in recovery["epoch_reports"]]
    recovery_validation = [recovery["initial_validation"]["student_accuracy"]] + [
        item["validation"]["student_accuracy"] for item in recovery["epoch_reports"]
    ]
    recovery_x = [2, 3, 4, 5]
    endpoint_axis.plot(
        recovery_x,
        recovery_validation,
        color=PURPLE,
        marker="o",
        markersize=5,
        linewidth=2.1,
        zorder=4,
    )
    for x_value, epoch, value in zip(
        recovery_x, recovery_epochs, recovery_validation, strict=True
    ):
        if epoch > 0:
            endpoint_axis.annotate(
                _percent(value),
                (x_value, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                color=PURPLE,
            )
    p0_test = next(
        item["selected_hwa_accuracy"]
        for item in paired
        if item["endpoint_seed"] == target["p0"]["endpoint_seed"]
    )
    selected_test = recovery["selected_test"]["student_accuracy"]
    endpoint_axis.scatter(
        [2], [p0_test], facecolors="white", edgecolors=PURPLE, marker="D", s=38, zorder=5
    )
    endpoint_axis.text(
        2.07,
        0.36,
        f"P0: {_percent(p0_test)} test / {_percent(recovery_validation[0])} val",
        ha="left",
        color=PURPLE,
    )
    endpoint_axis.scatter(
        [5.12], [selected_test], color=PURPLE, marker="D", s=42, zorder=5
    )
    endpoint_axis.annotate(
        f"test {_percent(selected_test)}",
        (5.12, selected_test),
        xytext=(-2, -14),
        textcoords="offset points",
        ha="right",
        color=PURPLE,
        fontweight="semibold",
    )
    endpoint_axis.text(
        3.55,
        0.76,
        "Exact seed-94301 P0 only",
        ha="center",
        color=PURPLE,
        fontweight="semibold",
    )
    endpoint_axis.axvline(1.5, color="#9CA3AF", linewidth=0.8)
    endpoint_axis.set_xticks(
        [0, 1, 2, 3, 4, 5],
        ["Epoch-0\nP&V", "Winner\nP&V", "P0", "Adam\n1", "Adam\n2", "Adam\n3"],
    )
    endpoint_axis.set_xlim(-0.25, 5.35)
    endpoint_axis.set_ylabel("Target accuracy")
    endpoint_axis.set_title("(c) Paired writes and endpoint-specific Adam")
    _finish_axis(endpoint_axis)

    endpoint_axis.legend(
        handles=[
            Line2D([0], [0], color=LIGHT_GRAY, marker="o", markerfacecolor=BLUE, linewidth=1.1, label="5 paired endpoint writes"),
            Line2D([0], [0], color="#111827", marker="D", linewidth=2, label="P&V mean (not CI)"),
            Line2D([0], [0], color=PURPLE, marker="o", linewidth=2, label="Adam validation"),
            Line2D([0], [0], color=PURPLE, marker="D", linewidth=0, label="Selected held-out test"),
        ],
        loc="lower right",
        frameon=False,
    )

    figure.subplots_adjust(left=0.055, right=0.992, top=0.86, bottom=0.205)
    figure.text(
        0.055,
        0.035,
        "Development points average 4 stochastic writes on identity 94003. Target P&V uses 5 paired writes "
        "(seeds 94301–94305) on one sealed array 94004—not 5 arrays or confidence intervals. "
        "Adam continues exact P0 seed 94301 only. Fitted AIHWKit 1.1.0 IBM-OM · exploratory_noncanonical.",
        ha="left",
        va="bottom",
        color=GRAY,
        fontsize=7.8,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Date": None,
            "Creator": "plot_ibm_om_corrupt_multi_source_hwa_tuned_recovery.py",
            "Title": "Multi-source HWA IBM-OM successor ladder",
        },
    )
    plt.close(figure)
    if output.suffix.lower() == ".svg":
        # Matplotlib leaves spaces at the ends of multiline SVG path commands.
        # Normalize generated text so the tracked figure passes diff checks.
        svg_text = output.read_text(encoding="utf-8")
        output.write_text(
            "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    arguments = _arguments()
    summary = _load(arguments.summary)
    _validate(summary)
    _plot(summary, arguments.output)


if __name__ == "__main__":
    main()
