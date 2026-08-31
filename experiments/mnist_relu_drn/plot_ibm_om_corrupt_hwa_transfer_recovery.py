"""Plot the corrupt-array HWA, transfer, and pulse-recovery ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected {name} to be a mapping.")
    return value


def _accuracy(value: Any, name: str) -> float:
    result = float(value) * 100.0
    if not 0.0 <= result <= 100.0:
        raise ValueError(f"Expected {name} to be an accuracy in [0,1].")
    return result


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(payload, "scientific summary")


def _annotate_points(axis: Any, x: Sequence[float], y: Sequence[float]) -> None:
    for x_value, y_value in zip(x, y, strict=True):
        axis.annotate(
            f"{y_value:.1f}%",
            (x_value, y_value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )


def plot(summary: Mapping[str, Any], output_prefix: Path) -> None:
    source = _mapping(summary["source_93001"], "source_93001")
    target = _mapping(summary["target_93002"], "target_93002")
    source_test = _mapping(source["test_after_selection_freeze"], "source test")
    ideal = _mapping(target["ideal_requested_no_write"], "target ideal")
    fault = _mapping(
        target["deterministic_fault_clamped_no_write"], "target fault-only"
    )
    program_verify = _mapping(target["program_verify"], "target P&V")
    recovery = _mapping(target["open_loop_pulse_adam"], "target recovery")

    epoch0 = [
        _accuracy(ideal["epoch_0"]["metrics"]["student_accuracy"], "epoch0 ideal"),
        _accuracy(
            fault["epoch_0"]["evaluation"]["metrics"]["student_accuracy"],
            "epoch0 fault-only",
        ),
        _accuracy(program_verify["epoch_0"]["mean"], "epoch0 P&V mean"),
    ]
    hwa = [
        _accuracy(
            ideal["selected_hwa"]["metrics"]["student_accuracy"], "HWA ideal"
        ),
        _accuracy(
            fault["selected_hwa"]["evaluation"]["metrics"]["student_accuracy"],
            "HWA fault-only",
        ),
        _accuracy(program_verify["selected_hwa"]["mean"], "HWA P&V mean"),
    ]
    source_epoch0 = _accuracy(
        source_test["epoch_0"]["persistent_mean_accuracy"], "source epoch0"
    )
    source_hwa = _accuracy(
        source_test["selected_hwa"]["persistent_mean_accuracy"], "source HWA"
    )
    p0_before = _accuracy(
        recovery["before_test"]["student_accuracy"], "target P0 before Adam"
    )
    p0_after = _accuracy(
        recovery["after_test"]["student_accuracy"], "target P0 after Adam"
    )

    blue = "#3274a1"
    orange = "#e1812c"
    gray = "#7f7f7f"
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)

    x = (0, 1, 2)
    labels = ("Ideal request\n(no write)", "Native stuck\ncells only", "Persistent P&V\nmean (5 seeds)")
    axes[0].plot(x, epoch0, marker="o", linewidth=2.4, color=gray, label="ReLU initialization")
    axes[0].plot(x, hwa, marker="o", linewidth=2.4, color=blue, label="Source-selected HWA")
    _annotate_points(axes[0], x, epoch0)
    _annotate_points(axes[0], x, hwa)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("MNIST test accuracy (%)")
    axes[0].set_title("Different target array: faults dominate the transfer loss")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="lower left")

    group_x = (0, 1)
    width = 0.32
    before = (source_epoch0, p0_before)
    after = (source_hwa, p0_after)
    axes[1].bar(
        [value - width / 2 for value in group_x],
        before,
        width,
        color=gray,
        label="Before intervention",
    )
    axes[1].bar(
        [value + width / 2 for value in group_x],
        after,
        width,
        color=(blue, orange),
        label="After intervention",
    )
    for x_value, y_value in zip(
        [group_x[0] - width / 2, group_x[1] - width / 2], before, strict=True
    ):
        axes[1].text(x_value, y_value + 2.0, f"{y_value:.1f}%", ha="center", fontsize=9)
    for x_value, y_value in zip(
        [group_x[0] + width / 2, group_x[1] + width / 2], after, strict=True
    ):
        axes[1].text(x_value, y_value + 2.0, f"{y_value:.1f}%", ha="center", fontsize=9)
    axes[1].set_xticks(
        group_x,
        ("Source P&V mean\nHWA intervention", "Target seed 93301\nAdam intervention"),
    )
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("MNIST test accuracy (%)")
    axes[1].set_title("Source HWA and exact-target pulse adaptation")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(
        handles=(
            Patch(facecolor=gray, label="Before intervention"),
            Patch(facecolor=blue, label="After source HWA"),
            Patch(facecolor=orange, label="After target pulse Adam"),
        ),
        frameon=False,
        loc="lower right",
    )

    figure.suptitle(
        "IBM-OM fitted-model ladder: one source array, one target array",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.02,
        "Exploratory, model-based AIHWKit OM; target P&V seeds are five writes on one array, not five arrays.",
        ha="center",
        fontsize=9,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    svg_path = output_prefix.with_suffix(".svg")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    # Matplotlib leaves spaces at the ends of multiline SVG path commands.
    # Normalize them so the tracked generated figure passes repository checks.
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
    plot(_load(arguments.summary), arguments.output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
