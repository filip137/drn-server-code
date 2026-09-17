#!/usr/bin/env python3
"""Plot IBM OM full-RESET/full-SET and achievable-range variation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_POPULATION_SHA256 = (
    "7e0bcfb76a0a613f0ba319786a39fdb540cce0a74f15d9fb007426a46ad49d2f"
)
EXPECTED_FINGERPRINT = (
    "4887ad89abdd16193448c54a0cbe97be915cb4b9bc04cf1151c5e2a96ee5a3ac"
)
EXPECTED_ASSIGNMENT_SEED = 84001
EXPECTED_NUM_CELLS = 158_800
ARRAY_WIDE_MINIMUM = -3.455834150314331
ARRAY_WIDE_MAXIMUM = 2.5384607315063477
ARRAY_WIDE_SPAN = ARRAY_WIDE_MAXIMUM - ARRAY_WIDE_MINIMUM


def _repository() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_population() -> Path:
    return _repository() / (
        "results/mnist-ibm-om-common-window-hwa-program-verify-pilot-"
        "20260824-v2/runs/train-exact-bounds-hwa/"
        "20260824T092611.622367Z-d7ba7637-7565051f/artifacts/"
        "ibm_om_population.training.npz"
    )


def _default_output() -> Path:
    return _repository() / "docs" / "ibm_om_cell_bound_variation.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_population(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _sha256(path) != EXPECTED_POPULATION_SHA256:
        raise ValueError(f"Unexpected population digest: {path}")

    with np.load(path, allow_pickle=False) as payload:
        if int(payload["assignment_seed"].item()) != EXPECTED_ASSIGNMENT_SEED:
            raise ValueError("Unexpected assignment seed")
        if str(payload["corruption_policy"].item()) != "counterfactual_repaired":
            raise ValueError("Expected the counterfactually repaired population")
        if str(payload["fingerprint"].item()) != EXPECTED_FINGERPRINT:
            raise ValueError("Unexpected population fingerprint")
        minimum = payload["min_bound"].astype(np.float64)
        maximum = payload["max_bound"].astype(np.float64)

    if minimum.shape != (EXPECTED_NUM_CELLS,) or maximum.shape != minimum.shape:
        raise ValueError("Unexpected population shape")
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise ValueError("Population bounds must be finite")
    if np.any(maximum <= minimum):
        raise ValueError("Every repaired cell must have a non-empty interval")

    reset = (minimum - ARRAY_WIDE_MINIMUM) / ARRAY_WIDE_SPAN
    set_state = (maximum - ARRAY_WIDE_MINIMUM) / ARRAY_WIDE_SPAN
    achievable_span = set_state - reset
    return reset, set_state, achievable_span


def _central_90(values: np.ndarray) -> tuple[float, float, float]:
    lower, median, upper = np.quantile(values, (0.05, 0.50, 0.95))
    return float(lower), float(median), float(upper)


def plot(population: Path, output: Path) -> None:
    reset, set_state, achievable_span = _load_population(population)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, (bounds_axis, span_axis) = plt.subplots(
        1,
        2,
        figsize=(8.6, 4.2),
        constrained_layout=False,
        gridspec_kw={"width_ratios": (1.15, 1.0)},
    )
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.76, wspace=0.22)
    figure.suptitle(
        "Cell-to-cell variation in full RESET/SET states and achievable range",
        fontsize=14,
        y=0.97,
    )
    figure.text(
        0.5,
        0.885,
        "Repaired development assignment · seed 84001 · 158,800 cells · shared global $g$ coordinate",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#555555",
    )

    bound_bins = np.linspace(0.0, 1.0, 81)
    bounds_axis.hist(
        reset,
        bins=bound_bins,
        density=True,
        color="#339cff",
        alpha=0.48,
        edgecolor="none",
        label=r"Full RESET $g_{\min}$",
    )
    bounds_axis.hist(
        set_state,
        bins=bound_bins,
        density=True,
        color="#e25507",
        alpha=0.42,
        edgecolor="none",
        label=r"Full SET $g_{\max}$",
    )
    for values, color in ((reset, "#176cae"), (set_state, "#a63b03")):
        lower, median, upper = _central_90(values)
        bounds_axis.axvspan(lower, upper, color=color, alpha=0.07)
        bounds_axis.axvline(median, color=color, linewidth=1.5, linestyle="--")
    reset_q = _central_90(reset)
    set_q = _central_90(set_state)
    bounds_axis.text(
        0.02,
        0.97,
        (
            f"RESET p5–p95: {reset_q[0]:.3f}–{reset_q[2]:.3f}\n"
            f"SET p5–p95: {set_q[0]:.3f}–{set_q[2]:.3f}"
        ),
        transform=bounds_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )
    bounds_axis.set_title("Full boundary-state distributions", fontsize=11)
    bounds_axis.set_xlabel("Boundary state in global $g$")
    bounds_axis.set_ylabel("Probability density")
    bounds_axis.set_xlim(0.0, 1.0)
    bounds_axis.legend(loc="upper right", frameon=False, fontsize=8.5)

    span_bins = np.linspace(0.0, 0.85, 69)
    span_axis.hist(
        achievable_span,
        bins=span_bins,
        density=True,
        color="#339cff",
        alpha=0.68,
        edgecolor="none",
    )
    span_lower, span_median, span_upper = _central_90(achievable_span)
    span_axis.axvspan(span_lower, span_upper, color="#339cff", alpha=0.10)
    span_axis.axvline(
        span_median,
        color="#176cae",
        linewidth=1.7,
        linestyle="--",
        label=f"median = {span_median:.3f}",
    )
    span_axis.text(
        0.98,
        0.97,
        f"p5–p95: {span_lower:.3f}–{span_upper:.3f}",
        transform=span_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
    )
    span_axis.set_title("Per-cell achievable span", fontsize=11)
    span_axis.set_xlabel(r"$g_{\max}-g_{\min}$")
    span_axis.set_ylabel("Probability density")
    span_axis.set_xlim(0.0, 0.85)
    span_axis.legend(loc="upper left", frameon=False, fontsize=8.5)

    for axis in (bounds_axis, span_axis):
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.65)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, default=_default_population())
    parser.add_argument("--output", type=Path, default=_default_output())
    arguments = parser.parse_args()
    plot(arguments.population.resolve(), arguments.output.resolve())


if __name__ == "__main__":
    main()
