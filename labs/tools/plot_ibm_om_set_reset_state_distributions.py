#!/usr/bin/env python3
"""Plot full IBM OM SET/RESET distributions with and without reference offset."""

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
    return _repository() / "docs" / "ibm_om_set_reset_state_distributions.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_population(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _sha256(path) != EXPECTED_POPULATION_SHA256:
        raise ValueError(f"Unexpected population digest: {path}")

    with np.load(path, allow_pickle=False) as payload:
        if int(payload["assignment_seed"].item()) != EXPECTED_ASSIGNMENT_SEED:
            raise ValueError("Unexpected assignment seed")
        if str(payload["corruption_policy"].item()) != "counterfactual_repaired":
            raise ValueError("Expected the counterfactually repaired population")
        if str(payload["fingerprint"].item()) != EXPECTED_FINGERPRINT:
            raise ValueError("Unexpected population fingerprint")
        reset_a = payload["min_bound"].astype(np.float64)
        set_a = payload["max_bound"].astype(np.float64)
        reference = payload["reference"].astype(np.float64)

    expected_shape = (EXPECTED_NUM_CELLS,)
    if (
        reset_a.shape != expected_shape
        or set_a.shape != expected_shape
        or reference.shape != expected_shape
    ):
        raise ValueError("Unexpected population shape")
    if not all(
        np.isfinite(values).all() for values in (reset_a, set_a, reference)
    ):
        raise ValueError("Population state tensors must be finite")
    if np.any(set_a <= reset_a):
        raise ValueError("Every repaired cell must have a non-empty interval")
    return reset_a, set_a, reference


def _range_text(reset: np.ndarray, set_state: np.ndarray) -> str:
    reset_min, reset_median, reset_max = np.quantile(reset, (0.0, 0.5, 1.0))
    set_min, set_median, set_max = np.quantile(set_state, (0.0, 0.5, 1.0))
    return (
        f"RESET min/median/max: {reset_min:.3f} / {reset_median:.3f} / {reset_max:.3f}\n"
        f"SET min/median/max: {set_min:.3f} / {set_median:.3f} / {set_max:.3f}"
    )


def plot(population: Path, output: Path) -> None:
    reset_a, set_a, reference = _load_population(population)
    panels = (
        (r"(a) Offset-subtracted state  $w=a-r$", reset_a - reference, set_a - reference),
        (r"(b) Pure active state  $a$", reset_a, set_a),
    )
    all_values = np.concatenate(
        [values for _, reset, set_state in panels for values in (reset, set_state)]
    )
    minimum = float(all_values.min())
    maximum = float(all_values.max())
    padding = 0.025 * (maximum - minimum)
    bins = np.linspace(minimum - padding, maximum + padding, 141)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 5.0),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.68, wspace=0.16)
    figure.suptitle(
        "IBM OM full SET and RESET endpoint distributions",
        fontsize=14,
        y=0.97,
    )
    figure.text(
        0.5,
        0.895,
        (
            "Repaired development assignment · seed 84001 · 158,800 cells · "
            "common bins and axes · complete observed range"
        ),
        ha="center",
        va="top",
        fontsize=8.5,
        color="#555555",
    )

    for axis, (title, reset, set_state) in zip(axes, panels, strict=True):
        axis.hist(
            reset,
            bins=bins,
            color="#339cff",
            alpha=0.26,
            edgecolor="#176cae",
            linewidth=0.8,
            label="Full RESET endpoint",
        )
        axis.hist(
            set_state,
            bins=bins,
            color="#e25507",
            alpha=0.22,
            edgecolor="#b74405",
            linewidth=0.8,
            label="Full SET endpoint",
        )
        axis.axvline(0.0, color="#777777", linewidth=0.9, linestyle="--")
        axis.text(
            0.025,
            0.97,
            _range_text(reset, set_state),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.3,
        )
        axis.set_title(title, fontsize=11, pad=10)
        axis.set_xlabel("AIHWKit normalized model-state coordinate")
        axis.set_yscale("symlog", linthresh=20.0, linscale=0.8)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.65)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel("Cells per common-width bin (symlog scale)")
    axes[0].set_xlim(float(bins[0]), float(bins[-1]))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.79),
        ncol=2,
        frameon=False,
        fontsize=8.3,
    )

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
