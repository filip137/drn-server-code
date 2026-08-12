"""Plot the accepted Conv3 centered-float64 voltage-read-noise comparison.

This is a read-only presentation layer over the validated production CSVs.  It
does not rerun equilibrium, inject new noise, or alter the canonical run bundle.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    ROOT
    / "results/perfectdiode-conv3-centered-float64-voltage-read-noise-20260811-v1"
    / "production"
)
DEFAULT_OUTPUT = DEFAULT_RUN.parent / "analysis/voltage_read_noise_joint_comparison.png"

SCHEMES = ("baseline", "ours", "legacy")
ROLES = ("reconstructed_initialization", "best_validation")
COLORS = {"baseline": "#4C78A8", "ours": "#F58518", "legacy": "#E45756"}
LINESTYLES = {"reconstructed_initialization": "--", "best_validation": "-"}
MARKER_FACE = {"reconstructed_initialization": "none", "best_validation": None}
ROLE_LABEL = {
    "reconstructed_initialization": "initialization",
    "best_validation": "best checkpoint",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _curve_rows(
    rows: list[dict[str, str]], scheme: str, role: str
) -> list[dict[str, str]]:
    return sorted(
        (
            row
            for row in rows
            if row["scheme"] == scheme
            and row["checkpoint_role"] == role
            and float(row["sigma_voltage"]) > 0.0
        ),
        key=lambda row: float(row["sigma_voltage"]),
    )


def _threshold_row(
    rows: list[dict[str, str]], scheme: str, role: str
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["scheme"] == scheme and row["checkpoint_role"] == role
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one threshold row for {scheme}/{role}.")
    return matches[0]


def _row_at_sigma(rows: list[dict[str, str]], sigma: float) -> dict[str, str]:
    matches = [row for row in rows if float(row["sigma_voltage"]) == sigma]
    if len(matches) != 1:
        raise ValueError(f"Expected one curve point at sigma={sigma:g}.")
    return matches[0]


def plot(run_dir: Path, output: Path) -> None:
    curves = _read_csv(run_dir / "configuration_summary.csv")
    thresholds = _read_csv(run_dir / "noise_thresholds.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.2), constrained_layout=True)
    cosine_ax, norm_ax, scale_ax = axes

    for scheme in SCHEMES:
        for role in ROLES:
            rows = _curve_rows(curves, scheme, role)
            if not rows:
                raise ValueError(f"Missing curve for {scheme}/{role}.")
            sigma = [float(row["sigma_voltage"]) for row in rows]
            cosine = [float(row["minimum_layer_task_cosine_p05"]) for row in rows]
            norm = [
                float(row["maximum_layer_task_symmetric_norm_delta_p95"])
                for row in rows
            ]
            common = dict(
                color=COLORS[scheme],
                linestyle=LINESTYLES[role],
                linewidth=2.1,
                marker="o",
                markersize=3.8,
                markerfacecolor=(
                    "none" if MARKER_FACE[role] == "none" else COLORS[scheme]
                ),
                markeredgewidth=1.0,
            )
            cosine_ax.plot(sigma, cosine, **common)
            norm_ax.plot(sigma, norm, **common)

            threshold = _threshold_row(thresholds, scheme, role)
            usable_sigma = float(threshold["usable_maximum_sustained_sigma"])
            threshold_curve_row = _row_at_sigma(rows, usable_sigma)
            cosine_ax.scatter(
                usable_sigma,
                float(threshold_curve_row["minimum_layer_task_cosine_p05"]),
                marker="*",
                s=125,
                color=COLORS[scheme],
                edgecolor="black",
                linewidth=0.55,
                zorder=5,
            )
            norm_ax.scatter(
                usable_sigma,
                float(
                    threshold_curve_row[
                        "maximum_layer_task_symmetric_norm_delta_p95"
                    ]
                ),
                marker="*",
                s=125,
                color=COLORS[scheme],
                edgecolor="black",
                linewidth=0.55,
                zorder=5,
            )

    for axis in (cosine_ax, norm_ax):
        axis.set_xscale("log")
        axis.grid(True, which="major", alpha=0.27)
        axis.grid(True, which="minor", alpha=0.09)
        axis.set_xlabel(r"Endpoint voltage-read noise $\sigma_v$")

    cosine_ax.axhline(0.99, color="0.25", linestyle=":", linewidth=1.35)
    cosine_ax.set_ylim(-0.18, 1.015)
    cosine_ax.set_ylabel("Worst-layer p05 cosine vs BPTT")
    cosine_ax.set_title("Gradient direction")
    cosine_ax.text(
        1.1e-12,
        0.975,
        "pass ≥ 0.99",
        fontsize=9,
        color="0.3",
        va="top",
    )

    norm_ax.axhline(0.1, color="0.25", linestyle=":", linewidth=1.35)
    norm_ax.set_yscale("log")
    norm_ax.set_ylabel("Worst-layer p95 symmetric norm error")
    norm_ax.set_title("Gradient magnitude")
    norm_ax.text(1.1e-12, 0.108, "pass ≤ 0.1", fontsize=9, color="0.3", va="bottom")

    role_offset = {"reconstructed_initialization": -0.13, "best_validation": 0.13}
    for scheme_index, scheme in enumerate(SCHEMES):
        for role in ROLES:
            row = _threshold_row(thresholds, scheme, role)
            x = scheme_index + role_offset[role]
            sigma = float(row["usable_maximum_sustained_sigma"])
            displacement = float(
                row["minimum_clean_positive_minus_negative_state_delta_rms"]
            )
            scale_ax.plot(
                [x, x],
                [sigma, displacement],
                color=COLORS[scheme],
                linestyle=LINESTYLES[role],
                linewidth=1.6,
                alpha=0.8,
            )
            scale_ax.scatter(
                x,
                sigma,
                marker="o",
                s=62,
                facecolor=(
                    "none" if role == "reconstructed_initialization" else COLORS[scheme]
                ),
                edgecolor=COLORS[scheme],
                linewidth=1.5,
                zorder=4,
            )
            scale_ax.scatter(
                x,
                displacement,
                marker="^",
                s=70,
                facecolor=(
                    "none" if role == "reconstructed_initialization" else COLORS[scheme]
                ),
                edgecolor=COLORS[scheme],
                linewidth=1.5,
                zorder=4,
            )

    scale_ax.set_yscale("log")
    scale_ax.set_xlim(-0.5, len(SCHEMES) - 0.5)
    scale_ax.set_xticks(range(len(SCHEMES)), SCHEMES)
    scale_ax.set_ylabel("Normalized voltage RMS")
    scale_ax.set_title("Tolerated noise vs clean phase separation")
    scale_ax.grid(True, axis="y", which="major", alpha=0.27)
    scale_ax.grid(True, axis="y", which="minor", alpha=0.09)
    scale_ax.annotate(
        "ours tolerates 1000×\nmore than legacy",
        xy=(2.13, 3e-10),
        xytext=(1.2, 1.5e-9),
        arrowprops=dict(arrowstyle="->", color="0.35", linewidth=1.0),
        fontsize=9,
        ha="center",
    )

    scheme_handles = [
        Line2D([0], [0], color=COLORS[name], linewidth=2.4, label=name)
        for name in SCHEMES
    ]
    role_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle=LINESTYLES[role],
            marker="o",
            markerfacecolor=("none" if role == "reconstructed_initialization" else "0.25"),
            label=ROLE_LABEL[role],
        )
        for role in ROLES
    ]
    meaning_handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="0.45", markeredgecolor="black", markersize=11, label=r"maximum usable $\sigma_v$ on curve"),
        Line2D([0], [0], marker="o", color="none", markeredgecolor="0.35", markersize=7, label=r"usable $\sigma_v$"),
        Line2D([0], [0], marker="^", color="none", markeredgecolor="0.35", markersize=8, label=r"minimum clean RMS$(V_+-V_-)$"),
    ]
    fig.legend(
        handles=[*scheme_handles, *role_handles, *meaning_handles],
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Conv3 centered EqProp: endpoint-voltage read-noise sensitivity\n"
        "native float64, T=K=64, fixed 16-example cohort, weights only",
        fontsize=15,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.run_dir.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
