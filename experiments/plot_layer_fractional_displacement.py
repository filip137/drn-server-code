"""Plot saved initialization displacement across layers; no simulation or replay."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CASES = (("conv1", 1.0), ("conv2", 4.0), ("conv3", 6.0))
STYLES = {
    "baseline": dict(color="#2ca02c", linestyle="--", marker="o",
                     markersize=8, markerfacecolor="none", zorder=4),
    "ours": dict(color="#ff7f0e", linestyle="-", marker="^",
                 markersize=6, zorder=3),
    "legacy": dict(color="#1f77b4", linestyle=":", marker="x",
                   markersize=6, zorder=5),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path,
                        default=Path("results/conv123-relative-noise-advantage-map-20260925-v1"))
    parser.add_argument("--source-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = args.source_csv or args.study_root / "analysis/stage_a/physical_state_controls.csv"
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    selected = [r for r in rows if
                (r["architecture"], float(r["relative_target"])) in CASES]
    lookup = {}
    for r in selected:
        key = (r["architecture"], r["scheme"], int(r["state_layer_index"]))
        assert key not in lookup, f"Duplicate measurement: {key}"
        for field in ("centered_displacement_rms", "endpoint_rms", "D_over_P"):
            assert math.isfinite(float(r[field])) and float(r[field]) > 0
        assert math.isclose(float(r["D_over_P"]),
                            float(r["centered_displacement_rms"]) / float(r["endpoint_rms"]),
                            rel_tol=1e-10)
        lookup[key] = r
    assert len(lookup) == 27
    out = args.output_dir or args.study_root / "analysis/layer_displacement"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "plotted_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)

    def draw(cases, filename):
        fig, axes = plt.subplots(len(cases), 2, figsize=(11, 3.25 * len(cases) + 1.0),
                                 squeeze=False)
        for row_index, (arch, target) in enumerate(cases):
            depth = int(arch[-1])
            ticks = list(range(depth + 1))
            labels = [f"Conv {i+1}" for i in range(depth)] + ["Readout"]
            for column, (field, scale, ylabel) in enumerate([
                ("centered_displacement_rms", 1.0, "Absolute displacement D (voltage units)"),
                ("D_over_P", 100.0, "Fractional displacement D/P (%)"),
            ]):
                ax = axes[row_index, column]
                for scheme, style in STYLES.items():
                    values = [scale * float(lookup[(arch, scheme, i)][field]) for i in ticks]
                    ax.plot(ticks, values, linewidth=1.9, markeredgewidth=1.4,
                            label=scheme.capitalize(), **style)
                ax.set_yscale("log")
                ax.set_xticks(ticks, labels)
                ax.set_ylabel(ylabel)
                ax.set_title(f"{arch.capitalize()} · matched output D/F = {target:g}", fontsize=11)
                ax.grid(axis="y", which="major", alpha=.23)
                ax.spines[["top", "right"]].set_visible(False)
                ax.margins(x=.12, y=.15)
        fig.suptitle("Initialization: displacement across layers", fontsize=15, y=.985)
        handles = [Line2D([], [], label=s.capitalize(), linewidth=1.9, **style)
                   for s, style in STYLES.items()]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .947),
                   ncol=3, frameon=False)
        fig.text(.5, .025,
                 "D = RMS[(v⁺ − v⁻)/2]; P = pooled RMS of nudged endpoints; F = free-output RMS.\n"
                 "Saved clean states · seed 0 · 576 examples · logarithmic y-axes",
                 ha="center", fontsize=9)
        fig.subplots_adjust(left=.105, right=.985, bottom=.16 if len(cases)==1 else .10,
                            top=.80 if len(cases)==1 else .89, wspace=.30, hspace=.52)
        fig.savefig(out / filename, dpi=180, facecolor="white", pil_kwargs={"quality": 95})
        plt.close(fig)

    draw(CASES, "conv123_absolute_and_fractional_displacement.jpg")
    for case in CASES:
        draw((case,), f"{case[0]}_absolute_and_fractional_displacement.jpg")
    (out / "README.md").write_text(
        "# Initialization displacement across layers\n\n"
        "[All architectures](conv123_absolute_and_fractional_displacement.jpg) · "
        "[Conv1](conv1_absolute_and_fractional_displacement.jpg) · "
        "[Conv2](conv2_absolute_and_fractional_displacement.jpg) · "
        "[Conv3](conv3_absolute_and_fractional_displacement.jpg)\n\n"
        "Conv1/2/3 use output D/F targets 1/4/6, respectively, matching the selected "
        "training settings at initialization. Original source: "
        "results/conv123-relative-noise-advantage-map-20260925-v1/analysis/stage_a/physical_state_controls.csv. "
        "These physical-state measurements use 576 examples; the stage-A noisy-gradient "
        "screen's smaller cohort is not used here. No stages are pooled.\n\n"
        "D is RMS((v_plus-v_minus)/2); P is the pooled RMS of both clean nudged "
        "endpoints. Each row represents one layer and scheme, pooled over the cohort "
        "and nodes. Both y-axes are logarithmic with independently fitted limits. "
        "Lines connect discrete layers; one initializer, no seed uncertainty. "
        "Baseline uses hollow green markers so its near-overlap with legacy remains visible. "
        "No noise is injected or simulation performed by this plotting script.\n\n"
        "[Plotted measurements](plotted_values.csv). Reproduce from the repository root: "
        f"`python -m experiments.plot_layer_fractional_displacement --source-csv "
        f"{out}/plotted_values.csv --output-dir {out}`.\n"
    )
    print(f"Verified and plotted {len(selected)} layer measurements into {out}")


if __name__ == "__main__":
    main()
