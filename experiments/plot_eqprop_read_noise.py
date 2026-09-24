"""Plot only collected, validated single-seed read-noise measurements."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-baseline", action="store_true",
                        help="Add the collected September 16 baseline extension")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / "paper_ready_results"
    rows = list(csv.DictReader((folder / "read_noise_run_status_20260914.csv").open()))
    if args.include_baseline:
        baseline = list(csv.DictReader((folder / "baseline_read_noise_run_status_20260916.csv").open()))
        for row in baseline:
            row["scheme"] = f"baseline β={row['beta']}"
        rows.extend(row for row in baseline if float(row["sigma"]) > 0)
    complete = [row for row in rows if row["status"] == "complete_collected_validated"
                and row["final_drop_pp"] != ""]
    if not complete:
        raise ValueError("No collected full trainings to plot")
    colors = {"ours": "#2468ac", "legacy": "#c66b22",
              "baseline β=100": "#37824a", "baseline β=200": "#8655a8",
              "baseline β=10": "#37824a"}
    markers = {"ours": "o", "legacy": "s", "baseline β=100": "^",
               "baseline β=200": "D", "baseline β=10": "^"}
    schemes = [scheme for scheme in colors if any(row["scheme"] == scheme for row in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    for depth, ax in enumerate(axes, 1):
        architecture = f"conv{depth}"
        selected = [row for row in complete if row["architecture"] == architecture]
        for scheme in schemes:
            scheme_rows = [row for row in selected if row["scheme"] == scheme]
            for group in sorted({row["noise_stream_group"] for row in scheme_rows}):
                group_rows = sorted((row for row in scheme_rows if row["noise_stream_group"] == group),
                                    key=lambda row: float(row["sigma"]))
                ampere = group.startswith(("rtx3090", "rtx3080"))
                akib = group.startswith("rtx3080")
                ax.plot([float(row["sigma"]) for row in group_rows],
                        [float(row["final_drop_pp"]) for row in group_rows],
                        color=colors[scheme], marker=markers[scheme], markersize=6,
                        markerfacecolor=colors[scheme] if akib or not ampere else "white",
                        markerfacecoloralt="white", fillstyle="left" if akib else "full",
                        linestyle=(":" if akib else "--" if ampere else "-") if len(group_rows) > 1 else "none",
                        linewidth=1.4, zorder=4 if ampere else 2)
        ax.axhline(0, color="#555555", linewidth=.8, linestyle="--", zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(7e-6, 7e-4)
        ax.set_xticks([1e-5, 3e-5, 1e-4, 3e-4, 5e-4],
                      ["1e-5", "3e-5", "1e-4", "3e-4", "5e-4"])
        ax.minorticks_off()
        ax.tick_params(axis="x", labelsize=8)
        expected = sum(row["architecture"] == architecture for row in rows)
        ax.set_title(f"Conv{depth} · {len(selected)}/{expected} complete", fontsize=11)
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        if not selected:
            ax.text(.5, .5, "Results pending", transform=ax.transAxes, ha="center", color="#777777")
    axes[0].set_ylabel("Final validation accuracy loss vs clean (pp)")
    handles = [plt.Line2D([], [], color=colors[scheme], marker=markers[scheme], label=scheme)
               for scheme in schemes]
    handles.append(plt.Line2D([], [], color="#555555", marker="o",
                             linestyle="-", label="5090"))
    handles.append(plt.Line2D([], [], color="#555555", marker="o", markerfacecolor="white",
                             linestyle="--", label="3090: local / Nom"))
    handles.append(plt.Line2D([], [], color="#555555", marker="o", markerfacecoloralt="white",
                             fillstyle="left", linestyle=":", label="3080: Akib"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.0 if args.include_baseline else .96),
               ncols=4 if args.include_baseline else 5, frameon=False, fontsize=9)
    fig.suptitle(f"EqProp read noise · seed 0 · {len(complete)}/{len(rows)} collected noisy runs", y=1.07, fontsize=12)
    fig.supxlabel("Read-noise standard deviation (simulator voltage units)", y=.055, fontsize=10)
    qualification = ("Validation only, one seed; Conv3 ours and baseline retain beta qualification caveats."
                     if args.include_baseline else
                     "Validation only, one seed; Conv3 ours retains its qualification exception.")
    fig.text(.5, -.075, "Positive values indicate degradation. Each panel has its own vertical scale. Markers are measured runs.\n"
             "Lines join points only within a noise-stream group. Some clean references use historical environments.\n"
             + qualification,
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, .10, 1, .86))
    for extension in ("png", "pdf"):
        stem = "read_noise_all_schemes_validation_20260916" if args.include_baseline else "read_noise_validation"
        path = folder / f"{stem}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
