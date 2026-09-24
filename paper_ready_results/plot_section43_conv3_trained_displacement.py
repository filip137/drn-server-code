#!/usr/bin/env python3
"""Verify and display trained Conv3 displacement from the existing phase replay."""
from collections import defaultdict
import json
from pathlib import Path
import sys

from plot_section43_mechanism import COLORS, LABELS, MARKERS, SCHEMES, read_csv, write_csv, save_all
from plot_section43_conv3_gradient_magnitudes import sha256
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent/"results/section43-eqprop-mechanism-20260918-v1/full-1"
STEM = "section43_conv3_trained_displacement"


def main():
    sys.path.insert(0, str(HERE.parent))
    from experiments.reporting import validate_run
    assert not validate_run(RUN)
    sources = {r["scheme"]: r for r in read_csv(HERE/"section43_checkpoint_sources.csv")
               if r["architecture"] == "conv3"}
    for source in sources.values():
        assert sha256(Path(source["run_dir"])/"weights_best.npz") == source["weights_best_npz_sha256"]
    groups = defaultdict(list)
    for r in read_csv(RUN/"displacement.csv"):
        if r["architecture"] == "conv3":
            groups[r["scheme"], r["state_layer"], r["kind"]].append(r)
    rows = [r for r in read_csv(HERE/"section43_phase_displacement_summary.csv") if r["architecture"] == "conv3"]
    assert len(rows) == len(groups) == 72
    for r in rows:
        raw = groups[r["scheme"], r["state_layer"], r["kind"]]
        assert len(raw) == 36 and {int(x["batch_index"]) for x in raw} == set(range(36))
        count = sum(int(x["element_count"]) for x in raw)
        rms = np.sqrt(sum(float(x["sum_squared_delta"]) for x in raw)/count)
        mean = sum(float(x["sum_delta"]) for x in raw)/count
        assert count == int(r["element_count"])
        np.testing.assert_allclose([rms, mean], [float(r["pooled_rms"]), float(r["signed_mean"])], rtol=1e-12)
        r["epoch"] = int(sources[r["scheme"]]["best_epoch"])
    write_csv(HERE/f"{STEM}.csv", rows)
    cells = {(r["scheme"], int(r["layer_index"]), r["kind"]): float(r["pooled_rms"]) for r in rows}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 4.2), sharey=True)
    fig.subplots_adjust(left=.10, right=.985, bottom=.25, top=.85, wspace=.18)
    for ax, kind, title in zip(axes,
        ("positive_minus_free", "centered_halfspan"),
        (r"Free → positive nudged: $v^+ - v_{\rm free}$",
         r"Centered contrast: $(v^+ - v^-)/2$"), strict=True):
        for scheme in SCHEMES:
            ax.plot(range(4), [cells[scheme, i, kind] for i in range(4)],
                    color=COLORS[scheme], marker=MARKERS[scheme], linewidth=1.8, label=LABELS[scheme])
        ax.set_yscale("log")
        ax.set_ylim(1e-10, .2)
        ax.set_xticks(range(4), ["H1", "H2", "H3", "Output"])
        ax.set_title(title, fontsize=11, pad=12)
        ax.grid(axis="y", alpha=.2, linewidth=.5)
        ax.set_xlabel("State layer (input → output)")
    axes[0].set_ylabel("Displacement RMS (voltage units)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               bbox_to_anchor=(.5, .065), frameon=False)
    fig.text(.5, .015, "Trained Conv3 · RMS pooled over 576 matched validation examples",
             ha="center", fontsize=9, color=".3")
    save_all(fig, HERE/"figures"/STEM)
    paths = [HERE/f"{STEM}.csv", *sorted((HERE/"figures").glob(STEM+".*"))]
    provenance = {"source_run": str(RUN.resolve()), "new_simulations": 0,
        "raw_rows_verified": sum(map(len, groups.values())), "summary_cells": len(rows),
        "checkpoint_sources": sources,
        "definition": "sqrt(sum(delta_v**2)/number_of_state_coordinates), pooled over samples, channels and spatial positions",
        "source_hashes": {str(p): sha256(p) for p in (
            RUN/"displacement.csv", RUN/"cohort.json", RUN/"config.used.json", RUN/"result.json",
            HERE/"section43_phase_displacement_summary.csv", Path(__file__))},
        "artifacts": {str(p.relative_to(HERE)): sha256(p) for p in paths}}
    (HERE/f"{STEM}_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps({"raw_rows_verified": provenance["raw_rows_verified"], "summary_cells": len(rows),
                      "figure": str(HERE/"figures"/f"{STEM}.jpg")}, indent=2))
    for i in range(4):
        print(i, "balanced/legacy centered RMS", cells["ours", i, "centered_halfspan"]/cells["legacy", i, "centered_halfspan"])


if __name__ == "__main__":
    main()
