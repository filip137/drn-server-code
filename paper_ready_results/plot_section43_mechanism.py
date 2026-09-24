#!/usr/bin/env python3
"""Aggregate the declared replay coverage and render the two Section 4.3 figures."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMES = ("baseline", "ours", "legacy")
LABELS = {"baseline": "Baseline", "ours": "Balanced", "legacy": "Legacy"}
COLORS = {"baseline": "#0072B2", "ours": "#D55E00", "legacy": "#009E73"}
MARKERS = {"baseline": "o", "ours": "s", "legacy": "^"}


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values):
    values = np.asarray([float(v) for v in values if v not in (None, "")])
    assert np.all(np.isfinite(values))
    if not len(values):
        return {"n_defined": 0, "mean": None, "median": None, "p10": None, "p90": None, "std": None}
    return {"n_defined": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.}


def aggregate(run, config):
    gradients = read_csv(run/"gradients.csv")
    assert len(gradients) == config["completion"]["gradient_rows"]
    grouped = defaultdict(list)
    seen = set()
    for row in gradients:
        key = (row["architecture"], row["scheme"], int(row["layer_index"]), row["parameter"], float(row["sigma"]))
        identity = (*key, int(row["batch_index"]), int(row["draw"]))
        assert identity not in seen, f"Duplicate comparison: {identity}"
        seen.add(identity)
        grouped[key].append(row)
    summary = []
    for depth in (1, 2, 3):
        for scheme in SCHEMES:
            for layer in range(depth+1):
                parameter = f"ConvWeight_{layer}" if layer < depth else "DenseWeight_0"
                for sigma in config["noise"]["sigmas"]:
                    rows = grouped[(f"conv{depth}", scheme, layer, parameter, sigma)]
                    draws = [-1] if sigma == 0 else range(config["noise"]["draws"])
                    expected = {(batch, draw) for batch in range(36) for draw in draws}
                    assert {(int(r["batch_index"]), int(r["draw"])) for r in rows} == expected
                    item = {"architecture": f"conv{depth}", "scheme": scheme, "layer_index": layer,
                            "parameter": parameter, "sigma": sigma, "n_comparisons": len(rows), "n_batches": 36}
                    for name in [k for k in rows[0] if k.startswith(("bptt_", "clean_ep_"))]:
                        item.update({f"{name}_{stat}": value for stat, value in distribution(r[name] for r in rows).items()})
                    summary.append(item)
    assert len(summary) == len(grouped) == config["completion"]["aggregate_cells"]
    displacements = read_csv(run/"displacement.csv")
    groups = defaultdict(list)
    for row in displacements:
        key = (row["architecture"], row["scheme"], int(row["layer_index"]), row["state_layer"], row["kind"])
        groups[key].append(row)
    displacement_summary = []
    for key, rows in sorted(groups.items()):
        assert {int(r["batch_index"]) for r in rows} == set(range(36)) and len(rows) == 36
        count = sum(int(r["element_count"]) for r in rows)
        squared = sum(float(r["sum_squared_delta"]) for r in rows)
        displacement_summary.append(dict(zip(("architecture", "scheme", "layer_index", "state_layer", "kind"), key)) |
                                    {"element_count": count, "sum_squared_delta": squared,
                                     "pooled_rms": math.sqrt(squared/count),
                                     "signed_mean": sum(float(r["sum_delta"]) for r in rows)/count,
                                     "mean_absolute_delta": sum(float(r["mean_absolute_delta"])*int(r["element_count"]) for r in rows)/count,
                                     **{f"batch_rms_{k}": v for k, v in distribution(r["rms"] for r in rows).items()}})
    outputs = {(r["architecture"], r["scheme"], r["kind"]): r["pooled_rms"]
               for r in displacement_summary if r["state_layer"] == "Output"}
    for row in displacement_summary:
        output = outputs[row["architecture"], row["scheme"], row["kind"]]
        row["rms_over_output_rms"] = row["pooled_rms"]/output if output > 0 else None
    assert len(displacement_summary) == 27*6
    return summary, displacement_summary


def save_all(fig, stem):
    for suffix in ("pdf", "jpg", "png", "svg"):
        options = {"pil_kwargs": {"quality": 95}} if suffix == "jpg" else {}
        fig.savefig(stem.with_suffix("."+suffix), dpi=300, facecolor="white",
                    bbox_inches="tight", pad_inches=.12, **options)
    plt.close(fig)


def render(summary, displacement, config, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    sigmas = config["noise"]["sigmas"]
    noise_labels = ["0", "1", "3", "10", "30", "50"]
    cells = {(r["architecture"], r["scheme"], r["layer_index"], r["sigma"]): r for r in summary}
    fig, axes = plt.subplots(3, 3, figsize=(10.4, 6.6))
    fig.subplots_adjust(left=.13, right=.895, bottom=.13, top=.93, hspace=.26, wspace=.38)
    cmap = plt.colormaps["RdBu"].copy()
    cmap.set_bad("#dddddd")
    for column, depth in enumerate((1, 2, 3)):
        for row, scheme in enumerate(SCHEMES):
            ax = axes[row, column]
            values = np.array([[cells[f"conv{depth}", scheme, layer, sigma]["bptt_cosine_mean"]
                                for sigma in sigmas] for layer in range(depth+1)], dtype=float)
            im = ax.imshow(values, vmin=-1, vmax=1, cmap=cmap, aspect="auto", interpolation="nearest")
            ax.set_xticks(range(len(sigmas)), noise_labels, fontsize=10)
            ax.set_yticks(range(depth+1), [f"Conv {i+1}" for i in range(depth)] + ["Readout"], fontsize=10)
            ax.tick_params(length=0)
            if row != 2:
                ax.tick_params(labelbottom=False)
            if row == 0:
                ax.set_title(f"Conv{depth}", pad=10, fontweight="bold")
            if column == 0:
                ax.text(-.34, .5, LABELS[scheme], transform=ax.transAxes, ha="center", va="center",
                        rotation=90, fontsize=12, fontweight="bold", color=COLORS[scheme])
            for (i, j), value in np.ndenumerate(values):
                label = "—" if np.isnan(value) else f"{0. if abs(value)<.005 else value:.2f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=10,
                        color="white" if np.isfinite(value) and abs(value) > .65 else "#222222")
            for spine in ax.spines.values():
                spine.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_axes([.925, .18, .015, .7]), ticks=[-1, -.5, 0, .5, 1])
    cb.set_label("Mean cosine similarity to noiseless BPTT", fontsize=10)
    fig.supxlabel(r"Endpoint read-noise standard deviation $\sigma$ ($\times\,10^{-5}$)", y=.035, fontsize=11)
    save_all(fig, output/"section43_gradient_cosine")

    contrast = {(r["architecture"], r["scheme"], r["layer_index"]): r["pooled_rms"]
                for r in displacement if r["kind"] == "centered_halfspan"}
    lower, upper = min(sigmas[1:])/math.sqrt(2), max(sigmas)/math.sqrt(2)
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.0), sharey=True)
    fig.subplots_adjust(left=.095, right=.985, bottom=.28, top=.86, wspace=.13)
    for ax, depth in zip(axes, (1, 2, 3), strict=True):
        ax.axhspan(lower, upper, color="#bdbdbd", alpha=.3, zorder=0)
        for sigma in sigmas[1:]:
            ax.axhline(sigma/math.sqrt(2), color="#bbbbbb", linewidth=.45, zorder=0)
        for scheme in SCHEMES:
            values = [contrast[f"conv{depth}", scheme, i] for i in range(depth+1)]
            ax.plot(range(depth+1), values, color=COLORS[scheme], marker=MARKERS[scheme], markersize=5.5,
                    linewidth=1.8, label=LABELS[scheme], zorder=3)
        ax.set_yscale("log")
        ax.set_title(f"Conv{depth}", fontweight="bold")
        ax.set_xticks(range(depth+1), [f"H{i+1}" for i in range(depth)] + ["Output"])
        ax.set_xlim(-.15, depth+.15)
        ax.grid(axis="y", which="major", linewidth=.45, alpha=.3)
        ax.set_xlabel("State layer (input → output)", labelpad=8)
    axes[0].set_ylabel(r"RMS of $(v^+ - v^-)/2$" + "\n(absolute voltage units)")
    minimum, maximum = min(contrast.values()), max(contrast.values())
    axes[0].set_ylim(10**math.floor(math.log10(min(minimum, lower))),
                     10**math.ceil(math.log10(max(maximum, upper))))
    handles = [Line2D([], [], color=COLORS[s], marker=MARKERS[s], label=LABELS[s], linewidth=1.8) for s in SCHEMES]
    handles.append(Patch(facecolor="#bdbdbd", alpha=.3, label=r"Voltage contrast noise scale: $\sigma/\sqrt{2}$"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005), ncol=4, frameon=False, fontsize=9)
    save_all(fig, output/"section43_phase_displacement")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE.parent/"results/section43-eqprop-mechanism-20260918-v1/full-1")
    parser.add_argument("--output", type=Path, default=HERE/"figures")
    args = parser.parse_args()
    result = json.loads((args.run/"result.json").read_text())
    assert result["terminal_metrics"]["case_count"] == 9
    config = json.loads((args.run/"config.used.json").read_text())
    summary, displacement = aggregate(args.run, config)
    args.output.mkdir(exist_ok=True, parents=True)
    write_csv(HERE/"section43_gradient_cosine_summary.csv", summary)
    write_csv(HERE/"section43_phase_displacement_summary.csv", displacement)
    render(summary, displacement, config, args.output)
    supporting_files = [HERE/name for name in (
        "section43_checkpoint_sources.csv", "section43_mechanism_captions.tex",
        "section43_mechanism_verification.json", "section43_mechanism.md") if (HERE/name).exists()]
    provenance = {"source_run": str(args.run.resolve()), "config_sha256": hashlib.sha256((args.run/"config.used.json").read_bytes()).hexdigest(),
                  "result_sha256": hashlib.sha256((args.run/"result.json").read_bytes()).hexdigest(),
                  "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "matplotlib_version": matplotlib.__version__, "gradient_rows": 39852, "cosine_cells": len(summary),
                  "aggregation": "mean of individual minibatch/draw cosines; pooled per-coordinate RMS displacement",
                  "spread": "percentiles describe minibatch/noise variation, not uncertainty across training seeds",
                  "artifacts": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [*args.output.glob("section43_*"), HERE/"section43_gradient_cosine_summary.csv",
                                          HERE/"section43_phase_displacement_summary.csv", *supporting_files]}}
    (HERE/"section43_mechanism_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps({"figures": str(args.output), "cosine_cells": len(summary), "displacement_groups": len(displacement)}, indent=2))


if __name__ == "__main__":
    main()
