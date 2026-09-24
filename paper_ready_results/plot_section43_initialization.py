#!/usr/bin/env python3
"""Compare the saved epoch-0 and trained epoch-30 Conv3 baseline checkpoint."""
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_section43_mechanism import distribution, read_csv, write_csv, save_all

HERE = Path(__file__).resolve().parent
PREFIX = "section43_conv3_baseline_initialization"
STUDY = HERE.parent/"results/section43-conv3-baseline-initialization-20260918-v1"
EPOCHS = (0, 30)
COLORS = {0: "#6a3d9a", 30: "#0072B2"}
LABELS = {0: "Initialization (epoch 0)", 30: "Trained (epoch 30)"}


def main():
    run = STUDY/"full"
    config = json.loads((run/"config.used.json").read_text())
    assert json.loads((run/"result.json").read_text())["terminal_metrics"]["passed"]
    previous = Path(config["trained_reference"]["run"])
    gradient_groups = defaultdict(list)
    displacement_groups = defaultdict(list)
    for epoch, source in [(0, run), (30, previous)]:
        rows = [r for r in read_csv(source/"gradients.csv") if r["architecture"] == "conv3" and r["scheme"] == "baseline"]
        assert len(rows) == 5904
        for row in rows:
            gradient_groups[epoch, int(row["layer_index"]), row["parameter"], float(row["sigma"])].append(row)
        rows = [r for r in read_csv(source/"displacement.csv") if r["architecture"] == "conv3" and r["scheme"] == "baseline"]
        assert len(rows) == 864
        for row in rows:
            displacement_groups[epoch, int(row["layer_index"]), row["state_layer"], row["kind"]].append(row)
    gradients = []
    for (epoch, layer, parameter, sigma), rows in sorted(gradient_groups.items()):
        draws = [-1] if sigma == 0 else range(8)
        assert {(int(r["batch_index"]), int(r["draw"])) for r in rows} == {(b, d) for b in range(36) for d in draws}
        assert len(rows) == 36*(1 if sigma == 0 else 8)
        result = {"epoch": epoch, "checkpoint_role": LABELS[epoch], "layer_index": layer, "parameter": parameter,
                  "sigma": sigma, "n_comparisons": len(rows)}
        for key in (k for k in rows[0] if k.startswith(("bptt_", "clean_ep_"))):
            result.update({f"{key}_{stat}": value for stat, value in distribution(r[key] for r in rows).items()})
        result.update({f"projection_onto_clean_{stat}": value for stat, value in distribution(
                       float(r["clean_ep_cosine"])*float(r["clean_ep_norm_ratio"]) for r in rows
                       if r["clean_ep_cosine"] and r["clean_ep_norm_ratio"]).items()})
        gradients.append(result)
    assert len(gradients) == 48
    displacement = []
    for (epoch, layer, name, kind), rows in sorted(displacement_groups.items()):
        assert len(rows) == 36 and {int(r["batch_index"]) for r in rows} == set(range(36))
        count = sum(int(r["element_count"]) for r in rows)
        rms = np.sqrt(sum(float(r["sum_squared_delta"]) for r in rows)/count)
        displacement.append({"epoch": epoch, "layer_index": layer, "state_layer": name, "kind": kind,
                             "element_count": count, "pooled_rms": float(rms),
                             "signed_mean": sum(float(r["sum_delta"]) for r in rows)/count})
    statistic_groups = defaultdict(list)
    for r in read_csv(run/"checkpoint_gradients.csv"):
        statistic_groups[int(r["epoch"]), r["estimator"], int(r["layer_index"]), r["parameter"]].append(r)
    statistics = []
    for (epoch, estimator, layer, name), rows in sorted(statistic_groups.items()):
        assert len(rows) == 36
        item = {"epoch": epoch, "estimator": estimator, "layer_index": layer, "parameter": name}
        for key in ("gradient_l2", "gradient_rms", "abs_le_1e_minus12_fraction", "exact_zero_fraction",
                    "norm_relative_to_initial", "cosine_to_initial_same_batch", "relative_l2_error_to_initial"):
            item.update({f"{key}_{stat}": value for stat, value in distribution(r[key] for r in rows).items()})
        statistics.append(item)
    voltage_groups = defaultdict(list)
    for r in read_csv(run/"checkpoint_voltages.csv"):
        voltage_groups[int(r["epoch"]), r["phase"], int(r["layer_index"]), r["state_layer"]].append(r)
    voltages = []
    for (epoch, phase, layer, name), rows in sorted(voltage_groups.items()):
        assert len(rows) == 36
        count = sum(int(r["element_count"]) for r in rows)
        mean = sum(float(r["state_sum"]) for r in rows)/count
        second = sum(float(r["state_squared_sum"]) for r in rows)/count
        clamp = [r for r in rows if r["clamp_occupancy"]]
        voltages.append({"epoch": epoch, "phase": phase, "layer_index": layer, "state_layer": name,
                         "element_count": count, "state_mean": mean, "state_rms": float(np.sqrt(second)),
                         "state_std": float(np.sqrt(max(second-mean*mean, 0))),
                         "state_min": min(float(r["state_min"]) for r in rows),
                         "state_max": max(float(r["state_max"]) for r in rows),
                         "clamp_occupancy": sum(float(r["clamp_occupancy"])*int(r["element_count"]) for r in clamp)/count if clamp else None})
    for suffix, rows in (("cosine", gradients), ("displacement", displacement), ("gradients", statistics), ("voltages", voltages)):
        write_csv(HERE/f"{PREFIX}_{suffix}.csv", rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    sigmas = config["noise"]["sigmas"]
    cells = {(r["epoch"], r["layer_index"], r["sigma"]): r["bptt_cosine_mean"] for r in gradients}
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6))
    fig.suptitle("Conv3 baseline", y=1.01, fontweight="bold")
    fig.subplots_adjust(left=.09, right=.87, bottom=.22, top=.84, wspace=.32)
    for epoch, ax in zip(EPOCHS, axes):
        data = np.array([[cells[epoch, i, sigma] for sigma in sigmas] for i in range(4)])
        im = ax.imshow(data, vmin=-1, vmax=1, cmap="RdBu", aspect="auto")
        ax.set_xticks(range(6), ["0", "1", "3", "10", "30", "50"])
        ax.set_yticks(range(4), ["Conv 1", "Conv 2", "Conv 3", "Readout"])
        ax.set_title(LABELS[epoch], fontweight="bold", pad=10)
        ax.tick_params(length=0)
        for (i, j), value in np.ndenumerate(data):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > .65 else "#222222")
        for spine in ax.spines.values():
            spine.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_axes([.9, .22, .018, .62]), ticks=[-1, -.5, 0, .5, 1])
    cb.set_label("Mean cosine to noiseless BPTT")
    fig.supxlabel(r"Endpoint read-noise standard deviation $\sigma$ ($\times\,10^{-5}$)", y=.04)
    save_all(fig, HERE/"figures"/f"{PREFIX}_cosine")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    fig.suptitle("Conv3 baseline", y=1.01, fontweight="bold")
    fig.subplots_adjust(left=.085, right=.98, top=.86, bottom=.23, wspace=.3)
    for epoch in EPOCHS:
        ds = [next(r["pooled_rms"] for r in displacement if r["epoch"] == epoch and r["kind"] == "centered_halfspan" and r["layer_index"] == i) for i in range(4)]
        gs = [next(r["gradient_l2_mean"] for r in statistics if r["epoch"] == epoch and r["estimator"] == "bptt" and r["layer_index"] == i) for i in range(4)]
        for ax, values in zip(axes, (ds, gs)):
            ax.plot(range(4), values, color=COLORS[epoch], marker="o" if epoch == 0 else "s", label=LABELS[epoch], linewidth=1.8)
    axes[0].axhspan(min(sigmas[1:])/np.sqrt(2), max(sigmas)/np.sqrt(2), color="gray", alpha=.15)
    axes[0].set_title("Clean centered phase contrast")
    axes[0].set_ylabel(r"RMS of $(v^+ - v^-)/2$" + "\n(voltage units)")
    axes[0].set_xticks(range(4), ["H1", "H2", "H3", "Output"])
    axes[1].set_title("Clean BPTT gradient strength")
    axes[1].set_ylabel("Mean gradient L2 norm")
    axes[1].set_xticks(range(4), ["Conv 1", "Conv 2", "Conv 3", "Readout"])
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles + [Patch(facecolor="gray", alpha=.15)],
               labels + [r"Voltage contrast noise: $\sigma/\sqrt{2}$"],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    save_all(fig, HERE/"figures"/f"{PREFIX}_signal")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    fig.suptitle("Conv3 baseline", y=1.01, fontweight="bold")
    fig.subplots_adjust(left=.1, right=.98, top=.85, bottom=.28, wspace=.3)
    palette = plt.colormaps["viridis"](np.linspace(.05, .85, 4))
    for i in range(4):
        rows = [next(r for r in voltages if r["epoch"] == epoch and r["phase"] == "post_T_free" and r["layer_index"] == i) for epoch in EPOCHS]
        for ax, key in zip(axes, ("state_mean", "state_rms")):
            ax.plot(EPOCHS, [r[key] for r in rows], linestyle="none", marker="o", color=palette[i], label=rows[0]["state_layer"])
            ax.set_xticks(EPOCHS)
            ax.set_xlabel("Checkpoint epoch")
            ax.grid(axis="y", alpha=.25)
    axes[0].set_title("Free-state signed mean voltage")
    axes[1].set_title("Free-state voltage RMS")
    axes[0].set_ylabel("Absolute voltage units")
    axes[1].set_yscale("log")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    save_all(fig, HERE/"figures"/f"{PREFIX}_voltages")

    paths = [*HERE.glob(f"{PREFIX}_*.csv"), *(HERE/"figures").glob(f"{PREFIX}_*")]
    provenance = {"study": str(STUDY), "trained_noise_source": str(previous), "matplotlib_version": matplotlib.__version__,
                  "cosine_cells": len(gradients), "batches_per_checkpoint": 36, "official_test_read": False,
                  "artifacts": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (HERE/f"{PREFIX}_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps({"cosine_cells": len(gradients), "figures": str(HERE/"figures"/f"{PREFIX}_cosine.jpg")}, indent=2))


if __name__ == "__main__":
    main()
