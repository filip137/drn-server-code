#!/usr/bin/env python3
"""Summarize trained Conv3 gradient magnitudes from the existing read-only replay."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

from plot_section43_mechanism import (
    COLORS, LABELS, MARKERS, SCHEMES, distribution, read_csv, save_all, write_csv,
)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "results/section43-eqprop-mechanism-20260918-v1/full-1"
STEM = "section43_conv3_gradient_magnitudes"
PARAMETERS = ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2", "DenseWeight_0")
LAYERS = ("Conv 1", "Conv 2", "Conv 3", "Readout")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.path.insert(0, str(HERE.parent))
    from experiments.reporting import validate_run

    errors = validate_run(RUN)
    assert not errors, errors
    config = json.loads((RUN / "config.used.json").read_text())
    sources = {r["scheme"]: r for r in read_csv(HERE / "section43_checkpoint_sources.csv")
               if r["architecture"] == "conv3"}
    shapes = {}
    for scheme, source in sources.items():
        checkpoint = Path(source["run_dir"]) / "weights_best.npz"
        assert sha256(checkpoint) == source["weights_best_npz_sha256"]
        with np.load(checkpoint, allow_pickle=False) as archive:
            shapes[scheme] = {p: archive[p].shape for p in PARAMETERS}
    assert shapes["baseline"] == shapes["ours"] == shapes["legacy"]

    groups = defaultdict(list)
    for row in read_csv(RUN / "gradients.csv"):
        if row["architecture"] == "conv3":
            groups[row["scheme"], row["parameter"], float(row["sigma"])].append(row)
    old = {(r["scheme"], r["parameter"], float(r["sigma"])): r
           for r in read_csv(HERE / "section43_gradient_cosine_summary.csv")
           if r["architecture"] == "conv3"}
    summary = []
    for scheme in SCHEMES:
        for layer, parameter in enumerate(PARAMETERS):
            count = int(np.prod(shapes[scheme][parameter]))
            for sigma in config["noise"]["sigmas"]:
                key = scheme, parameter, sigma
                rows = groups[key]
                draws = [-1] if sigma == 0 else range(config["noise"]["draws"])
                expected = {(b, d) for b in range(config["dataset"]["batch_count"]) for d in draws}
                assert len(rows) == len(expected)
                assert {(int(r["batch_index"]), int(r["draw"])) for r in rows} == expected
                item = dict(architecture="conv3", scheme=scheme, epoch=int(sources[scheme]["best_epoch"]),
                            checkpoint_role="maximum_validation_accuracy", parameter=parameter,
                            layer_index=layer, parameter_count=count, sigma=sigma,
                            n_batches=config["dataset"]["batch_count"], n_comparisons=len(rows))
                for name, field in (("ep", "bptt_estimate_l2"),
                                    ("bptt", "bptt_reference_l2"),
                                    ("clean_ep", "clean_ep_reference_l2")):
                    stats = distribution(r[field] for r in rows)
                    for stat, value in stats.items():
                        if stat == "n_defined":
                            assert value == len(rows)
                            continue
                        np.testing.assert_allclose(value, float(old[key][f"{field}_{stat}"]), rtol=1e-12)
                        item[f"{name}_l2_{stat}"] = value
                        item[f"{name}_rms_{stat}"] = value / np.sqrt(count)
                for name, field in (("ep_over_clean_ep", "clean_ep_norm_ratio"),
                                    ("ep_zero_fraction", "bptt_estimate_zero_fraction")):
                    for stat, value in distribution(r[field] for r in rows).items():
                        if stat != "n_defined":
                            item[f"{name}_{stat}"] = value
                summary.append(item)
    assert len(summary) == len(groups) == len(old) == 72
    csv_path = HERE / f"{STEM}.csv"
    write_csv(csv_path, summary)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
    fig.subplots_adjust(left=.065, right=.99, bottom=.25, top=.86, wspace=.34)
    cells = {(r["scheme"], r["layer_index"], r["sigma"]): r for r in summary}
    for ax, metric, sigma, title, ylabel in zip(
        axes, ("ep_l2", "ep_rms", "ep_over_clean_ep"), (0, 0, 5e-4),
        ("Clean gradient norm", "Clean gradient per weight", r"Noise effect ($\sigma=5\times10^{-4}$)"),
        (r"Gradient L2 norm $\|g\|_2$", r"Gradient RMS $\|g\|_2/\sqrt{N}$",
         r"Noisy / clean norm $\|g_\sigma\|_2/\|g_0\|_2$"), strict=True,
    ):
        for offset, scheme in zip((-.15, 0, .15), SCHEMES, strict=True):
            rows = [cells[scheme, i, sigma] for i in range(4)]
            x = np.arange(4) + offset
            mean = np.array([r[f"{metric}_mean"] for r in rows])
            lo, hi = (np.array([r[f"{metric}_{s}"] for r in rows]) for s in ("p10", "p90"))
            ax.vlines(x, lo, hi, colors=COLORS[scheme], alpha=.55, linewidth=1.7)
            ax.plot(x, mean, color=COLORS[scheme], marker=MARKERS[scheme], markersize=5,
                    linewidth=1.25, label=LABELS[scheme])
        if sigma:
            ax.axhline(1, color=".55", linewidth=.8, linestyle="--")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(4), LAYERS)
        ax.set_xlim(-.35, 3.35)
        ax.grid(axis="y", alpha=.2, linewidth=.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               bbox_to_anchor=(.5, .065), frameon=False)
    fig.text(.5, .015, "Conv3 · Mean with 10th–90th percentile range across minibatches / noise draws",
             ha="center", fontsize=9, color=".3")
    save_all(fig, HERE / "figures" / STEM)

    provenance = {
        "source_run": str(RUN.resolve()), "new_replay_or_optimizer_steps": 0,
        "canonical_run_validation_errors": errors, "raw_conv3_rows": sum(map(len, groups.values())),
        "summary_cells": len(summary), "parameter_shapes": shapes["baseline"],
        "checkpoints": sources,
        "cohort_sha256": sha256(RUN / "cohort.json"),
        "sources": {str(p.resolve()): sha256(p) for p in (
            RUN / "gradients.csv", RUN / "result.json", RUN / "config.used.json",
            HERE / "section43_gradient_cosine_summary.csv", HERE / "section43_checkpoint_sources.csv")},
        "statistic": "Mean, median, p10, p90 and sample std of individual batch/draw norms; RMS = L2/sqrt(parameter_count).",
        "ratio_statistic": "Aggregate of individual paired noisy-EP / clean-EP norm ratios, not ratio of aggregate means.",
        "spread": "Within-checkpoint minibatch/draw variation, not training-seed uncertainty.",
        "clean_reference": "Noiseless finite-K BPTT and centered frozen-current EqProp; T=K=8, batch size 16, float64.",
        "limitations": "One seed; different weights, betas and operating points; raw gradients are not Adam updates. Baseline post-T residual caveat remains; see section43_mechanism.md.",
        "generator_sha256": sha256(Path(__file__)), "matplotlib_version": matplotlib.__version__,
        "artifacts": {str(p.relative_to(HERE)): sha256(p) for p in (
            csv_path, *sorted((HERE / "figures").glob(STEM + ".*")))},
    }
    (HERE / f"{STEM}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"csv": str(csv_path), "figure": str(HERE / "figures" / f"{STEM}.jpg"),
                      "raw_rows": provenance["raw_conv3_rows"], "summary_cells": len(summary)}, indent=2))
    for layer, label in enumerate(LAYERS):
        print(label, {LABELS[s]: {k: cells[s, layer, 0][k] for k in ("ep_l2_mean", "ep_rms_mean")}
                      for s in SCHEMES})
    print("Noise/clean ratios", {LABELS[s]: [cells[s, i, 5e-4]["ep_over_clean_ep_mean"]
                                            for i in range(4)] for s in SCHEMES})


if __name__ == "__main__":
    main()
