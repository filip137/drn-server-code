#!/usr/bin/env python3
"""Compare clean Conv3 initialization/trained norms for all amplification schemes."""
from collections import defaultdict
import json
from pathlib import Path
import sys

from plot_section43_mechanism import (
    COLORS, LABELS, MARKERS, SCHEMES, distribution, read_csv, save_all, write_csv,
)
from plot_section43_conv3_gradient_magnitudes import sha256, PARAMETERS, LAYERS
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STEM = "section43_conv3_initialization_magnitudes"
STUDY = HERE.parent/"results/section43-conv3-initialization-magnitudes-20260918-v1"


def main():
    sys.path.insert(0, str(HERE.parent))
    from experiments.reporting import validate_run
    config = json.loads((STUDY/"full/config.used.json").read_text())
    baseline = Path(config["reuse_baseline_run"])
    trained = Path(config["trained_reference"]["run"])
    for run in (STUDY/"smoke", STUDY/"full", baseline, trained):
        assert not validate_run(run), run
    new = read_csv(STUDY/"full/gradients.csv")
    smoke = read_csv(STUDY/"smoke/gradients.csv")
    matched = [r for r in new if int(r["batch_index"]) == 0]
    assert [r for r in smoke if r["scheme"] != "baseline"] == matched
    baseline_clean = [r for r in read_csv(baseline/"gradients.csv") if float(r["sigma"]) == 0]
    assert [r for r in smoke if r["scheme"] == "baseline"] == [r for r in baseline_clean if int(r["batch_index"]) == 0]
    trained_clean = [r for r in read_csv(trained/"gradients.csv")
                     if r["architecture"] == "conv3" and float(r["sigma"]) == 0]
    assert len(new) == 288 and len(baseline_clean) == 144 and len(trained_clean) == 432
    counts = {r["parameter"]: int(r["element_count"])
              for r in read_csv(STUDY/"full/checkpoint_gradients.csv")}
    near_zero = {(r["scheme"], int(r["epoch"]), int(r["batch_index"]), r["estimator"], r["parameter"]):
                 float(r["abs_le_1e_minus12_fraction"])
                 for r in read_csv(STUDY/"full/checkpoint_gradients.csv")}
    for r in read_csv(baseline/"checkpoint_gradients.csv"):
        near_zero["baseline", int(r["epoch"]), int(r["batch_index"]), r["estimator"], r["parameter"]] = float(r["abs_le_1e_minus12_fraction"])
    epochs = {r["scheme"]: r["best_epoch"] for r in config["cases"]}
    groups = defaultdict(list)
    initial = {(r["scheme"], r["parameter"], int(r["batch_index"])): r for r in [*baseline_clean, *new]}
    for epoch_role, data in (("initialization", [*baseline_clean, *new]), ("trained", trained_clean)):
        for r in data:
            scheme, name, batch = r["scheme"], r["parameter"], int(r["batch_index"])
            epoch = 0 if epoch_role == "initialization" else epochs[scheme]
            for estimator, field, zero_field in (
                ("eqprop", "bptt_estimate_l2", "bptt_estimate_zero_fraction"),
                ("bptt", "bptt_reference_l2", "bptt_reference_zero_fraction"),
            ):
                norm, reference = float(r[field]), float(initial[scheme, name, batch][field])
                groups[scheme, epoch_role, epoch, name, estimator].append({
                    "batch_index": batch, "gradient_l2": norm, "gradient_rms": norm/np.sqrt(counts[name]),
                    "norm_relative_to_initial": norm/reference,
                    "exact_zero_fraction": float(r[zero_field]),
                    "abs_le_1e_minus12_fraction": near_zero.get((scheme, epoch, batch, estimator, name)),
                })
    summary = []
    for (scheme, role, epoch, name, estimator), rows in groups.items():
        assert len(rows) == 36 and {r["batch_index"] for r in rows} == set(range(36))
        item = {"scheme": scheme, "checkpoint_role": role, "epoch": epoch, "parameter": name,
                "layer_index": PARAMETERS.index(name), "estimator": estimator,
                "element_count": counts[name], "n_batches": len(rows)}
        for metric in ("gradient_l2", "gradient_rms", "norm_relative_to_initial",
                       "exact_zero_fraction", "abs_le_1e_minus12_fraction"):
            item.update({f"{metric}_{stat}": val for stat, val in distribution(r[metric] for r in rows).items()})
        summary.append(item)
    assert len(summary) == 48
    csv_path = HERE/f"{STEM}.csv"
    write_csv(csv_path, summary)
    cells = {(r["scheme"], r["checkpoint_role"], r["parameter"], r["estimator"]): r for r in summary}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3))
    fig.subplots_adjust(left=.075, right=.99, bottom=.25, top=.86, wspace=.35)
    limits = []
    for ax, role, metric, title, ylabel in zip(axes,
        ("initialization", "trained", "trained"),
        ("gradient_rms", "gradient_rms", "norm_relative_to_initial"),
        ("Initialization", "After training", "Change during training"),
        (r"Gradient RMS $\|g\|_2/\sqrt{N}$", r"Gradient RMS $\|g\|_2/\sqrt{N}$",
         r"Trained / initial norm $\|g_{\rm trained}\|_2/\|g_0\|_2$"), strict=True):
        for offset, scheme in zip((-.15, 0, .15), SCHEMES, strict=True):
            rows = [cells[scheme, role, p, "eqprop"] for p in PARAMETERS]
            x = np.arange(4) + offset
            mean = [r[f"{metric}_mean"] for r in rows]
            low, high = ([r[f"{metric}_{s}"] for r in rows] for s in ("p10", "p90"))
            ax.vlines(x, low, high, colors=COLORS[scheme], alpha=.5, linewidth=1.7)
            ax.plot(x, mean, color=COLORS[scheme], marker=MARKERS[scheme], markersize=5,
                    linewidth=1.3, label=LABELS[scheme])
            if metric == "gradient_rms":
                limits.extend(low+high)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(4), LAYERS)
        ax.set_xlim(-.35, 3.35)
        ax.grid(axis="y", alpha=.2, linewidth=.5)
    for ax in axes[:2]:
        ax.set_ylim(min(limits)/1.7, max(limits)*1.7)
    axes[2].axhline(1, color=".5", linestyle="--", linewidth=.8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               bbox_to_anchor=(.5, .065), frameon=False)
    fig.text(.5, .015, "Conv3 · Clean EqProp · Mean with 10th–90th percentile range across 36 matched minibatches",
             ha="center", fontsize=9, color=".3")
    save_all(fig, HERE/"figures"/STEM)

    guards = [json.loads(line) for line in (STUDY/"full/batch_guards.jsonl").read_text().splitlines()]
    assert len(guards) == 72
    guard_keys = ("parameter_tensors_unchanged_after_cast", "iteration_contract_passed",
                  "zero_noise_readout_matches_clean_bitwise", "checkpoint_unchanged_after_noise")
    assert all(all(g[k] for k in guard_keys) for g in guards)
    residuals = read_csv(STUDY/"full/residuals.csv")
    provenance = {"new_run": str(STUDY/"full"), "baseline_run": str(baseline), "trained_run": str(trained),
        "initializer": config["initializer"], "canonical_runs_validated": 4,
        "baseline_smoke_comparisons_identical": 4, "new_smoke_full_comparisons_identical": 8,
        "initialization_batches_total": 108, "new_initialization_batches": 72,
        "summary_cells": len(summary), "guard_flags_passed": guard_keys,
        "new_gpu_seconds_including_smoke": sum(json.loads(p.read_text())["gpu_seconds"] for p in STUDY.glob("*/usage.json")),
        "official_test_read": False, "optimizer_steps": 0, "matplotlib_version": matplotlib.__version__,
        "statistic": "Mean of individual batch gradient magnitudes; RMS=L2/sqrt(N). Ratios are computed within each paired batch, then summarized.",
        "near_zero": "Measured for all initialization cases and trained baseline; trained balanced/legacy unavailable from archived norms.",
        "spread": "p10/p90 across 36 minibatches; one training seed, not a seed confidence interval.",
        "sources": {str(p): sha256(p) for p in (
            STUDY/"full/gradients.csv", STUDY/"full/checkpoint_gradients.csv", STUDY/"full/config.used.json",
            baseline/"gradients.csv", baseline/"checkpoint_gradients.csv", trained/"gradients.csv",
            STUDY/"full/cohort.json", Path(__file__))},
        "artifacts": {str(p.relative_to(HERE)): sha256(p) for p in (
            csv_path, *sorted((HERE/"figures").glob(STEM+".*")))}}
    (HERE/f"{STEM}_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    for role in ("initialization", "trained"):
        print(role)
        for p in PARAMETERS:
            print(p, {LABELS[s]: {k: cells[s, role, p, "eqprop"][k] for k in (
                "gradient_l2_mean", "gradient_rms_mean", "norm_relative_to_initial_mean")}
                for s in SCHEMES})
    print(json.dumps({k: provenance[k] for k in ("canonical_runs_validated", "summary_cells", "new_gpu_seconds_including_smoke")}, indent=2))
    print("Residual columns:", list(residuals[0]))


if __name__ == "__main__":
    main()
