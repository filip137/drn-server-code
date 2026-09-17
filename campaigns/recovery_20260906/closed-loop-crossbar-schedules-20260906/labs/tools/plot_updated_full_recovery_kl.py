"""Update the recovery overview with explicitly sourced HWA noise-sweep results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
RECOVERY = ROOT / "analysis/notion_single_graph_corrected_20260907/provenance.json"
ORIGINAL = ROOT / "analysis/digital_hwa_open_loop_20260908/provenance.json"
SWEEP = WORK / "hwa-noise-sweep-20260908/runs"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", choices=("best-observed", "validation-selected"), default="best-observed")
    parser.add_argument("--drn-only", action="store_true", help="Export the three-stage DRN presentation figure.")
    args = parser.parse_args()
    out = ROOT / "analysis/full_recovery_updated_20260908" / args.selection
    if args.drn_only:
        out = out / "drn"
    out.mkdir(parents=True, exist_ok=True)
    old, original = read(RECOVERY), read(ORIGINAL)
    records, pools = [], []
    source_paths = [RECOVERY, ORIGINAL, Path(__file__)]
    architectures = (("drn", "DRN"),) if args.drn_only else (("crossbar", "Crossbar"), ("drn", "DRN"))
    recovery_stages = (("open_loop", 1),) if args.drn_only else (("open_loop", 1), ("closed_loop", 2))
    for arch, prefix in architectures:
        sweep_path = SWEEP / arch / "result.json"
        sweep = read(sweep_path)
        assert sweep["status"] == "complete" and not sweep["smoke"]
        assert not sweep["test_used_for_selection"]
        source_paths.append(sweep_path)
        healthy = next(s["values"] for s in old["series"] if s["label"] == f"{prefix} · Healthy")
        corrupt = next(s["values"] for s in old["series"] if s["label"] == f"{prefix} · Corrupt")
        digital = next(r["test_KL"] for r in original["records"] if r["architecture"] == arch and r["stage"] == "HWA digital weights")
        original_epoch = 10 if arch == "crossbar" else 9
        for stage, condition, old_value, metric in (
            ("clean_hwa", "shared", digital, "clean_test"),
            ("pv", "healthy", healthy[0], "healthy_pv_test"),
            ("pv", "corrupt", corrupt[0], "faulted_pv_test"),
        ):
            candidates = [dict(architecture=arch, stage=stage, condition=condition,
                test_KL=old_value, examples=10000, source_family="original_HWA",
                training_noise_multiplier=1.0, selected_epoch=original_epoch,
                source=str(ORIGINAL if stage == "clean_hwa" else RECOVERY),
                source_label="Original HWA")]
            for case in sweep["results"]:
                evaluation = case[metric] if stage == "clean_hwa" else case[metric]["apparent"]
                assert evaluation["examples"] == 10000
                multiplier = case["multiplier"]
                token = format(multiplier, "g").replace(".", "p")
                candidates.append(dict(architecture=arch, stage=stage, condition=condition,
                    test_KL=evaluation["kl_teacher_student"], examples=10000,
                    source_family="HWA_noise_sweep", training_noise_multiplier=multiplier,
                    selected_epoch=case["selected_epoch"], source=str(sweep_path),
                    checkpoint=str(SWEEP / arch / f"noise_{token}" / "selected.pt"),
                    sweep_validation_choice=case["selected_by_validation"],
                    source_label=f"Noise ×{multiplier:g}"))
            if args.selection == "best-observed":
                selected = min(candidates, key=lambda r: r["test_KL"])
            else:
                selected = next(r for r in candidates if r.get("sweep_validation_choice"))
            records.append(dict(selected))
            pools.append(dict(architecture=arch, stage=stage, condition=condition,
                candidates=candidates, displayed=selected))
        for stage, index in recovery_stages:
            for condition, values in (("healthy", healthy), ("corrupt", corrupt)):
                records.append(dict(architecture=arch, stage=stage, condition=condition,
                    test_KL=values[index], examples=10000, source_family="original_HWA_recovery",
                    training_noise_multiplier=1.0, selected_epoch=None, source=str(RECOVERY),
                    source_label="Original HWA deployment"))
    assert len(records) == (5 if args.drn_only else 14)
    colors = {"shared": "#738291", "healthy": "#26799D", "corrupt": "#CE6432"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20,
        "pdf.fonttype": 42, "svg.fonttype": "none"})
    legend = [Patch(facecolor=colors[k], label=label) for k, label in (
        ("shared", "After HWA"), ("healthy", "Healthy devices"), ("corrupt", "Corrupt devices"))]
    upper = max(0.08, np.ceil(1.3 * max(r["test_KL"] for r in records) / .02) * .02)
    outputs = []
    figures = ((False, "drn_recovery_kl", "drn_recovery_kl"),) if args.drn_only else (
        (True, "full_recovery_kl", "recovery_kl_with_hwa"),
        (False, "digital_hwa_pv_open_loop", "recovery_digital_hwa_open_loop"))
    for include_closed, stem, friendly in figures:
        stages = ["clean_hwa", "pv", "open_loop"] + (["closed_loop"] if include_closed else [])
        labels = ["After HWA", "After HWA and P&V",
                  "After HWA, P&V and\non-chip open-loop training"]
        if include_closed:
            labels[-1] = "After HWA, P&V and\non-chip open-loop\ntraining"
            labels.append("After HWA, P&V and\non-chip closed-loop\ntraining")
        centers = np.arange(len(stages)) * 2.2
        fig, axes = plt.subplots(len(architectures), 1, figsize=(16, 9), squeeze=False)
        fig.subplots_adjust(left=.105, right=.985, top=.835,
                            bottom=.185 if include_closed else .145,
                            hspace=1.0 if include_closed else .70)
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.55, .995),
            ncol=3, fontsize=22, frameon=False, handlelength=1.7, columnspacing=2.0)
        for axis, (arch, _) in zip(axes.flat, architectures):
            for row in records:
                if row["architecture"] != arch or row["stage"] not in stages:
                    continue
                stage_index = stages.index(row["stage"])
                separation = .4 if include_closed else .33
                offset = {"shared": 0, "healthy": -separation, "corrupt": separation}[row["condition"]]
                x = centers[stage_index] + offset
                bars = axis.bar(x, row["test_KL"], width=.78 if row["condition"] == "shared" else (.7 if include_closed else .58),
                    color=colors[row["condition"]], edgecolor="white", linewidth=.6, zorder=3)
                axis.bar_label(bars, labels=[f"{row['test_KL']:.6f}"], padding=7, fontsize=18)
            axis.set_title("Crossbar arrays" if arch == "crossbar" else "DRN", loc="left", fontsize=28, weight="bold", pad=15)
            axis.set_ylabel("KL divergence", fontsize=22, labelpad=17)
            axis.set_xticks(centers, labels, fontsize=22)
            axis.set_xlim(-.9, centers[-1]+.9)
            axis.set_ylim(0, upper)
            axis.set_yticks(np.arange(0, upper+.001, .02))
            axis.grid(axis="y", color="#e5e8ec")
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="x", length=0, pad=12)
            axis.tick_params(axis="y", labelsize=20)
        if args.selection == "best-observed":
            note = "Best observed test KL per stage across the compared checkpoints; stages summarize independent experiments."
        else:
            settings = "DRN ×0.5" if args.drn_only else "crossbar ×0.25, DRN ×0.5"
            note = f"HWA/P&V use validation-selected noise settings: {settings}. Recovery uses the original HWA deployment."
        for ext in ("png", "pdf", "svg"):
            path = out / f"{stem}.{ext}"
            fig.savefig(path, dpi=220, facecolor="white")
            shutil.copy2(path, WORK / f"{friendly}.{ext}")
            if not include_closed:
                shutil.copy2(path, WORK / f"recovery_kl_updated.{ext}")
            outputs.append(path)
        plt.close(fig)
    fields = ["architecture", "stage", "condition", "test_KL", "examples", "source_family",
              "training_noise_multiplier", "selected_epoch", "source", "source_label"]
    with (out / "kl_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    provenance = dict(status="complete", selection=args.selection, records=records, comparison_pools=pools,
        architectures=[arch for arch, _ in architectures],
        posthoc_test_minimum_display=args.selection == "best-observed", original_study_selections_modified=False,
        matched_single_deployment_trajectory=False, new_recovery_training_performed=False,
        recovery_values_preserved=2 if args.drn_only else 8,
        clean_hwa_state="No SET/RESET variation or write noise; nominal DRN mapping retained",
        presentation=dict(primary_stages=["After HWA", "After HWA and P&V",
            "After HWA, P&V and on-chip open-loop training"],
            size_inches=[16, 9], footnotes_in_figure=False, source_labels_in_figure=False,
            font_sizes=dict(panel_title=28, legend=22, stage_labels=22, axis_label=22, ticks=20, values=18)),
        source_sha256={str(p): digest(p) for p in source_paths},
        figure_sha256={p.name: digest(p) for p in outputs})
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    title = "# DRN recovery KL comparison" if args.drn_only else "# Updated full KL comparison"
    lines = [title, "", note, "",
        "The comparison pool consists of the original reported HWA/P&V checkpoints and "
        "the four noise-sweep checkpoints per architecture, each selected by its declared validation rule. "
        "Recovery retains the original validation-selected runs. "
        "This is an exploratory comparison of independent experiments; it does not show a single "
        "model passing through every stage. No recovery from the new HWA weights was run.", "",
        "| Architecture | Stage | Condition | Test KL | Source |", "|---|---|---|---:|---|"]
    for row in records:
        lines.append(f"| {row['architecture']} | {row['stage']} | {row['condition']} | {row['test_KL']:.6f} | {row['source_label']} |")
    lines += ["", "Clean HWA uses the frozen digital weights without SET/RESET variation or write noise. "
        "The DRN retains its nominal RESET=0.1 and SET=2.0 mapping, original solver and fixed gain. "
        "P&V and recovery use the held apparent device state. Teachers, splits and calibration remain "
        "the original ones within each architecture, with different teachers/device models between architectures.", "",
        "The best-observed mode is a post-hoc minimum over reported test readouts. It does not replace "
        "the frozen validation-selection receipts or establish an unbiased selected-model estimate. "
        "Source labels and explanatory notes are retained here and in the CSV/provenance, and omitted "
        "from the presentation figures at the user's request. The primary presentation ends after "
        "open-loop training. "
        "Original archived figures and all experiment artifacts are retained.", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(json.dumps({"selection": args.selection, "presentation_figure": str(WORK / "recovery_kl_updated.png"),
        "architectures": provenance["architectures"], "figures": [str(p) for p in outputs],
        "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
