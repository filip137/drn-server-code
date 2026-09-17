"""Plot validation-selected HWA noise ablations using Matplotlib."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

WORK = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=WORK / "hwa-noise-sweep-20260908")
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    runs = args.runs or args.study / "runs"
    output = args.output or args.study / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    records = {a: json.loads((runs / a / "result.json").read_text()) for a in ("crossbar", "drn")}
    rows = []
    for arch, result in records.items():
        assert result["status"] == "complete" and result["smoke"] == args.smoke
        assert len(result["results"]) == 4 and not result["test_used_for_selection"]
        assert sorted(r["multiplier"] for r in result["results"]) == [0, 0.25, 0.5, 1]
        selection = json.loads((runs / arch / "selection.json").read_text())
        assert not selection["test_used"]
        selected = min(result["results"], key=lambda r: (r["selected_validation"]["kl_teacher_student"],
            -r["selected_validation"]["student_accuracy"], r["multiplier"]))
        assert selected["multiplier"] == result["selected_multiplier"]
        for case in sorted(result["results"], key=lambda r: r["multiplier"]):
            row = dict(architecture=arch, noise_multiplier=case["multiplier"], selected_epoch=case["selected_epoch"],
                selected_multiplier=case["selected_by_validation"], validation_KL=case["selected_validation"]["kl_teacher_student"])
            metrics = {"clean": case["clean_test"], "healthy_pv": case["healthy_pv_test"]["apparent"],
                "faulted_pv": case["faulted_pv_test"]["apparent"],
                "healthy_pv_persistent_diagnostic": case["healthy_pv_test"]["persistent_secondary_diagnostic"],
                "faulted_pv_persistent_diagnostic": case["faulted_pv_test"]["persistent_secondary_diagnostic"]}
            for name, metric in metrics.items():
                assert metric["examples"] == (64 if args.smoke else 10000)
                row[f"{name}_accuracy_percent"] = metric["student_accuracy"] * 100
                row[f"{name}_KL"] = metric["kl_teacher_student"]
            rows.append(row)
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
        "axes.titlesize": 20, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "legend.fontsize": 15, "svg.fonttype": "none", "pdf.fonttype": 42})
    stages = (("clean", "Clean HWA weights", "#34383e", "o"),
        ("healthy_pv", "P&V healthy devices", "#2678b2", "s"),
        ("faulted_pv", "P&V corrupt devices", "#dd8235", "^"))
    for metric, ylabel, filename in (("accuracy_percent", "Test accuracy (%)", "hwa_noise_sweep_accuracy"),
                                    ("KL", "Test KL divergence", "hwa_noise_sweep_kl")):
        fig, axes = plt.subplots(2, 1, figsize=(10.6, 10.4), sharex=True)
        if args.smoke:
            fig.text(0.5, 0.995, "SMOKE CHECK · 64 images · not experiment results", ha="center", va="top", fontsize=12, color="#a43b32")
        for axis, arch in zip(axes, records):
            data = [r for r in rows if r["architecture"] == arch]
            xs = [r["noise_multiplier"] for r in data]
            for prefix, label, color, marker in stages:
                values = [r[f"{prefix}_{metric}"] for r in data]
                axis.plot(xs, values, label=label, color=color, marker=marker, markersize=8, linewidth=2.2)
                if prefix == "clean":
                    for x, y in zip(xs, values):
                        offset = (-23 if metric == "accuracy_percent" and y >= 99.9 else
                                  -18 if metric == "KL" and x == 1 else 11)
                        axis.annotate(f"{y:.2f}%" if metric == "accuracy_percent" else f"{y:.4f}", (x, y),
                            xytext=(0, offset), textcoords="offset points", ha="center", fontsize=12, color=color)
            axis.set_title("Crossbar arrays" if arch == "crossbar" else "DRN", pad=15)
            axis.set_ylabel(ylabel)
            axis.set_xticks([0, 0.25, 0.5, 1], ["0", "0.25", "0.5", "1"])
            axis.set_xlim(-0.055, 1.055)
            axis.grid(axis="y", color="#d9dce0", linewidth=0.8)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            lo, hi = axis.get_ylim()
            pad = max(0.15 if metric == "accuracy_percent" else 0.002, (hi-lo)*0.16)
            axis.set_ylim(lo if metric == "accuracy_percent" else 0, min(100, hi+pad) if metric == "accuracy_percent" else hi+pad)
        axes[-1].set_xlabel("HWA training-noise multiplier", labelpad=10)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False)
        fig.subplots_adjust(left=0.13, right=0.975, top=0.94, bottom=0.17, hspace=0.30)
        for extension in ("png", "pdf", "svg"):
            path = output / f"{filename}.{extension}"
            fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
            if not args.smoke:
                shutil.copyfile(path, WORK / path.name)
        plt.close(fig)
    lines = ["# HWA noise sweep (exploratory)", "",
        "All metrics use the validation-selected checkpoint for that noise multiplier. "
        "Clean HWA evaluation disables device endpoint variation and write noise. "
        "DRN retains its nominal 0.1/2.0 conductance mapping and original solver; "
        "crossbar uses the unprojected digital master. Deployment uses unchanged hardware effects.", "",
        "| Architecture | Training noise | Epoch | Clean accuracy | Healthy P&V accuracy | Corrupt P&V accuracy | Clean KL | Healthy P&V KL | Corrupt P&V KL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['architecture']} | {row['noise_multiplier']:g} | {row['selected_epoch']} | "
            f"{row['clean_accuracy_percent']:.2f}% | {row['healthy_pv_accuracy_percent']:.2f}% | "
            f"{row['faulted_pv_accuracy_percent']:.2f}% | {row['clean_KL']:.6f} | "
            f"{row['healthy_pv_KL']:.6f} | {row['faulted_pv_KL']:.6f} |")
    lines += ["", "Noise multiplier selected by healthy post-P&V validation KL:"]
    for arch, result in records.items():
        lines.append(f"- {arch}: {result['selected_multiplier']:g}.")
    lines += ["", "No test-based checkpoint or noise selection. Four models per architecture, "
        "ten full epochs each; one training seed, one development and one separate final deployment "
        "realization per architecture. The crossbar development population also supplies its training "
        "support bounds. Test data have been inspected previously. Different teachers and physical "
        "models mean this is not an isolated architecture comparison.", "",
        "The noise-free training arm still has nominal conductance/range constraints. "
        "It is not a proven digital optimum. The held apparent programming state is the primary "
        "readout; persistent-only diagnostics are in metrics.csv and result JSON. No on-chip "
        "recovery was retrained in this sweep. The old recovery results are unchanged.", ""]
    (output / "report.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "cases": len(rows), "selected": {a:r["selected_multiplier"] for a,r in records.items()}}))


if __name__ == "__main__":
    main()
