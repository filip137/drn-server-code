"""Regenerate the Notion recovery figure from completed, paired test readouts.

Both frozen schedule families are displayed. No test result selects a schedule
or checkpoint. Development-array results are excluded from this figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "analysis/paired_test_readouts.json")
    parser.add_argument("--output", type=Path, default=root / "analysis/notion_figure")
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    assert source["status"] == "complete"
    assert source["selection_used_test"] is False
    rows = [r for r in source["rows"] if r["cohort"] == "additional_array"]
    methods = ("p0", "constant", "exponential")
    expected_replicas = {
        "crossbar": {"array-2090501", "array-2090502"},
        "drn": {"array-2-write-1", "array-3-write-1"},
    }
    records = []
    grouped = {}
    for architecture in ("crossbar", "drn"):
        for condition in ("healthy", "faulted"):
            for method in methods:
                selected = sorted(
                    [r for r in rows if (r["architecture"], r["condition"], r["method"])
                     == (architecture, condition, method)], key=lambda r: r["replica"])
                assert len(selected) == 2
                assert {r["replica"] for r in selected} == expected_replicas[architecture]
                values = [r["test_kl"] for r in selected]
                record = dict(
                    architecture=architecture, condition=condition, method=method,
                    cohort="additional_array", split="test", state="held_apparent",
                    arrays=2, writes_per_array=1, examples=10000,
                    mean=statistics.mean(values), minimum=min(values), maximum=max(values),
                    replicas=";".join(r["replica"] for r in selected),
                    selected_epochs=";".join(str(r["selected_epoch"]) for r in selected),
                    initial_learning_rates=";".join(str(r["initial_learning_rate"]) for r in selected),
                    readouts=";".join(r["readout"] for r in selected),
                )
                grouped[architecture, condition, method] = record
                records.append(record)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    colors = {"healthy": "#26799D", "faulted": "#CE6432"}
    labels = {"healthy": "Without corrupt devices", "faulted": "With corrupt devices"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.35), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.245, top=0.70, wspace=0.17)
    fig.suptitle("KL divergence before and after on-chip training", x=0.075, y=0.965,
                 ha="left", fontsize=20, weight="bold")
    fig.text(0.075, 0.89, "Lower learning rates and selected checkpoints from 30-epoch recovery runs.", fontsize=12)
    fig.legend(handles=[Patch(facecolor=colors[c], label=labels[c]) for c in colors],
               loc="upper left", bbox_to_anchor=(0.065, 0.845), ncol=2, frameon=False,
               fontsize=12)

    for ax, architecture, title in zip(axes, ("crossbar", "drn"), ("a) Crossbar", "b) DRN")):
        for offset, condition in ((-0.185, "healthy"), (0.185, "faulted")):
            values = [grouped[architecture, condition, method] for method in methods]
            xs = [i + offset for i in range(len(methods))]
            means = [r["mean"] for r in values]
            ax.bar(xs, means, width=0.31, color=colors[condition], zorder=3)
            ax.errorbar(xs, means,
                        yerr=[[r["mean"] - r["minimum"] for r in values],
                              [r["maximum"] - r["mean"] for r in values]],
                        fmt="none", color="#333333", elinewidth=1.15,
                        capsize=3.5, zorder=4)
            for x, value in zip(xs, values):
                ax.annotate(f"{value['mean']:.5f}", (x, value["maximum"]),
                            xytext=(0, 6), textcoords="offset points", ha="center",
                            fontsize=10, color="#202020")
        ax.set_title(title, loc="left", fontsize=17, weight="bold", pad=18)
        ax.set_xticks([0, 1, 2], ["HWA + P&V\nBefore recovery", "Recovery\nConstant rate",
                                 "Recovery\nExponential decay"])
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(0, 0.06)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E5E8EC")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Test KL(teacher || student)", labelpad=8)
    fig.text(0.075, 0.112,
             "Held-apparent test KL on 10,000 MNIST images. Bars: mean; whiskers: range across two additional arrays per architecture.",
             fontsize=10, color="#444444")
    fig.text(0.075, 0.066,
             "Rates fixed from development runs; checkpoints selected by validation KL. Recovery starts from the corresponding HWA + P&V state.",
             fontsize=10, color="#444444")

    args.output.mkdir(parents=True, exist_ok=True)
    figure_files = []
    for extension in ("png", "svg", "pdf"):
        output = args.output / f"kl_before_after_onchip_training_improved.{extension}"
        fig.savefig(output, dpi=220, facecolor="white")
        figure_files.append(output)
    plt.close(fig)
    with (args.output / "kl_plot_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    reductions = {}
    for architecture in ("crossbar", "drn"):
        baseline = grouped[architecture, "faulted", "p0"]["mean"]
        reductions[architecture] = {
            method: 100 * (1 - grouped[architecture, "faulted", method]["mean"] / baseline)
            for method in ("constant", "exponential")
        }
    provenance = {
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": records,
        "faulted_reduction_percent_from_matched_p0": reductions,
        "schedule_display_policy": "Both frozen schedule families, without test-based selection between them.",
        "checkpoint_policy": "Saved validation-selected checkpoints; no reselection.",
        "old_figure_cohort": "Crossbar: one development-array validation result; DRN: three arrays x three writes, test.",
        "updated_figure_cohort": "Both architectures: common official MNIST test set, two additional arrays x one saved write.",
        "error_bars": "Full minimum-to-maximum range across two arrays; not confidence intervals.",
        "limitation": "Different teachers and device models; absolute KL is not a matched architecture ranking.",
        "figure_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in figure_files},
    }
    (args.output / "kl_plot_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "bars": len(records),
                      "faulted_reduction_percent": reductions}, indent=2))


if __name__ == "__main__":
    main()
