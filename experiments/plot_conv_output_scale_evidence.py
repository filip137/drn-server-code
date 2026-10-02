"""Plot existing voltage/accuracy evidence and the analytic output-scale loss.

This reads collected summaries only; it evaluates no dataset or model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-conv-output-scale")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STATE_SOURCE = ROOT / "results/conv3-relative-endpoint-noise-20260925-v1/analysis/physical_state_controls.csv"
ACCURACY_SOURCE = ROOT / "paper_ready_results/current_contract_validation_table_summary.csv"
COLORS = {"baseline": "#576578", "ours": "#3478b8", "legacy": "#cf672c", "normalized": "#16857e"}


def source_record(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def collect() -> dict:
    # Target 2 keeps all schemes on the same GPU/runtime. Only free-state RMS
    # is used; the nudging target does not define another free-state replicate.
    state_rows = [r for r in read_rows(STATE_SOURCE) if float(r["relative_target"]) == 2.0]
    state_lookup = {(r["scheme"], int(r["state_layer_index"])): r for r in state_rows}
    scaling = []
    for index in range(4):
        baseline = float(state_lookup[("baseline", index)]["free_rms"])
        legacy = float(state_lookup[("legacy", index)]["free_rms"])
        assert math.isfinite(baseline) and math.isfinite(legacy) and baseline > 0
        expected = 4 ** index
        scaling.append({"layer": index, "baseline_rms": baseline, "legacy_rms": legacy,
                        "ratio": legacy / baseline, "expected_ratio": expected,
                        "normalized_ratio": legacy / baseline / expected})

    sources = [source_record(STATE_SOURCE), source_record(ACCURACY_SOURCE)]
    accuracy = []
    rows = [r for r in read_rows(ACCURACY_SOURCE) if r["block"] == "T1_BPTT"]
    assert len(rows) == 9
    for row in rows:
        assert row["collected_count"] == "3" and row["collected_seeds"] == "0,1,2"
        assert float(row["G_max"]) == 100
        values = []
        for seed in range(3):
            path = (ROOT / "paper_ready_results/bundles/table1_wide_bptt"
                    / row["architecture"] / row["scheme"] / f"seed{seed}" / "result.json")
            result = json.loads(path.read_text())
            value = 100 * result["terminal_metrics"]["validation"]["best_accuracy"]
            assert math.isclose(value, float(row[f"seed{seed}_best_validation_percent"]), abs_tol=1e-10)
            values.append(value)
            sources.append(source_record(path))
        mean, sd = statistics.mean(values), statistics.stdev(values)
        assert math.isclose(mean, float(row["mean_best_validation_percent"]), abs_tol=1e-10)
        assert math.isclose(sd, float(row["sample_sd_best_validation_pp"]), abs_tol=1e-10)
        accuracy.append({"architecture": row["architecture"], "scheme": row["scheme"],
                         "seed_values": values, "mean": mean, "sample_sd": sd,
                         "T": int(row["T"]), "K": int(row["K"])})
    return {"created_at": datetime.now(timezone.utc).isoformat(),
            "kind": "existing-summary visualization and analytic illustration",
            "new_training_or_dataset_evaluation": False,
            "sources": sources, "script": source_record(Path(__file__)),
            "scaling": scaling, "accuracy": accuracy,
            "limitations": ["RMS agreement alone does not establish elementwise state or gradient identity.",
                            "Panel B is an analytic single-score loss slice, not measured training.",
                            "Historical accuracy arms use scheme-specific learning rates.",
                            "Best validation accuracy uses the 5000-example selection split, not official test.",
                            "Source metrics and aggregate arithmetic are checked; underlying simulations are not revalidated."]}


def plot(data: dict, destination: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#263444", "text.color": "#263444",
                         "xtick.color": "#263444", "ytick.color": "#263444",
                         "axes.edgecolor": "#bac2cc", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))
    fig.subplots_adjust(left=.055, right=.985, bottom=.24, top=.75, wspace=.34)
    fig.suptitle("A rescaled circuit can have a different training objective", x=.055, y=.955,
                 ha="left", fontsize=20, fontweight="bold")
    fig.text(.055, .897, "Existing MNIST measurements and an analytic illustration — the causal training control remains to be run.",
             fontsize=11, color="#647184")

    ax = axes[0]
    x = np.arange(4)
    ax.plot(x, [r["ratio"] for r in data["scaling"]], "o-", color=COLORS["legacy"],
            lw=2.3, ms=7, label="Measured legacy / baseline")
    ax.plot(x, [r["normalized_ratio"] for r in data["scaling"]], "s--",
            color=COLORS["normalized"], lw=2, ms=6, label=r"After dividing by $4^{\ell-1}$")
    for i, row in enumerate(data["scaling"]):
        ax.annotate(f"{row['expected_ratio']}×", (i, row["ratio"]), xytext=(0, 10),
                    textcoords="offset points", ha="center", color=COLORS["legacy"], fontweight="bold")
    ax.set_yscale("log", base=4)
    ax.set_yticks([1, 4, 16, 64], labels=["1", "4", "16", "64"])
    ax.set_ylim(.58, 120)
    ax.set_xticks(x, ["Conv 1", "Conv 2", "Conv 3", "Readout"])
    ax.set_ylabel("Layer voltage RMS ratio")
    ax.set_xlabel("Same-initializer Conv3 replay")
    ax.set_title("A   Voltage scaling is already measured", loc="left", pad=23)
    ax.text(0, 1.025, "One initialization · 576 examples · free states", transform=ax.transAxes,
            fontsize=9, color="#647184")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    ax = axes[1]
    f = np.linspace(-.04, 1.08, 401)
    for scale, label, color in [(1, "Baseline: target 1", COLORS["baseline"]),
                               (64, "Legacy: target 1/64", COLORS["legacy"])]:
        ax.plot(f, .5 * (f - 1 / scale) ** 2, color=color, lw=2.3, label=label)
        ax.scatter([1 / scale], [0], color=color, s=45, zorder=5)
        ax.axvline(1 / scale, color=color, lw=1, ls=":", alpha=.6)
    ax.set_xlim(-.04, 1.08)
    ax.set_ylim(-.025, .62)
    ax.set_xlabel(r"Unamplified true-class score $f$")
    ax.set_ylabel(r"Squared-error loss divided by $s^2$")
    ax.set_title("B   Fixed labels change the objective", loc="left", pad=23)
    ax.text(0, 1.025, "Analytic one-score slice · target y = 1", transform=ax.transAxes,
            fontsize=9, color="#647184")
    ax.text(.5, .96, r"$L_s/s^2=\frac{1}{2}(f-1/s)^2$", transform=ax.transAxes,
            ha="center", va="top", fontsize=14)
    ax.legend(loc="center", bbox_to_anchor=(.5, .55), frameon=False, fontsize=9)
    ax.text(.5, .74, "The same target voltage corresponds to\na different target in baseline coordinates.",
            transform=ax.transAxes, ha="center", fontsize=9, color="#647184")

    ax = axes[2]
    for scheme in ("baseline", "ours", "legacy"):
        rows = sorted((r for r in data["accuracy"] if r["scheme"] == scheme), key=lambda r: r["architecture"])
        ax.errorbar([1, 2, 3], [r["mean"] for r in rows], yerr=[r["sample_sd"] for r in rows],
                    color=COLORS[scheme], fmt="o-", lw=2, ms=5.5, capsize=3,
                    label=scheme.capitalize())
    ax.set_xticks([1, 2, 3], ["Conv1", "Conv2", "Conv3"])
    ax.set_xlim(.87, 3.13)
    ax.set_ylim(95.95, 99.15)
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_xlabel("Clean BPTT · wide conductance range")
    ax.set_title("C   The training advantage is observed", loc="left", pad=23)
    ax.text(0, 1.025, "Three seeds · mean ± sample SD", transform=ax.transAxes,
            fontsize=9, color="#647184")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(axis="y", color="#e7ebf0", lw=.8)
        ax.set_axisbelow(True)

    fig.text(.055, .125, "A: RMS ratios support scaling; they do not establish elementwise or gradient identity.  B: analytic curves, not training measurements.",
             fontsize=9, color="#647184")
    fig.text(.055, .087, "C: 5,000-example selection split; 10 / 30 / 30 epochs, Adam, zero biases, T=K of 4 / 6 / 8. Learning rates differ between schemes.",
             fontsize=9, color="#647184")
    fig.text(.055, .037, "Decisive next test: baseline + output gain, with legacy's optimizer, should reproduce legacy's logits, gradients and training trajectory.",
             fontsize=10, fontweight="bold")
    fig.savefig(destination, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/conv-output-scale-evidence-20260928-v1/analysis")
    args = parser.parse_args()
    data = collect()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "output_scale_evidence.jpg"
    plot(data, destination)
    data["figure"] = str(destination.resolve())
    (args.output_dir / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"figure": str(destination), "accuracy_results_checked": 27,
                      "voltage_ratios": [r["ratio"] for r in data["scaling"]]}))


if __name__ == "__main__":
    main()
