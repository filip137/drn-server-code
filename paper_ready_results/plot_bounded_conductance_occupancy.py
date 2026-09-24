#!/usr/bin/env python3
"""Read the manuscript's 81 bounded BPTT checkpoints and plot bound occupancy."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bounded_conductance_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHITECTURES = ("conv1", "conv2", "conv3")
SCHEMES = ("baseline", "ours", "legacy")
SCHEME_LABELS = {"baseline": "Baseline", "ours": "Balanced", "legacy": "Legacy"}
CEILINGS = (1e-4, 5e-4, 1e-3)
COLORS = ("#0072B2", "#D55E00", "#009E73")
MARKERS = ("o", "s", "^")
STEM = "bounded_conductance_occupancy_best"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def indexed_bytes(run, result, name):
    matches = [a for a in result["artifacts"] if Path(a["path"]).name == name]
    require(len(matches) == 1, f"Expected one indexed {name}: {run}")
    path = run / matches[0]["path"]
    data = path.read_bytes()
    require(digest(data) == matches[0]["sha256"], f"Hash mismatch: {path}")
    return data, matches[0]["sha256"]


def extract():
    with (ROOT / "three_table_overview_20260916.csv").open() as stream:
        inclusion = [r for r in csv.DictReader(stream) if r["table"] == "table2_native"]
    expected = {(a, s, g, seed) for a in ARCHITECTURES for s in SCHEMES
                for g in CEILINGS for seed in (0, 1, 2)}
    actual = [(r["architecture"], r["scheme"], float(r["G_max"]), int(r["seed"]))
              for r in inclusion]
    require(len(actual) == len(set(actual)) == 81 and set(actual) == expected,
            "Expected exactly the 81 manuscript shared-T/K bounded BPTT runs")
    runs, layers = [], []
    for entry in inclusion:
        result_path = ROOT.parent / entry["path"]
        run = result_path.parent
        result_data = result_path.read_bytes()
        require(digest(result_data) == entry["sha256"], f"Changed result: {run}")
        result = json.loads(result_data)
        status = read_json(run / "status.json")
        require(status["state"] == "complete" and result["completion"]["criteria_met"],
                f"Incomplete run: {run}")
        require(not result["smoke"] and not result["dataset"]["official_test_read"],
                f"Unexpected smoke or test-evaluated run: {run}")
        for name, hash_key in (("manifest.json", "manifest_sha256"),
                               ("metrics.jsonl", "metrics_sha256")):
            require(digest((run / name).read_bytes()) == result[hash_key],
                    f"Changed {name}: {run}")
        config_bytes, config_hash = indexed_bytes(run, result, "config.used.json")
        config = json.loads(config_bytes)
        model = config["model_base"]
        arch, scheme = entry["architecture"], entry["scheme"]
        gmin, gmax = float(entry["G_min"]), float(entry["G_max"])
        t = (4, 6, 8)[ARCHITECTURES.index(arch)]
        require(config["training_algorithm"] == "BP" and config["seed"] == int(entry["seed"]),
                f"Algorithm/seed mismatch: {run}")
        require(model["weight_min"] == gmin == 1e-5 and model["weight_max"] == gmax,
                f"Bound mismatch: {run}")
        require(model["num_iterations_inference"] == model["num_iterations_training"] == t,
                f"Unexpected T/K: {run}")
        require((model["voltage_amp"], model["current_amp"]) ==
                {"baseline": (1, 1), "ours": (4, 1), "legacy": (4, .25)}[scheme],
                f"Amplification mismatch: {run}")
        accuracy = 100 * result["terminal_metrics"]["validation"]["best_accuracy"]
        require(abs(accuracy - float(entry["best"])) < 1e-9, f"Accuracy mismatch: {run}")
        checkpoint_bytes, checkpoint_hash = indexed_bytes(run, result, "weights_best.npz")
        common = dict(architecture=arch, scheme=scheme, G_min=gmin, G_max=gmax,
                      seed=int(entry["seed"]), checkpoint_role="best_validation",
                      best_epoch=result["terminal_metrics"]["best_epoch"], T=t, K=t,
                      best_validation_accuracy_percent=accuracy,
                      run_path=str(run.relative_to(ROOT)),
                      checkpoint_sha256=checkpoint_hash, result_sha256=entry["sha256"],
                      config_sha256=config_hash)
        local = []
        with np.load(io.BytesIO(checkpoint_bytes), allow_pickle=False) as checkpoint:
            names = [n for n in checkpoint.files if n.startswith(("ConvWeight_", "DenseWeight_"))]
            expected_names = [f"ConvWeight_{i}" for i in range(ARCHITECTURES.index(arch) + 1)]
            require(set(names) == set(expected_names + ["DenseWeight_0"]),
                    f"Unexpected parameter set: {run}")
            for name in names:
                values = checkpoint[name]
                require(np.issubdtype(values.dtype, np.floating) and values.size > 0
                        and np.isfinite(values).all(), f"Invalid tensor {name}: {run}")
                lower, upper = np.asarray([gmin, gmax], dtype=values.dtype)
                require(np.all(values >= lower) and np.all(values <= upper),
                        f"Out-of-bounds tensor {name}: {run}")
                local.append(dict(parameter=name, num_weights=int(values.size),
                                  at_min_count=int(np.count_nonzero(values == lower)),
                                  at_max_count=int(np.count_nonzero(values == upper))))
            for row in local:
                row["at_min_percent"] = 100 * row["at_min_count"] / row["num_weights"]
                row["at_max_percent"] = 100 * row["at_max_count"] / row["num_weights"]
                layers.append({**common, **row})
        totals = {k: sum(r[k] for r in local) for k in ("num_weights", "at_min_count", "at_max_count")}
        totals.update({f"at_{bound}_percent": 100 * totals[f"at_{bound}_count"] / totals["num_weights"]
                       for bound in ("min", "max")})
        runs.append({**common, **totals})
    summary = []
    for arch in ARCHITECTURES:
        for scheme in SCHEMES:
            for ceiling in CEILINGS:
                group = [r for r in runs if (r["architecture"], r["scheme"], r["G_max"]) ==
                         (arch, scheme, ceiling)]
                require(sorted(r["seed"] for r in group) == [0, 1, 2], "Missing seed")
                row = dict(architecture=arch, scheme=scheme, G_min=1e-5, G_max=ceiling, n=3)
                for bound in ("min", "max"):
                    values = [r[f"at_{bound}_percent"] for r in group]
                    row[f"at_{bound}_mean_percent"] = float(np.mean(values))
                    row[f"at_{bound}_sd_percent"] = float(np.std(values, ddof=1))
                summary.append(row)
    return runs, layers, summary


def plot(summary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 4.8), sharex=True, sharey="row")
    x = np.array(CEILINGS)
    for j, arch in enumerate(ARCHITECTURES):
        axes[0, j].set_title(arch.capitalize(), fontsize=12, fontweight="medium", pad=10)
        for i, bound in enumerate(("min", "max")):
            ax = axes[i, j]
            for scheme, color, marker in zip(SCHEMES, COLORS, MARKERS):
                rows = [next(r for r in summary if (r["architecture"], r["scheme"], r["G_max"])
                             == (arch, scheme, ceiling)) for ceiling in CEILINGS]
                y = [r[f"at_{bound}_mean_percent"] for r in rows]
                sd = [r[f"at_{bound}_sd_percent"] for r in rows]
                ax.errorbar(x, y, yerr=sd, color=color, marker=marker, markersize=4.8,
                            linewidth=1.6, capsize=3, elinewidth=1, capthick=1)
            ax.set_xscale("log")
            ax.set_xticks(x, [r"$10^{-4}$", r"$5\!\times\!10^{-4}$", r"$10^{-3}$"])
            ax.minorticks_off()
            ax.set_xlim(8e-5, 1.25e-3)
            ax.grid(axis="y", color="0.9", linewidth=.7)
            ax.set_axisbelow(True)
            if i == 1:
                ax.set_xlabel(r"Upper conductance bound, $G_{\max}$", labelpad=7)
    for i, bound in enumerate(("min", "max")):
        ceiling = max(r[f"at_{bound}_mean_percent"] + r[f"at_{bound}_sd_percent"] for r in summary)
        top = min(100, max(5, np.ceil(ceiling * 1.15 / 5) * 5))
        axes[i, 0].set_ylim(0, top)
        axes[i, 0].set_ylabel(rf"Weights at $G_{{\{bound}}}$ (%)", labelpad=8)
    handles = [Line2D([], [], color=c, marker=m, linewidth=1.6, markersize=5, label=SCHEME_LABELS[s])
               for s, c, m in zip(SCHEMES, COLORS, MARKERS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1), ncol=3,
               frameon=False, handlelength=2.4, columnspacing=2.4)
    fig.text(.54, .014, r"Mean $\pm$ SD over 3 seeds; $G_{\min}=10^{-5}$",
             ha="center", fontsize=8.5, color="0.3")
    fig.subplots_adjust(left=.09, right=.985, top=.855, bottom=.14, hspace=.2, wspace=.13)
    for extension in ("png", "pdf", "svg", "jpg"):
        options = {"pil_kwargs": {"quality": 95, "subsampling": 0}} if extension == "jpg" else {}
        fig.savefig(ROOT / "figures" / f"{STEM}.{extension}", dpi=220,
                    bbox_inches="tight", facecolor="white", **options)
    plt.close(fig)


def main():
    runs, layers, summary = extract()
    write_csv(ROOT / f"{STEM}_per_seed.csv", runs)
    write_csv(ROOT / f"{STEM}_per_layer.csv", layers)
    write_csv(ROOT / f"{STEM}_summary.csv", summary)
    plot(summary)
    provenance = dict(source_inclusion="three_table_overview_20260916.csv: table2_native",
                      included_runs=len(runs), checkpoint_role="best_validation",
                      count_definition="Exact equality to bounds cast to checkpoint dtype; convolution and dense weights only",
                      aggregation="Pooled parameter counts within each seed; mean and sample SD across seeds 0,1,2",
                      checks="81 result, manifest, metrics.jsonl, config and selected checkpoint hashes; completion, contract, finiteness and bounds",
                      official_test_read=False, training_performed=False,
                      numpy_version=np.__version__, matplotlib_version=matplotlib.__version__,
                      command=f"{sys.executable} {Path(__file__).name}",
                      script_sha256=digest(Path(__file__).read_bytes()))
    (ROOT / f"{STEM}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Extracted {len(runs)} verified checkpoints, {len(layers)} weight tensors, {len(summary)} aggregates.")
    for r in summary:
        if r["G_max"] == 1e-4:
            print(f"{r['architecture']} {r['scheme']}: min {r['at_min_mean_percent']:.2f} +/- {r['at_min_sd_percent']:.2f}%; "
                  f"max {r['at_max_mean_percent']:.2f} +/- {r['at_max_sd_percent']:.2f}%")


if __name__ == "__main__":
    main()
