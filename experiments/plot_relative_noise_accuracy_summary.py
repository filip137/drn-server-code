"""Display saved exp014--019 best/final validation accuracies; no replay."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("exp014", "conv123-relative-noise-training-20260925-v1", 84),
    ("exp015", "conv23-relative-noise-epoch30-20260926-v1", 12),
    ("exp015", "conv23-relative-noise-epoch30-low-noise-20260926-v1", 8),
    ("exp016", "conv23-noiseless-three-seed-20260927-v1", 12),
    ("exp016", "conv23-noiseless-baseline-three-seed-20260927-v1", 6),
    ("exp017", "conv23-baseline-relative-noise-20260927-v1", 10),
    ("exp018", "conv1-clean-and-baseline-noise-20260927-v1", 14),
    ("exp019", "conv123-relative-noise-eta5em4-jeanzay-20260928-v1", 9),
]
SCHEMES = ("baseline", "ours", "legacy")
COLORS = {"baseline": "#15916b", "ours": "#ec861b", "legacy": "#63666a"}
TARGETS = {1: 1, 2: 4, 3: 6}


def read_json(path):
    return json.loads(path.read_text())


def history(path, expected):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    epochs = [r for r in rows if r.get("kind") == "epoch"]
    assert [r["epoch"] for r in epochs] == list(range(1, expected + 1)), path
    values = [100 * r["metrics"]["validation_accuracy"] for r in epochs]
    assert all(math.isfinite(v) and 0 <= v <= 100 for v in values), path
    return values


def load_rows():
    rows = []
    for study, name, expected in SOURCES:
        source = ROOT / "results" / name
        csv_path = source / "analysis" / ("coverage.csv" if study == "exp014" else "accuracy.csv")
        entries = list(csv.DictReader(csv_path.open()))
        assert len(entries) == expected, csv_path
        for entry in entries:
            run = Path(entry.get("run") or entry.get("path") or entry["run_path"])
            if not run.is_absolute():
                run = ROOT / run
            manifest = read_json(run / "manifest.json")
            config = manifest["configuration"]["resolved"]
            arm = manifest["arm_id"]
            depth = int(arm[4])
            scheme = arm.split("_")[1]
            assert scheme in SCHEMES and read_json(run / "status.json")["state"] == "complete"
            assert not manifest["smoke"]
            assert config["evaluation"]["official_test"]["policy"] == "disabled"
            count = config["lab"]["epochs"]
            values = history(run / "metrics.jsonl", count)
            parent = ""
            if study == "exp015":
                assert count == 20
                parent_path = source / "source_epoch10" / arm / "metrics.jsonl"
                values = history(parent_path, 10) + values
                parent = str(parent_path.relative_to(ROOT))
            final = values[-1]
            reference = (entry.get("final_accuracy_pct") or entry.get("epoch30")
                         or entry.get("epoch10_accuracy_percent") or entry.get("epoch10"))
            assert reference is not None and math.isclose(final, float(reference), abs_tol=1e-9), arm
            best = max(values)
            rows.append(dict(
                study=study, arm=arm, architecture=f"conv{depth}", depth=depth,
                scheme=scheme, seed=int(config["seed"]),
                initial_df=float(config["relative_output_calibration"]["target"]),
                eta=float(config["eqprop"]["endpoint_read_noise_eta"]),
                epochs=len(values), best_accuracy_pct=best, final_accuracy_pct=final,
                best_epoch=values.index(best) + 1,
                optimizer_history="reset_after_epoch10" if study == "exp015" else "uninterrupted",
                host=manifest["runtime"]["host"], run_path=str(run.relative_to(ROOT)),
                parent_metrics_path=parent,
            ))
    assert len(rows) == 155
    return sorted(rows, key=lambda r: (r["depth"], r["initial_df"], r["epochs"], r["eta"], r["seed"], r["scheme"]))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cell(row):
    if row is None:
        return "—"
    suffix = " †" if row["optimizer_history"] == "reset_after_epoch10" else ""
    return f'{row["best_accuracy_pct"]:.2f} / {row["final_accuracy_pct"]:.2f}{suffix}'


def tables(rows):
    lines = []
    for depth, target, epochs in sorted({(r["depth"], r["initial_df"], r["epochs"]) for r in rows}):
        selected = [r for r in rows if (r["depth"], r["initial_df"], r["epochs"]) == (depth, target, epochs)]
        lines += [f"### Conv{depth} · D/F={target:g} · {epochs} epochs", "",
                  "| Relative noise η | Seed | Baseline best / final | Ours best / final | Legacy best / final |",
                  "|---|---:|---:|---:|---:|"]
        for eta, seed in sorted({(r["eta"], r["seed"]) for r in selected}):
            cells = {r["scheme"]: r for r in selected if r["eta"] == eta and r["seed"] == seed}
            assert len(cells) == sum(r["eta"] == eta and r["seed"] == seed for r in selected)
            label = "0 (clean)" if eta == 0 else f"{eta:g}"
            lines.append(f'| {label} | {seed} | ' + " | ".join(cell(cells.get(s)) for s in SCHEMES) + " |")
        lines += [""]
    return lines


def plot_panels(rows, groups, out, filename, title):
    fig, axes = plt.subplots(len(groups), 2, figsize=(14, 3.5 * len(groups) + 1.6), squeeze=False)
    for pair, (depth, target, epochs) in zip(axes, groups):
        selected = [r for r in rows if r["depth"] == depth and r["initial_df"] == target
                    and r["epochs"] == epochs and r["eta"] > 0 and r["seed"] == 0]
        lo = min(r[k] for r in selected for k in ("best_accuracy_pct", "final_accuracy_pct"))
        hi = max(r[k] for r in selected for k in ("best_accuracy_pct", "final_accuracy_pct"))
        ticks = sorted({r["eta"] for r in selected})
        for ax, metric, label in zip(pair, ("best_accuracy_pct", "final_accuracy_pct"), ("Best over training", "Final epoch")):
            for scheme in SCHEMES:
                rr = sorted((r for r in selected if r["scheme"] == scheme), key=lambda r: r["eta"])
                ax.plot([r["eta"] for r in rr], [r[metric] for r in rr], color=COLORS[scheme], lw=1.8)
                for r in rr:
                    marker = "^" if r["optimizer_history"] == "reset_after_epoch10" else "o"
                    ax.plot(r["eta"], r[metric], marker=marker, color=COLORS[scheme], ms=6, linestyle="none")
            if 5e-4 in ticks:
                ax.axvline(5e-4, color="#d2d2d2", lw=.9, ls=":", zorder=0)
            ax.set(xscale="log", ylim=(math.floor(lo)-.6, math.ceil(hi)+.3),
                   title=f"Conv{depth} · D/F={target:g} · {epochs} epochs — {label}",
                   xlabel="Relative endpoint read noise η", ylabel="Validation accuracy (%)")
            ax.set_xticks(ticks, [f"{v:g}" for v in ticks], rotation=40, ha="right")
            ax.minorticks_off()
            ax.grid(alpha=.2)
    handles = [Line2D([], [], color=COLORS[s], lw=2, label=s.capitalize()) for s in SCHEMES]
    handles += [Line2D([], [], color="#444444", marker="o", ls="none", label="Uninterrupted"),
                Line2D([], [], color="#444444", marker="^", ls="none", label="Adam/RNG reset after epoch 10")]
    fig.suptitle(title, fontsize=16, y=.99)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .958), ncol=5, frameon=False, fontsize=10)
    fig.text(.5, .016, "Seed 0 · clean MNIST validation · best = maximum over all declared epochs · final = fixed last epoch\n"
             "Descriptive comparisons across studies/hardware; optimizer histories differ. No official-test accuracy or seed uncertainty.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .063, 1, .91 if len(groups) == 1 else .92))
    fig.savefig(out / filename, dpi=180, facecolor="white", pil_kwargs={"quality": 95})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "campaigns/read-noise-beta/series/007-relative-noise-training/results/figures/best-final-noise-summary")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    primary = [r for r in rows if r["seed"] == 0 and r["initial_df"] == TARGETS[r["depth"]]
               and r["epochs"] == (10 if r["depth"] == 1 else 30)]
    assert len(primary) == 67
    write_csv(out / "all_results.csv", rows)
    write_csv(out / "main_results.csv", primary)
    plot_panels(primary, [(1, 1, 10), (2, 4, 30), (3, 6, 30)], out,
                "best_and_final_vs_noise.jpg", "Relative read-noise training: best and final accuracy")
    originals = [r for r in rows if r["study"] == "exp014"]
    for depth in (1, 2, 3):
        groups = sorted({(r["depth"], r["initial_df"], r["epochs"]) for r in originals if r["depth"] == depth})
        plot_panels(originals, groups, out, f"conv{depth}_original_10epoch_sweep.jpg",
                    f"Conv{depth}: complete original ten-epoch sweep (exp014)")
    lines = ["# Best and final relative read-noise accuracy", "",
             "Saved exp014–019 results only; no training or replay. 155 recorded outcomes: 128 noisy and 27 clean. "
             "The 20 continuation outcomes extend their original ten-epoch parents; they are not independent replicates.", "",
             "Every table cell is **best / final validation accuracy (%)**. Best is the maximum over the full stated training history; "
             "for exp015 this includes parent epochs 1–10 and continuation epochs 11–30. Final is the fixed last epoch. "
             "These columns must not be interpreted as test accuracy.", "",
             "† Adam and data/noise streams reset after epoch 10. All other trajectories are uninterrupted. "
             "Noisy results use one seed; clean controls include seeds 0/1/2. Initial D/F is matched at seed 0 only. "
             "Hardware and optimizer-history differences prevent isolating noise effects across studies. "
             "A dash means that combination was not run in the corresponding budget; it is not a zero.", "",
             "## Main comparison: selected D/F, full budgets, seed 0", "",
             "Conv1 uses D/F=1 and 10 epochs; Conv2 D/F=4 and Conv3 D/F=6 use 30 epochs. "
             "The Conv2/3 1e-8 and 1e-7 measurements have only ten epochs and appear in the original-sweep tables below. "
             "Clean rows are in the tables but omitted from logarithmic noise plots.", "",
             "![Best and final versus noise](best_and_final_vs_noise.jpg)", ""]
    lines += tables(primary)
    lines += ["## Complete original ten-epoch sweep (84 outcomes)", ""]
    for depth in (1, 2, 3):
        lines += [f"[Conv{depth} complete original sweep JPG](conv{depth}_original_10epoch_sweep.jpg)", ""]
    lines += tables(originals)
    lines += ["## Clean controls: all three seeds (27 outcomes)", ""] + tables([r for r in rows if r["eta"] == 0])
    lines += ["## Sources and reproduction", "",
              "[All 155 outcomes and source paths](all_results.csv) · [Main 67 outcomes](main_results.csv). "
              "Source CSV final values were checked against the raw complete epoch histories; existing study validation was reused. "
              "Each CSV row links its run directory and, when applicable, parent metrics.", "",
              "The main tables plus original-sweep tables and clean-controls tables cover every source outcome; some appear in more than one view.", "",
              "Reproduce from repository root: `python -m experiments.plot_relative_noise_accuracy_summary`.", ""]
    (out / "README.md").write_text("\n".join(lines))
    print(json.dumps(dict(out=str(out),all_rows=len(rows),noisy=sum(r["eta"]>0 for r in rows),clean=sum(r["eta"]==0 for r in rows),main_rows=len(primary),figures=4)))


if __name__ == "__main__":
    main()
