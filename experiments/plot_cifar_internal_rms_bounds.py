"""Locate sampled layer-RMS cosine boundaries in the completed ours e0/1/5 replay."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


EPOCHS = (0, 1, 5)
FOCUS = ((1, 1), (1, 2), (2, 1), (2, 2))
COLORS = {0: "#7B3294", 1: "#D55E00", 5: "#009E73"}
ROOT = Path("results/cifar-ours-epoch-beta-gradients-20260930-v1")


def passes(row, threshold):
    cosine = row["native_cosine"]
    return cosine is not None and cosine >= threshold


def compact(row):
    if row is None:
        return None
    fields = ("case_index", "beta_multiplier", "beta_injected_B", "state_relative_rms",
              "state_absolute_rms", "state_free_rms", "native_cosine", "native_norm_ratio")
    record = {key: row[key] for key in fields}
    record["batch_cosine"] = row["batch_statistics"]["native_cosine"]
    return record


def intervals(rows, threshold):
    """Retain every connected run of passing samples and its adjacent failures."""
    segments, start = [], None
    for index in range(len(rows) + 1):
        good = index < len(rows) and passes(rows[index], threshold)
        if good and start is None:
            start = index
        if not good and start is not None:
            segments.append(dict(
                first_pass=compact(rows[start]), last_pass=compact(rows[index - 1]),
                lower_fail=compact(rows[start - 1]) if start else None,
                upper_fail=compact(rows[index]) if index < len(rows) else None,
                passing_sample_count=index - start,
            ))
            start = None
    return segments


def plot_lower(groups, bounds, metric, out, threshold):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.1), sharey=True)
    for block, layer in FOCUS:
        ax = axes[block - 1, layer - 1]
        limits = []
        for epoch in EPOCHS:
            points = groups[epoch, block, layer]
            segment = bounds[epoch, block, layer]["passing_intervals"][0]
            first, previous = segment["first_pass"], segment["lower_fail"]
            assert previous is not None, "A lower boundary is needed for the focused plot"
            limits.extend((previous[metric], first[metric]))
            ax.plot([p[metric] for p in points], [p["native_cosine"] for p in points],
                    "o-", markersize=3, linewidth=1.8, color=COLORS[epoch])
            ax.axvspan(previous[metric], first[metric], color=COLORS[epoch], alpha=.09)
            ax.scatter([previous[metric]], [previous["native_cosine"]], s=65,
                       facecolors="white", edgecolors=COLORS[epoch], linewidths=1.8, zorder=4)
            ax.scatter([first[metric]], [first["native_cosine"]], s=65,
                       color=COLORS[epoch], edgecolors="white", linewidths=.8, zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(min(limits) / 4, max(limits) * 6)
        ax.set_ylim(-.02, 1.025)
        ax.axhline(threshold, color="#555555", linestyle=":", linewidth=1.1)
        ax.grid(alpha=.18, which="major")
        ax.set_title(f"Block {block} · Conv {layer}", loc="left", fontsize=13)
        if layer == 1:
            ax.set_ylabel("EqProp / native BPTT cosine")
        if block == 2:
            ax.set_xlabel("This layer's relative centered RMS displacement" if metric == "state_relative_rms"
                          else "This layer's absolute centered RMS displacement")
    handles = [Line2D([], [], color=COLORS[e], linewidth=2, label=f"Epoch {e}") for e in EPOCHS]
    handles.extend((
        Line2D([], [], color="#444444", marker="o", markerfacecolor="white", linestyle="none",
               label="Adjacent lower failing sample"),
        Line2D([], [], color="#444444", marker="o", linestyle="none", label="First passing sample"),
    ))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .895), ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Where do early convolutions lose alignment at small internal displacement?\n"
                 "Ours · saved EqProp trajectory · epochs 0, 1 and 5", fontsize=16, y=.98)
    definition = ("R = RMS((s+ − s−)/2) / RMS(s_free), using this convolution's postsynaptic state."
                  if metric == "state_relative_rms" else
                  "D = RMS((s+ − s−)/2), in simulation units, using this convolution's postsynaptic state.")
    fig.text(.5, .025, definition + "\n"
             f"Shading: sampled lower crossing of cosine {threshold:g}; no interpolation or universal cutoff.\n"
             "256 fixed images · pooled layer gradients · lower-bound zoom; upper failures may be outside the view.",
             ha="center", fontsize=10, linespacing=1.5)
    fig.subplots_adjust(top=.79, bottom=.15, left=.075, right=.98, hspace=.28, wspace=.17)
    stem = "early_layer_relative_rms_bounds" if metric == "state_relative_rms" else "early_layer_absolute_rms_bounds"
    fig.savefig(out / f"{stem}.jpg", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--threshold", type=float, default=.95)
    args = parser.parse_args()
    assert 0 < args.threshold < 1
    out = args.study_root.resolve() / "analysis"
    original = json.loads((out / "summary.json").read_text())
    groups = defaultdict(list)
    for row in original["layers"]:
        groups[row["epoch"], row["block"], row["layer"]].append(row)
    assert len(original["layers"]) == 408 and len(groups) == 24
    bounds = {}
    for (epoch, block, layer), rows in groups.items():
        rows.sort(key=lambda r: r["beta_multiplier"])
        assert len(rows) == 17 and len({r["case_index"] for r in rows}) == 17
        for metric in ("state_relative_rms", "state_absolute_rms"):
            assert all(r[metric] is not None and math.isfinite(r[metric]) and r[metric] > 0 for r in rows)
            assert all(a[metric] <= b[metric] for a, b in zip(rows, rows[1:])), (epoch, block, layer, metric)
        bounds[epoch, block, layer] = dict(epoch=epoch, block=block, layer=layer,
            passing_intervals=intervals(rows, args.threshold))
    blocks = []
    for epoch in EPOCHS:
        for block in (1, 2):
            by_case = [[groups[epoch, block, layer][case] for layer in (1, 2, 3)] for case in range(17)]
            passing = [i for i, rows in enumerate(by_case) if all(passes(r, args.threshold) for r in rows)]
            assert passing and passing == list(range(min(passing), max(passing) + 1))
            low, high = min(passing), max(passing)
            blocks.append(dict(epoch=epoch, block=block,
                first_passing_multiplier=by_case[low][0]["beta_multiplier"],
                last_passing_multiplier=by_case[high][0]["beta_multiplier"],
                lower_adjacent_failing_layers=[r["layer"] for r in by_case[low - 1] if not passes(r, args.threshold)] if low else None,
                upper_adjacent_failing_layers=[r["layer"] for r in by_case[high + 1] if not passes(r, args.threshold)] if high < 16 else None))
    summary = dict(
        source="summary.json (validated exp-022 replay, reused without rerunning)",
        scope="Ours EqProp-trained checkpoints at epochs 0/1/5; distinct from earlier BPTT epochs 0/10/30/50.",
        criterion=f"Pooled native-BPTT cosine >= {args.threshold:g}, separately for each Conv tensor.",
        interpretation="Adjacent sampled crossings, not exact or universal RMS thresholds. No between-sample interpolation. Null edge means not bracketed by the grid. Batch cosine statistics retained separately.",
        layers=list(bounds.values()), common_block_bands=blocks,
    )
    (out / "internal_rms_bounds.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = []
    for (epoch, block, layer), record in sorted(bounds.items()):
        for index, segment in enumerate(record["passing_intervals"]):
            row = dict(epoch=epoch, block=block, layer=layer, interval=index + 1)
            for edge in ("lower_fail", "first_pass", "last_pass", "upper_fail"):
                point = segment[edge]
                for metric in ("state_relative_rms", "state_absolute_rms", "native_cosine", "native_norm_ratio", "beta_multiplier"):
                    row[f"{edge}_{metric}"] = point[metric] if point else None
            table.append(row)
    with (out / "internal_rms_bounds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for metric in ("state_relative_rms", "state_absolute_rms"):
        plot_lower(groups, bounds, metric, out, args.threshold)
    for epoch in EPOCHS:
        for block, layer in FOCUS:
            segment = bounds[epoch, block, layer]["passing_intervals"][0]
            lower, first = segment["lower_fail"], segment["first_pass"]
            print(f"e{epoch} B{block} C{layer}: relative [{lower['state_relative_rms']:.4g}, {first['state_relative_rms']:.4g}], "
                  f"absolute [{lower['state_absolute_rms']:.4g}, {first['state_absolute_rms']:.4g}], "
                  f"cosine {lower['native_cosine']:.4f} -> {first['native_cosine']:.4f}")
    print(f"Saved two JPGs and boundary evidence under {out}")


if __name__ == "__main__":
    main()
