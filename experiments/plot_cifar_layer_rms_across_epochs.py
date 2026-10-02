"""Plot saved CIFAR RMS sweeps by convolution; no model replay or GPU work."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SCHEMES = ("baseline", "ours", "legacy")
EPOCHS = (0, 10, 30, 50)
LAYERS = tuple((b, l) for b, depth in ((1, 3), (2, 3), (3, 2)) for l in range(1, depth + 1))
COLORS = {0: "#7B3294", 10: "#0072B2", 30: "#D55E00", 50: "#009E73"}
DEFAULT_ROOT = Path("results/cifar-block-rms-intermediate-20260929-v1")


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def source_root(point, root):
    results = root.parent
    scheme, epoch = point["scheme"], point["epoch"]
    if epoch == 0:
        if point["evidence"] == "expanded":
            return results / "cifar-block-rms-expanded-20260929-v1" / scheme
        if scheme == "ours":
            return results / "cifar-block-rms-focused-20260929-v1"
        return results / "cifar-block-rms-schemes-20260929-v1" / scheme
    if epoch == 50:
        name = "cifar-block-rms-e50-low-20260929-v1" if point["case_index"] >= 12 else "cifar-block-rms-e50-20260929-v1"
        return results / name / scheme
    return root / f"e{epoch}" / scheme


def load_rows(root):
    comparison = json.loads((root / "analysis/comparison.json").read_text())
    sources, rows = {}, []
    for point in comparison["points"]:
        source = source_root(point, root)
        if source not in sources:
            data = json.loads((source / "run/artifacts/comparison.json").read_text())
            batches = defaultdict(list)
            for row in data["batches"]:
                batches[row["case_index"], row["name"]].append(row)
            sources[source] = data["layers"], batches
        layers, batches = sources[source]
        selected = sorted(
            (r for r in layers if r["block"] == point["block"]
             and r["target_relative_rms"] is not None
             and close(r["target_relative_rms"], point["target"])
             and close(r["beta_injected_B"], point["beta"])),
            key=lambda r: r["layer"],
        )
        assert len(selected) == (3 if point["block"] < 3 else 2), (source, point)
        for index, row in enumerate(selected):
            native = row["native_bptt"]
            assert close(native["cosine"], point["native_cosines"][index])
            assert close(row["output_displacement"]["relative_rms"], point["relative_rms"])
            batch = batches[row["case_index"], row["name"]]
            assert len(batch) == 8 and {r["batch"] for r in batch} == set(range(8))
            assert row["examples"] == sum(r["examples"] for r in batch) == 256
            assert close(native["norm_ratio"], native["eqprop_norm"] / native["bptt_norm"])
            cosines = [r["native_bptt"]["cosine"] for r in batch]
            assert all(c is not None and math.isfinite(c) for c in cosines)
            rows.append(dict(
                scheme=point["scheme"], epoch=point["epoch"], block=row["block"],
                layer=row["layer"], name=row["name"], target=point["target"],
                beta=point["beta"], relative_output_rms=point["relative_rms"],
                absolute_output_rms=point["absolute_rms"], cosine=native["cosine"],
                local_cosine=row["local_bptt"]["cosine"],
                bptt_norm=native["bptt_norm"], eqprop_norm=native["eqprop_norm"],
                norm_ratio=native["norm_ratio"], batch_cosine_min=min(cosines),
                batch_cosine_median=statistics.median(cosines), batch_cosine_max=max(cosines),
                mean_batch_bptt_norm=statistics.mean(r["native_bptt"]["bptt_norm"] for r in batch),
                source=str(source.relative_to(root.parent)), source_case=row["case_index"],
            ))
    groups = defaultdict(list)
    for row in rows:
        groups[row["scheme"], row["epoch"], row["block"], row["layer"]].append(row)
    assert len(groups) == len(SCHEMES) * len(EPOCHS) * len(LAYERS)
    assert len(rows) == sum(len(p["native_cosines"]) for p in comparison["points"])
    for key, group in groups.items():
        assert len({r["target"] for r in group}) == len(group), key
        for metric in ("bptt_norm", "mean_batch_bptt_norm"):
            assert all(close(r[metric], group[0][metric]) for r in group), (key, metric)
    return rows, groups, sorted(str(p.relative_to(root.parent)) for p in sources)


def passing_spans(points, threshold=0.95):
    # Use the shared sampled domain; these spans are not certified continuous intervals.
    points = sorted((r for r in points if r["relative_output_rms"] <= 0.41),
                    key=lambda r: r["relative_output_rms"])
    spans, current = [], []
    for row in points:
        if row["cosine"] >= threshold:
            current.append(row["relative_output_rms"])
        elif current:
            spans.append(current)
            current = []
    if current:
        spans.append(current)
    return [[s[0], s[-1]] for s in spans]


def summarize(groups):
    scales, bands = [], []
    for (scheme, epoch, block, layer), group in sorted(groups.items()):
        row = group[0]
        initial = groups[scheme, 0, block, layer][0]
        scales.append(dict(
            scheme=scheme, epoch=epoch, block=block, layer=layer,
            bptt_norm=row["bptt_norm"], relative_to_initial=row["bptt_norm"] / initial["bptt_norm"],
            mean_batch_bptt_norm=row["mean_batch_bptt_norm"],
            batch_norm_relative_to_initial=row["mean_batch_bptt_norm"] / initial["mean_batch_bptt_norm"],
            pooled_to_mean_batch_norm=row["bptt_norm"] / row["mean_batch_bptt_norm"],
        ))
        spans = passing_spans(group)
        best = max(group, key=lambda r: r["cosine"])
        bands.append(dict(scheme=scheme, epoch=epoch, block=block, layer=layer,
                          sampled_cosine95_spans=spans,
                          widest_log10_span=max((math.log10(hi / lo) for lo, hi in spans), default=0.),
                          peak_rms=best["relative_output_rms"], peak_cosine=best["cosine"]))
    return scales, bands


def layer_figure(groups, scheme, metric, out, zoom=False):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.7), sharex=True)
    for block, layer in LAYERS:
        ax = axes[block - 1, layer - 1]
        for epoch in EPOCHS:
            points = sorted(groups[scheme, epoch, block, layer], key=lambda r: r["relative_output_rms"])
            ax.plot([r["relative_output_rms"] for r in points], [r[metric] for r in points],
                    color=COLORS[epoch], linestyle="--" if epoch == 0 else "-",
                    marker="o", markersize=2.6, linewidth=1.65, label=f"Epoch {epoch}")
        ax.set_title(f"Block {block} · Conv {layer}", fontsize=12, loc="left")
        ax.set_xscale("log")
        ax.set_xlim(4e-4, 1.8)
        ax.axvspan(0.41, 1.8, color="#e7e7e7", alpha=0.6, zorder=-1)
        ax.grid(alpha=0.18, which="major")
        if metric == "cosine":
            ax.set_ylim((0.70, 1.006) if zoom else (-1.025, 1.025))
            ax.axhline(.95, color="#555555", linestyle=":", linewidth=.9)
            ax.axhline(1., color="#aaaaaa", linewidth=.6)
            if layer == 1:
                ax.set_ylabel("EqProp / native BPTT cosine")
        else:
            ax.set_yscale("log")
            ax.axhspan(.9, 1.1, color="#dddddd", alpha=.6, zorder=-1)
            ax.axhline(1., color="#555555", linestyle=":", linewidth=.9)
            if layer == 1:
                ax.set_ylabel("EqProp / BPTT gradient norm")
        if block == 3 or (block == 2 and layer == 3):
            ax.set_xlabel("Relative block-output RMS displacement R")
    axes[2, 2].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[2, 2].legend(handles, labels, loc="upper left", frameon=False, fontsize=12)
    axes[2, 2].text(.02, .42,
        "Each panel: one Conv weight tensor.\n"
        "Gradient vectors averaged over 256 images\n"
        "before cosine / norms are computed.\n\n"
        "R is measured at the block output,\n"
        "not at each internal layer.\n"
        "Grey x-range: only initialization sampled.",
        transform=axes[2, 2].transAxes, fontsize=10, va="top", linespacing=1.5)
    title = "Gradient direction" if metric == "cosine" else "Gradient magnitude agreement"
    suffix = " · high-cosine zoom" if zoom else ""
    fig.suptitle(f"{scheme.capitalize()} · every convolution across BPTT training\n{title}{suffix}", fontsize=17, y=.985)
    fig.text(.5, .01, "Same 256 training images · seed 0 · batch 32 · noiseless · T = K = [6, 6, 4] · saved-data replot",
             ha="center", fontsize=10)
    fig.subplots_adjust(top=.88, bottom=.085, left=.065, right=.985, hspace=.38, wspace=.24)
    name = f"{scheme}_{metric}" + ("_zoom" if zoom else "") + ".jpg"
    fig.savefig(out / name, dpi=190)
    plt.close(fig)


def magnitude_figure(scales, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey="row")
    lookup = {(r["scheme"], r["epoch"], r["block"], r["layer"]): r for r in scales}
    palette = plt.get_cmap("tab10")
    for col, scheme in enumerate(SCHEMES):
        for index, (block, layer) in enumerate(LAYERS):
            points = [lookup[scheme, epoch, block, layer] for epoch in EPOCHS]
            for row, metric in enumerate(("bptt_norm", "relative_to_initial")):
                axes[row, col].plot(EPOCHS, [p[metric] for p in points], "o-", color=palette(index),
                                    linewidth=1.5, markersize=4, label=f"B{block} Conv {layer}")
        axes[0, col].set_title(scheme.capitalize(), fontsize=14)
        for row in range(2):
            axes[row, col].set_yscale("log")
            axes[row, col].set_xticks(EPOCHS)
            axes[row, col].grid(alpha=.2, which="major")
        axes[1, col].set_xlabel("BPTT checkpoint epoch")
    axes[0, 0].set_ylabel("L2 norm of cohort-mean BPTT gradient")
    axes[1, 0].set_ylabel("BPTT gradient norm / initialization norm")
    handles = [Line2D([], [], color=palette(i), marker="o", label=f"B{b} Conv {l}")
               for i, (b, l) in enumerate(LAYERS)]
    fig.legend(handles=handles, ncol=8, loc="lower center", bbox_to_anchor=(.5, .015), fontsize=10, frameon=False)
    fig.suptitle("Gradient magnitudes change strongly across training\nEach line is one convolution; no layers are combined", fontsize=17, y=.98)
    fig.subplots_adjust(top=.84, bottom=.14, left=.075, right=.985, hspace=.24, wspace=.18)
    fig.savefig(out / "bptt_gradient_magnitudes.jpg", dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.study_root.resolve()
    out = args.output or root / "analysis/layer_curves"
    rows, groups, sources = load_rows(root)
    scales, bands = summarize(groups)
    out.mkdir(parents=True, exist_ok=True)
    for scheme in SCHEMES:
        layer_figure(groups, scheme, "cosine", out)
        layer_figure(groups, scheme, "cosine", out, zoom=True)
        layer_figure(groups, scheme, "norm_ratio", out)
    magnitude_figure(scales, out)
    with (out / "plotted_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        scope="Existing seed-0, noiseless BPTT-trajectory checkpoints; no new replay or training.",
        x_axis="Relative centered block-output RMS; internal layer RMS was not recorded in these sweeps.",
        aggregation="Per-layer gradient vectors averaged over 256 examples; layers are never concatenated.",
        validation=dict(layer_rows=len(rows), layer_epoch_groups=len(groups), source_roots=sources,
                        all_points_match_prior_comparison=True, eight_batches_per_point=True,
                        reference_norms_invariant_across_beta=True),
        band_note="Connected sampled passing points within R<=0.41; unequal grids, no certified continuous intervals.",
        gradient_scales=scales, layer_bands=bands,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["validation"], indent=2))
    for scheme in SCHEMES:
        reductions = [1 / r["relative_to_initial"] for r in scales if r["scheme"] == scheme and r["epoch"] == 50]
        batch_reductions = [1 / r["batch_norm_relative_to_initial"] for r in scales if r["scheme"] == scheme and r["epoch"] == 50]
        print(f"{scheme}: e0/e50 pooled norm factors {min(reductions):.1f}–{max(reductions):.1f}; "
              f"mean minibatch norm factors {min(batch_reductions):.1f}–{max(batch_reductions):.1f}")
    print(f"Saved 10 JPG plots and numeric evidence under {out}")


if __name__ == "__main__":
    main()
