"""Plot existing initialization gradient summaries without new simulations."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments.plot_layer_fractional_displacement import CASES, STYLES


def parameters(arch):
    return [f"ConvWeight_{i}" for i in range(int(arch[-1]))] + ["DenseWeight_0"]


def labels(arch):
    return [f"Conv {i+1}" for i in range(int(arch[-1]))] + ["Readout"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=Path(
        "results/conv123-relative-noise-advantage-map-20260925-v1/analysis/stage_a/layer_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "campaigns/read-noise-beta/series/006-relative-endpoint-noise/results/figures/exp013-gradient-quality"))
    args = parser.parse_args()
    with args.source_csv.open() as stream:
        source = list(csv.DictReader(stream))
    rows = [r for r in source if (r["architecture"], float(r["relative_target"])) in CASES]
    lookup = {}
    for row in rows:
        key = (row["architecture"], row["scheme"], row["parameter_name"], float(row["noise_level"]))
        assert key not in lookup, f"Duplicate cell: {key}"
        assert int(row["draws"]) == (3 if key[-1] else 1)
        for metric in ("cosine", "noisy_clean_eqprop_cosine", "noise_over_clean_norm"):
            lo, mid, hi = [float(row[metric + suffix]) for suffix in ("_min", "", "_max")]
            assert all(math.isfinite(x) for x in (lo, mid, hi))
            assert lo - 1e-12 <= mid <= hi + 1e-12
            if metric.endswith("cosine"):
                assert -1.00000001 <= lo <= hi <= 1.00000001
            else:
                assert lo >= 0 and (not key[-1] or lo > 0)
        lookup[key] = row
    etas = sorted({float(r["noise_level"]) for r in rows if float(r["noise_level"]) > 0})
    expected = {(arch, scheme, param, eta) for arch, _ in CASES
                for scheme in STYLES for param in parameters(arch) for eta in [0., *etas]}
    assert set(lookup) == expected and len(rows) == 405
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "plotted_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    handles = [Line2D([], [], label=s.capitalize(), linewidth=1.8, **style)
               for s, style in STYLES.items()]

    for arch, target in CASES:
        params = parameters(arch)
        fig, axes = plt.subplots(2, len(params), figsize=(4 * len(params), 7.2), squeeze=False)
        for col, (param, label) in enumerate(zip(params, labels(arch))):
            for metric_index, metric in enumerate(("cosine", "noise_over_clean_norm")):
                ax = axes[metric_index, col]
                for scheme, style in STYLES.items():
                    cells = [lookup[(arch, scheme, param, eta)] for eta in etas]
                    y, lo, hi = [[float(r[metric + suffix]) for r in cells]
                                 for suffix in ("", "_min", "_max")]
                    ax.plot(etas, y, linewidth=1.7, markeredgewidth=1.1, **style)
                    ax.fill_between(etas, lo, hi, color=style["color"], alpha=.12, linewidth=0)
                ax.set_xscale("log")
                ax.set_xticks([1e-8, 1e-6, 1e-4, 1e-2])
                ax.grid(which="major", alpha=.2)
                ax.spines[["top", "right"]].set_visible(False)
                if metric_index == 0:
                    ax.set_title(label)
                    ax.set_ylim(-.12, 1.035)
                    ax.axhline(0, color="gray", linewidth=.6)
                else:
                    ax.set_yscale("log")
                    ax.axhline(1, color="gray", linewidth=.8, linestyle="--")
                    ax.set_xlabel(r"Relative read noise $\eta$")
                if col == 0:
                    ax.set_ylabel("Cosine to BPTT\n(higher is better)" if metric_index == 0 else
                                  "Noise error / clean EqProp norm\n(lower is better)")
        fig.suptitle(f"{arch.capitalize()} initialization · output D/F = {target:g}", fontsize=15, y=.99)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .947), ncol=3, frameon=False)
        fig.text(.5, .025,
                 "One relative-noise coefficient for every noninput layer. Bands: range of three noise-draw means.\n"
                 "192 examples · one initializer · finite-K BPTT · noise error = ‖g(noisy EP) − g(clean EP)‖ / ‖g(clean EP)‖",
                 ha="center", fontsize=9)
        fig.subplots_adjust(left=.10 if len(params)==2 else .07, right=.985,
                            bottom=.15, top=.85, hspace=.34, wspace=.30)
        fig.savefig(out / f"{arch}_gradient_quality_vs_noise.jpg", dpi=180,
                    facecolor="white", pil_kwargs={"quality": 95})
        plt.close(fig)

    profile_etas = (1e-6, 1e-5, 1e-4, 3e-4)
    for metric, name, title in (
        ("cosine", "cosine_to_bptt_across_layers", "Gradient cosine to BPTT across layers"),
        ("noisy_clean_eqprop_cosine", "cosine_to_clean_ep_across_layers",
         "Gradient cosine to clean EqProp across layers"),
    ):
        fig, axes = plt.subplots(3, len(profile_etas), figsize=(16.5, 9.7), sharey=True)
        for row_index, (arch, target) in enumerate(CASES):
            params = parameters(arch)
            x = list(range(len(params)))
            for col, eta in enumerate(profile_etas):
                ax = axes[row_index, col]
                for scheme, style in STYLES.items():
                    cells = [lookup[(arch, scheme, param, eta)] for param in params]
                    y, lo, hi = [[float(r[metric + suffix]) for r in cells]
                                 for suffix in ("", "_min", "_max")]
                    ax.plot(x, y, linewidth=1.7, markeredgewidth=1.2, **style)
                    ax.fill_between(x, lo, hi, color=style["color"], alpha=.12, linewidth=0)
                ax.set_xticks(x, labels(arch))
                ax.set_ylim(-.12, 1.035)
                ax.grid(axis="y", alpha=.2)
                ax.axhline(0, color="gray", linewidth=.6)
                ax.spines[["top", "right"]].set_visible(False)
                if row_index == 0:
                    exponent = math.floor(math.log10(eta))
                    mantissa = eta / 10 ** exponent
                    prefix = "" if math.isclose(mantissa, 1) else rf"{mantissa:g}\times "
                    ax.set_title(rf"$\eta = {prefix}10^{{{exponent}}}$")
                if col == 0:
                    ax.set_ylabel(f"{arch.capitalize()} · output D/F={target:g}\nGradient cosine")
        fig.suptitle(title + " at initialization", fontsize=15, y=.99)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .95), ncol=3, frameon=False)
        fig.text(.5, .025, "One relative-noise coefficient for every noninput layer.\n"
                 "192 examples · one initializer · bands: range of three noise-draw means, not confidence intervals",
                 ha="center", fontsize=9)
        fig.subplots_adjust(left=.085, right=.985, bottom=.11, top=.86, hspace=.32, wspace=.15)
        fig.savefig(out / f"{name}.jpg", dpi=180, facecolor="white", pil_kwargs={"quality": 95})
        plt.close(fig)

    (out / "README.md").write_text(
        "# Initialization gradient quality\n\n"
        "[Cosine to BPTT across layers](cosine_to_bptt_across_layers.jpg) · "
        "[Cosine to clean EqProp across layers](cosine_to_clean_ep_across_layers.jpg)\n\n"
        "The four layer-profile columns use eta=1e-6, 1e-5, 1e-4 and 3e-4.\n\n"
        "Noise sweeps, with cosine and relative noise-error norm: "
        "[Conv1](conv1_gradient_quality_vs_noise.jpg) · "
        "[Conv2](conv2_gradient_quality_vs_noise.jpg) · "
        "[Conv3](conv3_gradient_quality_vs_noise.jpg).\n\n"
        "Saved exp013 stage-A summaries, at matched initial output D/F=1/4/6 "
        "for Conv1/2/3. The 14 nonzero noise levels use 192 examples (12 batches of "
        "16), three acquisition draws and one model initializer. Every noninput "
        "layer uses the same relative coefficient eta. No training or new replay. "
        "Stages are not pooled, and uncertainty over model seeds is unavailable. "
        "Curves are means of per-batch metrics; bands span the three draw means, "
        "not confidence intervals. Finite-K BPTT is the reference, not a claimed "
        "exact equilibrium gradient.\n\n"
        "Relative noise error is ||g_noisy_EP - g_clean_EP|| / ||g_clean_EP||. "
        "This isolates acquisition perturbation; cosine to BPTT also includes "
        "any clean EqProp/BPTT discrepancy. The separate clean-EqProp cosine plot "
        "makes that distinction visible. Noise-sweep norm-error panels have "
        "individually scaled logarithmic y-axes; their horizontal line at 1 means "
        "error norm equals clean-gradient norm.\n\n"
        "The accompanying CSV contains all 405 selected summary rows, including "
        "27 eta=0 controls evaluated on the original 576-example clean cohort. "
        "Zero-noise rows are retained for provenance but are not plotted on the "
        "logarithmic noise axis or pooled with the 192-example noisy estimates. "
        "Original source: results/conv123-relative-noise-advantage-map-20260925-v1/"
        "analysis/stage_a/layer_summary.csv.\n\n"
        f"Reproduce: `python -m experiments.plot_layer_gradient_quality --source-csv "
        f"{out}/plotted_values.csv --output-dir {out}`.\n"
    )
    print(f"Verified {len(rows)} cells and wrote five JPGs to {out}")


if __name__ == "__main__":
    main()
