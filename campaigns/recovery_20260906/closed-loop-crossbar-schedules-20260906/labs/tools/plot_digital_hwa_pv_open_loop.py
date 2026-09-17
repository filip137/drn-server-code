"""Stacked crossbar/DRN KL plots: digital HWA, P&V, open-loop training."""

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
OUT = ROOT / "analysis/digital_hwa_open_loop_20260908"
RECOVERY = ROOT / "analysis/notion_single_graph_corrected_20260907/provenance.json"
CROSSBAR = ROOT / "analysis/stochastic_hwa_predeployment_20260908/crossbar_readout.json"
DRN = OUT / "drn_digital_readout.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = json.loads(RECOVERY.read_text())
    crossbar = json.loads(CROSSBAR.read_text())
    drn = json.loads(DRN.read_text())
    assert crossbar["status"] == drn["status"] == "complete"
    assert crossbar["unmodified_digital_master_diagnostic"]["examples"] == 10000
    assert drn["evaluation"]["examples"] == 10000 and not drn["stochastic_noise"]
    assert drn["writes"] == drn["optimizer_updates"] == 0
    digital = {"crossbar": crossbar["unmodified_digital_master_diagnostic"]["kl_teacher_student"],
               "drn": drn["evaluation"]["kl_teacher_student"]}
    values = {}
    records = []
    for architecture, prefix in (("crossbar", "Crossbar"), ("drn", "DRN")):
        healthy = next(s for s in old["series"] if s["label"] == f"{prefix} · Healthy")
        corrupt = next(s for s in old["series"] if s["label"] == f"{prefix} · Corrupt")
        values[architecture] = dict(digital=digital[architecture],
            pv=[healthy["values"][0], corrupt["values"][0]],
            open_loop=[healthy["values"][1], corrupt["values"][1]])
        records.append(dict(architecture=architecture, stage="HWA digital weights",
            condition="shared", test_KL=digital[architecture], test_examples=10000))
        for stage, key in (("P&V", "pv"), ("Open-loop on-chip training", "open_loop")):
            for condition, value in zip(("healthy", "corrupt"), values[architecture][key], strict=True):
                records.append(dict(architecture=architecture, stage=stage,
                    condition=condition, test_KL=value, test_examples=10000))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    colors = {"digital": "#738291", "healthy": "#26799D", "corrupt": "#CE6432"}
    upper = max(.082, 1.23 * max(r["test_KL"] for r in records))
    stage_x = np.array([0., 1.5, 3.])
    stage_labels = ["a) HWA digital weights", "b) P&V", "c) After open-loop\non-chip training"]
    legend = [Patch(facecolor=colors["healthy"], label="Healthy devices"),
              Patch(facecolor=colors["corrupt"], label="Corrupt devices")]

    def draw(ax, architecture):
        v = values[architecture]
        baseline = ax.bar(stage_x[0], v["digital"], width=.50,
            color=colors["digital"], edgecolor="white", linewidth=.6, zorder=3)
        ax.bar_label(baseline, labels=[f"{v['digital']:.6f}"], padding=5, fontsize=12)
        for offset, condition, index in ((-.19, "healthy", 0), (.19, "corrupt", 1)):
            bar_values = [v["pv"][index], v["open_loop"][index]]
            bars = ax.bar(stage_x[1:] + offset, bar_values, width=.35,
                color=colors[condition], edgecolor="white", linewidth=.6, zorder=3)
            ax.bar_label(bars, labels=[f"{k:.6f}" for k in bar_values], padding=5, fontsize=12)
        ax.set_title("Crossbar arrays" if architecture == "crossbar" else "DRN",
                     loc="left", fontsize=20, weight="bold", pad=17)
        ax.set_ylabel("Test KL(teacher || student)", fontsize=13, labelpad=12)
        ax.set_xticks(stage_x, stage_labels, fontsize=13)
        ax.set_xlim(-.65, 3.68)
        ax.set_ylim(0, upper)
        ax.set_yticks(np.arange(0, upper, .02))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E5E8EC")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0, pad=12)

    fig, axes = plt.subplots(2, 1, figsize=(11.6, 9.6))
    fig.subplots_adjust(left=.115, right=.975, bottom=.105, top=.885, hspace=.49)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.55, .985),
               ncol=2, frameon=False, fontsize=15)
    draw(axes[0], "crossbar")
    draw(axes[1], "drn")
    outputs = []
    for ext in ("png", "pdf", "svg"):
        path = OUT / f"digital_hwa_pv_open_loop.{ext}"
        fig.savefig(path, dpi=220, facecolor="white")
        shutil.copy2(path, ROOT.parent / f"recovery_digital_hwa_open_loop.{ext}")
        outputs.append(path)
    plt.close(fig)
    # Also export each panel independently for reuse in slides/manuscripts.
    for architecture in ("crossbar", "drn"):
        fig, ax = plt.subplots(figsize=(11.6, 5.3))
        fig.subplots_adjust(left=.115, right=.975, bottom=.22, top=.77)
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.57, .99),
                   ncol=2, frameon=False, fontsize=15)
        draw(ax, architecture)
        for ext in ("png", "pdf"):
            path = OUT / f"{architecture}_digital_hwa_pv_open_loop.{ext}"
            fig.savefig(path, dpi=220, facecolor="white")
            outputs.append(path)
        plt.close(fig)

    with (OUT / "kl_values.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    provenance = dict(status="complete", matplotlib_version=matplotlib.__version__,
        axes_order=["crossbar", "drn"], records=records, sources={str(p): digest(p)
        for p in (RECOVERY, CROSSBAR, DRN, Path(__file__), OUT / "readout_protocol.md")},
        previous_PV_and_open_loop_values_preserved=8,
        crossbar_digital_state="unmodified_HWA_master_without_device_noise_or_support_projection",
        drn_digital_state="HWA_master_in_nominal_Drn_RESET_0.1_SET_2.0_without_endpoint_noise",
        original_fixed_gains_retained=True, closed_loop_included=False,
        figure_sha256={p.name: digest(p) for p in outputs})
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    lines = ["# Digital HWA, P&V and open-loop on-chip training", "",
        "Two Matplotlib panels: crossbar arrays above DRN. All values use the official 10,000-image MNIST test set.", "",
        "| Architecture | Digital HWA | P&V healthy | P&V corrupt | Open-loop healthy | Open-loop corrupt |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for architecture, v in values.items():
        lines.append(f"| {architecture} | " + " | ".join(f"{x:.6f}" for x in
            (v["digital"], *v["pv"], *v["open_loop"])) + " |")
    lines += ["", "Digital HWA uses the frozen trained weights with hardware noise disabled. "
        "For DRN this retains the DRN solver and uses the original endpoint law's nominal RESET=0.1 and SET=2.0 for every cell. "
        "The output gain remains fixed at 3.1622776601683795. No ReLU surrogate or gain refit was substituted.", "",
        "P&V and open-loop training use the saved held apparent device states. The eight values are unchanged from the corrected "
        "September 7 comparison, including the DRN learning-rate-sweep correction. "
        "Open-loop checkpoints were selected by validation from their original 30-epoch budgets.", "",
        "The two architectures retain different teachers and device/fault models. Recovery coverage is one array/write, so no error bars "
        "are shown. Digital values are deterministic diagnostics. No training or programming was rerun.", "",
        "[Combined figure](digital_hwa_pv_open_loop.png) · [PDF](digital_hwa_pv_open_loop.pdf) · [Values](kl_values.csv) · "
        "[Protocol](readout_protocol.md) · [Provenance](provenance.json)", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print(ROOT.parent / "recovery_digital_hwa_open_loop.png")
    print(json.dumps(values))


if __name__ == "__main__":
    main()
