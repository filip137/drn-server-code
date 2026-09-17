"""Selected HWA-before-P&V, deployed, open-loop and closed-loop recovery KL."""

import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "analysis/notion_maxence/selected_cases.csv"
PREVIOUS = ROOT / "analysis/notion_single_graph_corrected_20260907/provenance.json"
OUT = ROOT / "analysis/stochastic_hwa_predeployment_20260908"
TUNED = ROOT.parent / "recovery-lr-schedule-20260906"
OUT.mkdir(parents=True, exist_ok=True)
previous = json.loads(PREVIOUS.read_text())
baselines = {}
for architecture in ("crossbar", "drn"):
    path = OUT / f"{architecture}_readout.json"
    baseline = json.loads(path.read_text())
    assert baseline["status"] == "complete" and baseline["architecture"] == architecture
    assert baseline["primary"]["examples"] == 10000
    assert baseline["p0_replay"]["passed"] and baseline["input_files_unchanged"]
    assert baseline["writes"] == baseline["optimizer_updates"] == 0
    assert baseline["validation_replay"]["passed"] and not baseline["deployment_array_used"]
    assert len(baseline["views"]) == baseline["primary"]["stochastic_views"] == 64
    assert not baseline["selection_performed"] and not baseline["calibration_refit"]
    assert baseline["shared_conditions"] == ["healthy", "faulted"]
    assert baseline["protocol_sha256"] == hashlib.sha256((OUT / "readout_protocol.md").read_bytes()).hexdigest()
    baselines[architecture] = baseline
with SOURCE.open() as f:
    rows = list(csv.DictReader(f))
series = []
corrections = []
for architecture, condition, label, color, hatch in (
    ("crossbar", "healthy", "Crossbar · Healthy", "#26799D", None),
    ("crossbar", "faulted", "Crossbar · Corrupt", "#CE6432", None),
    ("drn", "healthy", "DRN · Healthy", "#8AC1D5", "//"),
    ("drn", "faulted", "DRN · Corrupt", "#F0B192", "//"),
):
    op, cl = [next(r for r in rows if (r["architecture"], r["condition"], r["writer"]) ==
                  (architecture, condition, writer)) for writer in ("open_loop", "closed_loop_pv")]
    results = [json.loads(Path(r["source"]).read_text()) for r in (op, cl)]
    assert results[0]["manifest"]["initial_state"] == results[1]["manifest"]["initial_state"]
    assert results[0]["initial_test"] == results[1]["initial_test"]
    for row, r in zip((op, cl), results):
        assert r["status"] == "complete" and r["selected_replay_passed"]
        assert float(row["selected_test_KL"]) == r["selected_test"]["apparent"]["kl_teacher_student"]
    open_kl = float(op["selected_test_KL"])
    if architecture == "drn":
        candidates = [(p, json.loads(p.read_text())) for p in
                      sorted((TUNED / "drn_screen/runs").glob(f"{condition}-*/result.json"))]
        assert all(r["status"] == "complete" for _, r in candidates)
        selected_path, selected_run = min(candidates, key=lambda pair: (
            pair[1]["selected"]["validation"]["apparent"]["kl_teacher_student"],
            -pair[1]["selected"]["validation"]["apparent"]["student_accuracy"],
            pair[1]["selected"]["epoch"]))
        expected_arm = "healthy-constant-3e-5" if condition == "healthy" else "faulted-constant-3e-6"
        assert selected_path.parent.name == expected_arm
        readout_path = TUNED / f"drn_readouts/test/{condition}-array-1-write-1-constant.json"
        readout = json.loads(readout_path.read_text())
        p0 = json.loads((TUNED / f"drn_readouts/test/{condition}-array-1-write-1-p0.json").read_text())
        manifest = json.loads((selected_path.parent / "manifest.json").read_text())
        assert p0["evaluation"] == results[1]["initial_test"]
        assert selected_run["initial"] == results[1]["initial"]
        assert manifest["deployment_sha256"] == results[1]["manifest"]["inputs"]["deployment"]["sha256"]
        assert readout["selected_epoch"] == selected_run["selected"]["epoch"]
        assert readout["status"] == "complete" and readout["unchanged_state_check"]["passed"]
        assert selected_run["selected_replay_passed"]
        assert readout["evaluation"]["apparent"]["examples"] == 10000
        open_kl = readout["evaluation"]["apparent"]["kl_teacher_student"]
        corrections.append(dict(condition=condition, previous_open_KL=float(op["selected_test_KL"]),
                                corrected_open_KL=open_kl, result=str(selected_path), readout=str(readout_path),
                                readout_sha256=hashlib.sha256(readout_path.read_bytes()).hexdigest(),
                                validation_KL=selected_run["selected"]["validation"]["apparent"]["kl_teacher_student"],
                                matching_initial_validation_and_test=True, matching_deployment_hash=True,
                                candidates_considered=len(candidates), selected_epoch=readout["selected_epoch"]))
    prior_values = [float(op["initial_test_KL"]), open_kl, float(cl["selected_test_KL"])]
    assert prior_values == next(s["values"] for s in previous["series"] if s["label"] == label)
    hwa_kl = baselines[architecture]["primary"]["kl_teacher_student"]
    series.append(dict(architecture=architecture, condition=condition,
                       label=label, color=color, hatch=hatch,
                       values=[hwa_kl, *prior_values],
                       hwa_standard_deviation=baselines[architecture]["primary"]["kl_sd_across_views"]))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(16.8, 6.2))
fig.subplots_adjust(left=.065, right=.985, bottom=.20, top=.76)
x = np.arange(4) * 1.3
for offset, s in zip((-.315, -.105, .105, .315), series):
    bars = ax.bar(x + offset, s["values"], width=.195, color=s["color"],
                  hatch=s["hatch"], edgecolor="#FFFFFF", linewidth=.6, label=s["label"], zorder=3)
    ax.bar_label(bars, labels=["", *[f"{v:.6f}" for v in s["values"][1:]]], padding=5, fontsize=9)
    ax.errorbar(x[0] + offset, s["values"][0], yerr=s["hwa_standard_deviation"],
                fmt="none", ecolor="#303B46", capsize=3, elinewidth=1.1, zorder=4)
    ax.annotate(f"{s['values'][0]:.6f}",
                (x[0] + offset, s["values"][0] + s["hwa_standard_deviation"]),
                xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9)
stages = ["After stochastic HWA\nbefore deployment", "Before recovery\nHWA + P&V",
          "After open-loop\nrecovery", "After closed-loop\nP&V recovery"]
ax.set_xticks(x, stages, fontsize=13)
ax.set_ylabel("Test KL(teacher || student) ↓", labelpad=9)
ax.set_ylim(0, .082)
ax.set_yticks(np.arange(0, .081, .02))
ax.set_axisbelow(True)
ax.grid(axis="y", color="#E5E8EC")
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("KL divergence from stochastic HWA to recovery", x=.065, y=.97,
             ha="left", fontsize=22, weight="bold")
fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.055, .90),
           ncol=4, frameon=False, fontsize=14)
fig.text(.065, .063, "Before deployment: mean KL across 64 stochastic HWA views; whiskers show their standard deviation.", fontsize=11, color="#424A52")
fig.text(.065, .024, "Each HWA baseline is shared by the healthy and corrupt branches. Recovery bars show the selected array/write.", fontsize=11, color="#424A52")
paths = []
for ext in ("png", "svg", "pdf"):
    path = OUT / f"open_closed_recovery.{ext}"
    fig.savefig(path, dpi=220, facecolor="white")
    paths.append(path)
plt.close(fig)
(OUT / "provenance.json").write_text(json.dumps({
    "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "crossbar_and_closed_loop_selection_provenance": str(ROOT / "analysis/notion_maxence/selection_provenance.json"),
    "drn_open_loop_corrections": corrections,
    "correction_reason": "The earlier graph omitted the completed DRN learning-rate sweep and used the older open-loop controls.",
    "budgets": {"crossbar": 30, "drn_open_loop": 30, "drn_closed_loop": 3},
    "series": series, "axes_count": 1, "stages": stages,
    "previous_corrected_figure": str(PREVIOUS),
    "previous_provenance_sha256": hashlib.sha256(PREVIOUS.read_bytes()).hexdigest(),
    "all_previous_KL_values_preserved_exactly": True,
    "pre_pv_readouts": {a: {"path": str(OUT / f"{a}_readout.json"),
        "sha256": hashlib.sha256((OUT / f"{a}_readout.json").read_bytes()).hexdigest(),
        "test_KL": b["primary"]["kl_teacher_student"], "state": b["primary_state"],
        "standard_deviation_across_views": b["primary"]["kl_sd_across_views"],
        "stochastic_views": b["primary"]["stochastic_views"],
        "shared_across_healthy_and_corrupt": True} for a, b in baselines.items()},
    "protocol": str(OUT / "readout_protocol.md"),
    "figure_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
}, indent=2) + "\n")

with (OUT / "kl_values.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["architecture", "condition", "after_hwa_before_pv_KL",
        "after_hwa_pv_KL", "after_open_loop_KL", "after_closed_loop_KL", "stochastic_hwa_KL_SD", "HWA_views"])
    writer.writeheader()
    for s in series:
        writer.writerow(dict(zip(writer.fieldnames,
            [s["architecture"], s["condition"], *s["values"], s["hwa_standard_deviation"], 64], strict=True)))

lines = ["# Recovery KL with the stochastic HWA model before deployment", "",
    "All values are mean KL(teacher || student), in nats, on the official 10,000-image MNIST test set.", "",
    "| Architecture | Condition | After HWA, before P&V | After HWA + P&V | Open-loop recovery | Closed-loop recovery |",
    "| --- | --- | ---: | ---: | ---: | ---: |"]
for s in series:
    lines.append(f"| {s['architecture']} | {s['condition']} | " +
                 " | ".join(f"{v:.6f}" for v in s["values"]) + " |")
lines += ["", "The new bars evaluate the frozen stochastic models used during HWA, before deployment. "
    "The DRN averages all 64 views of its original endpoint training bank. The crossbar averages 64 Gaussian apparent-noise draws "
    "on its original HWA population (assignment 2090401). The mean is over individual KL values, not averaged predictions. "
    "No recovery-array population, P&V controller, fault injection or device updates enter the new baseline. "
    "Healthy and corrupt branches repeat the same HWA value because faults enter after P&V.", "",
    "Whiskers show the standard deviation of full-test KL across the 64 stochastic model views. "
    "The DRN views include related rotations of eight endpoint populations, so these are not confidence intervals or 64 independent arrays. "
    "Recovery bars retain their single-array/write coverage.", "",
    "The crossbar HWA surrogate uses unconditioned apparent noise. Its predeployment KL can therefore exceed the P&V-conditioned KL. "
    "For reference, the unmodified digital crossbar HWA master has test KL "
    f"{baselines['crossbar']['unmodified_digital_master_diagnostic']['kl_teacher_student']:.6f}; "
    "this is a separate noise-free diagnostic and is not the plotted HWA hardware-view value.", "",
    "The saved predeployment HWA validation readouts are: crossbar KL "
    f"{baselines['crossbar']['original_stochastic_validation_KL']:.6f} (one original held validation draw), and DRN KL "
    f"{baselines['drn']['validation_replay']['expected_KL']:.6f} (mean over three original development fields). "
    "These validation values do not enter the test plot.", "",
    "The DRN validation readout was reproduced. The crossbar population file and support-clamped master match their original hashes, "
    "and its deterministic validation predictions and KL were reproduced. The original Akib and local CUDA noise draws differ "
    "despite the same seed, so the crossbar test result estimates the same stochastic model with fresh draws; it is not a bitwise replay "
    "of the old stochastic validation draw. This operational deviation from the initial replay check is recorded in crossbar_readout.json.", "",
    "Both healthy P0 replays reproduced the previously saved KL exactly. Teachers, normalization, batch size and fixed scaling were retained. "
    "The 12 existing recovery values, including the corrected DRN open-loop selections, are preserved exactly. "
    "No training, programming, checkpoint selection or calibration refitting was performed.", "",
    "This remains an exploratory single-array/write comparison with different teachers and device/fault models across architectures. "
    "It does not isolate architecture as the cause of absolute KL differences. Crossbar recovery uses a 30-epoch budget; "
    "DRN open-loop uses 30 epochs and DRN closed-loop 3, with the original validation-based selections.", "",
    "[Figure](open_closed_recovery.png) · [PDF](open_closed_recovery.pdf) · [Values](kl_values.csv) · "
    "[Readout protocol](readout_protocol.md) · [Provenance](provenance.json)", ""]
(OUT / "report.md").write_text("\n".join(lines))
for ext in ("png", "pdf", "svg"):
    shutil.copy2(OUT / f"open_closed_recovery.{ext}", ROOT.parent / f"recovery_kl_with_hwa.{ext}")
print(paths[0])
