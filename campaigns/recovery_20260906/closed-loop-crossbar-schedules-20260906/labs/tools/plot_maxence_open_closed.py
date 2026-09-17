"""Extend the existing Maxence figure with the best current recovery cases.

The historical panels retain their original cohorts and metrics. The added
panels select among the completed paired recovery arms using validation KL.
"""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT.parent / "closed-loop-recovery-20260906"
HISTORICAL = ROOT.parent / "reviews/ibm_om_2026-09-04_05/message_to_maxence_20260906"
OUT = ROOT / "analysis/notion_maxence"
OUT.mkdir(parents=True, exist_ok=True)

assert hashlib.sha256((OUT / "original_figure.png").read_bytes()).hexdigest() == hashlib.sha256(
    (HISTORICAL / "kl_before_after_onchip_training.png").read_bytes()).hexdigest()
with (HISTORICAL / "kl_plot_values.csv").open() as f:
    historical = list(csv.DictReader(f))
sources = {str(HISTORICAL / "kl_plot_values.csv"): hashlib.sha256(
    (HISTORICAL / "kl_plot_values.csv").read_bytes()).hexdigest()}
selected = {}
candidate_audit = []
rows = []

def rank(r):
    m = r["selected"]["validation"]["apparent"]
    return (m["kl_teacher_student"], -m["student_accuracy"], r["selected"]["epoch"])

for architecture in ("crossbar", "drn"):
    for condition in ("healthy", "faulted"):
        for writer in ("open_loop", "closed_loop_pv"):
            if architecture == "drn":
                paths = [ORIGINAL / "runs" / f"drn-{condition}-{writer}" / "result.json"]
            else:
                paths = [ROOT / "runs" / f"{condition}-{schedule}-{writer}" / "result.json"
                         for schedule in ("constant", "exponential")]
                if condition == "healthy" and writer == "closed_loop_pv":
                    paths.append(ROOT / "healthy_lr1e5_control/run/result.json")
            candidates = []
            for path in paths:
                raw = path.read_bytes()
                result = json.loads(raw)
                assert result["status"] == "complete" and result["selected_replay_passed"]
                sources[str(path)] = hashlib.sha256(raw).hexdigest()
                candidates.append((path, result))
            path, result = min(candidates, key=lambda x: rank(x[1]))
            selected[architecture, condition, writer] = result
            for candidate_path, candidate in candidates:
                candidate_audit.append(dict(
                    architecture=architecture, condition=condition, writer=writer,
                    source=str(candidate_path), validation_KL=rank(candidate)[0],
                    selected=candidate_path == path))
            rows.append(dict(
                architecture=architecture, condition=condition, writer=writer,
                learning_rate=result["manifest"]["learning_rate"],
                schedule=result["manifest"].get("schedule_name", "constant"),
                epoch_budget=len(result["epochs"]), selected_epoch=result["selected"]["epoch"],
                initial_test_KL=result["initial_test"]["apparent"]["kl_teacher_student"],
                selected_validation_KL=rank(result)[0],
                selected_test_KL=result["selected_test"]["apparent"]["kl_teacher_student"],
                source=str(path)))
        a, b = [selected[architecture, condition, w] for w in ("open_loop", "closed_loop_pv")]
        for key in ("initial_state", "inputs", "seeds"):
            assert a["manifest"][key] == b["manifest"][key]
        assert a["initial_test"] == b["initial_test"]
        assert [e["data_stream_sha256"] for e in a["epochs"]] == [
            e["data_stream_sha256"] for e in b["epochs"]]
        assert a["selected_test"]["apparent"]["examples"] == b["selected_test"]["apparent"]["examples"] == 10000

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
colors = {"healthy": "#26799D", "faulted": "#CE6432"}
labels = {"healthy": "Without corrupt devices", "faulted": "With corrupt devices"}
fig = plt.figure(figsize=(14.4, 11.3))
fig.suptitle("KL divergence before and after on-chip training", x=.075, y=.98,
             ha="left", fontsize=22, weight="bold")
fig.text(.075, .941, "Open-loop and closed-loop recovery each start from the corresponding HWA + P&V state.",
         fontsize=12)
fig.legend(handles=[Patch(facecolor=colors[c], label=labels[c]) for c in colors],
           loc="upper left", bbox_to_anchor=(.065, .921), ncol=2, frameon=False, fontsize=12)
fig.text(.075, .866, "Previous one-epoch open-loop results", fontsize=13, weight="bold")
fig.text(.075, .526, "Best cases in the current open-loop / closed-loop comparison", fontsize=13, weight="bold")

def format_panel(ax, title, subtitle, stages, left):
    ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=28)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=10, color="#444444")
    ax.set_xticks(range(len(stages)), stages)
    ax.set_xlim(-.55, len(stages) - .45)
    ax.set_ylim(0, .082)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(MultipleLocator(.02))
    if left:
        ax.set_ylabel("KL(teacher || student)")
    else:
        ax.tick_params(labelleft=False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E8EC")
    ax.spines[["top", "right"]].set_visible(False)

for col, architecture in enumerate(("crossbar", "drn")):
    left = .075 if col == 0 else .56
    ax = fig.add_axes([left, .601, .415, .203])
    campaign = "crossbar_retuned" if architecture == "crossbar" else "drn"
    for offset, condition in ((-.185, "healthy"), (.185, "faulted")):
        values = [next(r for r in historical if (r["campaign"], r["condition"], r["stage"]) ==
                       (campaign, labels[condition], stage)) for stage in ("HWA + P&V", "After Adam")]
        xs = [i + offset for i in range(2)]
        means = [float(r["mean"]) for r in values]
        ax.bar(xs, means, width=.31, color=colors[condition], zorder=3)
        if architecture == "drn":
            ax.errorbar(xs, means,
                        yerr=[[float(r["mean"]) - float(r["minimum"]) for r in values],
                              [float(r["maximum"]) - float(r["mean"]) for r in values]],
                        fmt="none", color="#333333", capsize=3.5, zorder=4)
        for x, r in zip(xs, values):
            top = float(r["maximum"] or r["mean"])
            ax.annotate(f"{float(r['mean']):.5f}", (x, top), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=10)
    format_panel(ax, "a) Crossbar" if col == 0 else "b) DRN",
                 "Validation · 1 development array" if col == 0 else "Test · 3 arrays × 3 writes; range of array means",
                 ["HWA + P&V", "After 1 epoch\nof open-loop recovery"], col == 0)

    ax = fig.add_axes([left, .213, .415, .237])
    for offset, condition in ((-.185, "healthy"), (.185, "faulted")):
        op, cl = [selected[architecture, condition, w] for w in ("open_loop", "closed_loop_pv")]
        means = [op["initial_test"]["apparent"]["kl_teacher_student"],
                 op["selected_test"]["apparent"]["kl_teacher_student"],
                 cl["selected_test"]["apparent"]["kl_teacher_student"]]
        xs = [i + offset for i in range(3)]
        bars = ax.bar(xs, means, width=.31, color=colors[condition], zorder=3)
        ax.bar_label(bars, labels=[f"{v:.6f}" for v in means], padding=5, fontsize=10)
    format_panel(ax, "c) Crossbar" if col == 0 else "d) DRN",
                 "Test · 1 matched development array · 30 epochs" if col == 0 else "Test · 1 matched development array · 3 epochs",
                 ["HWA + P&V", "Open-loop\nrecovery", "Closed-loop P&V\nrecovery"], col == 0)

fig.text(.075, .126,
         "New comparison: schedules and checkpoints selected by validation KL, including the starting state; test uses 10,000 MNIST images.",
         fontsize=10, color="#444444")
fig.text(.075, .088,
         "Crossbar open loop: decay from LR 3e-6 (healthy) / 1e-5 (corrupt); P&V: constant LR 1e-5 for both. DRN: constant LR 1e-4 for both writers.",
         fontsize=10, color="#444444")
fig.text(.075, .05,
         "Held apparent state throughout. The two architectures use different teachers and device/fault models; absolute KL does not isolate architecture effects.",
         fontsize=10, color="#444444")
figure_paths = []
for ext in ("png", "svg", "pdf"):
    path = OUT / f"kl_open_closed_best_cases.{ext}"
    fig.savefig(path, dpi=220, facecolor="white")
    figure_paths.append(path)
plt.close(fig)

with (OUT / "selected_cases.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(OUT / "selection_provenance.json").write_text(json.dumps({
    "status": "complete", "historical_figure_bytes_matched": True,
    "historical_panels_preserved": True,
    "selection_rule": "Minimum selected validation KL within each architecture, condition and writer in the current paired comparison; ties use higher validation accuracy then earlier checkpoint.",
    "selection_used_test": False,
    "selection_scope": "Crossbar: eight schedule arms plus healthy LR=1e-5 P&V control; DRN: four original paired three-epoch arms.",
    "paired_starting_states_and_data_streams": True,
    "current_test_examples": 10000,
    "candidates": candidate_audit, "selected": rows, "source_sha256": sources,
    "figure_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in figure_paths},
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}, indent=2) + "\n")
print(json.dumps(rows, indent=2))
