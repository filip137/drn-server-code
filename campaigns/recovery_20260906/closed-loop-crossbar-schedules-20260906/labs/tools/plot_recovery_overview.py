"""Display completed DRN/crossbar recovery results, including the healthy control."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT.parent / "closed-loop-recovery-20260906"
OUT = ROOT / "analysis" / "updated_overview"
OUT.mkdir(parents=True, exist_ok=True)

panels = [
    ("DRN", "constant", 3, "DRN · constant rate\n3 recovery epochs"),
    ("Crossbar", "constant", 30, "Crossbar · constant rate\n30 recovery epochs"),
    ("Crossbar", "exponential", 30, "Crossbar · learning-rate decay\n30 recovery epochs"),
]
rows = []
sources = {}
for architecture, schedule, epochs, _ in panels:
    for condition in ("healthy", "faulted"):
        pair = []
        for writer in ("open_loop", "closed_loop_pv"):
            if architecture == "DRN":
                path = ORIGINAL / "runs" / f"drn-{condition}-{writer}" / "result.json"
            elif (condition, schedule, writer) == ("healthy", "constant", "closed_loop_pv"):
                path = ROOT / "healthy_lr1e5_control" / "run" / "result.json"
            else:
                path = ROOT / "runs" / f"{condition}-{schedule}-{writer}" / "result.json"
            raw = path.read_bytes()
            result = json.loads(raw)
            assert result["status"] == "complete" and result["selected_replay_passed"]
            assert len(result["epochs"]) == epochs
            assert result["selected_test"]["apparent"]["examples"] == 10000
            sources[str(path)] = hashlib.sha256(raw).hexdigest()
            pair.append(result)
        op, cl = pair
        assert op["manifest"]["initial_state"] == cl["manifest"]["initial_state"]
        assert op["initial_test"] == cl["initial_test"]
        assert op["manifest"]["inputs"] == cl["manifest"]["inputs"]
        assert op["manifest"]["seeds"] == cl["manifest"]["seeds"]
        assert [e["data_stream_sha256"] for e in op["epochs"]] == [
            e["data_stream_sha256"] for e in cl["epochs"]
        ]
        row = dict(
            architecture=architecture,
            condition="Healthy" if condition == "healthy" else "Corrupt",
            schedule=schedule,
            recovery_epochs=epochs,
            initial_test_KL=op["initial_test"]["apparent"]["kl_teacher_student"],
        )
        for prefix, r in (("open_loop", op), ("closed_loop", cl)):
            row[f"{prefix}_learning_rate"] = r["manifest"]["learning_rate"]
            row[f"{prefix}_selected_epoch"] = r["selected"]["epoch"]
            row[f"{prefix}_test_KL"] = r["selected_test"]["apparent"]["kl_teacher_student"]
            row[f"{prefix}_final_test_KL"] = r["final_test"]["apparent"]["kl_teacher_student"]
        rows.append(row)

with (OUT / "comparison.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(OUT / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.4), sharey=True)
fig.subplots_adjust(left=.065, right=.99, top=.725, bottom=.245, wspace=.16)
series = [
    ("initial_test_KL", "HWA + P&V starting state", "#a4acb5"),
    ("open_loop_test_KL", "Open-loop recovery", "#277da8"),
    ("closed_loop_test_KL", "Closed-loop P&V recovery", "#c16835"),
]
for index, (ax, panel) in enumerate(zip(axes, panels)):
    architecture, schedule, epochs, title = panel
    selected = rows[index * 2:index * 2 + 2]
    x = np.arange(2)
    for offset, (key, label, color) in zip((-.26, 0, .26), series):
        values = [r[key] for r in selected]
        bars = ax.bar(x + offset, values, width=.24, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.6f}" for v in values], padding=4, fontsize=8)
    ax.set_xticks(x, [r["condition"] for r in selected])
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_ylim(0, .082)
    ax.set_yticks(np.arange(0, .081, .02))
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Test KL(teacher || student) ↓")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .89),
           ncol=3, frameon=False, fontsize=11)
fig.suptitle("Open-loop vs closed-loop recovery", fontsize=18, y=.98)
fig.text(.5, .92, "Updated healthy crossbar control · Checkpoints selected using validation KL, including epoch 0",
         ha="center", fontsize=11)
fig.text(.5, .135,
         "Healthy crossbar, constant rate: open loop LR = 3e-6; closed loop LR = 1e-5. Corrupt: both LR = 1e-5.",
         ha="center", fontsize=10)
fig.text(.5, .04,
         "DRN: both LR = 1e-4. Crossbar decay: healthy starts at 3e-6; corrupt at 1e-5; ×0.01 by epoch 10.\n"
         "Held apparent state · One array/write · DRN and crossbar use different teachers, device models and training budgets",
         ha="center", fontsize=9)
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"open_closed_recovery.{ext}", dpi=200)
plt.close(fig)

lines = [
    "# Updated open-loop and closed-loop recovery", "",
    "Selected test KL(teacher || student), lower is better. Checkpoints were selected using validation KL with the starting state eligible. All test readouts contain 10,000 examples and use the held apparent state.", "",
    "![Open-loop and closed-loop recovery](open_closed_recovery.png)", "",
    "| Architecture | Devices | Schedule | Epoch budget | HWA + P&V start | Open loop | Closed-loop P&V | LR open / closed | Selected epoch open / closed |",
    "|---|---|---|---:|---:|---:|---:|---|---|",
]
for r in rows:
    lines.append(
        f"| {r['architecture']} | {r['condition']} | {r['schedule']} | {r['recovery_epochs']} | "
        f"{r['initial_test_KL']:.6f} | {r['open_loop_test_KL']:.6f} | {r['closed_loop_test_KL']:.6f} | "
        f"{r['open_loop_learning_rate']:.0e} / {r['closed_loop_learning_rate']:.0e} | "
        f"{r['open_loop_selected_epoch']} / {r['closed_loop_selected_epoch']} |"
    )
lines += [
    "",
    "The updated healthy constant-rate crossbar row combines the completed open-loop LR=3e-6 arm with the new closed-loop LR=1e-5 control. This row measures their achieved recovery under different rates. The paired LR=3e-6 closed-loop result remains 0.023479, versus 0.024159 open loop. There is no 30-epoch open-loop LR=1e-5 result in this display.",
    "",
    "Both crossbar decay P&V arms issued zero device pulses and retained the starting checkpoint. Decay reaches 1% of the initial rate at epoch 10 and holds through epoch 30. Fixed-final test KL and checkpoint epochs are preserved in the CSV so selected recovery is distinguishable from the endpoint of training.",
    "",
    "Each displayed pair has identical starting tensors, data order, source inputs and seeds. The original three-epoch DRN evidence and new thirty-epoch crossbar evidence use different teachers and device/fault models; their numerical KL values do not isolate architecture effects. This display uses one array/write per condition, without uncertainty estimates.",
    "",
    "[Full metrics](comparison.csv) · [PDF](open_closed_recovery.pdf) · [SVG](open_closed_recovery.svg) · [Sources and checks](verification.json)",
]
(OUT / "report.md").write_text("\n".join(lines) + "\n")
(OUT / "verification.json").write_text(json.dumps({
    "status": "complete",
    "source_sha256": sources,
    "completed_runs": len(sources),
    "paired_starting_states": True,
    "paired_data_streams": True,
    "selected_replay_passed": True,
    "healthy_crossbar_constant_rates_differ": True,
    "new_training_launched": False,
}, indent=2) + "\n")
print(OUT / "open_closed_recovery.png")
