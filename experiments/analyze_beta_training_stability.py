"""Summarize the ten-epoch larger-beta pilots, retaining scientific failures."""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.reporting import validate_run


def plot_progress(rows: list[dict], repo: Path, report: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#444444", "ours": "#0072B2", "legacy": "#D55E00"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for col, arch in enumerate(("conv1", "conv2", "conv3")):
        axes[0, col].set_title(arch.capitalize())
        axes[0, col].set_yscale("log")
        axes[0, col].set_xlabel("Training progress (epochs)")
        axes[1, col].set_xlabel("Completed epoch")
        axes[1, col].set_xlim(.8, 10.2)
        for row in [r for r in rows if r["architecture"] == arch]:
            scheme, color = row["scheme"], colors[row["scheme"]]
            label = f"{scheme}, beta={row['injected_beta']:g}"
            if "bundle" in row:
                trace_path = repo / row["bundle"] / "gradient_trace.jsonl"
                points = {}
                if trace_path.exists():
                    for line in trace_path.read_text().splitlines():
                        t = json.loads(line)
                        if t["parameter_name"].startswith("Bias_"):
                            continue
                        x = t["epoch"]-1 + t["batch"]/t["total_batches"]
                        points.setdefault(x, []).append(t["gradient_l2"])
                if points:
                    xs = sorted(points)
                    axes[0, col].plot(xs, [math.hypot(*points[x]) for x in xs],
                                      ".-", color=color, label=label)
                epochs = row.get("epoch_metric_records", [])
                epochs = [r for r in epochs if r.get("kind") == "epoch"]
                if epochs:
                    axes[1, col].plot([r["epoch"] for r in epochs],
                                      [100*r["metrics"]["validation_accuracy"] for r in epochs],
                                      "o-", markersize=3, color=color, label=label)
            if row.get("failure_epoch"):
                x = row["failure_epoch"]-1 + row["failure_batch"]/3438
                axes[0, col].axvline(x, color=color, linestyle=":", alpha=.7)
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)
            else:
                failed_arch = all(r["state"] == "failed" for r in rows if r["architecture"] == arch)
                empty_label = "Failed before validation" if failed_arch else "No measurements yet"
                ax.text(.5, .5, empty_label, transform=ax.transAxes,
                        ha="center", va="center", color="#666666")
    axes[0, 0].set_ylabel("Sampled whole-weight gradient L2 norm")
    axes[1, 0].set_ylabel("Validation accuracy (%)")
    fig.suptitle("Larger-beta training pilots — gradient and validation traces\nDotted vertical lines: nonfinite failure; seed 0, zero read noise", fontsize=12)
    fig.savefig(report.with_suffix(".png"), dpi=170)
    fig.savefig(report.with_suffix(".pdf"))
    plt.close(fig)


def analyze(study: Path, report: Path) -> dict:
    repo = Path(__file__).resolve().parents[1]
    rows = []
    for case in json.loads((study / "cases.json").read_text()):
        remote = study / case["target"]
        name = case["case"]
        config = json.loads((repo / case["config"]).read_text())
        paths = list((remote / "training").glob(f"000_{name}_*"))
        assert len(paths) <= 1, (name, paths)
        row = {"case": name, "architecture": config["stability_pilot"]["architecture"],
               "scheme": config["stability_pilot"]["scheme"],
               "injected_beta": case["injected_beta"], "target": case["target"], "state": "queued",
               "early_stable": None, "epochs_completed": 0}
        if paths:
            run = paths[0]
            row["bundle"] = str(run.relative_to(repo))
            status = json.loads((run / "status.json").read_text())
            row["state"] = status["state"]
            errors = validate_run(run)
            assert not errors, (run, errors)
            row["bundle_valid"] = True
            metrics_lines = (run / "metrics.jsonl").read_text().splitlines()
            row["epoch_metric_records"] = [json.loads(line) for line in metrics_lines]
            epoch_records = [r for r in row["epoch_metric_records"] if r.get("kind") == "epoch"]
            if epoch_records:
                last = epoch_records[-1]
                row["epochs_completed"] = last["epoch"]
                row["latest_validation_pct"] = 100 * last["metrics"]["validation_accuracy"]
                row["best_completed_epoch_validation_pct"] = 100 * max(
                    r["metrics"]["validation_accuracy"] for r in epoch_records)
                values = np.array([r["metrics"]["validation_accuracy"] for r in epoch_records])
                row["max_validation_drawdown_pp"] = 100 * float(
                    np.max(np.maximum.accumulate(values) - values))
            log = (remote / f"{name}.log").read_text()
            failures = re.findall(r"NonFiniteTrainingError: (.*)", log)
            if failures:
                row.update(early_stable=False, failure_kind="nonfinite_training",
                           failure=failures[-1])
                point = re.search(r"epoch=(\d+), batch=(\d+)", failures[-1])
                if point:
                    row["failure_epoch"], row["failure_batch"] = map(int, point.groups())
                    row["epochs_completed"] = row["failure_epoch"] - 1
            elif row["state"] == "failed":
                row["failure_kind"] = "unclassified_failure_requires_review"
            if row["state"] == "complete":
                metrics = json.loads((run / "metrics.json").read_text())
                assert metrics["official_test_evaluations"] == 0
                histories = {key: np.load(run / f"{key}.npy", allow_pickle=False)
                             for key in ("loss_train", "loss_test", "accuracy_train", "accuracy_test")}
                assert all(len(h) == 10 and np.isfinite(h).all() for h in histories.values())
                acc = histories["accuracy_test"]
                # Stored histories use fractions, as do canonical accuracy metrics.
                assert np.isclose(acc[-1], metrics["final_validation_accuracy"])
                best, final = float(acc.max()), float(acc[-1])
                control = repo / config["stability_pilot"]["comparison_control"]
                control_metrics = json.loads((control / "metrics.json").read_text())
                initial_match = (metrics["initial_parameter_state_sha256"] ==
                                 control_metrics["initial_parameter_state_sha256"])
                fields = ("train_indices_sha256", "validation_indices_sha256",
                          "first_epoch_batch_order_sha256")
                split_match = all(metrics["dataset_provenance"][k] ==
                                  control_metrics["dataset_provenance"][k] for k in fields)
                assert initial_match and split_match
                order = metrics["dataset_provenance"]["train_batch_order_sha256"]
                reference_order = control_metrics["dataset_provenance"]["train_batch_order_sha256"]
                assert order == reference_order[:len(order)]
                control_acc = np.load(control / "accuracy_test.npy", allow_pickle=False)[:10]
                assert len(control_acc) == 10
                row.update(epochs_completed=10, early_stable=(best-final < .05-1e-12),
                           best_validation_pct=100*best, final_validation_pct=100*final,
                           drop_from_best_pp=100*(best-final),
                           control_beta=json.loads((control/"config.used.json").read_text())["eqprop"]["injected_beta_B"],
                           control_final10_pct=100*float(control_acc[-1]),
                           final_accuracy_delta_pp=100*float(final-control_acc[-1]),
                           initializer_matched=True, split_order_matched=True)
                import torch
                for filename in ("best_model.pt", "final_model.pt"):
                    saved = torch.load(run/filename, map_location="cpu", weights_only=True)
                    for spec, tensor in zip(saved["schema"], saved["states"], strict=True):
                        assert tensor.dtype == torch.float64 and torch.isfinite(tensor).all()
                        if spec["name"].strip().startswith("Bias_"):
                            assert torch.count_nonzero(tensor) == 0
                trace = [json.loads(line) for line in (run/"gradient_trace.jsonl").read_text().splitlines()]
                row["max_sampled_gradient_l2"] = max(t["gradient_l2"] for t in trace)
                row["max_sampled_update_l2"] = max(t["applied_update_l2"] for t in trace)
        rows.append(row)
    terminal = sum(r["state"] in ("complete", "failed") for r in rows)
    failures = sum(r.get("failure_kind") == "nonfinite_training" for r in rows)
    stable = sum(r.get("early_stable") is True for r in rows)
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(), "expected": 9,
               "terminal": terminal, "nonfinite_failures": failures,
               "ten_epoch_stable": stable,
               "complete": terminal == 9, "official_test_read": False, "rows": rows}
    (study / "analysis.json").write_text(json.dumps(payload, indent=2)+"\n")
    text = ["# Larger-beta EqProp: ten-epoch stability pilots", "",
            f"Updated {payload['updated_at']}. **{terminal}/9 terminal; {stable} passed ten epochs; {failures} nonfinite training failures.**",
            "", "Seed 0; RTX5090 GPUs on Riri (two workers) and Trex (Conv3 ours only); clean endpoints; original Adam rates; T/K=4/6/8.",
            "Conv3 ours was moved before training when Trex became free. Its Trex/Riri smoke losses and gradient/update norms match exactly; the frozen config and initializer are unchanged. Host placement is recorded separately from the scientific cases.",
            "Betas were selected using whole-gradient cosine >=.99 plus norm mismatch <=.10.",
            "These are early stability diagnostics, not paper test accuracies or full 30-epoch qualifications.", "",
            "| Model | Scheme | Injected beta | Outcome | Best/final validation | Control epoch-10 |",
            "|---|---|---:|---|---|---|"]
    for row in rows:
        outcome = row["state"]
        score = control = "—"
        if row["state"] == "running":
            outcome = f"Running ({row['epochs_completed']}/10 epochs)"
            if "latest_validation_pct" in row:
                score = f"Latest {row['latest_validation_pct']:.2f}%"
        if row.get("failure_kind") == "nonfinite_training":
            outcome = f"Nonfinite: epoch {row.get('failure_epoch')}, batch {row.get('failure_batch')}"
            if "latest_validation_pct" in row:
                score = (f"Before failure: {row['best_completed_epoch_validation_pct']:.2f}/"
                         f"{row['latest_validation_pct']:.2f}%")
        elif row["state"] == "complete":
            outcome = "10-epoch stable" if row["early_stable"] else "Accuracy-collapse criterion failed"
            if row["early_stable"] and row.get("max_validation_drawdown_pp", 0) >= 5:
                outcome = "Passes final-drop rule; transient >=5pp collapse observed"
            score = f"{row['best_validation_pct']:.2f}/{row['final_validation_pct']:.2f}%"
            control = f"{row['control_final10_pct']:.2f}% (beta {row['control_beta']:g})"
        text.append(f"| {row['architecture']} | {row['scheme']} | {row['injected_beta']:g} | {outcome} | {score} | {control} |")
    if terminal == 9:
        completed = [r for r in rows if r["state"] == "complete"]
        text += ["", "All nine outcomes are locally collected and their canonical bundles validate; no scientific failures are excluded."]
        if completed:
            differences = "; ".join(
                f"{r['architecture']} {r['scheme']}: {r['final_accuracy_delta_pp']:+.2f}pp"
                for r in completed)
            text += [f"Final validation differences from the matching smaller-beta epoch-10 controls: {differences}.",
                     "The successful candidates pass the early stability criterion, but this single-seed clean comparison does not establish an accuracy improvement or longer/noisy training stability."]
    text += ["", "Stability requires ten finite epochs and a final validation drop strictly below 5pp from the run's best.",
             "Accuracy relative to the smaller-beta control is a separate question; finite poor learning is not evidence of satisfactory performance.",
             "Failed runs are retained and are not retried with changed scientific parameters.",
             "", "The observed failures show that clean gradient agreement at two fixed checkpoints does not guarantee stability along a new training trajectory.",
             "The whole-gradient criterion can mask layerwise errors; this pilot does not isolate that mechanism from finite-nudge effects during training.",
             "Conv1 ours at beta 1500 also passes the 0.95 per-matrix cosine rule with the norm gate: worst matrix cosine 0.951337 and norm mismatch 0.084590. Its epoch-1 failure also rejects this particular looser per-matrix candidate; this is not a training test of every 0.90/0.95/0.99 selection.",
             "", f"[Diagnostic figure]({report.with_suffix('.png').name}) · [PDF]({report.with_suffix('.pdf').name})",
             "Sampled gradient norms are absolute; their scales differ across schemes. Sparse samples may miss the final increase immediately before a failure.",
             "", "[Launch plan](../docs/eqprop_beta_training_stability_plan_20260918.md) · "
             "[Local evidence](../results/eqprop-beta-training-stability-20260918-v1/)", ""]
    report.write_text("\n".join(text))
    plot_progress(rows, repo, report)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.study.resolve(), args.report.resolve())
    print(json.dumps({k:v for k,v in summary.items() if k != "rows"}))
