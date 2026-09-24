"""Summarize the requested ten-epoch ours p99 T16 run against saved T8."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.reporting import validate_run


def load_run(directory: Path, epochs: int):
    issues = validate_run(directory)
    assert not issues, (directory, issues)
    assert json.loads((directory / "status.json").read_text())["state"] == "complete"
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row.get("kind") == "epoch"]
    assert [row["epoch"] for row in rows] == list(range(1, epochs + 1))
    manifest = json.loads((directory / "manifest.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["official_test_evaluations"] == 0
    assert metrics["runtime_dtype"] == "float64"
    with np.load(directory / "weights_final.npz") as weights:
        parameter_keys = [key for key in weights.files if key.startswith(("ConvWeight_", "DenseWeight_", "Bias_"))]
        assert len(parameter_keys) == 7
        assert all(np.isfinite(weights[key]).all() for key in parameter_keys)
        bias_keys = [key for key in weights.files if "Bias" in key]
        assert bias_keys and all((weights[key] == 0).all() for key in bias_keys)
    return rows, manifest, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    new_dirs = list((root / "production" / "runs").glob("000_*"))
    old_dirs = list((root / "reference-T8").glob("000_*"))
    assert len(new_dirs) == len(old_dirs) == 1
    new, new_manifest, new_metrics = load_run(new_dirs[0], 10)
    old, old_manifest, old_metrics = load_run(old_dirs[0], 30)
    assert new_manifest["git"]["source_archive_sha256"] == old_manifest["git"]["source_archive_sha256"]
    reference_environment = json.loads((root / "reference-T8-environment.json").read_text())
    for chunk in (0, 1):
        environment = json.loads((root / "production" / "segments" / f"chunk_{chunk}" / "environment.json").read_text())
        assert environment["gpu"] == reference_environment["name"]
        assert all(environment[key] == reference_environment[key] for key in ("torch", "cuda"))
    assert new_metrics["initial_parameter_state_sha256"] == old_metrics["initial_parameter_state_sha256"]
    for key in ("train_indices_sha256", "validation_indices_sha256", "first_epoch_batch_order_sha256"):
        assert new_metrics["dataset_provenance"][key] == old_metrics["dataset_provenance"][key]
    assert new_metrics["dataset_provenance"]["train_batch_order_sha256"] == old_metrics["dataset_provenance"]["train_batch_order_sha256"][:10]
    assert new_metrics["eqprop_endpoint_read_noise_draw_count"] * 3 == old_metrics["eqprop_endpoint_read_noise_draw_count"]
    for metadata in ("arm_id", "reporting", "qualification_source", "study_id", "stability_pilot"):
        new_manifest["configuration"]["resolved"].pop(metadata, None)
        old_manifest["configuration"]["resolved"].pop(metadata, None)
    old_config = old_manifest["configuration"]["resolved"]
    new_config = new_manifest["configuration"]["resolved"]
    assert old_config["lab"]["epochs"] == 30 and new_config["lab"]["epochs"] == 10
    assert old_config["model_base"]["num_iterations_inference"] == 8
    assert new_config["model_base"]["num_iterations_inference"] == 16
    old_config["lab"]["epochs"] = 10
    old_config["model_base"]["num_iterations_inference"] = 16
    assert old_config == new_config, "Unaccounted scientific config difference"
    assert new_config["model_base"]["num_iterations_training"] == 8
    assert new_config["eqprop"]["injected_beta_B"] == 0.987333678708

    output = root / "analysis"
    output.mkdir(exist_ok=True)
    table, summaries = [], []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for T, rows, bundle, color in [(8, old[:10], old_dirs[0], "#808080"), (16, new, new_dirs[0], "#1764ab")]:
        values = [row["metrics"] for row in rows]
        accuracy = [100 * row["validation_accuracy"] for row in values]
        losses = [row["validation_loss"] for row in values]
        assert np.isfinite(accuracy + losses).all()
        best_index = int(np.argmax(accuracy))
        summary = dict(T=T, K=8, epochs=10, final_validation_percent=accuracy[-1],
                       best_validation_percent=accuracy[best_index], best_epoch=best_index + 1,
                       final_drop_pp=max(accuracy) - accuracy[-1], finite=True,
                       drop_screen_passed=max(accuracy) - accuracy[-1] < 5,
                       source_bundle=str(bundle.relative_to(root)),
                       result_sha256=hashlib.sha256((bundle / "result.json").read_bytes()).hexdigest())
        summaries.append(summary)
        for epoch, row in enumerate(values, 1):
            table.append(dict(T=T, K=8, epoch=epoch, **{key: row[key] for key in
                ("train_accuracy", "train_loss", "validation_accuracy", "validation_loss")}))
        axes[0].plot(range(1, 11), accuracy, "o-", color=color, label=f"T = {T}, K = 8")
        axes[1].plot(range(1, 11), losses, "o-", color=color, label=f"T = {T}, K = 8")
    for axis, ylabel in zip(axes, ("Validation accuracy (%)", "Validation loss")):
        axis.set(xlabel="Epoch", ylabel=ylabel, xticks=range(1, 11))
        axis.grid(alpha=.2)
        axis.legend(frameon=False)
    fig.suptitle("Conv3 ours · p.99 β = 0.98733 · read noise σ = 5×10⁻⁴ · seed 0")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"validation_T8_vs_T16.{extension}", dpi=180)
    plt.close(fig)
    with (output / "epochs.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    result = dict(status="complete", included=summaries,
                  final_validation_difference_pp=summaries[1]["final_validation_percent"] - summaries[0]["final_validation_percent"],
                  maximum_absolute_epoch_validation_difference_pp=max(
                      100 * abs(a["metrics"]["validation_accuracy"] - b["metrics"]["validation_accuracy"])
                      for a, b in zip(new, old[:10])),
                  matched_initialization=True, matched_split=True, matched_ten_epoch_order=True,
                  expected_noise_draw_count=True, official_test_read=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Conv3 ours p.99: ten epochs at T16/K8", "",
             "The requested T16/K8 run completed ten finite epochs. Injected beta is",
             "0.987333678708, endpoint read noise sigma5e-4, and optimizer Adam with",
             "unchanged learning rates, saved seed0 initialization and exact-zero biases.", "",
             "| Free-phase T | Best validation, epochs1–10 | Epoch10 validation | Best epoch |",
             "|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['T']} | {row['best_validation_percent']:.2f}% | {row['final_validation_percent']:.2f}% | {row['best_epoch']} |")
    lines += ["", f"Epoch10 difference: {result['final_validation_difference_pp']:+.2f} percentage points.",
              f"The largest absolute epochwise accuracy difference is {result['maximum_absolute_epoch_validation_difference_pp']:.2f}pp.",
              "The trajectories are very similar; increasing T did not improve validation accuracy in this run.", "",
              "![Matched ten-epoch trajectories](validation_T8_vs_T16.png)", "",
              "T8 is the first ten epochs of the completed30-epoch p99 run. Both use",
              "the same frozen runtime on Jean Zay V100s; source configs differ",
              "scientifically only in T and terminal horizon. Initialization, split",
              "and all ten minibatch-order hashes match. The T change applies to",
              "both training free phases and validation inference. No official test is read.", "",
              "The T16 run used two five-epoch checkpoint segments, preserving optimizer",
              "and RNG/noise state. This is one-seed, ten-epoch exploratory evidence.",
              "It does not establish thirty-epoch stability or a large-beta stability boundary.", "",
              "[Measurements](epochs.csv) · [Summary and source hashes](summary.json)"]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
