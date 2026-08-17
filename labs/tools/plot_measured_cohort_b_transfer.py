"""Summarize cohort-A checkpoint deployment and fine-tuning on cohort B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _complete_result(run_dir: Path) -> dict[str, Any]:
    result = _json(run_dir / "result.json")
    if result.get("status") != "complete" or result.get("error") is not None:
        raise ValueError(
            "Expected a semantically complete training result. Provided "
            f"value: run={str(run_dir)!r}, status={result.get('status')!r}, "
            f"error={result.get('error')!r}."
        )
    return result


def _training_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            "Expected the cohort-B run directory to contain metrics.jsonl. "
            f"Provided value: {str(run_dir)!r}."
        )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    training = [row for row in rows if row.get("mode") == "train"]
    if not training:
        raise ValueError(
            "Expected metrics.jsonl to contain production train rows. "
            f"Provided value: {str(path)!r}."
        )
    return sorted(training, key=lambda row: int(row["completed_epochs"]))


def _test_metrics(path: Path) -> dict[str, Any]:
    result = _json(path)
    metrics = dict(result.get("metrics", {}))
    if (
        result.get("status") != "complete"
        or metrics.get("split_protocol", {}).get("effective")
        != "held_out_test"
        or int(metrics.get("examples", 0)) != 10000
    ):
        raise ValueError(
            "Expected a complete 10,000-example held_out_test result. "
            f"Provided value: path={str(path)!r}, result={result!r}."
        )
    return metrics


def summarize_transfer(
    source_run_dir: Path,
    cohort_b_run_dir: Path,
    *,
    source_test_result: Path,
    cohort_b_test_result: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_result = _complete_result(source_run_dir)
    cohort_b_result = _complete_result(cohort_b_run_dir)
    source_metrics = source_result["metrics"]
    cohort_b_metrics = cohort_b_result["metrics"]
    rows = _training_rows(cohort_b_run_dir)
    accuracies = [float(row["validation"]["accuracy"]) for row in rows]
    costs = [float(row["validation"]["mean_cost"]) for row in rows]
    source_test = _test_metrics(source_test_result)
    cohort_b_test = _test_metrics(cohort_b_test_result)
    pre = cohort_b_metrics["pre_deployment_validation"]
    post = cohort_b_metrics["initial_validation"]
    selected = cohort_b_metrics["selected"]
    programming = cohort_b_metrics["device_programming"]
    update_programming = cohort_b_metrics["update_programming"]
    active_cohort = update_programming.get("active_cohort")
    cohort_b_used = bool(
        programming.get("cohort_b_used")
        and update_programming.get("cohort_b_used")
        and active_cohort == "B"
    )
    gates = {
        "cohort_b_completed_without_failure": True,
        "only_cohort_b_active_during_transfer": cohort_b_used,
        "post_deployment_validation_at_least_85pct": (
            float(post["accuracy"]) >= 0.85
        ),
        "selected_finetuned_validation_at_least_90pct": (
            float(selected["accuracy"]) >= 0.90
        ),
        "cohort_b_test_within_5_percentage_points_of_source": (
            float(cohort_b_test["accuracy"])
            >= float(source_test["accuracy"]) - 0.05
        ),
        "cohort_b_late_drop_below_5_percentage_points": (
            max(accuracies) - accuracies[-1] < 0.05
        ),
    }
    deployment_projection = {
        key: value.get("initial_write")
        for key, value in programming["initialization"]["parameters"].items()
    }
    selection = cohort_b_metrics["learning_rate_selection"]
    summary = {
        "schema": "measured-cohort-b-transfer/v1",
        "source_cohort_a": {
            "run_dir": str(source_run_dir),
            "selected_validation_accuracy": float(
                source_metrics["selected"]["accuracy"]
            ),
            "official_test": source_test,
        },
        "cohort_b": {
            "run_dir": str(cohort_b_run_dir),
            "status": cohort_b_result["status"],
            "active_cohort": active_cohort,
            "cohort_b_used": cohort_b_used,
            "pre_deployment_validation": pre,
            "post_deployment_validation": post,
            "deployment_accuracy_delta": (
                float(post["accuracy"]) - float(pre["accuracy"])
            ),
            "selected_epoch": int(selected["epoch"]),
            "selected_validation_accuracy": float(selected["accuracy"]),
            "selected_validation_mean_cost": float(selected["value"]),
            "last_validation_accuracy": accuracies[-1],
            "peak_validation_accuracy": max(accuracies),
            "late_accuracy_drop_from_peak": max(accuracies) - accuracies[-1],
            "selected_lr_cell": selection["selected_cell_id"],
            "selected_relative_update_targets": selection[
                "selected_relative_update_targets"
            ],
            "selected_learning_rates_by_parameter": selection[
                "selected_learning_rates_by_parameter"
            ],
            "lr_grid_edge_selected": bool(selection["edge_selected"]),
            "deployment_projection_by_parameter": deployment_projection,
            "finetuning_projection_by_parameter": update_programming[
                "parameters"
            ],
            "official_test": cohort_b_test,
        },
        "feasibility_gates": gates,
        "automatic_accuracy_gates_pass": all(gates.values()),
        "projection_diagnostics_require_interpretation": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    epochs = [int(row["completed_epochs"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(
        epochs,
        [100.0 * accuracy for accuracy in accuracies],
        marker="o",
        markersize=3,
        label="Cohort-B fine-tuning",
    )
    axes[0].axhline(
        100.0 * float(pre["accuracy"]),
        color="tab:blue",
        linestyle="--",
        label="Loaded cohort-A checkpoint",
    )
    axes[0].scatter(
        [0],
        [100.0 * float(post["accuracy"])],
        color="tab:red",
        marker="x",
        s=55,
        label="After cohort-B deployment",
    )
    axes[0].axhline(90.0, color="0.4", linestyle=":", linewidth=1)
    axes[0].set(xlabel="Fine-tuning epoch", ylabel="Validation accuracy (%)")
    axes[1].plot(
        epochs,
        costs,
        marker="o",
        markersize=3,
        label="Cohort-B fine-tuning",
    )
    axes[1].axhline(
        float(pre["mean_cost"]),
        color="tab:blue",
        linestyle="--",
        label="Loaded cohort-A checkpoint",
    )
    axes[1].scatter(
        [0],
        [float(post["mean_cost"])],
        color="tab:red",
        marker="x",
        s=55,
        label="After cohort-B deployment",
    )
    axes[1].set(xlabel="Fine-tuning epoch", ylabel="Validation mean cost")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output_dir / "cohort_b_transfer.png", dpi=180)
    plt.close(figure)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--cohort-b-run-dir", type=Path, required=True)
    parser.add_argument("--source-test-result", type=Path, required=True)
    parser.add_argument("--cohort-b-test-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summarize_transfer(
        args.source_run_dir.expanduser().resolve(),
        args.cohort_b_run_dir.expanduser().resolve(),
        source_test_result=args.source_test_result.expanduser().resolve(),
        cohort_b_test_result=args.cohort_b_test_result.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
