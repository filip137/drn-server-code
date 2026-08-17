"""Summarize and plot RESET-trained KL and cross-entropy MNIST DRNs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _complete(path: Path, *, kind: str) -> dict[str, Any]:
    result = _json(path)
    if result.get("status") != "complete" or result.get("error") is not None:
        raise ValueError(
            f"Expected {kind} result to be semantically complete. "
            f"Provided value: path={str(path)!r}, "
            f"status={result.get('status')!r}, error={result.get('error')!r}."
        )
    return result


def _run(run_dir: Path, *, expected_objective: str) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not result_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            "Expected each run directory to contain result.json and "
            f"metrics.jsonl. Provided value: {str(run_dir)!r}."
        )
    result = _complete(result_path, kind="training")
    metrics = result["metrics"]
    if metrics.get("objective") != expected_objective:
        raise ValueError(
            "Expected training result objective to match its comparison arm. "
            f"Provided value: expected={expected_objective!r}, "
            f"objective={metrics.get('objective')!r}."
        )
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    initialization = [row for row in rows if row.get("mode") == "initialization"]
    training = sorted(
        (row for row in rows if row.get("mode") == "train"),
        key=lambda row: int(row["completed_epochs"]),
    )
    if len(initialization) != 1 or not training:
        raise ValueError(
            "Expected exactly one initialization metric and at least one "
            f"training metric. Provided value: initialization={len(initialization)}, "
            f"training={len(training)}."
        )
    return {"result": result, "initialization": initialization[0], "training": training}


def _pulse_changes(programming: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(value["pulse_changed_fraction"])
        for key, value in sorted(programming["parameters"].items())
    }


def compare(
    *,
    kl_run: Path,
    cross_entropy_run: Path,
    kl_test_result: Path,
    cross_entropy_test_result: Path,
    output_dir: Path,
) -> dict[str, Any]:
    runs = {
        "teacher_kl": _run(kl_run, expected_objective="teacher_kl"),
        "cross_entropy": _run(
            cross_entropy_run, expected_objective="cross_entropy"
        ),
    }
    tests = {
        "teacher_kl": _complete(kl_test_result, kind="test")["metrics"],
        "cross_entropy": _complete(
            cross_entropy_test_result, kind="test"
        )["metrics"],
    }
    arms = {}
    for label, run in runs.items():
        metrics = run["result"]["metrics"]
        arms[label] = {
            "run_dir": str(
                kl_run if label == "teacher_kl" else cross_entropy_run
            ),
            "initial": metrics["initial_validation"],
            "selected": metrics["selected"],
            "last": metrics["last_validation"],
            "selected_learning_rates": metrics["selected_learning_rates"],
            "pulse_changed_fraction_by_parameter": _pulse_changes(
                metrics["measured_programming"]
            ),
            "assignment_sha256_by_parameter": metrics["measured_programming"][
                "assignment_sha256_by_parameter"
            ],
            "teacher_sha256": metrics["teacher"]["sha256"],
            "posthoc_validation_calibration": metrics[
                "posthoc_validation_calibration"
            ],
            "test": tests[label],
        }

    same_assignments = (
        arms["teacher_kl"]["assignment_sha256_by_parameter"]
        == arms["cross_entropy"]["assignment_sha256_by_parameter"]
    )
    same_teacher = (
        arms["teacher_kl"]["teacher_sha256"]
        == arms["cross_entropy"]["teacher_sha256"]
    )
    kl_accuracy = float(tests["teacher_kl"]["student_accuracy"])
    ce_accuracy = float(tests["cross_entropy"]["student_accuracy"])
    nonzero_writes = all(
        value > 0.0
        for value in arms["teacher_kl"][
            "pulse_changed_fraction_by_parameter"
        ].values()
    )
    criteria = {
        "kl_test_accuracy_at_least_94pct": kl_accuracy >= 0.94,
        "kl_within_1pp_of_cross_entropy": kl_accuracy >= ce_accuracy - 0.01,
        "same_device_assignments": same_assignments,
        "same_teacher": same_teacher,
        "kl_updates_both_layers": nonzero_writes,
    }
    summary = {
        "schema": "mnist-relu-drn-reset-comparison/v1",
        "arms": arms,
        "comparison": {
            "kl_minus_cross_entropy_test_accuracy": kl_accuracy - ce_accuracy,
            "criteria": criteria,
            "hypothesis_supported": all(criteria.values()),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for label, run in runs.items():
        rows = run["training"]
        epochs = [0] + [int(row["completed_epochs"]) for row in rows]
        validation = [run["initialization"]["validation"]] + [
            row["validation"] for row in rows
        ]
        axes[0].plot(
            epochs,
            [float(item["kl_teacher_student"]) for item in validation],
            marker="o",
            markersize=3,
            label=label,
        )
        axes[1].plot(
            epochs,
            [100.0 * float(item["student_accuracy"]) for item in validation],
            marker="o",
            markersize=3,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set(
        xlabel="Completed epochs",
        ylabel="Validation KL(teacher || DRN)",
    )
    axes[1].axhline(94.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set(xlabel="Completed epochs", ylabel="Validation accuracy (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output_dir / "validation_comparison.png", dpi=180)
    plt.close(figure)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kl-run", type=Path, required=True)
    parser.add_argument("--cross-entropy-run", type=Path, required=True)
    parser.add_argument("--kl-test-result", type=Path, required=True)
    parser.add_argument("--cross-entropy-test-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    compare(
        kl_run=args.kl_run.expanduser().resolve(),
        cross_entropy_run=args.cross_entropy_run.expanduser().resolve(),
        kl_test_result=args.kl_test_result.expanduser().resolve(),
        cross_entropy_test_result=(
            args.cross_entropy_test_result.expanduser().resolve()
        ),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
