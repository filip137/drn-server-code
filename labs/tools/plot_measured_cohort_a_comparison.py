"""Compare completed raw and isotonic cohort-A MNIST training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            "Expected each run directory to contain metrics.jsonl. "
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
            "Expected each metrics.jsonl to contain production train rows. "
            f"Provided value: {str(path)!r}."
        )
    return sorted(training, key=lambda row: int(row["completed_epochs"]))


def _completed_result(run_dir: Path) -> dict[str, Any]:
    result = _json(run_dir / "result.json")
    if result.get("status") != "complete" or result.get("error") is not None:
        raise ValueError(
            "Expected a semantically complete training result. Provided "
            f"value: run={str(run_dir)!r}, status={result.get('status')!r}, "
            f"error={result.get('error')!r}."
        )
    return result


def _test_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    result = _json(path)
    if result.get("status") != "complete":
        raise ValueError(
            "Expected an official-test result with status='complete'. "
            f"Provided value: path={str(path)!r}, "
            f"status={result.get('status')!r}."
        )
    metrics = dict(result["metrics"])
    if metrics.get("split_protocol", {}).get("effective") != "held_out_test":
        raise ValueError(
            "Expected the supplied validation result to use held_out_test. "
            f"Provided value: {metrics.get('split_protocol')!r}."
        )
    return metrics


def _arm_summary(
    name: str,
    run_dir: Path,
    test_result: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = _completed_result(run_dir)
    rows = _training_rows(run_dir)
    accuracies = [float(row["validation"]["accuracy"]) for row in rows]
    costs = [float(row["validation"]["mean_cost"]) for row in rows]
    metrics = result["metrics"]
    selection = metrics["learning_rate_selection"]
    selected = metrics["selected"]
    projection = metrics["update_programming"]["parameters"]
    return (
        {
            "name": name,
            "run_dir": str(run_dir),
            "status": result["status"],
            "completed_epochs": int(metrics["completed_epochs"]),
            "selected_epoch": int(selected["epoch"]),
            "selected_validation_accuracy": float(selected["accuracy"]),
            "selected_validation_mean_cost": float(selected["value"]),
            "last_validation_accuracy": accuracies[-1],
            "last_validation_mean_cost": costs[-1],
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
            "projection_by_parameter": projection,
            "official_test": _test_metrics(test_result),
        },
        rows,
    )


def compare(
    raw_run_dir: Path,
    isotonic_run_dir: Path,
    *,
    raw_test_result: Path | None,
    isotonic_test_result: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    raw, raw_rows = _arm_summary("raw", raw_run_dir, raw_test_result)
    isotonic, isotonic_rows = _arm_summary(
        "isotonic",
        isotonic_run_dir,
        isotonic_test_result,
    )
    raw_accuracy = raw["selected_validation_accuracy"]
    isotonic_accuracy = isotonic["selected_validation_accuracy"]
    gates = {
        "raw_completed_without_failure": raw["status"] == "complete",
        "raw_validation_accuracy_at_least_90pct": raw_accuracy >= 0.90,
        "raw_within_5_percentage_points_of_isotonic": (
            raw_accuracy >= isotonic_accuracy - 0.05
        ),
        "raw_late_drop_below_5_percentage_points": (
            raw["late_accuracy_drop_from_peak"] < 0.05
        ),
    }
    summary = {
        "schema": "measured-cohort-a-comparison/v1",
        "raw": raw,
        "isotonic": isotonic,
        "validation_accuracy_difference_raw_minus_isotonic": (
            raw_accuracy - isotonic_accuracy
        ),
        "feasibility_gates": gates,
        "automatic_accuracy_gates_pass": all(gates.values()),
        "projection_diagnostics_require_interpretation": True,
        "cohort_b_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for label, rows in (("Raw measured", raw_rows), ("Isotonic", isotonic_rows)):
        epochs = [int(row["completed_epochs"]) for row in rows]
        axes[0].plot(
            epochs,
            [100.0 * float(row["validation"]["accuracy"]) for row in rows],
            marker="o",
            markersize=3,
            label=label,
        )
        axes[1].plot(
            epochs,
            [float(row["validation"]["mean_cost"]) for row in rows],
            marker="o",
            markersize=3,
            label=label,
        )
    axes[0].axhline(90.0, color="0.4", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Epoch", ylabel="Validation accuracy (%)")
    axes[1].set(xlabel="Epoch", ylabel="Validation mean cost")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output_dir / "validation_comparison.png", dpi=180)
    plt.close(figure)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run-dir", type=Path, required=True)
    parser.add_argument("--isotonic-run-dir", type=Path, required=True)
    parser.add_argument("--raw-test-result", type=Path)
    parser.add_argument("--isotonic-test-result", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    compare(
        args.raw_run_dir.expanduser().resolve(),
        args.isotonic_run_dir.expanduser().resolve(),
        raw_test_result=(
            None
            if args.raw_test_result is None
            else args.raw_test_result.expanduser().resolve()
        ),
        isotonic_test_result=(
            None
            if args.isotonic_test_result is None
            else args.isotonic_test_result.expanduser().resolve()
        ),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
