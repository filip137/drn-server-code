"""Compare cohort-B accumulated programming deadbands against no deadband."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _complete_result(run_dir: Path) -> dict[str, Any]:
    result = _json(run_dir / "result.json")
    if result.get("status") != "complete" or result.get("error") is not None:
        raise ValueError(
            "Expected a semantically complete cohort-B training result. "
            f"Provided value: run={str(run_dir)!r}, "
            f"status={result.get('status')!r}, error={result.get('error')!r}."
        )
    return result


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


def _weighted_programming_fraction(
    parameters: dict[str, dict[str, Any]],
    field: str,
    *,
    baseline_default: float | None = None,
) -> float:
    weighted = 0.0
    elements = 0
    for key, report in sorted(parameters.items()):
        shape = report.get("shape")
        if not isinstance(shape, list) or not shape:
            raise ValueError(
                "Expected every programming report to contain a non-empty "
                f"shape array. Provided value: key={key!r}, shape={shape!r}."
            )
        count = math.prod(int(value) for value in shape)
        value = report.get(field, baseline_default)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(
                f"Expected {field} to be finite for every parameter. "
                f"Provided value: key={key!r}, value={value!r}."
            )
        weighted += count * float(value)
        elements += count
    if elements == 0:
        raise ValueError(
            "Expected at least one measured dense-weight programming report. "
            "Provided value: none."
        )
    return weighted / elements


def _run_summary(
    run_dir: Path,
    test_result: Path,
    *,
    is_baseline: bool,
) -> dict[str, Any]:
    result = _complete_result(run_dir)
    metrics = result["metrics"]
    programming = metrics["update_programming"]
    if programming.get("active_cohort") != "B":
        raise ValueError(
            "Expected every sweep run to use cohort B exclusively. Provided "
            f"value: run={str(run_dir)!r}, "
            f"active_cohort={programming.get('active_cohort')!r}."
        )
    relative = float(programming.get("programming_deadband_relative", 0.0))
    mode = programming.get("programming_deadband_mode", "none")
    expected_baseline = relative == 0.0 and mode == "none"
    if is_baseline != expected_baseline:
        raise ValueError(
            "Expected exactly the baseline run to use no programming "
            f"deadband. Provided value: run={str(run_dir)!r}, mode={mode!r}, "
            f"relative={relative!r}, is_baseline={is_baseline!r}."
        )
    parameter_reports = dict(programming["parameters"])
    test = _test_metrics(test_result)
    selected = metrics["selected"]
    last = metrics["last_validation"]
    return {
        "run_dir": str(run_dir),
        "test_result": str(test_result),
        "programming_deadband_mode": mode,
        "programming_deadband_relative": relative,
        "selected_epoch": int(selected["epoch"]),
        "selected_validation_accuracy": float(selected["accuracy"]),
        "selected_validation_mean_cost": float(selected["value"]),
        "last_validation_accuracy": float(last["accuracy"]),
        "official_test_accuracy": float(test["accuracy"]),
        "official_test_mean_cost": float(test["mean_cost"]),
        "programming_event_fraction": _weighted_programming_fraction(
            parameter_reports,
            "programming_event_fraction",
            baseline_default=1.0 if is_baseline else None,
        ),
        "pulse_changed_fraction": _weighted_programming_fraction(
            parameter_reports,
            "pulse_changed_fraction",
        ),
        "parameters": parameter_reports,
    }


def summarize_threshold_sweep(
    baseline_run_dir: Path,
    threshold_run_dirs: Sequence[Path],
    *,
    baseline_test_result: Path,
    threshold_test_results: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    if not threshold_run_dirs or len(threshold_run_dirs) != len(
        threshold_test_results
    ):
        raise ValueError(
            "Expected matching non-empty threshold run and test-result "
            "sequences. Provided value: "
            f"runs={len(threshold_run_dirs)}, "
            f"tests={len(threshold_test_results)}."
        )
    baseline = _run_summary(
        baseline_run_dir,
        baseline_test_result,
        is_baseline=True,
    )
    threshold_runs = [
        _run_summary(run_dir, test_path, is_baseline=False)
        for run_dir, test_path in zip(
            threshold_run_dirs,
            threshold_test_results,
            strict=True,
        )
    ]
    threshold_runs.sort(key=lambda item: item["programming_deadband_relative"])
    relative_values = [
        item["programming_deadband_relative"] for item in threshold_runs
    ]
    if len(set(relative_values)) != len(relative_values):
        raise ValueError(
            "Expected unique positive programming deadbands. Provided value: "
            f"{relative_values!r}."
        )

    for item in threshold_runs:
        item["selected_validation_delta_vs_baseline"] = (
            item["selected_validation_accuracy"]
            - baseline["selected_validation_accuracy"]
        )
        item["official_test_delta_vs_baseline"] = (
            item["official_test_accuracy"]
            - baseline["official_test_accuracy"]
        )
        item["programming_event_reduction_vs_baseline"] = (
            1.0
            - item["programming_event_fraction"]
            / baseline["programming_event_fraction"]
        )
        item["pulse_change_reduction_vs_baseline"] = (
            1.0
            - item["pulse_changed_fraction"]
            / baseline["pulse_changed_fraction"]
        )

    best_test = max(
        threshold_runs,
        key=lambda item: (
            item["official_test_accuracy"],
            -item["programming_event_fraction"],
        ),
    )
    best_validation = max(
        threshold_runs,
        key=lambda item: (
            item["selected_validation_accuracy"],
            -item["programming_event_fraction"],
        ),
    )
    accuracy_tolerance = 0.0025
    retained_accuracy = [
        item
        for item in threshold_runs
        if item["official_test_accuracy"]
        >= baseline["official_test_accuracy"] - accuracy_tolerance
    ]
    most_efficient_retained = (
        None
        if not retained_accuracy
        else min(
            retained_accuracy,
            key=lambda item: item["programming_event_fraction"],
        )["programming_deadband_relative"]
    )
    conclusion = {
        "any_selected_validation_improvement": any(
            item["selected_validation_delta_vs_baseline"] > 0.0
            for item in threshold_runs
        ),
        "any_official_test_improvement": any(
            item["official_test_delta_vs_baseline"] > 0.0
            for item in threshold_runs
        ),
        "best_official_test_deadband_relative": best_test[
            "programming_deadband_relative"
        ],
        "best_official_test_delta": best_test[
            "official_test_delta_vs_baseline"
        ],
        "best_validation_deadband_relative": best_validation[
            "programming_deadband_relative"
        ],
        "best_validation_delta": best_validation[
            "selected_validation_delta_vs_baseline"
        ],
        "accuracy_retention_tolerance": accuracy_tolerance,
        "most_efficient_accuracy_retaining_deadband_relative": (
            most_efficient_retained
        ),
    }
    summary = {
        "schema": "measured-cohort-b-threshold-sweep/v1",
        "baseline": baseline,
        "threshold_runs": threshold_runs,
        "conclusion": conclusion,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    rows = [baseline, *threshold_runs]
    x = [item["programming_deadband_relative"] for item in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(
        x,
        [100.0 * item["selected_validation_accuracy"] for item in rows],
        marker="o",
        label="Selected validation",
    )
    axes[0].plot(
        x,
        [100.0 * item["official_test_accuracy"] for item in rows],
        marker="s",
        label="Official test",
    )
    axes[0].set(
        xlabel="Deadband / initial shadow RMS",
        ylabel="Accuracy (%)",
    )
    axes[0].legend()

    axes[1].plot(
        x,
        [100.0 * item["programming_event_fraction"] for item in rows],
        marker="o",
        label="Programming requests",
    )
    axes[1].plot(
        x,
        [100.0 * item["pulse_changed_fraction"] for item in rows],
        marker="s",
        label="Pulse-index changes",
    )
    axes[1].set(
        xlabel="Deadband / initial shadow RMS",
        ylabel="Fraction of element-steps (%)",
        yscale="log",
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(output_dir / "cohort_b_threshold_sweep.png", dpi=180)
    plt.close(figure)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--baseline-test-result", type=Path, required=True)
    parser.add_argument(
        "--threshold-run-dir",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--threshold-test-result",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summarize_threshold_sweep(
        args.baseline_run_dir.expanduser().resolve(),
        [path.expanduser().resolve() for path in args.threshold_run_dir],
        baseline_test_result=args.baseline_test_result.expanduser().resolve(),
        threshold_test_results=[
            path.expanduser().resolve()
            for path in args.threshold_test_result
        ],
        output_dir=args.output_dir.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
