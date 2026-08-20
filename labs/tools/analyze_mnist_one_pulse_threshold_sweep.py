#!/usr/bin/env python3
"""Summarize the MNIST one-pulse-down raw-gradient threshold sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


_CONFIG_LABELS = {
    "positive_p90_10ep.json": "p90",
    "positive_p99_10ep.json": "p99",
    "positive_p99_9_10ep.json": "p99.9",
}
_ORDER = ("zero", "p90", "p99", "p99.9")
_PARAMETERS = ("base.dense_weight.0", "base.dense_weight.1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--zero-control-study-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(
            "Expected JSON input to be an existing file. "
            f"Provided value: {str(path)!r}."
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(
            "Expected JSON input to contain an object. "
            f"Provided value: {type(value).__name__!r}."
        )
    return value


def _command_option(command: list[Any], option: str) -> str:
    if option not in command:
        raise ValueError(
            f"Expected run command to contain {option}. "
            f"Provided value: {command!r}."
        )
    index = command.index(option)
    if index + 1 >= len(command) or not isinstance(command[index + 1], str):
        raise ValueError(
            f"Expected run command {option} to have a string value. "
            f"Provided value: {command!r}."
        )
    return command[index + 1]


def _complete_runs(arm: Path) -> list[Path]:
    if not arm.is_dir():
        raise ValueError(
            "Expected study arm to be an existing directory. "
            f"Provided value: {str(arm)!r}."
        )
    result = []
    for run in sorted(path for path in arm.iterdir() if path.is_dir()):
        status = _load_json(run / "status.json")
        if status.get("status") == "complete":
            if not (run / "result.json").is_file():
                raise ValueError(
                    "Expected every complete run to contain result.json. "
                    f"Provided value: {str(run)!r}."
                )
            result.append(run)
    return result


def _labeled_runs(arm: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for run in _complete_runs(arm):
        manifest = _load_json(run / "manifest.json")
        configured = Path(
            _command_option(manifest.get("command", []), "--config")
        ).name
        label = _CONFIG_LABELS.get(configured)
        if label is None:
            raise ValueError(
                "Expected a declared threshold config filename. "
                f"Provided value: {configured!r}."
            )
        if label in result:
            raise ValueError(
                "Expected exactly one complete run per threshold label. "
                f"Provided value: duplicate {label!r}."
            )
        result[label] = run
    expected = set(_CONFIG_LABELS.values())
    if set(result) != expected:
        raise ValueError(
            "Expected complete runs for p90, p99, and p99.9. "
            f"Provided value: {sorted(result)!r}."
        )
    return result


def _only_complete_run(arm: Path) -> Path:
    runs = _complete_runs(arm)
    if len(runs) != 1:
        raise ValueError(
            "Expected exactly one complete zero-control run. "
            f"Provided value: {len(runs)!r}."
        )
    return runs[0]


def _metrics(run: Path) -> dict[str, Any]:
    return _load_json(run / "result.json")["metrics"]


def _stream(run: Path) -> list[dict[str, Any]]:
    path = run / "metrics.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _trajectory(run: Path) -> list[dict[str, Any]]:
    records = _stream(run)
    initialization = next(
        (item for item in records if item.get("mode") == "initialization"),
        None,
    )
    if initialization is None:
        raise ValueError(
            "Expected every train stream to contain initialization metrics. "
            f"Provided value: {str(run)!r}."
        )
    train_records = [item for item in records if item.get("mode") == "train"]
    manifest = _load_json(run / "manifest.json")
    resume_inputs = [
        item
        for item in manifest.get("inputs", [])
        if item.get("role") == "resume"
    ]
    if resume_inputs:
        if len(resume_inputs) != 1:
            raise ValueError(
                "Expected at most one resume input per train run. "
                f"Provided value: {resume_inputs!r}."
            )
        source_run = Path(resume_inputs[0]["path"]).resolve().parent.parent
        first_epoch = min(int(item["epoch"]) for item in train_records)
        train_records = [
            item
            for item in _stream(source_run)
            if item.get("mode") == "train"
            and int(item["epoch"]) < first_epoch
        ] + train_records
    by_epoch = {int(item["epoch"]): item for item in train_records}
    if sorted(by_epoch) != list(range(10)):
        raise ValueError(
            "Expected an exact ten-epoch trajectory numbered 0 through 9. "
            f"Provided value: {sorted(by_epoch)!r}."
        )
    result = [
        {
            "epoch": 0,
            "phase": "initialization",
            **initialization["validation"],
        }
    ]
    result.extend(
        {
            "epoch": epoch + 1,
            "phase": "train",
            **by_epoch[epoch]["validation"],
        }
        for epoch in range(10)
    )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(
    label: str,
    train_run: Path,
    test_run: Path,
) -> dict[str, Any]:
    train = _metrics(train_run)
    test = _metrics(test_run)
    programming = train["measured_programming"]
    parameters = programming["parameters"]
    configured_thresholds = programming.get(
        "positive_gradient_threshold_by_parameter"
    )
    thresholds = (
        {key: 0.0 for key in _PARAMETERS}
        if configured_thresholds is None
        else {key: float(configured_thresholds[key]) for key in _PARAMETERS}
    )
    checkpoint_thresholds = test.get("checkpoint_metadata", {}).get(
        "positive_gradient_threshold_by_parameter"
    )
    if label != "zero" and checkpoint_thresholds != thresholds:
        raise ValueError(
            "Expected selected-checkpoint thresholds to equal the final "
            "programming report. Provided value: "
            f"label={label!r}, report={thresholds!r}, "
            f"checkpoint={checkpoint_thresholds!r}."
        )
    initial = train["initial_validation"]
    selected = train["selected"]
    final = train["last_validation"]
    row: dict[str, Any] = {
        "label": label,
        "train_run_id": train_run.name,
        "test_run_id": test_run.name,
        "threshold_base_dense_weight_0": thresholds[_PARAMETERS[0]],
        "threshold_base_dense_weight_1": thresholds[_PARAMETERS[1]],
        "initial_validation_accuracy": initial["student_accuracy"],
        "initial_validation_kl": initial["kl_teacher_student"],
        "selected_epoch_zero_based": selected["epoch"],
        "selected_epoch_human": (
            0 if selected["epoch"] < 0 else selected["epoch"] + 1
        ),
        "selected_validation_accuracy": selected["student_accuracy"],
        "selected_validation_accuracy_gain_percentage_points": 100.0
        * (selected["student_accuracy"] - initial["student_accuracy"]),
        "selected_validation_kl": selected["kl_teacher_student"],
        "final_validation_accuracy": final["student_accuracy"],
        "final_validation_kl": final["kl_teacher_student"],
        "test_accuracy": test["student_accuracy"],
        "test_teacher_agreement": test["teacher_agreement"],
        "test_kl": test["kl_teacher_student"],
        "completed_epochs": _trajectory(train_run)[-1]["epoch"],
        "fine_tuning_update_rule": programming["fine_tuning_update_rule"],
        "global_nearest_fine_tuning": programming[
            "global_nearest_fine_tuning"
        ],
        "digital_shadow_accumulation": programming[
            "digital_shadow_accumulation"
        ],
    }
    for index, key in enumerate(_PARAMETERS):
        report = parameters[key]
        prefix = f"layer_{index}"
        applied = report["down_pulse_applied_fraction"]
        decrease = report["conductance_decrease_fraction"]
        positive_fraction = report.get("positive_gradient_fraction")
        if positive_fraction is None:
            positive_fraction = 1.0 - report[
                "negative_gradient_hold_fraction"
            ] - report["zero_gradient_hold_fraction"]
        eligible_fraction = report.get(
            "threshold_eligible_fraction",
            report["down_request_fraction"],
        )
        suppressed_fraction = report.get(
            "positive_gradient_suppressed_by_threshold_fraction"
        )
        if suppressed_fraction is None:
            suppressed_fraction = positive_fraction - eligible_fraction
        row.update(
            {
                f"{prefix}_positive_gradient_fraction": positive_fraction,
                f"{prefix}_threshold_eligible_fraction": eligible_fraction,
                f"{prefix}_positive_suppressed_fraction": suppressed_fraction,
                f"{prefix}_pulse_applied_fraction": applied,
                f"{prefix}_last_pulse_blocked_fraction": report[
                    "down_request_blocked_at_last_pulse_fraction"
                ],
                f"{prefix}_conductance_decrease_fraction": decrease,
                f"{prefix}_conductance_decrease_given_applied": (
                    decrease / applied if applied else None
                ),
                f"{prefix}_final_last_pulse_fraction": report[
                    "current_last_pulse_fraction"
                ],
                f"{prefix}_max_abs_pulse_jump": report[
                    "max_abs_pulse_jump"
                ],
                f"{prefix}_conductance_increase_fraction": report[
                    "conductance_increase_fraction"
                ],
                f"{prefix}_invariant_passed": report[
                    "one_pulse_down_invariant_passed"
                ],
            }
        )
    return row


def _plot(
    summary: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {
        "zero": "#777777",
        "p90": "#d95f02",
        "p99": "#1b9e77",
        "p99.9": "#7570b3",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for label in _ORDER:
        selected = [row for row in trajectory if row["label"] == label]
        axes[0, 0].plot(
            [row["epoch"] for row in selected],
            [100.0 * row["student_accuracy"] for row in selected],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            color=colors[label],
            label=label,
        )
        axes[0, 1].plot(
            [row["epoch"] for row in selected],
            [row["kl_teacher_student"] for row in selected],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            color=colors[label],
            label=label,
        )
    axes[0, 0].set(
        xlabel="completed epoch (0 = initialization)",
        ylabel="validation accuracy (%)",
        title="Validation accuracy trajectory",
    )
    axes[0, 1].set(
        xlabel="completed epoch (0 = initialization)",
        ylabel="teacher-student KL",
        title="Validation KL trajectory",
    )
    axes[0, 1].set_yscale("log")
    for axis in axes[0]:
        axis.grid(True, alpha=0.25)
        axis.legend()

    labels = [row["label"] for row in summary]
    x = np.arange(len(labels))
    axes[1, 0].bar(
        x,
        [100.0 * row["test_accuracy"] for row in summary],
        color=[colors[label] for label in labels],
    )
    axes[1, 0].axhline(
        100.0 * summary[0]["initial_validation_accuracy"],
        color="black",
        linestyle="--",
        linewidth=1.1,
        label="shared initialization validation",
    )
    axes[1, 0].set(
        xticks=x,
        xticklabels=labels,
        ylabel="fresh test accuracy (%)",
        title="KL-selected checkpoint",
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, axis="y", alpha=0.25)

    width = 0.36
    axes[1, 1].bar(
        x - width / 2,
        [100.0 * row["layer_0_final_last_pulse_fraction"] for row in summary],
        width,
        label="layer 0",
    )
    axes[1, 1].bar(
        x + width / 2,
        [100.0 * row["layer_1_final_last_pulse_fraction"] for row in summary],
        width,
        label="layer 1",
    )
    axes[1, 1].axhline(50.0, color="black", linestyle="--", linewidth=1.1)
    axes[1, 1].set(
        xticks=x,
        xticklabels=labels,
        ylabel="cells at final pulse after epoch 10 (%)",
        title="Final saturation",
    )
    axes[1, 1].legend()
    axes[1, 1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Cohort-A one-pulse-down raw-gradient threshold sweep")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    study = Path(args.study_dir).expanduser().resolve()
    zero_study = Path(args.zero_control_study_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    train_runs = _labeled_runs(study / "runs/threshold-train")
    test_runs = _labeled_runs(study / "runs/threshold-test")
    zero_train = _only_complete_run(zero_study / "runs/one-pulse-down-train")
    zero_test = _only_complete_run(zero_study / "runs/one-pulse-down-test")
    all_train = {"zero": zero_train, **train_runs}
    all_test = {"zero": zero_test, **test_runs}

    summary = [
        _summary_row(label, all_train[label], all_test[label])
        for label in _ORDER
    ]
    initial_values = {
        (row["initial_validation_accuracy"], row["initial_validation_kl"])
        for row in summary
    }
    if len(initial_values) != 1:
        raise ValueError(
            "Expected every arm and the control to have identical initial "
            f"validation metrics. Provided value: {sorted(initial_values)!r}."
        )
    zero_test_accuracy = summary[0]["test_accuracy"]
    for row in summary:
        row["test_accuracy_gain_vs_zero_percentage_points"] = 100.0 * (
            row["test_accuracy"] - zero_test_accuracy
        )

    trajectory: list[dict[str, Any]] = []
    for label in _ORDER:
        trajectory.extend(
            {"label": label, **row}
            for row in _trajectory(all_train[label])
        )
    _write_csv(output / "threshold_sweep_summary.csv", summary)
    _write_csv(output / "threshold_sweep_trajectory.csv", trajectory)
    _plot(summary, trajectory, output / "threshold_sweep_overview.png")

    supported = any(
        row["selected_validation_accuracy_gain_percentage_points"] >= 5.0
        and row["layer_0_final_last_pulse_fraction"] < 0.5
        and row["layer_1_final_last_pulse_fraction"] < 0.5
        for row in summary[1:]
    )
    report = {
        "schema": "mnist-one-pulse-threshold-sweep-analysis",
        "schema_version": 1,
        "hypothesis_supported": supported,
        "shared_initialization": {
            "validation_accuracy": summary[0]["initial_validation_accuracy"],
            "validation_kl": summary[0]["initial_validation_kl"],
        },
        "rows": summary,
    }
    (output / "threshold_sweep_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "hypothesis_supported": supported,
                "output_dir": str(output),
                "test_accuracy": {
                    row["label"]: row["test_accuracy"] for row in summary
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
