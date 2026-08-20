#!/usr/bin/env python3
"""Compare four- and eight-device one-pulse-down threshold sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


_ORDER = ("zero", "p90", "p99", "p99.9")
_CONFIG_BY_LABEL = {
    "zero": "isotonic_10ep.json",
    "p90": "positive_p90_10ep.json",
    "p99": "positive_p99_10ep.json",
    "p99.9": "positive_p99_9_10ep.json",
}
_EIGHT_KEYS = {
    0: ("base.conductance_plus.0", "base.conductance_minus.0"),
    1: ("base.conductance_plus.1", "base.conductance_minus.1"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--four-analysis", required=True)
    for label in ("zero", "p90", "p99", "p99-9"):
        parser.add_argument(f"--eight-{label}-train", required=True)
        parser.add_argument(f"--eight-{label}-test", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(
            "Expected a JSON input to be an existing file. "
            f"Provided value: {str(path)!r}."
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(
            "Expected a JSON input to contain an object. "
            f"Provided value: {type(value).__name__!r}."
        )
    return value


def _run_dir(value: str, *, role: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_file() and path.name == "result.json":
        path = path.parent
    if not path.is_dir() or not (path / "result.json").is_file():
        raise ValueError(
            f"Expected {role} to be a run directory containing result.json. "
            f"Provided value: {str(path)!r}."
        )
    result = _load_json(path / "result.json")
    if result.get("status") != "complete":
        raise ValueError(
            f"Expected {role} to have status 'complete'. "
            f"Provided value: {result.get('status')!r}."
        )
    return path


def _stream(run: Path) -> list[dict[str, Any]]:
    path = run / "metrics.jsonl"
    if not path.is_file():
        raise ValueError(
            "Expected a train run to contain metrics.jsonl. "
            f"Provided value: {str(run)!r}."
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _command_option(command: list[Any], option: str) -> str:
    if option not in command:
        raise ValueError(
            f"Expected a run command to contain {option}. "
            f"Provided value: {command!r}."
        )
    index = command.index(option)
    if index + 1 >= len(command) or not isinstance(command[index + 1], str):
        raise ValueError(
            f"Expected run command {option} to have a string value. "
            f"Provided value: {command!r}."
        )
    return command[index + 1]


def _eight_row(label: str, train: Path, test: Path) -> dict[str, Any]:
    manifest = _load_json(train / "manifest.json")
    configured = Path(
        _command_option(manifest.get("command", []), "--config")
    ).name
    if configured != _CONFIG_BY_LABEL[label]:
        raise ValueError(
            "Expected an eight-device arm to use its declared threshold "
            f"config. Provided value: label={label!r}, config={configured!r}."
        )
    train_metrics = _load_json(train / "result.json")["metrics"]
    test_metrics = _load_json(test / "result.json")["metrics"]
    records = _stream(train)
    initialization = [
        item for item in records if item.get("mode") == "initialization"
    ]
    epochs = {
        int(item["epoch"]): item
        for item in records
        if item.get("mode") == "train"
    }
    if len(initialization) != 1 or sorted(epochs) != list(range(10)):
        raise ValueError(
            "Expected an eight-device run to contain one initialization "
            "record and epochs 0 through 9. Provided value: "
            f"run={str(train)!r}, initialization={len(initialization)!r}, "
            f"epochs={sorted(epochs)!r}."
        )
    programming = train_metrics["measured_programming"]
    parameters = programming["parameters"]
    expected_keys = {key for keys in _EIGHT_KEYS.values() for key in keys}
    if set(parameters) != expected_keys:
        raise ValueError(
            "Expected eight-device programming parameters to use the four "
            "stable plus/minus keys. Provided value: "
            f"{sorted(parameters)!r}."
        )
    thresholds = programming.get("positive_gradient_threshold_by_parameter")
    if thresholds is None:
        thresholds = {key: 0.0 for key in expected_keys}
    thresholds = {key: float(thresholds[key]) for key in expected_keys}
    checkpoint_thresholds = test_metrics["checkpoint_metadata"].get(
        "positive_gradient_threshold_by_parameter"
    )
    if checkpoint_thresholds != thresholds:
        raise ValueError(
            "Expected test-checkpoint thresholds to match the train report. "
            f"Provided value: train={thresholds!r}, "
            f"test={checkpoint_thresholds!r}."
        )
    selected = train_metrics["selected"]
    row: dict[str, Any] = {
        "scheme": "eight-device",
        "label": label,
        "train_run_id": train.name,
        "test_run_id": test.name,
        "initial_validation_accuracy": train_metrics["initial_validation"][
            "student_accuracy"
        ],
        "initial_validation_kl": train_metrics["initial_validation"][
            "kl_teacher_student"
        ],
        "selected_epoch_zero_based": selected["epoch"],
        "selected_epoch_human": (
            0 if selected["epoch"] < 0 else selected["epoch"] + 1
        ),
        "selected_validation_accuracy": selected["student_accuracy"],
        "selected_validation_kl": selected["kl_teacher_student"],
        "final_validation_accuracy": train_metrics["last_validation"][
            "student_accuracy"
        ],
        "final_validation_kl": train_metrics["last_validation"][
            "kl_teacher_student"
        ],
        "test_accuracy": test_metrics["student_accuracy"],
        "test_kl": test_metrics["kl_teacher_student"],
        "test_teacher_agreement": test_metrics["teacher_agreement"],
        "thresholds_json": json.dumps(thresholds, sort_keys=True),
        "global_nearest_fine_tuning": programming[
            "global_nearest_fine_tuning"
        ],
        "digital_shadow_accumulation": programming[
            "digital_shadow_accumulation"
        ],
        "learning_rate_magnitude_used": programming[
            "learning_rate_magnitude_used"
        ],
    }
    for layer, keys in _EIGHT_KEYS.items():
        reports = [parameters[key] for key in keys]
        saturations = [item["current_last_pulse_fraction"] for item in reports]
        row.update(
            {
                f"layer_{layer}_final_last_pulse_fraction_min": min(
                    saturations
                ),
                f"layer_{layer}_final_last_pulse_fraction_max": max(
                    saturations
                ),
                f"layer_{layer}_max_abs_pulse_jump": max(
                    item["max_abs_pulse_jump"] for item in reports
                ),
                f"layer_{layer}_max_conductance_increase_s": max(
                    item["max_conductance_increase_s"] for item in reports
                ),
                f"layer_{layer}_invariant_passed": all(
                    item["one_pulse_down_invariant_passed"]
                    for item in reports
                ),
            }
        )
    return row


def _four_row(source: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: source[key]
        for key in (
            "label",
            "train_run_id",
            "test_run_id",
            "initial_validation_accuracy",
            "initial_validation_kl",
            "selected_epoch_zero_based",
            "selected_epoch_human",
            "selected_validation_accuracy",
            "selected_validation_kl",
            "final_validation_accuracy",
            "final_validation_kl",
            "test_accuracy",
            "test_kl",
            "test_teacher_agreement",
            "global_nearest_fine_tuning",
            "digital_shadow_accumulation",
        )
    }
    row["scheme"] = "four-device"
    row["learning_rate_magnitude_used"] = False
    row["thresholds_json"] = json.dumps(
        {
            "base.dense_weight.0": source[
                "threshold_base_dense_weight_0"
            ],
            "base.dense_weight.1": source[
                "threshold_base_dense_weight_1"
            ],
        },
        sort_keys=True,
    )
    for layer in (0, 1):
        saturation = source[f"layer_{layer}_final_last_pulse_fraction"]
        row[f"layer_{layer}_final_last_pulse_fraction_min"] = saturation
        row[f"layer_{layer}_final_last_pulse_fraction_max"] = saturation
        row[f"layer_{layer}_max_abs_pulse_jump"] = source[
            f"layer_{layer}_max_abs_pulse_jump"
        ]
        row[f"layer_{layer}_max_conductance_increase_s"] = source[
            f"layer_{layer}_conductance_increase_fraction"
        ]
        row[f"layer_{layer}_invariant_passed"] = source[
            f"layer_{layer}_invariant_passed"
        ]
    return row


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


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {(row["scheme"], row["label"]): row for row in rows}
    x = np.arange(len(_ORDER))
    width = 0.36
    colors = {"four-device": "#377eb8", "eight-device": "#e41a1c"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    panels = (
        ("selected_validation_accuracy", "Selected validation accuracy (%)"),
        ("test_accuracy", "Fresh test accuracy (%)"),
        ("final_validation_accuracy", "Epoch-10 validation accuracy (%)"),
    )
    for axis, (field, title) in zip(axes.flat[:3], panels, strict=True):
        for offset, scheme in ((-width / 2, "four-device"), (width / 2, "eight-device")):
            axis.bar(
                x + offset,
                [100.0 * lookup[(scheme, label)][field] for label in _ORDER],
                width,
                label=scheme,
                color=colors[scheme],
            )
        axis.set(xticks=x, xticklabels=_ORDER, ylabel="accuracy (%)", title=title)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    axis = axes.flat[3]
    for offset, scheme in ((-width / 2, "four-device"), (width / 2, "eight-device")):
        axis.bar(
            x + offset,
            [
                100.0
                * max(
                    lookup[(scheme, label)][
                        "layer_0_final_last_pulse_fraction_max"
                    ],
                    lookup[(scheme, label)][
                        "layer_1_final_last_pulse_fraction_max"
                    ],
                )
                for label in _ORDER
            ],
            width,
            label=scheme,
            color=colors[scheme],
        )
    axis.set(
        xticks=x,
        xticklabels=_ORDER,
        ylabel="maximum final-pulse occupancy (%)",
        title="Worst physical tensor after epoch 10",
    )
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    fig.suptitle("One-pulse-down threshold fine-tuning: four vs eight devices")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lookup = {(row["scheme"], row["label"]): row for row in rows}
    lines = [
        "# Four- versus eight-device one-pulse-down comparison",
        "",
        "| Gate | Four selected val. | Four test | Eight selected val. | Eight test | Selected epoch (4 / 8) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in _ORDER:
        four = lookup[("four-device", label)]
        eight = lookup[("eight-device", label)]
        lines.append(
            f"| {label} | {100 * four['selected_validation_accuracy']:.2f}% "
            f"| {100 * four['test_accuracy']:.2f}% "
            f"| {100 * eight['selected_validation_accuracy']:.2f}% "
            f"| {100 * eight['test_accuracy']:.2f}% "
            f"| {four['selected_epoch_human']} / {eight['selected_epoch_human']} |"
        )
    invariant_passed = all(
        row[f"layer_{layer}_invariant_passed"]
        and row[f"layer_{layer}_max_abs_pulse_jump"] <= 1
        and row[f"layer_{layer}_max_conductance_increase_s"] == 0.0
        and not row["global_nearest_fine_tuning"]
        and not row["digital_shadow_accumulation"]
        and not row["learning_rate_magnitude_used"]
        for row in rows
        for layer in (0, 1)
    )
    lines.extend(
        [
            "",
            "All physical update invariants passed."
            if invariant_passed
            else "At least one physical update invariant failed.",
            "",
            "The schemes use separate per-parameter thresholds calibrated at their own shared isotonic initialization; threshold magnitudes are therefore not compared directly across parameterizations.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    four = _load_json(Path(args.four_analysis).expanduser().resolve())
    four_rows = [_four_row(item) for item in four.get("rows", [])]
    if [row["label"] for row in four_rows] != list(_ORDER):
        raise ValueError(
            "Expected four-device analysis rows in zero, p90, p99, p99.9 "
            f"order. Provided value: {[row['label'] for row in four_rows]!r}."
        )
    values = {
        "zero": (args.eight_zero_train, args.eight_zero_test),
        "p90": (args.eight_p90_train, args.eight_p90_test),
        "p99": (args.eight_p99_train, args.eight_p99_test),
        "p99.9": (args.eight_p99_9_train, args.eight_p99_9_test),
    }
    eight_rows = [
        _eight_row(
            label,
            _run_dir(values[label][0], role=f"eight {label} train"),
            _run_dir(values[label][1], role=f"eight {label} test"),
        )
        for label in _ORDER
    ]
    reference_initial = eight_rows[0]
    initial_matches = all(
        row["initial_validation_accuracy"]
        == reference_initial["initial_validation_accuracy"]
        and math.isclose(
            row["initial_validation_kl"],
            reference_initial["initial_validation_kl"],
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        for row in eight_rows[1:]
    )
    if not initial_matches:
        provided_initial = [
            (
                row["initial_validation_accuracy"],
                row["initial_validation_kl"],
            )
            for row in eight_rows
        ]
        raise ValueError(
            "Expected all eight-device arms to share initialization accuracy "
            "exactly and KL within absolute tolerance 1e-7. Provided value: "
            f"{provided_initial!r}."
        )
    rows = [*four_rows, *eight_rows]
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "four_vs_eight_summary.csv", rows)
    _plot(rows, output / "four_vs_eight_accuracy.png")
    _write_report(output / "report.md", rows)
    payload = {
        "schema": "mnist-four-vs-eight-one-pulse-down-analysis",
        "schema_version": 1,
        "rows": rows,
    }
    (output / "four_vs_eight_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), "rows": len(rows)}))


if __name__ == "__main__":
    main()
