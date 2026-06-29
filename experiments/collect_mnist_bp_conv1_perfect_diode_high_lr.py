#!/usr/bin/env python3
"""Collect focused Conv1 perfect-diode high-LR rerun results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


RUNS = {
    "mnist_bp_amp_v1_c1": (1.0, 1.0, [0.012, 0.018, 0.024]),
    "mnist_bp_amp_v2_c1": (2.0, 1.0, [0.012, 0.018, 0.024]),
    "mnist_bp_amp_v1_c4": (1.0, 4.0, [0.012, 0.018, 0.024, 0.036, 0.048]),
}

SUMMARY_COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "lr",
    "lr_label",
    "input_gain",
    "iteration_count",
    "status",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "final_test_loss",
    "collapsed",
    "epoch_limited",
    "run_dir",
    "checkpoint_path",
    "weights_best_path",
    "weights_final_path",
]

SELECTED_COLUMNS = [
    "run_name",
    "voltage_amp",
    "current_amp",
    "selected_lr",
    "selected_lr_label",
    "input_gain",
    "iteration_count",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "final_test_loss",
    "epoch_limited",
    "run_dir",
    "checkpoint_path",
    "weights_best_path",
    "weights_final_path",
]


def _label_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _float(value: object, default: float = math.nan) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _config_values(run_dir: Path) -> tuple[float | str, int | str]:
    config = _read_json(run_dir / "config.json") or _read_json(run_dir / "source_config.json")
    if not config:
        return "", ""
    if "architecture" in config:
        architecture = config.get("architecture", {})
        training = config.get("training", {})
        return architecture.get("input_gain", ""), training.get("num_iterations_training", "")
    model_base = config.get("model_base", {})
    return model_base.get("input_gain", ""), model_base.get("num_iterations_training", "")


def _row(root: Path, run_name: str, voltage_amp: float, current_amp: float, lr: float) -> dict:
    lr_label = _label_float(lr)
    run_dir = root / f"lr_{lr_label}" / "perfect_diode" / run_name / "seed_0"
    metrics = _read_json(run_dir / "metrics.json")
    input_gain, iteration_count = _config_values(run_dir)
    row = {
        "run_name": run_name,
        "seed": 0,
        "voltage_amp": voltage_amp,
        "current_amp": current_amp,
        "lr": lr,
        "lr_label": lr_label,
        "input_gain": input_gain,
        "iteration_count": iteration_count,
        "status": "complete" if metrics is not None else "missing",
        "best_test_accuracy": "",
        "final_test_accuracy": "",
        "best_epoch": "",
        "final_train_loss": "",
        "final_test_loss": "",
        "collapsed": "",
        "epoch_limited": "",
        "run_dir": str(run_dir),
        "checkpoint_path": "",
        "weights_best_path": "",
        "weights_final_path": "",
    }
    if metrics is None:
        return row

    best_acc = _float(metrics.get("best_test_accuracy"))
    best_epoch = int(_float(metrics.get("best_epoch"), -1))
    row.update(
        {
            "best_test_accuracy": metrics.get("best_test_accuracy", ""),
            "final_test_accuracy": metrics.get("final_test_accuracy", ""),
            "best_epoch": metrics.get("best_epoch", ""),
            "final_train_loss": metrics.get("final_train_loss", ""),
            "final_test_loss": metrics.get("final_test_loss", ""),
            "collapsed": (not math.isfinite(best_acc)) or best_acc < 0.5,
            "epoch_limited": best_epoch >= 40,
            "checkpoint_path": metrics.get("checkpoint_path", ""),
            "weights_best_path": metrics.get("weights_best_path", ""),
            "weights_final_path": metrics.get("weights_final_path", ""),
        }
    )
    return row


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _selected(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["status"] != "complete" or row["collapsed"] is True:
            continue
        grouped.setdefault(str(row["run_name"]), []).append(row)

    selected = []
    for run_name, group in sorted(grouped.items()):
        def score(row: dict) -> tuple[float, float, float]:
            return (
                _float(row["best_test_accuracy"], -math.inf),
                _float(row["final_test_accuracy"], -math.inf),
                -_float(row["final_test_loss"], math.inf),
            )

        best = max(group, key=score)
        selected.append(
            {
                "run_name": run_name,
                "voltage_amp": best["voltage_amp"],
                "current_amp": best["current_amp"],
                "selected_lr": best["lr"],
                "selected_lr_label": best["lr_label"],
                "input_gain": best["input_gain"],
                "iteration_count": best["iteration_count"],
                "best_test_accuracy": best["best_test_accuracy"],
                "final_test_accuracy": best["final_test_accuracy"],
                "best_epoch": best["best_epoch"],
                "final_train_loss": best["final_train_loss"],
                "final_test_loss": best["final_test_loss"],
                "epoch_limited": best["epoch_limited"],
                "run_dir": best["run_dir"],
                "checkpoint_path": best["checkpoint_path"],
                "weights_best_path": best["weights_best_path"],
                "weights_final_path": best["weights_final_path"],
            }
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def main() -> None:
    root = Path(parse_args().root).expanduser().resolve()
    rows = []
    for run_name, (voltage_amp, current_amp, lrs) in RUNS.items():
        for lr in lrs:
            rows.append(_row(root, run_name, voltage_amp, current_amp, lr))
    selected = _selected(rows)
    _write_csv(root / "summary.csv", SUMMARY_COLUMNS, rows)
    _write_csv(root / "selected_lr_by_amp.csv", SELECTED_COLUMNS, selected)
    complete = sum(row["status"] == "complete" for row in rows)
    print(f"[collect-conv1-pd-high-lr] complete={complete}/{len(rows)} root={root}")
    print(f"[collect-conv1-pd-high-lr] wrote {root / 'summary.csv'}")
    print(f"[collect-conv1-pd-high-lr] wrote {root / 'selected_lr_by_amp.csv'}")


if __name__ == "__main__":
    main()
