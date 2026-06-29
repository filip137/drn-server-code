#!/usr/bin/env python3
"""Collect Conv1 hard-sigmoid input-gain-40 higher-LR 50-epoch reruns."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(
    "/home/filip/server_code/results/"
    "mnist_bp_conv1_hardsigmoid_inputgain40_higher_lr_seed0_50epoch_trex"
)
RUNS = [
    ("mnist_bp_amp_v1_c1", 1.0, 1.0, 0.024),
    ("mnist_bp_amp_v1_c1", 1.0, 1.0, 0.030),
    ("mnist_bp_amp_v1_c2", 1.0, 2.0, 0.048),
    ("mnist_bp_amp_v1_c2", 1.0, 2.0, 0.060),
]
BASELINES = {
    "mnist_bp_amp_v1_c1": {
        "lr": 0.018,
        "best_test_accuracy": 0.9552,
        "final_test_accuracy": 0.9537,
        "best_epoch": 47,
    },
    "mnist_bp_amp_v1_c2": {
        "lr": 0.036,
        "best_test_accuracy": 0.9290,
        "final_test_accuracy": 0.9260,
        "best_epoch": 46,
    },
}

COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "lr",
    "lr_label",
    "status",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "final_test_loss",
    "baseline_lr",
    "baseline_best_test_accuracy",
    "delta_best_vs_baseline",
    "baseline_final_test_accuracy",
    "delta_final_vs_baseline",
    "collapsed",
    "epoch_limited",
    "run_dir",
    "checkpoint_path",
    "weights_best_path",
    "weights_final_path",
]


def _label_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


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


def _run_dir(root: Path, run_name: str, lr: float) -> Path:
    return root / f"lr_{_label_float(lr)}" / "hard_sigmoid" / run_name / "seed_0"


def _row(root: Path, run_name: str, voltage_amp: float, current_amp: float, lr: float) -> dict:
    run_dir = _run_dir(root, run_name, lr)
    metrics = _read_json(run_dir / "metrics.json")
    baseline = BASELINES[run_name]
    row = {
        "run_name": run_name,
        "seed": 0,
        "voltage_amp": voltage_amp,
        "current_amp": current_amp,
        "lr": lr,
        "lr_label": _label_float(lr),
        "status": "complete" if metrics is not None else "missing",
        "best_test_accuracy": "",
        "final_test_accuracy": "",
        "best_epoch": "",
        "final_train_loss": "",
        "final_test_loss": "",
        "baseline_lr": baseline["lr"],
        "baseline_best_test_accuracy": baseline["best_test_accuracy"],
        "delta_best_vs_baseline": "",
        "baseline_final_test_accuracy": baseline["final_test_accuracy"],
        "delta_final_vs_baseline": "",
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
    final_acc = _float(metrics.get("final_test_accuracy"))
    best_epoch = int(_float(metrics.get("best_epoch"), -1))
    row.update(
        {
            "best_test_accuracy": metrics.get("best_test_accuracy", ""),
            "final_test_accuracy": metrics.get("final_test_accuracy", ""),
            "best_epoch": metrics.get("best_epoch", ""),
            "final_train_loss": metrics.get("final_train_loss", ""),
            "final_test_loss": metrics.get("final_test_loss", ""),
            "delta_best_vs_baseline": best_acc - baseline["best_test_accuracy"],
            "delta_final_vs_baseline": final_acc - baseline["final_test_accuracy"],
            "collapsed": (not math.isfinite(best_acc)) or best_acc < 0.5,
            "epoch_limited": best_epoch >= 50,
            "checkpoint_path": metrics.get("checkpoint_path", ""),
            "weights_best_path": metrics.get("weights_best_path", ""),
            "weights_final_path": metrics.get("weights_final_path", ""),
        }
    )
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    return parser.parse_args()


def main() -> None:
    root = Path(parse_args().root).expanduser().resolve()
    rows = [_row(root, *run) for run in RUNS]
    _write_csv(root / "summary.csv", rows)
    complete_rows = [row for row in rows if row["status"] == "complete"]
    selected = []
    for run_name in sorted({row["run_name"] for row in rows}):
        candidates = [row for row in complete_rows if row["run_name"] == run_name]
        if candidates:
            candidates.sort(
                key=lambda row: (
                    _float(row["best_test_accuracy"]),
                    _float(row["final_test_accuracy"]),
                    -_float(row["lr"]),
                ),
                reverse=True,
            )
            selected.append(candidates[0])
    _write_csv(root / "selected_higher_lr_by_run.csv", selected)
    complete = sum(row["status"] == "complete" for row in rows)
    print(f"[collect] complete={complete}/{len(rows)} root={root}")
    print(f"[collect] wrote {root / 'summary.csv'}")
    print(f"[collect] wrote {root / 'selected_higher_lr_by_run.csv'}")


if __name__ == "__main__":
    main()
