#!/usr/bin/env python3
"""Collect Conv1 input-gain-40 hard-sigmoid selected-LR 50-epoch runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(
    "/home/filip/server_code/results/"
    "mnist_bp_conv1_inputgain40_selected_lr_hardsigmoid_voff15_seed0_50epoch"
)
RUNS = [
    ("mnist_bp_amp_v1_c1", 1.0, 1.0, 0.018),
    ("mnist_bp_amp_v2_c1", 2.0, 1.0, 0.012),
    ("mnist_bp_amp_v4_c1", 4.0, 1.0, 0.009),
    ("mnist_bp_amp_v1_c2", 1.0, 2.0, 0.036),
    ("mnist_bp_amp_v1_c4", 1.0, 4.0, 0.048),
]

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
    _write_csv(root / "selected_lr_summary.csv", rows)
    complete = sum(row["status"] == "complete" for row in rows)
    print(f"[collect] complete={complete}/{len(rows)} root={root}")
    print(f"[collect] wrote {root / 'summary.csv'}")
    print(f"[collect] wrote {root / 'selected_lr_summary.csv'}")


if __name__ == "__main__":
    main()
