#!/usr/bin/env python3
"""Collect Conv1 input-gain-40 hard-sigmoid LR sweep results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path("/home/filip/server_code/results/mnist_bp_conv1_inputgain40_lr_sweep_hardsigmoid_voff15_seed0_10epoch")
RUNS = [
    ("mnist_bp_amp_v1_c1", 1.0, 1.0, [0.003, 0.006, 0.009, 0.012, 0.018, 0.024]),
    ("mnist_bp_amp_v2_c1", 2.0, 1.0, [0.003, 0.006, 0.009, 0.012, 0.018]),
    ("mnist_bp_amp_v4_c1", 4.0, 1.0, [0.003, 0.006, 0.009, 0.012, 0.018]),
    ("mnist_bp_amp_v1_c2", 1.0, 2.0, [0.006, 0.009, 0.012, 0.018, 0.024, 0.036]),
    ("mnist_bp_amp_v1_c4", 1.0, 4.0, [0.006, 0.009, 0.012, 0.018, 0.024, 0.036, 0.048]),
]

SUMMARY_COLUMNS = [
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
BY_AMP_LR_COLUMNS = [
    "run_name",
    "voltage_amp",
    "current_amp",
    "lr",
    "lr_label",
    "num_runs",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "final_test_loss",
    "collapsed",
    "epoch_limited",
]
SELECTED_COLUMNS = [
    "run_name",
    "voltage_amp",
    "current_amp",
    "selected_lr",
    "selected_lr_label",
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
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def _run_dir(root: Path, run_name: str, lr: float) -> Path:
    return root / f"lr_{_label_float(lr)}" / "hard_sigmoid" / run_name / "seed_0"


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
            "epoch_limited": best_epoch >= 10,
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


def _by_amp_lr(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        out.append(
            {
                "run_name": row["run_name"],
                "voltage_amp": row["voltage_amp"],
                "current_amp": row["current_amp"],
                "lr": row["lr"],
                "lr_label": row["lr_label"],
                "num_runs": 1 if row["status"] == "complete" else 0,
                "best_test_accuracy": row["best_test_accuracy"],
                "final_test_accuracy": row["final_test_accuracy"],
                "best_epoch": row["best_epoch"],
                "final_train_loss": row["final_train_loss"],
                "final_test_loss": row["final_test_loss"],
                "collapsed": row["collapsed"],
                "epoch_limited": row["epoch_limited"],
            }
        )
    return out


def _selected(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["status"] != "complete" or row["collapsed"] is True:
            continue
        grouped.setdefault(row["run_name"], []).append(row)

    out = []
    for run_name, group in sorted(grouped.items()):
        def score(row: dict) -> tuple[float, float, float]:
            return (
                _float(row["best_test_accuracy"], -math.inf),
                -_float(row["final_test_loss"], math.inf),
                -float(row["lr"]),
            )

        best = max(group, key=score)
        out.append(
            {
                "run_name": run_name,
                "voltage_amp": best["voltage_amp"],
                "current_amp": best["current_amp"],
                "selected_lr": best["lr"],
                "selected_lr_label": best["lr_label"],
                "best_test_accuracy": best["best_test_accuracy"],
                "final_test_accuracy": best["final_test_accuracy"],
                "best_epoch": best["best_epoch"],
                "final_train_loss": best["final_train_loss"],
                "final_test_loss": best["final_test_loss"],
                "collapsed": best["collapsed"],
                "epoch_limited": best["epoch_limited"],
                "run_dir": best["run_dir"],
                "checkpoint_path": best["checkpoint_path"],
                "weights_best_path": best["weights_best_path"],
                "weights_final_path": best["weights_final_path"],
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    rows = []
    for run_name, voltage_amp, current_amp, lrs in RUNS:
        for lr in lrs:
            rows.append(_row(root, run_name, voltage_amp, current_amp, lr))

    _write_csv(root / "summary.csv", SUMMARY_COLUMNS, rows)
    _write_csv(root / "summary_by_amp_lr.csv", BY_AMP_LR_COLUMNS, _by_amp_lr(rows))
    _write_csv(root / "selected_lr_by_amp.csv", SELECTED_COLUMNS, _selected(rows))
    complete = sum(row["status"] == "complete" for row in rows)
    print(f"[collect] complete={complete}/{len(rows)} root={root}")
    print(f"[collect] wrote {root / 'summary.csv'}")
    print(f"[collect] wrote {root / 'summary_by_amp_lr.csv'}")
    print(f"[collect] wrote {root / 'selected_lr_by_amp.csv'}")


if __name__ == "__main__":
    main()
