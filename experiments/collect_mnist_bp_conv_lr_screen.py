#!/usr/bin/env python3
"""Collect Conv DRN learning-rate screen results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROW_COLUMNS = [
    "lr_label",
    "lr",
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "input_gain",
    "iteration_count",
    "epochs",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "final_test_loss",
    "epoch_limited",
    "unstable",
    "collapsed",
    "run_dir",
    "checkpoint_path",
    "weights_best_path",
    "weights_final_path",
]

SELECTED_COLUMNS = [
    "non_linearity",
    "run_name",
    "voltage_amp",
    "current_amp",
    "selected_lr_label",
    "selected_lr",
    "input_gain",
    "iteration_count",
    "epochs",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "final_test_loss",
    "epoch_limited",
    "unstable",
    "collapsed",
    "run_dir",
    "checkpoint_path",
    "weights_best_path",
    "weights_final_path",
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _float(value: object, default: float = math.nan) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _lr_label(path: Path, lr: float) -> str:
    for parent in [path, *path.parents]:
        if parent.name.startswith("lr_"):
            return parent.name.removeprefix("lr_")
    return f"{lr:g}".replace(".", "p").replace("-", "m")


def _row(metrics_path: Path, *, epoch_limited_fraction: float, unstable_gap: float) -> dict:
    run_dir = metrics_path.parent
    metrics = _read_json(metrics_path)
    config = _read_json(run_dir / "config.json")
    architecture = config["architecture"]
    training = config["training"]
    optimizer = config["optimizer"]
    amplification = config["amplification"]
    lr_values = optimizer.get("learning_rate", [])
    lr = float(lr_values[0] if isinstance(lr_values, list) else lr_values)
    epochs = int(training["epochs"])
    best_epoch = int(metrics.get("best_epoch", 0))
    best_acc = _float(metrics.get("best_test_accuracy"))
    final_acc = _float(metrics.get("final_test_accuracy"))
    epoch_limited = best_epoch >= max(1, math.ceil((1.0 - epoch_limited_fraction) * epochs))
    unstable = math.isfinite(best_acc) and math.isfinite(final_acc) and (best_acc - final_acc) > unstable_gap
    collapsed = (not math.isfinite(best_acc)) or best_acc < 0.5
    return {
        "lr_label": _lr_label(run_dir, lr),
        "lr": lr,
        "non_linearity": architecture["non_linearity"],
        "run_name": run_dir.parent.name,
        "seed": config["seed"],
        "voltage_amp": amplification["voltage_amp"],
        "current_amp": amplification["current_amp"],
        "input_gain": architecture["input_gain"],
        "iteration_count": training["num_iterations_training"],
        "epochs": epochs,
        "best_test_accuracy": metrics.get("best_test_accuracy", ""),
        "final_test_accuracy": metrics.get("final_test_accuracy", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "final_train_loss": metrics.get("final_train_loss", ""),
        "final_test_loss": metrics.get("final_test_loss", ""),
        "epoch_limited": bool(epoch_limited),
        "unstable": bool(unstable),
        "collapsed": bool(collapsed),
        "run_dir": str(run_dir),
        "checkpoint_path": metrics.get("checkpoint_path", ""),
        "weights_best_path": metrics.get("weights_best_path", ""),
        "weights_final_path": metrics.get("weights_final_path", ""),
    }


def _rows(root: Path, *, epoch_limited_fraction: float, unstable_gap: float) -> list[dict]:
    rows = [
        _row(path, epoch_limited_fraction=epoch_limited_fraction, unstable_gap=unstable_gap)
        for path in sorted(root.glob("**/metrics.json"))
    ]
    return sorted(rows, key=lambda row: (row["non_linearity"], row["run_name"], float(row["lr"])))


def _selected(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["non_linearity"], row["run_name"]), []).append(row)
    out: list[dict] = []
    for (non_linearity, run_name), group in sorted(grouped.items()):
        candidates = [row for row in group if row["collapsed"] is not True] or group

        def score(row: dict) -> tuple[float, float, float, float]:
            return (
                _float(row["best_test_accuracy"], -math.inf),
                -_float(row["final_test_loss"], math.inf),
                -abs(_float(row["best_test_accuracy"], 0.0) - _float(row["final_test_accuracy"], 0.0)),
                -float(row["lr"]),
            )

        best = max(candidates, key=score)
        out.append(
            {
                "non_linearity": non_linearity,
                "run_name": run_name,
                "voltage_amp": best["voltage_amp"],
                "current_amp": best["current_amp"],
                "selected_lr_label": best["lr_label"],
                "selected_lr": best["lr"],
                "input_gain": best["input_gain"],
                "iteration_count": best["iteration_count"],
                "epochs": best["epochs"],
                "best_test_accuracy": best["best_test_accuracy"],
                "final_test_accuracy": best["final_test_accuracy"],
                "best_epoch": best["best_epoch"],
                "final_train_loss": best["final_train_loss"],
                "final_test_loss": best["final_test_loss"],
                "epoch_limited": best["epoch_limited"],
                "unstable": best["unstable"],
                "collapsed": best["collapsed"],
                "run_dir": best["run_dir"],
                "checkpoint_path": best["checkpoint_path"],
                "weights_best_path": best["weights_best_path"],
                "weights_final_path": best["weights_final_path"],
            }
        )
    return out


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--epoch-limited-fraction", type=float, default=0.2)
    parser.add_argument("--unstable-gap", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    rows = _rows(
        root,
        epoch_limited_fraction=float(args.epoch_limited_fraction),
        unstable_gap=float(args.unstable_gap),
    )
    selected = _selected(rows)
    _write_csv(root / "summary.csv", ROW_COLUMNS, rows)
    _write_csv(root / "summary_by_lr_nonlinearity_amp.csv", ROW_COLUMNS, rows)
    _write_csv(root / "selected_lr_by_nonlinearity_amp.csv", SELECTED_COLUMNS, selected)
    print(f"[collect-lr] rows={len(rows)} selected={len(selected)} root={root}")


if __name__ == "__main__":
    main()
