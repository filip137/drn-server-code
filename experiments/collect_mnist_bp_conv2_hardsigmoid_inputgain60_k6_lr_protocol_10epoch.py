#!/usr/bin/env python3
"""Collect Conv2 hard-sigmoid input-gain-60 K=6 LR protocol screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(
    "/home/filip/server_code/results/"
    "mnist_bp_conv2_hardsigmoid_inputgain60_k6_lr_protocol_seed0_10epoch"
)
LOSS_WINDOW = 3
MIN_ABS_LOSS_DROP = 1.0e-4
MIN_REL_LOSS_DROP = 1.0e-4

ROW_COLUMNS = [
    "lr_label",
    "lr",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "final_test_loss",
    "loss_window",
    "loss_window_start",
    "loss_window_end",
    "loss_window_drop",
    "loss_window_rel_drop",
    "loss_still_improving",
    "passes_lr_protocol",
    "run_dir",
    "checkpoint_path",
    "weights_best_path",
    "weights_final_path",
]

SELECTED_COLUMNS = [
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "selected_lr_label",
    "selected_lr",
    "selection_reason",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "final_test_loss",
    "loss_window_drop",
    "loss_window_rel_drop",
    "loss_still_improving",
    "run_dir",
    "checkpoint_path",
    "weights_best_path",
    "weights_final_path",
]


def _label_to_float(label: str) -> float | str:
    try:
        return float(label.replace("p", "."))
    except ValueError:
        return label


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _float(value: object, default: float = math.nan) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _loss_improvement(run_dir: Path) -> dict:
    path = run_dir / "loss_train.npy"
    out = {
        "loss_window": LOSS_WINDOW,
        "loss_window_start": "",
        "loss_window_end": "",
        "loss_window_drop": "",
        "loss_window_rel_drop": "",
        "loss_still_improving": False,
    }
    if not path.exists():
        return out
    values = np.load(path).astype(float)
    if values.size < 2:
        return out
    window = min(LOSS_WINDOW, values.size)
    start = float(values[-window])
    end = float(values[-1])
    drop = start - end
    rel_drop = drop / max(abs(start), 1.0e-12)
    out.update(
        {
            "loss_window": window,
            "loss_window_start": start,
            "loss_window_end": end,
            "loss_window_drop": drop,
            "loss_window_rel_drop": rel_drop,
            "loss_still_improving": (
                math.isfinite(drop)
                and math.isfinite(rel_drop)
                and drop > MIN_ABS_LOSS_DROP
                and rel_drop > MIN_REL_LOSS_DROP
            ),
        }
    )
    return out


def _rows(root: Path) -> list[dict]:
    rows = []
    for lr_root in sorted(root.glob("lr_*")):
        if not lr_root.is_dir():
            continue
        lr_label = lr_root.name.removeprefix("lr_")
        lr_value = _label_to_float(lr_label)
        for metrics_path in sorted(lr_root.glob("hard_sigmoid/*/seed_*/metrics.json")):
            run_dir = metrics_path.parent
            metrics = _read_json(metrics_path)
            config = _read_json(run_dir / "config.json")
            improvement = _loss_improvement(run_dir)
            best_acc = _float(metrics.get("best_test_accuracy"))
            row = {
                "lr_label": lr_label,
                "lr": lr_value,
                "run_name": run_dir.parent.name,
                "seed": config["seed"],
                "voltage_amp": config["amplification"]["voltage_amp"],
                "current_amp": config["amplification"]["current_amp"],
                "best_test_accuracy": metrics.get("best_test_accuracy"),
                "final_test_accuracy": metrics.get("final_test_accuracy"),
                "best_epoch": metrics.get("best_epoch"),
                "final_train_loss": metrics.get("final_train_loss"),
                "final_test_loss": metrics.get("final_test_loss"),
                "passes_lr_protocol": (
                    bool(improvement["loss_still_improving"])
                    and math.isfinite(best_acc)
                    and best_acc >= 0.5
                ),
                "run_dir": str(run_dir),
                "checkpoint_path": metrics.get("checkpoint_path"),
                "weights_best_path": metrics.get("weights_best_path"),
                "weights_final_path": metrics.get("weights_final_path"),
            }
            row.update(improvement)
            rows.append(row)
    return rows


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _selected_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, float, float], list[dict]] = {}
    for row in rows:
        key = (
            str(row["run_name"]),
            int(row["seed"]),
            float(row["voltage_amp"]),
            float(row["current_amp"]),
        )
        grouped.setdefault(key, []).append(row)

    def score(row: dict) -> tuple[float, float, float]:
        # Highest validation/test accuracy, then lower final test loss, then lower LR.
        return (
            _float(row["best_test_accuracy"]),
            -_float(row["final_test_loss"]),
            -_float(row["lr"]),
        )

    selected = []
    for key, group in sorted(grouped.items()):
        passing = [row for row in group if row["passes_lr_protocol"]]
        if passing:
            best = max(passing, key=score)
            reason = "highest_accuracy_among_loss_improving"
        else:
            best = max(group, key=score)
            reason = "fallback_highest_accuracy_no_loss_improving_candidate"
        selected.append(
            {
                "run_name": best["run_name"],
                "seed": best["seed"],
                "voltage_amp": best["voltage_amp"],
                "current_amp": best["current_amp"],
                "selected_lr_label": best["lr_label"],
                "selected_lr": best["lr"],
                "selection_reason": reason,
                "best_test_accuracy": best["best_test_accuracy"],
                "final_test_accuracy": best["final_test_accuracy"],
                "best_epoch": best["best_epoch"],
                "final_train_loss": best["final_train_loss"],
                "final_test_loss": best["final_test_loss"],
                "loss_window_drop": best["loss_window_drop"],
                "loss_window_rel_drop": best["loss_window_rel_drop"],
                "loss_still_improving": best["loss_still_improving"],
                "run_dir": best["run_dir"],
                "checkpoint_path": best["checkpoint_path"],
                "weights_best_path": best["weights_best_path"],
                "weights_final_path": best["weights_final_path"],
            }
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    return parser.parse_args()


def main() -> None:
    root = Path(parse_args().root).expanduser().resolve()
    rows = _rows(root)
    selected = _selected_rows(rows)
    _write_csv(root / "summary_all_lrs.csv", ROW_COLUMNS, rows)
    _write_csv(root / "selected_lr_by_amp_loss_improving.csv", SELECTED_COLUMNS, selected)
    print(f"[collect] runs={len(rows)} root={root}")
    print(f"[collect] wrote {root / 'summary_all_lrs.csv'}")
    print(f"[collect] wrote {root / 'selected_lr_by_amp_loss_improving.csv'}")


if __name__ == "__main__":
    main()
