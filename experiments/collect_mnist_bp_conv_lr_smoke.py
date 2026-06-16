#!/usr/bin/env python3
"""Collect conv MNIST LR-smoke results across lr_* output roots."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


ROW_COLUMNS = [
    "lr_label",
    "lr",
    "non_linearity",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "final_test_loss",
    "run_dir",
    "checkpoint_path",
    "weights_best_path",
    "weights_final_path",
]

GROUP_COLUMNS = [
    "lr_label",
    "lr",
    "non_linearity",
    "run_name",
    "voltage_amp",
    "current_amp",
    "num_runs",
    "mean_best_test_accuracy",
    "mean_final_test_accuracy",
    "mean_final_train_loss",
    "mean_final_test_loss",
]

SELECTED_COLUMNS = [
    "non_linearity",
    "run_name",
    "voltage_amp",
    "current_amp",
    "selected_lr_label",
    "selected_lr",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "final_test_loss",
    "run_dir",
    "checkpoint_path",
    "weights_best_path",
    "weights_final_path",
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _lr_label_value(path: Path) -> tuple[str, float | str]:
    label = path.name.removeprefix("lr_")
    try:
        value = float(label.replace("p", "."))
    except ValueError:
        value = label
    return label, value


def _rows(root: Path) -> list[dict]:
    rows = []
    for lr_root in sorted(root.glob("lr_*")):
        if not lr_root.is_dir():
            continue
        lr_label, lr_value = _lr_label_value(lr_root)
        for metrics_path in sorted(lr_root.glob("*/*/seed_*/metrics.json")):
            run_dir = metrics_path.parent
            metrics = _read_json(metrics_path)
            config = _read_json(run_dir / "config.json")
            rows.append(
                {
                    "lr_label": lr_label,
                    "lr": lr_value,
                    "non_linearity": config["architecture"]["non_linearity"],
                    "run_name": run_dir.parent.name,
                    "seed": config["seed"],
                    "voltage_amp": config["amplification"]["voltage_amp"],
                    "current_amp": config["amplification"]["current_amp"],
                    "best_test_accuracy": metrics.get("best_test_accuracy"),
                    "final_test_accuracy": metrics.get("final_test_accuracy"),
                    "best_epoch": metrics.get("best_epoch"),
                    "final_train_loss": metrics.get("final_train_loss"),
                    "final_test_loss": metrics.get("final_test_loss"),
                    "run_dir": str(run_dir),
                    "checkpoint_path": metrics.get("checkpoint_path"),
                    "weights_best_path": metrics.get("weights_best_path"),
                    "weights_final_path": metrics.get("weights_final_path"),
                }
            )
    return rows


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float | str:
    return statistics.fmean(values) if values else ""


def _group_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["lr_label"],
            row["lr"],
            row["non_linearity"],
            row["run_name"],
            float(row["voltage_amp"]),
            float(row["current_amp"]),
        )
        grouped.setdefault(key, []).append(row)

    out = []
    for key, group in sorted(grouped.items()):
        lr_label, lr, non_linearity, run_name, voltage_amp, current_amp = key
        out.append(
            {
                "lr_label": lr_label,
                "lr": lr,
                "non_linearity": non_linearity,
                "run_name": run_name,
                "voltage_amp": voltage_amp,
                "current_amp": current_amp,
                "num_runs": len(group),
                "mean_best_test_accuracy": _mean(
                    [float(row["best_test_accuracy"]) for row in group if row["best_test_accuracy"]]
                ),
                "mean_final_test_accuracy": _mean(
                    [float(row["final_test_accuracy"]) for row in group if row["final_test_accuracy"]]
                ),
                "mean_final_train_loss": _mean(
                    [float(row["final_train_loss"]) for row in group if row["final_train_loss"]]
                ),
                "mean_final_test_loss": _mean(
                    [float(row["final_test_loss"]) for row in group if row["final_test_loss"]]
                ),
            }
        )
    return out


def _selected_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, float, float], list[dict]] = {}
    for row in rows:
        key = (
            row["non_linearity"],
            row["run_name"],
            float(row["voltage_amp"]),
            float(row["current_amp"]),
        )
        grouped.setdefault(key, []).append(row)

    def score(row: dict) -> tuple[float, float, float]:
        # Prefer best accuracy, then lower final test loss, then lower LR.
        return (
            float(row["best_test_accuracy"]),
            -float(row["final_test_loss"]),
            -float(row["lr"]),
        )

    out = []
    for key, group in sorted(grouped.items()):
        non_linearity, run_name, voltage_amp, current_amp = key
        best = max(group, key=score)
        out.append(
            {
                "non_linearity": non_linearity,
                "run_name": run_name,
                "voltage_amp": voltage_amp,
                "current_amp": current_amp,
                "selected_lr_label": best["lr_label"],
                "selected_lr": best["lr"],
                "best_test_accuracy": best["best_test_accuracy"],
                "final_test_accuracy": best["final_test_accuracy"],
                "best_epoch": best["best_epoch"],
                "final_train_loss": best["final_train_loss"],
                "final_test_loss": best["final_test_loss"],
                "run_dir": best["run_dir"],
                "checkpoint_path": best["checkpoint_path"],
                "weights_best_path": best["weights_best_path"],
                "weights_final_path": best["weights_final_path"],
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    rows = _rows(root)
    all_path = root / "summary_all_lrs.csv"
    grouped_path = root / "summary_by_lr_nonlinearity_amp.csv"
    selected_path = root / "selected_best_by_run.csv"
    _write_csv(all_path, ROW_COLUMNS, rows)
    _write_csv(grouped_path, GROUP_COLUMNS, _group_rows(rows))
    _write_csv(selected_path, SELECTED_COLUMNS, _selected_rows(rows))
    print(f"[collect] runs={len(rows)} wrote {all_path}")
    print(f"[collect] wrote {grouped_path}")
    print(f"[collect] wrote {selected_path}")


if __name__ == "__main__":
    main()
