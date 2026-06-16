#!/usr/bin/env python3
"""Collect hard-sigmoid conv1 LR/input-gain probe results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


RUNS = [
    ("mnist_bp_amp_v1_c2", 1.0, 2.0),
    ("mnist_bp_amp_v1_c4", 1.0, 4.0),
]
LRS = [0.006, 0.009, 0.012]
INPUT_GAINS = [50.0, 100.0, 200.0]


def label_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p")


def read_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    rows = []
    for input_gain in INPUT_GAINS:
        for lr in LRS:
            subroot = root / f"input_gain_{label_float(input_gain)}" / f"lr_{label_float(lr)}"
            for run_name, voltage_amp, current_amp in RUNS:
                run_dir = subroot / "hard_sigmoid" / run_name / f"seed_{args.seed}"
                metrics = read_metrics(run_dir / "metrics.json")
                row = {
                    "input_gain": input_gain,
                    "lr": lr,
                    "non_linearity": "hard_sigmoid",
                    "run_name": run_name,
                    "seed": args.seed,
                    "voltage_amp": voltage_amp,
                    "current_amp": current_amp,
                    "status": "complete" if metrics is not None else "missing",
                    "best_test_accuracy": "",
                    "final_test_accuracy": "",
                    "best_epoch": "",
                    "final_train_loss": "",
                    "final_test_loss": "",
                    "checkpoint_path": "",
                    "weights_best_path": "",
                    "weights_final_path": "",
                    "run_dir": str(run_dir),
                }
                if metrics is not None:
                    row.update(
                        {
                            "best_test_accuracy": metrics.get("best_test_accuracy", ""),
                            "final_test_accuracy": metrics.get("final_test_accuracy", ""),
                            "best_epoch": metrics.get("best_epoch", ""),
                            "final_train_loss": metrics.get("final_train_loss", ""),
                            "final_test_loss": metrics.get("final_test_loss", ""),
                            "checkpoint_path": metrics.get("checkpoint_path", ""),
                            "weights_best_path": metrics.get("weights_best_path", ""),
                            "weights_final_path": metrics.get("weights_final_path", ""),
                        }
                    )
                rows.append(row)

    fieldnames = [
        "input_gain",
        "lr",
        "non_linearity",
        "run_name",
        "seed",
        "voltage_amp",
        "current_amp",
        "status",
        "best_test_accuracy",
        "final_test_accuracy",
        "best_epoch",
        "final_train_loss",
        "final_test_loss",
        "checkpoint_path",
        "weights_best_path",
        "weights_final_path",
        "run_dir",
    ]
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    complete = [row for row in rows if row["status"] == "complete"]
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in complete:
        grouped.setdefault((row["run_name"], float(row["input_gain"])), []).append(row)

    by_gain_rows = []
    for (run_name, input_gain), group in sorted(grouped.items()):
        finals = [float(row["final_test_accuracy"]) for row in group]
        bests = [float(row["best_test_accuracy"]) for row in group]
        best_row = max(group, key=lambda row: float(row["best_test_accuracy"]))
        by_gain_rows.append(
            {
                "run_name": run_name,
                "input_gain": input_gain,
                "num_lrs": len(group),
                "mean_best_test_accuracy": statistics.fmean(bests),
                "mean_final_test_accuracy": statistics.fmean(finals),
                "best_lr": best_row["lr"],
                "best_test_accuracy": best_row["best_test_accuracy"],
                "best_final_test_accuracy": best_row["final_test_accuracy"],
                "best_epoch": best_row["best_epoch"],
            }
        )

    by_gain_path = root / "summary_by_run_input_gain.csv"
    with by_gain_path.open("w", newline="") as handle:
        fieldnames_by_gain = [
            "run_name",
            "input_gain",
            "num_lrs",
            "mean_best_test_accuracy",
            "mean_final_test_accuracy",
            "best_lr",
            "best_test_accuracy",
            "best_final_test_accuracy",
            "best_epoch",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames_by_gain)
        writer.writeheader()
        writer.writerows(by_gain_rows)

    print(f"[collect] wrote {summary_path}")
    print(f"[collect] wrote {by_gain_path}")
    print(f"[collect] complete={len(complete)}/{len(rows)}")


if __name__ == "__main__":
    main()
