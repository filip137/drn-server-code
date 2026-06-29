#!/usr/bin/env python3
"""Write frozen Conv DRN final-run settings from selected LR rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLUMNS = [
    "non_linearity",
    "run_name",
    "input_gain",
    "iteration_count",
    "lr",
    "source_best_test_accuracy",
    "source_final_test_accuracy",
    "source_best_epoch",
    "source_run_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-lr-csv",
        required=True,
        help="CSV produced by collect_mnist_bp_conv_lr_screen.py.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_path = Path(args.selected_lr_csv).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    with selected_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "non_linearity": row["non_linearity"],
                "run_name": row["run_name"],
                "input_gain": row["input_gain"],
                "iteration_count": row["iteration_count"],
                "lr": row["selected_lr"],
                "source_best_test_accuracy": row.get("best_test_accuracy", ""),
                "source_final_test_accuracy": row.get("final_test_accuracy", ""),
                "source_best_epoch": row.get("best_epoch", ""),
                "source_run_dir": row.get("run_dir", ""),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[final-settings] rows={len(out_rows)} output={output_path}")


if __name__ == "__main__":
    main()
