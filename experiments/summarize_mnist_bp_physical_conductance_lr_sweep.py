#!/usr/bin/env python3
"""Aggregate physical-conductance MNIST BP amplification LR sweep results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "mnist_bp_amp_physical_conductance_lr_sweep"
)

SUMMARY_COLUMNS = [
    "lr",
    "lr_label",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "final_test_loss",
    "checkpoint_path",
    "weights_best_path",
    "weights_final_path",
]

REQUIRED_RUN_FILES = [
    "final_model.pt",
    "best_model.pt",
    "weights_final.npz",
    "weights_best.npz",
    "metrics.json",
    "accuracy_train.npy",
    "accuracy_test.npy",
    "loss_train.npy",
    "loss_test.npy",
    "config.json",
]


def _read_lr_from_config(row: dict) -> float | None:
    checkpoint_path = row.get("checkpoint_path")
    if not checkpoint_path:
        return None
    config_path = Path(checkpoint_path).parent / "config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return None
    learning_rate = config.get("optimizer", {}).get("learning_rate", [])
    if not learning_rate:
        return None
    return float(learning_rate[0])


def _run_dir_from_row(row: dict) -> Path | None:
    for key in ("checkpoint_path", "weights_best_path", "weights_final_path"):
        value = row.get(key)
        if value:
            return Path(value).expanduser().resolve().parent
    return None


def _validate_rows(rows: list[dict], expected_rows: int, require_artifacts: bool) -> list[str]:
    errors = []
    if expected_rows >= 0 and len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {len(rows)}")

    if not require_artifacts:
        return errors

    for row in rows:
        label = f"{row.get('lr_label', '?')}/{row.get('run_name', '?')}/seed_{row.get('seed', '?')}"
        run_dir = _run_dir_from_row(row)
        if run_dir is None:
            errors.append(f"{label}: cannot infer run directory from summary paths")
            continue
        for filename in REQUIRED_RUN_FILES:
            path = run_dir / filename
            if not path.exists():
                errors.append(f"{label}: missing {path}")
    return errors


def aggregate(output_root: Path) -> tuple[Path, list[dict]]:
    rows = []
    for lr_dir in sorted(output_root.glob("lr_*")):
        summary_path = lr_dir / "summary.csv"
        if not summary_path.exists():
            continue
        with summary_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                lr = _read_lr_from_config(row)
                rows.append(
                    {
                        "lr": lr if lr is not None else "",
                        "lr_label": lr_dir.name.removeprefix("lr_"),
                        **row,
                    }
                )

    summary_path = output_root / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return summary_path, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=60,
        help="Expected number of completed rows. Use -1 to disable this check.",
    )
    parser.add_argument(
        "--allow-missing-artifacts",
        action="store_true",
        help="Write summary even if per-run artifact files are incomplete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    summary_path, rows = aggregate(output_root)
    print(f"[summary] wrote {summary_path}")
    print(f"[summary] rows={len(rows)}")

    errors = _validate_rows(
        rows,
        expected_rows=args.expected_rows,
        require_artifacts=not args.allow_missing_artifacts,
    )
    if errors:
        for error in errors:
            print(f"[summary:error] {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
