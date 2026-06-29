#!/usr/bin/env python3
"""Collect Hopfield EqProp Conv MNIST LR-screen results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SUMMARY_COLUMNS = [
    "conv_depth",
    "seed",
    "lr_multiplier",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "learning_rates",
    "run_dir",
    "metrics_path",
]
SELECTED_COLUMNS = [
    "conv_depth",
    "lr_multiplier",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "learning_rates",
    "source_run_dir",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _discover_metrics(input_roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in input_roots:
        root = root.expanduser().resolve()
        if root.name == "metrics.json":
            candidates = [root]
        else:
            candidates = sorted(root.glob("**/metrics.json"))
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)
    return paths


def _row(metrics_path: Path) -> dict:
    metrics = _load_json(metrics_path)
    return {
        "conv_depth": int(metrics.get("conv_depth", -1)),
        "seed": int(metrics.get("seed", -1)),
        "lr_multiplier": float(metrics.get("lr_multiplier", math.nan)),
        "best_test_accuracy": float(metrics.get("best_test_accuracy", math.nan)),
        "final_test_accuracy": float(metrics.get("final_test_accuracy", math.nan)),
        "best_epoch": metrics.get("best_epoch", ""),
        "learning_rates": json.dumps(metrics.get("learning_rates", [])),
        "run_dir": metrics.get("run_dir", str(metrics_path.parent)),
        "metrics_path": str(metrics_path),
    }


def _score(row: dict) -> tuple[float, float, float]:
    best = float(row["best_test_accuracy"])
    final = float(row["final_test_accuracy"])
    epoch = float(row["best_epoch"] or 0)
    return (
        best if math.isfinite(best) else -1.0,
        final if math.isfinite(final) else -1.0,
        -epoch,
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--selected-csv", required=True)
    args = parser.parse_args()

    rows = [_row(path) for path in _discover_metrics([Path(value) for value in args.input_root])]
    rows = [
        row
        for row in rows
        if row["conv_depth"] > 0 and math.isfinite(float(row["best_test_accuracy"]))
    ]
    rows.sort(key=lambda row: (row["conv_depth"], row["seed"], row["lr_multiplier"], row["run_dir"]))

    selected = []
    by_depth: dict[int, list[dict]] = {}
    for row in rows:
        by_depth.setdefault(int(row["conv_depth"]), []).append(row)
    for depth, group in sorted(by_depth.items()):
        best = max(group, key=_score)
        selected.append(
            {
                "conv_depth": depth,
                "lr_multiplier": best["lr_multiplier"],
                "best_test_accuracy": best["best_test_accuracy"],
                "final_test_accuracy": best["final_test_accuracy"],
                "best_epoch": best["best_epoch"],
                "learning_rates": best["learning_rates"],
                "source_run_dir": best["run_dir"],
            }
        )

    _write_csv(Path(args.summary_csv), SUMMARY_COLUMNS, rows)
    _write_csv(Path(args.selected_csv), SELECTED_COLUMNS, selected)
    print(
        f"[collect] rows={len(rows)} selected={len(selected)} "
        f"summary={args.summary_csv} selected_csv={args.selected_csv}"
    )


if __name__ == "__main__":
    main()
