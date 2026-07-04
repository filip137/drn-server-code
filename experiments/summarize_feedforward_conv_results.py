#!/usr/bin/env python3
"""Summarize feed-forward ConvNet result roots into CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


RAW_COLUMNS = [
    "loss",
    "dataset",
    "activation",
    "conv_depth",
    "seed",
    "host",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_test_loss",
    "final_train_accuracy",
    "final_train_loss",
    "rotation_degrees",
    "rotation_seed",
    "translation_fraction",
    "scale_min",
    "scale_max",
    "shear_degrees",
    "run_dir",
]
SUMMARY_COLUMNS = [
    "loss",
    "dataset",
    "activation",
    "conv_depth",
    "num_seeds",
    "seeds",
    "mean_best_test_accuracy",
    "std_best_test_accuracy",
    "mean_final_test_accuracy",
    "std_final_test_accuracy",
    "mean_best_epoch",
    "mean_final_test_loss",
    "rotation_degrees",
    "rotation_seed",
    "translation_fraction",
    "scale_min",
    "scale_max",
    "shear_degrees",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--host", default="")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _float(value: object) -> float:
    return float(value)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _row_from_metrics(path: Path, host: str) -> dict:
    metrics = _load_json(path)
    return {
        "loss": metrics.get("loss", ""),
        "dataset": metrics.get("dataset", ""),
        "activation": metrics.get("activation", ""),
        "conv_depth": int(metrics.get("conv_depth", -1)),
        "seed": int(metrics.get("seed", -1)),
        "host": host,
        "best_test_accuracy": metrics.get("best_test_accuracy", ""),
        "final_test_accuracy": metrics.get("final_test_accuracy", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "final_test_loss": metrics.get("final_test_loss", ""),
        "final_train_accuracy": metrics.get("final_train_accuracy", ""),
        "final_train_loss": metrics.get("final_train_loss", ""),
        "rotation_degrees": metrics.get("rotation_degrees", ""),
        "rotation_seed": metrics.get("rotation_seed", ""),
        "translation_fraction": metrics.get("translation_fraction", ""),
        "scale_min": metrics.get("scale_min", ""),
        "scale_max": metrics.get("scale_max", ""),
        "shear_degrees": metrics.get("shear_degrees", ""),
        "run_dir": metrics.get("run_dir", str(path.parent)),
    }


def _discover_rows(output_root: Path, host: str) -> list[dict]:
    rows = [_row_from_metrics(path, host) for path in sorted(output_root.rglob("metrics.json"))]
    return sorted(
        rows,
        key=lambda row: (
            str(row["loss"]),
            str(row["dataset"]),
            str(row["activation"]),
            int(row["conv_depth"]),
            int(row["seed"]),
        ),
    )


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[object, ...], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["loss"], row["dataset"], row["activation"], row["conv_depth"])
        grouped[key].append(row)

    summary = []
    for (loss, dataset, activation, conv_depth), group in sorted(grouped.items()):
        seeds = sorted(int(row["seed"]) for row in group)
        best = [_float(row["best_test_accuracy"]) for row in group]
        final = [_float(row["final_test_accuracy"]) for row in group]
        best_epochs = [_float(row["best_epoch"]) for row in group]
        final_loss = [_float(row["final_test_loss"]) for row in group]
        rotation_degrees = sorted({str(row["rotation_degrees"]) for row in group})
        rotation_seed = sorted({str(row["rotation_seed"]) for row in group})
        translation_fraction = sorted({str(row["translation_fraction"]) for row in group})
        scale_min = sorted({str(row["scale_min"]) for row in group})
        scale_max = sorted({str(row["scale_max"]) for row in group})
        shear_degrees = sorted({str(row["shear_degrees"]) for row in group})
        summary.append(
            {
                "loss": loss,
                "dataset": dataset,
                "activation": activation,
                "conv_depth": conv_depth,
                "num_seeds": len(group),
                "seeds": " ".join(str(seed) for seed in seeds),
                "mean_best_test_accuracy": _mean(best),
                "std_best_test_accuracy": _std(best),
                "mean_final_test_accuracy": _mean(final),
                "std_final_test_accuracy": _std(final),
                "mean_best_epoch": _mean(best_epochs),
                "mean_final_test_loss": _mean(final_loss),
                "rotation_degrees": rotation_degrees[0] if len(rotation_degrees) == 1 else " ".join(rotation_degrees),
                "rotation_seed": rotation_seed[0] if len(rotation_seed) == 1 else " ".join(rotation_seed),
                "translation_fraction": translation_fraction[0]
                if len(translation_fraction) == 1
                else " ".join(translation_fraction),
                "scale_min": scale_min[0] if len(scale_min) == 1 else " ".join(scale_min),
                "scale_max": scale_max[0] if len(scale_max) == 1 else " ".join(scale_max),
                "shear_degrees": shear_degrees[0] if len(shear_degrees) == 1 else " ".join(shear_degrees),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.expanduser().resolve()
    rows = _discover_rows(output_root, args.host)
    summary = _aggregate(rows)
    raw_output = args.raw_output or output_root / "combined_raw_metrics.csv"
    summary_output = args.summary_output or output_root / "combined_summary_by_group.csv"
    _write_csv(raw_output, rows, RAW_COLUMNS)
    _write_csv(summary_output, summary, SUMMARY_COLUMNS)
    print(f"raw_rows={len(rows)} raw_output={raw_output}")
    print(f"summary_rows={len(summary)} summary_output={summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
