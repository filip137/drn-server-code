#!/usr/bin/env python3
"""Collect Conv2/Conv3 v1/c1 perfect-diode gain/LR calibration runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


RESULT_COLUMNS = [
    "conv_depth",
    "non_linearity",
    "run_name",
    "seed",
    "input_gain",
    "learning_rate",
    "iteration_count",
    "epochs",
    "strides",
    "paddings",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "final_test_loss",
    "run_dir",
    "checkpoint_path",
    "weights_best_path",
    "weights_final_path",
    "pruned",
    "pruned_after_epoch",
    "prune_min_best_test_accuracy",
    "best_test_accuracy_so_far",
]

SELECTED_COLUMNS = [
    "conv_depth",
    "non_linearity",
    "run_name",
    "seed",
    "selected_input_gain",
    "selected_lr",
    "selected_iteration_count",
    "selected_epochs",
    "strides",
    "paddings",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "final_test_loss",
    "num_candidates",
    "selection_rule",
    "run_dir",
    "checkpoint_path",
    "weights_best_path",
    "weights_final_path",
]


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


def _first_lr(config: dict) -> float:
    value = config.get("optimizer", {}).get("learning_rate", math.nan)
    if isinstance(value, list):
        return _float(value[0] if value else math.nan)
    return _float(value)


def _row(metrics_path: Path) -> dict | None:
    run_dir = metrics_path.parent
    metrics = _read_json(metrics_path) or {}
    config = _read_json(run_dir / "config.json")
    if not config:
        return None

    architecture = config.get("architecture", {})
    training = config.get("training", {})
    conv_pipeline = architecture.get("conv_pipeline", [])
    conv_depth = len(conv_pipeline)
    run_name = run_dir.parent.name
    non_linearity = architecture.get("non_linearity", "")
    if run_name != "mnist_bp_amp_v1_c1" or non_linearity != "perfect_diode":
        return None
    if conv_depth not in (2, 3):
        return None

    return {
        "conv_depth": conv_depth,
        "non_linearity": non_linearity,
        "run_name": run_name,
        "seed": config.get("seed", ""),
        "input_gain": architecture.get("input_gain", ""),
        "learning_rate": _first_lr(config),
        "iteration_count": training.get("num_iterations_training", ""),
        "epochs": training.get("epochs", ""),
        "strides": " ".join(str(item.get("stride")) for item in conv_pipeline),
        "paddings": " ".join(str(item.get("padding")) for item in conv_pipeline),
        "best_test_accuracy": metrics.get("best_test_accuracy", ""),
        "final_test_accuracy": metrics.get("final_test_accuracy", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "final_train_loss": metrics.get("final_train_loss", ""),
        "final_test_loss": metrics.get("final_test_loss", ""),
        "run_dir": str(run_dir),
        "checkpoint_path": metrics.get("checkpoint_path", ""),
        "weights_best_path": metrics.get("weights_best_path", ""),
        "weights_final_path": metrics.get("weights_final_path", ""),
        "pruned": bool(metrics.get("pruned", False)),
        "pruned_after_epoch": metrics.get("pruned_after_epoch", ""),
        "prune_min_best_test_accuracy": metrics.get("prune_min_best_test_accuracy", ""),
        "best_test_accuracy_so_far": metrics.get("best_test_accuracy_so_far", ""),
    }


def _selection_key(row: dict) -> tuple:
    best = _float(row.get("best_test_accuracy"), -math.inf)
    final = _float(row.get("final_test_accuracy"), -math.inf)
    final_loss = _float(row.get("final_test_loss"), math.inf)
    input_gain = _float(row.get("input_gain"), math.inf)
    lr = _float(row.get("learning_rate"), math.inf)
    # Maximize accuracy; for ties prefer lower loss, then the milder operating point.
    return (best, final, -final_loss, -input_gain, -lr)


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect(root: Path) -> tuple[list[dict], list[dict]]:
    rows = [
        row
        for metrics_path in sorted(root.glob("**/metrics.json"))
        if (row := _row(metrics_path)) is not None
    ]
    rows.sort(
        key=lambda row: (
            int(row["conv_depth"]),
            _float(row["input_gain"]),
            _float(row["learning_rate"]),
        )
    )

    selected = []
    for conv_depth in (2, 3):
        candidates = [
            row
            for row in rows
            if int(row["conv_depth"]) == conv_depth and not row.get("pruned", False)
        ]
        if not candidates:
            continue
        best = max(candidates, key=_selection_key)
        selected.append(
            {
                "conv_depth": best["conv_depth"],
                "non_linearity": best["non_linearity"],
                "run_name": best["run_name"],
                "seed": best["seed"],
                "selected_input_gain": best["input_gain"],
                "selected_lr": best["learning_rate"],
                "selected_iteration_count": best["iteration_count"],
                "selected_epochs": best["epochs"],
                "strides": best["strides"],
                "paddings": best["paddings"],
                "best_test_accuracy": best["best_test_accuracy"],
                "final_test_accuracy": best["final_test_accuracy"],
                "best_epoch": best["best_epoch"],
                "final_train_loss": best["final_train_loss"],
                "final_test_loss": best["final_test_loss"],
                "num_candidates": len(candidates),
                "selection_rule": (
                    "maximize best_test_accuracy, then final_test_accuracy, "
                    "then lower final_test_loss, lower input_gain, lower lr"
                ),
                "run_dir": best["run_dir"],
                "checkpoint_path": best["checkpoint_path"],
                "weights_best_path": best["weights_best_path"],
                "weights_final_path": best["weights_final_path"],
            }
        )
    return rows, selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Gain/LR grid result root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    rows, selected = collect(root)
    _write_csv(root / "grid_results.csv", RESULT_COLUMNS, rows)
    _write_csv(root / "selected_gain_lr_by_conv_depth.csv", SELECTED_COLUMNS, selected)
    print(f"[collect] complete_metrics={len(rows)}")
    print(f"[collect] selected_rows={len(selected)}")
    if len(selected) != 2:
        raise SystemExit("Expected selected rows for Conv2 and Conv3.")
    for row in selected:
        print(
            "[collect] conv{conv_depth} gain={selected_input_gain} lr={selected_lr} "
            "best={best_test_accuracy} final={final_test_accuracy}".format(**row)
        )


if __name__ == "__main__":
    main()
