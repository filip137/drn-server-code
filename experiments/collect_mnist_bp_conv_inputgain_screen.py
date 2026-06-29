#!/usr/bin/env python3
"""Collect Conv DRN input-gain screen results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np


RUNS = [
    ("mnist_bp_amp_v1_c1", 1.0, 1.0),
    ("mnist_bp_amp_v2_c1", 2.0, 1.0),
    ("mnist_bp_amp_v4_c1", 4.0, 1.0),
    ("mnist_bp_amp_v1_c2", 1.0, 2.0),
    ("mnist_bp_amp_v1_c4", 1.0, 4.0),
]
RUNS_BY_NAME = {name: (name, voltage_amp, current_amp) for name, voltage_amp, current_amp in RUNS}

SUMMARY_COLUMNS = [
    "non_linearity",
    "input_gain",
    "lr",
    "iteration_count",
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
    "initial_train_loss",
    "train_loss_delta",
    "train_loss_decreased",
    "loss_still_decreasing",
    "collapsed",
    "run_dir",
    "checkpoint_path",
    "best_checkpoint_path",
    "weights_best_path",
    "weights_final_path",
]

GROUP_COLUMNS = [
    "non_linearity",
    "input_gain",
    "lr",
    "iteration_count",
    "num_expected",
    "num_complete",
    "mean_best_test_accuracy",
    "min_best_test_accuracy",
    "mean_final_test_accuracy",
    "mean_final_train_loss",
    "mean_final_test_loss",
    "all_train_loss_decreased",
    "num_loss_still_decreasing",
    "any_collapsed",
    "eligible",
]

SELECTED_COLUMNS = [
    "non_linearity",
    "selected_input_gain",
    "selected_lr",
    "selected_iteration_count",
    "eligible",
    "mean_best_test_accuracy",
    "min_best_test_accuracy",
    "mean_final_test_accuracy",
    "mean_final_train_loss",
    "num_complete",
    "num_expected",
    "selection_rule",
]

SELECTED_BY_AMP_COLUMNS = [
    "non_linearity",
    "run_name",
    "voltage_amp",
    "current_amp",
    "selected_input_gain",
    "lr",
    "iteration_count",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "train_loss_decreased",
    "loss_still_decreasing",
    "collapsed",
    "run_dir",
]


def _label_float(value: float | int) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


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


def _mean(values: list[float]) -> float | str:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else ""


def _loss_summary(run_dir: Path) -> tuple[float | str, float | str, bool | str, bool | str]:
    path = run_dir / "loss_train.npy"
    if not path.exists():
        return "", "", "", ""
    values = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "", "", "", ""
    initial = float(values[0])
    final = float(values[-1])
    return initial, final - initial, bool(final < initial), bool(values.size >= 2 and values[-1] < values[-2])


def _config_values(config: dict) -> tuple[str, float, float, int, int, float, float]:
    architecture = config["architecture"]
    optimizer = config["optimizer"]
    training = config["training"]
    amplification = config["amplification"]
    lr_values = optimizer.get("learning_rate", [])
    lr = float(lr_values[0] if isinstance(lr_values, list) else lr_values)
    return (
        str(architecture["non_linearity"]),
        float(architecture["input_gain"]),
        lr,
        int(training["num_iterations_training"]),
        int(config["seed"]),
        float(amplification["voltage_amp"]),
        float(amplification["current_amp"]),
    )


def _actual_rows(root: Path) -> dict[tuple[str, float, str, int], dict]:
    rows: dict[tuple[str, float, str, int], dict] = {}
    for metrics_path in sorted(root.glob("**/metrics.json")):
        run_dir = metrics_path.parent
        metrics = _read_json(metrics_path) or {}
        config = _read_json(run_dir / "config.json")
        if not config:
            continue
        non_linearity, input_gain, lr, iteration_count, seed, voltage_amp, current_amp = _config_values(config)
        run_name = run_dir.parent.name
        initial_loss, loss_delta, train_decreased, still_decreasing = _loss_summary(run_dir)
        best_acc = _float(metrics.get("best_test_accuracy"))
        collapsed = (not math.isfinite(best_acc)) or best_acc < 0.5
        row = {
            "non_linearity": non_linearity,
            "input_gain": input_gain,
            "lr": lr,
            "iteration_count": iteration_count,
            "run_name": run_name,
            "seed": seed,
            "voltage_amp": voltage_amp,
            "current_amp": current_amp,
            "status": "complete",
            "best_test_accuracy": metrics.get("best_test_accuracy", ""),
            "final_test_accuracy": metrics.get("final_test_accuracy", ""),
            "best_epoch": metrics.get("best_epoch", ""),
            "final_train_loss": metrics.get("final_train_loss", ""),
            "final_test_loss": metrics.get("final_test_loss", ""),
            "initial_train_loss": initial_loss,
            "train_loss_delta": loss_delta,
            "train_loss_decreased": train_decreased,
            "loss_still_decreasing": still_decreasing,
            "collapsed": collapsed,
            "run_dir": str(run_dir),
            "checkpoint_path": metrics.get("checkpoint_path", ""),
            "best_checkpoint_path": metrics.get("best_checkpoint_path", ""),
            "weights_best_path": metrics.get("weights_best_path", ""),
            "weights_final_path": metrics.get("weights_final_path", ""),
        }
        rows[(non_linearity, input_gain, run_name, seed)] = row
    return rows


def _requested_runs(args: argparse.Namespace) -> list[tuple[str, float, float]]:
    if not args.run_names:
        return RUNS
    return [RUNS_BY_NAME[name] for name in args.run_names]


def _expected_rows(args: argparse.Namespace, actual: dict[tuple[str, float, str, int], dict]) -> list[dict]:
    rows: list[dict] = []
    requested_runs = _requested_runs(args)
    for non_linearity in args.non_linearities:
        for input_gain in args.input_gains:
            for run_name, voltage_amp, current_amp in requested_runs:
                key = (non_linearity, float(input_gain), run_name, int(args.seed))
                if key in actual:
                    rows.append(actual[key])
                    continue
                rows.append(
                    {
                        "non_linearity": non_linearity,
                        "input_gain": float(input_gain),
                        "lr": "",
                        "iteration_count": "",
                        "run_name": run_name,
                        "seed": int(args.seed),
                        "voltage_amp": voltage_amp,
                        "current_amp": current_amp,
                        "status": "missing",
                        "best_test_accuracy": "",
                        "final_test_accuracy": "",
                        "best_epoch": "",
                        "final_train_loss": "",
                        "final_test_loss": "",
                        "initial_train_loss": "",
                        "train_loss_delta": "",
                        "train_loss_decreased": "",
                        "loss_still_decreasing": "",
                        "collapsed": "",
                        "run_dir": "",
                        "checkpoint_path": "",
                        "best_checkpoint_path": "",
                        "weights_best_path": "",
                        "weights_final_path": "",
                    }
                )
    return rows


def _group_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["non_linearity"], float(row["input_gain"])), []).append(row)
    out: list[dict] = []
    for (non_linearity, input_gain), group in sorted(grouped.items()):
        complete = [row for row in group if row["status"] == "complete"]
        best = [_float(row["best_test_accuracy"]) for row in complete]
        final = [_float(row["final_test_accuracy"]) for row in complete]
        train_loss = [_float(row["final_train_loss"]) for row in complete]
        test_loss = [_float(row["final_test_loss"]) for row in complete]
        all_decreased = bool(complete) and len(complete) == len(group) and all(
            row["train_loss_decreased"] is True for row in complete
        )
        any_collapsed = any(row["collapsed"] is True for row in complete)
        lr_values = sorted({_float(row["lr"]) for row in complete if math.isfinite(_float(row["lr"]))})
        k_values = sorted(
            {int(_float(row["iteration_count"])) for row in complete if math.isfinite(_float(row["iteration_count"]))}
        )
        out.append(
            {
                "non_linearity": non_linearity,
                "input_gain": input_gain,
                "lr": lr_values[0] if len(lr_values) == 1 else " ".join(f"{value:g}" for value in lr_values),
                "iteration_count": k_values[0] if len(k_values) == 1 else " ".join(str(value) for value in k_values),
                "num_expected": len(group),
                "num_complete": len(complete),
                "mean_best_test_accuracy": _mean(best),
                "min_best_test_accuracy": min([value for value in best if math.isfinite(value)], default=""),
                "mean_final_test_accuracy": _mean(final),
                "mean_final_train_loss": _mean(train_loss),
                "mean_final_test_loss": _mean(test_loss),
                "all_train_loss_decreased": all_decreased,
                "num_loss_still_decreasing": sum(row["loss_still_decreasing"] is True for row in complete),
                "any_collapsed": any_collapsed,
                "eligible": len(complete) == len(group) and all_decreased and not any_collapsed,
            }
        )
    return out


def _selected_by_nonlinearity(group_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in group_rows:
        grouped.setdefault(str(row["non_linearity"]), []).append(row)
    out: list[dict] = []
    for non_linearity, rows in sorted(grouped.items()):
        eligible = [row for row in rows if row["eligible"] is True]
        candidates = eligible or [row for row in rows if int(row["num_complete"]) == int(row["num_expected"])]
        candidates = candidates or rows

        def score(row: dict) -> tuple[float, float, float]:
            return (
                _float(row["mean_best_test_accuracy"], -math.inf),
                _float(row["min_best_test_accuracy"], -math.inf),
                -float(row["input_gain"]),
            )

        selected = max(candidates, key=score)
        out.append(
            {
                "non_linearity": non_linearity,
                "selected_input_gain": selected["input_gain"],
                "selected_lr": selected["lr"],
                "selected_iteration_count": selected["iteration_count"],
                "eligible": selected["eligible"],
                "mean_best_test_accuracy": selected["mean_best_test_accuracy"],
                "min_best_test_accuracy": selected["min_best_test_accuracy"],
                "mean_final_test_accuracy": selected["mean_final_test_accuracy"],
                "mean_final_train_loss": selected["mean_final_train_loss"],
                "num_complete": selected["num_complete"],
                "num_expected": selected["num_expected"],
                "selection_rule": (
                    "highest mean best-test accuracy across amplification settings; "
                    "tie min accuracy then lower input_gain; eligible excludes missing/collapsed/non-decreasing"
                ),
            }
        )
    return out


def _selected_by_amp(rows: list[dict]) -> list[dict]:
    complete = [row for row in rows if row["status"] == "complete"]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in complete:
        grouped.setdefault((row["non_linearity"], row["run_name"]), []).append(row)
    out: list[dict] = []
    for (non_linearity, run_name), group in sorted(grouped.items()):
        best = max(
            group,
            key=lambda row: (_float(row["best_test_accuracy"], -math.inf), -float(row["input_gain"])),
        )
        out.append(
            {
                "non_linearity": non_linearity,
                "run_name": run_name,
                "voltage_amp": best["voltage_amp"],
                "current_amp": best["current_amp"],
                "selected_input_gain": best["input_gain"],
                "lr": best["lr"],
                "iteration_count": best["iteration_count"],
                "best_test_accuracy": best["best_test_accuracy"],
                "final_test_accuracy": best["final_test_accuracy"],
                "best_epoch": best["best_epoch"],
                "final_train_loss": best["final_train_loss"],
                "train_loss_decreased": best["train_loss_decreased"],
                "loss_still_decreasing": best["loss_still_decreasing"],
                "collapsed": best["collapsed"],
                "run_dir": best["run_dir"],
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
    parser.add_argument("--input-gains", type=float, nargs="+", required=True)
    parser.add_argument("--non-linearities", nargs="+", default=["hard_sigmoid", "perfect_diode"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run-names",
        nargs="+",
        choices=sorted(RUNS_BY_NAME),
        help="Restrict expected rows to the listed amplification run names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    actual = _actual_rows(root)
    rows = _expected_rows(args, actual)
    group_rows = _group_rows(rows)
    selected = _selected_by_nonlinearity(group_rows)
    selected_amp = _selected_by_amp(rows)
    _write_csv(root / "summary.csv", SUMMARY_COLUMNS, rows)
    _write_csv(root / "summary_by_gain_nonlinearity_amp.csv", SUMMARY_COLUMNS, rows)
    _write_csv(root / "summary_by_gain_nonlinearity.csv", GROUP_COLUMNS, group_rows)
    _write_csv(root / "selected_input_gain_by_nonlinearity.csv", SELECTED_COLUMNS, selected)
    _write_csv(root / "selected_input_gain_by_nonlinearity_amp.csv", SELECTED_BY_AMP_COLUMNS, selected_amp)
    print(f"[collect-input-gain] rows={len(rows)} actual={len(actual)} root={root}")


if __name__ == "__main__":
    main()
