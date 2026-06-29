#!/usr/bin/env python3
"""Collect hard-sigmoid conv input-gain training sweep results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RUNS = [
    ("mnist_bp_amp_v1_c1", 1.0, 1.0),
    ("mnist_bp_amp_v2_c1", 2.0, 1.0),
    ("mnist_bp_amp_v4_c1", 4.0, 1.0),
    ("mnist_bp_amp_v1_c2", 1.0, 2.0),
    ("mnist_bp_amp_v1_c4", 1.0, 4.0),
]
CONV1_GAINS = [10.0, 20.0, 30.0, 40.0, 50.0]
CONV2_GAINS = [25.0, 50.0, 75.0, 100.0]
CONV2_K = {
    "mnist_bp_amp_v1_c1": 12,
    "mnist_bp_amp_v2_c1": 8,
    "mnist_bp_amp_v4_c1": 12,
    "mnist_bp_amp_v1_c2": 16,
    "mnist_bp_amp_v1_c4": 16,
}


SUMMARY_COLUMNS = [
    "conv_depth",
    "input_gain",
    "iteration_count",
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
    "final_train_accuracy",
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

BY_DEPTH_GAIN_COLUMNS = [
    "conv_depth",
    "input_gain",
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
    "conv_depth",
    "selected_input_gain",
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
    "conv_depth",
    "run_name",
    "voltage_amp",
    "current_amp",
    "selected_input_gain",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_epoch",
    "final_train_loss",
    "train_loss_decreased",
    "loss_still_decreasing",
    "collapsed",
    "run_dir",
]


@dataclass(frozen=True)
class ExpectedRun:
    conv_depth: int
    input_gain: float
    iteration_count: int
    run_name: str
    voltage_amp: float
    current_amp: float
    seed: int = 0
    non_linearity: str = "hard_sigmoid"
    lr: float = 0.006


def _label_float(value: float | int) -> str:
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def _lr_for_gain(input_gain: float, *, lr_mode: str, fixed_lr: float, base_lr: float, base_gain: float) -> float:
    if lr_mode == "fixed":
        return fixed_lr
    if lr_mode == "gain_scaled":
        return base_lr * input_gain / base_gain
    raise ValueError(f"Expected lr_mode to be 'fixed' or 'gain_scaled', got {lr_mode!r}.")


def _expected_runs(
    *,
    lr_mode: str = "fixed",
    fixed_lr: float = 0.006,
    base_lr: float = 0.006,
    base_gain: float = 10.0,
) -> list[ExpectedRun]:
    expected: list[ExpectedRun] = []
    for gain in CONV1_GAINS:
        for run_name, voltage_amp, current_amp in RUNS:
            expected.append(
                ExpectedRun(
                    1,
                    gain,
                    4,
                    run_name,
                    voltage_amp,
                    current_amp,
                    lr=_lr_for_gain(gain, lr_mode=lr_mode, fixed_lr=fixed_lr, base_lr=base_lr, base_gain=base_gain),
                )
            )
    for gain in CONV2_GAINS:
        for run_name, voltage_amp, current_amp in RUNS:
            expected.append(
                ExpectedRun(
                    2,
                    gain,
                    CONV2_K[run_name],
                    run_name,
                    voltage_amp,
                    current_amp,
                    lr=_lr_for_gain(gain, lr_mode=lr_mode, fixed_lr=fixed_lr, base_lr=base_lr, base_gain=base_gain),
                )
            )
    return expected


def _run_dir(root: Path, spec: ExpectedRun, *, path_includes_lr: bool = False) -> Path:
    gain_label = _label_float(spec.input_gain)
    k_label = _label_float(spec.iteration_count)
    run_root = (
        root
        / f"conv{spec.conv_depth}"
        / f"input_gain_{gain_label}"
    )
    if path_includes_lr:
        run_root = run_root / f"lr_{_label_float(spec.lr)}"
    return run_root / f"k_{k_label}" / spec.non_linearity / spec.run_name / f"seed_{spec.seed}"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


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
    decreased = final < initial
    still_decreasing = bool(values.size >= 2 and values[-1] < values[-2])
    return initial, final - initial, decreased, still_decreasing


def _float(value: object, default: float = math.nan) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _row(root: Path, spec: ExpectedRun, *, path_includes_lr: bool = False) -> dict:
    run_dir = _run_dir(root, spec, path_includes_lr=path_includes_lr)
    metrics = _read_json(run_dir / "metrics.json")
    initial_loss, loss_delta, train_decreased, still_decreasing = _loss_summary(run_dir)
    row = {
        "conv_depth": spec.conv_depth,
        "input_gain": spec.input_gain,
        "iteration_count": spec.iteration_count,
        "lr": spec.lr,
        "non_linearity": spec.non_linearity,
        "run_name": spec.run_name,
        "seed": spec.seed,
        "voltage_amp": spec.voltage_amp,
        "current_amp": spec.current_amp,
        "status": "complete" if metrics is not None else "missing",
        "best_test_accuracy": "",
        "final_test_accuracy": "",
        "best_epoch": "",
        "final_train_accuracy": "",
        "final_train_loss": "",
        "final_test_loss": "",
        "initial_train_loss": initial_loss,
        "train_loss_delta": loss_delta,
        "train_loss_decreased": train_decreased,
        "loss_still_decreasing": still_decreasing,
        "collapsed": "",
        "run_dir": str(run_dir),
        "checkpoint_path": "",
        "best_checkpoint_path": "",
        "weights_best_path": "",
        "weights_final_path": "",
    }
    if metrics is None:
        return row

    best_acc = _float(metrics.get("best_test_accuracy"))
    collapsed = (not math.isfinite(best_acc)) or best_acc < 0.5
    row.update(
        {
            "best_test_accuracy": metrics.get("best_test_accuracy", ""),
            "final_test_accuracy": metrics.get("final_test_accuracy", ""),
            "best_epoch": metrics.get("best_epoch", ""),
            "final_train_accuracy": metrics.get("final_train_accuracy", ""),
            "final_train_loss": metrics.get("final_train_loss", ""),
            "final_test_loss": metrics.get("final_test_loss", ""),
            "collapsed": collapsed,
            "checkpoint_path": metrics.get("checkpoint_path", ""),
            "best_checkpoint_path": metrics.get("best_checkpoint_path", ""),
            "weights_best_path": metrics.get("weights_best_path", ""),
            "weights_final_path": metrics.get("weights_final_path", ""),
        }
    )
    return row


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float | str:
    return statistics.fmean(values) if values else ""


def _group_by_depth_gain(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["conv_depth"]), float(row["input_gain"])), []).append(row)

    out = []
    for (depth, gain), group in sorted(grouped.items()):
        complete = [row for row in group if row["status"] == "complete"]
        best = [_float(row["best_test_accuracy"]) for row in complete]
        best = [value for value in best if math.isfinite(value)]
        final = [_float(row["final_test_accuracy"]) for row in complete]
        final = [value for value in final if math.isfinite(value)]
        train_loss = [_float(row["final_train_loss"]) for row in complete]
        train_loss = [value for value in train_loss if math.isfinite(value)]
        test_loss = [_float(row["final_test_loss"]) for row in complete]
        test_loss = [value for value in test_loss if math.isfinite(value)]
        all_decreased = bool(complete) and len(complete) == len(group) and all(
            row["train_loss_decreased"] is True for row in complete
        )
        any_collapsed = any(row["collapsed"] is True for row in complete)
        eligible = len(complete) == len(group) and all_decreased and not any_collapsed
        out.append(
            {
                "conv_depth": depth,
                "input_gain": gain,
                "num_expected": len(group),
                "num_complete": len(complete),
                "mean_best_test_accuracy": _mean(best),
                "min_best_test_accuracy": min(best) if best else "",
                "mean_final_test_accuracy": _mean(final),
                "mean_final_train_loss": _mean(train_loss),
                "mean_final_test_loss": _mean(test_loss),
                "all_train_loss_decreased": all_decreased,
                "num_loss_still_decreasing": sum(row["loss_still_decreasing"] is True for row in complete),
                "any_collapsed": any_collapsed,
                "eligible": eligible,
            }
        )
    return out


def _group_by_depth_amp_gain(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (int(row["conv_depth"]), row["run_name"], float(row["input_gain"])))


def _selected_by_depth(grouped_rows: list[dict]) -> list[dict]:
    by_depth: dict[int, list[dict]] = {}
    for row in grouped_rows:
        by_depth.setdefault(int(row["conv_depth"]), []).append(row)

    out = []
    for depth, rows in sorted(by_depth.items()):
        eligible = [row for row in rows if row["eligible"] is True]
        candidates = eligible if eligible else [row for row in rows if int(row["num_complete"]) == int(row["num_expected"])]
        if not candidates:
            candidates = rows

        def score(row: dict) -> tuple[float, float, float]:
            return (
                _float(row["mean_best_test_accuracy"], -math.inf),
                _float(row["min_best_test_accuracy"], -math.inf),
                -float(row["input_gain"]),
            )

        selected = max(candidates, key=score)
        out.append(
            {
                "conv_depth": depth,
                "selected_input_gain": selected["input_gain"],
                "eligible": selected["eligible"],
                "mean_best_test_accuracy": selected["mean_best_test_accuracy"],
                "min_best_test_accuracy": selected["min_best_test_accuracy"],
                "mean_final_test_accuracy": selected["mean_final_test_accuracy"],
                "mean_final_train_loss": selected["mean_final_train_loss"],
                "num_complete": selected["num_complete"],
                "num_expected": selected["num_expected"],
                "selection_rule": (
                    "highest mean best-test accuracy, tie lower gain; "
                    "eligible requires all runs complete, decreasing train loss, no collapse"
                ),
            }
        )
    return out


def _selected_by_depth_amp(rows: list[dict]) -> list[dict]:
    complete = [row for row in rows if row["status"] == "complete"]
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in complete:
        grouped.setdefault((int(row["conv_depth"]), row["run_name"]), []).append(row)

    out = []
    for (depth, run_name), group in sorted(grouped.items()):
        def score(row: dict) -> tuple[float, float]:
            return (_float(row["best_test_accuracy"], -math.inf), -float(row["input_gain"]))

        best = max(group, key=score)
        out.append(
            {
                "conv_depth": depth,
                "run_name": run_name,
                "voltage_amp": best["voltage_amp"],
                "current_amp": best["current_amp"],
                "selected_input_gain": best["input_gain"],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--lr-mode", choices=["fixed", "gain_scaled"], default="fixed")
    parser.add_argument("--fixed-lr", type=float, default=0.006)
    parser.add_argument("--base-lr", type=float, default=0.006)
    parser.add_argument("--base-gain", type=float, default=10.0)
    parser.add_argument(
        "--path-includes-lr",
        action="store_true",
        help="Expect run directories to include an lr_<label> component after input_gain_<label>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    rows = [
        _row(root, spec, path_includes_lr=args.path_includes_lr)
        for spec in _expected_runs(
            lr_mode=args.lr_mode,
            fixed_lr=args.fixed_lr,
            base_lr=args.base_lr,
            base_gain=args.base_gain,
        )
    ]
    by_depth_gain = _group_by_depth_gain(rows)
    by_depth_amp_gain = _group_by_depth_amp_gain(rows)
    selected = _selected_by_depth(by_depth_gain)
    selected_by_amp = _selected_by_depth_amp(rows)

    _write_csv(root / "summary.csv", SUMMARY_COLUMNS, rows)
    _write_csv(root / "summary_by_depth_gain.csv", BY_DEPTH_GAIN_COLUMNS, by_depth_gain)
    _write_csv(root / "summary_by_depth_amp_gain.csv", SUMMARY_COLUMNS, by_depth_amp_gain)
    _write_csv(root / "selected_input_gain_by_depth.csv", SELECTED_COLUMNS, selected)
    _write_csv(root / "selected_input_gain_by_depth_amp.csv", SELECTED_BY_AMP_COLUMNS, selected_by_amp)
    complete = sum(row["status"] == "complete" for row in rows)
    print(f"[collect] complete={complete}/{len(rows)} root={root}")
    print(f"[collect] wrote {root / 'summary.csv'}")
    print(f"[collect] wrote {root / 'selected_input_gain_by_depth.csv'}")


if __name__ == "__main__":
    main()
