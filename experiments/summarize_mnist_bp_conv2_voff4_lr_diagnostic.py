#!/usr/bin/env python3
"""Summarize Conv2 v_off=4 LR diagnostic results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_ROOT = (
    REPO_ROOT
    / "results"
    / "mnist_bp_conv2_hardsigmoid_amp_saturation_targets_voff4_sat10_30_50_70_seed0_10epoch"
)
DEFAULT_LR_ROOT = (
    REPO_ROOT
    / "results"
    / "mnist_bp_conv2_hardsigmoid_voff4_existing_targets_lr_screen_seed0_10epoch"
)
DEFAULT_LONG_ROOT = (
    REPO_ROOT
    / "results"
    / "mnist_bp_conv2_hardsigmoid_voff4_existing_targets_longer_x1_seed0_30epoch"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "results"
    / "good_conv_saturation_analysis"
    / "conv2_voff4_lr_diagnostic_summary.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "results"
    / "good_conv_saturation_analysis"
    / "conv2_voff4_lr_diagnostic_summary.md"
)

RUN_ORDER = [
    "mnist_bp_amp_v1_c1",
    "mnist_bp_amp_v2_c1",
    "mnist_bp_amp_v4_c1",
    "mnist_bp_amp_v1_c2",
    "mnist_bp_amp_v1_c4",
    "mnist_bp_amp_v4_c0p25",
    "mnist_bp_amp_v2_c2",
]

OUTPUT_COLUMNS = [
    "control_type",
    "run_name",
    "target_label",
    "target_saturation",
    "lr_multiplier",
    "epochs",
    "input_gain",
    "learning_rate",
    "lr_times_input_gain",
    "best_test_accuracy",
    "final_test_accuracy",
    "best_test_accuracy_pct",
    "final_test_accuracy_pct",
    "best_epoch",
    "last3_test_accuracy_slope_pp_per_epoch",
    "final_hidden1_saturation",
    "final_hidden2_saturation",
    "final_all_hidden_saturation",
    "final_weight0_relative_grad_l2",
    "final_weight1_relative_grad_l2",
    "final_weight2_relative_grad_l2",
    "final_global_relative_grad_l2",
    "run_dir",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _model_cfg(config: dict) -> dict:
    model_key = config.get("lab", {}).get("model_key", "mnist_bp_conv_amp")
    return {
        **config.get("model_base", {}),
        **config.get("model_overrides", {}).get(model_key, {}),
    }


def _path_part(run_dir: Path, prefix: str) -> str:
    for part in reversed(run_dir.parts):
        if part.startswith(prefix):
            return part[len(prefix) :]
    return ""


def _label_to_float(value: str, default: float = math.nan) -> float:
    if not value:
        return default
    try:
        return float(value.replace("p", ".").replace("m", "-"))
    except ValueError:
        return default


def _target_from_label(label: str) -> float:
    if label.startswith("sat"):
        try:
            return int(label[3:]) / 100.0
        except ValueError:
            return math.nan
    return math.nan


def _first_lr(config: dict) -> float:
    lr = config.get("lr", [])
    if isinstance(lr, (int, float)):
        return float(lr)
    return float(lr[0]) if lr else math.nan


def _last3_slope(path: Path) -> float:
    if not path.exists():
        return math.nan
    values = np.asarray(np.load(path), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return math.nan
    return float((values[-1] - values[-3]) / 2.0 * 100.0)


def _key(row: dict) -> tuple:
    return (
        str(row.get("run_name", "")),
        str(row.get("target_label", "")),
        round(float(row.get("input_gain", math.nan)), 8),
        round(float(row.get("learning_rate", row.get("lr_reference", math.nan))), 14),
        int(float(row.get("epochs", 0) or 0)),
        round(float(row.get("lr_multiplier", 1.0)), 8),
    )


def _read_gradient_csv(path: Path | None) -> dict[tuple, dict]:
    if path is None or not path.exists():
        return {}
    result: dict[tuple, dict] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("phase") != "final":
                continue
            run_dir = Path(row.get("run_dir", ""))
            lr_mult = _label_to_float(_path_part(run_dir, "lr_mult_"), 1.0)
            learning_rate = float(row.get("lr_reference", math.nan))
            item = {
                "run_name": row.get("run_name", run_dir.parent.name),
                "target_label": row.get("target_label") or _path_part(run_dir, "target_"),
                "input_gain": float(row.get("input_gain", math.nan)),
                "learning_rate": learning_rate,
                "epochs": int(float(row.get("epochs", 0) or 0)),
                "lr_multiplier": lr_mult,
                "final_hidden1_saturation": float(row.get("hidden1_saturation", math.nan)),
                "final_hidden2_saturation": float(row.get("hidden2_saturation", math.nan)),
                "final_all_hidden_saturation": float(row.get("all_hidden_saturation", math.nan)),
                "final_weight0_relative_grad_l2": float(row.get("weight0_relative_grad_l2", math.nan)),
                "final_weight1_relative_grad_l2": float(row.get("weight1_relative_grad_l2", math.nan)),
                "final_weight2_relative_grad_l2": float(row.get("weight2_relative_grad_l2", math.nan)),
                "final_global_relative_grad_l2": float(row.get("global_relative_grad_l2", math.nan)),
            }
            result[_key(item)] = item
    return result


def _discover(root: Path, control_type: str) -> list[dict]:
    rows = []
    for metrics_path in sorted(root.glob("**/metrics.json")):
        run_dir = metrics_path.parent
        source_config = _load_json(run_dir / "source_config.json")
        metrics = _load_json(metrics_path)
        model_cfg = _model_cfg(source_config)
        conv_pipeline = model_cfg.get("conv_pipeline") or []
        target_label = _path_part(run_dir, "target_")
        if (
            len(conv_pipeline) != 2
            or model_cfg.get("non_linearity") != "hard_sigmoid"
            or target_label not in {"sat10", "sat30", "sat50", "sat70"}
            or abs(float(model_cfg.get("hard_sigmoid_param", {}).get("v_max", math.nan)) - 4.0) > 1e-9
        ):
            continue
        lr = _first_lr(source_config)
        input_gain = float(model_cfg.get("input_gain", math.nan))
        best = float(metrics.get("best_test_accuracy", math.nan))
        final = float(metrics.get("final_test_accuracy", math.nan))
        rows.append(
            {
                "control_type": control_type,
                "run_name": run_dir.parent.name,
                "target_label": target_label,
                "target_saturation": _target_from_label(target_label),
                "lr_multiplier": _label_to_float(_path_part(run_dir, "lr_mult_"), 1.0),
                "epochs": int(source_config.get("lab", {}).get("epochs", 0) or 0),
                "input_gain": input_gain,
                "learning_rate": lr,
                "lr_times_input_gain": lr * input_gain,
                "best_test_accuracy": best,
                "final_test_accuracy": final,
                "best_test_accuracy_pct": best * 100.0 if math.isfinite(best) else math.nan,
                "final_test_accuracy_pct": final * 100.0 if math.isfinite(final) else math.nan,
                "best_epoch": metrics.get("best_epoch", ""),
                "last3_test_accuracy_slope_pp_per_epoch": _last3_slope(run_dir / "accuracy_test.npy"),
                "run_dir": str(run_dir),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def _write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Conv2 v_off=4 LR Diagnostic",
        "",
        "| run | sat | lr_mult | epochs | best % | final % | slope pp/ep | h1 final | h2 final | W0 rel grad | W1 rel grad | W2 rel grad |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run_name"]),
                    str(row["target_label"]).replace("sat", ""),
                    _fmt(row["lr_multiplier"], 2),
                    str(row["epochs"]),
                    _fmt(row["best_test_accuracy_pct"], 2),
                    _fmt(row["final_test_accuracy_pct"], 2),
                    _fmt(row["last3_test_accuracy_slope_pp_per_epoch"], 3),
                    _fmt(row.get("final_hidden1_saturation"), 2),
                    _fmt(row.get("final_hidden2_saturation"), 2),
                    _fmt(row.get("final_weight0_relative_grad_l2"), 4),
                    _fmt(row.get("final_weight1_relative_grad_l2"), 4),
                    _fmt(row.get("final_weight2_relative_grad_l2"), 4),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--lr-screen-root", default=str(DEFAULT_LR_ROOT))
    parser.add_argument("--longer-root", default=str(DEFAULT_LONG_ROOT))
    parser.add_argument("--gradient-csv")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    args = parser.parse_args()

    rows = []
    rows.extend(_discover(Path(args.baseline_root).expanduser().resolve(), "baseline_x1_10epoch"))
    rows.extend(_discover(Path(args.lr_screen_root).expanduser().resolve(), "lr_screen_10epoch"))
    rows.extend(_discover(Path(args.longer_root).expanduser().resolve(), "longer_x1_30epoch"))

    gradients = _read_gradient_csv(Path(args.gradient_csv).expanduser().resolve() if args.gradient_csv else None)
    for row in rows:
        gradient = gradients.get(_key(row), {})
        for column in OUTPUT_COLUMNS:
            if column.startswith("final_") and column not in row:
                row[column] = gradient.get(column, math.nan)

    run_rank = {name: index for index, name in enumerate(RUN_ORDER)}
    rows.sort(
        key=lambda row: (
            run_rank.get(str(row["run_name"]), 999),
            float(row["target_saturation"]),
            int(row["epochs"]),
            float(row["lr_multiplier"]),
        )
    )
    _write_csv(Path(args.output_csv).expanduser().resolve(), rows)
    _write_markdown(Path(args.output_md).expanduser().resolve(), rows)
    print(f"[summary] rows={len(rows)} csv={Path(args.output_csv).expanduser().resolve()}")
    print(f"[summary] md={Path(args.output_md).expanduser().resolve()}")


if __name__ == "__main__":
    main()
