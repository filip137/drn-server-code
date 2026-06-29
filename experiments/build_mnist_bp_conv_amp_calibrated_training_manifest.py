#!/usr/bin/env python3
"""Build a row-per-training-job manifest from amp-aware saturation targets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


RUN_ORDER = [
    "mnist_bp_amp_v1_c1",
    "mnist_bp_amp_v2_c1",
    "mnist_bp_amp_v4_c1",
    "mnist_bp_amp_v1_c2",
    "mnist_bp_amp_v1_c4",
    "mnist_bp_amp_v4_c0p25",
    "mnist_bp_amp_v2_c2",
]


def _float_label(value: float) -> str:
    text = f"{value:.12g}"
    return text.replace("-", "m").replace(".", "p")


def _target_label(target: float) -> str:
    return f"sat{target * 100:.0f}"


def _read_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-csv", nargs="+", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--targets", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--sat50-lr-multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--default-lr-multipliers", type=float, nargs="+", default=[1.0])
    args = parser.parse_args()

    target_set = {round(value, 8) for value in args.targets}
    calibration_paths = [Path(path).expanduser().resolve() for path in args.calibration_csv]
    output_root = Path(args.output_root).expanduser()
    rows = []
    for row in _read_rows(calibration_paths):
        target = round(float(row["target_saturation"]), 8)
        if target not in target_set:
            continue
        multipliers = args.sat50_lr_multipliers if abs(target - 0.5) < 1e-8 else args.default_lr_multipliers
        for lr_multiplier in multipliers:
            base_lr = float(row["learning_rate"])
            learning_rate = base_lr * float(lr_multiplier)
            target_name = _target_label(target)
            gain_label = _float_label(float(row["input_gain"]))
            lr_label = _float_label(learning_rate)
            mult_label = _float_label(float(lr_multiplier))
            job_root = (
                output_root
                / f"target_{target_name}"
                / f"input_gain_{gain_label}"
                / f"lr_mult_{mult_label}"
                / f"lr_{lr_label}"
            )
            rows.append(
                {
                    "job_index": len(rows),
                    "run_name": row["run_name"],
                    "voltage_amp": row["voltage_amp"],
                    "current_amp": row["current_amp"],
                    "target_saturation": row["target_saturation"],
                    "target_label": target_name,
                    "input_gain": row["input_gain"],
                    "measured_saturation": row["measured_saturation"],
                    "layer_saturations": row["layer_saturations"],
                    "base_learning_rate": row["learning_rate"],
                    "lr_multiplier": f"{float(lr_multiplier):.12g}",
                    "learning_rate": f"{learning_rate:.12g}",
                    "v_off": row["v_off"],
                    "lr_numerator": row["lr_numerator"],
                    "seed": row["seed"],
                    "num_samples": row["num_samples"],
                    "num_iterations": row["num_iterations"],
                    "conv_depth": row["conv_depth"],
                    "padding": row["padding"],
                    "saturation_scope": row["saturation_scope"],
                    "job_root": str(job_root),
                }
            )

    run_rank = {name: idx for idx, name in enumerate(RUN_ORDER)}
    rows.sort(
        key=lambda row: (
            float(row["target_saturation"]),
            run_rank.get(row["run_name"], 999),
            float(row["lr_multiplier"]),
        )
    )
    for idx, row in enumerate(rows):
        row["job_index"] = idx

    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("No manifest rows produced.")
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[manifest] wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
