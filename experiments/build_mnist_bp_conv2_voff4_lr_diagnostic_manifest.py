#!/usr/bin/env python3
"""Build manifests for the Conv2 v_off=4 LR diagnostic screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


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

RUN_ORDER = [
    "mnist_bp_amp_v1_c1",
    "mnist_bp_amp_v2_c1",
    "mnist_bp_amp_v4_c1",
    "mnist_bp_amp_v1_c2",
    "mnist_bp_amp_v1_c4",
    "mnist_bp_amp_v4_c0p25",
    "mnist_bp_amp_v2_c2",
]

BASE_LR_MULTIPLIERS = [0.5, 2.0]
EXTRA_LR_MULTIPLIERS = {
    ("mnist_bp_amp_v4_c1", "sat10"): [4.0],
    ("mnist_bp_amp_v4_c1", "sat50"): [4.0],
    ("mnist_bp_amp_v4_c0p25", "sat30"): [0.25],
    ("mnist_bp_amp_v4_c0p25", "sat70"): [0.25],
}
LONG_CONTROLS = {
    ("mnist_bp_amp_v4_c1", "sat10"),
    ("mnist_bp_amp_v4_c1", "sat50"),
    ("mnist_bp_amp_v1_c4", "sat50"),
    ("mnist_bp_amp_v4_c0p25", "sat30"),
}

OUTPUT_COLUMNS = [
    "job_index",
    "run_name",
    "voltage_amp",
    "current_amp",
    "target_saturation",
    "target_label",
    "input_gain",
    "base_learning_rate",
    "lr_multiplier",
    "learning_rate",
    "v_off",
    "seed",
    "num_iterations",
    "conv_depth",
    "padding",
    "baseline_run_dir",
    "job_root",
]


def _float_label(value: float) -> str:
    text = f"{float(value):.12g}"
    return text.replace("-", "m").replace(".", "p")


def _path_part(run_dir: Path, prefix: str) -> str:
    for part in reversed(run_dir.parts):
        if part.startswith(prefix):
            return part[len(prefix) :]
    return ""


def _target_from_label(label: str) -> float:
    if label.startswith("sat"):
        try:
            return int(label[3:]) / 100.0
        except ValueError:
            return math.nan
    return math.nan


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _model_cfg(config: dict) -> dict:
    model_key = config.get("lab", {}).get("model_key", "mnist_bp_conv_amp")
    return {
        **config.get("model_base", {}),
        **config.get("model_overrides", {}).get(model_key, {}),
    }


def _first_lr(config: dict) -> float:
    lr = config.get("lr", [])
    if isinstance(lr, (int, float)):
        return float(lr)
    if not lr:
        raise ValueError("source_config.json has no lr values")
    return float(lr[0])


def _discover_baselines(baseline_root: Path) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in sorted(baseline_root.glob("**/metrics.json")):
        run_dir = metrics_path.parent
        source_config = _load_json(run_dir / "source_config.json")
        model_cfg = _model_cfg(source_config)
        conv_pipeline = model_cfg.get("conv_pipeline") or []
        target_label = _path_part(run_dir, "target_")
        if (
            len(conv_pipeline) != 2
            or model_cfg.get("non_linearity") != "hard_sigmoid"
            or int(source_config.get("seed", -1)) != 0
            or abs(float(model_cfg.get("hard_sigmoid_param", {}).get("v_max", math.nan)) - 4.0) > 1e-9
            or target_label not in {"sat10", "sat30", "sat50", "sat70"}
        ):
            continue
        run_name = run_dir.parent.name
        lr = _first_lr(source_config)
        rows.append(
            {
                "run_name": run_name,
                "voltage_amp": float(model_cfg.get("voltage_amp", math.nan)),
                "current_amp": float(model_cfg.get("current_amp", math.nan)),
                "target_saturation": _target_from_label(target_label),
                "target_label": target_label,
                "input_gain": float(model_cfg["input_gain"]),
                "base_learning_rate": lr,
                "v_off": float(model_cfg["hard_sigmoid_param"]["v_max"]),
                "seed": int(source_config.get("seed", 0)),
                "num_iterations": int(model_cfg["num_iterations_training"]),
                "conv_depth": len(conv_pipeline),
                "padding": int(conv_pipeline[0].get("padding", 0)),
                "baseline_run_dir": str(run_dir),
            }
        )
    run_rank = {name: index for index, name in enumerate(RUN_ORDER)}
    rows.sort(
        key=lambda row: (
            float(row["target_saturation"]),
            run_rank.get(str(row["run_name"]), 999),
        )
    )
    return rows


def _job_root(root: Path, base: dict, learning_rate: float, lr_multiplier: float) -> Path:
    return (
        root
        / f"target_{base['target_label']}"
        / f"input_gain_{_float_label(base['input_gain'])}"
        / f"lr_mult_{_float_label(lr_multiplier)}"
        / f"lr_{_float_label(learning_rate)}"
    )


def _make_row(base: dict, output_root: Path, lr_multiplier: float) -> dict:
    learning_rate = float(base["base_learning_rate"]) * float(lr_multiplier)
    return {
        **base,
        "lr_multiplier": f"{float(lr_multiplier):.12g}",
        "learning_rate": f"{learning_rate:.12g}",
        "job_root": str(_job_root(output_root, base, learning_rate, lr_multiplier)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _assign_indices(rows: list[dict]) -> list[dict]:
    indexed = []
    for index, row in enumerate(rows):
        indexed.append({"job_index": index, **row})
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--lr-screen-root", default=str(DEFAULT_LR_ROOT))
    parser.add_argument("--longer-root", default=str(DEFAULT_LONG_ROOT))
    parser.add_argument(
        "--lr-screen-manifest",
        default=str(DEFAULT_LR_ROOT / "manifests" / "lr_screen_manifest.csv"),
    )
    parser.add_argument(
        "--longer-manifest",
        default=str(DEFAULT_LONG_ROOT / "manifests" / "longer_x1_manifest.csv"),
    )
    args = parser.parse_args()

    baseline_root = Path(args.baseline_root).expanduser().resolve()
    lr_screen_root = Path(args.lr_screen_root).expanduser()
    longer_root = Path(args.longer_root).expanduser()
    baselines = _discover_baselines(baseline_root)
    if len(baselines) != 14:
        raise SystemExit(
            "Expected 14 completed baseline rows under "
            f"{baseline_root}, found {len(baselines)}."
        )

    lr_rows = []
    long_rows = []
    for base in baselines:
        key = (str(base["run_name"]), str(base["target_label"]))
        for lr_multiplier in BASE_LR_MULTIPLIERS + EXTRA_LR_MULTIPLIERS.get(key, []):
            lr_rows.append(_make_row(base, lr_screen_root, lr_multiplier))
        if key in LONG_CONTROLS:
            long_rows.append(_make_row(base, longer_root, 1.0))

    lr_rows = _assign_indices(lr_rows)
    long_rows = _assign_indices(long_rows)
    _write_csv(Path(args.lr_screen_manifest).expanduser().resolve(), lr_rows)
    _write_csv(Path(args.longer_manifest).expanduser().resolve(), long_rows)
    print(f"[manifest] baseline_rows={len(baselines)}")
    print(f"[manifest] lr_screen_rows={len(lr_rows)} path={Path(args.lr_screen_manifest).expanduser().resolve()}")
    print(f"[manifest] longer_rows={len(long_rows)} path={Path(args.longer_manifest).expanduser().resolve()}")


if __name__ == "__main__":
    main()
