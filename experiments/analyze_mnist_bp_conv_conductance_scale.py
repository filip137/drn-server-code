#!/usr/bin/env python3
"""Summarize learned conductance and bias scales for MNIST BP Conv runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


SKIP_NPZ_KEYS = {"param_names", "param_types", "param_shapes_json", "metadata_json"}
CHECKPOINTS = ("best", "final")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _model_cfg(config: dict[str, Any]) -> dict[str, Any]:
    model_key = config.get("lab", {}).get("model_key", "mnist_bp_conv_amp")
    return {
        **config.get("model_base", {}),
        **config.get("model_overrides", {}).get(model_key, {}),
    }


def _first_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return math.nan
    if isinstance(value, list):
        for item in value:
            found = _first_float(item)
            if math.isfinite(found):
                return found
        return math.nan
    if isinstance(value, dict):
        for item in value.values():
            found = _first_float(item)
            if math.isfinite(found):
                return found
        return math.nan
    return math.nan


def _parse_run_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw).expanduser()
        return path.name, path
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Expected non-empty label in --run LABEL=PATH, got: {raw}")
    return label, Path(path).expanduser()


def _run_name(run_dir: Path) -> str:
    if run_dir.parent.name.startswith("mnist_bp_amp_"):
        return run_dir.parent.name
    for part in reversed(run_dir.parts):
        if part.startswith("mnist_bp_amp_"):
            return part
    return ""


def _param_order(npz: np.lib.npyio.NpzFile) -> list[str]:
    if "param_names" in npz:
        return [str(name) for name in npz["param_names"].tolist()]
    return sorted(key for key in npz.files if key not in SKIP_NPZ_KEYS)


def _param_kind(name: str) -> str:
    if name.startswith(("ConvWeight_", "DenseWeight_")):
        return "inter_layer_conductance"
    if name.startswith("Bias_"):
        return "bias"
    return "other"


def _pre_layer_index(name: str, conv_depth: int) -> int | None:
    match = re.match(r"ConvWeight_(\d+)$", name)
    if match:
        return int(match.group(1))
    match = re.match(r"DenseWeight_(\d+)$", name)
    if match:
        return conv_depth + int(match.group(1))
    match = re.match(r"Bias_(\d+)$", name)
    if match:
        return int(match.group(1)) + 1
    return None


def _energy_interaction_factor(
    name: str,
    *,
    conv_depth: int,
    voltage_amp: float,
    current_amp: float,
) -> float:
    if not name.startswith(("ConvWeight_", "DenseWeight_")):
        return 1.0
    layer_index = _pre_layer_index(name, conv_depth)
    if layer_index is None:
        return math.nan
    if voltage_amp == 0:
        return math.nan
    return float((current_amp / voltage_amp) ** layer_index)


def _array_stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return {
            "numel": 0,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "min": math.nan,
            "p10": math.nan,
            "p90": math.nan,
            "p99": math.nan,
            "max": math.nan,
            "sum": math.nan,
            "l1_mean": math.nan,
            "l2_rms": math.nan,
            "zero_frac": math.nan,
        }
    return {
        "numel": int(flat.size),
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "p10": float(np.quantile(flat, 0.10)),
        "p90": float(np.quantile(flat, 0.90)),
        "p99": float(np.quantile(flat, 0.99)),
        "max": float(np.max(flat)),
        "sum": float(np.sum(flat)),
        "l1_mean": float(np.mean(np.abs(flat))),
        "l2_rms": float(np.sqrt(np.mean(flat * flat))),
        "zero_frac": float(np.mean(flat == 0.0)),
    }


def _rows_for_run(label: str, run_dir: Path) -> list[dict[str, Any]]:
    run_dir = run_dir.resolve()
    source_config = _load_json(run_dir / "source_config.json")
    metrics = _load_json(run_dir / "metrics.json")
    model_cfg = _model_cfg(source_config)
    conv_depth = len(model_cfg.get("conv_pipeline") or [])
    voltage_amp = float(model_cfg.get("voltage_amp", metrics.get("voltage_amp", math.nan)))
    current_amp = float(model_cfg.get("current_amp", metrics.get("current_amp", math.nan)))
    lr = _first_float(source_config.get("lr"))

    rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        npz_path = run_dir / f"weights_{checkpoint}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Expected weights file: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as npz:
            for name in _param_order(npz):
                if name in SKIP_NPZ_KEYS:
                    continue
                values = npz[name]
                if values.dtype.kind not in {"b", "i", "u", "f", "c"}:
                    continue
                stats = _array_stats(values)
                factor = _energy_interaction_factor(
                    name,
                    conv_depth=conv_depth,
                    voltage_amp=voltage_amp,
                    current_amp=current_amp,
                )
                layer_index = _pre_layer_index(name, conv_depth)
                rows.append(
                    {
                        "label": label,
                        "checkpoint": checkpoint,
                        "param": name,
                        "kind": _param_kind(name),
                        "shape": "x".join(str(dim) for dim in values.shape),
                        "layer_index": "" if layer_index is None else layer_index,
                        "energy_interaction_factor": factor,
                        "energy_scaled_mean": stats["mean"] * factor,
                        "energy_scaled_p90": stats["p90"] * factor,
                        "energy_scaled_sum": stats["sum"] * factor,
                        "run_name": _run_name(run_dir),
                        "conv_depth": conv_depth,
                        "non_linearity": model_cfg.get("non_linearity", ""),
                        "voltage_amp": voltage_amp,
                        "current_amp": current_amp,
                        "input_gain": model_cfg.get("input_gain", ""),
                        "lr": lr,
                        "epochs": source_config.get("lab", {}).get("epochs", ""),
                        "seed": source_config.get("seed", metrics.get("seed", "")),
                        "best_test_accuracy": metrics.get("best_test_accuracy", ""),
                        "final_test_accuracy": metrics.get("final_test_accuracy", ""),
                        "best_epoch": metrics.get("best_epoch", ""),
                        "run_dir": str(run_dir),
                        **stats,
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "checkpoint",
        "param",
        "kind",
        "shape",
        "layer_index",
        "energy_interaction_factor",
        "energy_scaled_mean",
        "energy_scaled_p90",
        "energy_scaled_sum",
        "run_name",
        "conv_depth",
        "non_linearity",
        "voltage_amp",
        "current_amp",
        "input_gain",
        "lr",
        "epochs",
        "seed",
        "best_test_accuracy",
        "final_test_accuracy",
        "best_epoch",
        "numel",
        "mean",
        "median",
        "std",
        "min",
        "p10",
        "p90",
        "p99",
        "max",
        "sum",
        "l1_mean",
        "l2_rms",
        "zero_frac",
        "run_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec as LABEL=RUN_DIR. Repeat for multiple runs.",
    )
    parser.add_argument("--output-csv", required=True, help="Destination CSV path.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for raw in args.run:
        label, run_dir = _parse_run_arg(raw)
        rows.extend(_rows_for_run(label, run_dir))
    _write_csv(Path(args.output_csv), rows)
    print(f"wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
