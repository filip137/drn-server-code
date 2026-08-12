#!/usr/bin/env python3
"""Diagnose LR adequacy in the bounded-wmax sweep without training new models.

The input is the pooled CSV produced by
``experiments.analyze_bounded_wmax_final_clipping``.  That upstream analyzer
validates the canonical run bundles and final checkpoints.  This script keeps
the analysis read-only, verifies the indexed ``metrics.jsonl`` digest again,
and concentrates on matched ``wmax=3e-4`` versus ``wmax=1e-3`` surfaces.

Endpoint occupancy is a useful boundary-pressure diagnostic, but it is not a
convergence test.  Consequently the output deliberately combines clipping
correlations with paired accuracy changes, late-epoch slopes, best-epoch
location, and the fraction of the available interval represented by the fixed
raw learning-rate vector.  Optional deterministic initialization replay adds
span-normalized final weight motion; it uses the same construction path as
``test_bounded_wmax_initializer_quantiles.py`` and never mutates a run bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ANALYSIS_SCHEMA = "perfectdiode-bounded-wmax-lr-adequacy-analysis/v1"
DEFAULT_LOW_WMAX = 3.0e-4
DEFAULT_HIGH_WMAX = 1.0e-3
WEIGHT_PREFIXES = ("ConvWeight_", "DenseWeight_")
ARCHITECTURE_COLORS = {
    "conv1": "#4c78a8",
    "conv2": "#f58518",
    "conv3": "#54a24b",
}
CLIPPING_FIELDS = (
    "exact_lower_endpoint_percent",
    "exact_upper_endpoint_percent",
    "exact_either_endpoint_percent",
)
ACCURACY_FIELDS = (
    "best_validation_accuracy_percent",
    "final_validation_accuracy_percent",
)


def _finite_float(value: Any, *, label: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected finite {label}, got {value!r}.") from error
    if not math.isfinite(output):
        raise ValueError(f"Expected finite {label}, got {value!r}.")
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def _load_pooled_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Pooled clipping CSV is empty: {path}.")

    required = {
        "arm_id",
        "architecture",
        "scheme",
        "optimizer",
        "weight_min",
        "weight_max",
        "learning_rate_vector_json",
        "run_dir",
        *CLIPPING_FIELDS,
        *ACCURACY_FIELDS,
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Pooled clipping CSV lacks columns: {missing!r}.")

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, float]] = set()
    for row in rows:
        if row.get("row_level") not in (None, "", "pooled"):
            raise ValueError(
                f"Expected only pooled rows, found {row.get('row_level')!r}."
            )
        try:
            learning_rates = json.loads(row["learning_rate_vector_json"])
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid learning-rate vector for {row['arm_id']!r}."
            ) from error
        if not isinstance(learning_rates, list) or not learning_rates:
            raise ValueError(f"Empty learning-rate vector for {row['arm_id']!r}.")
        learning_rates = [
            _finite_float(value, label="learning rate") for value in learning_rates
        ]
        if any(value <= 0.0 for value in learning_rates):
            raise ValueError(f"Non-positive learning rate for {row['arm_id']!r}.")

        parsed: dict[str, Any] = dict(row)
        for field in (*CLIPPING_FIELDS, *ACCURACY_FIELDS, "weight_min", "weight_max"):
            parsed[field] = _finite_float(row[field], label=field)
        parsed["optimizer"] = str(row["optimizer"]).lower()
        parsed["learning_rates"] = learning_rates
        parsed["run_dir"] = str(Path(row["run_dir"]).expanduser().resolve())
        key = (
            str(parsed["architecture"]),
            str(parsed["scheme"]),
            str(parsed["optimizer"]),
            float(parsed["weight_max"]),
        )
        if key in seen:
            raise ValueError(f"Duplicate pooled scientific point {key!r}.")
        seen.add(key)
        output.append(parsed)
    return output


def _metric_records(run_dir: Path, *, expected_arm_id: str) -> list[dict[str, float]]:
    manifest = _load_json(run_dir / "manifest.json")
    result = _load_json(run_dir / "result.json")
    if manifest.get("arm_id") != expected_arm_id or result.get("arm_id") != expected_arm_id:
        raise ValueError(f"arm_id mismatch in canonical run {run_dir}.")
    metrics_path = run_dir / "metrics.jsonl"
    expected_digest = result.get("metrics_sha256")
    if not isinstance(expected_digest, str) or _sha256(metrics_path) != expected_digest:
        raise ValueError(f"metrics.jsonl digest mismatch in {run_dir}.")

    records: list[dict[str, float]] = []
    for line_number, line in enumerate(
        metrics_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid metrics JSON at {metrics_path}:{line_number}."
            ) from error
        if record.get("kind") != "epoch":
            continue
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"Missing epoch metrics at {metrics_path}:{line_number}.")
        row = {
            "epoch": _finite_float(record.get("epoch"), label="epoch"),
            "best_epoch": _finite_float(
                metrics.get("best_epoch"), label="best epoch"
            ),
            "best_validation_accuracy": _finite_float(
                metrics.get("best_validation_accuracy"),
                label="best validation accuracy",
            ),
            "validation_accuracy": _finite_float(
                metrics.get("validation_accuracy"), label="validation accuracy"
            ),
            "train_accuracy": _finite_float(
                metrics.get("train_accuracy"), label="training accuracy"
            ),
            "validation_loss": _finite_float(
                metrics.get("validation_loss"), label="validation loss"
            ),
            "train_loss": _finite_float(metrics.get("train_loss"), label="training loss"),
        }
        records.append(row)
    if not records:
        raise ValueError(f"No epoch records in {metrics_path}.")
    epochs = [record["epoch"] for record in records]
    if epochs != sorted(epochs) or len(epochs) != len(set(epochs)):
        raise ValueError(f"Epoch records are not strictly ordered in {metrics_path}.")
    return records


def _weight_learning_rates(
    run_dir: Path, *, expected_arm_id: str
) -> dict[str, float]:
    manifest = _load_json(run_dir / "manifest.json")
    if manifest.get("arm_id") != expected_arm_id:
        raise ValueError(f"arm_id mismatch in canonical run {run_dir}.")
    configuration = manifest.get("configuration", {})
    resolved = configuration.get("resolved", {}) if isinstance(configuration, Mapping) else {}
    by_parameter = (
        resolved.get("learning_rates_by_parameter", {})
        if isinstance(resolved, Mapping)
        else {}
    )
    if not isinstance(by_parameter, Mapping):
        raise ValueError(f"Missing learning_rates_by_parameter in {run_dir}.")
    output = {
        str(name): _finite_float(value, label=f"learning rate for {name}")
        for name, value in by_parameter.items()
        if str(name).startswith(WEIGHT_PREFIXES)
    }
    if not output or any(value <= 0.0 for value in output.values()):
        raise ValueError(f"Missing or invalid bounded-weight learning rates in {run_dir}.")
    return dict(sorted(output.items()))


def _ols_slope(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("OLS slope requires at least two paired observations.")
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    centered = x_values - float(np.mean(x_values))
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        raise ValueError("OLS slope requires at least two distinct x values.")
    return float(np.dot(centered, y_values - float(np.mean(y_values))) / denominator)


def _trajectory_diagnostics(
    records: Sequence[Mapping[str, float]], *, final_window: int
) -> dict[str, Any]:
    if final_window < 2:
        raise ValueError("final_window must be at least two epochs.")
    window = list(records[-min(final_window, len(records)) :])
    epochs = [float(record["epoch"]) for record in window]
    validation = [100.0 * float(record["validation_accuracy"]) for record in window]
    training = [100.0 * float(record["train_accuracy"]) for record in window]
    validation_losses = [float(record["validation_loss"]) for record in window]
    training_losses = [float(record["train_loss"]) for record in window]

    final_record = records[-1]
    best_epoch = int(final_record["best_epoch"])
    final_epoch = int(final_record["epoch"])
    return {
        "epochs_observed": len(records),
        "first_epoch": int(records[0]["epoch"]),
        "final_epoch": final_epoch,
        "best_epoch_from_trace": best_epoch,
        "epochs_after_best": final_epoch - best_epoch,
        "best_in_final_three": final_epoch - best_epoch <= 2,
        "final_is_best": final_epoch == best_epoch,
        "final_minus_best_accuracy_pp": 100.0
        * (
            float(final_record["validation_accuracy"])
            - float(final_record["best_validation_accuracy"])
        ),
        "late_window_epochs": len(window),
        "late_validation_accuracy_slope_pp_per_epoch": _ols_slope(
            epochs, validation
        ),
        "late_train_accuracy_slope_pp_per_epoch": _ols_slope(epochs, training),
        "late_validation_loss_slope_per_epoch": _ols_slope(
            epochs, validation_losses
        ),
        "late_train_loss_slope_per_epoch": _ols_slope(epochs, training_losses),
    }


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=np.float64)
    start = 0
    while start < len(values_array):
        end = start + 1
        while end < len(values_array) and values_array[order[end]] == values_array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    x_centered = x_values - float(np.mean(x_values))
    y_centered = y_values - float(np.mean(y_values))
    denominator = math.sqrt(
        float(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    if denominator == 0.0:
        return None
    return float(np.dot(x_centered, y_centered) / denominator)


def _cosine_origin(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Return the exact pooled correlation after centering each two-run pair."""
    if len(x) != len(y) or not x:
        return None
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    denominator = math.sqrt(float(np.dot(x_values, x_values) * np.dot(y_values, y_values)))
    if denominator == 0.0:
        return None
    return float(np.dot(x_values, y_values) / denominator)


def _correlation_rows(
    run_rows: Sequence[Mapping[str, Any]], pair_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for accuracy_field in ACCURACY_FIELDS:
        for clipping_field in CLIPPING_FIELDS:
            raw_x = [float(row[clipping_field]) for row in run_rows]
            raw_y = [float(row[accuracy_field]) for row in run_rows]
            output.extend(
                [
                    {
                        "scope": "naive_selected_runs",
                        "accuracy_field": accuracy_field,
                        "clipping_field": clipping_field,
                        "correlation": "pearson",
                        "n": len(raw_x),
                        "value": _pearson(raw_x, raw_y),
                    },
                    {
                        "scope": "naive_selected_runs",
                        "accuracy_field": accuracy_field,
                        "clipping_field": clipping_field,
                        "correlation": "spearman",
                        "n": len(raw_x),
                        "value": _pearson(_rankdata(raw_x), _rankdata(raw_y)),
                    },
                ]
            )

            delta_x = [float(row[f"delta_{clipping_field}"]) for row in pair_rows]
            delta_y = [float(row[f"delta_{accuracy_field}"]) for row in pair_rows]
            # Centering each two-point surface yields +/- half its paired delta.
            # Pooling those centered observations gives the origin cosine of the
            # delta vectors (not ordinary Pearson after centering the deltas again).
            output.extend(
                [
                    {
                        "scope": "within_surface_centered",
                        "accuracy_field": accuracy_field,
                        "clipping_field": clipping_field,
                        "correlation": "pearson",
                        "n": 2 * len(delta_x),
                        "value": _cosine_origin(delta_x, delta_y),
                    },
                    {
                        "scope": "paired_deltas",
                        "accuracy_field": accuracy_field,
                        "clipping_field": clipping_field,
                        "correlation": "pearson",
                        "n": len(delta_x),
                        "value": _pearson(delta_x, delta_y),
                    },
                    {
                        "scope": "paired_deltas",
                        "accuracy_field": accuracy_field,
                        "clipping_field": clipping_field,
                        "correlation": "spearman",
                        "n": len(delta_x),
                        "value": _pearson(_rankdata(delta_x), _rankdata(delta_y)),
                    },
                ]
            )
    for accuracy_field in ACCURACY_FIELDS:
        accuracy_deltas = [
            float(row[f"delta_{accuracy_field}"]) for row in pair_rows
        ]
        for endpoint_field in (
            "exact_lower_endpoint_percent",
            "exact_upper_endpoint_percent",
            "exact_either_endpoint_percent",
        ):
            predictor = [
                float(row[f"low_{endpoint_field}"]) for row in pair_rows
            ]
            for correlation, value in (
                ("pearson", _pearson(predictor, accuracy_deltas)),
                (
                    "spearman",
                    _pearson(_rankdata(predictor), _rankdata(accuracy_deltas)),
                ),
            ):
                output.append(
                    {
                        "scope": "paired_low_wmax_endpoint_vs_accuracy_delta",
                        "accuracy_field": f"delta_{accuracy_field}",
                        "clipping_field": f"low_{endpoint_field}",
                        "correlation": correlation,
                        "n": len(pair_rows),
                        "value": value,
                    }
                )
        for motion_field in (
            "delta_final_motion_mean_abs_percent_span",
            "delta_final_motion_rms_percent_span",
        ):
            if not pair_rows or motion_field not in pair_rows[0]:
                continue
            predictor = [float(row[motion_field]) for row in pair_rows]
            for correlation, value in (
                ("pearson", _pearson(predictor, accuracy_deltas)),
                (
                    "spearman",
                    _pearson(_rankdata(predictor), _rankdata(accuracy_deltas)),
                ),
            ):
                output.append(
                    {
                        "scope": "paired_delta_normalized_motion_vs_accuracy_delta",
                        "accuracy_field": f"delta_{accuracy_field}",
                        "clipping_field": motion_field,
                        "correlation": correlation,
                        "n": len(pair_rows),
                        "value": value,
                    }
                )
    return output


def _initial_state(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Replay bounded initialization using the production construction path."""
    repo_root = Path(__file__).resolve().parents[1]
    labs_root = repo_root / "labs"
    if str(labs_root) not in sys.path:
        sys.path.insert(0, str(labs_root))
    import torch

    from custom_classes import FlexibleDeepResistiveEnergy
    from labs.mnist_train import _reset_name_counters, _set_seed

    _reset_name_counters()
    _set_seed(int(config["seed"]))
    model_key = str(config["lab"]["model_key"])
    model = {**config["model_base"], **config["model_overrides"][model_key]}
    energy = FlexibleDeepResistiveEnergy(
        layer_shapes=[tuple(shape) for shape in model["layer_shapes"]],
        conv_pipeline=model.get("conv_pipeline") or [],
        pooling_mode=model.get("pooling_mode"),
        weight_gains=model["weight_gains"],
        input_gain=model["input_gain"],
        non_linearity=model["non_linearity"],
        exponential_diode_param=model["exponential_diode_param"],
        quadratic_diode_param=model["quadratic_diode_param"],
        hard_sigmoid_param=model["hard_sigmoid_param"],
        voltage_amp=model["voltage_amp"],
        current_amp=model["current_amp"],
        weight_min=model["weight_min"],
        weight_max=model["weight_max"],
        weight_init_mode=model["weight_init_mode"],
        input_mode=config["input_mode"],
        trainable_amplification=model["trainable_amplification"],
        amplification_min=model["amplification_min"],
        amplification_max=model["amplification_max"],
    )
    energy.set_device("cpu")
    output: dict[str, np.ndarray] = {}
    for parameter in energy.params():
        name = str(parameter.name).strip()
        if name.startswith(WEIGHT_PREFIXES):
            output[name] = parameter.state.detach().cpu().numpy().copy()
    del energy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def _replayed_motion(row: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(row["run_dir"]))
    manifest = _load_json(run_dir / "manifest.json")
    config = manifest.get("configuration", {}).get("resolved", {})
    if not isinstance(config, Mapping):
        raise ValueError(f"Missing resolved config in {run_dir / 'manifest.json'}.")
    initial = _initial_state(config)
    checkpoint_path = Path(str(row.get("checkpoint_path", "")))
    if not checkpoint_path.is_file():
        result = _load_json(run_dir / "result.json")
        matches = [
            item
            for item in result.get("artifacts", [])
            if isinstance(item, Mapping)
            and Path(str(item.get("path", ""))).name == "weights_final.npz"
        ]
        if len(matches) != 1:
            raise ValueError(f"Cannot locate weights_final.npz in {run_dir}.")
        checkpoint_path = run_dir / str(matches[0]["path"])

    differences: list[np.ndarray] = []
    with np.load(checkpoint_path, allow_pickle=False) as payload:
        for name, initial_values in initial.items():
            if name not in payload.files:
                raise ValueError(f"Final checkpoint lacks replayed parameter {name!r}.")
            final_values = np.asarray(payload[name])
            if final_values.shape != initial_values.shape:
                raise ValueError(f"Shape mismatch for replayed parameter {name!r}.")
            differences.append(
                np.abs(
                    final_values.astype(np.float64)
                    - initial_values.astype(np.float64)
                ).reshape(-1)
            )
    if not differences:
        raise ValueError(f"No bounded weights replayed for {run_dir}.")
    absolute = np.concatenate(differences)
    span = float(row["weight_max"]) - float(row["weight_min"])
    normalized = absolute / span
    return {
        "motion_num_weights": int(absolute.size),
        "final_motion_mean_abs": float(np.mean(absolute)),
        "final_motion_rms": float(np.sqrt(np.mean(np.square(absolute)))),
        "final_motion_p50_abs": float(np.quantile(absolute, 0.50)),
        "final_motion_p90_abs": float(np.quantile(absolute, 0.90)),
        "final_motion_p99_abs": float(np.quantile(absolute, 0.99)),
        "final_motion_mean_abs_percent_span": 100.0 * float(np.mean(normalized)),
        "final_motion_rms_percent_span": 100.0
        * float(np.sqrt(np.mean(np.square(normalized)))),
        "final_motion_p50_percent_span": 100.0 * float(np.quantile(normalized, 0.50)),
        "final_motion_p90_percent_span": 100.0 * float(np.quantile(normalized, 0.90)),
        "final_motion_p99_percent_span": 100.0 * float(np.quantile(normalized, 0.99)),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _jsonable_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key not in {"learning_rates"}
        }
        for row in rows
    ]


def _save_plots(
    output_dir: Path,
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    include_motion: bool,
) -> list[str]:
    if not pair_rows:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: list[str] = []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for architecture in sorted({str(row["architecture"]) for row in pair_rows}):
        points = [row for row in pair_rows if row["architecture"] == architecture]
        color = ARCHITECTURE_COLORS.get(architecture, "#777777")
        axes[0].scatter(
            [float(row["delta_exact_either_endpoint_percent"]) for row in points],
            [float(row["delta_best_validation_accuracy_percent"]) for row in points],
            color=color,
            label=architecture,
            alpha=0.85,
        )
        axes[1].scatter(
            [
                float(row["delta_late_validation_accuracy_slope_pp_per_epoch"])
                for row in points
            ],
            [float(row["delta_best_validation_accuracy_percent"]) for row in points],
            color=color,
            label=architecture,
            alpha=0.85,
        )
    axes[0].set_xlabel("Change in final bound occupancy (pp)")
    axes[0].set_ylabel("Change in best validation accuracy (pp)")
    axes[1].set_xlabel("Change in final-5 validation slope (pp/epoch)")
    axes[1].set_ylabel("Change in best validation accuracy (pp)")
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Paired change: wmax 1e-3 minus 3e-4")
    fig.tight_layout()
    name = "paired_accuracy_clipping_and_late_slope.png"
    fig.savefig(output_dir / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs.append(name)

    if include_motion:
        fig, axis = plt.subplots(figsize=(6.5, 4.4))
        for architecture in sorted({str(row["architecture"]) for row in pair_rows}):
            points = [row for row in pair_rows if row["architecture"] == architecture]
            axis.scatter(
                [
                    float(row["delta_final_motion_mean_abs_percent_span"])
                    for row in points
                ],
                [float(row["delta_best_validation_accuracy_percent"]) for row in points],
                color=ARCHITECTURE_COLORS.get(architecture, "#777777"),
                label=architecture,
                alpha=0.85,
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_xlabel("Change in mean final motion (% of span)")
        axis.set_ylabel("Change in best validation accuracy (pp)")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
        axis.set_title("Paired change: wmax 1e-3 minus 3e-4")
        fig.tight_layout()
        name = "paired_accuracy_and_normalized_final_motion.png"
        fig.savefig(output_dir / name, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(name)
    return outputs


def analyze(
    *,
    pooled_csv: Path,
    output_dir: Path,
    low_wmax: float = DEFAULT_LOW_WMAX,
    high_wmax: float = DEFAULT_HIGH_WMAX,
    final_window: int = 5,
    replay_initialization: bool = False,
) -> dict[str, Any]:
    pooled_csv = Path(pooled_csv).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not (0.0 < low_wmax < high_wmax):
        raise ValueError("Expected 0 < low_wmax < high_wmax.")
    rows = _load_pooled_rows(pooled_csv)
    selected = [
        row
        for row in rows
        if math.isclose(float(row["weight_max"]), low_wmax, rel_tol=0.0, abs_tol=1e-15)
        or math.isclose(
            float(row["weight_max"]), high_wmax, rel_tol=0.0, abs_tol=1e-15
        )
    ]
    if not selected:
        raise ValueError("No rows match the requested wmax pair.")

    run_rows: list[dict[str, Any]] = []
    for row in selected:
        run_dir = Path(str(row["run_dir"]))
        weight_learning_rates = _weight_learning_rates(
            run_dir, expected_arm_id=str(row["arm_id"])
        )
        diagnostics = _trajectory_diagnostics(
            _metric_records(run_dir, expected_arm_id=str(row["arm_id"])),
            final_window=final_window,
        )
        span = float(row["weight_max"]) - float(row["weight_min"])
        enriched = {
            **row,
            "weight_span": span,
            "weight_learning_rates_json": json.dumps(
                weight_learning_rates, sort_keys=True, separators=(",", ":")
            ),
            "weight_learning_rate_min_percent_span": 100.0
            * min(weight_learning_rates.values())
            / span,
            "weight_learning_rate_max_percent_span": 100.0
            * max(weight_learning_rates.values())
            / span,
            **diagnostics,
        }
        if replay_initialization:
            enriched.update(_replayed_motion(enriched))
        run_rows.append(enriched)

    by_surface: dict[tuple[str, str, str], dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in run_rows:
        key = (str(row["architecture"]), str(row["scheme"]), str(row["optimizer"]))
        by_surface[key][float(row["weight_max"])] = row

    pair_rows: list[dict[str, Any]] = []
    missing_pairs: list[dict[str, Any]] = []
    delta_fields = [
        *ACCURACY_FIELDS,
        *CLIPPING_FIELDS,
        "weight_learning_rate_min_percent_span",
        "weight_learning_rate_max_percent_span",
        "best_epoch_from_trace",
        "epochs_after_best",
        "final_minus_best_accuracy_pp",
        "late_validation_accuracy_slope_pp_per_epoch",
        "late_train_accuracy_slope_pp_per_epoch",
        "late_validation_loss_slope_per_epoch",
    ]
    if replay_initialization:
        delta_fields.extend(
            [
                "final_motion_mean_abs_percent_span",
                "final_motion_rms_percent_span",
                "final_motion_p90_percent_span",
                "final_motion_p99_percent_span",
            ]
        )
    for key, candidates in sorted(by_surface.items()):
        low = next(
            (
                row
                for weight_max, row in candidates.items()
                if math.isclose(weight_max, low_wmax, rel_tol=0.0, abs_tol=1e-15)
            ),
            None,
        )
        high = next(
            (
                row
                for weight_max, row in candidates.items()
                if math.isclose(weight_max, high_wmax, rel_tol=0.0, abs_tol=1e-15)
            ),
            None,
        )
        if low is None or high is None:
            missing_pairs.append(
                {
                    "architecture": key[0],
                    "scheme": key[1],
                    "optimizer": key[2],
                    "has_low_wmax": low is not None,
                    "has_high_wmax": high is not None,
                }
            )
            continue
        if low["learning_rates"] != high["learning_rates"]:
            raise ValueError(f"Learning-rate vector changes within surface {key!r}.")
        if low["weight_learning_rates_json"] != high["weight_learning_rates_json"]:
            raise ValueError(
                f"Bounded-weight learning-rate mapping changes within surface {key!r}."
            )
        pair: dict[str, Any] = {
            "architecture": key[0],
            "scheme": key[1],
            "optimizer": key[2],
            "low_wmax": low_wmax,
            "high_wmax": high_wmax,
            "low_arm_id": low["arm_id"],
            "high_arm_id": high["arm_id"],
            "learning_rate_vector_json": low["learning_rate_vector_json"],
            "weight_learning_rates_json": low["weight_learning_rates_json"],
            "span_ratio_high_over_low": float(high["weight_span"])
            / float(low["weight_span"]),
            "lr_multiplier_to_match_low_normalized_step_at_high": float(
                high["weight_span"]
            )
            / float(low["weight_span"]),
            "high_best_in_final_three": high["best_in_final_three"],
            "low_best_in_final_three": low["best_in_final_three"],
            "high_final_is_best": high["final_is_best"],
            "low_final_is_best": low["final_is_best"],
        }
        for field in delta_fields:
            pair[f"low_{field}"] = low[field]
            pair[f"high_{field}"] = high[field]
            pair[f"delta_{field}"] = float(high[field]) - float(low[field])
        pair_rows.append(pair)
    if not pair_rows:
        raise ValueError("No complete matched surfaces for the requested wmax pair.")

    correlation_rows = _correlation_rows(
        [
            row
            for row in run_rows
            if any(
                row["arm_id"] in {pair["low_arm_id"], pair["high_arm_id"]}
                for pair in pair_rows
            )
        ],
        pair_rows,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_csv = output_dir / "matched_run_diagnostics.csv"
    pair_csv = output_dir / "paired_wmax_3em4_vs_1em3.csv"
    correlation_csv = output_dir / "clipping_accuracy_correlations.csv"
    serializable_runs = _jsonable_rows(run_rows)
    _write_csv(run_csv, serializable_runs)
    _write_csv(pair_csv, pair_rows)
    _write_csv(correlation_csv, correlation_rows)
    plots = _save_plots(
        output_dir, pair_rows, include_motion=replay_initialization
    )

    accuracy_wins = sum(
        float(pair["delta_best_validation_accuracy_percent"]) > 0.0
        for pair in pair_rows
    )
    high_later_best = sum(
        float(pair["delta_best_epoch_from_trace"]) > 0.0 for pair in pair_rows
    )
    high_steeper = sum(
        float(pair["delta_late_validation_accuracy_slope_pp_per_epoch"]) > 0.0
        for pair in pair_rows
    )
    high_final_three = sum(bool(pair["high_best_in_final_three"]) for pair in pair_rows)
    low_final_three = sum(bool(pair["low_best_in_final_three"]) for pair in pair_rows)
    report: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA,
        "input": {
            "pooled_csv": str(pooled_csv),
            "pooled_csv_sha256": _sha256(pooled_csv),
            "upstream_assumption": (
                "The pooled CSV was produced by the canonical-bundle-validating "
                "final-clipping analyzer; metrics digests were rechecked here."
            ),
        },
        "comparison": {
            "low_wmax": low_wmax,
            "high_wmax": high_wmax,
            "final_window": final_window,
            "matched_surface_count": len(pair_rows),
            "missing_surface_count": len(missing_pairs),
            "missing_surfaces": missing_pairs,
            "initialization_replay": replay_initialization,
            "high_wmax_best_accuracy_win_count": accuracy_wins,
            "high_wmax_best_epoch_later_count": high_later_best,
            "high_wmax_steeper_late_validation_count": high_steeper,
            "high_wmax_best_in_final_three_count": high_final_three,
            "low_wmax_best_in_final_three_count": low_final_three,
            "median_lr_multiplier_to_preserve_span_normalized_step": float(
                np.median(
                    [
                        float(pair["lr_multiplier_to_match_low_normalized_step_at_high"])
                        for pair in pair_rows
                    ]
                )
            ),
        },
        "interpretation_limits": [
            "A final endpoint count is not a cumulative projection count.",
            "A low endpoint count does not establish convergence or LR adequacy.",
            "Naive correlations mix architecture, optimizer, scheme, and range.",
            "Within-surface correlations use one seed and only two ranges per surface.",
            "Initialization changes with wmax in this study unless replayed motion is used only as a diagnostic.",
            "Ordinary-MNIST selection evidence is diagnostic, not paper-facing accuracy.",
        ],
        "correlations": correlation_rows,
        "paired_rows": pair_rows,
        "outputs": [
            run_csv.name,
            pair_csv.name,
            correlation_csv.name,
            *plots,
            "lr_adequacy_summary.json",
        ],
    }
    summary_path = output_dir / "lr_adequacy_summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-wmax", type=float, default=DEFAULT_LOW_WMAX)
    parser.add_argument("--high-wmax", type=float, default=DEFAULT_HIGH_WMAX)
    parser.add_argument("--final-window", type=int, default=5)
    parser.add_argument(
        "--replay-initialization",
        action="store_true",
        help=(
            "Deterministically reconstruct each model's initial bounded weights "
            "and measure span-normalized motion to the final checkpoint."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze(
        pooled_csv=args.pooled_csv,
        output_dir=args.output_dir,
        low_wmax=args.low_wmax,
        high_wmax=args.high_wmax,
        final_window=args.final_window,
        replay_initialization=args.replay_initialization,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                "matched_surface_count": report["comparison"][
                    "matched_surface_count"
                ],
                "missing_surface_count": report["comparison"][
                    "missing_surface_count"
                ],
                "initialization_replay": report["comparison"][
                    "initialization_replay"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
