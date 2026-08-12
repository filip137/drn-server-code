from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.analyze_bounded_wmax_lr_adequacy import (
    _correlation_rows,
    analyze,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _metric_lines(values: list[float]) -> str:
    best = -1.0
    best_epoch = 0
    records = []
    for epoch, value in enumerate(values, start=1):
        if value > best:
            best = value
            best_epoch = epoch
        records.append(
            {
                "schema_version": "experiment-run-metric/v1",
                "kind": "epoch",
                "epoch": epoch,
                "metrics": {
                    "best_epoch": best_epoch,
                    "best_validation_accuracy": best,
                    "validation_accuracy": value,
                    "train_accuracy": value + 0.01,
                    "validation_loss": 1.0 - value,
                    "train_loss": 0.9 - value,
                },
            }
        )
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


def _run(tmp_path: Path, arm_id: str, values: list[float]) -> Path:
    run_dir = tmp_path / arm_id
    run_dir.mkdir()
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text(_metric_lines(values), encoding="utf-8")
    digest = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    _write_json(
        run_dir / "manifest.json",
        {
            "arm_id": arm_id,
            "configuration": {
                "resolved": {
                    "learning_rates_by_parameter": {
                        "ConvWeight_0": 1.0e-6,
                        "DenseWeight_0": 2.0e-6,
                        "Bias_0": 9.0e-3,
                    }
                }
            },
        },
    )
    _write_json(
        run_dir / "result.json",
        {"arm_id": arm_id, "metrics_sha256": digest},
    )
    return run_dir


def _pooled_csv(tmp_path: Path) -> Path:
    rows = []
    cases = [
        (
            "conv1",
            "baseline",
            "sgd",
            3.0e-4,
            10.0,
            [0.50, 0.55, 0.60, 0.60, 0.60],
        ),
        (
            "conv1",
            "baseline",
            "sgd",
            1.0e-3,
            5.0,
            [0.40, 0.45, 0.50, 0.55, 0.60],
        ),
        (
            "conv2",
            "ours",
            "adam",
            3.0e-4,
            20.0,
            [0.70, 0.72, 0.74, 0.76, 0.80],
        ),
        (
            "conv2",
            "ours",
            "adam",
            1.0e-3,
            10.0,
            [0.70, 0.75, 0.80, 0.85, 0.90],
        ),
        # This surface deliberately has only the low range.
        (
            "conv3",
            "legacy",
            "sgd",
            3.0e-4,
            8.0,
            [0.70, 0.71, 0.72, 0.73, 0.74],
        ),
    ]
    for architecture, scheme, optimizer, wmax, clipped, values in cases:
        arm_id = f"{architecture}_{scheme}_{optimizer}_{wmax}"
        run_dir = _run(tmp_path, arm_id, values)
        rows.append(
            {
                "arm_id": arm_id,
                "architecture": architecture,
                "scheme": scheme,
                "optimizer": optimizer,
                "weight_min": 1.0e-5,
                "weight_max": wmax,
                "learning_rate_vector_json": "[1e-6,2e-6]",
                "run_dir": str(run_dir),
                "best_validation_accuracy_percent": 100.0 * max(values),
                "final_validation_accuracy_percent": 100.0 * values[-1],
                "exact_lower_endpoint_percent": 0.8 * clipped,
                "exact_upper_endpoint_percent": 0.2 * clipped,
                "exact_either_endpoint_percent": clipped,
                "row_level": "pooled",
            }
        )
    path = tmp_path / "pooled.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_matched_pair_correlations_and_late_diagnostics(tmp_path: Path) -> None:
    report = analyze(
        pooled_csv=_pooled_csv(tmp_path),
        output_dir=tmp_path / "analysis",
    )

    comparison = report["comparison"]
    assert comparison["matched_surface_count"] == 2
    assert comparison["missing_surface_count"] == 1
    assert comparison["median_lr_multiplier_to_preserve_span_normalized_step"] == pytest.approx(
        0.00099 / 0.00029
    )

    conv1 = next(row for row in report["paired_rows"] if row["architecture"] == "conv1")
    assert conv1["delta_best_validation_accuracy_percent"] == pytest.approx(0.0)
    assert conv1["delta_exact_either_endpoint_percent"] == pytest.approx(-5.0)
    assert conv1["low_best_epoch_from_trace"] == 3
    assert conv1["high_best_epoch_from_trace"] == 5
    assert conv1["delta_late_validation_accuracy_slope_pp_per_epoch"] == pytest.approx(
        2.5
    )

    centered = next(
        row
        for row in report["correlations"]
        if row["scope"] == "within_surface_centered"
        and row["accuracy_field"] == "best_validation_accuracy_percent"
        and row["clipping_field"] == "exact_either_endpoint_percent"
    )
    # Delta vectors are clipping [-5, -10] and accuracy [0, +10].
    assert centered["value"] == pytest.approx(-10.0 / (125.0**0.5))

    output_dir = tmp_path / "analysis"
    assert (output_dir / "matched_run_diagnostics.csv").is_file()
    assert (output_dir / "paired_wmax_3em4_vs_1em3.csv").is_file()
    assert (output_dir / "clipping_accuracy_correlations.csv").is_file()
    assert (output_dir / "lr_adequacy_summary.json").is_file()
    assert (output_dir / "paired_accuracy_clipping_and_late_slope.png").stat().st_size > 0


def test_metrics_digest_is_rechecked(tmp_path: Path) -> None:
    pooled = _pooled_csv(tmp_path)
    with pooled.open("r", encoding="utf-8", newline="") as handle:
        first = next(csv.DictReader(handle))
    metrics_path = Path(first["run_dir"]) / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="digest mismatch"):
        analyze(pooled_csv=pooled, output_dir=tmp_path / "analysis")


def test_motion_and_low_endpoint_pair_correlations_are_emitted() -> None:
    run_rows = [
        {
            "best_validation_accuracy_percent": value,
            "final_validation_accuracy_percent": value,
            "exact_lower_endpoint_percent": value,
            "exact_upper_endpoint_percent": value,
            "exact_either_endpoint_percent": value,
        }
        for value in (1.0, 2.0, 3.0, 4.0)
    ]
    pair_rows = [
        {
            "delta_best_validation_accuracy_percent": accuracy,
            "delta_final_validation_accuracy_percent": accuracy,
            "delta_exact_lower_endpoint_percent": clipping,
            "delta_exact_upper_endpoint_percent": clipping,
            "delta_exact_either_endpoint_percent": clipping,
            "low_exact_lower_endpoint_percent": low,
            "low_exact_upper_endpoint_percent": low,
            "low_exact_either_endpoint_percent": low,
            "delta_final_motion_mean_abs_percent_span": motion,
            "delta_final_motion_rms_percent_span": 2.0 * motion,
        }
        for accuracy, clipping, low, motion in (
            (1.0, -1.0, 4.0, 1.0),
            (2.0, -2.0, 3.0, 2.0),
            (3.0, -3.0, 2.0, 3.0),
        )
    ]

    rows = _correlation_rows(run_rows, pair_rows)
    motion_predictors = {
        row["clipping_field"]
        for row in rows
        if row["scope"]
        == "paired_delta_normalized_motion_vs_accuracy_delta"
    }
    assert motion_predictors == {
        "delta_final_motion_mean_abs_percent_span",
        "delta_final_motion_rms_percent_span",
    }
    upper = next(
        row
        for row in rows
        if row["scope"] == "paired_low_wmax_endpoint_vs_accuracy_delta"
        and row["accuracy_field"]
        == "delta_best_validation_accuracy_percent"
        and row["clipping_field"] == "low_exact_upper_endpoint_percent"
        and row["correlation"] == "pearson"
    )
    assert upper["value"] == pytest.approx(-1.0)
