from __future__ import annotations

import json
from pathlib import Path

import pytest

from labs.tools.plot_measured_cohort_b_threshold_sweep import (
    summarize_threshold_sweep,
)


def _write_run(
    path: Path,
    *,
    relative: float,
    validation_accuracy: float,
    event_fractions: tuple[float, float] | None,
    pulse_fractions: tuple[float, float],
) -> None:
    parameters = {
        "base.dense_weight.0": {
            "shape": [3, 2],
            "pulse_changed_fraction": pulse_fractions[0],
        },
        "base.dense_weight.1": {
            "shape": [2, 1],
            "pulse_changed_fraction": pulse_fractions[1],
        },
    }
    if event_fractions is not None:
        for report, fraction in zip(
            parameters.values(),
            event_fractions,
            strict=True,
        ):
            report["programming_event_fraction"] = fraction
    programming = {
        "active_cohort": "B",
        "parameters": parameters,
    }
    if relative > 0.0:
        programming.update(
            {
                "programming_deadband_mode": (
                    "accumulated_shadow_relative_rms"
                ),
                "programming_deadband_relative": relative,
            }
        )
    path.mkdir()
    (path / "result.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "error": None,
                "metrics": {
                    "selected": {
                        "epoch": 3,
                        "accuracy": validation_accuracy,
                        "value": 1.0 - validation_accuracy,
                    },
                    "last_validation": {
                        "accuracy": validation_accuracy - 0.001,
                    },
                    "update_programming": programming,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_test(path: Path, accuracy: float) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "metrics": {
                    "accuracy": accuracy,
                    "mean_cost": 1.0 - accuracy,
                    "examples": 10000,
                    "split_protocol": {"effective": "held_out_test"},
                },
            }
        ),
        encoding="utf-8",
    )


def test_threshold_summary_compares_accuracy_and_weighted_programming(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    threshold_low = tmp_path / "threshold_low"
    threshold_high = tmp_path / "threshold_high"
    _write_run(
        baseline,
        relative=0.0,
        validation_accuracy=0.950,
        event_fractions=None,
        pulse_fractions=(0.10, 0.20),
    )
    _write_run(
        threshold_low,
        relative=0.001,
        validation_accuracy=0.952,
        event_fractions=(0.50, 0.25),
        pulse_fractions=(0.05, 0.10),
    )
    _write_run(
        threshold_high,
        relative=0.01,
        validation_accuracy=0.948,
        event_fractions=(0.10, 0.05),
        pulse_fractions=(0.02, 0.04),
    )
    baseline_test = tmp_path / "baseline_test.json"
    low_test = tmp_path / "low_test.json"
    high_test = tmp_path / "high_test.json"
    _write_test(baseline_test, 0.951)
    _write_test(low_test, 0.953)
    _write_test(high_test, 0.949)

    output = tmp_path / "output"
    summary = summarize_threshold_sweep(
        baseline,
        [threshold_high, threshold_low],
        baseline_test_result=baseline_test,
        threshold_test_results=[high_test, low_test],
        output_dir=output,
    )

    low = summary["threshold_runs"][0]
    assert low["programming_deadband_relative"] == pytest.approx(0.001)
    assert low["programming_event_fraction"] == pytest.approx(0.4375)
    assert low["pulse_changed_fraction"] == pytest.approx(0.0625)
    assert low["official_test_delta_vs_baseline"] == pytest.approx(0.002)
    assert summary["conclusion"]["any_official_test_improvement"] is True
    assert summary["conclusion"][
        "most_efficient_accuracy_retaining_deadband_relative"
    ] == pytest.approx(0.01)
    assert (output / "summary.json").is_file()
    assert (output / "cohort_b_threshold_sweep.png").stat().st_size > 0
