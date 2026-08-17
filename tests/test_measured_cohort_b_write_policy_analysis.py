from __future__ import annotations

import json
from pathlib import Path

import pytest

from labs.tools.plot_measured_cohort_b_write_policies import (
    summarize_write_policies,
)


def _policy_parameters(
    mode: str,
    configured_value: float,
) -> dict[str, object]:
    parameters: dict[str, object] = {
        "curve_preprocessing": "raw",
        "programming_deadband_mode": "none",
        "programming_deadband_relative": 0.0,
        "probabilistic_write_mode": "none",
        "probabilistic_write_probability": 1.0,
        "probabilistic_write_scale_relative": 0.0,
        "probabilistic_write_seed": 0,
    }
    if mode == "deadband":
        parameters["programming_deadband_mode"] = (
            "accumulated_shadow_relative_rms"
        )
        parameters["programming_deadband_relative"] = configured_value
    elif mode == "uniform":
        parameters["probabilistic_write_mode"] = "uniform_bernoulli"
        parameters["probabilistic_write_probability"] = configured_value
        parameters["probabilistic_write_seed"] = 17
    elif mode == "displacement":
        parameters["probabilistic_write_mode"] = (
            "displacement_proportional"
        )
        parameters["probabilistic_write_scale_relative"] = configured_value
        parameters["probabilistic_write_seed"] = 17
    elif mode != "baseline":
        raise AssertionError(mode)
    return parameters


def _write_run(
    path: Path,
    *,
    mode: str,
    configured_value: float,
    validation_accuracy: float,
    validation_cost: float,
    event_fractions: tuple[float, float],
    pulse_fractions: tuple[float, float],
    mean_probabilities: tuple[float | None, float | None],
    learning_rate: float = 1e-7,
) -> None:
    path.mkdir()
    backend_parameters = _policy_parameters(mode, configured_value)
    (path / "config.resolved.json").write_text(
        json.dumps(
            {
                "modes": {
                    "train": {
                        "learning_rates": [learning_rate],
                        "update_backend": {
                            "type": "measured_cohort_b",
                            "parameters": backend_parameters,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "inputs": [
                    {"role": "weights", "sha256": "weights-sha"},
                    {"role": "device_data", "sha256": "device-sha"},
                ],
                "source": {
                    "commit": "commit",
                    "dirty": True,
                    "dirty_hash": "dirty-sha",
                },
            }
        ),
        encoding="utf-8",
    )
    parameter_reports = {
        "base.dense_weight.0": {
            "shape": [3, 2],
            "programming_event_fraction": event_fractions[0],
            "pulse_changed_fraction": pulse_fractions[0],
            "mean_write_probability": mean_probabilities[0],
        },
        "base.dense_weight.1": {
            "shape": [2, 1],
            "programming_event_fraction": event_fractions[1],
            "pulse_changed_fraction": pulse_fractions[1],
            "mean_write_probability": mean_probabilities[1],
        },
    }
    programming = {
        "active_cohort": "B",
        "source_sha256": "device-sha",
        "assignment_sha256_by_parameter": {
            "base.dense_weight.0": "assignment-0",
            "base.dense_weight.1": "assignment-1",
        },
        "parameters": parameter_reports,
        **backend_parameters,
    }
    (path / "result.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "error": None,
                "metrics": {
                    "completed_epochs": 20,
                    "global_step": 100,
                    "selected": {
                        "epoch": 4,
                        "accuracy": validation_accuracy,
                        "value": validation_cost,
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


def _write_test(path: Path, accuracy: float, mean_cost: float) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "error": None,
                "metrics": {
                    "accuracy": accuracy,
                    "mean_cost": mean_cost,
                    "examples": 10000,
                    "split_protocol": {"effective": "held_out_test"},
                },
            }
        ),
        encoding="utf-8",
    )


def test_write_policy_summary_compares_matched_activity_and_accuracy(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    deadband = tmp_path / "deadband"
    uniform = tmp_path / "uniform"
    displacement = tmp_path / "displacement"
    _write_run(
        baseline,
        mode="baseline",
        configured_value=1.0,
        validation_accuracy=0.9400,
        validation_cost=0.0900,
        event_fractions=(1.0, 1.0),
        pulse_fractions=(0.20, 0.10),
        mean_probabilities=(None, None),
    )
    _write_run(
        deadband,
        mode="deadband",
        configured_value=0.0003,
        validation_accuracy=0.9390,
        validation_cost=0.0910,
        event_fractions=(0.80, 0.60),
        pulse_fractions=(0.15, 0.05),
        mean_probabilities=(None, None),
    )
    _write_run(
        uniform,
        mode="uniform",
        configured_value=0.85,
        validation_accuracy=0.9400,
        validation_cost=0.0890,
        event_fractions=(0.85, 0.85),
        pulse_fractions=(0.16, 0.06),
        mean_probabilities=(0.85, 0.85),
    )
    _write_run(
        displacement,
        mode="displacement",
        configured_value=0.0005,
        validation_accuracy=0.9395,
        validation_cost=0.0895,
        event_fractions=(0.88, 0.84),
        pulse_fractions=(0.17, 0.07),
        mean_probabilities=(0.88, 0.84),
    )
    baseline_test = tmp_path / "baseline_test.json"
    deadband_test = tmp_path / "deadband_test.json"
    uniform_test = tmp_path / "uniform_test.json"
    displacement_test = tmp_path / "displacement_test.json"
    _write_test(baseline_test, 0.9450, 0.0860)
    _write_test(deadband_test, 0.9451, 0.0861)
    _write_test(uniform_test, 0.9449, 0.0850)
    _write_test(displacement_test, 0.9448, 0.0855)

    output = tmp_path / "output"
    summary = summarize_write_policies(
        baseline,
        [
            ("Deadband", deadband, deadband_test),
            ("Uniform", uniform, uniform_test),
            ("Displacement", displacement, displacement_test),
        ],
        baseline_test_result=baseline_test,
        output_dir=output,
    )

    rows = {item["label"]: item for item in summary["policies"]}
    assert rows["Deadband"]["programming_event_fraction"] == pytest.approx(
        0.75
    )
    assert rows["Uniform"]["programming_event_fraction"] == pytest.approx(
        0.85
    )
    assert rows["Uniform"]["official_test_accuracy_delta"] == pytest.approx(
        -0.0001
    )
    assert rows["Uniform"]["official_test_mean_cost_delta"] == pytest.approx(
        -0.001
    )
    assert summary["conclusion"]["recommended_probabilistic_policy"] == (
        "Uniform"
    )
    assert summary["conclusion"][
        "any_probabilistic_official_test_accuracy_improvement"
    ] is False
    assert summary["conclusion"][
        "any_probabilistic_official_test_cost_improvement"
    ] is True
    assert (output / "summary.json").is_file()
    assert (
        output / "cohort_b_write_policy_comparison.png"
    ).stat().st_size > 0


def test_write_policy_summary_rejects_non_policy_config_differences(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    uniform = tmp_path / "uniform"
    _write_run(
        baseline,
        mode="baseline",
        configured_value=1.0,
        validation_accuracy=0.94,
        validation_cost=0.09,
        event_fractions=(1.0, 1.0),
        pulse_fractions=(0.2, 0.1),
        mean_probabilities=(None, None),
    )
    _write_run(
        uniform,
        mode="uniform",
        configured_value=0.85,
        validation_accuracy=0.94,
        validation_cost=0.09,
        event_fractions=(0.85, 0.85),
        pulse_fractions=(0.16, 0.06),
        mean_probabilities=(0.85, 0.85),
        learning_rate=2e-7,
    )
    baseline_test = tmp_path / "baseline_test.json"
    uniform_test = tmp_path / "uniform_test.json"
    _write_test(baseline_test, 0.945, 0.086)
    _write_test(uniform_test, 0.945, 0.085)

    with pytest.raises(ValueError, match="differ only in write-policy"):
        summarize_write_policies(
            baseline,
            [("Uniform", uniform, uniform_test)],
            baseline_test_result=baseline_test,
            output_dir=tmp_path / "output",
        )
