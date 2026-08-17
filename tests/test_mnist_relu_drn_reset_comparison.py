from __future__ import annotations

import json
from pathlib import Path

from labs.tools.plot_mnist_relu_drn_reset_comparison import compare


def _validation(accuracy: float, kl: float) -> dict:
    return {
        "objective_loss": kl,
        "kl_teacher_student": kl,
        "student_accuracy": accuracy,
        "teacher_agreement": accuracy,
    }


def _write_run(path: Path, *, objective: str, accuracy: float) -> None:
    path.mkdir()
    initial = _validation(0.10, 2.2)
    selected = {"epoch": 0, **_validation(accuracy, 0.1)}
    rows = [
        {"mode": "initialization", "validation": initial},
        {"mode": "train", "completed_epochs": 1, "validation": selected},
    ]
    (path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    parameters = {
        "base.dense_weight.0": {"pulse_changed_fraction": 0.1},
        "base.dense_weight.1": {"pulse_changed_fraction": 0.2},
    }
    result = {
        "status": "complete",
        "error": None,
        "metrics": {
            "objective": objective,
            "initial_validation": initial,
            "selected": selected,
            "last_validation": selected,
            "selected_learning_rates": [1e-7, 1e-8],
            "measured_programming": {
                "parameters": parameters,
                "assignment_sha256_by_parameter": {
                    "base.dense_weight.0": "first",
                    "base.dense_weight.1": "second",
                },
            },
            "teacher": {"sha256": "teacher"},
            "posthoc_validation_calibration": {"gain": 4.0},
        },
    }
    (path / "result.json").write_text(json.dumps(result), encoding="utf-8")


def _write_test(path: Path, *, objective: str, accuracy: float) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "error": None,
                "metrics": {
                    "objective": objective,
                    "student_accuracy": accuracy,
                    "kl_teacher_student": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )


def test_reset_comparison_applies_accuracy_and_matching_gates(
    tmp_path: Path,
) -> None:
    kl_run = tmp_path / "kl"
    ce_run = tmp_path / "ce"
    _write_run(kl_run, objective="teacher_kl", accuracy=0.95)
    _write_run(ce_run, objective="cross_entropy", accuracy=0.955)
    kl_test = tmp_path / "kl_test.json"
    ce_test = tmp_path / "ce_test.json"
    _write_test(kl_test, objective="teacher_kl", accuracy=0.95)
    _write_test(ce_test, objective="cross_entropy", accuracy=0.955)
    output = tmp_path / "analysis"

    summary = compare(
        kl_run=kl_run,
        cross_entropy_run=ce_run,
        kl_test_result=kl_test,
        cross_entropy_test_result=ce_test,
        output_dir=output,
    )

    assert summary["comparison"]["hypothesis_supported"] is True
    assert summary["comparison"]["criteria"]["same_device_assignments"] is True
    assert (output / "summary.json").is_file()
    assert (output / "validation_comparison.png").stat().st_size > 0
