from __future__ import annotations

import json
from pathlib import Path

from labs.tools.plot_mnist_relu_drn_reset_bias_comparison import compare


OBJECTIVES = ("paired_squared_error", "cross_entropy", "teacher_kl")
TRAIN_INDICES = (
    {
        "resolved_pre_index": 3,
        "resolved_post_index": 4,
    },
    {
        "resolved_pre_index": 4,
        "resolved_post_index": 5,
    },
)
TEST_INDICES = (
    {
        "resolved_pre_index": 0,
        "resolved_post_index": 1,
    },
    {
        "resolved_pre_index": 1,
        "resolved_post_index": 2,
    },
)


def _validation(accuracy: float, offset: float = 0.0) -> dict:
    return {
        "objective_loss": 0.1 + offset,
        "kl_teacher_student": 0.2 + offset,
        "cross_entropy": 0.3 + offset,
        "paired_squared_error": 0.4 + offset,
        "student_accuracy": accuracy,
        "student_logit_rms": 0.5 + offset,
        "teacher_agreement": accuracy,
    }


def _write_run(path: Path, *, objective: str, accuracy: float) -> None:
    (path / "artifacts").mkdir(parents=True)
    initial = _validation(0.10, 2.0)
    selected = {"epoch": 0, **_validation(accuracy)}
    rows = [{"mode": "initialization", "validation": initial}]
    rows.extend(
        {
            "mode": "train",
            "completed_epochs": epoch,
            "validation": selected,
        }
        for epoch in range(1, 21)
    )
    (path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = {
        "status": "complete",
        "error": None,
        "metrics": {
            "objective": objective,
            "include_biases": True,
            "amplification_indexing": "legacy_process_global",
            "amplification_indices": TRAIN_INDICES,
            "initial_validation": initial,
            "selected": selected,
            "last_validation": selected,
            "selected_learning_rates": [1e-7, 1e-8, 1e-7],
            "measured_programming": {
                "parameters": {
                    "base.dense_weight.0": {"pulse_changed_fraction": 0.1},
                    "base.dense_weight.1": {"pulse_changed_fraction": 0.2},
                },
                "assignment_sha256_by_parameter": {
                    "base.dense_weight.0": "first",
                    "base.dense_weight.1": "second",
                },
            },
            "teacher": {"sha256": "teacher"},
        },
    }
    (path / "result.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )
    (path / "artifacts" / "learning_rate_selection.json").write_text(
        json.dumps({"reset_state_fingerprint": "same-reset"}),
        encoding="utf-8",
    )


def _write_test(path: Path, *, objective: str, accuracy: float) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "error": None,
                "metrics": {
                    "objective": objective,
                    "include_biases": True,
                    "amplification_indexing": "legacy_process_global",
                    "amplification_indices": TEST_INDICES,
                    "split": "test",
                    "student_accuracy": accuracy,
                },
            }
        ),
        encoding="utf-8",
    )


def test_legacy_bias_loss_comparison_checks_provenance_and_objective_gap(
    tmp_path: Path,
) -> None:
    accuracies = {
        "paired_squared_error": 0.9597,
        "cross_entropy": 0.9237,
        "teacher_kl": 0.8372,
    }
    runs = {}
    tests = {}
    for objective in OBJECTIVES:
        run = tmp_path / f"{objective}_run"
        test = tmp_path / f"{objective}_test.json"
        _write_run(run, objective=objective, accuracy=accuracies[objective])
        _write_test(test, objective=objective, accuracy=accuracies[objective])
        runs[objective] = run
        tests[objective] = test
    historical = tmp_path / "historical_test.json"
    historical.write_text(
        json.dumps(
            {
                "status": "complete",
                "error": None,
                "metrics": {"accuracy": 0.9597},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "analysis"

    summary = compare(
        training_runs=runs,
        test_results=tests,
        historical_test_result=historical,
        output_dir=output,
    )

    comparison = summary["comparison"]
    assert comparison["legacy_provenance_control_valid"] is True
    assert comparison[
        "supervision_objective_is_material_under_legacy_mismatch"
    ] is True
    assert comparison[
        "cross_entropy_recovers_at_least_94pct_and_within_1pp_of_paired"
    ] is False
    assert comparison[
        "teacher_kl_recovers_at_least_94pct_and_within_1pp_of_paired"
    ] is False
    assert comparison["invariants"][
        "all_training_indices_are_legacy_3_4_5"
    ] is True
    assert comparison["invariants"][
        "all_test_indices_are_fresh_0_1_2"
    ] is True
    assert (output / "summary.json").is_file()
    assert (output / "validation_and_test_comparison.png").stat().st_size > 0
