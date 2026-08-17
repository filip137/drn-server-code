from __future__ import annotations

import json
from pathlib import Path

import pytest

from labs.tools.plot_measured_cohort_b_transfer import summarize_transfer


def _write_test(path: Path, accuracy: float) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "metrics": {
                    "accuracy": accuracy,
                    "examples": 10000,
                    "split_protocol": {"effective": "held_out_test"},
                },
            }
        ),
        encoding="utf-8",
    )


def test_transfer_summary_writes_plot_and_passes_feasibility_gates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    cohort_b = tmp_path / "cohort_b"
    source.mkdir()
    cohort_b.mkdir()
    (source / "result.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "error": None,
                "metrics": {"selected": {"accuracy": 0.95}},
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "mode": "train",
            "completed_epochs": index + 1,
            "validation": {
                "accuracy": accuracy,
                "mean_cost": 1.0 - accuracy,
            },
        }
        for index, accuracy in enumerate([0.92, 0.94, 0.935])
    ]
    (cohort_b / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (cohort_b / "result.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "error": None,
                "metrics": {
                    "pre_deployment_validation": {
                        "accuracy": 0.95,
                        "mean_cost": 0.05,
                    },
                    "initial_validation": {
                        "accuracy": 0.90,
                        "mean_cost": 0.10,
                    },
                    "selected": {
                        "epoch": 1,
                        "accuracy": 0.94,
                        "value": 0.06,
                    },
                    "learning_rate_selection": {
                        "selected_cell_id": "cell_00",
                        "selected_relative_update_targets": [0.001, 0.003],
                        "selected_learning_rates_by_parameter": {
                            "weight": 1e-7
                        },
                        "edge_selected": True,
                    },
                    "device_programming": {
                        "cohort_b_used": True,
                        "initialization": {
                            "parameters": {
                                "base.dense_weight.0": {
                                    "initial_write": {
                                        "projection_mean_abs_error_s": 1e-6
                                    }
                                }
                            }
                        },
                    },
                    "update_programming": {
                        "active_cohort": "B",
                        "cohort_b_used": True,
                        "parameters": {
                            "base.dense_weight.0": {
                                "projection_mean_abs_error_s": 1e-6
                            }
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    source_test = tmp_path / "source_test.json"
    cohort_b_test = tmp_path / "cohort_b_test.json"
    _write_test(source_test, 0.95)
    _write_test(cohort_b_test, 0.94)
    output = tmp_path / "output"

    summary = summarize_transfer(
        source,
        cohort_b,
        source_test_result=source_test,
        cohort_b_test_result=cohort_b_test,
        output_dir=output,
    )

    assert summary["automatic_accuracy_gates_pass"] is True
    assert summary["cohort_b"]["cohort_b_used"] is True
    assert summary["cohort_b"]["deployment_accuracy_delta"] == pytest.approx(
        -0.05
    )
    assert (output / "summary.json").is_file()
    assert (output / "cohort_b_transfer.png").stat().st_size > 0
