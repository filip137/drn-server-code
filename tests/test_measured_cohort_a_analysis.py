from __future__ import annotations

import json
from pathlib import Path

from labs.tools.plot_measured_cohort_a_comparison import compare


def _write_run(path: Path, accuracies: list[float]) -> None:
    path.mkdir()
    rows = [
        {
            "mode": "train",
            "completed_epochs": index + 1,
            "validation": {
                "accuracy": accuracy,
                "mean_cost": 1.0 - accuracy,
            },
        }
        for index, accuracy in enumerate(accuracies)
    ]
    (path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = {
        "status": "complete",
        "error": None,
        "metrics": {
            "completed_epochs": len(rows),
            "selected": {
                "epoch": len(rows) - 1,
                "accuracy": accuracies[-1],
                "value": 1.0 - accuracies[-1],
            },
            "learning_rate_selection": {
                "selected_cell_id": "cell_00",
                "selected_relative_update_targets": [0.001, 0.003],
                "selected_learning_rates_by_parameter": {"weight": 1e-7},
                "edge_selected": True,
            },
            "update_programming": {
                "parameters": {"base.dense_weight.0": {"step_count": 2}}
            },
        },
    }
    (path / "result.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )


def _write_test_result(path: Path, accuracy: float) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "metrics": {
                    "accuracy": accuracy,
                    "split_protocol": {"effective": "held_out_test"},
                },
            }
        ),
        encoding="utf-8",
    )


def test_comparison_writes_matplotlib_plot_and_feasibility_summary(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    isotonic = tmp_path / "isotonic"
    _write_run(raw, [0.91, 0.92])
    _write_run(isotonic, [0.92, 0.94])
    raw_test = tmp_path / "raw_test.json"
    isotonic_test = tmp_path / "isotonic_test.json"
    _write_test_result(raw_test, 0.915)
    _write_test_result(isotonic_test, 0.935)
    output = tmp_path / "comparison"

    summary = compare(
        raw,
        isotonic,
        raw_test_result=raw_test,
        isotonic_test_result=isotonic_test,
        output_dir=output,
    )

    assert summary["automatic_accuracy_gates_pass"] is True
    assert summary["cohort_b_used"] is False
    assert summary["raw"]["official_test"]["accuracy"] == 0.915
    assert (output / "summary.json").is_file()
    assert (output / "validation_comparison.png").stat().st_size > 0
