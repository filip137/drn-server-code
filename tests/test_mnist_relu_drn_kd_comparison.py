from __future__ import annotations

import json
from pathlib import Path

from labs.tools.plot_mnist_relu_drn_kd_comparison import compare


def _write_run(
    path: Path,
    *,
    encoding: str,
    backend: str,
    initial_kl: float,
    selected_kl: float,
    accuracy: float,
    agreement: float,
) -> None:
    path.mkdir()
    initial = {
        "kl_teacher_student": initial_kl,
        "student_accuracy": accuracy - 0.01,
        "teacher_agreement": agreement - 0.01,
    }
    selected = {
        "epoch": 0,
        "kl_teacher_student": selected_kl,
        "student_accuracy": accuracy,
        "teacher_agreement": agreement,
    }
    rows = [
        {"mode": "initialization", "validation": initial},
        {
            "mode": "train",
            "completed_epochs": 1,
            "validation": selected,
        },
    ]
    (path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = {
        "status": "complete",
        "error": None,
        "metrics": {
            "encoding": encoding,
            "update_backend": backend,
            "initial_validation": initial,
            "selected": selected,
            "last_validation": selected,
            "relative_kl_recovery_from_initialization": (
                initial_kl - selected_kl
            )
            / initial_kl,
            "acceptance_gate": {"passed": True},
        },
    }
    (path / "result.json").write_text(json.dumps(result), encoding="utf-8")


def test_comparison_reports_two_device_criterion_and_writes_plot(
    tmp_path: Path,
) -> None:
    single = tmp_path / "single"
    differential = tmp_path / "differential"
    _write_run(
        single,
        encoding="single",
        backend="ideal",
        initial_kl=0.01,
        selected_kl=0.004,
        accuracy=0.970,
        agreement=0.990,
    )
    _write_run(
        differential,
        encoding="differential",
        backend="ideal",
        initial_kl=0.01,
        selected_kl=0.003,
        accuracy=0.969,
        agreement=0.989,
    )
    output = tmp_path / "analysis"

    summary = compare(
        {"ideal_single": single, "ideal_differential": differential},
        test_results={},
        output_dir=output,
    )

    ideal = summary["encoding_comparisons"]["ideal"]
    assert ideal["two_device_better"] is True
    assert ideal["relative_kl_reduction_differential_vs_single"] == 0.25
    assert (output / "summary.json").is_file()
    assert (output / "validation_comparison.png").stat().st_size > 0
