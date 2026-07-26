from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from labs.tools.curate_mnist_conv_lr_active_v3 import (
    CSV_FIELDS,
    compose_active_rows,
    derive_baseline_relative_rho,
    plot_active_rows,
)


def _frozen_rows() -> list[dict]:
    values = []
    for architecture, base_gain, ours_gain, legacy_gain in (
        ("conv1", 75.0, 85.0, 32.0),
        ("conv2", 253.0, 716.0, 661.0),
    ):
        for scheme, gain, suffix in (
            ("baseline", base_gain, "baseline_v1_c1"),
            ("ours", ours_gain, "ours_v4_c1"),
            ("legacy", legacy_gain, "legacy_v4_c0p25"),
        ):
            values.append(
                {
                    "row_id": f"{architecture}_{suffix}",
                    "architecture": architecture,
                    "scheme": scheme,
                    "input_gain": gain,
                    "inference_iterations": 4 if architecture == "conv1" else 16,
                    "training_iterations": 4 if architecture == "conv1" else 6,
                }
            )
    return values


def _selection_row(frozen: dict, *, status: str, value: float | None) -> dict:
    return {
        **frozen,
        "status": status,
        "reason": None if status == "frozen_seed0_screen" else "synthetic_unresolved",
        "selected_candidate_role": "fast" if value is not None else None,
        "selected_peak_learning_rate": value,
        "selected_rho_target": None if value is None else value / 10.0,
        "observed_peak_rho": value,
        "observed_peak_rho_relative": value,
        "observed_peak_rho_span": None if value is None else value / 100.0,
        "final_validation_loss": None if value is None else 0.3,
        "final_validation_accuracy": None if value is None else 0.6,
    }


def test_derives_worst_parameter_q90_from_exact_final_warmup_window(
    tmp_path: Path,
) -> None:
    study = tmp_path / "v1"
    initialization = study / "initialization"
    diagnostics_dir = (
        study / "stages/candidates/entries/conv1_baseline_v1_c1--fast"
    )
    initialization.mkdir(parents=True)
    diagnostics_dir.mkdir(parents=True)
    metadata = {
        "parameter_diagnostics": {
            "WeightA": {"bounded_gate": True, "rms": 2.0},
            "WeightB": {"bounded_gate": True, "rms": 4.0},
            "Bias": {"bounded_gate": False, "rms": 0.0},
        }
    }
    (initialization / "conv1.json").write_text(json.dumps(metadata), encoding="utf-8")
    path = diagnostics_dir / "parameter_diagnostics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "parameter", "bounded_gate", "proposed_update_rms"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "step": 828,
                "parameter": "WeightA",
                "bounded_gate": True,
                "proposed_update_rms": 1e9,
            }
        )
        for offset, step in enumerate(range(829, 861), start=1):
            writer.writerow(
                {
                    "step": step,
                    "parameter": "WeightA",
                    "bounded_gate": True,
                    "proposed_update_rms": 2.0 * offset,
                }
            )
            writer.writerow(
                {
                    "step": step,
                    "parameter": "WeightB",
                    "bounded_gate": True,
                    "proposed_update_rms": 2.0 * offset,
                }
            )

    result = derive_baseline_relative_rho(
        study,
        row_id="conv1_baseline_v1_c1",
        architecture="conv1",
        candidate_role="fast",
    )

    assert result["rho_relative_by_parameter"] == pytest.approx(
        {"WeightA": 28.9, "WeightB": 14.45}
    )
    assert result["rho_relative"] == pytest.approx(28.9)
    assert result["limiting_parameter"] == "WeightA"
    assert result["window_start_step"] == 829
    assert result["window_end_step"] == 860
    assert result["window_steps_per_parameter"] == 32
    assert result["candidate_diagnostics"]["sha256"]
    assert result["initialization_metadata"]["sha256"]


def test_composes_six_rows_without_mixing_native_rho_coordinates() -> None:
    frozen = _frozen_rows()
    v1_rows = []
    v3_rows = []
    baseline_derivations = {}
    for index, row in enumerate(frozen, start=1):
        if row["scheme"] == "baseline":
            selected = _selection_row(row, status="frozen_seed0_screen", value=float(index))
            v1_rows.append(selected)
            baseline_derivations[row["row_id"]] = {
                "rho_relative": index / 10.0,
                "rho_span_rederived": float(index),
            }
        else:
            v1_rows.append(_selection_row(row, status="unresolved", value=None))
            status = "unresolved" if row["scheme"] == "legacy" else "frozen_seed0_screen"
            value = None if status == "unresolved" else float(index)
            v3_rows.append(_selection_row(row, status=status, value=value))

    rows = compose_active_rows(
        frozen_six_rows=frozen,
        v1_selection={"schema_version": "mnist-conv-lr-selection/v1", "rows": v1_rows},
        v3_selection={"schema_version": "mnist-conv-lr-selection/v2", "rows": v3_rows},
        v1_study_id="lrstudy_" + "1" * 64,
        v3_study_id="lrstudy_" + "3" * 64,
        v1_selection_sha256="a" * 64,
        v3_selection_sha256="b" * 64,
        baseline_derivations=baseline_derivations,
    )

    assert len(rows) == 6
    assert list(rows[0]) == CSV_FIELDS
    assert [row["row_id"] for row in rows] == [row["row_id"] for row in frozen]
    baselines = [row for row in rows if row["scheme"] == "baseline"]
    amplified = [row for row in rows if row["scheme"] != "baseline"]
    assert all(
        row["selected_rho_target_coordinate"] == "conductance_bound_span"
        for row in baselines
    )
    assert all(
        row["observed_peak_rho_relative_source"]
        == "rederived_from_v1_candidate_log"
        for row in baselines
    )
    assert all(
        row["selected_rho_target_coordinate"] == "initial_parameter_rms"
        for row in amplified
    )
    assert all(
        row["observed_peak_rho_relative_source"] == "native_v3_selection"
        for row in amplified
    )
    assert baselines[0]["selected_peak_learning_rate"] == 1.0
    assert baselines[1]["selected_peak_learning_rate"] == 4.0


def test_writes_two_panel_log_plot_with_unresolved_rows_omitted(
    tmp_path: Path,
) -> None:
    rows = []
    for index, row in enumerate(_frozen_rows(), start=1):
        rows.append(
            {
                **row,
                "selected_peak_learning_rate": None if row["scheme"] == "legacy" else index / 10.0,
            }
        )
    path = tmp_path / "active.png"

    plot_active_rows(
        rows,
        path,
        field="selected_peak_learning_rate",
        ylabel="Selected raw peak learning rate",
    )

    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
