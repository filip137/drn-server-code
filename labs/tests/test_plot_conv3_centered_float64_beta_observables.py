from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments import plot_conv3_centered_float64_beta_observables as plotting


def _source_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    factors = {"baseline": 1.0, "ours": 64.0, "legacy": 4096.0}
    for scheme in plotting.SCHEMES:
        for role in plotting.CHECKPOINT_ROLES:
            for parameter in plotting.PARAMETERS:
                for beta_index, beta in enumerate(plotting.EXPECTED_BETAS):
                    scale = (beta_index + 1) * (plotting.PARAMETERS.index(parameter) + 1)
                    rows.append(
                        {
                            "schema": "test-summary/v1",
                            "eqprop_variant": "centered",
                            "scheme": scheme,
                            "checkpoint_role": role,
                            "precision": "float64",
                            "T": 64,
                            "K": 64,
                            "batch_index": 0,
                            "batch_payload_sha256": "a" * 64,
                            "batch_source_indices_sha256": "b" * 64,
                            "beta_index": beta_index,
                            "actual_base_beta": beta / factors[scheme],
                            "injected_beta": beta,
                            "amplification_factor": factors[scheme],
                            "parameter_name": parameter,
                            "gradient_layer": plotting.GRADIENT_LABELS[parameter],
                            "state_layer_name": plotting.EXPECTED_STATE_NAMES[parameter],
                            "state_layer": plotting.STATE_LABELS[parameter],
                            "bias_excluded": True,
                            "cosine_defined": True,
                            "cosine_undefined_reason": "",
                            "eqprop_vs_bptt_cosine": 1.0 - 0.0001 * scale,
                            "matched_zero_positive_relative_displacement": 1.0e-10 * scale,
                            "matched_zero_positive_delta_rms": 1.0e-11 * scale,
                            "matched_zero_negative_relative_displacement": 1.01e-10 * scale,
                            "matched_zero_negative_delta_rms": 1.01e-11 * scale,
                        }
                    )
    return rows


def _write_source(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_main_writes_complete_filtered_surface_and_plot(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    output = tmp_path / "analysis"
    _write_source(source, _source_rows())

    assert plotting.main(["--input-csv", str(source), "--output-dir", str(output)]) == 0

    output_csv = output / "centered_float64_beta_observables.csv"
    output_plot = output / "plots" / "centered_float64_cosine_relative_absolute_displacement_vs_beta.png"
    output_summary = output / "centered_float64_beta_observables.json"
    with output_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads(output_summary.read_text(encoding="utf-8"))

    assert len(rows) == plotting.EXPECTED_ROWS == 264
    assert output_plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output_plot.stat().st_size > 50_000
    assert summary["coverage"] == {
        "checkpoint_roles": list(plotting.CHECKPOINT_ROLES),
        "complete": True,
        "curve_count": 24,
        "expected_row_count": 264,
        "gradient_layers": [plotting.GRADIENT_LABELS[item] for item in plotting.PARAMETERS],
        "injected_betas": list(plotting.EXPECTED_BETAS),
        "points_per_curve": 11,
        "row_count": 264,
        "schemes": list(plotting.SCHEMES),
        "state_layers": [plotting.STATE_LABELS[item] for item in plotting.PARAMETERS],
    }
    assert summary["outputs"]["filtered_csv"]["sha256"]
    assert summary["outputs"]["plot_png"]["sha256"]
    assert summary["filters"]["plotted_endpoints"] == [
        "positive_nudged_state_matched_to_zero_phase",
        "negative_nudged_state_matched_to_zero_phase",
    ]
    assert summary["definitions"]["absolute_displacement"].startswith("per-node RMS")
    assert rows[0]["matched_zero_negative_delta_rms"]


def test_loader_rejects_missing_centered_float64_coordinate(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    rows = _source_rows()
    _write_source(source, rows[:-1])

    with pytest.raises(ValueError, match="Expected 264 centered-float64 rows"):
        plotting.load_centered_float64_rows(source)


def test_loader_rejects_undefined_cosine(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    rows = _source_rows()
    rows[0]["cosine_defined"] = False
    rows[0]["eqprop_vs_bptt_cosine"] = ""
    rows[0]["cosine_undefined_reason"] = "exact_zero_eqprop_gradient"
    _write_source(source, rows)

    with pytest.raises(ValueError, match="Centered float64 cosine is undefined"):
        plotting.load_centered_float64_rows(source)
