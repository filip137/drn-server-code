from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.mnist_conv import update_ratio_diagnostic as diagnostic


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "configs/conv/run_v1.diagnostic.example.json"
ENTRY_POINTS = {
    "experiments/create_mnist_bp_conv_tk_preflight.py": {
        "--base-run", "--calibration-records", "--output-sweep",
    },
    "experiments/diagnose_mnist_bp_conv_lr_update_ratio.py": {
        "--sweep", "--run-bundle", "--output-dir",
    },
    "experiments/diagnose_mnist_bp_conv_hardsigmoid_fixed_lr.py": {
        "--sweep", "--run-bundle", "--output-dir",
    },
}


@pytest.mark.parametrize(("relative_path", "expected_flags"), ENTRY_POINTS.items())
def test_compatibility_entry_points_have_canonical_help_only(relative_path, expected_flags):
    script = REPO_ROOT / relative_path
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert expected_flags <= set(completed.stdout.split())
    assert "train_mnist_bp_conv_amplification_sweep import" not in script.read_text()
    for retired_flag in (
        "--learning-rate", "--dataset-root", "--device",
        "--num-iterations", "--manifest-csv", "--output-root",
    ):
        assert retired_flag not in completed.stdout


def _write_weights(path: Path, values: np.ndarray) -> None:
    np.savez(
        path,
        weight_0=np.asarray(values, dtype=np.float32),
        param_names=np.asarray(["weight_0"]),
        param_types=np.asarray(["model.variable.parameter.Weight"]),
        param_shapes_json=np.asarray(json.dumps([list(values.shape)])),
        metadata_json=np.asarray("{}"),
    )


def _write_readable_bundle(bundle: Path) -> None:
    (bundle / "weights").mkdir(parents=True)
    (bundle / "config.resolved.json").write_text(EXAMPLE.read_text(), encoding="utf-8")
    (bundle / "metrics.json").write_text(
        json.dumps({
            "best_epoch": 1,
            "best_test_accuracy": 0.9,
            "final_test_accuracy": 0.8,
        }),
        encoding="utf-8",
    )
    with (bundle / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch", "train_loss", "train_accuracy", "test_loss",
                "test_accuracy", "learning_rate",
            ),
        )
        writer.writeheader()
        writer.writerow({
            "epoch": 1,
            "train_loss": 1.0,
            "train_accuracy": 0.5,
            "test_loss": 0.8,
            "test_accuracy": 0.8,
            "learning_rate": json.dumps([0.01, 0.02, 0.03]),
        })
    _write_weights(bundle / "weights/best.npz", np.asarray([3.0, 4.0]))
    _write_weights(bundle / "weights/final.npz", np.asarray([6.0, 8.0]))


def test_direct_bundle_diagnostic_reads_canonical_artifacts_and_writes_atomic_reports(
    tmp_path,
    monkeypatch,
):
    bundle = tmp_path / "bundle"
    _write_readable_bundle(bundle)
    run_id = "run_" + "1" * 64
    monkeypatch.setattr(
        diagnostic,
        "validate_bundle",
        lambda path, expected_run_id=None: {"run_id": run_id},
    )

    records = diagnostic.load_records(run_bundles=[bundle])
    result = diagnostic.analyze_records(
        records,
        tmp_path / "analysis",
        required_nonlinearity="hard_sigmoid",
    )

    assert result["schema_version"] == diagnostic.DIAGNOSTIC_SCHEMA_VERSION
    assert result["run_count"] == 1
    assert result["parameter_rows"] == 1
    assert result["runs"][0]["final_learning_rate"] == [0.01, 0.02, 0.03]
    assert result["runs"][0]["parameter_tensor_count"] == 1
    assert result["runs"][0]["parameter_scalar_count"] == 2
    with (tmp_path / "analysis/checkpoint_update_ratios.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["run_id"] == run_id
    assert row["element_count"] == "2"
    assert math.isclose(float(row["symmetric_relative_delta"]), 0.5)
    assert math.isclose(float(row["cosine"]), 1.0)
    assert not Path(row["run_bundle"]).is_absolute()
    written = json.loads((tmp_path / "analysis/diagnostic.json").read_text())
    assert written == result


def test_diagnostic_rejects_noncanonical_input_mode_combinations():
    with pytest.raises(diagnostic.DiagnosticInputError, match="exactly one input mode"):
        diagnostic.load_records()
    with pytest.raises(diagnostic.DiagnosticInputError, match="exactly one input mode"):
        diagnostic.load_records(run_bundles=["run"], sweep_dir="sweep")


def test_diagnostic_never_writes_inside_an_immutable_run_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    _write_readable_bundle(bundle)
    monkeypatch.setattr(
        diagnostic,
        "validate_bundle",
        lambda path, expected_run_id=None: {"run_id": "run_" + "2" * 64},
    )
    records = diagnostic.load_records(run_bundles=[bundle])

    with pytest.raises(diagnostic.DiagnosticInputError, match="outside every immutable"):
        diagnostic.analyze_records(records, bundle / "analysis")

    assert not (bundle / "analysis").exists()


def test_cli_reports_invalid_bundle_without_a_traceback(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/diagnose_mnist_bp_conv_lr_update_ratio.py"),
            "--run-bundle",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "analysis"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "Expected manifest.json as completion marker" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (tmp_path / "analysis").exists()


def test_zero_checkpoints_use_blank_cosine_instead_of_nan(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    _write_readable_bundle(bundle)
    _write_weights(bundle / "weights/best.npz", np.zeros(2))
    _write_weights(bundle / "weights/final.npz", np.zeros(2))
    monkeypatch.setattr(
        diagnostic,
        "validate_bundle",
        lambda path, expected_run_id=None: {"run_id": "run_" + "3" * 64},
    )

    result = diagnostic.analyze_records(
        diagnostic.load_records(run_bundles=[bundle]),
        tmp_path / "analysis",
    )

    assert result["runs"][0]["min_cosine"] is None
    parameter_csv = (tmp_path / "analysis/checkpoint_update_ratios.csv").read_text()
    assert "nan" not in parameter_csv.lower()
