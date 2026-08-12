from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.analyze_bounded_wmax_final_clipping import (
    IncompleteFinalWeightCoverageError,
    analyze,
)
from experiments.analyze_bounded_wmax_sweep import (
    EXPECTED_SOURCE_ARCHIVE_SHA256,
)
from experiments.reporting import MANIFEST_SCHEMA, RESULT_SCHEMA, STATUS_SCHEMA


STUDY_ID = "synthetic-bounded-wmax-final-clipping"
EVIDENCE_CLASS = "ordinary_mnist_bounded_wmax_sensitivity"
SOURCE_COMMIT = "0" * 40


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _study_manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    arm_id = "conv1_wmax_1em4_baseline_sgd_seed0"
    row: dict[str, object] = {
        "architecture": "conv1",
        "arm_id": arm_id,
        "config": f"conv1/00_{arm_id}.json",
        "config_sha256": hashlib.sha256(arm_id.encode()).hexdigest(),
        "epochs": 10,
        "global_array_index": 0,
        "optimizer": "sgd",
        "scheme": "baseline",
        "target": "akib",
        "weight_max": 1.0e-4,
        "weight_min": 1.0e-5,
        "weight_range_ratio": 10.0,
    }
    ordered_digest = hashlib.sha256(
        f"{row['config_sha256']}\n".encode("ascii")
    ).hexdigest()
    manifest = {
        "evidence_class": EVIDENCE_CLASS,
        "ordered_config_set_sha256": ordered_digest,
        "paper_facing": False,
        "parent": {"source_commit": SOURCE_COMMIT},
        "run_count": 1,
        "runs": [row],
        "schema_version": "perfectdiode-bounded-wmax-sweep-config-set/v1",
        "study_id": STUDY_ID,
        "varied_field": "model_base.weight_max",
        "weight_max_values": [1.0e-4],
    }
    path = tmp_path / "study_manifest.json"
    _write_json(path, manifest)
    return path, row


def _canonical_run(root: Path, row: dict[str, object]) -> Path:
    run_dir = root / "run_0"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    lower = np.float32(row["weight_min"])
    upper = np.float32(row["weight_max"])
    adjacent_lower = np.nextafter(lower, np.float32(np.inf))
    np.savez(
        checkpoint_dir / "weights_final.npz",
        ConvWeight_0=np.asarray(
            [lower, upper, adjacent_lower, np.float32(5.0e-5)], dtype=np.float32
        ),
        DenseWeight_0=np.asarray(
            [lower, lower, upper, np.float32(6.0e-5)], dtype=np.float32
        ),
        # Bias values deliberately equal the endpoints; they must be excluded.
        Bias_0=np.asarray([lower, upper], dtype=np.float32),
        param_names=np.asarray(
            ["ConvWeight_0", "DenseWeight_0", "Bias_0"], dtype="U13"
        ),
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": "2026-08-09T00:00:00+00:00",
        "study_id": STUDY_ID,
        "run_id": "run_0",
        "arm_id": row["arm_id"],
        "evidence_class": EVIDENCE_CLASS,
        "smoke": False,
        "configuration": {
            "sha256": row["config_sha256"],
            "epochs": row["epochs"],
            "resolved": {
                "arm_id": row["arm_id"],
                "study_id": STUDY_ID,
                "lab": {"epochs": row["epochs"]},
                "model_base": {
                    "weight_min": row["weight_min"],
                    "weight_max": row["weight_max"],
                },
                "optimizer": {
                    "name": "SGD",
                    "learning_rate": [1.0e-7, 2.0e-6],
                },
            },
        },
        "dataset": {"official_test_read": False},
        "git": {
            "commit": SOURCE_COMMIT,
            "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        },
        "runtime": {"target": row["target"]},
    }
    _write_json(run_dir / "manifest.json", manifest)
    checkpoint = checkpoint_dir / "weights_final.npz"
    result = {
        "schema_version": RESULT_SCHEMA,
        "study_id": STUDY_ID,
        "run_id": "run_0",
        "arm_id": row["arm_id"],
        "evidence_class": EVIDENCE_CLASS,
        "smoke": False,
        "completion": {"criteria_met": True},
        "manifest_sha256": _sha256(run_dir / "manifest.json"),
        "metrics_sha256": _sha256(run_dir / "metrics.jsonl"),
        "terminal_metrics": {
            "best_epoch": 3,
            "validation": {"best_accuracy": 0.85, "final_accuracy": 0.80},
        },
        "artifacts": [
            {
                "kind": "checkpoint",
                "path": "checkpoints/weights_final.npz",
                "sha256": _sha256(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
            }
        ],
    }
    _write_json(run_dir / "result.json", result)
    _write_json(
        run_dir / "status.json",
        {
            "schema_version": STATUS_SCHEMA,
            "study_id": STUDY_ID,
            "run_id": "run_0",
            "arm_id": row["arm_id"],
            "state": "complete",
        },
    )
    return run_dir


def test_exact_endpoint_occupancy_excludes_bias_and_keeps_tolerance_separate(
    tmp_path: Path,
) -> None:
    manifest_path, row = _study_manifest(tmp_path)
    result_root = tmp_path / "results"
    _canonical_run(result_root, row)
    analysis_dir = tmp_path / "analysis"

    report = analyze(
        study_manifest_path=manifest_path,
        result_roots=[result_root],
        analysis_dir=analysis_dir,
        require_complete=True,
    )

    assert report["coverage"]["complete"] is True
    assert report["coverage"]["final_weight_usable_arm_count"] == 1
    assert len(report["parameter_rows"]) == 2
    pooled = report["pooled_rows"][0]
    assert pooled["num_weights"] == 8
    assert pooled["exact_lower_endpoint_count"] == 3
    assert pooled["exact_upper_endpoint_count"] == 2
    assert pooled["exact_either_endpoint_count"] == 5
    assert pooled["exact_either_endpoint_percent"] == pytest.approx(62.5)
    assert pooled["audit_lower_endpoint_count"] == 4
    assert pooled["audit_extra_lower_count"] == 1
    assert pooled["outside_interval_count"] == 0
    assert pooled["best_validation_accuracy_percent"] == pytest.approx(85.0)
    assert pooled["final_validation_accuracy_percent"] == pytest.approx(80.0)
    assert pooled["learning_rate_vector_json"] == "[1e-07,2e-06]"
    assert report["learning_rate_consistency_by_surface"][0][
        "fixed_across_collected_wmax"
    ] is True
    assert (analysis_dir / "final_weight_endpoint_occupancy.csv").is_file()
    assert (
        analysis_dir / "accuracy_and_final_weight_endpoint_occupancy.csv"
    ).is_file()
    assert (analysis_dir / "best_observed_weight_range_by_surface.csv").is_file()
    assert len(report["best_observed_weight_range_by_surface"]) == 2
    assert all(
        value["selected_weight_max"] == pytest.approx(1.0e-4)
        for value in report["best_observed_weight_range_by_surface"]
    )
    assert (analysis_dir / "analysis_summary.json").is_file()
    assert (
        analysis_dir / "accuracy_and_final_endpoint_occupancy_conv1.png"
    ).stat().st_size > 0


def test_corrupted_indexed_checkpoint_is_excluded_and_partial_outputs_survive(
    tmp_path: Path,
) -> None:
    manifest_path, row = _study_manifest(tmp_path)
    result_root = tmp_path / "results"
    run_dir = _canonical_run(result_root, row)
    checkpoint = run_dir / "checkpoints" / "weights_final.npz"
    with checkpoint.open("ab") as handle:
        handle.write(b"corruption")
    analysis_dir = tmp_path / "analysis"

    report = analyze(
        study_manifest_path=manifest_path,
        result_roots=[result_root],
        analysis_dir=analysis_dir,
    )

    assert report["coverage"]["complete"] is False
    assert report["coverage"]["canonical_usable_arm_count"] == 0
    assert report["coverage"]["final_weight_usable_arm_count"] == 0
    assert "artifact hash mismatch" in " ".join(
        report["coverage"]["invalid_runs"][0]["errors"]
    )
    assert (analysis_dir / "final_weight_endpoint_occupancy.csv").is_file()
    assert (analysis_dir / "analysis_summary.json").is_file()

    with pytest.raises(IncompleteFinalWeightCoverageError):
        analyze(
            study_manifest_path=manifest_path,
            result_roots=[result_root],
            analysis_dir=analysis_dir,
            require_complete=True,
        )
