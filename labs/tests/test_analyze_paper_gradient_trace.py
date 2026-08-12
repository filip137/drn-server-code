from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.analyze_paper_gradient_trace import (
    ANOMALY_SCHEMA,
    EXPECTED_ARCHITECTURE_ARMS,
    SUMMARY_SCHEMA,
    _parse_arm,
    _stable_index_sequence_hash,
    analyze,
)
from experiments.gradient_trace import (
    GRADIENT_TRACE_METADATA_SCHEMA,
    GRADIENT_TRACE_SCHEMA,
)
from experiments.paper_gradient_trace_contract import (
    FROZEN_SOURCE_ARCHIVE_SHA256,
    FROZEN_SOURCE_COMMIT,
    FROZEN_STUDY_ID,
    expected_source_config_path,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_study(root: Path) -> None:
    provenance = {
        "schema": "mnist-train-validation-split/v1",
        "source_split": "train",
        "split_seed": 0,
        "shuffle_seed": 0,
        "train_indices_sha256": "a" * 64,
        "validation_indices_sha256": "b" * 64,
        "first_epoch_batch_order_sha256": "c" * 64,
        "train_batch_order_sha256": [f"{index:064x}" for index in range(10)],
    }
    sampled_sources = []
    for epoch in range(1, 11):
        for batch in range(1, 6):
            source_indices = [epoch * 100 + batch, epoch * 100 + batch + 1]
            sampled_sources.append(
                {
                    "epoch": epoch,
                    "batch": batch,
                    "source_indices": source_indices,
                    "source_indices_sha256": _stable_index_sequence_hash(
                        source_indices
                    ),
                }
            )

    run_index = 0
    for architecture, arm_ids in EXPECTED_ARCHITECTURE_ARMS.items():
        for arm_id in sorted(arm_ids):
            optimizer = "Adam" if "_adam_" in arm_id else "SGD"
            source_config_path = expected_source_config_path(arm_id)
            source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
            run_dir = root / "runs" / architecture / f"{run_index:03d}_{arm_id}"
            run_index += 1
            run_dir.mkdir(parents=True)
            manifest = {
                "arm_id": arm_id,
                "smoke": False,
                "study_id": FROZEN_STUDY_ID,
                "git": {
                    "commit": FROZEN_SOURCE_COMMIT,
                    "source_archive_sha256": FROZEN_SOURCE_ARCHIVE_SHA256,
                },
                "configuration": {
                    "path": str(source_config_path),
                    "sha256": _sha256(source_config_path),
                    "resolved": source_config,
                    "configured_epochs": source_config["lab"]["epochs"],
                    "epochs": 10,
                    "diagnostic_overrides": {
                        "gradient_trace_samples_per_epoch": 5,
                        "checkpoint_every_epoch": True,
                        "skip_terminal_official_test": True,
                    },
                },
            }
            status = {"arm_id": arm_id, "state": "complete"}
            result = {
                "arm_id": arm_id,
                "smoke": False,
                "completion": {"criteria_met": True},
            }
            config = {
                "optimizer": {"name": optimizer},
                "training": {"epochs": 10},
            }
            rows = []
            for source in sampled_sources:
                epoch = source["epoch"]
                batch = source["batch"]
                adam_scale = 0.25 if optimizer == "Adam" else 1.0
                adam_cosine = 0.25 if optimizer == "Adam" else 1.0
                rows.append(
                    {
                        "schema_version": GRADIENT_TRACE_SCHEMA,
                        "epoch": epoch,
                        "batch": batch,
                        "total_batches": 5,
                        "global_step": (epoch - 1) * 5 + batch,
                        "source_indices": source["source_indices"],
                        "source_indices_sha256": source["source_indices_sha256"],
                        "loss": 0.5 / epoch,
                        "optimizer": optimizer,
                        "parameter_name": "ConvWeight_0",
                        "learning_rate": 0.01,
                        "fresh_shadow_proposal_kind": (
                            "fresh_adam" if optimizer == "Adam" else "fresh_sgd"
                        ),
                        "gradient_rms": 1.0e-2 / epoch,
                        "gradient_zero_fraction": 0.1,
                        "raw_optimizer_update_rms": 1.0e-3,
                        "applied_update_rms": 7.0e-4,
                        "raw_optimizer_over_fresh_shadow_rms": adam_scale,
                        "applied_over_fresh_shadow_rms": 0.7 * adam_scale,
                        "raw_optimizer_fresh_shadow_cosine": adam_cosine,
                        "applied_fresh_shadow_cosine": 0.65,
                        "applied_update_over_parameter_rms": 1.0e-2,
                        "applied_update_over_initial_parameter_rms": 1.0e-2,
                        "projection_efficiency_l2": 0.70,
                        "projection_changed_fraction": 0.35,
                        "descent_cosine_raw_optimizer_update": 0.9,
                        "descent_cosine_applied_update": 0.65,
                        "lower_bound_fraction_before": 0.20,
                        "lower_bound_fraction_after": 0.30,
                        "upper_bound_fraction_before": 0.0,
                        "upper_bound_fraction_after": 0.0,
                        "optimizer_step": (
                            (epoch - 1) * 5 + batch if optimizer == "Adam" else None
                        ),
                        "gradient_momentum_cosine": (
                            0.2 if optimizer == "Adam" else None
                        ),
                        "gradient_momentum_sign_disagreement_fraction": (
                            0.4 if optimizer == "Adam" else None
                        ),
                        "adam_exp_avg_rms": 0.01 if optimizer == "Adam" else None,
                        "adam_bias_corrected_exp_avg_rms": (
                            0.02 if optimizer == "Adam" else None
                        ),
                        "adam_exp_avg_sq_mean": (
                            1.0e-4 if optimizer == "Adam" else None
                        ),
                    }
                )
            trace_path = run_dir / "gradient_trace.jsonl"
            trace_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            metadata = {
                "schema_version": GRADIENT_TRACE_METADATA_SCHEMA,
                "status": "complete",
                "epoch_count": 10,
                "effective_batches_per_epoch": 5,
                "samples_per_epoch": 5,
                "sampled_batch_positions": [1, 2, 3, 4, 5],
                "sampled_source_batches": sampled_sources,
                "parameter_names": ["ConvWeight_0"],
                "dataset_provenance": provenance,
                "recorded_steps": 50,
                "recorded_rows": 50,
                "trace_sha256": _sha256(trace_path),
                "trace_size_bytes": trace_path.stat().st_size,
            }
            _write_json(run_dir / "manifest.json", manifest)
            _write_json(run_dir / "status.json", status)
            _write_json(run_dir / "result.json", result)
            _write_json(run_dir / "config.json", config)
            _write_json(run_dir / "gradient_trace_metadata.json", metadata)


@pytest.mark.parametrize(
    ("arm_id", "expected"),
    [
        ("conv1_baseline_sgd_seed0", ("conv1", "baseline", "SGD", 0)),
        ("conv2_ours_adam_seed7", ("conv2", "ours", "Adam", 7)),
        ("conv3_legacy_sgd_seed12", ("conv3", "legacy", "SGD", 12)),
    ],
)
def test_arm_parser_preserves_canonical_optimizer_labels(arm_id, expected) -> None:
    assert _parse_arm(arm_id) == expected


def test_analyzer_writes_summaries_plots_and_optimizer_anomalies(tmp_path) -> None:
    study_root = tmp_path / "study"
    output_dir = tmp_path / "analysis"
    _make_study(study_root)

    summary = analyze(study_root, output_dir)

    assert summary["schema_version"] == SUMMARY_SCHEMA
    assert summary["run_count"] == 12
    assert summary["frozen_provenance_audit"]["status"] == "pass"
    assert summary["cohort_and_epoch_order_audit"]["status"] == "pass"
    assert (
        summary["optimizer_audit"]["sgd_fresh_shadow_verification"]["status"]
        == "pass"
    )
    assert summary["optimizer_audit"]["adam_actual_update_and_momentum"][
        "checked_row_count"
    ] == 150
    assert (output_dir / "run_inventory.csv").is_file()
    assert (output_dir / "gradient_trace_epoch_parameter_summary.csv").is_file()
    assert (output_dir / "gradient_trace_summary.json").is_file()
    summary_text = (output_dir / "gradient_trace_summary.json").read_text(
        encoding="utf-8"
    )
    assert "NaN" not in summary_text
    assert json.loads(summary_text)["schema_version"] == SUMMARY_SCHEMA
    anomaly_report = json.loads(
        (output_dir / "gradient_trace_anomalies.json").read_text(encoding="utf-8")
    )
    assert anomaly_report["schema_version"] == ANOMALY_SCHEMA
    assert anomaly_report["counts_by_kind"][
        "adam_actual_fresh_scale_divergence"
    ] == 30
    assert anomaly_report["counts_by_kind"][
        "adam_actual_fresh_direction_divergence"
    ] == 30
    assert anomaly_report["counts_by_kind"][
        "adam_gradient_momentum_sign_disagreement"
    ] == 30
    assert anomaly_report["counts_by_kind"]["low_projection_efficiency"] == 120
    for plot in summary["outputs"]["plots"]:
        assert (output_dir / plot).stat().st_size > 0


def test_sgd_fresh_shadow_audit_checks_direction_as_well_as_scale(tmp_path) -> None:
    study_root = tmp_path / "study"
    _make_study(study_root)
    run_dir = next(
        path.parent
        for path in study_root.rglob("gradient_trace.jsonl")
        if "_sgd_" in path.parent.name
    )
    trace_path = run_dir / "gradient_trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    rows[0]["raw_optimizer_fresh_shadow_cosine"] = -1.0
    trace_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata_path = run_dir / "gradient_trace_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["trace_sha256"] = _sha256(trace_path)
    metadata["trace_size_bytes"] = trace_path.stat().st_size
    _write_json(metadata_path, metadata)

    summary = analyze(study_root, tmp_path / "analysis")

    audit = summary["optimizer_audit"]["sgd_fresh_shadow_verification"]
    assert audit["status"] == "fail"
    assert audit["maximum_cosine_deviation_from_one"] == pytest.approx(2.0)


def test_trace_schema_rejects_missing_principal_metric(tmp_path) -> None:
    study_root = tmp_path / "study"
    _make_study(study_root)
    run_dir = next(path.parent for path in study_root.rglob("gradient_trace.jsonl"))
    trace_path = run_dir / "gradient_trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    del rows[0]["gradient_rms"]
    trace_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata_path = run_dir / "gradient_trace_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["trace_sha256"] = _sha256(trace_path)
    metadata["trace_size_bytes"] = trace_path.stat().st_size
    _write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="Missing trace metric 'gradient_rms'"):
        analyze(study_root, tmp_path / "analysis")


def test_analyzer_rejects_cross_arm_epoch_order_drift(tmp_path) -> None:
    study_root = tmp_path / "study"
    _make_study(study_root)
    manifest_path = next((study_root / "runs" / "conv3").glob("*/manifest.json"))
    run_dir = manifest_path.parent
    metadata_path = run_dir / "gradient_trace_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset_provenance"]["train_batch_order_sha256"][4] = "f" * 64
    _write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="cohort or epoch-order mismatch"):
        analyze(study_root, tmp_path / "analysis")


def test_analyzer_rejects_frozen_source_provenance_drift(tmp_path) -> None:
    study_root = tmp_path / "study"
    _make_study(study_root)
    manifest_path = next((study_root / "runs" / "conv1").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["git"]["source_archive_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="source archive SHA-256 mismatch"):
        analyze(study_root, tmp_path / "analysis")


def test_analyzer_rejects_smoke_or_missing_trace(tmp_path) -> None:
    study_root = tmp_path / "study"
    _make_study(study_root)
    result_path = next((study_root / "runs" / "conv1").glob("*/result.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["smoke"] = True
    _write_json(result_path, result)
    with pytest.raises(ValueError, match="smoke bundle"):
        analyze(study_root, tmp_path / "analysis-smoke")

    result["smoke"] = False
    _write_json(result_path, result)
    (result_path.parent / "gradient_trace.jsonl").unlink()
    with pytest.raises(ValueError, match="Missing or empty gradient trace"):
        analyze(study_root, tmp_path / "analysis-missing")
