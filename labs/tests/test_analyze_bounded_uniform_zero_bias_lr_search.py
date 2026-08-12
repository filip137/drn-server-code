from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.analyze_bounded_uniform_zero_bias_lr_search import (
    ANALYSIS_SCHEMA,
    CONV3_RECEIPT_SCHEMA,
    CONV3_RESOLVED_SCHEMA,
    CONV3_SELECTION_SCHEMA,
    CONV3_STUDY_SCHEMA,
    IncompleteCoverageError,
    _study_contract,
    analyze,
)
from experiments.reporting import MANIFEST_SCHEMA, RESULT_SCHEMA, STATUS_SCHEMA


STUDY_ID = "synthetic-bounded-uniform-zero-bias-lr-search"
SOURCE_COMMIT = "a" * 40
SOURCE_ARCHIVE_SHA256 = "b" * 64
WEIGHT_MIN = 1.0e-5
WEIGHT_MAX = 1.0e-4


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _surfaces(optimizers: list[str]) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "initializer": "bounded_uniform",
            "architecture": "conv1",
            "scheme": "baseline",
            "optimizer": optimizer,
            "surface_id": f"bounded_uniform__conv1__baseline__{optimizer.lower()}",
        }
        for index, optimizer in enumerate(optimizers)
    ]


def _study_config(tmp_path: Path, optimizers: list[str]) -> tuple[Path, list[dict[str, object]]]:
    surfaces = _surfaces(optimizers)
    path = tmp_path / "study.json"
    _write_json(
        path,
        {
            "schema_version": (
                "perfectdiode-conv123-bounded-uniform-zero-bias-rho-study/v1"
            ),
            "study_id": STUDY_ID,
            "evidence_class": "ordinary_mnist_selection",
            "scope": {
                "initializers": ["bounded_uniform"],
                "architectures": ["conv1"],
                "schemes": ["baseline"],
                "optimizers": optimizers,
                "excluded": [],
            },
            "bias_contract": {
                "initialization": "default_zero",
                "learning_rate": 0.0,
                "conductance_projection": False,
            },
            "model": {
                "weight_min": WEIGHT_MIN,
                "weight_max": WEIGHT_MAX,
            },
            "rho_search": {
                "minimum_validation_accuracy": 0.9,
                "select_best_safe_below_accuracy": False,
            },
        },
    )
    return path, surfaces


def test_conv3_successor_schema_resolves_dynamic_analysis_contract(
    tmp_path: Path,
) -> None:
    config_path, _ = _study_config(tmp_path, ["SGD", "Adam"])
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["schema_version"] = CONV3_STUDY_SCHEMA
    payload["scope"]["architectures"] = ["conv3"]
    payload["scope"]["schemes"] = ["baseline", "ours", "legacy"]
    payload["scope"]["excluded"] = ["conv1", "conv2"]
    _write_json(config_path, payload)

    contract = _study_contract(config_path)

    assert contract.architectures == ("conv3",)
    assert len(contract.surfaces) == 6
    assert contract.resolved_schema == CONV3_RESOLVED_SCHEMA
    assert contract.selection_schema == CONV3_SELECTION_SCHEMA
    assert contract.transport_receipt_schema == CONV3_RECEIPT_SCHEMA


def _shard(
    tmp_path: Path,
    config_path: Path,
    surfaces: list[dict[str, object]],
) -> tuple[Path, Path, Path]:
    study_root = tmp_path / "results" / STUDY_ID
    shard = study_root / "shards" / "main-r1"
    receipt_root = study_root / "transport_receipts" / "main-r1"
    init_path = shard / "assets" / "bounded_uniform" / "conv1" / "final_model.pt"
    init_path.parent.mkdir(parents=True)
    init_path.write_bytes(b"synthetic-shared-initialization")
    _write_json(
        shard / "study.resolved.json",
        {
            "schema_version": (
                "perfectdiode-conv123-bounded-uniform-zero-bias-rho-resolved/v1"
            ),
            "study_id": STUDY_ID,
            "study_config_sha256": _sha256(config_path),
            "official_test_read": False,
            "target": "main-r1",
            "surface_count": len(surfaces),
            "surfaces": surfaces,
            "code": {
                "commit": SOURCE_COMMIT,
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "source_kind": "frozen_archive",
            },
        },
    )
    return study_root, shard, receipt_root


def _cell_metadata(
    *,
    index: int,
    optimizer: str,
    rho_conv: float,
    rho_dense: float,
    status: str,
    selection_eligible: bool | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "index": index,
        "optimizer": optimizer,
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "status": status,
        "learning_rate_vector": [1.0e-7, 2.0e-7, 0.0],
        "learning_rates_by_parameter": {
            "ConvWeight_0": 1.0e-7,
            "DenseWeight_0": 2.0e-7,
            "Bias_0": 0.0,
        },
    }
    if selection_eligible is not None:
        payload["selection_eligible"] = selection_eligible
    return payload


def _manifest(
    *,
    spec: dict[str, object],
    cell: dict[str, object],
    state: str,
) -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": "2026-08-10T00:00:00+00:00",
        "study_id": STUDY_ID,
        "run_id": f"cell-{cell['index']}",
        "arm_id": f"{str(spec['optimizer']).lower()}-rho",
        "evidence_class": "ordinary_mnist_selection",
        "smoke": False,
        "command": {
            "surface_index": spec["index"],
            "cell_index": cell["index"],
        },
        "configuration": {
            "resolved": {
                "optimizer": {"name": spec["optimizer"]},
                "model_base": {
                    "weight_init_mode": "bounded_uniform",
                    "weight_min": WEIGHT_MIN,
                    "weight_max": WEIGHT_MAX,
                },
                "bias_contract": {"learning_rate": 0.0},
            }
        },
        "dataset": {"official_test_read": False},
        "git": {
            "commit": SOURCE_COMMIT,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        },
        "runtime": {"target": "main-r1", "state": state},
    }


def _canonical_complete_cell(
    surface_dir: Path,
    spec: dict[str, object],
    *,
    accuracy: float,
    index: int = 66,
    cell_id: str | None = None,
    rho_conv: float = 0.001,
    rho_dense: float = 1.0 / 300.0,
    selection_eligible: bool = False,
    conv_values: list[float] | None = None,
    dense_values: list[float] | None = None,
) -> tuple[Path, dict[str, object]]:
    if cell_id is None:
        cell_id = (
            "066_rc_0p001_rd_0p00333333333333"
            if index == 66
            else f"{index:03d}_synthetic"
        )
    run_dir = surface_dir / "rho" / "cells" / cell_id
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    lower = np.float32(WEIGHT_MIN)
    upper = np.float32(WEIGHT_MAX)
    np.savez(
        checkpoint_dir / "weights_final.npz",
        ConvWeight_0=np.asarray(
            conv_values or [lower, upper, 5.0e-5, lower], dtype=np.float32
        ),
        DenseWeight_0=np.asarray(
            dense_values or [upper, 6.0e-5, lower, 7.0e-5], dtype=np.float32
        ),
        Bias_0=np.asarray([lower, upper], dtype=np.float32),
        param_names=np.asarray(
            ["ConvWeight_0", "DenseWeight_0", "Bias_0"], dtype="U13"
        ),
    )
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"epoch": 1, "validation_accuracy": accuracy}) + "\n",
        encoding="utf-8",
    )
    cell = _cell_metadata(
        index=index,
        optimizer=str(spec["optimizer"]),
        rho_conv=rho_conv,
        rho_dense=rho_dense,
        status="complete",
        selection_eligible=selection_eligible,
    )
    _write_json(run_dir / "cell.json", cell)
    manifest = _manifest(spec=spec, cell=cell, state="complete")
    _write_json(run_dir / "manifest.json", manifest)
    checkpoint = checkpoint_dir / "weights_final.npz"
    result = {
        "schema_version": RESULT_SCHEMA,
        "study_id": STUDY_ID,
        "run_id": manifest["run_id"],
        "arm_id": manifest["arm_id"],
        "evidence_class": "ordinary_mnist_selection",
        "smoke": False,
        "dataset": {"official_test_read": False},
        "completion": {
            "criteria_met": accuracy >= 0.9,
            "rho_cell_complete": True,
            "safety_admissible": True,
            "minimum_validation_accuracy": 0.9,
            "official_test_read": False,
        },
        "manifest_sha256": _sha256(run_dir / "manifest.json"),
        "metrics_sha256": _sha256(metrics_path),
        "terminal_metrics": {
            "validation": {
                "final_accuracy": accuracy,
                "final_loss": 0.2,
            }
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
    result_sha256 = _sha256(run_dir / "result.json")
    _write_json(
        run_dir / "status.json",
        {
            "schema_version": STATUS_SCHEMA,
            "study_id": STUDY_ID,
            "run_id": manifest["run_id"],
            "arm_id": manifest["arm_id"],
            "state": "complete",
            "result_sha256": result_sha256,
        },
    )
    return run_dir, cell


def _canonical_rejected_cell(
    surface_dir: Path, spec: dict[str, object]
) -> tuple[Path, dict[str, object]]:
    cell_id = "067_rc_0p001_rd_0p01"
    run_dir = surface_dir / "rho" / "cells" / cell_id
    run_dir.mkdir(parents=True)
    cell = _cell_metadata(
        index=67,
        optimizer=str(spec["optimizer"]),
        rho_conv=0.001,
        rho_dense=0.01,
        status="canary_rejected_safety",
        selection_eligible=None,
    )
    _write_json(run_dir / "cell.json", cell)
    _write_json(run_dir / "manifest.json", _manifest(spec=spec, cell=cell, state="failed"))
    (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    _write_json(
        run_dir / "status.json",
        {
            "schema_version": STATUS_SCHEMA,
            "study_id": STUDY_ID,
            "run_id": f"cell-{cell['index']}",
            "arm_id": f"{str(spec['optimizer']).lower()}-rho",
            "state": "failed",
            "error": {"type": "SafetyRejection", "message": "synthetic"},
        },
    )
    return run_dir, cell


def _terminal_surface(
    *,
    shard: Path,
    receipt_root: Path,
    spec: dict[str, object],
    config_path: Path,
    accuracy: float = 0.89,
) -> Path:
    surface_dir = shard / "surfaces" / str(spec["surface_id"])
    _write_json(
        surface_dir / "rho" / "probe.json",
        {
            "schema_version": "conv-rho-probe/v1",
            "status": "complete",
            "official_test_read": False,
        },
    )
    complete_dir, _ = _canonical_complete_cell(surface_dir, spec, accuracy=accuracy)
    rejected_dir, _ = _canonical_rejected_cell(surface_dir, spec)
    selection = {
        "schema_version": (
            "perfectdiode-conv123-bounded-uniform-zero-bias-rho-surface/v1"
        ),
        **spec,
        "official_test_read": False,
        "status": "unresolved_no_safe_completed_core_candidate",
        "selected": None,
        "selection": {
            "accuracy_gate_met": False,
            "eligible": [],
            "plateau": [],
            "selected": None,
            "selection_basis": "none",
        },
        "candidates": [
            {
                "cell_id": complete_dir.name,
                "index": 66,
                "rho_conv": 0.001,
                "rho_dense": 1.0 / 300.0,
                "status": "complete",
                "selection_eligible": False,
                "final_validation_accuracy": accuracy,
                "final_validation_loss": 0.2,
            },
            {
                "cell_id": rejected_dir.name,
                "index": 67,
                "rho_conv": 0.001,
                "rho_dense": 0.01,
                "status": "canary_rejected_safety",
                "selection_eligible": False,
                "final_validation_accuracy": None,
                "final_validation_loss": None,
            },
        ],
    }
    selection_path = surface_dir / "selection.json"
    _write_json(selection_path, selection)
    init_path = shard / "assets" / "bounded_uniform" / "conv1" / "final_model.pt"
    _write_json(
        receipt_root / f"surface_{spec['index']}.json",
        {
            "schema_version": (
                "perfectdiode-conv123-bounded-uniform-zero-bias-rho-"
                "transport-receipt/v1"
            ),
            "study_id": STUDY_ID,
            "surface_index": spec["index"],
            "surface": {
                key: spec[key]
                for key in (
                    "initializer",
                    "architecture",
                    "scheme",
                    "optimizer",
                    "surface_id",
                )
            },
            "target": "main-r1",
            "environment_id": "synthetic:py312:cpu",
            "source_commit": SOURCE_COMMIT,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "study_config_sha256": _sha256(config_path),
            "selection_path": f"/remote/surfaces/{spec['surface_id']}/selection.json",
            "selection_status": selection["status"],
            "selected_run_dir": None,
            "official_test_read": False,
            "semantic_status": "pass",
            "shared_initialization_checkpoint": (
                "/remote/assets/bounded_uniform/conv1/final_model.pt"
            ),
            "shared_initialization_checkpoint_sha256": _sha256(init_path),
            "conv3_post_training_tk_validation": None,
        },
    )
    return surface_dir


def test_terminal_unresolved_rows_keep_lrs_and_exact_occupancy_without_winner(
    tmp_path: Path,
) -> None:
    config_path, surfaces = _study_config(tmp_path, ["SGD"])
    study_root, shard, receipt_root = _shard(tmp_path, config_path, surfaces)
    _terminal_surface(
        shard=shard,
        receipt_root=receipt_root,
        spec=surfaces[0],
        config_path=config_path,
    )
    output = tmp_path / "analysis"

    report = analyze(
        study_config_path=config_path,
        study_root=study_root,
        output_dir=output,
        require_complete=True,
    )

    assert report["schema_version"] == ANALYSIS_SCHEMA
    assert report["coverage"]["complete"] is True
    assert report["snapshot_status"] == "complete_coverage"
    assert report["scientific_handoff_complete"] is False
    assert report["winner_inferred"] is False
    assert report["unresolved_surface_ids"] == [surfaces[0]["surface_id"]]
    assert report["surfaces"][0]["receipt_status"] == "valid"
    assert report["surfaces"][0]["terminal_status"] == (
        "unresolved_no_safe_completed_core_candidate"
    )
    assert report["surfaces"][0]["terminal_status_interpretation"] == (
        "no_accuracy_eligible_candidate"
    )
    assert report["surfaces"][0][
        "canonical_safety_admissible_completed_candidate_count"
    ] == 1
    assert report["surfaces"][0][
        "canonical_safety_admissible_gate_qualified_candidate_count"
    ] == 0
    assert "not evidence that all candidates were unsafe" in report["surfaces"][0][
        "terminal_status_note"
    ]
    assert report["coverage"]["terminal_status_annotation_count"] == 1
    assert report["terminal_status_annotations"] == [
        {
            "surface_id": surfaces[0]["surface_id"],
            "raw_terminal_status": (
                "unresolved_no_safe_completed_core_candidate"
            ),
            "interpretation": "no_accuracy_eligible_candidate",
            "note": report["surfaces"][0]["terminal_status_note"],
            "canonical_safety_admissible_completed_candidate_count": 1,
            "canonical_safety_admissible_gate_qualified_candidate_count": 0,
        }
    ]
    assert report["surfaces"][0]["highest_observed_accuracy"] == pytest.approx(0.89)
    assert report["surfaces"][0]["gate_qualified_candidate_count"] == 0
    assert report["surfaces"][0]["reportable_selected_cell_id"] is None

    kinds = [row["row_kind"] for row in report["candidate_rows"]]
    assert kinds.count("surface") == 1
    assert kinds.count("probe") == 1
    assert kinds.count("candidate") == 2
    completed = next(
        row
        for row in report["candidate_rows"]
        if row["row_kind"] == "candidate" and row["status"] == "complete"
    )
    assert completed["bundle_valid"] is True
    assert completed["canonical_safety_admissible"] is True
    assert completed["num_bounded_weights"] == 8
    assert completed["exact_lower_bound_count"] == 3
    assert completed["exact_upper_bound_count"] == 2
    assert completed["exact_either_bound_count"] == 5
    assert completed["exact_either_bound_percent"] == pytest.approx(62.5)
    assert completed["all_bias_learning_rates_zero"] is True
    assert json.loads(completed["learning_rate_vector_json"]) == [1.0e-7, 2.0e-7, 0.0]
    assert json.loads(completed["learning_rates_by_parameter_json"])["Bias_0"] == 0.0
    parameter_rows = report["parameter_occupancy_rows"]
    assert [row["parameter_name"] for row in parameter_rows] == [
        "ConvWeight_0",
        "DenseWeight_0",
    ]
    assert all(not row["parameter_name"].startswith("Bias_") for row in parameter_rows)
    conv_parameter = parameter_rows[0]
    assert conv_parameter["learning_rate"] == pytest.approx(1.0e-7)
    assert conv_parameter["num_weights"] == 4
    assert conv_parameter["exact_lower_bound_count"] == 2
    assert conv_parameter["exact_upper_bound_count"] == 1
    assert conv_parameter["exact_either_bound_count"] == 3
    dense_parameter = parameter_rows[1]
    assert dense_parameter["learning_rate"] == pytest.approx(2.0e-7)
    assert dense_parameter["exact_lower_bound_count"] == 1
    assert dense_parameter["exact_upper_bound_count"] == 1
    assert dense_parameter["exact_either_bound_count"] == 2
    assert report["coverage"]["parameter_occupancy_row_count"] == 2
    assert report["accuracy_occupancy_correlations"][0]["status"] == (
        "insufficient_candidates"
    )
    rejected = next(
        row
        for row in report["candidate_rows"]
        if row["row_kind"] == "candidate"
        and row["status"] == "canary_rejected_safety"
    )
    assert rejected["bundle_state"] == "failed"
    assert rejected["bundle_valid"] is True
    assert rejected["checkpoint_path"] is None
    assert rejected["all_bias_learning_rates_zero"] is True

    for name in (
        "candidate_rows.csv",
        "surface_summary.csv",
        "parameter_endpoint_occupancy.csv",
        "parameter_endpoint_occupancy.json",
        "accuracy_occupancy_correlations.csv",
        "summary.json",
        "report.md",
        "plots/validation_accuracy.png",
        "plots/final_bound_occupancy.png",
        "plots/accuracy_vs_either_bound_occupancy.png",
    ):
        assert (output / name).stat().st_size > 0
    report_text = (output / "report.md").read_text(encoding="utf-8")
    assert "never labeled winners" in report_text
    assert "highest observed (below gate; observation, not winner)" in report_text
    assert '"Bias_0":0.0,"ConvWeight_0":1e-07' in report_text
    assert "Raw state" in report_text
    assert "no accuracy-eligible candidate" in report_text
    assert "It does not mean every candidate was unsafe" in report_text


def test_terminal_surface_reports_parameter_occupancy_and_descriptive_correlations(
    tmp_path: Path,
) -> None:
    config_path, surfaces = _study_config(tmp_path, ["SGD"])
    study_root, shard, receipt_root = _shard(tmp_path, config_path, surfaces)
    spec = surfaces[0]
    surface_dir = _terminal_surface(
        shard=shard,
        receipt_root=receipt_root,
        spec=spec,
        config_path=config_path,
    )
    lower = float(np.float32(WEIGHT_MIN))
    upper = float(np.float32(WEIGHT_MAX))
    second_dir, second_cell = _canonical_complete_cell(
        surface_dir,
        spec,
        accuracy=0.92,
        index=68,
        cell_id="068_rc_0p003_rd_0p00333333333333",
        rho_conv=0.003,
        rho_dense=1.0 / 300.0,
        selection_eligible=True,
        conv_values=[lower, upper, 5.0e-5, 6.0e-5],
        dense_values=[lower, upper, 6.0e-5, 7.0e-5],
    )
    third_dir, third_cell = _canonical_complete_cell(
        surface_dir,
        spec,
        accuracy=0.95,
        index=69,
        cell_id="069_rc_0p01_rd_0p00333333333333",
        rho_conv=0.01,
        rho_dense=1.0 / 300.0,
        selection_eligible=True,
        conv_values=[lower, 4.0e-5, 5.0e-5, 6.0e-5],
        dense_values=[upper, 4.0e-5, 6.0e-5, 7.0e-5],
    )
    selection_path = surface_dir / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    for run_dir, cell, accuracy in (
        (second_dir, second_cell, 0.92),
        (third_dir, third_cell, 0.95),
    ):
        selection["candidates"].append(
            {
                "cell_id": run_dir.name,
                "index": cell["index"],
                "rho_conv": cell["rho_conv"],
                "rho_dense": cell["rho_dense"],
                "status": "complete",
                "selection_eligible": True,
                "final_validation_accuracy": accuracy,
                "final_validation_loss": 0.2,
            }
        )
    selection["status"] = "selected"
    selection["selected"] = {"cell_id": third_dir.name}
    selection["selection"] = {
        "accuracy_gate_met": True,
        "eligible": [second_dir.name, third_dir.name],
        "plateau": [third_dir.name],
        "selected": third_dir.name,
        "selection_basis": "synthetic",
    }
    _write_json(selection_path, selection)
    receipt_path = receipt_root / f"surface_{spec['index']}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["selection_status"] = "selected"
    receipt["selected_run_dir"] = (
        f"/remote/surfaces/{spec['surface_id']}/rho/cells/{third_dir.name}"
    )
    _write_json(receipt_path, receipt)

    output = tmp_path / "correlation-analysis"
    report = analyze(
        study_config_path=config_path,
        study_root=study_root,
        output_dir=output,
        require_complete=True,
    )

    assert report["issues"] == []
    assert report["coverage"]["parameter_occupancy_row_count"] == 6
    assert {row["parameter_name"] for row in report["parameter_occupancy_rows"]} == {
        "ConvWeight_0",
        "DenseWeight_0",
    }
    correlation = report["accuracy_occupancy_correlations"][0]
    assert correlation["canonical_numeric_candidate_count"] == 3
    assert correlation["status"] == "computed"
    assert correlation["descriptive_non_causal"] is True
    assert correlation[
        "pearson_r_accuracy_vs_exact_either_bound_percent"
    ] < -0.9
    assert correlation[
        "spearman_r_accuracy_vs_exact_either_bound_percent"
    ] == pytest.approx(-1.0)
    surface = report["surfaces"][0]
    assert surface["highest_observed_row"]["cell_id"] == third_dir.name
    assert surface["reportable_selected_row"]["cell_id"] == third_dir.name
    assert surface["highest_observed_row"]["exact_either_bound_percent"] == pytest.approx(
        25.0
    )
    report_text = (output / "report.md").read_text(encoding="utf-8")
    assert "runner-selected (meets gate)" in report_text
    assert '"ConvWeight_0":1e-07' in report_text
    assert "descriptive associations, not causal evidence" in report_text
    assert (output / "accuracy_occupancy_correlations.csv").stat().st_size > 0
    assert (
        output / "plots/accuracy_vs_either_bound_occupancy.png"
    ).stat().st_size > 0


def test_partial_snapshot_preserves_probe_running_and_missing_surfaces(
    tmp_path: Path,
) -> None:
    config_path, surfaces = _study_config(tmp_path, ["SGD", "Adam"])
    study_root, shard, _ = _shard(tmp_path, config_path, surfaces)
    spec = surfaces[0]
    surface_dir = shard / "surfaces" / str(spec["surface_id"])
    _write_json(
        surface_dir / "rho" / "probe.json",
        {"status": "complete", "official_test_read": False},
    )
    run_dir = surface_dir / "rho" / "cells" / "066_running"
    cell = _cell_metadata(
        index=66,
        optimizer="SGD",
        rho_conv=0.001,
        rho_dense=1.0 / 300.0,
        status="running",
        selection_eligible=None,
    )
    _write_json(run_dir / "cell.json", cell)
    _write_json(run_dir / "manifest.json", _manifest(spec=spec, cell=cell, state="running"))
    (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    _write_json(
        run_dir / "status.json",
        {
            "schema_version": STATUS_SCHEMA,
            "study_id": STUDY_ID,
            "run_id": "cell-66",
            "arm_id": "sgd-rho",
            "state": "running",
        },
    )
    output = tmp_path / "partial-analysis"

    report = analyze(
        study_config_path=config_path,
        shard_roots=[shard],
        output_dir=output,
    )

    assert report["snapshot_status"] == "partial"
    assert report["coverage"]["complete"] is False
    assert report["coverage"]["terminal_surface_count"] == 0
    assert report["coverage"]["in_progress_surface_ids"] == [spec["surface_id"]]
    assert report["coverage"]["missing_surface_ids"] == [surfaces[1]["surface_id"]]
    assert any(
        row["row_kind"] == "probe" and row["status"] == "complete"
        for row in report["candidate_rows"]
    )
    assert any(
        row["row_kind"] == "candidate" and row["status"] == "running"
        for row in report["candidate_rows"]
    )
    assert any(
        row["row_kind"] == "surface" and row["status"] == "not_discovered"
        for row in report["candidate_rows"]
    )
    assert report["winner_inferred"] is False

    strict_output = tmp_path / "strict-partial-analysis"
    with pytest.raises(IncompleteCoverageError):
        analyze(
            study_config_path=config_path,
            shard_roots=[shard],
            output_dir=strict_output,
            require_complete=True,
        )
    assert (strict_output / "summary.json").is_file()


def test_corrupted_indexed_checkpoint_invalidates_completed_bundle_and_coverage(
    tmp_path: Path,
) -> None:
    config_path, surfaces = _study_config(tmp_path, ["SGD"])
    study_root, shard, receipt_root = _shard(tmp_path, config_path, surfaces)
    surface_dir = _terminal_surface(
        shard=shard,
        receipt_root=receipt_root,
        spec=surfaces[0],
        config_path=config_path,
    )
    checkpoint = (
        surface_dir
        / "rho/cells/066_rc_0p001_rd_0p00333333333333/checkpoints/weights_final.npz"
    )
    with checkpoint.open("ab") as handle:
        handle.write(b"corruption")

    report = analyze(
        study_config_path=config_path,
        study_root=study_root,
        output_dir=tmp_path / "corrupt-analysis",
    )

    assert report["coverage"]["complete"] is False
    assert report["coverage"]["invalid_completed_bundle_count"] == 1
    completed = next(
        row
        for row in report["candidate_rows"]
        if row["row_kind"] == "candidate" and row["status"] == "complete"
    )
    assert completed["bundle_valid"] is False
    assert completed["exact_either_bound_percent"] is None
    assert "artifact hash mismatch" in completed["validation_errors_json"]
    assert report["surfaces"][0]["receipt_status"] == "valid"
