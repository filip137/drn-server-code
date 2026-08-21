from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import analyze_conv12_directed_exploratory_lr_search as analysis


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_identity() -> dict[str, object]:
    archive = "a" * 64
    return {
        "commit": "b" * 40,
        "dirty": False,
        "source_kind": "frozen_archive",
        "source_archive_sha256": archive,
        "working_tree_sha256": archive,
    }


def test_exact_contract_preserves_sparse_coordinate_indices() -> None:
    contract = analysis.load_contract()
    surface = contract.surfaces[0]
    cells = analysis.surface_cells(contract.payload, surface)
    by_global = {
        (cell["rho_conv_ladder_index"], cell["rho_dense_ladder_index"]): cell
        for cell in cells
    }

    assert len(cells) == 39
    assert len({cell["index"] for cell in cells}) == 39
    assert by_global[(5, 4)]["index"] == 28
    assert by_global[(6, 0)]["index"] == 30
    assert (5, 5) not in by_global
    assert {29, 35, 41}.isdisjoint({cell["index"] for cell in cells})


@pytest.mark.parametrize("task", [3, 9])
def test_declared_jean_zay_qos_targets_map_to_runtime_jean_zay(
    tmp_path: Path, task: int
) -> None:
    contract = analysis.load_contract()
    surface = contract.surfaces[task]
    root = tmp_path / "shards" / f"jean-zay-task_{task}"
    (root / "surfaces" / str(surface["surface_id"])).mkdir(parents=True)
    resolved = {
        "schema_version": analysis.RESOLVED_SCHEMA,
        "study_id": contract.study_id,
        "evidence_class": "ordinary_mnist_exploratory",
        "study_config": f"/remote/configs/conv/{contract.path.name}",
        "study_config_sha256": contract.sha256,
        "parent_study_config": f"/remote/configs/conv/{contract.parent_path.name}",
        "parent_study_config_sha256": analysis.sha256_file(contract.parent_path),
        "initializer_checkpoint_sha256_by_architecture": analysis.EXPECTED_INITIALIZER_SHA256,
        "code": _source_identity(),
        "target": "jean-zay",
        "device": "cuda",
        "surface_count": 12,
        "total_cells": 257,
        "surfaces": list(contract.surfaces),
        "official_test_read": False,
        "restart_interrupted_cells": True,
    }
    _write_json(root / "study.resolved.json", resolved)

    shard = analysis._validate_resolved(root, contract)

    assert surface["target"] in {"jean-zay-dev", "jean-zay-t3"}
    assert shard.target == "jean-zay"


def _jean_zay_receipt_fixture(
    tmp_path: Path,
) -> tuple[
    analysis.StudyContract,
    analysis.Shard,
    dict[str, object],
    Path,
    dict[str, object],
]:
    contract = analysis.load_contract()
    surface = contract.surfaces[3]
    root = tmp_path / "shards" / "jean-zay-task_3"
    code = _source_identity()
    shard = analysis.Shard(
        root=root,
        surface=surface,
        resolved={"code": code},
        target="jean-zay",
    )
    summary_path = root / "surfaces" / str(surface["surface_id"]) / "summary.json"
    summary: dict[str, object] = {
        "status": "complete",
        "terminal_cells": 16,
        "status_counts": {"complete": 16},
        "best_observation": {"index": 4, "final_validation_accuracy": 0.91},
    }
    _write_json(summary_path, summary)
    receipt: dict[str, object] = {
        "schema_version": analysis.RECEIPT_SCHEMA,
        "study_id": contract.study_id,
        "surface": surface,
        "task_id": 3,
        "target": "jean-zay",
        "environment_id": "jean-zay:fmu-v100:py312",
        "source_commit": code["commit"],
        "source_archive_sha256": code["source_archive_sha256"],
        "study_config_sha256": contract.sha256,
        "initializer_checkpoint_sha256": analysis.EXPECTED_INITIALIZER_SHA256["conv1"],
        "summary_status": "complete",
        "expected_cells": 16,
        "terminal_cells": 16,
        "validated_cell_bundles": 16,
        "status_counts": {"complete": 16},
        "best_observation": summary["best_observation"],
        "summary_path": (
            "/lustre/results/study/shards/jean-zay-task_3/surfaces/"
            f"{surface['surface_id']}/summary.json"
        ),
        "summary_sha256": analysis.sha256_file(summary_path),
        "official_test_read": False,
        "semantic_pass": True,
    }
    receipt_path = tmp_path / "transport_receipts" / "jean-zay-task_3.json"
    _write_json(receipt_path, receipt)
    return contract, shard, summary, receipt_path, receipt


@pytest.mark.parametrize(
    ("tamper", "error_match"),
    [
        ("filename", "noncanonical filename"),
        ("summary_path", "does not bind the local summary"),
        ("task_id", "invalid task_id"),
        ("surface", "invalid surface"),
        ("summary_sha256", "does not bind the local summary"),
    ],
)
def test_jean_zay_receipt_tampering_fails_closed(
    tmp_path: Path, tamper: str, error_match: str
) -> None:
    contract, shard, summary, receipt_path, receipt = _jean_zay_receipt_fixture(tmp_path)
    valid = analysis._validate_receipt(
        shard, contract, summary=summary, record=(receipt_path, receipt)
    )
    assert valid["validated_cell_bundles"] == 16

    if tamper == "filename":
        receipt_path = receipt_path.with_name("receipt.json")
    elif tamper == "summary_path":
        receipt["summary_path"] = str(receipt["summary_path"]).replace(
            "jean-zay-task_3", "jean-zay-task_4"
        )
    elif tamper == "task_id":
        receipt["task_id"] = 4
    elif tamper == "surface":
        receipt["surface"] = contract.surfaces[4]
    elif tamper == "summary_sha256":
        receipt["summary_sha256"] = "0" * 64
    else:  # pragma: no cover
        raise AssertionError(tamper)

    with pytest.raises(ValueError, match=error_match):
        analysis._validate_receipt(
            shard, contract, summary=summary, record=(receipt_path, receipt)
        )


def test_production_asset_without_redundant_expected_sha_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = analysis.load_contract()
    surface = contract.surfaces[0]
    root = tmp_path / "shards" / "main-task_0"
    checkpoint = root / "assets" / "bounded_uniform" / "conv1" / "final_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"production initializer bytes")
    checkpoint_sha = analysis.sha256_file(checkpoint)
    monkeypatch.setattr(
        analysis,
        "EXPECTED_INITIALIZER_SHA256",
        {"conv1": checkpoint_sha, "conv2": analysis.EXPECTED_INITIALIZER_SHA256["conv2"]},
    )
    asset = {
        "architecture": "conv1",
        "initializer": "bounded_uniform",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        # Production schema deliberately has no expected_checkpoint_sha256.
        "official_test_read": False,
        "zero_bias_checkpoint_verification": {
            "all_exact_zero": True,
            "nonzero_bias_element_count": 0,
            "expected_bias_names": ["Bias_0"],
        },
    }
    _write_json(checkpoint.parent / "asset.json", asset)
    shard = analysis.Shard(
        root=root,
        surface=surface,
        resolved={"code": _source_identity()},
        target="main",
    )

    analysis._verify_initializer(shard)


def test_endpoint_occupancy_casts_bounds_to_checkpoint_dtype(tmp_path: Path) -> None:
    lower = np.float32(1e-5)
    upper = np.float32(1e-4)
    middle = np.float32(5e-5)
    checkpoint = tmp_path / "weights.npz"
    np.savez(
        checkpoint,
        ConvWeight_0=np.asarray([lower, middle, upper], dtype=np.float32),
        DenseWeight_0=np.asarray([lower, middle], dtype=np.float32),
        Bias_0=np.zeros(2, dtype=np.float32),
    )

    result = analysis._checkpoint_endpoint_diagnostics(
        checkpoint,
        architecture="conv1",
        reported_final_occupancy={"ConvWeight_0": 2 / 3, "DenseWeight_0": 1 / 2},
    )

    assert result["all"]["lower_count"] == 2
    assert result["all"]["upper_count"] == 1
    assert result["all"]["either_count"] == 3
    assert result["all"]["weight_count"] == 5
    assert result["all"]["either_fraction"] == pytest.approx(3 / 5)
    assert result["conv"]["either_count"] == 2
    assert result["dense"]["either_count"] == 1
    assert result["by_parameter"]["ConvWeight_0"]["dtype_lower_bound"] == float(lower)
    assert result["biases_exact_zero"] is True


def test_endpoint_occupancy_rejects_value_below_dtype_cast_bound(tmp_path: Path) -> None:
    lower = np.float32(1e-5)
    checkpoint = tmp_path / "outside.npz"
    np.savez(
        checkpoint,
        ConvWeight_0=np.asarray(
            [np.nextafter(lower, np.float32(-np.inf), dtype=np.float32)],
            dtype=np.float32,
        ),
        DenseWeight_0=np.asarray([np.float32(5e-5)], dtype=np.float32),
        Bias_0=np.zeros(1, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="leaves the dtype-cast interval"):
        analysis._checkpoint_endpoint_diagnostics(checkpoint, architecture="conv1")


def test_correlation_is_paired_tie_aware_and_endpoint_specific() -> None:
    stats = analysis.correlation_statistics([1.0, 2.0, 2.0, 4.0], [10.0, 20.0, 20.0, 40.0])
    assert stats == {"sample_count": 4, "pearson": 1.0, "spearman": 1.0}
    assert analysis.correlation_statistics([1.0, 1.0], [2.0, 3.0]) == {
        "sample_count": 2,
        "pearson": None,
        "spearman": None,
    }

    surface = {
        "surface_index": 0,
        "surface_id": "surface",
        "architecture": "conv1",
        "scheme": "baseline",
        "optimizer": "SGD",
    }
    rows = []
    for index in range(3):
        rows.append(
            {
                "canonical_outcome": "numeric_complete",
                "final_validation_accuracy": 0.7 + index / 10,
                "best_validation_accuracy": 0.9 - index / 10,
                "final_either_percent": 10.0 + index,
                "final_conv_either_percent": 20.0 + index,
                "final_dense_either_percent": 30.0 + index,
                "best_either_percent": 40.0 + index,
                "best_conv_either_percent": 50.0 + index,
                "best_dense_either_percent": 60.0 + index,
                "final_by_parameter": {
                    "ConvWeight_0": {"either_percent": 1.0 + index},
                    "DenseWeight_0": {"either_percent": 2.0 + index},
                },
                "best_by_parameter": {
                    "ConvWeight_0": {"either_percent": 3.0 + index},
                    "DenseWeight_0": {"either_percent": 4.0 + index},
                },
            }
        )
    correlations = analysis._correlation_rows(surface, rows)
    final_all = next(row for row in correlations if row["endpoint"] == "final" and row["scope"] == "all")
    best_all = next(row for row in correlations if row["endpoint"] == "best" and row["scope"] == "all")
    assert final_all["accuracy_field"] == "final_validation_accuracy"
    assert final_all["pearson"] == pytest.approx(1.0)
    assert best_all["accuracy_field"] == "best_validation_accuracy"
    assert best_all["pearson"] == pytest.approx(-1.0)


def test_global_ladder_neighbor_logic_distinguishes_bracket_open_unresolved() -> None:
    contract = analysis.load_contract()

    def row(conv: int, dense: int, accuracy: float, index: int) -> dict[str, object]:
        return {
            "rho_conv_ladder_index": conv,
            "rho_dense_ladder_index": dense,
            "final_validation_accuracy": accuracy,
            "canonical_outcome": "numeric_complete",
            "cell_index": index,
            "rho_conv": contract.payload["rho_search"]["rho_conv_ladder"][conv],
            "rho_dense": contract.payload["rho_search"]["rho_dense_ladder"][dense],
        }

    best = row(4, 4, 0.9, 0)
    rows = [best, row(3, 4, 0.8, 1), row(5, 4, 0.8, 2), row(4, 3, 0.8, 3), row(4, 5, 0.8, 4)]
    assert analysis._range_evidence(best, rows, contract)["range_status"] == "bracketed"

    open_rows = [*rows[:-1], row(4, 5, 0.9, 4)]
    open_result = analysis._range_evidence(best, open_rows, contract)
    assert open_result["range_status"] == "open"
    assert open_result["improving_or_tied_directions"] == ["upper_rho_dense"]

    unresolved = analysis._range_evidence(best, rows[:-1], contract)
    assert unresolved["range_status"] == "unresolved"
    assert unresolved["unresolved_directions"] == ["upper_rho_dense"]


def test_require_complete_writes_partial_snapshot_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = analysis.load_contract()
    shard_root = tmp_path / "study" / "shards" / "main-task_0"
    shard_root.mkdir(parents=True)
    (shard_root / "study.resolved.json").write_text("{}\n", encoding="utf-8")
    shard = analysis.Shard(
        root=shard_root,
        surface=contract.surfaces[0],
        resolved={},
        target="main",
    )
    snapshot = {
        "surface_index": 0,
        "surface_id": contract.surfaces[0]["surface_id"],
        "architecture": "conv1",
        "scheme": "baseline",
        "optimizer": "SGD",
        "declared_target": "main",
        "observed_target": "main",
        "surface_state": "running",
        "receipt_state": "pending",
        "expected_cell_count": 39,
        "observed_terminal_cell_count": 0,
        "numeric_cell_count": 0,
        "nonfinite_cell_count": 0,
        "missing_cell_count": 39,
        "shard_root": str(shard_root),
        "resolved_sha256": analysis.sha256_file(shard_root / "study.resolved.json"),
        "summary_sha256": "0" * 64,
    }
    monkeypatch.setattr(analysis, "load_contract", lambda _path: contract)
    monkeypatch.setattr(analysis, "discover_shard_roots", lambda **_kwargs: [shard_root])
    monkeypatch.setattr(analysis, "_load_shards", lambda _roots, _contract: [shard])
    monkeypatch.setattr(
        analysis,
        "_receipt_inventory",
        lambda _contract, **_kwargs: {index: [] for index in range(12)},
    )
    monkeypatch.setattr(
        analysis,
        "_read_surface",
        lambda *_args, **_kwargs: (snapshot, [], None),
    )
    monkeypatch.setattr(
        analysis,
        "_plot_accuracy_vs_clipping",
        lambda _rows, path: (path.parent.mkdir(parents=True, exist_ok=True), path.write_bytes(b"plot")),
    )
    output = tmp_path / "analysis"

    with pytest.raises(analysis.IncompleteCoverageError, match="partial snapshot was written") as caught:
        analysis.analyze(output_dir=output, require_complete=True)

    assert caught.value.report is not None
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["coverage"]["status"] == "partial"
    assert summary["coverage"]["discovered_surface_count"] == 1
    assert summary["coverage"]["missing_cell_count"] == 257
    assert (output / "cells.csv").is_file()
    assert (output / "surfaces.csv").is_file()
    assert (output / "correlations.csv").is_file()
    assert (output / "report.md").is_file()
    assert (output / "plots" / "accuracy_vs_clipping.png").is_file()
