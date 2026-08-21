from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.reporting import sha256_file
from experiments.run_conv12_bounded_rho import build_source_config
from experiments.run_conv12_directed_exploratory_lr_search import surface_cells
import experiments.validate_conv12_directed_exploratory_lr_search as validator


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_identity() -> dict[str, Any]:
    archive = "a" * 64
    return {
        "commit": "b" * 40,
        "dirty": False,
        "source_kind": "frozen_archive",
        "source_archive_sha256": archive,
        "working_tree_sha256": archive,
    }


def _authority(
    contract: validator.Contract, root: Path, task: int
) -> validator.Authority:
    surface = contract.surfaces[task]
    return validator.Authority(
        root=root,
        surface=surface,
        resolved={"code": _source_identity()},
        target=validator._canonical_target(surface),
    )


def _zero_bias_record(architecture: str) -> dict[str, Any]:
    names = list(validator._bias_names(architecture))
    checkpoint = {
        "all_exact_zero": True,
        "nonzero_bias_element_count": 0,
        "expected_bias_names": names,
    }
    return {
        "all_exact_zero": True,
        "checkpoint_count": 2,
        "checkpoints": [dict(checkpoint), dict(checkpoint)],
    }


def test_default_contract_has_exact_sparse_surface_counts() -> None:
    contract = validator.load_contract()
    counts = tuple(
        len(surface_cells(contract.study, surface)) for surface in contract.surfaces
    )
    assert counts == validator.EXPECTED_CELL_COUNTS
    assert sum(counts) == 257
    # Surface 0 is sparse: compact local indices have holes and cannot be
    # reconstructed with a dense-grid divmod.
    cells = surface_cells(contract.study, contract.surfaces[0])
    assert len(cells) == 39
    assert max(int(cell["index"]) for cell in cells) == 40
    assert {int(cell["index"]) for cell in cells} != set(range(39))


def test_empty_study_root_is_an_honest_partial_snapshot(tmp_path: Path) -> None:
    report = validator.validate_collection(study_root=tmp_path)
    assert report["status"] == "partial"
    assert report["coverage"]["authority_count"] == 0
    assert report["coverage"]["missing_task_ids"] == list(range(12))
    assert report["coverage"]["terminal_cell_count"] == 0

    with pytest.raises(validator.IncompleteCollectionError) as captured:
        validator.validate_collection(study_root=tmp_path, require_complete=True)
    assert captured.value.report["status"] == "partial"


def test_running_cell_is_nonterminal_not_an_integrity_failure() -> None:
    assert (
        validator.classify_cell_outcome(
            "running", None, reporting_state="running", result_present=False
        )
        == "running"
    )
    with pytest.raises(ValueError, match="canonical running bundle"):
        validator.classify_cell_outcome(
            "running", None, reporting_state="complete", result_present=False
        )


def test_discovery_rejects_nested_non_disjoint_shard_roots(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    inner = outer / "inner"
    _write_json(outer / "study.resolved.json", {})
    _write_json(inner / "study.resolved.json", {})
    with pytest.raises(ValueError, match="not disjoint"):
        validator.discover_shard_roots(
            study_root=None, shard_roots=[outer, inner]
        )


def test_duplicate_task_authorities_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = validator.load_contract()
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        root.mkdir()

    monkeypatch.setattr(
        validator,
        "_validate_resolved",
        lambda root, _contract: _authority(contract, root, 0),
    )
    with pytest.raises(ValueError, match="Duplicate task/surface authorities"):
        validator._load_authorities(roots, contract)


def _receipt_fixture(
    tmp_path: Path,
) -> tuple[
    validator.Contract,
    validator.Authority,
    dict[str, Any],
    Path,
    dict[str, Any],
]:
    contract = validator.load_contract()
    root = tmp_path / "shards" / "main-task_0"
    authority = _authority(contract, root, 0)
    surface = authority.surface
    summary_path = root / "surfaces" / str(surface["surface_id"]) / "summary.json"
    summary = {
        "status": "complete",
        "best_observation": {"index": 7, "final_validation_accuracy": 0.9},
    }
    _write_json(summary_path, summary)
    authority_report = {
        "state": "complete",
        "terminal_cells": 39,
        "status_counts": {"complete": 39},
        "summary_path": str(summary_path),
    }
    receipt = {
        "schema_version": validator.RECEIPT_SCHEMA,
        "study_id": contract.study_id,
        "surface": surface,
        "task_id": 0,
        "target": "main",
        "environment_id": "main-rtx3090-py312",
        "source_commit": authority.resolved["code"]["commit"],
        "source_archive_sha256": authority.resolved["code"][
            "source_archive_sha256"
        ],
        "study_config_sha256": contract.sha256,
        "initializer_checkpoint_sha256": validator.EXPECTED_INITIALIZER_SHA256[
            "conv1"
        ],
        "summary_status": "complete",
        "expected_cells": 39,
        "terminal_cells": 39,
        "validated_cell_bundles": 39,
        "status_counts": {"complete": 39},
        "best_observation": summary["best_observation"],
        # Deliberately remote: validation must rebind this suffix locally.
        "summary_path": (
            "/remote/results/study/shards/main-task_0/surfaces/"
            f"{surface['surface_id']}/summary.json"
        ),
        "summary_sha256": sha256_file(summary_path),
        "official_test_read": False,
        "semantic_pass": True,
    }
    receipt_path = tmp_path / "transport_receipts" / "main-task_0.json"
    _write_json(receipt_path, receipt)
    return contract, authority, authority_report, receipt_path, receipt


def test_receipt_rebinds_remote_summary_to_local_bytes(tmp_path: Path) -> None:
    contract, authority, report, receipt_path, receipt = _receipt_fixture(tmp_path)
    validated = validator._validate_receipt(
        authority, contract, report, (receipt_path, receipt)
    )
    assert validated["task_id"] == 0
    assert validated["validated_cell_bundles"] == 39
    assert validated["summary_sha256"] == receipt["summary_sha256"]


def test_receipt_rejects_tampered_local_summary(tmp_path: Path) -> None:
    contract, authority, report, receipt_path, receipt = _receipt_fixture(tmp_path)
    summary_path = Path(report["summary_path"])
    _write_json(summary_path, {"status": "complete", "best_observation": None})
    with pytest.raises(ValueError, match="locally collected summary"):
        validator._validate_receipt(
            authority, contract, report, (receipt_path, receipt)
        )


def test_unresolved_probe_receipt_is_valid_but_not_complete(tmp_path: Path) -> None:
    contract, authority, report, receipt_path, receipt = _receipt_fixture(tmp_path)
    summary_path = Path(report["summary_path"])
    summary = {"status": "unresolved_probe", "best_observation": None}
    _write_json(summary_path, summary)
    report.update(state="unresolved_probe", terminal_cells=0, status_counts={})
    receipt.update(
        summary_status="unresolved_probe",
        terminal_cells=0,
        validated_cell_bundles=0,
        status_counts={},
        best_observation=None,
        summary_sha256=sha256_file(summary_path),
    )
    _write_json(receipt_path, receipt)
    validated = validator._validate_receipt(
        authority, contract, report, (receipt_path, receipt)
    )
    assert validated["validated_cell_bundles"] == 0


def test_npz_bias_validation_checks_best_and_final(tmp_path: Path) -> None:
    run_dir = tmp_path / "cell"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    for name in ("weights_best.npz", "weights_final.npz"):
        np.savez(
            checkpoints / name,
            ConvWeight_0=np.asarray([1e-5], dtype=np.float32),
            DenseWeight_0=np.asarray([1e-4], dtype=np.float32),
            Bias_0=np.zeros((2,), dtype=np.float32),
        )
    result = {
        "artifacts": [
            {"kind": "checkpoint", "path": f"checkpoints/{name}"}
            for name in ("weights_best.npz", "weights_final.npz")
        ]
    }
    validator._validate_npz_biases(run_dir, result, "conv1")
    np.savez(
        checkpoints / "weights_final.npz",
        ConvWeight_0=np.asarray([1e-5], dtype=np.float32),
        DenseWeight_0=np.asarray([1e-4], dtype=np.float32),
        Bias_0=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="nonzero/missing Bias_0"):
        validator._validate_npz_biases(run_dir, result, "conv1")


def test_require_complete_accepts_exact_twelve_and_257_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = validator.load_contract()
    authorities = [
        _authority(contract, tmp_path / f"task_{task}", task) for task in range(12)
    ]
    reports = []
    receipts: dict[int, tuple[Path, dict[str, Any]]] = {}
    for task, (authority, count) in enumerate(
        zip(authorities, validator.EXPECTED_CELL_COUNTS)
    ):
        reports.append(
            {
                "task_id": task,
                "surface_id": authority.surface["surface_id"],
                "state": "complete",
                "expected_cells": count,
                "terminal_cells": count,
                "numeric_cells": count,
                "nonfinite_cells": 0,
                "status_counts": {"complete": count},
            }
        )
        receipts[task] = (tmp_path / f"receipt_{task}.json", {})

    monkeypatch.setattr(validator, "load_contract", lambda _path: contract)
    monkeypatch.setattr(validator, "discover_shard_roots", lambda **_kwargs: [])
    monkeypatch.setattr(validator, "_load_authorities", lambda _roots, _contract: authorities)
    monkeypatch.setattr(
        validator,
        "_validate_authority",
        lambda authority, _contract: dict(reports[int(authority.surface["index"])]),
    )
    monkeypatch.setattr(validator, "_receipt_inventory", lambda *args, **kwargs: receipts)
    monkeypatch.setattr(
        validator,
        "_validate_receipt",
        lambda authority, _contract, report, record: {
            "task_id": int(authority.surface["index"]),
            "validated_cell_bundles": int(report["terminal_cells"]),
        },
    )
    report = validator.validate_collection(
        study_root=tmp_path, require_complete=True
    )
    assert report["complete"] is True
    assert report["coverage"]["authority_count"] == 12
    assert report["coverage"]["validated_receipt_count"] == 12
    assert report["coverage"]["terminal_cell_count"] == 257


def test_numeric_cell_validation_calls_canonical_validator_and_checks_zero_bias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = validator.load_contract()
    surface = contract.surfaces[0]
    authority = _authority(contract, tmp_path / "main-task_0", 0)
    expected = surface_cells(contract.study, surface)[0]
    architecture = "conv1"
    names = validator._weight_names(architecture) + validator._bias_names(architecture)
    init_path = authority.root / "assets/bounded_uniform/conv1/final_model.pt"
    source = build_source_config(
        contract.parent,
        initializer="bounded_uniform",
        architecture=architecture,
        scheme="baseline",
        optimizer="SGD",
        init_checkpoint_path=init_path,
        dataset_root=tmp_path / "mnist",
    )
    source_sha = "c" * 64
    units = {name: 1.0 for name in validator._weight_names(architecture)}
    probe = {
        "parameter_names": list(names),
        "normalization_unit_by_weight": units,
    }
    probe_semantic_sha = "d" * 64
    probe_file_sha = "e" * 64
    rates = {
        "ConvWeight_0": float(expected["rho_conv"]),
        "DenseWeight_0": float(expected["rho_dense"]),
        "Bias_0": 0.0,
    }
    vector = [rates[name] for name in names]
    candidate = validator.prepare_training_config(
        source,
        "SGD",
        vector,
        epochs=3,
        max_batches=None,
        max_validation_batches=None,
        split_seed=0,
        shuffle_seed=0,
        validation_batch_size=64,
    )
    run_dir = authority.root / "cell"
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "source_config.json", candidate)
    _write_json(run_dir / "config.used.json", candidate)
    candidate_sha = sha256_file(run_dir / "source_config.json")
    zero = _zero_bias_record(architecture)
    cell = {
        "index": expected["index"],
        "optimizer": "SGD",
        "evidence_class": "ordinary_mnist_exploratory",
        "bias_policy": "zero",
        "rho_conv": expected["rho_conv"],
        "rho_dense": expected["rho_dense"],
        "status": "complete",
        "completed_steps": 10314,
        "selection_eligible": True,
        "learning_rates_by_parameter": rates,
        "learning_rate_vector": vector,
        "zero_bias_checkpoint_verification": zero,
        "signature": {
            "source_config_sha256": source_sha,
            "probe_sha256": probe_semantic_sha,
            "code": authority.resolved["code"],
            "epochs": 3,
            "expected_candidate_steps": 10314,
            "skip_canary": True,
            "evidence_class": "ordinary_mnist_exploratory",
            "split_seed": 0,
            "shuffle_seed": 0,
            "safety": {"rejections_enabled": False},
        },
    }
    _write_json(run_dir / "cell.json", cell)
    runtime = {"target": "main"}
    manifest = {
        "study_id": contract.study_id,
        "run_id": run_dir.name,
        "evidence_class": "ordinary_mnist_exploratory",
        "smoke": False,
        "dataset": {"evaluation_split": "validation", "official_test_read": False},
        "runtime": runtime,
        "command": {"surface_index": 0, "cell_index": expected["index"]},
        "configuration": {
            "resolved": candidate,
            "sha256": candidate_sha,
            "optimizer": "SGD",
        },
        "inputs": [
            {"role": "source_config", "sha256": source_sha},
            {"role": "optimizer_probe", "sha256": probe_file_sha},
        ],
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "status.json", {"state": "complete", "runtime": runtime})
    _write_json(
        run_dir / "metrics.json",
        {
            "official_test_accuracy": None,
            "official_test_loss": None,
            "official_test_examples": 0,
            "official_test_evaluations": 0,
        },
    )
    _write_json(
        run_dir / "safety_diagnostics.json",
        {
            "processed_steps": 10314,
            "report_only_diagnostics": {"used_for_rejection": False},
            "terminal_gates": {"scientific_rejections_enabled": False},
            "zero_bias_checkpoint_verification": zero,
        },
    )
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()
    for name in ("weights_best.npz", "weights_final.npz"):
        np.savez(
            checkpoints / name,
            ConvWeight_0=np.asarray([1e-5], dtype=np.float32),
            DenseWeight_0=np.asarray([1e-4], dtype=np.float32),
            Bias_0=np.zeros((1,), dtype=np.float32),
        )
    result = {
        "dataset": {"official_test_read": False},
        "completion": {
            "official_test_read": False,
            "criteria_met": True,
            "rho_cell_complete": True,
            "zero_bias_checkpoint_verification": zero,
        },
        "artifacts": [
            {"kind": "checkpoint", "path": f"checkpoints/{name}"}
            for name in ("weights_best.npz", "weights_final.npz")
        ],
    }
    _write_json(run_dir / "result.json", result)
    calls: list[Path] = []
    monkeypatch.setattr(
        validator, "validate_run", lambda path: calls.append(Path(path)) or []
    )
    row = validator._validate_cell(
        authority,
        contract,
        run_dir=run_dir,
        expected=expected,
        source_config=source,
        source_sha256=source_sha,
        probe=probe,
        probe_semantic_sha256=probe_semantic_sha,
        probe_file_sha256=probe_file_sha,
    )
    assert calls == [run_dir]
    assert row["canonical_outcome"] == "numeric_complete"
