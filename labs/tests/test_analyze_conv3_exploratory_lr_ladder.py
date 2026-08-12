from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import experiments.analyze_conv3_exploratory_lr_ladder as analyzer


LOCAL_TARGETS = ("akib", "trex", "akib", "trex")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _zero_bias_run_record() -> dict:
    return {
        "all_exact_zero": True,
        "checkpoint_count": 1,
        "checkpoints": [
            {
                "all_exact_zero": True,
                "nonzero_bias_element_count": 0,
                "expected_bias_names": list(analyzer.BIAS_NAMES),
            }
        ],
    }


def _scheduler_provenance_payloads(study_id: str) -> dict[str, dict]:
    common = {"study_id": study_id}
    return {
        "launch_plan.json": {
            **common,
            "schema_version": (
                "perfectdiode-conv3-exploratory-lr-ladder-launch-plan/v1"
            ),
        },
        "jean_zay_submission.json": {
            **common,
            "schema_version": (
                "perfectdiode-conv3-exploratory-lr-ladder-submission/v1"
            ),
            "job_id": "844032",
        },
        "replacement_plan.json": {
            **common,
            "schema_version": (
                "perfectdiode-conv3-exploratory-lr-ladder-replacement-plan/v1"
            ),
            "reason": {
                "superseded_job_id": "844032",
                "superseded_job_state": "CANCELLED before allocation",
                "superseded_job_elapsed": "00:00:00",
                "scientific_artifacts_produced": 0,
            },
        },
        "recovery_submission.json": {
            **common,
            "schema_version": (
                "perfectdiode-conv3-exploratory-lr-ladder-recovery-submission/v1"
            ),
            "supersedes": {
                "job_id": "844032",
                "state": "CANCELLED before allocation",
                "elapsed": "00:00:00",
                "artifacts": 0,
                "reason": "invalid pre-allocation source",
            },
            "replacement": {"job_id": "844396"},
        },
        "local_fallback_plan.json": {
            **common,
            "schema_version": (
                "perfectdiode-conv3-exploratory-lr-ladder-local-fallback-plan/v1"
            ),
            "reason": {
                "jean_zay_job_id": "844396",
                "state": (
                    "four array elements pending with QOSGrpCpuLimit and no "
                    "advertised start"
                ),
                "scientific_artifacts_produced": 0,
            },
        },
        "local_fallback_submission.json": {
            **common,
            "schema_version": (
                "perfectdiode-conv3-exploratory-lr-ladder-local-fallback-submission/v1"
            ),
            "superseded_jean_zay_job": {
                "job_id": "844396",
                "state_before_cancellation": (
                    "four PENDING array elements; QOSGrpCpuLimit; no "
                    "advertised start"
                ),
                "terminal_state": "CANCELLED by owner",
                "elapsed": "00:00:00",
                "scientific_artifacts": 0,
                "reason": "replaced by target-gated local workers",
            },
        },
        "jean_zay_terminal_accounting.json": {
            **common,
            "schema_version": (
                "perfectdiode-conv3-exploratory-lr-ladder-jean-zay-accounting/v1"
            ),
            "jobs": [
                {
                    "job_id": "844032",
                    "job_name": "pd-c3-lr-ladder",
                    "state": "CANCELLED by 306943",
                    "exit_code": "0:0",
                    "elapsed": "00:00:00",
                    "start": None,
                    "end": "2026-08-11T17:57:17",
                },
                {
                    "job_id": "844396",
                    "job_name": "pd-c3-lr2",
                    "state": "CANCELLED by 306943",
                    "exit_code": "0:0",
                    "elapsed": "00:00:00",
                    "start": None,
                    "end": "2026-08-11T21:45:08",
                },
            ],
        },
    }


def _write_scheduler_provenance(root: Path, study_id: str) -> dict[str, Path]:
    provenance = root / "provenance"
    paths: dict[str, Path] = {}
    for name, payload in _scheduler_provenance_payloads(study_id).items():
        path = provenance / name
        _write_json(path, payload)
        paths[name] = path
    return paths


def _study(tmp_path: Path, initializer_sha256: str) -> Path:
    conv, dense = analyzer._expected_axes()
    path = tmp_path / "study.json"
    _write_json(
        path,
        {
            "schema_version": analyzer.STUDY_SCHEMA,
            "study_id": "synthetic-conv3-exploratory-ladder",
            "evidence_class": "ordinary_mnist_exploratory",
            "parent": {
                "study_config": (
                    "configs/conv/"
                    "perfectdiode_bounded_uniform_zero_bias_lr_search_conv3_"
                    "occupancy_report_only_seed0_20260811_v1.json"
                ),
                "study_id": (
                    "perfectdiode-bounded-uniform-zero-bias-lr-search-conv3-"
                    "occupancy-report-only-seed0-20260811-v1"
                ),
                "study_config_sha256": (
                    "b845340aec78237dd0f6940a6dfcec76a4463bee0fe7164ae6ea51f5bc68d647"
                ),
                "initializer_checkpoint_sha256": initializer_sha256,
            },
            "scope": {
                "architectures": ["conv3"],
                "schemes": ["baseline", "ours"],
                "optimizers": ["SGD", "Adam"],
                "initializers": ["bounded_uniform"],
                "excluded": ["legacy", "conv1", "conv2"],
            },
            "rho_search": {
                "rho_conv": list(conv),
                "rho_dense": list(dense),
                "bias_policy": "zero",
                "candidate_epochs": 3,
                "expected_candidate_steps": 10314,
                "minimum_validation_accuracy": 0.0,
                "skip_canary": True,
                "canary_steps": 0,
                "disable_safety_rejections": True,
                "post_candidate_tk": False,
            },
        },
    )
    return path


def _runtime(target: str) -> dict:
    return {
        "target": target,
        "host": f"synthetic-{target}",
        "python": "3.12.synthetic",
    }


def _manifest(
    study_id: str,
    run_id: str,
    *,
    surface: dict,
    target: str,
    index: int,
    rho_conv: float,
    rho_dense: float,
    learning_rates: list[float],
    candidate: dict,
    candidate_sha256: str,
    source_sha256: str,
    probe_file_sha256: str,
    code: dict,
) -> dict:
    return {
        "schema_version": "experiment-run-manifest/v1",
        "study_id": study_id,
        "run_id": run_id,
        "arm_id": run_id,
        "evidence_class": "ordinary_mnist_exploratory",
        "dataset": {
            "key": "mnist",
            "variant": "ordinary",
            "evaluation_split": "validation",
            "official_test_read": False,
        },
        "smoke": False,
        "git": code,
        "runtime": _runtime(target),
        "command": {
            "surface_index": surface["index"],
            "cell_index": index,
        },
        "configuration": {
            "path": "source_config.json",
            "sha256": candidate_sha256,
            "resolved": candidate,
            "optimizer": surface["optimizer"],
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "learning_rates": learning_rates,
        },
        "inputs": [
            {
                "role": "source_config",
                "path": "../../source_config.json",
                "sha256": source_sha256,
            },
            {
                "role": "optimizer_probe",
                "path": "../../probe.json",
                "sha256": probe_file_sha256,
            },
        ],
    }


def _status(study_id: str, run_id: str, state: str, *, target: str) -> dict:
    return {
        "schema_version": "experiment-run-status/v1",
        "study_id": study_id,
        "run_id": run_id,
        "arm_id": run_id,
        "state": state,
        "runtime": _runtime(target),
    }


def _signature(
    *,
    optimizer: str,
    rho_conv: float,
    rho_dense: float,
    source_sha256: str,
    probe_sha256: str,
    code: dict,
) -> dict:
    return {
        "optimizer": optimizer,
        "evidence_class": "ordinary_mnist_exploratory",
        "bias_policy": "zero",
        "epochs": 3,
        "expected_candidate_steps": 10314,
        "max_batches": None,
        "max_validation_batches": None,
        "minimum_validation_accuracy": 0.0,
        "skip_canary": True,
        "restart_interrupted_cells": True,
        "canary_steps": 0,
        "source_config_sha256": source_sha256,
        "probe_sha256": probe_sha256,
        "code": code,
        "split_seed": 0,
        "shuffle_seed": 0,
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "safety": {
            "rejections_enabled": False,
            "bound_occupancy": "report_only",
            "projection_efficiency": "report_only",
            "bound_occupancy_increase_maximum": None,
            "projection_efficiency_minimum": None,
        },
    }


def _diagnostics(
    *,
    occupancy: dict[str, float],
    projection: float,
    processed_steps: int,
    failure: dict | None,
    complete: bool,
) -> dict:
    payload = {
        "schema_version": "conv-rho-training-safety/v1",
        "processed_steps": processed_steps,
        "terminal_gates": {
            "scientific_rejections_enabled": False,
            "bound_occupancy_increase_maximum": None,
            "projection_efficiency_minimum": None,
        },
        "report_only_diagnostics": {
            "bound_occupancy": True,
            "projection_efficiency": True,
            "used_for_rejection": False,
        },
        "final_bound_occupancy_by_parameter": occupancy,
        "median_projection_efficiency_by_parameter": {
            name: projection for name in occupancy
        },
        "median_projection_efficiency": projection if occupancy else None,
        "median_proposed_update_rms_by_parameter": {
            name: 1e-8 for name in occupancy
        },
        "median_applied_update_rms_by_parameter": {
            name: projection * 1e-8 for name in occupancy
        },
        "safety_failure": failure,
    }
    if complete:
        payload["zero_bias_checkpoint_verification"] = _zero_bias_run_record()
    return payload


def _make_numeric_cell(
    cell_dir: Path,
    *,
    study_id: str,
    surface: dict,
    index: int,
    rho_conv: float,
    rho_dense: float,
    source_sha256: str,
    probe_sha256: str,
    probe_file_sha256: str,
    code: dict,
    target: str,
    source_config: dict,
) -> dict:
    cell_dir.mkdir(parents=True)
    learning_rates = {
        "ConvWeight_0": rho_conv,
        "ConvWeight_1": rho_conv / 2,
        "ConvWeight_2": rho_conv / 3,
        "DenseWeight_0": rho_dense,
        "Bias_0": 0.0,
        "Bias_1": 0.0,
        "Bias_2": 0.0,
    }
    at_bound = index % 4 == 0
    bounded = np.asarray([1e-5 if at_bound else 5e-5, 5e-5], dtype=np.float32)
    checkpoint = cell_dir / "checkpoints" / "weights_final.npz"
    checkpoint.parent.mkdir()
    np.savez(
        checkpoint,
        ConvWeight_0=bounded,
        ConvWeight_1=bounded,
        ConvWeight_2=bounded,
        DenseWeight_0=bounded,
        Bias_0=np.zeros(1, dtype=np.float32),
        Bias_1=np.zeros(1, dtype=np.float32),
        Bias_2=np.zeros(1, dtype=np.float32),
    )
    occupancy = {name: 0.5 if at_bound else 0.0 for name in analyzer.WEIGHT_NAMES}
    projection = 1.0 - index / 1000.0
    accuracy = 0.2 + 0.1 * surface["index"] + index / 1000.0
    loss = 1.0 - accuracy
    learning_rate_vector = [
        learning_rates[name]
        for name in analyzer.WEIGHT_NAMES + analyzer.BIAS_NAMES
    ]
    cell = {
        "index": index,
        "optimizer": surface["optimizer"],
        "bias_policy": "zero",
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "status": "complete",
        "selection_eligible": True,
        "completed_steps": 10314,
        "learning_rates_by_parameter": learning_rates,
        "learning_rate_vector": learning_rate_vector,
        "signature": _signature(
            optimizer=surface["optimizer"],
            rho_conv=rho_conv,
            rho_dense=rho_dense,
            source_sha256=source_sha256,
            probe_sha256=probe_sha256,
            code=code,
        ),
        "evidence_class": "ordinary_mnist_exploratory",
        "zero_bias_checkpoint_verification": _zero_bias_run_record(),
    }
    _write_json(cell_dir / "cell.json", cell)
    candidate = json.loads(json.dumps(source_config))
    candidate["lr"] = learning_rate_vector
    candidate["optimizer"] = {
        **candidate["optimizer"],
        "name": surface["optimizer"],
        "learning_rate": learning_rate_vector,
    }
    _write_json(cell_dir / "source_config.json", candidate)
    _write_json(cell_dir / "config.used.json", candidate)
    metrics = {
        "final_validation_accuracy": accuracy,
        "final_validation_loss": loss,
        "final_test_accuracy": accuracy,
        "final_test_loss": loss,
        "official_test_accuracy": None,
        "official_test_loss": None,
        "official_test_examples": 0,
        "official_test_evaluations": 0,
    }
    _write_json(cell_dir / "metrics.json", metrics)
    _write_json(
        cell_dir / "safety_diagnostics.json",
        _diagnostics(
            occupancy=occupancy,
            projection=projection,
            processed_steps=10314,
            failure=None,
            complete=True,
        ),
    )
    manifest = _manifest(
        study_id,
        cell_dir.name,
        surface=surface,
        target=target,
        index=index,
        rho_conv=rho_conv,
        rho_dense=rho_dense,
        learning_rates=learning_rate_vector,
        candidate=candidate,
        candidate_sha256=_sha256(cell_dir / "source_config.json"),
        source_sha256=source_sha256,
        probe_file_sha256=probe_file_sha256,
        code=code,
    )
    _write_json(cell_dir / "manifest.json", manifest)
    (cell_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    result = {
        "schema_version": "experiment-run-result/v1",
        "study_id": study_id,
        "run_id": cell_dir.name,
        "arm_id": cell_dir.name,
        "evidence_class": "ordinary_mnist_exploratory",
        "dataset": manifest["dataset"],
        "smoke": False,
        "terminal_metrics": {
            "validation": {
                "final_accuracy": accuracy,
                "final_loss": loss,
            }
        },
        "completion": {
            "criteria_met": True,
            "rho_cell_complete": True,
            "official_test_read": False,
        },
        "manifest_sha256": _sha256(cell_dir / "manifest.json"),
        "metrics_sha256": _sha256(cell_dir / "metrics.jsonl"),
        "artifacts": [
            {
                "kind": "checkpoint",
                "path": "checkpoints/weights_final.npz",
                "sha256": _sha256(checkpoint),
            }
        ],
    }
    _write_json(cell_dir / "result.json", result)
    terminal_status = _status(
        study_id, cell_dir.name, "complete", target=target
    )
    terminal_status["result_sha256"] = _sha256(cell_dir / "result.json")
    _write_json(cell_dir / "status.json", terminal_status)
    return cell


def _make_nonfinite_cell(
    cell_dir: Path,
    *,
    study_id: str,
    surface: dict,
    index: int,
    rho_conv: float,
    rho_dense: float,
    source_sha256: str,
    probe_sha256: str,
    probe_file_sha256: str,
    code: dict,
    target: str,
    source_config: dict,
) -> dict:
    cell_dir.mkdir(parents=True)
    failure = {
        "kind": "nonfinite_training_value",
        "parameter": "ConvWeight_0",
        "confirmed_step": 7,
    }
    learning_rates = {
        "ConvWeight_0": rho_conv,
        "ConvWeight_1": rho_conv / 2,
        "ConvWeight_2": rho_conv / 3,
        "DenseWeight_0": rho_dense,
        "Bias_0": 0.0,
        "Bias_1": 0.0,
        "Bias_2": 0.0,
    }
    learning_rate_vector = [
        learning_rates[name]
        for name in analyzer.WEIGHT_NAMES + analyzer.BIAS_NAMES
    ]
    cell = {
        "index": index,
        "optimizer": surface["optimizer"],
        "bias_policy": "zero",
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "status": "candidate_rejected_nonfinite",
        "safety_failure": failure,
        "learning_rates_by_parameter": learning_rates,
        "learning_rate_vector": learning_rate_vector,
        "signature": _signature(
            optimizer=surface["optimizer"],
            rho_conv=rho_conv,
            rho_dense=rho_dense,
            source_sha256=source_sha256,
            probe_sha256=probe_sha256,
            code=code,
        ),
        "evidence_class": "ordinary_mnist_exploratory",
    }
    _write_json(cell_dir / "cell.json", cell)
    candidate = json.loads(json.dumps(source_config))
    candidate["lr"] = learning_rate_vector
    candidate["optimizer"] = {
        **candidate["optimizer"],
        "name": surface["optimizer"],
        "learning_rate": learning_rate_vector,
    }
    _write_json(cell_dir / "source_config.json", candidate)
    _write_json(cell_dir / "config.used.json", candidate)
    _write_json(
        cell_dir / "safety_diagnostics.json",
        _diagnostics(
            occupancy={},
            projection=0.0,
            processed_steps=7,
            failure=failure,
            complete=False,
        ),
    )
    _write_json(
        cell_dir / "manifest.json",
        _manifest(
            study_id,
            cell_dir.name,
            surface=surface,
            target=target,
            index=index,
            rho_conv=rho_conv,
            rho_dense=rho_dense,
            learning_rates=learning_rate_vector,
            candidate=candidate,
            candidate_sha256=_sha256(cell_dir / "source_config.json"),
            source_sha256=source_sha256,
            probe_file_sha256=probe_file_sha256,
            code=code,
        ),
    )
    (cell_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    _write_json(
        cell_dir / "status.json",
        _status(study_id, cell_dir.name, "failed", target=target),
    )
    return cell


def _make_shards(
    tmp_path: Path,
    *,
    targets: tuple[str, str, str, str] = (
        "jean-zay",
        "jean-zay",
        "jean-zay",
        "jean-zay",
    ),
) -> tuple[Path, list[Path]]:
    initializer_bytes = b"synthetic shared bounded-uniform initializer"
    initializer_sha256 = hashlib.sha256(initializer_bytes).hexdigest()
    study_path = _study(tmp_path, initializer_sha256)
    contract = analyzer.load_contract(study_path)
    roots = []
    for surface in contract.surfaces:
        target = targets[int(surface["index"])]
        code = {
            "commit": "1" * 40,
            "dirty": False,
            "source_archive_sha256": "2" * 64,
            "source_kind": "frozen_archive",
            "working_tree_sha256": "2" * 64,
        }
        root = tmp_path / "shards" / f"{target}-task_{surface['index']}"
        roots.append(root)
        _write_json(
            root / "study.resolved.json",
            {
                "schema_version": analyzer.RESOLVED_SCHEMA,
                "study_id": contract.study_id,
                "evidence_class": "ordinary_mnist_exploratory",
                "study_config_sha256": contract.sha256,
                "initializer_checkpoint_sha256": initializer_sha256,
                "code": code,
                "target": target,
                "surface_count": 4,
                "cells_per_surface": 64,
                "surfaces": list(contract.surfaces),
                "official_test_read": False,
                "restart_interrupted_cells": True,
            },
        )
        asset_root = root / "assets" / "bounded_uniform" / "conv3"
        asset_root.mkdir(parents=True)
        (asset_root / "final_model.pt").write_bytes(initializer_bytes)
        _write_json(
            asset_root / "asset.json",
            {
                "architecture": "conv3",
                "initializer": "bounded_uniform",
                "checkpoint": str(asset_root / "final_model.pt"),
                "checkpoint_sha256": initializer_sha256,
                "expected_checkpoint_sha256": initializer_sha256,
                "official_test_read": False,
                "zero_bias_checkpoint_verification": {
                    "all_exact_zero": True,
                    "nonzero_bias_element_count": 0,
                    "expected_bias_names": list(analyzer.BIAS_NAMES),
                },
            },
        )
        surface_root = root / "surfaces" / surface["surface_id"]
        parent_path = (
            Path(analyzer.__file__).resolve().parents[1]
            / "configs"
            / "conv"
            / "perfectdiode_bounded_uniform_zero_bias_lr_search_conv3_"
            "occupancy_report_only_seed0_20260811_v1.json"
        )
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        source = analyzer.build_source_config(
            parent,
            initializer="bounded_uniform",
            architecture="conv3",
            scheme=surface["scheme"],
            optimizer=surface["optimizer"],
            init_checkpoint_path=asset_root / "final_model.pt",
            dataset_root=tmp_path / f"datasets-{target}",
        )
        _write_json(surface_root / "source_config.json", source)
        source_sha256 = _sha256(surface_root / "source_config.json")
        search_signature = {
            "schema_version": "conv-rho-search/v1",
            "source_config": str(surface_root / "source_config.json"),
            "source_config_sha256": source_sha256,
            "code": code,
            "optimizer": surface["optimizer"],
            "evidence_class": "ordinary_mnist_exploratory",
            "rho_conv": list(contract.rho_conv),
            "rho_dense": list(contract.rho_dense),
            "bias_policy": "zero",
            "probe_stability_scope": "weights_only",
            "probe_batches": [128],
            "stability_tolerance": 0.5,
            "epochs": 3,
            "max_batches": None,
            "max_validation_batches": None,
            "split_seed": 0,
            "shuffle_seed": 0,
            "validation_batch_size": 64,
            "device": "cuda",
            "canary_steps": 0,
            "skip_canary": True,
            "restart_interrupted_cells": True,
            "expected_candidate_steps": 10314,
            "minimum_validation_accuracy": 0.0,
            "target": target,
            "safety": {
                "rejections_enabled": False,
                "bound_occupancy": "report_only",
                "projection_efficiency": "report_only",
                "bound_occupancy_increase_maximum": None,
                "projection_efficiency_minimum": None,
            },
        }
        _write_json(surface_root / "rho" / "resolved.json", search_signature)
        probe_path = surface_root / "rho" / "probe.json"
        _write_json(
            probe_path,
            {
                "schema_version": "conv-rho-probe/v1",
                "status": "complete",
                "probe_stable": True,
                "proposal_units_valid": True,
                "parameter_names": list(
                    analyzer.WEIGHT_NAMES + analyzer.BIAS_NAMES
                ),
                "weight_names": list(analyzer.WEIGHT_NAMES),
                "normalization_unit_by_weight": {
                    "ConvWeight_0": 1.0,
                    "ConvWeight_1": 2.0,
                    "ConvWeight_2": 3.0,
                    "DenseWeight_0": 1.0,
                },
                "initial_parameter_sha256": "3" * 64,
                "search_signature": search_signature,
                "zero_bias_checkpoint_verification": {"all_exact_zero": True},
            },
        )
        probe_payload = json.loads(probe_path.read_text(encoding="utf-8"))
        probe_sha256 = hashlib.sha256(
            json.dumps(
                probe_payload, sort_keys=True, allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        probe_file_sha256 = _sha256(probe_path)
        candidates = []
        for index in range(64):
            conv_index, dense_index = divmod(index, 8)
            cell_dir = surface_root / "rho" / "cells" / f"cell-{index:03d}"
            if surface["index"] == 3 and index == 63:
                cell = _make_nonfinite_cell(
                    cell_dir,
                    study_id=contract.study_id,
                    surface=surface,
                    index=index,
                    rho_conv=contract.rho_conv[conv_index],
                    rho_dense=contract.rho_dense[dense_index],
                    source_sha256=source_sha256,
                    probe_sha256=probe_sha256,
                    probe_file_sha256=probe_file_sha256,
                    code=code,
                    target=target,
                    source_config=source,
                )
            else:
                cell = _make_numeric_cell(
                    cell_dir,
                    study_id=contract.study_id,
                    surface=surface,
                    index=index,
                    rho_conv=contract.rho_conv[conv_index],
                    rho_dense=contract.rho_dense[dense_index],
                    source_sha256=source_sha256,
                    probe_sha256=probe_sha256,
                    probe_file_sha256=probe_file_sha256,
                    code=code,
                    target=target,
                    source_config=source,
                )
            candidates.append({"index": index, "status": cell["status"]})
        counts = dict(Counter(row["status"] for row in candidates))
        _write_json(
            surface_root / "summary.json",
            {
                "schema_version": analyzer.SURFACE_SUMMARY_SCHEMA,
                "study_id": contract.study_id,
                "surface": surface,
                "status": "complete",
                "complete": True,
                "expected_cells": 64,
                "terminal_cells": 64,
                "status_counts": counts,
                "candidates": candidates,
                "official_test_read": False,
            },
        )
        summary_path = surface_root / "summary.json"
        if target == "jean-zay":
            _write_json(
                tmp_path
                / "transport_receipts"
                / f"jean-zay-task_{surface['index']}.json",
                {
                    "schema_version": analyzer.RECEIPT_SCHEMA,
                    "study_id": contract.study_id,
                    "surface": surface,
                    "task_id": surface["index"],
                    "target": "jean-zay",
                    "environment_id": "pytorch-gpu/py3/2.5.0",
                    "source_commit": "1" * 40,
                    "source_archive_sha256": "2" * 64,
                    "study_config_sha256": contract.sha256,
                    "initializer_checkpoint_sha256": initializer_sha256,
                    "summary_status": "complete",
                    "expected_cells": 64,
                    "terminal_cells": 64,
                    "validated_cell_bundles": 64,
                    "status_counts": counts,
                    "summary_path": str(summary_path),
                    "summary_sha256": _sha256(summary_path),
                    "official_test_read": False,
                    "semantic_pass": True,
                },
            )
        else:
            _write_json(
                root / "transport_receipt.json",
                {
                    "schema_version": analyzer.LOCAL_RECEIPT_SCHEMA,
                    "study_id": contract.study_id,
                    "target": target,
                    "host": f"synthetic-{target}",
                    "surface_id": surface["surface_id"],
                    "surface_status": "complete",
                    "validated_cell_bundles": 64,
                    "initializer_checkpoint_sha256": initializer_sha256,
                    "source_commit": "1" * 40,
                    "source_archive_sha256": "2" * 64,
                    "python": "3.12.synthetic",
                    "torch": "2.5.synthetic",
                    "cuda_device": f"synthetic-{target}-gpu",
                    "official_test_read": False,
                    "recorded_at": "2026-08-11T20:00:00+00:00",
                },
            )
    return study_path, roots


def test_analyzer_validates_all_cells_and_writes_noncanonical_outputs(
    tmp_path: Path,
) -> None:
    study_path, roots = _make_shards(tmp_path)
    output = tmp_path / "analysis"

    report = analyzer.analyze(
        study_config_path=study_path,
        shard_roots=roots,
        output_dir=output,
    )

    assert report["exploratory_noncanonical"] is True
    assert report["canonical_lr_handoff"] is False
    assert report["paper_evidence"] is False
    assert report["analysis_generated_at"].endswith("+00:00")
    assert report["analyzer_source_sha256"] == _sha256(
        Path(analyzer.__file__).resolve()
    )
    assert report["probe_initial_parameter_sha256"] == "3" * 64
    assert report["coverage"] == {
        "surface_count": 4,
        "expected_surface_count": 4,
        "cell_count": 256,
        "expected_cell_count": 256,
        "cells_per_surface": 64,
        "numeric_cell_count": 255,
        "nonfinite_cell_count": 1,
        "canonical_bundle_count": 256,
    }
    assert len(report["cells"]) == 256
    assert report["cells"][0]["exact_bound_occupancy_percent"] == 50.0
    assert report["cells"][-1]["canonical_outcome"] == "nonfinite"
    assert report["cells"][-1]["final_validation_accuracy"] is None
    assert report["cells"][-1]["result_json_sha256"] is None
    assert all(row["bias_lr_zero"] for row in report["cells"])
    assert all(row["official_test_read"] is False for row in report["cells"])
    assert {row["target"] for row in report["cells"]} == {"jean-zay"}
    assert {row["target"] for row in report["surfaces"]} == {"jean-zay"}
    assert {row["target"] for row in report["source_shards"]} == {"jean-zay"}
    assert {
        row["schema_version"] for row in report["transport_receipts"]
    } == {analyzer.RECEIPT_SCHEMA}
    assert report["correlation_interpretation"] == {
        "causal": False,
        "scope": "within_surface_numeric_complete_cells",
        "caveat": analyzer.CORRELATION_CAVEAT,
    }
    assert report["target_assignment"]["targets_by_optimizer"] == {
        "SGD": "jean-zay",
        "Adam": "jean-zay",
    }
    assert report["target_assignment"]["cross_optimizer_comparison"] == {
        "same_target": True,
        "same_host_runtime_contract": True,
        "host_runtime_confounded": False,
        "causal_interpretation_allowed": False,
        "statement": (
            "SGD and Adam both ran on jean-zay; this target assignment adds no "
            "cross-optimizer host difference, but the exploratory design still "
            "does not support causal claims."
        ),
    }
    first_surface = report["surfaces"][0]
    assert first_surface["best_observed_cell_index"] == 63
    assert first_surface["best_observed_rho_conv"] == pytest.approx(0.009)
    assert first_surface["best_observed_rho_dense"] == pytest.approx(0.03)
    assert first_surface["best_observed_weight_learning_rates"] == pytest.approx(
        {
            "ConvWeight_0": 0.009,
            "ConvWeight_1": 0.0045,
            "ConvWeight_2": 0.003,
            "DenseWeight_0": 0.03,
        }
    )
    assert first_surface["best_observed_bias_learning_rates"] == {
        "Bias_0": 0.0,
        "Bias_1": 0.0,
        "Bias_2": 0.0,
    }
    assert first_surface["best_observed_bias_lrs_zero"] is True
    assert first_surface["range_status"] == "open_boundary"
    assert first_surface["locally_bracketed_by_accuracy"] is False
    assert first_surface["open_directions"] == [
        "upper_rho_conv",
        "upper_rho_dense",
    ]
    assert set(first_surface["axial_neighbors"]) == {
        "lower_rho_conv",
        "upper_rho_conv",
        "lower_rho_dense",
        "upper_rho_dense",
    }
    assert first_surface["axial_neighbors"]["lower_rho_conv"][
        "cell_index"
    ] == 55
    assert first_surface["axial_neighbors"]["upper_rho_conv"][
        "grid_state"
    ] == "outside_tested_range"
    first_correlations = report["surfaces"][0]["descriptive_correlations"]
    assert first_correlations["noncausal"] is True
    assert first_correlations["accuracy_vs_exact_bound_occupancy"][
        "sample_count"
    ] == 64
    assert first_correlations["accuracy_vs_exact_bound_occupancy"][
        "pearson"
    ] is not None
    assert first_correlations["accuracy_vs_exact_bound_occupancy"][
        "spearman"
    ] is not None
    by_parameter = first_correlations[
        "accuracy_vs_exact_bound_occupancy_by_parameter"
    ]
    assert set(by_parameter) == set(analyzer.WEIGHT_NAMES)
    assert all(value["sample_count"] == 64 for value in by_parameter.values())
    assert first_correlations["accuracy_vs_median_projection_efficiency"] == {
        "sample_count": 64,
        "pearson": pytest.approx(-1.0),
        "spearman": pytest.approx(-1.0),
    }

    assert len(report["included_runs"]) == 256
    assert len(report["numeric_summary_excluded_terminal_runs"]) == 1
    assert report["excluded_runs"] == []
    assert report["excluded_execution_attempts"] == []
    reconciliation = report["run_inventory"]["reconciliation"]
    assert reconciliation["reconciled"] is True
    assert reconciliation["canonical_terminal_cell_count"] == 256
    assert reconciliation["numeric_summary_included_count"] == 255
    assert reconciliation["numeric_summary_excluded_terminal_count"] == 1
    assert reconciliation["source_result_json_count"] == 255
    assert reconciliation["result_json_expected_absent_count"] == 1
    assert len(report["run_inventory"]["source_result_jsons"]) == 255
    authoritative_results = {
        (Path(row["result_json_path"]), row["result_json_sha256"])
        for row in report["included_runs"]
        if row["result_json_path"] is not None
    }
    assert len(authoritative_results) == 255
    assert all(path.is_file() and _sha256(path) == digest for path, digest in authoritative_results)
    retained = report["numeric_summary_excluded_terminal_runs"][0]
    assert retained["coverage_included"] is True
    assert retained["numeric_summary_included"] is False
    assert retained["result_json_sha256"] is None
    with pytest.raises(
        ValueError, match="cannot reconcile the 256 declared cells"
    ):
        analyzer._canonical_run_inventory(
            report["cells"][:-1], analyzer.load_contract(study_path)
        )

    expected = {
        "cells.csv",
        "summary.csv",
        "surfaces.csv",
        "summary.json",
        "report.md",
        "plots/final_validation_accuracy_heatmaps.png",
        "plots/endpoint_bound_occupancy_heatmaps.png",
        "plots/projection_efficiency_heatmaps.png",
        "plots/accuracy_vs_diagnostics.png",
    }
    assert all((output / relative).is_file() for relative in expected)
    summary_payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["surfaces"][0][
        "best_observed_weight_learning_rates"
    ] == pytest.approx(first_surface["best_observed_weight_learning_rates"])
    assert summary_payload["surfaces"][0][
        "best_observed_bias_learning_rates"
    ] == first_surface["best_observed_bias_learning_rates"]
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "NONCANONICAL" in markdown
    assert "Descriptive diagnostic correlations" in markdown
    assert "Four-direction axial range evidence" in markdown
    assert "`open_boundary`" in markdown
    assert "source `result.json` hashes" in markdown
    assert "Endpoint occupancy at each surface's highest-accuracy point" in markdown
    assert "Actual raw learning-rate vector at each highest-accuracy point" in markdown
    assert (
        "| baseline | SGD | jean-zay | 63 | 0.009 | 0.03 | 0.009 | "
        "0.0045 | 0.003 | 0.03 | `Bias_0=Bias_1=Bias_2=0` |"
    ) in markdown
    assert "Exact endpoint occupancy — ConvWeight_0" in markdown
    assert "Noncausal" in markdown
    with (output / "cells.csv").open(encoding="utf-8", newline="") as handle:
        cell_rows = list(csv.DictReader(handle))
    assert len(cell_rows) == 256
    assert {row["target"] for row in cell_rows} == {"jean-zay"}
    assert all(row["manifest_json_sha256"] for row in cell_rows)
    assert all(row["status_json_sha256"] for row in cell_rows)
    assert sum(bool(row["result_json_sha256"]) for row in cell_rows) == 255
    with (output / "surfaces.csv").open(encoding="utf-8", newline="") as handle:
        surface_rows = list(csv.DictReader(handle))
    assert len(surface_rows) == 4
    assert {row["target"] for row in surface_rows} == {"jean-zay"}
    assert float(surface_rows[0]["best_observed_lr_conv_weight_0"]) == pytest.approx(
        0.009
    )
    assert json.loads(
        surface_rows[0]["best_observed_weight_learning_rates_json"]
    ) == pytest.approx(first_surface["best_observed_weight_learning_rates"])
    assert json.loads(
        surface_rows[0]["best_observed_bias_learning_rates_json"]
    ) == first_surface["best_observed_bias_learning_rates"]
    assert {
        row["cross_optimizer_host_runtime_confounded"] for row in surface_rows
    } == {"False"}
    assert surface_rows[0]["range_status"] == "open_boundary"
    assert json.loads(surface_rows[0]["open_directions_json"]) == [
        "upper_rho_conv",
        "upper_rho_dense",
    ]
    assert set(json.loads(surface_rows[0]["axial_neighbors_json"])) == {
        "lower_rho_conv",
        "upper_rho_conv",
        "lower_rho_dense",
        "upper_rho_dense",
    }
    assert (output / "summary.csv").read_bytes() == (
        output / "surfaces.csv"
    ).read_bytes()

    no_best = analyzer._surface_summary(
        analyzer._surface_specs()[0],
        [{"target": "jean-zay", "canonical_outcome": "nonfinite"}],
    )
    assert no_best["best_observed_cell_index"] is None
    assert no_best["best_observed_weight_learning_rates"] is None
    assert no_best["best_observed_weight_learning_rates_json"] == "null"
    assert no_best["best_observed_bias_learning_rates"] is None
    assert no_best["best_observed_bias_learning_rates_json"] == "null"
    assert no_best["best_observed_bias_lrs_zero"] is None
    assert no_best["range_status"] == "no_numeric_observation"
    no_best_report = dict(report)
    no_best_report["surfaces"] = [no_best, *report["surfaces"][1:]]
    no_best_markdown = analyzer._markdown(no_best_report)
    assert (
        "| baseline | SGD | jean-zay | unavailable | unavailable | unavailable | "
        "unavailable | unavailable | unavailable | unavailable | unavailable |"
    ) in no_best_markdown

    duplicate_roots = [*roots[:-1], roots[0]]
    with pytest.raises(ValueError, match="four unique shard surfaces"):
        analyzer.analyze(
            study_config_path=study_path,
            shard_roots=duplicate_roots,
            output_dir=tmp_path / "duplicate-analysis",
        )

    unresolved_summary_path = (
        roots[3]
        / "surfaces"
        / analyzer._surface_specs()[3]["surface_id"]
        / "summary.json"
    )
    unresolved_summary = json.loads(unresolved_summary_path.read_text(encoding="utf-8"))
    unresolved_summary.update(
        {
            "status": "unresolved_probe",
            "complete": False,
            "terminal_cells": 0,
            "status_counts": {},
            "candidates": [],
        }
    )
    _write_json(unresolved_summary_path, unresolved_summary)
    receipt_path = tmp_path / "transport_receipts" / "jean-zay-task_3.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(
        {
            "summary_status": "unresolved_probe",
            "terminal_cells": 0,
            "validated_cell_bundles": 0,
            "status_counts": {},
            "summary_sha256": _sha256(unresolved_summary_path),
        }
    )
    _write_json(receipt_path, receipt)
    with pytest.raises(
        analyzer.IncompleteExploratoryCoverageError,
        match="not a transport failure",
    ):
        analyzer.analyze(
            study_config_path=study_path,
            shard_roots=roots,
            output_dir=tmp_path / "unresolved-analysis",
        )


def test_analyzer_accepts_and_discovers_mixed_akib_trex_local_receipts(
    tmp_path: Path,
) -> None:
    study_path, _roots = _make_shards(tmp_path, targets=LOCAL_TARGETS)

    report = analyzer.analyze(
        study_config_path=study_path,
        study_root=tmp_path,
        output_dir=tmp_path / "local-analysis",
    )

    assert [row["target"] for row in report["source_shards"]] == list(
        LOCAL_TARGETS
    )
    assert [row["target"] for row in report["transport_receipts"]] == list(
        LOCAL_TARGETS
    )
    assert {
        row["schema_version"] for row in report["transport_receipts"]
    } == {analyzer.LOCAL_RECEIPT_SCHEMA}
    assert report["coverage"]["surface_count"] == 4
    assert report["coverage"]["cell_count"] == 256
    assert [row["target"] for row in report["surfaces"]] == list(LOCAL_TARGETS)
    assignment = report["target_assignment"]
    assert assignment["derived_from_validated_transport_receipts"] is True
    assert assignment["targets_by_optimizer"] == {"SGD": "akib", "Adam": "trex"}
    assert [
        (row["optimizer"], row["target"], row["same_target"])
        for row in assignment["within_optimizer_scheme_comparisons"]
    ] == [("SGD", "akib", True), ("Adam", "trex", True)]
    assert assignment["cross_optimizer_comparison"] == {
        "same_target": False,
        "same_host_runtime_contract": False,
        "host_runtime_confounded": True,
        "causal_interpretation_allowed": False,
        "statement": (
            "SGD ran on akib while Adam ran on trex; their validated host/runtime "
            "contracts differ, so SGD-vs-Adam differences must not be "
            "interpreted causally as optimizer effects."
        ),
    }
    markdown = (tmp_path / "local-analysis" / "report.md").read_text(
        encoding="utf-8"
    )
    assert "| baseline | SGD | akib |" in markdown
    assert "| ours | Adam | trex |" in markdown
    assert "Baseline-vs-ours for SGD is a within-host/runtime comparison on akib." in markdown
    assert "SGD ran on akib while Adam ran on trex" in markdown
    assert "must not be interpreted causally as optimizer effects" in markdown
    with (tmp_path / "local-analysis" / "surfaces.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["target"] for row in csv_rows] == list(LOCAL_TARGETS)
    assert {row["baseline_vs_ours_within_target"] for row in csv_rows} == {
        "True"
    }
    assert {
        row["cross_optimizer_host_runtime_confounded"] for row in csv_rows
    } == {"True"}
    assert {
        row["cross_optimizer_causal_interpretation_allowed"] for row in csv_rows
    } == {"False"}
    assert all(
        "SGD ran on akib while Adam ran on trex"
        in row["cross_optimizer_comparison_statement"]
        for row in csv_rows
    )


def test_analyzer_rejects_host_split_baseline_ours_for_one_optimizer(
    tmp_path: Path,
) -> None:
    study_path, roots = _make_shards(
        tmp_path,
        targets=("akib", "trex", "trex", "trex"),
    )

    output = tmp_path / "host-split-analysis"
    with pytest.raises(
        ValueError,
        match="Matched baseline-vs-ours SGD surfaces are split across targets",
    ):
        analyzer.analyze(
            study_config_path=study_path,
            shard_roots=roots,
            output_dir=output,
        )
    assert not output.exists()


def test_same_target_but_different_receipt_runtime_is_reported_as_confound(
    tmp_path: Path,
) -> None:
    study_path, roots = _make_shards(tmp_path, targets=LOCAL_TARGETS)
    receipt_path = roots[2] / "transport_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["torch"] = "2.6.synthetic"
    _write_json(receipt_path, receipt)

    report = analyzer.analyze(
        study_config_path=study_path,
        shard_roots=roots,
        output_dir=tmp_path / "runtime-confound-analysis",
    )

    sgd = report["target_assignment"]["within_optimizer_scheme_comparisons"][0]
    assert sgd["same_target"] is True
    assert sgd["same_host_runtime_contract"] is False
    assert sgd["host_runtime_confounded"] is True
    assert sgd["host_target_confounded"] is True
    assert sgd["causal_scheme_interpretation_allowed"] is False
    assert sgd["baseline_execution_environment"] == {
        "target": "akib",
        "host": "synthetic-akib",
        "python": "3.12.synthetic",
        "torch": "2.5.synthetic",
        "cuda_device": "synthetic-akib-gpu",
        "environment_id": None,
    }
    assert sgd["ours_execution_environment"]["host"] == "synthetic-akib"
    assert sgd["ours_execution_environment"]["torch"] == "2.6.synthetic"
    assert "host/runtime tuples differ" in sgd["statement"]


def test_cell_runtime_must_match_local_transport_receipt(tmp_path: Path) -> None:
    study_path, roots = _make_shards(tmp_path, targets=LOCAL_TARGETS)
    surface_root = roots[2] / "surfaces" / analyzer._surface_specs()[2]["surface_id"]
    for cell_dir in (surface_root / "rho" / "cells").iterdir():
        manifest_path = cell_dir / "manifest.json"
        status_path = cell_dir / "status.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        manifest["runtime"]["host"] = "synthetic-akib-other-host"
        status["runtime"] = manifest["runtime"]
        _write_json(manifest_path, manifest)
        result_path = cell_dir / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["manifest_sha256"] = _sha256(manifest_path)
            _write_json(result_path, result)
            status["result_sha256"] = _sha256(result_path)
        _write_json(status_path, status)

    with pytest.raises(ValueError, match="does not match its transport receipt"):
        analyzer.analyze(
            study_config_path=study_path,
            shard_roots=roots,
            output_dir=tmp_path / "runtime-identity-analysis",
        )


def _coherently_rehash_surface_source(
    root: Path,
    surface: dict,
    *,
    training_algorithm: str | None = None,
    init_checkpoint_path: str | None = None,
) -> None:
    surface_root = root / "surfaces" / surface["surface_id"]
    source_path = surface_root / "source_config.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if training_algorithm is not None:
        source["training_algorithm"] = training_algorithm
    if init_checkpoint_path is not None:
        source["init_checkpoint_path"] = init_checkpoint_path
    _write_json(source_path, source)
    source_sha256 = _sha256(source_path)

    resolved_path = surface_root / "rho" / "resolved.json"
    probe_path = surface_root / "rho" / "probe.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved["source_config_sha256"] = source_sha256
    _write_json(resolved_path, resolved)
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe["search_signature"] = resolved
    _write_json(probe_path, probe)
    probe_file_sha256 = _sha256(probe_path)
    probe_semantic_sha256 = hashlib.sha256(
        json.dumps(probe, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()

    for cell_dir in (surface_root / "rho" / "cells").iterdir():
        cell_path = cell_dir / "cell.json"
        cell = json.loads(cell_path.read_text(encoding="utf-8"))
        cell["signature"]["source_config_sha256"] = source_sha256
        cell["signature"]["probe_sha256"] = probe_semantic_sha256
        _write_json(cell_path, cell)

        candidate = json.loads(json.dumps(source))
        vector = cell["learning_rate_vector"]
        candidate["lr"] = vector
        candidate["optimizer"]["learning_rate"] = vector
        candidate_path = cell_dir / "source_config.json"
        _write_json(candidate_path, candidate)
        _write_json(cell_dir / "config.used.json", candidate)

        manifest_path = cell_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["configuration"]["sha256"] = _sha256(candidate_path)
        manifest["configuration"]["resolved"] = candidate
        for item in manifest["inputs"]:
            if item["role"] == "source_config":
                item["sha256"] = source_sha256
            elif item["role"] == "optimizer_probe":
                item["sha256"] = probe_file_sha256
        _write_json(manifest_path, manifest)
        result_path = cell_dir / "result.json"
        status_path = cell_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["manifest_sha256"] = _sha256(manifest_path)
            _write_json(result_path, result)
            status["result_sha256"] = _sha256(result_path)
        _write_json(status_path, status)


@pytest.mark.parametrize(
    ("training_algorithm", "init_checkpoint_path", "message"),
    [
        ("EP", None, "complete governed Conv3 BP training contract"),
        (None, "/unrelated/final_model.pt", "pinned initializer asset"),
    ],
)
def test_coherently_rehashed_alien_source_contract_fails_closed(
    tmp_path: Path,
    training_algorithm: str | None,
    init_checkpoint_path: str | None,
    message: str,
) -> None:
    study_path, roots = _make_shards(tmp_path, targets=LOCAL_TARGETS)
    surface = analyzer._surface_specs()[0]
    _coherently_rehash_surface_source(
        roots[0],
        surface,
        training_algorithm=training_algorithm,
        init_checkpoint_path=init_checkpoint_path,
    )

    with pytest.raises(ValueError, match=message):
        analyzer.analyze(
            study_config_path=study_path,
            shard_roots=roots,
            output_dir=tmp_path / "alien-source-analysis",
        )


def test_coherently_rehashed_alien_single_cell_config_fails_closed(
    tmp_path: Path,
) -> None:
    study_path, roots = _make_shards(tmp_path, targets=LOCAL_TARGETS)
    cell_dir = (
        roots[0]
        / "surfaces"
        / analyzer._surface_specs()[0]["surface_id"]
        / "rho"
        / "cells"
        / "cell-000"
    )
    candidate_path = cell_dir / "source_config.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["training_algorithm"] = "EP"
    _write_json(candidate_path, candidate)
    _write_json(cell_dir / "config.used.json", candidate)
    manifest_path = cell_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration"]["resolved"] = candidate
    manifest["configuration"]["sha256"] = _sha256(candidate_path)
    _write_json(manifest_path, manifest)
    result_path = cell_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["manifest_sha256"] = _sha256(manifest_path)
    _write_json(result_path, result)
    status_path = cell_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["result_sha256"] = _sha256(result_path)
    _write_json(status_path, status)

    with pytest.raises(ValueError, match="resolved scientific configuration mismatch"):
        analyzer.analyze(
            study_config_path=study_path,
            shard_roots=roots,
            output_dir=tmp_path / "alien-cell-analysis",
        )


def test_collected_shards_may_be_relocated_with_stale_execution_paths(
    tmp_path: Path,
) -> None:
    study_path, roots = _make_shards(tmp_path, targets=LOCAL_TARGETS)
    relocated_parent = tmp_path / "relocated" / "shards"
    relocated_parent.mkdir(parents=True)
    relocated = []
    for root in roots:
        destination = relocated_parent / root.name
        root.rename(destination)
        relocated.append(destination)

    report = analyzer.analyze(
        study_config_path=study_path,
        shard_roots=relocated,
        output_dir=tmp_path / "relocated-analysis",
    )

    assert report["coverage"]["cell_count"] == 256


@pytest.mark.parametrize("tamper", ["wrong_receipt", "wrong_metadata"])
def test_current_recovery_binding_requires_canonical_receipt_and_metadata(
    tmp_path: Path, tamper: str
) -> None:
    study_path, roots = _make_shards(tmp_path, targets=LOCAL_TARGETS)
    cell_dir = (
        roots[0]
        / "surfaces"
        / analyzer._surface_specs()[0]["surface_id"]
        / "rho"
        / "cells"
        / "cell-000"
    )
    attempt_dir = cell_dir / "recovery_attempts" / "attempt_001"
    _write_json(attempt_dir / "cell.json", {"status": "running"})
    _write_json(
        attempt_dir / "recovery.json",
        {
            "schema_version": "conv-rho-interrupted-attempt/v1",
            "attempt": 1,
            "previous_cell_status": "running",
            "previous_reporting_state": "running",
            "archived_entries": ["cell.json"],
            "restart_mode": "clean_from_exact_initializer",
        },
    )
    binding = {
        "attempt": 1,
        "path": "recovery_attempts/attempt_001",
        "receipt": "recovery_attempts/attempt_001/recovery.json",
        "receipt_sha256": _sha256(attempt_dir / "recovery.json"),
        "restart_mode": "clean_from_exact_initializer",
    }
    if tamper == "wrong_receipt":
        binding["receipt"] = "recovery_attempts/attempt_001/cell.json"
        binding["receipt_sha256"] = _sha256(attempt_dir / "cell.json")
    else:
        binding["attempt"] = 999
        binding["restart_mode"] = "unsafe_resume"

    cell_path = cell_dir / "cell.json"
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    cell["recovery"] = binding
    _write_json(cell_path, cell)
    manifest_path = cell_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration"]["recovery"] = binding
    manifest["inputs"].append(
        {
            "role": "interrupted_attempt_receipt",
            "path": binding["receipt"],
            "sha256": binding["receipt_sha256"],
        }
    )
    _write_json(manifest_path, manifest)
    result_path = cell_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["manifest_sha256"] = _sha256(manifest_path)
    _write_json(result_path, result)
    status_path = cell_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["result_sha256"] = _sha256(result_path)
    _write_json(status_path, status)

    with pytest.raises(ValueError, match="recovery binding is noncanonical"):
        analyzer.analyze(
            study_config_path=study_path,
            shard_roots=roots,
            output_dir=tmp_path / "bad-current-recovery-analysis",
        )


def test_inventory_hashes_and_excludes_smoke_and_archived_attempt(
    tmp_path: Path,
) -> None:
    study_path, roots = _make_shards(tmp_path)
    smoke_result = tmp_path / "smoke" / "local-gate" / "smoke" / "result.json"
    _write_json(
        smoke_result,
        {
            "schema_version": "synthetic-smoke-result/v1",
            "run_id": "synthetic-smoke",
            "smoke": True,
        },
    )
    cell_dir = (
        roots[0]
        / "surfaces"
        / analyzer._surface_specs()[0]["surface_id"]
        / "rho"
        / "cells"
        / "cell-000"
    )
    attempt_dir = cell_dir / "recovery_attempts" / "attempt_001"
    _write_json(attempt_dir / "cell.json", {"status": "running"})
    _write_json(
        attempt_dir / "recovery.json",
        {
            "schema_version": "conv-rho-interrupted-attempt/v1",
            "attempt": 1,
            "previous_cell_status": "running",
            "previous_reporting_state": "running",
            "archived_entries": ["cell.json"],
            "restart_mode": "clean_from_exact_initializer",
        },
    )
    recovery_binding = {
        "attempt": 1,
        "path": "recovery_attempts/attempt_001",
        "receipt": "recovery_attempts/attempt_001/recovery.json",
        "receipt_sha256": _sha256(attempt_dir / "recovery.json"),
        "restart_mode": "clean_from_exact_initializer",
    }
    cell_path = cell_dir / "cell.json"
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    cell["recovery"] = recovery_binding
    _write_json(cell_path, cell)
    manifest_path = cell_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration"]["recovery"] = recovery_binding
    manifest["inputs"].append(
        {
            "role": "interrupted_attempt_receipt",
            "path": recovery_binding["receipt"],
            "sha256": recovery_binding["receipt_sha256"],
        }
    )
    _write_json(manifest_path, manifest)
    result_path = cell_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["manifest_sha256"] = _sha256(manifest_path)
    _write_json(result_path, result)
    status_path = cell_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["result_sha256"] = _sha256(result_path)
    _write_json(status_path, status)

    report = analyzer.analyze(
        study_config_path=study_path,
        shard_roots=roots,
        output_dir=tmp_path / "inventory-analysis",
    )

    by_kind = {row["evidence_kind"]: row for row in report["excluded_runs"]}
    assert set(by_kind) == {"smoke", "archived_interrupted_attempt"}
    assert by_kind["smoke"]["source_result_jsons"] == [
        {"path": str(smoke_result.resolve()), "sha256": _sha256(smoke_result)}
    ]
    archived = by_kind["archived_interrupted_attempt"]
    assert archived["source_result_jsons"] == []
    assert archived["coverage_included"] is False
    assert archived["recovery_receipt"]["sha256"] == _sha256(
        attempt_dir / "recovery.json"
    )
    source_results = report["run_inventory"]["source_result_jsons"]
    assert len(source_results) == 256
    assert sum(row["authority"] == "canonical_terminal_cell" for row in source_results) == 255
    assert sum(row["authority"] == "smoke" for row in source_results) == 1
    markdown = (tmp_path / "inventory-analysis" / "report.md").read_text(
        encoding="utf-8"
    )
    assert "`smoke`" in markdown
    assert "`archived_interrupted_attempt`" in markdown


def test_inventory_discovers_failed_gate_bundle_without_result_and_prefers_gate(
    tmp_path: Path,
) -> None:
    study_path, roots = _make_shards(tmp_path)
    gate = tmp_path / "gates" / "trex-target-gate" / "smoke" / "failed-cell"
    _write_json(
        gate / "manifest.json",
        {
            "schema_version": "experiment-run-manifest/v1",
            "study_id": "gate-smoke",
            "run_id": "failed-cell",
            "arm_id": "failed-cell",
        },
    )
    _write_json(
        gate / "status.json",
        {
            "schema_version": "experiment-run-status/v1",
            "study_id": "gate-smoke",
            "run_id": "failed-cell",
            "arm_id": "failed-cell",
            "state": "failed",
        },
    )

    report = analyzer.analyze(
        study_config_path=study_path,
        shard_roots=roots,
        output_dir=tmp_path / "gate-inventory-analysis",
    )

    gate_rows = [
        row
        for row in report["excluded_runs"]
        if row["path"] == str(gate.resolve())
    ]
    assert len(gate_rows) == 1
    assert gate_rows[0]["evidence_kind"] == "target_gate"
    assert gate_rows[0]["reporting_state"] == "failed"
    assert gate_rows[0]["source_result_jsons"] == []
    assert gate_rows[0]["reporting_files_present"] == [
        "manifest.json",
        "status.json",
    ]
    reconciliation = report["run_inventory"]["reconciliation"]
    assert reconciliation["discovered_operational_bundle_count"] == 1
    assert reconciliation["discovered_target_gate_count"] == 1
    assert reconciliation["all_discovered_source_result_json_count"] == 255


def test_inventory_retains_provenance_documented_gate_without_reporting_bundle(
    tmp_path: Path,
) -> None:
    study_path, roots = _make_shards(tmp_path)
    study_payload = json.loads(study_path.read_text(encoding="utf-8"))
    study_payload["study_id"] = analyzer.CANONICAL_STUDY_ID
    _write_json(study_path, study_payload)
    # The documented attempt can be absent from the collected tree; the
    # provenance receipt is still evidence that must be reconciled.
    remote_path = "/remote/study/gates/trex-target-gate"
    _write_json(
        tmp_path / "provenance" / "local_fallback_submission.json",
        {
            "schema_version": (
                "perfectdiode-conv3-exploratory-lr-ladder-local-fallback-submission/v1"
            ),
            "study_id": analyzer.CANONICAL_STUDY_ID,
            "target_gates": {
                "akib": {
                    "path": (
                        "/remote/study/gates/akib-target-gate/smoke/result.json"
                    ),
                    "status": "passed",
                },
                "trex": {
                    "failed_attempt": {
                        "path": remote_path,
                        "failure": "initializer serialization mismatch",
                        "diagnosis": "tensor values were identical",
                    },
                    "passing_attempt": {
                        "path": (
                            "/remote/study/gates/trex-target-gate-02/smoke/result.json"
                        ),
                        "status": "passed",
                    },
                }
            },
        },
    )

    records = analyzer._documented_operational_attempt_inventory(tmp_path, [])

    assert len(records) == 3
    assert {record["evidence_kind"] for record in records} == {"target_gate"}
    assert {
        (record["target"], record["reporting_state"])
        for record in records
    } == {
        ("akib", "passed_documented_remote_gate"),
        ("trex", "failed_before_reporting_bundle"),
        ("trex", "passed_documented_remote_gate"),
    }
    failed = next(
        record
        for record in records
        if record["reporting_state"] == "failed_before_reporting_bundle"
    )
    assert failed["documented_remote_path"] == remote_path
    assert all(record["source_result_jsons"] == [] for record in records)


@pytest.mark.parametrize("receipt_name", [None, "recovery-typo.json"])
def test_archived_recovery_attempts_fail_closed_on_missing_or_misnamed_receipt(
    tmp_path: Path, receipt_name: str | None
) -> None:
    study_path, roots = _make_shards(tmp_path)
    cell_dir = (
        roots[0]
        / "surfaces"
        / analyzer._surface_specs()[0]["surface_id"]
        / "rho"
        / "cells"
        / "cell-000"
    )
    attempt_dir = cell_dir / "recovery_attempts" / "attempt_001"
    _write_json(attempt_dir / "cell.json", {"status": "running"})
    if receipt_name is not None:
        _write_json(
            attempt_dir / receipt_name,
            {
                "schema_version": "conv-rho-interrupted-attempt/v1",
                "attempt": 1,
                "previous_cell_status": "running",
                "previous_reporting_state": "running",
                "archived_entries": ["cell.json"],
                "restart_mode": "clean_from_exact_initializer",
            },
        )

    with pytest.raises(ValueError, match="missing recovery.json"):
        analyzer.analyze(
            study_config_path=study_path,
            shard_roots=roots,
            output_dir=tmp_path / "bad-recovery-analysis",
        )


def test_noncanonical_recovery_attempt_directory_is_also_fail_closed(
    tmp_path: Path,
) -> None:
    study_path, roots = _make_shards(tmp_path)
    _write_json(
        tmp_path
        / "operational"
        / "failed-run"
        / "recovery_attempts"
        / "attempt_001"
        / "cell.json",
        {"status": "running"},
    )

    with pytest.raises(ValueError, match="missing recovery.json"):
        analyzer.analyze(
            study_config_path=study_path,
            shard_roots=roots,
            output_dir=tmp_path / "bad-noncanonical-recovery-analysis",
        )


def test_canonical_archived_attempt_requires_current_latest_recovery_binding(
    tmp_path: Path,
) -> None:
    study_path, roots = _make_shards(tmp_path)
    cell_dir = (
        roots[0]
        / "surfaces"
        / analyzer._surface_specs()[0]["surface_id"]
        / "rho"
        / "cells"
        / "cell-000"
    )
    attempt_dir = cell_dir / "recovery_attempts" / "attempt_001"
    _write_json(attempt_dir / "cell.json", {"status": "running"})
    _write_json(
        attempt_dir / "recovery.json",
        {
            "schema_version": "conv-rho-interrupted-attempt/v1",
            "attempt": 1,
            "previous_cell_status": "running",
            "previous_reporting_state": "running",
            "archived_entries": ["cell.json"],
            "restart_mode": "clean_from_exact_initializer",
        },
    )

    with pytest.raises(ValueError, match="omits recovery binding"):
        analyzer.analyze(
            study_config_path=study_path,
            shard_roots=roots,
            output_dir=tmp_path / "unbound-recovery-analysis",
        )


def test_archived_recovery_attempt_rejects_duplicate_misnamed_receipt(
    tmp_path: Path,
) -> None:
    study_path, roots = _make_shards(tmp_path)
    cell_dir = (
        roots[0]
        / "surfaces"
        / analyzer._surface_specs()[0]["surface_id"]
        / "rho"
        / "cells"
        / "cell-000"
    )
    attempt_dir = cell_dir / "recovery_attempts" / "attempt_001"
    receipt = {
        "schema_version": "conv-rho-interrupted-attempt/v1",
        "attempt": 1,
        "previous_cell_status": "running",
        "previous_reporting_state": "running",
        "archived_entries": ["cell.json"],
        "restart_mode": "clean_from_exact_initializer",
    }
    _write_json(attempt_dir / "cell.json", {"status": "running"})
    _write_json(attempt_dir / "recovery.json", receipt)
    _write_json(attempt_dir / "recovery-copy.json", receipt)

    with pytest.raises(ValueError, match="misnamed or duplicate receipt"):
        analyzer.analyze(
            study_config_path=study_path,
            shard_roots=roots,
            output_dir=tmp_path / "duplicate-recovery-analysis",
        )


def test_analyzer_rejects_named_learning_rate_map_tamper_of_123(
    tmp_path: Path,
) -> None:
    study_path, roots = _make_shards(tmp_path)
    cell_path = (
        roots[0]
        / "surfaces"
        / analyzer._surface_specs()[0]["surface_id"]
        / "rho"
        / "cells"
        / "cell-000"
        / "cell.json"
    )
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    cell["learning_rates_by_parameter"]["ConvWeight_0"] = 123
    _write_json(cell_path, cell)

    with pytest.raises(
        ValueError, match="Named learning-rate map does not match ordered vector"
    ):
        analyzer.analyze(
            study_config_path=study_path,
            shard_roots=roots,
            output_dir=tmp_path / "tampered-map-analysis",
        )


@pytest.mark.parametrize(
    ("relative_path", "mutation", "message"),
    [
        (
            Path("cell.json"),
            lambda payload: payload.update({"rho_conv": 123}),
            "Cell identity/grid mismatch",
        ),
        (
            Path("cell.json"),
            lambda payload: payload["signature"].update({"probe_sha256": "x" * 64}),
            "source/probe/code signature identity mismatch",
        ),
        (
            Path("manifest.json"),
            lambda payload: payload["configuration"].update({"rho_dense": 123}),
            "Manifest rho identity mismatch",
        ),
    ],
)
def test_analyzer_rejects_rho_probe_and_manifest_identity_tampering(
    tmp_path: Path,
    relative_path: Path,
    mutation,
    message: str,
) -> None:
    study_path, roots = _make_shards(tmp_path)
    cell_dir = (
        roots[0]
        / "surfaces"
        / analyzer._surface_specs()[0]["surface_id"]
        / "rho"
        / "cells"
        / "cell-000"
    )
    path = cell_dir / relative_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    _write_json(path, payload)
    if path.name == "manifest.json":
        result_path = cell_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["manifest_sha256"] = _sha256(path)
        _write_json(result_path, result)
        status_path = cell_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["result_sha256"] = _sha256(result_path)
        _write_json(status_path, status)

    with pytest.raises(ValueError, match=message):
        analyzer.analyze(
            study_config_path=study_path,
            shard_roots=roots,
            output_dir=tmp_path / "tampered-identity-analysis",
        )


def test_analyzer_rejects_probe_search_signature_resolved_mismatch(
    tmp_path: Path,
) -> None:
    study_path, roots = _make_shards(tmp_path)
    probe_path = (
        roots[0]
        / "surfaces"
        / analyzer._surface_specs()[0]["surface_id"]
        / "rho"
        / "probe.json"
    )
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe["search_signature"]["target"] = "trex"
    _write_json(probe_path, probe)

    with pytest.raises(ValueError, match="does not exactly match rho/resolved.json"):
        analyzer.analyze(
            study_config_path=study_path,
            shard_roots=roots,
            output_dir=tmp_path / "tampered-probe-analysis",
        )


def test_analyzer_rejects_result_changed_after_terminal_status(
    tmp_path: Path,
) -> None:
    study_path, roots = _make_shards(tmp_path)
    result_path = (
        roots[0]
        / "surfaces"
        / analyzer._surface_specs()[0]["surface_id"]
        / "rho"
        / "cells"
        / "cell-000"
        / "result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["benign_post_terminal_mutation"] = True
    _write_json(result_path, result)

    with pytest.raises(ValueError, match="does not bind result.json"):
        analyzer.analyze(
            study_config_path=study_path,
            shard_roots=roots,
            output_dir=tmp_path / "tampered-result-analysis",
        )


def test_canceled_jean_zay_job_inventory_is_zero_artifact_and_hash_bound(
    tmp_path: Path,
) -> None:
    study_path, _roots = _make_shards(tmp_path)
    source_contract = analyzer.load_contract(study_path)
    contract = analyzer.StudyContract(
        path=source_contract.path,
        payload=source_contract.payload,
        sha256=source_contract.sha256,
        study_id=analyzer.CANONICAL_STUDY_ID,
        initializer_sha256=source_contract.initializer_sha256,
        rho_conv=source_contract.rho_conv,
        rho_dense=source_contract.rho_dense,
        surfaces=source_contract.surfaces,
        weight_min=source_contract.weight_min,
        weight_max=source_contract.weight_max,
    )
    paths = _write_scheduler_provenance(tmp_path, contract.study_id)

    records = analyzer._canceled_job_inventory(tmp_path, contract)

    assert [row["job_id"] for row in records] == ["844032", "844396"]
    assert all(row["scientific_artifacts"] == 0 for row in records)
    assert all(row["source_result_jsons"] == [] for row in records)
    first_sources = {
        Path(row["path"]).name: row["sha256"]
        for row in records[0]["provenance_sources"]
    }
    assert first_sources == {
        name: _sha256(paths[name])
        for name in (
            "launch_plan.json",
            "jean_zay_submission.json",
            "recovery_submission.json",
            "replacement_plan.json",
            "jean_zay_terminal_accounting.json",
        )
    }


@pytest.mark.parametrize(
    ("filename", "field", "bad_value"),
    [
        ("launch_plan.json", "schema_version", "wrong/v1"),
        ("recovery_submission.json", "study_id", "wrong-study"),
        ("jean_zay_terminal_accounting.json", "schema_version", "wrong/v1"),
    ],
)
def test_canceled_job_inventory_rejects_provenance_schema_or_study_tamper(
    tmp_path: Path, filename: str, field: str, bad_value: str
) -> None:
    study_path, _roots = _make_shards(tmp_path)
    source_contract = analyzer.load_contract(study_path)
    contract = analyzer.StudyContract(
        path=source_contract.path,
        payload=source_contract.payload,
        sha256=source_contract.sha256,
        study_id=analyzer.CANONICAL_STUDY_ID,
        initializer_sha256=source_contract.initializer_sha256,
        rho_conv=source_contract.rho_conv,
        rho_dense=source_contract.rho_dense,
        surfaces=source_contract.surfaces,
        weight_min=source_contract.weight_min,
        weight_max=source_contract.weight_max,
    )
    paths = _write_scheduler_provenance(tmp_path, contract.study_id)
    payload = json.loads(paths[filename].read_text(encoding="utf-8"))
    payload[field] = bad_value
    _write_json(paths[filename], payload)

    with pytest.raises(ValueError, match="invalid schema/study identity"):
        analyzer._canceled_job_inventory(tmp_path, contract)


@pytest.mark.parametrize(
    ("missing_name", "message"),
    [
        ("launch_plan.json", "missing scheduler provenance"),
        ("replacement_plan.json", "missing scheduler provenance"),
        ("local_fallback_plan.json", "missing scheduler provenance"),
        ("jean_zay_terminal_accounting.json", "missing scheduler provenance"),
    ],
)
def test_canceled_job_inventory_requires_named_plan_provenance(
    tmp_path: Path, missing_name: str, message: str
) -> None:
    study_path, _roots = _make_shards(tmp_path)
    source_contract = analyzer.load_contract(study_path)
    contract = analyzer.StudyContract(
        path=source_contract.path,
        payload=source_contract.payload,
        sha256=source_contract.sha256,
        study_id=analyzer.CANONICAL_STUDY_ID,
        initializer_sha256=source_contract.initializer_sha256,
        rho_conv=source_contract.rho_conv,
        rho_dense=source_contract.rho_dense,
        surfaces=source_contract.surfaces,
        weight_min=source_contract.weight_min,
        weight_max=source_contract.weight_max,
    )
    paths = _write_scheduler_provenance(tmp_path, contract.study_id)
    paths[missing_name].unlink()

    with pytest.raises(ValueError, match=message):
        analyzer._canceled_job_inventory(tmp_path, contract)


@pytest.mark.parametrize(
    ("filename", "path", "bad_value", "message"),
    [
        (
            "recovery_submission.json",
            ("supersedes", "state"),
            "CANCELLED",
            "job 844032",
        ),
        (
            "local_fallback_submission.json",
            ("superseded_jean_zay_job", "terminal_state"),
            "CANCELLED+",
            "job 844396",
        ),
    ],
)
def test_canceled_job_inventory_requires_exact_terminal_states(
    tmp_path: Path,
    filename: str,
    path: tuple[str, str],
    bad_value: str,
    message: str,
) -> None:
    study_path, _roots = _make_shards(tmp_path)
    source_contract = analyzer.load_contract(study_path)
    contract = analyzer.StudyContract(
        path=source_contract.path,
        payload=source_contract.payload,
        sha256=source_contract.sha256,
        study_id=analyzer.CANONICAL_STUDY_ID,
        initializer_sha256=source_contract.initializer_sha256,
        rho_conv=source_contract.rho_conv,
        rho_dense=source_contract.rho_dense,
        surfaces=source_contract.surfaces,
        weight_min=source_contract.weight_min,
        weight_max=source_contract.weight_max,
    )
    paths = _write_scheduler_provenance(tmp_path, contract.study_id)
    payload = json.loads(paths[filename].read_text(encoding="utf-8"))
    payload[path[0]][path[1]] = bad_value
    _write_json(paths[filename], payload)

    with pytest.raises(ValueError, match=message):
        analyzer._canceled_job_inventory(tmp_path, contract)


def test_canceled_job_inventory_rejects_nonzero_terminal_accounting_runtime(
    tmp_path: Path,
) -> None:
    study_path, _roots = _make_shards(tmp_path)
    source_contract = analyzer.load_contract(study_path)
    contract = analyzer.StudyContract(
        path=source_contract.path,
        payload=source_contract.payload,
        sha256=source_contract.sha256,
        study_id=analyzer.CANONICAL_STUDY_ID,
        initializer_sha256=source_contract.initializer_sha256,
        rho_conv=source_contract.rho_conv,
        rho_dense=source_contract.rho_dense,
        surfaces=source_contract.surfaces,
        weight_min=source_contract.weight_min,
        weight_max=source_contract.weight_max,
    )
    paths = _write_scheduler_provenance(tmp_path, contract.study_id)
    path = paths["jean_zay_terminal_accounting.json"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["jobs"][1]["elapsed"] = "00:00:01"
    _write_json(path, payload)

    with pytest.raises(ValueError, match="zero-runtime pre-allocation"):
        analyzer._canceled_job_inventory(tmp_path, contract)


def test_local_receipts_fail_closed_on_missing_duplicate_target_or_surface(
    tmp_path: Path,
) -> None:
    study_path, roots = _make_shards(tmp_path, targets=LOCAL_TARGETS)
    contract = analyzer.load_contract(study_path)
    shards = analyzer._load_shards(roots, contract)
    assert len(
        analyzer._validate_receipts(shards, contract, receipt_roots=[])
    ) == 4

    receipt_path = roots[0] / "transport_receipt.json"
    original = json.loads(receipt_path.read_text(encoding="utf-8"))

    receipt_path.unlink()
    with pytest.raises(ValueError, match="found 0"):
        analyzer._validate_receipts(shards, contract, receipt_roots=[])
    _write_json(receipt_path, original)

    wrong_target = dict(original, target="trex")
    _write_json(receipt_path, wrong_target)
    with pytest.raises(ValueError, match="invalid target"):
        analyzer._validate_receipts(shards, contract, receipt_roots=[])
    _write_json(receipt_path, original)

    wrong_source = dict(original, source_archive_sha256="3" * 64)
    _write_json(receipt_path, wrong_source)
    with pytest.raises(ValueError, match="does not bind the shard's resolved source"):
        analyzer._validate_receipts(shards, contract, receipt_roots=[])
    _write_json(receipt_path, original)

    wrong_surface = dict(
        original,
        surface_id=analyzer._surface_specs()[1]["surface_id"],
    )
    _write_json(receipt_path, wrong_surface)
    with pytest.raises(ValueError, match="does not bind its containing shard surface"):
        analyzer._validate_receipts(shards, contract, receipt_roots=[])
    _write_json(receipt_path, original)

    duplicate_root = tmp_path / "duplicate-receipt"
    _write_json(duplicate_root / "transport_receipt.json", original)
    with pytest.raises(ValueError, match="found 2"):
        analyzer._validate_receipts(
            shards,
            contract,
            receipt_roots=[duplicate_root],
        )


def test_correlation_statistics_return_null_when_undefined() -> None:
    assert analyzer._correlation_statistics([1.0, 1.0], [2.0, 3.0]) == {
        "sample_count": 2,
        "pearson": None,
        "spearman": None,
    }
    assert analyzer._correlation_statistics([], []) == {
        "sample_count": 0,
        "pearson": None,
        "spearman": None,
    }


def _range_rows(best_index: int) -> list[dict]:
    conv_axis, dense_axis = analyzer._expected_axes()
    rows = []
    for index in range(64):
        conv_index, dense_index = divmod(index, 8)
        accuracy = 0.9 if index == best_index else 0.5
        rows.append(
            {
                "cell_index": index,
                "conv_axis_index": conv_index,
                "dense_axis_index": dense_index,
                "rho_conv": conv_axis[conv_index],
                "rho_dense": dense_axis[dense_index],
                "canonical_outcome": "numeric_complete",
                "final_validation_accuracy": accuracy,
            }
        )
    return rows


@pytest.mark.parametrize(
    ("best_index", "open_directions"),
    [
        (0, ["lower_rho_conv", "lower_rho_dense"]),
        (63, ["upper_rho_conv", "upper_rho_dense"]),
        (31, ["upper_rho_dense"]),
    ],
)
def test_range_evidence_marks_boundary_maxima_open(
    best_index: int, open_directions: list[str]
) -> None:
    rows = _range_rows(best_index)

    evidence = analyzer._range_evidence(rows[best_index], rows)

    assert evidence["range_status"] == "open_boundary"
    assert evidence["open_directions"] == open_directions
    assert evidence["locally_bracketed_by_accuracy"] is False
    assert set(evidence["axial_neighbors"]) == {
        "lower_rho_conv",
        "upper_rho_conv",
        "lower_rho_dense",
        "upper_rho_dense",
    }
    assert all(
        evidence["axial_neighbors"][direction]["grid_state"]
        == "outside_tested_range"
        for direction in open_directions
    )


def test_range_evidence_requires_four_strictly_lower_numeric_axial_neighbors() -> None:
    rows = _range_rows(27)

    evidence = analyzer._range_evidence(rows[27], rows)

    assert evidence["range_status"] == "locally_bracketed_by_accuracy"
    assert evidence["locally_bracketed_by_accuracy"] is True
    assert evidence["open_directions"] == []
    assert evidence["nonnumeric_axial_directions"] == []
    assert evidence["improving_or_tied_directions"] == []
    assert {
        direction: row["cell_index"]
        for direction, row in evidence["axial_neighbors"].items()
    } == {
        "lower_rho_conv": 19,
        "upper_rho_conv": 35,
        "lower_rho_dense": 26,
        "upper_rho_dense": 28,
    }
    assert {
        row["accuracy_relation_to_best"]
        for row in evidence["axial_neighbors"].values()
    } == {"strictly_lower"}


def test_range_evidence_is_unresolved_with_nonnumeric_interior_neighbor() -> None:
    rows = _range_rows(27)
    rows[28]["canonical_outcome"] = "nonfinite"
    rows[28]["final_validation_accuracy"] = None

    evidence = analyzer._range_evidence(rows[27], rows)

    assert evidence["range_status"] == "unresolved_nonnumeric_axial_neighbor"
    assert evidence["nonnumeric_axial_directions"] == ["upper_rho_dense"]
    assert evidence["locally_bracketed_by_accuracy"] is False
    assert evidence["axial_neighbors"]["upper_rho_dense"] == {
        "direction": "upper_rho_dense",
        "axis": "rho_dense",
        "grid_state": "observed",
        "cell_index": 28,
        "conv_axis_index": 3,
        "dense_axis_index": 4,
        "rho_conv": pytest.approx(rows[28]["rho_conv"]),
        "rho_dense": pytest.approx(rows[28]["rho_dense"]),
        "canonical_outcome": "nonfinite",
        "final_validation_accuracy": None,
        "final_validation_accuracy_percent": None,
        "accuracy_relation_to_best": "nonnumeric",
    }


def test_range_evidence_tied_axial_accuracy_is_not_bracketed() -> None:
    rows = _range_rows(27)
    rows[28]["final_validation_accuracy"] = rows[27]["final_validation_accuracy"]

    evidence = analyzer._range_evidence(rows[27], rows)

    assert evidence["range_status"] == "not_locally_bracketed"
    assert evidence["improving_or_tied_directions"] == ["upper_rho_dense"]
    assert evidence["axial_neighbors"]["upper_rho_dense"][
        "accuracy_relation_to_best"
    ] == "tied"
    assert evidence["locally_bracketed_by_accuracy"] is False


def test_co_maximum_range_evidence_is_conservative_when_boundary_ties_interior() -> None:
    rows = _range_rows(27)
    rows[63]["final_validation_accuracy"] = rows[27]["final_validation_accuracy"]
    for row in rows:
        row["final_validation_accuracy_percent"] = (
            100.0 * row["final_validation_accuracy"]
        )

    evidence = analyzer._co_maximum_range_evidence(
        selected_best=rows[27],
        co_maxima=[rows[27], rows[63]],
        rows=rows,
    )

    assert evidence["range_status"] == "open_boundary"
    assert evidence["locally_bracketed_by_accuracy"] is False
    assert evidence["selected_best_range_evidence"]["range_status"] == (
        "locally_bracketed_by_accuracy"
    )
    assert [
        (row["cell_index"], row["range_evidence"]["range_status"])
        for row in evidence["maximum_accuracy_co_maxima"]
    ] == [
        (27, "locally_bracketed_by_accuracy"),
        (63, "open_boundary"),
    ]


def test_analyzer_excludes_missing_projection_measurements_without_failing(
    tmp_path: Path,
) -> None:
    study_path, roots = _make_shards(tmp_path)
    surface = analyzer._surface_specs()[0]
    diagnostics_path = (
        roots[0]
        / "surfaces"
        / surface["surface_id"]
        / "rho"
        / "cells"
        / "cell-063"
        / "safety_diagnostics.json"
    )
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["median_projection_efficiency"] = None
    diagnostics["median_projection_efficiency_by_parameter"] = {
        name: None for name in analyzer.WEIGHT_NAMES
    }
    _write_json(diagnostics_path, diagnostics)

    output = tmp_path / "missing-projection-analysis"
    report = analyzer.analyze(
        study_config_path=study_path,
        shard_roots=roots,
        output_dir=output,
    )

    first_surface = report["surfaces"][0]
    assert first_surface["best_observed_cell_index"] == 63
    assert first_surface["best_observed_median_projection_efficiency"] is None
    assert first_surface[
        "accuracy_vs_median_projection_efficiency_sample_count"
    ] == 63
    assert "unavailable" in (output / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("raw_status", "failure", "state", "result_present", "expected"),
    [
        ("complete", None, "complete", True, ("numeric_complete", None)),
        (
            "candidate_rejected_nonfinite",
            {"kind": "nonfinite_training_value"},
            "failed",
            False,
            ("nonfinite", "nonfinite_training_value"),
        ),
        (
            "candidate_rejected_safety",
            {"kind": "nonfinite_diagnostic"},
            "failed",
            False,
            ("nonfinite", "nonfinite_diagnostic"),
        ),
    ],
)
def test_status_classifier_accepts_only_canonical_numeric_or_nonfinite(
    raw_status: str,
    failure: dict | None,
    state: str,
    result_present: bool,
    expected: tuple[str, str | None],
) -> None:
    assert analyzer.classify_cell_status(
        raw_status,
        failure,
        reporting_state=state,
        result_present=result_present,
    ) == expected


def test_status_classifier_rejects_disabled_safety_outcomes() -> None:
    with pytest.raises(ValueError, match="requires non-finite kind"):
        analyzer.classify_cell_status(
            "candidate_rejected_safety",
            {"kind": "gradient_rms_explosion"},
            reporting_state="failed",
            result_present=False,
        )
