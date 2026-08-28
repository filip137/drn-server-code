from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from experiments.artifacts import atomic_write_json, content_hash, sha256_file
from experiments.mnist_relu_drn.analyze_ibm_om_baseline_selection import (
    ANALYSIS_SCHEMA,
    BaselineSelectionAnalysisError,
    _torch_hash,
    analyze_study,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    BASELINE_POLICIES,
    LAYOUTS,
    build_physical_mapping,
    build_weight_error_decomposition,
    fit_weight_reconstruction_scales,
    hardware_instance_fingerprint,
    save_physical_mapping,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection_config import (
    CONTINUOUS_METRIC_DEFINITION,
    DEVELOPMENT_ASSIGNMENT_SEED,
    EXPECTED_WEIGHTS_SHA256,
    HELDOUT_ASSIGNMENT_SEEDS,
    STANDARD4DELTA_METRIC_DEFINITION,
    parse_baseline_selection_config,
    resolve_baseline_selection_spec,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection_runtime import (
    JOINT_SCHEMA,
    JOINT_SCHEMA_VERSION,
    PREDICTION_SCHEMA,
    PREDICTION_SCHEMA_VERSION,
    SUMMARY_SCHEMA,
    SUMMARY_SCHEMA_VERSION,
)
from experiments.schema import RunMode, to_plain_data
from experiments.study_workflow import prepare_study
from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    _native_seed,
    _population_fingerprint,
    save_om_array_population,
)
from training.ibm_reram_program_verify import OM_PRESET, derive_seed


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "studies/mnist-ibm-om-four-device-baseline-selection-20260828-v1.json"
ARM_POLICY = {
    "independent-cell-reset-mean": "independent_cell_reset_mean",
    "shared-quad-reset-max": "shared_quad_reset_max",
    "shared-destination-columns-reset-max": "shared_destination_columns_reset_max",
    "reference-enforced-destination-columns": "reference_enforced_destination_columns",
}
CORRECT = {
    "independent_cell_reset_mean": 8800,
    "shared_quad_reset_max": 9000,
    "shared_destination_columns_reset_max": 9300,
    "reference_enforced_destination_columns": 9100,
}
IDENTITY_FIELDS = (
    "max_bound",
    "min_bound",
    "dwmin_up",
    "dwmin_down",
    "reference",
    "corrupt",
    "published_corrupt",
)


def _npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def _joint(
    path: Path, *, seed: int, role: str, base_fingerprint: str
) -> tuple[str, dict[str, Any]]:
    shape = (2, 2)
    layer_values: dict[str, np.ndarray] = {}
    fingerprint_tensors: dict[str, torch.Tensor] = {}
    for layer in range(2):
        values = {
            "max_bound": np.ones(shape, dtype=np.float32),
            "min_bound": -np.ones(shape, dtype=np.float32),
            "dwmin_up": np.full(shape, 0.02, dtype=np.float32),
            "dwmin_down": np.full(shape, 0.02, dtype=np.float32),
            "reference": np.full(shape, -0.6, dtype=np.float32),
            "corrupt": np.zeros(shape, dtype=np.bool_),
            "published_corrupt": np.zeros(shape, dtype=np.bool_),
            "reset_mean_raw_a": np.full(shape, -0.6, dtype=np.float32),
            "reset_standard_error_raw_a": np.zeros(shape, dtype=np.float32),
        }
        for name, value in values.items():
            key = f"layer_{layer}_{name}"
            layer_values[key] = value
            fingerprint_tensors[key] = torch.from_numpy(value)
        layer_values[f"layer_{layer}_selected_attempt_grid"] = np.full(
            (1, 1), -1, dtype=np.int64
        )
    metadata = {
        "schema": JOINT_SCHEMA,
        "schema_version": JOINT_SCHEMA_VERSION,
        "assignment_seed": seed,
        "assignment_role": role,
        "base_population_fingerprint": base_fingerprint,
        "donor_population_fingerprint": None,
        "binding_keys": ["layer0", "layer1"],
        "binding_shapes": [[2, 2], [2, 2]],
        "layouts": list(LAYOUTS),
        "read_samples": 8,
        "maximum_attempts": 128,
        "selected_attempts": [],
        "failure_layer_index": [],
        "failure_flat_index": [],
    }
    hardware_id = hardware_instance_fingerprint(metadata, fingerprint_tensors)
    arrays = {
        "schema": np.asarray(JOINT_SCHEMA),
        "schema_version": np.asarray(JOINT_SCHEMA_VERSION, dtype=np.int64),
        "assignment_seed": np.asarray(seed, dtype=np.int64),
        "hardware_instance_id": np.asarray(hardware_id),
        "base_population_fingerprint": np.asarray(base_fingerprint),
        "donor_population_fingerprint": np.asarray("none"),
        "maximum_attempts": np.asarray(128, dtype=np.int64),
        "failure_layer_index": np.empty((0,), dtype=np.int64),
        "failure_flat_index": np.empty((0,), dtype=np.int64),
        "selected_attempt": np.empty((0,), dtype=np.int64),
        **layer_values,
    }
    _npz(path, arrays)
    report = {
        **metadata,
        "hardware_instance_id": hardware_id,
        "failure_count": 0,
        "failure_count_by_layer": [0, 0],
        "final_corrupt_count": 0,
        "final_corrupt_count_by_layer": [0, 0],
        "published_corrupt_count": 0,
        "published_corrupt_count_by_layer": [0, 0],
        "selected_attempt": {"minimum": None, "maximum": None, "mean": None},
        "final_policy_coverage": {"feasible_quads": [1, 1], "total_quads": [1, 1]},
        "commissioned_reset_mean_raw_a": [],
        "bounded_reset_baseline_unit": [],
        "reference_outside_public_coordinate_count": [0, 0],
    }
    receipt = {
        "schema": JOINT_SCHEMA,
        "schema_version": JOINT_SCHEMA_VERSION,
        "assignment_seed": seed,
        "hardware_instance_id": hardware_id,
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "report": report,
        "tensor_hashes": {
            name: sha256(value.tobytes(order="C")).hexdigest()
            for name, value in arrays.items()
            if value.ndim > 0
        },
    }
    atomic_write_json(path.with_suffix(".receipt.json"), receipt)
    return hardware_id, report


def _base_sources(
    run_dir: Path, *, role: str, seed: int
) -> tuple[dict[str, Any], dict[str, Any], tuple[tuple[Path, str], ...]]:
    keys = ("layer0", "layer1")
    shapes = ((2, 2), (2, 2))
    base_seeds = tuple(
        _native_seed(derive_seed(seed, OM_PRESET, key, "published")) for key in keys
    )
    donor_seeds = tuple(
        _native_seed(derive_seed(seed, OM_PRESET, key, "repair_donor")) for key in keys
    )
    size = 8
    tensors = {
        "max_bound": torch.ones(size, dtype=torch.float32),
        "min_bound": -torch.ones(size, dtype=torch.float32),
        "dwmin_up": torch.full((size,), 0.02, dtype=torch.float32),
        "dwmin_down": torch.full((size,), 0.02, dtype=torch.float32),
        "reference": torch.full((size,), -0.6, dtype=torch.float32),
        "corrupt": torch.zeros(size, dtype=torch.bool),
        "published_corrupt": torch.zeros(size, dtype=torch.bool),
    }
    scalar = {
        "aihwkit_version": "1.1.0",
        "nominal_dw_min": 0.02,
        "dw_min_std": 0.3,
        "write_noise_std": 0.1,
    }
    fingerprint = _population_fingerprint(
        assignment_seed=seed,
        corruption_policy="counterfactual_repaired",
        keys=keys,
        shapes=shapes,
        binding_sampling_seeds=base_seeds,
        donor_sampling_seeds=donor_seeds,
        scalar_parameters=scalar,
        tensors=tensors,
    )
    population = IbmReramArrayPopulation(
        assignment_seed=seed,
        corruption_policy="counterfactual_repaired",
        binding_keys=keys,
        binding_shapes=shapes,
        binding_sampling_seeds=base_seeds,
        donor_sampling_seeds=donor_seeds,
        nominal_dw_min=0.02,
        dw_min_std=0.3,
        write_noise_std=0.1,
        fingerprint=fingerprint,
        aihwkit_version="1.1.0",
        **tensors,
    )
    stem = f"4-device-assignment-{seed}"
    base = run_dir / "artifacts" / role / "base"
    population_path = base / "populations" / f"{stem}.npz"
    population_path.parent.mkdir(parents=True, exist_ok=True)
    save_om_array_population(population_path, population)
    request = {
        "preset": OM_PRESET,
        "assignment_seed": seed,
        "corruption_policy": "counterfactual_repaired",
        "binding_keys": list(keys),
        "binding_shapes": [list(shape) for shape in shapes],
        "required_aihwkit_version": "1.1.0",
    }
    population_receipt_path = population_path.with_suffix(".receipt.json")
    atomic_write_json(
        population_receipt_path,
        {
            "schema": "ebl.ibm_reram.om_array_population_receipt",
            "schema_version": 1,
            "backend": "external_pinned_aihwkit_python",
            "python_executable": "/pinned/python",
            "python_version": "3.10",
            "torch_version": "2.0",
            "aihwkit_version": "1.1.0",
            "request": request,
            "num_cells": size,
            "population_fingerprint": fingerprint,
            "population_sha256": sha256_file(population_path),
            "sampler_source_sha256": sha256_file(
                ROOT / "experiments/reram_program_verify/hwa_population_sampler.py"
            ),
            "population_implementation_sha256": sha256_file(
                ROOT / "training/ibm_reram_hwa.py"
            ),
        },
    )
    population_summary = {
        "assignment_seed": seed,
        "sha256": sha256_file(population_path),
        "receipt_sha256": sha256_file(population_receipt_path),
        "population_fingerprint": fingerprint,
        "cells": size,
        "noise_partition": {
            "commissioning_cycle_to_cycle_noise": "enabled",
            "commissioning_apparent_write_noise": "enabled",
            "mapped_target_write_noise": "disabled",
            "declared_dw_min_std": 0.3,
            "declared_write_noise_std": 0.1,
        },
    }
    commissioning_path = base / "commissioning" / f"{stem}.npz"
    commissioning_seed = seed + 123
    mean = np.full((size,), -0.6, dtype=np.float32)
    se = np.zeros((size,), dtype=np.float32)
    _npz(
        commissioning_path,
        {
            "schema": np.asarray("ebl.mnist_relu_drn.ibm_om_per_cell_raw_reset_commissioning"),
            "schema_version": np.asarray(1, dtype=np.int64),
            "population_fingerprint": np.asarray(fingerprint),
            "assignment_seed": np.asarray(seed, dtype=np.int64),
            "commissioning_seed": np.asarray(commissioning_seed, dtype=np.int64),
            "read_samples": np.asarray(8, dtype=np.int64),
            "binding_keys_json": np.asarray(json.dumps(list(keys), separators=(",", ":"))),
            "binding_shapes_json": np.asarray(json.dumps([list(shape) for shape in shapes], separators=(",", ":"))),
            "reset_mean_raw_a": mean,
            "reset_standard_error_raw_a": se,
        },
    )
    commissioning_receipt_path = commissioning_path.with_suffix(".receipt.json")
    commissioning_core = {
        "schema": "ebl.mnist_relu_drn.ibm_om_per_cell_raw_reset_commissioning",
        "schema_version": 1,
        "algorithm": "sequential_reset_pulse_read_per_cell_mean_raw_a",
        "population_fingerprint": fingerprint,
        "assignment_seed": seed,
        "topology": 4,
        "commissioning_seed": commissioning_seed,
        "read_samples": 8,
        "reset_pulses_per_cell": 8,
        "total_reset_pulses": size * 8,
        "baseline_estimator": "per_cell_arithmetic_mean",
        "cross_cell_pooling": "none",
        "standard_error_guard": 0.0,
        "read_coordinate": "apparent_raw_active_a",
        "reference_consumed_by_commissioner": False,
        "artifact": commissioning_path.name,
        "artifact_sha256": sha256_file(commissioning_path),
        "reset_mean_raw_a_sha256": _torch_hash(mean),
        "reset_standard_error_raw_a_sha256": _torch_hash(se),
    }
    atomic_write_json(
        commissioning_receipt_path,
        {**commissioning_core, "commissioner_report": {}},
    )
    commissioning_summary = {
        "sha256": sha256_file(commissioning_path),
        "receipt_sha256": sha256_file(commissioning_receipt_path),
        **commissioning_core,
    }
    artifacts = (
        (population_path, "ibm_om_base_population"),
        (population_receipt_path, "ibm_om_base_population_receipt"),
        (commissioning_path, "ibm_om_base_reset_commissioning"),
        (commissioning_receipt_path, "ibm_om_base_reset_commissioning_receipt"),
    )
    return population_summary, commissioning_summary, artifacts


def _prediction_metric(
    *, correct: int, teacher_correct: int, agreement: int, digest: str
) -> dict[str, Any]:
    return {
        "student_correct": correct,
        "examples": 10_000,
        "student_accuracy": correct / 10_000,
        "teacher_correct": teacher_correct,
        "teacher_accuracy": teacher_correct / 10_000,
        "teacher_agreement_count": agreement,
        "teacher_agreement": agreement / 10_000,
        "prediction_sha256": digest,
    }


def _predictions(
    path: Path, *, seed: int, policy: str, continuous_correct: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    labels = np.arange(10_000, dtype=np.int64) % 10
    teacher = labels.copy()
    continuous = labels.copy()
    standard = labels.copy()
    continuous[continuous_correct:] = (continuous[continuous_correct:] + 1) % 10
    standard_correct = continuous_correct - 200
    standard[standard_correct:] = (standard[standard_correct:] + 1) % 10
    arrays = {
        "schema": np.asarray(PREDICTION_SCHEMA),
        "schema_version": np.asarray(PREDICTION_SCHEMA_VERSION, dtype=np.int64),
        "assignment_seed": np.asarray(seed, dtype=np.int64),
        "policy": np.asarray(policy),
        "labels": labels,
        "teacher_prediction": teacher,
        "continuous_prediction": continuous,
        "standard4delta_prediction": standard,
    }
    _npz(path, arrays)
    receipt = {
        "schema": PREDICTION_SCHEMA,
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "assignment_seed": seed,
        "policy": policy,
        "examples": 10_000,
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "correct": {
            "teacher": 10_000,
            "continuous": continuous_correct,
            "standard4delta": standard_correct,
        },
        "labels_sha256": _torch_hash(labels),
        "prediction_hashes": {
            "teacher": _torch_hash(teacher),
            "continuous": _torch_hash(continuous),
            "standard4delta": _torch_hash(standard),
        },
    }
    atomic_write_json(path.with_suffix(".receipt.json"), receipt)
    continuous_metric = _prediction_metric(
        correct=continuous_correct,
        teacher_correct=10_000,
        agreement=continuous_correct,
        digest=receipt["prediction_hashes"]["continuous"],
    )
    standard_metric = _prediction_metric(
        correct=standard_correct,
        teacher_correct=10_000,
        agreement=standard_correct,
        digest=receipt["prediction_hashes"]["standard4delta"],
    )
    return receipt, continuous_metric, standard_metric


def _weight_errors(
    path: Path,
    *,
    seed: int,
    policy: str,
    logical_weights: tuple[torch.Tensor, ...],
    mapping,
    scales: tuple[float, ...],
) -> list[dict[str, Any]]:
    errors = build_weight_error_decomposition(logical_weights, mapping, scales)
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray("ebl.mnist_relu_drn.ibm_om_baseline_weight_errors"),
        "schema_version": np.asarray(1, dtype=np.int64),
        "assignment_seed": np.asarray(seed, dtype=np.int64),
        "policy": np.asarray(policy),
    }
    for error in errors:
        for field in fields(error):
            value = getattr(error, field.name)
            if isinstance(value, torch.Tensor):
                arrays[f"layer_{error.layer_index}_{field.name}"] = value.numpy()
    _npz(path, arrays)
    reports = [dict(error.report) for error in errors]
    atomic_write_json(
        path.with_suffix(".receipt.json"),
        {
            "schema": "ebl.mnist_relu_drn.ibm_om_baseline_weight_errors",
            "schema_version": 1,
            "assignment_seed": seed,
            "policy": policy,
            "artifact": path.name,
            "artifact_sha256": sha256_file(path),
            "analysis_scales": list(scales),
            "layers": reports,
        },
    )
    return reports


def _artifact(run_dir: Path, path: Path, kind: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "kind": kind,
    }


def _one_run(
    study_dir: Path,
    study: dict[str, Any],
    arm: dict[str, Any],
    config_record: dict[str, Any],
) -> Path:
    policy = ARM_POLICY[arm["arm_id"]]
    payload = json.loads(Path(config_record["resolved_path"]).read_text(encoding="utf-8"))
    document = parse_baseline_selection_config(payload)
    spec = resolve_baseline_selection_spec(document, RunMode.VALIDATE)
    resolved = to_plain_data(spec)
    seed = spec.protocol.assignments.heldout_seed
    run_dir = study_dir / "runs" / arm["arm_id"] / f"run-{seed}"
    (run_dir / "artifacts/development").mkdir(parents=True)
    (run_dir / "artifacts/heldout").mkdir(parents=True)
    dev_population, dev_commissioning, dev_source_artifacts = _base_sources(
        run_dir, role="development", seed=DEVELOPMENT_ASSIGNMENT_SEED
    )
    held_population, held_commissioning, held_source_artifacts = _base_sources(
        run_dir, role="heldout", seed=seed
    )
    dev_id, dev_report = _joint(
        run_dir / "artifacts/development/joint_assignment.npz",
        seed=DEVELOPMENT_ASSIGNMENT_SEED,
        role="development",
        base_fingerprint=dev_population["population_fingerprint"],
    )
    held_id, held_report = _joint(
        run_dir / "artifacts/heldout/joint_assignment.npz",
        seed=seed,
        role="heldout",
        base_fingerprint=held_population["population_fingerprint"],
    )
    logical = (torch.tensor([[0.8]], dtype=torch.float32), torch.tensor([[-0.6]], dtype=torch.float32))
    lower = (torch.zeros((2, 2)), torch.zeros((2, 2)))
    upper = (torch.ones((2, 2)), torch.ones((2, 2)))
    reset = (torch.full((2, 2), 0.2), torch.full((2, 2), 0.2))
    reference = (torch.full((2, 2), -0.6), torch.full((2, 2), -0.6))
    kwargs = {
        "policy": policy,
        "scale_fractions": (0.5, 0.5),
        "nominal_dw_min": 0.02,
        "conductance_min": 0.0,
        "conductance_max": 0.00011,
        "cell_lower_units": lower,
        "cell_upper_units": upper,
        "reset_baseline_units": reset,
        "intrinsic_references_native": reference,
    }
    dev_mapping = build_physical_mapping(logical, **kwargs)
    held_mapping = build_physical_mapping(logical, **kwargs)
    dev_map_artifact = save_physical_mapping(
        run_dir / "artifacts/development/selected_physical_mapping.npz",
        dev_mapping,
        assignment_seed=DEVELOPMENT_ASSIGNMENT_SEED,
        assignment_role="development",
        hardware_instance_id=dev_id,
    )
    save_physical_mapping(
        run_dir / "artifacts/heldout/physical_mapping.npz",
        held_mapping,
        assignment_seed=seed,
        assignment_role="heldout",
        hardware_instance_id=held_id,
    )
    scales = fit_weight_reconstruction_scales(logical, dev_mapping)
    candidate = {
        "scale_fractions": [0.5, 0.5],
        "calibration": {
            "gain": 1.0,
            "student_correct": 900,
            "calibrated_kl": 0.1,
            "test_labels_used": False,
        },
        "target_hashes": [layer["hashes"]["continuous_conductance"] for layer in dev_mapping.report["layers"]],
    }
    calibration = {
        "schema": "ebl.mnist_relu_drn.ibm_om_baseline_development_calibration",
        "schema_version": 1,
        "policy": policy,
        "assignment_seed": DEVELOPMENT_ASSIGNMENT_SEED,
        "hardware_instance_id": dev_id,
        "selection_domain": spec.protocol.mapping.selection_domain,
        "selection_metric": spec.protocol.mapping.selection_metric,
        "candidate_count": 16,
        "candidates": [candidate for _ in range(16)],
        "selected_index": 0,
        "selected": candidate,
        "analysis_scales": list(scales),
        "calibration_labels_sha256": "3" * 64,
        "calibration_indices_sha256": "4" * 64,
        "test_labels_used": False,
        "continuous_mapping_artifact_sha256": dev_map_artifact["sha256"],
    }
    calibration_path = run_dir / "artifacts/development/calibration.json"
    atomic_write_json(calibration_path, calibration)
    prediction_receipt, continuous, standard = _predictions(
        run_dir / "artifacts/heldout/predictions.npz",
        seed=seed,
        policy=policy,
        continuous_correct=CORRECT[policy],
    )
    weight_reports = _weight_errors(
        run_dir / "artifacts/heldout/weight_errors.npz",
        seed=seed,
        policy=policy,
        logical_weights=logical,
        mapping=held_mapping,
        scales=scales,
    )
    metric_ids = {
        "primary": CONTINUOUS_METRIC_DEFINITION,
        "diagnostic": STANDARD4DELTA_METRIC_DEFINITION,
    }
    data_provenance = {
        "dataset": "MNIST",
        "test_examples": 10_000,
        "test_data_sha256": "1" * 64,
        "test_targets_sha256": prediction_receipt["labels_sha256"],
        "training_examples": 60_000,
        "training_data_sha256": "7" * 64,
        "training_targets_sha256": "8" * 64,
        "calibration_examples": 1024,
        "calibration_indices_sha256": "4" * 64,
        "calibration_raw_images_sha256": "9" * 64,
        "calibration_raw_targets_sha256": "3" * 64,
        "calibration_labels_sha256": "3" * 64,
        "test_split_labels_used_for_mapping_or_calibration": False,
    }
    residual_arrays = {
        "schema": np.asarray(
            "ebl.mnist_relu_drn.ibm_om_baseline_rail_voltage_residual"
        ),
        "schema_version": np.asarray(1, dtype=np.int64),
        "examples": np.asarray(10_000, dtype=np.int64),
        **{
            f"layer_{layer}_rail_voltage_residual": np.zeros(
                (10_000, int(logical[layer].shape[1])), dtype=np.float32
            )
            for layer in range(2)
        },
    }
    baseline_residual = {
        "definition": "paired_rail_voltage_difference_with_all_offsets_d_equal_zero",
        "examples": 10_000,
        "layers": [
            {
                "layer": layer,
                "layout": LAYOUTS[layer],
                "values": 10_000 * int(logical[layer].shape[1]),
                "signed_mean": 0.0,
                "rms": 0.0,
                "p95_absolute": 0.0,
                "maximum_absolute": 0.0,
            }
            for layer in range(2)
        ],
        "residual_hashes": [
            _torch_hash(residual_arrays[f"layer_{layer}_rail_voltage_residual"])
            for layer in range(2)
        ],
    }
    baseline_residual_path = run_dir / "artifacts/heldout/baseline_rail_residual.npz"
    _npz(baseline_residual_path, residual_arrays)
    atomic_write_json(
        baseline_residual_path.with_suffix(".receipt.json"),
        {
            "schema": "ebl.mnist_relu_drn.ibm_om_baseline_rail_voltage_residual",
            "schema_version": 1,
            "artifact": baseline_residual_path.name,
            "artifact_sha256": sha256_file(baseline_residual_path),
            "report": baseline_residual,
            "tensor_hashes": {
                f"layer_{layer}_rail_voltage_residual": baseline_residual[
                    "residual_hashes"
                ][layer]
                for layer in range(2)
            },
        },
    )
    validity_gates = {
        "development_joint_policy_coverage_complete": True,
        "heldout_joint_policy_coverage_complete": True,
        "development_repaired_population_has_no_corrupt_cells": True,
        "heldout_repaired_population_has_no_corrupt_cells": True,
        "development_full_conductance_invariants": True,
        "heldout_full_conductance_invariants": True,
        "development_candidate_count_is_16": True,
        "standard4delta_uses_continuous_selected_calibration": True,
        "expected_evaluation_examples": True,
        "optimizer_updates_are_zero": True,
        "target_write_noise_is_zero": True,
        "inference_read_noise_is_zero": True,
    }
    observed_examples = {
        "continuous": 10_000,
        "standard4delta": 10_000,
        "baseline_only": 10_000,
        "baseline_rail_residual": 10_000,
        "predictions": 10_000,
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "complete",
        "claim_boundary": "test fixture",
        "metric_definition_ids": metric_ids,
        "policy": policy,
        "source": {
            "role": "frozen_relu_source_and_teacher",
            "path": "/frozen/teacher.pt",
            "sha256": EXPECTED_WEIGHTS_SHA256,
            "metadata": {},
            "logical_weight_hashes": ["logical-layer-0", "logical-layer-1"],
            "optimizer_updates": 0,
        },
        "data": data_provenance,
        "contract": resolved["protocol"],
        "development": {
            "hardware": {
                "hardware_instance_id": dev_id,
                "repair": dev_report,
                "nominal_dw_min": 0.02,
                "population": dev_population,
                "base_commissioning": dev_commissioning,
                "donor_population": None,
                "donor_commissioning": None,
            },
            "calibration": calibration,
            "mapping": dev_mapping.report,
            "invariants": {"valid": True},
        },
        "heldout": {
            "assignment_seed": seed,
            "hardware": {
                "hardware_instance_id": held_id,
                "repair": held_report,
                "nominal_dw_min": 0.02,
                "population": held_population,
                "base_commissioning": held_commissioning,
                "donor_population": None,
                "donor_commissioning": None,
            },
            "scale_fractions": [0.5, 0.5],
            "fixed_logit_gain": 1.0,
            "mapping": held_mapping.report,
            "invariants": {"valid": True},
            "ideal_bounded_continuous_init": continuous,
            "ideal_bounded_standard4delta_init": standard,
            "baseline_only_full_circuit": {"examples": 10_000},
            "baseline_rail_voltage_residual": baseline_residual,
            "weight_errors": weight_reports,
            "predictions": prediction_receipt,
        },
        "validity": {
            "coverage_valid": True,
            "gates": validity_gates,
            "expected_examples": 10_000,
            "observed_examples": observed_examples,
            "optimizer_updates": 0,
            "program_verify_enabled": False,
            "hardware_aware_training_enabled": False,
            "target_write_noise": 0.0,
            "inference_read_noise": 0.0,
            "retention_or_drift": False,
            "noise_partition": {
                "commissioning_reset_pulse_and_apparent_read_noise": "enabled_and_frozen",
                "mapped_target_write_noise": "disabled",
                "inference_read_noise": "disabled",
            },
        },
    }
    summary_path = run_dir / "artifacts/scientific_summary.json"
    atomic_write_json(summary_path, summary)
    artifact_specs = (
        *dev_source_artifacts,
        *held_source_artifacts,
        (run_dir / "artifacts/development/joint_assignment.npz", "ibm_om_joint_assignment"),
        (run_dir / "artifacts/development/joint_assignment.receipt.json", "ibm_om_joint_assignment_receipt"),
        (run_dir / "artifacts/heldout/joint_assignment.npz", "ibm_om_joint_assignment"),
        (run_dir / "artifacts/heldout/joint_assignment.receipt.json", "ibm_om_joint_assignment_receipt"),
        (run_dir / "artifacts/development/selected_physical_mapping.npz", "ibm_om_physical_mapping"),
        (run_dir / "artifacts/development/selected_physical_mapping.receipt.json", "ibm_om_physical_mapping_receipt"),
        (run_dir / "artifacts/heldout/physical_mapping.npz", "ibm_om_physical_mapping"),
        (run_dir / "artifacts/heldout/physical_mapping.receipt.json", "ibm_om_physical_mapping_receipt"),
        (calibration_path, "development_calibration"),
        (run_dir / "artifacts/heldout/predictions.npz", "predictions"),
        (run_dir / "artifacts/heldout/predictions.receipt.json", "predictions_receipt"),
        (run_dir / "artifacts/heldout/weight_errors.npz", "weight_error_decomposition"),
        (run_dir / "artifacts/heldout/weight_errors.receipt.json", "weight_error_decomposition_receipt"),
        (baseline_residual_path, "baseline_rail_residual"),
        (baseline_residual_path.with_suffix(".receipt.json"), "baseline_rail_residual_receipt"),
        (summary_path, "scientific_summary"),
    )
    artifacts = [_artifact(run_dir, path, kind) for path, kind in artifact_specs]
    terminal_keys = (
        "student_correct", "examples", "student_accuracy", "teacher_correct",
        "teacher_accuracy", "teacher_agreement_count", "teacher_agreement", "prediction_sha256",
    )
    metrics = {
        "metric_definition_ids": metric_ids,
        "policy": policy,
        "heldout_assignment_seed": seed,
        "development_hardware_instance_id": dev_id,
        "heldout_hardware_instance_id": held_id,
        "selected_scale_fractions": [0.5, 0.5],
        "fixed_logit_gain": 1.0,
        "ideal_bounded_continuous_init": {key: continuous[key] for key in terminal_keys},
        "ideal_bounded_standard4delta_init": {key: standard[key] for key in terminal_keys},
        "coverage_valid": True,
        "validity_gates": validity_gates,
        "expected_examples": 10_000,
        "optimizer_updates": 0,
    }
    run_id = run_dir.name
    atomic_write_json(run_dir / "config.resolved.json", resolved)
    atomic_write_json(
        run_dir / "manifest.json",
        {
            "schema": "ebl.run",
            "schema_version": 1,
            "run_id": run_id,
            "experiment_id": spec.experiment_id,
            "config": {"path": "config.resolved.json", "sha256": content_hash(resolved)},
            "inputs": [{"role": "weights", "path": "/frozen/teacher.pt", "sha256": EXPECTED_WEIGHTS_SHA256}],
            "study": {
                "study_id": study["study_id"],
                "arm_id": arm["arm_id"],
                "evidence_class": study["evidence_class"],
                "study_sha256": sha256_file(study_dir / "study.json"),
                "source_plan_sha256": study["source_plan"]["sha256"],
                "source_config_sha256": config_record["sha256"],
            },
        },
    )
    atomic_write_json(
        run_dir / "status.json",
        {"schema": "ebl.run", "schema_version": 1, "run_id": run_id, "status": "complete"},
    )
    atomic_write_json(
        run_dir / "result.json",
        {
            "schema": "ebl.run",
            "schema_version": 1,
            "run_id": run_id,
            "experiment_id": spec.experiment_id,
            "status": "complete",
            "metrics": metrics,
            "artifacts": artifacts,
            "error": None,
        },
    )
    return run_dir


def _study(tmp_path: Path) -> tuple[Path, list[Path]]:
    study_dir = prepare_study(PLAN, tmp_path / "results")
    study = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    runs = []
    for arm in study["arms"]:
        for config in arm["configs"]:
            runs.append(_one_run(study_dir, study, arm, config))
    return study_dir, runs


def test_analyzer_audits_full_4_by_3_study_and_applies_predeclared_gates(
    tmp_path: Path,
) -> None:
    study_dir, runs = _study(tmp_path)
    output = tmp_path / "explicit-analysis"
    report = analyze_study(study_dir, output)

    assert report["schema"] == ANALYSIS_SCHEMA
    assert report["audit"]["complete_run_count"] == 12
    assert report["audit"]["runtime_device"] == "cuda"
    assert report["decision"]["predeclared_unique_winner"] == (
        "shared_destination_columns_reset_max"
    )
    assert report["decision"]["initialization_gate"]["status"] == "passed"
    assert report["policies"]["shared_destination_columns_reset_max"][
        "continuous"
    ]["correct"] == [9300, 9300, 9300]
    assert (output / "baseline_selection_analysis.json").is_file()
    assert (output / "baseline_selection_runs.csv").read_text(encoding="utf-8").count("\n") == 13
    assert "diagnostic" in (output / "baseline_selection_analysis.md").read_text(encoding="utf-8")

    prediction_path = runs[0] / "artifacts/heldout/predictions.npz"
    prediction_path.write_bytes(prediction_path.read_bytes() + b"tamper")
    with pytest.raises(BaselineSelectionAnalysisError, match="(size|hash) mismatch"):
        analyze_study(study_dir, tmp_path / "tampered-analysis")


def test_analyzer_requires_all_twelve_complete_bundles(tmp_path: Path) -> None:
    study_dir, runs = _study(tmp_path)
    atomic_write_json(
        runs[-1] / "status.json",
        {"schema": "ebl.run", "schema_version": 1, "run_id": runs[-1].name, "status": "failed"},
    )
    with pytest.raises(
        BaselineSelectionAnalysisError,
        match="exactly 12 complete RunStore bundles",
    ):
        analyze_study(study_dir)
