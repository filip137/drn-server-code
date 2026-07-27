from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.run_mnist_conv_perfectdiode_conv3_hparam_v2 import main
from experiments.mnist_conv.identity import sha256_file, sha256_json
from experiments.mnist_conv.io import (
    atomic_write_bytes,
    atomic_write_json,
    read_json,
)
from experiments.mnist_conv.perfectdiode_conv3_hparam_v2_spec import (
    CANARY_STEPS,
    CANDIDATE_TOTAL_STEPS,
    DEFAULT_OPERATING_POINT_CONTRACT,
    DEFAULT_TEMPLATE,
    FIXED_HIGH_GRID,
    FIXED_HIGH_RHO_CONV,
    FIXED_HIGH_RHO_DENSE,
    PUBLIC_STAGE_SEQUENCE,
    PerfectDiodeConv3HparamStudySpec,
    PerfectDiodeConv3HparamValidationError,
    build_surface_manifest,
    load_frozen_template,
    materialize_study,
    representative_preflight_plan,
    validate_surface_manifest,
)
from experiments.mnist_conv.perfectdiode_tk_spec import (
    CONV3_CONV_WEIGHTS,
    K_GRID,
    T_CORE_GRID,
    T_EXTENSION_GRID,
    PerfectDiodeTKStudySpec,
    select_k_measurements,
    select_t_measurements,
    tk_entry_id,
)
from experiments.mnist_conv.perfectdiode_hparam_v2_execution import (
    ENVIRONMENT_RECEIPT_SCHEMA_VERSION,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
V1_CONFIG = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv12_sgd_adam_hparam_ordinary_mnist_v1.json"
)
TK_CONFIG = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv3_tk_ordinary_mnist_v1.json"
)
TK_LAUNCHER = (
    REPO_ROOT / "experiments" / "run_mnist_conv_perfectdiode_tk.py"
)
V1_FILES = {
    REPO_ROOT / "experiments/mnist_conv/perfectdiode_hparam_spec.py": (
        "47246847a1cf77296c15237c8632de4abbbc1c70bba29a786057b5d36551bace"
    ),
    REPO_ROOT / "experiments/mnist_conv/perfectdiode_hparam.py": (
        "fbedb2c5da9f0a06b7ee63cfcb79c4a946608dfa19629cfa05f37bcb58ade00a"
    ),
    REPO_ROOT / "experiments/mnist_conv/perfectdiode_hparam_runtime.py": (
        "c42c1c4cd78279506bf0d68d252c6105e3db8c83bb898dbbd2811407507d97d9"
    ),
    REPO_ROOT / "experiments/run_mnist_conv_perfectdiode_hparam.py": (
        "0a80bb080c01d52712bbb30ec1301a7fd9f8ae284cd792a914d5007ef39151c8"
    ),
    V1_CONFIG: (
        "647f92e01c8619951367d71cadd6320c6924c0223a2f5ed344f7ff1739bac5c1"
    ),
}

SCHEMES = (
    (
        "conv3_baseline_v1_c1",
        "baseline",
        "mnist_bp_amp_v1_c1",
        1.0,
        1.0,
        "main",
    ),
    (
        "conv3_ours_v4_c1",
        "ours",
        "mnist_bp_amp_v4_c1",
        4.0,
        1.0,
        "akibscomputer",
    ),
    (
        "conv3_legacy_v4_c0p25",
        "legacy",
        "mnist_bp_amp_v4_c0p25",
        4.0,
        0.25,
        "main",
    ),
)


def _artifact_record(path: Path, *, base: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _residual_stats(value: float) -> dict[str, float]:
    return {
        "mean": value,
        "median": value,
        "p90": value,
        "p99": value,
        "max": value,
    }


def _t_measurement(
    iteration: int,
    *,
    selected_threshold: int,
    t_cohort_sha: str,
    checkpoint_sha: str,
    tensor_sha: str,
) -> dict[str, object]:
    residual = 0.005 if iteration >= selected_threshold else 0.02
    layers = []
    for role, mode in (
        ("hidden_0", "projected_kkt"),
        ("hidden_1", "projected_kkt"),
        ("hidden_2", "projected_kkt"),
        ("output", "raw"),
    ):
        layer: dict[str, object] = {
            "role": role,
            "selection_residual_mode": mode,
            "selection_residual_p90": residual,
            "selection_residual": _residual_stats(residual),
            "raw_residual": _residual_stats(residual),
        }
        if role != "output":
            layer["clamped_occupancy"] = {
                "fraction": 0.25,
                "excitation_fraction": 0.125,
                "inhibition_fraction": 0.125,
            }
        layers.append(layer)
    return {
        "iteration_count": iteration,
        "num_examples": 1024,
        "batch_size": 64,
        "cohort_source_indices_sha256": t_cohort_sha,
        "initialization_checkpoint_sha256": checkpoint_sha,
        "initialization_tensor_sha256": tensor_sha,
        "official_test_read": False,
        "fixed_step_minimization": True,
        "batch_state_policy": "reset_each_batch",
        "layers": layers,
    }


def _k_parameter_diagnostic(
    name: str,
    *,
    passed: bool,
) -> dict[str, object]:
    gradient = 1.0 if passed else 0.5
    norm_delta = 0.0 if passed else 0.5
    relative_error = 0.0 if passed else 0.5
    batches = [
        {
            "batch_index": batch_index,
            "gradient_l2": gradient,
            "reference_gradient_l2": 1.0,
            "relative_gradient_l2_norm_delta": norm_delta,
            "gradient_vector_relative_error": relative_error,
            "gradient_zero_fraction": 0.0,
            "reference_gradient_zero_fraction": 0.0,
            "absolute_zero_fraction_delta": 0.0,
            "gradient_vector_cosine": 1.0,
            "reference_gradient_rms": 1.0,
        }
        for batch_index in range(8)
    ]
    return {
        "parameter": name,
        "batch_count": 8,
        "batches": batches,
        "gradient_l2": gradient,
        "reference_gradient_l2": 1.0,
        "relative_gradient_l2_norm_delta": norm_delta,
        "gradient_vector_relative_error": relative_error,
        "gradient_zero_fraction": 0.0,
        "reference_gradient_zero_fraction": 0.0,
        "absolute_zero_fraction_delta": 0.0,
        "gradient_vector_cosine": 1.0,
        "reference_median_gradient_rms": 1.0,
        "reference_q90_zero_fraction": 0.0,
        "initial_weight_rms": 1.0,
        "reference_nominal_update_unit": 1.0,
        "reference_viable": True,
        "passed": passed,
    }


def _k_measurement(
    candidate_k: int,
    *,
    selected_k: int,
    selected_t: int,
    k_cohort_sha: str,
    checkpoint_sha: str,
    tensor_sha: str,
    reference_k: int = 64,
) -> dict[str, object]:
    passed = candidate_k >= selected_k
    return {
        "candidate_k": candidate_k,
        "reference_k": reference_k,
        "batch_count": 8,
        "cohort_examples": 256,
        "batch_size": 32,
        "official_test_read": False,
        "fixed_step_minimization": True,
        "batch_state_policy": "reset_each_batch",
        "selected_t": selected_t,
        "cohort_source_indices_sha256": k_cohort_sha,
        "initialization_checkpoint_sha256": checkpoint_sha,
        "initialization_tensor_sha256": tensor_sha,
        "free_equilibrium_sha256_by_batch": ["1" * 64] * 8,
        "parameter_diagnostics": [
            _k_parameter_diagnostic(name, passed=passed)
            for name in CONV3_CONV_WEIGHTS
        ],
        "reference_viable": True,
        "passed": passed,
    }


def _shared_tk_asset_bundle(
    root: Path,
    tk_spec: PerfectDiodeTKStudySpec,
) -> dict[str, object]:
    asset_root = root / "shared_assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    checkpoint = asset_root / "initialization.pt"
    atomic_write_bytes(checkpoint, b"full-hash-bound-test-checkpoint")
    train_indices = list(range(55_000))
    validation_indices = list(range(55_000, 60_000))
    t_indices = list(range(1_024))
    k_indices = list(range(256))
    assets_path = asset_root / "assets.json"
    assets = {
        "schema_version": "mnist-conv-perfectdiode-tk-assets/v1",
        "study_id": tk_spec.study_id,
        "config_sha256": tk_spec.config_sha256,
        "architecture": "conv3",
        "model_seed": 0,
        "official_test_read": False,
        "initialization": {
            "path": checkpoint.name,
            "sha256": sha256_file(checkpoint),
            "parameter_tensor_sha256": "d" * 64,
            "parameter_names": [
                "ConvWeight_0",
                "Bias_0",
                "ConvWeight_1",
                "Bias_1",
                "ConvWeight_2",
                "Bias_2",
                "DenseWeight_0",
            ],
            "shared_across_schemes": True,
        },
        "split": {
            "source": "mnist_train",
            "train_size": 55_000,
            "validation_size": 5_000,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "train_indices_sha256": sha256_json(train_indices),
            "validation_indices_sha256": sha256_json(validation_indices),
        },
        "cohorts": {
            "base_batch_size": 16,
            "t": {
                "examples": 1_024,
                "batch_size": 64,
                "source_indices": t_indices,
                "source_indices_sha256": sha256_json(t_indices),
            },
            "k": {
                "examples": 256,
                "batch_size": 32,
                "source_indices": k_indices,
                "source_indices_sha256": sha256_json(k_indices),
            },
        },
    }
    atomic_write_json(assets_path, assets, canonical=True)
    completion_path = asset_root / "completion.json"
    atomic_write_json(
        completion_path,
        {
            "schema_version": (
                "mnist-conv-perfectdiode-tk-assets-completion/v1"
            ),
            "state": "complete",
            "study_id": tk_spec.study_id,
            "config_sha256": tk_spec.config_sha256,
            "official_test_read": False,
            "outputs": [
                _artifact_record(checkpoint, base=asset_root),
                _artifact_record(assets_path, base=asset_root),
            ],
        },
        canonical=True,
    )
    return {
        "schema_version": assets["schema_version"],
        "study_id": tk_spec.study_id,
        "config_sha256": tk_spec.config_sha256,
        "assets_sha256": sha256_file(assets_path),
        "completion_sha256": sha256_file(completion_path),
        "initialization_checkpoint_sha256": assets["initialization"]["sha256"],
        "initialization_tensor_sha256": assets["initialization"][
            "parameter_tensor_sha256"
        ],
        "train_indices_sha256": assets["split"]["train_indices_sha256"],
        "validation_indices_sha256": assets["split"][
            "validation_indices_sha256"
        ],
        "t_cohort_indices_sha256": assets["cohorts"]["t"][
            "source_indices_sha256"
        ],
        "k_cohort_indices_sha256": assets["cohorts"]["k"][
            "source_indices_sha256"
        ],
    }


def _tk_selection(
    root: Path,
    *,
    statuses: tuple[str, str, str] = ("selected", "selected", "selected"),
    audit_passed: bool = True,
    selected_ts: tuple[int, int, int] = (8, 10, 16),
    selected_ks: tuple[int, int, int] = (4, 6, 8),
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tk_spec = PerfectDiodeTKStudySpec.from_path(TK_CONFIG)
    resolved_config = root / "resolved_config.json"
    atomic_write_json(resolved_config, tk_spec.data, canonical=True)
    asset_binding = _shared_tk_asset_bundle(root, tk_spec)
    source_archive = root / "source" / "source.tar"
    source_archive.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(source_archive, b"exact-tk-producer-source-archive")
    entries = []
    for index, row in enumerate(tk_spec.rows):
        entry_id = tk_entry_id(tk_spec.study_id, row)
        entry_dir = f"entries/{entry_id}"
        entries.append(
            {
                "entry_index": index,
                "entry_id": entry_id,
                "row_id": row["row_id"],
                "scheme": row["scheme"],
                "architecture": "conv3",
                "execution": {
                    "backend": "tmux",
                    "lane": SCHEMES[index][-1],
                    "whole_row_on_one_lane": True,
                },
                "output_dir": entry_dir,
                "completion_path": f"{entry_dir}/completion.json",
                "outputs": [
                    f"{entry_dir}/environment.json",
                    f"{entry_dir}/t_measurements.json",
                    f"{entry_dir}/k_measurements.json",
                    f"{entry_dir}/operating_point_audit.json",
                    f"{entry_dir}/result.json",
                ],
            }
        )
    manifest = {
        "schema_version": "mnist-conv-perfectdiode-tk-manifest/v1",
        "study_id": tk_spec.study_id,
        "config_path": "resolved_config.json",
        "config_sha256": tk_spec.config_sha256,
        "resolved_config_file_sha256": sha256_file(resolved_config),
        "launcher_sha256": sha256_file(TK_LAUNCHER),
        "code_provenance": {
            "git_revision": "1" * 40,
            "dirty_source_digest": None,
            "effective_code_fingerprint": "2" * 64,
        },
        "staged_source": {
            "commit": "1" * 40,
            "archive_sha256": sha256_file(source_archive),
            "effective_code_fingerprint": "2" * 64,
        },
        "asset_binding": asset_binding,
        "entry_count": 3,
        "entries": entries,
        "selection_path": "selection.json",
    }
    manifest_path = root / "manifest.json"
    atomic_write_json(manifest_path, manifest, canonical=True)
    manifest_sha = sha256_file(manifest_path)
    rows = []
    t_cohort_sha = str(asset_binding["t_cohort_indices_sha256"])
    k_cohort_sha = str(asset_binding["k_cohort_indices_sha256"])
    checkpoint_sha = str(
        asset_binding["initialization_checkpoint_sha256"]
    )
    tensor_sha = str(asset_binding["initialization_tensor_sha256"])
    for index, (row, entry) in enumerate(zip(tk_spec.rows, entries)):
        requested_status = statuses[index]
        if requested_status not in {
            "selected",
            "unresolved_t_extension_sentinel",
        }:
            raise ValueError(requested_status)
        selected = requested_status == "selected"
        selected_t_target = selected_ts[index]
        selected_k_target = selected_ks[index]
        entry_root = root / entry["output_dir"]
        entry_root.mkdir(parents=True, exist_ok=True)
        execution_host = entry["execution"]["lane"]
        execution_environment_sha256 = (
            "6" * 64 if execution_host == "main" else "9" * 64
        )
        environment_path = entry_root / "environment.json"
        atomic_write_json(
            environment_path,
            {
                "schema_version": (
                    "mnist-conv-perfectdiode-tk-environment/v1"
                ),
                "execution_backend": "tmux",
                "execution_host": execution_host,
                "python_version": "test",
            },
            canonical=True,
        )
        execution_environment_sha256 = sha256_file(environment_path)
        t_iterations = (
            T_CORE_GRID if selected else T_CORE_GRID + T_EXTENSION_GRID
        )
        t_by_iteration = {
            count: _t_measurement(
                count,
                selected_threshold=(
                    selected_t_target if selected else T_EXTENSION_GRID[-1] + 1
                ),
                t_cohort_sha=t_cohort_sha,
                checkpoint_sha=checkpoint_sha,
                tensor_sha=tensor_sha,
            )
            for count in t_iterations
        }
        t_selection = select_t_measurements(t_by_iteration)
        assert t_selection["status"] == requested_status
        t_path = entry_root / "t_measurements.json"
        atomic_write_json(
            t_path,
            {
                "schema_version": (
                    "mnist-conv-perfectdiode-tk-t-measurements/v1"
                ),
                "study_id": tk_spec.study_id,
                "config_sha256": tk_spec.config_sha256,
                "entry_id": entry["entry_id"],
                "row_id": row["row_id"],
                "architecture": "conv3",
                "scheme": row["scheme"],
                "official_test_read": False,
                "cohort_source_indices_sha256": t_cohort_sha,
                "initialization_checkpoint_sha256": checkpoint_sha,
                "measurements": [
                    t_by_iteration[count] for count in t_iterations
                ],
                "selection": t_selection,
            },
            canonical=True,
        )
        k_by_iteration = (
            {
                count: _k_measurement(
                    count,
                    selected_k=selected_k_target,
                    selected_t=selected_t_target,
                    k_cohort_sha=k_cohort_sha,
                    checkpoint_sha=checkpoint_sha,
                    tensor_sha=tensor_sha,
                )
                for count in K_GRID
            }
            if selected
            else {}
        )
        k_selection = (
            select_k_measurements(k_by_iteration)
            if selected
            else {
                "status": "not_run_unresolved_t",
                "selected_k": None,
                "k128_sentinel_used": False,
                "candidates": [],
            }
        )
        if selected:
            assert k_selection["status"] == "selected"
            assert k_selection["selected_k"] == selected_k_target
        k_path = entry_root / "k_measurements.json"
        atomic_write_json(
            k_path,
            {
                "schema_version": (
                    "mnist-conv-perfectdiode-tk-k-measurements/v1"
                ),
                "study_id": tk_spec.study_id,
                "config_sha256": tk_spec.config_sha256,
                "entry_id": entry["entry_id"],
                "row_id": row["row_id"],
                "architecture": "conv3",
                "scheme": row["scheme"],
                "official_test_read": False,
                "selected_t": (
                    selected_t_target if selected else None
                ),
                "cohort_source_indices_sha256": k_cohort_sha,
                "initialization_checkpoint_sha256": checkpoint_sha,
                "measurements": [
                    k_by_iteration[count] for count in K_GRID
                ] if selected else [],
                "k128_sentinel": None,
                "selection": k_selection,
            },
            canonical=True,
        )
        audit_path = entry_root / "operating_point_audit.json"
        audit = {
            "schema_version": (
                "mnist-conv-perfectdiode-tk-operating-point-audit/v1"
            ),
            "study_id": tk_spec.study_id,
            "config_sha256": tk_spec.config_sha256,
            "entry_id": entry["entry_id"],
            "row_id": row["row_id"],
            "architecture": "conv3",
            "scheme": row["scheme"],
            "execution_backend": "tmux",
            "execution_host": execution_host,
            "execution_environment_sha256": execution_environment_sha256,
            "official_test_read": False,
            "selected_t": selected_t_target if selected else None,
            "selected_k": selected_k_target if selected else None,
            "status": "passed" if selected else "not_run_unresolved_selection",
            "passed": selected,
            "failure_reasons": [] if selected else [requested_status],
            "fresh_replay_after_selection": selected,
            "t_measurement": (
                t_by_iteration[selected_t_target] if selected else None
            ),
            "t64_sentinel_measurement": (
                t_by_iteration[64] if selected else None
            ),
            "t64_sentinel_passed": True if selected else None,
            "t256_extension_sentinel_measurement": None,
            "t256_extension_sentinel_passed": None,
            "t_extension_used": False if selected else True,
            "k_measurement": (
                k_by_iteration[selected_k_target] if selected else None
            ),
            "k_audit_reference": 64 if selected else None,
        }
        atomic_write_json(audit_path, audit, canonical=True)
        diagnostic_status = (
            "selected" if selected else requested_status
        )
        result_path = entry_root / "result.json"
        result = {
            "schema_version": "mnist-conv-perfectdiode-tk-row-result/v1",
            "study_id": tk_spec.study_id,
            "config_sha256": tk_spec.config_sha256,
            "manifest_sha256": manifest_sha,
            "entry_id": entry["entry_id"],
            "row_id": row["row_id"],
            "architecture": "conv3",
            "scheme": row["scheme"],
            "execution_backend": "tmux",
            "execution_host": execution_host,
            "execution_environment_sha256": execution_environment_sha256,
            "run_name": row["run_name"],
            "voltage_amp": row["voltage_amp"],
            "current_amp": row["current_amp"],
            "input_gain": row["input_gain"],
            "official_test_read": False,
            "status": diagnostic_status,
            "diagnostic_selection_status": diagnostic_status,
            "diagnostic_selected_t": t_selection.get("selected_t"),
            "diagnostic_selected_k": k_selection.get("selected_k"),
            "selected_t": selected_t_target if selected else None,
            "selected_k": selected_k_target if selected else None,
            "t_reference": 64,
            "k_reference": 64,
            "t_extension_used": bool(t_selection["extension_used"]),
            "k128_sentinel_used": bool(
                k_selection["k128_sentinel_used"]
            ),
            "initialization_checkpoint_sha256": checkpoint_sha,
            "initialization_tensor_sha256": tensor_sha,
            "train_indices_sha256": asset_binding[
                "train_indices_sha256"
            ],
            "validation_indices_sha256": asset_binding[
                "validation_indices_sha256"
            ],
            "t_cohort_indices_sha256": t_cohort_sha,
            "k_cohort_indices_sha256": k_cohort_sha,
            "operating_point_audit_path": audit_path.name,
            "operating_point_audit_sha256": sha256_file(audit_path),
            "operating_point_audit_passed": selected,
            "t_selection": t_selection,
            "k_selection": k_selection,
            "artifacts": [
                _artifact_record(t_path, base=entry_root),
                _artifact_record(k_path, base=entry_root),
                _artifact_record(audit_path, base=entry_root),
            ],
        }
        atomic_write_json(result_path, result, canonical=True)
        completion_path = entry_root / "completion.json"
        atomic_write_json(
            completion_path,
            {
                "schema_version": (
                    "mnist-conv-perfectdiode-tk-entry-completion/v1"
                ),
                "state": "complete",
                "study_id": tk_spec.study_id,
                "entry_id": entry["entry_id"],
                "manifest_sha256": manifest_sha,
                "official_test_read": False,
                "outputs": [
                    _artifact_record(root / relative, base=root)
                    for relative in entry["outputs"]
                ],
            },
            canonical=True,
        )
        rows.append(
            {
                "row_id": row["row_id"],
                "architecture": "conv3",
                "scheme": row["scheme"],
                "run_name": row["run_name"],
                "voltage_amp": row["voltage_amp"],
                "current_amp": row["current_amp"],
                "input_gain": 360.0,
                "status": diagnostic_status,
                "selected_t": selected_t_target if selected else None,
                "selected_k": selected_k_target if selected else None,
                "t_reference": 64,
                "k_reference": 64,
                "t_extension_used": bool(t_selection["extension_used"]),
                "k128_sentinel_used": False,
                "execution_host": execution_host,
                "execution_backend": "tmux",
                "execution_environment_path": entry["outputs"][0],
                "execution_environment_sha256": execution_environment_sha256,
                "t_cohort_indices_sha256": t_cohort_sha,
                "k_cohort_indices_sha256": k_cohort_sha,
                "result_path": entry["outputs"][-1],
                "result_sha256": sha256_file(result_path),
                "completion_path": entry["completion_path"],
                "completion_sha256": sha256_file(completion_path),
                "operating_point_audit_path": entry["outputs"][-2],
                "operating_point_audit_sha256": sha256_file(audit_path),
                "operating_point_audit_passed": selected,
            }
        )
    selection = {
        "schema_version": "mnist-conv-perfectdiode-tk-selection/v1",
        "study_id": tk_spec.study_id,
        "config_sha256": tk_spec.config_sha256,
        "manifest_sha256": manifest_sha,
        "official_test_read": False,
        "status": (
            "selected"
            if all(status == "selected" for status in statuses)
            else "unresolved"
        ),
        "rows": rows,
        "artifact_sha256s": {
            "asset_bundle": asset_binding,
            "rows": {
                row["row_id"]: {
                    "result_sha256": row["result_sha256"],
                    "completion_sha256": row["completion_sha256"],
                    "operating_point_audit_sha256": row[
                        "operating_point_audit_sha256"
                    ],
                    "execution_environment_sha256": row[
                        "execution_environment_sha256"
                    ],
                }
                for row in rows
            },
        },
    }
    if not audit_passed:
        selection["rows"][0]["operating_point_audit_passed"] = False
    path = root / "selection.json"
    atomic_write_json(path, selection, canonical=True)
    return path


def _refresh_row_hashes(root: Path, *, row_index: int) -> None:
    manifest = read_json(root / "manifest.json")
    selection = read_json(root / "selection.json")
    entry = manifest["entries"][row_index]
    entry_root = root / entry["output_dir"]
    result_path = root / entry["outputs"][-1]
    result = read_json(result_path)
    audit_path = root / entry["outputs"][-2]
    audit = read_json(audit_path)
    result["operating_point_audit_sha256"] = sha256_file(audit_path)
    result["operating_point_audit_passed"] = audit.get("passed") is True
    result["artifacts"] = [
        _artifact_record(root / relative, base=entry_root)
        for relative in entry["outputs"][1:4]
    ]
    atomic_write_json(result_path, result, canonical=True)
    completion_path = root / entry["completion_path"]
    completion = read_json(completion_path)
    completion["outputs"] = [
        _artifact_record(root / relative, base=root)
        for relative in entry["outputs"]
    ]
    atomic_write_json(completion_path, completion, canonical=True)
    selection_row = selection["rows"][row_index]
    selection_row["operating_point_audit_sha256"] = sha256_file(
        audit_path
    )
    selection_row["operating_point_audit_passed"] = (
        audit.get("passed") is True
    )
    selection_row["result_sha256"] = sha256_file(result_path)
    selection_row["completion_sha256"] = sha256_file(completion_path)
    row_hashes = selection["artifact_sha256s"]["rows"][
        selection_row["row_id"]
    ]
    row_hashes["result_sha256"] = selection_row["result_sha256"]
    row_hashes["completion_sha256"] = selection_row["completion_sha256"]
    row_hashes["operating_point_audit_sha256"] = selection_row[
        "operating_point_audit_sha256"
    ]
    atomic_write_json(root / "selection.json", selection, canonical=True)


def _materialize(
    tmp_path: Path,
    *,
    statuses: tuple[str, str, str] = ("selected", "selected", "selected"),
    audit_passed: bool = True,
    use_fixed_operating_point: bool = False,
    selected_ts: tuple[int, int, int] = (8, 10, 16),
    selected_ks: tuple[int, int, int] = (4, 6, 8),
) -> PerfectDiodeConv3HparamStudySpec:
    upstream = tmp_path / "tk"
    selection = _tk_selection(
        upstream,
        statuses=statuses,
        audit_passed=audit_passed,
        selected_ts=selected_ts,
        selected_ks=selected_ks,
    )
    return materialize_study(
        template_path=DEFAULT_TEMPLATE,
        tk_selection_path=selection,
        tk_selection_sha256=sha256_file(selection),
        operating_point_path=(
            DEFAULT_OPERATING_POINT_CONTRACT
            if use_fixed_operating_point
            else None
        ),
        output_dir=tmp_path / "lr",
    )


def _shared_assets(
    spec: PerfectDiodeConv3HparamStudySpec,
) -> dict[str, str]:
    upstream = spec.data["upstream_tk"]["artifact_sha256s"][
        "asset_bundle"
    ]
    return {
        key: upstream[key]
        for key in (
            "train_indices_sha256",
            "validation_indices_sha256",
            "initialization_checkpoint_sha256",
            "initialization_tensor_sha256",
            "t_cohort_indices_sha256",
            "k_cohort_indices_sha256",
        )
    } | {
        "train_batch_order_epoch_1_sha256": "e" * 64,
        "train_batch_order_epoch_2_sha256": "f" * 64,
        "train_batch_order_epoch_3_sha256": "0" * 64,
    }


def _execution_source(tmp_path: Path) -> dict[str, str]:
    return {
        "source_commit": "1" * 40,
        "source_archive_sha256": "2" * 64,
        "effective_code_fingerprint": "3" * 64,
        "worker_launcher_sha256": "4" * 64,
        "environment_contract_sha256": "5" * 64,
        "host_environment_sha256s": {
            "main": "6" * 64,
            "akibscomputer": "9" * 64,
        },
        "resolved_study_path": "study.resolved.json",
        "resolved_study_sha256": sha256_file(
            tmp_path / "lr" / "study.resolved.json"
        ),
        "output_root": "surfaces",
    }


def _host_environment_receipt(
    host: str,
    *,
    torch_version: str,
) -> dict[str, object]:
    delegated = host == "akibscomputer"
    return {
        "schema_version": ENVIRONMENT_RECEIPT_SCHEMA_VERSION,
        "execution_backend": "tmux",
        "execution_host": host,
        "tmux": {
            "attestation_mode": (
                "delegated_remote_tmux_session"
                if delegated
                else "direct_tmux_session"
            ),
            "session_name": host,
            "pane_token": "%8" if delegated else "%2",
        },
        "python": {
            "implementation": "CPython",
            "version": "3.12.9 (test build)",
            "executable": (
                "/home/filiposana/miniconda3/envs/py312/bin/python"
                if delegated
                else "/home/filip/miniconda3/envs/py312/bin/python"
            ),
        },
        "torch": {
            "version": torch_version,
            "cuda_build_version": "12.1",
            "cudnn_version": 90100,
        },
        "cuda": {
            "available": True,
            "visible_devices": "0",
            "device_count": 1,
            "devices": [
                {
                    "index": 0,
                    "name": "Test GPU",
                    "compute_capability": [8, 6],
                    "total_memory_bytes": 24_000_000_000,
                }
            ],
        },
        "official_test_read": False,
    }


def test_template_is_frozen_and_v1_is_byte_for_byte_unchanged() -> None:
    template, source = load_frozen_template()

    assert source == DEFAULT_TEMPLATE.resolve()
    assert template["base_contract"]["canonical_sha256"] == (
        "699804e1f7c35f65f50656db4af0b120dd3a3d0f2fb95e3a17b65ebfe7b78535"
    )
    assert template["training_data_contract"] == {
        "training_batch_size": 16,
        "validation_batch_size": 64,
        "candidate_epochs": 3,
        "steps_per_epoch": 3438,
        "candidate_total_steps": 10314,
        "bind_each_epoch_train_batch_order_sha256": True,
        "tk_diagnostic_batching_is_not_inherited": True,
    }
    for path, expected in V1_FILES.items():
        assert sha256_file(path) == expected


def test_public_stage_sequence_matches_protocol_literal() -> None:
    assert PUBLIC_STAGE_SEQUENCE == (
        "audit",
        "import_tk",
        "optimizer_probe",
        "rho_canary_core",
        "rho_core_candidates",
        "select_core",
        "rho_canary_expansion",
        "rho_expansion_candidates",
        "select_expanded",
        "post_training_tk",
        "finalize_lr",
    )
    template, _source = load_frozen_template()
    assert template["stages"]["public_sequence"] == list(
        PUBLIC_STAGE_SEQUENCE
    )


def test_materialization_binds_tk_and_freezes_conv3_runtime_contract(
    tmp_path: Path,
) -> None:
    spec = _materialize(tmp_path)
    data = spec.data

    assert data["schema_version"] == "mnist-conv-perfectdiode-hparam-study/v2"
    assert data["run_schema_version"] == "mnist-conv-perfectdiode-hparam-run/v2"
    assert data["dataset"]["variant"] == "ordinary"
    assert data["dataset"]["train"]["batch_size"] == 16
    assert data["dataset"]["validation"]["batch_size"] == 64
    assert data["dataset"]["official_test"] == {
        "enabled": False,
        "read_allowed": False,
    }
    assert data["model"]["model_seed"] == 0
    assert data["model"]["input_gain"]["conv3"] == 360.0
    assert data["model"]["architectures"]["conv3"]["channels"] == [64, 128, 256]
    assert data["model"]["architectures"]["conv3"]["strides"] == [2, 2, 1]
    assert data["candidate_training"]["total_steps"] == CANDIDATE_TOTAL_STEPS
    assert data["stages"]["public_sequence"] == list(PUBLIC_STAGE_SEQUENCE)
    assert data["long_confirm"]["enabled"] is False
    assert [row["scheme"] for row in spec.rows] == list(
        ("baseline", "ours", "legacy")
    )
    assert [
        (row["inference_iterations"], row["training_iterations"])
        for row in spec.rows
    ] == [(8, 4), (10, 6), (16, 8)]
    for row in spec.rows:
        assert row["lr_eligible"] is True
        assert row["upstream_tk"]["operating_point_audit_passed"] is True
        assert row["upstream_tk"]["t_extension_used"] is False
        assert row["upstream_tk"]["k128_sentinel_used"] is False
        assert len(row["upstream_tk"]["operating_point_audit_sha256"]) == 64
        assert (
            tmp_path
            / "lr"
            / "upstream_tk"
            / row["upstream_tk"]["t_measurements_path"]
        ).is_file()
        assert (
            tmp_path
            / "lr"
            / "upstream_tk"
            / row["upstream_tk"]["k_measurements_path"]
        ).is_file()
    assert (tmp_path / "lr" / "upstream_tk" / "manifest.json").is_file()
    assert (
        tmp_path / "lr" / "upstream_tk" / "resolved_config.json"
    ).is_file()
    assert (
        tmp_path
        / "lr"
        / "upstream_tk"
        / "shared_assets"
        / "initialization.pt"
    ).is_file()
    assert (
        tmp_path / "lr" / "upstream_tk" / "source" / "source.tar"
    ).is_file()

    reloaded = PerfectDiodeConv3HparamStudySpec.from_path(
        tmp_path / "lr" / "study.resolved.json"
    )
    assert reloaded.study_id == spec.study_id
    assert reloaded.config_sha256 == spec.config_sha256


def test_fixed_lr_operating_point_preserves_upstream_tk_provenance(
    tmp_path: Path,
) -> None:
    spec = _materialize(
        tmp_path,
        use_fixed_operating_point=True,
        selected_ts=(4, 4, 4),
        selected_ks=(4, 4, 4),
    )

    assert spec.data["operating_point"] == {
        "schema_version": "mnist-conv-perfectdiode-lr-operating-point/v1",
        "mode": "user_fixed_after_residual_gradient_review",
        "inference_iterations": 8,
        "training_iterations": 8,
        "reference_inference_iterations": 64,
        "reference_training_iterations": 64,
        "shared_across_schemes": True,
        "retain_upstream_diagnostic_selection": True,
        "require_fresh_manifest_bound_preflight_gate": True,
        "provenance": "user_directed_2026-07-27",
    }
    assert len(spec.data["operating_point_contract_sha256"]) == 64
    assert [
        (row["inference_iterations"], row["training_iterations"])
        for row in spec.rows
    ] == [(8, 8), (8, 8), (8, 8)]
    assert [
        (
            row["upstream_tk"]["selected_t"],
            row["upstream_tk"]["selected_k"],
        )
        for row in spec.rows
    ] == [(4, 4), (4, 4), (4, 4)]

    manifest = build_surface_manifest(
        spec,
        shared_asset_hashes=_shared_assets(spec),
        execution_source=_execution_source(tmp_path),
    )
    upstream_by_row = {
        row["row_id"]: (
            row["upstream_tk"]["selected_t"],
            row["upstream_tk"]["selected_k"],
        )
        for row in spec.rows
    }
    for surface in manifest["surfaces"]:
        assert (
            surface["inference_iterations"],
            surface["training_iterations"],
        ) == (8, 8)
        assert (
            surface["upstream_tk"]["selected_t"],
            surface["upstream_tk"]["selected_k"],
        ) == upstream_by_row[surface["row_id"]]

    reloaded = PerfectDiodeConv3HparamStudySpec.from_path(
        tmp_path / "lr" / "study.resolved.json"
    )
    assert reloaded.study_id == spec.study_id
    assert reloaded.config_sha256 == spec.config_sha256


def test_fixed_lr_operating_point_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    _materialize(
        tmp_path,
        use_fixed_operating_point=True,
        selected_ts=(4, 4, 4),
        selected_ks=(4, 4, 4),
    )
    resolved = read_json(tmp_path / "lr" / "study.resolved.json")

    contract_tamper = copy.deepcopy(resolved)
    contract_tamper["operating_point"]["training_iterations"] = 6
    contract_tamper_path = tmp_path / "lr" / "contract-tampered.json"
    atomic_write_json(contract_tamper_path, contract_tamper, canonical=True)
    with pytest.raises(
        PerfectDiodeConv3HparamValidationError,
        match="operating-point amendment",
    ):
        PerfectDiodeConv3HparamStudySpec.from_path(contract_tamper_path)

    row_tamper = copy.deepcopy(resolved)
    row_tamper["rows"][0]["inference_iterations"] = 4
    row_tamper_path = tmp_path / "lr" / "row-tampered.json"
    atomic_write_json(row_tamper_path, row_tamper, canonical=True)
    with pytest.raises(
        PerfectDiodeConv3HparamValidationError,
        match="exact template-plus-T/K materialization",
    ):
        PerfectDiodeConv3HparamStudySpec.from_path(row_tamper_path)


def test_selection_sha_and_selected_row_audit_fail_closed(tmp_path: Path) -> None:
    selection = _tk_selection(tmp_path / "tk")
    with pytest.raises(
        PerfectDiodeConv3HparamValidationError,
        match="selection SHA-256",
    ):
        materialize_study(
            template_path=DEFAULT_TEMPLATE,
            tk_selection_path=selection,
            tk_selection_sha256="0" * 64,
            output_dir=tmp_path / "lr-bad-sha",
        )

    failed_audit = _tk_selection(tmp_path / "tk-failed", audit_passed=False)
    with pytest.raises(
        PerfectDiodeConv3HparamValidationError,
        match="operating_point_audit_passed",
    ):
        materialize_study(
            template_path=DEFAULT_TEMPLATE,
            tk_selection_path=failed_audit,
            tk_selection_sha256=sha256_file(failed_audit),
            output_dir=tmp_path / "lr-bad-audit",
        )


def test_tk_consumer_rejects_wrong_frozen_identity_and_route(
    tmp_path: Path,
) -> None:
    wrong_identity = _tk_selection(tmp_path / "wrong-identity")
    identity_value = read_json(wrong_identity)
    identity_value["study_id"] = "tkstudy_" + "0" * 64
    atomic_write_json(wrong_identity, identity_value, canonical=True)
    with pytest.raises(
        PerfectDiodeConv3HparamValidationError,
        match="selection.study_id",
    ):
        materialize_study(
            template_path=DEFAULT_TEMPLATE,
            tk_selection_path=wrong_identity,
            tk_selection_sha256=sha256_file(wrong_identity),
            output_dir=tmp_path / "lr-wrong-identity",
        )

    wrong_route = _tk_selection(tmp_path / "wrong-route")
    route_value = read_json(wrong_route)
    route_value["rows"][1]["execution_host"] = "main"
    atomic_write_json(wrong_route, route_value, canonical=True)
    with pytest.raises(
        PerfectDiodeConv3HparamValidationError,
        match="exactly 'akibscomputer'",
    ):
        materialize_study(
            template_path=DEFAULT_TEMPLATE,
            tk_selection_path=wrong_route,
            tk_selection_sha256=sha256_file(wrong_route),
            output_dir=tmp_path / "lr-wrong-route",
        )


def test_tk_consumer_recomputes_exact_candidate_grids(
    tmp_path: Path,
) -> None:
    selection = _tk_selection(tmp_path / "wrong-grid")
    root = selection.parent
    manifest = read_json(root / "manifest.json")
    t_path = root / manifest["entries"][0]["outputs"][1]
    t_artifact = read_json(t_path)
    t_artifact["measurements"][0]["iteration_count"] = 9
    atomic_write_json(t_path, t_artifact, canonical=True)
    _refresh_row_hashes(root, row_index=0)
    selection = root / "selection.json"

    with pytest.raises(
        PerfectDiodeConv3HparamValidationError,
        match="exact ordered core grid",
    ):
        materialize_study(
            template_path=DEFAULT_TEMPLATE,
            tk_selection_path=selection,
            tk_selection_sha256=sha256_file(selection),
            output_dir=tmp_path / "lr-wrong-grid",
        )


def test_tk_consumer_requires_full_assets_and_source_archive(
    tmp_path: Path,
) -> None:
    bad_assets = _tk_selection(tmp_path / "bad-assets")
    atomic_write_bytes(
        bad_assets.parent / "shared_assets" / "initialization.pt",
        b"tampered-checkpoint",
    )
    with pytest.raises(
        PerfectDiodeConv3HparamValidationError,
        match="shared assets",
    ):
        materialize_study(
            template_path=DEFAULT_TEMPLATE,
            tk_selection_path=bad_assets,
            tk_selection_sha256=sha256_file(bad_assets),
            output_dir=tmp_path / "lr-bad-assets",
        )

    bad_source = _tk_selection(tmp_path / "bad-source")
    atomic_write_bytes(
        bad_source.parent / "source" / "source.tar",
        b"tampered-source",
    )
    with pytest.raises(
        PerfectDiodeConv3HparamValidationError,
        match="producer source archive",
    ):
        materialize_study(
            template_path=DEFAULT_TEMPLATE,
            tk_selection_path=bad_source,
            tk_selection_sha256=sha256_file(bad_source),
            output_dir=tmp_path / "lr-bad-source",
        )


def test_tk_consumer_rejects_unnecessary_t_extension(
    tmp_path: Path,
) -> None:
    selection_path = _tk_selection(tmp_path / "unnecessary-t-extension")
    root = selection_path.parent
    manifest = read_json(root / "manifest.json")
    entry = manifest["entries"][0]
    t_path = root / entry["outputs"][1]
    t_artifact = read_json(t_path)
    base_measurement = t_artifact["measurements"][0]
    for count in T_EXTENSION_GRID:
        t_artifact["measurements"].append(
            _t_measurement(
                count,
                selected_threshold=8,
                t_cohort_sha=base_measurement[
                    "cohort_source_indices_sha256"
                ],
                checkpoint_sha=base_measurement[
                    "initialization_checkpoint_sha256"
                ],
                tensor_sha=base_measurement[
                    "initialization_tensor_sha256"
                ],
            )
        )
    t_by_iteration = {
        measurement["iteration_count"]: measurement
        for measurement in t_artifact["measurements"]
    }
    t_artifact["selection"] = select_t_measurements(t_by_iteration)
    assert t_artifact["selection"]["selected_t"] == 8
    assert t_artifact["selection"]["extension_used"] is True
    atomic_write_json(t_path, t_artifact, canonical=True)

    audit_path = root / entry["outputs"][-2]
    audit = read_json(audit_path)
    audit["t_extension_used"] = True
    audit["t256_extension_sentinel_measurement"] = t_by_iteration[256]
    audit["t256_extension_sentinel_passed"] = True
    atomic_write_json(audit_path, audit, canonical=True)
    result_path = root / entry["outputs"][-1]
    result = read_json(result_path)
    result["t_selection"] = t_artifact["selection"]
    result["t_extension_used"] = True
    atomic_write_json(result_path, result, canonical=True)
    selection = read_json(selection_path)
    selection["rows"][0]["t_extension_used"] = True
    atomic_write_json(selection_path, selection, canonical=True)
    _refresh_row_hashes(root, row_index=0)

    with pytest.raises(
        PerfectDiodeConv3HparamValidationError,
        match="extension grid only after",
    ):
        materialize_study(
            template_path=DEFAULT_TEMPLATE,
            tk_selection_path=selection_path,
            tk_selection_sha256=sha256_file(selection_path),
            output_dir=tmp_path / "lr-unnecessary-t-extension",
        )


def test_tk_consumer_rejects_unnecessary_k128_sentinel(
    tmp_path: Path,
) -> None:
    selection_path = _tk_selection(tmp_path / "unnecessary-k128")
    root = selection_path.parent
    manifest = read_json(root / "manifest.json")
    entry = manifest["entries"][0]
    k_path = root / entry["outputs"][2]
    k_artifact = read_json(k_path)
    first = k_artifact["measurements"][0]
    sentinel = _k_measurement(
        64,
        selected_k=4,
        selected_t=8,
        k_cohort_sha=first["cohort_source_indices_sha256"],
        checkpoint_sha=first["initialization_checkpoint_sha256"],
        tensor_sha=first["initialization_tensor_sha256"],
        reference_k=128,
    )
    k_artifact["k128_sentinel"] = sentinel
    k_by_iteration = {
        measurement["candidate_k"]: measurement
        for measurement in k_artifact["measurements"]
    }
    k_artifact["selection"] = select_k_measurements(
        k_by_iteration,
        k128_sentinel=sentinel,
    )
    assert k_artifact["selection"]["selected_k"] == 4
    assert k_artifact["selection"]["k128_sentinel_used"] is False
    atomic_write_json(k_path, k_artifact, canonical=True)
    result_path = root / entry["outputs"][-1]
    result = read_json(result_path)
    result["k_selection"] = k_artifact["selection"]
    atomic_write_json(result_path, result, canonical=True)
    _refresh_row_hashes(root, row_index=0)

    with pytest.raises(
        PerfectDiodeConv3HparamValidationError,
        match="present if and only if",
    ):
        materialize_study(
            template_path=DEFAULT_TEMPLATE,
            tk_selection_path=selection_path,
            tk_selection_sha256=sha256_file(selection_path),
            output_dir=tmp_path / "lr-unnecessary-k128",
        )


def test_manifest_freezes_high_grid_expansion_routes_and_shared_assets(
    tmp_path: Path,
) -> None:
    spec = _materialize(tmp_path)
    manifest = build_surface_manifest(
        spec,
        shared_asset_hashes=_shared_assets(spec),
        execution_source=_execution_source(tmp_path),
    )

    assert manifest["surface_count"] == 6
    assert manifest["eligible_surface_count"] == 6
    assert manifest["zero_work_surface_count"] == 0
    assert manifest["maximum_core_cells"] == 54
    assert manifest["maximum_candidate_cells_after_one_expansion"] == 96
    assert manifest["long_confirmation"] is False
    assert manifest["public_stage_sequence"] == list(PUBLIC_STAGE_SEQUENCE)
    assert manifest["shared_asset_hashes"] == _shared_assets(spec)
    assert manifest["execution_source"] == _execution_source(tmp_path)
    assert manifest["relative_paths_resolve_from"] == "surface_manifest_parent"
    by_id = {surface["surface_id"]: surface for surface in manifest["surfaces"]}
    assert by_id["conv3_baseline_v1_c1--sgd"]["host"] == "main"
    assert by_id["conv3_baseline_v1_c1--adam"]["host"] == "main"
    assert by_id["conv3_ours_v4_c1--sgd"]["host"] == "akibscomputer"
    assert by_id["conv3_ours_v4_c1--adam"]["host"] == "akibscomputer"
    assert by_id["conv3_legacy_v4_c0p25--sgd"]["host"] == "main"
    assert by_id["conv3_legacy_v4_c0p25--adam"]["host"] == "main"
    for surface in manifest["surfaces"]:
        assert surface["execution_source"] == manifest["execution_source"]
        assert surface["output_path"] == f"surfaces/{surface['surface_id']}"
        assert surface["completion_path"] == (
            f"surfaces/{surface['surface_id']}/completion.json"
        )
        assert surface["finalization_path"] == (
            f"surfaces/{surface['surface_id']}/stages/finalize_lr/result.json"
        )
        assert surface["execution_environment_sha256"] == (
            "6" * 64 if surface["host"] == "main" else "9" * 64
        )
    for scheme in ("baseline", "ours"):
        surface = by_id[f"conv3_{scheme}_{'v1_c1' if scheme == 'baseline' else 'v4_c1'}--sgd"]
        policy = surface["rho_policy"]
        assert tuple(policy["rho_conv"]) == FIXED_HIGH_RHO_CONV
        assert tuple(policy["rho_dense"]) == FIXED_HIGH_RHO_DENSE
        assert len(policy["core_cells"]) == len(FIXED_HIGH_GRID) == 9
        assert policy["automatic_lower_center_search"] is False
        assert surface["maximum_expansion_waves"] == 1
        assert surface["maximum_cells"] == 16
    legacy = by_id["conv3_legacy_v4_c0p25--adam"]["rho_policy"]
    assert legacy["core_mode"] == "adaptive_safe_center_factor_three_3x3"
    assert legacy["maximum_center_attempts"] == 6

    preflight = representative_preflight_plan(
        manifest,
        spec=spec,
        host="main",
    )
    assert preflight == {
        "status": "required",
        "stage": "preflight_canary",
        "same_public_runner": True,
        "manifest_id": manifest["manifest_id"],
        "surface_id": "conv3_baseline_v1_c1--adam",
        "row_id": "conv3_baseline_v1_c1",
        "optimizer": "adam",
        "host": "main",
        "rho_conv": 0.009,
        "rho_dense": 0.03,
        "steps": CANARY_STEPS,
        "restart_from_shared_initialization": True,
        "shared_asset_hashes": _shared_assets(spec),
        "execution_source": _execution_source(tmp_path),
        "execution_environment_sha256": "6" * 64,
        "official_test_read": False,
    }
    akib_preflight = representative_preflight_plan(
        manifest,
        spec=spec,
        host="akibscomputer",
    )
    assert akib_preflight["surface_id"] == "conv3_ours_v4_c1--adam"
    assert akib_preflight["host"] == "akibscomputer"
    assert akib_preflight["rho_conv"] == 0.009
    assert akib_preflight["rho_dense"] == 0.03
    assert {
        (cell["rho_conv"], cell["rho_dense"])
        for cell in by_id["conv3_baseline_v1_c1--adam"]["rho_policy"][
            "core_cells"
        ]
    } == set(FIXED_HIGH_GRID)
    assert validate_surface_manifest(manifest, spec=spec) == manifest
    tampered = copy.deepcopy(manifest)
    tampered["surfaces"][0]["host"] = "akibscomputer"
    with pytest.raises(
        PerfectDiodeConv3HparamValidationError, match="manifest_id"
    ):
        validate_surface_manifest(tampered, spec=spec)

    self_consistent_tamper = copy.deepcopy(manifest)
    self_consistent_tamper["surfaces"][0]["rho_policy"]["rho_conv"][0] = 0.01
    self_consistent_tamper["surfaces"][0]["rho_policy"]["core_cells"][0][
        "rho_conv"
    ] = 0.01
    self_consistent_tamper.pop("manifest_id")
    self_consistent_tamper["manifest_id"] = (
        "pdlrmanifest_" + sha256_json(self_consistent_tamper)
    )
    with pytest.raises(
        PerfectDiodeConv3HparamValidationError,
        match=r"rho_policy to be exactly",
    ):
        validate_surface_manifest(self_consistent_tamper, spec=spec)

    zero_work_spoof = copy.deepcopy(manifest)
    for surface in zero_work_spoof["surfaces"]:
        surface["lr_eligible"] = False
        surface["zero_work"] = True
        surface["zero_work_reason"] = "fabricated_zero_work"
        surface["inference_iterations"] = None
        surface["training_iterations"] = None
        surface["rho_policy"] = None
    zero_work_spoof["eligible_surface_count"] = 0
    zero_work_spoof["zero_work_surface_count"] = 6
    zero_work_spoof["maximum_core_cells"] = 0
    zero_work_spoof["maximum_candidate_cells_after_one_expansion"] = 0
    zero_work_spoof.pop("manifest_id")
    zero_work_spoof["manifest_id"] = (
        "pdlrmanifest_" + sha256_json(zero_work_spoof)
    )
    with pytest.raises(
        PerfectDiodeConv3HparamValidationError,
        match=r"\.lr_eligible",
    ):
        validate_surface_manifest(zero_work_spoof, spec=spec)


def test_surface_routes_must_match_same_scheme_tk_host(tmp_path: Path) -> None:
    spec = _materialize(tmp_path)
    routes = copy.deepcopy(
        spec.data["execution"]["default_surface_routes"]
    )
    routes["conv3_ours_v4_c1--adam"] = "main"

    with pytest.raises(
        PerfectDiodeConv3HparamValidationError,
        match="same-scheme T/K execution host",
    ):
        build_surface_manifest(
            spec,
            shared_asset_hashes=_shared_assets(spec),
            execution_source=_execution_source(tmp_path),
            routes=routes,
        )


def test_unresolved_tk_row_materializes_two_zero_work_surfaces(
    tmp_path: Path,
) -> None:
    spec = _materialize(
        tmp_path,
        statuses=(
            "selected",
            "unresolved_t_extension_sentinel",
            "selected",
        ),
        use_fixed_operating_point=True,
        selected_ts=(4, 4, 4),
        selected_ks=(4, 4, 4),
    )
    ours = next(row for row in spec.rows if row["scheme"] == "ours")
    assert ours["lr_eligible"] is False
    assert ours["inference_iterations"] is None
    assert ours["training_iterations"] is None
    assert ours["zero_work_reason"] == (
        "upstream_tk_unresolved_t_extension_sentinel"
    )
    selected_rows = [row for row in spec.rows if row["lr_eligible"]]
    assert {
        (row["inference_iterations"], row["training_iterations"])
        for row in selected_rows
    } == {(8, 8)}

    manifest = build_surface_manifest(
        spec,
        shared_asset_hashes=_shared_assets(spec),
        execution_source=_execution_source(tmp_path),
    )
    zero = [surface for surface in manifest["surfaces"] if surface["zero_work"]]
    assert manifest["eligible_surface_count"] == 4
    assert manifest["zero_work_surface_count"] == 2
    assert {surface["optimizer"] for surface in zero} == {"sgd", "adam"}
    assert {surface["scheme"] for surface in zero} == {"ours"}
    assert all(surface["rho_policy"] is None for surface in zero)


def test_controller_plan_is_read_only_and_reports_no_launch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _materialize(
        tmp_path,
        use_fixed_operating_point=True,
        selected_ts=(4, 4, 4),
        selected_ks=(4, 4, 4),
    )

    assert main(["plan", "--study", str(tmp_path / "lr" / "study.resolved.json")]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "planned"
    assert output["eligible_surfaces"] == 6
    assert output["launched_jobs"] == 0
    assert output["long_confirmation"] is False
    assert output["operating_point"]["inference_iterations"] == 8
    assert output["operating_point"]["training_iterations"] == 8
    assert [
        (row["T"], row["K"]) for row in output["rows"]
    ] == [(8, 8), (8, 8), (8, 8)]
    assert [
        (row["diagnostic_T"], row["diagnostic_K"])
        for row in output["rows"]
    ] == [(4, 4), (4, 4), (4, 4)]


def test_controller_derives_all_nine_shared_asset_hashes_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _materialize(tmp_path)
    asset_dir = tmp_path / "lr" / "upstream_tk" / "shared_assets"
    metadata = read_json(asset_dir / "assets.json")
    epoch_hashes = ("e" * 64, "f" * 64, "0" * 64)
    bundle = SimpleNamespace(
        train_indices=tuple(metadata["split"]["train_indices"]),
        validation_indices=tuple(metadata["split"]["validation_indices"]),
        train_indices_hash=metadata["split"]["train_indices_sha256"],
        validation_indices_hash=metadata["split"][
            "validation_indices_sha256"
        ],
        train_batch_order_hashes=lambda *, num_epochs: (
            epoch_hashes if num_epochs == 3 else ()
        ),
    )
    loaded = SimpleNamespace(
        train_indices_sha256=bundle.train_indices_hash,
        validation_indices_sha256=bundle.validation_indices_hash,
        checkpoint_sha256=metadata["initialization"]["sha256"],
        parameter_tensor_sha256=metadata["initialization"][
            "parameter_tensor_sha256"
        ],
        t_indices_sha256=metadata["cohorts"]["t"][
            "source_indices_sha256"
        ],
        k_indices_sha256=metadata["cohorts"]["k"][
            "source_indices_sha256"
        ],
    )
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "load_tk_shared_assets",
        lambda *args, **kwargs: loaded,
    )
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "build_loader_bundle",
        lambda *args, **kwargs: bundle,
    )
    output = tmp_path / "lr" / "shared_asset_hashes.json"
    argv = [
        "derive-shared-assets",
        "--study",
        str(tmp_path / "lr" / "study.resolved.json"),
        "--data-root",
        str(tmp_path / "mnist"),
        "--output",
        str(output),
    ]

    assert main(argv) == 0
    capsys.readouterr()
    first_bytes = output.read_bytes()
    first_mtime = output.stat().st_mtime_ns
    value = read_json(output)
    assert value == _shared_assets(
        PerfectDiodeConv3HparamStudySpec.from_path(
            tmp_path / "lr" / "study.resolved.json"
        )
    )
    assert len(value) == 9

    assert main(argv) == 0
    capsys.readouterr()
    assert output.read_bytes() == first_bytes
    assert output.stat().st_mtime_ns == first_mtime

    bad_bundle = copy.copy(bundle)
    bad_bundle.train_indices_hash = "1" * 64
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "build_loader_bundle",
        lambda *args, **kwargs: bad_bundle,
    )
    with pytest.raises(
        RuntimeError, match="actual LR loader"
    ):
        main(
            [
                *argv[:-1],
                str(tmp_path / "lr" / "tampered_hashes.json"),
            ]
        )


def test_controller_captures_host_receipts_and_builds_canonical_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    receipts = {
        "main": _host_environment_receipt(
            "main", torch_version="2.5.1+cu121"
        ),
        "akibscomputer": _host_environment_receipt(
            "akibscomputer", torch_version="2.5.1"
        ),
    }
    monkeypatch.setattr(
        "experiments.run_mnist_conv_perfectdiode_conv3_hparam_v2."
        "observe_execution_environment",
        lambda host: copy.deepcopy(receipts[host]),
    )
    main_path = tmp_path / "main.environment.json"
    akib_path = tmp_path / "akibscomputer.environment.json"
    contract_path = tmp_path / "environment.contract.json"
    for host, path in (
        ("main", main_path),
        ("akibscomputer", akib_path),
    ):
        assert (
            main(
                [
                    "capture-environment",
                    "--host",
                    host,
                    "--output",
                    str(path),
                ]
            )
            == 0
        )
        capsys.readouterr()
    argv = [
        "build-environment-contract",
        "--main-receipt",
        str(main_path),
        "--akibscomputer-receipt",
        str(akib_path),
        "--output",
        str(contract_path),
    ]

    assert main(argv) == 0
    capsys.readouterr()
    first_bytes = contract_path.read_bytes()
    first_mtime = contract_path.stat().st_mtime_ns
    contract = read_json(contract_path)
    assert contract["host_receipts"] == receipts
    assert (
        contract["host_runtime_policy"][
            "cross_host_software_equality_required"
        ]
        is False
    )

    assert main(argv) == 0
    capsys.readouterr()
    assert contract_path.read_bytes() == first_bytes
    assert contract_path.stat().st_mtime_ns == first_mtime

    tampered = read_json(main_path)
    tampered["torch"]["version"] = "9.9.9"
    atomic_write_json(main_path, tampered, canonical=True)
    with pytest.raises(ValueError, match="canonically identical"):
        main(argv)
