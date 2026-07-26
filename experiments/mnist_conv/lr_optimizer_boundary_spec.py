"""Immutable contract for the Conv2 SGD/Adam boundary-extension diagnostic.

This schema is deliberately separate from ``mnist-conv-lr-study/v1`` through
``v6``.  It is content addressed, but it does not reinterpret or mutate any
completed study.  The only variable scientific inputs are the hashes that bind
the six reused SGD cells to their existing immutable artifacts.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .identity import sha256_json
from .specs import (
    ADAM_OPTIMIZER_V7,
    OPTIMIZER_BOUNDARY_PROTOCOL_ID,
    SGD_OPTIMIZER_V7,
    SpecValidationError,
    normalize_optimizer_v7,
)


OPTIMIZER_BOUNDARY_STUDY_SCHEMA_VERSION = (
    "mnist-conv-lr-optimizer-boundary/v1"
)
OPTIMIZER_BOUNDARY_STUDY_ID_SCHEMA = (
    "mnist-conv-lr-optimizer-boundary-id/v1"
)

OPTIMIZER_BOUNDARY_ROWS: tuple[dict[str, Any], ...] = (
    {
        "row_id": "conv2_baseline_v1_c1",
        "architecture": "conv2",
        "scheme": "baseline",
        "run_name": "mnist_bp_amp_v1_c1",
        "voltage_amp": 1.0,
        "current_amp": 1.0,
        "input_gain": 253.302230835,
        "inference_iterations": 16,
        "training_iterations": 6,
    },
    {
        "row_id": "conv2_ours_v4_c1",
        "architecture": "conv2",
        "scheme": "ours",
        "run_name": "mnist_bp_amp_v4_c1",
        "voltage_amp": 4.0,
        "current_amp": 1.0,
        "input_gain": 716.3439331055,
        "inference_iterations": 24,
        "training_iterations": 6,
    },
    {
        "row_id": "conv2_legacy_v4_c0p25",
        "architecture": "conv2",
        "scheme": "legacy",
        "run_name": "mnist_bp_amp_v4_c0p25",
        "voltage_amp": 4.0,
        "current_amp": 0.25,
        "input_gain": 661.4369506836,
        "inference_iterations": 8,
        "training_iterations": 4,
    },
)

_EXPECTED_REUSE_COORDINATES = {
    ("conv2_baseline_v1_c1", 0.01, 0.01),
    ("conv2_baseline_v1_c1", 0.01, 0.03),
    ("conv2_ours_v4_c1", 0.01, 0.003),
    ("conv2_ours_v4_c1", 0.01, 0.01),
    ("conv2_legacy_v4_c0p25", 0.01, 0.003),
    ("conv2_legacy_v4_c0p25", 0.01, 0.01),
}

_EXPECTED_REUSE_SOURCE_BY_COORDINATE = {
    ("conv2_baseline_v1_c1", 0.01, 0.01): (
        "conv2_baseline_v1_c1--grid-c03-d01",
        "lrstudy_c735d2beead9fcbadda89a65ffe86257"
        "da53f15e6cbfab7b4f04b14b08ff6c2f",
    ),
    ("conv2_baseline_v1_c1", 0.01, 0.03): (
        "conv2_baseline_v1_c1--grid-c03-d02",
        "lrstudy_c735d2beead9fcbadda89a65ffe86257"
        "da53f15e6cbfab7b4f04b14b08ff6c2f",
    ),
    ("conv2_ours_v4_c1", 0.01, 0.003): (
        "conv2_ours_v4_c1--scheme-rho--c03-d00",
        "lrsweep_07eed332608b684ff77bcf994071b9023"
        "0fd877345b2e5132d9144cc4fa40435",
    ),
    ("conv2_ours_v4_c1", 0.01, 0.01): (
        "conv2_ours_v4_c1--scheme-rho--c03-d01",
        "lrsweep_07eed332608b684ff77bcf994071b9023"
        "0fd877345b2e5132d9144cc4fa40435",
    ),
    ("conv2_legacy_v4_c0p25", 0.01, 0.003): (
        "conv2_legacy_v4_c0p25--scheme-rho--c03-d00",
        "lrsweep_07eed332608b684ff77bcf994071b9023"
        "0fd877345b2e5132d9144cc4fa40435",
    ),
    ("conv2_legacy_v4_c0p25", 0.01, 0.01): (
        "conv2_legacy_v4_c0p25--scheme-rho--c03-d01",
        "lrsweep_07eed332608b684ff77bcf994071b9023"
        "0fd877345b2e5132d9144cc4fa40435",
    ),
}

_TOP_LEVEL_KEYS = {
    "schema_version",
    "name",
    "protocol_id",
    "status",
    "dataset",
    "model",
    "solver",
    "optimizer_arms",
    "reuse",
    "probe",
    "rho_grid",
    "candidate_training",
    "selection",
    "sentinel",
    "bias_confirmation",
    "execution",
    "artifacts",
    "rows",
}

_FROZEN_SECTIONS: dict[str, Any] = {
    "status": {
        "study_role": "ordinary_mnist_optimization_diagnostic",
        "final_training_authorized": False,
        "replaces_medium_affine_handoff": False,
        "adam_scope": "local_high_rho_diagnostic",
    },
    "dataset": {
        "name": "mnist",
        "variant": "ordinary",
        "input_shape": [2, 28, 28],
        "normalization": {
            "mean": 0.1307,
            "std": 0.3081,
            "scale": 0.3,
        },
        "affine": {
            "enabled": False,
            "preset": "ordinary_identity",
            "degrees": 0.0,
            "translate": [0.0, 0.0],
            "scale": [1.0, 1.0],
            "shear": 0.0,
            "seed": 1729,
            "interpolation": "bilinear",
            "fill": 0.0,
            "deterministic_by_original_index": True,
        },
        "train": {
            "source": "mnist_train",
            "size": 55000,
            "batch_size": 16,
            "drop_last": False,
            "shuffle_method": "torch_randperm",
            "shuffle_seed": 0,
            "reset_shuffle_identically_before_each_candidate": True,
            "steps_per_epoch": 3438,
        },
        "validation": {
            "source": "mnist_train",
            "size": 5000,
            "class_count": 10,
            "samples_per_class": 500,
            "stratified": True,
            "selection_method": "classwise_torch_randperm",
            "split_seed": 0,
            "batch_size": 128,
            "store_exact_indices": True,
            "index_hash": "sha256",
        },
        "official_test": {
            "enabled": False,
            "read_allowed": False,
        },
    },
    "model": {
        "model_seed": 0,
        "architectures": {
            "conv2": {
                "channels": [64, 128],
                "kernel_sizes": [3, 3],
                "strides": [2, 2],
                "paddings": [1, 1],
                "pooling": "none",
                "output_dim": 20,
            },
        },
        "non_linearity": "hard_sigmoid",
        "quadratic_diode_param": {},
        "exponential_diode_param": {},
        "hard_sigmoid": {
            "g_on": 100.0,
            "g_off": 0.0,
            "v_off": 4.0,
        },
        "conductance_bounds": [0.0, 100.0],
        "weight_initialization": "kaiming_uniform",
        "weight_gains": 1.0,
        "paired_output_loss": "squared_error_paired_20",
        "amplification": "fixed",
        "input_gain_calibration": {
            "source_dataset": "deterministic_medium_affine_mnist",
            "settling_iterations": 64,
            "target_first_hidden_saturation": 0.3,
            "recalibrate_for_ordinary_mnist": False,
        },
        "shared_initialization": (
            "one_seed0_conv2_checkpoint_shared_across_all_schemes_and_candidates"
        ),
        "reset_global_name_counters_before_build": True,
    },
    "solver": {
        "energy_mode": "asynchronous",
        "fixed_step_minimization": True,
        "batch_state_policy": "reset_each_batch",
        "minimizer": {
            "double_diode_updater": "CustomExponentialDoubleDiodeUpdater",
            "adaptive_equilibrium": False,
            "overrelaxation_factor": 1.1,
            "single_diode_updater": "custom",
            "iv_data_path": None,
            "experimental_damping": 0.5,
            "experimental_newton_max_steps": 100,
            "settings": {
                "rel_tol": 1e-5,
                "vn_tol": 1e-6,
                "use_polish": True,
                "max_newton_iters": 32,
                "z_thresh": 1e10,
                "exp_clip": 1e5,
                "dynamic_polish": True,
                "overrelaxation_reject_steps": False,
                "overrelaxation_reject_max_tries": 3,
                "overrelaxation_reject_shrink": 0.5,
                "overrelaxation_reject_eps": 0.0,
                "experimental_exponential_newton_tol_progressive": True,
                "experimental_exponential_newton_tol_start": 1e-5,
                "experimental_exponential_newton_tol_end": 1e-5,
                "experimental_exponential_newton_tol_switch_hi": 0.01,
                "experimental_exponential_newton_tol_switch_lo": 0.0005,
            },
        },
    },
    "optimizer_arms": {
        "sgd": SGD_OPTIMIZER_V7,
        "adam": ADAM_OPTIMIZER_V7,
    },
    "probe": {
        "normalization_minibatches": 32,
        "replay_minibatches": 8,
        "batch_size": 16,
        "nominal_learning_rate": 1.0,
        "weight_unit": (
            "median_batch_rms_optimizer_proposal_over_initial_weight_rms"
        ),
        "bias_unit": (
            "q90_batch_rms_optimizer_proposal_over_attached_initial_weight_rms"
        ),
        "shared_frozen_minibatches": True,
        "normalization_is_optimizer_specific": True,
        "sgd_units_reused_for_adam": False,
        "adam_shadow_state_policy": "fresh_empty_state_per_minibatch",
        "replay_at": ["initialization", "each_epoch"],
        "restore_after_each_shadow": {
            "parameters": True,
            "optimizer_state": True,
            "moments": True,
            "variances": True,
            "step_counters": True,
        },
        "reject_non_finite_optimizer_state": True,
        "recorded_metrics": [
            "achieved_weight_rho",
            "bias_weight_step_ratio",
            "projection_efficiency",
            "weight_displacement",
            "hidden_saturation",
            "validation_loss",
            "validation_accuracy",
        ],
    },
    "rho_grid": {
        "rho_conv_main": [0.01, 0.015, 0.03],
        "rho_dense_by_scheme": {
            "baseline": [0.01, 0.03],
            "ours": [0.003, 0.01],
            "legacy": [0.003, 0.01],
        },
        "rho_conv_sentinel": 0.1,
        "main_cell_count_per_optimizer": 18,
        "reused_sgd_main_run_count": 6,
        "new_sgd_main_run_count": 12,
        "new_adam_main_run_count": 18,
        "new_main_run_count": 30,
        "maximum_sentinel_run_count": 12,
        "normal_new_run_count_including_confirmations": 36,
        "maximum_new_run_count_including_sentinels_and_confirmations": 48,
    },
    "candidate_training": {
        "model_seeds": [0],
        "epochs": 5,
        "training_examples": 55000,
        "batch_size": 16,
        "steps_per_epoch": 3438,
        "total_steps": 17190,
        "schedule": "constant",
        "scheduler_enabled": False,
        "one_optimizer_group_per_scientific_parameter": True,
        "explicit_learning_rate_per_group": True,
        "real_adam_state_policy": "preserve_across_optimizer_steps",
        "non_finite_optimizer_state_policy": "immediate_safety_failure",
        "official_test_evaluation": False,
        "complete_entry_policy": "skip_immutable",
        "incomplete_adam_policy": "restart_from_initialization",
    },
    "selection": {
        "independent_by": ["scheme", "optimizer"],
        "primary_metric": "minimum_final_validation_loss",
        "plateau_relative_to_minimum": 0.02,
        "tie_breakers": [
            "higher_final_validation_accuracy",
            "higher_projection_efficiency",
            "lower_maximum_rho",
            "lower_target_sum",
        ],
    },
    "sentinel": {
        "trigger": "plateau_reaches_rho_conv_0.03",
        "rho_conv": 0.1,
        "run_both_dense_slices": True,
        "high_status_if_plateau_reaches_sentinel": "unbracketed_high",
        "adam_low_edge": 0.01,
        "adam_low_status": "unbracketed_low",
        "extend_beyond_sentinel": False,
    },
    "bias_confirmation": {
        "one_per_scheme_optimizer": True,
        "unchanged_weight_learning_rates": True,
        "initial_bias_rho_cap": 0.001,
        "formula": "min(attached_lr,0.001/q90_bias_unit)",
        "cap_never_increases_bias_lr": True,
        "confirmation_count": 6,
    },
    "execution": {
        "local_role": "tests_and_probes",
        "production_location": "jean_zay",
        "slurm_account": "fmu@v100",
        "constraint": "v100-32g",
        "gpus_per_pack": 1,
        "cpus_per_pack": 16,
        "time_limit_hours": 20,
        "production_packs": {
            "baseline": {
                "new_main_runs": 10,
                "estimated_runtime_hours": [7, 8],
            },
            "ours": {
                "new_main_runs": 10,
                "estimated_runtime_hours": [9, 11],
            },
            "legacy": {
                "new_main_runs": 10,
                "estimated_runtime_hours": [3, 4],
            },
        },
        "benchmark_concurrency": [1, 2, 4, 8, 12, 16],
        "memory_headroom_fraction": 0.1,
        "minimum_relative_to_best_throughput": 0.9,
        "packing_rule": (
            "largest_safe_width_with_headroom_and_throughput_threshold"
        ),
    },
    "artifacts": {
        "hash": "sha256",
        "immutable": True,
        "completion_marker_written_last": True,
        "optimizer_parameters_required": True,
        "probe_hash_required": True,
        "parent_hashes_required": True,
        "raw_per_parameter_learning_rates_required": True,
        "minibatch_hashes_required": True,
        "checkpoint_hashes_required": True,
        "code_fingerprint_required": True,
        "official_test_read": False,
        "plots_backend": "matplotlib",
        "plots": [
            "final_validation_heatmaps",
            "epochwise_effective_rho",
            "epochwise_bias_weight_step_ratio",
        ],
    },
}


def _error(expected: str, provided: Any, path: str) -> SpecValidationError:
    return SpecValidationError(
        f"Expected {path} to be {expected}. Provided value: {provided!r}."
    )


def _strict_json_loads(text: str, source: Path) -> Any:
    def reject_constant(value: str) -> Any:
        raise _error("a finite JSON number", value, str(source))

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _error("JSON objects with unique keys", key, str(source))
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except json.JSONDecodeError as exc:
        raise SpecValidationError(
            f"Expected {source} to contain valid strict JSON. "
            f"Provided error: {exc}."
        ) from exc


def _reject_non_finite(value: Any, path: str = "study") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise _error("a finite JSON value", value, path)
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


def _assert_exact(expected: Any, provided: Any, path: str) -> None:
    """Recursively compare without Python's bool/int equivalence."""

    if type(expected) is not type(provided):
        raise _error(f"exactly {expected!r}", provided, path)
    if isinstance(expected, dict):
        if set(expected) != set(provided):
            missing = sorted(set(expected) - set(provided))
            extra = sorted(set(provided) - set(expected))
            raise _error(
                f"an object with exactly keys {sorted(expected)!r}",
                {"missing": missing, "extra": extra},
                path,
            )
        for key in expected:
            _assert_exact(expected[key], provided[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if len(expected) != len(provided):
            raise _error(f"a list of length {len(expected)}", provided, path)
        for index, (expected_item, provided_item) in enumerate(
            zip(expected, provided)
        ):
            _assert_exact(expected_item, provided_item, f"{path}[{index}]")
        return
    if expected != provided:
        raise _error(f"exactly {expected!r}", provided, path)


def _sha256(value: Any, path: str, *, prefix: str = "") -> str:
    pattern = re.escape(prefix) + r"[0-9a-f]{64}"
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        description = (
            f"{prefix!r} followed by 64 lowercase hexadecimal characters"
            if prefix
            else "a lowercase SHA-256 digest"
        )
        raise _error(description, value, path)
    return value


def _normalize_reuse(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error("a JSON object", value, "study.reuse")
    reuse = copy.deepcopy(dict(value))
    expected_keys = {"mode", "mismatch_policy", "optimizer", "cells"}
    if set(reuse) != expected_keys:
        raise _error(
            f"an object with exactly keys {sorted(expected_keys)!r}",
            {
                "missing": sorted(expected_keys - set(reuse)),
                "extra": sorted(set(reuse) - expected_keys),
            },
            "study.reuse",
        )
    if reuse["mode"] != "hash_verify_existing_sgd_cells":
        raise _error(
            "exactly 'hash_verify_existing_sgd_cells'",
            reuse["mode"],
            "study.reuse.mode",
        )
    if reuse["mismatch_policy"] != "abort_without_rerun":
        raise _error(
            "exactly 'abort_without_rerun'",
            reuse["mismatch_policy"],
            "study.reuse.mismatch_policy",
        )
    if reuse["optimizer"] != "SGD":
        raise _error("exactly 'SGD'", reuse["optimizer"], "study.reuse.optimizer")
    cells = reuse["cells"]
    if not isinstance(cells, list) or len(cells) != 6:
        raise _error("a list of exactly six reused cells", cells, "study.reuse.cells")

    normalized_cells: list[dict[str, Any]] = []
    coordinates: set[tuple[str, float, float]] = set()
    source_entry_ids: set[str] = set()
    for index, raw in enumerate(cells):
        path = f"study.reuse.cells[{index}]"
        if not isinstance(raw, Mapping):
            raise _error("a JSON object", raw, path)
        cell = copy.deepcopy(dict(raw))
        expected_cell_keys = {
            "row_id",
            "rho_conv",
            "rho_dense",
            "source_entry_id",
            "source_collection_id",
            "expected_completed_steps",
            "run_spec_sha256",
            "candidate_summary_sha256",
            "completion_sha256",
            "completion_bytes",
            "outputs",
        }
        if set(cell) != expected_cell_keys:
            raise _error(
                f"an object with exactly keys {sorted(expected_cell_keys)!r}",
                {
                    "missing": sorted(expected_cell_keys - set(cell)),
                    "extra": sorted(set(cell) - expected_cell_keys),
                },
                path,
            )
        row_id = cell["row_id"]
        if row_id not in {row["row_id"] for row in OPTIMIZER_BOUNDARY_ROWS}:
            raise _error(
                "one of the frozen Conv2 row ids",
                row_id,
                f"{path}.row_id",
            )
        for field in ("rho_conv", "rho_dense"):
            value = cell[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _error("a finite number", value, f"{path}.{field}")
            cell[field] = float(value)
        coordinate = (row_id, cell["rho_conv"], cell["rho_dense"])
        if coordinate in coordinates:
            raise _error("a unique reused-cell coordinate", coordinate, path)
        coordinates.add(coordinate)

        source_entry_id = cell["source_entry_id"]
        if (
            not isinstance(source_entry_id, str)
            or not source_entry_id
            or "/" in source_entry_id
            or "\\" in source_entry_id
            or source_entry_id in {".", ".."}
        ):
            raise _error(
                "a non-empty portable entry id without path separators",
                source_entry_id,
                f"{path}.source_entry_id",
            )
        if source_entry_id in source_entry_ids:
            raise _error(
                "a unique source entry id",
                source_entry_id,
                f"{path}.source_entry_id",
            )
        source_entry_ids.add(source_entry_id)

        collection_id = cell["source_collection_id"]
        if not isinstance(collection_id, str) or re.fullmatch(
            r"(?:lrstudy_|lrsweep_)[0-9a-f]{64}",
            collection_id,
        ) is None:
            raise _error(
                "lrstudy_ or lrsweep_ followed by 64 lowercase hexadecimal "
                "characters",
                collection_id,
                f"{path}.source_collection_id",
            )
        expected_entry_id, expected_collection_id = (
            _EXPECTED_REUSE_SOURCE_BY_COORDINATE[coordinate]
        )
        if source_entry_id != expected_entry_id:
            raise _error(
                f"exactly {expected_entry_id!r} for coordinate {coordinate!r}",
                source_entry_id,
                f"{path}.source_entry_id",
            )
        if collection_id != expected_collection_id:
            raise _error(
                f"exactly {expected_collection_id!r} for coordinate "
                f"{coordinate!r}",
                collection_id,
                f"{path}.source_collection_id",
            )

        completed_steps = cell["expected_completed_steps"]
        if (
            isinstance(completed_steps, bool)
            or not isinstance(completed_steps, int)
            or completed_steps != 17190
        ):
            raise _error(
                "exactly 17190",
                completed_steps,
                f"{path}.expected_completed_steps",
            )
        for field in (
            "run_spec_sha256",
            "candidate_summary_sha256",
            "completion_sha256",
        ):
            cell[field] = _sha256(cell[field], f"{path}.{field}")

        completion_bytes = cell["completion_bytes"]
        if (
            isinstance(completion_bytes, bool)
            or not isinstance(completion_bytes, int)
            or completion_bytes <= 0
        ):
            raise _error(
                "a positive integer byte count",
                completion_bytes,
                f"{path}.completion_bytes",
            )

        outputs = cell["outputs"]
        if not isinstance(outputs, list) or len(outputs) != 8:
            raise _error(
                "the eight completion-bound output files",
                outputs,
                f"{path}.outputs",
            )
        normalized_outputs: list[dict[str, Any]] = []
        output_names: set[str] = set()
        for output_index, raw_output in enumerate(outputs):
            output_path = f"{path}.outputs[{output_index}]"
            if not isinstance(raw_output, Mapping):
                raise _error("a JSON object", raw_output, output_path)
            output = copy.deepcopy(dict(raw_output))
            if set(output) != {"name", "bytes", "sha256"}:
                raise _error(
                    "an object with exactly keys ['bytes', 'name', 'sha256']",
                    output,
                    output_path,
                )
            name = output["name"]
            if (
                not isinstance(name, str)
                or not name
                or "/" in name
                or "\\" in name
                or name in {".", "..", "complete.json"}
            ):
                raise _error(
                    "a portable output basename other than complete.json",
                    name,
                    f"{output_path}.name",
                )
            if name in output_names:
                raise _error(
                    "a unique output basename",
                    name,
                    f"{output_path}.name",
                )
            output_names.add(name)
            byte_count = output["bytes"]
            if (
                isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count <= 0
            ):
                raise _error(
                    "a positive integer byte count",
                    byte_count,
                    f"{output_path}.bytes",
                )
            normalized_outputs.append(
                {
                    "name": name,
                    "bytes": byte_count,
                    "sha256": _sha256(
                        output["sha256"],
                        f"{output_path}.sha256",
                    ),
                }
            )

        fixed_output_names = {
            "best_validation.pt",
            "final.pt",
            "minibatches.json",
            "parameter_diagnostics.csv",
            "step_log.csv",
            "summary.json",
            "validation.json",
        }
        if not fixed_output_names.issubset(output_names):
            raise _error(
                "all seven frozen payload basenames plus one run_spec.v5.json "
                "or run_spec.v6.json",
                sorted(output_names),
                f"{path}.outputs",
            )
        run_spec_names = {
            name
            for name in output_names
            if re.fullmatch(r"run_spec\.v[56]\.json", name)
        }
        if len(run_spec_names) != 1 or output_names != (
            fixed_output_names | run_spec_names
        ):
            raise _error(
                "all seven frozen payload basenames plus exactly one "
                "run_spec.v5.json or run_spec.v6.json",
                sorted(output_names),
                f"{path}.outputs",
            )
        output_by_name = {output["name"]: output for output in normalized_outputs}
        run_spec_name = next(iter(run_spec_names))
        if cell["run_spec_sha256"] != output_by_name[run_spec_name]["sha256"]:
            raise _error(
                "equal to the run-spec output SHA-256",
                cell["run_spec_sha256"],
                f"{path}.run_spec_sha256",
            )
        if (
            cell["candidate_summary_sha256"]
            != output_by_name["summary.json"]["sha256"]
        ):
            raise _error(
                "equal to the summary.json output SHA-256",
                cell["candidate_summary_sha256"],
                f"{path}.candidate_summary_sha256",
            )
        normalized_outputs.sort(key=lambda output: output["name"])
        cell["outputs"] = normalized_outputs
        normalized_cells.append(cell)

    if coordinates != _EXPECTED_REUSE_COORDINATES:
        raise _error(
            f"exactly the six frozen rho_conv=0.01 coordinates "
            f"{sorted(_EXPECTED_REUSE_COORDINATES)!r}",
            sorted(coordinates),
            "study.reuse.cells",
        )
    normalized_cells.sort(
        key=lambda cell: (
            cell["row_id"],
            cell["rho_conv"],
            cell["rho_dense"],
        )
    )
    return {
        "mode": "hash_verify_existing_sgd_cells",
        "mismatch_policy": "abort_without_rerun",
        "optimizer": "SGD",
        "cells": normalized_cells,
    }


@dataclass(frozen=True)
class OptimizerBoundaryStudySpec:
    """Validated, content-addressed boundary-extension study."""

    data: dict[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OptimizerBoundaryStudySpec":
        if not isinstance(value, Mapping):
            raise _error("a JSON object", value, "study")
        study = copy.deepcopy(dict(value))
        _reject_non_finite(study)
        missing = sorted(_TOP_LEVEL_KEYS - set(study))
        extra = sorted(set(study) - _TOP_LEVEL_KEYS)
        if missing or extra:
            raise _error(
                f"an object with exactly keys {sorted(_TOP_LEVEL_KEYS)!r}",
                {"missing": missing, "extra": extra},
                "study",
            )
        if study["schema_version"] != OPTIMIZER_BOUNDARY_STUDY_SCHEMA_VERSION:
            raise _error(
                f"exactly {OPTIMIZER_BOUNDARY_STUDY_SCHEMA_VERSION!r}",
                study["schema_version"],
                "study.schema_version",
            )
        if not isinstance(study["name"], str) or not study["name"].strip():
            raise _error("a non-empty string", study["name"], "study.name")
        study["name"] = study["name"].strip()
        if study["protocol_id"] != OPTIMIZER_BOUNDARY_PROTOCOL_ID:
            raise _error(
                f"exactly {OPTIMIZER_BOUNDARY_PROTOCOL_ID!r}",
                study["protocol_id"],
                "study.protocol_id",
            )

        for section, expected in _FROZEN_SECTIONS.items():
            if section == "optimizer_arms":
                optimizers = study[section]
                if not isinstance(optimizers, Mapping) or set(optimizers) != {
                    "sgd",
                    "adam",
                }:
                    raise _error(
                        "an object with exactly keys ['adam', 'sgd']",
                        optimizers,
                        "study.optimizer_arms",
                    )
                normalized_optimizers = {
                    "sgd": normalize_optimizer_v7(
                        optimizers["sgd"], "study.optimizer_arms.sgd"
                    ),
                    "adam": normalize_optimizer_v7(
                        optimizers["adam"], "study.optimizer_arms.adam"
                    ),
                }
                _assert_exact(
                    expected,
                    normalized_optimizers,
                    "study.optimizer_arms",
                )
                study["optimizer_arms"] = normalized_optimizers
            else:
                _assert_exact(expected, study[section], f"study.{section}")

        study["reuse"] = _normalize_reuse(study["reuse"])
        _assert_exact(
            list(OPTIMIZER_BOUNDARY_ROWS),
            study["rows"],
            "study.rows",
        )
        return cls(study)

    @classmethod
    def from_path(cls, path: str | Path) -> "OptimizerBoundaryStudySpec":
        source = Path(path)
        try:
            text = source.read_text()
        except OSError as exc:
            raise SpecValidationError(
                f"Expected {source} to contain valid JSON. Provided error: {exc}."
            ) from exc
        return cls.from_dict(_strict_json_loads(text, source))

    from_file = from_path

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def identity_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("name")
        return payload

    @property
    def study_id(self) -> str:
        payload = {
            "schema_version": OPTIMIZER_BOUNDARY_STUDY_ID_SCHEMA,
            "study_spec": self.identity_payload(),
        }
        return "lrstudy_" + sha256_json(payload)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.data["rows"])


def optimizer_boundary_study_template(
    reuse_cells: Sequence[Mapping[str, Any]],
    *,
    name: str = "conv2-sgd-adam-boundary-extension-diagnostic",
) -> dict[str, Any]:
    """Return the complete v1 study document around six bound reuse records."""

    document: dict[str, Any] = {
        "schema_version": OPTIMIZER_BOUNDARY_STUDY_SCHEMA_VERSION,
        "name": name,
        "protocol_id": OPTIMIZER_BOUNDARY_PROTOCOL_ID,
        **copy.deepcopy(_FROZEN_SECTIONS),
        "reuse": {
            "mode": "hash_verify_existing_sgd_cells",
            "mismatch_policy": "abort_without_rerun",
            "optimizer": "SGD",
            "cells": [copy.deepcopy(dict(cell)) for cell in reuse_cells],
        },
        "rows": copy.deepcopy(list(OPTIMIZER_BOUNDARY_ROWS)),
    }
    return OptimizerBoundaryStudySpec.from_dict(document).to_dict()


# Short, schema-specific alias for callers.
StudySpec = OptimizerBoundaryStudySpec


__all__ = [
    "OPTIMIZER_BOUNDARY_ROWS",
    "OPTIMIZER_BOUNDARY_STUDY_ID_SCHEMA",
    "OPTIMIZER_BOUNDARY_STUDY_SCHEMA_VERSION",
    "OptimizerBoundaryStudySpec",
    "StudySpec",
    "optimizer_boundary_study_template",
]
