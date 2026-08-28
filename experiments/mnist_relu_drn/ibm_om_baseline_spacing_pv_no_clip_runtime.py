"""CUDA runtime for the exploratory unclipped IBM OM baseline/spacing rerun."""

from __future__ import annotations

from hashlib import sha256
from itertools import product
import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np
import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    fit_weight_reconstruction_scales,
    quad_contrast,
    quad_loading,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection_runtime import (
    _artifact_records,
    _atomic_save_npz,
    _dataset_provenance,
    _evaluate_detailed,
    _fit_calibration,
    _input,
    _mapping_invariants,
    _prepare_assignment,
    _sampler_python,
    _selected_candidate,
    _teacher_predictions,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv import (
    BaselineSpacingMapping,
    RawXConductanceEmbedding,
    UNCLIPPED_STUDY_CONDUCTANCE_CEILING,
    UNCLIPPED_STUDY_CONDUCTANCE_PER_RAW_X,
    UNCLIPPED_STUDY_RAW_X_MAXIMUM,
    UNCLIPPED_STUDY_RAW_X_MINIMUM,
    UNCLIPPED_STUDY_RAW_X_ORIGIN,
    apply_full_conductance_targets,
    build_unclipped_baseline_spacing_mapping,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_runtime import (
    _endpoint_conductance_diagnostics,
    _endpoint_residual_reports,
    _flatten,
    _hardware_summary,
    _joint_population,
    _maximum_supported_uniform_index,
    _numeric_summary,
    _pulse_distribution_reports,
    _requested_to_nearest_code_confusion,
    _save_design_mapping,
    _save_ideal_weight_errors,
    _save_persistent_weight_errors,
    _split_population_vector,
    _state_digest,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _float_summary,
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_ideal_mapping_scheme_screen import (
    _ordered_labels,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders
from experiments.schema import to_plain_data
from training.ibm_reram_hwa import IbmReramArrayPopulation
from training.ibm_reram_program_verify import ControllerSettings
from training.ibm_reram_raw_active_program_verify import (
    RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT,
    RawActiveProgramVerifyOutcome,
    classify_persistent_uniform_codes,
    run_raw_active_program_verify,
)


if TYPE_CHECKING:
    from ebl.cli import ValidateRequest


_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "ebl.mnist_relu_drn.ibm_om_baseline_spacing_pv_no_clip_result"
SUMMARY_SCHEMA_VERSION = 1
PV_SCHEMA = "ebl.mnist_relu_drn.ibm_om_baseline_spacing_pv_no_clip_endpoint"
PV_SCHEMA_VERSION = 1
PREDICTION_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_baseline_spacing_pv_no_clip_predictions"
)
PREDICTION_SCHEMA_VERSION = 1
LAYOUTS = ("halves", "paired")


def _compat_assignment_protocol(protocol: Any) -> Any:
    """Expose the exact predecessor assignment/commissioning contract."""

    return SimpleNamespace(
        device=protocol.device,
        reset_commissioning=protocol.commissioning,
        joint_assignment_repair=protocol.joint_assignment_repair,
    )


def _native_assignment_from_joint_artifact(
    hardware: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Recover native raw-x bounds and support-bounded RESET from the joint identity.

    The predecessor helper intentionally returns its historical public-[0,1]
    tensors.  This successor reads the same immutable joint artifact but never
    intersects native support with that interval.
    """

    joint = hardware["joint_assignment"]
    path = Path(str(joint["path"]))
    if sha256_file(path) != joint["sha256"]:
        raise RuntimeError("Joint-assignment artifact SHA-256 mismatch.")
    lower_layers: list[torch.Tensor] = []
    upper_layers: list[torch.Tensor] = []
    reset_layers: list[torch.Tensor] = []
    reset_mean_layers: list[torch.Tensor] = []
    native_clamp_counts: list[int] = []
    with np.load(path, allow_pickle=False) as payload:
        if (
            str(payload["schema"].item())
            != "ebl.mnist_relu_drn.ibm_om_baseline_joint_assignment"
            or int(payload["schema_version"].item()) != 1
            or int(payload["assignment_seed"].item())
            != int(hardware["assignment_seed"])
            or str(payload["hardware_instance_id"].item())
            != str(hardware["hardware_instance_id"])
        ):
            raise RuntimeError("Joint-assignment identity metadata mismatch.")
        for layer_index in range(2):
            mean_a = torch.from_numpy(
                payload[f"layer_{layer_index}_reset_mean_raw_a"].copy()
            ).to(torch.float64)
            minimum_a = torch.from_numpy(
                payload[f"layer_{layer_index}_min_bound"].copy()
            ).to(torch.float64)
            maximum_a = torch.from_numpy(
                payload[f"layer_{layer_index}_max_bound"].copy()
            ).to(torch.float64)
            if not (
                mean_a.shape == minimum_a.shape == maximum_a.shape
                and bool(torch.isfinite(mean_a).all())
                and bool(torch.isfinite(minimum_a).all())
                and bool(torch.isfinite(maximum_a).all())
            ):
                raise RuntimeError("Malformed native joint-assignment tensors.")
            lower = (minimum_a + 1.0) / 2.0
            upper = (maximum_a + 1.0) / 2.0
            mean = (mean_a + 1.0) / 2.0
            if bool(torch.any(upper < lower)):
                raise RuntimeError("Native raw-x device bounds are unordered.")
            reset = torch.minimum(torch.maximum(mean, lower), upper)
            lower_layers.append(lower)
            upper_layers.append(upper)
            reset_mean_layers.append(mean)
            reset_layers.append(reset)
            native_clamp_counts.append(int((reset != mean).sum().item()))
    return {
        "cell_lower_raw_x": tuple(lower_layers),
        "cell_upper_raw_x": tuple(upper_layers),
        "reset_mean_raw_x": tuple(reset_mean_layers),
        "reset_baseline_raw_x": tuple(reset_layers),
        "report": {
            "coordinate": "raw_x=(native_raw_active_a+1)/2_unclipped",
            "bound_policy": "bound_reset_mean_only_to_own_native_support",
            "public_0_1_intersection_applied": False,
            "native_support_bounded_reset_count_by_layer": native_clamp_counts,
            "native_support_bounded_reset_count": sum(native_clamp_counts),
            "cell_lower_raw_x": [_float_summary(value) for value in lower_layers],
            "cell_upper_raw_x": [_float_summary(value) for value in upper_layers],
            "reset_mean_raw_x": [
                _float_summary(value) for value in reset_mean_layers
            ],
            "reset_baseline_raw_x": [
                _float_summary(value) for value in reset_layers
            ],
        },
    }


def _validate_frozen_embedding(
    protocol: Any,
    *native_assignments: Mapping[str, Any],
) -> RawXConductanceEmbedding:
    contract = protocol.affine_embedding
    expected = (
        UNCLIPPED_STUDY_RAW_X_MINIMUM,
        UNCLIPPED_STUDY_RAW_X_ORIGIN,
        UNCLIPPED_STUDY_CONDUCTANCE_PER_RAW_X,
        UNCLIPPED_STUDY_RAW_X_MAXIMUM,
        UNCLIPPED_STUDY_CONDUCTANCE_CEILING,
    )
    observed = (
        float(contract.exact_support_floor_x),
        float(contract.support_origin_x),
        float(contract.slope_g_per_x),
        float(contract.exact_support_ceiling_x),
        float(contract.mapped_ceiling_g),
    )
    if observed != expected:
        raise RuntimeError(
            "Config and mapper disagree on the frozen study-wide affine embedding."
        )
    embedding = RawXConductanceEmbedding(
        raw_x_origin=contract.support_origin_x,
        conductance_per_raw_x=contract.slope_g_per_x,
        conductance_ceiling=contract.mapped_ceiling_g,
    )
    tolerance = 1e-12
    for native in native_assignments:
        lower = _flatten(native["cell_lower_raw_x"], dtype=torch.float64)
        upper = _flatten(native["cell_upper_raw_x"], dtype=torch.float64)
        if (
            float(lower.min().item()) < contract.exact_support_floor_x - tolerance
            or float(upper.max().item())
            > contract.exact_support_ceiling_x + tolerance
        ):
            raise RuntimeError("A reconstructed identity left the frozen raw-x cohort.")
        embedding.to_full_conductance(lower.to(torch.float32))
        embedding.to_full_conductance(upper.to(torch.float32))
    return embedding


def _mapping(
    logical_weights: Sequence[torch.Tensor],
    hardware: Mapping[str, Any],
    native: Mapping[str, Any],
    *,
    alpha: float,
    spacing: int,
    scales: Sequence[float],
    embedding: RawXConductanceEmbedding,
) -> BaselineSpacingMapping:
    return build_unclipped_baseline_spacing_mapping(
        logical_weights,
        baseline_position_fraction=alpha,
        spacing_delta_multiples=spacing,
        scale_fractions=scales,
        nominal_dw_min=float(hardware["nominal_dw_min"]),
        raw_x_origin=embedding.raw_x_origin,
        conductance_per_raw_x=embedding.conductance_per_raw_x,
        conductance_ceiling=embedding.conductance_ceiling,
        cell_lower_raw_x=native["cell_lower_raw_x"],
        cell_upper_raw_x=native["cell_upper_raw_x"],
        reset_baseline_raw_x=native["reset_baseline_raw_x"],
        intrinsic_references_native=hardware["intrinsic_references_native"],
    )


def _persistent_affine_handoff(
    outcome: RawActiveProgramVerifyOutcome,
    population: IbmReramArrayPopulation,
    embedding: RawXConductanceEmbedding,
) -> tuple[tuple[torch.Tensor, ...], Mapping[str, Any]]:
    """Apply persistent raw x directly; fail outside native support or affine range."""

    persistent = outcome.persistent_endpoint_unit.to(torch.float64)
    lower = outcome.raw_lower_unit.to(torch.float64)
    upper = outcome.raw_upper_unit.to(torch.float64)
    if not (persistent.shape == lower.shape == upper.shape == (population.size,)):
        raise RuntimeError("Persistent endpoint shape does not match its identity.")
    tolerance = RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT
    below = persistent < lower - tolerance
    above = persistent > upper + tolerance
    if bool(below.any()) or bool(above.any()):
        raise RuntimeError(
            "Persistent P&V endpoint left native support; no projection is permitted."
        )
    flat_g = embedding.to_full_conductance(
        outcome.persistent_endpoint_unit
    ).to(torch.float32)
    conductances = _split_population_vector(flat_g, population)
    return conductances, {
        "policy": "direct_study_wide_affine_no_projection",
        "coordinate_before": outcome.coordinate,
        "formula": "G=conductance_per_raw_x*(x_raw-raw_x_origin)",
        "raw_x_origin": embedding.raw_x_origin,
        "conductance_per_raw_x": embedding.conductance_per_raw_x,
        "conductance_ceiling": embedding.conductance_ceiling,
        "cells": population.size,
        "below_native_support": int(below.sum().item()),
        "above_native_support": int(above.sum().item()),
        "projected": 0,
        "raw_minimum": float(persistent.min().item()),
        "raw_maximum": float(persistent.max().item()),
        "full_conductance_minimum": float(flat_g.min().item()),
        "full_conductance_maximum": float(flat_g.max().item()),
        "full_conductance_sha256": _tensor_sha256(flat_g),
    }


def _apparent_raw_diagnostic(
    outcome: RawActiveProgramVerifyOutcome,
    embedding: RawXConductanceEmbedding,
) -> Mapping[str, Any]:
    apparent = outcome.apparent_endpoint_unit.to(torch.float64)
    lower = outcome.raw_lower_unit.to(torch.float64)
    upper = outcome.raw_upper_unit.to(torch.float64)
    return {
        "role": "controller_and_raw_diagnostic_only_never_applied_to_drn",
        "applied_to_drn": False,
        "raw_x": _float_summary(apparent),
        "below_native_support": int((apparent < lower).sum().item()),
        "above_native_support": int((apparent > upper).sum().item()),
        "at_or_below_affine_origin": int(
            (apparent <= embedding.raw_x_origin).sum().item()
        ),
        "above_affine_ceiling": int(
            (apparent > embedding.raw_x_ceiling).sum().item()
        ),
    }


def _save_pv_endpoint(
    path: Path,
    *,
    outcome: RawActiveProgramVerifyOutcome,
    population: IbmReramArrayPopulation,
    mapping: BaselineSpacingMapping,
    persistent_conductances: Sequence[torch.Tensor],
    code: Any,
    assignment_seed: int,
    baseline_position_fraction: float,
    spacing_delta_x_multiplier: int,
    persistent_metrics: Mapping[str, Any],
    persistent_handoff: Mapping[str, Any],
    apparent_diagnostic: Mapping[str, Any],
) -> Mapping[str, Any]:
    requested = _flatten(
        tuple(layer.integer_level_number for layer in mapping.physical.layers),
        dtype=torch.int64,
    )
    if not torch.equal(code.requested_index, requested):
        raise RuntimeError("Persistent-code classification lost requested-code order.")
    verify_intersection = (
        outcome.target_unit + outcome.tolerance_unit >= outcome.raw_lower_unit
    ) & (outcome.target_unit - outcome.tolerance_unit <= outcome.raw_upper_unit)
    programming = outcome.programming
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(PV_SCHEMA),
        "schema_version": np.asarray(PV_SCHEMA_VERSION, dtype=np.int64),
        "endpoint_seed": np.asarray(outcome.endpoint_seed, dtype=np.int64),
        "target_unit": outcome.target_unit.numpy(),
        "raw_lower_unit": outcome.raw_lower_unit.numpy(),
        "raw_upper_unit": outcome.raw_upper_unit.numpy(),
        "initial_persistent_unit": outcome.initial_persistent_unit.numpy(),
        "initial_apparent_unit": outcome.initial_apparent_unit.numpy(),
        "persistent_endpoint_unit": outcome.persistent_endpoint_unit.numpy(),
        "apparent_endpoint_unit": outcome.apparent_endpoint_unit.numpy(),
        "exact_target_in_support": outcome.exact_target_in_support.numpy(),
        "verify_window_intersects_support": verify_intersection.numpy(),
        "apparent_accepted": programming.accepted.numpy(),
        "persistent_inside_acceptance_window": (
            outcome.persistent_inside_acceptance_window.numpy()
        ),
        "nonfinite": programming.nonfinite.numpy(),
        "budget_exhausted": programming.budget_exhausted.numpy(),
        "saturated_lower": outcome.saturated_lower.numpy(),
        "saturated_upper": outcome.saturated_upper.numpy(),
        "set_count": programming.set_count.numpy(),
        "reset_count": programming.reset_count.numpy(),
        "total_pulses": programming.total_pulses.numpy(),
        "verify_count": programming.verify_count.numpy(),
        "reversals": programming.reversals.numpy(),
        "requested_level_index": requested.numpy(),
        "nearest_persistent_level_index": code.nearest_persistent_index.numpy(),
        "requested_code_correct": code.requested_code_correct.numpy(),
        "persistent_residual_to_requested": (
            code.persistent_residual_to_requested.numpy()
        ),
        "persistent_residual_to_nearest": (
            code.persistent_residual_to_nearest.numpy()
        ),
    }
    layer_reports = []
    for layer_index, (target, persistent, layout) in enumerate(
        zip(mapping.ideal_targets, persistent_conductances, LAYOUTS)
    ):
        if target.shape != persistent.shape:
            raise RuntimeError("Persistent full-conductance layer shape mismatch.")
        residual = persistent.to(torch.float64) - target.to(torch.float64)
        arrays[f"layer_{layer_index}_target_full_conductance"] = target.numpy()
        arrays[f"layer_{layer_index}_persistent_full_conductance"] = (
            persistent.numpy()
        )
        arrays[f"layer_{layer_index}_persistent_residual_full_conductance"] = (
            residual.numpy()
        )
        arrays[f"layer_{layer_index}_target_contrast"] = quad_contrast(
            target, layout=layout
        ).numpy()
        arrays[f"layer_{layer_index}_target_loading"] = quad_loading(
            target, layout=layout
        ).numpy()
        arrays[f"layer_{layer_index}_persistent_contrast"] = quad_contrast(
            persistent, layout=layout
        ).numpy()
        arrays[f"layer_{layer_index}_persistent_loading"] = quad_loading(
            persistent, layout=layout
        ).numpy()
        diagnostics = _endpoint_conductance_diagnostics(
            persistent, layout=layout
        )
        layer_reports.append(
            {
                "layer": layer_index,
                "layout": layout,
                "persistent_target_residual_full_conductance": _numeric_summary(
                    residual
                ),
                "persistent_loading": diagnostics["loading"],
                "persistent_rms_contrast_over_mean_loading": diagnostics[
                    "rms_contrast_over_mean_loading"
                ],
            }
        )
    for layer_index, (target, lower, upper) in enumerate(
        zip(
            _split_population_vector(outcome.target_unit, population),
            _split_population_vector(outcome.raw_lower_unit, population),
            _split_population_vector(outcome.raw_upper_unit, population),
        )
    ):
        arrays[f"layer_{layer_index}_target_unit"] = target.numpy()
        arrays[f"layer_{layer_index}_raw_lower_unit"] = lower.numpy()
        arrays[f"layer_{layer_index}_raw_upper_unit"] = upper.numpy()
    continuation = outcome.continuation_state
    for key in ("construction_seeds", "persistent", "apparent", "draw_indices"):
        value = continuation.get(key)
        if isinstance(value, torch.Tensor):
            arrays[f"continuation_{key}"] = value.numpy()
    seeds = continuation.get("seeds")
    if isinstance(seeds, list):
        arrays["continuation_trajectory_seeds"] = np.asarray(seeds, dtype=np.int64)
    for key in (
        "schema_version",
        "preset",
        "rng_backend",
        "maximum_random_draws",
        "state_coordinate",
    ):
        value = continuation.get(key)
        if value is not None:
            arrays[f"continuation_{key}"] = np.asarray(value)
    _atomic_save_npz(path, arrays)
    receipt_path = path.with_suffix(".receipt.json")
    receipt = {
        "schema": PV_SCHEMA,
        "schema_version": PV_SCHEMA_VERSION,
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "endpoint_seed": outcome.endpoint_seed,
        "assignment_seed": int(assignment_seed),
        "baseline_position_fraction": float(baseline_position_fraction),
        "spacing_delta_x_multiplier": int(spacing_delta_x_multiplier),
        "population_fingerprint": population.fingerprint,
        "coordinate": outcome.coordinate,
        "controller": outcome.controller,
        "tolerance_unit": outcome.tolerance_unit,
        "maximum_program_pulses": outcome.maximum_program_pulses,
        "inference_read_noise_enabled": outcome.inference_read_noise_enabled,
        "continuation_state_sha256": _state_digest(continuation),
        "cells": population.size,
        "counts": {
            "exact_target_in_support": int(
                outcome.exact_target_in_support.sum().item()
            ),
            "verify_window_intersects_support": int(verify_intersection.sum().item()),
            "apparent_accepted": int(programming.accepted.sum().item()),
            "persistent_inside_acceptance_window": int(
                outcome.persistent_inside_acceptance_window.sum().item()
            ),
            "requested_code_correct": int(code.requested_code_correct.sum().item()),
            "nearest_code_adjacent_to_requested": int(
                ((code.nearest_persistent_index - requested).abs() == 1).sum().item()
            ),
            "nearest_code_farther_than_adjacent": int(
                ((code.nearest_persistent_index - requested).abs() > 1).sum().item()
            ),
            "budget_exhausted": int(programming.budget_exhausted.sum().item()),
            "nonfinite": int(programming.nonfinite.sum().item()),
            "saturated_lower": int(outcome.saturated_lower.sum().item()),
            "saturated_upper": int(outcome.saturated_upper.sum().item()),
        },
        "pulse_totals": {
            "set": int(programming.set_count.sum().item()),
            "reset": int(programming.reset_count.sum().item()),
            "total": int(programming.total_pulses.sum().item()),
            "verify": int(programming.verify_count.sum().item()),
            "reversals": int(programming.reversals.sum().item()),
        },
        "pulse_distributions": _pulse_distribution_reports(programming),
        "requested_to_nearest_code_confusion": (
            _requested_to_nearest_code_confusion(
                requested, code.nearest_persistent_index
            )
        ),
        "persistent_network_metrics": dict(persistent_metrics),
        "persistent_affine_handoff": dict(persistent_handoff),
        "apparent_endpoint": dict(apparent_diagnostic),
        "full_conductance_residuals_by_layer": layer_reports,
        **_endpoint_residual_reports(
            requested,
            outcome.persistent_endpoint_unit - outcome.target_unit,
            outcome.apparent_endpoint_unit - outcome.target_unit,
        ),
        "tensor_hashes": {
            name: sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()
            for name, value in arrays.items()
            if isinstance(value, np.ndarray) and value.ndim > 0
        },
    }
    atomic_write_json(receipt_path, receipt)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "report": receipt,
    }


def _save_predictions(
    path: Path,
    *,
    labels: torch.Tensor,
    teacher: torch.Tensor,
    continuous: torch.Tensor,
    ideal: torch.Tensor,
    persistent: Sequence[torch.Tensor],
    endpoint_seeds: Sequence[int],
) -> Mapping[str, Any]:
    """Save only endpoints that were actually applied to the DRN."""

    if len(persistent) != len(endpoint_seeds):
        raise ValueError("Expected one persistent prediction per endpoint seed.")
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(PREDICTION_SCHEMA),
        "schema_version": np.asarray(PREDICTION_SCHEMA_VERSION, dtype=np.int64),
        "labels": labels.numpy(),
        "teacher_prediction": teacher.numpy(),
        "continuous_prediction": continuous.numpy(),
        "ideal_quantized_prediction": ideal.numpy(),
    }
    repeats = []
    ideal_correct = ideal.eq(labels)
    for repeat_index, (endpoint_seed, prediction) in enumerate(
        zip(endpoint_seeds, persistent)
    ):
        arrays[f"pv_persistent_prediction_seed_{int(endpoint_seed)}"] = (
            prediction.numpy()
        )
        correct = prediction.eq(labels)
        repeats.append(
            {
                "repeat_index": repeat_index,
                "endpoint_seed": int(endpoint_seed),
                "persistent_correct": int(correct.sum().item()),
                "ideal_to_persistent_wrong_to_correct": int(
                    ((~ideal_correct) & correct).sum().item()
                ),
                "ideal_to_persistent_correct_to_wrong": int(
                    (ideal_correct & (~correct)).sum().item()
                ),
                "ideal_to_persistent_prediction_flip_count": int(
                    ideal.ne(prediction).sum().item()
                ),
                "persistent_prediction_sha256": _tensor_sha256(prediction),
            }
        )
    _atomic_save_npz(path, arrays)
    receipt_path = path.with_suffix(".receipt.json")
    receipt = {
        "schema": PREDICTION_SCHEMA,
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "examples": int(labels.numel()),
        "labels_sha256": _tensor_sha256(labels),
        "teacher_prediction_sha256": _tensor_sha256(teacher),
        "continuous_prediction_sha256": _tensor_sha256(continuous),
        "ideal_prediction_sha256": _tensor_sha256(ideal),
        "ideal_correct": int(ideal_correct.sum().item()),
        "apparent_predictions_saved": False,
        "repeats": repeats,
    }
    atomic_write_json(receipt_path, receipt)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "report": receipt,
    }


def run_validate(request: "ValidateRequest") -> int:
    from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_no_clip_config import (
        BaselineSpacingPvNoClipValidateSpec,
    )

    spec = request.spec
    if not isinstance(spec, BaselineSpacingPvNoClipValidateSpec):
        raise TypeError("Expected the dedicated no-clipping validate spec.")
    if request.teacher_weights is not None or request.device_model is not None:
        raise ValueError("Expected only --weights for this frozen-source study.")
    protocol = spec.protocol
    weights_path = request.weights.expanduser().resolve()
    if sha256_file(weights_path) != protocol.source.expected_weights_sha256:
        raise ValueError("Frozen ReLU source SHA-256 does not match the contract.")
    sampler = _sampler_python()
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=(
            _input("weights", weights_path),
            _input("aihwkit_python", sampler),
        ),
        resume_capability="unsupported",
    )
    try:
        student_spec = spec.student
        if student_spec.runtime.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This study requires a live CUDA device; CPU is invalid.")
        torch.manual_seed(student_spec.runtime.seed)
        torch.cuda.manual_seed_all(student_spec.runtime.seed)
        device = torch.device("cuda")
        loaders = build_mnist_loaders(
            student_spec.data,
            data_seed=student_spec.runtime.data_seed,
            calibration_examples=student_spec.mapping.calibration_examples,
            calibration_batch_size=student_spec.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(
            weights_path, device=device, spec=student_spec
        )
        logical_weights = tuple(
            value.detach().cpu().clone() for value in teacher.parameters()
        )
        if len(logical_weights) != 2:
            raise RuntimeError("Expected exactly two frozen ReLU weight tensors.")

        artifacts: list[Mapping[str, Any]] = []
        artifact_root = store.run_dir / "artifacts"
        compat = _compat_assignment_protocol(protocol)
        alpha = float(protocol.baseline_position_fraction)
        spacing = int(protocol.spacing_delta_x_multiplier)

        dev_stack = build_student_stack(student_spec, enable_measured=False)
        development, development_artifacts = _prepare_assignment(
            assignment_seed=protocol.calibration.development_assignment_seed,
            assignment_role="development",
            logical_weights=logical_weights,
            stack=dev_stack,
            protocol=compat,
            aihwkit_python=sampler,
            artifact_root=artifact_root,
        )
        artifacts.extend(development_artifacts)
        if (
            development["hardware_instance_id"]
            != protocol.predecessor.development_hardware_instance_id
        ):
            raise RuntimeError("Development identity differs from the predecessor.")
        development_native = _native_assignment_from_joint_artifact(development)
        embedding = _validate_frozen_embedding(protocol, development_native)
        if (
            float(student_spec.model.conductance_min) != 0.0
            or float(student_spec.model.conductance_max)
            != embedding.conductance_ceiling
        ):
            raise RuntimeError("Student circuit range does not match affine capacity.")

        calibration_labels = _ordered_labels(loaders.calibration)
        data_provenance = _dataset_provenance(loaders, calibration_labels)
        candidates = []
        for scale_pair in product(protocol.calibration.scale_fractions, repeat=2):
            candidate_mapping = _mapping(
                logical_weights,
                development,
                development_native,
                alpha=alpha,
                spacing=spacing,
                scales=scale_pair,
                embedding=embedding,
            )
            apply_full_conductance_targets(
                dev_stack.bundle.catalog,
                candidate_mapping.continuous_targets,
                conductance_min=0.0,
                conductance_max=embedding.conductance_ceiling,
            )
            calibration = _fit_calibration(
                dev_stack,
                teacher,
                loaders.calibration,
                calibration_labels,
                student_spec,
            )
            candidates.append(
                {
                    "scale_fractions": list(map(float, scale_pair)),
                    "calibration": calibration,
                    "continuous_target_hashes": [
                        _tensor_sha256(value)
                        for value in candidate_mapping.continuous_targets
                    ],
                }
            )
        selected_index = _selected_candidate(candidates)
        selected = candidates[selected_index]
        selected_pair = tuple(map(float, selected["scale_fractions"]))
        selected_gain = float(selected["calibration"]["gain"])
        development_mapping = _mapping(
            logical_weights,
            development,
            development_native,
            alpha=alpha,
            spacing=spacing,
            scales=selected_pair,
            embedding=embedding,
        )
        development_invariants = _mapping_invariants(development_mapping.physical)
        if not development_invariants["valid"]:
            raise RuntimeError("Development full-conductance invariants failed.")
        analysis_scales = fit_weight_reconstruction_scales(
            logical_weights, development_mapping.physical
        )
        development_mapping_artifact = _save_design_mapping(
            artifact_root / "development" / "selected_physical_mapping.npz",
            development_mapping,
            assignment_seed=protocol.calibration.development_assignment_seed,
            assignment_role="development",
            hardware_instance_id=development["hardware_instance_id"],
        )
        artifacts.append(
            {**development_mapping_artifact, "kind": "ibm_om_physical_mapping"}
        )
        artifacts.append(
            {
                "path": development_mapping_artifact["context"],
                "kind": "mapping_context",
            }
        )
        calibration_receipt = {
            "schema": (
                "ebl.mnist_relu_drn.ibm_om_baseline_spacing_no_clip_"
                "development_calibration"
            ),
            "schema_version": 1,
            "baseline_position_fraction": alpha,
            "sharing": protocol.calibration.sharing,
            "assignment_seed": protocol.calibration.development_assignment_seed,
            "hardware_instance_id": development["hardware_instance_id"],
            "candidate_count": len(candidates),
            "candidates": candidates,
            "selected_index": selected_index,
            "selected": selected,
            "analysis_scales": list(map(float, analysis_scales)),
            "calibration_labels_sha256": data_provenance[
                "calibration_labels_sha256"
            ],
            "calibration_indices_sha256": data_provenance[
                "calibration_indices_sha256"
            ],
            "test_labels_used": False,
            "affine_embedding": dict(embedding.report()),
        }
        calibration_path = artifact_root / "development" / "calibration.json"
        atomic_write_json(calibration_path, calibration_receipt)
        artifacts.append(
            {"path": str(calibration_path), "kind": "development_calibration"}
        )
        print(
            f"calibrated no-clip alpha={alpha:g}: scales={selected_pair}, "
            f"gain={selected_gain:.8g}",
            flush=True,
        )

        del dev_stack
        gc.collect()
        torch.cuda.empty_cache()
        heldout_stack = build_student_stack(student_spec, enable_measured=False)
        heldout_seed = int(protocol.assignments.heldout_seed)
        heldout, heldout_artifacts = _prepare_assignment(
            assignment_seed=heldout_seed,
            assignment_role="heldout",
            logical_weights=logical_weights,
            stack=heldout_stack,
            protocol=compat,
            aihwkit_python=sampler,
            artifact_root=artifact_root,
        )
        artifacts.extend(heldout_artifacts)
        heldout_index = protocol.assignments.allowed_heldout_seeds.index(heldout_seed)
        if (
            heldout["hardware_instance_id"]
            != protocol.predecessor.heldout_hardware_instance_ids[heldout_index]
        ):
            raise RuntimeError("Held-out identity differs from the predecessor.")
        heldout_native = _native_assignment_from_joint_artifact(heldout)
        _validate_frozen_embedding(protocol, development_native, heldout_native)
        heldout_mapping = _mapping(
            logical_weights,
            heldout,
            heldout_native,
            alpha=alpha,
            spacing=spacing,
            scales=selected_pair,
            embedding=embedding,
        )
        heldout_invariants = _mapping_invariants(heldout_mapping.physical)
        if not heldout_invariants["valid"]:
            raise RuntimeError("Held-out full-conductance invariants failed.")
        heldout_mapping_artifact = _save_design_mapping(
            artifact_root / "heldout" / "ideal_physical_mapping.npz",
            heldout_mapping,
            assignment_seed=heldout_seed,
            assignment_role="heldout",
            hardware_instance_id=heldout["hardware_instance_id"],
        )
        artifacts.append(
            {
                **heldout_mapping_artifact,
                "kind": "ibm_om_baseline_spacing_no_clip_physical_mapping",
            }
        )
        artifacts.append(
            {"path": heldout_mapping_artifact["context"], "kind": "mapping_context"}
        )
        heldout_stack.cost.gain = selected_gain
        apply_full_conductance_targets(
            heldout_stack.bundle.catalog,
            heldout_mapping.continuous_targets,
            conductance_min=0.0,
            conductance_max=embedding.conductance_ceiling,
        )
        continuous_metrics, continuous_prediction = _evaluate_detailed(
            heldout_stack,
            teacher,
            loaders.test,
            sample_limit=student_spec.settings.sample_limit,
        )
        apply_full_conductance_targets(
            heldout_stack.bundle.catalog,
            heldout_mapping.ideal_targets,
            conductance_min=0.0,
            conductance_max=embedding.conductance_ceiling,
        )
        ideal_metrics, ideal_prediction = _evaluate_detailed(
            heldout_stack,
            teacher,
            loaders.test,
            sample_limit=student_spec.settings.sample_limit,
        )
        labels, teacher_prediction = _teacher_predictions(
            teacher,
            loaders.test,
            device=device,
            sample_limit=student_spec.settings.sample_limit,
        )
        expected_examples = 10_000
        if not (
            continuous_metrics["examples"]
            == ideal_metrics["examples"]
            == labels.numel()
            == expected_examples
        ):
            raise RuntimeError("Ideal endpoints do not have full test coverage.")

        population = _joint_population(heldout)
        expected_shapes = tuple(
            tuple(value.shape) for value in heldout_mapping.ideal_target_units
        )
        if population.binding_shapes != expected_shapes:
            raise RuntimeError("Joint P&V population does not match mapped rail order.")
        population_lower_raw_x = (
            population.min_bound.to(torch.float64) + 1.0
        ) / 2.0
        population_upper_raw_x = (
            population.max_bound.to(torch.float64) + 1.0
        ) / 2.0
        if not (
            torch.equal(
                population_lower_raw_x,
                _flatten(heldout_native["cell_lower_raw_x"], dtype=torch.float64),
            )
            and torch.equal(
                population_upper_raw_x,
                _flatten(heldout_native["cell_upper_raw_x"], dtype=torch.float64),
            )
        ):
            raise RuntimeError("P&V population bounds differ from raw mapping bounds.")
        target_raw_x = _flatten(
            heldout_mapping.ideal_target_units, dtype=torch.float32
        )
        target_full_g = embedding.to_full_conductance(target_raw_x).to(torch.float32)
        ideal_full_g = _flatten(heldout_mapping.ideal_targets, dtype=torch.float32)
        target_residual = target_full_g.to(torch.float64) - ideal_full_g.to(
            torch.float64
        )
        handoff_tolerance = max(1e-12, embedding.conductance_ceiling * 1e-6)
        if float(target_residual.abs().max().item()) > handoff_tolerance:
            raise RuntimeError("Raw-x ideal targets do not reconstruct full G.")
        if bool(
            torch.any(
                target_raw_x.to(torch.float64)
                < population_lower_raw_x - RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT
            )
        ) or bool(
            torch.any(
                target_raw_x.to(torch.float64)
                > population_upper_raw_x + RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT
            )
        ):
            raise RuntimeError("An ideal target left its native device support.")
        target_handoff_report = {
            "policy": "direct_study_wide_affine_no_projection",
            "projected": 0,
            "raw_x_sha256": _tensor_sha256(target_raw_x),
            "full_conductance_sha256": _tensor_sha256(target_full_g),
            "ideal_full_conductance_sha256": _tensor_sha256(ideal_full_g),
            "maximum_absolute_reconstruction_residual": float(
                target_residual.abs().max().item()
            ),
            "reconstruction_tolerance": handoff_tolerance,
            "affine_embedding": dict(embedding.report()),
        }

        baseline_raw_x = _flatten(
            heldout_mapping.baseline_raw_x, dtype=torch.float32
        )
        requested_index = _flatten(
            tuple(
                layer.integer_level_number for layer in heldout_mapping.physical.layers
            ),
            dtype=torch.int64,
        )
        maximum_index = _maximum_supported_uniform_index(
            population_upper_raw_x,
            baseline_raw_x,
            spacing_unit=float(heldout_mapping.report["level_spacing_unit"]),
        )
        if bool(torch.any(maximum_index < requested_index)):
            raise RuntimeError("Requested ideal code exceeds native cell capacity.")

        persistent_metrics = []
        persistent_predictions = []
        persistent_weight_artifacts = []
        pv_artifacts = []
        tolerance_unit = (
            protocol.program_verify.tolerance_delta_x_ratio
            * float(heldout_mapping.report["delta_x"])
        )
        for repeat_index, endpoint_seed in enumerate(
            protocol.assignments.endpoint_seeds
        ):
            outcome = run_raw_active_program_verify(
                population,
                targets_unit=target_raw_x,
                endpoint_seed=int(endpoint_seed),
                tolerance_unit=tolerance_unit,
                maximum_program_pulses=protocol.program_verify.maximum_program_pulses,
                device=device,
                settings=ControllerSettings(kind=protocol.program_verify.controller),
            )
            code = classify_persistent_uniform_codes(
                outcome.persistent_endpoint_unit,
                baseline_unit=baseline_raw_x,
                spacing_unit=float(heldout_mapping.report["level_spacing_unit"]),
                requested_index=requested_index,
                maximum_index=maximum_index,
            )
            persistent_g, persistent_handoff = _persistent_affine_handoff(
                outcome, population, embedding
            )
            apparent_diagnostic = _apparent_raw_diagnostic(outcome, embedding)
            apply_full_conductance_targets(
                heldout_stack.bundle.catalog,
                persistent_g,
                conductance_min=0.0,
                conductance_max=embedding.conductance_ceiling,
            )
            persistent_result, persistent_prediction = _evaluate_detailed(
                heldout_stack,
                teacher,
                loaders.test,
                sample_limit=student_spec.settings.sample_limit,
            )
            if persistent_result["examples"] != expected_examples:
                raise RuntimeError("Persistent endpoint lacks full test coverage.")
            pv_artifact = _save_pv_endpoint(
                artifact_root
                / "heldout"
                / f"program_verify_endpoint_seed_{int(endpoint_seed)}.npz",
                outcome=outcome,
                population=population,
                mapping=heldout_mapping,
                persistent_conductances=persistent_g,
                code=code,
                assignment_seed=heldout_seed,
                baseline_position_fraction=alpha,
                spacing_delta_x_multiplier=spacing,
                persistent_metrics=persistent_result,
                persistent_handoff=persistent_handoff,
                apparent_diagnostic=apparent_diagnostic,
            )
            artifacts.append(
                {**pv_artifact, "kind": "ibm_om_baseline_spacing_no_clip_endpoint"}
            )
            pv_artifacts.append(pv_artifact)
            persistent_metrics.append(persistent_result)
            persistent_predictions.append(persistent_prediction)
            weight_artifact = _save_persistent_weight_errors(
                artifact_root
                / "heldout"
                / f"program_verify_weight_errors_seed_{int(endpoint_seed)}.npz",
                endpoint_seed=int(endpoint_seed),
                logical_weights=logical_weights,
                conductances=persistent_g,
                analysis_scales=analysis_scales,
            )
            artifacts.append(
                {**weight_artifact, "kind": "program_verify_weight_errors"}
            )
            persistent_weight_artifacts.append(weight_artifact)
            print(
                f"heldout={heldout_seed} no-clip alpha={alpha:g} "
                f"h={spacing}delta pv[{repeat_index}]="
                f"{100.0 * persistent_result['student_accuracy']:.2f}% "
                f"accepted={pv_artifact['report']['counts']['apparent_accepted']}"
                f"/{population.size}",
                flush=True,
            )
            del outcome, code, persistent_g
            gc.collect()
            torch.cuda.empty_cache()

        prediction_artifact = _save_predictions(
            artifact_root / "heldout" / "predictions.npz",
            labels=labels,
            teacher=teacher_prediction,
            continuous=continuous_prediction,
            ideal=ideal_prediction,
            persistent=persistent_predictions,
            endpoint_seeds=protocol.assignments.endpoint_seeds,
        )
        artifacts.append({**prediction_artifact, "kind": "predictions"})
        ideal_weight_artifact = _save_ideal_weight_errors(
            artifact_root / "heldout" / "ideal_weight_errors.npz",
            logical_weights=logical_weights,
            mapping=heldout_mapping,
            analysis_scales=analysis_scales,
        )
        artifacts.append(
            {**ideal_weight_artifact, "kind": "ideal_weight_errors"}
        )

        pv_mean_accuracy = sum(
            value["student_accuracy"] for value in persistent_metrics
        ) / len(persistent_metrics)
        validity_gates = {
            "cuda": student_spec.runtime.device == "cuda"
            and torch.cuda.is_available(),
            "same_development_identity": development["hardware_instance_id"]
            == protocol.predecessor.development_hardware_instance_id,
            "same_heldout_identity": heldout["hardware_instance_id"]
            == protocol.predecessor.heldout_hardware_instance_ids[heldout_index],
            "development_invariants": bool(development_invariants["valid"]),
            "heldout_invariants": bool(heldout_invariants["valid"]),
            "candidate_count_16": len(candidates) == 16,
            "repeat_count": len(persistent_metrics)
            == protocol.program_verify.repeat_count,
            "all_targets_in_support": all(
                item["report"]["counts"]["exact_target_in_support"]
                == population.size
                for item in pv_artifacts
            ),
            "persistent_projection_count_zero": all(
                item["report"]["persistent_affine_handoff"]["projected"] == 0
                for item in pv_artifacts
            ),
            "persistent_endpoints_in_native_support": all(
                item["report"]["persistent_affine_handoff"][
                    "below_native_support"
                ]
                == 0
                and item["report"]["persistent_affine_handoff"][
                    "above_native_support"
                ]
                == 0
                for item in pv_artifacts
            ),
            "apparent_endpoint_never_applied": all(
                not item["report"]["apparent_endpoint"]["applied_to_drn"]
                for item in pv_artifacts
            ),
            "no_nonfinite_programming": all(
                item["report"]["counts"]["nonfinite"] == 0
                for item in pv_artifacts
            ),
            "optimizer_updates_zero": protocol.exclusions.optimizer_updates == 0,
        }
        if not all(validity_gates.values()):
            raise RuntimeError(
                "Refusing invalid no-clipping evidence: "
                f"{[name for name, value in validity_gates.items() if not value]!r}."
            )

        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "complete",
            "claim_boundary": (
                "Exploratory model-based AIHWKit 1.1.0 OM finite-cohort affine "
                "embedding; persistent raw-x P&V endpoints only. No public "
                "projection, apparent-state inference, optimizer update, HWA, "
                "retention, or fabricated-device claim."
            ),
            "source": {
                "path": str(weights_path),
                "sha256": sha256_file(weights_path),
                "metadata": teacher_metadata,
                "logical_weight_hashes": [
                    _tensor_sha256(value) for value in logical_weights
                ],
                "optimizer_updates": 0,
            },
            "data": data_provenance,
            "contract": to_plain_data(protocol),
            "affine_embedding": dict(embedding.report()),
            "design": {
                "baseline_position_fraction": alpha,
                "spacing_delta_x_multiplier": spacing,
                "level_spacing_raw_x": heldout_mapping.report[
                    "level_spacing_unit"
                ],
            },
            "development": {
                "hardware": _hardware_summary(development),
                "native_coordinate": development_native["report"],
                "calibration": calibration_receipt,
                "mapping": development_mapping.report,
                "invariants": development_invariants,
            },
            "heldout": {
                "assignment_seed": heldout_seed,
                "hardware": _hardware_summary(heldout),
                "native_coordinate": heldout_native["report"],
                "mapping": heldout_mapping.report,
                "invariants": heldout_invariants,
                "target_affine_handoff": target_handoff_report,
                "continuous": continuous_metrics,
                "ideal_quantized": ideal_metrics,
                "pv_persistent_repeats": persistent_metrics,
                "pv_persistent_mean_accuracy": pv_mean_accuracy,
                "pv_endpoint_artifacts": [
                    item["report"] for item in pv_artifacts
                ],
                "apparent_endpoint_role": (
                    "controller_and_raw_diagnostic_only_never_applied_to_drn"
                ),
                "weight_errors": ideal_weight_artifact["report"],
                "pv_persistent_weight_errors": [
                    item["report"] for item in persistent_weight_artifacts
                ],
                "predictions": prediction_artifact["report"],
            },
            "validity": {
                "coverage_valid": True,
                "gates": validity_gates,
                "expected_examples": expected_examples,
                "optimizer_updates": 0,
                "program_verify_enabled": True,
                "inference_read_noise": 0.0,
                "apparent_endpoint_applied_to_drn": False,
            },
        }
        summary_path = artifact_root / "scientific_summary.json"
        atomic_write_json(summary_path, summary)
        artifacts.append({"path": str(summary_path), "kind": "scientific_summary"})
        terminal_metrics = {
            "metric_definition_ids": {
                "ideal_quantized": protocol.metrics.ideal_quantized,
                "pv_persistent": protocol.metrics.pv_persistent,
                "continuous_diagnostic": protocol.metrics.continuous_diagnostic,
            },
            "baseline_position_fraction": alpha,
            "spacing_delta_x_multiplier": spacing,
            "heldout_assignment_seed": heldout_seed,
            "development_hardware_instance_id": development[
                "hardware_instance_id"
            ],
            "heldout_hardware_instance_id": heldout["hardware_instance_id"],
            "selected_scale_fractions": list(selected_pair),
            "fixed_logit_gain": selected_gain,
            "continuous": {
                key: continuous_metrics[key]
                for key in (
                    "student_correct",
                    "examples",
                    "student_accuracy",
                    "prediction_sha256",
                )
            },
            "ideal_quantized": {
                key: ideal_metrics[key]
                for key in (
                    "student_correct",
                    "examples",
                    "student_accuracy",
                    "prediction_sha256",
                )
            },
            "pv_persistent": [
                {
                    key: value[key]
                    for key in (
                        "student_correct",
                        "examples",
                        "student_accuracy",
                        "prediction_sha256",
                    )
                }
                for value in persistent_metrics
            ],
            "pv_persistent_mean_accuracy": pv_mean_accuracy,
            "persistent_projection_count": 0,
            "apparent_endpoint_applied_to_drn": False,
            "coverage_valid": True,
        }
        store.append_metric({"mode": "validate", **terminal_metrics})
        store.complete(
            metrics=terminal_metrics,
            artifacts=_artifact_records(store, artifacts),
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = ["run_validate"]
