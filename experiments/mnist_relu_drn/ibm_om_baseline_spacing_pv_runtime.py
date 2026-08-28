"""Native CUDA runtime for the IBM OM baseline-position/spacing P&V study."""

from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
from itertools import product
import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import numpy as np
import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    build_weight_error_decomposition,
    fit_weight_reconstruction_scales,
    quad_contrast,
    quad_loading,
    save_physical_mapping,
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
    apply_full_conductance_targets,
    build_baseline_spacing_mapping,
    persistent_weight_error_report,
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
from training.ibm_reram_hwa import IbmReramArrayPopulation, load_om_array_population
from training.ibm_reram_program_verify import ControllerSettings
from training.ibm_reram_raw_active_program_verify import (
    RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT,
    RawActiveProgramVerifyOutcome,
    classify_persistent_uniform_codes,
    project_raw_active_unit_to_full_conductance,
    run_raw_active_program_verify,
)


if TYPE_CHECKING:
    from ebl.cli import ValidateRequest


_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "ebl.mnist_relu_drn.ibm_om_baseline_spacing_pv_result"
SUMMARY_SCHEMA_VERSION = 1
PV_SCHEMA = "ebl.mnist_relu_drn.ibm_om_baseline_spacing_pv_endpoint"
PV_SCHEMA_VERSION = 1
PREDICTION_SCHEMA = "ebl.mnist_relu_drn.ibm_om_baseline_spacing_pv_predictions"
PREDICTION_SCHEMA_VERSION = 1
LAYOUTS = ("halves", "paired")
IDENTITY_FIELDS = (
    "max_bound",
    "min_bound",
    "dwmin_up",
    "dwmin_down",
    "reference",
    "corrupt",
    "published_corrupt",
)


def _compat_assignment_protocol(protocol: Any) -> Any:
    """Expose only the frozen fields consumed by the reference assignment builder."""

    return SimpleNamespace(
        device=protocol.device,
        reset_commissioning=protocol.commissioning,
        joint_assignment_repair=protocol.joint_assignment_repair,
    )


def _flatten(values: Sequence[torch.Tensor], *, dtype: torch.dtype) -> torch.Tensor:
    return torch.cat(
        tuple(value.detach().to(device="cpu", dtype=dtype).reshape(-1) for value in values)
    )


def _maximum_supported_uniform_index(
    cell_upper_unit: torch.Tensor,
    baseline_unit: torch.Tensor,
    *,
    spacing_unit: float,
) -> torch.Tensor:
    """Return code capacity using the same support tolerance as raw-active P&V."""

    spacing = float(spacing_unit)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Expected a finite positive uniform-code spacing.")
    upper = torch.as_tensor(cell_upper_unit, dtype=torch.float64, device="cpu")
    baseline = torch.as_tensor(baseline_unit, dtype=torch.float64, device="cpu")
    if upper.shape != baseline.shape:
        raise ValueError("Expected matched upper-bound and baseline tensors.")
    return torch.floor(
        (upper - baseline + RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT) / spacing
        + 1e-12
    ).to(torch.int64)


def _split_population_vector(
    value: torch.Tensor, population: IbmReramArrayPopulation
) -> tuple[torch.Tensor, ...]:
    flat = value.detach().to(device="cpu").reshape(-1)
    if flat.numel() != population.size:
        raise ValueError("Expected one endpoint value per frozen physical cell.")
    result = []
    offset = 0
    for shape in population.binding_shapes:
        count = math.prod(shape)
        result.append(flat[offset : offset + count].reshape(shape).clone())
        offset += count
    if offset != population.size:
        raise RuntimeError("Expected binding shapes to cover the population.")
    return tuple(result)


def _joint_population(hardware: Mapping[str, Any]) -> IbmReramArrayPopulation:
    """Rebuild the jointly repaired physical identity from its strict artifact."""

    population_path = Path(str(hardware["population"]["path"]))
    if sha256_file(population_path) != hardware["population"]["sha256"]:
        raise RuntimeError("Base-population artifact SHA-256 mismatch.")
    base = load_om_array_population(population_path)
    joint_path = Path(str(hardware["joint_assignment"]["path"]))
    if sha256_file(joint_path) != hardware["joint_assignment"]["sha256"]:
        raise RuntimeError("Joint-assignment artifact SHA-256 mismatch.")
    receipt_path = Path(str(hardware["joint_assignment"]["receipt"]))
    if sha256_file(receipt_path) != hardware["joint_assignment"]["receipt_sha256"]:
        raise RuntimeError("Joint-assignment receipt SHA-256 mismatch.")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema")
        != "ebl.mnist_relu_drn.ibm_om_baseline_joint_assignment"
        or receipt.get("schema_version") != 1
        or receipt.get("assignment_seed") != hardware["assignment_seed"]
        or receipt.get("hardware_instance_id") != hardware["hardware_instance_id"]
        or receipt.get("artifact_sha256") != hardware["joint_assignment"]["sha256"]
    ):
        raise RuntimeError("Joint-assignment receipt does not bind the frozen identity.")
    with np.load(joint_path, allow_pickle=False) as payload:
        if (
            str(payload["schema"].item())
            != "ebl.mnist_relu_drn.ibm_om_baseline_joint_assignment"
            or int(payload["schema_version"].item()) != 1
            or int(payload["assignment_seed"].item()) != hardware["assignment_seed"]
            or str(payload["hardware_instance_id"].item())
            != hardware["hardware_instance_id"]
            or str(payload["base_population_fingerprint"].item())
            != base.fingerprint
        ):
            raise RuntimeError("Joint-assignment hardware ID mismatch.")
        tensors: dict[str, torch.Tensor] = {}
        for name in IDENTITY_FIELDS:
            pieces = []
            for layer_index, shape in enumerate(base.binding_shapes):
                array = payload[f"layer_{layer_index}_{name}"]
                if tuple(array.shape) != tuple(shape):
                    raise RuntimeError("Joint identity tensor shape mismatch.")
                pieces.append(torch.from_numpy(array.copy()).reshape(-1))
            dtype = torch.bool if name in {"corrupt", "published_corrupt"} else torch.float32
            tensors[name] = torch.cat(pieces).to(dtype=dtype)
    return IbmReramArrayPopulation(
        assignment_seed=base.assignment_seed,
        corruption_policy=base.corruption_policy,
        binding_keys=base.binding_keys,
        binding_shapes=base.binding_shapes,
        binding_sampling_seeds=base.binding_sampling_seeds,
        donor_sampling_seeds=base.donor_sampling_seeds,
        nominal_dw_min=base.nominal_dw_min,
        dw_min_std=base.dw_min_std,
        write_noise_std=base.write_noise_std,
        max_bound=tensors["max_bound"],
        min_bound=tensors["min_bound"],
        dwmin_up=tensors["dwmin_up"],
        dwmin_down=tensors["dwmin_down"],
        reference=tensors["reference"],
        corrupt=tensors["corrupt"],
        published_corrupt=tensors["published_corrupt"],
        fingerprint=str(hardware["hardware_instance_id"]),
        aihwkit_version=base.aihwkit_version,
    )


def _historical_parity_gates(
    *,
    require_full_test_output: bool,
    parity: Any,
    parity_index: int,
    development_hardware_instance_id: str,
    heldout_hardware_instance_id: str,
    selected_pair: Sequence[float],
    selected_gain: float,
    ideal_metrics: Mapping[str, Any],
) -> tuple[bool, Mapping[str, bool]]:
    """Evaluate full-output parity only when the full test split was run."""

    full_test_output_required = bool(require_full_test_output)
    gates = {
        "development_hardware_instance_id": (
            development_hardware_instance_id
            == parity.development_hardware_instance_id
        ),
        "heldout_hardware_instance_id": (
            heldout_hardware_instance_id
            == parity.heldout_hardware_instance_ids[parity_index]
        ),
        "selected_scales": tuple(map(float, selected_pair))
        == tuple(parity.selected_scale_fractions),
        "fixed_gain": float(selected_gain) == parity.fixed_logit_gain,
        "ideal_correct": (
            not full_test_output_required
            or ideal_metrics["student_correct"] == parity.ideal_correct[parity_index]
        ),
        "ideal_prediction_sha256": (
            not full_test_output_required
            or ideal_metrics["prediction_sha256"]
            == parity.ideal_prediction_sha256[parity_index]
        ),
    }
    return full_test_output_required, gates


def _historical_parity_summary(
    *,
    design_arm: bool,
    full_test_output_required: bool,
    gates: Mapping[str, bool],
) -> Mapping[str, Any]:
    """Build the exact parity block consumed by the study analyzer."""

    return {
        "design_arm": bool(design_arm),
        "full_test_output_required": bool(full_test_output_required),
        "gates": dict(gates),
    }


_SPACING_NEUTRAL_IDEAL_REPORT_KEYS = {
    "standard4delta_endpoint": "ideal_quantized_endpoint",
    "standard4delta_total": "ideal_quantized_total",
}


def _spacing_neutral_ideal_report(value: Any) -> Any:
    """Rename inherited 4-delta labels in this variable-spacing study only."""

    if isinstance(value, Mapping):
        result: dict[Any, Any] = {}
        for key, item in value.items():
            renamed = _SPACING_NEUTRAL_IDEAL_REPORT_KEYS.get(key, key)
            if renamed in result:
                raise RuntimeError(
                    f"Spacing-neutral ideal report key collision at {renamed!r}."
                )
            result[renamed] = _spacing_neutral_ideal_report(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_spacing_neutral_ideal_report(item) for item in value]
    return value


def _mapping(
    logical_weights: Sequence[torch.Tensor],
    hardware: Mapping[str, Any],
    student_spec: Any,
    *,
    alpha: float,
    spacing: int,
    scales: Sequence[float],
) -> BaselineSpacingMapping:
    return build_baseline_spacing_mapping(
        logical_weights,
        baseline_position_fraction=alpha,
        spacing_delta_multiples=spacing,
        scale_fractions=scales,
        nominal_dw_min=float(hardware["nominal_dw_min"]),
        conductance_min=float(student_spec.model.conductance_min),
        conductance_max=float(student_spec.model.conductance_max),
        cell_lower_units=hardware["cell_lower_units"],
        cell_upper_units=hardware["cell_upper_units"],
        reset_baseline_units=hardware["reset_baseline_units"],
        intrinsic_references_native=hardware["intrinsic_references_native"],
    )


def _save_design_mapping(
    path: Path,
    mapping: BaselineSpacingMapping,
    *,
    assignment_seed: int,
    assignment_role: str,
    hardware_instance_id: str,
) -> Mapping[str, Any]:
    saved = save_physical_mapping(
        path,
        mapping.physical,
        assignment_seed=assignment_seed,
        assignment_role=assignment_role,
        hardware_instance_id=hardware_instance_id,
    )
    context_path = path.with_name(f"{path.stem}.design.json")
    context = {
        "schema": "ebl.mnist_relu_drn.ibm_om_baseline_spacing_mapping_context",
        "schema_version": 1,
        "assignment_seed": int(assignment_seed),
        "assignment_role": assignment_role,
        "hardware_instance_id": hardware_instance_id,
        "baseline_position_fraction": mapping.baseline_position_fraction,
        "spacing_delta_x_multiplier": mapping.spacing_delta_multiples,
        "nominal_dw_min": mapping.nominal_dw_min,
        "physical_mapping_artifact": path.name,
        "physical_mapping_sha256": saved["sha256"],
        "report": dict(mapping.report),
        "directional_capacity_hashes": [
            {
                name: _tensor_sha256(getattr(capacity, name))
                for name in (
                    "group_lower_unit",
                    "group_upper_unit",
                    "group_baseline_unit",
                    "downward_level_capacity",
                    "upward_level_capacity",
                )
            }
            for capacity in mapping.directional_capacity
        ],
    }
    atomic_write_json(context_path, context)
    return {**saved, "context": str(context_path), "context_sha256": sha256_file(context_path)}


def _state_digest(state: Mapping[str, object]) -> str:
    digest = sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            digest.update(_tensor_sha256(value).encode("ascii"))
        else:
            digest.update(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
    return digest.hexdigest()


def _numeric_summary(value: torch.Tensor) -> Mapping[str, float]:
    data = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if data.numel() == 0 or not bool(torch.isfinite(data).all()):
        raise ValueError("Expected non-empty finite values.")
    absolute = data.abs()
    return {
        "signed_mean": float(data.mean().item()),
        "mae": float(absolute.mean().item()),
        "rmse": float(data.square().mean().sqrt().item()),
        "p50_absolute": float(torch.quantile(absolute, 0.5).item()),
        "p95_absolute": float(torch.quantile(absolute, 0.95).item()),
        "maximum_absolute": float(absolute.max().item()),
    }


def _level_residual_report(
    requested: torch.Tensor, residual: torch.Tensor
) -> list[Mapping[str, Any]]:
    requested = requested.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    residual = residual.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    result = []
    for level in torch.unique(requested, sorted=True).tolist():
        mask = requested == int(level)
        result.append(
            {
                "requested_level": int(level),
                "cells": int(mask.sum().item()),
                "residual": _numeric_summary(residual[mask]),
            }
        )
    return result


def _endpoint_residual_reports(
    requested: torch.Tensor,
    persistent_residual: torch.Tensor,
    apparent_residual: torch.Tensor,
) -> Mapping[str, Any]:
    """Build symmetric persistent/apparent endpoint residual diagnostics."""

    return {
        "persistent_target_residual_unit": _numeric_summary(persistent_residual),
        "apparent_target_residual_unit": _numeric_summary(apparent_residual),
        "persistent_residual_by_requested_level": _level_residual_report(
            requested, persistent_residual
        ),
        "apparent_residual_by_requested_level": _level_residual_report(
            requested, apparent_residual
        ),
    }


def _pulse_distribution_reports(programming: Any) -> Mapping[str, Any]:
    """Summarize each directional and total one-pulse controller count."""

    return {
        name: _float_summary(getattr(programming, name).to(torch.float64))
        for name in (
            "set_count",
            "reset_count",
            "total_pulses",
            "verify_count",
            "reversals",
        )
    }


def _requested_to_nearest_code_confusion(
    requested: torch.Tensor, nearest: torch.Tensor
) -> list[Mapping[str, int]]:
    """Return the complete deterministic requested-index→nearest-index table."""

    requested_array = requested.detach().to(device="cpu", dtype=torch.int64).numpy()
    nearest_array = nearest.detach().to(device="cpu", dtype=torch.int64).numpy()
    if requested_array.shape != nearest_array.shape or requested_array.size == 0:
        raise ValueError("Expected matching non-empty requested and nearest indices.")
    pairs, counts = np.unique(
        np.stack((requested_array.reshape(-1), nearest_array.reshape(-1)), axis=1),
        axis=0,
        return_counts=True,
    )
    return [
        {
            "requested_index": int(pair[0]),
            "nearest_index": int(pair[1]),
            "cells": int(count),
        }
        for pair, count in zip(pairs, counts)
    ]


def _endpoint_conductance_diagnostics(
    conductance: torch.Tensor, *, layout: str
) -> Mapping[str, Any]:
    """Summarize full-G loading and signed-transfer/loading balance."""

    contrast = quad_contrast(conductance, layout=layout).to(torch.float64)
    loading = quad_loading(conductance, layout=layout).to(torch.float64)
    loading_mean = float(loading.mean().item())
    if not math.isfinite(loading_mean) or loading_mean <= 0.0:
        raise ValueError("Expected positive finite full-conductance loading.")
    return {
        "loading": _float_summary(loading),
        "rms_contrast_over_mean_loading": float(
            contrast.square().mean().sqrt().item()
        )
        / loading_mean,
    }


def _save_pv_endpoint(
    path: Path,
    *,
    outcome: RawActiveProgramVerifyOutcome,
    population: IbmReramArrayPopulation,
    mapping: BaselineSpacingMapping,
    persistent_conductances: Sequence[torch.Tensor],
    apparent_conductances: Sequence[torch.Tensor],
    code: Any,
    assignment_seed: int,
    baseline_position_fraction: float,
    spacing_delta_x_multiplier: int,
    persistent_metrics: Mapping[str, Any],
    apparent_metrics: Mapping[str, Any],
    persistent_projection: Any,
    apparent_projection: Any,
) -> Mapping[str, Any]:
    requested = _flatten(
        tuple(layer.integer_level_number for layer in mapping.physical.layers),
        dtype=torch.int64,
    )
    verify_intersection = (
        (outcome.target_unit + outcome.tolerance_unit >= outcome.raw_lower_unit)
        & (outcome.target_unit - outcome.tolerance_unit <= outcome.raw_upper_unit)
    )
    programming = outcome.programming
    if not torch.equal(code.requested_index, requested):
        raise RuntimeError("Persistent-code classification lost requested-code order.")
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
        "persistent_applied_public_unit": persistent_projection.applied_endpoint_unit.numpy(),
        "apparent_applied_public_unit": apparent_projection.applied_endpoint_unit.numpy(),
        "persistent_below_public_minimum": persistent_projection.below_public_minimum.numpy(),
        "persistent_above_public_maximum": persistent_projection.above_public_maximum.numpy(),
        "apparent_below_public_minimum": apparent_projection.below_public_minimum.numpy(),
        "apparent_above_public_maximum": apparent_projection.above_public_maximum.numpy(),
        "persistent_public_projection_delta_unit": persistent_projection.projection_delta_unit.numpy(),
        "apparent_public_projection_delta_unit": apparent_projection.projection_delta_unit.numpy(),
        "exact_target_in_support": outcome.exact_target_in_support.numpy(),
        "verify_window_intersects_support": verify_intersection.numpy(),
        "apparent_accepted": programming.accepted.numpy(),
        "persistent_inside_acceptance_window": outcome.persistent_inside_acceptance_window.numpy(),
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
        "persistent_residual_to_requested": code.persistent_residual_to_requested.numpy(),
        "persistent_residual_to_nearest": code.persistent_residual_to_nearest.numpy(),
    }
    if not (
        len(mapping.ideal_targets)
        == len(persistent_conductances)
        == len(apparent_conductances)
        == len(LAYOUTS)
    ):
        raise RuntimeError("Expected two complete target/persistent/apparent layers.")
    layer_conductance_reports = []
    for layer_index, (target, persistent, apparent, layout) in enumerate(
        zip(
            mapping.ideal_targets,
            persistent_conductances,
            apparent_conductances,
            LAYOUTS,
        )
    ):
        if target.shape != persistent.shape or target.shape != apparent.shape:
            raise RuntimeError("Endpoint full-conductance layer shape mismatch.")
        persistent_residual_g = persistent.to(torch.float64) - target.to(torch.float64)
        apparent_residual_g = apparent.to(torch.float64) - target.to(torch.float64)
        arrays[f"layer_{layer_index}_target_full_conductance"] = target.numpy()
        arrays[f"layer_{layer_index}_persistent_full_conductance"] = persistent.numpy()
        arrays[f"layer_{layer_index}_apparent_full_conductance"] = apparent.numpy()
        arrays[f"layer_{layer_index}_persistent_residual_full_conductance"] = (
            persistent_residual_g.numpy()
        )
        arrays[f"layer_{layer_index}_apparent_residual_full_conductance"] = (
            apparent_residual_g.numpy()
        )
        endpoint_conductance_reports: dict[str, Any] = {}
        for endpoint_name, conductance in (
            ("target", target),
            ("persistent", persistent),
            ("apparent", apparent),
        ):
            arrays[f"layer_{layer_index}_{endpoint_name}_contrast"] = quad_contrast(
                conductance, layout=layout
            ).numpy()
            arrays[f"layer_{layer_index}_{endpoint_name}_loading"] = quad_loading(
                conductance, layout=layout
            ).numpy()
            if endpoint_name in {"persistent", "apparent"}:
                diagnostics = _endpoint_conductance_diagnostics(
                    conductance, layout=layout
                )
                endpoint_conductance_reports[f"{endpoint_name}_loading"] = (
                    diagnostics["loading"]
                )
                endpoint_conductance_reports[
                    f"{endpoint_name}_rms_contrast_over_mean_loading"
                ] = diagnostics["rms_contrast_over_mean_loading"]
        layer_conductance_reports.append(
            {
                "layer": layer_index,
                "layout": layout,
                "persistent_target_residual_full_conductance": _numeric_summary(
                    persistent_residual_g
                ),
                "apparent_target_residual_full_conductance": _numeric_summary(
                    apparent_residual_g
                ),
                **endpoint_conductance_reports,
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
    target_residual = outcome.persistent_endpoint_unit - outcome.target_unit
    apparent_residual = outcome.apparent_endpoint_unit - outcome.target_unit
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
        "trajectory_seed_derivation": outcome.trajectory_seed_derivation,
        "continuation_state_sha256": _state_digest(continuation),
        "cells": population.size,
        "counts": {
            "exact_target_in_support": int(outcome.exact_target_in_support.sum().item()),
            "verify_window_intersects_support": int(verify_intersection.sum().item()),
            "apparent_accepted": int(programming.accepted.sum().item()),
            "persistent_inside_acceptance_window": int(outcome.persistent_inside_acceptance_window.sum().item()),
            "requested_code_correct": int(code.requested_code_correct.sum().item()),
            "nearest_code_adjacent_to_requested": int(
                (
                    (code.nearest_persistent_index - requested).abs() == 1
                ).sum().item()
            ),
            "nearest_code_farther_than_adjacent": int(
                (
                    (code.nearest_persistent_index - requested).abs() > 1
                ).sum().item()
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
                requested,
                code.nearest_persistent_index,
            )
        ),
        "persistent_network_metrics": dict(persistent_metrics),
        "apparent_network_metrics": dict(apparent_metrics),
        "persistent_public_conductance_projection": persistent_projection.report(),
        "apparent_public_conductance_projection": apparent_projection.report(),
        "full_conductance_residuals_by_layer": layer_conductance_reports,
        **_endpoint_residual_reports(
            requested,
            target_residual,
            apparent_residual,
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
    apparent: Sequence[torch.Tensor],
    endpoint_seeds: Sequence[int],
) -> Mapping[str, Any]:
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(PREDICTION_SCHEMA),
        "schema_version": np.asarray(PREDICTION_SCHEMA_VERSION, dtype=np.int64),
        "labels": labels.numpy(),
        "teacher_prediction": teacher.numpy(),
        "continuous_prediction": continuous.numpy(),
        "ideal_quantized_prediction": ideal.numpy(),
    }
    if not (len(persistent) == len(apparent) == len(endpoint_seeds)):
        raise ValueError("Expected one persistent/apparent prediction pair per endpoint seed.")
    for endpoint_seed, prediction in zip(endpoint_seeds, persistent):
        arrays[f"pv_persistent_prediction_seed_{int(endpoint_seed)}"] = prediction.numpy()
    for endpoint_seed, prediction in zip(endpoint_seeds, apparent):
        arrays[f"pv_apparent_prediction_seed_{int(endpoint_seed)}"] = prediction.numpy()
    _atomic_save_npz(path, arrays)
    receipt_path = path.with_suffix(".receipt.json")
    ideal_correct = ideal.eq(labels)
    repeat_rows = []
    for index, (endpoint_seed, persistent_prediction, apparent_prediction) in enumerate(
        zip(endpoint_seeds, persistent, apparent)
    ):
        persistent_correct = persistent_prediction.eq(labels)
        repeat_rows.append(
            {
                "repeat_index": index,
                "endpoint_seed": int(endpoint_seed),
                "persistent_correct": int(persistent_correct.sum().item()),
                "apparent_correct": int(apparent_prediction.eq(labels).sum().item()),
                "ideal_to_persistent_wrong_to_correct": int((~ideal_correct & persistent_correct).sum().item()),
                "ideal_to_persistent_correct_to_wrong": int((ideal_correct & ~persistent_correct).sum().item()),
                "ideal_to_persistent_prediction_flip_count": int(
                    ideal.ne(persistent_prediction).sum().item()
                ),
                "persistent_prediction_sha256": _tensor_sha256(persistent_prediction),
                "apparent_prediction_sha256": _tensor_sha256(apparent_prediction),
            }
        )
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
        "repeats": repeat_rows,
    }
    atomic_write_json(receipt_path, receipt)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "report": receipt,
    }


def _save_ideal_weight_errors(
    path: Path,
    *,
    logical_weights: Sequence[torch.Tensor],
    mapping: BaselineSpacingMapping,
    analysis_scales: Sequence[float],
) -> Mapping[str, Any]:
    ideal = build_weight_error_decomposition(
        logical_weights, mapping.physical, analysis_scales
    )
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray("ebl.mnist_relu_drn.ibm_om_baseline_spacing_weight_errors"),
        "schema_version": np.asarray(1, dtype=np.int64),
    }
    for value in ideal:
        for field in fields(value):
            tensor = getattr(value, field.name)
            if isinstance(tensor, torch.Tensor):
                arrays[f"ideal_layer_{value.layer_index}_{field.name}"] = tensor.numpy()
    _atomic_save_npz(path, arrays)
    receipt_path = path.with_suffix(".receipt.json")
    receipt = {
        "schema": "ebl.mnist_relu_drn.ibm_om_baseline_spacing_weight_errors",
        "schema_version": 1,
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "analysis_scales": list(map(float, analysis_scales)),
        "ideal_layers": [
            _spacing_neutral_ideal_report(value.report) for value in ideal
        ],
    }
    atomic_write_json(receipt_path, receipt)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "report": receipt,
    }


def _save_persistent_weight_errors(
    path: Path,
    *,
    endpoint_seed: int,
    logical_weights: Sequence[torch.Tensor],
    conductances: Sequence[torch.Tensor],
    analysis_scales: Sequence[float],
) -> Mapping[str, Any]:
    reports = persistent_weight_error_report(
        logical_weights, conductances, analysis_scales
    )
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(
            "ebl.mnist_relu_drn.ibm_om_baseline_spacing_pv_weight_errors"
        ),
        "schema_version": np.asarray(1, dtype=np.int64),
        "endpoint_seed": np.asarray(endpoint_seed, dtype=np.int64),
    }
    for layer_index, conductance in enumerate(conductances):
        arrays[f"layer_{layer_index}_persistent_full_conductance"] = (
            conductance.detach().cpu().numpy()
        )
    _atomic_save_npz(path, arrays)
    receipt_path = path.with_suffix(".receipt.json")
    receipt = {
        "schema": "ebl.mnist_relu_drn.ibm_om_baseline_spacing_pv_weight_errors",
        "schema_version": 1,
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "endpoint_seed": int(endpoint_seed),
        "analysis_scales": list(map(float, analysis_scales)),
        "layers": list(reports),
    }
    atomic_write_json(receipt_path, receipt)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "report": receipt,
    }


def _hardware_summary(hardware: Mapping[str, Any]) -> Mapping[str, Any]:
    excluded = {
        "cell_lower_units",
        "cell_upper_units",
        "reset_baseline_units",
        "intrinsic_references_native",
    }
    return {key: value for key, value in hardware.items() if key not in excluded}


def run_validate(request: "ValidateRequest") -> int:
    from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_config import (
        BaselineSpacingPvValidateSpec,
    )

    spec = request.spec
    if not isinstance(spec, BaselineSpacingPvValidateSpec):
        raise TypeError("Expected the dedicated baseline-spacing P&V validate spec.")
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
        logical_weights = tuple(value.detach().cpu().clone() for value in teacher.parameters())
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
        calibration_labels = _ordered_labels(loaders.calibration)
        data_provenance = _dataset_provenance(loaders, calibration_labels)
        candidates = []
        for scale_pair in product(protocol.calibration.scale_fractions, repeat=2):
            candidate_mapping = _mapping(
                logical_weights,
                development,
                student_spec,
                alpha=alpha,
                spacing=spacing,
                scales=scale_pair,
            )
            apply_full_conductance_targets(
                dev_stack.bundle.catalog,
                candidate_mapping.continuous_targets,
                conductance_min=student_spec.model.conductance_min,
                conductance_max=student_spec.model.conductance_max,
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
            student_spec,
            alpha=alpha,
            spacing=spacing,
            scales=selected_pair,
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
        artifacts.append({**development_mapping_artifact, "kind": "ibm_om_physical_mapping"})
        artifacts.append(
            {"path": development_mapping_artifact["context"], "kind": "mapping_context"}
        )
        calibration_receipt = {
            "schema": "ebl.mnist_relu_drn.ibm_om_baseline_spacing_development_calibration",
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
            "calibration_labels_sha256": data_provenance["calibration_labels_sha256"],
            "calibration_indices_sha256": data_provenance["calibration_indices_sha256"],
            "test_labels_used": False,
        }
        calibration_path = artifact_root / "development" / "calibration.json"
        atomic_write_json(calibration_path, calibration_receipt)
        artifacts.append({"path": str(calibration_path), "kind": "development_calibration"})
        print(
            f"calibrated alpha={alpha:g}: scales={selected_pair}, gain={selected_gain:.8g}",
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
        heldout_mapping = _mapping(
            logical_weights,
            heldout,
            student_spec,
            alpha=alpha,
            spacing=spacing,
            scales=selected_pair,
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
                "kind": "ibm_om_baseline_spacing_physical_mapping",
            }
        )
        artifacts.append({"path": heldout_mapping_artifact["context"], "kind": "mapping_context"})
        heldout_stack.cost.gain = selected_gain
        apply_full_conductance_targets(
            heldout_stack.bundle.catalog,
            heldout_mapping.continuous_targets,
            conductance_min=student_spec.model.conductance_min,
            conductance_max=student_spec.model.conductance_max,
        )
        continuous_metrics, continuous_prediction = _evaluate_detailed(
            heldout_stack, teacher, loaders.test, sample_limit=student_spec.settings.sample_limit
        )
        apply_full_conductance_targets(
            heldout_stack.bundle.catalog,
            heldout_mapping.ideal_targets,
            conductance_min=student_spec.model.conductance_min,
            conductance_max=student_spec.model.conductance_max,
        )
        ideal_metrics, ideal_prediction = _evaluate_detailed(
            heldout_stack, teacher, loaders.test, sample_limit=student_spec.settings.sample_limit
        )
        labels, teacher_prediction = _teacher_predictions(
            teacher,
            loaders.test,
            device=device,
            sample_limit=student_spec.settings.sample_limit,
        )
        expected_examples = 10_000 if protocol.execution.profile == "production" else 32
        if not (
            continuous_metrics["examples"]
            == ideal_metrics["examples"]
            == labels.numel()
            == expected_examples
        ):
            raise RuntimeError("Ideal endpoints do not have exact declared coverage.")

        population = _joint_population(heldout)
        expected_binding_shapes = tuple(
            tuple(value.shape) for value in heldout_mapping.ideal_target_units
        )
        if population.binding_shapes != expected_binding_shapes:
            raise RuntimeError("Joint P&V population does not match mapped rail order.")
        target_unit = _flatten(heldout_mapping.ideal_target_units, dtype=torch.float32)
        population_lower_unit = (
            (population.min_bound.to(torch.float64) + 1.0) / 2.0
        ).clamp(0.0, 1.0)
        population_upper_unit = (
            (population.max_bound.to(torch.float64) + 1.0) / 2.0
        ).clamp(0.0, 1.0)
        if not (
            torch.equal(
                population_lower_unit,
                _flatten(heldout["cell_lower_units"], dtype=torch.float64),
            )
            and torch.equal(
                population_upper_unit,
                _flatten(heldout["cell_upper_units"], dtype=torch.float64),
            )
        ):
            raise RuntimeError("Joint P&V population bounds do not match the mapping identity.")
        target_projection = project_raw_active_unit_to_full_conductance(
            target_unit,
            conductance_min=student_spec.model.conductance_min,
            conductance_max=student_spec.model.conductance_max,
        )
        if target_projection.projected_count:
            raise RuntimeError("An ideal P&V target left the public conductance range.")
        target_full_g = _flatten(heldout_mapping.ideal_targets, dtype=torch.float32)
        target_handoff_residual = (
            target_projection.full_conductance.to(torch.float64)
            - target_full_g.to(torch.float64)
        )
        target_handoff_tolerance = max(
            1e-12,
            (
                student_spec.model.conductance_max
                - student_spec.model.conductance_min
            )
            * 1e-6,
        )
        if float(target_handoff_residual.abs().max().item()) > target_handoff_tolerance:
            raise RuntimeError("Raw-x P&V targets do not reconstruct the ideal full G.")
        target_handoff_report = {
            **target_projection.report(),
            "ideal_full_conductance_sha256": _tensor_sha256(target_full_g),
            "maximum_absolute_reconstruction_residual": float(
                target_handoff_residual.abs().max().item()
            ),
            "reconstruction_tolerance": target_handoff_tolerance,
        }
        baseline_unit = _flatten(
            tuple(layer.baseline_unit for layer in heldout_mapping.physical.layers),
            dtype=torch.float32,
        )
        requested_index = _flatten(
            tuple(layer.integer_level_number for layer in heldout_mapping.physical.layers),
            dtype=torch.int64,
        )
        cell_upper = _flatten(heldout["cell_upper_units"], dtype=torch.float64)
        maximum_index = _maximum_supported_uniform_index(
            cell_upper,
            baseline_unit,
            spacing_unit=float(heldout_mapping.report["level_spacing_unit"]),
        )
        if bool(torch.any(maximum_index < requested_index)):
            raise RuntimeError("Requested ideal code exceeds a physical cell capacity.")

        persistent_metrics = []
        apparent_metrics = []
        persistent_predictions = []
        apparent_predictions = []
        persistent_conductance_repeats = []
        persistent_weight_artifacts = []
        pv_artifacts = []
        tolerance_unit = (
            protocol.program_verify.tolerance_delta_x_ratio
            * float(heldout_mapping.report["delta_x"])
        )
        for repeat_index, endpoint_seed in enumerate(protocol.assignments.endpoint_seeds):
            outcome = run_raw_active_program_verify(
                population,
                targets_unit=target_unit,
                endpoint_seed=int(endpoint_seed),
                tolerance_unit=tolerance_unit,
                maximum_program_pulses=protocol.program_verify.maximum_program_pulses,
                device=device,
                settings=ControllerSettings(kind=protocol.program_verify.controller),
            )
            code = classify_persistent_uniform_codes(
                outcome.persistent_endpoint_unit,
                baseline_unit=baseline_unit,
                spacing_unit=float(heldout_mapping.report["level_spacing_unit"]),
                requested_index=requested_index,
                maximum_index=maximum_index,
            )
            persistent_projection = project_raw_active_unit_to_full_conductance(
                outcome.persistent_endpoint_unit,
                conductance_min=student_spec.model.conductance_min,
                conductance_max=student_spec.model.conductance_max,
            )
            apparent_projection = project_raw_active_unit_to_full_conductance(
                outcome.apparent_endpoint_unit,
                conductance_min=student_spec.model.conductance_min,
                conductance_max=student_spec.model.conductance_max,
            )
            persistent_flat_g = persistent_projection.full_conductance.to(torch.float32)
            apparent_flat_g = apparent_projection.full_conductance.to(torch.float32)
            persistent_g = _split_population_vector(persistent_flat_g, population)
            apparent_g = _split_population_vector(apparent_flat_g, population)
            apply_full_conductance_targets(
                heldout_stack.bundle.catalog,
                persistent_g,
                conductance_min=student_spec.model.conductance_min,
                conductance_max=student_spec.model.conductance_max,
            )
            persistent_result, persistent_prediction = _evaluate_detailed(
                heldout_stack,
                teacher,
                loaders.test,
                sample_limit=student_spec.settings.sample_limit,
            )
            apply_full_conductance_targets(
                heldout_stack.bundle.catalog,
                apparent_g,
                conductance_min=student_spec.model.conductance_min,
                conductance_max=student_spec.model.conductance_max,
            )
            apparent_result, apparent_prediction = _evaluate_detailed(
                heldout_stack,
                teacher,
                loaders.test,
                sample_limit=student_spec.settings.sample_limit,
            )
            if persistent_result["examples"] != expected_examples or apparent_result["examples"] != expected_examples:
                raise RuntimeError("P&V endpoint does not have exact declared coverage.")
            pv_artifact = _save_pv_endpoint(
                artifact_root
                / "heldout"
                / f"program_verify_endpoint_seed_{int(endpoint_seed)}.npz",
                outcome=outcome,
                population=population,
                mapping=heldout_mapping,
                persistent_conductances=persistent_g,
                apparent_conductances=apparent_g,
                code=code,
                assignment_seed=heldout_seed,
                baseline_position_fraction=alpha,
                spacing_delta_x_multiplier=spacing,
                persistent_metrics=persistent_result,
                apparent_metrics=apparent_result,
                persistent_projection=persistent_projection,
                apparent_projection=apparent_projection,
            )
            artifacts.append(
                {
                    **pv_artifact,
                    "kind": "ibm_om_baseline_spacing_pv_endpoint",
                }
            )
            pv_artifacts.append(pv_artifact)
            persistent_metrics.append(persistent_result)
            apparent_metrics.append(apparent_result)
            persistent_predictions.append(persistent_prediction)
            apparent_predictions.append(apparent_prediction)
            persistent_conductance_repeats.append(persistent_g)
            persistent_weight_artifact = _save_persistent_weight_errors(
                artifact_root
                / "heldout"
                / f"program_verify_weight_errors_seed_{int(endpoint_seed)}.npz",
                endpoint_seed=int(endpoint_seed),
                logical_weights=logical_weights,
                conductances=persistent_g,
                analysis_scales=analysis_scales,
            )
            artifacts.append(
                {
                    **persistent_weight_artifact,
                    "kind": "program_verify_weight_errors",
                }
            )
            persistent_weight_artifacts.append(persistent_weight_artifact)
            print(
                f"heldout={heldout_seed} alpha={alpha:g} h={spacing}delta "
                f"pv[{repeat_index}]={100.0 * persistent_result['student_accuracy']:.2f}% "
                f"accepted={pv_artifact['report']['counts']['apparent_accepted']}/{population.size}",
                flush=True,
            )
            del outcome, code, persistent_projection, apparent_projection
            del persistent_flat_g, apparent_flat_g
            gc.collect()
            torch.cuda.empty_cache()

        prediction_artifact = _save_predictions(
            artifact_root / "heldout" / "predictions.npz",
            labels=labels,
            teacher=teacher_prediction,
            continuous=continuous_prediction,
            ideal=ideal_prediction,
            persistent=persistent_predictions,
            apparent=apparent_predictions,
            endpoint_seeds=protocol.assignments.endpoint_seeds,
        )
        artifacts.append({**prediction_artifact, "kind": "predictions"})
        weight_artifact = _save_ideal_weight_errors(
            artifact_root / "heldout" / "ideal_weight_errors.npz",
            logical_weights=logical_weights,
            mapping=heldout_mapping,
            analysis_scales=analysis_scales,
        )
        artifacts.append({**weight_artifact, "kind": "ideal_weight_errors"})

        parity = protocol.reference_study.alpha0_spacing4_parity
        parity_index = parity.heldout_assignment_seeds.index(heldout_seed)
        parity_design_arm = alpha == 0.0 and spacing == 4
        full_test_output_parity_required, parity_gates = _historical_parity_gates(
            require_full_test_output=(
                parity_design_arm and protocol.execution.profile == "production"
            ),
            parity=parity,
            parity_index=parity_index,
            development_hardware_instance_id=development["hardware_instance_id"],
            heldout_hardware_instance_id=heldout["hardware_instance_id"],
            selected_pair=selected_pair,
            selected_gain=selected_gain,
            ideal_metrics=ideal_metrics,
        )
        if parity_design_arm and not all(parity_gates.values()):
            raise RuntimeError(f"Alpha-zero/four-delta backward parity failed: {parity_gates!r}.")
        pv_mean_accuracy = sum(value["student_accuracy"] for value in persistent_metrics) / len(persistent_metrics)
        validity_gates = {
            "cuda": student_spec.runtime.device == "cuda" and torch.cuda.is_available(),
            "development_invariants": bool(development_invariants["valid"]),
            "heldout_invariants": bool(heldout_invariants["valid"]),
            "development_candidate_count_16": len(candidates) == 16,
            "repeat_count": len(persistent_metrics) == protocol.program_verify.repeat_count,
            "endpoint_seed_count": len(protocol.assignments.endpoint_seeds) == protocol.program_verify.repeat_count,
            "full_test_coverage": all(value["examples"] == expected_examples for value in persistent_metrics),
            "all_targets_in_support": all(
                item["report"]["counts"]["exact_target_in_support"] == population.size
                for item in pv_artifacts
            ),
            "no_nonfinite_programming": all(
                item["report"]["counts"]["nonfinite"] == 0 for item in pv_artifacts
            ),
            "inference_read_noise_off": not protocol.program_verify.inference_read_noise,
            "optimizer_updates_zero": protocol.exclusions.optimizer_updates == 0,
            "historical_development_hardware": parity_gates[
                "development_hardware_instance_id"
            ],
            "historical_heldout_hardware": parity_gates[
                "heldout_hardware_instance_id"
            ],
            "parity_if_required": (not parity_design_arm) or all(parity_gates.values()),
        }
        if not all(validity_gates.values()):
            raise RuntimeError(
                "Refusing to complete invalid baseline-spacing P&V evidence: "
                f"{[name for name, value in validity_gates.items() if not value]!r}."
            )

        source_report = {
            "path": str(weights_path),
            "sha256": sha256_file(weights_path),
            "metadata": teacher_metadata,
            "logical_weight_hashes": [_tensor_sha256(value) for value in logical_weights],
            "optimizer_updates": 0,
        }
        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "complete",
            "claim_boundary": (
                "Model-based AIHWKit 1.1.0 OM preset; four-device shared-destination "
                "initialization with raw-active pulse-resolved P&V. No inference read "
                "noise, optimizer update, HWA, retention, or fabricated-device claim."
            ),
            "source": source_report,
            "data": data_provenance,
            "contract": to_plain_data(protocol),
            "design": {
                "baseline_position_fraction": alpha,
                "spacing_delta_x_multiplier": spacing,
                "level_spacing_unit": heldout_mapping.report["level_spacing_unit"],
            },
            "development": {
                "hardware": _hardware_summary(development),
                "calibration": calibration_receipt,
                "mapping": development_mapping.report,
                "invariants": development_invariants,
            },
            "heldout": {
                "assignment_seed": heldout_seed,
                "hardware": _hardware_summary(heldout),
                "mapping": heldout_mapping.report,
                "invariants": heldout_invariants,
                "target_public_conductance_handoff": target_handoff_report,
                "continuous": continuous_metrics,
                "ideal_quantized": ideal_metrics,
                "pv_persistent_repeats": persistent_metrics,
                "pv_apparent_repeats": apparent_metrics,
                "pv_endpoint_artifacts": [item["report"] for item in pv_artifacts],
                "pv_persistent_mean_accuracy": pv_mean_accuracy,
                "weight_errors": weight_artifact["report"],
                "pv_persistent_weight_errors": [
                    item["report"] for item in persistent_weight_artifacts
                ],
                "predictions": prediction_artifact["report"],
            },
            "parity": _historical_parity_summary(
                design_arm=parity_design_arm,
                full_test_output_required=full_test_output_parity_required,
                gates=parity_gates,
            ),
            "validity": {
                "coverage_valid": True,
                "gates": validity_gates,
                "expected_examples": expected_examples,
                "optimizer_updates": 0,
                "program_verify_enabled": True,
                "inference_read_noise": 0.0,
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
            "development_hardware_instance_id": development["hardware_instance_id"],
            "heldout_hardware_instance_id": heldout["hardware_instance_id"],
            "selected_scale_fractions": list(selected_pair),
            "fixed_logit_gain": selected_gain,
            "continuous": {
                key: continuous_metrics[key]
                for key in ("student_correct", "examples", "student_accuracy", "prediction_sha256")
            },
            "ideal_quantized": {
                key: ideal_metrics[key]
                for key in ("student_correct", "examples", "student_accuracy", "prediction_sha256")
            },
            "pv_persistent": [
                {
                    key: value[key]
                    for key in ("student_correct", "examples", "student_accuracy", "prediction_sha256")
                }
                for value in persistent_metrics
            ],
            "pv_persistent_mean_accuracy": pv_mean_accuracy,
            "coverage_valid": True,
        }
        store.append_metric({"mode": "validate", **terminal_metrics})
        store.complete(metrics=terminal_metrics, artifacts=_artifact_records(store, artifacts))
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = ["run_validate"]
