"""Fail-closed analysis for the IBM OM baseline/spacing P&V study.

The analyzer performs no network inference.  It verifies all registered bytes,
reconstructs the full-conductance mapping and pulse-endpoint classifications,
checks the matched hardware/RNG/calibration design, and then applies the
predeclared ideal gate and persistent-P&V winner rule.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from itertools import product
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.artifacts import (
    atomic_write_json,
    canonical_json_bytes,
    content_hash,
    sha256_file,
)
from experiments.mnist_relu_drn.analyze_ibm_om_baseline_selection import (
    BaselineSelectionAnalysisError,
    _TENSOR_FIELDS,
    _WEIGHT_FIELDS,
    _artifact_index,
    _atomic_text,
    _digest,
    _equal,
    _error_summary,
    _exact_keys,
    _fail,
    _git_commit,
    _json,
    _kind_paths,
    _npz,
    _raw_hash,
    _registered,
    _scalar,
    _summary,
    _torch_hash,
    _validate_data_provenance,
    _validate_hardware_sources,
    _validate_joint,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    LAYOUTS,
    MAPPING_SCHEMA,
    MAPPING_SCHEMA_VERSION,
    SHARED_DESTINATION_COLUMNS_RESET_MAX,
    baseline_group_violation_mask,
    quad_column_loading,
    quad_contrast,
    quad_loading,
    quad_row_loading,
    quad_stack,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv import (
    build_baseline_spacing_mapping,
    persistent_weight_error_report,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_config import (
    BASELINE_POSITION_FRACTIONS,
    CONTINUOUS_DIAGNOSTIC_METRIC_DEFINITION,
    DEVELOPMENT_ASSIGNMENT_SEED,
    ENDPOINT_SEEDS_BY_ASSIGNMENT,
    EXPECTED_WEIGHTS_SHA256,
    EXPERIMENT_ID,
    HELDOUT_ASSIGNMENT_SEEDS,
    IDEAL_QUANTIZED_METRIC_DEFINITION,
    MAXIMUM_PROGRAM_PULSES,
    P_AND_V_REPEAT_COUNT,
    PV_PERSISTENT_METRIC_DEFINITION,
    SPACING_DELTA_X_MULTIPLIERS,
    STUDY_ID,
    VERIFY_TOLERANCE_DELTA_X_RATIO,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv_runtime import (
    PREDICTION_SCHEMA,
    PREDICTION_SCHEMA_VERSION,
    PV_SCHEMA,
    PV_SCHEMA_VERSION,
    SUMMARY_SCHEMA,
    SUMMARY_SCHEMA_VERSION,
    _level_residual_report,
    _maximum_supported_uniform_index,
    _numeric_summary,
    _state_digest,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _float_summary,
)
from experiments.study_workflow import load_study_record
from training.ibm_reram_hwa import load_om_array_population
from training.ibm_reram_raw_active_program_verify import (
    RAW_ACTIVE_COORDINATE,
    TRAJECTORY_SEED_DERIVATION,
    matched_trajectory_seeds,
    project_raw_active_unit_to_full_conductance,
)


BaselineSpacingPvAnalysisError = BaselineSelectionAnalysisError

ANALYSIS_SCHEMA = "ebl.mnist_relu_drn.ibm_om_baseline_spacing_pv_analysis"
ANALYSIS_SCHEMA_VERSION = 1
EXPECTED_EXAMPLES = 10_000
EXPECTED_RUNS = 27
PHYSICAL_POLICY = SHARED_DESTINATION_COLUMNS_RESET_MAX
MAPPING_CONTEXT_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_baseline_spacing_mapping_context"
)
CALIBRATION_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_baseline_spacing_development_calibration"
)
IDEAL_WEIGHT_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_baseline_spacing_weight_errors"
)
PV_WEIGHT_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_baseline_spacing_pv_weight_errors"
)
OUTPUT_JSON = "baseline_spacing_pv_analysis.json"
OUTPUT_CSV = "baseline_spacing_pv_runs.csv"
OUTPUT_MARKDOWN = "baseline_spacing_pv_analysis.md"

_IDEAL_ERROR_COMPONENT_KEYS = {
    "baseline",
    "envelope",
    "continuous_total",
    "quantization",
    "ideal_quantized_total",
}
_IDEAL_ENDPOINT_KEYS = {
    "relative_l2",
    "cosine_similarity",
    "opposite_sign_count",
    "erased_to_zero_count",
    "sign_error_union_count",
}


def _alpha_token(alpha: float) -> str:
    return {0.0: "000", 0.25: "025", 0.5: "050"}[float(alpha)]


def _arm_id(alpha: float, spacing: int) -> str:
    return f"alpha-{_alpha_token(alpha)}-spacing-{int(spacing)}delta"


DESIGNS = tuple(
    (float(alpha), int(spacing), _arm_id(alpha, spacing))
    for alpha in BASELINE_POSITION_FRACTIONS
    for spacing in SPACING_DELTA_X_MULTIPLIERS
)
ARM_DESIGNS = {arm: (alpha, spacing) for alpha, spacing, arm in DESIGNS}


def _integer_histogram(value: torch.Tensor) -> list[dict[str, int]]:
    flat = value.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    unique, counts = torch.unique(flat, sorted=True, return_counts=True)
    return [
        {"index": int(index), "count": int(count)}
        for index, count in zip(unique.tolist(), counts.tolist())
    ]


def _requested_to_nearest_code_confusion(
    requested: np.ndarray, nearest: np.ndarray
) -> list[dict[str, int]]:
    requested_array = np.asarray(requested, dtype=np.int64)
    nearest_array = np.asarray(nearest, dtype=np.int64)
    if requested_array.shape != nearest_array.shape or requested_array.size == 0:
        _fail("Expected matching non-empty requested and nearest code indices.")
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


def _zero_only_initialization_group_count(
    upward_level_capacity: torch.Tensor,
) -> int:
    """Count groups unable to express any positive initialization level."""

    return int((upward_level_capacity == 0).sum().item())


def _rms_contrast_over_mean_loading(
    contrast: torch.Tensor, loading: torch.Tensor
) -> float:
    contrast_rms = float(
        contrast.detach().to(torch.float64).square().mean().sqrt().item()
    )
    loading_mean = float(loading.detach().to(torch.float64).mean().item())
    if not math.isfinite(loading_mean) or loading_mean <= 0.0:
        _fail("Expected positive finite mean full-conductance loading.")
    return contrast_rms / loading_mean


def _finite_number(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"Expected finite numeric {label}.")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        _fail(f"Expected finite{' positive' if positive else ''} {label}.")
    return result


def _metric_from_prediction(
    metric: Mapping[str, Any],
    prediction: np.ndarray,
    labels: np.ndarray,
    *,
    teacher_prediction: np.ndarray,
    expected_gain: float,
) -> None:
    _exact_keys(
        metric,
        {
            "examples",
            "student_correct",
            "teacher_correct",
            "teacher_agreement_count",
            "kl_teacher_student",
            "raw_kl_teacher_student",
            "student_accuracy",
            "teacher_accuracy",
            "teacher_agreement",
            "raw_score_rms",
            "calibrated_score_rms",
            "teacher_logit_rms",
            "fixed_logit_gain",
            "prediction_sha256",
            "voltage",
        },
        label="detailed network metric",
    )
    examples = int(labels.size)
    if teacher_prediction.shape != labels.shape:
        _fail("Expected teacher prediction and label shapes to match.")
    _equal(metric.get("examples"), examples, label="metric examples")
    _equal(
        metric.get("student_correct"),
        int(np.count_nonzero(prediction == labels)),
        label="metric exact correct count",
    )
    _equal(
        metric.get("student_accuracy"),
        int(np.count_nonzero(prediction == labels)) / examples,
        label="metric exact accuracy",
    )
    _equal(
        metric.get("prediction_sha256"),
        _torch_hash(prediction),
        label="metric prediction hash",
    )
    teacher_correct = int(np.count_nonzero(teacher_prediction == labels))
    agreement = int(np.count_nonzero(prediction == teacher_prediction))
    _equal(metric["teacher_correct"], teacher_correct, label="metric teacher correct")
    _equal(
        metric["teacher_agreement_count"],
        agreement,
        label="metric teacher agreement count",
    )
    _equal(
        metric["teacher_accuracy"],
        teacher_correct / examples,
        label="metric teacher accuracy",
    )
    _equal(
        metric["teacher_agreement"],
        agreement / examples,
        label="metric teacher agreement",
    )
    _equal(metric["fixed_logit_gain"], expected_gain, label="metric frozen gain")
    for key in (
        "kl_teacher_student",
        "raw_kl_teacher_student",
        "raw_score_rms",
        "calibrated_score_rms",
        "teacher_logit_rms",
    ):
        value = _finite_number(metric[key], label=f"metric {key}")
        if value < -1e-12:
            _fail(f"Expected nonnegative detailed metric {key!r}.")
    voltage = metric.get("voltage")
    if not isinstance(voltage, list) or len(voltage) != 3:
        _fail("Expected input, hidden, and output voltage reports.")
    expected_values_per_example = (1568, 100, 20)
    voltage_keys = {
        "layer",
        "values",
        "mean",
        "rms",
        "standard_deviation",
        "minimum",
        "maximum",
    }
    for layer, (item, multiplier) in enumerate(
        zip(voltage, expected_values_per_example)
    ):
        if not isinstance(item, dict):
            _fail("Expected mapping-valued voltage report entries.")
        _exact_keys(item, voltage_keys, label=f"metric voltage layer {layer}")
        _equal(item["layer"], layer, label="metric voltage layer index")
        _equal(
            item["values"],
            examples * multiplier,
            label=f"metric voltage layer {layer} value count",
        )
        mean = _finite_number(item["mean"], label="voltage mean")
        rms = _finite_number(item["rms"], label="voltage RMS")
        deviation = _finite_number(
            item["standard_deviation"], label="voltage standard deviation"
        )
        minimum = _finite_number(item["minimum"], label="voltage minimum")
        maximum = _finite_number(item["maximum"], label="voltage maximum")
        if rms < 0.0 or deviation < 0.0 or not minimum <= mean <= maximum:
            _fail("Voltage report has invalid moments or extrema.")
        if not math.isclose(
            rms * rms,
            deviation * deviation + mean * mean,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            _fail("Voltage RMS is inconsistent with mean and standard deviation.")


def _validate_summary_parity(
    observed: Any,
    *,
    design_arm: bool,
    full_test_output_required: bool,
    gates: Mapping[str, bool],
) -> None:
    """Validate the exact parity block emitted by the native runtime."""

    _equal(
        observed,
        {
            "design_arm": bool(design_arm),
            "full_test_output_required": bool(full_test_output_required),
            "gates": dict(gates),
        },
        label="summary parity",
    )


def _validate_ideal_weight_report_shape(
    report: Mapping[str, Any], *, layer: int
) -> None:
    """Reject inherited fixed-spacing labels and require neutral report keys."""

    _exact_keys(
        report,
        {
            "layer",
            "analysis_scale",
            "raw_contrast_error",
            "normalized_conductance_coordinate_error",
            "normalized_relu_weight_error",
            "teacher_weight_error",
            "continuous_endpoint",
            "ideal_quantized_endpoint",
            "nonzero_source_count",
            "quantization_error_in_levels",
            "component_sum_max_abs_residual",
        },
        label=f"ideal weight-error layer {layer} report",
    )
    for key in (
        "raw_contrast_error",
        "normalized_conductance_coordinate_error",
        "normalized_relu_weight_error",
        "teacher_weight_error",
    ):
        value = report[key]
        if not isinstance(value, dict):
            _fail(f"Expected mapping-valued {key!r} in ideal weight report.")
        _exact_keys(
            value,
            _IDEAL_ERROR_COMPONENT_KEYS,
            label=f"ideal weight-error layer {layer} {key}",
        )
    for key in ("continuous_endpoint", "ideal_quantized_endpoint"):
        value = report[key]
        if not isinstance(value, dict):
            _fail(f"Expected mapping-valued {key!r} in ideal weight report.")
        _exact_keys(
            value,
            _IDEAL_ENDPOINT_KEYS,
            label=f"ideal weight-error layer {layer} {key}",
        )


def _mapping_fields() -> tuple[set[str], set[str]]:
    scalar = {
        "schema",
        "schema_version",
        "policy",
        "assignment_seed",
        "assignment_role",
        "hardware_instance_id",
        "conductance_min",
        "conductance_max",
    }
    tensors = {
        f"layer_{layer}_{name}"
        for layer in range(2)
        for name in _TENSOR_FIELDS
    }
    return scalar, tensors


def _validate_mapping(
    path: Path,
    receipt_path: Path,
    context_path: Path,
    *,
    alpha: float,
    spacing: int,
    assignment_seed: int,
    assignment_role: str,
    hardware_instance_id: str,
    nominal_dw_min: float,
    joint_arrays: Mapping[str, np.ndarray],
    source_logical_weight_hashes: Sequence[str],
    scale_fractions: Sequence[float],
    expected_conductance_min: float,
    expected_conductance_max: float,
) -> dict[str, Any]:
    arrays = _npz(path, label="baseline/spacing physical mapping")
    receipt = _json(receipt_path, label="physical-mapping receipt")
    context = _json(context_path, label="mapping design context")
    scalar_fields, tensor_fields = _mapping_fields()
    _exact_keys(arrays, scalar_fields | tensor_fields, label="physical-mapping NPZ")
    _exact_keys(
        receipt,
        {
            "schema",
            "schema_version",
            "policy",
            "assignment_seed",
            "assignment_role",
            "hardware_instance_id",
            "artifact",
            "artifact_sha256",
            "tensor_hashes",
            "mapping",
        },
        label="physical-mapping receipt",
    )
    _exact_keys(
        context,
        {
            "schema",
            "schema_version",
            "assignment_seed",
            "assignment_role",
            "hardware_instance_id",
            "baseline_position_fraction",
            "spacing_delta_x_multiplier",
            "nominal_dw_min",
            "physical_mapping_artifact",
            "physical_mapping_sha256",
            "report",
            "directional_capacity_hashes",
        },
        label="mapping design context",
    )
    expected = {
        "schema": MAPPING_SCHEMA,
        "schema_version": MAPPING_SCHEMA_VERSION,
        "policy": PHYSICAL_POLICY,
        "assignment_seed": int(assignment_seed),
        "assignment_role": assignment_role,
        "hardware_instance_id": hardware_instance_id,
    }
    for name, wanted in expected.items():
        _equal(receipt[name], wanted, label=f"mapping receipt {name}")
        observed = _scalar(arrays, name)
        if name in {"schema_version", "assignment_seed"}:
            observed = int(observed)
        else:
            observed = str(observed)
        _equal(observed, wanted, label=f"mapping NPZ {name}")
    _equal(receipt["artifact"], path.name, label="mapping artifact name")
    _equal(receipt["artifact_sha256"], sha256_file(path), label="mapping hash")
    _equal(
        receipt["tensor_hashes"],
        {name: _torch_hash(arrays[name]) for name in sorted(tensor_fields)},
        label="mapping tensor hashes",
    )
    _equal(context["schema"], MAPPING_CONTEXT_SCHEMA, label="mapping context schema")
    _equal(context["schema_version"], 1, label="mapping context version")
    _equal(context["assignment_seed"], assignment_seed, label="mapping context seed")
    _equal(context["assignment_role"], assignment_role, label="mapping context role")
    _equal(context["hardware_instance_id"], hardware_instance_id, label="mapping context hardware")
    _equal(context["baseline_position_fraction"], alpha, label="mapping alpha")
    _equal(context["spacing_delta_x_multiplier"], spacing, label="mapping spacing")
    _equal(context["nominal_dw_min"], nominal_dw_min, label="mapping nominal step")
    _equal(context["physical_mapping_artifact"], path.name, label="mapping context artifact")
    _equal(context["physical_mapping_sha256"], sha256_file(path), label="mapping context hash")

    g_min = float(_scalar(arrays, "conductance_min"))
    g_max = float(_scalar(arrays, "conductance_max"))
    if not (math.isfinite(g_min) and g_min >= 0.0 and g_max > g_min):
        _fail("Expected a finite increasing nonnegative conductance interval.")
    _equal(g_min, expected_conductance_min, label="configured conductance minimum")
    _equal(g_max, expected_conductance_max, label="configured conductance maximum")
    span = g_max - g_min
    delta_x = float(nominal_dw_min) / 2.0
    spacing_unit = int(spacing) * delta_x
    spacing_physical = span * spacing_unit
    report = context["report"]
    if not isinstance(report, dict):
        _fail("Expected a mapping design report object.")
    _equal(report.get("baseline_position_fraction"), alpha, label="design report alpha")
    _equal(report.get("spacing_delta_multiples"), spacing, label="design report spacing")
    _equal(report.get("delta_x"), delta_x, label="design report delta_x")
    _equal(report.get("level_spacing_unit"), spacing_unit, label="design report level spacing")
    _equal(report.get("physical"), receipt["mapping"], label="design/physical report binding")
    physical_report = receipt["mapping"]
    _equal(physical_report.get("policy"), PHYSICAL_POLICY, label="physical policy")
    _equal(physical_report.get("baseline_position_fraction"), alpha, label="physical report alpha")
    _equal(physical_report.get("spacing_delta_multiples"), spacing, label="physical report spacing")
    layer_reports = physical_report.get("layers")
    if not isinstance(layer_reports, list) or len(layer_reports) != 2:
        _fail("Expected two physical mapping layer reports.")
    directional_hashes = context["directional_capacity_hashes"]
    if not isinstance(directional_hashes, list) or len(directional_hashes) != 2:
        _fail("Expected two directional-capacity hash records.")

    if len(source_logical_weight_hashes) != 2 or len(scale_fractions) != 2:
        _fail("Expected two frozen source hashes and mapping scales.")
    source_weights = tuple(
        torch.from_numpy(
            np.ascontiguousarray(arrays[f"layer_{layer}_source_weight"])
        )
        for layer in range(2)
    )
    for layer, (source, digest) in enumerate(
        zip(source_weights, source_logical_weight_hashes)
    ):
        _digest(digest, label=f"source logical weight layer {layer}")
        _equal(
            _torch_hash(source.numpy()),
            digest,
            label=f"mapping/source logical weight layer {layer}",
        )
    lower_units = tuple(
        ((
            torch.from_numpy(
                np.ascontiguousarray(joint_arrays[f"layer_{layer}_min_bound"])
            ).to(torch.float64)
            + 1.0
        ) / 2.0).clamp(0.0, 1.0)
        for layer in range(2)
    )
    upper_units = tuple(
        ((
            torch.from_numpy(
                np.ascontiguousarray(joint_arrays[f"layer_{layer}_max_bound"])
            ).to(torch.float64)
            + 1.0
        ) / 2.0).clamp(0.0, 1.0)
        for layer in range(2)
    )
    reset_units = tuple(
        torch.minimum(
            torch.maximum(
                ((
                    torch.from_numpy(
                        np.ascontiguousarray(
                            joint_arrays[f"layer_{layer}_reset_mean_raw_a"]
                        )
                    ).to(torch.float64)
                    + 1.0
                ) / 2.0).clamp(0.0, 1.0),
                lower_units[layer],
            ),
            upper_units[layer],
        )
        for layer in range(2)
    )
    references = tuple(
        torch.from_numpy(
            np.ascontiguousarray(joint_arrays[f"layer_{layer}_reference"])
        )
        for layer in range(2)
    )
    reconstructed_mapping = build_baseline_spacing_mapping(
        logical_weights=source_weights,
        baseline_position_fraction=alpha,
        spacing_delta_multiples=spacing,
        scale_fractions=scale_fractions,
        nominal_dw_min=nominal_dw_min,
        conductance_min=expected_conductance_min,
        conductance_max=expected_conductance_max,
        cell_lower_units=lower_units,
        cell_upper_units=upper_units,
        reset_baseline_units=reset_units,
        intrinsic_references_native=references,
    )

    layer_values: list[dict[str, np.ndarray]] = []
    diagnostics = []
    for layer_index, layout in enumerate(LAYOUTS):
        values = {
            name: arrays[f"layer_{layer_index}_{name}"]
            for name in _TENSOR_FIELDS
        }
        layer_values.append(values)
        reconstructed_layer = reconstructed_mapping.physical.layers[layer_index]
        for name in _TENSOR_FIELDS:
            expected_tensor = getattr(reconstructed_layer, name)
            observed_tensor = torch.from_numpy(np.ascontiguousarray(values[name]))
            if not torch.equal(observed_tensor, expected_tensor):
                _fail(
                    f"Mapping tensor {name!r} is not reproducible from the "
                    "bound source, joint hardware, commissioning, and calibration."
                )
        baseline = torch.from_numpy(np.ascontiguousarray(values["baseline"]))
        baseline_unit = torch.from_numpy(np.ascontiguousarray(values["baseline_unit"]))
        continuous_offset = torch.from_numpy(
            np.ascontiguousarray(values["continuous_offset"])
        )
        quantized_offset = torch.from_numpy(
            np.ascontiguousarray(values["quantized_offset"])
        )
        continuous = torch.from_numpy(
            np.ascontiguousarray(values["continuous_conductance"])
        )
        ideal = torch.from_numpy(
            np.ascontiguousarray(values["quantized_conductance"])
        )
        integer = torch.from_numpy(
            np.ascontiguousarray(values["integer_level_number"])
        ).to(torch.int64)
        lower_unit = torch.from_numpy(
            np.ascontiguousarray(values["cell_lower_unit"])
        ).to(torch.float64)
        upper_unit = torch.from_numpy(
            np.ascontiguousarray(values["cell_upper_unit"])
        ).to(torch.float64)
        reset_unit = torch.from_numpy(
            np.ascontiguousarray(values["reset_baseline_unit"])
        ).to(torch.float64)
        if baseline.ndim != 2 or baseline.shape[0] % 2 or baseline.shape[1] % 2:
            _fail("Expected even rank-two physical conductance matrices.")
        if not torch.equal(continuous, baseline + continuous_offset):
            _fail("Continuous mapping violates exact G=B+d.")
        if not torch.equal(ideal, baseline + quantized_offset):
            _fail("Ideal mapping violates exact G=B+n*h.")
        expected_quantized_offset = (
            integer.to(torch.float64) * spacing_physical
        ).to(torch.float32)
        if not torch.equal(quantized_offset, expected_quantized_offset):
            _fail("Ideal offsets are not exact whole multiples of h.")
        expected_baseline = (g_min + span * baseline_unit.to(torch.float64)).to(
            torch.float32
        )
        if not torch.equal(baseline, expected_baseline):
            _fail("Physical baseline is not the saved normalized baseline.")
        lower = (g_min + span * lower_unit).to(torch.float32)
        upper = (g_min + span * upper_unit).to(torch.float32)
        tolerance = max(1e-12, span * 1e-6)
        for name, target in (
            ("baseline", baseline),
            ("continuous", continuous),
            ("ideal", ideal),
        ):
            if (
                not bool(torch.isfinite(target).all())
                or bool(torch.any(target < -tolerance))
                or bool(torch.any(target < lower - tolerance))
                or bool(torch.any(target > upper + tolerance))
            ):
                _fail(f"Expected finite per-cell bounded full {name} G.")

        reset_quad = quad_stack(reset_unit, layout=layout)
        upper_quad = quad_stack(upper_unit, layout=layout)
        group_lower = torch.stack(
            (
                torch.maximum(reset_quad[..., 0], reset_quad[..., 2]),
                torch.maximum(reset_quad[..., 1], reset_quad[..., 3]),
            ),
            dim=-1,
        )
        group_upper = torch.stack(
            (
                torch.minimum(upper_quad[..., 0], upper_quad[..., 2]),
                torch.minimum(upper_quad[..., 1], upper_quad[..., 3]),
            ),
            dim=-1,
        )
        group_baseline = group_lower + alpha * (group_upper - group_lower)
        expanded = torch.stack(
            (
                group_baseline[..., 0],
                group_baseline[..., 1],
                group_baseline[..., 0],
                group_baseline[..., 1],
            ),
            dim=-1,
        )
        saved_baseline_quad = quad_stack(
            baseline_unit.to(torch.float64), layout=layout
        )
        if not torch.equal(saved_baseline_quad, expanded.to(torch.float32).to(torch.float64)):
            _fail("Saved baseline does not equal L+alpha*(U-L) by destination column.")
        down = torch.floor(
            (group_baseline - group_lower) / spacing_unit + 1e-12
        ).to(torch.int64)
        up = torch.floor(
            (group_upper - group_baseline) / spacing_unit + 1e-12
        ).to(torch.int64)
        expected_directional = {
            "group_lower_unit": _torch_hash(group_lower.to(torch.float32).numpy()),
            "group_upper_unit": _torch_hash(group_upper.to(torch.float32).numpy()),
            "group_baseline_unit": _torch_hash(group_baseline.to(torch.float32).numpy()),
            "downward_level_capacity": _torch_hash(down.numpy()),
            "upward_level_capacity": _torch_hash(up.numpy()),
        }
        _equal(
            directional_hashes[layer_index],
            expected_directional,
            label="recomputed directional-capacity hashes",
        )
        if bool(torch.any(integer < 0)):
            _fail("Expected nonnegative one-sided level indices.")
        recomputed = {
            "baseline_contrast": quad_contrast(baseline, layout=layout),
            "continuous_contrast": quad_contrast(continuous, layout=layout),
            "quantized_contrast": quad_contrast(ideal, layout=layout),
            "baseline_loading": quad_loading(baseline, layout=layout),
            "continuous_loading": quad_loading(continuous, layout=layout),
            "quantized_loading": quad_loading(ideal, layout=layout),
            "baseline_row_loading": quad_row_loading(baseline, layout=layout),
            "baseline_column_loading": quad_column_loading(baseline, layout=layout),
            "continuous_row_loading": quad_row_loading(continuous, layout=layout),
            "continuous_column_loading": quad_column_loading(continuous, layout=layout),
            "quantized_row_loading": quad_row_loading(ideal, layout=layout),
            "quantized_column_loading": quad_column_loading(ideal, layout=layout),
        }
        for name, value in recomputed.items():
            if not torch.equal(
                value,
                torch.from_numpy(np.ascontiguousarray(values[name])),
            ):
                _fail(f"Saved {name} differs from full-G recomputation.")
        zero_max = float(recomputed["baseline_contrast"].abs().max().item())
        if zero_max != 0.0:
            _fail("Shared destination baseline is not exact logical zero.")
        violations = int(
            baseline_group_violation_mask(
                baseline,
                layout=layout,
                policy=PHYSICAL_POLICY,
            ).sum().item()
        )
        if violations:
            _fail("Shared destination baseline grouping is violated.")
        report_layer = layer_reports[layer_index]
        if not isinstance(report_layer, dict):
            _fail("Expected a physical mapping layer report object.")
        _equal(report_layer.get("layer"), layer_index, label="mapping report layer")
        _equal(report_layer.get("layout"), layout, label="mapping report layout")
        for name in (
            "baseline_contrast",
            "continuous_contrast",
            "ideal_contrast",
            "baseline_loading",
            "selected_headroom_unit",
        ):
            tensor_name = "quantized_contrast" if name == "ideal_contrast" else name
            _equal(
                report_layer.get(name),
                _summary(values[tensor_name]),
                label=f"mapping report {name}",
            )
        expected_hashes = {
            name: _torch_hash(values[name])
            for name in (
                "baseline",
                "continuous_offset",
                "quantized_offset",
                "continuous_conductance",
                "quantized_conductance",
                "baseline_contrast",
                "continuous_contrast",
                "quantized_contrast",
            )
        }
        _equal(report_layer.get("hashes"), expected_hashes, label="mapping report hashes")
        diagnostics.append(
            {
                "layer": layer_index,
                "layout": layout,
                "destination_column_groups": [
                    {
                        "destination_column": destination_column,
                        "lower_L_unit": _summary(
                            group_lower[..., destination_column].numpy()
                        ),
                        "upper_U_unit": _summary(
                            group_upper[..., destination_column].numpy()
                        ),
                        "baseline_B_unit": _summary(
                            group_baseline[..., destination_column].numpy()
                        ),
                        "downward_level_capacity": _summary(
                            down[..., destination_column].numpy().astype(np.float64)
                        ),
                        "upward_level_capacity": _summary(
                            up[..., destination_column].numpy().astype(np.float64)
                        ),
                        "zero_only_group_count": int(
                            _zero_only_initialization_group_count(
                                up[..., destination_column]
                            )
                        ),
                        "requested_index_histogram": _integer_histogram(
                            quad_stack(integer, layout=layout)[
                                ...,
                                (destination_column, destination_column + 2),
                            ]
                        ),
                        "requested_effective_level_count": len(
                            torch.unique(
                                quad_stack(integer, layout=layout)[
                                    ...,
                                    (destination_column, destination_column + 2),
                                ]
                            )
                        ),
                    }
                    for destination_column in range(2)
                ],
                "baseline_contrast": _summary(values["baseline_contrast"]),
                "baseline_loading": _summary(values["baseline_loading"]),
                "continuous_loading": _summary(values["continuous_loading"]),
                "ideal_loading": _summary(values["quantized_loading"]),
                "continuous_rms_contrast_over_mean_loading": (
                    _rms_contrast_over_mean_loading(
                        recomputed["continuous_contrast"],
                        recomputed["continuous_loading"],
                    )
                ),
                "ideal_rms_contrast_over_mean_loading": (
                    _rms_contrast_over_mean_loading(
                        recomputed["quantized_contrast"],
                        recomputed["quantized_loading"],
                    )
                ),
                "requested_index_histogram": _integer_histogram(integer),
                "requested_effective_level_count": len(torch.unique(integer)),
                "selected_level_capacity": _summary(
                    values["selected_level_capacity"].astype(np.float64)
                ),
                "downward_capacity": _summary(down.numpy().astype(np.float64)),
                "upward_capacity": _summary(up.numpy().astype(np.float64)),
                "zero_selected_capacity_count": int(
                    np.count_nonzero(values["selected_level_capacity"] == 0)
                ),
            }
        )
    return {
        "arrays": arrays,
        "layers": layer_values,
        "diagnostics": diagnostics,
        "conductance_min": g_min,
        "conductance_max": g_max,
        "span": span,
        "delta_x": delta_x,
        "spacing_unit": spacing_unit,
        "spacing_physical": spacing_physical,
        "mapping_report": physical_report,
    }


def _validate_calibration(
    path: Path,
    *,
    alpha: float,
    hardware_instance_id: str,
    protocol: Mapping[str, Any],
    data: Mapping[str, Any],
) -> dict[str, Any]:
    value = _json(path, label="development calibration")
    _exact_keys(
        value,
        {
            "schema",
            "schema_version",
            "baseline_position_fraction",
            "sharing",
            "assignment_seed",
            "hardware_instance_id",
            "candidate_count",
            "candidates",
            "selected_index",
            "selected",
            "analysis_scales",
            "calibration_labels_sha256",
            "calibration_indices_sha256",
            "test_labels_used",
        },
        label="development calibration",
    )
    _equal(value["schema"], CALIBRATION_SCHEMA, label="calibration schema")
    _equal(value["schema_version"], 1, label="calibration version")
    _equal(value["baseline_position_fraction"], alpha, label="calibration alpha")
    _equal(value["sharing"], protocol["calibration"]["sharing"], label="calibration sharing")
    _equal(value["assignment_seed"], DEVELOPMENT_ASSIGNMENT_SEED, label="calibration assignment")
    _equal(value["hardware_instance_id"], hardware_instance_id, label="calibration hardware")
    _equal(value["candidate_count"], 16, label="calibration candidate count")
    _equal(value["test_labels_used"], False, label="calibration test-label exclusion")
    _equal(value["calibration_labels_sha256"], data["calibration_labels_sha256"], label="calibration label provenance")
    _equal(value["calibration_indices_sha256"], data["calibration_indices_sha256"], label="calibration index provenance")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 16:
        _fail("Expected exactly sixteen continuous development candidates.")
    allowed_scales = protocol["calibration"]["scale_fractions"]
    if (
        not isinstance(allowed_scales, list)
        or len(allowed_scales) != 4
        or len(set(allowed_scales)) != 4
        or any(
            _finite_number(scale, label="calibration scale", positive=True) <= 0.0
            for scale in allowed_scales
        )
    ):
        _fail("Expected four unique finite positive calibration scales.")
    observed_pairs: list[tuple[float, float]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            _fail(f"Expected calibration candidate {index} to be an object.")
        _exact_keys(
            candidate,
            {"scale_fractions", "calibration", "continuous_target_hashes"},
            label=f"calibration candidate {index}",
        )
        scales = candidate["scale_fractions"]
        details = candidate["calibration"]
        hashes = candidate["continuous_target_hashes"]
        if (
            not isinstance(scales, list)
            or len(scales) != 2
            or any(scale not in allowed_scales for scale in scales)
            or not isinstance(details, dict)
            or isinstance(details.get("student_correct"), bool)
            or not isinstance(details.get("student_correct"), int)
            or not isinstance(details.get("calibrated_kl"), (int, float))
            or not math.isfinite(float(details["calibrated_kl"]))
            or isinstance(details.get("gain"), bool)
            or not isinstance(details.get("gain"), (int, float))
            or not math.isfinite(float(details["gain"]))
            or float(details["gain"]) <= 0.0
            or details.get("test_labels_used") is not False
            or not isinstance(hashes, list)
            or len(hashes) != 2
        ):
            _fail(f"Invalid calibration candidate {index}.")
        observed_pairs.append(tuple(map(float, scales)))
        for digest in hashes:
            _digest(digest, label="continuous target tensor")
    expected_pairs = {
        (float(left), float(right))
        for left, right in product(allowed_scales, repeat=2)
    }
    if len(set(observed_pairs)) != 16 or set(observed_pairs) != expected_pairs:
        _fail("Calibration candidates do not cover the exact 4x4 scale grid.")
    selected_index = min(
        range(len(candidates)),
        key=lambda index: (
            -int(candidates[index]["calibration"]["student_correct"]),
            float(candidates[index]["calibration"]["calibrated_kl"]),
            tuple(float(item) for item in candidates[index]["scale_fractions"]),
        ),
    )
    _equal(value["selected_index"], selected_index, label="calibration selected index")
    _equal(value["selected"], candidates[selected_index], label="calibration selected candidate")
    analysis_scales = value["analysis_scales"]
    if (
        not isinstance(analysis_scales, list)
        or len(analysis_scales) != 2
        or any(_finite_number(item, label="analysis scale", positive=True) <= 0 for item in analysis_scales)
    ):
        _fail("Expected two finite positive analysis scales.")
    return value


def _validate_predictions(
    path: Path,
    receipt_path: Path,
    *,
    endpoint_seeds: Sequence[int],
    expected_examples: int = EXPECTED_EXAMPLES,
) -> dict[str, Any]:
    arrays = _npz(path, label="baseline/spacing predictions")
    receipt = _json(receipt_path, label="prediction receipt")
    prediction_names = {
        "labels",
        "teacher_prediction",
        "continuous_prediction",
        "ideal_quantized_prediction",
        *(f"pv_persistent_prediction_seed_{seed}" for seed in endpoint_seeds),
        *(f"pv_apparent_prediction_seed_{seed}" for seed in endpoint_seeds),
    }
    _exact_keys(
        arrays,
        {"schema", "schema_version"} | prediction_names,
        label="prediction NPZ",
    )
    _exact_keys(
        receipt,
        {
            "schema",
            "schema_version",
            "artifact",
            "artifact_sha256",
            "examples",
            "labels_sha256",
            "teacher_prediction_sha256",
            "continuous_prediction_sha256",
            "ideal_prediction_sha256",
            "ideal_correct",
            "repeats",
        },
        label="prediction receipt",
    )
    _equal(str(_scalar(arrays, "schema")), PREDICTION_SCHEMA, label="prediction schema")
    _equal(int(_scalar(arrays, "schema_version")), PREDICTION_SCHEMA_VERSION, label="prediction version")
    _equal(receipt["schema"], PREDICTION_SCHEMA, label="prediction receipt schema")
    _equal(receipt["schema_version"], PREDICTION_SCHEMA_VERSION, label="prediction receipt version")
    _equal(receipt["artifact"], path.name, label="prediction artifact name")
    _equal(receipt["artifact_sha256"], sha256_file(path), label="prediction artifact hash")
    for name in prediction_names:
        value = arrays[name]
        if value.dtype != np.int64 or value.shape != (expected_examples,):
            _fail(f"Expected int64[{expected_examples}] prediction field {name!r}.")
        if bool(np.any(value < 0)) or bool(np.any(value > 9)):
            _fail(f"Expected MNIST classes in prediction field {name!r}.")
    labels = arrays["labels"]
    teacher = arrays["teacher_prediction"]
    continuous = arrays["continuous_prediction"]
    ideal = arrays["ideal_quantized_prediction"]
    _equal(receipt["examples"], expected_examples, label="prediction examples")
    _equal(receipt["labels_sha256"], _torch_hash(labels), label="labels hash")
    _equal(receipt["teacher_prediction_sha256"], _torch_hash(teacher), label="teacher prediction hash")
    _equal(receipt["continuous_prediction_sha256"], _torch_hash(continuous), label="continuous prediction hash")
    _equal(receipt["ideal_prediction_sha256"], _torch_hash(ideal), label="ideal prediction hash")
    ideal_correct_mask = ideal == labels
    _equal(receipt["ideal_correct"], int(np.count_nonzero(ideal_correct_mask)), label="ideal correct count")
    repeats = receipt["repeats"]
    if not isinstance(repeats, list) or len(repeats) != P_AND_V_REPEAT_COUNT:
        _fail("Expected five prediction repeat receipts.")
    repeat_reports = []
    for index, endpoint_seed in enumerate(endpoint_seeds):
        persistent = arrays[f"pv_persistent_prediction_seed_{endpoint_seed}"]
        apparent = arrays[f"pv_apparent_prediction_seed_{endpoint_seed}"]
        persistent_correct = persistent == labels
        expected = {
            "repeat_index": index,
            "endpoint_seed": int(endpoint_seed),
            "persistent_correct": int(np.count_nonzero(persistent_correct)),
            "apparent_correct": int(np.count_nonzero(apparent == labels)),
            "ideal_to_persistent_wrong_to_correct": int(
                np.count_nonzero(~ideal_correct_mask & persistent_correct)
            ),
            "ideal_to_persistent_correct_to_wrong": int(
                np.count_nonzero(ideal_correct_mask & ~persistent_correct)
            ),
            "ideal_to_persistent_prediction_flip_count": int(
                np.count_nonzero(ideal != persistent)
            ),
            "persistent_prediction_sha256": _torch_hash(persistent),
            "apparent_prediction_sha256": _torch_hash(apparent),
        }
        _equal(repeats[index], expected, label="recomputed prediction repeat")
        repeat_reports.append(expected)
    return {
        "arrays": arrays,
        "labels_sha256": receipt["labels_sha256"],
        "teacher_sha256": receipt["teacher_prediction_sha256"],
        "continuous_sha256": receipt["continuous_prediction_sha256"],
        "ideal_sha256": receipt["ideal_prediction_sha256"],
        "ideal_correct": receipt["ideal_correct"],
        "repeats": repeat_reports,
    }


def _expected_construction_seeds(population: Any) -> np.ndarray:
    return np.concatenate(
        tuple(
            np.full(math.prod(shape), int(seed), dtype=np.int64)
            for shape, seed in zip(
                population.binding_shapes,
                population.binding_sampling_seeds,
            )
        )
    )


def _validate_endpoint(
    path: Path,
    receipt_path: Path,
    *,
    endpoint_seed: int,
    assignment_seed: int,
    alpha: float,
    spacing: int,
    population: Any,
    joint_arrays: Mapping[str, np.ndarray],
    hardware_instance_id: str,
    mapping: Mapping[str, Any],
    persistent_prediction: np.ndarray,
    apparent_prediction: np.ndarray,
    teacher_prediction: np.ndarray,
    labels: np.ndarray,
    expected_gain: float,
) -> dict[str, Any]:
    arrays = _npz(path, label="raw-active P&V endpoint")
    receipt = _json(receipt_path, label="raw-active P&V endpoint receipt")
    flat_float = {
        "target_unit",
        "raw_lower_unit",
        "raw_upper_unit",
        "initial_persistent_unit",
        "initial_apparent_unit",
        "persistent_endpoint_unit",
        "apparent_endpoint_unit",
        "persistent_applied_public_unit",
        "apparent_applied_public_unit",
        "persistent_public_projection_delta_unit",
        "apparent_public_projection_delta_unit",
        "persistent_residual_to_requested",
        "persistent_residual_to_nearest",
    }
    flat_bool = {
        "persistent_below_public_minimum",
        "persistent_above_public_maximum",
        "apparent_below_public_minimum",
        "apparent_above_public_maximum",
        "exact_target_in_support",
        "verify_window_intersects_support",
        "apparent_accepted",
        "persistent_inside_acceptance_window",
        "nonfinite",
        "budget_exhausted",
        "saturated_lower",
        "saturated_upper",
        "requested_code_correct",
    }
    flat_int = {
        "set_count",
        "reset_count",
        "total_pulses",
        "verify_count",
        "reversals",
        "requested_level_index",
        "nearest_persistent_level_index",
        "continuation_construction_seeds",
        "continuation_draw_indices",
        "continuation_trajectory_seeds",
    }
    layer_fields = {
        f"layer_{layer}_{name}"
        for layer in range(2)
        for name in (
            "target_full_conductance",
            "persistent_full_conductance",
            "apparent_full_conductance",
            "persistent_residual_full_conductance",
            "apparent_residual_full_conductance",
            "target_contrast",
            "persistent_contrast",
            "apparent_contrast",
            "target_loading",
            "persistent_loading",
            "apparent_loading",
            "target_unit",
            "raw_lower_unit",
            "raw_upper_unit",
        )
    }
    continuation_scalars = {
        "continuation_schema_version",
        "continuation_preset",
        "continuation_rng_backend",
        "continuation_maximum_random_draws",
        "continuation_state_coordinate",
    }
    _exact_keys(
        arrays,
        {
            "schema",
            "schema_version",
            "endpoint_seed",
            "continuation_persistent",
            "continuation_apparent",
        }
        | flat_float
        | flat_bool
        | flat_int
        | layer_fields
        | continuation_scalars,
        label="P&V endpoint NPZ",
    )
    _exact_keys(
        receipt,
        {
            "schema",
            "schema_version",
            "artifact",
            "artifact_sha256",
            "endpoint_seed",
            "assignment_seed",
            "baseline_position_fraction",
            "spacing_delta_x_multiplier",
            "population_fingerprint",
            "coordinate",
            "controller",
            "tolerance_unit",
            "maximum_program_pulses",
            "inference_read_noise_enabled",
            "trajectory_seed_derivation",
            "continuation_state_sha256",
            "cells",
            "counts",
            "pulse_totals",
            "pulse_distributions",
            "requested_to_nearest_code_confusion",
            "persistent_network_metrics",
            "apparent_network_metrics",
            "persistent_public_conductance_projection",
            "apparent_public_conductance_projection",
            "full_conductance_residuals_by_layer",
            "persistent_target_residual_unit",
            "apparent_target_residual_unit",
            "persistent_residual_by_requested_level",
            "apparent_residual_by_requested_level",
            "tensor_hashes",
        },
        label="P&V endpoint receipt",
    )
    _equal(str(_scalar(arrays, "schema")), PV_SCHEMA, label="P&V endpoint schema")
    _equal(int(_scalar(arrays, "schema_version")), PV_SCHEMA_VERSION, label="P&V endpoint version")
    _equal(int(_scalar(arrays, "endpoint_seed")), endpoint_seed, label="P&V endpoint seed")
    _equal(receipt["schema"], PV_SCHEMA, label="P&V receipt schema")
    _equal(receipt["schema_version"], PV_SCHEMA_VERSION, label="P&V receipt version")
    _equal(receipt["artifact"], path.name, label="P&V artifact name")
    _equal(receipt["artifact_sha256"], sha256_file(path), label="P&V artifact hash")
    _equal(receipt["endpoint_seed"], endpoint_seed, label="P&V receipt endpoint seed")
    _equal(receipt["assignment_seed"], assignment_seed, label="P&V assignment seed")
    _equal(receipt["baseline_position_fraction"], alpha, label="P&V alpha")
    _equal(receipt["spacing_delta_x_multiplier"], spacing, label="P&V spacing")
    _equal(receipt["population_fingerprint"], hardware_instance_id, label="P&V population binding")
    _equal(receipt["coordinate"], RAW_ACTIVE_COORDINATE, label="P&V coordinate")
    _equal(receipt["controller"], "one_pulse", label="P&V controller")
    tolerance_unit = VERIFY_TOLERANCE_DELTA_X_RATIO * mapping["delta_x"]
    _equal(receipt["tolerance_unit"], tolerance_unit, label="P&V tolerance")
    _equal(receipt["maximum_program_pulses"], MAXIMUM_PROGRAM_PULSES, label="P&V pulse cap")
    _equal(receipt["inference_read_noise_enabled"], False, label="P&V inference noise")
    _equal(receipt["trajectory_seed_derivation"], TRAJECTORY_SEED_DERIVATION, label="trajectory derivation")

    cells = sum(int(np.prod(layer["baseline"].shape)) for layer in mapping["layers"])
    _equal(receipt["cells"], cells, label="P&V cell count")
    for name in flat_float | flat_bool | flat_int | {
        "continuation_persistent",
        "continuation_apparent",
    }:
        value = arrays[name]
        if value.shape != (cells,):
            _fail(f"Expected flat P&V field {name!r} to have {cells} cells.")
    for name in flat_bool:
        if arrays[name].dtype != np.bool_:
            _fail(f"Expected boolean P&V field {name!r}.")
    for name in flat_int:
        if arrays[name].dtype != np.int64:
            _fail(f"Expected int64 P&V field {name!r}.")
    for name in flat_float | {"continuation_persistent", "continuation_apparent"}:
        if arrays[name].dtype not in {np.dtype("float32"), np.dtype("float64")} or not bool(np.isfinite(arrays[name]).all()):
            _fail(f"Expected finite floating P&V field {name!r}.")

    expected_target_parts = []
    expected_lower_parts = []
    expected_upper_parts = []
    expected_baseline_parts = []
    expected_index_parts = []
    expected_code_upper_parts = []
    for layer, values in enumerate(mapping["layers"]):
        shape = values["baseline"].shape
        for suffix in (
            "persistent_full_conductance",
            "apparent_full_conductance",
            "target_unit",
            "raw_lower_unit",
            "raw_upper_unit",
        ):
            if arrays[f"layer_{layer}_{suffix}"].shape != shape:
                _fail("Expected endpoint layer tensors to match mapping shapes.")
        baseline_unit = values["baseline_unit"].astype(np.float32, copy=False)
        integer = values["integer_level_number"].astype(np.int64, copy=False)
        target = (
            baseline_unit.astype(np.float64)
            + integer.astype(np.float64) * mapping["spacing_unit"]
        ).astype(np.float32)
        native_lower = joint_arrays[f"layer_{layer}_min_bound"].astype(
            np.float32, copy=False
        )
        native_upper = joint_arrays[f"layer_{layer}_max_bound"].astype(
            np.float32, copy=False
        )
        lower = ((native_lower + np.float32(1.0)) / np.float32(2.0)).astype(
            np.float32, copy=False
        )
        upper = ((native_upper + np.float32(1.0)) / np.float32(2.0)).astype(
            np.float32, copy=False
        )
        if not np.array_equal(arrays[f"layer_{layer}_target_unit"], target):
            _fail("P&V target does not equal the ideal B+n*h target.")
        if not np.array_equal(arrays[f"layer_{layer}_raw_lower_unit"], lower):
            _fail("P&V raw lower support differs from mapped identity.")
        if not np.array_equal(arrays[f"layer_{layer}_raw_upper_unit"], upper):
            _fail("P&V raw upper support differs from mapped identity.")
        expected_target_parts.append(target.reshape(-1))
        expected_lower_parts.append(lower.reshape(-1))
        expected_upper_parts.append(upper.reshape(-1))
        expected_baseline_parts.append(baseline_unit.reshape(-1))
        expected_index_parts.append(integer.reshape(-1))
        expected_code_upper_parts.append(
            values["cell_upper_unit"].astype(np.float32, copy=False).reshape(-1)
        )
    target = np.concatenate(expected_target_parts)
    lower = np.concatenate(expected_lower_parts)
    upper = np.concatenate(expected_upper_parts)
    baseline = np.concatenate(expected_baseline_parts)
    requested = np.concatenate(expected_index_parts)
    code_upper = np.concatenate(expected_code_upper_parts)
    for name, expected in (
        ("target_unit", target),
        ("raw_lower_unit", lower),
        ("raw_upper_unit", upper),
        ("requested_level_index", requested),
    ):
        if not np.array_equal(arrays[name], expected):
            _fail(f"Flat endpoint field {name!r} differs from layer artifacts.")
    if not np.array_equal(arrays["initial_persistent_unit"], lower):
        _fail("P&V did not start from the sampled persistent lower state.")

    raw_persistent = torch.from_numpy(arrays["persistent_endpoint_unit"])
    raw_apparent = torch.from_numpy(arrays["apparent_endpoint_unit"])
    persistent_projection = project_raw_active_unit_to_full_conductance(
        raw_persistent,
        conductance_min=mapping["conductance_min"],
        conductance_max=mapping["conductance_max"],
    )
    apparent_projection = project_raw_active_unit_to_full_conductance(
        raw_apparent,
        conductance_min=mapping["conductance_min"],
        conductance_max=mapping["conductance_max"],
    )
    for prefix, projection in (
        ("persistent", persistent_projection),
        ("apparent", apparent_projection),
    ):
        expected_projection = {
            f"{prefix}_applied_public_unit": projection.applied_endpoint_unit.numpy(),
            f"{prefix}_below_public_minimum": projection.below_public_minimum.numpy(),
            f"{prefix}_above_public_maximum": projection.above_public_maximum.numpy(),
            f"{prefix}_public_projection_delta_unit": projection.projection_delta_unit.numpy(),
        }
        for name, expected in expected_projection.items():
            if not np.array_equal(arrays[name], expected):
                _fail(f"Saved public-coordinate projection field {name!r} is incorrect.")
        _equal(
            receipt[f"{prefix}_public_conductance_projection"],
            projection.report(),
            label=f"{prefix} public-coordinate projection report",
        )
        full = projection.full_conductance.to(torch.float32).numpy()
        offset = 0
        for layer, values in enumerate(mapping["layers"]):
            count = int(np.prod(values["baseline"].shape))
            expected = full[offset : offset + count].reshape(values["baseline"].shape)
            if not np.array_equal(arrays[f"layer_{layer}_{prefix}_full_conductance"], expected):
                _fail(f"Applied {prefix} full G does not match the declared projection.")
            offset += count

    full_residual_reports = []
    for layer, (layout, values) in enumerate(zip(LAYOUTS, mapping["layers"])):
        target_g = values["quantized_conductance"].astype(np.float32, copy=False)
        persistent_g = arrays[f"layer_{layer}_persistent_full_conductance"]
        apparent_g = arrays[f"layer_{layer}_apparent_full_conductance"]
        if not np.array_equal(
            arrays[f"layer_{layer}_target_full_conductance"], target_g
        ):
            _fail("Saved endpoint target full G differs from ideal mapping.")
        persistent_residual_g = (
            persistent_g.astype(np.float64) - target_g.astype(np.float64)
        )
        apparent_residual_g = (
            apparent_g.astype(np.float64) - target_g.astype(np.float64)
        )
        for name, expected in (
            ("persistent_residual_full_conductance", persistent_residual_g),
            ("apparent_residual_full_conductance", apparent_residual_g),
        ):
            if not np.array_equal(arrays[f"layer_{layer}_{name}"], expected):
                _fail(f"Saved full-conductance residual {name!r} is incorrect.")
        endpoint_conductance_reports: dict[str, Any] = {}
        for endpoint_name, conductance in (
            ("target", target_g),
            ("persistent", persistent_g),
            ("apparent", apparent_g),
        ):
            tensor = torch.from_numpy(np.ascontiguousarray(conductance))
            expected_contrast = quad_contrast(tensor, layout=layout)
            expected_loading = quad_loading(tensor, layout=layout)
            if not np.array_equal(
                arrays[f"layer_{layer}_{endpoint_name}_contrast"],
                expected_contrast.numpy(),
            ):
                _fail(f"Saved {endpoint_name} contrast does not match full G.")
            if not np.array_equal(
                arrays[f"layer_{layer}_{endpoint_name}_loading"],
                expected_loading.numpy(),
            ):
                _fail(f"Saved {endpoint_name} loading does not match full G.")
            if endpoint_name in {"persistent", "apparent"}:
                endpoint_conductance_reports[f"{endpoint_name}_loading"] = (
                    _float_summary(expected_loading.to(torch.float64))
                )
                endpoint_conductance_reports[
                    f"{endpoint_name}_rms_contrast_over_mean_loading"
                ] = _rms_contrast_over_mean_loading(
                    expected_contrast,
                    expected_loading,
                )
        full_residual_reports.append(
            {
                "layer": layer,
                "layout": layout,
                "persistent_target_residual_full_conductance": _numeric_summary(
                    torch.from_numpy(persistent_residual_g)
                ),
                "apparent_target_residual_full_conductance": _numeric_summary(
                    torch.from_numpy(apparent_residual_g)
                ),
                **endpoint_conductance_reports,
            }
        )
    _equal(
        receipt["full_conductance_residuals_by_layer"],
        full_residual_reports,
        label="full-conductance residual reports",
    )

    persistent = arrays["persistent_endpoint_unit"]
    apparent = arrays["apparent_endpoint_unit"]
    exact_support = (target >= lower - 1e-6) & (target <= upper + 1e-6)
    verify_intersection = (target + tolerance_unit >= lower) & (target - tolerance_unit <= upper)
    accepted = np.isfinite(apparent) & (np.abs(apparent - target) <= tolerance_unit)
    inside = np.abs(persistent - target) <= tolerance_unit
    nonfinite = ~np.isfinite(apparent)
    total = arrays["total_pulses"]
    exhausted = ~accepted & ~nonfinite & (total >= MAXIMUM_PROGRAM_PULSES)
    saturated_lower = np.abs((2.0 * persistent - 1.0) - (2.0 * lower - 1.0)) <= 2e-6
    saturated_upper = np.abs((2.0 * persistent - 1.0) - (2.0 * upper - 1.0)) <= 2e-6
    expected_bool = {
        "exact_target_in_support": exact_support,
        "verify_window_intersects_support": verify_intersection,
        "apparent_accepted": accepted,
        "persistent_inside_acceptance_window": inside,
        "nonfinite": nonfinite,
        "budget_exhausted": exhausted,
        "saturated_lower": saturated_lower,
        "saturated_upper": saturated_upper,
    }
    for name, expected in expected_bool.items():
        if not np.array_equal(arrays[name], expected):
            _fail(f"Endpoint classification {name!r} is not independently reproducible.")
    if not np.array_equal(arrays["set_count"] + arrays["reset_count"], total):
        _fail("SET+RESET pulse counts do not equal total pulses.")
    if bool(np.any(total < 0)) or bool(np.any(total > MAXIMUM_PROGRAM_PULSES)):
        _fail("P&V pulse count exceeds its declared budget.")
    if not np.array_equal(arrays["verify_count"], total + 1):
        _fail("One-pulse P&V verify count must equal total pulses plus one.")
    if bool(np.any(arrays["reversals"] < 0)) or bool(np.any(arrays["reversals"] > np.maximum(total - 1, 0))):
        _fail("P&V reversal count is invalid.")

    maximum_index = _maximum_supported_uniform_index(
        torch.from_numpy(np.ascontiguousarray(code_upper)),
        torch.from_numpy(np.ascontiguousarray(baseline)),
        spacing_unit=float(mapping["spacing_unit"]),
    ).numpy()
    nearest = np.floor(
        (persistent.astype(np.float64) - baseline.astype(np.float64))
        / mapping["spacing_unit"]
        + 0.5
    ).astype(np.int64)
    nearest = np.maximum(nearest, 0)
    nearest = np.minimum(nearest, maximum_index)
    requested_code = baseline.astype(np.float64) + mapping["spacing_unit"] * requested
    nearest_code = baseline.astype(np.float64) + mapping["spacing_unit"] * nearest
    classification = {
        "nearest_persistent_level_index": nearest,
        "requested_code_correct": nearest == requested,
        "persistent_residual_to_requested": persistent.astype(np.float64) - requested_code,
        "persistent_residual_to_nearest": persistent.astype(np.float64) - nearest_code,
    }
    for name, expected in classification.items():
        if not np.array_equal(arrays[name], expected):
            _fail(f"Persistent code classification {name!r} is incorrect.")
    confusion = _requested_to_nearest_code_confusion(requested, nearest)
    _equal(
        receipt["requested_to_nearest_code_confusion"],
        confusion,
        label="requested-to-nearest code confusion",
    )

    expected_counts = {
        "exact_target_in_support": int(np.count_nonzero(exact_support)),
        "verify_window_intersects_support": int(np.count_nonzero(verify_intersection)),
        "apparent_accepted": int(np.count_nonzero(accepted)),
        "persistent_inside_acceptance_window": int(np.count_nonzero(inside)),
        "requested_code_correct": int(np.count_nonzero(nearest == requested)),
        "nearest_code_adjacent_to_requested": int(
            np.count_nonzero(np.abs(nearest - requested) == 1)
        ),
        "nearest_code_farther_than_adjacent": int(
            np.count_nonzero(np.abs(nearest - requested) > 1)
        ),
        "budget_exhausted": int(np.count_nonzero(exhausted)),
        "nonfinite": int(np.count_nonzero(nonfinite)),
        "saturated_lower": int(np.count_nonzero(saturated_lower)),
        "saturated_upper": int(np.count_nonzero(saturated_upper)),
    }
    _equal(receipt["counts"], expected_counts, label="P&V endpoint counts")
    _equal(
        receipt["pulse_totals"],
        {
            "set": int(arrays["set_count"].sum()),
            "reset": int(arrays["reset_count"].sum()),
            "total": int(total.sum()),
            "verify": int(arrays["verify_count"].sum()),
            "reversals": int(arrays["reversals"].sum()),
        },
        label="P&V pulse totals",
    )
    _equal(
        receipt["pulse_distributions"],
        {
            name: _float_summary(torch.from_numpy(arrays[name]).to(torch.float64))
            for name in (
                "set_count",
                "reset_count",
                "total_pulses",
                "verify_count",
                "reversals",
            )
        },
        label="P&V pulse distributions",
    )
    persistent_residual = torch.from_numpy(persistent - target)
    apparent_residual = torch.from_numpy(apparent - target)
    _equal(receipt["persistent_target_residual_unit"], _numeric_summary(persistent_residual), label="persistent target residual")
    _equal(receipt["apparent_target_residual_unit"], _numeric_summary(apparent_residual), label="apparent target residual")
    _equal(
        receipt["persistent_residual_by_requested_level"],
        _level_residual_report(torch.from_numpy(requested), persistent_residual),
        label="persistent residual by requested level",
    )
    _equal(
        receipt["apparent_residual_by_requested_level"],
        _level_residual_report(torch.from_numpy(requested), apparent_residual),
        label="apparent residual by requested level",
    )
    _equal(
        receipt["tensor_hashes"],
        {name: _raw_hash(value) for name, value in arrays.items() if value.ndim > 0},
        label="P&V endpoint tensor hashes",
    )

    expected_seeds = np.asarray(
        matched_trajectory_seeds(population, endpoint_seed=endpoint_seed),
        dtype=np.int64,
    )
    if not np.array_equal(arrays["continuation_trajectory_seeds"], expected_seeds):
        _fail("Saved P&V trajectory seeds do not match explicit derivation.")
    if not np.array_equal(arrays["continuation_construction_seeds"], _expected_construction_seeds(population)):
        _fail("Saved construction seeds do not match the frozen assignment.")
    if not np.array_equal(arrays["continuation_draw_indices"], 1 + 2 * total):
        _fail("Buffered RNG draw indices do not match one initial and two per pulse draws.")
    continuation = {
        "schema_version": int(_scalar(arrays, "continuation_schema_version")),
        "preset": str(_scalar(arrays, "continuation_preset")),
        "construction_seeds": torch.from_numpy(arrays["continuation_construction_seeds"]),
        "persistent": torch.from_numpy(arrays["continuation_persistent"]),
        "apparent": torch.from_numpy(arrays["continuation_apparent"]),
        "rng_backend": str(_scalar(arrays, "continuation_rng_backend")),
        "seeds": arrays["continuation_trajectory_seeds"].tolist(),
        "maximum_random_draws": int(_scalar(arrays, "continuation_maximum_random_draws")),
        "draw_indices": torch.from_numpy(arrays["continuation_draw_indices"]),
        "state_coordinate": str(_scalar(arrays, "continuation_state_coordinate")),
    }
    _equal(continuation["schema_version"], 2, label="continuation schema version")
    _equal(continuation["preset"], "reram_array_om", label="continuation preset")
    _equal(continuation["rng_backend"], "per_trajectory_buffered_torch_cpu", label="continuation RNG backend")
    _equal(continuation["maximum_random_draws"], 257, label="continuation draw budget")
    _equal(continuation["state_coordinate"], "native_raw_active_a", label="continuation coordinate")
    if not torch.equal((continuation["persistent"] + 1.0) / 2.0, raw_persistent):
        _fail("Native persistent continuation does not reproduce the raw-x endpoint.")
    if not torch.equal((continuation["apparent"] + 1.0) / 2.0, raw_apparent):
        _fail("Native apparent continuation does not reproduce the raw-x endpoint.")
    _equal(receipt["continuation_state_sha256"], _state_digest(continuation), label="continuation state hash")

    _metric_from_prediction(
        receipt["persistent_network_metrics"],
        persistent_prediction,
        labels,
        teacher_prediction=teacher_prediction,
        expected_gain=expected_gain,
    )
    _metric_from_prediction(
        receipt["apparent_network_metrics"],
        apparent_prediction,
        labels,
        teacher_prediction=teacher_prediction,
        expected_gain=expected_gain,
    )
    return {
        "endpoint_seed": endpoint_seed,
        "cells": int(receipt["cells"]),
        "persistent_correct": int(np.count_nonzero(persistent_prediction == labels)),
        "apparent_correct": int(np.count_nonzero(apparent_prediction == labels)),
        "persistent_prediction_sha256": _torch_hash(persistent_prediction),
        "apparent_prediction_sha256": _torch_hash(apparent_prediction),
        "counts": expected_counts,
        "pulse_totals": dict(receipt["pulse_totals"]),
        "pulse_distributions": dict(receipt["pulse_distributions"]),
        "requested_to_nearest_code_confusion": list(confusion),
        "persistent_projection": dict(receipt["persistent_public_conductance_projection"]),
        "apparent_projection": dict(receipt["apparent_public_conductance_projection"]),
        "persistent_network_metrics": dict(receipt["persistent_network_metrics"]),
        "apparent_network_metrics": dict(receipt["apparent_network_metrics"]),
        "full_conductance_residuals_by_layer": list(
            receipt["full_conductance_residuals_by_layer"]
        ),
        "persistent_target_residual_unit": dict(
            receipt["persistent_target_residual_unit"]
        ),
        "apparent_target_residual_unit": dict(
            receipt["apparent_target_residual_unit"]
        ),
        "persistent_residual_by_requested_level": list(
            receipt["persistent_residual_by_requested_level"]
        ),
        "apparent_residual_by_requested_level": list(
            receipt["apparent_residual_by_requested_level"]
        ),
        "persistent_conductances": tuple(
            arrays[f"layer_{layer}_persistent_full_conductance"] for layer in range(2)
        ),
        "stream_match": {
            "population_fingerprint": receipt["population_fingerprint"],
            "raw_lower_sha256": _torch_hash(arrays["raw_lower_unit"]),
            "raw_upper_sha256": _torch_hash(arrays["raw_upper_unit"]),
            "initial_persistent_sha256": _torch_hash(arrays["initial_persistent_unit"]),
            "initial_apparent_sha256": _torch_hash(arrays["initial_apparent_unit"]),
            "construction_seeds_sha256": _torch_hash(arrays["continuation_construction_seeds"]),
            "trajectory_seeds_sha256": _torch_hash(arrays["continuation_trajectory_seeds"]),
        },
    }


def _validate_ideal_weight_errors(
    path: Path,
    receipt_path: Path,
    *,
    mapping: Mapping[str, Any],
    analysis_scales: Sequence[float],
) -> list[dict[str, Any]]:
    arrays = _npz(path, label="ideal weight errors")
    receipt = _json(receipt_path, label="ideal weight-error receipt")
    tensor_fields = {
        f"ideal_layer_{layer}_{name}"
        for layer in range(2)
        for name in _WEIGHT_FIELDS
    }
    _exact_keys(
        arrays,
        {"schema", "schema_version"} | tensor_fields,
        label="ideal weight-error NPZ",
    )
    _exact_keys(
        receipt,
        {
            "schema",
            "schema_version",
            "artifact",
            "artifact_sha256",
            "analysis_scales",
            "ideal_layers",
        },
        label="ideal weight-error receipt",
    )
    _equal(str(_scalar(arrays, "schema")), IDEAL_WEIGHT_SCHEMA, label="ideal weight schema")
    _equal(int(_scalar(arrays, "schema_version")), 1, label="ideal weight version")
    _equal(receipt["schema"], IDEAL_WEIGHT_SCHEMA, label="ideal weight receipt schema")
    _equal(receipt["schema_version"], 1, label="ideal weight receipt version")
    _equal(receipt["artifact"], path.name, label="ideal weight artifact name")
    _equal(receipt["artifact_sha256"], sha256_file(path), label="ideal weight artifact hash")
    _equal(receipt["analysis_scales"], list(analysis_scales), label="ideal weight analysis scales")
    reports = receipt["ideal_layers"]
    if not isinstance(reports, list) or len(reports) != 2:
        _fail("Expected two ideal weight-error reports.")
    verified = []
    for layer, values in enumerate(mapping["layers"]):
        source = torch.from_numpy(np.ascontiguousarray(values["source_weight"])).to(torch.float64)
        c0 = torch.from_numpy(np.ascontiguousarray(values["baseline_contrast"])).to(torch.float64)
        cc = torch.from_numpy(np.ascontiguousarray(values["continuous_contrast"])).to(torch.float64)
        cq = torch.from_numpy(np.ascontiguousarray(values["quantized_contrast"])).to(torch.float64)
        scale = float(analysis_scales[layer])
        maximum = float(source.abs().max().item())
        normalized = source / maximum
        reconstructed_continuous = maximum * cc / scale
        reconstructed_ideal = maximum * cq / scale
        baseline_error64 = maximum * c0 / scale
        envelope_error64 = maximum * ((cc - c0) / scale - normalized)
        continuous_total_error64 = maximum * (cc / scale - normalized)
        quantization_error64 = maximum * (cq - cc) / scale
        total_error64 = maximum * (cq / scale - normalized)
        expected = {
            "reconstructed_continuous_weight": reconstructed_continuous.to(torch.float32),
            "reconstructed_quantized_weight": reconstructed_ideal.to(torch.float32),
            "baseline_error": baseline_error64.to(torch.float32),
            "envelope_error": envelope_error64.to(torch.float32),
            "continuous_total_error": continuous_total_error64.to(torch.float32),
            "quantization_error": quantization_error64.to(torch.float32),
            "total_error": total_error64.to(torch.float32),
            "quantization_error_levels": ((cq - cc) / mapping["spacing_physical"]).to(torch.float32),
        }
        for name, wanted in expected.items():
            observed = torch.from_numpy(
                np.ascontiguousarray(arrays[f"ideal_layer_{layer}_{name}"])
            )
            if not torch.equal(observed, wanted):
                _fail(f"Ideal weight-error tensor {name!r} is not reproducible.")
        report = reports[layer]
        if not isinstance(report, dict):
            _fail("Expected an ideal weight-error layer report.")
        _validate_ideal_weight_report_shape(report, layer=layer)
        target_norm = float(source.norm().item())
        continuous_norm = float(reconstructed_continuous.norm().item())
        ideal_norm = float(reconstructed_ideal.norm().item())
        continuous_cosine = (
            float((source * reconstructed_continuous).sum().item())
            / (target_norm * continuous_norm)
            if target_norm > 0.0 and continuous_norm > 0.0
            else 0.0
        )
        ideal_cosine = (
            float((source * reconstructed_ideal).sum().item())
            / (target_norm * ideal_norm)
            if target_norm > 0.0 and ideal_norm > 0.0
            else 0.0
        )
        sign_tolerance = max(1e-12, 1e-9 * mapping["spacing_physical"])
        nonzero = source != 0.0
        continuous_erased = nonzero & (cc.abs() <= sign_tolerance)
        continuous_opposite = (
            nonzero
            & ~continuous_erased
            & (torch.sign(cc) != torch.sign(source))
        )
        ideal_erased = nonzero & (cq.abs() <= sign_tolerance)
        ideal_opposite = (
            nonzero & ~ideal_erased & (torch.sign(cq) != torch.sign(source))
        )
        conductance_span = mapping["span"]
        expected_report = {
            "layer": layer,
            "analysis_scale": scale,
            "raw_contrast_error": {
                "baseline": _error_summary(c0),
                "envelope": _error_summary((cc - c0) - scale * normalized),
                "continuous_total": _error_summary(cc - scale * normalized),
                "quantization": _error_summary(cq - cc),
                "ideal_quantized_total": _error_summary(cq - scale * normalized),
            },
            "normalized_conductance_coordinate_error": {
                "baseline": _error_summary(c0 / conductance_span),
                "envelope": _error_summary(
                    ((cc - c0) - scale * normalized) / conductance_span
                ),
                "continuous_total": _error_summary(
                    (cc - scale * normalized) / conductance_span
                ),
                "quantization": _error_summary((cq - cc) / conductance_span),
                "ideal_quantized_total": _error_summary(
                    (cq - scale * normalized) / conductance_span
                ),
            },
            "normalized_relu_weight_error": {
                "baseline": _error_summary(c0 / scale),
                "envelope": _error_summary((cc - c0) / scale - normalized),
                "continuous_total": _error_summary(cc / scale - normalized),
                "quantization": _error_summary((cq - cc) / scale),
                "ideal_quantized_total": _error_summary(cq / scale - normalized),
            },
            "teacher_weight_error": {
                "baseline": _error_summary(baseline_error64),
                "envelope": _error_summary(envelope_error64),
                "continuous_total": _error_summary(continuous_total_error64),
                "quantization": _error_summary(quantization_error64),
                "ideal_quantized_total": _error_summary(total_error64),
            },
            "continuous_endpoint": {
                "relative_l2": (
                    float(continuous_total_error64.norm().item()) / target_norm
                    if target_norm > 0.0
                    else 0.0
                ),
                "cosine_similarity": continuous_cosine,
                "opposite_sign_count": int(continuous_opposite.sum().item()),
                "erased_to_zero_count": int(continuous_erased.sum().item()),
                "sign_error_union_count": int(
                    (continuous_opposite | continuous_erased).sum().item()
                ),
            },
            "ideal_quantized_endpoint": {
                "relative_l2": (
                    float(total_error64.norm().item()) / target_norm
                    if target_norm > 0.0
                    else 0.0
                ),
                "cosine_similarity": ideal_cosine,
                "opposite_sign_count": int(ideal_opposite.sum().item()),
                "erased_to_zero_count": int(ideal_erased.sum().item()),
                "sign_error_union_count": int(
                    (ideal_opposite | ideal_erased).sum().item()
                ),
            },
            "nonzero_source_count": int(nonzero.sum().item()),
            "quantization_error_in_levels": _error_summary(
                (cq - cc) / mapping["spacing_physical"]
            ),
            "component_sum_max_abs_residual": float(
                (
                    total_error64
                    - (baseline_error64 + envelope_error64 + quantization_error64)
                ).abs().max().item()
            ),
        }
        _equal(report, expected_report, label=f"ideal weight report layer {layer}")
        verified.append(dict(report))
    return verified


def _validate_persistent_weight_errors(
    path: Path,
    receipt_path: Path,
    *,
    endpoint_seed: int,
    mapping: Mapping[str, Any],
    analysis_scales: Sequence[float],
    expected_conductances: Sequence[np.ndarray],
) -> list[dict[str, Any]]:
    arrays = _npz(path, label="persistent P&V weight errors")
    receipt = _json(receipt_path, label="persistent P&V weight-error receipt")
    _exact_keys(
        arrays,
        {
            "schema",
            "schema_version",
            "endpoint_seed",
            "layer_0_persistent_full_conductance",
            "layer_1_persistent_full_conductance",
        },
        label="persistent weight-error NPZ",
    )
    _exact_keys(
        receipt,
        {
            "schema",
            "schema_version",
            "artifact",
            "artifact_sha256",
            "endpoint_seed",
            "analysis_scales",
            "layers",
        },
        label="persistent weight-error receipt",
    )
    _equal(str(_scalar(arrays, "schema")), PV_WEIGHT_SCHEMA, label="persistent weight schema")
    _equal(int(_scalar(arrays, "schema_version")), 1, label="persistent weight version")
    _equal(int(_scalar(arrays, "endpoint_seed")), endpoint_seed, label="persistent weight seed")
    _equal(receipt["schema"], PV_WEIGHT_SCHEMA, label="persistent weight receipt schema")
    _equal(receipt["schema_version"], 1, label="persistent weight receipt version")
    _equal(receipt["artifact"], path.name, label="persistent weight artifact name")
    _equal(receipt["artifact_sha256"], sha256_file(path), label="persistent weight artifact hash")
    _equal(receipt["endpoint_seed"], endpoint_seed, label="persistent weight receipt seed")
    _equal(receipt["analysis_scales"], list(analysis_scales), label="persistent weight scales")
    conductances = []
    for layer, expected in enumerate(expected_conductances):
        observed = arrays[f"layer_{layer}_persistent_full_conductance"]
        if not np.array_equal(observed, expected):
            _fail("Persistent weight-error conductance is not bound to its endpoint.")
        conductances.append(torch.from_numpy(np.ascontiguousarray(observed)))
    source = [
        torch.from_numpy(np.ascontiguousarray(values["source_weight"]))
        for values in mapping["layers"]
    ]
    expected_reports = list(
        persistent_weight_error_report(
            source,
            conductances,
            analysis_scales,
        )
    )
    _equal(receipt["layers"], expected_reports, label="persistent weight-error reports")
    return expected_reports


def _validate_run(
    run_dir: Path,
    *,
    study: Mapping[str, Any],
    arm: Mapping[str, Any],
    expected_config_hash: str,
) -> dict[str, Any]:
    manifest = _json(run_dir / "manifest.json", label="run manifest")
    status = _json(run_dir / "status.json", label="run status")
    result = _json(run_dir / "result.json", label="run result")
    config = _json(run_dir / "config.resolved.json", label="resolved config")
    run_id = run_dir.name
    for value, label in ((manifest, "manifest"), (status, "status"), (result, "result")):
        _equal(value.get("schema"), "ebl.run", label=f"{label} schema")
        _equal(value.get("schema_version"), 1, label=f"{label} schema version")
        _equal(value.get("run_id"), run_id, label=f"{label} run ID")
    _equal(status.get("status"), "complete", label="run status")
    _equal(result.get("status"), "complete", label="result status")
    _equal(result.get("error"), None, label="result error")
    _equal(manifest.get("experiment_id"), EXPERIMENT_ID, label="manifest experiment")
    _equal(result.get("experiment_id"), EXPERIMENT_ID, label="result experiment")
    source_state = manifest.get("source")
    if not isinstance(source_state, dict):
        _fail("Expected source-controlled run provenance.")
    _exact_keys(source_state, {"available", "commit", "dirty", "dirty_hash"}, label="source state")
    _equal(source_state["available"], True, label="source availability")
    _git_commit(source_state["commit"], label="source commit")
    _equal(source_state["dirty"], False, label="clean source state")
    _equal(source_state["dirty_hash"], None, label="source dirty hash")
    _equal(manifest.get("config", {}).get("sha256"), content_hash(config), label="resolved config hash")
    link = manifest.get("study")
    if not isinstance(link, dict):
        _fail("Expected workflow-managed study linkage.")
    _equal(link.get("study_id"), STUDY_ID, label="manifest study ID")
    _equal(link.get("arm_id"), arm["arm_id"], label="manifest arm ID")
    _equal(link.get("source_config_sha256"), expected_config_hash, label="source config hash")
    _equal(link.get("study_sha256"), sha256_file(Path(study["_root"]) / "study.json"), label="prepared study hash")
    _equal(link.get("source_plan_sha256"), study["source_plan"]["sha256"], label="study-plan hash")

    _equal(config.get("experiment_id"), EXPERIMENT_ID, label="resolved experiment")
    protocol = config.get("protocol")
    student = config.get("student")
    if not isinstance(protocol, dict) or not isinstance(student, dict):
        _fail("Expected resolved protocol and student configuration.")
    _equal(protocol.get("execution", {}).get("profile"), "production", label="production profile")
    _equal(student.get("runtime", {}).get("device"), "cuda", label="production CUDA")
    _equal(student.get("settings", {}).get("sample_limit"), None, label="full production sample limit")
    _equal(protocol.get("source", {}).get("expected_weights_sha256"), EXPECTED_WEIGHTS_SHA256, label="source contract")
    alpha = float(protocol.get("baseline_position_fraction"))
    spacing = int(protocol.get("spacing_delta_x_multiplier"))
    _equal((alpha, spacing), ARM_DESIGNS[arm["arm_id"]], label="arm design")
    assignments = protocol.get("assignments")
    if not isinstance(assignments, dict):
        _fail("Expected assignment contract.")
    heldout_seed = int(assignments.get("heldout_seed"))
    if heldout_seed not in HELDOUT_ASSIGNMENT_SEEDS:
        _fail("Unexpected held-out assignment seed.")
    endpoint_seeds = tuple(int(value) for value in assignments.get("endpoint_seeds", ()))
    _equal(endpoint_seeds, ENDPOINT_SEEDS_BY_ASSIGNMENT[heldout_seed], label="endpoint seeds")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        _fail("Expected manifest inputs.")
    weights = [item for item in inputs if isinstance(item, dict) and item.get("role") == "weights"]
    if len(weights) != 1:
        _fail("Expected exactly one frozen weights input.")
    _equal(weights[0].get("sha256"), EXPECTED_WEIGHTS_SHA256, label="weights input hash")

    artifacts = _artifact_index(run_dir, result)
    required = {
        "summary": _registered(run_dir, artifacts, "artifacts/scientific_summary.json", kind="scientific_summary"),
        "calibration": _registered(run_dir, artifacts, "artifacts/development/calibration.json", kind="development_calibration"),
        "dev_joint": _registered(run_dir, artifacts, "artifacts/development/joint_assignment.npz", kind="ibm_om_joint_assignment"),
        "dev_joint_receipt": _registered(run_dir, artifacts, "artifacts/development/joint_assignment.receipt.json", kind="ibm_om_joint_assignment_receipt"),
        "dev_mapping": _registered(run_dir, artifacts, "artifacts/development/selected_physical_mapping.npz", kind="ibm_om_physical_mapping"),
        "dev_mapping_receipt": _registered(run_dir, artifacts, "artifacts/development/selected_physical_mapping.receipt.json", kind="ibm_om_physical_mapping_receipt"),
        "dev_mapping_context": _registered(run_dir, artifacts, "artifacts/development/selected_physical_mapping.design.json", kind="mapping_context"),
        "held_joint": _registered(run_dir, artifacts, "artifacts/heldout/joint_assignment.npz", kind="ibm_om_joint_assignment"),
        "held_joint_receipt": _registered(run_dir, artifacts, "artifacts/heldout/joint_assignment.receipt.json", kind="ibm_om_joint_assignment_receipt"),
        "held_mapping": _registered(run_dir, artifacts, "artifacts/heldout/ideal_physical_mapping.npz", kind="ibm_om_baseline_spacing_physical_mapping"),
        "held_mapping_receipt": _registered(run_dir, artifacts, "artifacts/heldout/ideal_physical_mapping.receipt.json", kind="ibm_om_baseline_spacing_physical_mapping_receipt"),
        "held_mapping_context": _registered(run_dir, artifacts, "artifacts/heldout/ideal_physical_mapping.design.json", kind="mapping_context"),
        "predictions": _registered(run_dir, artifacts, "artifacts/heldout/predictions.npz", kind="predictions"),
        "predictions_receipt": _registered(run_dir, artifacts, "artifacts/heldout/predictions.receipt.json", kind="predictions_receipt"),
        "ideal_errors": _registered(run_dir, artifacts, "artifacts/heldout/ideal_weight_errors.npz", kind="ideal_weight_errors"),
        "ideal_errors_receipt": _registered(run_dir, artifacts, "artifacts/heldout/ideal_weight_errors.receipt.json", kind="ideal_weight_errors_receipt"),
    }
    summary = _json(required["summary"], label="scientific summary")
    _equal(summary.get("schema"), SUMMARY_SCHEMA, label="scientific-summary schema")
    _equal(summary.get("schema_version"), SUMMARY_SCHEMA_VERSION, label="scientific-summary version")
    _equal(summary.get("status"), "complete", label="scientific-summary status")
    _equal(summary.get("contract"), protocol, label="summary contract")
    _equal(summary.get("source", {}).get("sha256"), EXPECTED_WEIGHTS_SHA256, label="summary source")
    _equal(summary.get("source", {}).get("optimizer_updates"), 0, label="summary optimizer updates")
    source_logical_weight_hashes = summary.get("source", {}).get(
        "logical_weight_hashes"
    )
    if not isinstance(source_logical_weight_hashes, list) or len(
        source_logical_weight_hashes
    ) != 2:
        _fail("Expected two frozen source logical-weight hashes.")
    for layer, digest in enumerate(source_logical_weight_hashes):
        _digest(digest, label=f"source logical weight layer {layer}")
    data = _validate_data_provenance(summary.get("data"))
    design = summary.get("design", {})
    _equal(design.get("baseline_position_fraction"), alpha, label="summary alpha")
    _equal(design.get("spacing_delta_x_multiplier"), spacing, label="summary spacing")
    development = summary.get("development")
    heldout = summary.get("heldout")
    if not isinstance(development, dict) or not isinstance(heldout, dict):
        _fail("Expected development and held-out summary sections.")
    _equal(heldout.get("assignment_seed"), heldout_seed, label="summary held-out seed")

    dev_joint = _validate_joint(required["dev_joint"], required["dev_joint_receipt"], assignment_seed=DEVELOPMENT_ASSIGNMENT_SEED, assignment_role="development")
    held_joint = _validate_joint(required["held_joint"], required["held_joint_receipt"], assignment_seed=heldout_seed, assignment_role="heldout")
    dev_hardware = development.get("hardware")
    held_hardware = heldout.get("hardware")
    if not isinstance(dev_hardware, dict) or not isinstance(held_hardware, dict):
        _fail("Expected hardware summaries.")
    _equal(dev_hardware.get("hardware_instance_id"), dev_joint["hardware_instance_id"], label="development hardware ID")
    _equal(held_hardware.get("hardware_instance_id"), held_joint["hardware_instance_id"], label="held-out hardware ID")
    _validate_hardware_sources(run_dir, artifacts, role="development", assignment_seed=DEVELOPMENT_ASSIGNMENT_SEED, protocol=protocol, hardware_summary=dev_hardware, joint_report=dev_joint["report"])
    _validate_hardware_sources(run_dir, artifacts, role="heldout", assignment_seed=heldout_seed, protocol=protocol, hardware_summary=held_hardware, joint_report=held_joint["report"])
    nominal_dev = _finite_number(dev_hardware.get("nominal_dw_min"), label="development nominal dw_min", positive=True)
    nominal_held = _finite_number(held_hardware.get("nominal_dw_min"), label="held-out nominal dw_min", positive=True)
    dev_population_path = _registered(
        run_dir,
        artifacts,
        f"artifacts/development/base/populations/4-device-assignment-{DEVELOPMENT_ASSIGNMENT_SEED}.npz",
        kind="ibm_om_base_population",
    )
    held_population_path = _registered(
        run_dir,
        artifacts,
        f"artifacts/heldout/base/populations/4-device-assignment-{heldout_seed}.npz",
        kind="ibm_om_base_population",
    )
    dev_population = load_om_array_population(dev_population_path)
    held_population = load_om_array_population(held_population_path)
    _equal(
        nominal_dev,
        float(dev_population.nominal_dw_min),
        label="development nominal step/population binding",
    )
    _equal(
        nominal_held,
        float(held_population.nominal_dw_min),
        label="held-out nominal step/population binding",
    )
    _equal(nominal_dev, nominal_held, label="matched assignment nominal step")
    dev_joint_arrays = _npz(required["dev_joint"], label="development joint assignment")
    held_joint_arrays = _npz(required["held_joint"], label="held-out joint assignment")
    calibration = _validate_calibration(required["calibration"], alpha=alpha, hardware_instance_id=dev_joint["hardware_instance_id"], protocol=protocol, data=data)
    _equal(development.get("calibration"), calibration, label="summary calibration")
    selected_scales = calibration["selected"]["scale_fractions"]
    model_config = student.get("model")
    if not isinstance(model_config, dict):
        _fail("Expected resolved student model configuration.")
    expected_g_min = _finite_number(
        model_config.get("conductance_min"), label="configured conductance minimum"
    )
    expected_g_max = _finite_number(
        model_config.get("conductance_max"),
        label="configured conductance maximum",
        positive=True,
    )
    if expected_g_max <= expected_g_min:
        _fail("Expected an increasing configured conductance interval.")
    dev_mapping = _validate_mapping(
        required["dev_mapping"],
        required["dev_mapping_receipt"],
        required["dev_mapping_context"],
        alpha=alpha,
        spacing=spacing,
        assignment_seed=DEVELOPMENT_ASSIGNMENT_SEED,
        assignment_role="development",
        hardware_instance_id=dev_joint["hardware_instance_id"],
        nominal_dw_min=nominal_dev,
        joint_arrays=dev_joint_arrays,
        source_logical_weight_hashes=source_logical_weight_hashes,
        scale_fractions=selected_scales,
        expected_conductance_min=expected_g_min,
        expected_conductance_max=expected_g_max,
    )
    held_mapping = _validate_mapping(
        required["held_mapping"],
        required["held_mapping_receipt"],
        required["held_mapping_context"],
        alpha=alpha,
        spacing=spacing,
        assignment_seed=heldout_seed,
        assignment_role="heldout",
        hardware_instance_id=held_joint["hardware_instance_id"],
        nominal_dw_min=nominal_held,
        joint_arrays=held_joint_arrays,
        source_logical_weight_hashes=source_logical_weight_hashes,
        scale_fractions=selected_scales,
        expected_conductance_min=expected_g_min,
        expected_conductance_max=expected_g_max,
    )
    _equal(development.get("mapping"), dev_mapping["mapping_report"], label="summary development mapping")
    _equal(heldout.get("mapping"), held_mapping["mapping_report"], label="summary held-out mapping")
    if development.get("invariants", {}).get("valid") is not True or heldout.get("invariants", {}).get("valid") is not True:
        _fail("Runtime full-G invariant gate did not pass.")

    predictions = _validate_predictions(required["predictions"], required["predictions_receipt"], endpoint_seeds=endpoint_seeds)
    _equal(
        predictions["labels_sha256"],
        data["test_targets_sha256"],
        label="prediction labels/test-target provenance",
    )
    prediction_arrays = predictions["arrays"]
    labels = prediction_arrays["labels"]
    continuous_metric = heldout.get("continuous")
    ideal_metric = heldout.get("ideal_quantized")
    if not isinstance(continuous_metric, dict) or not isinstance(ideal_metric, dict):
        _fail("Expected continuous and ideal metrics.")
    selected_gain = calibration["selected"]["calibration"]["gain"]
    _metric_from_prediction(
        continuous_metric,
        prediction_arrays["continuous_prediction"],
        labels,
        teacher_prediction=prediction_arrays["teacher_prediction"],
        expected_gain=selected_gain,
    )
    _metric_from_prediction(
        ideal_metric,
        prediction_arrays["ideal_quantized_prediction"],
        labels,
        teacher_prediction=prediction_arrays["teacher_prediction"],
        expected_gain=selected_gain,
    )
    ideal_weight_reports = _validate_ideal_weight_errors(
        required["ideal_errors"],
        required["ideal_errors_receipt"],
        mapping=held_mapping,
        analysis_scales=calibration["analysis_scales"],
    )

    population = held_population
    joint_arrays = held_joint_arrays
    endpoints = []
    endpoint_receipts = []
    persistent_weight_reports = []
    for endpoint_seed in endpoint_seeds:
        endpoint_path = _registered(run_dir, artifacts, f"artifacts/heldout/program_verify_endpoint_seed_{endpoint_seed}.npz", kind="ibm_om_baseline_spacing_pv_endpoint")
        endpoint_receipt = _registered(run_dir, artifacts, f"artifacts/heldout/program_verify_endpoint_seed_{endpoint_seed}.receipt.json", kind="ibm_om_baseline_spacing_pv_endpoint_receipt")
        endpoint = _validate_endpoint(
            endpoint_path,
            endpoint_receipt,
            endpoint_seed=endpoint_seed,
            assignment_seed=heldout_seed,
            alpha=alpha,
            spacing=spacing,
            population=population,
            joint_arrays=joint_arrays,
            hardware_instance_id=held_joint["hardware_instance_id"],
            mapping=held_mapping,
            persistent_prediction=prediction_arrays[f"pv_persistent_prediction_seed_{endpoint_seed}"],
            apparent_prediction=prediction_arrays[f"pv_apparent_prediction_seed_{endpoint_seed}"],
            teacher_prediction=prediction_arrays["teacher_prediction"],
            labels=labels,
            expected_gain=selected_gain,
        )
        endpoint_receipts.append(
            _json(endpoint_receipt, label="P&V endpoint receipt")
        )
        error_path = _registered(run_dir, artifacts, f"artifacts/heldout/program_verify_weight_errors_seed_{endpoint_seed}.npz", kind="program_verify_weight_errors")
        error_receipt = _registered(run_dir, artifacts, f"artifacts/heldout/program_verify_weight_errors_seed_{endpoint_seed}.receipt.json", kind="program_verify_weight_errors_receipt")
        persistent_weight_reports.append(
            _validate_persistent_weight_errors(
                error_path,
                error_receipt,
                endpoint_seed=endpoint_seed,
                mapping=held_mapping,
                analysis_scales=calibration["analysis_scales"],
                expected_conductances=endpoint["persistent_conductances"],
            )
        )
        endpoints.append(endpoint)
    _equal(
        [item["persistent_correct"] for item in endpoints],
        [item["student_correct"] for item in heldout.get("pv_persistent_repeats", [])],
        label="summary persistent repeat counts",
    )
    _equal(
        [item["apparent_correct"] for item in endpoints],
        [item["student_correct"] for item in heldout.get("pv_apparent_repeats", [])],
        label="summary apparent repeat counts",
    )
    _equal(
        heldout.get("pv_endpoint_artifacts"),
        endpoint_receipts,
        label="summary P&V endpoint receipts",
    )
    _equal(
        heldout.get("predictions"),
        _json(required["predictions_receipt"], label="prediction receipt"),
        label="summary prediction receipt",
    )
    _equal(
        heldout.get("weight_errors"),
        _json(required["ideal_errors_receipt"], label="ideal weight receipt"),
        label="summary ideal weight receipt",
    )
    expected_pv_mean = sum(item["persistent_correct"] for item in endpoints) / (P_AND_V_REPEAT_COUNT * EXPECTED_EXAMPLES)
    _equal(heldout.get("pv_persistent_mean_accuracy"), expected_pv_mean, label="summary P&V mean")
    _equal(heldout.get("pv_persistent_weight_errors"), [
        _json(
            run_dir / f"artifacts/heldout/program_verify_weight_errors_seed_{seed}.receipt.json",
            label="persistent weight receipt",
        )
        for seed in endpoint_seeds
    ], label="summary persistent weight reports")

    parity = protocol["reference_study"]["alpha0_spacing4_parity"]
    parity_index = parity["heldout_assignment_seeds"].index(heldout_seed)
    parity_design_arm = alpha == 0.0 and spacing == 4
    full_test_output_required = (
        parity_design_arm and protocol["execution"]["profile"] == "production"
    )
    parity_gates = {
        "development_hardware_instance_id": dev_joint["hardware_instance_id"] == parity["development_hardware_instance_id"],
        "heldout_hardware_instance_id": held_joint["hardware_instance_id"] == parity["heldout_hardware_instance_ids"][parity_index],
        "selected_scales": tuple(calibration["selected"]["scale_fractions"]) == tuple(parity["selected_scale_fractions"]),
        "fixed_gain": calibration["selected"]["calibration"]["gain"] == parity["fixed_logit_gain"],
        "ideal_correct": (
            not full_test_output_required
            or predictions["ideal_correct"] == parity["ideal_correct"][parity_index]
        ),
        "ideal_prediction_sha256": (
            not full_test_output_required
            or predictions["ideal_sha256"]
            == parity["ideal_prediction_sha256"][parity_index]
        ),
    }
    if not parity_gates["development_hardware_instance_id"] or not parity_gates["heldout_hardware_instance_id"]:
        _fail("Run does not reproduce the declared historical hardware identity.")
    if parity_design_arm and not all(parity_gates.values()):
        _fail(f"Historical alpha=0,h=4 ideal parity failed: {parity_gates!r}.")
    _validate_summary_parity(
        summary.get("parity"),
        design_arm=parity_design_arm,
        full_test_output_required=full_test_output_required,
        gates=parity_gates,
    )
    validity = summary.get("validity", {})
    _equal(validity.get("coverage_valid"), True, label="summary coverage validity")
    if not isinstance(validity.get("gates"), dict) or not all(validity["gates"].values()):
        _fail("One or more runtime validity gates did not pass.")
    _equal(validity.get("expected_examples"), EXPECTED_EXAMPLES, label="validity examples")
    _equal(validity.get("optimizer_updates"), 0, label="validity optimizer updates")
    _equal(validity.get("program_verify_enabled"), True, label="validity P&V enabled")
    _equal(validity.get("inference_read_noise"), 0.0, label="validity inference noise")

    terminal = result.get("metrics")
    if not isinstance(terminal, dict):
        _fail("Expected terminal metrics.")
    _equal(
        terminal.get("metric_definition_ids"),
        {
            "ideal_quantized": IDEAL_QUANTIZED_METRIC_DEFINITION,
            "pv_persistent": PV_PERSISTENT_METRIC_DEFINITION,
            "continuous_diagnostic": CONTINUOUS_DIAGNOSTIC_METRIC_DEFINITION,
        },
        label="terminal metric definitions",
    )
    _equal(terminal.get("baseline_position_fraction"), alpha, label="terminal alpha")
    _equal(terminal.get("spacing_delta_x_multiplier"), spacing, label="terminal spacing")
    _equal(terminal.get("heldout_assignment_seed"), heldout_seed, label="terminal held-out seed")
    _equal(terminal.get("development_hardware_instance_id"), dev_joint["hardware_instance_id"], label="terminal development hardware")
    _equal(terminal.get("heldout_hardware_instance_id"), held_joint["hardware_instance_id"], label="terminal held-out hardware")
    _equal(terminal.get("selected_scale_fractions"), calibration["selected"]["scale_fractions"], label="terminal selected scales")
    _equal(terminal.get("fixed_logit_gain"), calibration["selected"]["calibration"]["gain"], label="terminal gain")
    _equal(terminal.get("coverage_valid"), True, label="terminal coverage")
    _equal(terminal.get("pv_persistent_mean_accuracy"), expected_pv_mean, label="terminal P&V mean")
    _equal(
        [item["student_correct"] for item in terminal.get("pv_persistent", [])],
        [item["persistent_correct"] for item in endpoints],
        label="terminal P&V correct counts",
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "arm_id": arm["arm_id"],
        "alpha": alpha,
        "spacing_delta_x_multiplier": spacing,
        "heldout_assignment_seed": heldout_seed,
        "endpoint_seeds": list(endpoint_seeds),
        "run_source": source_state,
        "source_logical_weight_hashes": summary["source"]["logical_weight_hashes"],
        "data_provenance": data,
        "development_hardware_instance_id": dev_joint["hardware_instance_id"],
        "heldout_hardware_instance_id": held_joint["hardware_instance_id"],
        "calibration_sha256": sha256_file(required["calibration"]),
        "selected_scale_fractions": calibration["selected"]["scale_fractions"],
        "fixed_logit_gain": calibration["selected"]["calibration"]["gain"],
        "continuous_metrics": dict(continuous_metric),
        "continuous_correct": int(continuous_metric["student_correct"]),
        "continuous_prediction_sha256": continuous_metric["prediction_sha256"],
        "ideal_metrics": dict(ideal_metric),
        "ideal_correct": int(ideal_metric["student_correct"]),
        "ideal_prediction_sha256": ideal_metric["prediction_sha256"],
        "persistent_correct": [item["persistent_correct"] for item in endpoints],
        "persistent_prediction_sha256": [item["persistent_prediction_sha256"] for item in endpoints],
        "apparent_correct": [item["apparent_correct"] for item in endpoints],
        "paired_prediction_diagnostics": list(predictions["repeats"]),
        "endpoint_diagnostics": [
            {key: value for key, value in item.items() if key != "persistent_conductances"}
            for item in endpoints
        ],
        "stream_match": [item["stream_match"] for item in endpoints],
        "mapping_diagnostics": held_mapping["diagnostics"],
        "ideal_weight_errors": ideal_weight_reports,
        "pv_persistent_weight_errors": persistent_weight_reports,
        "registered_artifact_count": len(artifacts),
    }


def _discover(study_dir: Path, study: dict[str, Any]) -> list[dict[str, Any]]:
    study["_root"] = str(study_dir)
    _equal(study.get("study_id"), STUDY_ID, label="study ID")
    if len(study.get("arms", ())) != len(DESIGNS):
        _fail("Expected exactly nine declared alpha-by-spacing arms.")
    current_plan = Path(study["source_plan"]["path"])
    if not current_plan.is_file() or sha256_file(current_plan) != study["source_plan"]["sha256"]:
        _fail("Prepared study source-plan hash no longer matches its tracked file.")
    arm_by_id = {arm["arm_id"]: arm for arm in study["arms"]}
    if set(arm_by_id) != set(ARM_DESIGNS):
        _fail("Prepared study arms do not equal the declared 3x3 design.")
    config_owner: dict[str, Mapping[str, Any]] = {}
    complete: list[tuple[Path, Mapping[str, Any]]] = []
    for arm_id, arm in arm_by_id.items():
        if arm.get("experiment_id") != EXPERIMENT_ID or arm.get("mode") != "validate" or len(arm.get("configs", ())) != 3:
            _fail(f"Unexpected study arm contract for {arm_id!r}.")
        for config in arm["configs"]:
            config_path = Path(config["resolved_path"])
            if not config_path.is_file() or sha256_file(config_path) != config["sha256"]:
                _fail(f"Declared config hash no longer matches {config_path}.")
            if config["sha256"] in config_owner:
                _fail("Expected unique declared production config hashes.")
            config_owner[config["sha256"]] = arm
        arm_dir = study_dir / "runs" / arm_id
        if not arm_dir.is_dir():
            _fail(f"Missing run arm directory {arm_dir}.")
        for candidate in sorted(path for path in arm_dir.iterdir() if path.is_dir()):
            status_path = candidate / "status.json"
            if not status_path.is_file():
                _fail(f"Every native run directory must contain status.json: {candidate}.")
            state = _json(status_path, label="run status").get("status")
            if state == "complete":
                complete.append((candidate, arm))
            elif state == "running":
                _fail(f"Study still contains a running bundle: {candidate}.")
            elif state != "failed":
                _fail(f"Invalid run state {state!r}: {candidate}.")
    if len(complete) != EXPECTED_RUNS:
        _fail(f"Expected exactly {EXPECTED_RUNS} complete bundles; found {len(complete)}.")
    seen: set[str] = set()
    records = []
    for run_dir, directory_arm in complete:
        manifest = _json(run_dir / "manifest.json", label="run manifest")
        config_hash = manifest.get("study", {}).get("source_config_sha256")
        owner = config_owner.get(config_hash)
        if owner is None or owner["arm_id"] != directory_arm["arm_id"]:
            _fail(f"Complete run is not linked to a declared config in its arm: {run_dir}.")
        if config_hash in seen:
            _fail(f"More than one complete run claims config hash {config_hash}.")
        seen.add(config_hash)
        records.append(
            _validate_run(
                run_dir,
                study=study,
                arm=directory_arm,
                expected_config_hash=config_hash,
            )
        )
    if seen != set(config_owner):
        _fail("Complete run coverage does not equal all 27 declared configs.")
    expected_coverage = {
        (alpha, spacing, seed)
        for alpha in BASELINE_POSITION_FRACTIONS
        for spacing in SPACING_DELTA_X_MULTIPLIERS
        for seed in HELDOUT_ASSIGNMENT_SEEDS
    }
    observed_coverage = {
        (record["alpha"], record["spacing_delta_x_multiplier"], record["heldout_assignment_seed"])
        for record in records
    }
    if observed_coverage != expected_coverage:
        _fail("Observed run coverage is not exactly 3x3x3.")
    if len({canonical_json_bytes(record["run_source"]) for record in records}) != 1:
        _fail("Production runs do not share one clean source commit.")
    if len({canonical_json_bytes(record["source_logical_weight_hashes"]) for record in records}) != 1:
        _fail("Frozen logical ReLU source hashes differ across runs.")
    if len({canonical_json_bytes(record["data_provenance"]) for record in records}) != 1:
        _fail("MNIST data/calibration provenance differs across runs.")
    if len({record["development_hardware_instance_id"] for record in records}) != 1:
        _fail("Development hardware identity differs across the design.")
    for seed in HELDOUT_ASSIGNMENT_SEEDS:
        selected = [record for record in records if record["heldout_assignment_seed"] == seed]
        if len({record["heldout_hardware_instance_id"] for record in selected}) != 1:
            _fail(f"Held-out hardware identity differs at assignment {seed}.")
        for repeat_index, endpoint_seed in enumerate(ENDPOINT_SEEDS_BY_ASSIGNMENT[seed]):
            streams = {
                canonical_json_bytes(record["stream_match"][repeat_index])
                for record in selected
            }
            if len(streams) != 1:
                _fail(f"P&V latent stream pairing differs at assignment {seed}, endpoint {endpoint_seed}.")
    for alpha in BASELINE_POSITION_FRACTIONS:
        selected = [record for record in records if record["alpha"] == alpha]
        if len({record["calibration_sha256"] for record in selected}) != 1:
            _fail(f"Development calibration is not byte-identical for alpha={alpha}.")
        if len({canonical_json_bytes(record["selected_scale_fractions"]) for record in selected}) != 1 or len({record["fixed_logit_gain"] for record in selected}) != 1:
            _fail(f"Frozen selected calibration differs for alpha={alpha}.")
        for seed in HELDOUT_ASSIGNMENT_SEEDS:
            continuous_hashes = {
                record["continuous_prediction_sha256"]
                for record in selected
                if record["heldout_assignment_seed"] == seed
            }
            if len(continuous_hashes) != 1:
                _fail(f"Continuous held-out endpoint changes with spacing for alpha={alpha}, seed={seed}.")
    return sorted(
        records,
        key=lambda item: (
            BASELINE_POSITION_FRACTIONS.index(item["alpha"]),
            SPACING_DELTA_X_MULTIPLIERS.index(item["spacing_delta_x_multiplier"]),
            HELDOUT_ASSIGNMENT_SEEDS.index(item["heldout_assignment_seed"]),
        ),
    )


def _merge_requested_to_nearest_code_confusions(
    endpoints: Sequence[Mapping[str, Any]],
) -> list[dict[str, int]]:
    totals: dict[tuple[int, int], int] = {}
    for endpoint in endpoints:
        for row in endpoint["requested_to_nearest_code_confusion"]:
            key = (int(row["requested_index"]), int(row["nearest_index"]))
            totals[key] = totals.get(key, 0) + int(row["cells"])
    return [
        {
            "requested_index": requested,
            "nearest_index": nearest,
            "cells": totals[(requested, nearest)],
        }
        for requested, nearest in sorted(totals)
    ]


def _aggregate(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    designs: dict[str, Any] = {}
    for alpha, spacing, arm_id in DESIGNS:
        rows = [record for record in records if record["arm_id"] == arm_id]
        ideal_correct = [int(record["ideal_correct"]) for record in rows]
        assignment_persistent = [sum(int(value) for value in record["persistent_correct"]) for record in rows]
        repeat_correct = [int(value) for record in rows for value in record["persistent_correct"]]
        endpoint_rows = [
            endpoint
            for record in rows
            for endpoint in record["endpoint_diagnostics"]
        ]
        persistent_projection_count = sum(
            int(item["persistent_projection"]["projected"])
            for item in endpoint_rows
        )
        designs[arm_id] = {
            "baseline_position_fraction": alpha,
            "spacing_delta_x_multiplier": spacing,
            "heldout_assignment_seeds": list(HELDOUT_ASSIGNMENT_SEEDS),
            "ideal": {
                "correct_by_assignment": ideal_correct,
                "examples_each": EXPECTED_EXAMPLES,
                "correct_total": sum(ideal_correct),
                "examples_total": 3 * EXPECTED_EXAMPLES,
                "accuracy": sum(ideal_correct) / (3 * EXPECTED_EXAMPLES),
            },
            "pv_persistent": {
                "correct_by_repeat": repeat_correct,
                "correct_sum_by_assignment": assignment_persistent,
                "examples_per_assignment": P_AND_V_REPEAT_COUNT * EXPECTED_EXAMPLES,
                "correct_total": sum(repeat_correct),
                "examples_total": 3 * P_AND_V_REPEAT_COUNT * EXPECTED_EXAMPLES,
                "accuracy": sum(repeat_correct) / (3 * P_AND_V_REPEAT_COUNT * EXPECTED_EXAMPLES),
                "repeat_accuracy_minimum": min(repeat_correct) / EXPECTED_EXAMPLES,
                "repeat_accuracy_maximum": max(repeat_correct) / EXPECTED_EXAMPLES,
            },
            "ideal_to_pv_accuracy_drop": (
                sum(ideal_correct) / (3 * EXPECTED_EXAMPLES)
                - sum(repeat_correct) / (3 * P_AND_V_REPEAT_COUNT * EXPECTED_EXAMPLES)
            ),
            "persistent_public_projection_count": persistent_projection_count,
            "network_diagnostics": {
                "continuous": {
                    "teacher_agreement_mean": sum(
                        float(record["continuous_metrics"]["teacher_agreement"])
                        for record in rows
                    )
                    / len(rows),
                    "kl_teacher_student_mean": sum(
                        float(record["continuous_metrics"]["kl_teacher_student"])
                        for record in rows
                    )
                    / len(rows),
                    "voltage_rms_mean_by_state_layer": [
                        sum(
                            float(record["continuous_metrics"]["voltage"][layer]["rms"])
                            for record in rows
                        )
                        / len(rows)
                        for layer in range(3)
                    ],
                },
                "ideal_quantized": {
                    "teacher_agreement_mean": sum(
                        float(record["ideal_metrics"]["teacher_agreement"])
                        for record in rows
                    )
                    / len(rows),
                    "kl_teacher_student_mean": sum(
                        float(record["ideal_metrics"]["kl_teacher_student"])
                        for record in rows
                    )
                    / len(rows),
                    "voltage_rms_mean_by_state_layer": [
                        sum(
                            float(record["ideal_metrics"]["voltage"][layer]["rms"])
                            for record in rows
                        )
                        / len(rows)
                        for layer in range(3)
                    ],
                },
                "pv_persistent": {
                    "teacher_agreement_mean": sum(
                        float(item["persistent_network_metrics"]["teacher_agreement"])
                        for item in endpoint_rows
                    )
                    / len(endpoint_rows),
                    "kl_teacher_student_mean": sum(
                        float(item["persistent_network_metrics"]["kl_teacher_student"])
                        for item in endpoint_rows
                    )
                    / len(endpoint_rows),
                    "voltage_rms_mean_by_state_layer": [
                        sum(
                            float(item["persistent_network_metrics"]["voltage"][layer]["rms"])
                            for item in endpoint_rows
                        )
                        / len(endpoint_rows)
                        for layer in range(3)
                    ],
                },
                "pv_apparent": {
                    "teacher_agreement_mean": sum(
                        float(item["apparent_network_metrics"]["teacher_agreement"])
                        for item in endpoint_rows
                    )
                    / len(endpoint_rows),
                    "kl_teacher_student_mean": sum(
                        float(item["apparent_network_metrics"]["kl_teacher_student"])
                        for item in endpoint_rows
                    )
                    / len(endpoint_rows),
                    "voltage_rms_mean_by_state_layer": [
                        sum(
                            float(item["apparent_network_metrics"]["voltage"][layer]["rms"])
                            for item in endpoint_rows
                        )
                        / len(endpoint_rows)
                        for layer in range(3)
                    ],
                },
            },
            "mapping_diagnostics_by_assignment": [
                {
                    "heldout_assignment_seed": record["heldout_assignment_seed"],
                    "layers": record["mapping_diagnostics"],
                }
                for record in rows
            ],
            "program_verify_diagnostics": {
                "cells_total": sum(int(item["cells"]) for item in endpoint_rows),
                "counts_total": {
                    key: sum(int(item["counts"][key]) for item in endpoint_rows)
                    for key in endpoint_rows[0]["counts"]
                },
                "pulse_totals": {
                    key: sum(int(item["pulse_totals"][key]) for item in endpoint_rows)
                    for key in endpoint_rows[0]["pulse_totals"]
                },
                "public_projection_counts": {
                    "persistent": sum(
                        int(item["persistent_projection"]["projected"])
                        for item in endpoint_rows
                    ),
                    "apparent": sum(
                        int(item["apparent_projection"]["projected"])
                        for item in endpoint_rows
                    ),
                },
                "persistent_full_conductance_residual_rmse_mean_by_layer": [
                    sum(
                        float(
                            item["full_conductance_residuals_by_layer"][layer][
                                "persistent_target_residual_full_conductance"
                            ]["rmse"]
                        )
                        for item in endpoint_rows
                    )
                    / len(endpoint_rows)
                    for layer in range(2)
                ],
                "apparent_full_conductance_residual_rmse_mean_by_layer": [
                    sum(
                        float(
                            item["full_conductance_residuals_by_layer"][layer][
                                "apparent_target_residual_full_conductance"
                            ]["rmse"]
                        )
                        for item in endpoint_rows
                    )
                    / len(endpoint_rows)
                    for layer in range(2)
                ],
                "persistent_loading_mean_by_layer": [
                    sum(
                        float(
                            item["full_conductance_residuals_by_layer"][layer][
                                "persistent_loading"
                            ]["mean"]
                        )
                        for item in endpoint_rows
                    )
                    / len(endpoint_rows)
                    for layer in range(2)
                ],
                "apparent_loading_mean_by_layer": [
                    sum(
                        float(
                            item["full_conductance_residuals_by_layer"][layer][
                                "apparent_loading"
                            ]["mean"]
                        )
                        for item in endpoint_rows
                    )
                    / len(endpoint_rows)
                    for layer in range(2)
                ],
                "persistent_rms_contrast_over_mean_loading_by_layer": [
                    sum(
                        float(
                            item["full_conductance_residuals_by_layer"][layer][
                                "persistent_rms_contrast_over_mean_loading"
                            ]
                        )
                        for item in endpoint_rows
                    )
                    / len(endpoint_rows)
                    for layer in range(2)
                ],
                "apparent_rms_contrast_over_mean_loading_by_layer": [
                    sum(
                        float(
                            item["full_conductance_residuals_by_layer"][layer][
                                "apparent_rms_contrast_over_mean_loading"
                            ]
                        )
                        for item in endpoint_rows
                    )
                    / len(endpoint_rows)
                    for layer in range(2)
                ],
                "requested_to_nearest_code_confusion_total": (
                    _merge_requested_to_nearest_code_confusions(endpoint_rows)
                ),
                "mean_of_repeat_pulse_distributions": {
                    quantity: {
                        statistic: sum(
                            float(item["pulse_distributions"][quantity][statistic])
                            for item in endpoint_rows
                        )
                        / len(endpoint_rows)
                        for statistic in endpoint_rows[0]["pulse_distributions"][quantity]
                    }
                    for quantity in endpoint_rows[0]["pulse_distributions"]
                },
                "by_assignment": [
                    {
                        "heldout_assignment_seed": record["heldout_assignment_seed"],
                        "paired_prediction_diagnostics": record[
                            "paired_prediction_diagnostics"
                        ],
                        "repeats": [
                            {
                                key: endpoint[key]
                                for key in (
                                    "endpoint_seed",
                                    "cells",
                                    "counts",
                                    "pulse_totals",
                                    "pulse_distributions",
                                    "requested_to_nearest_code_confusion",
                                    "persistent_projection",
                                    "apparent_projection",
                                    "full_conductance_residuals_by_layer",
                                    "persistent_target_residual_unit",
                                    "apparent_target_residual_unit",
                                    "persistent_residual_by_requested_level",
                                    "apparent_residual_by_requested_level",
                                )
                            }
                            for endpoint in record["endpoint_diagnostics"]
                        ],
                    }
                    for record in rows
                ],
            },
            "weight_error": {
                "ideal_relative_l2_mean_by_layer": [
                    sum(
                        float(record["ideal_weight_errors"][layer]["ideal_quantized_endpoint"]["relative_l2"])
                        for record in rows
                    )
                    / len(rows)
                    for layer in range(2)
                ],
                "ideal_rmse_mean_by_layer": [
                    sum(
                        float(record["ideal_weight_errors"][layer]["teacher_weight_error"]["ideal_quantized_total"]["rmse"])
                        for record in rows
                    )
                    / len(rows)
                    for layer in range(2)
                ],
                "pv_persistent_relative_l2_mean_by_layer": [
                    sum(
                        float(repeat[layer]["relative_l2"])
                        for record in rows
                        for repeat in record["pv_persistent_weight_errors"]
                    )
                    / (len(rows) * P_AND_V_REPEAT_COUNT)
                    for layer in range(2)
                ],
                "pv_persistent_rmse_mean_by_layer": [
                    sum(
                        float(repeat[layer]["rmse"])
                        for record in rows
                        for repeat in record["pv_persistent_weight_errors"]
                    )
                    / (len(rows) * P_AND_V_REPEAT_COUNT)
                    for layer in range(2)
                ],
            },
        }
    ideal_order = sorted(
        designs,
        key=lambda arm: (-designs[arm]["ideal"]["accuracy"], arm),
    )
    eligible = [arm for arm in designs if designs[arm]["ideal"]["accuracy"] >= 0.90]
    persistent_order = sorted(
        eligible,
        key=lambda arm: (-designs[arm]["pv_persistent"]["accuracy"], arm),
    )
    dominant = []
    for candidate in eligible:
        values = designs[candidate]["pv_persistent"]["correct_sum_by_assignment"]
        if all(
            all(left > right for left, right in zip(values, designs[other]["pv_persistent"]["correct_sum_by_assignment"]))
            for other in eligible
            if other != candidate
        ):
            dominant.append(candidate)
    best_persistent = persistent_order[0] if persistent_order else None
    runner = persistent_order[1] if len(persistent_order) > 1 else None
    correct_count_lead = (
        designs[best_persistent]["pv_persistent"]["correct_total"]
        - designs[runner]["pv_persistent"]["correct_total"]
        if best_persistent is not None and runner is not None
        else None
    )
    persistent_comparison_examples = (
        3 * P_AND_V_REPEAT_COUNT * EXPECTED_EXAMPLES
    )
    minimum_correct_count_lead = 1500
    mean_lead = (
        correct_count_lead / persistent_comparison_examples
        if correct_count_lead is not None
        else None
    )
    winner = (
        dominant[0]
        if len(dominant) == 1
        and dominant[0] == best_persistent
        and correct_count_lead is not None
        and correct_count_lead >= minimum_correct_count_lead
        else None
    )
    decision = {
        "ideal_gate_threshold": 0.90,
        "eligible_designs": persistent_order,
        "best_ideal_design": ideal_order[0],
        "best_ideal_accuracy": designs[ideal_order[0]]["ideal"]["accuracy"],
        "best_observed_persistent_design": best_persistent,
        "runner_up_persistent_design": runner,
        "assignment_level_strict_dominance_candidates": dominant,
        "persistent_correct_count_lead_over_runner_up": correct_count_lead,
        "persistent_comparison_examples": persistent_comparison_examples,
        "persistent_mean_lead_over_runner_up": mean_lead,
        "minimum_correct_count_lead": minimum_correct_count_lead,
        "minimum_mean_lead": 0.01,
        "predeclared_unique_practical_winner": winner,
        "status": "winner" if winner is not None else "inconclusive_no_unique_winner",
    }
    matrices = {
        "alpha_order": list(BASELINE_POSITION_FRACTIONS),
        "spacing_order": list(SPACING_DELTA_X_MULTIPLIERS),
        "ideal_accuracy": [
            [designs[_arm_id(alpha, spacing)]["ideal"]["accuracy"] for spacing in SPACING_DELTA_X_MULTIPLIERS]
            for alpha in BASELINE_POSITION_FRACTIONS
        ],
        "pv_persistent_accuracy": [
            [designs[_arm_id(alpha, spacing)]["pv_persistent"]["accuracy"] for spacing in SPACING_DELTA_X_MULTIPLIERS]
            for alpha in BASELINE_POSITION_FRACTIONS
        ],
        "ideal_to_pv_drop": [
            [designs[_arm_id(alpha, spacing)]["ideal_to_pv_accuracy_drop"] for spacing in SPACING_DELTA_X_MULTIPLIERS]
            for alpha in BASELINE_POSITION_FRACTIONS
        ],
    }
    return {"designs": designs, "matrices": matrices}, decision


def _csv(records: Sequence[Mapping[str, Any]]) -> str:
    columns = [
        "arm_id", "alpha", "spacing_delta_x_multiplier", "heldout_assignment_seed", "run_id",
        "ideal_correct", "ideal_accuracy", "pv_persistent_correct_sum", "pv_persistent_accuracy",
        "ideal_relative_l2_layer0", "ideal_relative_l2_layer1",
        "ideal_rmse_layer0", "ideal_rmse_layer1",
        "pv_persistent_relative_l2_mean_layer0", "pv_persistent_relative_l2_mean_layer1",
        "pv_persistent_rmse_mean_layer0", "pv_persistent_rmse_mean_layer1",
        "selected_scale_layer0", "selected_scale_layer1", "fixed_logit_gain",
        "development_hardware_instance_id", "heldout_hardware_instance_id",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for record in records:
        persistent_sum = sum(record["persistent_correct"])
        writer.writerow(
            {
                "arm_id": record["arm_id"],
                "alpha": record["alpha"],
                "spacing_delta_x_multiplier": record["spacing_delta_x_multiplier"],
                "heldout_assignment_seed": record["heldout_assignment_seed"],
                "run_id": record["run_id"],
                "ideal_correct": record["ideal_correct"],
                "ideal_accuracy": record["ideal_correct"] / EXPECTED_EXAMPLES,
                "pv_persistent_correct_sum": persistent_sum,
                "pv_persistent_accuracy": persistent_sum / (P_AND_V_REPEAT_COUNT * EXPECTED_EXAMPLES),
                "ideal_relative_l2_layer0": record["ideal_weight_errors"][0]["ideal_quantized_endpoint"]["relative_l2"],
                "ideal_relative_l2_layer1": record["ideal_weight_errors"][1]["ideal_quantized_endpoint"]["relative_l2"],
                "ideal_rmse_layer0": record["ideal_weight_errors"][0]["teacher_weight_error"]["ideal_quantized_total"]["rmse"],
                "ideal_rmse_layer1": record["ideal_weight_errors"][1]["teacher_weight_error"]["ideal_quantized_total"]["rmse"],
                "pv_persistent_relative_l2_mean_layer0": sum(
                    float(repeat[0]["relative_l2"])
                    for repeat in record["pv_persistent_weight_errors"]
                ) / P_AND_V_REPEAT_COUNT,
                "pv_persistent_relative_l2_mean_layer1": sum(
                    float(repeat[1]["relative_l2"])
                    for repeat in record["pv_persistent_weight_errors"]
                ) / P_AND_V_REPEAT_COUNT,
                "pv_persistent_rmse_mean_layer0": sum(
                    float(repeat[0]["rmse"])
                    for repeat in record["pv_persistent_weight_errors"]
                ) / P_AND_V_REPEAT_COUNT,
                "pv_persistent_rmse_mean_layer1": sum(
                    float(repeat[1]["rmse"])
                    for repeat in record["pv_persistent_weight_errors"]
                ) / P_AND_V_REPEAT_COUNT,
                "selected_scale_layer0": record["selected_scale_fractions"][0],
                "selected_scale_layer1": record["selected_scale_fractions"][1],
                "fixed_logit_gain": record["fixed_logit_gain"],
                "development_hardware_instance_id": record["development_hardware_instance_id"],
                "heldout_hardware_instance_id": record["heldout_hardware_instance_id"],
            }
        )
    return stream.getvalue()


def _markdown(study: Mapping[str, Any], aggregate: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    lines = [
        f"# {study['title']} — audited analysis",
        "",
        "All 27 production CUDA bundles passed registered-byte, full-G mapping, frozen-calibration, matched-hardware, matched-RNG, raw-endpoint, public-projection, prediction, and historical-parity checks.",
        "",
        "| alpha | spacing | ideal | persistent P&V | ideal→P&V drop |",
        "|---:|---:|---:|---:|---:|",
    ]
    for alpha, spacing, arm_id in DESIGNS:
        value = aggregate["designs"][arm_id]
        lines.append(
            f"| {alpha:g} | {spacing} delta_x | {100 * value['ideal']['accuracy']:.2f}% | "
            f"{100 * value['pv_persistent']['accuracy']:.2f}% | "
            f"{100 * value['ideal_to_pv_accuracy_drop']:.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## DRN weight error versus frozen ReLU weights",
            "",
            "| alpha | spacing | ideal rel-L2 L0/L1 | persistent P&V rel-L2 L0/L1 | ideal RMSE L0/L1 | persistent P&V RMSE L0/L1 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for alpha, spacing, arm_id in DESIGNS:
        error = aggregate["designs"][arm_id]["weight_error"]
        ideal_l2 = error["ideal_relative_l2_mean_by_layer"]
        pv_l2 = error["pv_persistent_relative_l2_mean_by_layer"]
        ideal_rmse = error["ideal_rmse_mean_by_layer"]
        pv_rmse = error["pv_persistent_rmse_mean_by_layer"]
        lines.append(
            f"| {alpha:g} | {spacing} delta_x | {ideal_l2[0]:.6g} / {ideal_l2[1]:.6g} | "
            f"{pv_l2[0]:.6g} / {pv_l2[1]:.6g} | "
            f"{ideal_rmse[0]:.6g} / {ideal_rmse[1]:.6g} | "
            f"{pv_rmse[0]:.6g} / {pv_rmse[1]:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Voltage and program/verify diagnostics",
            "",
            "| alpha | spacing | ideal voltage RMS S0/S1/S2 | persistent voltage RMS S0/S1/S2 | requested-code correct | apparent accepted | exhausted | mean SET/RESET/total pulses per cell | persistent full-G RMSE L0/L1 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for alpha, spacing, arm_id in DESIGNS:
        value = aggregate["designs"][arm_id]
        ideal_voltage = value["network_diagnostics"]["ideal_quantized"][
            "voltage_rms_mean_by_state_layer"
        ]
        pv_voltage = value["network_diagnostics"]["pv_persistent"][
            "voltage_rms_mean_by_state_layer"
        ]
        pv = value["program_verify_diagnostics"]
        cells = pv["cells_total"]
        counts = pv["counts_total"]
        residual = pv[
            "persistent_full_conductance_residual_rmse_mean_by_layer"
        ]
        pulse_distributions = pv["mean_of_repeat_pulse_distributions"]
        lines.append(
            f"| {alpha:g} | {spacing} delta_x | "
            f"{ideal_voltage[0]:.5g} / {ideal_voltage[1]:.5g} / {ideal_voltage[2]:.5g} | "
            f"{pv_voltage[0]:.5g} / {pv_voltage[1]:.5g} / {pv_voltage[2]:.5g} | "
            f"{100 * counts['requested_code_correct'] / cells:.2f}% | "
            f"{100 * counts['apparent_accepted'] / cells:.2f}% | "
            f"{100 * counts['budget_exhausted'] / cells:.2f}% | "
            f"{pulse_distributions['set_count']['mean']:.3f} / "
            f"{pulse_distributions['reset_count']['mean']:.3f} / "
            f"{pulse_distributions['total_pulses']['mean']:.3f} | "
            f"{residual[0]:.6g} / {residual[1]:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Programmed endpoint loading",
            "",
            "| alpha | spacing | persistent mean loading L0/L1 | apparent mean loading L0/L1 | persistent RMS(C)/mean(L) L0/L1 | apparent RMS(C)/mean(L) L0/L1 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for alpha, spacing, arm_id in DESIGNS:
        pv = aggregate["designs"][arm_id]["program_verify_diagnostics"]
        persistent_loading = pv["persistent_loading_mean_by_layer"]
        apparent_loading = pv["apparent_loading_mean_by_layer"]
        persistent_ratio = pv[
            "persistent_rms_contrast_over_mean_loading_by_layer"
        ]
        apparent_ratio = pv[
            "apparent_rms_contrast_over_mean_loading_by_layer"
        ]
        lines.append(
            f"| {alpha:g} | {spacing} delta_x | "
            f"{persistent_loading[0]:.6g} / {persistent_loading[1]:.6g} | "
            f"{apparent_loading[0]:.6g} / {apparent_loading[1]:.6g} | "
            f"{persistent_ratio[0]:.6g} / {persistent_ratio[1]:.6g} | "
            f"{apparent_ratio[0]:.6g} / {apparent_ratio[1]:.6g} |"
        )
    lines.extend(["", "## Predeclared decision", ""])
    if decision["predeclared_unique_practical_winner"] is None:
        lines.append("No design satisfies the predeclared unique practical-winner rule.")
    else:
        lines.append(f"The predeclared unique practical winner is `{decision['predeclared_unique_practical_winner']}`.")
    lines.extend(["", f"Best ideal design: `{decision['best_ideal_design']}` ({100 * decision['best_ideal_accuracy']:.2f}%).", ""])
    return "\n".join(lines)


def analyze_study(
    study_dir: Path | str,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(study_dir).expanduser().resolve()
    study = load_study_record(root)
    records = _discover(root, study)
    aggregate, decision = _aggregate(records)
    output = root / "analysis" if output_dir is None else Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "claim_boundary": "four_device_shared_destination_model_based_om_ideal_and_raw_active_pv_without_training_or_inference_read_noise",
        "audit": {
            "complete_run_count": len(records),
            "expected_run_count": EXPECTED_RUNS,
            "runtime_device": "cuda",
            "examples_per_endpoint": EXPECTED_EXAMPLES,
            "pv_repeats_per_assignment": P_AND_V_REPEAT_COUNT,
            "source_sha256": EXPECTED_WEIGHTS_SHA256,
            "run_source_commit": records[0]["run_source"]["commit"],
            "registered_artifacts_verified": True,
            "full_g_mapping_recomputed": True,
            "raw_to_public_projection_recomputed": True,
            "continuation_state_hashes_recomputed": True,
            "prediction_counts_and_hashes_recomputed": True,
            "per_alpha_calibration_byte_identical": True,
            "matched_hardware_and_rng_streams": True,
            "historical_alpha0_spacing4_parity": True,
        },
        "runs": records,
        **aggregate,
        "decision": decision,
        "outputs": {"json": OUTPUT_JSON, "csv": OUTPUT_CSV, "markdown": OUTPUT_MARKDOWN},
    }
    atomic_write_json(output / OUTPUT_JSON, report)
    _atomic_text(output / OUTPUT_CSV, _csv(records))
    _atomic_text(output / OUTPUT_MARKDOWN, _markdown(study, aggregate, decision))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    analyze_study(args.study_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SCHEMA",
    "ANALYSIS_SCHEMA_VERSION",
    "BaselineSpacingPvAnalysisError",
    "analyze_study",
    "main",
]
