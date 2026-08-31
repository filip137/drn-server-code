"""Lightweight post-run analysis for the nominal-bound-Winsorized OM screen.

This module intentionally consumes the compact ``scientific_summary.json``
written by each exploratory CUDA run.  It does not rerun the network and it
does not describe nominal-bound Winsorization as the default IBM OM model.
The intervention is an analyst counterfactual: frozen sampled device bounds
are intersected with raw ``a in [-1, 1]`` before RESET commissioning and P&V.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from itertools import product
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from experiments.artifacts import atomic_write_json


ANALYSIS_SCHEMA = (
    "ebl.mnist_relu_drn."
    "ibm_om_baseline_spacing_pv_truncated_nominal_exploratory_analysis"
)
ANALYSIS_SCHEMA_VERSION = 1
RESULT_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_baseline_spacing_pv_truncated_nominal_result"
)
RESULT_SCHEMA_VERSION = 1
BASELINE_POSITION_FRACTIONS = (0.0, 0.25, 0.5)
SPACING_DELTA_X_MULTIPLIERS = (1, 2, 4)
HELDOUT_ASSIGNMENT_SEEDS = (87001, 87002, 87003)
ENDPOINT_SEEDS_BY_ASSIGNMENT = {
    87001: (89101, 89102, 89103, 89104, 89105),
    87002: (89201, 89202, 89203, 89204, 89205),
    87003: (89301, 89302, 89303, 89304, 89305),
}
EXPECTED_REPEATS = 5
EXPECTED_DESIGNS = tuple(
    product(BASELINE_POSITION_FRACTIONS, SPACING_DELTA_X_MULTIPLIERS)
)
EXPECTED_KEYS = frozenset(
    product(
        BASELINE_POSITION_FRACTIONS,
        SPACING_DELTA_X_MULTIPLIERS,
        HELDOUT_ASSIGNMENT_SEEDS,
    )
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIPPED_ROOT = (
    _REPO_ROOT
    / "results/mnist-ibm-om-baseline-spacing-pv-exploratory-20260828-v1"
)
DEFAULT_NO_CLIP_ROOT = (
    _REPO_ROOT
    / "results/mnist-ibm-om-baseline-spacing-pv-no-clip-exploratory-20260828-v1"
)


class WinsorizedNominalAnalysisError(RuntimeError):
    """Raised when the exploratory matrix cannot be interpreted safely."""


def _fail(message: str) -> None:
    raise WinsorizedNominalAnalysisError(message)


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"Cannot read JSON {path}: {error}")
    if not isinstance(value, dict):
        _fail(f"Expected a JSON object in {path}.")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(f"Expected object {label}.")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        _fail(f"Expected list {label}.")
    return value


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"Expected numeric {label}.")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"Expected finite {label}.")
    return result


def _integer(value: Any, *, label: str) -> int:
    number = _number(value, label=label)
    result = int(number)
    if float(result) != number:
        _fail(f"Expected integer {label}.")
    return result


def _accuracy(metric: Any, *, label: str) -> float:
    mapping = _mapping(metric, label=label)
    accuracy = _number(mapping.get("student_accuracy"), label=f"{label}.student_accuracy")
    if not 0.0 <= accuracy <= 1.0:
        _fail(f"Expected {label}.student_accuracy in [0,1].")
    examples = _integer(mapping.get("examples"), label=f"{label}.examples")
    correct = _integer(mapping.get("student_correct"), label=f"{label}.student_correct")
    if examples <= 0 or not 0 <= correct <= examples:
        _fail(f"Invalid correct/example counts in {label}.")
    if not math.isclose(accuracy, correct / examples, abs_tol=1e-12):
        _fail(f"Accuracy/count mismatch in {label}.")
    return accuracy


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        _fail("Cannot average an empty sequence.")
    return float(fmean(materialized))


def _stats(values: Iterable[float]) -> Mapping[str, Any]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"count": 0, "mean": None, "minimum": None, "maximum": None}
    if not all(math.isfinite(value) for value in materialized):
        _fail("Cannot summarize non-finite values.")
    return {
        "count": len(materialized),
        "mean": _mean(materialized),
        "minimum": min(materialized),
        "maximum": max(materialized),
    }


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _voltage_rows(metric: Mapping[str, Any], *, state: str) -> list[dict[str, Any]]:
    raw = metric.get("voltage")
    if not isinstance(raw, list):
        return []
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        layer = item.get("layer")
        rms = _optional_number(item.get("rms"))
        if isinstance(layer, int) and rms is not None:
            rows.append({"state": state, "layer": layer, "rms": rms})
    return rows


def _ideal_weight_rows(heldout: Mapping[str, Any]) -> list[dict[str, Any]]:
    report = heldout.get("weight_errors")
    if not isinstance(report, dict):
        return []
    raw_layers = report.get("ideal_layers")
    if not isinstance(raw_layers, list):
        return []
    rows = []
    for item in raw_layers:
        if not isinstance(item, dict) or not isinstance(item.get("layer"), int):
            continue
        normalized = _nested(
            item, "normalized_relu_weight_error", "ideal_quantized_total"
        )
        continuous = _nested(
            item, "normalized_relu_weight_error", "continuous_total"
        )
        endpoint = item.get("ideal_quantized_endpoint")
        if not isinstance(normalized, dict):
            normalized = {}
        if not isinstance(continuous, dict):
            continuous = {}
        if not isinstance(endpoint, dict):
            endpoint = {}
        rows.append(
            {
                "layer": int(item["layer"]),
                "continuous_rmse": _optional_number(continuous.get("rmse")),
                "continuous_mae": _optional_number(continuous.get("mae")),
                "ideal_quantized_rmse": _optional_number(normalized.get("rmse")),
                "ideal_quantized_mae": _optional_number(normalized.get("mae")),
                "ideal_quantized_relative_l2": _optional_number(
                    endpoint.get("relative_l2")
                ),
                "ideal_quantized_cosine_similarity": _optional_number(
                    endpoint.get("cosine_similarity")
                ),
            }
        )
    return rows


def _persistent_weight_rows(heldout: Mapping[str, Any]) -> list[dict[str, Any]]:
    reports = heldout.get("pv_persistent_weight_errors")
    if not isinstance(reports, list):
        return []
    rows = []
    for repeat_index, report in enumerate(reports):
        if not isinstance(report, dict) or not isinstance(report.get("layers"), list):
            continue
        endpoint_seed = report.get("endpoint_seed")
        for item in report["layers"]:
            if not isinstance(item, dict) or not isinstance(item.get("layer"), int):
                continue
            rows.append(
                {
                    "repeat_index": repeat_index,
                    "endpoint_seed": endpoint_seed,
                    "layer": int(item["layer"]),
                    "rmse": _optional_number(item.get("rmse")),
                    "mae": _optional_number(item.get("mae")),
                    "relative_l2": _optional_number(item.get("relative_l2")),
                    "cosine_similarity": _optional_number(
                        item.get("cosine_similarity")
                    ),
                }
            )
    return rows


def _mapping_load_rows(heldout: Mapping[str, Any]) -> list[dict[str, Any]]:
    invariants = heldout.get("invariants")
    mapping_report = heldout.get("mapping")
    invariant_layers = (
        invariants.get("layers") if isinstance(invariants, dict) else None
    )
    physical_layers = (
        _nested(mapping_report, "physical", "layers")
        if isinstance(mapping_report, dict)
        else None
    )
    if not isinstance(invariant_layers, list):
        invariant_layers = []
    if not isinstance(physical_layers, list):
        physical_layers = []
    physical_by_layer = {
        item.get("layer"): item
        for item in physical_layers
        if isinstance(item, dict) and isinstance(item.get("layer"), int)
    }
    rows = []
    for item in invariant_layers:
        if not isinstance(item, dict) or not isinstance(item.get("layer"), int):
            continue
        layer = int(item["layer"])
        physical = physical_by_layer.get(layer, {})
        rows.append(
            {
                "layer": layer,
                "baseline_loading_mean": _optional_number(
                    _nested(item, "baseline_loading", "mean")
                ),
                "continuous_loading_mean": _optional_number(
                    _nested(item, "continuous_loading", "mean")
                ),
                # The inherited invariant key is historical; in this runtime it
                # contains the selected ideal quantized target for the requested h.
                "ideal_quantized_loading_mean": _optional_number(
                    _nested(item, "standard4delta_loading", "mean")
                ),
                "ideal_contrast_rms": _optional_number(
                    _nested(physical, "ideal_contrast", "rms")
                ),
            }
        )
    return rows


def _endpoint_load_rows(endpoints: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    for repeat_index, endpoint in enumerate(endpoints):
        if not isinstance(endpoint, dict):
            continue
        raw_layers = endpoint.get("full_conductance_residuals_by_layer")
        if not isinstance(raw_layers, list):
            continue
        for item in raw_layers:
            if not isinstance(item, dict) or not isinstance(item.get("layer"), int):
                continue
            rows.append(
                {
                    "repeat_index": repeat_index,
                    "layer": int(item["layer"]),
                    "persistent_loading_mean": _optional_number(
                        _nested(item, "persistent_loading", "mean")
                    ),
                    "persistent_loading_rms": _optional_number(
                        _nested(item, "persistent_loading", "rms")
                    ),
                    "persistent_rms_contrast_over_mean_loading": _optional_number(
                        item.get("persistent_rms_contrast_over_mean_loading")
                    ),
                }
            )
    return rows


def _baseline_projection_report(value: Any, *, label: str) -> Mapping[str, Any]:
    report = _mapping(value, label=label)
    if (
        report.get("policy")
        != "project_requested_RESET_max_once_to_exact_common_support"
    ):
        _fail(f"Unexpected requested-baseline projection policy in {label}.")
    if report.get("stage") != "target_construction_before_quantization_and_pv":
        _fail(f"Unexpected requested-baseline projection stage in {label}.")
    if report.get("persistent_endpoint_projection") is not False:
        _fail(
            "Requested-baseline projection was conflated with endpoint "
            f"projection in {label}."
        )
    pairs = _integer(
        report.get("destination_pair_count"),
        label=f"{label}.destination_pair_count",
    )
    projected = _integer(
        report.get("projected_pair_count"),
        label=f"{label}.projected_pair_count",
    )
    fraction = _number(
        report.get("projected_pair_fraction"),
        label=f"{label}.projected_pair_fraction",
    )
    if pairs <= 0 or not 0 <= projected <= pairs or not math.isclose(
        fraction, projected / pairs, abs_tol=1e-12
    ):
        _fail(f"Inconsistent requested-baseline projection counts in {label}.")
    absolute_sum = _number(
        report.get("absolute_projection_sum_raw_x"),
        label=f"{label}.absolute_projection_sum_raw_x",
    )
    mean_all = _number(
        report.get("absolute_projection_mean_all_pairs_raw_x"),
        label=f"{label}.absolute_projection_mean_all_pairs_raw_x",
    )
    mean_projected = _number(
        report.get("absolute_projection_mean_projected_pairs_raw_x"),
        label=f"{label}.absolute_projection_mean_projected_pairs_raw_x",
    )
    maximum = _number(
        report.get("absolute_projection_maximum_raw_x"),
        label=f"{label}.absolute_projection_maximum_raw_x",
    )
    if not math.isclose(mean_all, absolute_sum / pairs, abs_tol=1e-12):
        _fail(f"Requested-baseline mean-all magnitude mismatch in {label}.")
    expected_projected_mean = absolute_sum / projected if projected else 0.0
    if not math.isclose(mean_projected, expected_projected_mean, abs_tol=1e-12):
        _fail(f"Requested-baseline projected-mean magnitude mismatch in {label}.")
    if min(absolute_sum, mean_all, mean_projected, maximum) < 0.0:
        _fail(f"Negative requested-baseline projection magnitude in {label}.")
    raw_layers = _sequence(report.get("layers"), label=f"{label}.layers")
    downward = 0
    upward = 0
    layer_pairs = 0
    layer_projected = 0
    for item_raw in raw_layers:
        item = _mapping(item_raw, label=f"{label}.layer")
        layer_pairs += _integer(
            item.get("destination_pair_count"),
            label=f"{label}.layer.destination_pair_count",
        )
        layer_projected += _integer(
            item.get("projected_pair_count"),
            label=f"{label}.layer.projected_pair_count",
        )
        downward += _integer(
            item.get("downward_projection_count"),
            label=f"{label}.layer.downward_projection_count",
        )
        upward += _integer(
            item.get("upward_projection_count"),
            label=f"{label}.layer.upward_projection_count",
        )
    if layer_pairs != pairs or layer_projected != projected or downward + upward != projected:
        _fail(f"Requested-baseline layer projection counts mismatch in {label}.")
    return {
        "policy": report["policy"],
        "stage": report["stage"],
        "persistent_endpoint_projection": False,
        "destination_pair_count": pairs,
        "projected_pair_count": projected,
        "projected_pair_fraction": fraction,
        "downward_projection_count": downward,
        "upward_projection_count": upward,
        "absolute_projection_sum_raw_x": absolute_sum,
        "absolute_projection_mean_all_pairs_raw_x": mean_all,
        "absolute_projection_mean_projected_pairs_raw_x": mean_projected,
        "absolute_projection_maximum_raw_x": maximum,
        "layers": raw_layers,
    }


def _require_complete_run(summary_path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    run_dir = summary_path.parent.parent
    status_path = run_dir / "status.json"
    manifest_path = run_dir / "manifest.json"
    if not status_path.is_file() or not manifest_path.is_file():
        _fail(f"Missing status.json or manifest.json beside {summary_path}.")
    status = _json(status_path)
    manifest = _json(manifest_path)
    if status.get("status") != "complete":
        _fail(f"Run {run_dir} is not complete.")
    if status.get("run_id") != manifest.get("run_id"):
        _fail(f"Run id mismatch in {run_dir}.")
    return status, manifest


def _current_record(summary_path: Path) -> Mapping[str, Any]:
    summary = _json(summary_path)
    if (
        summary.get("schema") != RESULT_SCHEMA
        or summary.get("schema_version") != RESULT_SCHEMA_VERSION
    ):
        _fail(f"Unexpected scientific-summary schema in {summary_path}.")
    if summary.get("status") != "complete":
        _fail(f"Scientific summary is not complete: {summary_path}.")
    status, manifest = _require_complete_run(summary_path)
    validity = _mapping(summary.get("validity"), label="validity")
    gates = _mapping(validity.get("gates"), label="validity.gates")
    if validity.get("coverage_valid") is not True or not gates or not all(
        value is True for value in gates.values()
    ):
        _fail(f"A validity gate failed in {summary_path}.")
    if validity.get("apparent_endpoint_applied_to_drn") is not False:
        _fail(f"Apparent endpoint was applied in {summary_path}.")
    if _number(validity.get("inference_read_noise"), label="inference_read_noise") != 0.0:
        _fail(f"Inference read noise is nonzero in {summary_path}.")

    design = _mapping(summary.get("design"), label="design")
    heldout = _mapping(summary.get("heldout"), label="heldout")
    alpha = _number(
        design.get("baseline_position_fraction"),
        label="design.baseline_position_fraction",
    )
    spacing = _integer(
        design.get("spacing_delta_x_multiplier"),
        label="design.spacing_delta_x_multiplier",
    )
    assignment_seed = _integer(
        heldout.get("assignment_seed"), label="heldout.assignment_seed"
    )
    key = (alpha, spacing, assignment_seed)
    if key not in EXPECTED_KEYS:
        _fail(f"Unexpected design/assignment key {key!r} in {summary_path}.")

    continuous_metric = _mapping(heldout.get("continuous"), label="heldout.continuous")
    ideal_metric = _mapping(
        heldout.get("ideal_quantized"), label="heldout.ideal_quantized"
    )
    continuous_accuracy = _accuracy(continuous_metric, label="heldout.continuous")
    ideal_accuracy = _accuracy(ideal_metric, label="heldout.ideal_quantized")
    repeat_metrics = _sequence(
        heldout.get("pv_persistent_repeats"),
        label="heldout.pv_persistent_repeats",
    )
    endpoints = _sequence(
        heldout.get("pv_endpoint_artifacts"),
        label="heldout.pv_endpoint_artifacts",
    )
    if len(repeat_metrics) != EXPECTED_REPEATS or len(endpoints) != EXPECTED_REPEATS:
        _fail(f"Expected five persistent P&V repeats in {summary_path}.")
    expected_endpoint_seeds = ENDPOINT_SEEDS_BY_ASSIGNMENT[assignment_seed]
    observed_endpoint_seeds = tuple(
        _integer(
            _mapping(endpoint, label="P&V endpoint").get("endpoint_seed"),
            label="P&V endpoint seed",
        )
        for endpoint in endpoints
    )
    if observed_endpoint_seeds != expected_endpoint_seeds:
        _fail(
            f"Endpoint seed mismatch in {summary_path}: "
            f"{observed_endpoint_seeds!r}."
        )
    contract_endpoint_seeds = _nested(summary, "contract", "assignments", "endpoint_seeds")
    if list(expected_endpoint_seeds) != contract_endpoint_seeds:
        _fail(f"Contract endpoint seeds mismatch in {summary_path}.")

    repeat_accuracies = []
    cell_events = 0
    counts = {
        key: 0
        for key in (
            "exact_target_in_support",
            "verify_window_intersects_support",
            "apparent_accepted",
            "persistent_inside_acceptance_window",
            "requested_code_correct",
            "nearest_code_adjacent_to_requested",
            "nearest_code_farther_than_adjacent",
            "budget_exhausted",
            "nonfinite",
            "saturated_lower",
            "saturated_upper",
        )
    }
    pulse_totals = {key: 0 for key in ("set", "reset", "total", "verify", "reversals")}
    projection_count = 0
    below_support = 0
    above_support = 0
    apparent_applied = 0
    residual_rmse_squared_cells = 0.0
    residual_mae_cells = 0.0
    for repeat_index, (metric_raw, endpoint_raw) in enumerate(zip(repeat_metrics, endpoints)):
        metric = _mapping(metric_raw, label="persistent repeat metric")
        endpoint = _mapping(endpoint_raw, label="P&V endpoint")
        accuracy = _accuracy(metric, label=f"persistent repeat {repeat_index}")
        endpoint_metric = _mapping(
            endpoint.get("persistent_network_metrics"),
            label="endpoint persistent network metrics",
        )
        endpoint_accuracy = _accuracy(
            endpoint_metric, label=f"endpoint persistent metric {repeat_index}"
        )
        if not math.isclose(accuracy, endpoint_accuracy, abs_tol=1e-12):
            _fail(f"Persistent metric mismatch in {summary_path}.")
        repeat_accuracies.append(accuracy)
        cells = _integer(endpoint.get("cells"), label="endpoint cells")
        if cells <= 0:
            _fail(f"Expected positive endpoint cell count in {summary_path}.")
        cell_events += cells
        endpoint_counts = _mapping(endpoint.get("counts"), label="endpoint counts")
        for name in counts:
            value = _integer(endpoint_counts.get(name), label=f"endpoint counts.{name}")
            if not 0 <= value <= cells:
                _fail(f"Invalid endpoint count {name} in {summary_path}.")
            counts[name] += value
        pulses = _mapping(endpoint.get("pulse_totals"), label="endpoint pulse_totals")
        for name in pulse_totals:
            value = _integer(pulses.get(name), label=f"pulse_totals.{name}")
            if value < 0:
                _fail(f"Negative pulse count in {summary_path}.")
            pulse_totals[name] += value
        handoff = _mapping(
            endpoint.get("persistent_affine_handoff"),
            label="persistent affine handoff",
        )
        projection_count += _integer(handoff.get("projected"), label="projected")
        below_support += _integer(
            handoff.get("below_native_support"), label="below native support"
        )
        above_support += _integer(
            handoff.get("above_native_support"), label="above native support"
        )
        apparent = _mapping(endpoint.get("apparent_endpoint"), label="apparent endpoint")
        apparent_applied += int(apparent.get("applied_to_drn") is True)
        residual = endpoint.get("persistent_target_residual_unit")
        if isinstance(residual, dict):
            rmse = _optional_number(residual.get("rmse"))
            mae = _optional_number(residual.get("mae"))
            if rmse is not None:
                residual_rmse_squared_cells += rmse * rmse * cells
            if mae is not None:
                residual_mae_cells += mae * cells

    recorded_pv_mean = _number(
        heldout.get("pv_persistent_mean_accuracy"),
        label="heldout.pv_persistent_mean_accuracy",
    )
    if not math.isclose(recorded_pv_mean, _mean(repeat_accuracies), abs_tol=1e-12):
        _fail(f"Persistent mean accuracy mismatch in {summary_path}.")
    target_handoff = _mapping(
        heldout.get("target_affine_handoff"), label="target affine handoff"
    )
    target_projection_count = _integer(
        target_handoff.get("projected"), label="target projected"
    )
    if (
        target_projection_count
        or projection_count
        or below_support
        or above_support
        or apparent_applied
        or counts["nonfinite"]
        or counts["exact_target_in_support"] != cell_events
    ):
        _fail(f"Physical handoff/support gate failed in {summary_path}.")

    heldout_mapping = _mapping(heldout.get("mapping"), label="heldout.mapping")
    baseline_projection = _baseline_projection_report(
        heldout_mapping.get("shared_destination_baseline_projection"),
        label="heldout.mapping.shared_destination_baseline_projection",
    )
    design_projection = design.get("heldout_shared_destination_baseline_projection")
    if design_projection != heldout_mapping.get(
        "shared_destination_baseline_projection"
    ):
        _fail(
            "Design/mapping requested-baseline projection mismatch in "
            f"{summary_path}."
        )

    native = _mapping(heldout.get("native_coordinate"), label="heldout.native_coordinate")
    winsor = _mapping(native.get("winsorization"), label="heldout winsorization")
    if (
        winsor.get("policy") != "nominal_bound_winsorization"
        or winsor.get("identity_resampled") is not False
    ):
        _fail(f"Unexpected Winsorization policy in {summary_path}.")
    cells = _integer(winsor.get("cells"), label="winsorization cells")
    if _integer(winsor.get("empty_intersection_count"), label="empty intersections") != 0:
        _fail(f"Empty Winsorization intersection in {summary_path}.")
    clipping = {"cells": cells}
    for stem in ("lower_bound", "upper_bound", "both_bounds", "any_bound"):
        count = _integer(winsor.get(f"{stem}_clipped_count"), label=f"{stem} clipped count")
        fraction = _number(
            winsor.get(f"{stem}_clipped_fraction"), label=f"{stem} clipped fraction"
        )
        if not 0 <= count <= cells or not math.isclose(
            fraction, count / cells, abs_tol=2e-7
        ):
            _fail(f"Inconsistent {stem} Winsorization count in {summary_path}.")
        clipping[f"{stem}_clipped_count"] = count
        clipping[f"{stem}_clipped_fraction"] = fraction

    hardware = _mapping(heldout.get("hardware"), label="heldout.hardware")
    hardware_instance_id = hardware.get("hardware_instance_id")
    if not isinstance(hardware_instance_id, str) or not hardware_instance_id:
        _fail(f"Missing held-out hardware identity in {summary_path}.")
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    return {
        "key": key,
        "alpha": alpha,
        "spacing_delta_x_multiplier": spacing,
        "heldout_assignment_seed": assignment_seed,
        "endpoint_seeds": list(observed_endpoint_seeds),
        "run_id": status.get("run_id"),
        "summary_path": str(summary_path),
        "hardware_instance_id": hardware_instance_id,
        "source_commit": source.get("commit"),
        "source_dirty": source.get("dirty"),
        "source_dirty_hash": source.get("dirty_hash"),
        "config_sha256": config.get("sha256"),
        "continuous_accuracy": continuous_accuracy,
        "ideal_quantized_accuracy": ideal_accuracy,
        "persistent_pv_repeat_accuracies": repeat_accuracies,
        "persistent_pv_mean_accuracy": recorded_pv_mean,
        "clipping": clipping,
        "pv": {
            "cell_events": cell_events,
            "counts": counts,
            "pulse_totals": pulse_totals,
            "persistent_target_residual_rmse": math.sqrt(
                residual_rmse_squared_cells / cell_events
            ),
            "persistent_target_residual_mae": residual_mae_cells / cell_events,
        },
        "handoff": {
            "target_projection_count": target_projection_count,
            "persistent_projection_count": projection_count,
            "below_native_support_count": below_support,
            "above_native_support_count": above_support,
            "apparent_endpoint_applied_count": apparent_applied,
        },
        "requested_baseline_common_window_projection": baseline_projection,
        "voltage": (
            _voltage_rows(continuous_metric, state="continuous")
            + _voltage_rows(ideal_metric, state="ideal_quantized")
            + [
                row
                for metric in repeat_metrics
                for row in _voltage_rows(
                    _mapping(metric, label="persistent repeat voltage metric"),
                    state="persistent_pv",
                )
            ]
        ),
        "mapping_loads": _mapping_load_rows(heldout),
        "persistent_loads": _endpoint_load_rows(endpoints),
        "ideal_weight_errors": _ideal_weight_rows(heldout),
        "persistent_weight_errors": _persistent_weight_rows(heldout),
    }


def _summary_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        _fail(f"Result root does not exist: {root}")
    return sorted(root.rglob("artifacts/scientific_summary.json"))


def _load_current_records(root: Path) -> list[Mapping[str, Any]]:
    paths = _summary_paths(root)
    if len(paths) != len(EXPECTED_KEYS):
        _fail(
            f"Expected exactly 27 scientific summaries below {root}, found {len(paths)}."
        )
    records = [_current_record(path) for path in paths]
    observed = [record["key"] for record in records]
    if len(set(observed)) != len(observed):
        _fail("Duplicate complete design/assignment combinations found.")
    missing = EXPECTED_KEYS - set(observed)
    extra = set(observed) - EXPECTED_KEYS
    if missing or extra:
        _fail(f"Matrix mismatch: missing={sorted(missing)!r}, extra={sorted(extra)!r}.")
    return sorted(records, key=lambda item: item["key"])


def _comparison_records(root: Path) -> list[Mapping[str, Any]]:
    paths = _summary_paths(root)
    rows = []
    for path in paths:
        summary = _json(path)
        if summary.get("status") != "complete":
            continue
        design = summary.get("design")
        heldout = summary.get("heldout")
        if not isinstance(design, dict) or not isinstance(heldout, dict):
            continue
        alpha = _number(design.get("baseline_position_fraction"), label="comparison alpha")
        spacing = _integer(design.get("spacing_delta_x_multiplier"), label="comparison spacing")
        seed = _integer(heldout.get("assignment_seed"), label="comparison assignment")
        key = (alpha, spacing, seed)
        if key not in EXPECTED_KEYS:
            continue
        repeats = _sequence(
            heldout.get("pv_persistent_repeats"), label="comparison P&V repeats"
        )
        if len(repeats) != EXPECTED_REPEATS:
            _fail(f"Comparison root has non-five-repeat run: {path}.")
        rows.append(
            {
                "key": key,
                "alpha": alpha,
                "spacing_delta_x_multiplier": spacing,
                "heldout_assignment_seed": seed,
                "continuous_accuracy": _accuracy(
                    heldout.get("continuous"), label="comparison continuous"
                ),
                "ideal_quantized_accuracy": _accuracy(
                    heldout.get("ideal_quantized"), label="comparison ideal"
                ),
                "persistent_pv_mean_accuracy": _mean(
                    _accuracy(metric, label="comparison persistent")
                    for metric in repeats
                ),
            }
        )
    keys = [row["key"] for row in rows]
    if len(rows) != len(EXPECTED_KEYS) or len(set(keys)) != len(EXPECTED_KEYS):
        _fail(
            f"Comparison root {root} does not contain exactly one complete 27-cell matrix."
        )
    return sorted(rows, key=lambda item: item["key"])


def _design_groups(
    records: Sequence[Mapping[str, Any]],
) -> Mapping[tuple[float, int], list[Mapping[str, Any]]]:
    groups = {design: [] for design in EXPECTED_DESIGNS}
    for record in records:
        groups[(record["alpha"], record["spacing_delta_x_multiplier"])].append(record)
    if any(len(group) != len(HELDOUT_ASSIGNMENT_SEEDS) for group in groups.values()):
        _fail("Expected three held-out assignments per design.")
    return groups


def _aggregate_named_rows(
    records: Sequence[Mapping[str, Any]],
    record_key: str,
    *,
    grouping_keys: Sequence[str],
    value_keys: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    for record in records:
        rows = record.get(record_key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            group = tuple(row.get(key) for key in grouping_keys)
            destination = grouped.setdefault(
                group, {key: [] for key in value_keys}
            )
            for key in value_keys:
                value = _optional_number(row.get(key))
                if value is not None:
                    destination[key].append(value)
    result = []
    for group, values in sorted(grouped.items()):
        row = dict(zip(grouping_keys, group))
        row.update({key: _stats(items) for key, items in values.items()})
        result.append(row)
    return result


def _design_row(
    design: tuple[float, int], records: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    alpha, spacing = design
    repeats = [
        accuracy
        for record in records
        for accuracy in record["persistent_pv_repeat_accuracies"]
    ]
    total_cells = sum(record["pv"]["cell_events"] for record in records)
    total_counts = {
        name: sum(record["pv"]["counts"][name] for record in records)
        for name in records[0]["pv"]["counts"]
    }
    total_pulses = {
        name: sum(record["pv"]["pulse_totals"][name] for record in records)
        for name in records[0]["pv"]["pulse_totals"]
    }
    voltage = _aggregate_named_rows(
        records,
        "voltage",
        grouping_keys=("state", "layer"),
        value_keys=("rms",),
    )
    mapping_loads = _aggregate_named_rows(
        records,
        "mapping_loads",
        grouping_keys=("layer",),
        value_keys=(
            "baseline_loading_mean",
            "continuous_loading_mean",
            "ideal_quantized_loading_mean",
            "ideal_contrast_rms",
        ),
    )
    persistent_loads = _aggregate_named_rows(
        records,
        "persistent_loads",
        grouping_keys=("layer",),
        value_keys=(
            "persistent_loading_mean",
            "persistent_loading_rms",
            "persistent_rms_contrast_over_mean_loading",
        ),
    )
    ideal_weights = _aggregate_named_rows(
        records,
        "ideal_weight_errors",
        grouping_keys=("layer",),
        value_keys=(
            "continuous_rmse",
            "continuous_mae",
            "ideal_quantized_rmse",
            "ideal_quantized_mae",
            "ideal_quantized_relative_l2",
            "ideal_quantized_cosine_similarity",
        ),
    )
    persistent_weights = _aggregate_named_rows(
        records,
        "persistent_weight_errors",
        grouping_keys=("layer",),
        value_keys=("rmse", "mae", "relative_l2", "cosine_similarity"),
    )
    baseline_pairs = sum(
        record["requested_baseline_common_window_projection"]["destination_pair_count"]
        for record in records
    )
    baseline_projected = sum(
        record["requested_baseline_common_window_projection"]["projected_pair_count"]
        for record in records
    )
    baseline_absolute_sum = sum(
        record["requested_baseline_common_window_projection"][
            "absolute_projection_sum_raw_x"
        ]
        for record in records
    )
    return {
        "alpha": alpha,
        "spacing_delta_x_multiplier": spacing,
        "continuous_accuracy": _mean(record["continuous_accuracy"] for record in records),
        "ideal_quantized_accuracy": _mean(
            record["ideal_quantized_accuracy"] for record in records
        ),
        "persistent_pv_accuracy": _mean(repeats),
        "persistent_pv_assignment_range": [
            min(record["persistent_pv_mean_accuracy"] for record in records),
            max(record["persistent_pv_mean_accuracy"] for record in records),
        ],
        "persistent_pv_repeat_range": [min(repeats), max(repeats)],
        "ideal_to_persistent_delta": _mean(repeats)
        - _mean(record["ideal_quantized_accuracy"] for record in records),
        "program_verify": {
            "cell_events": total_cells,
            "counts": total_counts,
            "fractions": {
                name: value / total_cells for name, value in total_counts.items()
            },
            "pulse_totals": total_pulses,
            "pulses_per_cell": total_pulses["total"] / total_cells,
            "persistent_target_residual_rmse": math.sqrt(
                sum(
                    record["pv"]["persistent_target_residual_rmse"] ** 2
                    * record["pv"]["cell_events"]
                    for record in records
                )
                / total_cells
            ),
            "persistent_target_residual_mae": sum(
                record["pv"]["persistent_target_residual_mae"]
                * record["pv"]["cell_events"]
                for record in records
            )
            / total_cells,
        },
        "requested_baseline_common_window_projection": {
            "destination_pair_count": baseline_pairs,
            "projected_pair_count": baseline_projected,
            "projected_pair_fraction": baseline_projected / baseline_pairs,
            "downward_projection_count": sum(
                record["requested_baseline_common_window_projection"][
                    "downward_projection_count"
                ]
                for record in records
            ),
            "upward_projection_count": sum(
                record["requested_baseline_common_window_projection"][
                    "upward_projection_count"
                ]
                for record in records
            ),
            "absolute_projection_sum_raw_x": baseline_absolute_sum,
            "absolute_projection_mean_all_pairs_raw_x": (
                baseline_absolute_sum / baseline_pairs
            ),
            "absolute_projection_mean_projected_pairs_raw_x": (
                baseline_absolute_sum / baseline_projected
                if baseline_projected
                else 0.0
            ),
            "absolute_projection_maximum_raw_x": max(
                record["requested_baseline_common_window_projection"][
                    "absolute_projection_maximum_raw_x"
                ]
                for record in records
            ),
        },
        "voltage_rms": voltage,
        "mapping_loads": mapping_loads,
        "persistent_loads": persistent_loads,
        "ideal_weight_errors": ideal_weights,
        "persistent_weight_errors": persistent_weights,
    }


def _assignment_row(record: Mapping[str, Any]) -> Mapping[str, Any]:
    row: dict[str, Any] = {
        key: record[key]
        for key in (
            "alpha",
            "spacing_delta_x_multiplier",
            "heldout_assignment_seed",
            "hardware_instance_id",
            "run_id",
            "config_sha256",
            "source_commit",
            "source_dirty",
            "source_dirty_hash",
            "continuous_accuracy",
            "ideal_quantized_accuracy",
            "persistent_pv_mean_accuracy",
            "persistent_pv_repeat_accuracies",
            "endpoint_seeds",
            "clipping",
            "handoff",
            "requested_baseline_common_window_projection",
        )
    }
    cells = record["pv"]["cell_events"]
    row["program_verify"] = {
        **record["pv"],
        "fractions": {
            name: value / cells for name, value in record["pv"]["counts"].items()
        },
        "pulses_per_cell": record["pv"]["pulse_totals"]["total"] / cells,
    }
    row["voltage"] = record["voltage"]
    row["mapping_loads"] = record["mapping_loads"]
    row["persistent_loads"] = record["persistent_loads"]
    row["ideal_weight_errors"] = record["ideal_weight_errors"]
    row["persistent_weight_errors"] = record["persistent_weight_errors"]
    return row


def _winsorization_by_assignment(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = []
    for seed in HELDOUT_ASSIGNMENT_SEEDS:
        matching = [record for record in records if record["heldout_assignment_seed"] == seed]
        canonical = matching[0]["clipping"]
        if any(record["clipping"] != canonical for record in matching[1:]):
            _fail(f"Winsorization receipt changed across designs for assignment {seed}.")
        result.append({"heldout_assignment_seed": seed, **canonical})
    return result


def _baseline_projection_by_assignment(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    result = []
    for seed in HELDOUT_ASSIGNMENT_SEEDS:
        matching = [record for record in records if record["heldout_assignment_seed"] == seed]
        canonical = matching[0]["requested_baseline_common_window_projection"]
        if any(
            record["requested_baseline_common_window_projection"] != canonical
            for record in matching[1:]
        ):
            _fail(
                "Requested-baseline common-window projection changed across "
                f"designs for assignment {seed}."
            )
        result.append({"heldout_assignment_seed": seed, **canonical})
    return result


def _comparison(
    label: str,
    root: Path,
    current_design_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    records = _comparison_records(root)
    groups = _design_groups(records)
    current = {
        (row["alpha"], row["spacing_delta_x_multiplier"]): row
        for row in current_design_rows
    }
    rows = []
    for design in EXPECTED_DESIGNS:
        prior = groups[design]
        prior_continuous = _mean(row["continuous_accuracy"] for row in prior)
        prior_ideal = _mean(row["ideal_quantized_accuracy"] for row in prior)
        prior_persistent = _mean(row["persistent_pv_mean_accuracy"] for row in prior)
        now = current[design]
        rows.append(
            {
                "alpha": design[0],
                "spacing_delta_x_multiplier": design[1],
                "prior_continuous_accuracy": prior_continuous,
                "current_continuous_accuracy": now["continuous_accuracy"],
                "continuous_delta_percentage_points": 100.0
                * (now["continuous_accuracy"] - prior_continuous),
                "prior_ideal_quantized_accuracy": prior_ideal,
                "current_ideal_quantized_accuracy": now["ideal_quantized_accuracy"],
                "ideal_quantized_delta_percentage_points": 100.0
                * (now["ideal_quantized_accuracy"] - prior_ideal),
                "prior_persistent_pv_accuracy": prior_persistent,
                "current_persistent_pv_accuracy": now["persistent_pv_accuracy"],
                "persistent_pv_delta_percentage_points": 100.0
                * (now["persistent_pv_accuracy"] - prior_persistent),
            }
        )
    return {"label": label, "root": str(root), "design_rows": rows}


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = [
        "alpha",
        "spacing_delta_x_multiplier",
        "heldout_assignment_seed",
        "hardware_instance_id",
        "run_id",
        "config_sha256",
        "continuous_accuracy",
        "ideal_quantized_accuracy",
        "persistent_pv_mean_accuracy",
        "persistent_pv_repeat_min",
        "persistent_pv_repeat_max",
        "lower_bound_clipped_fraction",
        "upper_bound_clipped_fraction",
        "both_bounds_clipped_fraction",
        "any_bound_clipped_fraction",
        "requested_baseline_projected_pair_count",
        "requested_baseline_projected_pair_fraction",
        "requested_baseline_projection_mean_projected_raw_x",
        "requested_baseline_projection_maximum_raw_x",
        "requested_code_correct_fraction",
        "apparent_accepted_fraction",
        "persistent_inside_acceptance_window_fraction",
        "saturated_lower_fraction",
        "saturated_upper_fraction",
        "pulses_per_cell",
        "persistent_target_residual_rmse",
        "persistent_target_residual_mae",
        "target_projection_count",
        "persistent_projection_count",
        "native_support_violation_count",
        "endpoint_seeds",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        clipping = row["clipping"]
        pv = row["program_verify"]
        fractions = pv["fractions"]
        handoff = row["handoff"]
        baseline_projection = row["requested_baseline_common_window_projection"]
        repeats = row["persistent_pv_repeat_accuracies"]
        writer.writerow(
            {
                "alpha": row["alpha"],
                "spacing_delta_x_multiplier": row["spacing_delta_x_multiplier"],
                "heldout_assignment_seed": row["heldout_assignment_seed"],
                "hardware_instance_id": row["hardware_instance_id"],
                "run_id": row["run_id"],
                "config_sha256": row["config_sha256"],
                "continuous_accuracy": row["continuous_accuracy"],
                "ideal_quantized_accuracy": row["ideal_quantized_accuracy"],
                "persistent_pv_mean_accuracy": row["persistent_pv_mean_accuracy"],
                "persistent_pv_repeat_min": min(repeats),
                "persistent_pv_repeat_max": max(repeats),
                "lower_bound_clipped_fraction": clipping["lower_bound_clipped_fraction"],
                "upper_bound_clipped_fraction": clipping["upper_bound_clipped_fraction"],
                "both_bounds_clipped_fraction": clipping["both_bounds_clipped_fraction"],
                "any_bound_clipped_fraction": clipping["any_bound_clipped_fraction"],
                "requested_baseline_projected_pair_count": baseline_projection[
                    "projected_pair_count"
                ],
                "requested_baseline_projected_pair_fraction": baseline_projection[
                    "projected_pair_fraction"
                ],
                "requested_baseline_projection_mean_projected_raw_x": (
                    baseline_projection[
                        "absolute_projection_mean_projected_pairs_raw_x"
                    ]
                ),
                "requested_baseline_projection_maximum_raw_x": baseline_projection[
                    "absolute_projection_maximum_raw_x"
                ],
                "requested_code_correct_fraction": fractions["requested_code_correct"],
                "apparent_accepted_fraction": fractions["apparent_accepted"],
                "persistent_inside_acceptance_window_fraction": fractions[
                    "persistent_inside_acceptance_window"
                ],
                "saturated_lower_fraction": fractions["saturated_lower"],
                "saturated_upper_fraction": fractions["saturated_upper"],
                "pulses_per_cell": pv["pulses_per_cell"],
                "persistent_target_residual_rmse": pv[
                    "persistent_target_residual_rmse"
                ],
                "persistent_target_residual_mae": pv[
                    "persistent_target_residual_mae"
                ],
                "target_projection_count": handoff["target_projection_count"],
                "persistent_projection_count": handoff[
                    "persistent_projection_count"
                ],
                "native_support_violation_count": handoff[
                    "below_native_support_count"
                ]
                + handoff["above_native_support_count"],
                "endpoint_seeds": ";".join(map(str, row["endpoint_seeds"])),
            }
        )
    return stream.getvalue()


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _mean_stat(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    for row in rows:
        stat = row.get(key)
        if isinstance(stat, dict) and stat.get("mean") is not None:
            return f"{float(stat['mean']):.6g}"
    return "not recorded"


def _markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Post-run analysis: nominal-bound-Winsorized IBM OM counterfactual",
        "",
        "This is an `exploratory_noncanonical` analyst counterfactual, not the "
        "default IBM optimized-material model. Frozen sampled OM identities are "
        "retained, but each sampled support is Winsorized to raw `a in [-1,1]` "
        "before repeated RESET commissioning and P&V. The circuit receives "
        "`G=a+1=2x` in `[0,2]`; no reference is subtracted and no persistent "
        "endpoint is projected after programming.",
        "",
        "## Coverage and gates",
        "",
        f"Coverage is **{analysis['coverage']['complete_configurations']}/27 configurations** "
        f"and **{analysis['coverage']['persistent_deployments']} persistent deployments** "
        "(five endpoint seeds for each of three assignments in each of nine designs).",
        "",
        "| gate | observed |",
        "|---|---:|",
    ]
    validity = analysis["validity"]
    for label, key in (
        ("target projections", "target_projection_count"),
        ("persistent projections", "persistent_projection_count"),
        ("persistent endpoints below support", "below_native_support_count"),
        ("persistent endpoints above support", "above_native_support_count"),
        ("apparent endpoints applied to DRN", "apparent_endpoint_applied_count"),
        ("non-finite programming outcomes", "nonfinite_programming_count"),
    ):
        lines.append(f"| {label} | {validity[key]} |")
    baseline_projection = analysis["requested_baseline_projection_unique_assignments"]
    lines.append(
        "| requested baselines moved into common support "
        f"(target-construction step, not endpoint projection) | "
        f"{baseline_projection['projected_pair_count']} / "
        f"{baseline_projection['destination_pair_count']} |"
    )
    lines.append(
        "| maximum requested-baseline movement | "
        f"{baseline_projection['absolute_projection_maximum_raw_x']:.6g} raw x |"
    )
    lines.extend(
        [
            "",
            "## Accuracy and programming matrix",
            "",
            "| α | spacing | continuous | ideal quantized | persistent P&V | "
            "ideal→P&V | baseline pairs moved | code correct | saturated "
            "low/high | pulses/cell |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["design_rows"]:
        pv = row["program_verify"]
        fractions = pv["fractions"]
        baseline = row["requested_baseline_common_window_projection"]
        lines.append(
            f"| {row['alpha']:.2f} | {row['spacing_delta_x_multiplier']}δx | "
            f"{_pct(row['continuous_accuracy'])} | {_pct(row['ideal_quantized_accuracy'])} | "
            f"{_pct(row['persistent_pv_accuracy'])} | "
            f"{100.0 * row['ideal_to_persistent_delta']:+.2f} pp | "
            f"{baseline['projected_pair_count']}/{baseline['destination_pair_count']} "
            f"({_pct(baseline['projected_pair_fraction'])}) | "
            f"{_pct(fractions['requested_code_correct'])} | "
            f"{_pct(fractions['saturated_lower'])}/{_pct(fractions['saturated_upper'])} | "
            f"{pv['pulses_per_cell']:.3f} |"
        )
    best_ideal = analysis["best_ideal_quantized_design"]
    best_pv = analysis["best_persistent_pv_design"]
    lines.extend(
        [
            "",
            f"The best ideal-mapped design is **α={best_ideal['alpha']:.2f}, "
            f"h={best_ideal['spacing_delta_x_multiplier']}δx** at "
            f"**{_pct(best_ideal['ideal_quantized_accuracy'])}**. The best "
            f"persistent design is **α={best_pv['alpha']:.2f}, "
            f"h={best_pv['spacing_delta_x_multiplier']}δx** at "
            f"**{_pct(best_pv['persistent_pv_accuracy'])}**.",
            "",
            "## Winsorized sampled bounds",
            "",
            "This is clipping of already sampled identities (Winsorization), not "
            "truncation by rejection/resampling. A clipped fraction therefore "
            "becomes point mass at `a=-1` or `a=+1` and changes the soft-bound "
            "pulse response of affected cells.",
            "",
            "| assignment | lower clipped | upper clipped | either clipped | "
            "both clipped | cells |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["winsorization_by_assignment"]:
        lines.append(
            f"| {row['heldout_assignment_seed']} | "
            f"{_pct(row['lower_bound_clipped_fraction'])} | "
            f"{_pct(row['upper_bound_clipped_fraction'])} | "
            f"{_pct(row['any_bound_clipped_fraction'])} | "
            f"{_pct(row['both_bounds_clipped_fraction'])} | {row['cells']} |"
        )

    lines.extend(
        [
            "",
            "## Voltage and full-conductance loading",
            "",
            "Loading values are in the analyst-defined normalized full-`G` units; "
            "they are not microSiemens or fabricated-array power measurements.",
            "",
            "| α | spacing | layer | ideal voltage RMS | persistent voltage "
            "RMS | baseline load mean | ideal load mean | persistent load "
            "mean | persistent RMS(C)/mean(load) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for design in analysis["design_rows"]:
        layers = sorted(
            {
                row["layer"]
                for collection in (
                    design["mapping_loads"],
                    design["persistent_loads"],
                )
                for row in collection
                if isinstance(row.get("layer"), int)
            }
        )
        # Network voltage layer 1 corresponds to physical W1 and layer 2 to W2;
        # physical load reports use zero-based weight-layer indices.
        for physical_layer in layers or (0, 1):
            mapping_rows = [
                row for row in design["mapping_loads"] if row.get("layer") == physical_layer
            ]
            persistent_rows = [
                row for row in design["persistent_loads"] if row.get("layer") == physical_layer
            ]
            ideal_voltage = [
                row
                for row in design["voltage_rms"]
                if row.get("state") == "ideal_quantized"
                and row.get("layer") == physical_layer + 1
            ]
            persistent_voltage = [
                row
                for row in design["voltage_rms"]
                if row.get("state") == "persistent_pv"
                and row.get("layer") == physical_layer + 1
            ]
            lines.append(
                f"| {design['alpha']:.2f} | {design['spacing_delta_x_multiplier']}δx | "
                f"W{physical_layer + 1} | {_mean_stat(ideal_voltage, 'rms')} | "
                f"{_mean_stat(persistent_voltage, 'rms')} | "
                f"{_mean_stat(mapping_rows, 'baseline_loading_mean')} | "
                f"{_mean_stat(mapping_rows, 'ideal_quantized_loading_mean')} | "
                f"{_mean_stat(persistent_rows, 'persistent_loading_mean')} | "
                f"{_mean_stat(persistent_rows, 'persistent_rms_contrast_over_mean_loading')} |"
            )

    lines.extend(
        [
            "",
            "## DRN-written versus ReLU weight errors",
            "",
            "The ideal columns use the saved `normalized_relu_weight_error`; "
            "persistent columns use weights reconstructed from the programmed "
            "four-conductance contrast. Values are averaged across assignments "
            "and, for P&V, all five endpoint repeats.",
            "",
            "| α | spacing | layer | ideal RMSE | ideal relative L2 | "
            "persistent RMSE | persistent relative L2 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for design in analysis["design_rows"]:
        layers = sorted(
            {row["layer"] for row in design["ideal_weight_errors"]}
            | {row["layer"] for row in design["persistent_weight_errors"]}
        )
        if not layers:
            lines.append(
                f"| {design['alpha']:.2f} | {design['spacing_delta_x_multiplier']}δx | "
                "not recorded | — | — | — | — |"
            )
        for layer in layers:
            ideal = [row for row in design["ideal_weight_errors"] if row["layer"] == layer]
            persistent = [
                row for row in design["persistent_weight_errors"] if row["layer"] == layer
            ]
            lines.append(
                f"| {design['alpha']:.2f} | {design['spacing_delta_x_multiplier']}δx | "
                f"W{layer + 1} | {_mean_stat(ideal, 'ideal_quantized_rmse')} | "
                f"{_mean_stat(ideal, 'ideal_quantized_relative_l2')} | "
                f"{_mean_stat(persistent, 'rmse')} | {_mean_stat(persistent, 'relative_l2')} |"
            )

    comparisons = analysis.get("comparisons", {})
    if comparisons:
        lines.extend(
            [
                "",
                "## Descriptive predecessor comparisons",
                "",
                "These deltas are descriptive, not isolated causal effects: the "
                "support intervention changes RESET commissioning, bounds, "
                "calibration, targets, loading, and P&V trajectories together.",
            ]
        )
        for comparison in comparisons.values():
            lines.extend(
                [
                    "",
                    f"### {comparison['label']}",
                    "",
                    "| α | spacing | Δ continuous | Δ ideal | Δ persistent P&V |",
                    "|---:|---:|---:|---:|---:|",
                ]
            )
            for row in comparison["design_rows"]:
                lines.append(
                    f"| {row['alpha']:.2f} | {row['spacing_delta_x_multiplier']}δx | "
                    f"{row['continuous_delta_percentage_points']:+.2f} pp | "
                    f"{row['ideal_quantized_delta_percentage_points']:+.2f} pp | "
                    f"{row['persistent_pv_delta_percentage_points']:+.2f} pp |"
                )

    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- Model-based AIHWKit OM finite-cohort evidence, not measured "
            "fabricated-device evidence.",
            "- Counterfactual repaired assignments and nominal-bound "
            "Winsorization; this is not IBM's default population sampler.",
            "- One-pulse P&V, cap 128, three assignments and five endpoint "
            "seeds; no HWA/QAT/BPTT, inference read noise, retention, or recovery.",
            "- The affine `G=a+1` scale is analyst-defined because the OM "
            "preset supplies no unique physical conductance calibration.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def analyze(
    result_root: Path,
    *,
    output_dir: Path | None = None,
    comparison_roots: Mapping[str, Path] | None = None,
) -> Mapping[str, Any]:
    """Analyze one complete 27-run Winsorized matrix and write three reports."""

    root = result_root.resolve()
    records = _load_current_records(root)
    groups = _design_groups(records)
    design_rows = [_design_row(design, groups[design]) for design in EXPECTED_DESIGNS]
    assignment_rows = [_assignment_row(record) for record in records]
    winsorization = _winsorization_by_assignment(records)
    baseline_by_assignment = _baseline_projection_by_assignment(records)
    unique_baseline_pairs = sum(
        row["destination_pair_count"] for row in baseline_by_assignment
    )
    unique_baseline_projected = sum(
        row["projected_pair_count"] for row in baseline_by_assignment
    )
    unique_baseline_absolute_sum = sum(
        row["absolute_projection_sum_raw_x"] for row in baseline_by_assignment
    )
    unique_baseline_projection = {
        "destination_pair_count": unique_baseline_pairs,
        "projected_pair_count": unique_baseline_projected,
        "projected_pair_fraction": unique_baseline_projected
        / unique_baseline_pairs,
        "downward_projection_count": sum(
            row["downward_projection_count"] for row in baseline_by_assignment
        ),
        "upward_projection_count": sum(
            row["upward_projection_count"] for row in baseline_by_assignment
        ),
        "absolute_projection_sum_raw_x": unique_baseline_absolute_sum,
        "absolute_projection_mean_all_pairs_raw_x": (
            unique_baseline_absolute_sum / unique_baseline_pairs
        ),
        "absolute_projection_mean_projected_pairs_raw_x": (
            unique_baseline_absolute_sum / unique_baseline_projected
            if unique_baseline_projected
            else 0.0
        ),
        "absolute_projection_maximum_raw_x": max(
            row["absolute_projection_maximum_raw_x"]
            for row in baseline_by_assignment
        ),
    }
    validity = {
        "target_projection_count": sum(
            row["handoff"]["target_projection_count"] for row in records
        ),
        "persistent_projection_count": sum(
            row["handoff"]["persistent_projection_count"] for row in records
        ),
        "below_native_support_count": sum(
            row["handoff"]["below_native_support_count"] for row in records
        ),
        "above_native_support_count": sum(
            row["handoff"]["above_native_support_count"] for row in records
        ),
        "apparent_endpoint_applied_count": sum(
            row["handoff"]["apparent_endpoint_applied_count"] for row in records
        ),
        "nonfinite_programming_count": sum(
            row["program_verify"]["counts"]["nonfinite"] for row in assignment_rows
        ),
        "all_gates_passed": True,
    }
    best_ideal = max(
        design_rows,
        key=lambda row: (
            row["ideal_quantized_accuracy"],
            -row["alpha"],
            -row["spacing_delta_x_multiplier"],
        ),
    )
    best_persistent = max(
        design_rows,
        key=lambda row: (
            row["persistent_pv_accuracy"],
            -row["alpha"],
            -row["spacing_delta_x_multiplier"],
        ),
    )
    comparison_output = {}
    for key, path in (comparison_roots or {}).items():
        comparison_output[key] = _comparison(key, path.resolve(), design_rows)
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "complete_exploratory_noncanonical_analysis",
        "claim_boundary": (
            "Analyst counterfactual nominal-bound Winsorization of frozen "
            "AIHWKit OM identities before RESET commissioning and P&V; not "
            "the default IBM OM population model or fabricated-device evidence."
        ),
        "result_root": str(root),
        "coverage": {
            "complete_configurations": len(records),
            "expected_configurations": len(EXPECTED_KEYS),
            "designs": len(design_rows),
            "heldout_assignments_per_design": len(HELDOUT_ASSIGNMENT_SEEDS),
            "persistent_repeats_per_assignment": EXPECTED_REPEATS,
            "persistent_deployments": len(records) * EXPECTED_REPEATS,
        },
        "intervention": {
            "scientific_name": "nominal_bound_winsorization",
            "legacy_filename_token": "truncated_nominal",
            "operation_order": (
                "sample frozen identity; Winsorize bounds to a in [-1,1]; "
                "repeat RESET commissioning; map/P&V; hand off persistent G=a+1"
            ),
            "conductance_formula": "G=a_winsorized+1=2*x",
            "conductance_interval": [0.0, 2.0],
            "identity_resampled": False,
            "default_ibm_model": False,
        },
        "validity": validity,
        "winsorization_by_assignment": winsorization,
        "requested_baseline_projection_by_assignment": baseline_by_assignment,
        "requested_baseline_projection_unique_assignments": (
            unique_baseline_projection
        ),
        "design_rows": design_rows,
        "assignment_rows": assignment_rows,
        "best_ideal_quantized_design": {
            key: best_ideal[key]
            for key in (
                "alpha",
                "spacing_delta_x_multiplier",
                "ideal_quantized_accuracy",
            )
        },
        "best_persistent_pv_design": {
            key: best_persistent[key]
            for key in (
                "alpha",
                "spacing_delta_x_multiplier",
                "persistent_pv_accuracy",
            )
        },
        "comparisons": comparison_output,
        "limitations": [
            "model_based_aihwkit_preset_not_measured_fabricated_array",
            "counterfactual_repaired_population",
            "nominal_bound_winsorization_not_default_ibm_sampling",
            "analyst_defined_affine_conductance_scale",
            "no_hwa_qat_bptt_read_noise_retention_or_recovery",
        ],
    }
    destination = (output_dir or (root / "analysis")).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / "exploratory_summary.json", analysis)
    _atomic_text(destination / "accuracy_by_assignment.csv", _csv_text(assignment_rows))
    _atomic_text(destination / "post_run_analysis.md", _markdown(analysis))
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--clipped-root", type=Path)
    parser.add_argument("--no-clip-root", type=Path)
    parser.add_argument(
        "--no-default-comparisons",
        action="store_true",
        help="Do not auto-compare predecessor roots when they exist.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    comparisons: dict[str, Path] = {}
    if not args.no_default_comparisons:
        if DEFAULT_CLIPPED_ROOT.is_dir():
            comparisons["legacy_clipped_predecessor"] = DEFAULT_CLIPPED_ROOT
        if DEFAULT_NO_CLIP_ROOT.is_dir():
            comparisons["affine_no_clip_predecessor"] = DEFAULT_NO_CLIP_ROOT
    if args.clipped_root is not None:
        comparisons["legacy_clipped_predecessor"] = args.clipped_root
    if args.no_clip_root is not None:
        comparisons["affine_no_clip_predecessor"] = args.no_clip_root
    result = analyze(
        args.result_root,
        output_dir=args.output_dir,
        comparison_roots=comparisons,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "coverage": result["coverage"],
                "best_ideal_quantized_design": result[
                    "best_ideal_quantized_design"
                ],
                "best_persistent_pv_design": result[
                    "best_persistent_pv_design"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SCHEMA",
    "ANALYSIS_SCHEMA_VERSION",
    "WinsorizedNominalAnalysisError",
    "analyze",
    "main",
]
