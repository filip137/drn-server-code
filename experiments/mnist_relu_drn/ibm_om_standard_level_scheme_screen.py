"""IBM OM ideal standard-level deployment-scheme screen.

This screen is the scheme-refitted successor to the historical deterministic
pulse-codebook screen.  It keeps the logical source and DRN solver in FP32,
but maps every active OM state to an analyst-declared uniform grid whose
adjacent targets are four nominal device increments apart.  No stochastic
write term, pulse controller, HWA modifier, optimizer update, or endpoint
sampler is used.

The no-reference arms anchor their grid at the sampled RESET/lower state.  The
fixed-reference arms retain each sampled reference ``r`` exactly and center
the active grid on it; if ``a=r`` is outside the active bounds, only ``a`` is
moved to the nearest in-bounds integer grid level.  Four-device signed weights
still use diagonal/off-diagonal placement, while eight-device edges retain
separate active and reference tensors.  Consequently transfer and loading are
always evaluated from the physical difference and sum, respectively.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gc
from itertools import product
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.definitions import resolve_experiment_config
from experiments.mnist_relu_drn.components import (
    apply_targets,
    build_student_stack,
    collect_calibration,
    fit_positive_logit_gain,
    signed_dual_rail_lift,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _binding_slices,
    _evaluate_detailed,
    _expanded_quad_minimum,
    _float_summary,
    _integer_summary,
    _native_to_unit,
    _population_for,
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_ideal_mapping_scheme_screen import (
    Scheme,
    _finite_fraction,
    _normalized,
    _ordered_labels,
    _quad_contrast,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders
from experiments.schema import RunMode
from training.ibm_reram_hwa import IbmReramArrayPopulation


SCHEMA = "ebl.mnist_relu_drn.ibm_om_standard_level_scheme_screen"
SCHEMA_VERSION = 1
CONTRACT_SCHEMA_VERSION = 1
_LAYOUTS = ("halves", "paired")


@dataclass(frozen=True)
class StandardLevelScreenContract:
    screen_id: str
    evidence_class: str
    preset: str
    required_aihwkit_version: str
    corruption_policy: str
    teacher_sha256: str
    model_config_sha256: str
    development_assignment_seed: int
    heldout_assignment_seeds: tuple[int, ...]
    spacing_delta_multiples: int
    scale_fractions: tuple[float, ...]
    sample_limit: int | None
    continuous_envelope_control: bool
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class StandardLevelGrid:
    """Per-cell uniform active-state grid in the repository unit coordinate."""

    origin: torch.Tensor
    active_zero: torch.Tensor
    minimum_indices: torch.Tensor
    zero_indices: torch.Tensor
    maximum_indices: torch.Tensor
    maximum_positive_steps: torch.Tensor
    maximum_negative_steps: torch.Tensor
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor
    spacing: float
    delta: float
    spacing_delta_multiples: int
    use_fixed_reference: bool
    report: Mapping[str, Any]

    @property
    def upper_values(self) -> torch.Tensor:
        return self.active_zero + self.maximum_positive_steps.to(
            self.active_zero.dtype
        ) * self.spacing


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the identity-aware IBM OM four/eight-device scheme screen "
            "with RESET-anchored or intrinsic-r-centered four-delta levels."
        )
    )
    parser.add_argument("--screen-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aihwkit-python", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"Expected {label} keys to be exactly {sorted(expected)!r}. "
            f"Provided missing={sorted(expected - set(value))!r}, "
            f"unknown={sorted(set(value) - expected)!r}."
        )


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"Expected {label} to be an integer >= {minimum}. "
            f"Provided value: {value!r}."
        )
    return value


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Expected a lowercase SHA-256 for {label}.")
    return value


def load_standard_level_contract(path: Path) -> StandardLevelScreenContract:
    source = path.expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Expected a readable strict screen config: {source}.") from error
    if not isinstance(raw, dict):
        raise ValueError("Expected the standard-level screen config to contain one object.")
    _exact_keys(
        raw,
        {
            "schema_version",
            "screen_id",
            "evidence_class",
            "preset",
            "required_aihwkit_version",
            "corruption_policy",
            "source",
            "development_assignment_seed",
            "heldout_assignment_seeds",
            "standard_levels",
            "per_scheme_calibration",
            "continuous_envelope_control",
            "sample_limit",
        },
        label="standard-level screen config",
    )
    if raw["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise ValueError("Expected standard-level screen schema_version to equal 1.")
    for name in ("screen_id", "evidence_class", "preset", "required_aihwkit_version"):
        if not isinstance(raw[name], str) or not raw[name]:
            raise ValueError(f"Expected {name} to be a non-empty string.")
    if raw["evidence_class"] != "model_based_aihwkit_preset":
        raise ValueError("Expected the OM screen evidence class to stay model-based.")
    if raw["preset"] != "reram_array_om" or raw["required_aihwkit_version"] != "1.1.0":
        raise ValueError("Expected the pinned AIHWKit 1.1.0 OM preset.")
    if raw["corruption_policy"] != "counterfactual_repaired":
        raise ValueError("Expected the declared counterfactual-repaired OM population.")

    source_contract = raw["source"]
    if not isinstance(source_contract, dict):
        raise ValueError("Expected source to be an object.")
    _exact_keys(
        source_contract,
        {"teacher_sha256", "model_config_sha256"},
        label="source",
    )
    teacher_sha256 = _sha256(
        source_contract["teacher_sha256"], label="source.teacher_sha256"
    )
    model_config_sha256 = _sha256(
        source_contract["model_config_sha256"],
        label="source.model_config_sha256",
    )

    development_seed = _integer(
        raw["development_assignment_seed"], label="development_assignment_seed"
    )
    heldout_raw = raw["heldout_assignment_seeds"]
    if not isinstance(heldout_raw, list) or not heldout_raw:
        raise ValueError("Expected at least one held-out assignment seed.")
    heldout = tuple(
        _integer(value, label=f"heldout_assignment_seeds[{index}]")
        for index, value in enumerate(heldout_raw)
    )
    if len(set(heldout)) != len(heldout) or development_seed in heldout:
        raise ValueError("Expected distinct development and held-out assignments.")

    levels = raw["standard_levels"]
    if not isinstance(levels, dict):
        raise ValueError("Expected standard_levels to be an object.")
    _exact_keys(
        levels,
        {
            "delta_definition",
            "conductance_coordinate",
            "minimum_spacing_delta_multiples",
            "without_fixed_r_origin",
            "with_fixed_r_origin",
            "with_fixed_r_active_zero",
            "active_programming_direction",
            "range_policy",
            "zero_positive_capacity_group",
            "no_in_bounds_standard_level",
            "logical_rounding",
            "cycle_to_cycle_random_term",
            "apparent_write_noise",
        },
        label="standard_levels",
    )
    expected_levels = {
        "delta_definition": "nominal_dw_min_in_native_a_coordinate",
        "conductance_coordinate": "clip((a+1)/2,0,1)",
        "without_fixed_r_origin": "sampled_min_bound_reset",
        "with_fixed_r_origin": "exact_sampled_reference_r",
        "with_fixed_r_active_zero": (
            "nearest_in_bounds_integer_level_without_changing_r"
        ),
        "active_programming_direction": "nonnegative_offset_with_dual_rail_sign",
        "range_policy": "whole_uniform_levels_within_sampled_active_bounds",
        "zero_positive_capacity_group": "retain_zero_level_only_and_report",
        "no_in_bounds_standard_level": (
            "retain_exact_reference_zero_only_and_report"
        ),
        "logical_rounding": "nearest_integer_half_away_from_zero",
        "cycle_to_cycle_random_term": 0.0,
        "apparent_write_noise": 0.0,
    }
    mismatches = {
        key: {"expected": expected_value, "provided": levels.get(key)}
        for key, expected_value in expected_levels.items()
        if levels.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Expected the standard four-delta level policy: {mismatches!r}.")
    spacing_delta_multiples = _integer(
        levels["minimum_spacing_delta_multiples"],
        label="minimum_spacing_delta_multiples",
        minimum=1,
    )
    if spacing_delta_multiples != 4:
        raise ValueError("Expected adjacent standard targets to be four delta apart.")

    calibration = raw["per_scheme_calibration"]
    if not isinstance(calibration, dict):
        raise ValueError("Expected per_scheme_calibration to be an object.")
    _exact_keys(
        calibration,
        {
            "scale_fractions",
            "selection_domain",
            "selection_metric",
            "logit_gain",
            "heldout_application",
        },
        label="per_scheme_calibration",
    )
    expected_calibration = {
        "selection_domain": "development_assignment_after_standard_level_mapping",
        "selection_metric": "mapped_accuracy_then_calibrated_kl_then_scale_pair",
        "logit_gain": "per_scheme_positive_kl_fit",
        "heldout_application": "freeze_selected_scale_pair_and_gain_per_scheme",
    }
    calibration_mismatches = {
        key: {"expected": expected_value, "provided": calibration.get(key)}
        for key, expected_value in expected_calibration.items()
        if calibration.get(key) != expected_value
    }
    if calibration_mismatches:
        raise ValueError(
            f"Expected the declared per-scheme calibration: {calibration_mismatches!r}."
        )
    fractions_raw = calibration["scale_fractions"]
    if not isinstance(fractions_raw, list) or not fractions_raw:
        raise ValueError("Expected a non-empty per-scheme scale-fraction grid.")
    fractions = tuple(
        _finite_fraction(
            value,
            name=f"per_scheme_calibration.scale_fractions[{index}]",
            include_zero=False,
            include_one=True,
        )
        for index, value in enumerate(fractions_raw)
    )
    if len(set(fractions)) != len(fractions):
        raise ValueError("Expected unique scale fractions.")
    if raw["continuous_envelope_control"] is not True:
        raise ValueError("Expected the continuous-envelope diagnostic control.")
    sample_limit = raw["sample_limit"]
    if sample_limit is not None:
        sample_limit = _integer(sample_limit, label="sample_limit", minimum=1)

    return StandardLevelScreenContract(
        screen_id=raw["screen_id"],
        evidence_class=raw["evidence_class"],
        preset=raw["preset"],
        required_aihwkit_version=raw["required_aihwkit_version"],
        corruption_policy=raw["corruption_policy"],
        teacher_sha256=teacher_sha256,
        model_config_sha256=model_config_sha256,
        development_assignment_seed=development_seed,
        heldout_assignment_seeds=heldout,
        spacing_delta_multiples=spacing_delta_multiples,
        scale_fractions=tuple(float(value) for value in fractions),
        sample_limit=sample_limit,
        continuous_envelope_control=True,
        raw=raw,
    )


def build_standard_level_grid(
    population: IbmReramArrayPopulation,
    *,
    use_fixed_reference: bool,
    spacing_delta_multiples: int = 4,
) -> StandardLevelGrid:
    """Build exact uniform levels while keeping a sampled ``r`` unchanged.

    ``dw_min`` is expressed in native ``a`` coordinates.  The repository maps
    native state to ``x=(a+1)/2``, so one nominal increment is
    ``delta_x=dw_min/2``.  The returned levels are integer multiples of
    ``spacing_delta_multiples * delta_x``.
    """

    if spacing_delta_multiples < 1:
        raise ValueError("Expected a positive standard-level spacing multiplier.")
    delta = float(population.nominal_dw_min) / 2.0
    spacing = float(spacing_delta_multiples) * delta
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Expected a finite positive nominal OM level spacing.")

    lower = _native_to_unit(
        population.min_bound.detach().to(device="cpu", dtype=torch.float64)
    )
    upper = _native_to_unit(
        population.max_bound.detach().to(device="cpu", dtype=torch.float64)
    )
    if bool(torch.any(upper < lower)):
        raise ValueError("Expected mapped OM upper bounds not below lower bounds.")

    if use_fixed_reference:
        origin = _native_to_unit(
            population.reference.detach().to(device="cpu", dtype=torch.float64)
        )
        minimum_indices = torch.ceil((lower - origin) / spacing - 1e-12).to(
            torch.int64
        )
        maximum_indices = torch.floor((upper - origin) / spacing + 1e-12).to(
            torch.int64
        )
        no_level = minimum_indices > maximum_indices
        # A vanishingly small tail of sampled identities has a narrow active
        # interval that falls entirely between adjacent four-delta levels.
        # Keep r exact and expose that identity as zero-only; never move r or
        # invent a shorter terminal interval.
        minimum_indices = torch.where(
            no_level, torch.zeros_like(minimum_indices), minimum_indices
        )
        maximum_indices = torch.where(
            no_level, torch.zeros_like(maximum_indices), maximum_indices
        )
        zeros = torch.zeros_like(minimum_indices)
        zero_indices = torch.minimum(
            torch.maximum(zeros, minimum_indices), maximum_indices
        )
        active_zero = origin + zero_indices.to(torch.float64) * spacing
        outside = (population.reference < population.min_bound) | (
            population.reference > population.max_bound
        )
        clipped_reference = (population.reference < -1.0) | (
            population.reference > 1.0
        )
        policy = "exact_sampled_reference_center_with_active_a_projection_only"
    else:
        origin = lower.clone()
        minimum_indices = torch.zeros(population.size, dtype=torch.int64)
        zero_indices = torch.zeros(population.size, dtype=torch.int64)
        maximum_indices = torch.floor((upper - origin) / spacing + 1e-12).to(
            torch.int64
        )
        active_zero = origin.clone()
        outside = torch.zeros(population.size, dtype=torch.bool)
        clipped_reference = torch.zeros(population.size, dtype=torch.bool)
        no_level = torch.zeros(population.size, dtype=torch.bool)
        policy = "sampled_min_bound_reset_origin"

    maximum_positive = maximum_indices - zero_indices
    maximum_negative = zero_indices - minimum_indices
    if bool(torch.any(maximum_positive < 0)) or bool(torch.any(maximum_negative < 0)):
        raise RuntimeError("Expected nonnegative standard-level capacities.")
    active_zero_outside = (active_zero < lower - 1e-10) | (
        active_zero > upper + 1e-10
    )
    permitted_outside = no_level if use_fixed_reference else torch.zeros_like(
        active_zero_outside
    )
    if bool(torch.any(active_zero_outside & ~permitted_outside)):
        raise RuntimeError("Expected every active zero code to remain within bounds.")
    active_zero = active_zero.to(torch.float32)
    origin = origin.to(torch.float32)
    lower = lower.to(torch.float32)
    upper = upper.to(torch.float32)
    residual = active_zero - origin
    report = {
        "policy": policy,
        "use_fixed_reference": use_fixed_reference,
        "native_delta": float(population.nominal_dw_min),
        "unit_coordinate_delta": delta,
        "spacing_delta_multiples": spacing_delta_multiples,
        "unit_coordinate_spacing": spacing,
        "native_coordinate_spacing": 2.0 * spacing,
        "origin": _float_summary(origin),
        "active_zero": _float_summary(active_zero),
        "active_zero_minus_origin": _float_summary(residual),
        "minimum_integer_index": _integer_summary(minimum_indices),
        "active_zero_integer_index": _integer_summary(zero_indices),
        "maximum_integer_index": _integer_summary(maximum_indices),
        "available_positive_steps": _integer_summary(maximum_positive),
        "available_negative_steps": _integer_summary(maximum_negative),
        "available_nonnegative_level_count": _integer_summary(
            maximum_positive + 1
        ),
        "available_centered_level_count": _integer_summary(
            maximum_indices - minimum_indices + 1
        ),
        "sampled_reference_outside_active_bounds_count": int(outside.sum().item()),
        "sampled_reference_outside_active_bounds_fraction": float(
            outside.to(torch.float64).mean().item()
        ),
        "sampled_reference_clipped_by_global_coordinate_count": int(
            clipped_reference.sum().item()
        ),
        "no_whole_in_bounds_standard_level_count": int(no_level.sum().item()),
        "no_whole_in_bounds_standard_level_fraction": float(
            no_level.to(torch.float64).mean().item()
        ),
        "active_zero_outside_active_bounds_count": int(
            active_zero_outside.sum().item()
        ),
        "hashes": {
            "origin": _tensor_sha256(origin),
            "active_zero": _tensor_sha256(active_zero),
            "minimum_integer_index": _tensor_sha256(minimum_indices),
            "active_zero_integer_index": _tensor_sha256(zero_indices),
            "maximum_integer_index": _tensor_sha256(maximum_indices),
            "available_positive_steps": _tensor_sha256(maximum_positive),
        },
    }
    return StandardLevelGrid(
        origin=origin,
        active_zero=active_zero,
        minimum_indices=minimum_indices,
        zero_indices=zero_indices,
        maximum_indices=maximum_indices,
        maximum_positive_steps=maximum_positive,
        maximum_negative_steps=maximum_negative,
        lower_bounds=lower,
        upper_bounds=upper,
        spacing=spacing,
        delta=delta,
        spacing_delta_multiples=spacing_delta_multiples,
        use_fixed_reference=use_fixed_reference,
        report=report,
    )


def _round_nonnegative_half_away_from_zero(value: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(value).all()) or bool(torch.any(value < 0.0)):
        raise ValueError("Expected finite nonnegative logical level indices.")
    return torch.floor(value + 0.5).to(torch.int64)


def _layer_report(
    *,
    layer_index: int,
    layout: str,
    logical_weight: torch.Tensor,
    scale_fraction: float,
    grid: StandardLevelGrid,
    binding_slice: slice,
    common_steps: torch.Tensor,
    continuous_steps: torch.Tensor,
    selected_steps: torch.Tensor,
    active_zero: torch.Tensor,
    reference: torch.Tensor,
    continuous_active: torch.Tensor,
    realized_active: torch.Tensor,
    difference: torch.Tensor,
    continuous_difference: torch.Tensor,
    baseline_difference: torch.Tensor,
    loading: torch.Tensor,
    conductance_min: float,
    conductance_max: float,
) -> dict[str, Any]:
    span = conductance_max - conductance_min
    normalized_difference = difference / span
    normalized_loading = loading / span
    contrast = _quad_contrast(normalized_difference, layout=layout) / 2.0
    continuous_contrast = _quad_contrast(
        continuous_difference / span, layout=layout
    ) / 2.0
    baseline_contrast = _quad_contrast(
        baseline_difference / span, layout=layout
    ) / 2.0
    requested_sign = torch.sign(_normalized(logical_weight).cpu())
    nonzero = requested_sign != 0
    sign_flip = nonzero & (requested_sign != torch.sign(contrast))
    continuous_sign_flip = nonzero & (
        requested_sign != torch.sign(continuous_contrast)
    )
    difference_rms = float(normalized_difference.double().square().mean().sqrt().item())
    loading_mean = float(normalized_loading.double().mean().item())
    active_error = realized_active - continuous_active
    maximum_positive = grid.maximum_positive_steps[binding_slice].reshape(
        active_zero.shape
    )
    lower_bound = grid.lower_bounds[binding_slice].reshape(active_zero.shape)
    upper_bound = grid.upper_bounds[binding_slice].reshape(active_zero.shape)
    active_outside = (realized_active < lower_bound - 1e-6) | (
        realized_active > upper_bound + 1e-6
    )
    zero_capacity = common_steps == 0
    return {
        "layer": layer_index,
        "layout": layout,
        "scale_fraction": scale_fraction,
        "standard_level_spacing": grid.spacing,
        "configured_conductance_standard_level_spacing": grid.spacing * span,
        "standard_level_spacing_delta_multiples": grid.spacing_delta_multiples,
        "active_origin": _float_summary(reference),
        "active_zero": _float_summary(active_zero),
        "active_zero_minus_origin": _float_summary(active_zero - reference),
        "per_cell_positive_step_capacity": _integer_summary(maximum_positive),
        "per_quad_common_positive_step_capacity": _integer_summary(common_steps),
        "zero_positive_capacity_quad_count": int(zero_capacity.sum().item()),
        "zero_positive_capacity_quad_fraction": float(
            zero_capacity.to(torch.float64).mean().item()
        ),
        "per_quad_common_signed_logical_level_count": _integer_summary(
            2 * common_steps + 1
        ),
        "continuous_level_index": _float_summary(continuous_steps),
        "selected_level_index": _integer_summary(selected_steps),
        "continuous_active_target": _float_summary(continuous_active),
        "realized_active_target": _float_summary(realized_active),
        "active_quantization_error": _float_summary(active_error),
        "active_target_outside_bounds_count": int(active_outside.sum().item()),
        "active_target_outside_bounds_fraction": float(
            active_outside.to(torch.float64).mean().item()
        ),
        "logical_contrast": _float_summary(contrast),
        "continuous_logical_contrast": _float_summary(continuous_contrast),
        "baseline_logical_contrast": _float_summary(baseline_contrast),
        "nonzero_logical_weight_count": int(nonzero.sum().item()),
        "logical_sign_flip_count": int(sign_flip.sum().item()),
        "logical_sign_flip_fraction_nonzero": (
            float(sign_flip.sum().item()) / int(nonzero.sum().item())
            if bool(torch.any(nonzero))
            else 0.0
        ),
        "continuous_logical_sign_flip_count": int(continuous_sign_flip.sum().item()),
        "continuous_logical_sign_flip_fraction_nonzero": (
            float(continuous_sign_flip.sum().item()) / int(nonzero.sum().item())
            if bool(torch.any(nonzero))
            else 0.0
        ),
        "difference_rms": difference_rms,
        "mean_edge_denominator_loading": loading_mean,
        "difference_rms_over_mean_loading": (
            difference_rms / loading_mean if loading_mean else None
        ),
        "mean_source_row_denominator_loading": float(
            normalized_loading.double().sum(dim=1).mean().item()
        ),
        "mean_destination_column_denominator_loading": float(
            normalized_loading.double().sum(dim=0).mean().item()
        ),
        "normalized_total_conductance_proxy": float(
            normalized_loading.double().sum().item()
        ),
        "hashes": {
            "common_steps": _tensor_sha256(common_steps),
            "continuous_level_index": _tensor_sha256(continuous_steps),
            "selected_level_index": _tensor_sha256(selected_steps),
            "continuous_active_target": _tensor_sha256(continuous_active),
            "realized_active_target": _tensor_sha256(realized_active),
            "baseline_difference": _tensor_sha256(baseline_difference),
            "difference": _tensor_sha256(difference),
            "loading": _tensor_sha256(loading),
        },
    }


def build_standard_scheme_targets(
    logical_weights: Sequence[torch.Tensor],
    population: IbmReramArrayPopulation,
    grid: StandardLevelGrid,
    *,
    scheme: Scheme,
    scale_fractions: tuple[float, float],
    conductance_min: float,
    conductance_max: float,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], dict[str, Any]]:
    """Return uniform-level and matched continuous targets plus evidence."""

    if len(logical_weights) != 2 or len(scale_fractions) != 2:
        raise ValueError("Expected exactly two logical matrices and scale fractions.")
    if scheme.device_count_per_logical_weight not in {4, 8}:
        raise ValueError("Expected a four- or eight-device scheme.")
    if grid.use_fixed_reference != scheme.use_fixed_reference:
        raise ValueError("Expected the standard grid reference policy to match the scheme.")
    lower = float(conductance_min)
    upper = float(conductance_max)
    if lower < 0.0 or upper <= lower or not all(math.isfinite(v) for v in (lower, upper)):
        raise ValueError("Expected finite nonnegative increasing conductance bounds.")
    expected_bindings = 2 if scheme.device_count_per_logical_weight == 4 else 4
    if len(population.binding_keys) != expected_bindings:
        raise ValueError("Expected population bindings to match the physical topology.")

    slices = _binding_slices(population)
    span = upper - lower
    exact_targets: list[torch.Tensor] = []
    continuous_targets: list[torch.Tensor] = []
    layer_reports = []
    for layer_index, (logical, fraction, layout) in enumerate(
        zip(logical_weights, scale_fractions, _LAYOUTS)
    ):
        scale_fraction = _finite_fraction(
            fraction,
            name=f"scale_fractions[{layer_index}]",
            include_zero=False,
            include_one=True,
        )
        lifted = signed_dual_rail_lift(
            _normalized(logical).cpu(), target_layout=layout
        )
        if scheme.device_count_per_logical_weight == 4:
            binding_index = layer_index
        else:
            binding_index = 2 * layer_index
        key = population.binding_keys[binding_index]
        binding_slice = slices[key]
        shape = population.binding_shapes[binding_index]
        active_zero = grid.active_zero[binding_slice].reshape(shape)
        origin = grid.origin[binding_slice].reshape(shape)
        positive_steps = grid.maximum_positive_steps[binding_slice].reshape(shape)
        common_steps = _expanded_quad_minimum(positive_steps, layout=layout)
        continuous_steps = (
            scale_fraction * common_steps.to(torch.float32) * lifted
        )
        selected_steps = _round_nonnegative_half_away_from_zero(continuous_steps)
        selected_steps = torch.minimum(selected_steps, common_steps)
        realized_active = active_zero + selected_steps.to(torch.float32) * grid.spacing
        continuous_active = active_zero + continuous_steps * grid.spacing

        if scheme.device_count_per_logical_weight == 4:
            if scheme.use_fixed_reference:
                exact_unit = torch.where(selected_steps > 0, realized_active, origin)
                continuous_unit = torch.where(lifted > 0.0, continuous_active, origin)
                baseline_unit = origin
            else:
                exact_unit = realized_active
                continuous_unit = continuous_active
                baseline_unit = active_zero
            exact_physical = lower + span * exact_unit
            continuous_physical = lower + span * continuous_unit
            exact_targets.append(exact_physical)
            continuous_targets.append(continuous_physical)
            difference = exact_physical
            continuous_difference = continuous_physical
            baseline_difference = lower + span * baseline_unit
            loading = exact_physical
            reference_for_report = origin if scheme.use_fixed_reference else active_zero
        else:
            reference_index = binding_index + 1
            reference_key = population.binding_keys[reference_index]
            reference_slice = slices[reference_key]
            reference_shape = population.binding_shapes[reference_index]
            if reference_shape != shape:
                raise ValueError("Expected matched active/reference branch shapes.")
            fixed_reference = (
                grid.origin[reference_slice]
                if scheme.use_fixed_reference
                else grid.active_zero[reference_slice]
            ).reshape(reference_shape)
            exact_plus = lower + span * realized_active
            continuous_plus = lower + span * continuous_active
            exact_minus = lower + span * fixed_reference
            exact_targets.extend((exact_plus, exact_minus))
            continuous_targets.extend((continuous_plus, exact_minus.clone()))
            difference = exact_plus - exact_minus
            continuous_difference = continuous_plus - exact_minus
            baseline_difference = (lower + span * active_zero) - exact_minus
            loading = exact_plus + exact_minus
            reference_for_report = origin if scheme.use_fixed_reference else active_zero

        layer_reports.append(
            _layer_report(
                layer_index=layer_index,
                layout=layout,
                logical_weight=logical,
                scale_fraction=scale_fraction,
                grid=grid,
                binding_slice=binding_slice,
                common_steps=common_steps,
                continuous_steps=continuous_steps,
                selected_steps=selected_steps,
                active_zero=active_zero,
                reference=reference_for_report,
                continuous_active=continuous_active,
                realized_active=realized_active,
                difference=difference,
                continuous_difference=continuous_difference,
                baseline_difference=baseline_difference,
                loading=loading,
                conductance_min=lower,
                conductance_max=upper,
            )
        )

    return tuple(exact_targets), tuple(continuous_targets), {
        "scheme": scheme.name,
        "encoding": scheme.encoding,
        "device_count_per_logical_weight": scheme.device_count_per_logical_weight,
        "use_fixed_reference": scheme.use_fixed_reference,
        "level_origin_policy": grid.report["policy"],
        "standard_level_spacing": grid.spacing,
        "standard_level_spacing_delta_multiples": grid.spacing_delta_multiples,
        "scale_fractions": [float(value) for value in scale_fractions],
        "layers": layer_reports,
        "hashes": {
            "standard_level_targets": [
                _tensor_sha256(value) for value in exact_targets
            ],
            "continuous_envelope_targets": [
                _tensor_sha256(value) for value in continuous_targets
            ],
        },
    }


def _schemes() -> tuple[Scheme, ...]:
    return (
        Scheme("four_without_fixed_r", 4, False, 0.0),
        Scheme("four_with_fixed_r", 4, True, 0.0),
        Scheme("eight_without_fixed_r", 8, False, 0.0),
        Scheme("eight_with_fixed_r", 8, True, 0.0),
    )


def _calibration_at_gain(
    raw_scores: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    gain: float,
    gain_source: str,
    gain_was_fit_on_this_scheme: bool,
) -> dict[str, Any]:
    scores = raw_scores.detach().to(torch.float64)
    teacher_scores = teacher_logits.detach().to(torch.float64)
    teacher_log_prob = F.log_softmax(teacher_scores, dim=1)
    teacher_prob = teacher_log_prob.exp()

    def kl(applied_gain: float) -> float:
        student_log_prob = F.log_softmax(scores * applied_gain, dim=1)
        return float(
            (teacher_prob * (teacher_log_prob - student_log_prob))
            .sum(dim=1)
            .mean()
            .item()
        )

    student_prediction = scores.cpu().argmax(dim=1)
    teacher_prediction = teacher_scores.cpu().argmax(dim=1)
    calibration_labels = labels.detach().cpu()
    if student_prediction.shape != calibration_labels.shape:
        raise RuntimeError("Expected calibration predictions and labels to match.")
    return {
        "gain": float(gain),
        "gain_source": gain_source,
        "gain_was_fit_on_this_scheme": gain_was_fit_on_this_scheme,
        "raw_kl": kl(1.0),
        "calibrated_kl": kl(float(gain)),
        "score_rms": float(scores.square().mean().sqrt().item()),
        "calibrated_score_rms": float(
            (scores * float(gain)).square().mean().sqrt().item()
        ),
        "teacher_logit_rms": float(teacher_scores.square().mean().sqrt().item()),
        "student_accuracy": float(
            student_prediction.eq(calibration_labels).double().mean().item()
        ),
        "teacher_accuracy": float(
            teacher_prediction.eq(calibration_labels).double().mean().item()
        ),
        "teacher_agreement": float(
            student_prediction.eq(teacher_prediction).double().mean().item()
        ),
    }


def _fit_scheme_calibration(
    stack,
    teacher,
    loader: Iterable,
    labels: torch.Tensor,
    *,
    gain_min: float,
    gain_max: float,
    gain_steps: int,
) -> dict[str, Any]:
    raw_scores, teacher_logits = collect_calibration(stack, teacher, loader)
    fitted = fit_positive_logit_gain(
        raw_scores,
        teacher_logits,
        gain_min=gain_min,
        gain_max=gain_max,
        steps=gain_steps,
    )
    report = _calibration_at_gain(
        raw_scores,
        teacher_logits,
        labels,
        gain=float(fitted["gain"]),
        gain_source="per_scheme_standard_level_development_fit",
        gain_was_fit_on_this_scheme=True,
    )
    report.update(
        {
            key: value
            for key, value in fitted.items()
            if key not in {"gain", "raw_kl", "calibrated_kl"}
        }
    )
    return report


def _diagnose_at_fixed_gain(
    stack,
    teacher,
    loader: Iterable,
    labels: torch.Tensor,
    *,
    gain: float,
) -> dict[str, Any]:
    raw_scores, teacher_logits = collect_calibration(stack, teacher, loader)
    return _calibration_at_gain(
        raw_scores,
        teacher_logits,
        labels,
        gain=gain,
        gain_source="frozen_selected_standard_level_scheme",
        gain_was_fit_on_this_scheme=False,
    )


def run_screen(
    *,
    screen_config_path: Path,
    model_config_path: Path,
    teacher_weights_path: Path,
    output_dir: Path,
    aihwkit_python: Path,
    device: str,
    sample_limit: int | None,
) -> dict[str, Any]:
    screen_config_path = screen_config_path.expanduser().resolve()
    model_config_path = model_config_path.expanduser().resolve()
    teacher_weights_path = teacher_weights_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    aihwkit_python = aihwkit_python.expanduser().resolve()
    contract = load_standard_level_contract(screen_config_path)
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("Expected --sample-limit to be positive or omitted.")
    effective_sample_limit = contract.sample_limit if sample_limit is None else sample_limit
    for path in (model_config_path, teacher_weights_path, aihwkit_python):
        if not path.is_file():
            raise FileNotFoundError(f"Expected required input file: {path}.")
    if sha256_file(model_config_path) != contract.model_config_sha256:
        raise ValueError("Model config SHA-256 does not match the standard-level contract.")
    if sha256_file(teacher_weights_path) != contract.teacher_sha256:
        raise ValueError("Teacher SHA-256 does not match the standard-level contract.")

    definition, source_spec = resolve_experiment_config(model_config_path, RunMode.TRAIN)
    if definition.experiment_id != "mnist_relu_drn_kd.v1":
        raise ValueError("Expected the MNIST ReLU-to-DRN composition root.")
    base_spec = replace(source_spec, runtime=replace(source_spec.runtime, device=device))
    loaders = build_mnist_loaders(
        source_spec.data,
        data_seed=source_spec.runtime.data_seed,
        calibration_examples=source_spec.mapping.calibration_examples,
        calibration_batch_size=source_spec.mapping.calibration_batch_size,
    )
    teacher, teacher_metadata = _load_teacher(
        teacher_weights_path, device=torch.device(device), spec=base_spec
    )
    logical_weights = tuple(parameter.detach().cpu() for parameter in teacher.parameters())
    if len(logical_weights) != 2:
        raise ValueError("Expected the bias-free teacher to expose two matrices.")
    calibration_labels = _ordered_labels(loaders.calibration)
    fraction_pairs = tuple(product(contract.scale_fractions, repeat=2))
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_dir / "screen_contract.json",
        {
            "screen_config": str(screen_config_path),
            "screen_config_sha256": sha256_file(screen_config_path),
            "model_config": str(model_config_path),
            "model_config_sha256": sha256_file(model_config_path),
            "teacher_weights": str(teacher_weights_path),
            "teacher_weights_sha256": sha256_file(teacher_weights_path),
            "aihwkit_python": str(aihwkit_python),
            "device": device,
            "sample_limit": effective_sample_limit,
            "contract": contract.raw,
        },
    )

    schemes = _schemes()
    development: dict[str, dict[str, Any]] = {}
    development_populations = []
    for topology in (4, 8):
        topology_spec = replace(
            base_spec,
            model=replace(
                base_spec.model,
                encoding="single" if topology == 4 else "differential",
            ),
        )
        torch.manual_seed(source_spec.runtime.seed)
        stack = build_student_stack(topology_spec, enable_measured=False)
        population, population_report = _population_for(
            stack=stack,
            topology=topology,
            assignment_seed=contract.development_assignment_seed,
            contract=contract,
            aihwkit_python=aihwkit_python,
            output_dir=output_dir,
        )
        population_report["standard_level_grids"] = {}
        for scheme in (
            value
            for value in schemes
            if value.device_count_per_logical_weight == topology
        ):
            grid = build_standard_level_grid(
                population,
                use_fixed_reference=scheme.use_fixed_reference,
                spacing_delta_multiples=contract.spacing_delta_multiples,
            )
            population_report["standard_level_grids"][scheme.name] = grid.report
            candidates = []
            selected_key = None
            selected_exact = None
            selected_continuous = None
            selected_mapping = None
            for pair in fraction_pairs:
                exact, continuous, mapping = build_standard_scheme_targets(
                    logical_weights,
                    population,
                    grid,
                    scheme=scheme,
                    scale_fractions=(float(pair[0]), float(pair[1])),
                    conductance_min=topology_spec.model.conductance_min,
                    conductance_max=topology_spec.model.conductance_max,
                )
                apply_targets(stack.bundle.catalog, exact)
                calibration = _fit_scheme_calibration(
                    stack,
                    teacher,
                    loaders.calibration,
                    calibration_labels,
                    gain_min=topology_spec.mapping.logit_gain_min,
                    gain_max=topology_spec.mapping.logit_gain_max,
                    gain_steps=topology_spec.mapping.logit_gain_steps,
                )
                candidate = {
                    "scale_fractions": [float(pair[0]), float(pair[1])],
                    "calibration": calibration,
                    "target_hashes": mapping["hashes"],
                }
                candidates.append(candidate)
                key = (
                    -calibration["student_accuracy"],
                    calibration["calibrated_kl"],
                    pair,
                )
                if selected_key is None or key < selected_key:
                    selected_key = key
                    selected_exact = tuple(value.detach().clone() for value in exact)
                    selected_continuous = tuple(
                        value.detach().clone() for value in continuous
                    )
                    selected_mapping = mapping
            if (
                selected_exact is None
                or selected_continuous is None
                or selected_mapping is None
            ):
                raise RuntimeError("Expected at least one per-scheme calibration candidate.")
            selected_index = min(
                range(len(candidates)),
                key=lambda index: (
                    -candidates[index]["calibration"]["student_accuracy"],
                    candidates[index]["calibration"]["calibrated_kl"],
                    tuple(candidates[index]["scale_fractions"]),
                ),
            )
            selected = candidates[selected_index]
            gain = float(selected["calibration"]["gain"])
            apply_targets(stack.bundle.catalog, selected_continuous)
            continuous_calibration = _diagnose_at_fixed_gain(
                stack,
                teacher,
                loaders.calibration,
                calibration_labels,
                gain=gain,
            )
            development[scheme.name] = {
                "scheme": scheme.name,
                "topology": topology,
                "selection_domain": (
                    "development_assignment_after_standard_level_mapping"
                ),
                "selection_metric": (
                    "mapped_accuracy_then_calibrated_kl_then_scale_pair"
                ),
                "per_scheme_refit_performed": True,
                "selected_index": selected_index,
                "selected": selected,
                "continuous_envelope_calibration_at_selected_gain": (
                    continuous_calibration
                ),
                "mapping": selected_mapping,
                "candidates": candidates,
            }
            print(
                f"calibrated {scheme.name}: scales="
                f"{selected['scale_fractions']}, gain={gain:.12g}, "
                f"accuracy={100.0 * selected['calibration']['student_accuracy']:.2f}%",
                flush=True,
            )
        development_populations.append(population_report)
        del population, stack
        gc.collect()

    heldout_reports = []
    for assignment_seed in contract.heldout_assignment_seeds:
        for topology in (4, 8):
            topology_spec = replace(
                base_spec,
                model=replace(
                    base_spec.model,
                    encoding="single" if topology == 4 else "differential",
                ),
            )
            torch.manual_seed(source_spec.runtime.seed)
            stack = build_student_stack(topology_spec, enable_measured=False)
            population, population_report = _population_for(
                stack=stack,
                topology=topology,
                assignment_seed=assignment_seed,
                contract=contract,
                aihwkit_python=aihwkit_python,
                output_dir=output_dir,
            )
            population_report["standard_level_grids"] = {}
            arm_reports = []
            for scheme in (
                value
                for value in schemes
                if value.device_count_per_logical_weight == topology
            ):
                grid = build_standard_level_grid(
                    population,
                    use_fixed_reference=scheme.use_fixed_reference,
                    spacing_delta_multiples=contract.spacing_delta_multiples,
                )
                population_report["standard_level_grids"][scheme.name] = grid.report
                selected = development[scheme.name]["selected"]
                pair = tuple(float(value) for value in selected["scale_fractions"])
                gain = float(selected["calibration"]["gain"])
                exact, continuous, mapping = build_standard_scheme_targets(
                    logical_weights,
                    population,
                    grid,
                    scheme=scheme,
                    scale_fractions=(pair[0], pair[1]),
                    conductance_min=topology_spec.model.conductance_min,
                    conductance_max=topology_spec.model.conductance_max,
                )
                apply_targets(stack.bundle.catalog, exact)
                stack.cost.gain = gain
                exact_metrics, exact_predictions = _evaluate_detailed(
                    stack,
                    teacher,
                    loaders.test,
                    sample_limit=effective_sample_limit,
                )
                apply_targets(stack.bundle.catalog, continuous)
                stack.cost.gain = gain
                continuous_metrics, continuous_predictions = _evaluate_detailed(
                    stack,
                    teacher,
                    loaders.test,
                    sample_limit=effective_sample_limit,
                )
                if exact_predictions.shape != continuous_predictions.shape:
                    raise RuntimeError("Expected matched continuous/level predictions.")
                flips = exact_predictions != continuous_predictions
                arm_reports.append(
                    {
                        "scheme": scheme.name,
                        "assignment_seed": assignment_seed,
                        "scale_fractions": list(pair),
                        "fixed_logit_gain": gain,
                        "calibration_source": (
                            "frozen_per_scheme_development_assignment"
                        ),
                        "per_scheme_refit_performed": True,
                        "mapping": mapping,
                        "standard_level_test": exact_metrics,
                        "continuous_envelope_test": continuous_metrics,
                        "continuous_to_standard_prediction_flip_count": int(
                            flips.sum().item()
                        ),
                        "continuous_to_standard_prediction_flip_fraction": float(
                            flips.to(torch.float64).mean().item()
                        ),
                    }
                )
                print(
                    f"heldout {assignment_seed} {scheme.name}: "
                    f"standard={100.0 * exact_metrics['student_accuracy']:.2f}%, "
                    f"continuous="
                    f"{100.0 * continuous_metrics['student_accuracy']:.2f}%",
                    flush=True,
                )
            heldout_reports.append(
                {
                    "assignment_seed": assignment_seed,
                    "topology": topology,
                    "population": population_report,
                    "arms": arm_reports,
                }
            )
            del population, stack
            gc.collect()

    aggregate = {}
    for scheme in schemes:
        reports = [
            arm
            for assignment in heldout_reports
            for arm in assignment["arms"]
            if arm["scheme"] == scheme.name
        ]
        standard_accuracy = torch.tensor(
            [arm["standard_level_test"]["student_accuracy"] for arm in reports],
            dtype=torch.float64,
        )
        continuous_accuracy = torch.tensor(
            [arm["continuous_envelope_test"]["student_accuracy"] for arm in reports],
            dtype=torch.float64,
        )
        aggregate[scheme.name] = {
            "assignments": [arm["assignment_seed"] for arm in reports],
            "standard_level_accuracy": _float_summary(standard_accuracy),
            "continuous_envelope_accuracy": _float_summary(continuous_accuracy),
            "standard_minus_continuous_accuracy": _float_summary(
                standard_accuracy - continuous_accuracy
            ),
            "passes_90_percent_mean_gate": bool(standard_accuracy.mean() >= 0.9),
        }

    comparisons = {
        "fixed_r_minus_no_r": {
            "four_devices": (
                aggregate["four_with_fixed_r"]["standard_level_accuracy"]["mean"]
                - aggregate["four_without_fixed_r"]["standard_level_accuracy"]["mean"]
            ),
            "eight_devices": (
                aggregate["eight_with_fixed_r"]["standard_level_accuracy"]["mean"]
                - aggregate["eight_without_fixed_r"]["standard_level_accuracy"]["mean"]
            ),
        },
        "eight_minus_four": {
            "without_fixed_r": (
                aggregate["eight_without_fixed_r"]["standard_level_accuracy"]["mean"]
                - aggregate["four_without_fixed_r"]["standard_level_accuracy"]["mean"]
            ),
            "with_fixed_r": (
                aggregate["eight_with_fixed_r"]["standard_level_accuracy"]["mean"]
                - aggregate["four_with_fixed_r"]["standard_level_accuracy"]["mean"]
            ),
        },
    }
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "identity_aware_ideal_standard_level_screen_complete",
        "screen_id": contract.screen_id,
        "claim_boundary": (
            "AIHWKit 1.1.0 normalized OM fitted-model control with repaired "
            "identities and analyst-standard uniform levels separated by four "
            "nominal increments. No stochastic write, program-and-verify, HWA, "
            "training, absolute conductance calibration, or fabricated-device claim."
        ),
        "source_precision": "fp32_teacher_and_drn_solver",
        "deployment_precision": "uniform_four_delta_standard_level_grid",
        "screen_config": str(screen_config_path),
        "screen_config_sha256": sha256_file(screen_config_path),
        "model_config": str(model_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "teacher_weights": str(teacher_weights_path),
        "teacher_weights_sha256": sha256_file(teacher_weights_path),
        "teacher_architecture": teacher_metadata.get("architecture"),
        "device": device,
        "sample_limit": effective_sample_limit,
        "development_assignment_seed": contract.development_assignment_seed,
        "heldout_assignment_seeds": list(contract.heldout_assignment_seeds),
        "calibration_policy": {
            "source": "per_scheme_development_assignment",
            "selection_domain": (
                "development_assignment_after_standard_level_mapping"
            ),
            "selection_metric": (
                "mapped_accuracy_then_calibrated_kl_then_scale_pair"
            ),
            "heldout_application": (
                "freeze_selected_scale_pair_and_gain_per_scheme"
            ),
        },
        "scheme_contract": {
            "without_fixed_r_origin": "sampled RESET/lower state",
            "with_fixed_r_origin": "exact intrinsic sampled r",
            "active_level_spacing": "4 * nominal dw_min/2 in x=(a+1)/2",
            "four_fixed_r_positive": "G++=G--=a; G+-=G-+=r",
            "four_fixed_r_negative": "G++=G--=r; G+-=G-+=a",
            "four_edge_transfer_and_loading": "D=G; S=G",
            "eight_edge_transfer": "D=G_a-G_r",
            "eight_edge_loading": "S=G_a+G_r",
        },
        "development_populations": development_populations,
        "development_calibration": development,
        "heldout": heldout_reports,
        "aggregate": aggregate,
        "matched_comparisons": comparisons,
    }


def main() -> None:
    args = _parser().parse_args()
    if args.torch_threads < 1:
        raise ValueError("Expected --torch-threads to be positive.")
    torch.set_num_threads(args.torch_threads)
    report = run_screen(
        screen_config_path=args.screen_config,
        model_config_path=args.model_config,
        teacher_weights_path=args.teacher_weights,
        output_dir=args.output_dir,
        aihwkit_python=args.aihwkit_python,
        device=args.device,
        sample_limit=args.sample_limit,
    )
    atomic_write_json(
        args.output_dir.expanduser().resolve() / "analysis" / "summary.json",
        report,
    )


if __name__ == "__main__":
    main()
