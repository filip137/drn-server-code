"""Identity-aware IBM OM ideal bounded-codebook deployment screen.

The frozen logical source and DRN solver remain FP32.  Literal AIHWKit OM
identities provide per-cell bounds, fitted references, and SET increments.
Each cell's deterministic lower-to-SET pulse trajectory is enumerated with
both stochastic terms disabled; requested conductances are then projected to
the nearest reachable code.  No program-and-verify controller, HWA modifier,
optimizer update, or endpoint sampler is used.

The screen crosses fixed-reference use with four versus eight devices per
original signed weight.  Four-device edges retain their physical conductance
in the nodal loading.  Eight-device edges retain separate active/reference
branches so transfer uses their difference and loading uses their sum.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gc
from hashlib import sha256
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
    signed_dual_rail_lift,
)
from experiments.mnist_relu_drn.ibm_om_ideal_mapping_scheme_screen import (
    Scheme,
    _finite_fraction,
    _normalized,
    _ordered_labels,
    _quad_contrast,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import RunMode
from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    load_om_array_population,
    sample_om_array_population_external,
)


SCHEMA = "ebl.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen"
SCHEMA_VERSION = 2
CONTRACT_SCHEMA_VERSION = 2
_LAYOUTS = ("halves", "paired")


@dataclass(frozen=True)
class ScreenContract:
    screen_id: str
    evidence_class: str
    preset: str
    required_aihwkit_version: str
    corruption_policy: str
    development_assignment_seed: int
    heldout_assignment_seeds: tuple[int, ...]
    maximum_pulses: int
    initial_reset_receipt_schema: str
    initial_reset_receipt_schema_version: int
    initial_reset_receipt_sha256: str
    initial_reset_teacher_sha256: str
    shared_scale_fractions: tuple[float, float]
    shared_fixed_logit_gain: float
    sample_limit: int | None
    continuous_envelope_control: bool
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class DeterministicCodebook:
    """Per-cell normalized-conductance states at pulse indices 0..K."""

    values: torch.Tensor
    effective_level_counts: torch.Tensor

    @property
    def maximum_pulses(self) -> int:
        return int(self.values.shape[0] - 1)


@dataclass(frozen=True)
class BaselineState:
    values: torch.Tensor
    pulse_indices: torch.Tensor
    upper_values: torch.Tensor
    report: Mapping[str, Any]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the identity-aware deterministic IBM OM bounded-codebook "
            "four/eight-device reference-policy screen."
        )
    )
    parser.add_argument("--screen-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument(
        "--initial-reset-calibration-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aihwkit-python", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--projection-chunk-size", type=int, default=16384)
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


def load_screen_contract(path: Path) -> ScreenContract:
    source = path.expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Expected a readable strict screen config: {source}.") from error
    if not isinstance(raw, dict):
        raise ValueError("Expected the screen config to contain one JSON object.")
    expected = {
        "schema_version",
        "screen_id",
        "evidence_class",
        "preset",
        "required_aihwkit_version",
        "corruption_policy",
        "development_assignment_seed",
        "heldout_assignment_seeds",
        "deterministic_codebook",
        "reference_policy",
        "shared_initial_reset_calibration",
        "continuous_envelope_control",
        "sample_limit",
    }
    _exact_keys(raw, expected, label="screen config")
    if raw["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise ValueError("Expected screen schema_version to equal 2.")
    for name in ("screen_id", "evidence_class", "preset", "required_aihwkit_version"):
        if not isinstance(raw[name], str) or not raw[name]:
            raise ValueError(f"Expected {name} to be a non-empty string.")
    if raw["evidence_class"] != "model_based_aihwkit_preset":
        raise ValueError("Expected the OM screen evidence class to stay model-based.")
    if raw["preset"] != "reram_array_om" or raw["required_aihwkit_version"] != "1.1.0":
        raise ValueError("Expected the pinned AIHWKit 1.1.0 OM preset.")
    if raw["corruption_policy"] != "counterfactual_repaired":
        raise ValueError(
            "Expected the first range/resolution screen to use the declared "
            "counterfactual-repaired population."
        )
    development_seed = _integer(
        raw["development_assignment_seed"],
        label="development_assignment_seed",
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

    codebook = raw["deterministic_codebook"]
    if not isinstance(codebook, dict):
        raise ValueError("Expected deterministic_codebook to be an object.")
    _exact_keys(
        codebook,
        {
            "start_state",
            "pulse_direction",
            "maximum_pulses",
            "cycle_to_cycle_random_term",
            "apparent_write_noise",
            "conductance_coordinate",
            "target_projection",
            "tie_break",
        },
        label="deterministic_codebook",
    )
    fixed_codebook = {
        "start_state": "sampled_min_bound",
        "pulse_direction": "set_only",
        "cycle_to_cycle_random_term": 0.0,
        "apparent_write_noise": 0.0,
        "conductance_coordinate": "clip((a+1)/2,0,1)",
        "target_projection": "nearest_absolute_error",
        "tie_break": "lowest_pulse_index",
    }
    mismatches = {
        key: {"expected": expected_value, "provided": codebook.get(key)}
        for key, expected_value in fixed_codebook.items()
        if codebook.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Expected the deterministic no-noise codebook: {mismatches!r}.")
    maximum_pulses = _integer(
        codebook["maximum_pulses"], label="maximum_pulses", minimum=1
    )

    reference = raw["reference_policy"]
    if not isinstance(reference, dict):
        raise ValueError("Expected reference_policy to be an object.")
    _exact_keys(
        reference,
        {"without_fixed_r", "with_fixed_r", "out_of_bounds_reference"},
        label="reference_policy",
    )
    if reference != {
        "without_fixed_r": "deterministic_pulse_zero_lower_state",
        "with_fixed_r": "nearest_deterministic_code_to_bounded_sampled_reference",
        "out_of_bounds_reference": "project_to_nearest_reachable_code_and_report",
    }:
        raise ValueError("Expected the predeclared lower/fitted-r reference policies.")

    calibration = raw["shared_initial_reset_calibration"]
    if not isinstance(calibration, dict):
        raise ValueError("Expected shared_initial_reset_calibration to be an object.")
    _exact_keys(
        calibration,
        {
            "source",
            "receipt_schema",
            "receipt_schema_version",
            "receipt_sha256",
            "teacher_sha256",
            "target_scale_fractions",
            "fixed_logit_gain",
            "application",
        },
        label="shared_initial_reset_calibration",
    )
    if calibration["source"] != "immutable_initial_reset_receipt":
        raise ValueError("Expected calibration to come from the initial-RESET receipt.")
    receipt_schema = calibration["receipt_schema"]
    if receipt_schema != "ebl.mnist_relu_drn.ibm_om_reset_relative_initialization":
        raise ValueError("Expected the IBM OM RESET-relative initialization receipt.")
    receipt_schema_version = _integer(
        calibration["receipt_schema_version"],
        label="shared_initial_reset_calibration.receipt_schema_version",
        minimum=1,
    )
    if receipt_schema_version != 1:
        raise ValueError("Expected RESET calibration receipt schema version 1.")
    receipt_sha256 = calibration["receipt_sha256"]
    teacher_sha256 = calibration["teacher_sha256"]
    for name, value in (
        ("receipt_sha256", receipt_sha256),
        ("teacher_sha256", teacher_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Expected a lowercase SHA-256 for {name}.")
    scales_raw = calibration["target_scale_fractions"]
    if not isinstance(scales_raw, list) or len(scales_raw) != 2:
        raise ValueError("Expected exactly two shared RESET scale fractions.")
    scales = tuple(
        _finite_fraction(
            value,
            name=f"target_scale_fractions[{index}]",
            include_zero=False,
            include_one=True,
        )
        for index, value in enumerate(scales_raw)
    )
    fixed_logit_gain = calibration["fixed_logit_gain"]
    if (
        isinstance(fixed_logit_gain, bool)
        or not isinstance(fixed_logit_gain, (int, float))
        or not math.isfinite(float(fixed_logit_gain))
        or float(fixed_logit_gain) <= 0.0
    ):
        raise ValueError("Expected a finite positive shared RESET logit gain.")
    if calibration["application"] != "all_schemes_and_endpoint_controls":
        raise ValueError("Expected one RESET calibration for every screen variant.")
    if raw["continuous_envelope_control"] is not True:
        raise ValueError("Expected the continuous-envelope diagnostic control.")
    sample_limit = raw["sample_limit"]
    if sample_limit is not None:
        sample_limit = _integer(sample_limit, label="sample_limit", minimum=1)

    return ScreenContract(
        screen_id=raw["screen_id"],
        evidence_class=raw["evidence_class"],
        preset=raw["preset"],
        required_aihwkit_version=raw["required_aihwkit_version"],
        corruption_policy=raw["corruption_policy"],
        development_assignment_seed=development_seed,
        heldout_assignment_seeds=heldout,
        maximum_pulses=maximum_pulses,
        initial_reset_receipt_schema=receipt_schema,
        initial_reset_receipt_schema_version=receipt_schema_version,
        initial_reset_receipt_sha256=receipt_sha256,
        initial_reset_teacher_sha256=teacher_sha256,
        shared_scale_fractions=(float(scales[0]), float(scales[1])),
        shared_fixed_logit_gain=float(fixed_logit_gain),
        sample_limit=sample_limit,
        continuous_envelope_control=True,
        raw=raw,
    )


def load_initial_reset_calibration_receipt(
    path: Path,
    *,
    contract: ScreenContract,
    teacher_weights_path: Path,
) -> dict[str, Any]:
    """Validate the immutable calibration anchor and return compact provenance."""

    source = path.expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Expected a readable initial-RESET calibration receipt: {source}."
        ) from error
    if not isinstance(raw, dict):
        raise ValueError("Expected the initial-RESET receipt to contain one object.")
    observed_receipt_sha256 = sha256_file(source)
    if observed_receipt_sha256 != contract.initial_reset_receipt_sha256:
        raise ValueError(
            "Initial-RESET calibration receipt SHA-256 mismatch: "
            f"expected {contract.initial_reset_receipt_sha256}, "
            f"observed {observed_receipt_sha256}."
        )
    observed_teacher_sha256 = sha256_file(teacher_weights_path)
    expected = {
        "schema": contract.initial_reset_receipt_schema,
        "schema_version": contract.initial_reset_receipt_schema_version,
        "teacher_sha256": contract.initial_reset_teacher_sha256,
        "policy": "expand_selected_lower_placed_layer_fraction_to_one",
        "optimizer_updates": 0,
    }
    mismatches = {
        key: {"expected": expected_value, "provided": raw.get(key)}
        for key, expected_value in expected.items()
        if raw.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"Initial-RESET calibration receipt contract mismatch: {mismatches!r}."
        )
    if observed_teacher_sha256 != contract.initial_reset_teacher_sha256:
        raise ValueError(
            "Teacher SHA-256 does not match the shared initial-RESET calibration: "
            f"expected {contract.initial_reset_teacher_sha256}, "
            f"observed {observed_teacher_sha256}."
        )
    receipt_scales = raw.get("target_scale_fractions")
    expected_scales = list(contract.shared_scale_fractions)
    if receipt_scales != expected_scales:
        raise ValueError(
            "Initial-RESET scale-fraction mismatch: "
            f"expected {expected_scales!r}, provided {receipt_scales!r}."
        )
    receipt_gain = raw.get("fixed_logit_gain")
    if (
        isinstance(receipt_gain, bool)
        or not isinstance(receipt_gain, (int, float))
        or float(receipt_gain) != contract.shared_fixed_logit_gain
    ):
        raise ValueError(
            "Initial-RESET fixed-logit-gain mismatch: "
            f"expected {contract.shared_fixed_logit_gain!r}, "
            f"provided {receipt_gain!r}."
        )
    return {
        "source": "immutable_initial_reset_receipt",
        "receipt_path": str(source),
        "receipt_sha256": observed_receipt_sha256,
        "receipt_schema": raw["schema"],
        "receipt_schema_version": raw["schema_version"],
        "policy": raw["policy"],
        "optimizer_updates": raw["optimizer_updates"],
        "teacher_sha256": observed_teacher_sha256,
        "source_checkpoint_sha256": raw.get("source_sha256"),
        "derived_checkpoint_sha256": raw.get("output_sha256"),
        "target_scale_fractions": expected_scales,
        "fixed_logit_gain": contract.shared_fixed_logit_gain,
        "application": "all_schemes_and_endpoint_controls",
        "per_scheme_refit_performed": False,
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    digest = sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _float_summary(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
        raise ValueError("Expected a non-empty finite tensor for summary.")
    quantiles = torch.quantile(
        flat,
        torch.tensor((0.01, 0.1, 0.5, 0.9, 0.99), dtype=torch.float64),
    )
    return {
        "minimum": float(flat.min().item()),
        "p01": float(quantiles[0].item()),
        "p10": float(quantiles[1].item()),
        "median": float(quantiles[2].item()),
        "mean": float(flat.mean().item()),
        "p90": float(quantiles[3].item()),
        "p99": float(quantiles[4].item()),
        "maximum": float(flat.max().item()),
        "rms": float(flat.square().mean().sqrt().item()),
    }


def _integer_summary(value: torch.Tensor) -> dict[str, float | int]:
    report = _float_summary(value.to(dtype=torch.float64))
    return {"minimum": int(report["minimum"]), **{
        key: report[key]
        for key in ("p01", "p10", "median", "mean", "p90", "p99")
    }, "maximum": int(report["maximum"])}


def _native_to_unit(state: torch.Tensor) -> torch.Tensor:
    return ((state + 1.0) / 2.0).clamp(0.0, 1.0)


def build_deterministic_set_codebook(
    population: IbmReramArrayPopulation,
    *,
    maximum_pulses: int,
) -> DeterministicCodebook:
    """Enumerate exact xi=0 SET states from each sampled lower bound."""

    if maximum_pulses < 1:
        raise ValueError("Expected at least one deterministic SET pulse.")
    physical = population.min_bound.detach().to(device="cpu", dtype=torch.float32).clone()
    maximum = population.max_bound.detach().to(device="cpu", dtype=torch.float32)
    step = population.dwmin_up.detach().to(device="cpu", dtype=torch.float32)
    levels = [_native_to_unit(physical)]
    for _pulse in range(maximum_pulses):
        normalized = torch.where(
            maximum > 0.0,
            physical / maximum,
            torch.zeros_like(physical),
        )
        response = step * (1.0 - normalized)
        candidate = physical + response
        candidate = torch.maximum(candidate, population.min_bound)
        candidate = torch.minimum(candidate, maximum)
        if bool(torch.any(candidate < physical)):
            raise RuntimeError("Expected deterministic SET states to be monotone.")
        physical = candidate
        levels.append(_native_to_unit(physical))
    values = torch.stack(levels)
    if not bool(torch.isfinite(values).all()) or bool(torch.any(values[1:] < values[:-1])):
        raise RuntimeError("Expected finite monotone normalized codebook levels.")
    effective = 1 + (values[1:] != values[:-1]).sum(dim=0).to(torch.int64)
    return DeterministicCodebook(values=values, effective_level_counts=effective)


def project_to_nearest_code(
    codebook_values: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project with the first (lowest-pulse) index winning exact ties."""

    if (
        codebook_values.ndim != 2
        or target.ndim != 1
        or codebook_values.shape[1] != target.numel()
        or chunk_size < 1
        or not bool(torch.isfinite(codebook_values).all())
        or not bool(torch.isfinite(target).all())
    ):
        raise ValueError("Expected a finite codebook and one finite target per cell.")
    realized = torch.empty_like(target)
    indices = torch.empty(target.shape, dtype=torch.int64, device=target.device)
    for start in range(0, target.numel(), chunk_size):
        stop = min(start + chunk_size, target.numel())
        values = codebook_values[:, start:stop]
        distance = torch.abs(values - target[start:stop].unsqueeze(0))
        _error, selected = torch.min(distance, dim=0)
        realized[start:stop] = torch.gather(
            values,
            0,
            selected.unsqueeze(0),
        ).squeeze(0)
        indices[start:stop] = selected
    return realized, indices


def build_baseline_state(
    population: IbmReramArrayPopulation,
    codebook: DeterministicCodebook,
    *,
    use_fixed_reference: bool,
    projection_chunk_size: int,
) -> BaselineState:
    if use_fixed_reference:
        bounded_native = torch.maximum(population.reference, population.min_bound)
        bounded_native = torch.minimum(bounded_native, population.max_bound)
        requested = _native_to_unit(bounded_native)
        values, indices = project_to_nearest_code(
            codebook.values,
            requested,
            chunk_size=projection_chunk_size,
        )
        outside = (population.reference < population.min_bound) | (
            population.reference > population.max_bound
        )
        error = values - requested
        report: dict[str, Any] = {
            "policy": "nearest_deterministic_code_to_bounded_sampled_reference",
            "raw_reference_outside_sampled_bounds_count": int(outside.sum().item()),
            "raw_reference_outside_sampled_bounds_fraction": float(
                outside.to(torch.float64).mean().item()
            ),
            "bounded_reference_target": _float_summary(requested),
            "realized_reference_code": _float_summary(values),
            "reference_projection_error": _float_summary(error),
            "reference_pulse_index": _integer_summary(indices),
        }
    else:
        values = codebook.values[0].clone()
        indices = torch.zeros(population.size, dtype=torch.int64)
        report = {
            "policy": "deterministic_pulse_zero_lower_state",
            "lower_code": _float_summary(values),
            "reference_pulse_index": _integer_summary(indices),
        }
    return BaselineState(
        values=values,
        pulse_indices=indices,
        upper_values=codebook.values[-1].clone(),
        report=report,
    )


def _binding_slices(population: IbmReramArrayPopulation) -> dict[str, slice]:
    result = {}
    offset = 0
    for key, shape in zip(population.binding_keys, population.binding_shapes):
        count = math.prod(shape)
        result[key] = slice(offset, offset + count)
        offset += count
    if offset != population.size:
        raise RuntimeError("Expected binding slices to cover the OM population.")
    return result


def _quad_columns(shape: tuple[int, int], layout: str, *, device: torch.device):
    rows, columns = shape
    if rows % 2 or columns % 2 or layout not in _LAYOUTS:
        raise ValueError("Expected an even dual-rail matrix and canonical layout.")
    logical_columns = columns // 2
    plus = (
        torch.arange(logical_columns, device=device)
        if layout == "halves"
        else 2 * torch.arange(logical_columns, device=device)
    )
    minus = plus + logical_columns if layout == "halves" else plus + 1
    return rows // 2, plus, minus


def _expanded_quad_minimum(value: torch.Tensor, *, layout: str) -> torch.Tensor:
    logical_rows, plus_columns, minus_columns = _quad_columns(
        tuple(value.shape), layout, device=value.device
    )
    quad = torch.stack(
        (
            value[:logical_rows][:, plus_columns],
            value[:logical_rows][:, minus_columns],
            value[logical_rows:][:, plus_columns],
            value[logical_rows:][:, minus_columns],
        )
    )
    common = quad.min(dim=0).values
    result = torch.empty_like(value)
    result[:logical_rows][:, plus_columns] = common
    result[:logical_rows][:, minus_columns] = common
    result[logical_rows:][:, plus_columns] = common
    result[logical_rows:][:, minus_columns] = common
    return result


def _layer_mapping_report(
    *,
    layer_index: int,
    layout: str,
    logical_weight: torch.Tensor,
    scale_fraction: float,
    baseline_active: torch.Tensor,
    upper_active: torch.Tensor,
    continuous_active: torch.Tensor,
    realized_active: torch.Tensor,
    active_pulse_indices: torch.Tensor,
    lifted: torch.Tensor,
    baseline_difference: torch.Tensor,
    difference: torch.Tensor,
    loading: torch.Tensor,
    continuous_difference: torch.Tensor,
    conductance_min: float,
    conductance_max: float,
) -> dict[str, Any]:
    span = conductance_max - conductance_min
    normalized_difference = (difference / span) if span else difference
    normalized_loading = (loading / span) if span else loading
    normalized_baseline_difference = (
        baseline_difference / span if span else baseline_difference
    )
    baseline_contrast = (
        _quad_contrast(normalized_baseline_difference, layout=layout) / 2.0
    )
    contrast = _quad_contrast(normalized_difference, layout=layout) / 2.0
    continuous_contrast = _quad_contrast(
        continuous_difference / span if span else continuous_difference,
        layout=layout,
    ) / 2.0
    requested_sign = torch.sign(_normalized(logical_weight).cpu())
    realized_sign = torch.sign(contrast)
    continuous_realized_sign = torch.sign(continuous_contrast)
    nonzero = requested_sign != 0
    sign_flip = nonzero & (requested_sign != realized_sign)
    continuous_sign_flip = nonzero & (
        requested_sign != continuous_realized_sign
    )
    codebook_incremental_contrast = contrast - baseline_contrast
    continuous_incremental_contrast = continuous_contrast - baseline_contrast
    codebook_incremental_sign_flip = nonzero & (
        requested_sign != torch.sign(codebook_incremental_contrast)
    )
    continuous_incremental_sign_flip = nonzero & (
        requested_sign != torch.sign(continuous_incremental_contrast)
    )
    raised = lifted > 0.0
    active_error = realized_active - continuous_active
    difference_rms = float(normalized_difference.double().square().mean().sqrt().item())
    loading_mean = float(normalized_loading.double().mean().item())
    return {
        "layer": layer_index,
        "layout": layout,
        "scale_fraction": scale_fraction,
        "baseline_active": _float_summary(baseline_active),
        "reachable_upper_active": _float_summary(upper_active),
        "common_headroom": _float_summary(
            _expanded_quad_minimum(upper_active - baseline_active, layout=layout)
        ),
        "continuous_active_target": _float_summary(continuous_active),
        "realized_active_code": _float_summary(realized_active),
        "active_projection_error": _float_summary(active_error),
        "active_projection_error_raised_only": (
            _float_summary(active_error[raised]) if bool(torch.any(raised)) else None
        ),
        "active_pulse_index": _integer_summary(active_pulse_indices),
        "active_pulse_index_raised_only": (
            _integer_summary(active_pulse_indices[raised])
            if bool(torch.any(raised))
            else None
        ),
        "logical_contrast": _float_summary(contrast),
        "continuous_logical_contrast": _float_summary(continuous_contrast),
        "baseline_logical_contrast": _float_summary(baseline_contrast),
        "codebook_incremental_logical_contrast": _float_summary(
            codebook_incremental_contrast
        ),
        "continuous_incremental_logical_contrast": _float_summary(
            continuous_incremental_contrast
        ),
        "nonzero_logical_weight_count": int(nonzero.sum().item()),
        "logical_sign_flip_count": int(sign_flip.sum().item()),
        "logical_sign_flip_fraction_nonzero": (
            float(sign_flip.sum().item()) / int(nonzero.sum().item())
            if bool(torch.any(nonzero))
            else 0.0
        ),
        "continuous_logical_sign_flip_count": int(
            continuous_sign_flip.sum().item()
        ),
        "continuous_logical_sign_flip_fraction_nonzero": (
            float(continuous_sign_flip.sum().item()) / int(nonzero.sum().item())
            if bool(torch.any(nonzero))
            else 0.0
        ),
        "codebook_incremental_sign_flip_count": int(
            codebook_incremental_sign_flip.sum().item()
        ),
        "codebook_incremental_sign_flip_fraction_nonzero": (
            float(codebook_incremental_sign_flip.sum().item())
            / int(nonzero.sum().item())
            if bool(torch.any(nonzero))
            else 0.0
        ),
        "continuous_incremental_sign_flip_count": int(
            continuous_incremental_sign_flip.sum().item()
        ),
        "continuous_incremental_sign_flip_fraction_nonzero": (
            float(continuous_incremental_sign_flip.sum().item())
            / int(nonzero.sum().item())
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
        "normalized_total_conductance_proxy": float(normalized_loading.double().sum().item()),
        "hashes": {
            "continuous_active_target": _tensor_sha256(continuous_active),
            "realized_active_code": _tensor_sha256(realized_active),
            "active_pulse_index": _tensor_sha256(active_pulse_indices),
            "baseline_difference": _tensor_sha256(baseline_difference),
            "difference": _tensor_sha256(difference),
            "loading": _tensor_sha256(loading),
        },
    }


def build_bounded_scheme_targets(
    logical_weights: Sequence[torch.Tensor],
    population: IbmReramArrayPopulation,
    codebook: DeterministicCodebook,
    *,
    scheme: Scheme,
    scale_fractions: tuple[float, float],
    conductance_min: float,
    conductance_max: float,
    projection_chunk_size: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], dict[str, Any]]:
    """Return exact codebook targets, continuous envelope targets, and evidence."""

    if len(logical_weights) != 2 or len(scale_fractions) != 2:
        raise ValueError("Expected exactly two logical matrices and scale fractions.")
    if scheme.device_count_per_logical_weight not in {4, 8}:
        raise ValueError("Expected a four- or eight-device scheme.")
    lower = float(conductance_min)
    upper = float(conductance_max)
    if lower < 0.0 or upper <= lower or not all(math.isfinite(v) for v in (lower, upper)):
        raise ValueError("Expected finite nonnegative increasing conductance bounds.")
    expected_bindings = 2 if scheme.device_count_per_logical_weight == 4 else 4
    if len(population.binding_keys) != expected_bindings:
        raise ValueError("Expected population bindings to match the physical topology.")

    baseline = build_baseline_state(
        population,
        codebook,
        use_fixed_reference=scheme.use_fixed_reference,
        projection_chunk_size=projection_chunk_size,
    )
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
            _normalized(logical).cpu(),
            target_layout=layout,
        )
        if scheme.device_count_per_logical_weight == 4:
            key = population.binding_keys[layer_index]
            binding_slice = slices[key]
            shape = population.binding_shapes[layer_index]
            baseline_active = baseline.values[binding_slice].reshape(shape)
            upper_active = baseline.upper_values[binding_slice].reshape(shape)
            common = _expanded_quad_minimum(
                upper_active - baseline_active,
                layout=layout,
            )
            continuous_active = baseline_active + scale_fraction * common * lifted
            realized_flat, pulse_indices_flat = project_to_nearest_code(
                codebook.values[:, binding_slice],
                continuous_active.reshape(-1),
                chunk_size=projection_chunk_size,
            )
            realized_active = realized_flat.reshape(shape)
            pulse_indices = pulse_indices_flat.reshape(shape)
            exact_physical = lower + span * realized_active
            continuous_physical = lower + span * continuous_active
            exact_targets.append(exact_physical)
            continuous_targets.append(continuous_physical)
            difference = exact_physical
            loading = exact_physical
            continuous_difference = continuous_physical
            baseline_difference = lower + span * baseline_active
        else:
            active_index = 2 * layer_index
            reference_index = active_index + 1
            active_key = population.binding_keys[active_index]
            reference_key = population.binding_keys[reference_index]
            active_slice = slices[active_key]
            reference_slice = slices[reference_key]
            active_shape = population.binding_shapes[active_index]
            reference_shape = population.binding_shapes[reference_index]
            if active_shape != reference_shape:
                raise ValueError("Expected matched active/reference branch shapes.")
            baseline_active = baseline.values[active_slice].reshape(active_shape)
            upper_active = baseline.upper_values[active_slice].reshape(active_shape)
            baseline_reference = baseline.values[reference_slice].reshape(reference_shape)
            common = _expanded_quad_minimum(
                upper_active - baseline_active,
                layout=layout,
            )
            continuous_active = baseline_active + scale_fraction * common * lifted
            realized_flat, pulse_indices_flat = project_to_nearest_code(
                codebook.values[:, active_slice],
                continuous_active.reshape(-1),
                chunk_size=projection_chunk_size,
            )
            realized_active = realized_flat.reshape(active_shape)
            pulse_indices = pulse_indices_flat.reshape(active_shape)
            exact_plus = lower + span * realized_active
            exact_minus = lower + span * baseline_reference
            continuous_plus = lower + span * continuous_active
            continuous_minus = exact_minus.clone()
            exact_targets.extend((exact_plus, exact_minus))
            continuous_targets.extend((continuous_plus, continuous_minus))
            difference = exact_plus - exact_minus
            loading = exact_plus + exact_minus
            continuous_difference = continuous_plus - continuous_minus
            baseline_difference = (
                lower + span * baseline_active
            ) - continuous_minus

        layer_reports.append(
            _layer_mapping_report(
                layer_index=layer_index,
                layout=layout,
                logical_weight=logical,
                scale_fraction=scale_fraction,
                baseline_active=baseline_active,
                upper_active=upper_active,
                continuous_active=continuous_active,
                realized_active=realized_active,
                active_pulse_indices=pulse_indices,
                lifted=lifted,
                baseline_difference=baseline_difference,
                difference=difference,
                loading=loading,
                continuous_difference=continuous_difference,
                conductance_min=lower,
                conductance_max=upper,
            )
        )

    return tuple(exact_targets), tuple(continuous_targets), {
        "scheme": scheme.name,
        "encoding": scheme.encoding,
        "device_count_per_logical_weight": scheme.device_count_per_logical_weight,
        "use_fixed_reference": scheme.use_fixed_reference,
        "reference_policy": baseline.report,
        "scale_fractions": [float(value) for value in scale_fractions],
        "layers": layer_reports,
        "hashes": {
            "bounded_codebook_targets": [
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


def _fixed_calibration_report(
    stack,
    teacher,
    loader: Iterable,
    labels: torch.Tensor,
    *,
    gain: float,
) -> dict[str, Any]:
    raw_scores, teacher_logits = collect_calibration(stack, teacher, loader)
    scores = raw_scores.detach().to(torch.float64)
    teacher_scores = teacher_logits.detach().to(torch.float64)
    teacher_log_prob = F.log_softmax(teacher_scores, dim=1)
    teacher_prob = teacher_log_prob.exp()

    def kl(applied_gain: float) -> float:
        student_log_prob = F.log_softmax(scores * applied_gain, dim=1)
        return float(
            (
                teacher_prob * (teacher_log_prob - student_log_prob)
            ).sum(dim=1).mean().item()
        )

    student_prediction = scores.detach().cpu().argmax(dim=1)
    teacher_prediction = teacher_scores.detach().cpu().argmax(dim=1)
    calibration_labels = labels.detach().cpu()
    if student_prediction.shape != calibration_labels.shape:
        raise RuntimeError("Expected calibration predictions and labels to match.")
    return {
        "gain": float(gain),
        "gain_source": "shared_initial_reset_receipt",
        "gain_was_fit_on_this_scheme": False,
        "raw_kl": kl(1.0),
        "calibrated_kl": kl(float(gain)),
        "score_rms": float(scores.square().mean().sqrt().item()),
        "calibrated_score_rms": float(
            (scores * float(gain)).square().mean().sqrt().item()
        ),
        "teacher_logit_rms": float(
            teacher_scores.square().mean().sqrt().item()
        ),
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


def _evaluate_detailed(
    stack,
    teacher,
    loader: Iterable,
    *,
    sample_limit: int | None,
) -> tuple[dict[str, Any], torch.Tensor]:
    totals = {
        "kl": 0.0,
        "raw_kl": 0.0,
        "student_correct": 0,
        "teacher_correct": 0,
        "agreement": 0,
        "raw_score_squared": 0.0,
        "calibrated_score_squared": 0.0,
        "teacher_score_squared": 0.0,
    }
    layers = tuple(stack.bundle.energy.layers())
    voltage = [
        {"count": 0, "sum": 0.0, "squared": 0.0, "minimum": None, "maximum": None}
        for _ in layers
    ]
    predictions = []
    examples = 0
    with torch.no_grad():
        for inputs, labels in limited(loader, None):
            if sample_limit is not None:
                remaining = sample_limit - examples
                if remaining <= 0:
                    break
                inputs = inputs[:remaining]
                labels = labels[:remaining]
            inputs = inputs.to(stack.device, dtype=torch.float32)
            labels = labels.to(stack.device, dtype=torch.long)
            teacher_logits = teacher.logits(inputs)
            stack.network.set_input(inputs, reset=True)
            stack.minimizer.compute_equilibrium()
            stack.cost.set_teacher(teacher_logits, labels)
            raw_scores = stack.cost.student_logits() / stack.cost.gain
            student_logits = raw_scores * stack.cost.gain
            teacher_log_prob = F.log_softmax(teacher_logits, dim=1)
            teacher_prob = teacher_log_prob.exp()
            student_log_prob = F.log_softmax(student_logits, dim=1)
            raw_student_log_prob = F.log_softmax(raw_scores, dim=1)
            totals["kl"] += float(
                (teacher_prob * (teacher_log_prob - student_log_prob)).sum().item()
            )
            totals["raw_kl"] += float(
                (teacher_prob * (teacher_log_prob - raw_student_log_prob)).sum().item()
            )
            student_prediction = student_logits.argmax(dim=1)
            teacher_prediction = teacher_logits.argmax(dim=1)
            predictions.append(student_prediction.detach().cpu())
            totals["student_correct"] += int(student_prediction.eq(labels).sum().item())
            totals["teacher_correct"] += int(teacher_prediction.eq(labels).sum().item())
            totals["agreement"] += int(
                student_prediction.eq(teacher_prediction).sum().item()
            )
            totals["raw_score_squared"] += float(raw_scores.square().sum().item())
            totals["calibrated_score_squared"] += float(student_logits.square().sum().item())
            totals["teacher_score_squared"] += float(teacher_logits.square().sum().item())
            for layer_index, layer in enumerate(layers):
                state = layer.state.detach()
                if not bool(torch.isfinite(state).all()):
                    raise RuntimeError("Expected finite layer voltages.")
                item = voltage[layer_index]
                item["count"] += state.numel()
                item["sum"] += float(state.double().sum().item())
                item["squared"] += float(state.double().square().sum().item())
                observed_minimum = float(state.min().item())
                observed_maximum = float(state.max().item())
                item["minimum"] = (
                    observed_minimum
                    if item["minimum"] is None
                    else min(float(item["minimum"]), observed_minimum)
                )
                item["maximum"] = (
                    observed_maximum
                    if item["maximum"] is None
                    else max(float(item["maximum"]), observed_maximum)
                )
            examples += int(labels.shape[0])
    if examples == 0:
        raise ValueError("Expected evaluation to process at least one example.")
    prediction = torch.cat(predictions).to(torch.long)
    score_values = examples * 10
    voltage_report = []
    for index, item in enumerate(voltage):
        count = int(item["count"])
        mean = float(item["sum"]) / count
        mean_square = float(item["squared"]) / count
        voltage_report.append(
            {
                "layer": index,
                "values": count,
                "mean": mean,
                "rms": math.sqrt(max(mean_square, 0.0)),
                "standard_deviation": math.sqrt(max(mean_square - mean * mean, 0.0)),
                "minimum": item["minimum"],
                "maximum": item["maximum"],
            }
        )
    return {
        "examples": examples,
        "student_correct": totals["student_correct"],
        "teacher_correct": totals["teacher_correct"],
        "teacher_agreement_count": totals["agreement"],
        "kl_teacher_student": totals["kl"] / examples,
        "raw_kl_teacher_student": totals["raw_kl"] / examples,
        "student_accuracy": totals["student_correct"] / examples,
        "teacher_accuracy": totals["teacher_correct"] / examples,
        "teacher_agreement": totals["agreement"] / examples,
        "raw_score_rms": math.sqrt(totals["raw_score_squared"] / score_values),
        "calibrated_score_rms": math.sqrt(
            totals["calibrated_score_squared"] / score_values
        ),
        "teacher_logit_rms": math.sqrt(totals["teacher_score_squared"] / score_values),
        "fixed_logit_gain": float(stack.cost.gain),
        "prediction_sha256": _tensor_sha256(prediction),
        "voltage": voltage_report,
    }, prediction


def _population_for(
    *,
    stack,
    topology: int,
    assignment_seed: int,
    contract: ScreenContract,
    aihwkit_python: Path,
    output_dir: Path,
) -> tuple[IbmReramArrayPopulation, dict[str, Any]]:
    stem = f"{topology}-device-assignment-{assignment_seed}"
    population_path = output_dir / "populations" / f"{stem}.npz"
    receipt_path = output_dir / "populations" / f"{stem}.receipt.json"
    if population_path.is_file():
        population = load_om_array_population(population_path)
        if not receipt_path.is_file():
            raise RuntimeError("Expected a sampling receipt beside an existing population.")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    else:
        population, receipt = sample_om_array_population_external(
            stack.bundle.catalog.trainable,
            assignment_seed=assignment_seed,
            corruption_policy=contract.corruption_policy,
            aihwkit_python=aihwkit_python,
            population_path=population_path,
            receipt_path=receipt_path,
        )
    expected_keys = tuple(binding.key for binding in stack.bundle.catalog.trainable)
    expected_shapes = tuple(
        tuple(binding.state.shape) for binding in stack.bundle.catalog.trainable
    )
    if (
        population.assignment_seed != assignment_seed
        or population.binding_keys != expected_keys
        or population.binding_shapes != expected_shapes
        or population.corruption_policy != contract.corruption_policy
        or population.aihwkit_version != contract.required_aihwkit_version
        or receipt.get("population_fingerprint") != population.fingerprint
        or receipt.get("population_sha256") != sha256_file(population_path)
    ):
        raise RuntimeError("Expected the frozen OM population and receipt to match.")
    return population, {
        "topology": topology,
        "assignment_seed": assignment_seed,
        "path": str(population_path),
        "sha256": sha256_file(population_path),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "population_fingerprint": population.fingerprint,
        "cells": population.size,
        "published_corrupt_cells_repaired": int(population.published_corrupt.sum().item()),
        "final_corrupt_cells": int(population.corrupt.sum().item()),
        "sampled_min_bound": _float_summary(population.min_bound),
        "sampled_max_bound": _float_summary(population.max_bound),
        "sampled_reference": _float_summary(population.reference),
        "sampled_set_step": _float_summary(population.dwmin_up),
        "nominal_dw_min": population.nominal_dw_min,
        "declared_dw_min_std_but_disabled": population.dw_min_std,
        "declared_write_noise_std_but_disabled": population.write_noise_std,
        "min_bound_below_nominal_minus_one_count": int(
            (population.min_bound < -1.0).sum().item()
        ),
        "max_bound_above_nominal_plus_one_count": int(
            (population.max_bound > 1.0).sum().item()
        ),
    }


def _codebook_report(
    population: IbmReramArrayPopulation,
    codebook: DeterministicCodebook,
) -> dict[str, Any]:
    return {
        "maximum_pulses": codebook.maximum_pulses,
        "states_per_cell_before_duplicate_collapse": codebook.maximum_pulses + 1,
        "effective_level_count_after_global_coordinate_clipping": _integer_summary(
            codebook.effective_level_counts
        ),
        "pulse_zero": _float_summary(codebook.values[0]),
        "pulse_cap": _float_summary(codebook.values[-1]),
        "pulse_zero_sha256": _tensor_sha256(codebook.values[0]),
        "pulse_cap_sha256": _tensor_sha256(codebook.values[-1]),
        "effective_level_count_sha256": _tensor_sha256(
            codebook.effective_level_counts
        ),
        "cycle_to_cycle_random_term": 0.0,
        "apparent_write_noise": 0.0,
        "population_fingerprint": population.fingerprint,
    }


def run_screen(
    *,
    screen_config_path: Path,
    model_config_path: Path,
    teacher_weights_path: Path,
    initial_reset_calibration_receipt_path: Path,
    output_dir: Path,
    aihwkit_python: Path,
    device: str,
    sample_limit: int | None,
    projection_chunk_size: int,
) -> dict[str, Any]:
    screen_config_path = screen_config_path.expanduser().resolve()
    model_config_path = model_config_path.expanduser().resolve()
    teacher_weights_path = teacher_weights_path.expanduser().resolve()
    initial_reset_calibration_receipt_path = (
        initial_reset_calibration_receipt_path.expanduser().resolve()
    )
    output_dir = output_dir.expanduser().resolve()
    aihwkit_python = aihwkit_python.expanduser().resolve()
    contract = load_screen_contract(screen_config_path)
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("Expected --sample-limit to be positive or omitted.")
    effective_sample_limit = contract.sample_limit if sample_limit is None else sample_limit
    if projection_chunk_size < 1:
        raise ValueError("Expected --projection-chunk-size to be positive.")
    for path in (
        model_config_path,
        teacher_weights_path,
        initial_reset_calibration_receipt_path,
        aihwkit_python,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Expected required input file: {path}.")
    shared_calibration = load_initial_reset_calibration_receipt(
        initial_reset_calibration_receipt_path,
        contract=contract,
        teacher_weights_path=teacher_weights_path,
    )

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
        teacher_weights_path,
        device=torch.device(device),
        spec=base_spec,
    )
    logical_weights = tuple(parameter.detach().cpu() for parameter in teacher.parameters())
    if len(logical_weights) != 2:
        raise ValueError("Expected the bias-free teacher to expose two matrices.")
    calibration_labels = _ordered_labels(loaders.calibration)
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
            "initial_reset_calibration_receipt": str(
                initial_reset_calibration_receipt_path
            ),
            "initial_reset_calibration_receipt_sha256": sha256_file(
                initial_reset_calibration_receipt_path
            ),
            "aihwkit_python": str(aihwkit_python),
            "device": device,
            "sample_limit": effective_sample_limit,
            "projection_chunk_size": projection_chunk_size,
            "contract": contract.raw,
            "validated_shared_initial_reset_calibration": shared_calibration,
        },
    )

    schemes = _schemes()
    development: dict[str, dict[str, Any]] = {}
    development_populations = []
    for topology in (4, 8):
        topology_schemes = tuple(
            scheme for scheme in schemes if scheme.device_count_per_logical_weight == topology
        )
        topology_spec = replace(
            base_spec,
            model=replace(
                base_spec.model,
                encoding="single" if topology == 4 else "differential",
            ),
        )
        stack = build_student_stack(topology_spec, enable_measured=False)
        population, population_report = _population_for(
            stack=stack,
            topology=topology,
            assignment_seed=contract.development_assignment_seed,
            contract=contract,
            aihwkit_python=aihwkit_python,
            output_dir=output_dir,
        )
        codebook = build_deterministic_set_codebook(
            population,
            maximum_pulses=contract.maximum_pulses,
        )
        population_report["codebook"] = _codebook_report(population, codebook)
        development_populations.append(population_report)
        for scheme in topology_schemes:
            pair = contract.shared_scale_fractions
            exact, continuous, mapping = build_bounded_scheme_targets(
                logical_weights,
                population,
                codebook,
                scheme=scheme,
                scale_fractions=(pair[0], pair[1]),
                conductance_min=topology_spec.model.conductance_min,
                conductance_max=topology_spec.model.conductance_max,
                projection_chunk_size=projection_chunk_size,
            )
            apply_targets(stack.bundle.catalog, exact)
            bounded_calibration = _fixed_calibration_report(
                stack,
                teacher,
                loaders.calibration,
                calibration_labels,
                gain=contract.shared_fixed_logit_gain,
            )
            apply_targets(stack.bundle.catalog, continuous)
            continuous_calibration = _fixed_calibration_report(
                stack,
                teacher,
                loaders.calibration,
                calibration_labels,
                gain=contract.shared_fixed_logit_gain,
            )
            development[scheme.name] = {
                "scheme": scheme.name,
                "topology": topology,
                "scale_fractions": list(pair),
                "fixed_logit_gain": contract.shared_fixed_logit_gain,
                "calibration_source": "shared_initial_reset_receipt",
                "per_scheme_refit_performed": False,
                "bounded_codebook_calibration": bounded_calibration,
                "continuous_envelope_calibration": continuous_calibration,
                "mapping": mapping,
            }
            print(
                f"diagnosed {scheme.name}: shared_scales={pair}, "
                f"shared_gain={contract.shared_fixed_logit_gain:.12g}, "
                f"calibration_accuracy="
                f"{100.0 * bounded_calibration['student_accuracy']:.2f}%",
                flush=True,
            )
        del codebook, population, stack
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
            stack = build_student_stack(topology_spec, enable_measured=False)
            population, population_report = _population_for(
                stack=stack,
                topology=topology,
                assignment_seed=assignment_seed,
                contract=contract,
                aihwkit_python=aihwkit_python,
                output_dir=output_dir,
            )
            codebook = build_deterministic_set_codebook(
                population,
                maximum_pulses=contract.maximum_pulses,
            )
            population_report["codebook"] = _codebook_report(population, codebook)
            arm_reports = []
            for scheme in (
                value
                for value in schemes
                if value.device_count_per_logical_weight == topology
            ):
                pair = contract.shared_scale_fractions
                exact, continuous, mapping = build_bounded_scheme_targets(
                    logical_weights,
                    population,
                    codebook,
                    scheme=scheme,
                    scale_fractions=(pair[0], pair[1]),
                    conductance_min=topology_spec.model.conductance_min,
                    conductance_max=topology_spec.model.conductance_max,
                    projection_chunk_size=projection_chunk_size,
                )
                apply_targets(stack.bundle.catalog, exact)
                stack.cost.gain = contract.shared_fixed_logit_gain
                exact_metrics, exact_predictions = _evaluate_detailed(
                    stack,
                    teacher,
                    loaders.test,
                    sample_limit=effective_sample_limit,
                )
                apply_targets(stack.bundle.catalog, continuous)
                stack.cost.gain = contract.shared_fixed_logit_gain
                continuous_metrics, continuous_predictions = _evaluate_detailed(
                    stack,
                    teacher,
                    loaders.test,
                    sample_limit=effective_sample_limit,
                )
                if exact_predictions.shape != continuous_predictions.shape:
                    raise RuntimeError("Expected matched continuous/codebook predictions.")
                flips = exact_predictions != continuous_predictions
                arm_reports.append(
                    {
                        "scheme": scheme.name,
                        "assignment_seed": assignment_seed,
                        "scale_fractions": list(pair),
                        "fixed_logit_gain": contract.shared_fixed_logit_gain,
                        "calibration_source": "shared_initial_reset_receipt",
                        "per_scheme_refit_performed": False,
                        "mapping": mapping,
                        "bounded_codebook_test": exact_metrics,
                        "continuous_envelope_test": continuous_metrics,
                        "continuous_to_codebook_prediction_flip_count": int(flips.sum().item()),
                        "continuous_to_codebook_prediction_flip_fraction": float(
                            flips.to(torch.float64).mean().item()
                        ),
                    }
                )
                print(
                    f"heldout {assignment_seed} {scheme.name}: "
                    f"codebook={100.0 * exact_metrics['student_accuracy']:.2f}%, "
                    f"continuous={100.0 * continuous_metrics['student_accuracy']:.2f}%",
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
            del codebook, population, stack
            gc.collect()

    aggregate = {}
    for scheme in schemes:
        reports = [
            arm
            for assignment in heldout_reports
            for arm in assignment["arms"]
            if arm["scheme"] == scheme.name
        ]
        codebook_accuracy = torch.tensor(
            [arm["bounded_codebook_test"]["student_accuracy"] for arm in reports],
            dtype=torch.float64,
        )
        continuous_accuracy = torch.tensor(
            [arm["continuous_envelope_test"]["student_accuracy"] for arm in reports],
            dtype=torch.float64,
        )
        delta = codebook_accuracy - continuous_accuracy
        aggregate[scheme.name] = {
            "assignments": [arm["assignment_seed"] for arm in reports],
            "bounded_codebook_accuracy": _float_summary(codebook_accuracy),
            "continuous_envelope_accuracy": _float_summary(continuous_accuracy),
            "codebook_minus_continuous_accuracy": _float_summary(delta),
            "passes_90_percent_mean_gate": bool(codebook_accuracy.mean() >= 0.9),
        }

    comparisons = {
        "fixed_r_minus_no_r": {
            "four_devices": (
                aggregate["four_with_fixed_r"]["bounded_codebook_accuracy"]["mean"]
                - aggregate["four_without_fixed_r"]["bounded_codebook_accuracy"]["mean"]
            ),
            "eight_devices": (
                aggregate["eight_with_fixed_r"]["bounded_codebook_accuracy"]["mean"]
                - aggregate["eight_without_fixed_r"]["bounded_codebook_accuracy"]["mean"]
            ),
        },
        "eight_minus_four": {
            "without_fixed_r": (
                aggregate["eight_without_fixed_r"]["bounded_codebook_accuracy"]["mean"]
                - aggregate["four_without_fixed_r"]["bounded_codebook_accuracy"]["mean"]
            ),
            "with_fixed_r": (
                aggregate["eight_with_fixed_r"]["bounded_codebook_accuracy"]["mean"]
                - aggregate["four_with_fixed_r"]["bounded_codebook_accuracy"]["mean"]
            ),
        },
    }
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "exploratory_identity_aware_ideal_bounded_codebook_complete",
        "screen_id": contract.screen_id,
        "claim_boundary": (
            "AIHWKit 1.1.0 normalized OM fitted-model control with repaired "
            "identities and deterministic oracle pulse codebooks. No stochastic "
            "write, program-and-verify controller, HWA, training, absolute "
            "conductance calibration, or fabricated-device claim."
        ),
        "source_precision": "fp32_teacher_and_drn_solver",
        "deployment_precision": "device_specific_deterministic_pulse_codebook",
        "screen_config": str(screen_config_path),
        "screen_config_sha256": sha256_file(screen_config_path),
        "model_config": str(model_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "teacher_weights": str(teacher_weights_path),
        "teacher_weights_sha256": sha256_file(teacher_weights_path),
        "initial_reset_calibration_receipt": str(
            initial_reset_calibration_receipt_path
        ),
        "initial_reset_calibration_receipt_sha256": sha256_file(
            initial_reset_calibration_receipt_path
        ),
        "teacher_architecture": teacher_metadata.get("architecture"),
        "device": device,
        "sample_limit": effective_sample_limit,
        "development_assignment_seed": contract.development_assignment_seed,
        "heldout_assignment_seeds": list(contract.heldout_assignment_seeds),
        "shared_initial_reset_calibration": shared_calibration,
        "scheme_contract": {
            "four_fixed_r_positive": "G++=G--=a; G+-=G-+=r",
            "four_fixed_r_negative": "G++=G--=r; G+-=G-+=a",
            "four_edge_transfer_and_loading": "D=G; S=G",
            "eight_edge_transfer": "D=G_a-G_r",
            "eight_edge_loading": "S=G_a+G_r",
        },
        "development_populations": development_populations,
        "development_diagnostics": development,
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
        initial_reset_calibration_receipt_path=(
            args.initial_reset_calibration_receipt
        ),
        output_dir=args.output_dir,
        aihwkit_python=args.aihwkit_python,
        device=args.device,
        sample_limit=args.sample_limit,
        projection_chunk_size=args.projection_chunk_size,
    )
    atomic_write_json(args.output_dir.expanduser().resolve() / "analysis" / "summary.json", report)


if __name__ == "__main__":
    main()
