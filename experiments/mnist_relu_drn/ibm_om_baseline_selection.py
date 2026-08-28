"""Pure four-device IBM OM baseline-selection mapping primitives.

The helpers in this module deliberately stop before experiment orchestration.
They construct explicit physical branch tensors ``G = B + d`` for the four
baseline policies in the baseline-selection study, validate those tensors,
and expose deterministic whole-quad repair and weight-error utilities.  The
four-device DRN consumes the returned full ``G`` matrices directly; no
baseline is subtracted before either transfer or loading is formed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from experiments.artifacts import atomic_write_json, sha256_file


INDEPENDENT_CELL_RESET_MEAN = "independent_cell_reset_mean"
SHARED_QUAD_RESET_MAX = "shared_quad_reset_max"
SHARED_DESTINATION_COLUMNS_RESET_MAX = (
    "shared_destination_columns_reset_max"
)
REFERENCE_ENFORCED_DESTINATION_COLUMNS = (
    "reference_enforced_destination_columns"
)
BASELINE_POLICIES = (
    INDEPENDENT_CELL_RESET_MEAN,
    SHARED_QUAD_RESET_MAX,
    SHARED_DESTINATION_COLUMNS_RESET_MAX,
    REFERENCE_ENFORCED_DESTINATION_COLUMNS,
)
LAYOUTS = ("halves", "paired")
MAPPING_SCHEMA = "ebl.mnist_relu_drn.ibm_om_baseline_physical_mapping"
MAPPING_SCHEMA_VERSION = 1


class InfeasibleBaselineError(ValueError):
    """Raised when a declared baseline cannot be represented by a quad."""


class DonorExhaustionError(RuntimeError):
    """Raised when a destination has no feasible donor in its frozen block."""


def _tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    digest = sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(
        json.dumps(list(cpu.shape), separators=(",", ":")).encode("utf-8")
    )
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_layout(shape: Sequence[int], layout: str) -> tuple[int, int]:
    if (
        len(shape) != 2
        or int(shape[0]) < 2
        or int(shape[1]) < 2
        or int(shape[0]) % 2
        or int(shape[1]) % 2
        or layout not in LAYOUTS
    ):
        raise ValueError(
            "Expected an even-by-even rank-2 rail matrix and layout "
            f"in {LAYOUTS!r}. Provided shape={tuple(shape)!r}, "
            f"layout={layout!r}."
        )
    return int(shape[0]) // 2, int(shape[1]) // 2


def _rail_columns(
    shape: Sequence[int], layout: str, *, device: torch.device
) -> tuple[int, torch.Tensor, torch.Tensor]:
    logical_rows, logical_columns = _validate_layout(shape, layout)
    plus = (
        torch.arange(logical_columns, device=device)
        if layout == "halves"
        else 2 * torch.arange(logical_columns, device=device)
    )
    minus = plus + logical_columns if layout == "halves" else plus + 1
    return logical_rows, plus, minus


def quad_stack(matrix: torch.Tensor, *, layout: str) -> torch.Tensor:
    """Return quads in canonical ``(++,+-,-+,--)`` order."""

    if matrix.ndim != 2:
        raise ValueError("Expected one rank-2 physical rail matrix.")
    rows, plus, minus = _rail_columns(
        matrix.shape, layout, device=matrix.device
    )
    return torch.stack(
        (
            matrix[:rows, plus],
            matrix[:rows, minus],
            matrix[rows:, plus],
            matrix[rows:, minus],
        ),
        dim=-1,
    )


def scatter_quads(
    quads: torch.Tensor, *, shape: Sequence[int], layout: str
) -> torch.Tensor:
    """Invert :func:`quad_stack` for canonical four-cell quads."""

    rows, logical_columns = _validate_layout(shape, layout)
    if quads.shape != (rows, logical_columns, 4):
        raise ValueError(
            "Expected quad tensor shape "
            f"{(rows, logical_columns, 4)!r}; provided {tuple(quads.shape)!r}."
        )
    result = torch.empty(
        tuple(int(value) for value in shape),
        dtype=quads.dtype,
        device=quads.device,
    )
    _rows, plus, minus = _rail_columns(shape, layout, device=quads.device)
    result[:rows, plus] = quads[..., 0]
    result[:rows, minus] = quads[..., 1]
    result[rows:, plus] = quads[..., 2]
    result[rows:, minus] = quads[..., 3]
    return result


def _quad_stack_field(value: torch.Tensor, *, layout: str) -> torch.Tensor:
    """Stack a rail matrix while preserving trailing per-cell field axes."""

    if value.ndim < 2:
        raise ValueError("Expected a field with two physical rail axes.")
    rows, plus, minus = _rail_columns(
        value.shape[:2], layout, device=value.device
    )
    return torch.stack(
        (
            value[:rows, plus, ...],
            value[:rows, minus, ...],
            value[rows:, plus, ...],
            value[rows:, minus, ...],
        ),
        dim=2,
    )


def _scatter_quad_field(
    quads: torch.Tensor, *, shape: Sequence[int], layout: str
) -> torch.Tensor:
    """Invert :func:`_quad_stack_field`, including trailing field axes."""

    if len(shape) < 2:
        raise ValueError("Expected a field with two physical rail axes.")
    rows, logical_columns = _validate_layout(shape[:2], layout)
    trailing = tuple(int(value) for value in shape[2:])
    expected = (rows, logical_columns, 4, *trailing)
    if quads.shape != expected:
        raise ValueError(
            f"Expected quad field shape {expected!r}; "
            f"provided {tuple(quads.shape)!r}."
        )
    result = torch.empty(
        tuple(int(value) for value in shape),
        dtype=quads.dtype,
        device=quads.device,
    )
    _rows, plus, minus = _rail_columns(
        shape[:2], layout, device=quads.device
    )
    result[:rows, plus, ...] = quads[:, :, 0, ...]
    result[:rows, minus, ...] = quads[:, :, 1, ...]
    result[rows:, plus, ...] = quads[:, :, 2, ...]
    result[rows:, minus, ...] = quads[:, :, 3, ...]
    return result


def quad_contrast(matrix: torch.Tensor, *, layout: str) -> torch.Tensor:
    """Return ``(G++ - G+- - G-+ + G--)/2`` from full conductances."""

    quad = quad_stack(matrix, layout=layout)
    # Pair the two conductances feeding each destination rail before combining
    # the rails.  This is algebraically identical to the expression in the
    # docstring, while preserving an exact zero when a baseline policy stores
    # G++ == G-+ and G+- == G-- in finite precision.
    return ((quad[..., 0] - quad[..., 2]) + (quad[..., 3] - quad[..., 1])) / 2.0


def baseline_group_violation_mask(
    matrix: torch.Tensor, *, layout: str, policy: str
) -> torch.Tensor:
    """Return logical quads that violate a policy's declared baseline groups."""

    quad = quad_stack(matrix, layout=layout)
    if policy == INDEPENDENT_CELL_RESET_MEAN:
        return torch.zeros(quad.shape[:-1], dtype=torch.bool, device=quad.device)
    if policy == SHARED_QUAD_RESET_MAX:
        return torch.any(quad != quad[..., :1], dim=-1)
    if policy in {
        SHARED_DESTINATION_COLUMNS_RESET_MAX,
        REFERENCE_ENFORCED_DESTINATION_COLUMNS,
    }:
        return (quad[..., 0] != quad[..., 2]) | (
            quad[..., 1] != quad[..., 3]
        )
    raise ValueError(f"Unexpected baseline policy: {policy!r}.")


def quad_loading(matrix: torch.Tensor, *, layout: str) -> torch.Tensor:
    """Return the sum of all four physical conductances in each quad."""

    return quad_stack(matrix, layout=layout).sum(dim=-1)


def quad_row_loading(matrix: torch.Tensor, *, layout: str) -> torch.Tensor:
    """Return ``(G+++G+-, G-++G--)`` for each logical connection."""

    quad = quad_stack(matrix, layout=layout)
    return torch.stack((quad[..., 0] + quad[..., 1], quad[..., 2] + quad[..., 3]), dim=-1)


def quad_column_loading(matrix: torch.Tensor, *, layout: str) -> torch.Tensor:
    """Return ``(G+++G-+, G+-+G--)`` for each logical connection."""

    quad = quad_stack(matrix, layout=layout)
    return torch.stack((quad[..., 0] + quad[..., 2], quad[..., 1] + quad[..., 3]), dim=-1)


def _normalized(weight: torch.Tensor) -> tuple[torch.Tensor, float]:
    value = weight.detach().to(device="cpu", dtype=torch.float64)
    if value.ndim != 2 or not bool(torch.isfinite(value).all()):
        raise ValueError("Expected a finite rank-2 logical weight matrix.")
    absolute_maximum = float(value.abs().max().item())
    if absolute_maximum <= 0.0:
        raise ValueError("Expected a logical matrix containing a nonzero weight.")
    return value / absolute_maximum, absolute_maximum


def _validate_quad_inputs(
    sign_source: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    reset_baseline: torch.Tensor,
    reference_unit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = tuple(
        value.detach().to(device="cpu", dtype=torch.float64)
        for value in (lower, upper, reset_baseline, reference_unit)
    )
    lower, upper, reset_baseline, reference_unit = tensors
    if (
        lower.shape[-1:] != (4,)
        or any(value.shape != lower.shape for value in tensors[1:])
        or sign_source.shape != lower.shape[:-1]
        or not all(bool(torch.isfinite(value).all()) for value in tensors)
        or not bool(torch.isfinite(sign_source).all())
        or bool(torch.any(lower < 0.0))
        or bool(torch.any(upper > 1.0))
        or bool(torch.any(upper < lower))
        or bool(torch.any(reset_baseline < lower))
        or bool(torch.any(reset_baseline > upper))
    ):
        raise ValueError(
            "Expected finite, matched (...,4) unit-coordinate bounds, "
            "bounded RESET baselines, references, and one sign source per quad."
        )
    return (
        sign_source.detach().to(device="cpu", dtype=torch.float64),
        lower,
        upper,
        reset_baseline,
        reference_unit,
    )


def policy_baseline_quads(
    sign_source: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    reset_baseline: torch.Tensor,
    reference_unit: torch.Tensor,
    *,
    policy: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unit baselines, active masks, and baseline-group IDs.

    ``sign_source == 0`` deterministically uses the positive active-role
    convention.  This affects only role metadata because its offset is zero.
    """

    if policy not in BASELINE_POLICIES:
        raise ValueError(f"Expected policy in {BASELINE_POLICIES!r}.")
    sign_source, lower, upper, reset_baseline, reference_unit = (
        _validate_quad_inputs(
            sign_source, lower, upper, reset_baseline, reference_unit
        )
    )
    positive = sign_source >= 0.0
    positive_active = torch.tensor(
        (True, False, False, True), dtype=torch.bool
    ).expand(lower.shape)
    negative_active = torch.tensor(
        (False, True, True, False), dtype=torch.bool
    ).expand(lower.shape)
    active = torch.where(positive.unsqueeze(-1), positive_active, negative_active)
    quad_count = int(math.prod(lower.shape[:-1]))
    quad_ids = torch.arange(quad_count, dtype=torch.int64).reshape(lower.shape[:-1])

    if policy == INDEPENDENT_CELL_RESET_MEAN:
        baseline = reset_baseline.clone()
        groups = 4 * quad_ids.unsqueeze(-1) + torch.arange(4, dtype=torch.int64)
    elif policy == SHARED_QUAD_RESET_MAX:
        shared = reset_baseline.max(dim=-1).values
        baseline = shared.unsqueeze(-1).expand_as(reset_baseline).clone()
        groups = quad_ids.unsqueeze(-1).expand_as(reset_baseline).clone()
    elif policy == SHARED_DESTINATION_COLUMNS_RESET_MAX:
        plus = torch.maximum(reset_baseline[..., 0], reset_baseline[..., 2])
        minus = torch.maximum(reset_baseline[..., 1], reset_baseline[..., 3])
        baseline = torch.stack((plus, minus, plus, minus), dim=-1)
        groups = torch.stack(
            (2 * quad_ids, 2 * quad_ids + 1, 2 * quad_ids, 2 * quad_ids + 1),
            dim=-1,
        )
    else:
        positive_baseline = torch.stack(
            (
                reference_unit[..., 2],
                reference_unit[..., 1],
                reference_unit[..., 2],
                reference_unit[..., 1],
            ),
            dim=-1,
        )
        negative_baseline = torch.stack(
            (
                reference_unit[..., 0],
                reference_unit[..., 3],
                reference_unit[..., 0],
                reference_unit[..., 3],
            ),
            dim=-1,
        )
        baseline = torch.where(
            positive.unsqueeze(-1), positive_baseline, negative_baseline
        )
        groups = torch.stack(
            (2 * quad_ids, 2 * quad_ids + 1, 2 * quad_ids, 2 * quad_ids + 1),
            dim=-1,
        )

    return baseline, active.clone(), groups


def all_policy_feasibility_from_quads(
    sign_source: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    reset_baseline: torch.Tensor,
    reference_unit: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Evaluate every baseline policy without using scale, labels, or accuracy."""

    result: dict[str, torch.Tensor] = {}
    for policy in BASELINE_POLICIES:
        baseline, _active, _groups = policy_baseline_quads(
            sign_source,
            lower,
            upper,
            reset_baseline,
            reference_unit,
            policy=policy,
        )
        result[policy] = (
            torch.isfinite(baseline).all(dim=-1)
            & (baseline >= 0.0).all(dim=-1)
            & (baseline <= 1.0).all(dim=-1)
            & (baseline >= lower).all(dim=-1)
            & (baseline <= upper).all(dim=-1)
        )
    result["all_policies"] = torch.stack(
        tuple(result[policy] for policy in BASELINE_POLICIES), dim=0
    ).all(dim=0)
    return result


@dataclass(frozen=True)
class LayerPhysicalMapping:
    layer_index: int
    layout: str
    policy: str
    scale_fraction: float
    teacher_absmax: float
    level_spacing_unit: float
    level_spacing_physical: float
    source_weight: torch.Tensor
    normalized_weight: torch.Tensor
    cell_lower_unit: torch.Tensor
    cell_upper_unit: torch.Tensor
    reset_baseline_unit: torch.Tensor
    intrinsic_reference_native: torch.Tensor
    intrinsic_reference_unit_unclipped: torch.Tensor
    baseline_group_id: torch.Tensor
    active_mask: torch.Tensor
    reference_mask: torch.Tensor
    baseline_unit: torch.Tensor
    baseline: torch.Tensor
    positive_headroom_unit: torch.Tensor
    negative_headroom_unit: torch.Tensor
    selected_headroom_unit: torch.Tensor
    positive_level_capacity: torch.Tensor
    negative_level_capacity: torch.Tensor
    selected_level_capacity: torch.Tensor
    integer_level_number: torch.Tensor
    continuous_offset: torch.Tensor
    quantized_offset: torch.Tensor
    continuous_conductance: torch.Tensor
    quantized_conductance: torch.Tensor
    baseline_contrast: torch.Tensor
    continuous_contrast: torch.Tensor
    quantized_contrast: torch.Tensor
    baseline_loading: torch.Tensor
    continuous_loading: torch.Tensor
    quantized_loading: torch.Tensor
    baseline_row_loading: torch.Tensor
    baseline_column_loading: torch.Tensor
    continuous_row_loading: torch.Tensor
    continuous_column_loading: torch.Tensor
    quantized_row_loading: torch.Tensor
    quantized_column_loading: torch.Tensor


@dataclass(frozen=True)
class PhysicalMapping:
    policy: str
    conductance_min: float
    conductance_max: float
    layers: tuple[LayerPhysicalMapping, ...]
    report: Mapping[str, Any]

    @property
    def continuous_targets(self) -> tuple[torch.Tensor, ...]:
        return tuple(layer.continuous_conductance for layer in self.layers)

    @property
    def quantized_targets(self) -> tuple[torch.Tensor, ...]:
        return tuple(layer.quantized_conductance for layer in self.layers)


def _summary(value: torch.Tensor) -> Mapping[str, float]:
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
        raise ValueError("Expected a non-empty finite tensor for summary.")
    return {
        "minimum": float(flat.min().item()),
        "mean": float(flat.mean().item()),
        "rms": float(flat.square().mean().sqrt().item()),
        "maximum": float(flat.max().item()),
    }


def build_layer_physical_mapping(
    logical_weight: torch.Tensor,
    *,
    layer_index: int,
    layout: str,
    policy: str,
    scale_fraction: float,
    nominal_dw_min: float,
    conductance_min: float,
    conductance_max: float,
    cell_lower_unit: torch.Tensor,
    cell_upper_unit: torch.Tensor,
    reset_baseline_unit: torch.Tensor,
    intrinsic_reference_native: torch.Tensor,
) -> LayerPhysicalMapping:
    """Build one continuous envelope and its frozen standard-4-delta probe."""

    normalized, teacher_absmax = _normalized(logical_weight)
    physical_shape = (2 * normalized.shape[0], 2 * normalized.shape[1])
    _validate_layout(physical_shape, layout)
    matrices = tuple(
        value.detach().to(device="cpu", dtype=torch.float64)
        for value in (
            cell_lower_unit,
            cell_upper_unit,
            reset_baseline_unit,
            intrinsic_reference_native,
        )
    )
    lower, upper, reset, reference_native = matrices
    if any(value.shape != physical_shape for value in matrices):
        raise ValueError(
            f"Expected physical tensors with shape {physical_shape!r}."
        )
    if not all(bool(torch.isfinite(value).all()) for value in matrices):
        raise ValueError("Expected finite physical mapping inputs.")
    if (
        bool(torch.any(lower < 0.0))
        or bool(torch.any(upper > 1.0))
        or bool(torch.any(upper < lower))
        or bool(torch.any(reset < lower))
        or bool(torch.any(reset > upper))
    ):
        raise ValueError("Expected unit bounds and bounded RESET means.")
    fraction = float(scale_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("Expected scale_fraction in (0,1].")
    native_delta = float(nominal_dw_min)
    level_spacing_unit = 4.0 * (native_delta / 2.0)
    lower_g = float(conductance_min)
    upper_g = float(conductance_max)
    if (
        not math.isfinite(native_delta)
        or native_delta <= 0.0
        or not math.isfinite(lower_g)
        or not math.isfinite(upper_g)
        or lower_g < 0.0
        or upper_g <= lower_g
    ):
        raise ValueError("Expected positive OM delta and increasing conductance bounds.")
    span = upper_g - lower_g
    level_spacing_physical = span * level_spacing_unit

    lower_quad = quad_stack(lower, layout=layout)
    upper_quad = quad_stack(upper, layout=layout)
    reset_quad = quad_stack(reset, layout=layout)
    reference_unit = (reference_native + 1.0) / 2.0
    reference_quad = quad_stack(reference_unit, layout=layout)
    baseline_quad, active_quad, group_quad = policy_baseline_quads(
        normalized,
        lower_quad,
        upper_quad,
        reset_quad,
        reference_quad,
        policy=policy,
    )
    feasible = (
        (baseline_quad >= lower_quad).all(dim=-1)
        & (baseline_quad <= upper_quad).all(dim=-1)
        & (baseline_quad >= 0.0).all(dim=-1)
        & (baseline_quad <= 1.0).all(dim=-1)
    )
    if not bool(feasible.all()):
        raise InfeasibleBaselineError(
            f"Baseline policy {policy!r} is infeasible for "
            f"{int((~feasible).sum().item())} quads."
        )

    headroom_quad = upper_quad - baseline_quad
    positive_headroom = torch.minimum(
        headroom_quad[..., 0], headroom_quad[..., 3]
    )
    negative_headroom = torch.minimum(
        headroom_quad[..., 1], headroom_quad[..., 2]
    )
    positive = normalized >= 0.0
    selected_headroom = torch.where(
        positive, positive_headroom, negative_headroom
    )
    continuous_level = fraction * selected_headroom * normalized.abs()
    continuous_offset_quad = (
        active_quad.to(torch.float64) * continuous_level.unsqueeze(-1)
    )

    positive_capacity = torch.floor(
        positive_headroom / level_spacing_unit + 1e-12
    ).to(torch.int64)
    negative_capacity = torch.floor(
        negative_headroom / level_spacing_unit + 1e-12
    ).to(torch.int64)
    selected_capacity = torch.where(
        positive, positive_capacity, negative_capacity
    )
    requested_level = continuous_level / level_spacing_unit
    selected_level = torch.floor(requested_level + 0.5).to(torch.int64)
    selected_level = torch.minimum(selected_level, selected_capacity)
    quantized_offset_quad = (
        active_quad.to(torch.float64)
        * selected_level.to(torch.float64).unsqueeze(-1)
        * level_spacing_unit
    )

    baseline_unit = scatter_quads(
        baseline_quad, shape=physical_shape, layout=layout
    )
    active_mask = scatter_quads(
        active_quad, shape=physical_shape, layout=layout
    )
    group_id = scatter_quads(group_quad, shape=physical_shape, layout=layout)
    continuous_offset_unit = scatter_quads(
        continuous_offset_quad, shape=physical_shape, layout=layout
    )
    quantized_offset_unit = scatter_quads(
        quantized_offset_quad, shape=physical_shape, layout=layout
    )
    integer_level_quad = (
        active_quad.to(torch.int64) * selected_level.unsqueeze(-1)
    )
    integer_level = scatter_quads(
        integer_level_quad, shape=physical_shape, layout=layout
    )

    baseline = (lower_g + span * baseline_unit).to(torch.float32)
    continuous_offset = (span * continuous_offset_unit).to(torch.float32)
    quantized_offset = (span * quantized_offset_unit).to(torch.float32)
    continuous = baseline + continuous_offset
    quantized = baseline + quantized_offset
    physical_lower = (lower_g + span * lower).to(torch.float32)
    physical_upper = (lower_g + span * upper).to(torch.float32)
    tolerance = max(1e-12, span * 1e-6)
    for name, target in (("continuous", continuous), ("quantized", quantized)):
        if (
            not bool(torch.isfinite(target).all())
            or bool(torch.any(target < -tolerance))
            or bool(torch.any(target < physical_lower - tolerance))
            or bool(torch.any(target > physical_upper + tolerance))
        ):
            raise RuntimeError(f"Expected valid in-bound {name} full conductances.")

    baseline_contrast = quad_contrast(baseline, layout=layout)
    continuous_contrast = quad_contrast(continuous, layout=layout)
    quantized_contrast = quad_contrast(quantized, layout=layout)
    return LayerPhysicalMapping(
        layer_index=int(layer_index),
        layout=layout,
        policy=policy,
        scale_fraction=fraction,
        teacher_absmax=teacher_absmax,
        level_spacing_unit=level_spacing_unit,
        level_spacing_physical=level_spacing_physical,
        source_weight=logical_weight.detach().to(device="cpu", dtype=torch.float32),
        normalized_weight=normalized.to(torch.float32),
        cell_lower_unit=lower.to(torch.float32),
        cell_upper_unit=upper.to(torch.float32),
        reset_baseline_unit=reset.to(torch.float32),
        intrinsic_reference_native=reference_native.to(torch.float32),
        intrinsic_reference_unit_unclipped=reference_unit.to(torch.float32),
        baseline_group_id=group_id,
        active_mask=active_mask,
        reference_mask=(~active_mask if policy == REFERENCE_ENFORCED_DESTINATION_COLUMNS else torch.zeros_like(active_mask)),
        baseline_unit=baseline_unit.to(torch.float32),
        baseline=baseline,
        positive_headroom_unit=positive_headroom.to(torch.float32),
        negative_headroom_unit=negative_headroom.to(torch.float32),
        selected_headroom_unit=selected_headroom.to(torch.float32),
        positive_level_capacity=positive_capacity,
        negative_level_capacity=negative_capacity,
        selected_level_capacity=selected_capacity,
        integer_level_number=integer_level,
        continuous_offset=continuous_offset,
        quantized_offset=quantized_offset,
        continuous_conductance=continuous,
        quantized_conductance=quantized,
        baseline_contrast=baseline_contrast,
        continuous_contrast=continuous_contrast,
        quantized_contrast=quantized_contrast,
        baseline_loading=quad_loading(baseline, layout=layout),
        continuous_loading=quad_loading(continuous, layout=layout),
        quantized_loading=quad_loading(quantized, layout=layout),
        baseline_row_loading=quad_row_loading(baseline, layout=layout),
        baseline_column_loading=quad_column_loading(baseline, layout=layout),
        continuous_row_loading=quad_row_loading(continuous, layout=layout),
        continuous_column_loading=quad_column_loading(continuous, layout=layout),
        quantized_row_loading=quad_row_loading(quantized, layout=layout),
        quantized_column_loading=quad_column_loading(quantized, layout=layout),
    )


def build_physical_mapping(
    logical_weights: Sequence[torch.Tensor],
    *,
    policy: str,
    scale_fractions: Sequence[float],
    nominal_dw_min: float,
    conductance_min: float,
    conductance_max: float,
    cell_lower_units: Sequence[torch.Tensor],
    cell_upper_units: Sequence[torch.Tensor],
    reset_baseline_units: Sequence[torch.Tensor],
    intrinsic_references_native: Sequence[torch.Tensor],
    layouts: Sequence[str] = LAYOUTS,
) -> PhysicalMapping:
    """Build the matched two-layer four-device physical mapping."""

    sequences = (
        tuple(logical_weights),
        tuple(scale_fractions),
        tuple(cell_lower_units),
        tuple(cell_upper_units),
        tuple(reset_baseline_units),
        tuple(intrinsic_references_native),
        tuple(layouts),
    )
    if any(len(value) != 2 for value in sequences):
        raise ValueError("Expected exactly two canonical MNIST DRN layers.")
    mapped = tuple(
        build_layer_physical_mapping(
            sequences[0][index],
            layer_index=index,
            layout=sequences[6][index],
            policy=policy,
            scale_fraction=float(sequences[1][index]),
            nominal_dw_min=nominal_dw_min,
            conductance_min=conductance_min,
            conductance_max=conductance_max,
            cell_lower_unit=sequences[2][index],
            cell_upper_unit=sequences[3][index],
            reset_baseline_unit=sequences[4][index],
            intrinsic_reference_native=sequences[5][index],
        )
        for index in range(2)
    )
    report = {
        "policy": policy,
        "decomposition": "every physical branch G=B+d",
        "circuit_accounting": "full G enters numerator and denominator",
        "continuous_metric": "ideal_bounded_continuous_init_accuracy",
        "quantized_diagnostic": "ideal_bounded_standard4delta_init_accuracy",
        "scale_fractions": [float(value) for value in scale_fractions],
        "layers": [
            {
                "layer": layer.layer_index,
                "layout": layer.layout,
                "baseline_contrast": _summary(layer.baseline_contrast),
                "continuous_contrast": _summary(layer.continuous_contrast),
                "quantized_contrast": _summary(layer.quantized_contrast),
                "baseline_loading": _summary(layer.baseline_loading),
                "selected_headroom_unit": _summary(layer.selected_headroom_unit),
                "zero_selected_capacity_count": int(
                    (layer.selected_level_capacity == 0).sum().item()
                ),
                "hashes": {
                    name: _tensor_sha256(getattr(layer, name))
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
                },
            }
            for layer in mapped
        ],
    }
    return PhysicalMapping(
        policy=policy,
        conductance_min=float(conductance_min),
        conductance_max=float(conductance_max),
        layers=mapped,
        report=report,
    )


def apply_checked_physical_targets(
    bindings_or_catalog: Any,
    mapping: PhysicalMapping,
    *,
    endpoint: str,
) -> None:
    """Validate every physical target before atomically mutating DRN tensors."""

    if endpoint not in {"continuous", "standard4delta"}:
        raise ValueError("Expected endpoint 'continuous' or 'standard4delta'.")
    bindings = tuple(
        bindings_or_catalog.trainable
        if hasattr(bindings_or_catalog, "trainable")
        else bindings_or_catalog
    )
    if len(bindings) != len(mapping.layers):
        raise ValueError("Expected one four-device binding per mapped layer.")
    checked: list[tuple[torch.Tensor, torch.Tensor]] = []
    for binding, layer in zip(bindings, mapping.layers):
        state = getattr(binding, "state", None)
        if not isinstance(state, torch.Tensor):
            raise TypeError("Expected each binding to expose a tensor state.")
        offset = (
            layer.continuous_offset
            if endpoint == "continuous"
            else layer.quantized_offset
        )
        target = (
            layer.continuous_conductance
            if endpoint == "continuous"
            else layer.quantized_conductance
        )
        if state.shape != target.shape:
            raise ValueError("Expected mapped and destination tensor shapes to match.")
        if not torch.equal(target, layer.baseline + offset):
            raise ValueError("Expected exact stored G=B+d decomposition.")
        lower = mapping.conductance_min + (
            mapping.conductance_max - mapping.conductance_min
        ) * layer.cell_lower_unit
        upper = mapping.conductance_min + (
            mapping.conductance_max - mapping.conductance_min
        ) * layer.cell_upper_unit
        tolerance = max(
            1e-12,
            (mapping.conductance_max - mapping.conductance_min) * 1e-6,
        )
        if (
            not bool(torch.isfinite(target).all())
            or bool(torch.any(target < -tolerance))
            or bool(torch.any(target < lower - tolerance))
            or bool(torch.any(target > upper + tolerance))
        ):
            raise ValueError("Expected finite, nonnegative, per-cell in-bound full G.")
        checked.append((state, target))
    with torch.no_grad():
        for state, target in checked:
            state.copy_(target.to(device=state.device, dtype=state.dtype))


@dataclass(frozen=True)
class JointRepairPlan:
    layouts: tuple[str, ...]
    attempts_by_layer: tuple[torch.Tensor, ...]
    failure_layer_index: torch.Tensor
    failure_flat_index: torch.Tensor
    selected_attempt: torch.Tensor
    maximum_attempts: int

    @property
    def failure_count(self) -> int:
        return int(self.selected_attempt.numel())

    @property
    def selected_donor_flat_index(self) -> torch.Tensor:
        """Return indices into the concatenated private donor blocks."""

        return (
            torch.arange(self.failure_count, dtype=torch.int64)
            * self.maximum_attempts
            + self.selected_attempt
        )


def build_joint_repair_plan(
    base_feasible_by_layer: Sequence[torch.Tensor],
    donor_feasible: torch.Tensor,
    *,
    layouts: Sequence[str] = LAYOUTS,
    maximum_attempts: int = 128,
) -> JointRepairPlan:
    """Choose the first feasible candidate in each destination's donor block."""

    masks = tuple(value.detach().to(device="cpu", dtype=torch.bool) for value in base_feasible_by_layer)
    selected_layouts = tuple(layouts)
    if (
        not masks
        or len(masks) != len(selected_layouts)
        or any(value.ndim != 2 or value.numel() == 0 for value in masks)
        or any(layout not in LAYOUTS for layout in selected_layouts)
        or isinstance(maximum_attempts, bool)
        or not isinstance(maximum_attempts, int)
        or maximum_attempts < 1
    ):
        raise ValueError("Expected valid per-layer feasibility masks and layouts.")
    failure_layers: list[int] = []
    failure_flats: list[int] = []
    attempts_by_layer = []
    for layer_index, mask in enumerate(masks):
        attempt = torch.full(mask.shape, -1, dtype=torch.int64)
        failures = torch.where(~mask.reshape(-1))[0]
        failure_layers.extend([layer_index] * int(failures.numel()))
        failure_flats.extend(int(value) for value in failures.tolist())
        attempts_by_layer.append(attempt)
    failure_count = len(failure_layers)
    candidates = donor_feasible.detach().to(device="cpu", dtype=torch.bool)
    if candidates.shape != (failure_count, maximum_attempts):
        raise ValueError(
            "Expected donor feasibility shape "
            f"{(failure_count, maximum_attempts)!r}; provided "
            f"{tuple(candidates.shape)!r}."
        )
    if failure_count:
        has_candidate = candidates.any(dim=1)
        if not bool(has_candidate.all()):
            missing = torch.where(~has_candidate)[0]
            raise DonorExhaustionError(
                "No feasible donor within the frozen attempt budget for "
                f"failure indices {missing.tolist()!r}."
            )
        selected = candidates.to(torch.int8).argmax(dim=1).to(torch.int64)
    else:
        selected = torch.empty((0,), dtype=torch.int64)
    for failure_index, (layer_index, flat_index) in enumerate(
        zip(failure_layers, failure_flats)
    ):
        attempts_by_layer[layer_index].reshape(-1)[flat_index] = selected[
            failure_index
        ]
    return JointRepairPlan(
        layouts=selected_layouts,
        attempts_by_layer=tuple(attempts_by_layer),
        failure_layer_index=torch.tensor(failure_layers, dtype=torch.int64),
        failure_flat_index=torch.tensor(failure_flats, dtype=torch.int64),
        selected_attempt=selected,
        maximum_attempts=maximum_attempts,
    )


def apply_joint_repair_field(
    base_layers: Sequence[torch.Tensor],
    donor_quads: torch.Tensor,
    plan: JointRepairPlan,
) -> tuple[torch.Tensor, ...]:
    """Apply one plan to an identity or commissioning field.

    Calling this function for every device tensor guarantees that bounds,
    pulse parameters, references, corruption provenance, RESET means, and
    standard errors move together under the same plan.  A field may carry
    trailing per-cell dimensions (for example, eight commissioning reads);
    its first two dimensions must remain the physical rail matrix.
    """

    bases = tuple(value.detach().cpu() for value in base_layers)
    if len(bases) != len(plan.layouts):
        raise ValueError("Expected one base field per repair-plan layer.")
    candidates = donor_quads.detach().cpu()
    if (
        candidates.ndim < 3
        or candidates.shape[:3]
        != (plan.failure_count, plan.maximum_attempts, 4)
    ):
        raise ValueError("Expected one disjoint donor block per failed destination.")
    if any(
        value.ndim < 2
        or value.dtype != candidates.dtype
        or value.shape[2:] != candidates.shape[3:]
        for value in bases
    ):
        raise ValueError("Expected base and donor field dtypes to match exactly.")
    quad_layers = [
        _quad_stack_field(value, layout=layout).clone()
        for value, layout in zip(bases, plan.layouts)
    ]
    for failure_index in range(plan.failure_count):
        layer_index = int(plan.failure_layer_index[failure_index].item())
        flat_index = int(plan.failure_flat_index[failure_index].item())
        attempt = int(plan.selected_attempt[failure_index].item())
        target = quad_layers[layer_index].reshape(
            -1, 4, *candidates.shape[3:]
        )[flat_index]
        target.copy_(candidates[failure_index, attempt])
    return tuple(
        _scatter_quad_field(quads, shape=base.shape, layout=layout)
        for quads, base, layout in zip(quad_layers, bases, plan.layouts)
    )


def fit_weight_reconstruction_scales(
    source_weights: Sequence[torch.Tensor],
    development_mapping: PhysicalMapping,
) -> tuple[float, ...]:
    """Fit one positive zero-intercept physical scale per layer on development."""

    weights = tuple(source_weights)
    if len(weights) != len(development_mapping.layers):
        raise ValueError("Expected source weights and mapped layers to match.")
    scales = []
    for weight, layer in zip(weights, development_mapping.layers):
        normalized, _maximum = _normalized(weight)
        contrast = (
            layer.continuous_contrast - layer.baseline_contrast
        ).to(torch.float64)
        if contrast.shape != normalized.shape:
            raise ValueError("Expected logical source and mapped contrast shapes to match.")
        denominator = normalized.square().sum()
        scale = float((normalized * contrast).sum().item() / denominator.item())
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("Expected a finite positive development analysis scale.")
        scales.append(scale)
    return tuple(scales)


def _error_summary(error: torch.Tensor) -> Mapping[str, float]:
    value = error.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    absolute = value.abs()
    return {
        "signed_mean": float(value.mean().item()),
        "mae": float(absolute.mean().item()),
        "rmse": float(value.square().mean().sqrt().item()),
        "p50_absolute": float(torch.quantile(absolute, 0.5).item()),
        "p95_absolute": float(torch.quantile(absolute, 0.95).item()),
        "maximum_absolute": float(absolute.max().item()),
    }


@dataclass(frozen=True)
class LayerWeightError:
    layer_index: int
    analysis_scale: float
    reconstructed_continuous_weight: torch.Tensor
    reconstructed_quantized_weight: torch.Tensor
    baseline_error: torch.Tensor
    envelope_error: torch.Tensor
    continuous_total_error: torch.Tensor
    quantization_error: torch.Tensor
    total_error: torch.Tensor
    quantization_error_levels: torch.Tensor
    report: Mapping[str, Any]


def build_weight_error_decomposition(
    source_weights: Sequence[torch.Tensor],
    mapping: PhysicalMapping,
    analysis_scales: Sequence[float],
) -> tuple[LayerWeightError, ...]:
    """Reconstruct source weights from full conductances and split the error."""

    weights = tuple(source_weights)
    scales = tuple(float(value) for value in analysis_scales)
    if len(weights) != len(mapping.layers) or len(scales) != len(mapping.layers):
        raise ValueError("Expected one source weight and frozen scale per layer.")
    result = []
    for weight, layer, scale in zip(weights, mapping.layers, scales):
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("Expected finite positive frozen analysis scales.")
        source = weight.detach().to(device="cpu", dtype=torch.float64)
        normalized, maximum = _normalized(source)
        c0 = layer.baseline_contrast.to(torch.float64)
        cc = layer.continuous_contrast.to(torch.float64)
        cq = layer.quantized_contrast.to(torch.float64)
        reconstructed_continuous = maximum * cc / scale
        reconstructed_quantized = maximum * cq / scale
        baseline_error = maximum * c0 / scale
        envelope_error = maximum * ((cc - c0) / scale - normalized)
        continuous_total_error = maximum * (cc / scale - normalized)
        quantization_error = maximum * (cq - cc) / scale
        total_error = maximum * (cq / scale - normalized)
        residual = total_error - (
            baseline_error + envelope_error + quantization_error
        )
        tolerance = max(1e-12, float(source.abs().max().item()) * 1e-6)
        if float(residual.abs().max().item()) > tolerance:
            raise RuntimeError("Expected exact baseline/envelope/quantization decomposition.")
        target_norm = float(source.norm().item())
        continuous_norm = float(reconstructed_continuous.norm().item())
        quantized_norm = float(reconstructed_quantized.norm().item())
        continuous_cosine = (
            float((source * reconstructed_continuous).sum().item())
            / (target_norm * continuous_norm)
            if target_norm > 0.0 and continuous_norm > 0.0
            else 0.0
        )
        quantized_cosine = (
            float((source * reconstructed_quantized).sum().item())
            / (target_norm * quantized_norm)
            if target_norm > 0.0 and quantized_norm > 0.0
            else 0.0
        )
        sign_tolerance = max(
            1e-12, 1e-9 * float(layer.level_spacing_physical)
        )
        nonzero = source != 0.0
        continuous_erased = nonzero & (cc.abs() <= sign_tolerance)
        continuous_opposite = (
            nonzero & ~continuous_erased & (torch.sign(cc) != torch.sign(source))
        )
        quantized_erased = nonzero & (cq.abs() <= sign_tolerance)
        quantized_opposite = (
            nonzero & ~quantized_erased & (torch.sign(cq) != torch.sign(source))
        )
        quant_levels = (cq - cc) / float(layer.level_spacing_physical)
        raw = {
            "baseline": _error_summary(c0),
            "envelope": _error_summary((cc - c0) - scale * normalized),
            "continuous_total": _error_summary(cc - scale * normalized),
            "quantization": _error_summary(cq - cc),
            "standard4delta_total": _error_summary(cq - scale * normalized),
        }
        conductance_span = mapping.conductance_max - mapping.conductance_min
        normalized_conductance_units = {
            "baseline": _error_summary(c0 / conductance_span),
            "envelope": _error_summary(
                ((cc - c0) - scale * normalized) / conductance_span
            ),
            "continuous_total": _error_summary(
                (cc - scale * normalized) / conductance_span
            ),
            "quantization": _error_summary((cq - cc) / conductance_span),
            "standard4delta_total": _error_summary(
                (cq - scale * normalized) / conductance_span
            ),
        }
        teacher_units = {
            "baseline": _error_summary(baseline_error),
            "envelope": _error_summary(envelope_error),
            "continuous_total": _error_summary(continuous_total_error),
            "quantization": _error_summary(quantization_error),
            "standard4delta_total": _error_summary(total_error),
        }
        normalized_units = {
            "baseline": _error_summary(c0 / scale),
            "envelope": _error_summary((cc - c0) / scale - normalized),
            "continuous_total": _error_summary(cc / scale - normalized),
            "quantization": _error_summary((cq - cc) / scale),
            "standard4delta_total": _error_summary(cq / scale - normalized),
        }
        report = {
            "layer": layer.layer_index,
            "analysis_scale": scale,
            "raw_contrast_error": raw,
            "normalized_conductance_coordinate_error": (
                normalized_conductance_units
            ),
            "normalized_relu_weight_error": normalized_units,
            "teacher_weight_error": teacher_units,
            "continuous_endpoint": {
                "relative_l2": (
                    float(continuous_total_error.norm().item()) / target_norm
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
            "standard4delta_endpoint": {
                "relative_l2": (
                    float(total_error.norm().item()) / target_norm
                    if target_norm > 0.0
                    else 0.0
                ),
                "cosine_similarity": quantized_cosine,
                "opposite_sign_count": int(quantized_opposite.sum().item()),
                "erased_to_zero_count": int(quantized_erased.sum().item()),
                "sign_error_union_count": int(
                    (quantized_opposite | quantized_erased).sum().item()
                ),
            },
            "nonzero_source_count": int(nonzero.sum().item()),
            "quantization_error_in_levels": _error_summary(quant_levels),
            "component_sum_max_abs_residual": float(residual.abs().max().item()),
        }
        result.append(
            LayerWeightError(
                layer_index=layer.layer_index,
                analysis_scale=scale,
                reconstructed_continuous_weight=reconstructed_continuous.to(torch.float32),
                reconstructed_quantized_weight=reconstructed_quantized.to(torch.float32),
                baseline_error=baseline_error.to(torch.float32),
                envelope_error=envelope_error.to(torch.float32),
                continuous_total_error=continuous_total_error.to(torch.float32),
                quantization_error=quantization_error.to(torch.float32),
                total_error=total_error.to(torch.float32),
                quantization_error_levels=quant_levels.to(torch.float32),
                report=report,
            )
        )
    return tuple(result)


def hardware_instance_fingerprint(
    metadata: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]
) -> str:
    """Hash only caller-declared semantic hardware metadata and tensor state."""

    digest = sha256()
    digest.update(
        json.dumps(
            dict(metadata),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name in sorted(tensors):
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_sha256(tensors[name]).encode("ascii"))
    return digest.hexdigest()


def save_physical_mapping(
    path: Path,
    mapping: PhysicalMapping,
    *,
    assignment_seed: int,
    assignment_role: str,
    hardware_instance_id: str,
) -> Mapping[str, Any]:
    """Persist a pickle-free full-``G`` mapping artifact and receipt."""

    destination = path.expanduser().resolve()
    if destination.suffix != ".npz":
        raise ValueError("Expected physical mapping artifact path ending in .npz.")
    if not assignment_role or not hardware_instance_id:
        raise ValueError("Expected assignment role and hardware instance ID.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(MAPPING_SCHEMA),
        "schema_version": np.asarray(MAPPING_SCHEMA_VERSION, dtype=np.int64),
        "policy": np.asarray(mapping.policy),
        "assignment_seed": np.asarray(int(assignment_seed), dtype=np.int64),
        "assignment_role": np.asarray(assignment_role),
        "hardware_instance_id": np.asarray(hardware_instance_id),
        "conductance_min": np.asarray(mapping.conductance_min, dtype=np.float64),
        "conductance_max": np.asarray(mapping.conductance_max, dtype=np.float64),
    }
    tensor_hashes: dict[str, str] = {}
    for layer in mapping.layers:
        for field in fields(layer):
            value = getattr(layer, field.name)
            if not isinstance(value, torch.Tensor):
                continue
            name = f"layer_{layer.layer_index}_{field.name}"
            arrays[name] = value.detach().contiguous().cpu().numpy()
            tensor_hashes[name] = _tensor_sha256(value)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    receipt_path = destination.with_suffix(".receipt.json")
    receipt = {
        "schema": MAPPING_SCHEMA,
        "schema_version": MAPPING_SCHEMA_VERSION,
        "policy": mapping.policy,
        "assignment_seed": int(assignment_seed),
        "assignment_role": assignment_role,
        "hardware_instance_id": hardware_instance_id,
        "artifact": destination.name,
        "artifact_sha256": sha256_file(destination),
        "tensor_hashes": tensor_hashes,
        "mapping": dict(mapping.report),
    }
    atomic_write_json(receipt_path, receipt)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "hardware_instance_id": hardware_instance_id,
    }


__all__ = [
    "BASELINE_POLICIES",
    "DonorExhaustionError",
    "INDEPENDENT_CELL_RESET_MEAN",
    "InfeasibleBaselineError",
    "JointRepairPlan",
    "LAYOUTS",
    "LayerPhysicalMapping",
    "LayerWeightError",
    "MAPPING_SCHEMA",
    "MAPPING_SCHEMA_VERSION",
    "PhysicalMapping",
    "REFERENCE_ENFORCED_DESTINATION_COLUMNS",
    "SHARED_DESTINATION_COLUMNS_RESET_MAX",
    "SHARED_QUAD_RESET_MAX",
    "all_policy_feasibility_from_quads",
    "apply_checked_physical_targets",
    "apply_joint_repair_field",
    "baseline_group_violation_mask",
    "build_joint_repair_plan",
    "build_layer_physical_mapping",
    "build_physical_mapping",
    "build_weight_error_decomposition",
    "fit_weight_reconstruction_scales",
    "hardware_instance_fingerprint",
    "policy_baseline_quads",
    "quad_column_loading",
    "quad_contrast",
    "quad_loading",
    "quad_row_loading",
    "quad_stack",
    "save_physical_mapping",
    "scatter_quads",
]
