"""Matched baseline-position and uniform-spacing mapping for IBM OM P&V.

This module extends the completed four-device baseline-selection mapper without
changing that experiment's frozen semantics.  The physical zero is shared by
the two cells in each destination column, every initialized offset is
non-negative, and every circuit target remains the complete conductance
``G = B + n h``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

import torch

from experiments.mnist_relu_drn.components import apply_targets
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    LayerPhysicalMapping,
    PhysicalMapping,
    SHARED_DESTINATION_COLUMNS_RESET_MAX,
    build_physical_mapping,
    quad_column_loading,
    quad_contrast,
    quad_loading,
    quad_row_loading,
    quad_stack,
    scatter_quads,
)
from training.ibm_reram_raw_active_program_verify import (
    raw_active_x_to_full_conductance,
)


BASELINE_POSITIONS = (0.0, 0.25, 0.5)
SPACING_DELTA_MULTIPLES = (1, 2, 4)

# One frozen translation for the corrected replay of the completed
# development/held-out identity cohort.  The origin sits 1e-6 raw-x below the
# smallest sampled native lower bound, so even that exact lower state maps to
# a strictly positive conductance.  The slope is the historical 110 uS per
# raw-x unit; unlike the rejected public-coordinate mapping, it is not used as
# an x=[0,1] ceiling.
UNCLIPPED_STUDY_RAW_X_MINIMUM = -1.3759238719940186
UNCLIPPED_STUDY_RAW_X_MAXIMUM = 1.8474750518798828
UNCLIPPED_STUDY_RAW_X_ORIGIN_MARGIN = 1e-6
UNCLIPPED_STUDY_RAW_X_ORIGIN = (
    UNCLIPPED_STUDY_RAW_X_MINIMUM - UNCLIPPED_STUDY_RAW_X_ORIGIN_MARGIN
)
UNCLIPPED_STUDY_CONDUCTANCE_PER_RAW_X = 1.10e-4
UNCLIPPED_STUDY_CONDUCTANCE_CEILING = (
    UNCLIPPED_STUDY_CONDUCTANCE_PER_RAW_X
    * (UNCLIPPED_STUDY_RAW_X_MAXIMUM - UNCLIPPED_STUDY_RAW_X_ORIGIN)
)


def _tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    digest = sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(
        json.dumps(list(cpu.shape), separators=(",", ":")).encode("utf-8")
    )
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


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


@dataclass(frozen=True)
class LayerDirectionalCapacity:
    """Common destination-column headroom around one selected baseline."""

    layer_index: int
    layout: str
    group_lower_unit: torch.Tensor
    group_upper_unit: torch.Tensor
    group_baseline_unit: torch.Tensor
    downward_level_capacity: torch.Tensor
    upward_level_capacity: torch.Tensor


@dataclass(frozen=True)
class RawXConductanceEmbedding:
    """One study-wide affine raw-device-coordinate embedding.

    The physical conversion is exactly

    ``G = conductance_per_raw_x * (x_raw - raw_x_origin)``.

    ``conductance_ceiling`` is a fail-closed circuit capacity, not a
    normalization slope and not permission to clip an endpoint.
    """

    raw_x_origin: float
    conductance_per_raw_x: float
    conductance_ceiling: float

    def __post_init__(self) -> None:
        values = (
            float(self.raw_x_origin),
            float(self.conductance_per_raw_x),
            float(self.conductance_ceiling),
        )
        if (
            not all(math.isfinite(value) for value in values)
            or self.conductance_per_raw_x <= 0.0
            or self.conductance_ceiling <= 0.0
        ):
            raise ValueError(
                "Expected a finite raw-x origin and positive finite affine "
                "conductance slope and ceiling."
            )

    @property
    def raw_x_span(self) -> float:
        return float(self.conductance_ceiling) / float(
            self.conductance_per_raw_x
        )

    @property
    def raw_x_ceiling(self) -> float:
        return float(self.raw_x_origin) + self.raw_x_span

    def to_full_conductance(self, raw_x: torch.Tensor) -> torch.Tensor:
        return raw_active_x_to_full_conductance(
            raw_x,
            raw_x_origin=self.raw_x_origin,
            conductance_per_raw_x=self.conductance_per_raw_x,
            conductance_ceiling=self.conductance_ceiling,
        )

    def to_mapping_unit(self, raw_x: torch.Tensor) -> torch.Tensor:
        """Return an internal [0,1] circuit fraction without clipping."""

        full = self.to_full_conductance(raw_x)
        unit = full / float(self.conductance_ceiling)
        if bool(torch.any(unit <= 0.0)) or bool(torch.any(unit > 1.0 + 1e-12)):
            raise ValueError("Raw-x endpoint leaves the affine mapping support.")
        return unit

    def from_mapping_unit(self, unit: torch.Tensor) -> torch.Tensor:
        """Invert the internal circuit fraction without projection."""

        value = torch.as_tensor(unit)
        if (
            value.numel() < 1
            or not value.is_floating_point()
            or not bool(torch.all(torch.isfinite(value)))
            or bool(torch.any(value <= 0.0))
            or bool(torch.any(value > 1.0 + 1e-12))
        ):
            raise ValueError("Expected a strictly positive in-range mapping unit.")
        raw64 = float(self.raw_x_origin) + self.raw_x_span * value.to(
            dtype=torch.float64
        )
        return raw64.to(dtype=value.dtype)

    def report(self) -> Mapping[str, float | str]:
        return {
            "policy": "study_wide_affine_raw_x_translation_no_clipping",
            "formula": "G=conductance_per_raw_x*(x_raw-raw_x_origin)",
            "raw_x_origin": float(self.raw_x_origin),
            "raw_x_ceiling": self.raw_x_ceiling,
            "conductance_per_raw_x": float(self.conductance_per_raw_x),
            "conductance_ceiling": float(self.conductance_ceiling),
        }


@dataclass(frozen=True)
class BaselineSpacingMapping:
    """One full two-layer physical map plus its baseline/spacing decision."""

    baseline_position_fraction: float
    spacing_delta_multiples: int
    nominal_dw_min: float
    physical: PhysicalMapping
    directional_capacity: tuple[LayerDirectionalCapacity, ...]
    report: Mapping[str, Any]
    raw_x_embedding: RawXConductanceEmbedding | None = None
    raw_x_baselines: tuple[torch.Tensor, ...] | None = None
    raw_x_ideal_targets: tuple[torch.Tensor, ...] | None = None

    @property
    def ideal_targets(self) -> tuple[torch.Tensor, ...]:
        return self.physical.quantized_targets

    @property
    def continuous_targets(self) -> tuple[torch.Tensor, ...]:
        return self.physical.continuous_targets

    @property
    def ideal_target_units(self) -> tuple[torch.Tensor, ...]:
        if self.raw_x_ideal_targets is not None:
            return tuple(value.clone() for value in self.raw_x_ideal_targets)
        return tuple(
            (
                layer.baseline_unit.to(torch.float64)
                + layer.integer_level_number.to(torch.float64)
                * layer.level_spacing_unit
            ).to(torch.float32)
            for layer in self.physical.layers
        )

    @property
    def baseline_raw_x(self) -> tuple[torch.Tensor, ...]:
        """Return shared full-rail baselines in the controller's raw-x frame."""

        if self.raw_x_baselines is not None:
            return tuple(value.clone() for value in self.raw_x_baselines)
        return tuple(layer.baseline_unit.clone() for layer in self.physical.layers)


def _destination_group_values(
    reset_quad: torch.Tensor,
    upper_quad: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lower = torch.stack(
        (
            torch.maximum(reset_quad[..., 0], reset_quad[..., 2]),
            torch.maximum(reset_quad[..., 1], reset_quad[..., 3]),
        ),
        dim=-1,
    )
    upper = torch.stack(
        (
            torch.minimum(upper_quad[..., 0], upper_quad[..., 2]),
            torch.minimum(upper_quad[..., 1], upper_quad[..., 3]),
        ),
        dim=-1,
    )
    if bool(torch.any(lower > upper)):
        raise ValueError("Expected a non-empty commissioned destination window.")
    baseline = lower + float(alpha) * (upper - lower)
    return lower, upper, baseline


def _expanded_destination_groups(group: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (group[..., 0], group[..., 1], group[..., 0], group[..., 1]),
        dim=-1,
    )


def _remap_layer(
    base: LayerPhysicalMapping,
    *,
    logical_weight: torch.Tensor,
    baseline_position_fraction: float,
    spacing_delta_multiples: int,
    nominal_dw_min: float,
    conductance_min: float,
    conductance_max: float,
) -> tuple[LayerPhysicalMapping, LayerDirectionalCapacity]:
    alpha = float(baseline_position_fraction)
    multiple = int(spacing_delta_multiples)
    spacing_unit = multiple * (float(nominal_dw_min) / 2.0)
    span = float(conductance_max) - float(conductance_min)
    spacing_physical = span * spacing_unit

    lower = base.cell_lower_unit.to(torch.float64)
    upper = base.cell_upper_unit.to(torch.float64)
    reset = base.reset_baseline_unit.to(torch.float64)
    lower_quad = quad_stack(lower, layout=base.layout)
    upper_quad = quad_stack(upper, layout=base.layout)
    reset_quad = quad_stack(reset, layout=base.layout)
    group_lower, group_upper, group_baseline = _destination_group_values(
        reset_quad, upper_quad, alpha
    )
    baseline_quad = _expanded_destination_groups(group_baseline)
    feasible = (baseline_quad >= lower_quad).all(dim=-1) & (
        baseline_quad <= upper_quad
    ).all(dim=-1)
    if not bool(feasible.all()):
        raise ValueError("Expected every selected destination baseline in bounds.")

    active_quad = quad_stack(base.active_mask, layout=base.layout).to(torch.bool)
    source = logical_weight.detach().to(device="cpu", dtype=torch.float64)
    if source.shape != base.source_weight.shape or not bool(
        torch.all(torch.isfinite(source))
    ):
        raise ValueError("Expected the original finite logical source matrix.")
    source_absmax = float(source.abs().max().item())
    if source_absmax <= 0.0 or source_absmax != base.teacher_absmax:
        raise ValueError("Expected the remap source to match the base mapping.")
    # Do not reconstruct from the stored float32 normalized diagnostic: doing
    # so changes continuous offsets by a few ULPs.  The original source is
    # required for tensor-exact alpha=0, h=4 parity with the finalized mapper.
    normalized = source / source_absmax
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
    continuous_level = (
        float(base.scale_fraction) * selected_headroom * normalized.abs()
    )
    continuous_offset_quad = (
        active_quad.to(torch.float64) * continuous_level.unsqueeze(-1)
    )
    positive_capacity = torch.floor(
        positive_headroom / spacing_unit + 1e-12
    ).to(torch.int64)
    negative_capacity = torch.floor(
        negative_headroom / spacing_unit + 1e-12
    ).to(torch.int64)
    selected_capacity = torch.where(
        positive, positive_capacity, negative_capacity
    )
    selected_level = torch.floor(continuous_level / spacing_unit + 0.5).to(
        torch.int64
    )
    selected_level = torch.minimum(selected_level, selected_capacity)
    integer_quad = active_quad.to(torch.int64) * selected_level.unsqueeze(-1)
    quantized_offset_quad = integer_quad.to(torch.float64) * spacing_unit

    physical_shape = tuple(base.baseline.shape)
    baseline_unit = scatter_quads(
        baseline_quad, shape=physical_shape, layout=base.layout
    )
    continuous_offset_unit = scatter_quads(
        continuous_offset_quad, shape=physical_shape, layout=base.layout
    )
    quantized_offset_unit = scatter_quads(
        quantized_offset_quad, shape=physical_shape, layout=base.layout
    )
    integer_level = scatter_quads(
        integer_quad, shape=physical_shape, layout=base.layout
    )
    baseline = (float(conductance_min) + span * baseline_unit).to(torch.float32)
    continuous_offset = (span * continuous_offset_unit).to(torch.float32)
    quantized_offset = (span * quantized_offset_unit).to(torch.float32)
    continuous = baseline + continuous_offset
    quantized = baseline + quantized_offset
    physical_lower = (float(conductance_min) + span * lower).to(torch.float32)
    physical_upper = (float(conductance_min) + span * upper).to(torch.float32)
    tolerance = max(1e-12, span * 1e-6)
    for name, target in (("continuous", continuous), ("ideal", quantized)):
        if (
            not bool(torch.isfinite(target).all())
            or bool(torch.any(target < -tolerance))
            or bool(torch.any(target < physical_lower - tolerance))
            or bool(torch.any(target > physical_upper + tolerance))
        ):
            raise RuntimeError(f"Expected an in-bound full-{name} G tensor.")

    remapped = replace(
        base,
        level_spacing_unit=spacing_unit,
        level_spacing_physical=spacing_physical,
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
        baseline_contrast=quad_contrast(baseline, layout=base.layout),
        continuous_contrast=quad_contrast(continuous, layout=base.layout),
        quantized_contrast=quad_contrast(quantized, layout=base.layout),
        baseline_loading=quad_loading(baseline, layout=base.layout),
        continuous_loading=quad_loading(continuous, layout=base.layout),
        quantized_loading=quad_loading(quantized, layout=base.layout),
        baseline_row_loading=quad_row_loading(baseline, layout=base.layout),
        baseline_column_loading=quad_column_loading(
            baseline, layout=base.layout
        ),
        continuous_row_loading=quad_row_loading(
            continuous, layout=base.layout
        ),
        continuous_column_loading=quad_column_loading(
            continuous, layout=base.layout
        ),
        quantized_row_loading=quad_row_loading(
            quantized, layout=base.layout
        ),
        quantized_column_loading=quad_column_loading(
            quantized, layout=base.layout
        ),
    )
    down_capacity = torch.floor(
        (group_baseline - group_lower) / spacing_unit + 1e-12
    ).to(torch.int64)
    up_capacity = torch.floor(
        (group_upper - group_baseline) / spacing_unit + 1e-12
    ).to(torch.int64)
    directional = LayerDirectionalCapacity(
        layer_index=base.layer_index,
        layout=base.layout,
        group_lower_unit=group_lower.to(torch.float32),
        group_upper_unit=group_upper.to(torch.float32),
        group_baseline_unit=group_baseline.to(torch.float32),
        downward_level_capacity=down_capacity,
        upward_level_capacity=up_capacity,
    )
    return remapped, directional


def build_baseline_spacing_mapping(
    logical_weights: Sequence[torch.Tensor],
    *,
    baseline_position_fraction: float,
    spacing_delta_multiples: int,
    scale_fractions: Sequence[float],
    nominal_dw_min: float,
    conductance_min: float,
    conductance_max: float,
    cell_lower_units: Sequence[torch.Tensor],
    cell_upper_units: Sequence[torch.Tensor],
    reset_baseline_units: Sequence[torch.Tensor],
    intrinsic_references_native: Sequence[torch.Tensor],
) -> BaselineSpacingMapping:
    """Build one shared-destination map for an ``alpha``/spacing decision."""

    alpha = float(baseline_position_fraction)
    if alpha not in BASELINE_POSITIONS:
        raise ValueError(f"Expected alpha in {BASELINE_POSITIONS!r}.")
    if (
        isinstance(spacing_delta_multiples, bool)
        or not isinstance(spacing_delta_multiples, int)
        or spacing_delta_multiples not in SPACING_DELTA_MULTIPLES
    ):
        raise ValueError(
            f"Expected spacing in {SPACING_DELTA_MULTIPLES!r} delta_x."
        )
    multiple = int(spacing_delta_multiples)
    base = build_physical_mapping(
        logical_weights,
        policy=SHARED_DESTINATION_COLUMNS_RESET_MAX,
        scale_fractions=scale_fractions,
        nominal_dw_min=nominal_dw_min,
        conductance_min=conductance_min,
        conductance_max=conductance_max,
        cell_lower_units=cell_lower_units,
        cell_upper_units=cell_upper_units,
        reset_baseline_units=reset_baseline_units,
        intrinsic_references_native=intrinsic_references_native,
    )
    layers_and_capacity = tuple(
        _remap_layer(
            layer,
            logical_weight=logical_weight,
            baseline_position_fraction=alpha,
            spacing_delta_multiples=multiple,
            nominal_dw_min=nominal_dw_min,
            conductance_min=conductance_min,
            conductance_max=conductance_max,
        )
        for layer, logical_weight in zip(base.layers, logical_weights)
    )
    layers = tuple(value[0] for value in layers_and_capacity)
    capacities = tuple(value[1] for value in layers_and_capacity)
    physical_report = {
        **dict(base.report),
        "policy": SHARED_DESTINATION_COLUMNS_RESET_MAX,
        "baseline_position_fraction": alpha,
        "spacing_delta_multiples": multiple,
        "quantized_diagnostic": "ideal_bounded_uniform_spacing_init_accuracy",
        "layers": [
            {
                "layer": layer.layer_index,
                "layout": layer.layout,
                "baseline_contrast": _summary(layer.baseline_contrast),
                "continuous_contrast": _summary(layer.continuous_contrast),
                "ideal_contrast": _summary(layer.quantized_contrast),
                "baseline_loading": _summary(layer.baseline_loading),
                "selected_headroom_unit": _summary(
                    layer.selected_headroom_unit
                ),
                "downward_level_capacity": _summary(
                    capacity.downward_level_capacity.to(torch.float64)
                ),
                "common_upward_level_capacity": _summary(
                    capacity.upward_level_capacity.to(torch.float64)
                ),
                "selected_level_capacity": _summary(
                    layer.selected_level_capacity.to(torch.float64)
                ),
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
            for layer, capacity in zip(layers, capacities)
        ],
    }
    physical = replace(base, layers=layers, report=physical_report)
    report = {
        "baseline_position_fraction": alpha,
        "spacing_delta_multiples": multiple,
        "delta_x": float(nominal_dw_min) / 2.0,
        "level_spacing_unit": multiple * float(nominal_dw_min) / 2.0,
        "initial_level_indices": "nonnegative_sign_selected",
        "downward_capacity_role": "latent_rewrite_diagnostic_only",
        "physical": physical_report,
    }
    return BaselineSpacingMapping(
        baseline_position_fraction=alpha,
        spacing_delta_multiples=multiple,
        nominal_dw_min=float(nominal_dw_min),
        physical=physical,
        directional_capacity=capacities,
        report=report,
    )


def build_unclipped_baseline_spacing_mapping(
    logical_weights: Sequence[torch.Tensor],
    *,
    baseline_position_fraction: float,
    spacing_delta_multiples: int,
    scale_fractions: Sequence[float],
    nominal_dw_min: float,
    raw_x_origin: float,
    conductance_per_raw_x: float,
    conductance_ceiling: float,
    cell_lower_raw_x: Sequence[torch.Tensor],
    cell_upper_raw_x: Sequence[torch.Tensor],
    reset_baseline_raw_x: Sequence[torch.Tensor],
    intrinsic_references_native: Sequence[torch.Tensor],
) -> BaselineSpacingMapping:
    """Build the corrected shared-destination map from unclipped raw ``x``.

    Baseline windows, level spacing, capacity, and P&V targets are all defined
    in the native raw coordinate ``x=(a+1)/2``.  A normalized mapping unit is
    used only as an internal compatibility representation for the established
    physical-mapping data structures.  Full conductance always follows the
    literal affine embedding and is never clipped at either end.
    """

    embedding = RawXConductanceEmbedding(
        raw_x_origin=float(raw_x_origin),
        conductance_per_raw_x=float(conductance_per_raw_x),
        conductance_ceiling=float(conductance_ceiling),
    )
    raw_sequences = (
        tuple(cell_lower_raw_x),
        tuple(cell_upper_raw_x),
        tuple(reset_baseline_raw_x),
    )
    if any(len(values) != 2 for values in raw_sequences):
        raise ValueError("Expected exactly two raw-x physical layers.")
    raw_lower, raw_upper, raw_reset = raw_sequences
    for layer_index, (lower, upper, reset) in enumerate(
        zip(raw_lower, raw_upper, raw_reset)
    ):
        matrices = tuple(
            torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
            for value in (lower, upper, reset)
        )
        lower64, upper64, reset64 = matrices
        if (
            lower64.shape != upper64.shape
            or lower64.shape != reset64.shape
            or lower64.numel() < 1
            or not all(bool(torch.isfinite(value).all()) for value in matrices)
            or bool(torch.any(upper64 < lower64))
            or bool(torch.any(reset64 < lower64))
            or bool(torch.any(reset64 > upper64))
        ):
            raise ValueError(
                f"Expected finite ordered raw-x bounds and bounded RESET in layer "
                f"{layer_index}."
            )

    native_delta = float(nominal_dw_min)
    if not math.isfinite(native_delta) or native_delta <= 0.0:
        raise ValueError("Expected a finite positive native OM dw_min.")
    normalized_delta = native_delta / embedding.raw_x_span
    lower_unit = tuple(embedding.to_mapping_unit(value) for value in raw_lower)
    upper_unit = tuple(embedding.to_mapping_unit(value) for value in raw_upper)
    reset_unit = tuple(embedding.to_mapping_unit(value) for value in raw_reset)
    mapped = build_baseline_spacing_mapping(
        logical_weights,
        baseline_position_fraction=baseline_position_fraction,
        spacing_delta_multiples=spacing_delta_multiples,
        scale_fractions=scale_fractions,
        nominal_dw_min=normalized_delta,
        conductance_min=0.0,
        conductance_max=embedding.conductance_ceiling,
        cell_lower_units=lower_unit,
        cell_upper_units=upper_unit,
        reset_baseline_units=reset_unit,
        intrinsic_references_native=intrinsic_references_native,
    )

    alpha = float(baseline_position_fraction)
    spacing_raw_x = int(spacing_delta_multiples) * native_delta / 2.0
    raw_baselines: list[torch.Tensor] = []
    raw_targets: list[torch.Tensor] = []
    tolerance_raw_x = max(1e-12, embedding.raw_x_span * 1e-6)
    tolerance_g = max(1e-12, embedding.conductance_ceiling * 1e-6)
    for layer, lower, upper, reset in zip(
        mapped.physical.layers,
        raw_lower,
        raw_upper,
        raw_reset,
    ):
        lower64 = torch.as_tensor(lower, dtype=torch.float64, device="cpu")
        upper64 = torch.as_tensor(upper, dtype=torch.float64, device="cpu")
        reset64 = torch.as_tensor(reset, dtype=torch.float64, device="cpu")
        reset_quad = quad_stack(reset64, layout=layer.layout)
        upper_quad = quad_stack(upper64, layout=layer.layout)
        _group_lower, _group_upper, group_baseline = _destination_group_values(
            reset_quad,
            upper_quad,
            alpha,
        )
        baseline_quad = _expanded_destination_groups(group_baseline)
        baseline_raw = scatter_quads(
            baseline_quad,
            shape=tuple(lower64.shape),
            layout=layer.layout,
        ).to(torch.float32)
        target_raw = (
            baseline_raw.to(torch.float64)
            + layer.integer_level_number.to(torch.float64) * spacing_raw_x
        ).to(torch.float32)
        if (
            bool(torch.any(baseline_raw.to(torch.float64) < lower64 - tolerance_raw_x))
            or bool(torch.any(baseline_raw.to(torch.float64) > upper64 + tolerance_raw_x))
            or bool(torch.any(target_raw.to(torch.float64) < lower64 - tolerance_raw_x))
            or bool(torch.any(target_raw.to(torch.float64) > upper64 + tolerance_raw_x))
        ):
            raise RuntimeError("Corrected raw-x baseline or target left native support.")
        baseline_g = embedding.to_full_conductance(baseline_raw)
        target_g = embedding.to_full_conductance(target_raw)
        if (
            not torch.allclose(
                baseline_g,
                layer.baseline,
                rtol=0.0,
                atol=tolerance_g,
            )
            or not torch.allclose(
                target_g,
                layer.quantized_conductance,
                rtol=0.0,
                atol=tolerance_g,
            )
        ):
            raise RuntimeError(
                "Internal mapping units do not reproduce the no-clipping affine G."
            )
        raw_baselines.append(baseline_raw)
        raw_targets.append(target_raw)

    report = {
        **dict(mapped.report),
        "coordinate": "raw_x=(native_raw_active_a+1)/2_unclipped",
        "delta_x": native_delta / 2.0,
        "level_spacing_unit": spacing_raw_x,
        "conductance_embedding": dict(embedding.report()),
        "internal_mapping_coordinate": (
            "z=G/conductance_ceiling; compatibility_only_not_device_coordinate"
        ),
        "internal_mapping_delta_x": normalized_delta / 2.0,
        "out_of_bounds_policy": "fail_no_projection_no_clipping",
    }
    return replace(
        mapped,
        nominal_dw_min=native_delta,
        report=report,
        raw_x_embedding=embedding,
        raw_x_baselines=tuple(raw_baselines),
        raw_x_ideal_targets=tuple(raw_targets),
    )


def apply_full_conductance_targets(
    catalog: Any,
    targets: Sequence[torch.Tensor],
    *,
    conductance_min: float,
    conductance_max: float,
) -> None:
    """Fail closed before applying full physical conductances to a DRN."""

    selected = tuple(targets)
    bindings = tuple(catalog.trainable)
    if len(selected) != len(bindings):
        raise ValueError("Expected one full-G target per trainable tensor.")
    tolerance = max(1e-12, (conductance_max - conductance_min) * 1e-6)
    for binding, target in zip(bindings, selected):
        if target.shape != binding.state.shape:
            raise ValueError("Expected full-G target shape to match its binding.")
        if (
            not bool(torch.isfinite(target).all())
            or bool(torch.any(target < conductance_min - tolerance))
            or bool(torch.any(target > conductance_max + tolerance))
        ):
            finite = target[torch.isfinite(target)]
            observed = (
                (float(finite.min().item()), float(finite.max().item()))
                if finite.numel()
                else (None, None)
            )
            raise ValueError(
                "Expected finite nonnegative in-range full G; "
                f"observed_min_max={observed!r}, declared="
                f"({conductance_min!r}, {conductance_max!r})."
            )
    apply_targets(catalog, selected)


def persistent_weight_error_report(
    source_weights: Sequence[torch.Tensor],
    persistent_conductances: Sequence[torch.Tensor],
    analysis_scales: Sequence[float],
    *,
    layouts: Sequence[str] = ("halves", "paired"),
) -> tuple[Mapping[str, Any], ...]:
    """Report ReLU-weight error reconstructed from persistent full G."""

    if not (
        len(source_weights)
        == len(persistent_conductances)
        == len(analysis_scales)
        == len(layouts)
        == 2
    ):
        raise ValueError("Expected two matched source and persistent layers.")
    reports = []
    for index, (source_raw, conductance, scale, layout) in enumerate(
        zip(source_weights, persistent_conductances, analysis_scales, layouts)
    ):
        source = source_raw.detach().to(device="cpu", dtype=torch.float64)
        maximum = float(source.abs().max().item())
        if maximum <= 0.0 or not math.isfinite(scale) or float(scale) <= 0.0:
            raise ValueError("Expected nonzero source weights and positive scales.")
        contrast = quad_contrast(
            conductance.detach().to(device="cpu", dtype=torch.float64),
            layout=layout,
        )
        reconstructed = maximum * contrast / float(scale)
        error = reconstructed - source
        source_norm = float(source.norm().item())
        reconstructed_norm = float(reconstructed.norm().item())
        cosine = (
            float((source * reconstructed).sum().item())
            / (source_norm * reconstructed_norm)
            if source_norm > 0.0 and reconstructed_norm > 0.0
            else 0.0
        )
        nonzero = source != 0.0
        erased = nonzero & (contrast == 0.0)
        opposite = nonzero & ~erased & (torch.sign(contrast) != torch.sign(source))
        absolute = error.abs().reshape(-1)
        reports.append(
            {
                "layer": index,
                "analysis_scale": float(scale),
                "signed_mean": float(error.mean().item()),
                "mae": float(absolute.mean().item()),
                "rmse": float(error.square().mean().sqrt().item()),
                "p50_absolute": float(torch.quantile(absolute, 0.5).item()),
                "p95_absolute": float(torch.quantile(absolute, 0.95).item()),
                "maximum_absolute": float(absolute.max().item()),
                "relative_l2": (
                    float(error.norm().item()) / source_norm
                    if source_norm > 0.0
                    else 0.0
                ),
                "cosine_similarity": cosine,
                "opposite_sign_count": int(opposite.sum().item()),
                "erased_to_zero_count": int(erased.sum().item()),
                "persistent_contrast_sha256": _tensor_sha256(contrast),
                "reconstructed_weight_sha256": _tensor_sha256(reconstructed),
            }
        )
    return tuple(reports)


__all__ = [
    "BASELINE_POSITIONS",
    "SPACING_DELTA_MULTIPLES",
    "BaselineSpacingMapping",
    "LayerDirectionalCapacity",
    "RawXConductanceEmbedding",
    "UNCLIPPED_STUDY_CONDUCTANCE_CEILING",
    "UNCLIPPED_STUDY_CONDUCTANCE_PER_RAW_X",
    "UNCLIPPED_STUDY_RAW_X_MAXIMUM",
    "UNCLIPPED_STUDY_RAW_X_MINIMUM",
    "UNCLIPPED_STUDY_RAW_X_ORIGIN",
    "UNCLIPPED_STUDY_RAW_X_ORIGIN_MARGIN",
    "apply_full_conductance_targets",
    "build_baseline_spacing_mapping",
    "build_unclipped_baseline_spacing_mapping",
    "persistent_weight_error_report",
]
