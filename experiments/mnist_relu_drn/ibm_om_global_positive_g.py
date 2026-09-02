"""Array-independent signed-master mapping to four positive DRN cells.

Each normalized logical ReLU weight ``u in [-1,1]`` is represented by the
canonical physical quad ``(++,+-,-+,--)`` as

``G = 2 * (relu(u), relu(-u), relu(-u), relu(u))``.

Thus every DRN parameter supplied to the circuit is a nonnegative physical
conductance in ``[0,2]``.  There is no Array-A bound, reference, baseline, or
identity in this mapping.  Array-specific shared baselines and device
assignment are introduced only later by the independent deployment mapper.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch

from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    quad_stack,
    scatter_quads,
)


GLOBAL_POSITIVE_G_LAYOUTS = ("halves", "paired")


@dataclass(frozen=True)
class GlobalPositiveGView:
    """One clean continuous or global-codebook physical DRN realization."""

    full_conductance: tuple[torch.Tensor, ...]
    realized_logical_master: tuple[torch.Tensor, ...]
    requested_level: tuple[torch.Tensor, ...] | None
    quantization_spacing_x: float | None
    mapping_report: Mapping[str, object]


def _validate_masters(
    masters: Sequence[torch.Tensor],
    layouts: Sequence[str],
) -> tuple[tuple[torch.Tensor, ...], tuple[str, ...]]:
    selected = tuple(masters)
    selected_layouts = tuple(layouts)
    if len(selected) != 2 or selected_layouts != GLOBAL_POSITIVE_G_LAYOUTS:
        raise ValueError(
            "Expected two logical masters with canonical halves/paired layouts."
        )
    for master in selected:
        if (
            not isinstance(master, torch.Tensor)
            or master.ndim != 2
            or not master.is_floating_point()
            or not bool(torch.all(torch.isfinite(master)))
            or bool(torch.any(master < -1.0))
            or bool(torch.any(master > 1.0))
        ):
            raise ValueError("Expected finite rank-two logical masters in [-1,1].")
    return selected, selected_layouts


def map_global_positive_g(
    masters: Sequence[torch.Tensor],
    *,
    quantization_spacing_x: float | None = None,
    layouts: Sequence[str] = GLOBAL_POSITIVE_G_LAYOUTS,
) -> GlobalPositiveGView:
    """Map signed logical masters into complete positive ``G in [0,2]``.

    When ``quantization_spacing_x`` is supplied, magnitude is rounded to the
    nearest global raw-``x`` level with half-away-from-zero tie handling.  The
    codebook is global and contains no cell- or array-specific capacity.
    """

    selected, selected_layouts = _validate_masters(masters, layouts)
    spacing = None
    if quantization_spacing_x is not None:
        spacing = float(quantization_spacing_x)
        if not math.isfinite(spacing) or spacing <= 0.0 or spacing > 1.0:
            raise ValueError("Expected quantization spacing in raw x within (0,1].")
    full: list[torch.Tensor] = []
    realized: list[torch.Tensor] = []
    levels: list[torch.Tensor] = []
    layer_reports: list[dict[str, object]] = []
    for layer, (master, layout) in enumerate(
        zip(selected, selected_layouts, strict=True)
    ):
        magnitude = master.abs()
        level = None
        if spacing is not None:
            level = torch.floor(magnitude / spacing + 0.5).to(torch.int64)
            maximum_level = int(math.floor(1.0 / spacing + 1e-12))
            level = torch.clamp(level, min=0, max=maximum_level)
            magnitude = torch.minimum(
                level.to(master.dtype) * spacing,
                torch.ones_like(master),
            )
            levels.append(level)
        realized_master = torch.where(master >= 0.0, magnitude, -magnitude)
        plus = torch.relu(realized_master)
        minus = torch.relu(-realized_master)
        quad_g = 2.0 * torch.stack((plus, minus, minus, plus), dim=-1)
        physical_shape = (2 * master.shape[0], 2 * master.shape[1])
        physical = scatter_quads(
            quad_g,
            shape=physical_shape,
            layout=layout,
        )
        if (
            not bool(torch.all(torch.isfinite(physical)))
            or bool(torch.any(physical < 0.0))
            or bool(torch.any(physical > 2.0))
        ):
            raise RuntimeError("Global positive-G mapping left [0,2].")
        full.append(physical)
        realized.append(realized_master)
        layer_reports.append(
            {
                "layer": layer,
                "layout": layout,
                "logical_shape": list(master.shape),
                "physical_shape": list(physical.shape),
                "minimum_g": float(physical.min().item()),
                "maximum_g": float(physical.max().item()),
                "zero_g_count": int((physical == 0.0).sum().item()),
                "physical_cells": int(physical.numel()),
                "quantized": spacing is not None,
                "quantization_spacing_x": spacing,
            }
        )
    return GlobalPositiveGView(
        full_conductance=tuple(full),
        realized_logical_master=tuple(realized),
        requested_level=None if spacing is None else tuple(levels),
        quantization_spacing_x=spacing,
        mapping_report={
            "policy": "global_four_cell_positive_conductance",
            "logical_master_bounds": [-1.0, 1.0],
            "physical_conductance_bounds": [0.0, 2.0],
            "coordinate": "x=G/2",
            "quad_order": ["G++", "G+-", "G-+", "G--"],
            "formula": "2*(relu(u),relu(-u),relu(-u),relu(u))",
            "baseline": 0.0,
            "array_specific_information_consumed": False,
            "layers": layer_reports,
        },
    )


def lift_global_positive_g_gradients(
    masters: Sequence[torch.Tensor],
    physical_gradients: Sequence[torch.Tensor],
    *,
    layouts: Sequence[str] = GLOBAL_POSITIVE_G_LAYOUTS,
) -> tuple[torch.Tensor, ...]:
    """Identity-STE lift from complete physical ``G`` back to signed masters."""

    selected, selected_layouts = _validate_masters(masters, layouts)
    gradients = tuple(physical_gradients)
    if len(gradients) != len(selected):
        raise ValueError("Expected one physical gradient per logical master.")
    result = []
    for master, physical, layout in zip(
        selected, gradients, selected_layouts, strict=True
    ):
        expected_shape = (2 * master.shape[0], 2 * master.shape[1])
        if (
            not isinstance(physical, torch.Tensor)
            or tuple(physical.shape) != expected_shape
            or not bool(torch.all(torch.isfinite(physical)))
        ):
            raise ValueError("Physical global-positive-G gradient shape mismatch.")
        quad = quad_stack(
            physical.to(device=master.device, dtype=master.dtype),
            layout=layout,
        )
        positive = 2.0 * (quad[..., 0] + quad[..., 3])
        negative = -2.0 * (quad[..., 1] + quad[..., 2])
        gradient = torch.where(master >= 0.0, positive, negative)
        if not bool(torch.all(torch.isfinite(gradient))):
            raise FloatingPointError("Global positive-G STE gradient is non-finite.")
        result.append(gradient)
    return tuple(result)


__all__ = [
    "GLOBAL_POSITIVE_G_LAYOUTS",
    "GlobalPositiveGView",
    "lift_global_positive_g_gradients",
    "map_global_positive_g",
]
