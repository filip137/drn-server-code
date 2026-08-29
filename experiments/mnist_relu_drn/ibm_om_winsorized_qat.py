"""Deterministic STE-QAT for the Winsorized IBM OM codebook.

The trainable shadow is one normalized signed logical tensor per ReLU layer.
Every forward materializes all four *complete* conductances of each DRN quad
from a frozen shared-destination baseline and a one-sided integer code.  No
reference is subtracted and no P&V endpoint noise is present in this training
primitive; persistent programming is a separate deployment evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    quad_stack,
    scatter_quads,
)


LAYOUTS = ("halves", "paired")


@dataclass(frozen=True)
class WinsorizedQatLayerTemplate:
    """Frozen cell constraints used by one logical layer's QAT quantizer."""

    layer_index: int
    layout: str
    baseline_full_g: torch.Tensor
    baseline_raw_x: torch.Tensor
    cell_upper_raw_x: torch.Tensor
    positive_headroom_raw_x: torch.Tensor
    negative_headroom_raw_x: torch.Tensor
    positive_capacity: torch.Tensor
    negative_capacity: torch.Tensor
    initial_normalized_weight: torch.Tensor
    level_spacing_raw_x: float

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return tuple(self.initial_normalized_weight.shape)

    def to(self, device: torch.device | str) -> "WinsorizedQatLayerTemplate":
        selected = torch.device(device)
        return WinsorizedQatLayerTemplate(
            layer_index=self.layer_index,
            layout=self.layout,
            baseline_full_g=self.baseline_full_g.to(selected),
            baseline_raw_x=self.baseline_raw_x.to(selected),
            cell_upper_raw_x=self.cell_upper_raw_x.to(selected),
            positive_headroom_raw_x=self.positive_headroom_raw_x.to(selected),
            negative_headroom_raw_x=self.negative_headroom_raw_x.to(selected),
            positive_capacity=self.positive_capacity.to(selected),
            negative_capacity=self.negative_capacity.to(selected),
            initial_normalized_weight=self.initial_normalized_weight.to(selected),
            level_spacing_raw_x=self.level_spacing_raw_x,
        )


@dataclass(frozen=True)
class WinsorizedQatView:
    """One deterministic quantized forward view of the logical shadow."""

    full_conductance_targets: tuple[torch.Tensor, ...]
    raw_x_targets: tuple[torch.Tensor, ...]
    requested_index: tuple[torch.Tensor, ...]
    realized_normalized_weight: tuple[torch.Tensor, ...]
    report: Mapping[str, Any]


def _tensor_summary(value: torch.Tensor) -> Mapping[str, float]:
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    return {
        "minimum": float(flat.min().item()),
        "mean": float(flat.mean().item()),
        "rms": float(flat.square().mean().sqrt().item()),
        "maximum": float(flat.max().item()),
    }


def _validate_template(template: WinsorizedQatLayerTemplate) -> None:
    if template.layout not in LAYOUTS:
        raise ValueError(f"Unsupported QAT quad layout: {template.layout!r}.")
    spacing = float(template.level_spacing_raw_x)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Expected a positive finite raw-x level spacing.")
    physical = (
        template.baseline_full_g,
        template.baseline_raw_x,
        template.cell_upper_raw_x,
    )
    if any(value.shape != physical[0].shape for value in physical):
        raise ValueError("Expected matched physical QAT template shapes.")
    if any(not bool(torch.isfinite(value).all()) for value in physical):
        raise ValueError("Expected finite QAT physical templates.")
    logical = (
        template.positive_headroom_raw_x,
        template.negative_headroom_raw_x,
        template.positive_capacity,
        template.negative_capacity,
        template.initial_normalized_weight,
    )
    if any(value.shape != logical[-1].shape for value in logical):
        raise ValueError("Expected matched logical QAT template shapes.")
    if (
        bool(torch.any(template.positive_headroom_raw_x < 0.0))
        or bool(torch.any(template.negative_headroom_raw_x < 0.0))
        or bool(torch.any(template.positive_capacity < 0))
        or bool(torch.any(template.negative_capacity < 0))
    ):
        raise ValueError("Expected nonnegative QAT headroom and capacity.")
    baseline_residual = (
        template.baseline_full_g.to(torch.float64)
        - 2.0 * template.baseline_raw_x.to(torch.float64)
    )
    if float(baseline_residual.abs().max().item()) > 2e-7:
        raise ValueError("Expected the Winsorized baseline to satisfy G=2*x.")


def load_winsorized_qat_templates(
    mapping_path: Path | str,
    *,
    level_spacing_raw_x: float,
    layouts: Sequence[str] = LAYOUTS,
) -> tuple[WinsorizedQatLayerTemplate, ...]:
    """Load the exact frozen mapping tensors needed by deterministic QAT."""

    path = Path(mapping_path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as source:
        templates = []
        for layer_index, layout in enumerate(tuple(layouts)):
            prefix = f"layer_{layer_index}_"
            template = WinsorizedQatLayerTemplate(
                layer_index=layer_index,
                layout=layout,
                baseline_full_g=torch.from_numpy(
                    np.array(source[prefix + "baseline"], copy=True)
                ).to(torch.float32),
                baseline_raw_x=torch.from_numpy(
                    np.array(source[prefix + "baseline_unit"], copy=True)
                ).to(torch.float32),
                cell_upper_raw_x=torch.from_numpy(
                    np.array(source[prefix + "cell_upper_unit"], copy=True)
                ).to(torch.float32),
                positive_headroom_raw_x=torch.from_numpy(
                    np.array(source[prefix + "positive_headroom_unit"], copy=True)
                ).to(torch.float32),
                negative_headroom_raw_x=torch.from_numpy(
                    np.array(source[prefix + "negative_headroom_unit"], copy=True)
                ).to(torch.float32),
                positive_capacity=torch.from_numpy(
                    np.array(source[prefix + "positive_level_capacity"], copy=True)
                ).to(torch.int64),
                negative_capacity=torch.from_numpy(
                    np.array(source[prefix + "negative_level_capacity"], copy=True)
                ).to(torch.int64),
                initial_normalized_weight=torch.from_numpy(
                    np.array(source[prefix + "normalized_weight"], copy=True)
                ).to(torch.float32),
                level_spacing_raw_x=float(level_spacing_raw_x),
            )
            _validate_template(template)
            templates.append(template)
    if len(templates) != 2:
        raise ValueError("Expected exactly two Winsorized QAT templates.")
    return tuple(templates)


def quantize_winsorized_logical_master(
    masters: Sequence[torch.Tensor],
    templates: Sequence[WinsorizedQatLayerTemplate],
    *,
    include_report: bool = True,
) -> WinsorizedQatView:
    """Materialize ``G=B+n*(2h_x)`` from a normalized logical shadow."""

    selected_masters = tuple(masters)
    selected_templates = tuple(templates)
    if len(selected_masters) != 2 or len(selected_templates) != 2:
        raise ValueError("Expected two logical masters and two QAT templates.")
    targets = []
    raw_targets = []
    requested = []
    realized = []
    reports = []
    for master, template in zip(selected_masters, selected_templates):
        _validate_template(template)
        if master.shape != template.logical_shape or not bool(
            torch.isfinite(master).all()
        ):
            raise ValueError("Expected a finite logical master with frozen shape.")
        if bool(torch.any(master < -1.0)) or bool(torch.any(master > 1.0)):
            raise ValueError("Expected normalized logical masters in [-1,1].")
        device = master.device
        positive_headroom = template.positive_headroom_raw_x.to(
            device=device, dtype=master.dtype
        )
        negative_headroom = template.negative_headroom_raw_x.to(
            device=device, dtype=master.dtype
        )
        positive_capacity = template.positive_capacity.to(device=device)
        negative_capacity = template.negative_capacity.to(device=device)
        positive = master >= 0.0
        headroom = torch.where(positive, positive_headroom, negative_headroom)
        capacity = torch.where(positive, positive_capacity, negative_capacity)
        continuous_index = headroom * master.abs() / template.level_spacing_raw_x
        level = torch.floor(continuous_index + 0.5).to(torch.int64)
        level = torch.minimum(level, capacity)
        level = torch.maximum(level, torch.zeros_like(level))

        level_quad = torch.zeros(
            (*master.shape, 4), dtype=torch.int64, device=device
        )
        level_quad[..., 0] = torch.where(positive, level, 0)
        level_quad[..., 3] = torch.where(positive, level, 0)
        level_quad[..., 1] = torch.where(positive, 0, level)
        level_quad[..., 2] = torch.where(positive, 0, level)
        physical_shape = tuple(template.baseline_full_g.shape)
        physical_level = scatter_quads(
            level_quad, shape=physical_shape, layout=template.layout
        )
        offset_raw64 = (
            physical_level.to(torch.float64) * template.level_spacing_raw_x
        )
        offset_g = (2.0 * offset_raw64).to(torch.float32)
        baseline_g = template.baseline_full_g.to(device=device, dtype=torch.float32)
        target_g = baseline_g + offset_g.to(device=device)
        target_raw = target_g / 2.0
        if (
            not bool(torch.isfinite(target_g).all())
            or bool(torch.any(target_g < 0.0))
            or bool(torch.any(target_g > 2.0 + 2e-6))
        ):
            raise RuntimeError("QAT quantizer produced an invalid full conductance.")
        upper = template.cell_upper_raw_x.to(device=device, dtype=torch.float32)
        if bool(torch.any(target_raw > upper + 1e-6)):
            raise RuntimeError("QAT quantizer exceeded one cell's Winsorized support.")

        signed_level = torch.where(positive, level, -level).to(master.dtype)
        safe_headroom = torch.where(headroom > 0.0, headroom, torch.ones_like(headroom))
        realized_u = signed_level * template.level_spacing_raw_x / safe_headroom
        realized_u = torch.where(headroom > 0.0, realized_u, torch.zeros_like(realized_u))
        if include_report:
            error = realized_u - master
            nonzero = master != 0.0
            reports.append(
                {
                    "layer": template.layer_index,
                    "layout": template.layout,
                    "level_spacing_raw_x": template.level_spacing_raw_x,
                    "zero_level_count": int((level == 0).sum().item()),
                    "weights": int(level.numel()),
                    "requested_level": _tensor_summary(level.to(torch.float64)),
                    "master": _tensor_summary(master),
                    "realized": _tensor_summary(realized_u),
                    "error_rmse": float(
                        error.to(torch.float64).square().mean().sqrt().item()
                    ),
                    "error_relative_l2": float(
                        error.to(torch.float64).norm().item()
                        / max(master.to(torch.float64).norm().item(), 1e-30)
                    ),
                    "erased_nonzero_count": int(
                        (nonzero & (level == 0)).sum().item()
                    ),
                }
            )
        targets.append(target_g)
        raw_targets.append(target_raw)
        requested.append(physical_level)
        realized.append(realized_u)
    return WinsorizedQatView(
        full_conductance_targets=tuple(targets),
        raw_x_targets=tuple(raw_targets),
        requested_index=tuple(requested),
        realized_normalized_weight=tuple(realized),
        report={"layers": reports} if include_report else {},
    )


def logical_master_gradients(
    masters: Sequence[torch.Tensor],
    physical_gradients: Sequence[torch.Tensor],
    templates: Sequence[WinsorizedQatLayerTemplate],
) -> tuple[torch.Tensor, ...]:
    """Apply the declared identity-STE through each sign-selected envelope.

    The circuit gradient is with respect to complete branch conductances.  A
    positive logical weight raises ``G++`` and ``G--`` by ``2 H_+ u``;
    a negative weight raises ``G+-`` and ``G-+`` by ``-2 H_- u``.
    """

    if not (len(masters) == len(physical_gradients) == len(templates) == 2):
        raise ValueError("Expected two matched QAT gradient layers.")
    result = []
    for master, physical_gradient, template in zip(
        masters, physical_gradients, templates
    ):
        if physical_gradient.shape != template.baseline_full_g.shape:
            raise ValueError("Physical QAT gradient shape mismatch.")
        quad = quad_stack(
            physical_gradient.to(device=master.device, dtype=master.dtype),
            layout=template.layout,
        )
        positive_headroom = template.positive_headroom_raw_x.to(
            device=master.device, dtype=master.dtype
        )
        negative_headroom = template.negative_headroom_raw_x.to(
            device=master.device, dtype=master.dtype
        )
        positive_gradient = 2.0 * positive_headroom * (quad[..., 0] + quad[..., 3])
        negative_gradient = -2.0 * negative_headroom * (quad[..., 1] + quad[..., 2])
        gradient = torch.where(master >= 0.0, positive_gradient, negative_gradient)
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("QAT logical STE gradient became non-finite.")
        result.append(gradient)
    return tuple(result)


__all__ = [
    "LAYOUTS",
    "WinsorizedQatLayerTemplate",
    "WinsorizedQatView",
    "load_winsorized_qat_templates",
    "logical_master_gradients",
    "quantize_winsorized_logical_master",
]
