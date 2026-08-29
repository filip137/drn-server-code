"""Nominal-bound-winsorized IBM OM single-device baseline/spacing mapping.

This exploratory intervention clips (statistically, winsorizes) each sampled
raw active-device bound to ``a in [-1, 1]`` *before* RESET commissioning and
program-and-verify.  It does not reject or resample outlying identities.  The
passive circuit then consumes the complete single-device conductance

``G = a + 1 = 2 x``, where ``x = (a + 1) / 2``.

There is no reference device and no baseline subtraction at circuit handoff.
The truncation changes the pulse plant's hard bounds; persistent endpoints are
therefore already physical and must never be projected after programming.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import torch

from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv import (
    BaselineSpacingMapping,
    build_baseline_spacing_mapping,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    quad_stack,
    scatter_quads,
)


NOMINAL_RAW_A_MINIMUM = -1.0
NOMINAL_RAW_A_MAXIMUM = 1.0
NOMINAL_RAW_X_MINIMUM = 0.0
NOMINAL_RAW_X_MAXIMUM = 1.0
NOMINAL_CONDUCTANCE_MINIMUM = 0.0
NOMINAL_CONDUCTANCE_MAXIMUM = 2.0


@dataclass(frozen=True)
class TruncatedNominalOmEmbedding:
    """Exact non-projecting conversion from truncated raw ``x`` to full ``G``."""

    raw_x_minimum: float = NOMINAL_RAW_X_MINIMUM
    raw_x_maximum: float = NOMINAL_RAW_X_MAXIMUM
    conductance_minimum: float = NOMINAL_CONDUCTANCE_MINIMUM
    conductance_maximum: float = NOMINAL_CONDUCTANCE_MAXIMUM

    def __post_init__(self) -> None:
        values = (
            self.raw_x_minimum,
            self.raw_x_maximum,
            self.conductance_minimum,
            self.conductance_maximum,
        )
        if (
            not all(math.isfinite(float(value)) for value in values)
            or float(self.raw_x_minimum) != 0.0
            or float(self.raw_x_maximum) != 1.0
            or float(self.conductance_minimum) != 0.0
            or float(self.conductance_maximum) != 2.0
        ):
            raise ValueError("Expected the frozen a in [-1,1], G=a+1 embedding.")

    @property
    def raw_x_origin(self) -> float:
        return self.raw_x_minimum

    @property
    def raw_x_ceiling(self) -> float:
        return self.raw_x_maximum

    @property
    def conductance_per_raw_x(self) -> float:
        return 2.0

    @property
    def conductance_ceiling(self) -> float:
        return self.conductance_maximum

    def to_full_conductance(self, raw_x: torch.Tensor) -> torch.Tensor:
        """Return ``G=2x`` while rejecting, rather than clipping, bad endpoints."""

        value = torch.as_tensor(raw_x)
        if value.numel() < 1 or not value.is_floating_point() or not bool(
            torch.all(torch.isfinite(value))
        ):
            raise ValueError("Expected finite floating truncated raw-x endpoints.")
        raw64 = value.to(dtype=torch.float64)
        if bool(torch.any(raw64 < 0.0)) or bool(torch.any(raw64 > 1.0)):
            raise ValueError(
                "Truncated nominal OM endpoint left x in [0,1]; no post-handoff "
                "projection is permitted."
            )
        full = (2.0 * raw64).to(dtype=value.dtype)
        if bool(torch.any(full < 0.0)) or not bool(torch.all(torch.isfinite(full))):
            raise ValueError(
                "Truncated nominal conductance became negative or non-finite."
            )
        return full

    def report(self) -> Mapping[str, float | str]:
        return {
            "policy": "nominal_bound_winsorization_before_commissioning_and_pv",
            "native_support_intervention": "a_min=max(a_min,-1);a_max=min(a_max,1)",
            "formula": "G=a_truncated+1=2*x",
            "raw_a_minimum": NOMINAL_RAW_A_MINIMUM,
            "raw_a_maximum": NOMINAL_RAW_A_MAXIMUM,
            "raw_x_minimum": self.raw_x_minimum,
            "raw_x_maximum": self.raw_x_maximum,
            "conductance_minimum": self.conductance_minimum,
            "conductance_maximum": self.conductance_maximum,
            "reference_device": "absent",
            "post_handoff_clipping": "forbidden",
        }


def project_shared_destination_reset_baselines(
    cell_lower_raw_x: Sequence[torch.Tensor],
    cell_upper_raw_x: Sequence[torch.Tensor],
    reset_baseline_raw_x: Sequence[torch.Tensor],
    *,
    layouts: Sequence[str] = ("halves", "paired"),
) -> tuple[tuple[torch.Tensor, ...], Mapping[str, Any]]:
    """Project each requested RESET-max baseline into its exact common support.

    The operation occurs once during target construction, before quantization
    or P&V.  It is not an endpoint handoff projection.  Each destination pair
    requests ``B_req=max(reset_0, reset_1)`` and receives
    ``clamp(B_req, max(lower_0,lower_1), min(upper_0,upper_1))``.  The selected
    scalar is then broadcast back to both cells in the destination pair.
    """

    lower_sequence = tuple(cell_lower_raw_x)
    upper_sequence = tuple(cell_upper_raw_x)
    reset_sequence = tuple(reset_baseline_raw_x)
    selected_layouts = tuple(layouts)
    if not (
        len(lower_sequence)
        == len(upper_sequence)
        == len(reset_sequence)
        == len(selected_layouts)
        == 2
    ):
        raise ValueError("Expected two physical layers and two layouts.")

    projected_layers: list[torch.Tensor] = []
    layer_reports: list[Mapping[str, Any]] = []
    total_pairs = 0
    total_projected = 0
    total_absolute = 0.0
    maximum_absolute = 0.0
    for layer_index, (lower_raw, upper_raw, reset_raw, layout) in enumerate(
        zip(lower_sequence, upper_sequence, reset_sequence, selected_layouts)
    ):
        lower = torch.as_tensor(lower_raw).detach().to(torch.float64)
        upper = torch.as_tensor(upper_raw).detach().to(torch.float64)
        reset = torch.as_tensor(reset_raw).detach().to(torch.float64)
        if (
            lower.shape != upper.shape
            or lower.shape != reset.shape
            or lower.numel() < 1
            or not all(
                bool(torch.all(torch.isfinite(value)))
                for value in (lower, upper, reset)
            )
            or bool(torch.any(upper < lower))
            or bool(torch.any(reset < lower))
            or bool(torch.any(reset > upper))
        ):
            raise ValueError(
                f"Expected finite ordered bounds and per-cell bounded RESET in "
                f"layer {layer_index}."
            )
        lower_quad = quad_stack(lower, layout=layout)
        upper_quad = quad_stack(upper, layout=layout)
        reset_quad = quad_stack(reset, layout=layout)
        common_lower = torch.stack(
            (
                torch.maximum(lower_quad[..., 0], lower_quad[..., 2]),
                torch.maximum(lower_quad[..., 1], lower_quad[..., 3]),
            ),
            dim=-1,
        )
        common_upper = torch.stack(
            (
                torch.minimum(upper_quad[..., 0], upper_quad[..., 2]),
                torch.minimum(upper_quad[..., 1], upper_quad[..., 3]),
            ),
            dim=-1,
        )
        empty = common_upper < common_lower
        if bool(empty.any()):
            raise ValueError(
                f"Winsorized destination pair has empty exact common support "
                f"in layer {layer_index}; empty_pairs={int(empty.sum().item())}."
            )
        requested = torch.stack(
            (
                torch.maximum(reset_quad[..., 0], reset_quad[..., 2]),
                torch.maximum(reset_quad[..., 1], reset_quad[..., 3]),
            ),
            dim=-1,
        )
        projected = torch.minimum(torch.maximum(requested, common_lower), common_upper)
        delta = projected - requested
        projected_mask = delta != 0.0
        expanded = torch.stack(
            (
                projected[..., 0],
                projected[..., 1],
                projected[..., 0],
                projected[..., 1],
            ),
            dim=-1,
        )
        projected_layer = scatter_quads(
            expanded,
            shape=tuple(reset.shape),
            layout=layout,
        ).to(dtype=torch.as_tensor(reset_raw).dtype)
        if bool(torch.any(projected_layer.to(torch.float64) < lower)) or bool(
            torch.any(projected_layer.to(torch.float64) > upper)
        ):
            raise RuntimeError("Projected shared RESET baseline left a cell support.")

        absolute = delta.abs()
        pair_count = int(requested.numel())
        projection_count = int(projected_mask.sum().item())
        layer_absolute = float(absolute.sum().item())
        layer_maximum = float(absolute.max().item()) if pair_count else 0.0
        total_pairs += pair_count
        total_projected += projection_count
        total_absolute += layer_absolute
        maximum_absolute = max(maximum_absolute, layer_maximum)
        projected_layers.append(projected_layer)
        layer_reports.append(
            {
                "layer": layer_index,
                "layout": layout,
                "destination_pair_count": pair_count,
                "projected_pair_count": projection_count,
                "projected_pair_fraction": (
                    projection_count / pair_count if pair_count else 0.0
                ),
                "downward_projection_count": int((delta < 0.0).sum().item()),
                "upward_projection_count": int((delta > 0.0).sum().item()),
                "absolute_projection_sum_raw_x": layer_absolute,
                "absolute_projection_mean_all_pairs_raw_x": (
                    layer_absolute / pair_count if pair_count else 0.0
                ),
                "absolute_projection_mean_projected_pairs_raw_x": (
                    layer_absolute / projection_count if projection_count else 0.0
                ),
                "absolute_projection_maximum_raw_x": layer_maximum,
            }
        )
    return tuple(projected_layers), {
        "policy": "project_requested_RESET_max_once_to_exact_common_support",
        "stage": "target_construction_before_quantization_and_pv",
        "persistent_endpoint_projection": False,
        "destination_pair_count": total_pairs,
        "projected_pair_count": total_projected,
        "projected_pair_fraction": (
            total_projected / total_pairs if total_pairs else 0.0
        ),
        "absolute_projection_sum_raw_x": total_absolute,
        "absolute_projection_mean_all_pairs_raw_x": (
            total_absolute / total_pairs if total_pairs else 0.0
        ),
        "absolute_projection_mean_projected_pairs_raw_x": (
            total_absolute / total_projected if total_projected else 0.0
        ),
        "absolute_projection_maximum_raw_x": maximum_absolute,
        "layers": layer_reports,
    }


def build_truncated_nominal_baseline_spacing_mapping(
    logical_weights: Sequence[torch.Tensor],
    *,
    baseline_position_fraction: float,
    spacing_delta_multiples: int,
    scale_fractions: Sequence[float],
    nominal_dw_min: float,
    cell_lower_raw_x: Sequence[torch.Tensor],
    cell_upper_raw_x: Sequence[torch.Tensor],
    reset_baseline_raw_x: Sequence[torch.Tensor],
    intrinsic_references_native: Sequence[torch.Tensor],
) -> BaselineSpacingMapping:
    """Build a shared-destination map on the pre-truncated pulse support.

    Baselines, spacing, and integer capacities use ``x=(a+1)/2``.  Complete
    circuit conductances use ``G=2x``.  The intrinsic OM ``reference`` tensor
    is retained only as frozen identity provenance by the established mapper;
    it is not subtracted from any branch conductance.
    """

    sequences = (
        tuple(cell_lower_raw_x),
        tuple(cell_upper_raw_x),
        tuple(reset_baseline_raw_x),
    )
    if any(len(values) != 2 for values in sequences):
        raise ValueError("Expected exactly two truncated physical layers.")
    tolerance = 1e-7
    for layer_index, (lower, upper, reset) in enumerate(zip(*sequences)):
        lower64, upper64, reset64 = (
            torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
            for value in (lower, upper, reset)
        )
        if (
            lower64.shape != upper64.shape
            or lower64.shape != reset64.shape
            or lower64.numel() < 1
            or not all(
                bool(torch.all(torch.isfinite(value)))
                for value in (lower64, upper64, reset64)
            )
            or bool(torch.any(lower64 < -tolerance))
            or bool(torch.any(upper64 > 1.0 + tolerance))
            or bool(torch.any(upper64 < lower64))
            or bool(torch.any(reset64 < lower64))
            or bool(torch.any(reset64 > upper64))
        ):
            raise ValueError(
                f"Expected finite ordered x in [0,1] and bounded commissioned "
                f"RESET in layer {layer_index}."
            )

    projected_reset, baseline_projection = (
        project_shared_destination_reset_baselines(
            sequences[0],
            sequences[1],
            sequences[2],
        )
    )
    mapped = build_baseline_spacing_mapping(
        logical_weights,
        baseline_position_fraction=baseline_position_fraction,
        spacing_delta_multiples=spacing_delta_multiples,
        scale_fractions=scale_fractions,
        nominal_dw_min=nominal_dw_min,
        conductance_min=NOMINAL_CONDUCTANCE_MINIMUM,
        conductance_max=NOMINAL_CONDUCTANCE_MAXIMUM,
        cell_lower_units=sequences[0],
        cell_upper_units=sequences[1],
        reset_baseline_units=projected_reset,
        intrinsic_references_native=intrinsic_references_native,
    )
    embedding = TruncatedNominalOmEmbedding()
    for target_x, target_g in zip(mapped.ideal_target_units, mapped.ideal_targets):
        reconstructed = embedding.to_full_conductance(target_x)
        if not torch.allclose(reconstructed, target_g, rtol=0.0, atol=2e-6):
            raise RuntimeError("The truncated target does not satisfy G=a+1=2x.")
    report = {
        **dict(mapped.report),
        "coordinate": "x=(a_truncated+1)/2",
        "delta_x": float(nominal_dw_min) / 2.0,
        "level_spacing_unit": (
            int(spacing_delta_multiples) * float(nominal_dw_min) / 2.0
        ),
        "conductance_embedding": dict(embedding.report()),
        "shared_destination_baseline_projection": baseline_projection,
        "support_policy": "winsorize_bounds_before_commissioning_and_pv",
        "out_of_bounds_policy": "fail_no_post_handoff_projection",
    }
    return replace(mapped, report=report)


__all__ = [
    "NOMINAL_CONDUCTANCE_MAXIMUM",
    "NOMINAL_CONDUCTANCE_MINIMUM",
    "NOMINAL_RAW_A_MAXIMUM",
    "NOMINAL_RAW_A_MINIMUM",
    "NOMINAL_RAW_X_MAXIMUM",
    "NOMINAL_RAW_X_MINIMUM",
    "TruncatedNominalOmEmbedding",
    "build_truncated_nominal_baseline_spacing_mapping",
    "project_shared_destination_reset_baselines",
]
