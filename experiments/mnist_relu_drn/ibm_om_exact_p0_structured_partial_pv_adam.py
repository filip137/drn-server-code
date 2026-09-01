"""Structured logical-quad masks for exact-P0 partial P&V recovery.

The selector never receives device bounds, the published-corrupt mask,
persistent hidden state, validation data, or test data.  Every selected
logical weight expands to all four physical conductances so that partial
fine-tuning does not silently change the DRN representation topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    quad_stack,
    scatter_quads,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)


LAYOUTS = ("halves", "paired")
ARM_IDS = ("full", "w1_only", "w2_only", "w1_top500", "global_top500")


@dataclass(frozen=True)
class StructuredPartialUpdateMask:
    arm_id: str
    logical_masks: tuple[torch.Tensor, torch.Tensor]
    physical_masks: tuple[torch.Tensor, torch.Tensor]

    @property
    def flat_physical_mask(self) -> torch.Tensor:
        return torch.cat(tuple(value.reshape(-1) for value in self.physical_masks))

    def report(self) -> dict[str, object]:
        logical_counts = [int(value.sum().item()) for value in self.logical_masks]
        physical_counts = [int(value.sum().item()) for value in self.physical_masks]
        return {
            "arm_id": self.arm_id,
            "logical_weights_by_layer": logical_counts,
            "logical_weights": sum(logical_counts),
            "physical_cells_by_layer": physical_counts,
            "physical_cells": sum(physical_counts),
            "logical_mask_sha256_by_layer": [
                _tensor_sha256(value) for value in self.logical_masks
            ],
            "physical_mask_sha256_by_layer": [
                _tensor_sha256(value) for value in self.physical_masks
            ],
            "flat_physical_mask_sha256": _tensor_sha256(
                self.flat_physical_mask
            ),
            "all_four_cells_per_selected_logical_weight": True,
        }


def _validate_scores(
    quad_scores: Sequence[torch.Tensor],
    *,
    binding_shapes: Sequence[Sequence[int]],
    layouts: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = tuple(torch.as_tensor(value, dtype=torch.float64) for value in quad_scores)
    shapes = tuple(tuple(int(item) for item in shape) for shape in binding_shapes)
    chosen_layouts = tuple(layouts)
    if len(scores) != 2 or len(shapes) != 2 or chosen_layouts != LAYOUTS:
        raise ValueError("Expected the canonical two-layer halves/paired DRN.")
    expected = tuple(
        quad_stack(torch.empty(shape), layout=layout).shape[:-1]
        for shape, layout in zip(shapes, chosen_layouts, strict=True)
    )
    for index, (score, wanted) in enumerate(zip(scores, expected, strict=True)):
        if score.shape != wanted or not bool(torch.all(torch.isfinite(score))):
            raise ValueError(
                f"Expected finite logical score shape {wanted!r} for layer {index}."
            )
        if bool(torch.any(score < 0.0)):
            raise ValueError("Logical importance scores must be nonnegative.")
    return scores  # type: ignore[return-value]


def _stable_top_mask(score: torch.Tensor, count: int) -> torch.Tensor:
    flat = score.reshape(-1)
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= flat.numel():
        raise ValueError("Top-k count must fit the score tensor.")
    result = torch.zeros(flat.numel(), dtype=torch.bool, device=flat.device)
    if count:
        # Stable sorting makes flat logical address the deterministic tie-break.
        order = torch.argsort(flat, descending=True, stable=True)
        result[order[:count]] = True
    return result.reshape(score.shape)


def _physical_mask(
    logical_mask: torch.Tensor, *, shape: Sequence[int], layout: str
) -> torch.Tensor:
    quads = logical_mask[..., None].expand(*logical_mask.shape, 4)
    physical = scatter_quads(quads, shape=shape, layout=layout)
    if physical.dtype is not torch.bool:
        raise RuntimeError("Structured partial mask lost its Boolean dtype.")
    if not bool(torch.all(quad_stack(physical, layout=layout).eq(quads))):
        raise RuntimeError("Structured partial mask did not preserve four-cell quads.")
    return physical


def build_structured_partial_update_masks(
    quad_scores: Sequence[torch.Tensor],
    *,
    binding_shapes: Sequence[Sequence[int]],
    layouts: Sequence[str] = LAYOUTS,
    matched_logical_count: int = 500,
) -> Mapping[str, StructuredPartialUpdateMask]:
    """Build the frozen full, layer, and cost-matched top-k masks."""

    scores = _validate_scores(
        quad_scores, binding_shapes=binding_shapes, layouts=layouts
    )
    shapes = tuple(tuple(int(item) for item in shape) for shape in binding_shapes)
    if scores[1].numel() != matched_logical_count:
        raise ValueError(
            "The primary cost match must equal the complete W2 logical layer."
        )
    full = tuple(torch.ones_like(value, dtype=torch.bool) for value in scores)
    empty = tuple(torch.zeros_like(value, dtype=torch.bool) for value in scores)
    w1_top = _stable_top_mask(scores[0], matched_logical_count)
    global_scores = torch.cat(tuple(value.reshape(-1) for value in scores))
    global_selected = _stable_top_mask(global_scores, matched_logical_count).reshape(-1)
    split = scores[0].numel()
    global_masks = (
        global_selected[:split].reshape(scores[0].shape),
        global_selected[split:].reshape(scores[1].shape),
    )
    logical_by_arm = {
        "full": full,
        "w1_only": (full[0], empty[1]),
        "w2_only": (empty[0], full[1]),
        "w1_top500": (w1_top, empty[1]),
        "global_top500": global_masks,
    }
    result: dict[str, StructuredPartialUpdateMask] = {}
    for arm_id in ARM_IDS:
        logical = tuple(value.clone() for value in logical_by_arm[arm_id])
        physical = tuple(
            _physical_mask(value, shape=shape, layout=layout)
            for value, shape, layout in zip(
                logical, shapes, tuple(layouts), strict=True
            )
        )
        result[arm_id] = StructuredPartialUpdateMask(
            arm_id=arm_id,
            logical_masks=logical,  # type: ignore[arg-type]
            physical_masks=physical,  # type: ignore[arg-type]
        )
    if tuple(result) != ARM_IDS:
        raise RuntimeError("Structured partial arm order changed.")
    for arm_id in ("w2_only", "w1_top500", "global_top500"):
        report = result[arm_id].report()
        if report["logical_weights"] != matched_logical_count or report[
            "physical_cells"
        ] != 4 * matched_logical_count:
            raise RuntimeError("The primary sparse arms are not cost matched.")
    return result


__all__ = [
    "ARM_IDS",
    "LAYOUTS",
    "StructuredPartialUpdateMask",
    "build_structured_partial_update_masks",
]
