"""Larger nested W2-plus-W1 masks for exact-P0 P&V recovery.

This stage-3 extension keeps all 500 W2 logical quads trainable and expands a
frozen stage-1 P0-gradient W1 prefix from the stage-2 top-500 anchor to 1,000,
2,000, and 4,000 quads.  The deterministic random-2,000 arm is an
address-only validation control.  Every selected logical weight expands to
all four nonnegative physical conductances in its canonical quad.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_exact_p0_hybrid_fraction_pv_adam import (
    RANDOM_SELECTOR_ALGORITHM,
    RANDOM_SELECTOR_SEED,
    _deterministic_random_mask,
    _selected_flat_indices_sha256,
)
from experiments.mnist_relu_drn.ibm_om_exact_p0_structured_partial_pv_adam import (
    LAYOUTS,
    _physical_mask,
    _stable_top_mask,
    _validate_scores,
)


ANCHOR_ARM_ID = "w2_plus_w1_top500"
LARGER_TOP_W1_COUNTS = (1_000, 2_000, 4_000)
TOP_W1_COUNTS = (500, *LARGER_TOP_W1_COUNTS)
RANDOM_W1_COUNT = 2_000
RANDOM_ARM_ID = "w2_plus_w1_random2000"
ARM_IDS = (
    ANCHOR_ARM_ID,
    "w2_plus_w1_top1000",
    "w2_plus_w1_top2000",
    "w2_plus_w1_top4000",
    RANDOM_ARM_ID,
)


@dataclass(frozen=True)
class HybridExtensionUpdateMask:
    arm_id: str
    w1_policy: str
    logical_masks: tuple[torch.Tensor, torch.Tensor]
    physical_masks: tuple[torch.Tensor, torch.Tensor]
    selector_seed: int | None = None
    selected_w1_flat_indices_sha256: str | None = None

    @property
    def flat_physical_mask(self) -> torch.Tensor:
        return torch.cat(tuple(value.reshape(-1) for value in self.physical_masks))

    def report(self) -> dict[str, object]:
        logical_counts = [int(value.sum().item()) for value in self.logical_masks]
        physical_counts = [int(value.sum().item()) for value in self.physical_masks]
        return {
            "arm_id": self.arm_id,
            "w1_policy": self.w1_policy,
            "selector_seed": self.selector_seed,
            "selector_algorithm": (
                RANDOM_SELECTOR_ALGORITHM if self.selector_seed is not None else None
            ),
            "selected_w1_flat_indices_sha256": (
                self.selected_w1_flat_indices_sha256
            ),
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
            "all_W2_logical_weights_selected": bool(
                torch.all(self.logical_masks[1]).item()
            ),
            "physical_conductance_domain": "nonnegative_G=a+1=2*x",
        }


def build_hybrid_extension_update_masks(
    quad_scores: Sequence[torch.Tensor],
    *,
    binding_shapes: Sequence[Sequence[int]],
    layouts: Sequence[str] = LAYOUTS,
    random_selector_seed: int = RANDOM_SELECTOR_SEED,
) -> Mapping[str, HybridExtensionUpdateMask]:
    """Build nested top-500--4,000 W1 unions and random-2,000 control."""

    scores = _validate_scores(
        quad_scores, binding_shapes=binding_shapes, layouts=layouts
    )
    shapes = tuple(tuple(int(item) for item in shape) for shape in binding_shapes)
    if scores[0].numel() != 39_200 or scores[1].numel() != 500:
        raise ValueError("Expected the canonical MNIST DRN W1/W2 logical shapes.")
    w2_full = torch.ones_like(scores[1], dtype=torch.bool)
    w1_random = _deterministic_random_mask(
        scores[0].shape, count=RANDOM_W1_COUNT, seed=random_selector_seed
    )
    logical_by_arm = {
        **{
            f"w2_plus_w1_top{count}": (
                _stable_top_mask(scores[0], count),
                w2_full,
            )
            for count in TOP_W1_COUNTS
        },
        RANDOM_ARM_ID: (w1_random, w2_full),
    }
    result: dict[str, HybridExtensionUpdateMask] = {}
    for arm_id in ARM_IDS:
        logical = tuple(value.clone() for value in logical_by_arm[arm_id])
        physical = tuple(
            _physical_mask(value, shape=shape, layout=layout)
            for value, shape, layout in zip(
                logical, shapes, tuple(layouts), strict=True
            )
        )
        is_random = arm_id == RANDOM_ARM_ID
        result[arm_id] = HybridExtensionUpdateMask(
            arm_id=arm_id,
            w1_policy=(
                "deterministic_random_flat_W1_addresses"
                if is_random
                else "nested_frozen_P0_gradient_rank_prefix"
            ),
            logical_masks=logical,  # type: ignore[arg-type]
            physical_masks=physical,  # type: ignore[arg-type]
            selector_seed=random_selector_seed if is_random else None,
            selected_w1_flat_indices_sha256=(
                _selected_flat_indices_sha256(logical[0]) if is_random else None
            ),
        )
    if tuple(result) != ARM_IDS:
        raise RuntimeError("Hybrid-extension arm order changed.")
    expected_w1 = (500, 1_000, 2_000, 4_000, 2_000)
    for arm_id, count in zip(ARM_IDS, expected_w1, strict=True):
        report = result[arm_id].report()
        if report["logical_weights_by_layer"] != [count, 500] or report[
            "physical_cells"
        ] != 4 * (count + 500):
            raise RuntimeError("Hybrid-extension mask cardinality changed.")
    ranked = [
        result[f"w2_plus_w1_top{count}"].logical_masks[0]
        for count in TOP_W1_COUNTS
    ]
    if any(
        not bool(torch.all(left <= right))
        for left, right in zip(ranked[:-1], ranked[1:], strict=True)
    ):
        raise RuntimeError("P0-gradient W1 prefixes are no longer nested.")
    return result


__all__ = [
    "ANCHOR_ARM_ID",
    "ARM_IDS",
    "HybridExtensionUpdateMask",
    "LARGER_TOP_W1_COUNTS",
    "RANDOM_ARM_ID",
    "RANDOM_SELECTOR_ALGORITHM",
    "RANDOM_SELECTOR_SEED",
    "RANDOM_W1_COUNT",
    "TOP_W1_COUNTS",
    "build_hybrid_extension_update_masks",
]
