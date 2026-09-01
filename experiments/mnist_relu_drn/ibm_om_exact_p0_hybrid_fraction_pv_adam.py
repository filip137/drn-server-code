"""Nested W2-plus-W1 masks for the exact-P0 partial P&V frontier.

Every arm updates all four nonnegative physical conductances belonging to a
selected logical quad.  W2 is always trainable.  W1 is either a nested prefix
of the frozen P0-gradient ranking or a deterministic address-only random
control.  The random selector is deliberately independent of scores, device
bounds, corruption identity, persistent state, validation data, and test data.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence

import torch

from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_exact_p0_structured_partial_pv_adam import (
    LAYOUTS,
    _physical_mask,
    _stable_top_mask,
    _validate_scores,
)


ARM_IDS = (
    "w2_only",
    "w2_plus_w1_top125",
    "w2_plus_w1_top250",
    "w2_plus_w1_top500",
    "w2_plus_w1_random500",
)
TOP_W1_COUNTS = (125, 250, 500)
RANDOM_W1_COUNT = 500
RANDOM_SELECTOR_SEED = 94_501
RANDOM_SELECTOR_ALGORITHM = (
    "sha256(seed_u64_be||flat_W1_index_u64_be), lexicographic_digest_then_index"
)


@dataclass(frozen=True)
class HybridFractionUpdateMask:
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


def _selected_flat_indices_sha256(mask: torch.Tensor) -> str:
    indices = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1).to(torch.int64)
    return _tensor_sha256(indices)


def _deterministic_random_mask(
    shape: Sequence[int], *, count: int, seed: int
) -> torch.Tensor:
    size = 1
    for value in shape:
        size *= int(value)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("Random selector seed must be an unsigned 64-bit integer.")
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= size:
        raise ValueError("Random selector count must fit the W1 logical layer.")
    seed_bytes = seed.to_bytes(8, byteorder="big", signed=False)
    ranked = sorted(
        range(size),
        key=lambda index: (
            sha256(
                seed_bytes + index.to_bytes(8, byteorder="big", signed=False)
            ).digest(),
            index,
        ),
    )
    result = torch.zeros(size, dtype=torch.bool)
    if count:
        result[torch.tensor(ranked[:count], dtype=torch.int64)] = True
    return result.reshape(tuple(int(value) for value in shape))


def build_hybrid_fraction_update_masks(
    quad_scores: Sequence[torch.Tensor],
    *,
    binding_shapes: Sequence[Sequence[int]],
    layouts: Sequence[str] = LAYOUTS,
    random_selector_seed: int = RANDOM_SELECTOR_SEED,
) -> Mapping[str, HybridFractionUpdateMask]:
    """Build the nested W2-plus-ranked-W1 screen and random-W1 control."""

    scores = _validate_scores(
        quad_scores, binding_shapes=binding_shapes, layouts=layouts
    )
    shapes = tuple(tuple(int(item) for item in shape) for shape in binding_shapes)
    if scores[0].numel() != 39_200 or scores[1].numel() != 500:
        raise ValueError("Expected the canonical MNIST DRN W1/W2 logical shapes.")
    w2_full = torch.ones_like(scores[1], dtype=torch.bool)
    w1_empty = torch.zeros_like(scores[0], dtype=torch.bool)
    w1_random = _deterministic_random_mask(
        scores[0].shape, count=RANDOM_W1_COUNT, seed=random_selector_seed
    )
    logical_by_arm = {
        "w2_only": (w1_empty, w2_full),
        **{
            f"w2_plus_w1_top{count}": (
                _stable_top_mask(scores[0], count),
                w2_full,
            )
            for count in TOP_W1_COUNTS
        },
        "w2_plus_w1_random500": (w1_random, w2_full),
    }
    result: dict[str, HybridFractionUpdateMask] = {}
    for arm_id in ARM_IDS:
        logical = tuple(value.clone() for value in logical_by_arm[arm_id])
        physical = tuple(
            _physical_mask(value, shape=shape, layout=layout)
            for value, shape, layout in zip(
                logical, shapes, tuple(layouts), strict=True
            )
        )
        is_random = arm_id == "w2_plus_w1_random500"
        result[arm_id] = HybridFractionUpdateMask(
            arm_id=arm_id,
            w1_policy=(
                "deterministic_random_flat_W1_addresses"
                if is_random
                else (
                    "none"
                    if arm_id == "w2_only"
                    else "nested_frozen_P0_gradient_rank_prefix"
                )
            ),
            logical_masks=logical,  # type: ignore[arg-type]
            physical_masks=physical,  # type: ignore[arg-type]
            selector_seed=random_selector_seed if is_random else None,
            selected_w1_flat_indices_sha256=(
                _selected_flat_indices_sha256(logical[0]) if is_random else None
            ),
        )
    if tuple(result) != ARM_IDS:
        raise RuntimeError("Hybrid-fraction arm order changed.")
    expected_w1 = (0, 125, 250, 500, 500)
    for arm_id, count in zip(ARM_IDS, expected_w1, strict=True):
        report = result[arm_id].report()
        if report["logical_weights_by_layer"] != [count, 500] or report[
            "physical_cells"
        ] != 4 * (count + 500):
            raise RuntimeError("Hybrid-fraction mask cardinality changed.")
    top125 = result["w2_plus_w1_top125"].logical_masks[0]
    top250 = result["w2_plus_w1_top250"].logical_masks[0]
    top500 = result["w2_plus_w1_top500"].logical_masks[0]
    if not bool(torch.all(top125 <= top250)) or not bool(torch.all(top250 <= top500)):
        raise RuntimeError("P0-gradient W1 prefixes are no longer nested.")
    return result


__all__ = [
    "ARM_IDS",
    "HybridFractionUpdateMask",
    "RANDOM_SELECTOR_ALGORITHM",
    "RANDOM_SELECTOR_SEED",
    "RANDOM_W1_COUNT",
    "TOP_W1_COUNTS",
    "build_hybrid_fraction_update_masks",
]
