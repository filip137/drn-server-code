from __future__ import annotations

import torch

from experiments.mnist_relu_drn.ibm_om_baseline_selection import quad_stack
from experiments.mnist_relu_drn.ibm_om_exact_p0_hybrid_extension_pv_adam import (
    ARM_IDS,
    RANDOM_ARM_ID,
    RANDOM_SELECTOR_SEED,
    build_hybrid_extension_update_masks,
)


SHAPES = ((1568, 100), (100, 20))


def _scores() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.arange(39_200, dtype=torch.float64).reshape(784, 50),
        torch.arange(500, dtype=torch.float64).reshape(50, 10),
    )


def test_extension_masks_are_nested_quad_coherent_and_keep_all_w2() -> None:
    masks = build_hybrid_extension_update_masks(_scores(), binding_shapes=SHAPES)
    assert tuple(masks) == ARM_IDS
    assert [masks[arm].report()["logical_weights_by_layer"] for arm in ARM_IDS] == [
        [500, 500],
        [1_000, 500],
        [2_000, 500],
        [4_000, 500],
        [2_000, 500],
    ]
    assert [masks[arm].report()["physical_cells"] for arm in ARM_IDS] == [
        4_000,
        6_000,
        10_000,
        18_000,
        10_000,
    ]
    ranked = [
        masks[arm].logical_masks[0]
        for arm in ARM_IDS
        if "top" in arm
    ]
    for left, right in zip(ranked[:-1], ranked[1:], strict=True):
        assert bool(torch.all(left <= right))
    for mask in masks.values():
        assert bool(torch.all(mask.logical_masks[1]))
        for physical, logical, layout in zip(
            mask.physical_masks,
            mask.logical_masks,
            ("halves", "paired"),
            strict=True,
        ):
            observed = quad_stack(physical, layout=layout)
            assert torch.equal(observed, logical[..., None].expand_as(observed))
        assert mask.report()["physical_conductance_domain"] == (
            "nonnegative_G=a+1=2*x"
        )


def test_random2000_is_reproducible_score_independent_and_hash_gated() -> None:
    first = build_hybrid_extension_update_masks(
        _scores(), binding_shapes=SHAPES, random_selector_seed=RANDOM_SELECTOR_SEED
    )[RANDOM_ARM_ID]
    changed_scores = (
        torch.flip(_scores()[0], dims=(0, 1)),
        torch.full((50, 10), 1.0, dtype=torch.float64),
    )
    second = build_hybrid_extension_update_masks(
        changed_scores,
        binding_shapes=SHAPES,
        random_selector_seed=RANDOM_SELECTOR_SEED,
    )[RANDOM_ARM_ID]
    third = build_hybrid_extension_update_masks(
        _scores(),
        binding_shapes=SHAPES,
        random_selector_seed=RANDOM_SELECTOR_SEED + 1,
    )[RANDOM_ARM_ID]
    assert torch.equal(first.logical_masks[0], second.logical_masks[0])
    assert not torch.equal(first.logical_masks[0], third.logical_masks[0])
    report = first.report()
    assert report["selector_seed"] == 94_501
    assert report["selected_w1_flat_indices_sha256"] == (
        "9048ca1fdae7e85ee6941a4d2553892e57ef1e6eea8d9794a7f66a5c3076d792"
    )
