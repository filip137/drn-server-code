from __future__ import annotations

import torch

from experiments.mnist_relu_drn.ibm_om_baseline_selection import quad_stack
from experiments.mnist_relu_drn.ibm_om_exact_p0_hybrid_fraction_pv_adam import (
    ARM_IDS,
    RANDOM_SELECTOR_SEED,
    build_hybrid_fraction_update_masks,
)


SHAPES = ((1568, 100), (100, 20))


def _scores() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.arange(39_200, dtype=torch.float64).reshape(784, 50),
        torch.arange(500, dtype=torch.float64).reshape(50, 10),
    )


def test_hybrid_masks_are_nested_quad_coherent_and_keep_all_w2() -> None:
    masks = build_hybrid_fraction_update_masks(_scores(), binding_shapes=SHAPES)
    assert tuple(masks) == ARM_IDS
    assert [masks[arm].report()["logical_weights_by_layer"] for arm in ARM_IDS] == [
        [0, 500],
        [125, 500],
        [250, 500],
        [500, 500],
        [500, 500],
    ]
    assert [masks[arm].report()["physical_cells"] for arm in ARM_IDS] == [
        2_000,
        2_500,
        3_000,
        4_000,
        4_000,
    ]
    top125 = masks["w2_plus_w1_top125"].logical_masks[0]
    top250 = masks["w2_plus_w1_top250"].logical_masks[0]
    top500 = masks["w2_plus_w1_top500"].logical_masks[0]
    assert bool(torch.all(top125 <= top250))
    assert bool(torch.all(top250 <= top500))
    for mask in masks.values():
        assert bool(torch.all(mask.logical_masks[1]))
        for physical, logical, layout in zip(
            mask.physical_masks, mask.logical_masks, ("halves", "paired"), strict=True
        ):
            observed = quad_stack(physical, layout=layout)
            assert torch.equal(observed, logical[..., None].expand_as(observed))
        assert mask.report()["physical_conductance_domain"] == "nonnegative_G=a+1=2*x"


def test_random_control_is_reproducible_score_independent_and_hash_gated() -> None:
    first = build_hybrid_fraction_update_masks(
        _scores(), binding_shapes=SHAPES, random_selector_seed=RANDOM_SELECTOR_SEED
    )["w2_plus_w1_random500"]
    changed_scores = (
        torch.flip(_scores()[0], dims=(0, 1)),
        torch.full((50, 10), 1.0, dtype=torch.float64),
    )
    second = build_hybrid_fraction_update_masks(
        changed_scores,
        binding_shapes=SHAPES,
        random_selector_seed=RANDOM_SELECTOR_SEED,
    )["w2_plus_w1_random500"]
    third = build_hybrid_fraction_update_masks(
        _scores(), binding_shapes=SHAPES, random_selector_seed=RANDOM_SELECTOR_SEED + 1
    )["w2_plus_w1_random500"]
    assert torch.equal(first.logical_masks[0], second.logical_masks[0])
    assert not torch.equal(first.logical_masks[0], third.logical_masks[0])
    report = first.report()
    assert report["selector_seed"] == 94_501
    assert report["selected_w1_flat_indices_sha256"] == (
        "85e1f3f7b1046d3acc8065b8460354950eda03973173159782a93f4b45b77887"
    )
