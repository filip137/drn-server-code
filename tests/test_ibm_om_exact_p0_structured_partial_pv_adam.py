from __future__ import annotations

import torch

from experiments.mnist_relu_drn.ibm_om_baseline_selection import quad_stack
from experiments.mnist_relu_drn.ibm_om_exact_p0_structured_partial_pv_adam import (
    ARM_IDS,
    LAYOUTS,
    build_structured_partial_update_masks,
)


SHAPES = ((1568, 100), (100, 20))


def test_structured_masks_are_exact_address_matched_canonical_quads() -> None:
    score0 = torch.arange(39_200, dtype=torch.float64).reshape(784, 50)
    score1 = (100_000 + torch.arange(500, dtype=torch.float64)).reshape(50, 10)
    masks = build_structured_partial_update_masks(
        (score0, score1), binding_shapes=SHAPES
    )
    assert tuple(masks) == ARM_IDS
    assert masks["full"].report()["logical_weights"] == 39_700
    assert masks["w1_only"].report()["logical_weights_by_layer"] == [39_200, 0]
    assert masks["w2_only"].report()["logical_weights_by_layer"] == [0, 500]
    assert masks["w1_top500"].report()["logical_weights_by_layer"] == [500, 0]
    # W2 scores dominate in this fixture, so the global top-500 is exactly W2.
    assert masks["global_top500"].report()["logical_weights_by_layer"] == [0, 500]
    for arm_id in ("w2_only", "w1_top500", "global_top500"):
        assert masks[arm_id].report()["physical_cells"] == 2_000
    for mask in masks.values():
        for physical, logical, layout in zip(
            mask.physical_masks, mask.logical_masks, LAYOUTS, strict=True
        ):
            observed = quad_stack(physical, layout=layout)
            assert torch.equal(observed, logical[..., None].expand_as(observed))


def test_top_k_uses_stable_flat_address_tie_break() -> None:
    score0 = torch.zeros((784, 50), dtype=torch.float64)
    score1 = torch.zeros((50, 10), dtype=torch.float64)
    masks = build_structured_partial_update_masks(
        (score0, score1), binding_shapes=SHAPES
    )
    w1 = masks["w1_top500"].logical_masks[0].reshape(-1)
    assert bool(torch.all(w1[:500]))
    assert not bool(torch.any(w1[500:]))
    global_mask = masks["global_top500"]
    assert torch.equal(global_mask.logical_masks[0], masks["w1_top500"].logical_masks[0])
    assert not bool(torch.any(global_mask.logical_masks[1]))
