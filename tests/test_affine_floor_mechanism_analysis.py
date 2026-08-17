from __future__ import annotations

import torch

from labs.tools.analyze_mnist_affine_floor_mechanism import trace_scores
from labs.tools.analyze_mnist_mapping_kl import paired_scores


def _fixture():
    generator = torch.Generator().manual_seed(23)
    inputs = torch.rand((7, 2), generator=generator)
    w1 = 0.05 + 0.4 * torch.rand((4, 4), generator=generator)
    w2 = 0.05 + 0.4 * torch.rand((4, 6), generator=generator)
    bias = 0.01 * torch.rand((4,), generator=generator)
    return inputs, w1, w2, bias


def test_trace_scores_matches_existing_paired_replay() -> None:
    inputs, w1, w2, bias = _fixture()
    traced = trace_scores(
        inputs,
        numerator_w1=w1,
        numerator_w2=w2,
        bias=bias,
        input_gain=3.0,
        iterations=4,
        batch_size=3,
    )
    expected = paired_scores(
        inputs,
        w1=w1,
        w2=w2,
        bias=bias,
        input_gain=3.0,
        iterations=4,
        batch_size=3,
    )
    torch.testing.assert_close(traced["scores"], expected)


def test_uniform_input_floor_cancels_in_first_hidden_update() -> None:
    inputs, w1, w2, bias = _fixture()
    floor = 0.2
    span = 1.0 - floor
    affine_w1 = floor + span * w1
    affine_w2 = floor + span * w2
    affine = trace_scores(
        inputs,
        numerator_w1=affine_w1,
        numerator_w2=affine_w2,
        bias=bias,
        input_gain=3.0,
        iterations=1,
        batch_size=7,
    )
    signal_only = trace_scores(
        inputs,
        numerator_w1=span * w1,
        numerator_w2=span * w2,
        denominator_w1=affine_w1,
        denominator_w2=affine_w2,
        bias=bias,
        input_gain=3.0,
        iterations=1,
        batch_size=7,
    )
    torch.testing.assert_close(
        affine["preactivations"][0],
        signal_only["preactivations"][0],
        rtol=1e-6,
        atol=1e-7,
    )


def test_common_scaling_of_signal_and_bias_is_exact_invariance() -> None:
    inputs, w1, w2, bias = _fixture()
    clean = trace_scores(
        inputs,
        numerator_w1=w1,
        numerator_w2=w2,
        bias=bias,
        input_gain=3.0,
        iterations=4,
        batch_size=7,
    )
    scale = 0.37
    scaled = trace_scores(
        inputs,
        numerator_w1=scale * w1,
        numerator_w2=scale * w2,
        bias=scale * bias,
        input_gain=3.0,
        iterations=4,
        batch_size=7,
    )
    torch.testing.assert_close(
        scaled["scores"],
        clean["scores"],
        rtol=2e-6,
        atol=2e-7,
    )
