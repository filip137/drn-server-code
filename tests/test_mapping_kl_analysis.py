"""Focused tests for mapping-divergence measurements."""

from __future__ import annotations

import math

import pytest
import torch

from labs.tools.analyze_mnist_mapping_kl import (
    histogram_divergence,
    predictive_divergence,
)


def test_predictive_divergence_matches_a_binary_reference() -> None:
    first = torch.log(torch.tensor([[0.8, 0.2], [0.8, 0.2]]))
    second = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))

    result = predictive_divergence(first, second, mode="raw")

    expected_forward = 0.8 * math.log(1.6) + 0.2 * math.log(0.4)
    expected_reverse = 0.5 * math.log(0.5 / 0.8) + 0.5 * math.log(
        0.5 / 0.2
    )
    assert result["first_to_second_kl_nats"]["mean"] == pytest.approx(
        expected_forward
    )
    assert result["second_to_first_kl_nats"]["mean"] == pytest.approx(
        expected_reverse
    )
    assert 0.0 <= result["jensen_shannon_nats"]["mean"] <= math.log(2.0)


def test_predictive_divergence_is_zero_for_identical_scores() -> None:
    scores = torch.tensor(
        [
            [0.2, -0.1, 0.7],
            [-0.3, 0.8, 0.1],
        ]
    )

    for mode in ("raw", "per_example_rms"):
        result = predictive_divergence(scores, scores, mode=mode)
        assert result["first_to_second_kl_nats"]["maximum"] == pytest.approx(
            0.0
        )
        assert result["second_to_first_kl_nats"]["maximum"] == pytest.approx(
            0.0
        )
        assert result["jensen_shannon_nats"]["maximum"] == pytest.approx(0.0)


def test_histogram_divergence_is_symmetric_only_in_its_symmetric_fields() -> None:
    first = torch.tensor([0.1, 0.1, 0.2, 0.3, 0.9])
    second = torch.tensor([0.1, 0.4, 0.4, 0.8, 0.8])

    forward = histogram_divergence(
        first,
        second,
        bins=8,
        minimum=0.0,
        maximum=1.0,
    )
    reverse = histogram_divergence(
        second,
        first,
        bins=8,
        minimum=0.0,
        maximum=1.0,
    )

    assert forward["first_to_second_kl_nats"] == pytest.approx(
        reverse["second_to_first_kl_nats"]
    )
    assert forward["second_to_first_kl_nats"] == pytest.approx(
        reverse["first_to_second_kl_nats"]
    )
    assert forward["symmetric_kl_nats"] == pytest.approx(
        reverse["symmetric_kl_nats"]
    )
    assert forward["jensen_shannon_nats"] == pytest.approx(
        reverse["jensen_shannon_nats"]
    )
