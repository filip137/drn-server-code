import math

import pytest
import torch

from experiments.mnist_conv.gradient_replay import (
    attached_weight_name,
    gradient_vector_comparison,
    summarize_channel_activity_gradients,
    summarize_gradient_vectors,
    validate_probe_batch_positions,
)


def test_default_sparse_probe_positions_are_valid():
    assert validate_probe_batch_positions((0, 4, 8, 12, 16, 20, 24, 28)) == (
        0,
        4,
        8,
        12,
        16,
        20,
        24,
        28,
    )


@pytest.mark.parametrize("positions", [(), (0, 0), (4, 0), (-1,), (32,)])
def test_invalid_probe_positions_fail_loudly(positions):
    with pytest.raises(ValueError, match="Expected"):
        validate_probe_batch_positions(positions)


def test_attached_weight_mapping_keeps_weights_and_maps_biases():
    mapping = {"Bias_0": "ConvWeight_0", "Bias_1": "ConvWeight_1"}
    assert attached_weight_name("ConvWeight_0", mapping) == "ConvWeight_0"
    assert attached_weight_name("DenseWeight_0", mapping) == "DenseWeight_0"
    assert attached_weight_name("Bias_1", mapping) == "ConvWeight_1"
    with pytest.raises(ValueError, match="explicit attached-weight"):
        attached_weight_name("Bias_2", mapping)


def test_gradient_vector_comparison_reports_exact_match():
    vector = torch.tensor([1.0, -2.0, 3.0])
    result = gradient_vector_comparison(vector, vector.clone())
    assert result["reference_cosine"] == pytest.approx(1.0)
    assert result["reference_relative_error"] == pytest.approx(0.0)
    assert result["reference_norm_ratio"] == pytest.approx(1.0)


def test_gradient_vector_comparison_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="identical shapes"):
        gradient_vector_comparison(torch.ones(2), torch.ones(3))


def test_gradient_summary_scale_sparsity_and_coherence():
    vectors = [
        torch.tensor([1.0, 0.0, -1.0]),
        torch.tensor([2.0, 0.0, -2.0]),
    ]
    result = summarize_gradient_vectors(vectors)
    assert result["gradient_rms_median"] == pytest.approx(
        math.sqrt(3.0 / 2.0)
    )
    assert result["gradient_zero_fraction_median"] == pytest.approx(1.0 / 3.0)
    assert result["gradient_batch_coherence"] > 0.94
    assert result["gradient_pairwise_cosine_median"] == pytest.approx(1.0)


def test_gradient_summary_rejects_empty_or_mismatched_vectors():
    with pytest.raises(ValueError, match="at least one"):
        summarize_gradient_vectors([])
    with pytest.raises(ValueError, match="identical shapes"):
        summarize_gradient_vectors([torch.ones(2), torch.ones(3)])


def test_channel_activity_gradient_summary_detects_active_channel_concentration():
    activity = [
        torch.tensor([0.1, 0.4, 0.8, 1.0]),
        torch.tensor([0.1, 0.5, 0.9, 1.0]),
    ]
    gradients = [
        torch.tensor([[0.1, 0.1], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0]]),
        torch.tensor([[0.1, 0.1], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0]]),
    ]
    aggregate, channels = summarize_channel_activity_gradients(
        activity, gradients
    )
    assert aggregate["channel_count"] == 4
    assert aggregate["channel_active_gradient_spearman"] == pytest.approx(1.0)
    assert (
        aggregate["least_active_quartile_gradient_energy_share"]
        < 0.01
    )
    assert channels[0]["in_least_active_quartile"] is True
    assert channels[-1]["gradient_energy_fraction"] > 0.7


def test_channel_activity_gradient_summary_rejects_axis_mismatch():
    with pytest.raises(ValueError, match="first axis"):
        summarize_channel_activity_gradients(
            [torch.ones(3)],
            [torch.ones(2, 4)],
        )
