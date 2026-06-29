import importlib.util
import sys
from pathlib import Path

import torch


TOOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "eval_relu_to_bounded_drn_zero_shot.py"
)
SPEC = importlib.util.spec_from_file_location("eval_relu_to_bounded_drn_zero_shot", TOOL_PATH)
zero_shot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = zero_shot
SPEC.loader.exec_module(zero_shot)


def _weights():
    return zero_shot.ReluTeacherWeights(
        w1=torch.tensor([[0.5, -2.0], [-0.25, 0.75]], dtype=torch.float32),
        b1=torch.tensor([0.1, -0.2], dtype=torch.float32),
        w2=torch.tensor([[1.5, -0.5], [-0.75, 0.25]], dtype=torch.float32),
        b2=torch.tensor([0.3, -0.4], dtype=torch.float32),
    )


def test_projection_conductances_are_bounded_and_finite():
    gmax = 1e-3
    projection = zero_shot.build_sign_split_projection(
        _weights(),
        rule="layer_max",
        gmax=gmax,
    )

    for tensor in (
        projection.current_conductance,
        projection.hidden_to_output_conductance,
    ):
        assert torch.isfinite(tensor).all()
        assert float(tensor.min()) >= 0.0
        assert float(tensor.max()) <= gmax + 1e-10


def test_effective_reconstruction_matches_scaled_clipped_weights():
    gmax = 1e-3
    weights = _weights()
    projection = zero_shot.build_sign_split_projection(
        weights,
        rule="layer_max",
        gmax=gmax,
    )

    w1_eff, w2_eff = zero_shot.reconstruct_effective_weights(projection)

    assert torch.allclose(w1_eff, projection.clipped_w1)
    assert torch.allclose(w2_eff, projection.clipped_w2)
    assert torch.allclose(
        projection.current_bias,
        torch.cat((weights.b1 * projection.scale_w1, -weights.b1 * projection.scale_w1)),
    )


def test_percentile_projection_saturates_outliers():
    weights = zero_shot.ReluTeacherWeights(
        w1=torch.tensor([[1.0, 2.0], [3.0, 100.0]], dtype=torch.float32),
        b1=torch.zeros(2, dtype=torch.float32),
        w2=torch.tensor([[1.0, -2.0], [3.0, -100.0]], dtype=torch.float32),
        b2=torch.zeros(2, dtype=torch.float32),
    )
    projection = zero_shot.build_sign_split_projection(
        weights,
        rule="layer_pct_50",
        gmax=1e-3,
    )
    stats = zero_shot.projection_stats(projection)

    assert stats["current_conductance"]["gmax_fraction"] > 0.0
    assert stats["hidden_to_output_conductance"]["gmax_fraction"] > 0.0
