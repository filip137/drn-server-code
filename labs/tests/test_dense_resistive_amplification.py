import pytest
import torch

from model.resistive.interaction import DenseResistive
from model.resistive.layer import ResistiveInputLayer
from model.variable.layer import Layer, LinearLayer
from model.variable.parameter import DenseWeight


def _reset_names():
    Layer._counter = 0
    DenseWeight._counter = 0


def _positive_dense_weight(pre_shape, post_shape):
    weight = DenseWeight(
        pre_shape,
        post_shape,
        gain=1.0,
        device="cpu",
        clamp=True,
        clamp_min=0.01,
        clamp_max=10.0,
        init_mode="kaiming_uniform",
    )
    with torch.no_grad():
        weight.state.copy_(torch.rand_like(weight.state) + 0.1)
    return weight


@pytest.mark.parametrize("pre_layer_index", [0, 1, 2])
@pytest.mark.parametrize(
    ("voltage_amp", "current_amp"),
    [(1.0, 1.0), (2.0, 1.0), (4.0, 1.0), (1.0, 2.0), (1.0, 4.0), (2.0, 3.0)],
)
def test_dense_resistive_amplification_coefficients_match_eval_gradient(
    pre_layer_index, voltage_amp, current_amp
):
    torch.manual_seed(0)
    _reset_names()

    batch_size = 5
    if pre_layer_index == 0:
        pre = ResistiveInputLayer((3,), gain=1.0)
    else:
        ResistiveInputLayer((2,), gain=1.0)
        for _ in range(1, pre_layer_index):
            LinearLayer((2,), batch_size=batch_size)
        pre = LinearLayer((3,), batch_size=batch_size)
    post = LinearLayer((4,), batch_size=batch_size)

    pre.state = torch.randn(batch_size, 3, requires_grad=True)
    post.state = torch.randn(batch_size, 4, requires_grad=True)
    weight = _positive_dense_weight(pre.shape, post.shape)

    interaction = DenseResistive(
        pre,
        post,
        weight,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
    )
    energy = interaction.eval().sum()
    grad_pre, grad_post = torch.autograd.grad(energy, (pre.state, post.state))

    coef_pre = (
        2.0 * interaction._a_coef_layer_pre() * pre.state
        + interaction._b_coef_layer_pre()
    )
    coef_post = (
        2.0 * interaction._a_coef_layer_post() * post.state
        + interaction._b_coef_layer_post()
    )

    assert torch.allclose(grad_pre, coef_pre, atol=2e-5, rtol=1e-5)
    assert torch.allclose(grad_post, coef_post, atol=2e-5, rtol=1e-5)
