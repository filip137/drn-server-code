import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from model.resistive.interaction import ConvResistive as CoreConvResistive
from model.resistive.interaction import DenseResistive
from model.resistive.layer import ConvLayer, ResistiveInputLayer
from model.variable.layer import Layer, LinearLayer
from model.variable.parameter import ConvWeight, DenseWeight


LABS_DIR = Path(__file__).resolve().parents[1]
if str(LABS_DIR) not in sys.path:
    sys.path.insert(0, str(LABS_DIR))

from custom_classes import ConvResistive as FlexibleConvResistive  # noqa: E402


def _reset_names():
    Layer._counter = 0
    ConvWeight._counter = 0


def _positive_conv_weight(shape):
    weight = ConvWeight(
        shape=shape,
        gain=1.0,
        device="cpu",
        clamp=True,
        clamp_min=0.01,
        clamp_max=10.0,
        init_mode="kaiming_uniform",
    )
    with torch.no_grad():
        weight.state.copy_(torch.rand_like(weight.state) + 0.1)
    weight.state.requires_grad_(True)
    return weight


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


def _make_pre_layer(pre_layer_index, shape, batch_size):
    if pre_layer_index == 0:
        return ResistiveInputLayer(shape, gain=1.0, batch_size=batch_size, device="cpu")
    ResistiveInputLayer(shape, gain=1.0, batch_size=batch_size, device="cpu")
    for _ in range(1, pre_layer_index):
        ConvLayer(shape, batch_size=batch_size, device="cpu", non_linearity="linear")
    return ConvLayer(shape, batch_size=batch_size, device="cpu", non_linearity="linear")


def _make_dense_pre_layer(pre_layer_index, shape, batch_size):
    if pre_layer_index == 0:
        return ResistiveInputLayer(shape, gain=1.0, batch_size=batch_size, device="cpu")
    ResistiveInputLayer(shape, gain=1.0, batch_size=batch_size, device="cpu")
    for _ in range(1, pre_layer_index):
        LinearLayer(shape, batch_size=batch_size, device="cpu")
    return LinearLayer(shape, batch_size=batch_size, device="cpu")


def _expected_conv_energy(interaction, pre, post, weight, voltage_amp, current_amp):
    kernel = (weight.shape[2], weight.shape[3])
    layer_pre_index = int(pre._name[-1])
    amp = (current_amp / voltage_amp) ** layer_pre_index
    pre_scale = 1.0 if pre.name == "Layer_0" else voltage_amp
    x = pre.state * pre_scale
    y = post.state
    cols = F.unfold(
        x,
        kernel,
        padding=interaction._P,
        stride=interaction._S,
        dilation=interaction._D,
    )
    batch_size, out_channels, h_out, w_out = y.shape
    patches = cols.transpose(1, 2).unsqueeze(2)
    kernels = weight.get().view(1, 1, out_channels, -1)
    targets = y.view(batch_size, out_channels, h_out * w_out).transpose(1, 2).unsqueeze(-1)
    return 0.5 * ((patches - targets).pow(2) * kernels).sum(dim=(1, 2, 3)) * amp


@pytest.mark.parametrize(
    "interaction_cls",
    [CoreConvResistive, FlexibleConvResistive],
    ids=["core", "flexible"],
)
@pytest.mark.parametrize("pre_layer_index", [0, 1, 2])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize(
    ("voltage_amp", "current_amp"),
    [(1.0, 1.0), (2.0, 1.0), (4.0, 1.0), (1.0, 2.0), (1.0, 4.0), (2.0, 3.0)],
)
def test_conv_resistive_amplification_matches_eval_coefficients_and_weight_gradient(
    interaction_cls, pre_layer_index, stride, voltage_amp, current_amp
):
    torch.manual_seed(0)
    _reset_names()

    batch_size = 4
    pre_shape = (2, 5, 5)
    kernel = (3, 3)
    padding = 0
    out_channels = 3
    h_out = (pre_shape[1] + 2 * padding - kernel[0]) // stride + 1
    w_out = (pre_shape[2] + 2 * padding - kernel[1]) // stride + 1
    post_shape = (out_channels, h_out, w_out)

    pre = _make_pre_layer(pre_layer_index, pre_shape, batch_size)
    post = ConvLayer(post_shape, batch_size=batch_size, device="cpu", non_linearity="linear")
    pre.state = torch.randn(batch_size, *pre_shape, requires_grad=True)
    post.state = torch.randn(batch_size, *post_shape, requires_grad=True)
    weight = _positive_conv_weight((out_channels, pre_shape[0], *kernel))

    interaction = interaction_cls(
        pre,
        post,
        weight,
        padding=padding,
        stride=stride,
        dilation=1,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
    )

    expected_energy = _expected_conv_energy(
        interaction,
        pre,
        post,
        weight,
        voltage_amp,
        current_amp,
    )
    assert torch.allclose(interaction.eval(), expected_energy, atol=2e-5, rtol=1e-5)

    grad_pre, grad_post = torch.autograd.grad(
        interaction.eval().sum(),
        (pre.state, post.state),
        retain_graph=True,
    )
    coef_pre = 2.0 * interaction._a_coef_layer_pre() * pre.state + interaction._b_coef_layer_pre()
    coef_post = (
        2.0 * interaction._a_coef_layer_post() * post.state
        + interaction._b_coef_layer_post()
    )

    assert torch.allclose(grad_pre, coef_pre, atol=2e-5, rtol=1e-5)
    assert torch.allclose(grad_post, coef_post, atol=2e-5, rtol=1e-5)

    grad_weight = torch.autograd.grad(interaction.eval().mean(), weight.state)[0]
    assert torch.allclose(grad_weight, interaction._grad_weight(), atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "interaction_cls",
    [CoreConvResistive, FlexibleConvResistive],
    ids=["core", "flexible"],
)
@pytest.mark.parametrize("pre_layer_index", [0, 1, 2])
@pytest.mark.parametrize(
    ("voltage_amp", "current_amp"),
    [(2.0, 1.0), (1.0, 2.0), (1.0, 4.0), (2.0, 3.0)],
)
def test_conv_1x1_amplification_matches_dense_layer(
    interaction_cls, pre_layer_index, voltage_amp, current_amp
):
    torch.manual_seed(1)

    batch_size = 3
    in_channels = 4
    out_channels = 5
    x = torch.randn(batch_size, in_channels)
    y = torch.randn(batch_size, out_channels)

    _reset_names()
    dense_pre = _make_dense_pre_layer(pre_layer_index, (in_channels,), batch_size)
    dense_post = LinearLayer((out_channels,), batch_size=batch_size, device="cpu")
    dense_pre.state = x.clone()
    dense_post.state = y.clone()
    dense_weight = _positive_dense_weight(dense_pre.shape, dense_post.shape)

    _reset_names()
    conv_pre = _make_pre_layer(pre_layer_index, (in_channels, 1, 1), batch_size)
    conv_post = ConvLayer(
        (out_channels, 1, 1),
        batch_size=batch_size,
        device="cpu",
        non_linearity="linear",
    )
    conv_pre.state = x.clone().view(batch_size, in_channels, 1, 1)
    conv_post.state = y.clone().view(batch_size, out_channels, 1, 1)
    conv_weight = _positive_conv_weight((out_channels, in_channels, 1, 1))
    with torch.no_grad():
        conv_weight.state.copy_(dense_weight.state.t().view(out_channels, in_channels, 1, 1))

    dense = DenseResistive(
        dense_pre,
        dense_post,
        dense_weight,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
    )
    conv = interaction_cls(
        conv_pre,
        conv_post,
        conv_weight,
        padding=0,
        stride=1,
        dilation=1,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
    )

    assert torch.allclose(conv.eval(), dense.eval(), atol=2e-5, rtol=1e-5)
    assert torch.allclose(
        conv._a_coef_layer_pre().view(batch_size, in_channels),
        dense._a_coef_layer_pre().expand(batch_size, in_channels),
        atol=2e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        conv._b_coef_layer_pre().view_as(dense._b_coef_layer_pre()),
        dense._b_coef_layer_pre(),
        atol=2e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        conv._a_coef_layer_post().view_as(dense._a_coef_layer_post()),
        dense._a_coef_layer_post(),
        atol=2e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        conv._b_coef_layer_post().view_as(dense._b_coef_layer_post()),
        dense._b_coef_layer_post(),
        atol=2e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        conv._grad_weight().view(out_channels, in_channels).t(),
        dense._grad_weight(),
        atol=2e-5,
        rtol=1e-5,
    )
