import torch

from model.function.network import Network
from model.hopfield.interaction import ConvHopfieldNoPool
from model.hopfield.layer import HardSigmoidLayer
from model.hopfield.minimizer import FixedPointMinimizer
from model.hopfield.network import FlexibleConvHopfieldEnergy
from model.variable.layer import InputLayer
from model.variable.parameter import ConvWeight


def test_no_pool_hopfield_conv_gradients_match_autograd():
    torch.manual_seed(0)
    batch_size = 3
    pre = InputLayer((1, 7, 7))
    post = HardSigmoidLayer((2, 4, 4))
    weight = ConvWeight(
        (2, 1, 3, 3),
        gain=0.1,
        device=None,
        clamp=False,
        clamp_min=None,
        clamp_max=None,
    )
    pre.state = torch.randn(batch_size, 1, 7, 7, requires_grad=True)
    post.state = torch.randn(batch_size, 2, 4, 4, requires_grad=True)
    weight.state = torch.randn(2, 1, 3, 3, requires_grad=True)

    interaction = ConvHopfieldNoPool(pre, post, weight, padding=1, stride=2)
    value = interaction.eval().sum()
    grad_pre, grad_post, grad_weight_sum = torch.autograd.grad(
        value,
        [pre.state, post.state, weight.state],
    )

    assert interaction.eval().shape == (batch_size,)
    torch.testing.assert_close(interaction.grad_layer_fn(pre)(), grad_pre)
    torch.testing.assert_close(interaction.grad_layer_fn(post)(), grad_post)
    torch.testing.assert_close(
        interaction.grad_param_fn(weight)(),
        grad_weight_sum / batch_size,
    )


def test_fixed_point_minimizer_constructs_and_runs_for_flexible_conv_hopfield():
    torch.manual_seed(1)
    energy = FlexibleConvHopfieldEnergy(
        layer_shapes=[(1, 8, 8), (4, 3, 3), (10,)],
        weight_gains=[0.6, 1.5],
        conv_pipeline=[
            {
                "mode": "convolution",
                "kernel": [3, 3],
                "stride": 2,
                "padding": 0,
            }
        ],
    )
    energy.set_device("cpu")
    network = Network(energy)
    network.set_input(torch.rand(2, 1, 8, 8), reset=True)
    minimizer = FixedPointMinimizer(
        energy,
        network.free_layers(),
        num_iterations=1,
        mode="asynchronous",
    )
    layers = minimizer.compute_equilibrium()

    assert set(layers) == {layer.name for layer in energy.layers()}
    assert energy.layers()[1].state.shape == (2, 4, 3, 3)
    assert energy.layers()[-1].state.shape == (2, 10)
