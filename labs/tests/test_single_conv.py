import torch

from custom_classes import ConvLayer, ConvResistive, DetailedSumSeparableFunction
from model.resistive.minimizer import QuadraticMinimizer
from model.variable.parameter import ConvWeight


def _default_quadratic_params():
    return {"diode_conductance": 1.0, "v_min": -1e6, "v_max": 1e6}


def _default_exponential_params():
    return {"I_s": 1e-6, "V_t": 0.025, "V_off": 0.0}


def test_single_conv_layer_quadratic_update_reduces_gradient():
    torch.manual_seed(0)

    batch_size = 2
    input_shape = (2, 6, 6)
    kernel_size = (3, 3)
    stride = 1
    padding = 0

    # Derived output spatial size: (6 - 3 + 1) = 4
    output_shape = (4, 4, 4)

    input_layer = ConvLayer(input_shape, device="cpu")
    output_layer = ConvLayer(output_shape, device="cpu")

    conv_weight = ConvWeight(
        shape=(output_shape[0], input_shape[0], kernel_size[0], kernel_size[1]),
        gain=0.5,
        device="cpu",
        clamp=False,
        clamp_min=None,
        clamp_max=None,
        mode="convolution",
    )

    conv_interaction = ConvResistive(
        input_layer,
        output_layer,
        conv_weight,
        padding=padding,
        stride=stride,
        dilation=1,
    )

    energy_fn = DetailedSumSeparableFunction(
        layers=[input_layer, output_layer],
        params=[conv_weight],
        interactions=[conv_interaction],
    )

    # Initialize states
    input_layer.state = torch.randn(batch_size, *input_shape)
    output_layer.state = torch.zeros(batch_size, *output_shape)

    minimizer = QuadraticMinimizer(
        energy_fn,
        free_layers=[output_layer],
        num_iterations=3,
        mode="forward",
        non_linearity="linear",
        quadratic_diode_param=_default_quadratic_params(),
        exponential_diode_param=_default_exponential_params(),
        voltage_amp=1.0,
        current_amp=1.0,
    )

    grad_fn = energy_fn.grad_layer_fn(output_layer)
    grad_before = grad_fn().norm().item()
    minimizer.compute_equilibrium()
    grad_after = grad_fn().norm().item()

    # Quadratic closed-form update should drive gradient close to zero.
    assert grad_after < grad_before * 1e-3, (
        f"Gradient did not decrease enough for single conv layer: "
        f"before={grad_before}, after={grad_after}"
    )
