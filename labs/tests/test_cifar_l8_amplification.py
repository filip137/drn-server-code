"""Independent physical-KCL and MNIST-solver checks for the CIFAR circuits.

Small synthetic circuits exercise the production assembly, including its
digital boundaries. No dataset, checkpoint, or optimizer step is needed.
"""

import copy
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from labs.cifar_l8_analog import (
    AnalogDenseReadout,
    AnalogSolverConfig,
    CifarL8Analog,
    VoltageClampedConvBlock,
)
from labs.custom_classes import ConvResistive as MnistConvResistive
from labs.custom_classes import TrackingQuadraticMinimizer
from model.function.interaction import SumSeparableFunction
from model.resistive.minimizer import QuadraticUpdater


SCHEMES = [(1., 1.), (4., 1.), (4., .25), (2., 3.)]
CONFIG = Path(__file__).resolve().parents[2] / 'configs/cifar/cifar10_l8_analog_bs16_seed0.json'


def config(voltage_amp, current_amp):
    result = json.loads(CONFIG.read_text())
    result['solver'].update(voltage_amp=voltage_amp, current_amp=current_amp)
    result.update(blocks=[[2, 2, 2], [3, 3, 3], [4, 4]], input_size=8)
    return result


def block(depth, voltage_amp, current_amp):
    c = config(voltage_amp, current_amp)
    return VoltageClampedConvBlock(
        logical_input_channels=2, analog_channels=[4] * depth, spatial_size=5,
        solver=AnalogSolverConfig.from_mapping(c['solver']), input_gain_init=2.,
        weight_gain=c['weight_gain'], weight_init_mode=c['weight_init_mode'],
        weight_min=c['weight_min'], weight_max=c['weight_max'], device='cpu',
    )


def physical_drive_and_degree(circuit, index):
    """Unscaled physical KCL, assembled without energy/coefficient methods.

    Incoming current: sum g*(alpha*v_previous - v).
    Outgoing reflected current: B*sum g*(v_next - A*v).
    Zero padding represents the same grounded boundary used by the circuit.
    """
    states = [circuit.boundary_layer.state] + [l.state for l in circuit.free_layers()]
    a, b = circuit.solver.voltage_amp, circuit.solver.current_amp
    incoming = circuit.weights[index].state
    pre_scale = 1. if index == 0 else a
    drive = F.conv2d(pre_scale * states[index], incoming, padding=1)
    degree = incoming.sum((1, 2, 3))[None, :, None, None]
    if index + 1 < len(circuit.weights):
        outgoing = circuit.weights[index + 1].state
        next_state = states[index + 2]
        drive = drive + b * F.conv_transpose2d(next_state, outgoing, padding=1)
        degree = degree + a * b * F.conv_transpose2d(
            torch.ones_like(next_state), outgoing, padding=1,
        )
    return drive, degree


@pytest.mark.parametrize('voltage_amp,current_amp', SCHEMES)
@pytest.mark.parametrize('depth', [2, 3])
def test_block_energy_and_coordinate_updates_match_physical_kcl(depth, voltage_amp, current_amp):
    torch.manual_seed(73)
    circuit = block(depth, voltage_amp, current_amp).double()
    circuit.set_input_voltage(torch.randn(2, 2, 5, 5, dtype=torch.float64))
    for layer in circuit.free_layers():
        layer.state = torch.randn(2, *layer.shape, dtype=torch.float64, requires_grad=True)
    derivatives = torch.autograd.grad(
        circuit.energy.eval().sum(), [l.state for l in circuit.free_layers()],
    )
    for index, (layer, derivative, updater) in enumerate(zip(
        circuit.free_layers(), derivatives, circuit.minimizer._updaters,
    )):
        drive, degree = physical_drive_and_degree(circuit, index)
        diagonal = (current_amp / voltage_amp) ** index
        expected_derivative = diagonal * (degree * layer.state - drive)
        torch.testing.assert_close(derivative, expected_derivative, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(updater.pre_activate(), drive / degree, rtol=2e-12, atol=2e-12)
        # The diode balances the residual only on a clamped coordinate.
        old_state = layer.state
        layer.state = updater.pre_activate()
        constrained = layer.activate()
        current = drive - degree * constrained
        positive, negative = constrained.chunk(2, dim=1)
        positive_current, negative_current = current.chunk(2, dim=1)
        assert positive_current[positive == 0].max().item() <= 2e-12
        assert negative_current[negative == 0].min().item() >= -2e-12
        torch.testing.assert_close(current[constrained != 0], torch.zeros_like(current[constrained != 0]),
                                   rtol=0, atol=2e-12)
        layer.state = old_state


@pytest.mark.parametrize('voltage_amp,current_amp', SCHEMES)
@pytest.mark.parametrize('depth', [2, 3])
def test_mnist_tracking_solver_matches_cifar_states_and_gradients(depth, voltage_amp, current_amp):
    torch.manual_seed(91)
    cifar = block(depth, voltage_amp, current_amp)
    mnist = block(depth, voltage_amp, current_amp)
    for original, other in zip(cifar.weights, mnist.weights):
        with torch.no_grad():
            other.state.copy_(original.state)
    layers = [mnist.boundary_layer, *mnist.free_layers()]
    edges = [MnistConvResistive(
        layers[i], layers[i + 1], weight, padding=1, stride=1, dilation=1,
        voltage_amp=voltage_amp, current_amp=current_amp,
    ) for i, weight in enumerate(mnist.weights)]
    # The production analog biases are exactly zero; retain those interactions.
    energy = SumSeparableFunction(
        layers=layers, params=[*mnist.weights, *mnist.biases],
        interactions=edges + mnist.energy._interactions[depth:],
    )
    s = mnist.solver
    mnist.minimizer = TrackingQuadraticMinimizer(
        fn=energy, free_layers=mnist.free_layers(), num_iterations=6,
        mode=s.mode, non_linearity=s.non_linearity,
        quadratic_diode_param=dict(s.quadratic_diode_param),
        exponential_diode_param=dict(s.exponential_diode_param),
        hard_sigmoid_param=dict(s.hard_sigmoid_param),
        voltage_amp=voltage_amp, current_amp=current_amp,
        iv_data=s.iv_data, iv_data_path=s.iv_data_path,
        double_diode_updater=s.double_diode_updater,
        single_diode_updater=s.single_diode_updater,
        adaptive_equilibrium=s.adaptive_equilibrium,
        overrelaxation_factor=s.overrelaxation_factor, minimizer_settings=s.minimizer_settings,
        damping=s.damping, experimental_newton_max_steps=s.experimental_newton_max_steps,
    )
    x = torch.randn(2, 2, 5, 5)
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    ya = cifar(xa, num_iterations=6, reset=True)
    yb = mnist(xb, num_iterations=6, reset=True)
    for left, right in zip(cifar.free_layers(), mnist.free_layers()):
        torch.testing.assert_close(left.state, right.state, rtol=0, atol=0)
    ga = torch.autograd.grad(ya.square().mean(), [xa, *cifar.analog_parameters()])
    gb = torch.autograd.grad(yb.square().mean(), [xb, *mnist.analog_parameters()])
    for left, right in zip(ga, gb):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize('voltage_amp,current_amp', SCHEMES)
def test_analog_classifier_matches_standard_updater_and_physical_kcl(voltage_amp, current_amp):
    torch.manual_seed(37)
    c = config(voltage_amp, current_amp)
    head = AnalogDenseReadout(6, 3, c, 'cpu')
    x = torch.randn(2, 6, requires_grad=True)
    actual = head(x)
    energy = SumSeparableFunction(
        layers=[head.boundary, head.output], params=[head.weight], interactions=[head.interaction],
    )
    standard = QuadraticUpdater(head.output, energy).pre_activate()
    torch.testing.assert_close(actual, standard, rtol=0, atol=0)
    w, voltage = head.weight.state, head.boundary.state
    branch_currents = w[None, :, :] * (voltage[:, :, None] - actual[:, None, :])
    residual = branch_currents.sum(dim=1)
    scale = branch_currents.abs().sum(dim=1).clamp_min(1.)
    assert (residual.abs() / scale).max().item() < 2e-6
    baseline = AnalogDenseReadout(6, 3, config(1., 1.), 'cpu')
    with torch.no_grad():
        baseline.weight.state.copy_(w)
    # A separately clamped one-edge circuit has no internal amplified edge.
    torch.testing.assert_close(actual, baseline(x), rtol=0, atol=0)


def test_legacy_block_scaling_and_bn_boundary_cancellation():
    torch.manual_seed(123)
    baseline = CifarL8Analog(config(1., 1.))
    legacy = CifarL8Analog(config(4., .25))
    legacy.restore(copy.deepcopy(baseline.snapshot()))
    baseline.train(); legacy.train()
    with torch.no_grad():
        for left, right in zip(baseline.analog_blocks, legacy.analog_blocks):
            x = torch.randn(4, left.logical_input_channels, left.spatial_size, left.spatial_size)
            a = left(x, num_iterations=6, reset=True)
            b = right(x, num_iterations=6, reset=True)
            scale = 4 ** (len(left.weights) - 1)
            torch.testing.assert_close(b / scale, a, rtol=0, atol=0)
        # Epsilon is disabled only in this algebraic test, never in production.
        for model in (baseline, legacy):
            for bridge in model.bridges:
                bridge[1].eps = 0.
        x = torch.randn(4, 3, 8, 8)
        torch.testing.assert_close(baseline(x, [6, 6, 4]), legacy(x, [6, 6, 4]), rtol=0, atol=0)


def test_voltage_normalization_matches_baseline_bptt_with_production_bn_epsilon():
    torch.manual_seed(43)
    c = config(1., 1.)
    baseline = CifarL8Analog(c)
    legacy_config = config(4., .25)
    legacy_config['block_output_normalization'] = 'voltage'
    legacy = CifarL8Analog(legacy_config)
    legacy.restore(baseline.snapshot())
    assert legacy.block_output_scales == (16., 16., 4.)
    assert all(b[1].eps == 1e-5 for b in legacy.bridges)
    baseline.train(); legacy.train()
    x = torch.randn(4, 3, 8, 8)
    y = torch.tensor([0, 1, 2, 3])
    outputs = []
    for model in (baseline, legacy):
        free, tracked = model.bptt(x, [6, 6, 4])
        F.cross_entropy(tracked, y).backward()
        outputs.append((free, tracked))
    for left, right in zip(*outputs):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for name, left in baseline.trainable_tensors().items():
        torch.testing.assert_close(left.grad, legacy.trainable_tensors()[name].grad, rtol=0, atol=0)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, legacy.state_dict()[name], rtol=0, atol=0)
    baseline.eval(); legacy.eval()
    with torch.no_grad():
        torch.testing.assert_close(baseline(x, [6, 6, 4]), legacy(x, [6, 6, 4]), rtol=0, atol=0)


def test_normalization_default_preserves_old_checkpoint_interface():
    torch.manual_seed(47)
    c = config(4., .25)
    old = CifarL8Analog(c)
    explicit = CifarL8Analog(dict(c, block_output_normalization='none'))
    explicit.restore(old.snapshot())
    x = torch.randn(4, 3, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(old(x, [6, 6, 4]), explicit(x, [6, 6, 4]), rtol=0, atol=0)
    with pytest.raises(ValueError, match='block_output_normalization'):
        CifarL8Analog(dict(c, block_output_normalization='invalid'))
