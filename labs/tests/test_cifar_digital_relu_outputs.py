"""Linear analog block outputs followed by a digital rectifier ('digital_relu')."""
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from experiments.calibrate_cifar_block_beta import capture_batches
from experiments.compare_cifar_block_beta_gradients import agreement, eqprop_gradients, local_bptt_gradients
from experiments.replay_cifar_precision_response import convert_model
from labs.cifar_l8_analog import CifarL8Analog


def config(activation):
    c = json.loads(Path('configs/cifar/cifar10_l8_analog_bs32_seed0.json').read_text())
    c.update(blocks=[[2, 2, 2], [3, 3, 3], [4, 4]], input_size=8, block_output_activation=activation)
    return c


def test_only_output_layers_are_analog_linear_and_initial_tensors_match():
    torch.manual_seed(5)
    linear = CifarL8Analog(config('linear'))
    torch.manual_seed(5)
    relu = CifarL8Analog(config('digital_relu'))
    assert [[layer.non_linearity for layer in block._free_layers] for block in relu.analog_blocks] == [
        ['perfect_diode', 'perfect_diode', 'linear'],
        ['perfect_diode', 'perfect_diode', 'linear'], ['perfect_diode', 'linear']]
    for group, tensors in linear.snapshot().items():
        for name, tensor in tensors.items():
            torch.testing.assert_close(tensor, relu.snapshot()[group][name], rtol=0, atol=0)


def test_digital_rectifier_matches_diode_clipping_and_gates_force():
    torch.manual_seed(11)
    block = CifarL8Analog(config('digital_relu')).analog_blocks[0]
    x = torch.randn(3, 3, 8, 8)
    decoded = block(x, num_iterations=6, reset=True)
    physical = block.free_layers()[-1].state
    positive, negative = physical.chunk(2, dim=1)
    assert (positive < 0).any() and (negative > 0).any(), 'test needs both rectified regions'
    torch.testing.assert_close(decoded, positive.clamp(min=0) - negative.clamp(max=0), rtol=0, atol=0)
    assert (decoded >= 0).all()
    force = torch.autograd.grad(decoded.sum(), physical)[0]
    expected = torch.cat(((positive > 0).to(force.dtype), -(negative < 0).to(force.dtype)), dim=1)
    torch.testing.assert_close(force, expected, rtol=0, atol=0)


def test_linear_and_digital_relu_share_the_analog_equilibrium():
    states = {}
    for activation in ('linear', 'digital_relu'):
        torch.manual_seed(3)
        block = CifarL8Analog(config(activation)).analog_blocks[0]
        with torch.no_grad():
            block(torch.ones(2, 3, 8, 8), num_iterations=6, reset=True)
        states[activation] = [layer.state.clone() for layer in block.free_layers()]
    for a, b in zip(states['linear'], states['digital_relu']):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_bptt_reaches_every_trainable_tensor():
    torch.manual_seed(13)
    model = CifarL8Analog(config('digital_relu'))
    _, tracked = model.bptt(torch.randn(3, 3, 8, 8), [6, 6, 4])
    F.cross_entropy(tracked, torch.tensor([0, 1, 2])).backward()
    for name, p in model.trainable_tensors().items():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0, name


@pytest.mark.parametrize('activation', ['perfect_diode', 'linear', 'digital_relu'])
def test_small_beta_eqprop_matches_local_bptt(activation):
    torch.manual_seed(29)
    model = CifarL8Analog(config(activation))
    convert_model(model, torch.float64)
    model.train()
    batch = [(torch.randn(4, 3, 8, 8, dtype=torch.float64), torch.tensor([0, 1, 2, 3]))]
    records, = capture_batches(model, batch, [6, 6, 4], torch.device('cpu'))
    for index, (block, saved) in enumerate(zip(model.analog_blocks, records)):
        iterations = [6, 6, 4][index]
        reference = local_bptt_gradients(block, saved, iterations, torch.device('cpu'))
        estimate, _ = eqprop_gradients(block, saved, 1e-8, iterations, torch.device('cpu'))
        for e, r in zip(estimate, reference):
            result = agreement(e, r)
            assert result['cosine'] > .999 and abs(result['norm_ratio'] - 1) < .01, (activation, index, result)
