import copy
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from labs.cifar_l8_analog import CifarL8Analog
from experiments.train_cifar_l8_analog import check_boundary_compatibility, make_optimizer


def config():
    c = json.loads(Path('configs/cifar/cifar10_l8_analog_bs32_seed0.json').read_text())
    c.update(blocks=[[2, 2, 2], [3, 3, 3], [4, 4]], input_size=8)
    return c


def test_only_block_outputs_become_linear_and_initial_tensors_match():
    c = config()
    torch.manual_seed(5)
    old = CifarL8Analog(c)
    c['block_output_activation'] = 'linear'
    torch.manual_seed(5)
    new = CifarL8Analog(c)
    assert [[layer.non_linearity for layer in block._free_layers] for block in new.analog_blocks] == [
        ['perfect_diode', 'perfect_diode', 'linear'],
        ['perfect_diode', 'perfect_diode', 'linear'], ['perfect_diode', 'linear']]
    for group, tensors in old.snapshot().items():
        for name, tensor in tensors.items():
            torch.testing.assert_close(tensor, new.snapshot()[group][name], rtol=0, atol=0)


def test_default_architecture_keeps_original_outputs():
    c = config()
    torch.manual_seed(7)
    implicit = CifarL8Analog(c).eval()
    c['block_output_activation'] = 'perfect_diode'
    explicit = CifarL8Analog(c).eval()
    explicit.restore(implicit.snapshot())
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(implicit(x, [6, 6, 4]), explicit(x, [6, 6, 4]), rtol=0, atol=0)


def test_solver_preserves_signed_output_and_unclamped_gradient():
    c = config()
    c.update(blocks=[[2], [3], [4]], block_output_activation='linear')
    model = CifarL8Analog(c)
    block = model.analog_blocks[0]
    with torch.no_grad():
        w = block.weights[0].state
        w.fill_(0.1)
        w[:2, :3] = 1.0
        w[2:, 3:] = 1.0
    x = -torch.ones(2, 3, 8, 8, requires_grad=True)
    decoded = block(x, num_iterations=1, reset=True)
    physical = block._free_layers[-1].state
    assert (physical[:, :2] < 0).all() and (physical[:, 2:] > 0).all()
    assert (decoded < 0).all()
    derivative = torch.autograd.grad(decoded.sum(), physical)[0]
    torch.testing.assert_close(derivative[:, :2], torch.ones_like(derivative[:, :2]))
    torch.testing.assert_close(derivative[:, 2:], -torch.ones_like(derivative[:, 2:]))


@pytest.mark.parametrize('current_amp', [0.25, 1.0])
def test_bptt_optimizer_and_checkpoint_roundtrip(current_amp):
    c = config()
    c['block_output_activation'] = 'linear'
    c['solver']['current_amp'] = current_amp
    torch.manual_seed(13)
    model = CifarL8Analog(c)
    optimizer, scheduler = make_optimizer(model, c)
    x = torch.randn(3, 3, 8, 8)
    y = torch.tensor([0, 1, 2])
    _, tracked = model.bptt(x, [6, 6, 4])
    F.cross_entropy(tracked, y).backward()
    for name, p in model.trainable_tensors().items():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0, name
    assert all(group['weight_decay'] == 0 for group in optimizer.param_groups)
    optimizer.step()
    model.project_()
    scheduler.step()
    restored = CifarL8Analog(c).eval()
    restored.restore(model.snapshot())
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x, [6, 6, 4]), restored(x, [6, 6, 4]), rtol=0, atol=0)


def test_incompatible_assets_and_invalid_activation_are_rejected():
    original = config()
    linear = copy.deepcopy(original)
    linear['block_output_activation'] = 'linear'
    with pytest.raises(ValueError, match='output activation mismatch'):
        check_boundary_compatibility(original, linear)
    with pytest.raises(ValueError, match='output activation mismatch'):
        check_boundary_compatibility(linear, original)
    linear['block_output_activation'] = 'typo'
    with pytest.raises(ValueError, match='block_output_activation'):
        CifarL8Analog(linear)
