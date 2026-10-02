"""Coupled nonlinear readout, gradient, checkpoint and precision regressions."""
import copy

import pytest
import torch
from torch.nn import functional as F

from experiments.replay_cifar_precision_response import convert_model, verify_dtype
from labs.cifar_l8_analog import AnalogDenseReadout, CifarL8Analog, NonlinearAnalogDenseReadout
from labs.tests.test_cifar_l8_analog import config as original_config


def config(hidden_units=4):
    result = original_config()
    result.update(head_hidden_units=hidden_units, head_iterations=4)
    result['solver']['current_amp'] = 1.
    return result


def test_coupled_head_polarities_equations_and_finite_difference():
    head = NonlinearAnalogDenseReadout(2, 2, config(), 'cpu')
    for variable in head.analog_variables():
        variable.state = variable.state.detach().double().requires_grad_(variable.state.requires_grad)
    head.double()
    with torch.no_grad():
        head.hidden_weight.state.copy_(torch.tensor([
            [.7, .4, .1, .2], [.2, .6, .3, .1],
            [.1, .2, .7, .4], [.3, .1, .2, .6]], dtype=torch.float64))
        head.weight.state.copy_(torch.tensor([
            [.4, .2], [.1, .5], [.2, .4], [.5, .1]], dtype=torch.float64))
    values = torch.tensor([[.7, .2]], dtype=torch.float64, requires_grad=True)
    actual = head(values)
    boundary = 100 * torch.cat((values, -values), dim=1)
    first, final = head.hidden_weight.state, head.weight.state
    expected = torch.zeros_like(actual)
    for _ in range(4):
        hidden = (boundary @ first + expected @ final.T) / (first.sum(0) + 4 * final.sum(1))
        # All four units remain strictly inside their permitted half-lines.
        assert (hidden[:, :2] > .1).all() and (hidden[:, 2:] < -.1).all()
        expected = 4 * (hidden @ final) / final.sum(0)
    torch.testing.assert_close(head.hidden.state, hidden)
    torch.testing.assert_close(actual, expected)
    assert [head.boundary.name, head.hidden.name, head.output.name] == ['Layer_0', 'Layer_1', 'Layer_2']
    assert head.minimizer.equilibrium_iteration_stats()['last_iterations'] == 4

    tensors = [values, first, final, head._input_gain_raw]
    gradients = torch.autograd.grad(actual.square().mean(), tensors)
    for tensor, gradient in zip(tensors, gradients):
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        direction = torch.linspace(.1, 1., tensor.numel(), dtype=tensor.dtype).reshape(tensor.shape)
        step = 1e-5
        with torch.no_grad():
            initial = tensor.clone()
            tensor.copy_(initial + step * direction)
            plus = head(values).square().mean().item()
            tensor.copy_(initial - step * direction)
            minus = head(values).square().mean().item()
            tensor.copy_(initial)
        numerical = torch.tensor((plus - minus) / (2 * step), dtype=tensor.dtype)
        torch.testing.assert_close((gradient * direction).sum(), numerical, rtol=2e-6, atol=1e-6)


def test_snapshot_restore_projection_and_optimizer_cover_both_matrices():
    torch.manual_seed(10)
    c = config()
    model = CifarL8Analog(c).eval()
    saved = model.snapshot()
    assert list(saved['conductances'])[-2:] == ['head.hidden_weight', 'head.weight']
    inputs = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        expected = model(inputs, c['iterations'])
        model.head.hidden_weight.state.fill_(-1)
        model.head.weight.state.fill_(100)
    model.project_()
    assert torch.all(model.head.hidden_weight.state == c['weight_min'])
    assert torch.all(model.head.weight.state == c['weight_max'])
    model.restore(saved)
    fresh = CifarL8Analog(c).eval()
    fresh.restore(saved)
    with torch.no_grad():
        for restored in (model, fresh):
            torch.testing.assert_close(restored(inputs, c['iterations']), expected, rtol=0, atol=0)
    groups = model.optimizer_groups(c['optimizer'])
    heads = [group for group in groups if group['name'] in ('head.hidden_weight', 'head.weight')]
    assert len(heads) == 2
    assert all(group['lr'] == c['optimizer']['head_learning_rate'] for group in heads)
    assert {id(p) for group in groups for p in group['params']} == {
        id(p) for p in model.trainable_tensors().values()}


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_conversion_reset_detach_and_full_model_bptt(dtype):
    torch.manual_seed(19)
    c = config()
    model = CifarL8Analog(c)
    convert_model(model, torch.float64)
    convert_model(model, dtype)
    x = torch.randn(2, 3, 8, 8, dtype=dtype)
    free, tracked = model.bptt(x, c['iterations'])
    assert free.dtype == tracked.dtype == dtype
    tracked.square().mean().backward()
    for name, parameter in model.trainable_tensors().items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    verify_dtype(model, dtype)
    model.detach_state_()
    assert all(layer.state.grad_fn is None for layer in
               (model.head.boundary, model.head.hidden, model.head.output))
    with pytest.raises(ValueError, match='different batch size'):
        model.head(torch.zeros(3, 4, dtype=dtype), reset=False)
    with torch.no_grad():
        model(torch.randn(3, 3, 8, 8, dtype=dtype), c['iterations'], reset=True)
    verify_dtype(model, dtype)


def test_default_single_head_keeps_exact_outputs_and_checkpoint_keys():
    implicit = original_config()
    explicit = copy.deepcopy(implicit)
    explicit.update(head_hidden_units=0, head_iterations=4)
    torch.manual_seed(37)
    first = CifarL8Analog(implicit)
    torch.manual_seed(37)
    second = CifarL8Analog(explicit)
    assert isinstance(first.head, AnalogDenseReadout)
    assert first.head.analog_variables() == (first.head.boundary, first.head.output, first.head.weight)
    for group, entries in first.snapshot().items():
        assert entries.keys() == second.snapshot()[group].keys()
        for name, value in entries.items():
            assert torch.equal(value, second.snapshot()[group][name])
    values = torch.randn(2, 4)
    voltage = F.softplus(first.head._input_gain_raw) * torch.cat((values, -values), 1)
    weight = first.head.weight.state
    expected = (voltage @ weight) / weight.sum(0)
    torch.testing.assert_close(first.head(values), expected)
    torch.testing.assert_close(first.head(values, reset=False), second.head(values), rtol=0, atol=0)


@pytest.mark.parametrize('hidden_units', [-2, 3, 2.5])
def test_invalid_hidden_width_is_rejected(hidden_units):
    with pytest.raises(ValueError, match='head_hidden_units'):
        CifarL8Analog(config(hidden_units))
