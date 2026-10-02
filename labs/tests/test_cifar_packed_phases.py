"""Packed centered phases preserve signs, sample normalization and block state."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path

import pytest
import torch

from experiments import compare_cifar_block_beta_gradients as comparison
from labs.cifar_l8_analog import AnalogSolverConfig, VoltageClampedConvBlock
from labs.custom_minimizer import CustomQuadraticMinimizer


def block_and_saved(batch_size):
    torch.manual_seed(71)
    config = json.loads((Path(__file__).resolve().parents[2] /
        'configs/cifar/fresh_readout_lr_20260928/ours.json').read_text())
    block = VoltageClampedConvBlock(logical_input_channels=1,
        analog_channels=[2, 4, 2], spatial_size=3,
        solver=AnalogSolverConfig.from_mapping(config['solver']),
        input_gain_init=1., weight_gain=1., weight_init_mode='kaiming_uniform',
        weight_min=1e-7, weight_max=10., device='cpu').double()
    with torch.no_grad():
        block(torch.randn(batch_size, 1, 3, 3, dtype=torch.float64),
              num_iterations=5, reset=True)
    saved = dict(input=block.boundary_layer.state.clone(),
        states=[layer.state.clone() for layer in block.free_layers()],
        gradient=torch.randn_like(block.free_layers()[-1].state) * .03)
    return block, saved


def assert_saved_equal(actual, expected):
    for a, b in zip([actual['input'], actual['gradient'], *actual['states']],
                    [expected['input'], expected['gradient'], *expected['states']]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def assert_displacement_close(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, list):
            assert len(actual[key]) == len(value)
            for a, b in zip(actual[key], value):
                assert_displacement_close(a, b)
        elif isinstance(value, float):
            assert actual[key] == pytest.approx(value, rel=1e-11, abs=1e-15)
        else:
            assert actual[key] == value


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('iterations', [1, 6])
def test_packed_matches_sequential_endpoints_gradients_and_displacements(
        batch_size, iterations, monkeypatch):
    block, saved = block_and_saved(batch_size)
    original_saved = copy.deepcopy(saved)
    original_weights = [weight.state.clone() for weight in block.weights]
    rng = torch.get_rng_state().clone()
    calls = []
    original_solve = CustomQuadraticMinimizer.compute_equilibrium

    def solve(self):
        calls.append((self.num_iterations, block.boundary_layer.state.shape[0]))
        return original_solve(self)

    monkeypatch.setattr(CustomQuadraticMinimizer, 'compute_equilibrium', solve)
    expected, expected_displacement = comparison.eqprop_gradients(
        block, saved, .17, iterations, 'cpu', measure_layer_displacement=True)
    expected_states = [layer.state.clone() for layer in block.free_layers()]
    assert calls == [(iterations, batch_size)] * 2
    calls.clear()
    actual, actual_displacement = comparison.eqprop_gradients(
        block, saved, .17, iterations, 'cpu', phase_mode='packed',
        cpu_output=False, measure_layer_displacement=True)
    assert calls == [(iterations, 2 * batch_size)]
    for a, b in zip(actual, expected):
        assert a.dtype == torch.float64 and not a.requires_grad
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-14)
    assert any(torch.any(g != 0) for g in actual)
    assert_displacement_close(actual_displacement, expected_displacement)
    for layer, state in zip(block.free_layers(), expected_states):
        assert layer.state.shape[0] == batch_size and not layer.state.requires_grad
        torch.testing.assert_close(layer.state, state, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(block.boundary_layer.state, saved['input'], rtol=0, atol=0)
    assert_saved_equal(saved, original_saved)
    for weight, state in zip(block.weights, original_weights):
        torch.testing.assert_close(weight.state, state, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)


def test_coefficient_cache_handles_sequential_packed_sequential_batches(monkeypatch):
    block, saved = block_and_saved(3)
    interaction = block.interactions[1]
    coefficient_batches = []
    original = interaction._a_col2im_uncached

    def coefficient():
        coefficient_batches.append(interaction._layer_post.state.shape[0])
        return original()

    monkeypatch.setattr(interaction, '_a_col2im_uncached', coefficient)
    results = []
    with ExitStack() as stack:
        for fn in block.interactions:
            stack.enter_context(fn.cache_a_col2im())
        for mode in ('sequential', 'packed', 'sequential'):
            results.append(comparison.eqprop_gradients(
                block, saved, .17, 6, 'cpu', phase_mode=mode,
                measure_displacement=False))
    assert coefficient_batches == [3, 6, 3]
    assert all(fn._a_col2im_cache is None for fn in block.interactions)
    for gradients, displacement in results:
        assert displacement == {}
        for a, b in zip(gradients, results[0][0]):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-14)


def test_packed_restores_original_batch_after_failed_solve(monkeypatch):
    block, saved = block_and_saved(3)
    original_saved = copy.deepcopy(saved)

    def fail(self):
        assert block.boundary_layer.state.shape[0] == 6
        raise RuntimeError('injected solver failure')

    monkeypatch.setattr(CustomQuadraticMinimizer, 'compute_equilibrium', fail)
    with pytest.raises(RuntimeError, match='injected solver failure'):
        comparison.eqprop_gradients(block, saved, .17, 6, 'cpu', phase_mode='packed')
    torch.testing.assert_close(block.boundary_layer.state, saved['input'], rtol=0, atol=0)
    for layer, state in zip(block.free_layers(), saved['states']):
        torch.testing.assert_close(layer.state, state, rtol=0, atol=0)
    assert_saved_equal(saved, original_saved)


def test_unknown_phase_mode_rejected_before_state_mutation():
    block, saved = block_and_saved(1)
    states = [v.state for v in [block.boundary_layer, *block.free_layers()]]
    with pytest.raises(ValueError, match='phase_mode'):
        comparison.eqprop_gradients(block, saved, .17, 6, 'cpu', phase_mode='invalid')
    assert all(v.state is state for v, state in
               zip([block.boundary_layer, *block.free_layers()], states))
