"""The inactive-channel force mask changes only nudging, never the BPTT target."""
import copy
import json
from pathlib import Path

import pytest
import torch

from experiments import cifar_conv_eqprop as hybrid
from experiments.calibrate_cifar_block_beta import (
    FrozenOutputCost, capture_batches, inactive_output_channels, mask_output_force, nudged_minimizer,
)
from experiments.compare_cifar_block_beta_gradients import (
    agreement, local_bptt_gradients, reference_record, restore_block, stable_centered_conv_gradient,
)
from experiments.train_cifar_l8_analog import deterministic_setup
from labs.cifar_l8_analog import AnalogSolverConfig, CifarL8Analog, VoltageClampedConvBlock
from labs.tests.test_cifar_conv_eqprop_training import config as hybrid_config
from model.resistive.interaction import ConvResistive
from training.sgd import AugmentedFunction, EquilibriumProp


def kill_output_channel(block, channel):
    """Wire one logical output channel so both diode halves stay off.

    Its excitatory unit sees the previous layer's nonpositive inhibitory states and its
    inhibitory unit sees the nonnegative excitatory states, through maximal conductances.
    """
    weight = block.weights[-1]
    out_half, in_half = weight.state.shape[0] // 2, weight.state.shape[1] // 2
    with torch.no_grad():
        weight.state[channel, :in_half] = block.weight_min
        weight.state[channel, in_half:] = block.weight_max
        weight.state[out_half + channel, :in_half] = block.weight_max
        weight.state[out_half + channel, in_half:] = block.weight_min


def small_block():
    c = json.loads((Path(__file__).resolve().parents[2] / 'configs/cifar/fresh_readout_lr_20260928/ours.json').read_text())
    return VoltageClampedConvBlock(logical_input_channels=2, analog_channels=[8, 4], spatial_size=4,
        solver=AnalogSolverConfig.from_mapping(c['solver']), input_gain_init=1., weight_gain=1.,
        weight_init_mode='kaiming_uniform', weight_min=1e-7, weight_max=10., device='cpu')


def phase_endpoints(block, saved, force, beta, iterations):
    restore_block(block, saved, 'cpu')
    cost = FrozenOutputCost(block.free_layers()[-1])
    cost.gradient = force
    augmented = AugmentedFunction(block.energy, cost, nudging_mode='current', current_scale=1.)
    estimator = EquilibriumProp(block.weights, block.free_layers(), augmented, cost,
                                nudged_minimizer(block, augmented, iterations), variant='centered', nudging=beta)
    ordered = [next(fn for fn in block.energy._interactions if isinstance(fn, ConvResistive) and fn._weight is w)
               for w in block.weights]
    endpoints = {}

    def gradients(minus, plus):
        endpoints.update(minus={k: v.clone() for k, v in minus.items()}, plus={k: v.clone() for k, v in plus.items()})
        return [stable_centered_conv_gradient(fn, minus, plus, beta) for fn in ordered]

    estimator._standard_param_grads = gradients
    with torch.no_grad():
        result = [g.detach().double() for g in estimator.compute_gradient()]
    block.detach_state_()
    return result, endpoints


def test_inactive_channels_require_both_halves_off():
    state = torch.randn(3, 8, 2, 2)
    state[:, [1, 5]] = 0.
    state[:, 2] = 0.  # only the excitatory half of logical channel 2 is off
    assert inactive_output_channels(state).tolist() == [False, True, False, False]
    with pytest.raises(ValueError, match='even channel count'):
        inactive_output_channels(torch.zeros(2, 3, 2, 2))


def test_mask_removes_only_inactive_physical_channels():
    state = torch.randn(3, 8, 2, 2)
    state[:, [1, 5]] = 0.
    force = torch.randn_like(state)
    same, record = mask_output_force(force, state, 'none')
    assert same is force and record == {'force_mask': 'none'}
    masked, record = mask_output_force(force, state, 'inactive_output_channels')
    keep = [0, 2, 3, 4, 6, 7]
    assert torch.equal(masked[:, keep], force[:, keep]) and not masked[:, [1, 5]].any()
    share = force[:, [1, 5]].double().square().sum() / force.double().square().sum()
    assert record['inactive_channels'] == 1
    assert record['masked_force_square_share'] == pytest.approx(float(share), rel=1e-12)
    with pytest.raises(ValueError, match='Expected force_mask'):
        mask_output_force(force, state, 'zero_everything')


def test_masked_force_keeps_reference_and_prevents_dead_channel_switching():
    torch.manual_seed(5)
    block = small_block()
    kill_output_channel(block, 0)
    with torch.no_grad():
        block(3 * torch.randn(4, 2, 4, 4), num_iterations=12, reset=True)
    saved = dict(input=block.boundary_layer.state.detach().clone(),
                 states=[layer.state.detach().clone() for layer in block.free_layers()])
    output = saved['states'][-1]
    assert inactive_output_channels(output).tolist() == [True, False]
    force = 1e-3 * torch.randn_like(output)
    force[:, [0, 2]] *= 1e5  # BatchNorm-like amplification on the dead channel
    masked, record = mask_output_force(force, output, 'inactive_output_channels')
    assert record['inactive_channels'] == 1 and record['masked_force_square_share'] > .99
    reference = local_bptt_gradients(block, dict(saved, gradient=force), 12, 'cpu')
    masked_reference = local_bptt_gradients(block, dict(saved, gradient=masked), 12, 'cpu')
    assert all(torch.equal(a, b) for a, b in zip(reference, masked_reference))
    beta = 3.
    unmasked_estimate, unmasked_ends = phase_endpoints(block, saved, force, beta, 12)
    masked_estimate, masked_ends = phase_endpoints(block, saved, masked, beta, 12)
    name = block.free_layers()[-1].name
    assert any(ends[name][:, [0, 2]].any() for ends in unmasked_ends.values())
    assert not any(ends[name][:, [0, 2]].any() for ends in masked_ends.values())
    unmasked_cosine = min(agreement(e, r)['cosine'] for e, r in zip(unmasked_estimate, reference))
    masked_cosine = min(agreement(e, r)['cosine'] for e, r in zip(masked_estimate, reference))
    assert masked_cosine > unmasked_cosine


def dead_block3_model():
    deterministic_setup(29)
    c = hybrid_config()
    model = CifarL8Analog(c)
    kill_output_channel(model.analog_blocks[2], 0)
    torch.manual_seed(3)
    return c, model, 3 * torch.randn(3, 3, 8, 8), torch.tensor([0, 1, 2])


def test_capture_keeps_default_records_and_optional_unmasked_force():
    c, model, x, y = dead_block3_model()
    plain = capture_batches(model, [(x, y)], c['iterations'], 'cpu')[0]
    masked = capture_batches(model, [(x, y)], c['iterations'], 'cpu',
                             force_mask='inactive_output_channels', keep_unmasked=True)[0]
    assert all(set(record) == {'input', 'states', 'gradient'} for record in plain)
    for before, after in zip(plain, masked):
        assert set(after) == {'input', 'states', 'gradient', 'force_mask', 'unmasked_gradient'}
        assert torch.equal(after['unmasked_gradient'], before['gradient'])
        expected, _ = mask_output_force(before['gradient'], before['states'][-1], 'inactive_output_channels')
        assert torch.equal(after['gradient'], expected)
        assert reference_record(after)['gradient'] is after['unmasked_gradient']
    assert masked[2]['force_mask']['inactive_channels'] == 1
    assert [r['force_mask']['inactive_channels'] for r in masked[:2]] == [0, 0]


def test_hybrid_backward_masks_injected_force_only(monkeypatch):
    c, model, x, y = dead_block3_model()
    copy_model = CifarL8Analog(c)
    copy_model.restore(model.snapshot())
    seen = []
    original = hybrid.eqprop_gradients

    def recording(block, saved, *args, **kwargs):
        seen.append(saved['gradient'].clone())
        return original(block, saved, *args, **kwargs)

    monkeypatch.setattr(hybrid, 'eqprop_gradients', recording)
    betas = c['conv_eqprop']['initial_betas']
    prediction, loss = hybrid.hybrid_backward(model, x, y, c['iterations'], betas)
    default = {n: p.grad.clone() for n, p in model.trainable_tensors().items()}
    records = []
    masked_prediction, masked_loss = hybrid.hybrid_backward(copy_model, x, y, c['iterations'], betas,
        force_mask='inactive_output_channels', mask_records=records)
    assert torch.equal(prediction, masked_prediction) and torch.equal(loss, masked_loss)
    assert [r['inactive_channels'] for r in records] == [0, 0, 1]
    plain_forces, masked_forces = seen[:3], seen[3:]
    for i in (0, 1):
        assert torch.equal(plain_forces[i], masked_forces[i])
    half = plain_forces[2].shape[1] // 2
    keep = [k for k in range(2 * half) if k not in (0, half)]
    assert torch.equal(masked_forces[2][:, keep], plain_forces[2][:, keep])
    assert not masked_forces[2][:, [0, half]].any() and plain_forces[2][:, [0, half]].any()
    for name, grad in copy_model.trainable_tensors().items():
        if not name.startswith('blocks.2.'):
            assert torch.equal(grad.grad, default[name]), name


def test_spec_accepts_only_declared_force_masks():
    c = hybrid_config()
    hybrid.validate_spec(c)
    accepted = copy.deepcopy(c)
    accepted['conv_eqprop']['force_mask'] = 'inactive_output_channels'
    hybrid.validate_spec(accepted)
    accepted['conv_eqprop']['force_mask'] = 'all_off_units'
    with pytest.raises(ValueError, match='Expected force_mask'):
        hybrid.validate_spec(accepted)


def test_training_target_rule_takes_larger_middle_of_tied_best_scores():
    from experiments.run_cifar_masked_training_pair import tied_median
    assert tied_median([.2, .9995, .9999, .9992, .5]) == (2, [1, 2, 3])
    assert tied_median([.2, .9995, .9999, .5]) == (2, [1, 2])
    assert tied_median([.97, .96, .95], tie=1e-3) == (0, [0])


def test_passing_run_starts_at_first_pass_and_stops_at_first_gap():
    from experiments.run_cifar_force_mask_replay import passing_run
    grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    assert passing_run(grid, [False, True, True, False, True]) == (1e-5, 1e-4)
    assert passing_run(grid, [True] * 5) == (1e-6, 1e-2)
    assert passing_run(grid, [False] * 5) == (None, None)
