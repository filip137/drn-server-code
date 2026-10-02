"""CPU checks for endpoint-only instrumentation and causal minibatch feedback."""
import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from experiments import cifar_conv_eqprop as hybrid
from experiments import train_cifar_beta_stages as staged
from experiments.train_cifar_l8_analog import deterministic_setup, make_optimizer
from labs.cifar_l8_analog import CifarL8Analog
from labs.custom_minimizer import CustomQuadraticMinimizer


def config():
    value = json.loads(Path('configs/cifar/cifar10_l8_analog_bs16_seed0.json').read_text())
    value.update(scheme='ours', blocks=[[2, 2, 2], [3, 3, 3], [4, 4]], input_size=8)
    value['solver'].update(voltage_amp=4., current_amp=1.)
    return value


def case(feedback=True):
    result = dict(method='hybrid_eqprop', betas=[.003, .001, .003], measure_training_rms=True)
    if feedback:
        result['beta_feedback'] = dict(targets=[.02, .035, .001], damping=.5,
            factor_bounds=[.5, 2.], beta_bounds=[1e-8, 1e4], rms_floor=1e-12)
    return result


def search_case():
    result = case(False)
    result['beta_search'] = dict(targets=[.02, .035, .001], relative_tolerance=.25,
        max_evaluations=5, beta_bounds=[1e-8, 1e4], factor_bounds=[.001, 1000.], rms_floor=1e-12)
    return result


def batch():
    return torch.randn(3, 3, 8, 8) + random.random() + np.random.random(), torch.tensor([0, 1, 2])


def exact(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            exact(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            exact(a, b)
    else:
        assert first == second


def test_instrumentation_preserves_gradients_bn_rng_and_solver_count(monkeypatch):
    deterministic_setup(13)
    c, arm = config(), case()
    a, b = CifarL8Analog(c), CifarL8Analog(c)
    b.restore(a.snapshot())
    x, y = batch()
    calls = []
    original = CustomQuadraticMinimizer.compute_equilibrium
    def counted(self, *args, **kwargs):
        calls.append(self.num_iterations)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(CustomQuadraticMinimizer, 'compute_equilibrium', counted)
    before_rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    prediction, loss = hybrid.hybrid_backward(a, x, y, c['iterations'], arm['betas'])
    baseline_calls = list(calls)
    calls.clear()
    measured_prediction, measured_loss, rms = hybrid.hybrid_backward(
        b, x, y, c['iterations'], arm['betas'], measure_displacement=True)
    assert calls == baseline_calls and len(calls) == 9
    exact(prediction, measured_prediction)
    exact(loss, measured_loss)
    exact(a.snapshot(), b.snapshot())
    exact({n:p.grad for n,p in a.trainable_tensors().items()},
          {n:p.grad for n,p in b.trainable_tensors().items()})
    exact(before_rng, (torch.get_rng_state(), np.random.get_state(), random.getstate()))
    assert len(rms) == 3 and all(m['free_rms'] > 0 and m['relative_rms'] >= 0 for m in rms)
    assert all(int(bridge[1].num_batches_tracked) == 1 for bridge in b.bridges)


def test_feedback_clips_zero_rms_factors_and_beta_bounds_without_mutation():
    arm = case()
    arm['beta_feedback']['beta_bounds'] = [.00075, .005]
    state = hybrid.initial_training_beta_state(arm)
    original = copy.deepcopy(state)
    rms = [{'relative_rms': v} for v in (0., .035 * 16, .001)]
    proposal, row = hybrid.next_training_beta_state(state, arm, rms)
    assert state == original
    assert proposal['values'] == [.005, .00075, .003]
    assert row['factor_clipped'] == row['beta_clipped'] == [True, True, False]
    assert proposal['clip_counts'] == {'factor': [1, 1, 0], 'beta': [1, 1, 0]}
    assert proposal['successful_updates'] == 1 and row['betas_used'] == arm['betas']
    fixed = case(False)
    fixed_state, fixed_row = hybrid.next_training_beta_state(hybrid.initial_training_beta_state(fixed), fixed, rms)
    assert fixed_state['values'] == fixed['betas']
    assert fixed_row['factor_clipped'] == fixed_row['beta_clipped'] == [False] * 3


@pytest.mark.parametrize('policy', ['feedback', 'search'])
def test_commit_follows_optimizer_projection_and_finite_check(monkeypatch, policy):
    deterministic_setup(21)
    c, arm = config(), search_case() if policy == 'search' else case()
    model = CifarL8Analog(c)
    optimizer, _ = make_optimizer(model, c)
    model.conv_beta_state = hybrid.initial_training_beta_state(arm)
    initial = copy.deepcopy(model.conv_beta_state)
    original_step, original_project = optimizer.step, model.project_
    events = []
    def step():
        assert model.conv_beta_state == initial
        events.append('optimizer')
        return original_step()
    def project():
        assert model.conv_beta_state == initial
        events.append('project')
        return original_project()
    monkeypatch.setattr(optimizer, 'step', step)
    monkeypatch.setattr(model, 'project_', project)
    _, _, row = staged.training_update(model, optimizer, *batch(), c['iterations'], arm)
    assert events == ['optimizer', 'project']
    assert row.get('starting_betas', row['betas_used']) == initial['values']
    assert model.conv_beta_state['values'] == row['betas_next']
    assert model.conv_beta_state['successful_updates'] == 1
    # A failed post-step parameter check must not publish the next beta.
    committed = copy.deepcopy(model.conv_beta_state)
    monkeypatch.setattr(optimizer, 'step', original_step)
    monkeypatch.setattr(model, 'project_', original_project)
    check = staged.finite_parameters
    def reject_after_step(value, *, gradients=False):
        if not gradients:
            raise FloatingPointError('Nonfinite parameter: injected test failure')
        check(value, gradients=True)
    monkeypatch.setattr(staged, 'finite_parameters', reject_after_step)
    optimizer.zero_grad(set_to_none=True)
    with pytest.raises(FloatingPointError):
        staged.training_update(model, optimizer, *batch(), c['iterations'], arm)
    assert model.conv_beta_state == committed


@pytest.mark.parametrize('feedback', [False, True, 'search'])
def test_instrumented_hybrid_checkpoint_next_update_is_exact(tmp_path, feedback):
    deterministic_setup(31)
    c, arm, identity = config(), search_case() if feedback == 'search' else case(feedback), {'sha256': {'initializer.pt': 'test'}}
    a = CifarL8Analog(c)
    oa, sa = make_optimizer(a, c)
    a.conv_beta_state = hybrid.initial_training_beta_state(arm)
    staged.training_update(a, oa, *batch(), c['iterations'], arm)
    middle = staged.cursor(0, 1, train_examples=6, batch_size=3)
    path = tmp_path / 'checkpoint.pt'
    staged.save_state(path, a, oa, sa, c, 'arm', arm, middle, identity)
    saved = torch.load(path, weights_only=False)
    oa.zero_grad(set_to_none=True)
    _, _, expected_row = staged.training_update(a, oa, *batch(), c['iterations'], arm)
    sa.step()
    rng_after = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    b = CifarL8Analog(c)
    ob, sb = make_optimizer(b, c)
    staged.restore_state(saved, b, ob, sb, c, 'arm', arm, identity, train_examples=6, batch_size=3)
    ob.zero_grad(set_to_none=True)
    _, _, actual_row = staged.training_update(b, ob, *batch(), c['iterations'], arm)
    sb.step()
    exact(expected_row, actual_row)
    exact(a.snapshot(), b.snapshot())
    exact(oa.state_dict(), ob.state_dict())
    exact(sa.state_dict(), sb.state_dict())
    exact(a.conv_beta_state, b.conv_beta_state)
    exact(rng_after, (torch.get_rng_state(), np.random.get_state(), random.getstate()))
    assert a.conv_beta_state['successful_updates'] == 2
    bad = torch.load(path, weights_only=False)
    bad['conv_beta_state']['successful_updates'] = 0
    with pytest.raises(ValueError, match='counter'):
        staged.restore_state(bad, b, ob, sb, c, 'arm', arm, identity, train_examples=6, batch_size=3)


def test_restore_checks_feedback_spec_bounds_and_clip_counters():
    arm = case()
    valid = hybrid.initial_training_beta_state(arm)
    for mutate in (lambda s: s['feedback_spec'].update(damping=.25),
                   lambda s: s['values'].__setitem__(0, 1e5),
                   lambda s: s['clip_counts']['factor'].__setitem__(0, 1)):
        bad = copy.deepcopy(valid)
        mutate(bad)
        with pytest.raises(ValueError):
            hybrid.validate_training_beta_state(bad, arm, 0)
    old_case = {'method': 'hybrid_eqprop', 'betas': arm['betas']}
    hybrid.validate_training_beta_state({'policy': 'fixed', 'values': arm['betas'],
        'calibrated_for_epoch': 1}, old_case, 1407)


@pytest.mark.parametrize('policy', ['feedback', 'search'])
def test_diagnostic_observes_current_feedback_beta_without_retuning(tmp_path, monkeypatch, policy):
    c, arm = config(), search_case() if policy == 'search' else case()
    model = CifarL8Analog(c)
    controller = hybrid.EpochBeta(model, staged.diagnostic_config(c, arm, list(range(256))),
                                 [batch()], torch.device('cpu'), tmp_path, float('inf'))
    model.conv_beta_state = hybrid.initial_training_beta_state(arm)
    model.conv_beta_state['values'] = [2 * v for v in arm['betas']]
    state = copy.deepcopy(model.conv_beta_state)
    monkeypatch.setattr(hybrid.reporting, 'update_status_progress', lambda *a, **k: None)
    monkeypatch.setattr(hybrid, 'search_beta', lambda *a, **k: pytest.fail('Unexpected calibration'))
    observation = controller.observe(0)
    assert [row['beta'] for row in observation['blocks']] == state['values']
    assert model.conv_beta_state == state


@pytest.mark.parametrize('field,value', [('unexpected', 1), ('damping', float('nan')),
    ('rms_floor', 0), ('targets', [1., 2.]), ('beta_bounds', [1., .1]), ('factor_bounds', [2., 3.])])
def test_malformed_feedback_rejected(field, value):
    arm = case()
    arm['beta_feedback'][field] = value
    with pytest.raises(ValueError):
        hybrid.validate_training_beta_case(arm)


@pytest.mark.parametrize('free,centered,error', [(0., 0., ValueError), (1., float('nan'), FloatingPointError)])
def test_training_displacement_requires_finite_positive_free_rms(monkeypatch, free, centered, error):
    c, arm = config(), case()
    model = CifarL8Analog(c)
    original = hybrid.eqprop_gradients
    def corrupt(*args, **kwargs):
        gradients, measured = original(*args, **kwargs)
        measured.update(free_squares=free, centered_squares=centered)
        return gradients, measured
    monkeypatch.setattr(hybrid, 'eqprop_gradients', corrupt)
    with pytest.raises(error):
        hybrid.hybrid_backward(model, *batch(), c['iterations'], arm['betas'], measure_displacement=True)


def test_current_batch_search_restores_every_trial_and_uses_cached_selected_gradient(monkeypatch):
    from experiments import compare_cifar_block_beta_gradients as comparison
    deterministic_setup(13)
    c, arm = config(), search_case()
    a, b = CifarL8Analog(c), CifarL8Analog(c)
    b.restore(a.snapshot())
    x, y = batch()
    prediction, loss = hybrid.hybrid_backward(a, x, y, c['iterations'], arm['betas'])
    reference = {name:p.grad.clone() for name,p in a.trainable_tensors().items()}
    block_indices = {id(block): i for i, block in enumerate(b.analog_blocks)}
    trial_counts, restored_records, actual_gradients = [0] * 3, {}, [[], [], []]
    original_ep, original_restore = hybrid.eqprop_gradients, comparison.restore_block
    def checked_restore(block, saved, device):
        index = block_indices[id(block)]
        if index not in restored_records:
            restored_records[index] = copy.deepcopy(saved)
        exact(saved, restored_records[index])
        original_restore(block, saved, device)
        exact(block.boundary_layer.state, saved['input'])
        exact([layer.state for layer in block.free_layers()], saved['states'])
    def force_five_trials(block, saved, beta, *args, **kwargs):
        index = block_indices[id(block)]
        gradients, displacement = original_ep(block, saved, beta, *args, **kwargs)
        actual_gradients[index].append([g.clone() for g in gradients])
        # Real gradients/endpoints are computed each time; controlled RMS forces
        # a bounded search whose first pair is best and whose last pair differs.
        ratio = [.5, 10., .1, 5., 4.][trial_counts[index]]
        trial_counts[index] += 1
        displacement['centered_squares'] = (arm['beta_search']['targets'][index] * ratio)**2 * displacement['free_squares']
        return gradients, displacement
    monkeypatch.setattr(comparison, 'restore_block', checked_restore)
    monkeypatch.setattr(hybrid, 'eqprop_gradients', force_five_trials)
    calls = []
    original_solver = CustomQuadraticMinimizer.compute_equilibrium
    def count_solver(self, *args, **kwargs):
        calls.append(self.num_iterations)
        return original_solver(self, *args, **kwargs)
    monkeypatch.setattr(CustomQuadraticMinimizer, 'compute_equilibrium', count_solver)
    before_rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    actual_prediction, actual_loss, measured = hybrid.hybrid_backward(b, x, y, c['iterations'], arm['betas'],
        measure_displacement=True, beta_search=arm['beta_search'])
    assert trial_counts == [5, 5, 5] and len(calls) == 3 + 2 * 15
    exact(prediction, actual_prediction)
    exact(loss, actual_loss)
    exact(reference, {name:p.grad for name,p in b.trainable_tensors().items()})
    exact(a.snapshot(), b.snapshot())
    exact(before_rng, (torch.get_rng_state(), np.random.get_state(), random.getstate()))
    assert all(int(bridge[1].num_batches_tracked) == 1 for bridge in b.bridges)
    assert any(not torch.equal(first, last) for trials in actual_gradients for first, last in zip(trials[0], trials[-1]))
    for i, measurement in enumerate(measured):
        record = measurement['search']
        assert record['chosen_index'] == 0 and record['selected_beta'] == arm['betas'][i]
        assert record['pair_count'] == 5 and record['tolerance_miss']
        assert record['trials'][2]['beta'] == pytest.approx((record['trials'][0]['beta'] * record['trials'][1]['beta'])**.5)
        hybrid.validate_beta_search_record(record, arm['betas'][i], arm['beta_search']['targets'][i], arm['beta_search'])


def test_search_tie_prefers_smaller_beta_repeated_bound_stops_and_nonfinite_trial_fails():
    spec = search_case()['beta_search']
    spec['max_evaluations'] = 3
    measurements = iter([2., .5, .5])
    def measure(beta):
        relative = next(measurements)
        return [torch.tensor(beta)], dict(relative_rms=relative, absolute_rms=relative, free_rms=1., elements=1)
    gradients, measured = hybrid.select_current_beta(measure, 2., 1., spec)
    record = measured['search']
    assert record['chosen_index'] == 1 and record['selected_beta'] == 1.
    assert gradients[0].item() == 1. and record['pair_count'] == 3
    hybrid.validate_beta_search_record(record, 2., 1., spec)
    def zero(beta):
        return [torch.tensor(beta)], dict(relative_rms=0., absolute_rms=0., free_rms=1., elements=1)
    _, measured = hybrid.select_current_beta(zero, 1e4, 1., spec)
    assert measured['search']['pair_count'] == 1 and measured['search']['stop_reason'] == 'repeated_beta'
    hybrid.validate_beta_search_record(measured['search'], 1e4, 1., spec)
    calls = []
    def nonfinite_second(beta):
        calls.append(beta)
        return [torch.tensor(float('nan') if len(calls) == 2 else 1.)], dict(relative_rms=.5, free_rms=1.)
    with pytest.raises(FloatingPointError, match='Nonfinite parameter gradient'):
        hybrid.select_current_beta(nonfinite_second, 1., 1., spec)


@pytest.mark.parametrize('field,value', [('unexpected', 1), ('relative_tolerance', 1.),
    ('max_evaluations', 6), ('max_evaluations', True), ('beta_bounds', [1., .1])])
def test_malformed_current_batch_spec_rejected(field, value):
    arm = search_case()
    arm['beta_search'][field] = value
    with pytest.raises(ValueError):
        hybrid.validate_training_beta_case(arm)
