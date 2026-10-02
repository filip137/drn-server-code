"""Probes must expose Adam's real gradient without changing its trajectory."""
import copy
import json
import math
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments import cifar_conv_eqprop as hybrid
from experiments import replay_cifar_selected_gradients as replay
from experiments.train_cifar_beta_stages import epoch_order
from experiments.train_cifar_l8_analog import deterministic_setup, make_optimizer
from labs.cifar_l8_analog import CifarL8Analog, batchnorm_minibatch_stats_only
from labs.tests.test_cifar_beta_feedback import batch, config, exact, search_case


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(3).double()
        self.head = torch.nn.Linear(3, 2).double()
        self.conv_beta_state = dict(values=[.1], successful_updates=0)

    def forward(self, x, iterations, reset=True):
        # Each stream affects the real next update and both diagnostic references.
        x = x + torch.randn_like(x) * (random.random() + np.random.random())
        return self.head(self.bn(x))

    def bptt(self, x, iterations):
        with torch.no_grad():
            free = self(x, iterations)
        with batchnorm_minibatch_stats_only(self):
            tracked = self(x, iterations)
        return free, tracked

    def trainable_tensors(self):
        return dict(self.named_parameters())

    def detach_state_(self):
        pass


def test_all101_updates_are_identical_with_probes_including_adam_bn_controller_order_rng(monkeypatch):
    def update(model, optimizer, x, y, iterations, case):
        logits = model(x, iterations)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        model.conv_beta_state['values'][0] *= 1.001
        model.conv_beta_state['successful_updates'] += 1
        return logits.detach(), loss.detach(), copy.deepcopy(model.conv_beta_state)

    monkeypatch.setattr(replay, 'training_update', update)

    def run(probed):
        deterministic_setup(31)
        model = Tiny()
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        order = epoch_order(list(range(404)), 0, 1)
        history = []
        for completed in range(101):
            ids = torch.tensor(order[completed*4:(completed+1)*4])
            x = torch.stack((ids, ids % 7, ids % 13), 1).double()
            y = ids % 2
            optimizer.zero_grad(set_to_none=True)
            if probed and completed in replay.PROBES:
                for method in ('native_bptt', 'free_t_bptt'):
                    replay.reference_gradients(model, x, y, [1], method)
                _, loss, row, actual, before, deltas = replay.measured_update(model, optimizer, x, y, [1], {})
                exact(actual, {n:p.grad.double() for n,p in model.trainable_tensors().items()})
                exact(deltas, {n:p.detach().double()-before[n] for n,p in model.trainable_tensors().items()})
            else:
                _, loss, row = update(model, optimizer, x, y, [1], {})
            history.append((ids.tolist(), float(loss), row))
        return dict(model=copy.deepcopy(model.state_dict()), optimizer=optimizer.state_dict(),
                    beta=model.conv_beta_state, history=history,
                    rng=(torch.get_rng_state(), np.random.get_state(), random.getstate()))

    unprobed, probed = run(False), run(True)
    exact(unprobed, probed)
    assert probed['model']['bn.num_batches_tracked'] == 101


def test_real_hybrid_capture_uses_selected_pair_when_first_trial_beats_last(monkeypatch):
    deterministic_setup(13)
    c, arm = config(), search_case()
    model = CifarL8Analog(c)
    optimizer, _ = make_optimizer(model, c)
    model.conv_beta_state = hybrid.initial_training_beta_state(arm)
    x, y = batch()
    before_model = model.snapshot()
    before_rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    free, free_loss = replay.reference_gradients(model, x, y, c['iterations'], 'free_t_bptt')
    native, _ = replay.reference_gradients(model, x, y, c['iterations'], 'native_bptt')
    exact(before_model, model.snapshot())
    exact(before_rng, (torch.get_rng_state(), np.random.get_state(), random.getstate()))
    assert all(p.grad is None for p in model.trainable_tensors().values())
    indices = {id(block): i for i, block in enumerate(model.analog_blocks)}
    gradients, counts = [[], [], []], [0, 0, 0]
    original = hybrid.eqprop_gradients

    def controlled(block, saved, beta, *args, **kwargs):
        i = indices[id(block)]
        actual, displacement = original(block, saved, beta, *args, **kwargs)
        gradients[i].append([value.cpu().double().clone() for value in actual])
        ratio = [.5, 10., .1, 5., 4.][counts[i]]
        counts[i] += 1
        displacement['centered_squares'] = (arm['beta_search']['targets'][i]*ratio)**2*displacement['free_squares']
        return actual, displacement

    monkeypatch.setattr(hybrid, 'eqprop_gradients', controlled)
    _, loss, row, actual, before, deltas = replay.measured_update(model, optimizer, x, y, c['iterations'], arm)
    assert counts == [5, 5, 5]
    assert all(record['chosen_index'] == 0 for record in row['beta_search'])
    assert any(not torch.equal(a, b) for block in gradients for a, b in zip(block[0], block[-1]))
    for name, value in actual.items():
        if name.startswith('blocks.'):
            _, block, _, weight = name.split('.')
            exact(value, gradients[int(block)][0][int(weight)])
        else:
            torch.testing.assert_close(value, free[name], rtol=.002, atol=1e-6)
    assert float(loss) == pytest.approx(free_loss, rel=1e-7)
    assert all(int(bridge[1].num_batches_tracked) == 1 for bridge in model.bridges)
    assert model.conv_beta_state['successful_updates'] == 1
    summary = replay.gradient_summary(actual, {'native_bptt': native, 'free_t_bptt': free}, before, deltas)
    assert {'block1', 'block2', 'block3', 'conv', 'digital', 'bn', 'gain', 'head'} <= summary['pooled'].keys()


def test_real_hybrid_references_leave_successive_selected_updates_exact():
    def run(probed):
        deterministic_setup(21)
        c, arm = config(), search_case()
        model = CifarL8Analog(c)
        optimizer, _ = make_optimizer(model, c)
        model.conv_beta_state = hybrid.initial_training_beta_state(arm)
        rows = []
        for _ in range(3):
            x, y = batch()
            optimizer.zero_grad(set_to_none=True)
            if probed:
                for method in ('native_bptt', 'free_t_bptt'):
                    replay.reference_gradients(model, x, y, c['iterations'], method)
                _, loss, row, _, _, _ = replay.measured_update(model, optimizer, x, y, c['iterations'], arm)
            else:
                _, loss, row = replay.training_update(model, optimizer, x, y, c['iterations'], arm)
            rows.append((float(loss), row))
        return dict(model=model.snapshot(), optimizer=optimizer.state_dict(), beta=model.conv_beta_state,
                    rows=rows, rng=(torch.get_rng_state(), np.random.get_state(), random.getstate()))

    exact(run(False), run(True))


@pytest.mark.parametrize('method', ['native_bptt', 'free_t_bptt'])
def test_reference_is_mean_ce_and_preserves_existing_gradients_modes_and_buffers(method):
    deterministic_setup(17)
    c = config()
    model = CifarL8Analog(c).double()
    model.head.weight.state = model.head.weight.state.detach().double().requires_grad_()
    x, y = batch()
    x = x.double()
    model.eval()
    for p in model.trainable_tensors().values():
        p.grad = torch.ones_like(p)
    saved = model.snapshot()
    first, first_loss = replay.reference_gradients(model, x, y, c['iterations'], method)
    second, second_loss = replay.reference_gradients(model, x.repeat(2, 1, 1, 1), y.repeat(2), c['iterations'], method)
    assert not model.training
    exact(saved, model.snapshot())
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.trainable_tensors().values())
    assert first_loss == pytest.approx(second_loss, rel=1e-6)
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=1e-6, atol=1e-10)


def test_summary_uses_pooled_vectors_and_leaves_zero_norm_cosines_undefined():
    actual = {'blocks.0.weights.0': torch.tensor([1., 0.]), 'head.weight': torch.zeros(2)}
    ref = {'blocks.0.weights.0': torch.tensor([0., 2.]), 'head.weight': torch.ones(2)}
    zero = {name: torch.zeros_like(t) for name, t in actual.items()}
    summary = replay.gradient_summary(actual, {'native_bptt': ref, 'free_t_bptt': zero}, zero, actual)
    conv = summary['pooled']['conv']
    assert conv['references']['native_bptt']['cosine'] == 0
    assert conv['references']['native_bptt']['norm_ratio'] == .5
    assert conv['references']['free_t_bptt']['cosine'] is None
    assert conv['references']['free_t_bptt']['norm_ratio'] is None
    assert summary['pooled']['digital']['references']['native_bptt']['cosine'] is None
    assert conv['update_norm'] == 1 and conv['relative_update_norm'] is None


def test_prefix_requires_every_loss_and_exact_equality(tmp_path):
    path = tmp_path/'metrics.jsonl'
    rows = [dict(stage='training', cursor=dict(total_updates=i), loss=1/i) for i in range(1, 102)]
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    prefix = replay.original_prefix(path)
    assert len(prefix) == 101
    replay.check_loss(prefix[20], .05, 'exp015', 20)
    with pytest.raises(RuntimeError, match='mismatch'):
        replay.check_loss(math.nextafter(.05, 1.), .05, 'exp015', 20)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows[1:]))
    with pytest.raises(ValueError, match='cover'):
        replay.original_prefix(path)


@pytest.mark.parametrize('damage', [None, 'missing_probe', 'bn_counter'])
def test_collector_requires_complete_probes_and_exact_bn_coverage(tmp_path, damage):
    from experiments import reporting

    arm = search_case()
    spec = dict(case=arm, case_id='current_batch', config={}, input_identity={})
    reporting.start_run(tmp_path, dict(study_id='test', run_id='replay', arm_id='both',
        replays={label: spec for label in ('exp015', 'exp017')}))
    rows, expected, bn = [], {}, {}
    for label in ('exp015', 'exp017'):
        state = hybrid.initial_training_beta_state(arm)
        expected[label] = {str(i): 1. for i in range(1, 102)}
        bn[label] = {'bn.num_batches_tracked': 0}
        for completed in range(101):
            measurements = []
            for i, beta in enumerate(state['values']):
                target = arm['beta_search']['targets'][i]
                _, measured = hybrid.select_current_beta(lambda value: ([torch.tensor(1.)],
                    dict(relative_rms=target, absolute_rms=target, free_rms=1., elements=1)),
                    beta, target, arm['beta_search'])
                measurements.append(measured)
            state, telemetry = hybrid.next_training_beta_state(state, arm, measurements)
            rows.append(dict(stage='training', replay=label, update=completed+1,
                             loss=1., loss_match=True, **telemetry))
            if completed in replay.PROBES and not (damage == 'missing_probe' and label == 'exp015' and completed == 100):
                values = {f'blocks.{i}.weights.0': torch.ones(1) for i in range(3)}
                values['head.weight'] = torch.ones(1)
                summary = replay.gradient_summary(values, {'native_bptt': values, 'free_t_bptt': values}, values, values)
                rows.append(dict(stage='probe', replay=label, completed_updates=completed, **summary, **telemetry))
        torch.save(dict(staged_cursor=replay.cursor(0, 101), scheduler={'last_epoch': 0}, epoch=0,
            native_epoch_resume_safe=False, config={}, case=arm, case_id='current_batch', input_identity={},
            optimizer={'state': {0: {'step': 101}}}, conv_beta_state=state,
            model={'module': {'bn.num_batches_tracked': torch.tensor(102 if damage == 'bn_counter' else 101)}}),
            tmp_path/'checkpoints'/f'{label}_prefix.pt')
    (tmp_path/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    reporting.atomic_write_json(tmp_path/'artifacts/original_prefix.json', expected)
    reporting.atomic_write_json(tmp_path/'artifacts/initial_bn_counts.json', bn)
    reporting.complete_run(tmp_path, terminal_metrics={})
    if damage:
        with pytest.raises(ValueError, match='coverage|BatchNorm'):
            replay.validate_replay(tmp_path)
    else:
        assert replay.validate_replay(tmp_path)['probes'] == 10
