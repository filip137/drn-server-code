"""CPU toy checks for exact staged order, RNG, Adam, BatchNorm and schedule."""
import copy
import json
import random
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset
from experiments import train_cifar_beta_stages as staged

from experiments.train_cifar_beta_stages import (
    advance_cursor, cursor, diagnostic_config, epoch_loader, epoch_order, restore_state,
    save_state, stage_bounds, stage_target, validate_config, verified_nonfinite,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(2).double()
        self.linear = torch.nn.Linear(2, 2).double()

    def snapshot(self):
        return {'module': {n:t.detach().clone() for n,t in self.state_dict().items()}}

    def restore(self, state):
        self.load_state_dict(state['module'])

    def trainable_tensors(self):
        return dict(self.named_parameters())


def setup():
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1 - e / 50)
    return model, optimizer, scheduler


def seed():
    torch.manual_seed(17)
    np.random.seed(19)
    random.seed(23)


def run(model, optimizer, scheduler, progress, target):
    seen = []
    while progress['total_updates'] < target:
        epoch = progress['next_epoch']
        order = epoch_order(list(range(11)), 0, epoch)
        ids = torch.tensor(order[progress['examples_in_epoch']:])
        for (indices,) in epoch_loader(TensorDataset(ids), 3, 0, False, 0, epoch):
            optimizer.zero_grad(set_to_none=True)
            model.train()
            x = torch.stack((indices.double(), indices.double().square()), 1)
            # All global RNG streams affect training; the loader must not advance them on resume.
            x = x + torch.randn_like(x) * (random.random() + np.random.random())
            loss = model.linear(model.bn(x)).square().mean()
            loss.backward()
            optimizer.step()
            seen.extend(indices.tolist())
            progress, full = advance_cursor(progress, train_examples=11, batch_size=3)
            if full:
                scheduler.step()
            if progress['total_updates'] == target:
                break
    return progress, seen


def assert_same(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_same(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_same(left, right)
    else:
        assert first == second


@pytest.mark.parametrize('split_after', [2, 4])
def test_resume_matches_uninterrupted_training_exactly(tmp_path, split_after):
    config, case, identity = {'seed': 0}, {'method': 'native_bptt'}, {'sha256': {'initializer.pt': 'same'}}
    seed()
    a, oa, sa = setup()
    final_a, seen_a = run(a, oa, sa, cursor(train_examples=11, batch_size=3), 8)
    rng_a = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    seed()
    b, ob, sb = setup()
    middle, seen_b = run(b, ob, sb, cursor(train_examples=11, batch_size=3), split_after)
    path = tmp_path / 'stage.pt'
    save_state(path, b, ob, sb, config, 'native', case, middle, identity)
    saved = torch.load(path, weights_only=False)
    assert saved['native_epoch_resume_safe'] == (split_after == 4)
    c, oc, sc = setup()  # Construction consumes RNG, then full restore must undo it.
    restored = restore_state(saved, c, oc, sc, config, 'native', case, identity,
                             train_examples=11, batch_size=3)
    final_c, seen_c = run(c, oc, sc, restored, 8)
    assert seen_a == seen_b + seen_c
    assert final_a == final_c
    assert_same(a.snapshot(), c.snapshot())
    assert_same(oa.state_dict(), oc.state_dict())
    assert_same(sa.state_dict(), sc.state_dict())
    assert_same(rng_a, (torch.get_rng_state(), np.random.get_state(), random.getstate()))
    assert int(c.bn.num_batches_tracked) == 8
    assert sc.last_epoch == 2


def test_partial_cursor_preserves_full_original_minibatches():
    partial = cursor(0, 704)
    assert partial == dict(completed_epoch=0, next_epoch=1, next_batch=704,
                        examples_in_epoch=22528, total_updates=704)
    order = epoch_order(list(range(45000)), 0, 1)
    assert order[:22528] + order[partial['examples_in_epoch']:] == order
    progress = partial
    for _ in range(1407 - 704):
        progress, full = advance_cursor(progress)
    assert full and progress == cursor(1)
    assert [stage_target(stage) for stage in ('1', '5', '10')] == [1407, 7035, 14070]


def test_restore_rejects_changed_case_and_broken_cursor(tmp_path):
    seed()
    m, o, s = setup()
    config, case, identity = {'seed': 0}, {'method': 'native_bptt'}, {'sha256': {}}
    progress, _ = run(m, o, s, cursor(train_examples=11, batch_size=3), 2)
    path = tmp_path / 'stage.pt'
    save_state(path, m, o, s, config, 'native', case, progress, identity)
    state = torch.load(path, weights_only=False)
    with pytest.raises(ValueError, match='case_id'):
        restore_state(state, m, o, s, config, 'different', case, identity, train_examples=11, batch_size=3)
    bad = copy.deepcopy(state)
    bad['staged_cursor']['examples_in_epoch'] += 1
    with pytest.raises(ValueError, match='cursor'):
        restore_state(bad, m, o, s, config, 'native', case, identity, train_examples=11, batch_size=3)


def test_only_verified_nonfinite_is_scientific_rejection():
    model = TinyModel()
    assert verified_nonfinite(FloatingPointError('Nonfinite hybrid loss'), model)
    assert not verified_nonfinite(FloatingPointError('Missing/nonfinite hybrid parameter gradient'), model)
    assert not verified_nonfinite(FloatingPointError('Output force is nonfinite or dead'), model)
    with torch.no_grad():
        model.linear.weight[0, 0] = float('nan')
    assert verified_nonfinite(FloatingPointError('Nonfinite parameter: linear.weight'), model)
    assert not verified_nonfinite(RuntimeError('shape mismatch'), model)


@pytest.mark.parametrize('method', ['native_bptt', 'free_t_bptt', 'hybrid_eqprop'])
def test_main_admits_supported_case_before_training(tmp_path, monkeypatch, method):
    source = tmp_path/'source'
    source.mkdir()
    inputs = tmp_path/'inputs'
    inputs.mkdir()
    for path, value in (
        (source/'source_identity.json', {'source_sha256': {}}),
        (source/'training-data-hashes.json', {f'data_batch_{i}': '' for i in range(1, 6)}),
        (inputs/'identity.json', {'sha256': {}}), (inputs/'config.json', {}),
        (inputs/'cases.json', {'control': {'method': method,
            'betas': [.1, .1, .1] if method == 'hybrid_eqprop' else []}}),
    ):
        path.write_text(json.dumps(value))
    monkeypatch.setattr(staged, '__file__', str(source/'experiments/train_cifar_beta_stages.py'))
    monkeypatch.setattr(staged, 'verify_hashes', lambda *a, **kw: None)
    monkeypatch.setattr(sys, 'argv', ['runner', '--root', str(tmp_path), '--case', 'control',
        '--stage', '1', '--output-dir', str(tmp_path/'out'), '--dataset-root', str(tmp_path/'data'),
        '--target', 'local', '--maximum-seconds', '600'])

    class ReachedConfigValidation(Exception):
        pass

    def stop_before_training(config):
        raise ReachedConfigValidation

    monkeypatch.setattr(staged, 'validate_config', stop_before_training)
    with pytest.raises(ReachedConfigValidation):
        staged.main()


@pytest.mark.parametrize('scheme', ['baseline', 'ours', 'legacy'])
def test_scheme_acceptance_preserves_config_and_original_schedule(scheme):
    config = dict(scheme=scheme, batch_size=32, seed=0, evaluation_batch_size=16,
                  iterations=[6, 6, 4], label_smoothing=0,
                  scheduler=dict(horizon_epochs=50, final_lr_ratio=.02))
    original = copy.deepcopy(config)
    validate_config(config)
    assert config == original
    config['scheduler']['horizon_epochs'] = 10
    with pytest.raises(ValueError, match='contract'):
        validate_config(config)


def test_longer_stage_admission_accepts_previous_and_recovery_boundaries():
    for stage, previous in [('20', 10), ('30', 20), ('50', 30)]:
        minimum, target = stage_bounds(stage)
        assert minimum == cursor(previous)['total_updates']
        assert target == cursor(int(stage))['total_updates']
        assert minimum <= cursor(previous, 17)['total_updates'] <= target
        assert cursor(previous - 1, 1406)['total_updates'] < minimum
        assert cursor(int(stage), 1)['total_updates'] > target
    with pytest.raises(ValueError, match='Stages'):
        stage_bounds('100')


def test_scheme_specific_diagnostic_anchors_do_not_change_beta_or_training_config():
    config = {'scheme': 'legacy', 'scheduler': {'horizon_epochs': 50}}
    case = {'method': 'hybrid_eqprop', 'betas': [.03, .1, .2], 'rms_targets': [.01, .02, .03]}
    resolved = diagnostic_config(config, case, list(range(256)))
    spec = resolved['conv_eqprop']
    assert spec['rms_targets'] == [.01, .02, .03]
    assert spec['initial_betas'] == case['betas']
    assert spec['policy'] == 'fixed' and spec['cosine_epochs'] == [0, 1, 5, 10, 20, 30, 50]
    spec['rms_targets'][0] = 99
    spec['initial_betas'][0] = 99
    assert case['rms_targets'] == [.01, .02, .03] and case['betas'] == [.03, .1, .2]
    assert config == {'scheme': 'legacy', 'scheduler': {'horizon_epochs': 50}}
    original_ours = diagnostic_config({'scheme': 'ours'}, {'betas': [1., 2., 3.]}, list(range(256)))
    assert original_ours['conv_eqprop']['rms_targets'] == [.02, .035, .001]
