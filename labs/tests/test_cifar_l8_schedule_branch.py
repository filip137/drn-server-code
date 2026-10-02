"""CPU-only branch tests: synthetic tensors, no CIFAR reads or solver training."""
import copy
import json
import math
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from experiments import train_cifar_l8_analog as runner


class TinyModel:
    def __init__(self):
        self.parameters = [torch.nn.Parameter(torch.randn(2, dtype=torch.float64)) for _ in range(3)]

    def optimizer_groups(self, config):
        return [{'params': [p], 'lr': lr} for p, lr in
                zip(self.parameters, [0.0045, config['bn_learning_rate'], config['gain_learning_rate']])]

    def snapshot(self):
        return [p.detach().clone() for p in self.parameters]

    def restore(self, values):
        with torch.no_grad():
            for p, value in zip(self.parameters, values):
                p.copy_(value)


def config():
    return json.loads(Path('configs/cifar/horizon_e50_e70_20260924/legacy_floor.json').read_text())


def exact(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def step(model, optimizer, scheduler, batches=1):
    for _ in range(batches):
        batch_step(model, optimizer)
    scheduler.step()


def batch_step(model, optimizer):
    optimizer.zero_grad()
    # All three global RNG streams affect the update and must survive recovery.
    offset = random.random() + np.random.random()
    sum((p * torch.randn_like(p) + offset).square().sum() for p in model.parameters).backward()
    optimizer.step()


def parent(tmp_path):
    runner.deterministic_setup(7)
    c = config()
    model = TinyModel()
    optimizer, scheduler = runner.make_optimizer(model, c)
    for _ in range(50):
        step(model, optimizer, scheduler)
    path = tmp_path / 'parent.pt'
    runner.save_checkpoint(path, model, optimizer, scheduler, 50, c)
    return torch.load(path, weights_only=False)


def provenance(checkpoint):
    return runner.schedule_branch_provenance(checkpoint, runner.SCHEDULE_RENEWAL,
        input_sha256='input-hash', parent_checkpoint_sha256='parent-hash',
        parent_manifest_sha256='manifest-hash')


def restore(checkpoint, c=None, branch=True):
    c = c or config()
    model = TinyModel()
    optimizer, scheduler = runner.make_optimizer(model, c)
    spec = runner.SCHEDULE_RENEWAL if branch else None
    if branch:
        c['schedule_branch'] = copy.deepcopy(spec)
    runner.restore_checkpoint(checkpoint, model, optimizer, scheduler, c,
                              spec, provenance(checkpoint) if branch else None)
    return model, optimizer, scheduler, c


def test_exact_endpoints_and_group_ratios(tmp_path):
    checkpoint = parent(tmp_path)
    model, optimizer, scheduler, c = restore(checkpoint)
    exact(optimizer.state_dict()['state'], checkpoint['optimizer']['state'])
    exact(model.snapshot(), checkpoint['model'])
    exact(torch.get_rng_state(), checkpoint['torch_rng'])
    exact(np.random.get_state(), checkpoint['numpy_rng'])
    exact(random.getstate(), checkpoint['python_rng'])
    assert scheduler.last_epoch == 50
    assert scheduler.schedule_branch['restored_learning_rates'] == [lr * 0.02 for lr in scheduler.base_lrs]
    assert runner.renewal_lr_ratio(51, runner.SCHEDULE_RENEWAL) == 0.10
    assert runner.renewal_lr_ratio(70, runner.SCHEDULE_RENEWAL) == 0.02
    assert runner.renewal_lr_ratio(71, runner.SCHEDULE_RENEWAL) == 0.02
    assert runner.renewal_lr_ratio(51.5, runner.SCHEDULE_RENEWAL) == pytest.approx(
        .02 + .08 * (1 + math.cos(math.pi * .5 / 19)) / 2)
    for epoch in range(51, 72):
        assert scheduler.get_last_lr() == [lr * runner.renewal_lr_ratio(epoch, runner.SCHEDULE_RENEWAL)
                                           for lr in scheduler.base_lrs]
        step(model, optimizer, scheduler)


def test_all_twenty_updates_and_epoch60_recovery_are_bitwise_equal(tmp_path):
    checkpoint = parent(tmp_path)
    batches = math.ceil(45000 / config()['batch_size'])
    # Preserve the production Adam counter while using tiny synthetic groups.
    for state in checkpoint['optimizer']['state'].values():
        state['step'].fill_(50 * batches)
    a, oa, sa, c = restore(checkpoint)
    trace = []
    for epoch in range(51, 71):
        trace.append(sa.get_last_lr())
        step(a, oa, sa, batches)
        if epoch == 60:
            runner.save_checkpoint(tmp_path / 'epoch60.pt', a, oa, sa, epoch, c)
    expected_rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    saved = torch.load(tmp_path / 'epoch60.pt', weights_only=False)
    b, ob, sb, _ = restore(saved)
    assert sb.last_epoch == 60
    assert sb.get_last_lr() == trace[10] and sb.get_last_lr() != trace[0]
    for epoch in range(61, 71):
        assert sb.get_last_lr() == trace[epoch - 51]
        step(b, ob, sb, batches)
    exact(a.snapshot(), b.snapshot())
    exact(oa.state_dict(), ob.state_dict())
    exact(sa.state_dict(), sb.state_dict())
    exact(expected_rng, (torch.get_rng_state(), np.random.get_state(), random.getstate()))
    assert len(trace) == 20 and sb.last_epoch == 70
    assert {int(v['step']) for v in ob.state.values()} == {98490}
    runner.save_checkpoint(tmp_path / 'final.pt', b, ob, sb, 70, c)
    final = torch.load(tmp_path / 'final.pt', weights_only=False)
    assert final['epoch'] == 70
    assert final['schedule_branch'] == saved['schedule_branch'] == provenance(checkpoint)
    assert final['config']['schedule_branch'] == runner.SCHEDULE_RENEWAL


@pytest.mark.parametrize('field', ['optimizer', 'scheduler', 'solver', 'batch_size', 'augmentation'])
def test_branch_keeps_original_science_strict(tmp_path, field):
    checkpoint = parent(tmp_path)
    c = config()
    if field == 'optimizer':
        c[field]['bn_learning_rate'] *= 2
    elif field == 'scheduler':
        c[field]['horizon_epochs'] = 70
    elif field == 'solver':
        c[field]['voltage_amp'] = 2
    elif field == 'augmentation':
        c[field]['flip_probability'] = 0
    else:
        c[field] = 64
    with pytest.raises(ValueError, match=field):
        restore(checkpoint, c)


def test_wrong_boundary_missing_flag_and_changed_spec_rejected(tmp_path):
    checkpoint = parent(tmp_path)
    wrong = copy.deepcopy(checkpoint)
    wrong['epoch'] = 49
    with pytest.raises(ValueError, match='epoch 50'):
        restore(wrong)
    model, optimizer, scheduler, c = restore(checkpoint)
    runner.save_checkpoint(tmp_path / 'branched.pt', model, optimizer, scheduler, 50, c)
    saved = torch.load(tmp_path / 'branched.pt', weights_only=False)
    with pytest.raises(ValueError, match='explicit --schedule-branch'):
        restore(saved, branch=False)
    saved['schedule_branch']['specification']['start_lr_ratio'] = 0.2
    with pytest.raises(ValueError, match='specification mismatch'):
        restore(saved)
    with pytest.raises(ValueError, match='target epoch 70'):
        runner.validate_schedule_branch(runner.SCHEDULE_RENEWAL, dict(config(), epochs=80))


def test_floor_schedule_and_restore_are_unchanged(tmp_path):
    checkpoint = parent(tmp_path)
    a, oa, sa, c = restore(checkpoint, branch=False)
    for epoch in range(51, 71):
        assert sa.get_last_lr() == [lr * .02 for lr in sa.base_lrs]
        step(a, oa, sa)
        if epoch == 60:
            runner.save_checkpoint(tmp_path / 'floor60.pt', a, oa, sa, epoch, c)
    saved = torch.load(tmp_path / 'floor60.pt', weights_only=False)
    assert 'schedule_branch' not in saved and 'schedule_branch' not in saved['scheduler']
    b, ob, sb, _ = restore(saved, branch=False)
    for _ in range(61, 71):
        step(b, ob, sb)
    exact(a.snapshot(), b.snapshot())
    exact(oa.state_dict(), ob.state_dict())
    exact(sa.state_dict(), sb.state_dict())


def test_renewal_configs_change_only_declared_labels():
    root = Path('configs/cifar/horizon_e50_e70_20260924')
    for scheme in ('legacy', 'ours'):
        floor = json.loads((root / f'{scheme}_floor.json').read_text())
        renewal = json.loads((root / f'{scheme}_renewal.json').read_text())
        assert {key for key in floor if floor[key] != renewal[key]} == {
            'study_id', 'arm_id', 'evidence_class'}
        assert floor.keys() == renewal.keys()
    assert json.loads((root / 'schedule_branch.json').read_text()) == runner.SCHEDULE_RENEWAL
