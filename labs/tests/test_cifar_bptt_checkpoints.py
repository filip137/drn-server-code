"""Real tiny nonlinear BPTT training with requested full epoch snapshots."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from experiments import reporting, train_cifar_l8_analog as runner
from labs.cifar_l8_analog import CifarL8Analog
from labs.tests.test_cifar_nonlinear_head import config


@pytest.mark.parametrize('keep_epochs, final_audit', [(None, None), ([1, 3], True)])
def test_prepare_and_nonlinear_bptt_save_requested_full_checkpoints(
        tmp_path, monkeypatch, keep_epochs, final_audit):
    c = config(hidden_units=8)
    c.update(epochs=3, input_size=32, batch_size=3, evaluation_batch_size=3, num_workers=0,
             precision='float32', gradient_method='bptt')
    if keep_epochs is not None:
        c['checkpoint_epochs'] = keep_epochs
    if final_audit is not None:
        c['final_solver_audit'] = final_audit
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(c))
    identity = tmp_path / 'source.json'
    identity.write_text('{"source_sha256": {}}')
    images = np.random.default_rng(19).integers(0, 256, (9, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(9)
    monkeypatch.setattr(runner, 'read_training_data', lambda root: (images, labels))
    monkeypatch.setattr(runner, 'make_split', lambda *args:
                        {'train': list(range(6)), 'validation': list(range(6, 9))})
    audit_calls = []

    def audit(*args):
        audit_calls.append(True)
        return {'passed': True, 'selected_iterations': c['iterations']}

    monkeypatch.setattr(runner, 'audit', audit)
    runner.deterministic_setup(c['seed'])
    source = CifarL8Analog(c)
    initializer = tmp_path / 'initializer.pt'
    torch.save({'epoch': 0, 'model': source.snapshot()}, initializer)
    common = [str(path), '--dataset-root', str(tmp_path), '--source-identity', str(identity),
              '--device', 'cpu']
    assets = tmp_path / 'assets'
    monkeypatch.setattr(sys, 'argv', ['runner', *common, '--mode', 'prepare', '--output-dir', str(assets),
                                    '--initial-checkpoint', str(initializer)])
    runner.main()
    assert audit_calls == []
    assert reporting.validate_run(assets) == []
    prepared = torch.load(assets / 'checkpoints/initial_model.pt', weights_only=False)
    for group, values in source.snapshot().items():
        for name, value in values.items():
            assert torch.equal(value, prepared['model'][group][name]), name

    run = tmp_path / 'train'
    monkeypatch.setattr(sys, 'argv', ['runner', *common, '--mode', 'train', '--output-dir', str(run),
                                    '--assets', str(assets)])
    runner.main()
    assert reporting.validate_run(run) == []
    assert len(audit_calls) == int(bool(final_audit))
    assert (run / 'artifacts/final_solver_audit.json').exists() == bool(final_audit)
    terminal = json.loads((run / 'result.json').read_text())['terminal_metrics']
    assert terminal['final_solver_audit_passed'] is (True if final_audit else None)
    assert terminal['epochs_completed'] == 3
    checkpoints = run / 'checkpoints'
    assert sorted(p.name for p in checkpoints.glob('epoch_*.pt')) == [
        f'epoch_{epoch:03d}.pt' for epoch in (keep_epochs or [])]
    paths = [checkpoints / 'final_model.pt'] + [checkpoints / f'epoch_{epoch:03d}.pt' for epoch in (keep_epochs or [])]
    for checkpoint_path in paths:
        saved = torch.load(checkpoint_path, weights_only=False)
        epoch = saved['epoch']
        assert saved['scheduler']['last_epoch'] == epoch
        assert {'torch_rng', 'numpy_rng', 'python_rng', 'cuda_rng'} <= saved.keys()
        assert saved['config'] == c and saved['precision'] == 'float32'
        for name in ('head.hidden_weight', 'head.weight'):
            weight = saved['model']['conductances'][name]
            assert weight.dtype == torch.float32 and torch.isfinite(weight).all()
            assert not torch.equal(weight, prepared['model']['conductances'][name]), name
            group, = [group for group in saved['optimizer']['param_groups'] if group['name'] == name]
            state = saved['optimizer']['state'][group['params'][0]]
            assert state['step'].item() == epoch * 2
            assert state['exp_avg'].dtype == state['exp_avg_sq'].dtype == torch.float32
            assert torch.isfinite(state['exp_avg']).all() and state['exp_avg'].norm() > 0
        for block in range(3):
            assert saved['model']['module'][f'bridges.{block}.1.num_batches_tracked'].item() == epoch * 2
        restored = CifarL8Analog(c)
        runner.configure_training_precision(restored, c)
        restored.restore(saved['model'])
        optimizer, scheduler = runner.make_optimizer(restored, c)
        optimizer.load_state_dict(saved['optimizer'])
        runner.verify_training_precision(restored, c, optimizer)
    rows = list(reporting.iter_metrics(run / 'metrics.jsonl'))
    for row in rows:
        assert row['first_batch_gradient_norms']['head.hidden_weight'] > 0
        assert row['first_batch_gradient_norms']['head.weight'] > 0


@pytest.mark.parametrize('epochs', [[0], [6], [1, 1], [True], [1.5], '1,5'])
def test_invalid_checkpoint_epoch_requests_fail_closed(epochs):
    with pytest.raises(ValueError, match='checkpoint_epochs'):
        runner.requested_checkpoint_epochs({'epochs': 5, 'checkpoint_epochs': epochs})
