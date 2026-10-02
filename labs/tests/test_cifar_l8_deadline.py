from datetime import timezone
import json
from pathlib import Path
import sys

import numpy as np

import pytest

from experiments.train_cifar_l8_analog import epoch_admission_seconds, parse_stop_before
from experiments import reporting, train_cifar_l8_analog as runner


def test_epoch_admission_accounts_for_slow_recent_epochs_and_shutdown():
    assert epoch_admission_seconds([], 900, 120) == 1245
    assert epoch_admission_seconds([750, 1100, 800], 900, 120) == 1495
    assert epoch_admission_seconds([1100, 750, 800, 760], 900, 120) == 1120


def test_deadline_timezone_is_explicit_and_converted_to_utc():
    parsed = parse_stop_before('2026-09-24T08:00:00+02:00')
    assert parsed.hour == 6 and parsed.tzinfo == timezone.utc
    assert parsed == parse_stop_before('2026-09-24T06:00:00Z')
    with pytest.raises(Exception, match='explicit timezone'):
        parse_stop_before('2026-09-24T08:00:00')


@pytest.mark.parametrize('qualify_solver', [False, True])
def test_deadline_writes_a_valid_paused_bundle_instead_of_failure(tmp_path, monkeypatch, qualify_solver):
    config = json.loads(Path('configs/cifar/cifar10_l8_analog_bs16_seed0.json').read_text())
    config.update(blocks=[[2, 2, 2], [3, 3, 3], [4, 4]], input_size=8, num_workers=0)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    identity = tmp_path / 'source.json'
    identity.write_text('{"source_sha256": {}}')
    monkeypatch.setattr(runner, 'read_training_data', lambda root:
                        (np.zeros((4, 32, 32, 3), dtype=np.uint8), np.arange(4)))
    monkeypatch.setattr(runner, 'make_split', lambda *args: {'train': [0, 1, 2], 'validation': [3]})
    audit_calls = []
    def audit(*args):
        assert qualify_solver, 'Default preparation must not perform a solver audit'
        audit_calls.append(True)
        return {'passed': True, 'selected_iterations': config['iterations']}
    monkeypatch.setattr(runner, 'audit', audit)
    monkeypatch.setattr(runner.signal, 'signal', lambda *args: None)
    common = [str(path), '--dataset-root', str(tmp_path), '--source-identity', str(identity), '--device', 'cpu']
    assets = tmp_path / 'assets'
    monkeypatch.setattr(sys, 'argv', ['runner', *common, '--mode', 'prepare', '--output-dir', str(assets),
                                    *(['--qualify-solver'] if qualify_solver else [])])
    runner.main()
    assert len(audit_calls) == int(qualify_solver)
    assert reporting.validate_run(assets) == []
    qualification = json.loads((assets / 'artifacts/qualification.json').read_text())
    if not qualify_solver:
        assert qualification['audit_performed'] is False and qualification['passed'] is None
        assert qualification['selected_iterations'] == config['iterations']
    run = tmp_path / 'paused'
    monkeypatch.setattr(sys, 'argv', ['runner', *common, '--mode', 'train', '--assets', str(assets),
                                    '--output-dir', str(run), '--stop-before', '2020-01-01T08:00:00+01:00'])
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code == 75
    status = json.loads((run / 'status.json').read_text())
    assert status['state'] == 'paused' and status['progress']['epoch'] == 0
    assert 'error' not in status and not (run / 'result.json').exists()
    assert reporting.validate_run(run) == []


def test_prepared_iterations_preserves_failure_and_config_guards():
    config = {'iterations': [6, 6, 4]}
    with pytest.raises(ValueError, match='did not qualify'):
        runner.prepared_iterations({'passed': False, 'selected_iterations': None}, config)
    record = {'status': 'configured_without_audit', 'audit_performed': False,
              'passed': None, 'selected_iterations': [8, 8, 8]}
    with pytest.raises(ValueError, match='differ from the declared config'):
        runner.prepared_iterations(record, config)
    record.update(selected_iterations=config['iterations'], passed=True)
    with pytest.raises(ValueError, match='cannot claim a solver pass'):
        runner.prepared_iterations(record, config)
    assert runner.prepared_iterations({'passed': True, 'selected_iterations': [8, 8, 8]}, config) == [8, 8, 8]
