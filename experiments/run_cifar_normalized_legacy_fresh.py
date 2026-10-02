"""Foreground queue transport for the fresh normalized-legacy CIFAR assignment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parent
sys.path.insert(0, str(SOURCE))
from experiments import reporting

OUT = ROOT / 'cells/legacy_voltage_normalized_fresh50'
CONFIG = SOURCE / 'configs/cifar/normalized_legacy_fresh50_20260927.json'


def validate():
    errors = reporting.validate_run(OUT)
    assert not errors, errors
    result = json.loads((OUT / 'result.json').read_text())
    rows = [json.loads(line) for line in (OUT / 'metrics.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in rows] == list(range(1, 51))
    assert all(row['steps'] == row['epoch'] * 1407 for row in rows)
    metrics = result['terminal_metrics']
    assert metrics['epochs_completed'] == 50 and metrics['steps'] == 70350
    assert metrics['starting_epoch'] == 0 and metrics['final_solver_audit_passed']
    assert result['completion']['criteria_met'] and not result['completion']['official_test_read']
    print('VALIDATED_FRESH_50_EPOCHS', flush=True)


def main(mode):
    if mode == 'validate':
        validate()
        return
    if mode == 'collect':
        queue = '/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py'
        status = json.loads(subprocess.check_output([sys.executable, queue, 'status', os.environ['RUN_WATCH_JOB_ID']], text=True))
        attempt = status['attempts'][-1]
        remote = '/home/filip/server_code/results/' + ROOT.name
        for subdir in ['cells', 'smoke']:
            destination = ROOT / subdir
            destination.mkdir(exist_ok=True)
            subprocess.run(['rsync', '-a', '-e', 'ssh -F /home/filip/.ssh/config -o BatchMode=yes',
                            f"filip@{attempt['target']}:{remote}/{subdir}/", str(destination) + '/'], check=True)
        validate()
        return
    assert mode == 'run'
    assert not OUT.exists(), 'Refuse duplicate production'
    data = Path(os.environ['CIFAR_DATA_ROOT'])
    for name, digest in json.loads((SOURCE / 'training-data-hashes.json').read_text()).items():
        assert reporting.sha256_file(data / name) == digest, name
    errors = reporting.validate_run(ROOT / 'prepare')
    assert not errors, errors
    token = os.environ['RUN_WATCH_ATTEMPT']
    attempt = Path.home() / '.local/state/run-watch/queue-workers/attempts' / token
    spec = json.loads((attempt / 'spec.json').read_text())
    receipt = json.loads((attempt / 'receipt.json').read_text())
    deadline = receipt['started_at'] + spec['allowed_seconds'] - 120
    if spec.get('stop_at'):
        deadline = min(deadline, spec['stop_at'] - 120)
    assert deadline - time.time() > 1500, 'Insufficient remaining queue budget'
    base = [sys.executable, '-u', '-m', 'experiments.train_cifar_l8_analog', str(CONFIG),
            '--device', 'cuda', '--target', os.environ['CIFAR_QUEUE_TARGET'], '--require-gpu-resident',
            '--assets', str(ROOT / 'prepare'), '--dataset-root', str(data),
            '--source-identity', str(SOURCE / 'source_identity.json')]
    smoke = ROOT / 'smoke' / token
    subprocess.run(base + ['--mode', 'smoke', '--smoke-batches', '1', '--output-dir', str(smoke)],
                   check=True, timeout=min(600, deadline - time.time()))
    assert not reporting.validate_run(smoke)
    rows = [json.loads(line) for line in (smoke / 'metrics.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and (rows[0]['epoch'], rows[0]['steps']) == (1, 1)
    assert (rows[0]['train_examples'], rows[0]['validation_examples']) == (32, 16)
    command = base + ['--mode', 'train', '--output-dir', str(OUT), '--stop-before',
                     datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
                     '--initial-epoch-seconds', '900', '--deadline-reserve-seconds', '180']
    print('FRESH_SMOKE_PASSED_STARTING_TRAIN', flush=True)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main(sys.argv[1])
