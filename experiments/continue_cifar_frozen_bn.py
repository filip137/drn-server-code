"""Queue transport for the three frozen-affine BN epoch-10 to 30 continuations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from experiments import reporting

SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parent


def validate(scheme):
    import torch

    out = ROOT / 'cells' / f'{scheme}_frozen_bn_e10_e30'
    errors = reporting.validate_run(out)
    assert not errors, errors
    rows = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows] == list(range(11, 31))
    assert all(r['steps'] == r['epoch'] * 1407 for r in rows)
    assert all((r['train_examples'], r['validation_examples']) == (45000, 5000) for r in rows)
    result = json.loads((out / 'result.json').read_text())
    metrics = result['terminal_metrics']
    assert (metrics['starting_epoch'], metrics['epochs_completed'], metrics['steps']) == (10, 30, 42210)
    assert metrics['final_solver_audit_passed']
    assert result['completion']['criteria_met'] and not result['completion']['official_test_read']
    checkpoint = torch.load(out / 'checkpoints/final_model.pt', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == checkpoint['scheduler']['last_epoch'] == 30
    assert checkpoint['config']['bn_affine_trainable'] is False
    assert checkpoint['config']['scheduler'] == {'final_lr_ratio': 0.02, 'horizon_epochs': 50}
    assert {int(v['step']) for v in checkpoint['optimizer']['state'].values()} == {42210}
    print(f'VALIDATED {scheme}: epochs 11-30, frozen BN affine, full validation', flush=True)


def main(mode, scheme):
    assert scheme in {'baseline', 'ours', 'legacy'}
    out = ROOT / 'cells' / f'{scheme}_frozen_bn_e10_e30'
    if mode == 'validate':
        validate(scheme)
        return
    if mode == 'collect':
        queue = '/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py'
        status = json.loads(subprocess.check_output(
            [sys.executable, queue, 'status', os.environ['RUN_WATCH_JOB_ID']], text=True))
        target = status['attempts'][-1]['target']
        if target != 'local':
            remote = f'/home/filip/server_code/results/{ROOT.name}/cells/{out.name}/'
            out.mkdir(parents=True, exist_ok=True)
            subprocess.run(['rsync', '-a', '-e', 'ssh -F /home/filip/.ssh/config -o BatchMode=yes',
                            f'filip@{target}:{remote}', str(out) + '/'], check=True)
        validate(scheme)
        reporting.atomic_write_json(ROOT / 'analysis' / f'{scheme}-collection.json', {
            'scheme': scheme, 'target': target, 'validated': True,
            'job_id': os.environ['RUN_WATCH_JOB_ID'],
            'collected_at': datetime.now(timezone.utc).isoformat(),
            'result': str(out / 'result.json')})
        return
    assert mode == 'run'
    assert not out.exists(), 'Refuse duplicate production'
    token = os.environ['RUN_WATCH_ATTEMPT']
    attempt = Path.home() / '.local/state/run-watch/queue-workers/attempts' / token
    spec = json.loads((attempt / 'spec.json').read_text())
    receipt = json.loads((attempt / 'receipt.json').read_text())
    deadline = receipt['started_at'] + spec['allowed_seconds'] - 120
    if spec.get('stop_at'):
        deadline = min(deadline, spec['stop_at'] - 120)
    assert deadline - time.time() > 2400, 'Insufficient remaining queue budget'
    config = SOURCE / 'configs/cifar/frozen_bn_e10_e30_20260928' / f'{scheme}.json'
    command = [sys.executable, '-u', '-m', 'experiments.train_cifar_l8_analog', str(config),
               '--mode', 'train', '--output-dir', str(out), '--device', 'cuda',
               '--target', os.environ['CIFAR_QUEUE_TARGET'], '--require-gpu-resident',
               '--assets', str(ROOT / 'assets' / scheme),
               '--resume-run', str(ROOT / 'parents' / scheme),
               '--dataset-root', os.environ['CIFAR_DATA_ROOT'],
               '--source-identity', str(SOURCE / 'source_identity.json'),
               '--stop-before', datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
               '--initial-epoch-seconds', os.environ['CIFAR_EPOCH_SECONDS'],
               '--deadline-reserve-seconds', '180']
    print(f'START_TRAIN_NO_SMOKE {scheme} epochs 11-30', flush=True)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main(*sys.argv[1:])
