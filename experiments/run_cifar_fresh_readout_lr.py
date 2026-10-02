"""Direct fresh CIFAR training and queue collection for the selected readout LRs."""
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
    out = ROOT / 'cells' / scheme
    assert not reporting.validate_run(out)
    rows = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows] == list(range(1, 11))
    assert all(r['steps'] == r['epoch'] * 1407 for r in rows)
    assert all((r['train_examples'], r['validation_examples']) == (45000, 5000) for r in rows)
    result = json.loads((out / 'result.json').read_text()); m = result['terminal_metrics']
    assert (m['starting_epoch'], m['epochs_completed'], m['steps']) == (0, 10, 14070)
    assert m['final_solver_audit_passed'] and result['completion']['criteria_met']
    assert not result['completion']['official_test_read']
    initial = torch.load(out / 'checkpoints/initial_model.pt', map_location='cpu', weights_only=False)
    final = torch.load(out / 'checkpoints/final_model.pt', map_location='cpu', weights_only=False)
    assert initial['epoch'] == initial['scheduler']['last_epoch'] == 0
    assert not initial['optimizer']['state'], 'Fresh Adam must have no inherited moments'
    assert final['epoch'] == final['scheduler']['last_epoch'] == 10
    assert final['config']['bn_affine_trainable'] is True
    assert {int(v['step']) for v in final['optimizer']['state'].values()} == {14070}
    module = final['model']['module']; groups = {g['name'] for g in final['optimizer']['param_groups']}
    tracked = [n for n in module if n.endswith('.num_batches_tracked')]
    assert len(tracked) == 3
    for name in tracked:
        prefix = name.removesuffix('.num_batches_tracked')
        assert int(module[name]) == 14070 and int(initial['model']['module'][name]) == 0
        for suffix in ['weight', 'bias']:
            key = prefix + '.' + suffix
            assert key in groups
            assert not torch.equal(module[key], initial['model']['module'][key]), key
    print('VALIDATED_FRESH_10_EPOCHS_TRAINABLE_BN', scheme, flush=True)


def main(mode, scheme):
    assert scheme in {'legacy', 'ours'}
    out = ROOT / 'cells' / scheme
    if mode == 'validate':
        validate(scheme)
        return
    if mode == 'collect':
        queue = '/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py'
        status = json.loads(subprocess.check_output([sys.executable, queue, 'status', os.environ['RUN_WATCH_JOB_ID']], text=True))
        target = status['attempts'][-1]['target']
        out.mkdir(parents=True, exist_ok=True)
        remote = f'/home/filip/server_code/results/{ROOT.name}/cells/{scheme}/'
        subprocess.run(['rsync', '-a', '-e', 'ssh -F /home/filip/.ssh/config -o BatchMode=yes',
                        f'filip@{target}:{remote}', str(out) + '/'], check=True)
        validate(scheme)
        reporting.atomic_write_json(ROOT / 'analysis' / f'{scheme}-collection.json', {
            'validated': True, 'job_id': os.environ['RUN_WATCH_JOB_ID'], 'target': target})
        subprocess.run([sys.executable, str(ROOT / 'analysis/review.py')], check=True,
                       env=dict(os.environ, MPLCONFIGDIR='/tmp/cifar-fresh-readout-mpl'))
        return
    assert mode == 'run' and not out.exists()
    attempt = Path.home() / '.local/state/run-watch/queue-workers/attempts' / os.environ['RUN_WATCH_ATTEMPT']
    spec = json.loads((attempt / 'spec.json').read_text()); receipt = json.loads((attempt / 'receipt.json').read_text())
    deadline = min(receipt['started_at'] + spec['allowed_seconds'], spec.get('stop_at') or float('inf')) - 120
    assert deadline - time.time() > 1500
    config = SOURCE / 'configs/cifar/fresh_readout_lr_20260928' / f'{scheme}.json'
    command = [sys.executable, '-u', '-m', 'experiments.train_cifar_l8_analog', str(config),
        '--mode', 'train', '--output-dir', str(out), '--device', 'cuda', '--require-gpu-resident',
        '--target', os.environ['CIFAR_QUEUE_TARGET'], '--assets', str(ROOT / 'assets' / scheme),
        '--dataset-root', os.environ['CIFAR_DATA_ROOT'], '--source-identity', str(SOURCE / 'source_identity.json'),
        '--stop-before', datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
        '--initial-epoch-seconds', '900', '--deadline-reserve-seconds', '180']
    print('START_FRESH_TRAINABLE_BN_NO_SMOKE', scheme, flush=True)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main(*sys.argv[1:])
