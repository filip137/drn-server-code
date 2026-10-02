"""Launch and collect exp-026 using the standard CIFAR trainer and shared queue."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments import reporting

SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parent
CONFIGS = SOURCE / 'configs/cifar/float64_periodic_beta_20261001'
ARMS = ('periodic', 'fixed_e5')


def validate(arm):
    import torch
    out = ROOT / 'cells' / arm
    errors = reporting.validate_run(out)
    if errors:
        raise ValueError(errors)
    manifest = json.loads((out / 'manifest.json').read_text())
    config = manifest['resolved_config']
    assert config['precision'] == 'float64' and config['epochs'] == 5
    assert config['gradient_method'] == 'conv_eqprop_autograd_boundaries'
    assert config['conv_eqprop']['policy'] == ('periodic' if arm == 'periodic' else 'fixed')
    assert config['conv_eqprop']['rms_targets'] == [.01, .005, 1e-5]
    rows = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    rows = [r for r in rows if 'validation_accuracy' in r]
    assert [r['epoch'] for r in rows] == list(range(1, 6))
    assert all(r['steps'] == r['epoch'] * 1407 and r['train_examples'] == 45000
               and r['validation_examples'] == 5000 for r in rows)
    initial = torch.load(out / 'checkpoints/initial_model.pt', map_location='cpu', weights_only=False)
    final = torch.load(out / 'checkpoints/final_model.pt', map_location='cpu', weights_only=False)
    assert initial['epoch'] == initial['scheduler']['last_epoch'] == 0 and not initial['optimizer']['state']
    assert final['epoch'] == final['scheduler']['last_epoch'] == 5
    assert {int(v['step']) for v in final['optimizer']['state'].values()} == {7035}
    for checkpoint in (initial, final):
        assert all(t.dtype == torch.float64 for tensors in checkpoint['model'].values()
                   for t in tensors.values() if t.is_floating_point())
        for state in checkpoint['optimizer']['state'].values():
            assert state['exp_avg'].dtype == state['exp_avg_sq'].dtype == torch.float64
    assert all(int(t) == 7035 for n, t in final['model']['module'].items() if n.endswith('num_batches_tracked'))
    assets = torch.load(ROOT / 'assets/checkpoints/initial_model.pt', map_location='cpu', weights_only=False)
    assert initial['model'].keys() == assets['model'].keys()
    assert all(torch.equal(t, assets['model'][group][name]) for group, tensors in initial['model'].items()
               for name, t in tensors.items())
    calibrations = [json.loads(p.read_text()) for p in (out / 'artifacts').glob('beta_calibration_*.json')]
    if arm == 'periodic':
        assert len(calibrations) == 20
        assert Counter(c['epoch'] for c in calibrations) == {e: 4 for e in range(1, 6)}
        for cal in calibrations:
            assert cal['complete'] and len(cal['blocks']) == 3
            for block in cal['blocks']:
                selected = next(t for t in block['trials'] if t['beta_injected_B'] == block['beta'])
                assert abs(selected['relative_rms'] / block['target'] - 1) <= .03 + 1e-12
    else:
        assert not calibrations
        assert final['conv_beta_state']['values'] == config['conv_eqprop']['initial_betas']
        assert all(r['beta_state']['values'] == config['conv_eqprop']['initial_betas'] for r in rows)
    assert not list((out / 'artifacts').glob('beta_boundary_*.json'))
    result = json.loads((out / 'result.json').read_text())
    assert result['completion']['full_epoch_budget'] and not result['completion']['official_test_read']
    assert result['terminal_metrics']['final_solver_audit_passed'] is None
    return dict(arm=arm, validated=True, epochs=5, steps=7035,
                precision='float64', calibration_count=len(calibrations),
                validation_accuracy=rows[-1]['validation_accuracy'],
                validation_loss=rows[-1]['validation_loss'])


def plot_comparison():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    summaries = {}
    for arm in ARMS:
        receipt = ROOT / 'analysis' / f'{arm}-collection.json'
        if not receipt.exists():
            continue
        summaries[arm] = json.loads(receipt.read_text())
        rows = [json.loads(s) for s in (ROOT / 'cells' / arm / 'metrics.jsonl').read_text().splitlines()]
        rows = [r for r in rows if 'validation_accuracy' in r]
        epochs = [r['epoch'] for r in rows]
        for ax, key, scale, title in zip(axes, ['validation_accuracy', 'validation_loss', 'train_loss'],
                                        [100, 1, 1], ['Validation accuracy (%)', 'Validation CE', 'Training CE']):
            ax.plot(epochs, [r[key] * scale for r in rows], 'o-', label=arm)
            ax.set(xlabel='Epoch', ylabel=title, xticks=epochs)
            ax.grid(alpha=.2)
    axes[0].legend()
    complete = len(summaries) == 2
    fig.suptitle('Ours, float64: periodic versus epoch-5 fixed beta' + ('' if complete else ' (partial collection)'))
    fig.tight_layout()
    fig.savefig(ROOT / 'analysis/comparison.jpg', dpi=160)
    plt.close(fig)
    summary = dict(complete=complete, arms=summaries, official_test_read=False)
    if complete:
        a, b = summaries['periodic'], summaries['fixed_e5']
        summary.update(periodic_minus_fixed_accuracy_pp=100 * (a['validation_accuracy'] - b['validation_accuracy']),
                       periodic_minus_fixed_cross_entropy=a['validation_loss'] - b['validation_loss'])
    reporting.atomic_write_json(ROOT / 'analysis/comparison.json', summary)


def main(mode, arm):
    assert arm in ARMS
    out = ROOT / 'cells' / arm
    if mode == 'validate':
        print(json.dumps(validate(arm)), flush=True)
        return
    if mode == 'collect':
        queue = '/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py'
        status = json.loads(subprocess.check_output([sys.executable, queue, 'status', os.environ['RUN_WATCH_JOB_ID']], text=True))
        target = status['attempts'][-1]['target']
        assert target in ('fifi', 'loulou')
        out.mkdir(parents=True, exist_ok=True)
        remote = f'{target}:/home/filip/server_code/results/{ROOT.name}/cells/{arm}/'
        subprocess.run(['rsync', '-a', '-e', 'ssh -F /home/filip/.ssh/config -o BatchMode=yes', remote, str(out) + '/'], check=True)
        result = validate(arm)
        reporting.atomic_write_json(ROOT / 'analysis' / f'{arm}-collection.json',
                                    dict(result, job_id=os.environ['RUN_WATCH_JOB_ID'], target=target))
        plot_comparison()
        return
    assert mode == 'run' and not out.exists()
    import torch
    assert str(torch.__version__) == '2.11.0+cu128' and torch.version.cuda == '12.8'
    attempt = Path.home() / '.local/state/run-watch/queue-workers/attempts' / os.environ['RUN_WATCH_ATTEMPT']
    spec, receipt = (json.loads((attempt / name).read_text()) for name in ('spec.json', 'receipt.json'))
    deadline = min(receipt['started_at'] + spec['allowed_seconds'], spec.get('stop_at') or float('inf')) - 180
    assert deadline - time.time() > 7200
    command = [sys.executable, '-u', '-m', 'experiments.train_cifar_l8_analog', str(CONFIGS / f'{arm}.json'),
               '--mode', 'train', '--output-dir', str(out), '--device', 'cuda', '--require-gpu-resident',
               '--target', os.environ['CIFAR_QUEUE_TARGET'], '--assets', str(ROOT / 'assets'),
               '--dataset-root', os.environ['CIFAR_DATA_ROOT'], '--source-identity', str(SOURCE / 'source_identity.json'),
               '--stop-before', datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
               '--initial-epoch-seconds', '5400', '--deadline-reserve-seconds', '300']
    print('START_FLOAT64_HYBRID_EQPROP', arm, flush=True)
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main(*sys.argv[1:])
