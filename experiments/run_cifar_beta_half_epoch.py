"""Queue transport, collection and comparison for the epoch10 beta screen."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments import reporting

SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parent


def validate(case):
    import torch
    out = ROOT / 'cells' / case
    errors = reporting.validate_run(out)
    if errors:
        raise ValueError(errors)
    initial = torch.load(out/'checkpoints/initial_model.pt', map_location='cpu', weights_only=False)
    final = torch.load(out/'checkpoints/final_model.pt', map_location='cpu', weights_only=False)
    parent = torch.load(ROOT/'inputs/parent.pt', map_location='cpu', weights_only=False)
    for group in parent['model']:
        assert all(torch.equal(t, initial['model'][group][name]) for name, t in parent['model'][group].items())
    assert initial['epoch'] == final['epoch'] == final['scheduler']['last_epoch'] == 10
    assert final['native_epoch_resume_safe'] is False
    assert final['partial_epoch']['updates_in_epoch'] == 704
    assert final['partial_epoch']['examples_in_epoch'] == 22500
    assert {int(v['step']) for v in initial['optimizer']['state'].values()} == {14070}
    assert {int(v['step']) for v in final['optimizer']['state'].values()} == {14774}
    assert final['scheduler'] == initial['scheduler']
    assert [g['lr'] for g in final['optimizer']['param_groups']] == [g['lr'] for g in parent['optimizer']['param_groups']]
    for name, value in final['model']['module'].items():
        if name.endswith('num_batches_tracked'):
            assert int(value) == int(parent['model']['module'][name]) + 704
    rows = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
    endpoint = [r for r in rows if r.get('stage') == 'final']
    assert len(endpoint) == 1
    assert endpoint[0]['measurements']['validation']['examples'] == 5000
    spec = json.loads((ROOT/'inputs/cases.json').read_text())[case]
    if spec['method'] == 'hybrid_eqprop':
        for index in [0, 1]:
            d = json.loads((out/'artifacts'/f'beta_boundary_e{index:03}.json').read_text())
            assert d['examples'] == 256 and len(d['layers']) == 8 and len(d['batches']) == 64
            assert d['beta_state']['values'] == spec['betas']
    result = json.loads((out/'result.json').read_text())
    assert result['terminal_metrics']['updates'] == 704
    assert result['terminal_metrics']['training_examples'] == 22500
    return {'case': case, 'validated': True, 'final': endpoint[0]['measurements'],
            'training_order_sha256': final['partial_epoch']['training_order_sha256']}


def compare():
    """Only completed, collected cases enter the comparison; GPU work is over."""
    import fcntl
    import torch
    cases = json.loads((ROOT/'inputs/cases.json').read_text())
    with (ROOT/'analysis/comparison.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        complete = {name: json.loads((ROOT/'analysis'/f'{name}-collection.json').read_text())
                    for name in cases if (ROOT/'analysis'/f'{name}-collection.json').exists()}
        report = {'complete': len(complete) == len(cases), 'expected': list(cases),
                  'collected': list(complete), 'ranking': [], 'scope': 'ours seed0 epoch10 to10.5, no noise'}
        if 'native_bptt' in complete:
            reference = torch.load(ROOT/'cells/native_bptt/checkpoints/final_model.pt', map_location='cpu', weights_only=False)
            parent = torch.load(ROOT/'inputs/parent.pt', map_location='cpu', weights_only=False)
            ref_metrics = complete['native_bptt']['final']['validation']
            for name, receipt in complete.items():
                assert receipt['training_order_sha256'] == complete['native_bptt']['training_order_sha256']
                metrics = receipt['final']['validation']
                final = torch.load(ROOT/'cells'/name/'checkpoints/final_model.pt', map_location='cpu', weights_only=False)
                dot = nr = ne = 0.
                layer_rows = []
                for key, initial in parent['model']['conductances'].items():
                    if not key.startswith('blocks.'):
                        continue
                    x = (final['model']['conductances'][key] - initial).double().flatten()
                    y = (reference['model']['conductances'][key] - initial).double().flatten()
                    xy, xx, yy = float(x@y), float(x@x), float(y@y)
                    dot += xy; ne += xx; nr += yy
                    layer_rows.append({'name': key, 'cosine': xy/(xx*yy)**.5 if xx*yy else None,
                                       'norm_ratio': (xx/yy)**.5 if yy else None})
                report['ranking'].append({'case': name, 'validation': metrics,
                    'validation_ce_gap': metrics['loss']-ref_metrics['loss'],
                    'validation_accuracy_gap_pp': 100*(metrics['accuracy']-ref_metrics['accuracy']),
                    'conv_update_cosine': dot/(nr*ne)**.5 if nr*ne else None,
                    'conv_update_norm_ratio': (ne/nr)**.5 if nr else None, 'layers': layer_rows})
            report['ranking'].sort(key=lambda r: abs(r['validation_ce_gap']))
        reporting.atomic_write_json(ROOT/'analysis/comparison.json', report)
        if report['ranking']:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            rows = report['ranking']
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            labels = [r['case'] for r in rows]
            axes[0].bar(labels, [r['validation_ce_gap'] for r in rows])
            axes[0].set_ylabel('Validation CE minus native BPTT')
            axes[1].bar(labels, [r['validation_accuracy_gap_pp'] for r in rows])
            axes[1].set_ylabel('Validation accuracy gap (pp)')
            for ax in axes:
                ax.axhline(0, color='black', linewidth=.6)
                ax.tick_params(axis='x', rotation=50)
            fig.suptitle('Ours: matched half-epoch continuations from epoch 10' + ('' if report['complete'] else ' (partial)'))
            fig.tight_layout(); fig.savefig(ROOT/'analysis/comparison.jpg', dpi=160); plt.close(fig)


def main(mode, case):
    assert case in json.loads((ROOT/'inputs/cases.json').read_text())
    out = ROOT/'cells'/case
    if mode == 'validate':
        print(json.dumps(validate(case))); return
    if mode == 'collect':
        queue = '/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py'
        status = json.loads(subprocess.check_output([sys.executable, queue, 'status', os.environ['RUN_WATCH_JOB_ID']], text=True))
        target = status['attempts'][-1]['target']
        if target != 'local':
            out.mkdir(parents=True, exist_ok=True)
            remote = f'/home/filip/server_code/results/{ROOT.name}/cells/{case}/'
            subprocess.run(['rsync', '-a', '-e', 'ssh -F /home/filip/.ssh/config -o BatchMode=yes',
                            target+':'+remote, str(out)+'/'], check=True)
        receipt = validate(case)
        reporting.atomic_write_json(ROOT/'analysis'/f'{case}-collection.json', dict(receipt,
            target=target, job_id=os.environ['RUN_WATCH_JOB_ID']))
        compare(); return
    assert mode == 'run' and not out.exists()
    attempt = Path.home()/'.local/state/run-watch/queue-workers/attempts'/os.environ['RUN_WATCH_ATTEMPT']
    spec = json.loads((attempt/'spec.json').read_text())
    receipt = json.loads((attempt/'receipt.json').read_text())
    deadline = min(receipt['started_at'] + spec['allowed_seconds'], spec.get('stop_at') or float('inf'))
    remaining = deadline-time.time()-90
    if remaining < 600:
        raise RuntimeError('Insufficient remaining half-epoch allowance')
    command = [sys.executable, '-u', '-m', 'experiments.screen_cifar_beta_half_epoch',
               '--root', str(ROOT), '--case', case, '--output-dir', str(out),
               '--dataset-root', os.environ['CIFAR_DATA_ROOT'], '--target', os.environ['CIFAR_QUEUE_TARGET'],
               '--maximum-seconds', str(remaining)]
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main(*sys.argv[1:])
