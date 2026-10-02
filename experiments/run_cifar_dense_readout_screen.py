"""Queue transport, collection and predeclared review of the dense-LR screen."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments import reporting

SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parent


def equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def validate(scheme):
    import torch
    out = ROOT / 'probes' / scheme
    assert not reporting.validate_run(out)
    manifest = json.loads((out / 'manifest.json').read_text())
    arms = manifest['intervention']['arms']
    assert list(arms.values()).count(1.) == 1 and len(arms) == 3
    rows = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    index = {(r['arm'], r['updates']): r for r in rows}
    assert len(rows) == len(index) == 9
    assert set(index) == {(a, n) for a in arms for n in [0, 100, 400]}
    assert len({r['training_order_sha256'] for r in rows}) == 1
    parent = torch.load(ROOT / 'inputs' / f'{scheme}_e10.pt', map_location='cpu', weights_only=False)
    for arm, multiplier in arms.items():
        assert index[arm, 0]['measurements'] == index['control', 0]['measurements']
        for n in [0, 100, 400]:
            r = index[arm, n]
            assert r['adam_steps'] == 14070 + n
            for cohort, count in [('train', 2048), ('validation', 1024)]:
                m = r['measurements'][cohort]
                assert m['examples'] == count and math.isfinite(m['loss'])
                assert 0 <= m['accuracy'] <= 1
            for name, lr in index['control', n]['learning_rates'].items():
                assert math.isclose(r['learning_rates'][name], lr * (multiplier if name == 'head.weight' else 1.), rel_tol=1e-13)
        initial = torch.load(out / 'checkpoints' / f'{arm}_initial_model.pt', map_location='cpu', weights_only=False)
        equal(initial['model'], parent['model'])
        equal(initial['optimizer']['state'], parent['optimizer']['state'])
        equal(initial['torch_rng'], parent['torch_rng'])
        final = torch.load(out / 'checkpoints' / f'{arm}_final_model.pt', map_location='cpu', weights_only=False)
        assert final['epoch'] == final['scheduler']['last_epoch'] == 10
        assert final['config']['bn_affine_trainable'] is False
        assert {int(v['step']) for v in final['optimizer']['state'].values()} == {14470}
        assert [g['lr'] for g in final['optimizer']['param_groups']] == final['scheduler']['_last_lr']
        assert final['config']['optimizer']['conv_learning_rates'] == parent['config']['optimizer']['conv_learning_rates']
        assert math.isclose(final['config']['optimizer']['head_learning_rate'], parent['config']['optimizer']['head_learning_rate'] * multiplier, rel_tol=1e-13)
        record = json.loads((out / 'artifacts' / f'{arm}_training.json').read_text())
        assert record['train_examples'] == 12800 and not record['native_epoch_resume_safe']
        del initial, final
    result = json.loads((out / 'result.json').read_text())
    assert result['completion']['criteria_met'] and not result['completion']['official_test_read']
    print('VALIDATED', scheme, 'three matched branches, 400 updates each', flush=True)


def review():
    if not all((ROOT / 'probes' / s / 'result.json').exists() for s in ['legacy', 'ours']):
        return
    response = {}
    for scheme in ['legacy', 'ours']:
        out = ROOT / 'probes' / scheme
        rows = [json.loads(x) for x in (out / 'metrics.jsonl').read_text().splitlines()]
        indexed = {(r['arm'], r['updates']): r for r in rows}
        arms = json.loads((out / 'manifest.json').read_text())['intervention']['arms']
        response[scheme] = []
        for arm, multiplier in arms.items():
            for step in [0, 100, 400]:
                a = indexed['control', step]['measurements']; b = indexed[arm, step]['measurements']
                response[scheme].append({'arm': arm, 'multiplier': multiplier, 'updates': step,
                    'measurements': b,
                    'relative_train_ce_improvement': (a['train']['loss'] - b['train']['loss']) / a['train']['loss'],
                    'relative_val_ce_change': (b['validation']['loss'] - a['validation']['loss']) / a['validation']['loss'],
                    'val_accuracy_change_pp': 100 * (b['validation']['accuracy'] - a['validation']['accuracy'])})
    low = min(r['multiplier'] for r in response['legacy'])
    endpoints = {s: next(r for r in response[s] if r['updates'] == 400 and r['multiplier'] == low) for s in response}
    legacy = endpoints['legacy']; ours = endpoints['ours']
    excess = legacy['relative_train_ce_improvement'] - ours['relative_train_ce_improvement']
    support = (legacy['relative_train_ce_improvement'] >= .01 and legacy['relative_val_ce_change'] <= .01
               and legacy['val_accuracy_change_pp'] >= -.5 and excess >= .005)
    contrary = all(r['relative_train_ce_improvement'] <= -.01 for r in response['legacy'] if r['updates'] == 400 and r['multiplier'] != 1.)
    verdict = 'supported' if support else 'contradicted' if contrary else 'inconclusive'
    reporting.atomic_write_json(ROOT / 'analysis/summary.json', {'response': response, 'verdict': verdict,
        'excess_relative_training_ce_improvement': excess, 'primary_multiplier': low,
        'scope': 'One-seed 400-update screen; not evidence of an optimal LR or long-run scheme-gap cause.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for i, scheme in enumerate(['legacy', 'ours']):
        for multiplier in sorted({r['multiplier'] for r in response[scheme]}, reverse=True):
            series = sorted((r for r in response[scheme] if r['multiplier'] == multiplier), key=lambda r:r['updates'])
            for ax, cohort in zip(axes[i], ['train', 'validation']):
                ax.plot([r['updates'] for r in series], [r['measurements'][cohort]['loss'] for r in series], '-o', label=f'Head LR ×{multiplier:.4g}')
                ax.set(title=f'{scheme}: fixed {cohort} CE', xlabel='Updates from epoch 10'); ax.grid(alpha=.2); ax.legend()
    fig.savefig(ROOT / 'analysis/dense_readout_response.jpg', dpi=180); plt.close(fig)
    repo = ROOT.parents[1]; campaign = repo / 'campaigns/cifar10-legacy-gap'
    series = campaign / 'series/002-residual-gap'; note = series / 'results/exp-016-dense-readout-lr.md'
    if note.exists():
        return
    lines = ['---', 'experiment: "exp-016"', 'evidence: "validated-local"',
        'summary: ' + json.dumps('Matched 400-update dense-LR screen: H-010 ' + verdict + ' in its predeclared short-window regime.'),
        'verdicts: {"H-010": "' + {'supported':'supports', 'contradicted':'contradicts', 'inconclusive':'inconclusive'}[verdict] + '"}', '---', '# Dense-readout LR screen', '',
        'All six branches completed400 updates from their original epoch10 full-state checkpoints.',
        'Within each scheme, starting predictions, model, Adam moments and RNG matched; only',
        'readout LR differed. Convolution rates were unchanged. All18 evaluation records validated.',
        'Frozen BN affine, original unnormalized legacy, fixed train2048/validation1024 cohorts;',
        'one seed, same Trex runtime. No smokes or official test read.', '',
        '| Scheme | Head LR multiplier | Training CE at400 | Training CE improvement | Validation CE change | Validation accuracy change |',
        '|---|---:|---:|---:|---:|---:|']
    for scheme in ['legacy', 'ours']:
        for r in sorted((r for r in response[scheme] if r['updates'] == 400), key=lambda r:-r['multiplier']):
            lines.append(f"| {scheme} | {r['multiplier']:.8g} | {r['measurements']['train']['loss']:.6f} | {100*r['relative_train_ce_improvement']:+.2f}% | {100*r['relative_val_ce_change']:+.2f}% | {r['val_accuracy_change_pp']:+.3f} pp |")
    lines += ['', f'H-010 verdict: **{verdict}**, applying the predeclared primary low-rate criterion.',
        f'Legacy-minus-ours relative training-loss improvement: {100*excess:+.3f} percentage points.',
        'The one-third dose is secondary and cannot replace the primary endpoint.',
        'This is a bounded fitting test, not an optimal-rate estimate or a causal explanation',
        'of the historical scheme ranking. No longer continuation is automatically launched.', '',
        'Evidence root: `results/cifar10-dense-readout-screen-20260928-v1/`; complete arithmetic',
        '`analysis/summary.json`, plot `analysis/dense_readout_response.jpg`, and canonical',
        '`probes/{legacy,ours}` bundles. Partial-epoch checkpoints are not native-resume-safe.']
    note.write_text('\n'.join(lines) + '\n')
    experiment = series / 'experiments/exp-016-dense-readout-lr.md'
    text = experiment.read_text().replace('status: "running"', 'status: "complete"')
    text = text[:text.index('Current handoff:')] + 'Current handoff: both schemes collected, validated and reviewed against H-010. See the result note.\n'
    experiment.write_text(text)
    ideas = campaign / 'ideas.md'
    ideas.write_text(ideas.read_text().replace('Active: matched 400-update dense-LR screen, frozen BN;', 'Completed 400-update screen, frozen BN; H-010 ' + verdict + ';'))
    subprocess.run([sys.executable, str(repo / '.agents/skills/experiment-loop/scripts/ledger.py'), str(campaign)], check=True)
    print('H-010', verdict, flush=True)


def main(mode, scheme):
    assert scheme in {'legacy', 'ours'}
    out = ROOT / 'probes' / scheme
    if mode == 'validate':
        validate(scheme)
    elif mode == 'collect':
        queue = '/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py'
        status = json.loads(subprocess.check_output([sys.executable, queue, 'status', os.environ['RUN_WATCH_JOB_ID']], text=True))
        target = status['attempts'][-1]['target']
        remote = f'/home/filip/server_code/results/{ROOT.name}/probes/{scheme}/'
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync', '-a', '-e', 'ssh -F /home/filip/.ssh/config -o BatchMode=yes', f'filip@{target}:{remote}', str(out) + '/'], check=True)
        validate(scheme)
        reporting.atomic_write_json(ROOT / 'analysis' / f'{scheme}-collection.json', {'validated': True, 'target': target, 'job': os.environ['RUN_WATCH_JOB_ID']})
        if all((ROOT / 'analysis' / f'{s}-collection.json').exists() for s in ['legacy', 'ours']):
            review()
    else:
        assert mode == 'run' and not out.exists()
        token = os.environ['RUN_WATCH_ATTEMPT']
        attempt = Path.home() / '.local/state/run-watch/queue-workers/attempts' / token
        spec = json.loads((attempt / 'spec.json').read_text()); receipt = json.loads((attempt / 'receipt.json').read_text())
        deadline = min(receipt['started_at'] + spec['allowed_seconds'], spec.get('stop_at') or float('inf')) - 120
        remaining = deadline - time.time()
        assert remaining > 600
        low = json.loads((ROOT / 'inputs/intervention.json').read_text())['low_multiplier']
        command = [sys.executable, '-u', '-m', 'experiments.screen_cifar_l8_conv_lr', '--root', str(ROOT),
            '--output-dir', str(out), '--dataset-root', os.environ['CIFAR_DATA_ROOT'], '--target', os.environ['CIFAR_QUEUE_TARGET'],
            '--parameter-family', 'head', '--checkpoint', f'{scheme}_e10.pt', '--multipliers', '1', str(1/3), str(low),
            '--updates', '400', '--maximum-seconds', str(remaining)]
        print('START_DENSE_LR_SCREEN_NO_SMOKE', scheme, flush=True)
        os.execv(sys.executable, command)


if __name__ == '__main__':
    main(*sys.argv[1:])
