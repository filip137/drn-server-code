"""Validate and summarize the three CIFAR epoch10-to30 continuations."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments import reporting


def verify_checkpoints(run, parent, config, rows):
    """Read-only checks of the actual resume state and terminal optimizer."""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and torch.equal(a, b)
        elif isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and np.array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        else:
            assert a == b
    def finite(value):
        if isinstance(value, torch.Tensor):
            assert bool(torch.isfinite(value).all())
        elif isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                finite(child)
    previous = torch.load(parent / 'checkpoints/final_model.pt', map_location='cpu', weights_only=False)
    initial = torch.load(run / 'checkpoints/initial_model.pt', map_location='cpu', weights_only=False)
    starting_epoch = config['continuation']['from_epoch']
    final_epoch = config['epochs']
    expected_steps = math.ceil(45000 / config['batch_size']) * final_epoch
    assert previous['epoch'] == initial['epoch'] == starting_epoch
    scheduler_metadata_note = None
    # PyTorch 2.5 adds its inert verbose=False field when restoring a 2.11
    # scheduler. Require the exact observed default; all numerical state is
    # still compared bitwise below.
    if 'verbose' not in previous['scheduler'] and initial['scheduler'].get('verbose') is False:
        initial['scheduler'] = dict(initial['scheduler'])
        initial['scheduler'].pop('verbose')
        scheduler_metadata_note = 'Restored scheduler adds inert verbose=False on PyTorch 2.5'
    for key in ('model', 'optimizer', 'scheduler', 'torch_rng', 'numpy_rng', 'python_rng'):
        equal(previous[key], initial[key])
    base_rates = previous['scheduler']['base_lrs']
    del previous, initial
    terminal = torch.load(run / 'checkpoints/final_model.pt', map_location='cpu', weights_only=False)
    assert terminal['epoch'] == terminal['scheduler']['last_epoch'] == final_epoch
    assert terminal['config'] == config
    finite(terminal['model']); finite(terminal['optimizer'])
    assert len(terminal['optimizer']['state']) == 19
    assert {int(x['step']) for x in terminal['optimizer']['state'].values()} == {expected_steps}
    counters = {k: int(v) for k, v in terminal['model']['module'].items() if k.endswith('num_batches_tracked')}
    assert len(counters) == 3 and set(counters.values()) == {expected_steps}
    assert len(terminal['model']['conductances']) == 9
    for tensor in terminal['model']['conductances'].values():
        assert float(tensor.min()) >= config['weight_min']
        assert float(tensor.max()) <= config['weight_max']
    def rates_at(epoch):
        schedule = config['scheduler']
        fraction = min(epoch / schedule['horizon_epochs'], 1)
        scale = schedule['final_lr_ratio'] + (1 - schedule['final_lr_ratio']) * .5 * (1 + math.cos(math.pi * fraction))
        return [x * scale for x in base_rates]
    def rates_equal(actual, expected):
        assert len(actual) == len(expected) == 19
        assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) for a, b in zip(actual, expected))
    for row in rows:
        rates_equal(row['learning_rates'], rates_at(row['epoch'] - 1))
    rates_equal([g['lr'] for g in terminal['optimizer']['param_groups']], rates_at(final_epoch))
    rates_equal(terminal['scheduler']['_last_lr'], rates_at(final_epoch))
    return {'resume_model_adam_scheduler_rng_exact': True,
            'scheduler_metadata_normalization': scheduler_metadata_note, 'all_checkpoint_tensors_finite': True,
            'optimizer_steps': expected_steps, 'bn_counters': counters, 'cosine_horizon_50_preserved': True,
            'all_nine_conductances_bounded': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--verify-checkpoints', action='store_true',
                        help='Additionally reconcile completed checkpoint tensors and optimizer states')
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    summary = {'complete': True, 'official_test_read': False, 'schemes': {}}
    checkpoint_checks = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = {'baseline': '#377eb8', 'ours': '#4daf4a', 'legacy': '#e41a1c'}
    lines = ['# CIFAR L8 continuation', '',
             'Exploratory seed-0 validation, batch32, Adam, trainable affine BN,',
             'crop/flip augmentation and cross-entropy. Original cosine horizon50.', '',
             '| Scheme | Epoch10 | Epoch20 | Epoch30 | Final CE | State |',
             '|---|---:|---:|---:|---:|---|']
    for scheme, color in colors.items():
        run = root / 'cells' / (scheme + '_e10_e30')
        manifest = json.loads((run / 'manifest.json').read_text())
        config = manifest['resolved_config']
        state = json.loads((run / 'status.json').read_text())['state']
        parent = root.parent / config['continuation']['parent_study'] / 'cells' / config['continuation']['parent_cell']
        before = [json.loads(x) for x in (parent / 'metrics.jsonl').read_text().splitlines()]
        after = [json.loads(x) for x in (run / 'metrics.jsonl').read_text().splitlines()] if (run / 'metrics.jsonl').exists() else []
        errors = []
        if state == 'complete':
            errors.extend(reporting.validate_run(run))
            if [x['epoch'] for x in after] != list(range(11, 31)):
                errors.append('Expected exactly epochs11 through30')
            final = json.loads((run / 'result.json').read_text())['terminal_metrics']
            if not final['final_solver_audit_passed']:
                errors.append('Final solver audit failed')
        else:
            summary['complete'] = False
            if not args.allow_partial:
                errors.append(f'Nonterminal or failed run: {state}')
        if reporting.sha256_file(parent / 'checkpoints/final_model.pt') != config['continuation']['checkpoint_sha256']:
            errors.append('Parent checkpoint SHA mismatch')
        parent_config = json.loads((parent / 'manifest.json').read_text())['resolved_config']
        allowed = {'study_id', 'arm_id', 'evidence_class', 'epochs', 'maximum_runtime_hours', 'continuation'}
        for key in set(config) | set(parent_config):
            if key not in allowed and config.get(key) != parent_config.get(key):
                errors.append(f'Scientific config changed: {key}')
        provenance = json.loads((run / 'artifacts/continuation.json').read_text())
        if provenance['from_epoch'] != 10 or provenance['parent_steps'] != 14070:
            errors.append('Incorrect resume epoch or step offset')
        for row in after:
            if (row['train_examples'], row['validation_examples'], row['steps']) != (45000, 5000, 1407 * row['epoch']):
                errors.append(f'Incomplete epoch {row["epoch"]}')
        if errors:
            raise ValueError({scheme: errors})
        if args.verify_checkpoints and state == 'complete':
            checkpoint_checks[scheme] = verify_checkpoints(run, parent, config, after)
        rows = before + after
        by_epoch = {x['epoch']: x for x in rows}
        item = {'state': state, 'target': manifest['runtime']['target'],
                'latest_epoch': rows[-1]['epoch'],
                'epoch10_accuracy': by_epoch[10]['validation_accuracy'],
                'latest_accuracy': rows[-1]['validation_accuracy'],
                'latest_loss': rows[-1]['validation_loss'],
                'new_epoch_seconds': sum(x['epoch_seconds'] for x in after)}
        summary['schemes'][scheme] = item
        values = [f"{100 * by_epoch[e]['validation_accuracy']:.2f}%" if e in by_epoch else 'pending' for e in (10, 20, 30)]
        ce = f"{by_epoch[30]['validation_loss']:.4f}" if 30 in by_epoch else 'pending'
        lines.append(f'| {scheme} | {" | ".join(values)} | {ce} | {state} |')
        for ax, metric in zip(axes, ('validation_accuracy', 'validation_loss')):
            scale = 100 if metric.endswith('accuracy') else 1
            ax.plot([x['epoch'] for x in rows], [scale * x[metric] for x in rows], color=color, label=scheme)
    for ax, label in zip(axes, ('Validation accuracy (%)', 'Validation cross-entropy')):
        ax.axvline(10, color='gray', linestyle='--', linewidth=1)
        ax.set(xlabel='Absolute epoch', ylabel=label, xlim=(1, 30))
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(output / 'learning_curves.png', dpi=180)
    plt.close(fig)
    lines += ['', 'The dashed line marks continuation from the saved epoch10 checkpoint.',
              'Schemes use different requested GPU/software stacks; small differences',
              'do not establish an amplification advantage. No official test was read.',
              'Best checkpoints in the continuation bundles cover epochs11–30 only.', '',
              '![Learning curves](learning_curves.png)', '']
    reporting.atomic_write_json(output / 'summary.json', summary)
    if args.verify_checkpoints:
        reporting.atomic_write_json(output / 'checkpoint_verification.json', checkpoint_checks)
    (output / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
