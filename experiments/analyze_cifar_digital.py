"""Validate a completed digital CIFAR study and plot its learning curves."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from experiments import reporting
from experiments.train_cifar_digital_l12 import DigitalL12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_state(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_state(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def analyze(root):
    run_dirs = sorted(root.glob('production-seed*'))
    require(len(run_dirs) == 1, 'Expected exactly one production bundle')
    run = run_dirs[0]
    for bundle in (root / 'smoke', run):
        errors = reporting.validate_run(bundle)
        require(not errors, f'{bundle}: {errors}')
        require(json.loads((bundle / 'status.json').read_text())['state'] == 'complete',
                f'Incomplete bundle: {bundle}')
    manifest = json.loads((run / 'manifest.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    config = manifest['resolved_config']
    rows = list(reporting.iter_metrics(run / 'metrics.jsonl'))
    split = config['dataset']['evaluation_split']
    prefix = 'validation' if split == 'validation' else 'test'
    evaluation_size = config['dataset'][f'{prefix}_size']
    epochs = config['epochs']
    steps_per_epoch = math.ceil(config['dataset']['train_size'] / config['batch_size'])
    require([r['epoch'] for r in rows] == list(range(1, epochs + 1)), 'Epoch coverage mismatch')
    for epoch, row in enumerate(rows, 1):
        require(row['train_examples'] == config['dataset']['train_size'], 'Training coverage mismatch')
        require(row[f'{prefix}_examples'] == evaluation_size, 'Evaluation coverage mismatch')
        require(row['optimizer_steps'] == epoch * steps_per_epoch, 'Optimizer step coverage mismatch')
        require(row['evaluation_split'] == split, 'Evaluation split mismatch')
    require(result['completion']['criteria_met'], 'Completion criteria unmet')
    require(result['terminal_metrics']['optimizer_steps'] == epochs * steps_per_epoch,
            'Terminal optimizer steps mismatch')
    best = max(rows, key=lambda r: (r[f'{prefix}_accuracy'], -r[f'{prefix}_loss']))
    require(result['terminal_metrics']['best_epoch'] == best['epoch'], 'Best checkpoint rule mismatch')
    require(result['terminal_metrics'][f'best_{prefix}_accuracy'] == best[f'{prefix}_accuracy'],
            'Best accuracy mismatch')
    require(result['terminal_metrics'][f'final_{prefix}_accuracy'] == rows[-1][f'{prefix}_accuracy'],
            'Final accuracy mismatch')
    if split == 'validation':
        require(manifest['dataset']['official_test_read'] is False and
                result['completion']['official_test_read'] is False, 'Unexpected official-test access')
        indices = json.loads((run / 'artifacts/split.json').read_text())
        require(len(indices['train']) == config['dataset']['train_size'] and
                len(indices['validation']) == evaluation_size, 'Split size mismatch')
        require(set(indices['train']).isdisjoint(indices['validation']), 'Overlapping split')
        require(len(set(indices['train']) | set(indices['validation'])) == 50000,
                'Incomplete split partition')
    model = DigitalL12(config)
    checkpoint_epochs = {}
    for name, epoch in [('initial', 0), ('best', best['epoch']), ('final', epochs)]:
        checkpoint = torch.load(run / f'checkpoints/{name}_model.pt', map_location='cpu', weights_only=False)
        require(checkpoint['config'] == config, f'{name}: configuration mismatch')
        require(checkpoint['epoch'] == epoch, f'{name}: epoch mismatch')
        require(finite_state(checkpoint['model']) and finite_state(checkpoint['optimizer']),
                f'{name}: nonfinite checkpoint')
        model.load_state_dict(checkpoint['model'], strict=True)
        for group in checkpoint['optimizer']['param_groups']:
            require(group['weight_decay'] in (0, config['optimizer']['weight_decay']),
                    f'{name}: unexpected optimizer weight decay')
        checkpoint_epochs[name] = epoch
    analysis = root / 'analysis'
    analysis.mkdir(exist_ok=True)
    summary = {
        'study_id': config['study_id'], 'evidence_class': config['evidence_class'],
        'evaluation_split': split, 'official_test_read': manifest['dataset']['official_test_read'],
        'terminal_metrics': result['terminal_metrics'],
        'best_epoch_metrics': best,
        'peak_reserved_memory_gib': max(r['peak_memory_reserved_bytes'] for r in rows) / 2**30,
        'weight_decay': config['optimizer']['weight_decay'],
        'block_output_activation': config.get('block_output_activation', 'relu'),
        'verification': {'bundle_errors': [], 'epochs': epochs, 'steps': epochs * steps_per_epoch,
                         'checkpoint_epochs': checkpoint_epochs, 'finite_checkpoints': True},
    }
    reporting.atomic_write_json(analysis / 'summary.json', summary)
    epoch_values = [r['epoch'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, label in [('train', 'Augmented training'), (prefix, prefix.capitalize())]:
        axes[0].plot(epoch_values, [100 * r[f'{key}_accuracy'] for r in rows], label=label)
        axes[1].plot(epoch_values, [r[f'{key}_loss'] for r in rows], label=label)
    axes[0].scatter([best['epoch']], [100 * best[f'{prefix}_accuracy']], color='black', s=24,
                    label=f"Best: {100 * best[f'{prefix}_accuracy']:.2f}% (epoch {best['epoch']})")
    for axis, ylabel in zip(axes, ['Accuracy (%)', 'Cross-entropy']):
        axis.set(xlabel='Epoch', ylabel=ylabel)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    fig.suptitle(f"Digital L8: {config.get('block_output_activation', 'relu')} block outputs, "
                 f"weight decay {config['optimizer']['weight_decay']:g}, seed {config['seed']}")
    fig.tight_layout()
    fig.savefig(analysis / 'learning_curves.jpg', dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study_root', type=Path)
    analyze(parser.parse_args().study_root)
