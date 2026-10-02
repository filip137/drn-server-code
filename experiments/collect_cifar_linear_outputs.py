"""Collect and validate the two analog linear-output BPTT pilot arms."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from experiments import reporting
from experiments.analyze_cifar_digital import finite_state


ARMS = ('legacy', 'ours')
QUEUE = '/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_metrics(rows, config):
    epochs = config['epochs']
    steps_per_epoch = math.ceil(45000 / config['batch_size'])
    require([row['epoch'] for row in rows] == list(range(1, epochs + 1)), 'Incomplete epoch coverage')
    for row in rows:
        require(row['steps'] == row['epoch'] * steps_per_epoch, 'Incomplete optimizer steps')
        require(row['train_examples'] == 45000 and row['validation_examples'] == 5000,
                'Incomplete data coverage')
        require(row['evaluation_split'] == 'validation', 'Unexpected evaluation split')
        require(all(math.isfinite(row[key]) for key in
                    ('train_loss', 'validation_loss', 'train_accuracy', 'validation_accuracy')),
                'Nonfinite metrics')
    return max(rows, key=lambda row: (row['validation_accuracy'], -row['validation_loss']))


def validate(root, arm, run_dir=None):
    run = Path(run_dir) if run_dir else root / 'cells' / arm
    errors = reporting.validate_run(run)
    require(not errors, f'Bundle validation: {errors}')
    manifest = json.loads((run / 'manifest.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    config = manifest['resolved_config']
    expected = json.loads((root / 'source/configs/cifar/linear_outputs_bptt_20261002' / f'{arm}.json').read_text())
    require(config == expected, 'Configuration mismatch')
    require(config['gradient_method'] == 'bptt' and config['block_output_activation'] == 'linear',
            'Unexpected model or gradient method')
    require(config['epochs'] == 10 and config['optimizer']['weight_decay'] == 0, 'Contract mismatch')
    require(config['precision'] == 'float32' and config['scheme'] == arm, 'Precision or scheme mismatch')
    require(not manifest['dataset']['official_test_read'] and not result['completion']['official_test_read'],
            'Official test access')
    require(result['completion']['criteria_met'] and result['completion']['full_epoch_budget'],
            'Incomplete run')
    rows = list(reporting.iter_metrics(run / 'metrics.jsonl'))
    best = check_metrics(rows, config)
    terminal = result['terminal_metrics']
    require(terminal['epochs_completed'] == 10 and terminal['steps'] == 14070, 'Terminal coverage mismatch')
    require(terminal['best_epoch'] == best['epoch'], 'Best checkpoint selection mismatch')
    require(terminal['final_validation_accuracy'] == rows[-1]['validation_accuracy'], 'Final metric mismatch')
    for name, epoch in [('initial_model', 0), ('best_model', best['epoch']), ('final_model', 10),
                        ('epoch_001', 1), ('epoch_005', 5), ('epoch_010', 10)]:
        saved = torch.load(run / f'checkpoints/{name}.pt', map_location='cpu', weights_only=False)
        require(saved['config'] == config and saved['epoch'] == epoch, f'{name}: config/epoch mismatch')
        require(finite_state(saved['model']) and finite_state(saved['optimizer']), f'{name}: nonfinite state')
        require(all(g['weight_decay'] == 0 for g in saved['optimizer']['param_groups']), f'{name}: weight decay')
        if epoch:
            require({int(v['step']) for v in saved['optimizer']['state'].values()} == {epoch * 1407},
                    f'{name}: Adam step mismatch')
        require(saved['scheduler']['last_epoch'] == epoch, f'{name}: scheduler mismatch')
        del saved
    return dict(arm=arm, validated=True, terminal_metrics=terminal, best_validation_loss=best['validation_loss'],
                epochs=10, steps=14070, official_test_read=False,
                validation_accuracy=[r['validation_accuracy'] for r in rows],
                validation_loss=[r['validation_loss'] for r in rows],
                train_accuracy=[r['train_accuracy'] for r in rows], train_loss=[r['train_loss'] for r in rows],
                peak_reserved_memory_gib=max(r['peak_memory_reserved_bytes'] for r in rows) / 2**30)


def plot_collected(root):
    summaries = [json.loads((root / 'analysis' / f'{arm}-collection.json').read_text())
                 for arm in ARMS if (root / 'analysis' / f'{arm}-collection.json').exists()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for summary in summaries:
        epochs = range(1, len(summary['validation_accuracy']) + 1)
        axes[0].plot(epochs, [100 * v for v in summary['validation_accuracy']], label=summary['arm'])
        axes[1].plot(epochs, summary['validation_loss'], label=summary['arm'])
    for ax, ylabel in zip(axes, ['Validation accuracy (%)', 'Validation cross-entropy']):
        ax.set(xlabel='Epoch', ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('Analog L8 linear block outputs — BPTT, seed 0, zero weight decay')
    fig.tight_layout()
    temporary = root / 'analysis' / f'.learning_curves.{os.getpid()}.jpg'
    fig.savefig(temporary, dpi=180)
    plt.close(fig)
    os.replace(temporary, root / 'analysis/learning_curves.jpg')
    reporting.atomic_write_json(root / 'analysis/comparison.json', {
        'coverage': [s['arm'] for s in summaries], 'complete': len(summaries) == len(ARMS),
        'review_status': 'awaiting_scientific_review',
        'measurements': {s['arm']: s['terminal_metrics'] for s in summaries}})


def collect(root, arm):
    job_id = os.environ.get('RUN_WATCH_JOB_ID', f'cifar-linear-bptt-{arm}-20261002')
    state = json.loads(subprocess.check_output([sys.executable, QUEUE, 'status', job_id], text=True))
    target = state['attempts'][-1]['target']
    placement = state['job']['spec']['placements'][target]
    output = root / 'cells' / arm
    if target != 'local':
        output.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync', '-a', '-e', 'ssh -F /home/filip/.ssh/config -o BatchMode=yes',
                        f"filip@{target}:{placement['run_dir']}/", str(output) + '/'], check=True)
    summary = dict(validate(root, arm), job_id=job_id, target=target)
    reporting.atomic_write_json(root / 'analysis' / f'{arm}-collection.json', summary)
    with (root / 'analysis/.collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plot_collected(root)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['collect', 'validate'])
    parser.add_argument('study_root', type=Path)
    parser.add_argument('arm', choices=ARMS)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    if args.action == 'collect':
        collect(args.study_root.resolve(), args.arm)
    else:
        print(json.dumps(validate(args.study_root.resolve(), args.arm, args.run_dir)), flush=True)
