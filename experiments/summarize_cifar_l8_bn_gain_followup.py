"""Summarize the bounded overnight CIFAR BN/gain follow-up and saved references."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments import reporting
from experiments.summarize_cifar_l8_e50 import ancestry
from experiments.summarize_cifar_l8_bn_ablation import training_rows


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def summarize(root):
    root = root.resolve()
    results = root.parent
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    original = results / 'cifar10-l8-analog-baseline-legacy-e30-e50-seed0-20260922-v1/cells'
    references = {
        'baseline': original / 'baseline_e30_e50',
        'proposed': results / 'cifar10-l8-analog-ours-e30-e50-seed0-20260922-v1/cells/ours_e30_e50',
        'legacy': original / 'legacy_e30_e50',
    }
    curves = {name: ancestry(path) for name, path in references.items()}
    gain_reference = results / 'cifar10-l8-bn-ablation-seed0-20260923-v1/cells/legacy_frozen_bn'
    curves['legacy frozen BN, original gain'] = training_rows(gain_reference, results)
    normalized_case = 'legacy_voltage_normalized_e10_e50'
    completion_case = 'legacy_voltage_normalized_e48_e50'
    if (root / 'cells' / completion_case / 'manifest.json').exists():
        normalized_case = completion_case
    cases = {
        normalized_case: 'normalized legacy',
        'legacy_frozen_bn_fixed_gain': 'legacy frozen BN, fixed gain',
        'legacy_frozen_bn_log_gain': 'legacy frozen BN, log gain',
    }
    coverage = []
    for case, label in cases.items():
        run = root / 'cells' / case
        status = read(run / 'status.json', {})
        manifest = read(run / 'manifest.json', {})
        config = manifest.get('resolved_config', {})
        pause = read(run / 'artifacts/pause.json', {})
        result = read(run / 'result.json', {})
        terminal = result.get('terminal_metrics', {})
        if manifest and 'continuation' in config:
            rows = ancestry(run)
        else:
            rows = training_rows(run, results)
        curves[label] = rows
        latest = rows[-1] if rows else {}
        own_rows = [r for r in rows if r['epoch'] > config.get('continuation', {}).get('from_epoch', 0)]
        if own_rows:
            assert all((r['train_examples'], r['validation_examples'], r['steps'])
                       == (45000, 5000, 1407*r['epoch']) for r in own_rows)
        if pause and latest:
            assert pause['completed_epoch'] == latest['epoch'], (case, pause, latest['epoch'])
        errors = reporting.validate_run(run) if status.get('state') in ('complete', 'failed') else []
        state = 'paused' if pause.get('planned_pause') else status.get('state', 'unstarted')
        audit = terminal.get('final_solver_audit_passed')
        replay_audit = None
        if state == 'paused' and label == 'normalized legacy':
            replay = root / 'diagnostics' / f"normalized_legacy_terminal_e{latest['epoch']}_replay"
            replay_result = read(replay / 'result.json', {})
            if replay_result:
                measured = replay_result['terminal_metrics']
                checkpoints = measured['checkpoints']
                assert len(checkpoints) == 1
                checkpoint = checkpoints[0]
                assert checkpoint['epoch'] == latest['epoch']
                assert checkpoint['sha256'] == pause['checkpoint_sha256']
                assert measured['all_checkpoint_guards_passed']
                replay_errors = reporting.validate_run(replay)
                replay_audit = {
                    'run': str(replay),
                    'result_sha256': reporting.sha256_file(replay / 'result.json'),
                    'checkpoint_sha256': checkpoint['sha256'],
                    'solver_audit_passed': checkpoint['solver_audit_passed'],
                    'validation_errors': replay_errors,
                }
        checkpoint_qualified = not errors and (
            audit is True or (replay_audit is not None
                              and replay_audit['solver_audit_passed'] is True
                              and not replay_audit['validation_errors']))
        qualified = state == 'complete' and audit is True and not errors
        coverage.append({
            'case': case, 'label': label, 'state': state, 'qualified_complete': qualified,
            'target_epoch': config.get('epochs', 50 if 'normalized' in case else 10),
            'starting_epoch': config.get('continuation', {}).get('from_epoch', 0),
            'continuation_parent': config.get('continuation'),
            'latest_epoch': latest.get('epoch'),
            'validation_accuracy': latest.get('validation_accuracy'),
            'validation_loss': latest.get('validation_loss'),
            'train_loss': latest.get('train_loss'),
            'input_gains': latest.get('input_gains'),
            'head_input_gain': latest.get('head_input_gain'),
            'measured_epoch_seconds': sum(r['epoch_seconds'] for r in own_rows),
            'final_solver_audit_passed': audit,
            'terminal_checkpoint_qualified': checkpoint_qualified,
            'separate_terminal_checkpoint_audit': replay_audit,
            'validation_errors': errors, 'pause': pause,
            'result_sha256': reporting.sha256_file(run / 'result.json') if result else None,
        })

    flat = []
    pairs = []
    fields = ('epoch', 'train_accuracy', 'train_loss', 'validation_accuracy',
              'validation_loss', 'epoch_seconds', 'head_input_gain')
    for case, rows in curves.items():
        for row in rows:
            item = {'case': case, **{key: row.get(key) for key in fields}}
            item.update({f'gain_block{i+1}': value for i, value in enumerate(row.get('input_gains', []))})
            flat.append(item)
    legacy_by_epoch = {r['epoch']: r for r in curves['legacy']}
    for row in curves['normalized legacy']:
        reference = legacy_by_epoch[row['epoch']]
        pairs.append({'epoch': row['epoch'],
                      'validation_accuracy_delta_pp': 100*(row['validation_accuracy']-reference['validation_accuracy']),
                      'validation_ce_delta': row['validation_loss']-reference['validation_loss'],
                      'train_ce_delta': row['train_loss']-reference['train_loss']})
    for name, rows in [('training.csv', flat), ('normalized_legacy_paired.csv', pairs)]:
        if rows:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            with (out / name).open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader(); writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plot_labels = {row['label']: f"{row['label']} (paused {row['latest_epoch']}/{row['target_epoch']})"
                   for row in coverage if row['state'] == 'paused'}
    for label in ('baseline', 'proposed', 'legacy', 'normalized legacy'):
        rows = curves[label]
        for ax, metric, scale in [(axes[0, 0], 'validation_accuracy', 100),
                                  (axes[0, 1], 'validation_loss', 1)]:
            ax.plot([r['epoch'] for r in rows], [scale*r[metric] for r in rows],
                    label=plot_labels.get(label, label))
    gain_labels = [label for label in curves if 'frozen BN' in label]
    colors = ['tab:gray', 'tab:blue', 'tab:orange']
    for label, color in zip(gain_labels, colors):
        rows = curves[label]
        axes[1, 0].plot([r['epoch'] for r in rows],
                        [100*r['validation_accuracy'] for r in rows], color=color, label=label)
        for index, style in enumerate(('-', '--', ':')):
            values = [r for r in rows if 'input_gains' in r]
            axes[1, 1].plot([r['epoch'] for r in values],
                            [r['input_gains'][index] for r in values], style, color=color,
                            label=f"{label.removeprefix('legacy frozen BN, ')} / block{index+1}")
    for ax, ylabel in zip(axes.flat, ('Validation accuracy (%)', 'Validation cross-entropy',
                                     'Validation accuracy (%)', 'Convolution-block input gain')):
        ax.set(xlabel='Absolute epoch', ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    for ax in axes[0]:
        ax.axvline(10, color='gray', linestyle=':', alpha=.6)
        ax.axvline(30, color='gray', linestyle=':', alpha=.6)
    fig.savefig(out / 'training_and_gains.png', dpi=170)
    plt.close(fig)

    summary = {'official_test_read': False, 'evidence_class': 'exploratory',
               'training_coverage': coverage,
               'all_training_qualified_complete': all(r['qualified_complete'] for r in coverage)}
    reporting.atomic_write_json(out / 'training_summary.json', summary)
    def fmt(value, percent=False):
        return 'pending' if value is None else f'{value*100:.2f}%' if percent else f'{value:.6f}'
    lines = ['# CIFAR L8 overnight training follow-up', '',
             'Exploratory seed 0, same 45,000/5,000 validation split, cross-entropy, crop/flip augmentation,',
             'batch 32 and selected legacy Adam rates. The normalized continuation retains trainable',
             'affine BN. The gain pair freezes affine BN while updating running statistics.', '',
             '| Case | Completed epoch / target | Validation accuracy | Validation CE | State | Available checkpoint audit |',
             '|---|---:|---:|---:|---|---|']
    for row in coverage:
        separate = row['separate_terminal_checkpoint_audit']
        passed = separate['solver_audit_passed'] if separate else row['final_solver_audit_passed']
        audit_label = ('pending' if passed is None else
                       ('passed' if passed else 'FAILED') + (' (read-only replay)' if separate else ' (native)'))
        lines.append(f"| {row['label']} | {row['latest_epoch']} / {row['target_epoch']} | "
                     f"{fmt(row['validation_accuracy'], True)} | {fmt(row['validation_loss'])} | "
                     f"{row['state']} | {audit_label} |")
    lines += ['', 'The original completed epoch-50 references are baseline 89.48%, proposed 89.92%, legacy 88.72%.',
              'Compare a partial normalized continuation only with references at the same epoch.',
              'A separate replay can qualify its saved checkpoint without completing the declared epoch target.',
              'The original frozen-BN softplus-gain reference reaches 73.02% at 10.',
              'Fixed/log gains differ in both parameterization and relative adaptation rate.',
              'A result with a failed solver audit is not a qualified equilibrium comparison.',
              'BN recalibration and Adam-step counterfactuals are reported separately from training.',
              '', '![Training and gains](training_and_gains.png)', '']
    (out / 'training_report.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
