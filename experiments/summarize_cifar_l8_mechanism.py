"""Collect the CIFAR mechanism controls without merging unlike replay cohorts."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments import reporting


CASES = ['baseline_reference', 'proposed_reference', 'legacy_reference',
         'legacy_baseline_lr', 'legacy_voltage_normalized']
COLORS = ['#377eb8', '#4daf4a', '#e41a1c', '#984ea3', '#ff7f00']


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def metrics(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def validate_cell(run, expected, root):
    errors = reporting.validate_run(run)
    if errors:
        raise ValueError({str(run): errors})
    manifest = read(run / 'manifest.json')
    assert manifest['resolved_config'] == expected
    expected_target = 'loulou' if run.name in CASES[3:] else 'fifi'
    assert manifest['runtime']['target'] == expected_target
    assert manifest['environment']['device'] == 'cuda'
    assert 'RTX 5090' in manifest['environment']['gpu']
    assert manifest['execution']['require_gpu_resident']
    assert not manifest['execution']['save_tensors_on_cpu']
    assert manifest['dataset']['official_test_read'] is False
    assets = read(run / 'artifacts/assets.json')
    assert assets['initial_sha256'] == reporting.sha256_file(root / 'inputs/initial_model.pt')
    assert assets['iterations'] == [6, 6, 4]
    prepared_split = read(root / ('prepare-' + run.name) / 'artifacts/split.json')
    assert prepared_split == read(root / 'inputs/split.json')
    rows = metrics(run / 'metrics.jsonl')
    assert [row['epoch'] for row in rows] == list(range(1, 11))
    assert all((row['train_examples'], row['validation_examples'], row['steps']) ==
               (45000, 5000, 1407 * row['epoch']) for row in rows)
    result = read(run / 'result.json')['terminal_metrics']
    assert result['epochs_completed'] == result['epochs_trained'] == 10
    assert result['starting_epoch'] == 0
    return {'canonical_bundle_valid': True,
            'solver_audit_passed': result['final_solver_audit_passed']}


def verify_checkpoints(run, root):
    """Verify real terminal tensors and their relation to the shared start."""
    import torch
    torch.set_num_threads(1)
    shared = torch.load(root / 'inputs/initial_model.pt', map_location='cpu', weights_only=False)
    initial = torch.load(run / 'checkpoints/initial_model.pt', map_location='cpu', weights_only=False)
    assert initial['epoch'] == 0 and not initial['optimizer']['state']
    for category, values in shared['model'].items():
        assert values.keys() == initial['model'][category].keys()
        for name, value in values.items():
            assert torch.equal(value, initial['model'][category][name]), name
    del shared, initial
    final = torch.load(run / 'checkpoints/final_model.pt', map_location='cpu', weights_only=False)
    config = read(run / 'manifest.json')['resolved_config']
    assert final['epoch'] == final['scheduler']['last_epoch'] == 10
    assert final['config'] == config
    assert len(final['optimizer']['state']) == 19
    assert {int(state['step']) for state in final['optimizer']['state'].values()} == {14070}
    for category in final['model'].values():
        assert all(torch.isfinite(value).all() for value in category.values())
    for state in final['optimizer']['state'].values():
        assert all(torch.isfinite(value).all() for value in state.values() if isinstance(value, torch.Tensor))
    counters = [int(value) for name, value in final['model']['module'].items() if name.endswith('num_batches_tracked')]
    assert len(counters) == 3 and set(counters) == {14070}
    assert len(final['model']['conductances']) == 9
    for value in final['model']['conductances'].values():
        assert float(value.min()) >= config['weight_min'] and float(value.max()) <= config['weight_max']
    base = final['scheduler']['base_lrs']
    schedule = config['scheduler']
    def expected_rates(epoch):
        scale = schedule['final_lr_ratio'] + (1 - schedule['final_lr_ratio']) * .5 * (
            1 + math.cos(math.pi * epoch / schedule['horizon_epochs']))
        return [rate * scale for rate in base]
    pairs = [(row['learning_rates'], expected_rates(row['epoch'] - 1)) for row in metrics(run / 'metrics.jsonl')]
    pairs.append(([group['lr'] for group in final['optimizer']['param_groups']], expected_rates(10)))
    for actual, expected in pairs:
        assert len(actual) == len(expected) == 19
        assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) for a, b in zip(actual, expected))
    return {'initializer_tensors_exact': True, 'optimizer_steps': 14070,
            'bn_steps': 14070, 'all_tensors_finite': True,
            'all_conductances_bounded': True, 'cosine_schedule_verified': True,
            'final_checkpoint_sha256': reporting.sha256_file(run / 'checkpoints/final_model.pt')}


def collect_replays(root):
    rows, completed, validation = [], {}, {}
    expected = {row['id']: row for row in read(root / 'inputs/checkpoints.json')}
    cohort = read(root / 'inputs/split.json')['train'][:128]
    for directory in sorted(root.glob('replay-*')):
        state = read(directory / 'status.json', {}).get('state', 'missing')
        if state != 'complete':
            validation[directory.name] = {'state': state}
            continue
        errors = reporting.validate_run(directory)
        if errors:
            raise ValueError({directory.name: errors})
        manifest = read(directory / 'manifest.json')
        assert manifest['cohort_indices'] == cohort
        assert manifest['dataset']['official_test_read'] is False
        assert manifest['runtime']['target'] == 'fifi'
        assert 'RTX 5090' in manifest['environment']['gpu']
        current = metrics(directory / 'metrics.jsonl')
        guards = [row for row in current if row['stage'] == 'checkpoint_guard']
        for guard in guards:
            name = guard['source']
            assert name not in completed, ('Duplicate checkpoint replay', name)
            assert guard['checkpoint_sha256'] == expected[name]['sha256']
            assert guard['parameters_unchanged'] and guard['bn_buffers_restored']
            assert guard['gradient_gate_examples'] == 8
            assert guard['gradient_gate_microbatch_size'] == 2
            measurements = [r for r in current if r.get('source') == name and r['stage'] == 'checkpoint']
            assert {(r['mode'], r['batch']) for r in measurements} == {
                (mode, batch) for mode in ('training', 'evaluation') for batch in range(4)}
            assert all(r['examples'] == 32 for r in measurements)
            long_states = [r for r in current if r.get('source') == name and r['stage'] == 'long_state']
            assert {(r['schedule'], r['batch']) for r in long_states} == {
                (schedule, batch) for schedule in ('reference', 'sentinel') for batch in range(4)}
            completed[name] = guard
        rows.extend(current)
        validation[directory.name] = {'state': state, 'bundle_valid': True,
                                      'checkpoint_count': len(guards)}
    return rows, completed, validation


def plot_replays(rows, output):
    bn_rows, state_rows, gradient_rows = [], [], []
    for row in rows:
        if row['stage'] not in ('checkpoint', 'long_state'):
            continue
        common = {key: row.get(key) for key in ('source', 'epoch', 'mode', 'batch')}
        common['schedule'] = row.get('schedule', 'operational')
        for state in row['states']:
            item = dict(common, block=state['block'], layer=state['layer'], phase=state['mode'],
                        clamped_fraction=state['clamped_fraction'],
                        projected_kkt_relative_l2=state['projected_kkt_relative_l2'],
                        normalized_downstream_loading_median=state['normalized_downstream_loading_median'])
            for scale in ('voltage', 'normalized_voltage'):
                item.update({scale + '_' + key: value for key, value in state[scale].items()})
            state_rows.append(item)
        for bn in row.get('bn', []):
            bn_rows.append(dict(common, block=bn['block'], phase=bn['phase'],
                median_batch_variance=float(np.median(bn['batch_variance'])),
                median_used_variance=float(np.median(bn['used_variance'])),
                epsilon=bn['epsilon'], epsilon_share_median=bn['epsilon_share_median']))
        for gradient in row.get('gradients', []):
            gradient_rows.append(dict(common, **gradient))
    write_csv(output / 'batchnorm.csv', bn_rows)
    write_csv(output / 'voltages_kkt.csv', state_rows)
    write_csv(output / 'gradients_adam.csv', gradient_rows)
    if bn_rows:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)
        for i, (mode, phase) in enumerate([('training', 'tracked'), ('evaluation', 'evaluation')]):
            for block in range(3):
                ax = axes[i, block]
                for scheme, color in zip(['baseline', 'ours', 'legacy'], COLORS):
                    selected = [r for r in bn_rows if r['source'].startswith(scheme + '_')
                                and r['mode'] == mode and r['phase'] == phase and r['block'] == block]
                    epochs = sorted({r['epoch'] for r in selected})
                    shares = [np.median([r['epsilon_share_median'] for r in selected if r['epoch'] == e]) for e in epochs]
                    ax.plot(epochs, shares, 'o-', color=color, label=scheme)
                ax.set(title=f'{mode}, block {block + 1}', xlabel='Saved checkpoint epoch', ylim=(0, 1))
                ax.grid(alpha=.2)
            axes[i, 0].set_ylabel('Median epsilon / (variance + epsilon)')
        axes[0, 0].legend()
        fig.savefig(output / 'batchnorm_epsilon.png', dpi=180); plt.close(fig)
    return bn_rows, state_rows, gradient_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--verify-checkpoints', action='store_true')
    args = parser.parse_args(); root = args.root.resolve(); output = root / 'analysis'
    output.mkdir(exist_ok=True)
    summary = {'complete': False, 'scientifically_qualified': False, 'official_test_read': False, 'cases': {}}
    lines = ['# CIFAR L8 mechanism controls', '',
             'Exploratory seed0 validation. References use Fifi RTX5090; the two legacy controls use Loulou RTX5090.',
             'Fresh10epoch runs, batch32, Adam, trainable affine BN and boundary gains,',
             'crop/flip augmentation, cross-entropy, and the original50epoch cosine horizon.', '',
             '| Case | Epochs | Train accuracy | Train CE | Validation accuracy | Validation CE | State |',
             '|---|---:|---:|---:|---:|---:|---|']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    training = []
    for case, color in zip(CASES, COLORS):
        run = root / 'cells' / case
        state = read(run / 'status.json', {}).get('state', 'pending')
        rows = metrics(run / 'metrics.jsonl')
        entry = {'state': state, 'epochs_recorded': len(rows)}
        if state == 'complete':
            config_dir = 'configs-loulou' if case in CASES[3:] else 'configs'
            entry.update(validate_cell(run, read(root / config_dir / (case + '.json')), root))
            if args.verify_checkpoints:
                entry['checkpoint_verification'] = verify_checkpoints(run, root)
        summary['cases'][case] = entry
        if not rows:
            lines.append(f'| {case} | 0 | — | — | — | — | {state} |')
            continue
        latest = rows[-1]
        entry.update({key: latest[key] for key in ['epoch', 'train_accuracy', 'train_loss',
                                                  'validation_accuracy', 'validation_loss']})
        lines.append(f"| {case} | {len(rows)} | {100*latest['train_accuracy']:.2f}% | {latest['train_loss']:.4f} | "
                     f"{100*latest['validation_accuracy']:.2f}% | {latest['validation_loss']:.4f} | {state} |")
        for row in rows:
            training.append(dict(case=case, **{key: row[key] for key in ['epoch', 'train_accuracy', 'train_loss',
                                                                       'validation_accuracy', 'validation_loss', 'epoch_seconds']}))
        for ax, metric in zip(axes.flat, ['train_accuracy', 'validation_accuracy', 'train_loss', 'validation_loss']):
            scale = 100 if metric.endswith('accuracy') else 1
            ax.plot([row['epoch'] for row in rows], [scale * row[metric] for row in rows],
                    color=color, label=case, linestyle='--' if case in CASES[3:] else '-')
    if training:
        for ax, label in zip(axes.flat, ['Training accuracy (%)', 'Validation accuracy (%)',
                                        'Training cross-entropy', 'Validation cross-entropy']):
            ax.set(xlabel='Epoch', ylabel=label, xlim=(1, 10)); ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=7)
        fig.savefig(output / 'learning_curves.png', dpi=180)
        lines += ['', '![Control learning curves](learning_curves.png)']
    plt.close(fig); write_csv(output / 'training.csv', training)
    eq = root / 'equivalence-200'
    summary['equivalence'] = {'state': read(eq / 'status.json', {}).get('state', 'pending')}
    if summary['equivalence']['state'] == 'complete':
        assert not reporting.validate_run(eq)
        measured = read(eq / 'result.json')['terminal_metrics']
        assert measured['steps'] == 200 and measured['passed'] and measured['maximum_relative_error'] <= 1e-5
        summary['equivalence'].update(measured)
    rows, guards, validation = collect_replays(root)
    summary['checkpoint_replays'] = guards
    summary['diagnostic_validation'] = validation
    bn, states, gradients = plot_replays(rows, output)
    matched = [row for row in rows if row['stage'] == 'matched_weights']
    write_csv(output / 'matched_weights.csv', [{key: value for key, value in row.items() if key != 'gradient_comparison'} |
               {'maximum_gradient_relative_error': row['gradient_comparison']['maximum_relative_error'],
                'worst_gradient_tensor': row['gradient_comparison']['worst_tensor']} for row in matched])
    expected_ids = {row['id'] for row in read(root / 'inputs/checkpoints.json')}
    summary['complete'] = (all(row['state'] == 'complete' for row in summary['cases'].values()) and
                           summary['equivalence']['state'] == 'complete' and set(guards) == expected_ids)
    observed_audits = {name: row['solver_audit_passed']
                       for name, row in list(summary['cases'].items()) + list(guards.items())
                       if 'solver_audit_passed' in row}
    summary['observed_solver_audits'] = observed_audits
    summary['all_observed_solver_audits_passed'] = bool(observed_audits) and all(observed_audits.values())
    summary['scientifically_qualified'] = summary['complete'] and summary['all_observed_solver_audits_passed']
    summary['launchers'] = {}
    for target in ['fifi', 'loulou']:
        launcher = read(root / 'launcher' / (target + '_parallel.json'), {})
        summary['launchers'][target] = {key: launcher.get(key) for key in ['state', 'spent_seconds', 'budget_seconds', 'exit_code']}
    lines += ['', f"Coverage complete: {summary['complete']}. All {len(observed_audits)} completed solver audits passed: {summary['all_observed_solver_audits_passed']}.",
              f"Checkpoint replays: {len(guards)}/12. Equivalence gate: {summary['equivalence']['state']}.", '',
              'Replay CSVs keep training/evaluation, free/tracked phases, and iteration schedules separate.',
              'Operational voltages and gradients use128 fixed examples in four batches32.',
              'Long-unroll gradient audits use the separately labelled eight-example cohort in microbatches2.',
              'BN plots show the median of four batchwise channel medians; this is a diagnostic statistic.',
              'Adam proposals use saved optimizer moments and the next scheduled learning rate; they are not executed updates.', '',
              'The LR intervention holds legacy voltage scale fixed. The normalization intervention holds its original LR fixed.',
              'Compare both with legacy_reference. A single seed and ten epochs cannot establish the cause of the final50epoch gap.',
              'The user requested parallel placement: reference cases run on Fifi, controls on Loulou with existing GPU jobs.',
              'Both use RTX5090/PyTorch2.11/cu128/cuDNN91900, but host placement remains an exploratory comparison limitation.',
              'Official test data is never read. Pending or failed coverage is not silently excluded.', '']
    if bn:
        lines += ['![BN epsilon share](batchnorm_epsilon.png)', '']
    reporting.atomic_write_json(output / 'summary.json', summary)
    (output / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))
    if not args.allow_partial and not summary['complete']:
        raise RuntimeError('Study coverage is incomplete; use --allow-partial for an interim report')


if __name__ == '__main__':
    main()
