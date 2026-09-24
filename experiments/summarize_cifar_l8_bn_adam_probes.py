"""Aggregate paired read-only CIFAR BN and saved-Adam diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import reporting


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write_csv(path, rows):
    if rows:
        with path.open('w') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def summarize(root):
    root = root.resolve(); out = root / 'analysis'; out.mkdir(exist_ok=True)
    coverage = []
    bn_rows, moment_rows, adam_rows, adam_summary = [], [], [], []
    expected = {
        'adam_e10': {f'{scheme}_e10' for scheme in ('baseline', 'ours', 'legacy')},
        'adam_e30': {f'{scheme}_e30' for scheme in ('baseline', 'ours', 'legacy')},
        'adam_e50': {f'{scheme}_e50' for scheme in ('baseline', 'ours', 'legacy')},
        'bn_recalibration': {f'{scheme}_e{epoch}' for scheme in ('baseline', 'ours', 'legacy')
                             for epoch in (10, 50)} | {'legacy_voltage_normalized_e10'},
    }
    for name, ids in expected.items():
        run = root / 'diagnostics' / name
        status = load(run / 'status.json', {})
        included = []
        for checkpoint in sorted(ids):
            path = run / 'artifacts' / f'{checkpoint}.json'
            value = load(path)
            if value is None:
                continue
            assert value['source'] == checkpoint and value['original_checkpoint_unchanged']
            included.append(checkpoint)
            if name == 'bn_recalibration':
                assert len(value['fit_indices']) == 4096 and len(value['validation_indices']) == 5000
                assert not set(value['fit_indices']) & set(value['validation_indices'])
                assert value['fitting_labels_used'] is False
                bn_rows.append({'checkpoint': checkpoint,
                    'saved_accuracy': value['before']['accuracy'], 'recalibrated_accuracy': value['after']['accuracy'],
                    'accuracy_delta_pp': 100*value['accuracy_delta'], 'saved_ce': value['before']['loss'],
                    'recalibrated_ce': value['after']['loss'], 'ce_delta': value['loss_delta'],
                    'checkpoint_sha256': value['sha256'], 'artifact_sha256': reporting.sha256_file(path)})
                for block in value['statistics']:
                    # All seven checkpoints retain physical BN epsilon1e-5;
                    # normalized legacy expresses its inputs in baseline units.
                    eps = 1e-5
                    moment_rows.append({'checkpoint': checkpoint, 'block': block['block'],
                        'old_variance_median': statistics.median(block['old_variance']),
                        'new_variance_median': statistics.median(block['new_variance']),
                        'old_epsilon_share_median': statistics.median(eps/(v+eps) for v in block['old_variance']),
                        'new_epsilon_share_median': statistics.median(eps/(v+eps) for v in block['new_variance'])})
            else:
                assert len(value['gradient_indices']) == len(value['holdout_indices']) == 128
                assert not set(value['gradient_indices']) & set(value['holdout_indices'])
                assert len(value['probes']) == 16
                by_batch = {}
                for row in value['probes']:
                    by_batch.setdefault(row['batch'], {})[row['multiplier']] = row
                assert len(by_batch) == 4
                for batch, by_step in by_batch.items():
                    assert set(by_step) == {0., .5, 1., 2.}
                    for multiplier, row in sorted(by_step.items()):
                        for probe in ('same_batch', 'holdout'):
                            base = by_step[0][probe]
                            adam_rows.append({'checkpoint': checkpoint, 'batch': batch, 'multiplier': multiplier,
                                'probe': probe, 'ce': row[probe]['loss'], 'ce_delta': row[probe]['loss']-base['loss'],
                                'accuracy': row[probe]['accuracy'],
                                'accuracy_delta_pp': 100*(row[probe]['accuracy']-base['accuracy']),
                                'logit_relative_l2': row[probe]['logit_relative_l2'],
                                'gradient_dot_update': row['update']['gradient_dot_update'],
                                'gradient_update_cosine': row['update']['gradient_update_cosine'],
                                'relative_update_l2': row['update']['relative_update_l2']})
                for probe in ('same_batch', 'holdout'):
                    for multiplier in (0., .5, 1., 2.):
                        rows = [r for r in adam_rows if r['checkpoint'] == checkpoint
                                and r['probe'] == probe and r['multiplier'] == multiplier]
                        adam_summary.append({'checkpoint': checkpoint, 'probe': probe, 'multiplier': multiplier,
                            'mean_ce_delta': statistics.mean(r['ce_delta'] for r in rows),
                            'minimum_ce_delta': min(r['ce_delta'] for r in rows),
                            'maximum_ce_delta': max(r['ce_delta'] for r in rows),
                            'directions_improving_ce': sum(r['ce_delta'] < 0 for r in rows),
                            'mean_logit_relative_l2': statistics.mean(r['logit_relative_l2'] for r in rows)})
        errors = reporting.validate_run(run) if status.get('state') in ('complete', 'failed') else []
        coverage.append({'bundle': name, 'state': status.get('state', 'not collected'),
                         'completed': included, 'missing': sorted(ids-set(included)), 'validation_errors': errors})
    for name, rows in [('bn_recalibration.csv', bn_rows), ('bn_moments.csv', moment_rows),
                       ('adam_steps.csv', adam_rows), ('adam_steps_summary.csv', adam_summary)]:
        write_csv(out / name, rows)
    summary = {'coverage': coverage, 'bn': bn_rows, 'adam': adam_summary,
               'official_test_read': False,
               'complete': all(r['state'] == 'complete' and not r['missing'] and not r['validation_errors']
                               for r in coverage)}
    reporting.atomic_write_json(out / 'diagnostic_summary.json', summary)
    lines = ['# CIFAR L8 read-only BN and Adam-step diagnostics', '',
        'All original checkpoint bytes are preserved. BN statistics are fit sequentially on4096',
        'training examples without augmentation; evaluation uses the same5000 validation examples.',
        'The original saved-model validation scores remain primary.', '',
        '| Checkpoint | Saved BN accuracy | Recalibrated accuracy | Difference | CE change |',
        '|---|---:|---:|---:|---:|']
    for row in bn_rows:
        lines.append(f"| {row['checkpoint']} | {100*row['saved_accuracy']:.2f}% | "
            f"{100*row['recalibrated_accuracy']:.2f}% | {row['accuracy_delta_pp']:+.2f}pp | {row['ce_delta']:+.5f} |")
    lines += ['', 'Adam directions use each checkpoint’s own saved moments and scheduled rates.',
              'Gradients use training-mode BN; paired loss/logit evaluations keep original saved BN',
              'buffers fixed. A loss increase can reflect this mode difference as well as update size.',
              'Four directions and one shared holdout cohort are diagnostics, not independent training seeds.', '',
              '| Checkpoint | Probe | Step multiplier | Mean CE change | Directions improving CE /4 | Mean relative logit change |',
              '|---|---|---:|---:|---:|---:|']
    for row in adam_summary:
        lines.append(f"| {row['checkpoint']} | {row['probe']} | {row['multiplier']:g} | "
            f"{row['mean_ce_delta']:+.6f} | {row['directions_improving_ce']} | {row['mean_logit_relative_l2']:.6f} |")
    lines += ['', 'Expected coverage and validation state:', '']
    for row in coverage:
        lines.append(f"- {row['bundle']}: {row['state']}; {len(row['completed'])} completed; missing {row['missing']}; validation errors {row['validation_errors']}.")
    if adam_summary:
        epochs = sorted({int(r['checkpoint'].rsplit('_e', 1)[1]) for r in adam_summary})
        fig, axes = plt.subplots(len(epochs), 2, figsize=(10, 3.6*len(epochs)), squeeze=False,
                                 constrained_layout=True)
        for pair, epoch in zip(axes, epochs):
            for ax, probe in zip(pair, ('same_batch', 'holdout')):
                for checkpoint in sorted({r['checkpoint'] for r in adam_summary
                                          if r['checkpoint'].endswith(f'_e{epoch}')}):
                    rows = [r for r in adam_summary if r['checkpoint'] == checkpoint and r['probe'] == probe]
                    ax.plot([r['multiplier'] for r in rows], [r['mean_ce_delta'] for r in rows],
                            marker='o', label=checkpoint.rsplit('_e', 1)[0].replace('ours', 'proposed'))
                    ax.fill_between([r['multiplier'] for r in rows],
                                    [r['minimum_ce_delta'] for r in rows],
                                    [r['maximum_ce_delta'] for r in rows], alpha=.1)
                ax.axhline(0, color='gray', linewidth=1)
                ax.set(xlabel='Projected Adam-step multiplier', ylabel='Mean change in cross-entropy',
                       title=f"Epoch {epoch}: " + ('gradient batch' if probe == 'same_batch' else 'separate training cohort'))
                ax.grid(alpha=.2); ax.legend()
        fig.savefig(out / 'adam_loss_probes.png', dpi=170); plt.close(fig)
        lines += ['', 'Curves show means; shaded bands show the minimum and maximum across four directions, not confidence intervals.',
                  '', '![Adam loss probes](adam_loss_probes.png)']
    (out / 'diagnostic_report.md').write_text('\n'.join(lines)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
