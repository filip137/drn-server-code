"""Compare existing matched CIFAR gradients, parameter scales and Adam proposals.

This aggregates saved read-only replay measurements; it performs no model replay
or optimizer step and leaves checkpoints unchanged.
"""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def spread(values):
    if not values:
        return {'median': None, 'minimum': None, 'maximum': None}
    return {'median': statistics.median(values), 'minimum': min(values),
            'maximum': max(values)}


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root.resolve()
    out = root / 'analysis'
    source = out / 'gradients_adam.csv'
    with source.open() as handle:
        rows = [r for r in csv.DictReader(handle)
                if r['mode'] == 'training' and r['schedule'] == 'operational']
    lookup = {(r['source'], r['parameter'], int(r['batch'])): r for r in rows}
    assert len(lookup) == len(rows)
    names = sorted({r['parameter'] for r in rows})
    conv = [p for p in names if p.startswith('blocks.')]
    epochs = [0, 10, 30, 50]
    metrics = ['gradient_rms', 'gradient_zero_fraction', 'parameter_l2',
               'next_step_lr', 'adam_proposal_l2', 'projected_adam_proposal_l2',
               'relative_projected_adam_proposal_l2']
    layer_rows, paired_rows, summaries = [], [], []
    for epoch in epochs:
        for scheme in ['baseline', 'ours', 'legacy']:
            for name in names:
                batches = [lookup[(f'{scheme}_e{epoch}', name, b)] for b in range(4)]
                row = {'scheme': scheme, 'epoch': epoch, 'parameter': name,
                       'replay_batches': 4}
                for metric in metrics:
                    for stat, value in spread([float(b[metric]) for b in batches if b[metric] != '']).items():
                        row[f'{metric}_{stat}'] = value
                retention = [float(b['projected_adam_proposal_l2']) / float(b['adam_proposal_l2'])
                             if float(b['adam_proposal_l2']) else 0. for b in batches]
                for stat, value in spread(retention).items():
                    row[f'projected_to_unprojected_norm_{stat}'] = value
                layer_rows.append(row)
        for reference in ['baseline', 'ours']:
            selected = []
            for name in conv:
                ratios = {k: [] for k in ['gradient_ratio', 'weight_norm_ratio',
                                         'relative_update_ratio', 'scale_adjusted_gradient_ratio']}
                for batch in range(4):
                    legacy = lookup[(f'legacy_e{epoch}', name, batch)]
                    other = lookup[(f'{reference}_e{epoch}', name, batch)]
                    gr = float(legacy['gradient_rms']) / float(other['gradient_rms'])
                    wr = float(legacy['parameter_l2']) / float(other['parameter_l2'])
                    ratios['gradient_ratio'].append(gr)
                    ratios['weight_norm_ratio'].append(wr)
                    ratios['scale_adjusted_gradient_ratio'].append(gr * wr)
                    ratios['relative_update_ratio'].append(
                        float(legacy['relative_projected_adam_proposal_l2']) /
                        float(other['relative_projected_adam_proposal_l2']))
                row = {'comparison': f'legacy/{reference}', 'epoch': epoch, 'parameter': name}
                for metric, values in ratios.items():
                    for stat, value in spread(values).items():
                        row[f'{metric}_{stat}'] = value
                selected.append(row)
                paired_rows.append(row)
            summary = {'comparison': f'legacy/{reference}', 'epoch': epoch,
                       'conv_layers': len(conv), 'replay_batches_per_layer': 4}
            for metric in ratios:
                summary[metric] = spread([r[f'{metric}_median'] for r in selected])
            summaries.append(summary)

    manifests = [json.loads(p.read_text()) for p in sorted(root.glob('replay-*/manifest.json'))]
    hashes = {m['cohort_sha256'] for m in manifests}
    assert len(hashes) == 1
    assert all(m['cohort_indices'] == manifests[0]['cohort_indices'] for m in manifests)
    assert len(manifests[0]['cohort_indices']) == 128
    write_csv(out / 'gradient_update_layers.csv', layer_rows)
    write_csv(out / 'gradient_update_paired_ratios.csv', paired_rows)
    record = {
        'input_csv_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'cohort_sha256': next(iter(hashes)), 'cohort_examples': 128,
        'batch_size': 32, 'augmentation': False, 'mode': 'training',
        'loss': 'cross_entropy', 'iterations': [6, 6, 4],
        'checkpoints': json.loads((root / 'inputs/checkpoints.json').read_text()),
        'aggregation': 'Pair identical batches; median across four batches per layer; then median/min/max across eight convolution layers. Spread is descriptive, not a confidence interval.',
        'update_definition': 'Read-only next Adam proposal using each checkpoint\'s own saved moments and scheduled LR, followed by conductance-bound projection and float32 rounding. Epoch50 is a hypothetical next step, not a logged training update.',
        'scale_adjusted_gradient_definition': 'Ratio of gradient_RMS * parameter_L2 for the same layer across schemes; invariant to uniform parameter-coordinate rescaling, not a measured loss change or log-parameter gradient.',
        'limitations': 'Different trained weights, BN parameters and Adam histories. Raw gradient vectors/directions and functional effects of the proposed updates were not retained in this CSV.',
        'convolution_summaries': summaries,
    }
    (out / 'gradient_update_summary.json').write_text(json.dumps(record, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for reference, color in [('baseline', '#377eb8'), ('ours', '#e41a1c')]:
        selected = [r for r in summaries if r['comparison'] == f'legacy/{reference}']
        for ax, key in zip(axes, ['gradient_ratio', 'relative_update_ratio']):
            values = [r[key] for r in selected]
            ax.plot(epochs, [v['median'] for v in values], marker='o', color=color,
                    label=f'Legacy / {reference}')
            ax.fill_between(epochs, [v['minimum'] for v in values],
                            [v['maximum'] for v in values], color=color, alpha=.13)
    for ax, title in zip(axes, ['Raw gradient RMS ratio', 'Relative Adam-update ratio']):
        ax.axhline(1, color='black', linestyle='--', linewidth=.8)
        ax.set(title=title, xlabel='Saved checkpoint epoch', xticks=epochs, yscale='log')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Eight conv layers: median ratios; bands span layer medians across four matched batches')
    fig.savefig(out / 'gradient_vs_adam_updates.png', dpi=170)
    plt.close(fig)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
