"""Validate and plot the matched-cohort post-BN voltage replay."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments import reporting


def analyze(root):
    errors = reporting.validate_run(root / 'replay')
    assert not errors, errors
    artifact = root / 'replay/artifacts/voltages.json'
    data = json.loads(artifact.read_text())
    terminal = json.loads((root / 'replay/result.json').read_text())['terminal_metrics']
    assert terminal['epochs'] == [0, 1, 5] and terminal['examples_per_epoch'] == 256
    for key in ('checkpoint_bytes_unchanged', 'parameters_and_bn_unchanged',
                'boundary_identities_pass', 'reference_rms_matched'):
        assert terminal[key]
    rows = data['rows']
    assert len(rows) == 84 and len(data['batches']) == 672
    lookup = {(r['epoch'], r['stage']): r for r in rows}
    assert len(lookup) == 84 and len(data['reference_comparisons']) == 24
    for key, row in lookup.items():
        batches = [b for b in data['batches'] if (b['epoch'], b['stage']) == key]
        assert len(batches) == 8 and {b['batch'] for b in batches} == set(range(8))
        for field in ('elements', 'sum', 'squares', 'zeros'):
            assert math.isclose(row[field], sum(b[field] for b in batches), rel_tol=1e-12, abs_tol=1e-10)
        assert math.isclose(row['rms']**2, row['std']**2 + row['mean']**2, rel_tol=1e-12, abs_tol=1e-14)
    boundary_rows = []
    for parameter in data['parameters']:
        epoch = parameter['epoch']
        for i, bn in enumerate(parameter['batch_norm'], 1):
            destination = f'block{i+1}' if i < 3 else 'head'
            post = lookup[epoch, f'block{i}/post_bn']
            physical = lookup[epoch, destination + '/injected_physical']
            gain = parameter['input_gains'][i]
            assert math.isclose(physical['rms'], gain * post['rms'], rel_tol=1e-7)
            assert abs(physical['mean']) < 1e-10
            batches = [b for b in data['bn_batches'] if b['epoch'] == epoch and b['block'] == i]
            assert len(batches) == 8
            expected_square = sum(sum(g*g*f + b*b for g, b, f in
                zip(bn['gamma'], bn['beta'], batch['variance_fraction'])) / len(bn['gamma'])
                for batch in batches) / 8
            assert math.isclose(post['rms']**2, expected_square, rel_tol=1e-6)
            boundary_rows.append(dict(epoch=epoch, boundary=f'block{i}_to_{destination}',
                pre_bn_rms=lookup[epoch, f'block{i}/pre_bn']['rms'],
                normalized_rms=lookup[epoch, f'block{i}/normalized_before_affine']['rms'],
                post_bn_mean=post['mean'], post_bn_rms=post['rms'], input_gain=gain,
                injected_rms=physical['rms'], gamma_rms=math.sqrt(sum(x*x for x in bn['gamma'])/len(bn['gamma'])),
                beta_rms=math.sqrt(sum(x*x for x in bn['beta'])/len(bn['beta'])), eps=bn['eps'],
                variance_below_eps_fraction=sum(v < bn['eps'] for b in batches for v in b['batch_variance']) /
                    (8 * len(bn['gamma']))))
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    for name, records in [('voltage_epoch_comparison', [{k: v for k, v in r.items() if k != 'channels'} for r in rows]),
                          ('boundary_summary', boundary_rows)]:
        with (out / f'{name}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    channel_rows = []
    for r in rows:
        if 'channels' in r:
            c = r['channels']
            for i in range(len(c['mean'])):
                channel_rows.append(dict(epoch=r['epoch'], stage=r['stage'], channel=i,
                    elements=c['elements'], **{k: c[k][i] for k in ('mean', 'std', 'rms')}))
    with (out / 'channel_voltages.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(channel_rows[0]))
        writer.writeheader(); writer.writerows(channel_rows)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    epochs = [0, 1, 5]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, layout='constrained')
    for i in range(1, 4):
        destination = f'block{i+1}' if i < 3 else 'head'
        axes[0, i-1].set_title(f'Block {i} → ' + (f'block {i+1}' if i < 3 else 'dense head'), fontweight='bold')
        stages = [f'block{i}/pre_bn', f'block{i}/post_bn', destination + '/injected_physical']
        for j, stage in enumerate(stages):
            ax = axes[j, i-1]
            values = [lookup[e, stage]['rms'] for e in epochs]
            ax.plot(epochs, values, 'o-', color=['#59636e', '#2568b4', '#d16b27'][j], lw=2)
            if j == 0:
                ax.set_yscale('log')
            else:
                ax.set_ylim(bottom=0, top=max(values) * 1.24)
            for e, value in zip(epochs, values):
                ax.annotate(f'{value:.3g}', (e, value), xytext=(0, 8), textcoords='offset points', ha='center')
            if j == 1:
                ax.plot(epochs, [lookup[e, f'block{i}/normalized_before_affine']['rms'] for e in epochs],
                    's--', color='#8e8e8e', alpha=.8, label='Before affine')
                ax.legend(loc='lower right', fontsize=8)
            ax.grid(alpha=.2); ax.set_xticks(epochs); ax.margins(x=.12)
            if i == 1:
                ax.set_ylabel(['Before BN: RMS (log scale)', 'After BN + affine: RMS', 'Injected physical voltage: RMS'][j])
            if j == 2:
                ax.set_xlabel('Training epoch')
    fig.suptitle('What reaches the next block?\nOurs fixed-0.3× EqProp trajectory · same 256 images · minibatch BN · T=[6,6,4]', fontsize=15)
    fig.savefig(out / 'boundary_voltage_rms.jpg', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout='constrained')
    for i, ax in enumerate(axes, 1):
        for stage, label in [('pre_bn', 'Before BN'), ('post_bn', 'After BN + affine')]:
            ax.plot(epochs, [lookup[e, f'block{i}/{stage}']['mean'] for e in epochs], 'o-', label=label)
        ax.axhline(0, color='grey', lw=.7); ax.grid(alpha=.2)
        ax.set(title=f'Block {i} boundary', xlabel='Training epoch', ylabel='Signed mean', xticks=epochs)
        ax.legend()
    fig.suptitle('Signed logical means; physical [v, −v] boundary means cancel by construction')
    fig.savefig(out / 'boundary_voltage_means.jpg', dpi=180); plt.close(fig)
    summary = dict(validation_errors=errors, terminal=terminal, boundary_rows=boundary_rows,
        artifact_sha256=reporting.sha256_file(artifact),
        maximum_reference_rms_relative_error=max(r['relative_error'] for r in data['reference_comparisons']),
        complete_stage_rows=84, complete_batch_stage_rows=672, pooled_statistics_checked=True,
        affine_and_gain_identities_checked=True)
    reporting.atomic_write_json(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
