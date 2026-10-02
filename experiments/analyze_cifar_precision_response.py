"""Validate matched CIFAR precision controls and plot their response curves."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments import reporting


ARMS = ['float32', 'float32_states_float64_arithmetic', 'float64']
LABELS = ['Float32', 'Float32 states / float64 gradient', 'Full float64']
COLORS = ['#777777', '#D97923', '#167BA1']
LAYERS = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2)]
OUTPUT_LAYERS = {block: max(layer for b, layer in LAYERS if b == block) for block in [1, 2, 3]}


def passes(row, norm=True):
    g = row['native_bptt']
    return g['cosine'] is not None and g['cosine'] >= .95 and (
        not norm or .9 <= g['norm_ratio'] <= 1.1)


def analyze(root, epochs):
    out = root / 'analysis'; out.mkdir(exist_ok=True)
    all_rows, boundaries, summaries, terminals, passing_sets = [], [], [], [], []
    cohort = None
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    for epoch in epochs:
        run = root / f'cells/e{epoch}'
        errors = reporting.validate_run(run); assert not errors, errors
        data = json.loads((run / 'artifacts/comparison.json').read_text())
        terminal = json.loads((run / 'result.json').read_text())['terminal_metrics']
        assert terminal['epoch'] == epoch and terminal['examples'] == 256
        assert terminal['rows'] == 480 and terminal['batch_rows'] == 3840
        assert terminal['baseline_rows_checked'] == 1088
        for key in ['precision_checks_passed', 'checkpoint_bytes_unchanged', 'parameters_and_bn_unchanged']:
            assert terminal[key]
        terminals.append(terminal)
        if cohort is None: cohort = data['cohort_indices']
        assert cohort == data['cohort_indices'] and len(set(cohort)) == 256
        rows = data['layers']; all_rows.extend(rows)
        lookup = {(r['arm'], r['block'], r['layer'], r['beta_multiplier']): r for r in rows}
        assert len(lookup) == len(rows) == 480
        assert len(data['batches']) == 3840 and len(data['references']) == 8
        for row in rows:
            output = lookup[row['arm'], row['block'], OUTPUT_LAYERS[row['block']], row['beta_multiplier']]
            assert output['case_index'] == row['case_index']
            assert output['beta_injected_B'] == row['beta_injected_B']
            row['block_output_layer'] = output['layer']
            row['block_output_displacement'] = {key: output['state_displacement'][key]
                for key in ['absolute_rms', 'relative_rms', 'free_rms']}
            batches = [b for b in data['batches'] if (b['arm'], b['name'], b['case_index']) ==
                       (row['arm'], row['name'], row['case_index'])]
            assert len(batches) == 8 and {b['batch'] for b in batches} == set(range(8))
            for key in ['elements', 'centered_squares', 'free_squares', 'unchanged_elements',
                        'active_elements', 'active_unchanged_elements', 'active_sub_ulp_elements']:
                assert math.isclose(row['state_displacement'][key],
                    sum(b['state_displacement'][key] for b in batches), rel_tol=1e-12, abs_tol=1e-25)
            if row['arm'] == ARMS[1]:
                original = lookup[ARMS[0], row['block'], row['layer'], row['beta_multiplier']]
                assert original['state_displacement'] == row['state_displacement']
            row['batch_cosine_min'] = min(b['native_bptt']['cosine'] for b in batches
                if b['native_bptt']['cosine'] is not None) if any(b['native_bptt']['cosine'] is not None for b in batches) else None
            row['passing_batches'] = sum(passes(b) for b in batches)
        for arm in ARMS:
            for block in [1, 2, 3]:
                pooled_pass, batch_pass = [], []
                for multiplier in sorted({r['beta_multiplier'] for r in rows}):
                    group = [r for r in rows if (r['arm'], r['block'], r['beta_multiplier']) == (arm, block, multiplier)]
                    batches = [r for r in data['batches'] if (r['arm'], r['block'], r['beta_multiplier']) == (arm, block, multiplier)]
                    if all(passes(r) for r in group): pooled_pass.append(multiplier)
                    if all(passes(r) for r in batches): batch_pass.append(multiplier)
                passing_sets.append(dict(epoch=epoch, arm=arm, block=block,
                    pooled=pooled_pass, every_batch=batch_pass))
            for block, layer in LAYERS:
                group = sorted([r for r in rows if (r['arm'], r['block'], r['layer']) == (arm, block, layer)],
                               key=lambda r: r['beta_multiplier'])
                assert len(group) == 20
                for require_norm in [False, True]:
                    good = [i for i, r in enumerate(group) if passes(r, require_norm)]
                    first = group[good[0]] if good else None
                    previous = group[good[0]-1] if good and good[0] else None
                    boundaries.append(dict(epoch=epoch, arm=arm, block=block, layer=layer,
                        criterion='cosine_and_norm' if require_norm else 'cosine',
                        first_passing_multiplier=first['beta_multiplier'] if first else None,
                        previous_failing_multiplier=previous['beta_multiplier'] if previous else None,
                        first_passing_absolute_rms=first['state_displacement']['absolute_rms'] if first else None,
                        first_passing_relative_rms=first['state_displacement']['relative_rms'] if first else None,
                        first_passing_cosine=first['native_bptt']['cosine'] if first else None,
                        first_passing_norm_ratio=first['native_bptt']['norm_ratio'] if first else None,
                        first_passing_batches=first['passing_batches'] if first else None,
                        lower_boundary_bracketed=previous is not None))
        changes, recovery = [], dict(float32_failing_small_beta=0, mixed_recovered=0, full64_recovered=0)
        for r in rows:
            if r['arm'] != 'float32': continue
            m = lookup[ARMS[1], r['block'], r['layer'], r['beta_multiplier']]
            full = lookup[ARMS[2], r['block'], r['layer'], r['beta_multiplier']]
            if r['native_bptt']['cosine'] is not None and m['native_bptt']['cosine'] is not None:
                changes.append(abs(m['native_bptt']['cosine'] - r['native_bptt']['cosine']))
            if r['beta_multiplier'] <= 1 and not passes(r):
                recovery['float32_failing_small_beta'] += 1
                recovery['mixed_recovered'] += passes(m)
                recovery['full64_recovered'] += passes(full)
        free_changes = []
        for block, layer in LAYERS:
            pair = [next(r for r in rows if (r['arm'], r['block'], r['layer'], r['case_index']) ==
                         (arm, block, layer, 0)) for arm in ['float32', 'float64']]
            free_changes.append(dict(block=block, layer=layer,
                float32_rms=pair[0]['state_displacement']['free_rms'],
                float64_rms=pair[1]['state_displacement']['free_rms'],
                ratio=pair[1]['state_displacement']['free_rms']/pair[0]['state_displacement']['free_rms']))
        force_changes = []
        for block in [1, 2, 3]:
            rms = {}
            for precision in ['float32', 'float64']:
                group = [r for r in data['forcing'] if r['precision']==precision and r['block']==block]
                assert len(group) == 8
                rms[precision] = math.sqrt(sum(r['force_squares'] for r in group) / sum(r['elements'] for r in group))
            force_changes.append(dict(block=block, **rms, ratio=rms['float64']/rms['float32']))
        summaries.append(dict(epoch=epoch, validation_errors=errors, recovery=recovery,
            free_state_changes=free_changes, output_force_changes=force_changes,
            max_mixed_vs_float32_cosine_change=max(changes),
            minimum_native_reference_cosine=min(r['native']['cosine'] for r in data['references']),
            minimum_batch_native_reference_cosine=min(r['native']['cosine'] for r in data['reference_batches']),
            maximum_native_reference_norm_ratio_error=max(abs(r['native']['norm_ratio']-1) for r in data['references']),
            minimum_full64_local_vs_native_cosine=min(r['local_vs_native_bptt']['cosine'] for r in rows if r['arm']=='float64'),
            artifact_sha256=reporting.sha256_file(run / 'artifacts/comparison.json')))
        for metric, label in [('beta_multiplier', 'Beta multiplier'),
                              ('absolute_rms', 'Absolute centered state RMS displacement'),
                              ('relative_rms', 'Relative centered state RMS displacement'),
                              ('block_output_absolute_rms', 'Block output absolute RMS displacement'),
                              ('block_output_relative_rms', 'Block output relative RMS displacement')]:
            block_output_axis = metric.startswith('block_output_')
            state_metric = metric.removeprefix('block_output_')
            state_source = 'block_output_displacement' if block_output_axis else 'state_displacement'
            fig, axes = plt.subplots(3, 3, figsize=(14, 10.5), sharey=True,
                                     sharex='row' if block_output_axis else False, layout='constrained')
            for block, layer in LAYERS:
                ax = axes[block-1, layer-1]
                for arm, color in zip(ARMS, COLORS):
                    group = sorted([r for r in rows if (r['arm'], r['block'], r['layer']) == (arm, block, layer)],
                                   key=lambda r: r['beta_multiplier'])
                    points = [(r['beta_multiplier'] if metric=='beta_multiplier' else r[state_source][state_metric],
                               r['native_bptt']['cosine']) for r in group]
                    points = [(x, y) for x, y in points if x is not None and x > 0 and y is not None]
                    if points:
                        ax.plot(*zip(*points), marker='o', ms=3.5, lw=1.7, color=color,
                                linestyle='--' if arm == ARMS[1] else '-')
                finite_cosines = [r['native_bptt']['cosine'] for r in rows if r['native_bptt']['cosine'] is not None]
                ax.set_xscale('log'); ax.set_ylim(min(-.02, min(finite_cosines)-.04), 1.025); ax.grid(alpha=.18)
                ax.axhline(.95, color='#555', ls=':', lw=1)
                ax.set_title(f'Block {block} · Conv {layer}', loc='left')
                ax.set_xlabel(label)
                if layer == 1: ax.set_ylabel('EqProp / native BPTT cosine')
            axes[2, 2].axis('off')
            handles = [Line2D([], [], color=c, marker='o', ms=4,
                       ls='--' if a == ARMS[1] else '-', label=l) for a,c,l in zip(ARMS,COLORS,LABELS)]
            axes[2, 2].legend(handles=handles, loc='upper left', frameon=False, fontsize=10)
            detail = ('Block output = last analog state\nB1/B2: Conv3 · B3: Conv2\nBefore pooling and BN\nSame x values within each block\n\n'
                      if block_output_axis else '')
            detail += ('256 images · Batch32 · T/K=[6,6,4]\nEach arm uses its native reference\n'
                       'Zero / undefined points omitted.\nLines connect sampled beta values.')
            axes[2, 2].text(.02, .53 if block_output_axis else .47, detail,
                           transform=axes[2, 2].transAxes, va='top', fontsize=10)
            fig.suptitle(f'CIFAR precision controls · epoch {epoch}\nCosine versus {label.lower()}', fontsize=15)
            fig.savefig(out / f'e{epoch}_cosine_vs_{metric}.jpg', dpi=170); plt.close(fig)
    flat = []
    for r in all_rows:
        row = {k: v for k, v in r.items() if not isinstance(v, dict)}
        for parent in ['state_displacement', 'block_output_displacement', 'native_bptt', 'local_bptt']:
            row.update({parent+'_'+k: v for k, v in r[parent].items()})
        flat.append(row)
    for name, records in [('layers', flat), ('lower_boundaries', boundaries)]:
        with (out / f'{name}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    common = []
    for arm in ARMS:
        for block in [1, 2, 3]:
            group = [r for r in passing_sets if r['arm']==arm and r['block']==block]
            common.append(dict(arm=arm, block=block, **{key: sorted(set.intersection(
                *(set(r[key]) for r in group))) for key in ['pooled', 'every_batch']}))
    summary = dict(epochs=epochs, complete_coverage=epochs==[0,1,5], rows=len(all_rows),
        summaries=summaries, terminal_metrics=terminals, lower_boundaries=boundaries,
        passing_multipliers_by_epoch=passing_sets, common_passing_multipliers=common,
        same_float32_endpoints_checked=True, displacement_pooling_checked=True,
        block_output_axes=dict(output_layers=OUTPUT_LAYERS, state='last physical analog state before pooling and BN',
            relative_definition='RMS((v_plus-v_minus)/2) / RMS(v_free) at block output',
            matching=['epoch', 'arm', 'block', 'beta_multiplier'], shared_x_limits_within_block=True))
    reporting.atomic_write_json(out / 'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'lower_boundaries'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--epochs', nargs='+', type=int, default=[0,1,5])
    args = parser.parse_args(); analyze(args.root, args.epochs)
