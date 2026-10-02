"""Validate and compare sampled checkpoint RMS bands for two resistive heads."""
from __future__ import annotations

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
from experiments.replay_cifar_nonlinear_head import (
    DEPTHS, PRECISIONS, checkpoint_epoch, validate_run, validate_training_provenance,
)

ARMS = ('single', 'hidden512')
REFERENCE = 'common_full64_local_bptt'
THRESHOLD = .95


def read_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_pair(runs):
    first, second = (runs[arm] for arm in ARMS)
    for field in ('source_sha256', 'training_data_sha256'):
        require(first['manifest'][field] == second['manifest'][field], f'Cross-head {field} differs')
    configs = [run['manifest']['resolved_config'] for run in (first, second)]
    epochs = [checkpoint_epoch(config) for config in configs]
    require(epochs[0] == epochs[1], 'Cross-head checkpoint epoch differs')
    epoch = epochs[0]
    for field in ('study_id', 'cohort_indices', 'relative_rms_targets'):
        require(configs[0][field] == configs[1][field], f'Cross-head {field} differs')
    # At initialization preserve the original exact match (only head width differs).
    # Trained arms also have distinct descriptive arm/study labels, never distinct science.
    excluded = {'head_hidden_units', 'arm_id', 'study_id'} if epoch else {'head_hidden_units'}
    trunk_configs = [{k: v for k, v in c['model_config'].items() if k not in excluded} for c in configs]
    require(trunk_configs[0] == trunk_configs[1], 'Cross-head trunk configuration differs')
    for arm, expected in zip(ARMS, (0, 512)):
        config = runs[arm]['manifest']['resolved_config']
        require(config['arm_id'] == arm and config['model_config'].get('head_hidden_units', 0) == expected,
                f'Unexpected head architecture in {arm}')
    captures = [{(r['precision'], r['batch'], r['block']): (r['input'], r['states'])
                 for r in run['identity']['captures']} for run in (first, second)]
    matching = dict(same_source=True, same_training_data=True, same_ordered_cohort=True,
                    same_trunk_config=True, checkpoint_epoch=epoch,
                    checkpoint_hashes={arm: runs[arm]['manifest']['checkpoint_sha256'] for arm in ARMS})
    if epoch:
        for run, config in zip((first, second), configs):
            validate_training_provenance(config, run['manifest']['checkpoint_sha256'])
        provenance = [config['training_provenance'] for config in configs]
        for field in ('initial_trunk_sha256', 'training_source_sha256'):
            require(provenance[0][field] == provenance[1][field], f'Cross-head training {field} differs')
        require(captures[0].keys() == captures[1].keys(), 'Cross-head capture coverage differs')
        matching.update(same_training_protocol=True, same_training_source=True, same_initializer_trunk=True,
            initializer_trunk_sha256=provenance[0]['initial_trunk_sha256'],
            training_runs={arm: config['training_provenance']['training_run'] for arm, config in zip(ARMS, configs)},
            trained_trunk_divergence_expected=True,
            cross_head_trunk_equal={precision: first['identity']['models'][precision]['trunk_sha256'] ==
                second['identity']['models'][precision]['trunk_sha256'] for precision in PRECISIONS},
            cross_head_free_states_equal=captures[0] == captures[1],
            changed_capture_count=sum(captures[0][key] != captures[1][key] for key in captures[0]),
            comparison_scope='Matched training protocols from the same starting trunk; learned trunks may differ.')
    else:
        for precision in PRECISIONS:
            models = [run['identity']['models'][precision] for run in (first, second)]
            require(models[0]['trunk_sha256'] == models[1]['trunk_sha256'],
                    f'Cross-head {precision} trunk hash differs')
            tensors = [{group: {name: digest for name, digest in values.items() if not name.startswith('head.')}
                        for group, values in model['tensors'].items()} for model in models]
            require(tensors[0] == tensors[1], f'Cross-head {precision} trunk tensor hashes differ')
        require(captures[0] == captures[1], 'Cross-head block input/free-state captures differ')
        matching.update(exact_trunk_tensors_both_precisions=True,
                        exact_block_inputs_and_free_states_all_24_captures=True)
    return matching


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}_{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            require(not isinstance(item, float) or math.isfinite(item), f'Nonfinite metric: {name}')
            result[name] = item
    return result


def extrema(values):
    present = [value for value in values if value is not None]
    return (min(present) if present else None, max(present) if present else None,
            len(values) - len(present))


def make_tables(runs):
    points, layers = [], []
    for arm, run in runs.items():
        comparison = run['comparison']
        for scope, field in (('pooled', 'layers'), ('batch', 'batches')):
            for row in comparison[field]:
                layers.append(dict(arm=arm, scope=scope, **flatten(row)))
        for cal in sorted(run['calibration'], key=lambda row: (row['block'], row['target_index'])):
            for precision in ('float32', 'float64'):
                point = dict(arm=arm, precision=precision, block=cal['block'],
                             target_index=cal['target_index'], target_relative_rms=cal['target_relative_rms'],
                             calibration_status=cal['status'], beta_injected_B=cal['beta_injected_B'],
                             passing=None, missing_reason=cal.get('reason'))
                if cal['status'] == 'missing':
                    points.append(point)
                    continue
                groups = [[row for row in comparison[field] if
                           (row['precision'], row['block'], row['target_index']) ==
                           (precision, cal['block'], cal['target_index'])] for field in ('layers', 'batches')]
                pooled, batches = groups
                require(len(pooled) == DEPTHS[cal['block']-1] and len(batches) == 4 * len(pooled),
                        'Incomplete target comparison')
                output = pooled[0]['output_displacement']
                require(all(row['output_displacement'] == output for row in pooled),
                        'Pooled block-output displacement differs across layers')
                point.update(flatten(output, 'output_displacement'))
                point.update(output_force_rms=pooled[0]['output_force_rms'],
                             injected_force_rms=pooled[0]['injected_force_rms'])
                cosines = [row[REFERENCE]['cosine'] for row in batches]
                point['passing'] = all(value is not None and value > THRESHOLD for value in cosines)
                for scope, rows in zip(('pooled', 'batch'), groups):
                    for metric in ('cosine', 'norm_ratio', 'eqprop_norm', 'bptt_norm', 'eqprop_rms', 'bptt_rms'):
                        low, high, absent = extrema([row[REFERENCE][metric] for row in rows])
                        point.update({f'{scope}_{metric}_min': low, f'{scope}_{metric}_max': high,
                                      f'{scope}_{metric}_undefined_count': absent})
                point['passing_layer_batches'] = sum(value is not None and value > THRESHOLD for value in cosines)
                point['layer_batches'] = len(cosines)
                points.append(point)
    return points, layers


def sampled_spans(points):
    """Only adjacent passing grid samples form a span; missing samples break it."""
    points = sorted(points, key=lambda row: row['target_index'])
    require([row['target_index'] for row in points] == list(range(len(points))), 'Noncontiguous declared grid')
    segments, active = [], []
    for point in points:
        if point['passing'] is True:
            active.append(point)
        elif active:
            segments.append(active)
            active = []
    if active:
        segments.append(active)
    result = []
    for segment in segments:
        first, last = segment[0], segment[-1]
        before = points[first['target_index']-1] if first['target_index'] else None
        after = points[last['target_index']+1] if last['target_index'] + 1 < len(points) else None
        boundary = lambda row: ('grid_edge_unresolved' if row is None else
                                'calibration_missing_unresolved' if row['passing'] is None else
                                'bracketed_by_sampled_failure')
        actual_lower = first['output_displacement_relative_rms']
        actual_upper = last['output_displacement_relative_rms']
        result.append(dict(first_target_index=first['target_index'], last_target_index=last['target_index'],
            sampled_points=len(segment), lower_target_relative_rms=first['target_relative_rms'],
            upper_target_relative_rms=last['target_relative_rms'],
            target_upper_lower_ratio=last['target_relative_rms'] / first['target_relative_rms'],
            lower_actual_output_relative_rms=actual_lower, upper_actual_output_relative_rms=actual_upper,
            actual_output_upper_lower_ratio=actual_upper / actual_lower if actual_lower else None,
            lower_boundary=boundary(before), upper_boundary=boundary(after)))
    return dict(spans=result, largest=max(result, key=lambda row: (row['target_upper_lower_ratio'],
                row['sampled_points'])) if result else None)


def summarize_bands(points):
    bands, comparisons = [], []
    for precision in ('float32', 'float64'):
        for block in (1, 2, 3):
            arms = {}
            for arm in ARMS:
                selected = [row for row in points if (row['arm'], row['precision'], row['block']) ==
                            (arm, precision, block)]
                band = dict(arm=arm, precision=precision, block=block, **sampled_spans(selected),
                            missing_target_indices=[row['target_index'] for row in selected if row['passing'] is None])
                bands.append(band)
                arms[arm] = band
            control, hidden = (arms[arm]['largest'] for arm in ARMS)
            ratio = (hidden['target_upper_lower_ratio'] / control['target_upper_lower_ratio']
                     if control and hidden else None)
            actual_ratio = (hidden['actual_output_upper_lower_ratio'] / control['actual_output_upper_lower_ratio']
                            if control and hidden and control['actual_output_upper_lower_ratio'] and
                            hidden['actual_output_upper_lower_ratio'] is not None else None)
            comparisons.append(dict(precision=precision, block=block,
                hidden_over_single_sampled_span_ratio=ratio,
                hidden_over_single_actual_endpoint_span_ratio=actual_ratio,
                material_sampled_widening=ratio >= 3 if ratio is not None else None,
                sampled_span_regression=ratio < 1 if ratio is not None else None,
                control_has_passing_samples=control is not None, hidden_has_passing_samples=hidden is not None,
                ratio_undefined_reason=('no_control_passing_band' if control is None else
                                        'no_hidden_passing_band' if hidden is None else None)))
    return bands, comparisons


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot(points, path, checkpoint_epoch=0):
    colors = dict(single='#64748b', hidden512='#d97706')
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        figure, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)
        for row, precision in enumerate(('float32', 'float64')):
            for col, block in enumerate((1, 2, 3)):
                axis = axes[row, col]
                positive_x = []
                for arm in ARMS:
                    selected = sorted([p for p in points if (p['arm'], p['precision'], p['block']) ==
                                       (arm, precision, block)], key=lambda p: p['target_index'])
                    x = [p.get('output_displacement_relative_rms') or math.nan for p in selected]
                    positive_x.extend(value for value in x if math.isfinite(value) and value > 0)
                    for scope, style in (('batch', '-'), ('pooled', '--')):
                        y = [p.get(f'{scope}_cosine_min') if p.get(f'{scope}_cosine_min') is not None and
                             p.get(f'{scope}_cosine_undefined_count') == 0 else math.nan for p in selected]
                        axis.plot(x, y, style, marker='o', markersize=3, color=colors[arm])
                axis.axhline(THRESHOLD, color='black', linewidth=.8, linestyle=':')
                axis.set_xscale('log')
                if positive_x:
                    axis.set_xlim(min(positive_x) / 1.4, max(positive_x) * 1.4)
                else:
                    axis.set_xlim(1e-4, 1)
                    axis.text(.5, .4, 'No positive measured RMS', transform=axis.transAxes, ha='center')
                axis.set_title(f'{precision} · block {block}')
                axis.set_xlabel('Actual pooled output relative centered RMS')
                axis.set_ylabel('Minimum layer cosine' if col == 0 else '')
                axis.grid(alpha=.2)
        handles = [Line2D([], [], color=colors[arm], label=arm) for arm in ARMS]
        handles += [Line2D([], [], color='black', linestyle=style, label=label) for style, label in
                    (('-', 'Minimum over layers and four batches'), ('--', 'Minimum over pooled layer gradients'),
                     (':', 'Strict passing threshold: cosine > 0.95'))]
        figure.legend(handles=handles, loc='lower center', ncol=3, frameon=False)
        label = f'After BPTT epoch {checkpoint_epoch}' if checkpoint_epoch else 'Initialization only'
        figure.suptitle(f'{label} · common float64 local BPTT reference · missing targets break curves')
        figure.tight_layout(rect=(0, .095, 1, .95))
        figure.savefig(path, dpi=180, format='jpg')
        plt.close(figure)


def analyze(root):
    runs = {}
    for arm in ARMS:
        path = root / arm
        require((path / 'status.json').is_file(), f'{arm}: no run status; cannot analyze a partial run')
        state = read_json(path / 'status.json')['state']
        require(state == 'complete', f'{arm}: run state is {state!r}; refusing partial-run analysis')
        coverage = validate_run(path)
        runs[arm] = dict(coverage=coverage, manifest=read_json(path / 'manifest.json'),
                         **{name: read_json(path / f'artifacts/{name}.json')
                            for name in ('calibration', 'comparison', 'identity')})
    matching = validate_pair(runs)
    points, layers = make_tables(runs)
    bands, comparisons = summarize_bands(points)
    epoch = matching['checkpoint_epoch']
    summary = dict(evidence_class='trained_checkpoint_diagnostic' if epoch else 'initialization_only_diagnostic',
        checkpoint_epoch=epoch, canonical_validation_passed=True,
        matching=matching, coverage={arm: run['coverage'] for arm, run in runs.items()},
        all_calibrations_matched=all(run['coverage']['scientific_coverage_complete'] for run in runs.values()),
        criterion=dict(reference=REFERENCE, cosine_strictly_greater_than=THRESHOLD,
            requires_every_layer_and_all_four_batches=True, pooled_cosine_is_diagnostic_only=True,
            span_measure='largest contiguous passing sampled target upper/lower ratio',
            material_hidden_over_single_ratio=3, interpolation=False, missing_calibration_breaks_span=True),
        bands=bands, comparisons=comparisons,
        limitations=['Passing spans describe sampled targets, not continuous intervals.',
                     'Grid-edge and calibration-missing boundaries remain unresolved.',
                     ('Matched trained-head comparison includes expected learned-trunk differences; it does not isolate head-only forcing.'
                      if epoch else 'Initialization replay does not establish training or learned accuracy benefits.'),
                     *(['Gradient agreement does not establish an accuracy benefit.'] if epoch else [])])
    out = root / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'points.csv', points)
    write_csv(out / 'per-layer.csv', layers)
    plot(points, out / 'cosine-vs-output-rms.jpg', checkpoint_epoch=epoch)
    reporting.atomic_write_json(out / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
