"""Validate and plot the compact Conv3 absolute/relative endpoint-noise replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

SCHEMES = ('baseline', 'ours', 'legacy')
LAYERS = ('ConvWeight_0', 'ConvWeight_1', 'ConvWeight_2', 'DenseWeight_0')
TARGETS = (2., 6.)
LEVELS = (1e-4, 1e-3)
MODELS = ('absolute', 'nodewise_relative')
SEEDS = (101000000, 102000000, 103000000)
METRICS = ('cosine', 'noisy_clean_eqprop_cosine')


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    if value is None or str(value).lower() in ('', 'none', 'null'):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'Nonfinite measurement: {value}')
    return result


def summarize(rows):
    """Require the full declared grid; average individual cosines within draws."""
    acquisition = [('clean', 0., -1)] + list(itertools.product(MODELS, LEVELS, range(3)))
    expected = {(s, r, p, m, level, b, d)
                for s, r, p, (m, level, d), b in itertools.product(
                    SCHEMES, TARGETS, LAYERS, acquisition, range(36))}
    observed = set()
    groups = defaultdict(list)
    for row in rows:
        s, r, p = row['scheme'], float(row['relative_target']), row['parameter_name']
        m, level = row['noise_model'], float(row['noise_level'])
        b, d = int(row['batch_index']), int(row['noise_draw_index'])
        key = (s, r, p, m, level, b, d)
        if key not in expected or key in observed:
            raise ValueError(f'Unexpected or repeated row: {key}')
        observed.add(key)
        seed = number(row['noise_seed'])
        if seed != (None if m == 'clean' else SEEDS[d]):
            raise ValueError(f'Unexpected noise seed: {key}/{seed}')
        for metric in METRICS:
            value = number(row[metric])
            if value is not None and abs(value) > 1 + 1e-10:
                raise ValueError(f'Cosine outside [-1,1]: {key}')
        groups[(s, r, p, m, level, d)].append(row)
    if observed != expected:
        raise ValueError(f'Incomplete coverage: {len(observed)}/{len(expected)} rows')
    per_draw = []
    fields = (*METRICS, 'clean_eqprop_bptt_cosine', 'noise_over_clean_norm',
              'relative_l2_difference_over_bptt')
    for key, samples in sorted(groups.items()):
        s, r, p, m, level, d = key
        entry = dict(scheme=s, relative_target=r, parameter_name=p,
                     noise_model=m, noise_level=level, noise_draw_index=d,
                     samples=len(samples))
        for field in fields:
            values = [number(row[field]) for row in samples]
            entry[field] = mean(values) if all(v is not None for v in values) else None
            entry[field + '_undefined'] = sum(v is None for v in values)
        per_draw.append(entry)
    cell_groups = defaultdict(list)
    for row in per_draw:
        cell_groups[tuple(row[k] for k in ('scheme', 'relative_target', 'parameter_name',
                                          'noise_model', 'noise_level'))].append(row)
    cells = []
    for key, draws in sorted(cell_groups.items()):
        entry = dict(zip(('scheme', 'relative_target', 'parameter_name',
                          'noise_model', 'noise_level'), key))
        entry['draws'] = len(draws)
        for field in fields:
            values = [r[field] for r in draws]
            valid = all(v is not None for v in values)
            entry[field] = mean(values) if valid else None
            entry[field + '_min'] = min(values) if valid else None
            entry[field + '_max'] = max(values) if valid else None
        cells.append(entry)
    lookup = {tuple(row[k] for k in ('scheme', 'relative_target', 'parameter_name',
                                    'noise_model', 'noise_level', 'noise_draw_index')): row
              for row in per_draw}
    contrasts = []
    for s, r, p, level, d, metric in itertools.product(
            ('baseline', 'ours'), TARGETS, LAYERS, LEVELS, range(3), METRICS):
        gaps = {}
        for model in MODELS:
            left = lookup[('legacy', r, p, model, level, d)][metric]
            right = lookup[(s, r, p, model, level, d)][metric]
            gaps[model] = left - right if left is not None and right is not None else None
        contrasts.append(dict(competitor=s, relative_target=r, parameter_name=p,
            noise_level=level, noise_draw_index=d, metric=metric,
            absolute_legacy_advantage=gaps['absolute'],
            relative_legacy_advantage=gaps['nodewise_relative'],
            advantage_reduction=(gaps['absolute'] - gaps['nodewise_relative'])
                if all(v is not None for v in gaps.values()) else None))
    return cells, per_draw, contrasts


def load_runs(study):
    runs = sorted(study.glob('local/runs/*/result.json')) + sorted(
        study.glob('collected/akib/**/runs/*/result.json'))
    if len(runs) != 6:
        raise ValueError(f'Expected six authoritative local bundles, got {len(runs)}')
    rows, provenance, states, noise_rows = [], [], [], []
    bptt_hashes, raw_hashes = {}, {}
    for result_path in runs:
        run = result_path.parent
        result = json.loads(result_path.read_text())
        status = json.loads((run / 'status.json').read_text())
        manifest = json.loads((run / 'manifest.json').read_text())
        if status['state'] != 'complete' or manifest.get('smoke'):
            raise ValueError(f'Nonproduction or incomplete bundle: {run}')
        config = manifest['configuration']['resolved']
        if manifest['cohort_sha256'] != config['expected_cohort_sha256']:
            raise ValueError(f'Cohort mismatch: {run}')
        if manifest['official_test_read'] or manifest['optimizer_steps_applied']:
            raise ValueError(f'Unexpected scientific scope: {run}')
        if manifest['runtime']['device'] != 'cuda':
            raise ValueError(f'Non-CUDA execution: {run}')
        terminal = result['terminal_metrics']
        required = ('source_bytes_unchanged', 'all_parameter_tensors_unchanged',
                    'all_bias_tensors_exact_zero',
                    'saved_activation_cpu_offload')
        if not all(terminal[k] for k in required) or not result['completion']['coverage_complete']:
            raise ValueError(f'Failed scientific guard: {run}')
        if not math.isclose(terminal['actual_relative_output_rms'], terminal['relative_target'], rel_tol=.01):
            raise ValueError(f'Output displacement not matched: {run}')
        values = read_csv(run / 'layer_metrics.csv')
        for row in values:
            key = (row['scheme'], row['batch_index'], row['parameter_name'])
            digest = row['bptt_gradient_sha256']
            # User waived cross-device reference qualification. Preserve native
            # reference identities without treating different GPU hashes as failure.
            bptt_hashes.setdefault(key, set()).add(digest)
        run_noise = read_csv(run / 'endpoint_read_noise.csv')
        expected = set(itertools.product(MODELS, LEVELS, range(36), range(3),
                                         ('negative', 'positive'), range(4)))
        observed = set()
        for row in run_noise:
            model, level = row['noise_model'], float(row['noise_level'])
            batch, draw = int(row['batch_index']), int(row['noise_draw_index'])
            phase, layer = row['phase'], int(row['state_layer_index'])
            identity = (model, level, batch, draw, phase, layer)
            if identity not in expected or identity in observed:
                raise ValueError(f'Duplicate/unexpected endpoint read: {identity}')
            observed.add(identity)
            seed = SEEDS[draw] + 1000 * batch + 100 * (1 if phase == 'negative' else 2) + layer
            if int(row['seed']) != seed or int(row['noise_seed']) != SEEDS[draw]:
                raise ValueError(f'Unpaired endpoint seed: {identity}')
            raw = (row['standard_normal_sha256'], int(row['element_count']))
            key = (batch, draw, phase, layer)
            if raw_hashes.setdefault(key, raw) != raw:
                raise ValueError(f'Raw noise changed across scheme/beta/model/level: {key}')
            expected_rms = level if model == 'absolute' else level * float(row['clean_endpoint_rms'])
            if not math.isclose(expected_rms, float(row['expected_noise_rms']), rel_tol=1e-12, abs_tol=1e-18):
                raise ValueError(f'Noise scaling mismatch: {identity}')
            if model == 'nodewise_relative' and int(row['noise_nonzero_at_clean_zero_count']):
                raise ValueError(f'Unexpected relative-noise floor: {identity}')
        if observed != expected:
            raise ValueError(f'Incomplete endpoint read coverage: {run}')
        noise_rows.extend(run_noise)
        states.extend(read_csv(run / 'state_signal.csv'))
        rows.extend(values)
        provenance.append(dict(run=str(run), rows=len(values),
                               result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                               runner_sha256=manifest['git']['runner_sha256'],
                               config_sha256=manifest['configuration']['sha256'],
                               cross_device_bptt_equivalence_checked=False,
                               terminal_metrics=result.get('terminal_metrics', {})))
    return rows, provenance, states, noise_rows


def scale_controls(states, noise_rows):
    state_groups, noise_groups = defaultdict(list), defaultdict(list)
    for row in states:
        state_groups[(row['scheme'], float(row['relative_target']), int(row['state_layer_index']))].append(row)
    physical = []
    for (scheme, target, layer), records in sorted(state_groups.items()):
        if sorted(int(r['batch_index']) for r in records) != list(range(36)):
            raise ValueError('Incomplete/duplicate state controls')
        n = sum(int(r['element_count']) for r in records)
        rms = lambda field: math.sqrt(sum(int(r['element_count']) * float(r[field])**2 for r in records) / n)
        D, F, P = rms('centered_half_difference_rms'), rms('free_rms'), rms('endpoint_rms')
        physical.append(dict(scheme=scheme, relative_target=target, state_layer_index=layer,
            free_rms=F, centered_displacement_rms=D, endpoint_rms=P,
            D_over_F=D/F if F else None, D_over_P=D/P if P else None))
    for row in noise_rows:
        key = (row['scheme'], float(row['relative_target']), int(row['state_layer_index']),
               row['noise_model'], float(row['noise_level']))
        noise_groups[key].append(row)
    noise = []
    for key, records in sorted(noise_groups.items()):
        n = sum(int(r['element_count']) for r in records)
        entry = dict(zip(('scheme', 'relative_target', 'state_layer_index', 'noise_model', 'noise_level'), key))
        for field in ('expected_noise_rms', 'actual_noise_rms', 'clean_endpoint_rms'):
            entry[field] = math.sqrt(sum(int(r['element_count']) * float(r[field])**2 for r in records) / n)
        noise.append(entry)
    return physical, noise


def plot(cells, destination, metric):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup = {(r['scheme'], r['relative_target'], r['parameter_name'],
               r['noise_model'], r['noise_level']): r for r in cells}
    colors = dict(baseline='#4477AA', ours='#228833', legacy='#CC6677')
    styles = dict(baseline=dict(linestyle='--', marker='o', markersize=9,
                                markerfacecolor='none', markeredgewidth=1.5, zorder=5),
                  ours=dict(linestyle='-', marker='^', markersize=6, zorder=4),
                  legacy=dict(linestyle='-.', marker='x', markersize=6, zorder=3))
    fig, axes = plt.subplots(4, 4, figsize=(15, 12), sharex=True, constrained_layout=True)
    for row_index, (model, level) in enumerate(itertools.product(MODELS, LEVELS)):
        for col, layer in enumerate(LAYERS):
            ax = axes[row_index, col]
            for scheme in SCHEMES:
                selected = [lookup[(scheme, target, layer, model, level)] for target in TARGETS]
                values = [r[metric] if r[metric] is not None else math.nan for r in selected]
                low = [r[metric + '_min'] if r[metric + '_min'] is not None else math.nan for r in selected]
                high = [r[metric + '_max'] if r[metric + '_max'] is not None else math.nan for r in selected]
                ax.plot(TARGETS, values, color=colors[scheme], label=scheme,
                        **styles[scheme])
                ax.fill_between(TARGETS, low, high, color=colors[scheme], alpha=.15)
                clean = [lookup[(scheme, target, layer, 'clean', 0.)][metric] for target in TARGETS]
                ax.plot(TARGETS, clean, ':', color=colors[scheme], alpha=.6)
            finite = [line.get_ydata() for line in ax.lines]
            minimum = min((float(v) for line in finite for v in line if v is not None and math.isfinite(v)), default=0.)
            ax.set_ylim(min(-.06, minimum - .03), 1.025)
            ax.set_xticks(TARGETS)
            ax.grid(alpha=.2)
            if row_index == 0:
                ax.set_title(('Convolution 1', 'Convolution 2', 'Convolution 3', 'Readout')[col])
            if col == 0:
                unit = 'sigma (voltage)' if model == 'absolute' else 'eta (fraction)'
                label = 'Absolute noise' if model == 'absolute' else 'Relative noise'
                ax.set_ylabel(f'{label}\n{unit}={level:g}\ncosine')
            if row_index == 3:
                ax.set_xlabel('Normalized output displacement D/F')
    axes[0, 0].legend(fontsize=9)
    reference = 'finite-K BPTT' if metric == 'cosine' else 'clean EqProp'
    fig.suptitle(f'Conv3 initialization: cosine to {reference}\n'
                 'Shading: range of 3 noise-draw means; dotted: clean control. '
                 'Absolute and relative levels have different units.')
    for extension in ('png', 'pdf'):
        fig.savefig(destination / f'{metric}.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    args = parser.parse_args()
    rows, provenance, states, noise_rows = load_runs(args.study_root)
    cells, per_draw, contrasts = summarize(rows)
    physical, noise = scale_controls(states, noise_rows)
    out = args.study_root / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    for name, values in (('layer_summary', cells), ('per_draw_summary', per_draw),
                         ('paired_advantage_reduction', contrasts),
                         ('physical_state_controls', physical), ('noise_scale_controls', noise)):
        write_csv(out / f'{name}.csv', values)
    (out / 'validation.json').write_text(json.dumps(dict(
        run_count=len(provenance), layer_rows=len(rows), runs=provenance,
        evidence_scope='exploratory; user waived additional qualification checks',
        canonical_bundle_revalidation_performed=False,
        cross_device_bptt_equivalence_checked=False), indent=2) + '\n')
    for metric in METRICS:
        plot(cells, out, metric)
    lines = ['# Compact Conv3 endpoint read-noise comparison', '',
             f'Collected {len(provenance)} completed production bundles and summarized {len(rows):,} layer/batch/draw rows.', '',
             'Exploratory: Filip waived further qualification checks. Cross-device BPTT numerical equivalence is unchecked; local and Akib use native GPU references.',
             'All results are initialization-only, one model seed and one fixed validation cohort.',
             'Absolute sigma is in voltage units; relative eta is dimensionless. Equal numbers do not match noise power.', '',
             '## Predeclared third-convolution diagnostic', '',
             '| Competitor | Alignment reference | Positive reduction draws/cells | Reduction range |',
             '|---|---|---:|---:|']
    for scheme, metric in itertools.product(('baseline', 'ours'), METRICS):
        selected = [r['advantage_reduction'] for r in contrasts
                    if r['competitor'] == scheme and r['metric'] == metric
                    and r['parameter_name'] == 'ConvWeight_2']
        defined = [v for v in selected if v is not None]
        span = f'{min(defined):+.6g} to {max(defined):+.6g}' if defined else 'undefined'
        lines.append(f'| {scheme} | {metric} | {sum(v > 0 for v in defined)}/{len(selected)} | {span} |')
    lines += ['', 'A positive reduction means the legacy-minus-competitor cosine gap is smaller under relative noise.',
              'This exploratory pattern does not isolate voltage scale from different noise severity/covariance.', '',
              '[BPTT alignment](cosine.png) · [Acquisition alignment](noisy_clean_eqprop_cosine.png)',
              '[Layer means](layer_summary.csv) · [Paired draw comparisons](paired_advantage_reduction.csv)', '']
    (out / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(runs=len(provenance), rows=len(rows), layer_cells=len(cells), output=str(out))))


if __name__ == '__main__':
    main()
