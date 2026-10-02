"""Validate and plot exp-025 full-float64 RMS sweeps on CPU; never replay a model.

Run: python experiments/analyze_cifar_layer_rms_transfer.py [--root ROOT]
Use --reference local_bptt for matched block-only comparisons in analysis/local_bptt/.
Partial mode excludes entire unfinished/invalid cases and cannot certify universality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_ROOT = Path('results/cifar-layer-rms-all-schemes-float64-20260930-v1')
SCHEMES = ('baseline', 'ours', 'legacy')
EPOCHS = {'eqprop': (0, 1, 5), 'bptt': (0, 10, 30, 50)}
TARGETS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, .00025, .0005, .001,
           .0025, .005, .01, .02, .035, .05, .065, .075, .085, .1, .125, .2, .4)
DEPTHS = (3, 3, 2)
LAYERS = tuple((b, l) for b, depth in enumerate(DEPTHS, 1) for l in range(1, depth + 1))
CRITERIA = ('cosine95', 'cosine99', 'cosine95_norm90_110')
COLORS = {'baseline': '#0072B2', 'ours': '#D55E00', 'legacy': '#009E73'}
SUMS = ('centered_squares', 'free_squares', 'elements', 'unchanged_elements')
REFERENCES = ('native_bptt', 'local_bptt', 'local_vs_native_bptt')
REFERENCE_LABELS = {'native_bptt': 'Full-network BPTT', 'local_bptt': 'Block-only BPTT'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=2e-10, abs_tol=1e-25)


def ratio(a, b):
    return a / b if a is not None and b else None


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rowkey(row, batch=False):
    key = row['case_index'], row['block'], row['layer']
    return (*key, row['batch']) if batch else key


def finite_tree(value, path='row'):
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, path + '/' + key)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child, path)
    elif isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite ' + path)


def index_rows(rows, batch=False):
    expected = {(t, b, l) for t in range(len(TARGETS)) for b, l in LAYERS}
    if batch:
        expected = {(*key, n) for key in expected for n in range(8)}
    indexed = {}
    for row in rows:
        key = rowkey(row, batch)
        require(key not in indexed, f'Duplicate row {key}')
        finite_tree(row)
        indexed[key] = row
    require(set(indexed) == expected,
            f'Coverage mismatch: missing {len(expected - set(indexed))}, unexpected {len(set(indexed) - expected)}')
    for key, row in indexed.items():
        t, b, l = key[:3]
        require(row['name'] == f'blocks.{b-1}.weights.{l-1}', f'Parameter identity {key}')
        require(row['examples'] == (32 if batch else 256), f'Cohort size {key}')
        require(close(row['target_relative_rms'], TARGETS[t]), f'Target identity {key}')
        require(row['reference_iterations'] == row['nudged_iterations'] == (6, 6, 4)[b-1], f'T/K {key}')
        require(row['beta_injected_B'] > 0 and close(row['beta_injected_B'], row['beta_multiplier']), f'Base beta != 1: {key}')
        state = row['state_displacement']
        require(state['layer'] == l and state['elements'] > 0, f'State identity {key}')
        require(all(state[k] >= 0 for k in SUMS), f'Negative state sums {key}')
        require(state['unchanged_elements'] <= state['elements'], f'Unchanged count {key}')
        for field, expected_value in {
            'absolute_rms': math.sqrt(state['centered_squares'] / state['elements']),
            'free_rms': math.sqrt(state['free_squares'] / state['elements']),
            'relative_rms': math.sqrt(state['centered_squares'] / state['free_squares']) if state['free_squares'] else None,
            'unchanged_fraction': state['unchanged_elements'] / state['elements'],
        }.items():
            require(close(state[field], expected_value), f'State {field}: {key}')
        require(close(row['injected_force_rms'], row['beta_injected_B'] * row['output_force_rms']), f'Force scaling {key}')
        require(close(row['native_bptt']['eqprop_norm'], row['local_bptt']['eqprop_norm']), f'EP norm {key}')
        for reference in REFERENCES:
            g = row[reference]
            require(close(g['zero_epsilon'], 1e-12), f'Zero threshold {key}')
            require(g['eqprop_norm'] >= 0 and g['bptt_norm'] >= 0, f'Gradient norm {key}')
            require(close(g['norm_ratio'], ratio(g['eqprop_norm'], g['bptt_norm'])), f'Norm ratio {key}')
            require(g['cosine'] is None if not g['eqprop_norm'] or not g['bptt_norm']
                    else g['cosine'] is not None and abs(g['cosine']) <= 1 + 1e-10, f'Cosine validity {key}')
            for who in ('eqprop', 'bptt'):
                require(0 <= g[who + '_zero_fraction'] <= g[who + '_below_epsilon_fraction'] <= 1, f'Zero fractions {key}')
        first = indexed[(0, *key[1:])]
        for reference in ('native_bptt', 'local_bptt'):
            for field in ('bptt_norm', 'bptt_rms', 'bptt_zero_fraction', 'bptt_below_epsilon_fraction'):
                require(close(row[reference][field], first[reference][field]), f'Reference varies with beta {key}/{field}')
        require(row['local_vs_native_bptt'] == first['local_vs_native_bptt'], f'Local/native reference varies {key}')
        for field in ('elements', 'free_squares'):
            require(close(state[field], first['state_displacement'][field]), f'Free state varies {key}')
        require(close(row['output_force_rms'], first['output_force_rms']), f'Force varies {key}')
        last = indexed[(t, b, DEPTHS[b-1], *key[3:])]
        for field in ('absolute_rms', 'relative_rms', 'free_rms', 'elements'):
            require(close(row['output_displacement'][field], last['state_displacement'][field]), f'Final state/output mismatch {key}/{field}')
        require(row['force_elements'] == last['state_displacement']['elements'], f'Force count {key}')
        require(close(row['beta_injected_B'], last['beta_injected_B']), f'Layer beta differs {key}')
        if not batch:
            achieved = row['output_displacement']['relative_rms']
            require(achieved is not None and abs(achieved / TARGETS[t] - 1) <= .03 + 1e-10,
                    f'Target missed by >3%: {key}, {achieved} vs {TARGETS[t]}')
    return indexed


def expected_cases():
    return {f'{"init" if epoch == 0 else group}-{scheme}-e{epoch}':
            (scheme, epoch, 'initialization' if epoch == 0 else group)
            for scheme in SCHEMES for group, epochs in EPOCHS.items() for epoch in epochs}


def load_case(root, case):
    cell = root / 'cells' / case['id']
    manifest, result = read(cell / 'manifest.json'), read(cell / 'result.json')
    cfg, terminal = manifest['resolved_config'], result['terminal_metrics']
    require(result['manifest_sha256'] == digest(cell / 'manifest.json'), 'Manifest hash mismatch')
    artifact = cell / 'artifacts/comparison.json'
    hashes = [a['sha256'] for a in result['artifacts'] if a['path'] == 'artifacts/comparison.json']
    require(hashes == [digest(artifact)], 'Comparison artifact hash mismatch')
    require(not manifest['smoke'] and not result['smoke'], 'Smoke artifact')
    for key in ('criteria_met', 'compared_all_conv_weights', 'compared_all_selected_conv_weights'):
        require(result['completion'][key] is True, 'Completion: ' + key)
    require(result['completion']['official_test_read'] is False, 'Official test accessed')
    for key in ('checkpoint_bytes_unchanged', 'parameters_and_bn_unchanged', 'measure_layer_displacement'):
        require(terminal[key] is True, 'Terminal assertion: ' + key)
    require(terminal['checkpoint_epoch'] == manifest['checkpoint_epoch'] == case['epoch'], 'Checkpoint epoch')
    require(terminal['scheme'] == manifest['scheme'] == case['scheme'], 'Checkpoint scheme')
    require(terminal['checkpoint_file'] == manifest['checkpoint_file'] == cfg['checkpoint_file'], 'Checkpoint path')
    require(terminal['checkpoint_sha256'] == manifest['checkpoint_sha256'] ==
            cfg['input_sha256'][cfg['checkpoint_file']], 'Checkpoint hash declarations')
    for key in ('checkpoint_sha256',):
        if key in case:
            require(case[key] == terminal[key], 'Case checkpoint hash mismatch')
    require(manifest['environment'] == {'torch': '2.11.0+cu128', 'cuda': '12.8'}, 'Replay environment')
    require(any(gpu in manifest['gpu'] for gpu in ('RTX 5090', 'RTX 3090')), 'Replay GPU')
    require(cfg.get('precision') == manifest.get('precision') == terminal.get('precision') == 'float64',
            'Replay requires explicit float64 config, manifest and terminal precision')
    require(cfg.get('dtype', 'float64') == 'float64', 'Conflicting replay dtype')
    require(terminal.get('precision_checks_passed') is True,
            'Missing or failed full-float64 precision checks')
    require(cfg['base_betas'] == [1, 1, 1] and cfg['batch_size'] == 32 and cfg['noise'] == 0, 'Replay config')
    require(cfg['parameters_and_bn_read_only'] is True, 'Read-only config')
    require(cfg['displacement_calibration']['relative_tolerance'] == .03
            and cfg['displacement_calibration']['max_evaluations'] == 24, 'Calibration contract')
    require(tuple(cfg['relative_rms_targets']) == tuple(terminal['relative_rms_targets']) == TARGETS, 'Target grid')
    require(terminal['examples'] == 256 and terminal['reference_iterations'] == [6, 6, 4], 'Replay cohort/T')
    data = read(artifact)
    pooled, batches = index_rows(data['layers']), index_rows(data['batches'], batch=True)
    require(terminal['layers'] == data['layers'], 'Terminal rows differ from artifact')
    for key, row in pooled.items():
        bs = [batches[(*key, n)] for n in range(8)]
        for field in SUMS:
            require(close(row['state_displacement'][field], math.fsum(r['state_displacement'][field] for r in bs)), f'Raw-sum pooling {key}/{field}')
        for field in ('output_force_rms', 'injected_force_rms'):
            expected = math.sqrt(math.fsum(r[field]**2 * r['force_elements'] for r in bs) / sum(r['force_elements'] for r in bs))
            require(close(row[field], expected), f'Force pooling {key}/{field}')
        require(all(close(r['beta_injected_B'], row['beta_injected_B']) for r in bs), f'Batch beta differs {key}')
        require(close(row['beta_multiplier'], terminal['beta_multiplier_vectors'][key[0]][key[1]-1]), f'Terminal beta differs {key}')
    return dict(case=case, pooled=pooled, batches=batches, provenance=dict(
        gpu=manifest['gpu'], environment=manifest['environment'],
        checkpoint_sha256=terminal['checkpoint_sha256'],
        cohort_sha256=cfg['input_sha256']['inputs/cohort.json'],
        source_sha256=manifest['source_identity']['source_sha256'],
        manifest_sha256=result['manifest_sha256'], comparison_sha256=hashes[0]))


def load(root, allow_partial):
    cases = read(root / 'cases.json')
    require(isinstance(cases, list), 'cases.json must be a list')
    expected = expected_cases()
    require(len(cases) == len(expected) and {c['id'] for c in cases} == set(expected), 'Expected all 18 unique physical cases in inventory')
    for case in cases:
        require((case['scheme'], case['epoch'], case['trajectory']) == expected[case['id']], 'Case identity: ' + case['id'])
        require(set(case['groups']) == ({'eqprop', 'bptt'} if case['epoch'] == 0 else {case['trajectory']}), 'Group membership')
    loaded, excluded = {}, []
    for case in cases:
        try:
            loaded[case['id']] = load_case(root, case)
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            if not allow_partial:
                raise ValueError(f"{case['id']}: {exc}") from exc
            excluded.append(dict(id=case['id'], reason=f'{type(exc).__name__}: {exc}'))
    cohorts = {d['provenance']['cohort_sha256'] for d in loaded.values()}
    require(len(cohorts) <= 1, 'Different cohorts across cases')
    # Source inventories can contain different configs; compare the actual numerical sources.
    numerical_sources = ('experiments/compare_cifar_block_beta_gradients.py',
                         'experiments/calibrate_cifar_block_beta.py',
                         'experiments/train_cifar_l8_analog.py', 'labs/cifar_l8_analog.py',
                         'model/resistive/interaction.py', 'training/sgd.py')
    source_checks = {}
    for path in numerical_sources:
        values = [d['provenance']['source_sha256'].get(path) for d in loaded.values()]
        require(not values or (None not in values and len(set(values)) == 1),
                'Numerical source missing or differs across cases: ' + path)
        source_checks[path] = values[0] if values else None
    return cases, loaded, excluded, source_checks


def flatten(case, row):
    record = dict(physical_case=case['id'], scheme=case['scheme'], trajectory=case['trajectory'],
                  epoch=case['epoch'], groups='|'.join(case['groups']), precision='float64')
    for field in ('case_index', 'block', 'layer', 'name', 'examples', 'beta_injected_B',
                  'beta_multiplier', 'target_relative_rms', 'force_elements', 'output_force_rms', 'injected_force_rms'):
        record[field] = row[field]
    if 'batch' in row:
        record['batch'] = row['batch']
    for field in ('state_displacement', 'output_displacement', *REFERENCES):
        record.update({field + '_' + k: v for k, v in row[field].items()})
    return record


def stats(values):
    defined = [v for v in values if v is not None]
    return dict(min=min(defined) if defined else None, median=median(defined) if defined else None,
                defined_count=len(defined), undefined_count=len(values)-len(defined))


def passes(row, criterion, reference='native_bptt'):
    g = row[reference]
    cosine, norm = g['cosine'], g['norm_ratio']
    if cosine is None or not g['eqprop_norm'] or not g['bptt_norm']:
        return False
    return cosine >= (.99 if criterion == 'cosine99' else .95) and (
        criterion != 'cosine95_norm90_110' or norm is not None and .9 <= norm <= 1.1)


def tables(loaded, reference='native_bptt'):
    pooled, batches = [], []
    for cid, data in loaded.items():
        case = data['case']
        initial = loaded.get(f"init-{case['scheme']}-e0")
        for key, row in sorted(data['pooled'].items()):
            record = flatten(case, row)
            record['diagnostic_reference'] = reference
            bs = [data['batches'][(*key, n)] for n in range(8)]
            for ref in ('native_bptt', 'local_bptt'):
                for metric in ('cosine', 'norm_ratio', 'eqprop_norm', 'bptt_norm'):
                    record.update({f'batch_{ref}_{metric}_{k}': v for k, v in stats([r[ref][metric] for r in bs]).items()})
            for criterion in CRITERIA:
                record['pooled_pass_' + criterion] = passes(row, criterion, reference)
                record['batch_pass_count_' + criterion] = sum(passes(r, criterion, reference) for r in bs)
            for ref, metric in (('native_bptt', 'eqprop_norm'), ('native_bptt', 'bptt_norm'), ('local_bptt', 'bptt_norm')):
                record[f'{ref}_{metric}_vs_init_matched_target'] = ratio(row[ref][metric], initial['pooled'][key][ref][metric]) if initial else None
            pooled.append(record)
            for n, br in enumerate(bs):
                flat = flatten(case, br)
                flat['diagnostic_reference'] = reference
                for ref, metric in (('native_bptt', 'eqprop_norm'), ('native_bptt', 'bptt_norm'), ('local_bptt', 'bptt_norm')):
                    flat[f'{ref}_{metric}_vs_init_matched_target'] = ratio(br[ref][metric], initial['batches'][(*key, n)][ref][metric]) if initial else None
                batches.append(flat)
    return pooled, batches


def margins(row, criterion, reference='native_bptt'):
    g = row[reference]
    cm = g['cosine'] - (.99 if criterion == 'cosine99' else .95) if g['cosine'] is not None else None
    nm = min(g['norm_ratio'] - .9, 1.1 - g['norm_ratio']) if g['norm_ratio'] is not None else None
    return cm, nm


def sample_point(loaded, ids, block, t, criterion, reference='native_bptt'):
    items = [(cid, loaded[cid]['pooled'][t, block, layer]) for cid in ids for layer in range(1, DEPTHS[block-1]+1)]
    # Undefined gradients rank below any defined cosine and can never pass.
    worst_id, worst = min(items, key=lambda p: -2 if p[1][reference]['cosine'] is None else p[1][reference]['cosine'])
    cms, nms = zip(*(margins(row, criterion, reference) for _, row in items))
    batch_worst = []
    batch_pass = 0
    for n in range(8):
        rows = [loaded[cid]['batches'][t, block, layer, n] for cid in ids for layer in range(1, DEPTHS[block-1]+1)]
        cos = [r[reference]['cosine'] for r in rows]
        batch_worst.append(min(cos) if all(c is not None for c in cos) else None)
        batch_pass += all(passes(r, criterion, reference) for r in rows)
    candidates = []
    for cid in ids:
        row = loaded[cid]['pooled'][t, block, 1]
        candidates.append(dict(physical_case=cid, epoch=loaded[cid]['case']['epoch'], scheme=loaded[cid]['case']['scheme'],
                               beta=row['beta_injected_B'], achieved_output_relative_rms=row['output_displacement']['relative_rms']))
    cms_defined = all(v is not None for v in cms)
    nms_defined = all(v is not None for v in nms)
    min_cm = min(cms) if cms_defined else None
    min_nm = min(nms) if nms_defined else None
    limiting = min(items, key=lambda p: min(v for v in margins(p[1], criterion, reference) if v is not None)
                   if all(v is not None for v in margins(p[1], criterion, reference)) else -math.inf) if criterion == CRITERIA[2] else (worst_id, worst)
    return dict(target=TARGETS[t], sampled_index=t,
                passed=all(passes(row, criterion, reference) for _, row in items),
                worst_cosine=worst[reference]['cosine'], worst_checkpoint=worst_id, worst_layer=worst['name'],
                limiting_checkpoint=limiting[0], limiting_layer=limiting[1]['name'],
                cosine_margin=min_cm, norm_margin=min_nm,
                worst_criterion_margin=(min(min_cm, min_nm) if min_cm is not None and min_nm is not None else None)
                    if criterion == CRITERIA[2] else min_cm,
                undefined_layer_count=sum(r[reference]['cosine'] is None for _, r in items),
                batch_worst_cosine=stats(batch_worst), batch_all_layers_pass_count=batch_pass,
                achieved_output_relative_rms_min=min(c['achieved_output_relative_rms'] for c in candidates),
                achieved_output_relative_rms_max=max(c['achieved_output_relative_rms'] for c in candidates),
                checkpoint_betas=candidates)


def sampled_bands(points):
    bands, current = [], []
    for point in points:
        if point['passed']:
            current.append(point)
        elif current:
            bands.append(current)
            current = []
    if current:
        bands.append(current)
    return [dict(targets=[p['target'] for p in band],
                 achieved_min=min(p['achieved_output_relative_rms_min'] for p in band),
                 achieved_max=max(p['achieved_output_relative_rms_max'] for p in band)) for band in bands]


def summarize(cases, loaded, complete, reference='native_bptt'):
    scopes = []
    # Initial cases enter two labelled views, never a cross-group intersection.
    for group in EPOCHS:
        group_cases = [c for c in cases if group in c['groups']]
        for case in group_cases:
            scopes.append((group, 'checkpoint', case['scheme'], 'initialization' if case['epoch'] == 0 else 'trained', [case['id']]))
        for inclusion in ('trained_only', 'initialization_inclusive'):
            selected = [c for c in group_cases if inclusion != 'trained_only' or c['epoch'] > 0]
            for scheme in SCHEMES:
                scopes.append((group, 'across_epochs', scheme, inclusion, [c['id'] for c in selected if c['scheme'] == scheme]))
            scopes.append((group, 'across_schemes', 'all', inclusion, [c['id'] for c in selected]))
    summaries = []
    for group, scope, scheme, inclusion, expected in scopes:
        available = [cid for cid in expected if cid in loaded]
        missing = sorted(set(expected) - set(available))
        for block in range(1, 4):
            for criterion in CRITERIA:
                points = [sample_point(loaded, available, block, t, criterion, reference) for t in range(len(TARGETS))] if available else []
                best = max(points, key=lambda p: p['worst_criterion_margin'] if p['worst_criterion_margin'] is not None else -math.inf) if points else None
                passing = [p['target'] for p in points if p['passed']]
                certified = complete and not missing
                summaries.append(dict(group=group, scope=scope, scheme=scheme, inclusion=inclusion, diagnostic_reference=reference,
                    block=block, criterion=criterion, expected_cases=expected, available_cases=available, missing_cases=missing,
                    evidence_status='complete' if certified else 'partial_observed_cases_only',
                    shared_sampled_target_exists=bool(passing) if certified else None,
                    observed_passing_targets=passing, observed_sampled_bands=sampled_bands(points),
                    best_worst_case_compromise=best, points=points,
                    conclusion=('Sampled common targets pass.' if passing else 'No sampled common target passes.') if certified
                               else 'Partial evidence; no complete or universal claim.'))
    return summaries


def save_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def own_layer_figures(loaded, out, partial, reference='native_bptt'):
    for group, epochs in EPOCHS.items():
        for scheme in SCHEMES:
            fig, axes = plt.subplots(3, 3, figsize=(14, 10))
            for b, l in LAYERS:
                ax = axes[b-1, l-1]
                for i, epoch in enumerate(epochs):
                    cid = f'{"init" if epoch == 0 else group}-{scheme}-e{epoch}'
                    if cid not in loaded:
                        continue
                    data = loaded[cid]
                    rows = [data['pooled'][t, b, l] for t in range(len(TARGETS))]
                    color = plt.get_cmap('tab10')(i)
                    x = [r['state_displacement']['relative_rms'] for r in rows]
                    y = [r[reference]['cosine'] for r in rows]
                    ax.plot(x, y, 'o-', color=color, markersize=2.5, label=f'Epoch {epoch}: pooled')
                    bs = [stats([data['batches'][t, b, l, n][reference]['cosine'] for n in range(8)]) for t in range(len(TARGETS))]
                    ax.plot(x, [s['min'] for s in bs], ':', color=color, alpha=.6)
                    ax.plot(x, [s['median'] for s in bs], '--', color=color, alpha=.65, linewidth=.8)
                ax.set(xscale='log', ylim=(-1.03, 1.03), title=f'Block {b} / Conv {l}', xlabel='Achieved own-layer centered d/f', ylabel=f'Cosine vs {REFERENCE_LABELS[reference]}')
                ax.axhline(.95, color='grey', linewidth=.7)
                ax.axhline(.99, color='grey', linewidth=.7, linestyle=':')
                ax.grid(alpha=.15)
            axes[2, 2].axis('off')
            handles, labels = axes[0, 0].get_legend_handles_labels()
            axes[2, 2].legend(handles, labels, loc='upper left', frameon=False)
            axes[2, 2].text(0, .46, 'Solid: cosine of pooled vectors\nDotted: minibatch minimum\nDashed: minibatch median\nEight batches / 256 training images\nfloat64; descriptive same-cohort fit\nUndefined values leave gaps', transform=axes[2, 2].transAxes, va='top')
            fig.suptitle(f'{scheme} / {group.upper()} checkpoints: EqProp vs {REFERENCE_LABELS[reference]}' + (' [PARTIAL]' if partial else ''))
            fig.tight_layout(rect=(0, 0, 1, .96))
            fig.savefig(out / f'{group}_{scheme}_own_layer_cosine.jpg', dpi=170)
            plt.close(fig)


def output_figures(loaded, summaries, out, partial, reference='native_bptt'):
    for group, epochs in EPOCHS.items():
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for block in range(1, 4):
            ax = axes[0, block-1]
            for scheme in SCHEMES:
                for epoch in epochs:
                    cid = f'{"init" if epoch == 0 else group}-{scheme}-e{epoch}'
                    if cid not in loaded:
                        continue
                    points = [sample_point(loaded, [cid], block, t, 'cosine95', reference) for t in range(len(TARGETS))]
                    ax.plot([p['achieved_output_relative_rms_min'] for p in points], [p['worst_cosine'] for p in points],
                            marker='o', markersize=2, linewidth=1, color=COLORS[scheme],
                            linestyle=('--', '-', ':', '-.')[epochs.index(epoch)], label=f'{scheme} e{epoch}')
            ax.set(xscale='log', ylim=(-1.03, 1.03), title=f'Block {block}: worst pooled layer', ylabel=f'Minimum cosine vs {REFERENCE_LABELS[reference]}')
            ax.axhline(.95, color='grey', linewidth=.8)
            ax.axhline(.99, color='grey', linewidth=.8, linestyle=':')
            ax.grid(alpha=.2)
            band_ax = axes[1, block-1]
            selected = [s for s in summaries if s['group'] == group and s['block'] == block and
                        s['scope'] == 'across_schemes']
            labels = []
            for y, s in enumerate(selected):
                labels.append(s['inclusion'].replace('_', ' ') + '\n' + s['criterion'])
                # Every glyph is a sampled nominal target; horizontal whiskers show the
                # spread of achieved d/f across checkpoint calibrations, never a band.
                for p in s['points']:
                    lo, hi = p['achieved_output_relative_rms_min'], p['achieved_output_relative_rms_max']
                    mid = math.sqrt(lo * hi)
                    band_ax.errorbar(mid, y, xerr=[[mid-lo], [hi-mid]], marker='o' if p['passed'] else 'x',
                                     markersize=4 if p['passed'] else 2.5, color='#16823b' if p['passed'] else '#bbb', linewidth=.6)
            band_ax.set(xscale='log', yticks=range(len(labels)), yticklabels=labels,
                        xlabel='Achieved block-output centered d/f', title='Across schemes: sampled intersections')
            band_ax.grid(axis='x', alpha=.2)
            band_ax.tick_params(axis='y', labelsize=7)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=8)
        fig.suptitle(f'{group.upper()} checkpoints / EqProp vs {REFERENCE_LABELS[reference]} / common output d/f targets' + (' [PARTIAL; OBSERVED CASES ONLY]' if partial else ''))
        fig.text(.5, .053, 'Green dots: all observed layers/checkpoints pass; grey crosses: fail. Whiskers: achieved d/f spread at one target. No interpolation or independent validation.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .09, 1, .96))
        fig.savefig(out / f'{group}_output_worst_layer_and_common_targets.jpg', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--reference', choices=tuple(REFERENCE_LABELS), default='native_bptt',
                        help='Reference for plots and passing targets; local_bptt writes a separate analysis/local_bptt directory')
    parser.add_argument('--allow-partial', action='store_true', help='Exclude entire invalid/missing cases; prohibit complete/universal conclusions')
    args = parser.parse_args()
    root = args.root.resolve()
    cases, loaded, excluded, source_checks = load(root, args.allow_partial)
    complete = len(loaded) == 18 and not excluded
    pooled, batches = tables(loaded, args.reference)
    summaries = summarize(cases, loaded, complete, args.reference)
    out = root / ('analysis/partial' if not complete else 'analysis')
    if args.reference != 'native_bptt':
        out = out / args.reference
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / 'layer_metrics.csv', pooled)
    save_csv(out / 'batch_metrics.csv', batches)
    points_csv = []
    betas_csv = []
    for s in summaries:
        context = {k: s[k] for k in ('group', 'scope', 'scheme', 'inclusion', 'block', 'criterion', 'evidence_status', 'diagnostic_reference')}
        context['expected_cases'] = '|'.join(s['expected_cases'])
        for p in s['points']:
            flat = dict(context, **{k: v for k, v in p.items() if k not in ('checkpoint_betas', 'batch_worst_cosine')})
            flat.update({'batch_worst_cosine_' + k: v for k, v in p['batch_worst_cosine'].items()})
            points_csv.append(flat)
            for beta in p['checkpoint_betas']:
                betas_csv.append(dict(context, target=p['target'], all_observed_pass=p['passed'],
                                      best_compromise=p is s['best_worst_case_compromise'], **{('checkpoint_scheme' if k == 'scheme' else k): v for k, v in beta.items()}))
    save_csv(out / 'common_target_points.csv', points_csv)
    save_csv(out / 'candidate_betas.csv', betas_csv)
    summary = dict(study='exp-025', evidence_class='read-only finite-iteration diagnostic', precision='float64',
        diagnostic_reference=args.reference, reference_label=REFERENCE_LABELS[args.reference],
        precision_policy='float64_model_states_forcing_and_gradients',
        complete=complete, expected_physical_cases=18, loaded_physical_cases=len(loaded),
        labelled_group_views=sum(len(d['case']['groups']) for d in loaded.values()), excluded_cases=excluded,
        physical_pooled_rows=len(pooled), physical_batch_rows=len(batches), targets=TARGETS,
        notes=[
            'Local comparisons fix each block input, initial free states and output loss force from the free pass; full-network BPTT advances all blocks before evaluating its loss.',
            'EqProp-trained and BPTT-trained checkpoint groups are never intersected or pooled together.',
            'Initialization is one physical replay per scheme with two labelled group views.',
            'Gradients are example-weighted vector means in the source artifact; scalar batch cosines/norms cannot reconstruct pooled vectors.',
            'Displacement and force RMS pooling is independently checked against raw sums/counts.',
            'Eight minibatches use the same 256 calibration images; legacy calibration/confirmation row labels do not establish independent validation.',
            'Shared targets refer to the nominal grid with achieved output d/f within 3%; plots use achieved d/f.',
            'Passing bands are contiguous sampled grid points only; unsampled gaps and interpolated continuous intervals are not certified.',
            'Checkpoint immutability/model/BN checks are runner terminal assertions, cross-checked against manifest; tensors are not loaded by this analyzer.',
            'Initialization tensor equivalence, epoch5 parent identity and source-to-input provenance are preparation/collection responsibilities.',
            'Full float64 is the fixed replay contract: float64_model_states_forcing_and_gradients. Float32 artifacts are rejected.',
            'Matched-target scale ratios compare each gradient tensor to initialization at the same target, with separately calibrated beta.',
        ],
        validation=dict(validation_scope='Loaded cases only; exclusions listed above',
                        terminal_precision_checks_passed=True,
                        unique_full_case_layer_batch_coverage=True, raw_sum_pooling=True,
                        references_invariant_across_targets=True, final_state_matches_block_output=True,
                        target_tolerance=.03, numerical_source_hashes=source_checks),
        provenance={cid: d['provenance'] for cid, d in loaded.items()}, intersections=summaries)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if loaded:
        own_layer_figures(loaded, out, not complete, args.reference)
        output_figures(loaded, summaries, out, not complete, args.reference)
    print(json.dumps(dict(complete=complete, loaded_cases=len(loaded), excluded_cases=excluded,
                         diagnostic_reference=args.reference, pooled_rows=len(pooled), batch_rows=len(batches), output=str(out)), indent=2))


if __name__ == '__main__':
    main()
