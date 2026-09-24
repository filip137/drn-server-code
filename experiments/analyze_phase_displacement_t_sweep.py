"""Pool the T-only replay and separate zero-nudge drift from nudging response."""
from __future__ import annotations

from collections import defaultdict
import csv
import math
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiments.analyze_phase_displacement import pooled, csv_write
from experiments.run_phase_displacement_t_sweep import ROOT, OUT, SURFACES, checked, read, sha, write

ROLES = ('reconstructed_initialization', 'best_validation')
LAYERS = ('#266fb4', '#d27b19', '#399262', '#8854a6')


def figure(rows, metric, label, filename, *, threshold=None):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    plot_floor = min(r[metric] for r in rows if r[metric] > 0) / 10
    axis_label = {'raw_to_control_ratio': 'Raw / controlled RMS',
                  'drift_to_control_ratio': 'Drift / response RMS',
                  'controlled_rms': 'Response RMS [voltage units]',
                  'zero_drift_rms': 'Drift RMS [voltage units]'}[metric]
    for col, (architecture, scheme, ts, roles) in enumerate(SURFACES):
        depth = int(architecture[-1])
        for ri, role in enumerate(ROLES):
            ax = axes[ri, col]
            if role not in roles:
                ax.axis('off')
                ax.text(.05, .8, 'Conv2 legacy after training:\nraw/control RMS difference was <1%\nin the previous study; not replayed here.',
                        transform=ax.transAxes, va='top', fontsize=10)
                continue
            for layer in range(1, depth + 2):
                color = LAYERS[3 if layer == depth + 1 else layer - 1]
                for phase, style, marker in [('positive', '-', 'o'), ('negative', '--', 'v')]:
                    selected = sorted([r for r in rows if r['architecture'] == architecture
                        and r['scheme'] == scheme and r['checkpoint_role'] == role
                        and r['layer_index'] == layer and r['phase'] == phase], key=lambda r: r['T'])
                    # A zero value has no logarithm; display it at an explicit plotting floor.
                    yy = [r[metric] if r[metric] > 0 else plot_floor for r in selected]
                    ax.plot([r['T'] for r in selected], yy, style, marker=marker,
                            color=color, markersize=4, linewidth=1.3)
            ax.set(yscale='log', xticks=ts, xlabel='Free-phase iterations T')
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
            if threshold is not None:
                ax.axhline(threshold, color='black', linestyle=':', linewidth=1)
            if col == 0:
                ax.set_ylabel(('Initialization\n' if ri == 0 else 'Best BPTT checkpoint\n') + axis_label)
            if ri == 0:
                case = next(r for r in rows if r['architecture'] == architecture and r['scheme'] == scheme)
                ax.set_title(f'{architecture.capitalize()} {scheme} | K={case["K"]}, beta={case["injected_beta"]:g}', fontsize=10)
    handles = [Line2D([0], [0], color=c, label=n) for c, n in zip(LAYERS, ('H1', 'H2', 'H3', 'Output'))]
    handles += [Line2D([0], [0], color='black', linestyle=ls, marker=m, label=n)
                for ls, m, n in [('-', 'o', 'positive'), ('--', 'v', 'negative')]]
    fig.legend(handles=handles, ncol=6, loc='upper center', bbox_to_anchor=(.5, .925), fontsize=9)
    fig.suptitle(label + ' versus T\nSeed 0; same 576 validation examples; fixed checkpoint, K and beta within each curve', fontsize=11)
    note = 'Conv3 baseline beta 10 remains unconfirmed. T changes only for replay; no training setting adopted.'
    if threshold is not None:
        note += f' Dotted line: {threshold:g}.'
    if any(r[metric] == 0 for r in rows):
        note += f' Exact zeros shown at {plot_floor:.1e}.'
    fig.text(.5, .012, note, ha='center', fontsize=7.5)
    fig.tight_layout(rect=(0, .05, 1, .94))
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / 'analysis' / f'{filename}.{extension}', dpi=180)
    plt.close(fig)


def main():
    execution = read(OUT / 'execution.json')
    if execution['state'] != 'complete':
        raise ValueError('Wait for all declared cases.')
    expected = {(a, s, t) for a, s, ts, roles in SURFACES for t in ts}
    assert len(execution['runs']) == 15
    assert {(c['architecture'], c['scheme'], c['T']) for c in execution['runs']} == expected
    checked(execution['smoke'], smoke=True)
    if 'initial_smoke' in execution:
        checked(execution['initial_smoke'], smoke=True)
    assert read(OUT / 'smoke_regression.json')['shared_measurements_identical']
    analysis = OUT / 'analysis'
    analysis.mkdir(exist_ok=True)
    raw_rows, sources = [], []
    expected_guards = None
    total_replays = total_comparisons = 0
    for case in execution['runs']:
        verified = checked(case)
        assert verified['result_sha256'] == case['result_sha256']
        total_replays += verified['replay_count']
        total_comparisons += verified['layer_comparison_count']
        config = read(ROOT / case['path'])
        guards = config['dataset']['expected_batch_guards']
        if expected_guards is None:
            expected_guards = guards
        else:
            assert guards == expected_guards
        source = config['cases'][0]
        best_epoch = read(ROOT / source['run_dir'] / 'metrics.json')['best_epoch']
        directory = (ROOT / case['result']).parent
        if case['T'] == case['native_T']:
            assert case['native_regression']['shared_measurements_identical']
        for row in csv.DictReader((directory / 'state_displacement.csv').open()):
            if row['state_layer_name'] == '__all__':
                continue
            assert row['outcome'] == 'ok'
            guard = guards[int(row['batch_index'])]
            assert row['batch_payload_sha256'] == guard['payload_sha256']
            assert row['batch_source_indices_sha256'] == guard['source_indices_sha256']
            row.update(injected_beta=case['beta'], base_beta=case['base_beta'], model_seed=0,
                layer_index=int(re.search(r'(\d+)$', row['state_layer_name']).group(1)),
                source_checkpoint_epoch=0 if row['checkpoint_role'] == ROLES[0] else best_epoch,
                run_name=case['name'])
            raw_rows.append(row)
        sources.append(dict(architecture=case['architecture'], scheme=case['scheme'], T=case['T'], K=case['K'],
            injected_beta=case['beta'], base_beta=case['base_beta'], roles=';'.join(case['roles']),
            source_run=source['run_dir'], best_epoch=best_epoch,
            initializer=source['initializer_checkpoint_path'], initializer_sha256=source['initializer_checkpoint_sha256'],
            best_checkpoint_sha256=source['source_file_sha256']['best_model.pt'],
            result=case['result'], result_sha256=case['result_sha256'],
            gradient_pass=case['gradient_fidelity_all_passed'], equilibrium_pass=case['equilibrium_residual_all_passed']))
    assert total_replays == 900 and total_comparisons == 3420 and len(raw_rows) == 20520
    keys = ('architecture', 'scheme', 'checkpoint_role', 'T', 'phase', 'reference_kind', 'state_layer_name')
    groups = defaultdict(list)
    for row in raw_rows:
        groups[tuple(row[k] for k in keys)].append(row)
    summaries = []
    for key, rows in groups.items():
        assert len(rows) == 36 and len({r['batch_index'] for r in rows}) == 36
        r = rows[0]
        summary = dict(zip(keys, key), K=int(r['K']), layer_index=r['layer_index'],
            injected_beta=r['injected_beta'], base_beta=r['base_beta'], model_seed=0,
            source_checkpoint_epoch=r['source_checkpoint_epoch'], **pooled(rows))
        summary['T'] = int(summary['T'])
        summaries.append(summary)
    assert len(summaries) == 570
    lookup = {(r['architecture'], r['scheme'], r['checkpoint_role'], r['T'], r['layer_index'], r['phase'], r['reference_kind']): r for r in summaries}
    batch_lookup = {(r['architecture'], r['scheme'], r['checkpoint_role'], int(r['T']), r['layer_index'], r['batch_index'], r['phase'], r['reference_kind']): r for r in raw_rows}
    comparisons = []
    for z in summaries:
        if z['phase'] != 'zero':
            continue
        prefix = (z['architecture'], z['scheme'], z['checkpoint_role'], z['T'], z['layer_index'])
        max_t = max(ts for a, s, ts_grid, roles in SURFACES if a == z['architecture'] and s == z['scheme'] for ts in ts_grid)
        for phase in ('positive', 'negative'):
            raw = lookup[(*prefix, phase, 'post_T_free')]
            controlled = lookup[(*prefix, phase, 'matched_zero_K')]
            endpoint = lookup[(*prefix[:3], max_t, prefix[4], phase, 'matched_zero_K')]
            assert controlled['delta_rms'] > 0 and endpoint['delta_rms'] > 0
            assert raw['reference_rms'] == z['reference_rms']
            assert controlled['reference_rms'] == z['current_rms']
            tolerance = 1e-12 * max(raw['delta_rms'], controlled['delta_rms'], 1e-30)
            assert abs(raw['delta_rms'] - controlled['delta_rms']) <= z['delta_rms'] + tolerance
            assert z['delta_rms'] <= raw['delta_rms'] + controlled['delta_rms'] + tolerance
            batch_ratios = []
            for bi in range(36):
                batch_prefix = (*prefix, str(bi))
                d = float(batch_lookup[(*batch_prefix, 'zero', 'post_T_free')]['delta_rms'])
                n = float(batch_lookup[(*batch_prefix, phase, 'matched_zero_K')]['delta_rms'])
                assert n > 0
                batch_ratios.append(d / n)
            comparisons.append(dict(architecture=z['architecture'], scheme=z['scheme'], checkpoint_role=z['checkpoint_role'],
                T=z['T'], K=z['K'], injected_beta=z['injected_beta'], base_beta=z['base_beta'],
                layer_index=z['layer_index'], state_layer_name=z['state_layer_name'], phase=phase,
                raw_rms=raw['delta_rms'], controlled_rms=controlled['delta_rms'], zero_drift_rms=z['delta_rms'],
                raw_to_control_ratio=raw['delta_rms'] / controlled['delta_rms'],
                drift_to_control_ratio=z['delta_rms'] / controlled['delta_rms'],
                response_over_max_T_response=controlled['delta_rms'] / endpoint['delta_rms'], max_reference_T=max_t,
                free_state_rms=raw['reference_rms'], free_state_mean=raw['reference_mean'],
                batch_median_drift_to_control=float(np.median(batch_ratios)),
                batch_p95_drift_to_control=float(np.quantile(batch_ratios, .95)),
                batch_max_drift_to_control=max(batch_ratios),
                source_checkpoint_epoch=z['source_checkpoint_epoch']))
    assert len(comparisons) == 190
    qualifications = []
    for architecture, scheme, ts, roles in SURFACES:
        for role in roles:
            by_t = {}
            for t in ts:
                values = [r for r in comparisons if r['architecture'] == architecture and r['scheme'] == scheme
                          and r['checkpoint_role'] == role and r['T'] == t]
                by_t[t] = dict(worst_pooled_drift_to_control=max(r['drift_to_control_ratio'] for r in values),
                    worst_batch_drift_to_control=max(r['batch_max_drift_to_control'] for r in values),
                    maximum_controlled_response_change_from_max_T=max(abs(r['response_over_max_T_response'] - 1) for r in values))
            accepted = next((t for t in ts if all(by_t[u]['worst_pooled_drift_to_control'] <= .01 for u in ts if u >= t)), None)
            strict = next((t for t in ts if all(by_t[u]['worst_batch_drift_to_control'] <= .01 for u in ts if u >= t)), None)
            qualifications.append(dict(architecture=architecture, scheme=scheme, checkpoint_role=role,
                descriptive_threshold=.01, smallest_sustained_tested_T=accepted,
                smallest_sustained_all_batches_T=strict, by_T=by_t))
    csv_write(analysis / 'layer_batch_displacement.csv', raw_rows)
    csv_write(analysis / 'layer_displacement_summary.csv', summaries)
    csv_write(analysis / 'drift_response_comparison.csv', comparisons)
    csv_write(analysis / 'sources.csv', sources)
    figure(comparisons, 'raw_to_control_ratio', 'Raw / controlled RMS displacement', 'phase_t_raw_control', threshold=1)
    figure(comparisons, 'drift_to_control_ratio', 'Zero-nudge drift / controlled response RMS', 'phase_t_drift_fraction', threshold=.01)
    figure(comparisons, 'controlled_rms', 'Controlled nudging-response RMS [voltage units]', 'phase_t_controlled_response')
    figure(comparisons, 'zero_drift_rms', 'Zero-nudge drift RMS [voltage units]', 'phase_t_zero_drift')
    summary = dict(study_id=OUT.name, execution=execution, qualifications=qualifications,
        formal_cases=15, checkpoint_batch_replays=900, gradient_comparisons=3420,
        individual_layer_rows=20520, pooled_context_rows=570, comparison_rows=190,
        sources=sources, analysis_source_sha256=sha(__file__), pooling_helper_sha256=sha(ROOT / 'experiments/analyze_phase_displacement.py'),
        cohort_payloads_match=True, native_measurements_reproduced=True,
        recovery=execution.get('recovery'),
        training_started=False, official_test_read=False)
    write(analysis / 'summary.json', summary)
    for name in ('layer_displacement_summary.csv', 'drift_response_comparison.csv', 'sources.csv', 'summary.json'):
        shutil.copy2(analysis / name, ROOT / 'paper_ready_results' / f'phase_t_sweep_20260916_{name}')
    for stem in ('phase_t_raw_control', 'phase_t_drift_fraction', 'phase_t_controlled_response', 'phase_t_zero_drift'):
        for extension in ('png', 'pdf'):
            shutil.copy2(analysis / f'{stem}.{extension}', ROOT / 'paper_ready_results' / f'{stem}_20260916.{extension}')
    print('Validated15 cases;900 replays;20,520 individual-layer rows;190 drift/response comparisons.')
    for q in qualifications:
        print(q['architecture'], q['scheme'], q['checkpoint_role'], 'pooled T', q['smallest_sustained_tested_T'],
              'all-batch T', q['smallest_sustained_all_batches_T'])


if __name__ == '__main__':
    main()
