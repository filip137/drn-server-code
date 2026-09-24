"""Aggregate the compact trained-checkpoint T/K cosine comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.reporting import validate_run

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / 'results/eqprop-conv3-cosine-tk-schemes-20260922-v1'
CONFIGS = ROOT / 'configs/conv/eqprop_conv3_cosine_tk_schemes_20260922_v1'
SCHEMES = ['baseline', 'ours', 'legacy']
GRID = [8, 16, 32]
PARAMETERS = ['ConvWeight_0', 'ConvWeight_1', 'ConvWeight_2', 'DenseWeight_0']
LAYERS = ['Conv1', 'Conv2', 'Conv3', 'Readout']
SEEDS = [2026092101, 2026092111, 2026092121, 2026092131]


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmaps(records, directory, condition, partial):
    lookup = {(r['scheme'], r['T'], r['K'], r['parameter_name']): r
              for r in records if r['condition'] == condition}
    fig, axes = plt.subplots(3, 4, figsize=(12.5, 8))
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad('#eeeeee')
    for si, scheme in enumerate(SCHEMES):
        for pi, (parameter, label) in enumerate(zip(PARAMETERS, LAYERS)):
            ax = axes[si, pi]
            values = np.asarray([[lookup.get((scheme, t, k, parameter), {}).get('median_cosine', np.nan)
                                  for k in GRID] for t in GRID])
            im = ax.imshow(values, vmin=-1, vmax=1, cmap=cmap, aspect='equal')
            for ti, t in enumerate(GRID):
                for ki, k in enumerate(GRID):
                    value = values[ti, ki]
                    text = f'{value:.3f}' if np.isfinite(value) else '—'
                    ax.text(ki, ti, text, ha='center', va='center', fontsize=10,
                            color='white' if np.isfinite(value) and abs(value) > .65 else 'black')
            ax.set(xticks=range(3), xticklabels=GRID, yticks=range(3), yticklabels=GRID, xlabel='K')
            if pi == 0:
                ax.set_ylabel(f'{scheme}\nT')
            if si == 0:
                ax.set_title(label)
    title = ('PARTIAL · ' if partial else '') + f'{condition.capitalize()} median EP–BPTT cosine'
    fig.suptitle(title + '\nEpoch 30 · training σ = 5×10⁻⁴ · each scheme’s training β · same 576 examples', y=.98)
    fig.subplots_adjust(left=.075, right=.90, bottom=.07, top=.87, wspace=.28, hspace=.45)
    color_ax = fig.add_axes([.925, .17, .018, .60])
    fig.colorbar(im, cax=color_ax, label='Cosine')
    for suffix in ['png', 'pdf']:
        fig.savefig(directory / f'{condition}_cosine_TK.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_slices(records, directory, varied_axis, partial):
    lookup = {(r['scheme'], r['T'], r['K'], r['parameter_name']): r
              for r in records if r['condition'] == 'noisy'}
    fixed_axis = 'K' if varied_axis == 'T' else 'T'
    fig, axes = plt.subplots(4, 3, figsize=(12.5, 10), sharex=True, sharey=True)
    for pi, (parameter, layer) in enumerate(zip(PARAMETERS, LAYERS)):
        for si, scheme in enumerate(SCHEMES):
            ax = axes[pi, si]
            for fixed, color in zip(GRID, ['#0072b2', '#d55e00', '#009e73']):
                keys = [(scheme, x, fixed, parameter) if varied_axis == 'T'
                        else (scheme, fixed, x, parameter) for x in GRID]
                selected = [lookup.get(key, {}) for key in keys]
                ax.plot(GRID, [r.get('median_cosine', np.nan) for r in selected],
                        marker='o', color=color, label=f'{fixed_axis}={fixed}')
                ax.fill_between(GRID, [r.get('p10_cosine', np.nan) for r in selected],
                                [r.get('p90_cosine', np.nan) for r in selected], color=color, alpha=.10)
            ax.axhline(0, color='gray', linewidth=.7)
            ax.set(xticks=GRID, ylim=(-1.03, 1.03))
            ax.grid(alpha=.18)
            if pi == 0:
                ax.set_title(scheme)
            if si == 0:
                ax.set_ylabel(f'{layer}\nCosine')
            if pi == 3:
                ax.set_xlabel(varied_axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .965), ncol=3, frameon=False)
    fig.suptitle(('PARTIAL · ' if partial else '') + f'Noisy cosine versus {varied_axis} at fixed {fixed_axis}'
                 + '\nTraining/read σ = 5×10⁻⁴ · each training β · bands: 10–90% of batches/draws', y=1.01)
    fig.tight_layout(rect=[0, 0, 1, .92])
    for suffix in ['png', 'pdf']:
        fig.savefig(directory / f'cosine_vs_{varied_axis}_slices.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    study = args.study_root.resolve()
    analysis = study / ('analysis_partial' if args.partial else 'analysis')
    analysis.mkdir(exist_ok=True)
    cached = json.loads((study / 'cached_cells.json').read_text())
    sources = [{**cell, 'cached': True, 'run': ROOT / cell['run']} for cell in cached]
    cases = {}
    for config_path in sorted(CONFIGS.glob('T*_K*.json')):
        cfg = json.loads(config_path.read_text())
        for case in cfg['cases']:
            cases[case['scheme']] = case
            sources.append(dict(T=cfg['T'], K=cfg['K'], scheme=case['scheme'], cached=False,
                run=study / 'collected' / f'T{cfg["T"]}_K{cfg["K"]}' / 'runs' / (case['name'] + '_factor_1'),
                config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest()))
    assert len(sources) == 27
    assert len({(s['scheme'], s['T'], s['K']) for s in sources}) == 27
    records, included, missing, batch_identities = [], [], [], {}
    post_t_force_identities = {}
    for source in sorted(sources, key=lambda s: (SCHEMES.index(s['scheme']), s['T'], s['K'])):
        run, scheme, t, k = source['run'], source['scheme'], source['T'], source['K']
        if not (run / 'result.json').exists():
            missing.append(dict(scheme=scheme, T=t, K=k))
            continue
        assert not validate_run(run), run
        digest = hashlib.sha256((run / 'result.json').read_bytes()).hexdigest()
        if source['cached']:
            assert digest == source['result_sha256']
        else:
            manifest = json.loads((run / 'manifest.json').read_text())
            assert manifest['configuration']['sha256'] == source['config_sha256']
        rows = list(csv.DictReader((run / 'layer_metrics.csv').open()))
        assert len(rows) == 720
        guards = json.loads((run / 'read_only_guards.json').read_text())
        assert len(guards) == 36
        for batch, guard in enumerate(guards):
            identity = guard['common_post_T_state_sha256'], guard['frozen_current_force_sha256']
            assert post_t_force_identities.setdefault((scheme, t, batch), identity) == identity
            assert guard['bptt_start_state_sha256'] == identity[0]
            for field in ['bptt_gradient_iterations_K', 'bptt_minimizer_iterations_K',
                          'eqprop_minimizer_iterations_K']:
                assert guard[field] == k, (run, field)
            assert guard['frozen_current_force_unchanged_across_phases']
        identity_set = set()
        for row in rows:
            assert (int(row['T']), int(row['K']), row['scheme']) == (t, k, scheme)
            assert row['checkpoint_sha256'] == cases[scheme]['source_hashes']['final_model.pt']
            assert int(row['checkpoint_epoch']) == 30 and float(row['training_sigma']) == .0005
            assert float(row['injected_beta']) == cases[scheme]['beta'] and float(row['beta_factor']) == 1
            batch, draw = int(row['batch_index']), int(row['noise_draw_index'])
            assert batch in range(36) and draw in range(5)
            identity = (row['batch_payload_sha256'], row['batch_source_indices_sha256'])
            assert batch_identities.setdefault(batch, identity) == identity
            assert float(row['endpoint_read_noise_std']) == (0 if draw == 0 else .0005)
            assert row['endpoint_read_noise_seed'] == ('' if draw == 0 else str(SEEDS[draw - 1]))
            cell_identity = batch, draw, row['parameter_name']
            assert cell_identity not in identity_set
            identity_set.add(cell_identity)
            assert np.isfinite(float(row['cosine']))
        assert len(batch_identities) == 36
        for parameter, label in zip(PARAMETERS, LAYERS):
            for condition in ['clean', 'noisy']:
                selected = [r for r in rows if r['parameter_name'] == parameter
                            and (int(r['noise_draw_index']) == 0) == (condition == 'clean')]
                values = np.asarray([float(r['cosine']) for r in selected])
                assert len(values) == (36 if condition == 'clean' else 144)
                records.append(dict(scheme=scheme, T=t, K=k, parameter_name=parameter, layer=label,
                    condition=condition, injected_beta=cases[scheme]['beta'], comparisons=len(values),
                    median_cosine=float(np.median(values)), p10_cosine=float(np.quantile(values, .1)),
                    p90_cosine=float(np.quantile(values, .9)), minimum_cosine=float(values.min()),
                    positive_fraction=float((values > 0).mean()),
                    median_bptt_l2=float(np.median([float(r['bptt_l2']) for r in selected])),
                    median_eqprop_l2=float(np.median([float(r['eqprop_l2']) for r in selected]))))
        included.append(dict(scheme=scheme, T=t, K=k, cached=source['cached'],
                             run=str(run.relative_to(ROOT)), result_sha256=digest))
    if missing and not args.partial:
        raise RuntimeError(f'Incomplete declared grid: {missing}')
    assert records
    write_csv(analysis / 'layer_summary.csv', records)
    lookup = {(r['scheme'], r['T'], r['K'], r['parameter_name'], r['condition']): r for r in records}
    effects = []
    for scheme in SCHEMES:
        for fixed in GRID:
            for parameter, label in zip(PARAMETERS, LAYERS):
                for condition in ['clean', 'noisy']:
                    for axis in ['T', 'K']:
                        low_key = (scheme, 8, fixed, parameter, condition) if axis == 'T' else (scheme, fixed, 8, parameter, condition)
                        high_key = (scheme, 32, fixed, parameter, condition) if axis == 'T' else (scheme, fixed, 32, parameter, condition)
                        if low_key in lookup and high_key in lookup:
                            low, high = lookup[low_key]['median_cosine'], lookup[high_key]['median_cosine']
                            effects.append(dict(scheme=scheme, layer=label, condition=condition,
                                varied_axis=axis, fixed_axis_value=fixed, from_value=8, to_value=32,
                                from_median=low, to_median=high, median_change=high - low))
    if effects:
        write_csv(analysis / 'cosine_changes.csv', effects)
    for condition in ['noisy', 'clean']:
        plot_heatmaps(records, analysis, condition, bool(missing))
    for axis in ['T', 'K']:
        plot_slices(records, analysis, axis, bool(missing))
    report = ['# Conv3 cosine across T and K', '',
        f'Coverage: {len(included)}/27 declared cells; {sum(not r["cached"] for r in included)}/22 new, '
        f'{sum(r["cached"] for r in included)}/5 cached.', '',
        'Final epoch-30 seed-0 baseline/ours/legacy checkpoints trained at sigma 5e-4. '
        'Injected beta stays at each scheme’s training value: baseline 404.141105702, '
        'ours 5.26875648112, legacy 4.42250110273. T and K each span 8/16/32.', '',
        'Each cell compares centered frozen-current EP after K steps per phase with '
        'BPTT through K zero-nudge steps from the same post-T state. K changes both '
        'estimators; the reference is not a single fixed equilibrium gradient.', '',
        '![Noisy layer cosines](noisy_cosine_TK.png)', '',
        '![Clean layer cosines](clean_cosine_TK.png)', '',
        '[K curves at fixed T](cosine_vs_K_slices.png) · [T curves at fixed K](cosine_vs_T_slices.png)', '',
        '## Noisy layer medians', '', '| Scheme | T | K | Conv1 | Conv2 | Conv3 | Readout |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for source in included:
        scheme, t, k = source['scheme'], source['T'], source['K']
        values = [lookup[scheme, t, k, p, 'noisy']['median_cosine'] for p in PARAMETERS]
        report.append(f'| {scheme} | {t} | {k} | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    report += ['', 'Medians use 36 matched batches × four read draws (144 comparisons per '
        'layer). Clean controls use the same 36 batches. CSVs include 10–90% ranges '
        'and positive fractions; these describe batches/draws, not uncertainty across training seeds.', '',
        'All comparisons use the same 576 validation examples and deterministic read-noise '
        'seeds, exact-zero biases, explicit perfect diodes, float64 computations and unchanged '
        'source weights. Residuals do not filter or select these points. No training, accuracy '
        'evaluation or official-test read occurs. A finite grid does not establish an exact '
        'minimum T/K or gradient convergence.', '',
        '[Layer measurements](layer_summary.csv) · [Cosine changes](cosine_changes.csv) · '
        '[Validation](validation.json)']
    (analysis / 'report.md').write_text('\n'.join(report) + '\n')
    validation = dict(state='partial' if missing else 'complete', expected_cells=27,
        included_cells=len(included), new_cells=sum(not s['cached'] for s in included),
        cached_cells=sum(s['cached'] for s in included), layer_comparisons=720 * len(included),
        matched_batches=len(batch_identities), identical_post_T_state_and_force_groups=len(post_t_force_identities),
        verified_EP_and_BPTT_K=True, missing_cells=missing, sources=included,
        official_test_read=False, optimizer_steps_applied=False, accuracy_evaluation=False)
    (analysis / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(dict(included_cells=len(included), missing_cells=len(missing),
                         noisy_rows=[r for r in records if r['condition'] == 'noisy']), indent=2))


if __name__ == '__main__':
    main()
