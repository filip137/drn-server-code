"""Validate and plot the fixed-checkpoint noisy beta sweep."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.reporting import atomic_write_json, sha256_file, validate_run

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ('ConvWeight_0', 'ConvWeight_1', 'ConvWeight_2', 'DenseWeight_0')
LABELS = ('Conv 1', 'Conv 2', 'Conv 3', 'Readout')
SCHEMES = ('baseline', 'ours', 'legacy')


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantiles(row, key, values):
    finite = [float(v) for v in values if v not in (None, '') and np.isfinite(float(v))]
    row[key + '_defined_count'] = len(finite)
    for label, q in [('min', 0), ('p10', .1), ('median', .5), ('p90', .9), ('max', 1)]:
        row[key + '_' + label] = float(np.quantile(finite, q)) if finite else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    study = ROOT / cfg['output_root']
    collected = study / 'collected'
    out = study / ('analysis_partial' if args.partial else 'analysis')
    out.mkdir(exist_ok=True)
    groups, state_groups = defaultdict(list), defaultdict(list)
    sources, missing, targets = [], [], set()
    for case in cfg['cases']:
        for factor in cfg['beta_factors']:
            name = case['name'] + '_factor_' + f'{factor:g}'.replace('.', 'p')
            run = collected / 'runs' / name
            if not (run / 'result.json').exists():
                missing.append(name)
                continue
            errors = validate_run(run)
            assert not errors, (name, errors)
            manifest = json.loads((run / 'manifest.json').read_text())
            targets.add(manifest['runtime']['target'])
            assert manifest['configuration']['sha256'] == sha256_file(args.config)
            result = json.loads((run / 'result.json').read_text())
            terminal = result['terminal_metrics']
            assert terminal['replay_batches'] == cfg['expected_batches']
            assert terminal['layer_comparisons'] == cfg['expected_batches'] * 5 * 4
            assert terminal['checkpoint_epoch'] == 30 and terminal['checkpoint_role'] == 'final'
            assert terminal['source_bytes_unchanged'] and terminal['all_parameter_tensors_unchanged']
            assert terminal['bptt_invariant_across_beta'] and terminal['noise_draws_matched']
            assert not terminal['official_test_read'] and not terminal['optimizer_steps_applied']
            rows = read_csv(run / 'layer_metrics.csv')
            assert len(rows) == terminal['layer_comparisons']
            keys = [(r['batch_index'], r['noise_draw_index'], r['parameter_name']) for r in rows]
            assert len(set(keys)) == len(keys)
            for r in rows:
                assert int(r['T']) == cfg['T'] and int(r['K']) == cfg['K']
                assert r['checkpoint_sha256'] == case['source_hashes']['final_model.pt']
                reference_rms = float(r['bptt_rms'])
                r['added_noise_relative_l2'] = float(r['added_noise_rms']) / reference_rms if reference_rms else None
                groups[(case['name'], factor, float(r['endpoint_read_noise_std']), r['parameter_name'])].append(r)
            for r in read_csv(run / 'state_signals.csv'):
                state_groups[(case['name'], factor, int(r['layer_index']))].append(r)
            sources.append(dict(name=name, result_sha256=sha256_file(run / 'result.json'),
                source_checkpoint=case['bundle'] + '/final_model.pt',
                checkpoint_sha256=case['source_hashes']['final_model.pt'],
                elapsed_seconds=terminal['elapsed_seconds'], residual_failed_rows=terminal['residual_failed_rows']))
    if not args.partial:
        assert not missing, missing
        completed = json.loads((collected / 'summary.json').read_text())
        assert completed['state'] == 'complete' and len(completed['cells']) == cfg['expected_production_cells']
        assert completed['config_sha256'] == sha256_file(args.config)
        assert sum(len(v) for v in groups.values()) == cfg['expected_layer_comparisons']
    cases = {c['name']: c for c in cfg['cases']}
    summary = []
    metrics = ('cosine', 'eqprop_over_bptt_norm_ratio', 'relative_l2_difference_over_bptt',
               'bptt_rms', 'eqprop_rms', 'clean_eqprop_rms', 'added_noise_rms', 'noise_over_clean_norm',
               'added_noise_relative_l2', 'bptt_near_zero_fraction', 'eqprop_near_zero_fraction')
    for (checkpoint, factor, sigma, parameter), rows in sorted(groups.items()):
        assert len(rows) == cfg['expected_batches'] * (4 if sigma else 1)
        case = cases[checkpoint]
        row = dict(checkpoint=checkpoint, scheme=case['scheme'], training_sigma=case['sigma'],
            readout_sigma=sigma, parameter=parameter, beta_factor=factor,
            injected_beta=case['beta']*factor, training_beta=case['beta'], comparisons=len(rows))
        for metric in metrics:
            quantiles(row, metric, [r[metric] for r in rows])
        row['all_residual_gates_passed'] = all(r['all_endpoint_residual_gates_passed'] == 'True' for r in rows)
        row['fraction_cosine_above_0p90'] = sum(r['cosine'] != '' and float(r['cosine']) > .9 for r in rows)/len(rows)
        row['fraction_cosine_above_0p99'] = sum(r['cosine'] != '' and float(r['cosine']) > .99 for r in rows)/len(rows)
        summary.append(row)
    write_csv(out / 'layer_summary.csv', summary)
    t_comparisons = []
    if cfg.get('source_T8_study'):
        previous_summary = read_csv(ROOT / cfg['source_T8_study'] / 'analysis/layer_summary.csv')
        previous_by_key = {(r['checkpoint'], float(r['beta_factor']), float(r['readout_sigma']), r['parameter']): r
                           for r in previous_summary}
        for r in summary:
            old = previous_by_key[(r['checkpoint'], r['beta_factor'], r['readout_sigma'], r['parameter'])]
            t_comparisons.append(dict(checkpoint=r['checkpoint'], scheme=r['scheme'], training_sigma=r['training_sigma'],
                readout_sigma=r['readout_sigma'], parameter=r['parameter'], beta_factor=r['beta_factor'], injected_beta=r['injected_beta'],
                previous_T=8, replay_T=cfg['T'], K=cfg['K'], T8_median_cosine=float(old['cosine_median']),
                new_median_cosine=r['cosine_median'], median_cosine_change=r['cosine_median']-float(old['cosine_median'])))
        write_csv(out / 'cosine_change_with_T.csv', t_comparisons)
    state_summary = []
    for (checkpoint, factor, layer), rows in sorted(state_groups.items()):
        row = dict(checkpoint=checkpoint, beta_factor=factor, injected_beta=cases[checkpoint]['beta']*factor, layer_index=layer)
        for metric in ('free_rms', 'positive_free_rms', 'negative_free_rms', 'centered_half_difference_rms',
                       'zero_free_rms', 'midpoint_minus_zero_rms', 'positive_activity_change_fraction',
                       'negative_activity_change_fraction', 'phase_activity_disagreement_fraction'):
            quantiles(row, metric, [r[metric] for r in rows])
        state_summary.append(row)
    write_csv(out / 'state_summary.csv', state_summary)

    choices, layer_choices = [], []
    for case in cfg['cases']:
        noisy = [r for r in summary if r['checkpoint'] == case['name'] and r['readout_sigma'] > 0]
        if not noisy:
            continue
        eligible = [r for r in noisy if r['cosine_defined_count'] == r['comparisons']]
        scores = []
        for factor in cfg['beta_factors']:
            point = [r for r in eligible if r['beta_factor'] == factor]
            if len(point) == 4:
                scores.append((min(r['cosine_median'] for r in point), factor, point))
        if not scores:
            continue
        best = max(scores, key=lambda v: (v[0], -v[1]))
        ordered = {r['parameter']: r for r in best[2]}
        selected = dict(checkpoint=case['name'], scheme=case['scheme'], training_sigma=case['sigma'],
            best_shared_beta_factor=best[1], best_shared_injected_beta=case['beta']*best[1],
            minimum_layer_median_cosine=best[0],
            all_residual_gates_passed=all(r['all_residual_gates_passed'] for r in best[2]),
            at_tested_boundary=best[1] in (cfg['beta_factors'][0], cfg['beta_factors'][-1]),
            common_factors_median_gt_0p90=json.dumps([f for score, f, _ in scores if score > .90]),
            common_factors_median_gt_0p99=json.dumps([f for score, f, _ in scores if score > .99]))
        qualified = [item for item in scores if all(r['all_residual_gates_passed'] for r in item[2])]
        qualified_best = max(qualified, key=lambda v: (v[0], -v[1])) if qualified else None
        selected['best_residual_qualified_beta'] = case['beta']*qualified_best[1] if qualified_best else None
        selected['best_residual_qualified_minimum_layer_median_cosine'] = qualified_best[0] if qualified_best else None
        for i, parameter in enumerate(PARAMETERS):
            selected[f'layer_{i+1}_median_cosine'] = ordered[parameter]['cosine_median']
            selected[f'layer_{i+1}_median_norm_ratio'] = ordered[parameter]['eqprop_over_bptt_norm_ratio_median']
            selected[f'layer_{i+1}_median_noise_relative_l2'] = ordered[parameter]['added_noise_relative_l2_median']
            clean_selected = next(r for r in summary if r['checkpoint'] == case['name'] and r['readout_sigma'] == 0
                                  and r['parameter'] == parameter and r['beta_factor'] == best[1])
            selected[f'layer_{i+1}_median_clean_relative_l2'] = clean_selected['relative_l2_difference_over_bptt_median']
            points = [r for r in eligible if r['parameter'] == parameter]
            best_layer = max(points, key=lambda r: (r['cosine_median'], -r['beta_factor']))
            layer_choices.append(dict(checkpoint=case['name'], scheme=case['scheme'], training_sigma=case['sigma'],
                parameter=parameter, best_injected_beta=best_layer['injected_beta'], best_beta_factor=best_layer['beta_factor'],
                best_median_cosine=best_layer['cosine_median'], median_norm_ratio=best_layer['eqprop_over_bptt_norm_ratio_median'],
                all_residual_gates_passed=best_layer['all_residual_gates_passed'],
                factors_median_gt_0p90=json.dumps([r['beta_factor'] for r in points if r['cosine_median'] > .90]),
                factors_median_gt_0p99=json.dumps([r['beta_factor'] for r in points if r['cosine_median'] > .99])))
        choices.append(selected)
    write_csv(out / 'best_shared_beta.csv', choices)
    write_csv(out / 'best_per_layer_beta.csv', layer_choices)

    sigmas = sorted({c['sigma'] for c in cfg['cases']})
    colors = plt.get_cmap('tab10').colors
    for metric, ylabel, filename in [('cosine', 'EP–BPTT gradient cosine', 'cosine_vs_beta'),
            ('eqprop_over_bptt_norm_ratio', 'EP / BPTT gradient norm', 'norm_ratio_vs_beta')]:
        fig, axes = plt.subplots(len(sigmas), 3, figsize=(16, 8), sharex='col', sharey=True, squeeze=False)
        for ri, sigma in enumerate(sigmas):
            for ci, scheme in enumerate(SCHEMES):
                ax = axes[ri, ci]
                for li, parameter in enumerate(PARAMETERS):
                    points = sorted([r for r in summary if r['scheme'] == scheme and r['training_sigma'] == sigma
                                     and r['readout_sigma'] > 0 and r['parameter'] == parameter], key=lambda r:r['injected_beta'])
                    clean = sorted([r for r in summary if r['scheme'] == scheme and r['training_sigma'] == sigma
                                    and r['readout_sigma'] == 0 and r['parameter'] == parameter], key=lambda r:r['injected_beta'])
                    ax.plot([r['injected_beta'] for r in points], [r[metric+'_median'] for r in points], 'o-',
                            color=colors[li], label=LABELS[li], markersize=3)
                    ax.fill_between([r['injected_beta'] for r in points], [r[metric+'_p10'] for r in points],
                                    [r[metric+'_p90'] for r in points], color=colors[li], alpha=.10)
                    ax.plot([r['injected_beta'] for r in clean], [r[metric+'_median'] for r in clean], ':', color=colors[li], alpha=.85)
                ax.axvline(next(c['beta'] for c in cfg['cases'] if c['scheme'] == scheme), color='black', lw=.8, ls='--')
                ax.set_xscale('log')
                if metric == 'cosine':
                    ax.set_ylim(-1.03, 1.03)
                    ax.axhline(.9, color='gray', lw=.6, alpha=.6)
                else:
                    ax.set_yscale('log')
                    ax.axhline(1, color='gray', lw=.6)
                ax.grid(alpha=.15)
                ax.set_title(f'{scheme} · training/read σ = {sigma:g}')
                if ri == len(sigmas)-1:
                    ax.set_xlabel('Injected β')
                if ci == 0:
                    ax.set_ylabel(ylabel)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=4, frameon=False)
        prefix = f'PARTIAL {len(sources)}/{cfg["expected_production_cells"]} cells · ' if missing else ''
        fig.suptitle(prefix + f'Epoch-30 checkpoints · T={cfg["T"]}, K={cfg["K"]} · solid: noisy · dotted: clean · dashed: training β', y=.945, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .91))
        for extension in ('png', 'pdf'):
            fig.savefig(out / f'{filename}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    if cfg.get('source_T8_study'):
        previous = read_csv(ROOT / cfg['source_T8_study'] / 'analysis/layer_summary.csv')
        fig, axes = plt.subplots(len(sigmas), 3, figsize=(16, 8), sharex='col', sharey=True, squeeze=False)
        for ri, sigma in enumerate(sigmas):
            for ci, scheme in enumerate(SCHEMES):
                ax = axes[ri, ci]
                for li, parameter in enumerate(PARAMETERS):
                    points = sorted([r for r in summary if r['scheme'] == scheme and r['training_sigma'] == sigma
                                     and r['readout_sigma'] > 0 and r['parameter'] == parameter], key=lambda r:r['injected_beta'])
                    old_points = sorted([r for r in previous if r['scheme'] == scheme and float(r['training_sigma']) == sigma
                                         and float(r['readout_sigma']) > 0 and r['parameter'] == parameter
                                         and float(r['beta_factor']) in cfg['beta_factors']], key=lambda r:float(r['injected_beta']))
                    ax.plot([r['injected_beta'] for r in points], [r['cosine_median'] for r in points], 'o-',
                            color=colors[li], label=LABELS[li], markersize=3)
                    ax.plot([float(r['injected_beta']) for r in old_points], [float(r['cosine_median']) for r in old_points],
                            '--', color=colors[li], alpha=.55)
                ax.set_xscale('log')
                ax.set_ylim(-1.03, 1.03)
                ax.grid(alpha=.15)
                ax.set_title(f'{scheme} · training/read σ = {sigma:g}')
                if ri == len(sigmas)-1:
                    ax.set_xlabel('Injected β')
                if ci == 0:
                    ax.set_ylabel('Noisy EP–BPTT gradient cosine')
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=4, frameon=False)
        fig.suptitle(f'Matched beta grid · solid: T={cfg["T"]} · dashed: T=8 · K={cfg["K"]} throughout', y=.945, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .91))
        for extension in ('png', 'pdf'):
            fig.savefig(out / f'cosine_T8_vs_T{cfg["T"]}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    validation = dict(state='partial' if missing else 'complete', completed_cells=len(sources),
        expected_cells=cfg['expected_production_cells'], missing_cells=missing,
        layer_comparisons=sum(len(v) for v in groups.values()), sources=sources,
        total_residual_failed_rows=sum(s['residual_failed_rows'] for s in sources),
        production_seconds=sum(s['elapsed_seconds'] for s in sources),
        recorded_targets=sorted(targets), replay_T=cfg['T'], replay_K=cfg['K'],
        official_test_read=False, optimizer_steps_applied=False)
    atomic_write_json(out / 'validation.json', validation)
    lines = ['# Trained Conv3 beta sweep under read noise', '',
        f"Coverage: {len(sources)}/{cfg['expected_production_cells']} cells; {validation['layer_comparisons']} layer comparisons.", '',
        'The objective is noisy gradient alignment at fixed final epoch-30 weights. Clean replay is a control.', '',
        '| Training/read σ | Scheme | Best tested shared injected β | Conv1 | Conv2 | Conv3 | Readout | Worst layer median |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in choices:
        lines.append(f"| {r['training_sigma']:g} | {r['scheme']} | {r['best_shared_injected_beta']:.6g} | " +
                     ' | '.join(f"{r[f'layer_{i}_median_cosine']:.4f}" for i in range(1, 5)) +
                     f" | {r['minimum_layer_median_cosine']:.4f} |")
    lines += ['', 'Shared beta maximizes the minimum of the four layer median noisy cosines over tested finite points; ties prefer smaller beta. This is a description of the sampled interval, not a training-beta recommendation.',
        'Per-layer maxima and sampled .90/.99 windows are in `best_per_layer_beta.csv`; norm ratios remain separate diagnostics.', '',
        'Solid lines show medians across36 matched batches ×4 draws; shading is the10–90% range, not a confidence interval. Dotted clean controls use36 batches.',
        f'All six checkpoints were trained on A100s. Recorded replay targets: {", ".join(sorted(targets))}. Source weights, zero biases, cohorts, frozen forces and BPTT invariance within each shard are checked.',
        f'This is a same-cohort exploratory sweep at T={cfg["T"]}, K={cfg["K"]}, not a training-beta selection or an exact-equilibrium guarantee. Per-layer independent beta choices do not constitute one physical EqProp update.',
        'Operational smokes and unsuccessful attempts are excluded from these aggregates; see the study plan for their inventory.', '',
        '![Noisy and clean gradient alignment](cosine_vs_beta.png)', '', '![Gradient norm ratios](norm_ratio_vs_beta.png)', '']
    cohort = json.loads((collected / 'cohort.json').read_text())
    lines += ['## Replay contract and sources', '',
        f"Cohort SHA-256: `{cohort['cohort_sha256']}`. The 576 examples come from the deterministic 55,000/5,000 ordinary-MNIST training/validation split; official-test reads and optimizer steps are zero.", '',
        f'Float64 centered frozen-current EqProp; T={cfg["T"]}, K={cfg["K"]}; input gain 360; paired 20-output squared loss; exact-zero biases; explicit perfect diodes; original [0,100] weights. BPTT differentiates exactly K zero-nudge steps from the same post-T state.', '',
        'Independent Gaussian endpoint reads use the four saved seeds, with matched standard-normal draws across schemes and beta. The physical phase states and clamped inputs remain clean.', '',
        'Every source listed below is the final epoch-30 checkpoint, selected by declared condition rather than accuracy.', '',
        '| Scheme | Training σ | Source checkpoint |', '|---|---:|---|']
    for case in cfg['cases']:
        lines.append(f"| {case['scheme']} | {case['sigma']:g} | [{case['name']}](../../../{case['bundle']}/final_model.pt) |")
    lines += ['', '[Layer summary](layer_summary.csv) · [Best shared beta](best_shared_beta.csv) · [Per-layer beta windows](best_per_layer_beta.csv) · [Displacement and diode activity](state_summary.csv) · [Coverage and hashes](validation.json)', '']
    if cfg.get('source_T8_study'):
        lines += ['## Cosine changes at the original training beta', '',
            'Each cell below shows the noisy median at T8 → the noisy median at the new T; K stays fixed.', '',
            '| Training/read σ | Scheme | Conv1 | Conv2 | Conv3 | Readout |', '|---:|---|---:|---:|---:|---:|']
        for case in cfg['cases']:
            points = {r['parameter']: r for r in t_comparisons if r['checkpoint'] == case['name']
                      and r['readout_sigma'] > 0 and r['beta_factor'] == 1}
            if len(points) == 4:
                lines.append(f"| {case['sigma']:g} | {case['scheme']} | " + ' | '.join(
                    f"{points[p]['T8_median_cosine']:.4f} → {points[p]['new_median_cosine']:.4f}" for p in PARAMETERS) + ' |')
        lines += ['', f'![Matched T8 versus T{cfg["T"]} comparison](cosine_T8_vs_T{cfg["T"]}.png)', '',
                  '[All matched cosine changes](cosine_change_with_T.csv)', '']
    (out / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in validation.items() if k not in ('sources', 'missing_cells')}, indent=2))
    print(json.dumps(choices, indent=2))


if __name__ == '__main__':
    main()
