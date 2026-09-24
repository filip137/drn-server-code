"""Summarize beta/K replay against both matching-K and fixed-K references."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.reporting import atomic_write_json, sha256_file, validate_run
from experiments.replay_conv3_trained_beta_noise import replay_contexts, cell_identity
from experiments.summarize_conv3_trained_beta_noise import read_csv, write_csv, quantiles

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ['ConvWeight_0', 'ConvWeight_1', 'ConvWeight_2', 'DenseWeight_0']
LABELS = ['Conv1', 'Conv2', 'Conv3', 'Readout']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    study = ROOT/cfg['output_root']
    out = study/('analysis_partial' if args.partial else 'analysis')
    out.mkdir(exist_ok=True)
    summary, residuals, signals, included, missing = [], [], [], [], []
    batches, references, fixed_references, cohorts, targets = {}, {}, {}, set(), set()
    metrics = ['cosine', 'relative_l2_difference_over_bptt', 'eqprop_over_bptt_norm_ratio',
               'bptt_rms', 'eqprop_rms', 'added_noise_rms', 'noise_over_clean_norm',
               'bptt_near_zero_fraction', 'eqprop_near_zero_fraction']
    metrics += [prefix+'_'+metric for prefix in ['bptt_vs_reference',
                'clean_ep_vs_reference_ep', 'ep_vs_reference_bptt']
                for metric in ['cosine', 'relative_l2', 'norm_ratio']]
    for context in replay_contexts(cfg):
        for beta in cfg['injected_betas']:
            name = cell_identity(context, beta, absolute=True)
            run = study/'collected/runs'/name
            if not (run/'result.json').exists():
                missing.append(name)
                continue
            assert not validate_run(run), name
            manifest = json.loads((run/'manifest.json').read_text())
            terminal = json.loads((run/'result.json').read_text())['terminal_metrics']
            assert manifest['configuration']['sha256'] == sha256_file(args.config)
            assert manifest['replay_K'] == context['replay_K']
            assert manifest['replay_T'] == context['replay_T']
            assert terminal['replay_batches'] == 36
            for key in ['source_bytes_unchanged', 'all_parameter_tensors_unchanged',
                        'all_bias_tensors_exact_zero', 'bptt_invariant_across_beta', 'noise_draws_matched']:
                assert terminal[key], (name, key)
            assert not terminal['official_test_read'] and not terminal['optimizer_steps_applied']
            guards = json.loads((run/'read_only_guards.json').read_text())
            assert len(guards) == 36
            for g in guards:
                assert g['iteration_contract_passed']
                assert g['inference_minimizer_iterations_T'] == context['replay_T']
                assert g['bptt_minimizer_iterations_K'] == g['eqprop_minimizer_iterations_K'] == context['replay_K']
            cohorts.add(manifest['cohort_sha256'])
            targets.add(manifest['runtime']['target'])
            common = dict(checkpoint=context['name'], scheme=context['scheme'],
                          epoch=context['checkpoint_epoch'], T=context['replay_T'], K=context['replay_K'],
                          injected_beta=beta, reference_K=max(cfg['K_values']))
            rows = read_csv(run/'layer_metrics.csv')
            assert len(rows) == terminal['layer_comparisons'] == 720
            assert len({(r['batch_index'], r['noise_draw_index'], r['parameter_name']) for r in rows}) == 720
            grouped = defaultdict(list)
            for r in rows:
                assert int(r['K']) == context['replay_K'] and int(r['T']) == context['replay_T']
                checkpoint_file = 'final_model.pt' if context['checkpoint_role'] == 'final' else 'best_model.pt'
                assert r['checkpoint_sha256'] == context['source_hashes'][checkpoint_file]
                batch = r['batch_index']
                payload = (r['batch_payload_sha256'], r['batch_source_indices_sha256'])
                assert batches.setdefault(batch, payload) == payload
                key = (context['name'], context['replay_T'], batch, r['parameter_name'])
                assert fixed_references.setdefault(key, r['reference_bptt_sha256']) == r['reference_bptt_sha256']
                key += (context['replay_K'],)
                assert references.setdefault(key, r['bptt_gradient_sha256']) == r['bptt_gradient_sha256']
                grouped[(float(r['endpoint_read_noise_std']), r['parameter_name'])].append(r)
            for (sigma, parameter), group in grouped.items():
                assert len(group) == (144 if sigma else 36)
                row = dict(common, sigma=sigma, parameter=parameter, comparisons=len(group))
                for metric in metrics:
                    quantiles(row, metric, [r[metric] for r in group])
                summary.append(row)
            grouped = defaultdict(list)
            for r in read_csv(run/'equilibrium_residuals.csv'):
                grouped[(r['phase'], r['state_layer_name'], r['residual_mode'])].append(r)
            for (phase, layer, mode), group in grouped.items():
                assert len(group) == 36
                row = dict(common, phase=phase, layer=layer, mode=mode,
                           failed_batches=sum(r['gate_passed'] != 'True' for r in group))
                quantiles(row, 'batch_p90', [r['selected_max_p90'] for r in group])
                quantiles(row, 'batch_maximum', [r['selected_maximum'] for r in group])
                residuals.append(row)
            grouped = defaultdict(list)
            for r in read_csv(run/'state_signals.csv'):
                grouped[r['layer_index']].append(r)
            for layer, group in grouped.items():
                assert len(group) == 36
                row = dict(common, layer_index=int(layer))
                for metric in ['free_rms', 'centered_half_difference_rms', 'zero_free_rms',
                               'midpoint_minus_zero_rms', 'phase_activity_disagreement_fraction']:
                    quantiles(row, metric, [r[metric] for r in group])
                signals.append(row)
            included.append(dict(name=name, result_sha256=sha256_file(run/'result.json')))
    assert len(cohorts) == len(targets) == 1
    if not args.partial:
        assert not missing
        terminal = json.loads((study/'collected/summary.json').read_text())
        assert terminal['state'] == 'complete'
        assert len(included) == len(terminal['cells']) == cfg['expected_production_cells']
        assert sum(r['comparisons'] for r in summary) == cfg['expected_layer_comparisons']
    selections = []
    for checkpoint in cfg['cases']:
        for k in cfg['K_values']:
            for sigma in [0., .0005]:
                for metric in ['cosine', 'ep_vs_reference_bptt_cosine']:
                    for criterion in ['minimum_layer_median', 'conv3_median']:
                        candidates = []
                        for beta in cfg['injected_betas']:
                            rows = [r for r in summary if (r['checkpoint'], r['K'], r['sigma'], r['injected_beta']) ==
                                    (checkpoint['name'], k, sigma, beta)]
                            if len(rows) != 4:
                                continue
                            score = min(r[metric+'_median'] for r in rows) if criterion == 'minimum_layer_median' else next(
                                r[metric+'_median'] for r in rows if r['parameter'] == 'ConvWeight_2')
                            candidates.append((score, beta))
                        if candidates:
                            score, beta = max(candidates, key=lambda x: (x[0], -x[1]))
                            selections.append(dict(checkpoint=checkpoint['name'], K=k, sigma=sigma,
                                                   metric=metric, criterion=criterion, best_sampled_beta=beta, score=score))
    for name, rows in [('layer_summary', summary), ('residual_summary', residuals),
                       ('state_summary', signals), ('selected_points', selections)]:
        write_csv(out/(name+'.csv'), rows)
    for metric, ylabel, filename, sigma in [
            ('cosine', 'EqProp–BPTT cosine (matching K)', 'cosine_vs_k', .0005),
            ('ep_vs_reference_bptt_cosine', 'EqProp–BPTT(K64) cosine', 'fixed_reference_cosine_vs_k', .0005),
            ('clean_ep_vs_reference_ep_relative_l2', 'Clean EqProp error vs K64', 'eqprop_convergence', 0.),
            ('bptt_vs_reference_relative_l2', 'BPTT error vs K64', 'bptt_convergence', 0.)]:
        fig, axes = plt.subplots(3, 4, figsize=(15, 9), constrained_layout=True)
        for i, checkpoint in enumerate(cfg['cases']):
            for j, (parameter, label) in enumerate(zip(PARAMETERS, LABELS)):
                ax = axes[i, j]
                for beta in cfg['injected_betas']:
                    rows = sorted([r for r in summary if (r['checkpoint'], r['sigma'], r['parameter'], r['injected_beta']) ==
                                   (checkpoint['name'], sigma, parameter, beta)], key=lambda r: r['K'])
                    if not rows:
                        continue
                    x = [r['K'] for r in rows]
                    ax.plot(x, [r[metric+'_median'] for r in rows], 'o-', label=f'β={beta:.3g}')
                    ax.fill_between(x, [r[metric+'_p10'] for r in rows], [r[metric+'_p90'] for r in rows], alpha=.12)
                if i == 0:
                    ax.set_title(label)
                if j == 0:
                    ax.set_ylabel(checkpoint['name']+'\n'+ylabel)
                ax.set_xscale('log', base=2)
                ax.set_xticks(cfg['K_values'], [str(k) for k in cfg['K_values']])
                ax.set_xlabel('K')
                ax.grid(alpha=.2)
                if metric.endswith('relative_l2'):
                    ax.set_yscale('symlog', linthresh=1e-12)
                    upper = max((r[metric+'_p90'] for r in summary
                                 if r['checkpoint'] == checkpoint['name'] and
                                 r['sigma'] == sigma and r['parameter'] == parameter), default=0.)
                    ax.set_ylim(0, max(1e-11, 1.2*upper))
                else:
                    ax.set_ylim(-.15, 1.05)
        axes[0, -1].legend(fontsize=8)
        fig.suptitle(f'T=16 · σ={sigma:g} · median and 10–90% across matched replay batches/draws')
        for extension in ['png', 'pdf']:
            fig.savefig(out/(filename+'.'+extension), dpi=160)
        plt.close(fig)
    atomic_write_json(out/'verification.json', dict(complete=not missing, included=included, missing=missing,
        cells=len(included), layer_comparisons=sum(r['comparisons'] for r in summary),
        matched_batches=len(batches), cohort_sha256=next(iter(cohorts)), target=next(iter(targets)),
        residual_failed_batches=sum(r['failed_batches'] for r in residuals),
        fixed_reference_hashes_match=True, bptt_beta_invariance=True,
        official_test_read=False, optimizer_steps_applied=False))
    print(json.dumps(dict(cells=len(included), missing=len(missing), output=str(out))))


if __name__ == '__main__':
    main()
