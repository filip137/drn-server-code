"""Validate and summarize the p99 checkpoint beta/T replay, without training."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.reporting import atomic_write_json, sha256_file, validate_run
from experiments.replay_conv3_trained_beta_noise import replay_contexts, cell_identity
from experiments.summarize_conv3_trained_beta_noise import read_csv, write_csv, quantiles

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ['ConvWeight_0', 'ConvWeight_1', 'ConvWeight_2', 'DenseWeight_0']
LABELS = ['Conv1', 'Conv2', 'Conv3', 'Readout']


def training_context(cfg, out):
    """Read existing matched training traces; do not construct an optimizer."""
    epochs, trace_summary, sources, batch_hashes = [], [], [], {}
    for case in cfg['cases']:
        if case['checkpoint_role'] != 'final':
            continue
        bundle = ROOT / case['bundle']
        assert not validate_run(bundle), bundle
        metrics = json.loads((bundle/'metrics.json').read_text())
        assert metrics['official_test_evaluations'] == 0
        rows = [json.loads(line) for line in (bundle/'metrics.jsonl').read_text().splitlines()]
        rows = [r for r in rows if r.get('kind') == 'epoch']
        assert [r['epoch'] for r in rows] == list(range(1,31))
        for r in rows:
            assert r['evaluation_split'] == 'validation'
            epochs.append(dict(scheme=case['scheme'],epoch=r['epoch'],**r['metrics']))
        groups = defaultdict(list)
        for line in (bundle/'gradient_trace.jsonl').read_text().splitlines():
            r = json.loads(line)
            if r['parameter_name'] not in PARAMETERS:
                continue
            key = (r['epoch'],r['batch'])
            assert batch_hashes.setdefault(key,r['source_indices_sha256']) == r['source_indices_sha256']
            groups[(r['epoch'],r['parameter_name'])].append(r)
        for (epoch,parameter), group in sorted(groups.items()):
            row = dict(scheme=case['scheme'],epoch=epoch,parameter=parameter,sampled_batches=len(group),
                       learning_rate=group[0]['learning_rate'])
            for metric in ['gradient_rms','applied_update_rms','applied_update_over_parameter_rms',
                           'parameter_rms_before','projection_changed_fraction',
                           'lower_bound_fraction_after','gradient_momentum_cosine']:
                quantiles(row,metric,[r[metric] for r in group])
            trace_summary.append(row)
        sources.append(dict(scheme=case['scheme'],bundle=case['bundle'],
                            file_sha256={f:sha256_file(bundle/f) for f in
                            ['metrics.json','metrics.jsonl','gradient_trace.jsonl']},
                            initialization_sha256=metrics['initial_parameter_state_sha256']))
    assert len({s['initialization_sha256'] for s in sources}) == 1
    write_csv(out/'training_epochs.csv',epochs)
    write_csv(out/'training_trace_summary.csv',trace_summary)
    atomic_write_json(out/'training_context_sources.json',dict(sources=sources,
        matched_sampled_training_batches=len(batch_hashes),new_optimizer_steps=False,official_test_read=False))
    weight_stats, weights = [], {}
    for case in cfg['cases']:
        filename='weights_final.npz' if case['checkpoint_role']=='final' else 'weights_best.npz'
        with np.load(ROOT/case['bundle']/filename) as saved:
            weights[case['name']]={name:saved[name].copy() for name in PARAMETERS}
        for parameter, values in weights[case['name']].items():
            weight_stats.append(dict(checkpoint=case['name'],parameter=parameter,
                rms=float(np.sqrt(np.mean(values**2))),exact_zero_fraction=float(np.mean(values==0)),
                maximum=float(values.max())))
    write_csv(out/'checkpoint_weight_statistics.csv',weight_stats)
    comparison=[]
    for parameter in PARAMETERS:
        legacy=weights['legacy_epoch30'][parameter]
        ours=weights['ours_epoch30'][parameter]
        comparison.append(dict(parameter=parameter,
            ours_over_legacy_rms=float(np.linalg.norm(ours)/np.linalg.norm(legacy)),
            between_scheme_weight_cosine=float(np.vdot(legacy,ours)/(np.linalg.norm(legacy)*np.linalg.norm(ours)))))
    write_csv(out/'checkpoint_weight_comparison.csv',comparison)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for scheme,color in [('legacy','#276FBF'),('ours','#D46A21')]:
        rows=[r for r in epochs if r['scheme']==scheme]
        axes[0].plot([r['epoch'] for r in rows],[100*r['validation_accuracy'] for r in rows],label=scheme,color=color)
        axes[1].plot([r['epoch'] for r in rows],[r['validation_loss'] for r in rows],label=scheme,color=color)
    axes[0].set_ylabel('Validation accuracy (%)')
    axes[1].set_ylabel('Validation paired squared-error loss')
    for ax in axes: ax.set_xlabel('Training epoch'); ax.grid(alpha=.15); ax.legend(frameon=False)
    fig.suptitle('Existing p99 training runs · seed0 · σ=5e-4 · T=K=8')
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(out/f'training_context.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


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
    training_context(cfg,out)
    summary, state_summary, residual_summary, included, missing = [], [], [], [], []
    reference_hashes, batch_hashes, targets, cohort_hashes = {}, {}, set(), set()
    metrics = ['cosine', 'eqprop_over_bptt_norm_ratio', 'relative_l2_difference_over_bptt',
               'bptt_rms', 'eqprop_rms', 'clean_eqprop_rms', 'added_noise_rms',
               'noise_over_clean_norm', 'added_noise_over_bptt',
               'bptt_near_zero_fraction', 'eqprop_near_zero_fraction']
    for case in replay_contexts(cfg):
        T = case['replay_T']
        role = case['checkpoint_role']
        checkpoint_file = 'final_model.pt' if role == 'final' else 'best_model.pt'
        checkpoint_hash = case['source_hashes'][checkpoint_file]
        for beta in cfg['injected_betas']:
            name = cell_identity(case, beta, absolute=True)
            run = collected / 'runs' / name
            if not (run / 'result.json').exists():
                missing.append(name)
                continue
            assert not validate_run(run), run
            manifest = json.loads((run / 'manifest.json').read_text())
            terminal = json.loads((run / 'result.json').read_text())['terminal_metrics']
            assert manifest['configuration']['sha256'] == sha256_file(args.config)
            assert manifest['checkpoint_role'] == role and manifest['checkpoint_epoch'] == case['checkpoint_epoch']
            assert manifest['replay_T'] == T and manifest['replay_K'] == cfg['K']
            assert terminal['replay_batches'] == cfg['expected_batches']
            assert terminal['source_bytes_unchanged'] and terminal['all_parameter_tensors_unchanged']
            assert terminal['all_bias_tensors_exact_zero'] and terminal['noise_draws_matched']
            assert terminal['bptt_invariant_across_beta']
            assert not terminal['optimizer_steps_applied'] and not terminal['official_test_read']
            guards = json.loads((run / 'read_only_guards.json').read_text())
            assert len(guards) == 36
            for guard in guards:
                assert guard['iteration_contract_passed']
                assert guard['inference_minimizer_iterations_T'] == T
                assert guard['bptt_minimizer_iterations_K'] == guard['eqprop_minimizer_iterations_K'] == cfg['K']
                assert guard['frozen_current_force_matches_post_T_cost_gradient']
                assert guard['frozen_current_force_unchanged_across_phases']
            targets.add(manifest['runtime']['target'])
            cohort_hashes.add(manifest['cohort_sha256'])
            rows = read_csv(run / 'layer_metrics.csv')
            assert len(rows) == terminal['layer_comparisons'] == 36*5*4
            assert len({(r['batch_index'], r['noise_draw_index'], r['parameter_name']) for r in rows}) == len(rows)
            groups = defaultdict(list)
            context = dict(checkpoint=case['name'], scheme=case['scheme'], epoch=case['checkpoint_epoch'],
                           T=T, K=cfg['K'], injected_beta=beta, training_beta=case['beta'])
            for r in rows:
                assert int(r['T']) == T and int(r['K']) == cfg['K']
                assert r['checkpoint_sha256'] == checkpoint_hash
                key = (case['name'], T, r['batch_index'], r['parameter_name'])
                assert reference_hashes.setdefault(key, r['bptt_gradient_sha256']) == r['bptt_gradient_sha256']
                identity = (r['batch_payload_sha256'], r['batch_source_indices_sha256'])
                assert batch_hashes.setdefault(r['batch_index'], identity) == identity
                rms = float(r['bptt_rms'])
                r['added_noise_over_bptt'] = float(r['added_noise_rms'])/rms if rms else None
                groups[(float(r['endpoint_read_noise_std']), r['parameter_name'])].append(r)
            for (sigma, parameter), group in groups.items():
                assert len(group) == 36*(4 if sigma else 1)
                row = dict(context, sigma=sigma, parameter=parameter, comparisons=len(group))
                for metric in metrics:
                    quantiles(row, metric, [r[metric] for r in group])
                row['all_residual_gates_passed'] = all(r['all_endpoint_residual_gates_passed'] == 'True' for r in group)
                summary.append(row)
            groups = defaultdict(list)
            for r in read_csv(run / 'state_signals.csv'):
                groups[r['layer_index']].append(r)
            for layer, group in groups.items():
                assert len(group) == 36
                row = dict(context, layer_index=int(layer))
                for metric in ['free_rms', 'centered_half_difference_rms', 'zero_free_rms',
                               'midpoint_minus_zero_rms', 'positive_activity_change_fraction',
                               'negative_activity_change_fraction', 'phase_activity_disagreement_fraction']:
                    quantiles(row, metric, [r[metric] for r in group])
                state_summary.append(row)
            groups = defaultdict(list)
            for r in read_csv(run / 'equilibrium_residuals.csv'):
                groups[(r['phase'], r['state_layer_name'], r['residual_mode'])].append(r)
            for (phase, layer, mode), group in groups.items():
                assert len(group) == 36
                row = dict(context, phase=phase, layer=layer, residual_mode=mode,
                           failed_batches=sum(r['gate_passed'] != 'True' for r in group), batches=36)
                quantiles(row, 'batch_p90', [r['selected_max_p90'] for r in group])
                quantiles(row, 'batch_maximum', [r['selected_maximum'] for r in group])
                residual_summary.append(row)
            included.append(dict(name=name, result_sha256=sha256_file(run/'result.json'),
                                 elapsed_seconds=terminal['elapsed_seconds'],
                                 checkpoint_sha256=checkpoint_hash,
                                 residual_failed_rows=terminal['residual_failed_rows']))
    if not args.partial:
        assert not missing, missing
        terminal = json.loads((collected/'summary.json').read_text())
        assert terminal['state'] == 'complete'
        assert len(included) == len(terminal['cells']) == cfg['expected_production_cells']
        assert sum(r['comparisons'] for r in summary) == cfg['expected_layer_comparisons']
    assert len(targets) <= 1 and len(cohort_hashes) <= 1
    for filename, rows in [('layer_summary', summary), ('state_summary', state_summary),
                           ('residual_summary', residual_summary)]:
        write_csv(out/f'{filename}.csv', rows)

    choices = []
    for case in replay_contexts(cfg):
        for sigma in [0., .0005]:
            points = []
            for beta in cfg['injected_betas']:
                rows = [r for r in summary if (r['checkpoint'], r['T'], r['sigma'], r['injected_beta']) ==
                        (case['name'], case['replay_T'], sigma, beta)]
                if len(rows) == 4 and all(r['cosine_defined_count'] == r['comparisons'] for r in rows):
                    points.append((min(r['cosine_median'] for r in rows), beta, rows))
            if not points:
                continue
            selections = [('best_sampled', max(points, key=lambda x: (x[0], -x[1])))]
            selections += [('training_beta', p) for p in points if p[1] == case['beta']]
            for label, (score, beta, rows) in selections:
                row = dict(checkpoint=case['name'], T=case['replay_T'], K=cfg['K'], sigma=sigma,
                           selection=label, beta=beta, worst_layer_median_cosine=score,
                           residuals_pass=all(r['all_residual_gates_passed'] for r in rows),
                           boundary=beta in [cfg['injected_betas'][0], cfg['injected_betas'][-1]])
                for parameter, layer in zip(PARAMETERS, LABELS):
                    r = next(r for r in rows if r['parameter'] == parameter)
                    for metric in ['cosine', 'eqprop_over_bptt_norm_ratio', 'relative_l2_difference_over_bptt',
                                   'added_noise_over_bptt', 'noise_over_clean_norm']:
                        row[layer+'_'+metric] = r[metric+'_median']
                choices.append(row)
    write_csv(out/'selected_points.csv', choices)
    for metric, label, filename in [('cosine', 'EP–BPTT cosine', 'cosine_vs_beta'),
            ('relative_l2_difference_over_bptt', 'Relative gradient error', 'error_vs_beta'),
            ('added_noise_over_bptt', 'Added noise / BPTT norm', 'noise_vs_beta')]:
        fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True, sharey=True)
        for ci, case in enumerate(cfg['cases']):
            for li, parameter in enumerate(PARAMETERS):
                ax = axes[ci, li]
                for ti, T in enumerate(cfg['T_values']):
                    for sigma in ([.0005] if metric == 'added_noise_over_bptt' else [0., .0005]):
                        rows = sorted([r for r in summary if (r['checkpoint'],r['T'],r['sigma'],r['parameter']) ==
                                       (case['name'],T,sigma,parameter)], key=lambda r:r['injected_beta'])
                        x = [r['injected_beta'] for r in rows]
                        y = [r[metric+'_median'] for r in rows]
                        color = ['#276FBF','#D46A21'][ti]
                        ax.plot(x,y,('o-' if sigma else ':'), color=color, markersize=3,
                                label=f'T={T}, '+('noisy' if sigma else 'clean'))
                        if sigma:
                            ax.fill_between(x,[r[metric+'_p10'] for r in rows],
                                            [r[metric+'_p90'] for r in rows],color=color,alpha=.10)
                ax.axvline(case['beta'],color='black',ls='--',lw=.8)
                ax.set_xscale('log')
                if metric == 'cosine':
                    ax.set_ylim(-1.03,1.03)
                    ax.axhline(.9,color='gray',lw=.5)
                else:
                    ax.set_yscale('log')
                    ax.axhline(1,color='gray',lw=.5)
                ax.grid(alpha=.15)
                if ci == 0: ax.set_title(LABELS[li])
                if li == 0: ax.set_ylabel(case['name'].replace('_',' ')+'\n'+label)
                if ci == 2: ax.set_xlabel('Injected β')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False)
        fig.suptitle(('PARTIAL · ' if missing else '')+'Trained p99 checkpoints · K=8 · σ=5e-4 · bands: 10–90% · dashed: training β',y=.945,fontsize=11)
        fig.tight_layout(rect=(0,0,1,.92))
        for ext in ['png','pdf']: fig.savefig(out/f'{filename}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    atomic_write_json(out/'verification.json',dict(complete=not missing, included=included, missing=missing,
        source_config_sha256=sha256_file(args.config), targets=sorted(targets), cohort_sha256=sorted(cohort_hashes),
        layer_comparisons=sum(r['comparisons'] for r in summary), matched_batches=len(batch_hashes),
        bptt_hashes_invariant_across_beta=True, batch_bytes_matched=True,
        official_test_read=False, optimizer_steps_applied=False))
    print(json.dumps(dict(included=len(included),missing=len(missing),output=str(out))))


if __name__ == '__main__':
    main()
