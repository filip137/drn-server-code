"""Summarize matched initial-output-displacement beta checkpoint replays."""
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
from experiments.replay_conv3_trained_beta_noise import beta_points, cell_identity, replay_contexts
from experiments.summarize_conv3_trained_beta_noise import read_csv, write_csv, quantiles

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ['ConvWeight_0', 'ConvWeight_1', 'ConvWeight_2', 'DenseWeight_0']
LABELS = ['Conv 1', 'Conv 2', 'Conv 3', 'Readout']
METRICS = ['cosine', 'relative_l2_difference_over_bptt', 'eqprop_over_bptt_norm_ratio',
           'bptt_rms', 'clean_eqprop_rms', 'eqprop_rms', 'added_noise_rms',
           'noise_over_clean_norm', 'bptt_near_zero_fraction', 'eqprop_near_zero_fraction']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    study = ROOT / cfg['output_root']
    out = study / 'analysis'
    out.mkdir(exist_ok=True)
    groups, signals = defaultdict(list), defaultdict(list)
    batches, cohort_hashes, targets = {}, set(), set()
    sources, residual_summary = [], []
    all_rows = []
    for case in replay_contexts(cfg):
        for beta in beta_points(cfg, case=case):
            name = cell_identity(case, beta, absolute=True)
            run = study / 'collected/runs' / name
            assert not validate_run(run), name
            manifest = json.loads((run / 'manifest.json').read_text())
            result = json.loads((run / 'result.json').read_text())['terminal_metrics']
            assert manifest['configuration']['sha256'] == sha256_file(args.config)
            assert result['replay_batches'] == 36 and result['layer_comparisons'] == 720
            for guard in ['source_bytes_unchanged', 'all_parameter_tensors_unchanged',
                          'all_bias_tensors_exact_zero', 'noise_draws_matched']:
                assert result[guard], (name, guard)
            assert not result['official_test_read'] and not result['optimizer_steps_applied']
            assert result['checkpoint_epoch'] == case['checkpoint_epoch']
            expected_sha = (cfg['initializer_checkpoint_sha256'] if case['checkpoint_epoch'] == 0
                            else case['source_hashes']['final_model.pt'])
            cohort_hashes.add(manifest['cohort_sha256'])
            targets.add(manifest['runtime']['target'])
            rows = read_csv(run / 'layer_metrics.csv')
            assert len(rows) == len({(r['batch_index'], r['noise_draw_index'], r['parameter_name']) for r in rows}) == 720
            for r in rows:
                batch = int(r['batch_index'])
                identity = (r['batch_payload_sha256'], r['batch_source_indices_sha256'])
                assert batches.setdefault(batch, identity) == identity
                assert r['checkpoint_sha256'] == expected_sha
                assert float(r['injected_beta']) == beta
                assert int(r['T']) == case['replay_T'] and int(r['K']) == case['replay_K']
                draw = int(r['noise_draw_index'])
                sigma = float(r['endpoint_read_noise_std'])
                assert sigma == (0. if draw == 0 else .0005)
                if draw:
                    assert int(r['endpoint_read_noise_seed']) == cfg['read_noise_seeds'][draw-1]
                groups[(case['scheme'], case['checkpoint_epoch'], sigma, r['parameter_name'])].append(r)
            all_rows.extend(rows)
            for r in read_csv(run / 'state_signals.csv'):
                signals[(case['scheme'], case['checkpoint_epoch'], int(r['layer_index']))].append(r)
            residuals = read_csv(run / 'equilibrium_residuals.csv')
            assert residuals and all(r['gate_passed'] == 'True' for r in residuals)
            residual_summary.append(dict(checkpoint=case['name'], rows=len(residuals),
                failures=0, maximum=float(max(float(r['selected_maximum']) for r in residuals))))
            sources.append(dict(name=name, result_sha256=sha256_file(run/'result.json'),
                                checkpoint_sha256=expected_sha, source_bundle=case['bundle']))
    assert len(sources) == cfg['expected_production_cells'] == 4
    assert len(all_rows) == cfg['expected_layer_comparisons'] == 2880
    assert len(batches) == 36 and len(cohort_hashes) == len(targets) == 1
    summary = []
    for (scheme, epoch, sigma, parameter), rows in sorted(groups.items()):
        assert len(rows) == (144 if sigma else 36)
        r = dict(scheme=scheme, epoch=epoch, sigma=sigma, parameter=parameter,
                 beta=float(rows[0]['injected_beta']), comparisons=len(rows))
        for metric in METRICS:
            quantiles(r, metric, [x[metric] for x in rows])
        r['noise_dominated_fraction'] = float(np.mean([float(x['noise_over_clean_norm']) >= 1 for x in rows]))
        summary.append(r)
    state_summary = []
    for (scheme, epoch, layer), rows in sorted(signals.items()):
        assert len(rows) == 36
        r = dict(scheme=scheme, epoch=epoch, layer=LABELS[layer], layer_index=layer,
                 beta=float(rows[0]['injected_beta']), batches=len(rows))
        for metric in ['free_rms', 'positive_free_rms', 'negative_free_rms',
                       'centered_half_difference_rms', 'zero_free_rms']:
            quantiles(r, metric, [x[metric] for x in rows])
            r[metric+'_pooled'] = float(np.sqrt(sum(float(x[metric])**2*int(x['element_count']) for x in rows)
                                                /sum(int(x['element_count']) for x in rows)))
        state_summary.append(r)
    paired, effects = [], []
    for epoch in [0, 30]:
        for sigma in [0., .0005]:
            for parameter in PARAMETERS:
                pairs = []
                for batch in range(36):
                    r = dict(epoch=epoch, sigma=sigma, parameter=parameter, batch=batch)
                    for metric in ['cosine', 'relative_l2_difference_over_bptt', 'noise_over_clean_norm']:
                        for scheme in ['ours', 'legacy']:
                            selected = [x for x in groups[(scheme,epoch,sigma,parameter)] if int(x['batch_index']) == batch]
                            r[scheme+'_'+metric] = float(np.mean([float(x[metric]) for x in selected]))
                        r[metric+'_ours_minus_legacy'] = r['ours_'+metric]-r['legacy_'+metric]
                    pairs.append(r)
                paired.extend(pairs)
                effect = dict(epoch=epoch, sigma=sigma, parameter=parameter, paired_batches=36)
                for metric in ['cosine', 'relative_l2_difference_over_bptt', 'noise_over_clean_norm']:
                    quantiles(effect, metric+'_ours_minus_legacy', [r[metric+'_ours_minus_legacy'] for r in pairs])
                effect['ours_higher_cosine_fraction'] = float(np.mean([r['cosine_ours_minus_legacy'] > 0 for r in pairs]))
                effects.append(effect)
    scales = []
    for scheme in ['legacy','ours']:
        for parameter in PARAMETERS:
            r = dict(scheme=scheme, parameter=parameter)
            for metric in ['bptt_rms', 'clean_eqprop_rms']:
                ratios = []
                for batch in range(36):
                    init = next(x for x in groups[(scheme,0,0.,parameter)] if int(x['batch_index'])==batch)
                    trained = next(x for x in groups[(scheme,30,0.,parameter)] if int(x['batch_index'])==batch)
                    ratios.append(float(trained[metric])/float(init[metric]))
                quantiles(r, metric+'_epoch30_over_init', ratios)
            scales.append(r)
    for filename, rows in [('layer_summary',summary),('phase_summary',state_summary),
                           ('paired_batches',paired),('paired_effects',effects),
                           ('gradient_epoch_ratios',scales),('residual_summary',residual_summary)]:
        write_csv(out/(filename+'.csv'), rows)
    fig, axes = plt.subplots(2,2,figsize=(11,7),sharex=True)
    for col,epoch in enumerate([0,30]):
        for row,sigma in enumerate([0.,.0005]):
            ax=axes[row,col]
            for scheme,color,offset in [('legacy','#276FBF',-.04),('ours','#D46A21',.04)]:
                cells=[next(r for r in summary if (r['scheme'],r['epoch'],r['sigma'],r['parameter'])==(scheme,epoch,sigma,p)) for p in PARAMETERS]
                y=np.array([r['cosine_median'] for r in cells]); lo=np.array([r['cosine_p10'] for r in cells]); hi=np.array([r['cosine_p90'] for r in cells])
                ax.errorbar(np.arange(4)+offset,y,yerr=[y-lo,hi-y],fmt='o-',capsize=3,color=color,label=scheme)
            ax.set_title(('Initialization' if epoch==0 else 'P99-trained epoch 30')+' · '+('clean' if not sigma else 'σ=5×10⁻⁴'))
            ax.set_xticks(range(4),LABELS); ax.grid(alpha=.2); ax.set_ylabel('EqProp–BPTT cosine')
            clean_low = min(r['cosine_p10'] for r in summary if r['epoch']==epoch and r['sigma']==0.)
            noisy_low = min(r['cosine_p10'] for r in summary if r['sigma']>0.)
            ax.set_ylim((min(.95,clean_low-.005),1.003) if not sigma else (min(-.1,noisy_low-.02),1.05)); ax.legend(frameon=False)
    assert len(cfg['T_values']) == 1
    fig.suptitle(f"Fixed β: ours 1.385, legacy 0.02173 · T{cfg['T_values'][0]}/K{cfg['K']}\n"
                 'Median and 10–90% spread across matched validation measurements')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'gradient_quality.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,epoch in zip(axes,[0,30]):
        for scheme,color in [('legacy','#276FBF'),('ours','#D46A21')]:
            rows=[next(r for r in state_summary if (r['scheme'],r['epoch'],r['layer_index'])==(scheme,epoch,i)) for i in range(4)]
            ax.semilogy(range(4),[r['centered_half_difference_rms_pooled'] for r in rows],'o-',label=scheme,color=color)
        ax.axhline(.0005/np.sqrt(2),color='grey',linestyle='--',label='Centered voltage noise RMS')
        ax.set_xticks(range(4),['Hidden 1','Hidden 2','Hidden 3','Output']); ax.grid(alpha=.2)
        ax.set_title('Initialization' if epoch==0 else 'P99-trained epoch 30');ax.set_ylabel('Pooled centered phase RMS');ax.legend(frameon=False,fontsize=8)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'phase_signal.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    atomic_write_json(out/'validation.json',dict(complete=True,cases=len(sources),gradient_rows=len(all_rows),
        cohort_sha256=next(iter(cohort_hashes)),target=next(iter(targets)),matched_batches=len(batches),
        sources=sources,residuals=residual_summary,official_test_read=False,optimizer_steps_applied=False))
    for r in summary:
        print(r['epoch'],r['scheme'],r['sigma'],r['parameter'], 'cos',round(r['cosine_median'],6),
              'relerr',round(r['relative_l2_difference_over_bptt_median'],6),
              'noise/clean',round(r['noise_over_clean_norm_median'],6))


if __name__ == '__main__':
    main()
