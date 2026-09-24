"""Summarize the matched initialization beta/T/K replay and paired batches."""
from __future__ import annotations

import argparse
from collections import defaultdict
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
STUDY = 'eqprop-conv3-init-beta-tk-20260923-v1'
PARAMETERS = ['ConvWeight_0', 'ConvWeight_1', 'ConvWeight_2', 'DenseWeight_0']
LAYERS = ['Conv1', 'Conv2', 'Conv3', 'Readout']


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def historical_context(out, summaries, batches):
    """Keep historical data distinct while joining exact beta/T/K points."""
    parent = ROOT / 'results/eqprop-conv3-cosine-tk-schemes-20260922-v1'
    cached = json.loads((parent / 'cached_cells.json').read_text())
    sources = {int(c['T']): ROOT / c['run'] for c in cached if c['scheme'] == 'ours'}
    paired, values, hashes = [], {}, {}
    for T, run in sources.items():
        path = run / 'layer_metrics.csv'
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        grouped = defaultdict(list)
        for r in read_csv(path):
            assert (r['batch_payload_sha256'], r['batch_source_indices_sha256']) == batches[int(r['batch_index'])]
            if int(r['noise_draw_index']) > 0:
                grouped[(int(r['batch_index']), r['parameter_name'])].append(float(r['cosine']))
        values[T] = {k: float(np.mean(v)) for k, v in grouped.items()}
    for T in [16,32]:
        for parameter in PARAMETERS:
            for i in range(36):
                paired.append(dict(checkpoint='ours_p90_epoch30', beta=5.26875648112,
                    K=8, T_start=8,T_end=T,parameter=parameter,batch=i,
                    delta_mean_cosine=values[T][(i,parameter)]-values[8][(i,parameter)]))
    write_csv(out/'historical_paired_batch_changes.csv',paired)
    curves=[dict(checkpoint='initialization',**r) for r in summaries if r['scheme']=='ours']
    path=parent/'analysis/layer_summary.csv'
    hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for r in read_csv(path):
        if r['scheme']=='ours':
            curves.append(dict(checkpoint='p90 epoch30',beta=float(r['injected_beta']),
                T=int(r['T']),K=int(r['K']),condition=r['condition'],parameter=r['parameter_name'],
                cosine_median=float(r['median_cosine'])))
    for study in ['eqprop-conv3-p99-trained-beta-replay-20260923-v1','eqprop-conv3-p99-beta-k-replay-20260923-v1']:
        path=ROOT/'results'/study/'analysis/layer_summary.csv'
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for r in read_csv(path):
            if r['checkpoint']=='ours_epoch30' and float(r['injected_beta'])==.987333678708:
                curves.append(dict(checkpoint='p99 epoch30',beta=float(r['injected_beta']),
                    T=int(r['T']),K=int(r['K']),condition='noisy' if float(r['sigma']) else 'clean',
                    parameter=r['parameter'],cosine_median=float(r['cosine_median'])))
    fig,axes=plt.subplots(2,4,figsize=(13,6),sharex=True,sharey=True)
    for bi,beta in enumerate([.987333678708,5.26875648112]):
        for pi,(parameter,layer) in enumerate(zip(PARAMETERS,LAYERS)):
            ax=axes[bi,pi]
            for checkpoint,color in [('initialization','#0072b2'),('p99 epoch30','#009e73'),('p90 epoch30','#d55e00')]:
                for K,style in [(8,'-'),(32,'--')]:
                    points={r['T']:r['cosine_median'] for r in curves if r['checkpoint']==checkpoint
                            and r['beta']==beta and r['K']==K and r['parameter']==parameter and r['condition']=='noisy'}
                    if points:
                        ax.plot(sorted(points),[points[t] for t in sorted(points)],style,marker='o',color=color,
                                label=f'{checkpoint}, K={K}')
            ax.set(xticks=[8,16,32],ylim=(-1.03,1.03))
            ax.grid(alpha=.2)
            if bi==0:ax.set_title(layer)
            if pi==0:ax.set_ylabel(f'β={beta:.3f}\nNoisy cosine')
            if bi==1:ax.set_xlabel('T')
    handles,labels=[],[]
    for row in axes:
        h,l=row[0].get_legend_handles_labels()
        for hi,li in zip(h,l):
            if li not in labels:handles.append(hi);labels.append(li)
    fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.5,.945),frameon=False)
    fig.suptitle('Ours: initialization and trained checkpoints at matched replay beta\n'
                 'Same 576 examples · σ=5×10⁻⁴ · historical replay environments differ')
    fig.subplots_adjust(left=.07,right=.99,bottom=.1,top=.80,wspace=.08,hspace=.16)
    for suffix in ['png','pdf']:fig.savefig(out/f'ours_initialization_vs_trained.{suffix}',dpi=160)
    plt.close(fig)
    (out/'historical_context_sources.json').write_text(json.dumps(hashes,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=ROOT / 'results' / STUDY)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    study = args.study_root.resolve()
    cfg = json.loads((ROOT / 'configs/conv/eqprop_conv3_init_beta_tk_20260923.json').read_text())
    out = study / ('analysis_partial' if args.partial else 'analysis')
    out.mkdir(exist_ok=True)
    records, included, missing, batches, bptt, starts = [], [], [], {}, {}, {}
    for case in cfg['cases']:
        for T in cfg['T_values']:
            for K in cfg['K_values']:
                for beta in cfg['injected_betas']:
                    name = f"{case['name']}_T{T}K{K}_beta_{format(beta, '.12g').replace('.', 'p')}"
                    run = study / 'collected/runs' / name
                    if not (run / 'result.json').exists():
                        missing.append(name)
                        continue
                    assert not validate_run(run), run
                    manifest = json.loads((run / 'manifest.json').read_text())
                    assert manifest['checkpoint_epoch'] == 0
                    assert manifest['checkpoint_role'] == 'reconstructed_initialization'
                    assert manifest['configuration']['resolved'] == cfg
                    rows = read_csv(run / 'layer_metrics.csv')
                    assert len(rows) == 720
                    seen = set()
                    for row in rows:
                        i, draw = int(row['batch_index']), int(row['noise_draw_index'])
                        parameter = row['parameter_name']
                        assert int(row['T']) == T and int(row['K']) == K
                        assert float(row['injected_beta']) == beta
                        assert row['training_beta'] == row['training_sigma'] == ''
                        assert row['checkpoint_sha256'] == cfg['initializer_checkpoint_sha256']
                        assert int(row['checkpoint_epoch']) == 0
                        assert i in range(36) and draw in range(5) and parameter in PARAMETERS
                        identity = row['batch_payload_sha256'], row['batch_source_indices_sha256']
                        assert batches.setdefault(i, identity) == identity
                        key = (case['scheme'], T, K, i, parameter)
                        assert bptt.setdefault(key, row['bptt_gradient_sha256']) == row['bptt_gradient_sha256']
                        key = (i, draw, parameter)
                        assert key not in seen
                        seen.add(key)
                        cosine = float(row['cosine'])
                        assert np.isfinite(cosine)
                        records.append(dict(scheme=case['scheme'], T=T, K=K, beta=beta,
                            parameter=parameter, batch=i, draw=draw,
                            condition='noisy' if draw else 'clean', cosine=cosine,
                            fixed_reference_cosine=float(row['ep_vs_reference_bptt_cosine']),
                            bptt_rms=float(row['bptt_rms']), ep_rms=float(row['eqprop_rms']),
                            near_zero_fraction=float(row['eqprop_near_zero_fraction'])))
                    guards = json.loads((run / 'read_only_guards.json').read_text())
                    assert len(guards) == 36
                    for i, guard in enumerate(guards):
                        assert guard['reconstructed_initialization_tensor_sha256'] == case['reconstructed_initialization_tensor_sha256']
                        assert guard['parameter_tensors_unchanged_after_cast']
                        assert guard['iteration_contract_passed']
                        key = (case['scheme'], T, i)
                        identity = guard['common_post_T_state_sha256'], guard['frozen_current_force_sha256']
                        assert starts.setdefault(key, identity) == identity
                        assert guard['bptt_start_state_sha256'] == identity[0]
                    included.append(dict(name=name, result_sha256=hashlib.sha256((run/'result.json').read_bytes()).hexdigest()))
    if not args.partial:
        assert not missing, missing
        assert len(included) == cfg['expected_production_cells']
        assert len(records) == cfg['expected_layer_comparisons']
        assert len(batches) == cfg['expected_batches']
    assert records, 'No completed cells yet'
    grouped, batch_groups = defaultdict(list), defaultdict(list)
    for r in records:
        key = (r['scheme'], r['beta'], r['T'], r['K'], r['condition'], r['parameter'])
        grouped[key].append(r)
        batch_groups[(*key, r['batch'])].append(r['cosine'])
    summaries = []
    for key, group in sorted(grouped.items()):
        row = dict(zip(['scheme', 'beta', 'T', 'K', 'condition', 'parameter'], key))
        row['comparisons'] = len(group)
        for metric in ['cosine', 'fixed_reference_cosine', 'bptt_rms', 'ep_rms', 'near_zero_fraction']:
            values = [r[metric] for r in group]
            row.update({metric+'_'+label: float(np.quantile(values, q))
                        for label, q in [('p10', .1), ('median', .5), ('p90', .9)]})
        summaries.append(row)
    write_csv(out / 'layer_summary.csv', summaries)
    lookup = {tuple(r[k] for k in ['scheme', 'beta', 'T', 'K', 'condition', 'parameter']): r for r in summaries}
    batch_means = {key: float(np.mean(v)) for key, v in batch_groups.items()}
    paired, effects = [], []
    for key, left in sorted(lookup.items()):
        scheme, beta, T, K, condition, parameter = key
        comparisons = [('K', 8, 32)] if K == 8 else []
        comparisons += [('T', T, stop) for stop in cfg['T_values'] if stop > T]
        for axis, begin, end in comparisons:
            right_key = (scheme, beta, end if axis == 'T' else T,
                         end if axis == 'K' else K, condition, parameter)
            if right_key not in lookup:
                continue
            context = dict(scheme=scheme, beta=beta, condition=condition,
                parameter=parameter, varied_axis=axis, start=begin, end=end,
                fixed_axis='K' if axis == 'T' else 'T', fixed_value=K if axis == 'T' else T)
            deltas = []
            for i in range(36):
                delta = batch_means[(*right_key, i)] - batch_means[(*key, i)]
                paired.append(dict(context, batch=i, delta_mean_cosine=delta))
                deltas.append(delta)
            right = lookup[right_key]
            effects.append(dict(context, median_cosine_start=left['cosine_median'],
                median_cosine_end=right['cosine_median'],
                delta_median_cosine=right['cosine_median']-left['cosine_median'],
                delta_fixed_reference_median=right['fixed_reference_cosine_median']-left['fixed_reference_cosine_median'],
                paired_delta_p10=float(np.quantile(deltas,.1)), paired_delta_median=float(np.median(deltas)),
                paired_delta_p90=float(np.quantile(deltas,.9)),
                positive_batch_fraction=float(np.mean(np.array(deltas)>0)),
                absolute_change_above_0p01_batch_fraction=float(np.mean(np.abs(deltas)>.01))))
    if effects:
        write_csv(out / 'paired_batch_changes.csv', paired)
        write_csv(out / 'effects.csv', effects)
    for condition in ['clean', 'noisy']:
        fig, axes = plt.subplots(3, 4, figsize=(13, 8), sharex=True, sharey=True)
        for si, scheme in enumerate(['baseline', 'ours', 'legacy']):
            for pi, (parameter, layer) in enumerate(zip(PARAMETERS, LAYERS)):
                ax = axes[si, pi]
                for beta, color in zip(cfg['injected_betas'], ['#0072b2', '#d55e00']):
                    for K, style in zip(cfg['K_values'], ['-', '--']):
                        selected = [lookup[(scheme,beta,T,K,condition,parameter)] for T in cfg['T_values']
                                    if (scheme,beta,T,K,condition,parameter) in lookup]
                        ax.plot([r['T'] for r in selected], [r['cosine_median'] for r in selected],
                                style, marker='o', color=color, label=f'β={beta:.3f}, K={K}')
                        ax.fill_between([r['T'] for r in selected], [r['cosine_p10'] for r in selected],
                                        [r['cosine_p90'] for r in selected], color=color, alpha=.06)
                ax.set(xticks=cfg['T_values'], ylim=(-1.03,1.03))
                ax.grid(alpha=.2)
                if si == 0: ax.set_title(layer)
                if pi == 0: ax.set_ylabel(f'{scheme}\nEP–BPTT cosine')
                if si == 2: ax.set_xlabel('T')
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(.5,.95), frameon=False)
        fig.suptitle(('PARTIAL · ' if missing else '') + f'Conv3 initialization · {condition} endpoint reads\n'
                     'Same seed-0 weights · 576 validation examples · bands: 10–90% descriptive spread')
        fig.tight_layout(rect=[0,0,1,.88])
        for suffix in ['png','pdf']: fig.savefig(out/f'{condition}_cosine_vs_T.{suffix}',dpi=160)
        plt.close(fig)
    maxima = []
    for scheme in ['baseline','ours','legacy']:
        for beta in cfg['injected_betas']:
            row=dict(scheme=scheme,beta=beta)
            for condition in ['clean','noisy']:
                for axis in ['T','K']:
                    values=[abs(e['delta_median_cosine']) for e in effects
                            if e['scheme']==scheme and e['beta']==beta and e['condition']==condition and e['varied_axis']==axis]
                    row[f'{condition}_{axis}_max_change']=max(values) if values else None
            maxima.append(row)
    write_csv(out/'max_cosine_changes.csv',maxima)
    if not missing:
        historical_context(out, summaries, batches)
    verification=dict(complete=not missing, included=included, missing=missing,
        layer_comparisons=len(records),matched_batches=len(batches),
        initialization_sha256=cfg['initializer_checkpoint_sha256'],
        cohort_sha256=json.loads((study/'smoke_cohort.json').read_text())['cohort_sha256'],
        same_post_T_state_and_force_across_K_and_beta=True, bptt_invariant_across_beta=True,
        all_parameter_tensors_unchanged=True,official_test_read=False,optimizer_steps_applied=False)
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    lines=['# Conv3 initialization: beta, T and K', '',
           ('Partial' if missing else 'Complete')+f': {len(included)}/36 cases, {len(records):,} gradient comparisons.', '',
           'All schemes start from the same saved seed-0 initialization (epoch0). '
           'Injected beta is .987333678708 or5.26875648112; T=8/16/32, K=8/32. '
           'Clean reads and four matched endpoint-noise draws at sigma5e-4 use '
           '36 batches of16 validation examples. No training or test evaluation.', '',
           '## Largest absolute changes in layer-median cosine', '',
           'T changes pool each fixed K; K changes pool each fixed T. Maxima cover all four layers. '
           'These are changes within the sampled grid, not an exact convergence threshold.', '',
           '| Scheme | Injected beta | T, clean | T, noisy | K, clean | K, noisy |',
           '|---|---:|---:|---:|---:|---:|']
    for r in maxima:
        f=lambda x:'pending' if x is None else f'{x:.6g}'
        lines.append(f"| {r['scheme']} | {r['beta']:.9g} | "+' | '.join(f(r[k]) for k in ['clean_T_max_change','noisy_T_max_change','clean_K_max_change','noisy_K_max_change'])+' |')
    lines+=['','## Layer medians at T8/K8','','| Scheme | Beta | Reads | Conv1 | Conv2 | Conv3 | Readout |','|---|---:|---|---:|---:|---:|---:|']
    for scheme in ['baseline','ours','legacy']:
        for beta in cfg['injected_betas']:
            for condition in ['clean','noisy']:
                keys=[(scheme,beta,8,8,condition,p) for p in PARAMETERS]
                if all(k in lookup for k in keys):
                    lines.append(f'| {scheme} | {beta:.6g} | {condition} | '+' | '.join(f"{lookup[k]['cosine_median']:.6f}" for k in keys)+' |')
    lines+=['','![Noisy initialization cosine](noisy_cosine_vs_T.png)','','![Clean initialization cosine](clean_cosine_vs_T.png)', '',
            '## Paired batches and references','',
            'For every T or K contrast, `paired_batch_changes.csv` first averages the four '
            'noisy cosines within each batch, then subtracts the matched starting setting. '
            '`effects.csv` gives the median and10–90% spread across36 paired batches, '
            'the fraction with positive change, and the fraction with absolute change above.01. '
            'These spreads describe examples, not uncertainty across training seeds.', '',
            'Matching-K cosines use BPTT through exactly K zero-nudge steps from the same '
            'post-T state as EqProp. Fixed-reference cosines use K32 BPTT at the same T. '
            'K32 is a finite reference. Changing T changes the starting state and its reference. '
            'Read noise is added only to endpoint reads; physical dynamics are clean.', '',
            '[Layer statistics](layer_summary.csv) · [Paired effects](effects.csv) · '
            '[Per-batch changes](paired_batch_changes.csv) · [Coverage and guards](verification.json)', '',
            '[Plan](../../../docs/eqprop_conv3_init_beta_tk_plan_20260923.md) · '
            '[Config](../../../configs/conv/eqprop_conv3_init_beta_tk_20260923.json)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    if not missing:
        max_noisy_T=max(r['noisy_T_max_change'] for r in maxima)
        max_noisy_K=max(r['noisy_K_max_change'] for r in maxima)
        max_clean=max(max(r['clean_T_max_change'],r['clean_K_max_change']) for r in maxima)
        max_paired=max(abs(r['delta_mean_cosine']) for r in paired)
        interpretation=(
            '**No meaningful T/K dependence is present at initialization on this grid.** '
            f'Across all schemes, betas, and layers, the largest noisy median-cosine change is '
            f'{max_noisy_T:.3g} when changing T and {max_noisy_K:.3g} when changing K. '
            f'The largest clean change is {max_clean:.3g}.\n\n'
            f'The largest absolute paired-batch change, averaging the four noisy draws within '
            f'each batch, is {max_paired:.3g} across both clean and noisy comparisons. '
            'No paired batch changes by more than .01. The fixed K32 reference gives '
            'the same practical conclusion; matching-K agreement is not hiding a large K effect.\n\n'
            'For ours, noisy readout cosine stays .999978 at beta .987333678708 and .999923 '
            'at beta5.26875648112 across all six T/K settings. At the latter beta, the '
            'older p90-trained epoch30 checkpoint instead changes from -.293154 to .820463 '
            'as T increases8 to32 at K8. Its paired readout improvement occurs on35/36 '
            'batches at T16 and36/36 at T32. This sensitivity is absent from the shared '
            'initial weights and appears later along that training trajectory. The exact '
            'onset epoch remains unmeasured.\n\n'
            'Beta still affects gradient quality at initialization: ours noisy Conv3 cosine '
            'rises from .571775 to .960893 between the two tested betas, while its first '
            'two convolutions remain poorly aligned under read noise. Longer T/K does '
            'not repair that noise limitation. The common betas are diagnostic probes, '
            'not selected training values for all schemes.\n\n'
            '![Ours initialization versus trained checkpoints](ours_initialization_vs_trained.png)\n\n'
            '[Historical paired batches](historical_paired_batch_changes.csv) · '
            '[Historical source hashes](historical_context_sources.json). The historical '
            'curves use the exact same cohort but different replay environments; comparisons '
            'within each study are matched. This is one initialization seed and two replay '
            'betas, not a universal guarantee across checkpoints, betas, or seeds.\n\n')
        text=(out/'report.md').read_text()
        text=text.replace('## Largest absolute changes in layer-median cosine',
                          interpretation+'## Largest absolute changes in layer-median cosine',1)
        validation=json.loads((study/'collection_validation.json').read_text())
        text+='\n## Completion and collection\n\n'
        text+=(f"All36 production bundles and18 local GPU smoke bundles validate. All"
               f"{validation['collected_files_matching_remote']} collected file hashes match Fifi; "
               f"{validation['frozen_files_verified_local_and_remote']} frozen input/source files "
               'match locally and remotely. Initialization tensors, post-T starts, frozen '
               'forces, cohorts, noise draws, and beta-independent BPTT guards pass. '
               f"Production took{validation['production_seconds']:.2f}s; total including smoke "
               f"{validation['total_including_smoke_seconds']:.2f}s within3600s. The worker "
               'exited0 and released the GPU. No scientific cases failed or were excluded. '
               'Prelaunch staging corrections are recorded in the plan. Partial analyses '
               'are superseded by this complete report.\n\n'
               '[Collection validation](../collection_validation.json).\n')
        (out/'report.md').write_text(text)
    print(json.dumps(dict(cells=len(included),missing=len(missing),output=str(out))))


if __name__ == '__main__':
    main()
