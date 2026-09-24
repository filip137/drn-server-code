"""Collect local refined replays and show measured cosine boundary intervals."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.reporting import validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'results/eqprop-beta-refinement-20260919-v1'
OUT = ROOT / 'paper_ready_results'


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    old = read_csv(OUT / 'beta_rule_comparison_20260918_layer_metrics.csv')
    groups = defaultdict(list)
    origins = {}
    for r in old:
        if r['architecture'] in ('conv2', 'conv3'):
            k = r['architecture'], r['scheme'], float(r['injected_beta'])
            groups[k].append(r)
            origins[k] = 'original_grid'
    states = []
    valid = []
    for folder in ('local', 'local_conv3'):
        p = STUDY / folder
        progress = json.loads((p / 'progress.json').read_text())
        states.append(progress['state'])
        spec = json.loads((p / 'study.json').read_text())
        for name, expected_hash in spec['source_hashes'].items():
            assert hashlib.sha256((p / name).read_bytes()).hexdigest() == expected_hash, name
        if progress['state'] == 'complete':
            refined = json.loads((p / 'refinement_betas.json').read_text())
            expected = {(scheme, float(beta)) for scheme, betas in spec['initial_betas'].items() for beta in betas}
            expected |= {(scheme, float(beta)) for scheme, betas in refined.items() for beta in betas}
            observed = {(r['scheme'], float(r['beta'])) for r in progress['runs']}
            assert expected == observed and len(observed) == len(progress['runs']), p
            assert (p / 'exit_code').read_text().strip() == '0', p
        for row in progress['runs']:
            bundle = Path(row['bundle'])
            assert not validate_run(bundle), bundle
            assert row['state'] == 'complete', 'Explicitly reconcile numerical failures before final analysis.'
            metrics = json.loads((bundle / 'result.json').read_text())['terminal_metrics']
            assert metrics['replay_count'] == 72 and not metrics['official_test_read']
            assert not metrics['optimizer_steps_applied'] and metrics['source_bytes_unchanged']
            assert metrics['all_bias_tensors_exact_zero'] and metrics['endpoint_read_noise_std'] == 0
            rows = read_csv(bundle / 'layer_metrics.csv')
            k = rows[0]['architecture'], rows[0]['scheme'], float(rows[0]['injected_beta'])
            assert k not in groups and len(rows) == 72 * (int(k[0][-1])+1)
            groups[k] = rows
            origins[k] = 'refined_grid'
            valid.append(str(bundle.relative_to(ROOT)))
    summary = []
    for (arch, scheme, beta), rows in sorted(groups.items()):
        assert len({(r['checkpoint_role'],r['batch_index']) for r in rows}) == 72
        limiting = min(rows, key=lambda r:float(r['cosine']))
        summary.append(dict(architecture=arch, scheme=scheme, beta=beta,
            minimum_cosine=min(float(r['cosine']) for r in rows),
            maximum_norm_mismatch=max(float(r['symmetric_norm_delta']) for r in rows),
            maximum_relative_l2_error=max(float(r['relative_l2_difference_over_bptt']) for r in rows),
            limiting_matrix=limiting['parameter_name'], limiting_checkpoint=limiting['checkpoint_role'],
            limiting_batch=int(limiting['batch_index']),
            origin=origins[arch, scheme, beta]))
    complete = all(s == 'complete' for s in states)
    selections = []
    for arch in ('conv2', 'conv3'):
        for scheme in ('baseline', 'ours', 'legacy'):
            rows = [r for r in summary if (r['architecture'],r['scheme']) == (arch,scheme)]
            original = [r for r in rows if r['origin'] == 'original_grid']
            for t in (.9,.95):
                chosen = max((r for r in rows if r['minimum_cosine'] > t), key=lambda r:r['beta'])
                above = [r for r in rows if r['beta'] > chosen['beta']]
                hi = min(above, key=lambda r:r['beta']) if above else None
                previous = max(r['beta'] for r in original if r['minimum_cosine'] > t)
                width = hi['beta']/chosen['beta']-1 if hi else None
                if complete and width is not None:
                    assert width <= .050000001
                    assert hi['minimum_cosine'] <= t
                selections.append(dict(architecture=arch, scheme=scheme, threshold=t,
                    previous_beta=previous, beta=chosen['beta'], multiplier=chosen['beta']/previous,
                    minimum_cosine=chosen['minimum_cosine'], maximum_norm_mismatch=chosen['maximum_norm_mismatch'],
                    limiting_matrix=chosen['limiting_matrix'], limiting_checkpoint=chosen['limiting_checkpoint'],
                    limiting_batch=chosen['limiting_batch'],
                    next_failing_beta=hi['beta'] if hi else None, bracket_width=width,
                    open_upper_edge=hi is None))
    for name, rows in (('beta_refinement_20260919_metrics.csv',summary),('beta_refinement_20260919_selections.csv',selections)):
        with (OUT/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    payload=dict(complete=complete,new_cases=len(valid),total_cases=len(summary),validated_bundles=valid,selections=selections)
    (STUDY/'analysis.json').write_text(json.dumps(payload,indent=2)+'\n')
    text=['# Refined beta cosine boundaries','',f"Status: {'complete' if complete else 'partial'}. {len(valid)} new beta settings collected and validated.",
          '', 'Zero-noise Conv2/Conv3, seed0, T=K6/8; unchanged 36-batch validation-partition cohort at initializer and saved BPTT checkpoint. Every weight matrix and every replay must pass strictly. Norm mismatch is diagnostic, without a norm gate.',
          '', '| Model | Scheme | Threshold | Previous beta | Refined passing beta | Next failing beta | Bracket width | Worst cosine |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in selections:
        hi=f"{r['next_failing_beta']:.6g}" if r['next_failing_beta'] is not None else 'open'
        width=f"{100*r['bracket_width']:.2f}%" if r['bracket_width'] is not None else '—'
        text.append(f"| {r['architecture']} | {r['scheme']} | {r['threshold']:.2f} | {r['previous_beta']:g} | {r['beta']:.6g} | {hi} | {width} | {r['minimum_cosine']:.6f} |")
    text += ['', 'These are sampled crossing intervals, not a proof of a globally monotone cosine function or an exact maximum. Passing values are calibration-qualified only: the newly refined betas have not undergone new training or noisy-training qualification.',
             'The figure zooms around the threshold crossings; it does not display every measured beta or every training outcome. The CSVs retain the complete grids.',
             'Original measurements and all new measurements are retained. Both production workers used the local RTX3090; remote staging smokes and refused/aborted admissions are excluded. See the plan and handoff for transport revisions.',
             '', '[Figure](beta_refinement_20260919.png) · [PDF](beta_refinement_20260919.pdf) · [Measurements](beta_refinement_20260919_metrics.csv) · [Selections](beta_refinement_20260919_selections.csv) · [Plan](../docs/eqprop_beta_refinement_plan_20260919.md)']
    (OUT/'beta_refinement_20260919.md').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    training=read_csv(OUT/'beta_protocol_training_evidence_20260919.csv')
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    windows={('conv2','baseline'):(450,800),('conv2','ours'):(27,85),('conv2','legacy'):(25,380),
             ('conv3','baseline'):(90,550),('conv3','ours'):(.8,7),('conv3','legacy'):(.85,13)}
    for i,arch in enumerate(('conv2','conv3')):
        for j,scheme in enumerate(('baseline','ours','legacy')):
            ax=axes[i,j];rows=[r for r in summary if (r['architecture'],r['scheme'])==(arch,scheme)]
            ax.axhline(.90,color='gray',ls='--',lw=1);ax.axhline(.95,color='gray',ls=':',lw=1)
            for origin,marker in (('original_grid','o'),('refined_grid','.')):
                rs=[r for r in rows if r['origin']==origin]
                ax.scatter([r['beta'] for r in rs],[r['minimum_cosine'] for r in rs],s=25 if marker=='o' else 18,marker=marker,label=origin.replace('_',' '))
            for r in training:
                if (r['architecture'],r['scheme'])==(arch,scheme):
                    stable=r['early_stable']=='True'
                    ax.scatter(float(r['beta']),float(r['matrix_minimum_cosine']),marker='*' if stable else 'x',color='green' if stable else 'red',s=70,zorder=4)
            ax.set(xscale='log',xlim=windows[arch,scheme],ylim=(.75,1.005),title=f'{arch.capitalize()} {scheme}',xlabel='Injected beta',ylabel='Worst per-matrix cosine')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Refined static cosine boundaries\nGreen star: ten-epoch stable; red cross: nonfinite training (existing evidence)')
    fig.savefig(OUT/'beta_refinement_20260919.png',dpi=160);fig.savefig(OUT/'beta_refinement_20260919.pdf');plt.close(fig)
    print(json.dumps({k:v for k,v in payload.items() if k not in ('validated_bundles','selections')}))


if __name__=='__main__':
    main()
