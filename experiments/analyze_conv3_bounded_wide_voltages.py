"""Validate and plot trained Conv3 voltage magnitudes and bounded/wide ratios."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from experiments.reporting import atomic_write_json, sha256_file, validate_run


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    production = root/'production'
    cfg = json.loads((production/'manifest.json').read_text())['configuration']
    assert len(cfg['cases']) == 36
    expected = {c['name'] for c in cfg['cases']}
    assert {p.name for p in (production/'cases').iterdir()} == expected
    for path in [production, *sorted((production/'cases').iterdir())]:
        errors = validate_run(path)
        assert not errors, (path, errors)
        assert json.loads((path/'status.json').read_text())['state'] == 'complete'
    summary = json.loads((production/'summary.json').read_text())
    assert summary['full_cohort'] and summary['examples_per_case'] == 576 and summary['profile_rows'] == 144
    with (production/'voltage_profiles.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 144
    keys = [(r['name'], int(r['layer_index'])) for r in rows]
    assert len(set(keys)) == 144
    assert set(keys) == {(name, i) for name in expected for i in range(4)}
    metrics = ['rms','mean_absolute','signed_mean','std','zero_fraction','minimum','maximum']
    for r in rows:
        for key in metrics:
            r[key] = float(r[key])
            assert np.isfinite(r[key])
        r['seed'], r['layer_index'] = int(r['seed']), int(r['layer_index'])
        assert int(r['element_count']) == 576 * [64*14*14,128*7*7,256*7*7,20][r['layer_index']]
    lookup = {(r['scheme'],r['ratio'],r['seed'],r['layer_index']):r for r in rows}
    output = root/'analysis'
    output.mkdir(exist_ok=True)
    paired = []
    pooled = []
    for scheme in ['baseline','ours','legacy']:
        for ratio in ['wide','10','50','100']:
            for i in range(4):
                subset = [lookup[scheme,ratio,s,i] for s in range(3)]
                row = dict(scheme=scheme, ratio=ratio, layer=subset[0]['layer'], layer_index=i, seeds=3)
                for metric in metrics:
                    values = np.array([r[metric] for r in subset])
                    row.update({f'{metric}_mean':float(values.mean()), f'{metric}_sd':float(values.std(ddof=1)),
                        f'{metric}_seed_min':float(values.min()), f'{metric}_seed_max':float(values.max())})
                pooled.append(row)
                if ratio != 'wide':
                    for seed in range(3):
                        finite, wide = lookup[scheme,ratio,seed,i], lookup[scheme,'wide',seed,i]
                        paired.append(dict(scheme=scheme,ratio=ratio,seed=seed,layer=finite['layer'],layer_index=i,
                            bounded_checkpoint=finite['checkpoint_path'],wide_checkpoint=wide['checkpoint_path'],
                            bounded_epoch=finite['checkpoint_epoch'],wide_epoch=wide['checkpoint_epoch'],
                            **{f'{m}_bounded_over_wide':finite[m]/wide[m] for m in ['rms','mean_absolute']}))
    write_csv(output/'voltage_profiles.csv', rows)
    write_csv(output/'voltage_summary.csv', pooled)
    write_csv(output/'bounded_over_wide.csv', paired)
    colors = {'wide':'#252525','10':'#d95f02','50':'#1b9e77','100':'#7570b3'}
    labels = {'wide':'Wide [0, 100]','10':'Ratio 10','50':'Ratio 50','100':'Ratio 100'}
    titles = {'baseline':'Baseline (1, 1)','ours':'Ours (4, 1)','legacy':'Legacy (4, 0.25)'}
    x = np.arange(4)
    for metric, ylabel in [('rms','RMS voltage'),('mean_absolute','Mean absolute voltage'),('signed_mean','Signed mean voltage')]:
        fig, axes = plt.subplots(1,3,figsize=(13.2,4.5),sharey=True)
        for ax, scheme in zip(axes, titles):
            for ratio in colors:
                values = np.array([[lookup[scheme,ratio,s,i][metric] for i in range(4)] for s in range(3)])
                mean = values.mean(axis=0)
                ax.plot(x,mean,'o-',color=colors[ratio],label=labels[ratio],lw=2,ms=5)
                ax.fill_between(x,values.min(axis=0),values.max(axis=0),color=colors[ratio],alpha=.15)
            ax.set_title(titles[scheme])
            ax.set_xticks(x,['H1','H2','H3','Readout'])
            ax.grid(alpha=.2,which='both')
            if metric != 'signed_mean':
                ax.set_yscale('log')
        axes[0].set_ylabel(ylabel+' (simulator units)')
        axes[-1].legend(fontsize=9)
        fig.suptitle('Trained Conv3 · BPTT/Adam · best validation · T=8',fontsize=14)
        fig.text(.5,.02,'Mean across 3 seeds; shading spans seeds · Same 576 validation images · Finite ratios use wmin=1e−5',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.05,1,.94))
        fig.savefig(output/f'{metric}_by_layer.jpg',dpi=200)
        plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(13.2,4.5),sharey=True)
    for ax,scheme in zip(axes,titles):
        for ratio in ['10','50','100']:
            values=np.array([[lookup[scheme,ratio,s,i]['rms']/lookup[scheme,'wide',s,i]['rms'] for i in range(4)] for s in range(3)])
            ax.plot(x,values.mean(axis=0),'o-',color=colors[ratio],label=labels[ratio],lw=2)
            ax.fill_between(x,values.min(axis=0),values.max(axis=0),color=colors[ratio],alpha=.15)
        ax.axhline(1,color='black',ls='--',lw=1)
        ax.set_title(titles[scheme]);ax.set_xticks(x,['H1','H2','H3','Readout'])
        ax.set_yscale('log');ax.grid(alpha=.2,which='both')
    axes[0].set_ylabel('Bounded / wide RMS (paired by seed)')
    axes[-1].legend()
    fig.suptitle('Change in trained voltage magnitude under finite conductance ratios')
    fig.text(.5,.02,'Mean of 3 paired ratios; shading spans seeds · Dashed line: unchanged voltage magnitude',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(output/'rms_bounded_over_wide.jpg',dpi=200);plt.close(fig)
    atomic_write_json(output/'validation.json',dict(cases=36,profiles=144,examples_per_case=576,
        validated_bundles=37,cohort_sha256=cfg['expected_cohort_sha256'],official_test_read=False,
        source_result_sha256=sha256_file(production/'result.json'),analyzer_sha256=sha256_file(Path(__file__)),
        finite_statistics=True,complete_coverage=True))
    for r in pooled:
        if r['layer']=='H3':
            print(r['scheme'],r['ratio'],'H3 RMS',round(r['rms_mean'],6),'SD',round(r['rms_sd'],6))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
