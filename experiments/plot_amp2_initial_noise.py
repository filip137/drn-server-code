"""Add A=4, B=2 to the established initialization noise-versus-BPTT curves."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiments.reporting import sha256_file, validate_run
from experiments.plot_layer_fractional_displacement import CASES, STYLES


def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,required=True)
    args=parser.parse_args()
    root=args.study_root.resolve();out=root/'initialization/analysis';out.mkdir(exist_ok=True)
    source=Path('results/conv123-relative-noise-advantage-map-20260925-v1/analysis/stage_a/layer_summary.csv')
    metrics=('cosine','noisy_clean_eqprop_cosine','noise_over_clean_norm')
    rows=[]
    for r in read(source):
        if (r['architecture'],float(r['relative_target'])) not in CASES:continue
        row={k:r[k] for k in ('architecture','scheme','parameter_name')}
        row.update(relative_target=float(r['relative_target']),noise_level=float(r['noise_level']))
        for field in metrics:
            for suffix in ('','_min','_max'):row[field+suffix]=float(r[field+suffix])
        rows.append(row)
    errors={};matches={};count=0
    for arch,target in CASES:
        lane=root/'initialization/production'/arch
        errors[arch]=validate_run(lane)
        case=lane/'cases'/f'{arch}_amp2_relative{int(target)}'
        errors[arch+'_case']=validate_run(case)
        assert not errors[arch] and not errors[arch+'_case'],errors
        cfg=json.loads((lane/'config.used.json').read_text())
        assert cfg['noisy_batches']==12 and cfg['expected_batches']==36
        assert cfg['read_noise_seeds']==[111000000,112000000,113000000]
        matching=json.loads((case/'matching.json').read_text());assert matching['matched']
        matches[arch]=matching
        raw=read(case/'layer_metrics.csv');count+=len(raw)
        assert len(raw)==(int(arch[-1])+1)*(36+12*3*14)
        groups=defaultdict(list)
        for r in raw:groups[(r['parameter_name'],float(r['noise_level']))].append(r)
        for (layer,eta),values in sorted(groups.items()):
            row=dict(architecture=arch,scheme='amp2',parameter_name=layer,relative_target=target,noise_level=eta)
            by_draw=defaultdict(list)
            for r in values:by_draw[int(r['noise_draw_index'])].append(r)
            assert len(by_draw)==(3 if eta else 1)
            assert all(len(v)==(12 if eta else 36) for v in by_draw.values())
            for metric in metrics:
                means=[mean(float(r[metric]) for r in draw) for draw in by_draw.values()]
                assert all(math.isfinite(x) for x in means)
                row.update({metric:mean(means),metric+'_min':min(means),metric+'_max':max(means)})
            rows.append(row)
    assert len(rows)==540
    with (out/'plotted_values.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    styles={**STYLES,'amp2':dict(color='#9467bd',linestyle='-',marker='s',markersize=4,zorder=6)}
    names={'baseline':'Baseline (A=1, B=1)','ours':'Ours (A=4, B=1)',
           'legacy':'Legacy (A=4, B=0.25)','amp2':'A=4, B=2'}
    handles=[Line2D([],[],label=names[s],**style) for s,style in styles.items()]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

    def draw(ax,arch,layer):
        for scheme,style in styles.items():
            points=sorted([r for r in rows if (r['architecture'],r['scheme'],r['parameter_name'])==(arch,scheme,layer) and r['noise_level']>0],key=lambda r:r['noise_level'])
            assert len(points)==14
            x=[r['noise_level'] for r in points]
            ax.plot(x,[r['cosine'] for r in points],lw=1.5,**style)
            ax.fill_between(x,[r['cosine_min'] for r in points],[r['cosine_max'] for r in points],color=style['color'],alpha=.10)
        ax.set(xscale='log',xlim=(1.7e-9,1.7e-2),ylim=(-.07,1.05))
        ax.set_xticks([1e-8,1e-6,1e-4,1e-2]);ax.grid(alpha=.17)
        ax.set_title('Readout' if layer.startswith('Dense') else 'Convolution '+str(int(layer.split('_')[-1])+1))

    fig,axes=plt.subplots(3,4,figsize=(14,9),sharex=True,sharey=True)
    for row,(arch,target) in enumerate(CASES):
        depth=int(arch[-1]);columns=list(range(depth))+[3]
        layers=[f'ConvWeight_{i}' for i in range(depth)]+['DenseWeight_0']
        for col,layer in zip(columns,layers):draw(axes[row,col],arch,layer)
        for col in set(range(4))-set(columns):axes[row,col].set_visible(False)
        axes[row,0].set_ylabel(f'{arch.capitalize()} · output D/F={target:g}\nCosine to BPTT')
        f,axs=plt.subplots(1,len(layers),figsize=(4*len(layers),4),sharey=True)
        for ax,layer in zip(axs,layers):draw(ax,arch,layer);ax.set_xlabel('Relative read noise η')
        axs[0].set_ylabel('Cosine to BPTT')
        f.suptitle(f'{arch.capitalize()} initialization · output D/F={target:g}')
        f.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.92),ncol=4,frameon=False)
        f.text(.5,.02,'192 examples · 3 paired draws · one initializer · native finite-K BPTT · unit targets',ha='center',fontsize=9)
        f.tight_layout(rect=(0,.06,1,.83));f.savefig(out/f'{arch}_initialization_amp2_noise_vs_bptt.jpg',dpi=200);plt.close(f)
    for ax in axes[-1]:ax.set_xlabel('Relative read noise η')
    fig.suptitle('Initialization: does increasing current amplification preserve gradient agreement?',y=.99,fontsize=14)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.955),ncol=4,frameon=False)
    fig.text(.5,.012,'192 validation images · bands: range over 3 paired noise draws · one common initializer per depth\nUnit targets · matched output D/F · baseline/ours/legacy reused from the prior screen; B=2 is newly replayed',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.92));fig.savefig(out/'initialization_amp2_noise_vs_bptt.jpg',dpi=200);plt.close(fig)
    validation=dict(complete=True,new_cases=3,new_layer_comparisons=count,plotted_cells=len(rows),bundle_errors=errors,
        source_summary_sha256=sha256_file(source),matching=matches,
        amp2_clean_cosines=[{k:r[k] for k in ('architecture','parameter_name','cosine')} for r in rows if r['scheme']=='amp2' and r['noise_level']==0])
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    for arch,eta in [('conv1',1e-3),('conv2',1e-5),('conv3',3e-8),('conv3',1e-4)]:
        for i in range(int(arch[-1])):
            layer=f'ConvWeight_{i}'
            print(arch,layer,eta,{r['scheme']:round(r['cosine'],5) for r in rows if (r['architecture'],r['parameter_name'],r['noise_level'])==(arch,layer,eta)})


if __name__=='__main__':main()
