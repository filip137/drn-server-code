"""Plot all validated initialization screen and confirmation layer curves; CPU only."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

SCHEMES=('baseline','ours','legacy')
COLORS={'baseline':'#3969AC','ours':'#D55E00','legacy':'#16966A'}
SIGMAS=(0.,.0001,.0005,.001,.002,.005)
TARGETS=(.2,.8,2.,6.,10.)


def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root',type=Path,required=True)
    args=p.parse_args();root=args.study_root
    screen=read(root/'analysis/layer_summary.csv')
    physical=read(root/'analysis/physical_state_controls.csv')
    confirm=read(root/'confirmation/analysis/per_draw_layer_summary.csv')
    assert len(screen)==810 and all(r['complete']=='True' for r in screen)
    assert all(r['complete']=='True' for r in confirm)
    states={(r['architecture'],r['scheme'],float(r['relative_target']),int(r['state_layer_index'])):r for r in physical}
    lookup={(r['architecture'],r['scheme'],float(r['relative_target']),float(r['readout_sigma']),r['parameter_name']):r for r in screen}
    extra=defaultdict(list)
    for r in confirm:
        key=(r['architecture'],r['scheme'],float(r['relative_target']),float(r['readout_sigma']),r['parameter_name'])
        extra[key].append(float(r['cosine_mean']))
    out=root/'analysis/displacement_curves';out.mkdir(exist_ok=True)
    exported=[]
    variants=[('relative_output','Output normalized displacement D/F'),
              ('absolute_output','Output RMS displacement D (simulator voltage units)'),
              ('absolute_layer','This layer’s RMS displacement D (simulator voltage units)')]
    with PdfPages(out/'all_layer_cosine_displacement.pdf') as book:
        for variant,xlabel in variants:
            for depth in (1,2,3):
                arch=f'conv{depth}';params=[f'ConvWeight_{i}' for i in range(depth)]+['DenseWeight_0']
                fig,axes=plt.subplots(6,depth+1,figsize=(4.3*(depth+1),15.5),squeeze=False)
                own=[r for r in screen if r['architecture']==arch]
                ymin=min(-.035,min(float(r['cosine_p10']) for r in own)-.035)
                for row,sigma in enumerate(SIGMAS):
                    for col,param in enumerate(params):
                        ax=axes[row,col]
                        ax.axhline(0,color='#888888',lw=.55,zorder=0)
                        for scheme in SCHEMES:
                            xs=[];ys=[];low=[];high=[]
                            for target in TARGETS:
                                key=(arch,scheme,target,sigma,param);r=lookup[key]
                                output=states[(arch,scheme,target,depth)]
                                state=states[(arch,scheme,target,col)]
                                x=float(output['measured_relative_displacement'] if variant=='relative_output' else
                                        (output if variant=='absolute_output' else state)['centered_displacement_rms'])
                                y=float(r['cosine_mean']);xs.append(x);ys.append(y)
                                low.append(float(r['cosine_p10']));high.append(float(r['cosine_p90']))
                                if variant=='relative_output':
                                    exported.append(dict(architecture=arch,scheme=scheme,parameter=param,noise_sigma=sigma,
                                        target_D_over_F=target,measured_output_D_over_F=output['measured_relative_displacement'],
                                        output_D=output['centered_displacement_rms'],layer_D=state['centered_displacement_rms'],
                                        screen_mean=y,screen_p10=r['cosine_p10'],screen_p90=r['cosine_p90'],
                                        confirmation_draw_means=json.dumps(extra.get(key,[]))))
                                if key in extra:
                                    values=extra[key];assert len(values)==(1 if sigma==0 else 5)
                                    mean=sum(values)/len(values)
                                    ax.errorbar(x,mean,yerr=[[mean-min(values)],[max(values)-mean]],fmt='D',
                                        color=COLORS[scheme],mfc='white',mec=COLORS[scheme],ms=5.5,mew=1.2,
                                        capsize=3,elinewidth=1.2,zorder=5)
                            ax.plot(xs,ys,'o-',color=COLORS[scheme],lw=1.45,ms=3.6,label=scheme,zorder=3)
                            ax.fill_between(xs,low,high,color=COLORS[scheme],alpha=.10,lw=0)
                        ax.set_xscale('log');ax.set_ylim(max(-1.025,ymin),1.035)
                        ax.grid(True,which='major',alpha=.2)
                        if variant=='relative_output':
                            ax.set_xticks(TARGETS,[f'{x:g}' for x in TARGETS]);ax.minorticks_off()
                        if row==0:ax.set_title(f'Convolution {col+1}' if col<depth else 'Readout',fontweight='bold')
                        if col==0:ax.set_ylabel(('Clean (σ = 0)' if sigma==0 else f'Noise σ = {sigma:g}')+'\nCosine vs BPTT')
                        if row==5:ax.set_xlabel(xlabel,fontsize=9)
                handles=[Line2D([],[],color=COLORS[s],marker='o',label=s) for s in SCHEMES]
                handles.append(Line2D([],[],color='#444444',marker='D',mfc='white',ls='none',label='Confirmation: mean + draw range'))
                fig.suptitle(f'Conv{depth} · initialization · each layer vs displacement',fontsize=16,y=.995)
                fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.975),ncol=2,frameon=False,fontsize=10)
                fig.text(.5,.017,'Lines: screen mean; shading: 10–90% batch/draw spread (not confidence intervals).\n'
                         'Diamonds: confirmation at selected points; whiskers: range of five fresh-draw means.\n'
                         'D = RMS((v+ − v−)/2); F = free-output RMS. Fixed weights; no training.',ha='center',fontsize=9)
                fig.subplots_adjust(top=.93,bottom=.085,left=.10,right=.985,hspace=.24,wspace=.22)
                base=out/f'{arch}_{variant}'
                fig.savefig(base.with_suffix('.png'),dpi=180)
                fig.savefig(base.with_suffix('.pdf'));book.savefig(fig);plt.close(fig)
    with (out/'plotted_values.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(exported[0]));w.writeheader();w.writerows(exported)
    (out/'README.md').write_text('''# Layerwise cosine versus displacement

All 810 validated layer/noise/scheme/target screen summaries are plotted, with
selected confirmation runs overlaid separately. Every architecture includes all
its convolution layers and the readout, three schemes, five displacement targets,
and six noise settings including clean. Screen and confirmation draws are not pooled.

Each architecture has three PNG/PDF versions:
- `relative_output`: measured output D/F, matching the experiment coordinate.
- `absolute_output`: measured absolute output centered RMS displacement D.
- `absolute_layer`: measured centered RMS displacement of the layer shown.

Rows are noise levels; columns are layers. Solid lines show mean cosine of
individual batch/draw comparisons; shading is their 10–90% spread, not a confidence
interval. Hollow diamonds show mean confirmation cosine, with the range across
five fresh-draw means. Clean confirmation has only one reference measurement.
The log x-axis uses actual measured displacements; lines connect sampled points
without asserting behavior between them. The y range is shared within a network
and includes every displayed screen band; it may differ between architectures.

`all_layer_cosine_displacement.pdf` contains all nine figures;
`plotted_values.csv` contains the 810 underlying screen summaries and confirmation
values. Existing validated CSVs are the only inputs: no new GPU work or replays.

Reproduce: `python -m experiments.plot_layer_cosine_displacement_curves --study-root
results/conv123-initial-displacement-noise-map-20260925-v1`.
''')
    print(json.dumps(dict(screen_layer_cells=len(exported),confirmation_layer_draw_rows=len(confirm),
                         figures=9,output=str(out))))

if __name__=='__main__':main()
