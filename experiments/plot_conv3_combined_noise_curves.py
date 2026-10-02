"""Overlay all measured exp013 noise levels in Conv3 layer panels (CPU only)."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,required=True)
    root=parser.parse_args().study_root
    out=root/'analysis/displacement_curves'
    with (out/'plotted_values.csv').open() as stream:
        rows=[r for r in csv.DictReader(stream) if r['architecture']=='conv3']
    for r in rows:
        for k in ['eta','relative_output','cosine']:
            r[k]=float(r[k])
    etas=sorted({r['eta'] for r in rows})
    middle=(len(etas)+1)//2
    groups=[('low_noise',etas[:middle]),('high_noise',etas[middle:])]
    colors={'baseline':'#16966A','ours':'#F28E2B'}
    markers=['o','s','^','v','D','P','X','<','>','h']
    line_styles=['-','--','-.',':']
    params=['ConvWeight_0','ConvWeight_1','ConvWeight_2','DenseWeight_0']
    titles=['Convolution 1','Convolution 2','Convolution 3','Readout']
    def panel(ax,param,levels):
        for idx,eta in enumerate(levels):
            marker=markers[idx]
            style=line_styles[idx % len(line_styles)]
            for scheme,color in colors.items():
                for stage in ['stage_a','stage_b']:
                    rr=sorted((r for r in rows if r['parameter']==param and r['eta']==eta and r['scheme']==scheme and r['stage']==stage),key=lambda r:r['relative_output'])
                    if not rr: continue
                    x=[r['relative_output'] for r in rr];y=[r['cosine'] for r in rr]
                    # Refinement stays separate: new cohort/noise draws, never pooled.
                    if stage=='stage_a':
                        ax.plot(x,y,color=color,ls=style,marker=marker,ms=3.3,lw=1.6,alpha=.9)
                    else:
                        ax.plot(x,y,color=color,ls='none',marker=marker,ms=6,mfc='white',mew=1)
        ax.set_xscale('log');ax.set_xlim(.45,9);ax.set_ylim(-.10,1.035)
        ax.set_xticks([.5,1,2,4,6,8],['0.5','1','2','4','6','8']);ax.minorticks_off()
        ax.set_xlabel('Measured output normalized RMS displacement D/F')
        ax.set_ylabel('Gradient cosine similarity vs BPTT')
        ax.grid(alpha=.2);ax.axhline(0,color='gray',lw=.5)
    for group,levels in groups:
        noise_handles=[Line2D([],[],color='#555555',ls=line_styles[i % len(line_styles)],marker=markers[i],ms=4,lw=1.5,label=f'eta = {eta:.3g}') for i,eta in enumerate(levels)]
        scheme_handles=[Line2D([],[],color=color,lw=2,label=scheme) for scheme,color in colors.items()]
        draw_handles=[Line2D([],[],color='#333333',ls='none',marker='o',label='Screen: filled'), Line2D([],[],color='#333333',ls='none',marker='o',mfc='white',label='Refinement: hollow')]
        fig,axes=plt.subplots(2,2,figsize=(15,10),sharex=True,sharey=True)
        for ax,param,title in zip(axes.flat,params,titles):
            panel(ax,param,levels);ax.set_title(title,fontweight='bold')
        fig.suptitle(f'Conv3 initialization: cosine vs displacement — {group.replace("_"," ")}\nOrange = ours; green = baseline; marker / line style = noise eta',fontsize=15,y=.985)
        fig.legend(handles=noise_handles,loc='upper center',bbox_to_anchor=(.5,.925),ncol=5,frameon=False,fontsize=9)
        fig.legend(handles=scheme_handles+draw_handles,loc='lower center',bbox_to_anchor=(.5,.035),ncol=6,frameon=False,fontsize=9)
        fig.text(.5,.015,'Lines: screen (192 examples). Hollow markers: refinement (576 examples), separate estimates. Same eta across all noninput layers.',ha='center',fontsize=9)
        fig.subplots_adjust(top=.80,bottom=.14,left=.07,right=.985,hspace=.29,wspace=.15)
        fig.savefig(out/f'conv3_combined_{group}.jpg',dpi=180,facecolor='white',pil_kwargs={'quality':95});plt.close(fig)
        for param,title in zip(params,titles):
            fig,ax=plt.subplots(figsize=(11,6.7));panel(ax,param,levels)
            ax.set_title(f'Conv3 · {title} · {group.replace("_"," ")}')
            legend=ax.legend(handles=noise_handles,loc='upper left',bbox_to_anchor=(1.01,1),frameon=False,fontsize=9)
            ax.add_artist(legend)
            ax.legend(handles=scheme_handles+draw_handles,loc='lower left',bbox_to_anchor=(1.01,0),frameon=False,fontsize=9)
            fig.subplots_adjust(right=.78,bottom=.13)
            fig.savefig(out/f'conv3_{param}_combined_{group}.jpg',dpi=180,facecolor='white',pil_kwargs={'quality':95});plt.close(fig)
    readme=out/'README.md';text=readme.read_text();heading='## Conv3: overlaid noise levels'
    if heading in text: text=text.split(heading)[0].rstrip()+'\n'
    text+='\n'+heading+'\n\nOrange identifies ours and green baseline; legacy is omitted. Marker shape and line style identify eta. Filled markers show the screen, hollow markers refinement estimates separately. All 19 measured levels are split into groups of 10 and 9; bands are omitted for readability.\n\n'
    for group,_ in groups:
        text+=f'- [{group.replace("_"," ").title()}: all layers](conv3_combined_{group}.jpg)\n'
        for param,title in zip(params,titles):
            text+=f'  - [{title}](conv3_{param}_combined_{group}.jpg)\n'
    readme.write_text(text)
    print(f'Created 2 combined JPGs and 8 individual layer JPGs; {len(etas)} noise levels.')

if __name__=='__main__': main()
