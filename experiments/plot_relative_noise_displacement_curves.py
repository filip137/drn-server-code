"""CPU-only curves from exp013's collected screen and refinement summaries."""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from experiments.plot_layer_cosine_displacement_curves import COLORS, SCHEMES, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    args = parser.parse_args()
    analysis = args.study_root / 'analysis'
    out = analysis / 'displacement_curves'
    out.mkdir(exist_ok=True)
    values = []
    for stage in ('stage_a', 'stage_b'):
        physical = {(r['architecture'], r['scheme'], float(r['relative_target']),
                     int(r['state_layer_index'])): r
                    for r in read(analysis / stage / 'physical_state_controls.csv')}
        for row in read(analysis / stage / 'layer_summary.csv'):
            arch, scheme = row['architecture'], row['scheme']
            depth = int(arch[-1])
            layer = depth if row['parameter_name'].startswith('Dense') else int(row['parameter_name'].split('_')[-1])
            key = (arch, scheme, float(row['relative_target']))
            output, state = physical[(*key, depth)], physical[(*key, layer)]
            record = dict(stage=stage, architecture=arch, scheme=scheme,
                          parameter=row['parameter_name'], eta=float(row['noise_level']),
                          target=float(row['relative_target']),
                          relative_output=float(output['D_over_F']),
                          absolute_output=float(output['centered_displacement_rms']),
                          absolute_layer=float(state['centered_displacement_rms']),
                          cosine=float(row['cosine']), low=float(row['cosine_min']),
                          high=float(row['cosine_max']))
            assert all(math.isfinite(v) for v in record.values() if isinstance(v, float))
            assert record['relative_output'] > 0
            values.append(record)
    etas = sorted({r['eta'] for r in values})
    labels = {'relative_output': 'Measured output normalized RMS displacement D/F',
              'absolute_output': 'Measured output RMS displacement D (voltage units)',
              'absolute_layer': 'Measured layer RMS displacement D (voltage units)'}
    links = []
    for variant, xlabel in labels.items():
        with PdfPages(out / f'all_layers_{variant}.pdf') as book:
            for depth in (1, 2, 3):
                arch = f'conv{depth}'
                params = [f'ConvWeight_{i}' for i in range(depth)] + ['DenseWeight_0']
                for start in range(0, len(etas), 5):
                    chunk = etas[start:start + 5]
                    fig, axes = plt.subplots(len(chunk), depth + 1, figsize=(4.1*(depth+1), 2.35*len(chunk)+1.7), squeeze=False)
                    for i, eta in enumerate(chunk):
                        for j, param in enumerate(params):
                            ax = axes[i, j]
                            for scheme in SCHEMES:
                                for stage in ('stage_a', 'stage_b'):
                                    points = sorted((r for r in values if r['architecture']==arch and r['parameter']==param and r['scheme']==scheme and r['eta']==eta and r['stage']==stage), key=lambda r:r[variant])
                                    if not points:
                                        continue
                                    x, y, lo, hi = ([r[k] for r in points] for k in (variant, 'cosine', 'low', 'high'))
                                    if stage == 'stage_a':
                                        ax.plot(x, y, 'o-', color=COLORS[scheme], ms=3.5, lw=1.3)
                                        ax.fill_between(x, lo, hi, color=COLORS[scheme], alpha=.10, linewidth=0)
                                    else:
                                        ax.errorbar(x, y, yerr=[[max(0,b-a) for a,b in zip(lo,y)], [max(0,b-a) for a,b in zip(y,hi)]], fmt='D--', mfc='white', color=COLORS[scheme], ms=4.5, lw=1, capsize=2)
                            ax.set_xscale('log')
                            ax.set_ylim(-.12, 1.03)
                            ax.axhline(0, color='gray', lw=.5)
                            ax.grid(alpha=.2)
                            if variant == 'relative_output':
                                ax.set_xlim(.45, 9)
                                ax.set_xticks([.5,1,2,4,8], ['0.5','1','2','4','8'])
                                ax.minorticks_off()
                            if i==0:
                                ax.set_title(f'Convolution {j+1}' if j<depth else 'Readout')
                            if j==0:
                                ax.set_ylabel(f'eta = {eta:.3g}\nCosine vs BPTT')
                            if i==len(chunk)-1:
                                ax.set_xlabel('Output D/F' if variant=='relative_output' else 'Output RMS D' if variant=='absolute_output' else 'Layer RMS D')
                    handles = [Line2D([],[],color=COLORS[s],label=s,lw=2) for s in SCHEMES]
                    handles += [Line2D([],[],color='gray',marker='o',label='Screen (192 examples)'), Line2D([],[],color='gray',marker='D',mfc='white',ls='--',label='Refinement (576 examples)')]
                    fig.suptitle(f'Conv{depth}: layerwise cosine vs displacement at initialization\n{xlabel}', y=.995, fontsize=13)
                    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.948),ncol=3,fontsize=8,frameon=False)
                    fig.text(.5,.015, 'One global relative eta for all noninput layers. D = RMS((v+ - v-)/2); F = free-output RMS.\nBands/whiskers: range of three noise-draw means, not confidence intervals. Stages are not pooled.\nLines connect measured points only. Clean references use 576 examples in both stages. One model seed; no training.', ha='center',fontsize=8)
                    fig.subplots_adjust(top=.87,bottom=.115,left=.10,right=.985,hspace=.35,wspace=.24)
                    name = f'{arch}_{variant}_{start//5+1}'
                    fig.savefig(out/f'{name}.jpg',dpi=140,facecolor='white',pil_kwargs={'quality':95})
                    book.savefig(fig)
                    plt.close(fig)
                    if variant=='relative_output':
                        links.append(f'- [Conv{depth}, noise rows {start+1}–{start+len(chunk)}]({name}.jpg)')
    with (out/'plotted_values.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    (out/'README.md').write_text('# Cosine versus RMS displacement\n\nAll layers, schemes and measured noise levels from both stages; no additional simulations.\n\n[Normalized output D/F: all plots](all_layers_relative_output.pdf) · [Absolute output RMS](all_layers_absolute_output.pdf) · [Layer RMS](all_layers_absolute_layer.pdf)\n\nRows are noise levels; columns are layers. Screen curves and refinement curves are separate because cohort sizes and noise draws differ. Refinement-only noise levels have only diamond curves. Three-draw ranges are not confidence intervals. No interpolation, extrapolation or pooling between stages.\n\n'+ '\n'.join(links)+'\n\n[Plotted values](plotted_values.csv)\n')
    print(f'Wrote {len(values)} plotted rows, {len(etas)} noise levels, three PDF collections to {out}')

if __name__ == '__main__':
    main()
