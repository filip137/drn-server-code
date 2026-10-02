"""Collect exp014 epoch-10 comparisons; report missing/failed arms without selection."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from experiments.plot_initial_relative_output_training import (
    read, discover, epoch_history, scientific_config,
)


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root',type=Path,required=True)
    p.add_argument('--config-root',type=Path,default=Path('configs/conv/relative_noise_training_20260925'))
    p.add_argument('--production-root',type=Path,action='append',required=True)
    p.add_argument('--exclude-run',type=Path,action='append',default=[])
    args=p.parse_args()
    configs={c['arm_id']:c for c in (read(f) for f in args.config_root.glob('*.json'))}
    expected={(f'conv{d}',s,float(r),float(e)) for d,targets in [(1,[1]),(2,[1,2,4]),(3,[4,6])]
              for s in ['ours','legacy'] for r in targets for e in [1e-8,1e-7,1e-6,1e-5,1e-4,3e-4,1e-3]}
    keys={(c['arm_id'].split('_')[0],c['arm_id'].split('_')[1],float(c['relative_output_calibration']['target']),float(c['eqprop']['endpoint_read_noise_eta'])) for c in configs.values()}
    if keys!=expected or len(configs)!=84:raise ValueError('Config grid differs from assigned 84 cases')
    grouped,attempts,ignored=discover(args.production_root,set(configs),args.exclude_run)
    rows=[];histories={}
    for arm,c in sorted(configs.items()):
        row=dict(arm_id=arm,architecture=arm.split('_')[0],scheme=arm.split('_')[1],target=float(c['relative_output_calibration']['target']),eta=float(c['eqprop']['endpoint_read_noise_eta']),state='not_collected',epoch10_accuracy_percent=None,epoch10_loss=None,host='',source='',run='',issue='')
        candidates=[a for a in grouped.get(arm,[]) if not a['excluded']]
        unique={a['identity']:a for a in candidates}
        if len(unique)>1:
            row.update(state='ambiguous',issue='Multiple nonidentical attempts; name operational exclusions explicitly')
        elif unique:
            a=next(iter(unique.values()));row.update(state=a['state'],run=a['run'])
            if a['state']=='complete':
                try:
                    m=a['manifest'];actual=m['configuration']['resolved']
                    if scientific_config(actual)!=scientific_config(c):raise ValueError('Resolved scientific config differs')
                    if m['runtime']['device']!='cuda':raise ValueError('Non-CUDA execution')
                    if actual['evaluation']['official_test']['policy']!='disabled':raise ValueError('Official test enabled')
                    history=epoch_history(Path(a['run']));final=history[-1]['metrics']
                    row.update(epoch10_accuracy_percent=100*final['validation_accuracy'],epoch10_loss=final['validation_loss'],host=m['runtime']['host'],source=m['git'].get('source_archive_sha256',''))
                    histories[arm]=history
                except (ValueError,KeyError,OSError) as error:
                    row.update(state='invalid',issue=str(error))
        rows.append(row)
    lookup={(r['architecture'],r['scheme'],r['target'],r['eta']):r for r in rows}
    pairs=[]
    for arch,target,eta in sorted({(r['architecture'],r['target'],r['eta']) for r in rows}):
        ours,legacy=[lookup[(arch,s,target,eta)] for s in ['ours','legacy']]
        pair=dict(architecture=arch,target=target,eta=eta,state='incomplete',ours_accuracy_percent=ours['epoch10_accuracy_percent'],legacy_accuracy_percent=legacy['epoch10_accuracy_percent'],ours_minus_legacy_pp=None)
        if ours['state']==legacy['state']=='complete':
            if not ours['source'] or (ours['host'],ours['source'])!=(legacy['host'],legacy['source']):pair['state']='placement_or_source_mismatch'
            else:pair.update(state='complete',ours_minus_legacy_pp=ours['epoch10_accuracy_percent']-legacy['epoch10_accuracy_percent'])
        pairs.append(pair)
    out=args.study_root/'analysis';out.mkdir(exist_ok=True)
    write_csv(out/'coverage.csv',rows);write_csv(out/'epoch10_differences.csv',pairs)
    summary=dict(expected_runs=84,complete_runs=sum(r['state']=='complete' for r in rows),expected_pairs=42,complete_pairs=sum(p['state']=='complete' for p in pairs),states={state:sum(r['state']==state for r in rows) for state in sorted({r['state'] for r in rows})},excluded_runs=[str(p) for p in args.exclude_run],ignored_smokes=ignored,architecture={})
    lines=['# Relative-noise training: epoch-10 validation','',f"Complete locally validated runs: {summary['complete_runs']}/84; paired comparisons: {summary['complete_pairs']}/42.",'','One initializer, uniform relative eta across noninput layers. Initial D/F calibration; beta fixed during training. No official-test accuracy. Missing/failed outcomes stay in coverage.','']
    for arch in ['conv1','conv2','conv3']:
        selected=[p for p in pairs if p['architecture']==arch];complete=[p for p in selected if p['state']=='complete'];positive=sum(p['ours_minus_legacy_pp']>0 for p in complete)
        avg=sum(p['ours_minus_legacy_pp'] for p in complete)/len(selected) if len(complete)==len(selected) else None
        verdict='supports' if positive else 'contradicts' if len(complete)==len(selected) else 'inconclusive'
        summary['architecture'][arch]=dict(expected_pairs=len(selected),complete_pairs=len(complete),positive_pairs=positive,full_grid_mean_pp=avg,sampled_existence_verdict=verdict)
        lines.append(f'- {arch}: {len(complete)}/{len(selected)} paired cells, {positive} positive; full-grid mean {avg if avg is not None else "pending"} percentage points; sampled-existence criterion {verdict}.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if histories:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        fig,axes=plt.subplots(1,3,figsize=(15,4.8),sharey=True)
        for ax,arch in zip(axes,['conv1','conv2','conv3']):
            for target in sorted({r['target'] for r in rows if r['architecture']==arch}):
                for scheme,color in [('ours','#F28E2B'),('legacy','#6E6E6E')]:
                    rr=sorted((r for r in rows if r['architecture']==arch and r['target']==target and r['scheme']==scheme),key=lambda r:r['eta'])
                    ax.plot([r['eta'] for r in rr],[r['epoch10_accuracy_percent'] if r['state']=='complete' else np.nan for r in rr],marker={1:'o',2:'s',4:'^',6:'D'}[int(target)],color=color,ls='-' if scheme=='ours' else '--',label=f'{scheme}, D/F={target:g}')
            ax.set_xscale('log');ax.set_title(arch.upper());ax.set_xlabel('Relative noise eta');ax.grid(alpha=.2);ax.legend(fontsize=7)
        axes[0].set_ylabel('Epoch-10 validation accuracy (%)');fig.tight_layout();fig.savefig(out/'epoch10_accuracy.jpg',dpi=170,pil_kwargs={'quality':95});plt.close(fig)
        lines += ['', '[Epoch-10 accuracy JPG](epoch10_accuracy.jpg)']
    lines += ['', '[All run coverage](coverage.csv) · [All paired differences](epoch10_differences.csv)']
    (out/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
