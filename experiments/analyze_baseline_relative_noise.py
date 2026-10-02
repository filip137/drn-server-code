"""Summarize uninterrupted baseline noisy controls and existing reset comparisons."""
import argparse,csv,json
from pathlib import Path
from experiments.reporting import validate_run

def main():
 p=argparse.ArgumentParser();p.add_argument('--study-root',type=Path,required=True);a=p.parse_args();root=a.study_root.resolve();out=root/'analysis';out.mkdir(exist_ok=True)
 configs=list((root/'source-training-v1/configs/conv/baseline_relative_noise_20260927').glob('*.json'));roots=[root/'production/local']+list((root/'collected').glob('*/production/*'));rows=[];errors=[]
 for cp in sorted(configs):
  c=json.loads(cp.read_text());arm=cp.stem;matches=[f for r in roots for f in r.glob(arm+'/*/result.json')]
  if len(matches)!=1:errors.append(f'{arm}: {len(matches)} terminal bundles');continue
  run=matches[0].parent;err=validate_run(run)
  if err:errors.extend(f'{arm}: {e}' for e in err);continue
  es=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines() if json.loads(x).get('kind')=='epoch']
  if [e['epoch'] for e in es]!=list(range(1,31)):errors.append(f'{arm}: incomplete epochs');continue
  rows.append(dict(arm=arm,architecture=arm.split('_')[0],scheme='baseline',eta=c['eqprop']['endpoint_read_noise_eta'],epoch10=100*es[9]['metrics']['validation_accuracy'],epoch30=100*es[-1]['metrics']['validation_accuracy'],path=str(run)))
 if len(configs)!=10:errors.append(f'{len(configs)}/10 configs')
 with (out/'accuracy.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['arm','architecture','scheme','eta','epoch10','epoch30','path']);w.writeheader();w.writerows(rows)
 comparisons=[]
 for sibling in ['conv23-relative-noise-epoch30-20260926-v1','conv23-relative-noise-epoch30-low-noise-20260926-v1']:
  f=root.parent/sibling/'analysis/accuracy.csv'
  if f.exists():
   for r in csv.DictReader(f.open()):
    for b in rows:
     if b['architecture']==r['architecture'] and b['eta']==float(r['eta']):comparisons.append(dict(architecture=b['architecture'],eta=b['eta'],other_scheme=r['scheme'],baseline=b['epoch30'],other=float(r['epoch30']),baseline_minus_other_pp=b['epoch30']-float(r['epoch30'])))
 summary=dict(complete=len(rows)==10 and not errors,completed=len(rows),expected=10,errors=errors,comparisons=comparisons,qualification='Baseline uninterrupted30epochs; ours/legacy resetAdam/RNG atepoch10. Optimizerhistory confounds crossscheme epoch30comparison. Same seed0 validation only.')
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['# Exp017 baseline relative-noise training','',summary['qualification'],'',f'Validated coverage: {len(rows)}/10.','','| Model | eta | Epoch10 | Epoch30 |','|---|---:|---:|---:|']
 lines += [f"| {r['architecture']} | {r['eta']:g} | {r['epoch10']:.2f}% | {r['epoch30']:.2f}% |" for r in rows];lines+=['','Errors: '+('; '.join(errors) or 'none')];(out/'report.md').write_text('\n'.join(lines)+'\n')
 if rows:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,2,figsize=(10,4))
  for ax,depth in zip(axes,['conv2','conv3']):
   rr=sorted([r for r in rows if r['architecture']==depth],key=lambda r:r['eta']);ax.plot([r['eta'] for r in rr],[r['epoch30'] for r in rr],'-o',color='tab:green',label='baseline (uninterrupted)')
   for scheme,color in [('ours','tab:orange'),('legacy','gray')]:
    cc=sorted([r for r in comparisons if r['architecture']==depth and r['other_scheme']==scheme],key=lambda r:r['eta'])
    if cc:ax.plot([r['eta'] for r in cc],[r['other'] for r in cc],'--o',color=color,label=scheme+' (reset at10)')
   ax.set(xscale='log',xlabel='Relative noise eta',ylabel='Epoch30 validation accuracy (%)',title=depth);ax.legend(fontsize=8)
  fig.tight_layout();fig.savefig(out/'epoch30_accuracy.jpg',dpi=180);plt.close(fig)
 print(json.dumps(summary));return 0 if summary['complete'] else 1
if __name__=='__main__':raise SystemExit(main())
