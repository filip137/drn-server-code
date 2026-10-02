"""Review collected exp015 weights-only continuations at cumulative epoch30."""
import argparse,csv,json
from pathlib import Path
from experiments.reporting import validate_run

def main():
 p=argparse.ArgumentParser();p.add_argument('--study-root',type=Path,required=True);p.add_argument('--config-subdir',default='relative_noise_epoch30_20260926');p.add_argument('--expected-cases',type=int,default=12);a=p.parse_args();root=a.study_root.resolve();out=root/'analysis';out.mkdir(exist_ok=True)
 configs=list((root/'source-training-v1/configs/conv'/a.config_subdir).glob('*.json'))
 roots=[root/'production/local']+list((root/'collected').glob('*/production/*'))
 rows=[];errors=[]
 for cp in sorted(configs):
  c=json.loads(cp.read_text());arm=cp.stem;matches=[f for r in roots for f in r.glob(arm+'/*/result.json')]
  if len(matches)!=1:errors.append(f'{arm}: {len(matches)} terminal bundles');continue
  run=matches[0].parent;err=validate_run(run)
  if err:errors.extend(f'{arm}: {e}' for e in err);continue
  epochs=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines() if json.loads(x).get('kind')=='epoch']
  end=[e for e in epochs if e.get('epoch')==20]
  if len(end)!=1:errors.append(f'{arm}: no unique additional epoch20');continue
  parent=[json.loads(x) for x in (root/'source_epoch10'/arm/'metrics.jsonl').read_text().splitlines() if json.loads(x).get('kind')=='epoch' and json.loads(x).get('epoch')==10][0]
  rows.append(dict(arm=arm,architecture=arm.split('_')[0],scheme=arm.split('_')[1],eta=c['eqprop']['endpoint_read_noise_eta'],initial_df=c['relative_output_calibration']['target'],epoch10=100*parent['metrics']['validation_accuracy'],epoch30=100*end[0]['metrics']['validation_accuracy'],path=str(run)))
 if len(configs)!=a.expected_cases:errors.append(f'{len(configs)}/{a.expected_cases} prepared configs')
 with (out/'accuracy.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['arm','architecture','scheme','eta','initial_df','epoch10','epoch30','path']);w.writeheader();w.writerows(rows)
 pairs=[]
 for depth in ['conv2','conv3']:
  for eta in sorted({r['eta'] for r in rows if r['architecture']==depth}):
   cell={r['scheme']:r for r in rows if r['architecture']==depth and r['eta']==eta}
   if len(cell)==2:pairs.append(dict(architecture=depth,eta=eta,ours=cell['ours']['epoch30'],legacy=cell['legacy']['epoch30'],gap_pp=cell['ours']['epoch30']-cell['legacy']['epoch30']))
 summary=dict(complete=len(rows)==a.expected_cases and not errors,completed=len(rows),expected=a.expected_cases,errors=errors,pairs=pairs,qualification='Single seed; fresh Adam and reset RNG at epoch10, not uninterrupted training.')
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['# Exp015 cumulative epoch30','',summary['qualification'],'',f'Coverage: {len(rows)}/{a.expected_cases} validated runs.','', '| Model | eta | Ours | Legacy | Gap (pp) |','|---|---:|---:|---:|---:|']
 lines += [f"| {r['architecture']} | {r['eta']:g} | {r['ours']:.2f}% | {r['legacy']:.2f}% | {r['gap_pp']:+.2f} |" for r in pairs]
 lines+=['','Errors: '+('; '.join(errors) or 'none')];(out/'report.md').write_text('\n'.join(lines)+'\n')
 if rows:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,2,figsize=(10,4))
  for ax,depth in zip(axes,['conv2','conv3']):
   for scheme,color in [('ours','tab:orange'),('legacy','gray')]:
    rr=sorted([r for r in rows if r['architecture']==depth and r['scheme']==scheme],key=lambda r:r['eta']);ax.plot([r['eta'] for r in rr],[r['epoch30'] for r in rr],'-o',color=color,label=scheme)
   ax.set(xscale='log',xlabel='Relative noise eta',ylabel='Epoch30 validation accuracy (%)',title=depth);ax.legend()
  fig.tight_layout();fig.savefig(out/'epoch30_accuracy.jpg',dpi=180);plt.close(fig)
 print(json.dumps(summary));return 0 if summary['complete'] else 1
if __name__=='__main__':raise SystemExit(main())
