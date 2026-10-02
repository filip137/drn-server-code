"""Review ten-epoch Conv1 clean controls and baseline relative noise."""
import argparse,csv,json,statistics
from pathlib import Path
from experiments.reporting import validate_run

def main():
 p=argparse.ArgumentParser();p.add_argument('--study-root',type=Path,required=True);a=p.parse_args();root=a.study_root.resolve();out=root/'analysis';out.mkdir(exist_ok=True);rows=[];errors=[]
 configs=list((root/'source-training-v1/configs/conv/conv1_clean_baseline_noise_20260927').glob('*.json'));roots=list((root/'collected').glob('*/production/*'))
 for cp in sorted(configs):
  c=json.loads(cp.read_text());arm=cp.stem;fs=[f for r in roots for f in r.glob(arm+'/*/result.json')]
  if len(fs)!=1:errors.append(f'{arm}: {len(fs)} terminal bundles');continue
  run=fs[0].parent;err=validate_run(run)
  if err:errors.extend(f'{arm}: {e}' for e in err);continue
  es=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines() if json.loads(x).get('kind')=='epoch']
  if [e['epoch'] for e in es]!=list(range(1,11)):errors.append(f'{arm}: incomplete epochs');continue
  rows.append(dict(arm=arm,scheme=arm.split('_')[1],seed=c['seed'],eta=c['eqprop']['endpoint_read_noise_eta'],epoch10=100*es[-1]['metrics']['validation_accuracy'],path=str(run)))
 if len(configs)!=14:errors.append(f'{len(configs)}/14 configs')
 with (out/'accuracy.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['arm','scheme','seed','eta','epoch10','path']);w.writeheader();w.writerows(rows)
 stats=[]
 for scheme in ['baseline','ours','legacy']:
  vals=[r['epoch10'] for r in rows if r['eta']==0 and r['scheme']==scheme]
  if len(vals)==3:stats.append(dict(scheme=scheme,mean=statistics.mean(vals),sample_std=statistics.stdev(vals)))
 comparisons=[];old=root.parent/'conv123-relative-noise-training-20260925-v1/production/local'
 for r in rows:
  if not r['eta']:continue
  tag=f"{r['eta']:.0e}".replace('e-0','em').replace('e-','em')
  for scheme in ['ours','legacy']:
   fs=list(old.glob(f'conv1_{scheme}_relative1_eta{tag}_seed0/*/result.json'))
   if len(fs)==1:
    e=[json.loads(x) for x in fs[0].with_name('metrics.jsonl').read_text().splitlines() if json.loads(x).get('kind')=='epoch' and json.loads(x).get('epoch')==10][0];v=100*e['metrics']['validation_accuracy'];comparisons.append(dict(eta=r['eta'],other_scheme=scheme,baseline=r['epoch10'],other=v,gap_pp=r['epoch10']-v))
 summary=dict(complete=len(rows)==14 and not errors,completed=len(rows),expected=14,errors=errors,clean_statistics=stats,noisy_comparisons=comparisons,qualification='MNISTvalidation10epochs; clean3seeds, noise1seed; fixedseed0betaD/F1.')
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');lines=['# Conv1 ten-epoch controls','',summary['qualification'],'',f'Coverage:{len(rows)}/14','','| Scheme | Clean mean (%) | Sample SD (pp) |','|---|---:|---:|']+[f"| {r['scheme']} | {r['mean']:.2f} | {r['sample_std']:.2f} |" for r in stats];(out/'report.md').write_text('\n'.join(lines)+'\n')
 if rows:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,2,figsize=(10,4))
  for scheme,color in [('baseline','tab:green'),('ours','tab:orange'),('legacy','gray')]:
   rr=sorted([r for r in rows if r['eta']==0 and r['scheme']==scheme],key=lambda r:r['seed']);axes[0].plot([r['seed'] for r in rr],[r['epoch10'] for r in rr],'-o',color=color,label=scheme)
   if scheme=='baseline':rr=sorted([r for r in rows if r['eta']>0],key=lambda r:r['eta']);xx=[r['eta'] for r in rr];yy=[r['epoch10'] for r in rr]
   else:rr=sorted([r for r in comparisons if r['other_scheme']==scheme],key=lambda r:r['eta']);xx=[r['eta'] for r in rr];yy=[r['other'] for r in rr]
   if xx:axes[1].plot(xx,yy,'-o',color=color,label=scheme)
  axes[0].set(xlabel='Seed',ylabel='Validation accuracy (%)',title='Clean',xticks=[0,1,2]);axes[1].set(xlabel='Relative noise eta',xscale='log',title='Noisy seed0');axes[0].legend();axes[1].legend();fig.tight_layout();fig.savefig(out/'epoch10_accuracy.jpg',dpi=180);plt.close(fig)
 print(json.dumps(summary));return 0 if summary['complete'] else 1
if __name__=='__main__':raise SystemExit(main())
