"""Summarize fixed-epoch clean controls without best-epoch selection."""
import argparse,csv,json,statistics
from pathlib import Path
from experiments.reporting import validate_run

def main():
 p=argparse.ArgumentParser();p.add_argument('--study-root',type=Path,required=True);p.add_argument('--config-subdir',default='noiseless_three_seed_20260927');p.add_argument('--expected-cases',type=int,default=12);p.add_argument('--schemes',nargs='+',default=['ours','legacy']);a=p.parse_args();root=a.study_root.resolve();out=root/'analysis';out.mkdir(exist_ok=True)
 configs=list((root/'source-training-v1/configs/conv'/a.config_subdir).glob('*.json'))
 roots=[root/'production/local']+list((root/'collected').glob('*/production/*'));rows=[];errors=[]
 for cp in sorted(configs):
  c=json.loads(cp.read_text());arm=cp.stem;matches=[f for r in roots for f in r.glob(arm+'/*/result.json')]
  if len(matches)!=1:errors.append(f'{arm}: {len(matches)} completed bundles');continue
  run=matches[0].parent;err=validate_run(run)
  if err:errors.extend(f'{arm}: {e}' for e in err);continue
  epochs=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines() if json.loads(x).get('kind')=='epoch']
  if [e['epoch'] for e in epochs]!=list(range(1,31)):errors.append(f'{arm}: incomplete epoch sequence');continue
  manifest=json.loads((run/'manifest.json').read_text());resolved=manifest['configuration']['resolved']
  assert resolved['eqprop']['endpoint_read_noise_eta']==0 and resolved['eqprop']['endpoint_read_noise_std']==0
  assert resolved['evaluation']['official_test']['policy']=='disabled'
  rows.append(dict(arm=arm,architecture=arm.split('_')[0],scheme=arm.split('_')[1],seed=c['seed'],epoch10=100*epochs[9]['metrics']['validation_accuracy'],epoch30=100*epochs[-1]['metrics']['validation_accuracy'],path=str(run)))
 if len(configs)!=a.expected_cases:errors.append(f'{len(configs)}/{a.expected_cases} configs')
 with (out/'accuracy.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['arm','architecture','scheme','seed','epoch10','epoch30','path']);w.writeheader();w.writerows(rows)
 stats=[];pairs=[]
 for depth in ['conv2','conv3']:
  for scheme in a.schemes:
   rr=[r for r in rows if r['architecture']==depth and r['scheme']==scheme]
   if len(rr)==3:stats.append(dict(architecture=depth,scheme=scheme,mean=statistics.mean(r['epoch30'] for r in rr),sample_std=statistics.stdev(r['epoch30'] for r in rr)))
  for seed in [0,1,2]:
   cell={r['scheme']:r for r in rows if r['architecture']==depth and r['seed']==seed}
   if 'ours' in cell and 'legacy' in cell:pairs.append(dict(architecture=depth,seed=seed,ours=cell['ours']['epoch30'],legacy=cell['legacy']['epoch30'],gap_pp=cell['ours']['epoch30']-cell['legacy']['epoch30']))
 summary=dict(complete=len(rows)==a.expected_cases and not errors,completed=len(rows),expected=a.expected_cases,errors=errors,statistics=stats,pairs=pairs,qualification='Clean30epochs frominitialization with uninterrupted Adam; noisyexp015 resetAdam/RNG atepoch10. Fixed beta selectedseed0, not rematched forseeds1/2. Validation only,3seeds.')
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['# Exp016 noiseless three-seed controls','',summary['qualification'],'',f'Validated coverage: {len(rows)}/{a.expected_cases}.','','| Model | Scheme | Mean accuracy | Sample SD (pp) |','|---|---|---:|---:|']
 lines += [f"| {r['architecture']} | {r['scheme']} | {r['mean']:.2f}% | {r['sample_std']:.2f} |" for r in stats]
 lines+=['','Errors: '+('; '.join(errors) or 'none')];(out/'report.md').write_text('\n'.join(lines)+'\n')
 if rows:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,2,figsize=(9,4))
  for ax,depth in zip(axes,['conv2','conv3']):
   for scheme in a.schemes:
    color={'ours':'tab:orange','legacy':'gray','baseline':'tab:green'}[scheme]
    rr=sorted([r for r in rows if r['architecture']==depth and r['scheme']==scheme],key=lambda r:r['seed']);ax.plot([r['seed'] for r in rr],[r['epoch30'] for r in rr],'-o',color=color,label=scheme)
   ax.set(xlabel='Seed',ylabel='Epoch30 validation accuracy (%)',title=depth,xticks=[0,1,2]);ax.legend()
  fig.tight_layout();fig.savefig(out/'epoch30_accuracy.jpg',dpi=180);plt.close(fig)
 print(json.dumps(summary));return 0 if summary['complete'] else 1
if __name__=='__main__':raise SystemExit(main())
