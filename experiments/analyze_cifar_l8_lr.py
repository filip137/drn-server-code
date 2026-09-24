"""Validate a collected CIFAR L8 LR grid and write per-scheme selections."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from experiments import reporting

SCHEMES=('baseline','ours','legacy')


def validate_cell(path, config):
    errors=reporting.validate_run(path)
    if errors:raise ValueError(f'{path}: {errors}')
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest['resolved_config']!=config:raise ValueError(f'Config mismatch: {path}')
    status=json.loads((path/'status.json').read_text())
    if status['state']=='failed':return {'state':'failed','error':status.get('error')}
    if status['state']!='complete':raise ValueError(f'Active cell cannot be selected: {path}')
    result=json.loads((path/'result.json').read_text())
    metrics=[json.loads(l) for l in (path/'metrics.jsonl').read_text().splitlines()]
    epochs=config['epochs'];assert [m['epoch'] for m in metrics]==list(range(1,epochs+1))
    assert not manifest['dataset']['official_test_read'] and not result['completion']['official_test_read']
    assert all(m['train_examples']==45000 and m['validation_examples']==5000 for m in metrics)
    assert metrics[-1]['steps']==epochs*math.ceil(45000/config['batch_size'])
    assert all(len(m['first_batch_gradient_norms'])==19 and all(math.isfinite(v) and v>0 for v in m['first_batch_gradient_norms'].values()) for m in metrics)
    assets=json.loads((path/'artifacts/assets.json').read_text())
    initial=torch.load(path/'checkpoints/initial_model.pt',map_location='cpu',weights_only=False)['model']
    checkpoint=torch.load(path/'checkpoints/final_model.pt',map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==epochs and checkpoint['config']==config
    final=checkpoint['model'];bn_changes={}
    for section in final.values():
        assert all(torch.isfinite(value).all() for value in section.values())
    for name,value in final['module'].items():
        if name.endswith('num_batches_tracked'):assert value.item()==metrics[-1]['steps']
        if name.startswith('bridges.') and name.endswith(('.weight','.bias')):
            bn_changes[name]=float((value-initial['module'][name]).abs().mean())
            assert bn_changes[name]>0,name
    for name,value in final['conductances'].items():
        assert not torch.equal(value,initial['conductances'][name]),name
        assert value.min()>=config['weight_min'] and value.max()<=config['weight_max'],name
    terminal=result['terminal_metrics'];last=metrics[-1]
    assert terminal['final_validation_accuracy']==last['validation_accuracy']
    assert terminal['final_validation_loss']==last['validation_loss']
    assert all(math.isfinite(m[k]) for m in metrics for k in ('train_loss','train_accuracy','validation_loss','validation_accuracy'))
    return {'state':'complete','eligible':bool(terminal['final_solver_audit_passed']),
            'validation_accuracy':last['validation_accuracy'],'cross_entropy':last['validation_loss'],
            'elapsed_seconds':terminal['elapsed_seconds'],'target':manifest['runtime']['target'],
            'epochs':epochs,'final_solver_audit_passed':terminal['final_solver_audit_passed'],
            'bn_mean_absolute_changes':bn_changes,
            'initial_sha256':assets['initial_sha256'],'split_sha256':assets['split_sha256'],
            'curve':[{'epoch':m['epoch'],'accuracy':m['validation_accuracy'],'ce':m['validation_loss']} for m in metrics]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--configs',type=Path,default=Path('configs/cifar/lr_search_20260921'))
    p.add_argument('--extra-configs',type=Path,action='append',default=[],
                   help='Additional declared configs at the same epoch budget, such as boundary probes')
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--label',default='core')
    a=p.parse_args();root=a.root;rows=[];missing=[]
    receipts={}
    operational_failure_seconds=0.
    for path in (root/'operational_failures').glob('*/launcher.json'):
        failed=json.loads(path.read_text())
        operational_failure_seconds+=sum(r.get('elapsed_seconds',0) for r in failed.get('runs',[]))
    for path in (root/'launcher').glob('*.json'):
        lane=json.loads(path.read_text())
        for run in lane.get('runs',[]):
            if 'exit_code' in run:
                if run['run_id'] in receipts:raise ValueError(f'Duplicate native receipt: {run["run_id"]}')
                receipts[run['run_id']]=run
    config_directories=[a.configs,*a.extra_configs]
    configs=sorted(cfg for directory in config_directories for cfg in directory.glob('*.json'))
    if not configs:raise ValueError('No declared configs')
    if len({cfg.stem for cfg in configs})!=len(configs):raise ValueError('Duplicate declared run ids')
    epoch_budgets={json.loads(cfg.read_text())['epochs'] for cfg in configs}
    if len(epoch_budgets)!=1:raise ValueError('Cannot rank different epoch budgets together')
    for cfg in configs:
        c=json.loads(cfg.read_text());run=root/'cells'/cfg.stem
        if not (run/'status.json').exists():missing.append(cfg.stem);continue
        status=json.loads((run/'status.json').read_text())
        if status['state']=='running':missing.append(cfg.stem);continue
        row={'run_id':cfg.stem,'scheme':c['scheme'],
             'conv_multiplier':c['lr_search']['conv_multiplier'],
             'head_multiplier':c['lr_search']['head_multiplier'],
             'conv_learning_rates':c['optimizer']['conv_learning_rates'],
             'head_learning_rate':c['optimizer']['head_learning_rate'],
             **validate_cell(run,c)}
        receipt=receipts.get(cfg.stem)
        if receipt is None and not a.allow_partial:raise ValueError(f'Missing native exit: {cfg.stem}')
        if receipt is not None:
            if row['state']=='complete':assert receipt['exit_code']==0,cfg.stem
            row['native_exit_code']=receipt['exit_code']
            row['accounted_seconds']=receipt['elapsed_seconds']
        rows.append(row)
    if missing and not a.allow_partial:raise ValueError(f'Incomplete declared coverage: {missing}')
    completed=[r for r in rows if r['state']=='complete']
    for key in ['initial_sha256','split_sha256']:
        if completed:assert len({r[key] for r in completed})==1,key
    top={}
    for scheme in SCHEMES:
        eligible=[r for r in rows if r['scheme']==scheme and r.get('eligible')]
        top[scheme]=sorted(eligible,key=lambda r:(-r['validation_accuracy'],r['cross_entropy'],r['conv_multiplier'],r['head_multiplier']))[:2]
    out=root/'analysis'/a.label;out.mkdir(parents=True,exist_ok=True)
    summary={'declared':len(configs),'terminal':len(rows),'missing':missing,
             'configured_epochs':next(iter(epoch_budgets)),
             'config_directories':[str(p) for p in config_directories],
             'ready_for_selection':not missing,'rows':rows,'top_two':{s:[r['run_id'] for r in t] for s,t in top.items()},
             'successful_run_gpu_hours':sum(r.get('elapsed_seconds',0) for r in rows)/3600,
             'accounted_candidate_gpu_hours':sum(r.get('accounted_seconds',0) for r in rows)/3600,
             'operational_failure_gpu_hours':operational_failure_seconds/3600,
             'accounted_candidate_and_recovery_gpu_hours':(sum(r.get('accounted_seconds',0) for r in rows)+operational_failure_seconds)/3600,
             'official_test_read':False,'evidence_class':'exploratory_single_seed_validation'}
    reporting.atomic_write_json(out/'summary.json',summary)
    fields=['run_id','scheme','state','eligible','conv_multiplier','head_multiplier','head_learning_rate',
            'target','epochs','validation_accuracy','cross_entropy','elapsed_seconds','final_solver_audit_passed']
    with (out/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    if completed:
        fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for scheme,ax in zip(SCHEMES,axes):
            subset=[r for r in completed if r['scheme']==scheme]
            xs=sorted({r['head_multiplier'] for r in subset});ys=sorted({r['conv_multiplier'] for r in subset})
            if not xs or not ys:ax.set_visible(False);continue
            grid=np.full((len(ys),len(xs)),np.nan)
            for r in subset:grid[ys.index(r['conv_multiplier']),xs.index(r['head_multiplier'])]=100*r['validation_accuracy']
            ax.imshow(grid,origin='lower',aspect='auto',vmin=0,vmax=100,cmap='viridis')
            for r in subset:
                ax.text(xs.index(r['head_multiplier']),ys.index(r['conv_multiplier']),f"{100*r['validation_accuracy']:.2f}"+('' if r['eligible'] else '*'),ha='center',va='center',color='white',fontsize=9)
            ax.set_xticks(range(len(xs)),[f'{x:g}' for x in xs]);ax.set_yticks(range(len(ys)),[f'{y:g}' for y in ys])
            ax.set_title(scheme);ax.set_xlabel('Dense LR multiplier');ax.set_ylabel('Conv LR multiplier')
        fig.suptitle('CIFAR L8 validation accuracy (%) — * final solver audit failed')
        fig.savefig(out/'lr_grid.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
