"""Execute a plain list of CIFAR configs with bounded subprocesses and receipts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--list',type=Path,required=True)
    p.add_argument('--target',required=True)
    p.add_argument('--receipt-label',help='Distinct receipt name for a later stage on this target')
    p.add_argument('--budget-hours',type=float,default=9)
    a=p.parse_args();root=a.root.resolve();source=root/'source-v2'
    if a.budget_hours<=0:raise ValueError('Budget must be positive')
    label=a.receipt_label or a.target
    if Path(label).name!=label or label in ('.','..'):raise ValueError('Invalid receipt label')
    configs=[line.strip() for line in a.list.read_text().splitlines() if line.strip() and not line.startswith('#')]
    smoke=json.loads((root/f'smoke-{a.target}/status.json').read_text())
    assert smoke['state']=='complete','Destination smoke must complete before training'
    out=root/'launcher';out.mkdir(exist_ok=True)
    receipt=out/f'{label}.json'
    if receipt.exists():raise FileExistsError(receipt)
    (root/'cells').mkdir(exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    child_env=os.environ.copy()
    child_env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    state={'target':a.target,'receipt_label':label,'pid':os.getpid(),'started_at':now(),'state':'running',
           'environment':{key:child_env.get(key) for key in ('CUBLAS_WORKSPACE_CONFIG','LD_PRELOAD','LD_LIBRARY_PATH')},
           'budget_seconds':a.budget_hours*3600,'expected_configs':configs,'runs':[]}
    write(receipt,state);start=time.monotonic();bad=False
    for name in configs:
        remaining=a.budget_hours*3600-(time.monotonic()-start)
        if remaining<=0:bad=True;state['stop_reason']='lane_budget_exhausted';break
        cfg=source/name;c=json.loads(cfg.read_text());run=root/'cells'/cfg.stem
        if run.exists():raise FileExistsError(run)
        cmd=[sys.executable,'-m','experiments.train_cifar_l8_analog',str(cfg),
             '--mode','train','--assets',str(root/f"prepare-{c['scheme']}"),
             '--output-dir',str(run),'--dataset-root',str(root/'data/cifar-10-batches-py'),
             '--source-identity',str(source/'source_identity.json'),'--target',a.target]
        item={'config':name,'run_id':run.name,'command':cmd,'started_at':now()}
        state['runs'].append(item);tick=time.monotonic()
        with (root/'logs'/f'{run.name}.log').open('x') as log:
            child=subprocess.Popen(cmd,cwd=source,env=child_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            item['pid']=child.pid;state['current_run']=run.name;write(receipt,state)
            try:code=child.wait(timeout=min(remaining,c['maximum_runtime_hours']*3600))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=124
        item.update(exit_code=code,finished_at=now(),elapsed_seconds=time.monotonic()-tick)
        if code==0:
            if not (run/'result.json').is_file():
                item['error']='exit0_without_result';bad=True
        else:bad=True
        state['elapsed_seconds']=time.monotonic()-start;write(receipt,state)
    state.update(state='failed' if bad else 'complete',finished_at=now(),elapsed_seconds=time.monotonic()-start)
    state.pop('current_run',None);write(receipt,state)
    return int(bad)

if __name__=='__main__':raise SystemExit(main())
