"""Execute predeclared boundary extensions and independent-array schedule checks."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from analyze import screen_results, choose

HERE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)

def main(args):
    arch=args.architecture
    args.plan=args.plan.resolve()
    if args.additional_screen_status:
        args.additional_screen_status=args.additional_screen_status.resolve()
    plan=read(args.plan)
    out=HERE/(arch+'_confirmation')
    out.mkdir(exist_ok=False)
    (out/'logs').mkdir()
    status=dict(status='waiting_for_screen',phase='confirmation',architecture=arch,pid=os.getpid(),completed=[],heartbeat=time.time())
    save(out/'status.json',status)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',EBL_DEFER_CURRENT_SIMULATIONS='1',EBL_MNIST_ROOT=args.mnist_root)
    code=HERE/'crossbar_code' if arch=='crossbar' else HERE.parent/'ibm-om-cell-aware-quantized'
    def execute(label,cmd):
        status.update(status='running',current_case=label,command=cmd,heartbeat=time.time())
        save(out/'status.json',status)
        print('START',label,flush=True)
        with (out/'logs'/(label+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=child.pid
            while child.poll() is None:
                status['heartbeat']=time.time();save(out/'status.json',status);time.sleep(15)
        if child.returncode:
            status.update(status='failed',exit_code=child.returncode,heartbeat=time.time());save(out/'status.json',status)
            raise RuntimeError(f'{label} failed with {child.returncode}; inspect its retained log.')
        status['completed'].append(label);status.update(child_pid=None,heartbeat=time.time());save(out/'status.json',status)
        print('COMPLETE',label,flush=True)
    screen=HERE/(arch+'_screen')
    while True:
        if (screen/'status.json').exists():
            s=read(screen/'status.json')
            if s['status']=='complete':break
            if s['status']=='failed':
                status.update(status='blocked_by_failed_screen',heartbeat=time.time());save(out/'status.json',status)
                raise RuntimeError('Screen failed; preserve state and request operational diagnosis.')
        status['heartbeat']=time.time();save(out/'status.json',status);time.sleep(30)
    if args.additional_screen_status:
        while True:
            if args.additional_screen_status.exists():
                extra=read(args.additional_screen_status)
                if extra['status']=='complete':break
                if extra['status']=='failed':raise RuntimeError('Additional screen failed; inspect its retained log.')
            status.update(status='waiting_for_additional_screen',heartbeat=time.time());save(out/'status.json',status);time.sleep(30)
    rows=screen_results(screen,arch)
    if len(rows)!=len(plan[arch]['arms']):raise RuntimeError('Screen terminal coverage is incomplete.')
    winners=choose(rows)
    save(out/'screen_selection.json',dict(rule='Minimum selected validation KL, then accuracy, pulses, lower rate; independently by condition and schedule type',winners=winners))
    # A boundary extension repeats the exact original P0/RNG under a separately
    # frozen 60-epoch contract; the first 30 epochs must remain reproducible.
    for i,winner in enumerate(winners):
        if not winner['extension_eligible']:continue
        arm=copy.deepcopy(next(a for a in plan[arch]['arms'] if a['arm_id']==winner['arm_id']))
        arm['epochs']=60
        extension_root=HERE/(arch+'_extensions')
        if arch=='drn':
            extension_plan=copy.deepcopy(plan)
            extension_plan['drn']['arms']=[arm]
            extension_plan_path=out/(arm['arm_id']+'-extension-plan.json');save(extension_plan_path,extension_plan)
            target=extension_root/'runs'/arm['arm_id']
            cmd=[sys.executable,str(HERE/'run_drn.py'),'--plan',str(extension_plan_path),'--arm',arm['arm_id'],'--replica','array-1-write-1','--output',str(target)]
        else:
            cfg=read(code/'examples/mnist_analog_relu/long_kl_schedules_20260906'/(arm['arm_id']+'.json'))
            cfg['stage']['epochs']=60
            config_path=code/'examples/mnist_analog_relu/long_kl_extensions_20260906'/(arm['arm_id']+'.json');save(config_path,cfg)
            target=extension_root/'results/extensions/runs'/arm['arm_id']
            # Ordinary standalone native ebl output: extension configs are
            # recorded here rather than mutating the prepared screen study.
            target=extension_root/'native'/arm['arm_id']
            inputs=Path(args.crossbar_inputs)
            cmd=[sys.executable,'-m','ebl','train','--config',str(config_path),'--output-dir',str(target),'--teacher-weights',str(inputs/'teacher_weights.pt'),'--device-state',str(inputs/('hwa_healthy_p0.pt' if arm['condition']=='healthy' else 'hwa_faulted_p0.pt'))]
        execute('extension-'+arm['arm_id'],cmd)
        # Preserve the selected schedule and its new epoch budget for array checks.
        winner['confirmation_epochs']=60
        winner['extension_result_root']=str(target)
    for w in winners:w.setdefault('confirmation_epochs',30)
    save(out/'frozen_selection.json',dict(winners=winners,selection_frozen_at=time.time(),scope='Same source teachers and HWA masters, two additional saved arrays; no test-based selection.'))
    if arch=='drn':
        jobs=[]
        for w in winners:
            arm=copy.deepcopy(next(a for a in plan['drn']['arms'] if a['arm_id']==w['arm_id']))
            arm['epochs']=w['confirmation_epochs']
            for replica in plan['drn']['replicas'][1:]:jobs.append(dict(arm=arm,replica=replica['id']))
        cplan=copy.deepcopy(plan)
        cplan['drn']['arms']=[dict(next(a for a in plan['drn']['arms'] if a['arm_id']==w['arm_id']),epochs=w['confirmation_epochs']) for w in winners]
        cplan_path=out/'frozen_plan.json';save(cplan_path,cplan)
        save(out/'cases.json',jobs)
        parallel_jobs=[]
        for job in jobs:
            case=job['arm']['arm_id']+'-'+job['replica']
            target=out/'runs'/case
            cmd=[sys.executable,str(HERE/'run_drn.py'),'--plan',str(cplan_path),'--arm',job['arm']['arm_id'],'--replica',job['replica'],'--output',str(target)]
            if args.workers>1:
                parallel_jobs.append(dict(label=case,command=cmd,result=target/'result.json',epochs=job['arm']['epochs']))
            else:
                execute(case,cmd)
                if read(target/'result.json')['status']!='complete':raise RuntimeError(case)
        if parallel_jobs:
            from parallel_runs import run_jobs
            run_jobs(parallel_jobs,out=out,status=status,cwd=code,env=env,workers=args.workers)
    else:
        inputs=read(HERE/'confirmation_inputs/receipt.json')
        study_id='mnist-ibm-om-crossbar-long-kl-confirmation-20260906-v1'
        study=copy.deepcopy(read(code/'studies'/(plan['crossbar']['study_id']+'.json')))
        study.update(study_id=study_id,title='Frozen low-rate and decaying schedules on two additional crossbar arrays',hypothesis='The selected longer schedules reduce held-apparent teacher KL on additional saved arrays.',arms=[],completion_criteria=['Eight frozen-schedule runs and four one-epoch reference runs complete on two additional saved arrays.','Teacher and HWA master hashes match the screening inputs.','All selection uses validation KL with P0 eligible; the official test set remains closed.'])
        jobs=[]
        for condition in ['healthy','faulted']:
            selected=[w for w in winners if w['condition']==condition]
            for inp in [v for v in inputs if v['condition']==condition]:
                for w in selected+[None]:
                    reference=w is None
                    base_id=condition+('-constant-1e-5' if condition=='healthy' else '-constant-3e-5') if reference else w['arm_id']
                    cfg=read(code/'examples/mnist_analog_relu/long_kl_schedules_20260906'/(base_id+'.json'))
                    cfg['device']['assignment_seed']=inp['assignment_seed'];cfg['device']['endpoint_seed']=inp['endpoint_seed']
                    cfg['stage']['epochs']=1 if reference else w['confirmation_epochs']
                    case=(condition+'-previous-one-epoch' if reference else base_id)+'-array-'+str(inp['assignment_seed'])
                    path=code/'examples/mnist_analog_relu/long_kl_confirmation_20260906'/(case+'.json');save(path,cfg)
                    study['arms'].append(dict(arm_id=case,configs=['../'+str(path.relative_to(code))],description='Frozen '+('previous one-epoch reference' if reference else w['schedule_kind'])+' recovery on a held-out saved array',experiment_id='mnist_ibm_om_crossbar_relu.v2',mode='train'))
                    jobs.append(dict(case=case,config=str(path),input=inp,reference=reference,schedule_source=base_id))
        study_path=code/'studies'/(study_id+'.json');save(study_path,study);save(out/'cases.json',jobs)
        execute('prepare-confirmation',[sys.executable,'-m','ebl','study','prepare','--plan',str(study_path),'--results-root',str(out/'results')])
        for job in jobs:
            target=out/'results'/study_id/'runs'/job['case']
            cmd=[sys.executable,'-m','ebl','train','--config',job['config'],'--output-dir',str(target),'--teacher-weights',str(Path(args.crossbar_inputs)/'teacher_weights.pt'),'--device-state',job['input']['local_path']]
            execute(job['case'],cmd)
            results=list(target.glob('*/result.json'))
            if len(results)!=1 or read(results[0])['status']!='complete':raise RuntimeError(job['case'])
        execute('verify-confirmation',[sys.executable,'-m','ebl','study','summarize','--study-dir',str(out/'results'/study_id),'--verify-artifacts'])
        execute('verify-screen',[sys.executable,'-m','ebl','study','summarize','--study-dir',str(screen/'results'/plan['crossbar']['study_id']),'--verify-artifacts'])
    status.update(status='complete',heartbeat=time.time(),completed_at=time.time());save(out/'status.json',status)
    print('CONFIRMATION_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('architecture',choices=['drn','crossbar']);p.add_argument('--mnist-root',required=True);p.add_argument('--crossbar-inputs',default='');p.add_argument('--workers',type=int,default=1);p.add_argument('--plan',type=Path,default=HERE/'plan.json');p.add_argument('--additional-screen-status',type=Path);main(p.parse_args())
